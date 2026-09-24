#!/usr/bin/env python3
"""Export all MiniCPM-V 4.6 text shards and native runtime assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import AutoTokenizer, DynamicCache
from transformers.models.minicpmv4_6.modeling_minicpmv4_6 import (
    MiniCPMV4_6ForConditionalGeneration,
)

from export_decode_probes import (
    FullAttentionDecodeLayer,
    LinearAttentionDecodeLayer,
    export_onnx,
    tensor_error,
)
from export_prefill_probes import FullAttentionPrefillLayer, LinearAttentionPrefillLayer


SEQUENCE = 87
VISUAL_TOKENS = 64
HIDDEN = 1024
CAPACITY = 256
IMAGE_TOKEN_ID = 248056
LOGITS_CHUNK = 61_440


def build_prompt() -> str:
    return (
        "<|im_start|>user\n<image_id>0</image_id><image>"
        + "<|image_pad|>" * VISUAL_TOKENS
        + "</image>\nPlease describe this image.<|im_end|>\n"
        + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def load_visual_reference(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npz":
        with np.load(path) as fixture:
            if "image_embeds" not in fixture:
                raise RuntimeError(f"Missing image_embeds in {path}")
            return np.asarray(fixture["image_embeds"], dtype=np.float32).reshape(-1)
    return np.fromfile(path, dtype=np.float32)


class FinalNormLmHead(nn.Module):
    def __init__(self, norm: nn.Module, lm_head: nn.Module) -> None:
        super().__init__()
        self.norm = norm
        self.lm_head = lm_head

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, ...]:
        hidden_states = self.norm(hidden_states)[:, -1, :]
        outputs = []
        for start in range(0, self.lm_head.weight.shape[0], LOGITS_CHUNK):
            end = min(start + LOGITS_CHUNK, self.lm_head.weight.shape[0])
            outputs.append(F.linear(hidden_states, self.lm_head.weight[start:end]))
        return tuple(outputs)


def errors_ok(actual: tuple[Tensor, ...], expected: tuple[Tensor, ...], tolerance: float) -> bool:
    return max(tensor_error(a, e)["max_abs"] for a, e in zip(actual, expected, strict=True)) <= tolerance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vision-reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=CAPACITY)
    args = parser.parse_args()
    if args.capacity <= SEQUENCE:
        raise ValueError(f"capacity must exceed {SEQUENCE}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "prefill").mkdir(exist_ok=True)
    (args.output_dir / "decode").mkdir(exist_ok=True)
    (args.output_dir / "head").mkdir(exist_ok=True)
    (args.output_dir / "assets").mkdir(exist_ok=True)

    print("[1/7] Loading tokenizer and original FP32 weights", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompt = build_prompt()
    token_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    if len(token_ids) != SEQUENCE or token_ids.count(IMAGE_TOKEN_ID) != VISUAL_TOKENS:
        raise RuntimeError(
            f"Prompt contract mismatch: total={len(token_ids)}, image={token_ids.count(IMAGE_TOKEN_ID)}"
        )
    source = MiniCPMV4_6ForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval()
    text_model = source.model.language_model
    config = source.config.text_config

    print("[2/7] Building fixed prompt and native lookup assets", flush=True)
    ids = torch.tensor([token_ids], dtype=torch.int64)
    with torch.no_grad():
        prompt_embeddings = text_model.embed_tokens(ids).float()
    visual = load_visual_reference(args.vision_reference)
    if visual.size != VISUAL_TOKENS * HIDDEN:
        raise RuntimeError(f"Unexpected vision reference size: {visual.size}")
    visual = torch.from_numpy(visual.reshape(VISUAL_TOKENS, HIDDEN))
    image_indices = [index for index, token in enumerate(token_ids) if token == IMAGE_TOKEN_ID]
    prompt_embeddings[0, image_indices, :] = visual

    assets = args.output_dir / "assets"
    np.asarray(token_ids, dtype=np.int32).tofile(assets / "prompt_token_ids_i32.bin")
    np.asarray(image_indices, dtype=np.int32).tofile(assets / "image_indices_i32.bin")
    prompt_embeddings.numpy().tofile(assets / "prompt_embeddings_fp32.bin")
    text_model.embed_tokens.weight.detach().half().numpy().tofile(assets / "token_embeddings_fp16.bin")
    positions = torch.arange(args.capacity, dtype=torch.int64).unsqueeze(0)
    with torch.inference_mode():
        rope_cos, rope_sin = text_model.rotary_emb(
            torch.zeros(1, args.capacity, HIDDEN), positions
        )
    rope_cos.half().numpy().tofile(assets / "rope_cos_fp16.bin")
    rope_sin.half().numpy().tofile(assets / "rope_sin_fp16.bin")

    prefill_positions = torch.arange(SEQUENCE, dtype=torch.int64).unsqueeze(0)
    with torch.inference_mode():
        prefill_cos, prefill_sin = text_model.rotary_emb(prompt_embeddings, prefill_positions)
    causal_mask = torch.full(
        (1, 1, SEQUENCE, SEQUENCE),
        torch.finfo(torch.float32).min,
        dtype=torch.float32,
    ).triu(diagonal=1)

    print("[3/7] Exporting and validating all 24 prefill layers", flush=True)
    cache = DynamicCache(config=config)
    hidden = prompt_embeddings
    layer_types = []
    prefill_metrics = []
    for index, layer in enumerate(text_model.layers):
        layer_types.append(layer.layer_type)
        layer_dir = args.output_dir / "prefill" / f"layer_{index:02d}"
        if layer.layer_type == "linear_attention":
            wrapper = LinearAttentionPrefillLayer(layer).eval()
            with torch.inference_mode():
                expected_hidden = layer(
                    hidden.clone(),
                    position_embeddings=(prefill_cos, prefill_sin),
                    past_key_values=cache,
                )
                actual = wrapper(hidden.clone())
            expected = (
                expected_hidden,
                cache.layers[index].conv_states,
                cache.layers[index].recurrent_states,
            )
            tolerance = 5e-3
        else:
            wrapper = FullAttentionPrefillLayer(
                layer, prefill_cos, prefill_sin, causal_mask
            ).eval()
            with torch.inference_mode():
                expected_hidden = layer(
                    hidden.clone(),
                    position_embeddings=(prefill_cos, prefill_sin),
                    attention_mask=causal_mask,
                    past_key_values=cache,
                )
                actual = wrapper(hidden.clone())
            expected = (
                expected_hidden,
                cache.layers[index].keys,
                cache.layers[index].values,
            )
            tolerance = 2e-5
        if not errors_ok(actual, expected, tolerance):
            raise RuntimeError(f"Prefill layer {index} wrapper mismatch")
        metric = tensor_error(actual[0], expected_hidden)
        prefill_metrics.append(metric)
        export_onnx(
            wrapper,
            (hidden,),
            layer_dir / "model.onnx",
            ["hidden_states", "hidden_out", "state_0", "state_1"],
        )
        hidden = expected_hidden
        print(
            f"      prefill {index + 1:02d}/24 {layer.layer_type} "
            f"max_abs={metric['max_abs']:.3g}",
            flush=True,
        )

    print("[4/7] Exporting final norm and chunked LM head", flush=True)
    head = FinalNormLmHead(text_model.norm, source.lm_head).eval()
    with torch.inference_mode():
        prefill_logits_parts = head(hidden[:, -1:, :])
        prefill_logits = torch.cat(prefill_logits_parts, dim=-1)
    first_token = int(prefill_logits.argmax(dim=-1).item())
    head_names = [f"logits_{index}" for index in range(len(prefill_logits_parts))]
    head_path = args.output_dir / "head" / "model.onnx"
    head_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            head,
            (hidden[:, -1:, :],),
            str(head_path),
            input_names=["hidden_states"],
            output_names=head_names,
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
            external_data=True,
        )
    import onnx

    onnx.checker.check_model(str(head_path))
    print(
        f"      first token={first_token} {tokenizer.decode([first_token])!r}",
        flush=True,
    )

    print("[5/7] Exporting and validating all 24 fixed-capacity decode layers", flush=True)
    decode_hidden = text_model.embed_tokens(torch.tensor([[first_token]], dtype=torch.int64))
    decode_position = torch.tensor([[SEQUENCE]], dtype=torch.int64)
    with torch.inference_mode():
        decode_cos, decode_sin = text_model.rotary_emb(decode_hidden, decode_position)
    decode_mask = torch.full(
        (1, 1, 1, args.capacity + 1),
        torch.finfo(torch.float32).min,
        dtype=torch.float32,
    )
    decode_mask[..., :SEQUENCE] = 0
    decode_mask[..., -1] = 0
    decode_metrics = []
    for index, layer in enumerate(text_model.layers):
        layer_dir = args.output_dir / "decode" / f"layer_{index:02d}"
        if layer.layer_type == "linear_attention":
            conv_state = cache.layers[index].conv_states.clone()
            recurrent_state = cache.layers[index].recurrent_states.clone()
            wrapper = LinearAttentionDecodeLayer(layer).eval()
            with torch.inference_mode():
                expected_hidden = layer(
                    decode_hidden.clone(),
                    position_embeddings=(decode_cos, decode_sin),
                    past_key_values=cache,
                )
                actual = wrapper(decode_hidden.clone(), conv_state, recurrent_state)
            expected = (
                expected_hidden,
                cache.layers[index].conv_states,
                cache.layers[index].recurrent_states,
            )
            inputs = (decode_hidden, conv_state, recurrent_state)
            names = [
                "hidden_states", "conv_state", "recurrent_state",
                "hidden_out", "new_conv_state", "new_recurrent_state",
            ]
        else:
            valid_key = cache.layers[index].keys.clone()
            valid_value = cache.layers[index].values.clone()
            past_key = F.pad(valid_key, (0, 0, 0, args.capacity - SEQUENCE))
            past_value = F.pad(valid_value, (0, 0, 0, args.capacity - SEQUENCE))
            wrapper = FullAttentionDecodeLayer(layer).eval()
            with torch.inference_mode():
                expected_hidden = layer(
                    decode_hidden.clone(),
                    position_embeddings=(decode_cos, decode_sin),
                    attention_mask=None,
                    past_key_values=cache,
                )
                actual = wrapper(
                    decode_hidden.clone(), decode_cos, decode_sin,
                    past_key, past_value, decode_mask,
                )
            expected = (
                expected_hidden,
                cache.layers[index].keys[..., -1:, :],
                cache.layers[index].values[..., -1:, :],
            )
            inputs = (
                decode_hidden, decode_cos, decode_sin,
                past_key, past_value, decode_mask,
            )
            names = [
                "hidden_states", "position_cos", "position_sin",
                "past_key", "past_value", "attention_mask",
                "hidden_out", "new_key", "new_value",
            ]
        if not errors_ok(actual, expected, 2e-5):
            raise RuntimeError(f"Decode layer {index} wrapper mismatch")
        metric = tensor_error(actual[0], expected_hidden)
        decode_metrics.append(metric)
        export_onnx(wrapper, inputs, layer_dir / "model.onnx", names)
        decode_hidden = expected_hidden
        print(
            f"      decode {index + 1:02d}/24 {layer.layer_type} "
            f"max_abs={metric['max_abs']:.3g}",
            flush=True,
        )

    print("[6/7] Computing second-token reference", flush=True)
    with torch.inference_mode():
        decode_logits = torch.cat(head(decode_hidden), dim=-1)
    second_token = int(decode_logits.argmax(dim=-1).item())
    print(
        f"      second token={second_token} {tokenizer.decode([second_token])!r}",
        flush=True,
    )

    print("[7/7] Writing manifest", flush=True)
    manifest = {
        "model": "MiniCPM-V 4.6",
        "platform": "mt6899",
        "sequence": SEQUENCE,
        "visual_tokens": VISUAL_TOKENS,
        "hidden_size": HIDDEN,
        "capacity": args.capacity,
        "vocab_size": config.vocab_size,
        "eos_token_id": source.config.eos_token_id,
        "image_token_id": IMAGE_TOKEN_ID,
        "layer_types": layer_types,
        "logits_chunks": [int(part.shape[-1]) for part in prefill_logits_parts],
        "first_token_id": first_token,
        "first_token_text": tokenizer.decode([first_token]),
        "second_token_id": second_token,
        "second_token_text": tokenizer.decode([second_token]),
        "prompt": prompt,
        "prompt_token_ids": token_ids,
        "image_indices": image_indices,
        "prefill_hidden_metrics": prefill_metrics,
        "decode_hidden_metrics": decode_metrics,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True) + "\n"
    )
    print(f"All shards exported to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
