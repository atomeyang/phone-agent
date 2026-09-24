#!/usr/bin/env python3
"""Compare MiniCPM-V 4.6 HF and MNN-export Python LLM checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


IMAGE_TOKEN_ID = 248056
HIDDEN_SIZE = 1024


def build_prompt(user_prompt: str, visual_tokens: int) -> str:
    return (
        "<|im_start|>user\n<image_id>0</image_id><image>"
        + "<|image_pad|>" * visual_tokens
        + "</image>\n"
        + user_prompt
        + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def output_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    raise TypeError(f"Hook output has no tensor: {type(output)!r}")


def capture_hook(checkpoints: dict[str, torch.Tensor], name: str):
    def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        checkpoints[name] = output_tensor(output).detach().float().cpu().contiguous()

    return hook


def install_hooks(model: Any, wrapper: bool) -> tuple[dict[str, torch.Tensor], list[Any]]:
    checkpoints: dict[str, torch.Tensor] = {}
    handles = []
    layers = model.blocks if wrapper else model.layers
    for index, layer in enumerate(layers):
        prefix = f"layer_{index:02d}"
        mixer = layer.self_attn if wrapper else (
            layer.linear_attn if layer.layer_type == "linear_attention" else layer.self_attn
        )
        modules = {
            f"{prefix}.input_norm": layer.input_layernorm,
            f"{prefix}.mixer": mixer,
            f"{prefix}.post_attention_norm": layer.post_attention_layernorm,
            f"{prefix}.mlp": layer.mlp,
            f"{prefix}.output": layer,
        }
        for name, module in modules.items():
            handles.append(module.register_forward_hook(capture_hook(checkpoints, name)))
    final_norm = model.final_layernorm if wrapper else model.norm
    handles.append(final_norm.register_forward_hook(capture_hook(checkpoints, "final_norm")))
    return checkpoints, handles


def load_inputs(args: argparse.Namespace, tokenizer: Any, embedding: torch.nn.Module):
    visual = np.fromfile(args.visual_embedding, dtype=np.float32)
    if visual.size % HIDDEN_SIZE != 0:
        raise RuntimeError(f"Invalid visual embedding element count: {visual.size}")
    visual = torch.from_numpy(visual.reshape(-1, HIDDEN_SIZE).copy())
    user_prompt = args.prompt.read_text(encoding="utf-8").rstrip("\r\n")
    prompt = build_prompt(user_prompt, visual.shape[0])
    token_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    image_indices = [i for i, token_id in enumerate(token_ids) if token_id == IMAGE_TOKEN_ID]
    if len(image_indices) != visual.shape[0]:
        raise RuntimeError(
            f"Image token mismatch: tokens={len(image_indices)}, embeddings={visual.shape[0]}"
        )

    ids = torch.tensor([token_ids], dtype=torch.long)
    with torch.inference_mode():
        text_embeddings = embedding(ids).detach().float().cpu()
    if text_embeddings.shape != (1, len(token_ids), HIDDEN_SIZE):
        text_embeddings = text_embeddings.reshape(1, len(token_ids), HIDDEN_SIZE)
    inputs_embeds = text_embeddings.clone()
    inputs_embeds[0, image_indices] = visual
    return ids, text_embeddings, inputs_embeds, image_indices


def run_official(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer
    from transformers.models.minicpmv4_6.modeling_minicpmv4_6 import (
        MiniCPMV4_6ForConditionalGeneration,
    )

    print("[official 1/4] Loading MiniCPM-V 4.6 FP32", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    source = MiniCPMV4_6ForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        local_files_only=True,
    ).eval()
    text_model = source.model.language_model

    print("[official 2/4] Building the fixed multimodal input", flush=True)
    ids, text_embeddings, inputs_embeds, image_indices = load_inputs(
        args, tokenizer, text_model.embed_tokens
    )
    print(
        f"tokens={ids.shape[1]} image_tokens={len(image_indices)} hidden={inputs_embeds.shape[-1]}",
        flush=True,
    )
    checkpoints, handles = install_hooks(text_model, wrapper=False)
    attention_mask = torch.ones_like(ids)
    position_ids = torch.arange(ids.shape[1], dtype=torch.long).unsqueeze(0)

    print("[official 3/4] Running and capturing all 24 layers", flush=True)
    with torch.inference_mode():
        output = text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
        )
        logits = source.lm_head(output.last_hidden_state[:, -1, :]).detach().float().cpu()
    for handle in handles:
        handle.remove()

    payload = {
        "token_ids": ids.cpu(),
        "image_indices": torch.tensor(image_indices, dtype=torch.int64),
        "text_embeddings": text_embeddings,
        "inputs_embeds": inputs_embeds,
        "checkpoints": checkpoints,
        "logits": logits,
    }
    args.reference.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.reference)
    print(f"[official 4/4] Saved {len(checkpoints)} checkpoints to {args.reference}", flush=True)


def tensor_metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    actual = actual.detach().float().cpu().reshape(-1)
    expected = expected.detach().float().cpu().reshape(-1)
    if actual.shape != expected.shape:
        return {
            "shape_match": False,
            "actual_shape": list(actual.shape),
            "expected_shape": list(expected.shape),
        }
    delta = actual - expected
    actual_norm = torch.linalg.vector_norm(actual)
    expected_norm = torch.linalg.vector_norm(expected)
    denominator = actual_norm * expected_norm
    cosine = float(torch.dot(actual, expected) / denominator) if denominator > 0 else math.nan
    return {
        "shape_match": True,
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "rmse": float(torch.sqrt(torch.mean(delta * delta))),
        "cosine": cosine,
    }


def run_wrapper(args: argparse.Namespace) -> None:
    export_path = args.mnn_repo / "transformers" / "llm" / "export"
    sys.path.insert(0, str(export_path))
    from utils.model import LlmModel

    print("[wrapper 1/5] Loading official checkpoint reference", flush=True)
    reference = torch.load(args.reference, map_location="cpu", weights_only=True)
    export_args = SimpleNamespace(
        transformer_c4=True,
        lora_split=False,
        test=True,
        eagle_path=None,
        dflash_target_layer_ids=None,
        skip_weight=False,
    )
    print("[wrapper 2/5] Loading MNN export Python wrapper in FP32", flush=True)
    model = LlmModel.from_pretrained(str(args.model), args=export_args).float().eval()

    ids = reference["token_ids"]
    inputs_embeds = reference["inputs_embeds"]
    with torch.inference_mode():
        wrapper_text_embeddings = model.embedding(ids).reshape(1, ids.shape[1], HIDDEN_SIZE)
    embedding_metrics = tensor_metrics(wrapper_text_embeddings, reference["text_embeddings"])
    print(f"[wrapper 3/5] Text embedding: {json.dumps(embedding_metrics)}", flush=True)

    checkpoints, handles = install_hooks(model, wrapper=True)
    attention_mask = model.get_attention_mask(ids.shape[1])
    position_ids = model.get_position_ids(ids.shape[1], input_ids=ids)
    print("[wrapper 4/5] Running and comparing all 24 layers", flush=True)
    with torch.inference_mode():
        logits, _final_norm, _talker = model(
            inputs_embeds,
            attention_mask,
            position_ids,
            torch.tensor([-1], dtype=torch.int32),
        )
    for handle in handles:
        handle.remove()

    metrics = {"text_embeddings": embedding_metrics, "checkpoints": {}}
    first_divergence = None
    for name, expected in reference["checkpoints"].items():
        if name not in checkpoints:
            metrics["checkpoints"][name] = {"missing": True}
            if first_divergence is None:
                first_divergence = name
            continue
        actual = checkpoints[name]
        if name == "final_norm" and actual.shape != expected.shape:
            expected = expected[:, -actual.shape[1] :, :]
        item = tensor_metrics(actual, expected)
        metrics["checkpoints"][name] = item
        if (
            first_divergence is None
            and item.get("shape_match")
            and (item["cosine"] < args.cosine_threshold or item["max_abs"] > args.max_abs_threshold)
        ):
            first_divergence = name
        print(
            f"{name:36s} max={item.get('max_abs', float('nan')):.7g} "
            f"mean={item.get('mean_abs', float('nan')):.7g} "
            f"cos={item.get('cosine', float('nan')):.9f}",
            flush=True,
        )

    metrics["logits"] = tensor_metrics(logits, reference["logits"])
    metrics["first_divergence"] = first_divergence
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"[wrapper 5/5] First divergence: {first_divergence or 'none'}", flush=True)
    print(f"Report: {args.report}", flush=True)


def run_all(args: argparse.Namespace) -> None:
    common = [
        "--model",
        str(args.model),
        "--visual-embedding",
        str(args.visual_embedding),
        "--prompt",
        str(args.prompt),
        "--reference",
        str(args.reference),
        "--report",
        str(args.report),
        "--mnn-repo",
        str(args.mnn_repo),
        "--max-abs-threshold",
        str(args.max_abs_threshold),
        "--cosine-threshold",
        str(args.cosine_threshold),
    ]
    for stage in ("official", "wrapper"):
        command = [sys.executable, str(Path(__file__).resolve()), stage, *common]
        print(f"[all] Starting {stage} stage", flush=True)
        subprocess.run(command, check=True, env=os.environ.copy())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("official", "wrapper", "all"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--visual-embedding", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--mnn-repo", type=Path, required=True)
    parser.add_argument("--max-abs-threshold", type=float, default=5e-3)
    parser.add_argument("--cosine-threshold", type=float, default=0.99999)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "official":
        run_official(args)
    elif args.stage == "wrapper":
        run_wrapper(args)
    else:
        run_all(args)


if __name__ == "__main__":
    main()
