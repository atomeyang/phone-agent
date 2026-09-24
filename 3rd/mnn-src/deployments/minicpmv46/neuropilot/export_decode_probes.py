#!/usr/bin/env python3
"""Export one MiniCPM-V 4.6 decode layer of each attention type.

The exported graphs expose every cache tensor explicitly.  This keeps Python
cache mutation out of ONNX and gives the NeuroPilot compiler static shapes.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import DynamicCache
from transformers.models.minicpmv4_6.modeling_minicpmv4_6 import (
    MiniCPMV4_6ForConditionalGeneration,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb


def l2norm(value: Tensor, eps: float = 1e-6) -> Tensor:
    return value * torch.rsqrt((value * value).sum(dim=-1, keepdim=True) + eps)


def decomposed_softplus(value: Tensor) -> Tensor:
    """Numerically stable Softplus using operators supported by MTK Converter."""
    return F.relu(value) + torch.log(1.0 + torch.exp(-torch.abs(value)))


def repeat_kv_static(value: Tensor, repeats: int) -> Tensor:
    if repeats == 1:
        return value
    return torch.cat(
        tuple(
            value[:, head : head + 1]
            for head in range(value.shape[1])
            for _ in range(repeats)
        ),
        dim=1,
    )


class LinearAttentionDecodeLayer(nn.Module):
    """One Qwen3.5 Gated Delta decoder layer with explicit recurrent state."""

    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        if layer.layer_type != "linear_attention":
            raise ValueError(f"Expected linear_attention, got {layer.layer_type}")
        self.input_layernorm = layer.input_layernorm
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.mlp = layer.mlp
        self.linear_attn = layer.linear_attn

    def forward(
        self,
        hidden_states: Tensor,
        conv_state: Tensor,
        recurrent_state: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        residual = hidden_states
        normalized = self.input_layernorm(hidden_states)
        mixer = self.linear_attn

        mixed_qkv = mixer.in_proj_qkv(normalized).transpose(1, 2)
        conv_input = torch.cat((conv_state, mixed_qkv), dim=-1)
        new_conv_state = conv_input[:, :, -mixer.conv_kernel_size :]
        mixed_qkv = F.conv1d(
            conv_input,
            mixer.conv1d.weight,
            mixer.conv1d.bias,
            groups=mixer.conv_dim,
        )
        mixed_qkv = F.silu(mixed_qkv[:, :, -1:]).transpose(1, 2)

        query, key, value = torch.split(
            mixed_qkv,
            (mixer.key_dim, mixer.key_dim, mixer.value_dim),
            dim=-1,
        )
        query = l2norm(query.reshape(1, 1, mixer.num_k_heads, mixer.head_k_dim))
        key = l2norm(key.reshape(1, 1, mixer.num_k_heads, mixer.head_k_dim))
        value = value.reshape(1, 1, mixer.num_v_heads, mixer.head_v_dim)
        query = query.transpose(1, 2)[:, :, 0, :] * (mixer.head_k_dim**-0.5)
        key = key.transpose(1, 2)[:, :, 0, :]
        value = value.transpose(1, 2)[:, :, 0, :]

        beta = torch.sigmoid(mixer.in_proj_b(normalized)).transpose(1, 2)
        decay = (
            -mixer.A_log.float().exp()
            * decomposed_softplus(mixer.in_proj_a(normalized).float() + mixer.dt_bias)
        )
        decay = decay.transpose(1, 2).exp().unsqueeze(-1)

        new_recurrent_state = recurrent_state * decay
        memory = (new_recurrent_state * key.unsqueeze(-1)).sum(dim=-2)
        delta = (value - memory) * beta
        new_recurrent_state = new_recurrent_state + key.unsqueeze(-1) * delta.unsqueeze(-2)
        mixed = (new_recurrent_state * query.unsqueeze(-1)).sum(dim=-2)
        mixed = mixed.unsqueeze(1).reshape(-1, mixer.head_v_dim)

        gate = mixer.in_proj_z(normalized).reshape(-1, mixer.head_v_dim)
        mixed = mixer.norm(mixed, gate).reshape(1, 1, mixer.value_dim)
        hidden_states = residual + mixer.out_proj(mixed)
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, new_conv_state, new_recurrent_state


class FullAttentionDecodeLayer(nn.Module):
    """One full-attention decoder layer with a fixed-capacity host KV cache."""

    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        if layer.layer_type != "full_attention":
            raise ValueError(f"Expected full_attention, got {layer.layer_type}")
        self.input_layernorm = layer.input_layernorm
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.mlp = layer.mlp
        self.self_attn = layer.self_attn

    def forward(
        self,
        hidden_states: Tensor,
        position_cos: Tensor,
        position_sin: Tensor,
        past_key: Tensor,
        past_value: Tensor,
        attention_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        residual = hidden_states
        normalized = self.input_layernorm(hidden_states)
        attention = self.self_attn
        input_shape = normalized.shape[:-1]
        hidden_shape = (*input_shape, -1, attention.head_dim)

        query, gate = torch.chunk(
            attention.q_proj(normalized).view(*input_shape, -1, attention.head_dim * 2),
            2,
            dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)
        query = attention.q_norm(query.view(hidden_shape)).transpose(1, 2)
        new_key = attention.k_norm(attention.k_proj(normalized).view(hidden_shape)).transpose(1, 2)
        new_value = attention.v_proj(normalized).view(hidden_shape).transpose(1, 2)
        query, new_key = apply_rotary_pos_emb(query, new_key, position_cos, position_sin)

        all_key = torch.cat((past_key, new_key), dim=2)
        all_value = torch.cat((past_value, new_value), dim=2)
        all_key = repeat_kv_static(all_key, attention.num_key_value_groups)
        all_value = repeat_kv_static(all_value, attention.num_key_value_groups)
        weights = torch.matmul(query, all_key.transpose(2, 3)) * attention.scaling
        weights = F.softmax(weights + attention_mask, dim=-1, dtype=torch.float32).to(query.dtype)
        mixed = torch.matmul(weights, all_value)
        mixed = mixed.transpose(1, 2).reshape(*input_shape, -1)
        mixed = mixed * torch.sigmoid(gate)

        hidden_states = residual + attention.o_proj(mixed)
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, new_key, new_value


def tensor_error(actual: Tensor, expected: Tensor) -> dict[str, float]:
    actual64 = actual.detach().double().reshape(1, -1)
    expected64 = expected.detach().double().reshape(1, -1)
    return {
        "max_abs": float((actual64 - expected64).abs().max()),
        "cosine": float(F.cosine_similarity(actual64, expected64).item()),
    }


def export_onnx(model: nn.Module, inputs: tuple[Tensor, ...], path: Path, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        torch.onnx.export(
            model,
            inputs,
            str(path),
            input_names=names[:-3],
            output_names=names[-3:],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
            external_data=True,
        )
    import onnx

    onnx.checker.check_model(str(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=256)
    parser.add_argument("--past-length", type=int, default=86)
    args = parser.parse_args()
    if not 0 < args.past_length < args.capacity:
        raise ValueError("past-length must be positive and smaller than capacity")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(20260805)
    print("[1/6] Loading original FP32 MiniCPM-V 4.6", flush=True)
    source = MiniCPMV4_6ForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval()
    text_model = source.model.language_model
    linear_layer = text_model.layers[0]
    full_layer = text_model.layers[3]
    rotary = text_model.rotary_emb
    source.model.vision_tower = None
    source.model.merger = None

    hidden = torch.randn(1, 1, 1024) * 0.2
    conv_state = torch.randn(1, 6144, 4) * 0.1
    recurrent_state = torch.randn(1, 16, 128, 128) * 0.01
    position_ids = torch.tensor([[args.past_length]], dtype=torch.int64)
    with torch.inference_mode():
        position_cos, position_sin = rotary(hidden, position_ids)

    print("[2/6] Validating explicit linear-attention state", flush=True)
    linear_probe = LinearAttentionDecodeLayer(linear_layer).eval()
    linear_cache = DynamicCache(config=source.config.text_config)
    linear_cache.update_conv_state(conv_state.clone(), 0)
    linear_cache.update_recurrent_state(recurrent_state.clone(), 0)
    with torch.inference_mode():
        expected_linear = linear_layer(
            hidden.clone(),
            position_embeddings=(position_cos, position_sin),
            past_key_values=linear_cache,
        )
        actual_linear = linear_probe(hidden.clone(), conv_state.clone(), recurrent_state.clone())
    linear_metrics = {
        "hidden": tensor_error(actual_linear[0], expected_linear),
        "conv_state": tensor_error(actual_linear[1], linear_cache.layers[0].conv_states),
        "recurrent_state": tensor_error(actual_linear[2], linear_cache.layers[0].recurrent_states),
    }
    print(json.dumps(linear_metrics, indent=2), flush=True)
    if max(item["max_abs"] for item in linear_metrics.values()) > 2e-5:
        raise RuntimeError("Explicit linear-attention decode does not match Transformers")

    print("[3/6] Validating explicit fixed-capacity full attention", flush=True)
    past_key_valid = torch.randn(1, 2, args.past_length, 256) * 0.1
    past_value_valid = torch.randn(1, 2, args.past_length, 256) * 0.1
    past_key = F.pad(past_key_valid, (0, 0, 0, args.capacity - args.past_length))
    past_value = F.pad(past_value_valid, (0, 0, 0, args.capacity - args.past_length))
    attention_mask = torch.full(
        (1, 1, 1, args.capacity + 1), torch.finfo(hidden.dtype).min, dtype=hidden.dtype
    )
    attention_mask[..., : args.past_length] = 0
    attention_mask[..., -1] = 0
    full_probe = FullAttentionDecodeLayer(full_layer).eval()
    full_cache = DynamicCache(config=source.config.text_config)
    full_cache.update(past_key_valid.clone(), past_value_valid.clone(), 3)
    with torch.inference_mode():
        expected_full = full_layer(
            hidden.clone(),
            position_embeddings=(position_cos, position_sin),
            attention_mask=None,
            past_key_values=full_cache,
        )
        actual_full = full_probe(
            hidden.clone(),
            position_cos,
            position_sin,
            past_key,
            past_value,
            attention_mask,
        )
    full_metrics = {
        "hidden": tensor_error(actual_full[0], expected_full),
        "new_key": tensor_error(actual_full[1], full_cache.layers[3].keys[..., -1:, :]),
        "new_value": tensor_error(actual_full[2], full_cache.layers[3].values[..., -1:, :]),
    }
    print(json.dumps(full_metrics, indent=2), flush=True)
    if max(item["max_abs"] for item in full_metrics.values()) > 2e-5:
        raise RuntimeError("Explicit full-attention decode does not match Transformers")

    metrics = {
        "capacity": args.capacity,
        "past_length": args.past_length,
        "linear": linear_metrics,
        "full": full_metrics,
    }
    (args.output_dir / "validation.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez(
        args.output_dir / "probe_inputs.npz",
        hidden=hidden.numpy(),
        conv_state=conv_state.numpy(),
        recurrent_state=recurrent_state.numpy(),
        position_cos=position_cos.numpy(),
        position_sin=position_sin.numpy(),
        past_key=past_key.numpy(),
        past_value=past_value.numpy(),
        attention_mask=attention_mask.numpy(),
    )

    print("[4/6] Exporting linear-attention decode ONNX", flush=True)
    export_onnx(
        linear_probe,
        (hidden, conv_state, recurrent_state),
        args.output_dir / "linear_decode" / "model.onnx",
        [
            "hidden_states",
            "conv_state",
            "recurrent_state",
            "hidden_out",
            "new_conv_state",
            "new_recurrent_state",
        ],
    )

    print("[5/6] Exporting full-attention decode ONNX", flush=True)
    export_onnx(
        full_probe,
        (hidden, position_cos, position_sin, past_key, past_value, attention_mask),
        args.output_dir / "full_decode" / "model.onnx",
        [
            "hidden_states",
            "position_cos",
            "position_sin",
            "past_key",
            "past_value",
            "attention_mask",
            "hidden_out",
            "new_key",
            "new_value",
        ],
    )
    print("[6/6] Decode probes exported", flush=True)

    del source, text_model, linear_layer, full_layer, linear_probe, full_probe
    gc.collect()


if __name__ == "__main__":
    main()
