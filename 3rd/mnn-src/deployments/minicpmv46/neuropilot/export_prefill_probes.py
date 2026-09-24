#!/usr/bin/env python3
"""Export one MiniCPM-V 4.6 prefill layer of each attention type."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import DynamicCache
from transformers.models.minicpmv4_6.modeling_minicpmv4_6 import (
    MiniCPMV4_6ForConditionalGeneration,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb

from export_decode_probes import decomposed_softplus, export_onnx, l2norm, repeat_kv_static, tensor_error


class LinearAttentionPrefillLayer(nn.Module):
    """One Gated Delta prefill layer using an explicit recurrent scan."""

    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        if layer.layer_type != "linear_attention":
            raise ValueError(f"Expected linear_attention, got {layer.layer_type}")
        self.input_layernorm = layer.input_layernorm
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.mlp = layer.mlp
        self.linear_attn = layer.linear_attn

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        residual = hidden_states
        normalized = self.input_layernorm(hidden_states)
        mixer = self.linear_attn
        batch, sequence, _ = normalized.shape

        mixed_qkv = mixer.in_proj_qkv(normalized).transpose(1, 2)
        new_conv_state = mixed_qkv[:, :, -mixer.conv_kernel_size :]
        mixed_qkv = F.conv1d(
            mixed_qkv,
            mixer.conv1d.weight,
            mixer.conv1d.bias,
            padding=mixer.conv_kernel_size - 1,
            groups=mixer.conv_dim,
        )[:, :, :sequence]
        mixed_qkv = F.silu(mixed_qkv).transpose(1, 2)

        query, key, value = torch.split(
            mixed_qkv,
            (mixer.key_dim, mixer.key_dim, mixer.value_dim),
            dim=-1,
        )
        query = l2norm(query.reshape(batch, sequence, mixer.num_k_heads, mixer.head_k_dim))
        key = l2norm(key.reshape(batch, sequence, mixer.num_k_heads, mixer.head_k_dim))
        value = value.reshape(batch, sequence, mixer.num_v_heads, mixer.head_v_dim)
        query = query.transpose(1, 2) * (mixer.head_k_dim**-0.5)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)

        beta = torch.sigmoid(mixer.in_proj_b(normalized)).transpose(1, 2)
        decay = (
            -mixer.A_log.float().exp()
            * decomposed_softplus(mixer.in_proj_a(normalized).float() + mixer.dt_bias)
        ).transpose(1, 2)

        state = torch.zeros(
            batch,
            mixer.num_v_heads,
            mixer.head_k_dim,
            mixer.head_v_dim,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        outputs = []
        for token in range(sequence):
            state = state * decay[:, :, token].exp().unsqueeze(-1).unsqueeze(-1)
            key_token = key[:, :, token]
            value_token = value[:, :, token]
            memory = (state * key_token.unsqueeze(-1)).sum(dim=-2)
            delta = (value_token - memory) * beta[:, :, token].unsqueeze(-1)
            state = state + key_token.unsqueeze(-1) * delta.unsqueeze(-2)
            outputs.append((state * query[:, :, token].unsqueeze(-1)).sum(dim=-2))

        mixed = torch.stack(outputs, dim=2).transpose(1, 2).reshape(-1, mixer.head_v_dim)
        gate = mixer.in_proj_z(normalized).reshape(-1, mixer.head_v_dim)
        mixed = mixer.norm(mixed, gate).reshape(batch, sequence, mixer.value_dim)
        hidden_states = residual + mixer.out_proj(mixed)
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, new_conv_state, state


class FullAttentionPrefillLayer(nn.Module):
    """One causal full-attention prefill layer with static RoPE and mask."""

    def __init__(self, layer: nn.Module, position_cos: Tensor, position_sin: Tensor, mask: Tensor) -> None:
        super().__init__()
        if layer.layer_type != "full_attention":
            raise ValueError(f"Expected full_attention, got {layer.layer_type}")
        self.input_layernorm = layer.input_layernorm
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.mlp = layer.mlp
        self.self_attn = layer.self_attn
        self.register_buffer("position_cos", position_cos)
        self.register_buffer("position_sin", position_sin)
        self.register_buffer("mask", mask)

    def forward(self, hidden_states: Tensor) -> tuple[Tensor, Tensor, Tensor]:
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
        key = attention.k_norm(attention.k_proj(normalized).view(hidden_shape)).transpose(1, 2)
        value = attention.v_proj(normalized).view(hidden_shape).transpose(1, 2)
        query, key = apply_rotary_pos_emb(query, key, self.position_cos, self.position_sin)

        repeated_key = repeat_kv_static(key, attention.num_key_value_groups)
        repeated_value = repeat_kv_static(value, attention.num_key_value_groups)
        weights = torch.matmul(query, repeated_key.transpose(2, 3)) * attention.scaling
        weights = F.softmax(weights + self.mask, dim=-1, dtype=torch.float32).to(query.dtype)
        mixed = torch.matmul(weights, repeated_value)
        mixed = mixed.transpose(1, 2).reshape(*input_shape, -1) * torch.sigmoid(gate)
        hidden_states = residual + attention.o_proj(mixed)
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, key, value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence", type=int, default=86)
    args = parser.parse_args()
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
    hidden = torch.randn(1, args.sequence, 1024) * 0.2
    position_ids = torch.arange(args.sequence, dtype=torch.int64).unsqueeze(0)
    with torch.inference_mode():
        position_cos, position_sin = text_model.rotary_emb(hidden, position_ids)
    causal_mask = torch.full(
        (1, 1, args.sequence, args.sequence),
        torch.finfo(hidden.dtype).min,
        dtype=hidden.dtype,
    ).triu(diagonal=1)

    print("[2/6] Validating recurrent-scan linear prefill", flush=True)
    linear_probe = LinearAttentionPrefillLayer(linear_layer).eval()
    linear_cache = DynamicCache(config=source.config.text_config)
    with torch.inference_mode():
        expected_linear = linear_layer(
            hidden.clone(),
            position_embeddings=(position_cos, position_sin),
            attention_mask=None,
            past_key_values=linear_cache,
        )
        actual_linear = linear_probe(hidden.clone())
    linear_metrics = {
        "hidden": tensor_error(actual_linear[0], expected_linear),
        "conv_state": tensor_error(actual_linear[1], linear_cache.layers[0].conv_states),
        "recurrent_state": tensor_error(actual_linear[2], linear_cache.layers[0].recurrent_states),
    }
    print(json.dumps(linear_metrics, indent=2), flush=True)
    if linear_metrics["hidden"]["cosine"] < 0.99999 or linear_metrics["hidden"]["max_abs"] > 5e-3:
        raise RuntimeError("Recurrent-scan linear prefill differs too much from chunked Transformers prefill")

    print("[3/6] Validating static full-attention prefill", flush=True)
    full_probe = FullAttentionPrefillLayer(
        full_layer, position_cos, position_sin, causal_mask
    ).eval()
    full_cache = DynamicCache(config=source.config.text_config)
    with torch.inference_mode():
        expected_full = full_layer(
            hidden.clone(),
            position_embeddings=(position_cos, position_sin),
            attention_mask=causal_mask,
            past_key_values=full_cache,
        )
        actual_full = full_probe(hidden.clone())
    full_metrics = {
        "hidden": tensor_error(actual_full[0], expected_full),
        "key": tensor_error(actual_full[1], full_cache.layers[3].keys),
        "value": tensor_error(actual_full[2], full_cache.layers[3].values),
    }
    print(json.dumps(full_metrics, indent=2), flush=True)
    if max(item["max_abs"] for item in full_metrics.values()) > 2e-5:
        raise RuntimeError("Static full-attention prefill does not match Transformers")

    metrics = {"sequence": args.sequence, "linear": linear_metrics, "full": full_metrics}
    (args.output_dir / "validation.json").write_text(json.dumps(metrics, indent=2) + "\n")

    print("[4/6] Exporting linear-attention prefill ONNX", flush=True)
    export_onnx(
        linear_probe,
        (hidden,),
        args.output_dir / "linear_prefill" / "model.onnx",
        ["hidden_states", "hidden_out", "conv_state", "recurrent_state"],
    )
    print("[5/6] Exporting full-attention prefill ONNX", flush=True)
    export_onnx(
        full_probe,
        (hidden,),
        args.output_dir / "full_prefill" / "model.onnx",
        ["hidden_states", "hidden_out", "key", "value"],
    )
    print("[6/6] Prefill probes exported", flush=True)


if __name__ == "__main__":
    main()
