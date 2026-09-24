#!/usr/bin/env python3
"""Export the MiniCPM-V 4.6 vision stack for one dynamic-resolution tile."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from transformers import AutoProcessor
from transformers.models.minicpmv4_6.modeling_minicpmv4_6 import (
    MiniCPMV4_6ForConditionalGeneration,
)


class DynamicTileVision(nn.Module):
    """Vectorized single-tile graph without trace-time image dimensions."""

    def __init__(self, vision_tower: nn.Module, merger: nn.Module) -> None:
        super().__init__()
        self.embeddings = vision_tower.embeddings
        self.layers = vision_tower.encoder.layers
        self.insert_layer_id = vision_tower.config.insert_layer_id
        self.vit_merger = vision_tower.vit_merger
        self.post_layernorm = vision_tower.post_layernorm
        self.merger = merger
        self.patch_size = vision_tower.config.patch_size

    @staticmethod
    def attention(module: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, sequence, _ = hidden_states.shape
        heads = module.num_heads
        head_dim = module.head_dim
        query = module.q_proj(hidden_states).view(batch, sequence, heads, head_dim).transpose(1, 2)
        key = module.k_proj(hidden_states).view(batch, sequence, heads, head_dim).transpose(1, 2)
        value = module.v_proj(hidden_states).view(batch, sequence, heads, head_dim).transpose(1, 2)
        weights = torch.matmul(query, key.transpose(2, 3)) * module.scaling
        weights = torch.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
        attended = torch.matmul(weights, value).transpose(1, 2).contiguous()
        return module.out_proj(attended.reshape(batch, sequence, heads * head_dim))

    def encoder_layer(self, layer: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.attention(layer.self_attn, layer.layer_norm1(hidden_states))
        hidden_states = residual + hidden_states
        residual = hidden_states
        return residual + layer.mlp(layer.layer_norm2(hidden_states))

    def merge_vit_windows(
        self, hidden_states: torch.Tensor, height: int, width: int
    ) -> tuple[torch.Tensor, int, int]:
        module = self.vit_merger
        residual = hidden_states
        normalized = module.layer_norm1(hidden_states)
        channels = normalized.shape[-1]
        windows = (
            normalized.view(1, height // 2, 2, width // 2, 2, channels)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(-1, 4, channels)
        )
        attended = self.attention(module.self_attn, windows)
        attended = (
            attended.view(1, height // 2, width // 2, 2, 2, channels)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(1, height * width, channels)
        )
        hidden_states = residual + attended

        windows = (
            hidden_states.view(1, height // 2, 2, width // 2, 2, channels)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(-1, 4, channels)
        )
        merged = module.pre_norm(windows.reshape(-1, 4 * channels))
        merged = module.linear_2(module.act(module.linear_1(merged)))
        merged = merged + windows.mean(dim=1)
        return merged.unsqueeze(0), height // 2, width // 2

    def merge_to_llm(
        self, hidden_states: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        channels = hidden_states.shape[-1]
        windows = (
            hidden_states.view(1, height // 2, 2, width // 2, 2, channels)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(-1, 4 * channels)
        )
        return self.merger.mlp[0](windows)

    def forward(
        self, pixel_values: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        height = pixel_values.shape[2] // self.patch_size
        width = pixel_values.shape[3] // self.patch_size

        patches = self.embeddings.patch_embedding(pixel_values)
        hidden_states = patches.flatten(2).transpose(1, 2)
        hidden_states = hidden_states + self.embeddings.position_embedding(position_ids).unsqueeze(0)

        for index, layer in enumerate(self.layers):
            hidden_states = self.encoder_layer(layer, hidden_states)
            if index == self.insert_layer_id:
                hidden_states, height, width = self.merge_vit_windows(
                    hidden_states, height, width
                )

        hidden_states = self.post_layernorm(hidden_states)
        return self.merge_to_llm(hidden_states, height, width).unsqueeze(1)


def unpatchify(pixel_values: torch.Tensor, target_size: torch.Tensor) -> torch.Tensor:
    height = int(target_size[0])
    width = int(target_size[1])
    channels = pixel_values.shape[1]
    patch = pixel_values.shape[2]
    return (
        pixel_values[0]
        .reshape(channels, patch, height, width, patch)
        .permute(0, 2, 1, 3, 4)
        .reshape(1, channels, height * patch, width * patch)
    )


def make_position_ids(height: int, width: int, side: int) -> torch.Tensor:
    rows = torch.div(torch.arange(height) * side, height, rounding_mode="floor")
    columns = torch.div(torch.arange(width) * side, width, rounding_mode="floor")
    return (rows[:, None] * side + columns).flatten().to(torch.int32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.output_dir / "visual_dynamic.onnx"

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    image = Image.open(args.image).convert("RGB")
    batch = processor.image_processor(
        image,
        return_tensors="pt",
        max_slice_nums=1,
    )
    target_size = batch["target_sizes"][0]
    pixel_values = unpatchify(batch["pixel_values"].float(), target_size)

    source = MiniCPMV4_6ForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval()
    vision_tower = source.model.vision_tower.float()
    merger = source.model.merger.float()

    with torch.inference_mode():
        official = vision_tower(
            batch["pixel_values"].float(),
            target_sizes=batch["target_sizes"],
            use_vit_merger=True,
        )
        official_features = torch.cat(
            merger(official.last_hidden_state, batch["target_sizes"] // 2), dim=0
        ).float()

    source.model.vision_tower = None
    source.model.merger = None
    del source
    dynamic = DynamicTileVision(vision_tower, merger).float().eval()
    del vision_tower, merger
    gc.collect()

    position_ids = make_position_ids(
        int(target_size[0]),
        int(target_size[1]),
        dynamic.embeddings.num_patches_per_side,
    )
    with torch.inference_mode():
        actual = dynamic(pixel_values, position_ids)
    actual_features = actual[:, 0, :]
    max_error = float((actual_features - official_features).abs().max())
    cosine = float(
        torch.nn.functional.cosine_similarity(
            actual_features.flatten(), official_features.flatten(), dim=0
        )
    )
    if max_error > 5e-4 or cosine < 0.999999:
        raise RuntimeError(
            f"dynamic wrapper mismatch: max_error={max_error}, cosine={cosine}"
        )

    with torch.inference_mode():
        torch.onnx.export(
            dynamic,
            (pixel_values, position_ids),
            str(onnx_path),
            input_names=["pixel_values", "position_ids"],
            output_names=["image_embeds"],
            dynamic_axes={
                "pixel_values": {2: "image_height", 3: "image_width"},
                "position_ids": {0: "patch_count"},
                "image_embeds": {0: "visual_tokens"},
            },
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )

    import onnx

    onnx.checker.check_model(str(onnx_path))
    metadata = {
        "source_model": str(args.model),
        "source_image": str(args.image),
        "reference_shape": list(pixel_values.shape),
        "reference_target_size": target_size.tolist(),
        "reference_output_shape": list(actual.shape),
        "reference_max_error": max_error,
        "reference_cosine": cosine,
        "dynamic_dimensions": ["image_height", "image_width", "visual_tokens"],
    }
    (args.output_dir / "visual_dynamic_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    np.asarray(pixel_values, dtype=np.float32).tofile(
        args.output_dir / "visual_dynamic_reference_input.bin"
    )
    np.asarray(position_ids, dtype=np.int32).tofile(
        args.output_dir / "visual_dynamic_reference_positions.bin"
    )
    np.asarray(actual, dtype=np.float32).tofile(
        args.output_dir / "visual_dynamic_reference_output.bin"
    )
    print(
        f"saved {onnx_path}: input={tuple(pixel_values.shape)}, "
        f"output={tuple(actual.shape)}, max_error={max_error:.6g}, cosine={cosine:.9f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
