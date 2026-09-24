#!/usr/bin/env python3
"""Export exact-shape MiniCPM-V 4.6 vision profiles for NeuroPilot."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from transformers.models.minicpmv4_6.modeling_minicpmv4_6 import (
    MiniCPMV4_6ForConditionalGeneration,
)

from export_vision_dynamic import DynamicTileVision, make_position_ids


PATCH_SIZE = 14
SIZE_DIVISOR = 56


class StaticVisionProfile(nn.Module):
    def __init__(self, visual: nn.Module, position_ids: torch.Tensor) -> None:
        super().__init__()
        self.visual = visual
        self.register_buffer("position_ids", position_ids)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.visual(pixel_values, self.position_ids)


def parse_shape(value: str) -> tuple[int, int]:
    try:
        height, width = (int(part) for part in value.lower().split("x", 1))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"invalid tile shape: {value}") from exc
    if height <= 0 or width <= 0 or height % SIZE_DIVISOR or width % SIZE_DIVISOR:
        raise argparse.ArgumentTypeError(
            f"tile shape must contain positive multiples of {SIZE_DIVISOR}: {value}"
        )
    return height, width


def prepare_pixels(image: Image.Image, height: int, width: int) -> torch.Tensor:
    resized = image.resize((width, height), Image.Resampling.BICUBIC)
    values = np.asarray(resized, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(values).permute(2, 0, 1).unsqueeze(0).contiguous()


def patchify(pixel_values: torch.Tensor) -> torch.Tensor:
    _, channels, height, width = pixel_values.shape
    return (
        pixel_values[0]
        .reshape(channels, height // PATCH_SIZE, PATCH_SIZE, width // PATCH_SIZE, PATCH_SIZE)
        .permute(0, 2, 1, 3, 4)
        .reshape(1, channels, PATCH_SIZE, -1)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shape", type=parse_shape, action="append", required=True)
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(args.image).convert("RGB")

    print("[1/3] Loading MiniCPM-V 4.6 vision weights", flush=True)
    source = MiniCPMV4_6ForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval()
    vision_tower = source.model.vision_tower.float()
    merger = source.model.merger.float()
    source.model.vision_tower = None
    source.model.merger = None
    del source
    gc.collect()

    visual = DynamicTileVision(vision_tower, merger).float().eval()
    profiles = []
    print("[2/3] Verifying and exporting exact-shape profiles", flush=True)
    for height, width in dict.fromkeys(args.shape):
        pixel_values = prepare_pixels(image, height, width)
        target_sizes = torch.tensor(
            [[height // PATCH_SIZE, width // PATCH_SIZE]], dtype=torch.int64
        )
        position_ids = make_position_ids(
            height // PATCH_SIZE,
            width // PATCH_SIZE,
            visual.embeddings.num_patches_per_side,
        )
        with torch.inference_mode():
            official = vision_tower(
                patchify(pixel_values),
                target_sizes=target_sizes,
                use_vit_merger=True,
            )
            official_features = torch.cat(
                merger(official.last_hidden_state, target_sizes // 2), dim=0
            ).float()
            actual = visual(pixel_values, position_ids)

        actual_features = actual[:, 0, :]
        max_error = float((actual_features - official_features).abs().max())
        cosine = float(
            torch.nn.functional.cosine_similarity(
                actual_features.flatten(), official_features.flatten(), dim=0
            )
        )
        if max_error > 5e-4 or cosine < 0.999999:
            raise RuntimeError(
                f"profile {height}x{width} mismatch: "
                f"max_error={max_error}, cosine={cosine}"
            )

        stem = f"visual_{height}x{width}"
        profile_dir = args.output_dir / stem
        profile_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = profile_dir / "visual.onnx"
        profile_model = StaticVisionProfile(visual, position_ids).eval()
        with torch.inference_mode():
            torch.onnx.export(
                profile_model,
                (pixel_values,),
                str(onnx_path),
                input_names=["pixel_values"],
                output_names=["image_embeds"],
                opset_version=args.opset,
                do_constant_folding=True,
                dynamo=False,
                external_data=True,
            )

        import onnx

        onnx.checker.check_model(str(onnx_path))
        np.asarray(pixel_values, dtype=np.float32).tofile(
            profile_dir / "reference_input.bin"
        )
        np.asarray(actual, dtype=np.float32).tofile(
            profile_dir / "reference_output.bin"
        )
        profile = {
            "height": height,
            "width": width,
            "patch_count": int(position_ids.numel()),
            "visual_tokens": int(actual.shape[0]),
            "max_error": max_error,
            "cosine": cosine,
            "onnx": str(onnx_path.relative_to(args.output_dir)),
        }
        profiles.append(profile)
        print(
            f"      {height}x{width}: tokens={actual.shape[0]}, "
            f"max_error={max_error:.6g}, cosine={cosine:.9f}",
            flush=True,
        )

    (args.output_dir / "visual_profiles.json").write_text(
        json.dumps({"profiles": profiles}, indent=2) + "\n", encoding="utf-8"
    )
    print("[3/3] Static vision profiles exported", flush=True)


if __name__ == "__main__":
    main()
