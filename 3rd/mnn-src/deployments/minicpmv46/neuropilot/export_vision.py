#!/usr/bin/env python3
"""Export MiniCPM-V 4.6 vision tower and merger for one native image tile."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from transformers import AutoProcessor
from transformers.models.minicpmv4_6.modeling_minicpmv4_6 import (
    MiniCPMV4_6ForConditionalGeneration,
)


class MiniCPMV46Vision(nn.Module):
    def __init__(
        self,
        vision_tower: nn.Module,
        merger: nn.Module,
        target_sizes: torch.Tensor,
    ) -> None:
        super().__init__()
        self.vision_tower = vision_tower
        self.merger = merger
        # ONNX shape tensors are int64. Keeping fixed sizes int64 avoids mixed
        # int32/int64 Concat and Slice inputs in the traced graph.
        self.register_buffer("target_sizes", target_sizes.to(torch.int64))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        output = self.vision_tower(
            pixel_values,
            target_sizes=self.target_sizes,
            use_vit_merger=True,
        )
        image_features = self.merger(
            output.last_hidden_state,
            self.target_sizes // 2,
        )
        # MNN multimodal runtimes consume [sequence, batch, hidden].
        return torch.cat(image_features, dim=0).unsqueeze(1)


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
    onnx_path = args.output_dir / "visual.onnx"
    fixture_path = args.output_dir / "visual.reference.npz"

    print("[1/6] Preparing one native MiniCPM-V image tile", flush=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    image = Image.open(args.image).convert("RGB")
    batch = processor(
        images=image,
        text="<image>\nPlease describe this image.",
        return_tensors="pt",
        max_slice_nums=1,
    )
    pixel_values = batch["pixel_values"].float()
    target_sizes = batch["target_sizes"].to(torch.int64)
    if target_sizes.shape[0] != 1:
        raise RuntimeError(f"Expected one tile, got target_sizes={target_sizes.tolist()}")
    print(
        f"      pixel_values={tuple(pixel_values.shape)}, "
        f"target_sizes={target_sizes.tolist()}",
        flush=True,
    )

    print("[2/6] Loading original BF16 weights", flush=True)
    source = MiniCPMV4_6ForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval()
    vision_tower = source.model.vision_tower
    merger = source.model.merger
    source.model.vision_tower = None
    source.model.merger = None
    del source
    gc.collect()

    print("[3/6] Building FP32 export graph", flush=True)
    visual = MiniCPMV46Vision(vision_tower, merger, target_sizes).float().eval()
    del vision_tower, merger
    gc.collect()

    print("[4/6] Computing PyTorch reference", flush=True)
    started = time.perf_counter()
    with torch.inference_mode():
        reference = visual(pixel_values)
    elapsed = time.perf_counter() - started
    reference_np = reference.cpu().numpy()
    if not np.isfinite(reference_np).all():
        raise RuntimeError("PyTorch visual output contains non-finite values")
    np.savez(
        fixture_path,
        pixel_values=pixel_values.numpy(),
        target_sizes=target_sizes.numpy(),
        image_embeds=reference_np,
    )
    pixel_values.numpy().astype(np.float32).tofile(
        args.output_dir / "visual_input_fp32.bin"
    )
    reference_np.astype(np.float32).tofile(
        args.output_dir / "visual_reference_fp32.bin"
    )
    print(
        f"      output={reference_np.shape}, time={elapsed:.3f}s, "
        f"range=[{reference_np.min():.6g}, {reference_np.max():.6g}]",
        flush=True,
    )

    print(f"[5/6] Exporting ONNX opset {args.opset}", flush=True)
    with torch.inference_mode():
        torch.onnx.export(
            visual,
            (pixel_values,),
            str(onnx_path),
            input_names=["pixel_values"],
            output_names=["image_embeds"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
            external_data=True,
        )

    print("[6/6] Checking ONNX graph", flush=True)
    import onnx

    # Passing the file path lets ONNX resolve external data relative to the model.
    onnx.checker.check_model(str(onnx_path))
    metadata = {
        "source_model": str(args.model),
        "source_image": str(args.image),
        "pixel_values_shape": list(pixel_values.shape),
        "target_sizes": target_sizes.tolist(),
        "image_embeds_shape": list(reference_np.shape),
        "visual_tokens_per_tile": int(reference_np.shape[0]),
        "hidden_size": int(reference_np.shape[-1]),
        "pytorch_wall_time_seconds": elapsed,
        "downsample_mode": "16x",
        "max_slice_nums_during_fixture_creation": 1,
    }
    (args.output_dir / "visual_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"      saved {onnx_path}", flush=True)


if __name__ == "__main__":
    main()
