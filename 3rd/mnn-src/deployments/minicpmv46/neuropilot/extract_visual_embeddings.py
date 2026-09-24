#!/usr/bin/env python3
"""Extract one-tile MiniCPM-V 4.6 visual embeddings without exporting ONNX."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor
from transformers.models.minicpmv4_6.modeling_minicpmv4_6 import (
    MiniCPMV4_6ForConditionalGeneration,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

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
        raise RuntimeError(f"expected one tile, got {target_sizes.tolist()}")

    source = MiniCPMV4_6ForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval()
    vision_tower = source.model.vision_tower.float().eval()
    merger = source.model.merger.float().eval()
    source.model.vision_tower = None
    source.model.merger = None
    del source
    gc.collect()

    with torch.inference_mode():
        vision_output = vision_tower(
            pixel_values,
            target_sizes=target_sizes,
            use_vit_merger=True,
        )
        image_features = merger(
            vision_output.last_hidden_state,
            target_sizes // 2,
        )
        embeddings = torch.cat(image_features, dim=0).unsqueeze(1).float().cpu().numpy()

    if embeddings.ndim != 3 or embeddings.shape[1:] != (1, 1024):
        raise RuntimeError(f"unexpected visual embedding shape: {embeddings.shape}")
    if not np.isfinite(embeddings).all():
        raise RuntimeError("visual embeddings contain non-finite values")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    embeddings.tofile(args.output)
    print(
        f"saved {args.output}: shape={embeddings.shape}, "
        f"range=[{embeddings.min():.6g}, {embeddings.max():.6g}]"
    )


if __name__ == "__main__":
    main()
