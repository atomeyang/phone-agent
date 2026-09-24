#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:?Set MODEL to the MiniCPM-V 4.6 model directory}"
IMAGE="${IMAGE:?Set IMAGE to the test image used to fix the one-tile graph}"
OUTPUT="${OUTPUT:-$ROOT/build/vision}"
EXPORT_PYTHON="${EXPORT_PYTHON:-python3}"
CONVERTER_PYTHON="${CONVERTER_PYTHON:-python3}"
NEURON_SDK_ROOT="${NEURON_SDK_ROOT:?Set NEURON_SDK_ROOT to the SDK neuron_sdk directory}"
NCC_ROOT="${NCC_ROOT:-$NEURON_SDK_ROOT/linux-x86_64}"
MTK_PLATFORM="${MTK_PLATFORM:-mt6899}"

mkdir -p "$OUTPUT"

if [[ ! -s "$OUTPUT/visual.onnx" ]]; then
    "$EXPORT_PYTHON" "$ROOT/export_vision.py" \
        --model "$MODEL" \
        --image "$IMAGE" \
        --output-dir "$OUTPUT" \
        2>&1 | tee "$OUTPUT/export.log"
fi

if [[ ! -s "$OUTPUT/visual.npu.onnx" ]]; then
    "$EXPORT_PYTHON" "$ROOT/staticize_onnx.py" \
        --input "$OUTPUT/visual.onnx" \
        --output "$OUTPUT/visual.npu.onnx"
fi

if [[ ! -s "$OUTPUT/visual.tflite" ]]; then
    "$CONVERTER_PYTHON" "$ROOT/mtk_convert_large_onnx.py" \
        --input_model_file "$OUTPUT/visual.npu.onnx" \
        --output_file "$OUTPUT/visual.tflite" \
        >"$OUTPUT/convert.log" 2>&1
fi

LD_LIBRARY_PATH="$NCC_ROOT/lib" "$NCC_ROOT/bin/ncc-tflite" \
    "$OUTPUT/visual.tflite" \
    --platform-config="$MTK_PLATFORM" \
    --disallow-bridge \
    --show-exec-plan \
    --gen-mem-info \
    --cast-fp32=fp16 \
    -O 3 \
    --mem-opt 3 \
    --dla-opt 2 \
    -d "$OUTPUT/visual_fp16_${MTK_PLATFORM}.dla" \
    >"$OUTPUT/compile.log" 2>&1

printf 'Vision DLA: %s/visual_fp16_%s.dla\n' "$OUTPUT" "$MTK_PLATFORM"
