#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:?Set MODEL to the MiniCPM-V 4.6 model directory}"
OUTPUT="${OUTPUT:-$ROOT/build/prefill-probes}"
EXPORT_PYTHON="${EXPORT_PYTHON:-python3}"
CONVERTER_PYTHON="${CONVERTER_PYTHON:-python3}"
CONVERTER="${CONVERTER:-$ROOT/mtk_convert_large_onnx.py}"
NEURON_SDK_ROOT="${NEURON_SDK_ROOT:?Set NEURON_SDK_ROOT to the SDK neuron_sdk directory}"
NCC_ROOT="${NCC_ROOT:-$NEURON_SDK_ROOT/linux-x86_64}"
MTK_PLATFORM="${MTK_PLATFORM:-mt6899}"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT"

"$EXPORT_PYTHON" "$ROOT/export_prefill_probes.py" \
    --model "$MODEL" --output-dir "$OUTPUT" --sequence 86 \
    2>&1 | tee "$OUTPUT/export.log"

for kind in linear_prefill full_prefill; do
    "$CONVERTER_PYTHON" "$CONVERTER" \
        --input_model_file "$OUTPUT/$kind/model.onnx" \
        --output_file "$OUTPUT/$kind/model.tflite" \
        >"$OUTPUT/$kind/convert.log" 2>&1
    LD_LIBRARY_PATH="$NCC_ROOT/lib" "$NCC_ROOT/bin/ncc-tflite" \
        "$OUTPUT/$kind/model.tflite" \
        --platform-config="$MTK_PLATFORM" \
        --disallow-bridge \
        --show-exec-plan \
        --gen-mem-info \
        --cast-fp32=bf16 \
        --disable-edpa \
        --disable-mvpu \
        -O 3 --mem-opt 3 --dla-opt 2 \
        -d "$OUTPUT/$kind/model_bf16_${MTK_PLATFORM}.dla" \
        >"$OUTPUT/$kind/compile.log" 2>&1
done

printf 'Both prefill layer types compiled as pure MDLA: %s\n' "$OUTPUT"
