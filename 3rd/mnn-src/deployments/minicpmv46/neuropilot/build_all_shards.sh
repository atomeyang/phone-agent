#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:?Set MODEL to the MiniCPM-V 4.6 model directory}"
VISION_REFERENCE="${VISION_REFERENCE:-$ROOT/build/vision/visual.reference.npz}"
OUTPUT="${OUTPUT:-$ROOT/build/all-npu}"
EXPORT_PYTHON="${EXPORT_PYTHON:-python3}"
CONVERTER_PYTHON="${CONVERTER_PYTHON:-python3}"
CONVERTER="${CONVERTER:-$ROOT/mtk_convert_large_onnx.py}"
NEURON_SDK_ROOT="${NEURON_SDK_ROOT:?Set NEURON_SDK_ROOT to the SDK neuron_sdk directory}"
NCC_ROOT="${NCC_ROOT:-$NEURON_SDK_ROOT/linux-x86_64}"
MTK_PLATFORM="${MTK_PLATFORM:-mt6899}"
CAPACITY="${CAPACITY:-256}"

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT"

if [[ ! -s "$OUTPUT/manifest.json" ]]; then
    "$EXPORT_PYTHON" "$ROOT/export_all_shards.py" \
        --model "$MODEL" \
        --vision-reference "$VISION_REFERENCE" \
        --output-dir "$OUTPUT" \
        --capacity "$CAPACITY" \
        2>&1 | tee "$OUTPUT/export.log"
else
    printf 'Reusing exported shards in %s\n' "$OUTPUT"
fi

convert_compile() {
    local directory="$1"
    local label="$2"
    if [[ ! -s "$directory/model.tflite" ]]; then
        "$CONVERTER_PYTHON" "$CONVERTER" \
            --input_model_file "$directory/model.onnx" \
            --output_file "$directory/model.tflite" \
            >"$directory/convert.log" 2>&1
    fi
    if [[ ! -s "$directory/model_bf16_mt6899.dla" ]]; then
        LD_LIBRARY_PATH="$NCC_ROOT/lib" "$NCC_ROOT/bin/ncc-tflite" \
            "$directory/model.tflite" \
            --platform-config="$MTK_PLATFORM" \
            --disallow-bridge \
            --show-exec-plan \
            --gen-mem-info \
            --cast-fp32=bf16 \
            --disable-edpa \
            --disable-mvpu \
            -O 3 --mem-opt 3 --dla-opt 2 \
            -d "$directory/model_bf16_${MTK_PLATFORM}.dla" \
            >"$directory/compile.log" 2>&1
    fi
    printf 'Compiled %s\n' "$label"
}

for stage in prefill decode; do
    for index in $(seq -w 0 23); do
        convert_compile "$OUTPUT/$stage/layer_$index" "$stage layer $index"
    done
done
convert_compile "$OUTPUT/head" "final norm + LM head"

printf 'All text graphs compiled as pure MDLA: %s\n' "$OUTPUT"
