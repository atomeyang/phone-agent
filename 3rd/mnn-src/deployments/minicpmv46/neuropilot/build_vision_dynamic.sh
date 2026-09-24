#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:?Set MODEL to the MiniCPM-V 4.6 model directory}"
IMAGE="${IMAGE:?Set IMAGE to a reference image}"
OUTPUT="${OUTPUT:-$ROOT/build/vision-profiles}"
EXPORT_PYTHON="${EXPORT_PYTHON:-python3}"
CONVERTER_PYTHON="${CONVERTER_PYTHON:-python3}"
NEURON_SDK_ROOT="${NEURON_SDK_ROOT:?Set NEURON_SDK_ROOT to the SDK neuron_sdk directory}"
NCC_ROOT="${NCC_ROOT:-$NEURON_SDK_ROOT/linux-x86_64}"
MTK_PLATFORM="${MTK_PLATFORM:-mt6899}"

mkdir -p "$OUTPUT"

PROFILES=(560x392 392x504)
EXPORT_ARGS=()
for profile in "${PROFILES[@]}"; do
    EXPORT_ARGS+=(--shape "$profile")
done

if [[ ! -s "$OUTPUT/visual_profiles.json" ]]; then
    "$EXPORT_PYTHON" "$ROOT/export_vision_profiles.py" \
        --model "$MODEL" \
        --image "$IMAGE" \
        --output-dir "$OUTPUT" \
        "${EXPORT_ARGS[@]}" \
        2>&1 | tee "$OUTPUT/export.log"
fi

for profile in "${PROFILES[@]}"; do
    stem="visual_${profile}"
    profile_dir="$OUTPUT/$stem"
    dla="$profile_dir/visual_fp16_${MTK_PLATFORM}.dla"
    if [[ -s "$dla" ]]; then
        printf 'Vision profile DLA: %s\n' "$dla"
        continue
    fi
    if [[ ! -s "$profile_dir/visual.tflite" ]]; then
        "$CONVERTER_PYTHON" "$ROOT/mtk_convert_large_onnx.py" \
            --input_model_file "$profile_dir/visual.onnx" \
            --output_file "$profile_dir/visual.tflite" \
            >"$OUTPUT/convert-$profile.log" 2>&1
    fi

    LD_LIBRARY_PATH="$NCC_ROOT/lib" "$NCC_ROOT/bin/ncc-tflite" \
        "$profile_dir/visual.tflite" \
        --platform-config="$MTK_PLATFORM" \
        --disallow-bridge \
        --show-exec-plan \
        --gen-mem-info \
        --cast-fp32=fp16 \
        -O 3 \
        --mem-opt 3 \
        --dla-opt 2 \
        -d "$dla" \
        >"$OUTPUT/compile-$profile.log" 2>&1
    printf 'Vision profile DLA: %s/visual_fp16_%s.dla\n' \
        "$profile_dir" "$MTK_PLATFORM"
done
