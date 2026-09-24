#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"
MNN_ROOT="${MNN_ROOT:-$REPO_ROOT}"
MODEL="${MODEL:?Set MODEL to the MiniCPM-V 4.6 model directory}"
IMAGE="${IMAGE:?Set IMAGE to the test image used to fix the one-tile graph}"
OUTPUT="${OUTPUT:-$ROOT/build/mnn}"
EXPORT_PYTHON="${EXPORT_PYTHON:-python3}"
MNNCONVERT="${MNNCONVERT:-$MNN_ROOT/build/MNNConvert}"

if [[ ! -x "$MNNCONVERT" ]]; then
    printf 'MNNConvert is not executable: %s\n' "$MNNCONVERT" >&2
    exit 1
fi

mkdir -p "$OUTPUT" "$OUTPUT/vision-onnx"

if [[ ! -s "$OUTPUT/llm.mnn" || ! -s "$OUTPUT/llm.mnn.weight" ]]; then
    "$EXPORT_PYTHON" "$MNN_ROOT/transformers/llm/export/llmexport.py" \
        --path "$MODEL" \
        --tokenizer_path "$MODEL" \
        --dst_path "$OUTPUT" \
        --export mnn \
        --quant_bit 4 \
        --quant_block 64 \
        --lm_quant_bit 4 \
        --lm_quant_block 64 \
        --mnnconvert "$MNNCONVERT"
else
    printf 'Reusing MNN Q4 language model in %s\n' "$OUTPUT"
fi

if [[ ! -s "$OUTPUT/vision-onnx/visual.onnx" ]]; then
    "$EXPORT_PYTHON" "$ROOT/neuropilot/export_vision.py" \
        --model "$MODEL" \
        --image "$IMAGE" \
        --output-dir "$OUTPUT/vision-onnx"
fi

if [[ ! -s "$OUTPUT/visual.mnn" || ! -s "$OUTPUT/visual.mnn.weight" ]]; then
    "$MNNCONVERT" \
        -f ONNX \
        --modelFile "$OUTPUT/vision-onnx/visual.onnx" \
        --MNNModel "$OUTPUT/visual.mnn" \
        --bizCode MNN \
        --weightQuantBits 8 \
        --weightQuantBlock 128 \
        --saveExternalData \
        --transformerFuse=0 \
        2>&1 | tee "$OUTPUT/visual-convert.log"
else
    printf 'Reusing MNN Q8 vision model in %s\n' "$OUTPUT"
fi

printf 'MNN deployment models are ready in %s\n' "$OUTPUT"
