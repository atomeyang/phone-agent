#!/bin/bash
set -euo pipefail

export SUDO_ASKPASS="${SUDO_ASKPASS:-/tmp/askpass.sh}"
sudo() { command sudo -A "$@"; }

# ============================================================
# MiniCPM-V 4.6 ???????- Step 6: ?? NPU ?????
# ============================================================
# ??: ???MiniCPM-V 4.6 vision tower + merger ?????ONNX,
#       ?????MTK ?????? DLA (Deep Learning Accelerator) ??
# ============================================================

WORK_DIR="${HOME}/minicpmv46-workdir"
MODEL_DIR="${HOME}/models/minicpm-v-4.6"
NEURON_SDK_ROOT="${NEURON_SDK_ROOT:-${HOME}/neuron_sdk}"

echo "=========================================="
echo "Step 6: ?? NPU ??"
echo "=========================================="

source "${HOME}/minicpmv46-env/bin/activate"

cd "$WORK_DIR"
mkdir -p npu_vision
cd npu_vision

export PYTHONPATH="${HOME}/minicpmv46-letterbox/transformers/llm/export:${PYTHONPATH:-}"

# --- 6.1 ???? shape ONNX (Letterbox ???504x392) ---
echo ""
echo "[1/4] ???? shape ONNX (504x392)..."
python "${HOME}/minicpmv46-letterbox/deployments/minicpmv46/neuropilot/export_vision.py" \
    --model "$MODEL_DIR" \
    --image "$WORK_DIR/test.jpg" \
    --output-dir ./vision_504x392 \
    2>&1 | tee export_vision.log

# --- 6.2 ??????shape ONNX (Adaptive ??? ---
echo ""
echo "[2/4] ??????shape ONNX (for Adaptive profiles)..."
python "${HOME}/minicpmv46-letterbox/deployments/minicpmv46/neuropilot/export_vision_dynamic.py" \
    --model "$MODEL_DIR" \
    --image "$WORK_DIR/test.jpg" \
    --output-dir ./vision_dynamic \
    2>&1 | tee export_vision_dynamic.log

# --- 6.3 ??? ONNX (???? shape) ---
echo ""
echo "[3/4] ??? ONNX (??????)..."
python "${HOME}/minicpmv46-letterbox/deployments/minicpmv46/neuropilot/staticize_onnx.py" \
    --input ./vision_504x392/visual.onnx \
    --output ./vision_504x392/visual_static.onnx \
    2>&1 | tee staticize.log

# --- 6.4 ???MTK ?????? DLA ---
echo ""
echo "[4/4] ???MTK NPU ?????? DLA..."
echo "  NEURON_SDK_ROOT: $NEURON_SDK_ROOT"

if [ ! -d "$NEURON_SDK_ROOT" ]; then
    echo "  [!!] NEURON_SDK_ROOT ?????????"
    echo "  ????? export NEURON_SDK_ROOT=/path/to/neuron_sdk"
    echo "  ??????..."
else
    # ?? MTK ?????
    MTK_CONVERTER="${NEURON_SDK_ROOT}/linux-x86_64/bin/neuron compiler"  # ?????????
    if [ -f "$MTK_CONVERTER" ]; then
        echo "  ??????? $MTK_CONVERTER"
    else
        echo "  ?? MTK ?????.."
        find "$NEURON_SDK_ROOT" -name "neuron" -o -name "*.compiler" 2>/dev/null | head -5
        echo ""
        echo "  ?????? (????:"
        echo "    $MTK_CONVERTER \\"
        echo "      --input ./vision_504x392/visual_static.onnx \\"
        echo "      --output ./vision_504x392/visual_fp16_mt6899.dla \\"
        echo "      --target mt6899 \\"
        echo "      --data-type fp16"
    fi

    echo ""
    echo "  DLA ????: ./vision_504x392/"
    echo "  ????: visual_fp16_mt6899.dla"
fi

# --- 6.5 ?? profile ?? (Adaptive ????560x392 ???392x504) ---
echo ""
echo "[Bonus] ?? Adaptive profile ??..."
if [ -f "$WORK_DIR/test_wide.jpg" ]; then
    python "${HOME}/minicpmv46-letterbox/deployments/minicpmv46/neuropilot/export_vision.py" \
        --model "$MODEL_DIR" \
        --image "$WORK_DIR/test_wide.jpg" \
        --output-dir ./vision_392x504 \
        2>&1 | tee export_vision_392x504.log
fi

echo ""
echo "=========================================="
echo "Step 6 ??!"
echo "=========================================="
echo ""
echo "????:"
echo "  - vision_504x392/visual.onnx"
echo "  - vision_504x392/visual_fp16_mt6899.dla (??????)"
echo "  - vision_504x392/visual.reference.npz (PyTorch FP32 ????"
echo ""
echo "????? Step 7 - ?? NPU ?????(24???LLM)"
