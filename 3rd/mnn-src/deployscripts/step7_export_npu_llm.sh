#!/bin/bash
set -euo pipefail

export SUDO_ASKPASS="${SUDO_ASKPASS:-/tmp/askpass.sh}"
sudo() { command sudo -A "$@"; }

# ============================================================
# MiniCPM-V 4.6 ???????- Step 7: ?? NPU ?????
# ============================================================
# ??: ???MiniCPM-V 4.6 ???Qwen3.5 ???????24 ?????? ONNX???
#       ?????MTK ?????? DLA
#       - 24 ???prefill ???(?????? = 86 tokens)
#       - 24 ???decode ???(?? KV-cache ?? = 256)
#       - 1 ???final_norm + LM_head (???5 ?????logits)
# ============================================================

WORK_DIR="${HOME}/minicpmv46-workdir"
MODEL_DIR="${HOME}/models/minicpm-v-4.6"
NEURON_SDK_ROOT="${NEURON_SDK_ROOT:-${HOME}/neuron_sdk}"

echo "=========================================="
echo "Step 7: ?? NPU ?????(24???LLM)"
echo "=========================================="

source "${HOME}/minicpmv46-env/bin/activate"

cd "$WORK_DIR"
mkdir -p npu_llm
cd npu_llm

# --- 7.1 ???? reference (?? prefill ????? ---
echo ""
echo "[1/5] ???? embedding reference..."
if [ -f "../npu_vision/vision_504x392/visual.reference.npz" ]; then
    cp ../npu_vision/vision_504x392/visual.reference.npz ./visual.reference.npz
    echo "  ??????????? reference"
else
    echo "  ????? reference ???????? Step 6"
    echo "  ?? placeholder reference..."
    python -c "
import numpy as np
import os
# ???????reference
os.makedirs('.', exist_ok=True)
# Placeholder - ????? MiniCPM-V 4.6 vision tower ????
np.savez_compressed('visual.reference.npz', image_embeds=np.zeros((63, 1024), dtype=np.float32))
print('Created placeholder reference')
"
fi

# --- 7.2 ??????prefill + decode ???---
echo ""
echo "[2/5] ?? 24 ???prefill ???+ 24 ???decode ???+ head..."
# ??????????????
python "${HOME}/minicpmv46-letterbox/deployments/minicpmv46/neuropilot/export_all_shards.py" \
    --model "$MODEL_DIR" \
    --vision-reference ./visual.reference.npz \
    --output-dir ./npu_shards \
    --capacity 256 \
    2>&1 | tee export_shards.log

# --- 7.3 ?? prefill ?????(??) ---
echo ""
echo "[3/5] ?? prefill ?? (????)..."
# export_prefill_probes.py ???export_decode_probes.py ??????ONNX??vision reference ???????????
python "${HOME}/minicpmv46-letterbox/deployments/minicpmv46/neuropilot/export_prefill_probes.py" \
    --model "$MODEL_DIR" \
    --output-dir ./prefill_probe \
    --sequence 87 \
    2>&1 | tee export_prefill_probe.log

# --- 7.4 ?? decode ?????(??) ---
echo ""
echo "[4/5] ?? decode ?? (????)..."
python "${HOME}/minicpmv46-letterbox/deployments/minicpmv46/neuropilot/export_decode_probes.py" \
    --model "$MODEL_DIR" \
    --output-dir ./decode_probe \
    --capacity 256 \
    --past-length 87 \
    2>&1 | tee export_decode_probe.log

# --- 7.5 ???? (?? NeuroPilot SDK ??) ---
echo ""
echo "[5/5] ?? NPU ??..."
if [ ! -d "$NEURON_SDK_ROOT" ]; then
    echo "  [!!] NEURON_SDK_ROOT ???"
    echo "  ????? export NEURON_SDK_ROOT=/path/to/neuron_sdk"
    echo "  ?????? (??):"
    echo "    bash ${HOME}/minicpmv46-letterbox/deployments/minicpmv46/neuropilot/build_all_npu_android.sh"
else
    echo "  ??????24 ???prefill + 24 ???decode + head..."
    VISION_REFERENCE="$(pwd)/visual.reference.npz" \
    bash "${HOME}/minicpmv46-letterbox/deployments/minicpmv46/neuropilot/build_all_npu_android.sh" \
        2>&1 | tee build_all_npu.log
fi

echo ""
echo "=========================================="
echo "Step 7 ??!"
echo "=========================================="
echo ""
echo "????: npu_llm/npu_shards/"
echo "  prefill/layer_00~23/ - 24 ???prefill ONNX"
echo "  decode/layer_00~23/   - 24 ???decode ONNX"
echo "  head/                 - final_norm + LM_head"
echo "  assets/               - prompt tokens, RoPE ? embedding"
echo ""
echo "????? Step 8 - ?? MNN ??"
