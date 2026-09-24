#!/bin/bash
set -euo pipefail

export SUDO_ASKPASS="${SUDO_ASKPASS:-/tmp/askpass.sh}"
sudo() { command sudo -A "$@"; }

# ============================================================
# MiniCPM-V 4.6 全流程复???- Step 10: 准备 runtime assets 目录结构
# ============================================================

WORK_DIR="${HOME}/minicpmv46-workdir"
ASSETS_DIR="${WORK_DIR}/runtime-assets"
LETTERBOX_ASSETS_DIR="${WORK_DIR}/runtime-assets-letterbox"

echo "=========================================="
echo "Step 10: 准备 runtime assets"
echo "=========================================="

mkdir -p "$ASSETS_DIR"
mkdir -p "$LETTERBOX_ASSETS_DIR"

# --- 10.1 创建 Hybrid Runtime Assets (Adaptive) ---
echo ""
echo "[1/3] 创建 Hybrid (Adaptive) runtime assets..."
cd "$ASSETS_DIR"

# 10.1.1 创建子目???
mkdir -p models

# 10.1.2 复制 LLM 文件
echo "  复制 LLM 模型..."
cp "$WORK_DIR/mnn_export/llm.mnn" ./llm.mnn
cp "$WORK_DIR/mnn_export/llm.mnn.weight" ./llm.mnn.weight 2>/dev/null || {
    echo "  注意: llm.mnn.weight 不存在，可能权重已合并到 llm.mnn"
}
cp "$WORK_DIR/mnn_export/tokenizer.mtok" ./tokenizer.mtok

# 10.1.3 复制 LLM 配置
echo "  创建 LLM 配置..."
cat > llm_config.json << 'LLEOF'
{
  "model_path": "llm.mnn",
  "tokenizer_path": "tokenizer.mtok",
  "llm_type": "minicpmv4_6",
  "quantize": "q4",
  "max_length": 4096
}
LLEOF

# 10.1.4 复制视觉 DLA 文件 (Adaptive 需???3 ???profile)
echo "  复制 NPU 视觉 DLA..."
if [ -f "$WORK_DIR/npu_vision/vision_504x392/visual_fp16_mt6899.dla" ]; then
    cp "$WORK_DIR/npu_vision/vision_504x392/visual_fp16_mt6899.dla" ./models/visual_504x392_fp16_mt6899.dla
    echo "  ???visual_504x392_fp16_mt6899.dla"
else
    echo "  ???504x392 DLA 不存???
fi

# 如果有其???profile
for PROFILE_DIR in vision_560x392 vision_392x504; do
    if [ -d "$WORK_DIR/npu_vision/$PROFILE_DIR" ]; then
        # 提取尺寸
        HEIGHT=$(echo "$PROFILE_DIR" | cut -d'x' -f1 | tr -d 'vision_')
        WIDTH=$(echo "$PROFILE_DIR" | cut -d'x' -f2)
        if [ -f "$WORK_DIR/npu_vision/$PROFILE_DIR/visual_fp16_mt6899.dla" ]; then
            cp "$WORK_DIR/npu_vision/$PROFILE_DIR/visual_fp16_mt6899.dla" \
               "./models/visual_${HEIGHT}x${WIDTH}_fp16_mt6899.dla"
            echo "  ???visual_${HEIGHT}x${WIDTH}_fp16_mt6899.dla"
        fi
    fi
done

# 10.1.5 复制 NeuroPilot ???
echo "  复制 NeuroPilot SDK ???.."
if [ -n "${NEURON_SDK_ROOT:-}" ] && [ -d "$NEURON_SDK_ROOT" ]; then
    cp "${NEURON_SDK_ROOT}/mt6899/lib/libneuron_runtime.so.9.3.1" ./libneuron_runtime.so.9.3.1
    cp "${NEURON_SDK_ROOT}/mt6899/lib/libneuron_adapter.so.9.3.1" ./libneuron_adapter.so.9.3.1
    echo "  ???libneuron_runtime.so.9.3.1"
    echo "  ???libneuron_adapter.so.9.3.1"
else
    echo "  ???NEURON_SDK_ROOT 未设置，跳过 NeuroPilot ???
fi

# 10.1.6 复制 runner ???bridge
echo "  复制 runner ???bridge..."
cp "$WORK_DIR/build/android/minicpmv46-hybrid-runner" ./minicpmv46-hybrid-runner
cp "${HOME}/minicpmv46-letterbox/deployments/minicpmv46/android/minicpmv46-neuropilot-bridge.sh" ./minicpmv46-neuropilot-bridge.sh
chmod +x ./minicpmv46-hybrid-runner ./minicpmv46-neuropilot-bridge.sh

# 10.1.7 创建 config.json
cat > config.json << 'CONFIGEOF'
{
  "vision": {
    "npu": true,
    "profiles": [
      {"width": 560, "height": 392, "file": "models/visual_560x392_fp16_mt6899.dla"},
      {"width": 392, "height": 504, "file": "models/visual_392x504_fp16_mt6899.dla"},
      {"width": 504, "height": 392, "file": "models/visual_504x392_fp16_mt6899.dla"}
    ]
  },
  "llm": {
    "type": "mnn",
    "model": "llm.mnn",
    "tokenizer": "tokenizer.mtok",
    "quantization": "q4"
  }
}
CONFIGEOF

echo "  ???config.json"

# --- 10.2 创建 Letterbox Runtime Assets ---
echo ""
echo "[2/3] 创建 Letterbox runtime assets..."
cd "$LETTERBOX_ASSETS_DIR"

mkdir -p models

# 复制 LLM 文件
cp "$WORK_DIR/mnn_export/llm.mnn" ./llm.mnn
cp "$WORK_DIR/mnn_export/tokenizer.mtok" ./tokenizer.mtok

cat > llm_config.json << 'LLEOF'
{
  "model_path": "llm.mnn",
  "tokenizer_path": "tokenizer.mtok",
  "llm_type": "minicpmv4_6",
  "quantize": "q4",
  "max_length": 4096
}
LLEOF

# Letterbox 只有 504x392 一???profile
if [ -f "$WORK_DIR/npu_vision/vision_504x392/visual_fp16_mt6899.dla" ]; then
    cp "$WORK_DIR/npu_vision/vision_504x392/visual_fp16_mt6899.dla" ./models/visual_504x392_fp16_mt6899.dla
fi

# 复制 NeuroPilot ???
if [ -n "${NEURON_SDK_ROOT:-}" ] && [ -d "$NEURON_SDK_ROOT" ]; then
    cp "${NEURON_SDK_ROOT}/mt6899/lib/libneuron_runtime.so.9.3.1" ./libneuron_runtime.so.9.3.1
    cp "${NEURON_SDK_ROOT}/mt6899/lib/libneuron_adapter.so.9.3.1" ./libneuron_adapter.so.9.3.1
fi

# Letterbox runner
if [ -f "$WORK_DIR/build/android/minicpmv46-letterbox-runner" ]; then
    cp "$WORK_DIR/build/android/minicpmv46-letterbox-runner" ./minicpmv46-letterbox-runner
fi
cp "${HOME}/minicpmv46-letterbox/deployments/minicpmv46/android/minicpmv46-letterbox-bridge.sh" ./minicpmv46-letterbox-bridge.sh
chmod +x ./*.sh ./*-runner 2>/dev/null || true

cat > config.json << 'CONFIGEOF'
{
  "vision": {
    "npu": true,
    "profiles": [
      {"width": 504, "height": 392, "file": "models/visual_504x392_fp16_mt6899.dla"}
    ]
  },
  "llm": {
    "type": "mnn",
    "model": "llm.mnn",
    "tokenizer": "tokenizer.mtok",
    "quantization": "q4"
  }
}
CONFIGEOF

# --- 10.3 统计大小 ---
echo ""
echo "[3/3] 统计 runtime assets 大小..."
echo ""
echo "  Adaptive (${ASSETS_DIR}):"
du -sh "$ASSETS_DIR" 2>/dev/null
find "$ASSETS_DIR" -type f | wc -l
echo ""
echo "  Letterbox (${LETTERBOX_ASSETS_DIR}):"
du -sh "$LETTERBOX_ASSETS_DIR" 2>/dev/null
find "$LETTERBOX_ASSETS_DIR" -type f | wc -l

echo ""
echo "=========================================="
echo "Step 10 完成!"
echo "=========================================="
echo ""
echo "Assets 目录:"
echo "  Adaptive: $ASSETS_DIR ($(du -sh "$ASSETS_DIR" 2>/dev/null | cut -f1))"
echo "  Letterbox: $LETTERBOX_ASSETS_DIR ($(du -sh "$LETTERBOX_ASSETS_DIR" 2>/dev/null | cut -f1))"
echo ""
echo "下一??? Step 11 - 构建 Android APK"
