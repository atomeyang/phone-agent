#!/bin/bash
set -euo pipefail

export SUDO_ASKPASS="${SUDO_ASKPASS:-/tmp/askpass.sh}"
sudo() { command sudo -A "$@"; }

# ============================================================
# MiniCPM-V 4.6 全流程复???- Step 8: 导出 MNN Q4 LLM + MNN Q8 Vision
# ============================================================

WORK_DIR="${HOME}/minicpmv46-workdir"
MODEL_DIR="${HOME}/models/minicpm-v-4.6"
MNN_CONVERTER="${HOME}/minicpmv46-letterbox/build_converter/MNNConvert"

echo "=========================================="
echo "Step 8: 导出 MNN Q4 LLM + MNN Q8 Vision"
echo "=========================================="

source "${HOME}/minicpmv46-env/bin/activate"

cd "$WORK_DIR"
mkdir -p mnn_export
cd mnn_export

export PYTHONPATH="${HOME}/minicpmv46-letterbox/transformers/llm/export:${PYTHONPATH:-}"

# --- 8.1 导出 Q4 LLM (语言模型) ---
echo ""
echo "[1/2] 导出 MNN Q4 LLM..."
python "${HOME}/minicpmv46-letterbox/transformers/llm/export/llmexport.py" \
    --path "$MODEL_DIR" \
    --model_type minicpmv4_6 \
    --export mnn \
    --quantize q4 \
    --dst_path ./mnn_q4_llm \
    2>&1 | tee export_llm.log

# --- 8.2 导出 Q8 Vision (视觉模型) ---
echo ""
echo "[2/2] 导出 MNN Q8 Vision..."
# MiniCPM-V 4.6 ???vision tower 需要单独导???
# 使用 MNNConvert 直接转换
if [ -f "$MNN_CONVERTER" ]; then
    echo "  使用 MNNConvert 转换 vision ONNX..."
    
    # Q8 vision from ONNX
    if [ -f "../npu_vision/vision_504x392/visual.onnx" ]; then
        "$MNN_CONVERTER" \
            --modelFile ../npu_vision/vision_504x392/visual.onnx \
            --MNNModel ./visual_q8.mnn \
            --fp16 \
            --quantizeWeight 8 \
            --bizCode MNN \
            2>&1 | tee convert_vision.log
    fi
else
    echo "  MNNConvert 未找到，请先运行 Step 5"
fi

# --- 8.3 导出 tokenizer ---
echo ""
echo "[3/3] 复制 tokenizer..."
cp "$MODEL_DIR/tokenizer.json" ./tokenizer.json
cp "$MODEL_DIR/tokenizer_config.json" ./tokenizer_config.json
cp "$MODEL_DIR/special_tokens_map.json" ./special_tokens_map.json 2>/dev/null || true
cp "$MODEL_DIR/tokenizer.model" ./tokenizer.model 2>/dev/null || true

# 创建 .mtok 版本
python -c "
import os
import sys
sys.path.insert(0, '${HOME}/minicpmv46-letterbox/transformers/llm/export')
from utils.tokenizer import MNNTokenizer
tokenizer = MNNTokenizer.from_pretrained('$MODEL_DIR')
tokenizer.save('$WORK_DIR/mnn_export/tokenizer.mtok')
print('Tokenizer saved as tokenizer.mtok')
" 2>&1

echo ""
echo "=========================================="
echo "Step 8 完成!"
echo "=========================================="
echo ""
echo "输出目录: mnn_export/"
echo "  llm.mnn + llm.mnn.weight  - Q4 量化语言模型"
echo "  visual.mnn + visual.mnn.weight - Q8 量化视觉模型"
echo "  tokenizer.mtok - Tokenizer"
echo ""
echo "下一??? Step 9 - 构建 Android Native Runner"
