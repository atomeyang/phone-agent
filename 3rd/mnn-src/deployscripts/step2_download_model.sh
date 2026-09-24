#!/bin/bash
set -euo pipefail

# ============================================================
# MiniCPM-V 4.6 全流程复�?- Step 2: 下载模型权重
# ============================================================
# 需�? ~15GB 磁盘空间, HuggingFace 访问权限
# ============================================================

MODEL_DIR="${1:-${HOME}/models/minicpm-v-4.6}"
VENV_DIR="${HOME}/minicpmv46-env"

echo "=========================================="
echo "Step 2: 下载 MiniCPM-V 4.6 模型"
echo "=========================================="
echo "目标目录: $MODEL_DIR"

# 检�?venv
if [ ! -d "$VENV_DIR" ]; then
    echo "ERROR: 虚拟环境不存�? $VENV_DIR"
    echo "请先运行: bash step1_install_deps.sh"
    exit 1
fi

source "$VENV_DIR/bin/activate"

# 检�?transformers 版本
python -c "
from transformers import AutoProcessor, AutoModel
from transformers.models.minicpmv4_6.modeling_minicpmv4_6 import MiniCPMV4_6ForConditionalGeneration
print('All imports OK')
"

echo ""
echo "[1/3] 创建目录..."
mkdir -p "$MODEL_DIR"

echo ""
echo "[2/3] 下载 MiniCPM-V 4.6 from HuggingFace..."
echo "模型: openbmb/MiniCPM-V-4_6"
echo "大小: ~15GB (包括 vision + language model)"
echo ""

# 使用 huggingface-cli �?Python download
python -c "
import os
from transformers import AutoProcessor, AutoModelForCausalLM
from huggingface_hub import snapshot_download

model_id = 'openbmb/MiniCPM-V-4_6'
save_dir = os.path.expanduser('${MODEL_DIR}')

print(f'Downloading {model_id} to {save_dir}...')
print('This will take a while. Press Ctrl+C to cancel.')

# 方法1: snapshot_download (推荐，支持断点续�?
snapshot_download(
    repo_id=model_id,
    local_dir=save_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
)
print(f'Download complete: {save_dir}')
" 2>&1 | tail -20

echo ""
echo "[3/3] 验证模型文件..."
ls -lh "$MODEL_DIR/" | head -20

# 检查关键文�?
REQUIRED_FILES=("config.json" "model.safetensors" "preprocessor_config.json" "tokenizer.json" "tokenizer_config.json")
for f in "${REQUIRED_FILES[@]}"; do
    if [ -f "$MODEL_DIR/$f" ]; then
        echo "  �?$f"
    else
        echo "  �?$f MISSING"
    fi
done

echo ""
echo "=========================================="
echo "Step 2 完成!"
echo "=========================================="
echo "模型目录: $MODEL_DIR"
echo ""
echo "下一�? Step 3 - 准备测试图片"
