#!/bin/bash
set -euo pipefail

export SUDO_ASKPASS="${SUDO_ASKPASS:-/tmp/askpass.sh}"

# Non-interactive sudo askpass enabled
# ============================================================
# MiniCPM-V 4.6 全流程复�?- Step 1: Python 依赖安装
# ============================================================
# 需�? WSL Ubuntu, ~20GB 磁盘空间
# 预计时间: 30-60 分钟 (取决于网�?
# ============================================================

echo "=========================================="
echo "Step 1: 安装 Python 依赖"
echo "=========================================="
echo ""
echo "注意: 此脚本运行于 WSL Ubuntu 上"

# --- 1.1 系统依赖 ---
echo ""
echo "[1/7] 安装系统依赖..."
sudo -A apt-get update -qq
sudo -A apt-get install -y -qq \
    python3-pip \
    python3-dev \
    python3-venv \
    build-essential \
    git \
    wget \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    pkg-config \
    2>&1 | tail -5

# --- 1.2 创建 venv ---
VENV_DIR="$HOME/minicpmv46-env"
echo ""
echo "[2/7] 创建 Python 虚拟环境: $VENV_DIR"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "Python: $(python --version 2>&1)"
echo "pip:    $(pip --version 2>&1)"

# --- 1.3 安装 PyTorch (CPU + CUDA) ---
echo ""
echo "[3/7] 安装 PyTorch (CPU + CUDA 12.1)..."
# 优先尝试 CUDA 版本
pip install --upgrade pip wheel setuptools

# 先试�?CUDA 12.1 (较新系统兼容性更�?
pip install \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -5

# 检查是否安装成�?
python -c "import torch; print('PyTorch OK:', torch.__version__)" 2>/dev/null || {
    echo "CUDA 版本失败，尝�?CPU 版本..."
    pip install \
        torch==2.5.1 \
        torchvision==0.20.1 \
        torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -5
}

python -c "import torch; print('PyTorch OK:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

# --- 1.4 安装 transformers 和模型相�?---
echo ""
echo "[4/7] 安装 transformers, accelerate, huggingface_hub..."
pip install \
    transformers==4.51.0 \
    accelerate==1.5.0 \
    huggingface_hub==0.30.0 \
    tokenizers \
    2>&1 | tail -5

python -c "import transformers; print('transformers OK:', transformers.__version__)"

# --- 1.5 安装数值计算和图像�?---
echo ""
echo "[5/7] 安装 numpy, scipy, Pillow, opencv-python..."
pip install \
    numpy==1.26.4 \
    scipy==1.15.1 \
    pillow==11.1.0 \
    opencv-python-headless==4.11.0.86 \
    2>&1 | tail -5

python -c "import numpy; print('numpy OK:', numpy.__version__)"
python -c "from PIL import Image; print('PIL OK')"
python -c "import cv2; print('cv2 OK:', cv2.__version__)"

# --- 1.6 安装 ONNX ---
echo ""
echo "[6/7] 安装 ONNX..."
pip install \
    onnx==1.17.0 \
    onnxruntime==1.20.1 \
    2>&1 | tail -5

python -c "import onnx; print('onnx OK:', onnx.__version__)"

# --- 1.7 验证 MiniCPMV4_6 可导�?---
echo ""
echo "[7/7] 验证 MiniCPM-V 4.6 可导�?.."
pip install safetensors safetensors-align 2>&1 | tail -3

python -c "
from transformers.models.minicpmv4_6.modeling_minicpmv4_6 import MiniCPMV4_6ForConditionalGeneration
print('MiniCPMV4_6ForConditionalGeneration: OK')
from transformers import AutoProcessor
print('AutoProcessor: OK')
"

echo ""
echo "=========================================="
echo "Step 1 完成!"
echo "=========================================="
echo "激活虚拟环境命�?"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "下一�? Step 2 - 下载模型"
