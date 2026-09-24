#!/bin/bash
set -euo pipefail

export SUDO_ASKPASS="${SUDO_ASKPASS:-/tmp/askpass.sh}"
sudo() { command sudo -A "$@"; }

# ============================================================
# MiniCPM-V 4.6 全流程复???- Step 5: 构建 MNN Host Converter
# ============================================================

WORK_DIR="${HOME}/minicpmv46-workdir"
MNN_REPO="${HOME}/minicpmv46-letterbox"

echo "=========================================="
echo "Step 5: 构建 MNN Host Converter"
echo "=========================================="

# 检查依???
if ! command -v cmake &>/dev/null; then
    echo "安装 cmake..."
    sudo apt-get install -y cmake
fi
if ! command -v ninja &>/dev/null; then
    echo "安装 ninja..."
    sudo apt-get install -y ninja-build
fi

# --- 5.1 链接或克???MNN 仓库 ---
echo ""
echo "[1/3] 准备 MNN 源码..."
if [ -d "$MNN_REPO" ]; then
    echo "  MNN repo 已存??? $MNN_REPO"
    # 确保是最???
    cd "$MNN_REPO"
    git fetch origin
    echo "  当前分支: $(git branch --show-current)"
    echo "  当前 commit: $(git rev-parse --short HEAD)"
else
    echo "  克隆 MNN 仓库..."
    # ???phonevlm-letterbox ???origin 拉取
    # 注意：实际项目在 GitHub alibaba/MNN
    echo "  注意: 使用官方 alibaba/MNN (phonevlm-letterbox 是定制分???"
    cd "$HOME"
    git clone https://github.com/alibaba/MNN.git "$MNN_REPO"
    cd "$MNN_REPO"
    # 切换到包???phonevlm-letterbox 修复的分???
    # 由于不确定是否有对应分支，我们使用最新的 main
    echo "  使用最???main 分支"
fi

# --- 5.2 CMake 配置 ---
echo ""
echo "[2/3] CMake 配置 MNN Converter..."
cd "$MNN_REPO"
mkdir -p build_converter
cd build_converter

# MNNConvert 需??? Converter ONNX TF ???
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DMNN_BUILD_CONVERTER=ON \
    -DMNN_BUILD_TRAIN=OFF \
    -DMNN_BUILD_TEST=OFF \
    -DMNN_BUILD_LLM=OFF \
    -DMNN_BUILD_TOOLS=OFF \
    -DMNN_SUPPORT_TRANSFORMER_FUSE=OFF \
    2>&1 | tail -10

# --- 5.3 编译 ---
echo ""
echo "[3/3] 编译 MNNConvert..."
make -j$(nproc) MNNConvert 2>&1 | tail -10

# 验证
if [ -f "./MNNConvert" ]; then
    echo "  ???MNNConvert 编译成功: $(realpath ./MNNConvert)"
    echo "  版本: $(./MNNConvert -v 2>&1 | head -3)"
else
    echo "  ???MNNConvert 编译失败"
    exit 1
fi

echo ""
echo "=========================================="
echo "Step 5 完成!"
echo "=========================================="
echo "MNNConvert: $MNN_REPO/build_converter/MNNConvert"
echo ""
echo "下一步: Step 6 - 导出 NPU 视觉"
