#!/bin/bash
set -euo pipefail

# ============================================================
# MiniCPM-V 4.6 全流程复�?- Step 4: 安装 NDK �?NeuroPilot SDK
# ============================================================

echo "=========================================="
echo "Step 4: 安装 Android NDK �?NeuroPilot SDK"
echo "=========================================="

# --- 4.1 Android NDK ---
NDK_VERSION="android-ndk-r27b"
NDK_DEST="${ANDROID_NDK_ROOT:-$HOME/android-ndk}"

echo ""
echo "[1/2] Android NDK..."
if [ -d "$NDK_DEST" ]; then
    echo "  NDK 已安�? $NDK_DEST"
else
    echo "  下载 $NDK_VERSION (Linux)..."
    cd /tmp
    # WSL Ubuntu 直接下载 Linux 版本
    wget -q --show-progress \
        "https://dl.google.com/android/repository/${NDK_VERSION}-linux.zip" \
        -O "${NDK_VERSION}-linux.zip" \
        || curl -fsSL \
            "https://dl.google.com/android/repository/${NDK_VERSION}-linux.zip" \
            -o "${NDK_VERSION}-linux.zip"
    
    echo "  解压�?$HOME..."
    unzip -q "${NDK_VERSION}-linux.zip" -d "$HOME"
    if [ -d "$HOME/${NDK_VERSION}" ]; then
        mv "$HOME/${NDK_VERSION}" "$NDK_DEST"
    fi
    rm -f /tmp/${NDK_VERSION}-linux.zip
    
    echo "  NDK 安装完成: $NDK_DEST"
fi

# 验证 NDK
if [ -f "${NDK_DEST}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++" ]; then
    echo "  �?NDK clang++: ${NDK_DEST}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++"
else
    echo "  �?NDK clang++ 未找�?"
    echo "  提示: WSL 默认 x86_64, NDK 必须 Linux 版本"
fi

# --- 4.2 NeuroPilot SDK ---
echo ""
echo "[2/2] NeuroPilot SDK..."
echo ""
echo "  NeuroPilot SDK 需要从 MTK 开发者网站下载�?
echo "  下载地址: https://neuropilot.mediatek.com/resources/latest/library"
echo "  需要注�?MTK 开发者账号�?
echo ""
echo "  需要下载的文件:"
echo "    - neuron_sdk_<version>.tar.gz (Linux 版本)"
echo "    - neuropilot_converter_<version>.tar.gz (ONNX 编译�?"
echo ""
# Auto-detected SDK; set NEURON_SDK_ROOT_OVERRIDE env var to override
SDK_PATH="${NEURON_SDK_ROOT_OVERRIDE:-/mnt/d/codes/Engineering/mtk/neuropilot-sdk-premium-9.0.9-build20260629/neuron_sdk}"

if [ -n "$SDK_PATH" ] && [ -d "$SDK_PATH" ]; then
    export NEURON_SDK_ROOT="$SDK_PATH"
    echo "  NEURON_SDK_ROOT=$NEURON_SDK_ROOT"
    
    # 验证
    if [ -f "${NEURON_SDK_ROOT}/linux-x86_64/include/neuron/api/RuntimeV2.h" ]; then
        echo "  �?RuntimeV2.h 找到"
    fi
    if [ -d "${NEURON_SDK_ROOT}/mt6899/lib" ]; then
        echo "  �?MT6899 lib 找到"
    fi
else
    echo "  跳过 - 稍后手动设置 NEURON_SDK_ROOT"
fi

# --- 4.3 设置环境变量 ---
echo ""
echo "[3/3] 环境变量..."
echo ""
echo "请在 ~/.bashrc �?~/.zshrc 中添加以下内�?"
echo ""
cat << 'EOF'
# MiniCPM-V 4.6 部署
export ANDROID_NDK_ROOT="$HOME/android-ndk"
export NEURON_SDK_ROOT="$HOME/path/to/neuron_sdk"  # 替换为实际路�?
export PATH="$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin:$PATH"
EOF

# 立即生效
export ANDROID_NDK_ROOT="$NDK_DEST"
echo "ANDROID_NDK_ROOT=$ANDROID_NDK_ROOT"

if [ -n "${SDK_PATH:-}" ]; then
    export NEURON_SDK_ROOT="$SDK_PATH"
    echo "NEURON_SDK_ROOT=$NEURON_SDK_ROOT"
fi

echo ""
echo "=========================================="
echo "Step 4 完成!"
echo "=========================================="
echo ""
echo "下一�? Step 5 - 构建 MNN Host Converter"
echo ""
echo "如果 NeuroPilot SDK 未安装，请先从以下地址获取:"
echo "  https://neuropilot.mediatek.com/resources/latest/library"
