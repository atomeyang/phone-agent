#!/bin/bash
set -euo pipefail

export SUDO_ASKPASS="${SUDO_ASKPASS:-/tmp/askpass.sh}"
sudo() { command sudo -A "$@"; }

# ============================================================
# MiniCPM-V 4.6 全流程复???- Step 9: 构建 Android Native Runner
# ============================================================

WORK_DIR="${HOME}/minicpmv46-workdir"
MNN_REPO="${HOME}/minicpmv46-letterbox"

echo "=========================================="
echo "Step 9: 构建 Android Native Runner"
echo "=========================================="

# --- 9.1 设置环境变量 ---
export ANDROID_NDK_ROOT="${ANDROID_NDK_ROOT:-${HOME}/android-ndk}"
export NEURON_SDK_ROOT="${NEURON_SDK_ROOT:-}"

if [ ! -d "$ANDROID_NDK_ROOT" ]; then
    echo "ERROR: Android NDK 未安???
    echo "请先运行: bash step4_install_ndk.sh"
    exit 1
fi

echo "  ANDROID_NDK_ROOT: $ANDROID_NDK_ROOT"
echo "  NEURON_SDK_ROOT: $NEURON_SDK_ROOT"

# --- 9.2 构建 MNN Android ---
echo ""
echo "[1/2] 构建 MNN Android ???.."
cd "$WORK_DIR"
mkdir -p build_mnn_android
cd build_mnn_android

cmake "$MNN_REPO" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="arm64-v8a" \
    -DANDROID_NATIVE_API_LEVEL=29 \
    -DMNN_BUILD_LLM=ON \
    -DMNN_BUILD_CONVERTER=OFF \
    -DMNN_BUILD_TRAIN=OFF \
    -DMNN_BUILD_TEST=OFF \
    -DMNN_LOW_MEMORY=ON \
    -DMNN_SUPPORT_TRANSFORMER_FUSE=ON \
    -DMNN_ARM82=ON \
    -DMNN_OPENCL=ON \
    2>&1 | tail -20

make -j$(nproc) MNN 2>&1 | tail -10

# 查找输出
MNN_LIB=""
for f in libMNN.so libMNN.a; do
    if [ -f "./libMNN.so" ]; then
        MNN_LIB="$(pwd)/libMNN.so"
        break
    fi
done

if [ -f "./libMNN.so" ]; then
    echo "  ???MNN Android ??? $(realpath ./libMNN.so)"
else
    echo "  ???MNN Android 库构建失???
    ls -la . 2>&1 | head -20
    exit 1
fi

# --- 9.3 构建 Hybrid Runner ---
echo ""
echo "[2/2] 构建 Hybrid Runner (minicpmv46-hybrid-runner)..."
export MNN_ANDROID_BUILD="$(pwd)"

cd "$WORK_DIR"

# 使用项目提供的构建脚???
bash "${MNN_REPO}/deployments/minicpmv46/runtime/build_hybrid_android.sh" \
    2>&1 | tee build_runner.log

# 验证输出
if [ -f "./build/android/minicpmv46-hybrid-runner" ]; then
    echo "  ???Hybrid Runner: $(realpath ./build/android/minicpmv46-hybrid-runner)"
else
    echo "  ???Hybrid Runner 构建失败"
    echo "  查看日志: ./build_runner.log"
fi

# --- 9.4 构建 Letterbox Runner (固定 504x392 专用) ---
echo ""
echo "[Bonus] 构建 Letterbox Runner..."
# Letterbox 只需要一个固???profile ???runner
# 可以直接复用 hybrid-runner，但可以减少编译的模???
echo "  使用已编译的 minicpmv46-hybrid-runner"
cp ./build/android/minicpmv46-hybrid-runner ./build/android/minicpmv46-letterbox-runner
echo "  ???Letterbox Runner: $(realpath ./build/android/minicpmv46-letterbox-runner)"

echo ""
echo "=========================================="
echo "Step 9 完成!"
echo "=========================================="
echo ""
echo "输出文件:"
echo "  build_mnn_android/libMNN.so"
echo "  build/android/minicpmv46-hybrid-runner"
echo "  build/android/minicpmv46-letterbox-runner"
echo ""
echo "下一??? Step 10 - 准备 runtime assets"
