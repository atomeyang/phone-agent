#!/bin/bash
set -euo pipefail

export SUDO_ASKPASS="${SUDO_ASKPASS:-/tmp/askpass.sh}"
sudo() { command sudo -A "$@"; }

# ============================================================
# MiniCPM-V 4.6 全流程复???- Step 11: 构建 Android APK
# ============================================================

WORK_DIR="${HOME}/minicpmv46-workdir"
ASSETS_DIR="${WORK_DIR}/runtime-assets"
LETTERBOX_ASSETS_DIR="${WORK_DIR}/runtime-assets-letterbox"
MNN_REPO="${HOME}/minicpmv46-letterbox"

echo "=========================================="
echo "Step 11: 构建 Android APK"
echo "=========================================="

# --- 11.1 检???Android SDK ---
echo ""
echo "[1/3] 检???Android SDK..."
if ! command -v gradle &>/dev/null; then
    echo "  安装 Gradle..."
    # Android Studio 自带 Gradle，但命令行可能需要单独安???
    export ANDROID_SDK_ROOT="${ANDROID_SDK_ROOT:-${HOME}/Android/Sdk}"
    if [ -d "$ANDROID_SDK_ROOT" ]; then
        export PATH="$ANDROID_SDK_ROOT/cmdline-tools/latest/bin:$PATH"
    fi
fi

# 检???Android SDK
export ANDROID_SDK_ROOT="${ANDROID_SDK_ROOT:-${HOME}/Android/Sdk}"
if [ ! -d "$ANDROID_SDK_ROOT" ]; then
    echo "  注意: 未找???Android SDK"
    echo "  Windows 路径通常??? C:\\Users\\<name>\\AppData\\Local\\Android\\Sdk"
    echo "  可以???~/.bashrc 中设??? export ANDROID_SDK_ROOT=/mnt/c/Users/<name>/AppData/Local/Android/Sdk"
fi

# --- 11.2 准备 Gradle Wrapper ---
echo ""
echo "[2/3] 准备 Gradle..."
cd "$MNN_REPO/deployments/minicpmv46/android"

# 如果没有 gradlew，创建一???
if [ ! -f "gradlew" ]; then
    echo "  创建 Gradle Wrapper..."
    # ???Gradle 官网下载 wrapper
    GRADLE_VERSION="8.7"
    mkdir -p gradle/wrapper
    cat > gradle/wrapper/gradle-wrapper.properties << 'GRADLEOF'
distributionBase=GRADLE_USER_HOME
distributionPath=wrapper/dists
distributionUrl=https\://services.gradle.org/distributions/gradle-8.7-bin.zip
networkTimeout=10000
validateDistributionUrl=true
zipStoreBase=GRADLE_USER_HOME
zipStorePath=wrapper/dists
GRADLEOF
    
    wget -q "https://raw.githubusercontent.com/gradle/gradle/v${GRADLE_VERSION}/gradlew" -O gradlew
    wget -q "https://raw.githubusercontent.com/gradle/gradle/v${GRADLE_VERSION}/gradlew.bat" -O gradlew.bat
    wget -q "https://raw.githubusercontent.com/gradle/gradle/v${GRADLE_VERSION}/gradle/wrapper/gradle-wrapper.jar" -O gradle/wrapper/gradle-wrapper.jar
    chmod +x gradlew gradlew.bat
fi

# --- 11.3 构建 Adaptive APK ---
echo ""
echo "[3/3] 构建 Adaptive APK..."
cd "$MNN_REPO/deployments/minicpmv46/android"

# 检???assets 目录
if [ ! -d "$ASSETS_DIR" ]; then
    echo "  ???Adaptive runtime assets 不存??? $ASSETS_DIR"
    echo "  请先运行: bash step10_prepare_assets.sh"
else
    # 创建软链接到 assets
    mkdir -p app/src/dynamic/assets
    ln -sf "$ASSETS_DIR" app/src/dynamic/assets/runtime 2>/dev/null || \
        cp -r "$ASSETS_DIR" app/src/dynamic/assets/runtime
    
    echo "  构建 Dynamic Debug APK..."
    ./gradlew :app:assembleDynamicDebug \
        -PdynamicRuntimeAssetsDir="$ASSETS_DIR" \
        --no-daemon \
        2>&1 | tail -30
    
    APK_PATH="app/build/outputs/apk/dynamic/debug/app-dynamic-debug.apk"
    if [ -f "$APK_PATH" ]; then
        echo "  ???Adaptive APK: $(realpath $APK_PATH)"
    fi
fi

# --- 11.4 构建 Letterbox APK ---
echo ""
echo "[Bonus] 构建 Letterbox APK..."
if [ ! -d "$LETTERBOX_ASSETS_DIR" ]; then
    echo "  [!!] Letterbox runtime assets 不存在"
else
    mkdir -p app/src/letterbox/assets
    ln -sf "$LETTERBOX_ASSETS_DIR" app/src/letterbox/assets/runtime 2>/dev/null || \
        cp -r "$LETTERBOX_ASSETS_DIR" app/src/letterbox/assets/runtime
    
    echo "  构建 Letterbox Debug APK..."
    ./gradlew :app:assembleLetterboxDebug \
        -PletterboxRuntimeAssetsDir="$LETTERBOX_ASSETS_DIR" \
        --no-daemon \
        2>&1 | tail -30
    
    APK_PATH="app/build/outputs/apk/letterbox/debug/app-letterbox-debug.apk"
    if [ -f "$APK_PATH" ]; then
        echo "  ???Letterbox APK: $(realpath $APK_PATH)"
        cp "$APK_PATH" "$WORK_DIR/PhoneVLM-Letterbox-v1.2.1.apk"
        echo "  复制??? $WORK_DIR/PhoneVLM-Letterbox-v1.2.1.apk"
    fi
fi

echo ""
echo "=========================================="
echo "Step 11 完成!"
echo "=========================================="
echo ""
echo "APK 文件:"
ls -lh "$MNN_REPO/deployments/minicpmv46/android/app/build/outputs/apk/"*/*/*.apk 2>/dev/null || echo "  未找???APK"
echo ""
echo "下一步: Step 12 - 部署到 MT6899 真机"
