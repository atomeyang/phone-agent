#!/bin/bash
set -euo pipefail

export SUDO_ASKPASS="${SUDO_ASKPASS:-/tmp/askpass.sh}"
sudo() { command sudo -A "$@"; }

# ============================================================
# Step 12 简化版: WSL Ubuntu 部署???MT6899 真机
# ============================================================
# 直接调用 Windows adb.exe,无需切换???Windows
# ============================================================

WORK_DIR="${WORK_DIR:-$HOME/minicpmv46-workdir}"
MNN_REPO="${HOME}/minicpmv46-letterbox"

echo "=========================================="
echo "Step 12: 部署???MT6899 真机 (WSL 模式)"
echo "=========================================="

# --- 定位 adb ---
echo ""
echo "[1/4] 定位 adb..."

ADB=""
# 优先??? WSL 自己???adb ???Windows adb.exe
if command -v adb &>/dev/null; then
    ADB="adb"
    echo "  ???WSL adb: $(which adb)"
else
    # 常见 Windows Android SDK 路径
    for path in \
        "/mnt/c/Users/Administrator/AppData/Local/Android/Sdk/platform-tools/adb.exe" \
        "/mnt/c/Users/$USER/AppData/Local/Android/Sdk/platform-tools/adb.exe" \
        "/mnt/c/Users/admin/AppData/Local/Android/Sdk/platform-tools/adb.exe" \
        "/mnt/c/Android/Sdk/platform-tools/adb.exe" \
        "$HOME/Android/Sdk/platform-tools/adb.exe"; do
        if [ -f "$path" ]; then
            ADB="$path"
            echo "  ???Windows adb.exe: $ADB"
            break
        fi
    done
fi

if [ -z "$ADB" ]; then
    echo "  ???未找???adb"
    echo "  安装方法:"
    echo "    - WSL: sudo apt install -y adb"
    echo "    - Windows: 安装 Android Studio (自动???Android SDK)"
    exit 1
fi

# --- 检查设???---
echo ""
echo "[2/4] 检查设备连???.."
$ADB devices

DEVICE_MODEL=$($ADB shell getprop ro.product.model 2>/dev/null | tr -d '\r\n' || echo "unknown")
DEVICE_PLATFORM=$($ADB shell getprop ro.board.platform 2>/dev/null | tr -d '\r\n' || echo "unknown")
echo "  设备: $DEVICE_MODEL"
echo "  平台: $DEVICE_PLATFORM"

if [ "$DEVICE_PLATFORM" != "mt6899" ]; then
    echo ""
    echo "  ???WARNING: 平台不是 mt6899 ($DEVICE_PLATFORM)"
    echo "  NeuroPilot NPU 可能不工作，???MNN CPU 模式可以"
fi

# --- root 检???---
echo ""
echo "[3/4] 检???root ADB..."
ADB_UID=$($ADB shell id -u 2>/dev/null | tr -d '\r\n' || echo "")
if [ "$ADB_UID" != "0" ]; then
    echo "  请求 root..."
    $ADB root
    sleep 2
    ADB_UID=$($ADB shell id -u 2>/dev/null | tr -d '\r\n' || echo "")
fi
echo "  UID: $ADB_UID"

# --- 部署 APK + assets ---
echo ""
echo "[4/4] 部署..."

LETTERBOX_APK="$WORK_DIR/PhoneVLM-Letterbox-v1.2.1.apk"
LETTERBOX_ASSETS="$WORK_DIR/runtime-assets-letterbox"

if [ ! -f "$LETTERBOX_APK" ]; then
    echo "  ???APK 不存??? $LETTERBOX_APK"
    echo "  请先运行 Step 1-11"
    exit 1
fi

echo "  APK:     $LETTERBOX_APK"
echo "  Assets:  $LETTERBOX_ASSETS"

# 转换???Windows 路径 (如果???adb.exe)
if [[ "$ADB" == *.exe ]]; then
    if command -v wslpath &>/dev/null; then
        WIN_APK=$(wslpath -w "$LETTERBOX_APK")
        WIN_ASSETS=$(wslpath -w "$LETTERBOX_ASSETS")
        echo "  Windows APK path: $WIN_APK"
    fi
fi

REMOTE_ROOT="/data/local/tmp/minicpmv46-neuropilot-letterbox"
PACKAGE_NAME="com.example.minicpmv46.neuropilot.letterbox"
ACTIVITY="$PACKAGE_NAME/com.example.minicpmv46.neuropilot.MainActivity"

# 安装 APK
echo ""
echo "  [a] 安装 APK..."
if [[ "$ADB" == *.exe ]] && [ -n "${WIN_APK:-}" ]; then
    $ADB install -r -d "$WIN_APK"
else
    $ADB install -r -d "$LETTERBOX_APK"
fi

# 创建远程目录
echo "  [b] 创建设备目录..."
$ADB shell "mkdir -p '$REMOTE_ROOT' && chmod -R 777 '$REMOTE_ROOT'"

# 推???assets
echo "  [c] 推???runtime assets (~1.5GB)..."
if [[ "$ADB" == *.exe ]] && [ -n "${WIN_ASSETS:-}" ]; then
    # adb.exe 需???Windows 路径
    $ADB push "$WIN_ASSETS\\." "$REMOTE_ROOT/" 2>&1 | tail -3
else
    $ADB push "$LETTERBOX_ASSETS/." "$REMOTE_ROOT/" 2>&1 | tail -3
fi

# 清理旧进???
echo "  [d] 清理旧进???.."
$ADB shell "for pid in \$(pidof minicpmv46-letterbox-runner 2>/dev/null); do kill -9 \$pid 2>/dev/null; done"

# 启动应用
echo "  [e] 启动应用..."
$ADB shell am force-stop "$PACKAGE_NAME"
$ADB shell am start -S -W -f 0x10008000 -n "$ACTIVITY"

echo ""
echo "=========================================="
echo "Step 12 完成!"
echo "=========================================="
echo ""
echo "后续操作:"
echo "  1. ???MT6899 屏幕上点 'Choose image' 选图"
echo "  2. 编辑 prompt (默认 'Please describe this image.')"
echo "  3. 点 'Run' 开始推理"
echo "  4. 看输出文???+ TTFT 指标"
echo ""
echo "日志查看:"
echo "  \$ADB logcat -c && \$ADB logcat | grep -iE 'minicpm|mnn|neuropilot'"