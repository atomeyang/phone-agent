#!/bin/bash
set -euo pipefail

# ============================================================
# MiniCPM-V 4.6 全流程复�?- �?WSL 执行脚本
# ============================================================
#
# �?Windows PowerShell 中触�?
#   wsl -e bash /mnt/d/codes/Gallery/UDIdemos/PhoneVLM/phonevlm-letterbox/run_wsl.sh
#
# 或在 WSL Ubuntu �?
#   bash /mnt/d/codes/Gallery/UDIdemos/PhoneVLM/phonevlm-letterbox/run_wsl.sh
#
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="${WORK_DIR:-$HOME/minicpmv46-workdir}"
MODEL_DIR="${MODEL_DIR:-$HOME/models/minicpm-v-4.6}"
VENV_DIR="${VENV_DIR:-$HOME/minicpmv46-env}"
ANDROID_NDK_ROOT="${ANDROID_NDK_ROOT:-$HOME/android-ndk}"
NEURON_SDK_ROOT="${NEURON_SDK_ROOT:-}"

echo "=========================================="
echo "MiniCPM-V 4.6 全流�?(�?WSL 模式)"
echo "=========================================="
echo ""
echo "工作目录:    $WORK_DIR"
echo "模型目录:    $MODEL_DIR"
echo "虚拟环境:    $VENV_DIR"
echo "Android NDK: $ANDROID_NDK_ROOT"
echo "NeuroPilot:  $NEURON_SDK_ROOT"
echo "WSL 环境:    $(uname -a)"
echo ""

# 检�?WSL
if ! uname -a | grep -qi "microsoft\|WSL"; then
    echo "�?WARNING: 不在 WSL 环境，可能不是预期的 Ubuntu"
    read -p "继续? (y/N): " CONFIRM
    [ "$CONFIRM" = "y" ] || exit 1
fi

# 询问起始步骤
echo ""
echo "请选择执行模式:"
echo "  1) 从头开�?(Step 1-12)"
echo "  2) 从特定步骤开�?
echo "  3) 只运行指定步�?
echo ""
read -p "选择 (1/2/3): " MODE

case "$MODE" in
    1) START=1; END=12 ;;
    2) read -p "�?Step 几开�? (1-12): " START; END=12 ;;
    3) read -p "起始步骤: " START; read -p "结束步骤: " END ;;
    *) echo "无效选择"; exit 1 ;;
esac

echo ""
echo "将执�?Step $START �?Step $END"
read -p "确认? (y/N): " CONFIRM
[ "$CONFIRM" = "y" ] || exit 0

# 执行步骤
STeps=(
    "step1_install_deps.sh"
    "step2_download_model.sh"
    "step3_prepare_image.sh"
    "step4_install_ndk.sh"
    "step5_build_mnn_converter.sh"
    "step6_export_npu_vision.sh"
    "step7_export_npu_llm.sh"
    "step8_export_mnn.sh"
    "step9_build_runner.sh"
    "step10_prepare_assets.sh"
    "step11_build_apk.sh"
    "step12_deploy.sh"
)

for i in $(seq $START $END); do
    idx=$((i - 1))
    script="${STeps[$idx]}"
    echo ""
    echo "=========================================="
    echo ">>> Step $i: $script"
    echo "=========================================="
    
    if [ ! -f "$SCRIPT_DIR/$script" ]; then
        echo "�?脚本不存�? $script"
        exit 1
    fi
    
    bash "$SCRIPT_DIR/$script"
    result=$?
    
    if [ $result -ne 0 ]; then
        echo ""
        echo "�?Step $i 失败 (exit code: $result)"
        echo ""
        echo "调试提示:"
        echo "  - 查看日志: cat $WORK_DIR/*/step$i.log 2>/dev/null"
        echo "  - 重新运行: bash $SCRIPT_DIR/$script"
        echo "  - 跳过此步: 编辑此脚本调整起始步�?
        exit $result
    fi
    
    echo "�?Step $i 完成"
done

echo ""
echo "=========================================="
echo "�?全流程完�?"
echo "=========================================="