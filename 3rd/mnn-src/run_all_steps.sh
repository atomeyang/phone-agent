#!/bin/bash
set -euo pipefail

# ============================================================
# MiniCPM-V 4.6 全流程复�?- 总控脚本
# ============================================================
#
# 用法: bash run_all_steps.sh [STEP]
#   不带参数: 依次执行 Step 1-12
#   bash run_all_steps.sh 6  �?Step 6 开�?
#   bash run_all_steps.sh 6 9  只执�?Step 6-9
#
# ============================================================

set +e  # 让每�?step 独立运行
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -eq 0 ]; then
    # 执行所有步�?
    START=1
    END=12
else
    START=$1
    END=${2:-$1}
fi

echo "=========================================="
echo "MiniCPM-V 4.6 全流程复�?
echo "=========================================="
echo "将执�?Step $START �?Step $END"
echo ""

for i in $(seq $START $END); do
    idx=$((i - 1))
    script="${STeps[$idx]}"
    echo ""
    echo "=========================================="
    echo ">>> 执行 Step $i: $script"
    echo "=========================================="
    
    bash "$SCRIPT_DIR/$script"
    result=$?
    
    if [ $result -ne 0 ]; then
        echo ""
        echo "=========================================="
        echo "�?Step $i 失败 (exit code: $result)"
        echo "=========================================="
        echo ""
        echo "下一�? 修复问题后，重新运行:"
        echo "  bash $script"
        echo ""
        echo "或从这里继续:"
        for j in $(seq $i 12); do
            echo "  Step $j: bash ${STeps[$((j-1))]}"
        done
        exit $result
    fi
    
    echo ""
    echo "�?Step $i 完成"
done

echo ""
echo "=========================================="
echo "✓✓�?全流程完�? ✓✓�?
echo "=========================================="
echo ""
echo "总结:"
echo "  模型导出: \$WORK_DIR/npu_vision/ + npu_llm/"
echo "  MNN 模型: \$WORK_DIR/mnn_export/"
echo "  Runtime:  \$WORK_DIR/runtime-assets/"
echo "  APK:      \$WORK_DIR/PhoneVLM-Letterbox-v1.2.1.apk"
echo ""
echo "目录变量:"
echo "  WORK_DIR=\$HOME/minicpmv46-workdir"
echo "  MODEL_DIR=\$HOME/models/minicpm-v-4.6"
echo "  VENV_DIR=\$HOME/minicpmv46-env"
