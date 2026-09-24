#!/usr/bin/env bash
# Gemma 4 E2B hybrid deployment for MediaTek Dimensity 8400 (MT6899).
#
#   vision : Gemma 4 vision tower -> NeuroPilot W8A16 DLA on the MDLA NPU
#   text   : Gemma 4 text backbone -> MNN Q4 on the CPU
#
# Usage:  source "$(dirname "$0")/env.sh"

export MTKG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- inputs ------------------------------------------------------------------
export MODEL="${MODEL:-/mnt/e/codes/Engineering/mtkgemma/gemma-4-E2B-it}"
export IMAGE_ROOT="$MTKG/helpers/inputs/images"
# Full calibration pool (21 categories). Falls back to the copied subset.
export CALIB_ROOT="${CALIB_ROOT:-/mnt/d/datas/simpletestsets}"
[[ -d "$CALIB_ROOT" ]] || export CALIB_ROOT="$IMAGE_ROOT"
export CALIB_LIMIT="${CALIB_LIMIT:-16}"
export CALIB_TAIL_LIMIT="${CALIB_TAIL_LIMIT:-8}"
export IMAGE="${IMAGE:-$IMAGE_ROOT/dog/2147.jpg}"

# --- vision canvas geometry --------------------------------------------------
# The vision graph is fully static, so the canvas is a build-time decision.
# GEMMA4_GRID_W x GEMMA4_GRID_H are patch counts (each patch is 16 px), i.e.
# 54x45 -> 864x720 px / 270 soft tokens (the original profile) and
# 30x39 -> 480x624 px / 130 soft tokens (the fast profile, see helpers/profiles/).
# Both sides must be a multiple of 3 (the 3x3 pooling) and of 3 at the patch
# level; anything else makes the pooling matrix image dependent.
export GEMMA4_GRID_W="${GEMMA4_GRID_W:-54}"
export GEMMA4_GRID_H="${GEMMA4_GRID_H:-45}"
export GEMMA4_PATCH="${GEMMA4_PATCH:-16}"
export GEMMA4_CANVAS_W="$((GEMMA4_GRID_W * GEMMA4_PATCH))"
export GEMMA4_CANVAS_H="$((GEMMA4_GRID_H * GEMMA4_PATCH))"
export GEMMA4_SOFT_TOKENS="$(( (GEMMA4_GRID_W / 3) * (GEMMA4_GRID_H / 3) ))"
# Geometry profile name; also the suffix of the per-profile output directory so
# a rebuild at a new canvas never overwrites the previous delivery.
export GEMMA4_PROFILE="${GEMMA4_PROFILE:-${GEMMA4_GRID_W}x${GEMMA4_GRID_H}}"
if (( GEMMA4_GRID_W % 3 != 0 || GEMMA4_GRID_H % 3 != 0 )); then
    echo "env.sh: GEMMA4_GRID_W/GEMMA4_GRID_H must be multiples of 3 (got ${GEMMA4_GRID_W}x${GEMMA4_GRID_H})" >&2
    return 1 2>/dev/null || exit 1
fi

# --- MediaTek NeuroPilot Premium SDK (MT6899 target) -------------------------
export SDK_ROOT="${SDK_ROOT:-/mnt/d/codes/Engineering/mtk/neuropilot-sdk-premium-9.0.9-build20260629}"
export NEURON_SDK_ROOT="$SDK_ROOT/neuron_sdk"
export NCC_ROOT="$NEURON_SDK_ROOT/linux-x86_64"
export NCC_TFLITE="$NCC_ROOT/bin/ncc-tflite"
export MTK_PLATFORM="${MTK_PLATFORM:-mt6899}"

# --- python environments (kept apart on purpose) -----------------------------
# export    : torch / transformers / onnx 1.22 -> safetensors to ONNX
# converter : mtk_converter 9.12 -> ONNX to quantised TFLite
# mtk_converter pins onnx<1.14, protobuf<4 and numpy<2, so it cannot share the
# export environment.
export EXPORT_PYTHON="${EXPORT_PYTHON:-$HOME/miniforge3/envs/d7400export/bin/python}"
export CONVERTER_PYTHON="${CONVERTER_PYTHON:-$HOME/miniforge3/envs/mtkcvt/bin/python}"

# --- MNN (language model side, CPU) ------------------------------------------
export MNN_SRC="$MTKG/3rd/mnn-src"
export MNNCONVERT="$MTKG/3rd/mnn-host/MNNConvert"
export MNN_ANDROID_LIB="$MTKG/3rd/mnn-android/libMNN.so"

# --- android toolchain -------------------------------------------------------
export ANDROID_NDK_ROOT="${ANDROID_NDK_ROOT:-/mnt/d/tools/android/ndk/ubuntu/android-ndk-r25c}"
export ANDROID_SDK_ROOT="${ANDROID_SDK_ROOT:-/mnt/d/tools/android/sdk/ubuntu}"
export JAVA_HOME="${JAVA_HOME:-/usr/lib/jvm/jdk-17}"
export GRADLE="${GRADLE:-/mnt/d/gradle-cache/bpt9gzteqjrbo1mjrsomdt32c/gradle-8.11.1/bin/gradle}"
export GRADLE_USER_HOME="${GRADLE_USER_HOME:-/mnt/d/gradle-cache-user-home}"
export ADB="${ADB:-/usr/local/bin/adb}"

# --- outputs -----------------------------------------------------------------
# OUT_DIR wins if given; otherwise the default 54x45 profile keeps the original
# `out/` tree and every other geometry gets its own `out-<profile>/` tree.
if [[ -n "${OUT_DIR:-}" ]]; then
    export OUT="$OUT_DIR"
elif [[ "$GEMMA4_PROFILE" == "54x45" ]]; then
    export OUT="$MTKG/out"
else
    export OUT="$MTKG/out-$GEMMA4_PROFILE"
fi
export LOGS="$MTKG/logs"
export OUT_VISION="$OUT/vision"
export OUT_VISION_FRONT="$OUT/vision/front"
export OUT_VISION_TAIL="$OUT/vision/tail"
# The text backbone does not depend on the vision canvas, so its 2.6 GB of
# assets are shared by every geometry profile instead of being duplicated.
export OUT_MNN="${MNN_OUT_DIR:-$MTKG/out/mnn}"
export OUT_RUNNER="$OUT/runner"
export OUT_ASSETS="$OUT/runtime-assets"
export OUT_APK="$OUT/apk"

# --- target device -----------------------------------------------------------
export DEVICE_SOC="MT6899"
export DLA_FRONT="visual_front_w8a16_${MTK_PLATFORM}.dla"
export DLA_TAIL="visual_tail_w8a16_${MTK_PLATFORM}.dla"

# --- helpers -----------------------------------------------------------------
# Only ncc-tflite gets the SDK host libraries: exporting that directory globally
# shadows the system libstdc++ and breaks torch and MNNConvert.
ncc_run() { LD_LIBRARY_PATH="$NCC_ROOT/lib" "$@"; }

MNN_HOST_LIBDIRS="$MTKG/3rd/mnn-host:$MNN_SRC/build/tools/converter:$MNN_SRC/build/express:$MNN_SRC/build"
mnn_run() { LD_LIBRARY_PATH="$MNN_HOST_LIBDIRS" "$@"; }

mkdir -p "$LOGS" "$OUT_VISION" "$OUT_VISION_FRONT" "$OUT_VISION_TAIL" "$OUT_MNN" \
         "$OUT_RUNNER" "$OUT_ASSETS" "$OUT_APK"
