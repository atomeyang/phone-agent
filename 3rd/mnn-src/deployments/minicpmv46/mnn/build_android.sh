#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"
MNN_ROOT="${MNN_ROOT:-$REPO_ROOT}"
OUTPUT="${MNN_ANDROID_BUILD:-$ROOT/build/mnn-android}"
NDK_ROOT="${ANDROID_NDK_ROOT:?Set ANDROID_NDK_ROOT to an installed Android NDK}"
JOBS="${JOBS:-4}"

cmake -S "$MNN_ROOT" -B "$OUTPUT" \
    -DCMAKE_TOOLCHAIN_FILE="$NDK_ROOT/build/cmake/android.toolchain.cmake" \
    -DCMAKE_BUILD_TYPE=Release \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-29 \
    -DANDROID_STL=c++_static \
    -DMNN_ARM82=ON \
    -DMNN_BUILD_FOR_ANDROID_COMMAND=ON \
    -DMNN_BUILD_LLM=ON \
    -DMNN_BUILD_SHARED_LIBS=ON \
    -DMNN_BUILD_TEST=OFF \
    -DMNN_BUILD_TOOLS=OFF \
    -DMNN_KLEIDIAI=ON \
    -DMNN_LOW_MEMORY=ON \
    -DMNN_SEP_BUILD=OFF \
    -DMNN_SME2=ON \
    -DMNN_SUPPORT_TRANSFORMER_FUSE=ON \
    -DMNN_USE_LOGCAT=OFF

cmake --build "$OUTPUT" --target MNN --parallel "$JOBS"
printf 'MNN Android runtime: %s/libMNN.so\n' "$OUTPUT"
