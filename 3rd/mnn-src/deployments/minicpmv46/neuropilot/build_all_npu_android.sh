#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NEURON_SDK_ROOT="${NEURON_SDK_ROOT:?Set NEURON_SDK_ROOT to the SDK neuron_sdk directory}"
NDK_ROOT="${ANDROID_NDK_ROOT:?Set ANDROID_NDK_ROOT to an installed Android NDK}"
MTK_PLATFORM="${MTK_PLATFORM:-mt6899}"
NEURON_RUNTIME_SONAME="${NEURON_RUNTIME_SONAME:-libneuron_runtime.so.9.3.1}"
CXX="$NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++"

mkdir -p "$ROOT/build/android"
"$CXX" \
    -std=c++17 \
    -O3 \
    -DNDEBUG \
    -static-libstdc++ \
    -I"$NEURON_SDK_ROOT/linux-x86_64/include" \
    "$ROOT/../runtime/all_npu_runner.cpp" \
    -L"$NEURON_SDK_ROOT/$MTK_PLATFORM/lib" \
    -Wl,--allow-shlib-undefined \
    -Wl,-rpath,'$ORIGIN' \
    -l:"$NEURON_RUNTIME_SONAME" \
    -o "$ROOT/build/android/minicpmv46-all-npu-runner"

printf '%s\n' "$ROOT/build/android/minicpmv46-all-npu-runner"
