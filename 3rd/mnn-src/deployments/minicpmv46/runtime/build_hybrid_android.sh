#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT/../.." && pwd)"
MNN_ROOT="${MNN_ROOT:-$REPO_ROOT}"
MNN_BUILD="${MNN_ANDROID_BUILD:-$ROOT/build/mnn-android}"
NEURON_SDK_ROOT="${NEURON_SDK_ROOT:?Set NEURON_SDK_ROOT to the SDK neuron_sdk directory}"
NDK_ROOT="${ANDROID_NDK_ROOT:?Set ANDROID_NDK_ROOT to an installed Android NDK}"
MTK_PLATFORM="${MTK_PLATFORM:-mt6899}"
NEURON_RUNTIME_SONAME="${NEURON_RUNTIME_SONAME:-libneuron_runtime.so.9.3.1}"
CXX="$NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android29-clang++"
OUTPUT="$ROOT/build/android"

mkdir -p "$OUTPUT"
"$CXX" \
    -std=c++17 \
    -O3 \
    -DNDEBUG \
    -static-libstdc++ \
    -I"$MNN_ROOT/include" \
    -I"$MNN_ROOT/express" \
    -I"$MNN_ROOT/transformers/llm/engine/include" \
    -I"$NEURON_SDK_ROOT/linux-x86_64/include" \
    "$ROOT/runtime/hybrid_runner.cpp" \
    -L"$MNN_BUILD" \
    -L"$NEURON_SDK_ROOT/$MTK_PLATFORM/lib" \
    -Wl,--allow-shlib-undefined \
    -Wl,-rpath,'$ORIGIN' \
    -lMNN \
    -l:"$NEURON_RUNTIME_SONAME" \
    -llog \
    -landroid \
    -o "$OUTPUT/minicpmv46-hybrid-runner"

printf 'Hybrid Android runner: %s/minicpmv46-hybrid-runner\n' "$OUTPUT"
