# MiniCPM-V 4.6 deployment on MT6899

This directory records the source, conversion workflow, and device results for
three MiniCPM-V 4.6 deployment routes:

1. NeuroPilot FP16 vision plus MNN Q4 language model (recommended).
2. MNN Q8 vision plus MNN Q4 language model (portable CPU baseline).
3. NeuroPilot for vision, all 24 prefill/decode layers, final norm, and LM head.

No model weights, generated graphs, proprietary SDK files, runtime libraries,
or device packages are stored in Git.

## Result summary

Device: MTK 8400 platform, MT6899, Android 16.

| Route | TTFT | Decode | 16-token total |
|---|---:|---:|---:|
| NeuroPilot vision + MNN Q4 LLM | 0.835 s | 61.62 token/s | about 1.079 s |
| Full MNN | 1.825 s | 65.11 token/s | about 2.056 s |
| Full NeuroPilot | 1.200 s | 2.58 token/s | 7.033 s |

The full NeuroPilot route is numerically correct, but it invokes 24 layer DLAs
plus the LM-head DLA for every generated token. That graph boundary overhead
makes decode substantially slower than the MNN CPU path. The recommended route
keeps the accurate, faster vision graph on MDLA and uses MNN Q4 for text.

See `results/benchmark.md` and `results/raw/` for complete metrics and output.

## External prerequisites

- MiniCPM-V 4.6 model directory with the original `model.safetensors`.
- A Python environment containing the packages in `requirements.txt` and a
  Transformers build that includes `minicpmv4_6` and `qwen3_5`.
- Android NDK with an `aarch64-linux-android29-clang++` toolchain.
- NeuroPilot Premium 9.0.9 for the NPU routes. Point `NEURON_SDK_ROOT` at its
  `neuron_sdk` directory.
- A host-built `MNNConvert` for MNN export.

The Android demo supports three exact-shape FP16 visual profiles on MT6899:
`560x392`, `392x504`, and `504x392`. The adaptive route uses the official
dynamic multi-tile preprocessing whenever all required profiles are available.
Unsupported combinations fall back to proportional resize plus centered
padding on a `504x392` canvas. The letterbox route always uses that fixed
single-tile path. Neither route crops or distorts the source image.

## MNN route

This branch adds MiniCPM-V 4.6 text-backbone mapping to MNN's exporter and a
token callback used for exact wall-clock TTFT measurement.

Build the host converter, then export the Q4 LLM and Q8 visual graph:

```bash
export MODEL=/path/to/minicpm-v-4.6
export IMAGE=/path/to/test.jpg
export MNNCONVERT=/path/to/MNNConvert
bash deployments/minicpmv46/mnn/export_models.sh
```

Build the Android MNN runtime:

```bash
export ANDROID_NDK_ROOT=/path/to/android-ndk
bash deployments/minicpmv46/mnn/build_android.sh
```

## NeuroPilot route

Export and compile the visual graph:

```bash
export MODEL=/path/to/minicpm-v-4.6
export IMAGE=/path/to/test.jpg
export NEURON_SDK_ROOT=/path/to/neuron_sdk
export EXPORT_PYTHON=/path/to/export/python
export CONVERTER_PYTHON=/path/to/mtk-converter/python
bash deployments/minicpmv46/neuropilot/build_vision.sh
```

The exporter writes both `visual.reference.npz` and raw FP32 input/reference
files. The NPU compiler runs with `--disallow-bridge`, so a successful build has
no CPU/OpenCL fallback inside the DLA.

The full-NPU text route is built with:

```bash
export VISION_REFERENCE=deployments/minicpmv46/neuropilot/build/vision/visual.reference.npz
bash deployments/minicpmv46/neuropilot/build_all_shards.sh
bash deployments/minicpmv46/neuropilot/build_all_npu_android.sh
```

`build_decode_probes.sh` and `build_prefill_probes.sh` are smaller validation
gates for one layer of each attention architecture.

## Android runners

Build the runner shared by the recommended and full-MNN routes:

```bash
export MNN_ANDROID_BUILD=deployments/minicpmv46/build/mnn-android
bash deployments/minicpmv46/runtime/build_hybrid_android.sh
```

Its two modes keep the prompt, tokenizer, MNN Q4 LLM, and greedy decoding
identical:

```text
minicpmv46-hybrid-runner npu CONFIG VISION_DLA INPUT_BIN REFERENCE_BIN 64
minicpmv46-hybrid-runner mnn CONFIG VISION_MNN INPUT_BIN REFERENCE_BIN 64
```

The all-NPU runner uses:

```text
minicpmv46-all-npu-runner MODEL_ROOT VISION_DLA INPUT_BIN TOKENS_OUTPUT 16
```

## Android demo

The Android project has two product flavors that share the resident hybrid
runner and streaming UI:

| Flavor | Version | Image path |
|---|---|---|
| `dynamic` | PhoneVLM Adaptive 1.3.1 | Official multi-tile with letterbox fallback |
| `letterbox` | PhoneVLM Letterbox 1.2.1 | Fixed 504x392 proportional fit and padding |

Build either flavor after supplying its external runtime asset directory:

```bash
cd deployments/minicpmv46/android
gradle :app:assembleDynamicDebug \
  -PdynamicRuntimeAssetsDir=/path/to/adaptive/runtime-assets
gradle :app:assembleLetterboxDebug \
  -PletterboxRuntimeAssetsDir=/path/to/letterbox/runtime-assets
```

Runtime assets contain generated model files and proprietary SDK libraries, so
they are intentionally excluded from Git. See `android/README.txt` and
`android/README-letterbox.txt` for device installation and route behavior.

On this host, every device operation must use the Windows SDK executable:

```text
/mnt/d/Users/admin/AppData/Local/Android/Sdk/platform-tools/adb.exe
```
