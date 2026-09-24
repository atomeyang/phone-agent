# Qwen3-VL Android MNN fixes

## Scope

This branch is a source-only MNN 3.6.0 snapshot containing the framework
fixes used by the Qwen3-VL Android CPU deployment. It does not contain the
Android application, model files, APKs, or prebuilt MNN libraries.

The tested runtime configuration was:

- Android arm64-v8a
- Q4 LLM weights
- BF16 vision model and BF16 CPU attention path
- Dynamic image resolution and visual token count unchanged
- `MNN_BUILD_LLM=ON`
- `MNN_SUPPORT_TRANSFORMER_FUSE=ON`
- `MNN_ARM82=ON`
- `MNN_LOW_MEMORY=ON`
- `MNN_OPENCL=ON`

## Fixes

### Qwen3-VL decode MRoPE

File: `transformers/llm/engine/src/omni.cpp`

Generate all three MRoPE position-id planes for every decode token. Decode
positions advance from the final multimodal prompt position and the generated
sequence length instead of reusing an incomplete position layout.

### BF16 CPU attention

Files:

- `source/backend/cpu/CPUAttention.cpp`
- `source/backend/cpu/CPUKVCacheManager.cpp`
- `source/backend/cpu/CPUKVCacheManager.hpp`

Keep attention masks, QK values, and KV-cache packing consistent with the
active CPU core element width. Buffer allocation, address calculation, cache
copying, and typed loads/stores use `core->bytes` instead of assuming FP32.
This prevents the BF16 path from interpreting two-byte data as four-byte
floats and producing NaN attention values.

The value-cache path also handles packed NC4HW4 input explicitly, including
single-token decode, so prefill and decode use the same logical values.

## Observed failure before the fix

- Vision output and the initial Q/K/V tensors were finite.
- Attention later became all-NaN.
- The language model emitted a valid first token and then repeated a token.

## Verification

The fixes were built as a Release arm64-v8a shared MNN library and exercised
through complete on-device Qwen3-VL inference.

- No NaN attention values were observed after the fix.
- Generated output remained semantically coherent through decode.
- Dynamic visual resolution and token count were not reduced.
- App wall time was 13.90 s on the original test phone.
- App wall time was 6.45 s on an MTK 8400 platform (MT6899).

The verified library SHA-256 was
`a7a0b6d7c2741b754e68bad8a143cfd97ae15d0e60ef1a1fd858577faa6a3460`.
The library itself is intentionally not committed to this source branch.
