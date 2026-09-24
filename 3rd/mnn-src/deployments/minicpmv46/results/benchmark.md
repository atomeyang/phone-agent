# MiniCPM-V 4.6 Android Deployment Results

Test device: Infinix X6877, MT6899, Android 16. All Android commands used the
Windows SDK `adb.exe`. The test used `test.jpg`, the official one-tile
preprocessing path, 86 prompt tokens, 63 visual tokens, and greedy decoding.
The hybrid and MNN runs generated 64 tokens; the full-NPU run generated 16.

| Metric | NPU vision + MNN Q4 LLM | MNN Q8 vision + MNN Q4 LLM |
|---|---:|---:|
| Vision | 380.365 ms | 1384.915 ms |
| LLM prefill | 447.900 ms | 437.826 ms |
| Real TTFT | 835.182 ms | 1825.296 ms |
| Decode | 16.228 ms/token | 15.359 ms/token |
| Decode speed | 61.62 token/s | 65.11 token/s |
| 64-token inference wall time | 1891.899 ms | 2826.814 ms |
| Visual cosine vs PyTorch FP32 | 0.999973 | 0.982223 |
| Visual MAE vs PyTorch FP32 | 0.000822 | 0.023599 |
| Visual maximum error | 0.025592 | 1.039744 |

The NPU path is the recommended deployment. It reduces TTFT by 990 ms (54.2%)
and has substantially better visual numerical fidelity. Both paths generated
all 64 requested tokens continuously and repeated with identical token IDs on a
second run.

## Full-NPU comparison

| Route | TTFT | Decode | 16-token total |
|---|---:|---:|---:|
| NPU vision + MNN Q4 LLM | 0.835 s | 61.62 token/s | about 1.079 s |
| MNN Q8 vision + MNN Q4 LLM | 1.825 s | 65.11 token/s | about 2.056 s |
| Full NPU | 1.200 s | 2.58 token/s | 7.033 s |

The full-NPU first two token IDs (`760, 2099`) match the FP32 reference, and
its 16-token output is:

> The image shows a person wearing a black t-shirt and glasses. The background is

All 49 text graphs compiled with `--disallow-bridge`: 24 prefill layers, 24
fixed-capacity decode layers, and one final norm plus five-way LM-head graph.
The slow decode is caused by 25 NPU invocations per generated token and exposed
recurrent state at every graph boundary, not by a correctness failure.

## Generated text

NPU vision + MNN Q4 LLM:

> The image shows a person with a neutral, neutral-appearing expression. The
> individual is wearing a simple, casual outfit, with no visible clothing
> accessories or additional accessories. The background is plain and
> unobtrusive, ensuring that the focus remains on the individual. The overall
> impression is one of simplicity and neutrality.

MNN Q8 vision + MNN Q4 LLM:

> The image shows a person with a neutral, neutral-appearing expression, likely
> a professional or professional portrait. The individual is wearing a simple,
> plain-colored outfit, which suggests a formal or professional context. The
> background is plain and uncluttered, ensuring the focus remains on the
> individual. The overall impression is one

Original BF16 PyTorch reference (64-token cutoff):

> The image shows a person standing against a white background. They have
> short, dark, slightly wavy hair and are wearing black-rimmed glasses. They are
> dressed in a plain black t-shirt with a round neck and short sleeves. The
> person's posture is straight, and they are looking directly at the camera
> with a

The deployed Q4 LLM remains semantically valid but loses specific attributes
present in the BF16 reference. This is a language-model quantization quality
limit, not an NPU vision failure: NPU visual cosine is 0.999973.

## Generated artifacts

- NPU vision: `neuropilot/build/vision/`
- Full-NPU text graphs: `neuropilot/build/all-npu/`
- MNN Q4 LLM and Q8 vision: `build/mnn/`
- Android runners: `build/android/`
- Raw benchmark logs and BF16 reference: `results/raw/`

The failed fused MNN vision artifact was deleted. The retained MNN vision graph
uses the working no-fusion conversion because the fused graph failed shape
inference on the first attention layer.

## Checksums

```text
76f632700f096f5186c071d7caf1e456c5882d006df72c9b642f9e8db9dcc4a5  visual_fp16_mt6899.dla
179e76131faa4bf370fe6d6489cef06f281c80c42acd53eb4ca156896a076f57  minicpmv46-runner
a6e535b229c728a26d87a621111e8be7af639718f96190877ccb861dd3df361d  llm.mnn
0a49c4d55496675b1e888cc15282b7447de168572ffa994b6fec4e5431374ae8  llm.mnn.weight
8e15fa125eadd03ebf821c91a2d3894222d3e01853759453a67da7ddc48ce16e  visual.mnn
0fbc6cf935e1094cb9ca95d6e6ba328c6ab78406534e745eb064093a5948c9d4  visual.mnn.weight
```
