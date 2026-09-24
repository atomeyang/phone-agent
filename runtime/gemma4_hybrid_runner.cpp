// Gemma 4 E2B hybrid runner: NPU vision (NeuroPilot W8A16 DLA) + MNN Q4 text,
// plus the resident on-device agent (memory, planning, tools, history).
//
//   npu    CONFIG VISION_DLA VISION_INPUT REFERENCE [MAX_TOKENS]
//   --service CONFIG VISION_DIR VISION_INPUT PROMPT REQUEST STOP LOG READY [MAX_TOKENS]
//   --agent-service CONFIG VISION_DLA ROOT AGENT_DIR [MAX_TOKENS]
//
// VISION_INPUT is the letterboxed canvas written by the Android demo:
//   int32 magic, int32 version, int32 valid_patch_rows, int32 valid_patch_cols,
//   int32 tile_count, then per tile: int32 height, int32 width and
//   height*width*3 float32 in [0, 1] (R plane, G plane, B plane).
//
// The vision graph is static: 2430 patch vectors in, 270 pooled tokens out.  The
// runner patchifies the canvas, masks the padding patches, runs the DLA, drops
// the pooled bins that only cover padding and splices the surviving tokens into
// the MNN prompt as `<|image|>` embeddings.

#include "llm/llm.hpp"

#include "agent/agent_json.h"
#include "agent/agent_llm.h"
#include "agent/agent_runtime.h"
#include "agent/agent_service.h"
#include "agent/agent_util.h"

#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>
#include "neuron/api/RuntimeV2.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <sched.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using MNN::Express::VARP;
using MNN::Transformer::Llm;

// --- vision graph geometry (must match helpers/src/gemma4_vision.py) ----------------
constexpr int kPatch = 16;
constexpr int kPool = 3;
#ifndef GEMMA4_GRID_COLUMNS
#define GEMMA4_GRID_COLUMNS 54
#endif
#ifndef GEMMA4_GRID_ROWS
#define GEMMA4_GRID_ROWS 45
#endif
constexpr int kGridWidth = GEMMA4_GRID_COLUMNS;   // patch columns in the canvas
constexpr int kGridHeight = GEMMA4_GRID_ROWS;     // patch rows in the canvas
constexpr int kCanvasWidth = kGridWidth * kPatch;    // 864
constexpr int kCanvasHeight = kGridHeight * kPatch;  // 720
constexpr int kPatches = kGridWidth * kGridHeight;   // 2430
constexpr int kPatchDim = 3 * kPatch * kPatch;       // 768
constexpr int kHidden = kPatchDim;                    // vision hidden size
constexpr int kBinWidth = kGridWidth / kPool;        // 18
constexpr int kBinHeight = kGridHeight / kPool;      // 15
constexpr int kSoftTokens = kBinWidth * kBinHeight;  // 270
constexpr int kTextHidden = 1536;

constexpr float kMaskNegative = -30000.0F;
// The tail shard starts with a scale-invariant RMSNorm, so the pooled states
// are pre-scaled into the range the tail's int16 activation quantiser expects;
// without it MDLA returns an all-zero output for the real (large) values.
constexpr float kTailInputScale = 1.0F / 256.0F;

constexpr int32_t kTileFileMagic = 0x4d435034;
constexpr int32_t kTileFileVersion = 1;

// The MNN graph applies Gemma 4's `embed_scale` (sqrt(text hidden size)) to the
// embeddings it receives, and MNN's own Gemma 4 vision path compensates with
// `* (1/39.25f)` (transformers/llm/engine/src/omni.cpp).  Injected soft tokens
// must therefore be divided by the same factor, otherwise the image is 39x too
// loud and the model hallucinates.
constexpr float kEmbedScale = 39.19F;

// Gemma 4 special tokens.
constexpr int kImageTokenId = 258880;
constexpr const char* kImageStart = "<|image>";
constexpr const char* kImageEnd = "<image|>";
constexpr const char* kImageToken = "<|image|>";
constexpr const char* kDefaultUserPrompt = "Describe this image.";

double milliseconds(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

void check_neuron(int status, const std::string& operation) {
    if (status != 0) {
        throw std::runtime_error(operation + " failed with status " + std::to_string(status));
    }
}

float half_to_float(uint16_t bits) {
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000) << 16;
    uint32_t exponent = (bits >> 10) & 0x1f;
    uint32_t mantissa = bits & 0x3ff;
    uint32_t result = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            result = sign;
        } else {
            exponent = 127 - 15 + 1;
            while ((mantissa & 0x400) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 0x3ff;
            result = sign | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1f) {
        result = sign | 0x7f800000u | (mantissa << 13);
    } else {
        result = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
    }
    float value;
    std::memcpy(&value, &result, sizeof(value));
    return value;
}

// Keep the text backbone on the four performance cores, like the MT6899
// MiniCPM-V delivery does.
void pin_to_performance_cores() {
    cpu_set_t cores;
    CPU_ZERO(&cores);
    for (int cpu = 4; cpu <= 7; ++cpu) {
        CPU_SET(cpu, &cores);
    }
    sched_setaffinity(0, sizeof(cores), &cores);
}

std::vector<float> read_floats(const std::string& path, size_t expected) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open " + path);
    }
    std::vector<float> values(expected);
    stream.read(reinterpret_cast<char*>(values.data()),
                static_cast<std::streamsize>(expected * sizeof(float)));
    if (!stream) {
        throw std::runtime_error("truncated float file " + path);
    }
    return values;
}

struct Accuracy {
    double cosine = 0.0;
    double mae = 0.0;
    double max_abs = 0.0;
};

Accuracy compare(const std::vector<float>& actual, const std::vector<float>& reference) {
    if (actual.size() != reference.size()) {
        throw std::runtime_error("accuracy comparison size mismatch");
    }
    double dot = 0.0;
    double left = 0.0;
    double right = 0.0;
    double abs_sum = 0.0;
    double max_abs = 0.0;
    for (size_t index = 0; index < actual.size(); ++index) {
        const double a = actual[index];
        const double b = reference[index];
        dot += a * b;
        left += a * a;
        right += b * b;
        abs_sum += std::fabs(a - b);
        max_abs = std::max(max_abs, std::fabs(a - b));
    }
    const double denominator = std::sqrt(left) * std::sqrt(right);
    return Accuracy{
        denominator > 0.0 ? dot / denominator : 0.0,
        abs_sum / static_cast<double>(actual.size()),
        max_abs,
    };
}

// --- vision input -----------------------------------------------------------

struct CanvasInput {
    std::vector<float> pixels;  // kCanvasHeight x kCanvasWidth x 3, planar RGB
    int patch_rows = 0;         // valid patch rows
    int patch_columns = 0;      // valid patch columns

    int soft_tokens() const {
        return (patch_rows / kPool) * (patch_columns / kPool);
    }
};

CanvasInput read_canvas_input(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open " + path);
    }
    auto read_int = [&]() {
        int32_t value = 0;
        stream.read(reinterpret_cast<char*>(&value), sizeof(value));
        if (!stream) {
            throw std::runtime_error("truncated canvas header");
        }
        return value;
    };

    const int32_t magic = read_int();
    const int32_t version = read_int();
    CanvasInput canvas;
    canvas.patch_rows = read_int();
    canvas.patch_columns = read_int();
    const int32_t tile_count = read_int();
    if (magic != kTileFileMagic || version != kTileFileVersion || tile_count != 1) {
        throw std::runtime_error("unsupported canvas file");
    }
    if (canvas.patch_rows <= 0 || canvas.patch_columns <= 0 ||
        canvas.patch_rows % kPool != 0 || canvas.patch_columns % kPool != 0 ||
        canvas.patch_rows > kGridHeight || canvas.patch_columns > kGridWidth) {
        throw std::runtime_error("invalid canvas patch grid");
    }

    const int32_t height = read_int();
    const int32_t width = read_int();
    if (height != kCanvasHeight || width != kCanvasWidth) {
        throw std::runtime_error("canvas must be " + std::to_string(kCanvasWidth) + "x" +
                                 std::to_string(kCanvasHeight));
    }
    canvas.pixels.resize(static_cast<size_t>(3) * height * width);
    stream.read(reinterpret_cast<char*>(canvas.pixels.data()),
                static_cast<std::streamsize>(canvas.pixels.size() * sizeof(float)));
    if (!stream) {
        throw std::runtime_error("truncated canvas data");
    }
    return canvas;
}

struct VisionInput {
    // RGBA canvas: the RGB planes carry the pixels and the alpha plane is 1.0
    // on every padding patch.  The DLA convolves the coverage plane into the
    // additive attention mask, so the runner only has to fill the planes.
    std::vector<float> planes;  // [4, kCanvasHeight, kCanvasWidth]
};

VisionInput build_vision_input(const CanvasInput& canvas) {
    VisionInput input;
    const size_t plane = static_cast<size_t>(kCanvasWidth) * kCanvasHeight;
    input.planes.assign(4 * plane, 0.0F);
    std::memcpy(input.planes.data(), canvas.pixels.data(), 3 * plane * sizeof(float));
    float* alpha = input.planes.data() + 3 * plane;
    for (int row = 0; row < kCanvasHeight; ++row) {
        const int patch_row = row / kPatch;
        const bool row_padding = patch_row >= canvas.patch_rows;
        float* line = alpha + static_cast<size_t>(row) * kCanvasWidth;
        if (row_padding) {
            std::fill(line, line + kCanvasWidth, 1.0F);
            continue;
        }
        const int boundary = canvas.patch_columns * kPatch;
        std::fill(line + boundary, line + kCanvasWidth, 1.0F);
    }
    return input;
}

// --- NPU vision -------------------------------------------------------------

class NpuVision {
public:
    enum class Role { Front, Tail, Single };

    explicit NpuVision(const std::string& path, Role role = Role::Single,
                       bool check_geometry = true) {
        role_ = role;
        check_geometry_ = check_geometry;
        check_neuron(NeuronRuntimeV2_create(path.c_str(), 1, &runtime_, 8),
                     "create vision DLA");
        NeuronRuntimeV2_setQoS(runtime_, NEURONRUNTIME_SYSTEM_SCENARIO_HINT,
                               NEURONRUNTIME_SYSTEM_SCENARIO_HINT_PERFORMANCE);
        NeuronRuntimeV2_setQoS(runtime_, NEURONRUNTIME_SYSTEM_CPU_BOOST, 100);
        NeuronRuntimeV2_setQoS(runtime_, NEURONRUNTIME_SYSTEM_DDR_BOOST, 100);

        size_t input_count = 0;
        size_t output_count = 0;
        check_neuron(NeuronRuntimeV2_getInputNumber(runtime_, &input_count),
                     "get NPU input count");
        check_neuron(NeuronRuntimeV2_getOutputNumber(runtime_, &output_count),
                     "get NPU output count");
        if (!check_geometry_) {
            input_bytes_.resize(input_count);
            for (size_t index = 0; index < input_count; ++index) {
                check_neuron(NeuronRuntimeV2_getInputPaddedSize(runtime_, index,
                                                               &input_bytes_[index]),
                             "get NPU input size");
            }
            check_neuron(NeuronRuntimeV2_getOutputPaddedSize(runtime_, 0, &output_bytes_),
                         "get NPU output size");
            for (size_t index = 0; index < input_count; ++index) {
                void* buffer = nullptr;
                if (posix_memalign(&buffer, 4096, input_bytes_[index]) != 0) {
                    throw std::bad_alloc();
                }
                inputs_.push_back(buffer);
            }
            if (posix_memalign(&output_, 4096, output_bytes_) != 0) {
                throw std::bad_alloc();
            }
            return;
        }
        if (input_count != 1 || output_count != 1) {
            throw std::runtime_error(
                "vision DLA must have one input (patches with fused mask) and one output");
        }
        for (size_t index = 0; index < input_count; ++index) {
            size_t required = 0;
            check_neuron(NeuronRuntimeV2_getInputPaddedSize(runtime_, index, &required),
                         "get NPU input size");
            input_bytes_.push_back(required);
        }
        check_neuron(NeuronRuntimeV2_getOutputPaddedSize(runtime_, 0, &output_bytes_),
                     "get NPU output size");
        if (std::getenv("GEMMA4_DEBUG") != nullptr) {
            std::cout << "DEBUG npu_input_bytes=" << input_bytes_[0]
                      << " npu_output_bytes=" << output_bytes_
                      << " expected_input="
                      << static_cast<size_t>(4) * kCanvasWidth * kCanvasHeight *
                             sizeof(float)
                      << " expected_output="
                      << static_cast<size_t>(kSoftTokens) * kTextHidden * sizeof(float)
                      << std::endl;
        }
        const size_t expected_input =
            static_cast<size_t>(4) * kCanvasWidth * kCanvasHeight * sizeof(float);
        const size_t expected_input_tail =
            static_cast<size_t>(kSoftTokens) * kPatchDim * sizeof(float);
        const size_t expected_output_front =
            static_cast<size_t>(kPatches) * kPatchDim * sizeof(float);
        const size_t expected_output_tail =
            static_cast<size_t>(kSoftTokens) * kTextHidden * sizeof(float);
        const bool input_ok =
            role_ == Role::Tail ? input_bytes_[0] >= expected_input_tail
                                : input_bytes_[0] >= expected_input;
        const size_t expected_output =
            role_ == Role::Front ? expected_output_front : expected_output_tail;
        if (!input_ok || output_bytes_ < expected_output) {
            throw std::runtime_error("vision DLA buffer shape mismatch");
        }
        for (size_t index = 0; index < input_count; ++index) {
            void* buffer = nullptr;
            if (posix_memalign(&buffer, 4096, input_bytes_[index]) != 0) {
                throw std::bad_alloc();
            }
            inputs_.push_back(buffer);
        }
        if (posix_memalign(&output_, 4096, output_bytes_) != 0) {
            throw std::bad_alloc();
        }
    }

    std::vector<float> run_raw(const std::vector<float>& input) {
        const size_t input_count = input_bytes_[0] / sizeof(float);
        if (input.size() != input_count) {
            throw std::runtime_error(
                "raw input size mismatch: got " + std::to_string(input.size()) +
                " expected " + std::to_string(input_count));
        }
        std::memset(inputs_[0], 0, input_bytes_[0]);
        std::memset(output_, 0, output_bytes_);
        std::memcpy(inputs_[0], input.data(), input.size() * sizeof(float));
        IOBuffer input_buffer(inputs_[0], input_bytes_[0], -1, 0);
        IOBuffer output_buffer(output_, output_bytes_, -1, 0);
        SyncInferenceRequest request{&input_buffer, &output_buffer};
        check_neuron(NeuronRuntimeV2_run(runtime_, request), "run vision DLA (raw)");
        const auto* values = static_cast<const float*>(output_);
        return std::vector<float>(values, values + output_bytes_ / sizeof(float));
    }

    size_t input_elements() const { return input_bytes_[0] / sizeof(float); }
    size_t output_elements() const { return output_bytes_ / sizeof(float); }

    /** Sizes of the first input and first output, in floats. */
    size_t input_floats() const { return input_bytes_[0] / sizeof(float); }
    size_t output_floats() const { return output_bytes_ / sizeof(float); }

    void describe() const {
        for (size_t index = 0; index < inputs_.size(); ++index) {
            uint32_t rank = 0;
            RuntimeAPIDimensions dims{};
            if (NeuronRuntimeV2_getInputRank(runtime_, index, &rank) == 0 &&
                NeuronRuntimeV2_getInputPaddedDimensions(runtime_, index, &dims) == 0) {
                std::cout << "DEBUG input[" << index << "] rank=" << rank << " padded=(";
                for (uint32_t axis = 0; axis < rank && axis < 8; ++axis) {
                    std::cout << (axis ? "," : "") << dims.dimensions[axis];
                }
                std::cout << ") bytes=" << input_bytes_[index] << std::endl;
            }
        }
        uint32_t rank = 0;
        RuntimeAPIDimensions dims{};
        if (NeuronRuntimeV2_getOutputRank(runtime_, 0, &rank) == 0 &&
            NeuronRuntimeV2_getOutputPaddedDimensions(runtime_, 0, &dims) == 0) {
            std::cout << "DEBUG output rank=" << rank << " padded=(";
            for (uint32_t axis = 0; axis < rank && axis < 8; ++axis) {
                std::cout << (axis ? "," : "") << dims.dimensions[axis];
            }
            std::cout << ") bytes=" << output_bytes_ << std::endl;
        }
    }

    ~NpuVision() {
        for (void* buffer : inputs_) {
            std::free(buffer);
        }
        std::free(output_);
        if (runtime_ != nullptr) {
            NeuronRuntimeV2_release(runtime_);
        }
    }

    std::vector<float> run(const VisionInput& input) {
        std::memset(inputs_[0], 0, input_bytes_[0]);
        std::memset(output_, 0, output_bytes_);
        std::memcpy(inputs_[0], input.planes.data(), input.planes.size() * sizeof(float));

        IOBuffer input_buffer(inputs_[0], input_bytes_[0], -1, 0);
        IOBuffer output_buffer(output_, output_bytes_, -1, 0);
        SyncInferenceRequest request{&input_buffer, &output_buffer};
        check_neuron(NeuronRuntimeV2_run(runtime_, request), "run vision DLA");
        const auto* values = static_cast<const float*>(output_);
        if (std::getenv("GEMMA4_DEBUG") != nullptr) {
            // Some runtimes only fill the output from the second run onwards.
            check_neuron(NeuronRuntimeV2_run(runtime_, request), "run vision DLA (retry)");
            double sum = 0.0;
            double max_abs = 0.0;
            const size_t count = static_cast<size_t>(kSoftTokens) * kTextHidden;
            for (size_t index = 0; index < count; ++index) {
                sum += std::fabs(values[index]);
                max_abs = std::max(max_abs, std::fabs(static_cast<double>(values[index])));
            }
            std::cout << "DEBUG npu_output mean_abs=" << (sum / count)
                      << " max_abs=" << max_abs
                      << " first=" << values[0] << "," << values[1] << "," << values[2]
                      << std::endl;
            // The graph asks for a float output, but check the fp16
            // interpretation as well in case the boundary cast was dropped.
            const auto* halves = reinterpret_cast<const uint16_t*>(output_);
            double half_sum = 0.0;
            double half_max = 0.0;
            const size_t half_count = count / 2;
            for (size_t index = 0; index < half_count; ++index) {
                const float value = half_to_float(halves[index]);
                half_sum += std::fabs(value);
                half_max = std::max(half_max, std::fabs(static_cast<double>(value)));
            }
            std::cout << "DEBUG npu_output_as_fp16 mean_abs=" << (half_sum / half_count)
                      << " max_abs=" << half_max << std::endl;
        }
        return std::vector<float>(values, values + output_bytes_ / sizeof(float));
    }

private:
    Role role_ = Role::Single;
    bool check_geometry_ = true;
    void* runtime_ = nullptr;
    std::vector<void*> inputs_;
    std::vector<size_t> input_bytes_;
    void* output_ = nullptr;
    size_t output_bytes_ = 0;
};

// Divides soft tokens by the text embedding scale; see kEmbedScale.
void apply_embed_scale(std::vector<float>& embeddings) {
    for (float& value : embeddings) {
        value /= kEmbedScale;
    }
}

// Selects the pooled bins that lie inside the valid patch rectangle.  Bins are
// laid out row major over the canvas (`bins_per_row = kGridWidth / kPool`), the
// same order the HuggingFace pooler uses for a letterboxed image.
// 3x3 box average over the canvas patch grid, then the sqrt(hidden) scale that
// Gemma4VisionPooler applies.  This runs on the CPU: expressing it as a
// constant-times-activation MatMul inside the DLA made the MDLA program return
// an all-zero output on this firmware.
std::vector<float> pool_patches(const std::vector<float>& hidden) {
    std::vector<float> pooled(static_cast<size_t>(kSoftTokens) * kHidden, 0.0F);
    const float scale = std::sqrt(static_cast<float>(kHidden)) / (kPool * kPool);
    for (int bin_row = 0; bin_row < kBinHeight; ++bin_row) {
        for (int bin_column = 0; bin_column < kBinWidth; ++bin_column) {
            const int bin = bin_row * kBinWidth + bin_column;
            float* destination = pooled.data() + static_cast<size_t>(bin) * kHidden;
            for (int row = 0; row < kPool; ++row) {
                for (int column = 0; column < kPool; ++column) {
                    const int patch =
                        (bin_row * kPool + row) * kGridWidth + bin_column * kPool + column;
                    const float* source =
                        hidden.data() + static_cast<size_t>(patch) * kHidden;
                    for (int index = 0; index < kHidden; ++index) {
                        destination[index] += source[index] * scale * kTailInputScale;
                    }
                }
            }
        }
    }
    return pooled;
}

std::vector<float> select_soft_tokens(
    const std::vector<float>& canvas_tokens, const CanvasInput& canvas) {
    const int tokens = canvas.soft_tokens();
    std::vector<float> selected(static_cast<size_t>(tokens) * kTextHidden);
    const int bin_rows = canvas.patch_rows / kPool;
    const int bin_columns = canvas.patch_columns / kPool;
    int cursor = 0;
    for (int row = 0; row < bin_rows; ++row) {
        for (int column = 0; column < bin_columns; ++column) {
            const int bin = row * kBinWidth + column;
            std::memcpy(selected.data() + static_cast<size_t>(cursor) * kTextHidden,
                        canvas_tokens.data() + static_cast<size_t>(bin) * kTextHidden,
                        kTextHidden * sizeof(float));
            ++cursor;
        }
    }
    return selected;
}

// --- prompt -----------------------------------------------------------------

std::string build_prompt(int image_tokens, const std::string& user_prompt) {
    std::string prompt = "<bos><|turn>user\n";
    prompt += kImageStart;
    for (int index = 0; index < image_tokens; ++index) {
        prompt += kImageToken;
    }
    prompt += kImageEnd;
    prompt += user_prompt;
    prompt += "<turn|>\n<|turn>model\n";
    return prompt;
}

std::string read_text(const std::string& path) {
    if (path.empty()) {
        return kDefaultUserPrompt;
    }
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return kDefaultUserPrompt;
    }
    std::ostringstream contents;
    contents << stream.rdbuf();
    std::string value = contents.str();
    while (!value.empty() && (value.back() == '\n' || value.back() == '\r')) {
        value.pop_back();
    }
    return value.empty() ? kDefaultUserPrompt : value;
}

bool consume_file(const std::string& path) {
    if (path.empty() || access(path.c_str(), F_OK) != 0) {
        return false;
    }
    unlink(path.c_str());
    return true;
}

void write_ready_file(const std::string& path, const std::string& contents) {
    const std::string temporary = path + ".tmp";
    {
        std::ofstream stream(temporary, std::ios::trunc);
        if (!stream) {
            throw std::runtime_error("cannot write " + temporary);
        }
        stream << contents << '\n';
    }
    if (rename(temporary.c_str(), path.c_str()) != 0) {
        throw std::runtime_error("cannot publish " + path);
    }
}

std::string base64_encode(const std::string& value) {
    static constexpr char kAlphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string encoded;
    encoded.reserve((value.size() + 2) / 3 * 4);
    for (size_t offset = 0; offset < value.size(); offset += 3) {
        const uint32_t first = static_cast<unsigned char>(value[offset]);
        const uint32_t second = offset + 1 < value.size()
            ? static_cast<unsigned char>(value[offset + 1]) : 0;
        const uint32_t third = offset + 2 < value.size()
            ? static_cast<unsigned char>(value[offset + 2]) : 0;
        const uint32_t packed = (first << 16) | (second << 8) | third;
        encoded.push_back(kAlphabet[(packed >> 18) & 0x3f]);
        encoded.push_back(kAlphabet[(packed >> 12) & 0x3f]);
        encoded.push_back(offset + 1 < value.size() ? kAlphabet[(packed >> 6) & 0x3f] : '=');
        encoded.push_back(offset + 2 < value.size() ? kAlphabet[packed & 0x3f] : '=');
    }
    return encoded;
}

const char* model_platform() {
    const char* value = std::getenv("GEMMA4_PLATFORM");
    return value != nullptr && *value != '\0' ? value : "mt6899";
}

// Thread count for the MNN text backbone.  The MT6899 delivery used the four
// performance cores; the little cores are slower per clock and drag the
// quantised kernels down, but it is worth measuring on a given device.
int text_threads() {
    const char* value = std::getenv("GEMMA4_THREADS");
    if (value != nullptr && *value != '\0') {
        const int parsed = std::atoi(value);
        if (parsed > 0 && parsed <= 16) {
            return parsed;
        }
    }
    return 4;
}

std::string llm_config_json() {
    // Gemma 4 E2B's context window is close to 128k tokens; the KV cache we
    // allocate is the only limit we impose (see agent-config.json
    // kv_total_tokens / GEMMA4_MAX_ALL_TOKENS).  The old 2048 default is what
    // made the agent compact every single turn.
    const char* configured = std::getenv("GEMMA4_MAX_ALL_TOKENS");
    const int max_all_tokens = configured != nullptr && *configured != '\0'
        ? std::max(512, std::atoi(configured))
        : 2048;
    // prompt_cache + reuse_kv make MNN compare the rendered chat template with
    // the text its KV cache already holds and prefill only the new suffix.  This
    // line was missing, so every text turn re-prefilled the whole conversation
    // (the phone logged 68-212 s per answer with cache_hit=false while the host
    // probe, which did set it, was 10x faster).
    // The sampler is a first-class quality knob for this export: on the same 40
    // turns the host scored 97.5% with the model's own "mixed" configuration and
    // 82.5% with greedy, and seven failures appeared *only* under greedy.  The
    // runner therefore leaves the model's sampler alone unless GEMMA4_SAMPLER
    // pins one explicitly.
    const char* sampler = std::getenv("GEMMA4_SAMPLER");
    const bool pin_sampler = sampler != nullptr && *sampler != '\0';
    return "{\"backend_type\":\"cpu\",\"thread_num\":" +
           std::to_string(text_threads()) +
           ",\"precision\":\"low\",\"memory\":\"low\",\"power\":\"high\","
           "\"async\":false," +
           (pin_sampler ? "\"sampler_type\":\"" + std::string(sampler) + "\"," : "") +
           "\"prompt_cache\":true,\"reuse_kv\":true,\"max_all_tokens\":" +
           std::to_string(max_all_tokens) + "}";
}

// Captures the token text MNN writes to the stream passed to Llm::response().
// MNN flushes after every token, so intercepting the streambuf gives the app
// one callback per decoded token, which is what drives its streaming text and
// the TTFT read-out (`FIRST_TOKEN ttft_ms=...`).
class TokenCaptureBuf : public std::streambuf {
public:
    struct Callback {
        std::ostream* downstream;  // captured text is also forwarded here
        std::function<bool(int token_id, int token_index, const std::string& text)> on_token;
        std::function<void()> on_first_token;
    };

    TokenCaptureBuf(std::streambuf* inner, Callback cb)
        : inner_(inner), callback_(std::move(cb)) {}

    void bind(const Llm* llm) { llm_ = llm; }

protected:
    int overflow(int ch) override {
        if (ch == EOF) {
            return 0;
        }
        const char character = static_cast<char>(ch);
        flush_chunk(&character, 1);
        return ch;
    }

    std::streamsize xsputn(const char* data, std::streamsize count) override {
        flush_chunk(data, count);
        return count;
    }

    int sync() override { return inner_->pubsync(); }

private:
    void flush_chunk(const char* data, std::streamsize count) {
        if (count <= 0) {
            return;
        }
        chunk_.append(data, static_cast<size_t>(count));
        callback_.on_first_token();
        if (callback_.on_token) {
            int token_id = -1;
            int token_index = -1;
            if (llm_ != nullptr && llm_->getContext() != nullptr &&
                !llm_->getContext()->output_tokens.empty()) {
                const auto& tokens = llm_->getContext()->output_tokens;
                token_id = tokens.back();
                token_index = static_cast<int>(tokens.size()) - 1;
            }
            callback_.on_token(token_id, token_index, chunk_);
        }
        if (callback_.downstream != nullptr) {
            callback_.downstream->write(data, count);
            callback_.downstream->flush();
        }
        chunk_.clear();
    }

    std::streambuf* inner_;
    Callback callback_;
    std::string chunk_;
    const Llm* llm_ = nullptr;
};

// --- engine -----------------------------------------------------------------

class HybridEngine {
public:
    HybridEngine(const std::string& config_path, const std::string& vision_path)
        : vision_(vision_path, NpuVision::Role::Front),
          vision_tail_(tail_path_for(vision_path), NpuVision::Role::Tail),
          llm_(Llm::createLLM(config_path)) {
        config_path_ = config_path;
        reload_llm_per_request_ = std::getenv("GEMMA4_NO_RELOAD_LLM") == nullptr;
        if (!llm_) {
            throw std::runtime_error("cannot create MNN LLM");
        }
        if (!llm_->set_config(llm_config_json())) {
            throw std::runtime_error("cannot configure MNN LLM");
        }
        const auto load_begin = Clock::now();
        if (!llm_->load()) {
            throw std::runtime_error("cannot load MNN LLM");
        }
        llm_load_ms_ = milliseconds(load_begin, Clock::now());
        llm_->tuning(MNN::Transformer::OP_ENCODER_NUMBER, {1, 5, 10, 20, 30, 50, 100});
    }

    // MNN's Llm::reset() rewinds the KV cache through the KVMeta remove counter,
    // but with Gemma 4's mixed sliding-window attention the rewound cache keeps
    // leaking context: request 1 is exact, request 2 drifts, request 3 is off
    // topic, even though the injected soft tokens are bit-identical (verified
    // with GEMMA4_DUMP_TOKENS).  Re-creating the engine costs the model load
    // (~1.9 s) and restores a pristine state, so the demo stays correct.
    void reload_llm() {
        llm_.reset();
        llm_.reset(Llm::createLLM(config_path_));
        if (!llm_) {
            throw std::runtime_error("cannot create MNN LLM");
        }
        if (!llm_->set_config(llm_config_json())) {
            throw std::runtime_error("cannot configure MNN LLM");
        }
        const auto load_begin = Clock::now();
        if (!llm_->load()) {
            throw std::runtime_error("cannot load MNN LLM");
        }
        llm_load_ms_ = milliseconds(load_begin, Clock::now());
        llm_->tuning(MNN::Transformer::OP_ENCODER_NUMBER, {1, 5, 10, 20, 30, 50, 100});
    }

    void run(const std::string& input_path, const std::string& prompt_path,
             const std::string& stop_path, int max_tokens, std::ostream& output) {
        consume_file(stop_path);
        // TTFT is measured from here so that a per-request LLM rebuild (trap
        // T15) is visible in the number the UI shows, exactly like any other
        // work between the request and the first token.
        const auto inference_begin = Clock::now();
        // The first request of a fresh process already starts from an empty KV
        // cache, so only later requests pay for the rebuild.
        if (reload_llm_per_request_ && request_count_ > 0) {
            reload_llm();
        }
        ++request_count_;
        llm_->reset();

        const auto prompt_begin = Clock::now();
        const CanvasInput canvas = read_canvas_input(input_path);
        const int image_tokens = canvas.soft_tokens();
        const std::string user_prompt = read_text(prompt_path);
        const auto input_ids = llm_->tokenizer_encode(build_prompt(image_tokens, user_prompt));
        const int prompt_image_tokens = static_cast<int>(
            std::count(input_ids.begin(), input_ids.end(), kImageTokenId));
        if (prompt_image_tokens != image_tokens) {
            throw std::runtime_error(
                "prompt token contract mismatch: image=" +
                std::to_string(prompt_image_tokens) + " expected=" +
                std::to_string(image_tokens));
        }
        const double prompt_prepare_ms = milliseconds(prompt_begin, Clock::now());

        const auto vision_begin = Clock::now();
        const VisionInput vision_input = build_vision_input(canvas);
        const std::vector<float> patch_states = vision_.run(vision_input);
        const std::vector<float> pooled = pool_patches(patch_states);
        const std::vector<float> canvas_tokens = vision_tail_.run_raw(pooled);
        std::vector<float> visual_embeddings = select_soft_tokens(canvas_tokens, canvas);
        apply_embed_scale(visual_embeddings);
        const double vision_ms = milliseconds(vision_begin, Clock::now());
        // Diagnostics: dump exactly what the splice below injects, so a service
        // run can be compared against `npu`/`inject` mode byte for byte.
        if (const char* path = std::getenv("GEMMA4_DUMP_TOKENS")) {
            std::ofstream out(path, std::ios::binary);
            out.write(reinterpret_cast<const char*>(visual_embeddings.data()),
                      static_cast<std::streamsize>(visual_embeddings.size() *
                                                   sizeof(float)));
        }

        const auto inject_begin = Clock::now();
        VARP input_embeddings = llm_->embedding(input_ids);
        if (input_embeddings == nullptr || input_embeddings->getInfo() == nullptr ||
            input_embeddings->getInfo()->size !=
                static_cast<int>(input_ids.size()) * kTextHidden) {
            throw std::runtime_error("LLM embedding shape mismatch");
        }
        float* embedding_data = input_embeddings->writeMap<float>();
        int image_index = 0;
        for (size_t token_index = 0; token_index < input_ids.size(); ++token_index) {
            if (input_ids[token_index] != kImageTokenId) {
                continue;
            }
            std::memcpy(embedding_data + token_index * kTextHidden,
                        visual_embeddings.data() + static_cast<size_t>(image_index) * kTextHidden,
                        kTextHidden * sizeof(float));
            ++image_index;
        }
        const double inject_ms = milliseconds(inject_begin, Clock::now());

        const auto decode_begin = Clock::now();
        bool has_first_token = false;
        bool stopped = false;
        Clock::time_point first_token_time;
        TokenCaptureBuf capture(
            output.rdbuf(),
            TokenCaptureBuf::Callback{
                /*downstream*/ &output,
                /*on_token*/ [&](int token, int index, const std::string& text) {
                    if (consume_file(stop_path)) {
                        stopped = true;
                        return false;
                    }
                    output << "TOKEN index=" << index
                           << " id=" << token
                           << " text_b64=" << base64_encode(text)
                           << std::endl;
                    return true;
                },
                /*on_first_token*/ [&]() {
                    if (has_first_token) {
                        return;
                    }
                    has_first_token = true;
                    first_token_time = Clock::now();
                    const double ttft_ms = milliseconds(inference_begin, first_token_time);
                    output << "FIRST_TOKEN ttft_ms=" << ttft_ms
                           << " wall_ms=" << ttft_ms << std::endl;
                }});
        capture.bind(llm_.get());
        std::ostream capture_stream(&capture);
        if (std::getenv("GEMMA4_NO_CAPTURE") != nullptr) {
            // Diagnostic: hand MNN the log stream directly, like the reference
            // MiniCPM-V runner does (it disabled its capture buffer too).
            llm_->response(input_embeddings, &output, nullptr, max_tokens);
        } else {
            llm_->response(input_embeddings, &capture_stream, nullptr, max_tokens);
        }
        const double decode_wall_ms = milliseconds(decode_begin, Clock::now());

        const auto* context = llm_->getContext();
        const double prefill_ms = context->prefill_us / 1000.0;
        const double decode_ms = context->decode_us / 1000.0;
        const int decode_intervals = std::max(0, context->gen_seq_len - 1);
        output << std::fixed << std::setprecision(3)
               << "\nRESULT prompt_prepare_ms=" << prompt_prepare_ms << '\n'
               << "RESULT llm_load_ms=" << llm_load_ms_ << '\n'
               << "RESULT prompt_tokens=" << input_ids.size() << '\n'
               << "RESULT image_tokens=" << image_tokens << '\n'
               << "RESULT vision_grid_rows=" << canvas.patch_rows << '\n'
               << "RESULT vision_grid_columns=" << canvas.patch_columns << '\n'
               << "RESULT npu_vision_ms=" << vision_ms << '\n'
               << "RESULT embedding_inject_ms=" << inject_ms << '\n'
               << "RESULT llm_prefill_ms=" << prefill_ms << '\n'
               << "RESULT generated_tokens=" << context->gen_seq_len << '\n'
               << "RESULT llm_decode_ms=" << decode_ms << '\n'
               << "RESULT decode_ms_per_token="
               << (decode_intervals > 0 ? decode_ms / decode_intervals : 0.0) << '\n'
               << "RESULT decode_tokens_per_second="
               << (decode_ms > 0.0 ? decode_intervals * 1000.0 / decode_ms : 0.0) << '\n'
               << "RESULT decode_wall_ms=" << decode_wall_ms << '\n'
               << "RESULT ttft_ms="
               << (has_first_token
                       ? milliseconds(inference_begin, first_token_time)
                       : 0.0)
               << '\n'
               << "RESULT stopped=" << stopped << '\n'
               << "RESULT inference_wall_ms="
               << milliseconds(inference_begin, Clock::now()) << '\n'
               << "RESULT eos_reached=" << llm_->stoped() << '\n'
               << "RESULT request_complete=1" << std::endl;
    }

    double llm_load_ms() const { return llm_load_ms_; }

    // --- vision, shared by the legacy request path and the agent ------------
    struct VisionResult {
        std::vector<float> embeddings;   // soft tokens, embed scale applied
        int image_tokens = 0;
        int grid_rows = 0;
        int grid_columns = 0;
        double vision_ms = 0.0;
    };

    VisionResult run_vision(const std::string& input_path) {
        const auto vision_begin = Clock::now();
        const CanvasInput canvas = read_canvas_input(input_path);
        const VisionInput vision_input = build_vision_input(canvas);
        const std::vector<float> patch_states = vision_.run(vision_input);
        const std::vector<float> pooled = pool_patches(patch_states);
        const std::vector<float> canvas_tokens = vision_tail_.run_raw(pooled);
        VisionResult result;
        result.embeddings = select_soft_tokens(canvas_tokens, canvas);
        apply_embed_scale(result.embeddings);
        result.image_tokens = canvas.soft_tokens();
        result.grid_rows = canvas.patch_rows;
        result.grid_columns = canvas.patch_columns;
        result.vision_ms = milliseconds(vision_begin, Clock::now());
        return result;
    }

    // Reads only the canvas header, so the app can learn what the current
    // image costs (visual tokens, fingerprint) without running the NPU.
    bool inspect_canvas(const std::string& path, int* image_tokens, int* grid_rows,
                        int* grid_columns, std::string* sha256, std::string* error) const {
        std::string contents;
        if (!agent::read_file(path, &contents)) {
            if (error != nullptr) {
                *error = "cannot open " + path;
            }
            return false;
        }
        if (contents.size() < 28) {
            if (error != nullptr) {
                *error = "canvas file is too short";
            }
            return false;
        }
        int32_t header[5] = {0, 0, 0, 0, 0};
        std::memcpy(header, contents.data(), sizeof(header));
        if (header[0] != kTileFileMagic || header[1] != kTileFileVersion || header[4] != 1) {
            if (error != nullptr) {
                *error = "unsupported canvas file";
            }
            return false;
        }
        const int rows = header[2];
        const int columns = header[3];
        if (rows <= 0 || columns <= 0 || rows % kPool != 0 || columns % kPool != 0 ||
            rows > kGridHeight || columns > kGridWidth) {
            if (error != nullptr) {
                *error = "invalid canvas patch grid";
            }
            return false;
        }
        if (sha256 != nullptr) {
            *sha256 = agent::sha256_hex(contents);
        }
        if (image_tokens != nullptr) {
            *image_tokens = (rows / kPool) * (columns / kPool);
        }
        if (grid_rows != nullptr) {
            *grid_rows = rows;
        }
        if (grid_columns != nullptr) {
            *grid_columns = columns;
        }
        return true;
    }

    // The front DLA is named ..._front_...dla, the tail ..._tail_...dla.
    static std::string tail_path_for(const std::string& front) {
        const std::string marker = "_front_";
        const size_t position = front.find(marker);
        if (position == std::string::npos) {
            throw std::runtime_error(
                "vision DLA path must contain '_front_' so the tail shard can be found: " +
                front);
        }
        std::string tail = front;
        tail.replace(position, marker.size(), "_tail_");
        return tail;
    }

    NpuVision vision_;
    NpuVision vision_tail_;
    std::string config_path_;
    std::unique_ptr<Llm> llm_;
    double llm_load_ms_ = 0.0;
    bool reload_llm_per_request_ = false;
    int request_count_ = 0;
};

void print_usage(const char* program) {
    std::cerr << "Usage: " << program
              << " npu CONFIG VISION_DLA CANVAS_BIN [REFERENCE_BIN] [MAX_TOKENS]\n"
              << "   or: " << program
              << " --service CONFIG VISION_DLA CANVAS_BIN PROMPT REQUEST STOP LOG READY "
                 "[MAX_TOKENS]\n"
              << "   or: " << program
              << " --agent-service CONFIG VISION_DLA RUNTIME_ROOT AGENT_DIR "
                 "[MAX_TOKENS]\n";
}

// --- agent driver -----------------------------------------------------------
//
// Wraps the hybrid engine for the agent core:
//   * a text turn continues the resident KV cache (MNN prompt cache prefills
//     only the new suffix, so a follow-up question costs a fraction of a full
//     prefill),
//   * a turn that has to change the conversation (new session, compaction, an
//     image, a cancelled turn) re-creates the engine first: `Llm::reset()` does
//     not fully clear Gemma 4's mixed sliding-window KV cache (see helpers/STATUS.md),
//     while a fresh engine is bit-exact, at the cost of the model load.
class MnnAgentLlm : public agent::AgentLlm {
public:
    MnnAgentLlm(const std::string& config_path, const std::string& vision_path,
                HybridEngine* engine)
        : engine_(engine), config_path_(config_path), vision_path_(vision_path) {
        pristine_engine_ = std::getenv("GEMMA4_PRISTINE_ENGINE") != nullptr;
    }

    std::string generate_delta(
        const agent::ChatMessages& messages, const std::vector<agent::ImageSplice>& splices,
        int max_tokens, const std::function<bool(int, const std::string&)>& on_token,
        agent::GenerationMetrics* metrics, std::string* error) override {
        Llm* llm = engine_->llm_.get();
        // Only a prompt that *extends* what the KV cache holds may be prefilled
        // incrementally.  MNN answers a rewritten prompt by erasing the KV and
        // re-prefilling it in place, and on this device that leaves Gemma 4's
        // mixed sliding-window cache inconsistent: from the next turn on the
        // model emits its end-of-turn token immediately (verified on the phone
        // with the mixed conversation probe - "rewind" produced a one-token
        // answer on every later turn while the re-created engine answered).  So
        // the driver checks the contract itself and falls back to a pristine
        // engine instead of letting the engine rewind.
        if (!cache_covers(messages)) {
            if (std::getenv("GEMMA4_DEBUG_PHASES") != nullptr) {
                size_t index = 0;
                while (index < cached_messages_.size() && index < messages.size() &&
                       cached_messages_[index].first == messages[index].first &&
                       cached_messages_[index].second == messages[index].second) {
                    ++index;
                }
                std::cerr << "[cache] miss at message " << index << "/"
                          << cached_messages_.size() << " new=" << messages.size();
                if (index < cached_messages_.size()) {
                    std::cerr << " cached[" << index << "]=" << cached_messages_[index].first
                              << ":" << cached_messages_[index].second.substr(0, 60);
                }
                if (index < messages.size()) {
                    std::cerr << " new[" << index << "]=" << messages[index].first << ":"
                              << messages[index].second.substr(0, 60);
                }
                std::cerr << std::endl;
            }
            return generate_full(messages, splices, max_tokens, on_token, metrics, error);
        }
        const auto phase_begin = Clock::now();
        const std::string prompt = llm->apply_chat_template(to_mnn(messages));
        dump_prompt(prompt, splices, messages);
        const auto phase_template = Clock::now();
        const std::vector<int> ids = llm->tokenizer_encode(prompt);
        const auto phase_encode = Clock::now();
        const int placeholders =
            static_cast<int>(std::count(ids.begin(), ids.end(), kImageTokenId));
        if (std::getenv("GEMMA4_DEBUG_PHASES") != nullptr) {
            std::cerr << "[phase] template=" << milliseconds(phase_begin, phase_template)
                      << "ms encode=" << milliseconds(phase_template, phase_encode)
                      << "ms tokens=" << ids.size() << std::endl;
        }
        int expected = 0;
        for (const auto& splice : splices) {
            expected += splice.tokens;
        }
        if (placeholders != expected) {
            if (error != nullptr) {
                *error = "prompt/image contract mismatch: placeholders=" +
                         std::to_string(placeholders) + " expected=" +
                         std::to_string(expected);
            }
            return std::string();
        }
        // An image turn no longer re-creates the engine: the vision vectors are
        // handed to MNN as a soft-token provider, so the engine can prefill just
        // the new suffix and splice the pictures the suffix carries.  Placeholders
        // that already sit in the KV cache (an image from an earlier turn) are
        // never re-encoded, so a follow-up turn costs no vision work at all.
        install_soft_token_provider(splices);
        if (metrics != nullptr) {
            metrics->full_prefill = !cache_covers(messages);
        }
        const std::string output =
            run_generation(messages, ids, nullptr, max_tokens, on_token, metrics, error);
        if (std::getenv("GEMMA4_DEBUG_PHASES") != nullptr) {
            std::cerr << "[phase] engine=" << milliseconds(phase_encode, Clock::now()) << "ms"
                      << std::endl;
        }
        if (metrics != nullptr) {
            metrics->npu_ms += splice_vision_ms_;
        }
        if (!splice_error_.empty()) {
            // The splice never happened, so whatever the model produced was
            // conditioned on an empty patch of image embeddings.
            if (error != nullptr) {
                *error = splice_error_;
            }
            splice_error_.clear();
            return std::string();
        }
        return output;
    }

    std::string generate_full(
        const agent::ChatMessages& messages, const std::vector<agent::ImageSplice>& splices,
        int max_tokens, const std::function<bool(int, const std::string&)>& on_token,
        agent::GenerationMetrics* metrics, std::string* error) override {
        // A full prefill starts from a pristine engine.  Rewinding the KV through
        // MNN's erase counter *looked* like a 2.3 s win, but on this device the
        // rewound Gemma 4 cache is left inconsistent: the model answers with its
        // end-of-turn token on every following turn (trap T15, reproduced with
        // the mixed conversation probe; the same turns answer normally once the
        // engine is re-created).  The reload stays; what is *kept* is the
        // incremental path - a turn whose prompt extends the cache, images
        // included, never comes through here.
        engine_->reload_llm();
        Llm* llm = engine_->llm_.get();
        const bool pristine = pristine_engine_;
        const std::string prompt = llm->apply_chat_template(to_mnn(messages));
        dump_prompt(prompt, splices, messages);
        const std::vector<int> ids = llm->tokenizer_encode(prompt);
        const int placeholders =
            static_cast<int>(std::count(ids.begin(), ids.end(), kImageTokenId));

        std::vector<agent::ImageSplice> ordered = splices;
        std::sort(ordered.begin(), ordered.end(),
                  [](const agent::ImageSplice& left, const agent::ImageSplice& right) {
                      return left.message_index < right.message_index;
                  });
        int expected = 0;
        for (const auto& splice : ordered) {
            expected += splice.tokens;
        }
        if (placeholders != expected) {
            if (error != nullptr) {
                *error = "prompt/image contract mismatch: placeholders=" +
                         std::to_string(placeholders) + " expected=" + std::to_string(expected);
            }
            return std::string();
        }

        if (!pristine) {
            // Fresh engine plus MNN's own prompt pipeline: tokenize, embed, splice
            // the pictures the prompt carries through the soft-token provider,
            // prefill.  No embedding materialisation and no separate cache
            // bookkeeping are needed - the engine updates its prompt cache itself.
            install_soft_token_provider(splices);
            if (metrics != nullptr) {
                metrics->full_prefill = true;
                metrics->load_ms = engine_->llm_load_ms();
            }
            const std::string output =
                run_generation(messages, ids, nullptr, max_tokens, on_token, metrics, error);
            if (metrics != nullptr) {
                metrics->npu_ms += splice_vision_ms_;
            }
            if (!splice_error_.empty()) {
                if (error != nullptr) {
                    *error = splice_error_;
                }
                splice_error_.clear();
                return std::string();
            }
            return output;
        }

        VARP input_embeddings = llm->embedding(ids);
        if (input_embeddings == nullptr || input_embeddings->getInfo() == nullptr ||
            input_embeddings->getInfo()->size !=
                static_cast<int>(ids.size()) * kTextHidden) {
            if (error != nullptr) {
                *error = "LLM embedding shape mismatch";
            }
            return std::string();
        }
        float* embedding_data = input_embeddings->writeMap<float>();
        // Vision runs once per spliced image; the soft tokens are then written
        // into the placeholder positions in prompt order.
        std::vector<float> soft_tokens;
        soft_tokens.reserve(static_cast<size_t>(expected) * kTextHidden);
        for (const auto& splice : ordered) {
            const HybridEngine::VisionResult vision = engine_->run_vision(splice.canvas_path);
            if (vision.image_tokens != splice.tokens) {
                if (error != nullptr) {
                    *error = "cached canvas produced " + std::to_string(vision.image_tokens) +
                             " soft tokens, expected " + std::to_string(splice.tokens);
                }
                return std::string();
            }
            if (metrics != nullptr) {
                metrics->npu_ms += vision.vision_ms;
            }
            soft_tokens.insert(soft_tokens.end(), vision.embeddings.begin(),
                               vision.embeddings.end());
        }
        if (static_cast<int>(soft_tokens.size() / kTextHidden) != expected) {
            if (error != nullptr) {
                *error = "vision produced " +
                         std::to_string(soft_tokens.size() / kTextHidden) +
                         " soft tokens, expected " + std::to_string(expected);
            }
            return std::string();
        }
        int ordinal = 0;
        for (size_t token = 0; token < ids.size(); ++token) {
            if (ids[token] != kImageTokenId) {
                continue;
            }
            std::memcpy(embedding_data + token * kTextHidden,
                        soft_tokens.data() + static_cast<size_t>(ordinal) * kTextHidden,
                        kTextHidden * sizeof(float));
            ++ordinal;
        }
        if (metrics != nullptr) {
            metrics->full_prefill = true;
            metrics->load_ms = engine_->llm_load_ms();
        }
        const std::string output = run_generation(messages, ids, &input_embeddings, max_tokens,
                                                  on_token, metrics, error);
        // Tell MNN's prompt cache what this KV state holds (the spliced prompt
        // plus the response) so the rest of the turn can prefill only its
        // suffix instead of re-running the vision tower and the full prefill.
        agent::ChatMessages with_response = messages;
        with_response.emplace_back("assistant", output);
        llm->syncPromptCache(to_mnn(with_response));
        cached_messages_ = std::move(with_response);
        return output;
    }

    bool reset_conversation(std::string* error) override {
        try {
            engine_->reload_llm();
            cached_messages_.clear();
            return true;
        } catch (const std::exception& failure) {
            if (error != nullptr) {
                *error = failure.what();
            }
            return false;
        }
    }

    void sync_transcript(const agent::ChatMessages& messages) override {
        // Tell MNN what this KV state holds *as the transcript will render it*,
        // so a turn whose stored answer was cleaned (a `<turn|>` artefact or a
        // stray space removed) still continues incrementally.
        engine_->llm_->syncPromptCache(to_mnn(messages));
        cached_messages_ = messages;
    }

    int count_tokens(const agent::ChatMessages& messages) const override {
        Llm* llm = engine_->llm_.get();
        const std::string prompt = llm->apply_chat_template(to_mnn(messages));
        return static_cast<int>(llm->tokenizer_encode(prompt).size());
    }

    bool inspect_image(const std::string& canvas_path, agent::ImageInput* image,
                       std::string* error) const override {
        int tokens = 0;
        int rows = 0;
        int columns = 0;
        std::string sha;
        if (!engine_->inspect_canvas(canvas_path, &tokens, &rows, &columns, &sha, error)) {
            return false;
        }
        image->valid = true;
        image->canvas_path = canvas_path;
        image->tokens = tokens;
        image->canvas_width = columns * kPatch;
        image->canvas_height = rows * kPatch;
        image->sha256 = sha;
        image->note = std::to_string(image->canvas_width) + "x" +
                      std::to_string(image->canvas_height) + " letterboxed canvas, " +
                      std::to_string(tokens) + " visual tokens (W8A16 vision DLA)";
        return true;
    }

    agent::Json describe() const override {
        agent::Json out = agent::Json::object();
        out.set("backend", agent::Json::string("mnn-q4 + neuron-w8a16"));
        out.set("platform", agent::Json::string(model_platform()));
        out.set("threads", agent::Json::integer(text_threads()));
        out.set("canvas", agent::Json::string(std::to_string(kCanvasWidth) + "x" +
                                              std::to_string(kCanvasHeight)));
        out.set("soft_tokens", agent::Json::integer(kSoftTokens));
        out.set("grid", agent::Json::string(std::to_string(kGridWidth) + "x" +
                                            std::to_string(kGridHeight)));
        out.set("vision_dla", agent::Json::string(agent::file_name(vision_path_)));
        out.set("text_model", agent::Json::string(agent::file_name(config_path_)));
        return out;
    }

private:
    // GEMMA4_DUMP_PROMPT=<path> writes the exact prompt text the engine is
    // handed (image placeholders collapsed to `<|image|> xN` so the dump stays
    // readable), one block per generation.
    void dump_prompt(const std::string& prompt, const std::vector<agent::ImageSplice>& splices,
                     const agent::ChatMessages& messages) const {
        const char* path = std::getenv("GEMMA4_DUMP_PROMPT");
        if (path == nullptr || *path == '\0') {
            return;
        }
        static int call = 0;
        ++call;
        std::string collapsed;
        collapsed.reserve(prompt.size());
        size_t index = 0;
        int pads = 0;
        while (index < prompt.size()) {
            if (prompt.compare(index, 9, kImageToken) == 0) {
                ++pads;
                index += 9;
                continue;
            }
            if (pads > 0) {
                collapsed += "[" + std::to_string(pads) + " x <|image|>]";
                pads = 0;
            }
            collapsed += prompt[index++];
        }
        if (pads > 0) {
            collapsed += "[" + std::to_string(pads) + " x <|image|>]";
        }
        std::ofstream out(path, std::ios::app);
        if (!out) {
            return;
        }
        out << "\n=== call " << call << " messages=" << messages.size()
            << " splices=" << splices.size() << " prompt_tokens≈" << prompt.size() << "\n"
            << collapsed << "\n";
    }

    // One picture of the current turn, with the vision vectors materialised the
    // first time MNN asks for one of its soft tokens.  The engine only requests
    // the occurrences that live in the tokens it is about to prefill, so an
    // image that is already part of the KV cache costs nothing.
    struct SoftTokenSlot {
        std::string canvas_path;
        int base = 0;      // first occurrence index inside this prompt
        int tokens = 0;    // visual tokens this picture contributes
        bool ready = false;
        std::vector<float> embeddings;
    };

    void install_soft_token_provider(const std::vector<agent::ImageSplice>& splices) {
        soft_slots_.clear();
        splice_vision_ms_ = 0.0;
        splice_error_.clear();
        std::vector<agent::ImageSplice> ordered = splices;
        std::sort(ordered.begin(), ordered.end(),
                  [](const agent::ImageSplice& left, const agent::ImageSplice& right) {
                      return left.message_index < right.message_index;
                  });
        int base = 0;
        for (const auto& splice : ordered) {
            SoftTokenSlot slot;
            slot.canvas_path = splice.canvas_path;
            slot.base = base;
            slot.tokens = splice.tokens;
            base += splice.tokens;
            soft_slots_.push_back(std::move(slot));
        }
        engine_->llm_->setSoftTokenProvider(
            [this](int occurrence) -> const float* { return soft_token_row(occurrence); });
    }

    const float* soft_token_row(int occurrence) {
        for (auto& slot : soft_slots_) {
            if (occurrence < slot.base || occurrence >= slot.base + slot.tokens) {
                continue;
            }
            if (!slot.ready) {
                slot.ready = true;
                try {
                    const HybridEngine::VisionResult vision =
                        engine_->run_vision(slot.canvas_path);
                    if (vision.image_tokens != slot.tokens) {
                        splice_error_ =
                            "cached canvas produced " + std::to_string(vision.image_tokens) +
                            " soft tokens, expected " + std::to_string(slot.tokens);
                        return nullptr;
                    }
                    splice_vision_ms_ += vision.vision_ms;
                    slot.embeddings = vision.embeddings;
                } catch (const std::exception& failure) {
                    splice_error_ = failure.what();
                    return nullptr;
                }
            }
            if (slot.embeddings.empty()) {
                return nullptr;
            }
            const int index = occurrence - slot.base;
            return slot.embeddings.data() + static_cast<size_t>(index) * kTextHidden;
        }
        return nullptr;
    }

    // True when the KV cache was built from a prefix of `messages`, i.e. the
    // engine only has to prefill the remaining messages.  Comparing the message
    // list avoids the whitespace/thinking-channel differences that a comparison
    // of the rendered template text runs into.
    bool cache_covers(const agent::ChatMessages& messages) const {
        if (cached_messages_.empty() || messages.size() < cached_messages_.size()) {
            return false;
        }
        for (size_t index = 0; index < cached_messages_.size(); ++index) {
            if (messages[index].first != cached_messages_[index].first ||
                messages[index].second != cached_messages_[index].second) {
                return false;
            }
        }
        return true;
    }

    static MNN::Transformer::ChatMessages to_mnn(const agent::ChatMessages& messages) {
        MNN::Transformer::ChatMessages out;
        out.reserve(messages.size());
        for (const auto& message : messages) {
            out.emplace_back(message.first, message.second);
        }
        return out;
    }

    std::string run_generation(
        const agent::ChatMessages& messages, const std::vector<int>& ids,
        MNN::Express::VARP* input_embeddings, int max_tokens,
        const std::function<bool(int, const std::string&)>& on_token,
        agent::GenerationMetrics* metrics, std::string* error) {
        Llm* llm = engine_->llm_.get();
        const auto* before = llm->getContext();
        const int before_tokens = before != nullptr ? before->all_seq_len : 0;
        const auto begin = Clock::now();
        bool first_token = false;
        bool cancelled = false;
        double ttft_ms = 0.0;
        std::string streamed;
        // The shipped arm64 MNN runtime does not export setTokenCallback(), so
        // the per-token hook is the streambuf the engine writes to (the same
        // mechanism the legacy request path uses).  The return value of the
        // callback cannot abort a running decode: a cancelled turn stops being
        // streamed and the loop ends after the current step (bounded by
        // max_tokens per step / the engine's timeout_ms).
        // TokenCaptureBuf forwards everything to `inner` as well, so give it a
        // sink that throws the text away (the agent returns the engine's own
        // generate_str as the authoritative text).
        class NullBuf : public std::streambuf {
        protected:
            int overflow(int character) override {
                return character;
            }
        };
        NullBuf null_buffer;
        TokenCaptureBuf capture(
            &null_buffer,
            TokenCaptureBuf::Callback{
                /*downstream*/ nullptr,
                /*on_token*/
                [&](int, int index, const std::string& text) {
                    if (!first_token) {
                        first_token = true;
                        ttft_ms = milliseconds(begin, Clock::now());
                    }
                    if (cancelled) {
                        return false;
                    }
                    streamed += text;
                    if (on_token && !on_token(index, text)) {
                        cancelled = true;
                    }
                    return false;
                },
                /*on_first_token*/ []() {}});
        capture.bind(llm);
        std::ostream capture_stream(&capture);
        try {
            if (input_embeddings != nullptr) {
                llm->response(*input_embeddings, &capture_stream, "", max_tokens);
            } else {
                // The prompt-cache path compares the rendered chat template and
                // prefills only the suffix, which keeps a text turn cheap.
                llm->response(to_mnn(messages), &capture_stream, "", max_tokens);
            }
        } catch (const std::exception& failure) {
            if (error != nullptr) {
                *error = failure.what();
            }
            return streamed;
        }

        const auto* context = llm->getContext();
        if (context == nullptr) {
            if (error != nullptr) {
                *error = "the engine lost its context";
            }
            return streamed;
        }
        if (metrics != nullptr) {
            const int added = context->all_seq_len - before_tokens;
            metrics->context_tokens = context->all_seq_len;
            metrics->generated_tokens = context->gen_seq_len;
            // `prompt_len` is what the engine tokenised and pushed in this call:
            // the suffix when the prompt cache matched, the whole conversation
            // when it had to re-prefill.  It is the honest cache-hit signal.
            metrics->engine_prefill_tokens = context->prompt_len;
            metrics->prompt_tokens = std::max(0, added - context->gen_seq_len);
            metrics->cache_hit = !metrics->full_prefill &&
                                 metrics->prompt_tokens < static_cast<int>(ids.size());
            metrics->cache_delta_tokens =
                metrics->cache_hit ? metrics->prompt_tokens : -1;
            metrics->prefill_ms = context->prefill_us / 1000.0;
            metrics->decode_ms = context->decode_us / 1000.0;
            const int intervals = std::max(0, context->gen_seq_len - 1);
            metrics->decode_tokens_per_second =
                context->decode_us > 0
                    ? intervals * 1e6 / static_cast<double>(context->decode_us)
                    : 0.0;
            metrics->ttft_ms = first_token ? ttft_ms : 0.0;
            if (cancelled) {
                metrics->cancelled = true;
                metrics->stop_reason = "cancelled";
            } else if (llm->stoped()) {
                metrics->stop_reason = "eos";
            } else {
                metrics->stop_reason = "max_tokens";
            }
        }
        // Remember what this KV state holds so the next call can tell whether its
        // prompt extends it (delta) or replaces it (full prefill).
        cached_messages_ = messages;
        if (!context->generate_str.empty()) {
            cached_messages_.emplace_back("assistant", context->generate_str);
        }
        // Keep the engine's own idea of the cached prompt in sync with ours: the
        // driver's prefix check and MNN's text comparison then agree, and a turn
        // that does not extend the cache is routed to a pristine engine instead
        // of being re-prefilled over a rewound one.
        llm->syncPromptCache(to_mnn(cached_messages_));
        // The engine's own text is authoritative: it is what the KV cache holds.
        return context->generate_str;
    }

    HybridEngine* engine_;
    std::string config_path_;
    std::string vision_path_;
    // Messages the current KV state was built from (plus the response), used to
    // decide between the delta and the full-prefill path.
    agent::ChatMessages cached_messages_;
    std::vector<SoftTokenSlot> soft_slots_;
    double splice_vision_ms_ = 0.0;
    std::string splice_error_;
    bool pristine_engine_ = false;
};

// --- agent service ----------------------------------------------------------
//
// Resident process shared by the app's chat UI and the legacy single-shot
// request path:
//   <root>/agent/request.json   request document, written by the app
//   <root>/agent/events.jsonl   append-only event stream the app tails
//   <root>/agent/response.json  reply to the last request
//   <root>/agent/cancel         sentinel that stops the running turn
//   <root>/agent/service-ready  written once the models are loaded
//   <root>/app-request          legacy demo/benchmark trigger (kept working)
int run_agent_service(const std::string& config_path, const std::string& vision_path,
                      const std::string& root, const std::string& agent_dir, int max_tokens) {
    agent::make_directories(root);
    agent::make_directories(agent_dir);

    agent::AgentConfig config =
        agent::AgentConfig::load(agent::path_join(root, "agent-config.json"), root);
    // The KV allocation is a model-load parameter, so it has to be in the
    // environment before the engine is created.
    setenv("GEMMA4_MAX_ALL_TOKENS", std::to_string(config.kv_total_tokens).c_str(), 0);

    const auto startup_begin = Clock::now();
    HybridEngine engine(config_path, vision_path);
    MnnAgentLlm driver(config_path, vision_path, &engine);
    const double startup_ms = milliseconds(startup_begin, Clock::now());
    if (!agent_dir.empty()) {
        config.agent_dir = agent_dir;
        config.memory_dir = agent::path_join(agent_dir, "memory");
        config.sessions_dir = agent::path_join(agent_dir, "sessions");
        config.work_dir = agent::path_join(agent_dir, "work");
    }

    agent::AgentService::Options service_options;
    service_options.agent_dir = config.agent_dir;
    service_options.root = root;
    service_options.poll_interval_ms = config.poll_interval_ms;
    service_options.max_tokens = max_tokens;
    service_options.echo = std::getenv("GEMMA4_AGENT_QUIET") == nullptr;
    agent::AgentService service(std::move(service_options));
    std::string error;
    if (!service.prepare(&error)) {
        std::cerr << "ERROR cannot prepare the agent directory: " << error << std::endl;
        return 1;
    }

    agent::Json device = agent::Json::object();
    device.set("platform", agent::Json::string(model_platform()));
    device.set("vision", agent::Json::string("W8A16"));
    device.set("text_model", agent::Json::string("Q4"));
    device.set("threads", agent::Json::integer(text_threads()));
    device.set("canvas_width", agent::Json::integer(kCanvasWidth));
    device.set("canvas_height", agent::Json::integer(kCanvasHeight));
    device.set("soft_tokens", agent::Json::integer(kSoftTokens));
    device.set("agent_dir", agent::Json::string(config.agent_dir));

    agent::AgentRuntime::Options options;
    options.config = config;
    options.root = root;
    options.device = device;
    if (const char* value = std::getenv("GEMMA4_AGENT_MODEL_MEMORY")) {
        options.model_memory_extraction = std::string(value) != "0";
    }

    agent::AgentRuntime runtime(
        options, driver, [&](const agent::Json& event) { service.emit(event); },
        [&]() { return service.cancel_requested(); });
    if (!runtime.initialize(&error)) {
        std::cerr << "ERROR cannot initialise the agent: " << error << std::endl;
        return 1;
    }

    service.emit(service.hello(runtime, startup_ms, engine.llm_load_ms(), root));
    service.write_ready("READY startup_ms=" + std::to_string(startup_ms) +
                        " llm_load_ms=" + std::to_string(engine.llm_load_ms()) +
                        " platform=" + model_platform() +
                        " session=" + runtime.status().string_or("session", "") +
                        " protocol=gemma4-agent/1");
    std::cout << "AGENT READY startup_ms=" << startup_ms
              << " llm_load_ms=" << engine.llm_load_ms()
              << " platform=" << model_platform()
              << " canvas=" << kCanvasWidth << "x" << kCanvasHeight
              << " soft_tokens=" << kSoftTokens
              // Which sampler is actually in effect: the model's own config
              // unless GEMMA4_SAMPLER pins one (the evaluation showed ~15 points
              // between "mixed" and "greedy", so this must be visible in the log).
              << " sampler=" << (std::getenv("GEMMA4_SAMPLER") != nullptr
                                     ? std::getenv("GEMMA4_SAMPLER")
                                     : "model-default(mixed)")
              << std::endl;

    const auto legacy = [&]() {
        // The benchmark/demo path: one image, one answer, the legacy log format
        // the existing UI and BenchmarkReceiver parse.
        std::ofstream log(agent::path_join(root, "app-run.log"), std::ios::trunc);
        try {
            engine.run(agent::path_join(root, "app-input-fp32.bin"),
                       agent::path_join(root, "app-prompt.txt"),
                       agent::path_join(root, "app-stop"), max_tokens, log);
        } catch (const std::exception& failure) {
            log << "\nERROR " << failure.what() << std::endl;
        }
    };

    std::cout << "AGENT LOOP" << std::endl;
    while (true) {
        if (service.handle_pending(runtime, legacy)) {
            continue;
        }
        if (service.cancel_requested()) {
            service.clear_cancel();
        }
        std::this_thread::sleep_for(
            std::chrono::milliseconds(std::max(5, config.poll_interval_ms)));
    }
    return 0;
}
}  // namespace

int main(int argc, char** argv) {
    const bool service_mode = argc > 1 && std::string(argv[1]) == "--service";
    const bool agent_service_mode = argc > 1 &&
                                    std::string(argv[1]) == "--agent-service";
    if (agent_service_mode) {
        if (argc < 6 || argc > 7) {
            print_usage(argv[0]);
            return 2;
        }
    }
    if (!service_mode && (argc < 5 || argc > 7)) {
        if (!agent_service_mode) {
            print_usage(argv[0]);
            return 2;
        }
    }
    if (service_mode && (argc < 10 || argc > 11)) {
        print_usage(argv[0]);
        return 2;
    }

    try {
        pin_to_performance_cores();
        if (agent_service_mode) {
            const int max_tokens = argc == 7 ? std::stoi(argv[6]) : 512;
            return run_agent_service(argv[2], argv[3], argv[4], argv[5], max_tokens);
        }
        if (service_mode) {
            const auto startup_begin = Clock::now();
            HybridEngine engine(argv[2], argv[3]);
            const double startup_ms = milliseconds(startup_begin, Clock::now());
            const std::string input_path = argv[4];
            const std::string prompt_path = argv[5];
            const std::string request_path = argv[6];
            const std::string stop_path = argv[7];
            const std::string log_path = argv[8];
            const std::string ready_path = argv[9];
            const int max_tokens = argc == 11 ? std::stoi(argv[10]) : 512;
            write_ready_file(
                ready_path,
                "READY startup_ms=" + std::to_string(startup_ms) +
                    " llm_load_ms=" + std::to_string(engine.llm_load_ms()) +
                    " platform=" + model_platform());
            while (true) {
                if (!consume_file(request_path)) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    continue;
                }
                try {
                    std::ofstream log(log_path, std::ios::trunc);
                    if (!log) {
                        throw std::runtime_error("cannot open request log");
                    }
                    engine.run(input_path, prompt_path, stop_path, max_tokens, log);
                } catch (const std::exception& error) {
                    std::ofstream log(log_path, std::ios::app);
                    log << "\nERROR " << error.what() << std::endl;
                }
            }
        }

        const int max_tokens = argc == 7 ? std::stoi(argv[6]) : 0;
        const bool has_reference = argc >= 6;

        if (std::string(argv[1]) == "raw") {
            // raw DLA INPUT_BIN [DUMP_BIN]: run any DLA with a raw float tensor.
            NpuVision dla(argv[3], NpuVision::Role::Single, false);
            dla.describe();
            const auto input = read_floats(argv[4], dla.input_elements());
            const auto run_begin = Clock::now();
            const auto output = dla.run_raw(input);
            const double run_ms = milliseconds(run_begin, Clock::now());
            double sum = 0.0;
            double max_abs = 0.0;
            for (float value : output) {
                sum += std::fabs(value);
                max_abs = std::max(max_abs, std::fabs(static_cast<double>(value)));
            }
            std::cout << std::fixed << std::setprecision(6)
                      << "RAW run_ms=" << run_ms
                      << " input_elements=" << dla.input_elements()
                      << " output_elements=" << dla.output_elements()
                      << " mean_abs=" << (sum / std::max<size_t>(1, output.size()))
                      << " max_abs=" << max_abs
                      << " first=" << output[0] << "," << output[1] << "," << output[2]
                      << std::endl;
            if (argc > 5) {
                std::ofstream dump(argv[5], std::ios::binary);
                dump.write(reinterpret_cast<const char*>(output.data()),
                           static_cast<std::streamsize>(output.size() * sizeof(float)));
            }
            return 0;
        }

        if (std::string(argv[1]) == "text") {
            // text CONFIG PROMPT [MAX_TOKENS]: exercise the MNN Q4 backbone
            // without the vision stage.
            std::unique_ptr<Llm> llm(Llm::createLLM(argv[2]));
            if (!llm) {
                throw std::runtime_error("cannot create MNN LLM");
            }
            if (!llm->set_config(llm_config_json())) {
                throw std::runtime_error("cannot configure MNN LLM");
            }
            const auto load_begin = Clock::now();
            if (!llm->load()) {
                throw std::runtime_error("cannot load MNN LLM");
            }
            llm->tuning(MNN::Transformer::OP_ENCODER_NUMBER, {1, 5, 10, 20, 30, 50, 100});
            const int max_tokens = argc > 4 ? std::stoi(argv[4]) : 32;
            const std::string prompt = read_text(argc > 3 ? argv[3] : "");
            const auto ids = llm->tokenizer_encode(prompt);
            std::cout << "TEXT llm_load_ms=" << milliseconds(load_begin, Clock::now())
                      << " prompt_tokens=" << ids.size() << "\n---\n"
                      << std::flush;
            llm->response(ids, &std::cout, nullptr, max_tokens);
            const auto* context = llm->getContext();
            std::cout << "\n---\nTEXT generated_tokens=" << context->gen_seq_len
                      << " decode_tokens_per_second="
                      << (context->decode_us > 0
                              ? (std::max(0, context->gen_seq_len - 1)) * 1e6 / context->decode_us
                              : 0.0)
                      << " prefill_ms=" << context->prefill_us / 1000.0 << std::endl;
            return 0;
        }

        if (std::string(argv[1]) == "inject") {
            // inject CONFIG EMBEDDINGS PROMPT [MAX_TOKENS]
            // Runs the full prompt/embedding-splice path with visual tokens
            // produced elsewhere (the host fixture, or an NPU run).
            std::unique_ptr<Llm> llm(Llm::createLLM(argv[2]));
            if (!llm) {
                throw std::runtime_error("cannot create MNN LLM");
            }
            if (!llm->set_config(llm_config_json())) {
                throw std::runtime_error("cannot configure MNN LLM");
            }
            if (!llm->load()) {
                throw std::runtime_error("cannot load MNN LLM");
            }
            llm->tuning(MNN::Transformer::OP_ENCODER_NUMBER, {1, 5, 10, 20, 30, 50, 100});

            std::ifstream stream(argv[3], std::ios::binary | std::ios::ate);
            if (!stream) {
                throw std::runtime_error("cannot open " + std::string(argv[3]));
            }
            const size_t bytes = static_cast<size_t>(stream.tellg());
            stream.seekg(0);
            std::vector<float> embeddings(bytes / sizeof(float));
            stream.read(reinterpret_cast<char*>(embeddings.data()),
                        static_cast<std::streamsize>(bytes));
            apply_embed_scale(embeddings);
            const int image_tokens = static_cast<int>(embeddings.size() / kTextHidden);
            const std::string user_prompt = read_text(argc > 4 ? argv[4] : "");
            const auto input_ids = llm->tokenizer_encode(build_prompt(image_tokens, user_prompt));
            const int prompt_image_tokens = static_cast<int>(
                std::count(input_ids.begin(), input_ids.end(), kImageTokenId));
            if (prompt_image_tokens != image_tokens) {
                throw std::runtime_error("prompt/image token mismatch");
            }
            VARP input_embeddings = llm->embedding(input_ids);
            float* data = input_embeddings->writeMap<float>();
            int index = 0;
            for (size_t token = 0; token < input_ids.size(); ++token) {
                if (input_ids[token] != kImageTokenId) {
                    continue;
                }
                std::memcpy(data + token * kTextHidden,
                            embeddings.data() + static_cast<size_t>(index) * kTextHidden,
                            kTextHidden * sizeof(float));
                ++index;
            }
            const int max_tokens = argc > 5 ? std::stoi(argv[5]) : 64;
            std::cout << "INJECT image_tokens=" << image_tokens
                      << " prompt_tokens=" << input_ids.size() << "\n---\n" << std::flush;
            llm->response(input_embeddings, &std::cout, nullptr, max_tokens);
            const auto* context = llm->getContext();
            std::cout << "\n---\nINJECT generated_tokens=" << context->gen_seq_len
                      << " decode_tokens_per_second="
                      << (context->decode_us > 0
                              ? (std::max(0, context->gen_seq_len - 1)) * 1e6 / context->decode_us
                              : 0.0)
                      << " prefill_ms=" << context->prefill_us / 1000.0 << std::endl;
            return 0;
        }

        // Test mode: run one image, print accuracy when a reference is given.
        const CanvasInput canvas = read_canvas_input(argv[4]);
        const bool debug = std::getenv("GEMMA4_DEBUG") != nullptr;
        const auto prepare_begin = Clock::now();
        const VisionInput vision_input = build_vision_input(canvas);
        const double prepare_ms = milliseconds(prepare_begin, Clock::now());
        const auto vision_begin = Clock::now();
        NpuVision front(argv[3], NpuVision::Role::Front);
        NpuVision tail(HybridEngine::tail_path_for(argv[3]), NpuVision::Role::Tail);
        const auto front_begin = Clock::now();
        const std::vector<float> patch_states = front.run(vision_input);
        const double front_ms = milliseconds(front_begin, Clock::now());
        if (std::getenv("GEMMA4_DEBUG") != nullptr) {
            std::cout << "DEBUG patch_states=" << patch_states.size() << std::endl;
        }
        const auto pool_begin = Clock::now();
        const std::vector<float> pooled = pool_patches(patch_states);
        const double pool_ms = milliseconds(pool_begin, Clock::now());
        if (const char* path = std::getenv("GEMMA4_DUMP_STATES")) {
            std::ofstream out(path, std::ios::binary);
            out.write(reinterpret_cast<const char*>(patch_states.data()),
                      static_cast<std::streamsize>(patch_states.size() * sizeof(float)));
        }
        if (const char* path = std::getenv("GEMMA4_DUMP_POOLED")) {
            std::ofstream out(path, std::ios::binary);
            out.write(reinterpret_cast<const char*>(pooled.data()),
                      static_cast<std::streamsize>(pooled.size() * sizeof(float)));
        }
        if (std::getenv("GEMMA4_DEBUG") != nullptr) {
            std::cout << "DEBUG pooled=" << pooled.size() << std::endl;
        }
        const auto tail_begin = Clock::now();
        const std::vector<float> canvas_tokens = tail.run_raw(pooled);
        const double tail_ms = milliseconds(tail_begin, Clock::now());
        if (const char* path = std::getenv("GEMMA4_DUMP_TAIL")) {
            std::ofstream out(path, std::ios::binary);
            out.write(reinterpret_cast<const char*>(canvas_tokens.data()),
                      static_cast<std::streamsize>(canvas_tokens.size() * sizeof(float)));
        }
        if (std::getenv("GEMMA4_DEBUG") != nullptr) {
            double sum = 0.0, mx = 0.0;
            for (float value : canvas_tokens) {
                sum += std::fabs(value);
                mx = std::max(mx, std::fabs(static_cast<double>(value)));
            }
            std::cout << "DEBUG tail_tokens=" << canvas_tokens.size()
                      << " mean_abs=" << (sum / std::max<size_t>(1, canvas_tokens.size()))
                      << " max_abs=" << mx
                      << " first=" << canvas_tokens[0] << "," << canvas_tokens[1] << std::endl;
        }
        const double vision_ms = milliseconds(vision_begin, Clock::now());
        const std::vector<float> selected = select_soft_tokens(canvas_tokens, canvas);

        if (const char* dump = std::getenv("GEMMA4_DUMP_TOKENS")) {
            std::ofstream out(dump, std::ios::binary);
            out.write(reinterpret_cast<const char*>(selected.data()),
                      static_cast<std::streamsize>(selected.size() * sizeof(float)));
            std::cout << "VISION dumped tokens to " << dump << std::endl;
        }
        std::cout << std::fixed << std::setprecision(3)
                  << "VISION platform=" << model_platform()
                  << " tokens=" << canvas.soft_tokens()
                  << " vision_ms=" << vision_ms
                  << " front_ms=" << front_ms
                  << " pool_ms=" << pool_ms
                  << " tail_ms=" << tail_ms
                  << " canvas_copy_ms=" << prepare_ms << std::endl;
        if (has_reference) {
            const auto reference = read_floats(
                argv[5], static_cast<size_t>(canvas.soft_tokens()) * kTextHidden);
            const Accuracy accuracy = compare(selected, reference);
            std::cout << "VISION_ACCURACY cosine=" << accuracy.cosine
                      << " mae=" << accuracy.mae
                      << " max_abs=" << accuracy.max_abs << std::endl;
        }
        if (max_tokens > 0) {
            HybridEngine engine(argv[2], argv[3]);
            engine.run(argv[4], "", "", max_tokens, std::cout);
        }
    } catch (const std::exception& error) {
        std::cerr << "ERROR " << error.what() << std::endl;
        return 1;
    }
    return 0;
}
