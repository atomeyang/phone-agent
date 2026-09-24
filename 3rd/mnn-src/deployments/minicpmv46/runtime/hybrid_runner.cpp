#include "llm/llm.hpp"

#include <MNN/Interpreter.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/Module.hpp>
#include "neuron/api/RuntimeV2.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <map>
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
using MNN::Express::Module;
using MNN::Express::VARP;
using MNN::Transformer::Llm;

constexpr int kStaticVisionHeight = 504;
constexpr int kStaticVisionWidth = 392;
constexpr int kStaticVisionTokens = 63;
constexpr int kVisionSizeDivisor = 56;
constexpr int kHiddenSize = 1024;
constexpr int kImageTokenId = 248056;
constexpr int32_t kTileFileMagic = 0x4d435034;
constexpr int32_t kTileFileVersion = 1;

double milliseconds(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

void check_neuron(int status, const std::string& operation) {
    if (status != 0) {
        throw std::runtime_error(operation + " failed with status " + std::to_string(status));
    }
}

void pin_to_performance_cores() {
    cpu_set_t cores;
    CPU_ZERO(&cores);
    for (int cpu = 4; cpu <= 7; ++cpu) {
        CPU_SET(cpu, &cores);
    }
    sched_setaffinity(0, sizeof(cores), &cores);
}

std::vector<float> read_floats(const std::string& path, size_t expected) {
    std::vector<float> values(expected);
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open " + path);
    }
    stream.read(reinterpret_cast<char*>(values.data()),
                static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (stream.gcount() != static_cast<std::streamsize>(values.size() * sizeof(float))) {
        throw std::runtime_error("unexpected byte count in " + path);
    }
    return values;
}

struct Accuracy {
    double cosine = 0.0;
    double mean_absolute_error = 0.0;
    double max_absolute_error = 0.0;
    bool finite = true;
};

Accuracy compare(const std::vector<float>& actual, const std::vector<float>& reference) {
    if (actual.size() != reference.size()) {
        throw std::runtime_error("accuracy vector size mismatch");
    }
    double dot = 0.0;
    double norm_actual = 0.0;
    double norm_reference = 0.0;
    double absolute_sum = 0.0;
    double absolute_max = 0.0;
    bool finite = true;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double a = actual[i];
        const double b = reference[i];
        finite = finite && std::isfinite(a) && std::isfinite(b);
        dot += a * b;
        norm_actual += a * a;
        norm_reference += b * b;
        const double error = std::abs(a - b);
        absolute_sum += error;
        absolute_max = std::max(absolute_max, error);
    }
    Accuracy result;
    result.cosine = dot / std::sqrt(norm_actual * norm_reference);
    result.mean_absolute_error = absolute_sum / actual.size();
    result.max_absolute_error = absolute_max;
    result.finite = finite;
    return result;
}

class VisionBackend {
public:
    virtual ~VisionBackend() = default;
    virtual std::vector<float> run(
        const std::vector<float>& input, int height, int width) = 0;
    virtual const char* name() const = 0;
};

class NpuVisionProfile {
public:
    NpuVisionProfile(const std::string& path, int height, int width)
        : height_(height), width_(width) {
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
        if (input_count != 1 || output_count != 1) {
            throw std::runtime_error("NPU vision profile must have one input and one output");
        }
    }

    ~NpuVisionProfile() {
        std::free(input_);
        std::free(output_);
        if (runtime_ != nullptr) {
            NeuronRuntimeV2_release(runtime_);
        }
    }

    std::vector<float> run(const std::vector<float>& input) {
        if (input.size() != static_cast<size_t>(3) * height_ * width_) {
            throw std::runtime_error("vision profile input size mismatch");
        }
        size_t required_input_bytes = 0;
        size_t required_output_bytes = 0;
        check_neuron(
            NeuronRuntimeV2_getInputPaddedSize(runtime_, 0, &required_input_bytes),
            "get dynamic NPU input size");
        check_neuron(
            NeuronRuntimeV2_getOutputPaddedSize(runtime_, 0, &required_output_bytes),
            "get dynamic NPU output size");
        const int visual_tokens =
            (height_ / kVisionSizeDivisor) * (width_ / kVisionSizeDivisor);
        if (required_input_bytes < input.size() * sizeof(float) ||
            required_output_bytes <
                static_cast<size_t>(visual_tokens) * kHiddenSize * sizeof(float)) {
            throw std::runtime_error("static NPU profile buffer shape mismatch");
        }
        resize_buffer(input_, input_bytes_, required_input_bytes);
        resize_buffer(output_, output_bytes_, required_output_bytes);
        std::memset(input_, 0, input_bytes_);
        std::memset(output_, 0, output_bytes_);
        std::memcpy(input_, input.data(), input.size() * sizeof(float));
        IOBuffer input_buffer(input_, input_bytes_, -1, 0);
        IOBuffer output_buffer(output_, output_bytes_, -1, 0);
        SyncInferenceRequest request{&input_buffer, &output_buffer};
        check_neuron(NeuronRuntimeV2_run(runtime_, request), "run vision DLA");
        const auto* output = static_cast<const float*>(output_);
        return std::vector<float>(
            output, output + static_cast<size_t>(visual_tokens) * kHiddenSize);
    }

private:
    static void resize_buffer(void*& buffer, size_t& capacity, size_t required) {
        if (capacity >= required) {
            return;
        }
        std::free(buffer);
        buffer = nullptr;
        capacity = 0;
        if (posix_memalign(&buffer, 4096, required) != 0) {
            throw std::bad_alloc();
        }
        capacity = required;
    }

    void* runtime_ = nullptr;
    void* input_ = nullptr;
    void* output_ = nullptr;
    size_t input_bytes_ = 0;
    size_t output_bytes_ = 0;
    int height_ = 0;
    int width_ = 0;
};

class NpuVisionBackend final : public VisionBackend {
public:
    explicit NpuVisionBackend(std::string model_directory)
        : model_directory_(std::move(model_directory)) {}

    std::vector<float> run(
        const std::vector<float>& input, int height, int width) override {
        if (height <= 0 || width <= 0 ||
            height % kVisionSizeDivisor != 0 || width % kVisionSizeDivisor != 0 ||
            input.size() != static_cast<size_t>(3) * height * width) {
            throw std::runtime_error("invalid vision tile shape");
        }
        const auto key = std::make_pair(height, width);
        auto profile = profiles_.find(key);
        if (profile == profiles_.end()) {
            const std::string path = model_directory_ + "/visual_" +
                std::to_string(height) + "x" + std::to_string(width) +
                "_fp16_mt6899.dla";
            if (access(path.c_str(), R_OK) != 0) {
                throw std::runtime_error(
                    "no exact NPU vision profile for " +
                    std::to_string(height) + "x" + std::to_string(width));
            }
            profile = profiles_.emplace(
                key, std::make_unique<NpuVisionProfile>(path, height, width)).first;
        }
        return profile->second->run(input);
    }

    const char* name() const override { return "npu"; }

private:
    std::string model_directory_;
    std::map<std::pair<int, int>, std::unique_ptr<NpuVisionProfile>> profiles_;
};

class MnnVisionBackend final : public VisionBackend {
public:
    explicit MnnVisionBackend(const std::string& path) {
        MNN::ScheduleConfig schedule;
        MNN::BackendConfig backend;
        schedule.type = MNN_FORWARD_CPU;
        schedule.numThread = 4;
        backend.precision = MNN::BackendConfig::Precision_Low;
        backend.power = MNN::BackendConfig::Power_High;
        backend.memory = MNN::BackendConfig::Memory_Low;
        schedule.backendConfig = &backend;
        runtime_.reset(
            MNN::Express::Executor::RuntimeManager::createRuntimeManager(schedule),
            MNN::Express::Executor::RuntimeManager::destroy);
        if (!runtime_) {
            throw std::runtime_error("cannot create MNN vision runtime");
        }
        MNN::Express::Module::Config module_config;
        module_config.shapeMutable = false;
        module_config.rearrange = true;
        module_.reset(Module::load({}, {}, path.c_str(), runtime_, &module_config),
                      Module::destroy);
        if (!module_) {
            throw std::runtime_error("cannot load MNN vision graph");
        }
    }

    std::vector<float> run(
        const std::vector<float>& input, int height, int width) override {
        if (height != kStaticVisionHeight || width != kStaticVisionWidth ||
            input.size() != static_cast<size_t>(3) * height * width) {
            throw std::runtime_error("static MNN vision only supports 504x392");
        }
        auto input_var = MNN::Express::_Input(
            {1, 3, 14, 14112}, MNN::Express::NCHW, halide_type_of<float>());
        std::memcpy(input_var->writeMap<float>(), input.data(),
                    input.size() * sizeof(float));
        auto outputs = module_->onForward({input_var});
        if (outputs.size() != 1 || outputs[0] == nullptr ||
            outputs[0]->getInfo() == nullptr ||
            outputs[0]->getInfo()->size != kStaticVisionTokens * kHiddenSize) {
            throw std::runtime_error("MNN vision output shape mismatch");
        }
        const float* output = outputs[0]->readMap<float>();
        if (output == nullptr) {
            throw std::runtime_error("cannot read MNN vision output");
        }
        return std::vector<float>(
            output, output + kStaticVisionTokens * kHiddenSize);
    }

    const char* name() const override { return "mnn"; }

private:
    std::shared_ptr<MNN::Express::Executor::RuntimeManager> runtime_;
    std::shared_ptr<Module> module_;
};

constexpr const char* kDefaultUserPrompt = "Please describe this image.";

struct VisionTile {
    int height = 0;
    int width = 0;
    std::vector<float> pixels;

    int token_count() const {
        return (height / kVisionSizeDivisor) * (width / kVisionSizeDivisor);
    }
};

struct VisionInput {
    int grid_rows = 0;
    int grid_columns = 0;
    std::vector<VisionTile> tiles;

    int token_count() const {
        return std::accumulate(
            tiles.begin(), tiles.end(), 0,
            [](int total, const VisionTile& tile) {
                return total + tile.token_count();
            });
    }
};

VisionInput read_vision_input(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open " + path);
    }
    auto read_int = [&]() {
        int32_t value = 0;
        stream.read(reinterpret_cast<char*>(&value), sizeof(value));
        if (!stream) {
            throw std::runtime_error("truncated vision input header");
        }
        return value;
    };

    const int32_t magic = read_int();
    const int32_t version = read_int();
    VisionInput input;
    input.grid_rows = read_int();
    input.grid_columns = read_int();
    const int32_t tile_count = read_int();
    if (magic != kTileFileMagic || version != kTileFileVersion ||
        tile_count <= 0 || tile_count > 10 || input.grid_rows < 0 ||
        input.grid_columns < 0 ||
        (tile_count != 1 &&
         tile_count != 1 + input.grid_rows * input.grid_columns)) {
        throw std::runtime_error("invalid vision input metadata");
    }

    input.tiles.reserve(tile_count);
    for (int tile_index = 0; tile_index < tile_count; ++tile_index) {
        VisionTile tile;
        tile.height = read_int();
        tile.width = read_int();
        if (tile.height <= 0 || tile.width <= 0 ||
            tile.height > 1120 || tile.width > 1120 ||
            tile.height % kVisionSizeDivisor != 0 ||
            tile.width % kVisionSizeDivisor != 0) {
            throw std::runtime_error("invalid vision tile dimensions");
        }
        tile.pixels.resize(static_cast<size_t>(3) * tile.height * tile.width);
        stream.read(
            reinterpret_cast<char*>(tile.pixels.data()),
            static_cast<std::streamsize>(tile.pixels.size() * sizeof(float)));
        if (!stream) {
            throw std::runtime_error("truncated vision tile data");
        }
        input.tiles.push_back(std::move(tile));
    }
    return input;
}

void append_image_tokens(std::string& prompt, int count) {
    for (int index = 0; index < count; ++index) {
        prompt += "<|image_pad|>";
    }
}

std::string build_prompt(
    const VisionInput& vision, const std::string& user_prompt = kDefaultUserPrompt) {
    if (vision.tiles.empty()) {
        throw std::runtime_error("vision prompt requires at least one tile");
    }
    std::string prompt = "<|im_start|>user\n<image_id>0</image_id><image>";
    append_image_tokens(prompt, vision.tiles[0].token_count());
    prompt += "</image>";
    size_t tile_index = 1;
    for (int row = 0; row < vision.grid_rows; ++row) {
        for (int column = 0; column < vision.grid_columns; ++column) {
            prompt += "<slice>";
            append_image_tokens(prompt, vision.tiles.at(tile_index).token_count());
            prompt += "</slice>";
            ++tile_index;
        }
        if (row + 1 < vision.grid_rows) {
            prompt += "\n";
        }
    }
    prompt += "\n" + user_prompt + "<|im_end|>\n";
    prompt += "<|im_start|>assistant\n<think>\n\n</think>\n\n";
    return prompt;
}

std::string read_text(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        throw std::runtime_error("cannot open " + path);
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
        encoded.push_back(offset + 1 < value.size()
            ? kAlphabet[(packed >> 6) & 0x3f] : '=');
        encoded.push_back(offset + 2 < value.size()
            ? kAlphabet[packed & 0x3f] : '=');
    }
    return encoded;
}

class HybridInferenceEngine {
public:
    HybridInferenceEngine(const std::string& config_path,
                          const std::string& vision_path)
        : vision_(vision_path), llm_(Llm::createLLM(config_path)) {
        if (!llm_) {
            throw std::runtime_error("cannot create MNN LLM");
        }
        if (!llm_->set_config(
                R"({"backend_type":"cpu","thread_num":4,"precision":"low","memory":"low","power":"high","async":false,"sampler_type":"greedy"})")) {
            throw std::runtime_error("cannot configure MNN LLM");
        }
        const auto load_begin = Clock::now();
        if (!llm_->load()) {
            throw std::runtime_error("cannot load MNN LLM");
        }
        llm_load_ms_ = milliseconds(load_begin, Clock::now());
        llm_->tuning(MNN::Transformer::OP_ENCODER_NUMBER,
                     {1, 5, 10, 20, 30, 50, 100});

    }

    void run(const std::string& input_path, const std::string& prompt_path,
             const std::string& stop_path, int max_tokens, std::ostream& output) {
        consume_file(stop_path);
        llm_->reset();

        const VisionInput vision_input = read_vision_input(input_path);
        const auto inference_begin = Clock::now();
        const auto prompt_begin = Clock::now();
        const std::string user_prompt = read_text(prompt_path);
        const auto input_ids = llm_->tokenizer_encode(
            build_prompt(vision_input, user_prompt));
        const int image_count = static_cast<int>(
            std::count(input_ids.begin(), input_ids.end(), kImageTokenId));
        if (image_count != vision_input.token_count()) {
            throw std::runtime_error(
                "prompt token contract mismatch: total=" +
                std::to_string(input_ids.size()) + ", image=" +
                std::to_string(image_count));
        }
        const double prompt_prepare_ms = milliseconds(prompt_begin, Clock::now());

        const auto vision_begin = Clock::now();
        std::vector<float> visual_embeddings;
        visual_embeddings.reserve(
            static_cast<size_t>(image_count) * kHiddenSize);
        for (const VisionTile& tile : vision_input.tiles) {
            auto tile_embeddings = vision_.run(
                tile.pixels, tile.height, tile.width);
            visual_embeddings.insert(
                visual_embeddings.end(),
                tile_embeddings.begin(), tile_embeddings.end());
        }
        const double vision_ms = milliseconds(vision_begin, Clock::now());

        const auto inject_begin = Clock::now();
        VARP input_embeddings = llm_->embedding(input_ids);
        if (input_embeddings == nullptr || input_embeddings->getInfo() == nullptr ||
            input_embeddings->getInfo()->size !=
                static_cast<int>(input_ids.size()) * kHiddenSize) {
            throw std::runtime_error("LLM embedding shape mismatch");
        }
        float* embedding_data = input_embeddings->writeMap<float>();
        int image_index = 0;
        for (size_t token_index = 0; token_index < input_ids.size(); ++token_index) {
            if (input_ids[token_index] != kImageTokenId) {
                continue;
            }
            std::memcpy(embedding_data + token_index * kHiddenSize,
                        visual_embeddings.data() + image_index * kHiddenSize,
                        kHiddenSize * sizeof(float));
            ++image_index;
        }
        const double inject_ms = milliseconds(inject_begin, Clock::now());

        bool has_first_token = false;
        bool stopped = false;
        Clock::time_point first_token_time;
        llm_->setTokenCallback([&](int token, int index) {
            if (consume_file(stop_path)) {
                stopped = true;
                return false;
            }
            if (!has_first_token) {
                first_token_time = Clock::now();
                has_first_token = true;
                const double ttft_ms = milliseconds(inference_begin, first_token_time);
                output << "FIRST_TOKEN id=" << token
                       << " ttft_ms=" << ttft_ms
                       << " wall_ms=" << ttft_ms << std::endl;
            }
            output << "TOKEN index=" << index
                   << " id=" << token
                   << " text_b64=" << base64_encode(llm_->tokenizer_decode(token))
                   << std::endl;
            return true;
        });

        llm_->response(input_embeddings, nullptr, nullptr, max_tokens);
        const auto inference_end = Clock::now();
        llm_->setTokenCallback({});

        const auto* context = llm_->getContext();
        const double prefill_ms = context->prefill_us / 1000.0;
        const double decode_ms = context->decode_us / 1000.0;
        const int decode_intervals = std::max(0, context->gen_seq_len - 1);
        output << std::fixed << std::setprecision(3)
               << "RESULT prompt_prepare_ms=" << prompt_prepare_ms << '\n'
               << "RESULT prompt_tokens=" << input_ids.size() << '\n'
               << "RESULT image_tokens=" << image_count << '\n'
               << "RESULT vision_tiles=" << vision_input.tiles.size() << '\n'
               << "RESULT vision_grid_rows=" << vision_input.grid_rows << '\n'
               << "RESULT vision_grid_columns=" << vision_input.grid_columns << '\n'
               << "RESULT npu_vision_ms=" << vision_ms << '\n'
               << "RESULT embedding_inject_ms=" << inject_ms << '\n'
               << "RESULT llm_prefill_ms=" << prefill_ms << '\n'
               << "RESULT generated_tokens=" << context->gen_seq_len << '\n'
               << "RESULT llm_decode_ms=" << decode_ms << '\n'
               << "RESULT decode_ms_per_token="
               << (decode_intervals > 0 ? decode_ms / decode_intervals : 0.0) << '\n'
               << "RESULT decode_tokens_per_second="
               << (decode_ms > 0.0 ? decode_intervals * 1000.0 / decode_ms : 0.0) << '\n'
               << "RESULT inference_wall_ms="
               << milliseconds(inference_begin, inference_end) << '\n'
               << "RESULT eos_reached=" << llm_->stoped() << '\n'
               << "RESULT stopped=" << stopped << '\n'
               << "RESULT request_complete=1" << std::endl;
    }

    double llm_load_ms() const { return llm_load_ms_; }

private:
    NpuVisionBackend vision_;
    std::unique_ptr<Llm> llm_;
    double llm_load_ms_ = 0.0;
};

void print_usage(const char* program) {
    std::cerr << "Usage: " << program
              << " npu|mnn CONFIG VISION_MODEL VISION_INPUT REFERENCE [MAX_TOKENS]\n"
              << "   or: " << program
              << " --service CONFIG VISION_DLA VISION_INPUT PROMPT REQUEST STOP LOG READY "
                 "[MAX_TOKENS]\n";
}

}  // namespace

int main(int argc, char** argv) {
    const bool service_mode = argc > 1 && std::string(argv[1]) == "--service";
    if ((!service_mode && (argc < 6 || argc > 7)) ||
        (service_mode && (argc < 10 || argc > 11))) {
        print_usage(argv[0]);
        return 2;
    }

    try {
        pin_to_performance_cores();
        if (service_mode) {
            const auto startup_begin = Clock::now();
            HybridInferenceEngine engine(argv[2], argv[3]);
            const double startup_ms = milliseconds(startup_begin, Clock::now());
            const std::string input_path = argv[4];
            const std::string prompt_path = argv[5];
            const std::string request_path = argv[6];
            const std::string stop_path = argv[7];
            const std::string log_path = argv[8];
            const std::string ready_path = argv[9];
            const int max_tokens = argc == 11 ? std::stoi(argv[10]) : 1024;
            write_ready_file(
                ready_path,
                "READY startup_ms=" + std::to_string(startup_ms) +
                    " llm_load_ms=" + std::to_string(engine.llm_load_ms()));

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
                    log << "ERROR " << error.what() << std::endl;
                }
            }
        }

        const std::string mode = argv[1];
        const int max_tokens = argc == 7 ? std::stoi(argv[6]) : 64;
        const auto wall_begin = Clock::now();

        std::unique_ptr<VisionBackend> vision;
        const auto vision_load_begin = Clock::now();
        if (mode == "npu") {
            vision = std::make_unique<NpuVisionBackend>(argv[3]);
        } else if (mode == "mnn") {
            vision = std::make_unique<MnnVisionBackend>(argv[3]);
        } else {
            throw std::runtime_error("mode must be npu or mnn");
        }
        const double vision_load_ms = milliseconds(vision_load_begin, Clock::now());

        std::unique_ptr<Llm> llm(Llm::createLLM(argv[2]));
        if (!llm) {
            throw std::runtime_error("cannot create MNN LLM");
        }
        if (!llm->set_config(
                R"({"backend_type":"cpu","thread_num":4,"precision":"low","memory":"low","power":"high","async":false,"sampler_type":"greedy"})")) {
            throw std::runtime_error("cannot configure MNN LLM");
        }
        const auto llm_load_begin = Clock::now();
        if (!llm->load()) {
            throw std::runtime_error("cannot load MNN LLM");
        }
        const double llm_load_ms = milliseconds(llm_load_begin, Clock::now());
        llm->tuning(MNN::Transformer::OP_ENCODER_NUMBER,
                    {1, 5, 10, 20, 30, 50, 100});

        const auto input = read_floats(
            argv[4], static_cast<size_t>(3) * kStaticVisionHeight * kStaticVisionWidth);
        const auto reference = read_floats(
            argv[5], kStaticVisionTokens * kHiddenSize);

        const auto inference_begin = Clock::now();
        const auto vision_begin = Clock::now();
        const auto visual_embeddings = vision->run(
            input, kStaticVisionHeight, kStaticVisionWidth);
        const double vision_ms = milliseconds(vision_begin, Clock::now());
        const Accuracy accuracy = compare(visual_embeddings, reference);

        VisionInput static_vision;
        static_vision.tiles.push_back(
            VisionTile{kStaticVisionHeight, kStaticVisionWidth, {}});
        const std::string prompt = build_prompt(static_vision);
        const auto input_ids = llm->tokenizer_encode(prompt);
        const int image_count = static_cast<int>(
            std::count(input_ids.begin(), input_ids.end(), kImageTokenId));
        if (input_ids.size() != 86 || image_count != kStaticVisionTokens) {
            throw std::runtime_error(
                "prompt token contract mismatch: total=" +
                std::to_string(input_ids.size()) + ", image=" +
                std::to_string(image_count));
        }

        const auto inject_begin = Clock::now();
        VARP input_embeddings = llm->embedding(input_ids);
        if (input_embeddings == nullptr || input_embeddings->getInfo() == nullptr ||
            input_embeddings->getInfo()->size !=
                static_cast<int>(input_ids.size()) * kHiddenSize) {
            throw std::runtime_error("LLM embedding shape mismatch");
        }
        float* embedding_data = input_embeddings->writeMap<float>();
        int image_index = 0;
        for (size_t token_index = 0; token_index < input_ids.size(); ++token_index) {
            if (input_ids[token_index] != kImageTokenId) {
                continue;
            }
            std::memcpy(embedding_data + token_index * kHiddenSize,
                        visual_embeddings.data() + image_index * kHiddenSize,
                        kHiddenSize * sizeof(float));
            ++image_index;
        }
        const double inject_ms = milliseconds(inject_begin, Clock::now());

        std::vector<int> output_ids;
        Clock::time_point first_token_time;
        bool has_first_token = false;
        llm->setTokenCallback([&](int token, int index) {
            if (!has_first_token) {
                first_token_time = Clock::now();
                has_first_token = true;
            }
            output_ids.push_back(token);
            std::cout << "TOKEN index=" << index << " id=" << token << std::endl;
            return true;
        });
        std::ostringstream generated_text;
        llm->response(input_embeddings, &generated_text, nullptr, max_tokens);
        const auto inference_end = Clock::now();
        llm->setTokenCallback({});

        const auto* context = llm->getContext();
        const double prefill_ms = context->prefill_us / 1000.0;
        const double decode_ms = context->decode_us / 1000.0;
        const double sample_ms = context->sample_us / 1000.0;
        const double ttft_ms = has_first_token
            ? milliseconds(inference_begin, first_token_time)
            : -1.0;
        const int decode_intervals = std::max(0, context->gen_seq_len - 1);

        std::cout << std::fixed << std::setprecision(6)
                  << "RESULT backend=" << vision->name() << '\n'
                  << "RESULT vision_load_ms=" << vision_load_ms << '\n'
                  << "RESULT llm_load_ms=" << llm_load_ms << '\n'
                  << "RESULT prompt_tokens=" << input_ids.size() << '\n'
                  << "RESULT image_tokens=" << image_count << '\n'
                  << "RESULT visual_cosine=" << accuracy.cosine << '\n'
                  << "RESULT visual_mae=" << accuracy.mean_absolute_error << '\n'
                  << "RESULT visual_max_error=" << accuracy.max_absolute_error << '\n'
                  << "RESULT visual_all_finite=" << (accuracy.finite ? 1 : 0) << '\n'
                  << "RESULT vision_ms=" << vision_ms << '\n'
                  << "RESULT embedding_inject_ms=" << inject_ms << '\n'
                  << "RESULT llm_prefill_ms=" << prefill_ms << '\n'
                  << "RESULT sample_ms=" << sample_ms << '\n'
                  << "RESULT ttft_ms=" << ttft_ms << '\n'
                  << "RESULT generated_tokens=" << context->gen_seq_len << '\n'
                  << "RESULT llm_decode_ms=" << decode_ms << '\n'
                  << "RESULT decode_ms_per_token="
                  << (decode_intervals > 0 ? decode_ms / decode_intervals : 0.0) << '\n'
                  << "RESULT decode_tokens_per_second="
                  << (decode_ms > 0.0 ? decode_intervals * 1000.0 / decode_ms : 0.0) << '\n'
                  << "RESULT inference_wall_ms="
                  << milliseconds(inference_begin, inference_end) << '\n'
                  << "RESULT process_wall_ms="
                  << milliseconds(wall_begin, inference_end) << '\n'
                  << "RESULT token_ids=";
        for (size_t i = 0; i < output_ids.size(); ++i) {
            std::cout << (i == 0 ? "" : ",") << output_ids[i];
        }
        std::cout << '\n' << "RESULT text_begin\n"
                  << generated_text.str() << "\nRESULT text_end\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR " << error.what() << std::endl;
        return 1;
    }
}
