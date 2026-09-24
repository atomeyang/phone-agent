#include "neuron/api/RuntimeV2.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

constexpr size_t kLayers = 24;
constexpr size_t kSequence = 87;
constexpr size_t kVisualTokens = 64;
constexpr size_t kHidden = 1024;
constexpr size_t kCapacity = 256;
constexpr size_t kRotary = 64;
constexpr size_t kLinearChannels = 6144;
constexpr size_t kConvWidth = 4;
constexpr size_t kLinearHeads = 16;
constexpr size_t kLinearHeadDim = 128;
constexpr size_t kKvHeads = 2;
constexpr size_t kAttentionHeadDim = 256;
constexpr size_t kVocab = 248094;
constexpr int kEos = 248044;
constexpr size_t kVisionInputElements = 1 * 3 * 14 * 14112;
constexpr size_t kLogitsChunks[] = {61440, 61440, 61440, 61440, 2334};

constexpr bool is_full_attention(size_t layer) { return (layer + 1) % 4 == 0; }

double milliseconds(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

void check_neuron(int status, const std::string& operation) {
    if (status != 0) {
        throw std::runtime_error(operation + " failed with status " + std::to_string(status));
    }
}

uint16_t float_to_half(float value) {
    __fp16 converted = static_cast<__fp16>(value);
    uint16_t bits;
    std::memcpy(&bits, &converted, sizeof(bits));
    return bits;
}

float half_to_float(uint16_t bits) {
    __fp16 value;
    std::memcpy(&value, &bits, sizeof(value));
    return static_cast<float>(value);
}

class AlignedBuffer {
public:
    explicit AlignedBuffer(size_t size) : size_(size) {
        if (posix_memalign(&data_, 4096, size_) != 0) {
            throw std::bad_alloc();
        }
        std::memset(data_, 0, size_);
    }
    ~AlignedBuffer() { std::free(data_); }
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    void* data() { return data_; }
    const void* data() const { return data_; }
    size_t size() const { return size_; }

private:
    void* data_ = nullptr;
    size_t size_ = 0;
};

class MappedFile {
public:
    explicit MappedFile(const std::string& path) {
        fd_ = open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd_ < 0) {
            throw std::runtime_error("Cannot open " + path);
        }
        struct stat info {};
        if (fstat(fd_, &info) != 0) {
            throw std::runtime_error("Cannot stat " + path);
        }
        size_ = static_cast<size_t>(info.st_size);
        data_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (data_ == MAP_FAILED) {
            throw std::runtime_error("Cannot mmap " + path);
        }
    }
    ~MappedFile() {
        if (data_ != MAP_FAILED) {
            munmap(data_, size_);
        }
        if (fd_ >= 0) {
            close(fd_);
        }
    }
    const uint8_t* bytes() const { return static_cast<const uint8_t*>(data_); }
    size_t size() const { return size_; }

private:
    int fd_ = -1;
    void* data_ = MAP_FAILED;
    size_t size_ = 0;
};

class Runtime {
public:
    explicit Runtime(const std::string& path) : path_(path) {
        auto begin = Clock::now();
        check_neuron(NeuronRuntimeV2_create(path.c_str(), 1, &runtime_, 8), "create " + path);
        load_ms_ = milliseconds(begin, Clock::now());
        NeuronRuntimeV2_setQoS(
            runtime_, NEURONRUNTIME_SYSTEM_SCENARIO_HINT,
            NEURONRUNTIME_SYSTEM_SCENARIO_HINT_PERFORMANCE);
        NeuronRuntimeV2_setQoS(runtime_, NEURONRUNTIME_SYSTEM_CPU_BOOST, 100);
        NeuronRuntimeV2_setQoS(runtime_, NEURONRUNTIME_SYSTEM_DDR_BOOST, 100);

        size_t input_count = 0;
        size_t output_count = 0;
        check_neuron(NeuronRuntimeV2_getInputNumber(runtime_, &input_count), "get input count");
        check_neuron(NeuronRuntimeV2_getOutputNumber(runtime_, &output_count), "get output count");
        allocate(input_count, true, inputs_, input_descriptors_);
        allocate(output_count, false, outputs_, output_descriptors_);
    }
    ~Runtime() {
        if (runtime_ != nullptr) {
            NeuronRuntimeV2_release(runtime_);
        }
    }
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    double run() {
        SyncInferenceRequest request{input_descriptors_.data(), output_descriptors_.data()};
        auto begin = Clock::now();
        check_neuron(NeuronRuntimeV2_run(runtime_, request), "run " + path_);
        return milliseconds(begin, Clock::now());
    }
    AlignedBuffer& input(size_t index) { return *inputs_.at(index); }
    AlignedBuffer& output(size_t index) { return *outputs_.at(index); }
    size_t input_count() const { return inputs_.size(); }
    size_t output_count() const { return outputs_.size(); }
    double load_ms() const { return load_ms_; }

private:
    void allocate(
        size_t count, bool input,
        std::vector<std::unique_ptr<AlignedBuffer>>& buffers,
        std::vector<IOBuffer>& descriptors) {
        for (size_t index = 0; index < count; ++index) {
            size_t size = 0;
            int status = input
                ? NeuronRuntimeV2_getInputPaddedSize(runtime_, index, &size)
                : NeuronRuntimeV2_getOutputPaddedSize(runtime_, index, &size);
            check_neuron(status, input ? "get padded input size" : "get padded output size");
            buffers.push_back(std::make_unique<AlignedBuffer>(size));
            descriptors.emplace_back(buffers.back()->data(), size, -1, 0);
        }
    }

    std::string path_;
    void* runtime_ = nullptr;
    double load_ms_ = 0;
    std::vector<std::unique_ptr<AlignedBuffer>> inputs_;
    std::vector<std::unique_ptr<AlignedBuffer>> outputs_;
    std::vector<IOBuffer> input_descriptors_;
    std::vector<IOBuffer> output_descriptors_;
};

void require_size(const AlignedBuffer& buffer, size_t bytes, const std::string& label) {
    if (buffer.size() < bytes) {
        throw std::runtime_error(label + " buffer is too small");
    }
}

void copy_floats(AlignedBuffer& destination, const float* source, size_t count) {
    require_size(destination, count * sizeof(float), "destination");
    std::memcpy(destination.data(), source, count * sizeof(float));
}

void copy_floats(float* destination, const AlignedBuffer& source, size_t count) {
    require_size(source, count * sizeof(float), "source");
    std::memcpy(destination, source.data(), count * sizeof(float));
}

void copy_buffer(AlignedBuffer& destination, const AlignedBuffer& source, size_t count) {
    require_size(destination, count * sizeof(float), "destination");
    require_size(source, count * sizeof(float), "source");
    std::memcpy(destination.data(), source.data(), count * sizeof(float));
}

std::string layer_path(const std::string& root, const char* stage, size_t layer) {
    char index[3];
    std::snprintf(index, sizeof(index), "%02zu", layer);
    return root + "/" + stage + "/layer_" + index + "/model_bf16_mt6899.dla";
}

int greedy_token(Runtime& head) {
    int best_token = -1;
    float best_value = -std::numeric_limits<float>::infinity();
    size_t offset = 0;
    for (size_t chunk = 0; chunk < std::size(kLogitsChunks); ++chunk) {
        require_size(head.output(chunk), kLogitsChunks[chunk] * sizeof(float), "logits");
        const float* values = static_cast<const float*>(head.output(chunk).data());
        for (size_t index = 0; index < kLogitsChunks[chunk]; ++index) {
            if (values[index] > best_value) {
                best_value = values[index];
                best_token = static_cast<int>(offset + index);
            }
        }
        offset += kLogitsChunks[chunk];
    }
    if (offset != kVocab || best_token < 0) {
        throw std::runtime_error("Invalid LM-head outputs");
    }
    return best_token;
}

void write_tokens(const std::string& path, const std::vector<int>& tokens) {
    std::ofstream stream(path, std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("Cannot write " + path);
    }
    for (size_t index = 0; index < tokens.size(); ++index) {
        if (index != 0) {
            stream << ' ';
        }
        stream << tokens[index];
    }
    stream << '\n';
}

double percentile(std::vector<double> values, double fraction) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    size_t index = static_cast<size_t>(std::ceil(fraction * values.size())) - 1;
    return values[std::min(index, values.size() - 1)];
}

class Engine {
public:
    Engine(const std::string& root, const std::string& vision_path)
        : root_(root),
          prompt_(root + "/assets/prompt_embeddings_fp32.bin"),
          image_indices_(root + "/assets/image_indices_i32.bin"),
          embeddings_(root + "/assets/token_embeddings_fp16.bin"),
          rope_cos_(root + "/assets/rope_cos_fp16.bin"),
          rope_sin_(root + "/assets/rope_sin_fp16.bin"),
          vision_(vision_path),
          head_(root + "/head/model_bf16_mt6899.dla") {
        validate_assets();
        for (size_t layer = 0; layer < kLayers; ++layer) {
            prefill_.push_back(std::make_unique<Runtime>(layer_path(root, "prefill", layer)));
            decode_.push_back(std::make_unique<Runtime>(layer_path(root, "decode", layer)));
        }
        validate_graphs();
    }

    void run(const std::string& vision_input_path, const std::string& token_output, int max_tokens) {
        if (max_tokens <= 0 || max_tokens > static_cast<int>(kCapacity - kSequence + 1)) {
            throw std::runtime_error("max_tokens must be in [1, 171]");
        }
        std::vector<float> hidden(kSequence * kHidden);
        std::memcpy(hidden.data(), prompt_.bytes(), hidden.size() * sizeof(float));

        std::ifstream vision_input(vision_input_path, std::ios::binary);
        if (!vision_input) {
            throw std::runtime_error("Cannot open " + vision_input_path);
        }
        require_size(vision_.input(0), kVisionInputElements * sizeof(float), "vision input");
        vision_input.read(
            static_cast<char*>(vision_.input(0).data()),
            kVisionInputElements * sizeof(float));
        if (vision_input.gcount() != static_cast<std::streamsize>(kVisionInputElements * sizeof(float))) {
            throw std::runtime_error("Unexpected vision input byte count");
        }

        auto wall_begin = Clock::now();
        const double vision_ms = vision_.run();
        require_size(vision_.output(0), kVisualTokens * kHidden * sizeof(float), "vision output");
        const float* visual = static_cast<const float*>(vision_.output(0).data());
        const int32_t* image_positions = reinterpret_cast<const int32_t*>(image_indices_.bytes());
        for (size_t token = 0; token < kVisualTokens; ++token) {
            std::memcpy(
                hidden.data() + static_cast<size_t>(image_positions[token]) * kHidden,
                visual + token * kHidden,
                kHidden * sizeof(float));
        }

        double prefill_ms = 0.0;
        for (size_t layer = 0; layer < kLayers; ++layer) {
            Runtime& graph = *prefill_[layer];
            copy_floats(graph.input(0), hidden.data(), hidden.size());
            const double layer_ms = graph.run();
            prefill_ms += layer_ms;
            copy_floats(hidden.data(), graph.output(0), hidden.size());
            initialize_decode_state(layer, graph);
            std::cout << "PREFILL layer=" << layer << " ms=" << layer_ms << std::endl;
        }

        copy_floats(head_.input(0), hidden.data() + (kSequence - 1) * kHidden, kHidden);
        const double first_head_ms = head_.run();
        const int first_token = greedy_token(head_);
        auto first_token_wall = Clock::now();
        std::vector<int> tokens{first_token};
        std::cout << "TOKEN index=1 id=" << first_token << std::endl;
        std::cout << std::fixed << std::setprecision(6)
                  << "FIRST_TOKEN vision_ms=" << vision_ms
                  << " prefill_layers_ms=" << prefill_ms
                  << " head_ms=" << first_head_ms
                  << " npu_ms=" << (vision_ms + prefill_ms + first_head_ms)
                  << " wall_ms=" << milliseconds(wall_begin, first_token_wall)
                  << std::endl;

        std::vector<double> decode_npu_times;
        std::vector<double> decode_wall_times;
        for (size_t position = kSequence;
             tokens.size() < static_cast<size_t>(max_tokens) && tokens.back() != kEos;
             ++position) {
            auto decode_wall_begin = Clock::now();
            prepare_token_embedding(tokens.back(), hidden);
            double token_npu_ms = 0.0;
            for (size_t layer = 0; layer < kLayers; ++layer) {
                Runtime& graph = *decode_[layer];
                copy_floats(graph.input(0), hidden.data(), kHidden);
                if (is_full_attention(layer)) {
                    prepare_full_attention_inputs(graph, position);
                }
                token_npu_ms += graph.run();
                copy_floats(hidden.data(), graph.output(0), kHidden);
                update_decode_state(layer, graph, position);
            }
            copy_floats(head_.input(0), hidden.data(), kHidden);
            token_npu_ms += head_.run();
            const int token = greedy_token(head_);
            tokens.push_back(token);
            const double token_wall_ms = milliseconds(decode_wall_begin, Clock::now());
            decode_npu_times.push_back(token_npu_ms);
            decode_wall_times.push_back(token_wall_ms);
            std::cout << "TOKEN index=" << tokens.size() << " id=" << token
                      << " npu_ms=" << token_npu_ms
                      << " wall_ms=" << token_wall_ms << std::endl;
        }

        write_tokens(token_output, tokens);
        const double decode_npu_total =
            std::accumulate(decode_npu_times.begin(), decode_npu_times.end(), 0.0);
        const double decode_wall_total =
            std::accumulate(decode_wall_times.begin(), decode_wall_times.end(), 0.0);
        std::cout << "RESULT vision_ms=" << vision_ms << '\n'
                  << "RESULT prefill_layers_ms=" << prefill_ms << '\n'
                  << "RESULT prefill_head_ms=" << first_head_ms << '\n'
                  << "RESULT first_token_id=" << first_token << '\n'
                  << "RESULT ttft_npu_ms=" << (vision_ms + prefill_ms + first_head_ms) << '\n'
                  << "RESULT ttft_wall_ms=" << milliseconds(wall_begin, first_token_wall) << '\n'
                  << "RESULT decode_count=" << decode_npu_times.size() << '\n'
                  << "RESULT decode_npu_average_ms="
                  << (decode_npu_times.empty() ? 0.0 : decode_npu_total / decode_npu_times.size()) << '\n'
                  << "RESULT decode_wall_average_ms="
                  << (decode_wall_times.empty() ? 0.0 : decode_wall_total / decode_wall_times.size()) << '\n'
                  << "RESULT decode_wall_p50_ms=" << percentile(decode_wall_times, 0.50) << '\n'
                  << "RESULT decode_wall_p90_ms=" << percentile(decode_wall_times, 0.90) << '\n'
                  << "RESULT decode_tokens_per_second="
                  << (decode_wall_total == 0.0 ? 0.0 : decode_wall_times.size() * 1000.0 / decode_wall_total) << '\n'
                  << "RESULT generated_tokens=" << tokens.size() << '\n'
                  << "RESULT eos_reached=" << (tokens.back() == kEos ? 1 : 0) << '\n'
                  << "RESULT total_wall_ms=" << milliseconds(wall_begin, Clock::now())
                  << std::endl;
    }

    double total_load_ms() const {
        double total = vision_.load_ms() + head_.load_ms();
        for (const auto& runtime : prefill_) total += runtime->load_ms();
        for (const auto& runtime : decode_) total += runtime->load_ms();
        return total;
    }

private:
    void validate_assets() {
        if (prompt_.size() != kSequence * kHidden * sizeof(float) ||
            image_indices_.size() != kVisualTokens * sizeof(int32_t) ||
            embeddings_.size() != kVocab * kHidden * sizeof(uint16_t) ||
            rope_cos_.size() != kCapacity * kRotary * sizeof(uint16_t) ||
            rope_sin_.size() != kCapacity * kRotary * sizeof(uint16_t)) {
            throw std::runtime_error("Runtime asset shape mismatch");
        }
    }

    void validate_graphs() {
        if (vision_.input_count() != 1 || vision_.output_count() != 1 ||
            head_.input_count() != 1 || head_.output_count() != std::size(kLogitsChunks)) {
            throw std::runtime_error("Vision or LM-head I/O count mismatch");
        }
        for (size_t layer = 0; layer < kLayers; ++layer) {
            if (prefill_[layer]->input_count() != 1 || prefill_[layer]->output_count() != 3) {
                throw std::runtime_error("Prefill I/O count mismatch at layer " + std::to_string(layer));
            }
            const size_t decode_inputs = is_full_attention(layer) ? 6 : 3;
            if (decode_[layer]->input_count() != decode_inputs || decode_[layer]->output_count() != 3) {
                throw std::runtime_error("Decode I/O count mismatch at layer " + std::to_string(layer));
            }
        }
    }

    void initialize_decode_state(size_t layer, Runtime& prefill) {
        Runtime& decode = *decode_[layer];
        if (!is_full_attention(layer)) {
            copy_buffer(decode.input(1), prefill.output(1), kLinearChannels * kConvWidth);
            copy_buffer(
                decode.input(2), prefill.output(2),
                kLinearHeads * kLinearHeadDim * kLinearHeadDim);
            return;
        }
        const float* source_key = static_cast<const float*>(prefill.output(1).data());
        const float* source_value = static_cast<const float*>(prefill.output(2).data());
        float* destination_key = static_cast<float*>(decode.input(3).data());
        float* destination_value = static_cast<float*>(decode.input(4).data());
        require_size(prefill.output(1), kKvHeads * kSequence * kAttentionHeadDim * sizeof(float), "prefill key");
        require_size(prefill.output(2), kKvHeads * kSequence * kAttentionHeadDim * sizeof(float), "prefill value");
        require_size(decode.input(3), kKvHeads * kCapacity * kAttentionHeadDim * sizeof(float), "decode key");
        require_size(decode.input(4), kKvHeads * kCapacity * kAttentionHeadDim * sizeof(float), "decode value");
        for (size_t head = 0; head < kKvHeads; ++head) {
            std::memcpy(
                destination_key + head * kCapacity * kAttentionHeadDim,
                source_key + head * kSequence * kAttentionHeadDim,
                kSequence * kAttentionHeadDim * sizeof(float));
            std::memcpy(
                destination_value + head * kCapacity * kAttentionHeadDim,
                source_value + head * kSequence * kAttentionHeadDim,
                kSequence * kAttentionHeadDim * sizeof(float));
        }
    }

    void prepare_token_embedding(int token, std::vector<float>& hidden) {
        if (token < 0 || token >= static_cast<int>(kVocab)) {
            throw std::runtime_error("Token is outside vocabulary");
        }
        const uint16_t* row = reinterpret_cast<const uint16_t*>(embeddings_.bytes())
            + static_cast<size_t>(token) * kHidden;
        hidden.resize(kHidden);
        for (size_t index = 0; index < kHidden; ++index) {
            hidden[index] = half_to_float(row[index]);
        }
    }

    void prepare_full_attention_inputs(Runtime& graph, size_t position) {
        const uint16_t* cos = reinterpret_cast<const uint16_t*>(rope_cos_.bytes()) + position * kRotary;
        const uint16_t* sin = reinterpret_cast<const uint16_t*>(rope_sin_.bytes()) + position * kRotary;
        float* cos_input = static_cast<float*>(graph.input(1).data());
        float* sin_input = static_cast<float*>(graph.input(2).data());
        for (size_t index = 0; index < kRotary; ++index) {
            cos_input[index] = half_to_float(cos[index]);
            sin_input[index] = half_to_float(sin[index]);
        }
        float* mask = static_cast<float*>(graph.input(5).data());
        require_size(graph.input(5), (kCapacity + 1) * sizeof(float), "attention mask");
        std::fill(mask, mask + kCapacity + 1, -std::numeric_limits<float>::max());
        std::fill(mask, mask + position, 0.0f);
        mask[kCapacity] = 0.0f;
    }

    void update_decode_state(size_t layer, Runtime& graph, size_t position) {
        if (!is_full_attention(layer)) {
            copy_buffer(graph.input(1), graph.output(1), kLinearChannels * kConvWidth);
            copy_buffer(
                graph.input(2), graph.output(2),
                kLinearHeads * kLinearHeadDim * kLinearHeadDim);
            return;
        }
        const float* new_key = static_cast<const float*>(graph.output(1).data());
        const float* new_value = static_cast<const float*>(graph.output(2).data());
        float* cache_key = static_cast<float*>(graph.input(3).data());
        float* cache_value = static_cast<float*>(graph.input(4).data());
        for (size_t head = 0; head < kKvHeads; ++head) {
            std::memcpy(
                cache_key + (head * kCapacity + position) * kAttentionHeadDim,
                new_key + head * kAttentionHeadDim,
                kAttentionHeadDim * sizeof(float));
            std::memcpy(
                cache_value + (head * kCapacity + position) * kAttentionHeadDim,
                new_value + head * kAttentionHeadDim,
                kAttentionHeadDim * sizeof(float));
        }
    }

    std::string root_;
    MappedFile prompt_;
    MappedFile image_indices_;
    MappedFile embeddings_;
    MappedFile rope_cos_;
    MappedFile rope_sin_;
    Runtime vision_;
    Runtime head_;
    std::vector<std::unique_ptr<Runtime>> prefill_;
    std::vector<std::unique_ptr<Runtime>> decode_;
};

void usage(const char* program) {
    std::cerr << "Usage: " << program
              << " MODEL_ROOT VISION_DLA VISION_INPUT TOKENS_OUTPUT [MAX_TOKENS]\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5 || argc > 6) {
        usage(argv[0]);
        return 2;
    }
    try {
        const int max_tokens = argc == 6 ? std::stoi(argv[5]) : 32;
        auto load_begin = Clock::now();
        Engine engine(argv[1], argv[2]);
        std::cout << std::fixed << std::setprecision(6)
                  << "LOAD summed_runtime_ms=" << engine.total_load_ms()
                  << " wall_ms=" << milliseconds(load_begin, Clock::now()) << std::endl;
        engine.run(argv[3], argv[4], max_tokens);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR " << error.what() << std::endl;
        return 1;
    }
}
