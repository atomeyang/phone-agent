#include "llm/llm.hpp"

#include <MNN/expr/Expr.hpp>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using MNN::Express::VARP;
using MNN::Transformer::Llm;

constexpr int kHiddenSize = 1024;
constexpr int kImageTokenId = 248056;

std::vector<float> read_visual_embeddings(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open visual embeddings: " + path);
    }
    input.seekg(0, std::ios::end);
    const auto bytes = input.tellg();
    input.seekg(0, std::ios::beg);
    if (bytes <= 0 || bytes % (kHiddenSize * sizeof(float)) != 0) {
        throw std::runtime_error("invalid visual embedding byte count");
    }
    std::vector<float> values(
        static_cast<size_t>(bytes) / sizeof(float));
    input.read(reinterpret_cast<char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (input.gcount() != static_cast<std::streamsize>(values.size() * sizeof(float))) {
        throw std::runtime_error("unexpected visual embedding byte count");
    }
    return values;
}

std::string read_text(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("cannot open prompt: " + path);
    }
    std::ostringstream text;
    text << input.rdbuf();
    std::string result = text.str();
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r')) {
        result.pop_back();
    }
    return result;
}

std::string build_prompt(const std::string& user_prompt, int vision_tokens) {
    std::string prompt = "<|im_start|>user\n<image_id>0</image_id><image>";
    for (int i = 0; i < vision_tokens; ++i) {
        prompt += "<|image_pad|>";
    }
    prompt += "</image>\n" + user_prompt + "<|im_end|>\n";
    prompt += "<|im_start|>assistant\n<think>\n\n</think>\n\n";
    return prompt;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4 || argc > 5) {
        std::cerr << "Usage: " << argv[0]
                  << " CONFIG VISUAL_EMBEDDINGS PROMPT [MAX_TOKENS]\n";
        return 2;
    }

    try {
        const int max_tokens = argc == 5 ? std::stoi(argv[4]) : 256;
        const auto visual_embeddings = read_visual_embeddings(argv[2]);
        const int vision_tokens = static_cast<int>(
            visual_embeddings.size() / kHiddenSize);
        const std::string user_prompt = read_text(argv[3]);

        std::unique_ptr<Llm> llm(Llm::createLLM(argv[1]));
        if (!llm || !llm->set_config(
                R"({"backend_type":"cpu","thread_num":4,"precision":"low","memory":"low","power":"high","async":false,"sampler_type":"greedy"})") ||
            !llm->load()) {
            throw std::runtime_error("cannot load MNN LLM");
        }

        const auto input_ids = llm->tokenizer_encode(
            build_prompt(user_prompt, vision_tokens));
        const int image_count = static_cast<int>(
            std::count(input_ids.begin(), input_ids.end(), kImageTokenId));
        if (image_count != vision_tokens) {
            throw std::runtime_error("image-token count mismatch: " +
                                     std::to_string(image_count));
        }
        std::cout << "RESULT input_ids=";
        for (size_t index = 0; index < input_ids.size(); ++index) {
            std::cout << (index == 0 ? "" : ",") << input_ids[index];
        }
        std::cout << '\n';

        VARP input_embeddings = llm->embedding(input_ids);
        if (input_embeddings == nullptr || !input_embeddings->getInfo() ||
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

        std::ostringstream output;
        llm->response(input_embeddings, &output, nullptr, max_tokens);
        std::cout << "RESULT prompt_tokens=" << input_ids.size() << '\n'
                  << "RESULT image_tokens=" << image_count << '\n'
                  << "RESULT generated_tokens=" << llm->getContext()->gen_seq_len << '\n'
                  << "RESULT text_begin\n"
                  << output.str() << "\nRESULT text_end\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR " << error.what() << '\n';
        return 1;
    }
}
