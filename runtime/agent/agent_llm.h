// Inference driver interface.
//
// The agent core never touches MNN directly: it asks this interface for one
// generation at a time.  The runner implements it with the hybrid engine (NPU
// vision + MNN Q4 text backbone); the host tests implement it with a scripted
// stub, which is what makes the whole agent loop testable without a phone.
#ifndef GEMMA4_AGENT_LLM_H
#define GEMMA4_AGENT_LLM_H

#include "agent_json.h"
#include "agent_types.h"

#include <functional>
#include <string>
#include <vector>

namespace agent {

struct GenerationMetrics {
    int context_tokens = 0;      // all_seq_len after the call
    int prompt_tokens = 0;       // tokens prefilled by this call (the delta)
    int engine_prefill_tokens = 0;  // tokens the engine really pushed this call
    int generated_tokens = 0;
    int cache_delta_tokens = -1; // -1 when the model's prompt cache was not used
    bool cache_hit = false;
    bool full_prefill = false;
    bool cancelled = false;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    double ttft_ms = 0.0;
    double npu_ms = 0.0;
    double load_ms = 0.0;
    double decode_tokens_per_second = 0.0;
    std::string stop_reason;     // eos | max_tokens | cancelled | error
};

struct ImageInput {
    bool valid = false;
    std::string canvas_path;
    int tokens = 0;              // soft tokens the canvas produces
    int canvas_width = 0;
    int canvas_height = 0;
    std::string note;            // human readable description for the prompt/UI
    std::string sha256;
};

// One image that has to be spliced into a full prefill.  message_index points
// at the ChatMessages entry whose content carries the `<|image|>` placeholders.
struct ImageSplice {
    int message_index = -1;
    std::string canvas_path;
    int tokens = 0;
};

class AgentLlm {
public:
    virtual ~AgentLlm() = default;

    // Incremental generation: the driver keeps the KV cache and prefills only
    // the new suffix of `messages`.  `splices` describe the images referenced by
    // the messages; a driver uses them when it has to fall back to a full
    // prefill (for example the first call that carries an image, or when the
    // KV cache no longer matches the prompt).
    virtual std::string generate_delta(const ChatMessages& messages,
                                       const std::vector<ImageSplice>& splices, int max_tokens,
                                       const std::function<bool(int, const std::string&)>& on_token,
                                       GenerationMetrics* metrics, std::string* error) = 0;

    // Full prefill from a pristine KV cache, with optional NPU soft tokens
    // spliced into the requested messages.
    virtual std::string generate_full(const ChatMessages& messages,
                                      const std::vector<ImageSplice>& splices, int max_tokens,
                                      const std::function<bool(int, const std::string&)>& on_token,
                                      GenerationMetrics* metrics, std::string* error) = 0;

    // Drops the conversation state.  The next generation is a pristine full
    // prefill; on device this re-creates the MNN engine (~1.9 s).
    virtual bool reset_conversation(std::string* error) = 0;

    // The transcript keeps a cleaned version of the model's own answer (a stray
    // tool-protocol artefact or a template token removed).  The KV cache still
    // holds the raw text, so the engine's own prompt-cache comparison would see
    // a mismatch and re-prefill the whole conversation on the next turn.  This
    // hook lets the driver re-anchor the cached prompt text on what the runtime
    // actually stores.  Drivers without a KV cache ignore it.
    virtual void sync_transcript(const ChatMessages& messages) { (void)messages; }

    // Exact token count of the prompt that `generate_*` would build.
    virtual int count_tokens(const ChatMessages& messages) const = 0;

    // Reads the canvas header and computes the soft token count/fingerprint.
    virtual bool inspect_image(const std::string& canvas_path, ImageInput* image,
                               std::string* error) const = 0;

    virtual Json describe() const = 0;
};

}  // namespace agent

#endif  // GEMMA4_AGENT_LLM_H
