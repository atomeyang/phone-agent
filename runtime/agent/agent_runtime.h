// The on-device agent loop.
//
// One user turn is: recall memory -> build the context under the token budget
// -> plan -> (tool call -> observation)* -> answer -> write memory -> persist the
// transcript.  Everything in that sentence happens in native code; the app only
// renders the events this class emits.
#ifndef GEMMA4_AGENT_RUNTIME_H
#define GEMMA4_AGENT_RUNTIME_H

#include "agent_config.h"
#include "agent_json.h"
#include "agent_intent.h"
#include "agent_llm.h"
#include "agent_memory.h"
#include "agent_planner.h"
#include "agent_session.h"
#include "agent_tools.h"
#include "agent_types.h"

#include <atomic>
#include <functional>
#include <string>
#include <vector>

namespace agent {

class AgentRuntime {
public:
    using EventSink = std::function<void(const Json& event)>;
    using CancelCheck = std::function<bool()>;

    struct Options {
        AgentConfig config;
        std::string root;
        Json device = Json::object();
        bool model_memory_extraction = false;   // extra model pass after a turn
        int summary_max_bullets = 8;
        int image_cache_turns = 3;              // cached canvases kept per session
    };

    AgentRuntime(Options options, AgentLlm& llm, EventSink sink, CancelCheck cancel);

    bool initialize(std::string* error);

    // Dispatches one request document; the result is also what the service
    // writes to agent/response.json.
    Json handle(const Json& request);

    Json capabilities() const;
    Json status() const;
    const AgentConfig& config() const { return options_.config; }

private:
    Json run_turn(const Json& request);

    // History (system prompt, rolling summary, retained turns) plus the given
    // tail, with the image splices the tail needs.
    ChatMessages render_history(const ChatMessages& tail,
                                std::vector<ImageSplice>* splices,
                                const std::string& current_canvas, int current_tokens);
    // Two-tier history: the newest `image_window_turns`/`tool_detail_window_turns`
    // turns keep their full payload, older ones are folded to text.  Returns true
    // when something changed (the caller then re-prefills from a pristine KV).
    bool degraded_history_ = false;
    std::vector<ScoredMemory> recall(const std::string& query) const;
    void compact_history(bool force);
    void write_memories(const std::string& user_text, const std::string& answer,
                        SessionTurn* turn);
    // Notetaker: append one bounded bullet about what this turn established.
    void update_session_notes(const SessionTurn& turn);
    bool auxiliary_generate(const ChatMessages& messages, int max_tokens, std::string* text,
                            GenerationMetrics* metrics);
    std::string session_image_dir() const;
    bool cache_canvas(SessionTurn* turn, const std::string& canvas_path);
    void emit(const std::string& type, Json payload);
    void emit_error(const std::string& message, const std::string& stage);
    bool cancelled() const;
    // True when the generation starts with a registered tool name followed by
    // '{' or ':' but could not be parsed into a call - the phone sees this when
    // the model tries to hand the image bytes to image_info.
    bool looks_like_broken_call(const std::string& output) const;
    static bool looks_like_refusal(const std::string& answer);
    static bool mentions_image(const std::string& answer);
    // Strips the protocol/markup debris the Q4 export leaks into answers
    // (`<|"|>`, `<bos>`, `<b>…</b>`, a leading "answer:", bare "/image/analyze").
    static std::string clean_answer(const std::string& answer);
    // True when the answer actually uses one of the recalled facts.
    static bool answer_mentions_memory(const std::string& answer,
                                       const std::vector<ScoredMemory>& memories);
    static std::string memory_answer(const std::vector<ScoredMemory>& memories);
    // An image answer that refused, bounced the question back or is too short to
    // describe anything: the caller falls back to the shipped vision prompt.
    static bool looks_like_unusable_image_answer(const std::string& answer,
                                                const IntentPolicy& intent);
    // Last resort for an image turn the agent prompt made the model refuse: run
    // the shipped single-shot description prompt with the same soft tokens.
    std::string describe_image_directly(const std::string& canvas_path,
                                        const ImageInput& image);
    std::string system_prompt() const;
    // Chooses the prompt variant for this intent and reports whether it changed
    // (a change invalidates the KV prefix, so the turn is prefilled in full).
    bool select_system_prompt(IntentKind kind);
    static bool intent_needs_tools(IntentKind kind);
    int reserve_tokens() const;
    int history_budget_tokens() const;

    Options options_;
    AgentLlm& llm_;
    EventSink sink_;
    CancelCheck cancel_;
    ToolRegistry tools_;
    MemoryStore memory_;
    SessionStore sessions_;
    AgentPlanner planner_;
    std::string system_prompt_;        // tool-bearing variant
    std::string system_prompt_small_;  // chat/描述 variant (no tools, no protocol)
    int system_prompt_tokens_ = 0;
    bool last_prompt_had_tools_ = false;
    std::string last_tool_summary_;   // summary of the last successful tool call of this turn
    bool needs_full_prefill_ = false;
    bool kv_dirty_ = false;
    long long sequence_ = 0;
    long long turn_sequence_ = 0;
    std::atomic<bool> cancel_requested_{false};
};

}  // namespace agent

#endif  // GEMMA4_AGENT_RUNTIME_H
