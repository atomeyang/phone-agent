// Prompt construction and output parsing for the agent protocol.
//
// Gemma 4 is post-trained on an explicit function-calling syntax, and the
// checkpoint's chat template defines it:
//
//   tool declaration (inside the system turn)
//     <|tool>declaration:NAME{description:<|"|>...<|"|>,parameters:{...}}<tool|>
//   the model asks for a call with
//     <|tool_call>call:NAME{arg:value,arg2:value}<tool_call|>
//   and the tool answer comes back as
//     <|tool_response>response:NAME{key:value}<tool_response|>
//
// Using that syntax instead of an invented JSON protocol is what makes a 2B
// quantised model call tools reliably.  The rendered prompts stay byte-stable
// so MNN's prompt cache can reuse the KV prefix across turns.
#ifndef GEMMA4_AGENT_PROMPT_H
#define GEMMA4_AGENT_PROMPT_H

#include "agent_config.h"
#include "agent_json.h"
#include "agent_tools.h"
#include "agent_types.h"

#include <string>
#include <vector>

namespace agent {

constexpr const char* kToolCallMarker = "TOOL_CALL";
constexpr const char* kNativeToolCallOpen = "<|tool_call>";
constexpr const char* kNativeToolCallClose = "<tool_call|>";
constexpr const char* kNativeToolResponseOpen = "<|tool_response>";
constexpr const char* kNativeToolResponseClose = "<tool_response|>";

// include_tools=false builds the small "just talk to me" prompt: no tool
// protocol section and no tool declarations (about 250 tokens less).
std::string build_system_prompt(const AgentConfig& config, const ToolRegistry& tools,
                                const Json& device, bool include_tools = true);

// The user message for one turn.  When image_tokens > 0 the content starts with
// the Gemma 4 image delimiters and exactly that many placeholder tokens, which
// the runner then replaces with the NPU soft tokens.
std::string build_user_message(const std::string& user_text,
                               const std::vector<ScoredMemory>& memories, int image_tokens,
                               const std::string& session_notes = std::string());

// Deterministic, explainable routing hint computed in native code and appended
// to the user message (never to the system prompt, which has to stay byte
// identical for the prompt cache).  It nudges the model towards the tool that
// fits the request; the model still decides.
std::string build_task_hint(const std::string& user_text, bool has_image);

std::string build_memory_block(const std::vector<ScoredMemory>& memories);

std::string format_observation(const ToolResult& result, const std::string& plan_render,
                               int remaining_calls);

// --- native gemma4 function-calling syntax ---------------------------------

// <|tool>declaration:NAME{description:...,parameters:{...}}<tool|>
std::string format_tool_declaration(const ToolSpec& spec);
// <|tool_call>call:NAME{arg:value,...}<tool_call|>
std::string format_tool_call_block(const std::string& name, const Json& arguments);
// <|tool_response>response:NAME{ok:...,result:...}<tool_response|>
std::string format_tool_response_block(const std::string& name, bool ok,
                                       const std::string& summary, const Json& value);
// Serialises a JSON value the way the gemma4 template does (strings wrapped in
// <|"|>, unquoted keys, objects in {}, arrays in []).
std::string format_argument(const Json& value, bool escape_keys);
// Parses `key:value,key2:value2` as produced by the model.
bool parse_arguments(const std::string& text, Json* out, std::string* error);

std::string build_compaction_prompt(const std::string& transcript, int max_bullets);
std::string build_memory_prompt(const std::string& user_text, const std::string& answer);
std::string build_forced_answer_prompt(const std::string& reason);

// Returns a call with valid=false and error="no tool call" when the text is a
// final answer rather than a tool call.
// `tools` enables the recovery form the deployed Q4 export falls back to on the
// phone: a bare first line like `calculator:17*23`.  Without it only the
// explicit forms are accepted.
ToolCall parse_tool_call(const std::string& text, const ToolRegistry* tools = nullptr);

// Heuristic, always-on memory extraction.  Runs without an extra model pass so
// that a turn never costs two generations just to remember something.
struct ExtractedMemory {
    std::string text;
    std::string kind;
    double importance = 0.4;
    bool pinned = false;
    bool correction = false;   // the user is replacing an earlier statement
};

std::vector<ExtractedMemory> extract_memories(const std::string& user_text,
                                              const std::string& answer,
                                              const std::string& language);

// Parses the model's JSON answer for the optional model-side extraction pass.
std::vector<ExtractedMemory> parse_memory_json(const std::string& text);

// Removes the tool call line so a mixed answer can still be shown to the user.
std::string strip_tool_call(const std::string& text);

// Removes the model's thinking channel, exactly like the checkpoint's chat
// template does (`<|channel>thought ... <channel|>`).  The deployed Q4 export
// emits those blocks as plain text, and they must never reach the user.
std::string strip_thinking(const std::string& text);

}  // namespace agent

#endif  // GEMMA4_AGENT_PROMPT_H
