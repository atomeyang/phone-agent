// Shared value types for the native on-device agent.
#ifndef GEMMA4_AGENT_TYPES_H
#define GEMMA4_AGENT_TYPES_H

#include "agent_json.h"

#include <string>
#include <utility>
#include <vector>

namespace agent {

// A chat message in the shape MNN's tokenizer/chat template consumes:
// role is "system" / "user" / "assistant"; content is plain UTF-8 text.
using ChatMessage = std::pair<std::string, std::string>;
using ChatMessages = std::vector<ChatMessage>;

// One tool invocation decoded from the model's output.
struct ToolCall {
    std::string name;
    Json arguments = Json::object();
    std::string raw;      // the JSON object exactly as the model wrote it
    bool valid = false;   // parsed and passed the schema check
    std::string error;    // why the call could not be executed
};

// The result of running a native tool.
struct ToolResult {
    std::string name;
    bool ok = false;
    Json value = Json::object();   // tool specific payload
    std::string error;
    std::string summary;           // one-line rendering for the observation
    double duration_ms = 0.0;
};

enum class StepStatus { Pending, Active, Done, Failed, Skipped };

const char* to_string(StepStatus status);
StepStatus step_status_from_string(const std::string& value);

// A planning step.  The plan is owned by the native planner; the model creates
// and updates steps through the make_plan/update_plan tools, and the runtime
// injects the rendered plan into every observation.
struct PlanStep {
    int index = 0;
    std::string text;
    StepStatus status = StepStatus::Pending;
    std::string note;
    std::string tool;
};

struct TokenUsage {
    long long prompt_tokens = 0;
    long long generated_tokens = 0;
};

// One completed agent turn as stored in the session transcript.
struct SessionTurn {
    int index = 0;
    long long started_ms = 0;
    long long duration_ms = 0;
    std::string user_text;
    std::string answer;
    bool has_image = false;
    int image_tokens = 0;
    std::string image_note;      // human readable description of the image
    std::string image_file;      // cached canvas file, relative to the session
    // The exact (role, content) list the model saw, including the tool calls
    // and observations in between.  Replaying it verbatim is what lets the
    // prompt cache reuse the KV prefix across turns.
    Json messages = Json::array();
    std::vector<PlanStep> plan;
    Json tool_calls = Json::array();   // [{name, arguments, ok, summary, duration_ms}]
    Json recalled = Json::array();     // [{id, text, score, kind}]
    std::vector<std::string> memories_written;
    int steps = 0;
    bool cancelled = false;
    bool truncated = false;
    TokenUsage usage;
    double ttft_ms = 0.0;
    double decode_tokens_per_second = 0.0;
    double prefill_ms = 0.0;
    double npu_ms = 0.0;
    std::string stop_reason;
};

// A memory record in the persistent store.
struct MemoryRecord {
    std::string id;
    std::string text;
    std::string kind;    // fact | preference | note | episode
    std::vector<std::string> tags;
    double importance = 0.5;
    bool pinned = false;
    long long created_ms = 0;
    long long updated_ms = 0;
    long long last_access_ms = 0;
    int access_count = 0;
    std::string session_id;
    int turn_index = 0;
    // Set when the user is correcting an earlier statement ("改成…", "instead");
    // the store then replaces the conflicting record instead of adding one.
    bool correction = false;
};

struct ScoredMemory {
    MemoryRecord record;
    double score = 0.0;
    double lexical = 0.0;
    double recency = 1.0;
};

}  // namespace agent

#endif  // GEMMA4_AGENT_TYPES_H
