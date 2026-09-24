// Intent understanding - the router that runs before every turn.
//
// The agent does not treat every message the same way.  A native, deterministic
// classifier decides what the turn is about and, from that, the *policy* for the
// turn: whether to consult long term memory, whether tools are expected at all,
// and whether the answer must stay plain text.  That is what keeps a greeting
// from dragging the memory store and the tool protocol into the prompt, and what
// stops the model from answering an image question with a tool call.
//
// No model call is involved: this is pattern matching over the user text plus
// the turn's context, so it costs microseconds and is fully reproducible.
#ifndef GEMMA4_AGENT_INTENT_H
#define GEMMA4_AGENT_INTENT_H

#include "agent_tools.h"
#include "agent_types.h"

#include <string>

namespace agent {

enum class IntentKind {
    Chat,           // small talk, general questions: answer directly
    Greeting,       // "Nihao" / "你好" - no task, no tools, no image
    Arithmetic,     // "17*23?" - the calculator is expected
    Clock,          // "what time is it" - the now tool is expected
    Convert,        // unit conversion
    TextStats,      // "how many characters is this" - the text_stats tool
    MemoryWrite,    // "remember that ..."
    MemoryRecall,   // "what is my name / do you remember ..."
    ImageDescribe,  // an image plus "what is in it" - describe in plain text
    ImageQuestion,  // an image plus a specific question - tools allowed
    ImageAdvice,    // "how to make it better / brighter" - advise from the image
    MultiStep,      // a task that plausibly needs several tools
};

struct IntentPolicy {
    IntentKind kind = IntentKind::Chat;
    std::string label;              // human readable, shown in the UI
    double confidence = 0.0;        // 0..1, heuristic
    bool inject_memory = false;     // recall long term memory for this turn
    bool allow_tools = true;        // may execute a tool the model asks for
    bool prefer_plain_answer = false;  // a tool call is noise -> ask for text
    std::string preferred_tool;     // "" when the turn is not tool-shaped
    ToolCall seed;                  // exact call the router could derive itself
};

// Classifies one turn and derives its policy.  `has_image` is true when the app
// attached an image to this message.
IntentPolicy classify_intent(const std::string& user_text, bool has_image,
                             const ToolRegistry& tools);

const char* to_string(IntentKind kind);

}  // namespace agent

#endif  // GEMMA4_AGENT_INTENT_H
