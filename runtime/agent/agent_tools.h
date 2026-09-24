// Native tool registry.
//
// Every tool is a pure function of its arguments plus the turn context: no
// network, no shell, no hidden state.  The registry renders the JSON schemas
// that go into the system prompt, validates the arguments the model produced
// and executes the handler, so a malformed call becomes an observation the
// model can recover from instead of a crash.
#ifndef GEMMA4_AGENT_TOOLS_H
#define GEMMA4_AGENT_TOOLS_H

#include "agent_config.h"
#include "agent_json.h"
#include "agent_memory.h"
#include "agent_planner.h"
#include "agent_session.h"
#include "agent_types.h"

#include <functional>
#include <string>
#include <vector>

namespace agent {

// Everything a tool may look at.  The runner fills the device/vision fields,
// which keeps the tool implementations free of platform dependencies.
struct ToolContext {
    const AgentConfig* config = nullptr;
    MemoryStore* memory = nullptr;
    SessionStore* sessions = nullptr;
    AgentPlanner* planner = nullptr;
    std::string session_id;
    int turn_index = 0;
    bool has_image = false;
    int image_tokens = 0;
    std::string image_note;
    std::string image_sha256;
    std::string image_file;
    Json device = Json::object();
    std::string language = "auto";
    std::vector<std::string>* written_memories = nullptr;
    std::string* last_memory_text = nullptr;
};

struct ToolSpec {
    std::string name;
    std::string description;
    Json parameters = Json::object();   // {type: object, properties: {...}, required: [...]}
    // First required argument (or first declared property).  Used to rescue a
    // call the model wrote as `calculator:17*23` without the native markers.
    std::string primary_argument;

    ToolSpec() = default;
    ToolSpec(std::string tool_name, std::string tool_description, Json tool_parameters)
        : name(std::move(tool_name)),
          description(std::move(tool_description)),
          parameters(std::move(tool_parameters)) {}
};

struct ToolDefinition {
    ToolSpec spec;
    std::function<ToolResult(const Json& arguments, ToolContext& context)> handler;
};

class ToolRegistry {
public:
    void add(ToolDefinition definition);
    bool has(const std::string& name) const;
    const ToolSpec* spec(const std::string& name) const;
    const std::vector<ToolDefinition>& definitions() const { return definitions_; }
    std::vector<std::string> names() const;

    ToolResult call(const std::string& name, const Json& arguments, ToolContext& context) const;
    // Maps the name the model wrote onto a registered tool: exact match, then a
    // small alias table, then an edit-distance match.  A 2B model regularly
    // writes `get_time` or `calc`, and a resolved alias is a much better answer
    // than an "unknown tool" round trip.
    std::string resolve(const std::string& name, std::string* note = nullptr) const;
    // Compact multi-line schema block appended to the system prompt.
    std::string render_for_prompt() const;
    Json to_json() const;

private:
    std::vector<ToolDefinition> definitions_;
};

// Registers calculator, now, unit_convert, text_stats, remember, recall,
// forget, search_history, image_info, device_info, make_plan, update_plan.
ToolRegistry build_default_tools();

// Exposed for the host tests.
double evaluate_expression(const std::string& expression, bool* ok, std::string* error);

}  // namespace agent

#endif  // GEMMA4_AGENT_TOOLS_H
