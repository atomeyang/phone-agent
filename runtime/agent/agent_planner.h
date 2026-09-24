// Native plan state for a turn.
//
// The model proposes the steps (make_plan) and reports progress
// (update_plan); the planner owns the step list, enforces the step budget,
// tracks which step a tool call belongs to and renders the plan back into every
// observation so the small model always sees where it is.
#ifndef GEMMA4_AGENT_PLANNER_H
#define GEMMA4_AGENT_PLANNER_H

#include "agent_config.h"
#include "agent_json.h"
#include "agent_types.h"

#include <string>
#include <vector>

namespace agent {

class AgentPlanner {
public:
    explicit AgentPlanner(const AgentConfig& config) : config_(config) {}

    void begin_turn(const std::string& goal);
    int set_plan(const std::vector<std::string>& steps);
    bool update(int index, StepStatus status, const std::string& note);
    bool update_from_json(const Json& step, std::string* error);
    // Attaches the tool call to the active step and advances the status.
    void note_tool(const std::string& tool, bool ok);
    void finish_all();
    int activate_next();

    bool has_plan() const { return !steps_.empty(); }
    bool complete() const;
    int active_index() const;
    int completed_steps() const;
    const std::vector<PlanStep>& steps() const { return steps_; }
    const std::string& goal() const { return goal_; }

    std::string render() const;
    Json to_json() const;
    int step_count() const { return static_cast<int>(steps_.size()); }

private:
    AgentConfig config_;
    std::string goal_;
    std::vector<PlanStep> steps_;
};

}  // namespace agent

#endif  // GEMMA4_AGENT_PLANNER_H
