#include "agent_planner.h"

#include "agent_util.h"

namespace agent {

void AgentPlanner::begin_turn(const std::string& goal) {
    goal_ = collapse_whitespace(goal);
    steps_.clear();
}

int AgentPlanner::set_plan(const std::vector<std::string>& steps) {
    steps_.clear();
    for (const auto& text : steps) {
        const std::string cleaned = collapse_whitespace(text);
        if (cleaned.empty()) {
            continue;
        }
        if (static_cast<int>(steps_.size()) >= config_.max_plan_steps) {
            break;
        }
        PlanStep step;
        step.index = static_cast<int>(steps_.size()) + 1;
        step.text = head(cleaned, 240);
        step.status = steps_.empty() ? StepStatus::Active : StepStatus::Pending;
        steps_.push_back(std::move(step));
    }
    return static_cast<int>(steps_.size());
}

bool AgentPlanner::update(int index, StepStatus status, const std::string& note) {
    if (index <= 0 || index > static_cast<int>(steps_.size())) {
        return false;
    }
    PlanStep& step = steps_[static_cast<size_t>(index - 1)];
    step.status = status;
    if (!note.empty()) {
        step.note = head(collapse_whitespace(note), 160);
    }
    if (status == StepStatus::Done || status == StepStatus::Failed) {
        activate_next();
    }
    return true;
}

bool AgentPlanner::update_from_json(const Json& step, std::string* error) {
    const long long index = step.integer_or("index", 0);
    if (index <= 0) {
        if (error != nullptr) {
            *error = "step index must be >= 1";
        }
        return false;
    }
    const std::string status = step.string_or("status", "");
    if (status.empty()) {
        if (error != nullptr) {
            *error = "missing status";
        }
        return false;
    }
    const std::string note = step.string_or("note", "");
    if (!update(static_cast<int>(index), step_status_from_string(status), note)) {
        if (error != nullptr) {
            *error = "unknown step " + std::to_string(index);
        }
        return false;
    }
    return true;
}

void AgentPlanner::note_tool(const std::string& tool, bool ok) {
    int index = active_index();
    if (index <= 0) {
        index = activate_next();
    }
    if (index <= 0) {
        return;
    }
    PlanStep& step = steps_[static_cast<size_t>(index - 1)];
    if (step.tool.empty()) {
        step.tool = tool;
    } else if (step.tool != tool) {
        step.tool += "," + tool;
    }
    if (!ok) {
        step.status = StepStatus::Failed;
        activate_next();
    }
}

void AgentPlanner::finish_all() {
    for (auto& step : steps_) {
        if (step.status == StepStatus::Active || step.status == StepStatus::Pending) {
            step.status = StepStatus::Done;
        }
    }
}

int AgentPlanner::activate_next() {
    for (auto& step : steps_) {
        if (step.status == StepStatus::Active) {
            return step.index;
        }
    }
    for (auto& step : steps_) {
        if (step.status == StepStatus::Pending) {
            step.status = StepStatus::Active;
            return step.index;
        }
    }
    return 0;
}

int AgentPlanner::active_index() const {
    for (const auto& step : steps_) {
        if (step.status == StepStatus::Active) {
            return step.index;
        }
    }
    return 0;
}

int AgentPlanner::completed_steps() const {
    int completed = 0;
    for (const auto& step : steps_) {
        if (step.status == StepStatus::Done) {
            ++completed;
        }
    }
    return completed;
}

bool AgentPlanner::complete() const {
    if (steps_.empty()) {
        return false;
    }
    for (const auto& step : steps_) {
        if (step.status == StepStatus::Active || step.status == StepStatus::Pending) {
            return false;
        }
    }
    return true;
}

std::string AgentPlanner::render() const {
    if (steps_.empty()) {
        return std::string();
    }
    std::string out = "PLAN";
    if (!goal_.empty()) {
        out += " (goal: " + head(goal_, 120) + ")";
    }
    out += ":\n";
    for (const auto& step : steps_) {
        out += "  ";
        out += std::to_string(step.index);
        out += ". [";
        out += to_string(step.status);
        out += "] ";
        out += step.text;
        if (!step.tool.empty()) {
            out += " (tool: " + step.tool + ")";
        }
        if (!step.note.empty()) {
            out += " - " + step.note;
        }
        out += "\n";
    }
    const int active = active_index();
    if (active > 0) {
        out += "CURRENT STEP: " + std::to_string(active) + "\n";
    }
    return out;
}

Json AgentPlanner::to_json() const {
    Json array = Json::array();
    for (const auto& step : steps_) {
        Json entry = Json::object();
        entry.set("index", Json::integer(step.index));
        entry.set("text", Json::string(step.text));
        entry.set("status", Json::string(to_string(step.status)));
        entry.set("note", Json::string(step.note));
        entry.set("tool", Json::string(step.tool));
        array.push(std::move(entry));
    }
    return array;
}

}  // namespace agent
