#include "agent_types.h"

namespace agent {

const char* to_string(StepStatus status) {
    switch (status) {
        case StepStatus::Pending: return "pending";
        case StepStatus::Active: return "active";
        case StepStatus::Done: return "done";
        case StepStatus::Failed: return "failed";
        case StepStatus::Skipped: return "skipped";
    }
    return "pending";
}

StepStatus step_status_from_string(const std::string& value) {
    if (value == "active" || value == "running") {
        return StepStatus::Active;
    }
    if (value == "done" || value == "complete" || value == "completed") {
        return StepStatus::Done;
    }
    if (value == "failed" || value == "error") {
        return StepStatus::Failed;
    }
    if (value == "skipped") {
        return StepStatus::Skipped;
    }
    return StepStatus::Pending;
}

}  // namespace agent
