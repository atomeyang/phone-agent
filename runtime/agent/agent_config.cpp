#include "agent_config.h"

#include "agent_util.h"

#include <cmath>
#include <cstdlib>

namespace agent {
namespace {

int env_int(const char* name, int fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return fallback;
    }
    const int parsed = std::atoi(value);
    return parsed > 0 ? parsed : fallback;
}

bool env_flag(const char* name, bool fallback) {
    const char* value = std::getenv(name);
    if (value == nullptr || *value == '\0') {
        return fallback;
    }
    return std::string(value) != "0";
}

}  // namespace

AgentConfig AgentConfig::load(const std::string& path, const std::string& root) {
    AgentConfig config;
    config.root = root;
    config.agent_dir = path_join(root, "agent");
    config.memory_dir = path_join(config.agent_dir, "memory");
    config.sessions_dir = path_join(config.agent_dir, "sessions");
    config.work_dir = path_join(config.agent_dir, "work");

    std::string text;
    if (!path.empty() && read_text_file(path, &text)) {
        std::string error;
        const Json parsed = Json::parse(text, &error);
        if (parsed.is_object()) {
            config.max_tool_calls =
                static_cast<int>(parsed.integer_or("max_tool_calls", config.max_tool_calls));
            config.max_steps =
                static_cast<int>(parsed.integer_or("max_steps", config.max_steps));
            config.max_tokens_per_step = static_cast<int>(
                parsed.integer_or("max_tokens_per_step", config.max_tokens_per_step));
            config.max_answer_tokens = static_cast<int>(
                parsed.integer_or("max_answer_tokens", config.max_answer_tokens));
            config.max_turn_tokens =
                parsed.integer_or("max_turn_tokens", config.max_turn_tokens);
            config.max_observation_chars = static_cast<int>(
                parsed.integer_or("max_observation_chars", config.max_observation_chars));
            config.max_memory_chars = static_cast<int>(
                parsed.integer_or("max_memory_chars", config.max_memory_chars));
            config.max_plan_steps =
                static_cast<int>(parsed.integer_or("max_plan_steps", config.max_plan_steps));
            config.context_budget_tokens = static_cast<int>(
                parsed.integer_or("context_budget_tokens", config.context_budget_tokens));
            config.history_budget_tokens = static_cast<int>(parsed.integer_or(
                "history_budget_tokens",
                parsed.integer_or("compaction_trigger_tokens",
                                  config.history_budget_tokens)));
            config.recent_turns_min =
                static_cast<int>(parsed.integer_or("recent_turns_min", config.recent_turns_min));
            config.recent_turns_max =
                static_cast<int>(parsed.integer_or("recent_turns_max", config.recent_turns_max));
            config.compaction_min_turns = static_cast<int>(
                parsed.integer_or("compaction_min_turns", config.compaction_min_turns));
            config.max_context_turns =
                static_cast<int>(parsed.integer_or("max_context_turns", config.max_context_turns));
            config.compaction_min_turns =
        env_int("GEMMA4_AGENT_COMPACT_MIN_TURNS", config.compaction_min_turns);
    config.image_window_turns = static_cast<int>(
                parsed.integer_or("image_window_turns", config.image_window_turns));
            config.tool_detail_window_turns = static_cast<int>(
                parsed.integer_or("tool_detail_window_turns", config.tool_detail_window_turns));
            config.session_notes = parsed.bool_or("session_notes", config.session_notes);
            config.session_notes_lines = static_cast<int>(
                parsed.integer_or("session_notes_lines", config.session_notes_lines));
            config.session_notes_chars = static_cast<int>(
                parsed.integer_or("session_notes_chars", config.session_notes_chars));
            if (const Json* tips = parsed.find("system_tips")) {
                for (const auto& entry : tips->items()) {
                    if (entry.is_string() && !entry.as_string().empty()) {
                        config.system_tips.push_back(entry.as_string());
                    }
                }
            }
            config.memory_enabled = parsed.bool_or("memory", config.memory_enabled);
            config.include_history =
                parsed.bool_or("include_history", config.include_history);
            config.system_prompt_mode =
                parsed.string_or("system_prompt_mode", config.system_prompt_mode);
            config.reuse_history_images =
                parsed.bool_or("reuse_history_images", config.reuse_history_images);
            config.image_delta_prefill =
                parsed.bool_or("image_delta_prefill", config.image_delta_prefill);
            config.store_episodes = parsed.bool_or("store_episodes", config.store_episodes);
            config.recall_top_k =
                static_cast<int>(parsed.integer_or("recall_top_k", config.recall_top_k));
            config.memory_max_records = static_cast<int>(
                parsed.integer_or("memory_max_records", config.memory_max_records));
            config.memory_half_life_days =
                parsed.number_or("memory_half_life_days", config.memory_half_life_days);
            config.memory_episode_weight =
                parsed.number_or("memory_episode_weight", config.memory_episode_weight);
            config.memory_min_score =
                parsed.number_or("memory_min_score", config.memory_min_score);
            config.planning_enabled = parsed.bool_or("planning", config.planning_enabled);
            config.stream_steps = parsed.bool_or("stream_steps", config.stream_steps);
            config.allow_tool_retry = parsed.bool_or("allow_tool_retry", config.allow_tool_retry);
            config.language = parsed.string_or("language", config.language);
            config.sampler = parsed.string_or("sampler", config.sampler);
            config.max_sessions =
                static_cast<int>(parsed.integer_or("max_sessions", config.max_sessions));
            config.session_title_chars = static_cast<int>(
                parsed.integer_or("session_title_chars", config.session_title_chars));
            config.debug = parsed.bool_or("debug", config.debug);
            config.poll_interval_ms =
                static_cast<int>(parsed.integer_or("poll_interval_ms", config.poll_interval_ms));
            config.max_event_bytes = static_cast<int>(
                parsed.integer_or("max_event_bytes", config.max_event_bytes));
            const std::string agent_dir = parsed.string_or("agent_dir", "");
            if (!agent_dir.empty()) {
                config.agent_dir = agent_dir;
                config.memory_dir = path_join(agent_dir, "memory");
                config.sessions_dir = path_join(agent_dir, "sessions");
                config.work_dir = path_join(agent_dir, "work");
            }
        }
    }

    const char* agent_root = std::getenv("GEMMA4_AGENT_DIR");
    if (agent_root != nullptr && *agent_root != '\0') {
        config.agent_dir = agent_root;
        config.memory_dir = path_join(config.agent_dir, "memory");
        config.sessions_dir = path_join(config.agent_dir, "sessions");
        config.work_dir = path_join(config.agent_dir, "work");
    }

    config.max_tool_calls = env_int("GEMMA4_AGENT_MAX_TOOL_CALLS", config.max_tool_calls);
    config.max_steps = env_int("GEMMA4_AGENT_MAX_STEPS", config.max_steps);
    config.max_answer_tokens =
        env_int("GEMMA4_AGENT_MAX_ANSWER_TOKENS", config.max_answer_tokens);
    config.max_tokens_per_step =
        env_int("GEMMA4_AGENT_MAX_STEP_TOKENS", config.max_tokens_per_step);
    config.context_budget_tokens =
        env_int("GEMMA4_AGENT_CONTEXT_BUDGET", config.context_budget_tokens);
    config.history_budget_tokens =
        env_int("GEMMA4_AGENT_HISTORY_BUDGET", config.history_budget_tokens);
    config.recall_top_k = env_int("GEMMA4_AGENT_RECALL_K", config.recall_top_k);
    config.image_window_turns =
        env_int("GEMMA4_AGENT_IMAGE_WINDOW", config.image_window_turns);
    config.tool_detail_window_turns =
        env_int("GEMMA4_AGENT_TOOL_WINDOW", config.tool_detail_window_turns);
    config.session_notes = env_flag("GEMMA4_AGENT_SESSION_NOTES", config.session_notes);
    config.memory_max_records =
        env_int("GEMMA4_AGENT_MEMORY_MAX", config.memory_max_records);
    config.memory_enabled = env_flag("GEMMA4_AGENT_MEMORY", config.memory_enabled);
    config.include_history = env_flag("GEMMA4_AGENT_HISTORY", config.include_history);
    config.store_episodes = env_flag("GEMMA4_AGENT_STORE_EPISODES", config.store_episodes);
    config.planning_enabled = env_flag("GEMMA4_AGENT_PLANNING", config.planning_enabled);
    config.stream_steps = env_flag("GEMMA4_AGENT_STREAM_STEPS", config.stream_steps);
    config.debug = env_flag("GEMMA4_AGENT_DEBUG", config.debug);

    if (config.history_budget_tokens > 0 &&
        config.history_budget_tokens >= config.context_budget_tokens) {
        config.history_budget_tokens =
            std::max(128, config.context_budget_tokens / 3);
    }
    if (config.recent_turns_min < 1) {
        config.recent_turns_min = 1;
    }
    if (config.recent_turns_max < config.recent_turns_min) {
        config.recent_turns_max = config.recent_turns_min;
    }
    return config;
}

Json AgentConfig::describe() const {
    Json out = Json::object();
    out.set("agent_dir", Json::string(agent_dir));
    out.set("max_tool_calls", Json::integer(max_tool_calls));
    out.set("max_steps", Json::integer(max_steps));
    out.set("max_tokens_per_step", Json::integer(max_tokens_per_step));
    out.set("max_answer_tokens", Json::integer(max_answer_tokens));
    out.set("max_turn_tokens", Json::integer(max_turn_tokens));
    out.set("context_budget_tokens", Json::integer(context_budget_tokens));
    out.set("history_budget_tokens", Json::integer(history_budget_tokens));
    out.set("image_window_turns", Json::integer(image_window_turns));
    out.set("tool_detail_window_turns", Json::integer(tool_detail_window_turns));
    out.set("session_notes", Json::boolean(session_notes));
    out.set("system_tips", Json::integer(static_cast<long long>(system_tips.size())));
    out.set("recent_turns_min", Json::integer(recent_turns_min));
    out.set("recent_turns_max", Json::integer(recent_turns_max));
    out.set("compaction_min_turns", Json::integer(compaction_min_turns));
    out.set("max_context_turns", Json::integer(max_context_turns));
    out.set("memory_enabled", Json::boolean(memory_enabled));
    out.set("include_history", Json::boolean(include_history));
    out.set("system_prompt_mode", Json::string(system_prompt_mode));
    out.set("reuse_history_images", Json::boolean(reuse_history_images));
    out.set("image_delta_prefill", Json::boolean(image_delta_prefill));
    out.set("store_episodes", Json::boolean(store_episodes));
    out.set("recall_top_k", Json::integer(recall_top_k));
    out.set("memory_max_records", Json::integer(memory_max_records));
    out.set("memory_half_life_days", Json::number(memory_half_life_days));
    out.set("memory_min_score", Json::number(memory_min_score));
    out.set("planning_enabled", Json::boolean(planning_enabled));
    out.set("stream_steps", Json::boolean(stream_steps));
    out.set("language", Json::string(language));
    out.set("sampler", Json::string(sampler));
    out.set("max_sessions", Json::integer(max_sessions));
    out.set("debug", Json::boolean(debug));
    return out;
}

}  // namespace agent
