// Runtime limits for the on-device agent.  Defaults are conservative because
// the text backbone runs a 2k token KV cache on four CPU cores; every value can
// be overridden from agent-config.json or an environment variable so a device
// experiment never needs a rebuild.
#ifndef GEMMA4_AGENT_CONFIG_H
#define GEMMA4_AGENT_CONFIG_H

#include "agent_json.h"

#include <string>
#include <vector>

namespace agent {

struct AgentConfig {
    // Layout: <root> is the runner's runtime directory.
    std::string root;
    std::string agent_dir;
    std::string memory_dir;
    std::string sessions_dir;
    std::string work_dir;

    // Agent loop budgets (per user turn).
    int max_tool_calls = 4;
    int max_steps = 6;
    int max_tokens_per_step = 224;
    int max_answer_tokens = 320;
    long long max_turn_tokens = 1400;         // prompt + generated for one turn
    int max_observation_chars = 600;
    int max_memory_chars = 400;
    int max_plan_steps = 6;

    // Conversation context management.
    // The engine's KV cache holds max_all_tokens (2048 by default) prompt +
    // generated tokens, and the tool declarations alone cost ~830 of them, so
    // the prompt budget and the history budget are deliberately separate.
    int context_budget_tokens = 1700;         // hard ceiling for one prompt
    // 0 = derive the history budget from the prompt budget minus the system
    // prompt and one turn's allowance; set it to pin an explicit value.
    int history_budget_tokens = 0;
    int kv_total_tokens = 2048;               // MNN max_all_tokens
    int recent_turns_min = 2;                 // always kept verbatim
    int recent_turns_max = 8;                 // verbatim window (the rest is carried by summary + notes)
    // A compaction folds at least this many turns at once; folding one turn per
    // user message (what the phone log showed: 17 compactions in 28 turns) costs
    // a summary generation plus an engine re-creation for almost no gain.
    int compaction_min_turns = 4;
    int max_context_turns = 24;
    // Two-tier history (borrowed from Mobile-Agent-v3.5's build_messages): only
    // the newest turns keep their expensive payload - the NPU soft tokens of an
    // image and the full tool call/observation text.  Older turns are folded into
    // text the model can still reason about.
    int image_window_turns = 2;               // image turns that keep soft tokens
    int tool_detail_window_turns = 2;         // turns that keep full tool detail

    // Memory system.
    bool memory_enabled = true;
    // Turn history in the prompt.  Off = "fresh context" chat: only the system
    // prompt, the current message and this turn's tool exchanges (fastest, and
    // the model cannot fixate on earlier context).
    bool include_history = true;
    // System prompt shape: "adaptive" picks a small prompt for chat/描述/记忆
    // turns and the tool-bearing one only when a tool is expected; "full"/"minimal"
    // pin one of them.
    std::string system_prompt_mode = "adaptive";
    // Images are injected only in the turn that carried them.  With this off the
    // history keeps a text description instead of the 130 soft-token
    // placeholders, so a full prefill never re-runs the vision tower for an old
    // picture (and the prompt stays small).
    bool reuse_history_images = false;
    // An image turn continues the resident KV cache and lets the engine prefill
    // only the new suffix (the driver splices the vision vectors through MNN's
    // soft-token provider).  Off = the turn starts from a pristine engine, which
    // costs the model load but is the delivery that was validated first.
    bool image_delta_prefill = true;
    // Notetaker (Mobile-Agent-E): a bounded deterministic digest of this session
    // - what the images showed, what was stored, which tools ran - injected into
    // the *current* user message so it never invalidates the prompt cache.  It is
    // what keeps an older image usable once its soft tokens leave the window.
    bool session_notes = true;
    int session_notes_lines = 8;
    int session_notes_chars = 900;
    // Advice the deployment pins into the system prompt (Mobile-Agent-E's
    // INIT_TIPS idea).
    std::vector<std::string> system_tips;
    // Turn episodes stay in the session transcript by default; only durable
    // user facts/preferences/notes go to the cross-session memory store.
    bool store_episodes = false;
    int recall_top_k = 4;
    int memory_max_records = 500;
    double memory_half_life_days = 14.0;
    double memory_episode_weight = 0.55;
    double memory_min_score = 0.12;

    // Reasoning / tool protocol.
    bool planning_enabled = true;
    bool stream_steps = true;
    bool allow_tool_retry = true;
    std::string language = "auto";
    std::string sampler = "greedy";

    // Session store.
    int max_sessions = 40;
    int session_title_chars = 48;

    // Service behaviour.
    bool debug = false;
    int poll_interval_ms = 40;
    int max_event_bytes = 2000000;   // truncate the event log past this size

    // Loads <path> if it exists (missing file = defaults), then applies the
    // GEMMA4_AGENT_* environment overrides.
    static AgentConfig load(const std::string& path, const std::string& root);

    // True when the config carries non-default values that the UI should show.
    Json describe() const;
};

}  // namespace agent

#endif  // GEMMA4_AGENT_CONFIG_H
