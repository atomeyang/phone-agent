#include "agent_session.h"

#include "agent_util.h"

#include <algorithm>
#include <sstream>

#include <unistd.h>

namespace agent {

SessionStore::SessionStore(std::string directory, const AgentConfig& config,
                           TokenCounter counter)
    : directory_(std::move(directory)), config_(config), counter_(std::move(counter)) {
    if (directory_.empty()) {
        directory_ = "agent-sessions";
    }
}

std::string SessionStore::session_dir(const std::string& id) const {
    return path_join(directory_, id);
}

bool SessionStore::load_index() {
    index_.clear();
    if (!make_directories(directory_)) {
        return false;
    }
    std::string contents;
    if (read_text_file(path_join(directory_, "index.json"), &contents)) {
        const Json parsed = Json::parse(contents);
        if (parsed.is_object()) {
            if (const Json* sessions = parsed.find("sessions")) {
                for (const auto& entry : sessions->items()) {
                    index_.push_back(meta_from_json(entry));
                }
            }
            id_sequence_ = static_cast<int>(parsed.integer_or("sequence", 1));
        }
    }
    if (index_.empty()) {
        // Rebuild from the directories so a lost index never hides history.
        for (const auto& name : list_directory(directory_)) {
            if (starts_with(name, "index.") || !is_directory(session_dir(name))) {
                continue;
            }
            std::string meta_text;
            SessionMeta meta;
            meta.id = name;
            if (read_text_file(path_join(session_dir(name), "meta.json"), &meta_text)) {
                meta = meta_from_json(Json::parse(meta_text));
                meta.id = name;
            } else {
                meta.title = name;
                meta.updated_ms = now_ms();
            }
            index_.push_back(meta);
        }
    }
    for (const auto& meta : index_) {
        if (starts_with(meta.id, "s")) {
            try {
                id_sequence_ = std::max(id_sequence_, std::stoi(meta.id.substr(1)) + 1);
            } catch (...) {
            }
        }
    }
    std::sort(index_.begin(), index_.end(), [](const SessionMeta& left, const SessionMeta& right) {
        return left.updated_ms > right.updated_ms;
    });
    return true;
}

bool SessionStore::write_index() const {
    Json root = Json::object();
    root.set("sequence", Json::integer(id_sequence_));
    root.set("updated_ms", Json::integer(now_ms()));
    Json sessions = Json::array();
    for (const auto& meta : index_) {
        sessions.push(meta_to_json(meta));
    }
    root.set("sessions", std::move(sessions));
    return write_file(path_join(directory_, "index.json"), root.dump(2));
}

bool SessionStore::create(const std::string& title_hint, const std::string& requested_id) {
    std::string id = requested_id;
    if (id.empty()) {
        id = "s" + std::to_string(id_sequence_++);
    }
    if (!make_directories(session_dir(id))) {
        return false;
    }
    meta_ = SessionMeta();
    meta_.id = id;
    meta_.title = title_hint.empty()
        ? "New session"
        : head(collapse_whitespace(title_hint),
               static_cast<size_t>(config_.session_title_chars) * 3);
    meta_.created_ms = now_ms();
    meta_.updated_ms = meta_.created_ms;
    turns_.clear();
    index_.erase(std::remove_if(index_.begin(), index_.end(),
                                [&](const SessionMeta& entry) { return entry.id == id; }),
                 index_.end());
    index_.insert(index_.begin(), meta_);
    if (static_cast<int>(index_.size()) > config_.max_sessions) {
        for (size_t position = index_.size(); position-- > static_cast<size_t>(config_.max_sessions);) {
            const std::string victim = index_[position].id;
            if (victim == id) {
                continue;
            }
            if (!is_directory(session_dir(victim))) {
                continue;
            }
            const auto files = list_directory(session_dir(victim));
            for (const auto& name : files) {
                remove_file(path_join(session_dir(victim), name));
            }
        }
        index_.resize(static_cast<size_t>(config_.max_sessions));
    }
    write_index();
    return persist_meta();
}

bool SessionStore::open(const std::string& id) {
    std::string meta_text;
    meta_ = SessionMeta();
    meta_.id = id;
    if (read_text_file(path_join(session_dir(id), "meta.json"), &meta_text)) {
        meta_ = meta_from_json(Json::parse(meta_text));
        meta_.id = id;
    } else {
        if (!make_directories(session_dir(id))) {
            return false;
        }
        meta_.title = id;
        meta_.created_ms = now_ms();
        meta_.updated_ms = meta_.created_ms;
    }
    if (!read_turns(id, &turns_)) {
        turns_.clear();
    }
    meta_.turns = static_cast<int>(turns_.size());
    return true;
}

bool SessionStore::read_turns(const std::string& id, std::vector<SessionTurn>* turns) const {
    turns->clear();
    std::string contents;
    if (!read_file(path_join(session_dir(id), "turns.jsonl"), &contents)) {
        return false;
    }
    std::istringstream stream(contents);
    std::string line;
    while (std::getline(stream, line)) {
        const std::string trimmed = trim(line);
        if (trimmed.empty()) {
            continue;
        }
        const Json parsed = Json::parse(trimmed);
        if (parsed.is_object()) {
            turns->push_back(turn_from_json(parsed));
        }
    }
    return true;
}

bool SessionStore::persist_meta() {
    if (meta_.id.empty()) {
        return false;
    }
    if (!make_directories(session_dir(meta_.id))) {
        return false;
    }
    for (auto& entry : index_) {
        if (entry.id == meta_.id) {
            entry = meta_;
        }
    }
    bool found = false;
    for (const auto& entry : index_) {
        found = found || entry.id == meta_.id;
    }
    if (!found) {
        index_.insert(index_.begin(), meta_);
    }
    std::sort(index_.begin(), index_.end(), [](const SessionMeta& left, const SessionMeta& right) {
        return left.updated_ms > right.updated_ms;
    });
    write_index();
    return write_file(path_join(session_dir(meta_.id), "meta.json"), meta_to_json(meta_).dump(2));
}

bool SessionStore::append_turn(const SessionTurn& turn) {
    if (meta_.id.empty()) {
        return false;
    }
    if (!make_directories(session_dir(meta_.id))) {
        return false;
    }
    if (!append_line(path_join(session_dir(meta_.id), "turns.jsonl"),
                     to_json(turn).dump())) {
        return false;
    }
    turns_.push_back(turn);
    meta_.turns = static_cast<int>(turns_.size());
    meta_.updated_ms = turn.started_ms != 0 ? turn.started_ms : now_ms();
    if (meta_.title.empty() || meta_.title == "New session") {
        meta_.title = head(collapse_whitespace(turn.user_text),
                           static_cast<size_t>(config_.session_title_chars) * 3);
        if (meta_.title.empty()) {
            meta_.title = "Session " + meta_.id;
        }
    }
    return persist_meta();
}

bool SessionStore::rewrite_turns() {
    if (meta_.id.empty()) {
        return false;
    }
    std::string contents;
    for (const auto& turn : turns_) {
        contents += to_json(turn).dump();
        contents.push_back('\n');
    }
    return write_file(path_join(session_dir(meta_.id), "turns.jsonl"), contents);
}

bool SessionStore::remove(const std::string& id) {
    // Session directories can hold subdirectories (the cached canvases of an
    // image turn), so remove the whole tree.
    remove_tree(session_dir(id));
    index_.erase(std::remove_if(index_.begin(), index_.end(),
                                [&](const SessionMeta& entry) { return entry.id == id; }),
                 index_.end());
    if (meta_.id == id) {
        meta_ = SessionMeta();
        turns_.clear();
    }
    return write_index();
}

bool SessionStore::rename(const std::string& id, const std::string& title) {
    const std::string trimmed = head(collapse_whitespace(title),
                                     static_cast<size_t>(config_.session_title_chars) * 3);
    for (auto& entry : index_) {
        if (entry.id != id) {
            continue;
        }
        entry.title = trimmed;
        entry.updated_ms = now_ms();
        if (meta_.id == id) {
            meta_.title = trimmed;
        }
        write_file(path_join(session_dir(id), "meta.json"), meta_to_json(entry).dump(2));
        return write_index();
    }
    return false;
}

ChatMessages SessionStore::render_turn(const SessionTurn& turn) const {
    ChatMessages messages;
    for (const auto& entry : turn.messages.items()) {
        const std::string role = entry.string_or("role", "");
        const std::string content = entry.string_or("content", "");
        if (role.empty() || content.empty()) {
            continue;
        }
        messages.emplace_back(role, content);
    }
    if (messages.empty() && !turn.user_text.empty()) {
        messages.emplace_back("user", turn.user_text);
        if (!turn.answer.empty()) {
            messages.emplace_back("assistant", turn.answer);
        }
    }
    return messages;
}

int SessionStore::transcript_tokens(int reserve_tokens) const {
    if (turns_.empty()) {
        return 0;
    }
    ChatMessages messages;
    if (!meta_.summary.empty()) {
        messages.emplace_back("user", "CONVERSATION SUMMARY:\n" + meta_.summary);
    }
    for (const auto& turn : turns_) {
        if (turn.index <= meta_.summary_upto) {
            continue;
        }
        const auto rendered = render_turn(turn);
        messages.insert(messages.end(), rendered.begin(), rendered.end());
    }
    if (messages.empty()) {
        return 0;
    }
    return counter_ ? counter_(messages) + reserve_tokens : 0;
}

bool SessionStore::needs_compaction(int budget_tokens, int reserve_tokens) const {
    if (turns_.size() <= static_cast<size_t>(config_.recent_turns_min)) {
        return false;
    }
    return transcript_tokens(reserve_tokens) > budget_tokens;
}

std::vector<SessionTurn> SessionStore::compaction_candidates() const {
    std::vector<SessionTurn> candidates;
    // The newest `recent_turns_min` turns always stay verbatim in the context.
    const int total = static_cast<int>(turns_.size());
    const int summarisable = total - config_.recent_turns_min;
    for (int position = 0; position < summarisable; ++position) {
        const auto& turn = turns_[static_cast<size_t>(position)];
        if (turn.index <= meta_.summary_upto) {
            continue;
        }
        candidates.push_back(turn);
    }
    return candidates;
}

bool SessionStore::apply_compaction(const std::string& summary, int upto_turn) {
    meta_.summary = summary;
    meta_.summary_upto = std::max(meta_.summary_upto, upto_turn);
    meta_.compactions += 1;
    meta_.updated_ms = now_ms();
    return persist_meta();
}

Json SessionStore::to_json(const SessionTurn& turn) {
    Json value = Json::object();
    value.set("index", Json::integer(turn.index));
    value.set("started_ms", Json::integer(turn.started_ms));
    value.set("duration_ms", Json::integer(turn.duration_ms));
    value.set("user", Json::string(turn.user_text));
    value.set("answer", Json::string(turn.answer));
    value.set("has_image", Json::boolean(turn.has_image));
    value.set("image_tokens", Json::integer(turn.image_tokens));
    value.set("image_note", Json::string(turn.image_note));
    value.set("image_file", Json::string(turn.image_file));
    value.set("messages", turn.messages);
    Json plan = Json::array();
    for (const auto& step : turn.plan) {
        Json entry = Json::object();
        entry.set("index", Json::integer(step.index));
        entry.set("text", Json::string(step.text));
        entry.set("status", Json::string(to_string(step.status)));
        entry.set("note", Json::string(step.note));
        entry.set("tool", Json::string(step.tool));
        plan.push(std::move(entry));
    }
    value.set("plan", std::move(plan));
    value.set("tool_calls", turn.tool_calls);
    value.set("recalled", turn.recalled);
    Json written = Json::array();
    for (const auto& id : turn.memories_written) {
        written.push(Json::string(id));
    }
    value.set("memories_written", std::move(written));
    value.set("steps", Json::integer(turn.steps));
    value.set("cancelled", Json::boolean(turn.cancelled));
    value.set("truncated", Json::boolean(turn.truncated));
    value.set("prompt_tokens", Json::integer(turn.usage.prompt_tokens));
    value.set("generated_tokens", Json::integer(turn.usage.generated_tokens));
    value.set("ttft_ms", Json::number(turn.ttft_ms));
    value.set("decode_tps", Json::number(turn.decode_tokens_per_second));
    value.set("prefill_ms", Json::number(turn.prefill_ms));
    value.set("npu_ms", Json::number(turn.npu_ms));
    value.set("stop_reason", Json::string(turn.stop_reason));
    return value;
}

SessionTurn SessionStore::turn_from_json(const Json& value) {
    SessionTurn turn;
    turn.index = static_cast<int>(value.integer_or("index", 0));
    turn.started_ms = value.integer_or("started_ms", 0);
    turn.duration_ms = value.integer_or("duration_ms", 0);
    turn.user_text = value.string_or("user", "");
    turn.answer = value.string_or("answer", "");
    turn.has_image = value.bool_or("has_image", false);
    turn.image_tokens = static_cast<int>(value.integer_or("image_tokens", 0));
    turn.image_note = value.string_or("image_note", "");
    turn.image_file = value.string_or("image_file", "");
    if (const Json* messages = value.find("messages")) {
        turn.messages = *messages;
    }
    if (const Json* plan = value.find("plan")) {
        for (const auto& entry : plan->items()) {
            PlanStep step;
            step.index = static_cast<int>(entry.integer_or("index", 0));
            step.text = entry.string_or("text", "");
            step.status = step_status_from_string(entry.string_or("status", "pending"));
            step.note = entry.string_or("note", "");
            step.tool = entry.string_or("tool", "");
            turn.plan.push_back(std::move(step));
        }
    }
    if (const Json* calls = value.find("tool_calls")) {
        turn.tool_calls = *calls;
    }
    if (const Json* recalled = value.find("recalled")) {
        turn.recalled = *recalled;
    }
    if (const Json* written = value.find("memories_written")) {
        for (const auto& entry : written->items()) {
            if (entry.is_string()) {
                turn.memories_written.push_back(entry.as_string());
            }
        }
    }
    turn.steps = static_cast<int>(value.integer_or("steps", 0));
    turn.cancelled = value.bool_or("cancelled", false);
    turn.truncated = value.bool_or("truncated", false);
    turn.usage.prompt_tokens = value.integer_or("prompt_tokens", 0);
    turn.usage.generated_tokens = value.integer_or("generated_tokens", 0);
    turn.ttft_ms = value.number_or("ttft_ms", 0.0);
    turn.decode_tokens_per_second = value.number_or("decode_tps", 0.0);
    turn.prefill_ms = value.number_or("prefill_ms", 0.0);
    turn.npu_ms = value.number_or("npu_ms", 0.0);
    turn.stop_reason = value.string_or("stop_reason", "");
    return turn;
}

Json SessionStore::meta_to_json(const SessionMeta& meta) {
    Json value = Json::object();
    value.set("id", Json::string(meta.id));
    value.set("title", Json::string(meta.title));
    value.set("created_ms", Json::integer(meta.created_ms));
    value.set("updated_ms", Json::integer(meta.updated_ms));
    value.set("turns", Json::integer(meta.turns));
    value.set("compactions", Json::integer(meta.compactions));
    value.set("summary", Json::string(meta.summary));
    value.set("notes", Json::string(meta.notes));
    value.set("summary_upto", Json::integer(meta.summary_upto));
    return value;
}

SessionMeta SessionStore::meta_from_json(const Json& value) {
    SessionMeta meta;
    meta.id = value.string_or("id", "");
    meta.title = value.string_or("title", "");
    meta.created_ms = value.integer_or("created_ms", 0);
    meta.updated_ms = value.integer_or("updated_ms", 0);
    meta.turns = static_cast<int>(value.integer_or("turns", 0));
    meta.compactions = static_cast<int>(value.integer_or("compactions", 0));
    meta.summary = value.string_or("summary", "");
    meta.notes = value.string_or("notes", "");
    meta.summary_upto = static_cast<int>(value.integer_or("summary_upto", 0));
    return meta;
}

}  // namespace agent
