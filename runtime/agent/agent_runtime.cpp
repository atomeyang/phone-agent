#include "agent_runtime.h"

#include "agent_intent.h"
#include "agent_prompt.h"
#include "agent_util.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <sstream>

namespace agent {
namespace {

Json memory_to_json(const ScoredMemory& memory) {
    Json entry = Json::object();
    entry.set("id", Json::string(memory.record.id));
    entry.set("kind", Json::string(memory.record.kind));
    entry.set("text", Json::string(memory.record.text));
    entry.set("score", Json::number(memory.score));
    entry.set("lexical", Json::number(memory.lexical));
    entry.set("recency", Json::number(memory.recency));
    entry.set("importance", Json::number(memory.record.importance));
    entry.set("pinned", Json::boolean(memory.record.pinned));
    entry.set("updated_ms", Json::integer(memory.record.updated_ms));
    return entry;
}

Json message_entries(const SessionTurn& turn) {
    Json array = Json::array();
    for (const auto& entry : turn.messages.items()) {
        if (!entry.is_object()) {
            continue;
        }
        Json message = Json::object();
        message.set("role", Json::string(entry.string_or("role", "")));
        message.set("content", Json::string(entry.string_or("content", "")));
        array.push(std::move(message));
    }
    return array;
}

Json tool_call_entry(const ToolResult& result, const ToolCall& call) {
    Json entry = Json::object();
    entry.set("name", Json::string(call.name));
    entry.set("arguments", call.arguments);
    entry.set("ok", Json::boolean(result.ok));
    entry.set("error", Json::string(result.error));
    entry.set("summary", Json::string(result.summary));
    entry.set("duration_ms", Json::number(result.duration_ms));
    return entry;
}

// "characters=15 words=0 lines=1" -> a sentence the user can read.
std::string format_count_line(const std::string& summary, const std::string& user_text) {
    std::string characters;
    std::string words;
    for (const std::string& part : split(summary, ' ')) {
        if (starts_with(part, "characters=")) {
            characters = part.substr(std::string("characters=").size());
        } else if (starts_with(part, "words=")) {
            words = part.substr(std::string("words=").size());
        }
    }
    if (characters.empty()) {
        return std::string();
    }
    const bool cjk = user_text.find_first_not_of(
                         "abcdefghijklmnopqrstuvwxyz"
                         "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?:;'\"()[]{}"
                         "<>+-*/=_@#$%^&|\\~`\t\r\n") != std::string::npos;
    return cjk ? "这段文字有 " + characters + " 个字符" +
                     (words.empty() ? "。" : "（" + words + " 个词）。")
               : "That text has " + characters + " characters" +
                     (words.empty() ? "." : " (" + words + " words).");
}

// The runtime sometimes replaces what the model wrote (the tool's own sentence
// for a value question, a memory answer, a native fallback).  The transcript has
// to keep what the user is shown: otherwise the next turn renders the model's
// original sentence while the KV cache was anchored on the replacement, the
// prompt no longer extends the cache and the whole conversation is re-prefilled.
void replace_last_assistant(Json* messages, const std::string& text) {
    if (messages == nullptr || !messages->is_array()) {
        return;
    }
    const auto& items = messages->items();
    if (!items.empty() &&
        items.back().string_or("role", "") == "assistant") {
        Json updated = Json::array();
        for (size_t index = 0; index + 1 < items.size(); ++index) {
            updated.push(items[index]);
        }
        Json entry = Json::object();
        entry.set("role", Json::string("assistant"));
        entry.set("content", Json::string(text));
        updated.push(std::move(entry));
        *messages = std::move(updated);
        return;
    }
    Json entry = Json::object();
    entry.set("role", Json::string("assistant"));
    entry.set("content", Json::string(text));
    messages->push(std::move(entry));
}

int count_image_tokens(const std::string& content) {
    static const std::string placeholder = "<|image|>";
    int count = 0;
    size_t position = 0;
    while ((position = content.find(placeholder, position)) != std::string::npos) {
        ++count;
        position += placeholder.size();
    }
    return count;
}

std::string strip_placeholders(const std::string& content, const std::string& note) {
    std::string out;
    size_t position = 0;
    const std::string begin = "<|image>";
    const std::string end = "<image|>";
    while (true) {
        const size_t start = content.find(begin, position);
        if (start == std::string::npos) {
            out += content.substr(position);
            break;
        }
        out += content.substr(position, start - position);
        const size_t finish = content.find(end, start);
        if (finish == std::string::npos) {
            out += content.substr(start);
            break;
        }
        out += "[image: " + note + "]";
        position = finish + end.size();
    }
    return out;
}

}  // namespace

AgentRuntime::AgentRuntime(Options options, AgentLlm& llm, EventSink sink, CancelCheck cancel)
    : options_(std::move(options)),
      llm_(llm),
      sink_(std::move(sink)),
      cancel_(std::move(cancel)),
      tools_(build_default_tools()),
      memory_(options_.config.memory_dir, options_.config),
      sessions_(options_.config.sessions_dir, options_.config,
                [this](const ChatMessages& messages) { return llm_.count_tokens(messages); }),
      planner_(options_.config) {}

bool AgentRuntime::initialize(std::string* error) {
    if (!make_directories(options_.config.agent_dir) ||
        !make_directories(options_.config.memory_dir) ||
        !make_directories(options_.config.sessions_dir) ||
        !make_directories(options_.config.work_dir)) {
        if (error != nullptr) {
            *error = "cannot create the agent directories under " + options_.config.agent_dir;
        }
        return false;
    }
    if (!memory_.load()) {
        if (error != nullptr) {
            *error = "cannot load the memory store";
        }
        return false;
    }
    if (!sessions_.load_index()) {
        if (error != nullptr) {
            *error = "cannot load the session index";
        }
        return false;
    }
    if (!sessions_.is_open()) {
        if (!sessions_.index().empty()) {
            sessions_.open(sessions_.index().front().id);
            needs_full_prefill_ = true;
        } else {
            sessions_.create("New session");
        }
    }
    system_prompt_ = build_system_prompt(options_.config, tools_, options_.device, true);
    system_prompt_small_ =
        build_system_prompt(options_.config, tools_, options_.device, false);
    const bool want_tools = options_.config.system_prompt_mode != "minimal";
    system_prompt_ = want_tools ? system_prompt_ : system_prompt_small_;
    last_prompt_had_tools_ = want_tools;
    system_prompt_tokens_ = llm_.count_tokens({{"system", system_prompt_}});
    return true;
}

std::string AgentRuntime::system_prompt() const {
    return system_prompt_;
}

bool AgentRuntime::intent_needs_tools(IntentKind kind) {
    switch (kind) {
        case IntentKind::Arithmetic:
        case IntentKind::Clock:
        case IntentKind::Convert:
        case IntentKind::TextStats:
        case IntentKind::MemoryWrite:
        case IntentKind::ImageQuestion:
        case IntentKind::MultiStep:
            return true;
        default:
            return false;
    }
}

bool AgentRuntime::select_system_prompt(IntentKind kind) {
    const std::string mode = options_.config.system_prompt_mode;
    bool want_tools = true;
    if (mode == "minimal") {
        want_tools = false;
    } else if (mode == "adaptive") {
        want_tools = intent_needs_tools(kind);
    }
    system_prompt_ = want_tools ? build_system_prompt(options_.config, tools_,
                                                     options_.device, true)
                                : system_prompt_small_;
    if (want_tools == last_prompt_had_tools_) {
        return false;
    }
    last_prompt_had_tools_ = want_tools;
    system_prompt_tokens_ = llm_.count_tokens({{"system", system_prompt_}});
    return true;
}

void AgentRuntime::emit(const std::string& type, Json payload) {
    Json event = Json::object();
    event.set("type", Json::string(type));
    event.set("seq", Json::integer(++sequence_));
    event.set("ts_ms", Json::integer(now_ms()));
    event.set("session", Json::string(sessions_.meta().id));
    if (payload.is_object()) {
        for (const auto& member : payload.members()) {
            event.set(member.first, member.second);
        }
    }
    if (sink_) {
        sink_(event);
    }
}

void AgentRuntime::emit_error(const std::string& message, const std::string& stage) {
    Json payload = Json::object();
    payload.set("message", Json::string(message));
    payload.set("stage", Json::string(stage));
    emit("error", std::move(payload));
}

bool AgentRuntime::cancelled() const {
    if (cancel_requested_.load()) {
        return true;
    }
    return cancel_ && cancel_();
}

bool AgentRuntime::looks_like_broken_call(const std::string& output) const {
    const std::string text = trim(output);
    if (text.empty()) {
        return false;
    }
    const size_t separator = text.find_first_of(":{");
    if (separator == std::string::npos || separator == 0 || separator > 32) {
        return false;
    }
    const std::string head = trim(text.substr(0, separator));
    if (head.empty()) {
        return false;
    }
    for (const char character : head) {
        if (std::isalnum(static_cast<unsigned char>(character)) == 0 && character != '_') {
            return false;
        }
    }
    return !tools_.resolve(head, nullptr).empty();
}

bool AgentRuntime::looks_like_unusable_image_answer(const std::string& answer,
                                                  const IntentPolicy& intent) {
    if (looks_like_refusal(answer)) {
        return true;
    }
    if (intent.kind != IntentKind::ImageDescribe) {
        return false;   // other intents may legitimately answer briefly
    }
    const std::string trimmed = trim(answer);
    if (trimmed.empty()) {
        return true;
    }
    const char last = trimmed.back();
    if (last == '?' || (trimmed.size() >= 3 && trimmed.substr(trimmed.size() - 3) == "？")) {
        // "What do you want me to do with this image?" - the model bounced the
        // question back instead of describing what it can see.
        return true;
    }
    return utf8_length(trimmed) < 12;   // too short to be a description
}

std::string AgentRuntime::clean_answer(const std::string& answer) {
    std::string out = answer;
    static const char* kArtifacts[] = {"<|\"|>", "<bos>", "<eos>", "<pad>", "</b>", "<b>",
                                       "</i>", "<i>", "<|image|>", "<image|>", "<|image>",
                                       "<|turn>", "<turn|>"};
    for (const char* artifact : kArtifacts) {
        out = replace_all(out, artifact, "");
    }
    // A bare tool-protocol fragment such as "answer:当前时间…" loses its prefix.
    const std::string lowered = to_lower(out);
    if (lowered.rfind("answer:", 0) == 0) {
        out = out.substr(7);
    }
    out = trim(out);
    // "/image/analyze" style lines are model debris, not an answer.
    if (!out.empty() && out.front() == '/' && out.find(' ') == std::string::npos &&
        utf8_length(out) < 32) {
        out.clear();
    }
    return trim(out);
}

bool AgentRuntime::answer_mentions_memory(const std::string& answer,
                                          const std::vector<ScoredMemory>& memories) {
    const std::string lowered = to_lower(answer);
    for (const auto& memory : memories) {
        // Compare the *distinctive* tokens of the fact with the answer.  Single
        // CJK characters are not distinctive - "我叫Gem" shares 我/叫 with "我叫
        // 杨雷" and would look like a memory answer - so only bigrams (6 bytes)
        // and ASCII words of at least four characters count.
        const auto tokens = MemoryStore::tokenize(memory.record.text);
        int hits = 0;
        for (const auto& token : tokens) {
            const bool distinctive = token.size() >= 6 ||
                                     (token.size() >= 4 && token.find_first_of(
                                          "abcdefghijklmnopqrstuvwxyz") != std::string::npos);
            if (distinctive && contains(lowered, to_lower(token))) {
                ++hits;
            }
        }
        if (hits >= 2) {
            return true;
        }
    }
    return false;
}

std::string AgentRuntime::memory_answer(const std::vector<ScoredMemory>& memories) {
    std::string out;
    int count = 0;
    for (const auto& memory : memories) {
        if (count++ >= 3) {
            break;
        }
        out += "- (" + memory.record.kind + ") " + memory.record.text + "\n";
    }
    return trim(out);
}

bool AgentRuntime::mentions_image(const std::string& answer) {
    const std::string lowered = to_lower(answer);
    static const char* kWords[] = {"image", "picture", "photo", "photo you", "attachment",
                                   "图片", "照片", "图像", "这张图", "您上传"};
    for (const char* word : kWords) {
        if (contains(lowered, word) || contains(answer, word)) {
            return true;
        }
    }
    return false;
}

bool AgentRuntime::looks_like_refusal(const std::string& answer) {
    const std::string lowered = to_lower(answer);
    static const char* kRefusals[] = {
        "unable to process", "cannot process", "can not process", "can't process",
        "unable to help",   "cannot help",   "can't help",       "i am unable",
        "i'm unable",       "无法处理",       "无法回答",          "不能处理",
        "抱歉，我无法",
        // Denials that the model cannot see an image that is actually in its
        // input - the phone produced exactly this one.
        "无法看到",   "看不到",     "不能看到",   "无法查看",     "不能查看",
        "没有看到",   "看不到您上传", "没有图片",   "看不到图片",
        "cannot see", "can't see", "unable to see", "do not see", "don't see",
        "no image",   "text processing model", "text-only model", "text based model",
        "文本处理模型", "文本模型", "纯文本模型",
        // Refusals of a result the runtime just handed over (the phone log had
        // "I can't do that. I do not have access to the current time." right
        // after the now tool answered).
        "can't do that",  "cannot do that", "can not do that", "don't have access",
        "do not have access", "no access to", "don't have that information",
        "do not have that information", "cannot fulfill", "can't fulfill",
        "我没有访问", "无法访问",
    };
    for (const char* phrase : kRefusals) {
        if (contains(lowered, phrase)) {
            return true;
        }
    }
    return false;
}

int AgentRuntime::reserve_tokens() const {
    return std::max(64, options_.config.max_answer_tokens);
}

int AgentRuntime::history_budget_tokens() const {
    if (options_.config.history_budget_tokens > 0) {
        return options_.config.history_budget_tokens;
    }
    // Start from the prompt budget, subtract the fixed part (the system prompt
    // with its tool declarations) and one turn's own content (user message, one
    // or two observations and the answer) because compaction cannot shrink
    // those; whatever is left is what the transcript may occupy.
    const int turn_allowance = 420;
    const int available =
        options_.config.context_budget_tokens - system_prompt_tokens_ - turn_allowance;
    return std::max(128, available);
}

Json AgentRuntime::capabilities() const {
    Json out = Json::object();
    out.set("agent", Json::string("gemma4-e2b-on-device-agent"));
    out.set("version", Json::string("1.0.0"));
    out.set("native", Json::boolean(true));
    out.set("local", Json::boolean(true));
    Json functions = Json::array();
    for (const auto& definition : tools_.definitions()) {
        Json entry = Json::object();
        entry.set("name", Json::string(definition.spec.name));
        entry.set("description", Json::string(definition.spec.description));
        entry.set("parameters", definition.spec.parameters);
        functions.push(std::move(entry));
    }
    out.set("functions", std::move(functions));
    out.set("config", options_.config.describe());
    out.set("device", options_.device);
    out.set("llm", llm_.describe());
    out.set("memory_records", Json::integer(static_cast<long long>(memory_.size())));
    out.set("sessions", Json::integer(static_cast<long long>(sessions_.index().size())));
    out.set("session", Json::string(sessions_.meta().id));
    out.set("system_prompt_chars", Json::integer(static_cast<long long>(system_prompt_.size())));
    return out;
}

Json AgentRuntime::status() const {
    Json out = Json::object();
    out.set("ok", Json::boolean(true));
    out.set("session", Json::string(sessions_.meta().id));
    out.set("session_title", Json::string(sessions_.meta().title));
    out.set("turns", Json::integer(sessions_.meta().turns));
    out.set("compactions", Json::integer(sessions_.meta().compactions));
    out.set("summary_upto", Json::integer(sessions_.meta().summary_upto));
    out.set("memory_records", Json::integer(static_cast<long long>(memory_.size())));
    out.set("context_tokens", Json::integer(sessions_.transcript_tokens(reserve_tokens())));
    out.set("context_budget", Json::integer(options_.config.context_budget_tokens));
    out.set("history_budget", Json::integer(history_budget_tokens()));
    out.set("system_prompt_tokens", Json::integer(system_prompt_tokens_));
    out.set("kv_dirty", Json::boolean(kv_dirty_));
    out.set("needs_full_prefill", Json::boolean(needs_full_prefill_));
    return out;
}

Json AgentRuntime::handle(const Json& request) {
    Json document = request;
    if (document.is_string()) {
        Json parsed = Json::parse(document.as_string());
        if (parsed.is_object()) {
            document = std::move(parsed);
        }
    }
    if (!document.is_object()) {
        Json out = Json::object();
        out.set("ok", Json::boolean(false));
        out.set("error", Json::string("request must be a JSON object"));
        return out;
    }
    const std::string type = document.string_or("type", "turn");

    if (type == "turn" || type == "ask") {
        return run_turn(document);
    }
    if (type == "status" || type == "capabilities") {
        Json out = capabilities();
        emit("capabilities", out);
        return out;
    }
    if (type == "new_session") {
        const std::string title = document.string_or("title", "");
        const std::string id = document.string_or("session", "");
        sessions_.create(title, id);
        needs_full_prefill_ = true;
        Json opened = Json::object();
        opened.set("session", Json::string(sessions_.meta().id));
        opened.set("title", Json::string(sessions_.meta().title));
        opened.set("created_ms", Json::integer(sessions_.meta().created_ms));
        emit("session_opened", std::move(opened));
        Json out = Json::object();
        out.set("ok", Json::boolean(true));
        out.set("session", Json::string(sessions_.meta().id));
        out.set("title", Json::string(sessions_.meta().title));
        return out;
    }
    if (type == "open_session") {
        const std::string id = document.string_or("session", "");
        if (id.empty() || !sessions_.open(id)) {
            Json out = Json::object();
            out.set("ok", Json::boolean(false));
            out.set("error", Json::string("unknown session " + id));
            return out;
        }
        needs_full_prefill_ = true;
        Json opened = Json::object();
        opened.set("session", Json::string(sessions_.meta().id));
        opened.set("title", Json::string(sessions_.meta().title));
        opened.set("turns", Json::integer(sessions_.meta().turns));
        opened.set("summary_upto", Json::integer(sessions_.meta().summary_upto));
        emit("session_opened", std::move(opened));
        Json out = Json::object();
        out.set("ok", Json::boolean(true));
        out.set("session", Json::string(sessions_.meta().id));
        out.set("title", Json::string(sessions_.meta().title));
        return out;
    }
    if (type == "delete_session") {
        const std::string id = document.string_or("session", "");
        const bool removed = !id.empty() && sessions_.remove(id);
        if (removed && !sessions_.is_open()) {
            if (!sessions_.index().empty()) {
                sessions_.open(sessions_.index().front().id);
            } else {
                sessions_.create("New session");
            }
            needs_full_prefill_ = true;
        }
        Json out = Json::object();
        out.set("ok", Json::boolean(removed));
        out.set("session", Json::string(sessions_.meta().id));
        return out;
    }
    if (type == "rename_session") {
        const std::string id = document.string_or("session", sessions_.meta().id);
        const std::string title = document.string_or("title", "");
        const bool renamed = sessions_.rename(id, title);
        Json out = Json::object();
        out.set("ok", Json::boolean(renamed));
        return out;
    }
    if (type == "delete_all_sessions") {
        // One tap in the UI: wipe the conversation history and start a fresh
        // session so the agent stays usable.
        std::vector<std::string> ids;
        for (const auto& meta : sessions_.index()) {
            ids.push_back(meta.id);
        }
        int removed = 0;
        for (const auto& id : ids) {
            if (sessions_.remove(id)) {
                ++removed;
            }
        }
        sessions_.create("New session");
        needs_full_prefill_ = true;
        Json opened = Json::object();
        opened.set("session", Json::string(sessions_.meta().id));
        opened.set("title", Json::string(sessions_.meta().title));
        emit("session_opened", std::move(opened));
        Json out = Json::object();
        out.set("ok", Json::boolean(true));
        out.set("removed", Json::integer(removed));
        out.set("session", Json::string(sessions_.meta().id));
        return out;
    }
    if (type == "list_sessions") {
        Json out = Json::object();
        out.set("ok", Json::boolean(true));
        Json list = Json::array();
        for (const auto& meta : sessions_.index()) {
            Json entry = SessionStore::meta_to_json(meta);
            entry.set("current", Json::boolean(meta.id == sessions_.meta().id));
            list.push(std::move(entry));
        }
        out.set("sessions", std::move(list));
        out.set("current", Json::string(sessions_.meta().id));
        return out;
    }
    if (type == "history") {
        const std::string requested = document.string_or("session", "");
        if (!requested.empty() && requested != sessions_.meta().id) {
            sessions_.open(requested);
        }
        Json out = Json::object();
        out.set("ok", Json::boolean(true));
        out.set("session", Json::string(sessions_.meta().id));
        out.set("title", Json::string(sessions_.meta().title));
        out.set("summary", Json::string(sessions_.meta().summary));
        out.set("summary_upto", Json::integer(sessions_.meta().summary_upto));
        out.set("compactions", Json::integer(sessions_.meta().compactions));
        Json turns = Json::array();
        for (const auto& turn : sessions_.turns()) {
            turns.push(SessionStore::to_json(turn));
        }
        out.set("turns", std::move(turns));
        out.set("context_tokens", Json::integer(sessions_.transcript_tokens(reserve_tokens())));
        out.set("context_budget", Json::integer(options_.config.context_budget_tokens));
        return out;
    }
    if (type == "list_memories") {
        Json out = Json::object();
        out.set("ok", Json::boolean(true));
        Json items = Json::array();
        std::vector<MemoryRecord> records = memory_.records();
        std::sort(records.begin(), records.end(), [](const MemoryRecord& left,
                                                     const MemoryRecord& right) {
            if (left.pinned != right.pinned) {
                return left.pinned;
            }
            return left.updated_ms > right.updated_ms;
        });
        for (const auto& record : records) {
            items.push(MemoryStore::to_json(record));
        }
        out.set("memories", std::move(items));
        out.set("count", Json::integer(static_cast<long long>(memory_.size())));
        return out;
    }
    if (type == "delete_memory") {
        const std::string id = document.string_or("id", "");
        const bool removed = !id.empty() && memory_.remove(id);
        if (removed) {
            memory_.save();
        }
        Json out = Json::object();
        out.set("ok", Json::boolean(removed));
        out.set("id", Json::string(id));
        return out;
    }
    if (type == "clear_memories") {
        const std::string kind = document.string_or("kind", "");
        const int removed = memory_.clear(kind, document.bool_or("keep_pinned", false));
        memory_.save();
        Json out = Json::object();
        out.set("ok", Json::boolean(true));
        out.set("removed", Json::integer(removed));
        return out;
    }
    if (type == "remember") {
        MemoryRecord record;
        record.text = document.string_or("text", "");
        record.kind = document.string_or("kind", "note");
        record.importance = document.number_or("importance", 0.8);
        record.pinned = document.bool_or("pinned", true);
        bool merged = false;
        const std::string id = memory_.add(record, &merged);
        memory_.save();
        Json out = Json::object();
        out.set("ok", Json::boolean(!id.empty()));
        out.set("id", Json::string(id));
        out.set("merged", Json::boolean(merged));
        return out;
    }
    if (type == "compact") {
        compact_history(true);
        Json out = Json::object();
        out.set("ok", Json::boolean(true));
        out.set("summary_upto", Json::integer(sessions_.meta().summary_upto));
        out.set("compactions", Json::integer(sessions_.meta().compactions));
        return out;
    }
    if (type == "reset") {
        std::string error;
        if (!llm_.reset_conversation(&error)) {
            Json out = Json::object();
            out.set("ok", Json::boolean(false));
            out.set("error", Json::string(error));
            return out;
        }
        kv_dirty_ = false;
        needs_full_prefill_ = true;
        Json out = Json::object();
        out.set("ok", Json::boolean(true));
        return out;
    }
    if (type == "config") {
        // Runtime toggles the app exposes (memory / conversation history).
        if (document.has("memory")) {
            options_.config.memory_enabled = document.bool_or("memory", true);
        }
        if (document.has("history")) {
            const bool include = document.bool_or("history", true);
            if (include != options_.config.include_history) {
                options_.config.include_history = include;
                needs_full_prefill_ = true;   // the prompt shape changed
            }
        }
        if (document.has("session_notes")) {
            options_.config.session_notes = document.bool_or("session_notes", true);
        }
        Json out = Json::object();
        out.set("ok", Json::boolean(true));
        out.set("memory", Json::boolean(options_.config.memory_enabled));
        out.set("history", Json::boolean(options_.config.include_history));
        out.set("session_notes", Json::boolean(options_.config.session_notes));
        emit("options", out);
        return out;
    }
    if (type == "cancel") {
        cancel_requested_.store(true);
        Json out = Json::object();
        out.set("ok", Json::boolean(true));
        return out;
    }

    Json out = Json::object();
    out.set("ok", Json::boolean(false));
    out.set("error", Json::string("unknown request type '" + type + "'"));
    return out;
}

std::vector<ScoredMemory> AgentRuntime::recall(const std::string& query) const {
    if (!options_.config.memory_enabled) {
        return {};
    }
    std::vector<ScoredMemory> hits = memory_.search(query, options_.config.recall_top_k);
    // A fact this conversation just established beats an older session's answer to
    // the same question: the phone asked "我最喜欢的颜色是什么？" right after
    // storing 紫色 and answered 绿色, a colour from an earlier session that the
    // store still held.  Sessions are long-lived on the phone, so the current one
    // gets a clear preference.
    const std::string current_session = sessions_.meta().id;
    for (auto& hit : hits) {
        if (!current_session.empty() && hit.record.session_id == current_session) {
            hit.score *= 1.6;
        }
    }
    std::stable_sort(hits.begin(), hits.end(),
                     [](const ScoredMemory& left, const ScoredMemory& right) {
                         return left.score > right.score;
                     });
    const auto pinned = memory_.pinned();
    for (const auto& record : pinned) {
        const bool present = std::any_of(hits.begin(), hits.end(),
                                         [&](const ScoredMemory& hit) {
                                             return hit.record.id == record.id;
                                         });
        if (present) {
            continue;
        }
        ScoredMemory entry;
        entry.record = record;
        entry.score = 0.5 + 0.5 * record.importance;
        entry.lexical = 0.0;
        entry.recency = 1.0;
        hits.push_back(std::move(entry));
    }
    if (static_cast<int>(hits.size()) > options_.config.recall_top_k + 4) {
        hits.resize(static_cast<size_t>(options_.config.recall_top_k + 4));
    }
    return hits;
}

ChatMessages AgentRuntime::render_history(const ChatMessages& tail,
                                         std::vector<ImageSplice>* splices,
                                         const std::string& current_canvas,
                                         int current_tokens) {
    // The history is rendered VERBATIM.  Rewriting it (folding tool detail,
    // dropping old images, trimming turns) changes the middle of the prompt,
    // which invalidates the KV prefix and forces a full prefill - the phone did
    // that on every single turn (68-128 s per answer).  All rewriting now
    // happens once per compaction instead (see compact_history).
    ChatMessages messages;
    messages.emplace_back("system", system_prompt_);
    if (!options_.config.include_history) {
        // Fresh-context mode: no summary, no notes, no earlier turns - just the
        // current message (and this turn's own tool exchanges).
        messages.insert(messages.end(), tail.begin(), tail.end());
        return messages;
    }
    if (!sessions_.meta().summary.empty()) {
        messages.emplace_back("user", "CONVERSATION SUMMARY (earlier turns, already "
                                      "completed):\n" + sessions_.meta().summary);
    }
    bool transcript_changed = false;
    for (auto& turn : sessions_.mutable_turns()) {
        if (turn.index <= sessions_.meta().summary_upto) {
            continue;
        }
        const std::string cached_canvas =
            turn.image_file.empty()
                ? std::string()
                : path_join(path_join(sessions_.directory_hint(), sessions_.meta().id),
                            turn.image_file);
        const bool is_current_turn = &turn == &sessions_.mutable_turns().back();
        const bool use_current = !current_canvas.empty() && current_tokens > 0 && is_current_turn;
        // An image is injected only in the turn that carried it.  Older image
        // turns keep a text description instead (no soft tokens, no vision
        // re-run, ~130 tokens smaller prompt per old picture).
        const bool keep_image = use_current || (options_.config.reuse_history_images &&
                                                !cached_canvas.empty() &&
                                                file_exists(cached_canvas));
        const std::string canvas = use_current ? current_canvas : cached_canvas;
        const int tokens = use_current ? current_tokens : turn.image_tokens;
        Json entries = message_entries(turn);
        Json updated_entries = Json::array();
        bool rewrote_turn = false;
        for (const auto& entry : entries.items()) {
            const std::string role = entry.string_or("role", "");
            std::string content = entry.string_or("content", "");
            if (role.empty()) {
                continue;
            }
            if (count_image_tokens(content) > 0) {
                if (keep_image) {
                    if (splices != nullptr) {
                        ImageSplice splice;
                        splice.message_index = static_cast<int>(messages.size());
                        splice.canvas_path = canvas;
                        splice.tokens = tokens;
                        splices->push_back(std::move(splice));
                    }
                } else {
                    // The picture is only in its own turn: substitute a text note
                    // that carries what the model said about it, persist it, and
                    // every later turn renders the same (small) prompt.
                    std::string note = turn.image_note;
                    if (!turn.answer.empty()) {
                        note += "; described as: " + head(collapse_whitespace(turn.answer), 160);
                    }
                    content = strip_placeholders(content, note);
                    rewrote_turn = true;
                }
            }
            messages.emplace_back(role, content);
            Json kept = Json::object();
            kept.set("role", Json::string(role));
            kept.set("content", Json::string(content));
            updated_entries.push(std::move(kept));
        }
        if (rewrote_turn) {
            turn.messages = std::move(updated_entries);
            transcript_changed = true;
        }
    }
    if (transcript_changed) {
        sessions_.rewrite_turns();
        degraded_history_ = true;
    }
    messages.insert(messages.end(), tail.begin(), tail.end());
    return messages;
}

bool AgentRuntime::auxiliary_generate(const ChatMessages& messages, int max_tokens,
                                     std::string* text, GenerationMetrics* metrics) {
    std::string error;
    // generate_full always starts from a pristine KV cache, so the auxiliary
    // conversation can never leak into the user's transcript.
    std::string output = llm_.generate_full(
        messages, {}, max_tokens, [](int, const std::string&) { return true; }, metrics, &error);
    kv_dirty_ = true;
    needs_full_prefill_ = true;
    if (!error.empty() && output.empty()) {
        emit_error("auxiliary generation failed: " + error, "aux");
        return false;
    }
    // No extra reset: `needs_full_prefill_` makes the next user-facing
    // generation start from a pristine engine anyway, and on the device every
    // engine re-creation costs ~1.9 s.
    if (text != nullptr) {
        *text = output;
    }
    return true;
}

void AgentRuntime::compact_history(bool force) {
    if (!options_.config.memory_enabled && !force) {
        return;
    }
    const int budget = history_budget_tokens();
    if (!force && !sessions_.needs_compaction(budget, 0)) {
        return;
    }
    auto candidates = sessions_.compaction_candidates();
    if (candidates.empty()) {
        return;
    }
    if (!force && static_cast<int>(candidates.size()) < options_.config.compaction_min_turns) {
        // Folding one or two turns does not free enough context to be worth a
        // summary generation and an engine re-creation; the verbatim window and
        // the session notes carry them instead.
        return;
    }
    std::string transcript;
    for (const auto& turn : candidates) {
        transcript += "USER: " + turn.user_text + "\n";
        if (!turn.image_note.empty()) {
            transcript += "(image: " + turn.image_note + ")\n";
        }
        transcript += "ASSISTANT: " + turn.answer + "\n";
    }
    if (!sessions_.meta().summary.empty()) {
        transcript = "EXISTING SUMMARY:\n" + sessions_.meta().summary + "\n\nNEW TURNS:\n" +
                     transcript;
    }
    const ChatMessages prompt = {
        {"system", "You compress conversations for an on-device assistant."},
        {"user", build_compaction_prompt(transcript, options_.summary_max_bullets)}};
    GenerationMetrics metrics;
    std::string summary;
    if (!auxiliary_generate(prompt, 220, &summary, &metrics)) {
        return;
    }
    summary = trim(summary);
    if (summary.empty()) {
        return;
    }
    const int upto = candidates.back().index;
    sessions_.apply_compaction(summary, upto);
    needs_full_prefill_ = true;
    Json payload = Json::object();
    payload.set("upto_turn", Json::integer(upto));
    payload.set("turns", Json::integer(static_cast<long long>(candidates.size())));
    payload.set("summary", Json::string(summary));
    payload.set("generated_tokens", Json::integer(metrics.generated_tokens));
    payload.set("duration_ms", Json::number(static_cast<double>(metrics.decode_ms)));
    emit("context_compacted", std::move(payload));
}

void AgentRuntime::update_session_notes(const SessionTurn& turn) {
    if (!options_.config.session_notes) {
        return;
    }
    // One bullet per turn, and only for turns that left something behind: what an
    // image showed, what was stored in long term memory, or which tools ran.  The
    // digest is bounded, so it can be injected on every later turn for the price
    // of a couple of hundred tokens - and it is what keeps an older image usable
    // after its soft tokens left the history window.
    std::string bullet;
    if (turn.has_image && !turn.answer.empty()) {
        bullet = "[image t" + std::to_string(turn.index) + "] " +
                 head(collapse_whitespace(turn.answer), 140);
    } else if (!turn.memories_written.empty()) {
        bullet = "[memory t" + std::to_string(turn.index) + "] " +
                 head(collapse_whitespace(turn.user_text), 100) +
                 (turn.answer.empty() ? "" : " -> " + head(collapse_whitespace(turn.answer), 80));
    } else if (!turn.tool_calls.items().empty()) {
        std::string tools;
        for (const auto& call : turn.tool_calls.items()) {
            if (!tools.empty()) {
                tools += ", ";
            }
            tools += call.string_or("name", "?");
        }
        bullet = "[tools t" + std::to_string(turn.index) + "] " + tools + " -> " +
                 head(collapse_whitespace(turn.answer), 100);
    }
    if (bullet.empty()) {
        return;
    }
    std::vector<std::string> lines = split(sessions_.meta().notes, '\n');
    lines.erase(std::remove_if(lines.begin(), lines.end(),
                               [](const std::string& line) { return trim(line).empty(); }),
                lines.end());
    lines.push_back("- " + bullet);
    while (static_cast<int>(lines.size()) > options_.config.session_notes_lines) {
        lines.erase(lines.begin());
    }
    std::string notes;
    for (const auto& line : lines) {
        notes += line + "\n";
    }
    if (static_cast<int>(notes.size()) > options_.config.session_notes_chars) {
        notes = tail(notes, static_cast<size_t>(options_.config.session_notes_chars));
        const size_t newline = notes.find('\n');
        if (newline != std::string::npos) {
            notes = notes.substr(newline + 1);
        }
    }
    sessions_.mutable_meta().notes = notes;
    sessions_.persist_meta();
    Json payload = Json::object();
    payload.set("turn", Json::integer(turn.index));
    payload.set("notes", Json::string(notes));
    emit("session_notes", std::move(payload));
}

void AgentRuntime::write_memories(const std::string& user_text, const std::string& answer,
                                 SessionTurn* turn) {
    if (!options_.config.memory_enabled) {
        return;
    }
    auto extracted = extract_memories(user_text, answer, options_.config.language);
    if (options_.model_memory_extraction) {
        const ChatMessages prompt = {
            {"system", "You extract durable memory records as JSON."},
            {"user", build_memory_prompt(user_text, answer)}};
        GenerationMetrics metrics;
        std::string response;
        if (auxiliary_generate(prompt, 160, &response, &metrics)) {
            auto model_memories = parse_memory_json(response);
            if (!model_memories.empty()) {
                extracted = std::move(model_memories);
            }
        }
    }
    for (const auto& memory : extracted) {
        // Episodes describe *this conversation*; they belong to the session
        // transcript (searchable through search_history), not to the long term
        // store.  Only durable facts, preferences and notes are promoted.
        if (memory.kind == "episode" && !options_.config.store_episodes) {
            continue;
        }
        MemoryRecord record;
        record.text = memory.text;
        record.kind = memory.kind;
        record.importance = memory.importance;
        record.pinned = memory.pinned;
        record.correction = memory.correction;
        record.session_id = sessions_.meta().id;
        record.turn_index = turn->index;
        bool merged = false;
        std::string action;
        const std::string id = memory_.add(record, &merged, &action);
        if (id.empty()) {
            continue;
        }
        if (turn != nullptr) {
            turn->memories_written.push_back(id);
        }
        Json payload = Json::object();
        payload.set("id", Json::string(id));
        payload.set("kind", Json::string(memory.kind));
        payload.set("text", Json::string(record.text));
        payload.set("importance", Json::number(memory.importance));
        payload.set("merged", Json::boolean(merged));
        payload.set("action", Json::string(action));
        payload.set("source", Json::string("turn"));
        emit("memory_write", std::move(payload));
    }
    memory_.save();
}

std::string AgentRuntime::session_image_dir() const {
    return path_join(path_join(sessions_.directory_hint(), sessions_.meta().id), "images");
}

bool AgentRuntime::cache_canvas(SessionTurn* turn, const std::string& canvas_path) {
    if (canvas_path.empty() || !file_exists(canvas_path)) {
        return false;
    }
    const std::string directory = session_image_dir();
    if (!make_directories(directory)) {
        return false;
    }
    std::string payload;
    if (!read_file(canvas_path, &payload)) {
        return false;
    }
    const std::string relative = "images/turn-" + std::to_string(turn->index) + ".bin";
    const std::string target = path_join(path_join(sessions_.directory_hint(),
                                                   sessions_.meta().id),
                                         relative);
    if (!write_file(target, payload)) {
        return false;
    }
    turn->image_file = relative;
    // Keep only the canvases that can still appear in the retained context.
    const int keep_from = std::max(
        1, sessions_.meta().turns + 1 - options_.image_cache_turns);
    for (const auto& name : list_directory(directory)) {
        if (!starts_with(name, "turn-")) {
            continue;
        }
        int index = 0;
        try {
            index = std::stoi(name.substr(5));
        } catch (...) {
            continue;
        }
        if (index < keep_from) {
            remove_file(path_join(directory, name));
        }
    }
    return true;
}

std::string AgentRuntime::describe_image_directly(const std::string& canvas_path,
                                                 const ImageInput& image) {
    if (canvas_path.empty() || image.tokens <= 0) {
        return std::string();
    }
    // Exactly the prompt of the shipped demo (see README: "Describe this
    // image."), with the same soft tokens the agent turn already computed.
    agent::ChatMessages prompt = {
        {"user", build_user_message("Describe this image.", {}, image.tokens)}};
    std::vector<ImageSplice> splices;
    ImageSplice splice;
    splice.message_index = 0;
    splice.canvas_path = canvas_path;
    splice.tokens = image.tokens;
    splices.push_back(std::move(splice));

    GenerationMetrics metrics;
    std::string error;
    const std::string text =
        llm_.generate_full(prompt, splices, options_.config.max_answer_tokens,
                           [](int, const std::string&) { return true; }, &metrics, &error);
    if (!error.empty() && text.empty()) {
        emit_error("vision fallback failed: " + error, "vision_fallback");
        return std::string();
    }
    return trim(strip_thinking(text));
}

Json AgentRuntime::run_turn(const Json& request) {
    const std::string requested_session = request.string_or("session", "");
    if (!requested_session.empty() && requested_session != sessions_.meta().id) {
        if (sessions_.open(requested_session)) {
            needs_full_prefill_ = true;
        }
    }
    if (!sessions_.is_open()) {
        sessions_.create("New session");
        needs_full_prefill_ = true;
    }
    cancel_requested_.store(false);

    std::string text = trim(request.string_or("text", ""));
    std::string image_path = request.string_or("image", "");
    if (!image_path.empty() && image_path[0] != '/') {
        image_path = path_join(options_.root, image_path);
    }
    if (text.empty() && image_path.empty()) {
        emit_error("the request has neither text nor an image", "turn");
        Json out = Json::object();
        out.set("ok", Json::boolean(false));
        out.set("error", Json::string("empty turn"));
        return out;
    }

    const long long turn_begin = now_ms();
    const long long turn_id = ++turn_sequence_;
    SessionTurn turn;
    turn.index = sessions_.meta().turns + 1;
    turn.started_ms = turn_begin;
    turn.user_text = text.empty() ? std::string("[image only]") : text;

    Json start = Json::object();
    start.set("turn", Json::integer(turn.index));
    start.set("turn_id", Json::integer(turn_id));
    start.set("text", Json::string(turn.user_text));
    start.set("has_image", Json::boolean(!image_path.empty()));
    start.set("mode", Json::string(needs_full_prefill_ ? "full" : "delta"));
    emit("turn_start", std::move(start));

    // --- vision -------------------------------------------------------------
    ImageInput image;
    if (!image_path.empty()) {
        std::string error;
        if (!llm_.inspect_image(image_path, &image, &error)) {
            emit_error("cannot read the image: " + error, "vision");
            image = ImageInput();
        } else {
            turn.has_image = true;
            turn.image_tokens = image.tokens;
            turn.image_note = image.note;
            cache_canvas(&turn, image_path);
            Json payload = Json::object();
            payload.set("tokens", Json::integer(image.tokens));
            payload.set("canvas_width", Json::integer(image.canvas_width));
            payload.set("canvas_height", Json::integer(image.canvas_height));
            payload.set("note", Json::string(image.note));
            payload.set("sha256", Json::string(image.sha256));
            emit("image", std::move(payload));
        }
    }

    // --- intent understanding ------------------------------------------------
    // The router decides what this turn is about and, with it, the policy: does
    // it consult long term memory, are tools expected, must the answer stay
    // plain text?  Everything below follows that policy instead of treating
    // every message the same way.
    const IntentPolicy intent = classify_intent(turn.user_text, turn.has_image, tools_);
    Json intent_event = Json::object();
    intent_event.set("kind", Json::string(to_string(intent.kind)));
    intent_event.set("label", Json::string(intent.label));
    intent_event.set("confidence", Json::number(intent.confidence));
    intent_event.set("memory", Json::boolean(intent.inject_memory));
    intent_event.set("tools", Json::boolean(intent.allow_tools));
    intent_event.set("plain_text", Json::boolean(intent.prefer_plain_answer));
    intent_event.set("preferred_tool", Json::string(intent.preferred_tool));
    emit("intent", std::move(intent_event));
    if (select_system_prompt(intent.kind)) {
        // The prompt variant changed: the whole KV prefix is different now.
        needs_full_prefill_ = true;
    }

    // --- memory recall (only when the intent asks for it) --------------------
    std::string query = turn.user_text;
    if (!turn.image_note.empty()) {
        query += " " + turn.image_note;
    }
    const std::vector<ScoredMemory> recalled =
        intent.inject_memory ? recall(query) : std::vector<ScoredMemory>();
    Json recall_payload = Json::object();
    Json recall_items = Json::array();
    std::vector<std::string> accessed;
    for (const auto& memory : recalled) {
        recall_items.push(memory_to_json(memory));
        accessed.push_back(memory.record.id);
    }
    recall_payload.set("items", std::move(recall_items));
    recall_payload.set("count", Json::integer(static_cast<long long>(recalled.size())));
    recall_payload.set("query", Json::string(head(query, 160)));
    recall_payload.set("records", Json::integer(static_cast<long long>(memory_.size())));
    emit("recall", std::move(recall_payload));
    if (!accessed.empty()) {
        memory_.mark_accessed(accessed);
    }
    turn.recalled = Json::array();
    for (const auto& memory : recalled) {
        Json entry = Json::object();
        entry.set("id", Json::string(memory.record.id));
        entry.set("kind", Json::string(memory.record.kind));
        entry.set("text", Json::string(memory.record.text));
        entry.set("score", Json::number(memory.score));
        turn.recalled.push(std::move(entry));
    }

    // --- context ------------------------------------------------------------
    // The native routing hint sometimes spells out the exact call (arithmetic,
    // clock).  Keep the parsed form as a deterministic fallback: the deployed
    // Q4 export regularly drops the arguments even when the line is in front of
    // it, and a tool result is what the user actually asked for.
    ToolCall hinted = intent.seed;
    bool hint_used = false;
    std::string notes_block;
    // The Notetaker digest keeps an older image or tool result usable, but on a
    // turn whose answer the runtime can already spell out (a calculation, the
    // clock, a conversion, a character count) it is pure noise: the evaluation
    // A/B caught an arithmetic turn that answered *nothing* with the notes in
    // the prompt and `42+16 = 58` with them removed.
    const bool value_turn = intent.kind == IntentKind::Arithmetic ||
                            intent.kind == IntentKind::Clock ||
                            intent.kind == IntentKind::Convert ||
                            intent.kind == IntentKind::TextStats;
    if (options_.config.session_notes && !value_turn && !sessions_.meta().notes.empty()) {
        notes_block = sessions_.meta().notes;
    }
    const std::string user_message = build_user_message(
        turn.user_text, recalled, turn.has_image ? image.tokens : 0, notes_block);
    Json user_entry = Json::object();
    user_entry.set("role", Json::string("user"));
    user_entry.set("content", Json::string(user_message));
    turn.messages.push(user_entry);
    // One turn is a normal alternating transcript: the model asks for a tool,
    // the tool answer is delivered as a user message, and the loop continues.
    // This keeps the rendered prompt a strict prefix extension of the previous
    // one, which is what MNN's prompt cache needs to prefill only the suffix.
    auto rebuild_working = [&]() {
        ChatMessages working = render_history({}, nullptr, std::string(), 0);
        for (const auto& entry : turn.messages.items()) {
            if (!entry.is_object()) {
                continue;
            }
            const std::string role = entry.string_or("role", "");
            const std::string content = entry.string_or("content", "");
            if (role.empty()) {
                continue;
            }
            working.emplace_back(role, content);
        }
        return working;
    };
    ChatMessages working = rebuild_working();
    const int image_message_index = static_cast<int>(working.size()) - 1;
    int context_tokens = llm_.count_tokens(working);
    const int history_tokens = sessions_.transcript_tokens(0);
    const bool over_budget =
        context_tokens + reserve_tokens() > options_.config.context_budget_tokens;
    const bool history_over = history_tokens > history_budget_tokens();
    Json context_payload = Json::object();
    context_payload.set("tokens", Json::integer(context_tokens));
    context_payload.set("budget", Json::integer(options_.config.context_budget_tokens));
    context_payload.set("reserve", Json::integer(reserve_tokens()));
    context_payload.set("history_turns", Json::integer(sessions_.meta().turns));
    context_payload.set("summary_upto", Json::integer(sessions_.meta().summary_upto));
    context_payload.set("history_tokens", Json::integer(history_tokens));
    context_payload.set("history_budget", Json::integer(history_budget_tokens()));
    context_payload.set("compacted", Json::boolean(over_budget || history_over));
    emit("context", std::move(context_payload));
    if (over_budget || history_over) {
        compact_history(true);
        working = rebuild_working();
        context_tokens = llm_.count_tokens(working);
    }

    // The image descriptions travel with every call of this turn: the driver
    // needs them whenever it (re)does a full prefill - the first step, a KV
    // state that no longer matches, or the wrap-up - and passing them costs
    // nothing otherwise.
    std::vector<ImageSplice> turn_splices;
    auto collect_splices = [&]() {
        turn_splices.clear();
        render_history({}, &turn_splices, std::string(), 0);
        if (image.valid) {
            ImageSplice splice;
            splice.message_index = image_message_index;
            splice.canvas_path = image_path;
            splice.tokens = image.tokens;
            turn_splices.push_back(std::move(splice));
        }
    };
    collect_splices();
    if (degraded_history_) {
        // The transcript was rewritten (an image left its window, or an old tool
        // exchange was folded), so the KV cache no longer matches it.
        degraded_history_ = false;
        needs_full_prefill_ = true;
    }

    // --- plan / tool loop ---------------------------------------------------
    planner_.begin_turn(turn.user_text);
    int tool_calls = 0;
    int step_index = 0;
    std::string answer;
    bool truncated = false;
    bool cancelled_turn = false;
    bool answer_generated = false;
    bool answer_verbatim = false;
    bool retried_empty = false;
    bool retried_broken_call = false;
    bool retried_refusal = false;
    bool retried_greeting = false;
    int language_retries = 0;
    // True when the assistant turn that ends the transcript is a cleaned copy of
    // this turn's own generation: the driver may then re-anchor the KV cache on
    // the transcript instead of paying a full re-prefill next turn.
    bool transcript_matches_kv = false;
    std::string last_error_signature;
    std::string last_tool_summary;
    TokenUsage usage;
    GenerationMetrics last_metrics;
    long long first_token_ms = 0;
    std::string stop_reason;

    while (step_index < options_.config.max_steps) {
        if (cancelled()) {
            cancelled_turn = true;
            stop_reason = "cancelled";
            break;
        }
        ++step_index;
        // An image turn is only forced onto the pristine-engine path when the
        // deployment asks for it (`image_delta_prefill=false`): the driver can
        // otherwise prefill the image message as a suffix of the resident KV
        // cache and splice the vision vectors into it.
        const bool force_full =
            needs_full_prefill_ ||
            (image.valid && step_index == 1 && !options_.config.image_delta_prefill);
        const int max_tokens = options_.config.max_tokens_per_step;
        Json step_start = Json::object();
        step_start.set("step", Json::integer(step_index));
        step_start.set("mode", Json::string(force_full ? "full" : "delta"));
        step_start.set("splices", Json::integer(static_cast<long long>(turn_splices.size())));
        step_start.set("max_tokens", Json::integer(max_tokens));
        step_start.set("context_tokens", Json::integer(context_tokens));
        emit("step_start", std::move(step_start));

        GenerationMetrics metrics;
        std::string error;
        const long long step_begin = now_ms();
        auto on_token = [&](int index, const std::string& piece) -> bool {
            if (first_token_ms == 0) {
                first_token_ms = now_ms();
            }
            if (!options_.config.stream_steps) {
                return !cancelled();
            }
            Json payload = Json::object();
            payload.set("step", Json::integer(step_index));
            payload.set("index", Json::integer(index));
            payload.set("text_b64", Json::string(base64_encode(piece)));
            emit("step_delta", std::move(payload));
            return !cancelled();
        };
        std::string output;
        if (force_full) {
            output = llm_.generate_full(working, turn_splices, max_tokens, on_token, &metrics,
                                        &error);
        } else {
            output = llm_.generate_delta(working, turn_splices, max_tokens, on_token, &metrics,
                                         &error);
        }
        needs_full_prefill_ = false;
        // The deployed Q4 export sometimes writes its thinking channel as plain
        // text; it is stripped the same way the checkpoint's template strips it.
        output = strip_thinking(output);
        if (!output.empty()) {
            kv_dirty_ = true;
        }
        usage.prompt_tokens += metrics.prompt_tokens;
        usage.generated_tokens += metrics.generated_tokens;
        last_metrics = metrics;
        if (first_token_ms == 0) {
            first_token_ms = now_ms();
        }
        const long long step_ms = now_ms() - step_begin;
        if (!error.empty() && trim(output).empty()) {
            emit_error("generation failed: " + error, "generate");
            stop_reason = "error";
            break;
        }
        if (trim(output).empty() && !retried_empty && step_index < options_.config.max_steps) {
            // The model stopped without emitting anything, which happens after a
            // failed observation.  Nudge it once; the nudge is not part of the
            // transcript, so the next turn rebuilds its prompt from a pristine
            // engine instead of reusing a KV state that no longer matches.
            retried_empty = true;
            const std::string nudge =
                "Continue: either call the next tool, or answer the user in plain text. "
                "Do not reply with an empty message.";
            working.emplace_back("user", nudge);
            Json nudge_entry = Json::object();
            nudge_entry.set("role", Json::string("user"));
            nudge_entry.set("content", Json::string(nudge));
            turn.messages.push(std::move(nudge_entry));
            Json payload = Json::object();
            payload.set("step", Json::integer(step_index));
            payload.set("kind", Json::string("retry"));
            payload.set("text", Json::string(""));
            payload.set("note", Json::string("empty generation, nudging once"));
            emit("step_end", std::move(payload));
            continue;
        }

        ToolCall call = parse_tool_call(output, &tools_);
        if (!call.valid && !retried_broken_call && step_index < options_.config.max_steps &&
            looks_like_broken_call(output)) {
            // The model wrote something tool-shaped that could not be executed -
            // on the phone this happens when it tries to pass the image bytes to
            // image_info instead of answering.  Ask once for plain text; the
            // nudge stays out of the transcript.
            retried_broken_call = true;
            const std::string nudge =
                "Answer in plain text only. Do not call a tool for this message and do not "
                "repeat the question.";
            working.emplace_back("user", nudge);
            Json nudge_entry = Json::object();
            nudge_entry.set("role", Json::string("user"));
            nudge_entry.set("content", Json::string(nudge));
            turn.messages.push(std::move(nudge_entry));
            Json payload = Json::object();
            payload.set("step", Json::integer(step_index));
            payload.set("kind", Json::string("retry"));
            payload.set("text", Json::string(head(trim(output), 120)));
            payload.set("note", Json::string("unusable tool-shaped output, asking for plain text"));
            emit("step_end", std::move(payload));
            continue;
        }
        if (!intent.allow_tools && call.valid && !retried_broken_call) {
            // The router says this turn is a plain-text answer (an image
            // description, a memory question, small talk). A tool call here is
            // the model wandering off; ask once for text instead of running it.
            retried_broken_call = true;
            const std::string nudge =
                "Reply with plain text only, in the user's language. Do not call a tool "
                "and do not repeat the question.";
            working.emplace_back("user", nudge);
            Json nudge_entry = Json::object();
            nudge_entry.set("role", Json::string("user"));
            nudge_entry.set("content", Json::string(nudge));
            turn.messages.push(std::move(nudge_entry));
            Json payload = Json::object();
            payload.set("step", Json::integer(step_index));
            payload.set("kind", Json::string("retry"));
            payload.set("text", Json::string(head(trim(output), 120)));
            payload.set("note", Json::string(std::string("tool call not expected for this "
                                                         "intent (") +
                                             to_string(intent.kind) +
                                             "), asking for plain text"));
            emit("step_end", std::move(payload));
            continue;
        }

        std::string resolution;
        std::string canonical =
            call.valid ? tools_.resolve(call.name, &resolution) : std::string();
        // Rescue the turn when the model either wrote a malformed call or named
        // the hinted tool without its arguments; a genuine call to a *different*
        // tool stays the model's decision.
        const bool malformed = !call.valid;
        const bool dropped_arguments =
            hinted.valid && !canonical.empty() && canonical == hinted.name &&
            call.arguments.empty();
        // The runtime spelled the exact call out for arithmetic/clock questions.
        // If the model answers them from thin air instead (it has no clock and no
        // calculator), run the hinted tool: that is what the user asked for, and
        // the model still writes the final sentence afterwards.
        const bool answered_without_tool =
            !call.valid && !trim(output).empty() && hinted.valid;
        if (!hint_used && hinted.valid && tool_calls == 0 &&
            (malformed || dropped_arguments || answered_without_tool)) {
            Json notice = Json::object();
            notice.set("step", Json::integer(step_index));
            notice.set("requested", Json::string(head(trim(output), 120)));
            notice.set("resolved", Json::string(hinted.name));
            notice.set("note", Json::string(
                malformed ? "the model wrote no usable call; the runtime used the call "
                            "from its own hint"
                          : "the model answered directly; the runtime ran the hinted tool"));
            emit("tool_alias", std::move(notice));
            call = hinted;
            canonical = hinted.name;
            resolution.clear();
            hint_used = true;
        }
        if (!call.valid) {
            answer = trim(strip_tool_call(output));
            answer = clean_answer(answer);
            // Punctuation-only debris ("." after a tool error) is not an answer:
            // treat it like an empty generation so the retry/fallback chain runs.
            if (answer.size() <= 8 &&
                answer.find_first_not_of(".,:;!?-_*/\"' \n\t()<>|") == std::string::npos) {
                answer.clear();
            }
            // Set when the answer shown is rewritten from something other than
            // this step's raw output (the runtime's own clock/memory/vision
            // answer); those cannot be re-anchored on the KV.
            bool answer_replaced = false;
            // On-device the Q4 export regularly ignores what it was just given:
            // it answers a clock question with "I have no access to the time"
            // right after the now tool returned it, and a memory question with
            // "I don't have that information" right after recall injected it.
            // When that happens the runtime answers from the authoritative
            // source instead of showing the denial.
            if (turn.has_image == false && !last_tool_summary.empty() &&
                intent.kind == IntentKind::Clock && looks_like_refusal(answer)) {
                answer = "现在是 " + last_tool_summary + "（设备时钟）";
                replace_last_assistant(&turn.messages, answer);
                answer_generated = true;
                answer_verbatim = true;   // the transcript stores this answer below
                answer_replaced = true;
                transcript_matches_kv = true;   // the driver re-anchors the cache
            } else if (intent.kind == IntentKind::MemoryRecall && !recalled.empty() &&
                       (answer.empty() || looks_like_refusal(answer) ||
                        // Only the *top* record counts: an answer that mentions a
                        // superseded fact from an older session is not an answer
                        // to what this conversation stored.
                        !answer_mentions_memory(answer, {recalled.front()}))) {
                answer = "根据我的长期记忆：\n" + memory_answer(recalled);
                replace_last_assistant(&turn.messages, answer);
                answer_generated = true;
                answer_verbatim = true;   // the transcript stores this answer below
                answer_replaced = true;
                transcript_matches_kv = true;   // the driver re-anchors the cache
            }
            // A greeting that answers about an image nobody sent: the earlier
            // image turn is still in the context and the small model fixates on
            // it.  Ask once for a plain greeting instead of showing that.
            // The same export also drifts into another language ("手机上 AI를
            // 개발하는 것은 ..." for a Chinese question, "我 기억할게요" for
            // "请记住：...").  Ask once, explicitly, before showing that.
            auto count_script = [](const std::string& text, uint32_t low, uint32_t high) {
                int count = 0;
                for (size_t index = 0; index < text.size();) {
                    const unsigned char byte = static_cast<unsigned char>(text[index]);
                    uint32_t code_point = byte;
                    size_t length = 1;
                    if (byte >= 0xF0) {
                        code_point = static_cast<uint32_t>(byte & 0x07) << 18;
                        length = 4;
                    } else if (byte >= 0xE0) {
                        code_point = static_cast<uint32_t>(byte & 0x0F) << 12;
                        length = 3;
                    } else if (byte >= 0xC0) {
                        code_point = static_cast<uint32_t>(byte & 0x1F) << 6;
                        length = 2;
                    }
                    for (size_t extra = 1; extra < length && index + extra < text.size();
                         ++extra) {
                        code_point |= static_cast<uint32_t>(
                                          static_cast<unsigned char>(text[index + extra]) & 0x3F)
                                      << (6 * (length - extra - 1));
                    }
                    if (code_point >= low && code_point <= high) {
                        ++count;
                    }
                    index += length;
                }
                return count;
            };
            const int asked_cjk = count_script(turn.user_text, 0x4E00, 0x9FFF);
            const int answered_cjk = count_script(answer, 0x4E00, 0x9FFF);
            const int answered_hangul = count_script(answer, 0xAC00, 0xD7AF);
            const bool wrong_language =
                asked_cjk > 0 && (answered_cjk == 0 || answered_hangul > answered_cjk);
            // A pure formula ("17*23 = 391") carries no language at all: there is
            // nothing to re-ask about.
            const bool has_letters =
                answered_cjk > 0 || answered_hangul > 0 ||
                std::any_of(answer.begin(), answer.end(), [](unsigned char byte) {
                    return std::isalpha(byte) != 0;
                });
            if (language_retries < 2 && wrong_language && has_letters && !answer.empty() &&
                step_index < options_.config.max_steps) {
                ++language_retries;
                // Repeat the question in Chinese: an English "reply in Chinese"
                // instruction was ignored, while the user's own words anchor both
                // the topic and the language.
                const std::string nudge =
                    "请用简体中文重新回答这个问题：" + turn.user_text;
                working.emplace_back("user", nudge);
                Json nudge_entry = Json::object();
                nudge_entry.set("role", Json::string("user"));
                nudge_entry.set("content", Json::string(nudge));
                turn.messages.push(std::move(nudge_entry));
                Json payload = Json::object();
                payload.set("step", Json::integer(step_index));
                payload.set("kind", Json::string("retry"));
                payload.set("text", Json::string(head(answer, 120)));
                payload.set("note", Json::string("answered a Chinese message in another "
                                                 "language, asking once more"));
                emit("step_end", std::move(payload));
                continue;
            }
            if (!retried_greeting && intent.kind == IntentKind::Greeting &&
                !turn.has_image && mentions_image(answer)) {
                retried_greeting = true;
                const std::string nudge =
                    "This message is only a greeting and no image was sent. Greet the user "
                    "back briefly in their language.";
                working.emplace_back("user", nudge);
                Json nudge_entry = Json::object();
                nudge_entry.set("role", Json::string("user"));
                nudge_entry.set("content", Json::string(nudge));
                turn.messages.push(std::move(nudge_entry));
                Json payload = Json::object();
                payload.set("step", Json::integer(step_index));
                payload.set("kind", Json::string("retry"));
                payload.set("text", Json::string(head(answer, 120)));
                payload.set("note", Json::string("greeting answered about an image, asking once "
                                                 "more"));
                emit("step_end", std::move(payload));
                continue;
            }
            // The deployed Q4 export sometimes answers an image question with a
            // flat refusal even though the image is in its input ("I am unable
            // to process that request.").  Give the image one more chance before
            // showing that to the user.
            if (image.valid && !answer.empty() &&
                looks_like_unusable_image_answer(answer, intent)) {
                if (!retried_refusal && step_index < options_.config.max_steps) {
                    // First: ask once more, with the image explicitly restated.
                    retried_refusal = true;
                    const std::string nudge =
                        "The image is already part of your input and you can see it. "
                        "Describe what is in it in plain text, in two or three sentences, now.";
                    working.emplace_back("user", nudge);
                    Json nudge_entry = Json::object();
                    nudge_entry.set("role", Json::string("user"));
                    nudge_entry.set("content", Json::string(nudge));
                    turn.messages.push(std::move(nudge_entry));
                    Json payload = Json::object();
                    payload.set("step", Json::integer(step_index));
                    payload.set("kind", Json::string("retry"));
                    payload.set("text", Json::string(head(answer, 120)));
                    payload.set("note", Json::string("refusal on an image turn, asking once more"));
                    emit("step_end", std::move(payload));
                    continue;
                }
                // Still refusing: the agent prompt (tools + instructions) is what
                // pushes this export into a denial, so fall back to the plain
                // vision prompt the shipped single-shot demo was validated with.
                const std::string described = describe_image_directly(image_path, image);
                if (!described.empty()) {
                    answer = described;
                    replace_last_assistant(&turn.messages, answer);
                    answer_generated = true;
                    answer_verbatim = true;
                    answer_replaced = true;
                    transcript_matches_kv = true;   // the driver re-anchors the cache
                    Json payload = Json::object();
                    payload.set("step", Json::integer(step_index));
                    payload.set("kind", Json::string("vision_fallback"));
                    payload.set("text", Json::string(answer));
                    payload.set("note", Json::string("model refused the image; answered with the "
                                                     "shipped vision prompt"));
                    emit("step_end", std::move(payload));
                    stop_reason = "answer";
                    break;
                }
            }
            answer_generated = !answer.empty();
            answer_verbatim = trim(output) == answer;
            transcript_matches_kv = answer_generated && !answer_replaced && !trim(output).empty();
            Json assistant_entry = Json::object();
            assistant_entry.set("role", Json::string("assistant"));
            // Store what the user is shown (a native fallback or a cleaned answer
            // replaces the raw generation); the KV then still matches the
            // transcript closely enough for the next turn to stay incremental.
            assistant_entry.set("content", Json::string(answer.empty() ? output : answer));
            turn.messages.push(std::move(assistant_entry));
            Json payload = Json::object();
            payload.set("step", Json::integer(step_index));
            payload.set("kind", Json::string("answer"));
            payload.set("text", Json::string(answer));
            payload.set("generated_tokens", Json::integer(metrics.generated_tokens));
            payload.set("prompt_tokens", Json::integer(metrics.prompt_tokens));
            payload.set("context_tokens", Json::integer(metrics.context_tokens));
            payload.set("engine_prefill_tokens",
                        Json::integer(metrics.engine_prefill_tokens));
            payload.set("mode", Json::string(metrics.full_prefill ? "full" : "delta"));
            payload.set("duration_ms", Json::integer(step_ms));
            payload.set("stop_reason", Json::string(metrics.stop_reason));
            emit("step_end", std::move(payload));
            stop_reason = "answer";
            break;
        }

        Json step_end = Json::object();
        step_end.set("step", Json::integer(step_index));
        step_end.set("kind", Json::string("tool"));
        step_end.set("text", Json::string(trim(output)));
        step_end.set("call_valid", Json::boolean(call.valid));
        step_end.set("call_error", Json::string(call.error));
        step_end.set("generated_tokens", Json::integer(metrics.generated_tokens));
        step_end.set("prompt_tokens", Json::integer(metrics.prompt_tokens));
        step_end.set("context_tokens", Json::integer(metrics.context_tokens));
        step_end.set("engine_prefill_tokens", Json::integer(metrics.engine_prefill_tokens));
        step_end.set("mode", Json::string(metrics.full_prefill ? "full" : "delta"));
        step_end.set("duration_ms", Json::integer(step_ms));
        emit("step_end", std::move(step_end));

        // Resolve aliases and near-miss names before declaring the tool unknown:
        // a 2B model writes `get_time`, `calc` or a one-letter typo regularly.
        if (!call.valid || canonical.empty()) {
            // Hand the failure back as an observation so the model can retry.
            ToolResult result;
            result.name = call.name.empty() ? "unknown" : call.name;
            result.ok = false;
            result.error = call.valid ? "unknown tool '" + call.name + "'; available: " +
                                            join(tools_.names(), ", ")
                                      : "the tool call could not be parsed: " + call.error;
            result.summary = "error: " + result.error;
            Json call_event = Json::object();
            call_event.set("step", Json::integer(step_index));
            call_event.set("name", Json::string(result.name));
            call_event.set("arguments", call.arguments);
            call_event.set("raw", Json::string(head(call.raw, 400)));
            call_event.set("ok", Json::boolean(false));
            emit("tool_call", std::move(call_event));
            Json result_event = Json::object();
            result_event.set("step", Json::integer(step_index));
            result_event.set("name", Json::string(result.name));
            result_event.set("ok", Json::boolean(false));
            result_event.set("error", Json::string(result.error));
            result_event.set("summary", Json::string(result.summary));
            emit("tool_result", std::move(result_event));
            const std::string observation =
                format_observation(result, planner_.render(),
                                   options_.config.max_tool_calls - tool_calls);
            Json assistant_entry = Json::object();
            assistant_entry.set("role", Json::string("assistant"));
            assistant_entry.set("content", Json::string(output));
            Json observation_entry = Json::object();
            observation_entry.set("role", Json::string("user"));
            observation_entry.set("content", Json::string(observation));
            turn.messages.push(std::move(assistant_entry));
            turn.messages.push(std::move(observation_entry));
            working = rebuild_working();
            continue;
        }
        if (!resolution.empty()) {
            Json notice = Json::object();
            notice.set("step", Json::integer(step_index));
            notice.set("requested", Json::string(call.name));
            notice.set("resolved", Json::string(canonical));
            notice.set("note", Json::string(resolution));
            emit("tool_alias", std::move(notice));
        }
        call.name = canonical;
        // The router derived the exact call from the user's own sentence for the
        // cases it can ("算一下 17*23 等于多少？" -> calculator{17*23}).  The Q4
        // export sometimes invents its own operands instead (the phone answered
        // 1*3); for those hints the derived arguments win.
        // Any hinted call the router could derive itself (calculator expression,
        // unit conversion value/from/to, the pasted text to count) is
        // authoritative: the Q4 export regularly invents its own operands
        // ("1*3" for 17*23, a made-up conversion) when it should just repeat the
        // call it was given.
        const bool hinted_has_arguments =
            hinted.valid && hinted.name == canonical && hinted.arguments.dump() != "{}";
        // A value turn the router could spell out is answered by *its* call: the
        // phone answered "把 100华氏度换算成摄氏度" with the calculator ("32 = 32")
        // because the model picked the wrong tool, and a wrong tool with the right
        // arguments is just as wrong as a right tool with invented arguments.
        const bool hinted_wins = hinted.valid && tool_calls == 0 &&
                                 (!canonical.empty() ? canonical != hinted.name : true);
        if (!hint_used && hinted.valid && (hinted_wins || (hinted_has_arguments &&
            call.arguments.dump() != hinted.arguments.dump()))) {
            Json notice = Json::object();
            notice.set("step", Json::integer(step_index));
            notice.set("requested", Json::string(call.name + call.arguments.dump()));
            notice.set("resolved", Json::string(hinted.arguments.dump()));
            notice.set("note", Json::string("the model invented the arguments; the runtime used "
                                            "the call it derived from the user's message"));
            emit("tool_alias", std::move(notice));
            call = hinted;          // name *and* arguments: the model may have picked
            canonical = hinted.name;  // a different tool for the same value question
            hint_used = true;
        }

        ToolContext context;
        context.config = &options_.config;
        context.memory = &memory_;
        context.sessions = &sessions_;
        context.planner = &planner_;
        context.session_id = sessions_.meta().id;
        context.turn_index = turn.index;
        context.has_image = turn.has_image;
        context.image_tokens = turn.image_tokens;
        context.image_note = turn.image_note;
        context.image_sha256 = image.sha256;
        context.image_file = turn.image_file;
        context.device = options_.device;
        context.language = options_.config.language;
        std::vector<std::string> written;
        context.written_memories = &written;
        ToolResult result = tools_.call(call.name, call.arguments, context);
        if (result.ok) {
            // Remember the last successful result so the runtime can answer from
            // it when the model ignores what it was just handed.
            last_tool_summary = result.summary;
        }
        // Failure escalation (Mobile-Agent-E's err_to_manager_thresh): the same
        // tool failing the same way twice tells the model to change strategy
        // instead of retrying the identical call.
        if (!result.ok) {
            const std::string signature = call.name + "|" + result.error;
            if (signature == last_error_signature) {
                result.summary = result.summary + "\n[NOTE] This tool already failed the same "
                                                "way once. Do not repeat the same call: change "
                                                "the arguments, use another tool, or answer "
                                                "without a tool.";
            }
            last_error_signature = signature;
        } else {
            last_error_signature.clear();
        }

        Json call_event = Json::object();
        call_event.set("step", Json::integer(step_index));
        call_event.set("name", Json::string(call.name));
        call_event.set("arguments", call.arguments);
        call_event.set("raw", Json::string(head(call.raw, 400)));
        call_event.set("ok", Json::boolean(result.ok));
        emit("tool_call", std::move(call_event));
        Json result_event = Json::object();
        result_event.set("step", Json::integer(step_index));
        result_event.set("name", Json::string(result.name));
        result_event.set("ok", Json::boolean(result.ok));
        result_event.set("error", Json::string(result.error));
        result_event.set("summary", Json::string(result.summary));
        result_event.set("data", result.value);
        result_event.set("duration_ms", Json::number(result.duration_ms));
        emit("tool_result", std::move(result_event));
        for (const auto& id : written) {
            turn.memories_written.push_back(id);
        }
        if (!planner_.has_plan()) {
            // Every tool call becomes a plan step, so the UI always shows what
            // the agent did even when the model did not call make_plan.
            std::vector<std::string> implicit;
            implicit.push_back(call.name + ": " + head(result.summary, 120));
            planner_.set_plan(implicit);
        }
        planner_.note_tool(call.name, result.ok);
        Json plan_payload = Json::object();
        plan_payload.set("steps", planner_.to_json());
        plan_payload.set("goal", Json::string(planner_.goal()));
        plan_payload.set("source", Json::string("update"));
        emit("plan", std::move(plan_payload));

        turn.tool_calls.push(tool_call_entry(result, call));
        const std::string observation =
            format_observation(result, planner_.render(),
                               options_.config.max_tool_calls - tool_calls - 1);
        Json assistant_entry = Json::object();
        assistant_entry.set("role", Json::string("assistant"));
        assistant_entry.set("content", Json::string(output));
        Json observation_entry = Json::object();
        observation_entry.set("role", Json::string("user"));
        observation_entry.set("content", Json::string(observation));
        turn.messages.push(std::move(assistant_entry));
        turn.messages.push(std::move(observation_entry));
        working = rebuild_working();
        ++tool_calls;

        context_tokens = metrics.context_tokens;
        const bool budget_left = tool_calls < options_.config.max_tool_calls &&
                                 step_index < options_.config.max_steps;
        if (!budget_left) {
            truncated = tool_calls >= options_.config.max_tool_calls;
            stop_reason = truncated ? "tool_budget" : "step_budget";
            break;
        }
    }

    // --- forced wrap-up -----------------------------------------------------
    if (answer.empty() && !cancelled_turn) {
        const std::string reason = truncated ? "the tool call budget is exhausted"
                                             : "the step budget is exhausted";
        ChatMessages wrap = working;
        wrap.emplace_back("user", build_forced_answer_prompt(reason));
        GenerationMetrics metrics;
        std::string error;
        auto on_token = [&](int index, const std::string& piece) -> bool {
            if (first_token_ms == 0) {
                first_token_ms = now_ms();
            }
            if (!options_.config.stream_steps) {
                return !cancelled();
            }
            Json payload = Json::object();
            payload.set("step", Json::integer(step_index + 1));
            payload.set("index", Json::integer(index));
            payload.set("text_b64", Json::string(base64_encode(piece)));
            emit("step_delta", std::move(payload));
            return !cancelled();
        };
        std::string output = llm_.generate_delta(wrap, turn_splices,
                                                options_.config.max_answer_tokens,
                                                on_token, &metrics, &error);
        // The wrap-up prompt carries an extra instruction that the transcript
        // does not keep, so the next turn rebuilds from a pristine engine
        // instead of reusing this KV state.
        needs_full_prefill_ = true;
        if (!output.empty()) {
            Json assistant_entry = Json::object();
            assistant_entry.set("role", Json::string("assistant"));
            assistant_entry.set("content", Json::string(output));
            turn.messages.push(std::move(assistant_entry));
        }
        usage.prompt_tokens += metrics.prompt_tokens;
        usage.generated_tokens += metrics.generated_tokens;
        last_metrics = metrics;
        if (!trim(output).empty()) {
            answer = trim(strip_tool_call(output));
            answer_generated = !answer.empty();
            answer_verbatim = trim(output) == answer;
        }
        Json payload = Json::object();
        payload.set("step", Json::integer(step_index + 1));
        payload.set("kind", Json::string("answer"));
        payload.set("text", Json::string(answer));
        payload.set("forced", Json::boolean(true));
        payload.set("generated_tokens", Json::integer(metrics.generated_tokens));
        payload.set("prompt_tokens", Json::integer(metrics.prompt_tokens));
        payload.set("mode", Json::string(metrics.full_prefill ? "full" : "delta"));
        payload.set("stop_reason", Json::string(metrics.stop_reason));
        emit("step_end", std::move(payload));
    }
    // A calculated answer has to carry the number.  The Q4 export likes to
    // answer a solved arithmetic turn with a pleasant sentence that drops the
    // result entirely ("I can help you with that."), which reads as "it did not
    // answer" - the calculator already phrased it, so use that sentence.
    const bool value_must_show =
        intent.kind == IntentKind::Arithmetic || intent.kind == IntentKind::Convert ||
        intent.kind == IntentKind::Clock;
    // The tool is authoritative for a value question: the phone produced
    // "The current time is 1:300AM" one second after the now tool returned
    // 14:53, so an answer that does not carry the tool's own value is replaced
    // by it.  The turn's own stop reason is left alone (the UI still shows a
    // tool-budget wrap-up as such).
    if (!answer.empty() && !cancelled_turn && value_must_show && !last_tool_summary.empty()) {
        const size_t equals = last_tool_summary.rfind("= ");
        const std::string full_value =
            equals == std::string::npos ? std::string() : trim(last_tool_summary.substr(equals + 2));
        std::string value = full_value;
        const size_t space = value.find(' ');
        if (space != std::string::npos) {
            value = value.substr(0, space);   // the probe: does the answer carry the value?
        }
        if (value.size() >= 1 && answer.find(value) == std::string::npos) {
            const std::string replacement =
                intent.kind == IntentKind::Clock ? "现在是 " + full_value + "（设备时钟）"
                                                 : last_tool_summary;
            Json payload = Json::object();
            payload.set("step", Json::integer(step_index));
            payload.set("kind", Json::string("tool_answer"));
            payload.set("text", Json::string(replacement));
            payload.set("note", Json::string("the model's sentence dropped the result; the "
                                             "runtime answered with the tool's own line"));
            emit("step_end", std::move(payload));
            answer = replacement;
            replace_last_assistant(&turn.messages, answer);
            answer_generated = true;
            answer_verbatim = false;
            transcript_matches_kv = true;   // the driver re-anchors the cache
        }
    }
    if (answer.empty()) {
        if (cancelled_turn) {
            answer = "(stopped)";
        } else if (!turn.tool_calls.items().empty()) {
            const Json* last = &turn.tool_calls.items().back();
            const std::string summary = last->string_or("summary", "no result");
            const bool tool_answered =
                last->bool_or("ok", false) && !summary.empty() &&
                (intent.kind == IntentKind::Arithmetic || intent.kind == IntentKind::Convert ||
                 intent.kind == IntentKind::Clock || intent.kind == IntentKind::TextStats);
            if (tool_answered) {
                // The tool already phrased the answer ("17*23 = 391") and the
                // model went silent after reading it; show the tool's sentence
                // instead of a failure line.
                answer = summary;
                if (intent.kind == IntentKind::TextStats) {
                    const std::string counted = format_count_line(summary, turn.user_text);
                    if (!counted.empty()) {
                        answer = counted;
                    }
                }
                replace_last_assistant(&turn.messages, answer);
                transcript_matches_kv = true;   // the driver re-anchors the cache
                if (intent.kind == IntentKind::Clock) {
                    const size_t equals = summary.rfind("= ");
                    if (equals != std::string::npos) {
                        answer = "现在是 " + trim(summary.substr(equals + 2)) + "（设备时钟）";
                    }
                }
                answer_generated = true;
                transcript_matches_kv = true;   // the driver re-anchors the cache
                Json payload = Json::object();
                payload.set("step", Json::integer(step_index));
                payload.set("kind", Json::string("tool_answer"));
                payload.set("text", Json::string(answer));
                payload.set("note", Json::string("the model stayed silent after the tool; the "
                                                 "runtime answered from its result"));
                emit("step_end", std::move(payload));
            } else {
                answer = "I could not finish the answer. Last tool result: " +
                         last->string_or("summary", "no result");
            }
        } else {
            answer = "I could not produce an answer for that request.";
        }
        // The transcript keeps what the user actually sees, even when the model
        // produced nothing usable.
        replace_last_assistant(&turn.messages, answer);
        stop_reason = stop_reason.empty() ? "fallback" : stop_reason;
    }

    // Same for a counting request: the tool counted, so the answer has to carry
    // the number instead of a sentence that forgets it.
    if (!answer.empty() && !cancelled_turn && intent.kind == IntentKind::TextStats &&
        !last_tool_summary.empty()) {
        std::string characters;
        std::string words;
        for (const std::string& part : split(last_tool_summary, ' ')) {
            if (starts_with(part, "characters=")) {
                characters = part.substr(std::string("characters=").size());
            } else if (starts_with(part, "words=")) {
                words = part.substr(std::string("words=").size());
            }
        }
        // The phone answered this with "444" while the tool had counted 15
        // characters, so requiring "no digits at all" was not enough: the answer
        // has to carry the count the tool produced.
        if (!characters.empty() && answer.find(characters) == std::string::npos) {
            const bool cjk_turn =
                turn.user_text.end() != std::find_if(turn.user_text.begin(),
                                                     turn.user_text.end(), [](unsigned char byte) {
                                                         return byte >= 0x80;
                                                     });
            const std::string counted = cjk_turn
                         ? "这段文字有 " + characters + " 个字符" +
                               (words.empty() ? "。" : "（" + words + " 个词）。")
                         : "That text has " + characters + " characters" +
                               (words.empty() ? "." : " (" + words + " words).");
            answer = counted;
            replace_last_assistant(&turn.messages, answer);
            Json payload = Json::object();
            payload.set("step", Json::integer(step_index));
            payload.set("kind", Json::string("tool_answer"));
            payload.set("text", Json::string(answer));
            payload.set("note", Json::string("the model's sentence dropped the count; the runtime "
                                             "answered with the tool's own numbers"));
            emit("step_end", std::move(payload));
            answer_generated = true;
            answer_verbatim = false;
            transcript_matches_kv = true;   // the driver re-anchors the cache
        }
    }
    // "请记住：我叫杨雷" is answered with "我没听见你" (and once with Korean) on
    // the phone even though the fact is stored.  The store is authoritative for
    // this intent, so acknowledge with it when the model's reply does not
    // mention the fact at all.
    // Never *claim* to have stored something when the memory switch is off: the
    // phone answered "好的，我记住了：…" with `records: 0` in the store and the
    // switch set to false, which is simply a lie to the user.
    if (!cancelled_turn && intent.kind == IntentKind::MemoryWrite &&
        options_.config.memory_enabled) {
        // Use the same extraction the store uses ("改成记住我叫杨雷" stores
        // "我叫杨雷"), so the acknowledgement can never confirm the *old* fact -
        // the phone answered a correction with the superseded name once.
        std::string fact;
        const auto extracted = extract_memories(turn.user_text, answer, options_.config.language);
        if (!extracted.empty()) {
            fact = trim(extracted.front().text);
        }
        if (fact.empty()) {
            fact = trim(turn.user_text);
        }
        std::vector<ScoredMemory> stored;
        ScoredMemory record;
        record.record.text = fact;
        record.record.kind = "fact";
        stored.push_back(std::move(record));
        if (!fact.empty() && (answer.empty() || !answer_mentions_memory(answer, stored))) {
            const bool cjk_fact = fact.find_first_not_of(
                                      "abcdefghijklmnopqrstuvwxyz"
                                      "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?:;'\"()[]{}"
                                      "<>+-*/=_@#$%^&|\\~`\t\r\n") != std::string::npos;
            answer = cjk_fact ? "好的，我记住了：" + fact : "Got it, I stored: " + fact;
            replace_last_assistant(&turn.messages, answer);
            Json payload = Json::object();
            payload.set("step", Json::integer(step_index));
            payload.set("kind", Json::string("memory_ack"));
            payload.set("text", Json::string(answer));
            payload.set("note", Json::string("the model's reply did not acknowledge the fact; "
                                             "the runtime confirmed it from the memory store"));
            emit("step_end", std::move(payload));
            answer_generated = true;
            answer_verbatim = false;
            transcript_matches_kv = true;
        }
    }
    // Memory is switched off: say so instead of pretending, and keep the answer
    // useful if the model produced nothing.
    if (!cancelled_turn && intent.kind == IntentKind::MemoryWrite &&
        !options_.config.memory_enabled) {
        const bool quiet = answer.empty() || looks_like_refusal(answer);
        if (quiet) {
            answer = "\u8bb0\u5fc6\u529f\u80fd\u5f53\u524d\u662f\u5173\u95ed\u7684\uff0c"
                     "\u8fd9\u6761\u5185\u5bb9\u4e0d\u4f1a\u88ab\u4fdd\u5b58\u3002"
                     "\u6253\u5f00\u300c\u8bb0\u5fc6\u300d\u5f00\u5173\u540e\u6211\u5c31"
                     "\u80fd\u8bb0\u4f4f\u4e86\u3002";
        } else {
            answer += " \uff08\u8bb0\u5fc6\u529f\u80fd\u5f53\u524d\u5173\u95ed\uff0c"
                      "\u8fd9\u6761\u5185\u5bb9\u672a\u4fdd\u5b58\uff09";
        }
        replace_last_assistant(&turn.messages, answer);
        Json payload = Json::object();
        payload.set("step", Json::integer(step_index));
        payload.set("kind", Json::string("memory_disabled_note"));
        payload.set("text", Json::string(answer));
        payload.set("note", Json::string("the memory switch is off; the runtime tells the user "
                                         "instead of acknowledging a write"));
        emit("step_end", std::move(payload));
        answer_generated = true;
        answer_verbatim = false;
        transcript_matches_kv = true;
    }

    // --- memory write-back --------------------------------------------------
    if (!answer_generated || (!answer_verbatim && !transcript_matches_kv) || cancelled_turn ||
        stop_reason == "error") {
        // The KV state does not match the transcript (cancelled, failed or a
        // rewritten answer): the next turn starts from a pristine full prefill.
        needs_full_prefill_ = true;
    } else if (transcript_matches_kv) {
        // Tell the driver what its KV state holds *as the transcript renders it*.
        // The engine's own bookkeeping re-decodes the generated tokens, which
        // picks up template debris (`<turn|>`, a leading space) that the runtime
        // strips before storing; without this anchor the next turn would see a
        // mismatch and re-prefill the whole conversation.
        ChatMessages anchor = working;
        anchor.emplace_back("assistant", answer);
        llm_.sync_transcript(anchor);
    }
    write_memories(turn.user_text, answer, &turn);

    // --- finalise -----------------------------------------------------------
    planner_.finish_all();
    turn.answer = answer;
    turn.plan = planner_.steps();
    turn.steps = step_index;
    turn.usage = usage;
    turn.duration_ms = now_ms() - turn_begin;
    turn.ttft_ms = first_token_ms > 0
        ? static_cast<double>(first_token_ms - turn_begin)
        : 0.0;
    turn.decode_tokens_per_second = last_metrics.decode_tokens_per_second;
    turn.prefill_ms = last_metrics.prefill_ms;
    turn.npu_ms = last_metrics.npu_ms;
    turn.cancelled = cancelled_turn;
    turn.truncated = truncated;
    turn.stop_reason = stop_reason.empty() ? "answer" : stop_reason;
    update_session_notes(turn);
    sessions_.append_turn(turn);

    Json plan_payload = Json::object();
    plan_payload.set("steps", planner_.to_json());
    plan_payload.set("goal", Json::string(planner_.goal()));
    plan_payload.set("source", Json::string("final"));
    emit("plan", std::move(plan_payload));

    Json answer_payload = Json::object();
    answer_payload.set("turn", Json::integer(turn.index));
    answer_payload.set("text", Json::string(answer));
    answer_payload.set("steps", Json::integer(step_index));
    answer_payload.set("tool_calls", Json::integer(tool_calls));
    answer_payload.set("cancelled", Json::boolean(cancelled_turn));
    answer_payload.set("truncated", Json::boolean(truncated));
    answer_payload.set("stop_reason", Json::string(turn.stop_reason));
    emit("answer", std::move(answer_payload));

    Json metrics_payload = Json::object();
    metrics_payload.set("turn", Json::integer(turn.index));
    metrics_payload.set("ttft_ms", Json::number(turn.ttft_ms));
    metrics_payload.set("duration_ms", Json::integer(turn.duration_ms));
    metrics_payload.set("prompt_tokens", Json::integer(usage.prompt_tokens));
    metrics_payload.set("generated_tokens", Json::integer(usage.generated_tokens));
    metrics_payload.set("decode_tokens_per_second",
                        Json::number(turn.decode_tokens_per_second));
    metrics_payload.set("prefill_ms", Json::number(turn.prefill_ms));
    metrics_payload.set("npu_ms", Json::number(turn.npu_ms));
    metrics_payload.set("context_tokens", Json::integer(last_metrics.context_tokens));
    metrics_payload.set("context_budget", Json::integer(options_.config.context_budget_tokens));
    metrics_payload.set("cache_hit", Json::boolean(last_metrics.cache_hit));
    metrics_payload.set("cache_delta_tokens", Json::integer(last_metrics.cache_delta_tokens));
    metrics_payload.set("engine_prefill_tokens",
                        Json::integer(last_metrics.engine_prefill_tokens));
    metrics_payload.set("steps", Json::integer(step_index));
    metrics_payload.set("tool_calls", Json::integer(tool_calls));
    metrics_payload.set("memory_records", Json::integer(static_cast<long long>(memory_.size())));
    metrics_payload.set("memories_written",
                        Json::integer(static_cast<long long>(turn.memories_written.size())));
    emit("metrics", std::move(metrics_payload));

    Json end_payload = Json::object();
    end_payload.set("turn", Json::integer(turn.index));
    end_payload.set("ok", Json::boolean(!cancelled_turn && stop_reason != "error"));
    end_payload.set("stop_reason", Json::string(turn.stop_reason));
    end_payload.set("session_turns", Json::integer(sessions_.meta().turns));
    end_payload.set("context_tokens", Json::integer(last_metrics.context_tokens));
    end_payload.set("compactions", Json::integer(sessions_.meta().compactions));
    emit("turn_end", std::move(end_payload));

    if (sessions_.needs_compaction(history_budget_tokens(), 0)) {
        compact_history(true);
    }

    Json out = Json::object();
    out.set("ok", Json::boolean(!cancelled_turn && stop_reason != "error"));
    out.set("answer", Json::string(answer));
    out.set("turn", Json::integer(turn.index));
    out.set("session", Json::string(sessions_.meta().id));
    out.set("ttft_ms", Json::number(turn.ttft_ms));
    out.set("duration_ms", Json::integer(turn.duration_ms));
    out.set("tool_calls", Json::integer(tool_calls));
    out.set("steps", Json::integer(step_index));
    out.set("memories_written", Json::integer(static_cast<long long>(
                                    turn.memories_written.size())));
    out.set("stop_reason", Json::string(turn.stop_reason));
    return out;
}

}  // namespace agent
