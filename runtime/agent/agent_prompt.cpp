#include "agent_prompt.h"

#include "agent_intent.h"
#include "agent_util.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cstring>
#include <sstream>

namespace agent {
namespace {

// True when the text is mostly CJK (the model tends to answer those in English
// unless it is told again in the turn itself).
bool is_cjk(const std::string& text) {
    size_t cjk = 0;
    size_t other = 0;
    for (size_t index = 0; index < text.size();) {
        const unsigned char byte = static_cast<unsigned char>(text[index]);
        size_t length = 1;
        if ((byte & 0x80) == 0) {
            if (std::isalnum(byte) != 0) {
                ++other;
            }
        } else if ((byte & 0xe0) == 0xc0) {
            length = 2;
        } else if ((byte & 0xf0) == 0xe0) {
            length = 3;
        } else {
            length = 4;
        }
        if (length >= 3) {
            ++cjk;
        }
        index += length;
    }
    return cjk >= 2 && cjk * 2 >= other;
}

const char* kImageStart = "<|image>";
const char* kImageEnd = "<image|>";
const char* kImageToken = "<|image|>";
const char* kQuote = "<|\"|>";
const size_t kQuoteLength = 5;   // <|"|>

std::string quoted(const std::string& value) {
    return std::string(kQuote) + value + kQuote;
}

std::string schema_type(const Json& schema) {
    const std::string type = to_lower(schema.string_or("type", "string"));
    if (type == "number" || type == "integer") {
        return "NUMBER";
    }
    if (type == "boolean") {
        return "BOOLEAN";
    }
    if (type == "array") {
        return "ARRAY";
    }
    if (type == "object") {
        return "OBJECT";
    }
    return "STRING";
}

std::vector<std::string> required_list(const Json& schema) {
    std::vector<std::string> required;
    const Json* list = schema.find("required");
    if (list == nullptr || !list->is_array()) {
        return required;
    }
    for (const auto& entry : list->items()) {
        if (entry.is_string()) {
            required.push_back(entry.as_string());
        }
    }
    return required;
}

// Mirrors the checkpoint's format_parameters() macro, minus the optional
// `required`/`type` trailer: the model only needs the argument names, and every
// token saved here is paid again on each full prefill.
std::string format_properties(const Json& properties,
                              const std::vector<std::string>& required) {
    (void)required;
    std::ostringstream out;
    out << "{";
    bool first = true;
    if (properties.is_object()) {
        for (const auto& member : properties.members()) {
            const Json& schema = member.second;
            if (!first) {
                out << ",";
            }
            first = false;
            out << member.first << ":{";
            const std::string description = schema.string_or("description", "");
            if (!description.empty()) {
                out << "description:" << quoted(description) << ",";
            }
            out << "type:" << quoted(schema_type(schema)) << "}";
        }
    }
    out << "}";
    return out.str();
}

// --- native argument reading ------------------------------------------------

class ArgumentParser {
public:
    ArgumentParser(const std::string& text, std::string* error)
        : text_(text), error_(error) {}

    bool parse(Json* out) {
        skip_space();
        if (position_ >= text_.size() || text_[position_] != '{') {
            return fail("expected '{'");
        }
        return parse_object(out);
    }

private:
    bool fail(const std::string& message) {
        if (error_ != nullptr && error_->empty()) {
            *error_ = message;
        }
        return false;
    }

    void skip_space() {
        while (position_ < text_.size() &&
               (text_[position_] == ' ' || text_[position_] == '\t' ||
                text_[position_] == '\n' || text_[position_] == '\r')) {
            ++position_;
        }
    }

    bool consume(char character) {
        skip_space();
        if (position_ < text_.size() && text_[position_] == character) {
            ++position_;
            return true;
        }
        return false;
    }

    bool consume_token(const char* token) {
        skip_space();
        const size_t length = std::strlen(token);
        if (text_.compare(position_, length, token) == 0) {
            position_ += length;
            return true;
        }
        return false;
    }

    bool parse_object(Json* out) {
        if (!consume('{')) {
            return fail("expected '{'");
        }
        Json object = Json::object();
        skip_space();
        if (consume('}')) {
            *out = std::move(object);
            return true;
        }
        while (true) {
            std::string key;
            if (!parse_key(&key)) {
                return false;
            }
            if (!consume(':')) {
                return fail("expected ':' after '" + key + "'");
            }
            Json value;
            if (!parse_value(&value)) {
                return false;
            }
            object.set(key, std::move(value));
            skip_space();
            if (consume(',')) {
                continue;
            }
            if (consume('}')) {
                break;
            }
            return fail("expected ',' or '}'");
        }
        *out = std::move(object);
        return true;
    }

    bool consume_string(std::string* value) {
        if (!consume_token(kQuote)) {
            return false;
        }
        const size_t begin = position_;
        while (position_ < text_.size() &&
               text_.compare(position_, kQuoteLength, kQuote) != 0) {
            ++position_;
        }
        if (position_ >= text_.size()) {
            fail("unterminated <|\"|> string");
            return false;
        }
        *value = text_.substr(begin, position_ - begin);
        position_ += kQuoteLength;
        return true;
    }

    bool parse_key(std::string* key) {
        skip_space();
        if (consume_string(key)) {
            return true;
        }
        const size_t begin = position_;
        while (position_ < text_.size() && text_[position_] != ':' &&
               text_[position_] != ',' && text_[position_] != '}') {
            ++position_;
        }
        std::string raw = trim(text_.substr(begin, position_ - begin));
        if (raw.size() >= 2 && raw.front() == '"' && raw.back() == '"') {
            raw = raw.substr(1, raw.size() - 2);
        }
        if (raw.empty()) {
            return fail("empty argument name");
        }
        *key = raw;
        return true;
    }

    bool parse_value(Json* out) {
        skip_space();
        if (position_ >= text_.size()) {
            return fail("unexpected end of arguments");
        }
        std::string quoted_value;
        if (consume_string(&quoted_value)) {
            *out = Json::string(quoted_value);
            return true;
        }
        if (text_[position_] == '{') {
            return parse_object(out);
        }
        if (text_[position_] == '[') {
            Json array = Json::array();
            consume('[');
            skip_space();
            if (consume(']')) {
                *out = std::move(array);
                return true;
            }
            while (true) {
                Json entry;
                if (!parse_value(&entry)) {
                    return false;
                }
                array.push(std::move(entry));
                skip_space();
                if (consume(',')) {
                    continue;
                }
                if (consume(']')) {
                    break;
                }
                return fail("expected ',' or ']'");
            }
            *out = std::move(array);
            return true;
        }
        // Bare value: read up to the next top level ',' or closing bracket.
        const size_t begin = position_;
        int depth = 0;
        while (position_ < text_.size()) {
            const char character = text_[position_];
            if (character == '{' || character == '[') {
                ++depth;
            } else if (character == '}' || character == ']') {
                if (depth == 0) {
                    break;
                }
                --depth;
            } else if (character == ',' && depth == 0) {
                break;
            }
            ++position_;
        }
        const std::string raw = trim(text_.substr(begin, position_ - begin));
        if (raw == "true" || raw == "True") {
            *out = Json::boolean(true);
            return true;
        }
        if (raw == "false" || raw == "False") {
            *out = Json::boolean(false);
            return true;
        }
        if (!raw.empty()) {
            char* end = nullptr;
            const double number = std::strtod(raw.c_str(), &end);
            if (end != nullptr && *end == '\0' && end != raw.c_str()) {
                *out = Json::number(number);
                return true;
            }
        }
        std::string text_value = raw;
        if (text_value.size() >= 2 && text_value.front() == '"' && text_value.back() == '"') {
            text_value = text_value.substr(1, text_value.size() - 2);
        }
        *out = Json::string(text_value);
        return true;
    }

    const std::string& text_;
    std::string* error_;
    size_t position_ = 0;
};

// Extracts the balanced `{...}` block that starts at `begin`, ignoring braces
// inside <|"|> strings.
bool extract_braces(const std::string& text, size_t begin, std::string* out) {
    int depth = 0;
    bool in_json_string = false;
    for (size_t index = begin; index < text.size(); ++index) {
        const char character = text[index];
        if (text.compare(index, kQuoteLength, kQuote) == 0) {
            const size_t close = text.find(kQuote, index + kQuoteLength);
            if (close == std::string::npos) {
                return false;
            }
            index = close + kQuoteLength - 1;
            continue;
        }
        if (in_json_string) {
            if (character == '"') {
                in_json_string = false;
            }
            continue;
        }
        if (character == '"') {
            in_json_string = true;
        } else if (character == '{') {
            ++depth;
        } else if (character == '}') {
            --depth;
            if (depth == 0) {
                *out = text.substr(begin, index - begin + 1);
                return true;
            }
        }
    }
    return false;
}

ToolCall call_from_json(const std::string& json_text) {
    ToolCall call;
    call.raw = json_text;
    std::string error;
    const Json parsed = Json::parse(json_text, &error);
    if (!parsed.is_object()) {
        call.error = "tool call is not a JSON object";
        return call;
    }
    std::string name = parsed.string_or("name", "");
    if (name.empty()) {
        name = parsed.string_or("tool", "");
    }
    if (name.empty()) {
        call.error = "tool call has no \"name\"";
        return call;
    }
    const Json* arguments = parsed.find("arguments");
    if (arguments == nullptr) {
        arguments = parsed.find("parameters");
    }
    if (arguments == nullptr) {
        arguments = parsed.find("args");
    }
    call.name = trim(name);
    if (arguments != nullptr && arguments->is_object()) {
        call.arguments = *arguments;
    }
    call.valid = true;
    return call;
}

// Reads `call:NAME{...}` from the body of a <|tool_call> block.
std::string native_call_name(const std::string& body, size_t* arguments_begin) {
    const std::string lowered = to_lower(body);
    const size_t call = lowered.find("call:");
    if (call == std::string::npos) {
        return std::string();
    }
    size_t start = call + 5;
    while (start < body.size() && (body[start] == ' ' || body[start] == '\t')) {
        ++start;
    }
    // Tool names are identifiers.  The deployed Q4 export sometimes repeats the
    // name (`call:calculator::calculator{...}`) or appends stray tokens
    // (`call:time<|"|>`), so take the leading identifier only.
    size_t end = start;
    while (end < body.size() &&
           (std::isalnum(static_cast<unsigned char>(body[end])) != 0 ||
            body[end] == '_')) {
        ++end;
    }
    if (end == start) {
        return std::string();
    }
    size_t brace = body.find('{', end);
    const size_t parenthesis = body.find('(', end);
    if (brace == std::string::npos ||
        (parenthesis != std::string::npos && parenthesis < brace)) {
        brace = parenthesis;
    }
    *arguments_begin = brace == std::string::npos ? body.size() : brace;
    return trim(body.substr(start, end - start));
}

bool is_question(const std::string& text) {
    const std::string trimmed = trim(text);
    if (trimmed.empty()) {
        return false;
    }
    const std::string suffix = trimmed.size() >= 3 ? trimmed.substr(trimmed.size() - 3) : trimmed;
    if (trimmed.back() == '?' || suffix == "？") {
        return true;
    }
    return contains(trimmed, "吗？") || contains(trimmed, "吗?") ||
           contains(trimmed, "什么名字") || contains(trimmed, "是谁") ||
           contains(trimmed, "多少？");
}

}  // namespace

std::string format_argument(const Json& value, bool escape_keys) {
    switch (value.type()) {
        case Json::Type::String:
            return quoted(value.as_string());
        case Json::Type::Bool:
            return value.as_bool() ? "true" : "false";
        case Json::Type::Number: {
            const double number = value.as_number();
            char buffer[40];
            if (number == std::floor(number) && std::fabs(number) < 1e15) {
                std::snprintf(buffer, sizeof(buffer), "%lld", static_cast<long long>(number));
            } else {
                std::snprintf(buffer, sizeof(buffer), "%.10g", number);
            }
            return buffer;
        }
        case Json::Type::Array: {
            std::string out = "[";
            const auto& items = value.items();
            for (size_t index = 0; index < items.size(); ++index) {
                if (index != 0) {
                    out += ",";
                }
                out += format_argument(items[index], escape_keys);
            }
            out += "]";
            return out;
        }
        case Json::Type::Object: {
            std::string out = "{";
            bool first = true;
            for (const auto& member : value.members()) {
                if (!first) {
                    out += ",";
                }
                first = false;
                out += escape_keys ? quoted(member.first) : member.first;
                out += ":";
                out += format_argument(member.second, escape_keys);
            }
            out += "}";
            return out;
        }
        case Json::Type::Null:
        default:
            return quoted(std::string());
    }
}

bool parse_arguments(const std::string& text, Json* out, std::string* error) {
    ArgumentParser parser(text, error);
    return parser.parse(out);
}

std::string format_tool_declaration(const ToolSpec& spec) {
    std::ostringstream out;
    out << "<|tool>declaration:" << spec.name << "{description:" << quoted(spec.description);
    if (spec.parameters.is_object() && !spec.parameters.empty()) {
        const Json* properties = spec.parameters.find("properties");
        out << ",parameters:{properties:"
            << format_properties(properties != nullptr ? *properties : Json::object(),
                                 required_list(spec.parameters))
            << "}";
    }
    out << "}<tool|>";
    return out.str();
}

std::string format_tool_call_block(const std::string& name, const Json& arguments) {
    std::string out = std::string(kNativeToolCallOpen) + "call:" + name + "{";
    bool first = true;
    if (arguments.is_object()) {
        for (const auto& member : arguments.members()) {
            if (!first) {
                out += ",";
            }
            first = false;
            out += member.first + ":";
            out += format_argument(member.second, false);
        }
    }
    out += "}";
    out += kNativeToolCallClose;
    return out;
}

std::string format_tool_response_block(const std::string& name, bool ok,
                                       const std::string& summary, const Json& value) {
    Json payload = Json::object();
    payload.set("ok", Json::boolean(ok));
    payload.set("result", Json::string(summary));
    if (value.is_object() && !value.empty()) {
        payload.set("data", value);
    }
    std::string out = std::string(kNativeToolResponseOpen) + "response:" + name + "{";
    bool first = true;
    for (const auto& member : payload.members()) {
        if (!first) {
            out += ",";
        }
        first = false;
        out += member.first + ":";
        out += format_argument(member.second, false);
    }
    out += "}";
    out += kNativeToolResponseClose;
    return out;
}

std::string build_system_prompt(const AgentConfig& config, const ToolRegistry& tools,
                                const Json& device, bool include_tools) {
    const std::string platform = device.string_or("platform", "mt6899");
    const std::string vision = device.string_or("vision", "W8A16");
    const std::string text_model = device.string_or("text_model", "Q4");

    std::ostringstream out;
    if (!include_tools) {
        // Small "just talk to me" prompt: identity, language and style only.
        // Without the tool protocol and the declarations a full prefill costs
        // ~250 tokens less, and the Q4 export stops drifting into protocol talk.
        out << "You are Gemma Agent, an offline assistant on this phone (" << platform
            << "). Answer in the user's language.\n"
            << "Be concise and direct; the user reads the answer on a phone screen.\n"
            << "Two to six sentences, no jargon, no code fences.\n"
            // A rule, not a request, so the model cannot answer *with* it: the
            // Q4 export drifts into Korean/English on Chinese turns unless the
            // language anchor sits in the system prompt as well.
            << "\u7528\u6237\u7528\u4e2d\u6587\u63d0\u95ee\u65f6\uff0c\u5fc5\u987b\u7528"
               "\u7b80\u4f53\u4e2d\u6587\u56de\u7b54\u3002\n";
        if (config.language != "auto" && !config.language.empty()) {
            out << "Preferred answer language: " << config.language << ".\n";
        }
        if (!config.system_tips.empty()) {
            out << "\nDEPLOYMENT NOTES (always apply)\n";
            for (const auto& tip : config.system_tips) {
                out << "- " << tip << "\n";
            }
        }
        return out.str();
    }
    out << "You are Gemma Agent, an offline assistant on this phone (" << platform << ", "
        << vision << " vision NPU + " << text_model
        << " text model, no network). Answer in the user's language.\n"
        << "\u7528\u6237\u7528\u4e2d\u6587\u63d0\u95ee\u65f6\uff0c\u5fc5\u987b\u7528"
           "\u7b80\u4f53\u4e2d\u6587\u56de\u7b54\u3002\n"
        << "TOOLS: reply with exactly one line to call one:\n"
        << "<|tool_call>call:calculator{expression:<|\"|>17*23<|\"|>}<tool_call|>\n"
        << "The system then answers with "
           "<|tool_response>response:NAME{...}<tool_response|>; never write that answer "
           "yourself and never invent a result.\n"
        << "Rules:\n"
        << "1. One call per reply, at most " << config.max_tool_calls
        << " calls per user message.\n"
        << "2. Use the calculator for any arithmetic and unit_convert for units.\n"
        << "3. Call make_plan first when a task needs two or more calls.\n"
        << "4. When you know the answer, reply with plain text only: no call, no JSON.\n"
        << "5. If a tool fails, fix the arguments or answer with what you know.\n"
        << "Style: two to six sentences, direct, no mention of tools or plans.\n";
    if (config.language != "auto" && !config.language.empty()) {
        out << "- Preferred answer language: " << config.language << ".\n";
    }
    if (!config.system_tips.empty()) {
        out << "\nDEPLOYMENT NOTES (always apply)\n";
        for (const auto& tip : config.system_tips) {
            out << "- " << tip << "\n";
        }
    }
    out << "\n";
    for (const auto& definition : tools.definitions()) {
        out << format_tool_declaration(definition.spec) << "\n";
    }
    return out.str();
}

std::string build_memory_block(const std::vector<ScoredMemory>& memories) {
    if (memories.empty()) {
        return std::string();
    }
    std::ostringstream out;
    // The block is an offer, not an instruction: the phone answered "它是什么
    // 颜色的？" (about a picture) with the *remembered* favourite colour because
    // the recalled fact looked like an answer.  Saying when to ignore it is the
    // cheap fix for that whole family of mistakes.
    out << "[MEMORY] earlier sessions may have stored this about the user. Use it only when "
           "it answers the current message; ignore it otherwise:\n";
    for (const auto& memory : memories) {
        out << "- (" << memory.record.kind << " " << memory.record.id << ") "
            << memory.record.text << "\n";
    }
    out << "[END MEMORY]\n";
    return out.str();
}

std::string build_user_message(const std::string& user_text,
                               const std::vector<ScoredMemory>& memories, int image_tokens,
                               const std::string& session_notes) {
    std::string message;
    if (image_tokens > 0) {
        message += kImageStart;
        for (int index = 0; index < image_tokens; ++index) {
            message += kImageToken;
        }
        message += kImageEnd;
    }
    if (!session_notes.empty()) {
        // The Notetaker digest travels in the *current* message: putting it
        // before the history would change the prefix on every turn and throw
        // away the KV cache.
        message += "[SESSION NOTES] earlier in this conversation:\n";
        message += head(session_notes, 900);
        message += "[END SESSION NOTES]\n";
    }
    message += build_memory_block(memories);
    message += user_text;
    const std::string hint = build_task_hint(user_text, image_tokens > 0);
    if (!hint.empty()) {
        message += "\n" + hint;
    }
    return message;
}

std::string build_task_hint(const std::string& user_text, bool has_image) {
    // The hint is derived from the intent policy (agent_intent.cpp): when the
    // router could derive the exact call it is spelled out, when the turn must
    // stay plain text the model is told so, and a plain conversation gets no
    // hint at all.
    agent::ToolRegistry registry = build_default_tools();
    const IntentPolicy policy = classify_intent(user_text, has_image, registry);
    if (policy.seed.valid) {
        return "[hint] Emit exactly this line and nothing else:\n" +
               format_tool_call_block(policy.seed.name, policy.seed.arguments);
    }
    if (policy.prefer_plain_answer) {
        // No meta-instructions the Q4 export can parrot back: the phone answered
        // twelve image questions with "I will do my best to keep my answers in the
        // language you asked." because the hint *said* that.  The hint now only
        // describes the content the answer should have; language lives in the
        // system prompt, and the runtime re-asks if the answer ignores it.
        if (policy.kind == IntentKind::Greeting) {
            return std::string("[hint] This message is only a greeting: greet the user back "
                               "briefly. There is no task and no image in it.");
        }
        if (has_image) {
            // A Chinese user gets a Chinese instruction: an English hint with a
            // Chinese tag still had the export answering in Korean, while a hint
            // written in the user's language anchors the output language.
            if (is_cjk(user_text)) {
                return std::string("[\u63d0\u793a] \u56fe\u7247\u5df2\u7ecf\u5728\u4f60\u7684"
                                   "\u8f93\u5165\u91cc\u3002\u8bf7\u7528\u4e24\u4e09\u53e5"
                                   "\u8bdd\u63cf\u8ff0\u4f60\u770b\u5230\u7684\u5185"
                                   "\u5bb9\uff1a\u4e3b\u4f53\u3001\u573a\u666f\u3001"
                                   "\u989c\u8272\u3002\u4e0d\u8981\u8c03\u7528\u5de5"
                                   "\u5177\uff0c\u4e5f\u4e0d\u8981\u56de\u7b54\u5176"
                                   "\u4ed6\u8bed\u8a00\u3002");
            }
            return std::string("[hint] The image is part of your input. Describe what you can "
                               "see in it - subject, setting, colours - in two or three short "
                               "sentences. Do not call a tool.");
        }
        if (is_cjk(user_text)) {
            return std::string("[hint] Answer the message directly in two or three sentences. "
                               "Do not call a tool. 请用简体中文回答。");
        }
        return std::string("[hint] Answer the message directly in two or three sentences. "
                           "Do not call a tool.");
    }
    if (policy.kind == IntentKind::ImageAdvice) {
        return std::string("[hint] The user wants advice about the image that is already part of "
                           "your input. Reply in plain text with two or three concrete, "
                           "photography-level suggestions (light, exposure, framing, subject, "
                           "background). Do not call a tool.") +
               (is_cjk(user_text) ? " Reply in 简体中文." : "");
    }
    if (policy.kind == IntentKind::Convert) {
        return "[hint] Use the unit_convert tool with value/from/to.";
    }
    if (policy.kind == IntentKind::MemoryWrite) {
        return "[hint] The user wants this stored: call remember with the fact.";
    }
    return std::string();
}

std::string format_observation(const ToolResult& result, const std::string& plan_render,
                               int remaining_calls) {
    // Placement matters: the probe in helpers/tools/agent_live_probe.cpp shows that a
    // user turn which *starts* with a <|tool_response> block makes the model
    // close its turn immediately, while the same block after a plain-text line
    // answers normally.  So the observation leads with text and keeps the
    // native block as the machine readable tail.
    std::string summary = result.ok ? result.summary : ("error: " + result.error);
    std::string out = "TOOL_RESULT from " + result.name + " (" +
                      (result.ok ? "ok" : "failed") + "):\n" + summary + "\n";
    // Keep the machine readable half small: the observation is part of the
    // prompt on every later turn, so a tool that returns a large object is
    // represented by its summary instead.
    Json payload = result.value;
    if (!payload.is_object() || payload.dump().size() > 700) {
        payload = Json::object();
    }
    out += format_tool_response_block(result.name, result.ok, summary, payload);
    out += "\n";
    if (!plan_render.empty()) {
        out += plan_render;
    }
    out += "TOOL CALLS LEFT: " + std::to_string(remaining_calls) +
           " - reply with the next tool call, or with your final answer in plain text.";
    return out;
}

std::string build_compaction_prompt(const std::string& transcript, int max_bullets) {
    std::ostringstream out;
    out << "Summarise the conversation below so that an assistant can continue it "
           "without the original messages.\n"
        << "Keep: who the user is, their preferences, decisions already made, concrete "
           "facts (names, numbers, ids, dates) and open tasks.\n"
        << "Drop: greetings, repetition and anything already resolved.\n"
        << "Write at most " << max_bullets << " short bullet points in the language of "
           "the conversation. Output the bullets only.\n\n"
        << "CONVERSATION:\n"
        << transcript << "\n"
        << "SUMMARY:\n";
    return out.str();
}

std::string build_memory_prompt(const std::string& user_text, const std::string& answer) {
    std::ostringstream out;
    out << "Extract durable memories from one conversation turn.\n"
        << "Reply with JSON only, in this shape:\n"
        << "{\"memories\":[{\"text\":\"...\",\"kind\":\"fact|preference|note\","
           "\"importance\":0.0}]}\n"
        << "Only include something that stays true beyond this turn (identity, "
           "preferences, constraints, decisions). Use an empty array when nothing is "
           "durable. Never include questions.\n\n"
        << "USER: " << head(user_text, 800) << "\n"
        << "ASSISTANT: " << head(answer, 800) << "\n"
        << "JSON:\n";
    return out.str();
}

std::string build_forced_answer_prompt(const std::string& reason) {
    return "You have reached the limit (" + reason +
           "). Answer the user now with the best information you already have. Plain "
           "text only, no tool call.";
}

ToolCall parse_tool_call(const std::string& text, const ToolRegistry* tools) {
    ToolCall none;
    none.error = "no tool call";
    const std::string cleaned = strip_thinking(text);
    const std::string& body_text = cleaned;

    // 1. The native gemma4 form: <|tool_call>call:NAME{args}<tool_call|>
    size_t open = body_text.find(kNativeToolCallOpen);
    if (open == std::string::npos) {
        open = text.find("call:");
    }
    if (open != std::string::npos) {
        const size_t body_begin =
            text.compare(open, std::strlen(kNativeToolCallOpen), kNativeToolCallOpen) == 0
                ? open + std::strlen(kNativeToolCallOpen)
                : open;
        const size_t close = text.find(kNativeToolCallClose, body_begin);
        const std::string body = close == std::string::npos
            ? text.substr(body_begin)
            : text.substr(body_begin, close - body_begin);
        size_t arguments_begin = std::string::npos;
        const std::string name = native_call_name(body, &arguments_begin);
        if (!name.empty() && arguments_begin < body.size()) {
            std::string arguments_text;
            if (extract_braces(body, arguments_begin, &arguments_text)) {
                ToolCall call;
                call.name = name;
                call.raw = body;
                std::string error;
                if (parse_arguments(arguments_text, &call.arguments, &error)) {
                    call.valid = true;
                    return call;
                }
                call.error = "cannot parse the arguments: " + error;
                return call;
            }
        }
        if (!name.empty()) {
            ToolCall call;
            call.name = name;
            call.raw = body;
            call.arguments = Json::object();
            call.valid = true;
            return call;
        }
    }

    // 2. The textual fallback: TOOL_CALL {...}
    const std::string lowered = to_lower(text);
    const size_t marker = lowered.find(to_lower(kToolCallMarker));
    if (marker != std::string::npos) {
        const size_t begin = text.find('{', marker);
        if (begin != std::string::npos) {
            std::string object;
            if (extract_braces(text, begin, &object)) {
                return call_from_json(object);
            }
        }
        ToolCall broken;
        broken.error = "TOOL_CALL marker without a complete JSON object";
        return broken;
    }

    // 3. A bare JSON object that carries a tool name.
    size_t begin = body_text.find('{');
    while (begin != std::string::npos) {
        std::string object;
        if (extract_braces(body_text, begin, &object)) {
            const Json parsed = Json::parse(object);
            if (parsed.is_object() && (parsed.has("name") || parsed.has("tool"))) {
                return call_from_json(object);
            }
        }
        begin = text.find('{', begin + 1);
    }

    // 4. Recovery for the phone: the Q4 export sometimes writes the call without
    //    the native markers, as a bare first line `calculator:17*23`.  Only a
    //    line whose head resolves to a registered tool is accepted, and the
    //    remainder becomes that tool's primary argument.
    if (tools != nullptr) {
        const std::string first_line = trim(text.substr(0, text.find('\n')));
        const size_t separator = first_line.find_first_of(":{");
        if (separator != std::string::npos && separator > 0) {
            const std::string head = trim(first_line.substr(0, separator));
            bool identifier = !head.empty();
            for (const char character : head) {
                identifier = identifier &&
                             (std::isalnum(static_cast<unsigned char>(character)) != 0 ||
                              character == '_');
            }
            const std::string canonical = identifier ? tools->resolve(head, nullptr)
                                                     : std::string();
            const ToolSpec* spec =
                canonical.empty() ? nullptr : tools->spec(canonical);
            std::string remainder = trim(first_line.substr(separator + 1));
            if (spec != nullptr && !remainder.empty() && !spec->primary_argument.empty()) {
                ToolCall call;
                call.name = canonical;
                call.raw = first_line;
                if (remainder.front() == '{') {
                    std::string object;
                    std::string error;
                    if (extract_braces(remainder, 0, &object) &&
                        parse_arguments(object, &call.arguments, &error)) {
                        call.valid = true;
                        return call;
                    }
                    return none;
                }
                call.arguments.set(spec->primary_argument, Json::string(remainder));
                call.valid = true;
                return call;
            }
        }
    }
    return none;
}

std::string strip_thinking(const std::string& text) {
    std::string out;
    size_t position = 0;
    while (position <= text.size()) {
        const size_t separator = text.find("<channel|>", position);
        const std::string part =
            separator == std::string::npos ? text.substr(position)
                                           : text.substr(position, separator - position);
        const size_t channel = part.find("<|channel>");
        out += channel == std::string::npos ? part : part.substr(0, channel);
        if (separator == std::string::npos) {
            break;
        }
        position = separator + 10;
    }
    // A dangling opening marker (generation stopped mid-thought).
    const size_t dangling = out.find("<|channel>");
    if (dangling != std::string::npos) {
        out = out.substr(0, dangling);
    }
    return trim(out);
}

std::string strip_tool_call(const std::string& text) {
    std::string out = text;
    const size_t open = out.find(kNativeToolCallOpen);
    if (open != std::string::npos) {
        const size_t close = out.find(kNativeToolCallClose, open);
        const size_t end =
            close == std::string::npos ? out.size() : close + std::strlen(kNativeToolCallClose);
        out.erase(open, end - open);
    }
    const size_t marker = to_lower(out).find(to_lower(kToolCallMarker));
    if (marker != std::string::npos) {
        const size_t newline = out.find('\n', marker);
        out.erase(marker, newline == std::string::npos ? std::string::npos : newline - marker);
    }
    return trim(out);
}

std::vector<ExtractedMemory> extract_memories(const std::string& user_text,
                                              const std::string& answer,
                                              const std::string& language) {
    (void)language;
    std::vector<ExtractedMemory> memories;
    const std::string collapsed = collapse_whitespace(user_text);
    if (collapsed.empty()) {
        return memories;
    }
    // A correction replaces what was said before: keep only the corrected part
    // ("记住我叫李雷，改成记住我叫杨雷" stores the 杨雷 statement) and flag it so
    // the store overwrites the conflicting record instead of adding a second one.
    std::string subject = collapsed;
    bool correction = false;
    static const char* kCorrections[] = {"改成", "改为", "更正", "纠正", "改一下",
                                         "instead", "actually"};
    size_t last_marker = std::string::npos;
    size_t marker_length = 0;
    const std::string lowered_subject = to_lower(collapsed);
    for (const char* marker : kCorrections) {
        size_t position = lowered_subject.rfind(to_lower(marker));
        if (position != std::string::npos && (last_marker == std::string::npos ||
                                              position > last_marker)) {
            last_marker = position;
            marker_length = std::strlen(marker);
        }
    }
    if (last_marker != std::string::npos) {
        const std::string tail =
            trim(collapsed.substr(last_marker + marker_length));
        if (!tail.empty()) {
            subject = tail;
            correction = true;
        }
    }
    const std::string lowered = to_lower(subject);
    const bool explicit_request =
        contains(subject, "记住") || contains(subject, "记一下") ||
        contains(subject, "别忘") || contains(subject, "帮我记") ||
        contains(lowered, "remember") || contains(lowered, "don't forget") ||
        contains(lowered, "do not forget") || contains(lowered, "keep in mind");
    // Store the fact, not the instruction that asked for it: "记住: 我叫杨雷，是
    // 一名嵌入式工程师。" belongs in the store as "我叫杨雷，是一名嵌入式工程师。".
    // The phone compared the two spellings and both have to behave the same.
    auto strip_imperative = [](const std::string& value) {
        std::string trimmed = trim(value);
        static const char* kPrefixes[] = {
            "请记住：",     "请记住:",     "请记住",       "记住：",   "记住:",   "记住",
            "帮我记住：",   "帮我记住:",   "帮我记住",     "记一下：", "记一下:", "记一下",
            "别忘：",       "别忘:",       "别忘",         "remember that ", "remember: ",
            "remember ",    "don't forget: ", "don't forget ", "do not forget: ",
            "do not forget ", "keep in mind: ", "keep in mind "};
        for (const char* prefix : kPrefixes) {
            const size_t length = std::strlen(prefix);
            const bool ascii_prefix = static_cast<unsigned char>(prefix[0]) < 0x80;
            const std::string head_text =
                ascii_prefix ? to_lower(trimmed.substr(0, length)) : trimmed.substr(0, length);
            const std::string wanted = ascii_prefix ? to_lower(prefix) : prefix;
            if (head_text == wanted) {
                trimmed = trim(trimmed.substr(length));
                break;
            }
        }
        return trimmed;
    };
    subject = strip_imperative(subject);
    if (subject.empty()) {
        return memories;
    }
    if (explicit_request) {
        ExtractedMemory memory;
        memory.text = subject;
        memory.kind = "fact";
        memory.importance = 0.9;
        memory.correction = correction;
        memories.push_back(std::move(memory));
    } else if (!is_question(subject)) {
        static const char* kIdentity[] = {"我叫",  "我的名字", "我是", "我住在",
                                          "我的手机", "my name is", "i am ",
                                          "i'm ",  "i live in",   "call me"};
        static const char* kPreference[] = {"我喜欢", "我偏好", "我习惯", "我讨厌",
                                            "我不喜欢", "i like", "i prefer",
                                            "i hate",   "i usually", "i always"};
        for (const char* pattern : kIdentity) {
            if (contains(subject, pattern) || contains(lowered, pattern)) {
                ExtractedMemory memory;
                memory.text = head(subject, 240);
                memory.kind = "fact";
                memory.importance = 0.8;
                memory.correction = correction;
                memories.push_back(std::move(memory));
                break;
            }
        }
        if (memories.empty()) {
            for (const char* pattern : kPreference) {
                if (contains(subject, pattern) || contains(lowered, pattern)) {
                    ExtractedMemory memory;
                    memory.text = head(subject, 240);
                    memory.kind = "preference";
                    memory.importance = 0.7;
                    memory.correction = correction;
                    memories.push_back(std::move(memory));
                    break;
                }
            }
        }
    }
    // Episodes are recorded for every substantive turn (they are what session
    // recall searches); facts and preferences never come from a question.
    if (memories.empty() && !answer.empty() && utf8_length(collapsed) >= 8) {
        ExtractedMemory episode;
        episode.text = "Q: " + head(collapsed, 160) + " -> A: " +
                       head(collapse_whitespace(answer), 200);
        episode.kind = "episode";
        episode.importance = 0.3;
        memories.push_back(std::move(episode));
    }
    return memories;
}

std::vector<ExtractedMemory> parse_memory_json(const std::string& text) {
    std::vector<ExtractedMemory> memories;
    const size_t begin = text.find('{');
    if (begin == std::string::npos) {
        return memories;
    }
    std::string object;
    if (!extract_braces(text, begin, &object)) {
        return memories;
    }
    const Json parsed = Json::parse(object);
    const Json* list = parsed.find("memories");
    if (list == nullptr || !list->is_array()) {
        return memories;
    }
    for (const auto& entry : list->items()) {
        const std::string text_value = entry.string_or("text", "");
        const std::string kind = entry.string_or("kind", "note");
        if (text_value.empty()) {
            continue;
        }
        if (kind != "fact" && kind != "preference" && kind != "note" && kind != "episode") {
            continue;
        }
        ExtractedMemory memory;
        memory.text = head(collapse_whitespace(text_value), 240);
        memory.kind = kind;
        memory.importance = std::min(1.0, std::max(0.0, entry.number_or("importance", 0.6)));
        memories.push_back(std::move(memory));
    }
    return memories;
}

}  // namespace agent
