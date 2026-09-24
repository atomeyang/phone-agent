#include "agent_intent.h"

#include "agent_prompt.h"
#include "agent_util.h"

#include <algorithm>
#include <cctype>
#include <cstring>

namespace agent {
namespace {

bool any_of(const std::string& text, const std::string& lowered,
            std::initializer_list<const char*> needles) {
    for (const char* needle : needles) {
        if (contains(text, needle) || contains(lowered, needle)) {
            return true;
        }
    }
    return false;
}

// Extracts the arithmetic expression of a sentence ("算一下 (12+8)/4" -> "(12+8)/4").
std::string find_expression(const std::string& text) {
    std::vector<std::string> candidates;
    std::string current;
    auto flush = [&]() {
        const std::string trimmed = trim(current);
        int digits = 0;
        bool operators = false;
        for (const char character : trimmed) {
            if (character >= '0' && character <= '9') {
                ++digits;
            } else if (character == '+' || character == '-' || character == '*' ||
                       character == '/' || character == '%' || character == '^') {
                operators = true;
            }
        }
        if (digits >= 2 && operators && trimmed.size() <= 64) {
            candidates.push_back(trimmed);
        }
        current.clear();
    };
    for (const char character : text) {
        const bool allowed = (character >= '0' && character <= '9') || character == '+' ||
                             character == '-' || character == '*' || character == '/' ||
                             character == '%' || character == '^' || character == '(' ||
                             character == ')' || character == '.' || character == ' ' ||
                             character == 'x' || character == 'X';
        if (!allowed) {
            flush();
            continue;
        }
        current.push_back(character == 'x' || character == 'X' ? '*' : character);
    }
    flush();
    std::string best;
    for (const auto& candidate : candidates) {
        if (candidate.size() > best.size()) {
            best = candidate;
        }
    }
    return replace_all(best, " ", "");
}

// Unit words as people write them, mapped onto the names unit_convert knows.
// Longest first so 厘米 wins over 米 and 千米 over 米; bare single letters are
// left out on purpose (a lone "m" matches inside ordinary words).
struct UnitWord {
    const char* word;
    const char* unit;
};

const UnitWord kUnitWords[] = {
    {"公里", "km"},        {"千米", "km"},      {"英里", "mile"},   {"厘米", "cm"},
    {"毫米", "mm"},        {"英尺", "ft"},      {"英寸", "inch"},   {"公斤", "kg"},
    {"千克", "kg"},        {"毫克", "mg"},      {"摄氏度", "celsius"},
    {"华氏度", "fahrenheit"}, {"摄氏", "celsius"}, {"华氏", "fahrenheit"},
    {"开尔文", "kelvin"},  {"毫升", "ml"},      {"加仑", "gal"},    {"克", "g"},
    {"磅", "lb"},          {"升", "l"},         {"米", "m"},
    {"kilometers", "km"},  {"kilometres", "km"}, {"kilometer", "km"},
    {"centimeters", "cm"}, {"millimeters", "mm"}, {"kilograms", "kg"},
    {"kilogram", "kg"},    {"milligrams", "mg"}, {"grams", "g"},
    {"gram", "g"},         {"meters", "m"},     {"meter", "m"},
    {"miles", "mile"},     {"pounds", "lb"},    {"pound", "lb"},
    {"inches", "inch"},    {"feet", "ft"},      {"celsius", "celsius"},
    {"fahrenheit", "fahrenheit"}, {"kelvin", "kelvin"}, {"liters", "l"},
    {"litres", "l"},       {"milliliters", "ml"}, {"gallons", "gal"},
    {"km", "km"},          {"cm", "cm"},        {"mm", "mm"},
    {"kg", "kg"},          {"mg", "mg"},        {"ml", "ml"},
    {"lb", "lb"},          {"ft", "ft"},        {"mph", "mph"},
    {"gal", "gal"},
};

std::vector<std::string> unit_mentions(const std::string& lowered) {
    std::vector<std::string> found;
    size_t position = 0;
    while (position < lowered.size()) {
        bool matched = false;
        for (const auto& entry : kUnitWords) {
            const size_t length = std::strlen(entry.word);
            if (lowered.compare(position, length, entry.word) != 0) {
                continue;
            }
            // A trailing ASCII letter would mean we are inside a longer word
            // ("g" in "grams" is fine because grams itself matches first).
            if (position + length < lowered.size() &&
                std::isalpha(static_cast<unsigned char>(lowered[position + length])) != 0) {
                continue;
            }
            found.push_back(entry.unit);
            position += length;
            matched = true;
            break;
        }
        if (!matched) {
            ++position;
        }
    }
    return found;
}

// "100 华氏度是多少摄氏度" / "5 km in miles" / "3 公斤等于多少克": one number and
// two units is all unit_convert needs, so the runtime can spell the call out.
bool find_conversion(const std::string& text, const std::string& lowered, double* value,
                     std::string* from, std::string* to) {
    const size_t number = lowered.find_first_of("0123456789");
    if (number == std::string::npos) {
        return false;
    }
    size_t end = number;
    while (end < text.size() &&
           (std::isdigit(static_cast<unsigned char>(text[end])) != 0 || text[end] == '.')) {
        ++end;
    }
    const std::string digits = text.substr(number, end - number);
    try {
        *value = std::stod(digits);
    } catch (const std::exception&) {
        return false;
    }
    const std::vector<std::string> mentions = unit_mentions(lowered);
    for (size_t index = 0; index + 1 < mentions.size(); ++index) {
        if (mentions[index] != mentions[index + 1]) {
            *from = mentions[index];
            *to = mentions[index + 1];
            return true;
        }
    }
    return false;
}

// "统计一下这段文字的字数：..." - everything after the colon is the payload.
bool find_count_request(const std::string& text, std::string* payload) {
    const std::string lowered = to_lower(text);
    const bool asked = any_of(text, lowered, {"统计", "字数", "多少个字", "多少字", "字符数",
                                              "word count", "count the characters",
                                              "count the words", "how many characters",
                                              "how many words"});
    if (!asked) {
        return false;
    }
    for (const char* separator : {u8"：", ":"}) {
        const size_t position = text.find(separator);
        if (position != std::string::npos) {
            const std::string tail = trim(text.substr(position + std::strlen(separator)));
            if (!tail.empty()) {
                *payload = tail;
                return true;
            }
        }
    }
    *payload = text;
    return true;
}

ToolCall seeded_call(const std::string& tool, const Json& arguments) {
    ToolCall call;
    call.name = tool;
    call.arguments = arguments;
    call.raw = format_tool_call_block(tool, arguments);
    call.valid = true;
    return call;
}

}  // namespace

const char* to_string(IntentKind kind) {
    switch (kind) {
        case IntentKind::Chat: return "chat";
        case IntentKind::Greeting: return "greeting";
        case IntentKind::Arithmetic: return "arithmetic";
        case IntentKind::Clock: return "clock";
        case IntentKind::Convert: return "convert";
        case IntentKind::TextStats: return "text_stats";
        case IntentKind::MemoryWrite: return "memory_write";
        case IntentKind::MemoryRecall: return "memory_recall";
        case IntentKind::ImageDescribe: return "image_describe";
        case IntentKind::ImageQuestion: return "image_question";
        case IntentKind::ImageAdvice: return "image_advice";
        case IntentKind::MultiStep: return "multi_step";
    }
    return "chat";
}

IntentPolicy classify_intent(const std::string& user_text, bool has_image,
                             const ToolRegistry& tools) {
    IntentPolicy policy;
    const std::string text = trim(user_text);
    const std::string lowered = to_lower(text);
    const std::string compact =
        replace_all(replace_all(lowered, " ", ""), "!", "?");
    (void)tools;

    const bool asks_to_remember =
        any_of(text, lowered, {"记住", "记一下", "别忘", "帮我记", "remember", "don't forget"});
    const bool asks_about_self =
        any_of(text, lowered, {"我叫什么", "我是谁", "我的名字", "你记得", "还记得", "我之前",
                               "我说过", "what do you know about me", "do you remember",
                               "what is my name", "who am i", "我是做什么", "我做什么工作",
                               "我的职业", "我的工作", "我干什么", "what do i do",
                               "what is my job", "what's my job", "my job"});
    const bool asks_time =
        any_of(text, lowered, {"几点", "什么时间", "现在时间", "今天几号", "what time",
                               "the time", "current time", "today's date", "what day"});
    const bool asks_conversion =
        any_of(text, lowered, {"换算", "转换", "convert", "in miles", "in km", "多少公里",
                               "多少米", "多少公斤", "in celsius", "in fahrenheit"});
    const bool multi_step =
        any_of(text, lowered, {"先", "然后", "再", "步骤", "一步一步", "step by step",
                               "multi-step", "first,", "then "});

    // 1. Memory writes: the user is telling the agent something durable.
    if (asks_to_remember) {
        policy.kind = IntentKind::MemoryWrite;
        policy.label = "记住信息";
        policy.confidence = 0.9;
        policy.preferred_tool = "remember";
        policy.allow_tools = true;
        return policy;
    }
    // 1b. A greeting is not a task: answer it, do not reach for tools and do not
    //     drag an earlier image into the reply (the phone saw exactly that -
    //     "Nihao" after an image turn produced "what do you want me to do with
    //     the image").
    const bool greeting =
        any_of(text, lowered, {"nihao", "ni hao", "hello", "hi there", "hey", "good morning",
                               "good evening", "你好", "您好", "早上好", "晚上好", "哈喽",
                               "在吗", "hola"}) ||
        compact == "hi" || compact == "hello" || compact == "nihao" || compact == "hey";
    if (greeting && !has_image && utf8_length(text) <= 12) {
        policy.kind = IntentKind::Greeting;
        policy.label = "问候";
        policy.confidence = 0.8;
        policy.inject_memory = false;
        policy.allow_tools = false;
        policy.prefer_plain_answer = true;
        return policy;
    }
    // 2. Memory reads: the question is about the user, so consult the store.
    if (asks_about_self) {
        policy.kind = IntentKind::MemoryRecall;
        policy.label = "回忆用户信息";
        policy.confidence = 0.85;
        policy.inject_memory = true;
        policy.allow_tools = false;
        policy.prefer_plain_answer = true;
        return policy;
    }
    // 3. Arithmetic: the calculator is expected, not the model's own maths.
    //    Units and text counting come first: both carry numbers, and the model
    //    should call their tool rather than do the work itself.
    double convert_value = 0.0;
    std::string convert_from;
    std::string convert_to;
    if (find_conversion(text, lowered, &convert_value, &convert_from, &convert_to)) {
        Json arguments = Json::object();
        arguments.set("value", Json::number(convert_value));
        arguments.set("from", Json::string(convert_from));
        arguments.set("to", Json::string(convert_to));
        policy.kind = IntentKind::Convert;
        policy.label = "单位换算";
        policy.confidence = 0.85;
        policy.preferred_tool = "unit_convert";
        policy.seed = seeded_call("unit_convert", arguments);
        return policy;
    }
    std::string count_payload;
    if (find_count_request(text, &count_payload)) {
        Json arguments = Json::object();
        arguments.set("text", Json::string(count_payload));
        policy.kind = IntentKind::TextStats;
        policy.label = "字数统计";
        policy.confidence = 0.8;
        policy.preferred_tool = "text_stats";
        policy.seed = seeded_call("text_stats", arguments);
        return policy;
    }
    const std::string expression = find_expression(text);
    // A question that carries a real operator ("What is 17*23?") is arithmetic
    // even without the usual keywords; the keyword list stays for the phrasings
    // that hide the operator behind words ("how much is ... times ...").  A bare
    // hyphen is excluded so a date ("2026-09-22") is not read as a subtraction.
    const bool arithmetic_keyword =
        any_of(text, lowered, {"=", "等于", "多少", "计算", "算一算", "算一下", "calculate",
                               "how much", "times", "plus", "multiplied"});
    const bool question_shaped = contains(text, "?") || contains(text, "？");
    const bool hard_operator = expression.find_first_of("*/+^%") != std::string::npos;
    if (!expression.empty() && (arithmetic_keyword || (question_shaped && hard_operator))) {
        Json arguments = Json::object();
        arguments.set("expression", Json::string(expression));
        policy.kind = IntentKind::Arithmetic;
        policy.label = "算术计算";
        policy.confidence = 0.9;
        policy.preferred_tool = "calculator";
        policy.seed = seeded_call("calculator", arguments);
        return policy;
    }
    // 4. Clock.
    if (asks_time) {
        policy.kind = IntentKind::Clock;
        policy.label = "查询时间";
        policy.confidence = 0.85;
        policy.preferred_tool = "now";
        policy.seed = seeded_call("now", Json::object());
        return policy;
    }
    // 5. Units.
    if (asks_conversion) {
        policy.kind = IntentKind::Convert;
        policy.label = "单位换算";
        policy.confidence = 0.6;
        policy.preferred_tool = "unit_convert";
        return policy;
    }
    // 6. Images: describing is a plain-text task; a specific question may use
    //    image_info, but never the empty "describe" case.
    if (has_image) {
        // Advice/edit requests first: they are image turns too, but the answer
        // must reason about the picture instead of calling a tool.
        const bool advice =
            any_of(text, lowered, {"make it better", "make this better", "make it brighter",
                                   "improve", "enhance", "edit this", "edit the image",
                                   "怎么改", "怎么调", "修图", "调亮", "改善", "更好看",
                                   "how to make"});
        if (advice) {
            policy.kind = IntentKind::ImageAdvice;
            policy.label = "图像改进建议";
            policy.confidence = 0.8;
            policy.allow_tools = false;
            policy.prefer_plain_answer = true;
            return policy;
        }
        // Only describing *verbs* count: "is the light red in this image?" is a
        // question about the image, not a request to describe it.
        const bool describes =
            any_of(text, lowered, {"描述", "看看", "有什么", "什么内容", "what can you see",
                                   "what do you see", "what is in", "what's in", "describe",
                                   "tell me about", "这张图", "这张图片", "what is this",
                                   "what's this", "what is that", "describe it",
                                   "describe the image", "describe this", "这是什么",
                                   "说什么", "它是什么"});
        if (describes) {
            policy.kind = IntentKind::ImageDescribe;
            policy.label = "描述图片";
            policy.confidence = 0.85;
            policy.allow_tools = false;
            policy.prefer_plain_answer = true;
            return policy;
        }
        policy.kind = IntentKind::ImageQuestion;
        policy.label = "图片问答";
        policy.confidence = 0.7;
        policy.allow_tools = true;
        return policy;
    }
    // 7. Multi-step tasks keep the tools, but memory stays out unless asked.
    if (multi_step) {
        policy.kind = IntentKind::MultiStep;
        policy.label = "多步任务";
        policy.confidence = 0.6;
        policy.allow_tools = true;
        return policy;
    }
    // 8. Everything else is conversation: no tools expected, no memory lookup.
    policy.kind = IntentKind::Chat;
    policy.label = "直接对话";
    policy.confidence = 0.5;
    policy.inject_memory = false;
    policy.allow_tools = true;   // the model may still ask for a tool
    return policy;
}

}  // namespace agent
