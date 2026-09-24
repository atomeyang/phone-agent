#include "agent_tools.h"

#include "agent_util.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <set>
#include <sstream>

namespace agent {
namespace {

// --- calculator -------------------------------------------------------------

class ExpressionParser {
public:
    ExpressionParser(const std::string& text, bool* ok, std::string* error)
        : text_(text), ok_(ok), error_(error) {}

    double parse() {
        const double value = parse_sum();
        skip_space();
        if (position_ != text_.size() && *ok_) {
            fail("unexpected token in expression");
        }
        return *ok_ ? value : 0.0;
    }

private:
    void fail(const std::string& message) {
        if (*ok_) {
            *ok_ = false;
            *error_ = message;
        }
    }

    void skip_space() {
        while (position_ < text_.size() &&
               (text_[position_] == ' ' || text_[position_] == '\t')) {
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

    bool consume(const char* word) {
        skip_space();
        const size_t length = std::strlen(word);
        if (text_.compare(position_, length, word) == 0) {
            position_ += length;
            return true;
        }
        return false;
    }

    double parse_sum() {
        double value = parse_product();
        while (*ok_) {
            if (consume('+')) {
                value += parse_product();
            } else if (consume('-')) {
                value -= parse_product();
            } else {
                break;
            }
        }
        return value;
    }

    double parse_product() {
        double value = parse_power();
        while (*ok_) {
            skip_space();
            if (position_ < text_.size() && text_[position_] == '*' &&
                !(position_ + 1 < text_.size() && text_[position_ + 1] == '*')) {
                ++position_;
                value *= parse_power();
            } else if (consume('/')) {
                const double divisor = parse_power();
                if (std::fabs(divisor) < 1e-12) {
                    fail("division by zero");
                    return 0.0;
                }
                value /= divisor;
            } else if (consume('%')) {
                const double divisor = parse_power();
                if (std::fabs(divisor) < 1e-12) {
                    fail("modulo by zero");
                    return 0.0;
                }
                value = std::fmod(value, divisor);
            } else {
                break;
            }
        }
        return value;
    }

    double parse_power() {
        const double base = parse_unary();
        if (consume("**") || consume('^')) {
            return std::pow(base, parse_power());
        }
        return base;
    }

    double parse_unary() {
        skip_space();
        if (consume('-')) {
            return -parse_unary();
        }
        if (consume('+')) {
            return parse_unary();
        }
        return parse_primary();
    }

    double parse_primary() {
        skip_space();
        if (position_ >= text_.size()) {
            fail("unexpected end of expression");
            return 0.0;
        }
        if (consume('(')) {
            const double value = parse_sum();
            if (!consume(')')) {
                fail("missing ')'");
                return 0.0;
            }
            return value;
        }
        if (std::isdigit(static_cast<unsigned char>(text_[position_])) != 0 ||
            text_[position_] == '.') {
            const char* begin = text_.c_str() + position_;
            char* end = nullptr;
            const double value = std::strtod(begin, &end);
            position_ += static_cast<size_t>(end - begin);
            return value;
        }
        if (std::isalpha(static_cast<unsigned char>(text_[position_])) != 0 ||
            text_[position_] == '_') {
            const size_t begin = position_;
            while (position_ < text_.size() &&
                   (std::isalnum(static_cast<unsigned char>(text_[position_])) != 0 ||
                    text_[position_] == '_')) {
                ++position_;
            }
            const std::string name = to_lower(text_.substr(begin, position_ - begin));
            if (name == "pi") {
                return 3.14159265358979323846;
            }
            if (name == "e") {
                return 2.71828182845904523536;
            }
            if (!consume('(')) {
                fail("unknown constant '" + name + "'");
                return 0.0;
            }
            std::vector<double> arguments;
            if (!consume(')')) {
                while (true) {
                    arguments.push_back(parse_sum());
                    if (consume(',')) {
                        continue;
                    }
                    if (consume(')')) {
                        break;
                    }
                    fail("missing ')' in call to " + name);
                    return 0.0;
                }
            }
            return apply_function(name, arguments);
        }
        fail(std::string("unexpected character '") + text_[position_] + "'");
        return 0.0;
    }

    double apply_function(const std::string& name, const std::vector<double>& arguments) {
        auto need = [&](size_t count) {
            if (arguments.size() != count) {
                fail(name + "() expects " + std::to_string(count) + " argument(s)");
                return false;
            }
            return true;
        };
        if (name == "sqrt") {
            return need(1) ? std::sqrt(std::max(0.0, arguments[0])) : 0.0;
        }
        if (name == "abs") {
            return need(1) ? std::fabs(arguments[0]) : 0.0;
        }
        if (name == "round") {
            return need(1) ? std::round(arguments[0]) : 0.0;
        }
        if (name == "floor") {
            return need(1) ? std::floor(arguments[0]) : 0.0;
        }
        if (name == "ceil") {
            return need(1) ? std::ceil(arguments[0]) : 0.0;
        }
        if (name == "exp") {
            return need(1) ? std::exp(arguments[0]) : 0.0;
        }
        if (name == "ln") {
            return need(1) && arguments[0] > 0 ? std::log(arguments[0]) : 0.0;
        }
        if (name == "log" || name == "log10") {
            return need(1) && arguments[0] > 0 ? std::log10(arguments[0]) : 0.0;
        }
        if (name == "log2") {
            return need(1) && arguments[0] > 0 ? std::log2(arguments[0]) : 0.0;
        }
        if (name == "sin") {
            return need(1) ? std::sin(arguments[0]) : 0.0;
        }
        if (name == "cos") {
            return need(1) ? std::cos(arguments[0]) : 0.0;
        }
        if (name == "tan") {
            return need(1) ? std::tan(arguments[0]) : 0.0;
        }
        if (name == "pow") {
            return need(2) ? std::pow(arguments[0], arguments[1]) : 0.0;
        }
        if (name == "min") {
            return arguments.empty() ? 0.0
                                     : *std::min_element(arguments.begin(), arguments.end());
        }
        if (name == "max") {
            return arguments.empty() ? 0.0
                                     : *std::max_element(arguments.begin(), arguments.end());
        }
        fail("unknown function '" + name + "'");
        return 0.0;
    }

    const std::string& text_;
    bool* ok_;
    std::string* error_;
    size_t position_ = 0;
};

// --- units ------------------------------------------------------------------

struct UnitDefinition {
    const char* name;
    const char* category;
    double factor;      // multiply to reach the category base unit
    double offset;      // applied after the factor (temperatures)
};

const UnitDefinition kUnits[] = {
    {"mm", "length", 0.001, 0.0},   {"cm", "length", 0.01, 0.0},
    {"m", "length", 1.0, 0.0},      {"km", "length", 1000.0, 0.0},
    {"inch", "length", 0.0254, 0.0}, {"in", "length", 0.0254, 0.0},
    {"ft", "length", 0.3048, 0.0},  {"foot", "length", 0.3048, 0.0},
    {"yd", "length", 0.9144, 0.0},  {"mile", "length", 1609.344, 0.0},
    {"mi", "length", 1609.344, 0.0},
    {"mg", "mass", 1e-6, 0.0},      {"g", "mass", 0.001, 0.0},
    {"kg", "mass", 1.0, 0.0},       {"t", "mass", 1000.0, 0.0},
    {"lb", "mass", 0.45359237, 0.0}, {"oz", "mass", 0.028349523125, 0.0},
    {"c", "temperature", 1.0, 273.15},   {"celsius", "temperature", 1.0, 273.15},
    {"f", "temperature", 5.0 / 9.0, 255.3722222222222},
    {"fahrenheit", "temperature", 5.0 / 9.0, 255.3722222222222},
    {"k", "temperature", 1.0, 0.0},      {"kelvin", "temperature", 1.0, 0.0},
    {"ms", "time", 0.001, 0.0},     {"s", "time", 1.0, 0.0},
    {"sec", "time", 1.0, 0.0},      {"min", "time", 60.0, 0.0},
    {"h", "time", 3600.0, 0.0},     {"hour", "time", 3600.0, 0.0},
    {"day", "time", 86400.0, 0.0},
    {"b", "data", 1.0, 0.0},        {"kb", "data", 1024.0, 0.0},
    {"mb", "data", 1048576.0, 0.0}, {"gb", "data", 1073741824.0, 0.0},
    {"tb", "data", 1099511627776.0, 0.0},
    {"mps", "speed", 1.0, 0.0},     {"kmh", "speed", 1000.0 / 3600.0, 0.0},
    {"kph", "speed", 1000.0 / 3600.0, 0.0}, {"mph", "speed", 0.44704, 0.0},
    {"knot", "speed", 0.514444, 0.0},
    {"ml", "volume", 0.001, 0.0},   {"l", "volume", 1.0, 0.0},
    {"gal", "volume", 3.785411784, 0.0},
    {"byte", "data", 1.0, 0.0},     {"bit", "data", 0.125, 0.0},
};

const UnitDefinition* find_unit(const std::string& name) {
    const std::string lowered = to_lower(trim(name));
    for (const auto& unit : kUnits) {
        if (lowered == unit.name) {
            return &unit;
        }
    }
    return nullptr;
}

// --- helpers ----------------------------------------------------------------

ToolResult failure(const std::string& name, const std::string& message) {
    ToolResult result;
    result.name = name;
    result.ok = false;
    result.error = message;
    result.summary = "error: " + message;
    return result;
}

size_t edit_distance(const std::string& left, const std::string& right) {
    std::vector<size_t> previous(right.size() + 1);
    std::vector<size_t> current(right.size() + 1);
    for (size_t index = 0; index <= right.size(); ++index) {
        previous[index] = index;
    }
    for (size_t row = 1; row <= left.size(); ++row) {
        current[0] = row;
        for (size_t column = 1; column <= right.size(); ++column) {
            const size_t substitution = previous[column - 1] +
                                        (left[row - 1] == right[column - 1] ? 0 : 1);
            current[column] = std::min(
                {previous[column] + 1, current[column - 1] + 1, substitution});
        }
        previous.swap(current);
    }
    return previous[right.size()];
}

ToolResult success(const std::string& name, Json value, const std::string& summary) {
    ToolResult result;
    result.name = name;
    result.ok = true;
    result.value = std::move(value);
    result.summary = summary;
    return result;
}

std::string format_number(double value) {
    char buffer[64];
    if (value == std::floor(value) && std::fabs(value) < 1e15) {
        std::snprintf(buffer, sizeof(buffer), "%.0f", value);
    } else {
        std::snprintf(buffer, sizeof(buffer), "%.6g", value);
    }
    return buffer;
}

Json required_string(const Json& arguments, const std::string& key) {
    const Json* value = arguments.find(key);
    return value != nullptr ? *value : Json();
}

// Reads every session transcript under `root` and returns matching turns.
Json search_transcripts(const std::string& sessions_dir, const std::string& query, int top_k,
                        const std::string& exclude_session) {
    Json matches = Json::array();
    const auto tokens = MemoryStore::tokenize(query);
    std::set<std::string> unique(tokens.begin(), tokens.end());
    if (unique.empty()) {
        return matches;
    }
    struct Hit {
        double score;
        Json entry;
    };
    std::vector<Hit> hits;
    for (const auto& session : list_directory(sessions_dir)) {
        const std::string path = path_join(path_join(sessions_dir, session), "turns.jsonl");
        std::string contents;
        if (!read_file(path, &contents)) {
            continue;
        }
        std::istringstream stream(contents);
        std::string line;
        while (std::getline(stream, line)) {
            const std::string trimmed = trim(line);
            if (trimmed.empty()) {
                continue;
            }
            const Json parsed = Json::parse(trimmed);
            if (!parsed.is_object()) {
                continue;
            }
            const std::string user = parsed.string_or("user", "");
            const std::string answer = parsed.string_or("answer", "");
            const std::string haystack = to_lower(user + " " + answer);
            double score = 0.0;
            for (const auto& token : unique) {
                if (contains(haystack, to_lower(token))) {
                    score += 1.0;
                }
            }
            if (score <= 0.0) {
                continue;
            }
            if (contains(to_lower(user + " " + answer), to_lower(trim(query)))) {
                score += 2.0;
            }
            Json entry = Json::object();
            entry.set("session", Json::string(session));
            entry.set("turn", Json::integer(parsed.integer_or("index", 0)));
            entry.set("user", Json::string(head(collapse_whitespace(user), 200)));
            entry.set("answer", Json::string(head(collapse_whitespace(answer), 300)));
            entry.set("current", Json::boolean(session == exclude_session));
            hits.push_back(Hit{score, std::move(entry)});
        }
    }
    std::sort(hits.begin(), hits.end(), [](const Hit& left, const Hit& right) {
        return left.score > right.score;
    });
    const size_t limit = top_k > 0 ? static_cast<size_t>(top_k) : 5;
    for (size_t index = 0; index < hits.size() && index < limit; ++index) {
        matches.push(hits[index].entry);
    }
    return matches;
}

}  // namespace

double evaluate_expression(const std::string& expression, bool* ok, std::string* error) {
    *ok = true;
    error->clear();
    ExpressionParser parser(expression, ok, error);
    const double value = parser.parse();
    if (!std::isfinite(value)) {
        *ok = false;
        *error = "result is not a finite number";
        return 0.0;
    }
    return value;
}

void ToolRegistry::add(ToolDefinition definition) {
    if (definition.spec.primary_argument.empty() &&
        definition.spec.parameters.is_object()) {
        const Json* required = definition.spec.parameters.find("required");
        if (required != nullptr && required->is_array() && !required->items().empty() &&
            required->items().front().is_string()) {
            definition.spec.primary_argument = required->items().front().as_string();
        } else if (const Json* properties = definition.spec.parameters.find("properties")) {
            if (properties->is_object() && !properties->members().empty()) {
                definition.spec.primary_argument = properties->members().front().first;
            }
        }
    }
    for (auto& existing : definitions_) {
        if (existing.spec.name == definition.spec.name) {
            existing = std::move(definition);
            return;
        }
    }
    definitions_.push_back(std::move(definition));
}

bool ToolRegistry::has(const std::string& name) const {
    return spec(name) != nullptr;
}

const ToolSpec* ToolRegistry::spec(const std::string& name) const {
    for (const auto& definition : definitions_) {
        if (definition.spec.name == name) {
            return &definition.spec;
        }
    }
    return nullptr;
}

std::vector<std::string> ToolRegistry::names() const {
    std::vector<std::string> out;
    out.reserve(definitions_.size());
    for (const auto& definition : definitions_) {
        out.push_back(definition.spec.name);
    }
    return out;
}

ToolResult ToolRegistry::call(const std::string& name, const Json& arguments,
                              ToolContext& context) const {
    std::string note;
    const std::string canonical = resolve(name, &note);
    for (const auto& definition : definitions_) {
        if (definition.spec.name != canonical) {
            continue;
        }
        if (!arguments.is_object()) {
            return failure(canonical, "arguments must be a JSON object");
        }
        const long long begin = now_ms();
        ToolResult result = definition.handler(arguments, context);
        result.duration_ms = static_cast<double>(now_ms() - begin);
        if (result.name.empty()) {
            result.name = canonical;
        } else {
            result.name = canonical;
        }
        if (!note.empty()) {
            result.summary = note + "; " + result.summary;
        }
        return result;
    }
    std::string available;
    for (const auto& definition : definitions_) {
        available += (available.empty() ? "" : ", ") + definition.spec.name;
    }
    return failure(name, "unknown tool, available: " + available);
}

std::string ToolRegistry::resolve(const std::string& name, std::string* note) const {
    const std::string cleaned = to_lower(trim(name));
    std::string compact;
    for (const char character : cleaned) {
        if (std::isalnum(static_cast<unsigned char>(character)) != 0) {
            compact.push_back(character);
        }
    }
    if (compact.empty()) {
        return std::string();
    }
    auto canonical_for = [&](const std::string& candidate) -> std::string {
        for (const auto& definition : definitions_) {
            std::string normalised;
            for (const char character : definition.spec.name) {
                if (std::isalnum(static_cast<unsigned char>(character)) != 0) {
                    normalised.push_back(
                        static_cast<char>(std::tolower(static_cast<unsigned char>(character))));
                }
            }
            if (normalised == candidate) {
                return definition.spec.name;
            }
        }
        return std::string();
    };
    const std::string exact = canonical_for(compact);
    if (!exact.empty()) {
        return exact;
    }
    static const std::pair<const char*, const char*> kAliases[] = {
        {"gettime", "now"},        {"currenttime", "now"},
        {"time", "now"},           {"date", "now"},
        {"clock", "now"},          {"getcurrenttime", "now"},
        {"calc", "calculator"},    {"compute", "calculator"},
        {"calculate", "calculator"}, {"math", "calculator"},
        {"evaluate", "calculator"}, {"converter", "unit_convert"},
        {"convert", "unit_convert"}, {"unitconversion", "unit_convert"},
        {"rememberfact", "remember"}, {"store", "remember"},
        {"save", "remember"},      {"searchmemory", "recall"},
        {"retrievememory", "recall"}, {"memorysearch", "recall"},
        {"memory", "recall"},      {"deletememory", "forget"},
        {"searchhistory", "search_history"}, {"history", "search_history"},
        {"conversationsearch", "search_history"}, {"image", "image_info"},
        {"imageinfo", "image_info"}, {"describeimage", "image_info"},
        {"vision", "image_info"},  {"device", "device_info"},
        {"deviceinfo", "device_info"}, {"system", "device_info"},
        {"status", "device_info"}, {"plan", "make_plan"},
        {"createplan", "make_plan"}, {"updateplan", "update_plan"},
        {"wordcount", "text_stats"}, {"textstats", "text_stats"},
        {"stats", "text_stats"},
    };
    for (const auto& alias : kAliases) {
        if (compact == alias.first) {
            const std::string target = canonical_for(to_lower(alias.second));
            if (!target.empty()) {
                if (note != nullptr) {
                    *note = "matched '" + name + "' to the " + target + " tool";
                }
                return target;
            }
        }
    }
    // Last resort: one edit away from a registered name (missing letter, typo).
    std::string best;
    size_t best_distance = 3;
    for (const auto& definition : definitions_) {
        std::string normalised;
        for (const char character : definition.spec.name) {
            if (std::isalnum(static_cast<unsigned char>(character)) != 0) {
                normalised.push_back(
                    static_cast<char>(std::tolower(static_cast<unsigned char>(character))));
            }
        }
        const size_t distance = edit_distance(compact, normalised);
        if (distance < best_distance) {
            best_distance = distance;
            best = definition.spec.name;
        }
    }
    // Only accept a single-character slip on a reasonably long name: "nope"
    // must not silently become "now".
    if (!best.empty() && best_distance <= 1 && compact.size() >= 4 && best.size() >= 4) {
        if (note != nullptr) {
            *note = "matched '" + name + "' to the " + best + " tool";
        }
        return best;
    }
    return std::string();
}

std::string ToolRegistry::render_for_prompt() const {
    std::string out;
    for (const auto& definition : definitions_) {
        out += "- " + definition.spec.name + ": " + definition.spec.description;
        const std::string parameters = definition.spec.parameters.dump();
        if (parameters != "{}") {
            out += "\n  arguments: " + parameters;
        }
        out += "\n";
    }
    return out;
}

Json ToolRegistry::to_json() const {
    Json array = Json::array();
    for (const auto& definition : definitions_) {
        Json entry = Json::object();
        entry.set("name", Json::string(definition.spec.name));
        entry.set("description", Json::string(definition.spec.description));
        entry.set("parameters", definition.spec.parameters);
        array.push(std::move(entry));
    }
    return array;
}

ToolRegistry build_default_tools() {
    ToolRegistry registry;

    registry.add(ToolDefinition{
        ToolSpec{"calculator",
                 "Evaluate a math expression.",
                 Json::parse("{\"type\":\"object\",\"properties\":{\"expression\":"
                             "{\"type\":\"string\",\"description\":\"expression\"}},"
                             "\"required\":[\"expression\"]}")},
        [](const Json& arguments, ToolContext&) -> ToolResult {
            const Json expression = required_string(arguments, "expression");
            if (!expression.is_string() || expression.as_string().empty()) {
                return failure("calculator", "missing 'expression'");
            }
            bool ok = false;
            std::string error;
            const double value =
                evaluate_expression(expression.as_string(), &ok, &error);
            if (!ok) {
                return failure("calculator", error);
            }
            Json out = Json::object();
            out.set("expression", Json::string(expression.as_string()));
            out.set("value", Json::number(value));
            out.set("formatted", Json::string(format_number(value)));
            return success("calculator", out,
                           expression.as_string() + " = " + format_number(value));
        }});

    registry.add(ToolDefinition{
        ToolSpec{"now",
                 "Device date and time.",
                 Json::parse("{\"type\":\"object\",\"properties\":{\"field\":{\"type\":\"string\","
                             "\"enum\":[\"datetime\",\"date\",\"time\",\"weekday\",\"epoch\"]}},"
                             "\"required\":[]}")},
        [](const Json& arguments, ToolContext&) -> ToolResult {
            const std::string field = arguments.string_or("field", "datetime");
            const long long timestamp = now_ms();
            Json out = Json::object();
            out.set("epoch_ms", Json::integer(timestamp));
            out.set("date", Json::string(local_time_string(timestamp, "%Y-%m-%d")));
            out.set("time", Json::string(local_time_string(timestamp, "%H:%M:%S")));
            out.set("datetime", Json::string(local_time_string(timestamp, "%Y-%m-%d %H:%M:%S")));
            out.set("weekday", Json::string(local_time_string(timestamp, "%A")));
            out.set("utc", Json::string(iso8601(timestamp)));
            const std::string summary =
                field == "date"   ? local_time_string(timestamp, "%Y-%m-%d")
                : field == "time"       ? local_time_string(timestamp, "%H:%M:%S")
                : field == "weekday"    ? local_time_string(timestamp, "%A")
                : field == "epoch"      ? std::to_string(timestamp)
                                        : local_time_string(timestamp, "%Y-%m-%d %H:%M:%S");
            return success("now", out, "now = " + summary + " (local)");
        }});

    registry.add(ToolDefinition{
        ToolSpec{"unit_convert",
                 "Convert a value between units.",
                 Json::parse("{\"type\":\"object\",\"properties\":{\"value\":{\"type\":\"number\"},"
                             "\"from\":{\"type\":\"string\"},\"to\":{\"type\":\"string\"}},"
                             "\"required\":[\"value\",\"from\",\"to\"]}")},
        [](const Json& arguments, ToolContext&) -> ToolResult {
            if (!arguments.has("value") || !arguments.has("from") || !arguments.has("to")) {
                return failure("unit_convert", "value, from and to are required");
            }
            const double value = arguments.number_or("value", std::nan(""));
            if (!std::isfinite(value)) {
                return failure("unit_convert", "'value' must be a number");
            }
            const UnitDefinition* from = find_unit(arguments.string_or("from", ""));
            const UnitDefinition* to = find_unit(arguments.string_or("to", ""));
            if (from == nullptr) {
                return failure("unit_convert", "unknown unit '" + arguments.string_or("from", "") + "'");
            }
            if (to == nullptr) {
                return failure("unit_convert", "unknown unit '" + arguments.string_or("to", "") + "'");
            }
            if (std::string(from->category) != to->category) {
                return failure("unit_convert",
                               std::string("cannot convert ") + from->category + " to " + to->category);
            }
            const double base = value * from->factor + from->offset;
            const double converted = (base - to->offset) / to->factor;
            Json out = Json::object();
            out.set("value", Json::number(value));
            out.set("from", Json::string(from->name));
            out.set("to", Json::string(to->name));
            out.set("result", Json::number(converted));
            out.set("formatted", Json::string(format_number(converted)));
            out.set("category", Json::string(from->category));
            return success("unit_convert", out,
                           format_number(value) + " " + from->name + " = " +
                               format_number(converted) + " " + to->name);
        }});

    registry.add(ToolDefinition{
        ToolSpec{"text_stats",
                 "Count characters and words of a text.",
                 Json::parse("{\"type\":\"object\",\"properties\":{\"text\":{\"type\":\"string\"}},"
                             "\"required\":[\"text\"]}")},
        [](const Json& arguments, ToolContext&) -> ToolResult {
            const Json text = required_string(arguments, "text");
            if (!text.is_string()) {
                return failure("text_stats", "missing 'text'");
            }
            const std::string value = text.as_string();
            int words = 0;
            bool in_word = false;
            size_t cjk = 0;
            for (size_t index = 0; index < value.size();) {
                const unsigned char byte = static_cast<unsigned char>(value[index]);
                if (byte < 0x80) {
                    const bool separator = byte == ' ' || byte == '\t' || byte == '\n' ||
                                           byte == '\r';
                    if (!separator && !in_word) {
                        ++words;
                    }
                    in_word = !separator;
                    ++index;
                    continue;
                }
                in_word = false;
                size_t length = 1;
                if ((byte & 0xe0) == 0xc0) {
                    length = 2;
                } else if ((byte & 0xf0) == 0xe0) {
                    length = 3;
                } else if ((byte & 0xf8) == 0xf0) {
                    length = 4;
                }
                if (length >= 3) {
                    ++cjk;   // CJK ideographs live in the 3 and 4 byte planes
                }
                index += length;
            }
            Json out = Json::object();
            out.set("characters", Json::integer(static_cast<long long>(utf8_length(value))));
            out.set("bytes", Json::integer(static_cast<long long>(value.size())));
            out.set("words", Json::integer(words));
            out.set("cjk_characters", Json::integer(static_cast<long long>(cjk)));
            out.set("lines", Json::integer(
                                 static_cast<long long>(std::count(value.begin(), value.end(), '\n')) + 1));
            out.set("sha256", Json::string(sha256_hex(value)));
            return success("text_stats", out,
                           "characters=" + std::to_string(utf8_length(value)) + " words=" +
                               std::to_string(words) + " lines=" +
                               std::to_string(std::count(value.begin(), value.end(), '\n') + 1));
        }});

    registry.add(ToolDefinition{
        ToolSpec{"remember",
                 "Store a durable fact, preference or note.",
                 Json::parse("{\"type\":\"object\",\"properties\":{"
                             "\"text\":{\"type\":\"string\"},"
                             "\"kind\":{\"type\":\"string\",\"enum\":[\"fact\",\"preference\",\"note\"]},"
                             "\"tags\":{\"type\":\"array\",\"items\":{\"type\":\"string\"}},"
                             "\"importance\":{\"type\":\"number\"},"
                             "\"pinned\":{\"type\":\"boolean\"}},"
                             "\"required\":[\"text\"]}")},
        [](const Json& arguments, ToolContext& context) -> ToolResult {
            if (context.memory == nullptr) {
                return failure("remember", "memory store unavailable");
            }
            const Json text = required_string(arguments, "text");
            if (!text.is_string() || text.as_string().empty()) {
                return failure("remember", "missing 'text'");
            }
            MemoryRecord record;
            record.text = text.as_string();
            record.kind = arguments.string_or("kind", "note");
            record.importance = arguments.number_or("importance", 0.7);
            record.pinned = arguments.bool_or("pinned", false);
            record.session_id = context.session_id;
            record.turn_index = context.turn_index;
            if (const Json* tags = arguments.find("tags")) {
                for (const auto& tag : tags->items()) {
                    if (tag.is_string()) {
                        record.tags.push_back(tag.as_string());
                    }
                }
            }
            bool merged = false;
            const std::string id = context.memory->add(record, &merged);
            if (id.empty()) {
                return failure("remember", "could not store the memory");
            }
            if (context.written_memories != nullptr && !merged) {
                context.written_memories->push_back(id);
            }
            if (context.last_memory_text != nullptr) {
                *context.last_memory_text = record.text;
            }
            Json out = Json::object();
            out.set("id", Json::string(id));
            out.set("merged", Json::boolean(merged));
            out.set("kind", Json::string(record.kind));
            return success("remember", out,
                           (merged ? "updated memory " : "stored memory ") + id);
        }});

    registry.add(ToolDefinition{
        ToolSpec{"recall",
                 "Search long term memory.",
                 Json::parse("{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\"},"
                             "\"k\":{\"type\":\"integer\"}},\"required\":[\"query\"]}")},
        [](const Json& arguments, ToolContext& context) -> ToolResult {
            if (context.memory == nullptr) {
                return failure("recall", "memory store unavailable");
            }
            const std::string query = arguments.string_or("query", "");
            const int top_k = static_cast<int>(arguments.integer_or(
                "k", context.config != nullptr ? context.config->recall_top_k : 4));
            const auto hits = context.memory->search(query, top_k);
            Json items = Json::array();
            for (const auto& hit : hits) {
                Json entry = Json::object();
                entry.set("id", Json::string(hit.record.id));
                entry.set("kind", Json::string(hit.record.kind));
                entry.set("text", Json::string(hit.record.text));
                entry.set("score", Json::number(hit.score));
                items.push(std::move(entry));
            }
            std::vector<std::string> ids;
            for (const auto& hit : hits) {
                ids.push_back(hit.record.id);
            }
            context.memory->mark_accessed(ids);
            Json out = Json::object();
            out.set("query", Json::string(query));
            out.set("count", Json::integer(static_cast<long long>(hits.size())));
            out.set("items", std::move(items));
            std::string summary = "no memory matched";
            if (!hits.empty()) {
                summary = std::to_string(hits.size()) + " memory hit(s): ";
                for (size_t index = 0; index < hits.size() && index < 3; ++index) {
                    summary += (index == 0 ? "" : " | ") + head(hits[index].record.text, 80);
                }
            }
            return success("recall", out, summary);
        }});

    registry.add(ToolDefinition{
        ToolSpec{"forget",
                 "Delete a memory.",
                 Json::parse("{\"type\":\"object\",\"properties\":{\"id\":{\"type\":\"string\"},"
                             "\"query\":{\"type\":\"string\"}},\"required\":[]}")},
        [](const Json& arguments, ToolContext& context) -> ToolResult {
            if (context.memory == nullptr) {
                return failure("forget", "memory store unavailable");
            }
            const std::string id = arguments.string_or("id", "");
            if (!id.empty()) {
                const bool removed = context.memory->remove(id);
                return removed ? success("forget", Json::object(), "deleted memory " + id)
                               : failure("forget", "no memory with id " + id);
            }
            const std::string query = arguments.string_or("query", "");
            if (query.empty()) {
                return failure("forget", "'id' or 'query' is required");
            }
            const auto hits = context.memory->search(query, 1);
            if (hits.empty()) {
                return failure("forget", "nothing matched '" + query + "'");
            }
            const std::string victim = hits.front().record.id;
            context.memory->remove(victim);
            return success("forget", Json::object(),
                           "deleted memory " + victim + " (" + head(hits.front().record.text, 80) + ")");
        }});

    registry.add(ToolDefinition{
        ToolSpec{"search_history",
                 "Search earlier conversations.",
                 Json::parse("{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\"},"
                             "\"k\":{\"type\":\"integer\"}},\"required\":[\"query\"]}")},
        [](const Json& arguments, ToolContext& context) -> ToolResult {
            if (context.sessions == nullptr) {
                return failure("search_history", "session store unavailable");
            }
            const std::string query = arguments.string_or("query", "");
            if (query.empty()) {
                return failure("search_history", "missing 'query'");
            }
            const int top_k = static_cast<int>(arguments.integer_or("k", 4));
            Json matches = search_transcripts(context.sessions->directory_hint(), query, top_k,
                                              context.session_id);
            Json out = Json::object();
            out.set("query", Json::string(query));
            out.set("count", Json::integer(matches.size()));
            out.set("items", matches);
            return success("search_history", out,
                           std::to_string(static_cast<int>(matches.size())) +
                               " conversation hit(s) for '" + head(query, 48) + "'");
        }});

    registry.add(ToolDefinition{
        ToolSpec{"image_info",
                 "Describe the attached image.",
                 Json::parse("{\"type\":\"object\",\"properties\":{},\"required\":[]}")},
        [](const Json&, ToolContext& context) -> ToolResult {
            if (!context.has_image) {
                return failure("image_info", "no image attached to this turn");
            }
            Json out = Json::object();
            out.set("attached", Json::boolean(true));
            out.set("visual_tokens", Json::integer(context.image_tokens));
            out.set("note", Json::string(context.image_note));
            out.set("sha256", Json::string(context.image_sha256));
            out.set("file", Json::string(context.image_file));
            return success("image_info", out,
                           "image with " + std::to_string(context.image_tokens) +
                               " visual tokens, sha256 " +
                               head(context.image_sha256, 12));
        }});

    registry.add(ToolDefinition{
        ToolSpec{"device_info",
                 "Device, model and agent state.",
                 Json::parse("{\"type\":\"object\",\"properties\":{},\"required\":[]}")},
        [](const Json&, ToolContext& context) -> ToolResult {
            Json out = context.device.is_object() ? context.device : Json::object();
            if (context.memory != nullptr) {
                out.set("memory_records", Json::integer(static_cast<long long>(context.memory->size())));
            }
            if (context.sessions != nullptr) {
                out.set("sessions", Json::integer(static_cast<long long>(context.sessions->index().size())));
                out.set("session", Json::string(context.session_id));
                out.set("turn", Json::integer(context.turn_index));
            }
            if (context.config != nullptr) {
                out.set("context_budget_tokens",
                        Json::integer(context.config->context_budget_tokens));
                out.set("max_tool_calls", Json::integer(context.config->max_tool_calls));
            }
            return success("device_info", out, "device and agent state");
        }});

    registry.add(ToolDefinition{
        ToolSpec{"make_plan",
                 "Write the steps for a task with two or more tool calls.",
                 Json::parse("{\"type\":\"object\",\"properties\":{\"goal\":{\"type\":\"string\"},"
                             "\"steps\":{\"type\":\"array\",\"items\":{\"type\":\"string\"}}},"
                             "\"required\":[\"steps\"]}")},
        [](const Json& arguments, ToolContext& context) -> ToolResult {
            if (context.planner == nullptr) {
                return failure("make_plan", "planner unavailable");
            }
            std::vector<std::string> steps;
            const Json* list = arguments.find("steps");
            if (list == nullptr || !list->is_array()) {
                return failure("make_plan", "'steps' must be an array of strings");
            }
            for (const auto& entry : list->items()) {
                if (entry.is_string()) {
                    steps.push_back(entry.as_string());
                }
            }
            if (steps.empty()) {
                return failure("make_plan", "'steps' is empty");
            }
            const std::string goal = arguments.string_or("goal", "");
            if (!goal.empty()) {
                context.planner->begin_turn(goal);
            }
            const int count = context.planner->set_plan(steps);
            Json out = Json::object();
            out.set("steps", context.planner->to_json());
            out.set("count", Json::integer(count));
            return success("make_plan", out, "plan with " + std::to_string(count) + " step(s)");
        }});

    registry.add(ToolDefinition{
        ToolSpec{"update_plan",
                 "Mark a plan step done.",
                 Json::parse("{\"type\":\"object\",\"properties\":{\"index\":{\"type\":\"integer\"},"
                             "\"status\":{\"type\":\"string\",\"enum\":[\"active\",\"done\",\"failed\","
                             "\"skipped\"]},\"note\":{\"type\":\"string\"}},"
                             "\"required\":[\"index\",\"status\"]}")},
        [](const Json& arguments, ToolContext& context) -> ToolResult {
            if (context.planner == nullptr) {
                return failure("update_plan", "planner unavailable");
            }
            std::string error;
            if (!context.planner->update_from_json(arguments, &error)) {
                return failure("update_plan", error);
            }
            Json out = Json::object();
            out.set("steps", context.planner->to_json());
            return success("update_plan", out, "plan updated");
        }});

    return registry;
}

}  // namespace agent
