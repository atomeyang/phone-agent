#include "agent_json.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>

namespace agent {
namespace {

void append_utf8(std::string& out, unsigned int code_point) {
    if (code_point <= 0x7f) {
        out.push_back(static_cast<char>(code_point));
    } else if (code_point <= 0x7ff) {
        out.push_back(static_cast<char>(0xc0 | (code_point >> 6)));
        out.push_back(static_cast<char>(0x80 | (code_point & 0x3f)));
    } else if (code_point <= 0xffff) {
        out.push_back(static_cast<char>(0xe0 | (code_point >> 12)));
        out.push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (code_point & 0x3f)));
    } else {
        out.push_back(static_cast<char>(0xf0 | (code_point >> 18)));
        out.push_back(static_cast<char>(0x80 | ((code_point >> 12) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3f)));
        out.push_back(static_cast<char>(0x80 | (code_point & 0x3f)));
    }
}

int hex_value(char character) {
    if (character >= '0' && character <= '9') {
        return character - '0';
    }
    if (character >= 'a' && character <= 'f') {
        return character - 'a' + 10;
    }
    if (character >= 'A' && character <= 'F') {
        return character - 'A' + 10;
    }
    return -1;
}

class Parser {
public:
    Parser(const std::string& text, std::string* error) : text_(text), error_(error) {}

    bool parse(Json* out) {
        skip_space();
        if (!parse_value(out)) {
            return false;
        }
        skip_space();
        if (position_ != text_.size()) {
            return fail("trailing characters after JSON value");
        }
        return true;
    }

private:
    bool fail(const std::string& message) {
        if (error_ != nullptr && error_->empty()) {
            std::ostringstream stream;
            stream << message << " at offset " << position_;
            *error_ = stream.str();
        }
        return false;
    }

    void skip_space() {
        while (position_ < text_.size()) {
            const char character = text_[position_];
            if (character == ' ' || character == '\t' || character == '\n' ||
                character == '\r') {
                ++position_;
            } else {
                break;
            }
        }
    }

    bool literal(const char* word) {
        const size_t length = std::strlen(word);
        if (text_.compare(position_, length, word) != 0) {
            return false;
        }
        position_ += length;
        return true;
    }

    bool parse_value(Json* out) {
        if (position_ >= text_.size()) {
            return fail("unexpected end of input");
        }
        const char character = text_[position_];
        switch (character) {
            case '{': return parse_object(out);
            case '[': return parse_array(out);
            case '"': {
                std::string value;
                if (!parse_string(&value)) {
                    return false;
                }
                *out = Json::string(std::move(value));
                return true;
            }
            case 't':
                if (!literal("true")) {
                    return fail("invalid literal");
                }
                *out = Json::boolean(true);
                return true;
            case 'f':
                if (!literal("false")) {
                    return fail("invalid literal");
                }
                *out = Json::boolean(false);
                return true;
            case 'n':
                if (!literal("null")) {
                    return fail("invalid literal");
                }
                *out = Json::null();
                return true;
            default: return parse_number(out);
        }
    }

    bool parse_object(Json* out) {
        ++position_;  // '{'
        Json result = Json::object();
        skip_space();
        if (position_ < text_.size() && text_[position_] == '}') {
            ++position_;
            *out = std::move(result);
            return true;
        }
        while (true) {
            skip_space();
            std::string key;
            if (!parse_string(&key)) {
                return false;
            }
            skip_space();
            if (position_ >= text_.size() || text_[position_] != ':') {
                return fail("expected ':'");
            }
            ++position_;
            skip_space();
            Json value;
            if (!parse_value(&value)) {
                return false;
            }
            result.set(key, std::move(value));
            skip_space();
            if (position_ >= text_.size()) {
                return fail("unterminated object");
            }
            if (text_[position_] == ',') {
                ++position_;
                continue;
            }
            if (text_[position_] == '}') {
                ++position_;
                *out = std::move(result);
                return true;
            }
            return fail("expected ',' or '}'");
        }
    }

    bool parse_array(Json* out) {
        ++position_;  // '['
        Json result = Json::array();
        skip_space();
        if (position_ < text_.size() && text_[position_] == ']') {
            ++position_;
            *out = std::move(result);
            return true;
        }
        while (true) {
            skip_space();
            Json value;
            if (!parse_value(&value)) {
                return false;
            }
            result.push(std::move(value));
            skip_space();
            if (position_ >= text_.size()) {
                return fail("unterminated array");
            }
            if (text_[position_] == ',') {
                ++position_;
                continue;
            }
            if (text_[position_] == ']') {
                ++position_;
                *out = std::move(result);
                return true;
            }
            return fail("expected ',' or ']'");
        }
    }

    bool parse_string(std::string* out) {
        if (position_ >= text_.size() || text_[position_] != '"') {
            return fail("expected string");
        }
        ++position_;
        std::string value;
        while (true) {
            if (position_ >= text_.size()) {
                return fail("unterminated string");
            }
            const char character = text_[position_++];
            if (character == '"') {
                *out = std::move(value);
                return true;
            }
            if (character != '\\') {
                value.push_back(character);
                continue;
            }
            if (position_ >= text_.size()) {
                return fail("unterminated escape");
            }
            const char escape = text_[position_++];
            switch (escape) {
                case '"': value.push_back('"'); break;
                case '\\': value.push_back('\\'); break;
                case '/': value.push_back('/'); break;
                case 'b': value.push_back('\b'); break;
                case 'f': value.push_back('\f'); break;
                case 'n': value.push_back('\n'); break;
                case 'r': value.push_back('\r'); break;
                case 't': value.push_back('\t'); break;
                case 'u': {
                    unsigned int code_point = 0;
                    if (!parse_hex4(&code_point)) {
                        return false;
                    }
                    if (code_point >= 0xd800 && code_point <= 0xdbff &&
                        position_ + 1 < text_.size() && text_[position_] == '\\' &&
                        text_[position_ + 1] == 'u') {
                        const size_t saved = position_;
                        position_ += 2;
                        unsigned int low = 0;
                        if (!parse_hex4(&low)) {
                            return false;
                        }
                        if (low >= 0xdc00 && low <= 0xdfff) {
                            code_point = 0x10000 +
                                ((code_point - 0xd800) << 10) + (low - 0xdc00);
                        } else {
                            position_ = saved;
                        }
                    }
                    append_utf8(value, code_point);
                    break;
                }
                default: return fail("invalid escape");
            }
        }
    }

    bool parse_hex4(unsigned int* out) {
        if (position_ + 4 > text_.size()) {
            return fail("truncated \\u escape");
        }
        unsigned int value = 0;
        for (int index = 0; index < 4; ++index) {
            const int digit = hex_value(text_[position_ + index]);
            if (digit < 0) {
                return fail("invalid \\u escape");
            }
            value = (value << 4) | static_cast<unsigned int>(digit);
        }
        position_ += 4;
        *out = value;
        return true;
    }

    bool parse_number(Json* out) {
        const size_t begin = position_;
        if (position_ < text_.size() && (text_[position_] == '-' || text_[position_] == '+')) {
            ++position_;
        }
        bool digits = false;
        while (position_ < text_.size() && text_[position_] >= '0' && text_[position_] <= '9') {
            ++position_;
            digits = true;
        }
        if (position_ < text_.size() && text_[position_] == '.') {
            ++position_;
            while (position_ < text_.size() && text_[position_] >= '0' &&
                   text_[position_] <= '9') {
                ++position_;
                digits = true;
            }
        }
        if (!digits) {
            return fail("invalid number");
        }
        if (position_ < text_.size() &&
            (text_[position_] == 'e' || text_[position_] == 'E')) {
            ++position_;
            if (position_ < text_.size() &&
                (text_[position_] == '-' || text_[position_] == '+')) {
                ++position_;
            }
            while (position_ < text_.size() && text_[position_] >= '0' &&
                   text_[position_] <= '9') {
                ++position_;
            }
        }
        const std::string token = text_.substr(begin, position_ - begin);
        *out = Json::number(std::strtod(token.c_str(), nullptr));
        return true;
    }

    const std::string& text_;
    std::string* error_;
    size_t position_ = 0;
};

void dump_into(const Json& value, int indent, int depth, std::string* out) {
    const std::string pad(indent > 0 ? static_cast<size_t>(indent) * (depth + 1) : 0, ' ');
    const std::string pad_end(indent > 0 ? static_cast<size_t>(indent) * depth : 0, ' ');
    switch (value.type()) {
        case Json::Type::Null: out->append("null"); break;
        case Json::Type::Bool: out->append(value.as_bool() ? "true" : "false"); break;
        case Json::Type::Number: {
            const double number = value.as_number();
            char buffer[40];
            if (number == std::floor(number) && std::fabs(number) < 1e15) {
                std::snprintf(buffer, sizeof(buffer), "%lld",
                              static_cast<long long>(number));
            } else {
                std::snprintf(buffer, sizeof(buffer), "%.10g", number);
            }
            out->append(buffer);
            break;
        }
        case Json::Type::String: out->append(Json::escape(value.as_string())); break;
        case Json::Type::Array: {
            const auto& items = value.items();
            if (items.empty()) {
                out->append("[]");
                break;
            }
            out->push_back('[');
            for (size_t index = 0; index < items.size(); ++index) {
                if (index != 0) {
                    out->push_back(',');
                }
                if (indent > 0) {
                    out->push_back('\n');
                    out->append(pad);
                }
                dump_into(items[index], indent, depth + 1, out);
            }
            if (indent > 0) {
                out->push_back('\n');
                out->append(pad_end);
            }
            out->push_back(']');
            break;
        }
        case Json::Type::Object: {
            const auto& members = value.members();
            if (members.empty()) {
                out->append("{}");
                break;
            }
            out->push_back('{');
            for (size_t index = 0; index < members.size(); ++index) {
                if (index != 0) {
                    out->push_back(',');
                }
                if (indent > 0) {
                    out->push_back('\n');
                    out->append(pad);
                }
                out->append(Json::escape(members[index].first));
                out->push_back(':');
                if (indent > 0) {
                    out->push_back(' ');
                }
                dump_into(members[index].second, indent, depth + 1, out);
            }
            if (indent > 0) {
                out->push_back('\n');
                out->append(pad_end);
            }
            out->push_back('}');
            break;
        }
    }
}

}  // namespace

Json Json::object() {
    Json value;
    value.type_ = Type::Object;
    return value;
}

Json Json::array() {
    Json value;
    value.type_ = Type::Array;
    return value;
}

Json Json::string(std::string text) {
    Json value;
    value.type_ = Type::String;
    value.string_ = std::move(text);
    return value;
}

Json Json::number(double number) {
    Json value;
    value.type_ = Type::Number;
    value.number_ = number;
    return value;
}

Json Json::integer(long long number) {
    return Json::number(static_cast<double>(number));
}

Json Json::boolean(bool flag) {
    Json value;
    value.type_ = Type::Bool;
    value.bool_ = flag;
    return value;
}

Json Json::null() {
    return Json();
}

bool Json::as_bool(bool fallback) const {
    if (type_ == Type::Bool) {
        return bool_;
    }
    if (type_ == Type::Number) {
        return number_ != 0.0;
    }
    return fallback;
}

double Json::as_number(double fallback) const {
    if (type_ == Type::Number) {
        return number_;
    }
    if (type_ == Type::Bool) {
        return bool_ ? 1.0 : 0.0;
    }
    if (type_ == Type::String) {
        char* end = nullptr;
        const double parsed = std::strtod(string_.c_str(), &end);
        if (end != nullptr && end != string_.c_str() && *end == '\0') {
            return parsed;
        }
    }
    return fallback;
}

long long Json::as_integer(long long fallback) const {
    if (type_ == Type::Number) {
        return static_cast<long long>(number_);
    }
    if (type_ == Type::Bool) {
        return bool_ ? 1 : 0;
    }
    return fallback;
}

const std::string& Json::as_string() const {
    static const std::string empty;
    return type_ == Type::String ? string_ : empty;
}

bool Json::has(const std::string& key) const {
    return find(key) != nullptr;
}

const Json* Json::find(const std::string& key) const {
    for (const auto& member : object_) {
        if (member.first == key) {
            return &member.second;
        }
    }
    return nullptr;
}

Json& Json::operator[](const std::string& key) {
    for (auto& member : object_) {
        if (member.first == key) {
            return member.second;
        }
    }
    if (type_ != Type::Object) {
        type_ = Type::Object;
    }
    object_.emplace_back(key, Json());
    return object_.back().second;
}

void Json::set(const std::string& key, Json value) {
    for (auto& member : object_) {
        if (member.first == key) {
            member.second = std::move(value);
            return;
        }
    }
    if (type_ != Type::Object) {
        type_ = Type::Object;
    }
    object_.emplace_back(key, std::move(value));
}

bool Json::erase(const std::string& key) {
    for (size_t index = 0; index < object_.size(); ++index) {
        if (object_[index].first == key) {
            object_.erase(object_.begin() + static_cast<long>(index));
            return true;
        }
    }
    return false;
}

std::string Json::string_or(const std::string& key, const std::string& fallback) const {
    const Json* value = find(key);
    return value != nullptr && value->is_string() ? value->as_string() : fallback;
}

double Json::number_or(const std::string& key, double fallback) const {
    const Json* value = find(key);
    return value != nullptr ? value->as_number(fallback) : fallback;
}

long long Json::integer_or(const std::string& key, long long fallback) const {
    const Json* value = find(key);
    return value != nullptr ? value->as_integer(fallback) : fallback;
}

bool Json::bool_or(const std::string& key, bool fallback) const {
    const Json* value = find(key);
    return value != nullptr ? value->as_bool(fallback) : fallback;
}

void Json::push(Json value) {
    if (type_ != Type::Array) {
        type_ = Type::Array;
    }
    array_.push_back(std::move(value));
}

const Json* Json::at(size_t index) const {
    return index < array_.size() ? &array_[index] : nullptr;
}

size_t Json::size() const {
    if (type_ == Type::Array) {
        return array_.size();
    }
    if (type_ == Type::Object) {
        return object_.size();
    }
    if (type_ == Type::String) {
        return string_.size();
    }
    return 0;
}

std::string Json::dump() const {
    std::string out;
    dump_into(*this, 0, 0, &out);
    return out;
}

std::string Json::dump(int indent) const {
    std::string out;
    dump_into(*this, indent, 0, &out);
    return out;
}

Json Json::parse(const std::string& text, std::string* error) {
    if (error != nullptr) {
        error->clear();
    }
    Json value;
    Parser parser(text, error);
    if (!parser.parse(&value)) {
        if (error != nullptr && error->empty()) {
            *error = "invalid JSON";
        }
        return Json();
    }
    return value;
}

std::string Json::escape(const std::string& value) {
    std::string out;
    out.reserve(value.size() + 2);
    out.push_back('"');
    for (const char character : value) {
        switch (character) {
            case '"': out.append("\\\""); break;
            case '\\': out.append("\\\\"); break;
            case '\n': out.append("\\n"); break;
            case '\r': out.append("\\r"); break;
            case '\t': out.append("\\t"); break;
            case '\b': out.append("\\b"); break;
            case '\f': out.append("\\f"); break;
            default:
                if (static_cast<unsigned char>(character) < 0x20) {
                    char buffer[8];
                    std::snprintf(buffer, sizeof(buffer), "\\u%04x",
                                  static_cast<unsigned int>(
                                      static_cast<unsigned char>(character)));
                    out.append(buffer);
                } else {
                    out.push_back(character);
                }
                break;
        }
    }
    out.push_back('"');
    return out;
}

}  // namespace agent
