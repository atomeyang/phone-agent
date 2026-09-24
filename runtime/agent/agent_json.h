// Minimal dependency-free JSON value used by the on-device agent.
//
// The agent has to parse tool calls produced by the language model and to
// serialise its event log, so it needs a JSON implementation that is small,
// allocation-cheap and available in the runner, on the Android device and in
// the host test binaries.  Objects keep insertion order, which makes the
// serialised prompts and transcripts stable and diffable.
#ifndef GEMMA4_AGENT_JSON_H
#define GEMMA4_AGENT_JSON_H

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace agent {

class Json {
public:
    enum class Type { Null, Bool, Number, String, Array, Object };

    Json() = default;

    static Json object();
    static Json array();
    static Json string(std::string value);
    static Json number(double value);
    static Json integer(long long value);
    static Json boolean(bool value);
    static Json null();

    Type type() const { return type_; }
    bool is_null() const { return type_ == Type::Null; }
    bool is_bool() const { return type_ == Type::Bool; }
    bool is_number() const { return type_ == Type::Number; }
    bool is_string() const { return type_ == Type::String; }
    bool is_array() const { return type_ == Type::Array; }
    bool is_object() const { return type_ == Type::Object; }

    bool as_bool(bool fallback = false) const;
    double as_number(double fallback = 0.0) const;
    long long as_integer(long long fallback = 0) const;
    const std::string& as_string() const;

    // Object access.  operator[] inserts a null member when missing, which is
    // what the event writers want; find()/get_*() never insert.
    bool has(const std::string& key) const;
    const Json* find(const std::string& key) const;
    Json& operator[](const std::string& key);
    void set(const std::string& key, Json value);
    bool erase(const std::string& key);

    std::string string_or(const std::string& key, const std::string& fallback) const;
    double number_or(const std::string& key, double fallback) const;
    long long integer_or(const std::string& key, long long fallback) const;
    bool bool_or(const std::string& key, bool fallback) const;

    // Array access.
    void push(Json value);
    const Json* at(size_t index) const;
    const std::vector<Json>& items() const { return array_; }
    const std::vector<std::pair<std::string, Json>>& members() const { return object_; }

    size_t size() const;
    bool empty() const { return size() == 0; }

    std::string dump() const;
    std::string dump(int indent) const;

    static Json parse(const std::string& text, std::string* error = nullptr);
    static std::string escape(const std::string& value);

private:
    Type type_ = Type::Null;
    bool bool_ = false;
    double number_ = 0.0;
    std::string string_;
    std::vector<Json> array_;
    std::vector<std::pair<std::string, Json>> object_;
};

}  // namespace agent

#endif  // GEMMA4_AGENT_JSON_H
