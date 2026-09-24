#include "agent_memory.h"

#include "agent_util.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <cmath>
#include <set>
#include <sstream>

namespace agent {
namespace {

constexpr double kDayMs = 86400000.0;

// UTF-8 -> code points.  Invalid bytes are skipped; the helpers here only feed
// retrieval, never the model, so being lenient is the right trade-off.
std::vector<uint32_t> decode_code_points(const std::string& text) {
    std::vector<uint32_t> points;
    points.reserve(text.size());
    size_t index = 0;
    while (index < text.size()) {
        const unsigned char byte = static_cast<unsigned char>(text[index]);
        uint32_t code_point = byte;
        size_t length = 1;
        if ((byte & 0x80) == 0x00) {
            length = 1;
            code_point = byte;
        } else if ((byte & 0xe0) == 0xc0 && index + 1 < text.size()) {
            length = 2;
            code_point = static_cast<uint32_t>(byte & 0x1f) << 6;
            code_point |= static_cast<unsigned char>(text[index + 1]) & 0x3f;
        } else if ((byte & 0xf0) == 0xe0 && index + 2 < text.size()) {
            length = 3;
            code_point = static_cast<uint32_t>(byte & 0x0f) << 12;
            code_point |= (static_cast<unsigned char>(text[index + 1]) & 0x3f) << 6;
            code_point |= static_cast<unsigned char>(text[index + 2]) & 0x3f;
        } else if ((byte & 0xf8) == 0xf0 && index + 3 < text.size()) {
            length = 4;
            code_point = static_cast<uint32_t>(byte & 0x07) << 18;
            code_point |= (static_cast<unsigned char>(text[index + 1]) & 0x3f) << 12;
            code_point |= (static_cast<unsigned char>(text[index + 2]) & 0x3f) << 6;
            code_point |= static_cast<unsigned char>(text[index + 3]) & 0x3f;
        }
        points.push_back(code_point);
        index += length;
    }
    return points;
}

std::string encode_code_point(uint32_t code_point) {
    std::string out;
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
    return out;
}

bool is_ascii_word(uint32_t code_point) {
    if (code_point > 0x7f) {
        return false;
    }
    const char character = static_cast<char>(code_point);
    return std::isalnum(static_cast<unsigned char>(character)) != 0 || character == '_' ||
           character == '-' || character == '+';
}

bool is_cjk(uint32_t code_point) {
    return (code_point >= 0x3040 && code_point <= 0x30ff) ||   // kana
           (code_point >= 0x3400 && code_point <= 0x4dbf) ||   // ext A
           (code_point >= 0x4e00 && code_point <= 0x9fff) ||   // unified
           (code_point >= 0xf900 && code_point <= 0xfaff) ||   // compat
           (code_point >= 0xac00 && code_point <= 0xd7af);     // hangul
}

double jaccard(const std::set<std::string>& left, const std::set<std::string>& right) {
    if (left.empty() || right.empty()) {
        return 0.0;
    }
    size_t intersection = 0;
    for (const auto& token : left) {
        if (right.count(token) != 0) {
            ++intersection;
        }
    }
    const size_t union_size = left.size() + right.size() - intersection;
    return union_size == 0 ? 0.0 : static_cast<double>(intersection) / union_size;
}

std::set<std::string> token_set(const std::string& text) {
    const auto tokens = MemoryStore::tokenize(text);
    return std::set<std::string>(tokens.begin(), tokens.end());
}

double kind_weight(const std::string& kind, const AgentConfig& config) {
    if (kind == "episode") {
        return config.memory_episode_weight;
    }
    if (kind == "fact" || kind == "preference") {
        return 1.0;
    }
    return 0.9;
}

}  // namespace

MemoryStore::MemoryStore(std::string directory, const AgentConfig& config)
    : directory_(std::move(directory)), config_(config) {
    path_ = path_join(directory_, "memories.jsonl");
    if (directory_.empty()) {
        directory_ = "agent-memory";
        path_ = path_join(directory_, "memories.jsonl");
    }
}

bool MemoryStore::load() {
    records_.clear();
    sequence_ = 1;
    if (!make_directories(directory_)) {
        return false;
    }
    std::string contents;
    if (!read_file(path_, &contents)) {
        rebuild_index();
        return true;
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
        MemoryRecord record = from_json(parsed);
        if (record.id.empty()) {
            record.id = "m" + std::to_string(sequence_);
        }
        long long numeric = 0;
        if (record.id.size() > 1 && record.id[0] == 'm') {
            try {
                numeric = std::stoll(record.id.substr(1));
            } catch (...) {
                numeric = 0;
            }
        }
        sequence_ = std::max(sequence_, numeric + 1);
        records_.push_back(std::move(record));
    }
    std::string state;
    if (read_text_file(path_join(directory_, "state.json"), &state)) {
        const Json parsed = Json::parse(state);
        if (parsed.is_object()) {
            sequence_ = std::max(sequence_, parsed.integer_or("sequence", sequence_));
        }
    }
    rebuild_index();
    return true;
}

bool MemoryStore::save() const {
    if (!make_directories(directory_)) {
        return false;
    }
    std::string contents;
    contents.reserve(records_.size() * 160);
    for (const auto& record : records_) {
        contents += to_json(record).dump();
        contents.push_back('\n');
    }
    Json state = Json::object();
    state.set("sequence", Json::integer(sequence_));
    state.set("updated_ms", Json::integer(now_ms()));
    state.set("records", Json::integer(static_cast<long long>(records_.size())));
    return write_file(path_, contents) &&
           write_file(path_join(directory_, "state.json"), state.dump(2));
}

void MemoryStore::rebuild_index() {
    document_frequency_.clear();
    record_lengths_.clear();
    long long total = 0;
    for (const auto& record : records_) {
        const auto tokens = tokenize(record.text + " " + join(record.tags, " "));
        std::set<std::string> unique(tokens.begin(), tokens.end());
        for (const auto& token : unique) {
            document_frequency_[token] += 1;
        }
        record_lengths_[record.id] = static_cast<int>(tokens.size());
        total += static_cast<long long>(tokens.size());
    }
    average_length_ = records_.empty()
        ? 1.0
        : std::max(1.0, static_cast<double>(total) / static_cast<double>(records_.size()));
}

int MemoryStore::record_token_count(const MemoryRecord& record) const {
    const auto found = record_lengths_.find(record.id);
    if (found != record_lengths_.end()) {
        return found->second;
    }
    return static_cast<int>(tokenize(record.text).size());
}

MemoryStore::Slot MemoryStore::extract_slot(const std::string& text) {
    // Ordered from the most specific pattern to the loosest one.
    static const std::pair<const char*, const char*> kSlots[] = {
        {"我的名字是", "name"}, {"我的名字叫", "name"}, {"我叫", "name"},
        {"叫我", "name"},       {"my name is", "name"}, {"call me", "name"},
        {"我住在", "city"},     {"i live in", "city"},  {"我的职业是", "job"},
        {"我是", "job"},        {"i work as", "job"},
        {"我喜欢", "preference"}, {"我偏好", "preference"}, {"我不喜欢", "preference"},
        {"我讨厌", "preference"}, {"i like", "preference"}, {"i prefer", "preference"},
        {"i hate", "preference"},
    };
    Slot slot;
    const std::string lowered = to_lower(text);
    for (const auto& pattern : kSlots) {
        size_t position = text.find(pattern.first);
        if (position == std::string::npos) {
            position = lowered.find(pattern.first);
            if (position == std::string::npos) {
                continue;
            }
        }
        std::string value = text.substr(position + std::strlen(pattern.first));
        // The value ends at the first delimiter.
        const size_t delimiter = value.find_first_of(",.;:!?，。；：！？\n\t");
        if (delimiter != std::string::npos) {
            value = value.substr(0, delimiter);
        }
        value = trim(collapse_whitespace(value));
        if (value.empty()) {
            continue;
        }
        slot.key = pattern.second;
        slot.value = head(value, 64);
        slot.found = true;
        return slot;
    }
    return slot;
}

bool MemoryStore::has_correction_marker(const std::string& text) {
    static const char* kMarkers[] = {"改成", "改为", "更正", "纠正", "改一下",
                                     "不是", "不对", "应该是", "重记", "我说的是",
                                     "instead", "actually", "correction"};
    const std::string lowered = to_lower(text);
    for (const char* marker : kMarkers) {
        if (contains(text, marker) || contains(lowered, marker)) {
            return true;
        }
    }
    return false;
}

std::string MemoryStore::add(MemoryRecord record, bool* merged, std::string* action) {
    if (merged != nullptr) {
        *merged = false;
    }
    if (action != nullptr) {
        *action = "new";
    }
    record.text = trim(collapse_whitespace(record.text));
    if (record.text.empty()) {
        return std::string();
    }
    if (record.text.size() > static_cast<size_t>(config_.max_memory_chars) * 3) {
        record.text = head(record.text, static_cast<size_t>(config_.max_memory_chars) * 3);
    }
    if (record.kind.empty()) {
        record.kind = "note";
    }
    const long long timestamp = now_ms();
    if (record.created_ms == 0) {
        record.created_ms = timestamp;
    }
    record.updated_ms = timestamp;
    if (record.last_access_ms == 0) {
        record.last_access_ms = timestamp;
    }
    record.importance = std::min(1.0, std::max(0.0, record.importance));

    const std::set<std::string> candidate = token_set(record.text);
    // 1. Slot conflict: the user replaced a value ("我叫李雷" -> "我叫杨雷").
    const Slot slot = extract_slot(record.text);
    if (slot.found) {
        for (auto& existing : records_) {
            if (existing.kind != record.kind) {
                continue;
            }
            const Slot other = extract_slot(existing.text);
            if (!other.found || other.key != slot.key) {
                continue;
            }
            if (other.value == slot.value) {
                // Same fact said differently ("我叫杨雷" / "再说一遍：我叫杨雷"):
                // absorb it instead of growing the store with a duplicate.
                if (record.text.size() > existing.text.size()) {
                    existing.text = record.text;
                }
                existing.importance = std::max(existing.importance, record.importance);
                existing.pinned = existing.pinned || record.pinned;
                existing.updated_ms = timestamp;
                existing.access_count += 1;
                if (merged != nullptr) {
                    *merged = true;
                }
                if (action != nullptr) {
                    *action = "merged";
                }
                rebuild_index();
                return existing.id;
            }
            existing.text = record.text;
            existing.importance = std::max(existing.importance, record.importance);
            existing.pinned = existing.pinned || record.pinned;
            existing.updated_ms = timestamp;
            existing.last_access_ms = timestamp;
            existing.access_count += 1;
            if (merged != nullptr) {
                *merged = true;
            }
            if (action != nullptr) {
                *action = "updated";
            }
            rebuild_index();
            return existing.id;
        }
    }
    // 2. An explicit correction ("改成…", "instead"): replace the closest memory
    //    of the same kind even when the wording differs a lot.
    if (record.correction || has_correction_marker(record.text)) {
        size_t best_index = records_.size();
        double best_similarity = 0.0;
        for (size_t index = 0; index < records_.size(); ++index) {
            if (records_[index].kind != record.kind) {
                continue;
            }
            const double similarity = jaccard(candidate, token_set(records_[index].text));
            if (similarity > best_similarity) {
                best_similarity = similarity;
                best_index = index;
            }
        }
        if (best_index < records_.size() && best_similarity >= 0.2) {
            MemoryRecord& existing = records_[best_index];
            existing.text = record.text;
            existing.importance = std::max(existing.importance, record.importance);
            existing.pinned = existing.pinned || record.pinned;
            existing.updated_ms = timestamp;
            existing.last_access_ms = timestamp;
            existing.access_count += 1;
            if (merged != nullptr) {
                *merged = true;
            }
            if (action != nullptr) {
                *action = "updated";
            }
            rebuild_index();
            return existing.id;
        }
    }
    // 3. Near duplicate: merge into the existing record.
    for (auto& existing : records_) {
        if (existing.kind != record.kind) {
            continue;
        }
        const double similarity = jaccard(candidate, token_set(existing.text));
        if (similarity < 0.72) {
            continue;
        }
        if (record.text.size() > existing.text.size()) {
            existing.text = record.text;
        }
        existing.importance = std::max(existing.importance, record.importance);
        existing.pinned = existing.pinned || record.pinned;
        for (const auto& tag : record.tags) {
            if (std::find(existing.tags.begin(), existing.tags.end(), tag) ==
                existing.tags.end()) {
                existing.tags.push_back(tag);
            }
        }
        existing.updated_ms = timestamp;
        existing.access_count += 1;
        if (merged != nullptr) {
            *merged = true;
        }
        rebuild_index();
        return existing.id;
    }

    record.id = "m" + std::to_string(sequence_++);
    records_.push_back(record);
    rebuild_index();
    prune();
    return record.id;
}

bool MemoryStore::remove(const std::string& id) {
    for (size_t index = 0; index < records_.size(); ++index) {
        if (records_[index].id == id) {
            records_.erase(records_.begin() + static_cast<long>(index));
            rebuild_index();
            return true;
        }
    }
    return false;
}

int MemoryStore::clear(const std::string& kind, bool keep_pinned) {
    int removed = 0;
    std::vector<MemoryRecord> kept;
    kept.reserve(records_.size());
    for (auto& record : records_) {
        const bool matches = kind.empty() || record.kind == kind;
        if (matches && !(keep_pinned && record.pinned)) {
            ++removed;
            continue;
        }
        kept.push_back(record);
    }
    records_ = std::move(kept);
    rebuild_index();
    return removed;
}

void MemoryStore::mark_accessed(const std::vector<std::string>& ids) {
    const long long timestamp = now_ms();
    for (auto& record : records_) {
        if (std::find(ids.begin(), ids.end(), record.id) == ids.end()) {
            continue;
        }
        record.last_access_ms = timestamp;
        record.access_count += 1;
    }
}

int MemoryStore::prune() {
    if (static_cast<int>(records_.size()) <= config_.memory_max_records) {
        return 0;
    }
    const long long timestamp = now_ms();
    struct Ranked {
        size_t index;
        double value;
        bool pinned;
    };
    std::vector<Ranked> ranked;
    ranked.reserve(records_.size());
    for (size_t index = 0; index < records_.size(); ++index) {
        const auto& record = records_[index];
        const double age_days = static_cast<double>(timestamp - record.updated_ms) / kDayMs;
        const double recency = std::pow(0.5, age_days / config_.memory_half_life_days);
        double value = record.importance * 2.0 + recency +
                       0.1 * std::min(record.access_count, 8);
        if (record.kind == "episode") {
            value *= 0.6;
        }
        ranked.push_back(Ranked{index, value, record.pinned});
    }
    std::sort(ranked.begin(), ranked.end(), [](const Ranked& left, const Ranked& right) {
        if (left.pinned != right.pinned) {
            return left.pinned;
        }
        return left.value > right.value;
    });
    const size_t keep = static_cast<size_t>(config_.memory_max_records);
    std::vector<bool> keep_flags(records_.size(), false);
    size_t kept = 0;
    for (const auto& entry : ranked) {
        if (kept >= keep && !entry.pinned) {
            continue;
        }
        keep_flags[entry.index] = true;
        ++kept;
    }
    std::vector<MemoryRecord> survivors;
    survivors.reserve(kept);
    for (size_t index = 0; index < records_.size(); ++index) {
        if (keep_flags[index]) {
            survivors.push_back(records_[index]);
        }
    }
    const int removed = static_cast<int>(records_.size() - survivors.size());
    records_ = std::move(survivors);
    rebuild_index();
    return removed;
}

std::vector<ScoredMemory> MemoryStore::search(const std::string& query, int top_k) const {
    std::vector<ScoredMemory> results;
    if (records_.empty()) {
        return results;
    }
    const long long timestamp = now_ms();
    const double count = static_cast<double>(records_.size());
    std::vector<std::string> query_tokens = tokenize(query);
    std::set<std::string> unique_query(query_tokens.begin(), query_tokens.end());
    const std::string lowered_query = to_lower(trim(query));
    constexpr double k1 = 1.2;
    constexpr double b = 0.6;

    for (const auto& record : records_) {
        double lexical = 0.0;
        if (!unique_query.empty()) {
            const auto tokens = tokenize(record.text + " " + join(record.tags, " "));
            std::unordered_map<std::string, int> frequencies;
            for (const auto& token : tokens) {
                frequencies[token] += 1;
            }
            const double length = std::max(1.0, static_cast<double>(tokens.size()));
            for (const auto& term : unique_query) {
                const auto found = frequencies.find(term);
                if (found == frequencies.end()) {
                    continue;
                }
                const double tf = found->second;
                const auto df = document_frequency_.find(term);
                const double documents = df == document_frequency_.end() ? 1.0 : df->second;
                const double idf = std::log(1.0 + (count - documents + 0.5) / (documents + 0.5));
                lexical += idf * (tf * (k1 + 1.0)) /
                           (tf + k1 * (1.0 - b + b * length / average_length_));
            }
            if (!lowered_query.empty() && contains(to_lower(record.text), lowered_query)) {
                lexical += 1.5;  // exact phrase bonus
            }
        }
        const double age_days = static_cast<double>(timestamp - record.updated_ms) / kDayMs;
        const double recency = std::pow(0.5, age_days / config_.memory_half_life_days);
        double score = lexical * kind_weight(record.kind, config_) *
                       (0.6 + 0.8 * record.importance) * (0.7 + 0.3 * recency);
        score *= 1.0 + 0.05 * std::min(record.access_count, 8);
        if (record.pinned) {
            score += 1.0 + 0.5 * record.importance;
        }
        if (unique_query.empty()) {
            score = record.importance + 0.5 * recency;
            if (record.pinned) {
                score += 1.0;
            }
        }
        if (score < config_.memory_min_score) {
            continue;
        }
        ScoredMemory scored;
        scored.record = record;
        scored.lexical = lexical;
        scored.recency = recency;
        scored.score = score;
        results.push_back(std::move(scored));
    }
    std::sort(results.begin(), results.end(),
              [](const ScoredMemory& left, const ScoredMemory& right) {
                  if (left.score != right.score) {
                      return left.score > right.score;
                  }
                  return left.record.updated_ms > right.record.updated_ms;
              });
    if (top_k > 0 && results.size() > static_cast<size_t>(top_k)) {
        results.resize(static_cast<size_t>(top_k));
    }
    return results;
}

std::vector<MemoryRecord> MemoryStore::pinned() const {
    std::vector<MemoryRecord> out;
    for (const auto& record : records_) {
        if (record.pinned) {
            out.push_back(record);
        }
    }
    return out;
}

const MemoryRecord* MemoryStore::find(const std::string& id) const {
    for (const auto& record : records_) {
        if (record.id == id) {
            return &record;
        }
    }
    return nullptr;
}

Json MemoryStore::to_json(const MemoryRecord& record) {
    Json value = Json::object();
    value.set("id", Json::string(record.id));
    value.set("text", Json::string(record.text));
    value.set("kind", Json::string(record.kind));
    Json tags = Json::array();
    for (const auto& tag : record.tags) {
        tags.push(Json::string(tag));
    }
    value.set("tags", std::move(tags));
    value.set("importance", Json::number(record.importance));
    value.set("pinned", Json::boolean(record.pinned));
    value.set("created_ms", Json::integer(record.created_ms));
    value.set("updated_ms", Json::integer(record.updated_ms));
    value.set("last_access_ms", Json::integer(record.last_access_ms));
    value.set("access_count", Json::integer(record.access_count));
    value.set("session", Json::string(record.session_id));
    value.set("turn", Json::integer(record.turn_index));
    return value;
}

MemoryRecord MemoryStore::from_json(const Json& value) {
    MemoryRecord record;
    record.id = value.string_or("id", "");
    record.text = value.string_or("text", "");
    record.kind = value.string_or("kind", "note");
    if (const Json* tags = value.find("tags")) {
        for (const auto& tag : tags->items()) {
            if (tag.is_string()) {
                record.tags.push_back(tag.as_string());
            }
        }
    }
    record.importance = value.number_or("importance", 0.5);
    record.pinned = value.bool_or("pinned", false);
    record.created_ms = value.integer_or("created_ms", 0);
    record.updated_ms = value.integer_or("updated_ms", 0);
    record.last_access_ms = value.integer_or("last_access_ms", 0);
    record.access_count = static_cast<int>(value.integer_or("access_count", 0));
    record.session_id = value.string_or("session", "");
    record.turn_index = static_cast<int>(value.integer_or("turn", 0));
    return record;
}

std::vector<std::string> MemoryStore::tokenize(const std::string& text) {
    std::vector<std::string> tokens;
    const auto points = decode_code_points(text);
    std::string ascii;
    auto flush = [&]() {
        if (!ascii.empty()) {
            tokens.push_back(to_lower(ascii));
            ascii.clear();
        }
    };
    for (size_t index = 0; index < points.size(); ++index) {
        const uint32_t code_point = points[index];
        if (is_ascii_word(code_point)) {
            ascii.push_back(static_cast<char>(code_point));
            continue;
        }
        flush();
        if (!is_cjk(code_point)) {
            continue;
        }
        tokens.push_back(encode_code_point(code_point));
        if (index + 1 < points.size() && is_cjk(points[index + 1])) {
            // CJK bigrams: "手机" and "机壳" both match a "手机壳" record.
            tokens.push_back(encode_code_point(code_point) +
                             encode_code_point(points[index + 1]));
        }
    }
    flush();
    return tokens;
}

}  // namespace agent
