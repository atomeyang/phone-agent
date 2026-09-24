// Persistent memory store for the native agent.
//
// The store is a bounded, self-contained set of records on the device's
// filesystem: durable facts and preferences the user told the model, notes the
// model decided to keep, and short episode summaries of past turns.  Retrieval
// combines a BM25 lexical score (CJK aware tokenisation, because the assistant
// is bilingual) with recency decay, importance and access frequency, so that a
// "remember that ..." fact survives while stale small talk ages out.
#ifndef GEMMA4_AGENT_MEMORY_H
#define GEMMA4_AGENT_MEMORY_H

#include "agent_config.h"
#include "agent_json.h"
#include "agent_types.h"

#include <string>
#include <unordered_map>
#include <vector>

namespace agent {

class MemoryStore {
public:
    MemoryStore(std::string directory, const AgentConfig& config);

    bool load();
    bool save() const;

    // Inserts a record, merging near duplicates.  Returns the record id and
    // sets *merged when an existing record absorbed the new text.
    // Inserts a record.  Returns the id and reports what happened:
    // "new" (nothing similar), "merged" (near duplicate absorbed),
    // "updated" (a slot of an earlier record was replaced, e.g. a name).
    std::string add(MemoryRecord record, bool* merged = nullptr,
                    std::string* action = nullptr);
    bool remove(const std::string& id);
    int clear(const std::string& kind, bool keep_pinned);
    void mark_accessed(const std::vector<std::string>& ids);
    int prune();

    std::vector<ScoredMemory> search(const std::string& query, int top_k) const;
    std::vector<MemoryRecord> pinned() const;
    const MemoryRecord* find(const std::string& id) const;
    const std::vector<MemoryRecord>& records() const { return records_; }
    size_t size() const { return records_.size(); }

    static Json to_json(const MemoryRecord& record);
    static MemoryRecord from_json(const Json& value);
    static std::vector<std::string> tokenize(const std::string& text);

    // Identity/preference slots used to detect a correction: "我叫李雷" and
    // "我叫杨雷" share the key `name` with different values.
    struct Slot {
        std::string key;
        std::string value;
        bool found = false;
    };
    static Slot extract_slot(const std::string& text);
    static bool has_correction_marker(const std::string& text);

private:
    void rebuild_index();
    int record_token_count(const MemoryRecord& record) const;

    std::string directory_;
    std::string path_;
    AgentConfig config_;
    std::vector<MemoryRecord> records_;
    std::unordered_map<std::string, int> document_frequency_;
    std::unordered_map<std::string, int> record_lengths_;
    double average_length_ = 1.0;
    mutable long long sequence_ = 1;
};

}  // namespace agent

#endif  // GEMMA4_AGENT_MEMORY_H
