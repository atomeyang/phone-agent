// Conversation history: sessions, transcripts and the context budget.
//
// Every session lives in its own directory as an append-only transcript plus a
// small meta file, and the store owns the policy that decides what the model
// actually sees: newest turns verbatim, older turns folded into a rolling
// summary once the token budget is exceeded.
#ifndef GEMMA4_AGENT_SESSION_H
#define GEMMA4_AGENT_SESSION_H

#include "agent_config.h"
#include "agent_json.h"
#include "agent_types.h"

#include <functional>
#include <string>
#include <vector>

namespace agent {

struct SessionMeta {
    std::string id;
    std::string title;
    long long created_ms = 0;
    long long updated_ms = 0;
    int turns = 0;
    int compactions = 0;
    std::string summary;      // rolling summary of the turns below
    int summary_upto = 0;     // last turn index covered by summary
    // Notetaker digest: bounded bullets about what the session established (what
    // an image showed, what was stored, which tools ran).  Persisted with the
    // session so it survives restarts.
    std::string notes;
};

class SessionStore {
public:
    // Exact token counting is delegated to the runner (MNN tokenizer); the host
    // tests inject a word-based approximation.
    using TokenCounter = std::function<int(const ChatMessages&)>;

    SessionStore(std::string directory, const AgentConfig& config, TokenCounter counter);

    bool load_index();
    bool create(const std::string& title_hint, const std::string& requested_id = std::string());
    bool open(const std::string& id);
    bool remove(const std::string& id);
    bool rename(const std::string& id, const std::string& title);
    bool persist_meta();
    bool append_turn(const SessionTurn& turn);
    // Rewrites turns.jsonl from the in-memory turns (used when an image had to
    // be dropped from the retained context and the transcript must match).
    bool rewrite_turns();

    const SessionMeta& meta() const { return meta_; }
    SessionMeta& mutable_meta() { return meta_; }
    const std::vector<SessionTurn>& turns() const { return turns_; }
    std::vector<SessionTurn>& mutable_turns() { return turns_; }
    const std::vector<SessionMeta>& index() const { return index_; }
    bool is_open() const { return !meta_.id.empty(); }
    const std::string& directory_hint() const { return directory_; }

    // Tokens the full transcript would occupy if it were replayed verbatim.
    int transcript_tokens(int reserve_tokens) const;
    bool needs_compaction(int budget_tokens, int reserve_tokens) const;
    // Turns that have to be summarised to get back under the budget.
    std::vector<SessionTurn> compaction_candidates() const;
    bool apply_compaction(const std::string& summary, int upto_turn);

    // Messages for one turn, including its observations and tool calls.
    ChatMessages render_turn(const SessionTurn& turn) const;

    static Json to_json(const SessionTurn& turn);
    static SessionTurn turn_from_json(const Json& value);
    static Json meta_to_json(const SessionMeta& meta);
    static SessionMeta meta_from_json(const Json& value);

private:
    std::string session_dir(const std::string& id) const;
    bool read_turns(const std::string& id, std::vector<SessionTurn>* turns) const;
    bool write_index() const;

    std::string directory_;
    AgentConfig config_;
    TokenCounter counter_;
    SessionMeta meta_;
    std::vector<SessionTurn> turns_;
    std::vector<SessionMeta> index_;
    int id_sequence_ = 1;
};

}  // namespace agent

#endif  // GEMMA4_AGENT_SESSION_H
