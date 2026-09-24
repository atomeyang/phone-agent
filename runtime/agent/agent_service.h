// File protocol shared by the device runner and the host tests.
//
// The app writes one request document, the runner answers with an append-only
// event log plus a response document.  Keeping this loop in the agent core (and
// not in the runner's main()) is what makes the contract testable without a
// phone: the host suite drives the same code with a stub model.
#ifndef GEMMA4_AGENT_SERVICE_H
#define GEMMA4_AGENT_SERVICE_H

#include "agent_json.h"
#include "agent_runtime.h"

#include <functional>
#include <string>

namespace agent {

class AgentService {
public:
    struct Options {
        std::string agent_dir;
        std::string root;                 // legacy app-* files live here
        std::string request_file;         // default: <agent_dir>/request.json
        std::string response_file;        // default: <agent_dir>/response.json
        std::string events_file;          // default: <agent_dir>/events.jsonl
        std::string cancel_file;          // default: <agent_dir>/cancel
        std::string ready_file;           // default: <agent_dir>/service-ready
        std::string legacy_request_file;  // empty disables the legacy path
        bool echo = false;                // also print events to stdout
        int poll_interval_ms = 40;
        int max_tokens = 512;
    };

    explicit AgentService(Options options);

    const Options& options() const { return options_; }
    const std::string& events_file() const { return options_.events_file; }

    // Creates the directories and truncates the event log.
    bool prepare(std::string* error);

    // Event sink for the runtime.
    void emit(const Json& event);

    // Handles one request and writes the response document.
    Json dispatch(AgentRuntime& runtime, const Json& request);

    // Handles a pending request if there is one (returns true when it did).
    // `legacy` is invoked for a legacy app-request file; the runner passes a
    // function that runs the single-shot vision path.
    bool handle_pending(AgentRuntime& runtime, const std::function<void()>& legacy);

    // The "hello" event the app uses to learn the capabilities.
    Json hello(AgentRuntime& runtime, double startup_ms, double llm_load_ms,
               const std::string& root) const;

    void write_ready(const std::string& contents) const;
    bool cancel_requested() const;
    void clear_cancel() const;
    std::string read_request(std::string* error) const;

private:
    Options options_;
};

}  // namespace agent

#endif  // GEMMA4_AGENT_SERVICE_H
