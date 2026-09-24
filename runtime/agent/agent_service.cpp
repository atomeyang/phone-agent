#include "agent_service.h"

#include "agent_util.h"

#include <iostream>
#include <cstdio>

namespace agent {
namespace {

std::string default_path(const std::string& directory, const std::string& name) {
    return path_join(directory, name);
}

}  // namespace

AgentService::AgentService(Options options) : options_(std::move(options)) {
    if (options_.agent_dir.empty()) {
        options_.agent_dir = "agent";
    }
    if (options_.root.empty()) {
        options_.root = parent_path(options_.agent_dir);
    }
    if (options_.request_file.empty()) {
        options_.request_file = default_path(options_.agent_dir, "request.json");
    }
    if (options_.response_file.empty()) {
        options_.response_file = default_path(options_.agent_dir, "response.json");
    }
    if (options_.events_file.empty()) {
        options_.events_file = default_path(options_.agent_dir, "events.jsonl");
    }
    if (options_.cancel_file.empty()) {
        options_.cancel_file = default_path(options_.agent_dir, "cancel");
    }
    if (options_.ready_file.empty()) {
        options_.ready_file = default_path(options_.agent_dir, "service-ready");
    }
    if (options_.legacy_request_file.empty() && !options_.root.empty()) {
        options_.legacy_request_file = default_path(options_.root, "app-request");
    }
}

bool AgentService::prepare(std::string* error) {
    if (!make_directories(options_.agent_dir)) {
        if (error != nullptr) {
            *error = "cannot create " + options_.agent_dir;
        }
        return false;
    }
    remove_file(options_.events_file);
    remove_file(options_.ready_file);
    remove_file(options_.cancel_file);
    return true;
}

void AgentService::emit(const Json& event) {
    const std::string line = event.dump();
    append_line(options_.events_file, line);
    if (options_.echo) {
        std::cout << "EVENT " << line << std::endl;
    }
}

Json AgentService::dispatch(AgentRuntime& runtime, const Json& request) {
    Json response;
    try {
        response = runtime.handle(request);
    } catch (const std::exception& failure) {
        response = Json::object();
        response.set("ok", Json::boolean(false));
        response.set("error", Json::string(failure.what()));
        Json event = Json::object();
        event.set("type", Json::string("error"));
        event.set("seq", Json::integer(0));
        event.set("ts_ms", Json::integer(now_ms()));
        event.set("session", Json::string(""));
        event.set("message", Json::string(failure.what()));
        event.set("stage", Json::string("service"));
        emit(event);
    }
    write_file(options_.response_file, response.dump());
    clear_cancel();
    return response;
}

bool AgentService::handle_pending(AgentRuntime& runtime,
                                  const std::function<void()>& legacy) {
    if (file_exists(options_.request_file)) {
        std::string error;
        const std::string contents = read_request(&error);
        if (!error.empty()) {
            Json response = Json::object();
            response.set("ok", Json::boolean(false));
            response.set("error", Json::string(error));
            write_file(options_.response_file, response.dump());
            return true;
        }
        Json request = Json::parse(contents);
        if (!request.is_object()) {
            // A bare text request is the friendly form the app may send.
            request = Json::object();
            request.set("type", Json::string("turn"));
            request.set("text", Json::string(trim(contents)));
        }
        dispatch(runtime, request);
        return true;
    }
    if (!options_.legacy_request_file.empty() &&
        file_exists(options_.legacy_request_file)) {
        // Claim the trigger before running it, otherwise the loop would run the
        // benchmark again on the next poll.
        remove_file(options_.legacy_request_file);
        if (legacy) {
            legacy();
        }
        // The legacy path resets the shared KV cache; the agent then starts its
        // next turn from a pristine engine.
        Json reset = Json::object();
        reset.set("type", Json::string("reset"));
        runtime.handle(reset);
        return true;
    }
    return false;
}

Json AgentService::hello(AgentRuntime& runtime, double startup_ms, double llm_load_ms,
                         const std::string& root) const {
    Json hello = runtime.capabilities();
    hello.set("type", Json::string("hello"));
    hello.set("startup_ms", Json::number(startup_ms));
    hello.set("llm_load_ms", Json::number(llm_load_ms));
    hello.set("root", Json::string(root));
    hello.set("events", Json::string(options_.events_file));
    hello.set("protocol", Json::string("gemma4-agent/1"));
    return hello;
}

void AgentService::write_ready(const std::string& contents) const {
    const std::string temporary = options_.ready_file + ".tmp";
    if (!write_file(temporary, contents + "\n")) {
        return;
    }
    rename(temporary.c_str(), options_.ready_file.c_str());
}

bool AgentService::cancel_requested() const {
    return file_exists(options_.cancel_file);
}

void AgentService::clear_cancel() const {
    remove_file(options_.cancel_file);
}

std::string AgentService::read_request(std::string* error) const {
    std::string contents;
    if (!read_file(options_.request_file, &contents)) {
        if (error != nullptr) {
            *error = "cannot read " + options_.request_file;
        }
        return std::string();
    }
    // Claim the request before parsing so a slow dispatch cannot process it twice.
    remove_file(options_.request_file);
    return trim(contents);
}

}  // namespace agent
