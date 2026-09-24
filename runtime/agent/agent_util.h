// Small helpers shared by the agent modules: text, time, files and ids.
// Kept free of MNN/Android dependencies so the host test build can use them.
#ifndef GEMMA4_AGENT_UTIL_H
#define GEMMA4_AGENT_UTIL_H

#include <cstdint>
#include <string>
#include <vector>

namespace agent {

long long now_ms();
std::string iso8601(long long epoch_ms);
std::string local_time_string(long long epoch_ms, const char* format);

std::string trim(const std::string& value);
std::string to_lower(const std::string& value);
std::string replace_all(std::string value, const std::string& from, const std::string& to);
std::vector<std::string> split(const std::string& value, char separator);
std::string join(const std::vector<std::string>& parts, const std::string& separator);
bool starts_with(const std::string& value, const std::string& prefix);
bool ends_with(const std::string& value, const std::string& suffix);
bool contains(const std::string& haystack, const std::string& needle);
size_t utf8_length(const std::string& value);
std::string head(const std::string& value, size_t max_bytes);
std::string tail(const std::string& value, size_t max_bytes);
std::string collapse_whitespace(const std::string& value);

std::string base64_encode(const std::string& value);
bool base64_decode(const std::string& value, std::string* out);
std::string sha256_hex(const std::string& data);

bool file_exists(const std::string& path);
bool is_directory(const std::string& path);
bool read_file(const std::string& path, std::string* out);
bool read_text_file(const std::string& path, std::string* out);
bool write_file(const std::string& path, const std::string& contents);
bool append_line(const std::string& path, const std::string& line);
bool make_directories(const std::string& path);
bool remove_file(const std::string& path);
// Removes a file, or a directory with everything below it.  Only ever called on
// paths the agent created itself (a session directory, its cached canvases).
bool remove_tree(const std::string& path);
long long file_size(const std::string& path);
std::vector<std::string> list_directory(const std::string& path);
std::string path_join(const std::string& left, const std::string& right);
std::string parent_path(const std::string& path);
std::string file_name(const std::string& path);

}  // namespace agent

#endif  // GEMMA4_AGENT_UTIL_H
