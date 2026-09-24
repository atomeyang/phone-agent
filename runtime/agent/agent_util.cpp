#include "agent_util.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <sstream>

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

namespace agent {
namespace {

bool is_space(char character) {
    return character == ' ' || character == '\t' || character == '\n' ||
           character == '\r' || character == '\f' || character == '\v';
}

// Minimal SHA-256 (FIPS 180-4).  Used for image fingerprints in the tool
// output; the implementation only has to be correct, not fast.
struct Sha256 {
    uint32_t state[8] = {0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
                         0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
    uint64_t length = 0;
    unsigned char buffer[64] = {};
    size_t buffered = 0;

    static uint32_t rotate_right(uint32_t value, uint32_t count) {
        return (value >> count) | (value << (32 - count));
    }

    void transform(const unsigned char* block) {
        static const uint32_t k[64] = {
            0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU,
            0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U,
            0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U,
            0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
            0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U,
            0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
            0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
            0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
            0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U,
            0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U, 0x1e376c08U,
            0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU,
            0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
            0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};
        uint32_t w[64];
        for (int index = 0; index < 16; ++index) {
            w[index] = (static_cast<uint32_t>(block[index * 4]) << 24) |
                       (static_cast<uint32_t>(block[index * 4 + 1]) << 16) |
                       (static_cast<uint32_t>(block[index * 4 + 2]) << 8) |
                       static_cast<uint32_t>(block[index * 4 + 3]);
        }
        for (int index = 16; index < 64; ++index) {
            const uint32_t s0 = rotate_right(w[index - 15], 7) ^
                                rotate_right(w[index - 15], 18) ^ (w[index - 15] >> 3);
            const uint32_t s1 = rotate_right(w[index - 2], 17) ^
                                rotate_right(w[index - 2], 19) ^ (w[index - 2] >> 10);
            w[index] = w[index - 16] + s0 + w[index - 7] + s1;
        }
        uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
        uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
        for (int index = 0; index < 64; ++index) {
            const uint32_t s1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
            const uint32_t ch = (e & f) ^ ((~e) & g);
            const uint32_t temp1 = h + s1 + ch + k[index] + w[index];
            const uint32_t s0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temp2 = s0 + maj;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        state[0] += a; state[1] += b; state[2] += c; state[3] += d;
        state[4] += e; state[5] += f; state[6] += g; state[7] += h;
    }

    void update(const std::string& data) {
        length += data.size();
        size_t offset = 0;
        while (offset < data.size()) {
            const size_t take = std::min(sizeof(buffer) - buffered, data.size() - offset);
            std::memcpy(buffer + buffered, data.data() + offset, take);
            buffered += take;
            offset += take;
            if (buffered == sizeof(buffer)) {
                transform(buffer);
                buffered = 0;
            }
        }
    }

    std::string finish() {
        const uint64_t bits = length * 8;
        unsigned char padding[72];
        std::memset(padding, 0, sizeof(padding));
        padding[0] = 0x80;
        const size_t pad_length = buffered < 56 ? 56 - buffered : 120 - buffered;
        std::string tail_bytes(reinterpret_cast<const char*>(padding), pad_length);
        std::string length_bytes(8, '\0');
        for (int index = 0; index < 8; ++index) {
            length_bytes[index] = static_cast<char>((bits >> (56 - index * 8)) & 0xff);
        }
        update(tail_bytes);
        update(length_bytes);
        char out[65];
        for (int index = 0; index < 8; ++index) {
            std::snprintf(out + index * 8, 9, "%08x", state[index]);
        }
        return std::string(out, 64);
    }
};

}  // namespace

long long now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::string iso8601(long long epoch_ms) {
    const std::time_t seconds = static_cast<std::time_t>(epoch_ms / 1000);
    std::tm parts{};
    gmtime_r(&seconds, &parts);
    char buffer[32];
    std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &parts);
    return buffer;
}

std::string local_time_string(long long epoch_ms, const char* format) {
    const std::time_t seconds = static_cast<std::time_t>(epoch_ms / 1000);
    std::tm parts{};
    localtime_r(&seconds, &parts);
    char buffer[64];
    std::strftime(buffer, sizeof(buffer), format, &parts);
    return buffer;
}

std::string trim(const std::string& value) {
    size_t begin = 0;
    size_t end = value.size();
    while (begin < end && is_space(value[begin])) {
        ++begin;
    }
    while (end > begin && is_space(value[end - 1])) {
        --end;
    }
    return value.substr(begin, end - begin);
}

std::string to_lower(const std::string& value) {
    std::string out(value);
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char character) {
        return static_cast<char>(std::tolower(character));
    });
    return out;
}

std::string replace_all(std::string value, const std::string& from, const std::string& to) {
    if (from.empty()) {
        return value;
    }
    size_t position = 0;
    while ((position = value.find(from, position)) != std::string::npos) {
        value.replace(position, from.size(), to);
        position += to.size();
    }
    return value;
}

std::vector<std::string> split(const std::string& value, char separator) {
    std::vector<std::string> parts;
    std::string current;
    for (const char character : value) {
        if (character == separator) {
            parts.push_back(current);
            current.clear();
        } else {
            current.push_back(character);
        }
    }
    parts.push_back(current);
    return parts;
}

std::string join(const std::vector<std::string>& parts, const std::string& separator) {
    std::string out;
    for (size_t index = 0; index < parts.size(); ++index) {
        if (index != 0) {
            out += separator;
        }
        out += parts[index];
    }
    return out;
}

bool starts_with(const std::string& value, const std::string& prefix) {
    return value.size() >= prefix.size() &&
           value.compare(0, prefix.size(), prefix) == 0;
}

bool ends_with(const std::string& value, const std::string& suffix) {
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

size_t utf8_length(const std::string& value) {
    size_t count = 0;
    for (const char character : value) {
        if ((static_cast<unsigned char>(character) & 0xc0) != 0x80) {
            ++count;
        }
    }
    return count;
}

std::string head(const std::string& value, size_t max_bytes) {
    if (value.size() <= max_bytes) {
        return value;
    }
    size_t end = max_bytes;
    while (end > 0 && (static_cast<unsigned char>(value[end]) & 0xc0) == 0x80) {
        --end;
    }
    return value.substr(0, end);
}

std::string tail(const std::string& value, size_t max_bytes) {
    if (value.size() <= max_bytes) {
        return value;
    }
    size_t begin = value.size() - max_bytes;
    while (begin < value.size() &&
           (static_cast<unsigned char>(value[begin]) & 0xc0) == 0x80) {
        ++begin;
    }
    return value.substr(begin);
}

std::string collapse_whitespace(const std::string& value) {
    std::string out;
    out.reserve(value.size());
    bool pending_space = false;
    for (const char character : value) {
        if (is_space(character)) {
            pending_space = !out.empty();
            continue;
        }
        if (pending_space) {
            out.push_back(' ');
            pending_space = false;
        }
        out.push_back(character);
    }
    return out;
}

std::string base64_encode(const std::string& value) {
    static constexpr char kAlphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string encoded;
    encoded.reserve((value.size() + 2) / 3 * 4);
    for (size_t offset = 0; offset < value.size(); offset += 3) {
        const uint32_t first = static_cast<unsigned char>(value[offset]);
        const uint32_t second = offset + 1 < value.size()
            ? static_cast<unsigned char>(value[offset + 1]) : 0;
        const uint32_t third = offset + 2 < value.size()
            ? static_cast<unsigned char>(value[offset + 2]) : 0;
        const uint32_t packed = (first << 16) | (second << 8) | third;
        encoded.push_back(kAlphabet[(packed >> 18) & 0x3f]);
        encoded.push_back(kAlphabet[(packed >> 12) & 0x3f]);
        encoded.push_back(offset + 1 < value.size() ? kAlphabet[(packed >> 6) & 0x3f] : '=');
        encoded.push_back(offset + 2 < value.size() ? kAlphabet[packed & 0x3f] : '=');
    }
    return encoded;
}

bool base64_decode(const std::string& value, std::string* out) {
    static int table[256];
    static bool ready = false;
    if (!ready) {
        std::memset(table, -1, sizeof(table));
        const char* alphabet =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        for (int index = 0; index < 64; ++index) {
            table[static_cast<unsigned char>(alphabet[index])] = index;
        }
        ready = true;
    }
    out->clear();
    int accumulator = 0;
    int bits = 0;
    for (const char character : value) {
        if (character == '=') {
            break;
        }
        if (is_space(character)) {
            continue;
        }
        const int digit = table[static_cast<unsigned char>(character)];
        if (digit < 0) {
            return false;
        }
        accumulator = (accumulator << 6) | digit;
        bits += 6;
        if (bits >= 8) {
            bits -= 8;
            out->push_back(static_cast<char>((accumulator >> bits) & 0xff));
        }
    }
    return true;
}

std::string sha256_hex(const std::string& data) {
    Sha256 hash;
    hash.update(data);
    return hash.finish();
}

bool file_exists(const std::string& path) {
    struct stat info {};
    return stat(path.c_str(), &info) == 0;
}

bool is_directory(const std::string& path) {
    struct stat info {};
    return stat(path.c_str(), &info) == 0 && S_ISDIR(info.st_mode);
}

bool read_file(const std::string& path, std::string* out) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return false;
    }
    std::ostringstream contents;
    contents << stream.rdbuf();
    *out = contents.str();
    return true;
}

bool read_text_file(const std::string& path, std::string* out) {
    if (!read_file(path, out)) {
        return false;
    }
    *out = trim(*out);
    return true;
}

bool write_file(const std::string& path, const std::string& contents) {
    const std::string temporary = path + ".part";
    {
        std::ofstream stream(temporary, std::ios::trunc | std::ios::binary);
        if (!stream) {
            return false;
        }
        stream.write(contents.data(), static_cast<std::streamsize>(contents.size()));
        stream.flush();
        if (!stream) {
            return false;
        }
    }
    if (rename(temporary.c_str(), path.c_str()) != 0) {
        remove_file(temporary);
        return false;
    }
    chmod(path.c_str(), 0666);
    return true;
}

bool append_line(const std::string& path, const std::string& line) {
    std::ofstream stream(path, std::ios::app | std::ios::binary);
    if (!stream) {
        return false;
    }
    stream.write(line.data(), static_cast<std::streamsize>(line.size()));
    stream.put('\n');
    stream.flush();
    chmod(path.c_str(), 0666);
    return static_cast<bool>(stream);
}

bool make_directories(const std::string& path) {
    if (path.empty() || is_directory(path)) {
        return true;
    }
    const std::string parent = parent_path(path);
    if (!parent.empty() && parent != path && !is_directory(parent)) {
        if (!make_directories(parent)) {
            return false;
        }
    }
    if (mkdir(path.c_str(), 0777) == 0) {
        chmod(path.c_str(), 0777);
        return true;
    }
    return is_directory(path);
}

bool remove_file(const std::string& path) {
    return unlink(path.c_str()) == 0 || !file_exists(path);
}

bool remove_tree(const std::string& path) {
    if (!is_directory(path)) {
        return remove_file(path);
    }
    for (const auto& name : list_directory(path)) {
        remove_tree(path_join(path, name));
    }
    return rmdir(path.c_str()) == 0 || !file_exists(path);
}

long long file_size(const std::string& path) {
    struct stat info {};
    if (stat(path.c_str(), &info) != 0) {
        return -1;
    }
    return static_cast<long long>(info.st_size);
}

std::vector<std::string> list_directory(const std::string& path) {
    std::vector<std::string> entries;
    DIR* directory = opendir(path.c_str());
    if (directory == nullptr) {
        return entries;
    }
    while (struct dirent* entry = readdir(directory)) {
        const std::string name = entry->d_name;
        if (name == "." || name == "..") {
            continue;
        }
        entries.push_back(name);
    }
    closedir(directory);
    std::sort(entries.begin(), entries.end());
    return entries;
}

std::string path_join(const std::string& left, const std::string& right) {
    if (left.empty()) {
        return right;
    }
    if (right.empty()) {
        return left;
    }
    if (left.back() == '/') {
        return left + right;
    }
    return left + "/" + right;
}

std::string parent_path(const std::string& path) {
    const size_t position = path.find_last_of('/');
    if (position == std::string::npos) {
        return std::string();
    }
    if (position == 0) {
        return "/";
    }
    return path.substr(0, position);
}

std::string file_name(const std::string& path) {
    const size_t position = path.find_last_of('/');
    return position == std::string::npos ? path : path.substr(position + 1);
}

}  // namespace agent
