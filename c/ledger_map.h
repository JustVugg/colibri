/* The memory map and the measurement window for Ledger (github.com/sec4si/ledger), an
 * LD_PRELOAD CUPTI library that records the parameters of every kernel launch and every
 * memory copy, and resolves their pointers against the program's memory map: which launch
 * read which weight, which copy carried which expert, how much copy time no kernel hid.
 *
 * With LEDGER_OUT set, a GPU backend keeps the list of its device allocations (address,
 * size, a name, a shape and a dtype) and writes it as $LEDGER_OUT/regions-<pid>-<tag>.json,
 * the file Ledger reads, whenever it changed and the engine finishes a decode step.
 * COLI_LEDGER_WINDOW=<skip>:<take> brackets decode steps skip .. skip+take-1 with
 * cudaProfilerStart/Stop, Ledger's window. Without LEDGER_OUT nothing is recorded and
 * nothing is written; under HIP the map is always off.
 *
 * A region that overlaps a new one is dropped when the new one is added: two live device
 * allocations never overlap, so the old one was freed (or was the arena the new one is
 * carved from, and Ledger resolves a pointer to one region only). The free paths are
 * left alone. */
#ifndef COLIBRI_LEDGER_MAP_H
#define COLIBRI_LEDGER_MAP_H

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#ifdef _WIN32
#include <process.h>
#define coli_ledger_getpid _getpid
#else
#include <unistd.h>
#define coli_ledger_getpid getpid
#endif

namespace coli_ledger {

struct Region {
    uintptr_t addr;
    size_t size;
    std::string kind;   /* "weight", "cache" or "buffer" */
    std::string name;
    std::vector<long long> shape;
    const char *dtype;  /* a string literal */
};

struct Map {
    std::mutex mu;
    std::map<uintptr_t, Region> live;   /* by address; no two overlap */
    bool dirty = false;
    long long steps = 0;
};

inline Map &map() { static Map m; return m; }

inline const char *out_dir() {
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIP__)
    return nullptr;
#else
    static const char *dir = [] { const char *d = std::getenv("LEDGER_OUT"); return d && *d ? d : nullptr; }();
    return dir;
#endif
}

inline bool on() { return out_dir() != nullptr; }

/* What the engine calls the next allocations on this thread (an upload names its weights
 * and scales `<name>.weight` and `<name>.scale`); no name: the backend's own. */
struct Label {
    std::string kind, name;
};
inline Label &label() { thread_local Label l; return l; }
inline void set_label(const char *kind, const char *name) {
    label().kind = kind && name ? kind : "";
    label().name = kind && name ? name : "";
}

inline void add(const void *addr, size_t size, const std::string &kind, const std::string &name,
                std::vector<long long> shape, const char *dtype) {
    if (!on() || !addr || !size) return;
    const uintptr_t a = (uintptr_t)addr;
    Map &m = map();
    std::lock_guard<std::mutex> lock(m.mu);
    auto it = m.live.lower_bound(a);
    if (it != m.live.begin() && std::prev(it)->second.addr + std::prev(it)->second.size > a) --it;
    while (it != m.live.end() && it->first < a + size) it = m.live.erase(it);
    m.live.emplace(a, Region{a, size, kind, name, std::move(shape), dtype});
    m.dirty = true;
}

/* `base.<n>`, numbered in allocation order: the backend's own name for a buffer */
inline std::string numbered(const char *base) {
    static std::atomic<long long> n{0};
    return std::string(base) + "." + std::to_string(n++);
}

/* The label's name, or the backend's own */
inline std::string name(const char *fallback) {
    const Label &l = label();
    return l.name.empty() ? numbered(fallback) : l.name;
}

/* The label's kind, or `fallback` */
inline std::string kind(const char *fallback) {
    const Label &l = label();
    return l.name.empty() ? fallback : l.kind;
}

inline void json_string(std::string &out, const std::string &s) {
    out += '"';
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { out += '\\'; out += (char)c; }
        else if (c < 0x20) { char e[8]; std::snprintf(e, sizeof e, "\\u%04x", c); out += e; }
        else out += (char)c;
    }
    out += '"';
}

inline std::string json(const std::map<uintptr_t, Region> &v) {
    std::string out = "[";
    for (const auto &entry : v) {
        const Region &r = entry.second;
        out += out.size() > 1 ? ",\n{\"kind\":" : "\n{\"kind\":";
        json_string(out, r.kind);
        out += ",\"name\":";
        json_string(out, r.name);
        out += ",\"addr\":" + std::to_string((unsigned long long)r.addr) +
               ",\"size\":" + std::to_string((unsigned long long)r.size) + ",\"shape\":[";
        for (size_t d = 0; d < r.shape.size(); d++) out += (d ? "," : "") + std::to_string(r.shape[d]);
        out += "],\"dtype\":";
        json_string(out, r.dtype);
        out += '}';
    }
    return out + "\n]\n";
}

/* Write the map if it changed, through a temporary file so a reader never sees half of it. */
inline bool write(const char *tag) {
    if (!on()) return false;
    Map &m = map();
    std::string text;
    {
        std::lock_guard<std::mutex> lock(m.mu);
        if (!m.dirty) return true;
        text = json(m.live);
        m.dirty = false;
    }
    const std::string path = std::string(out_dir()) + "/regions-" + std::to_string((long long)coli_ledger_getpid()) + "-" + tag + ".json";
    const std::string tmp = path + ".tmp";
    FILE *f = std::fopen(tmp.c_str(), "wb");
    if (!f) { std::fprintf(stderr, "[LEDGER] cannot write %s\n", tmp.c_str()); return false; }
    const bool ok = std::fwrite(text.data(), 1, text.size(), f) == text.size();
    if (std::fclose(f) != 0 || !ok) { std::remove(tmp.c_str()); return false; }
#ifdef _WIN32
    std::remove(path.c_str());
#endif
    if (std::rename(tmp.c_str(), path.c_str()) != 0) { std::remove(tmp.c_str()); return false; }
    return true;
}

/* COLI_LEDGER_WINDOW=<skip>:<take> -> {skip, take}; {-1, 0} when unset or malformed.
 * skip is at least 1: the window opens after a finished step (the first is never steady). */
inline void parse_window(const char *s, long long *skip, long long *take) {
    *skip = -1; *take = 0;
    if (!s) return;
    char *end = nullptr;
    const long long a = std::strtoll(s, &end, 10);
    if (end == s || *end != ':' || a < 1) return;
    const char *t = end + 1;
    const long long b = std::strtoll(t, &end, 10);
    if (end == t || *end || b < 1) return;
    *skip = a; *take = b;
}

/* One decode step has finished. Returns +1 when the window opens before the next step,
 * -1 when it closes, 0 otherwise; the caller synchronizes its devices and calls
 * cudaProfilerStart/Stop. The map is written first, so it is on disk when the window
 * opens. */
inline int step(const char *tag) {
    if (!on()) return 0;
    static const std::pair<long long, long long> w = [] {
        long long skip, take;
        parse_window(std::getenv("COLI_LEDGER_WINDOW"), &skip, &take);
        return std::make_pair(skip, take);
    }();
    write(tag);
    const long long done = ++map().steps;
    if (w.first < 0) return 0;
    return done == w.first ? 1 : done == w.first + w.second ? -1 : 0;
}

}  // namespace coli_ledger

#endif
