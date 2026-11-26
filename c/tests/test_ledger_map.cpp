/* ledger_map.h without a GPU: the regions file Ledger reads, the overlap rule that stands
 * in for the free paths, the names, and the COLI_LEDGER_WINDOW steps. */
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>
#include "../ledger_map.h"

#ifdef _WIN32
static void set_env(const char *k, const char *v) { _putenv_s(k, v); }
#else
static void set_env(const char *k, const char *v) { setenv(k, v, 1); }
#endif

static std::string slurp(const std::string &path) {
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) return "";
    std::string s;
    char buf[4096];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, n);
    std::fclose(f);
    return s;
}

int main(void) {
    long long skip, take;
    coli_ledger::parse_window("8:40", &skip, &take);
    assert(skip == 8 && take == 40);
    for (const char *bad : {"", ":4", "8:", "8:0", "0:4", "-1:4", "8:4x", "8", "a:b"}) {
        coli_ledger::parse_window(bad, &skip, &take);
        assert(skip == -1 && take == 0);
    }
    coli_ledger::parse_window(nullptr, &skip, &take);
    assert(skip == -1);

    const std::string dir = ".";
    set_env("LEDGER_OUT", dir.c_str());
    set_env("COLI_LEDGER_WINDOW", "2:3");
    assert(coli_ledger::on());

    /* a name of the engine's (a label) or the backend's own, numbered */
    assert(coli_ledger::name("tensor") == "tensor.0");
    assert(coli_ledger::kind("weight") == "weight");
    coli_ledger::set_label("cache", "layers.12.kv");
    assert(coli_ledger::name("tensor") == "layers.12.kv" && coli_ledger::kind("buffer") == "cache");
    coli_ledger::set_label("weight", "layers.12.experts.slot3.w13");
    assert(coli_ledger::name("tensor") == "layers.12.experts.slot3.w13");
    coli_ledger::set_label("weight", nullptr);
    assert(coli_ledger::name("tensor") == "tensor.1" && coli_ledger::kind("buffer") == "buffer");

    coli_ledger::add((void *)0x1000, 0x100, "weight", "a", {16, 4}, "uint8");
    coli_ledger::add((void *)0x2000, 0x100, "cache", "b\"\\\n", {}, "float32");
    coli_ledger::add(nullptr, 0x100, "weight", "null", {}, "uint8");   /* nothing to map */
    coli_ledger::add((void *)0x3000, 0, "weight", "empty", {}, "uint8");
    assert(coli_ledger::map().live.size() == 2);
    /* a new region overlapping an old one replaces it; one that only touches it does not */
    coli_ledger::add((void *)0x10f0, 0x20, "weight", "c", {32}, "uint8");
    coli_ledger::add((void *)0x1110, 0x10, "weight", "d", {16}, "uint8");
    assert(coli_ledger::map().live.size() == 3);
    assert(coli_ledger::map().live.begin()->second.name == "c");

    /* in address order */
    const std::string expect =
        "[\n{\"kind\":\"weight\",\"name\":\"c\",\"addr\":4336,\"size\":32,\"shape\":[32],\"dtype\":\"uint8\"},\n"
        "{\"kind\":\"weight\",\"name\":\"d\",\"addr\":4368,\"size\":16,\"shape\":[16],\"dtype\":\"uint8\"},\n"
        "{\"kind\":\"cache\",\"name\":\"b\\\"\\\\\\u000a\",\"addr\":8192,\"size\":256,\"shape\":[],\"dtype\":\"float32\"}\n]\n";
    assert(coli_ledger::json(coli_ledger::map().live) == expect);

    /* the steps write the map when it changed, and open the window after step 2 and close
     * it after step 5 (steps 2, 3, 4 counted from 0 are inside) */
    const std::string path = dir + "/regions-" + std::to_string((long long)coli_ledger_getpid()) + "-ledger-map-test.json";
    std::remove(path.c_str());
    assert(coli_ledger::step("ledger-map-test") == 0);
    assert(slurp(path) == expect);
    std::remove(path.c_str());
    assert(coli_ledger::step("ledger-map-test") == 1);
    assert(slurp(path).empty());                 /* unchanged: not written again */
    coli_ledger::add((void *)0x4000, 0x40, "weight", "e", {}, "uint8");
    assert(coli_ledger::step("ledger-map-test") == 0);
    assert(slurp(path).find("\"name\":\"e\"") != std::string::npos);
    assert(coli_ledger::step("ledger-map-test") == 0);
    assert(coli_ledger::step("ledger-map-test") == -1);
    assert(coli_ledger::step("ledger-map-test") == 0);
    assert(slurp(path + ".tmp").empty());
    std::remove(path.c_str());
    /* one region over several drops them all; one starting inside an earlier one drops it */
    coli_ledger::add((void *)0x1000, 0x1100, "weight", "f", {}, "uint8");
    assert(coli_ledger::map().live.size() == 2 && coli_ledger::map().live.count(0x4000));
    coli_ledger::add((void *)0x4020, 0x10, "cache", "g", {}, "uint8");
    assert(coli_ledger::map().live.size() == 2 && !coli_ledger::map().live.count(0x4000));
    assert(coli_ledger::map().live.begin()->second.name == "f");
    std::puts("ledger map tests: ok");
    return 0;
}
