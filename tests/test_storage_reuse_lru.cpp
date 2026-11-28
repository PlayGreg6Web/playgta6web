// Storage relocation LRU (gpu/storage_reuse_lru.h): the indexed reuse lookup gives the same home as
// the linear reuse scan, on random insert / bind / erase / evict sequences with stale bind presents.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

#include "gpu/storage_reuse_lru.h"

namespace {

int g_failures = 0;

#define CHECK(cond)                                                                \
    do {                                                                           \
        if (!(cond)) {                                                             \
            std::fprintf(stderr, "FAIL [%s:%d]: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                          \
        }                                                                          \
    } while (0)

struct Entry;
using Lru = playgta6::gpu::d3d12_guest::StorageReuseLru<Entry>;
struct Entry {
    Lru::Node node;
    bool live = false;
};

uint64_t random_capacity(std::mt19937_64& rng) {
    switch (rng() % 4) {
    case 0: return uint64_t{256} << (rng() % 15);                // powers of two
    case 1: return (uint64_t{512} << (rng() % 15)) - 256;      // just below the next one
    case 2: return 256 * (1 + rng() % 64);                         // dense small sizes
    default: return 256 * (1 + rng() % (uint64_t{1} << 16));       // anything up to 16 MiB
    }
}

// Order and the reuse lookups agree with the linear scan at every step.
void run_sequence(uint64_t seed, size_t window, size_t pool, size_t ops) {
    std::mt19937_64 rng(seed);
    Lru lru(window);
    std::vector<std::unique_ptr<Entry>> entries;
    for (size_t i = 0; i < pool; ++i) {
        entries.push_back(std::make_unique<Entry>());
    }
    std::vector<Entry*> live;
    uint64_t present = 10;
    size_t found_both = 0, queries = 0;
    for (size_t op = 0; op < ops; ++op) {
        const unsigned kind = static_cast<unsigned>(rng() % 100);
        // Binds may carry a present read before the lock (up to two behind).
        const uint64_t stale = rng() % 8 == 0 ? std::min<uint64_t>(present, rng() % 3) : 0;
        if (kind < 3) {
            ++present;
        } else if (kind < 30) {
            Entry* e = entries[rng() % pool].get();
            if (!e->live) {
                lru.push_back(&e->node, e, present - stale, random_capacity(rng), rng() % 4 == 0);
                e->live = true;
                live.push_back(e);
            }
        } else if (kind < 70 && !live.empty()) {
            lru.touch(&live[rng() % live.size()]->node, present - stale);
        } else if (kind < 80 && !live.empty()) {
            const size_t i = rng() % live.size();
            Entry* e = rng() % 2 == 0 ? live[i] : lru.front()->entry;  // anywhere, or the eviction victim
            lru.erase(&e->node);
            e->live = false;
            live.erase(std::find(live.begin(), live.end(), e));
        } else {
            const uint64_t need = 256 * (1 + rng() % (rng() % 2 == 0 ? 64 : (uint64_t{1} << 16)));
            const bool dedicated = rng() % 4 == 0;
            const bool forced = rng() % 8 == 0;
            const uint64_t reuse_after = 1 + rng() % 4;
            size_t steps_linear = 0, steps_indexed = 0;
            const Lru::Node* a = lru.find_reuse_linear(present, forced, need, dedicated, reuse_after, &steps_linear);
            const Lru::Node* b = lru.find_reuse(present, forced, need, dedicated, reuse_after, &steps_indexed);
            CHECK(a == b);
            if (a != b) {
                std::fprintf(stderr, "  seed %llu op %zu: linear %p indexed %p\n",
                             static_cast<unsigned long long>(seed), op, static_cast<const void*>(a),
                             static_cast<const void*>(b));
                return;
            }
            ++queries;
            found_both += a != nullptr ? 1 : 0;
        }
        CHECK(lru.size() == live.size());
    }
    // Both outcomes were exercised.
    CHECK(queries > 0 && found_both > 0 && found_both < queries);
}

void test_random_sequences() {
    for (uint64_t seed = 1; seed <= 40; ++seed) {
        run_sequence(seed, 8, 32, 4000);       // window smaller than the population
        run_sequence(seed + 1000, 64, 48, 4000);  // window larger than the population
        run_sequence(seed + 2000, 128, 1024, 20000);
    }
}

void test_window_and_settled_stop() {
    Lru lru(2);
    Entry e[4];
    lru.push_back(&e[0].node, &e[0], 1, 256, false);   // too small
    lru.push_back(&e[1].node, &e[1], 1, 256, false);   // too small
    lru.push_back(&e[2].node, &e[2], 1, 4096, false);  // rank 2: outside a window of 2
    CHECK(lru.find_reuse(10, false, 1024, false, 3, nullptr) == nullptr);
    lru.touch(&e[0].node, 1);  // e[2] moves into the window
    CHECK(lru.find_reuse(10, false, 1024, false, 3, nullptr) == &e[2].node);
    // A recently bound node ahead of the candidate stops the walk.
    lru.erase(&e[0].node);
    lru.touch(&e[2].node, 9);
    lru.push_back(&e[3].node, &e[3], 1, 8192, false);
    CHECK(lru.find_reuse(10, false, 1024, false, 3, nullptr) == nullptr);   // e[1], e[2] (busy) | e[3]
    CHECK(lru.find_reuse(10, true, 1024, false, 3, nullptr) == &e[2].node);  // forced: no settle stop
    CHECK(lru.find_reuse(12, false, 1024, false, 3, nullptr) == &e[2].node);
    lru.clear();
    CHECK(lru.empty() && lru.front() == nullptr);
}

}  // namespace

int main() {
    test_window_and_settled_stop();
    test_random_sequences();
    if (g_failures != 0) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("storage reuse lru: ok\n");
    return 0;
}
