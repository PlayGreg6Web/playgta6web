// Contention stress test for XMem write tracking (O33): several threads each own a few 64 KiB
// physical pages mapped in a shared (window) reservation and loop over
//   write (faults on an armed view) -> take_written_pages on their own pages (re-arms them)
//   -> now and then unmap + remap one page (view tracking),
// while one "render" thread keeps taking the written pages of the whole reservation (as the GPU
// upload path does). Every write must be reported to its owner's next take, so the test doubles
// as a correctness check of the tracker under concurrency.
//
//   bench_xmem_contention [--full] [--threads N]
//
// Without --full (CTest) it runs a short smoke pass; --full prints iterations per second.
// It also times write-tracking harvests on a single thread (the GPU submit thread's pattern: many
// small per-resource takes per frame, a few pages written in between): ns per take over a
// MEM_WRITE_WATCH arena and over shared page views, checking every write is reported.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include "os/vmem.h"
#include "os/win32_kernel.h"
#include "os/xbox_scarlett.h"

namespace {

using namespace playgta6::os;
using namespace playgta6::os::scarlett;
using Clock = std::chrono::steady_clock;

constexpr uint32_t kWindowType = 0x20402000u;  // MEM_RESERVE | MEM_64K_PAGES | MEM_PHYSICAL
constexpr uint32_t kCpuProtect = 0x4u;         // PAGE_READWRITE
constexpr size_t kPagesPerThread = 4;

std::atomic<bool> g_stop{false};
std::atomic<int> g_failures{0};
std::atomic<int> g_started{0};  // threads that reached their loop

struct Result {
    std::atomic<uint64_t> iterations{0};  // (published as it goes: the smoke pass waits for progress)
};

void worker(uint64_t base_va, const uint64_t* pfns, int seed, Result* out) {
    auto* arena = win32::get_vmem_arena();
    uint8_t* p = arena->translate(base_va);
    const uint64_t len = kPagesPerThread * kXMemPageSize64K;
    WriteCursor cursor;
    std::vector<uint64_t> pages;
    take_written_pages(p, len, &cursor, &pages);  // (initial: everything new)
    g_started.fetch_add(1);
    uint64_t it = 0;
    while (!g_stop.load(std::memory_order_relaxed)) {
        const size_t i = static_cast<size_t>(it % kPagesPerThread);
        if (it % 64 == 63) {
            // The map path: unmap and remap the page (contents kept).
            const uint64_t va = base_va + i * kXMemPageSize64K;
            if (pg6_xbox_XMemMapPhysicalPagesScatter(&va, 1, nullptr) != 1 ||
                pg6_xbox_XMemMapPhysicalPagesScatter(&va, 1, &pfns[i]) != 1) {
                std::fprintf(stderr, "FAIL: remap (thread %d)\n", seed);
                g_failures.fetch_add(1);
                return;
            }
            take_written_pages(p, len, &cursor, &pages);  // (a new mapping counts as written)
        }
        const uint64_t off = i * kXMemPageSize64K + (it * 4096 + 17) % kXMemPageSize64K;
        p[off] = static_cast<uint8_t>(seed + it);
        pages.clear();
        if (!take_written_pages(p, len, &cursor, &pages) ||
            std::find(pages.begin(), pages.end(), off & ~uint64_t{kWrittenPageSize - 1}) == pages.end()) {
            std::fprintf(stderr, "FAIL: write at +0x%llx not reported (thread %d)\n",
                         static_cast<unsigned long long>(off), seed);
            g_failures.fetch_add(1);
            return;
        }
        if (p[off] != static_cast<uint8_t>(seed + it)) {
            std::fprintf(stderr, "FAIL: contents (thread %d)\n", seed);
            g_failures.fetch_add(1);
            return;
        }
        ++it;
        out->iterations.store(it, std::memory_order_relaxed);
    }
}

// ns per take_written_pages call, `takes` resource-sized takes over `res` resources of `res_len`
// bytes starting at `p`, after writing one byte into every 8th resource. Fails if a write is missed.
double harvest_ns(uint8_t* p, uint64_t res, uint64_t res_len, int rounds, const char* what) {
    std::vector<WriteCursor> cursors(static_cast<size_t>(res));
    std::vector<uint64_t> pages;
    for (uint64_t r = 0; r < res; ++r) {
        take_written_pages(p + r * res_len, res_len, &cursors[static_cast<size_t>(r)], &pages);
    }
    uint64_t takes = 0;
    const auto t0 = Clock::now();
    for (int round = 0; round < rounds; ++round) {
        for (uint64_t r = round % 8; r < res; r += 8) {
            p[r * res_len + (round * 4096 + 5) % res_len] = static_cast<uint8_t>(round);
        }
        for (uint64_t r = 0; r < res; ++r) {
            pages.clear();
            const uint64_t want = (round * 4096 + 5) % res_len & ~uint64_t{kWrittenPageSize - 1};
            if (!take_written_pages(p + r * res_len, res_len, &cursors[static_cast<size_t>(r)], &pages) ||
                (r % 8 == static_cast<uint64_t>(round % 8) &&
                 std::find(pages.begin(), pages.end(), want) == pages.end())) {
                std::fprintf(stderr, "FAIL: harvest (%s) missed a write\n", what);
                g_failures.fetch_add(1);
                return 0;
            }
            ++takes;
        }
    }
    return std::chrono::duration<double, std::nano>(Clock::now() - t0).count() / static_cast<double>(takes);
}

}  // namespace

int main(int argc, char** argv) {
    bool full = false;
    int threads = 4;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--full") == 0) {
            full = true;
        } else if (std::strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            threads = std::max(1, std::atoi(argv[++i]));
        } else {
            std::fprintf(stderr, "usage: bench_xmem_contention [--full] [--threads N]\n");
            return 64;
        }
    }
#if !defined(_WIN32)
    std::printf("bench_xmem_contention: shared views are write tracked on Windows only, skipped\n");
    return 0;
#endif
    xmem_reset_for_testing();
    auto* arena = win32::get_vmem_arena();
    if (!arena->has_shared_zone()) {
        std::printf("bench_xmem_contention: no shared zone on this host, skipped\n");
        return 0;
    }
    const size_t total = kPagesPerThread * static_cast<size_t>(threads);
    void* res = pg6_xbox_XMemVirtualAlloc(nullptr, uint64_t{total} * kXMemPageSize64K, kWindowType, 0x23u, kCpuProtect);
    if (res == nullptr || !arena->is_shared(arena->guest_address(res))) {
        std::fprintf(stderr, "FAIL: shared reservation\n");
        return 1;
    }
    const uint64_t base = arena->guest_address(res);
    std::vector<uint64_t> pfns(total);
    size_t n = total;
    if (pg6_xbox_XMemAllocatePhysicalPages(0x380u, &n, pfns.data()) != 1 || n != total) {
        std::fprintf(stderr, "FAIL: physical pages\n");
        return 1;
    }
    std::vector<uint64_t> vas(total);
    for (size_t i = 0; i < total; ++i) {
        vas[i] = base + i * kXMemPageSize64K;
    }
    if (pg6_xbox_XMemMapPhysicalPagesScatter(vas.data(), total, pfns.data()) != 1) {
        std::fprintf(stderr, "FAIL: map\n");
        return 1;
    }

    // Harvest cost: 256 resources of 64 KiB in a MEM_WRITE_WATCH arena, then the shared views.
    {
        set_write_tracking(true);
        VirtualMemoryArena watched;
        set_write_tracking(false);
        constexpr uint64_t kRes = 256, kResLen = 64 * 1024;
        const uint64_t a = watched.reserve(0, kRes * kResLen);
        uint8_t* wp = a != 0 && watched.commit(a, kRes * kResLen, PageProtect::ReadWrite)
                          ? watched.host_ptr(a, kRes * kResLen)
                          : nullptr;
        const int rounds = full ? 400 : 20;
        if (wp != nullptr) {
            std::printf("harvest: write-watch arena, %llu x 64 KiB takes: %.0f ns/take\n",
                        static_cast<unsigned long long>(kRes), harvest_ns(wp, kRes, kResLen, rounds, "watch"));
        }
        // An arena without MEM_WRITE_WATCH: every take answers "unknown" (false).
        VirtualMemoryArena plain;
        const uint64_t b = plain.reserve(0, kResLen);
        uint8_t* pp = b != 0 && plain.commit(b, kResLen, PageProtect::ReadWrite) ? plain.host_ptr(b, kResLen) : nullptr;
        if (pp != nullptr) {
            WriteCursor cursor;
            std::vector<uint64_t> pages;
            take_written_pages(pp, kResLen, &cursor, &pages);
            const int n = rounds * 256;
            const auto t0 = Clock::now();
            for (int i = 0; i < n; ++i) {
                pages.clear();
                if (take_written_pages(pp, kResLen, &cursor, &pages)) {
                    std::fprintf(stderr, "FAIL: harvest of an unwatched arena claims tracking\n");
                    g_failures.fetch_add(1);
                    break;
                }
            }
            std::printf("harvest: unwatched arena, 64 KiB takes: %.0f ns/take\n",
                        std::chrono::duration<double, std::nano>(Clock::now() - t0).count() / n);
        }
        std::printf("harvest: shared views, %zu x 64 KiB takes: %.0f ns/take\n", total,
                    harvest_ns(arena->translate(base), total, kXMemPageSize64K, rounds * 4, "views"));
        std::printf("harvest: shared views, one %zu KiB take: %.0f ns/take\n", total * 64,
                    harvest_ns(arena->translate(base), 1, total * kXMemPageSize64K, rounds * 16, "range"));
    }

    std::vector<Result> results(static_cast<size_t>(threads));
    std::atomic<uint64_t> render_takes{0};
    std::vector<std::thread> pool;
    const auto created = Clock::now();
    for (int w = 0; w < threads; ++w) {
        pool.emplace_back(worker, base + uint64_t{kPagesPerThread} * w * kXMemPageSize64K,
                          &pfns[kPagesPerThread * static_cast<size_t>(w)], w * 31 + 7, &results[static_cast<size_t>(w)]);
    }
    std::thread render([&] {
        WriteCursor cursor;
        std::vector<uint64_t> pages;
        const uint8_t* p = arena->translate(base);
        g_started.fetch_add(1);
        while (!g_stop.load(std::memory_order_relaxed)) {
            pages.clear();
            take_written_pages(p, uint64_t{total} * kXMemPageSize64K, &cursor, &pages);
            render_takes.fetch_add(1, std::memory_order_relaxed);
        }
    });
    // The window starts once every thread runs. A fixed 300 ms window from thread creation made the
    // smoke pass measure the scheduler, not the tracker: on an oversubscribed runner (CTest --parallel
    // on 4 vCPUs, Windows Server's long fixed quanta) new threads can wait longer than that for their
    // first time slice, and a lock holder preempted mid-take stalls the others for a whole quantum,
    // so it ended with 0 iterations, 0 takes and 0 faults. The smoke pass therefore runs at least
    // 300 ms and until every worker has done kMinIterations (two remaps each) and the render thread
    // kMinIterations takes, failing only if that takes longer than kDeadline.
    constexpr uint64_t kMinIterations = 128;
    constexpr auto kDeadline = std::chrono::seconds(60);
    const int all = threads + 1;
    while (g_started.load() < all && Clock::now() - created < kDeadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    const auto t0 = Clock::now();
    // (Counted from here: a thread that started early has a head start the rates leave out.)
    const uint64_t faults0 = shared_view_write_faults();
    const uint64_t takes0 = render_takes.load();
    uint64_t iters0 = 0;
    for (const Result& r : results) {
        iters0 += r.iterations.load();
    }
    const double start_ms = std::chrono::duration<double, std::milli>(t0 - created).count();
    auto progressed = [&] {
        if (render_takes.load(std::memory_order_relaxed) < kMinIterations) {
            return false;
        }
        for (const Result& r : results) {
            if (r.iterations.load(std::memory_order_relaxed) < kMinIterations) {
                return false;
            }
        }
        return true;
    };
    const auto window = std::chrono::milliseconds(full ? 3000 : 300);
    bool timed_out = false;
    std::this_thread::sleep_for(window);
    while (!full && !progressed()) {
        if (Clock::now() - created >= kDeadline) {
            timed_out = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    g_stop.store(true);
    for (auto& t : pool) {
        t.join();
    }
    render.join();
    const double secs = std::chrono::duration<double>(Clock::now() - t0).count();
    uint64_t iters = 0;
    uint64_t slowest = ~uint64_t{0};
    for (const Result& r : results) {
        iters += r.iterations.load();  // (minus iters0 below)
        slowest = std::min(slowest, r.iterations.load());
    }
    iters -= iters0;
    std::printf("%d thread(s) + render taker, %.2f s: %.0f write/take iterations/s (%.0f per thread), "
                "%.0f render takes/s, %llu faults\n",
                threads, secs, iters / secs, iters / secs / threads, (render_takes.load() - takes0) / secs,
                static_cast<unsigned long long>(shared_view_write_faults() - faults0));
    std::printf("threads running %.1f ms after creation; slowest worker %llu iterations\n", start_ms,
                static_cast<unsigned long long>(slowest));
    if (timed_out) {
        std::fprintf(stderr, "FAIL: no progress within %lld s (%d/%d threads started, slowest worker %llu iterations, "
                     "%llu render takes)\n",
                     static_cast<long long>(kDeadline.count()), g_started.load(), all,
                     static_cast<unsigned long long>(slowest), static_cast<unsigned long long>(render_takes.load()));
        g_failures.fetch_add(1);
    }
    pg6_xbox_XMemMapPhysicalPagesScatter(vas.data(), total, nullptr);
    pg6_xbox_XMemFreePhysicalPages(total, pfns.data());
    xmem_virtual_free(res);
    if (g_failures.load() != 0 || slowest == 0 || render_takes.load() == 0) {
        std::fprintf(stderr, "bench_xmem_contention: FAILED\n");
        return 1;
    }
    return 0;
}
