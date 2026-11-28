// LivePump (gpu/native_exec/live_pump.h): the bounded queue between the submit thread and the
// executor thread, backpressure, window close and focus forwarding, with a fake executor.
// No GPU, no Dawn.
#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "gpu/native_exec/live_pump.h"

namespace {
int g_failures = 0;
void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_live_pump.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}
#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6::gpu::native_exec;
using namespace std::chrono_literals;

// Waits until `cond` holds (10 s at most). The tests used fixed sleeps for "the other thread got
// there by now", which fails on an oversubscribed CI runner where a thread can wait longer than
// that for its first time slice.
template <class F>
bool wait_until(F cond) {
    const auto deadline = std::chrono::steady_clock::now() + 10s;
    while (!cond()) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::sleep_for(1ms);
    }
    return true;
}

// State shared with the test body (the executor itself lives on the pump thread).
struct Shared {
    std::mutex m;
    std::vector<uint8_t> first_bytes;  // first byte of every executed batch, in order
    std::thread::id exec_thread, create_thread, destroy_thread;
    std::atomic<bool> gate{true};      // false = execute() blocks (a slow executor)
    std::atomic<bool> close{false};    // poll() returns false
    std::atomic<bool> focus{false};
    std::atomic<int> waits{0};
    std::atomic<bool> destroyed{false};
};

struct FakeExecutor {
    explicit FakeExecutor(Shared* s) : sh(s) { sh->create_thread = std::this_thread::get_id(); }
    ~FakeExecutor() {
        sh->destroy_thread = std::this_thread::get_id();
        sh->destroyed = true;
    }
    uint32_t execute(const uint8_t* data, size_t len) {
        while (!sh->gate.load()) std::this_thread::sleep_for(1ms);
        std::lock_guard<std::mutex> lock(sh->m);
        sh->exec_thread = std::this_thread::get_id();
        sh->first_bytes.push_back(data[0]);
        return len > 1 ? 1u : 0u;  // pretend multi-byte batches carry a Present
    }
    bool poll() { return !sh->close.load(); }
    bool window_focused() const { return sh->focus.load(); }
    void wait_idle() { ++sh->waits; }
    Shared* sh;
};

using Pump = LivePump<FakeExecutor>;

bool start(Pump& pump, Shared& sh, std::function<void()> on_close = {}, std::function<void(bool)> on_focus = {}) {
    std::string error;
    return pump.start([&sh](std::string*) { return std::make_unique<FakeExecutor>(&sh); }, std::move(on_close),
                      std::move(on_focus), &error);
}

template <class P>
void submit_byte(P& p, uint8_t b, size_t len = 2) {
    std::vector<uint8_t> v(len, b);
    p.submit(v.data(), v.size());
}

void test_order_and_threads() {
    Shared sh;
    Pump pump;
    CHECK(start(pump, sh));
    for (uint8_t i = 0; i < 50; ++i) submit_byte(pump, i);
    pump.flush();
    const LivePumpStats s = pump.stats();
    CHECK(s.submitted == 50 && s.executed == 50 && s.presents == 50);
    CHECK(s.blocked_submits == 0);
    pump.stop();
    CHECK(sh.first_bytes.size() == 50);
    for (size_t i = 0; i < sh.first_bytes.size(); ++i) CHECK(sh.first_bytes[i] == i);
    // The executor is created, used and destroyed on the pump thread, never on the caller's.
    CHECK(sh.create_thread != std::this_thread::get_id());
    CHECK(sh.create_thread == sh.exec_thread && sh.exec_thread == sh.destroy_thread);
    CHECK(sh.destroyed.load() && sh.waits.load() == 1);
    CHECK(!pump.window_closed());
}

void test_backpressure_blocks_and_counts() {
    Shared sh;
    sh.gate = false;
    LivePumpOptions opt;
    opt.max_queued_batches = 4;
    Pump pump(opt);
    CHECK(start(pump, sh));
    std::atomic<int> accepted{0};
    std::thread producer([&] {
        for (uint8_t i = 0; i < 20; ++i) {
            submit_byte(pump, i);
            ++accepted;
        }
    });
    // One batch is inside the stuck execute(), 4 are queued: the producer must be stopped there.
    CHECK(wait_until([&] { return accepted.load() == 5 && pump.stats().blocked_now == 1; }));
    std::this_thread::sleep_for(120ms);  // (still stopped, and the in-flight times below grow past 100 ms)
    CHECK(accepted.load() == 5);
    CHECK(pump.stats().max_queue_depth == 4);
    // The in-flight snapshot shows the never-ending wait and batch (the totals do not yet).
    const LivePumpStats stuck = pump.stats();
    CHECK(stuck.blocked_now == 1 && stuck.blocked_now_ns >= 100'000'000ull);
    CHECK(stuck.queued_now == 4 && stuck.executing_ns >= 100'000'000ull);
    CHECK(pump.stats_json().find("\"executing_ms\": ") != std::string::npos);
    sh.gate = true;
    producer.join();
    pump.flush();
    const LivePumpStats s = pump.stats();
    CHECK(s.blocked_now == 0 && s.blocked_now_ns == 0 && s.queued_now == 0 && s.executing_ns == 0);
    CHECK(s.executed == 20 && s.dropped_after_close == 0);  // nothing is dropped, only delayed
    CHECK(s.blocked_submits >= 1 && s.blocked_ns > 0);
    CHECK(s.max_queue_depth <= 4);
    pump.stop();
    for (size_t i = 0; i < sh.first_bytes.size(); ++i) CHECK(sh.first_bytes[i] == i);
}

// idle(): false while a batch executes or waits in the queue, true again once the executor is between
// batches (the pipeline prewarm builds only then, so it never queues ahead of the guest).
void test_idle_tracks_queue_and_execution() {
    Shared sh;
    Pump pump;
    CHECK(start(pump, sh));
    CHECK(pump.idle());
    sh.gate = false;
    submit_byte(pump, 1);
    submit_byte(pump, 2);
    CHECK(!pump.idle());  // one queued or executing, the other queued
    std::this_thread::sleep_for(50ms);
    CHECK(!pump.idle());  // stuck inside execute()
    sh.gate = true;
    pump.flush();
    bool idle = false;
    for (int i = 0; i < 200 && !idle; ++i) {
        idle = pump.idle();
        if (!idle) std::this_thread::sleep_for(5ms);
    }
    CHECK(idle);
    pump.stop();
    CHECK(pump.idle());
}

void test_byte_bound_and_oversized_batch() {
    Shared sh;
    sh.gate = false;
    LivePumpOptions opt;
    opt.max_queued_batches = 100;
    opt.max_queued_bytes = 1000;
    Pump pump(opt);
    CHECK(start(pump, sh));
    submit_byte(pump, 0, 10);  // taken by the stuck executor
    CHECK(wait_until([&] { return pump.stats().queued_now == 0; }));
    submit_byte(pump, 1, 5000);  // oversized: admitted into an empty queue, never deadlocks
    std::atomic<bool> done{false};
    std::thread producer([&] {
        submit_byte(pump, 2, 10);  // 5000 queued already: waits
        done = true;
    });
    CHECK(wait_until([&] { return pump.stats().blocked_now == 1; }));
    CHECK(!done.load());
    sh.gate = true;
    producer.join();
    pump.flush();
    CHECK(pump.stats().executed == 3 && pump.stats().blocked_submits == 1);
    pump.stop();
}

void test_close_unblocks_and_requests_shutdown() {
    Shared sh;
    LivePumpOptions opt;
    opt.max_queued_batches = 2;
    opt.idle_poll_ms = 2;
    Pump pump(opt);
    std::atomic<int> closes{0};
    std::thread::id close_thread;
    CHECK(start(pump, sh, [&] {
        ++closes;
        close_thread = std::this_thread::get_id();
    }));
    sh.gate = false;  // executor stuck, producer fills the queue and blocks
    std::atomic<int> accepted{0};
    std::thread producer([&] {
        for (uint8_t i = 0; i < 100; ++i) {
            submit_byte(pump, i);
            ++accepted;
        }
    });
    std::this_thread::sleep_for(100ms);
    CHECK(accepted.load() < 100);
    sh.close = true;  // the user closes the window
    sh.gate = true;
    producer.join();  // must not hang: the dead consumer releases the blocked submitter
    CHECK(pump.window_closed());
    pump.stop();
    CHECK(closes.load() == 1 && close_thread != std::this_thread::get_id());
    const LivePumpStats s = pump.stats();
    CHECK(s.window_closed && s.dropped_after_close + s.executed + s.discarded_on_close == 100);
    submit_byte(pump, 1);  // after close: returns at once, counted
    CHECK(pump.stats().dropped_after_close == s.dropped_after_close + 1);
    CHECK(sh.destroyed.load());
}

void test_focus_forwarding() {
    Shared sh;
    LivePumpOptions opt;
    opt.idle_poll_ms = 2;
    Pump pump(opt);
    std::mutex m;
    std::vector<bool> changes;
    CHECK(start(pump, sh, {}, [&](bool f) {
        std::lock_guard<std::mutex> l(m);
        changes.push_back(f);
    }));
    std::this_thread::sleep_for(30ms);
    sh.focus = true;  // picked up by the idle poll, no batches needed
    std::this_thread::sleep_for(50ms);
    sh.focus = false;
    std::this_thread::sleep_for(50ms);
    pump.stop();
    std::lock_guard<std::mutex> l(m);
    CHECK(changes.size() == 3);  // initial state, gained, lost: reported only on change
    if (changes.size() == 3) CHECK(!changes[0] && changes[1] && !changes[2]);
}

void test_create_failure() {
    Pump pump;
    std::string error;
    CHECK(!pump.start(
        [](std::string* e) -> std::unique_ptr<FakeExecutor> {
            *e = "no adapter";
            return nullptr;
        },
        {}, {}, &error));
    CHECK(error == "no adapter");
    pump.stop();  // harmless on a pump that never ran
}

void test_stop_drains_and_stats_json() {
    Shared sh;
    sh.gate = false;
    Pump pump;
    CHECK(start(pump, sh));
    for (uint8_t i = 0; i < 6; ++i) submit_byte(pump, i);
    std::thread releaser([&] {
        std::this_thread::sleep_for(50ms);
        sh.gate = true;
    });
    pump.stop();  // everything accepted before stop is executed
    releaser.join();
    CHECK(sh.first_bytes.size() == 6);
    const std::string json = pump.stats_json();
    CHECK(json.find("\"submitted\": 6") != std::string::npos);
    CHECK(json.find("\"window_closed\": false") != std::string::npos);
}

void test_tick_hook_runs_on_pump_thread() {
    Shared sh;
    Pump pump;
    std::atomic<int> ticks{0};
    std::thread::id tick_thread;
    std::mutex m;
    pump.set_tick_hook(
        [&](FakeExecutor&) {
            std::lock_guard<std::mutex> lock(m);
            tick_thread = std::this_thread::get_id();
            ++ticks;
        },
        5);
    CHECK(start(pump, sh));
    std::this_thread::sleep_for(100ms);
    pump.stop();
    CHECK(ticks.load() >= 2);
    CHECK(tick_thread == sh.create_thread);
}

// A guest blocked behind a stuck executor (backpressure) must be released by abort(): the --max-ms
// deadline of a live run cannot wait for the executor.
void test_abort_releases_blocked_submit() {
    Shared sh;
    sh.gate = false;
    LivePumpOptions opt;
    opt.max_queued_batches = 2;
    Pump pump(opt);
    CHECK(start(pump, sh));
    std::atomic<int> returned{0};
    std::thread producer([&] {
        for (uint8_t i = 0; i < 10; ++i) {
            submit_byte(pump, i);
            ++returned;
        }
    });
    std::this_thread::sleep_for(100ms);
    CHECK(returned.load() == 3);  // 1 inside execute(), 2 queued, the 4th is blocked
    pump.abort();
    producer.join();              // would hang without abort()
    CHECK(returned.load() == 10);
    const LivePumpStats s = pump.stats();
    CHECK(s.discarded_on_close == 2);
    CHECK(s.dropped_after_close >= 1);
    sh.gate = true;               // the batch inside execute() finishes, then the pump exits
    pump.stop();
    CHECK(sh.destroyed.load());
    CHECK(sh.first_bytes.size() == 1);  // nothing queued ran after the abort
}

// --live-executor: the executor has no window (no window_focused(), poll() never reports a close).
// The pump must run the same path: order, presents counted, no close, on_focus never called, clean stop.
struct HeadlessExecutor {
    explicit HeadlessExecutor(Shared* s) : sh(s) {}
    uint32_t execute(const uint8_t* data, size_t len) {
        std::lock_guard<std::mutex> lock(sh->m);
        sh->first_bytes.push_back(data[0]);
        return len > 1 ? 1u : 0u;
    }
    bool poll() { return true; }
    void wait_idle() { ++sh->waits; }
    Shared* sh;
};

void test_headless_live_executor() {
    Shared sh;
    LivePump<HeadlessExecutor> pump;
    std::atomic<int> closes{0}, focus_calls{0};
    std::string error;
    CHECK(pump.start([&sh](std::string*) { return std::make_unique<HeadlessExecutor>(&sh); }, [&] { ++closes; },
                     [&](bool) { ++focus_calls; }, &error));
    for (uint8_t i = 0; i < 20; ++i) submit_byte(pump, i, (i % 4 == 0) ? 1 : 2);  // 15 of the 20 carry a Present
    pump.flush();
    std::this_thread::sleep_for(50ms);
    CHECK(!pump.window_closed());
    CHECK(closes.load() == 0);
    CHECK(focus_calls.load() == 0);
    pump.stop();
    const LivePumpStats s = pump.stats();
    CHECK(s.executed == 20);
    CHECK(s.presents == 15);
    CHECK(!s.window_closed);
    CHECK(s.discarded_on_close == 0 && s.dropped_after_close == 0);
    CHECK(sh.first_bytes.size() == 20);
    for (size_t i = 0; i < sh.first_bytes.size(); ++i) CHECK(sh.first_bytes[i] == i);
}
}  // namespace

int main() {
    test_order_and_threads();
    test_backpressure_blocks_and_counts();
    test_idle_tracks_queue_and_execution();
    test_byte_bound_and_oversized_batch();
    test_close_unblocks_and_requests_shutdown();
    test_focus_forwarding();
    test_create_failure();
    test_stop_drains_and_stats_json();
    test_tick_hook_runs_on_pump_thread();
    test_abort_releases_blocked_submit();
    test_headless_live_executor();
    if (g_failures == 0) std::printf("test_live_pump: ok\n");
    return g_failures == 0 ? 0 : 1;
}
