#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#ifndef __EMSCRIPTEN__
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif
#endif

#include "abi/playgta6_abi.h"
#include <thread>

#include "gpu/d3d12_guest_objects.h"
#include "gpu/d3d12_xs_exports.h"
#include "gpu/d3d12_xs_xbox.h"
#include "gpu/present_pacing.h"
#include "guest/boot.h"
#include "wasm_boot.h"
#include "guest/com_bridge.h"
#include "guest/minidump.h"
#include "guest/guest_runtime.h"
#include "os/image_loader.h"
#include "os/os_shim.h"
#include "os/ucrt_bridge.h"
#include "os/vmem.h"
#include "os/win32_kernel.h"
#include "os/win32_compat.h"
#include "os/win32_sync.h"
#include "os/xgameruntime_provider.h"

// Boot sequence tests: a synthetic PGIM image (PE headers with a TLS directory, IAT, TLS template)
// and hand-written guest-convention functions standing in for lifted code. Like lifted code they
// keep the guest stack in lockstep (push a return address before a call, pop it on `ret`), but
// they access guest memory through the arena instead of 1:1 pointers.

namespace {

int g_failures = 0;

void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_guest_boot.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}

#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6;
using namespace playgta6::guest;
using playgta6::os::VirtualMemoryArena;

// ---------------------------------------------------------------------------
// Minimal JSON reader (enough to parse the report back)
// ---------------------------------------------------------------------------

struct Json {
    enum Type { Null, Bool, Number, String, Array, Object } type = Null;
    bool boolean = false;
    double number = 0;
    std::string text;
    std::vector<Json> items;
    std::vector<std::pair<std::string, Json>> fields;

    const Json& operator[](const std::string& key) const {
        static const Json kNull;
        for (const auto& [k, v] : fields) {
            if (k == key) {
                return v;
            }
        }
        return kNull;
    }
    const Json& operator[](size_t i) const {
        static const Json kNull;
        return i < items.size() ? items[i] : kNull;
    }
    bool has(const std::string& key) const {
        for (const auto& f : fields) {
            if (f.first == key) {
                return true;
            }
        }
        return false;
    }
    uint64_t hex() const { return type == String ? std::strtoull(text.c_str(), nullptr, 16) : 0; }
};

struct JsonParser {
    const std::string& s;
    size_t i = 0;
    bool ok = true;

    void ws() {
        while (i < s.size() && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) ++i;
    }
    bool eat(char c) {
        ws();
        if (i < s.size() && s[i] == c) {
            ++i;
            return true;
        }
        return false;
    }
    std::string str() {
        std::string out;
        if (!eat('"')) {
            ok = false;
            return out;
        }
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\' && i + 1 < s.size()) {
                ++i;
                const char e = s[i];
                if (e == 'n') out += '\n';
                else if (e == 't') out += '\t';
                else if (e == 'r') out += '\r';
                else if (e == 'u') {
                    out += static_cast<char>(std::strtol(s.substr(i + 1, 4).c_str(), nullptr, 16));
                    i += 4;
                } else out += e;
                ++i;
            } else {
                out += s[i++];
            }
        }
        if (i >= s.size()) ok = false;
        ++i;
        return out;
    }
    Json value() {
        Json v;
        ws();
        if (i >= s.size()) {
            ok = false;
            return v;
        }
        const char c = s[i];
        if (c == '{') {
            ++i;
            v.type = Json::Object;
            if (eat('}')) return v;
            do {
                std::string key = str();
                if (!eat(':')) ok = false;
                v.fields.emplace_back(key, value());
            } while (ok && eat(','));
            if (!eat('}')) ok = false;
        } else if (c == '[') {
            ++i;
            v.type = Json::Array;
            if (eat(']')) return v;
            do {
                v.items.push_back(value());
            } while (ok && eat(','));
            if (!eat(']')) ok = false;
        } else if (c == '"') {
            v.type = Json::String;
            v.text = str();
        } else if (s.compare(i, 4, "null") == 0) {
            i += 4;
        } else if (s.compare(i, 4, "true") == 0) {
            v.type = Json::Bool;
            v.boolean = true;
            i += 4;
        } else if (s.compare(i, 5, "false") == 0) {
            v.type = Json::Bool;
            i += 5;
        } else {
            v.type = Json::Number;
            char* end = nullptr;
            v.number = std::strtod(s.c_str() + i, &end);
            if (end == s.c_str() + i) ok = false;
            i = static_cast<size_t>(end - s.c_str());
        }
        return v;
    }
};

Json parse_json(const std::string& text, bool* ok) {
    JsonParser p{text};
    Json v = p.value();
    p.ws();
    *ok = p.ok && p.i == text.size();
    return v;
}

// ---------------------------------------------------------------------------
// Synthetic image
// ---------------------------------------------------------------------------

constexpr uint64_t kBase = 0x140000000ULL;
constexpr uint32_t kSizeOfImage = 0x6000;
constexpr uint32_t kEntryRva = 0x1000;
constexpr uint64_t kEntry = kBase + kEntryRva;
constexpr uint64_t kMid = kBase + 0x1100;
constexpr uint64_t kLeaf = kBase + 0x1180;
constexpr uint64_t kCallback1 = kBase + 0x1200;
constexpr uint64_t kCallback2 = kBase + 0x1240;
constexpr uint64_t kWorker = kBase + 0x1300;
// Return addresses inside the functions above, each preceded by a call instruction in .text.
constexpr uint64_t kEntryCallsMid = kBase + 0x1010;   // after `call rel32` at 0x100B
constexpr uint64_t kMidCallsLeaf = kBase + 0x1120;    // after `call [rip+disp]` at 0x111A
constexpr uint64_t kEntryCallsIat = kBase + 0x1030;   // after `call [rip+disp]` at 0x102A
constexpr uint64_t kEntryStaleValue = kBase + 0x1050;  // inside entry, NOT after a call
constexpr uint64_t kUnknownInsideEntry = kBase + 0x1234 - 0x200;  // 0x1034: not an entry

constexpr uint32_t kTlsDirRva = 0x2000;
constexpr uint32_t kCallbacksRva = 0x2040;
constexpr uint32_t kIatRva = 0x3000;
constexpr uint32_t kTlsIndexRva = 0x3100;
constexpr uint32_t kTlsTemplateRva = 0x4000;
constexpr uint32_t kTlsTemplateSize = 0x40;
constexpr uint32_t kTlsZeroFill = 0x30;
constexpr uint32_t kLfanew = 0x80;
constexpr uint64_t kPeStackReserve = 0x40000;

// IAT slots.
constexpr uint32_t kSlotExit = kIatRva + 0x00;
constexpr uint32_t kSlotSetLastError = kIatRva + 0x08;
constexpr uint32_t kSlotGetLastError = kIatRva + 0x10;
constexpr uint32_t kSlotMissingOne = kIatRva + 0x18;
constexpr uint32_t kSlotMissingTwo = kIatRva + 0x20;
constexpr uint32_t kSlotAbort = kIatRva + 0x28;
constexpr uint32_t kSlotTerminate = kIatRva + 0x30;
constexpr uint32_t kSlotCaptureContext = kIatRva + 0x38;
constexpr uint32_t kPdataRva = 0x2080;
constexpr uint32_t kLeafInfoRva = 0x20B0;
constexpr uint32_t kAlloc8InfoRva = 0x20B4;

void put32(std::vector<uint8_t>& out, uint32_t v) {
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<uint8_t>(v >> (8 * i)));
}
void put64(std::vector<uint8_t>& out, uint64_t v) {
    for (int i = 0; i < 8; ++i) out.push_back(static_cast<uint8_t>(v >> (8 * i)));
}
void set16(std::vector<uint8_t>& b, size_t at, uint16_t v) {
    b[at] = static_cast<uint8_t>(v);
    b[at + 1] = static_cast<uint8_t>(v >> 8);
}
void set32(std::vector<uint8_t>& b, size_t at, uint32_t v) {
    for (int i = 0; i < 4; ++i) b[at + i] = static_cast<uint8_t>(v >> (8 * i));
}
void set64(std::vector<uint8_t>& b, size_t at, uint64_t v) {
    for (int i = 0; i < 8; ++i) b[at + i] = static_cast<uint8_t>(v >> (8 * i));
}

struct Sec {
    std::string name;
    uint32_t rva;
    uint32_t vsize;
    uint32_t prot;
    uint32_t flags;
    std::vector<uint8_t> data;
};

std::vector<uint8_t> write_container(const std::vector<Sec>& secs, uint32_t flags, uint64_t base) {
    const uint64_t tables = 64 + 64 * secs.size();
    std::vector<uint64_t> offsets;
    uint64_t cursor = tables;
    for (const Sec& s : secs) {
        if (s.data.empty()) {
            offsets.push_back(0);
            continue;
        }
        const uint64_t off = (cursor + 4095) / 4096 * 4096;
        offsets.push_back(off);
        cursor = off + s.data.size();
    }
    std::vector<uint8_t> out = {'P', 'G', 'I', 'M'};
    put32(out, 1);
    put32(out, 64);
    put32(out, 64);
    put32(out, flags);
    put32(out, 4096);
    put64(out, base);
    put32(out, kSizeOfImage);
    put32(out, 0x1000);
    put32(out, kEntryRva);
    put32(out, static_cast<uint32_t>(secs.size()));
    put32(out, 0);
    put32(out, 0);
    put64(out, cursor);
    for (size_t i = 0; i < secs.size(); ++i) {
        std::string name = secs[i].name;
        name.resize(16, '\0');
        out.insert(out.end(), name.begin(), name.end());
        put32(out, secs[i].rva);
        put32(out, secs[i].vsize);
        put32(out, secs[i].prot);
        put32(out, secs[i].flags);
        put64(out, offsets[i]);
        put64(out, secs[i].data.size());
        put64(out, os::pgim::fnv1a_64(secs[i].data.data(), secs[i].data.size()));
        put32(out, 0);
        put32(out, 0);
    }
    for (size_t i = 0; i < secs.size(); ++i) {
        if (!secs[i].data.empty()) {
            out.resize(static_cast<size_t>(offsets[i]), 0);
            out.insert(out.end(), secs[i].data.begin(), secs[i].data.end());
        }
    }
    return out;
}

uint8_t template_byte(size_t i) { return static_cast<uint8_t>(0xA0 + i); }

// with_tls = false drops the TLS data directory.
std::vector<uint8_t> build_image(bool with_tls = true, uint64_t base = kBase,
                                 std::vector<uint64_t> callbacks = {kCallback1, kCallback2},
                                 uint8_t seed = 0xA0, bool with_pdata = true) {
    using namespace os::pgim;
    // PE headers: DOS stub + PE32+ optional header with 16 data directories.
    std::vector<uint8_t> headers(0x400, 0);
    set16(headers, 0, 0x5A4D);
    set32(headers, 0x3C, kLfanew);
    set32(headers, kLfanew, 0x00004550);
    set16(headers, kLfanew + 4, 0x8664);  // machine
    set16(headers, kLfanew + 20, 240);    // SizeOfOptionalHeader
    const size_t opt = kLfanew + 24;
    set16(headers, opt, 0x20B);
    set32(headers, opt + 16, kEntryRva);
    set64(headers, opt + 24, base);
    set16(headers, opt + 68, 2);  // IMAGE_SUBSYSTEM_WINDOWS_GUI
    set64(headers, opt + 72, kPeStackReserve);
    set32(headers, opt + 108, 16);
    set32(headers, opt + 56, kSizeOfImage);
    if (with_pdata) {
        set32(headers, opt + 112 + 8 * 3, kPdataRva);
        set32(headers, opt + 112 + 8 * 3 + 4, 36);
    }
    if (with_tls) {
        set32(headers, opt + 112 + 8 * 9, kTlsDirRva);
        set32(headers, opt + 112 + 8 * 9 + 4, 40);
    }

    // .text: int3 filler with call instructions right before each return address used below.
    std::vector<uint8_t> text(0x400, 0xCC);
    auto at = [](uint64_t va) { return static_cast<size_t>(va - kBase - 0x1000); };
    text[at(kEntryCallsMid) - 5] = 0xE8;  // call rel32
    text[at(kMidCallsLeaf) - 6] = 0xFF;   // call [rip+disp32]
    text[at(kMidCallsLeaf) - 5] = 0x15;
    text[at(kEntryCallsIat) - 6] = 0xFF;
    text[at(kEntryCallsIat) - 5] = 0x15;

    // .rdata: TLS directory + callback array.
    std::vector<uint8_t> rdata(0x100, 0);
    set64(rdata, 0x00, base + kTlsTemplateRva);
    set64(rdata, 0x08, base + kTlsTemplateRva + kTlsTemplateSize);
    set64(rdata, 0x10, base + kTlsIndexRva);
    set64(rdata, 0x18, base + kCallbacksRva);
    if (with_pdata) {
        // .pdata (RUNTIME_FUNCTION begin, end, unwind info) + UNWIND_INFO: entry and mid allocate
        // 8 bytes (UWOP_ALLOC_SMALL at prolog offset 1), leaf has no unwind codes.
        const uint32_t pdata[3][3] = {{0x1000, 0x1100, kAlloc8InfoRva},
                                      {0x1100, 0x1180, kAlloc8InfoRva},
                                      {0x1180, 0x11C0, kLeafInfoRva}};
        for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) set32(rdata, kPdataRva - 0x2000 + 12 * i + 4 * j, pdata[i][j]);
        }
        set32(rdata, kLeafInfoRva - 0x2000, 0x00000001u);   // version 1, no codes
        set32(rdata, kAlloc8InfoRva - 0x2000, 0x00010101u); // version 1, prolog 1, 1 code
        set16(rdata, kAlloc8InfoRva - 0x2000 + 4, 0x0201);  // offset 1: UWOP_ALLOC_SMALL 8
    }
    set32(rdata, 0x20, kTlsZeroFill);
    set32(rdata, 0x24, 0x00500000);  // IMAGE_SCN_ALIGN_16BYTES
    for (size_t i = 0; i < callbacks.size(); ++i) set64(rdata, kCallbacksRva - kTlsDirRva + 8 * i, callbacks[i]);

    // .data: IAT (filled by the binder) and the TLS index (0xFFFFFFFF until the loader writes 0).
    std::vector<uint8_t> data(0x200, 0);
    set32(data, kTlsIndexRva - kIatRva, 0xFFFFFFFFu);

    // .tls: template; the bytes after it in the section are garbage that must not be copied.
    std::vector<uint8_t> tls(0x100, 0xEE);
    for (size_t i = 0; i < kTlsTemplateSize; ++i) tls[i] = static_cast<uint8_t>(seed + i);

    std::vector<Sec> secs = {
        {"<headers>", 0, 0x400, kProtRead, kSectionFlagHeaders, headers},
        {".text", 0x1000, 0x400, kProtRead | kProtExecute, 0, text},
        {".rdata", 0x2000, 0x100, kProtRead, 0, rdata},
        {".data", 0x3000, 0x200, kProtRead | kProtWrite, 0, data},
        {".tls", 0x4000, 0x100, kProtRead | kProtWrite, 0, tls},
    };
    return write_container(secs, kFlagHeadersMapped, base);
}

// ---------------------------------------------------------------------------
// Guest-convention functions
// ---------------------------------------------------------------------------

VirtualMemoryArena* g_arena = nullptr;

// A COM interface whose slot 7 has no implementation (policy Trap): calling it is an
// unimplemented-slot fault, named in the fault record.
struct IFaultProbeHost {
    virtual int32_t Known() = 0;
};
constexpr com::Guid kIidFaultProbe = com::make_guid("5a1f0c11-7436-11ce-8034-00aa006009fa");
using FaultProbeM = com::Methods<IFaultProbeHost>;
constexpr com::MethodDesc kFaultProbeMethods[] = {FaultProbeM::method<&IFaultProbeHost::Known>("Known")};
constexpr auto kIFaultProbe = com::declare_interface<IFaultProbeHost>("IFaultProbe", kIidFaultProbe,
                                                                      kFaultProbeMethods, {.slot_count = 12});

enum class Scenario {
    ReturnCode,
    ExitViaCrt,
    RequestExit,
    Trap,
    Unimplemented,
    DivideError,
    UnknownTarget,
    StackImbalance,
    BadStack,
    Unresolved,
    SurveyLoop,
    SurveyReturn,
    CrtAbort,
    LastError,
    PresentMark,
    TerminateViaShim,
    OffThreadFault,
    HostCallable,
    OffThreadUnresolved,
    O42PostPresentBatch,
    O42PostPresentFault,
    O42PostPresentConcurrent,
    DeepTrapWithRegs,
    OffThreadTrapWithRegs,
    OffThreadNullIndirect,
    OffThreadUnknownTarget,
    OffThreadComSlot,
};

// HostCallable: the entry calls a named host callable (like a COM method) twice.
void host_answer(void* raw) {
    auto* s = static_cast<GuestState*>(raw);
    s->gpr[GuestReg::Rax] = 42;
    s->gpr[GuestReg::Rsp] += 8;  // pops the return address
}

// OffThreadFault: a host thread that runs guest code faults while the boot thread waits.
std::atomic<bool> g_off_thread_reported{false};
std::string g_off_thread_report;

// O42 test observations captured mid-run.
struct O42Observed {
    uint64_t mid_dispatches = 0;
    uint32_t main_tid = 0;
    std::string live_last_import;
    std::vector<uint32_t> worker_tids;
} g_o42;

Scenario g_scenario = Scenario::ReturnCode;
std::vector<std::string> g_events;

struct Observed {
    uint64_t teb_self = 0, stack_base = 0, stack_limit = 0, pid = 0, tid = 0;
    uint64_t tls_pointer = 0, peb = 0, peb_image_base = 0, peb_params = 0;
    uint64_t tls_block = 0;
    std::vector<uint8_t> tls_bytes;
    uint32_t tls_index = 0;
    uint64_t callback_args[2][3] = {};
    uint32_t mxcsr = 0;
    uint64_t limit_low = 0, limit_high = 0;
    std::string command_line_a;
    std::u16string command_line;
} g_seen;

uint64_t load64(uint64_t va) {
    uint64_t v = 0;
    CHECK(g_arena->read(va, &v, sizeof(v)));
    return v;
}
uint32_t load32(uint64_t va) {
    uint32_t v = 0;
    CHECK(g_arena->read(va, &v, sizeof(v)));
    return v;
}
void store64(uint64_t va, uint64_t v) { CHECK(g_arena->write(va, &v, sizeof(v))); }
void store32(uint64_t va, uint32_t v) { CHECK(g_arena->write(va, &v, sizeof(v))); }

uint64_t& reg(GuestState* s, uint32_t r) { return s->gpr[r]; }
void push(GuestState* s, uint64_t v) {
    reg(s, GuestReg::Rsp) -= 8;
    store64(reg(s, GuestReg::Rsp), v);
}
void ret(GuestState* s) { reg(s, GuestReg::Rsp) += 8; }
// `call [rip + iat slot]` from `return_address - 6`.
void call_iat(GuestState* s, uint32_t slot_rva, uint64_t return_address) {
    push(s, return_address);
    __playgta6_indirect_call(s, load64(kBase + slot_rva));
}

// Inline TLS access as MSVC emits it: mov rax, gs:[0x58]; mov ecx, [index]; mov rax, [rax+rcx*8].
uint64_t tls_block_of(GuestState* s) {
    const uint64_t vector = load64(s->gs_base + 0x58);
    const uint32_t index = load32(kBase + kTlsIndexRva);
    return load64(vector + 8ull * index);
}

void sub_leaf(GuestState* s) {
    __playgta6_trap(s, kLeaf + 4);
}

void sub_mid(GuestState* s) {
    push(s, 0x1234);  // a saved register that is not a return address
    push(s, kMidCallsLeaf);
    sub_leaf(s);
    ret(s);
    reg(s, GuestReg::Rsp) += 8;
    ret(s);
}

void sub_callback(GuestState* s, int which) {
    g_events.push_back(which == 0 ? "callback1" : "callback2");
    g_seen.callback_args[which][0] = reg(s, GuestReg::Rcx);
    g_seen.callback_args[which][1] = reg(s, GuestReg::Rdx);
    g_seen.callback_args[which][2] = reg(s, GuestReg::R8);
    // TLS must already be usable in callbacks.
    if (which == 0) {
        const uint64_t block = tls_block_of(s);
        g_events.push_back(load32(block) == 0xA3A2A1A0u ? "callback1-tls-ok" : "callback1-tls-bad");
    }
    ret(s);
}
void sub_callback1(GuestState* s) { sub_callback(s, 0); }
void sub_callback2(GuestState* s) { sub_callback(s, 1); }

void sub_worker(GuestState* s) {
    // Second-thread body: report its TLS block through rax.
    reg(s, GuestReg::Rax) = tls_block_of(s);
    ret(s);
}

void sub_entry(GuestState* s) {
    g_events.push_back("entry");
    switch (g_scenario) {
        case Scenario::ReturnCode: {
            g_seen.teb_self = load64(s->gs_base + 0x30);
            g_seen.stack_base = load64(s->gs_base + 0x08);
            g_seen.stack_limit = load64(s->gs_base + 0x10);
            g_seen.pid = load64(s->gs_base + 0x40);
            g_seen.tid = load64(s->gs_base + 0x48);
            g_seen.tls_pointer = load64(s->gs_base + 0x58);
            g_seen.peb = load64(s->gs_base + 0x60);
            g_seen.peb_image_base = load64(g_seen.peb + 0x10);
            g_seen.peb_params = load64(g_seen.peb + 0x20);
            g_seen.tls_index = load32(kBase + kTlsIndexRva);
            if (g_seen.tls_pointer != 0) {
                g_seen.tls_block = tls_block_of(s);
                g_seen.tls_bytes.resize(kTlsTemplateSize + kTlsZeroFill);
                CHECK(g_arena->read(g_seen.tls_block, g_seen.tls_bytes.data(), g_seen.tls_bytes.size()));
            }
            g_seen.mxcsr = s->mxcsr;
            pg6_win32_get_current_thread_stack_limits(&g_seen.limit_low, &g_seen.limit_high);
            g_seen.command_line_a = pg6_win32_get_command_line_a();
            // RTL_USER_PROCESS_PARAMETERS.CommandLine
            const uint16_t length = static_cast<uint16_t>(load32(g_seen.peb_params + 0x70));
            const uint64_t buffer = load64(g_seen.peb_params + 0x78);
            g_seen.command_line.resize(length / 2);
            CHECK(g_arena->read(buffer, g_seen.command_line.data(), length));
            reg(s, GuestReg::Rax) = 0xFFFFFFFF00000000ull | 42;  // exit code = eax
            ret(s);
            return;
        }
        case Scenario::ExitViaCrt:
            reg(s, GuestReg::Rcx) = 7;
            call_iat(s, kSlotExit, kEntryCallsIat);
            g_events.push_back("after-exit");  // must not run
            ret(s);
            return;
        case Scenario::RequestExit:
            pg6_guest_boot_request_exit(9);
            g_events.push_back("after-request-exit");
            ret(s);
            return;
        case Scenario::Trap:
            push(s, kEntryStaleValue);  // stale non-return address
            push(s, kEntryCallsMid);
            sub_mid(s);
            ret(s);
            return;
        case Scenario::Unimplemented:
            __playgta6_unimplemented(s, kEntry + 0x20);
        case Scenario::DivideError:
            __playgta6_divide_error(s, kEntry + 0x24);
        case Scenario::UnknownTarget:
            push(s, kEntryCallsIat);
            __playgta6_indirect_call(s, kUnknownInsideEntry);
            ret(s);
            return;
        case Scenario::StackImbalance:
            reg(s, GuestReg::Rax) = 1;
            return;  // forgets `ret`: rsp is 8 bytes too low
        case Scenario::BadStack:
            reg(s, GuestReg::Rsp) = 0x20;
            call_guest(*g_arena, *s, kMid);
            return;
        case Scenario::Unresolved:
            call_iat(s, kSlotMissingOne, kEntryCallsIat);
            ret(s);
            return;
        case Scenario::SurveyLoop:
            for (int i = 0; i < 1000; ++i) {
                call_iat(s, kSlotMissingTwo, kEntryCallsIat);
                call_iat(s, kSlotMissingOne, kEntryCallsIat);
                CHECK(reg(s, GuestReg::Rax) == 0);
            }
            ret(s);
            return;
        case Scenario::SurveyReturn:
            reg(s, GuestReg::Rax) = 0x55;
            call_iat(s, kSlotMissingTwo, kEntryCallsIat);
            CHECK(reg(s, GuestReg::Rax) == 0);
            call_iat(s, kSlotMissingOne, kEntryCallsIat);
            call_iat(s, kSlotMissingTwo, kEntryCallsIat);
            reg(s, GuestReg::Rax) = 5;
            ret(s);
            return;
        case Scenario::TerminateViaShim:
            reg(s, GuestReg::Rcx) = ~uint64_t{0};  // GetCurrentProcess()
            reg(s, GuestReg::Rdx) = 11;
            call_iat(s, kSlotTerminate, kEntryCallsIat);
            g_events.push_back("after-terminate");
            ret(s);
            return;
        case Scenario::CrtAbort:
            call_iat(s, kSlotAbort, kEntryCallsIat);
            ret(s);
            return;
        case Scenario::LastError: {
            reg(s, GuestReg::Rcx) = 1234;
            call_iat(s, kSlotSetLastError, kEntryCallsIat);
            const uint32_t from_teb = load32(s->gs_base + 0x68);  // inline gs:[0x68] read
            store32(s->gs_base + 0x68, 77);                        // inline write
            call_iat(s, kSlotGetLastError, kEntryCallsIat);
            reg(s, GuestReg::Rax) = (reg(s, GuestReg::Rax) == 77 && from_teb == 1234) ? 0 : 1;
            ret(s);
            return;
        }
        case Scenario::PresentMark: {
            // Two import calls, the first Present, one more call.
            call_iat(s, kSlotSetLastError, kEntryCallsIat);
            call_iat(s, kSlotGetLastError, kEntryCallsIat);
            // A PresentX reaches the display (present pacer) without any GPU capture running:
            // that alone marks the first Present.
            gpu::present_pacer().present(0);
            gpu::present_pacer().present(0);
            call_iat(s, kSlotSetLastError, kEntryCallsIat);
            mark_first_present();  // later marks are ignored
            reg(s, GuestReg::Rax) = 0;
            ret(s);
            return;
        }
        case Scenario::HostCallable: {
            const uint64_t answer = register_host_callable(&host_answer, "ITest::Answer");
            for (int i = 0; i < 2; ++i) {
                push(s, kEntryCallsIat);
                __playgta6_indirect_call(s, answer);
            }
            CHECK(reg(s, GuestReg::Rax) == 42);
            call_iat(s, kSlotGetLastError, kEntryCallsIat);
            __playgta6_unimplemented(s, kEntry + 0x20);  // a fault, so the report lists the dispatches
        }
        case Scenario::OffThreadUnresolved: {
            // A worker calls an import that did not resolve (parked by the hook, detached).
            std::thread([] {
                GuestState& state = com::current_guest_state();
                call_iat(&state, kSlotMissingOne, kWorker + 4);
            }).detach();
            for (int i = 0; i < 5000 && !g_off_thread_reported.load(); ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            reg(s, GuestReg::Rax) = 0;
            ret(s);
            return;
        }
        case Scenario::OffThreadFault: {
            // The worker never returns from the fault (the hook parks it); it is detached.
            std::thread([] {
                GuestState& state = com::current_guest_state();
                __playgta6_unimplemented(&state, kWorker + 4);
            }).detach();
            for (int i = 0; i < 5000 && !g_off_thread_reported.load(); ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            reg(s, GuestReg::Rax) = 0;
            ret(s);
            return;
        }
        case Scenario::OffThreadNullIndirect:
        case Scenario::OffThreadUnknownTarget:
        case Scenario::OffThreadComSlot: {
            // A worker's indirect call that cannot be dispatched (the browser run's INDIRECT_CALL_MISS at
            // 0x0 was one of these on a non-boot thread). The hook parks it; the worker is detached.
            static uint64_t target = 0;
            target = g_scenario == Scenario::OffThreadNullIndirect      ? 0
                     : g_scenario == Scenario::OffThreadUnknownTarget   ? kUnknownInsideEntry
                                                                        : com::slot_address(kIFaultProbe.info, 7);
            std::thread([] {
                GuestState& state = com::current_guest_state();
                push(&state, kWorker + 4);
                __playgta6_indirect_call(&state, target);
            }).detach();
            for (int i = 0; i < 5000 && !g_off_thread_reported.load(); ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            reg(s, GuestReg::Rax) = 0;
            ret(s);
            return;
        }
        case Scenario::O42PostPresentBatch: {
            g_o42.main_tid = os::win32::GetCurrentThreadId();
            // 3 pre-present calls (SetLastError x2, GetLastError x1).
            reg(s, GuestReg::Rcx) = 0x11;
            call_iat(s, kSlotSetLastError, kEntryCallsIat);
            reg(s, GuestReg::Rcx) = 0x22;
            call_iat(s, kSlotSetLastError, kEntryCallsIat);
            call_iat(s, kSlotGetLastError, kEntryCallsIat);
            // Mid-run live_counters() must flush the calling thread's pending batch (< 256).
            g_o42.mid_dispatches = live_counters().dispatches;
            mark_first_present();
            // 600 post-present calls (spans 2 full 256-call batches + 88 remainder).
            for (uint32_t i = 0; i < 600; ++i) {
                if ((i & 1u) == 0) {
                    reg(s, GuestReg::Rcx) = 0x1000u + i;
                    call_iat(s, kSlotSetLastError, kEntryCallsIat);
                } else {
                    call_iat(s, kSlotGetLastError, kEntryCallsIat);
                }
            }
            g_o42.live_last_import = guest_thread_last_import(g_o42.main_tid);
            reg(s, GuestReg::Rax) = 0;
            ret(s);
            return;
        }
        case Scenario::O42PostPresentFault: {
            g_o42.main_tid = os::win32::GetCurrentThreadId();
            reg(s, GuestReg::Rcx) = 0xAA;
            call_iat(s, kSlotSetLastError, kEntryCallsIat);
            mark_first_present();
            reg(s, GuestReg::Rcx) = 0xBB;
            call_iat(s, kSlotSetLastError, kEntryCallsIat);
            call_iat(s, kSlotGetLastError, kEntryCallsIat);
            // Fault with pending dispatches < 256: boot_fault_handler must flush & preserve post-present state.
            __playgta6_unimplemented(s, kEntry + 0x20);
        }
        case Scenario::O42PostPresentConcurrent: {
            g_o42.main_tid = os::win32::GetCurrentThreadId();
            // 2 pre-present import calls.
            reg(s, GuestReg::Rcx) = 0x5;
            call_iat(s, kSlotSetLastError, kEntryCallsIat);
            call_iat(s, kSlotGetLastError, kEntryCallsIat);
            mark_first_present();

            constexpr int kWorkers = 4;
            constexpr uint32_t kCallsPerWorker = 300;
            g_o42.worker_tids.assign(kWorkers, 0);
            std::atomic<int> ready{0};
            std::atomic<bool> go{false};
            std::vector<std::thread> workers;
            workers.reserve(kWorkers);
            for (int w = 0; w < kWorkers; ++w) {
                workers.emplace_back([w, &ready, &go] {
                    GuestState& ws = com::current_guest_state();
                    g_o42.worker_tids[static_cast<size_t>(w)] = os::win32::GetCurrentThreadId();
                    ready.fetch_add(1, std::memory_order_release);
                    while (!go.load(std::memory_order_acquire)) {
                        std::this_thread::yield();
                    }
                    for (uint32_t i = 0; i < kCallsPerWorker; ++i) {
                        const bool is_last = (i + 1 == kCallsPerWorker);
                        const bool call_get = is_last ? ((w & 1) == 0) : ((i & 1u) != 0);
                        if (call_get) {
                            call_iat(&ws, kSlotGetLastError, kEntryCallsIat);
                        } else {
                            reg(&ws, GuestReg::Rcx) = (static_cast<uint64_t>(w) << 16) | i;
                            call_iat(&ws, kSlotSetLastError, kEntryCallsIat);
                        }
                    }
                });
            }
            while (ready.load(std::memory_order_acquire) < kWorkers) {
                std::this_thread::yield();
            }
            go.store(true, std::memory_order_release);
            // Main thread also runs 100 concurrent post-present calls, ending with SetLastError.
            for (uint32_t i = 0; i < 100; ++i) {
                if (i == 99 || (i & 1u) == 0) {
                    reg(s, GuestReg::Rcx) = 0x9000u + i;
                    call_iat(s, kSlotSetLastError, kEntryCallsIat);
                } else {
                    call_iat(s, kSlotGetLastError, kEntryCallsIat);
                }
            }
            for (std::thread& t : workers) {
                t.join();
            }
            reg(s, GuestReg::Rax) = 0;
            ret(s);
            return;
        }
        case Scenario::DeepTrapWithRegs: {
            // 5 verified call-site return addresses (frame[4] is distinct to check the 4-frame cap).
            push(s, kEntryCallsIat);   // frame[4]: sub_140001000+0x30 (must be omitted from callers=)
            push(s, kEntryCallsMid);   // frame[3]: sub_140001000+0x10
            push(s, kMidCallsLeaf);    // frame[2]: sub_140001100+0x20
            push(s, kEntryCallsMid);   // frame[1]: sub_140001000+0x10
            push(s, kMidCallsLeaf);    // frame[0]: sub_140001100+0x20
            reg(s, GuestReg::Rax) = 0x1111222233334444ULL;
            reg(s, GuestReg::Rcx) = 0x2222333344445555ULL;
            reg(s, GuestReg::Rdx) = 0x3333444455556666ULL;
            reg(s, GuestReg::R8) = 0x88889999aaaabbbbULL;
            reg(s, GuestReg::R9) = 0x9999aaaabbbbccccULL;
            reg(s, GuestReg::Rbx) = 0xbbbbccccddddeeeeULL;
            reg(s, GuestReg::Rsi) = 0x5151515151515151ULL;
            reg(s, GuestReg::Rdi) = 0xd1d1d1d1d1d1d1d1ULL;
            __playgta6_trap(s, kLeaf + 4);
        }
        case Scenario::OffThreadTrapWithRegs: {
            std::thread([] {
                GuestState& state = com::current_guest_state();
                push(&state, kEntryCallsMid);  // frame[1]
                push(&state, kMidCallsLeaf);   // frame[0]
                reg(&state, GuestReg::Rax) = 0x1010101010101010ULL;
                reg(&state, GuestReg::Rcx) = 0x2020202020202020ULL;
                reg(&state, GuestReg::Rdx) = 0x3030303030303030ULL;
                reg(&state, GuestReg::R8) = 0x8080808080808080ULL;
                reg(&state, GuestReg::R9) = 0x9090909090909090ULL;
                reg(&state, GuestReg::Rbx) = 0xb0b0b0b0b0b0b0b0ULL;
                reg(&state, GuestReg::Rsi) = 0x5050505050505050ULL;
                reg(&state, GuestReg::Rdi) = 0xd0d0d0d0d0d0d0d0ULL;
                __playgta6_trap(&state, kLeaf + 4);
            }).detach();
            for (int i = 0; i < 5000 && !g_off_thread_reported.load(); ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            reg(s, GuestReg::Rax) = 0;
            ret(s);
            return;
        }
    }
}

// Import thunks (R5 shape: called after the guest pushed its return address).
void thunk_exit(void* raw) {
    auto* s = static_cast<GuestState*>(raw);
    pg6_crt_exit(static_cast<int32_t>(reg(s, GuestReg::Rcx)));
    ret(s);
}
void thunk_terminate(void* raw) {
    auto* s = static_cast<GuestState*>(raw);
    reg(s, GuestReg::Rax) = static_cast<uint64_t>(
        pg6_win32_terminate_process(reg(s, GuestReg::Rcx), static_cast<uint32_t>(reg(s, GuestReg::Rdx))));
    ret(s);
}
void thunk_abort(void* raw) {
    auto* s = static_cast<GuestState*>(raw);
    pg6_crt_abort();
    ret(s);
}
void thunk_set_last_error(void* raw) {
    auto* s = static_cast<GuestState*>(raw);
    os::win32::SetLastError(static_cast<uint32_t>(reg(s, GuestReg::Rcx)));
    ret(s);
}
void thunk_get_last_error(void* raw) {
    auto* s = static_cast<GuestState*>(raw);
    reg(s, GuestReg::Rax) = os::win32::GetLastError();
    ret(s);
}
void thunk_generated_unresolved(void*) {
    std::fprintf(stderr, "generated unresolved thunk must be replaced by the boot\n");
    std::abort();
}

const LiftedFunctionEntry kTable[] = {
    {kEntry, &sub_entry},         {kMid, &sub_mid},       {kLeaf, &sub_leaf},
    {kCallback1, &sub_callback1}, {kCallback2, &sub_callback2}, {kWorker, &sub_worker},
};

const BootImport kImports[] = {
    {kSlotExit, kPg6BootImportBound, 0, "api-ms-win-crt-runtime-l1-1-0.dll", "exit", "pg6_crt_exit", &thunk_exit},
    {kSlotSetLastError, kPg6BootImportBound, 0, "KERNEL32.dll", "SetLastError", "pg6_win32_set_last_error",
     &thunk_set_last_error},
    {kSlotGetLastError, kPg6BootImportBound, 0, "KERNEL32.dll", "GetLastError", "pg6_win32_get_last_error",
     &thunk_get_last_error},
    {kSlotMissingOne, 0, 0, "KERNEL32.dll", "MissingOne", nullptr, &thunk_generated_unresolved},
    {kSlotMissingTwo, kPg6BootImportDelayLoad, 0, "USER32.dll", "MissingTwo", nullptr,
     &thunk_generated_unresolved},
    {kSlotAbort, kPg6BootImportBound, 0, "api-ms-win-crt-runtime-l1-1-0.dll", "abort", "pg6_crt_abort",
     &thunk_abort},
    {kSlotTerminate, kPg6BootImportBound, 0, "KERNEL32.dll", "TerminateProcess", "pg6_win32_terminate_process",
     &thunk_terminate},
    // Left unresolved by the import bridge; the boot binds it to R11's runtime thunk.
    {kSlotCaptureContext, 0, 0, "KERNEL32.dll", "RtlCaptureContext", nullptr, &thunk_generated_unresolved},
};

std::vector<uint8_t> g_image;

BootConfig make_config(os::pgim::ImageReader& reader) {
    BootConfig cfg;
    BootModule exe;
    exe.image = &reader;
    exe.name = "boot_test.exe";
    exe.lifted_tables.push_back(std::span<const LiftedFunctionEntry>(kTable));
    exe.imports = std::span<const BootImport>(kImports);
    cfg.modules.push_back(std::move(exe));
    cfg.arena = g_arena;
    cfg.command_line = "boot_test.exe -flag \xC3\xA4";  // non-ASCII survives as UTF-16
    cfg.working_directory = "C:\\game";
    cfg.log_progress = false;
    return cfg;
}

struct Run {
    BootResult result;
    Json report;
};

Run boot_scenario(Scenario scenario, void (*tweak)(BootConfig&) = nullptr) {
    g_scenario = scenario;
    g_events.clear();
    g_seen = Observed{};
    os::pgim::MemoryImageReader reader(g_image.data(), g_image.size());
    BootConfig cfg = make_config(reader);
    if (tweak != nullptr) {
        tweak(cfg);
    }
    Run run;
    run.result = run_boot(cfg);
    bool ok = false;
    run.report = parse_json(run.result.report_json, &ok);
    CHECK(ok);
    CHECK(run.report["schema"].text == "playgta6-boot-report");
    CHECK(run.report["schema_version"].number == 2);
    CHECK(run.report["status"].text == boot_status_name(run.result.status));
    CHECK(!run.result.summary.empty());
    if (!ok) {
        std::fprintf(stderr, "%s\n", run.result.report_json.c_str());
    }
    return run;
}

void check_fault(const Run& run, const char* kind, uint64_t va, uint64_t function_entry) {
    const Json& fault = run.report["fault"];
    CHECK(fault.type == Json::Object);
    CHECK(fault["kind"].text == kind);
    CHECK(fault["va"].hex() == va);
    if (function_entry != 0) {
        CHECK(fault["function"]["entry"].hex() == function_entry);
        CHECK(fault["function"]["offset"].hex() == va - function_entry);
    }
    const Json& regs = run.report["registers"];
    CHECK(regs.type == Json::Object);
    CHECK(regs.has("rax") && regs.has("r15") && regs.has("rsp") && regs.has("gs_base"));
    CHECK(regs["gs_base"].hex() == run.report["startup"]["main_thread"]["teb"].hex());
    if (std::string(kind) != "unresolved_import") {
        CHECK(run.result.status == BootStatus::Fault || run.result.status == BootStatus::GuestAbort);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

void test_layout_offsets() {
    // The static_asserts in boot.h pin the layouts; spot-check the sizes here too.
    CHECK(sizeof(win64::Teb64) == 0x2000);
    CHECK(sizeof(win64::Peb64) == 0x1000);
    CHECK(sizeof(BootImport) == 48);
}

// Report schema 2: milestone import indices, the import calls up to the first Present, and the
// members a report extension adds.
void test_report_import_fields() {
    Run run = boot_scenario(Scenario::ReturnCode);
    CHECK(run.report["milestones"].items.size() == 3);
    for (const Json& m : run.report["milestones"].items) {
        CHECK(m.has("import_seq") && m["import_seq"].number == 0);  // no import call before the entry returned
    }
    CHECK(run.report["first_present"].type == Json::Null);  // nothing marked a Present
    CHECK(!run.report.has("presents"));

    set_report_extension([]() -> std::string { return "\"presents\": [{\"index\": 0}]"; });
    run = boot_scenario(Scenario::PresentMark);
    set_report_extension(nullptr);
    CHECK(run.result.status == BootStatus::Exited);
    CHECK(run.report["presents"].type == Json::Array && run.report["presents"].items.size() == 1);
    CHECK(run.report["presents"][0]["index"].number == 0);
    // R40: the two paced presents are sampled without any capture.
    CHECK(run.report["present_samples"].type == Json::Array && run.report["present_samples"].items.size() == 2);
    CHECK(run.report["present_samples"][1]["index"].number == 1);
    CHECK(run.report["present_samples"][1].has("wall_ms") && run.report["present_samples"][1].has("draws"));
    CHECK(run.report.has("frame_times") && run.report["frame_times"]["phases"].type == Json::Array);
    const Json& fp = run.report["first_present"];
    CHECK(fp.type == Json::Object);
    CHECK(fp["import_seq"].number == 2);
    // ... and is a FIRST_PRESENT milestone at that import call (the last milestone; once).
    int first_present_milestones = 0;
    for (const Json& m : run.report["milestones"].items) {
        if (m["milestone"].text == "FIRST_PRESENT") {
            ++first_present_milestones;
            CHECK(m["import_seq"].number == 2);
        }
    }
    CHECK(first_present_milestones == 1);
    CHECK(fp["imports"].items.size() == 2);  // SetLastError and GetLastError, one thread, one call each
    uint64_t total = 0;
    bool names = false;
    for (const Json& row : fp["imports"].items) {
        total += static_cast<uint64_t>(row["count"].number);
        names = names || (row["symbol"].text == "SetLastError" && row["dll"].text == "KERNEL32.dll");
    }
    CHECK(total == 2 && names);  // the call after the Present is not counted
    CHECK(fp["imports"][0]["thread"].number == fp["imports"][1]["thread"].number);
    CHECK(run.report["counters"]["import_calls"].number == 3);
    // A later boot starts clean.
    run = boot_scenario(Scenario::ReturnCode);
    CHECK(run.report["first_present"].type == Json::Null);
}

void test_clean_boot_and_teb() {
    const Run run = boot_scenario(Scenario::ReturnCode);
    CHECK(run.result.status == BootStatus::Exited);
    CHECK(run.result.stats.exit_code == 42);
    CHECK(run.report["exit_code"].number == 42);
    CHECK(run.report["fault"].type == Json::Null);
    CHECK(run.report["phase"].text == "entry");
    {
        const Json& ms = run.report["milestones"];
        CHECK(ms.items.size() == 3 && ms[0]["milestone"].text == "PGIM_LOAD" &&
              ms[1]["milestone"].text == "TLS_INIT" && ms[2]["milestone"].text == "ENTRY_CALL");
    }
    // TLS callbacks ran first, in order, with (image base, DLL_PROCESS_ATTACH, null).
    CHECK(g_events.size() == 4);
    CHECK(g_events.size() == 4 && g_events[0] == "callback1" && g_events[1] == "callback1-tls-ok" &&
          g_events[2] == "callback2" && g_events[3] == "entry");
    for (int i = 0; i < 2; ++i) {
        CHECK(g_seen.callback_args[i][0] == kBase);
        CHECK(g_seen.callback_args[i][1] == win64::kDllProcessAttach);
        CHECK(g_seen.callback_args[i][2] == 0);
    }
    CHECK(run.result.stats.tls_callbacks == 2);
    CHECK(run.result.stats.tls_callbacks_run == 2);
    CHECK(run.result.stats.entry_started == 1);
    // TEB through gs.
    const Json& thread = run.report["startup"]["main_thread"];
    const uint64_t teb = thread["teb"].hex();
    CHECK(teb != 0 && teb % 0x1000 == 0);
    CHECK(g_seen.teb_self == teb);
    CHECK(g_seen.stack_base == thread["stack_base"].hex());
    CHECK(g_seen.stack_limit == thread["stack_limit"].hex());
    CHECK(g_seen.stack_base - g_seen.stack_limit == kPeStackReserve);  // PE SizeOfStackReserve
    CHECK(g_seen.pid == 0x1000);
    CHECK(g_seen.tid == os::win32::GetCurrentThreadId());
    CHECK(g_seen.peb == run.report["startup"]["process"]["peb"].hex());
    CHECK(g_seen.peb_image_base == kBase);
    CHECK(g_seen.mxcsr == kGuestMxcsrDefault);
    CHECK(g_seen.limit_low == g_seen.stack_limit && g_seen.limit_high == g_seen.stack_base);
    CHECK(g_seen.command_line_a == "boot_test.exe -flag ä");
    CHECK(g_seen.command_line == u"boot_test.exe -flag \u00E4");
    // Implicit TLS through gs:[0x58].
    CHECK(g_seen.tls_index == 0);
    CHECK(g_seen.tls_pointer == thread["tls_vector"].hex());
    CHECK(g_seen.tls_block == thread["tls_block"].hex());
    CHECK(g_seen.tls_block % 16 == 0);
    bool template_ok = g_seen.tls_bytes.size() == kTlsTemplateSize + kTlsZeroFill;
    for (size_t i = 0; template_ok && i < g_seen.tls_bytes.size(); ++i) {
        template_ok = g_seen.tls_bytes[i] == (i < kTlsTemplateSize ? template_byte(i) : 0);
    }
    CHECK(template_ok);
    const Json& tls = run.report["startup"]["modules"][0]["tls"];
    CHECK(tls["present"].boolean);
    CHECK(tls["template_size"].number == kTlsTemplateSize);
    CHECK(tls["zero_fill"].number == kTlsZeroFill);
    CHECK(tls["alignment"].number == 16);
    CHECK(tls["callbacks"].items.size() == 2 && tls["callbacks"][0].hex() == kCallback1);
    // Startup summary.
    const Json& imports = run.report["startup"]["imports"];
    CHECK(imports["total"].number == 8);
    CHECK(imports["bound"].number == 6);
    CHECK(run.report["startup"]["modules"][0]["imports"]["runtime_thunks"].number == 1);
    CHECK(imports["unresolved"].number == 2);
    CHECK(imports["by_dll"].items.size() == 3);
    CHECK(run.report["startup"]["functions_registered"].number == 6);
    CHECK(run.report["startup"]["image_base"].hex() == kBase);
    // The process keeps the guest thread's last-error slot bound until reset.
    CHECK(os::win32::GetLastError() == 0);
}

void test_exit_paths() {
    Run run = boot_scenario(Scenario::ExitViaCrt);
    CHECK(run.result.status == BootStatus::Exited);
    CHECK(run.result.stats.exit_code == 7);
    CHECK(run.report["exit_code"].number == 7);
    CHECK(g_events.size() == 4 && g_events.back() == "entry");  // nothing after exit ran
    CHECK(run.report["counters"]["import_calls"].number == 1);
    CHECK(run.report["counters"]["recent"].items.size() == 1);
    CHECK(run.report["counters"]["recent"][0]["symbol"].text == "exit");

    run = boot_scenario(Scenario::RequestExit);
    CHECK(run.result.status == BootStatus::Exited);
    CHECK(run.result.stats.exit_code == 9);
    CHECK(g_events.back() == "entry");

    run = boot_scenario(Scenario::TerminateViaShim);
    CHECK(run.result.status == BootStatus::Exited);
    CHECK(run.result.stats.exit_code == 11);
    CHECK(g_events.back() == "entry");

    run = boot_scenario(Scenario::CrtAbort);
    CHECK(run.result.status == BootStatus::GuestAbort);
    CHECK(run.report["fault"]["kind"].text == "crt_abort");
    CHECK(run.report["exit_code"].type == Json::Null);
}

void test_fault_reports() {
    // Trap two calls deep: function + backtrace, walked by R11's unwinder over the image's
    // .pdata, and by the call-verified stack scan when the image has no .pdata.
    const std::vector<uint8_t> saved = g_image;
    Run run;
    for (const bool pdata : {true, false}) {
        g_image = build_image(true, kBase, {kCallback1, kCallback2}, 0xA0, pdata);
        run = boot_scenario(Scenario::Trap);
        CHECK(run.result.status == BootStatus::Fault);
        CHECK(run.report["phase"].text == "entry");
        CHECK(run.report["startup"]["modules"][0]["pdata"].boolean == pdata);
        check_fault(run, "trap", kLeaf + 4, kLeaf);
        CHECK(run.report["fault"]["trap_kind"].text == "GUEST_TRAP");
        CHECK(run.report["code_misses"]["trap"]["count"].number == 1);
        CHECK(run.report["code_misses"]["trap"]["targets"][0].hex() == kLeaf + 4);
        CHECK(run.report["code_misses"]["unknown_target"]["count"].number == 0);
        const Json& bt = run.report["backtrace"];
        CHECK(bt["method"].text == (pdata ? "rtl-virtual-unwind" : "stack-scan-call-verified"));
        CHECK(bt["frames"].items.size() == 2);
        CHECK(bt["frames"][0]["return_address"].hex() == kMidCallsLeaf);
        CHECK(bt["frames"][0]["function"].hex() == kMid);
        CHECK(bt["frames"][0]["offset"].hex() == kMidCallsLeaf - kMid);
        CHECK(bt["frames"][1]["return_address"].hex() == kEntryCallsMid);
        CHECK(bt["frames"][1]["function"].hex() == kEntry);
        CHECK(bt["frames"][0]["stack_address"].hex() < bt["frames"][1]["stack_address"].hex());
        CHECK(run.result.summary.find("sub_140001180+0x4") != std::string::npos);
    }
    g_image = saved;

    run = boot_scenario(Scenario::Unimplemented);
    check_fault(run, "unimplemented", kEntry + 0x20, kEntry);
    run = boot_scenario(Scenario::DivideError);
    check_fault(run, "divide_error", kEntry + 0x24, kEntry);

    run = boot_scenario(Scenario::UnknownTarget);
    check_fault(run, "unknown_target", kUnknownInsideEntry, kEntry);
    CHECK(run.report["fault"]["trap_kind"].text == "INDIRECT_CALL_MISS");
    CHECK(run.report["fault"]["detail"].hex() == kEntry);  // nearest entry below the target
    {
        // The dispatch miss is counted and its target listed for recomp seeding (`--entry`).
        const Json& misses = run.report["code_misses"];
        CHECK(misses["unknown_target"]["count"].number == 1);
        CHECK(misses["unknown_target"]["targets"].items.size() == 1);
        CHECK(misses["unknown_target"]["targets"][0].hex() == kUnknownInsideEntry);
        CHECK(misses["unknown_resume_target"]["count"].number == 0);
        CHECK(misses["trap"]["targets"].items.empty());
    }
    {
        // Fault record: rip/target, containing function, indirect source, entry ring, register snapshot.
        const Json& fr = run.report["fault_record"];
        CHECK(fr["kind"].text == "unknown_target");
        CHECK(fr["code_miss"].boolean);
        CHECK(fr["target"].hex() == kUnknownInsideEntry);
        CHECK(fr["rip"].hex() == kUnknownInsideEntry);
        CHECK(fr["function"]["entry"].hex() == kEntry);
        CHECK(fr["function"]["offset"].hex() == kUnknownInsideEntry - kEntry);
        CHECK(fr["source"]["return_address"].hex() == kEntryCallsIat);
        CHECK(fr["source"]["location"]["entry"].hex() == kEntry);
        CHECK(!fr["registers"]["rsp"].text.empty());
        CHECK(fr["elapsed_ms"].type == Json::Number);
        CHECK(!fr["recent_entries"].items.empty());
        const Json& last = fr["recent_entries"][fr["recent_entries"].items.size() - 1];
        CHECK(last["target"].hex() == kUnknownInsideEntry);
        CHECK(last["from"].hex() == kEntryCallsIat);
        CHECK(!fr["call_stack"]["frames"].items.empty());
    }
    // Toggled off: no record and no entry ring in the report.
    {
        const Run off = boot_scenario(Scenario::UnknownTarget, [](BootConfig& c) { c.fault_record = false; });
        CHECK(off.report["fault_record"].type == Json::Null);
        CHECK(off.report["recent_dispatches"].items.empty());
        CHECK(off.report["code_misses"]["unknown_target"]["count"].number == 1);
    }
    // (the synthetic unwind info describes the Trap scenario's frames; only the caller is exact here)
    CHECK(!run.report["backtrace"]["frames"].items.empty());
    CHECK(run.report["backtrace"]["frames"][0]["return_address"].hex() == kEntryCallsIat);

    run = boot_scenario(Scenario::StackImbalance);
    check_fault(run, "stack_imbalance", kEntry, kEntry);

    run = boot_scenario(Scenario::BadStack);
    check_fault(run, "bad_stack", kMid, kMid);
    CHECK(run.report["fault"]["detail"].hex() == 0x20);

    // Unresolved import: DLL!symbol, the call site and no survey entry.
    run = boot_scenario(Scenario::Unresolved);
    CHECK(run.result.status == BootStatus::Fault);
    check_fault(run, "unresolved_import", kEntryCallsIat, kEntry);
    const Json& imp = run.report["fault"]["import"];
    CHECK(imp["dll"].text == "KERNEL32.dll");
    CHECK(imp["symbol"].text == "MissingOne");
    CHECK(imp["module"].text == "boot_test.exe");
    CHECK(run.report["fault"]["trap_kind"].text == "UNRESOLVED_IMPORT");
    CHECK(imp["slot_rva"].hex() == kSlotMissingOne);
    CHECK(!imp["delay_load"].boolean);
    CHECK(run.report["counters"]["imports"].items.size() == 1);
    CHECK(run.report["counters"]["imports"][0]["resolved"].type == Json::Bool &&
          !run.report["counters"]["imports"][0]["resolved"].boolean);
    CHECK(run.report["survey"]["unresolved_called"].items.empty());
    CHECK(run.result.summary.find("KERNEL32.dll!MissingOne") != std::string::npos);

    // A fault in a TLS callback is attributed to that phase (entry never starts).
    run = boot_scenario(Scenario::ReturnCode, [](BootConfig& cfg) {
        static const LiftedFunctionEntry kBroken[] = {{kEntry, &sub_entry}, {kCallback1, &sub_leaf}};
        cfg.modules[0].lifted_tables.clear();
        cfg.modules[0].lifted_tables.push_back(std::span<const LiftedFunctionEntry>(kBroken));
    });
    CHECK(run.result.status == BootStatus::Fault);
    CHECK(run.report["phase"].text == "tls_callback");
    CHECK(run.result.stats.entry_started == 0);
    CHECK(run.report["fault"]["kind"].text == "trap");
}

void test_survey_mode() {
    // Bounded: the loop would make 2000 import calls; the bound stops it.
    Run run = boot_scenario(Scenario::SurveyLoop, [](BootConfig& cfg) {
        cfg.survey = true;
        cfg.max_dispatches = 50;
        cfg.recent_import_count = 4;
    });
    CHECK(run.result.status == BootStatus::LimitReached);
    CHECK(run.report["fault"]["kind"].text == "dispatch_limit");
    CHECK(run.report["fault"]["trap_kind"].type == Json::Null);
    CHECK(run.report["survey"]["enabled"].boolean);
    const Json& called = run.report["survey"]["unresolved_called"];
    CHECK(called.items.size() == 2);
    CHECK(called[0]["symbol"].text == "MissingTwo" && called[0]["dll"].text == "USER32.dll");
    CHECK(called[1]["symbol"].text == "MissingOne");
    CHECK(called[0]["first_seq"].number == 1 && called[1]["first_seq"].number == 2);
    // Dispatches 1-2 are the TLS callbacks, 3 the entry point, 4.. the import calls; dispatch 51
    // (import call 48) hits the bound before its thunk runs.
    CHECK(called[0]["count"].number + called[1]["count"].number == 47);
    CHECK(run.report["counters"]["dispatches"].number == 51);
    CHECK(run.report["counters"]["import_calls"].number == 48);
    CHECK(run.report["counters"]["recent"].items.size() == 4);
    CHECK(run.report["counters"]["recent"][3]["seq"].number == 48);
    CHECK(run.report["counters"]["recent"][0]["seq"].number == 45);
    CHECK(run.result.stats.unresolved_called == 2);

    // Unbounded: survives past unresolved calls and returns normally.
    run = boot_scenario(Scenario::SurveyReturn, [](BootConfig& cfg) { cfg.survey = true; });
    CHECK(run.result.status == BootStatus::Exited);
    CHECK(run.result.stats.exit_code == 5);
    const Json& list = run.report["survey"]["unresolved_called"];
    CHECK(list.items.size() == 2);
    CHECK(list[0]["symbol"].text == "MissingTwo" && list[0]["count"].number == 2);
    CHECK(list[1]["symbol"].text == "MissingOne" && list[1]["count"].number == 1);

    // Wall-time bound alone (checked every 256 dispatches).
    run = boot_scenario(Scenario::SurveyLoop, [](BootConfig& cfg) {
        cfg.survey = true;
        cfg.max_wall_ms = 1;
    });
    CHECK(run.result.status == BootStatus::LimitReached || run.result.status == BootStatus::Exited);
}

void test_last_error_shared_with_teb() {
    const Run run = boot_scenario(Scenario::LastError);
    CHECK(run.result.status == BootStatus::Exited);
    CHECK(run.result.stats.exit_code == 0);
}

// Observability: import arguments, the import trace, threads and the address-space log.
void test_observability() {
    const Run run = boot_scenario(Scenario::LastError, [](BootConfig& cfg) { cfg.import_trace_limit = 16; });
    CHECK(run.result.status == BootStatus::Exited);
    const Json& recent = run.report["counters"]["recent"];
    CHECK(recent.items.size() == 2);
    CHECK(recent[0]["symbol"].text == "SetLastError");
    CHECK(recent[0]["args"][0].hex() == 1234);  // rcx at the call
    CHECK(recent[0]["thread"].number == os::win32::GetCurrentThreadId());
    const std::string trace = import_trace_text();
    CHECK(trace.find("1 ") == 0);
    CHECK(trace.find("KERNEL32.dll!SetLastError 4d2 ") != std::string::npos);
    CHECK(trace.find("KERNEL32.dll!GetLastError") != std::string::npos);
    const Json& threads = run.report["threads"];
    CHECK(threads.items.size() == 1);
    CHECK(threads[0]["teb"].hex() == run.report["startup"]["main_thread"]["teb"].hex());
    CHECK(threads[0]["last_import"]["symbol"].text == "GetLastError");
    // Loading the image, the process block, the stack and the TEB all go through the arena.
    const Json& space = run.report["address_space"];
    CHECK(space["events"].number >= 4);
    CHECK(space["failures"].number == 0);
    CHECK(space["reserved_peak"].hex() >= kSizeOfImage);
    bool image_reserve = false;
    for (const Json& e : space["log"].items) {
        if (e["event"].text == "reserve" && e["va"].hex() == kBase && e["ok"].boolean) {
            image_reserve = true;
            CHECK(e["import"].type == Json::Null);  // the runtime's own request, no import
        }
    }
    CHECK(image_reserve);
    // The log keeps at most BootConfig::arena_log_limit events (all are counted).
    {
        const Run limited = boot_scenario(Scenario::LastError, [](BootConfig& cfg) { cfg.arena_log_limit = 2; });
        const Json& s = limited.report["address_space"];
        CHECK(s["log"].items.size() == 2);
        CHECK(s["events"].number > 2);
        CHECK(s["log_truncated"].boolean);
    }
    CHECK(run.report["first_calls"].type == Json::Array);
    CHECK(run.report["recent_dispatches"].type == Json::Array);

    // The boot seeds the console-mode value the title reads at startup (default: Series X).
    auto console_mode = []() -> int64_t {
        Pg6Handle key = 0;
        if (pg6_win32_reg_open_key_w(0x80000002u, u"Software\\Microsoft\\Windows NT\\CurrentVersion\\OEM",
                                     &key) != 0) {
            return -1;
        }
        uint32_t type = 0, value = 0, size = sizeof(value);
        const int32_t status = pg6_win32_reg_query_value_ex_w(key, u"ConsoleMode", nullptr, &type,
                                                              reinterpret_cast<uint8_t*>(&value), &size);
        pg6_win32_reg_close_key(key);
        return status == 0 && type == 4 && size == 4 ? static_cast<int64_t>(value) : -1;
    };
    os::win32::registry_clear();
    CHECK(console_mode() == -1);
    boot_scenario(Scenario::LastError);
    CHECK(console_mode() == static_cast<int64_t>(kConsoleModeSeriesX));
    os::win32::registry_clear();
    boot_scenario(Scenario::LastError, [](BootConfig& cfg) { cfg.console_mode = kConsoleModeDevKit; });
    CHECK(console_mode() == static_cast<int64_t>(kConsoleModeDevKit));
    os::win32::registry_clear();
    boot_scenario(Scenario::LastError, [](BootConfig& cfg) { cfg.console_mode = 0; });
    CHECK(console_mode() == -1);

    // The emulated console's sandbox reaches the xgameruntime provider (default: retail).
    boot_scenario(Scenario::LastError);
    CHECK(os::gdk::provider::xbox_live_sandbox_id() == "RETAIL");
    boot_scenario(Scenario::LastError, [](BootConfig& cfg) { cfg.sandbox_id.clear(); });
    CHECK(os::gdk::provider::xbox_live_sandbox_id().empty());

    // The local user's age group reaches the provider (default: Unknown).
    os::gdk::provider::set_user_age_group(os::gdk::provider::UserAgeGroup::Teen);
    boot_scenario(Scenario::LastError);
    CHECK(os::gdk::provider::user_age_group() == os::gdk::provider::UserAgeGroup::Unknown);
    boot_scenario(Scenario::LastError,
                  [](BootConfig& cfg) { cfg.user_age_group = os::gdk::provider::UserAgeGroup::Child; });
    CHECK(os::gdk::provider::user_age_group() == os::gdk::provider::UserAgeGroup::Child);
    os::gdk::provider::set_user_age_group(os::gdk::provider::UserAgeGroup::Unknown);

    // The thread hook sees the main guest thread, on the thread that runs it.
    static uint32_t started_id = 0;
    static int started_calls = 0;
    started_id = 0;
    started_calls = 0;
    boot_scenario(Scenario::LastError, [](BootConfig& cfg) {
        cfg.guest_thread_started = [](uint32_t id) {
            started_id = id;
            ++started_calls;
        };
    });
    CHECK(started_calls >= 1 && started_id != 0);
    CHECK(guest::guest_thread_last_import(0xFFFFFFF0u).empty());
}

// host_callable_calls counts calls to named host callables in first-call order; com_stub_hits
// lists COM stub slots (none here); recent_dispatches name host callables and imports.
void test_host_callable_statistics() {
    const Run run = boot_scenario(Scenario::HostCallable);
    CHECK(run.result.status != BootStatus::Exited);
    bool answer_named = false;
    bool import_named = false;
    size_t without_caller = 0;
    for (const Json& d : run.report["recent_dispatches"].items) {
        // Every dispatch was reached by a call: the observer read its return address off the
        // guest thread's stack.
        without_caller += d["return_address"].hex() == 0 ? 1 : 0;
        answer_named = answer_named || d["name"].text == "ITest::Answer";
        import_named = import_named || (d["name"].type == Json::String &&
                                        d["name"].text.find("!GetLastError") != std::string::npos);
    }
    CHECK(answer_named);
    CHECK(import_named);
    CHECK(!run.report["recent_dispatches"].items.empty() && without_caller == 0);
    const Json& calls = run.report["host_callable_calls"];
    CHECK(calls.type == Json::Array);
    bool found = false;
    for (const Json& c : calls.items) {
        if (c["name"].text == "ITest::Answer") {
            found = true;
            CHECK(c["calls"].number == 2);
        }
    }
    CHECK(found);
    CHECK(run.report["com_stub_hits"].type == Json::Array);
}

void test_off_thread_fault() {
    // Without the hook the boot leaves such faults to the default handler: not exercised here.
    const Run run = boot_scenario(Scenario::OffThreadFault, [](BootConfig& cfg) {
        cfg.off_thread_fault = [] {
            g_off_thread_report = last_report_json();
            g_off_thread_reported = true;
            for (;;) {
                std::this_thread::sleep_for(std::chrono::hours(1));  // the harness terminates here
            }
        };
    });
    CHECK(g_off_thread_reported.load());
    bool ok = false;
    const Json report = parse_json(g_off_thread_report, &ok);
    CHECK(ok);
    CHECK(report["status"].text == "fault");
    CHECK(report["fault"]["kind"].text == "unimplemented");
    CHECK(report["fault"]["va"].hex() == kWorker + 4);
    CHECK(report["message"].text.find("on guest thread") != std::string::npos);
    (void)run;
}

// An unresolved import called on another guest thread is reported through the same hook
// (before, the process aborted without a report).
void test_off_thread_unresolved_import() {
    g_off_thread_reported = false;
    g_off_thread_report.clear();
    const Run run = boot_scenario(Scenario::OffThreadUnresolved, [](BootConfig& cfg) {
        cfg.off_thread_fault = [] {
            g_off_thread_report = last_report_json();
            g_off_thread_reported = true;
            for (;;) {
                std::this_thread::sleep_for(std::chrono::hours(1));
            }
        };
    });
    CHECK(g_off_thread_reported.load());
    bool ok = false;
    const Json report = parse_json(g_off_thread_report, &ok);
    CHECK(ok);
    CHECK(report["status"].text == "fault");
    CHECK(report["fault"]["kind"].text == "unresolved_import");
    CHECK(report["message"].text.find("call to unresolved import") != std::string::npos);
    CHECK(report["message"].text.find("on guest thread") != std::string::npos);
    (void)run;
}

// Log lines of the runtime (os::log tap), read by the fault-record tests below. Works the same on the
// native build and the wasm64 build (where os::log goes to the host_log import, not stderr).
std::mutex g_tap_mutex;
std::vector<std::string> g_tap_lines;
void tap_line(std::string_view line) {
    std::lock_guard<std::mutex> lock(g_tap_mutex);
    g_tap_lines.emplace_back(line);
}
std::vector<std::string> tap_snapshot() {
    std::lock_guard<std::mutex> lock(g_tap_mutex);
    return g_tap_lines;
}
size_t tap_find(const std::vector<std::string>& lines, const char* prefix) {
    for (size_t i = lines.size(); i-- > 0;) {
        if (lines[i].rfind(prefix, 0) == 0) {
            return i;
        }
    }
    return lines.size();
}

// The tail of the wasm boot's off-thread path: the boot records the fault and logs the report and the
// `[fault-record]` line, then calls wasm_off_thread_fault (the browser build's BootConfig hook), which logs
// a marker and ends the run. Here the run's end is replaced by parking the worker.
std::vector<std::string> g_hook_lines;
Json g_hook_record;
std::atomic<bool> g_hook_exit_called{false};
void hook_exit_park() {
    g_hook_lines = tap_snapshot();
    g_hook_exit_called = true;
    // Releases the boot thread's wait in the scenario (it polls this flag; on Windows each 1 ms sleep
    // lasts ~15 ms, so waiting out the 5000-iteration cap takes over a minute per scenario).
    g_off_thread_reported = true;
    for (;;) {
        std::this_thread::sleep_for(std::chrono::hours(1));
    }
}

Json run_off_thread_record(Scenario scenario, std::vector<std::string>* lines_out = nullptr) {
    g_off_thread_reported = false;
    g_hook_exit_called = false;
    g_hook_lines.clear();
    {
        std::lock_guard<std::mutex> lock(g_tap_mutex);
        g_tap_lines.clear();
    }
    os::set_log_tap(&tap_line);
    boot::set_wasm_off_thread_fault_exit(&hook_exit_park);
    const Run run = boot_scenario(scenario, [](BootConfig& cfg) {
        cfg.log_progress = true;
        cfg.fault_record = true;
        cfg.off_thread_fault = &boot::wasm_off_thread_fault;
    });
    (void)run;
    boot::set_wasm_off_thread_fault_exit(nullptr);
    os::set_log_tap(nullptr);
    CHECK(g_hook_exit_called.load());
    const std::vector<std::string>& lines = g_hook_lines;
    if (lines_out != nullptr) {
        *lines_out = lines;
    }
    const size_t rec = tap_find(lines, "[fault-record] ");
    const size_t marker = tap_find(lines, "[boot] off-thread guest fault captured");
    CHECK(rec < lines.size());
    CHECK(marker < lines.size());
    CHECK(rec < marker);  // the record is logged before the run ends
    if (rec >= lines.size()) {
        return Json{};
    }
    bool ok = false;
    const Json j = parse_json(lines[rec].substr(std::strlen("[fault-record] ")), &ok);
    CHECK(ok);
    return j;
}

// Off-thread guest faults end in wasm_off_thread_fault with a `[fault-record]` line that names the
// fault (what tools/run_browser_boot.py reads from the tail of a browser log).
void test_wasm_off_thread_fault_records() {
    {
        const Json j = run_off_thread_record(Scenario::OffThreadNullIndirect);
        CHECK(j["kind"].text == "unknown_target");
        CHECK(j["va"].hex() == 0);
        CHECK(j["unimplemented_slot"].type == Json::Null);
        CHECK(j["callers"].type == Json::Array);
        // Locating info: the worker pushed return address kWorker + 4 before the null call.
        CHECK(j["source"].type == Json::Object);
        CHECK(j["source"]["return_address"].hex() == kWorker + 4);
        CHECK(j["source"]["location"]["entry"].hex() == kWorker);
        CHECK(j["registers"].type == Json::Object);
        CHECK(j["registers"]["rsp"].hex() != 0);
        CHECK(j["call_stack"]["frames"].type == Json::Array);
        CHECK(j["recent_entries"].type == Json::Array);
        CHECK(!j["recent_entries"].items.empty());
        CHECK(j["recent_entries"].items.back()["target"].hex() == 0);
        CHECK(j["recent_entries"].items.back()["from"].hex() == kWorker + 4);
    }
    {
        const Json j = run_off_thread_record(Scenario::OffThreadUnknownTarget);
        CHECK(j["kind"].text == "unknown_target");
        CHECK(j["va"].hex() == kUnknownInsideEntry);
    }
    {
        const Json j = run_off_thread_record(Scenario::OffThreadComSlot);
        CHECK(j["kind"].text == "unimplemented_slot");
        CHECK(j["unimplemented_slot"]["interface"].text == "IFaultProbe");
        CHECK(j["unimplemented_slot"]["slot"].number == 7);
    }
    {
        const Json j = run_off_thread_record(Scenario::OffThreadUnresolved);
        CHECK(j["kind"].text == "unresolved_import");
        CHECK(j["import"].type == Json::Object);
        CHECK(!j["import"]["symbol"].text.empty());
        CHECK(!j["import"]["dll"].text.empty());
    }
    {
        const Json j = run_off_thread_record(Scenario::OffThreadFault);
        CHECK(j["kind"].text == "unimplemented");
        CHECK(j["va"].hex() == kWorker + 4);
        CHECK(j["function"].type == Json::Object);  // names the lifted function containing the va
    }
}

void test_second_thread() {
    Run run = boot_scenario(Scenario::ReturnCode);
    CHECK(run.result.status == BootStatus::Exited);
    const uint64_t main_teb = run.report["startup"]["main_thread"]["teb"].hex();
    const uint64_t main_block = run.report["startup"]["main_thread"]["tls_block"].hex();
    // The main thread changes its TLS copy; the new thread must still get the pristine template.
    store32(main_block, 0x11111111u);

    GuestThreadContext worker{};
    CHECK(prepare_guest_thread(0x10000, 77, worker));
    CHECK(worker.teb != 0 && worker.teb != main_teb);
    CHECK(worker.tls_block != 0 && worker.tls_block != main_block);
    CHECK(worker.thread.state.gs_base == worker.teb);
    CHECK(load64(worker.teb + 0x30) == worker.teb);
    CHECK(load64(worker.teb + 0x48) == 77);
    CHECK(load64(worker.teb + 0x58) == worker.tls_vector);
    CHECK(load64(worker.teb + 0x60) == run.report["startup"]["process"]["peb"].hex());
    CHECK(load64(worker.teb + 0x08) == worker.thread.stack_top);
    CHECK(load32(worker.tls_block) == 0xA3A2A1A0u);
    CHECK(load32(worker.tls_block + kTlsTemplateSize) == 0);
    // Guest code on the new thread reaches its own block through gs.
    CHECK(call_guest(*g_arena, worker.thread.state, kWorker) == worker.tls_block);
    // Its last-error slot is separate from the main thread's.
    GuestThreadContext* previous = activate_guest_thread(&worker);
    os::win32::SetLastError(5);
    CHECK(load32(worker.teb + 0x68) == 5);
    CHECK(load32(main_teb + 0x68) != 5);
    activate_guest_thread(previous);
    CHECK(run_tls_callbacks(worker, win64::kDllThreadAttach) == 2);
    CHECK(g_seen.callback_args[0][1] == win64::kDllThreadAttach);
    release_guest_thread(worker);
    CHECK(worker.teb == 0);

    // Host threads that O16 gives a guest thread lazily (CreateThread, pool callbacks) get a TEB
    // and TLS from the boot, and the TLS callbacks run with DLL_THREAD_ATTACH.
    g_events.clear();
    g_seen.callback_args[0][1] = 0;
    uint64_t lazy_teb = 0, lazy_self = 0, lazy_block = 0, lazy_rax = 0;
    uint32_t lazy_first = 0;
    std::thread host([&] {
        GuestState& state = com::current_guest_state();
        lazy_teb = state.gs_base;
        lazy_self = load64(state.gs_base + 0x30);
        lazy_block = load64(load64(state.gs_base + 0x58));
        lazy_first = load32(lazy_block);
        lazy_rax = call_guest(*g_arena, state, kWorker);
    });
    host.join();
    CHECK(lazy_teb != 0 && lazy_teb != main_teb);
    CHECK(lazy_self == lazy_teb);
    CHECK(lazy_block != 0 && lazy_block != main_block);
    CHECK(lazy_first == 0xA3A2A1A0u);
    CHECK(lazy_rax == lazy_block);
    CHECK(g_events.size() == 3 && g_events[0] == "callback1");
    CHECK(g_seen.callback_args[0][1] == win64::kDllThreadAttach);
}

void test_no_tls_and_load_failure() {
    const std::vector<uint8_t> saved = g_image;
    g_image = build_image(false);
    Run run = boot_scenario(Scenario::ReturnCode);
    CHECK(run.result.status == BootStatus::Exited);
    CHECK(!run.report["startup"]["modules"][0]["tls"]["present"].boolean);
    CHECK(run.report["startup"]["main_thread"]["tls_vector"].type == Json::Null);
    CHECK(g_seen.tls_pointer == 0);
    CHECK(g_events.size() == 1 && g_events[0] == "entry");

    g_image.assign(g_image.size(), 0);  // not a PGIM container
    run = boot_scenario(Scenario::ReturnCode);
    CHECK(run.result.status == BootStatus::LoadFailed);
    CHECK(run.report["phase"].text == "load");
    CHECK(run.report["startup"]["image_base"].type == Json::Null);
    g_image = saved;

    // C ABI: in-memory image, one table, quiet.
    g_scenario = Scenario::ReturnCode;
    const Pg6LiftedFunctionEntry* tables[] = {kTable};
    const size_t counts[] = {std::size(kTable)};
    Pg6GuestBootModule module{};
    module.image_bytes = g_image.data();
    module.image_size = g_image.size();
    module.name = "boot_test.exe";
    module.lifted_tables = tables;
    module.lifted_table_counts = counts;
    module.lifted_table_count = 1;
    module.imports = kImports;
    module.import_count = std::size(kImports);
    Pg6GuestBootConfig cfg{};
    cfg.modules = &module;
    cfg.module_count = 1;
    cfg.quiet = 1;
    Pg6GuestBootStats stats{};
    os::win32::set_vmem_arena(g_arena);
    CHECK(pg6_guest_boot_run(&cfg, &stats) == PG6_BOOT_EXITED);
    CHECK(stats.exit_code == 42);
    CHECK(stats.imports_total == 8);
    char small[16];
    const size_t full = pg6_guest_boot_report_json(small, sizeof(small));
    CHECK(full > sizeof(small) && std::strlen(small) == sizeof(small) - 1);
}

// A second guest module (a DLL) with its own TLS directory, callback and IAT.
constexpr uint64_t kDllBase = 0x180000000ULL;
constexpr uint64_t kDllCallback = kDllBase + 0x1000;

uint32_t g_dll_tls_index = 0;
uint32_t g_dll_tls_first = 0;

void sub_dll_callback(GuestState* s) {
    g_events.push_back("dll-callback");
    g_dll_tls_index = load32(kDllBase + kTlsIndexRva);
    const uint64_t vector = load64(s->gs_base + 0x58);
    g_dll_tls_first = load32(load64(vector + 8ull * g_dll_tls_index));
    ret(s);
}

const LiftedFunctionEntry kDllTable[] = {{kDllCallback, &sub_dll_callback}};
const BootImport kDllImports[] = {
    {kIatRva, 0, 0, "VCRUNTIME140.dll", "memcpy", nullptr, &thunk_generated_unresolved},
};

void test_two_modules() {
    const std::vector<uint8_t> dll_image = build_image(true, kDllBase, {kDllCallback}, 0xB0);
    os::pgim::MemoryImageReader dll_reader(dll_image.data(), dll_image.size());
    g_scenario = Scenario::ReturnCode;
    g_events.clear();
    g_seen = Observed{};
    os::pgim::MemoryImageReader reader(g_image.data(), g_image.size());
    BootConfig cfg = make_config(reader);
    BootModule dll;
    dll.image = &dll_reader;
    dll.name = "helper.dll";
    dll.lifted_tables.push_back(std::span<const LiftedFunctionEntry>(kDllTable));
    dll.imports = std::span<const BootImport>(kDllImports);
    cfg.modules.push_back(std::move(dll));
    const BootResult result = run_boot(cfg);
    bool ok = false;
    const Json report = parse_json(result.report_json, &ok);
    CHECK(ok);
    CHECK(result.status == BootStatus::Exited);
    CHECK(result.stats.exit_code == 42);
    // DLL callbacks first, then the executable's, then the entry point.
    CHECK(g_events.size() == 5 && g_events[0] == "dll-callback" && g_events[1] == "callback1" &&
          g_events[4] == "entry");
    // TLS indices in module order: the executable 0, the DLL 1; each block has its own template.
    CHECK(g_seen.tls_index == 0);
    CHECK(g_dll_tls_index == 1);
    CHECK(g_dll_tls_first == 0xB3B2B1B0u);
    CHECK(g_seen.tls_bytes.size() > 0 && g_seen.tls_bytes[0] == 0xA0);
    const Json& modules = report["startup"]["modules"];
    CHECK(modules.items.size() == 2);
    CHECK(modules[1]["name"].text == "helper.dll");
    CHECK(modules[1]["image_base"].hex() == kDllBase);
    CHECK(modules[1]["tls"]["index"].number == 1);
    CHECK(modules[1]["tls"]["callbacks_run"].number == 1);
    CHECK(modules[1]["imports"]["unresolved"].number == 1);
    CHECK(modules[0]["tls"]["index"].number == 0);
    CHECK(report["startup"]["functions_registered"].number == 7);
    CHECK(report["startup"]["imports"]["total"].number == 9);
    // The DLL's IAT slot got its own synthetic address (bound to the boot's unresolved thunk).
    SyntheticTarget target;
    CHECK(describe_synthetic(load64(kDllBase + kIatRva), &target));
    CHECK(target.is_import && target.image_base == kDllBase && target.slot_rva == kIatRva);
    CHECK(result.stats.tls_callbacks == 3);
}

void test_host_fault_mapping() {
    Run run = boot_scenario(Scenario::ReturnCode);
    const auto host_mid = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&sub_mid));
    uint64_t offset = 99;
    // Function addresses of hand-written functions are not ordered like guest VAs, but an exact
    // function address always maps back to its own entry.
    CHECK(guest_function_for_host_pc(host_mid, &offset) == kMid);
    CHECK(offset == 0);
    CHECK(report_host_fault("host_access_violation", 0xDEAD0000, host_mid));
    bool ok = false;
    const Json report = parse_json(last_report_json(), &ok);
    CHECK(ok);
    CHECK(report["status"].text == "fault");
    CHECK(report["fault"]["kind"].text == "host_access_violation");
    CHECK(report["fault"]["host_address"].hex() == 0xDEAD0000);
    CHECK(report["fault"]["function"]["entry"].hex() == kMid);
}

// ---------------------------------------------------------------------------
// R37c: Shard-only dispatch boot tests (empty lifted_tables)
// ---------------------------------------------------------------------------

enum class ShardScenario {
    IndirectAndImport,
    UnknownTargetFault,
    TrapFault,
};

ShardScenario g_shard_scenario = ShardScenario::IndirectAndImport;
std::vector<std::string> g_shard_events;
uint64_t g_shard_helper_result = 0;
uint64_t g_shard_last_error = 0;

void shard0_dispatch(uint32_t local_idx, Pg6GuestState* state) {
    switch (local_idx) {
        case 0: {  // kEntry
            g_shard_events.push_back("shard0:entry");
            if (g_shard_scenario == ShardScenario::IndirectAndImport) {
                // 1) Indirect call into a function living in shard 1.
                reg(state, GuestReg::Rcx) = 19;
                push(state, kEntryCallsMid);
                __playgta6_indirect_call(state, kMid);
                g_shard_helper_result = reg(state, GuestReg::Rax);

                // 2) Synthetic import thunk calls (SetLastError / GetLastError).
                reg(state, GuestReg::Rcx) = 0x2468;
                call_iat(state, kSlotSetLastError, kEntryCallsIat);
                call_iat(state, kSlotGetLastError, kEntryCallsIat);
                g_shard_last_error = reg(state, GuestReg::Rax);

                // 3) Synthetic host callable (COM-style) alongside shard dispatch.
                const uint64_t answer = register_host_callable(&host_answer, "IShard::Answer");
                push(state, kEntryCallsIat);
                __playgta6_indirect_call(state, answer);
                const uint64_t host_val = reg(state, GuestReg::Rax);

                reg(state, GuestReg::Rax) = g_shard_helper_result + host_val;
                ret(state);
                return;
            }
            if (g_shard_scenario == ShardScenario::UnknownTargetFault) {
                push(state, kEntryCallsIat);
                __playgta6_indirect_call(state, kUnknownInsideEntry);
                ret(state);
                return;
            }
            // TrapFault: push 8-byte prolog matching .pdata UWOP_ALLOC_SMALL 8, then call kMid -> kLeaf.
            push(state, kEntryStaleValue);
            push(state, kEntryCallsMid);
            __playgta6_indirect_call(state, kMid);
            reg(state, GuestReg::Rsp) += 8;
            ret(state);
            return;
        }
        case 1:  // kCallback1
            g_shard_events.push_back("shard0:cb1");
            ret(state);
            return;
        case 2:  // kCallback2
            g_shard_events.push_back("shard0:cb2");
            ret(state);
            return;
        default:
            __playgta6_trap(state, kEntry);
    }
}

void shard1_dispatch(uint32_t local_idx, Pg6GuestState* state) {
    switch (local_idx) {
        case 0: {  // kMid
            g_shard_events.push_back("shard1:mid");
            if (g_shard_scenario == ShardScenario::TrapFault) {
                push(state, 0x1234);  // 8-byte prolog matching .pdata UWOP_ALLOC_SMALL 8
                push(state, kMidCallsLeaf);
                __playgta6_indirect_call(state, kLeaf);
                reg(state, GuestReg::Rsp) += 8;
                ret(state);
                return;
            }
            reg(state, GuestReg::Rax) = reg(state, GuestReg::Rcx) * 2 + 3;
            ret(state);
            return;
        }
        case 1:  // kLeaf
            g_shard_events.push_back("shard1:leaf");
            __playgta6_trap(state, kLeaf + 4);
        case 2:  // kWorker
            g_shard_events.push_back("shard1:worker");
            reg(state, GuestReg::Rax) = reg(state, GuestReg::Rcx) + 100;
            ret(state);
            return;
        default:
            __playgta6_trap(state, kMid);
    }
}

const Pg6ShardDispatchFn kTestShards[] = {&shard0_dispatch, &shard1_dispatch};
const Pg6ShardFunctionEntry kTestShardEntries[] = {
    {kEntry, 0u, 0u},
    {kMid, 1u, 0u},
    {kLeaf, 1u, 1u},
    {kCallback1, 0u, 1u},
    {kCallback2, 0u, 2u},
    {kWorker, 1u, 2u},
};
const Pg6ShardDispatchTable kTestShardTable{
    kPg6ShardDispatchVersion,
    static_cast<uint32_t>(std::size(kTestShards)),
    static_cast<uint64_t>(std::size(kTestShardEntries)),
    kTestShards,
    kTestShardEntries,
};

Run boot_shard_only(ShardScenario scenario) {
    g_shard_scenario = scenario;
    g_shard_events.clear();
    g_shard_helper_result = 0;
    g_shard_last_error = 0;
    os::pgim::MemoryImageReader reader(g_image.data(), g_image.size());
    BootConfig cfg;
    BootModule exe;
    exe.image = &reader;
    exe.name = "boot_test.exe";
    // Explicitly empty lifted_tables: everything resolves through Pg6ShardDispatchTable.
    exe.shard_dispatch_tables.push_back(&kTestShardTable);
    exe.imports = std::span<const BootImport>(kImports);
    cfg.modules.push_back(std::move(exe));
    cfg.arena = g_arena;
    cfg.log_progress = false;
    Run run;
    run.result = run_boot(cfg);
    bool ok = false;
    run.report = parse_json(run.result.report_json, &ok);
    CHECK(ok);
    return run;
}

void test_shard_dispatch_boot() {
    // 1. Direct entry point + TLS callbacks + cross-shard indirect call + synthetic import thunk +
    //    synthetic host callable + call_guest callback on a second thread.
    {
        const Run run = boot_shard_only(ShardScenario::IndirectAndImport);
        CHECK(run.result.status == BootStatus::Exited);
        CHECK(g_shard_helper_result == 41);  // 19 * 2 + 3
        CHECK(g_shard_last_error == 0x2468);
        CHECK(run.result.stats.exit_code == 41 + 42);
        CHECK(run.result.stats.functions_registered == std::size(kTestShardEntries));
        CHECK(run.report["startup"]["lifted_tables"].number == 0);
        CHECK(run.report["startup"]["shard_dispatch_tables"].number == 1);
        CHECK(run.report["startup"]["functions_registered"].number == std::size(kTestShardEntries));
        CHECK(g_shard_events.size() == 4);
        if (g_shard_events.size() == 4) {
            CHECK(g_shard_events[0] == "shard0:cb1" && g_shard_events[1] == "shard0:cb2" &&
                  g_shard_events[2] == "shard0:entry" && g_shard_events[3] == "shard1:mid");
        }

        // call_guest invocation into a shard-dispatched function on a newly prepared thread.
        GuestThreadContext worker{};
        CHECK(prepare_guest_thread(0x10000, 88, worker));
        CHECK(call_guest_args(*g_arena, worker.thread.state, kWorker, uint64_t{23}) == 123);
        CHECK(g_shard_events.back() == "shard1:worker");
        release_guest_thread(worker);

        // Lazy guest thread (CreateThread / pool callback path) dispatches TLS + worker via shard.
        uint64_t lazy_res = 0;
        std::thread host([&] {
            GuestState& state = com::current_guest_state();
            lazy_res = call_guest_args(*g_arena, state, kWorker, uint64_t{55});
        });
        host.join();
        CHECK(lazy_res == 155);
    }

    // 2. Nearest-function fault reporting on unknown_target (`sub_XXXX+0xYY` + fault.detail).
    {
        const Run run = boot_shard_only(ShardScenario::UnknownTargetFault);
        CHECK(run.result.status == BootStatus::Fault);
        check_fault(run, "unknown_target", kUnknownInsideEntry, kEntry);
        CHECK(run.report["fault"]["detail"].hex() == kEntry);
        CHECK(run.result.summary.find("sub_140001000+0x34") != std::string::npos);
    }

    // 3. Nearest-function fault reporting + backtrace on a trap inside a shard-dispatched leaf.
    {
        const Run run = boot_shard_only(ShardScenario::TrapFault);
        CHECK(run.result.status == BootStatus::Fault);
        check_fault(run, "trap", kLeaf + 4, kLeaf);
        CHECK(run.result.summary.find("sub_140001180+0x4") != std::string::npos);
        const Json& bt = run.report["backtrace"];
        CHECK(bt["frames"].items.size() == 2);
        CHECK(bt["frames"][0]["function"].hex() == kMid);
        CHECK(bt["frames"][1]["function"].hex() == kEntry);
    }

    // 4. Malformed shard dispatch table fails registration cleanly with BootStatus::RegisterFailed.
    {
        const Pg6ShardDispatchTable bad_table{
            kPg6ShardDispatchVersion + 99u, 1u, 1u, kTestShards, kTestShardEntries,
        };
        os::pgim::MemoryImageReader reader(g_image.data(), g_image.size());
        BootConfig cfg;
        BootModule exe;
        exe.image = &reader;
        exe.name = "boot_test.exe";
        exe.shard_dispatch_tables.push_back(&bad_table);
        cfg.modules.push_back(std::move(exe));
        cfg.arena = g_arena;
        cfg.log_progress = false;
        const BootResult res = run_boot(cfg);
        CHECK(res.status == BootStatus::RegisterFailed);
        CHECK(res.summary.find("shard dispatch table") != std::string::npos);
    }

    // 4b. Shard dispatch table with synthetic guest VA is rejected cleanly.
    {
        const Pg6ShardFunctionEntry synthetic_entries[] = {{kSyntheticBase + kSyntheticStride, 0u, 0u}};
        const Pg6ShardDispatchTable synth_table{
            kPg6ShardDispatchVersion, 1u, 1u, kTestShards, synthetic_entries,
        };
        os::pgim::MemoryImageReader reader(g_image.data(), g_image.size());
        BootConfig cfg;
        BootModule exe;
        exe.image = &reader;
        exe.name = "boot_test.exe";
        exe.shard_dispatch_tables.push_back(&synth_table);
        cfg.modules.push_back(std::move(exe));
        cfg.arena = g_arena;
        cfg.log_progress = false;
        const BootResult res = run_boot(cfg);
        CHECK(res.status == BootStatus::RegisterFailed);
        CHECK(res.summary.find("synthetic") != std::string::npos);
    }

    // 5. C ABI Pg6GuestBootModule forwards shard_dispatch_tables.
    {
        g_shard_scenario = ShardScenario::IndirectAndImport;
        g_shard_events.clear();
        const Pg6ShardDispatchTable* shard_tables[] = {&kTestShardTable};
        Pg6GuestBootModule module{};
        module.image_bytes = g_image.data();
        module.image_size = g_image.size();
        module.name = "boot_test.exe";
        module.imports = kImports;
        module.import_count = std::size(kImports);
        module.shard_dispatch_tables = shard_tables;
        module.shard_dispatch_table_count = 1;
        Pg6GuestBootConfig cfg{};
        cfg.modules = &module;
        cfg.module_count = 1;
        cfg.quiet = 1;
        Pg6GuestBootStats stats{};
        os::win32::set_vmem_arena(g_arena);
        CHECK(pg6_guest_boot_run(&cfg, &stats) == PG6_BOOT_EXITED);
        CHECK(stats.exit_code == 41 + 42);
        CHECK(stats.functions_registered == std::size(kTestShardEntries));
    }
}

// O36: run options for a boot without a command line, and the report through a sink (the browser
// boot logs it in chunks, which concatenate back to the report).
std::string g_sunk_report;
int g_sink_calls = 0;

void test_boot_options_and_report_sink() {
    BootConfig cfg;
    CHECK(!apply_boot_options_json("", cfg));
    CHECK(!apply_boot_options_json("[1]", cfg));
    CHECK(apply_boot_options_json("{}", cfg));
    CHECK(cfg.max_wall_ms == 0 && !cfg.report_sink);
    CHECK(apply_boot_options_json(
        " {\"max_wall_ms\": 300000, \"max_dispatches\":7,\n \"arena_log_limit\": 12, "
        "\"import_trace_limit\": 5, \"unknown\": \"x\", \"report_to_log\": true}\n", cfg));
    CHECK(cfg.max_wall_ms == 300000 && cfg.max_dispatches == 7);
    CHECK(cfg.arena_log_limit == 12 && cfg.import_trace_limit == 5);
    CHECK(static_cast<bool>(cfg.report_sink));
    CHECK(apply_boot_options_json("{\"report_to_log\": false, \"max_wall_ms\": \"x\"}", cfg));
    CHECK(!cfg.report_sink && cfg.max_wall_ms == 300000);

    // O36: runtime switches default on (the fence publish lag off, #663); each key flips only its own field.
    {
        BootConfig d;
        CHECK(apply_boot_options_json("{}", d));
        CHECK(!d.fence_publish_lag);
        CHECK(apply_boot_options_json("{\"fence_publish_lag\": true}", d) && d.fence_publish_lag);
        CHECK(d.fence_publish_lag && d.defer_gpu_objects && d.xmem_shared_pages && d.present_frame_waits &&
              d.audio_endpoint);
        const char* keys[] = {"fence_publish_lag", "defer_gpu_objects", "xmem_shared_pages",
                              "present_frame_waits", "audio_endpoint"};
        for (int i = 0; i < 5; ++i) {
            BootConfig c;
            c.fence_publish_lag = true;  // every switch on, so each key flips exactly one off
            CHECK(apply_boot_options_json(std::string("{\"") + keys[i] + "\": false}", c));
            const bool v[] = {c.fence_publish_lag, c.defer_gpu_objects, c.xmem_shared_pages,
                              c.present_frame_waits, c.audio_endpoint};
            for (int j = 0; j < 5; ++j) {
                CHECK(v[j] == (i != j));
            }
            CHECK(apply_boot_options_json(std::string("{\"") + keys[i] + "\": \"x\"}", c));  // junk keeps it
            CHECK(!c.fence_publish_lag + !c.defer_gpu_objects + !c.xmem_shared_pages + !c.present_frame_waits +
                      !c.audio_endpoint == 1);
            CHECK(apply_boot_options_json(std::string("{\"") + keys[i] + "\": true}", c));
            CHECK(c.fence_publish_lag && c.defer_gpu_objects && c.xmem_shared_pages && c.present_frame_waits &&
                  c.audio_endpoint);
        }
    }

    // Chunks split on UTF-8 boundaries and concatenate back to the text.
    const std::string text = "{\"a\": \"\xC3\xA9\xE2\x82\xAC\xF0\x9F\x98\x80 \xC3\xA9\"}\n{}";
    std::vector<std::string> lines;
    log_report_json(text, 8, [&](std::string_view line) { lines.emplace_back(line); });
    std::string joined;
    for (size_t i = 0; i < lines.size(); ++i) {
        const std::string head = "[boot-report " + std::to_string(i + 1) + "/" + std::to_string(lines.size()) + "] ";
        CHECK(lines[i].rfind(head, 0) == 0);
        const std::string chunk = lines[i].substr(head.size());
        CHECK(!chunk.empty() && chunk.size() <= 8);
        CHECK((static_cast<unsigned char>(chunk.front()) & 0xC0u) != 0x80u);
        joined += chunk;
    }
    CHECK(lines.size() > 1 && joined == text);

    // The sink receives the finished report, the same text run_boot returns.
    g_sunk_report.clear();
    g_sink_calls = 0;
    Run run = boot_scenario(Scenario::ReturnCode, [](BootConfig& c) {
        c.report_sink = [](std::string_view report) {
            g_sunk_report.assign(report);
            ++g_sink_calls;
        };
    });
    CHECK(g_sink_calls == 1);
    CHECK(g_sunk_report == run.result.report_json);

    // Absent options file: nothing changes; a file that is not an object is reported.
    BootConfig opts;
    CHECK(load_boot_options(opts, [](const std::string&, std::vector<uint8_t>&) { return false; }));
    CHECK(opts.max_wall_ms == 0);
    CHECK(load_boot_options(opts, [](const std::string& path, std::vector<uint8_t>& out) {
        const std::string body = path == "boot/boot-options.json" ? "{\"max_wall_ms\": 5}" : "";
        out.assign(body.begin(), body.end());
        return true;
    }));
    CHECK(opts.max_wall_ms == 5);
    CHECK(!load_boot_options(opts, [](const std::string&, std::vector<uint8_t>& out) {
        out.assign({'n', 'o'});
        return true;
    }));
}

// G71 (#663, #1508): fence_publish_lag_frames, fence_publish_cap_ms, and present_max_interval
// boot options: defaults, JSON parsing, validation, run_boot wiring, and PresentX interval clamping.
void test_fence_lag_and_present_max_interval() {
    // 1. Defaults match G69 (#1508) and console-faithful pacing (0 = no clamp).
    {
        BootConfig cfg;
        CHECK(!cfg.fence_publish_lag);  // R40 (#663): queue fences publish at issue
        CHECK(cfg.fence_publish_lag_frames == 1u);
        CHECK(cfg.fence_publish_cap_ms == 33u);
        CHECK(cfg.present_max_interval == 0u);
    }

    // 2. JSON parsing: valid overrides, zeros, independent fields, no key-prefix collision.
    {
        BootConfig cfg;
        CHECK(apply_boot_options_json(
            "{\"fence_publish_lag_frames\": 2, \"fence_publish_cap_ms\": 50, \"present_max_interval\": 1}",
            cfg));
        CHECK(!cfg.fence_publish_lag);
        CHECK(cfg.fence_publish_lag_frames == 2u);
        CHECK(cfg.fence_publish_cap_ms == 50u);
        CHECK(cfg.present_max_interval == 1u);

        // fence_publish_lag: false disables lag cleanly without corrupting numeric lag fields.
        CHECK(apply_boot_options_json("{\"fence_publish_lag\": false}", cfg));
        CHECK(!cfg.fence_publish_lag);
        CHECK(cfg.fence_publish_lag_frames == 2u);
        CHECK(cfg.fence_publish_cap_ms == 50u);

        // Zero values are accepted for all three uint32 fields.
        CHECK(apply_boot_options_json(
            "{\"fence_publish_lag\": true, \"fence_publish_lag_frames\": 0, "
            "\"fence_publish_cap_ms\": 0, \"present_max_interval\": 0}",
            cfg));
        CHECK(cfg.fence_publish_lag);
        CHECK(cfg.fence_publish_lag_frames == 0u);
        CHECK(cfg.fence_publish_cap_ms == 0u);
        CHECK(cfg.present_max_interval == 0u);

        // Max uint32 boundary is accepted.
        CHECK(apply_boot_options_json(
            "{\"fence_publish_lag_frames\": 4294967295, \"fence_publish_cap_ms\": 4294967295, "
            "\"present_max_interval\": 4294967295}",
            cfg));
        CHECK(cfg.fence_publish_lag_frames == 4294967295u);
        CHECK(cfg.fence_publish_cap_ms == 4294967295u);
        CHECK(cfg.present_max_interval == 4294967295u);
    }

    // 3. Validation: negative, string, bool, float, uint32 overflow, and uint64 overflow leave fields untouched.
    {
        BootConfig cfg;
        cfg.fence_publish_lag_frames = 2u;
        cfg.fence_publish_cap_ms = 40u;
        cfg.present_max_interval = 1u;

        const char* bad_inputs[] = {
            "{\"fence_publish_lag_frames\": -1, \"fence_publish_cap_ms\": -5, \"present_max_interval\": -2}",
            "{\"fence_publish_lag_frames\": \"2\", \"fence_publish_cap_ms\": \"16\", \"present_max_interval\": \"1\"}",
            "{\"fence_publish_lag_frames\": true, \"fence_publish_cap_ms\": false, \"present_max_interval\": null}",
            "{\"fence_publish_lag_frames\": 1.5, \"fence_publish_cap_ms\": 33.3, \"present_max_interval\": 1.0}",
            "{\"fence_publish_lag_frames\": 4294967296, \"fence_publish_cap_ms\": 4294967296, \"present_max_interval\": 4294967296}",
            "{\"fence_publish_lag_frames\": 99999999999999999999999, \"fence_publish_cap_ms\": 99999999999999999999999, \"present_max_interval\": 99999999999999999999999}",
        };
        for (const char* json : bad_inputs) {
            CHECK(apply_boot_options_json(json, cfg));
            CHECK(cfg.fence_publish_lag_frames == 2u);
            CHECK(cfg.fence_publish_cap_ms == 40u);
            CHECK(cfg.present_max_interval == 1u);
        }
    }

    // 4. run_boot wiring to DeviceOptions and xbox::present_max_interval().
    {
        gpu::d3d12_xs_exports::reset_for_tests();
        const Run r1 = boot_scenario(Scenario::ReturnCode, [](BootConfig& c) {
            CHECK(apply_boot_options_json(
                "{\"fence_publish_lag\": true, \"fence_publish_lag_frames\": 2, \"fence_publish_cap_ms\": 66, "
                "\"present_max_interval\": 1}",
                c));
        });
        CHECK(r1.result.status == BootStatus::Exited);
        const gpu::d3d12_guest::DeviceOptions opts1 = gpu::d3d12_xs_exports::device_options();
        CHECK(opts1.fence_publish_lag);
        CHECK(opts1.fence_publish_lag_frames == 2u);
        CHECK(opts1.fence_publish_cap_ms == 66u);
        CHECK(gpu::d3d12_guest::xbox::present_max_interval() == 1u);

        // fence_publish_lag: false disables lag cleanly on DeviceOptions and Device even with lag_frames > 0.
        const Run r2 = boot_scenario(Scenario::ReturnCode, [](BootConfig& c) {
            CHECK(apply_boot_options_json(
                "{\"fence_publish_lag\": false, \"fence_publish_lag_frames\": 2, \"fence_publish_cap_ms\": 50}",
                c));
        });
        CHECK(r2.result.status == BootStatus::Exited);
        const gpu::d3d12_guest::DeviceOptions opts2 = gpu::d3d12_xs_exports::device_options();
        CHECK(!opts2.fence_publish_lag);
        CHECK(opts2.fence_publish_lag_frames == 2u);
        CHECK(opts2.fence_publish_cap_ms == 50u);
        CHECK(gpu::d3d12_guest::xbox::present_max_interval() == 0u);
        {
            gpu::d3d12_guest::DeviceOptions dev_opts = opts2;
            dev_opts.sink = [](const uint8_t*, size_t) {};
            gpu::d3d12_guest::Device dev(dev_opts);
            CHECK(!dev.fence_publish_lag());
            CHECK(dev.fence_publish_lag_frames() == 0u);
            CHECK(dev.fence_publish_cap_ms() == 50u);
            dev.shutdown_frame_events();
        }

        // Subsequent default boot restores lag_frames=1, cap_ms=33, present_max_interval=0.
        gpu::d3d12_xs_exports::reset_for_tests();
        const Run r3 = boot_scenario(Scenario::ReturnCode);
        CHECK(r3.result.status == BootStatus::Exited);
        const gpu::d3d12_guest::DeviceOptions opts3 = gpu::d3d12_xs_exports::device_options();
        CHECK(!opts3.fence_publish_lag);
        CHECK(opts3.fence_publish_lag_frames == 1u);
        CHECK(opts3.fence_publish_cap_ms == 33u);
        CHECK(gpu::d3d12_guest::xbox::present_max_interval() == 0u);
    }

    // 5. CommandQueue::PresentX interval clamping right before present_pacer().present(...).
    {
        namespace dg = gpu::d3d12_guest;
        com::set_guest_memory(g_arena);
        gpu::present_pacer().reset_for_testing();
        gpu::present_pacer().set_enabled(true);

        auto now = gpu::PresentPacer::Clock::time_point{};
        gpu::present_pacer().set_clock_for_testing(
            [&] { return now; },
            [&](gpu::PresentPacer::Clock::time_point target) { now = target; },
            now);

        dg::DeviceOptions opts;
        opts.sink = [](const uint8_t*, size_t) {};
        opts.fence_publish_lag = false;
        com::Ref<dg::Device> dev = com::create_object<dg::Device>(com::implements(dg::kID3D12Device), opts);
        CHECK(static_cast<bool>(dev));
        dev->initialize_guest_payload();

        dg::GuestCommandQueueDesc qdesc{};
        com::Ref<dg::CommandQueue> queue =
            com::create_object<dg::CommandQueue>(com::implements(dg::kID3D12CommandQueue), dev, qdesc);
        CHECK(static_cast<bool>(queue));

        // Title requests 30 Hz (33333 us -> 2 vblanks on a 60 Hz display).
        CHECK(dev->SetFrameIntervalX(0, 33333u, 1u, 0u) == dg::kSOk);
        CHECK(dev->present_interval_vblanks() == 2u);
        // The title drives the frame interval, so presents add no 60 Hz wait on top of it.
        CHECK(gpu::present_pacer().title_driven());
        // Interval clamping below is covered with the display pacing on.
        gpu::present_pacer().set_title_driven(false);

        const uint64_t guest_page = g_arena->reserve(0, os::kPageSize, os::PageProtect::ReadWrite);
        CHECK(guest_page != 0 && g_arena->commit(guest_page, os::kPageSize, os::PageProtect::ReadWrite));
        dg::xbox::GuestPresentPlaneX plane{};
        CHECK(g_arena->write(guest_page, &plane, sizeof(plane)));
        const uint64_t params_va = guest_page + 0x100;
        const uint32_t non_immediate = 0u;
        const uint32_t immediate = 1u;
        CHECK(g_arena->write(params_va, &non_immediate, sizeof(non_immediate)));

        // Default present_max_interval = 0 -> no clamp, honors title's 2-vblank (30 Hz) interval.
        dg::xbox::set_present_max_interval(0u);
        CHECK(queue->PresentX(1u, guest_page, params_va) == dg::kSOk);
        CHECK(gpu::present_pacer().stats().last_interval == 2u);
        const auto t_after_first = now;
        CHECK(queue->PresentX(1u, guest_page, params_va) == dg::kSOk);
        CHECK(gpu::present_pacer().stats().last_interval == 2u);
        CHECK(std::chrono::duration_cast<std::chrono::nanoseconds>(now - t_after_first).count() ==
              2 * gpu::PresentPacer::kVblankNs);

        // present_max_interval = 1 -> clamps 2-vblank title interval to 1 vblank (60 Hz cap).
        dg::xbox::set_present_max_interval(1u);
        const auto t_before_clamp = now;
        CHECK(queue->PresentX(1u, guest_page, params_va) == dg::kSOk);
        CHECK(gpu::present_pacer().stats().last_interval == 1u);
        CHECK(std::chrono::duration_cast<std::chrono::nanoseconds>(now - t_before_clamp).count() ==
              1 * gpu::PresentPacer::kVblankNs);

        // present_max_interval = 3 (> title's 2) -> keeps title's 2-vblank interval.
        dg::xbox::set_present_max_interval(3u);
        const auto t_before_higher = now;
        CHECK(queue->PresentX(1u, guest_page, params_va) == dg::kSOk);
        CHECK(gpu::present_pacer().stats().last_interval == 2u);
        CHECK(std::chrono::duration_cast<std::chrono::nanoseconds>(now - t_before_higher).count() ==
              2 * gpu::PresentPacer::kVblankNs);

        // Immediate present (parameters[0] != 0) stays interval 0 even when present_max_interval = 1.
        CHECK(g_arena->write(params_va, &immediate, sizeof(immediate)));
        dg::xbox::set_present_max_interval(1u);
        const auto t_before_imm = now;
        CHECK(queue->PresentX(1u, guest_page, params_va) == dg::kSOk);
        CHECK(gpu::present_pacer().stats().last_interval == 0u);
        CHECK(now == t_before_imm);

        // Clean up global test state.
        dg::xbox::set_present_max_interval(0u);
        dev->shutdown_frame_events();
        queue.reset();
        dev.reset();
        g_arena->release(guest_page);
        com::reset_com_bridge();
        gpu::d3d12_xs_exports::reset_for_tests();
        gpu::present_pacer().reset_for_testing();
    }
}

// O42 (#663): per-thread dispatch batching and lock-free post-present import observer.
void test_lockfree_dispatch_observer() {
    // 1. Batched dispatch flushing across multiple 256-call batches + remainder, mid-run
    //    live_counters() flush, and post-present recent/thread_last accuracy.
    {
        g_o42 = O42Observed{};
        const Run run = boot_scenario(Scenario::O42PostPresentBatch);
        CHECK(run.result.status == BootStatus::Exited);
        // 2 TLS callbacks + 1 entry + 3 pre-present calls = 6 dispatches at mid-run check.
        CHECK(g_o42.mid_dispatches == 6);
        // Total dispatches: 6 + 600 post-present calls = 606 exact.
        CHECK(run.result.stats.dispatches == 606);
        CHECK(run.report["counters"]["dispatches"].number == 606);
        CHECK(run.result.stats.import_calls == 603);
        CHECK(run.report["counters"]["import_calls"].number == 603);
        // First present counts only the 3 pre-present calls.
        const Json& fp = run.report["first_present"];
        CHECK(fp.type == Json::Object);
        CHECK(fp["import_seq"].number == 3);
        uint64_t pre_total = 0;
        for (const Json& row : fp["imports"].items) {
            pre_total += static_cast<uint64_t>(row["count"].number);
        }
        CHECK(pre_total == 3);
        // Recent ring has the last 32 post-present calls (seq 572..603) in ascending order.
        const Json& recent = run.report["counters"]["recent"];
        CHECK(recent.items.size() == 32);
        if (recent.items.size() == 32) {
            CHECK(recent[0]["seq"].number == 572);
            CHECK(recent[31]["seq"].number == 603);
            // Call i = 599 is odd -> GetLastError; i = 598 is even -> SetLastError(0x1000 + 598).
            CHECK(recent[31]["symbol"].text == "GetLastError");
            CHECK(recent[30]["symbol"].text == "SetLastError");
            CHECK(recent[30]["args"][0].hex() == 0x1000u + 598u);
        }
        // thread_last and live guest_thread_last_import reflect the final post-present call.
        CHECK(g_o42.live_last_import == "KERNEL32.dll!GetLastError");
        CHECK(guest_thread_last_import(g_o42.main_tid) == "KERNEL32.dll!GetLastError");
        const Json& threads = run.report["threads"];
        CHECK(threads.items.size() == 1);
        CHECK(threads[0]["last_import"]["seq"].number == 603);
        CHECK(threads[0]["last_import"]["symbol"].text == "GetLastError");
        CHECK(run.result.summary.find("#603 KERNEL32.dll!GetLastError") != std::string::npos);
    }

    // 2. Fault after present_seen with < 256 pending dispatches flushes exact count and preserves
    //    both pre-present and post-present recent/thread_last entries.
    {
        g_o42 = O42Observed{};
        const Run run = boot_scenario(Scenario::O42PostPresentFault);
        CHECK(run.result.status == BootStatus::Fault);
        CHECK(run.report["fault"]["kind"].text == "unimplemented");
        // 2 TLS callbacks + 1 entry + 3 import calls = 6 dispatches.
        CHECK(run.result.stats.dispatches == 6);
        CHECK(run.report["counters"]["dispatches"].number == 6);
        CHECK(run.result.stats.import_calls == 3);
        const Json& recent = run.report["counters"]["recent"];
        CHECK(recent.items.size() == 3);
        if (recent.items.size() == 3) {
            CHECK(recent[0]["seq"].number == 1 && recent[0]["args"][0].hex() == 0xAA);
            CHECK(recent[1]["seq"].number == 2 && recent[1]["args"][0].hex() == 0xBB);
            CHECK(recent[2]["seq"].number == 3 && recent[2]["symbol"].text == "GetLastError");
        }
        const Json& threads = run.report["threads"];
        CHECK(threads.items.size() == 1);
        CHECK(threads[0]["last_import"]["seq"].number == 3);
        CHECK(threads[0]["last_import"]["symbol"].text == "GetLastError");
        CHECK(guest_thread_last_import(g_o42.main_tid) == "KERNEL32.dll!GetLastError");
    }

    // 3. Multi-threaded concurrent synthetic import calls after present_seen: exact dispatch &
    //    import counts across all threads (including thread-exit batch flushes) and accurate
    //    per-thread last_import + merged recent ring.
    {
        const std::vector<uint8_t> saved = g_image;
        // Empty TLS callbacks so worker threads do not touch non-thread-safe test callback globals.
        g_image = build_image(true, kBase, {});
        g_o42 = O42Observed{};
        const Run run = boot_scenario(Scenario::O42PostPresentConcurrent);
        g_image = saved;

        CHECK(run.result.status == BootStatus::Exited);
        // 1 entry + 2 pre-present + 100 main post-present + 4 * 300 worker post-present = 1303 dispatches.
        CHECK(run.result.stats.dispatches == 1303);
        CHECK(run.report["counters"]["dispatches"].number == 1303);
        CHECK(run.result.stats.import_calls == 1302);
        CHECK(run.report["counters"]["import_calls"].number == 1302);
        uint64_t sum_imports = 0;
        for (const Json& row : run.report["counters"]["imports"].items) {
            sum_imports += static_cast<uint64_t>(row["count"].number);
        }
        CHECK(sum_imports == 1302);
        // First present saw only the 2 pre-present calls.
        CHECK(run.report["first_present"]["import_seq"].number == 2);
        // Merged recent ring has exact 32 newest calls (seq 1271..1302), strictly ascending.
        const Json& recent = run.report["counters"]["recent"];
        CHECK(recent.items.size() == 32);
        if (recent.items.size() == 32) {
            CHECK(recent[0]["seq"].number == 1271);
            CHECK(recent[31]["seq"].number == 1302);
            for (size_t i = 1; i < recent.items.size(); ++i) {
                CHECK(recent[i]["seq"].number == recent[i - 1]["seq"].number + 1);
            }
        }
        // All 5 threads (1 main + 4 workers) report exact post-present last_import.
        const Json& threads = run.report["threads"];
        CHECK(threads.items.size() == 5);
        std::map<uint32_t, std::string> expected_symbol;
        expected_symbol[g_o42.main_tid] = "SetLastError";
        for (size_t w = 0; w < g_o42.worker_tids.size(); ++w) {
            expected_symbol[g_o42.worker_tids[w]] = ((w & 1u) == 0) ? "GetLastError" : "SetLastError";
        }
        for (const Json& t : threads.items) {
            const auto tid = static_cast<uint32_t>(t["thread_id"].number);
            auto it = expected_symbol.find(tid);
            CHECK(it != expected_symbol.end());
            if (it != expected_symbol.end()) {
                CHECK(t["last_import"].type == Json::Object);
                CHECK(t["last_import"]["seq"].number > 2);
                CHECK(t["last_import"]["symbol"].text == it->second);
                CHECK(guest_thread_last_import(tid) == "KERNEL32.dll!" + it->second);
            }
        }
    }
}

#ifndef __EMSCRIPTEN__
struct ScopedStderrCapture {
    int saved_fd = -1;
    std::filesystem::path temp_path;

    ScopedStderrCapture() {
        std::fflush(stderr);
        const auto tick = std::chrono::steady_clock::now().time_since_epoch().count();
        temp_path = std::filesystem::temp_directory_path() /
                    ("playgta6_fault_log_test_" + std::to_string(tick) + ".log");
        FILE* f = std::fopen(temp_path.string().c_str(), "wb");
        if (f == nullptr) {
            return;
        }
#ifdef _WIN32
        saved_fd = _dup(_fileno(stderr));
        if (saved_fd >= 0) {
            _dup2(_fileno(f), _fileno(stderr));
        }
#else
        saved_fd = dup(fileno(stderr));
        if (saved_fd >= 0) {
            dup2(fileno(f), fileno(stderr));
        }
#endif
        std::fclose(f);
    }

    std::string read_flushed() const {
        std::string content;
        if (temp_path.empty()) {
            return content;
        }
        if (FILE* f = std::fopen(temp_path.string().c_str(), "rb")) {
            char buf[4096];
            size_t n = 0;
            while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) {
                content.append(buf, n);
            }
            std::fclose(f);
        }
        return content;
    }

    std::string finish() {
        if (saved_fd >= 0) {
            std::fflush(stderr);
#ifdef _WIN32
            _dup2(saved_fd, _fileno(stderr));
            _close(saved_fd);
#else
            dup2(saved_fd, fileno(stderr));
            close(saved_fd);
#endif
            saved_fd = -1;
        }
        std::string content = read_flushed();
        if (!temp_path.empty()) {
            std::error_code ec;
            std::filesystem::remove(temp_path, ec);
            temp_path.clear();
        }
        return content;
    }

    ~ScopedStderrCapture() {
        if (saved_fd >= 0 || !temp_path.empty()) {
            (void)finish();
        }
    }
};

std::string find_log_line(const std::string& logs, const std::string& prefix) {
    size_t pos = 0;
    while (pos < logs.size()) {
        size_t end = logs.find('\n', pos);
        if (end == std::string::npos) {
            end = logs.size();
        }
        std::string line = logs.substr(pos, end - pos);
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.rfind(prefix, 0) == 0) {
            return line;
        }
        pos = end + 1;
    }
    return {};
}
#endif

// #663: sanitized `[fault]` backward compatibility + `callers=` (up to 4 frames) + `[fault-regs]`
// integer GPR summary, including off-thread trap flush before `off_thread_fault`.
void test_fault_log_gprs_and_callers() {
#ifndef __EMSCRIPTEN__
    const std::vector<uint8_t> saved = g_image;

    // 1. 5-frame trap without .pdata (stack-scan-call-verified): verify `[fault]` keeps existing
    //    fields intact, appends `callers=` capped at 4 frames, and logs `[fault-regs]` with exact
    //    integer register values.
    {
        g_image = build_image(true, kBase, {kCallback1, kCallback2}, 0xA0, /*with_pdata=*/false);
        ScopedStderrCapture capture;
        const Run run = boot_scenario(Scenario::DeepTrapWithRegs, [](BootConfig& cfg) {
            cfg.log_progress = true;
            cfg.fault_record = true;
        });
        const std::string logs = capture.finish();
        g_image = saved;

        CHECK(run.result.status == BootStatus::Fault);
        CHECK(run.report["backtrace"]["frames"].items.size() == 5);
        const std::string fault_line = find_log_line(logs, "[fault] ");
        const std::string regs_line = find_log_line(logs, "[fault-regs] ");
        CHECK(!fault_line.empty());
        CHECK(!regs_line.empty());

        // Backward-compatible fields on `[fault]`.
        CHECK(fault_line.find("[fault] trap rip=0x") == 0);
        CHECK(fault_line.find(" target=0x140001184 fn=sub_140001180+0x4 thread=") != std::string::npos);
        CHECK(fault_line.find("ms caller=sub_140001100+0x20 ") != std::string::npos);
        // Up to 4 frames in `callers=`, 5th frame (`sub_140001000+0x30`) omitted.
        CHECK(fault_line.find(
                  " callers=sub_140001100+0x20,sub_140001000+0x10,sub_140001100+0x20,sub_140001000+0x10") !=
              std::string::npos);
        CHECK(fault_line.find("sub_140001000+0x30") == std::string::npos);

        // Exact `[fault-regs]` ordering & values (including captured rsp from report).
        const uint64_t expected_rsp = run.report["registers"]["rsp"].hex();
        char expected_regs[384];
        std::snprintf(expected_regs, sizeof(expected_regs),
                      "[fault-regs] rax=0x1111222233334444 rcx=0x2222333344445555 rdx=0x3333444455556666"
                      " r8=0x88889999aaaabbbb r9=0x9999aaaabbbbcccc rsp=0x%" PRIx64
                      " rbx=0xbbbbccccddddeeee rsi=0x5151515151515151 rdi=0xd1d1d1d1d1d1d1d1",
                      expected_rsp);
        CHECK(regs_line == expected_regs);
    }

    // 2. Standard 2-frame `.pdata` trap: `caller=` and 2-entry `callers=` + `[fault-regs]`.
    {
        ScopedStderrCapture capture;
        const Run run = boot_scenario(Scenario::Trap, [](BootConfig& cfg) {
            cfg.log_progress = true;
            cfg.fault_record = true;
        });
        const std::string logs = capture.finish();
        CHECK(run.result.status == BootStatus::Fault);
        const std::string fault_line = find_log_line(logs, "[fault] ");
        const std::string regs_line = find_log_line(logs, "[fault-regs] ");
        CHECK(fault_line.find(" caller=sub_140001100+0x20 callers=sub_140001100+0x20,sub_140001000+0x10") !=
              std::string::npos);
        CHECK(regs_line.rfind("[fault-regs] rax=0x", 0) == 0);
        // The compact `[fault-record]` line comes after the report; the kind's first char is \u-escaped
        // so web/telemetry/boot_tracer.js does not read the bare word "trap" as a wasm trap.
        const std::string brief = find_log_line(logs, "[fault-record] ");
        CHECK(brief.find("{\"kind\": \"\\u0074rap\"") != std::string::npos);
        CHECK(brief.find("\"unimplemented_slot\": null") != std::string::npos);
        CHECK(brief.find("\"callers\": [") != std::string::npos);
    }

    // 3. Off-thread trap: both `[fault]` and `[fault-regs]` must be flushed BEFORE `off_thread_fault` runs.
    {
        g_image = build_image(true, kBase, {}, 0xA0, /*with_pdata=*/false);
        g_off_thread_reported = false;
        g_off_thread_report.clear();
        static ScopedStderrCapture* active_capture = nullptr;
        static std::string flushed_before_hook;
        flushed_before_hook.clear();
        ScopedStderrCapture capture;
        active_capture = &capture;
        const Run run = boot_scenario(Scenario::OffThreadTrapWithRegs, [](BootConfig& cfg) {
            cfg.log_progress = true;
            cfg.fault_record = true;
            cfg.off_thread_fault = [] {
                if (active_capture != nullptr) {
                    flushed_before_hook = active_capture->read_flushed();
                }
                g_off_thread_report = last_report_json();
                g_off_thread_reported = true;
                for (;;) {
                    std::this_thread::sleep_for(std::chrono::hours(1));
                }
            };
        });
        active_capture = nullptr;
        (void)capture.finish();
        g_image = saved;
        (void)run;

        CHECK(g_off_thread_reported.load());
        const std::string fault_line = find_log_line(flushed_before_hook, "[fault] ");
        const std::string regs_line = find_log_line(flushed_before_hook, "[fault-regs] ");
        CHECK(!fault_line.empty());
        CHECK(!regs_line.empty());
        CHECK(!find_log_line(flushed_before_hook, "[fault-record] ").empty());
        CHECK(fault_line.find(" target=0x140001184 fn=sub_140001180+0x4 thread=") != std::string::npos);
        CHECK(fault_line.find(" caller=sub_140001100+0x20 callers=sub_140001100+0x20,sub_140001000+0x10") !=
              std::string::npos);
        CHECK(regs_line.find("rax=0x1010101010101010 rcx=0x2020202020202020 rdx=0x3030303030303030"
                             " r8=0x8080808080808080 r9=0x9090909090909090 rsp=0x") != std::string::npos);
        CHECK(regs_line.find(" rbx=0xb0b0b0b0b0b0b0b0 rsi=0x5050505050505050 rdi=0xd0d0d0d0d0d0d0d0") !=
              std::string::npos);
    }

    // 4. `fault_record = false`: neither `[fault]` nor `[fault-regs]` is emitted.
    {
        ScopedStderrCapture capture;
        const Run run = boot_scenario(Scenario::Trap, [](BootConfig& cfg) {
            cfg.log_progress = true;
            cfg.fault_record = false;
        });
        const std::string logs = capture.finish();
        CHECK(run.result.status == BootStatus::Fault);
        CHECK(find_log_line(logs, "[fault] ").empty());
        CHECK(find_log_line(logs, "[fault-regs] ").empty());
    }

    // 5. Fault without guest registers (`report_host_fault` with no active thread): `[fault]` has
    //    `rip=0x0` and no `[fault-regs]` line is emitted.
    {
        const Run run = boot_scenario(Scenario::ReturnCode, [](BootConfig& cfg) {
            cfg.log_progress = true;
            cfg.fault_record = true;
        });
        CHECK(run.result.status == BootStatus::Exited);
        GuestThreadContext* prev = activate_guest_thread(nullptr);
        ScopedStderrCapture capture;
        const auto host_mid = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(&sub_mid));
        CHECK(report_host_fault("host_access_violation", 0xDEAD0000, host_mid));
        const std::string logs = capture.finish();
        activate_guest_thread(prev);

        const std::string fault_line = find_log_line(logs, "[fault] ");
        const std::string regs_line = find_log_line(logs, "[fault-regs] ");
        CHECK(fault_line.find("[fault] host_access_violation rip=0x0 target=0x140001100 fn=sub_140001100+0x0") == 0);
        CHECK(regs_line.empty());
    }
#endif
}

}  // namespace

void test_minidump_paths() {
    namespace md = playgta6::guest::minidump;
    CHECK(md::select_path("C:/runs/a/boot-report.json", "host", "host_access_violation", 42) ==
          "C:/runs/a/host-fault-host_access_violation-42.dmp");
    CHECK(md::select_path("boot-report.json", "guest", "a b/c", 7) == "guest-fault-a_b_c-7.dmp");
    CHECK(md::select_path("out/r.json", "host", "", 1) == "out/host-fault-unknown-1.dmp");
    const std::string r = "{\"fault_record\": {\"kind\": \"x\"}}";
    CHECK(md::annotate_report(r, "C:/d.dmp") == "{\"fault_record\": {\"minidump\": \"C:/d.dmp\", \"kind\": \"x\"}}");
    CHECK(md::annotate_report("{}", "p") == "{}");
    CHECK(md::annotate_report(r, "") == r);
}

void test_dump_pipelines_dir_and_present_interval_boot_options() {
    // 1. Defaults and valid JSON string parsing for dump_pipelines_dir and present_interval.
    {
        BootConfig cfg;
        CHECK(cfg.dump_pipelines_dir.empty());
        CHECK(!cfg.present_honor_interval);

        CHECK(apply_boot_options_json(
            "{\"dump_pipelines_dir\": \"/pg6_dump\", \"present_interval\": \"honor\"}", cfg));
        CHECK(cfg.dump_pipelines_dir == "/pg6_dump");
        CHECK(cfg.present_honor_interval);

        CHECK(apply_boot_options_json("{\"present_interval\": \"min1\"}", cfg));
        CHECK(!cfg.present_honor_interval);
        CHECK(cfg.dump_pipelines_dir == "/pg6_dump");

        // Escaped path characters (including escaped slashes and \u002f) decode cleanly.
        CHECK(apply_boot_options_json(
            "{\"dump_pipelines_dir\": \"\\u002fpg6_dump\\/sub\"}", cfg));
        CHECK(cfg.dump_pipelines_dir == "/pg6_dump/sub");

        // Empty string, non-string types, and malformed/unterminated strings leave the field untouched.
        const char* bad_inputs[] = {
            "{\"dump_pipelines_dir\": \"\"}",
            "{\"dump_pipelines_dir\": 123}",
            "{\"dump_pipelines_dir\": true}",
            "{\"dump_pipelines_dir\": null}",
            "{\"dump_pipelines_dir\": \"/unterminated}",
            "{\"present_interval\": \"invalid\"}",
            "{\"present_interval\": 2}",
        };
        for (const char* json : bad_inputs) {
            CHECK(apply_boot_options_json(json, cfg));
            CHECK(cfg.dump_pipelines_dir == "/pg6_dump/sub");
            CHECK(!cfg.present_honor_interval);
        }
    }

    // 2. load_boot_options reads dump_pipelines_dir from boot/boot-options.json.
    {
        BootConfig cfg;
        CHECK(load_boot_options(cfg, [](const std::string& path, std::vector<uint8_t>& out) {
            if (path != "boot/boot-options.json") {
                return false;
            }
            const std::string body = "{\"dump_pipelines_dir\": \"/pg6_dump\", \"report_to_log\": true}";
            out.assign(body.begin(), body.end());
            return true;
        }));
        CHECK(cfg.dump_pipelines_dir == "/pg6_dump");
        CHECK(static_cast<bool>(cfg.report_sink));
    }

    // 3. run_boot wires dump_pipelines_dir into DeviceOptions::dump_dir and preserves pre-existing
    //    DeviceOptions::dump_dir when BootConfig::dump_pipelines_dir is empty.
    {
        gpu::d3d12_xs_exports::reset_for_tests();
        const Run r1 = boot_scenario(Scenario::ReturnCode, [](BootConfig& c) {
            CHECK(apply_boot_options_json("{\"dump_pipelines_dir\": \"/pg6_dump\"}", c));
        });
        CHECK(r1.result.status == BootStatus::Exited);
        CHECK(gpu::d3d12_xs_exports::device_options().dump_dir == "/pg6_dump");

        // Empty BootConfig::dump_pipelines_dir must not clear a harness-configured dump_dir.
        gpu::d3d12_xs_exports::reset_for_tests();
        {
            gpu::d3d12_guest::DeviceOptions pre;
            pre.dump_dir = "/preconfigured_dump";
            gpu::d3d12_xs_exports::set_device_options(pre);
        }
        const Run r2 = boot_scenario(Scenario::ReturnCode);
        CHECK(r2.result.status == BootStatus::Exited);
        CHECK(gpu::d3d12_xs_exports::device_options().dump_dir == "/preconfigured_dump");
        gpu::d3d12_xs_exports::reset_for_tests();
    }
}

int main() {
    test_minidump_paths();
#ifdef __EMSCRIPTEN__
    // The default wasm arena is 2 GiB ([4, 6) GiB) but the helper DLL is mapped at 6 GiB.
    VirtualMemoryArena arena(os::kDefaultArenaBase, uint64_t{4} << 30);
#else
    VirtualMemoryArena arena;
#endif
    g_arena = &arena;
    os::win32::set_vmem_arena(&arena);
    g_image = build_image();

    test_layout_offsets();
    test_clean_boot_and_teb();
    test_exit_paths();
    test_fault_reports();
    test_survey_mode();
    test_last_error_shared_with_teb();
    test_observability();
    test_off_thread_fault();
    test_report_import_fields();
    test_host_callable_statistics();
    test_off_thread_unresolved_import();
    test_second_thread();
    test_no_tls_and_load_failure();
    test_two_modules();
    test_host_fault_mapping();
    test_shard_dispatch_boot();
    test_boot_options_and_report_sink();
    test_fence_lag_and_present_max_interval();
    test_dump_pipelines_dir_and_present_interval_boot_options();
    test_lockfree_dispatch_observer();
    test_fault_log_gprs_and_callers();
    test_wasm_off_thread_fault_records();

    reset_boot();
    os::win32::set_vmem_arena(nullptr);
    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("test_guest_boot: all checks passed\n");
    return 0;
}
