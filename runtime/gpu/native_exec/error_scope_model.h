// Diagnostic error-scope attribution (--gpu-error-scopes): Dawn uncaptured device errors carry no shader or
// resource context, so the executor wraps each packet (pipeline creation, bind group creation, pass
// commands, the present's encode + submit) in Push/PopErrorScope and records the error with the phase, the
// present index and the shader hashes / resource ids in scope. This header is the pure part: the phase of
// an opcode, the on/off policy and the bounded, deduplicated log. No GPU involved.
#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "abi/playgta6_abi.h"

namespace playgta6::gpu::native_exec {

inline constexpr size_t kErrorScopeMaxEntries = 64;     // distinct (phase, message, context) rows kept
inline constexpr size_t kErrorScopeMaxMessages = 128;   // distinct messages counted
inline constexpr uint64_t kErrorScopeAutoErrors = 200;  // auto mode stops after this many scoped errors
inline constexpr uint64_t kErrorScopeAutoPresents = 120;  // auto mode stays armed this long after an error

// The phase an opcode belongs to: where the error was raised.
inline const char* error_scope_phase(uint16_t opcode) {
    switch (static_cast<GpuOpcode>(opcode)) {
        case GpuOpcode::CreateRenderPipeline:
        case GpuOpcode::CreateComputePipeline:
        case GpuOpcode::CreateRenderPipeline2:
        case GpuOpcode::CreateComputePipeline2:
        case GpuOpcode::CreatePipelineLayout:
        case GpuOpcode::CreateShaderModule:
            return "pipeline_create";
        case GpuOpcode::CreateBindGroup:
        case GpuOpcode::CreateBindGroup2:
        case GpuOpcode::CreateBindGroupLayout:
            return "bind_group_create";
        case GpuOpcode::Present:
            return "present_submit";  // the frame's passes are encoded, finished and submitted here
        case GpuOpcode::BeginRenderPass:
        case GpuOpcode::EndRenderPass:
        case GpuOpcode::Draw:
        case GpuOpcode::DrawIndexed:
        case GpuOpcode::DrawIndirect:
        case GpuOpcode::DrawIndexedIndirect:
        case GpuOpcode::DispatchCompute:
        case GpuOpcode::DispatchIndirect:
        case GpuOpcode::SetPipeline:
        case GpuOpcode::SetComputePipeline:
        case GpuOpcode::SetBindGroup:
        case GpuOpcode::SetBindGroup2:
            return "pass";
        default:
            return "other";
    }
}

// Scoping is explicit (--gpu-error-scopes), or armed automatically by a device error nothing attributed yet
// (an uncaptured one on the executor thread, or a scoped one) and kept armed for kErrorScopeAutoPresents
// presents after the latest such error, disarmed for good after kErrorScopeAutoErrors scoped errors. Errors
// an object creation already caught in its own scope (created_valid: counted with their opcode) and shader
// worker errors (scopes are per thread) do not arm it, so a run with only those pays nothing per packet.
inline bool error_scope_active(bool explicit_on, uint64_t unattributed_errors, uint64_t last_error_present,
                               uint64_t present, uint64_t scoped_errors) {
    if (explicit_on) return true;
    return unattributed_errors > 0 && present <= last_error_present + kErrorScopeAutoPresents &&
           scoped_errors < kErrorScopeAutoErrors;
}

struct ErrorScopeEntry {
    std::string message;
    std::string kind;  // validation / out-of-memory / internal
    std::string phase;
    std::string context;  // "vs=<hash> fs=<hash> pipeline=<id> bind_group=<id> ..."
    uint64_t present_index = 0;
    uint64_t packet = 0;
    uint16_t opcode = 0;
    uint64_t count = 0;
};

class ErrorScopeLog {
public:
    void record(const std::string& kind, const std::string& message, const char* phase, const std::string& context,
                uint64_t present_index, uint64_t packet, uint16_t opcode) {
        std::lock_guard<std::mutex> lock(mu_);
        ++total_;
        std::string msg = message.size() > 400 ? message.substr(0, 400) : message;
        if (by_message_.size() < kErrorScopeMaxMessages || by_message_.count(msg)) ++by_message_[msg];
        else ++messages_dropped_;
        for (ErrorScopeEntry& e : entries_) {
            if (e.message == msg && e.phase == phase && e.context == context) {
                ++e.count;
                return;
            }
        }
        if (entries_.size() >= kErrorScopeMaxEntries) {
            ++entries_dropped_;
            return;
        }
        ErrorScopeEntry e;
        e.message = std::move(msg);
        e.kind = kind;
        e.phase = phase;
        e.context = context;
        e.present_index = present_index;
        e.packet = packet;
        e.opcode = opcode;
        e.count = 1;
        entries_.push_back(std::move(e));
    }

    uint64_t total() const {
        std::lock_guard<std::mutex> lock(mu_);
        return total_;
    }
    std::vector<ErrorScopeEntry> entries() const {
        std::lock_guard<std::mutex> lock(mu_);
        return entries_;
    }
    std::map<std::string, uint64_t> by_message() const {
        std::lock_guard<std::mutex> lock(mu_);
        return by_message_;
    }
    uint64_t entries_dropped() const {
        std::lock_guard<std::mutex> lock(mu_);
        return entries_dropped_;
    }

    static std::string json_escape(const std::string& s) {
        std::string o;
        for (char c : s) {
            if (c == '"' || c == '\\') {
                o += '\\';
                o += c;
            } else if (c == '\n') {
                o += "\\n";
            } else if (static_cast<unsigned char>(c) >= 0x20) {
                o += c;
            }
        }
        return o;
    }

    // {"total": N, "entries_dropped": N, "by_message": [{"count", "text"}...], "entries": [...]}
    std::string json() const {
        std::lock_guard<std::mutex> lock(mu_);
        std::string j = "{\"total\": " + std::to_string(total_) + ", \"entries_dropped\": " + std::to_string(entries_dropped_) +
                        ", \"messages_dropped\": " + std::to_string(messages_dropped_) + ", \"by_message\": [";
        std::vector<std::pair<std::string, uint64_t>> rows(by_message_.begin(), by_message_.end());
        std::sort(rows.begin(), rows.end(), [](auto& a, auto& b) { return a.second > b.second; });
        for (size_t i = 0; i < rows.size(); ++i) {
            if (i) j += ", ";
            j += "{\"count\": " + std::to_string(rows[i].second) + ", \"text\": \"" + json_escape(rows[i].first) + "\"}";
        }
        j += "], \"entries\": [";
        for (size_t i = 0; i < entries_.size(); ++i) {
            const ErrorScopeEntry& e = entries_[i];
            if (i) j += ", ";
            j += "{\"count\": " + std::to_string(e.count) + ", \"kind\": \"" + e.kind + "\", \"phase\": \"" + e.phase +
                 "\", \"present_index\": " + std::to_string(e.present_index) + ", \"packet\": " + std::to_string(e.packet) +
                 ", \"opcode\": " + std::to_string(e.opcode) + ", \"context\": \"" + json_escape(e.context) +
                 "\", \"message\": \"" + json_escape(e.message) + "\"}";
        }
        j += "]}";
        return j;
    }

private:
    mutable std::mutex mu_;
    std::vector<ErrorScopeEntry> entries_;
    std::map<std::string, uint64_t> by_message_;
    uint64_t total_ = 0, entries_dropped_ = 0, messages_dropped_ = 0;
};

// Always-on histogram of every uncaptured device error by kind: the message with its numbers folded
// ("storage buffers (26)" and "(29)" are one kind), keyed with the opcode in flight when Dawn raised it
// (0xffff = a shader worker thread). Cheap (error path only) and bounded, so a long boot's
// device_errors total can be split by cause without --gpu-error-scopes, which only sees the first 200.
inline constexpr size_t kDeviceErrorKindsMax = 48;
inline constexpr size_t kDeviceErrorKindChars = 320;
inline constexpr uint16_t kDeviceErrorWorkerOpcode = 0xffff;
inline constexpr uint16_t kDeviceErrorDeferredOpcode = 0xfffe;  // pump_deferred ahead of a packet

inline std::string normalize_device_error(const std::string& m) {
    std::string o;
    o.reserve(std::min(m.size(), kDeviceErrorKindChars));
    for (size_t i = 0; i < m.size() && o.size() < kDeviceErrorKindChars;) {
        const char c = m[i];
        if (c >= '0' && c <= '9') {  // a number (decimal or 0x hex) becomes N
            const bool hex = c == '0' && i + 1 < m.size() && (m[i + 1] == 'x' || m[i + 1] == 'X');
            if (hex) i += 2;
            while (i < m.size() && ((m[i] >= '0' && m[i] <= '9') ||
                                    (hex && ((m[i] >= 'a' && m[i] <= 'f') || (m[i] >= 'A' && m[i] <= 'F')))))
                ++i;
            o += 'N';
            continue;
        }
        if (c == '\n' || c == '\r' || c == '\t') {
            if (!o.empty() && o.back() != ' ') o += ' ';
        } else if (c != ' ' || (!o.empty() && o.back() != ' ')) {
            o += c;
        }
        ++i;
    }
    while (!o.empty() && o.back() == ' ') o.pop_back();
    return o;
}

// A cascade error is a consequence of an earlier root error rather than a cause: the deliberate invalid
// placeholder pipeline (entry pg6_missing_entry) made after a real pipeline build failed, and Dawn's
// "[Invalid CommandBuffer/BindGroup/BindGroupLayout/PipelineLayout/...] is invalid" follow-ups.
inline bool is_cascade_device_error(const std::string& m) {
    if (m.find("pg6_missing_entry") != std::string::npos || m.find("pgN_missing_entry") != std::string::npos) return true;
    static const char* const kObjs[] = {"CommandBuffer", "BindGroup", "BindGroupLayout", "PipelineLayout", "RenderPipeline", "ComputePipeline"};
    for (const char* o : kObjs) {
        const std::string a = std::string("Invalid ") + o, b = std::string("invalid ") + o;
        if (m.find(a) != std::string::npos || m.find(b) != std::string::npos) return true;
    }
    return false;
}

class DeviceErrorKinds {
public:
    void record(uint16_t opcode, const std::string& message, uint64_t n = 1) {
        std::string key = normalize_device_error(message);
        const bool cascade = is_cascade_device_error(message);
        std::lock_guard<std::mutex> lock(mu_);
        total_ += n;
        (cascade ? cascade_ : root_) += n;
        auto it = rows_.find({opcode, key});
        if (it != rows_.end()) {
            it->second += n;
        } else if (rows_.size() < kDeviceErrorKindsMax) {
            rows_.emplace(std::make_pair(opcode, std::move(key)), n);
        } else {
            other_ += n;
        }
    }
    uint64_t total() const {
        std::lock_guard<std::mutex> lock(mu_);
        return total_;
    }
    uint64_t count(uint16_t opcode, const std::string& message) const {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = rows_.find({opcode, normalize_device_error(message)});
        return it == rows_.end() ? 0 : it->second;
    }
    uint64_t cascade_total() const {
        std::lock_guard<std::mutex> lock(mu_);
        return cascade_;
    }
    uint64_t root_total() const {
        std::lock_guard<std::mutex> lock(mu_);
        return root_;
    }
    // {"total": N, "root_total": N, "cascade_total": N, "other": N, "kinds": [{"count", "opcode", "cascade", "text"}...]}
    // sorted by count; cascade rows are consequences of an earlier root error (see is_cascade_device_error)
    std::string json() const {
        std::lock_guard<std::mutex> lock(mu_);
        std::vector<std::pair<const std::pair<uint16_t, std::string>*, uint64_t>> rows;
        for (const auto& kv : rows_) rows.emplace_back(&kv.first, kv.second);
        std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
        std::string j = "{\"total\": " + std::to_string(total_) + ", \"root_total\": " + std::to_string(root_) +
                        ", \"cascade_total\": " + std::to_string(cascade_) + ", \"other\": " + std::to_string(other_) + ", \"kinds\": [";
        for (size_t i = 0; i < rows.size(); ++i) {
            if (i) j += ", ";
            j += "{\"count\": " + std::to_string(rows[i].second) + ", \"opcode\": " + std::to_string(rows[i].first->first) +
                 ", \"cascade\": " + (is_cascade_device_error(rows[i].first->second) ? "true" : "false") + ", \"text\": \"" + ErrorScopeLog::json_escape(rows[i].first->second) + "\"}";
        }
        return j + "]}";
    }

private:
    mutable std::mutex mu_;
    std::map<std::pair<uint16_t, std::string>, uint64_t> rows_;
    uint64_t total_ = 0, other_ = 0, root_ = 0, cascade_ = 0;
};

}  // namespace playgta6::gpu::native_exec
