// Native executor: consumes the GPU packet stream (docs/gpu_commands.md) with a native WebGPU
// implementation (Dawn, the library Chrome's WebGPU is built on) and presents to an offscreen
// swapchain texture (headless) or a native window. It mirrors the semantics of the browser worker
// (web/workers/wgpu_worker.js) packet by packet; packets it does not implement yet are skipped by
// payload_bytes and counted per opcode, never approximated. Built only with
// -DPLAYGTA6_NATIVE_EXECUTOR=ON. Design and milestones: docs/NATIVE_EXECUTOR_DESIGN.md.
//
// Threading (milestone 0): an executor and its window belong to the thread that created them;
// execute() and every other call must come from that thread.
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <map>
#include <string>

#include "gpu/native_exec/present_model.h"
#include "gpu/native_exec/shader_quarantine.h"
#include <vector>

#include "gpu/native_exec/m2_model.h"

namespace playgta6::gpu::d3d12_guest {
class GpuTimeTable;
}

namespace playgta6::gpu::native_exec {

struct ExecutorOptions {
    // Shader hashes whose draws / dispatches are skipped (shader_quarantine.h; --skip-shader, --skip-shader-file, repo list).
    ShaderQuarantine quarantine;
    bool window = false;  // present to a native window instead of the offscreen swapchain
    uint32_t width = 1280;  // swapchain (and initial client area) size
    uint32_t height = 720;
    std::string title = "playgta6 native executor";
    // Swapchain blit (WgpuSwapchain's intermediate texture + fullscreen blit, plus letterboxing):
    // headless, a canvas of present_width x present_height (both non-zero) receives the stream
    // (width x height) scaled into the largest centred aspect-preserving rectangle, black bars
    // around it. With a window, fixed_stream_size keeps the stream at width x height and does the same
    // into the window's client area (default: the stream follows the window size). A stream format
    // that is not a canvas format (e.g. rgba8unorm-srgb) always goes through the blit.
    uint32_t present_width = 0;
    uint32_t present_height = 0;
    bool fixed_stream_size = false;
    // Test hook (--window-script, window_script.h): timed resize / minimize / restore / fullscreen
    // actions and read-back probes driven from poll(), counted from the first presented frame.
    std::string window_script;
    // GPU pass timestamps (#1608): measured with timestamp queries when the adapter has the
    // feature and fed into this table (nullptr = the process-wide gpu_time_table(), the one the
    // boot report prints). Nothing is estimated: without the feature the table says unavailable.
    // Test hook: do not request float32-filterable even when the adapter has it, so the code paths of a
    // device without it (r16float depth twins, r32float dummies) run on real hardware.
    bool disable_float32_filterable = false;
    bool gpu_timing = true;
    // Guest query heaps (kGpuOpcodeCreateQuerySet / Query / ResolveQuery): timestamp-query is requested
    // for them too when the adapter has it (else timestamp heaps are counted as unsupported).
    bool guest_queries = true;
    // Software adapter (--adapter warp): request Dawn's fallback adapter (D3D12 WARP on Windows) instead of a GPU,
    // so GPU-less hosts (CI runners) can replay for real. Default off (the high-performance GPU).
    bool force_fallback_adapter = false;
    // Per-pipeline attribution inside a pass (--gpu-time-split): a pass is timed as segments, one per
    // run of the same pipeline, with a timestamp written inside the pass at every pipeline change
    // (Dawn's chromium-experimental-timestamp-query-inside-passes). Off: a pass is one measurement,
    // keyed to the pipeline its first SetPipeline selects. Needs gpu_timing and the adapter feature.
    bool gpu_time_split = false;
    d3d12_guest::GpuTimeTable* gpu_time = nullptr;
    // Persistent Dawn blob cache (blob_cache.h): compiled shaders / pipelines are stored under
    // <blob_cache_root>/<blob_cache_title>/<Dawn build>-<adapter>/ and loaded by later runs, so DXC and
    // the driver compile each pipeline once per machine instead of once per run. Empty root = off
    // (the default: tests and tools never touch the user's cache); boot_main passes the default root.
    std::string blob_cache_root;
    std::string blob_cache_title = "title";
    uint64_t blob_cache_max_bytes = uint64_t{2} << 30;  // LRU-trimmed to this (0 = unbounded)
    // Live play: v2 render / compute pipelines are created with CreateRenderPipelineAsync /
    // CreateComputePipelineAsync, so Dawn compiles them on its own worker threads while the executor
    // keeps consuming packets (the prewarm batches create hundreds ahead of use). The pipeline is
    // awaited, never skipped, when a pass first uses it, so output is identical to sync creation.
    bool async_pipelines = false;
    // Per-frame wait budget for a still-compiling async pipeline (pipe_budget_model.h): past it the draw
    // is skipped (drop table reason pipeline_pending) and the pipeline is used once ready. ns;
    // UINT64_MAX = wait forever (the old behaviour), 0 = never wait. Fullscreen = vertex-buffer-less
    // pipelines. Overridable: --pipeline-wait-ms / --pipeline-wait-fullscreen-ms, env
    // PG6_PIPELINE_WAIT_MS / PG6_PIPELINE_WAIT_FULLSCREEN_MS.
    uint64_t pipeline_wait_frame_ns = 500ull * 1000000ull;
    uint64_t pipeline_wait_fullscreen_ns = 1000ull * 1000000ull;
    // Strict validation (--gpu-validation): Dawn backend validation (D3D12 debug layer when installed),
    // immediate error handling, and an error scope around every packet. Errors are deduped by text in
    // stats_json()'s "validation" block with the first packet index / opcode. Slower; diagnosis only.
    bool gpu_validation = false;
    // Error-scope attribution (--gpu-error-scopes, error_scope_model.h): an error scope around every packet
    // (pipeline / bind group creation, pass commands, the present's encode + submit) records each device
    // error with its phase, present index and the shader hashes / resource ids in scope into stats_json()'s
    // "error_scopes" block (bounded). Off = no scopes until the first uncaptured device error arms it
    // (then for up to 200 scoped errors), so a healthy run pays nothing.
    bool gpu_error_scopes = false;
    // One command buffer per pass (opt-in, --split-passes): each render / compute pass of a frame or
    // read-back flush is finished and submitted on its own, in order. Dawn invalidates a whole command
    // buffer on its first validation error (an invalid bind group, writable storage aliasing, ...), so
    // with one buffer per frame a single bad pass dropped every pass of that frame, the present copy and
    // the read-backs with it (black frames); split, only the bad pass is lost.
    // Off by default: in the in-world scene the passes that a bad pass used to take down with it then run,
    // and some of them keep the GPU busy for tens of seconds; Dawn's Submit blocks in WaitForSerial (the
    // command allocators are all in flight), the live executor stops, the title's 30 s main-loop watchdog
    // fires (std-pack --live-executor boot of 3d57fb4a: 0 fps from ~90 s, sub_1423CF620 at 120 s; with
    // the split off the same build runs past 170 s).
    bool split_passes = false;
    // Per-submit GPU time budget (--submit-budget-ms, 0 = off): a frame's passes are submitted early, at a
    // pass boundary, once the GPU time they are estimated to take (the measured average of each pass's
    // timing key, gpu_time) reaches it, so no single submit runs for the ~2 s Windows' TDR allows (one
    // in-world frame of the std-pack --live-executor boot took ~3 s on the GPU: DXGI_ERROR_DEVICE_HUNG).
    uint64_t submit_budget_ns = 250ull * 1000000ull;
    // Usage-scope snapshots (opt-in, --usage-snapshots; usage_scope_model.h): a read-only binding of a
    // resource the same draw / dispatch writes is bound from a copy of its read range taken right before
    // it. Off by default: in the in-world scene of the std-pack --live-executor boot they copy ~950 GB in
    // ~90 s (~34k snapshots of large read-only ranges) and the frames they make valid hang the GPU
    // (DXGI_ERROR_DEVICE_HUNG at ~90-98 s, then the device is lost for good); with them off the same build
    // runs past 170 s. Off, such a conflict is left to the buffer aliasing pass (resolve_buffer_aliases).
    bool usage_snapshots = false;
    // With usage_snapshots off, small snapshots are still taken where nothing cheaper exists (a render draw's
    // read side, a compute read of a buffer range the dispatch does not write): at most `cap` bytes each and
    // `budget` bytes per command encoder. A conflict left after that skips the draw / dispatch (counted by
    // reason, skip_usage_*) instead of recording it: Dawn would reject the whole command buffer. 0 cap: none.
    uint64_t usage_snapshot_cap = 4ull << 20;
    uint64_t usage_snapshot_budget = 64ull << 20;
    // Device-loss recovery (docs/NATIVE_EXECUTOR_DESIGN.md "Device loss"): the executor journals the packets
    // that create resources and fill them so a lost device can be rebuilt and the resources re-created
    // with their uploaded contents. Buffer writes are folded into one image per buffer (64 KiB pages, a
    // rewrite costs nothing), a texture write drops the older writes its box covers, destroyed ids are
    // pruned. Journal budget in bytes; 0 = no journal, so a device loss ends the run with a fault record.
    // Over the budget, content goes first (the most often rewritten resources, then the incoming write):
    // those resources come back empty after a recovery (counted, the next upload fills them); only
    // creations that outgrow the budget make a recovery impossible.
    static constexpr uint64_t kDefaultRecoveryJournalBytes = 512ull << 20;
    uint64_t recovery_journal_bytes = kDefaultRecoveryJournalBytes;
    // With the default budget: raise it to 1/8 of the free physical memory at start (at most 2 GiB).
    bool recovery_journal_auto = true;
    // Live-vs-replay check (tools/live_vs_replay.py): when non-empty, every presented frame's stream image
    // (before the blit; 4-byte formats) is read back and a line `digest frame N: <16 hex> WxH <format>` is
    // appended here, N counting presents from 0 - the same digest pg6_native_exec --digest-at prints.
    // Slow (one synchronous read-back per frame); a window surface then gets CopySrc usage.
    std::string frame_digest_path;
    // Window present mode (--present-mode) and host-side frame cap (--max-fps, 0 = off). An unsupported
    // mode falls back (logged); the guest's own pacing is not changed.
    PresentChoice present_mode = PresentChoice::Auto;
    uint32_t max_fps = 0;
    // Screenshots: F12 in the window, and these 0-based present indices (--screenshot-at) save the presented
    // stream image (before the letterbox) as a PNG into screenshot_dir (empty: %LOCALAPPDATA%/playgta6/screenshots).
    std::vector<uint64_t> screenshot_at;
    std::string screenshot_dir;
    // --show-target <selector> (show_target_model.h): present this render target / texture instead of the
    // swapchain image. Read-only, the guest's frame is unchanged. Empty = off. With a window F7 / F8 / F9 /
    // F10 (mip, layer, previous, next target) take over; set_show_target changes it at run time.
    std::string show_target;
    // --dump-targets-at N[,N...] (target_dump_model.h): at these 0-based presents every render target the frame drew
    // to (each colour attachment and the depth of every render pass) and every storage texture a dispatch wrote is
    // drawn raw into an rgba16float copy (at most dump_targets_max_dim on its larger side) and written as
    // <dump_targets_dir>/present-<N>/<pass>-tex-<id>[...].png (scaled by its own max) plus index.txt: per target
    // its format / size, value stats (BLACK / FLAT / ok, max, mean, NaN) and the draws / dispatches that ran and
    // were skipped from that pass to the next. Empty dir: <screenshot dir>/targets. Read-only like show_target.
    std::vector<uint64_t> dump_targets_at;
    std::string dump_targets_dir;
    uint32_t dump_targets_max_dim = 640;
    // On-screen stats overlay (overlay_model.h) at start; F3 toggles it in a window, set_overlay() at run time.
    // Drawn after every copy of the guest's frame: digests and screenshots are unchanged.
    bool overlay = false;
};

inline constexpr size_t kOpcodeSlots = 64;

// A direct dispatch's workgroups in all are cut to kMaxDispatchGroups (z first, then y, then x), as the
// ExecuteIndirect clamp pass cuts indirect ones (state_bridge indirect_clamp_wgsl MAX_GROUPS, 2^18): a garbage
// count would keep the GPU busy past the TDR limit. True: the counts were cut.
inline constexpr uint64_t kMaxDispatchGroups = uint64_t{1} << 22;
inline bool clamp_dispatch_groups(uint32_t* x, uint32_t* y, uint32_t* z) {
    if (uint64_t{*x} * *y * *z <= kMaxDispatchGroups) return false;
    if (*x > kMaxDispatchGroups) *x = static_cast<uint32_t>(kMaxDispatchGroups);
    if (uint64_t{*x} * *y > kMaxDispatchGroups) {
        *z = 1;
        *y = static_cast<uint32_t>(std::max<uint64_t>(1, kMaxDispatchGroups / *x));
    } else {
        *z = static_cast<uint32_t>(std::max<uint64_t>(1, kMaxDispatchGroups / (uint64_t{*x} * *y)));
    }
    return true;
}

// "DXGI_ERROR_DEVICE_HUNG" out of Dawn's device-loss message ("Device removed reason: <code> (0x...)", from
// ID3D12Device::GetDeviceRemovedReason); else its first DXGI_ERROR_ code; empty when it has none.
inline std::string device_removed_reason_of(const std::string& msg) {
    static const std::string kKey = "Device removed reason: ";
    size_t p = msg.find(kKey);
    if (p != std::string::npos) p += kKey.size();
    else p = msg.find("DXGI_ERROR_");
    if (p == std::string::npos) return {};
    size_t e = p;
    while (e < msg.size() && ((msg[e] >= 'A' && msg[e] <= 'Z') || (msg[e] >= '0' && msg[e] <= '9') || msg[e] == '_')) ++e;
    return msg.substr(p, e - p);
}

struct ExecutorStats {
    uint64_t batches = 0;
    uint64_t packets = 0;
    uint64_t bytes = 0;
    uint64_t present_packets = 0;
    uint64_t frames_submitted = 0;   // the worker's frameCount
    uint64_t present_failures = 0;   // no swapchain texture / submit failed: the frame is dropped
    uint64_t truncated_batches = 0;  // CommandStream "truncated packet" (rest of the batch dropped)
    uint64_t op_errors = 0;          // the worker's _recordOpValidationError for payload checks
    uint64_t device_errors = 0;      // uncaptured WebGPU errors (validation / OOM / internal)
    uint64_t device_lost = 0;
    // Device loss recovery. device_lost_events counts every loss that was not our own teardown;
    // recoveries the ones that ended with a rebuilt device, recovery_failures the ones that ended the run.
    uint64_t device_lost_events = 0;
    uint64_t recoveries = 0;
    uint64_t recovery_failures = 0;
    uint64_t recovery_ns_total = 0;
    uint64_t recovery_replayed_packets = 0;   // journal packets re-dispatched onto the new device
    uint64_t recovery_replay_errors = 0;      // ... that the executor rejected (packet errors during the replay)
    uint64_t recovery_frames_dropped = 0;     // the frame that was being built when the device was lost
    uint64_t recovery_gpu_resources_reset = 0;  // render-target / storage resources re-created empty (GPU-written)
    uint64_t journal_bytes = 0;               // live journal size (creations + content)
    uint64_t journal_create_bytes = 0;        // ... of it creations (shaders, pipelines, layouts, bind groups, ...)
    uint64_t journal_budget = 0;              // the effective budget (recovery_journal_bytes, raised by _auto)
    uint64_t journal_overflows = 0;           // writes that did not fit the budget (their resource's content dropped)
    uint64_t journal_evicted_resources = 0;   // resources whose journaled content was dropped for the budget
    uint64_t journal_evicted_bytes = 0;
    uint64_t recovery_contents_lost = 0;      // live resources a recovery re-created without their uploaded content
    // Device loss diagnostics: Dawn's lost reason / message (also in the fault record), and the submits the
    // GPU had not finished when the device was lost (JSON array, oldest first: age, frame, work counts).
    std::string device_lost_reason;
    std::string device_lost_message;
    std::string device_lost_inflight;
    std::string device_removed_reason;        // GetDeviceRemovedReason of the last loss (e.g. DXGI_ERROR_DEVICE_HUNG)
    uint64_t recovery_attempts = 0;           // device creations tried by recoveries
    uint64_t recovery_instance_resets = 0;    // ... after dropping the instance (the removed adapter's D3D12 device)
    uint64_t recovery_bind_groups_deferred = 0;  // bind groups a recovery left to their next bind (not replayed)
    uint64_t recovery_adapter_mismatches = 0;    // recovery attempts that were offered another adapter (e.g. WARP)
    uint64_t recovery_shader_jobs_dropped = 0;   // queued shader-worker jobs of the lost device dropped by a recovery
    std::string recovery_timeline;            // JSON: the last recovery's phases and attempts (ms)
    uint64_t debug_gpu_hangs = 0;             // GPU hangs injected on purpose (debug_hang_gpu / PG6_DEBUG_GPU_HANG_AT_FRAME)
    uint64_t submit_budget_splits = 0;        // early submits at a pass boundary (submit_budget_ns)
    uint64_t submit_est_max_ns = 0;           // the largest estimated GPU time of one submit
    uint64_t submits_tracked = 0;             // frame submits watched for completion
    uint64_t submit_gpu_max_ns = 0;           // the longest submit -> completion time seen
    uint64_t submit_gpu_max_frame = 0;        // ... and the frame (frames_submitted) it belonged to
    uint64_t draws_clamped = 0;               // direct draws whose vertex x instance count was clamped (kMaxDrawWork)
    bool faulted = false;                     // the run cannot continue (see fault_record); poll() returns false
    std::string fault_record;                 // JSON object, empty unless faulted
    uint64_t dropped_while_lost = 0;  // batches ignored after device loss
    uint64_t render_pass_missing_texture = 0;  // BeginRenderPass dropped: unknown attachment texture
    uint64_t render_pass_dropped_colors = 0;   // attachments dropped for a size mismatch
    uint64_t render_pass_proxied_colors = 0;   // colors of another size than the depth, rendered through a proxy
    uint64_t render_pass_proxy_textures = 0;   // proxy textures created for them
    uint64_t swapchain_format_unsupported = 0;  // stream asked for a swapchain format that is no colour format
    uint64_t blit_frames = 0;  // frames presented through the intermediate texture + blit
    uint64_t surface_configures = 0;         // surface (re)configurations (resize, format change, lost surface)
    uint64_t surface_minimized_skips = 0;    // presents skipped because the client area was empty (minimized)
    uint64_t surface_acquire_retries = 0;    // acquires retried after Outdated / Lost / Error
    uint64_t window_actions = 0;             // --window-script actions applied
    uint64_t overlay_frames = 0;             // presents / idle redraws that drew the stats overlay
    uint64_t probes = 0;                     // letterbox probes run
    uint64_t screenshots_saved = 0, screenshots_failed = 0;  // F12 / --screenshot-at PNGs written / failed
    uint64_t probes_failed = 0;              // ... whose bars were not black or whose picture was blank
    uint64_t quarantine_skips = 0;           // draws / dispatches skipped by the shader quarantine (also in the drop table)
    uint64_t show_target_frames = 0;         // presents that drew the --show-target texture instead of the swapchain image
    uint64_t show_target_unavailable = 0;    // presents where the chosen target could not be shown (the swapchain image stays)
    uint64_t targets_dumped = 0, targets_dump_failed = 0;  // --dump-targets-at PNGs written / targets that could not be
    // Milestone 1 (resources, pipelines, draws). "packet errors" are the worker's `_packetError`
    // (a v2 / texture packet it rejects); op errors above are its `_recordOpValidationError`.
    uint64_t packet_errors = 0;
    std::array<uint64_t, kOpcodeSlots> packet_errors_by_opcode{};
    std::string last_packet_error;
    uint64_t shader_modules = 0;
    uint64_t shader_pack_hits = 0;
    uint64_t shader_code_rereads = 0;  // WGSL texts re-read from the pack after being dropped (rescan of a dropped shader)
    uint64_t shader_pack_misses = 0;  // a hash-only shader the pack does not hold (compiled as text: invalid)
    uint64_t shader_binding_shifted = 0;  // pack shaders requested with renumbered @group(0) bindings (legacy fragments)
    // Pack shaders materialized on first pipeline use (on the executor thread or a shader worker): the
    // module compile (WGSL parse + resolve) and the memoized text scans (entry point, @group(0) bindings).
    // shader_materialize_* is the time spent on the executor (queue) thread only.
    uint64_t shader_materialized = 0;
    uint64_t shader_materialize_ns = 0;
    uint64_t shader_materialize_max_ns = 0;
    uint64_t shader_materialize_max_hash = 0;
    uint64_t shader_materialize_max_bytes = 0;
    uint64_t shader_scans = 0;
    uint64_t shader_scan_ns = 0;
    // Live mode (async_pipelines): shader workers (shader_workers.h) compile them instead. Handed over,
    // finished (and their compile time / slowest one, host private MB after a compile), times the
    // executor thread had to wait for one (a draw / dispatch needed its pipeline) and how long.
    uint64_t shader_workers = 0;
    uint64_t shader_async_queued = 0;
    uint64_t shader_async = 0;
    uint64_t shader_async_ns = 0;
    uint64_t shader_async_max_ns = 0;
    uint64_t shader_async_max_hash = 0;
    uint64_t shader_async_max_bytes = 0;
    uint64_t shader_async_private_mb_max = 0;
    uint64_t shader_async_max_cpu_ns = 0;    // CPU time of the slowest one (far below its wall time: it waited)
    uint64_t shader_async_slow = 0;          // modules that took >= 1 s
    uint64_t shader_async_slow_blocked = 0;  // ... of them mostly off the CPU (disk, paging or a lock)
    uint64_t shader_async_waits = 0;
    uint64_t shader_async_wait_ns = 0;
    // Pipelines created after their packet because a shader was still on a worker: deferred, built
    // so far, the most deferred at once.
    uint64_t pipelines_deferred = 0;
    uint64_t pipelines_deferred_built = 0;
    uint64_t pipelines_deferred_max = 0;
    // Packets that took >= 1 s in the executor (each also logged, the first 32, with its shader).
    uint64_t slow_packets = 0;
    uint64_t slow_packet_max_ns = 0;
    uint64_t textures_created = 0;
    uint64_t texture_uploads = 0;
    uint64_t texture_upload_bytes = 0;
    // ClearTexture (clear_model.h): textures zeroed on the GPU, the zero-buffer copies and texel bytes
    // that took, the extra submits a write forced, clears whose texels were all written before they
    // were encoded (no copy), and clears of textures this executor never created.
    uint64_t textures_cleared = 0;
    uint64_t clear_copies = 0;
    uint64_t clear_bytes = 0;
    uint64_t clear_flushes = 0;
    uint64_t clears_covered = 0;
    uint64_t clears_missing = 0;
    uint64_t texture_copies = 0;  // CopyTexture (opcode 62) copies encoded
    uint64_t samplers_created = 0;
    uint64_t bind_group_layouts = 0;
    uint64_t pipeline_layouts = 0;
    uint64_t render_pipelines = 0;
    uint64_t compute_pipelines = 0;
    uint64_t bind_groups = 0;
    uint64_t draws = 0;
    uint64_t draws_skipped = 0;        // _executePresentRenderDraw skips (no pipeline / group 0 unbound)
    uint64_t dispatches = 0;
    uint64_t dispatches_skipped = 0;   // zero-size or missing pipeline (_preparePassCommandsForFlush)
    uint64_t dispatches_clamped = 0;
    uint64_t auto_layout_pipelines = 0;  // layout 0 ("auto"): the worker's WGSL remap is not mirrored
    uint64_t manifest_layouts_unavailable = 0;
    // layout:'auto' bind group remap (the worker's _getOrCreateRemappedBindGroup).
    uint64_t auto_layout_remap_pipelines = 0;  // an explicit layout that did not fit the WGSL @group(0) bindings
    uint64_t remapped_bind_groups = 0;         // group 0 bind groups built for such pipelines
    uint64_t remap_cache_hits = 0;
    uint64_t remap_dynamic_offsets = 0;        // dynamic offsets folded into static buffer offsets
    uint64_t remap_dynamic_clamps = 0;         // folded offsets clamped to the buffer / out of bounds (dummy)
    uint64_t remap_fallback_samplers = 0;      // sampler bindings no resource matched: default sampler
    uint64_t remap_dummies = 0;                // bindings no resource matched: dummy resource
    // Milestone 2: read-backs (39/40), v1 pipelines / bind groups (8/15/16/19), pass timestamps.
    uint64_t readbacks_requested = 0;
    uint64_t readbacks_delivered = 0;  // bytes written to guest memory
    uint64_t readbacks_failed = 0;     // counter advanced without data (copy / map failed, no memory, out of range)
    uint64_t readback_bytes = 0;
    uint64_t early_readback_flushes = 0;  // submits that ran the recorded work up to a read-back (no present)
    uint64_t command_buffers = 0;         // command buffers submitted by presents and read-back flushes (split_passes: one per pass)
    // Usage scopes (usage_scope_model.h): a resource written and read through bindings of one render pass /
    // one dispatch. Render passes split at a draw that conflicts with an earlier draw of the pass; passes
    // split to bind snapshots (copies of the read range) for a conflict inside one draw / dispatch; those
    // snapshots (buffer / texture) and their bytes (texture bytes estimated at 4 per texel); snapshots
    // that could not be made (the conflict stays); splits of a pass whose attachments store Discard.
    uint64_t usage_scope_splits = 0;
    uint64_t usage_snapshot_splits = 0;
    uint64_t usage_buffer_snapshots = 0;
    uint64_t usage_texture_snapshots = 0;
    uint64_t usage_snapshot_bytes = 0;
    uint64_t usage_snapshot_skipped = 0;
    uint64_t usage_snapshot_reused = 0;  // buffer snapshots served from the encoder cache (no copy, no split)
    uint64_t usage_snapshot_bytes_saved = 0;
    // Shader binding access (binding_access_model.h): buffer bindings the bound pipeline never references that
    // touch another binding of the same buffer are bound to a dummy instead (no snapshot, no alias copy).
    uint64_t usage_unref_dummies = 0;        // bind-group entries bound to such a dummy
    uint64_t usage_unref_groups = 0;         // substitute bind groups created for them
    uint64_t usage_never_written_uses = 0;   // writable buffer uses the pipeline never writes (reads of them stay cacheable)
    uint64_t usage_writes_not_noted = 0;     // writable buffer uses that did not invalidate cached snapshots (never written)
    uint64_t shader_access_scans = 0;        // WGSL modules scanned for their binding access
    uint64_t shader_access_unknown = 0;      // ... the scan could not account for every binding (all assumed written)
    uint64_t shader_rw_never_written = 0;    // ... read_write storage / storage-texture bindings the module never writes
    uint64_t usage_snapshots_to_writer_twin = 0;  // big read-only snapshots avoided by twinning the small writable binding
    uint64_t usage_snapshots_off = 0;  // snapshots not taken: ExecutorOptions::usage_snapshots is off
    uint64_t usage_split_discards = 0;
    uint64_t usage_small_snapshots = 0;        // snapshots taken under usage_snapshot_cap with usage_snapshots off
    uint64_t usage_snapshot_over_cap = 0;      // ... not taken: bigger than the cap
    uint64_t usage_snapshot_over_budget = 0;   // ... not taken: the encoder's snapshot budget is spent
    // Draws / dispatches skipped because what they bind still breaks a usage scope rule (scope_violation in
    // usage_scope_model.h): read-only + writable use of a buffer / a texture subresource, overlapping writable
    // buffer ranges, one texture subresource writable twice. Recorded, Dawn rejects the whole command buffer.
    uint64_t skip_usage_rw_buffer = 0;
    uint64_t skip_usage_rw_texture = 0;
    uint64_t skip_usage_alias_buffer = 0;
    uint64_t skip_usage_alias_texture = 0;
    uint64_t v1_render_pipelines = 0;
    uint64_t v1_compute_pipelines = 0;
    uint64_t v1_bind_groups = 0;
    uint64_t reflection_unavailable = 0;  // v1 pipelines built without a matching reflection record (layout:auto)
    // Reflection loader (reflection_model.h: web/workers/reflection_loader.js): .reflect.json sidecars
    // (v1..v3) and layout-manifest.json documents feed v1 pipelines (layout, entry points, vertex
    // attributes), v1 bind groups (layouts, fallback samplers) and ManifestReference layouts.
    uint64_t reflection_documents = 0;      // documents loaded (sidecars + manifests)
    uint64_t reflection_load_errors = 0;    // documents the loader rejected (nothing of them is registered)
    uint64_t reflection_shaders = 0;        // stage / shader records in the registry
    uint64_t reflected_render_pipelines = 0;   // v1 render pipelines built with a reflected layout
    uint64_t reflected_compute_pipelines = 0;  // v1 compute pipelines built with a reflected layout
    uint64_t reflected_vertex_layouts = 0;     // v1 render pipelines with the reflected vertex buffer layout
    uint64_t reflection_vertex_fallbacks = 0;  // reflected vertex layout rejected: the single-attribute default
    uint64_t reflection_layout_errors = 0;     // a reflected layout the loader's merge rejected (pipeline not created)
    uint64_t reflection_binding_conflicts = 0; // ... of which two stages disagreed on a binding
    uint64_t norm16_twins_created = 0;     // float copies of r16/rg16/rgba16 unorm/snorm textures (float entry meets a norm16 view)
    uint64_t norm16_twin_copies = 0;       // ... refreshes
    uint64_t norm16_twin_binds = 0;        // ... bind-group entries that read a twin (instead of a dummy)
    uint64_t unfilterable_dummy_not_norm16 = 0;   // ... of the dummies: not a norm16 format (f32 without float32-filterable)
    uint64_t unfilterable_dummy_view_format = 0;  // ... norm16 view whose format differs from the texture's
    uint64_t unfilterable_dummy_no_twin = 0;      // ... norm16 but the texture cannot have a twin (msaa, non-2d, promoted 1d, create failed)
    uint64_t bind_group_unfilterable_dummies = 0;  // float layout entry vs an unfilterable-float-only view (norm16, f32 without float32-filterable): dummy bound
    uint64_t bind_group_sample_type_dummies = 0;  // integer view vs float layout entry (or the reverse): dummy bound
    uint64_t buffer_bindings_clamped = 0;  // buffer ranges cut to the device's binding-size rules (fit_buffer_binding)
    uint64_t buffer_bindings_dummied = 0;  // buffer bindings no range of which fits those rules: dummy bound
    // Remapped group 0 buffer ranges shorter than the WGSL binding's minimum size (its reflection):
    uint64_t remap_min_size_extended = 0;  // grown within the buffer
    uint64_t remap_min_size_twins = 0;     // bound as a zero-padded copy (the buffer ends first)
    uint64_t remap_min_size_splits = 0;    // passes split to encode those copies
    uint64_t remap_min_size_dummies = 0;   // no copy possible: a dummy
    // The same for explicit (CreateBindGroup2, manifest layouts included) and v1 bind groups: the layout's / the
    // reflected group's minBindingSize
    uint64_t explicit_min_size_extended = 0;
    uint64_t explicit_min_size_twins = 0;   // bound as a zero-padded copy taken at creation
    uint64_t explicit_min_size_dummies = 0;
    uint64_t v1_min_size_extended = 0;
    uint64_t v1_min_size_twins = 0;
    uint64_t v1_min_size_unfit = 0;  // no fit (left as the title bound it)
    uint64_t view_dim_coerced = 0;         // texture views re-created with the layout entry's viewDimension (cube-array view meeting a 2d entry, ...)
    uint64_t view_dim_unfixable = 0;       // view dimension mismatches the texture cannot satisfy (cube entry over < 6 layers)
    uint64_t view_layers_clamped = 0;      // texture views whose array layer range ran past the texture, cut to its layers
    uint64_t view_mips_clamped = 0;        // texture views whose mip range ran past the texture, cut to its mips
    uint64_t view_range_unfixable = 0;     // texture views starting past the texture (or a cube over < 6 layers): left to Dawn
    uint64_t storage_alias_copies = 0;     // read-only buffer bindings overlapping a writable one, bound through a copy
    // Guest query heaps (kGpuOpcodeCreateQuerySet / Query / ResolveQuery).
    uint64_t query_sets_created = 0;       // guest query heaps with a query set (occlusion results / timestamp set)
    uint64_t query_begins = 0;             // occlusion queries begun
    uint64_t query_ends = 0;               // occlusion queries ended (their samples folded into the heap's results)
    uint64_t query_occlusion_segments = 0;  // pool occlusion queries written (one per render-pass segment a query's draws fell in)
    uint64_t query_timestamps = 0;         // timestamps written
    uint64_t query_timestamp_splits = 0;   // passes split to write a timestamp between their commands
    uint64_t query_resolves = 0;           // ResolveQuery packets that wrote their results
    uint64_t query_results_written = 0;    // 64-bit results those wrote into the destination buffers
    uint64_t query_unsupported = 0;        // query packets not run: no timestamp feature, unknown set / buffer, nested or unmatched begin / end, out-of-range destination
    uint64_t query_pool_exhausted = 0;     // occlusion segments past the per-submit pool: the query's result is not written
    uint64_t storage_alias_dispatch_restarts = 0;  // compute passes split to take those copies
    uint64_t storage_alias_unfixable = 0;  // overlaps no copy could separate (zero-size / out-of-range / depth, MS, BC textures, two texture writers)
    // ... the unfixable ones by reason (they sum to storage_alias_unfixable)
    uint64_t storage_alias_unfixable_tex_writers = 0;  // two writable storage-texture bindings of one subresource
    uint64_t storage_alias_unfixable_tex_ms = 0;       // texture twin impossible: multisampled
    uint64_t storage_alias_unfixable_tex_depth = 0;    // ... depth / stencil format
    uint64_t storage_alias_unfixable_tex_bc = 0;       // ... block-compressed format
    uint64_t storage_alias_unfixable_tex_view = 0;     // ... view format or aspect differs from the texture's
    uint64_t storage_alias_unfixable_tex_copysrc = 0;  // ... texture without CopySrc usage
    uint64_t storage_alias_unfixable_buf_range = 0;    // buffer copy impossible: zero words / range past the buffer end
    uint64_t storage_alias_twin_writers = 0;  // later writable buffer bindings run on a scratch copy whose changed words are merged back
    uint64_t storage_alias_copy_dispatches = 0;  // copies / merges done by a compute kernel inside the pass (no split)
    uint64_t storage_tex_alias_copies = 0;  // texture bindings overlapping a storage-texture write, bound through a twin texture
    uint64_t storage_tex_alias_restarts = 0;  // compute passes split to take texture twins
    uint64_t bind_group_depth_layout_dummies = 0;  // depth view vs float/sint/uint layout entry (or the reverse): dummy bound
    uint64_t depth_float_copies = 0;       // depth aspect re-copied into an r32float twin (float layout entry meets a depth texture)
    uint64_t depth_twins_created = 0;      // ... of which r32uint/r32sint (uint/sint entry on a depth texture): depth_int_twins_created
    uint64_t depth_int_twins_created = 0;
    uint64_t depth_half_twins_created = 0; // r16float twins (float entry, device without float32-filterable): half-precision depth
    uint64_t pipeline_variants_created = 0;     // render pipelines rebuilt with the attachment formats / samples of the pass they are drawn into
    uint64_t pipeline_variants_unsupported = 0; // pipelines that differ from their pass but cannot be rebuilt (auto layout)
    uint64_t pipeline_variants_failed = 0;      // variants whose compile failed: draws dropped instead of using the mismatching base pipeline
    uint64_t depth_stencil_direct_binds = 0;  // stencil-aspect views bound to a uint entry (no twin: the stencil aspect is uint)
    uint64_t reflection_layout_unsupported = 0;  // layout entries Dawn cannot take (externalTexture, unknown storage format)
    uint64_t manifest_layouts_resolved = 0;    // ManifestReference layouts served from a loaded layout-manifest.json
    uint64_t v1_synth_samplers = 0;            // v1 bind group samplers synthesized from the layout (default samplers)
    std::string last_reflection_error;
    uint64_t timer_frames = 0;            // submits with timestamp queries attached
    uint64_t timer_passes = 0;            // passes that carried timestamp writes
    uint64_t timer_segments = 0;          // extra per-pipeline segments written inside passes (gpu_time_split)
    bool timer_split_supported = false;   // the device has timestamp-query-inside-passes (gpu_time_split)
    uint64_t timer_frames_skipped = 0;    // no free query slot (the previous one was still in flight)
    bool timer_supported = false;
    bool limits_raised = false;  // device created with requestNegotiatedDevice's raised limits
    std::array<uint64_t, kOpcodeSlots> handled{};    // by opcode
    std::array<uint64_t, kOpcodeSlots> unhandled{};  // by opcode: not implemented yet, skipped
    uint64_t unknown_opcodes = 0;                    // opcode >= kOpcodeSlots
    // Profile (ns, wall time on the executor thread): per opcode, and the phases of a batch / present.
    std::array<uint64_t, kOpcodeSlots> opcode_ns{};
    uint64_t ns_execute = 0, ns_process_events = 0, ns_flush_readbacks = 0, ns_poll = 0;
    uint64_t ns_present_acquire = 0, ns_present_encode = 0, ns_present_submit = 0, ns_present_swap = 0;
    // Persistent blob cache (blob_cache.h; a snapshot taken when stats() is read).
    bool blob_cache_enabled = false;
    std::string blob_cache_dir;
    uint64_t blob_hits = 0;       // Dawn loads served from disk (shader / pipeline compiles skipped)
    uint64_t blob_misses = 0;     // lookups Dawn had to compile for
    uint64_t blob_stores = 0;
    uint64_t blob_bytes_loaded = 0;
    uint64_t blob_bytes_stored = 0;
    uint64_t blob_evictions = 0;  // LRU removals to stay under the cap
    uint64_t blob_corrupt = 0;
    uint64_t blob_io_max_ns = 0;  // slowest single cache file read / write
    uint64_t blob_io_slow = 0;    // cache file reads / writes that took over 1 s
    uint64_t blob_entries = 0;
    uint64_t blob_total_bytes = 0;
    uint64_t blob_max_bytes = 0;
    // Async pipeline creation (ExecutorOptions::async_pipelines).
    uint64_t async_pipelines = 0;          // pipelines handed to Dawn's worker threads
    uint64_t async_pipeline_waits = 0;     // first uses that still had to wait for the compile
    uint64_t async_pipeline_wait_ns = 0;
    uint64_t pipeline_pending_skips = 0;   // draws / dispatches skipped: the pipeline was still compiling past the budget
    // Pipeline layouts fitted to the WGSL (fit_pipeline_layout): the title's layout lacked a group the
    // WGSL binds, or was over the device's per-stage limits.
    uint64_t layout_fit_pipelines = 0;      // pipelines built with a fitted layout
    uint64_t layout_fit_wgsl_groups = 0;    // groups (above 0) built from the WGSL
    uint64_t layout_fit_over_limits = 0;    // of the fitted: the title's layout was over a per-stage limit
    uint64_t layout_fit_still_over = 0;     // over the limits even fitted: built with the title's layout
    uint64_t layout_fit_dummy_binds = 0;    // fitted groups bound with dummies (the pass bound nothing there)
    uint64_t layout_fit_unused_groups = 0;  // the title's groups the WGSL does not bind, padded to get under the limits
    uint64_t layout_fit_restores = 0;
    uint64_t over_limit_bind_group_layouts = 0;  // title bind group layouts over a per-stage limit on their own
    uint64_t over_limit_binds_skipped = 0;       // their bind groups set outside a fitted pipeline's remap: not encoded       // the stream's own group re-bound after a padded pipeline's draw
    std::map<std::string, uint64_t> async_failure_kinds;  // classify_pipeline_failure -> count
    // The device's per-stage limits the layouts are checked against.
    uint32_t limit_uniform_buffers_per_stage = 0;
    uint32_t limit_storage_buffers_per_stage = 0;
    uint32_t limit_sampled_textures_per_stage = 0;
    uint32_t limit_samplers_per_stage = 0;
    uint32_t limit_storage_textures_per_stage = 0;
    uint64_t remap_reflected_layouts = 0;  // v2 remap pipelines built with an explicit reflection layout (no wait for the compiled pipeline)
    uint64_t pipeline_failed_skips = 0;  // draws / dispatches dropped because their async pipeline failed to compile
    uint64_t write_masks_cleared = 0;  // color targets the fragment stage does not write: writeMask forced to 0
    uint64_t async_pipeline_failures = 0;  // creation failed: the pipeline became an error object
    std::vector<uint64_t> failed_pipeline_hashes;  // unique perf hashes whose pipeline creation failed (auto-quarantine input)
    // Object validity (the executor never records work Dawn will reject: one invalid object in a pass
    // invalidates the whole command buffer, a black frame). Objects created with a validation error are
    // tracked as invalid; the work that would use one is skipped and counted by reason.
    uint64_t invalid_bind_group_layouts = 0;   // created with an error (or over a per-stage limit)
    uint64_t invalid_pipeline_layouts = 0;     // a group's layout is invalid: never created
    uint64_t invalid_bind_groups = 0;          // created with an error (an invalid view, a usage mismatch, ...)
    uint64_t scope_pop_waits = 0;              // created_valid pops that were not complete at once (WaitAny)
    uint64_t invalid_pipelines = 0;            // sync creation error, or an invalid explicit layout: never handed to a pass
    uint64_t skip_invalid_bind_group = 0;      // draws / dispatches skipped: a bound group is invalid
    uint64_t skip_invalid_bind_group_sets = 0; // SetBindGroup of an invalid group: not encoded
    std::map<std::string, uint64_t> packet_error_kinds;  // packet errors by message (numbers folded), bounded
    std::string last_device_error;
    std::vector<std::string> device_error_log;  // the first 20 uncaptured device errors
    // Object lifetime (memory_model.h): ids DestroyResource dropped per kind, and the cache entries
    // (view_meta / bg_infos) purged because they still referenced a destroyed texture, buffer or group.
    uint64_t textures_destroyed = 0;
    uint64_t buffers_destroyed = 0;
    uint64_t bind_groups_destroyed = 0;
    uint64_t cache_purges = 0;
    uint64_t cache_views_purged = 0;
    uint64_t cache_bg_infos_purged = 0;
    // Bind group budget (PG6_BIND_GROUP_BUDGET, default 131072 live Dawn groups): groups whose Dawn object
    // was dropped, rebuilt from the journal when bound again, and rebuilds that failed.
    uint64_t bind_groups_evicted = 0;
    uint64_t bind_groups_recreated = 0;
    uint64_t bind_groups_recreate_failed = 0;
    uint64_t bind_group_budget_passes = 0;
};

struct ShowTargetCandidate {
    uint32_t id = 0;
    bool depth = false;
};

struct AdapterDescription {
    std::string backend;
    std::string vendor;
    std::string architecture;
    std::string device;
    std::string description;
};

// A presented frame read back from the GPU: rows tightly packed (width * 4 bytes, or * 8 for
// rgba16float), in the swapchain's own format.
struct FrameImage {
    uint32_t width = 0;
    uint32_t height = 0;
    std::string format;  // WebGPU name: "bgra8unorm", "rgba8unorm", "rgba16float"
    uint32_t bytes_per_pixel = 4;
    std::vector<uint8_t> pixels;
};

class NativeExecutor {
public:
    enum class CreateStatus { Ok, NoAdapter, Failed };

    // nullptr with *status NoAdapter when the host has no WebGPU adapter (tests skip), Failed for
    // any other error (*error says what).
    static std::unique_ptr<NativeExecutor> create(const ExecutorOptions& options, CreateStatus* status,
                                                  std::string* error);
    ~NativeExecutor();
    NativeExecutor(const NativeExecutor&) = delete;
    NativeExecutor& operator=(const NativeExecutor&) = delete;

    // Executes one batch of complete packets (as handed to os::wgpu_submit). Returns the number of
    // Present packets in it (the worker's `presents`).
    uint32_t execute(const uint8_t* data, size_t len);

    // Loads a .wgsl.pack (WGPK v1) for hash-only CreateShaderModule payloads, like the replay page's
    // `processor.loadWgslPack`. Only the entry table is read now; WGSL is read on demand.
    // A `<path>.reflect.json` sidecar next to the pack is loaded too (the replay page / boot
    // orchestrator derive the same URL); a bad sidecar is counted (reflection_load_errors), it does
    // not fail the pack.
    bool load_wgsl_pack(const std::string& path, std::string* error);

    // Loads a reflection document: a `playgta6-stage-reflection` sidecar (`*.reflect.json`, versions
    // 1..3) or an S6 `layout-manifest.json`, like `processor.loadReflectionManifest`. Later documents
    // overwrite records with the same hash. False + *error when the loader rejects the document.
    bool load_reflection(const std::string& path, std::string* error);
    bool load_reflection_text(const std::string& json, std::string* error);

    // Processes WebGPU callbacks and, with a window, window messages. Returns false once the window
    // was closed (always true headless).
    bool poll();
    // True while the presenting window has keyboard focus (false headless).
    bool window_focused() const;
    // Title bar text of the live window (no-op without one). Call from the pump thread.
    void set_window_title(const std::string& title);
    // Stats overlay (F3): on / off, and the two things only the host knows: live-audio underruns (< 0: no audio) and
    // the reason the guest stopped ("": running; non-empty is drawn in red and the last frame is redrawn with it).
    // Call from the pump thread.
    void set_overlay(bool on);
    bool overlay_enabled() const;
    void set_overlay_extra(int64_t audio_underruns, const std::string& fault);
    // The stop message (guest/stop_message.h): a box that is drawn on the last frame even with the F3 overlay off.
    void set_overlay_banner(const std::string& text);
    // --show-target at run time: selector grammar in show_target_model.h; "" switches it off. False + *error
    // for a selector that does not parse.
    bool set_show_target(const std::string& selector, std::string* error);
    // The hotkeys as bits (show_target_model.h kHotkeyMip / Layer / Prev / Next): what poll() does with F7..F10.
    void show_target_hotkeys(uint32_t bits);
    // What the last present showed ("tex 12 depth 1280x720 depth32float mip 0/1 layer 0/1"), the reason it could
    // not ("unavailable: ..."), "" when the swapchain image was shown.
    std::string shown_target_label() const;
    // Render targets of the last presented frame (the hotkey cycle list), sorted by id.
    std::vector<ShowTargetCandidate> shown_target_candidates() const;
    // Present mode in effect once the surface is configured ("" before / headless).
    std::string present_mode_name() const;
    // Path of the newest screenshot written (wait_idle() first), "" if none.
    std::string last_screenshot_path() const;
    // Blocks until the GPU finished all submitted work and every pending read-back / timestamp
    // read was delivered.
    void wait_idle();

    // Where read-back results (ReadbackBuffer / ReadbackTexture) are written: the worker's
    // attachGuestMemory. Without it a read-back writes nothing and does not move its counter.
    // The memory must outlive the executor's pending read-backs (wait_idle() drains them).
    void attach_guest_memory(const GuestMemory& memory);

    // Test hook: loses the device now (wgpu Device::ForceLoss, reason Unknown) so the recovery runs on
    // the next packet / poll exactly as for a driver reset.
    void debug_force_device_lost(const std::string& message);
    // Test hook: the next `count` device creations fail, so a recovery ends fatal without a device.
    void debug_fail_device_creations(int count);
    // Test hook: submits a compute dispatch that never ends (a real GPU hang: on Windows the driver resets the
    // adapter after ~2 s, DXGI_ERROR_DEVICE_HUNG), so a recovery from a real TDR can be timed. Counted in
    // stats().debug_gpu_hangs. PG6_DEBUG_GPU_HANG_AT_FRAME=<n> does the same right after frame n is submitted.
    void debug_hang_gpu();
    bool faulted() const;

    uint64_t frame_count() const;
    const ExecutorStats& stats() const;
    const AdapterDescription& adapter() const;
    std::string stats_json() const;

    // Headless only: the swapchain contents after the last submitted frame.
    bool read_last_frame(FrameImage* out, std::string* error);
    // Reads back mip `mip` of a stream texture (needs COPY_SRC), tightly packed rows. Uncompressed
    // 2D textures only.
    bool read_texture(uint32_t id, uint32_t mip, std::vector<uint8_t>* out, uint32_t* width, uint32_t* height,
                      std::string* error);
    // Reads a stream buffer (it needs COPY_SRC usage; WebGPU rejects the copy otherwise).
    bool read_buffer(uint32_t id, uint64_t offset, uint64_t size, std::vector<uint8_t>* out, std::string* error);

    struct Impl;

private:
    explicit NativeExecutor(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

}  // namespace playgta6::gpu::native_exec
