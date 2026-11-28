// ID3D12GraphicsCommandList on the state bridge, command list creation and ExecuteCommandLists.
// See d3d12_guest_resources.h for the data-flow model.

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <memory>

#include "gpu/d3d12_guest_resources.h"
#include "gpu/binding_fallbacks.h"
#include "gpu/drop_table.h"
#include "gpu/gpu_va_watch.h"
#include "os/guest_window.h"
#include "os/thread_spans.h"

namespace playgta6::gpu::d3d12_guest {

namespace {
std::atomic<uint64_t> g_work[static_cast<size_t>(Work::Count)] = {};
}  // namespace

void count_work(Work w, uint64_t n) {
    g_work[static_cast<size_t>(w)].fetch_add(n, std::memory_order_relaxed);
}

uint64_t work_total(Work w) {
    return g_work[static_cast<size_t>(w)].load(std::memory_order_relaxed);
}

const char* work_name(Work w) {
    switch (w) {
    case Work::DrawsIssued: return "draws_issued";
    case Work::DrawsEmitted: return "draws_emitted";
    case Work::DrawsNoShaderModule: return "draws_no_shader_module";
    case Work::DrawsMeshPipeline: return "draws_mesh_pipeline";
    case Work::DrawsRectList: return "draws_rect_list";
    case Work::DrawsDroppedByBridge: return "draws_dropped_by_bridge";
    case Work::DispatchesIssued: return "dispatches_issued";
    case Work::DispatchesEmitted: return "dispatches_emitted";
    case Work::DispatchesNoShaderModule: return "dispatches_no_shader_module";
    case Work::DispatchesDroppedByBridge: return "dispatches_dropped_by_bridge";
    case Work::IndirectExecutionsSkipped: return "indirect_executions_skipped";
    case Work::IndirectExecutionsEmitted: return "indirect_executions_emitted";
    case Work::IndirectVertexIndexArgsSkipped: return "indirect_vertex_index_args_skipped";
    case Work::IndirectRootViewArgsSkipped: return "indirect_root_view_args_skipped";
    case Work::TextureUploads: return "texture_uploads";
    case Work::TextureSyncViewSkips: return "texture_sync_view_skips";
    case Work::TextureSyncViewTracked: return "texture_sync_view_tracked";
    case Work::TextureSyncUntracked: return "texture_sync_untracked";
    case Work::TextureSyncOwnedChecks: return "texture_sync_owned_checks";
    case Work::TextureSyncOwnedHashSkips: return "texture_sync_owned_hash_skips";
    case Work::BufferUploads: return "buffer_uploads";
    case Work::CopiesSkipped: return "copies_skipped";
    case Work::CopySkipGateOff: return "copy_skip_gate_off";
    case Work::CopySkipNoGpuTexture: return "copy_skip_no_gpu_texture";
    case Work::CopySkipMsaa: return "copy_skip_msaa";
    case Work::CopySkipSwapchain: return "copy_skip_swapchain";
    case Work::CopySkipDimension: return "copy_skip_dimension";
    case Work::CopySkipFormat: return "copy_skip_format";
    case Work::CopySkipOverlap: return "copy_skip_overlap";
    case Work::CopySkipEmitFailed: return "copy_skip_emit_failed";
    case Work::ResourcesCreated: return "resources_created";
    case Work::ResourcesDestroyed: return "resources_destroyed";
    case Work::ReflectionTableSlotsOutsideRs: return "reflection_table_slots_outside_rs";
    case Work::GpuObjectsDeferred: return "gpu_objects_deferred";
    case Work::GpuObjectsCreatedOnUse: return "gpu_objects_created_on_use";
    case Work::GpuObjectsNeverCreated: return "gpu_objects_never_created";
    case Work::ReflectionLaneIndexedUnbound: return "reflection_lane_indexed_unbound";
    case Work::BindGroupCacheHits: return "bind_group_cache_hits";
    case Work::BindGroupCacheMisses: return "bind_group_cache_misses";
    case Work::BindGroupsCreated: return "bind_groups_created";
    case Work::PipelinesCreated: return "pipelines_created";
    case Work::ReflectionRootDescriptorTables: return "reflection_root_descriptor_tables";
    case Work::ReflectionVertexInputsDefaulted: return "reflection_vertex_inputs_defaulted";
    case Work::ReflectionUnresolvedBindings: return "reflection_unresolved_bindings";
    case Work::ReadbacksIssued: return "readbacks_issued";
    case Work::ReadbackBytes: return "readback_bytes";
    case Work::ReadbacksCompleted: return "readbacks_completed";
    case Work::ReadbackSignalsHeld: return "readback_signals_held";
    case Work::ReadbackTimeouts: return "readback_timeouts";
    case Work::MipClampedBinds: return "mip_clamped_binds";
    case Work::StorageViewsDepthUntwinned: return "storage_views_depth_untwinned";
    case Work::DepthTwinBindings: return "depth_twin_bindings";
    case Work::DepthTwinCopiesIn: return "depth_twin_copies_in";
    case Work::DepthTwinCopiesOut: return "depth_twin_copies_out";
    case Work::QueryBegins: return "query_begins";
    case Work::QueryEnds: return "query_ends";
    case Work::QueryResolves: return "query_resolves";
    case Work::QueryOcclusionOps: return "query_occlusion_ops";
    case Work::QueryTimestampOps: return "query_timestamp_ops";
    case Work::QueryOtherOps: return "query_other_ops";
    case Work::QueryResultsUnwritten: return "query_results_unwritten";
    case Work::QueryPackets: return "query_packets";
    case Work::QueryResultsWritten: return "query_results_written";
    case Work::ComputeTextureReadbacks: return "compute_texture_readbacks";
    case Work::ComputeTextureRetiled: return "compute_texture_retiled";
    case Work::ComputeTextureRetileDiscarded: return "compute_texture_retile_discarded";
    case Work::StencilTwinBindings: return "stencil_twin_bindings";
    case Work::StencilTwinCopiesIn: return "stencil_twin_copies_in";
    case Work::StencilTwinCopiesOut: return "stencil_twin_copies_out";
    case Work::StencilTwinDraws: return "stencil_twin_draws";
    case Work::ShadowComparedBytes: return "shadow_compared_bytes";
    case Work::ShadowFullCompares: return "shadow_full_compares";
    case Work::ShadowFullCompareBytes: return "shadow_full_compare_bytes";
    case Work::FlushIndexTakes: return "flush_index_takes";
    case Work::FlushIndexTakeFailures: return "flush_index_take_failures";
    case Work::FlushIndexDirtyPages: return "flush_index_dirty_pages";
    case Work::FlushBuffersVisited: return "flush_buffers_visited";
    case Work::FlushBuffersSkipped: return "flush_buffers_skipped";
    case Work::FlushBuffersUntracked: return "flush_buffers_untracked";
    case Work::FlushUniformsCompared: return "flush_uniforms_compared";
    case Work::FlushUniformsSkipped: return "flush_uniforms_skipped";
    case Work::FlushStorageCompared: return "flush_storage_compared";
    case Work::FlushStorageSkipped: return "flush_storage_skipped";
    case Work::StorageRelocationMisses: return "storage_relocation_misses";
    case Work::StorageReuseScanSteps: return "storage_reuse_scan_steps";
    case Work::FlushIaCompared: return "flush_ia_compared";
    case Work::FlushIaSkipped: return "flush_ia_skipped";
    case Work::FlushCrosscheckLate: return "flush_crosscheck_late";
    case Work::FlushCrosscheckMisses: return "flush_crosscheck_misses";
    case Work::FlushIndexedCalls: return "flush_indexed_calls";
    case Work::FlushIndexedNs: return "flush_indexed_ns";
    case Work::FlushBufferNs: return "flush_buffer_ns";
    case Work::FlushUniformNs: return "flush_uniform_ns";
    case Work::Count: break;
    }
    return "?";
}

namespace {
void count_object_reuse(const bridge::BridgeStats& a, const bridge::BridgeStats& b) {
    count_work(Work::BindGroupCacheHits, b.bind_group_cache_hits - a.bind_group_cache_hits);
    count_work(Work::BindGroupCacheMisses, b.bind_group_cache_misses - a.bind_group_cache_misses);
    count_work(Work::BindGroupsCreated, b.bind_groups_created - a.bind_groups_created);
    count_work(Work::PipelinesCreated, b.pipelines_created - a.pipelines_created);
    count_work(Work::MipClampedBinds, b.mip_clamped_binds - a.mip_clamped_binds);
    count_work(Work::StorageViewsDepthUntwinned, b.storage_views_depth_untwinned - a.storage_views_depth_untwinned);
    count_work(Work::DepthTwinBindings, b.depth_twin_bindings - a.depth_twin_bindings);
    count_work(Work::DepthTwinCopiesIn, b.depth_twin_copies_in - a.depth_twin_copies_in);
    count_work(Work::DepthTwinCopiesOut, b.depth_twin_copies_out - a.depth_twin_copies_out);
    count_work(Work::StencilTwinBindings, b.stencil_twin_bindings - a.stencil_twin_bindings);
    count_work(Work::StencilTwinCopiesIn, b.stencil_twin_copies_in - a.stencil_twin_copies_in);
    count_work(Work::StencilTwinCopiesOut, b.stencil_twin_copies_out - a.stencil_twin_copies_out);
    count_work(Work::StencilTwinDraws, b.stencil_twin_draws - a.stencil_twin_draws);
}

// Counts the outcome of one recorded draw from the state bridge's statistics: emitted if the
// bridge drew, dropped by the bridge if it skipped, otherwise `fallback` (set by the caller's own
// skip paths).
// Draws / dispatches that bound a null/dummy resource (BridgeStats::null_bound_*), per shader identity.
// Runs only when the counters moved, so a clean draw pays two compares.
void note_binding_fallbacks(const std::vector<uint64_t>* identities, const bridge::BridgeStats& a,
                            const bridge::BridgeStats& b, bool dispatch) {
    if (identities == nullptr || (a.null_bound_unresolved == b.null_bound_unresolved &&
                                  a.null_bound_empty == b.null_bound_empty)) {
        return;
    }
    const bool unresolved = b.null_bound_unresolved != a.null_bound_unresolved;
    for (uint64_t id : *identities) {
        if (id != 0) {
            binding_fallback_table().record(id, unresolved ? BindingFallbackTable::Reason::Unresolved
                                                           : BindingFallbackTable::Reason::EmptySlot,
                                            dispatch);
        }
    }
}

class DrawOutcome {
public:
    explicit DrawOutcome(const bridge::CommandListBridge& b)
        : bridge_(b), drawn_(b.stats().draws), dropped_(b.stats().draws_skipped), before_(b.stats()) {
        count_work(Work::DrawsIssued);
    }
    ~DrawOutcome() {
        count_object_reuse(before_, bridge_.stats());
        note_binding_fallbacks(identities, before_, bridge_.stats(), false);
        if (bridge_.stats().draws != drawn_) {
            count_work(Work::DrawsEmitted);
        } else if (bridge_.stats().draws_skipped != dropped_) {
            count_work(Work::DrawsDroppedByBridge);
            note_drop(DropTable::Reason::Bridge);
        } else {
            count_work(fallback);
            note_drop(fallback == Work::DrawsNoShaderModule ? DropTable::Reason::NoModule
                      : fallback == Work::DrawsMeshPipeline ? DropTable::Reason::MeshRefused
                      : fallback == Work::DrawsRectList     ? DropTable::Reason::RectList
                                                            : DropTable::Reason::Bridge);
        }
    }
    Work fallback = Work::DrawsDroppedByBridge;
    // Shader identities of the bound pipeline state (null: none); drops are tabled per identity.
    const std::vector<uint64_t>* identities = nullptr;

private:
    void note_drop(DropTable::Reason r) const {
        if (identities == nullptr) {
            return;
        }
        for (uint64_t id : *identities) {
            if (id != 0) {
                drop_table().record(id, r, false);
            }
        }
    }
    const bridge::CommandListBridge& bridge_;
    uint64_t drawn_;
    uint64_t dropped_;
    bridge::BridgeStats before_;
};
}  // namespace

namespace {

constexpr uint32_t kMaxViewports = 16;
constexpr uint32_t kMaxRects = 64;
constexpr uint32_t kMaxBarriers = 4096;
constexpr uint32_t kMaxListsPerExecute = 1024;
constexpr uint64_t kMaxWriteChunk = 1u << 20;

std::string hex(uint64_t v) {
    char b[24];
    std::snprintf(b, sizeof(b), "0x%" PRIx64, v);
    return b;
}

template <typename T>
bool read_n(uint64_t va, uint32_t count, std::vector<T>* out) {
    out->assign(count, T{});
    return count == 0 || (va != 0 && com::read_guest(va, out->data(), sizeof(T) * count));
}

Resource* as_resource(uint64_t va) {
    auto arg = InterfaceArg<IResourceHost>::resolve(va);
    return arg ? static_cast<Resource*>(arg.get()) : nullptr;
}

constexpr uint64_t align_up_u64(uint64_t v, uint64_t a) {
    return a <= 1u ? v : ((v + a - 1u) / a) * a;
}

constexpr uint32_t align_up_u32(uint32_t v, uint32_t a) {
    return a <= 1u ? v : ((v + a - 1u) / a) * a;
}

bool is_depth_dxgi_format(uint32_t format) {
    switch (static_cast<pso::DxgiFormat>(format)) {
    case pso::DxgiFormat::R32G8X24_TYPELESS:
    case pso::DxgiFormat::D32_FLOAT_S8X24_UINT:
    case pso::DxgiFormat::R32_FLOAT_X8X24_TYPELESS:
    case pso::DxgiFormat::X32_TYPELESS_G8X24_UINT:
    case pso::DxgiFormat::R32_TYPELESS:
    case pso::DxgiFormat::D32_FLOAT:
    case pso::DxgiFormat::R24G8_TYPELESS:
    case pso::DxgiFormat::D24_UNORM_S8_UINT:
    case pso::DxgiFormat::R24_UNORM_X8_TYPELESS:
    case pso::DxgiFormat::X24_TYPELESS_G8_UINT:
    case pso::DxgiFormat::R16_TYPELESS:
    case pso::DxgiFormat::D16_UNORM:
    case pso::DxgiFormat::D16_UNORM_S8_UINT:
    case pso::DxgiFormat::R16_UNORM_X8_TYPELESS:
    case pso::DxgiFormat::X16_TYPELESS_G8_UINT:
        return true;
    default:
        return false;
    }
}

struct SubresourceInfo {
    uint32_t mip = 0;
    uint32_t layer = 0;
    uint32_t plane = 0;
    uint32_t mip_w = 1;
    uint32_t mip_h = 1;
    uint32_t mip_d = 1;
    uint32_t phys_w = 1;
    uint32_t phys_h = 1;
    bool is_1d = false;
    bool is_3d = false;
};

bool decode_texture_subresource(const Resource* tex, uint32_t sub, const DxgiFormatLayout& layout,
                                SubresourceInfo* out) {
    if (tex == nullptr || tex->is_buffer() || sub >= tex->subresource_count()) {
        return false;
    }
    const uint32_t mips = std::max<uint32_t>(1u, tex->mip_levels());
    const uint32_t layers = std::max<uint32_t>(1u, tex->array_layers());
    const bool is_1d = tex->desc().Dimension == ResourceDimension::Texture1D;
    const bool is_3d = tex->desc().Dimension == ResourceDimension::Texture3D;
    const uint32_t mip = sub % mips;
    const uint32_t layer = (sub / mips) % layers;
    const uint32_t plane = sub / (mips * layers);
    const uint32_t bw = std::max<uint32_t>(1u, layout.block_width);
    const uint32_t bh = std::max<uint32_t>(1u, layout.block_height);
    const uint32_t mip_w = std::max<uint32_t>(1u, static_cast<uint32_t>(tex->desc().Width >> mip));
    const uint32_t mip_h = is_1d ? 1u : std::max<uint32_t>(1u, tex->desc().Height >> mip);
    const uint32_t mip_d = is_3d ? std::max<uint32_t>(1u, static_cast<uint32_t>(tex->desc().DepthOrArraySize) >> mip) : 1u;
    out->mip = mip;
    out->layer = layer;
    out->plane = plane;
    out->mip_w = mip_w;
    out->mip_h = mip_h;
    out->mip_d = mip_d;
    out->phys_w = align_up_u32(mip_w, bw);
    out->phys_h = is_1d ? 1u : align_up_u32(mip_h, bh);
    out->is_1d = is_1d;
    out->is_3d = is_3d;
    return true;
}

bool formats_compatible_for_resolve(uint32_t a, uint32_t b) {
    if (a == b) {
        return true;
    }
    const DxgiFormatLayout la = dxgi_format_layout(a);
    const DxgiFormatLayout lb = dxgi_format_layout(b);
    if (!la.known || !lb.known) {
        return false;
    }
    return la.block_bytes == lb.block_bytes && la.block_width == lb.block_width &&
           la.block_height == lb.block_height && la.planes == lb.planes;
}

float normalize_zero(float v) {
    return v == 0.0f ? 0.0f : v;
}

// Clamps a D3D12 viewport to WebGPU-valid bounds:
//   x >= 0, y >= 0, width >= 0, height >= 0, x + width <= limit_w, y + height <= limit_h,
//   0 <= minDepth <= maxDepth <= 1, all finite.
pso::D3D12Viewport clamp_viewport_webgpu(const pso::D3D12Viewport& vp, uint32_t rt_w, uint32_t rt_h,
                                         uint32_t max_dim) {
    const float limit_w = static_cast<float>(rt_w > 0 ? std::min(rt_w, max_dim) : max_dim);
    const float limit_h = static_cast<float>(rt_h > 0 ? std::min(rt_h, max_dim) : max_dim);
    float x = std::isfinite(vp.TopLeftX) ? vp.TopLeftX : 0.0f;
    float y = std::isfinite(vp.TopLeftY) ? vp.TopLeftY : 0.0f;
    float w = std::isfinite(vp.Width) ? vp.Width : 0.0f;
    float h = std::isfinite(vp.Height) ? vp.Height : 0.0f;
    if (w < 0.0f) {
        w = 0.0f;
    }
    if (h < 0.0f) {
        h = 0.0f;
    }
    if (x < 0.0f) {
        w = std::max(0.0f, w + x);
        x = 0.0f;
    }
    if (y < 0.0f) {
        h = std::max(0.0f, h + y);
        y = 0.0f;
    }
    x = std::clamp(x, 0.0f, limit_w);
    y = std::clamp(y, 0.0f, limit_h);
    w = std::clamp(w, 0.0f, std::max(0.0f, limit_w - x));
    h = std::clamp(h, 0.0f, std::max(0.0f, limit_h - y));

    float min_d = std::isfinite(vp.MinDepth) ? std::clamp(vp.MinDepth, 0.0f, 1.0f) : 0.0f;
    float max_d = std::isfinite(vp.MaxDepth) ? std::clamp(vp.MaxDepth, 0.0f, 1.0f) : 1.0f;
    if (min_d > max_d) {
        std::swap(min_d, max_d);
    }
    return pso::D3D12Viewport{normalize_zero(x), normalize_zero(y), normalize_zero(w), normalize_zero(h),
                              normalize_zero(min_d), normalize_zero(max_d)};
}

// Clamps a D3D12 scissor rect against render-target extents (or max_dim when no RT is bound yet).
pso::D3D12Rect clamp_scissor_webgpu(const pso::D3D12Rect& r, uint32_t rt_w, uint32_t rt_h, uint32_t max_dim) {
    const int64_t limit_w = static_cast<int64_t>(rt_w > 0 ? std::min(rt_w, max_dim) : max_dim);
    const int64_t limit_h = static_cast<int64_t>(rt_h > 0 ? std::min(rt_h, max_dim) : max_dim);
    const int64_t x0 = std::clamp<int64_t>(r.left, 0, limit_w);
    const int64_t y0 = std::clamp<int64_t>(r.top, 0, limit_h);
    const int64_t x1 = std::clamp<int64_t>(r.right, x0, limit_w);
    const int64_t y1 = std::clamp<int64_t>(r.bottom, y0, limit_h);
    return pso::D3D12Rect{static_cast<int32_t>(x0), static_cast<int32_t>(y0), static_cast<int32_t>(x1),
                          static_cast<int32_t>(y1)};
}

// Per-CommandList auxiliary state for G79 (kept in this TU to preserve d3d12_guest_resources.h unchanged).
struct CommandListAux {
    struct BoundDsv {
        uint32_t resource_id = 0;
        uint32_t mip = 0;
        uint32_t base_layer = 0;
        uint32_t layer_count = 1;
        bool depth_read_only = false;
    };
    struct DeferredMeta {
        uint64_t dst_cpu_va = 0;
        uint64_t dst_width = 0;
        uint64_t dst_allocated = 0;
        bool needs_rmw_align = false;
    };
    struct ComputeRootArg {
        uint8_t kind = 0;  // 0 unset, 1 table
        uint64_t value = 0;
    };

    std::vector<pso::D3D12Viewport> raw_viewports;
    std::vector<pso::D3D12Viewport> clamped_viewports;
    std::vector<pso::D3D12Rect> raw_scissors;
    std::vector<pso::D3D12Rect> clamped_scissors;
    float blend_factor[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    uint32_t stencil_ref = 0;
    uint32_t rt_width = 0;
    uint32_t rt_height = 0;
    BoundDsv bound_dsv{};
    const d3d::RootSignature* compute_root_signature = nullptr;
    std::array<ComputeRootArg, 64> compute_root_args{};
    bridge::DepthTwinTracker depth_twin;
    std::vector<DeferredMeta> deferred_meta;
    uint64_t depth_srv_sync_flushes = 0;
    uint64_t depth_srv_sync_refreshes = 0;
    uint64_t resolves_emitted = 0;
    uint64_t resolves_fallback = 0;

    void reset_dynamic_state() {
        raw_viewports.clear();
        clamped_viewports.clear();
        raw_scissors.clear();
        clamped_scissors.clear();
        blend_factor[0] = blend_factor[1] = blend_factor[2] = blend_factor[3] = 1.0f;
        stencil_ref = 0;
        rt_width = 0;
        rt_height = 0;
        bound_dsv = {};
        compute_root_signature = nullptr;
        compute_root_args = {};
    }

    void reset_all() {
        reset_dynamic_state();
        depth_twin.reset();
        deferred_meta.clear();
        depth_srv_sync_flushes = 0;
        depth_srv_sync_refreshes = 0;
        resolves_emitted = 0;
        resolves_fallback = 0;
    }
};

std::mutex g_aux_mutex;
std::unordered_map<const CommandList*, std::unique_ptr<CommandListAux>> g_aux_map;

CommandListAux& aux_of(const CommandList* list) {
    std::lock_guard<std::mutex> lock(g_aux_mutex);
    auto& ptr = g_aux_map[list];
    if (!ptr) {
        ptr = std::make_unique<CommandListAux>();
    }
    return *ptr;
}

void erase_aux(const CommandList* list) {
    std::lock_guard<std::mutex> lock(g_aux_mutex);
    g_aux_map.erase(list);
}

// ---------------------------------------------------------------------------------------------
// GPU-to-GPU texture copy passes (ABI v2 render-pass blit & compute-pass storage copy)
// ---------------------------------------------------------------------------------------------

enum class CopySampleKind : uint8_t {
    Unsupported = 0,
    Float,
    Uint,
    Sint,
    Depth,
};

CopySampleKind copy_sample_kind(pso::WgpuTextureFormat fmt) noexcept {
    using F = pso::WgpuTextureFormat;
    switch (fmt) {
    case F::Depth16Unorm:
    case F::Depth24Plus:
    case F::Depth24PlusStencil8:
    case F::Depth32Float:
    case F::Depth32FloatStencil8:
        return CopySampleKind::Depth;
    case F::R8Uint:
    case F::R16Uint:
    case F::R32Uint:
    case F::Rg8Uint:
    case F::Rg16Uint:
    case F::Rg32Uint:
    case F::Rgba8Uint:
    case F::Rgba16Uint:
    case F::Rgba32Uint:
    case F::Rgb10a2Uint:
        return CopySampleKind::Uint;
    case F::R8Sint:
    case F::R16Sint:
    case F::R32Sint:
    case F::Rg8Sint:
    case F::Rg16Sint:
    case F::Rg32Sint:
    case F::Rgba8Sint:
    case F::Rgba16Sint:
    case F::Rgba32Sint:
        return CopySampleKind::Sint;
    case F::R8Unorm:
    case F::R8Snorm:
    case F::R16Unorm:
    case F::R16Snorm:
    case F::R16Float:
    case F::R32Float:
    case F::Rg8Unorm:
    case F::Rg8Snorm:
    case F::Rg16Unorm:
    case F::Rg16Snorm:
    case F::Rg16Float:
    case F::Rg32Float:
    case F::Rgba8Unorm:
    case F::Rgba8UnormSrgb:
    case F::Rgba8Snorm:
    case F::Bgra8Unorm:
    case F::Bgra8UnormSrgb:
    case F::Rgb10a2Unorm:
    case F::Rg11b10Ufloat:
    case F::Rgb9e5Ufloat:
    case F::Rgba16Unorm:
    case F::Rgba16Snorm:
    case F::Rgba16Float:
    case F::Rgba32Float:
        return CopySampleKind::Float;
    default:
        return CopySampleKind::Unsupported;
    }
}

pso::WgpuTextureFormat linear_copy_view_format(pso::WgpuTextureFormat fmt) noexcept {
    if (fmt == pso::WgpuTextureFormat::Rgba8UnormSrgb) {
        return pso::WgpuTextureFormat::Rgba8Unorm;
    }
    if (fmt == pso::WgpuTextureFormat::Bgra8UnormSrgb) {
        return pso::WgpuTextureFormat::Bgra8Unorm;
    }
    return fmt;
}

const char* wgsl_storage_format_name(pso::WgpuTextureFormat fmt, uint32_t features) noexcept {
    using F = pso::WgpuTextureFormat;
    switch (fmt) {
    case F::Rgba8Unorm: return "rgba8unorm";
    case F::Rgba8Snorm: return "rgba8snorm";
    case F::Rgba8Uint: return "rgba8uint";
    case F::Rgba8Sint: return "rgba8sint";
    case F::Rgba16Uint: return "rgba16uint";
    case F::Rgba16Sint: return "rgba16sint";
    case F::Rgba16Float: return "rgba16float";
    case F::R32Uint: return "r32uint";
    case F::R32Sint: return "r32sint";
    case F::R32Float: return "r32float";
    case F::Rg32Uint: return "rg32uint";
    case F::Rg32Sint: return "rg32sint";
    case F::Rg32Float: return "rg32float";
    case F::Rgba32Uint: return "rgba32uint";
    case F::Rgba32Sint: return "rgba32sint";
    case F::Rgba32Float: return "rgba32float";
    case F::Bgra8Unorm:
        return (features & pso::WgpuFeature::Bgra8UnormStorage) != 0u ? "bgra8unorm" : nullptr;
    default:
        return nullptr;
    }
}

pso::WgpuTextureFormat resource_wgpu_format(const Resource* r, uint32_t features) noexcept {
    if (r == nullptr || r->is_buffer()) {
        return pso::WgpuTextureFormat::Undefined;
    }
    const bool depth = (r->desc().Flags & ResourceFlags::AllowDepthStencil) != 0u;
    const bool rt = (r->desc().Flags & ResourceFlags::AllowRenderTarget) != 0u;
    const pso::FormatUsage usage = depth ? pso::FormatUsage::DepthStencil
                                         : (rt ? pso::FormatUsage::RenderTarget : pso::FormatUsage::Sampled);
    const pso::TextureFormatMapping m =
        pso::map_texture_format(static_cast<pso::DxgiFormat>(r->desc().Format), usage, features);
    return m.supported() ? m.format : pso::WgpuTextureFormat::Undefined;
}

bool copy_kinds_compatible(CopySampleKind src_kind, CopySampleKind dst_kind, bool allow_depth_cross) noexcept {
    if (src_kind == CopySampleKind::Unsupported || dst_kind == CopySampleKind::Unsupported) {
        return false;
    }
    if (src_kind == dst_kind) {
        return true;
    }
    if (allow_depth_cross) {
        return (src_kind == CopySampleKind::Depth && dst_kind == CopySampleKind::Float) ||
               (src_kind == CopySampleKind::Float && dst_kind == CopySampleKind::Depth);
    }
    return false;
}

uint32_t pack_signed_delta_i16(int32_t dx, int32_t dy) noexcept {
    const uint32_t ux = static_cast<uint32_t>(static_cast<uint16_t>(dx));
    const uint32_t uy = static_cast<uint32_t>(static_cast<uint16_t>(dy));
    return ux | (uy << 16u);
}

constexpr const char* kTexCopyBlitVs = R"(
struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) @interpolate(flat) delta: vec2<i32>,
};
@vertex fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) inst: u32) -> VsOut {
  let p = vec2<f32>(f32((vi << 1u) & 2u), f32(vi & 2u));
  let raw_x = i32(inst & 0xffffu);
  let raw_y = i32((inst >> 16u) & 0xffffu);
  let dx = select(raw_x, raw_x - 65536, raw_x >= 32768);
  let dy = select(raw_y, raw_y - 65536, raw_y >= 32768);
  return VsOut(vec4<f32>(p * 2.0 - 1.0, 0.0, 1.0), vec2<i32>(dx, dy));
}
)";

constexpr const char* kTexCopyFloatFs = R"(
struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) @interpolate(flat) delta: vec2<i32>,
};
@group(0) @binding(0) var src_tex: texture_2d<f32>;
@fragment fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
  return textureLoad(src_tex, vec2<i32>(in.pos.xy) + in.delta, 0);
}
)";

constexpr const char* kTexCopyUintFs = R"(
struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) @interpolate(flat) delta: vec2<i32>,
};
@group(0) @binding(0) var src_tex: texture_2d<u32>;
@fragment fn fs_main(in: VsOut) -> @location(0) vec4<u32> {
  return textureLoad(src_tex, vec2<i32>(in.pos.xy) + in.delta, 0);
}
)";

constexpr const char* kTexCopySintFs = R"(
struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) @interpolate(flat) delta: vec2<i32>,
};
@group(0) @binding(0) var src_tex: texture_2d<i32>;
@fragment fn fs_main(in: VsOut) -> @location(0) vec4<i32> {
  return textureLoad(src_tex, vec2<i32>(in.pos.xy) + in.delta, 0);
}
)";

constexpr const char* kTexCopyDepthFs = R"(
struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) @interpolate(flat) delta: vec2<i32>,
};
struct FsOut { @builtin(frag_depth) depth: f32 };
@group(0) @binding(0) var src_tex: texture_depth_2d;
@fragment fn fs_main(in: VsOut) -> FsOut {
  return FsOut(textureLoad(src_tex, vec2<i32>(in.pos.xy) + in.delta, 0));
}
)";

constexpr const char* kTexCopyDepthToFloatFs = R"(
struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) @interpolate(flat) delta: vec2<i32>,
};
@group(0) @binding(0) var src_tex: texture_depth_2d;
@fragment fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
  let d = textureLoad(src_tex, vec2<i32>(in.pos.xy) + in.delta, 0);
  return vec4<f32>(d, 0.0, 0.0, 1.0);
}
)";

constexpr const char* kTexCopyFloatToDepthFs = R"(
struct VsOut {
  @builtin(position) pos: vec4<f32>,
  @location(0) @interpolate(flat) delta: vec2<i32>,
};
struct FsOut { @builtin(frag_depth) depth: f32 };
@group(0) @binding(0) var src_tex: texture_2d<f32>;
@fragment fn fs_main(in: VsOut) -> FsOut {
  let v = textureLoad(src_tex, vec2<i32>(in.pos.xy) + in.delta, 0);
  return FsOut(clamp(v.x, 0.0, 1.0));
}
)";

std::string build_compute_copy_wgsl(const char* storage_fmt_name, CopySampleKind src_kind, bool is_3d,
                                    bool with_params) {
    const bool depth_src = src_kind == CopySampleKind::Depth;
    const char* scalar = src_kind == CopySampleKind::Uint ? "u32" : (src_kind == CopySampleKind::Sint ? "i32" : "f32");
    std::string s;
    if (with_params) {
        s += "struct CopyParams { src_origin: vec4<u32>, dst_origin: vec4<u32>, extent: vec4<u32>, };\n";
    }
    if (depth_src) {
        s += "@group(0) @binding(0) var src_tex: texture_depth_2d;\n";
    } else if (is_3d) {
        s += std::string("@group(0) @binding(0) var src_tex: texture_3d<") + scalar + ">;\n";
    } else {
        s += std::string("@group(0) @binding(0) var src_tex: texture_2d<") + scalar + ">;\n";
    }
    s += std::string("@group(0) @binding(1) var dst_tex: ") +
         (is_3d ? "texture_storage_3d<" : "texture_storage_2d<") + storage_fmt_name + ", write>;\n";
    if (with_params) {
        s += "@group(0) @binding(2) var<uniform> params: CopyParams;\n";
    }
    s += is_3d ? "@compute @workgroup_size(4, 4, 4)\n" : "@compute @workgroup_size(8, 8, 1)\n";
    s += "fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {\n";
    if (with_params) {
        if (is_3d) {
            s += "  if (any(id >= params.extent.xyz)) { return; }\n";
            s += "  let sc = vec3<i32>(params.src_origin.xyz + id);\n";
            s += "  let dc = vec3<i32>(params.dst_origin.xyz + id);\n";
            s += "  textureStore(dst_tex, dc, textureLoad(src_tex, sc, 0));\n";
        } else {
            s += "  if (any(id.xy >= params.extent.xy)) { return; }\n";
            s += "  let sc = vec2<i32>(params.src_origin.xy + id.xy);\n";
            s += "  let dc = vec2<i32>(params.dst_origin.xy + id.xy);\n";
            if (depth_src) {
                s += "  let d = textureLoad(src_tex, sc, 0);\n";
                s += "  textureStore(dst_tex, dc, vec4<f32>(d, 0.0, 0.0, 1.0));\n";
            } else {
                s += "  textureStore(dst_tex, dc, textureLoad(src_tex, sc, 0));\n";
            }
        }
    } else {
        if (is_3d) {
            s += "  if (any(id >= textureDimensions(dst_tex))) { return; }\n";
            s += "  textureStore(dst_tex, vec3<i32>(id), textureLoad(src_tex, vec3<i32>(id), 0));\n";
        } else {
            s += "  if (any(id.xy >= textureDimensions(dst_tex))) { return; }\n";
            if (depth_src) {
                s += "  let d = textureLoad(src_tex, vec2<i32>(id.xy), 0);\n";
                s += "  textureStore(dst_tex, vec2<i32>(id.xy), vec4<f32>(d, 0.0, 0.0, 1.0));\n";
            } else {
                s += "  textureStore(dst_tex, vec2<i32>(id.xy), textureLoad(src_tex, vec2<i32>(id.xy), 0));\n";
            }
        }
    }
    s += "}\n";
    return s;
}

struct CopyParamsKey {
    uint32_t sx = 0, sy = 0, sz = 0;
    uint32_t dx = 0, dy = 0, dz = 0;
    uint32_t ex = 0, ey = 0, ez = 0;
    bool operator==(const CopyParamsKey& o) const noexcept {
        return sx == o.sx && sy == o.sy && sz == o.sz && dx == o.dx && dy == o.dy && dz == o.dz && ex == o.ex &&
               ey == o.ey && ez == o.ez;
    }
};

struct CopyParamsKeyHash {
    size_t operator()(const CopyParamsKey& k) const noexcept {
        size_t h = 14695981039346656037ull;
        const uint32_t vals[9] = {k.sx, k.sy, k.sz, k.dx, k.dy, k.dz, k.ex, k.ey, k.ez};
        for (uint32_t v : vals) {
            h ^= v;
            h *= 1099511628211ull;
        }
        return h;
    }
};

// Per-Device cache of ABI v2 texture-copy shader modules, layouts, pipelines, and uniform buffers.
// Refcounted by live CommandList instances (which hold com::Ref<Device>), so the entry is always
// erased before Device destruction — avoiding both leaks and pointer-reuse ABA hazards in tests.
struct DeviceTexCopyCache {
    uint64_t live_lists = 0;
    uint32_t blit_vs = 0;
    // Keyed by (src_kind << 4) | dst_kind.
    std::unordered_map<uint32_t, uint32_t> blit_fs;
    // Keyed by src_kind (1=Float, 2=Uint, 3=Sint, 4=Depth).
    std::unordered_map<uint8_t, uint32_t> blit_bgl;
    std::unordered_map<uint8_t, uint32_t> blit_pl;
    // Keyed by (dst_wgpu_code << 16) | (src_kind << 8) | dst_kind.
    std::unordered_map<uint32_t, uint32_t> blit_pipelines;

    struct ComputeCopyPipeline {
        uint32_t bgl = 0;
        uint32_t pipeline = 0;
    };
    // Keyed by (dst_wgpu_code << 16) | (src_kind << 8) | (is_3d ? 2 : 0) | (with_params ? 1 : 0).
    std::unordered_map<uint32_t, ComputeCopyPipeline> compute_pipelines;
    std::unordered_map<CopyParamsKey, uint32_t, CopyParamsKeyHash> params_buffers;
};

std::mutex g_tex_copy_cache_mutex;
std::unordered_map<const Device*, std::unique_ptr<DeviceTexCopyCache>> g_tex_copy_cache;

void retain_device_copy_cache(const Device* dev) {
    if (dev == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_tex_copy_cache_mutex);
    auto& entry = g_tex_copy_cache[dev];
    if (!entry) {
        entry = std::make_unique<DeviceTexCopyCache>();
    }
    ++entry->live_lists;
}

void release_device_copy_cache(const Device* dev) {
    if (dev == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_tex_copy_cache_mutex);
    auto it = g_tex_copy_cache.find(dev);
    if (it == g_tex_copy_cache.end() || !it->second) {
        return;
    }
    if (it->second->live_lists <= 1u) {
        g_tex_copy_cache.erase(it);
    } else {
        --it->second->live_lists;
    }
}

struct SubresourceCopySpec {
    uint32_t src_mip = 0;
    uint32_t src_layer = 0;
    uint32_t src_x = 0;
    uint32_t src_y = 0;
    uint32_t src_z = 0;
    uint32_t dst_mip = 0;
    uint32_t dst_layer = 0;
    uint32_t dst_x = 0;
    uint32_t dst_y = 0;
    uint32_t dst_z = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t depth = 1;
    bool full_subresource = false;
};

struct TextureCopyPlan {
    enum class Mode : uint8_t { None, RenderBlit, ComputeStorage, NativeCopy };
    Mode mode = Mode::None;
    CopySampleKind src_kind = CopySampleKind::Unsupported;
    CopySampleKind dst_kind = CopySampleKind::Unsupported;
    pso::WgpuTextureFormat src_view_fmt = pso::WgpuTextureFormat::Undefined;
    pso::WgpuTextureFormat dst_view_fmt = pso::WgpuTextureFormat::Undefined;
    const char* storage_fmt_name = nullptr;
    bool is_3d = false;
    bool dst_swapchain = false;
    // NativeCopy: block size of the (equal) formats, and whether a copy must cover whole subresources.
    uint32_t block_w = 1;
    uint32_t block_h = 1;
    bool whole_only = false;
    // Mode::None: why no route exists (the Work::CopySkip* counter).
    Work skip = Work::CopySkipEmitFailed;
};

// A plain copyTextureToTexture (kGpuOpcodeCopyTexture) for what the shader routes cannot write: the
// destination is neither a render target nor a UAV (or the source has no shader-resource use).
TextureCopyPlan plan_native_texture_copy(Device& dev, const Resource* dst, const Resource* src) {
    TextureCopyPlan plan{};
    if (!gpu_texture_copy()) {
        plan.skip = Work::CopySkipGateOff;
        return plan;
    }
    if (dst == nullptr || src == nullptr || dst->is_buffer() || src->is_buffer() || !dst->has_gpu_texture() ||
        !src->has_gpu_texture()) {
        plan.skip = Work::CopySkipNoGpuTexture;
        return plan;
    }
    if (src->desc().SampleCount > 1u || dst->desc().SampleCount > 1u) {
        plan.skip = Work::CopySkipMsaa;
        return plan;
    }
    bridge::TextureInfo dst_info{}, src_info{};
    if (!dev.state_bridge().texture_info(dst->id(), &dst_info) ||
        !dev.state_bridge().texture_info(src->id(), &src_info)) {
        plan.skip = Work::CopySkipNoGpuTexture;
        return plan;
    }
    if (dst_info.swapchain || src_info.swapchain) {
        plan.skip = Work::CopySkipSwapchain;
        return plan;
    }
    const bool both_2d = src_info.dimension == GpuTextureDimension::Dim2D &&
                         dst_info.dimension == GpuTextureDimension::Dim2D;
    const bool both_3d = src_info.dimension == GpuTextureDimension::Dim3D &&
                         dst_info.dimension == GpuTextureDimension::Dim3D;
    if (!both_2d && !both_3d) {
        plan.skip = Work::CopySkipDimension;
        return plan;
    }
    const uint32_t features = dev.state_bridge().options().features;
    const pso::WgpuTextureFormat src_wgpu = resource_wgpu_format(src, features);
    const pso::WgpuTextureFormat dst_wgpu = resource_wgpu_format(dst, features);
    if (src_wgpu == pso::WgpuTextureFormat::Undefined || dst_wgpu == pso::WgpuTextureFormat::Undefined ||
        linear_copy_view_format(src_wgpu) != linear_copy_view_format(dst_wgpu)) {
        plan.skip = Work::CopySkipFormat;
        return plan;  // not copy-compatible (equal up to sRGB-ness)
    }
    const DxgiFormatLayout l = dxgi_format_layout(dst->desc().Format);
    plan.mode = TextureCopyPlan::Mode::NativeCopy;
    plan.is_3d = both_3d;
    plan.block_w = l.known ? std::max<uint32_t>(1u, l.block_width) : 1u;
    plan.block_h = l.known ? std::max<uint32_t>(1u, l.block_height) : 1u;
    plan.whole_only = is_depth_dxgi_format(dst->desc().Format) || is_depth_dxgi_format(src->desc().Format);
    return plan;
}

TextureCopyPlan plan_gpu_texture_copy_shader(Device& dev, const Resource* dst, const Resource* src);

TextureCopyPlan plan_gpu_texture_copy(Device& dev, const Resource* dst, const Resource* src) {
    TextureCopyPlan plan = plan_gpu_texture_copy_shader(dev, dst, src);
    if (plan.mode != TextureCopyPlan::Mode::None) {
        return plan;
    }
    return plan_native_texture_copy(dev, dst, src);
}

TextureCopyPlan plan_gpu_texture_copy_shader(Device& dev, const Resource* dst, const Resource* src) {
    TextureCopyPlan plan{};
    if (dst == nullptr || src == nullptr || dst->is_buffer() || src->is_buffer() || !dst->has_gpu_texture() ||
        !src->has_gpu_texture()) {
        return plan;
    }
    if (src->desc().SampleCount > 1u || dst->desc().SampleCount > 1u) {
        return plan;
    }
    if ((src->desc().Flags & ResourceFlags::DenyShaderResource) != 0u) {
        return plan;
    }
    bridge::TextureInfo dst_info{}, src_info{};
    if (!dev.state_bridge().texture_info(dst->id(), &dst_info) ||
        !dev.state_bridge().texture_info(src->id(), &src_info)) {
        return plan;
    }
    if (src_info.swapchain) {
        return plan;  // canvas context lacks TEXTURE_BINDING
    }
    const uint32_t features = dev.state_bridge().options().features;
    const pso::WgpuTextureFormat src_wgpu = resource_wgpu_format(src, features);
    const pso::WgpuTextureFormat dst_wgpu = resource_wgpu_format(dst, features);
    const CopySampleKind src_kind = copy_sample_kind(src_wgpu);
    const CopySampleKind dst_kind = copy_sample_kind(dst_wgpu);
    if (!copy_kinds_compatible(src_kind, dst_kind, /*allow_depth_cross=*/true)) {
        return plan;
    }
    // Use linear view formats for sRGB textures so the copy is bit-exact (no gamma decode/encode).
    const pso::WgpuTextureFormat src_view_fmt = linear_copy_view_format(src_wgpu);
    const pso::WgpuTextureFormat dst_view_fmt = dst_info.swapchain ? dst_wgpu : linear_copy_view_format(dst_wgpu);

    const bool both_2d = src_info.dimension == GpuTextureDimension::Dim2D &&
                         dst_info.dimension == GpuTextureDimension::Dim2D;
    const bool both_3d = src_info.dimension == GpuTextureDimension::Dim3D &&
                         dst_info.dimension == GpuTextureDimension::Dim3D;

    // Path 1: Render-pass blit for 2D / 2D-Array / Cube RT or Depth-Stencil destinations.
    if (both_2d) {
        const bool dst_is_depth = dst_kind == CopySampleKind::Depth;
        const bool can_render =
            dst_is_depth
                ? ((dst->desc().Flags & ResourceFlags::AllowDepthStencil) != 0u && pso::has_depth_aspect(dst_view_fmt))
                : (((dst->desc().Flags & ResourceFlags::AllowRenderTarget) != 0u || dst_info.swapchain) &&
                   pso::is_color_renderable(dst_view_fmt, features));
        if (can_render) {
            plan.mode = TextureCopyPlan::Mode::RenderBlit;
            plan.src_kind = src_kind;
            plan.dst_kind = dst_kind;
            plan.src_view_fmt = src_view_fmt;
            plan.dst_view_fmt = dst_view_fmt;
            plan.is_3d = false;
            plan.dst_swapchain = dst_info.swapchain;
            return plan;
        }
    }

    // Path 2: Compute-pass copy for 2D or 3D StorageBinding (UAV) destinations.
    if ((both_2d || both_3d) && dst_info.storage && !dst_info.swapchain && dst_kind != CopySampleKind::Depth &&
        !(both_3d && src_kind == CopySampleKind::Depth)) {
        const char* storage_name = wgsl_storage_format_name(dst_view_fmt, features);
        if (storage_name != nullptr) {
            plan.mode = TextureCopyPlan::Mode::ComputeStorage;
            plan.src_kind = src_kind;
            plan.dst_kind = dst_kind;
            plan.src_view_fmt = src_view_fmt;
            plan.dst_view_fmt = dst_view_fmt;
            plan.storage_fmt_name = storage_name;
            plan.is_3d = both_3d;
            plan.dst_swapchain = false;
            return plan;
        }
    }

    return plan;
}

bool emit_gpu_texture_subresource_copy(Device& dev, const TextureCopyPlan& plan, uint32_t dst_id, uint32_t src_id,
                                       const SubresourceCopySpec& spec, CommandEncoder& enc) {
    if (plan.mode == TextureCopyPlan::Mode::None || spec.width == 0 || spec.height == 0 || spec.depth == 0) {
        return false;
    }
    uint32_t host_src_mip = spec.src_mip;
    uint32_t host_dst_mip = spec.dst_mip;
    if (!dev.state_bridge().translate_host_mip(src_id, &host_src_mip) ||
        !dev.state_bridge().translate_host_mip(dst_id, &host_dst_mip)) {
        return true;  // mip trimmed by texture_budget: cleanly skipped
    }

    if (plan.mode == TextureCopyPlan::Mode::NativeCopy) {
        if (plan.whole_only && !spec.full_subresource) {
            return false;  // depth/stencil are copied whole (copyTextureToTexture)
        }
        GpuCopyTexturePayload c{};
        c.dst_id = dst_id;
        c.src_id = src_id;
        c.dst_mip = host_dst_mip;
        c.src_mip = host_src_mip;
        c.dst_x = spec.dst_x;
        c.dst_y = spec.dst_y;
        c.dst_z = spec.dst_z;
        c.src_x = spec.src_x;
        c.src_y = spec.src_y;
        c.src_z = spec.src_z;
        c.width = align_up_u32(spec.width, plan.block_w);
        c.height = align_up_u32(spec.height, plan.block_h);
        c.depth_or_layers = spec.depth;
        enc.emit_packet(kGpuOpcodeCopyTexture, &c, sizeof(c));
        return true;
    }

    if (plan.mode == TextureCopyPlan::Mode::RenderBlit) {
        const int32_t dx = static_cast<int32_t>(spec.src_x) - static_cast<int32_t>(spec.dst_x);
        const int32_t dy = static_cast<int32_t>(spec.src_y) - static_cast<int32_t>(spec.dst_y);
        if (dx < -32768 || dx > 32767 || dy < -32768 || dy > 32767) {
            return false;
        }
        const uint8_t src_kind_u8 = static_cast<uint8_t>(plan.src_kind);
        const uint8_t dst_kind_u8 = static_cast<uint8_t>(plan.dst_kind);
        const uint8_t dst_fmt_code = static_cast<uint8_t>(plan.dst_view_fmt);
        const uint32_t fs_key = (uint32_t{src_kind_u8} << 4u) | dst_kind_u8;
        const uint32_t pipe_key = (uint32_t{dst_fmt_code} << 16u) | (uint32_t{src_kind_u8} << 8u) | dst_kind_u8;

        uint32_t group_layout_id = 0;
        uint32_t pipeline_id = 0;
        CommandEncoder setup;
        {
            std::lock_guard<std::mutex> lock(g_tex_copy_cache_mutex);
            auto& entry_ptr = g_tex_copy_cache[&dev];
            if (!entry_ptr) {
                entry_ptr = std::make_unique<DeviceTexCopyCache>();
            }
            DeviceTexCopyCache& cache = *entry_ptr;
            if (cache.blit_vs == 0) {
                cache.blit_vs = dev.allocate_id();
                setup.create_shader_module(cache.blit_vs, GpuShaderStage::Vertex, kTexCopyBlitVs);
            }
            uint32_t& fs_id = cache.blit_fs[fs_key];
            if (fs_id == 0) {
                const char* fs_wgsl = kTexCopyFloatFs;
                if (plan.src_kind == CopySampleKind::Depth && plan.dst_kind == CopySampleKind::Depth) {
                    fs_wgsl = kTexCopyDepthFs;
                } else if (plan.src_kind == CopySampleKind::Depth && plan.dst_kind == CopySampleKind::Float) {
                    fs_wgsl = kTexCopyDepthToFloatFs;
                } else if (plan.src_kind == CopySampleKind::Float && plan.dst_kind == CopySampleKind::Depth) {
                    fs_wgsl = kTexCopyFloatToDepthFs;
                } else if (plan.src_kind == CopySampleKind::Uint) {
                    fs_wgsl = kTexCopyUintFs;
                } else if (plan.src_kind == CopySampleKind::Sint) {
                    fs_wgsl = kTexCopySintFs;
                }
                fs_id = dev.allocate_id();
                setup.create_shader_module(fs_id, GpuShaderStage::Fragment, fs_wgsl);
            }
            uint32_t& bgl_id = cache.blit_bgl[src_kind_u8];
            uint32_t& pl_id = cache.blit_pl[src_kind_u8];
            if (bgl_id == 0) {
                GpuBindGroupLayoutEntry e{};
                e.binding = 0;
                e.visibility = GpuShaderStageBits::Fragment;
                e.type = static_cast<uint8_t>(GpuBindingLayoutType::Texture);
                e.binding_type = static_cast<uint8_t>(
                    plan.src_kind == CopySampleKind::Depth
                        ? GpuTextureSampleType::Depth
                        : (plan.src_kind == CopySampleKind::Uint
                               ? GpuTextureSampleType::Uint
                               : (plan.src_kind == CopySampleKind::Sint ? GpuTextureSampleType::Sint
                                                                        : GpuTextureSampleType::UnfilterableFloat)));
                e.view_dimension = static_cast<uint8_t>(GpuTextureViewDimension::Dim2D);
                bgl_id = dev.allocate_id();
                GpuCreateBindGroupLayoutPayload bgl{};
                bgl.id = bgl_id;
                bgl.entry_count = 1;
                setup.emit_packet(GpuOpcode::CreateBindGroupLayout, &bgl, sizeof(bgl), &e, sizeof(e));
                pl_id = dev.allocate_id();
                GpuCreatePipelineLayoutPayload pl{};
                pl.id = pl_id;
                pl.bind_group_layout_count = 1;
                setup.emit_packet(GpuOpcode::CreatePipelineLayout, &pl, sizeof(pl), &bgl_id, sizeof(uint32_t));
            }
            group_layout_id = bgl_id;
            uint32_t& pipe = cache.blit_pipelines[pipe_key];
            if (pipe == 0) {
                pipe = dev.allocate_id();
                GpuCreateRenderPipeline2Payload rp{};
                rp.id = pipe;
                rp.vs_module_id = cache.blit_vs;
                rp.fs_module_id = fs_id;
                rp.pipeline_layout_id = pl_id;
                rp.key.vs_shader = 1;
                rp.key.ps_shader = 1;
                if (plan.dst_kind == CopySampleKind::Depth) {
                    rp.key.color_target_count = 0;
                    rp.key.depth_stencil.format = dst_fmt_code;
                    rp.key.depth_stencil.depth_write_enabled = 1;
                    rp.key.depth_stencil.depth_compare = static_cast<uint8_t>(pso::WgpuCompareFunction::Always);
                    rp.key.depth_stencil.stencil_front.compare = static_cast<uint8_t>(pso::WgpuCompareFunction::Always);
                    rp.key.depth_stencil.stencil_back.compare = static_cast<uint8_t>(pso::WgpuCompareFunction::Always);
                } else {
                    rp.key.color_target_count = 1;
                    rp.key.color_targets[0].format = dst_fmt_code;
                    rp.key.color_targets[0].write_mask = 0xFu;
                }
                rp.key.primitive.topology = static_cast<uint8_t>(GpuPrimitiveTopology::TriangleList);
                rp.key.multisample.count = 1;
                rp.key.multisample.mask = 0xFFFFFFFFu;
                setup.emit_packet(GpuOpcode::CreateRenderPipeline2, &rp, sizeof(rp));
            }
            pipeline_id = pipe;
        }
        if (!setup.empty()) {
            dev.submit(setup);
        }

        const uint32_t group_id = dev.allocate_id();
        GpuBindGroupEntry2 entry{};
        entry.binding = 0;
        entry.resource_type = static_cast<uint32_t>(GpuBindResourceType::TextureView);
        entry.resource_id = src_id;
        entry.view.format =
            plan.src_kind == CopySampleKind::Depth ? 0u : static_cast<uint8_t>(plan.src_view_fmt);
        entry.view.dimension = static_cast<uint8_t>(GpuTextureViewDimension::Dim2D);
        entry.view.aspect = static_cast<uint8_t>(
            plan.src_kind == CopySampleKind::Depth ? GpuTextureAspect::DepthOnly : GpuTextureAspect::All);
        entry.view.base_mip_level = static_cast<uint16_t>(host_src_mip);
        entry.view.mip_level_count = 1;
        entry.view.base_array_layer = spec.src_layer;
        entry.view.array_layer_count = 1;
        GpuCreateBindGroup2Payload bg{};
        bg.id = group_id;
        bg.layout_id = group_layout_id;
        bg.entry_count = 1;
        enc.emit_packet(GpuOpcode::CreateBindGroup2, &bg, sizeof(bg), &entry, sizeof(entry));

        GpuBeginRenderPassPayload pass{};
        const uint8_t load_op =
            static_cast<uint8_t>(spec.full_subresource ? GpuLoadOp::Clear : GpuLoadOp::Load);
        if (plan.dst_kind == CopySampleKind::Depth) {
            pass.color_count = 0;
            GpuRenderPassDepthStencilAttachment& d = pass.depth_stencil;
            d.texture_id = dst_id;
            d.depth_load_op = load_op;
            d.depth_store_op = static_cast<uint8_t>(GpuStoreOp::Store);
            d.depth_clear_value = 0.0f;
            if (pso::has_stencil_aspect(plan.dst_view_fmt)) {
                d.stencil_read_only = 1;
            }
            d.view.format = dst_fmt_code;
            d.view.dimension = static_cast<uint8_t>(GpuTextureViewDimension::Dim2D);
            d.view.aspect = static_cast<uint8_t>(GpuTextureAspect::All);
            d.view.base_mip_level = static_cast<uint16_t>(host_dst_mip);
            d.view.mip_level_count = 1;
            d.view.base_array_layer = spec.dst_layer;
            d.view.array_layer_count = 1;
        } else {
            pass.color_count = 1;
            GpuRenderPassColorAttachment& c = pass.color[0];
            c.texture_id = plan.dst_swapchain ? kGpuSwapchainTextureId : dst_id;
            c.load_op = load_op;
            c.store_op = static_cast<uint8_t>(GpuStoreOp::Store);
            c.view.format = dst_fmt_code;
            c.view.dimension = static_cast<uint8_t>(GpuTextureViewDimension::Dim2D);
            c.view.aspect = static_cast<uint8_t>(GpuTextureAspect::All);
            c.view.base_mip_level = static_cast<uint16_t>(host_dst_mip);
            c.view.mip_level_count = 1;
            c.view.base_array_layer = spec.dst_layer;
            c.view.array_layer_count = 1;
        }
        enc.emit_packet(GpuOpcode::BeginRenderPass, &pass, sizeof(pass));
        enc.set_pipeline(pipeline_id);
        const GpuSetBindGroup2Payload set{0u, group_id, 0u, 0u};
        enc.emit_packet(GpuOpcode::SetBindGroup2, &set, sizeof(set));
        if (!spec.full_subresource) {
            GpuSetViewportPayload vp{};
            vp.x = static_cast<float>(spec.dst_x);
            vp.y = static_cast<float>(spec.dst_y);
            vp.width = static_cast<float>(spec.width);
            vp.height = static_cast<float>(spec.height);
            vp.min_depth = 0.0f;
            vp.max_depth = 1.0f;
            enc.emit_packet(GpuOpcode::SetViewport, &vp, sizeof(vp));
            GpuSetScissorRectPayload sc{};
            sc.x = spec.dst_x;
            sc.y = spec.dst_y;
            sc.width = spec.width;
            sc.height = spec.height;
            enc.emit_packet(GpuOpcode::SetScissorRect, &sc, sizeof(sc));
        }
        enc.draw(3, 1, 0, pack_signed_delta_i16(dx, dy));
        enc.emit_packet(GpuOpcode::EndRenderPass, nullptr, 0);
        // The group is single-use (one per copy): the executor resolved it at SetBindGroup2, so
        // destroying it now frees the Dawn bind group and prunes its journaled create packet.
        enc.destroy_resource(group_id);
        return true;
    }

    // Path 2: Compute-pass copy for 2D / 3D storage-capable destinations.
    const bool with_params = !spec.full_subresource || spec.src_x != 0 || spec.src_y != 0 || spec.src_z != 0 ||
                             spec.dst_x != 0 || spec.dst_y != 0 || spec.dst_z != 0;
    const uint8_t dst_fmt_code = static_cast<uint8_t>(plan.dst_view_fmt);
    const uint8_t src_kind_u8 = static_cast<uint8_t>(plan.src_kind);
    const uint8_t dim_code = static_cast<uint8_t>(plan.is_3d ? GpuTextureViewDimension::Dim3D
                                                             : GpuTextureViewDimension::Dim2D);
    const uint32_t cs_key = (uint32_t{dst_fmt_code} << 16u) | (uint32_t{src_kind_u8} << 8u) |
                            (plan.is_3d ? 2u : 0u) | (with_params ? 1u : 0u);

    uint32_t group_layout_id = 0;
    uint32_t pipeline_id = 0;
    uint32_t params_buf_id = 0;
    CommandEncoder setup;
    {
        std::lock_guard<std::mutex> lock(g_tex_copy_cache_mutex);
        auto& entry_ptr = g_tex_copy_cache[&dev];
        if (!entry_ptr) {
            entry_ptr = std::make_unique<DeviceTexCopyCache>();
        }
        DeviceTexCopyCache& cache = *entry_ptr;
        auto& cp = cache.compute_pipelines[cs_key];
        if (cp.pipeline == 0) {
            const uint32_t mod_id = dev.allocate_id();
            setup.create_shader_module(
                mod_id, GpuShaderStage::Compute,
                build_compute_copy_wgsl(plan.storage_fmt_name, plan.src_kind, plan.is_3d, with_params));
            GpuBindGroupLayoutEntry entries[3] = {};
            entries[0].binding = 0;
            entries[0].visibility = GpuShaderStageBits::Compute;
            entries[0].type = static_cast<uint8_t>(GpuBindingLayoutType::Texture);
            entries[0].binding_type = static_cast<uint8_t>(
                plan.src_kind == CopySampleKind::Depth
                    ? GpuTextureSampleType::Depth
                    : (plan.src_kind == CopySampleKind::Uint
                           ? GpuTextureSampleType::Uint
                           : (plan.src_kind == CopySampleKind::Sint ? GpuTextureSampleType::Sint
                                                                    : GpuTextureSampleType::UnfilterableFloat)));
            entries[0].view_dimension = dim_code;
            entries[1].binding = 1;
            entries[1].visibility = GpuShaderStageBits::Compute;
            entries[1].type = static_cast<uint8_t>(GpuBindingLayoutType::StorageTexture);
            entries[1].binding_type = static_cast<uint8_t>(GpuStorageTextureAccess::WriteOnly);
            entries[1].view_dimension = dim_code;
            entries[1].storage_format = dst_fmt_code;
            uint32_t entry_count = 2;
            if (with_params) {
                entries[2].binding = 2;
                entries[2].visibility = GpuShaderStageBits::Compute;
                entries[2].type = static_cast<uint8_t>(GpuBindingLayoutType::Buffer);
                entries[2].binding_type = static_cast<uint8_t>(GpuBufferBindingType::Uniform);
                entries[2].min_binding_size = 48;
                entry_count = 3;
            }
            cp.bgl = dev.allocate_id();
            GpuCreateBindGroupLayoutPayload bgl{};
            bgl.id = cp.bgl;
            bgl.entry_count = entry_count;
            setup.emit_packet(GpuOpcode::CreateBindGroupLayout, &bgl, sizeof(bgl), entries,
                              entry_count * sizeof(GpuBindGroupLayoutEntry));
            const uint32_t pl_id = dev.allocate_id();
            GpuCreatePipelineLayoutPayload pl{};
            pl.id = pl_id;
            pl.bind_group_layout_count = 1;
            setup.emit_packet(GpuOpcode::CreatePipelineLayout, &pl, sizeof(pl), &cp.bgl, sizeof(uint32_t));
            cp.pipeline = dev.allocate_id();
            GpuCreateComputePipeline2Payload pipe{};
            pipe.id = cp.pipeline;
            pipe.cs_module_id = mod_id;
            pipe.pipeline_layout_id = pl_id;
            setup.emit_packet(GpuOpcode::CreateComputePipeline2, &pipe, sizeof(pipe));
        }
        group_layout_id = cp.bgl;
        pipeline_id = cp.pipeline;
        if (with_params) {
            const CopyParamsKey pk{spec.src_x, spec.src_y, spec.src_z, spec.dst_x, spec.dst_y,
                                   spec.dst_z, spec.width, spec.height, spec.depth};
            auto it = cache.params_buffers.find(pk);
            if (it != cache.params_buffers.end()) {
                params_buf_id = it->second;
            } else {
                params_buf_id = dev.allocate_id();
                const uint32_t words[12] = {spec.src_x, spec.src_y, spec.src_z, 0u,
                                            spec.dst_x, spec.dst_y, spec.dst_z, 0u,
                                            spec.width, spec.height, spec.depth, 0u};
                setup.create_buffer(params_buf_id, GpuBufferUsage::Uniform | GpuBufferUsage::CopyDst, sizeof(words));
                setup.write_buffer(params_buf_id, 0, words, sizeof(words));
                if (cache.params_buffers.size() < 256u) {
                    cache.params_buffers.emplace(pk, params_buf_id);
                }
            }
        }
    }
    if (!setup.empty()) {
        dev.submit(setup);
    }

    const uint32_t group_id = dev.allocate_id();
    GpuBindGroupEntry2 e[3] = {};
    e[0].binding = 0;
    e[0].resource_type = static_cast<uint32_t>(GpuBindResourceType::TextureView);
    e[0].resource_id = src_id;
    e[0].view.format = plan.src_kind == CopySampleKind::Depth ? 0u : static_cast<uint8_t>(plan.src_view_fmt);
    e[0].view.dimension = dim_code;
    e[0].view.aspect = static_cast<uint8_t>(
        plan.src_kind == CopySampleKind::Depth ? GpuTextureAspect::DepthOnly : GpuTextureAspect::All);
    e[0].view.base_mip_level = static_cast<uint16_t>(host_src_mip);
    e[0].view.mip_level_count = 1;
    e[0].view.base_array_layer = plan.is_3d ? 0u : spec.src_layer;
    e[0].view.array_layer_count = 1;

    e[1].binding = 1;
    e[1].resource_type = static_cast<uint32_t>(GpuBindResourceType::TextureView);
    e[1].resource_id = dst_id;
    e[1].view.format = dst_fmt_code;
    e[1].view.dimension = dim_code;
    e[1].view.aspect = static_cast<uint8_t>(GpuTextureAspect::All);
    e[1].view.base_mip_level = static_cast<uint16_t>(host_dst_mip);
    e[1].view.mip_level_count = 1;
    e[1].view.base_array_layer = plan.is_3d ? 0u : spec.dst_layer;
    e[1].view.array_layer_count = 1;

    uint32_t entry_count = 2;
    if (with_params) {
        e[2].binding = 2;
        e[2].resource_type = static_cast<uint32_t>(GpuBindResourceType::Buffer);
        e[2].resource_id = params_buf_id;
        e[2].offset = 0;
        e[2].size = 48;
        entry_count = 3;
    }
    GpuCreateBindGroup2Payload bg{};
    bg.id = group_id;
    bg.layout_id = group_layout_id;
    bg.entry_count = entry_count;
    enc.emit_packet(GpuOpcode::CreateBindGroup2, &bg, sizeof(bg), e, entry_count * sizeof(GpuBindGroupEntry2));

    enc.set_compute_pipeline(pipeline_id);
    const GpuSetBindGroup2Payload set{0u, group_id, 0u, 0u};
    enc.emit_packet(GpuOpcode::SetBindGroup2, &set, sizeof(set));
    const uint32_t wg_xy = plan.is_3d ? 4u : 8u;
    const uint32_t gx = (spec.width + wg_xy - 1u) / wg_xy;
    const uint32_t gy = (spec.height + wg_xy - 1u) / wg_xy;
    const uint32_t gz = plan.is_3d ? (spec.depth + 3u) / 4u : 1u;
    enc.dispatch_compute(gx, gy, gz);
    enc.destroy_resource(group_id);  // single-use group (see the render path)
    return true;
}

// Synchronizes any depth-as-SRV descriptor table bindings against `aux.depth_twin` so pending twin
// writes are flushed and stale SRV twins are refreshed accurately across mip levels and array layers.
void sync_depth_srvs_for_tables(Device& dev, CommandListAux& aux, const d3d::RootSignature* rs,
                                const std::function<uint64_t(size_t)>& table_gpu_handle) {
    if (rs == nullptr) {
        return;
    }
    const d3d::RootSignatureDesc& rd = rs->desc();
    for (size_t p = 0; p < rd.parameters.size(); ++p) {
        const d3d::RootParameterDesc& param = rd.parameters[p];
        if (param.parameter_type != d3d::RootParameterType::DescriptorTable) {
            continue;
        }
        const uint64_t gpu_handle = table_gpu_handle(p);
        if (gpu_handle == 0) {
            continue;
        }
        d3d::ResolvedDescriptor base{};
        if (dev.heaps().resolve_gpu(d3d::GpuDescriptorHandle{gpu_handle}, &base) != d3d::Status::Ok ||
            base.heap == nullptr) {
            continue;
        }
        uint32_t table_offset = 0;
        for (const d3d::DescriptorRangeDesc& range : param.ranges) {
            if (range.offset_in_descriptors_from_table_start != 0xFFFFFFFFu) {
                table_offset = range.offset_in_descriptors_from_table_start;
            }
            const uint32_t count =
                range.num_descriptors == d3d::kUnboundedDescriptorCount ? 16u : std::min<uint32_t>(range.num_descriptors, 64u);
            if (range.range_type == d3d::DescriptorRangeType::Srv) {
                for (uint32_t i = 0; i < count; ++i) {
                    const uint32_t slot = base.index + table_offset + i;
                    if (slot >= base.heap->capacity()) {
                        break;
                    }
                    const d3d::ViewDescriptor* v = base.heap->view(slot);
                    if (v == nullptr || v->kind != d3d::DescriptorKind::Srv || v->resource_id == 0) {
                        continue;
                    }
                    bridge::TextureInfo ti{};
                    if (!dev.state_bridge().texture_info(v->resource_id, &ti) ||
                        !is_depth_dxgi_format(static_cast<uint32_t>(ti.format))) {
                        continue;
                    }
                    const uint32_t total_mips = std::max<uint32_t>(1u, ti.mip_levels);
                    const uint32_t total_layers =
                        ti.dimension == GpuTextureDimension::Dim3D ? 1u : std::max<uint32_t>(1u, ti.depth_or_array_size);
                    const uint32_t base_mip = std::min<uint32_t>(v->first_mip, total_mips - 1u);
                    const uint32_t mips =
                        (v->mip_levels == 0 || v->mip_levels == 0xFFFFu)
                            ? (total_mips - base_mip)
                            : std::min<uint32_t>(v->mip_levels, total_mips - base_mip);
                    const uint32_t base_layer = std::min(v->first_array_slice, total_layers - 1u);
                    const uint32_t layers =
                        (v->array_size == 0 || v->array_size == 0xFFFFFFFFu)
                            ? (total_layers - base_layer)
                            : std::min(v->array_size, total_layers - base_layer);
                    const bridge::DepthTwinTracker::SrvSyncPlan plan =
                        aux.depth_twin.plan_srv_read(v->resource_id, base_mip, mips, base_layer, layers);
                    if (plan.any()) {
                        aux.depth_srv_sync_flushes += plan.flush_twin_to_depth.size();
                        aux.depth_srv_sync_refreshes += plan.refresh_depth_to_srv.size();
                        aux.depth_twin.apply_srv_sync(v->resource_id, plan);
                    }
                }
            }
            table_offset += count;
        }
    }
}

}  // namespace

CommandListDynamicStateSnapshot inspect_command_list_state(const CommandList* list) {
    CommandListDynamicStateSnapshot snap{};
    if (list == nullptr) {
        return snap;
    }
    std::lock_guard<std::mutex> lock(g_aux_mutex);
    const auto it = g_aux_map.find(list);
    if (it == g_aux_map.end() || !it->second) {
        return snap;
    }
    const CommandListAux& aux = *it->second;
    snap.viewports = aux.clamped_viewports;
    snap.scissors = aux.clamped_scissors;
    std::memcpy(snap.blend_factor, aux.blend_factor, sizeof(snap.blend_factor));
    snap.stencil_ref = aux.stencil_ref;
    snap.bound_dsv_texture = aux.bound_dsv.resource_id;
    snap.bound_rt_width = aux.rt_width;
    snap.bound_rt_height = aux.rt_height;
    snap.depth_srv_sync_flushes = aux.depth_srv_sync_flushes;
    snap.depth_srv_sync_refreshes = aux.depth_srv_sync_refreshes;
    snap.resolves_emitted = aux.resolves_emitted;
    snap.resolves_fallback = aux.resolves_fallback;
    return snap;
}

const bridge::DepthTwinTracker* inspect_command_list_depth_twin(const CommandList* list) {
    if (list == nullptr) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(g_aux_mutex);
    const auto it = g_aux_map.find(list);
    return (it != g_aux_map.end() && it->second) ? &it->second->depth_twin : nullptr;
}

// ---------------------------------------------------------------------------------------------
// Device: command lists
// ---------------------------------------------------------------------------------------------

HResult Device::create_command_list(uint32_t type, InterfaceArg<IPipelineStateHost> initial_state, const Guid& riid,
                                    OutPtr out) {
    if (type > 3) {
        note_bad_argument("CreateCommandList: unsupported type " + std::to_string(type));
        return kEInvalidArg;
    }
    if (initial_state.invalid()) {
        note_bad_argument("CreateCommandList: invalid initial pipeline state " + hex(initial_state.va()));
        return kEInvalidArg;
    }
    auto list = com::create_object<CommandList>(com::implements(kID3D12GraphicsCommandList), device_ref(this), type);
    if (list) {
        list->initialize_guest_payload();
        list->reset(initial_state ? static_cast<PipelineState*>(initial_state.get()) : nullptr);
    }
    return hand_out(std::move(list), riid, out);
}

HResult Device::CreateCommandList(uint32_t, uint32_t type, InterfaceArg<ICommandAllocatorHost> allocator,
                                  InterfaceArg<IPipelineStateHost> initial_state, const Guid& riid, OutPtr out) {
    if (!allocator) {
        note_bad_argument("CreateCommandList: invalid allocator " + hex(allocator.va()));
        return kEInvalidArg;
    }
    return create_command_list(type, std::move(initial_state), riid, out);
}

HResult Device::CreateCommandListX(const GuestCommandListDescX* desc, InterfaceArg<ICommandAllocatorHost> allocator,
                                   InterfaceArg<IPipelineStateHost> initial_state, const Guid& riid, OutPtr out) {
    if (desc == nullptr || !allocator) {
        note_bad_argument("CreateCommandList (Xbox slot 73): unreadable descriptor or invalid allocator");
        return kEInvalidArg;
    }
    return create_command_list(desc->Type, std::move(initial_state), riid, out);
}

// ---------------------------------------------------------------------------------------------
// Queue: ExecuteCommandLists
// ---------------------------------------------------------------------------------------------

void CommandQueue::ExecuteCommandListsX(com::CallContext& ctx) {
    const os::ThreadSpan frame_span(os::SpanKind::D3dExecute);  // frame_breakdown
    // (NumCommandLists, ppCommandLists, frame token, flags): the token (device slot 80's output, 0
    // outside a frame) is recorded; the flags are not interpreted.
    const uint32_t count = static_cast<uint32_t>(ctx.int_arg(1));
    const uint64_t lists = ctx.int_arg(2);
    if (ctx.int_arg(3) != 0) {
        const uint64_t tok = ctx.int_arg(3);
        uint64_t prev = device_->last_execute_token.load(std::memory_order_relaxed);
        while (tok > prev && !device_->last_execute_token.compare_exchange_weak(
                                 prev, tok, std::memory_order_acq_rel, std::memory_order_relaxed)) {
        }
        device_->count(xbox::Counter::ExecutesWithToken);
    }
    std::vector<uint64_t> pointers;
    if (count > kMaxListsPerExecute || !read_n(lists, count, &pointers)) {
        device_->note_bad_argument("ExecuteCommandLists: unreadable list array " + hex(lists));
        return;
    }
    execute(pointers);
}

void CommandQueue::execute(const std::vector<uint64_t>& lists) {
    std::vector<std::shared_ptr<com::ComObject>> pins;
    std::vector<CommandList*> resolved;
    for (uint64_t p : lists) {
        com::detail::Resolved r;
        if (!com::detail::lookup_interface(p, &r) || r.iface != &kID3D12GraphicsCommandList.info) {
            device_->note_bad_argument("ExecuteCommandLists: not a command list " + hex(p));
            continue;
        }
        auto* list = static_cast<CommandList*>(static_cast<IGraphicsCommandListHost*>(r.host));
        if (list->is_open()) {
            device_->note_bad_argument("ExecuteCommandLists: command list " + hex(p) + " is not closed");
            continue;
        }
        pins.push_back(std::move(r.object));
        resolved.push_back(list);
    }
    // One submission: changed CPU-visible buffers, then each list's CPU-sourced copies and packets.
    std::vector<CommandEncoder> parts(1 + resolved.size());
    device_->begin_submission();
    for (CommandList* list : resolved) {
        list->issue_readbacks();  // G66: numbered (and counted) before any signal can follow them
    }
    {
        const os::ThreadSpan part_span(os::SpanKind::D3dFlush);
        device_->flush_cpu_buffers(parts[0]);
    }
    std::vector<d3d::StateDiagnostic> state_diags;
    {
        const os::ThreadSpan part_span(os::SpanKind::D3dAppend);
        for (size_t i = 0; i < resolved.size(); ++i) {
            resolved[i]->append_submission(parts[i + 1], &state_diags);
        }
    }
    device_->end_submission();
    std::vector<std::pair<const uint8_t*, size_t>> spans;
    for (const CommandEncoder& e : parts) {
        spans.emplace_back(e.data(), e.size());
    }
    for (CommandList* list : resolved) {
        spans.emplace_back(list->encoder().data(), list->encoder().size());
    }
    // parts[i + 1] (deferred copies of list i) must precede list i's packets: interleave.
    std::vector<std::pair<const uint8_t*, size_t>> ordered;
    ordered.push_back(spans[0]);
    for (size_t i = 0; i < resolved.size(); ++i) {
        ordered.push_back(spans[1 + i]);
        ordered.push_back(spans[1 + resolved.size() + i]);
    }
    {
        const os::ThreadSpan part_span(os::SpanKind::D3dSubmit);
        device_->submit_parts(ordered);
    }
    for (CommandList* list : resolved) {
        list->apply_immediate_writes();
    }
    for (const d3d::StateDiagnostic& d : state_diags) {
        device_->note_bad_argument(std::string("resource state at ExecuteCommandLists: ") +
                                   d3d::state_diagnostic_name(d.code) + " (resource " + std::to_string(d.resource_id) +
                                   ")");
    }
}

// ---------------------------------------------------------------------------------------------
// Command list
// ---------------------------------------------------------------------------------------------

CommandList::CommandList(com::Ref<Device> device, uint32_t type)
    : ChildBase(std::move(device)), type_(type), bridge_(device_->state_bridge(), encoder_) {
    retain_device_copy_cache(device_.get());
    device_->command_lists.fetch_add(1);
}

CommandList::~CommandList() {
    erase_aux(this);
    release_device_copy_cache(device_.get());
    device_->command_lists.fetch_sub(1);
}

void CommandList::note(const std::string& message) {
    ++diag_count_;
    device_->note_bad_argument(message);
}

bool CommandList::usable() {
    if (!open_) {
        note("ID3D12GraphicsCommandList: call on a closed command list ignored");
        return false;
    }
    return true;
}

Resource* CommandList::resource(const InterfaceArg<IResourceHost>& arg) const {
    return arg ? static_cast<Resource*>(arg.get()) : nullptr;
}

void CommandList::reset(PipelineState* initial_state) {
    encoder_.clear();
    deferred_.clear();
    immediate_writes_.clear();
    readbacks_.clear();
    compute_written_.clear();
    graphics_root_args_ = {};
    graphics_root_signature_ = nullptr;
    vertex_buffers_ = {};
    index_buffer_ = {};
    rtv_handles_.clear();
    dsv_handle_ = 0;
    rect_list_ = false;
    rect_expansions_.clear();
    rect_buffers_used_ = 0;
    bridge_vb_views_ = {};
    pinned_.clear();
    graphics_pso_ = nullptr;
    compute_pso_ = nullptr;
    aux_of(this).reset_all();
    bridge_.Reset(nullptr);
    open_ = true;
    if (initial_state != nullptr) {
        bind_pipeline(initial_state);
    }
}

HResult CommandList::Close() {
    if (!open_) {
        note("ID3D12GraphicsCommandList::Close on a closed list");
        return kEFail;
    }
    emit_compute_texture_readbacks();
    bridge_.Close();
    open_ = false;
    return kSOk;
}

HResult CommandList::Reset(InterfaceArg<ICommandAllocatorHost> allocator, InterfaceArg<IPipelineStateHost> initial_state) {
    if (!allocator || initial_state.invalid()) {
        note("ID3D12GraphicsCommandList::Reset: invalid allocator or pipeline state");
        return kEInvalidArg;
    }
    if (open_) {
        note("ID3D12GraphicsCommandList::Reset on an open list");
        return kEFail;
    }
    reset(initial_state ? static_cast<PipelineState*>(initial_state.get()) : nullptr);
    return kSOk;
}

// Xbox slot 139. The executable's command-list pool takes a closed list from its ring and calls
// slot 139 with (an ID3D12CommandAllocator, null, 0), checks the HRESULT, then records into the
// list: a Reset onto that allocator. The third argument is recorded when non-zero (its meaning
// is unknown; the boot has only shown 0).
HResult CommandList::ResetX(InterfaceArg<ICommandAllocatorHost> allocator, InterfaceArg<IPipelineStateHost> initial_state,
                            uint64_t extra) {
    if (extra != 0) {
        note("ID3D12GraphicsCommandList Xbox slot 139: third argument " + hex(extra) + " is not interpreted");
    }
    return Reset(std::move(allocator), std::move(initial_state));
}

void CommandList::ClearState(InterfaceArg<IPipelineStateHost> state) {
    if (!usable()) {
        return;
    }
    // Keep the recorded packets; drop the bound state to D3D12 defaults and rebind `state`.
    graphics_pso_ = nullptr;
    compute_pso_ = nullptr;
    graphics_root_args_ = {};
    graphics_root_signature_ = nullptr;
    vertex_buffers_ = {};
    index_buffer_ = {};
    rtv_handles_.clear();
    dsv_handle_ = 0;
    rect_list_ = false;
    bridge_vb_views_ = {};
    aux_of(this).reset_dynamic_state();
    bridge_.SetPipelineState(nullptr);
    bridge_.SetComputePipelineState(nullptr);
    bridge_.RSSetViewports(0, nullptr);
    bridge_.RSSetScissorRects(0, nullptr);
    bridge_.OMSetBlendFactor(nullptr);
    bridge_.OMSetStencilRef(0);
    bridge_.OMSetRenderTargets(0, nullptr, false, nullptr);
    bridge_.IASetIndexBuffer(nullptr);
    if (state) {
        bind_pipeline(static_cast<PipelineState*>(state.get()));
    }
}

void CommandList::bind_pipeline(PipelineState* state) {
    pinned_.push_back(com::Ref<com::ComObject>::share(state));
    if (state->is_compute()) {
        compute_pso_ = state;
        bridge_.SetComputePipelineState(state->resolved() ? state->compute() : nullptr);
    } else if (state->mesh() != nullptr) {
        graphics_pso_ = state;
        bridge_.SetMeshPipelineState(state->resolved() ? state->mesh() : nullptr);
    } else {
        graphics_pso_ = state;
        bridge_.SetPipelineState(state->resolved() ? state->graphics() : nullptr);
    }
    if (!state->resolved()) {
        note("pipeline state without shader modules bound: its draws/dispatches are skipped");
    }
}

void CommandList::SetPipelineState(InterfaceArg<IPipelineStateHost> state) {
    if (!usable()) {
        return;
    }
    if (!state) {
        note("SetPipelineState: invalid pipeline state " + hex(state.va()));
        return;
    }
    bind_pipeline(static_cast<PipelineState*>(state.get()));
}

void CommandList::record_root_arg(uint32_t index, uint8_t kind, uint64_t value, uint32_t count) {
    if (index < graphics_root_args_.size()) {
        RootArgRecord& r = graphics_root_args_[index];
        r.kind = kind;
        r.value = value;
        r.count = kind == 2 ? std::max(r.count, count) : count;
    }
}

namespace {
const char* range_type_name(d3d::DescriptorRangeType t) {
    switch (t) {
        case d3d::DescriptorRangeType::Srv:
            return "SRV";
        case d3d::DescriptorRangeType::Uav:
            return "UAV";
        case d3d::DescriptorRangeType::Cbv:
            return "CBV";
        default:
            return "Sampler";
    }
}
const char* descriptor_kind_name(d3d::DescriptorKind k) {
    switch (k) {
        case d3d::DescriptorKind::Cbv:
            return "CBV";
        case d3d::DescriptorKind::Srv:
            return "SRV";
        case d3d::DescriptorKind::Uav:
            return "UAV";
        case d3d::DescriptorKind::Sampler:
            return "sampler";
        case d3d::DescriptorKind::Rtv:
            return "RTV";
        case d3d::DescriptorKind::Dsv:
            return "DSV";
        default:
            return "empty";
    }
}
}  // namespace

// The first skipped draw of each graphics pipeline state, described with counts, formats and
// sizes only (no bytecode, no identities): what the draw would need to become visible.
void CommandList::describe_dropped_draw(const char* call, uint32_t count, uint32_t instances) {
    char reason[160];
    std::snprintf(reason, sizeof(reason), "%s (state bridge diagnostics: state 0x%" PRIx64 ", pipeline 0x%" PRIx64 ")",
                  graphics_pso_ == nullptr ? "no graphics pipeline state bound" : "dropped by the state bridge",
                  bridge_.state_diagnostics(), bridge_.pipeline_diagnostics());
    describe_skipped_draw(call, count, instances, reason);
}

void CommandList::describe_skipped_draw(const char* call, uint32_t count, uint32_t instances, const char* reason) {
    static const char kNoPipeline = 0;
    const void* key = graphics_pso_ != nullptr ? static_cast<const void*>(graphics_pso_) : &kNoPipeline;
    if (!device_->claim_skipped_draw_report(key)) {
        return;
    }
    std::string s = std::string("d3d12_xs: skipped ") + call + " (" + std::to_string(count) + " x " +
                    std::to_string(instances) + "): " + reason + ";";
    if (compute_pso_ != nullptr && graphics_pso_ == nullptr) {
        s += " a compute pipeline state is bound";
        for (uint64_t id : compute_pso_->shader_identities()) {
            s += " (CS " + std::to_string(device_->shader_size(id)) + " B, DXIL kind " +
                 std::to_string(device_->shader_kind(id)) + ")";
        }
        s += ";";
    }
    const GraphicsPsoSource* src = graphics_pso_ != nullptr ? graphics_pso_->graphics_source() : nullptr;
    if (graphics_pso_ != nullptr && src == nullptr) {
        s += std::string(" the bound pipeline state has no graphics description") +
             (graphics_pso_->unsupported_reason().empty() ? "" : " (" + graphics_pso_->unsupported_reason() + ")") + ";";
    }
    if (const PipelineState::Native* n = graphics_pso_ != nullptr ? graphics_pso_->native() : nullptr) {
        const xbox::NativePipeline& l = n->layout;
        s += " native stages: vertex " + std::to_string(l.vertex.code.size) + " B code + " +
             std::to_string(l.vertex.metadata.size) + " B metadata (ranges " + (l.vertex_alt ? "6/7" : "4/5") + ")";
        s += l.pixel.present() ? ", pixel " + std::to_string(l.pixel.code.size) + " B code + " +
                                     std::to_string(l.pixel.metadata.size) + " B metadata"
                               : ", no pixel stage";
        s += "; header record " + std::to_string(l.header.size) + " B, pipeline metadata " +
             std::to_string(l.metadata.size) + " B;";
    }
    // Shaders, topology, input layout and formats exist only for a graphics description.
    if (src != nullptr) {
        const pso::D3D12GraphicsPsoDesc& d = src->desc;
        const char* stage_names[] = {"VS", "PS", "DS", "HS", "GS", "AS", "MS"};
        const uint64_t stages[] = {d.vs, d.ps, d.ds, d.hs, d.gs, d.as, d.ms};
        s += " shaders";
        for (size_t i = 0; i < 7; ++i) {
            if (stages[i] != 0) {
                s += std::string(" ") + stage_names[i] + " " + std::to_string(device_->shader_size(stages[i])) +
                     " B DXIL kind " + std::to_string(device_->shader_kind(stages[i])) +
                     (device_->shader_known(stages[i]) ? "" : " (no module)");
            }
        }
        for (const auto& [name, n] : {std::pair{"VS", &src->native_vs}, std::pair{"PS", &src->native_ps}}) {
            if (n->identity != 0) {
                s += std::string("; ") + name + " joins native stage " + std::to_string(n->code_bytes) + " B code + " +
                     std::to_string(n->metadata_bytes) + " B metadata";
            }
        }
        const pso::D3D12RenderTargetBlendDesc& rt0 = d.BlendState.RenderTarget[0];
        s += "; blend rt0 " + std::string(rt0.BlendEnable ? "on" : "off") + " mask " +
             std::to_string(static_cast<uint32_t>(rt0.RenderTargetWriteMask)) + ", depth " +
             (d.DepthStencilState.DepthEnable ? "on func " + std::to_string(static_cast<uint32_t>(d.DepthStencilState.DepthFunc))
                                              : std::string("off")) +
             ", cull " + std::to_string(static_cast<uint32_t>(d.RasterizerState.CullMode));
        s += "; topology type " + std::to_string(static_cast<uint32_t>(d.PrimitiveTopologyType)) +
             (src->rect_list ? " (rectangle list)" : "");
        s += "; input layout " + std::to_string(d.NumInputElements) + " element(s)";
        for (uint32_t i = 0; i < d.NumInputElements && d.pInputElementDescs != nullptr; ++i) {
            const pso::D3D12InputElementDesc& e = d.pInputElementDescs[i];
            s += std::string(i == 0 ? " [" : ", ") + (e.SemanticName != nullptr ? e.SemanticName : "?") +
                 std::to_string(e.SemanticIndex) + " fmt " + std::to_string(static_cast<uint32_t>(e.Format)) +
                 " slot " + std::to_string(e.InputSlot) + (i + 1 == d.NumInputElements ? "]" : "");
        }
        s += "; render targets " + std::to_string(d.NumRenderTargets);
        for (uint32_t i = 0; i < d.NumRenderTargets && i < pso::kD3D12SimultaneousRenderTargetCount; ++i) {
            s += std::string(i == 0 ? " [fmt " : ", ") + std::to_string(static_cast<uint32_t>(d.RTVFormats[i])) +
                 (i + 1 == d.NumRenderTargets ? "]" : "");
        }
        s += " depth fmt " + std::to_string(static_cast<uint32_t>(d.DSVFormat)) + " samples " +
             std::to_string(d.SampleDesc.Count);
    } else {
        s += " no graphics description";
    }

    // Root signature layout and what is bound for each parameter.
    const d3d::RootSignature* rs = src != nullptr && src->root_signature ? &src->root_signature->signature()
                                                                         : graphics_root_signature_;
    if (rs == nullptr) {
        s += "; no root signature";
    } else {
        const d3d::RootSignatureDesc& rd = rs->desc();
        s += "; root signature 1." + std::to_string(static_cast<uint32_t>(rd.version) - 1) + ", " +
             std::to_string(rd.parameters.size()) + " parameter(s), " + std::to_string(rd.static_samplers.size()) +
             " static sampler(s)";
        for (size_t p = 0; p < rd.parameters.size(); ++p) {
            const d3d::RootParameterDesc& param = rd.parameters[p];
            s += "; [" + std::to_string(p) + " vis " + std::to_string(static_cast<uint32_t>(param.shader_visibility)) +
                 "] ";
            const RootArgRecord arg = p < graphics_root_args_.size() ? graphics_root_args_[p] : RootArgRecord{};
            switch (param.parameter_type) {
                case d3d::RootParameterType::DescriptorTable: {
                    s += "table";
                    uint32_t total = 0;
                    for (const d3d::DescriptorRangeDesc& r : param.ranges) {
                        s += std::string(" ") + range_type_name(r.range_type) + " x" +
                             (r.num_descriptors == d3d::kUnboundedDescriptorCount ? std::string("unbounded")
                                                                                  : std::to_string(r.num_descriptors));
                        total += r.num_descriptors == d3d::kUnboundedDescriptorCount ? 4 : r.num_descriptors;
                    }
                    if (arg.kind != 1) {
                        s += " -> not set";
                        break;
                    }
                    d3d::ResolvedDescriptor res;
                    if (device_->heaps().resolve_gpu(d3d::GpuDescriptorHandle{arg.value}, &res) != d3d::Status::Ok) {
                        s += " -> unresolvable handle";
                        break;
                    }
                    s += " ->";
                    for (uint32_t i = 0; i < std::min<uint32_t>(total, 8); ++i) {
                        if (const d3d::ViewDescriptor* v = res.heap->view(res.index + i)) {
                            s += std::string(" ") + descriptor_kind_name(v->kind) + " dim " + std::to_string(v->dimension) +
                                 " fmt " + std::to_string(v->format);
                            bridge::TextureInfo t;
                            if (v->resource_id != 0 &&
                                device_->state_bridge().texture_info(v->resource_id, &t)) {
                                s += " tex " + std::to_string(t.width) + "x" + std::to_string(t.height) + " fmt " +
                                     std::to_string(static_cast<uint32_t>(t.format));
                            } else if (v->kind == d3d::DescriptorKind::Cbv) {
                                s += " " + std::to_string(v->num_elements_or_size) + " B";
                            } else if (v->resource_id != 0) {
                                s += " buffer " + std::to_string(v->num_elements_or_size) + " x " +
                                     std::to_string(v->structure_byte_stride) + " B";
                            }
                        } else if (res.heap->sampler(res.index + i) != nullptr) {
                            s += " sampler";
                        }
                    }
                    break;
                }
                case d3d::RootParameterType::Constants32Bit:
                    s += std::to_string(param.num_32bit_values) + " root constants" +
                         (arg.kind == 2 ? "" : " (not set)");
                    break;
                default:
                    s += std::string(param.parameter_type == d3d::RootParameterType::Cbv   ? "root CBV"
                                     : param.parameter_type == d3d::RootParameterType::Srv ? "root SRV"
                                                                                           : "root UAV") +
                         (arg.kind >= 3 ? "" : " (not set)");
                    break;
            }
        }
    }

    // Vertex / index buffers and render targets bound now.
    uint32_t vbs = 0;
    for (const VertexBufferRecord& v : vertex_buffers_) {
        if (v.va != 0) {
            s += std::string(vbs++ == 0 ? "; vertex buffers" : ",") + " " + std::to_string(v.size) + " B stride " +
                 std::to_string(v.stride);
        }
    }
    if (vbs == 0) {
        s += "; no vertex buffers";
    }
    s += index_buffer_.va != 0
             ? "; index buffer " + std::to_string(index_buffer_.size) + " B fmt " + std::to_string(index_buffer_.stride)
             : "; no index buffer";
    for (size_t i = 0; i < rtv_handles_.size(); ++i) {
        d3d::ResolvedDescriptor res;
        const d3d::ViewDescriptor* v = nullptr;
        if (device_->heaps().resolve_cpu(d3d::CpuDescriptorHandle{rtv_handles_[i]}, &res) == d3d::Status::Ok) {
            v = res.heap->view(res.index);
        }
        bridge::TextureInfo t;
        s += std::string(i == 0 ? "; bound render targets" : ",");
        if (v != nullptr && v->resource_id != 0 && device_->state_bridge().texture_info(v->resource_id, &t)) {
            s += " " + std::to_string(t.width) + "x" + std::to_string(t.height) + " fmt " +
                 std::to_string(static_cast<uint32_t>(t.format)) + (t.swapchain ? " (back buffer)" : "");
        } else {
            s += " unresolved";
        }
    }
    device_->add_skipped_draw_report(s);
}

void CommandList::DrawInstanced(uint32_t vertices, uint32_t instances, uint32_t start_vertex, uint32_t start_instance) {
    if (!usable()) {
        return;
    }
    note_draw_for_dump();
    DrawOutcome outcome(bridge_);
    outcome.fallback = Work::DrawsRectList;
    outcome.identities = graphics_pso_ != nullptr ? &graphics_pso_->shader_identities() : nullptr;
    if (graphics_pso_ != nullptr && !graphics_pso_->resolved()) {
        outcome.fallback = graphics_pso_->unsupported_reason().empty() ? Work::DrawsNoShaderModule
                                                                       : Work::DrawsMeshPipeline;
        device_->note_skipped_draw();
        describe_skipped_draw("DrawInstanced", vertices, instances, "pipeline state without shader modules");
        return;
    }
    CommandListAux& aux = aux_of(this);
    const d3d::RootSignature* active_rs =
        graphics_root_signature_ != nullptr
            ? graphics_root_signature_
            : ((graphics_pso_ != nullptr && graphics_pso_->graphics_source() != nullptr &&
                graphics_pso_->graphics_source()->root_signature)
                   ? &graphics_pso_->graphics_source()->root_signature->signature()
                   : nullptr);
    sync_depth_srvs_for_tables(*device_, aux, active_rs, [&](size_t p) -> uint64_t {
        return p < graphics_root_args_.size() && graphics_root_args_[p].kind == 1 ? graphics_root_args_[p].value : 0ull;
    });
    if (rect_list_) {
        std::string why;
        if (!expand_rect_list_draw(vertices, instances, start_vertex, start_instance, &why)) {
            skip_rect_list_draw("DrawInstanced", vertices, instances, why.c_str());
        }
        return;
    }
    const uint64_t drawn_before = bridge_.stats().draws;
    const uint64_t dropped = bridge_.stats().draws_skipped;
    bridge_.DrawInstanced(vertices, instances, start_vertex, start_instance);
    if (bridge_.stats().draws != drawn_before && aux.bound_dsv.resource_id != 0 && !aux.bound_dsv.depth_read_only) {
        const GraphicsPsoSource* src = graphics_pso_ != nullptr ? graphics_pso_->graphics_source() : nullptr;
        const bool depth_writes = src == nullptr || (src->desc.DepthStencilState.DepthEnable &&
                                                     src->desc.DepthStencilState.DepthWriteMask != pso::D3D12DepthWriteMask::Zero);
        if (depth_writes) {
            aux.depth_twin.note_depth_written(aux.bound_dsv.resource_id, aux.bound_dsv.mip, 1u,
                                              aux.bound_dsv.base_layer, aux.bound_dsv.layer_count);
        }
    }
    if (bridge_.stats().draws_skipped != dropped) {
        describe_dropped_draw("DrawInstanced", vertices, instances);
    }
}

namespace {

// Per-vertex attribute formats a rectangle's implied corner can be computed for: component count
// and encoding. Integer formats use wrapping integer arithmetic, the others float arithmetic
// (normalized formats clamped and rounded).
enum class RectComp { F32, F16, Unorm8, Snorm8, Unorm16, Snorm16, U32, S32 };
struct RectFormat {
    RectComp comp;
    uint32_t count;
};

bool rect_format(uint32_t dxgi, RectFormat* out) {
    switch (dxgi) {
        case 2: *out = {RectComp::F32, 4}; return true;   // R32G32B32A32_FLOAT
        case 6: *out = {RectComp::F32, 3}; return true;   // R32G32B32_FLOAT
        case 16: *out = {RectComp::F32, 2}; return true;  // R32G32_FLOAT
        case 41: *out = {RectComp::F32, 1}; return true;  // R32_FLOAT
        case 10: *out = {RectComp::F16, 4}; return true;  // R16G16B16A16_FLOAT
        case 34: *out = {RectComp::F16, 2}; return true;  // R16G16_FLOAT
        case 54: *out = {RectComp::F16, 1}; return true;  // R16_FLOAT
        case 28: case 87: case 88: *out = {RectComp::Unorm8, 4}; return true;  // R8G8B8A8 / B8G8R8A8 / B8G8R8X8_UNORM
        case 49: *out = {RectComp::Unorm8, 2}; return true;  // R8G8_UNORM
        case 61: *out = {RectComp::Unorm8, 1}; return true;  // R8_UNORM
        case 31: *out = {RectComp::Snorm8, 4}; return true;  // R8G8B8A8_SNORM
        case 51: *out = {RectComp::Snorm8, 2}; return true;  // R8G8_SNORM
        case 11: *out = {RectComp::Unorm16, 4}; return true;  // R16G16B16A16_UNORM
        case 35: *out = {RectComp::Unorm16, 2}; return true;  // R16G16_UNORM
        case 13: *out = {RectComp::Snorm16, 4}; return true;  // R16G16B16A16_SNORM
        case 37: *out = {RectComp::Snorm16, 2}; return true;  // R16G16_SNORM
        case 3: *out = {RectComp::U32, 4}; return true;  // R32G32B32A32_UINT
        case 7: *out = {RectComp::U32, 3}; return true;
        case 17: *out = {RectComp::U32, 2}; return true;
        case 42: *out = {RectComp::U32, 1}; return true;
        case 4: *out = {RectComp::S32, 4}; return true;  // R32G32B32A32_SINT
        case 8: *out = {RectComp::S32, 3}; return true;
        case 18: *out = {RectComp::S32, 2}; return true;
        case 43: *out = {RectComp::S32, 1}; return true;
        default: return false;
    }
}

uint32_t rect_comp_bytes(RectComp c) {
    switch (c) {
        case RectComp::F32: case RectComp::U32: case RectComp::S32: return 4;
        case RectComp::F16: case RectComp::Unorm16: case RectComp::Snorm16: return 2;
        default: return 1;
    }
}

float half_to_float(uint16_t h) {
    const uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1F, man = h & 0x3FF;
    float v;
    if (exp == 0) {
        v = std::ldexp(static_cast<float>(man), -24);
    } else if (exp == 31) {
        v = man != 0 ? std::numeric_limits<float>::quiet_NaN() : std::numeric_limits<float>::infinity();
    } else {
        v = std::ldexp(static_cast<float>(man | 0x400), static_cast<int>(exp) - 25);
    }
    return sign ? -v : v;
}

uint16_t float_to_half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint16_t sign = static_cast<uint16_t>((x >> 16) & 0x8000);
    const int32_t exp = static_cast<int32_t>((x >> 23) & 0xFF) - 127 + 15;
    uint32_t man = x & 0x7FFFFF;
    if (((x >> 23) & 0xFF) == 0xFF) {
        return sign | 0x7C00 | (man != 0 ? 0x200 : 0);
    }
    if (exp >= 31) {
        return sign | 0x7C00;
    }
    if (exp <= 0) {
        if (exp < -10) {
            return sign;
        }
        man |= 0x800000;
        const uint32_t shift = static_cast<uint32_t>(14 - exp);
        uint32_t m = man >> shift;
        if ((man >> (shift - 1)) & 1) {
            ++m;  // round half up
        }
        return sign | static_cast<uint16_t>(m);
    }
    uint32_t h = (static_cast<uint32_t>(exp) << 10) | (man >> 13);
    if (man & 0x1000) {
        ++h;  // round half up (may carry into the exponent, which is correct)
    }
    return sign | static_cast<uint16_t>(h);
}

// One component of the implied corner: d = a + c - b.
void rect_corner_component(RectComp k, const uint8_t* a, const uint8_t* b, const uint8_t* c, uint8_t* d) {
    auto rd = [](const uint8_t* p, auto t) {
        decltype(t) v;
        std::memcpy(&v, p, sizeof(v));
        return v;
    };
    switch (k) {
        case RectComp::F32: {
            const float v = rd(a, 0.0f) + rd(c, 0.0f) - rd(b, 0.0f);
            std::memcpy(d, &v, 4);
            break;
        }
        case RectComp::F16: {
            const uint16_t v = float_to_half(half_to_float(rd(a, uint16_t{})) + half_to_float(rd(c, uint16_t{})) -
                                             half_to_float(rd(b, uint16_t{})));
            std::memcpy(d, &v, 2);
            break;
        }
        case RectComp::Unorm8: {
            const int v = int{*a} + int{*c} - int{*b};
            *d = static_cast<uint8_t>(std::clamp(v, 0, 255));
            break;
        }
        case RectComp::Snorm8: {
            const int v = int{static_cast<int8_t>(*a)} + int{static_cast<int8_t>(*c)} - int{static_cast<int8_t>(*b)};
            *d = static_cast<uint8_t>(static_cast<int8_t>(std::clamp(v, -127, 127)));
            break;
        }
        case RectComp::Unorm16: {
            const int v = int{rd(a, uint16_t{})} + int{rd(c, uint16_t{})} - int{rd(b, uint16_t{})};
            const uint16_t o = static_cast<uint16_t>(std::clamp(v, 0, 65535));
            std::memcpy(d, &o, 2);
            break;
        }
        case RectComp::Snorm16: {
            const int v = int{rd(a, int16_t{})} + int{rd(c, int16_t{})} - int{rd(b, int16_t{})};
            const int16_t o = static_cast<int16_t>(std::clamp(v, -32767, 32767));
            std::memcpy(d, &o, 2);
            break;
        }
        case RectComp::U32:
        case RectComp::S32: {
            const uint32_t v = rd(a, uint32_t{}) + rd(c, uint32_t{}) - rd(b, uint32_t{});
            std::memcpy(d, &v, 4);
            break;
        }
    }
}

}  // namespace

bool CommandList::expand_rect_list_draw(uint32_t vertices, uint32_t instances, uint32_t start_vertex,
                                        uint32_t start_instance, std::string* why) {
    const GraphicsPsoSource* src = graphics_pso_ != nullptr ? graphics_pso_->graphics_source() : nullptr;
    if (src == nullptr) {
        *why = "rectangle-list topology without a graphics pipeline description";
        return false;
    }
    if (vertices == 0 || vertices % 3 != 0) {
        *why = "rectangle-list draw with a vertex count that is not a multiple of 3";
        return false;
    }
    const pso::D3D12GraphicsPsoDesc& d = src->desc;
    // Per-vertex slots and their elements (per-instance slots are left as they are).
    std::map<uint32_t, std::vector<std::pair<uint32_t, uint32_t>>> slots;
    std::array<uint32_t, 32> next_offset{};
    for (uint32_t i = 0; i < d.NumInputElements && d.pInputElementDescs != nullptr; ++i) {
        const pso::D3D12InputElementDesc& e = d.pInputElementDescs[i];
        const uint32_t fmt = static_cast<uint32_t>(e.Format);
        RectFormat rf{};
        if (e.InputSlot >= 32) {
            *why = "rectangle-list draw with an input slot out of range";
            return false;
        }
        const uint32_t offset = e.AlignedByteOffset == pso::kD3D12AppendAlignedElement ? next_offset[e.InputSlot]
                                                                                       : e.AlignedByteOffset;
        if (!rect_format(fmt, &rf)) {
            *why = "rectangle-list draw with vertex format " + std::to_string(fmt) + " (implied corner not computed)";
            return false;
        }
        next_offset[e.InputSlot] = offset + rf.count * rect_comp_bytes(rf.comp);
        if (e.InputSlotClass == pso::D3D12InputClassification::PerVertexData) {
            slots[e.InputSlot].emplace_back(offset, fmt);
        }
    }
    const uint32_t rects = vertices / 3;
    std::vector<std::pair<uint32_t, RectExpansion>> plan;
    for (const auto& [slot, elements] : slots) {
        const VertexBufferRecord& vb = vertex_buffers_[slot];
        if (vb.va == 0 || vb.stride == 0) {
            *why = "rectangle-list draw without a vertex buffer in slot " + std::to_string(slot);
            return false;
        }
        for (const auto& [offset, fmt] : elements) {
            RectFormat rf{};
            rect_format(fmt, &rf);
            if (offset + rf.count * rect_comp_bytes(rf.comp) > vb.stride) {
                *why = "rectangle-list draw with an element beyond the vertex stride";
                return false;
            }
        }
        if ((uint64_t{start_vertex} + uint64_t{vertices}) * vb.stride > vb.size) {
            *why = "rectangle-list draw beyond its vertex buffer";
            return false;
        }
        const uint64_t cpu = device_->cpu_address_for_gpu(vb.va);
        uint8_t probe = 0;
        if (!com::read_guest(cpu, &probe, 1)) {
            *why = "rectangle-list draw from a vertex buffer without CPU-visible memory";
            return false;
        }
        RectExpansion x;
        x.src = cpu + uint64_t{start_vertex} * vb.stride;
        x.stride = vb.stride;
        x.rects = rects;
        x.elements = elements;
        plan.emplace_back(slot, std::move(x));
    }
    // List-owned vertex buffers, reused across recordings.
    std::vector<std::pair<uint32_t, d3d12_xs::D3D12VertexBufferView>> views;
    for (auto& [slot, x] : plan) {
        const uint64_t bytes = uint64_t{x.rects} * 6 * x.stride;
        if (rect_buffers_used_ == rect_buffers_.size() || rect_buffers_[rect_buffers_used_]->desc().Width < bytes) {
            com::Ref<Resource> buffer = device_->create_internal_buffer(std::max<uint64_t>(bytes, 4096));
            if (!buffer) {
                *why = "rectangle-list draw: no vertex buffer for the expansion";
                return false;
            }
            if (rect_buffers_used_ == rect_buffers_.size()) {
                rect_buffers_.push_back(std::move(buffer));
            } else {
                rect_buffers_[rect_buffers_used_] = std::move(buffer);
            }
        }
        Resource* buffer = rect_buffers_[rect_buffers_used_++].get();
        x.buffer_id = buffer->id();
        d3d12_xs::D3D12VertexBufferView v{};
        v.buffer_id = buffer->id();
        v.offset = 0;
        v.size_in_bytes = static_cast<uint32_t>(bytes);
        v.stride_in_bytes = x.stride;
        views.emplace_back(slot, v);
        rect_expansions_.push_back(std::move(x));
    }
    for (const auto& [slot, v] : views) {
        bridge_.IASetVertexBuffers(slot, 1, &v);
    }
    bridge_.IASetPrimitiveTopology(pso::D3DPrimitiveTopology::TriangleList);
    const uint64_t dropped = bridge_.stats().draws_skipped;
    bridge_.DrawInstanced(rects * 6, instances, 0, start_instance);
    if (bridge_.stats().draws_skipped != dropped) {
        describe_dropped_draw("DrawInstanced (expanded rectangle list)", rects * 6, instances);
    }
    for (const auto& [slot, v] : views) {
        bridge_.IASetVertexBuffers(slot, 1, &bridge_vb_views_[slot]);
    }
    bridge_.IASetPrimitiveTopology(static_cast<pso::D3DPrimitiveTopology>(xbox::kTopologyRectList));
    device_->count(xbox::Counter::RectListDrawsExpanded);
    return true;
}

void CommandList::note_draw_for_dump() {
    if (graphics_pso_ != nullptr && graphics_pso_->dump_index() != PipelineState::kNoDumpIndex) {
        if (PipelineDump* dump = device_->pipeline_dump()) {
            dump->draw(graphics_pso_->dump_index());
        }
    }
}

// Rectangle-list draws that cannot be expanded are skipped: as triangle lists they would cover
// half of each rectangle.
bool CommandList::skip_rect_list_draw(const char* call, uint32_t count, uint32_t instances, const char* reason) {
    if (!rect_list_) {
        return false;
    }
    device_->note_skipped_draw();
    device_->count(xbox::Counter::RectListDrawsSkipped);
    describe_skipped_draw(call, count, instances, reason);
    return true;
}

void CommandList::DrawIndexedInstanced(uint32_t indices, uint32_t instances, uint32_t start_index, int32_t base_vertex,
                                       uint32_t start_instance) {
    if (!usable()) {
        return;
    }
    note_draw_for_dump();
    DrawOutcome outcome(bridge_);
    outcome.fallback = Work::DrawsRectList;
    outcome.identities = graphics_pso_ != nullptr ? &graphics_pso_->shader_identities() : nullptr;
    if (graphics_pso_ != nullptr && !graphics_pso_->resolved()) {
        outcome.fallback = graphics_pso_->unsupported_reason().empty() ? Work::DrawsNoShaderModule
                                                                       : Work::DrawsMeshPipeline;
        device_->note_skipped_draw();
        describe_skipped_draw("DrawIndexedInstanced", indices, instances, "pipeline state without shader modules");
        return;
    }
    CommandListAux& aux = aux_of(this);
    const d3d::RootSignature* active_rs =
        graphics_root_signature_ != nullptr
            ? graphics_root_signature_
            : ((graphics_pso_ != nullptr && graphics_pso_->graphics_source() != nullptr &&
                graphics_pso_->graphics_source()->root_signature)
                   ? &graphics_pso_->graphics_source()->root_signature->signature()
                   : nullptr);
    sync_depth_srvs_for_tables(*device_, aux, active_rs, [&](size_t p) -> uint64_t {
        return p < graphics_root_args_.size() && graphics_root_args_[p].kind == 1 ? graphics_root_args_[p].value : 0ull;
    });
    if (skip_rect_list_draw("DrawIndexedInstanced", indices, instances,
                            "indexed rectangle-list topology (not expanded; the title draws none)")) {
        return;
    }
    const uint64_t drawn_before = bridge_.stats().draws;
    const uint64_t dropped = bridge_.stats().draws_skipped;
    bridge_.DrawIndexedInstanced(indices, instances, start_index, base_vertex, start_instance);
    if (bridge_.stats().draws != drawn_before && aux.bound_dsv.resource_id != 0 && !aux.bound_dsv.depth_read_only) {
        const GraphicsPsoSource* src = graphics_pso_ != nullptr ? graphics_pso_->graphics_source() : nullptr;
        const bool depth_writes = src == nullptr || (src->desc.DepthStencilState.DepthEnable &&
                                                     src->desc.DepthStencilState.DepthWriteMask != pso::D3D12DepthWriteMask::Zero);
        if (depth_writes) {
            aux.depth_twin.note_depth_written(aux.bound_dsv.resource_id, aux.bound_dsv.mip, 1u,
                                              aux.bound_dsv.base_layer, aux.bound_dsv.layer_count);
        }
    }
    if (bridge_.stats().draws_skipped != dropped) {
        describe_dropped_draw("DrawIndexedInstanced", indices, instances);
    }
}

void CommandList::Dispatch(uint32_t x, uint32_t y, uint32_t z) {
    if (!usable()) {
        return;
    }
    if (compute_pso_ != nullptr && compute_pso_->dump_index() != PipelineState::kNoDumpIndex) {
        if (PipelineDump* dump = device_->pipeline_dump()) {
            dump->dispatch(compute_pso_->dump_index());
        }
    }
    count_work(Work::DispatchesIssued);
    if (compute_pso_ != nullptr && !compute_pso_->resolved()) {
        count_work(Work::DispatchesNoShaderModule);
        for (uint64_t id : compute_pso_->shader_identities()) {
            drop_table().record(id, DropTable::Reason::NoModule, true);
        }
        device_->note_skipped_draw();
        return;
    }
    CommandListAux& aux = aux_of(this);
    const d3d::RootSignature* active_cs_rs =
        aux.compute_root_signature != nullptr
            ? aux.compute_root_signature
            : ((compute_pso_ != nullptr && compute_pso_->compute() != nullptr)
                   ? compute_pso_->compute()->root_signature()
                   : nullptr);
    sync_depth_srvs_for_tables(*device_, aux, active_cs_rs, [&](size_t p) -> uint64_t {
        return p < aux.compute_root_args.size() && aux.compute_root_args[p].kind == 1
                   ? aux.compute_root_args[p].value
                   : 0ull;
    });
    const uint64_t emitted = bridge_.stats().dispatches;
    const bridge::BridgeStats before = bridge_.stats();
    bridge_.Dispatch(x, y, z);
    bridge_.take_written_textures(&compute_written_);
    count_object_reuse(before, bridge_.stats());
    note_binding_fallbacks(compute_pso_ != nullptr ? &compute_pso_->shader_identities() : nullptr, before,
                           bridge_.stats(), true);
    if (bridge_.stats().dispatches != emitted) {
        count_work(Work::DispatchesEmitted);
    } else {
        count_work(Work::DispatchesDroppedByBridge);
        if (compute_pso_ != nullptr) {
            for (uint64_t id : compute_pso_->shader_identities()) {
                drop_table().record(id, DropTable::Reason::Bridge, true);
            }
        }
    }
}

// ---- Copies ----

void CommandList::CopyBufferRegion(InterfaceArg<IResourceHost> dst_arg, uint64_t dst_offset,
                                   InterfaceArg<IResourceHost> src_arg, uint64_t src_offset, uint64_t bytes) {
    if (!usable()) {
        return;
    }
    Resource* dst = resource(dst_arg);
    Resource* src = resource(src_arg);
    va_watch_resource("copy source (CopyBufferRegion)", src, "to resource id " + std::to_string(dst ? dst->id() : 0));
    va_watch_resource("copy destination (CopyBufferRegion)", dst, "from resource id " + std::to_string(src ? src->id() : 0));
    if (dst == nullptr || src == nullptr || !dst->is_buffer() || !src->is_buffer()) {
        note("CopyBufferRegion: invalid resources or range");
        return;
    }
    const uint64_t dst_w = dst->desc().Width;
    const uint64_t src_w = src->desc().Width;
    if (dst_offset > dst_w || bytes > dst_w - dst_offset || src_offset > src_w || bytes > src_w - src_offset) {
        note("CopyBufferRegion: invalid resources or range");
        return;
    }
    // Zero-byte copy is a valid D3D12 no-op.
    if (bytes == 0) {
        return;
    }
    if (src->cpu_address() == 0 && dst->cpu_address() != 0) {
        // G66: GPU memory into CPU-visible memory (a readback heap): read back.
        pinned_.push_back(com::Ref<com::ComObject>::share(src));
        pinned_.push_back(com::Ref<com::ComObject>::share(dst));
        GpuReadbackBufferPayload p{};
        p.buffer_id = src->id();
        p.src_offset = src_offset;
        p.size = bytes;
        if (readback_target(dst->cpu_address() + dst_offset, bytes, &p.dst_addr, &p.done_addr)) {
            record_readback(GpuOpcode::ReadbackBuffer, &p, sizeof(p), src->id(), dst->cpu_address() + dst_offset, bytes);
        }
        return;
    }
    if (src->cpu_address() == 0) {
        device_->note_unsupported_copy("CopyBufferRegion: GPU-to-GPU buffer copy has no packet (skipped)");
        return;
    }
    // Identical self-copy on a CPU-visible buffer is a safe no-op.
    if (src == dst && dst_offset == src_offset) {
        return;
    }
    // Guard against 32-bit truncation in GpuWriteBufferPayload::dst_offset.
    const uint64_t aligned_end = align_up_u64(dst_offset + bytes, 4u);
    if (dst_offset > UINT32_MAX || aligned_end > UINT32_MAX) {
        note("CopyBufferRegion: destination range exceeds 32-bit WriteBuffer offset limit");
        return;
    }
    // WebGPU queue.writeBuffer requires 4-byte aligned bufferOffset and size.
    const bool dst_unaligned = (dst_offset & 3u) != 0u;
    const bool len_unaligned = (bytes & 3u) != 0u;
    if (dst_unaligned && dst->cpu_address() == 0) {
        note("CopyBufferRegion: unaligned destination offset " + std::to_string(dst_offset) +
             " on GPU-only buffer (skipped for WebGPU 4-byte alignment)");
        return;
    }
    pinned_.push_back(com::Ref<com::ComObject>::share(src));
    pinned_.push_back(com::Ref<com::ComObject>::share(dst));
    DeferredWrite w;
    w.id = dst->id();
    w.dst_offset = dst_offset;
    w.src = src->cpu_address() + src_offset;
    w.size = bytes;
    CommandListAux::DeferredMeta meta{};
    meta.dst_cpu_va = dst->cpu_address();
    meta.dst_width = dst_w;
    meta.dst_allocated = std::max<uint64_t>(device_->state_bridge().buffer_allocated_size(dst->id()),
                                            align_up_u64(dst_w, 4u));
    meta.needs_rmw_align = dst_unaligned || len_unaligned;
    CommandListAux& aux = aux_of(this);
    if (aux.deferred_meta.size() < deferred_.size()) {
        aux.deferred_meta.resize(deferred_.size());
    }
    deferred_.push_back(w);
    aux.deferred_meta.push_back(meta);
}

void CommandList::CopyResource(InterfaceArg<IResourceHost> dst_arg, InterfaceArg<IResourceHost> src_arg) {
    if (!usable()) {
        return;
    }
    Resource* dst = resource(dst_arg);
    Resource* src = resource(src_arg);
    va_watch_resource("copy source (CopyResource)", src, "to resource id " + std::to_string(dst ? dst->id() : 0));
    va_watch_resource("copy destination (CopyResource)", dst, "from resource id " + std::to_string(src ? src->id() : 0));
    if (dst == nullptr || src == nullptr) {
        note("CopyResource: invalid resources");
        return;
    }
    if (dst->is_buffer() && src->is_buffer() && src->cpu_address() != 0) {
        const uint64_t bytes = std::min(dst->desc().Width, src->desc().Width);
        if (bytes == 0 || src == dst) {
            return;
        }
        const uint64_t aligned_end = align_up_u64(bytes, 4u);
        if (aligned_end > UINT32_MAX) {
            note("CopyResource: buffer size exceeds 32-bit WriteBuffer limit");
            return;
        }
        pinned_.push_back(com::Ref<com::ComObject>::share(src));
        pinned_.push_back(com::Ref<com::ComObject>::share(dst));
        DeferredWrite w;
        w.id = dst->id();
        w.src = src->cpu_address();
        w.size = bytes;
        CommandListAux::DeferredMeta meta{};
        meta.dst_cpu_va = dst->cpu_address();
        meta.dst_width = dst->desc().Width;
        meta.dst_allocated = std::max<uint64_t>(device_->state_bridge().buffer_allocated_size(dst->id()),
                                                align_up_u64(dst->desc().Width, 4u));
        meta.needs_rmw_align = (bytes & 3u) != 0u;
        CommandListAux& aux = aux_of(this);
        if (aux.deferred_meta.size() < deferred_.size()) {
            aux.deferred_meta.resize(deferred_.size());
        }
        deferred_.push_back(w);
        aux.deferred_meta.push_back(meta);
        return;
    }
    if (dst->is_buffer() && src->is_buffer() && src->cpu_address() == 0 && dst->cpu_address() != 0) {
        // G66: a whole GPU buffer into CPU-visible memory: read back.
        const uint64_t bytes = std::min(dst->desc().Width, src->desc().Width);
        if (bytes == 0) {
            return;
        }
        pinned_.push_back(com::Ref<com::ComObject>::share(src));
        pinned_.push_back(com::Ref<com::ComObject>::share(dst));
        GpuReadbackBufferPayload p{};
        p.buffer_id = src->id();
        p.size = bytes;
        if (readback_target(dst->cpu_address(), bytes, &p.dst_addr, &p.done_addr)) {
            record_readback(GpuOpcode::ReadbackBuffer, &p, sizeof(p), src->id(), dst->cpu_address(), bytes);
        }
        return;
    }
    if (!dst->is_buffer() && !src->is_buffer()) {
        if (src == dst) {
            return;  // self-copy is a no-op
        }
        if (dst->desc().Dimension != src->desc().Dimension || dst->desc().Width != src->desc().Width ||
            dst->desc().Height != src->desc().Height ||
            dst->desc().DepthOrArraySize != src->desc().DepthOrArraySize ||
            dst->mip_levels() != src->mip_levels() || dst->desc().SampleCount != src->desc().SampleCount) {
            note("CopyResource: incompatible source and destination texture dimensions");
            return;
        }
        const TextureCopyPlan plan = plan_gpu_texture_copy(*device_, dst, src);
        if (plan.mode != TextureCopyPlan::Mode::None) {
            pinned_.push_back(com::Ref<com::ComObject>::share(src));
            pinned_.push_back(com::Ref<com::ComObject>::share(dst));
            device_->sync_bound_texture_memory(src->id(), encoder_);
            bridge_.ResourceBarrier(0, nullptr);
            const uint32_t mips = std::max<uint32_t>(1u, dst->mip_levels());
            const uint32_t layers = plan.is_3d ? 1u : std::max<uint32_t>(1u, dst->array_layers());
            bool all_ok = true;
            for (uint32_t mip = 0; mip < mips && all_ok; ++mip) {
                const uint32_t mw = std::max<uint32_t>(1u, static_cast<uint32_t>(dst->desc().Width >> mip));
                const uint32_t mh = std::max<uint32_t>(1u, dst->desc().Height >> mip);
                const uint32_t md =
                    plan.is_3d ? std::max<uint32_t>(1u, static_cast<uint32_t>(dst->desc().DepthOrArraySize) >> mip) : 1u;
                for (uint32_t layer = 0; layer < layers; ++layer) {
                    SubresourceCopySpec spec{};
                    spec.src_mip = mip;
                    spec.src_layer = layer;
                    spec.dst_mip = mip;
                    spec.dst_layer = layer;
                    spec.width = mw;
                    spec.height = mh;
                    spec.depth = md;
                    spec.full_subresource = true;
                    if (!emit_gpu_texture_subresource_copy(*device_, plan, dst->id(), src->id(), spec, encoder_)) {
                        all_ok = false;
                        break;
                    }
                }
            }
            if (is_depth_dxgi_format(dst->desc().Format)) {
                aux_of(this).depth_twin.note_depth_copied(dst->id(), 0, 0, 0, 0);
            }
            device_->note_texture_copy_destination(dst->id());
            if (all_ok) {
                return;
            }
        }
        if (is_depth_dxgi_format(dst->desc().Format)) {
            aux_of(this).depth_twin.note_depth_copied(dst->id(), 0, 0, 0, 0);
        }
        count_work(plan.mode == TextureCopyPlan::Mode::None ? plan.skip : Work::CopySkipEmitFailed);
        device_->note_unsupported_copy(
            "CopyResource: GPU texture-to-texture copy has no route (skipped): " + std::string(work_name(plan.skip)) +
            ", DXGI " + std::to_string(dst->desc().Format) + " <- " + std::to_string(src->desc().Format) +
            ", dst flags " + std::to_string(static_cast<uint32_t>(dst->desc().Flags)));
        return;
    }
    device_->note_unsupported_copy("CopyResource: only CPU-visible buffer sources are supported (skipped)");
}

void CommandList::CopyTextureRegion(com::CallContext& ctx) {
    // (pDst, DstX, DstY, DstZ, pSrc, pSrcBox)
    if (!usable()) {
        return;
    }
    GuestTextureCopyLocation dst{}, src{};
    if (!com::read_guest(ctx.int_arg(1), dst) || !com::read_guest(ctx.int_arg(5), src)) {
        note("CopyTextureRegion: unreadable copy locations");
        return;
    }
    Resource* tex = as_resource(dst.pResource);
    Resource* buf = as_resource(src.pResource);
    va_watch_resource("copy source (CopyTextureRegion)", buf, "to resource id " + std::to_string(tex ? tex->id() : 0));
    va_watch_resource("copy destination (CopyTextureRegion)", tex, "from resource id " + std::to_string(buf ? buf->id() : 0));
    if (dst.Type == 1 && src.Type == 0 && tex != nullptr && buf != nullptr && tex->is_buffer() && !buf->is_buffer()) {
        // Here `tex` is the footprint buffer (destination) and `buf` the texture (source).
        copy_texture_to_footprint(ctx, dst, src, tex, buf);
        return;
    }
    if (dst.Type == 0 && src.Type == 0 && tex != nullptr && buf != nullptr && !tex->is_buffer() && !buf->is_buffer()) {
        // Texture -> texture subresource copy: validate subresources, box, and emit GPU copy pass.
        Resource* dst_tex = tex;
        Resource* src_tex = buf;
        const DxgiFormatLayout dst_l = dxgi_format_layout(dst_tex->desc().Format);
        const DxgiFormatLayout src_l = dxgi_format_layout(src_tex->desc().Format);
        SubresourceInfo dst_sub{}, src_sub{};
        const uint32_t dst_sub_idx = static_cast<uint32_t>(dst.PlacedFootprint.Offset);
        const uint32_t src_sub_idx = static_cast<uint32_t>(src.PlacedFootprint.Offset);
        if (!dst_l.known || !src_l.known || !decode_texture_subresource(dst_tex, dst_sub_idx, dst_l, &dst_sub) ||
            !decode_texture_subresource(src_tex, src_sub_idx, src_l, &src_sub)) {
            note("CopyTextureRegion: texture-to-texture subresource or format invalid");
            return;
        }
        const uint32_t dst_x = static_cast<uint32_t>(ctx.int_arg(2));
        const uint32_t dst_y = static_cast<uint32_t>(ctx.int_arg(3));
        const uint32_t dst_z = static_cast<uint32_t>(ctx.int_arg(4));
        if ((dst_x % dst_l.block_width) != 0 || (dst_y % dst_l.block_height) != 0) {
            note("CopyTextureRegion: destination origin not aligned to format block size");
            return;
        }
        if ((dst_sub.is_1d && (dst_y != 0 || dst_z != 0)) || (!dst_sub.is_3d && dst_z != 0)) {
            note("CopyTextureRegion: texture-to-texture destination origin invalid for texture dimension");
            return;
        }
        if (dst_x >= dst_sub.phys_w || dst_y >= dst_sub.phys_h || (dst_sub.is_3d && dst_z >= dst_sub.mip_d)) {
            note("CopyTextureRegion: texture-to-texture destination origin exceeds subresource dimensions");
            return;
        }
        GuestBox box{0, 0, 0, src_sub.phys_w, src_sub.phys_h, src_sub.mip_d};
        if (ctx.int_arg(6) != 0 && !com::read_guest(ctx.int_arg(6), box)) {
            note("CopyTextureRegion: unreadable source box");
            return;
        }
        if (box.right <= box.left || box.bottom <= box.top || box.back <= box.front) {
            return;  // empty box is a valid no-op
        }
        if ((box.left % src_l.block_width) != 0 || (box.top % src_l.block_height) != 0) {
            note("CopyTextureRegion: source box origin not aligned to format block size");
            return;
        }
        if (box.left >= src_sub.phys_w || box.top >= src_sub.phys_h || box.front >= src_sub.mip_d) {
            note("CopyTextureRegion: source box origin exceeds subresource dimensions");
            return;
        }
        box.right = std::min(box.right, src_sub.phys_w);
        box.bottom = std::min(box.bottom, src_sub.phys_h);
        box.back = std::min(box.back, src_sub.mip_d);

        // Compressed <-> uncompressed copy of equal bytes per block: the executor copies through a buffer.
        if (gpu_texture_copy_transcode() && src_l.block_bytes != 0 && src_l.block_bytes == dst_l.block_bytes &&
            (src_l.block_width != dst_l.block_width || src_l.block_height != dst_l.block_height) &&
            (src_l.block_width == 1 || dst_l.block_width == 1) && (src_l.block_height == 1 || dst_l.block_height == 1) &&
            !src_sub.is_3d && !dst_sub.is_3d && !src_sub.is_1d && !dst_sub.is_1d && src_tex != dst_tex &&
            src_l.planes == 1 && dst_l.planes == 1 && !is_depth_dxgi_format(src_tex->desc().Format) &&
            !is_depth_dxgi_format(dst_tex->desc().Format) && src_tex->desc().SampleCount <= 1u &&
            dst_tex->desc().SampleCount <= 1u && src_tex->has_gpu_texture() && dst_tex->has_gpu_texture()) {
            bridge::TextureInfo ti_dst{}, ti_src{};
            if (device_->state_bridge().texture_info(dst_tex->id(), &ti_dst) &&
                device_->state_bridge().texture_info(src_tex->id(), &ti_src) && !ti_dst.swapchain && !ti_src.swapchain &&
                ti_dst.dimension == GpuTextureDimension::Dim2D && ti_src.dimension == GpuTextureDimension::Dim2D) {
                // Whole blocks of the source box: a column / row of the box is one block of both textures.
                const uint32_t cols = std::min((align_up_u32(box.right - box.left, src_l.block_width) / src_l.block_width),
                                               std::min((src_sub.phys_w - box.left) / src_l.block_width,
                                                        (dst_sub.phys_w - dst_x) / dst_l.block_width));
                const uint32_t rows = std::min((align_up_u32(box.bottom - box.top, src_l.block_height) / src_l.block_height),
                                               std::min((src_sub.phys_h - box.top) / src_l.block_height,
                                                        (dst_sub.phys_h - dst_y) / dst_l.block_height));
                uint32_t host_src_mip = src_sub.mip;
                uint32_t host_dst_mip = dst_sub.mip;
                if (cols == 0 || rows == 0 ||
                    !device_->state_bridge().translate_host_mip(src_tex->id(), &host_src_mip) ||
                    !device_->state_bridge().translate_host_mip(dst_tex->id(), &host_dst_mip)) {
                    return;  // empty, or a mip trimmed by texture_budget: cleanly skipped
                }
                GpuCopyTexturePayload c{};
                c.dst_id = dst_tex->id();
                c.src_id = src_tex->id();
                c.dst_mip = host_dst_mip;
                c.src_mip = host_src_mip;
                c.dst_x = dst_x;
                c.dst_y = dst_y;
                c.dst_z = dst_sub.layer;
                c.src_x = box.left;
                c.src_y = box.top;
                c.src_z = src_sub.layer;
                c.width = cols * src_l.block_width;
                c.height = rows * src_l.block_height;
                c.depth_or_layers = 1;
                c.reserved = kCopyTextureTranscode;
                pinned_.push_back(com::Ref<com::ComObject>::share(src_tex));
                pinned_.push_back(com::Ref<com::ComObject>::share(dst_tex));
                device_->sync_bound_texture_memory(src_tex->id(), encoder_);
                bridge_.ResourceBarrier(0, nullptr);
                encoder_.emit_packet(kGpuOpcodeCopyTexture, &c, sizeof(c));
                device_->note_texture_copy_destination(dst_tex->id());
                return;
            }
        }

        uint32_t copy_w = align_up_u32(box.right - box.left, src_l.block_width);
        uint32_t copy_h = align_up_u32(box.bottom - box.top, src_l.block_height);
        uint32_t copy_d = box.back - box.front;
        copy_w = std::min(copy_w, std::min(src_sub.phys_w - box.left, dst_sub.phys_w - dst_x));
        copy_h = std::min(copy_h, std::min(src_sub.phys_h - box.top, dst_sub.phys_h - dst_y));
        copy_d = std::min(copy_d, (src_sub.is_3d && dst_sub.is_3d)
                                      ? std::min(src_sub.mip_d - box.front, dst_sub.mip_d - dst_z)
                                      : 1u);
        if (copy_w == 0 || copy_h == 0 || copy_d == 0) {
            return;
        }
        const bool same_subresource =
            src_tex == dst_tex && src_sub.mip == dst_sub.mip &&
            ((!src_sub.is_3d && !dst_sub.is_3d && src_sub.layer == dst_sub.layer) || (src_sub.is_3d && dst_sub.is_3d));
        if (same_subresource) {
            if (box.left == dst_x && box.top == dst_y && (!src_sub.is_3d || box.front == dst_z)) {
                return;  // identical in-place copy is a no-op
            }
            if (is_depth_dxgi_format(dst_tex->desc().Format)) {
                aux_of(this).depth_twin.note_depth_copied(dst_tex->id(), dst_sub.mip, 1u,
                                                          dst_sub.is_3d ? dst_z : dst_sub.layer, copy_d);
            }
            count_work(Work::CopySkipOverlap);
            device_->note_unsupported_copy(
                "CopyTextureRegion: same-subresource overlapping texture copy cannot alias in WebGPU (skipped)");
            return;
        }

        const TextureCopyPlan plan = plan_gpu_texture_copy(*device_, dst_tex, src_tex);
        if (plan.mode != TextureCopyPlan::Mode::None) {
            SubresourceCopySpec spec{};
            spec.src_mip = src_sub.mip;
            spec.src_layer = src_sub.layer;
            spec.src_x = box.left;
            spec.src_y = box.top;
            spec.src_z = src_sub.is_3d ? box.front : 0u;
            spec.dst_mip = dst_sub.mip;
            spec.dst_layer = dst_sub.layer;
            spec.dst_x = dst_x;
            spec.dst_y = dst_y;
            spec.dst_z = dst_sub.is_3d ? dst_z : 0u;
            spec.width = std::min(copy_w, dst_sub.mip_w - std::min(dst_x, dst_sub.mip_w));
            spec.height = std::min(copy_h, dst_sub.mip_h - std::min(dst_y, dst_sub.mip_h));
            spec.depth = copy_d;
            spec.full_subresource =
                dst_x == 0 && dst_y == 0 && (!dst_sub.is_3d || dst_z == 0) && spec.width >= dst_sub.mip_w &&
                spec.height >= dst_sub.mip_h && (!dst_sub.is_3d || spec.depth >= dst_sub.mip_d);
            pinned_.push_back(com::Ref<com::ComObject>::share(src_tex));
            pinned_.push_back(com::Ref<com::ComObject>::share(dst_tex));
            device_->sync_bound_texture_memory(src_tex->id(), encoder_);
            bridge_.ResourceBarrier(0, nullptr);
            if (emit_gpu_texture_subresource_copy(*device_, plan, dst_tex->id(), src_tex->id(), spec, encoder_)) {
                if (is_depth_dxgi_format(dst_tex->desc().Format)) {
                    aux_of(this).depth_twin.note_depth_copied(dst_tex->id(), dst_sub.mip, 1u,
                                                              dst_sub.is_3d ? dst_z : dst_sub.layer, copy_d);
                }
                device_->note_texture_copy_destination(dst_tex->id());
                return;
            }
        }
        if (is_depth_dxgi_format(dst_tex->desc().Format)) {
            aux_of(this).depth_twin.note_depth_copied(dst_tex->id(), dst_sub.mip, 1u,
                                                      dst_sub.is_3d ? dst_z : dst_sub.layer, copy_d);
        }
        count_work(plan.mode == TextureCopyPlan::Mode::None ? plan.skip : Work::CopySkipEmitFailed);
        device_->note_unsupported_copy(
            "CopyTextureRegion: GPU texture-to-texture copy has no route (skipped): " + std::string(work_name(plan.skip)) +
            ", DXGI " + std::to_string(dst_tex->desc().Format) + " <- " + std::to_string(src_tex->desc().Format) +
            ", dst flags " + std::to_string(static_cast<uint32_t>(dst_tex->desc().Flags)));
        return;
    }
    if (tex == nullptr || buf == nullptr || dst.Type != 0 || src.Type != 1 || tex->is_buffer() || !buf->is_buffer()) {
        device_->note_unsupported_copy("CopyTextureRegion: only buffer footprint -> texture subresource is supported");
        return;
    }
    if (buf->cpu_address() == 0 || !tex->has_gpu_texture()) {
        device_->note_unsupported_copy("CopyTextureRegion: source not CPU-visible or texture without GPU object");
        return;
    }
    const uint32_t sub = static_cast<uint32_t>(dst.PlacedFootprint.Offset);  // SubresourceIndex
    const GuestSubresourceFootprint& fp = src.PlacedFootprint.Footprint;
    const DxgiFormatLayout l = dxgi_format_layout(fp.Format);
    if (!l.known || fp.RowPitch == 0 || fp.Width == 0 || fp.Height == 0 || fp.Depth == 0) {
        note("CopyTextureRegion: unknown footprint format " + std::to_string(fp.Format));
        return;
    }
    SubresourceInfo sub_info{};
    if (!decode_texture_subresource(tex, sub, l, &sub_info)) {
        note("CopyTextureRegion: subresource " + std::to_string(sub) + " out of range");
        return;
    }
    const uint32_t dst_x = static_cast<uint32_t>(ctx.int_arg(2));
    const uint32_t dst_y = static_cast<uint32_t>(ctx.int_arg(3));
    const uint32_t dst_z = static_cast<uint32_t>(ctx.int_arg(4));
    if ((dst_x % l.block_width) != 0 || (dst_y % l.block_height) != 0) {
        note("CopyTextureRegion: destination origin not aligned to format block size");
        return;
    }
    if (sub_info.is_1d && (dst_y != 0 || dst_z != 0)) {
        note("CopyTextureRegion: 1D texture destination origin y/z must be 0");
        return;
    }
    if (!sub_info.is_3d && dst_z != 0) {
        note("CopyTextureRegion: 2D/array/cubemap destination origin z must be 0");
        return;
    }
    if (dst_x >= sub_info.phys_w || dst_y >= sub_info.phys_h || (sub_info.is_3d && dst_z >= sub_info.mip_d)) {
        note("CopyTextureRegion: destination origin exceeds subresource dimensions");
        return;
    }
    const uint32_t fp_phys_w = align_up_u32(fp.Width, l.block_width);
    const uint32_t fp_phys_h = align_up_u32(fp.Height, l.block_height);
    GuestBox box{0, 0, 0, fp_phys_w, fp_phys_h, fp.Depth};
    if (ctx.int_arg(6) != 0 && !com::read_guest(ctx.int_arg(6), box)) {
        note("CopyTextureRegion: unreadable source box");
        return;
    }
    if (box.right <= box.left || box.bottom <= box.top || box.back <= box.front) {
        return;  // empty box is a valid no-op
    }
    if ((box.left % l.block_width) != 0 || (box.top % l.block_height) != 0) {
        note("CopyTextureRegion: source box origin not aligned to format block size");
        return;
    }
    if (box.left >= fp_phys_w || box.top >= fp_phys_h || box.front >= fp.Depth) {
        note("CopyTextureRegion: source box origin exceeds footprint dimensions");
        return;
    }
    box.right = std::min(box.right, fp_phys_w);
    box.bottom = std::min(box.bottom, fp_phys_h);
    box.back = std::min(box.back, fp.Depth);

    // For block-compressed formats (e.g. BC1-BC7 4x4), round mip-tail copy extents up to whole
    // blocks so WebGPU writeTexture/copyBufferToTexture never rejects 2x2 or 1x1 mip tails.
    uint32_t w = align_up_u32(box.right - box.left, l.block_width);
    uint32_t h = align_up_u32(box.bottom - box.top, l.block_height);
    uint32_t d = box.back - box.front;
    w = std::min(w, sub_info.phys_w - dst_x);
    h = std::min(h, sub_info.phys_h - dst_y);
    d = std::min(d, sub_info.is_3d ? (sub_info.mip_d - dst_z) : 1u);
    if (w == 0 || h == 0 || d == 0) {
        return;
    }
    const uint32_t elem_bytes = l.planes == 2 ? plane_block_bytes(fp.Format, sub_info.plane) : l.block_bytes;
    const uint32_t blocks_x = (w + l.block_width - 1u) / l.block_width;
    const uint32_t rows = (h + l.block_height - 1u) / l.block_height;
    const uint32_t rows_per_slice = std::max(rows, (fp.Height + l.block_height - 1u) / l.block_height);
    const uint64_t row_bytes = uint64_t{blocks_x} * elem_bytes;
    if (row_bytes > fp.RowPitch) {
        note("CopyTextureRegion: row bytes exceed footprint RowPitch");
        return;
    }
    const uint64_t slice_stride = uint64_t{rows_per_slice} * uint64_t{fp.RowPitch};
    const uint64_t rel_start = uint64_t{box.front} * slice_stride +
                               uint64_t{box.top / l.block_height} * fp.RowPitch +
                               uint64_t{box.left / l.block_width} * elem_bytes;
    const uint64_t buf_w = buf->desc().Width;
    if (src.PlacedFootprint.Offset > buf_w || rel_start > buf_w - src.PlacedFootprint.Offset) {
        note("CopyTextureRegion: footprint exceeds the source buffer");
        return;
    }
    const uint64_t start = src.PlacedFootprint.Offset + rel_start;
    const uint64_t len = uint64_t{d - 1u} * slice_stride + uint64_t{rows - 1u} * fp.RowPitch + row_bytes;
    if (len > buf_w - start || len > UINT32_MAX) {
        note("CopyTextureRegion: footprint exceeds the source buffer");
        return;
    }
    if (device_->upload_needs_conversion(tex->desc().Format)) {
        device_->note_unsupported_copy("CopyTextureRegion: DXGI format " + std::to_string(tex->desc().Format) +
                                       " needs CPU conversion for its WebGPU format (not implemented); skipped");
        return;
    }
    const uint32_t origin_z = sub_info.is_3d ? dst_z : sub_info.layer;
    pinned_.push_back(com::Ref<com::ComObject>::share(buf));
    pinned_.push_back(com::Ref<com::ComObject>::share(tex));
    DeferredWrite wr;
    wr.texture = true;
    wr.id = tex->id();
    wr.src = buf->cpu_address() + start;
    wr.size = len;
    wr.tex.texture_id = tex->id();
    wr.tex.mip_level = sub_info.mip;
    wr.tex.origin_x = dst_x;
    wr.tex.origin_y = dst_y;
    wr.tex.origin_z = origin_z;
    wr.tex.width = w;
    wr.tex.height = h;
    wr.tex.depth_or_layers = d;
    wr.tex.bytes_per_row = fp.RowPitch;
    wr.tex.rows_per_image = rows_per_slice;
    wr.widen_a8 = tex->desc().Format == static_cast<uint32_t>(pso::DxgiFormat::A8_UNORM);
    CommandListAux& aux = aux_of(this);
    if (aux.deferred_meta.size() < deferred_.size()) {
        aux.deferred_meta.resize(deferred_.size());
    }
    deferred_.push_back(wr);
    aux.deferred_meta.push_back(CommandListAux::DeferredMeta{});
    if (is_depth_dxgi_format(tex->desc().Format)) {
        aux.depth_twin.note_depth_copied(tex->id(), sub_info.mip, 1u, origin_z, d);
    }
    device_->note_texture_copy_destination(tex->id());
}

// ---- GPU -> CPU read-back (G66) ----

bool CommandList::readback_target(uint64_t va, uint64_t bytes, uint64_t* dst_addr, uint64_t* done_addr) {
    void* host = bytes == 0 ? nullptr : com::guest_to_host(va, static_cast<size_t>(bytes));
    if (host == nullptr) {
        note("read-back: destination " + hex(va) + " (+" + std::to_string(bytes) + ") is not guest memory");
        return false;
    }
    *dst_addr = reinterpret_cast<uintptr_t>(host);
    *done_addr = device_->readbacks().done_address();
    return true;
}

void CommandList::record_readback(GpuOpcode opcode, const void* payload, uint32_t payload_bytes, uint32_t resource_id,
                                  uint64_t va, uint64_t bytes) {
    const size_t at = bridge_.Readback(opcode, payload, payload_bytes, resource_id);
    if (at == SIZE_MAX) {
        return;
    }
    ReadbackRecord r;
    r.seq_offset = at;  // seq is the payload's first field
    r.guest_va = va;
    r.bytes = bytes;
    readbacks_.push_back(r);
}

void CommandList::issue_readbacks() {
    for (const ReadbackRecord& r : readbacks_) {
        if (!r.retile) {
            const uint32_t seq = device_->readbacks().issue(r.guest_va, r.bytes);
            encoder_.patch(r.seq_offset, &seq, sizeof(seq));
            continue;
        }
        // Backlog 8: the bytes land in a scratch buffer of this execution (pre-filled with a sentinel
        // so a read-back that never wrote is told apart); the completion re-tiles them into the guest
        // surface and marks it written. Nothing is written unless the GPU result arrived.
        auto scratch = std::make_shared<std::vector<uint8_t>>(static_cast<size_t>(r.retile_plan.linear_bytes));
        compute_readback::fill_sentinel(scratch->data(), scratch->size());
        const uint64_t dst = reinterpret_cast<uintptr_t>(scratch->data());
        encoder_.patch(r.seq_offset + offsetof(GpuReadbackTexturePayload, dst_addr), &dst, sizeof(dst));
        const compute_readback::Plan plan = r.retile_plan;
        const uint64_t cpu_va = r.retile_cpu_va;
        Device* const dev = device_.get();
        const uint32_t tex_id = r.retile_texture_id;
        const uint32_t seq = device_->readbacks().issue_with_completion(
            0, 0, ReadbackKind::Texture, 0u, [scratch, plan, cpu_va, dev, tex_id]() {
                void* host = com::guest_to_host(cpu_va, static_cast<size_t>(plan.surface_bytes));
                if (host != nullptr &&
                    compute_readback::retile_into_guest(plan, scratch->data(), scratch->size(),
                                                        static_cast<uint8_t*>(host),
                                                        static_cast<size_t>(plan.surface_bytes))) {
                    os::guest_window::mark_written(cpu_va, static_cast<size_t>(plan.surface_bytes));
                    // Host-originated write: remember its hash so it is not mistaken for a guest write.
                    dev->note_texture_readback_retiled(tex_id, cpu_va, static_cast<size_t>(plan.surface_bytes));
                    count_work(Work::ComputeTextureRetiled);
                } else {
                    count_work(Work::ComputeTextureRetileDiscarded);
                }
            });
        encoder_.patch(r.seq_offset, &seq, sizeof(seq));
    }
}

// Backlog 8: a compute pass wrote a float texture the title reads on the CPU (ocean height /
// displacement maps). The host GPU wrote its own texture, so read it back (async: the packet joins
// the list's read-back ring, the title is never blocked on a map) and re-tile the rows into the
// texture's console-layout guest memory when the read-back completes. Only small, tracked,
// guest-backed float 2D textures qualify (compute_texture_readback.h). The texture's guest memory
// then follows the GPU (the CPU->GPU upload sync stops, as for a copy destination), so the
// read-back result is not uploaded back over the next dispatch's output.
void CommandList::emit_compute_texture_readbacks() {
    if (compute_written_.empty()) {
        return;
    }
    constexpr size_t kMaxPerList = 4;
    constexpr uint64_t kMaxBytesPerList = 2ull << 20;
    std::vector<bridge::CommandListBridge::WrittenTexture> written;
    written.swap(compute_written_);
    size_t emitted = 0;
    uint64_t total = 0;
    for (const auto& w : written) {
        if (emitted >= kMaxPerList) {
            break;
        }
        TrackedTexture tt{};
        if (!device_->tracked_texture_info(w.resource_id, &tt) || tt.gpu_va == 0 ||
            tt.dimension != ResourceDimension::Texture2D || tt.sample_count > 1 || tt.host_floor != 0 ||
            tt.cap_floor_applied != 0 || device_->upload_needs_conversion(tt.dxgi_format)) {
            continue;
        }
        compute_readback::Target target;
        target.dxgi_format = tt.dxgi_format;
        target.width = tt.width;
        target.height = tt.height;
        target.array_size = std::max<uint32_t>(1u, tt.depth);
        target.mip_levels = tt.mip_levels;
        target.d3d12_layout = tt.layout;
        target.is_2d = true;
        const compute_readback::Plan plan = compute_readback::plan_readback(target, w.mip);
        if (!plan.ok || total + plan.linear_bytes > kMaxBytesPerList) {
            continue;
        }
        const uint64_t cpu_va = device_->cpu_address_for_gpu(tt.gpu_va);
        if (cpu_va == 0 || com::guest_to_host(cpu_va, static_cast<size_t>(plan.surface_bytes)) == nullptr) {
            continue;
        }
        GpuReadbackTexturePayload p{};
        p.texture_id = tt.resource_id;
        p.mip_level = plan.mip;
        p.width = plan.mip_width;
        p.height = plan.mip_height;
        p.depth_or_layers = 1;
        p.bytes_per_row = plan.row_bytes;
        p.rows_per_image = plan.mip_height;
        p.dst_size = plan.linear_bytes;
        p.done_addr = device_->readbacks().done_address();
        const size_t at = bridge_.Readback(GpuOpcode::ReadbackTexture, &p, sizeof(p), tt.resource_id);
        if (at == SIZE_MAX) {
            continue;
        }
        ReadbackRecord r;
        r.seq_offset = at;
        r.retile = true;
        r.retile_plan = plan;
        r.retile_cpu_va = cpu_va;
        r.retile_texture_id = tt.resource_id;
        readbacks_.push_back(r);
        device_->note_texture_readback_pending(tt.resource_id);
        count_work(Work::ComputeTextureReadbacks);
        total += plan.linear_bytes;
        ++emitted;
    }
}

// CopyTextureRegion with a texture subresource source and a placed-footprint destination in
// CPU-visible memory: the region is read back into the footprint's rows.
void CommandList::copy_texture_to_footprint(com::CallContext& ctx, const GuestTextureCopyLocation& dst,
                                            const GuestTextureCopyLocation& src, Resource* buf, Resource* tex) {
    if (buf->cpu_address() == 0 || !tex->has_gpu_texture()) {
        device_->note_unsupported_copy("CopyTextureRegion: texture -> footprint needs a CPU-visible buffer and a GPU texture");
        return;
    }
    const uint32_t sub = static_cast<uint32_t>(src.PlacedFootprint.Offset);  // SubresourceIndex
    const GuestSubresourceFootprint& fp = dst.PlacedFootprint.Footprint;
    const DxgiFormatLayout l = dxgi_format_layout(fp.Format);
    if (!l.known || fp.RowPitch == 0 || fp.Width == 0 || fp.Height == 0 || fp.Depth == 0) {
        note("CopyTextureRegion: unknown footprint format " + std::to_string(fp.Format));
        return;
    }
    SubresourceInfo sub_info{};
    if (!decode_texture_subresource(tex, sub, l, &sub_info)) {
        note("CopyTextureRegion: source subresource " + std::to_string(sub) + " out of range");
        return;
    }
    if (device_->upload_needs_conversion(tex->desc().Format)) {
        device_->note_unsupported_copy("CopyTextureRegion: DXGI format " + std::to_string(tex->desc().Format) +
                                       " has a converted WebGPU format; read-back skipped");
        return;
    }
    GuestBox box{0, 0, 0, sub_info.phys_w, sub_info.phys_h, sub_info.mip_d};
    if (ctx.int_arg(6) != 0 && !com::read_guest(ctx.int_arg(6), box)) {
        note("CopyTextureRegion: unreadable source box");
        return;
    }
    if (box.right <= box.left || box.bottom <= box.top || box.back <= box.front) {
        return;  // empty box is a valid no-op
    }
    if ((box.left % l.block_width) != 0 || (box.top % l.block_height) != 0) {
        note("CopyTextureRegion: read-back source box origin not aligned to format block size");
        return;
    }
    if (box.left >= sub_info.phys_w || box.top >= sub_info.phys_h || box.front >= sub_info.mip_d) {
        note("CopyTextureRegion: read-back source box origin exceeds subresource dimensions");
        return;
    }
    box.right = std::min(box.right, sub_info.phys_w);
    box.bottom = std::min(box.bottom, sub_info.phys_h);
    box.back = std::min(box.back, sub_info.mip_d);

    uint32_t w = align_up_u32(box.right - box.left, l.block_width);
    uint32_t h = align_up_u32(box.bottom - box.top, l.block_height);
    uint32_t d = box.back - box.front;
    w = std::min(w, sub_info.phys_w - box.left);
    h = std::min(h, sub_info.phys_h - box.top);
    d = std::min(d, sub_info.is_3d ? (sub_info.mip_d - box.front) : 1u);
    if (w == 0 || h == 0 || d == 0) {
        return;
    }
    const uint32_t dst_x = static_cast<uint32_t>(ctx.int_arg(2));
    const uint32_t dst_y = static_cast<uint32_t>(ctx.int_arg(3));
    const uint32_t dst_z = static_cast<uint32_t>(ctx.int_arg(4));
    if ((dst_x % l.block_width) != 0 || (dst_y % l.block_height) != 0) {
        note("CopyTextureRegion: read-back destination origin not aligned to format block size");
        return;
    }
    const uint32_t elem_bytes = l.planes == 2 ? plane_block_bytes(fp.Format, sub_info.plane) : l.block_bytes;
    const uint32_t rows = (h + l.block_height - 1u) / l.block_height;
    const uint32_t rows_per_slice = std::max(rows, (fp.Height + l.block_height - 1u) / l.block_height);
    const uint64_t row_bytes = uint64_t{(w + l.block_width - 1u) / l.block_width} * elem_bytes;
    const uint64_t slice_stride = uint64_t{rows_per_slice} * uint64_t{fp.RowPitch};
    const uint64_t rel_start = uint64_t{dst_z} * slice_stride +
                               uint64_t{dst_y / l.block_height} * fp.RowPitch +
                               uint64_t{dst_x / l.block_width} * elem_bytes;
    const uint64_t buf_w = buf->desc().Width;
    if (row_bytes > fp.RowPitch || dst.PlacedFootprint.Offset > buf_w ||
        rel_start > buf_w - dst.PlacedFootprint.Offset) {
        note("CopyTextureRegion: read-back region exceeds the destination footprint");
        return;
    }
    const uint64_t start = dst.PlacedFootprint.Offset + rel_start;
    const uint64_t len = uint64_t{d - 1u} * slice_stride + uint64_t{rows - 1u} * fp.RowPitch + row_bytes;
    if (len > buf_w - start) {
        note("CopyTextureRegion: read-back region exceeds the destination footprint");
        return;
    }
    const uint32_t origin_z = sub_info.is_3d ? box.front : sub_info.layer;
    if (is_depth_dxgi_format(tex->desc().Format)) {
        CommandListAux& aux = aux_of(this);
        for (const auto& pending : aux.depth_twin.pending_out(tex->id(), sub_info.mip, 1u, origin_z, d)) {
            aux.depth_twin.note_copied_out(tex->id(), pending.mip, pending.layer);
            ++aux.depth_srv_sync_flushes;
        }
    }
    pinned_.push_back(com::Ref<com::ComObject>::share(buf));
    pinned_.push_back(com::Ref<com::ComObject>::share(tex));
    GpuReadbackTexturePayload p{};
    p.texture_id = tex->id();
    p.mip_level = sub_info.mip;
    p.origin_x = box.left;
    p.origin_y = box.top;
    p.origin_z = origin_z;
    p.width = w;
    p.height = h;
    p.depth_or_layers = d;
    p.bytes_per_row = fp.RowPitch;
    p.rows_per_image = rows_per_slice;
    p.dst_size = len;
    const uint64_t va = buf->cpu_address() + start;
    if (readback_target(va, len, &p.dst_addr, &p.done_addr)) {
        record_readback(GpuOpcode::ReadbackTexture, &p, sizeof(p), tex->id(), va, len);
    }
}

// Xbox slot 85: (pDst, DstX, DstY, DstZ, pSrc, pSrcBox, flags) -> void. The title's texture upload
// path copies rows into a staging buffer and calls slot 85 with the CopyTextureRegion arguments
// followed by a trailing flags word. The flags are a 32-bit argument: the executable stores only
// the low half of the stack slot (movl), so the upper half holds whatever was there before.
void CommandList::CopyTextureRegionX(com::CallContext& ctx) {
    if (!usable()) {
        return;
    }
    const uint32_t flags = static_cast<uint32_t>(ctx.int_arg(7));
    if (flags != 0) {
        note("ID3D12GraphicsCommandList Xbox slot 85 (CopyTextureRegionX): trailing flags " + hex(flags) +
             " are not interpreted");
    }
    CopyTextureRegion(ctx);
}

void CommandList::ResolveSubresource(com::CallContext& ctx) {
    // (pDstResource, DstSubresource, pSrcResource, SrcSubresource, Format)
    if (!usable()) {
        return;
    }
    Resource* dst = as_resource(ctx.int_arg(1));
    const uint32_t dst_sub_idx = static_cast<uint32_t>(ctx.int_arg(2));
    Resource* src = as_resource(ctx.int_arg(3));
    const uint32_t src_sub_idx = static_cast<uint32_t>(ctx.int_arg(4));
    const uint32_t format = static_cast<uint32_t>(ctx.int_arg(5));
    va_watch_resource("resolve source (ResolveSubresource)", src,
                      "to resource id " + std::to_string(dst ? dst->id() : 0));
    va_watch_resource("resolve destination (ResolveSubresource)", dst,
                      "from resource id " + std::to_string(src ? src->id() : 0));
    if (dst == nullptr || src == nullptr || dst->is_buffer() || src->is_buffer() || !dst->has_gpu_texture() ||
        !src->has_gpu_texture()) {
        note("ResolveSubresource: invalid source or destination texture");
        return;
    }
    const DxgiFormatLayout fmt_layout = dxgi_format_layout(format);
    if (format == 0 || !fmt_layout.known || !formats_compatible_for_resolve(format, dst->desc().Format) ||
        !formats_compatible_for_resolve(format, src->desc().Format)) {
        note("ResolveSubresource: invalid or incompatible resolve format " + std::to_string(format));
        return;
    }
    SubresourceInfo dst_sub{}, src_sub{};
    if (!decode_texture_subresource(dst, dst_sub_idx, fmt_layout, &dst_sub) ||
        !decode_texture_subresource(src, src_sub_idx, fmt_layout, &src_sub)) {
        note("ResolveSubresource: subresource index out of range");
        return;
    }
    if (dst->desc().Dimension != ResourceDimension::Texture2D || src->desc().Dimension != ResourceDimension::Texture2D ||
        dst->desc().SampleCount > 1u || dst_sub.mip_w != src_sub.mip_w || dst_sub.mip_h != src_sub.mip_h) {
        note("ResolveSubresource: incompatible texture dimensions or sample counts");
        return;
    }

    pinned_.push_back(com::Ref<com::ComObject>::share(src));
    pinned_.push_back(com::Ref<com::ComObject>::share(dst));
    CommandListAux& aux = aux_of(this);
    const bool depth_resolve = is_depth_dxgi_format(format) || is_depth_dxgi_format(dst->desc().Format);
    if (depth_resolve) {
        aux.depth_twin.note_depth_copied(dst->id(), dst_sub.mip, 1u, dst_sub.layer, 1u);
    }
    device_->note_texture_copy_destination(dst->id());

    // Flush pending clears and close any active render pass before emitting the resolve pass.
    bridge_.ResourceBarrier(0, nullptr);

    const bool src_sampleable = (src->desc().Flags & ResourceFlags::DenyShaderResource) == 0u;
    const bool dst_attachable =
        depth_resolve ? ((dst->desc().Flags & ResourceFlags::AllowDepthStencil) != 0u)
                      : ((dst->desc().Flags & ResourceFlags::AllowRenderTarget) != 0u);
    if (src->desc().SampleCount > 1u && src_sub.mip == 0u && src_sampleable && dst_attachable &&
        device_->emit_msaa_resolve(dst->id(), dst_sub.mip, dst_sub.layer, src->id(), src_sub.layer, format, encoder_)) {
        ++aux.resolves_emitted;
        return;
    }
    ++aux.resolves_fallback;
    device_->note_unsupported_copy("ResolveSubresource: unsupported resolve configuration (safe fallback)");
}

void CommandList::ClearUnorderedAccessView(com::CallContext&) {
    if (usable()) {
        device_->note_unsupported_copy("ClearUnorderedAccessView: no UAV clear packet (skipped)");
    }
}

namespace {
// D3D12_QUERY_TYPE: 0 OCCLUSION, 1 BINARY_OCCLUSION, 2 TIMESTAMP, 3.. pipeline statistics / stream output.
void count_query_type(uint64_t type) {
    count_work(type <= 1u ? Work::QueryOcclusionOps : type == 2u ? Work::QueryTimestampOps : Work::QueryOtherOps);
}
}  // namespace

// Queries run on the consumer's query sets (gpu_queries(), kGpuOpcodeQuery / kGpuOpcodeResolveQuery):
// occlusion and timestamp heaps. Without one (no consumer support, pipeline / stream-output statistics)
// begin / end are only counted and a resolve writes nothing: the destination keeps its previous bytes,
// nothing is invented as a query result.
namespace {
const QueryHeap* query_heap_arg(uint64_t pointer) {
    auto arg = InterfaceArg<IQueryHeapHost>::resolve(pointer);
    return arg ? static_cast<const QueryHeap*>(arg.get()) : nullptr;
}

// The consumer query set a query of D3D12 `type` on `heap` runs on (0: none).
uint32_t query_set_for(const QueryHeap* heap, uint64_t type, uint64_t index, uint64_t count) {
    if (heap == nullptr || heap->query_set_id() == 0 || index > heap->desc().Count ||
        count > heap->desc().Count - index) {
        return 0;
    }
    const bool occlusion = type <= 1u && heap->query_set_kind() == GpuQuerySetKind::Occlusion;
    const bool timestamp = type == 2u && heap->query_set_kind() == GpuQuerySetKind::Timestamp;
    return occlusion || timestamp ? heap->query_set_id() : 0u;
}
}  // namespace

// The D3D12_QUERY_TYPE and UINT arguments are 32-bit: the upper half of their registers is not theirs.
static uint32_t u32_arg(com::CallContext& ctx, int i) {
    return static_cast<uint32_t>(ctx.int_arg(i));
}

void CommandList::emit_query(com::CallContext& ctx, uint32_t op) {
    // this, heap, type, index
    const QueryHeap* heap = query_heap_arg(ctx.int_arg(1));
    const uint32_t type = u32_arg(ctx, 2);
    const uint32_t set = query_set_for(heap, type, u32_arg(ctx, 3), 1);
    if (!usable() || set == 0 || (op == 0 && type == 2u)) {
        return;  // a timestamp only ends (BeginQuery on one is invalid D3D12)
    }
    pinned_.push_back(com::Ref<com::ComObject>::share(const_cast<QueryHeap*>(heap)));
    GpuQueryPayload p{};
    p.set_id = set;
    p.index = u32_arg(ctx, 3);
    p.op = op;
    p.type = type;
    bridge_.QueryPacket(kGpuOpcodeQuery, &p, sizeof(p), 0, false);
    count_work(Work::QueryPackets);
}

void CommandList::BeginQuery(com::CallContext& ctx) {
    count_work(Work::QueryBegins);
    count_query_type(u32_arg(ctx, 2));
    emit_query(ctx, 0);
}

void CommandList::EndQuery(com::CallContext& ctx) {
    count_work(Work::QueryEnds);
    count_query_type(u32_arg(ctx, 2));
    emit_query(ctx, 1);
}

void CommandList::ResolveQueryData(com::CallContext& ctx) {
    // this, heap, type, start index, count, destination buffer, destination offset.
    count_work(Work::QueryResolves);
    const uint32_t type = u32_arg(ctx, 2);
    const uint64_t first = u32_arg(ctx, 3);
    const uint64_t count = u32_arg(ctx, 4);
    const uint64_t offset = ctx.int_arg(6);
    count_query_type(type);
    if (!usable()) {
        return;
    }
    const QueryHeap* heap = query_heap_arg(ctx.int_arg(1));
    Resource* dst = resource(InterfaceArg<IResourceHost>::resolve(ctx.int_arg(5)));
    const uint32_t set = query_set_for(heap, type, first, count);
    const bool dst_ok = dst != nullptr && dst->is_buffer() && (offset & 7u) == 0 && count <= (1u << 12) &&
                        offset <= dst->desc().Width && count * 8 <= dst->desc().Width - offset;
    if (set == 0 || !dst_ok || count == 0) {
        count_work(Work::QueryResultsUnwritten, static_cast<uint32_t>(count));
        device_->note_bad_argument(std::string("ResolveQueryData: ") +
                                   (set == 0 ? "no GPU query set for type " + std::to_string(type)
                                             : std::string("invalid destination"))  +
                                   ", results not written (destination keeps its previous bytes)");
        return;
    }
    pinned_.push_back(com::Ref<com::ComObject>::share(const_cast<QueryHeap*>(heap)));
    pinned_.push_back(com::Ref<com::ComObject>::share(dst));
    GpuResolveQueryPayload p{};
    p.set_id = set;
    p.type = type;
    p.first = static_cast<uint32_t>(first);
    p.count = static_cast<uint32_t>(count);
    p.buffer_id = dst->id();
    p.dst_offset = offset;
    bridge_.QueryPacket(kGpuOpcodeResolveQuery, &p, sizeof(p), dst->id(), true);
    count_work(Work::QueryPackets);
    count_work(Work::QueryResultsWritten, static_cast<uint32_t>(count));
    if (dst->cpu_address() != 0) {
        // A CPU-visible destination (a readback heap): the resolved bytes also go back to its memory.
        GpuReadbackBufferPayload rb{};
        rb.buffer_id = dst->id();
        rb.src_offset = offset;
        rb.size = count * 8;
        if (readback_target(dst->cpu_address() + offset, count * 8, &rb.dst_addr, &rb.done_addr)) {
            record_readback(GpuOpcode::ReadbackBuffer, &rb, sizeof(rb), dst->id(), dst->cpu_address() + offset,
                            count * 8);
        }
    }
}

void CommandList::append_submission(CommandEncoder& out, std::vector<d3d::StateDiagnostic>* state_diagnostics) {
    // Root tables over descriptor memory: the GPU reads the slots when the list executes, so they are
    // read again now and the recorded bind groups are rewritten where the title changed them.
    bridge_.refresh_memory_bind_groups(out);
    CommandListAux& aux = aux_of(this);
    for (size_t idx = 0; idx < deferred_.size(); ++idx) {
        const DeferredWrite& w = deferred_[idx];
        const CommandListAux::DeferredMeta meta =
            idx < aux.deferred_meta.size() ? aux.deferred_meta[idx] : CommandListAux::DeferredMeta{};
        std::vector<uint8_t> bytes(static_cast<size_t>(w.size));
        if (!com::read_guest(w.src, bytes.data(), bytes.size())) {
            device_->note_bad_argument("ExecuteCommandLists: copy source " + hex(w.src) + " no longer readable");
            continue;
        }
        uint32_t host_mip = w.texture ? w.tex.mip_level : 0u;
        if (w.texture && !device_->state_bridge().translate_host_mip(w.tex.texture_id, &host_mip)) {
            continue;  // a level the capped host texture lacks (gpu/texture_budget.h)
        }
        if (w.texture && w.widen_a8) {
            count_work(Work::TextureUploads);
            std::vector<uint8_t> wide;
            uint32_t pitch = 0;
            if (widen_a8_rows(bytes.data(), bytes.size(), w.tex.width, w.tex.height, w.tex.depth_or_layers,
                              w.tex.bytes_per_row, uint64_t{w.tex.rows_per_image} * w.tex.bytes_per_row, &wide, &pitch)) {
                out.write_texture(w.tex.texture_id, host_mip, w.tex.origin_x, w.tex.origin_y, w.tex.origin_z,
                                  w.tex.width, w.tex.height, w.tex.depth_or_layers, pitch, w.tex.height, wide.data(),
                                  static_cast<uint32_t>(wide.size()));
            }
        } else if (w.texture) {
            count_work(Work::TextureUploads);
            out.write_texture(w.tex.texture_id, host_mip, w.tex.origin_x, w.tex.origin_y, w.tex.origin_z,
                              w.tex.width, w.tex.height, w.tex.depth_or_layers, w.tex.bytes_per_row,
                              w.tex.rows_per_image, bytes.data(), static_cast<uint32_t>(bytes.size()));
        } else {
            count_work(Work::BufferUploads);
            uint64_t write_offset = w.dst_offset;
            const uint8_t* write_ptr = bytes.data();
            uint64_t write_size = w.size;
            std::vector<uint8_t> aligned;
            if (meta.needs_rmw_align || (write_offset & 3u) != 0u || (write_size & 3u) != 0u) {
                const uint64_t aligned_start = write_offset & ~uint64_t{3};
                const uint64_t aligned_end = align_up_u64(write_offset + write_size, 4u);
                const uint64_t aligned_len = aligned_end - aligned_start;
                if (aligned_end > UINT32_MAX || (meta.dst_allocated != 0 && aligned_end > meta.dst_allocated)) {
                    device_->note_bad_argument("ExecuteCommandLists: aligned CopyBufferRegion exceeds destination buffer");
                    continue;
                }
                aligned.assign(static_cast<size_t>(aligned_len), 0u);
                if (meta.dst_cpu_va != 0) {
                    const uint64_t readable =
                        aligned_start < meta.dst_width ? std::min(aligned_len, meta.dst_width - aligned_start) : 0u;
                    if (readable != 0) {
                        com::read_guest(meta.dst_cpu_va + aligned_start, aligned.data(), static_cast<size_t>(readable));
                    }
                }
                std::memcpy(aligned.data() + static_cast<size_t>(write_offset - aligned_start), bytes.data(),
                            bytes.size());
                write_offset = aligned_start;
                write_ptr = aligned.data();
                write_size = aligned_len;
            }
            for (uint64_t off = 0; off < write_size; off += kMaxWriteChunk) {
                const uint64_t n = std::min(kMaxWriteChunk, write_size - off);
                out.write_buffer(w.id, static_cast<uint32_t>(write_offset + off), write_ptr + off,
                                 static_cast<uint32_t>(n));
            }
        }
        device_->note_upload(w.size);
    }
    // Rectangle-list expansions: the source vertices as they are now, three per rectangle, become
    // two triangles (v0 v1 v2, v2 v3 v0) with v3 = v0 + v2 - v1 per per-vertex attribute.
    for (const RectExpansion& x : rect_expansions_) {
        std::vector<uint8_t> in(static_cast<size_t>(uint64_t{x.rects} * 3 * x.stride));
        if (!com::read_guest(x.src, in.data(), in.size())) {
            device_->note_bad_argument("ExecuteCommandLists: rectangle-list vertices at " + hex(x.src) +
                                       " no longer readable");
            continue;
        }
        std::vector<uint8_t> out_bytes(static_cast<size_t>(uint64_t{x.rects} * 6 * x.stride));
        for (uint32_t r = 0; r < x.rects; ++r) {
            const uint8_t* v0 = in.data() + size_t{r} * 3 * x.stride;
            const uint8_t* v1 = v0 + x.stride;
            const uint8_t* v2 = v1 + x.stride;
            uint8_t* o = out_bytes.data() + size_t{r} * 6 * x.stride;
            std::vector<uint8_t> v3(v0, v0 + x.stride);
            for (const auto& [offset, fmt] : x.elements) {
                RectFormat rf{};
                rect_format(fmt, &rf);
                const uint32_t cb = rect_comp_bytes(rf.comp);
                for (uint32_t c = 0; c < rf.count; ++c) {
                    const uint32_t at = offset + c * cb;
                    rect_corner_component(rf.comp, v0 + at, v1 + at, v2 + at, v3.data() + at);
                }
            }
            for (const uint8_t* v : {v0, v1, v2, v2, static_cast<const uint8_t*>(v3.data()), v0}) {
                std::memcpy(o, v, x.stride);
                o += x.stride;
            }
        }
        for (uint64_t off = 0; off < out_bytes.size(); off += kMaxWriteChunk) {
            const uint64_t n = std::min<uint64_t>(kMaxWriteChunk, out_bytes.size() - off);
            out.write_buffer(x.buffer_id, static_cast<uint32_t>(off), out_bytes.data() + off, static_cast<uint32_t>(n));
        }
        device_->note_upload(out_bytes.size());
    }
    device_->resource_states().execute(bridge_.resource_states(), state_diagnostics);
}

// ---- Fixed-function state ----

void CommandList::IASetPrimitiveTopology(uint32_t topology) {
    if (usable()) {
        rect_list_ = topology == xbox::kTopologyRectList;
        bridge_.IASetPrimitiveTopology(static_cast<pso::D3DPrimitiveTopology>(topology));
    }
}

void CommandList::RSSetViewports(uint32_t count, uint64_t viewports) {
    std::vector<pso::D3D12Viewport> v;
    static_assert(sizeof(pso::D3D12Viewport) == sizeof(GuestViewport));
    if (!usable()) {
        return;
    }
    if (count > kMaxViewports || !read_n(viewports, count, &v)) {
        note("RSSetViewports: unreadable viewports");
        return;
    }
    CommandListAux& aux = aux_of(this);
    aux.raw_viewports = v;
    aux.clamped_viewports.resize(count);
    const uint32_t max_dim = device_->state_bridge().options().max_texture_dimension_2d;
    for (uint32_t i = 0; i < count; ++i) {
        aux.clamped_viewports[i] = clamp_viewport_webgpu(v[i], aux.rt_width, aux.rt_height, max_dim);
    }
    bridge_.RSSetViewports(count, aux.clamped_viewports.empty() ? nullptr : aux.clamped_viewports.data());
}

void CommandList::RSSetScissorRects(uint32_t count, uint64_t rects) {
    std::vector<pso::D3D12Rect> r;
    static_assert(sizeof(pso::D3D12Rect) == sizeof(GuestRect));
    if (!usable()) {
        return;
    }
    if (count > kMaxViewports || !read_n(rects, count, &r)) {
        note("RSSetScissorRects: unreadable rects");
        return;
    }
    CommandListAux& aux = aux_of(this);
    aux.raw_scissors = r;
    aux.clamped_scissors.resize(count);
    const uint32_t max_dim = device_->state_bridge().options().max_texture_dimension_2d;
    for (uint32_t i = 0; i < count; ++i) {
        aux.clamped_scissors[i] = clamp_scissor_webgpu(r[i], aux.rt_width, aux.rt_height, max_dim);
    }
    bridge_.RSSetScissorRects(count, aux.clamped_scissors.empty() ? nullptr : aux.clamped_scissors.data());
}

void CommandList::OMSetBlendFactor(uint64_t factor) {
    if (!usable()) {
        return;
    }
    float f[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    if (factor != 0 && !com::read_guest(factor, f, sizeof(f))) {
        note("OMSetBlendFactor: unreadable factor");
        return;
    }
    for (float& c : f) {
        c = std::isfinite(c) ? normalize_zero(c) : 0.0f;
    }
    CommandListAux& aux = aux_of(this);
    std::memcpy(aux.blend_factor, f, sizeof(f));
    bridge_.OMSetBlendFactor(f);
}

void CommandList::OMSetStencilRef(uint32_t ref) {
    if (usable()) {
        const uint32_t masked = ref & 0xFFu;
        aux_of(this).stencil_ref = masked;
        bridge_.OMSetStencilRef(masked);
    }
}

void CommandList::WriteValue32(uint64_t address, uint32_t value, uint32_t a3, uint32_t a4) {
    if (!usable()) {
        return;
    }
    if (address == 0 || (address & 3) != 0) {
        note("ID3D12GraphicsCommandList Xbox slot 71: invalid address " + hex(address));
        return;
    }
    if (a3 != 0 || a4 != 0) {
        note("ID3D12GraphicsCommandList Xbox slot 71: arguments " + hex(a3) + ", " + hex(a4) + " are not interpreted");
    }
    immediate_writes_.push_back(ImmediateWrite{address, value, 4});
}

void CommandList::WriteValue64(uint64_t address, uint32_t a2, uint64_t value, uint32_t a4) {
    if (!usable()) {
        return;
    }
    if (address == 0 || (address & 7) != 0) {
        note("ID3D12GraphicsCommandList Xbox slot 74: invalid address " + hex(address));
        return;
    }
    if (a2 != 7 || a4 != 1) {
        note("ID3D12GraphicsCommandList Xbox slot 74: arguments " + hex(a2) + ", " + hex(a4) +
             " differ from the observed (7, 1); not interpreted");
    }
    immediate_writes_.push_back(ImmediateWrite{address, value, 8});
}

void CommandList::apply_immediate_writes() {
    for (const ImmediateWrite& w : immediate_writes_) {
        // The address is a GPU VA: inside an alias window it names the CPU window's bytes (a
        // phantom GPU window has none of its own; O29: the CPU window may be translated).
        if (!com::write_guest(device_->cpu_address_for_gpu(w.address), &w.value, w.size)) {
            device_->note_bad_argument("ExecuteCommandLists: immediate write to " + hex(w.address) +
                                       " is not guest memory");
        }
    }
}

namespace {
// D3D12_BARRIER_GROUP {D3D12_BARRIER_TYPE Type; UINT32 NumBarriers; const void* pBarriers}.
struct GuestBarrierGroup {
    uint32_t Type;
    uint32_t NumBarriers;
    uint64_t pBarriers;
};
static_assert(sizeof(GuestBarrierGroup) == 16);
// Element sizes of the three barrier types (global, texture, buffer) and the offset of their
// resource pointer (none for global barriers).
constexpr uint32_t kBarrierBytes[3] = {16, 64, 40};
constexpr uint32_t kBarrierResourceOffset[3] = {0, 24, 16};
constexpr uint32_t kMaxBarrierGroups = 4096;
constexpr uint32_t kMaxBarriersPerGroup = 65536;
}  // namespace

void CommandList::Barrier(uint32_t group_count, uint64_t groups) {
    if (!usable()) {
        return;
    }
    std::vector<GuestBarrierGroup> g;
    if (group_count > kMaxBarrierGroups || !read_n(groups, group_count, &g)) {
        note("Barrier (Xbox slot 146): unreadable barrier group array " + hex(groups));
        return;
    }
    // Enhanced barriers order GPU work and change layouts; WebGPU orders its own work and has no
    // layouts, so nothing is recorded. The groups are still checked: the resources they name must
    // be resources of this device.
    for (const GuestBarrierGroup& group : g) {
        if (group.Type > 2 || group.NumBarriers > kMaxBarriersPerGroup ||
            (group.NumBarriers != 0 && group.pBarriers == 0)) {
            note("Barrier (Xbox slot 146): invalid barrier group (type " + std::to_string(group.Type) + ", " +
                 std::to_string(group.NumBarriers) + " barriers)");
            continue;
        }
        if (group.Type == 0) {
            continue;
        }
        for (uint32_t i = 0; i < group.NumBarriers; ++i) {
            uint64_t resource_va = 0;
            const uint64_t at = group.pBarriers + uint64_t{i} * kBarrierBytes[group.Type] +
                                kBarrierResourceOffset[group.Type];
            if (!com::read_guest(at, resource_va) || as_resource(resource_va) == nullptr) {
                note("Barrier (Xbox slot 146): barrier " + std::to_string(i) + " names an unknown resource " +
                     hex(resource_va));
            }
        }
    }
}

void CommandList::ResourceBarrier(uint32_t count, uint64_t barriers) {
    if (!usable()) {
        return;
    }
    std::vector<GuestResourceBarrier> g;
    if (count > kMaxBarriers || !read_n(barriers, count, &g)) {
        note("ResourceBarrier: unreadable barrier array");
        return;
    }
    std::vector<d3d::ResourceBarrier> b;
    b.reserve(count);
    auto id_of = [&](uint64_t va) -> uint32_t {
        Resource* r = as_resource(va);
        return r != nullptr ? r->id() : 0u;
    };
    for (const GuestResourceBarrier& x : g) {
        d3d::ResourceBarrier r;
        r.type = static_cast<d3d::ResourceBarrierType>(x.Type);
        r.flags = x.Flags;
        switch (x.Type) {
        case 0:
            r.resource_id = id_of(x.pResource);
            r.subresource = x.u0;
            r.state_before = x.u1;
            r.state_after = x.StateAfter;
            if (r.resource_id == 0) {
                note("ResourceBarrier: transition of an unknown resource " + hex(x.pResource));
                continue;
            }
            break;
        case 1:
            r.resource_before = id_of(x.pResource);
            r.resource_after = id_of(uint64_t{x.u0} | (uint64_t{x.u1} << 32));
            break;
        case 2:
            r.resource_id = id_of(x.pResource);
            break;
        default:
            note("ResourceBarrier: invalid type " + std::to_string(x.Type));
            continue;
        }
        b.push_back(r);
    }
    bridge_.ResourceBarrier(static_cast<uint32_t>(b.size()), b.data());
}

void CommandList::SetDescriptorHeaps(uint32_t count, uint64_t heaps) {
    if (!usable()) {
        return;
    }
    std::vector<uint64_t> ptrs;
    if (count > 2 || !read_n(heaps, count, &ptrs)) {
        note("SetDescriptorHeaps: unreadable heap array or more than 2 heaps");
        return;
    }
    std::vector<d3d::DescriptorHeap*> list;
    for (uint64_t p : ptrs) {
        auto arg = InterfaceArg<IDescriptorHeapHost>::resolve(p);
        if (!arg) {
            note("SetDescriptorHeaps: invalid heap " + hex(p));
            return;
        }
        pinned_.push_back(com::Ref<com::ComObject>::share(arg.object()));
        list.push_back(static_cast<DescriptorHeap*>(arg.get())->heap());
    }
    const d3d::Status s = bridge_.SetDescriptorHeaps(static_cast<uint32_t>(list.size()), list.data());
    if (s != d3d::Status::Ok) {
        note(std::string("SetDescriptorHeaps: ") + d3d::status_name(s));
    }
}

namespace {
const d3d::RootSignature* signature_of(const InterfaceArg<IRootSignatureHost>& arg) {
    return arg ? &static_cast<RootSignature*>(arg.get())->signature() : nullptr;
}
}  // namespace

void CommandList::SetComputeRootSignature(InterfaceArg<IRootSignatureHost> signature) {
    if (!usable()) {
        return;
    }
    if (signature) {
        pinned_.push_back(com::Ref<com::ComObject>::share(signature.object()));
    }
    const d3d::RootSignature* sig = signature_of(signature);
    aux_of(this).compute_root_signature = sig;
    const d3d::Status s = bridge_.SetComputeRootSignature(sig);
    if (s != d3d::Status::Ok) {
        note(std::string("SetComputeRootSignature: ") + d3d::status_name(s));
    }
}

void CommandList::SetGraphicsRootSignature(InterfaceArg<IRootSignatureHost> signature) {
    if (!usable()) {
        return;
    }
    if (signature) {
        pinned_.push_back(com::Ref<com::ComObject>::share(signature.object()));
    }
    graphics_root_signature_ = signature_of(signature);
    const d3d::Status s = bridge_.SetGraphicsRootSignature(graphics_root_signature_);
    if (s != d3d::Status::Ok) {
        note(std::string("SetGraphicsRootSignature: ") + d3d::status_name(s));
    }
}

#define PG6_ROOT_CALL(expr, what)                                       \
    do {                                                                \
        if (!usable()) {                                                \
            return;                                                     \
        }                                                               \
        const d3d::Status s_ = (expr);                                  \
        if (s_ != d3d::Status::Ok) {                                    \
            note(std::string(what ": ") + d3d::status_name(s_));        \
        }                                                               \
    } while (0)

void CommandList::SetComputeRootDescriptorTable(uint32_t index, uint64_t base) {
    CommandListAux& aux = aux_of(this);
    if (index < aux.compute_root_args.size()) {
        aux.compute_root_args[index] = {1, base};
    }
    PG6_ROOT_CALL(bridge_.SetComputeRootDescriptorTable(index, d3d::GpuDescriptorHandle{base}), "SetComputeRootDescriptorTable");
}

void CommandList::SetGraphicsRootDescriptorTable(uint32_t index, uint64_t base) {
    record_root_arg(index, 1, base, 0);
    PG6_ROOT_CALL(bridge_.SetGraphicsRootDescriptorTable(index, d3d::GpuDescriptorHandle{base}), "SetGraphicsRootDescriptorTable");
}

void CommandList::SetComputeRoot32BitConstant(uint32_t index, uint32_t value, uint32_t offset) {
    PG6_ROOT_CALL(bridge_.SetComputeRoot32BitConstants(index, 1, &value, offset), "SetComputeRoot32BitConstant");
}

void CommandList::SetGraphicsRoot32BitConstant(uint32_t index, uint32_t value, uint32_t offset) {
    record_root_arg(index, 2, 0, offset + 1);
    PG6_ROOT_CALL(bridge_.SetGraphicsRoot32BitConstants(index, 1, &value, offset), "SetGraphicsRoot32BitConstant");
}

void CommandList::SetComputeRoot32BitConstants(uint32_t index, uint32_t count, uint64_t values, uint32_t offset) {
    std::vector<uint32_t> v;
    if (count > d3d::kMaxRootSignatureDwords || !read_n(values, count, &v)) {
        note("SetComputeRoot32BitConstants: unreadable values");
        return;
    }
    PG6_ROOT_CALL(bridge_.SetComputeRoot32BitConstants(index, count, v.data(), offset), "SetComputeRoot32BitConstants");
}

void CommandList::SetGraphicsRoot32BitConstants(uint32_t index, uint32_t count, uint64_t values, uint32_t offset) {
    std::vector<uint32_t> v;
    if (count > d3d::kMaxRootSignatureDwords || !read_n(values, count, &v)) {
        note("SetGraphicsRoot32BitConstants: unreadable values");
        return;
    }
    record_root_arg(index, 2, 0, offset + count);
    PG6_ROOT_CALL(bridge_.SetGraphicsRoot32BitConstants(index, count, v.data(), offset), "SetGraphicsRoot32BitConstants");
}

void CommandList::SetComputeRootConstantBufferView(uint32_t index, uint64_t va) {
    va_watch::touch("root CBV (compute)", va, 1, "root parameter " + std::to_string(index));
    PG6_ROOT_CALL(bridge_.SetComputeRootConstantBufferView(index, va), "SetComputeRootConstantBufferView");
}

void CommandList::SetGraphicsRootConstantBufferView(uint32_t index, uint64_t va) {
    va_watch::touch("root CBV (graphics)", va, 1, "root parameter " + std::to_string(index));
    record_root_arg(index, 3, va, 0);
    PG6_ROOT_CALL(bridge_.SetGraphicsRootConstantBufferView(index, va), "SetGraphicsRootConstantBufferView");
}

void CommandList::SetComputeRootShaderResourceView(uint32_t index, uint64_t va) {
    va_watch::touch("root SRV (compute)", va, 1, "root parameter " + std::to_string(index));
    if (usable()) {
        note("SetComputeRootShaderResourceView: compute root SRVs are not supported by the state bridge (ignored)");
    }
}

void CommandList::SetGraphicsRootShaderResourceView(uint32_t index, uint64_t va) {
    va_watch::touch("root SRV (graphics)", va, 1, "root parameter " + std::to_string(index));
    record_root_arg(index, 4, va, 0);
    PG6_ROOT_CALL(bridge_.SetGraphicsRootShaderResourceView(index, va), "SetGraphicsRootShaderResourceView");
}

void CommandList::SetComputeRootUnorderedAccessView(uint32_t index, uint64_t va) {
    va_watch::touch("root UAV (compute)", va, 1, "root parameter " + std::to_string(index));
    if (usable()) {
        note("SetComputeRootUnorderedAccessView: compute root UAVs are not supported by the state bridge (ignored)");
    }
}

void CommandList::SetGraphicsRootUnorderedAccessView(uint32_t index, uint64_t va) {
    va_watch::touch("root UAV (graphics)", va, 1, "root parameter " + std::to_string(index));
    record_root_arg(index, 5, va, 0);
    PG6_ROOT_CALL(bridge_.SetGraphicsRootUnorderedAccessView(index, va), "SetGraphicsRootUnorderedAccessView");
}

#undef PG6_ROOT_CALL

// ---- Input assembler ----

void CommandList::IASetIndexBuffer(uint64_t view) {
    if (!usable()) {
        return;
    }
    if (view == 0) {
        bridge_.IASetIndexBuffer(nullptr);
        return;
    }
    GuestIndexBufferView g{};
    if (!com::read_guest(view, g)) {
        note("IASetIndexBuffer: unreadable view");
        return;
    }
    index_buffer_ = VertexBufferRecord{g.BufferLocation, g.SizeInBytes, g.Format};
    d3d::GpuVaResolution res;
    if (!device_->state_bridge().resolve_gpu_va(g.BufferLocation, &res)) {
        // Address-only guest memory (G58: the title's dynamic ring buffers): a runtime-owned mirror.
        uint32_t buf_id = 0;
        uint64_t buf_off = 0;
        if (g.BufferLocation == 0 || g.SizeInBytes == 0 ||
            !device_->relocate_ia_buffer(g.BufferLocation, g.SizeInBytes, &buf_id, &buf_off)) {
            note("IASetIndexBuffer: unknown GPU virtual address " + hex(g.BufferLocation));
            bridge_.IASetIndexBuffer(nullptr);
            return;
        }
        res.resource_id = buf_id;
        res.offset = buf_off;
    }
    if (g.Format != 57 && g.Format != 42) {
        note("IASetIndexBuffer: invalid index format " + std::to_string(g.Format));
        return;
    }
    d3d12_xs::D3D12IndexBufferView v;
    v.buffer_id = res.resource_id;
    v.offset = res.offset;
    v.size_in_bytes = g.SizeInBytes;
    v.format = g.Format == 57 ? GpuIndexFormat::Uint16 : GpuIndexFormat::Uint32;
    bridge_.IASetIndexBuffer(&v);
}

void CommandList::IASetVertexBuffers(uint32_t start, uint32_t count, uint64_t views) {
    if (!usable()) {
        return;
    }
    if (start >= pso::kD3D12IaVertexInputSlotCount || count > pso::kD3D12IaVertexInputSlotCount - start) {
        note("IASetVertexBuffers: slot range out of bounds");
        return;
    }
    if (views == 0) {
        for (uint32_t i = 0; i < count; ++i) {
            vertex_buffers_[start + i] = VertexBufferRecord{};
            bridge_vb_views_[start + i] = d3d12_xs::D3D12VertexBufferView{};
        }
        bridge_.IASetVertexBuffers(start, count, nullptr);
        return;
    }
    std::vector<GuestVertexBufferView> g;
    if (!read_n(views, count, &g)) {
        note("IASetVertexBuffers: unreadable views");
        return;
    }
    std::vector<d3d12_xs::D3D12VertexBufferView> v(count);
    for (uint32_t i = 0; i < count; ++i) {
        vertex_buffers_[start + i] = VertexBufferRecord{g[i].BufferLocation, g[i].SizeInBytes, g[i].StrideInBytes};
        d3d::GpuVaResolution res;
        if (g[i].BufferLocation != 0 && device_->state_bridge().resolve_gpu_va(g[i].BufferLocation, &res)) {
            v[i].buffer_id = res.resource_id;
            v[i].offset = res.offset;
        } else if (g[i].BufferLocation != 0) {
            // Address-only guest memory (G58: the title's dynamic ring buffers): a runtime-owned mirror.
            uint32_t buf_id = 0;
            uint64_t buf_off = 0;
            if (g[i].SizeInBytes != 0 &&
                device_->relocate_ia_buffer(g[i].BufferLocation, g[i].SizeInBytes, &buf_id, &buf_off)) {
                v[i].buffer_id = buf_id;
                v[i].offset = buf_off;
            } else {
                note("IASetVertexBuffers: unknown GPU virtual address " + hex(g[i].BufferLocation));
            }
        }
        v[i].size_in_bytes = g[i].SizeInBytes;
        v[i].stride_in_bytes = g[i].StrideInBytes;
        bridge_vb_views_[start + i] = v[i];
    }
    bridge_.IASetVertexBuffers(start, count, v.data());
}

void CommandList::SOSetTargets(uint32_t, uint32_t count, uint64_t) {
    if (usable() && count != 0) {
        note("SOSetTargets: stream output is not supported (ignored)");
    }
}

// ---- Output merger ----

void CommandList::OMSetRenderTargets(uint32_t count, uint64_t rtvs, bool single_range, uint64_t dsv) {
    if (!usable()) {
        return;
    }
    if (count > 8) {
        note("OMSetRenderTargets: more than 8 render targets");
        return;
    }
    std::vector<d3d::CpuDescriptorHandle> handles;
    if (!read_n(rtvs, count == 0 ? 0u : (single_range ? 1u : count), &handles)) {
        note("OMSetRenderTargets: unreadable render target handles");
        return;
    }
    d3d::CpuDescriptorHandle depth{};
    if (dsv != 0 && !com::read_guest(dsv, depth)) {
        note("OMSetRenderTargets: unreadable depth-stencil handle");
        return;
    }
    rtv_handles_.clear();
    for (uint32_t i = 0; i < count; ++i) {
        rtv_handles_.push_back(single_range ? handles[0].ptr + uint64_t{i} * d3d::descriptor_handle_increment_size(
                                                                    d3d::DescriptorHeapType::Rtv)
                                            : handles[i].ptr);
    }
    dsv_handle_ = dsv != 0 ? depth.ptr : 0;

    // Update bound RT/DSV extents and active DSV metadata for viewport/scissor clamping and depth-twin sync.
    CommandListAux& aux = aux_of(this);
    uint32_t min_w = UINT32_MAX;
    uint32_t min_h = UINT32_MAX;
    auto accumulate_handle = [&](uint64_t cpu_handle) -> const d3d::ViewDescriptor* {
        if (cpu_handle == 0) {
            return nullptr;
        }
        d3d::ResolvedDescriptor res{};
        if (device_->heaps().resolve_cpu(d3d::CpuDescriptorHandle{cpu_handle}, &res) != d3d::Status::Ok ||
            res.heap == nullptr) {
            return nullptr;
        }
        const d3d::ViewDescriptor* v = res.heap->view(res.index);
        if (v == nullptr || v->resource_id == 0) {
            return nullptr;
        }
        bridge::TextureInfo ti{};
        if (device_->state_bridge().texture_info(v->resource_id, &ti)) {
            const uint32_t mw = std::max<uint32_t>(1u, ti.width >> v->first_mip);
            const uint32_t mh = std::max<uint32_t>(1u, ti.height >> v->first_mip);
            min_w = std::min(min_w, mw);
            min_h = std::min(min_h, mh);
        }
        return v;
    };
    for (uint64_t h : rtv_handles_) {
        accumulate_handle(h);
    }
    aux.bound_dsv = {};
    if (const d3d::ViewDescriptor* dv = accumulate_handle(dsv_handle_); dv != nullptr) {
        aux.bound_dsv.resource_id = dv->resource_id;
        aux.bound_dsv.mip = dv->first_mip;
        aux.bound_dsv.base_layer = dv->first_array_slice;
        aux.bound_dsv.layer_count = dv->array_size != 0 ? dv->array_size : 1u;
        aux.bound_dsv.depth_read_only = (dv->flags & 0x1u) != 0u;  // D3D12_DSV_FLAG_READ_ONLY_DEPTH
    }
    aux.rt_width = (min_w != UINT32_MAX) ? min_w : 0u;
    aux.rt_height = (min_h != UINT32_MAX) ? min_h : 0u;

    bridge_.OMSetRenderTargets(count, handles.empty() ? nullptr : handles.data(), single_range,
                               dsv != 0 ? &depth : nullptr);

    // Re-clamp any viewports/scissors set prior to OMSetRenderTargets against the new RT extents.
    const uint32_t max_dim = device_->state_bridge().options().max_texture_dimension_2d;
    if (!aux.raw_viewports.empty()) {
        aux.clamped_viewports.resize(aux.raw_viewports.size());
        for (size_t i = 0; i < aux.raw_viewports.size(); ++i) {
            aux.clamped_viewports[i] =
                clamp_viewport_webgpu(aux.raw_viewports[i], aux.rt_width, aux.rt_height, max_dim);
        }
        bridge_.RSSetViewports(static_cast<uint32_t>(aux.clamped_viewports.size()), aux.clamped_viewports.data());
    }
    if (!aux.raw_scissors.empty()) {
        aux.clamped_scissors.resize(aux.raw_scissors.size());
        for (size_t i = 0; i < aux.raw_scissors.size(); ++i) {
            aux.clamped_scissors[i] = clamp_scissor_webgpu(aux.raw_scissors[i], aux.rt_width, aux.rt_height, max_dim);
        }
        bridge_.RSSetScissorRects(static_cast<uint32_t>(aux.clamped_scissors.size()), aux.clamped_scissors.data());
    }
}

void CommandList::ClearDepthStencilView(uint64_t dsv, uint32_t flags, float depth, uint8_t stencil,
                                        uint32_t rect_count, uint64_t rects) {
    if (!usable()) {
        return;
    }
    std::vector<pso::D3D12Rect> r;
    if (rect_count > kMaxRects || !read_n(rects, rect_count, &r)) {
        note("ClearDepthStencilView: unreadable rects");
        return;
    }
    flags &= (bridge::DepthTwinTracker::kClearFlagDepth | bridge::DepthTwinTracker::kClearFlagStencil);
    if (flags == 0u) {
        return;  // neither depth nor stencil clear flag set: valid no-op
    }
    // WebGPU depthClearValue must be finite and in [0.0, 1.0].
    const float clean_depth = std::isfinite(depth) ? std::clamp(depth, 0.0f, 1.0f) : 1.0f;

    uint32_t mip_w = device_->state_bridge().options().max_texture_dimension_2d;
    uint32_t mip_h = mip_w;
    d3d::ResolvedDescriptor res{};
    const d3d::ViewDescriptor* dv = nullptr;
    if (dsv != 0 && device_->heaps().resolve_cpu(d3d::CpuDescriptorHandle{dsv}, &res) == d3d::Status::Ok &&
        res.heap != nullptr) {
        dv = res.heap->view(res.index);
        if (dv != nullptr && dv->resource_id != 0) {
            bridge::TextureInfo ti{};
            if (device_->state_bridge().texture_info(dv->resource_id, &ti)) {
                mip_w = std::max<uint32_t>(1u, ti.width >> dv->first_mip);
                mip_h = std::max<uint32_t>(1u, ti.height >> dv->first_mip);
            }
        }
    }

    std::vector<pso::D3D12Rect> clamped_rects;
    if (rect_count > 0) {
        clamped_rects.reserve(rect_count);
        for (const pso::D3D12Rect& rect : r) {
            const pso::D3D12Rect c = clamp_scissor_webgpu(rect, mip_w, mip_h, mip_w > mip_h ? mip_w : mip_h);
            if (c.right > c.left && c.bottom > c.top) {
                clamped_rects.push_back(c);
            }
        }
        if (clamped_rects.empty()) {
            return;  // all clear rects fell outside the DSV extents
        }
    }

    if ((flags & bridge::DepthTwinTracker::kClearFlagDepth) != 0u && dv != nullptr && dv->resource_id != 0) {
        const uint32_t layers = dv->array_size != 0 ? dv->array_size : 1u;
        aux_of(this).depth_twin.note_depth_cleared(dv->resource_id, flags, dv->first_mip, 1u, dv->first_array_slice,
                                                   layers, /*full_subresource=*/rect_count == 0u);
    }

    bridge_.ClearDepthStencilView(d3d::CpuDescriptorHandle{dsv}, flags, clean_depth, stencil,
                                  static_cast<uint32_t>(clamped_rects.size()),
                                  clamped_rects.empty() ? nullptr : clamped_rects.data());
}

void CommandList::ClearRenderTargetView(uint64_t rtv, uint64_t color, uint32_t rect_count, uint64_t rects) {
    if (!usable()) {
        return;
    }
    float c[4] = {};
    std::vector<pso::D3D12Rect> r;
    if (color == 0 || !com::read_guest(color, c, sizeof(c)) || rect_count > kMaxRects || !read_n(rects, rect_count, &r)) {
        note("ClearRenderTargetView: unreadable color or rects");
        return;
    }
    bridge_.ClearRenderTargetView(d3d::CpuDescriptorHandle{rtv}, c, rect_count, r.empty() ? nullptr : r.data());
}

// ---------------------------------------------------------------------------------------------
// Vtable (Xbox ID3D12GraphicsCommandList order, see d3d12_xs_xbox.h)
// ---------------------------------------------------------------------------------------------

namespace {

using CL = Methods<IGraphicsCommandListHost>;
using H = IGraphicsCommandListHost;
constexpr com::MethodDesc kCommandListMethods[] = {
    CL::method<&IObjectHost::GetPrivateData>("GetPrivateData"),                       // 3
    CL::method<&IObjectHost::SetPrivateData>("SetPrivateData"),                       // 4
    CL::method<&IObjectHost::SetPrivateDataInterface>("SetPrivateDataInterface"),     // 5
    CL::method<&IObjectHost::SetName>("SetName"),                                     // 6
    CL::method<&IDeviceChildHost::GetDevice>("GetDevice"),                            // 7
    CL::method<&H::GetType>("GetType"),                                               // 8
    CL::method<&H::Close>("Close"),                                                   // 9
    CL::method<&H::Reset>("Reset"),                                                   // 10
    CL::method<&H::ClearState>("ClearState"),                                         // 11
    CL::method<&H::DrawInstanced>("DrawInstanced"),                                   // 12
    CL::method<&H::DrawIndexedInstanced>("DrawIndexedInstanced"),                     // 13
    CL::method<&H::Dispatch>("Dispatch"),                                             // 14
    CL::method<&H::CopyBufferRegion>("CopyBufferRegion"),                             // 15
    CL::method<&H::CopyTextureRegion>("CopyTextureRegion"),                           // 16
    CL::method<&H::CopyResource>("CopyResource"),                                     // 17
    CL::stub("CopyTiles"),                                                            // 18
    CL::method<&H::ResolveSubresource>("ResolveSubresource"),                         // 19
    CL::method<&H::IASetPrimitiveTopology>("IASetPrimitiveTopology"),                 // 20
    CL::method<&H::RSSetViewports>("RSSetViewports"),                                 // 21
    CL::method<&H::RSSetScissorRects>("RSSetScissorRects"),                           // 22
    CL::method<&H::OMSetBlendFactor>("OMSetBlendFactor"),                             // 23
    CL::method<&H::OMSetStencilRef>("OMSetStencilRef"),                               // 24
    CL::method<&H::SetPipelineStateX>("SetPipelineState (staged)"),                   // 25
    CL::method<&H::ResourceBarrier>("ResourceBarrier"),                               // 26
    CL::stub("ExecuteBundle"),                                                        // 27
    // Xbox order from here on (d3d12_xs_xbox.h): queries and predication moved up, public 28..51 +4.
    CL::method<&H::BeginQuery>("BeginQuery"),                                     // 28
    CL::method<&H::EndQuery>("EndQuery"),                                       // 29
    CL::method<&H::ResolveQueryData>("ResolveQueryData"),                               // 30
    CL::method<&H::SetPredication>("SetPredication (inferred)"),                      // 31
    CL::method<&H::SetDescriptorHeaps>("SetDescriptorHeaps"),                         // 32
    CL::method<&H::SetComputeRootSignature>("SetComputeRootSignature"),               // 33
    CL::method<&H::SetGraphicsRootSignature>("SetGraphicsRootSignature"),             // 34
    CL::method<&H::SetComputeRootDescriptorTable>("SetComputeRootDescriptorTable"),   // 35
    CL::method<&H::SetGraphicsRootDescriptorTable>("SetGraphicsRootDescriptorTable"), // 36
    CL::method<&H::SetComputeRoot32BitConstant>("SetComputeRoot32BitConstant"),       // 37
    CL::method<&H::SetGraphicsRoot32BitConstant>("SetGraphicsRoot32BitConstant"),     // 38
    CL::method<&H::SetComputeRoot32BitConstants>("SetComputeRoot32BitConstants"),     // 39
    CL::method<&H::SetGraphicsRoot32BitConstants>("SetGraphicsRoot32BitConstants"),   // 40
    CL::method<&H::SetComputeRootConstantBufferView>("SetComputeRootConstantBufferView"),    // 41
    CL::method<&H::SetGraphicsRootConstantBufferView>("SetGraphicsRootConstantBufferView"),  // 42
    CL::method<&H::SetComputeRootShaderResourceView>("SetComputeRootShaderResourceView"),    // 43
    CL::method<&H::SetGraphicsRootShaderResourceView>("SetGraphicsRootShaderResourceView"),  // 44
    CL::method<&H::SetComputeRootUnorderedAccessView>("SetComputeRootUnorderedAccessView"),  // 45
    CL::method<&H::SetGraphicsRootUnorderedAccessView>("SetGraphicsRootUnorderedAccessView"),  // 46
    CL::method<&H::IASetIndexBuffer>("IASetIndexBuffer"),                             // 47
    CL::method<&H::IASetVertexBuffers>("IASetVertexBuffers"),                         // 48
    CL::method<&H::SOSetTargets>("SOSetTargets"),                                     // 49
    CL::method<&H::OMSetRenderTargets>("OMSetRenderTargets"),                         // 50
    CL::method<&H::ClearDepthStencilView>("ClearDepthStencilView"),                   // 51
    CL::method<&H::ClearRenderTargetView>("ClearRenderTargetView"),                   // 52
    CL::method<&H::ClearUnorderedAccessView>("ClearUnorderedAccessViewUint"),         // 53
    CL::method<&H::ClearUnorderedAccessView>("ClearUnorderedAccessViewFloat"),        // 54
    CL::method<&H::DiscardResource>("DiscardResource"),                               // 55
    CL::method<&H::DebugMarker>("SetMarker (inferred)"),                              // 56
    CL::method<&H::DebugMarker>("BeginEvent (inferred)"),                             // 57
    CL::method<&H::DebugMarker>("EndEvent (inferred)"),                               // 58
    CL::method<&H::XboxSlot<59>>("Xbox slot 59"),                                     // 59
    CL::method<&H::XboxSlot<60>>("Xbox slot 60"),                                     // 60
    CL::method<&H::ExecuteIndirect>("ExecuteIndirect"),                               // 61
    CL::method<&H::XboxSlot<62>>("Xbox slot 62"),                                     // 62
    CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr),       // 63..66
    CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr),                          // 67..69
    CL::method<&H::OMSetDepthBounds>("OMSetDepthBounds"),                             // 70
    CL::method<&H::WriteValue32>("write value 32 (Xbox slot 71)"),                    // 71
    CL::method<&H::XboxSlot<72>>("Xbox slot 72"),                                     // 72
    CL::stub(nullptr),                                                                // 73
    CL::method<&H::WriteValue64>("write value 64 (Xbox slot 74)"),                    // 74
    CL::method<&H::XboxSlot<75>>("Xbox slot 75"),                                     // 75
    CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr),       // 76..79
    CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr),       // 80..83
    CL::stub(nullptr),                                                                // 84
    CL::method<&H::CopyTextureRegionX>("CopyTextureRegion (Xbox slot 85)"),           // 85
    CL::method<&H::XboxSlot<86>>("Xbox slot 86"),                                     // 86
    CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr),                          // 87..89
    CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr),       // 90..93
    CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr),       // 94..97
    CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr),                          // 98..100
    // Slot 101 was hit once late in a browser boot (issue 663) and is UNIDENTIFIED. Public headers cannot name it:
    // ID3D12GraphicsCommandList..10 end at slot 85, and this table follows the Xbox order (RSSetShadingRate is 126
    // here, 77 in public; Barrier 146 vs 80). No static call site in the exe applies it to a command list: all 27
    // `call [vtable+0x328]` sites (22 functions, no register / CFG-indirect forms, none in the bundled DLLs) act on
    // title objects or the device (device slot 101 = CreateStateObject). So the hit points to a confused or stale
    // object pointer, not to a missing method. Kept an E_NOTIMPL stub on purpose (logged once with `this` and the
    // return address, hits counted: com::stub_hits); the call sites' return addresses are in docs/gpu_commands.md.
    CL::stub("Xbox slot 101 (unidentified)"),                                         // 101
    CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr),       // 102..105
    CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr),       // 106..109
    CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr),       // 110..113
    CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr),       // 114..117
    CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr),       // 118..121
    CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr),       // 122..125
    CL::method<&H::RSSetShadingRate>("RSSetShadingRate"),                             // 126
    CL::method<&H::RSSetShadingRateImage>("RSSetShadingRateImage"),                   // 127
    CL::method<&H::XboxSlot<128>>("Xbox slot 128"),                                   // 128
    CL::method<&H::XboxSlot<129>>("Xbox slot 129"),                                   // 129
    CL::stub(nullptr),                                                                // 130
    CL::method<&H::XboxSlot<131>>("Xbox slot 131 (register write)"),                  // 131
    CL::method<&H::XboxSlot<132>>("Xbox slot 132 (register write)"),                  // 132
    CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr),       // 133..136
    CL::stub(nullptr), CL::stub(nullptr),                                             // 137..138
    CL::method<&H::ResetX>("Reset (Xbox slot 139)"),                                  // 139
    CL::stub(nullptr),                                                                // 140
    CL::stub(nullptr),                                                                // 141
    CL::method<&H::XboxSlot<142>>("Xbox slot 142"),                                   // 142
    CL::stub(nullptr), CL::stub(nullptr), CL::stub(nullptr),                          // 143..145
    CL::method<&H::Barrier>("Barrier (Xbox slot 146)"),                               // 146
};
constexpr Guid kCommandListIids[] = {
    IID_ID3D12Object,
    IID_ID3D12DeviceChild,
    IID_ID3D12CommandList,
    com::make_guid("553103fb-1fe7-4557-bb38-946d7d0e7ca7"),  // ID3D12GraphicsCommandList1
    com::make_guid("38c3e585-ff17-412c-9150-4fc6f9d72a28"),  // ID3D12GraphicsCommandList2
    com::make_guid("6fda83a7-b84c-4e38-9ac8-c7bd22016b3d"),  // ID3D12GraphicsCommandList3
    com::make_guid("8754318e-d3a9-4541-98cf-645b50dc4874"),  // ID3D12GraphicsCommandList4
    com::make_guid("55050859-4024-474c-87f5-6472eaee44ea"),  // ID3D12GraphicsCommandList5
    com::make_guid("c3827890-e548-4cfa-96cf-5689a9370f80"),  // ID3D12GraphicsCommandList6
    // Requested by the title from CreateCommandList (Xbox slot 73), first for a direct list: the
    // Xbox runtime's own graphics-command-list IID (the vtable this interface implements, Xbox
    // slots included, was built from the title's command-list calls).
    com::make_guid("c3827890-4024-474c-98cf-645b51dc4874"),
};

constexpr com::InterfaceOptions list_options() {
    com::InterfaceOptions o;
    o.slot_count = 192;  // Xbox-specific slots up to at least 132 are called (logging stubs)
    o.policy = com::SlotPolicy::ReturnNotImpl;
    o.extra_iids = kCommandListIids;
    return o;
}

}  // namespace

const com::Interface<IGraphicsCommandListHost> kID3D12GraphicsCommandList =
    com::declare_interface<IGraphicsCommandListHost>("ID3D12GraphicsCommandList", IID_ID3D12GraphicsCommandList,
                                                     kCommandListMethods, list_options());

}  // namespace playgta6::gpu::d3d12_guest
