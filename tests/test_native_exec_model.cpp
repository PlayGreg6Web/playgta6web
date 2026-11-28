// Native executor stream model (gpu/native_exec/stream_model.h, replay_frames.h): packet walking,
// payload decoding and the replay frame split / digest, checked against the browser worker's rules.
// No GPU involved.
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "gpu/cmd_encoder.h"
#include "gpu/native_exec/binding_access_model.h"
#include "gpu/native_exec/blit_model.h"
#include "gpu/native_exec/clear_model.h"
#include "gpu/native_exec/adapter_model.h"
#include "gpu/native_exec/binding_fit_model.h"
#include "gpu/native_exec/error_scope_model.h"
#include "gpu/native_exec/fragment_outputs_model.h"
#include "gpu/native_exec/gpu_stall_model.h"
#include "gpu/native_exec/layout_model.h"
#include "gpu/native_exec/m1_model.h"
#include "gpu/native_exec/memory_model.h"
#include "gpu/native_exec/pipe_budget_model.h"
#include "gpu/native_exec/shader_text_model.h"
#include "gpu/native_exec/view_alias_model.h"
#include "gpu/native_exec/m2_model.h"
#include "gpu/native_exec/present_model.h"
#include "gpu/native_exec/replay_frames.h"
#include "gpu/native_exec/shader_workers.h"
#include "gpu/native_exec/show_target_model.h"
#include "gpu/native_exec/stream_model.h"
#include "gpu/native_exec/target_dump_model.h"
#include "gpu/native_exec/usage_scope_model.h"

namespace {
int g_failures = 0;
void check_impl(bool cond, const char* expr, int line) {
    if (!cond) {
        std::fprintf(stderr, "FAIL [test_native_exec_model.cpp:%d]: %s\n", line, expr);
        ++g_failures;
    }
}
#define CHECK(expr) check_impl(static_cast<bool>(expr), #expr, __LINE__)

using namespace playgta6;
using namespace playgta6::gpu::native_exec;

std::vector<Packet> walk_all(const uint8_t* data, size_t len, WalkResult* out = nullptr) {
    std::vector<Packet> v;
    WalkResult r = walk_packets(data, len, [&](const Packet& p) {
        v.push_back(p);
        return true;
    });
    if (out) *out = r;
    return v;
}

Packet single(const gpu::CommandEncoder& enc) {
    auto v = walk_all(enc.data(), enc.size());
    return v.empty() ? Packet{} : v[0];
}

void test_walk() {
    gpu::CommandEncoder enc;
    enc.begin_frame();
    enc.clear_color(0.25f, 0.5f, 0.75f, 1.0f);
    enc.present();
    WalkResult r;
    auto packets = walk_all(enc.data(), enc.size(), &r);
    CHECK(packets.size() == 3);
    CHECK(!r.truncated && r.bytes == enc.size());
    CHECK(packets[1].opcode == static_cast<uint16_t>(GpuOpcode::ClearColor) && packets[1].payload_bytes == 16);
    CHECK(packets[2].offset == 8 + 24);
    CHECK(complete_prefix(enc.data(), enc.size()) == enc.size());

    // A header whose payload runs past the end stops the walk (the worker's CommandStream error).
    std::vector<uint8_t> bytes(enc.data(), enc.data() + enc.size());
    const GpuPacketHeader bad{static_cast<uint16_t>(GpuOpcode::WriteBuffer), 0, 100};
    const size_t at = bytes.size();
    bytes.resize(at + 8 + 10);
    std::memcpy(bytes.data() + at, &bad, 8);
    packets = walk_all(bytes.data(), bytes.size(), &r);
    CHECK(packets.size() == 3);
    CHECK(r.truncated && r.truncated_opcode == 4 && r.truncated_need == 100 && r.truncated_remaining == 10);
    CHECK(complete_prefix(bytes.data(), bytes.size()) == at);
    // Fewer than 8 trailing bytes are ignored silently.
    bytes.resize(at + 5);
    walk_all(bytes.data(), bytes.size(), &r);
    CHECK(!r.truncated && r.packets == 3);

    // The visitor can stop the walk (device lost mid-batch).
    int seen = 0;
    r = walk_packets(enc.data(), enc.size(), [&](const Packet&) { return ++seen < 2; });
    CHECK(seen == 2 && r.stopped && r.packets == 2);
}

void test_clear_color() {
    gpu::CommandEncoder enc;
    enc.clear_color(1.0f, 0.1f, 0.0f, 0.5f);
    auto c = decode_clear_color(single(enc));
    CHECK(c.has_value());
    CHECK(c->r == 1.0 && c->g == static_cast<double>(0.1f) && c->b == 0.0 && c->a == 0.5);
    gpu::CommandEncoder shorter;
    const float three[3] = {1, 1, 1};
    shorter.emit_packet(GpuOpcode::ClearColor, three, sizeof three);
    CHECK(!decode_clear_color(single(shorter)).has_value());
    CHECK(kDefaultClearColor == (ClearValue{0.05, 0.07, 0.11, 1.0}));

    CHECK(sanitize_clear_component(std::numeric_limits<float>::quiet_NaN(), 1.0) == 1.0);
    CHECK(sanitize_clear_component(std::numeric_limits<float>::infinity(), 0.0) == 0.0);
    const double z = sanitize_clear_component(-0.0f, 0.0);
    CHECK(z == 0.0 && !std::signbit(z));
    CHECK(sanitize_clear_component(0.5f, 0.0) == 0.5);
}

GpuBeginRenderPassPayload empty_pass() {
    GpuBeginRenderPassPayload bp;
    std::memset(&bp, 0, sizeof bp);
    return bp;
}

void test_begin_render_pass() {
    GpuBeginRenderPassPayload bp = empty_pass();
    bp.color_count = 12;  // clamped to 8
    bp.color[0].texture_id = kGpuSwapchainTextureId;
    bp.color[0].load_op = static_cast<uint8_t>(GpuLoadOp::Clear);
    bp.color[0].store_op = static_cast<uint8_t>(GpuStoreOp::Discard);
    bp.color[0].clear_value[0] = std::numeric_limits<float>::quiet_NaN();
    bp.color[0].clear_value[1] = 0.25f;
    bp.color[0].clear_value[2] = -0.0f;
    bp.color[0].clear_value[3] = std::numeric_limits<float>::infinity();
    bp.color[1].texture_id = 9;
    bp.color[1].load_op = 0;   // undefined -> load
    bp.color[1].store_op = 7;  // unknown -> store
    bp.color[1].view.format = 27;
    gpu::CommandEncoder enc;
    enc.emit_packet(GpuOpcode::BeginRenderPass, &bp, sizeof bp);
    auto plan = decode_begin_render_pass(single(enc));
    CHECK(plan.has_value());
    CHECK(plan->color_count == 8);
    CHECK(plan->colors[0].load == LoadOp::Clear && plan->colors[0].store == StoreOp::Discard);
    CHECK(plan->colors[0].clear == (ClearValue{0.0, 0.25, 0.0, 1.0}));
    CHECK(plan->colors[1].texture_id == 9 && plan->colors[1].load == LoadOp::Load &&
          plan->colors[1].store == StoreOp::Store && plan->colors[1].view.format == 27);
    CHECK(plan->colors[2].texture_id == 0);
    CHECK(!plan->has_depth_stencil);

    gpu::CommandEncoder shorter;
    shorter.emit_packet(GpuOpcode::BeginRenderPass, &bp, 367);
    CHECK(!decode_begin_render_pass(single(shorter)).has_value());

    // Size sanitizer: the first sized color attachment is the reference.
    RenderPassPlan p;
    p.color_count = 3;
    p.colors[0].texture_id = 1;
    p.colors[1].texture_id = 2;
    p.colors[2].texture_id = 3;
    std::array<AttachmentSize, kGpuMaxColorAttachments> sizes{};
    sizes[0] = {};          // unknown: skipped
    sizes[1] = {64, 32};    // reference
    sizes[2] = {32, 32};    // dropped
    CHECK(drop_mismatched_colors(p, sizes, {}) == 0b100);
    sizes[2] = {64, 32};
    CHECK(drop_mismatched_colors(p, sizes, {}) == 0);
    // Null attachments are never the reference (nor dropped); the depth-stencil attachment does not
    // outrank a sized color attachment.
    sizes = {};
    sizes[0] = {16, 16};
    p.colors[0].texture_id = 0;
    p.has_depth_stencil = true;
    sizes[1] = {8, 8};
    sizes[2] = {4, 4};
    CHECK(drop_mismatched_colors(p, sizes, {4, 4}) == 0b100);
    CHECK(mip_size({64, 32}, 2) == (AttachmentSize{16, 8}));
    CHECK(mip_size({64, 32}, 6) == (AttachmentSize{1, 1}));
    CHECK(mip_size({64, 32}, 40) == (AttachmentSize{1, 1}));

    // A depth-stencil attachment of another size: the pass takes the depth size and the mismatched colors
    // render through proxies (instead of the sanitizer dropping the depth attachment).
    RenderPassPlan q;
    q.color_count = 2;
    q.colors[0].texture_id = 1;
    q.colors[1].texture_id = 2;
    q.has_depth_stencil = true;
    std::array<AttachmentSize, kGpuMaxColorAttachments> cs{};
    cs[0] = {1920, 1080};
    cs[1] = {1920, 1088};
    auto pp = plan_attachment_proxies(q, cs, {1920, 1088}, 0b11);
    CHECK(pp.has_value() && pp->proxied == 0b01 && pp->pass == (AttachmentSize{1920, 1088}));
    CHECK(pp->copy[0] == (AttachmentSize{1920, 1080}) && pp->area == (AttachmentSize{1920, 1080}));
    cs[0] = {3840, 2160};  // a color larger than the depth: the shared region is the depth's
    pp = plan_attachment_proxies(q, cs, {1920, 1088}, 0b11);
    CHECK(pp.has_value() && pp->copy[0] == (AttachmentSize{1920, 1088}) && pp->area == (AttachmentSize{1920, 1088}));
    CHECK(!plan_attachment_proxies(q, cs, {1920, 1088}, 0b10).has_value());  // slot 0 cannot be proxied
    cs[0] = {1920, 1088};
    CHECK(!plan_attachment_proxies(q, cs, {1920, 1088}, 0b11).has_value());  // sizes agree
    q.has_depth_stencil = false;
    cs[0] = {64, 64};
    CHECK(!plan_attachment_proxies(q, cs, {}, 0b11).has_value());  // no depth: the sanitizer decides
}

void test_write_buffer() {
    const uint8_t data[9] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<std::unique_ptr<gpu::CommandEncoder>> encs;  // the plans point into the encoders' bytes
    auto plan_for = [&](uint32_t offset, uint32_t len, uint64_t size) {
        gpu::CommandEncoder& enc = *encs.emplace_back(std::make_unique<gpu::CommandEncoder>());
        enc.write_buffer(5, offset, data, len);
        return plan_write_buffer(single(enc), size);
    };
    WriteBufferPlan p = plan_for(4, 5, 16);  // staged: padded to 8 bytes
    CHECK(p.path == WriteBufferPath::Padded && p.copy_size == 8 && p.data_len == 5 && p.id == 5);
    CHECK(p.data && p.data[0] == 1 && p.data[4] == 5);
    p = plan_for(8, 8, 16);
    CHECK(p.path == WriteBufferPath::Padded && p.copy_size == 8);
    p = plan_for(12, 5, 16);  // the padded copy would overflow: direct (WebGPU rejects the size)
    CHECK(p.path == WriteBufferPath::Direct);
    p = plan_for(2, 4, 16);  // unaligned offset: direct (WebGPU rejects the offset)
    CHECK(p.path == WriteBufferPath::Direct && p.dst_offset == 2);
    p = plan_for(0, 0, 16);
    CHECK(p.path == WriteBufferPath::Direct && p.data_len == 0);

    // data_len past the payload: the worker returns without writing.
    GpuWriteBufferPayload w{5, 0, 64, 0};
    gpu::CommandEncoder enc;
    enc.emit_packet(GpuOpcode::WriteBuffer, &w, sizeof w, data, sizeof data);
    CHECK(plan_write_buffer(single(enc), 1024).path == WriteBufferPath::Skip);
}

void test_split_frames() {
    gpu::CommandEncoder enc;
    enc.create_buffer(1, GpuBufferUsage::CopyDst, 64);
    enc.begin_frame();
    enc.present();
    const size_t frame0 = enc.size();
    // A thinned upload: data_len 0, reserved == 'thin'.
    GpuWriteBufferPayload thin{1, 0, 0, kThinMarker};
    enc.emit_packet(GpuOpcode::WriteBuffer, &thin, sizeof thin);
    enc.clear_color(1, 0, 0, 1);
    enc.present();
    enc.begin_frame();  // trailing packets: a final frame
    auto frames = split_replay_frames(enc.data(), enc.size());
    CHECK(frames.size() == 3);
    CHECK(frames[0].size() == frame0);
    CHECK(frames[1].size() == 24 + 8);  // the thin packet is gone
    CHECK(frames[2].size() == 8);
    WalkResult r;
    auto packets = walk_all(frames[1].data(), frames[1].size(), &r);
    CHECK(packets.size() == 2 && packets[0].opcode == static_cast<uint16_t>(GpuOpcode::ClearColor));
}

void test_digest() {
    // Reference value computed by tools/replay/replay.html's digest code on the same pixels.
    uint8_t px[3 * 2 * 4];
    for (size_t i = 0; i < sizeof px; ++i) px[i] = static_cast<uint8_t>((i * 37 + 11) & 255);
    CHECK(replay_digest(px, 3, 2) == "1876ef7b336afd4d");

    std::vector<uint8_t> img(8 * 8 * 4, 0);
    // Sampled pixels are (0,0), (4,0), (0,4), (4,4); alpha does not count.
    img[3] = 255;                       // (0,0): alpha only -> black
    img[(0 * 8 + 4) * 4 + 1] = 10;      // (4,0)
    img[(4 * 8 + 4) * 4 + 1] = 10;      // (4,4): same colour
    img[(1 * 8 + 1) * 4] = 99;          // not sampled
    const SampledStats s = replay_sampled_stats(img.data(), 8, 8);
    CHECK(s.total == 4 && s.non_black == 2 && s.distinct == 2);
}

}  // namespace

// ---- milestone 1 (m1_model.h) ----

void test_shader_payloads() {
    // parseHashOnlyShaderPayload
    CHECK(parse_hash_only_shader_payload("0x006d0f6d4b911457") == 0x006d0f6d4b911457ull);
    CHECK(parse_hash_only_shader_payload("hash:0xFF") == 0xffull);
    CHECK(parse_hash_only_shader_payload("WGPK : 12ab") == 0x12abull);
    CHECK(parse_hash_only_shader_payload("0123456789abcdef") == 0x0123456789abcdefull);
    CHECK(!parse_hash_only_shader_payload("0123456789abcde"));  // bare hex needs exactly 16 digits
    CHECK(!parse_hash_only_shader_payload("0x"));
    CHECK(!parse_hash_only_shader_payload("0x12345678901234567"));  // 17 digits
    CHECK(!parse_hash_only_shader_payload("@vertex fn vs_main() {}"));
    CHECK(!parse_hash_only_shader_payload(""));
    CHECK(trim_view("  \t x \n") == "x");

    // Entry points: comments are ignored by the declared-entry scan, not by the recorded one.
    const std::string code =
        "// @vertex fn commented() {}\n"
        "/* @vertex fn block() {} */\n"
        "@group(0) @binding(0) var<uniform> u: vec4f;\n"
        "@vertex\nfn vs_other(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f { return u; }\n"
        "@fragment @must_use fn fs_main() -> @location(0) vec4f { return u; }\n"
        "@vertex @invariant(x) fn vs_main() -> @builtin(position) vec4f { return u; }\n";
    const auto vs = find_wgsl_stage_entry_points(code, "vertex");
    CHECK(vs.size() == 2 && vs[0] == "vs_other" && vs[1] == "vs_main");
    CHECK((find_wgsl_stage_entry_points(code, "fragment") == std::vector<std::string>{"fs_main"}));
    CHECK(find_wgsl_stage_entry_points(code, "compute").empty());
    CHECK(extract_wgsl_entry_point(code, "vertex") == std::string("commented"));  // regex over raw text
    CHECK(extract_wgsl_entry_point("fn helper() {}\n@compute fn cs() {}", "") == std::string("helper"));
    CHECK(!extract_wgsl_entry_point("", "vertex"));
    // resolveShaderEntryPoint: the default name wins among several; else the explicit one when
    // declared; else the first declared.
    CHECK(resolve_shader_entry_point(code, std::string("vs_other"), "vertex") == "vs_other");
    CHECK(resolve_shader_entry_point(code, std::string("commented"), "vertex") == "vs_main");
    CHECK(resolve_shader_entry_point("@vertex fn only() {}", std::nullopt, "vertex") == "only");
    CHECK(resolve_shader_entry_point("", std::nullopt, "compute") == "cs_main");
    CHECK(resolve_shader_entry_point("", std::string("custom"), "fragment") == "custom");
    // All cached stage resolutions match the old individual path for alternate-stage reuse,
    // comments, absent stages and explicit names that do or do not exist in this module.
    for (const std::string& source : {code, std::string{}, std::string("@compute fn custom() {}"),
                                      std::string("/* @fragment fn ignored() {} */ @vertex fn v() {}")}) {
        for (const std::optional<std::string>& explicit_ep : {std::optional<std::string>{},
                                                               std::optional<std::string>{"vs_other"},
                                                               std::optional<std::string>{"bad-name"}}) {
            const auto all = resolve_all_shader_entry_points(source, explicit_ep);
            CHECK(all[0] == resolve_shader_entry_point(source, explicit_ep, "vertex"));
            CHECK(all[1] == resolve_shader_entry_point(source, explicit_ep, "fragment"));
            CHECK(all[2] == resolve_shader_entry_point(source, explicit_ep, "compute"));
        }
    }
    CHECK(wgsl_declares_group0("@group( 0 ) @binding(1) var s: sampler;"));
    CHECK(!wgsl_declares_group0("@group(1) @binding(0) var s: sampler;"));
    CHECK(strip_wgsl_comments("a /* b /* c */ d */ e // f\ng") == "a    e \ng");
}

void test_pack() {
    const std::string a = "@vertex fn vs_main() {}", b = "@fragment fn fs_main() {}";
    std::vector<uint8_t> pack = {'W', 'G', 'P', 'K', 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0};
    auto u32 = [&](uint32_t v) { for (int i = 0; i < 4; ++i) pack.push_back(static_cast<uint8_t>(v >> (8 * i))); };
    auto u64 = [&](uint64_t v) { for (int i = 0; i < 8; ++i) pack.push_back(static_cast<uint8_t>(v >> (8 * i))); };
    const uint32_t data_start = 16 + 2 * 20;
    u64(0x1111); pack.push_back(1); pack.insert(pack.end(), 3, 0); u32(data_start); u32(static_cast<uint32_t>(a.size()));
    u64(0x2222); pack.push_back(2); pack.insert(pack.end(), 3, 0); u32(data_start + static_cast<uint32_t>(a.size()));
    u32(static_cast<uint32_t>(b.size()));
    pack.insert(pack.end(), a.begin(), a.end());
    pack.insert(pack.end(), b.begin(), b.end());
    const char* path = "test_native_exec_model.wgsl.pack";
    FILE* f = std::fopen(path, "wb");
    CHECK(f != nullptr);
    if (!f) return;
    std::fwrite(pack.data(), 1, pack.size(), f);
    std::fclose(f);
    {
        WgslPack p;
        std::string err;
        CHECK(p.load(path, &err));
        CHECK(p.size() == 2 && p.has(0x1111) && p.has(0x2222) && !p.has(0x3333));
        CHECK(p.get(0x2222) == b);
        CHECK(p.get(0x1111) == a);
        CHECK(!p.get(0x3333));
        WgslPack bad;
        CHECK(!bad.load("test_native_exec_model.does-not-exist", &err));
    }
    std::remove(path);

    // WGPK v2 (merged packs): 24-byte entries {hash, stage, pad, length u32, offset u64}; the live run
    // logged shader_pack_hits 0 because the executor's own loader accepted v1 only.
    std::vector<uint8_t> v2 = {'W', 'G', 'P', 'K', 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0};
    auto w32 = [&](uint32_t v) { for (int i = 0; i < 4; ++i) v2.push_back(static_cast<uint8_t>(v >> (8 * i))); };
    auto w64 = [&](uint64_t v) { for (int i = 0; i < 8; ++i) v2.push_back(static_cast<uint8_t>(v >> (8 * i))); };
    const uint64_t ds2 = 16 + 2 * 24;
    w64(0x1111); v2.push_back(1); v2.insert(v2.end(), 3, 0); w32(static_cast<uint32_t>(a.size())); w64(ds2);
    w64(0x2222); v2.push_back(2); v2.insert(v2.end(), 3, 0); w32(static_cast<uint32_t>(b.size())); w64(ds2 + a.size());
    v2.insert(v2.end(), a.begin(), a.end());
    v2.insert(v2.end(), b.begin(), b.end());
    const char* path2 = "test_native_exec_model_v2.wgsl.pack";
    FILE* f2 = std::fopen(path2, "wb");
    CHECK(f2 != nullptr);
    if (!f2) return;
    std::fwrite(v2.data(), 1, v2.size(), f2);
    std::fclose(f2);
    {
        WgslPack p;
        std::string err;
        CHECK(p.load(path2, &err));
        CHECK(p.size() == 2 && p.has(0x1111) && p.has(0x2222));
        CHECK(p.get(0x2222) == b);
        CHECK(p.get(0x1111) == a);
    }
    std::remove(path2);
}

void test_formats() {
    CHECK(std::string(format_info(22).name) == "rgba8unorm" && std::string(format_info(62).name) == "bc7-rgba-unorm");
    CHECK(std::string(format_info(63).name) == "bc7-rgba-unorm-srgb" && std::string(format_info(49).name) == "depth32float-stencil8");
    CHECK(format_info(62).compressed && format_info(62).block_w == 4 && format_info(62).bytes == 16);
    CHECK(format_info(50).bytes == 8 && format_info(43).bytes == 16 && format_info(1).bytes == 1);
    CHECK(!format_valid(0) && format_valid(63) && !format_valid(64));
    CHECK(format_is_bc(50) && format_is_bc(63) && !format_is_bc(49));
    CHECK(format_is_srgb(23) && format_is_srgb(63) && !format_is_srgb(22));
    CHECK(format_is_depth(46) && format_is_depth(49) && !format_is_depth(44) && format_is_depth_or_stencil(44));
    CHECK(format_is_float32(43) && !format_is_float32(40));
    CHECK(format_is_integer(25) && format_is_integer(30) && !format_is_integer(31) && !format_is_integer(22));
    CHECK(srgb_sibling(22) == 23 && srgb_sibling(23) == 22 && srgb_sibling(62) == 63 && srgb_sibling(40) == 0);
    CHECK(v1_format_code(5) == 62 && v1_format_code(3) == 46 && v1_format_code(99) == 22);
    // resolveCompatibleViewFormats
    bool added = false;
    auto v = resolve_compatible_view_formats(22, {}, &added);
    CHECK((v == std::vector<uint8_t>{23}) && added);
    v = resolve_compatible_view_formats(22, {22, 40, 23, 22}, &added);
    CHECK((v == std::vector<uint8_t>{22, 23}) && !added);
    CHECK(resolve_compatible_view_formats(40, {40, 22}).size() == 1);
    CHECK(resolve_compatible_view_formats(40, {}).empty());
    // Mips and samples
    CHECK(max_mip_levels(256, 64, 1, 1) == 9 && max_mip_levels(8, 8, 32, 2) == 6 && max_mip_levels(5, 5, 1, 0) == 1);
    CHECK(clamp_mip_levels(0, 16, 16, 1, 1) == 1 && clamp_mip_levels(99, 16, 16, 1, 1) == 5);
    CHECK(clamp_sample_count(4, 1, 1, 1) == 4 && clamp_sample_count(4, 2, 1, 1) == 1 && clamp_sample_count(4, 1, 2, 1) == 1);
    CHECK(mip_dimension(100, 0) == 100 && mip_dimension(100, 3) == 12 && mip_dimension(1, 5) == 1 && mip_dimension(8, 40) == 1);
}

void test_write_texture_plan() {
    TextureGeometry g;
    g.width = 64;
    g.height = 64;
    g.mips = 7;
    g.format = 22;  // rgba8unorm
    WriteTextureArgs a;
    a.width = 64; a.height = 64; a.data_len = 64 * 64 * 4;
    // No pitch sent: inferred from the data length, rows_per_image only for multi-slice uploads.
    WriteTexturePlan p = plan_write_texture(a, g);
    CHECK(p.status == WriteTexturePlan::Status::Write && p.bytes_per_row == 256 && p.rows_per_image == 0);
    CHECK(p.width == 64 && p.height == 64 && p.depth == 1);
    // Explicit pitch and rows per image.
    a.bytes_per_row = 512; a.rows_per_image = 64; a.data_len = 512 * 64;
    p = plan_write_texture(a, g);
    CHECK(p.status == WriteTexturePlan::Status::Write && p.bytes_per_row == 512 && p.rows_per_image == 64);
    // A region past the mip bounds is clamped (mip 1 is 32x32).
    a = WriteTextureArgs{};
    a.mip = 1; a.ox = 16; a.width = 64; a.height = 64; a.data_len = 64 * 64 * 4; a.bytes_per_row = 256;
    p = plan_write_texture(a, g);
    CHECK(p.status == WriteTexturePlan::Status::Write && p.clamped && p.width == 16 && p.height == 32 && p.mip == 1);
    // Mip out of range / origin outside the mip / empty region.
    a.mip = 7;
    CHECK(plan_write_texture(a, g).status == WriteTexturePlan::Status::BadMip);
    a.mip = 1; a.ox = 40;
    CHECK(plan_write_texture(a, g).status == WriteTexturePlan::Status::Skip);
    a = WriteTextureArgs{};
    CHECK(plan_write_texture(a, g).status == WriteTexturePlan::Status::Skip);

    // BC7 4x4 tail mip of an 8x8 texture: the copy extent is one 4x4 block, 16 bytes per row.
    TextureGeometry bc;
    bc.width = 8; bc.height = 8; bc.mips = 4; bc.format = 62;
    a = WriteTextureArgs{};
    a.mip = 2; a.width = 2; a.height = 2; a.bytes_per_row = 16; a.rows_per_image = 2; a.data_len = 16;
    p = plan_write_texture(a, bc);
    CHECK(p.status == WriteTexturePlan::Status::Write && p.width == 4 && p.height == 4 && p.bytes_per_row == 16);
    CHECK(p.rows_per_image == 1);  // one block row
    // Compressed, no pitch sent: derived from the data (2x2 blocks of 16 bytes = 32 bytes per row).
    a = WriteTextureArgs{};
    a.width = 8; a.height = 8; a.data_len = 64;
    p = plan_write_texture(a, bc);
    CHECK(p.status == WriteTexturePlan::Status::Write && p.bytes_per_row == 32);
    // A compressed pitch below one row of blocks is an error.
    a.bytes_per_row = 16; a.data_len = 128;
    CHECK(plan_write_texture(a, bc).status == WriteTexturePlan::Status::BadRowBytes);
    // 1D (promoted) textures: height 0 means 1.
    TextureGeometry one;
    one.width = 256; one.height = 1; one.dimension = 1; one.requested_1d = true; one.format = 9;  // r16float
    a = WriteTextureArgs{};
    a.width = 256; a.height = 0; a.data_len = 512;
    p = plan_write_texture(a, one);
    CHECK(p.status == WriteTexturePlan::Status::Write && p.height == 1 && p.width == 256);
}

void test_clamps() {
    // clampViewportToTarget: inside, partly outside, depth range swap, no target.
    Viewport v{0, 0, 1280, 720, 0, 1};
    Viewport c = clamp_viewport_to_target(v, 1920, 1080);
    CHECK(c.x == 0 && c.width == 1280 && c.height == 720 && c.min_depth == 0 && c.max_depth == 1);
    v = Viewport{-10, 100, 4000, 4000, 0.9, 0.1};
    c = clamp_viewport_to_target(v, 1920, 1080);
    CHECK(c.x == 0 && c.y == 100 && c.width == 1920 && c.height == 980 && c.min_depth == 0.1 && c.max_depth == 0.9);
    v = Viewport{5, 5, std::nan(""), 10, 0, 1};
    c = clamp_viewport_to_target(v, 0, 0);
    CHECK(c.x == 5 && c.width == 0 && c.height == 10);
    // clampScissorToTarget
    Rect r = clamp_scissor_to_target(10, 10, 100, 100, 640, 480);
    CHECK(r.x == 10 && r.width == 100 && r.height == 100);
    r = clamp_scissor_to_target(600, 400, 100, 100, 640, 480);
    CHECK(r.x == 600 && r.width == 40 && r.y == 400 && r.height == 80);
    r = clamp_scissor_to_target(700, 10, 5, 5, 640, 480);
    CHECK(r.x == 640 && r.width == 0);
    r = clamp_scissor_to_target(1, 2, 3, 4, 0, 0);
    CHECK(r.x == 1 && r.width == 3 && r.height == 4);
}

// ---- milestone 2: read-backs and v1 pipeline rules (m2_model.h) ----

Packet readback_packet(std::vector<uint8_t>& storage, GpuOpcode op, const void* payload, uint32_t bytes) {
    gpu::CommandEncoder enc;
    enc.emit_packet(op, payload, bytes);
    storage.assign(enc.data(), enc.data() + enc.size());
    auto v = walk_all(storage.data(), storage.size());
    return v.empty() ? Packet{} : v[0];
}

void test_readback_decode() {
    GpuReadbackBufferPayload b{};
    b.seq = 7;
    b.buffer_id = 11;
    b.src_offset = 5;
    b.size = 6;
    b.dst_addr = 0x1000;
    b.done_addr = 0x2000;
    std::vector<uint8_t> st;
    Packet p = readback_packet(st, GpuOpcode::ReadbackBuffer, &b, sizeof b);
    auto r = decode_readback(p);
    CHECK(r && r->kind == ReadbackRequest::Kind::Buffer && r->seq == 7 && r->id == 11);
    CHECK(r && r->src_offset == 5 && r->size == 6 && r->dst_addr == 0x1000 && r->done_addr == 0x2000);

    GpuReadbackTexturePayload t{};
    t.seq = 8;
    t.texture_id = 12;
    t.mip_level = 1;
    t.origin_x = 2;
    t.origin_y = 3;
    t.origin_z = 4;
    t.width = 5;
    t.height = 6;
    t.depth_or_layers = 7;
    t.bytes_per_row = 8;
    t.rows_per_image = 9;
    t.dst_addr = 0x3000;
    t.done_addr = 0x4000;
    t.dst_size = 99;
    p = readback_packet(st, GpuOpcode::ReadbackTexture, &t, sizeof t);
    r = decode_readback(p);
    CHECK(r && r->kind == ReadbackRequest::Kind::Texture && r->id == 12 && r->mip_level == 1);
    CHECK(r && r->origin_x == 2 && r->origin_y == 3 && r->origin_z == 4 && r->width == 5 && r->height == 6);
    CHECK(r && r->depth_or_layers == 7 && r->bytes_per_row == 8 && r->rows_per_image == 9);
    CHECK(r && r->dst_addr == 0x3000 && r->done_addr == 0x4000 && r->size == 99);

    // Truncated payloads are the worker's validation error (nullopt).
    p = readback_packet(st, GpuOpcode::ReadbackBuffer, &b, sizeof b - 1);
    CHECK(!decode_readback(p));
    p = readback_packet(st, GpuOpcode::ReadbackTexture, &t, sizeof t - 1);
    CHECK(!decode_readback(p));
    p = readback_packet(st, GpuOpcode::Nop, &b, sizeof b);
    CHECK(!decode_readback(p));
}

void test_readback_layout() {
    // Buffer: the copy starts at the 4-aligned offset below src_offset and ends 4-aligned.
    ReadbackRequest b;
    b.kind = ReadbackRequest::Kind::Buffer;
    b.src_offset = 5;
    b.size = 6;
    ReadbackLayout l = readback_staging_layout(b);
    CHECK(l.valid && l.copy_offset == 4 && l.skip == 1 && l.staging_size == 8);
    b.src_offset = 8;
    b.size = 8;
    l = readback_staging_layout(b);
    CHECK(l.copy_offset == 8 && l.skip == 0 && l.staging_size == 8);
    b.size = 0;
    l = readback_staging_layout(b);
    CHECK(l.staging_size == 4);  // Math.max(4, ...)
    b.src_offset = uint64_t{1} << 60;
    CHECK(!readback_staging_layout(b).valid);

    // Texture: 2x2 rgba8 into a guest footprint of pitch 16 (24 bytes in all): rows of 8 bytes.
    ReadbackRequest t;
    t.kind = ReadbackRequest::Kind::Texture;
    t.width = 2;
    t.height = 2;
    t.depth_or_layers = 1;
    t.bytes_per_row = 16;
    t.rows_per_image = 2;
    t.size = 24;
    l = readback_staging_layout(t);
    CHECK(l.valid && l.rows == 2 && l.depth == 1 && l.row_bytes == 8);
    CHECK(l.staging_row == 256 && l.staging_rows_per_image == 2 && l.staging_size == 512);
    auto pieces = readback_pieces(t, l);
    CHECK(pieces.size() == 2);
    CHECK(pieces.size() == 2 && pieces[0].dst_offset == 0 && pieces[0].src_offset == 0 && pieces[0].length == 8);
    CHECK(pieces.size() == 2 && pieces[1].dst_offset == 16 && pieces[1].src_offset == 256 && pieces[1].length == 8);

    // A block-compressed region reads 4-row blocks: 8x8 bc7 = 2 block rows of 2 blocks (32 bytes).
    t.width = 8;
    t.height = 8;
    t.bytes_per_row = 32;
    t.rows_per_image = 2;
    t.size = 32 * 2;
    l = readback_staging_layout(t, readback_block_height(62, false));
    CHECK(l.rows == 2 && l.row_bytes == 32);
    CHECK(readback_block_height(62, false) == 4 && readback_block_height(62, true) == 1 && readback_block_height(22, false) == 1);
    // 3D: two slices, rows_per_image 3 (one more than the 2 rows read).
    t.width = 2;
    t.height = 2;
    t.depth_or_layers = 2;
    t.bytes_per_row = 16;
    t.rows_per_image = 3;
    t.size = (1 * 3 + 1) * 16 + 8;
    l = readback_staging_layout(t);
    CHECK(l.row_bytes == 8 && l.staging_size == 256 * 2 * 2);
    pieces = readback_pieces(t, l);
    CHECK(pieces.size() == 4 && pieces[2].dst_offset == 3 * 16 && pieces[2].src_offset == 2 * 256);
    // A footprint pitch smaller than the row, or a size that cuts the last row: nothing / what fits.
    t.bytes_per_row = 4;
    t.size = (1 * 3 + 1) * 4 + 8;
    l = readback_staging_layout(t);
    CHECK(readback_pieces(t, l).empty());  // row_bytes (8) > pitch
    t.bytes_per_row = 16;
    t.rows_per_image = 3;
    t.size = (1 * 3 + 1) * 16 + 4;  // the last row's 8 bytes do not fit: row_bytes = 4 < the copy
    l = readback_staging_layout(t);
    CHECK(l.row_bytes == 4);
}

void test_readback_deliver() {
    std::vector<uint8_t> mem(256, 0xEE);
    std::memset(mem.data(), 0, 8);  // two counters
    GuestMemory gm;
    gm.base = mem.data();
    gm.size = mem.size();

    ReadbackRequest b;
    b.kind = ReadbackRequest::Kind::Buffer;
    b.src_offset = 5;
    b.size = 6;
    b.dst_addr = 64;
    b.done_addr = 0;
    const ReadbackLayout l = readback_staging_layout(b);
    const uint8_t staged[8] = {10, 11, 12, 13, 14, 15, 16, 17};  // bytes [4, 12) of the buffer
    CHECK(deliver_readback(gm, b, l, staged, sizeof staged));
    const uint8_t want[6] = {11, 12, 13, 14, 15, 16};  // bytes [5, 11)
    CHECK(std::memcmp(mem.data() + 64, want, 6) == 0);
    CHECK(mem[63] == 0xEE && mem[70] == 0xEE);
    uint32_t counter;
    std::memcpy(&counter, mem.data(), 4);
    CHECK(counter == 1);

    // A failed copy (no mapped bytes) writes nothing and still advances the counter.
    CHECK(!deliver_readback(gm, b, l, nullptr, 0));
    std::memcpy(&counter, mem.data(), 4);
    CHECK(counter == 2);

    // A destination outside the memory: no data, counter advances. An unaligned or outside counter
    // does not move.
    ReadbackRequest far = b;
    far.dst_addr = 255;
    far.done_addr = 4;
    CHECK(!deliver_readback(gm, far, l, staged, sizeof staged));
    std::memcpy(&counter, mem.data() + 4, 4);
    CHECK(counter == 1);
    far.dst_addr = 64;
    far.done_addr = 6;
    CHECK(deliver_readback(gm, far, l, staged, sizeof staged));
    std::memcpy(&counter, mem.data() + 4, 4);
    CHECK(counter == 1);
    far.done_addr = 254;  // 4 bytes do not fit
    CHECK(deliver_readback(gm, far, l, staged, sizeof staged));
    far.dst_addr = ~uint64_t{0} - 2;  // address arithmetic must not wrap
    CHECK(!deliver_readback(gm, far, l, staged, sizeof staged));

    // Nothing attached: nothing at all happens (the worker's `!memory`).
    CHECK(!GuestMemory{}.attached());
    std::memcpy(&counter, mem.data(), 4);
    CHECK(!deliver_readback(GuestMemory{}, b, l, staged, sizeof staged));
    uint32_t after;
    std::memcpy(&after, mem.data(), 4);
    CHECK(after == counter);

    // Host-address mode: dst_addr / done_addr are absolute pointers (the native ReadbackGate's).
    std::vector<uint8_t> host(32, 0xEE);
    uint32_t done = 0;
    GuestMemory abs;
    abs.host_addresses = true;
    ReadbackRequest h = b;
    h.dst_addr = reinterpret_cast<uintptr_t>(host.data());
    h.done_addr = reinterpret_cast<uintptr_t>(&done);
    CHECK(abs.attached());
    CHECK(deliver_readback(abs, h, l, staged, sizeof staged));
    CHECK(std::memcmp(host.data(), want, 6) == 0 && host[6] == 0xEE && done == 1);

    // Texture rows are repacked into the guest pitch; a short mapped range copies what exists.
    ReadbackRequest t;
    t.kind = ReadbackRequest::Kind::Texture;
    t.width = 2;
    t.height = 2;
    t.depth_or_layers = 1;
    t.bytes_per_row = 16;
    t.rows_per_image = 2;
    t.size = 24;
    t.dst_addr = 128;
    t.done_addr = 8;
    std::memset(mem.data() + 8, 0, 4);
    const ReadbackLayout tl = readback_staging_layout(t);
    std::vector<uint8_t> tstage(tl.staging_size, 0);
    for (int i = 0; i < 8; ++i) tstage[i] = static_cast<uint8_t>(1 + i);
    for (int i = 0; i < 8; ++i) tstage[256 + i] = static_cast<uint8_t>(9 + i);
    CHECK(deliver_readback(gm, t, tl, tstage.data(), tstage.size()));
    for (int i = 0; i < 8; ++i) CHECK(mem[128 + i] == 1 + i && mem[144 + i] == 9 + i);
    CHECK(mem[136] == 0xEE && mem[143] == 0xEE);
    std::memcpy(&counter, mem.data() + 8, 4);
    CHECK(counter == 1);
}

void test_v1_vertex_layout() {
    CHECK(!plan_v1_vertex_layout(0).has_buffer);
    auto l = plan_v1_vertex_layout(4);
    CHECK(l.has_buffer && l.stride == 4 && l.components == 1);
    l = plan_v1_vertex_layout(8);
    CHECK(l.components == 2);
    l = plan_v1_vertex_layout(11);
    CHECK(l.components == 2);
    l = plan_v1_vertex_layout(12);
    CHECK(l.components == 3);
    l = plan_v1_vertex_layout(24);
    CHECK(l.stride == 24 && l.components == 3);
}

static void test_negotiated_limits() {
    const LimitSpec& tex = kOptionalLimitSpecs[0];  // maxSampledTexturesPerShaderStage: 16 / cap 256
    CHECK(!negotiated_limit(tex, 16));              // at the default: nothing requested
    CHECK(!negotiated_limit(tex, 8));
    CHECK(negotiated_limit(tex, 17) == 17u);
    CHECK(negotiated_limit(tex, 1000000) == 256u);  // capped
    const LimitSpec& buf = kOptionalLimitSpecs[6];  // maxBufferSize: 256 MiB / cap 1 GiB
    CHECK(!negotiated_limit(buf, 256ull << 20));
    CHECK(negotiated_limit(buf, 4ull << 30) == (1ull << 30));
    CHECK(negotiated_limit(buf, 512ull << 20) == (512ull << 20));
    CHECK(kOptionalLimitCount == 9);
}


void test_layout_remap() {
    // parseWgslGroup0Bindings: comments stripped, both attribute orders, usage filter, kinds.
    const char* wgsl = R"(
// @group(0) @binding(9) var<uniform> commented: vec4<f32>;
@group(0) @binding(0) var<uniform> u: Params;
@binding(1) @group(0) var<storage, read> ro: array<f32>;
@group(0) @binding(2) var<storage, read_write> rw: array<f32>;
@group(0) @binding(3) var smp: sampler;
@group(0) @binding(4) var cmp: sampler_comparison;
@group(0) @binding(5) var t2: texture_2d<f32>;
@group(0) @binding(6) var tu: texture_2d_array<u32>;
@group(0) @binding(7) var td: texture_depth_multisampled_2d;
@group(0) @binding(8) var st: texture_storage_3d<rgba16float, write>;
@group(0) @binding(10) var unused: texture_cube<f32>;
@group(1) @binding(0) var<uniform> other: Params;
@vertex fn vs_main() -> @builtin(position) vec4<f32> {
  let a = u.x + ro[0] + rw[0] + textureSample(t2, smp, vec2<f32>(0.0)).x;
  let b = textureLoad(tu, vec2<i32>(0), 0, 0).x; let c = textureLoad(td, vec2<i32>(0), 0);
  let d = textureSampleCompare(t2, cmp, vec2<f32>(0.0), 0.5); textureStore(st, vec3<i32>(0), vec4<f32>(0.0));
  return vec4<f32>(0.0);
})";
    const auto b = parse_wgsl_group0_bindings(wgsl);
    CHECK(b.size() == 9);
    if (b.size() == 9) {
        CHECK(b[0].binding == 0 && b[0].kind == WgslBindingKind::Uniform);
        CHECK(b[1].kind == WgslBindingKind::ReadOnlyStorage);
        CHECK(b[2].kind == WgslBindingKind::Storage);
        CHECK(b[3].kind == WgslBindingKind::Sampler && !b[3].comparison);
        CHECK(b[4].kind == WgslBindingKind::Sampler && b[4].comparison);
        CHECK(b[5].kind == WgslBindingKind::Texture && b[5].view_dim == 2 && b[5].sample_type == WgslSampleType::Float);
        CHECK(b[6].view_dim == 3 && b[6].sample_type == WgslSampleType::Uint);
        CHECK(b[7].sample_type == WgslSampleType::Depth && b[7].multisampled);
        CHECK(b[8].kind == WgslBindingKind::StorageTexture && b[8].view_dim == 6 && b[8].storage_format == "rgba16float");
    }
    // No function: every declaration counts (the worker's `hasFunction` false branch).
    CHECK(parse_wgsl_group0_bindings("@group(0) @binding(3) var<uniform> z: vec4<f32>;").size() == 1);
    CHECK(parse_wgsl_group0_bindings("@group(1) @binding(3) var<uniform> z: vec4<f32>;").empty());
    CHECK(parse_wgsl_group0_bindings("").empty());
    // A name used only as a member (`.u`) or inside another identifier is not a reference.
    CHECK(parse_wgsl_group0_bindings("@group(0) @binding(0) var<uniform> u: P;\nfn f() { let a = x.u + uu + u_; }").empty());

    // merge: vertex first, a fragment `storage` upgrades a vertex `read-only-storage`.
    std::vector<WgslBinding> vs(2), fs(2);
    vs[0].binding = 0; vs[0].kind = WgslBindingKind::ReadOnlyStorage;
    vs[1].binding = 2; vs[1].kind = WgslBindingKind::Uniform;
    fs[0].binding = 0; fs[0].kind = WgslBindingKind::Storage;
    fs[1].binding = 1; fs[1].kind = WgslBindingKind::Sampler;
    const auto m = merge_wgsl_group0_bindings(vs, fs);
    CHECK(m.size() == 3 && m[0].binding == 0 && m[0].kind == WgslBindingKind::Storage && m[1].binding == 1);

    // explicit layout compatibility
    std::vector<WgslBinding> need(2);
    need[0].binding = 0; need[0].kind = WgslBindingKind::ReadOnlyStorage;
    need[1].binding = 1; need[1].kind = WgslBindingKind::Sampler;
    std::vector<Group0LayoutEntry> have = {{0, 0, 1}, {1, 1, 0}};
    CHECK(!explicit_layout_compatible(&have, need));  // a read-write layout entry does not serve read-only (Dawn)
    have[0].subtype = 2;
    CHECK(explicit_layout_compatible(&have, need));
    have[0].subtype = 0;  // uniform
    CHECK(!explicit_layout_compatible(&have, need));
    CHECK(explicit_layout_compatible(nullptr, need));
    CHECK(explicit_layout_compatible(&have, {}));
    have = {{0, 0, 2}};
    CHECK(!explicit_layout_compatible(&have, need));  // binding 1 missing

    // texture sample types
    CHECK(texture_sample_type(22, true) == WgslSampleType::Float);
    CHECK(texture_sample_type(48, true) == WgslSampleType::Depth);
    CHECK(depth_sample_type_mismatch(0, 48) && depth_sample_type_mismatch(3, 48) && depth_sample_type_mismatch(4, 48));
    CHECK(!depth_sample_type_mismatch(1, 48) && !depth_sample_type_mismatch(2, 48));  // unfilterable-float / depth take it
    CHECK(depth_sample_type_mismatch(2, 22) && !depth_sample_type_mismatch(0, 22));
    // pick_depth_read(layout type, layout ms, format, aspect, samples, dim, float32_filterable)
    CHECK(pick_depth_read(0, false, 48, 0, 1, 1, true) == DepthRead::TwinF32);  // float entry, filterable r32float
    CHECK(pick_depth_read(0, false, 48, 0, 1, 1, false) == DepthRead::TwinF16);  // no float32-filterable: r16float
    CHECK(pick_depth_read(0, false, 48, 0, 4, 1, true) == DepthRead::Dummy);  // non-ms entry vs ms texture
    CHECK(pick_depth_read(0, true, 48, 0, 4, 1, true) == DepthRead::Direct);  // ms float entry is created unfilterable
    CHECK(pick_depth_read(1, true, 49, 1, 4, 1, false) == DepthRead::Direct &&
          pick_depth_read(2, false, 45, 0, 1, 1, false) == DepthRead::Direct);
    CHECK(pick_depth_read(4, false, 48, 0, 1, 1, true) == DepthRead::TwinU32);  // depth plane through uint: raw bits
    CHECK(pick_depth_read(3, false, 45, 1, 1, 1, true) == DepthRead::TwinS32);
    CHECK(pick_depth_read(3, true, 48, 0, 4, 1, true) == DepthRead::Dummy &&
          pick_depth_read(4, true, 48, 0, 4, 1, true) == DepthRead::Dummy);
    CHECK(pick_depth_read(4, false, 47, 2, 1, 1, true) == DepthRead::StencilDirect);  // stencil plane: direct, also ms
    CHECK(pick_depth_read(4, true, 49, 0, 4, 1, true) == DepthRead::StencilDirect);
    CHECK(pick_depth_read(4, false, 49, 1, 1, 1, true) == DepthRead::TwinU32);  // explicit depth aspect: depth bits
    CHECK(pick_depth_read(2, false, 47, 2, 1, 1, true) == DepthRead::Dummy &&
          pick_depth_read(0, false, 47, 2, 1, 1, true) == DepthRead::Dummy);
    CHECK(pick_depth_read(4, false, 48, 2, 1, 1, true) == DepthRead::Dummy);  // depth32float has no stencil plane
    CHECK(pick_depth_read(0, false, 22, 0, 1, 1, true) == DepthRead::Dummy &&
          pick_depth_read(0, false, 48, 0, 1, 2, true) == DepthRead::Dummy);
    CHECK(depth_twin_wgsl(DepthRead::TwinU32, 48).find("bitcast<u32>(d)") != std::string::npos);
    CHECK(depth_twin_wgsl(DepthRead::TwinU32, 45).find("65535.0") != std::string::npos);
    CHECK(depth_twin_wgsl(DepthRead::TwinS32, 47).find("@location(0) i32") != std::string::npos &&
          depth_twin_wgsl(DepthRead::TwinS32, 47).find("bitcast<i32>(u32(round(d * 16777215.0)))") != std::string::npos);
    CHECK(texture_sample_type(25, true) == WgslSampleType::Uint);  // rgba8uint
    CHECK(texture_sample_type(26, true) == WgslSampleType::Sint);  // rgba8sint
    CHECK(texture_sample_type(44, true) == WgslSampleType::Uint);  // stencil8
    CHECK(texture_sample_type(43, true) == WgslSampleType::Float && texture_sample_type(43, false) == WgslSampleType::UnfilterableFloat);

    // picker: pass 0 prefers the exact kind, pass 1 relaxes buffers; dynamic offsets fold in.
    std::vector<RemapResource> av(4);
    av[0].binding = 5; av[0].kind = WgslBindingKind::Storage; av[0].buffer_usage = 0x80 | 0x40;
    av[0].buffer_total = 1024;
    av[1].binding = 6; av[1].kind = WgslBindingKind::Uniform; av[1].buffer_usage = 0x40;
    av[1].buffer_total = 1024; av[1].buffer_offset = 16; av[1].buffer_size = 512;
    av[1].has_dynamic_offset = true; av[1].dynamic_offset_index = 0;
    av[2].binding = 7; av[2].kind = WgslBindingKind::Texture; av[2].format = 22; av[2].view_dim = 2; av[2].multisampled = false;
    av[3].binding = 8; av[3].kind = WgslBindingKind::Sampler; av[3].comparison = false;
    std::vector<uint32_t> dyn = {256};
    RemapPicker pk(av, dyn, true);
    WgslBinding wu; wu.kind = WgslBindingKind::Uniform;
    PickResult r = pk.pick(wu);
    CHECK(r.kind == PickResult::Kind::Resource && r.index == 1 && r.folded && r.offset == 16 + 256 && r.size == 512);
    CHECK(!r.dyn_clamped);
    WgslBinding wr; wr.kind = WgslBindingKind::ReadOnlyStorage;
    r = pk.pick(wr);
    CHECK(r.kind == PickResult::Kind::Resource && r.index == 0 && !r.folded);  // storage serves read-only
    WgslBinding wt; wt.kind = WgslBindingKind::Texture; wt.sample_type = WgslSampleType::Depth;
    CHECK(pk.pick(wt).kind == PickResult::Kind::Dummy);  // a float texture does not serve a depth binding
    wt.sample_type = WgslSampleType::Float;
    CHECK(pk.pick(wt).index == 2);
    WgslBinding ws; ws.kind = WgslBindingKind::Sampler; ws.comparison = true;
    CHECK(pk.pick(ws).kind == PickResult::Kind::FallbackSampler);  // comparison mismatch: default sampler
    ws.comparison = false;
    CHECK(pk.pick(ws).index == 3);
    CHECK(pk.pick(wu).kind == PickResult::Kind::Dummy);  // all buffers used

    // dynamic offset clamping and out-of-bounds
    std::vector<RemapResource> one(1);
    one[0].binding = 0; one[0].kind = WgslBindingKind::Uniform; one[0].buffer_usage = 0x40;
    one[0].buffer_total = 300; one[0].buffer_size = 200;
    one[0].has_dynamic_offset = true;
    std::vector<uint32_t> d2 = {200};
    RemapPicker p2(one, d2, true);
    r = p2.pick(wu);
    CHECK(r.folded && r.dyn_clamped && r.offset == 200 && r.size == 100);
    std::vector<uint32_t> d3 = {300};
    RemapPicker p3(one, d3, true);
    r = p3.pick(wu);
    CHECK(r.kind == PickResult::Kind::Dummy && r.dyn_oob);

    // pick_binding: a buffer bound at the WGSL's number still needs the usage the binding reads it with
    std::vector<RemapResource> pb(1);
    pb[0].binding = 3; pb[0].kind = WgslBindingKind::Storage; pb[0].buffer_usage = 0x80 | 0x08;
    pb[0].buffer_total = 256;
    WgslBinding wb3; wb3.kind = WgslBindingKind::Uniform; wb3.binding = 3;
    RemapPicker pk3(pb, {}, true);
    CHECK(pk3.pick_binding(wb3).kind == PickResult::Kind::Dummy);  // storage-only buffer, uniform binding
    pb[0].buffer_usage = buffer_usage_with_uniform(0x80 | 0x08);
    RemapPicker pk4(pb, {}, true);
    CHECK(pk4.pick_binding(wb3).kind == PickResult::Kind::Resource);
    CHECK(buffer_usage_with_uniform(0x80 | 0x08) == (0x80 | 0x08 | 0x40));
    CHECK(buffer_usage_with_uniform(0x20) == 0x20);  // a vertex buffer stays as it is

    // synthesized layout entries for dummies
    WgslBinding sb; sb.kind = WgslBindingKind::StorageTexture; sb.storage_format = "r32float"; sb.view_dim = 2;
    const SynthLayoutEntry se = synth_layout_entry(sb);
    CHECK(se.kind == 3 && se.storage_format == format_code_by_name("r32float"));
    CHECK(synth_layout_entry(wu).min_binding_size == 4096);
}

void test_blit_model() {
    // Which stream formats need the blit (WgpuSwapchain isValidCanvasFormat).
    CHECK(is_canvas_format_code(22) && is_canvas_format_code(27) && is_canvas_format_code(40));
    CHECK(!is_blit_stream_format_code(22) && !is_blit_stream_format_code(27) && !is_blit_stream_format_code(40));
    CHECK(is_blit_stream_format_code(23) && is_blit_stream_format_code(28) && is_blit_stream_format_code(31) &&
          is_blit_stream_format_code(32) && is_blit_stream_format_code(36));
    CHECK(!is_blit_stream_format_code(0) && !is_blit_stream_format_code(44) && !is_blit_stream_format_code(45) &&
          !is_blit_stream_format_code(49) && !is_blit_stream_format_code(255));
    CHECK(!blit_needed(true, 64, 32, 64, 32));
    CHECK(blit_needed(false, 64, 32, 64, 32));
    CHECK(blit_needed(true, 64, 32, 128, 128));
    // --live-executor (headless): the offscreen canvas is the --window-size, the stream the same size, so a
    // canvas-format stream presents without a blit exactly like the window's fixed-size surface; a 1:1 aspect
    // canvas letterboxes the stream the way the window's client area does.
    CHECK(!blit_needed(true, 1280, 720, 1280, 720));
    CHECK(letterbox_rect(1280, 720, 1280, 720) == (PresentRect{0, 0, 1280, 720}));
    CHECK(letterbox_rect(1280, 720, 1920, 1080) == (PresentRect{0, 0, 1920, 1080}));

    // Letterbox: largest centred rectangle of the stream's aspect ratio.
    CHECK(letterbox_rect(64, 32, 64, 32) == (PresentRect{0, 0, 64, 32}));
    CHECK(letterbox_rect(64, 32, 128, 128) == (PresentRect{0, 32, 128, 64}));    // bars top and bottom
    CHECK(letterbox_rect(64, 32, 100, 20) == (PresentRect{30, 0, 40, 20}));      // limited by height
    CHECK(letterbox_rect(1920, 1080, 3840, 2160) == (PresentRect{0, 0, 3840, 2160}));
    CHECK(letterbox_rect(1920, 1080, 1280, 1024) == (PresentRect{0, 152, 1280, 720}));
    CHECK(letterbox_rect(1280, 720, 1000, 1000) == (PresentRect{0, 218, 1000, 563}));
    CHECK(letterbox_rect(16, 9, 1, 1).w == 1 && letterbox_rect(16, 9, 1, 1).h == 1);  // never an empty rect
    CHECK(letterbox_rect(0, 9, 100, 100) == PresentRect{});
    CHECK(letterbox_rect(16, 9, 0, 100) == PresentRect{});
    // Extreme aspect: the short side clamps to at least one pixel and stays inside the target.
    const PresentRect thin = letterbox_rect(10000, 1, 100, 100);
    CHECK(thin.w == 100 && thin.h == 1 && thin.y == 49);
    const PresentRect big = letterbox_rect(4000000000u, 3000000000u, 4000000000u - 1, 3000000000u);
    CHECK(big.w <= 4000000000u - 1 && big.h <= 3000000000u && big.w >= 1 && big.h >= 1);
    CHECK(std::string(kBlitWgsl).find("textureSample(blit_tex, blit_sampler, in.uv)") != std::string::npos);
}

void test_present_model() {
    PresentChoice c = PresentChoice::Fifo;
    CHECK(parse_present_choice("mailbox", &c) && c == PresentChoice::Mailbox);
    CHECK(parse_present_choice("auto", &c) && c == PresentChoice::Auto);
    CHECK(!parse_present_choice("vsync", &c));
    const uint8_t all = kModeFifo | kModeMailbox | kModeImmediate;
    CHECK(select_present_mode(PresentChoice::Auto, all).actual == PresentChoice::Mailbox);
    CHECK(select_present_mode(PresentChoice::Auto, kModeFifo | kModeImmediate).actual == PresentChoice::Immediate);
    CHECK(select_present_mode(PresentChoice::Auto, 0).actual == PresentChoice::Fifo);
    PresentSelection s = select_present_mode(PresentChoice::Immediate, all);
    CHECK(s.actual == PresentChoice::Immediate && !s.fell_back && s.note.empty());
    s = select_present_mode(PresentChoice::Mailbox, kModeFifo | kModeImmediate);
    CHECK(s.fell_back && s.actual == PresentChoice::Immediate && !s.note.empty());
    s = select_present_mode(PresentChoice::Immediate, kModeFifo);
    CHECK(s.fell_back && s.actual == PresentChoice::Fifo);
    s = select_present_mode(PresentChoice::Fifo, 0);
    CHECK(!s.fell_back && s.actual == PresentChoice::Fifo);
    // 100 fps = 10 ms interval.
    CHECK(frame_cap_wait_ns(0, 5, 6) == 0);
    CHECK(frame_cap_wait_ns(100, 0, 6) == 0);
    CHECK(frame_cap_wait_ns(100, 1000, 4000) == 10000000ull - 3000);
    CHECK(frame_cap_wait_ns(100, 1000, 20000000ull) == 0);
    CHECK(frame_cap_next_anchor(100, 1000, 10001000ull) == 10001000ull);        // on schedule
    CHECK(frame_cap_next_anchor(100, 1000, 100000000ull) == 100000000ull);      // very late: no burst
    CHECK(frame_cap_next_anchor(100, 1000, 15000000ull) == 10001000ull);        // slightly late: keep cadence
}

void test_show_target_model() {
    ShowTargetSpec s;
    std::string err;
    CHECK(parse_show_target("42", &s, &err) && s.kind == ShowTargetSpec::Kind::Tex && s.id == 42 && s.layer == -1 && s.mip == -1);
    CHECK(parse_show_target("tex:7@layer=2@mip=1", &s, &err) && s.id == 7 && s.layer == 2 && s.mip == 1);
    CHECK(parse_show_target("depth:9@range=0.25:0.75", &s, &err) && s.kind == ShowTargetSpec::Kind::Depth && s.lo == 0.25f && s.hi == 0.75f);
    CHECK(parse_show_target("pass:3", &s, &err) && s.kind == ShowTargetSpec::Kind::Pass && s.pass == 3);
    CHECK(parse_show_target("pass:0:depth", &s, &err) && s.kind == ShowTargetSpec::Kind::PassDepth && s.pass == 0);
    CHECK(parse_show_target("last", &s, &err) && s.kind == ShowTargetSpec::Kind::Last);
    CHECK(parse_show_target("cycle", &s, &err) && s.kind == ShowTargetSpec::Kind::Cycle);
    for (const char* bad : {"", "tex:", "0", "tex:0", "depth:x", "pass:", "pass:1:color", "pass:-1", "12@layer", "12@layer=-1",
                            "12@range=1:1", "12@range=1", "12@bogus=1", "foo", "99999999999", "12@mip=x"}) {
        err.clear();
        CHECK(!parse_show_target(bad, &s, &err));
        CHECK(!err.empty());
    }

    // Resolution against a frame: pass 0 colour 10 (mip 1 layer 2), pass 1 depth-only 11, pass 2 swapchain only,
    // pass 3 colour 12 + depth 11.
    std::vector<PassTarget> passes(4);
    passes[0] = {0, 10, 1, 2, 0, 0, 0};
    passes[1] = {1, 0, 0, 0, 11, 0, 0};
    passes[2] = {2, 0, 0, 0, 0, 0, 0};
    passes[3] = {3, 12, 0, 0, 11, 0, 0};
    auto res = [&](const char* sel) {
        ShowTargetSpec sp;
        std::string e;
        CHECK(parse_show_target(sel, &sp, &e));
        return resolve_show_target(sp, passes);
    };
    CHECK(res("pass:0").target && res("pass:0").target->id == 10 && res("pass:0").target->mip == 1 && res("pass:0").target->layer == 2);
    CHECK(res("pass:1").target && res("pass:1").target->id == 11 && res("pass:1").target->depth);
    CHECK(!res("pass:2").target && !res("pass:2").reason.empty());
    CHECK(!res("pass:3:depth").reason.empty() == false && res("pass:3:depth").target->depth);
    CHECK(!res("pass:2:depth").target);
    CHECK(!res("pass:9").target && res("pass:9").reason.find("4 passes") != std::string::npos);
    CHECK(res("last").target && res("last").target->id == 12 && !res("last").target->depth);
    CHECK(res("depth:5").target && res("depth:5").target->depth && res("5").target && !res("5").target->depth);
    CHECK(!res("cycle").target);
    CHECK(!resolve_show_target(ShowTargetSpec{ShowTargetSpec::Kind::Last}, {}).target);

    // Cycle list: distinct, sorted by id, colour before depth; next / previous walk it with an "off" stop.
    const auto c = show_target_candidates(passes);
    CHECK(c.size() == 3);
    CHECK(c.size() == 3 && c[0].id == 10 && c[1].id == 11 && c[1].depth && c[2].id == 12 && !c[2].depth);
    std::optional<TargetRef> cur;
    cur = show_target_cycle(c, cur, true);
    CHECK(cur && cur->id == 10);
    cur = show_target_cycle(c, cur, true);
    CHECK(cur && cur->id == 11 && cur->depth);
    cur = show_target_cycle(c, cur, true);
    CHECK(cur && cur->id == 12);
    cur = show_target_cycle(c, cur, true);
    CHECK(!cur);
    cur = show_target_cycle(c, cur, false);  // previous from off: the last one
    CHECK(cur && cur->id == 12);
    cur = show_target_cycle(c, cur, false);
    CHECK(cur && cur->id == 11);
    CHECK(!show_target_cycle(c, TargetRef{10, false, 0, 0}, false));  // before the first: off
    CHECK(show_target_cycle(c, TargetRef{500, false, 0, 0}, true) == std::nullopt);
    CHECK(show_target_cycle(c, TargetRef{11, false, 0, 0}, true)->id == 11);  // not in the list as colour: the next larger key
    CHECK(!show_target_cycle({}, std::nullopt, true));
    CHECK(show_target_step(0, 3) == 1 && show_target_step(2, 3) == 0 && show_target_step(0, 0) == 0);

    // What can be shown.
    ShowTexture t;
    t.format = 22;  // rgba8unorm
    t.usage = 0x14;
    CHECK(show_target_unsupported(t, false).empty());
    t.usage = 0x10;
    CHECK(show_target_unsupported(t, false).find("TextureBinding") != std::string::npos);
    t.usage = 0x14;
    t.format = 14;  // r32uint
    CHECK(show_target_unsupported(t, false).find("integer") != std::string::npos);
    t.format = 48;  // depth32float
    CHECK(show_target_unsupported(t, false).find("depth") != std::string::npos);
    CHECK(show_target_unsupported(t, true).empty());
    t.sample_count = 4;
    CHECK(show_target_unsupported(t, true).find("multisampled") != std::string::npos);
    t.sample_count = 1;
    t.format = 22;
    CHECK(show_target_unsupported(t, true) == "not a depth format");
    t.dimension = 2;
    CHECK(show_target_unsupported(t, false) == "3d texture");
    t.dimension = 1;
    t.sample_count = 4;
    CHECK(show_target_unsupported(t, false).empty());  // msaa colour is shown (sample 0)

    // Subresource: the selector wins over the attachment's own, both are clamped.
    t.sample_count = 1;
    t.layers = 4;
    t.mips = 3;
    ShowTargetSpec sp;
    sp.layer = -1;
    sp.mip = -1;
    TargetRef ref{1, false, 1, 2};
    ShowSubresource sub = show_target_subresource(sp, ref, t);
    CHECK(sub.mip == 1 && sub.layer == 2 && !sub.clamped);
    sp.layer = 3;
    sp.mip = 0;
    sub = show_target_subresource(sp, ref, t);
    CHECK(sub.mip == 0 && sub.layer == 3 && !sub.clamped);
    sp.layer = 9;
    sp.mip = 7;
    sub = show_target_subresource(sp, ref, t);
    CHECK(sub.mip == 2 && sub.layer == 3 && sub.clamped);
    CHECK(show_target_mip_extent(64, 0) == 64 && show_target_mip_extent(64, 2) == 16 && show_target_mip_extent(3, 5) == 1);

    // The display transform and the shader text.
    CHECK(show_target_map(0.5f, 0, 1) == 0.5f && show_target_map(2.0f, 0, 1) == 1.0f && show_target_map(-1.0f, 0, 1) == 0.0f);
    CHECK(show_target_map(0.5f, 0.5f, 1.5f) == 0.0f && show_target_map(0.5f, 0, 0.5f) == 1.0f);
    CHECK(show_target_wgsl(false).find("texture_2d<f32>") != std::string::npos);
    CHECK(show_target_wgsl(true).find("texture_multisampled_2d<f32>") != std::string::npos);

    // Title text.
    t.width = 1920;
    t.height = 1080;
    t.format = 48;
    sub = ShowSubresource{0, 1, false};
    sp = ShowTargetSpec{};
    const std::string label = show_target_label(TargetRef{7, true, 0, 0}, t, sub, sp);
    CHECK(label.find("tex 7 depth 1920x1080 depth32float") == 0);
    CHECK(label.find("mip 0/3 layer 1/4") != std::string::npos);
    CHECK(show_target_title("base", "") == "base");
    CHECK(show_target_title("base", label).find("base | SHOWING tex 7") == 0);
}

// Legacy fragment stages renumbered on request (gpu::pack_key_code(key, shift)).
void test_shifted_pack_shader() {
    using namespace playgta6::gpu::native_exec;
    const auto s = parse_shifted_pack_shader_payload("wgpk:0x00a5459e8d5891ed;group0_binding_shift=128");
    CHECK(s && s->hash == 0x00a5459e8d5891edull && s->group0_binding_shift == 128);
    CHECK(!parse_hash_only_shader_payload("wgpk:0x00a5459e8d5891ed;group0_binding_shift=128"));
    CHECK(!parse_shifted_pack_shader_payload("0x00a5459e8d5891ed"));
    CHECK(!parse_shifted_pack_shader_payload("wgpk:0x00a5459e8d5891ed;group0_binding_shift=0"));
    CHECK(!parse_shifted_pack_shader_payload("wgpk:0x00a5459e8d5891ed;group0_binding_shift=12x"));
    CHECK(!parse_shifted_pack_shader_payload("wgpk:0xzz;group0_binding_shift=128"));

    const std::string wgsl =
        "@group(0) @binding(0) var<uniform> cb0: CB;\n"
        "@group(0) @binding(1)\nvar tex1: texture_2d<f32>;\n"
        "@binding( 2 ) @group( 0 ) var samp2: sampler;\n"
        "@group(3) @binding(16) var<storage, read> rt: array<u32>;\n"
        "@group(1) @binding(0) var other: sampler;\n"
        "@fragment fn main() -> @location(0) vec4<f32> { return vec4<f32>(0.0); }\n";
    const std::string out = shift_wgsl_group0_bindings(wgsl, 128);
    CHECK(out ==
          "@group(0) @binding(128) var<uniform> cb0: CB;\n"
          "@group(0) @binding(129)\nvar tex1: texture_2d<f32>;\n"
          "@binding( 130 ) @group( 0 ) var samp2: sampler;\n"
          "@group(3) @binding(16) var<storage, read> rt: array<u32>;\n"
          "@group(1) @binding(0) var other: sampler;\n"
          "@fragment fn main() -> @location(0) vec4<f32> { return vec4<f32>(0.0); }\n");
    const auto g0 = parse_wgsl_group0_bindings(out);
    CHECK(g0.size() == 0 || g0.front().binding >= 128);  // unreferenced bindings are not listed
    CHECK(shift_wgsl_group0_bindings("", 128).empty());
}

void test_remap_bindings_match() {
    using namespace playgta6::gpu::native_exec;
    WgslBinding a, b;
    a.binding = 3;
    b.binding = 1;
    CHECK(remap_bindings_match({3}, {a}));
    CHECK(remap_bindings_match({3, 1}, {a, b}));  // order-insensitive
    CHECK(!remap_bindings_match({3, 1}, {a}));     // layout has an entry the remap would not fill
    CHECK(!remap_bindings_match({3}, {a, b}));     // the WGSL binds one the layout lacks
    CHECK(!remap_bindings_match({}, {a}));
    CHECK(remap_bindings_match({}, {}));
}

// Layout fitting: every @group of a WGSL is parsed (group 0 exactly as the remap parses it).
void test_wgsl_group_bindings() {
    using namespace playgta6::gpu::native_exec;
    const char* src = R"(
@group(0) @binding(0) var<uniform> cb0: vec4<u32>;
@group(0) @binding(9) var<storage> sbuf0: array<u32>;
@group(0) @binding(12) var<storage> unused: array<u32>;
@group(3) @binding(0) var<storage, read_write> ms_out: array<u32>;
@compute @workgroup_size(64) fn cs_main() { ms_out[0] = cb0.x + sbuf0[0]; }
)";
    const auto groups = parse_wgsl_group_bindings(src);
    CHECK(groups.size() == 4);  // trimmed after the last used group
    if (groups.size() == 4) {
        CHECK(groups[0] == parse_wgsl_group0_bindings(src));
        CHECK(groups[0].size() == 2);  // binding 12 is not referenced
        CHECK(groups[1].empty() && groups[2].empty());
        CHECK(groups[3].size() == 1 && groups[3][0].binding == 0 && groups[3][0].kind == WgslBindingKind::Storage);
    }
    CHECK(parse_wgsl_group_bindings("@compute @workgroup_size(1) fn main() {}").empty());
    // `var<storage>` is read-only in WGSL: a read-write layout entry does not serve it (Dawn rejects it).
    const auto ro = parse_wgsl_group0_bindings("@group(0) @binding(4) var<storage> s: array<u32>;\nfn f() { _ = s[0]; }");
    CHECK(ro.size() == 1 && ro[0].kind == WgslBindingKind::ReadOnlyStorage);
    const std::vector<Group0LayoutEntry> rw = {{4, 0, 1}}, rd = {{4, 0, 2}};
    CHECK(!explicit_layout_compatible(&rw, ro));
    CHECK(explicit_layout_compatible(&rd, ro));
    const auto g0 = parse_wgsl_group_bindings("@group(0) @binding(1) var t: texture_2d<f32>;\nfn f() { _ = t; }");
    CHECK(g0.size() == 1 && g0[0].size() == 1 && g0[0][0].kind == WgslBindingKind::Texture);
}

// Missing groups: the title's group serves the WGSL where it is compatible, a group the layout lacks
// (or binds differently) is built from the WGSL, a gap below it is padding.
void test_fit_missing_groups() {
    using namespace playgta6::gpu::native_exec;
    WgslBinding u;  // uniform at 0
    WgslBinding out;
    out.kind = WgslBindingKind::Storage;
    const std::vector<std::vector<WgslBinding>> cs = {{u}, {}, {}, {out}};
    const std::vector<FitStage> stages = {{kFitCompute, &cs}};
    std::vector<std::vector<FitBinding>> wg;
    for (size_t g = 0; g < cs.size(); ++g) wg.push_back(merge_fit_group(stages, g));
    CHECK(wg[3].size() == 1 && wg[3][0].visibility == kFitCompute);

    FitExplicitGroup g0;
    g0.entries = std::vector<Group0LayoutEntry>{{0, 0, 0}};
    auto src = fit_layout_groups({g0}, wg, false);
    CHECK(src.size() == 4);
    if (src.size() == 4) {
        CHECK(src[0] == FitSource::Explicit);
        CHECK(src[1] == FitSource::Empty && src[2] == FitSource::Empty);
        CHECK(src[3] == FitSource::Wgsl);
    }
    CHECK(layout_needs_fit(src, 1, false));

    // The title's layout has all four groups and its group 3 matches: nothing to fit.
    FitExplicitGroup empty;
    empty.entries = std::vector<Group0LayoutEntry>{};
    FitExplicitGroup g3;
    g3.entries = std::vector<Group0LayoutEntry>{{0, 0, 1}};
    src = fit_layout_groups({g0, empty, empty, g3}, wg, false);
    CHECK(!layout_needs_fit(src, 4, false));
    // Its group 3 binds a uniform where the WGSL writes storage: rebuilt from the WGSL.
    g3.entries = std::vector<Group0LayoutEntry>{{0, 0, 0}};
    src = fit_layout_groups({g0, empty, empty, g3}, wg, false);
    CHECK(src.size() == 4 && src[3] == FitSource::Wgsl);
    CHECK(layout_needs_fit(src, 4, false));
    // Group 0 the remap binds is always the WGSL's.
    src = fit_layout_groups({g0}, wg, true);
    CHECK(!src.empty() && src[0] == FitSource::Wgsl);
    // Only group 0 used and compatible: nothing to fit.
    const std::vector<std::vector<FitBinding>> only0 = {wg[0]};
    CHECK(!layout_needs_fit(fit_layout_groups({g0}, only0, false), 1, false));
}

// Per-stage limits: a root-signature layout visible to every stage counts its buffers in each stage;
// the WGSL-fitted group counts only what each stage binds.
void test_fit_stage_limits() {
    using namespace playgta6::gpu::native_exec;
    StageLimits lim;  // Dawn D3D12: 12 uniform, 16 storage buffers per stage
    lim.uniform_buffers = 12;
    lim.storage_buffers = 16;
    std::vector<LimitEntry> title;
    for (int i = 0; i < 15; ++i) title.push_back({kFitVertex | kFitFragment, 0, 0});
    CHECK(stage_limit_excess(title, lim) == "uniform buffers 15 > 12 (vertex)");
    std::vector<LimitEntry> storage;
    for (int i = 0; i < 22; ++i) storage.push_back({kFitCompute, 0, 2});  // read-only storage counts as storage
    CHECK(stage_limit_excess(storage, lim) == "storage buffers 22 > 16 (compute)");
    CHECK(stage_limit_excess({{kFitCompute, 3, 0}}, lim).empty());
    std::vector<LimitEntry> tex;
    for (int i = 0; i < 5; ++i) tex.push_back({kFitCompute, 3, 0});
    CHECK(stage_limit_excess(tex, lim) == "storage textures 5 > 4 (compute)");

    // Vertex binds 4 uniforms, fragment 6 (2 shared): fitted, the vertex stage counts 4, fragment 6.
    std::vector<WgslBinding> vs, fs;
    for (uint32_t b = 0; b < 4; ++b) { WgslBinding w; w.binding = b; vs.push_back(w); }
    for (uint32_t b = 2; b < 8; ++b) { WgslBinding w; w.binding = b; fs.push_back(w); }
    const std::vector<std::vector<WgslBinding>> vg = {vs}, fg = {fs};
    const auto merged = merge_fit_group({{kFitVertex, &vg}, {kFitFragment, &fg}}, 0);
    CHECK(merged.size() == 8);
    if (merged.size() == 8) {
        CHECK(merged[0].visibility == kFitVertex);
        CHECK(merged[2].visibility == (kFitVertex | kFitFragment));
        CHECK(merged[7].visibility == kFitFragment);
    }
    std::vector<LimitEntry> fitted;
    for (const FitBinding& f : merged) fitted.push_back(limit_entry_of(f));
    CHECK(stage_limit_excess(fitted, lim).empty());
    // Over the limits: a fit is needed even when every group is the title's.
    CHECK(layout_needs_fit({FitSource::Explicit}, 1, true));

    // A writable storage binding loses the vertex bit when the fragment stage binds it too.
    WgslBinding rw;
    rw.kind = WgslBindingKind::Storage;
    const std::vector<std::vector<WgslBinding>> rwg = {{rw}};
    const auto m2 = merge_fit_group({{kFitVertex, &rwg}, {kFitFragment, &rwg}}, 0);
    CHECK(m2.size() == 1 && m2[0].visibility == kFitFragment);
}

void test_fit_failure_classes() {
    using namespace playgta6::gpu::native_exec;
    CHECK(classify_pipeline_failure("The entry-point uses bindings in group 3 but [PipelineLayout (unlabeled)] doesn't "
                                    "have a BindGroupLayout for this index.") == "missing_group");
    CHECK(classify_pipeline_failure("The number of storage textures (9) in the Compute stage exceeds the maximum "
                                    "per-stage limit (8).") == "stage_limit_storage_textures");
    CHECK(classify_pipeline_failure("The number of uniform buffers (15) in the Vertex stage exceeds the maximum "
                                    "per-stage limit (12).") == "stage_limit_uniform_buffers");
    CHECK(classify_pipeline_failure("[Invalid PipelineLayout (unlabeled)] is invalid due to a previous error.") ==
          "invalid_layout");
    CHECK(classify_pipeline_failure("The buffer type in the shader (BufferBindingType::ReadOnlyStorage) is not "
                                    "compatible with the type in the layout (BufferBindingType::Storage).") ==
          "binding_mismatch");
    CHECK(classify_pipeline_failure("something else") == "other");

    // Unused title groups become padding; used and WGSL-built ones stay.
    WgslBinding u;
    std::vector<FitSource> src = {FitSource::Wgsl, FitSource::Explicit, FitSource::Explicit, FitSource::Wgsl};
    const std::vector<std::vector<FitBinding>> wg = {{FitBinding{u, kFitCompute}}, {}, {FitBinding{u, kFitCompute}},
                                                     {FitBinding{u, kFitCompute}}};
    CHECK(replace_unused_groups(&src, wg) == 0x2);
    CHECK(src[1] == FitSource::Empty && src[2] == FitSource::Explicit && src[0] == FitSource::Wgsl);
}

void test_pipe_wait_budget() {
    using namespace playgta6::gpu::native_exec;
    uint64_t v = 7;
    CHECK(parse_pipe_wait_ms("500", &v) && v == 500ull * 1000000ull);
    CHECK(parse_pipe_wait_ms("0", &v) && v == 0);
    CHECK(parse_pipe_wait_ms("inf", &v) && v == kPipeWaitUnlimited);
    CHECK(parse_pipe_wait_ms("-1", &v) && v == kPipeWaitUnlimited);
    v = 7;
    CHECK(!parse_pipe_wait_ms("abc", &v) && v == 7);
    CHECK(!parse_pipe_wait_ms("-5", &v) && !parse_pipe_wait_ms("", &v) && !parse_pipe_wait_ms(nullptr, &v));

    CHECK(shader_text_droppable(0x1234, false));  // pack-backed: re-readable from the pack
    CHECK(!shader_text_droppable(0, false));      // inline WGSL: the only copy
    CHECK(!shader_text_droppable(0x1234, true));  // pack miss: the hash text is what compiled

    PipeWaitBudget b;  // defaults: 0.5 s per frame, 1 s fullscreen
    CHECK(b.allowance(false) == 500ull * 1000000ull);
    CHECK(b.allowance(true) == 1000ull * 1000000ull);
    b.charge(false, 200ull * 1000000ull);
    CHECK(b.allowance(false) == 300ull * 1000000ull);
    CHECK(b.allowance(true) == 1000ull * 1000000ull);  // separate pools
    b.charge(false, 400ull * 1000000ull);                // overspent: skip, never wrap
    CHECK(b.allowance(false) == 0);
    b.charge(true, 1000ull * 1000000ull);
    CHECK(b.allowance(true) == 0);
    b.begin_frame();  // a new frame refills both
    CHECK(b.allowance(false) == 500ull * 1000000ull && b.allowance(true) == 1000ull * 1000000ull);

    b.configure(0, kPipeWaitUnlimited);  // 0 = never wait, unlimited = the old behaviour
    CHECK(b.allowance(false) == 0);
    b.charge(true, 1ull << 40);
    CHECK(b.allowance(true) == kPipeWaitUnlimited);
}

void test_shader_workers() {
    CHECK(shader_worker_count(24, nullptr) == 4);  // a quarter, at most 4
    CHECK(shader_worker_count(8, "") == 2);
    CHECK(shader_worker_count(2, nullptr) == 1);  // at least one
    CHECK(shader_worker_count(24, "0") == 0);     // PG6_SHADER_WORKERS=0: off
    CHECK(shader_worker_count(24, "7") == 7);
    CHECK(shader_worker_count(24, "x") == 4);     // not a number: the default

    ShaderWorkers w;
    CHECK(!w.running());
    w.start(2);
    CHECK(w.running() && w.size() == 2);
    std::atomic<bool> release{false};
    std::atomic<int> done{0};
    w.submit([&] {
        while (!release.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
        ++done;
    });
    w.submit([&] { ++done; });
    // A bounded wait for an unfinished job times out instead of blocking.
    CHECK(!w.wait([&] { return done.load() == 2; }, 20ull * 1000000ull));
    release = true;
    CHECK(w.wait([&] { return done.load() == 2; }, kPipeWaitUnlimited));
    CHECK(w.wait([&] { return w.completed() == 2; }, 1000ull * 1000000ull));
    CHECK(w.outstanding() == 0);
    // Recovery: jobs not started are dropped, a running one is waited for, the workers stay usable.
    std::atomic<int> started{0};
    release = false;
    for (int i = 0; i < 2; ++i)
        w.submit([&] {
            ++started;
            while (!release.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
            ++done;
        });
    while (started.load() < 2) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    for (int i = 0; i < 3; ++i) w.submit([&] { ++done; });
    std::thread releaser([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        release = true;
    });
    const uint64_t dropped = w.cancel_pending();
    releaser.join();
    CHECK(dropped == 3);  // two workers ran the two blocking jobs; the three queued ones were dropped
    CHECK(done.load() == 4);
    CHECK(w.outstanding() == 0);
    w.submit([&] { ++done; });
    CHECK(w.wait([&] { return done.load() == 5; }, 1000ull * 1000000ull));
    w.stop();
    CHECK(!w.running());
}

void test_fragment_outputs() {
    CHECK(wgsl_fragment_output_mask("@fragment fn fs_main() -> @location(0) vec4f { return vec4f(); }", "fs_main") == 1u);
    CHECK(wgsl_fragment_output_mask("struct O { @location(0) a: vec4f, @location(2) b: vec4f }\n"
                                    "@fragment fn fs_main() -> O { var o: O; return o; }", "fs_main") == 5u);
    CHECK(wgsl_fragment_output_mask("@fragment fn fs_main() {}", "fs_main") == 0u);
    CHECK(!wgsl_fragment_output_mask("@fragment fn fs_main() -> Unknown { }", "fs_main").has_value());
    CHECK(wgsl_fragment_output_mask("@fragment fn a() -> @location(1) vec4f { return vec4f(); }\n"
                                    "@fragment fn b() -> @location(3) vec4f { return vec4f(); }", "b") == 8u);
}

void test_error_scope_model() {
    CHECK(std::string(error_scope_phase(static_cast<uint16_t>(GpuOpcode::CreateRenderPipeline2))) == "pipeline_create");
    CHECK(std::string(error_scope_phase(static_cast<uint16_t>(GpuOpcode::CreateBindGroup2))) == "bind_group_create");
    CHECK(std::string(error_scope_phase(static_cast<uint16_t>(GpuOpcode::Draw))) == "pass");
    CHECK(std::string(error_scope_phase(static_cast<uint16_t>(GpuOpcode::Present))) == "present_submit");
    CHECK(std::string(error_scope_phase(static_cast<uint16_t>(GpuOpcode::WriteBuffer))) == "other");
    // off until an unattributed device error arms it; armed for a window of presents after the latest one;
    // explicit stays on; auto disarms after the error budget
    CHECK(!error_scope_active(false, 0, 0, 0, 0));
    CHECK(!error_scope_active(false, 0, 0, 5, 0));
    CHECK(error_scope_active(false, 1, 10, 10, 0));
    CHECK(error_scope_active(false, 1, 10, 10 + kErrorScopeAutoPresents, 0));
    CHECK(!error_scope_active(false, 1, 10, 11 + kErrorScopeAutoPresents, 0));
    CHECK(!error_scope_active(false, 5, 10, 10, kErrorScopeAutoErrors));
    CHECK(error_scope_active(true, 0, 0, 1000, kErrorScopeAutoErrors + 10));
    // dedupe by (message, phase, context); bounded entries, counts by message
    ErrorScopeLog log;
    log.record("validation", "bad \"thing\"", "pipeline_create", "rp1(vs=a fs=b)", 3, 10, 24);
    log.record("validation", "bad \"thing\"", "pipeline_create", "rp1(vs=a fs=b)", 4, 11, 24);
    log.record("validation", "bad \"thing\"", "present_submit", "rp1(vs=a fs=b)", 4, 12, 7);
    CHECK(log.total() == 3);
    CHECK(log.entries().size() == 2);
    CHECK(log.entries()[0].count == 2 && log.entries()[0].present_index == 3);
    CHECK(log.by_message().at("bad \"thing\"") == 3);
    CHECK(log.json().find("\\\"thing\\\"") != std::string::npos);
    for (int i = 0; i < 200; ++i) log.record("validation", "m" + std::to_string(i), "pass", "", 0, i, 6);
    CHECK(log.entries().size() == kErrorScopeMaxEntries);
    CHECK(log.entries_dropped() > 0);
    CHECK(log.by_message().size() <= kErrorScopeMaxMessages);
    CHECK(log.total() == 203);
}

void test_view_alias_model() {
    // view dimension vs the entry's (6-layer 2d texture)
    EntryViewFit f = fit_view_to_entry(1, 6, 5, 0, 6, 2);  // cube-array view, 2d entry
    CHECK(f.fit == ViewFit::Coerced && f.dim == 2 && f.layer_count == 1 && f.base_layer == 0);
    f = fit_view_to_entry(1, 6, 4, 0, 6, 3);  // cube view, 2d-array entry
    CHECK(f.fit == ViewFit::Coerced && f.dim == 3 && f.layer_count == 6);
    f = fit_view_to_entry(1, 6, 3, 0, 0, 4);  // all-layers array view, cube entry
    CHECK(f.fit == ViewFit::Coerced && f.dim == 4 && f.layer_count == 6);
    f = fit_view_to_entry(1, 12, 3, 0, 0, 5);  // 12 layers -> cube-array of 12
    CHECK(f.fit == ViewFit::Coerced && f.dim == 5 && f.layer_count == 12);
    f = fit_view_to_entry(1, 8, 3, 0, 0, 5);  // 8 layers -> cube-array of 6
    CHECK(f.fit == ViewFit::Coerced && f.layer_count == 6);
    f = fit_view_to_entry(1, 2, 3, 0, 0, 4);  // cube entry, 2 layers: no
    CHECK(f.fit == ViewFit::Unfixable);
    f = fit_view_to_entry(1, 6, 3, 2, 0, 4);  // 4 layers remain: no
    CHECK(f.fit == ViewFit::Unfixable);
    f = fit_view_to_entry(1, 6, 0, 0, 0, 3);  // default view of a layered texture is 2d-array already
    CHECK(f.fit == ViewFit::Same);
    f = fit_view_to_entry(1, 1, 0, 0, 0, 2);  // default view of a single layer is 2d
    CHECK(f.fit == ViewFit::Same);
    f = fit_view_to_entry(1, 6, 0, 0, 0, 2);  // default 2d-array view, 2d entry: slice 0
    CHECK(f.fit == ViewFit::Coerced && f.dim == 2 && f.layer_count == 1);
    f = fit_view_to_entry(2, 1, 6, 0, 0, 2);  // a 3d texture is never touched
    CHECK(f.fit == ViewFit::Same);
    f = fit_view_to_entry(1, 6, 5, 0, 6, 6);  // neither is a 3d entry
    CHECK(f.fit == ViewFit::Same);

    // view range vs the texture's layers / mips
    ViewRangeFit r = fit_view_range(1, 1, 1, 3, 0, 1, 0, 4);  // 2d-array view of 4 layers over 1
    CHECK(r.fit == RangeFit::Clamped && r.layers_clamped && !r.mips_clamped && r.layer_count == 1);
    r = fit_view_range(1, 6, 1, 3, 0, 0, 2, 0);  // all remaining: untouched
    CHECK(r.fit == RangeFit::Same && r.base_layer == 2 && r.layer_count == 0);
    r = fit_view_range(1, 8, 1, 3, 0, 1, 2, 8);  // layers 2..9 over 8: 2..7
    CHECK(r.fit == RangeFit::Clamped && r.base_layer == 2 && r.layer_count == 6);
    r = fit_view_range(1, 1, 3, 2, 1, 5, 0, 1);  // mips 1..5 over 3: 1..2
    CHECK(r.fit == RangeFit::Clamped && r.mips_clamped && r.base_mip == 1 && r.mip_count == 2);
    r = fit_view_range(1, 4, 1, 3, 0, 1, 4, 1);  // starts past the last layer
    CHECK(r.fit == RangeFit::Unfixable && r.base_layer == 4 && r.layer_count == 1);
    r = fit_view_range(1, 1, 2, 2, 2, 1, 0, 1);  // starts past the last mip
    CHECK(r.fit == RangeFit::Unfixable);
    r = fit_view_range(1, 4, 1, 4, 0, 1, 0, 6);  // a cube over 4 layers
    CHECK(r.fit == RangeFit::Unfixable);
    r = fit_view_range(1, 14, 1, 5, 0, 1, 0, 18);  // cube-array of 3 cubes over 14 layers: 2 cubes
    CHECK(r.fit == RangeFit::Clamped && r.layer_count == 12);
    r = fit_view_range(2, 8, 1, 6, 0, 1, 0, 0);  // a 3d texture's depth is not layers: default view is fine
    CHECK(r.fit == RangeFit::Same);
    r = fit_view_range(2, 8, 1, 6, 0, 1, 0, 8);  // ... and a layer count over it is one layer
    CHECK(r.fit == RangeFit::Clamped && r.layer_count == 1);
    r = fit_view_range(1, 6, 4, 4, 0, 4, 0, 6);  // inside
    CHECK(r.fit == RangeFit::Same);

    // buffer aliasing
    auto bind = [](uint64_t buf, uint32_t g, uint32_t b, uint64_t off, uint64_t size, bool w) {
        AliasBinding a;
        a.buffer = buf;
        a.group = g;
        a.binding = b;
        a.offset = off;
        a.size = size;
        a.writable = w;
        return a;
    };
    AliasPlan p = plan_alias_copies({bind(1, 0, 0, 0, 64, true), bind(1, 0, 1, 0, 64, false)});
    CHECK(p.copy.size() == 1 && p.copy[0] == 1 && p.unfixable == 0);
    p = plan_alias_copies({bind(1, 0, 0, 0, 64, false), bind(1, 1, 0, 32, 64, true)});  // across groups, order swapped
    CHECK(p.copy.size() == 1 && p.copy[0] == 0 && p.unfixable == 0);
    p = plan_alias_copies({bind(1, 0, 0, 0, 64, true), bind(1, 1, 0, 64, 64, false)});  // adjacent, no overlap
    CHECK(p.copy.empty() && p.unfixable == 0);
    p = plan_alias_copies({bind(1, 0, 0, 0, 64, true), bind(2, 1, 0, 0, 64, false)});  // other buffer
    CHECK(p.copy.empty());
    p = plan_alias_copies({bind(1, 0, 0, 0, 64, false), bind(1, 1, 0, 0, 64, false)});  // read-only twice is fine
    CHECK(p.copy.empty() && p.unfixable == 0);
    p = plan_alias_copies({bind(1, 0, 0, 0, 64, true), bind(1, 1, 0, 0, 64, true)});  // two writers: the later one twins
    CHECK(p.copy.empty() && p.twin.size() == 1 && p.twin[0] == 1 && p.unfixable == 0);
    p = plan_alias_copies({bind(1, 0, 0, 0, 64, true), bind(1, 0, 1, 64, 64, true), bind(1, 1, 0, 32, 64, true)});
    CHECK(p.twin.size() == 1 && p.twin[0] == 2);  // 0 and 1 are disjoint: both stay on the buffer
    p = plan_alias_copies({bind(1, 0, 0, 0, 64, true), bind(2, 0, 1, 0, 64, true)});  // other buffer
    CHECK(p.twin.empty());
    p = plan_alias_copies({bind(1, 0, 0, 0, 64, true), bind(1, 0, 1, 0, 16, false), bind(1, 1, 0, 8, 16, false)});
    CHECK(p.copy.size() == 2 && p.copy[0] == 1 && p.copy[1] == 2);

    p = plan_alias_copies({bind(1, 0, 0, 0, 1 << 20, false), bind(1, 0, 1, 512, 32, true)});  // big reader, tiny writer
    CHECK(p.copy.empty() && p.twin.size() == 1 && p.twin[0] == 1);
    p = plan_alias_copies({bind(1, 0, 0, 0, 1 << 20, false), bind(1, 0, 1, 0, 1 << 19, true)});  // comparable: copy the reader
    CHECK(p.copy.size() == 1 && p.twin.empty());

    // texture aliasing
    auto tb = [](uint64_t tex, uint32_t mip, uint32_t mips, uint32_t layer, uint32_t layers, bool w) {
        AliasTexBinding t;
        t.texture = tex;
        t.base_mip = mip;
        t.mip_count = mips;
        t.base_layer = layer;
        t.layer_count = layers;
        t.writable = w;
        return t;
    };
    TexAliasPlan tp = plan_texture_alias({tb(1, 0, 1, 0, 1, true), tb(1, 0, 1, 0, 1, false)});
    CHECK(tp.copy.size() == 1 && tp.copy[0] == 1 && tp.unfixable == 0);
    tp = plan_texture_alias({tb(1, 0, 1, 0, 1, false), tb(1, 0, 4, 0, 1, true)});  // writer's mips cover the reader's
    CHECK(tp.copy.size() == 1 && tp.copy[0] == 0);
    tp = plan_texture_alias({tb(1, 0, 1, 0, 1, true), tb(1, 1, 1, 0, 1, false)});  // other mip
    CHECK(tp.copy.empty() && tp.unfixable == 0);
    tp = plan_texture_alias({tb(1, 0, 1, 0, 2, true), tb(1, 0, 1, 2, 2, false)});  // other layers
    CHECK(tp.copy.empty());
    tp = plan_texture_alias({tb(1, 0, 1, 0, 1, true), tb(2, 0, 1, 0, 1, false)});  // other texture
    CHECK(tp.copy.empty());
    tp = plan_texture_alias({tb(1, 0, 1, 0, 1, false), tb(1, 0, 1, 0, 1, false)});  // two readers
    CHECK(tp.copy.empty() && tp.unfixable == 0);
    tp = plan_texture_alias({tb(1, 0, 1, 0, 1, true), tb(1, 0, 1, 0, 1, true)});  // two writers: counted
    CHECK(tp.copy.empty() && tp.unfixable == 1);
}

void test_binding_fit() {
    // whole 128 KiB uniform buffer -> cut to 64 KiB; a 1 KiB range stays as is
    BufferBindingFit f = fit_buffer_binding(false, 131072, 0, 0, 0, 65536);
    CHECK(f.ok && f.size == 65536 && f.clamped);
    f = fit_buffer_binding(false, 131072, 256, 1024, 0, 65536);
    CHECK(f.ok && f.size == 1024 && !f.clamped);
    // requested past the end -> the remainder; offset past the end -> no fit
    f = fit_buffer_binding(false, 4096, 3840, 1024, 0, 65536);
    CHECK(f.ok && f.size == 256);
    CHECK(!fit_buffer_binding(false, 4096, 4096, 0, 0, 65536).ok);
    // storage: a multiple of 4
    f = fit_buffer_binding(true, 1030, 0, 0, 0, 1u << 27);
    CHECK(f.ok && f.size == 1028 && f.clamped);
    CHECK(!fit_buffer_binding(true, 2, 0, 0, 0, 1u << 27).ok);
    // under minBindingSize: grown when the buffer has room, else no fit
    f = fit_buffer_binding(false, 4096, 0, 64, 256, 65536);
    CHECK(f.ok && f.size == 256 && f.extended && !f.padded);
    CHECK(!fit_buffer_binding(false, 128, 0, 0, 256, 65536).ok);
    // ... with allow_pad (the remap of a compacted stage's block): a zero-padded twin of min bytes holding
    // the rest of the buffer, the copy in whole words
    f = fit_buffer_binding(false, 128, 0, 0, 256, 65536, true);
    CHECK(f.ok && f.padded && f.size == 256 && f.data_size == 128);
    f = fit_buffer_binding(false, 4096, 3968, 64, 912, 65536, true);  // a 64-byte view near the end, a 912-byte block
    CHECK(f.ok && f.padded && f.size == 912 && f.data_size == 128);
    f = fit_buffer_binding(true, 1030, 1024, 0, 64, 1u << 27, true);  // storage: 6 bytes left -> 4 copied
    CHECK(f.ok && f.padded && f.size == 64 && f.data_size == 4);
    CHECK(!fit_buffer_binding(false, 4096, 4094, 0, 256, 65536, true).ok);  // nothing whole to copy
    CHECK(!fit_buffer_binding(false, 4096, 0, 0, 131072, 65536, true).ok);  // minimum over the device limit
    f = fit_buffer_binding(false, 4096, 0, 64, 256, 65536, true);  // room in the buffer: extended, no twin
    CHECK(f.ok && f.extended && !f.padded && f.size == 256);
    // explicit bind groups (CreateBindGroup2 / manifest layouts / v1 over a reflected group): a 16-byte
    // uniform view against a 48-byte minimum, near the end of the buffer and at it
    f = fit_buffer_binding(false, 4096, 4000, 16, 48, 65536, true);  // 96 bytes left: extended in place
    CHECK(f.ok && f.extended && !f.padded && f.size == 48);
    f = fit_buffer_binding(false, 4096, 4064, 16, 48, 65536, true);  // 32 bytes left: a 48-byte twin of 32 data bytes
    CHECK(f.ok && f.padded && f.size == 48 && f.data_size == 32);
    f = fit_buffer_binding(true, 4096, 4088, 8, 50, 1u << 27, true);  // storage: min rounded up to whole words
    CHECK(f.ok && f.padded && f.size == 52 && f.data_size == 8);
    // size unknown: unchanged
    f = fit_buffer_binding(false, 0, 0, 0, 0, 65536);
    CHECK(f.ok && f.size == 0);
    // integer formats vs sample types (0 float, 1 unfilterable, 2 depth, 3 sint, 4 uint)
    CHECK(integer_sample_type_mismatch(0, 14));   // r32uint on float
    CHECK(integer_sample_type_mismatch(1, 15));   // r32sint on unfilterable-float
    CHECK(!integer_sample_type_mismatch(4, 14));  // r32uint on uint
    CHECK(integer_sample_type_mismatch(3, 14));   // r32uint on sint
    CHECK(integer_sample_type_mismatch(4, 22));   // rgba8unorm on uint
    CHECK(!integer_sample_type_mismatch(0, 22));  // rgba8unorm on float
    CHECK(!integer_sample_type_mismatch(1, 16));  // r32float on unfilterable-float
    // unfilterable-float-only views on a float entry: norm16 always, float32 without float32-filterable
    CHECK(unfilterable_float_mismatch(0, format_code_by_name("rg16unorm"), true));
    CHECK(unfilterable_float_mismatch(0, format_code_by_name("r16snorm"), true));
    CHECK(unfilterable_float_mismatch(0, format_code_by_name("rgba16unorm"), false));
    CHECK(!unfilterable_float_mismatch(1, format_code_by_name("rg16unorm"), true));  // unfilterable entry takes it
    CHECK(!unfilterable_float_mismatch(0, 22, true));                                   // rgba8unorm
    CHECK(!unfilterable_float_mismatch(0, format_code_by_name("depth16unorm"), true));  // depth: pick_depth_read
    CHECK(unfilterable_float_mismatch(0, 16, false) && !unfilterable_float_mismatch(0, 16, true));  // r32float
    CHECK(norm16_components(format_code_by_name("r16snorm")) == 1 && norm16_components(format_code_by_name("rg16unorm")) == 2 &&
          norm16_components(format_code_by_name("rgba16unorm")) == 4 && norm16_components(22) == 0 && norm16_components(16) == 0);
}

void test_device_error_kinds() {
    // numbers fold (decimal and 0x hex), newlines collapse; words with digits after a letter keep the letter
    CHECK(normalize_device_error("The number of storage buffers (26) in the Compute stage exceeds the maximum per-stage limit (16).\n"
                                 " - While validating binding counts\n") ==
          "The number of storage buffers (N) in the Compute stage exceeds the maximum per-stage limit (N). - While "
          "validating binding counts");
    CHECK(normalize_device_error("offset 0x1F0 texture_2d") == "offset N texture_Nd");
    DeviceErrorKinds k;
    k.record(31, "limit (13) exceeded");
    k.record(31, "limit (29) exceeded");
    k.record(7, "limit (29) exceeded");
    k.record(kDeviceErrorWorkerOpcode, "worker thing", 5);
    CHECK(k.total() == 8);
    CHECK(k.count(31, "limit (1) exceeded") == 2);
    CHECK(k.count(7, "limit (1) exceeded") == 1);
    const std::string j = k.json();
    CHECK(j.find("{\"count\": 5, \"opcode\": 65535, \"cascade\": false, \"text\": \"worker thing\"}") < j.find("\"opcode\": 31"));
    for (int i = 0; i < 100; ++i) k.record(9, "kind " + std::string(1, static_cast<char>('A' + i % 26)) + std::string(1, static_cast<char>('a' + i / 26)));
    CHECK(k.total() == 108);
    CHECK(j.size() < k.json().size());
    {
        DeviceErrorKinds c;
        c.record(1, "root failure", 2);
        c.record(1, "Entry point pg6_missing_entry doesn't exist", 3);
        c.record(2, "[Invalid CommandBuffer \"x\"] is invalid due to a previous error.");
        c.record(2, "[Invalid BindGroup] is invalid.");
        c.record(2, "[Invalid BindGroupLayout] is invalid.");
        c.record(2, "[Invalid PipelineLayout] is invalid.");
        CHECK(c.root_total() == 2 && c.cascade_total() == 7 && c.total() == 9);
        const std::string cj = c.json();
        CHECK(cj.find("\"cascade_total\": 7") != std::string::npos);
        CHECK(cj.find("\"cascade\": true, \"text\": \"Entry point pgN_missing_entry") != std::string::npos);
        CHECK(cj.find("\"cascade\": false, \"text\": \"root failure\"") != std::string::npos);
    }
    CHECK(k.json().find("\"other\": 0") == std::string::npos);  // past the row cap the rest is "other"
}

// Bytes the copies zero per (mip, slice, row) texel row: every pending texel exactly once.
uint64_t copied_bytes(const std::vector<ZeroCopy>& copies) {
    uint64_t n = 0;
    for (const ZeroCopy& c : copies) n += c.bytes;
    return n;
}

void test_adapter_model() {
    const AdapterIdentity gpu{"d3d12", 0x10de, 0x2783, false, "NVIDIA GeForce RTX 4070 SUPER"};
    const AdapterIdentity warp{"d3d12", 0x1414, 0x8c, true, "Microsoft Basic Render Driver"};
    CHECK(same_adapter(gpu, gpu));
    CHECK(!same_adapter(gpu, warp));  // the CPU adapter a new instance hands out while the GPU is not open
    AdapterIdentity other = gpu;
    other.description = "renamed";
    CHECK(same_adapter(gpu, other));  // the description is for the log only
    other.backend = "vulkan";
    CHECK(!same_adapter(gpu, other));
    CHECK(same_adapter(warp, warp));  // --adapter warp runs stay on WARP
}
void test_gpu_stall_watch() {
    constexpr uint64_t ms = 1000000ull;
    GpuStallWatch w(8000 * ms);
    CHECK(!w.sample(0, 0, 1 * ms));            // idle
    CHECK(!w.sample(3, 0, 10 * ms));           // work pending: the clock starts
    CHECK(!w.sample(3, 1, 5000 * ms));         // progress restarts it
    CHECK(!w.sample(3, 1, 12000 * ms));        // 7 s without progress: not yet
    CHECK(w.stalled_ns(12000 * ms) == 7000 * ms);
    CHECK(w.sample(3, 1, 13001 * ms));         // > 8 s: remove the device
    CHECK(!w.sample(3, 1, 20000 * ms));        // once per stall
    CHECK(!w.sample(3, UINT64_MAX, 20100 * ms));  // removed: the fence reports UINT64_MAX, idle again
    CHECK(w.stalled_ns(20100 * ms) == 0);
    CHECK(!w.sample(5, 3, 30000 * ms));        // a new stall needs its own 8 s
    CHECK(!w.sample(5, 3, 37000 * ms));
    CHECK(w.sample(5, 3, 38001 * ms));
    w.reset();
    CHECK(!w.sample(1, 0, 40000 * ms));
    GpuStallWatch off(0);
    CHECK(!off.sample(1, 0, 0));
    CHECK(!off.sample(1, 0, UINT64_MAX / 2));
    CHECK(gpu_stall_threshold_ns(nullptr) == kDefaultGpuStallNs);
    CHECK(gpu_stall_threshold_ns("0") == 0);
    CHECK(gpu_stall_threshold_ns("2500") == 2500 * ms);
    CHECK(gpu_stall_threshold_ns("x") == kDefaultGpuStallNs);
    CHECK(gpu_stall_threshold_ns("-1") == kDefaultGpuStallNs);
}
void test_clear_model() {
    // 8x8 rgba8 with 4 mips, nothing written: every texel of every mip, one copy per mip.
    ClearGeometry g;
    g.width = 8;
    g.height = 8;
    g.mips = 4;
    PendingClear pc;
    pc.reset(g);
    CHECK(pc.pending() && pc.pending_subresources() == 4);
    std::vector<ZeroCopy> copies;
    pc.zero_copies(1u << 20, &copies);
    CHECK(copies.size() == 4);
    CHECK(copied_bytes(copies) == (64 + 16 + 4 + 1) * 4);
    for (const ZeroCopy& c : copies) {
        CHECK(c.bytes_per_row == 256 && c.y == 0 && c.z == 0 && c.depth == 1);
        CHECK(c.width == (8u >> c.mip) && c.height == (8u >> c.mip) && c.rows_per_image == c.height);
    }

    // A full write of mip 0 and the first band of mip 1: mip 0 needs no copy, mip 1 only its last rows.
    CHECK(pc.note_write(0, 0, 0, 0, 8, 8, 1));
    CHECK(pc.note_write(1, 0, 0, 0, 4, 2, 1));
    CHECK(pc.note_write(1, 0, 2, 0, 4, 1, 1));  // the next band extends the covered rows
    copies.clear();
    pc.zero_copies(1u << 20, &copies);
    CHECK(copies.size() == 3);
    CHECK(copies[0].mip == 1 && copies[0].y == 3 && copies[0].height == 1);
    CHECK(copied_bytes(copies) == (4 + 4 + 1) * 4);
    // Inside the covered rows: fine. A partial-width write or one past the covered rows: needs the clear first.
    CHECK(pc.note_write(1, 1, 0, 0, 2, 2, 1));
    CHECK(!pc.note_write(1, 0, 3, 0, 2, 1, 1));
    CHECK(!pc.note_write(2, 0, 1, 0, 2, 1, 1));
    // Every texel written: nothing left to copy.
    PendingClear all;
    all.reset(g);
    for (uint32_t m = 0; m < 4; ++m) CHECK(all.note_write(m, 0, 0, 0, 8u >> m, 8u >> m, 1));
    CHECK(!all.pending());
    copies.clear();
    all.zero_copies(1u << 20, &copies);
    CHECK(copies.empty());

    // BC7 (4x4 blocks, 16 B) 10x6, 2 mips: physical extents 12x8 and 8x4 (block-rounded).
    ClearGeometry bc;
    bc.width = 10;
    bc.height = 6;
    bc.mips = 2;
    bc.block_w = 4;
    bc.block_h = 4;
    bc.block_bytes = 16;
    PendingClear pb;
    pb.reset(bc);
    copies.clear();
    pb.zero_copies(1u << 20, &copies);
    CHECK(copies.size() == 2);
    CHECK(copies[0].width == 12 && copies[0].height == 8 && copies[0].rows_per_image == 2);
    CHECK(copies[1].width == 8 && copies[1].height == 4 && copies[1].rows_per_image == 1);
    CHECK(copied_bytes(copies) == 3 * 2 * 16 + 2 * 1 * 16);
    CHECK(pb.note_write(0, 0, 0, 0, 12, 4, 1));  // one block row
    copies.clear();
    pb.zero_copies(1u << 20, &copies);
    CHECK(copies.size() == 2 && copies[0].y == 4 && copies[0].height == 4 && copies[0].rows_per_image == 1);

    // 2D array of 6 layers 16x16 rgba8: untouched layers share a copy; a written layer is skipped.
    ClearGeometry arr;
    arr.width = 16;
    arr.height = 16;
    arr.depth_or_layers = 6;
    PendingClear pa;
    pa.reset(arr);
    CHECK(pa.note_write(0, 0, 0, 2, 16, 16, 1));  // layer 2 written whole
    copies.clear();
    pa.zero_copies(1u << 20, &copies);
    CHECK(copies.size() == 2);
    CHECK(copies[0].z == 0 && copies[0].depth == 2 && copies[1].z == 3 && copies[1].depth == 3);
    CHECK(copied_bytes(copies) == 5ull * 16 * 16 * 4);
    // A small zero buffer splits the layers, then the rows: 16 rows of 256 B = 4 KiB per layer.
    copies.clear();
    pa.zero_copies(4096, &copies);
    CHECK(copies.size() == 5 && copied_bytes(copies) == 5ull * 16 * 16 * 4);
    copies.clear();
    pa.zero_copies(1024, &copies);  // 4 rows per copy
    CHECK(copies.size() == 20 && copied_bytes(copies) == 5ull * 16 * 16 * 4);
    for (const ZeroCopy& c : copies) CHECK(uint64_t{c.bytes_per_row} * c.rows_per_image * c.depth <= 1024);

    // 3D 8x8x8, 2 mips: slices are the unit; full-slice writes from the front shrink the copy.
    ClearGeometry vol;
    vol.width = 8;
    vol.height = 8;
    vol.depth_or_layers = 8;
    vol.mips = 2;
    vol.dimension = 2;
    PendingClear pv;
    pv.reset(vol);
    CHECK(pv.note_write(0, 0, 0, 0, 8, 8, 3));
    CHECK(!pv.note_write(0, 0, 0, 5, 8, 8, 1));  // a gap: clear first
    copies.clear();
    pv.zero_copies(1u << 20, &copies);
    CHECK(copies.size() == 2);
    CHECK(copies[0].mip == 0 && copies[0].z == 3 && copies[0].depth == 5);
    CHECK(copies[1].mip == 1 && copies[1].z == 0 && copies[1].depth == 4 && copies[1].width == 4);
    CHECK(copied_bytes(copies) == 5ull * 64 * 4 + 4ull * 16 * 4);
}

void test_usage_scope_model() {
    int buf_a = 0, buf_b = 0, tex = 0;
    auto buf = [](const void* r, UseKind k, uint64_t off, uint64_t size, int32_t dyn = -1) {
        ResUse u;
        u.res = r;
        u.kind = k;
        u.offset = off;
        u.size = size;
        u.dyn_index = dyn;
        return u;
    };
    auto img = [](const void* r, UseKind k, uint32_t mip, uint32_t mips, uint32_t layer = 0, uint32_t layers = 1) {
        ResUse u;
        u.res = r;
        u.texture = true;
        u.kind = k;
        u.mip_base = mip;
        u.mip_count = mips;
        u.layer_base = layer;
        u.layer_count = layers;
        return u;
    };
    // one group: uniform [0, 16) + storage [256, 272) of one buffer -> a snapshot, not volatile
    BgUses g0;
    g0.add(buf(&buf_a, UseKind::Uniform, 0, 16));
    g0.add(buf(&buf_a, UseKind::StorageWrite, 256, 16));
    CHECK(g0.any_writable);
    std::vector<BoundUses> bound{{0, &g0, nullptr}};
    UsageScope scope;
    BindPlan p = plan_binds(bound, scope, false);
    CHECK(p.split && !p.scope_conflict && p.snapshots.size() == 1);
    CHECK(p.snapshots[0].group == 0 && p.snapshots[0].use == 0 && !p.snapshots[0].volatile_);
    // a dynamic offset moves the uniform onto the written range: volatile
    BgUses g1;
    g1.add(buf(&buf_a, UseKind::StorageRead, 0, 16, 0));
    g1.add(buf(&buf_a, UseKind::StorageWrite, 256, 16));
    const std::vector<uint32_t> dyn{256};
    bound = {{0, &g1, &dyn}};
    p = plan_binds(bound, scope, false);
    CHECK(p.snapshots.size() == 1 && p.snapshots[0].volatile_);
    // ... unless the pipeline never writes through the writable binding: still a snapshot, not volatile
    g1.uses[1].never_written = true;
    p = plan_binds(bound, scope, false);
    CHECK(p.snapshots.size() == 1 && !p.snapshots[0].volatile_);
    g1.uses[1].never_written = false;
    // two writable uses (aliasing) and reads of another buffer are not usage conflicts
    BgUses g2;
    g2.add(buf(&buf_a, UseKind::StorageWrite, 0, 16));
    g2.add(buf(&buf_a, UseKind::StorageWrite, 0, 16));
    g2.add(buf(&buf_b, UseKind::Uniform, 0, 16));
    bound = {{0, &g2, nullptr}};
    CHECK(!plan_binds(bound, scope, true).split);
    // render: a write in one draw, a read of the same buffer in a later one -> scope split
    for (const ResUse& u : g2.uses) scope.add(u);
    BgUses g3;
    g3.add(buf(&buf_a, UseKind::Uniform, 512, 16));
    bound = {{1, &g3, nullptr}};
    p = plan_binds(bound, scope, true);
    CHECK(p.split && p.scope_conflict && p.snapshots.empty());
    CHECK(!plan_binds(bound, scope, false).split);  // compute: per-dispatch scopes
    BgUses g4;
    g4.add(buf(&buf_b, UseKind::Uniform, 0, 16));
    bound = {{1, &g4, nullptr}};
    CHECK(!plan_binds(bound, scope, true).split);
    // textures: sampled mips 0..3 against a storage write of mip 1 conflict; of mip 4 they do not
    BgUses t0;
    t0.add(img(&tex, UseKind::Sampled, 0, 4));
    t0.add(img(&tex, UseKind::StorageTexWrite, 1, 1));
    bound = {{0, &t0, nullptr}};
    UsageScope empty;
    p = plan_binds(bound, empty, false);
    CHECK(p.snapshots.size() == 1 && p.snapshots[0].use == 0 && p.snapshots[0].volatile_);
    BgUses t1;
    t1.add(img(&tex, UseKind::Sampled, 0, 4));
    t1.add(img(&tex, UseKind::StorageTexWrite, 4, 1));
    t1.add(img(&tex, UseKind::StorageTexWrite, 0, 4, 1, 1));  // another layer
    bound = {{0, &t1, nullptr}};
    CHECK(!plan_binds(bound, empty, false).split);
    // scope_violation: what is left after snapshots / copies
    bound = {{0, &g0, nullptr}};
    CHECK(scope_violation(bound) == ScopeViolation::RwBuffer);  // disjoint ranges still conflict (whole buffer)
    std::vector<std::vector<uint8_t>> repl{{1, 0}};             // the uniform bound from a snapshot
    CHECK(scope_violation(bound, &repl) == ScopeViolation::None);
    bound = {{0, &g2, nullptr}};
    CHECK(scope_violation(bound) == ScopeViolation::AliasBuffer);  // two writers of [0, 16)
    repl = {{0, 1, 0}};                                            // the second one on a scratch twin
    CHECK(scope_violation(bound, &repl) == ScopeViolation::None);
    BgUses g5;
    g5.add(buf(&buf_a, UseKind::StorageWrite, 0, 16));
    g5.add(buf(&buf_a, UseKind::StorageWrite, 16, 16));
    bound = {{0, &g5, nullptr}};
    CHECK(scope_violation(bound) == ScopeViolation::None);  // disjoint writers
    bound = {{0, &t0, nullptr}};
    CHECK(scope_violation(bound) == ScopeViolation::RwTexture);
    bound = {{0, &t1, nullptr}};
    CHECK(scope_violation(bound) == ScopeViolation::None);  // other mips / layers
    BgUses t2;
    t2.add(img(&tex, UseKind::StorageTexWrite, 0, 1));
    t2.add(img(&tex, UseKind::StorageTexWrite, 0, 1));
    bound = {{0, &t2, nullptr}};
    CHECK(scope_violation(bound) == ScopeViolation::AliasTexture);
    // reads only, across groups: none
    bound = {{0, &g3, nullptr}, {1, &g4, nullptr}};
    CHECK(scope_violation(bound) == ScopeViolation::None);
    // view ranges: WebGPU defaults
    ViewRange r = resolve_view_range(1, 5, 6, 4, 0, 0, 0, 0);  // cube of a 6-layer 2d texture
    CHECK(r.mip_count == 5 && r.layer_count == 6 && r.view_dim == 4);
    r = resolve_view_range(1, 5, 6, 0, 2, 0, 1, 0);  // undefined dim: 2d-array, layers 1..5
    CHECK(r.view_dim == 3 && r.mip_base == 2 && r.mip_count == 3 && r.layer_base == 1 && r.layer_count == 5);
    r = resolve_view_range(1, 1, 6, 2, 0, 0, 3, 0);  // a 2d view: one layer
    CHECK(r.layer_base == 3 && r.layer_count == 1);
    r = resolve_view_range(2, 3, 8, 0, 0, 0, 0, 0);  // 3d
    CHECK(r.view_dim == 6 && r.layer_count == 1 && r.mip_count == 3);
}

void test_binding_access_model() {
    using A = BindingAccess;
    // read_write bindings: written by assignment / compound / increment, by an atomic, through a pointer;
    // read-only when only loaded or measured; unreferenced when no function names them.
    const char* src = R"(
struct S { data: array<vec4<u32>, 4>, }
@group(0) @binding(0) var<uniform> cb0_: S;
@group(0) @binding(1) var<storage> sbuf0_: array<u32>;
@group(0) @binding(2) var<storage, read_write> w_assign: array<u32>;
@group(0) @binding(3) var<storage, read_write> r_only: array<u32>;
@binding(4) @group(0) var<storage, read_write> unused: array<u32>;
@group(1) @binding(0) var<storage, read_write> w_compound: array<u32>;
@group(1) @binding(1) var<storage, read_write> w_atomic: array<atomic<u32>>;
@group(1) @binding(2) var<storage, read_write> r_atomic_load: array<atomic<u32>>;
@group(1) @binding(3) var<storage, read_write> w_ptr: array<u32>;
@group(1) @binding(4) var<storage, read_write> w_inc: array<u32>;
@group(1) @binding(5) var<storage, read_write> w_member: S;
@group(1) @binding(6) var<storage, read_write> w_shift: array<u32>;
@group(1) @binding(7) var<storage, read_write> w_paren: array<u32>;
@group(2) @binding(0) var tex0_: texture_2d<f32>;
@group(2) @binding(1) var samp0_: sampler;
@group(2) @binding(2) var img: texture_storage_2d<rgba8unorm, write>;
fn helper(p: ptr<storage, array<u32>, read_write>) { (*p)[0] = 1u; }
@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    // unused[0] = 1u;  (a comment is not a reference)
    let n = arrayLength((&r_only));
    let a = r_only[min(id.x, n - 1u)] + sbuf0_[0] + cb0_.data[0].x;
    if (r_only[0] == 3u && r_only[1] <= 4u && r_only[2] != a) { w_assign[(id.x + 0u)] = a; }
    w_compound[id.x] += 2u;
    atomicAdd(&w_atomic[0], 1u);
    let l = atomicLoad((&r_atomic_load[id.x]));
    helper(&w_ptr);
    w_inc[0]++;
    w_member.data[1].y = l & r_only[3];
    w_shift[0] <<= 1u;
    (w_paren)[0] = 5u;
    textureStore(img, vec2<i32>(0, 0), textureSampleLevel(tex0_, samp0_, vec2<f32>(0.0), 0.0));
}
)";
    const ShaderBindingAccess acc = scan_binding_access(src);
    CHECK(acc.known);
    CHECK(acc.of(0, 0) == A::Read);
    CHECK(acc.of(0, 1) == A::Read);
    CHECK(acc.of(0, 2) == A::Written);
    CHECK(acc.of(0, 3) == A::Read);
    CHECK(acc.of(0, 4) == A::Unreferenced);
    CHECK(acc.of(0, 9) == A::Unreferenced);  // not declared at all
    CHECK(acc.of(1, 0) == A::Written);
    CHECK(acc.of(1, 1) == A::Written);
    CHECK(acc.of(1, 2) == A::Read);
    CHECK(acc.of(1, 3) == A::Written);
    CHECK(acc.of(1, 4) == A::Written);
    CHECK(acc.of(1, 5) == A::Written);
    CHECK(acc.of(1, 6) == A::Written);
    CHECK(acc.of(1, 7) == A::Written);
    CHECK(acc.of(2, 0) == A::Read);
    CHECK(acc.of(2, 1) == A::Read);
    CHECK(acc.of(2, 2) == A::Written);
    CHECK(acc.of(3, 0) == A::Unreferenced);
    // declared read_write but never written: r_only, unused, r_atomic_load
    CHECK(acc.writable_never_written() == 3);

    // Conservative: an @binding the scan cannot match, or no function bodies -> unknown (all written).
    const ShaderBindingAccess odd = scan_binding_access(
        "@group(0) @binding(0) @id(3) var<storage, read_write> b: array<u32>;\n"
        "@compute @workgroup_size(1) fn main() { }\n");
    CHECK(!odd.known && odd.of(0, 0) == A::Written && odd.of(0, 7) == A::Written);
    const ShaderBindingAccess nofn = scan_binding_access("@group(0) @binding(0) var<storage, read_write> b: array<u32>;\n");
    CHECK(!nofn.known);
    CHECK(scan_binding_access("").of(0, 0) == A::Written);
    // a local shadowing a binding's name: its stores count as stores to the binding
    const ShaderBindingAccess shadow = scan_binding_access(
        "@group(0) @binding(0) var<storage, read_write> b: array<u32>;\n"
        "@compute @workgroup_size(1) fn main() { var b: u32; b = 1u; }\n");
    CHECK(shadow.known && shadow.of(0, 0) == A::Written);
    // a member named like a binding is not a reference to it
    const ShaderBindingAccess member = scan_binding_access(
        "struct T { b: u32, }\n"
        "@group(0) @binding(0) var<storage, read_write> b: array<u32>;\n"
        "@group(0) @binding(1) var<uniform> t: T;\n"
        "@compute @workgroup_size(1) fn main() { let x = t.b; }\n");
    CHECK(member.known && member.of(0, 0) == A::Unreferenced && member.of(0, 1) == A::Read);
    // a binary & / && before the name is a read
    const ShaderBindingAccess band = scan_binding_access(
        "@group(0) @binding(0) var<storage, read_write> b: array<u32>;\n"
        "@group(0) @binding(1) var<storage, read_write> o: array<u32>;\n"
        "@compute @workgroup_size(1) fn main() { let x = 3u & b[0]; let y = (x == 1u) && (b[1] == 2u); o[0] = x; }\n");
    CHECK(band.known && band.of(0, 0) == A::Read && band.of(0, 1) == A::Written);

    // A pipeline: the strongest access over its stages; an unknown stage makes it unknown.
    const ShaderBindingAccess vs = scan_binding_access(
        "@group(0) @binding(0) var<storage, read_write> b: array<u32>;\n"
        "@group(0) @binding(1) var<uniform> u: vec4<f32>;\n"
        "@vertex fn vs_main() -> @builtin(position) vec4<f32> { return u + vec4<f32>(f32(b[0])); }\n");
    const ShaderBindingAccess fs = scan_binding_access(
        "@group(0) @binding(0) var<storage, read_write> b: array<u32>;\n"
        "@group(0) @binding(2) var<storage, read_write> c: array<u32>;\n"
        "@fragment fn fs_main() -> @location(0) vec4<f32> { b[1] = 2u; return vec4<f32>(0.0); }\n");
    const ShaderBindingAccess pl = merge_binding_access(vs, &fs);
    CHECK(pl.known && pl.of(0, 0) == A::Written && pl.of(0, 1) == A::Read && pl.of(0, 2) == A::Unreferenced);
    CHECK(merge_binding_access(vs, nullptr).of(0, 0) == A::Read);
    CHECK(!merge_binding_access(vs, &odd).known);
}

// memory_model.h: texture sizes, the bounded timeline, and the purge of destroyed handles from the caches.
struct FakeHandle {
    const void* p = nullptr;
    const void* Get() const { return p; }
    explicit operator bool() const { return p != nullptr; }
};
void test_memory_model() {
    // 256x256 rgba8, full mip chain: 4/3 of the top level (rounded down per mip, 1x1 at the end).
    CHECK(texture_storage_bytes(256, 256, 1, 1, 1, 32, false) == 262144);
    CHECK(texture_storage_bytes(256, 256, 1, 9, 1, 32, false) == 349524);
    CHECK(texture_storage_bytes(64, 64, 6, 1, 1, 32, false) == 6 * 64 * 64 * 4);  // array layers do not halve
    CHECK(texture_storage_bytes(64, 64, 8, 2, 1, 32, true) == 64 * 64 * 8 * 4 + 32 * 32 * 4 * 4);  // 3D depth halves
    CHECK(texture_storage_bytes(1920, 1080, 1, 1, 4, 32, false) == 1920ull * 1080 * 4 * 4);  // MSAA x4
    CHECK(texture_storage_bytes(4, 4, 1, 1, 1, 4, false) == 8);  // BC1: 8 bytes per 4x4 block
    CHECK(texture_storage_bytes(0, 0, 0, 0, 0, 32, false) == 0);

    MemoryTimeline tl(8, 1000);
    CHECK(tl.due(0));
    for (uint64_t t = 0; t < 8; ++t) {
        MemorySample s;
        s.t_ms = t * 1000;
        s.gpu_local_usage = t == 3 ? 900 : t;
        if (tl.due(s.t_ms)) tl.push(s);
    }
    // the 8th sample filled it: every second one kept, the interval doubled
    CHECK(tl.samples().size() == 4);
    CHECK(tl.interval_ms() == 2000);
    CHECK(tl.samples()[0].t_ms == 0 && tl.samples()[1].t_ms == 2000 && tl.samples()[3].t_ms == 6000);
    CHECK(tl.peak().t_ms == 3000);  // the peak survives the thinning
    CHECK(!tl.due(8000) && tl.due(9000));
    MemorySample lost;
    lost.tag = "device_lost";
    lost.t_ms = 8500;
    tl.push(lost);
    CHECK(tl.tagged().size() == 1 && tl.samples().size() == 4);  // tagged samples are kept apart

    int a = 0, b = 0, c = 0;
    RetiredSet<FakeHandle> rs;
    rs.add(FakeHandle{&a});
    rs.add(FakeHandle{&a});  // deduplicated
    rs.add(FakeHandle{});    // null ignored
    CHECK(rs.size() == 1 && rs.contains(&a) && !rs.contains(&b) && !rs.contains(nullptr));
    std::unordered_map<const void*, const void*> cache{{&a, &c}, {&b, &a}, {&c, &c}};  // view -> texture
    const size_t n = purge_cache(cache, [&](const auto& kv) { return rs.contains(kv.second) || rs.contains(kv.first); });
    CHECK(n == 2 && cache.size() == 1 && cache.count(&c) == 1);
    rs.clear();
    CHECK(rs.empty());
}

// --dump-targets-at: output size, half read-back, value stats, auto-scaled PNG pixels, names.
void test_target_dump_model() {
    DumpExtent d = target_dump_extent(1920, 1080, 640);
    CHECK(d.w == 640 && d.h == 360);
    d = target_dump_extent(100, 50, 640);
    CHECK(d.w == 100 && d.h == 50);
    d = target_dump_extent(4, 4096, 640);
    CHECK(d.w == 1 && d.h == 640);
    d = target_dump_extent(0, 0, 0);
    CHECK(d.w == 1 && d.h == 1);

    CHECK(dump_half_to_float(0x3C00) == 1.0f && dump_half_to_float(0xC000) == -2.0f && dump_half_to_float(0) == 0.0f);
    CHECK(std::isnan(dump_half_to_float(0x7E00)) && std::isinf(dump_half_to_float(0x7C00)));
    CHECK(std::fabs(dump_half_to_float(0x0001) - std::ldexp(1.0f, -24)) < 1e-12f);
    const uint16_t row[2][4] = {{0x3C00, 0x3800, 0, 0x3C00}, {0, 0, 0, 0}};  // (1, 0.5, 0, 1), (0, 0, 0, 0)
    std::vector<uint8_t> padded(256 + 16, 0);
    std::memcpy(padded.data(), row[0], 8);
    std::memcpy(padded.data() + 256, row[1], 8);
    std::vector<float> px = dump_rgba16f_to_float(padded.data(), 256, 1, 2);
    CHECK(px.size() == 8 && px[0] == 1.0f && px[1] == 0.5f && px[3] == 1.0f && px[4] == 0.0f);

    TargetDumpStats s = target_dump_stats(px.data(), 2);
    CHECK(!s.black() && !s.flat() && s.nonzero_rgb == 1 && s.max_rgb == 1.0f && s.mean[1] == 0.25);
    CHECK(target_dump_stats_text(s).find("ok nonzero 50.0%") == 0);
    const std::vector<float> zero(4 * 3, 0.0f);
    s = target_dump_stats(zero.data(), 3);
    CHECK(s.black() && s.flat() && target_dump_stats_text(s).find("BLACK") == 0);
    std::vector<uint8_t> rgb = target_dump_rgb8(zero.data(), 3, s.max_rgb);
    CHECK(rgb.size() == 9 && rgb[0] == 0 && rgb[8] == 0);
    // HDR / dim targets are scaled by their own max; NaN is counted, not averaged.
    const float hdr[8] = {8.0f, 2.0f, 0.0f, 1.0f, std::nanf(""), 0.0f, 0.0f, 1.0f};
    s = target_dump_stats(hdr, 2);
    CHECK(s.nan == 1 && s.max_rgb == 8.0f && s.mean[0] == 8.0);
    rgb = target_dump_rgb8(hdr, 2, s.max_rgb);
    CHECK(rgb[0] == 255 && rgb[1] == 64 && rgb[2] == 0 && rgb[3] == 0);
    const float dim[4] = {0.002f, 0.001f, 0.0f, 1.0f};
    s = target_dump_stats(dim, 1);
    rgb = target_dump_rgb8(dim, 1, s.max_rgb);
    CHECK(rgb[0] == 255 && rgb[1] >= 127 && rgb[1] <= 128);

    CHECK(target_dump_stem(7, 1234, "", 0) == "pass-007-tex-1234");
    CHECK(target_dump_stem(12, 5, "depth", 0) == "pass-012-tex-5-depth");
    CHECK(target_dump_stem(3, 9, "", 2) == "pass-003-tex-9-c2");
    CHECK(show_target_wgsl(false).find("u.mode.y == 1u") != std::string::npos);
}

int main() {
    test_memory_model();
    test_present_model();
    test_shader_payloads();
    test_pack();
    test_formats();
    test_write_texture_plan();
    test_clamps();
    test_walk();
    test_clear_color();
    test_begin_render_pass();
    test_write_buffer();
    test_split_frames();
    test_digest();
    test_readback_decode();
    test_readback_layout();
    test_readback_deliver();
    test_v1_vertex_layout();
    test_negotiated_limits();
    test_layout_remap();
    test_blit_model();
    test_show_target_model();
    test_target_dump_model();
    test_pipe_wait_budget();
    test_shader_workers();
    test_remap_bindings_match();
    test_shifted_pack_shader();
    test_wgsl_group_bindings();
    test_fit_missing_groups();
    test_fit_stage_limits();
    test_fit_failure_classes();
    test_fragment_outputs();
    test_error_scope_model();
    test_device_error_kinds();
    test_binding_fit();
    test_view_alias_model();
    test_clear_model();
    test_gpu_stall_watch();
    test_adapter_model();
    test_usage_scope_model();
    test_binding_access_model();
    if (g_failures) {
        std::fprintf(stderr, "%d failure(s)\n", g_failures);
        return 1;
    }
    std::printf("test_native_exec_model: ok\n");
    return 0;
}
