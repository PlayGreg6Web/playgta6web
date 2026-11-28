cmake_minimum_required(VERSION 3.24)
project(playgta6_runtime LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(playgta6_core STATIC
    os/os_shim.cpp
    os/vmem.cpp
    os/process_memory.cpp
    os/vmem_backing.cpp
    os/threading.cpp
    os/thread_name.cpp
    os/game_input.cpp
    os/httpfs_mailbox.cpp
    os/win32_kernel.cpp
    os/win32_sync.cpp
    os/win32_compat_file.cpp
    os/win32_compat_sys.cpp
    os/host_sleep.cpp
    os/win32_compat_text.cpp
    os/win32_compat_misc.cpp
    os/win32_rtl.cpp
    os/win32_rtl_text.cpp
    os/win32_rtl_file.cpp
    os/xgameruntime.cpp
    os/xasync_provider.cpp
    os/xbox_offline.cpp
    os/xbox_scarlett.cpp
    os/xbox_middleware.cpp
    os/win32_loader.cpp
    os/guest_heap.cpp
    os/ucrt_bridge.cpp
    os/crt_guest.cpp
    os/ucrt_ext.cpp
    os/crt_format.cpp
    os/crt_scan.cpp
    os/shim_exports.cpp
    os/image_loader.cpp
    os/winsock_ordinals.cpp
    os/host_marshallers.cpp
    gpu/cmd_encoder.cpp
    gpu/pso_cache.cpp
    gpu/descriptor_heap.cpp
    gpu/state_bridge.cpp
    gpu/d3d12_guest_objects.cpp
    gpu/d3d12_xs_exports.cpp
    gpu/d3d12_guest_resources.cpp
    gpu/flush_page_index.cpp
    gpu/d3d12_guest_command_list.cpp
    gpu/d3d12_xs_xbox.cpp
    gpu/xbox_embedded_native.cpp
    gpu/present_pacing.cpp
    gpu/texture_budget.cpp
    gpu/small_texture_pool.cpp
    gpu/d3d12_xs_dump.cpp
    gpu/shader_pack_index.cpp
    gpu/pipeline_prewarm.cpp
    gpu/pipeline_prewarm_host.cpp
    gpu/prewarm_pipeline_desc.cpp
    gpu/stage_reflection.cpp
    gpu/gpu_va_watch.cpp
    gpu/drop_table.cpp
    gpu/binding_fallbacks.cpp
    gpu/rt_arena.cpp
    gpu/rt_bvh.cpp
    gpu/rt_residency.cpp
    gpu/gpu_time.cpp
    gpu/d3d12_readback.cpp
    gpu/gfx10_descriptors.cpp
    gpu/gfx10_detile.cpp
    gpu/compute_texture_readback.cpp
    gpu/descriptor_words.cpp
    gpu/frame_capture.cpp
    os/xmem_windows.cpp
    os/guest_window.cpp
    guest/guest_runtime.cpp
    guest/guest_exceptions.cpp
    guest/com_bridge.cpp
    guest/guest_callbacks.cpp
    guest/export_thunks.cpp
    guest/guest_crt_exit.cpp
    guest/guest_rtl.cpp
    guest/cpu_helpers.cpp
    guest/guest_memory.cpp
    guest/memory_model.cpp
    guest/boot.cpp
    guest/export_counters.cpp
    guest/audio_probes.cpp
    guest/guest_watch.cpp
    guest/guest_probe.cpp
    guest/input_script.cpp
    guest/host_input.cpp
    guest/keymap.cpp
    guest/input_record.cpp
    wasm_boot.cpp
    os/xgameruntime_provider.cpp
    os/xgamesave_local.cpp
    os/xbox_scarlett_guest.cpp
    os/bcpack.cpp
    guest/program_loader.cpp
    guest/shard_dispatch.cpp
    os/xbox_scarlett_guest_input.cpp
    os/acp_hal_guest.cpp
    os/xma_decoder.cpp
    os/mmdevice_guest.cpp
    os/audio_downmix.cpp
    os/xapu_opus.cpp
)
target_include_directories(playgta6_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

option(PLAYGTA6_PERFORMANCE_BENCHMARKS "Build synthetic production-path performance benchmarks" OFF)
if(PLAYGTA6_PERFORMANCE_BENCHMARKS)
    add_executable(bench_fragmented_buffer_flush ../tests/bench_fragmented_buffer_flush.cpp)
    target_link_libraries(bench_fragmented_buffer_flush PRIVATE playgta6_core)
    add_executable(bench_buffer_shadows ../tests/bench_buffer_shadows.cpp)
    target_link_libraries(bench_buffer_shadows PRIVATE playgta6_core)
endif()

# Precompiled standard headers (MSVC only, standard library only: no project or Windows headers,
# so no TU's macro setup is affected). The ~70 core TUs spend most of their compile time parsing
# the same STL headers; a fresh build directory (every agent worktree, every harness build) pays it.
if(MSVC)
    target_precompile_headers(playgta6_core PRIVATE
        <algorithm> <array> <atomic> <chrono> <cstddef> <cstdint> <cstdio> <cstdlib> <cstring>
        <functional> <map> <memory> <mutex> <string> <string_view> <thread> <unordered_map>
        <utility> <vector>)
endif()

# XApu decode commands carry Opus packets (os/xapu_opus.h). libopus (BSD-3-Clause, Xiph.Org
# Foundation) is fetched at configure time from the release tarball, pinned by hash; with the
# option off the XApu model posts silence instead.
option(PLAYGTA6_OPUS "Decode XApu Opus packets with libopus (fetched at configure time)" ON)
if(PLAYGTA6_OPUS)
    include(FetchContent)
    set(OPUS_BUILD_SHARED_LIBRARY OFF CACHE BOOL "" FORCE)
    set(OPUS_BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(OPUS_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
    set(OPUS_INSTALL_PKG_CONFIG_MODULE OFF CACHE BOOL "" FORCE)
    set(OPUS_INSTALL_CMAKE_CONFIG_MODULE OFF CACHE BOOL "" FORCE)
    if(EMSCRIPTEN)
        set(OPUS_DISABLE_INTRINSICS ON CACHE BOOL "" FORCE)
        set(OPUS_STACK_PROTECTOR OFF CACHE BOOL "" FORCE)
        set(OPUS_FORTIFY_SOURCE OFF CACHE BOOL "" FORCE)
    endif()
    FetchContent_Declare(opus
        URL https://downloads.xiph.org/releases/opus/opus-1.5.2.tar.gz
            https://github.com/xiph/opus/releases/download/v1.5.2/opus-1.5.2.tar.gz
        URL_HASH SHA256=65c1d2f78b9f2fb20082c38cbe47c951ad5839345876e46941612ee87f9a7ce1
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(opus)
    if(EMSCRIPTEN)
        target_compile_options(opus PRIVATE -pthread -sMEMORY64=1)
    endif()
    target_link_libraries(playgta6_core PRIVATE opus)
    target_compile_definitions(playgta6_core PUBLIC PLAYGTA6_HAVE_OPUS=1)
endif()

# Build metadata for the browser build's boot report (`build` block, wasm_boot.cpp). Configure-time
# git state: the commit and whether the tree had uncommitted changes.
find_package(Git QUIET)
set(_pg6_commit "unknown")
set(_pg6_dirty 0)
if(GIT_FOUND)
    execute_process(COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" OUTPUT_VARIABLE _pg6_rev
        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET RESULT_VARIABLE _pg6_rev_rc)
    if(_pg6_rev_rc EQUAL 0 AND _pg6_rev)
        set(_pg6_commit "${_pg6_rev}")
        execute_process(COMMAND "${GIT_EXECUTABLE}" status --porcelain --untracked-files=no
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" OUTPUT_VARIABLE _pg6_status
            OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
        if(_pg6_status)
            set(_pg6_dirty 1)
        endif()
    endif()
endif()
set_source_files_properties(wasm_boot.cpp PROPERTIES COMPILE_DEFINITIONS
    "PG6_BUILD_COMMIT=\"${_pg6_commit}\";PG6_BUILD_DIRTY=${_pg6_dirty};PG6_BUILD_PROFILE=\"$<CONFIG>\"")

# Local-only wasm64 test build (tools/run_wasm_tests.py): keeps the CTest executables below enabled
# under Emscripten instead of building the browser `runtime` target. Off by default.
option(PLAYGTA6_WASM_TESTS "Build the CTest executables as wasm64 + pthreads Node tests (Emscripten only)" OFF)

if(EMSCRIPTEN AND NOT PLAYGTA6_WASM_TESTS)
    add_executable(runtime main.cpp)
    target_link_libraries(runtime PRIVATE playgta6_core)
    set_target_properties(runtime PROPERTIES
        OUTPUT_NAME "runtime"
        SUFFIX ".js"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../web/dist"
    )
    target_compile_options(runtime PRIVATE -pthread -sMEMORY64=1 -msimd128 -mbulk-memory)
    if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_link_options(runtime PRIVATE -sASSERTIONS=0 -sSAFE_HEAP=0)
    endif()
    target_link_options(runtime PRIVATE
        -pthread
        -sMEMORY64=1
        -sPTHREAD_POOL_SIZE=4
        -sMODULARIZE=1
        -sEXPORT_NAME=createRuntimeModule
        -sALLOW_MEMORY_GROWTH=1
        # Guest address space (os/vmem.h): linear memory [4 GiB, 16 GiB) is the identity-mapped
        # guest pool, grown on commit; sbrk is fenced below 4 GiB so malloc never enters it.
        -sMAXIMUM_MEMORY=16GB
        -Wl,--wrap=sbrk
        -sENVIRONMENT=worker
        -sERROR_ON_UNDEFINED_SYMBOLS=0
        -lexports.js
        # Pull the lifted-code helpers (guest/guest_runtime.cpp, EMSCRIPTEN_KEEPALIVE) out of the
        # static library even before recompiled code references them.
        -Wl,--undefined=__playgta6_indirect_call
        -Wl,--undefined=__playgta6_trap
        -Wl,--undefined=__playgta6_unimplemented
        -Wl,--undefined=__playgta6_divide_error
        -Wl,--undefined=__playgta6_cpuid
        -Wl,--undefined=__playgta6_rdtsc
        -Wl,--undefined=__playgta6_rdtscp
        -Wl,--undefined=__playgta6_xgetbv
    )
else()
    if(NOT EMSCRIPTEN)
        find_package(Threads REQUIRED)
        target_link_libraries(playgta6_core PUBLIC Threads::Threads)
    endif()

    enable_testing()
    add_executable(runtime_tests ../tests/test_runtime_abi.cpp)
    target_link_libraries(runtime_tests PRIVATE playgta6_core)
    add_test(NAME runtime_abi_tests COMMAND runtime_tests)

    add_executable(os_shim_tests ../tests/test_os_shims.cpp)
    target_link_libraries(os_shim_tests PRIVATE playgta6_core)
    add_test(NAME os_shim_tests COMMAND os_shim_tests)

    add_executable(test_httpfs_mailbox ../tests/test_httpfs_mailbox.cpp)
    target_link_libraries(test_httpfs_mailbox PRIVATE playgta6_core)
    add_test(NAME playgta6_httpfs_mailbox_tests COMMAND test_httpfs_mailbox)

    add_executable(test_xgameruntime ../tests/test_xgameruntime.cpp)
    target_link_libraries(test_xgameruntime PRIVATE playgta6_core)
    add_test(NAME playgta6_xgameruntime_tests COMMAND test_xgameruntime)

    add_executable(test_xbox_offline ../tests/test_xbox_offline.cpp)
    target_link_libraries(test_xbox_offline PRIVATE playgta6_core)
    add_test(NAME playgta6_xbox_offline_tests COMMAND test_xbox_offline)

    add_executable(test_win32_loader ../tests/test_win32_loader.cpp)
    target_link_libraries(test_win32_loader PRIVATE playgta6_core)
    add_test(NAME playgta6_win32_loader_tests COMMAND test_win32_loader)

    add_executable(test_shim_exports ../tests/test_shim_exports.cpp)
    target_link_libraries(test_shim_exports PRIVATE playgta6_core)
    add_test(NAME playgta6_shim_exports_tests COMMAND test_shim_exports)

    add_executable(test_shim_audit ../tests/test_shim_audit.cpp)
    target_link_libraries(test_shim_audit PRIVATE playgta6_core)
    add_test(NAME playgta6_shim_audit_tests COMMAND test_shim_audit)

    add_executable(test_crt_format ../tests/test_crt_format.cpp)
    target_link_libraries(test_crt_format PRIVATE playgta6_core)
    add_test(NAME playgta6_crt_format_tests COMMAND test_crt_format)

    add_executable(test_crt_scan ../tests/test_crt_scan.cpp)
    target_link_libraries(test_crt_scan PRIVATE playgta6_core)
    add_test(NAME playgta6_crt_scan_tests COMMAND test_crt_scan)

    add_executable(test_indirect_targets ../tests/test_indirect_targets.cpp)
    target_link_libraries(test_indirect_targets PRIVATE playgta6_core)
    add_test(NAME playgta6_indirect_targets_tests COMMAND test_indirect_targets)

    add_executable(test_xmem_hooks ../tests/test_xmem_hooks.cpp)
    target_link_libraries(test_xmem_hooks PRIVATE playgta6_core)
    add_test(NAME playgta6_xmem_hooks_tests COMMAND test_xmem_hooks)

    add_executable(test_image_loader ../tests/test_image_loader.cpp)
    target_link_libraries(test_image_loader PRIVATE playgta6_core)
    add_test(NAME playgta6_image_loader_tests COMMAND test_image_loader)

    add_executable(test_pso_cache ../tests/test_pso_cache.cpp)
    target_link_libraries(test_pso_cache PRIVATE playgta6_core)
    add_test(NAME playgta6_pso_cache_tests COMMAND test_pso_cache)

    add_executable(test_descriptor_heap ../tests/test_descriptor_heap.cpp)
    target_link_libraries(test_descriptor_heap PRIVATE playgta6_core)
    add_test(NAME descriptor_heap_tests COMMAND test_descriptor_heap)

    add_executable(test_gfx10_descriptors ../tests/test_gfx10_descriptors.cpp)
    target_link_libraries(test_gfx10_descriptors PRIVATE playgta6_core)
    add_test(NAME gfx10_descriptors_tests COMMAND test_gfx10_descriptors)

    add_executable(test_descriptor_words ../tests/test_descriptor_words.cpp)
    target_link_libraries(test_descriptor_words PRIVATE playgta6_core)
    add_test(NAME descriptor_words_tests COMMAND test_descriptor_words)

    add_executable(test_d3d12_guest_command_list_copy ../tests/test_d3d12_guest_command_list_copy.cpp)
    target_link_libraries(test_d3d12_guest_command_list_copy PRIVATE playgta6_core)
    add_test(NAME d3d12_guest_command_list_copy_tests COMMAND test_d3d12_guest_command_list_copy)

    add_executable(test_gfx10_detile ../tests/test_gfx10_detile.cpp)
    target_link_libraries(test_gfx10_detile PRIVATE playgta6_core)
    add_test(NAME test_gfx10_detile COMMAND test_gfx10_detile)

    add_executable(test_compute_texture_readback ../tests/test_compute_texture_readback.cpp)
    target_link_libraries(test_compute_texture_readback PRIVATE playgta6_core)
    add_test(NAME test_compute_texture_readback COMMAND test_compute_texture_readback)

    add_executable(test_console_layout_sizes ../tests/test_console_layout_sizes.cpp)
    target_link_libraries(test_console_layout_sizes PRIVATE playgta6_core)
    add_test(NAME test_console_layout_sizes COMMAND test_console_layout_sizes)

    add_executable(test_xbox_embedded_native ../tests/test_xbox_embedded_native.cpp)
    target_link_libraries(test_xbox_embedded_native PRIVATE playgta6_core)
    add_test(NAME test_xbox_embedded_native COMMAND test_xbox_embedded_native)

    add_executable(test_stage_reflection ../tests/test_stage_reflection.cpp)
    target_link_libraries(test_stage_reflection PRIVATE playgta6_core)
    add_test(NAME test_stage_reflection COMMAND test_stage_reflection)

    add_executable(test_drop_table ../tests/test_drop_table.cpp)
    target_link_libraries(test_drop_table PRIVATE playgta6_core)
    add_test(NAME test_drop_table COMMAND test_drop_table)
    add_executable(test_binding_fallbacks ../tests/test_binding_fallbacks.cpp)
    target_link_libraries(test_binding_fallbacks PRIVATE playgta6_core)
    add_test(NAME test_binding_fallbacks COMMAND test_binding_fallbacks)
    add_executable(test_rt_arena ../tests/test_rt_arena.cpp)
    target_link_libraries(test_rt_arena PRIVATE playgta6_core)
    add_test(NAME test_rt_arena COMMAND test_rt_arena)
    add_executable(test_rt_bvh ../tests/test_rt_bvh.cpp)
    target_link_libraries(test_rt_bvh PRIVATE playgta6_core)
    target_include_directories(test_rt_bvh PRIVATE ../tests)
    add_test(NAME test_rt_bvh COMMAND test_rt_bvh)
    add_executable(test_rt_residency ../tests/test_rt_residency.cpp)
    target_link_libraries(test_rt_residency PRIVATE playgta6_core)
    add_test(NAME test_rt_residency COMMAND test_rt_residency)
    add_executable(test_rt_vectors_json ../tests/test_rt_vectors_json.cpp)
    target_link_libraries(test_rt_vectors_json PRIVATE playgta6_core)
    target_include_directories(test_rt_vectors_json PRIVATE ../tests)
    add_test(NAME test_rt_vectors_json COMMAND test_rt_vectors_json ${CMAKE_CURRENT_SOURCE_DIR}/../tests/data/bvh_vectors.json)
    add_executable(test_gpu_time ../tests/test_gpu_time.cpp)
    target_link_libraries(test_gpu_time PRIVATE playgta6_core)
    add_test(NAME test_gpu_time COMMAND test_gpu_time)

    add_executable(test_gpu_va_watch ../tests/test_gpu_va_watch.cpp)
    target_link_libraries(test_gpu_va_watch PRIVATE playgta6_core)
    add_test(NAME test_gpu_va_watch COMMAND test_gpu_va_watch)

    add_executable(test_state_bridge ../tests/test_state_bridge.cpp)
    target_link_libraries(test_state_bridge PRIVATE playgta6_core)
    target_compile_definitions(test_state_bridge PRIVATE
        PG6_TEST_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../tests/fixtures")
    add_test(NAME state_bridge_tests COMMAND test_state_bridge)

    add_executable(test_win32_compat ../tests/test_win32_compat.cpp)
    target_link_libraries(test_win32_compat PRIVATE playgta6_core)
    add_test(NAME playgta6_win32_compat_tests COMMAND test_win32_compat)

    add_executable(test_win32_rtl ../tests/test_win32_rtl.cpp)
    target_link_libraries(test_win32_rtl PRIVATE playgta6_core)
    add_test(NAME playgta6_win32_rtl_tests COMMAND test_win32_rtl)

    add_executable(test_win32_sync ../tests/test_win32_sync.cpp)
    target_link_libraries(test_win32_sync PRIVATE playgta6_core)
    add_test(NAME playgta6_win32_sync_tests COMMAND test_win32_sync)

    add_executable(test_xbox_scarlett ../tests/test_xbox_scarlett.cpp)
    target_link_libraries(test_xbox_scarlett PRIVATE playgta6_core)
    add_test(NAME playgta6_xbox_scarlett_tests COMMAND test_xbox_scarlett)

    add_executable(test_xbox_middleware ../tests/test_xbox_middleware.cpp)
    target_link_libraries(test_xbox_middleware PRIVATE playgta6_core)
    add_test(NAME test_xbox_middleware COMMAND test_xbox_middleware)

    add_executable(test_lazy_host_access ../tests/test_lazy_host_access.cpp)
    target_link_libraries(test_lazy_host_access PRIVATE playgta6_core)
    add_test(NAME playgta6_lazy_host_access_tests COMMAND test_lazy_host_access)

    add_executable(test_ucrt_bridge ../tests/test_ucrt_bridge.cpp)
    target_link_libraries(test_ucrt_bridge PRIVATE playgta6_core)
    add_test(NAME playgta6_ucrt_bridge_tests COMMAND test_ucrt_bridge)

    add_executable(test_guest_heap ../tests/test_guest_heap.cpp)
    target_link_libraries(test_guest_heap PRIVATE playgta6_core)
    add_test(NAME playgta6_guest_heap_tests COMMAND test_guest_heap)

    add_executable(test_vmem_contiguous ../tests/test_vmem_contiguous.cpp)
    target_link_libraries(test_vmem_contiguous PRIVATE playgta6_core)
    add_test(NAME playgta6_vmem_contiguous_tests COMMAND test_vmem_contiguous)

    add_executable(test_guest_address_space ../tests/test_guest_address_space.cpp)
    target_link_libraries(test_guest_address_space PRIVATE playgta6_core)
    add_test(NAME playgta6_guest_address_space_tests COMMAND test_guest_address_space)

    add_executable(test_guest_runtime ../tests/test_guest_runtime.cpp)
    target_link_libraries(test_guest_runtime PRIVATE playgta6_core)
    add_test(NAME playgta6_guest_runtime_tests COMMAND test_guest_runtime)

    add_executable(test_guest_exceptions ../tests/test_guest_exceptions.cpp)
    target_link_libraries(test_guest_exceptions PRIVATE playgta6_core)
    add_test(NAME playgta6_guest_exceptions_tests COMMAND test_guest_exceptions)

    add_executable(test_com_bridge ../tests/test_com_bridge.cpp)
    target_link_libraries(test_com_bridge PRIVATE playgta6_core)
    add_test(NAME playgta6_com_bridge_tests COMMAND test_com_bridge)

    add_executable(test_cpu_helpers ../tests/test_cpu_helpers.cpp)
    target_link_libraries(test_cpu_helpers PRIVATE playgta6_core)
    add_test(NAME playgta6_cpu_helpers_tests COMMAND test_cpu_helpers)

    # Native first-boot harness without lifted code (tools/build_native_boot.py links the full one).
    add_executable(playgta6_boot guest/boot_main.cpp os/audio_out_wasapi.cpp)
    target_link_libraries(playgta6_boot PRIVATE playgta6_core)
    if(WIN32)
        target_link_libraries(playgta6_boot PRIVATE ole32)
    endif()

    add_executable(test_memory_budget ../tests/test_memory_budget.cpp)
    target_link_libraries(test_memory_budget PRIVATE playgta6_core)
    add_test(NAME playgta6_memory_budget_tests COMMAND test_memory_budget)
    add_executable(test_texture_budget ../tests/test_texture_budget.cpp)
    target_link_libraries(test_texture_budget PRIVATE playgta6_core)
    add_test(NAME playgta6_texture_budget_tests COMMAND test_texture_budget)
    add_executable(test_depth_twin ../tests/test_depth_twin.cpp)
    target_link_libraries(test_depth_twin PRIVATE playgta6_core)
    add_test(NAME playgta6_depth_twin_tests COMMAND test_depth_twin)

    add_executable(test_small_texture_pool ../tests/test_small_texture_pool.cpp)
    target_link_libraries(test_small_texture_pool PRIVATE playgta6_core)
    add_test(NAME playgta6_small_texture_pool_tests COMMAND test_small_texture_pool)
    add_executable(test_pipeline_prewarm ../tests/test_pipeline_prewarm.cpp)
    target_link_libraries(test_pipeline_prewarm PRIVATE playgta6_core)
    add_test(NAME playgta6_pipeline_prewarm_tests COMMAND test_pipeline_prewarm)
    add_executable(test_present_pacing ../tests/test_present_pacing.cpp)
    target_link_libraries(test_present_pacing PRIVATE playgta6_core)
    add_test(NAME playgta6_present_pacing_tests COMMAND test_present_pacing)
    add_executable(test_flush_page_index ../tests/test_flush_page_index.cpp)
    target_link_libraries(test_flush_page_index PRIVATE playgta6_core)
    add_test(NAME playgta6_flush_page_index_tests COMMAND test_flush_page_index)
    add_executable(test_frame_breakdown ../tests/test_frame_breakdown.cpp)
    target_link_libraries(test_frame_breakdown PRIVATE playgta6_core)
    add_test(NAME playgta6_frame_breakdown_tests COMMAND test_frame_breakdown)
    add_executable(test_window_layout ../tests/test_window_layout.cpp)
    target_include_directories(test_window_layout PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME playgta6_window_layout_tests COMMAND test_window_layout)
    add_executable(test_log_rate_limiter ../tests/test_log_rate_limiter.cpp)
    target_link_libraries(test_log_rate_limiter PRIVATE playgta6_core)
    add_test(NAME playgta6_log_rate_limiter_tests COMMAND test_log_rate_limiter)
    add_executable(test_input_script ../tests/test_input_script.cpp)
    target_link_libraries(test_input_script PRIVATE playgta6_core)
    add_test(NAME playgta6_input_script_tests COMMAND test_input_script)
    add_executable(test_host_input ../tests/test_host_input.cpp)
    target_link_libraries(test_host_input PRIVATE playgta6_core)
    add_test(NAME playgta6_host_input_tests COMMAND test_host_input)
    add_executable(test_keymap ../tests/test_keymap.cpp)
    target_link_libraries(test_keymap PRIVATE playgta6_core)
    add_test(NAME playgta6_keymap_tests COMMAND test_keymap)

    add_executable(test_guest_watch ../tests/test_guest_watch.cpp)
    target_link_libraries(test_guest_watch PRIVATE playgta6_core)
    add_test(NAME playgta6_guest_watch_tests COMMAND test_guest_watch)
    add_executable(test_guest_probe ../tests/test_guest_probe.cpp)
    target_link_libraries(test_guest_probe PRIVATE playgta6_core)
    add_test(NAME playgta6_guest_probe_tests COMMAND test_guest_probe)

    add_executable(test_audio_probes ../tests/test_audio_probes.cpp)
    target_link_libraries(test_audio_probes PRIVATE playgta6_core)
    add_test(NAME playgta6_audio_probes_tests COMMAND test_audio_probes)

    add_executable(test_export_counters ../tests/test_export_counters.cpp)
    target_link_libraries(test_export_counters PRIVATE playgta6_core)
    add_test(NAME playgta6_export_counters_tests COMMAND test_export_counters)

    add_executable(test_guest_boot ../tests/test_guest_boot.cpp)
    target_link_libraries(test_guest_boot PRIVATE playgta6_core)
    add_test(NAME playgta6_guest_boot_tests COMMAND test_guest_boot)

    if(WIN32)
        add_executable(test_minidump_fault ../tests/test_minidump_fault.cpp os/host_sleep.cpp)
        target_include_directories(test_minidump_fault PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        add_test(NAME playgta6_minidump_fault_tests COMMAND test_minidump_fault)
        set_tests_properties(playgta6_minidump_fault_tests PROPERTIES
            ENVIRONMENT "PG6_MINIDUMP_STACK_PY=${CMAKE_CURRENT_SOURCE_DIR}/../tools/minidump_stack.py;PG6_MINIDUMP_NO_CDB=1")
        add_test(NAME playgta6_minidump_guest_fault_tests COMMAND test_minidump_fault --guest)
        set_tests_properties(playgta6_minidump_guest_fault_tests PROPERTIES
            ENVIRONMENT "PG6_MINIDUMP_STACK_PY=${CMAKE_CURRENT_SOURCE_DIR}/../tools/minidump_stack.py;PG6_MINIDUMP_NO_CDB=1")
        # A thread holding the process heap lock must not freeze the dump (#663).
        add_test(NAME playgta6_minidump_locked_heap_tests COMMAND test_minidump_fault --locked)
        # Watchdog path: every thread's stack printed and an all-thread dump.
        add_test(NAME playgta6_minidump_hang_tests COMMAND test_minidump_fault --hang)
        set_tests_properties(playgta6_minidump_hang_tests PROPERTIES
            ENVIRONMENT "PG6_MINIDUMP_STACK_PY=${CMAKE_CURRENT_SOURCE_DIR}/../tools/minidump_stack.py;PG6_MINIDUMP_NO_CDB=1")

        # End to end: host fault, watchdog hang and stuck report path (backstop) each leave a
        # parseable non-empty minidump within a time bound.
        add_executable(test_minidump_e2e ../tests/test_minidump_e2e.cpp os/host_sleep.cpp)
        target_include_directories(test_minidump_e2e PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        add_test(NAME playgta6_minidump_e2e_tests COMMAND test_minidump_e2e)
        set_tests_properties(playgta6_minidump_e2e_tests PROPERTIES
            ENVIRONMENT "PG6_MINIDUMP_STACK_PY=${CMAKE_CURRENT_SOURCE_DIR}/../tools/minidump_stack.py;PG6_MINIDUMP_NO_CDB=1")

        # The real playgta6_boot watchdog path (stall seam --test-stall, synthetic image): stacks, dump, report.
        add_executable(test_boot_watchdog_e2e ../tests/test_boot_watchdog_e2e.cpp)
        target_include_directories(test_boot_watchdog_e2e PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        add_test(NAME playgta6_boot_watchdog_e2e_tests COMMAND test_boot_watchdog_e2e $<TARGET_FILE:playgta6_boot>
                 ${CMAKE_CURRENT_SOURCE_DIR}/../tools/minidump_stack.py)
        # The tests check minidump_stack.py's own output format; a cdb on the machine (CI runners have one)
        # would replace it with cdb's, so pin the built-in parser.
        set_tests_properties(playgta6_boot_watchdog_e2e_tests PROPERTIES ENVIRONMENT PG6_MINIDUMP_NO_CDB=1)
    endif()

    add_executable(test_host_marshallers ../tests/test_host_marshallers.cpp)
    target_link_libraries(test_host_marshallers PRIVATE playgta6_core)
    add_test(NAME playgta6_host_marshallers_tests COMMAND test_host_marshallers)

    add_executable(test_winsock_ordinals ../tests/test_winsock_ordinals.cpp)
    target_link_libraries(test_winsock_ordinals PRIVATE playgta6_core)
    add_test(NAME playgta6_winsock_ordinals_tests COMMAND test_winsock_ordinals)

    add_executable(test_xgameruntime_provider ../tests/test_xgameruntime_provider.cpp)
    target_link_libraries(test_xgameruntime_provider PRIVATE playgta6_core)
    add_test(NAME playgta6_xgameruntime_provider_tests COMMAND test_xgameruntime_provider)

    add_executable(test_xgamesave_local ../tests/test_xgamesave_local.cpp)
    target_link_libraries(test_xgamesave_local PRIVATE playgta6_core)
    add_test(NAME playgta6_xgamesave_local_tests COMMAND test_xgamesave_local)

    add_executable(test_guest_crt_exit ../tests/test_guest_crt_exit.cpp)
    target_link_libraries(test_guest_crt_exit PRIVATE playgta6_core)
    add_test(NAME playgta6_guest_crt_exit_tests COMMAND test_guest_crt_exit)

    add_executable(test_guest_callback_unwind ../tests/test_guest_callback_unwind.cpp)
    target_link_libraries(test_guest_callback_unwind PRIVATE playgta6_core)
    add_test(NAME playgta6_guest_callback_unwind_tests COMMAND test_guest_callback_unwind)

    add_executable(test_audio_downmix ../tests/test_audio_downmix.cpp)
    target_link_libraries(test_audio_downmix PRIVATE playgta6_core)
    add_test(NAME playgta6_audio_downmix_tests COMMAND test_audio_downmix)

    add_executable(test_xma_decoder ../tests/test_xma_decoder.cpp)
    target_link_libraries(test_xma_decoder PRIVATE playgta6_core)
    add_test(NAME playgta6_xma_decoder_tests COMMAND test_xma_decoder)

    add_executable(test_xma_bitreader ../tests/test_xma_bitreader.cpp os/xapu_opus.cpp)
    target_include_directories(test_xma_bitreader PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME playgta6_xma_bitreader_tests COMMAND test_xma_bitreader)

    add_executable(test_audio_wav_dump ../tests/test_audio_wav_dump.cpp)
    target_link_libraries(test_audio_wav_dump PRIVATE playgta6_core)
    add_test(NAME playgta6_audio_wav_dump_tests COMMAND test_audio_wav_dump)

    add_executable(test_audio_out ../tests/test_audio_out.cpp)
    target_link_libraries(test_audio_out PRIVATE playgta6_core)
    add_test(NAME playgta6_audio_out_tests COMMAND test_audio_out)

    add_executable(test_audio_report ../tests/test_audio_report.cpp)
    target_link_libraries(test_audio_report PRIVATE playgta6_core)
    add_test(NAME playgta6_audio_report_tests COMMAND test_audio_report)

    add_executable(test_bcpack ../tests/test_bcpack.cpp)
    target_link_libraries(test_bcpack PRIVATE playgta6_core)
    add_test(NAME playgta6_bcpack_tests COMMAND test_bcpack)

    add_executable(test_xbox_scarlett_guest ../tests/test_xbox_scarlett_guest.cpp)
    target_link_libraries(test_xbox_scarlett_guest PRIVATE playgta6_core)
    if(PLAYGTA6_OPUS)
        target_link_libraries(test_xbox_scarlett_guest PRIVATE opus)  # encodes the test packets
    endif()
    add_test(NAME playgta6_xbox_scarlett_guest_tests COMMAND test_xbox_scarlett_guest)

    add_executable(test_frame_capture ../tests/test_frame_capture.cpp)
    target_link_libraries(test_frame_capture PRIVATE playgta6_core)
    add_test(NAME playgta6_frame_capture_tests COMMAND test_frame_capture)

    add_executable(test_program_loader ../tests/test_program_loader.cpp)
    target_link_libraries(test_program_loader PRIVATE playgta6_core)
    add_test(NAME playgta6_program_loader_tests COMMAND test_program_loader)

    # R18: links the checked-in output of `playgta6-recomp export-thunks --only ...`
    # (tests/fixtures/export_thunks, kept in sync with the generator by a Rust test).
    add_executable(test_export_thunks
        ../tests/test_export_thunks.cpp
        ../tests/fixtures/export_thunks/pg6_export_thunks.cpp)
    target_include_directories(test_export_thunks PRIVATE ../tests/fixtures/export_thunks)
    target_link_libraries(test_export_thunks PRIVATE playgta6_core)
    add_test(NAME playgta6_guest_export_thunks_tests COMMAND test_export_thunks)

    add_executable(test_xasync_provider ../tests/test_xasync_provider.cpp)
    target_link_libraries(test_xasync_provider PRIVATE playgta6_core)
    add_test(NAME playgta6_xasync_provider_tests COMMAND test_xasync_provider)

    add_executable(test_xmem_windows ../tests/test_xmem_windows.cpp)
    target_link_libraries(test_xmem_windows PRIVATE playgta6_core)
    add_test(NAME playgta6_xmem_windows_tests COMMAND test_xmem_windows)

    add_executable(test_guest_window ../tests/test_guest_window.cpp)
    target_link_libraries(test_guest_window PRIVATE playgta6_core)
    add_test(NAME playgta6_guest_window_tests COMMAND test_guest_window)

    add_executable(test_vmem_window_stress ../tests/test_vmem_window_stress.cpp)
    target_link_libraries(test_vmem_window_stress PRIVATE playgta6_core)
    add_test(NAME playgta6_vmem_window_stress_tests COMMAND test_vmem_window_stress)

    # O29: the checked-in output of `playgta6-recomp export-thunks --memory-model window --only ...`
    # (tests/fixtures/export_thunks_window, kept in sync with the generator by a Rust test).
    add_executable(test_window_marshal
        ../tests/test_window_marshal.cpp
        ../tests/fixtures/export_thunks_window/pg6_export_thunks.cpp)
    target_include_directories(test_window_marshal PRIVATE ../tests/fixtures/export_thunks_window)
    target_link_libraries(test_window_marshal PRIVATE playgta6_core)
    add_test(NAME playgta6_window_marshal_tests COMMAND test_window_marshal)

    add_executable(test_guest_window_e2e ../tests/test_guest_window_e2e.cpp)
    target_link_libraries(test_guest_window_e2e PRIVATE playgta6_core)
    add_test(NAME playgta6_guest_window_e2e_tests COMMAND test_guest_window_e2e)

    add_executable(test_fragmented_buffer_flush ../tests/bench_fragmented_buffer_flush.cpp)
    target_link_libraries(test_fragmented_buffer_flush PRIVATE playgta6_core)
    add_test(NAME fragmented_buffer_sparse COMMAND test_fragmented_buffer_flush 2 2 fragmented)
    add_test(NAME fragmented_buffer_unaligned_tail COMMAND test_fragmented_buffer_flush 2 2 fragmented-unaligned)
    add_test(NAME fragmented_buffer_dense COMMAND test_fragmented_buffer_flush 2 1 fragmented-dense)

    add_executable(test_d3d12_xs_exports ../tests/test_d3d12_xs_exports.cpp)
    target_link_libraries(test_d3d12_xs_exports PRIVATE playgta6_core)
    add_test(NAME playgta6_d3d12_xs_exports_tests COMMAND test_d3d12_xs_exports)

    add_executable(test_wasm_boot ../tests/test_wasm_boot.cpp)
    target_link_libraries(test_wasm_boot PRIVATE playgta6_core)
    add_test(NAME test_wasm_boot COMMAND test_wasm_boot)

    add_executable(test_shard_dispatch ../tests/test_shard_dispatch.cpp)
    target_link_libraries(test_shard_dispatch PRIVATE playgta6_core)
    add_test(NAME test_shard_dispatch COMMAND test_shard_dispatch)

    # Indirect-call dispatch micro-benchmark (O33); CTest runs the smoke variant
    # (no arguments; `bench_indirect_call --full` for numbers).
    add_executable(bench_indirect_call ../tests/bench_indirect_call.cpp)
    target_link_libraries(bench_indirect_call PRIVATE playgta6_core)
    add_test(NAME bench_indirect_call_smoke COMMAND bench_indirect_call)

    # XMem physical-page commit path micro-benchmark (R40); CTest runs the smoke variant
    # (`bench_xmem_map --full [--threads N]` for numbers).
    add_executable(bench_xmem_map ../tests/bench_xmem_map.cpp)
    target_link_libraries(bench_xmem_map PRIVATE playgta6_core)
    add_test(NAME bench_xmem_map_smoke COMMAND bench_xmem_map)

    # XMem write-tracking contention stress test (O33); CTest runs the smoke variant
    # (`bench_xmem_contention --full [--threads N]` for numbers).
    add_executable(bench_xmem_contention ../tests/bench_xmem_contention.cpp)
    target_link_libraries(bench_xmem_contention PRIVATE playgta6_core)
    add_test(NAME bench_xmem_contention_smoke COMMAND bench_xmem_contention)

    # Native executor stream model (gpu/native_exec/stream_model.h, replay_frames.h): no WebGPU,
    # so it runs on every host build.
    add_executable(test_live_pump ../tests/test_live_pump.cpp)
    target_link_libraries(test_live_pump PRIVATE playgta6_core)
    add_test(NAME test_live_pump COMMAND test_live_pump)
    add_executable(test_native_exec_model ../tests/test_native_exec_model.cpp)
    target_link_libraries(test_native_exec_model PRIVATE playgta6_core)
    add_test(NAME test_native_exec_model COMMAND test_native_exec_model)
    add_executable(test_native_exec_quarantine ../tests/test_native_exec_quarantine.cpp)
    target_link_libraries(test_native_exec_quarantine PRIVATE playgta6_core)
    add_test(NAME test_native_exec_quarantine COMMAND test_native_exec_quarantine)
    add_executable(test_overlay_model ../tests/test_overlay_model.cpp)
    target_link_libraries(test_overlay_model PRIVATE playgta6_core)
    add_test(NAME test_overlay_model COMMAND test_overlay_model)
    add_executable(test_screenshot_model ../tests/test_screenshot_model.cpp)
    target_link_libraries(test_screenshot_model PRIVATE playgta6_core)
    target_include_directories(test_screenshot_model PRIVATE ../tests)
    add_test(NAME test_screenshot_model COMMAND test_screenshot_model)
    add_executable(test_native_exec_reflection ../tests/test_native_exec_reflection.cpp)
    target_link_libraries(test_native_exec_reflection PRIVATE playgta6_core)
    add_test(NAME test_native_exec_reflection COMMAND test_native_exec_reflection)
    add_executable(test_blob_cache ../tests/test_blob_cache.cpp)
    target_link_libraries(test_blob_cache PRIVATE playgta6_core)
    add_test(NAME test_blob_cache COMMAND test_blob_cache)

    # Native WebGPU executor (docs/NATIVE_EXECUTOR_DESIGN.md): consumes the packet stream with Dawn,
    # the WebGPU implementation inside Chrome (BSD-3-Clause). Off by default; when on, the official
    # prebuilt Dawn release for the host is fetched at configure time (pinned by SHA-256, nothing
    # is vendored). The prebuilt libraries are Release builds: use a Release or RelWithDebInfo
    # configuration.
    option(PLAYGTA6_NATIVE_EXECUTOR "Build the Dawn-backed native GPU executor (prebuilt Dawn fetched at configure time)" OFF)
    if(PLAYGTA6_NATIVE_EXECUTOR AND NOT EMSCRIPTEN)
        set(_pg6_dawn_release "https://github.com/google/dawn/releases/download/v20261109.214834")
        set(_pg6_dawn_commit "7435fd87c506828e9b119c0052bfa19a8ba589c5")
        if(WIN32 AND CMAKE_SIZEOF_VOID_P EQUAL 8)
            set(_pg6_dawn_url "${_pg6_dawn_release}/Dawn-${_pg6_dawn_commit}-windows-latest-Release.tar.gz")
            set(_pg6_dawn_sha256 "7570acce74fef379c477a634e8ff86f882ec9bb8f51218a9a1e0930393cb6196")
        elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
            set(_pg6_dawn_url "${_pg6_dawn_release}/Dawn-${_pg6_dawn_commit}-ubuntu-latest-Release.tar.gz")
            set(_pg6_dawn_sha256 "2125482707ba81132ecd356b78418f72d02b55007abc51d4335a91b8b4cae266")
        else()
            message(FATAL_ERROR "PLAYGTA6_NATIVE_EXECUTOR: no pinned prebuilt Dawn for this host")
        endif()
        include(FetchContent)
        FetchContent_Declare(dawn_prebuilt
            URL "${_pg6_dawn_url}"
            URL_HASH SHA256=${_pg6_dawn_sha256}
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
        FetchContent_MakeAvailable(dawn_prebuilt)
        find_package(Threads REQUIRED)
        find_package(Dawn CONFIG REQUIRED PATHS "${dawn_prebuilt_SOURCE_DIR}" NO_DEFAULT_PATH)

        add_library(playgta6_native_exec STATIC
            gpu/native_exec/native_executor.cpp
            gpu/native_exec/native_window.cpp
        )
        target_include_directories(playgta6_native_exec PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
        target_compile_definitions(playgta6_native_exec PRIVATE PG6_DAWN_COMMIT="${_pg6_dawn_commit}")
        if(MSVC)
            target_compile_options(playgta6_native_exec PRIVATE /bigobj)  # native_executor.cpp exceeds the default section limit (C1128)
        endif()
        target_link_libraries(playgta6_native_exec PUBLIC dawn::webgpu_dawn PRIVATE playgta6_core)  # PRIVATE: gpu/gpu_time.cpp (the GPU-time table the boot report prints)

        # `playgta6_boot --window`: the live packet stream goes to the executor (gpu/native_exec/live_pump.h).
        if(TARGET playgta6_boot)
            target_link_libraries(playgta6_boot PRIVATE playgta6_native_exec)
            target_compile_definitions(playgta6_boot PRIVATE PLAYGTA6_NATIVE_EXECUTOR=1)
        endif()

        add_executable(pg6_native_exec gpu/native_exec/pg6_native_exec_main.cpp)
        target_link_libraries(pg6_native_exec PRIVATE playgta6_native_exec)

        # Writes a tiny synthetic .pg6trace (swapchain clears only, no game data) for tests/native_exec_warp_smoke.py.
        add_executable(gen_native_exec_trace ../tests/gen_native_exec_trace.cpp)
        target_link_libraries(gen_native_exec_trace PRIVATE playgta6_core)

        # Runs on the host's GPU; exits 77 (skipped) when there is no WebGPU adapter.
        add_executable(test_native_executor ../tests/test_native_executor.cpp)
        target_link_libraries(test_native_executor PRIVATE playgta6_native_exec playgta6_core)
        target_include_directories(test_native_executor PRIVATE ../tests)
        add_test(NAME test_native_executor COMMAND test_native_executor)
        set_tests_properties(test_native_executor PROPERTIES SKIP_RETURN_CODE 77)

        # The WGSL prelude (gpu/rt_bvh.h) compiled by Dawn/Tint and run on the GPU against the CPU twin.
        add_executable(test_rt_bvh_gpu ../tests/test_rt_bvh_gpu.cpp)
        target_link_libraries(test_rt_bvh_gpu PRIVATE playgta6_native_exec playgta6_core)
        target_include_directories(test_rt_bvh_gpu PRIVATE ../tests)
        add_test(NAME test_rt_bvh_gpu COMMAND test_rt_bvh_gpu)
        set_tests_properties(test_rt_bvh_gpu PROPERTIES SKIP_RETURN_CODE 77)

        # Dawn's D3D12 backend compiles shaders with DXC (as Chrome does) and loads dxcompiler.dll /
        # dxil.dll from the executable's directory: fetch the official DXC release (NCSA/MIT, dxil.dll
        # under Microsoft's redistributable licence) and copy both next to every executable.
        if(WIN32)
            FetchContent_Declare(dxc_prebuilt
                URL "https://github.com/microsoft/DirectXShaderCompiler/releases/download/v1.9.2609/dxc_2026_09_29.zip"
                URL_HASH SHA256=ad31b1fc8443175d204f77a611fdb3ef2ec42759bdc2f1167368de24a4a7e7f1
                DOWNLOAD_EXTRACT_TIMESTAMP TRUE
            )
            FetchContent_MakeAvailable(dxc_prebuilt)
            foreach(_t pg6_native_exec test_native_executor test_rt_bvh_gpu playgta6_boot)
                add_custom_command(TARGET ${_t} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${dxc_prebuilt_SOURCE_DIR}/bin/x64/dxcompiler.dll"
                        "${dxc_prebuilt_SOURCE_DIR}/bin/x64/dxil.dll"
                        "$<TARGET_FILE_DIR:${_t}>")
            endforeach()
        endif()
    endif()
endif()

if(EMSCRIPTEN AND PLAYGTA6_WASM_TESTS)
    # Same memory model as the browser runtime (MEMORY64, pthreads, 16 GiB max, sbrk fenced below
    # the guest pool); -fexceptions because test fault handlers throw. The host_* imports that
    # os_shim.cpp declares are resolved by a Node-side JS library instead of the page's glue.
    set(PLAYGTA6_WASM_TEST_JS_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/../tools/wasm_test_support/host_imports.js"
        CACHE FILEPATH "JS library providing the env.host_* imports for Node-run wasm64 tests")
    target_compile_options(playgta6_core PUBLIC -pthread -sMEMORY64=1 -fexceptions)
    # The browser's process arena is the whole 12 GiB guest pool, which leaves no room for the extra
    # arenas several tests create next to it (offset-arena, rebase and multi-arena checks): tests
    # get a 4 GiB process arena instead (vmem.h: sizes are build parameters), 8 GiB stay free.
    target_compile_definitions(playgta6_core PUBLIC PLAYGTA6_PROCESS_GUEST_SPACE_SIZE=4294967296ULL)
    get_directory_property(_pg6_wasm_targets BUILDSYSTEM_TARGETS)
    foreach(_t IN LISTS _pg6_wasm_targets)
        get_target_property(_type ${_t} TYPE)
        if(_type STREQUAL "EXECUTABLE")
            set_target_properties(${_t} PROPERTIES SUFFIX ".js")
            target_link_options(${_t} PRIVATE
                -pthread -sMEMORY64=1 -fexceptions
                -sPTHREAD_POOL_SIZE=24
                -sPROXY_TO_PTHREAD=1
                -sEXIT_RUNTIME=1
                -sSTACK_SIZE=1MB
                -sALLOW_MEMORY_GROWTH=1
                -sMAXIMUM_MEMORY=16GB
                -Wl,--wrap=sbrk
                -sENVIRONMENT=node
                -sERROR_ON_UNDEFINED_SYMBOLS=1
                -sNODERAWFS=1  # tests read tests/fixtures from the host, like the native build
                --js-library=${PLAYGTA6_WASM_TEST_JS_LIBRARY}
            )
            set_property(TARGET ${_t} APPEND PROPERTY LINK_DEPENDS "${PLAYGTA6_WASM_TEST_JS_LIBRARY}")
        endif()
    endforeach()
endif()
