cmake_minimum_required(VERSION 3.24)

project(dsmvc VERSION 0.1.0 LANGUAGES CXX)

if(MSVC)
    # The plugin boundary is a C ABI. Static CRT linkage prevents the host's
    # bundled, potentially older MSVC runtime from changing STL synchronization ABI.
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

include(CTest)

set(DSMVC_VAPOURSYNTH_SDK "" CACHE PATH "Path to the VapourSynth SDK directory")
set(DSMVC_VS_PYTHON "" CACHE FILEPATH "Python executable from the target VapourSynth environment")
set(DSMVC_BASELINE_PLUGIN "" CACHE FILEPATH "Baseline descale plugin used by integration tests")
set(DSMVC_BASELINE_WRAPPER "" CACHE FILEPATH "Baseline descale.py used by integration tests")
option(DSMVC_ENABLE_CUDA
       "Build the optional CUDA backend on Windows or Linux" OFF)
option(DSMVC_ENABLE_VULKAN
       "Build the optional Vulkan 1.2 backend on Windows or Linux" OFF)
set(DSMVC_CUDA_ARCHITECTURES
    "75;86;89;120" CACHE STRING
    "Native SM targets embedded in the CUDA module")
set(DSMVC_CUDA_PTX_ARCHITECTURES "75;120" CACHE STRING
    "Virtual compute targets embedded as CUDA PTX fallbacks")
option(DSMVC_BUILD_BENCHMARKS "Build CPU and numerical contract benchmarks" OFF)
option(DSMVC_ENABLE_NATIVE_CPU_SIMD
       "Build the native AVX2 or NEON CPU implementation" ON)

set(DSMVC_TARGET_PROCESSOR "${CMAKE_SYSTEM_PROCESSOR}")
if(APPLE AND CMAKE_OSX_ARCHITECTURES)
    list(LENGTH CMAKE_OSX_ARCHITECTURES _dsmvc_osx_arch_count)
    if(NOT _dsmvc_osx_arch_count EQUAL 1)
        message(FATAL_ERROR
            "Universal macOS builds require separate arm64 and x86_64 builds")
    endif()
    list(GET CMAKE_OSX_ARCHITECTURES 0 DSMVC_TARGET_PROCESSOR)
endif()
string(TOLOWER "${DSMVC_TARGET_PROCESSOR}" DSMVC_TARGET_PROCESSOR)

set(_dsmvc_metal_default OFF)
if(APPLE AND DSMVC_TARGET_PROCESSOR MATCHES "^(arm64|aarch64)$")
    set(_dsmvc_metal_default ON)
endif()
if(DEFINED CACHE{DSMVC_BUILD_METAL_EXPERIMENTS})
    set(_dsmvc_legacy_metal_option_defined ON)
else()
    set(_dsmvc_legacy_metal_option_defined OFF)
endif()
option(DSMVC_ENABLE_METAL
    "Build the Apple Silicon UMA Metal backend" ${_dsmvc_metal_default})
option(DSMVC_BUILD_METAL_EXPERIMENTS
    "Deprecated alias for DSMVC_ENABLE_METAL" ${DSMVC_ENABLE_METAL})
if(_dsmvc_legacy_metal_option_defined)
    set(DSMVC_ENABLE_METAL ${DSMVC_BUILD_METAL_EXPERIMENTS})
else()
    set(DSMVC_BUILD_METAL_EXPERIMENTS ${DSMVC_ENABLE_METAL})
endif()

if(APPLE AND DSMVC_TARGET_PROCESSOR MATCHES "^(arm64|aarch64)$")
    add_compile_options(
        "$<$<CONFIG:Release>:-O3;-flto=full>"
        "$<$<CONFIG:RelWithDebInfo>:-O3;-flto=full>")
    add_link_options(
        "$<$<CONFIG:Release>:-flto=full>"
        "$<$<CONFIG:RelWithDebInfo>:-flto=full>")
endif()

if(DSMVC_VAPOURSYNTH_SDK)
    set(_dsmvc_vs_hints "${DSMVC_VAPOURSYNTH_SDK}/include")
endif()
find_path(DSMVC_VAPOURSYNTH_INCLUDE_DIR VapourSynth4.h
    HINTS
        ${_dsmvc_vs_hints}
        "$ENV{VAPOURSYNTH_SDK}/include"
        "D:/okegui/OKEGui/tools/vapoursynth/sdk/include"
)
if(NOT DSMVC_VAPOURSYNTH_INCLUDE_DIR)
    message(FATAL_ERROR "VapourSynth API4 headers were not found; set DSMVC_VAPOURSYNTH_SDK")
endif()

set(DSMVC_ENGINE_SOURCES
    src/axis_plan.cpp
    src/backend.cpp
    src/cpu_executor.cpp
    src/executor.cpp
)

set(DSMVC_CAN_BUILD_AVX2 OFF)
set(DSMVC_CAN_BUILD_NEON OFF)
if(DSMVC_ENABLE_NATIVE_CPU_SIMD
   AND CMAKE_SIZEOF_VOID_P EQUAL 8
   AND DSMVC_TARGET_PROCESSOR MATCHES "^(amd64|x86_64)$")
    set(DSMVC_CAN_BUILD_AVX2 ON)
    list(APPEND DSMVC_ENGINE_SOURCES src/cpu_executor_avx2.cpp)
elseif(DSMVC_ENABLE_NATIVE_CPU_SIMD
       AND CMAKE_SIZEOF_VOID_P EQUAL 8
       AND DSMVC_TARGET_PROCESSOR MATCHES "^(arm64|aarch64)$")
    set(DSMVC_CAN_BUILD_NEON ON)
    list(APPEND DSMVC_ENGINE_SOURCES src/cpu_executor_neon.cpp)
endif()

if(DSMVC_ENABLE_CUDA)
    if(NOT WIN32 AND NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        message(FATAL_ERROR "DSMVC_ENABLE_CUDA currently requires Windows or Linux")
    endif()
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    if(TARGET CUDA::nvtx3)
        set(_dsmvc_nvtx_target CUDA::nvtx3)
    else()
        add_library(dsmvc_nvtx3 INTERFACE)
        target_include_directories(dsmvc_nvtx3 SYSTEM INTERFACE
            "${CUDAToolkit_INCLUDE_DIRS}")
        target_link_libraries(dsmvc_nvtx3 INTERFACE ${CMAKE_DL_LIBS})
        set(_dsmvc_nvtx_target dsmvc_nvtx3)
    endif()
    find_program(DSMVC_CUOBJDUMP_EXECUTABLE
        NAMES cuobjdump.exe cuobjdump
        HINTS "${CUDAToolkit_BIN_DIR}")
    if(BUILD_TESTING AND NOT DSMVC_CUOBJDUMP_EXECUTABLE)
        message(FATAL_ERROR "CUDA artifact tests require cuobjdump")
    endif()
    if(DSMVC_CUDA_ARCHITECTURES STREQUAL ""
       OR DSMVC_CUDA_PTX_ARCHITECTURES STREQUAL "")
        message(FATAL_ERROR "CUDA native and PTX architecture lists must not be empty")
    endif()
    list(REMOVE_DUPLICATES DSMVC_CUDA_ARCHITECTURES)
    list(REMOVE_DUPLICATES DSMVC_CUDA_PTX_ARCHITECTURES)
    foreach(_architecture IN LISTS
            DSMVC_CUDA_ARCHITECTURES DSMVC_CUDA_PTX_ARCHITECTURES)
        if(NOT _architecture MATCHES "^[0-9]+$")
            message(FATAL_ERROR "CUDA architecture lists must contain numeric SM values")
        endif()
    endforeach()

    set(_dsmvc_cuda_architectures)
    foreach(_architecture IN LISTS DSMVC_CUDA_ARCHITECTURES)
        list(APPEND _dsmvc_cuda_architectures "${_architecture}-real")
    endforeach()
    foreach(_architecture IN LISTS DSMVC_CUDA_PTX_ARCHITECTURES)
        list(APPEND _dsmvc_cuda_architectures "${_architecture}-virtual")
    endforeach()

    list(APPEND DSMVC_ENGINE_SOURCES
        src/cuda/cuda_executor.cpp
        src/cuda/dsmvc_cuda.cu)
endif()

if(DSMVC_ENABLE_VULKAN)
    if(NOT WIN32 AND NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        message(FATAL_ERROR "DSMVC_ENABLE_VULKAN requires Windows or Linux")
    endif()
    find_package(Vulkan 1.2 REQUIRED)
    if(NOT DSMVC_GLSLC_EXECUTABLE AND Vulkan_GLSLC_EXECUTABLE)
        set(DSMVC_GLSLC_EXECUTABLE "${Vulkan_GLSLC_EXECUTABLE}"
            CACHE FILEPATH "Path to the glslc shader compiler")
    endif()
    if(NOT DSMVC_GLSLC_EXECUTABLE)
        find_program(DSMVC_GLSLC_EXECUTABLE
            NAMES glslc.exe glslc
            HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")
    endif()
    if(NOT DSMVC_GLSLC_EXECUTABLE)
        message(FATAL_ERROR "Vulkan builds require glslc")
    endif()
    if(BUILD_TESTING)
        find_program(DSMVC_SPIRV_VAL_EXECUTABLE
            NAMES spirv-val.exe spirv-val
            HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")
        if(NOT DSMVC_SPIRV_VAL_EXECUTABLE)
            message(FATAL_ERROR "Vulkan test builds require spirv-val")
        endif()
    endif()

    set(_dsmvc_vulkan_generated_dir
        "${CMAKE_CURRENT_BINARY_DIR}/generated/vulkan")
    file(MAKE_DIRECTORY "${_dsmvc_vulkan_generated_dir}")
    function(dsmvc_compile_vulkan_shader name source)
        set(_definitions ${ARGN})
        set(_spv "${_dsmvc_vulkan_generated_dir}/${name}.spv")
        set(_header
            "${_dsmvc_vulkan_generated_dir}/vulkan_${name}_spv.hpp")
        set(_symbol "vulkan_${name}_spv")
        add_custom_command(
            OUTPUT "${_spv}"
            COMMAND "${DSMVC_GLSLC_EXECUTABLE}"
                --target-env=vulkan1.2 -O ${_definitions}
                -o "${_spv}" "${CMAKE_CURRENT_SOURCE_DIR}/${source}"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${source}"
            VERBATIM)
        add_custom_command(
            OUTPUT "${_header}"
            COMMAND "${CMAKE_COMMAND}"
                "-DINPUT=${_spv}"
                "-DOUTPUT=${_header}"
                "-DSYMBOL=${_symbol}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spirv.cmake"
            DEPENDS "${_spv}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spirv.cmake"
            VERBATIM)
        set(DSMVC_VULKAN_SPIRV
            ${DSMVC_VULKAN_SPIRV} "${_spv}" PARENT_SCOPE)
        set(DSMVC_VULKAN_GENERATED_HEADERS
            ${DSMVC_VULKAN_GENERATED_HEADERS} "${_header}" PARENT_SCOPE)
    endfunction()

    dsmvc_compile_vulkan_shader(
        transpose_256 src/vulkan/shaders/transpose.comp
        -DDSMVC_LOCAL_X=32 -DDSMVC_LOCAL_Y=8)
    dsmvc_compile_vulkan_shader(
        transpose_128 src/vulkan/shaders/transpose.comp
        -DDSMVC_LOCAL_X=16 -DDSMVC_LOCAL_Y=8)
    dsmvc_compile_vulkan_shader(
        rhs_256 src/vulkan/shaders/rhs.comp
        -DDSMVC_LOCAL_X=32 -DDSMVC_LOCAL_Y=8)
    dsmvc_compile_vulkan_shader(
        rhs_128 src/vulkan/shaders/rhs.comp
        -DDSMVC_LOCAL_X=16 -DDSMVC_LOCAL_Y=8)
    dsmvc_compile_vulkan_shader(
        inverse_32 src/vulkan/shaders/inverse.comp
        -DDSMVC_LOCAL_SIZE=32 -DDSMVC_SOLVE_ONLY=0)
    dsmvc_compile_vulkan_shader(
        inverse_64 src/vulkan/shaders/inverse.comp
        -DDSMVC_LOCAL_SIZE=64 -DDSMVC_SOLVE_ONLY=0)
    dsmvc_compile_vulkan_shader(
        solve_32 src/vulkan/shaders/inverse.comp
        -DDSMVC_LOCAL_SIZE=32 -DDSMVC_SOLVE_ONLY=1)
    dsmvc_compile_vulkan_shader(
        solve_64 src/vulkan/shaders/inverse.comp
        -DDSMVC_LOCAL_SIZE=64 -DDSMVC_SOLVE_ONLY=1)
    dsmvc_compile_vulkan_shader(
        convert_256 src/vulkan/shaders/convert.comp
        -DDSMVC_LOCAL_SIZE=256)
    dsmvc_compile_vulkan_shader(
        convert_128 src/vulkan/shaders/convert.comp
        -DDSMVC_LOCAL_SIZE=128)

    list(APPEND DSMVC_ENGINE_SOURCES
        src/vulkan/vulkan_executor.cpp
        ${DSMVC_VULKAN_GENERATED_HEADERS})
endif()

add_library(dsmvc_engine STATIC ${DSMVC_ENGINE_SOURCES})
target_compile_features(dsmvc_engine PUBLIC cxx_std_23)
target_include_directories(dsmvc_engine PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
set_target_properties(dsmvc_engine PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(DSMVC_ENABLE_CUDA)
    target_compile_definitions(dsmvc_engine PRIVATE DSMVC_HAS_CUDA=1)
    target_include_directories(dsmvc_engine PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda")
    target_link_libraries(dsmvc_engine PRIVATE
        CUDA::cudart_static ${_dsmvc_nvtx_target})
    set_target_properties(dsmvc_engine PROPERTIES
        CUDA_ARCHITECTURES "${_dsmvc_cuda_architectures}"
        CUDA_RUNTIME_LIBRARY Static
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON)
    target_compile_options(dsmvc_engine PRIVATE
        "$<$<COMPILE_LANGUAGE:CUDA>:-O3;-lineinfo>")
endif()

if(DSMVC_ENABLE_VULKAN)
    target_compile_definitions(dsmvc_engine PRIVATE DSMVC_HAS_VULKAN=1)
    target_include_directories(dsmvc_engine PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan"
        "${_dsmvc_vulkan_generated_dir}")
    target_link_libraries(dsmvc_engine PRIVATE Vulkan::Vulkan)
endif()

if(DSMVC_CAN_BUILD_AVX2)
    target_compile_definitions(dsmvc_engine PRIVATE DSMVC_HAS_AVX2_OBJECT=1)
    if(MSVC)
        set_source_files_properties(src/cpu_executor_avx2.cpp PROPERTIES
            COMPILE_OPTIONS "/arch:AVX2;/Ob3;/fp:strict")
    else()
        set_source_files_properties(src/cpu_executor_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
    endif()
elseif(DSMVC_CAN_BUILD_NEON)
    target_compile_definitions(dsmvc_engine PRIVATE DSMVC_HAS_NEON_OBJECT=1)
endif()

add_library(dsmvc MODULE src/vs_plugin.cpp)
target_compile_features(dsmvc PRIVATE cxx_std_23)
target_include_directories(dsmvc PRIVATE "${DSMVC_VAPOURSYNTH_INCLUDE_DIR}")
target_link_libraries(dsmvc PRIVATE dsmvc_engine)
if(DSMVC_ENABLE_CUDA)
    target_compile_definitions(dsmvc PRIVATE DSMVC_HAS_CUDA=1)
    target_include_directories(dsmvc PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/cuda")
    target_link_libraries(dsmvc PRIVATE ${_dsmvc_nvtx_target})
endif()
if(DSMVC_ENABLE_VULKAN)
    target_compile_definitions(dsmvc PRIVATE DSMVC_HAS_VULKAN=1)
endif()
set_target_properties(dsmvc PROPERTIES PREFIX "" OUTPUT_NAME "dsmvc")
if(APPLE)
    target_link_options(dsmvc PRIVATE
        "LINKER:-exported_symbol,_VapourSynthPluginInit2")
endif()
if(DSMVC_CAN_BUILD_NEON)
    target_compile_definitions(dsmvc PRIVATE DSMVC_HAS_NEON_OBJECT=1)
endif()

foreach(_target dsmvc_engine dsmvc)
    if(MSVC)
        target_compile_options(${_target} PRIVATE
            "$<$<COMPILE_LANGUAGE:CXX>:/W4;/permissive-;/EHsc;/fp:strict;/Zc:__cplusplus>")
        target_compile_definitions(${_target} PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
    else()
        target_compile_options(${_target} PRIVATE
            "$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-ffp-contract=off>")
    endif()
endforeach()

install(TARGETS dsmvc LIBRARY DESTINATION vapoursynth RUNTIME DESTINATION vapoursynth)
install(FILES dsmvc.py DESTINATION python)
install(FILES README.md LICENSE THIRD_PARTY_NOTICES.md DESTINATION share/dsmvc)

if(BUILD_TESTING)
    include(cmake/numerical_contract.cmake)

    add_executable(dsmvc_engine_tests tests/engine_tests.cpp)
    target_compile_features(dsmvc_engine_tests PRIVATE cxx_std_23)
    target_include_directories(dsmvc_engine_tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(dsmvc_engine_tests PRIVATE dsmvc_engine)
    add_test(NAME dsmvc_engine_tests COMMAND dsmvc_engine_tests)

    add_executable(dsmvc_simd_buffer_contract_tests
        tests/simd_buffer_contract_tests.cpp)
    target_compile_features(dsmvc_simd_buffer_contract_tests PRIVATE cxx_std_23)
    target_include_directories(dsmvc_simd_buffer_contract_tests PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src")
    target_link_libraries(dsmvc_simd_buffer_contract_tests PRIVATE dsmvc_engine)
    if(MSVC)
        target_compile_options(dsmvc_simd_buffer_contract_tests PRIVATE
            /W4 /permissive- /EHsc /fp:strict /Zc:__cplusplus)
        target_compile_definitions(dsmvc_simd_buffer_contract_tests PRIVATE
            _CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
    else()
        target_compile_options(dsmvc_simd_buffer_contract_tests PRIVATE
            -Wall -Wextra -Wpedantic -ffp-contract=off)
    endif()
    add_test(NAME dsmvc_simd_buffer_contract_tests
        COMMAND dsmvc_simd_buffer_contract_tests)

    dsmvc_add_numerical_contract_test(
        dsmvc_numerical_contract_tests
        BACKEND shared
        SOURCES tests/numerical_contract_tests.cpp)

    dsmvc_add_numerical_contract_test(
        dsmvc_f64_contract_tests
        BACKEND shared
        SOURCES tests/f64_contract_tests.cpp)

    dsmvc_add_numerical_contract_test(
        dsmvc_integer_contract_tests
        BACKEND shared
        SOURCES tests/integer_contract_tests.cpp)

    if(DSMVC_ENABLE_CUDA)
        add_test(NAME dsmvc_cuda_artifact_inventory
            COMMAND "${CMAKE_COMMAND}"
                "-DFATBIN=$<TARGET_FILE:dsmvc>"
                "-DCUOBJDUMP=${DSMVC_CUOBJDUMP_EXECUTABLE}"
                "-DNATIVE_ARCHITECTURES=${DSMVC_CUDA_ARCHITECTURES}"
                "-DPTX_ARCHITECTURES=${DSMVC_CUDA_PTX_ARCHITECTURES}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/verify_cuda_fatbin.cmake")

        add_executable(dsmvc_cuda_f64_conformance_tests
            tests/cuda_f64_conformance_tests.cpp)
        target_compile_features(
            dsmvc_cuda_f64_conformance_tests PRIVATE cxx_std_23)
        target_include_directories(dsmvc_cuda_f64_conformance_tests PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests")
        target_link_libraries(dsmvc_cuda_f64_conformance_tests PRIVATE
            dsmvc_engine CUDA::cudart_static)
        if(MSVC)
            target_compile_options(dsmvc_cuda_f64_conformance_tests PRIVATE
                /W4 /permissive- /EHsc /fp:strict /Zc:__cplusplus)
            target_compile_definitions(
                dsmvc_cuda_f64_conformance_tests PRIVATE
                _CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
        else()
            target_compile_options(dsmvc_cuda_f64_conformance_tests PRIVATE
                -Wall -Wextra -Wpedantic -ffp-contract=off)
        endif()
        add_test(NAME dsmvc_cuda_f64_conformance
            COMMAND dsmvc_cuda_f64_conformance_tests)
    endif()

    if(DSMVC_ENABLE_VULKAN)
        foreach(_spirv IN LISTS DSMVC_VULKAN_SPIRV)
            get_filename_component(_shader_name "${_spirv}" NAME_WE)
            add_test(NAME "dsmvc_vulkan_spirv_${_shader_name}"
                COMMAND "${DSMVC_SPIRV_VAL_EXECUTABLE}"
                    --target-env vulkan1.2 "${_spirv}")
        endforeach()

        function(dsmvc_add_vulkan_runtime_test suffix)
            add_test(NAME "dsmvc_vulkan_${suffix}" COMMAND dsmvc_engine_tests)
            set_tests_properties("dsmvc_vulkan_${suffix}" PROPERTIES
                ENVIRONMENT "${ARGN}")
        endfunction()
        dsmvc_add_vulkan_runtime_test(
            forced_fused "DSMVC_VULKAN_SPLIT_RHS=0")
        dsmvc_add_vulkan_runtime_test(
            forced_split_fallback
            "DSMVC_VULKAN_SPLIT_RHS=force;DSMVC_VULKAN_WORKGROUP=128;DSMVC_VULKAN_TIMELINE=0")
        dsmvc_add_vulkan_runtime_test(
            non_coherent "DSMVC_VULKAN_FORCE_NON_COHERENT=1")
        dsmvc_add_vulkan_runtime_test(
            cache_eviction
            "DSMVC_VULKAN_PLAN_CACHE_MB=1;DSMVC_VULKAN_INPUT_CACHE_MB=1;DSMVC_TEST_VULKAN_CACHE_EVICTION=1")

        foreach(_f64_test IN ITEMS
                vulkan_f64_runtime_tests
                vulkan_f64_policy_tests)
            add_executable(dsmvc_${_f64_test}
                "src/vulkan/${_f64_test}.cpp")
            target_compile_features(dsmvc_${_f64_test} PRIVATE cxx_std_23)
            target_include_directories(dsmvc_${_f64_test} PRIVATE
                "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan"
                "${_dsmvc_vulkan_generated_dir}")
            target_link_libraries(dsmvc_${_f64_test} PRIVATE
                dsmvc_engine Vulkan::Vulkan)
            if(MSVC)
                target_compile_options(dsmvc_${_f64_test} PRIVATE
                    /W4 /permissive- /EHsc /fp:strict /Zc:__cplusplus)
                target_compile_definitions(dsmvc_${_f64_test} PRIVATE
                    _CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
            else()
                target_compile_options(dsmvc_${_f64_test} PRIVATE
                    -Wall -Wextra -Wpedantic -ffp-contract=off)
            endif()
            add_test(NAME dsmvc_${_f64_test} COMMAND dsmvc_${_f64_test})
        endforeach()
    endif()

    if(NOT DSMVC_VS_PYTHON AND EXISTS "D:/okegui/OKEGui/tools/vapoursynth/python.exe")
        set(DSMVC_VS_PYTHON "D:/okegui/OKEGui/tools/vapoursynth/python.exe")
    endif()
    if(NOT DSMVC_BASELINE_PLUGIN AND EXISTS "D:/okegui/OKEGui/tools/vapoursynth/vapoursynth64/plugins/descale.dll")
        set(DSMVC_BASELINE_PLUGIN
            "D:/okegui/OKEGui/tools/vapoursynth/vapoursynth64/plugins/descale.dll")
    endif()
    if(DSMVC_VS_PYTHON AND DSMVC_BASELINE_PLUGIN)
        set(_dsmvc_vs_integration_args
            --plugin "$<TARGET_FILE:dsmvc>"
            --old-plugin "${DSMVC_BASELINE_PLUGIN}"
            --repo-root "${CMAKE_CURRENT_SOURCE_DIR}")
        if(DSMVC_BASELINE_WRAPPER)
            list(APPEND _dsmvc_vs_integration_args
                --old-wrapper "${DSMVC_BASELINE_WRAPPER}")
        endif()
        if(DSMVC_ENABLE_CUDA)
            list(APPEND _dsmvc_vs_integration_args --cuda-enabled)
        endif()
        if(DSMVC_ENABLE_VULKAN)
            list(APPEND _dsmvc_vs_integration_args --vulkan-enabled)
        endif()
        add_test(
            NAME dsmvc_vs_integration
            COMMAND "${DSMVC_VS_PYTHON}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/vs_integration.py"
                ${_dsmvc_vs_integration_args}
        )
        set_tests_properties(dsmvc_vs_integration PROPERTIES TIMEOUT 60)
    endif()
    set(_dsmvc_vs_backend_args)
    if(DSMVC_ENABLE_CUDA)
        list(APPEND _dsmvc_vs_backend_args --cuda-enabled)
    endif()
    if(DSMVC_ENABLE_VULKAN)
        list(APPEND _dsmvc_vs_backend_args --vulkan-enabled)
    endif()
    if(DSMVC_VS_PYTHON AND NOT DSMVC_ENABLE_METAL)
        add_test(
            NAME dsmvc_vs_api4_smoke
            COMMAND "${DSMVC_VS_PYTHON}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/vs_metal_integration.py"
                --plugin "$<TARGET_FILE:dsmvc>"
                --expect-metal disabled
                ${_dsmvc_vs_backend_args})
        set_tests_properties(dsmvc_vs_api4_smoke PROPERTIES TIMEOUT 60)
    endif()
    if(DSMVC_ENABLE_METAL AND DSMVC_VS_PYTHON)
        add_test(
            NAME dsmvc_vs_conditioned_float64_metal
            COMMAND "${DSMVC_VS_PYTHON}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/vs_metal_integration.py"
                --plugin "$<TARGET_FILE:dsmvc>"
                --expect-metal enabled
                ${_dsmvc_vs_backend_args}
                --conditioned-only)
        set_tests_properties(dsmvc_vs_conditioned_float64_metal PROPERTIES
            TIMEOUT 60
            RESOURCE_LOCK dsmvc_metal_device)
        add_test(
            NAME dsmvc_vs_metal_integration
            COMMAND "${DSMVC_VS_PYTHON}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/vs_metal_integration.py"
                --plugin "$<TARGET_FILE:dsmvc>"
                --expect-metal enabled
                ${_dsmvc_vs_backend_args})
        set_tests_properties(dsmvc_vs_metal_integration PROPERTIES
            TIMEOUT 300
            RESOURCE_LOCK dsmvc_metal_device)
    endif()
endif()

if(DSMVC_BUILD_BENCHMARKS)
    add_executable(dsmvc_cpu_profile_benchmark
        benchmarks/cpu_profile_benchmark.cpp)
    target_compile_features(dsmvc_cpu_profile_benchmark PRIVATE cxx_std_23)
    target_link_libraries(dsmvc_cpu_profile_benchmark PRIVATE dsmvc_engine)
    if(MSVC)
        target_compile_options(dsmvc_cpu_profile_benchmark PRIVATE
            /W4 /permissive- /EHsc /fp:strict /Zc:__cplusplus)
        target_compile_definitions(dsmvc_cpu_profile_benchmark PRIVATE
            _CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
    else()
        target_compile_options(dsmvc_cpu_profile_benchmark PRIVATE
            -Wall -Wextra -Wpedantic -ffp-contract=off)
    endif()

    add_executable(dsmvc_numerical_contract_benchmark
        benchmarks/numerical_contract_benchmark.cpp)
    target_compile_features(
        dsmvc_numerical_contract_benchmark PRIVATE cxx_std_23)
    target_include_directories(dsmvc_numerical_contract_benchmark PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    target_link_libraries(
        dsmvc_numerical_contract_benchmark PRIVATE dsmvc_engine)
    if(MSVC)
        target_compile_options(dsmvc_numerical_contract_benchmark PRIVATE
            /W4 /permissive- /EHsc /fp:strict /Zc:__cplusplus)
        target_compile_definitions(dsmvc_numerical_contract_benchmark PRIVATE
            _CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
    else()
        target_compile_options(dsmvc_numerical_contract_benchmark PRIVATE
            -Wall -Wextra -Wpedantic -ffp-contract=off -fno-fast-math)
    endif()

    add_executable(dsmvc_cpu_f64_avx2_benchmark
        benchmarks/cpu_f64_avx2_benchmark.cpp)
    target_compile_features(
        dsmvc_cpu_f64_avx2_benchmark PRIVATE cxx_std_23)
    target_link_libraries(
        dsmvc_cpu_f64_avx2_benchmark PRIVATE dsmvc_engine)
    if(MSVC)
        target_compile_options(dsmvc_cpu_f64_avx2_benchmark PRIVATE
            /W4 /permissive- /EHsc /fp:strict /Zc:__cplusplus)
        target_compile_definitions(dsmvc_cpu_f64_avx2_benchmark PRIVATE
            _CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
    else()
        target_compile_options(dsmvc_cpu_f64_avx2_benchmark PRIVATE
            -Wall -Wextra -Wpedantic -ffp-contract=off -fno-fast-math)
    endif()

    if(DSMVC_ENABLE_CUDA)
        add_executable(dsmvc_cuda_f64_executor_benchmark
            benchmarks/cuda_f64_executor_benchmark.cpp)
        target_compile_features(
            dsmvc_cuda_f64_executor_benchmark PRIVATE cxx_std_23)
        target_include_directories(
            dsmvc_cuda_f64_executor_benchmark PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src")
        target_link_libraries(dsmvc_cuda_f64_executor_benchmark PRIVATE
            dsmvc_engine CUDA::cudart_static)
        if(MSVC)
            target_compile_options(dsmvc_cuda_f64_executor_benchmark PRIVATE
                /W4 /permissive- /EHsc /fp:strict /Zc:__cplusplus)
            target_compile_definitions(
                dsmvc_cuda_f64_executor_benchmark PRIVATE
                _CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
        else()
            target_compile_options(dsmvc_cuda_f64_executor_benchmark PRIVATE
                -Wall -Wextra -Wpedantic -ffp-contract=off -fno-fast-math)
        endif()
    endif()

    if(DSMVC_ENABLE_VULKAN)
        foreach(_suffix IN ITEMS f64 f32)
            set(_vulkan_bench_target "dsmvc_vulkan_${_suffix}_benchmark")
            add_executable(${_vulkan_bench_target}
                src/vulkan/vulkan_f64_benchmark.cpp)
            target_compile_features(
                ${_vulkan_bench_target} PRIVATE cxx_std_23)
            target_include_directories(${_vulkan_bench_target} PRIVATE
                "${CMAKE_CURRENT_SOURCE_DIR}/src/vulkan"
                "${_dsmvc_vulkan_generated_dir}")
            target_link_libraries(${_vulkan_bench_target} PRIVATE
                dsmvc_engine Vulkan::Vulkan)
            if(_suffix STREQUAL "f32")
                target_compile_definitions(${_vulkan_bench_target} PRIVATE
                    DSMVC_VULKAN_F64_BENCHMARK_F32_ONLY=1)
            endif()
            if(MSVC)
                target_compile_options(${_vulkan_bench_target} PRIVATE
                    /W4 /permissive- /EHsc /fp:strict /Zc:__cplusplus)
                target_compile_definitions(${_vulkan_bench_target} PRIVATE
                    _CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
            else()
                target_compile_options(${_vulkan_bench_target} PRIVATE
                    -Wall -Wextra -Wpedantic -ffp-contract=off -fno-fast-math)
            endif()
        endforeach()
    endif()
endif()

if(DSMVC_ENABLE_METAL)
    if(NOT APPLE OR NOT DSMVC_TARGET_PROCESSOR MATCHES "^(arm64|aarch64)$")
        message(FATAL_ERROR
            "DSMVC_ENABLE_METAL requires a native Apple ARM64 build")
    endif()

    find_program(DSMVC_XCRUN xcrun REQUIRED)
    execute_process(
        COMMAND "${DSMVC_XCRUN}" metal --version
        RESULT_VARIABLE DSMVC_METAL_COMPILER_RESULT
        OUTPUT_QUIET ERROR_QUIET)
    execute_process(
        COMMAND "${DSMVC_XCRUN}" metallib --version
        RESULT_VARIABLE DSMVC_METALLIB_RESULT
        OUTPUT_QUIET ERROR_QUIET)
    if(NOT DSMVC_METAL_COMPILER_RESULT EQUAL 0
       OR NOT DSMVC_METALLIB_RESULT EQUAL 0)
        message(FATAL_ERROR
            "DSMVC_ENABLE_METAL requires the Xcode Metal toolchain")
    endif()

    enable_language(OBJCXX)
    find_library(DSMVC_METAL_FRAMEWORK Metal REQUIRED)
    find_library(DSMVC_FOUNDATION_FRAMEWORK Foundation REQUIRED)

    set(DSMVC_METAL_GENERATED_DIR
        "${CMAKE_CURRENT_BINARY_DIR}/generated/metal-routes")
    set(DSMVC_METAL_AIR
        "${DSMVC_METAL_GENERATED_DIR}/dsmvc_metal_routes.air")
    set(DSMVC_METALLIB
        "${DSMVC_METAL_GENERATED_DIR}/dsmvc_metal_routes.metallib")
    set(DSMVC_METALLIB_HEADER
        "${DSMVC_METAL_GENERATED_DIR}/dsmvc_metal_routes_metallib.hpp")

    add_custom_command(
        OUTPUT "${DSMVC_METAL_AIR}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory
                "${DSMVC_METAL_GENERATED_DIR}"
        COMMAND "${DSMVC_XCRUN}" -sdk macosx metal
                -Werror -fno-fast-math -ffp-contract=off
                -c "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/metal_routes.metal"
                -o "${DSMVC_METAL_AIR}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/metal/metal_routes.metal"
        VERBATIM)
    add_custom_command(
        OUTPUT "${DSMVC_METALLIB}"
        COMMAND "${DSMVC_XCRUN}" -sdk macosx metallib
                "${DSMVC_METAL_AIR}" -o "${DSMVC_METALLIB}"
        DEPENDS "${DSMVC_METAL_AIR}"
        VERBATIM)
    add_custom_command(
        OUTPUT "${DSMVC_METALLIB_HEADER}"
        COMMAND "${CMAKE_COMMAND}"
                "-DINPUT=${DSMVC_METALLIB}"
                "-DOUTPUT=${DSMVC_METALLIB_HEADER}"
                -DSYMBOL=dsmvc_metal_routes_metallib
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.cmake"
        DEPENDS "${DSMVC_METALLIB}"
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.cmake"
        VERBATIM)

    target_sources(dsmvc PRIVATE
        src/metal_scheduler_apple.mm
        "${DSMVC_METALLIB_HEADER}")
    target_sources(dsmvc_engine PRIVATE src/metal_device_apple.mm)
    target_include_directories(dsmvc PRIVATE "${DSMVC_METAL_GENERATED_DIR}")
    target_compile_definitions(dsmvc PRIVATE DSMVC_HAS_METAL=1)
    target_compile_definitions(dsmvc_engine PRIVATE DSMVC_HAS_METAL=1)
    set_source_files_properties(
        src/metal_scheduler_apple.mm src/metal_device_apple.mm
        PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
    target_link_libraries(dsmvc_engine PRIVATE
        "${DSMVC_METAL_FRAMEWORK}"
        "${DSMVC_FOUNDATION_FRAMEWORK}")
    target_link_libraries(dsmvc PRIVATE
        "${DSMVC_METAL_FRAMEWORK}"
        "${DSMVC_FOUNDATION_FRAMEWORK}")

    if(DSMVC_BUILD_BENCHMARKS)
        add_executable(dsmvc_metal_route_benchmark
            benchmarks/metal_route_benchmark.mm
            "${DSMVC_METALLIB_HEADER}")
        target_compile_features(dsmvc_metal_route_benchmark PRIVATE cxx_std_23)
        target_include_directories(dsmvc_metal_route_benchmark PRIVATE
            "${DSMVC_METAL_GENERATED_DIR}")
        target_compile_options(dsmvc_metal_route_benchmark PRIVATE
            -fobjc-arc -Wall -Wextra -Wpedantic -ffp-contract=off)
        target_link_libraries(dsmvc_metal_route_benchmark PRIVATE
            dsmvc_engine
            "${DSMVC_METAL_FRAMEWORK}"
            "${DSMVC_FOUNDATION_FRAMEWORK}")

        add_executable(dsmvc_metal_yuv_benchmark
            benchmarks/metal_yuv_benchmark.mm
            "${DSMVC_METALLIB_HEADER}")
        target_compile_features(dsmvc_metal_yuv_benchmark PRIVATE cxx_std_23)
        target_include_directories(dsmvc_metal_yuv_benchmark PRIVATE
            "${DSMVC_METAL_GENERATED_DIR}")
        target_compile_options(dsmvc_metal_yuv_benchmark PRIVATE
            -fobjc-arc -Wall -Wextra -Wpedantic -ffp-contract=off)
        target_link_libraries(dsmvc_metal_yuv_benchmark PRIVATE
            dsmvc_engine
            "${DSMVC_METAL_FRAMEWORK}"
            "${DSMVC_FOUNDATION_FRAMEWORK}")
    endif()

    if(BUILD_TESTING)
        add_test(NAME dsmvc_metal_artifact_inventory
            COMMAND "${CMAKE_COMMAND}"
                "-DMETALLIB=${DSMVC_METALLIB}"
                "-DEXPECTED_SYMBOLS=${CMAKE_CURRENT_SOURCE_DIR}/cmake/metal_entrypoints.txt"
                "-DXCRUN=${DSMVC_XCRUN}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/verify_metal_metallib.cmake")
        set_tests_properties(
            dsmvc_metal_artifact_inventory PROPERTIES TIMEOUT 30)

        dsmvc_add_numerical_contract_test(
            dsmvc_metal_numerical_contract_tests
            BACKEND metal
            SOURCES
                tests/metal_numerical_contract_tests.mm
                src/metal_float_executor_apple.mm
                src/metal_yuv_executor_apple.mm
                "${DSMVC_METALLIB_HEADER}"
            LIBRARIES
                "${DSMVC_METAL_FRAMEWORK}"
                "${DSMVC_FOUNDATION_FRAMEWORK}"
            INCLUDE_DIRECTORIES
                "${DSMVC_METAL_GENERATED_DIR}")
        set_source_files_properties(
            src/metal_float_executor_apple.mm src/metal_yuv_executor_apple.mm
            PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
        set_tests_properties(
            dsmvc_metal_numerical_contract_tests PROPERTIES
            TIMEOUT 60
            RESOURCE_LOCK dsmvc_metal_device)

        add_executable(dsmvc_hybrid_batch_coordinator_tests
            tests/hybrid_batch_coordinator_tests.cpp
            src/hybrid_batch_coordinator.cpp)
        target_compile_features(
            dsmvc_hybrid_batch_coordinator_tests PRIVATE cxx_std_23)
        target_compile_options(dsmvc_hybrid_batch_coordinator_tests PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
        add_test(NAME dsmvc_hybrid_batch_coordinator_tests
            COMMAND dsmvc_hybrid_batch_coordinator_tests)

        add_executable(dsmvc_metal_yuv_staging_tests
            tests/metal_yuv_staging_tests.cpp)
        target_compile_features(dsmvc_metal_yuv_staging_tests PRIVATE cxx_std_23)
        target_include_directories(dsmvc_metal_yuv_staging_tests PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${CMAKE_CURRENT_SOURCE_DIR}/include")
        target_compile_options(dsmvc_metal_yuv_staging_tests PRIVATE
            -Wall -Wextra -Wpedantic -Werror)
        add_test(NAME dsmvc_metal_yuv_staging_tests
            COMMAND dsmvc_metal_yuv_staging_tests)

        add_executable(dsmvc_metal_scheduler_tests
            tests/metal_scheduler_tests.cpp
            src/metal_scheduler_apple.mm
            "${DSMVC_METALLIB_HEADER}")
        target_compile_features(dsmvc_metal_scheduler_tests PRIVATE cxx_std_23)
        target_compile_definitions(dsmvc_metal_scheduler_tests PRIVATE
            DSMVC_METAL_SCHEDULER_TESTING=1)
        target_include_directories(dsmvc_metal_scheduler_tests PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            "${DSMVC_METAL_GENERATED_DIR}")
        target_compile_options(dsmvc_metal_scheduler_tests PRIVATE
            -Wall -Wextra -Wpedantic -Werror -ffp-contract=off)
        target_link_libraries(dsmvc_metal_scheduler_tests PRIVATE
            dsmvc_engine
            "${DSMVC_METAL_FRAMEWORK}"
            "${DSMVC_FOUNDATION_FRAMEWORK}")
        add_test(NAME dsmvc_metal_scheduler_tests
            COMMAND dsmvc_metal_scheduler_tests)
        set_tests_properties(dsmvc_metal_scheduler_tests PROPERTIES
            TIMEOUT 60
            RESOURCE_LOCK dsmvc_metal_device)
    endif()
endif()
