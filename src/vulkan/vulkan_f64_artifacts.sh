#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
sdk=${VULKAN_SDK:-/home/owen/Downloads/vulkansdk-linux-x86_64-1.4.357.0/1.4.357.0/x86_64}
glslc=${DSMVC_GLSLC_EXECUTABLE:-"$sdk/bin/glslc"}
spirv_dis=${DSMVC_SPIRV_DIS_EXECUTABLE:-"$sdk/bin/spirv-dis"}
spirv_as=${DSMVC_SPIRV_AS_EXECUTABLE:-"$sdk/bin/spirv-as"}
spirv_val=${DSMVC_SPIRV_VAL_EXECUTABLE:-"$sdk/bin/spirv-val"}
output=${1:-"$root/build-vulkan-f64/generated/vulkan_f64"}
embed=${DSMVC_VULKAN_F64_EMBED:-0}
capability_report_file=${DSMVC_VULKAN_F64_CAPABILITY_REPORT_FILE:-}

runtime_capability_report=not-recorded
if [[ -n "$capability_report_file" ]]; then
    if [[ ! -r "$capability_report_file" ]]; then
        echo "Float64 capability report is not readable: $capability_report_file" >&2
        exit 1
    fi
    runtime_capability_report=$(grep -m1 \
        'shaderDenormPreserveFloat64=[01]' "$capability_report_file" || true)
    if [[ -z "$runtime_capability_report" ]]; then
        echo "Float64 capability report does not record denorm preservation" >&2
        exit 1
    fi
fi

mkdir -p "$output"

compile_module() {
    local name=$1
    local source=$2
    shift 2
    local plain="$output/$name.plain.spv"
    local assembly="$output/$name.plain.spvasm"
    local controlled="$output/$name.spvasm"
    local binary="$output/$name.spv"

    "$glslc" --target-env=vulkan1.2 -O "$@" -o "$plain" "$root/$source"
    "$spirv_dis" "$plain" -o "$assembly"
    local entry
    entry=$(awk '$1 == "OpEntryPoint" && $2 == "GLCompute" { print $3; exit }' "$assembly")
    if [[ -z "$entry" ]]; then
        echo "$name has no GLCompute entry point" >&2
        return 1
    fi
    awk -v entry="$entry" '
        !capabilities && (index($0, "OpExtInstImport") != 0 || $1 == "OpMemoryModel") {
            print "               OpCapability SignedZeroInfNanPreserve"
            print "               OpCapability RoundingModeRTE"
            capabilities = 1
        }
        !modes && $1 == "OpExecutionMode" {
            print "               OpExecutionMode " entry " SignedZeroInfNanPreserve 64"
            print "               OpExecutionMode " entry " RoundingModeRTE 64"
            modes = 1
        }
        { print }
    ' "$assembly" > "$controlled"
    "$spirv_as" --target-env vulkan1.2 "$controlled" -o "$binary"
    "$spirv_val" --target-env vulkan1.2 "$binary"
    "$spirv_dis" "$binary" -o "$output/$name.dis"
    for expected in \
        'OpCapability Float64' \
        'OpCapability SignedZeroInfNanPreserve' \
        'OpCapability RoundingModeRTE' \
        'OpExecutionMode .* SignedZeroInfNanPreserve 64' \
        'OpExecutionMode .* RoundingModeRTE 64' \
        'OpTypeFloat 64'; do
        if ! grep -Eq "$expected" "$output/$name.dis"; then
            echo "$name is missing SPIR-V evidence: $expected" >&2
            return 1
        fi
    done
    # Denorm preservation is probe-only. On devices reporting false, Float64
    # subnormal intermediates may flush to zero; normal-range values are unaffected.
    if grep -q 'DenormPreserve' "$output/$name.dis"; then
        echo "$name unexpectedly requires SPIR-V DenormPreserve" >&2
        return 1
    fi
    local generated_header="$output/vulkan_${name}_spv.hpp"
    local source_header="$root/src/vulkan/vulkan_${name}_spv.hpp"
    "${CMAKE_COMMAND:-cmake}" \
        "-DINPUT=$binary" \
        "-DOUTPUT=$generated_header" \
        "-DSYMBOL=vulkan_${name}_spv" \
        -P "$root/cmake/embed_spirv.cmake"
    if [[ "$embed" == 1 ]]; then
        "${CMAKE_COMMAND:-cmake}" -E copy_if_different \
            "$generated_header" "$source_header"
    fi
    if ! cmp -s "$generated_header" "$source_header"; then
        echo "$name embedded header does not match the validated SPIR-V" >&2
        return 1
    fi
}

compile_module transpose_f64 src/vulkan/shaders/transpose_f64.comp
compile_module rhs_f64 src/vulkan/shaders/rhs_f64.comp
compile_module inverse_f64 src/vulkan/shaders/inverse_f64.comp \
    -DDSMVC_SOLVE_ONLY=0
compile_module solve_f64 src/vulkan/shaders/inverse_f64.comp \
    -DDSMVC_SOLVE_ONLY=1
compile_module convert_f64 src/vulkan/shaders/convert_f64.comp

for module in rhs_f64 inverse_f64 solve_f64; do
    if ! grep -q 'OpFMul %double' "$output/$module.dis" \
        || ! grep -q 'NoContraction' "$output/$module.dis"; then
        echo "$module does not preserve ordered Double multiply/add steps" >&2
        exit 1
    fi
done
if ! grep -q 'NoContraction' "$output/convert_f64.dis"; then
    echo "convert_f64 does not preserve the separate Double multiply/add" >&2
    exit 1
fi

{
    echo "source_sha=$(git -C "$root" rev-parse HEAD)"
    if git -C "$root" diff --quiet -- src/vulkan; then
        echo "source_state=clean"
    else
        echo "source_state=dirty"
    fi
    echo "float64_denorm_preserve_policy=probe-only"
    echo "float64_denorm_preserve_spirv=not-requested"
    echo "float64_denorm_boundary=subnormal intermediates may flush to zero; normal-range values unaffected"
    echo "runtime_capability_report=$runtime_capability_report"
    "$glslc" --version
    "$spirv_val" --version
    sha256sum \
        "$root/src/vulkan/shaders/transpose_f64.comp" \
        "$root/src/vulkan/shaders/rhs_f64.comp" \
        "$root/src/vulkan/shaders/inverse_f64.comp" \
        "$root/src/vulkan/shaders/convert_f64.comp" \
        "$output/transpose_f64.spv" \
        "$output/rhs_f64.spv" \
        "$output/inverse_f64.spv" \
        "$output/solve_f64.spv" \
        "$output/convert_f64.spv" \
        "$root/src/vulkan/vulkan_transpose_f64_spv.hpp" \
        "$root/src/vulkan/vulkan_rhs_f64_spv.hpp" \
        "$root/src/vulkan/vulkan_inverse_f64_spv.hpp" \
        "$root/src/vulkan/vulkan_solve_f64_spv.hpp" \
        "$root/src/vulkan/vulkan_convert_f64_spv.hpp"
} > "$output/inventory.txt"

echo "Vulkan Float64 artifacts validated in $output"
