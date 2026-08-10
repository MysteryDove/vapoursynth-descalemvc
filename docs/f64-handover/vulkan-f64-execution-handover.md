# Vulkan Float64 Execution Handover

> Status: capability-gated native retained-F64 Vulkan execution is
> implemented. The staged sections below are retained as design history; the
> arithmetic and acceptance contracts describe the current implementation.

## Goal

Add native Vulkan Double execution on devices that explicitly expose the
required capabilities. Preserve the current Float32 pipelines on every device,
and return a precise error for an explicit F64 request on unsupported hardware.

Vulkan FP64 is optional. `shaderFloat64` is defined in the official
[Vulkan feature specification](https://docs.vulkan.org/spec/latest/chapters/features.html),
and Float64 rounding, denormal, and signed-zero/NaN/Inf properties are described
in the [device limits and float controls specification](https://docs.vulkan.org/spec/latest/chapters/limits.html).

## Current State

- [vulkan_executor.cpp](../../src/vulkan/vulkan_executor.cpp) creates only F32
  transpose, RHS, inverse/solve, and conversion pipelines.
- [inverse.comp](../../src/vulkan/shaders/inverse.comp) and
  [rhs.comp](../../src/vulkan/shaders/rhs.comp) decode float values from
  word-addressed SSBO storage and use explicit `fma`.
- Plan packing stores Float32 weights, lower/upper bands, and inverse diagonal.
- Runtime device selection does not establish an F64 execution contract.
- [executor.cpp](../../src/executor.cpp) rejects retained Float64 plans before
  `VulkanExecutor` sees them.
- The shared prework candidate based on `e411298` retains the original Double
  normal matrix and provides immutable contract fixtures, an ordered F32/F64
  oracle, an axis benchmark, and backend-local numerical-test registration.
- There is no target-device Vulkan F64 evidence for this candidate.

## Lane Entry Order

Begin only after the shared candidate is frozen. This one Vulkan lane performs:

1. query the exact device float controls and converge current F32 shaders to the
   shared ordered-FMA fixtures, including tails and workgroup variants;
2. validate and disassemble the exact SPIR-V, retaining F32 changes only when
   plugin E2E stays within `3%` and executor-only cases within `5%`;
3. implement capability-gated native Vulkan Float64 end to end; and
4. evaluate mixed-precision IR only after native Float64 correctness and timing
   are established on target hardware.

Use `dsmvc_add_numerical_contract_test()` for a proposed Vulkan-local
conformance target and keep shared fixture hashes immutable. The integrator
performs top-level registration and later routing changes.

## Ownership

The Vulkan implementer owns:

- `src/vulkan/vulkan_executor.cpp`
- `src/vulkan/vulkan_executor.hpp`
- new files under `src/vulkan/shaders/` with an unambiguous F64 name
- new Vulkan-only tests, benchmarks, and artifact scripts named with
  `vulkan_f64`

Do not edit shared dispatch, CMake, public headers, planner code, plugin code,
existing shared tests, other backend files, README, or release reports. The
integrator registers shaders and removes the shared Vulkan rejection only after
admission.

## Capability Contract

At physical-device selection, query and retain:

- `VkPhysicalDeviceFeatures::shaderFloat64`;
- `VkPhysicalDeviceFloatControlsProperties::shaderRoundingModeRTEFloat64`;
- `shaderSignedZeroInfNanPreserveFloat64`;
- `shaderDenormPreserveFloat64`; and
- the relevant storage-buffer alignment and range limits.

Use `vkGetPhysicalDeviceFeatures2` and `vkGetPhysicalDeviceProperties2` so the
result belongs to the selected device. Do not infer support from API version,
GPU vendor, shader compiler success, or a software implementation.

The initial strict route should require native Float64, round-to-nearest-even,
and preservation behavior compatible with the CPU oracle. If a device lacks a
required float-control property, explicit F64 execution returns a capability
error that names the missing property. F32 pipelines remain available.

## Required Design

### Separate shader modules and pipelines

Compile separate F64 modules, such as:

- `transpose_f64.comp`
- `rhs_f64.comp`
- `inverse_f64.comp`
- `convert_f64.comp`

Do not add `double` declarations to the existing F32 modules. A module that
contains Float64 SPIR-V requires the feature even if host dispatch never selects
that function. Separate modules keep F32 startup and compatibility unchanged.

Create F64 pipelines only after the selected device passes capability probing.
F32-only devices must not fail backend construction because an optional F64
pipeline was requested eagerly.

### Double plan and workspace layout

The existing SSBOs are arrays of 32-bit words. Either introduce typed Double
buffers with correct `std430` alignment or encode each Double as an aligned
two-word value and reconstruct it in the F64 shader. Whichever representation
is selected must have one host-side checked layout definition and matching
shader helpers.

Upload retained `transpose_weights_f64` and raw `ldlt_bands_f64`; never widen
the F32 lower/upper arrays. Include precision and all Double byte counts in plan
cache identity, arena allocation, eviction, and diagnostics.

The F64 workspace uses two words per Double or a typed Double element. All
offset arithmetic must be checked in bytes and words, including non-coherent
flush/invalidate alignment.

### Arithmetic and traversal

The scalar CPU path is the arithmetic oracle. Match its separately rounded
multiply and add/subtract order and its raw-band indexing. The current shader
uses `precise` operations with `NoContraction`; retained diagonal division uses
a reciprocal plus residual correction because the supported NVIDIA toolchain
fails to compile the direct FP64 `OpFDiv` form. Inspect generated SPIR-V instead
of assuming source qualifiers survived compilation.

For 2D, once either axis is risky, transpose/normalize to Double, execute both
axes in Double, and convert only after the final solve. A safe companion axis
uses its F32 fields widened in shader execution. Integer normalization and
final scale/clamp/round execute in Double with the existing conversion
constants widened from float.

### Dispatch and failure behavior

`VulkanExecutor::prepare` may cache F64 host layout independently of device
support, but the first operation that requires an unavailable F64 feature must
fail before command recording or large allocation. Every rows, columns, 2D,
U8, and U16 entrypoint validates the plan precision.

Pipeline/cache keys include precision. Descriptor sets and command-buffer
reuse must not bind an F32 plan or workspace to an F64 pipeline.

## Execution Stages

### VK-0: Capability probe

1. Add a backend-local capability report for the selected physical device.
2. Add deterministic tests for supported and unsupported feature structs by
   isolating policy from Vulkan calls.
3. Define the exact user-facing error for missing `shaderFloat64` and each hard
   float-control requirement.

Do not change shared routing in this stage.

### VK-1: Shader and layout baseline

1. Add generic F64 RHS and solve shaders.
2. Add one checked host/shader plan layout for offsets, indices, Double weights,
   raw bands, and optional reciprocals.
3. Compile with Vulkan 1.2 target settings and validate each SPIR-V module.
4. Disassemble the result and assert `OpTypeFloat 64`, required capabilities,
   and the intended contraction decorations.

### VK-2: Rows and columns

1. Add row and column F64 execution with arbitrary valid strides and tails.
2. Cover B1/B3/B5/B7 and generic bandwidth.
3. Compare with scalar CPU F64 on the same frozen plan.
4. Exercise coherent and forced non-coherent memory paths.

### VK-3: Complete 2D and integer paths

1. Add Double transpose/normalization and a Double intermediate.
2. Cover risky/risky and both mixed-axis directions.
3. Add Float32 and integer final conversion after the last Double solve.
4. Exercise fused and forced split-RHS modes, timeline on/off, input cache,
   eviction, and re-upload.

### VK-4: Integration and artifact gates

1. Give the integrator a complete shader/generated-header/test inventory.
2. Register every F64 SPIR-V artifact and `spirv-val --target-env vulkan1.2`
   test through the integrator lane.
3. Preserve existing F32 shader hashes or explain any intentional rebuild-only
   difference with unchanged disassembly.
4. Ask the integrator to delegate the shared rejection only after all backend
   tests pass.

### VK-5: Hardware and performance admission

Run a fresh Release plugin on real Windows or Linux Vulkan hardware with the
required features. Record source SHA, plugin SHA-256, shader hashes, Vulkan
loader/runtime version, driver, GPU and device ID, queried feature/property
values, command lines, warmups, and raw paired samples.

Software Vulkan is useful for deterministic CI but is not target-hardware
evidence. Test at least one unsupported physical or mocked capability profile
to prove explicit failure semantics.

Report executor timing and plugin E2E separately. Do not add an automatic
Vulkan F64 route unless paired plugin median throughput is at least `1.03x` CPU
F64 and existing F32/CPU controls remain within 3%.

### VK-6: optional mixed-precision IR

This stage is optional and begins only after VK-0 through VK-5 establish a
correct native Vulkan Float64 fallback on the same physical device. Follow
[mixed-precision-ir-handover.md](mixed-precision-ir-handover.md).

1. Use queried Float32/Float64 controls to define the residual and correction
   contract; capability presence is not performance evidence.
2. Upload original Double transpose and unfactored normal bands for native
   Float64 residual/update shaders. Reuse F32 factors only for correction.
3. Retain complete residual histories, iteration counts, and forced fallback
   evidence.
4. On IR failure, execute native Vulkan Float64. Explicit Vulkan never falls
   back to CPU.
5. Retain IR only when it passes the same numerical gates and improves plugin
   E2E on the exact target artifact. Native Float64 remains the baseline.

## Numerical Matrix

| Dimension | Required cases |
|---|---|
| Capability | full strict F64, missing `shaderFloat64`, missing each required float control |
| Precision | conditioned automatic, safe automatic, forced F32, forced F64 |
| Operation | rows, columns, one-axis 2D, two-axis 2D |
| Axis pair | risky/risky, risky/safe, safe/risky |
| Bandwidth | 1, 3, 5, 7, generic custom support |
| Samples | float, U8, U10, U16 |
| Memory | coherent, non-coherent, cache hit, cache eviction/re-upload |
| Scheduling | fused RHS, split RHS, timeline on/off, 128/256 workgroup variants |

Native Vulkan F64 final float output should be identical to scalar CPU output
or within one output ULP with no high-precision-oracle regression. Integer
output is within one code of scalar CPU F64. Repeated execution on the same
Vulkan route remains bit exact. Any NaN, infinity, validation-layer error, or
device loss fails admission.

## Explicit Backend Semantics

- `backend=vulkan` executes Vulkan or returns an error.
- Unsupported F64 is not a reason to run CPU.
- `backend=auto` remains unchanged.
- A device may support F32 Vulkan while rejecting only retained F64 plans.
- Plugin evidence must distinguish backend availability from per-plan F64
  capability.

## Stop Conditions

Stop without shared integration if:

- device capability is inferred rather than queried;
- an F64 module is required on F32-only devices;
- retained plans use widened F32 coefficients;
- precision is missing from a plan, pipeline, descriptor, or command cache key;
- a 2D intermediate is Float32;
- integer output differs from scalar CPU F64 by more than one code;
- SPIR-V inspection does not prove the intended Float64 operations; or
- explicit Vulkan can silently execute on CPU.

## Definition of Done

Vulkan F64 is complete only after backend-local and shared tests, validation
layers, SPIR-V validation/disassembly, API4 plugin tests, unsupported-device
tests, and a reproducible real-hardware run all pass. Shader compilation alone
is not completion.
