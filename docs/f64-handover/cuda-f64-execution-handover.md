# CUDA Float64 Execution Handover

> Status: native retained-F64 CUDA execution is implemented. The staged
> sections below are retained as design history; the arithmetic and acceptance
> contracts describe the current implementation.

## Goal

Add native CUDA Double execution for retained Float64 plans while preserving
the current Float32 kernels and strict explicit-backend semantics. This branch
does not change planning, padding, plugin arguments, or automatic routing.

CUDA compute capability 2.0 and later supports Double arithmetic; this project
currently emits native code for SM 75, 86, 89, and 120 and PTX for SM 75 and
120. NVIDIA documents IEEE-754 Double and DFMA behavior in its
[Floating Point and IEEE 754 guide](https://docs.nvidia.com/cuda/floating-point/index.html).
Compiler behavior and per-translation-unit controls are documented in the
[NVCC guide](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html).

## Current State

- [cuda_executor.cpp](../../src/cuda/cuda_executor.cpp) packs only
  `transpose_weights`, `lower_ld`, `upper_l`, and `inverse_diagonal` as float.
- [dsmvc_cuda.cu](../../src/cuda/dsmvc_cuda.cu) stores Float32 workspace data
  and exposes only F32 inverse, RHS, solve, transpose, and conversion kernels.
- [cuda_launch.hpp](../../src/cuda/cuda_launch.hpp) exposes only float plan and
  workspace pointers.
- [executor.cpp](../../src/executor.cpp) rejects a retained Float64 plan before
  `CudaExecutor` receives `prepare` or an execution call.
- The shared prework candidate based on `e411298` retains the original Double
  normal matrix and provides immutable contract fixtures, an ordered F32/F64
  oracle, an axis benchmark, and backend-local numerical-test registration.
- There is no CUDA target-device F64 evidence for this candidate.

## Lane Entry Order

Begin only after the shared candidate is frozen. This one CUDA lane performs:

1. converge the current F32 RHS/lower/upper grouping to the shared ordered-FMA
   fixtures, including tails and fused/split variants;
2. inspect the exact fatbin/SASS/PTX and retain the F32 change only when plugin
   E2E stays within `3%` and executor-only admitted geometries within `5%`;
3. implement and validate native CUDA Double end to end; and
4. evaluate mixed-precision IR only after native FP64 correctness and timing are
   established on target hardware.

Use `dsmvc_add_numerical_contract_test()` for a proposed CUDA-local conformance
target and keep shared fixture hashes immutable. The integrator performs the
top-level registration and later routing change.

## Ownership

The CUDA implementer owns:

- `src/cuda/cuda_executor.cpp`
- `src/cuda/cuda_executor.hpp`
- `src/cuda/cuda_kernel.hpp`
- `src/cuda/cuda_launch.hpp`
- `src/cuda/dsmvc_cuda.cu`
- new CUDA-only test, benchmark, and artifact files named with `cuda_f64`

Do not edit `src/executor.cpp`, `CMakeLists.txt`, public engine headers,
planner code, plugin code, existing shared tests, Metal/Vulkan code, README, or
release reports. Ask the integrator to register new files and remove the shared
rejection after CUDA passes admission.

## Required Design

### Precision-specific packed plans

Extend the private `PackedPlan` with an explicit precision tag and disjoint
storage layouts. A retained plan uploads:

- existing `transpose_offsets` and `transpose_indices` as 32-bit integers;
- `transpose_weights_f64` as `double`;
- `ldlt_bands_f64` as `double`; and
- `inverse_diagonal_f64` as `double` only if a multiplication variant is
  admitted against the scalar division oracle.

Prefer uploading the raw `ldlt_bands_f64` layout and using the same indices as
`detail::inverse_axis_f64`. Do not manufacture Double data by widening
`lower_ld`, `upper_l`, or `inverse_diagonal`.

The cache key must continue to distinguish the immutable `AxisPlan`. Validate
all byte-size products, align Double regions, and include Double storage in
cache/residency accounting.

### Double kernels

Add precision-specific entrypoints rather than templating the existing exported
F32 names in a way that changes their generated code. Required operations are:

- F32 source to F64 transpose for float input;
- U8/U16 normalization directly to F64 for integer input;
- horizontal and vertical F64 RHS accumulation;
- horizontal and vertical banded LDLT solve using the retained bands;
- F64-to-F32 final conversion;
- F64-to-U8/U16 scale, clamp, round-to-nearest-even, and conversion; and
- fused and split-RHS variants where the current executor can select either.

Parallelize independent rows or columns. Do not parallelize the recurrence
inside one axis solve. Specialize half-bandwidths 1, 3, 5, and 7 only after the
generic F64 kernel passes conformance.

For a 2D call where either axis requires Float64, both axes and the complete
intermediate stay Double. If the companion axis is safe, widen its Float32
fields during execution exactly as the CPU reference does; it has no retained
Double fields.

### Arithmetic policy

The scalar CPU path is the arithmetic oracle. CUDA uses explicit
round-to-nearest Double multiply followed by add/subtract, preserves the frozen
coefficient/band traversal, and applies the retained pivot as division by
`D + epsilon`. Do not infer the contract from an NVCC default and do not alter
existing F32 translation-unit behavior globally.

A DFMA variant may be evaluated separately. Retain it only if
final float output is identical or within one output ULP, integer output is
within one code of scalar CPU F64, and the independent high-precision error
does not regress.

### Executor dispatch

`CudaExecutor::prepare` must validate and upload either F32 or F64 plan data.
Every rows, columns, 2D, U8, and U16 entrypoint must validate precision before
launch. A malformed or unsupported precision state raises a CUDA-specific
error; it never selects CPU.

Keep F32 buffers, kernels, cache behavior, graph/stream behavior, and input
cache behavior unchanged. Precision must be visible in any reusable graph or
launch-cache key so an F32 graph cannot be reused for an F64 call.

## Execution Stages

### CUDA-0: Local conformance shell

1. Add backend-local tests that instantiate `CudaExecutor` directly.
2. Import the integrator-owned conditioned, forced-F64, mixed-axis, and integer
   fixtures without editing their source.
3. Confirm the current branch fails only because CUDA lacks precision support.

Stop if tests require planner or plugin semantic changes.

### CUDA-1: Plan packing

1. Add checked Double plan storage and upload.
2. Make `prepare`, lazy upload, cache eviction, and `seal` precision aware.
3. Add tests for plan-cache identity, byte accounting, upload completion, and
   concurrent prepare.

No execution routing changes belong in this stage.

### CUDA-2: Rows and columns

1. Add generic F64 row and column kernels.
2. Match CPU accumulation and band traversal order.
3. Cover vector counts below a block, odd tails, arbitrary valid strides, and
   half-bandwidths outside 1/3/5/7 through the generic path.
4. Compare final F32 output with scalar CPU F64.

### CUDA-3: Complete 2D and integer paths

1. Allocate a Double intermediate when either axis is risky.
2. Cover risky/risky and risky/safe axis pairs in both orientations.
3. Normalize integer input in Double and perform final conversion after the
   vertical Double solve.
4. Verify buffered and streamed public calls reach the same CUDA implementation
   and produce bit-identical integer output.

### CUDA-4: Integration and artifacts

1. Give the integrator the exact new source/header/test list for CMake.
2. Add an artifact inventory that proves all required F64 symbols are embedded
   for the configured native and PTX targets.
3. Use `cuobjdump` or equivalent disassembly to confirm Double instructions in
   F64 entrypoints and no unintended changes to F32 entrypoints.
4. Only then ask the integrator to remove CUDA's centralized F64 rejection.

### CUDA-5: Hardware and performance admission

Run on at least one real SM75-or-newer GPU on a supported Windows or Linux
VapourSynth environment. Record source SHA, plugin SHA-256, CUDA toolkit,
driver, GPU name/compute capability, clocks/power mode, command lines, warmups,
and all raw paired samples.

Report separately:

- kernel/executor timing without plugin overhead;
- explicit `backend=cuda` plugin E2E against `backend=cpu` F64; and
- unchanged F32 explicit CUDA and CPU controls.

Native consumer-GPU FP64 may be slower than CPU. Correct explicit CUDA F64 can
still be documented as supported, but it must not enter any automatic route
unless paired plugin median throughput is at least `1.03x` CPU F64.

### CUDA-6: optional mixed-precision IR

This stage is optional and begins only after CUDA-0 through CUDA-5 establish a
correct native FP64 fallback on the same device. Follow
[mixed-precision-ir-handover.md](mixed-precision-ir-handover.md).

1. Query `cudaDevAttrSingleToDoublePrecisionPerfRatio` as a profiling hint, not
   an admission decision.
2. Upload original Double transpose and unfactored normal bands for Double
   residual/update kernels; reuse the existing F32 factors only for correction.
3. Retain full residual histories, iteration counts, and forced fallback tests.
4. On IR failure, execute native CUDA FP64. Explicit CUDA never falls back to
   CPU.
5. Retain IR only if it passes the same numerical gates and materially improves
   plugin E2E on the measured target. Native FP64 remains available regardless.

## Numerical Matrix

At minimum cover:

| Dimension | Required cases |
|---|---|
| Precision selection | conditioned automatic, safe automatic, forced F32, forced F64 |
| Axis operation | rows, columns, 2D horizontal-only, vertical-only, and two-axis |
| Mixed axes | risky horizontal/safe vertical and safe horizontal/risky vertical |
| Bandwidth | 1, 3, 5, 7, generic custom support |
| Input/output | float, U8 limited, U10 limited/chroma, U16 full |
| Layout | contiguous, padded stride, non-block tail, small vector count |
| Cache | eager prepare, deferred prepare, sealed lookup, eviction, concurrent request |

Use scalar CPU F64 as the per-pixel implementation oracle and retain the direct
QR/high-precision conditioned anchors from the shared fixture. Float results
outside one output ULP or integer results outside one code fail admission.

## Explicit Backend Semantics

- `backend=cuda` on a capable build executes CUDA or returns an error.
- A device without usable native Double support returns a stable, specific
  error before allocating large workspaces.
- There is no CPU fallback inside `CudaExecutor`.
- `backend=auto` remains unchanged by this branch.
- Route and error evidence must come from the plugin, not only the standalone
  executor.

## Stop Conditions

Stop and hand back the branch without shared integration if:

- retained Double fields are not used end to end;
- the 2D intermediate is Float32;
- an F64 graph or plan cache can alias F32 state;
- integer output differs from scalar CPU F64 by more than one code;
- a target artifact lacks the expected F64 code;
- a supported explicit request can silently use CPU; or
- target-device correctness cannot be reproduced from an exact binary.

## Definition of Done

CUDA F64 is done only when backend-local tests, shared engine tests, API4 plugin
tests, fatbin inspection, and real target-hardware runs all pass. A local build
without a CUDA device, PTX compilation alone, or standalone kernel timing does
not complete this handover.
