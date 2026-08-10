# Metal High-Precision Execution Handover

## Goal and Hard Constraint

Determine whether either of two emulated high-precision Metal strategies can
meet the shared result contract. Performance is a later routing question, not
part of numerical correctness:

1. float-float residual and update with the existing F32 factor solve as the
   correction preconditioner; or
2. a complete float-float RHS and triangular solve.

Metal Shading Language does not support the `double` scalar type. Neither
strategy is called native Metal FP64. Retaining CPU F64 fallback is a valid and
expected final result when accuracy does not pass or no production GPU route is
selected.

See the official
[Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
and the shared
[mixed-precision IR handover](mixed-precision-ir-handover.md).

## Current State

- Metal shaders execute F32 plans only. Fixed B1/B3/B5/B7 and generic routes
  now use the same explicit ordered-FMA recurrence.
- The scheduler excludes a complete frame job from GPU eligibility when any
  processed plane axis requires F64.
- `backend=metal` is a heterogeneous plugin scheduler; `backend=auto` has a
  separate admission policy.
- Current conditioned Metal tests prove CPU fallback, not an independent Metal
  high-precision result.
- A direct Apple ARM64 numerical contract test covers horizontal and vertical
  B1/B3/B5/B7 plus generic B9 at `0 ULP` against ordered F32 and rejects retained
  F64 plans before shader execution.
- Existing Metal compilation uses `-fno-fast-math -ffp-contract=off`; explicit
  `fma` is still required inside error-free transforms and the unified F32 path.

## Ownership

The Metal implementer owns:

- new prototype files under `src/metal/`;
- Metal-specific Apple host files;
- new Metal-only probes, tests, and benchmarks named with `metal_f64` or
  `metal_ir`; and
- a proposed entrypoint/CMake inventory supplied to the integrator.

The implementer does not edit shared planner/public headers, top-level CMake,
plugin routing, existing shared fixtures, other backends, README, or release
reports. Phase 0 is callable only from backend-local probes.

## Shared Prerequisites

The shared prework candidate now contains:

- the unified strict F32 oracle and fixtures;
- retained `normal_bands_f64` and `normal_inf_norm`;
- conditioned and forced-F64 shared fixtures;
- backend-local CMake/test registration hooks; and
- immutable CPU direct-F64 and high-precision reference outputs; and
- `dsmvc_add_numerical_contract_test()` plus the machine-readable numerical
  contract benchmark.

Metal work begins only after that exact candidate is frozen as the common
integration commit.

## F32 Entry Gate: Complete on Apple ARM64

Metal F32 now follows the shared ordered-FMA contract for RHS, lower, diagonal,
and upper recurrence in both fixed-wide and generic routes. Fresh Apple ARM64
execution passes the immutable direct-Metal fixtures at `0 ULP`; the broader
Metal/plugin suite and artifact inventory also pass. This change is admitted as
a correctness repair without a throughput threshold and does not change plugin
routing. METAL-0A through METAL-0E remain unimplemented optional high-precision
work.

## Float-Float Representation

Represent one value as normalized `(hi, lo)` floats packed from the original
Double:

```text
hi = float(value)
lo = float(value - double(hi))
```

Use fixed-order, round-to-nearest transforms:

- `two_sum` and `quick_two_sum`;
- `two_prod` with explicit `fma(a, b, -product)`;
- renormalized add, subtract, and multiply; and
- residual-corrected reciprocal/division for the direct-solve candidate.

Float-float retains F32 exponent range and usually carries fewer effective bits
than IEEE Double. Compiler reassociation, a lost low component, or dependence on
flushed subnormals invalidates the strategy.

## Candidate A: Mixed-Precision IR

### Data

Pack into Metal buffers:

- F32 transpose and factor arrays for the initial/correction solve;
- `transpose_weights_f64` split into `(hi, lo)`;
- original `normal_bands_f64` split into `(hi, lo)`;
- `normal_inf_norm`; and
- per-vector convergence/status storage.

Do not upload widened F32 normal bands. The CPU retains direct F64 factors for
fallback. The risky IR axis does not need a high-precision triangular solve;
full mixed-axis support still needs the limited direct float-float safe-companion
solve described below.

### Execution

For each independent vector:

1. run the existing ordered F32 inverse to obtain `x0`;
2. promote `x0` to a float-float working solution;
3. form float-float `b=A^T y` and `r=b-Gx`;
4. convert the residual to F32 and run a solve-only F32 correction kernel using
   the existing factors;
5. add the correction to the float-float working solution; and
6. repeat with the shared convergence/fallback policy for at most eight
   corrections.

The correction kernel consumes a destination-sized RHS and does not repeat the
transpose stage.

Evaluate two scheduling shapes:

- one thread owns one vector and performs the complete iteration sequence; and
- coordinate-parallel residual kernels plus vector-parallel correction kernels
  in one command buffer.

Do not perform a CPU readback between corrections. Fixed bounded iterations,
device-side status, and one final completion/readback are the baseline. Record
actual iterations per vector.

For 2D, the horizontal working solution and full intermediate stay `(hi, lo)`.
The vertical stage consumes both components. In a safe/risky pair, Candidate A
refines the risky axis and uses a direct float-float solve for the safe companion
with its frozen F32 coefficients widened to `(value, 0)`. This limited direct
solver is required to match current CPU mixed-axis semantics; running the safe
axis in F32 is not admitted.

## Candidate B: Complete Float-Float Direct Solve

Pack `transpose_weights_f64`, `ldlt_bands_f64`, and any admitted reciprocal into
`(hi, lo)` pairs. Keep the pair through:

- RHS accumulation;
- forward substitution;
- diagonal application/division;
- backward substitution; and
- both axes and the complete 2D intermediate.

Parallelize independent rows, columns, planes, or frames. Do not parallelize the
recurrence inside one axis. Start with a generic bandwidth implementation;
specialize 1, 3, 5, and 7 only after generic conformance.

This candidate is the emulated-direct comparator. It may be more predictable
than IR but has expensive serial high-precision recurrence operations.

## Phase-0 Stages

### METAL-0A: arithmetic micro-oracle

1. Add isolated float-float arithmetic kernels used by both candidates.
2. Test cancellation, exponent gaps, conditioned pivots, signed zero, normal
   boundaries, and subnormal-adjacent values.
3. Compare each primitive with CPU Double and inspect generated AIR/metallib.
4. Stop if the scalar oracle's separately rounded operation order cannot be
   preserved or the low component is lost.

### METAL-0B: plan packing and status shell

1. Pack retained Double transpose, normal bands, factors, and norms into
   candidate-specific buffers.
2. Validate checked offsets, byte accounting, cache identity, and finite values.
3. Add a device-side residual/status record and forced failure injection.
4. Keep all prototypes outside plugin routing.

### METAL-0C: one-axis IR

1. Implement initial F32 solve, high-precision residual, solve-only F32
   correction, high-precision update, and bounded convergence.
2. Run Lanczos2 `978.1`, bilinear `974.3`, threshold-adjacent controls, and more
   extreme rcond cases.
3. Retain complete residual/iteration histories and forced fallback evidence.

### METAL-0D: one-axis direct float-float

1. Implement generic RHS and LDLT solve with the same packed Double plan.
2. Run the identical fixture/input matrix.
3. Compare both candidates with CPU direct F64, QR/high-precision anchors, and
   existing Metal F32.

### METAL-0E: optional production-route feasibility

After a candidate passes all numerical gates, measure IR, direct float-float,
and CPU F64 from the same source/plugin binary provenance and thermal window.
Include plan packing, scheduler-shaped batching, command buffers,
synchronization, status handling, and fallback cost. This evidence decides
whether to integrate or automatically route the Metal candidate; it does not
invalidate the correct CPU F64 fallback or the candidate's numerical result.

## Full Admission After Phase 0

Only a passing candidate proceeds to:

1. rows, columns, risky/risky 2D, and both mixed-axis directions;
2. F32, U8, U10, and U16 with integer output within one code of the
   same-precision CPU scalar reference;
3. bounded plan/workspace caching and concurrent scheduler clients;
4. tails, strides, generic bandwidth, command failure, and eviction/re-upload;
5. plugin integration by the shared integrator; and
6. exact AIR/metallib/source/plugin artifact evidence on Apple ARM64 hardware.

The losing candidate remains an isolated benchmark/reference or is removed. Do
not carry two unadmitted production strategies into shared routing.

## Numerical Gates

- Final F32 output is identical to CPU direct F64 when practical, otherwise at
  most one output ULP with no independent-oracle regression.
- Retain the existing `3e-6` absolute guard near zero.
- Integer output is within one code of the same-precision CPU scalar reference;
  repeated execution on one concrete Metal route remains bit exact.
- IR converges within eight corrections for every admitted fixture and records
  all residual histories.
- Direct float-float keeps the low component through the final axis.
- Any NaN, infinity, unexplained residual growth, or compiler reassociation
  fails admission.

## Plugin and Route Semantics

The integrator, not the Metal implementer, changes routing after full admission:

- `backend=metal` remains a mixed CPU/Metal scheduler;
- `_DSMVCMetal=1` means the final frame includes actual Metal-assigned work;
- an IR result discarded and entirely recomputed on CPU is reported as CPU;
- `backend=auto` remains unchanged until a separate paired admission passes;
- iteration/fallback diagnostics are observable in test builds or route audit;
  and
- no result is described as native Metal FP64.

Plane-level safe-GPU/risky-CPU execution is a separate scheduler experiment. It
must not be bundled with either numerical candidate.

## Stop Conditions

Stop Phase 0 and retain CPU fallback when:

- error-free transforms cannot be reproduced on the exact Metal artifact;
- a candidate starts from widened F32 high-precision operator data;
- IR rounds its working solution/intermediate to F32 or violates the shared
  convergence policy;
- direct solve loses the low component during RHS, recurrence, or intermediate;
- float or integer numerical gates fail;
- memory grows without a strict concurrent cap;
- passing would require changing geometry, padding, regularization, or output
  tolerances.

## Definition of Done

Metal high-precision correctness is complete in one of two states:

1. an emulated strategy passes all numerical, fallback, artifact, and hardware
   gates; it may remain an isolated backend result until a separate production
   route decision; or
2. Phase 0 records reproducible failure evidence and CPU F64 remains supported.

Automatic Metal routing is a separate completion state requiring plugin-shaped
performance evidence and truthful route reporting. It is not required to close
the numerical-correctness work.
