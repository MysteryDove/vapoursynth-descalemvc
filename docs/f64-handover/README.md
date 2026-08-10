# Float32 Stability and Float64 Backend Handovers

## Authority and Readiness

The integration authority is
[f32-cross-backend-contract-prework.md](../../.omx/plans/f32-cross-backend-contract-prework.md).
This directory splits its high-precision work into backend-owned handovers from
a shared prework candidate based on source revision `e411298`.

The integrator-owned Stage 0/1 implementation is complete and locally
validated: immutable conformance fixtures, ordered F32/F64 oracles, retained raw
Double normal bands, benchmark hooks, and backend-local registration points are
present. The repository is ready to freeze this candidate and launch parallel
backend work. Backend branches must begin from the same resulting commit, not
from the still-dirty working tree.

This is planning material. It is not evidence that CUDA, Vulkan, or Metal
high-precision execution is implemented.

## Validated Starting Point

The following Release checks were recorded on 2026-08-09:

- Apple ARM64 with Metal: 9 of 9 tests passed, including the numerical contract,
  direct Metal F32 numerical contract, conditioned CPU F64 fallback, and the
  32-entrypoint Metal artifact inventory.
- Apple ARM64 CPU-only: 3 of 3 tests passed.
- macOS x86_64 CPU-only: 2 of 2 tests passed; conditioned rows, columns, and 2D
  fixtures matched their scalar output fixtures. The numerical benchmark
  reports `native_cpu_path=scalar`, so this is not AVX2 hardware evidence.
- CUDA and Vulkan target-device F64 have not been built or run.

The local `build-release-f64-padding*` and `build-prework-*` directories are
untracked evidence. They are not release artifacts and must not be committed or
removed by backend work.

The shared candidate also establishes:

- exact `f32_contract_identity` values for request/padding/precision metadata
  and every logical F32 planner array;
- a separate `complete_identity` containing Double bits, used only to bind a
  concrete host/libm plan rather than require cross-libm bit identity;
- ordered F32/F64 result identities, four padding modes, B1/B3/B5/B7 tails,
  forced-F32/forced-F64/automatic-risk fixtures, mixed-axis 2D, cache and
  malformed-metadata tests;
- an independent QR gate for conditioned `978.1` (`4.57457e-9` ARM64 and
  `4.5922e-9` x86_64 endpoint error versus a `2.69779e-8` bound); and
- a machine-readable axis benchmark for ordered/current/native CPU F32,
  automatic-risk F64, and forced F64.

The Apple ARM64 correctness lane additionally establishes:

- NEON F32 rows, columns, scalar fallbacks, and SIMD tails match the ordered-FMA
  contract at `0 ULP` for B1/B3/B5/B7 fixtures;
- NEON F64 rows, columns, safe/risky and risky/safe 2D, and forced-F64/F64 2D
  match the ordered Double contract at `0 ULP`; and
- direct Metal F32 execution matches the ordered-FMA contract at `0 ULP` for
  horizontal and vertical B1/B3/B5/B7 plus generic B9. A retained F64 plan is
  rejected by the direct Metal executor and remains observable CPU F64 work in
  the plugin scheduler.

The axis benchmark is not the CPU lane's complete rows/columns/2D/integer/plugin
performance harness. Its timings cannot be used as a release speed claim.

## Result Precision Versus Implementation Strategy

`f64mode` defines the required result, not a specific algorithm:

| Mode | Required result |
|---|---|
| `f64mode=0` | F32 for safe plans; high precision when `normal_rcond < 1e-4` |
| `f64mode=1` | forced F32; never direct F64 or IR |
| `f64mode=2` | high precision for every processed axis |

`backend=auto` keeps CPU direct F64 as its reliable high-precision route until
an exact GPU strategy passes automatic admission. Any later auto fallback to CPU
remains observable. This does not permit explicit CUDA/Vulkan CPU fallback.

An implementation may satisfy the high-precision requirement through:

1. a direct native/emulated high-precision RHS and solve; or
2. admitted mixed-precision iterative refinement with an independent
   high-precision residual and update.

Running the old F32 solve without convergence evidence never satisfies the
high-precision requirement.

## Backend Strategy Matrix

| Lane | Primary strategy | Optional strategy | Current behavior |
|---|---|---|---|
| CPU | direct Double; optimize AVX2 F64 and ARM integer NEON | no first-wave IR | complete CPU result/fallback oracle |
| CUDA | native FP64 direct solve | IR only after native correctness and timing | retained F64 rejected before CUDA |
| Vulkan | native Float64 with queried capabilities | IR only after native correctness and timing | retained F64 rejected before Vulkan |
| Metal | compare float-float residual/F32-correction IR with complete float-float direct solve | plane-level safe-GPU/risky-CPU split | retained F64 executes in CPU scheduler lane |

CUDA/Vulkan IR, if later admitted, falls back to native high precision on the
same named backend or returns an error. It never silently executes CPU. Metal is
a heterogeneous plugin scheduler and may rerun a failed high-precision GPU
attempt on CPU, but the final route must remain observable.

## Frozen Common Result Contract

1. `AxisPlan::normal_rcond` remains available in every precision mode.
2. A retained direct plan consumes original `transpose_weights_f64` and
   `ldlt_bands_f64`; risky-axis direct execution never widens F32 factors.
3. Shared Stage 1 retains original unfactored
   `normal_bands_f64` and `normal_inf_norm` for independent IR residuals.
4. Direct high precision covers RHS accumulation, forward substitution,
   diagonal operation, backward substitution, and the complete two-axis
   intermediate.
5. IR may reuse F32 factors only for the correction solve. Residual evaluation
   and `x += correction` stay at admitted high precision, use original Double
   operator data, and pass convergence/fallback gates.
6. If either 2D axis requires high precision, both axes and their intermediate
   remain high precision. A safe companion uses its frozen F32 coefficients
   widened to the selected high-precision representation.
7. Float output converts once after the final solve. Integer normalization,
   scale, clamp, and rounding preserve the existing semantics. U8/U10/U16
   output is within one code of the same-precision CPU scalar reference.
8. Padding, geometry, coefficient order, band order, and custom-kernel behavior
   do not change. Backends consume planner output and never regenerate it.

## Shared Integrator Lane

Only the integrator edits:

- `include/dsmvc/engine.hpp`, `src/axis_plan.cpp`, and
  `src/axis_plan_internal.hpp`: raw normal bands, norm metadata, validation,
  storage accounting, and shared oracles;
- `src/executor.cpp`, `src/backend.cpp`, `src/vs_plugin.cpp`, and `dsmvc.py`:
  public dispatch, explicit error/fallback behavior, and route properties;
- top-level `CMakeLists.txt`: serialized backend registration and artifact gates;
- shared tests and fixtures under `tests/`;
- README, API, handover, and release documentation.

The following items are implemented in the shared candidate and must be frozen
in one integration commit before backend work:

1. immutable F32/F64 plan and output fixtures plus backend-independent
   comparison utilities;
2. ordered F32 and F64 arithmetic oracles plus F32/F64 benchmark baselines;
3. behavior-preserving raw Double normal-band/norm retention and cache
   accounting;
4. backend-local CMake/test hooks without routing changes.

The shared CUDA/Vulkan F64 rejection remains until each backend validates its
own capability and precision behavior. Metal plugin routing remains unchanged
during all Phase-0 prototypes.

## Ownership Matrix

| Lane | May edit | Must not edit |
|---|---|---|
| CPU | `src/cpu_executor.cpp`, one SIMD source per sub-lane, CPU-only tests/benchmarks | planner semantics, GPU files, public API |
| CUDA | `src/cuda/**`, CUDA-only tests/benchmarks/artifact scripts | shared files and every Vulkan/Metal file |
| Vulkan | `src/vulkan/**`, Vulkan-only tests/benchmarks/artifact scripts | shared files and every CUDA/Metal file |
| Metal | `src/metal/**`, Metal Apple host files, Metal-only probes/tests/benchmarks | shared plugin routing and other backends |
| Integrator | shared files listed above | backend internals except reviewed integration hunks |
| Verifier | immutable runner and result artifacts | production numerical implementation |

Backend branches use the same frozen integration commit. They do not resolve
conflicts by taking complete shared files from another branch.

## Parallel Launch Set

Launch four lanes after the shared commit is frozen:

1. **CPU:** converge scalar/AVX2/NEON F32 and tails first, extend the F64
   rows/columns/2D/integer benchmark, then optimize direct F64.
2. **CUDA:** converge ordered F32 and inspect the exact CUDA artifact first,
   then implement native FP64. Evaluate IR only after the native baseline.
3. **Vulkan:** converge F32 under queried float controls first, then implement
   capability-gated native Float64. Evaluate IR only after the native baseline.
4. **Metal:** converge fixed and generic F32 routes first, then run the isolated
   float-float IR versus direct float-float Phase 0.

Do not launch a separate CPU-F64 agent alongside the CPU-F32 lane because both
own `cpu_executor.cpp` and the same SIMD files. Do not launch a standalone IR
integration lane; IR remains subordinate to Metal Phase 0 or to a validated
native CUDA/Vulkan lane.

## Common Numerical Gates

Every backend follows this order:

1. **Plan gate:** prove plan hashes, precision metadata, cache identity, and
   original high-precision fields are consumed correctly.
2. **F32 control gate:** prove the backend's F32 path matches the ordered strict
   fixtures, including vector-width boundaries and scalar tails.
3. **High-precision gate:** compare rows, columns, mixed-axis 2D, float, U8,
   U10, and U16 against CPU direct F64 and independent conditioned anchors.
4. **IR gate where applicable:** retain complete residual history, scaled
   backward error, iteration count, and forced fallback results. Every admitted
   fixture converges in no more than eight corrections.
5. **Engine/plugin gate:** prove buffered/streamed behavior, explicit backend
   semantics, route properties, API4 frames, and all failure paths.
6. **Artifact gate:** inspect the exact object, fatbin, SPIR-V, AIR, or metallib
   for expected precision operations and bind it to source/binary hashes.
7. **Hardware gate:** run on a real supported target; compile-only and software
   emulation cannot close the gate.
8. **Optional route gate:** after correctness is complete, report executor and
   plugin E2E separately before changing automatic routing. This gate chooses
   where a correct result executes; it does not reject a correct CPU or backend
   F64 implementation because high precision is slower.

Native high precision should normally produce identical final F32 output; any
difference is limited to one output ULP and cannot worsen the independent
high-precision reference. Integer output is within one code of the
same-precision CPU scalar reference. Repeated execution on one concrete route,
and CPU buffered/streamed or fused/two-pass route pairs, remain bit exact. NaN
or infinity fails with `std::runtime_error`.

## Merge Order

1. Freeze the completed shared conformance, benchmark, arithmetic-oracle, and
   raw-normal-band candidate as one common branch point.
2. Merge F32 arithmetic convergence one backend at a time, rerunning shared
   controls after every merge.
3. Optimize CPU direct F64 and keep it as the oracle.
4. Implement CUDA and Vulkan native F64 independently; remove each shared
   rejection only after its target-hardware gate.
5. Run Metal IR and full float-float Phase 0 as backend-local A/B candidates.
6. Evaluate optional CUDA/Vulkan IR only after native FP64 evidence exists.
7. Integrate a winning high-precision route and then consider plane-level Metal
   split or automatic routing as separate measured changes.
8. Update release reports only from fresh exact-binary evidence. Historical old
   descale and existing CPU throughput data are not rerun or rewritten without
   explicit authorization.

## Backend Documents

- [Mixed-precision iterative-refinement handover](mixed-precision-ir-handover.md)
- [CUDA F64 execution handover](cuda-f64-execution-handover.md)
- [Vulkan F64 execution handover](vulkan-f64-execution-handover.md)
- [Metal high-precision execution handover](metal-f64-execution-handover.md)
- [CPU F64 optimization research](cpu-f64-optimization-research.md)

## Global Stop Conditions

Stop a backend branch without changing shared routing when:

- a backend regenerates planner data or computes residuals from widened F32
  operator coefficients;
- direct risky execution consumes F32 factors, or IR rounds its working solution
  or two-axis intermediate to F32;
- explicit CUDA/Vulkan can silently execute CPU;
- high-precision float output exceeds one ULP or worsens an independent anchor;
- integer output differs from the same-precision CPU scalar reference by more
  than one code;
- target features or exact artifact evidence are missing;
- an IR residual grows twice, stagnates, becomes nonfinite, or fails its
  eight-correction cap; or
- correctness requires changing padding, geometry, regularization, or the
  least-squares operator.

Performance regression is not a global correctness stop condition. It may stop
an optional optimization or automatic-route proposal, but the correct direct
F64 path remains admissible even when it is materially slower than F32.
