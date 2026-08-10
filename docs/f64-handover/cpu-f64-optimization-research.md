# CPU Float64 Optimization Research

## Conclusion

The complete CPU Float64 path is implemented and suitable as the backend
oracle, but it is not performance-complete. The x86 AVX2 four-RHS Double path
is now implemented. Remaining opportunities include ARM integer-input NEON
F64 and broader executor/plugin performance coverage.

A lightweight benchmark now records ordered-reference, current-scalar, and
native-CPU axis solves for automatic-risk and forced F64 plans. It is diagnostic
baseline evidence, not a complete optimization measurement. Every optimization
below remains a hypothesis until it passes the proposed paired executor and
plugin benchmarks plus the numerical oracle.

Mixed-precision iterative refinement is intentionally excluded from the first
CPU optimization wave. The CPU already has complete native Double execution;
IR would add a high-precision residual, one or more F32 correction solves, and
convergence bookkeeping. It may be revisited only if optimized direct F64 has a
measured architecture-specific deficit. See
[mixed-precision-ir-handover.md](mixed-precision-ir-handover.md).

The shared ordered F64 oracle and conditioned QR anchor now exist and pass on
ARM64 and x86_64. The scalar CPU path defines the cross-backend arithmetic
contract: each multiply and add/subtract is rounded separately, and retained
F64 diagonal application divides by `D + epsilon`. AVX2 and NEON vector bodies
and scalar-width tails preserve that order.

The CPU lane first converges scalar/AVX2/NEON F32 behavior and tails, then
extends the F64 benchmark and optimizes direct F64. These are serial phases of
one lane, not separate agents editing the same executor and SIMD files.

## Established Implementation

The current source directly establishes that:

- [axis_plan.cpp](../../src/axis_plan.cpp) retains original Double transpose
  weights, unfactored normal bands, normal infinity norm, raw LDLT bands,
  inverse diagonal, and `normal_rcond` below the automatic `1e-4` threshold or
  under forced F64.
- [cpu_executor.cpp](../../src/cpu_executor.cpp) executes rows and columns in
  Double and uses a complete Double intermediate when either 2D axis is risky.
- Float output converts only after the solve; integer input normalization and
  final scale/clamp/`nearbyint` conversion occur in Double.
- [cpu_executor_neon.cpp](../../src/cpu_executor_neon.cpp) batches four
  independent F64 RHS using two `float64x2_t` vectors.
- ARM float-input rows, columns, and 2D use that NEON path, including Double
  intermediate producers and consumers.
- Apple ARM64 contract tests now cover NEON F32 rows, columns, vector-width
  boundaries, and scalar tails at `0 ULP` for B1/B3/B5/B7. NEON F64 rows,
  columns, both safe/risky axis orders, and forced-F64/F64 2D also match the
  ordered Double result at `0 ULP`.
- [cpu_executor_avx2.cpp](../../src/cpu_executor_avx2.cpp) batches four
  independent F64 RHS in `__m256d` lanes for rows, columns, and float 2D.
- Both buffered and streamed integer F64 calls invoke `inverse_2d_f64` with
  `use_neon=false`, so ARM integer F64 is scalar.
- Internal CPU parallelism is capped at four workers and the F64 path shares
  the F32 `262144` work threshold.
- Scalar F64 rows/columns allocate source and destination vectors per worker
  range, and each 2D call allocates a full Double intermediate.
- `dsmvc_numerical_contract_benchmark` now supplies one-axis F32,
  automatic-risk F64, and forced-F64 ordered/current/native baselines. The
  existing `dsmvc_cpu_profile_benchmark` remains F32-only. Neither benchmark yet
  establishes complete F64 rows/columns/2D/integer or plugin speedups.

## Ranked Opportunities

| Rank | Opportunity | Expected value | Confidence | Main risk |
|---:|---|---|---|---|
| 1 | Tune the implemented x86 AVX2 four-RHS Double rows/columns/2D path | High | High | arithmetic-order changes and tail handling |
| 2 | Add ARM NEON integer F64 normalization and output | High for U8/U16 risky plans | High | exact conversion/rounding parity |
| 3 | Extend the F64 benchmark to rows/columns/2D/integer/plugin coverage | Enables all retention decisions | High | fixture bias if only one conditioned geometry is used |
| 4 | Specialize F64 bandwidths 1, 3, 5, and 7 | Medium-high | Medium | code size and register pressure |
| 5 | Reuse bounded scalar/2D scratch storage | Medium | Medium | concurrency, lifetime, and memory retention |
| 6 | Tune F64 worker count and work thresholds | Medium | Medium | nested VS concurrency can reverse local gains |
| 7 | Evaluate wider ARM RHS batches/coefficient reuse | Medium-low | Medium-low | spills and larger workspace traffic |
| 8 | Tile or fuse the 2D F64 pipeline | Experimental | Low | accumulation order and semantic drift |
| 9 | Replace scalar diagonal division with retained reciprocal multiplication | Small to medium | Medium-low | one extra rounding difference from scalar oracle |

## Stage 0 for Optional CPU F64 Optimization

Correctness coverage is broader than the benchmark: ARM rows, columns, mixed
2D, forced-F64/F64 2D, and SIMD tails are now numerical-contract tests. The
performance harness remains partial and covers one-axis ordered, current-scalar,
and native-CPU routes for an F32 control plus automatic-risk and forced F64.
Before retaining an optional optimization, extend that separate benchmark to
the matrix below rather than changing the historical F32 profile format.

This stage is not required to admit a numerical-correctness repair. A slower
direct F64 implementation remains valid when it is the route that satisfies the
precision contract.

### Required cases

- Precision: conditioned `f64mode=0`, safe `f64mode=2`, and `f64mode=1` F32
  control.
- Operations: rows, columns, horizontal-only 2D, vertical-only 2D, and full
  two-axis 2D.
- Axis combinations: risky/risky, risky/safe, and safe/risky.
- Kernels/bands: B1, B3, B5, B7, and one generic custom support.
- Scale: small non-parallel work, threshold-adjacent work, and representative
  full frames.
- Samples: float, U8 limited luma, U10 limited chroma, and U16 full.
- Paths: scalar vs native SIMD on ARM and x86.

### Measurement contract

1. Use a fresh Release build and record source SHA, binary SHA-256, compiler,
   architecture, CPU, OS, and command line.
2. Alternate scalar/native order, warm both paths, and retain all raw samples.
3. Report median, spread, and paired ratio per case. Keep one-axis executor,
   2D executor, and plugin E2E separate.
4. Verify the plan actually retains F64 data before timing.
5. Track allocation count/bytes, peak resident memory, worker utilization,
   cache misses, and hardware counters where available.
6. Require the shared one-code integer contract, finite output, and the shared
   F64 numerical gate before considering speed.
7. Retain a change only when representative paired median improves by at least
   3% and no important case regresses by more than 3%, unless the commit has a
   documented architecture-specific scope.

## 1. x86 AVX2 Double Path

Status: implemented. The design below records the accepted shape of the path;
future changes remain subject to the same arithmetic and numerical gates.

### Evidence

F64 dispatch in `CpuExecutor::inverse_rows`, `inverse_columns`, and float 2D now
uses four independent `__m256d` RHS lanes when AVX2 is explicitly or
automatically selected. Ordered scalar tails cover incomplete groups.

### Proposed implementation

1. Add a four-independent-RHS `__m256d` solver analogous to ARM `F64Quad`.
   SIMD lanes represent rows or columns; the dependency within one axis remains
   scalar in axis index.
2. Add float-to-Double row loads, Double intermediate stores, Double
   intermediate column loads, and one final Double-to-float conversion.
3. Dispatch retained rows/columns through AVX2, then connect both stages of 2D.
4. Keep scalar tails for fewer than four independent RHS.
5. Start with the generic runtime bandwidth loop. Specialize only after the
   baseline is correct and measured.

The x86 source is compiled with AVX2/FMA flags. Freeze the contraction policy
for this path and compare both final output and the independent oracle. Do not
change the existing F32 code generation as a side effect.

### Acceptance

- all conditioned and mixed-axis float fixtures pass;
- final integer output remains within one code of scalar CPU F64 once integer
  support is connected;
- x86 Release plugin smoke passes on Windows/Linux target hardware, not only a
  macOS cross-architecture build; and
- representative F64 executor and plugin medians improve by at least 3%.

## 2. ARM Integer F64 NEON

### Evidence

The templated integer paths call `inverse_2d_f64(..., false)` for both buffered
and streamed APIs. The current NEON helper accepts float input and writes float
or Double intermediate, but has no U8/U16 loader or Double-to-integer final
store.

### Proposed implementation

1. Add four-lane U8/U16 loaders that convert integer samples to Double and then
   apply `(sample - double(input_offset)) * double(input_scale)` in the same
   order as the scalar template.
2. Reuse the F64 quad solver to write the Double horizontal intermediate.
3. Add a vertical Double-input solve whose final store applies Double
   `output_scale`, Double `output_offset`, clamp, and round-to-nearest-even.
4. Use scalar conversion for tails until a vector rounding implementation is
   proven to satisfy the one-code contract for every supported output range.
5. Enable `use_neon` for integer F64 only after U8, U10, and U16 parity passes.

### Acceptance

Buffered and streamed calls must match each other exactly. Each integer sample
must remain within one code of scalar CPU F64, including half-way rounding
cases, clamps, padded strides, odd widths, and fewer than four columns. A
throughput gain cannot compensate for exceeding that bound.

## 3. Bandwidth Specialization

### Evidence

The F64 NEON solver uses runtime `min` and distance loops. Established F32 NEON
and CUDA paths already use dedicated bandwidth 1/3/5/7 dispatch because these
are the common built-in plan shapes.

### Proposed implementation

- Template the F64 core on bandwidth 1, 3, 5, and 7, with a generic fallback.
- Preserve the exact forward descending-distance and backward descending-
  distance order used by the F64 reference. Do not copy B3's F32 ascending
  optimization without a separate numerical gate.
- Hoist coefficient addresses and width once per solve where this reduces
  repeated indexing.
- Inspect generated assembly for unrolling, spills, and instruction-cache
  growth before retaining each specialization.

Measure each specialization independently. A single combined patch would make
it impossible to identify a code-size or register-pressure regression.

## 4. Bounded Scratch Reuse

### Evidence

The scalar path creates `std::vector<double>` source/destination storage inside
each worker-range callback. The 2D path creates a full
`vertical.source_size * horizontal.destination_size` Double vector per call.
The NEON one-axis helper uses thread-local storage, but broad use of per-thread
storage in VapourSynth could retain one large allocation per callback thread.

### Proposed implementation

- Give each fixed internal worker a bounded source/destination scratch slot.
- Add a bounded executor-level lease pool for 2D intermediates, keyed by needed
  capacity and capped by measured concurrent demand.
- If all slots are busy, use a local allocation or bounded wait; never grow an
  unbounded TLS cache across VapourSynth threads.
- Release logical ownership after every call while retaining capacity only
  under an explicit byte cap.
- Include allocation failures and concurrent exceptions in tests.

Measure allocator calls and resident memory as well as time. Reject scratch
reuse that speeds one frame but creates unbounded multi-filter retention.

## 5. F64 Parallelism Tuning

### Evidence

The internal pool is shared, capped at four, non-reentrant through an atomic
in-use flag, and activated at the same `262144` output-work threshold as F32.
F64 has a different arithmetic/memory ratio, while VapourSynth may already run
several frames concurrently.

### Proposed experiment

- Sweep worker counts 1 through the smaller of physical performance cores and
  8.
- Sweep thresholds around 32K, 64K, 128K, 256K, and 512K output elements.
- Test one callback and realistic concurrent callbacks separately.
- Record worker-pool contention/fallback frequency and system CPU time.
- Select by architecture and operation only if the gain is stable; avoid a
  universal lower threshold based on one full-frame case.

## 6. Wider ARM RHS Batches

The current four-RHS design is a sound baseline. An eight-RHS variant could
reuse each coefficient across twice as much work, but doubles live vector state
and workspace traffic. Prototype it only after bandwidth specialization and
profile register spills. Retain it only for row/column counts large enough to
avoid tail duplication and only with a measured gain beyond noise.

## 7. Experimental 2D Restructuring

A full Double intermediate is expensive, but removing it is not a local memory
optimization. Vertical RHS formation needs horizontally solved values, and a
streaming/tiled design can change accumulation order or repeat horizontal
recurrences. Either change is visible on an ill-conditioned problem.

Treat this as a separate algorithm experiment:

1. write down the exact proposed order and storage reduction;
2. compare every pixel with the current Double intermediate path and the
   independent oracle;
3. measure memory traffic and plugin E2E, not just kernel time; and
4. reject it if it changes final float output beyond one ULP or integer output
   beyond one code relative to scalar CPU F64.

It should not be assigned in the first parallel optimization wave.

## 8. Division Versus Reciprocal Multiplication

The scalar retained path divides by `factor + epsilon`; the NEON F64 path
multiplies by the planner-retained reciprocal. Multiplication can be faster but
introduces a different rounding sequence. Before unifying the scalar path on
the reciprocal, compare:

- final float and integer output;
- direct QR/high-precision error on conditioned plans;
- scalar/NEON cross-platform consistency; and
- actual division cost in profiles.

This is accuracy-sensitive and lower priority than SIMD and integer coverage.

## Recommended Commit Sequence

1. `bench(cpu): extend isolated f64 benchmark to complete executor matrix`
2. `perf(cpu-x86): add generic four-rhs avx2 f64 solve`
3. `perf(cpu-arm): vectorize integer f64 normalization and conversion`
4. One commit per architecture for B1/B3/B5/B7 specialization, retaining only
   measured wins.
5. `perf(cpu): add bounded f64 scratch reuse`
6. Architecture-specific threshold tuning with raw result artifacts.
7. Optional wider NEON, 2D restructuring, and reciprocal experiments as
   independent branches.

Do not combine x86, ARM integer, scratch, and scheduling changes in one commit.
Each has a different correctness and performance failure mode.

## Research Boundaries

Established by current code and existing tests:

- the CPU F64 numerical chain is complete;
- ARM float-input F64 SIMD exists;
- x86 F64 SIMD and ARM integer F64 SIMD do not;
- the numerical benchmark has an F64 axis baseline but not the complete
  rows/columns/2D/integer/plugin matrix; and
- local Release correctness tests passed on Apple ARM64 and macOS x86_64.

Not established without new measurements or target systems:

- the speedup of any proposed optimization;
- the best F64 worker count or threshold;
- Windows/Linux AVX2 F64 behavior;
- memory savings under real concurrent VapourSynth workloads; or
- whether a restructured 2D path can preserve the current output order.
