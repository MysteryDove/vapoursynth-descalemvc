# Mixed-Precision Iterative-Refinement Handover

## Goal

Evaluate whether the existing high-throughput F32 factor solve can act as a
preconditioner while a higher-precision residual and update recover the current
high-precision result contract.

IR is an alternative execution strategy, not a change to `f64mode`, padding,
geometry, or the least-squares operator. It is admitted only when it matches the
CPU direct-F64 result/oracle gates. Whether a correct IR implementation replaces
or supplements a direct-F64 route is decided separately from correctness.

## Implementation Status

The shared prework candidate retains the operator data needed for an independent
residual and provides ordered F32/F64 oracles, immutable fixtures, hash layers,
and a numerical axis benchmark. The IR loop, correction-only solve, convergence
telemetry, and backend fallbacks are not implemented.

IR is not an independent parallel integration lane. Metal F32 convergence is
complete on Apple ARM64, but Metal IR and direct float-float remain unimplemented.
CUDA and Vulkan may evaluate IR only after their native F64 baselines pass. The
first CPU optimization wave keeps direct Double and does not implement IR.

## Mathematical Contract

For one axis:

```text
G x = b
G = A^T A
b = A^T y
```

Let `S32` be the actual ordered F32 correction solve using the packed F32
factors. Classical conditioning is only a prefilter. Runtime convergence is
governed by the actual iteration operator:

```text
E = I - S32 G
rho(E) < 1
```

The two release anchors have `kappa(G) * epsilon_F32` near `0.0302` and
`0.0526`. They are promising candidates, not a convergence proof. The report
uses `epsilon_F32=2^-23`; conventional unit roundoff is `u=2^-24`.

## Shared Plan Data

The integrator-owned shared candidate provides:

- original unfactored `normal_bands_f64`;
- `normal_inf_norm`;
- validation and storage/cache accounting for both fields; and
- immutable conditioned, forced-precision, safe-control, padding, tail,
  mixed-axis, cache, and malformed-metadata fixtures.

Deliberately non-convergent/adversarial residual-history fixtures are still part
of the backend IR implementation stage because no convergence loop exists yet.

The existing F32 transpose/factor arrays remain the correction preconditioner.
The existing F64 transpose/factors remain available for the CPU or native-F64
fallback. An IR backend never reconstructs `G` from F32 factors.

## Required Algorithm

```text
x_hp = promote(existing_f32_inverse(y))

for correction = 0 .. 7:
    b_hp = high_precision(A^T y)
    r_hp = b_hp - high_precision(G x_hp)
    eta = ||r_hp||inf / (||G||inf ||x_hp||inf + ||b_hp||inf)

    if residual, correction, and final-F32 rounding are stable:
        return x_hp

    d32 = f32_factor_solve(round_to_f32(r_hp))
    x_hp = high_precision(x_hp + d32)

fallback
```

Production implementations may cache `b_hp` or recompute it. They may use a
banded `G*x` or the original `A^T(y-Ax)` form, but the selected form and order
must be frozen and compared against the same CPU high-precision reference. The
first implementation should use retained Double normal bands because they
match the current normal-equation operator without requiring a forward CSR or a
source-sized temporary.

The correction solver consumes a destination-sized residual directly. It must
not rerun the transpose/RHS stage as though the residual were source samples.

## High-Precision Representations

### Native Double

CUDA, qualifying Vulkan devices, and CPU may hold `b`, `r`, `x`, and updates in
Double while using F32 factors only for the correction solve. This is an
optional performance experiment after their native direct-F64 baselines pass.

### Float-float

Metal represents one value as normalized `(hi, lo)` floats packed from an
original Double:

```text
hi = float(value)
lo = float(value - double(hi))
```

Required primitives include `two_sum`, `quick_two_sum`, FMA-based `two_prod`,
renormalized add/subtract/multiply, and accurate accumulation. Fast math and
reassociation are disabled. Float-float keeps the F32 exponent range and is not
called native FP64.

For 2D, `x_hp` and the entire axis intermediate remain Double or float-float.
If only one axis is risky, refine that axis and execute the safe companion with
direct high-precision arithmetic over its frozen F32 coefficients, matching the
current CPU semantics. Do not run the companion axis in F32 and do not retain
Double operator data for every safe plan solely to force it through IR. A final
F32 conversion occurs only after the last axis.

## Convergence and Fallback Policy

An admitted route checks all of:

- finite `b`, residual, correction, and working solution;
- scaled backward error against a fixture-derived target;
- meaningful residual reduction;
- correction small enough that final F32 rounding is stable; and
- no independent QR/high-precision-oracle regression.

Fallback is mandatory when:

- residual grows on two corrections;
- residual ratio remains above `0.5` for two corrections;
- a nonzero high-precision residual rounds to an unusable F32 correction;
- the update stagnates above the target error floor;
- any value becomes nonfinite; or
- eight corrections complete without success.

The `0.5` ratio is a failure diagnostic, not the final convergence criterion.
Every solve records iteration count, initial/final backward error, residual
history, convergence reason, and fallback reason.

## Backend Failure Semantics

- CPU: direct F64 is always available; first-wave CPU optimization does not use
  IR.
- CUDA: optional IR falls back to native CUDA FP64 on the same backend. If
  native FP64 is unavailable or not admitted, explicit CUDA returns an error.
- Vulkan: optional IR falls back to native Vulkan Float64 on the same backend.
  Missing capabilities produce an explicit error.
- Metal: the heterogeneous scheduler may discard a failed IR result and rerun
  the affected final work through CPU F64. A CPU-produced final frame is not
  labeled as an independent Metal high-precision result.
- `auto`: remains unchanged until the candidate passes paired plugin admission.

## Phase-0 Matrix

At minimum cover:

| Dimension | Cases |
|---|---|
| Conditioning | Lanczos2 `978.1`, bilinear `974.3`, threshold-adjacent safe/risky plans, and more extreme rcond |
| Precision mode | automatic risky, forced F64 safe/risky, forced F32 control |
| Operation | rows, columns, horizontal-only 2D, vertical-only 2D, risky/risky and both mixed-axis directions |
| Bandwidth | 1, 3, 5, 7, and one generic custom support |
| Samples | F32 first; then U8, U10, and U16 before full admission |
| Failure | residual growth, stagnation, nonfinite, iteration cap, backend capability failure |
| Scheduling | one vector, tails, full frame, realistic batches, concurrent plugin callbacks |

Report the full iteration histogram, not only mean iterations. Separate:

- residual and correction kernel timing;
- standalone executor timing;
- plugin E2E including packing, command submission, synchronization, and any
  fallback; and
- unchanged F32 controls.

## Optional Routing and Performance Decision

- Correctness admission has no throughput threshold. F64 and emulated
  high-precision execution are expected to be slower than F32.
- CUDA/Vulkan/Metal routing may consider whether IR materially extends the
  usefulness of the direct high-precision route on the same device. A device
  capability/performance ratio is only a routing hint, not benchmark evidence.
- Metal Phase 0 compares IR, complete float-float direct solve, and CPU F64 from
  the same source/binary provenance and thermal window.
- A fast one-correction microcase cannot admit a route whose conditioned p95
  needs enough corrections to lose plugin E2E.

## Definition of Done

IR numerical correctness is complete only when shared fixtures, independent
references, forced fallbacks, the one-code integer contract, artifact
inspection, target hardware, and complete residual histories all pass. A
convergent standalone kernel or a float-float arithmetic microtest alone is not
completion. Production or automatic routing additionally requires separately
authorized plugin E2E evaluation.
