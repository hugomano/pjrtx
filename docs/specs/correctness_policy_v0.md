# Correctness Policy V0

This spec defines the strict mathematical correctness defaults for the new
PjRTx architecture under `//pjrtx/...`.

Performance work is valid only when semantic obligations remain visible and
tested. V0 prefers rejecting programs over silently accepting approximate or
underspecified behavior.

## Requirements

Correctness policy must be represented at every layer:

- StableHLO import and verification
- typed graph payloads
- target legality
- cost and lowering records
- schedule verification
- backend binding
- runtime execution
- profiling and reports

## Default Mode

V0 default mode is strict:

```text
math_mode = strict
fast_math = false
allow_reassociation = false
allow_contracting = false unless StableHLO semantics permit it
allow_approx_transcendentals = false
allow_dtype_narrowing = false
allow_layout_semantic_change = false
allow_implicit_host_execution = false
```

If a future mode relaxes any of these, the compile options, lowering records,
backend binding, and explain report must all record the relaxation.

## Shape And Type Correctness

The graph importer and verifier must reject:

- dynamic dimensions in V0
- unsupported tuple or token forms
- rank forms not covered by V0 typed payloads
- invalid dot contracting dimensions
- incompatible elementwise shapes
- unsupported dtype conversions
- unknown layout semantics

Shape/type failure happens before target legality and before backend binding.

## Floating-Point Semantics

V0 does not assume all targets implement floating-point operations identically.
The compiler must track:

- input dtype
- output dtype
- accumulation dtype when different
- operation class
- whether a backend operation is exact enough for the selected math mode
- any documented approximation

`tanh` is classified as `transcendental`, not generic elementwise work. A target
that cannot provide strict enough `tanh` for V0 must fail compilation for that
region.

## Lowering Correctness

Lowering records must preserve source provenance and semantic obligations.

Required fields:

- graph instruction IDs
- source refs
- selected lowering decision
- rejected alternatives when meaningful
- dtype and accumulation policy
- approximation note, empty when none
- correctness status

Unsupported is a valid compile result. It is not a runtime event.

## Schedule Correctness

Schedule verification must prove:

- every command dependency exists
- data dependencies dominate use
- H2D commands complete before backend use
- D2H commands wait for producing backend commands
- backend commands have lowering provenance
- backend commands have backend bindings
- profile instrumentation does not add hidden synchronization unless recorded

Runtime executes verified schedules. It does not repair invalid schedules.

## Reference Oracle

Reference execution is allowed only in tests and correctness checking. It may be
JAX CPU, NumPy, or a dedicated reference interpreter.

Rules:

- reference execution never produces PJRT outputs in production
- reference tolerance is explicit per dtype and operation
- golden tests record the tolerance policy
- failures report max absolute error, max relative error, dtype, shape, and
  source operation IDs when available

V0 tolerance defaults:

```text
f32: rtol=1e-5 atol=1e-6
bf16: rtol=5e-2 atol=5e-2
i32/i64: exact
bool: exact
```

Backend-specific tests may tighten tolerances. They may not loosen tolerances
without recording the reason in the test and explain report.

## Profiling Correctness

Profiling must not change semantics.

Instrumentation rules:

- record timestamps around existing command boundaries
- record forced synchronization explicitly
- do not insert synchronization silently
- do not hide data races by making execution more serial unless the profile
  mode says so
- redact timing values in golden reports while preserving event topology

## Report Requirements

Correctness report sections must include:

```text
math_mode
relaxations
shape_status
dtype_status
lowering_correctness_status
schedule_correctness_status
reference_oracle_status
tolerance_policy
```

Every failure must include the first failing stage and enough source/graph IDs
to trace it back to the program.
