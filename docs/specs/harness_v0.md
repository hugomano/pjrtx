# Harness V0

This spec defines the fast, debuggable development harness for the new PjRTx
architecture under `//pjrtx/...`.

The harness exists to keep iteration tight:

```text
edit
  -> focused Bazel test
  -> stable report diff
  -> optional local JAX execution
  -> profile/explain inspection
  -> broader regression test
```

V0 harness quality is part of the architecture. If reports are unstable,
diagnostics are vague, or tests require full end-to-end execution for every
change, the architecture will become hard to evolve.

## Requirements

The harness must:

- run fast for core/compiler/report tests
- produce deterministic report output
- redact timing fields in golden tests
- make diagnostics testable
- make missing provenance testable
- make it testable that unlowerable programs never reach runtime
- support optional execution through Metal
- support future NPU tests
- keep current `//src/...` tests intact

## Target Test Layers

### Level 1: Boundary And Fact Unit Tests

No MLIR, no backend, no JAX.

Targets:

```text
//pjrtx/target:unit_tests
//pjrtx/compiler/facts:unit_tests
//pjrtx/backend/facts:unit_tests
//pjrtx/runtime/facts:unit_tests
//pjrtx/report:unit_tests
```

Coverage:

- typed ID formatting and ordering
- tensor type validation
- target model validation
- report normalization from extracted facts
- cost formulas
- schedule dependency validation
- report sorting
- explain record validation
- ownership/deinit smoke tests

Expected runtime: tiny.

### Level 2: Compiler Import Tests

Uses MLIR/StableHLO/Shardy C API. No runtime execution.

Targets:

```text
//pjrtx/compiler:unit_tests
//pjrtx/tests/vertical_slice:import_tests
```

Coverage:

- parse StableHLO fixture
- verify module
- optional canonicalization
- import typed graph
- reject unsupported operations
- diagnostics include pass/op/shape/dtype/feature
- emit stable MLIR lowering pass summaries

Expected runtime: small.

### Level 2B: Compiler Middle Tests

Uses imported graph and target descriptions. No backend executable submission.

Targets:

```text
//pjrtx/compiler:unit_tests
//pjrtx/tests/vertical_slice:lowering_tests
```

Coverage:

- MLIR pass-pipeline reports
- fusion candidate discovery
- accepted and rejected fusion records
- layout and tile records
- memory-space assignment records
- allocation summaries that prove fused-internal values are not materialized as
  standalone device buffers
- unsupported collective lowering diagnostics
- no XLA-as-hidden-compiler or runtime fallback path

Expected runtime: small to moderate.

### Level 3: Report Tests

Build graph, MLIR-extracted cost ledger, MLIR-extracted schedule, backend
binding records, explain records, and normalized report. Execution is optional.

Targets:

```text
//pjrtx/tests/vertical_slice:report_test
```

Coverage:

- report section order
- stable IDs
- graph-to-cost links
- lowering-to-schedule links
- backend binding links
- profile event join slots
- redacted timings

Expected runtime: small.

### Level 4: Backend Binding Tests

Uses Metal or NPU backend records. May not execute full JAX path.

Targets:

```text
//pjrtx/backend:unit_tests
//pjrtx/tests/vertical_slice:backend_binding_test
```

Coverage:

- backend capability records
- backend binding records
- unsupported backend feature diagnostics
- NPU target execution-unit binding

Expected runtime: moderate.

### Level 5: PJRT/JAX Smoke Tests

Full integration path.

Targets:

```text
//pjrtx/tests/vertical_slice:execution_test
```

Coverage:

- compile through PJRT path
- execute supported workload
- compare with JAX CPU or NumPy
- emit profile events
- emit explain/cost report

Expected runtime: slower. Do not require this for every small edit.

## Fixtures

Fixtures should be small and explicit.

```text
pjrtx/tests/fixtures/tanh_dot_bias.mlir
pjrtx/tests/fixtures/elementwise_chain.mlir
pjrtx/tests/fixtures/unsupported_convolution.mlir
pjrtx/tests/fixtures/bad_dot_shape.mlir
pjrtx/tests/fixtures/shardy_tiny_mesh.mlir
pjrtx/tests/fixtures/elementwise_fusion.mlir
pjrtx/tests/fixtures/unsupported_all_reduce.mlir
```

Fixture purposes:

```text
tanh_dot_bias.mlir
  Main V0 workload.

elementwise_chain.mlir
  Pure elementwise cost/fusion/report test.

unsupported_convolution.mlir
  Failure diagnostics and unsupported feature path.

bad_dot_shape.mlir
  Shape verification failure before backend execution.

shardy_tiny_mesh.mlir
  Shardy metadata import and future sharding trace.

elementwise_fusion.mlir
  Fusion candidate and accepted fusion report test.

unsupported_all_reduce.mlir
  Collective legality and no-fallback failure test.
```

Fixture files should be reviewed like code. Avoid giant generated MLIR in V0.

## Golden Reports

Golden reports should compare normalized text first. JSON can come later.

Rules:

- sort records by typed ID
- redact `start_ns`, `duration_ns`, and wall-clock totals
- preserve event kind, command ID, bytes, logical ops, and forced sync flag
- avoid pointer addresses
- avoid nondeterministic map ordering
- represent unknown fields as `unknown`
- include formulas as stable strings

Example redaction:

```text
profile.event.3 backend_execute duration_ns=<redacted>
```

Golden report sections:

```text
program
target
graph
cost
lowering
schedule
backend
profile
correctness
explain
```

## Debug Flags

The exact names can change, but the harness should provide these capabilities:

```text
PJRTX_TRACE=1
PJRTX_DUMP_GRAPH=1
PJRTX_DUMP_MLIR_PASSES=1
PJRTX_DUMP_FUSION=1
PJRTX_DUMP_TILING=1
PJRTX_DUMP_MEMORY_PLAN=1
PJRTX_DUMP_COLLECTIVES=1
PJRTX_DUMP_COST=1
PJRTX_DUMP_SCHEDULE=1
PJRTX_DUMP_PROFILE=1
PJRTX_DUMP_EXPLAIN=1
PJRTX_REQUIRE_BACKEND_EXECUTABLE=1
PJRTX_ASSERT_NO_RUNTIME_FALLBACK=1
```

Debug dumps should write through `std.Io.Writer` and should be stable enough to
read in tests when useful.

## Failure Tests

Failure tests are required in V0.

Assert:

- unsupported ops fail during compile
- unsupported lowering passes fail before typed executable artifacts
- diagnostics include pass, op, shape/dtype, and feature
- invalid shapes fail before backend execution
- invalid fusion, layout, tile, memory-space, or collective choices fail before
  schedule build
- missing backend capability fails with an explanation
- unlowerable programs never reach runtime execution
- malformed report links fail validation

## Boundary Tests

The new package tree has a first live architecture boundary test:
`//pjrtx/tests/architecture:boundary_test`.

It is a migration guardrail, not the final visibility model. It pins the current
bootstrap `//pjrtx/core` import baseline exactly, forbids new `core` references,
and checks the active compiler/backend/runtime/plugin package edges. When a
package migrates records out of `core`, the same change should reduce the pinned
baseline.

Rules:

- `//pjrtx/target` cannot import compiler/runtime/backend/plugin/report
- no package imports `//pjrtx/core` after migration
- `//pjrtx/compiler` cannot import runtime/backend/plugin
- `//pjrtx/runtime` cannot import backend implementations or Metal C symbols
- `//pjrtx/plugin` cannot mention Metal backend shim symbols
- backend implementations cannot leak into target/compiler/runtime fact packages
- graph/cost/schedule modules cannot import compiler/runtime/backend/plugin

Bazel visibility should enforce most final boundaries. String-based smoke tests
are acceptable during migration when they preserve exact baselines and fail on
new debt.

## Profile Harness

Tests should emit lightweight profile events:

- compile/import
- h2d
- coarse backend_execute command events
- backend lowering-region events for each generated or selected kernel region
- d2h

Golden tests redact timings but preserve:

- event kind
- command ID
- graph instruction IDs
- bytes
- logical ops
- lowering-region matches
- forced synchronization flag
- status

Manual performance runs may keep real timings.

## Fast Commands

Expected command families:

```sh
bazel test //pjrtx/target:unit_tests
bazel test //pjrtx/compiler/facts:unit_tests
bazel test //pjrtx/compiler:unit_tests
bazel test //pjrtx/tests/vertical_slice:lowering_tests
bazel test //pjrtx/tests/vertical_slice:report_test
bazel test //pjrtx/tests/vertical_slice:backend_binding_test
bazel test //pjrtx/tests/vertical_slice:execution_test
```

Exact targets may change during implementation. The layered loop should not.

## V0 Decisions

Fixtures live under `pjrtx/tests/fixtures`. Golden reports and normalized
expected outputs live under `pjrtx/tests/vertical_slice/testdata`. This keeps
source programs reusable across multiple test layers while keeping report
expectations close to the vertical slice that owns them.

Debug dumps write human-readable text to stderr by default. If `PJRTX_DUMP_DIR`
is set, dumps are also written as deterministic files named by stage and stable
test case ID. Report tests compare captured report sections, not arbitrary
stderr, unless the test is specifically about diagnostics.

Execution tests skip gracefully when local Metal is unavailable. CI should
still run NPU and report-only tests. If `PJRTX_REQUIRE_BACKEND_EXECUTABLE=1`
is set, missing Metal is a hard failure. Skipping hardware execution never
permits fallback execution.

Golden reports normalize target-dependent device names to `<device-name>` while
preserving target kind, memory-space names, execution-unit names, dtype rates,
and transfer-edge structure.
