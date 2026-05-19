# PjRTx V0 Specs

These specs are the implementation contract for the new architecture under
`//next/pjrtx/...`. They intentionally live beside the current README/bootstrap code
without replacing it.

## Source Of Truth

- `../pjrtx_architecture_vision.md` is the long-range architecture.
- `../vertical_slice_v0.md` defines the first end-to-end scope.
- `mlir_state_machine_compiler_v0.md` defines the intended MLIR-first compiler
  state machine. Current Zig records are extracted views and scaffolding, not
  the long-term compiler truth.
- `pjrtx_mlir_dialect_v0.md` defines the first concrete PjRTx MLIR dialect
  surface, Zig `MlirSession` API shape, verifier boundaries, and fusion
  extraction slice.
- `final_mlir_dialect_op_pass_architecture_v0.md` defines the destination
  dialect/op/pass architecture after the current attribute bridge.
- `package_boundaries_v0.md` defines the segregated package graph and the
  no-`//next/pjrtx/core` ownership rule.
- `trace_schema_v0.md` defines the extracted records that make the compiler
  explainable to tests, runtime handoff, and users.
- `target_model_v0.md` defines hardware, memory, transfer, and dtype-rate data.
- `compile_pipeline_v0.md` defines the no-fallback compile gates.
- `compiler_pass_pipeline_v0.md` defines the XLA-like, explainable compiler
  pass families and pass contracts.
- `xla_coverage_map_v0.md` records what the local XLA tree already covers and
  how PjRTx should match that compiler/runtime breadth without inheriting XLA's
  opacity.
- `harness_v0.md` defines the fast/debuggable iteration loop.
- `correctness_policy_v0.md` defines strict mathematical correctness defaults.
- `implementation_workplan_v0.md` defines the first concrete package and task
  breakdown.

The specs may be narrower than the vision, but they must not weaken these
properties:

- no runtime fallback
- explicit target legality
- stable source-to-backend traceability
- MLIR-owned compiler-middle state, with Zig records extracted from verified
  MLIR instead of acting as the permanent lowering representation
- Zig-owned compiler policy where practical, including MLIR external passes
  through the C API, with minimal C++/TableGen reserved for real MLIR dialect
  plumbing and rewrite facilities unavailable through C
- strict mathematical correctness by default
- deterministic reports
- debuggable diagnostics written through `std.Io.Writer`
- TigerStyle-inspired implementation discipline for safety, performance, and
  developer experience
- standard-library-first implementation: new helpers should express PjRTx
  domain concepts, not duplicate Zig `std`
- intent-oriented comments and docs: explain invariants, design intent, and
  contracts, not obvious mechanics
- continuous refactoring when the current shape obscures ownership,
  correctness, performance, or explainability
- segregated package ownership: no central `//next/pjrtx/core` bucket for new
  architecture code

## Canonical Ownership

To avoid duplicate specs drifting apart, each topic has exactly one canonical
home:

| Topic | Canonical spec | Other docs should do this |
| --- | --- | --- |
| Long-range goal and principles | `../pjrtx_architecture_vision.md` | Summarize briefly, then link back. |
| Package graph and no-core migration | `package_boundaries_v0.md` | Do not restate full package layouts. |
| Final MLIR dialect/op/pass destination | `final_mlir_dialect_op_pass_architecture_v0.md` | Do not duplicate op/pass family lists. |
| Current V0 MLIR attribute bridge | `pjrtx_mlir_dialect_v0.md` | Describe only the live bridge and extraction. |
| MLIR compiler-state rule | `mlir_state_machine_compiler_v0.md` | Keep "MLIR is truth" here, not in every phase. |
| Pass contracts and V0 pass order | `compiler_pass_pipeline_v0.md` | Workplans may reference phase names only. |
| Compile stage gates | `compile_pipeline_v0.md` | Avoid duplicating stage contracts elsewhere. |
| Extracted fact schema | `trace_schema_v0.md` | Package docs state ownership, not schemas. |
| Hardware model | `target_model_v0.md` | Backend/runtime docs consume it, not redefine it. |
| Harness and fast loop | `harness_v0.md` | Other docs list target names only when needed. |
| Concrete queue | `implementation_workplan_v0.md` | Keep it operational, not architectural. |

When a document needs detail owned by another spec, prefer a short intent
sentence plus the canonical link. The logbook may repeat historical context,
but current specs should not maintain parallel versions of the same rule.

## Implementation Order

1. Establish package boundaries from `package_boundaries_v0.md`: target,
   MLIR, dialects, compiler facts, backend facts, runtime facts, report, and
   plugin packages, with no new dependency on `//next/pjrtx/core`.
2. Move existing bootstrap records out of `//next/pjrtx/core` by ownership while
   keeping tests green after each move.
3. Add report writer and normalization tests under `//next/pjrtx/report`.
4. Add StableHLO/MLIR ingest through Bazel MLIR/StableHLO/Shardy C API targets.
5. Import the V0 typed graph for `tanh(dot(x, w) + b)`.
6. Add target selection and legality gates for `metal_v0` and
   `npu_v0`.
7. Add the compiler-middle spine: MLIR lowering pass reports, fusion candidate
   and decision records, layout/tile/memory-space plans, and collective
   lowering diagnostics. Treat these Zig records as schema prototypes for
   PjRTx MLIR dialect/state-machine facts.
   Use `compiler_pass_pipeline_v0.md` and
   `mlir_state_machine_compiler_v0.md` for pass ordering and contracts. Use
   `pjrtx_mlir_dialect_v0.md` for the first MLIR dialect/API implementation
   slice.
8. Add a Zig external MLIR pass proof through the MLIR C API pass manager, then
   introduce minimal TableGen/C++ dialect plumbing for real PjRTx dialect
   syntax and verifiers.
9. Add cost ledger, lowering records, schedule records, backend binding
   records, and explain records that join those compiler-middle artifacts.
10. Add report-only vertical slice tests.
11. Bind Metal execution only after compile gates and report validation pass.
12. Add NPU synthetic profile events to test hardware-aware traceability.
13. Add PJRT/JAX smoke only after lower layers are stable.

For the detailed build queue, use `implementation_workplan_v0.md`.

## Hard Rules

Unsupported programs fail during compile. Reference execution is a test oracle,
not an execution strategy. Library calls are valid backend lowerings when they
are selected intentionally and recorded as backend bindings; they are not a
fallback.

PjRTx must own compiler decisions instead of treating XLA as a hidden compiler
and implementing only a StreamExecutor-like backend. MLIR, StableHLO, Shardy,
and XLA concepts are references and building blocks, but fusion, tiling,
layout, memory-space, collective, schedule, and backend-binding decisions must
be represented in PjRTx artifacts. The long-term source of truth for those
compiler-middle decisions is MLIR dialect state with verifiers; Zig records are
the extracted report and handoff surface.

PjRTx may create its own MLIR dialects. The implementation rule is Zig-first
for compiler policy and minimal MLIR-native C++/TableGen for the dialect
surface itself. Zig external passes are the preferred proof point before adding
larger dialect infrastructure.

Every new artifact should have:

- a typed ID when it appears in reports
- validation
- `std.Io.Writer` printing
- a deinit/ownership story
- at least one focused test
- a path back to source operations when it affects computation
- a performance sketch when it can affect compile time, runtime, allocation,
  transfer, or synchronization
- clear names with units where relevant, such as `_bytes`, `_ns`, `_ops`, or
  `_bytes_per_second`
- a reason when it introduces a helper that replaces or wraps an existing Zig
  `std` primitive
- comments for non-obvious intent, invariants, and contracts
- typed initialization style where it makes construction clearer
- tests that avoid inline `@as(T, value)` acceptance noise
