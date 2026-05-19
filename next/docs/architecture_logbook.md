# Architecture Logbook

This logbook records architecture decisions, reversals, open questions, and
follow-up threads for the new PjRTx architecture under `//next/pjrtx/...`.

It exists so the project does not lose reasoning in chat history. When a design
changes, add a short entry here with the date, decision, why it changed, and
where to look next.

The current README and `//src/...` tree remain the bootstrap implementation.
This logbook tracks the new architecture described in `next/docs/` and implemented
under `//next/pjrtx/...`.

Historical entries are not current instructions. When older entries disagree
with current specs, follow `next/docs/specs/README.md` and the canonical spec named
there. The logbook explains why the project moved; it does not override the
current contract.

## How To Use This Logbook

Add an entry whenever:

- a new architecture decision is made
- a prior decision is reversed or narrowed
- an open question is answered
- an implementation discovers the design was wrong
- a test or benchmark changes the performance/correctness direction
- a document is split, merged, or made canonical

Entry template:

```text
## YYYY-MM-DD - Short Title

Status: proposed | accepted | changed | superseded | rejected

Context:
What question or problem led to this?

Decision:
What did we decide?

Reasoning:
Why is this the right choice now?

Consequences:
What changes because of this?

Follow-up:
What should be checked or implemented next?

Links:
- next/docs/...
- code target or test target
```

Keep entries short. The goal is not to duplicate the design docs; it is to
preserve the trail of why the docs look the way they do.

## 2026-05-18 - Two Versions Live Side By Side

Status: accepted

Context:
The existing README and `//src/...` implementation describe the current
bootstrap PJRT plugin. The new architecture is broader and should not
accidentally rewrite or destabilize that work.

Decision:
Keep two versions side by side:

```text
README.md + //src/...      current/bootstrap implementation
next/docs/*.md + //next/pjrtx/...    new architecture and implementation
```

Reasoning:
This lets the current implementation remain usable while the new compiler,
runtime, backend, profiling, and traceability architecture is built cleanly.

Consequences:
New architecture code should be rooted under `//next/pjrtx/...`. Bridges to
`//src/...` are allowed during migration, but new public architecture should
not be mixed into `//src/...`.

Follow-up:
Create Bazel packages under `//next/pjrtx/...` when implementation starts.

Links:
- `next/docs/pjrtx_architecture_vision.md`
- `next/docs/vertical_slice_v0.md`

## 2026-05-18 - Main Architecture Doc Is Canonical

Status: accepted

Context:
There were separate docs for architecture vision, vertical slice, and backend
kernel generation. The backend document overlapped heavily with the vision.

Decision:
Merge backend/kernel-generation content into `next/docs/pjrtx_architecture_vision.md`
and delete the standalone backend doc.

Reasoning:
Backend-specific kernel generation is not separate from the architecture. It is
the bridge from graph/schedule to real performance and should live in the
canonical vision doc.

Consequences:
`next/docs/pjrtx_architecture_vision.md` is the canonical architecture document.
`next/docs/vertical_slice_v0.md` is the first scoped implementation slice.

Follow-up:
Keep future cross-cutting design material in the architecture doc unless it is
specific to one implementation slice.

Links:
- `next/docs/pjrtx_architecture_vision.md`
- `next/docs/vertical_slice_v0.md`

## 2026-05-18 - Vertical Slice V0 Is Scope, Not General Style

Status: accepted

Context:
The vertical slice doc originally contained reusable implementation guidance:
Zig IO, package layout, diagnostics, stable reports, and harness details. That
made the slice document too broad.

Decision:
Move reusable implementation and harness guidance into the main architecture
doc. Keep `next/docs/vertical_slice_v0.md` focused on the V0 workload, artifacts,
scope, tests, report shape, and exit criteria.

Reasoning:
The vertical slice should answer "what is V0?" not become a second architecture
manual.

Consequences:
General implementation discipline now lives in the architecture doc.

Follow-up:
When a new slice is added, it should reference the architecture doc for shared
style and only define its own workload/scope/exits.

Links:
- `next/docs/pjrtx_architecture_vision.md`
- `next/docs/vertical_slice_v0.md`

## 2026-05-18 - Vertical Traceability Is A Requirement

Status: accepted

Context:
XLA can make some computations difficult to explain because provenance gets
lost or spread across backend-specific side channels. PjRTx wants a stronger
PJRT-to-hardware trace.

Decision:
Make vertical traceability a design requirement. A user should be able to trace
from PJRT-visible computation through StableHLO, graph IR, lowering, schedule
commands, generated or selected kernels, hardware units, profile events, and
correctness records.

Reasoning:
Performance data detached from program semantics is not explainable. Correct
graphs detached from hardware binding are not performance-engineered.

Consequences:
Core records need stable source IDs, cost ledgers, backend bindings, profile
events, and explain records.

Follow-up:
V0 should prove this on `tanh(dot(x, w) + b)`.

Links:
- `next/docs/pjrtx_architecture_vision.md`
- `next/docs/vertical_slice_v0.md`

## 2026-05-18 - Harness And Fast Iteration Are Architecture

Status: accepted

Context:
The project will become complex quickly. Without a fast harness, architecture
decisions will become hard to validate and regressions will be difficult to
debug.

Decision:
Document the harness and iteration loop as part of the architecture:
focused Bazel tests, stable report diffs, debug dump modes, profile/explain
inspection, boundary tests, and failure-quality tests.

Reasoning:
A compiler/runtime architecture is only useful if contributors can quickly see
what changed and why.

Consequences:
Implementation should prioritize stable reports, fixture tests, and debug
output alongside the first records.

Follow-up:
Add initial `//next/pjrtx/vertical_slice` report tests during V0
implementation.

Links:
- `next/docs/pjrtx_architecture_vision.md`

## 2026-05-18 - No Runtime Fallback

Status: accepted

Context:
The new architecture is performance and explainability oriented. Runtime
fallback would hide unsupported lowering, detach performance data from program
semantics, and make correctness/performance behavior ambiguous.

Decision:
There is no runtime fallback path. If a program cannot be fully legalized,
lowered, scheduled, and backend-bound for the selected target, compilation
fails before runtime. Reference execution is only a test oracle.

Reasoning:
The runtime should execute verified schedules, not rescue incomplete compiler
work.

Consequences:
The compile pipeline needs explicit legality gates. Harness failure tests must
assert that unlowerable programs never reach runtime.

Follow-up:
Implement compile pipeline stage checks before executable creation.

Links:
- `next/docs/pjrtx_architecture_vision.md`
- `next/docs/specs/compile_pipeline_v0.md`
- `next/docs/specs/harness_v0.md`

## 2026-05-18 - V0 Specs Added

Status: accepted

Context:
The vision and vertical-slice docs were still too broad to implement directly.
The next step was to define small implementation-facing specs.

Decision:
Add focused V0 specs for trace schema, harness, target model, and compile
pipeline.

Reasoning:
These specs define the records and gates needed before writing code under
`//next/pjrtx/...`.

Consequences:
Implementation should start from the trace schema and compile pipeline gates,
not from ad hoc runtime execution.

Follow-up:
Create `//next/pjrtx/core` skeleton with trace schema types and report validation.

Links:
- `next/docs/specs/trace_schema_v0.md`
- `next/docs/specs/harness_v0.md`
- `next/docs/specs/target_model_v0.md`
- `next/docs/specs/compile_pipeline_v0.md`
- `next/docs/specs/implementation_workplan_v0.md`

## 2026-05-18 - Documentation Hardening Pass

Date:
2026-05-18

Context:
The first docs established the architecture, but several V0 specs still ended
with open questions. That made implementation too easy to steer from memory or
chat history.

Decision:
Convert remaining open questions into V0 decisions. Add a specs index. Add an
explicit correctness policy spec. Treat no-fallback, stable reports, source
provenance, target fingerprints, debug dumps, and hardware execution skips as
specified behavior.

Consequences:
Implementation can begin from docs without inventing policy at the keyboard.
When a design choice changes, update the spec and add a logbook entry before
changing code.

Links:
- `next/docs/specs/README.md`
- `next/docs/specs/correctness_policy_v0.md`
- `next/docs/specs/trace_schema_v0.md`
- `next/docs/specs/harness_v0.md`
- `next/docs/specs/target_model_v0.md`
- `next/docs/specs/compile_pipeline_v0.md`

## Next Implementation Threads

- Create `//next/pjrtx/core` with typed IDs, trace records, target records, report
  writer, validators, and ownership tests.
- Implement V0 as report-first compile infrastructure, then bind the supported
  Metal path only after no-fallback compile gates exist.
- Implement `npu_v0` before generated Metal kernels so memory spaces,
  dtype rates, transfer edges, execution units, and profile joins are testable
  without vendor-specific codegen.
- Use normalized text reports for the first stable golden format.
- Keep `README.md` and `//src/...` as bootstrap/legacy source of truth only.
  New architecture work references docs and lands under `//next/pjrtx/...`.

## 2026-05-18 - New Architecture Skeleton Started

Context:
Implementation began without creating a branch. The goal was to establish the
new package namespace before moving compiler logic.

Decision:
Create the first `//next/pjrtx/...` Bazel packages beside the existing `//src/...`
implementation. Add a short coding policy, core typed IDs, stable writer smoke
tests, and placeholder package tests for compiler/backend/runtime/plugin and
vertical slice targets.

Consequences:
The new architecture has a compilable home that does not import the legacy
`//src/...` namespace. Future work can grow `//next/pjrtx/core` toward the trace
schema and target model specs while keeping package boundaries visible.

Verification:
`bazel query //next/pjrtx/...` discovers the new package tree.
`bazel test //next/pjrtx/...` passes.

Links:
- `pjrtx/CODING_POLICY.md`
- `pjrtx/core/core.zig`
- `pjrtx/core/BUILD.bazel`
- `next/pjrtx/vertical_slice/BUILD.bazel`

## 2026-05-18 - Core Trace And Target Records Started

Context:
After the package skeleton passed, the next useful slice was to make
`//next/pjrtx/core` carry real architecture vocabulary instead of only placeholders.

Decision:
Add V0 core records for buffer types, tensor types, source refs, graph values,
graph instruction payloads, target descriptions, memory spaces, transfer edges,
execution units, target validation, and stable target summary writing.

Consequences:
The new architecture now has a small but real foundation for trace schema and
target model implementation. Validation writes diagnostics through
`std.Io.Writer`, unknown target fields render explicitly, and tests avoid
inline `@as(T, value)` acceptance casts.

Verification:
`bazel test //next/pjrtx/core:unit_tests --test_output=errors` passes.
`bazel test //next/pjrtx/... --test_output=errors` passes.

Links:
- `pjrtx/core/core.zig`
- `next/docs/specs/trace_schema_v0.md`
- `next/docs/specs/target_model_v0.md`

## 2026-05-18 - Trace Report Joins Started

Context:
The core package had graph and target records, but the remaining V0 trace
records were still only documented.

Decision:
Add cost ledger, lowering, schedule command, backend binding, profile event,
and explain records to `//next/pjrtx/core`. Add `TraceReport` validation for stable
ID order and cross-record joins. Add a stable summary writer that preserves
section order and redacts profile durations. Refactor placeholder backend and
runtime code to reuse core `BackendKind` and `CommandKind`.

Consequences:
Later compiler stages can now emit connected provenance records instead of
inventing per-package placeholders. The validator starts enforcing the
no-fallback invariant that backend execution commands require lowering
provenance.

Verification:
`bazel test //next/pjrtx/core:unit_tests --test_output=errors` passes.
`bazel test //next/pjrtx/... --test_output=errors` passes.
No `@as(` or `@import("src/...")` remains in `pjrtx/**/*.zig`.

Links:
- `pjrtx/core/core.zig`
- `pjrtx/backend/backend.zig`
- `pjrtx/runtime/runtime.zig`
- `next/docs/specs/trace_schema_v0.md`

## 2026-05-18 - Core Inline Documentation Policy

Context:
The design specs exist under `next/docs/`. The implemented `//next/pjrtx/core` contracts
need nearby intent, but standalone package schema files would duplicate the
central specs and create drift.

Decision:
Keep schema docs centralized under `next/docs/specs`. Add inline intent comments to
core validators and summary writers where the code encodes architecture
contracts. Keep `pjrtx/core/README.md` as a package orientation document that
points back to the central specs.

Consequences:
A reader can understand why core validators and writers exist without creating
a second schema source of truth. Schema changes happen in `next/docs/specs`; code
comments explain implemented intent.

Verification:
`bazel test //next/pjrtx/core:unit_tests --test_output=errors` passes.
`bazel test //next/pjrtx/... --test_output=errors` passes.

Links:
- `pjrtx/core/README.md`
- `pjrtx/core/core.zig`
- `pjrtx/CODING_POLICY.md`

## 2026-05-18 - Vertical Slice Golden Report Started

Context:
The core report writer existed, but the vertical slice harness still only had a
smoke test for ID formatting.

Decision:
Add a runfiles-backed golden report fixture under
`next/pjrtx/vertical_slice/testdata`. Update the report test to construct a
tiny connected trace, validate it, write the stable summary, read the expected
report from Bazel runfiles, and compare the exact text.

Consequences:
The report path now exercises the same style of golden file comparison the
compiler pipeline will use later. The fixture is runtime test data, not an
`@embedFile`, so it follows Bazel runfiles behavior.

Verification:
`bazel test //next/pjrtx/vertical_slice:report_test --test_output=errors`
passes.
`bazel test //next/pjrtx/... --test_output=errors` passes.
No `@embedFile`, `@as(`, or `@import("src/...")` remains in `pjrtx/**/*.zig`.

Links:
- `next/pjrtx/vertical_slice/report_test.zig`
- `next/pjrtx/vertical_slice/testdata/tiny_trace_report.txt`
- `next/pjrtx/vertical_slice/BUILD.bazel`

## 2026-05-18 - Compiler Gates Started

Context:
The core trace report existed, but compiler code still had only stage names.
The next implementation step was to make no-fallback compile gates executable
before adding real StableHLO import.

Decision:
Add V0 target selection for `metal_v0` and `npu_v0`, backend binding
verification, and executable-view creation. Treat executable creation as a
verified view over a trace report until runtime owns real command buffers.

Consequences:
The compiler now enforces that every backend command has exactly one binding
and that the binding backend matches the selected target. Runtime-facing
objects can only be created after trace validation and backend binding
verification pass.

Verification:
`bazel test //next/pjrtx/compiler:unit_tests --test_output=errors` passes.
`bazel test //next/pjrtx/... --test_output=errors` passes.
No `@embedFile`, `@as(`, or `@import("src/...")` remains in `pjrtx/**/*.zig`.

Links:
- `pjrtx/compiler/compiler.zig`

## 2026-05-18 - Vertical Execution Golden Report Started

Status: accepted

Context:
The vertical execution test checked counts and a few summary substrings, but
runtime allocation, stream, and predicted-versus-profiled output were not yet
protected by a stable full-report diff.

Decision:
Add a runfiles-backed golden execution report for the V0 StableHLO fixture. The
report combines allocation summary, stream summary, and runtime execution
summary in deterministic order.

Reasoning:
The harness should make performance-structure changes reviewable. Exact text
goldens are blunt, but they are very effective for this early architecture
phase because they show buffer, lifetime, event, and profile joins changing in
one place.

Consequences:
The vertical execution test now compares the generated execution report against
`next/pjrtx/vertical_slice/testdata/v0_execution_report.txt`. Future changes
to planning, allocation, stream events, or profile metrics must update the
golden intentionally.

Verification:
`bazel test //next/pjrtx/vertical_slice:execution_test --test_output=errors`
passes.

Links:
- `next/pjrtx/vertical_slice/execution_test.zig`
- `next/pjrtx/vertical_slice/testdata/v0_execution_report.txt`
- `next/pjrtx/vertical_slice/BUILD.bazel`

## 2026-05-18 - Hardware Utilization Summary Started

Status: accepted

Context:
The execution golden showed allocation, streams, and predicted-versus-profiled
command metrics, but it did not yet summarize the target model view: which
memory spaces and execution units the workload pressures.

Decision:
Add a runtime hardware utilization summary that reports per-memory-space
peak-live bytes and allocated bytes, per-execution-unit predicted ops and bytes
from the cost ledger, and target dtype/op-class peak-rate rows joined with
predicted work. Rate rows also include an ideal compute time estimate derived
from predicted logical ops and target peak ops/second.

Reasoning:
The architecture goal is to trace from hardware specifications to lowered work
and profile data. Even before real counters exist, the report should expose how
the target model is being used.

Consequences:
The V0 execution golden now includes memory-space pressure for host pinned,
device HBM, and local SRAM, predicted work for synthetic matrix, vector, DMA, and
collective units, and target-rate rows for f32/bf16 matmul plus f32
elementwise/transcendental work. The golden now shows picosecond-scale ideal
compute estimates for the tiny V0 fixture.

Verification:
`bazel test //next/pjrtx/vertical_slice:execution_test --test_output=errors`
passes.

Links:
- `pjrtx/runtime/runtime.zig`
- `next/pjrtx/vertical_slice/testdata/v0_execution_report.txt`
- `next/docs/specs/compile_pipeline_v0.md`
- `next/docs/specs/target_model_v0.md`

## 2026-05-18 - Transfer Bandwidth Reporting Added

Status: accepted

Context:
The first hardware utilization report explained memory pressure and compute
roofline rows, but H2D/D2H movement still appeared only as command bytes. That
left the bandwidth side of the target model disconnected from execution.

Decision:
Give `npu_v0` synthetic host/device transfer bandwidth and have the
runtime report compute ideal transfer time per transfer command and per target
transfer edge. The hardware summary now lists all transfer edges, including
zero-use HBM/SRAM edges, so unused modeled hardware stays visible.

Consequences:
The V0 golden shows H2D and D2H command bytes mapped to memory.0->memory.1 and
memory.1->memory.0 edges with picosecond-scale ideal transfer estimates. This
keeps the vertical slice closer to the intended PJRT-to-hardware explanation:
kernel work maps to execution units, and IO work maps to transfer edges.

Verification:
`bazel test //next/pjrtx/vertical_slice:execution_test --test_output=errors`
passes and covers the golden report update.

Links:
- `pjrtx/compiler/compiler.zig`
- `pjrtx/runtime/runtime.zig`
- `next/pjrtx/vertical_slice/testdata/v0_execution_report.txt`

## 2026-05-18 - Backend Names Aligned With Metal And TRN2-like NPU

Status: accepted

Context:
The V0 architecture still carried old bootstrap naming from the earlier
Metal-library idea. That implied the new path would depend on a library bridge,
while the intended backend direction is generated Metal/MLS graph and kernel
planning.

Decision:
Use `metal_v0` for the Metal backend and replace backend operation names with
`metal_mls_graph_execute` plus per-instruction `metal_mls_*_kernel`
operations. Use `npu_v0` for the NPU target model and make it TRN2-like with
tensor, vector, DMA, collective, HBM, and local SRAM concepts.
Lowering vocabulary now uses `backend_kernel_graph` rather than backend library
calls for the V0 kernel/graph path.

Consequences:
The new `//next/pjrtx/...` tree now uses Metal and NPU naming consistently.
Docs and golden reports now describe Metal/MLS kernel graph planning and an
NPU target model suitable for TRN2-style performance reasoning.

Verification:
`bazel test //next/pjrtx/... --test_output=errors` passes.

Links:
- `pjrtx/core/core.zig`
- `pjrtx/backend/backend.zig`
- `pjrtx/compiler/compiler.zig`
- `next/pjrtx/vertical_slice/metal_bridge_test.zig`
- `next/docs/specs/target_model_v0.md`

## 2026-05-18 - Metal MLS Kernel Graph Plan Started

Status: accepted

Context:
The Metal bridge could expand a verified backend binding into ordered kernel
operations, but the report still did not show kernel-to-kernel dataflow. That
made it harder to reason from graph values to generated MLS graph structure.

Decision:
Add `BackendKernelGraphPlan`, `BackendKernelGraphNode`, and
`BackendKernelGraphEdge` to `//next/pjrtx/backend`. V0 creates one Metal MLS graph
node per executable call and adds an edge whenever a node consumes a graph value
produced by an earlier node.

Consequences:
The Metal vertical slice now proves that the four planned kernels form a graph
with three value-flow edges: matmul output and broadcast output feed add, and
add output feeds tanh. NPU kernel graph planning remains invalid in V0 until it
has a target-specific graph representation.

Verification:
`bazel test //next/pjrtx/backend:unit_tests //next/pjrtx/vertical_slice:metal_bridge_test --test_output=errors`
passes.

Links:
- `pjrtx/backend/backend.zig`
- `next/pjrtx/vertical_slice/metal_bridge_test.zig`

## 2026-05-18 - Metal MLS Kernel Descriptors Added

Status: accepted

Context:
The MLS kernel graph had topology, but each node still needed enough typed
metadata for future codegen. Without descriptors, later layers would have to
infer dtype, shape, and StableHLO attributes from value IDs or operation
strings.

Decision:
Add output tensor descriptors and typed kernel attributes to
`BackendKernelGraphNode`. The Metal graph planner now copies output dtype,
shape, layout, and attributes such as dot contracting dimensions or broadcast
dimensions from the verified trace report.

Consequences:
The Metal bridge summary can now explain each MLS node as a kernel with
operation, dtype, rank, input/output counts, and attributes. This is still
pre-command-buffer work, but it gives future Metal codegen a typed contract.

Verification:
`bazel test //next/pjrtx/vertical_slice:metal_bridge_test --test_output=errors`
passes.

Links:
- `pjrtx/backend/backend.zig`
- `next/pjrtx/vertical_slice/metal_bridge_test.zig`

## 2026-05-18 - Metal Backend Executable Plan Started

Status: accepted

Context:
The compiler could create Metal backend binding records, but the binding
still collapsed all work into a single operation string. Before runtime submits
anything to Metal, the backend layer needs to prove which graph instructions map
to which backend calls.

Decision:
Add `BackendExecutablePlan` and `BackendExecutableCall` to `//next/pjrtx/backend`.
Executable planning validates the trace report, checks that the binding belongs
to the report, checks selected target/backend agreement, verifies the backend
executable operation string, and expands each graph instruction into a concrete
backend operation such as `metal_mls_matmul_kernel`, `metal_mls_broadcast_kernel`, `metal_mls_add_kernel`, or
`metal_mls_tanh_kernel`.

Consequences:
The new vertical Metal bridge test compiles `tanh(dot(x, w) + b)` for
`metal_v0` and proves the binding expands into four deterministic Metal
MLS kernel-planning calls. This is still pre-submission: no reference execution
and no submitted Metal command-buffer claim.

Verification:
`bazel test //next/pjrtx/backend:unit_tests //next/pjrtx/vertical_slice:metal_bridge_test --test_output=errors`
passes.

Links:
- `pjrtx/backend/backend.zig`
- `next/pjrtx/vertical_slice/metal_bridge_test.zig`
- `next/pjrtx/vertical_slice/BUILD.bazel`

## 2026-05-18 - Runtime Allocation Preconditions Hardened

Status: accepted

Context:
Transfer edges became report-visible, but the helper used for report printing
could still treat a missing transfer edge as an unknown estimate. That is fine
for absent bandwidth, but not for absent topology. With no fallback path, a
scheduled transfer without a modeled edge must fail before execution.

Decision:
Make allocation planning verify two target/workload contracts: peak live bytes
must not exceed known memory-space capacity, and every H2D/D2H command must map
to an explicit target transfer edge. Add runtime unit tests for both failures.

Consequences:
Runtime planning now separates unknown performance facts from impossible
hardware paths. Unknown bandwidth can still render as `ideal_transfer_ps=0`,
but a missing transfer edge is an invalid schedule.

Verification:
`bazel test //next/pjrtx/runtime:unit_tests --test_output=errors` passes.

Links:
- `pjrtx/runtime/runtime.zig`
- `next/docs/specs/target_model_v0.md`

## 2026-05-18 - Input Setup Gate Added

Context:
The compiler had target and backend binding gates, but Stage 1 of the V0
pipeline still lacked a typed artifact.

Decision:
Add `CompileInput`, `ProgramFormat`, and `CompileOptions` to
`//next/pjrtx/compiler`. Input setup now parses program format, target kind, and
compile options, then ingests program bytes through `std.Io.Reader` into an
owned compile artifact. Unsupported formats, unknown targets, invalid options,
and empty programs fail with `std.Io.Writer` diagnostics at `input_setup`.

Consequences:
StableHLO/VHLO ingest can now receive a typed owned input instead of raw
strings. The first compile gate is explicit and testable, and failure happens
before target legality or runtime-shaped artifacts.

Verification:
`bazel test //next/pjrtx/compiler:unit_tests --test_output=errors` passes.
`bazel test //next/pjrtx/... --test_output=errors` passes.
No `@embedFile`, `@as(`, or `@import("src/...")` remains in `pjrtx/**/*.zig`.

Links:
- `pjrtx/compiler/compiler.zig`
- `next/docs/specs/compile_pipeline_v0.md`

## 2026-05-18 - StableHLO C API Ingest Started

Context:
Stage 1 produced a typed `CompileInput`. Stage 2 needed to prove that the new
architecture uses the real MLIR/StableHLO/Shardy C API boundary instead of an
ad hoc parser.

Decision:
Add a new `//next/pjrtx/compiler:mlir_c_api` C/C++ boundary, register the func,
shape, CHLO, Shardy, and StableHLO dialects, and implement StableHLO text parse
and verification through MLIR C API. Add a `tanh_dot_bias.mlir` fixture and a
vertical-slice `import_tests` target that reads the fixture through Bazel
runfiles.

Consequences:
The new compiler namespace now has a real Stage 2 ingest path for StableHLO
text. Portable artifact/bytecode ingest still fails explicitly until that path
is implemented; it does not fall back to another parser.

Verification:
`bazel test //next/pjrtx/compiler:unit_tests //next/pjrtx/vertical_slice:import_tests //next/pjrtx/vertical_slice:report_test --test_output=errors`
passes.
`bazel test //next/pjrtx/... --test_output=errors` passes.
No `@embedFile`, `@as(`, or `@import("src/...")` remains in `pjrtx/**/*.zig`.

Links:
- `pjrtx/compiler/compiler.zig`
- `pjrtx/compiler/mlir_c_api.h`
- `pjrtx/compiler/mlir_register.cc`
- `next/pjrtx/fixtures/tanh_dot_bias.mlir`
- `next/pjrtx/vertical_slice/import_test.zig`

## 2026-05-18 - Typed Graph Import And Verification Started

Status: accepted

Context:
StableHLO ingest proved the MLIR C API boundary, but the compiler still needed
owned graph records with stage-local failure for unsupported operations,
dynamic shapes, and shape-contract mistakes.

Decision:
Import the V0 StableHLO operation set into `GraphModule`, then verify graph
IDs, source provenance, tensor types, rank-2 dot shape contracts, broadcast
dimension maps, elementwise type equality, and return shape behavior before
target legality or backend binding.

Reasoning:
Parseable MLIR is not yet PjRTx compiler data. The graph layer must establish
mathematical correctness and explainable provenance before performance
planning begins.

Consequences:
Unsupported StableHLO fails in `graph_import`; dynamic dimensions fail in V0;
invalid dot shapes fail in `graph_verify`; every imported instruction carries
a canonical source reference.

Verification:
`bazel test //next/pjrtx/compiler:unit_tests --test_output=errors` passes.
`bazel test //next/pjrtx/vertical_slice:import_tests --test_output=errors`
passes.

Links:
- `pjrtx/compiler/compiler.zig`
- `next/pjrtx/vertical_slice/import_test.zig`

## 2026-05-18 - Runtime Allocation Plan Started

Status: accepted

Context:
The runtime could emit synthetic profile events from a verified schedule, but
the schedule still had no allocator-shaped artifact describing which graph
values live in host or device memory and which commands read or write them.

Decision:
Add a runtime allocation plan with buffer IDs, host/device placement, target
memory-space IDs, value sizes, command buffer uses, buffer lifetimes, a
peak-live-device-byte summary, and a stable text summary writer. V0 allocates
a device buffer for every graph value and host buffers only at explicit
H2D/D2H boundaries.

Reasoning:
Performance explainability needs memory to be visible before real allocator
calls land. This plan is deliberately simple but gives future stream execution,
device allocation, and memory-space optimization a typed contract to refine.

Consequences:
The vertical execution test now checks allocation count, command buffer-use
count, lifetime count, peak live device bytes, and summary text for the
StableHLO fixture. Real allocator calls, lifetime reuse, and SRAM/HBM
assignment remain future work.

Verification:
`bazel test //next/pjrtx/vertical_slice:execution_test --test_output=errors`
passes.

Links:
- `pjrtx/runtime/runtime.zig`
- `next/pjrtx/vertical_slice/execution_test.zig`

## 2026-05-18 - Backend Capability Legality Gate Started

Status: accepted

Context:
The compiler could plan the V0 workload, but backend support was still mostly
implicit in the planner and backend binding names. Unsupported work needed an
explicit no-fallback gate before lowering and runtime artifacts.

Decision:
Add backend capability sets under `//next/pjrtx/backend` with feature and dtype
support for `npu_v0` and `metal_v0`. Add compiler target-legality
verification that checks every executable graph instruction against the
selected backend capability set before cost/lowering/schedule planning.

Reasoning:
Backend capability is a contract, not a comment. A graph may be mathematically
valid and still illegal for a target because the backend lacks a dtype or
operation implementation.

Consequences:
Unsupported dtypes now fail at `target_legality` with diagnostics such as the
backend kind, feature, dtype, and graph instruction ID. The V0 workload still
passes because `f32` dot/broadcast/add/tanh are declared supported.

Verification:
`bazel test //next/pjrtx/backend:unit_tests //next/pjrtx/compiler:unit_tests --test_output=errors`
passes.

Links:
- `pjrtx/backend/backend.zig`
- `pjrtx/compiler/compiler.zig`

## 2026-05-18 - Compiler Schedule Verification Hardened

Status: accepted

Context:
Runtime stream planning rejected invalid dependency ordering, but compiler
executable creation still relied mostly on structural trace validation and
backend binding checks.

Decision:
Add compiler-owned schedule verification before backend binding verification
in executable creation. The verifier rejects command ID order mismatches,
future/self dependencies, and backend execute commands that do not depend on an
earlier transfer or wait command.

Reasoning:
Invalid schedules should fail before they become executable-shaped. Runtime
will keep checking dependencies, but the compiler owns the schedule and should
reject bad command graphs first.

Consequences:
Backend binding tests now use valid H2D -> backend schedules. Separate negative
tests cover missing backend dependencies and future dependencies.

Verification:
`bazel test //next/pjrtx/compiler:unit_tests --test_output=errors` passes.

Links:
- `pjrtx/compiler/compiler.zig`
- `pjrtx/compiler/BUILD.bazel`

## 2026-05-18 - Planner Uses Backend Capabilities

Status: accepted

Context:
Backend capability checks existed, but the cost ledger and backend binding
still had planner-local execution-unit defaults. That risked the legality gate
and emitted report disagreeing.

Decision:
Make compiler planning derive per-instruction expected execution units from
the selected backend capability set. A combined backend binding now reports a
single expected unit only when all executable instructions share one; otherwise
it records `null` rather than pretending mixed-unit work is single-unit.

Reasoning:
The trace report should preserve what the backend declared. Capability-derived
planning keeps target legality, cost estimates, backend binding, and runtime
profiling aligned.

Consequences:
The NPU V0 report now records the dot on the matrix unit and elementwise
work on the vector unit, while the single backend execute binding records no
single expected unit because the command spans both.

Verification:
`bazel test //next/pjrtx/backend:unit_tests //next/pjrtx/compiler:unit_tests --test_output=errors`
passes.

Links:
- `pjrtx/backend/backend.zig`
- `pjrtx/compiler/compiler.zig`

## 2026-05-18 - Runtime Predicted Versus Profiled Summary Started

Status: accepted

Context:
Profile events, allocation plans, and stream plans existed, but predicted
compiler costs and runtime synthetic observations were still inspected through
separate artifacts.

Decision:
Add a runtime execution summary writer that joins schedule commands, stream
steps, allocation peak bytes, predicted cost metrics, and profile events. Also
make synthetic profiling validate dependency ordering through the same runtime
dependency checks as stream planning.

Reasoning:
Performance explainability requires comparison, not just collection. A command
should show what the compiler predicted and what the runtime observed, even
when the observation is synthetic in V0.

Consequences:
The vertical execution test now asserts predicted and observed bytes/ops for
the backend command. Future hardware profile events can reuse this summary
shape when real counters are available.

Verification:
`bazel test //next/pjrtx/runtime:unit_tests //next/pjrtx/vertical_slice:execution_test --test_output=errors`
passes.

Links:
- `pjrtx/runtime/runtime.zig`
- `next/pjrtx/vertical_slice/execution_test.zig`

## 2026-05-18 - Runtime Stream Plan Started

Status: accepted

Context:
The runtime allocation plan exposed buffers and lifetimes, but the runtime
still lacked a stream/event artifact that made schedule dependencies explicit
before synthetic profiling.

Decision:
Add a runtime stream plan with one step per schedule command, deterministic
start/done event IDs, dependency wait event IDs, and a stable summary writer.
The planner validates that command dependencies point to earlier commands.

Reasoning:
Stream and event structure is where asynchronous execution bugs usually hide.
Even before a real stream executor exists, the runtime should make dependency
ordering visible and reject future/self dependencies instead of relying on
implicit linear execution.

Consequences:
The vertical execution test now checks stream steps and event wait summaries.
Runtime unit tests reject invalid dependency ordering. Real stream submission,
priorities, overlap, and asynchronous error propagation remain future work.

Verification:
`bazel test //next/pjrtx/runtime:unit_tests //next/pjrtx/vertical_slice:execution_test --test_output=errors`
passes.

Links:
- `pjrtx/runtime/runtime.zig`
- `next/pjrtx/vertical_slice/execution_test.zig`

## 2026-05-18 - Synthetic Runtime Profile Path Started

Status: accepted

Context:
The compiler could produce a verified V0 trace report with schedule commands
and backend binding, but runtime still had no command-linked profile path.

Decision:
Add a synthetic runtime profiling function that validates the trace report,
requires exactly one backend binding for each backend execute command, and
emits deterministic profile events for H2D, backend execute, and D2H commands.

Reasoning:
This gives the architecture a runtime-shaped feedback edge without pretending
to execute unsupported work and without introducing a reference fallback. It
also makes profile event linkage testable before real device streams and
allocators exist.

Consequences:
The vertical execution test now crosses StableHLO ingest, graph import,
verification, target selection, trace planning, backend binding, and runtime
profile event production. Real allocator/stream execution remains future work.

Verification:
`bazel test //next/pjrtx/runtime:unit_tests //next/pjrtx/vertical_slice:execution_test --test_output=errors`
passes.

Links:
- `pjrtx/runtime/runtime.zig`
- `next/pjrtx/vertical_slice/execution_test.zig`

## 2026-05-18 - V0 Trace Planner Started

Status: accepted

Context:
The graph layer was typed and verified, but the vertical slice still needed a
compiler-owned bridge into performance records: cost ledger, lowering,
schedule, backend binding, and explain output.

Decision:
Add `planV0TraceReport`, which turns a verified graph and selected target into
a trace report with per-instruction cost estimates, lowering records, a simple
H2D/backend/D2H schedule, exactly one backend binding for the execute command,
and an explain record.

Reasoning:
The first planner should be conservative and legible. It does not try to be a
global optimizer yet; it creates the minimum complete path that proves no
runtime fallback and makes FLOPs/bytes/lowering choices visible.

Consequences:
The vertical slice can now validate a full compile-shaped report before any
runtime execution exists. Future work can refine fusion, allocation, and kernel
generation without losing the report edges.

Verification:
`bazel test //next/pjrtx/compiler:unit_tests --test_output=errors` passes.
`bazel test //next/pjrtx/vertical_slice:import_tests --test_output=errors`
passes.

Links:
- `pjrtx/compiler/compiler.zig`
- `next/pjrtx/vertical_slice/import_test.zig`

## 2026-05-18 - Compiler Middle Reroute Accepted

Status: accepted

Context:
The new implementation successfully built trace records, target legality,
runtime allocation/profile summaries, and Metal/MLS backend executable
descriptors. That progress also exposed drift from the vision: the workplan was
starting to deepen backend/runtime layers before the compiler owned MLIR
lowering, fusion, tiling, memory-space assignment, and collectives. That risks
copying the common plugin pattern where XLA remains the real compiler and a
project only implements the StreamExecutor-style runtime/backend layer.

Decision:
Reroute the next work to the compiler middle. Add explicit MLIR pass-pipeline
reports, fusion candidate and decision records, layout/tile/memory-space plans,
and collective lowering records before continuing deeper Metal command-buffer
execution or PJRT/JAX smoke work.

Reasoning:
PjRTx exists to make performance and correctness explainable from PJRT-level
programs down to target hardware units. That cannot be recovered after the
fact from backend submission APIs. The compiler must preserve the decisions
that create FLOPs, bytes, memory movement, collectives, and kernels.

Consequences:
Phase 6 is split into compiler-middle subphases. Phase 9 Metal execution is
paused after report-only MLS graph descriptors until those subphases exist.
The compile pipeline now includes MLIR lowering, fusion, layout/tile/memory,
and collective stages before cost, schedule, and backend binding.

Follow-up:
Implement a `lowering_tests` vertical target with pass/fusion/tile/memory and
unsupported-collective goldens. Then make the planner consume those artifacts
instead of building the schedule directly from raw graph instructions.

Links:
- `next/docs/specs/implementation_workplan_v0.md`
- `next/docs/specs/compile_pipeline_v0.md`
- `next/docs/specs/harness_v0.md`
- `next/docs/pjrtx_architecture_vision.md`

## 2026-05-18 - Compiler Middle Artifacts Started

Status: accepted

Context:
After the reroute, the next implementation step was to make the compiler
middle visible in code before continuing backend execution. Shardy also needed
to remain part of the pass-pipeline contract, not an afterthought.

Decision:
Add first compiler-owned artifacts for MLIR pass-pipeline reporting, fusion
planning, layout/tile/memory placement, and collective lowering. The pass
report records Shardy metadata presence, whether Shardy propagation was
requested, and whether the metadata is preserved. The new vertical
`lowering_tests` target checks the pass report, accepted/rejected fusion,
NPU placement, and unsupported collective failure.

Reasoning:
These records make the next planner refactor possible: cost, schedule, and
backend binding should consume explicit compiler-middle decisions instead of
building directly from raw graph instructions.

Consequences:
Phase 6A through 6D now have initial report-only code. The artifacts are not
yet joined into `TraceReport`; that join is the next important refactor.

Verification:
`bazel test //next/pjrtx/compiler:unit_tests //next/pjrtx/vertical_slice:lowering_tests --test_output=errors`
passes.

Links:
- `pjrtx/compiler/compiler.zig`
- `next/pjrtx/vertical_slice/lowering_test.zig`

## 2026-05-18 - First Operational Graph Normalization Pass

Status: accepted

Context:
The pass catalog made the compiler pipeline visible, but the first rows were
still mostly records and conservative gates. The next drift risk was letting
algebraic rewrites hide in later fusion or lowering helpers.

Decision:
Add `GraphRewriteRecord` to the core trace schema and implement
`broadcast_simplify` as the first operational graph normalization pass. The
pass runs after typed graph import and verification. It removes only identity
broadcasts whose input and output tensor types are identical and whose
dimension map is exactly identity. Non-identity broadcasts emit rejected records
so later performance work can explain why the broadcast stayed.

Reasoning:
This is a small but real correctness-preserving rewrite. It proves that graph
normalization can mutate the compiler-owned graph while preserving a visible
audit trail before fusion, placement, cost, lowering, and backend binding.

Consequences:
The compile orchestrator now normalizes the graph before target selection and
trace planning. Reports can include graph rewrite records. Future algebraic
passes should follow the same pattern: strict legality, owned graph output,
deterministic records, and no silent semantic relaxation.

Verification:
`bazel test //next/pjrtx/... --test_output=errors` passes.

Links:
- `pjrtx/core/core.zig`
- `pjrtx/compiler/compiler.zig`
- `next/pjrtx/fixtures/identity_broadcast.mlir`
- `next/pjrtx/vertical_slice/lowering_test.zig`
- `next/docs/specs/trace_schema_v0.md`

## 2026-05-18 - Identity Reshape Transpose Fold

Status: accepted

Context:
`broadcast_simplify` proved the graph normalization path, but the next planned
pass required the typed graph to understand StableHLO reshape and transpose.
Adding the ops without lowering support would be dangerous unless non-identity
cases stayed on the no-fallback rejection path.

Decision:
Add typed graph support for `stablehlo.reshape` and `stablehlo.transpose`.
Extend `reshape_transpose_fold` to remove only identity reshapes and identity
transposes after graph import. Keep non-identity reshape/transpose unsupported
by target/backend capabilities so they fail target legality before cost,
lowering, scheduling, allocation, or backend binding.

Reasoning:
Identity shape/layout rewrites are mathematically exact and useful for cleaning
frontend artifacts. Non-identity transforms affect layout, memory traffic, and
allocation, so accepting them before those layers exist would hide performance
and correctness decisions.

Consequences:
The graph model is slightly wider, but executable V0 behavior remains
conservative. Future layout-aware lowering can reuse these graph op kinds and
replace the target-legality rejection with explicit layout, memory traffic, and
kernel-generation records.

Verification:
`bazel test //next/pjrtx/... --test_output=errors` passes.

Links:
- `pjrtx/core/core.zig`
- `pjrtx/compiler/compiler.zig`
- `pjrtx/backend/backend.zig`
- `next/pjrtx/fixtures/identity_reshape_transpose.mlir`
- `next/pjrtx/fixtures/non_identity_transpose.mlir`
- `next/pjrtx/vertical_slice/lowering_test.zig`

## 2026-05-18 - Explicit Compiler Pass Catalog

Status: accepted

Context:
The architecture docs now require an XLA-like pass pipeline, but the code still
expressed pass structure mostly through function names and a short MLIR pass
report. That made future fusion, tiling, collective, allocation, and backend
codegen work too easy to hide in ad hoc planner code.

Decision:
Add a typed V0 compiler pass catalog in `//next/pjrtx/compiler`. Each catalog entry
records the stable pass name, stage, input and output artifact kinds, required
target facts, preserved invariants, invalidated analyses, emitted record names,
and pass effect. Make the current MLIR pass report use catalog names for
`stablehlo_parse`, `mlir_verify`, `collective_graph_payload_import`,
`mlir_canonicalize_cse`, and `shardy_metadata_propagation_report`.

Reasoning:
PjRTx should have many passes, but the pass list itself must be inspectable and
testable. A typed catalog gives future lowering work a place to land without
building a callback-heavy framework too early.

Consequences:
The pass catalog is now part of the tested compiler API. V0 still uses
straight-line orchestration, but future passes must fit into the catalog or
extend it intentionally. Collective handling now reports the stable
`collective_graph_payload_import` name before algorithm selection.

Verification:
`bazel test //next/pjrtx/... --test_output=errors` passes.

Links:
- `pjrtx/compiler/compiler.zig`
- `next/pjrtx/vertical_slice/lowering_test.zig`
- `next/docs/specs/compiler_pass_pipeline_v0.md`

## 2026-05-18 - Runtime Allocation Consumes Lowering Regions

Status: accepted

Context:
After backend executable planning started consuming fused lowering regions, the
runtime allocation planner still materialized every graph value as a device
buffer. That meant the broadcast and add intermediates inside the fused
elementwise region still consumed allocation/lifetime space even though the
compiler had selected fusion specifically to avoid that pressure.

Decision:
Make runtime allocation planning consume schedule lowering records. A
single-instruction lowering reads and writes that instruction's operands. A
fused lowering reads only values produced outside the region and writes only
values consumed outside the region. Fused-internal values remain visible in the
trace for correctness and provenance but do not receive standalone device
buffers.

Reasoning:
Performance explainability must reach allocation, not stop at backend naming.
If fusion is accepted, memory pressure and buffer lifetime reports should show
the effect immediately.

Consequences:
The V0 execution golden drops from 11 to 9 allocations and from 18 to 14
command-buffer uses. Peak device bytes for `tanh(dot(x, w) + b)` drops from 188
to 140 while source-level predicted bytes and FLOPs remain unchanged.

Verification:
`bazel test //next/pjrtx/runtime:unit_tests //next/pjrtx/vertical_slice:execution_test --test_output=errors`
passes.

Links:
- `pjrtx/runtime/runtime.zig`
- `next/pjrtx/vertical_slice/execution_test.zig`
- `next/pjrtx/vertical_slice/testdata/v0_execution_report.txt`

## 2026-05-18 - Compiler Boundary No Longer Imports Backend

Status: accepted

Context:
The vision and harness boundary rules say `//next/pjrtx/compiler` must not depend on
`//next/pjrtx/backend`, but the V0 compiler was importing backend capability helpers
for target legality, cost unit assignment, and backend binding operation names.
That let an implementation package become part of compiler decision making.

Decision:
Remove the `//next/pjrtx/backend` dependency from `//next/pjrtx/compiler`. The compiler
now owns a small V0 target-capability view for legality and binding planning.
The backend package still independently validates capabilities when it expands
a verified binding into executable calls.

Reasoning:
Compiler decisions need to be explainable without depending on backend
submission code. Backend revalidation is still useful as a pre-runtime guard,
but it should not be the source of compiler legality.

Consequences:
The Bazel compiler target now depends on `//next/pjrtx/core` and MLIR/StableHLO/
Shardy C API targets, not `//next/pjrtx/backend`. This preserves the intended
layering while keeping V0 capability data explicit.

Verification:
`bazel test //next/pjrtx/compiler:unit_tests //next/pjrtx/vertical_slice:lowering_tests //next/pjrtx/vertical_slice:backend_binding_test --test_output=errors`
passes.

Links:
- `pjrtx/compiler/BUILD.bazel`
- `pjrtx/compiler/compiler.zig`

## 2026-05-18 - Profiles Join To Lowering Regions

Status: accepted

Context:
Runtime profile events existed only at the coarse schedule-command level.
After compiler lowerings, backend calls, kernel graph nodes, and allocation all
became region-aware, profiling was the remaining coarse layer. That made it
impossible to match observed or synthetic backend work to the matmul kernel
versus the fused elementwise kernel.

Decision:
Keep command-level synthetic profile events, and add deterministic
backend-lowering profile events for each lowering record on a backend execute
command. Add a runtime lowering-profile summary and a backend call-profile
summary. The backend summary joins exact graph-instruction provenance back to
Metal/MLS executable calls.

Reasoning:
Performance explainability needs the same unit of work across lowering,
backend planning, allocation, and profiling. The command event remains useful
for stream and schedule accounting, while lowering-region events let kernel
and fusion decisions be inspected independently.

Consequences:
The V0 profile stream for `tanh(dot(x, w) + b)` now has five synthetic events:
H2D, coarse backend execute, matmul lowering, fused elementwise lowering, and
D2H. The execution golden includes a `lowering profiles` section, and the Metal
bridge test verifies profile joins for `metal_mls_matmul_kernel` and
`metal_mls_elementwise_fusion_kernel`.

Verification:
`bazel test //next/pjrtx/runtime:unit_tests //next/pjrtx/backend:unit_tests //next/pjrtx/vertical_slice:metal_bridge_test //next/pjrtx/vertical_slice:execution_test --test_output=errors`
passes.

Links:
- `pjrtx/runtime/runtime.zig`
- `pjrtx/backend/backend.zig`
- `next/pjrtx/vertical_slice/execution_test.zig`
- `next/pjrtx/vertical_slice/metal_bridge_test.zig`
- `next/pjrtx/vertical_slice/testdata/v0_execution_report.txt`

## 2026-05-18 - Core Profile Summary Preserves Join Keys

Status: accepted

Context:
The profile harness spec required stable reports to preserve event kind,
command ID, graph instruction IDs, bytes, logical ops, status, and forced
synchronization. The core trace summary still only printed event kind and a
redacted duration, so the richer region-aware profile events were not visible
in the central trace report.

Decision:
Expand `writeTraceReportSummary` profile rows to include command ID, graph
instruction IDs, bytes, logical ops, status, and forced synchronization while
continuing to redact `start_ns` and `duration_ns`.

Reasoning:
Raw timings are unstable in golden tests, but profile join keys are not. The
stable trace must expose enough information to connect profile events back to
schedule commands, lowering regions, backend calls, and explain records.

Consequences:
The tiny trace golden now shows profile provenance and counters without leaking
timing values. This makes the core summary consistent with the execution and
backend call profile summaries.

Verification:
`bazel test //next/pjrtx/core:unit_tests //next/pjrtx/vertical_slice:report_test --test_output=errors`
passes.

Links:
- `pjrtx/core/core.zig`
- `next/pjrtx/vertical_slice/testdata/tiny_trace_report.txt`

## 2026-05-18 - Explain Records Join To Profiles

Status: accepted

Context:
Lowering-region profile events existed, but explain records still mostly
described the coarse backend binding and did not point back to profile events.
That meant the trace could show what was measured and separately why lowering
happened, but not join those two facts directly.

Decision:
Emit compiler explain records for each lowering record. When runtime creates a
profiled report, copy explain records and attach matching `profile_event_ids`
by lowering provenance, backend binding command/instruction IDs, schedule
command, or graph instruction membership. Expand the core summary to print
explain subjects and profile IDs.

Reasoning:
PjRTx should answer both "why did this lowering happen?" and "which event
measured it?" without hidden side channels. Profile links belong in the trace,
while raw timing remains redacted in stable summaries.

Consequences:
The V0 compile trace now has lowering-level explain records. The profiled trace
links matmul lowering to `profile.2`, fused elementwise lowering to
`profile.3`, and the coarse backend binding explanation to `profile.1`.

Verification:
`bazel test //next/pjrtx/core:unit_tests //next/pjrtx/vertical_slice:report_test //next/pjrtx/vertical_slice:lowering_tests //next/pjrtx/vertical_slice:execution_test --test_output=errors`
passes.

Links:
- `pjrtx/compiler/compiler.zig`
- `pjrtx/runtime/runtime.zig`
- `pjrtx/core/core.zig`
- `next/pjrtx/vertical_slice/lowering_test.zig`
- `next/pjrtx/vertical_slice/execution_test.zig`

## 2026-05-18 - Backend Call Performance Ledger Started

Status: accepted

Context:
Backend call profiles could join predicted and observed bytes/ops to exact
Metal or NPU calls, but they did not yet expose the hardware performance
expectation for that call. The vision requires each generated or selected
kernel region to explain expected unit binding and ideal time from hardware
specs.

Decision:
Extend backend call profile summaries with expected unit ID and
`ideal_compute_ps`. The calculation sums each cost ledger entry in the call
against the selected target execution unit's dtype/op-class rate. Unknown
target rates produce `0`, which keeps Metal honest until rates are modeled.

Reasoning:
Per-call ideal time is the smallest useful kernel-performance ledger for V0.
It makes the lowering-to-kernel path comparable to hardware capability without
pretending we have real counters or full roofline modeling yet.

Consequences:
Metal V0 call profiles now show `ideal_compute_ps=0` because its rates are
unknown. The TRN2-like NPU path shows `npu_matmul` at `2ps` and fused
elementwise at `16ps`, with the fused value preserving per-cost-entry rounding
for broadcast, add, and tanh.

Verification:
`bazel test //next/pjrtx/backend:unit_tests //next/pjrtx/vertical_slice:metal_bridge_test //next/pjrtx/vertical_slice:execution_test --test_output=errors`
passes.

Links:
- `pjrtx/backend/backend.zig`
- `next/pjrtx/vertical_slice/metal_bridge_test.zig`
- `next/pjrtx/vertical_slice/execution_test.zig`

## 2026-05-18 - Backend Call Roofline Memory Slice Started

Status: accepted

Context:
The backend call ledger exposed ideal compute time, but memory bandwidth was
still hidden. That made V0 too FLOP-centric and weakened the NPU story, where
the same lowering can be limited by HBM, SRAM movement, tensor units, vector
units, or collectives rather than arithmetic alone.

Decision:
Extend backend call profile summaries with `ideal_memory_ps` and
`limiting`. V0 uses the target's primary device memory space, preferring HBM
over unified memory, and computes an ideal lower bound from predicted
per-call bytes and target bandwidth. Unknown bandwidth keeps the result at
`0` and marks the limiting resource as `unknown`.

Reasoning:
This gives every generated or selected backend call a compact roofline row:
lowering IDs, kernel operation, expected unit, predicted/observed work,
ideal compute time, ideal memory time, limiting resource, and profile event.
It keeps the report explainable from PJRT down toward kernel generation without
pretending V0 has full cache, SRAM, occupancy, or counter modeling yet.

Consequences:
Metal/MLS remains honest with unknown ideal compute and memory timing until
the Metal target describes rates and bandwidth. The TRN2-like NPU path now
shows the V0 matmul and fused elementwise calls as memory-limited because their
ideal HBM time exceeds their ideal compute time.

Verification:
`bazel test //next/pjrtx/backend:unit_tests //next/pjrtx/vertical_slice:metal_bridge_test //next/pjrtx/vertical_slice:execution_test --test_output=errors`
passes.

Links:
- `pjrtx/backend/backend.zig`
- `next/pjrtx/vertical_slice/metal_bridge_test.zig`
- `next/pjrtx/vertical_slice/execution_test.zig`

## 2026-05-18 - Backend Call Memory Spaces Joined To Placement

Status: accepted

Context:
The first backend-call roofline row used one primary device memory bandwidth.
That was still too coarse for NPUs because a lowering can have HBM boundary
traffic and local tile/scratchpad traffic at the same time. During the slice,
the test also exposed that synthetic profiled reports were dropping
compiler-middle records, which meant placement facts disappeared exactly where
profile analysis needed them.

Decision:
Keep the public schema stable and derive per-call memory rows inside the
backend summary. Global device-memory bytes come from backend-call external
inputs and outputs. Local SRAM/scratchpad bytes come from cost entries whose
instructions have placement records using that tile memory. Preserve
MLIR/Shardy pass, fusion, placement, and collective records when creating a
profiled report.

Reasoning:
This gives V0 a useful memory hierarchy view without adding a premature public
traffic schema. The fused elementwise lowering now shows the important
distinction: HBM sees only the region boundary, while local SRAM accounts for
the internal per-op traffic.

Consequences:
Backend call profile rows now include a `memory=` list such as
`memory.1:60B/60ps,memory.2:156B/8ps`. The call-level `ideal_memory_ps` is the
maximum ideal time across memory spaces, so limiting-resource selection is no
longer based on a single primary bandwidth.

Verification:
`bazel test //next/pjrtx/backend:unit_tests //next/pjrtx/runtime:unit_tests //next/pjrtx/vertical_slice:metal_bridge_test //next/pjrtx/vertical_slice:execution_test --test_output=errors`
passes.

Links:
- `pjrtx/backend/backend.zig`
- `pjrtx/runtime/runtime.zig`
- `next/pjrtx/vertical_slice/execution_test.zig`
- `next/pjrtx/vertical_slice/metal_bridge_test.zig`

## 2026-05-18 - Memory Traffic Becomes A Trace Record

Status: accepted

Context:
The backend could now print per-memory-space traffic, but the facts were still
derived locally from placement and cost records. That made the report useful
for humans but weak as an architecture contract for kernel generation,
autotuning, and future profiling consumers.

Decision:
Add `MemoryTrafficRecord` to the core trace schema. The compiler now emits
traffic records per lowering for device-boundary memory and tile memory. The
backend call profile summary consumes those records directly when computing
per-memory-space bytes, ideal memory time, and limiting resource. Profiled
reports preserve memory traffic records alongside the rest of the compiler
middle.

Reasoning:
Memory traffic is not a backend formatting concern; it is one of the central
facts needed to explain why a lowering should be fast or slow on NPUs and TPUs.
Making it first-class keeps the path from PJRT/StableHLO to kernel generation
inspectable without rebuilding hidden state in every layer.

Consequences:
The V0 `tanh(dot(x, w) + b)` trace now has four traffic records: matmul device
boundary, matmul tile memory, fused elementwise device boundary, and fused
elementwise tile memory. Backend summaries still print the same compact
`memory=` rows, but those rows now come from trace records.

Verification:
`bazel test //next/pjrtx/core:unit_tests //next/pjrtx/compiler:unit_tests //next/pjrtx/backend:unit_tests //next/pjrtx/runtime:unit_tests //next/pjrtx/vertical_slice:report_test //next/pjrtx/vertical_slice:lowering_tests //next/pjrtx/vertical_slice:metal_bridge_test //next/pjrtx/vertical_slice:execution_test --test_output=errors`
passes.

Links:
- `pjrtx/core/core.zig`
- `pjrtx/compiler/compiler.zig`
- `pjrtx/backend/backend.zig`
- `pjrtx/runtime/runtime.zig`

## 2026-05-18 - Hardware Summary Prints Memory Traffic

Status: accepted

Context:
`MemoryTrafficRecord` was first-class and backend call summaries consumed it,
but the runtime hardware utilization report still jumped from allocation
pressure to transfer edges. That hid the lowering-level HBM versus local SRAM
story unless a backend executable summary was also printed.

Decision:
Add a `memory traffic` section to the hardware utilization summary. Each row
prints traffic ID, lowering ID, memory space, traffic kind, read/write/total
bytes, target memory bandwidth, and ideal memory time.

Reasoning:
The hardware report is the right common place to audit memory hierarchy
pressure across backends. Backend rows can stay compact, while hardware
summaries show the trace-native facts that should feed future scheduling,
tiling, kernel generation, and profiling analysis.

Consequences:
The V0 execution golden now shows matmul and fused elementwise traffic at both
HBM boundary and local SRAM tile levels. For the fused elementwise region, HBM
traffic is 60 bytes while tile-memory traffic is 156 bytes, making fusion's
boundary reduction visible without backend-private derivation.

Verification:
`bazel test //next/pjrtx/runtime:unit_tests //next/pjrtx/vertical_slice:execution_test --test_output=errors`
passes.

Links:
- `pjrtx/runtime/runtime.zig`
- `next/pjrtx/vertical_slice/testdata/v0_execution_report.txt`
- `next/pjrtx/vertical_slice/execution_test.zig`

## 2026-05-18 - Hardware Summary Adds Lowering Roofline

Status: accepted

Context:
The hardware summary printed memory traffic and execution-unit rates, but a
reviewer still had to mentally join cost entries and traffic rows to determine
whether a lowering was compute-limited or memory-limited. Backend call summaries
had this compact row, but the hardware report needed the same backend-neutral
view.

Decision:
Add a `lowering roofline` section to the hardware utilization summary. Each row
prints lowering ID, decision, predicted ops/bytes, ideal compute time, ideal
memory time, and limiting resource. Ideal compute time comes from target
execution-unit dtype/op-class rates. Ideal memory time is the maximum ideal
time across memory traffic records for that lowering.

Reasoning:
Lowering is the right architecture level for composable performance reasoning:
it sits after fusion/placement and before backend-specific kernel graph details.
This lets us ask whether the compiler made a sensible lowering decision before
debugging generated kernels.

Consequences:
The V0 execution golden now marks both matmul and fused elementwise lowerings
as memory-limited. The numbers are derived from trace facts, so backend
summaries and hardware summaries agree without sharing backend-private logic.

Verification:
`bazel test //next/pjrtx/runtime:unit_tests //next/pjrtx/vertical_slice:execution_test --test_output=errors`
passes.

Links:
- `pjrtx/runtime/runtime.zig`
- `next/pjrtx/vertical_slice/testdata/v0_execution_report.txt`
- `next/pjrtx/vertical_slice/execution_test.zig`

## 2026-05-18 - Lowering Roofline Joins Profile Events

Status: accepted

Context:
The hardware summary could classify each lowering as compute- or
memory-limited, but the row only showed predicted work and ideal lower bounds.
The measured synthetic profile data existed in nearby lowering-profile rows,
forcing readers to cross-reference manually.

Decision:
Join lowering roofline rows to lowering-region profile events. Each row now
prints predicted ops/bytes, observed ops/bytes, ideal compute time, ideal memory
time, limiting resource, and `event=profile.N`.

Reasoning:
This keeps the performance explanation stable and backend-independent while
making prediction-versus-measurement visible at the lowering boundary. Raw
durations remain out of the golden row; the stable profile ID carries the join
to detailed timing/counter data.

Consequences:
The V0 hardware summary directly connects matmul lowering to `profile.2` and
fused elementwise lowering to `profile.3`. Backend call summaries and hardware
summaries now agree on profile joins without sharing backend-private state.

Verification:
`bazel test //next/pjrtx/runtime:unit_tests //next/pjrtx/vertical_slice:execution_test --test_output=errors`
passes.

Links:
- `pjrtx/runtime/runtime.zig`
- `next/pjrtx/vertical_slice/testdata/v0_execution_report.txt`
- `next/pjrtx/vertical_slice/execution_test.zig`

## 2026-05-18 - Compiler Pass Pipeline Spec Added

Status: accepted

Context:
XLA has many HLO passes because high-performance compilation needs many
separate decisions: canonicalization, legality, fusion, tiling, collectives,
layout, scheduling, buffer assignment, and backend lowering. PjRTx needs that
kind of structure, but without creating an opaque pass pile where decisions are
hard to explain.

Decision:
Add `next/docs/specs/compiler_pass_pipeline_v0.md`. The spec defines pass families,
pass contracts, required invariants, V0 pass ordering, next passes to add, pass
manager shape, test requirements, and anti-patterns. It explicitly treats XLA
as a reference for hard problems and pass families, not as a hidden compiler
behind PjRTx.

Reasoning:
The project was already growing compiler-middle records. A dedicated pass spec
keeps that growth intentional: every pass that changes correctness,
performance, memory, collectives, scheduling, or backend binding must emit
records that join to trace, target, roofline, and profile data.

Consequences:
`compile_pipeline_v0.md` remains the no-fallback stage gate document.
`compiler_pass_pipeline_v0.md` now owns the detailed pass taxonomy and
contracts. Future implementation work should add passes behind those contracts
instead of pushing hidden decisions into backend code.

Verification:
Documentation-only change; checked links and references with ripgrep.

Links:
- `next/docs/specs/compiler_pass_pipeline_v0.md`
- `next/docs/specs/README.md`
- `next/docs/specs/compile_pipeline_v0.md`
- `next/docs/specs/implementation_workplan_v0.md`
- `next/docs/pjrtx_architecture_vision.md`

## 2026-05-18 - Backend Executable Consumes Fused Lowerings

Status: accepted

Context:
The trace planner now emitted region-level lowerings, but the Metal backend
bridge still expanded every graph instruction into its own backend call. That
kept the call surface operationally close to the old per-op model even though
the compiler had already selected an elementwise fusion region.

Decision:
Make backend executable planning iterate schedule lowering records instead of
raw backend-binding instruction lists. A non-fused lowering becomes one backend
call. An `elementwise_fusion` lowering becomes one backend call with all fused
instruction IDs, explicit external inputs, one external output, and the backend
operation `metal_mls_elementwise_fusion_kernel`. The Metal kernel graph now has
one matmul node feeding one fused elementwise node.

Reasoning:
The backend bridge must preserve compiler decisions all the way to codegen.
Otherwise fusion, tiling, placement, and later Shardy-aware collective
decisions become documentation rather than executable architecture.

Consequences:
For `tanh(dot(x, w) + b)`, Metal V0 now exposes two backend calls and one
kernel-graph edge: matmul produces the value consumed by the fused
broadcast/add/tanh node. The summary includes both the primary instruction ID
and full fused instruction provenance.

Verification:
`bazel test //next/pjrtx/backend:unit_tests //next/pjrtx/vertical_slice:metal_bridge_test --test_output=errors`
passes.

Links:
- `pjrtx/backend/backend.zig`
- `next/pjrtx/vertical_slice/metal_bridge_test.zig`

## 2026-05-18 - Lowering Consumes Fusion And Placement

Status: accepted

Context:
Fusion and placement records were present in the final trace, but cost and
lowering still behaved mostly as if every StableHLO instruction lowered
independently. That made fusion explainable but not yet operational.

Decision:
Keep per-instruction cost entries, but emit lowering records from
compiler-middle regions. The V0 planner now lowers matmul as one backend
kernel-graph boundary and lowers the broadcast/add/tanh chain as one accepted
elementwise-fusion region. Lowering requires placement records for every
instruction in the region.

Reasoning:
Cost should remain granular enough to explain FLOPs and bytes per source
operation, while lowering should represent the region that backend codegen or
kernel graph planning will actually consume.

Consequences:
The V0 trace now has two lowering regions for `tanh(dot(x, w) + b)`: matmul and
fused elementwise. At the time of this decision, the backend binding still
expanded graph instructions into per-backend calls; the later backend bridge
decision superseded that consequence and now consumes the fused lowering region
directly.

Verification:
`bazel test //next/pjrtx/compiler:unit_tests //next/pjrtx/vertical_slice:lowering_tests //next/pjrtx/vertical_slice:import_tests //next/pjrtx/vertical_slice:execution_test --test_output=errors`
passes.

Links:
- `pjrtx/compiler/compiler.zig`
- `next/pjrtx/vertical_slice/lowering_test.zig`
- `next/pjrtx/fixtures/shardy_tiny_mesh.mlir`
- `next/pjrtx/fixtures/unsupported_all_reduce.mlir`

## 2026-05-18 - Compiler Middle Records Joined Into Trace

Status: accepted

Context:
The first compiler-middle artifacts existed, but the executable trace still
mostly started at cost/lowering/schedule. That left the planner able to bypass
fusion, placement, and collective decisions even though those decisions were now
defined.

Decision:
Move compiler-middle report records into the core `TraceReport`: MLIR pass
records, fusion groups, placement records, and collective plan records. Update
trace validation and summary writing for those sections. Make
`planV0TraceReport` build fusion, placement, and collective records before cost
ledger, lowering, schedule, and backend binding. Keep MLIR pass records empty
in the graph-only planner until compile orchestration passes the MLIR/input
artifact through the report path.

Reasoning:
Schedule and backend binding should be downstream from compiler decisions, not
peers that can ignore them. The trace now has a visible place for those
decisions, which makes the next cost/schedule refactor smaller and safer.

Consequences:
Golden trace summaries now include empty compiler-middle sections even for
hand-built reports. Compiled V0 reports carry fusion, placement, and collective
records. MLIR pass records remain covered by `lowering_tests` and are ready to
join once the compile orchestrator owns all stage artifacts.

Verification:
`bazel test //next/pjrtx/... --test_output=errors` passes.

Links:
- `pjrtx/core/core.zig`
- `pjrtx/compiler/compiler.zig`
- `next/pjrtx/vertical_slice/testdata/tiny_trace_report.txt`

## 2026-05-18 - Compile Orchestrator Carries Shardy Pass Records

Status: accepted

Context:
MLIR pass records were part of the core trace schema, and standalone lowering
tests proved Shardy metadata preservation. But the graph-only planner could not
honestly attach MLIR pass records because it starts after MLIR ingest.

Decision:
Add `compileV0FromReader`, an owned V0 compile orchestrator that runs input
setup, StableHLO ingest, Shardy-aware MLIR pass reporting, graph import, graph
verification, target selection, and trace planning. The returned compiled
artifact owns the graph and planned trace, and its final report includes MLIR
pass records.

Reasoning:
Pass provenance must flow through the real compile path, not just a side
artifact. Keeping graph-only planning separate preserves honest stage
boundaries while giving the end-to-end compile path complete traceability.

Consequences:
The final compiled V0 trace now contains Shardy-aware MLIR pass records,
fusion groups, placement records, collective records, cost, lowering, schedule,
backend binding, and explain output. The next step is to make cost and schedule
consume placement/fusion more deeply instead of only carrying them beside the
current baseline schedule.

Verification:
`bazel test //next/pjrtx/compiler:unit_tests //next/pjrtx/vertical_slice:lowering_tests --test_output=errors`
passes.

Links:
- `pjrtx/compiler/compiler.zig`
- `next/pjrtx/vertical_slice/lowering_test.zig`

## 2026-05-18 - XLA Coverage Map Becomes A PjRTx Design Guardrail

Status: accepted

Context:
The local XLA tree shows that XLA already covers most responsibilities PjRTx
cares about: StableHLO/HLO import, HLO passes, simplification, decomposition,
fusion, tiling, Shardy/SPMD, collectives, layout, memory spaces, buffer
assignment, scheduling, backend codegen, PJRT/StreamExecutor runtime,
host/device IO, allocation, profiling, and autotuning. That breadth validates
the compiler-middle direction, but it also highlights XLA's opacity from the
PjRT-to-kernel point of view.

Decision:
Add `next/docs/specs/xla_coverage_map_v0.md` and link it from the architecture
vision, spec README, compiler pass pipeline, and implementation workplan. Treat
XLA as a coverage checklist for responsibilities, not as a hidden compiler or a
structure to copy mechanically.

Reasoning:
PjRTx should match XLA-scale compiler/runtime responsibilities over time while
making every important decision typed, validated, printable, and joinable from
source operation to generated or selected kernel, collective command, hardware
fact, and profile event.

Consequences:
Future work should compare proposed slices against the coverage map. A slice is
on track when it increases PjRTx ownership of compiler decisions and their
explain records. A slice is drifting when it deepens runtime/backend submission
without improving the source-to-hardware causal chain.

Verification:
Documentation-only change.

Links:
- `next/docs/specs/xla_coverage_map_v0.md`
- `next/docs/pjrtx_architecture_vision.md`
- `next/docs/specs/compiler_pass_pipeline_v0.md`

## 2026-05-18 - Matmul Epilogue Fusion Is Explicitly Rejected In V0

Status: accepted

Context:
The next compiler-middle pass after the XLA coverage map was
`matmul_epilogue_fusion_select`. Without it, the V0 planner said the
elementwise chain was fused but did not explicitly answer whether
`dot+broadcast/add/tanh` should become a matmul epilogue. That is exactly the
kind of backend fusion decision that PjRTx should not hide.

Decision:
Add `matmul_epilogue_fusion_select` to the compiler pass catalog. Teach
`planFusion` to recognize the V0 dot-plus-bias-plus-activation pattern and emit
a rejected `FusionGroup` with all four source instruction IDs. Keep the current
lowering behavior: matmul remains a backend kernel-graph boundary and
broadcast/add/tanh lower as an accepted elementwise fusion region.

Reasoning:
Rejecting the epilogue is fine for V0; leaving the decision implicit is not.
The record now explains that accepted epilogue fusion needs explicit kernel IR,
math-policy, tile-pressure, and backend epilogue support before it can become a
target-specific lowering.

Consequences:
Fusion reports now show both the rejected matmul epilogue candidate and the
accepted elementwise chain. Later work can turn that rejected record into an
accepted target-specific lowering without changing the trace shape.

Verification:
`bazel test //next/pjrtx/compiler:unit_tests //next/pjrtx/vertical_slice:lowering_tests --test_output=errors`
passes.

Links:
- `pjrtx/compiler/compiler.zig`
- `next/pjrtx/vertical_slice/lowering_test.zig`
- `next/docs/specs/compiler_pass_pipeline_v0.md`

## 2026-05-18 - Tile Shape Selection Becomes Target-Aware

Status: accepted

Context:
Placement records existed, but V0 still copied the output tensor shape into the
tile shape. That kept the report stable, but it did not prove the compiler was
owning tiling decisions for NPU-like targets with local memory.

Decision:
Rename the placement pass contract to `tile_shape_select` and make
`planPlacement` choose target-aware tile shapes. Metal V0 keeps whole-tensor
logical tiles. NPU V0 now bounds matrix and fusible elementwise tile dimensions
with a conservative synthetic limit before recording local SRAM as tile memory.
Add a large StableHLO fixture proving the compiler records `128x128` NPU tiles
for a `4096x4096` matmul output without relying on backend codegen.

Reasoning:
The tile does not need to be optimal yet, but it must be a compiler decision
joined to target memory facts. This keeps future kernel generation, roofline
rows, and memory traffic refinement anchored to explicit placement records.

Consequences:
The next tiling work should replace the synthetic dimension cap with
capacity-checked tile legality that accounts for input tiles, output tiles,
accumulation buffers, scratch/workspace bytes, and backend kernel constraints.

Verification:
`bazel test //next/pjrtx/compiler:unit_tests //next/pjrtx/vertical_slice:lowering_tests --test_output=errors`
passes.

Links:
- `pjrtx/compiler/compiler.zig`
- `next/pjrtx/fixtures/large_tiling.mlir`
- `next/pjrtx/vertical_slice/lowering_test.zig`
- `next/docs/specs/trace_schema_v0.md`

## 2026-05-18 - Memory Traffic Kinds Become Hierarchy-Aware

Status: accepted

Context:
Memory traffic records existed, but the kind names were still generic:
`device_boundary` and `tile_memory`. That made reports less aligned with the
hardware hierarchy PjRTx wants to explain: HBM/global memory, local SRAM or
scratchpad, DMA, and interconnect traffic.

Decision:
Rename emitted lowering traffic kinds to `global_memory` and `local_memory`.
Add schema-level `host_device_dma` and `interconnect` kinds, but do not emit
them yet because transfer commands and collective commands do not currently
have their own memory-traffic provenance records. Add `memory_traffic_refine`
to the compiler pass catalog after lowering-region formation.

Reasoning:
The report should name the memory hierarchy role directly, not force the reader
to infer it from memory-space IDs. At the same time, PjRTx should not fake DMA
or interconnect traffic by attaching it to compute lowerings before transfer
and collective lowering records exist.

Consequences:
Hardware utilization summaries and golden reports now print
`kind=global_memory` for HBM/unified-device traffic and `kind=local_memory` for
local SRAM tile traffic. Future transfer and collective slices can populate the
reserved DMA/interconnect kinds without changing the enum shape again.

Verification:
`bazel test //next/pjrtx/compiler:unit_tests //next/pjrtx/core:unit_tests //next/pjrtx/vertical_slice:report_test //next/pjrtx/vertical_slice:lowering_tests --test_output=errors`
passes.

Links:
- `pjrtx/core/core.zig`
- `pjrtx/compiler/compiler.zig`
- `next/pjrtx/vertical_slice/testdata/v0_execution_report.txt`
- `next/docs/specs/trace_schema_v0.md`

## 2026-05-18 - Collective Algorithm Selection Becomes Explicit

Status: accepted

Context:
Collective planning existed as a count-only record. Unsupported StableHLO
collectives failed early, but the report did not yet name an algorithm decision
or explain which collective algorithms were unavailable.

Decision:
Add `CollectivePlanDecision` and `CollectiveAlgorithm` to the core trace
schema. Extend `CollectivePlanRecord` with `decision`, `algorithm`,
`estimated_bytes`, and `estimated_latency_ns`. Add
`collective_algorithm_select` to the compiler pass catalog. V0 records
`decision=no_collectives` and `algorithm=none` for graphs without collective
work. Unsupported collectives are imported as graph payloads before
`collective_algorithm_select` rejects them with direct, ring, tree, and split
algorithm diagnostics.

Reasoning:
PjRTx should not claim collective support through silence. Even before V0 has
executable collective commands, the trace needs an explicit algorithm field and
the failure path should say which algorithm family could not be selected.

Consequences:
The next collective slice should deepen group, channel, and token verification
so rejected algorithms can become richer per-collective records and eventually
collective command records.

Verification:
`bazel test //next/pjrtx/core:unit_tests //next/pjrtx/compiler:unit_tests //next/pjrtx/vertical_slice:lowering_tests --test_output=errors`
passes.

Links:
- `pjrtx/core/core.zig`
- `pjrtx/compiler/compiler.zig`
- `next/pjrtx/vertical_slice/lowering_test.zig`
- `next/docs/specs/trace_schema_v0.md`

## 2026-05-18 - Collective Payload Import Moves Failure Into Compiler Middle

Status: accepted

Context:
The last collective slice made algorithm selection explicit, but
`stablehlo.all_reduce` still died before graph import. That was a drift from
the vision: PjRTx should explain communication from StableHLO/Shardy metadata
through target topology and selected or rejected algorithms, not stop at a raw
MLIR gate.

Decision:
Rename the catalog boundary to `collective_graph_payload_import`. Import
`stablehlo.all_reduce` as a typed `GraphPayload.collective` with add reduction,
replica group count, replica group size, optional channel ID, and token-use
state. Keep the no-fallback rule by rejecting imported collective payloads at
`collective_algorithm_select` before cost, lowering, schedule, backend binding,
or runtime execution.

Reasoning:
The failure point now has the facts needed for explainability: source op,
graph instruction ID, target backend, group shape, channel state, and rejected
direct/ring/tree/split algorithms. This matches the architecture goal better
than an early text-level gate while still refusing unsupported execution.

Consequences:
Collective ops are hard fusion/backend boundaries in V0. Backend capability code
does not treat collective payloads as executable kernels. The next slice should
verify channel handles, token dependencies, and compile topology, then connect
collective payloads to Shardy-derived communication requirements and
schedule-overlap planning.

Verification:
`bazel test //next/pjrtx/... --test_output=errors` passes.

Links:
- `pjrtx/core/core.zig`
- `pjrtx/compiler/compiler.zig`
- `pjrtx/backend/backend.zig`
- `next/pjrtx/vertical_slice/lowering_test.zig`
- `next/docs/specs/compiler_pass_pipeline_v0.md`
- `next/docs/specs/trace_schema_v0.md`

## 2026-05-18 - Collective Group And Channel Verification Lands

Status: accepted

Context:
Collective payload import preserved only the group shape and optional channel
slot. That was enough to move failure into the compiler middle, but not enough
to explain whether the collective matched the compile topology.

Decision:
Add `collective_group_channel_verify` to the compiler pass catalog. Store
explicit replica-group participant IDs plus StableHLO channel handle ID/type in
`CollectiveSpec`. Verify imported collective participants against
`replicas * partitions`, reject duplicate participants, reject incomplete
channel handles, and keep tokenized collectives off the V0 executable path.

Reasoning:
Collective correctness is mathematical and topological, not just an operation
name. The compiler needs to know which participants communicate, which channel
binds the rendezvous, and whether token ordering exists before it can choose
direct, ring, tree, split, or unsupported behavior.

Consequences:
Unsupported all-reduce diagnostics now include verified group/channel facts
before algorithm selection. The next compiler-middle step can move to
`schedule_overlap_plan`: modeling compute, DMA, and future collective overlap
with explicit event and stream constraints.

Verification:
`bazel test //next/pjrtx/... --test_output=errors` passes.

Links:
- `pjrtx/core/core.zig`
- `pjrtx/compiler/compiler.zig`
- `next/pjrtx/fixtures/all_reduce_channel.mlir`
- `next/pjrtx/vertical_slice/lowering_test.zig`
- `next/docs/specs/compiler_pass_pipeline_v0.md`
- `next/docs/specs/trace_schema_v0.md`

## 2026-05-18 - Schedule Overlap Planning Starts As Explicit Serialized Edges

Status: accepted

Context:
The V0 schedule had H2D, backend execution, and D2H commands, but overlap was
still implicit: the runtime simply executed commands in dependency order. That
left no trace surface for future async DMA, event waits, collective engines, or
stream planning.

Decision:
Add `ScheduleOverlapRecord` to the trace report and add `schedule_overlap_plan`
to the compiler pass catalog between `schedule_build` and backend binding. The
V0 planner records the H2D-to-compute and compute-to-D2H data edges as
serialized overlap decisions with command IDs, stream IDs, dependency kind, and
stable reasons.

Reasoning:
Overlap is a performance decision, and performance decisions must be visible
before the backend or runtime can act on them. Recording serialized edges now
keeps V0 mathematically conservative while giving future passes the exact rows
they need to upgrade edges into selected or rejected async plans.

Consequences:
Backend binding now consumes a schedule that already exposes overlap facts.
Future work can add transfer memory traffic, buffer lifetime, event dependency,
collective engine, and stream-capacity constraints without changing the basic
report shape.

Verification:
`bazel test //next/pjrtx/... --test_output=errors` passes.

Links:
- `pjrtx/core/core.zig`
- `pjrtx/compiler/compiler.zig`
- `pjrtx/runtime/runtime.zig`
- `next/pjrtx/vertical_slice/import_test.zig`
- `next/pjrtx/vertical_slice/lowering_test.zig`
- `next/pjrtx/vertical_slice/testdata/tiny_trace_report.txt`
- `next/docs/specs/compiler_pass_pipeline_v0.md`
- `next/docs/specs/trace_schema_v0.md`
- `next/docs/specs/implementation_workplan_v0.md`

## 2026-05-18 - Kernel Codegen Planning Enters The Compiler Middle

Status: accepted

Context:
Backend binding previously jumped from schedule commands directly to a backend
operation string. That was too coarse for the vision: generated kernels,
library calls, MLS graph nodes, and NPU unit choices must be explainable from
lowering records and memory traffic before backend executable planning runs.

Decision:
Add `KernelCodegenRecord` and `kernel_codegen_plan`. The planner now emits one
record per executable lowering region, linking lowering ID, backend execute
command, backend kind, operation name, graph instruction IDs, cost ledger IDs,
memory traffic IDs, expected unit, and a stable reason.

Reasoning:
This creates the missing vertical bridge from compiler-middle lowering to
backend-specific generated or selected kernels. The first records are still
small, but they make the next hard work tractable: matmul epilogue fusion,
tile legality, kernel IR shape, and Metal/MLS or NPU generated kernel details
can attach to a stable codegen row instead of being buried in backend code.

Consequences:
Backend binding now follows codegen planning in the pass catalog. The V0 NPU
fixture exposes separate `npu_matmul` and `npu_elementwise_fusion` records, and
Metal can later map the same shape to MLS graph/kernel descriptors without
losing source, cost, memory, or lowering provenance.

Verification:
`bazel test //next/pjrtx/... --test_output=errors` passes.

Links:
- `pjrtx/core/core.zig`
- `pjrtx/compiler/compiler.zig`
- `pjrtx/runtime/runtime.zig`
- `next/pjrtx/vertical_slice/import_test.zig`
- `next/pjrtx/vertical_slice/lowering_test.zig`
- `next/docs/specs/compiler_pass_pipeline_v0.md`
- `next/docs/specs/trace_schema_v0.md`
- `next/docs/specs/compile_pipeline_v0.md`
- `next/docs/specs/implementation_workplan_v0.md`

## 2026-05-18 - Kernel Codegen Shape Makes Region Boundaries Measurable

Status: accepted

Context:
`KernelCodegenRecord` named generated or selected kernel candidates, but the
first version still lacked a backend-neutral way to describe the region shape.
That made it hard to review whether later fusion acceptance really changes the
generated kernel boundary or only changes an operation string.

Decision:
Add `KernelCodegenShape` and explicit value-flow lists to each codegen record.
The shape records operation count, external input count, external output count,
and intermediate value count for the lowering region. The value-flow lists name
the exact graph values behind those counts.

Reasoning:
This is intentionally smaller than a full kernel IR. It is enough to validate
the current V0 regions: matmul is one operation with two external inputs and
one output, while fused broadcast/add/tanh is three operations with two
external inputs, one output, and two internal intermediates. The ID lists make
that boundary actionable for allocation and future matmul epilogue acceptance,
not just human-readable.

Consequences:
Codegen records now carry stable shape and value-flow facts before backend
binding. The next step can attach tile/memory pressure to those same records or
use them to accept the matmul epilogue fusion under strict math and target
capacity checks.

Verification:
`bazel test //next/pjrtx/... --test_output=errors` passes.

Links:
- `pjrtx/core/core.zig`
- `pjrtx/compiler/compiler.zig`
- `next/pjrtx/vertical_slice/import_test.zig`
- `next/pjrtx/vertical_slice/lowering_test.zig`
- `next/docs/specs/compiler_pass_pipeline_v0.md`
- `next/docs/specs/trace_schema_v0.md`
- `next/docs/specs/implementation_workplan_v0.md`

## 2026-05-18 - Codegen Records Carry Tile And Memory Pressure

Status: accepted

Context:
Codegen records had operation names, region shape, and value-flow IDs, but a
future fusion accept pass still had to chase placement and memory traffic
separately to know whether a generated region fit the target. That was a small
but real drift from the goal that performance decisions be visible at every
layer.

Decision:
Attach logical tile shape, result memory space, optional tile memory space, and
global/local byte pressure to each `KernelCodegenRecord`. The byte pressure is
derived from the existing `MemoryTrafficRecord` rows, so memory traffic remains
the source of truth.

Reasoning:
Generated kernels are where shape, memory hierarchy, expected unit, and value
flow meet. Carrying those facts on the codegen row lets future matmul epilogue
fusion and tile legality checks decide from one typed artifact without
rediscovering facts from StableHLO strings or backend graph construction.

Consequences:
The V0 fixture now proves two regions: `npu_matmul` carries a `2x3` tile with
HBM/global and local-SRAM pressure, while `npu_elementwise_fusion` carries the
same tile plus fused-region intermediate pressure. The next step can move from
summary pressure to capacity-checked tile legality or accept matmul epilogue
fusion with explicit pressure deltas.

Verification:
`bazel test //next/pjrtx/... --test_output=errors` passes.

Links:
- `pjrtx/core/core.zig`
- `pjrtx/compiler/compiler.zig`
- `next/pjrtx/vertical_slice/import_test.zig`
- `next/docs/specs/compiler_pass_pipeline_v0.md`
- `next/docs/specs/trace_schema_v0.md`
- `next/docs/specs/implementation_workplan_v0.md`

## 2026-05-18 - Tile Legality Becomes A Compile Gate

Status: accepted

Context:
Codegen records carried tile shape, value-flow IDs, and global/local pressure,
but those facts were still descriptive. A future fusion accept pass could read
the pressure, but the compiler had not yet proven that the selected tile and
memory-space decision fit the target.

Decision:
Add `tile_legality_verify` after `kernel_codegen_plan` and before backend
binding. The pass checks result-memory output bytes and tile-memory live bytes
against target memory-space capacities. It fails compilation with a diagnostic
instead of allowing backend binding to repair or reinterpret the plan.

Reasoning:
This is the first capacity gate over the generated-kernel view of the program.
The check is intentionally conservative, but it keeps the no-fallback contract:
if codegen says a region uses local SRAM, the compiler must prove the region
fits local SRAM before any backend sees it.

Consequences:
Backend binding now consumes only capacity-checked codegen records. A test with
an intentionally tiny `local_sram` target proves that the V0 workload fails at
`tile_legality_verify` with `tile live bytes exceed memory capacity`.

Verification:
`bazel test //next/pjrtx/... --test_output=errors` passes.

Links:
- `pjrtx/compiler/compiler.zig`
- `next/pjrtx/vertical_slice/lowering_test.zig`
- `next/docs/specs/compiler_pass_pipeline_v0.md`
- `next/docs/specs/compile_pipeline_v0.md`
- `next/docs/specs/implementation_workplan_v0.md`

## 2026-05-18 - Matmul Epilogue Fusion Gets A Pressure Delta

Status: accepted

Context:
The compiler already recognized `dot+broadcast/add/tanh` as a legal-looking
epilogue candidate, but it rejected the fusion without quantifying the tradeoff.
That left the future accept decision under-explained: we knew launch count and
some bytes could improve, but not the live-memory cost of making one larger
kernel.

Decision:
Extend `FusionGroup` with `FusionPressureDelta`. For the rejected matmul
epilogue candidate, record split kernel count, fused kernel count, split peak
live bytes, fused live bytes, additional live bytes, and global bytes saved.

Reasoning:
This keeps acceptance separate from measurement. V0 still rejects the epilogue,
but the trace now says what would change: the current fixture moves from two
kernels to one, saves 72 global bytes, and would require 80 additional live
bytes compared with the split peak.

Consequences:
The next fusion step can make a real decision from explicit pressure data and
tile legality instead of changing a boolean. Backend codegen still does not
need to infer the candidate from StableHLO strings.

Verification:
`bazel test //next/pjrtx/... --test_output=errors` passes.

Links:
- `pjrtx/core/core.zig`
- `pjrtx/compiler/compiler.zig`
- `next/pjrtx/vertical_slice/lowering_test.zig`
- `next/docs/specs/trace_schema_v0.md`
- `next/docs/specs/compiler_pass_pipeline_v0.md`
- `next/docs/specs/implementation_workplan_v0.md`

## 2026-05-19 - MLIR State Machine Becomes Compiler Truth

Status: accepted

Context:
The V0 implementation has accumulated useful Zig records for pass reports,
fusion, placement, collectives, lowering, memory traffic, schedule overlap, and
kernel codegen. Those records made the vocabulary explicit, but keeping them as
the permanent lowering representation would drift away from the compiler
vision: PjRTx should explain and verify transformations across MLIR boundaries,
not rebuild an MLIR/HLO compiler as parallel Zig structs.

Decision:
Define `mlir_state_machine_compiler_v0.md` as the correction. MLIR owns
compiler-middle state, transformation facts, legality proofs, pressure facts,
collective facts, codegen facts, and schedule facts. Zig records remain as V0
schema prototypes, extracted report views, runtime handoff structures, and
golden-test artifacts.

Reasoning:
Fusion, tiling, bufferization, collectives, and codegen are naturally MLIR
problems: they need dialect conversion, op/region attributes, verifier
boundaries, canonical IR snapshots, and pass-pipeline tests. Zig should be used
where it is strongest: PJRT/API edges, target/runtime ownership, deterministic
extraction, validation, `std.Io.Reader`/`std.Io.Writer` plumbing, runfiles, and
debuggable harnesses.

Consequences:
The next implementation work should add a minimal PjRTx MLIR dialect/state
machine skeleton before adding more Zig-only compiler-middle records. Existing
records should either be extracted from MLIR or explicitly marked as temporary
scaffolding until the corresponding MLIR state exists.

Verification:
Docs-only change; no Bazel tests required.

Links:
- `next/docs/specs/mlir_state_machine_compiler_v0.md`
- `next/docs/specs/README.md`
- `next/docs/pjrtx_architecture_vision.md`
- `next/docs/specs/compiler_pass_pipeline_v0.md`
- `next/docs/specs/trace_schema_v0.md`
- `next/docs/specs/implementation_workplan_v0.md`

## 2026-05-19 - First PjRTx MLIR Dialect Slice Is Fusion Extraction

Status: accepted

Context:
After deciding that MLIR is the compiler truth, the next risk was staying at
the vision level and continuing to add Zig records anyway. The project needed a
small, implementable MLIR dialect/API slice. The local ZML project already has
useful technical patterns for MLIR in Zig: a registry/context/module/pass
manager owner, explicit dialect loading, module attributes, operation
construction, pass-manager pipelines, and `std.Io` threaded through compile
setup.

Decision:
Add `pjrtx_mlir_dialect_v0.md`. The first slice is module state, target
attachment, fusion candidate/decision, pressure delta, verifier rules, and
extraction from MLIR fusion facts to the current `FusionGroup` report view.
Zig should add a thin `MlirSession` wrapper inspired by ZML, while keeping
compiler facts in MLIR.

Reasoning:
Fusion plus pressure delta is already the most concrete compiler-middle
decision in V0. Moving that fact first proves the architectural direction
without boiling the ocean. ZML shows how to keep MLIR handling ergonomic in Zig,
but PjRTx must adapt it for compiler-state verification and no-fallback
diagnostics rather than frontend DSL emission.

Consequences:
The next implementation step should be `//next/pjrtx/compiler/mlir_state` or an
equivalent package that owns `MlirSession`, state attributes, target attachment,
fusion fact attachment, verification, and extraction. More Zig-only
compiler-middle records should wait unless the spec says what MLIR fact they
prototype.

Verification:
Docs-only change; no Bazel tests required.

Links:
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`
- `/Users/hugo/Developer/zml/zml/module.zig`
- `/Users/hugo/Developer/zml/zml/mlirx.zig`
- `next/docs/specs/mlir_state_machine_compiler_v0.md`
- `next/docs/specs/implementation_workplan_v0.md`

## 2026-05-19 - `mlir_state` Adds The First MLIR-Owned Compiler Fact

Status: accepted

Context:
The MLIR dialect slice needed code before it could steer the rest of the
compiler. Without a concrete target, the project would keep adding Zig-only
records and postpone the harder MLIR ownership question.

Decision:
Add `//next/pjrtx/compiler/mlir_state`. It owns a small `MlirSession`, parses
StableHLO text into MLIR, attaches `pjrtx.state` and target identity
attributes, verifies state transitions, attaches fusion/pressure facts to MLIR,
and extracts those facts back to `core.FusionGroup`.

Reasoning:
This is deliberately a dialect shim rather than the final TableGen dialect.
String attributes are enough to prove the direction: compiler facts can live on
MLIR, be verified before use, appear in MLIR snapshots, and become the source
for existing Zig report views.

Consequences:
The next implementation step should route the existing `planFusion` output
through `mlir_state` during `compileV0FromReader`, then replace that with an
MLIR-native fusion planning pass. After that, tile/memory/codegen/schedule
facts can move by following the same pattern.

Verification:
`bazel test //next/pjrtx/compiler/mlir_state:unit_tests --test_output=errors` passes.

Links:
- `pjrtx/compiler/mlir_state/package.zig`
- `pjrtx/compiler/BUILD.bazel`
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`
- `next/docs/specs/implementation_workplan_v0.md`

## 2026-05-19 - Compile V0 Routes Fusion Through MLIR State

Status: accepted

Context:
`mlir_state` proved that fusion facts could be attached to MLIR and extracted
back to `FusionGroup`, but the end-to-end compile path still used the
Zig-produced fusion plan directly. That left the new MLIR path as a sidecar
test rather than the compiler route.

Decision:
Change `compileV0FromReader` to create an `MlirSession`, attach target state,
mark target legality after the existing target check, attach the V0 fusion plan
to MLIR, and replace the builder's fusion plan with groups extracted from MLIR.
The lower-level `planV0TraceReport` helper remains graph-only for tests that
intentionally start after MLIR.

Reasoning:
This keeps behavior stable while moving ownership in the right direction. The
current Zig planner still discovers the groups, but the final trace receives
fusion groups from MLIR state on the orchestrated compile path. That creates the
bridge needed to later replace the Zig planner with an MLIR-native pass.

Consequences:
The next implementation step should replace the temporary `pjrtx.fusion.plan`
string attribute with structured attributes or dialect ops, then move fusion
candidate discovery itself into MLIR. Ownership now tracks extracted strings so
report deinit remains correct.

Verification:
`bazel test //next/pjrtx/compiler:unit_tests //next/pjrtx/compiler/mlir_state:unit_tests //next/pjrtx/vertical_slice:lowering_tests --test_output=errors` passes.

Links:
- `pjrtx/compiler/compiler.zig`
- `pjrtx/compiler/mlir_state/package.zig`
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`
- `next/docs/specs/implementation_workplan_v0.md`

## 2026-05-19 - Fusion Plan Becomes A Structured MLIR Attribute

Status: accepted

Context:
The first `mlir_state` bridge attached `pjrtx.fusion.plan` as a delimited
string. That proved MLIR ownership, but the representation was still too close
to ad hoc serialization and did not give verifiers or extractors a structured
MLIR shape to inspect.

Decision:
Change `pjrtx.fusion.plan` to an MLIR array attribute of dictionary attributes.
Each fusion entry carries typed scalar fields, an instruction ID array, nested
pressure-delta dictionary, and string-encoded U128 byte counters.

Reasoning:
This is still a shim before the real dialect, but it removes the text blob and
makes extraction read MLIR attribute structure directly. U128 byte counters stay
string-encoded because the final dialect has not yet defined the integer-width
policy for large byte quantities.

Consequences:
The next step should be dialect attrs or ops with verifier hooks, then moving
fusion discovery itself into MLIR. The current structured shim is good enough
to preserve the "MLIR fact -> Zig report view" direction.

Verification:
`bazel test //next/pjrtx/compiler/mlir_state:unit_tests --test_output=errors` passes.

Links:
- `pjrtx/compiler/mlir_state/package.zig`
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`
- `next/docs/specs/implementation_workplan_v0.md`

## 2026-05-19 - First MLIR-Owned Fusion Candidate Discovery Fact

Status: accepted

Context:
Fusion planning had moved its state transition into a Zig external MLIR pass,
but candidate discovery was still entirely a Zig graph-planner concept. That
kept PjRTx close to the old drift: MLIR carried decisions only after a parallel
representation had already discovered them.

Decision:
Add `runFusionCandidateDiscoveryExternalPass` and call it on the orchestrated
compile path after target legality and before the V0 fusion planner. The pass
walks MLIR operations through the C API, recognizes the current V0
`tanh(add(dot_general, broadcast_in_dim))` shape, and stamps
`pjrtx.fusion.candidates.matmul_epilogue`,
`pjrtx.fusion.candidates.elementwise_chain`, and `pjrtx.fusion_candidate.pass`
on the module.

Reasoning:
This is deliberately a small discovery slice rather than a full fusion planner.
It proves the direction: candidate facts can originate in MLIR pass-manager
execution before Zig report extraction and before backend decisions.

Consequences:
The next fusion work can compare V0 planner groups against MLIR candidate facts,
then move accepted/rejected group construction into MLIR as richer attrs or
real dialect ops. The current slice still has no fallback path and does not
claim a detailed decision from the candidate counters alone.

Verification:
`bazel test //next/pjrtx/compiler/mlir_state:unit_tests //next/pjrtx/compiler:unit_tests --test_output=errors` passes.

Links:
- `pjrtx/compiler/mlir_state/package.zig`
- `pjrtx/compiler/compiler.zig`
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`
- `next/docs/specs/implementation_workplan_v0.md`

## 2026-05-19 - Fusion Decisions Must Match MLIR Candidate Facts

Status: accepted

Context:
The compiler had MLIR-owned fusion candidate counts, but the V0 planner could
still attach arbitrary fusion groups afterward. That left a drift path between
MLIR discovery and Zig decision records.

Decision:
Make `attachFusionGroups` verify V0 fusion groups against
`pjrtx.fusion.candidates.matmul_epilogue` and
`pjrtx.fusion.candidates.elementwise_chain` before it attaches
`pjrtx.fusion.plan`. Missing candidate facts or count mismatches now fail the
compile boundary.

Reasoning:
The detailed fusion decision still lives in the temporary V0 planner, but it no
longer defines the candidate universe alone. MLIR discovery is now a required
source fact for decision attachment.

Consequences:
Direct MLIR-state tests and boundary tests must run candidate discovery before
attaching fusion groups. The next step can either move group construction into
the MLIR pass or replace candidate counters with richer candidate attrs/ops.

Verification:
`bazel test //next/pjrtx/compiler/mlir_state:unit_tests //next/pjrtx/vertical_slice:mlir_state_test //next/pjrtx/compiler:unit_tests --test_output=errors` passes.

Links:
- `pjrtx/compiler/mlir_state/package.zig`
- `next/pjrtx/vertical_slice/mlir_state_test.zig`
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`

## 2026-05-19 - Fusion Candidates Become Structured MLIR Facts

Status: accepted

Context:
Fusion candidate discovery was MLIR-owned, but it only exposed per-kind count
attributes. Counts were enough to block drift, but not enough to explain what
the compiler saw or to become the base for future candidate-to-decision
lowering.

Decision:
Change `runFusionCandidateDiscoveryExternalPass` to stamp a structured
`pjrtx.fusion.candidates` array attribute. Each entry carries `index`, `kind`,
`root`, `operation_count`, and `reason`. Keep the older per-kind count
attributes as derived compatibility markers, but make decision attachment verify
against the structured array.

Reasoning:
This is the next step from "MLIR discovered something" to "MLIR carries a
candidate object." It remains an attribute shim, not the final dialect op, but
it gives the verifier and golden harness a real MLIR-owned shape to inspect.

Consequences:
The next migration step can add decision fields to these candidate entries or
replace the structured attr with real PjRTx dialect attrs/ops. The V0 planner
still produces final `FusionGroup` records, but it is now checked against
structured MLIR candidate facts.

Verification:
`bazel test //next/pjrtx/compiler/mlir_state:unit_tests //next/pjrtx/vertical_slice:mlir_state_test //next/pjrtx/compiler:unit_tests --test_output=errors` passes.

Links:
- `pjrtx/compiler/mlir_state/package.zig`
- `next/pjrtx/vertical_slice/testdata/mlir_state_after_fusion.txt`
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`

## 2026-05-19 - Fusion Extraction Reads Enriched Candidates

Status: accepted

Context:
Candidates carried decision links, but `extractFusionGroups` still read the
temporary `pjrtx.fusion.plan` attribute. That meant the final report view was
not yet extracted from the discovery-to-decision object.

Decision:
Enrich candidate entries with `instructions` and `pressure_delta`, then change
`extractFusionGroups` to read `pjrtx.fusion.candidates`. The temporary
`pjrtx.fusion.plan` attribute remains the attachment surface for the V0 planner,
but it is no longer the final extraction source.

Reasoning:
This moves one more responsibility into the MLIR-owned candidate object. The
candidate now carries enough information to produce the public `FusionGroup`
view after the fusion-plan pass has run.

Consequences:
The next step can shrink or replace `pjrtx.fusion.plan` itself by having the
MLIR pass construct decisions directly on candidates, or by replacing the attr
shim with real PjRTx dialect attrs/ops.

Verification:
`bazel test //next/pjrtx/compiler/mlir_state:unit_tests //next/pjrtx/vertical_slice:mlir_state_test //next/pjrtx/compiler:unit_tests --test_output=errors` passes.

Links:
- `pjrtx/compiler/mlir_state/package.zig`
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`

## 2026-05-19 - Fusion Candidates Carry Decision Links

Status: accepted

Context:
Structured fusion candidates existed, and planner groups had to match them, but
the candidate entries still stopped at discovery. The chosen or rejected
decision lived only in `pjrtx.fusion.plan`, leaving two MLIR facts that were
related by convention rather than by structure.

Decision:
Extend `runFusionPlanExternalPass` so it enriches each
`pjrtx.fusion.candidates` entry with the matching planner decision. Candidate
entries now carry `plan_index`, `decision`, `bytes_saved`,
`launch_count_reduction`, and `decision_reason` after the fusion-plan pass
runs.

Reasoning:
This makes the candidate object the bridge from discovery to decision. The V0
planner still supplies the decision, but MLIR now carries the joined fact before
extraction.

Consequences:
The next step can either make the fusion-plan extractor read candidates instead
of `pjrtx.fusion.plan`, or move decision construction itself into the MLIR pass
and shrink the V0 Zig planner role.

Verification:
`bazel test //next/pjrtx/compiler/mlir_state:unit_tests //next/pjrtx/vertical_slice:mlir_state_test //next/pjrtx/compiler:unit_tests --test_output=errors` passes.

Links:
- `pjrtx/compiler/mlir_state/package.zig`
- `next/pjrtx/vertical_slice/testdata/mlir_state_after_fusion.txt`
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`

## 2026-05-19 - Fusion Planned State Moves Into A Zig External MLIR Pass

Status: accepted

Context:
After target legality moved into a Zig external MLIR pass, fusion still used a
direct Zig state transition. That kept the structured `pjrtx.fusion.plan`
attribute useful for extraction, but the `fusion_planned` boundary was not yet
owned by the MLIR pass manager.

Decision:
Change `attachFusionGroups` so it attaches the structured fusion plan attribute
and then runs `runFusionPlanExternalPass`. The pass verifies `target_legal`
state, checks that `pjrtx.fusion.plan` is an array of dictionary attributes, and
validates required decision, instruction, pressure, byte, and reason fields
before setting `pjrtx.state = "fusion_planned"` and stamping
`pjrtx.fusion_plan.pass`.

Reasoning:
This keeps the V0 planner intact while shifting another compiler boundary into
the intended architecture: MLIR state plus pass-manager verification first,
extracted Zig report view second.

Consequences:
The next step should move beyond attach-then-verify toward MLIR-native fusion
candidate discovery or start replacing the attribute shim with first real
dialect attrs/ops. Either path now has a pass-manager execution slot and stable
failure tests.

Verification:
`bazel test //next/pjrtx/compiler/mlir_state:unit_tests --test_output=errors` passes.

Links:
- `pjrtx/compiler/mlir_state/package.zig`
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`
- `next/docs/specs/implementation_workplan_v0.md`

## 2026-05-19 - MLIR State Gets A Runfiles-Backed Boundary Golden

Status: accepted

Context:
`mlir_state` had unit tests, but the MLIR state boundary still lacked a stable
artifact in the vertical slice harness. Without a golden, it would be easy to
change the MLIR-owned target or fusion attributes without noticing that the
explainability surface drifted.

Decision:
Add `//next/pjrtx/vertical_slice:mlir_state_test` and
`testdata/mlir_state_after_fusion.txt`. The test reads the real
`tanh_dot_bias.mlir` fixture through runfiles, builds the V0 graph and fusion
plan, attaches target/fusion state to `MlirSession`, then compares a stable
MLIR-state summary.

Reasoning:
Raw MLIR printing is still too tied to temporary attribute spelling and
printer formatting. The golden summary checks the compiler facts that matter:
module state, target identity, target fingerprint, fusion count, instruction
IDs, pressure delta, bytes saved, launch delta, and reasons. It still reads
from MLIR attributes, not from the original Zig plan.

Consequences:
The boundary harness now has a place to grow as the shim becomes a real dialect.
The next step can add raw MLIR/FileCheck-style snapshots after dialect spelling
stabilizes.

Verification:
`bazel test //next/pjrtx/vertical_slice:mlir_state_test --test_output=errors`
passes.

Links:
- `next/pjrtx/vertical_slice/mlir_state_test.zig`
- `next/pjrtx/vertical_slice/testdata/mlir_state_after_fusion.txt`
- `pjrtx/compiler/mlir_state/package.zig`
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`

## 2026-05-19 - Zig-First MLIR Dialect Direction

Status: accepted

Context:
After moving compiler truth into MLIR state, the next design question was
whether PjRTx can define its own dialects and passes mostly in Zig with minimal
C code. The previous docs said to avoid callback-heavy Zig pass infrastructure,
but that was too broad: MLIR exposes external pass callbacks through the C API,
which lets Zig own PjRTx pass policy without inventing a parallel IR.

Decision:
Document a Zig-first, MLIR-native split. Zig owns compiler policy, cost models,
target legality, diagnostics, extraction, harnesses, and external MLIR pass
callbacks when the MLIR C API is sufficient. Minimal MLIR-native
C++/TableGen owns real dialect registration, generated ops/types/attrs,
parser/printer/verifier integration, and rewrite or dialect-conversion helpers
that are not practical through the C API.

Reasoning:
Pure Zig is a good goal for PjRTx behavior, but pure Zig is not a realistic
implementation strategy for full MLIR dialects today. MLIR dialect classes,
ODS/TableGen definitions, parser/printer hooks, verifier integration, and
advanced rewrite infrastructure are C++-first. Keeping that layer narrow gives
PjRTx real dialects without turning the project into an opaque C++ compiler.

Consequences:
The immediate next implementation milestone should be a Zig external MLIR pass
proof through `MlirSession` and MLIR's pass manager. After that, introduce
`//next/pjrtx/dialects/...` and `//next/pjrtx/compiler/mlir/...` only for narrow dialect
and shim responsibilities. No meaningful compiler decision may be hidden inside
the shim; every pass still needs a stable name, state transition, diagnostics,
verifier boundary, and extracted explanation record.

Verification:
Docs-only change.

Links:
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`
- `next/docs/specs/mlir_state_machine_compiler_v0.md`
- `next/docs/specs/compiler_pass_pipeline_v0.md`
- `next/docs/specs/implementation_workplan_v0.md`
- `next/docs/pjrtx_architecture_vision.md`

## 2026-05-19 - Zig External MLIR Pass Proof Lands

Status: accepted

Context:
The docs now require a Zig-first MLIR pass direction, but the implementation
still only used straight-line Zig functions that mutate MLIR attributes. That
left an important question unproven: whether Zig-owned pass policy can actually
enter MLIR's pass manager through the external pass C API.

Decision:
Extend `MlirSession` with a pass manager lifetime and add
`runExternalStateProbePass`. The proof pass is implemented in Zig with
`callconv(.c)` MLIR external-pass callbacks. It verifies that `pjrtx.state`
exists on the module and stamps `pjrtx.external_pass.proof = "ran"` through the
MLIR operation handle.

Reasoning:
This is intentionally tiny. The value is architectural, not in the mutation
itself: PjRTx can run Zig-owned policy inside MLIR's pass manager without
building a separate Zig pass framework and without moving policy into C++.

Consequences:
Future PjRTx policy passes should prefer this path when the MLIR C API exposes
the needed operation/attribute access. Minimal C++/TableGen remains reserved for
real dialect definitions and rewrite/dialect-conversion support that cannot be
expressed cleanly through C.

Verification:
`bazel test //next/pjrtx/compiler/mlir_state:unit_tests --test_output=errors` passes.

Links:
- `pjrtx/compiler/mlir_state/package.zig`
- `pjrtx/compiler/mlir_c_api.h`
- `pjrtx/compiler/BUILD.bazel`
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`
- `next/docs/specs/implementation_workplan_v0.md`

## 2026-05-19 - Target Legal State Moves Into A Zig External MLIR Pass

Status: accepted

Context:
The first external pass only proved the mechanism. The actual state machine
still used a direct Zig helper to set `target_legal`, which meant the
documented pass-manager direction was not yet shaping compiler state.

Decision:
Make `markTargetLegal` run `runTargetLegalExternalPass`. The pass is implemented
in Zig as an MLIR external pass, checks that the module is in `target_attached`
state, verifies required target attributes, stamps `pjrtx.target_legal.pass`,
and sets `pjrtx.state = "target_legal"`.

Reasoning:
This turns the proof into a real compiler boundary while keeping the public
API stable. Graph target legality remains in the orchestrator before this pass;
the external pass owns the MLIR state transition and target-attachment
verification.

Consequences:
Future state transitions should move one at a time into Zig external MLIR
passes when the C API is sufficient. The next candidate is fusion attachment or
fusion candidate discovery, because it already has structured MLIR facts and
stable extraction tests.

Verification:
`bazel test //next/pjrtx/compiler/mlir_state:unit_tests --test_output=errors` passes.

Links:
- `pjrtx/compiler/mlir_state/package.zig`
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`
- `next/docs/specs/implementation_workplan_v0.md`

## 2026-05-19 - Fusion Plan Becomes An Ephemeral MLIR Handoff

Status: accepted

Context:
The MLIR fusion-planning slice still kept two overlapping facts after planning:
`pjrtx.fusion.plan` held the original V0 planner records, while
`pjrtx.fusion.candidates` held the MLIR-discovered candidates enriched with the
matched decisions. That duplication made the final module less explainable
because a reader had to decide which fact was authoritative.

Decision:
Keep `pjrtx.fusion.plan` only as the temporary input surface from the current
V0 Zig planner into `runFusionPlanExternalPass`. The pass verifies it, matches
it against MLIR-owned fusion candidates, enriches those candidates with the
decision fields, and then removes `pjrtx.fusion.plan` from the module before
setting `pjrtx.state = "fusion_planned"`.

Reasoning:
The durable compiler fact should follow the MLIR discovery object from
candidate to decision, not create a second final representation. This keeps the
explanation chain vertical: StableHLO pattern discovery, planner decision,
pressure delta, extraction, and report summary all read through the same
candidate stream.

Consequences:
Post-`fusion_planned` snapshots should contain `pjrtx.fusion.candidates` and
should not contain `pjrtx.fusion.plan`. The human `fusion_plan` summary remains
for compatibility, but it is rendered from enriched candidates rather than the
temporary handoff attribute.

Verification:
`bazel test //next/pjrtx/... --test_output=errors` passes.

Links:
- `pjrtx/compiler/mlir_state/package.zig`
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`
- `next/docs/specs/implementation_workplan_v0.md`

## 2026-05-19 - V0 Fusion Decisions Move Into The MLIR Pass Path

Status: accepted

Context:
The post-`fusion_planned` module no longer kept `pjrtx.fusion.plan`, but the
orchestrated compile path still called `planFusion` first and then handed those
decisions to MLIR. That meant MLIR owned the final state, but the real decision
still originated in a parallel Zig graph planner.

Decision:
Make `runFusionCandidateDiscoveryExternalPass` carry enough facts for V0
decision construction: graph instruction ids, tensor-byte-derived savings,
launch reduction, and pressure delta. Then let `runFusionPlanExternalPass`
apply the V0 decision policy directly from `pjrtx.fusion.candidates` when no
temporary `pjrtx.fusion.plan` handoff is attached. The orchestrated compile path
now uses this direct MLIR route and extracts `FusionGroup` records afterward.

Reasoning:
This is still a V0 policy, not the final fusion optimizer, but the policy now
runs at the MLIR pass boundary that owns the state transition. The explanation
chain is tighter: StableHLO walk, candidate facts, byte/pressure facts,
decision, extraction, and report summary all flow through MLIR-owned
attributes.

Consequences:
`attachFusionGroups` remains only as a compatibility shim for tests or callers
that already have graph-produced fusion groups. The next fusion step should
replace the attribute shim with real PjRTx dialect attrs/ops or broaden the
MLIR-side candidate discovery beyond the first dot/broadcast/add/tanh shape.

Verification:
`bazel test //next/pjrtx/... --test_output=errors` passes.

Links:
- `pjrtx/compiler/mlir_state/package.zig`
- `pjrtx/compiler/compiler.zig`
- `next/pjrtx/vertical_slice/mlir_state_test.zig`
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`
- `next/docs/specs/implementation_workplan_v0.md`

## 2026-05-19 - Fusion Handoff Dead Code Removed

Status: accepted

Context:
After V0 fusion decisions moved into `runFusionPlanExternalPass`, the old
`attachFusionGroups` path was no longer used by the orchestrated compiler. It
kept a second way to enter `fusion_planned` by constructing `pjrtx.fusion.plan`
from graph-produced groups, which contradicted the no-fallback direction.

Decision:
Remove `attachFusionGroups`, the temporary `pjrtx.fusion.plan` handoff support,
and the unused string/dictionary parsing helpers that existed for older fusion
plan representations. `planFusionFromCandidates` is now the only public MLIR
fusion-planning transition.

Reasoning:
The compiler should not carry a compatibility path that lets graph-side fusion
facts bypass MLIR candidate ownership. Keeping one transition makes the code and
explanation model easier to audit: candidate discovery creates the facts,
fusion planning decides on those facts, and extraction reads those same facts.

Consequences:
This was the intermediate step before the full graph-only planner removal.
Future fusion work should grow candidate discovery and replace the attribute
shim with real dialect attrs/ops, not restore a side-channel plan handoff.

Verification:
`bazel test //next/pjrtx/... --test_output=errors` passes.

Links:
- `pjrtx/compiler/mlir_state/package.zig`
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`
- `next/docs/specs/implementation_workplan_v0.md`

## 2026-05-19 - Graph-Only Fusion Planner Removed

Status: accepted

Context:
The old full-Zig fusion planner still provided a compatibility path after MLIR
candidate discovery and `runFusionPlanExternalPass` became the live compiler
path. That kept two explanations for fusion decisions: one from graph-only Zig
records and one from MLIR-owned candidate facts.

Decision:
Remove the graph-only `FusionPlan`, `planFusion`, `writeFusionPlan`, and
graph-only `planV0TraceReport` path. Full trace planning now requires a live
`MlirSession`; callers that start after graph import use
`planV0TraceReportFromMlirSession` so fusion decisions still come from MLIR.
Compiler pass contracts now distinguish `fusion_candidates` from
`fusion_decisions` instead of using the old generic `fusion_plan` artifact
label.

Reasoning:
PjRTx cannot become explainable from PJRT to generated kernels if major
performance decisions can bypass the compiler IR. The MLIR state machine is the
compiler truth; Zig records are extracted report and runtime views. Removing
the compatibility planner reduces code debt and forces future fusion,
placement, tiling, collective, and codegen work to grow the MLIR path.

Consequences:
Tests that need a full planned trace must create an `MlirSession` from the
fixture text and pass it through the trace planner. There is intentionally no
fallback path for fusion decisions outside the MLIR candidate-to-decision
transition.

Verification:
`bazel test //next/pjrtx/... --test_output=errors` passes.

Links:
- `pjrtx/compiler/compiler.zig`
- `pjrtx/compiler/mlir_state/package.zig`
- `next/pjrtx/vertical_slice/lowering_test.zig`
- `next/docs/specs/compile_pipeline_v0.md`
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`
- `next/docs/specs/implementation_workplan_v0.md`

## 2026-05-19 - Placement Facts Enter MLIR State

Status: accepted

Context:
After removing the graph-only fusion planner, placement was the next compiler
middle fact still flowing straight from Zig planning into the trace report.
That preserved useful V0 behavior, but it kept layout, tile, and memory-space
decisions beside the MLIR state machine instead of inside it.

Decision:
Add `placement_planned` to the MLIR module state machine. The V0 placement
planner now produces temporary records, commits them to
`pjrtx.placement.records`, runs `runPlacementPlanExternalPass` to verify the
MLIR facts, stamps `pjrtx.placement_plan.pass`, and extracts the final
`PlacementRecord` report view from MLIR.

Reasoning:
Placement is where target memory hierarchy starts to become performance truth:
whole-tensor Metal/MLS placement, bounded NPU local-SRAM tiles, result memory
spaces, and later DMA/prefetch constraints. Those decisions must be visible at
an MLIR boundary before cost, lowering, kernel codegen, schedule, and backend
binding consume them.

Consequences:
The local Zig placement planner is now a temporary producer for MLIR placement
facts, not the report source of truth. The next step should move tile selection
and memory-pressure calculation themselves into MLIR passes or dialect attrs,
then add collective plan/group/channel facts to the same state-machine path.

Verification:
`bazel test //next/pjrtx/... --test_output=errors` passes.

Links:
- `pjrtx/compiler/mlir_state/package.zig`
- `pjrtx/compiler/compiler.zig`
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`
- `next/docs/specs/mlir_state_machine_compiler_v0.md`
- `next/docs/specs/implementation_workplan_v0.md`

## 2026-05-19 - Backend Executable Calls Enter MLIR State

Status: accepted

Context:
Executable readiness entered MLIR, but concrete backend executable calls were
still only derived from the extracted report by `//next/pjrtx/backend`.

Decision:
Add `backend_executable_planned` to the MLIR module state machine and introduce
`//next/pjrtx/backend:mlir_bridge`. Backend planning still owns the policy for
turning a verified binding into backend calls, then the bridge commits the
result to `pjrtx.backend.executable`, runs
`runBackendExecutablePlanExternalPass`, stamps `pjrtx.backend_executable.pass`,
and verifies the plan against the selected target and executable contract.

Reasoning:
The compiler must not import backend implementation details, but generated or
selected backend calls must still be explainable in the MLIR state path. A
separate bridge target keeps ownership clean: backend derives the calls, MLIR
records the facts, and tests can inspect the exact transition.

Consequences:
Metal executable-call planning now has a durable MLIR fact boundary before MLS
kernel graph planning. The next slice should move Metal/MLS kernel graph nodes
or allocator reservations into MLIR, then add extraction helpers when runtime
starts consuming the MLIR facts directly.

Verification:
`bazel test //next/pjrtx/... --test_output=errors` passes.

Links:
- `pjrtx/backend/mlir_bridge.zig`
- `pjrtx/backend/BUILD.bazel`
- `pjrtx/compiler/mlir_state/package.zig`
- `next/pjrtx/vertical_slice/metal_bridge_test.zig`
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`
- `next/docs/specs/mlir_state_machine_compiler_v0.md`
- `next/docs/specs/implementation_workplan_v0.md`

## 2026-05-19 - Metal Kernel Graph Enters MLIR State

Status: accepted

Context:
Backend executable calls entered MLIR, but the Metal/MLS kernel graph topology
was still only derived inside `//next/pjrtx/backend` immediately before summary
printing.

Decision:
Add `backend_kernel_graph_planned` to the MLIR module state machine. The
backend MLIR bridge now commits the Metal/MLS kernel graph to
`pjrtx.backend.kernel_graph`, including nodes, output tensor descriptors,
compact attribute tags, and value-flow edges. `runBackendKernelGraphExternalPass`
verifies the graph against the committed backend executable plan and stamps
`pjrtx.backend_kernel_graph.pass`.

Reasoning:
Kernel graph topology is where generated or selected backend calls start to
look like a real command-buffer/codegen problem. It needs to be visible in MLIR
before allocator joins, profiling joins, or Metal command generation can claim
an explainable path from PJRT input to kernels.

Consequences:
The Metal vertical slice now proves both backend executable calls and Metal/MLS
kernel graph facts in the live MLIR session. The next slice should move runtime
allocator reservations or richer kernel attributes into MLIR, then eventually
replace compact attribute tags with generated-kernel IR facts.

Verification:
`bazel test //next/pjrtx/... --test_output=errors` passes.

Links:
- `pjrtx/backend/mlir_bridge.zig`
- `pjrtx/compiler/mlir_state/package.zig`
- `next/pjrtx/vertical_slice/metal_bridge_test.zig`
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`
- `next/docs/specs/mlir_state_machine_compiler_v0.md`
- `next/docs/specs/implementation_workplan_v0.md`

## 2026-05-19 - Runtime Allocation Enters MLIR State

Status: accepted

Context:
Kernel graph topology entered MLIR, but runtime allocator reservations were
still only derived inside `//next/pjrtx/runtime` before stream planning and profile
summaries.

Decision:
Add `runtime_allocation_planned` to the MLIR module state machine and introduce
`//next/pjrtx/runtime:mlir_bridge`. Runtime planning still owns capacity checks,
transfer-edge checks, buffer placement, lifetimes, and peak live bytes. The
bridge commits the verified result to `pjrtx.runtime.allocation` and
`runRuntimeAllocationExternalPass` stamps `pjrtx.runtime_allocation.pass`.

Reasoning:
Allocation is a performance and correctness boundary: memory-space choice,
bytes, lifetime, and command use determine whether peak pressure fits the
target and whether profiling can explain memory limits. Those facts need a
durable MLIR boundary before stream planning, command submission, or measured
profile joins.

Consequences:
The NPU execution vertical slice now proves backend executable calls and
runtime allocation reservations in the live MLIR session. The next slice should
move stream/event planning into MLIR, then join profile events against commands,
allocations, and backend calls from MLIR facts instead of report-only views.

Verification:
`bazel test //next/pjrtx/... --test_output=errors` passes.

Links:
- `pjrtx/runtime/mlir_bridge.zig`

## 2026-05-19 - Runtime Profile Joins Enter MLIR State

Status: accepted

Context:
Runtime profile events were in MLIR, but the relation between an observed event
and the schedule command, lowering record, or explain record it measured still
had to be inferred from report-side joins.

Decision:
Add `runtime_profile_joined` to the MLIR module state machine. The runtime MLIR
bridge now commits `pjrtx.runtime.profile_joins` rows that name subject kind,
subject ID, optional command ID, instruction IDs, and profile event IDs.

Reasoning:
Observed performance must be explainable from the IR boundary itself. A user
should be able to trace from profile event to lowering decision to generated or
selected kernel without trusting a separate report reconstruction.

Consequences:
The NPU execution vertical slice now proves the runtime MLIR chain through
allocation, streams, profile events, and profile joins. Backend-call-specific
join rows are added by the follow-up backend profile join slice without
reversing the compiler/backend dependency boundary.

Verification:
`bazel test //next/pjrtx/... --test_output=errors` passes.

Links:
- `pjrtx/compiler/mlir_state/package.zig`
- `pjrtx/runtime/mlir_bridge.zig`
- `next/pjrtx/vertical_slice/execution_test.zig`
- `pjrtx/runtime/BUILD.bazel`
- `pjrtx/compiler/mlir_state/package.zig`
- `next/pjrtx/vertical_slice/execution_test.zig`
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`
- `next/docs/specs/mlir_state_machine_compiler_v0.md`
- `next/docs/specs/implementation_workplan_v0.md`

## 2026-05-19 - Backend Call Profile Joins Enter MLIR State

Status: accepted

Context:
Runtime profile joins connected observed events to schedule commands, lowering
records, and explain records, but concrete backend executable calls were still
joined to profile events only in backend summary text.

Decision:
Add `backend_profile_joined` to the MLIR module state machine. The backend MLIR
bridge now commits `pjrtx.backend.profile_joins` rows that name backend call
index, command ID, instruction IDs, and the matching profile event ID.

Reasoning:
The generated or selected kernel call is the performance-critical boundary.
Keeping call-to-event joins in MLIR lets a user trace from hardware observation
to backend executable call to lowering region without depending on a separate
report reconstruction.

Consequences:
The NPU execution vertical slice now proves call-level joins for the matmul and
elementwise-fusion backend calls. The compiler still does not import backend
internals; the backend bridge owns the concrete call shape and commits the MLIR
fact after runtime profile joins exist.

Verification:
`bazel test //next/pjrtx/... --test_output=errors` passes.

Links:
- `pjrtx/compiler/mlir_state/package.zig`
- `pjrtx/backend/mlir_bridge.zig`
- `next/pjrtx/vertical_slice/execution_test.zig`
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`
- `next/docs/specs/mlir_state_machine_compiler_v0.md`
- `next/docs/specs/implementation_workplan_v0.md`

## 2026-05-19 - Profile Join Facts Become Extractable

Status: accepted

Context:
Runtime and backend profile joins were verified MLIR attributes, but tests still
mostly consumed them through state summaries. That made the facts visible, but
not yet reusable as typed MLIR-extracted views.

Decision:
Add `extractRuntimeProfileJoins` and `extractBackendProfileJoins` to
`//next/pjrtx/compiler/mlir_state`. Runtime join extraction accepts
`runtime_profile_joined` and later `backend_profile_joined` states because the
runtime join attribute remains valid after backend joins are committed.

Reasoning:
The migration rule is `MLIR fact -> extracted Zig view`. Profile joins are a
core explainability relation, so they need the same typed readback path as
fusion, placement, schedule, backend binding, and executable contract facts.

Consequences:
The NPU execution vertical slice now checks extracted runtime and backend
profile join objects directly. Later report/explanation code can consume the
MLIR-extracted joins instead of rebuilding relations from the original trace
objects or summary strings.

Verification:
`bazel test //next/pjrtx/... --test_output=errors` passes.

Links:
- `pjrtx/compiler/mlir_state/package.zig`
- `next/pjrtx/vertical_slice/execution_test.zig`
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`
- `next/docs/specs/implementation_workplan_v0.md`

## 2026-05-19 - Extracted MLIR Facts Feed Reports

Status: accepted

Context:
Runtime allocation, stream, profile, backend executable, kernel graph, and
backend profile-join facts were committed to MLIR and extractable, but several
human summaries still rendered from the original runtime/backend Zig planner
objects. That made the reports look MLIR-backed while leaving old objects as a
hidden source of truth.

Decision:
Add typed `mlir_state` summary writers for extracted runtime allocation,
runtime streams, runtime execution, lowering profiles, hardware utilization,
backend executable calls, backend kernel graphs, and backend call profiles.
The execution vertical slice now renders its stable golden report from
extracted MLIR allocation, stream, and profile-event facts. The backend call
profile summary uses extracted backend executable calls and backend-call
profile joins. The Metal slice renders executable and kernel graph summaries
from extracted MLIR facts.

Reasoning:
Committing facts to MLIR is only half of the architecture. The next layer must
consume the extracted facts so missing MLIR state is felt immediately by tests
and reports. Schedule, target, and cost context initially still came from the
compiler trace while those views were being migrated, but committed
runtime/backend facts should be read back from MLIR before they are printed.

Consequences:
The old Zig planner objects are less able to mask missing extraction coverage.
The remaining work is to continue converting compiler-side schedule, cost,
target, lowering, fusion, tiling, Shardy, and codegen-region views into MLIR
facts so reports can eventually join almost entirely over extracted state.

Verification:
`bazel test //next/pjrtx/compiler/mlir_state:unit_tests //next/pjrtx/vertical_slice:execution_test //next/pjrtx/vertical_slice:metal_bridge_test --test_output=errors` passes.

Links:
- `pjrtx/compiler/mlir_state/package.zig`
- `next/pjrtx/vertical_slice/execution_test.zig`
- `next/pjrtx/vertical_slice/metal_bridge_test.zig`
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`
- `next/docs/specs/implementation_workplan_v0.md`

## 2026-05-19 - Schedule Facts Feed Runtime Reports

Status: accepted

Context:
Schedule commands were already committed to MLIR and extracted immediately
after scheduling, but the runtime report writers still iterated over
`TraceReport.schedule_commands` when joining extracted allocation, stream, and
profile facts. That kept command order and dependency membership partly hidden
behind the old trace object.

Decision:
Allow schedule command and overlap extraction from `scheduled` or any later V0
state. Update runtime execution, lowering-profile, and hardware-utilization
fact summary writers to receive extracted schedule commands. The execution
vertical slice now extracts schedule commands after `backend_profile_joined`
and uses those extracted commands for stable report rendering.

Reasoning:
Schedule is the spine that ties host/device IO, backend execution, streams,
events, allocation lifetimes, and profile observations together. If runtime
reports consume extracted runtime facts but still take command order from the
old report, the report is not yet telling the full MLIR-state story.

Consequences:
Committed schedule facts now survive and remain reportable through the full V0
backend/runtime/profile sequence. At this point in the log, remaining trace
dependencies in these writers were narrower: target, graph values, cost ledger,
and lowering records.

Verification:
`bazel test //next/pjrtx/compiler/mlir_state:unit_tests //next/pjrtx/vertical_slice:execution_test --test_output=errors` passes.

Links:
- `pjrtx/compiler/mlir_state/package.zig`
- `next/pjrtx/vertical_slice/execution_test.zig`
- `next/pjrtx/vertical_slice/BUILD.bazel`
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`
- `next/docs/specs/implementation_workplan_v0.md`

## 2026-05-19 - Full Target Specs Enter MLIR State

Status: accepted

Context:
Target attachment stored target name, kind, fingerprint, replicas, and
partitions in MLIR, but hardware summaries still read memory spaces, transfer
edges, execution units, and dtype rates from the original `TraceReport.target`.

Decision:
Commit the full V0 hardware contract as `pjrtx.target.spec` during target
attachment. Add `extractTargetDescription` to rebuild a typed
`core.TargetDescription` from MLIR and validate it with the core target
validator. Hardware-utilization fact summaries now receive the extracted target
instead of reading `TraceReport.target`.

Reasoning:
Performance explainability starts with the hardware model. Bandwidths, memory
capacities, execution-unit kinds, dtype rates, and transfer edges must be
visible in MLIR state so predicted roofline and allocation pressure can be
traced back to the same hardware spec that drove lowering decisions.

Consequences:
The hardware summary no longer depends on the old target object for memory and
execution-unit facts. Cost ledger, lowering records, graph values, and memory
traffic records are still report-side dependencies and should move into MLIR
facts in later slices.

Verification:
`bazel test //next/pjrtx/compiler/mlir_state:unit_tests //next/pjrtx/vertical_slice:execution_test --test_output=errors` passes.

Links:
- `pjrtx/compiler/mlir_state/package.zig`
- `next/pjrtx/vertical_slice/execution_test.zig`
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`
- `next/docs/specs/implementation_workplan_v0.md`

## 2026-05-19 - Backend Call Profiles Use Extracted Runtime Events

Status: accepted

Context:
Backend call-profile summaries used extracted backend executable calls and
backend-call profile joins, but the observed bytes, ops, and event IDs still
came from `TraceReport.profile_events`.

Decision:
Change the backend call-profile fact writer to consume extracted
`RuntimeProfileEventFact` rows. The summary now joins extracted backend calls,
extracted runtime profile events, and extracted backend profile joins.

Reasoning:
Profile observations are runtime facts committed to MLIR. Once they are
extractable, backend reports should use that extracted profile view instead of
the original runtime-owned report array.

Consequences:
Observed backend-call metrics no longer depend on the old runtime profile array
for this summary path. Predicted metrics still depend on the compiler cost
ledger and target rates, which should become explicit MLIR facts in a later
slice.

Verification:
`bazel test //next/pjrtx/compiler/mlir_state:unit_tests //next/pjrtx/vertical_slice:execution_test --test_output=errors` passes.

Links:
- `pjrtx/compiler/mlir_state/package.zig`
- `next/pjrtx/vertical_slice/execution_test.zig`
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`
- `next/docs/specs/implementation_workplan_v0.md`

## 2026-05-19 - Backend Executable And Kernel Graph Facts Become Extractable

Status: accepted

Context:
Backend executable calls and Metal/MLS kernel graph facts were committed to
MLIR and visible in summaries, but typed tests still mainly trusted the
backend-owned Zig plans after commit.

Decision:
Add `extractBackendExecutablePlan` and `extractBackendKernelGraph` to
`//next/pjrtx/compiler/mlir_state`. Backend executable extraction accepts
`backend_executable_planned` and later backend/runtime states. Kernel graph
extraction accepts `backend_kernel_graph_planned` and later runtime/profile
states.

Reasoning:
Concrete backend calls and generated graph nodes are the handoff between
lowering/codegen decisions and runtime execution. They must be readable back
from MLIR before reports can stop treating backend planner outputs as hidden
truth.

Consequences:
The NPU execution slice now validates extracted backend executable calls after
the final backend profile state. The Metal bridge slice validates extracted
executable calls plus kernel graph nodes, output tensor descriptors, and
value-flow edges from MLIR.

Verification:
`bazel test //next/pjrtx/... --test_output=errors` passes.

Links:
- `pjrtx/compiler/mlir_state/package.zig`
- `next/pjrtx/vertical_slice/execution_test.zig`
- `next/pjrtx/vertical_slice/metal_bridge_test.zig`
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`
- `next/docs/specs/implementation_workplan_v0.md`

## 2026-05-19 - Runtime Allocation And Stream Facts Become Extractable

Status: accepted

Context:
Runtime allocation and stream facts were committed to MLIR and visible in state
summaries, but typed test and report code still consumed the original runtime
planner outputs for allocator reservations and event sequencing.

Decision:
Add `extractRuntimeAllocationPlan` and `extractRuntimeStreamPlan` to
`//next/pjrtx/compiler/mlir_state`. Allocation extraction accepts
`runtime_allocation_planned` and later runtime/profile states. Stream
extraction accepts `runtime_stream_planned` and later profile states.

Reasoning:
Allocator reservations, buffer-use access, peak memory, stream assignment, and
event waits are performance and correctness facts. They need typed MLIR
readback before summaries can stop treating the original runtime planner
objects as hidden truth.

Consequences:
The NPU execution vertical slice now validates allocation and stream payloads
from extracted MLIR facts after the final backend profile state. This extends
the readback chain below profile events and joins.

Verification:
`bazel test //next/pjrtx/... --test_output=errors` passes.

Links:
- `pjrtx/compiler/mlir_state/package.zig`
- `next/pjrtx/vertical_slice/execution_test.zig`
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`
- `next/docs/specs/implementation_workplan_v0.md`

## 2026-05-19 - Runtime Profile Events Become Extractable

Status: accepted

Context:
Runtime profile events were verified MLIR attributes and visible in summaries,
but typed code still consumed the original runtime trace rows when it needed
event payloads.

Decision:
Add `extractRuntimeProfileEvents` to `//next/pjrtx/compiler/mlir_state`. Extraction
accepts `runtime_profiled`, `runtime_profile_joined`, and
`backend_profile_joined` states because later profile joins do not invalidate
the original observation rows.

Reasoning:
Observed bytes, ops, status, synchronization, command IDs, and instruction IDs
are performance evidence. They must follow the same `MLIR fact -> extracted
Zig view` rule as the explanation joins that reference them.

Consequences:
The NPU execution vertical slice now checks extracted profile events directly
from MLIR, including a backend-execute region event. Later report code can read
profile observations from MLIR instead of retaining a parallel runtime-owned
source of truth.

Verification:
`bazel test //next/pjrtx/... --test_output=errors` passes.

Links:
- `pjrtx/compiler/mlir_state/package.zig`
- `next/pjrtx/vertical_slice/execution_test.zig`
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`
- `next/docs/specs/implementation_workplan_v0.md`

## 2026-05-19 - Runtime Streams Enter MLIR State

Status: accepted

Context:
Runtime allocation entered MLIR, but stream/event planning still lived only in
`//next/pjrtx/runtime` and report summaries.

Decision:
Add `runtime_stream_planned` to the MLIR module state machine. The runtime MLIR
bridge now commits stream steps to `pjrtx.runtime.streams`, including command
ID, stream ID, wait events, start event, and done event.
`runRuntimeStreamExternalPass` verifies the records and stamps
`pjrtx.runtime_stream.pass`.

Reasoning:
Streams and events are the first executable ordering contract after allocation.
They must be explainable before command submission, profiling joins, or overlap
experiments can claim performance behavior.

Consequences:
The NPU execution vertical slice now proves allocation and stream/event facts in
the live MLIR session. The next slice should move synthetic profile events or
profile joins into MLIR so observed metrics can point back to commands,
allocations, backend calls, and stream events.

Verification:
`bazel test //next/pjrtx/... --test_output=errors` passes.

Links:
- `pjrtx/runtime/mlir_bridge.zig`
- `pjrtx/compiler/mlir_state/package.zig`
- `next/pjrtx/vertical_slice/execution_test.zig`
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`
- `next/docs/specs/mlir_state_machine_compiler_v0.md`
- `next/docs/specs/implementation_workplan_v0.md`

## 2026-05-19 - Runtime Profile Events Enter MLIR State

Status: accepted

Context:
Stream/event facts entered MLIR, but synthetic profile events and observed
metrics were still only present in profiled trace reports.

Decision:
Add `runtime_profiled` to the MLIR module state machine. The runtime MLIR
bridge now commits profile events to `pjrtx.runtime.profile_events`, including
event ID, command join, instruction join, kind, timing, bytes, logical ops,
status, and forced synchronization. `runRuntimeProfileExternalPass` verifies
the rows and stamps `pjrtx.runtime_profile.pass`.

Reasoning:
Profile events are where predicted execution meets observed behavior. They
must join back to MLIR command, stream, allocation, lowering, and backend-call
facts so performance explanations do not depend on report-only side channels.

Consequences:
The NPU execution vertical slice now proves the runtime MLIR chain through
allocation, streams, and profile events. The next slice should either extract
these runtime facts back from MLIR for summaries, or add profile join facts that
explicitly connect events to lowerings, backend calls, memory traffic, and
explain records.

Verification:
`bazel test //next/pjrtx/... --test_output=errors` passes.

Links:
- `pjrtx/runtime/mlir_bridge.zig`
- `pjrtx/compiler/mlir_state/package.zig`
- `next/pjrtx/vertical_slice/execution_test.zig`
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`
- `next/docs/specs/mlir_state_machine_compiler_v0.md`
- `next/docs/specs/implementation_workplan_v0.md`

## 2026-05-19 - Executable Readiness Enters MLIR State

Status: accepted

Context:
Backend binding facts entered MLIR, but the final “this module has enough
verified compiler facts to become executable-shaped” decision still happened
only in the Zig executable view path.

Decision:
Add `executable_ready` to the MLIR module state machine. The V0 compiler now
commits a small `pjrtx.executable.contract` with selected target kind, schedule
command count, backend binding count, and kernel codegen record count, runs
`runExecutableReadyExternalPass`, stamps `pjrtx.executable_ready.pass`, and
requires the contract to match the verified MLIR schedule, backend binding, and
codegen facts.

Reasoning:
Executable readiness is a compiler-owned contract, not a backend fallback. Even
before generated kernel handles, allocator reservations, or backend graph calls
become first-class MLIR facts, the state machine should explicitly prove that
the executable view is consuming a coherent set of verified compiler facts.

Consequences:
The Zig executable view still performs structural report validation, schedule
verification, and backend binding verification, but it now runs after MLIR has
advanced to `executable_ready`. The next slice should move backend executable
calls, Metal/MLS kernel graph nodes, or allocator reservations into MLIR rather
than only deriving them from extracted reports.

Verification:
`bazel test //next/pjrtx/... --test_output=errors` passes.

Links:
- `pjrtx/compiler/mlir_state/package.zig`
- `pjrtx/compiler/compiler.zig`
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`
- `next/docs/specs/mlir_state_machine_compiler_v0.md`
- `next/docs/specs/implementation_workplan_v0.md`

## 2026-05-19 - Backend Binding Facts Enter MLIR State

Status: accepted

Context:
Schedule and kernel codegen facts now entered MLIR before reports consumed
them, but backend binding still flowed directly from Zig planning into the
trace report and executable view.

Decision:
Add `backend_bound` to the MLIR module state machine. The V0 backend binder now
produces temporary command-to-backend records, commits them to
`pjrtx.backend.bindings`, runs `runBackendBindingExternalPass` to verify the
MLIR facts, stamps `pjrtx.backend_binding.pass`, and extracts the final
`BackendBinding` report view from MLIR.

Reasoning:
Backend binding is where compiler-middle decisions become a concrete target
contract. It must consume verified schedule/codegen facts, preserve source
instruction and cost IDs, and expose the selected backend operation before
Metal/MLS graph expansion, executable creation, runtime allocation, and profile
joins can proceed.

Consequences:
The local Zig backend binder is now a temporary producer for MLIR backend
binding facts, not the report source of truth. The next step should move
executable-readiness, generated kernel handles, allocator reservations, and
backend graph calls into the MLIR state path.

Verification:
`bazel test //next/pjrtx/... --test_output=errors` passes.

Links:
- `pjrtx/compiler/mlir_state/package.zig`
- `pjrtx/compiler/compiler.zig`
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`
- `next/docs/specs/mlir_state_machine_compiler_v0.md`
- `next/docs/specs/implementation_workplan_v0.md`

## 2026-05-19 - Schedule Facts Enter MLIR State

Status: accepted

Context:
Kernel codegen facts now entered MLIR before tile legality and backend binding,
but the executable command sequence and overlap records still flowed directly
from Zig planning to reports and backend binding.

Decision:
Add `scheduled` to the MLIR module state machine. The V0 schedule builder now
produces temporary command and overlap records, commits them to
`pjrtx.schedule.commands` and `pjrtx.schedule.overlaps`, runs
`runSchedulePlanExternalPass` to verify the MLIR facts, stamps
`pjrtx.schedule_plan.pass`, and extracts the final `ScheduleCommand` and
`ScheduleOverlapRecord` report views from MLIR.

Reasoning:
The schedule is where data dependencies, stream assignment, transfer/compute
ordering, and future overlap opportunities become executable constraints. Those
facts must be visible before backend binding, runtime allocation, stream
submission, and profiling join against them.

Consequences:
The local Zig schedule builder is now a temporary producer for MLIR schedule
facts, not the report source of truth. The next step should move backend
binding or executable-readiness facts into MLIR, then replace temporary schedule
attributes with real stream/event/bufferization state.

Verification:
`bazel test //next/pjrtx/... --test_output=errors` passes.

Links:
- `pjrtx/compiler/mlir_state/package.zig`
- `pjrtx/compiler/compiler.zig`
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`
- `next/docs/specs/mlir_state_machine_compiler_v0.md`
- `next/docs/specs/implementation_workplan_v0.md`

## 2026-05-19 - Kernel Codegen Facts Enter MLIR State

Status: accepted

Context:
Fusion, placement, and collective facts now entered MLIR before the trace report
consumed them. Kernel codegen records still flowed directly from Zig planning
to tile legality, backend binding, Metal/MLS graph planning, and reports.

Decision:
Add `codegen_planned` to the MLIR module state machine. The V0 kernel codegen
planner now produces temporary records, commits them to `pjrtx.codegen.records`,
runs `runKernelCodegenPlanExternalPass` to verify the MLIR facts, stamps
`pjrtx.codegen_plan.pass`, and extracts the final `KernelCodegenRecord` report
view from MLIR.

Reasoning:
Codegen is the bridge from lowering to generated or selected kernels. It must
carry backend kind, kernel kind, operation name, value-flow shape, tile and
memory-space decisions, memory pressure, expected execution unit, source
instructions, costs, and memory traffic before tile legality or backend binding
can claim an executable path.

Consequences:
The local Zig codegen planner is now a temporary producer for MLIR codegen
facts, not the report source of truth. The next step should move tile legality
and schedule/overlap facts into MLIR, then replace the temporary codegen
attribute records with real kernel IR or dialect attrs.

Verification:
`bazel test //next/pjrtx/... --test_output=errors` passes.

Links:
- `pjrtx/compiler/mlir_state/package.zig`
- `pjrtx/compiler/compiler.zig`
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`
- `next/docs/specs/mlir_state_machine_compiler_v0.md`
- `next/docs/specs/implementation_workplan_v0.md`

## 2026-05-19 - Collective Decisions Enter MLIR State

Status: accepted

Context:
Placement now entered MLIR before the report consumed it, but collective
planning still flowed directly from Zig planner output to the trace report. That
left all-reduce rejection/no-op decisions outside the compiler state machine
even though collectives are correctness-, topology-, and performance-critical.

Decision:
Add `collectives_planned` to the MLIR module state machine. The V0 collective
planner now produces a temporary decision record, commits it to
`pjrtx.collective.records`, runs `runCollectivePlanExternalPass` to verify the
MLIR facts, stamps `pjrtx.collective_plan.pass`, and extracts the final
`CollectivePlanRecord` report view from MLIR.

Reasoning:
Collective lowering must never become a backend or runtime fallback. Even when
V0 records `algorithm=none` or rejects executable collectives, the compiler
should expose the decision with checked participant counts, selected or rejected
algorithm, estimated traffic, and a reason before schedule construction.

Consequences:
The local Zig collective planner is now a temporary producer for MLIR collective
facts, not the report source of truth. The next step should move group/channel
facts, token constraints, algorithm selection, and eventual collective command
lowering into MLIR passes or dialect attrs.

Verification:
`bazel test //next/pjrtx/... --test_output=errors` passes.

Links:
- `pjrtx/compiler/mlir_state/package.zig`
- `pjrtx/compiler/compiler.zig`
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`
- `next/docs/specs/mlir_state_machine_compiler_v0.md`
- `next/docs/specs/implementation_workplan_v0.md`

## 2026-05-19 - Performance Facts Enter MLIR State

Status: accepted

Context:
Target, schedule, runtime, backend executable, and profile facts were already
being committed to MLIR and read back as typed views. Cost ledger entries and
memory traffic records still came from the original Zig trace arrays, which made
predicted bytes, logical ops, and roofline memory terms a hidden source of
truth.

Decision:
Add `performance_modeled` to the MLIR module state machine between
`collectives_planned` and `codegen_planned`. The compiler now commits cost
ledger and memory traffic facts to `pjrtx.performance.cost_ledger` and
`pjrtx.performance.memory_traffic`, verifies them through
`runPerformanceFactsExternalPass`, extracts typed `CostLedgerEntry` and
`MemoryTrafficRecord` views, and feeds runtime/lowering/hardware/backend-call
summaries from those extracted views.

Reasoning:
Performance explanation has to be vertical: source instruction, lowering,
backend call, memory space, target bandwidth, dtype peak, predicted work, and
observed profile event must meet at explicit boundaries. MLIR state is the right
boundary because it can be verified, summarized, and later replaced by real
passes/dialect attrs without preserving the old Zig planner as truth.

Consequences:
Kernel codegen now requires `performance_modeled`, not just
`collectives_planned`. The temporary Zig cost model remains a producer, but the
report layer consumes MLIR-extracted performance facts. Empty source/location or
approximation strings are sanitized before becoming MLIR string attrs.

Verification:
`bazel test //next/pjrtx/compiler/mlir_state:unit_tests //next/pjrtx/compiler:unit_tests //next/pjrtx/vertical_slice:execution_test --test_output=errors` passes.

Links:
- `pjrtx/compiler/mlir_state/package.zig`
- `pjrtx/compiler/compiler.zig`
- `next/pjrtx/vertical_slice/execution_test.zig`
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`
- `next/docs/specs/mlir_state_machine_compiler_v0.md`
- `next/docs/specs/implementation_workplan_v0.md`

## 2026-05-19 - Lowering Regions Enter MLIR State

Status: accepted

Context:
Cost, memory traffic, and later schedule/backend/profile facts had started to
enter MLIR state, but the lowering-region boundary itself was still mostly a
Zig trace fact. That made performance facts point at lowering IDs whose source
of truth had not passed through the same MLIR state verifier.

Decision:
Add `lowering_planned` between `collectives_planned` and
`performance_modeled`. The V0 lowering planner now produces temporary lowering
records, commits them to `pjrtx.lowering.records`, runs
`runLoweringPlanExternalPass` to verify graph-instruction and cost-ledger
references, stamps `pjrtx.lowering_plan.pass`, and extracts the final
`LoweringRecord` report view from MLIR before memory traffic and performance
modeling run.

Reasoning:
Lowering regions are the unit that kernels, memory traffic, schedules,
allocation, stream events, roofline rows, and profile joins all need to agree
on. If that unit stays outside MLIR while later facts enter MLIR, the compiler
would still have a hidden source of truth at the most important performance
boundary.

Consequences:
The local Zig lowering planner is now only a temporary producer for MLIR
lowering facts. The next compiler-middle work should replace temporary
committed records with real MLIR lowering-region formation from fusion,
placement, Shardy-aware partitioning, collective legality, tiling, and backend
codegen constraints.

Verification:
`bazel test //next/pjrtx/compiler/mlir_state:unit_tests //next/pjrtx/compiler:unit_tests //next/pjrtx/vertical_slice:execution_test --test_output=errors` passes.
`bazel test //next/pjrtx/... --test_output=errors` passes.

Links:
- `pjrtx/compiler/mlir_state/package.zig`
- `pjrtx/compiler/compiler.zig`
- `next/pjrtx/vertical_slice/execution_test.zig`
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`
- `next/docs/specs/mlir_state_machine_compiler_v0.md`
- `next/docs/specs/compiler_pass_pipeline_v0.md`
- `next/docs/specs/implementation_workplan_v0.md`

## 2026-05-19 - Lowering Region Facts Join Fusion, Tile, Memory, And Codegen Intent

Status: accepted

Context:
`lowering_planned` made the final lowering records pass through MLIR before
performance modeling. The next risk was that the region's surrounding compiler
evidence still had to be inferred from separate Zig arrays: fusion groups,
placement records, tiles, memory spaces, and future codegen-region intent.

Decision:
Add `pjrtx.lowering.region_facts` beside `pjrtx.lowering.records`.
`runLoweringPlanExternalPass` now verifies both attributes before advancing to
`lowering_planned`. The extracted `LoweringRegionFact` view carries the
lowering ID, optional fusion-group index, placement record coverage, logical
tile shape, result memory, tile memory, codegen-region decision, and reason.

Reasoning:
Lowering-region formation should be the place where fusion, tiling,
memory-space planning, and backend codegen intent first meet. Keeping those
links explicit in MLIR prevents later memory traffic, schedule, kernel graph,
and profiling code from quietly reconstructing why a region exists.

Consequences:
The V0 Zig lowering planner is still the temporary producer, but it now has to
produce an MLIR-verifiable region-analysis fact, not just a report row. The
next step is to move the actual region formation into MLIR passes so
`LoweringRegionFact` becomes an extracted view of real dialect facts rather
than a committed temporary bridge.

Verification:
`bazel test //next/pjrtx/compiler/mlir_state:unit_tests //next/pjrtx/compiler:unit_tests //next/pjrtx/vertical_slice:execution_test --test_output=errors` passes.

Links:
- `pjrtx/compiler/mlir_state/package.zig`
- `pjrtx/compiler/compiler.zig`
- `next/pjrtx/vertical_slice/execution_test.zig`
- `next/pjrtx/vertical_slice/lowering_test.zig`
- `next/docs/specs/trace_schema_v0.md`
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`
- `next/docs/specs/compiler_pass_pipeline_v0.md`
- `next/docs/specs/mlir_state_machine_compiler_v0.md`

## 2026-05-19 - Lowering Region Facts Are Derived From MLIR State

Status: accepted

Context:
The first `LoweringRegionFact` slice made fusion, placement, tile, memory, and
codegen intent visible in MLIR, but the compiler-side builder still assembled
those facts from local Zig arrays before commit.

Decision:
Move `LoweringRegionFact` construction into `mlir_state`. `commitLoweringPlan`
now accepts only lowering records and cost-count context. It derives
`pjrtx.lowering.region_facts` from MLIR-visible `pjrtx.fusion.candidates` and
`pjrtx.placement.records`, then verifies the derived facts with
`pjrtx.lowering.records` before `runLoweringPlanExternalPass` advances to
`lowering_planned`.

Reasoning:
This is the first concrete step from MLIR-as-ledger toward MLIR-as-compiler in
the lowering pipeline. The lowering record proposal is still temporary, but the
region-analysis evidence now comes from MLIR state instead of being composed by
the trace builder.

Consequences:
The compiler trace builder no longer owns fusion-to-placement-to-region fact
assembly. Missing MLIR fusion or placement state now fails at
`lowering_region_form`. The next step is to move the executable region record
formation itself into an MLIR pass so even `pjrtx.lowering.records` stop being
Zig-produced.

Verification:
`bazel test //next/pjrtx/compiler/mlir_state:unit_tests //next/pjrtx/compiler:unit_tests //next/pjrtx/vertical_slice:execution_test //next/pjrtx/vertical_slice:lowering_tests --test_output=errors` passes.

Links:
- `pjrtx/compiler/mlir_state/package.zig`
- `pjrtx/compiler/compiler.zig`
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`
- `next/docs/specs/compiler_pass_pipeline_v0.md`
- `next/docs/specs/mlir_state_machine_compiler_v0.md`

## 2026-05-19 - MLIR Pass Runner Boundary Materialized

Status: accepted

Context:
The compiler had many Zig external MLIR passes, but each pass runner repeated
the same pass-manager lifecycle in `mlir_state/package.zig`: create manager,
enable verifier, add owned pass, run on module, read the result, destroy the
manager, and clear the session slot. That made the code look like pass-manager
plumbing with callbacks nearby, instead of an explicit pass boundary.

Decision:
Add `pjrtx/compiler/mlir_state/passes.zig` and route every external pass
execution through `PassRunner.runModulePass`. The package facade still creates
pass-local data and preserves each semantic diagnostic, but MLIR
create/add/run/destroy ownership now has one implementation point. The
`mlir_state` Bazel target and README name `passes.zig` as the pass-manager
lifecycle boundary.

Reasoning:
This is a small but important step toward real MLIR pass architecture. It
separates pass execution mechanics from state-machine semantics and gives the
next refactor a natural place to move pass families out of the large package
facade. It also keeps the no-fallback rule honest: a pass-manager failure is a
compiler failure, not a signal to silently use a Zig side path.

Consequences:
`package.zig` remains too large because callback structs, attribute helpers,
and verifier logic still live together. The next split should move pass
callback families behind meaningful files, then continue replacing temporary
bounded attributes with PjRTx dialect attrs/ops and MLIR-native rewrite or
dialect-conversion passes where the C API boundary is too narrow.

Verification:
`bazel test //next/pjrtx/compiler/mlir_state:unit_tests //next/pjrtx/compiler:unit_tests //next/pjrtx/architecture:boundary_test --test_output=errors` passes.

Links:
- `pjrtx/compiler/mlir_state/passes.zig`
- `pjrtx/compiler/mlir_state/package.zig`
- `pjrtx/compiler/mlir_state/BUILD.bazel`
- `pjrtx/compiler/mlir_state/README.md`
- `next/docs/specs/implementation_workplan_v0.md`

## 2026-05-19 - State And Target Pass Family Leaves The Facade

Status: accepted

Context:
After the pass-runner boundary landed, `mlir_state/package.zig` still owned all
external pass callback structs. That kept callback policy, attribute access,
state transitions, extraction, summaries, and tests in one very large file.

Decision:
Add `attrs.zig` for the small MLIR C API attribute helpers needed by pass
families, and move the state-probe plus target-legality external pass callbacks
into `state_target_passes.zig`. The package facade still owns the public
`runExternalStateProbePass` and `runTargetLegalExternalPass` APIs, diagnostics,
and post-pass state checks.

Reasoning:
This is the first real pass-family split. It keeps the public compiler-state
surface stable while proving that pass callbacks can live behind meaningful
files with their own dependencies. Starting with state and target legality is
intentionally conservative because those passes use only basic module
attributes and do not require the larger verifier helpers used by fusion,
lowering, performance, runtime, or backend plan passes.

Consequences:
`package.zig` is smaller but still too large. The next pass-family moves should
extract fusion/candidate passes, then placement/collective/lowering passes,
while moving shared verifier helpers into named files instead of growing a new
central bucket.

Verification:
`bazel test //next/pjrtx/compiler/mlir_state:unit_tests //next/pjrtx/compiler:unit_tests //next/pjrtx/architecture:boundary_test --test_output=errors` passes.

Links:
- `pjrtx/compiler/mlir_state/attrs.zig`
- `pjrtx/compiler/mlir_state/state_target_passes.zig`
- `pjrtx/compiler/mlir_state/package.zig`
- `next/docs/specs/implementation_workplan_v0.md`

## 2026-05-19 - Fusion Pass Family Leaves The Facade

Status: accepted

Context:
State and target legality had been split into a pass-family file, but fusion
remained embedded in `mlir_state/package.zig`. That was the first real
compiler-middle policy pass: it walks StableHLO operations, derives candidate
IDs and pressure metrics, verifies candidate attributes, stamps V0 decisions,
and advances the module to `fusion_planned`.

Decision:
Add `fusion_passes.zig` and move `FusionCandidateDiscoveryExternalPass` plus
`FusionPlanExternalPass` into it. Keep fusion-specific helpers local to that
file: operation walking, tensor byte sizing, candidate array construction,
pressure attribute construction, and V0 decision stamping. The package facade
continues to expose `runFusionCandidateDiscoveryExternalPass` and
`runFusionPlanExternalPass`, preserving diagnostics and state checks.

Reasoning:
Fusion is an architectural boundary, not a random helper cluster. Keeping its
StableHLO pattern matching and pressure math beside its pass callbacks makes
the code easier to audit for performance and correctness. It also prevents the
new split from becoming another central utility layer.

Consequences:
`package.zig` is materially smaller, but placement, collective, lowering,
performance, codegen, schedule, backend, and runtime pass callbacks still need
the same treatment. The next natural family split is placement plus
collectives, followed by lowering/performance/codegen.

Verification:
`bazel test //next/pjrtx/compiler/mlir_state:unit_tests //next/pjrtx/compiler:unit_tests //next/pjrtx/architecture:boundary_test --test_output=errors` passes.

Links:
- `pjrtx/compiler/mlir_state/fusion_passes.zig`
- `pjrtx/compiler/mlir_state/package.zig`
- `pjrtx/compiler/mlir_state/README.md`
- `next/docs/specs/implementation_workplan_v0.md`

## 2026-05-19 - Placement And Collective Pass Family Leaves The Facade

Status: accepted

Context:
After fusion moved out, `mlir_state/package.zig` still carried the placement
and collective external pass callbacks. Those callbacks validate bounded MLIR
record arrays and advance the state machine from `fusion_planned` to
`placement_planned`, then from `placement_planned` to `collectives_planned`.

Decision:
Add `placement_collective_passes.zig` and move `PlacementPlanExternalPass` plus
`CollectivePlanExternalPass` into it. Placement-specific array/tile validation
and collective-specific decision/algorithm/metric validation now live beside
the callbacks that use them. The package facade keeps the public runner
functions, diagnostics, and post-pass state checks.

Reasoning:
Placement and collectives are adjacent compiler-middle boundaries. Keeping
their verifier callbacks together makes the current V0 state progression
clear without making a generic pass dumping ground. It also prepares the next
split: lowering, performance, and codegen can become their own compiler-middle
family.

Consequences:
`package.zig` is smaller again, but still owns typed commit/extract validators,
summary writers, and the larger lowering/performance/runtime/backend callback
families. Future extraction should keep following the same rule: helpers move
with the smallest pass or record family that owns their invariants.

Verification:
`bazel test //next/pjrtx/compiler/mlir_state:unit_tests //next/pjrtx/compiler:unit_tests //next/pjrtx/architecture:boundary_test --test_output=errors` passes. Full `bazel test //next/pjrtx/... --test_output=errors` also passes.

Links:
- `pjrtx/compiler/mlir_state/placement_collective_passes.zig`
- `pjrtx/compiler/mlir_state/package.zig`
- `pjrtx/compiler/mlir_state/README.md`
- `next/docs/specs/implementation_workplan_v0.md`

## 2026-05-19 - MLIR State Gets Its Own Bazel Package

Status: accepted

Context:
`//next/pjrtx/compiler/mlir_state` had become a large implementation target inside
the parent compiler package. That made the Bazel tree less meaningful than the
architecture: compiler orchestration and MLIR state-machine ownership were
separate concepts but shared one package.

Decision:
Move the public `pjrtx/compiler/mlir_state` import into the
`//next/pjrtx/compiler/mlir_state` Bazel package. Split public state-machine and
extracted fact types into `types.zig`, bounded temporary attribute limits into
`limits.zig`, and keep the session/pass/commit/verify/extract implementation
in `package.zig`.

Reasoning:
The package tree should describe ownership. `//next/pjrtx/compiler` orchestrates
compile policy; `//next/pjrtx/compiler/mlir_state` owns MLIR state transitions and
the temporary bounded attribute bridge. This also gives future splits a
natural home, such as attrs, parsers, verifiers, pass callbacks, extraction,
and summary writers.

Consequences:
Existing Zig imports remain `@import("pjrtx/compiler/mlir_state")`, but Bazel
deps now point at `//next/pjrtx/compiler/mlir_state`. The parent compiler package no
longer owns the MLIR-state target. The next shrink should continue within that
subpackage instead of adding more logic to the parent compiler target.

Verification:
`bazel test //next/pjrtx/compiler/mlir_state:unit_tests //next/pjrtx/compiler:unit_tests //next/pjrtx/architecture:boundary_test --test_output=errors` passes.

Links:
- `pjrtx/compiler/mlir_state/BUILD.bazel`
- `pjrtx/compiler/mlir_state/types.zig`
- `pjrtx/compiler/mlir_state/limits.zig`
- `pjrtx/compiler/mlir_state/package.zig`

## 2026-05-19 - Specs Deduplicated Around Canonical Owners

Status: accepted

Context:
The documentation had begun repeating the same architecture in several places:
package boundaries were described in the vision, workplan, harness, coding
policy, and boundary spec; the final MLIR dialect split was described in the
V0 dialect spec, state-machine spec, and final dialect spec. That made future
drift likely.

Decision:
Add a canonical ownership table to `next/docs/specs/README.md` and reduce repeated
sections elsewhere. `package_boundaries_v0.md` is now the single canonical
package graph and no-core migration spec. `final_mlir_dialect_op_pass_architecture_v0.md`
is the single final dialect/op/pass destination. `pjrtx_mlir_dialect_v0.md`
now owns only the current V0 attribute bridge. `mlir_state_machine_compiler_v0.md`
keeps the "MLIR is compiler truth" rule without duplicating the op list.
`implementation_workplan_v0.md` now references the package boundary spec
instead of restating the full layout.

Reasoning:
Each durable rule should have one owner. Workplans should queue work; specs
should define contracts; the vision should explain why the contracts exist.
When detail is needed elsewhere, a short pointer is safer than a copied block.

Consequences:
Future doc changes should first identify the canonical owner. The logbook may
repeat historical context, but current specs should not maintain parallel
versions of package graphs, dialect op lists, pass families, or no-fallback
rules.

Verification:
Documentation sweep no longer finds the repeated package-layout, MLIR-layering,
or shared-core guidance blocks that were collapsed into canonical specs.
`bazel test //next/pjrtx/... --test_output=errors` should still be run after this
docs-only pass.

Links:
- `next/docs/specs/README.md`
- `next/docs/specs/package_boundaries_v0.md`
- `next/docs/specs/final_mlir_dialect_op_pass_architecture_v0.md`
- `next/docs/specs/implementation_workplan_v0.md`
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`

## 2026-05-19 - Package Boundary And Final Dialect Specs Bootstrapped

Status: accepted

Context:
The next correction is structural: `//next/pjrtx/core` became a central vocabulary
bucket. That makes all layers import the same package and weakens the boundary
discipline needed for the final MLIR dialect/op/pass architecture.

Decision:
Add two specs. `package_boundaries_v0.md` defines the no-core package graph,
dependency edges, owner table, migration steps, and boundary-test expectations.
`final_mlir_dialect_op_pass_architecture_v0.md` defines the final PjRTx dialect
families, op families, pass families, state machine, extraction outputs, and
bootstrap acceptance criteria. Add README intent files for the future target,
MLIR, dialect, compiler facts, compiler passes, backend facts, runtime facts,
and report packages so the code migration has concrete landing zones.

Reasoning:
Removing `core` should not mean scattering types randomly. It should mean each
fact moves to the package that owns its invariants, and packages compose
through extracted outputs. The final MLIR architecture must also be visible
before implementation starts: TableGen/C++ owns real dialect plumbing, Zig owns
policy and external passes where the C API is sufficient, and reports consume
verified extracted facts.

Consequences:
The next code refactor should move target records first, then compiler facts,
backend facts, runtime facts, and report rendering. Boundary tests should fail
on new `//next/pjrtx/core` imports. The dialect implementation should bootstrap
`//next/pjrtx/dialects` and `//next/pjrtx/mlir` before replacing individual temporary
attributes with real dialect attrs or ops.

Verification:
Docs were updated and package intent files were added. This is a documentation
and boundary bootstrap; code-moving tests remain for the next implementation
slice.

Links:
- `next/docs/specs/package_boundaries_v0.md`
- `next/docs/specs/final_mlir_dialect_op_pass_architecture_v0.md`
- `next/docs/specs/implementation_workplan_v0.md`
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`
- `pjrtx/target/README.md`
- `pjrtx/mlir/README.md`
- `pjrtx/dialects/README.md`

## 2026-05-19 - Cost Ledger Enters MLIR Before Lowering

Status: accepted

Context:
Lowering records were derived in `mlir_state`, but their cost references still
pointed at a cost ledger that only lived in the trace builder until
`performance_modeled`.

Decision:
Have `commitLoweringPlan` stamp `pjrtx.performance.cost_ledger` before deriving
`pjrtx.lowering.records` and `pjrtx.lowering.region_facts`. After lowering is
planned, the compiler extracts the MLIR cost ledger and replaces its local cost
array before memory traffic is derived.

Reasoning:
Lowering regions are defined by source instructions, accepted fusion,
placement, and cost references. If cost IDs are part of the region contract,
the cost ledger must be visible in MLIR before lowering verification, not only
after performance modeling.

Consequences:
This made the cost ledger an MLIR fact before lowering and forced all later
compiler stages to read back the extracted cost view. The follow-up step moved
cost-ledger row construction itself into `mlir_state`, leaving compiler target
legality to supply only compact backend capability facts.

Verification:
`bazel test //next/pjrtx/compiler/mlir_state:unit_tests //next/pjrtx/compiler:unit_tests //next/pjrtx/vertical_slice:execution_test //next/pjrtx/vertical_slice:lowering_tests --test_output=errors` passes.

Links:
- `pjrtx/compiler/mlir_state/package.zig`
- `pjrtx/compiler/compiler.zig`
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`
- `next/docs/specs/compiler_pass_pipeline_v0.md`
- `next/docs/specs/mlir_state_machine_compiler_v0.md`

## 2026-05-19 - Lowering Records Are Derived In MLIR State

Status: accepted

Context:
`LoweringRegionFact` was already derived from MLIR fusion and placement facts,
but `LoweringRecord` itself was still assembled in the compiler trace builder
and then committed to MLIR. That left the executable-region row as a remaining
Zig-produced compiler-middle decision.

Decision:
Move V0 lowering-record formation into `mlir_state`. `commitLoweringPlan` now
receives the cost ledger, derives `pjrtx.lowering.records` from
`pjrtx.fusion.candidates`, `pjrtx.placement.records`, and cost-ledger
instruction links, derives matching `pjrtx.lowering.region_facts`, verifies
both attributes, and only then advances to `lowering_planned`.

Reasoning:
The compiler should not secretly decide executable regions in a local trace
builder after MLIR has already planned fusion and placement. The lowering
boundary is where graph instructions, accepted fusion, placement, tile/memory
intent, and cost IDs become one executable unit, so that join belongs in MLIR
state.

Consequences:
The trace builder now produces costs, then asks `mlir_state` to form and verify
lowerings. It no longer owns `addLoweringsFromCompilerMiddle`,
`appendLoweringRecord`, or local lowering decision strings. A later follow-up
also moved cost-ledger row construction into `mlir_state`, so compiler-local
trace records keep shrinking toward target legality, orchestration, and
extracted views.

Verification:
`bazel test //next/pjrtx/compiler/mlir_state:unit_tests //next/pjrtx/compiler:unit_tests //next/pjrtx/vertical_slice:execution_test //next/pjrtx/vertical_slice:lowering_tests --test_output=errors` passes.

Links:
- `pjrtx/compiler/mlir_state/package.zig`
- `pjrtx/compiler/compiler.zig`
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`
- `next/docs/specs/compiler_pass_pipeline_v0.md`
- `next/docs/specs/mlir_state_machine_compiler_v0.md`

## 2026-05-19 - Cost Ledger Rows Are Derived In MLIR State

Status: accepted

Context:
The cost ledger had become an MLIR fact before lowering, but the row
construction still lived in `TracePlanBuilder.addInstructionCosts`. That kept
FLOPs, bytes, op class, dtype, formulas, and expected execution units too close
to the old Zig trace builder.

Decision:
Add `CostCapabilityFact` as the narrow compiler-to-MLIR-state bridge. Compiler
target legality now checks backend support and supplies only instruction ID plus
expected execution unit. `mlir_state.deriveCostLedgerEntries` constructs the
typed `CostLedgerEntry` rows from graph values, graph instructions, op/type
facts, byte counts, logical ops, formulas, approximations, and capability
facts. The derived ledger is then stamped into
`pjrtx.performance.cost_ledger` before lowering-region formation.

Reasoning:
Backend capability selection is target policy and belongs near target legality.
Cost-row construction is compiler-middle explanation data and must live on the
MLIR-state path so lowering, memory traffic, schedules, kernel codegen, and
reports all consume the same extracted fact view.

Consequences:
The compiler trace builder no longer owns cost formulas or FLOP/byte row
construction. It still orchestrates capability checks and passes graph facts
into `mlir_state`, but the hidden source of truth for cost ledger content has
moved closer to the MLIR boundary. A later follow-up also moved memory-traffic
row derivation into `mlir_state`, so cost and traffic are now both derived on
the MLIR-state path before performance facts are committed. The next deeper
step is to turn this attribute bridge into real MLIR op/type/cost/traffic
passes and then feed fusion, tiling, Shardy partitioning, and kernel-region
planning from those facts.

Verification:
`bazel test //next/pjrtx/compiler/mlir_state:unit_tests //next/pjrtx/compiler:unit_tests //next/pjrtx/vertical_slice:execution_test //next/pjrtx/vertical_slice:lowering_tests --test_output=errors` passes.

Links:
- `pjrtx/compiler/mlir_state/package.zig`
- `pjrtx/compiler/compiler.zig`
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`
- `next/docs/specs/compiler_pass_pipeline_v0.md`
- `next/docs/specs/mlir_state_machine_compiler_v0.md`

## 2026-05-19 - Memory Traffic Rows Are Derived In MLIR State

Status: accepted

Context:
Cost ledger rows were derived in `mlir_state`, but `TracePlanBuilder` still
assembled `MemoryTrafficRecord` rows from local helper functions. That left HBM
boundary traffic, local tile traffic, value-flow joins, and cost-byte joins as
another hidden compiler-local source of truth before `performance_modeled`.

Decision:
Add `mlir_state.deriveMemoryTrafficRecords`. The compiler now passes target
memory spaces, graph values, graph instructions, extracted cost-ledger rows,
extracted lowering records, and extracted placement records into `mlir_state`.
`mlir_state` derives global-memory traffic for external lowering inputs and
outputs, derives local-memory traffic from cost entries whose placements use
local SRAM or scratchpad, owns the traffic reason strings, verifies the rows,
and returns typed `MemoryTrafficRecord` views for commit/extraction.

Reasoning:
Memory traffic is part of the compiler-middle performance contract, not a
backend formatting detail and not a schedule-side guess. Keeping this join near
MLIR state makes the path from target memory hierarchy to lowering to bytes
explicit before kernel codegen, scheduling, backend binding, hardware summaries,
and profile joins consume it.

Consequences:
The compiler trace builder no longer owns device-boundary or tile-memory row
construction. It still orchestrates the stage and commits the derived rows to
`pjrtx.performance.memory_traffic`, but performance reports now consume cost
and traffic rows that were both derived through `mlir_state`. A later follow-up
also moved kernel codegen row derivation into `mlir_state`. The next deeper
step is to make these bridges real MLIR passes over region/value/memory-space
facts and then add transfer DMA and collective/interconnect traffic without
pretending they are compute-lowering traffic.

Verification:
`bazel test //next/pjrtx/compiler/mlir_state:unit_tests //next/pjrtx/compiler:unit_tests //next/pjrtx/vertical_slice:execution_test //next/pjrtx/vertical_slice:lowering_tests --test_output=errors` passes.

Links:
- `pjrtx/compiler/mlir_state/package.zig`
- `pjrtx/compiler/compiler.zig`
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`
- `next/docs/specs/compiler_pass_pipeline_v0.md`
- `next/docs/specs/mlir_state_machine_compiler_v0.md`

## 2026-05-19 - Kernel Codegen Rows Are Derived In MLIR State

Status: accepted

Context:
Cost, lowering, region facts, and memory traffic had moved onto the
`mlir_state` path, but `TracePlanBuilder` still assembled
`KernelCodegenRecord` rows. That kept value-flow shape, tile shape,
memory-pressure joins, operation selection, expected unit, and reason strings
as compiler-local facts before `codegen_planned`.

Decision:
Add `KernelCodegenCapabilityFact` and
`mlir_state.deriveKernelCodegenRecords`. Compiler backend policy now supplies
backend kind, backend execute command ID, whole-graph execute operation,
fused elementwise operation, and per-instruction backend operation/expected-unit
facts. `mlir_state` derives one `KernelCodegenRecord` per lowering from those
facts plus graph instructions, extracted lowerings, placement records, and
memory traffic records, then the compiler commits and extracts the MLIR
`pjrtx.codegen.records` view as before.

Reasoning:
Kernel codegen is the bridge from executable lowering regions to generated or
selected kernels. It must join lowering, value flow, tile/memory placement,
memory pressure, cost IDs, backend operation choice, and expected unit in one
explainable record. Keeping that join in `mlir_state` prevents backend binding
or reports from depending on hidden compiler-local assembly.

Consequences:
The compiler trace builder no longer owns codegen record shape/value-flow,
memory-pressure, or operation/reason row construction. It still contributes
backend policy facts and orchestrates the MLIR commit/extract boundary. A later
follow-up also moved schedule-command and overlap derivation into `mlir_state`.
The next natural step is backend-binding derivation, followed by turning the
current attribute bridges into real MLIR codegen-region and schedule operations
and passes.

Verification:
`bazel test //next/pjrtx/compiler/mlir_state:unit_tests //next/pjrtx/compiler:unit_tests //next/pjrtx/vertical_slice:execution_test //next/pjrtx/vertical_slice:lowering_tests --test_output=errors` passes.

Links:
- `pjrtx/compiler/mlir_state/package.zig`
- `pjrtx/compiler/compiler.zig`
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`
- `next/docs/specs/compiler_pass_pipeline_v0.md`
- `next/docs/specs/mlir_state_machine_compiler_v0.md`

## 2026-05-19 - Documentation Audit Before Plan Status

Status: accepted

Context:
Before saying the V0 plan was back on track, the vision and spec documents
needed to be checked against the implementation shape. The risk was that the
code had corrected several MLIR-state drift points while the docs still carried
older wording from the Zig trace and backend/runtime summary phases.

Decision:
Audit the canonical vision/spec set before giving the next status summary. The
result is that the main architecture still points in the intended direction:
MLIR is the compiler truth, Zig records are extracted views, no fallback is
allowed, and compiler-middle depth remains the priority before deeper backend
submission. The audit found wording and ordering drift rather than a new design
reversal:

- graph-execute wording that implied an execution fallback was wrong and is now
  described as an intentional whole-graph execute operation.
- V0 pass-order docs now distinguish early schedule-row proposal from the
  verified MLIR `scheduled` state, which currently follows `codegen_planned`.
- workplan priorities now separate completed MLIR-state bridge milestones from
  the remaining next steps: backend-binding derivation, real dialect attrs/ops,
  transfer/collective traffic, bufferization, and backend kernel generation.
- harness language now names MLIR-extracted backend bindings instead of
  placeholders.

Reasoning:
The docs are the contract for this project. If the status sentence says the
plan is healthy, that claim must be traceable to the vision, V0 scope,
compiler-pass spec, MLIR-state spec, dialect spec, harness, and workplan rather
than to chat memory.

Consequences:
Future status updates should start from the docs and mention the caveat
precisely: V0 is now mostly an MLIR attribute/state bridge with Zig external
passes and extracted typed views, not yet the final custom PjRTx dialect with
native ops, canonicalization, dialect conversion, bufferization, and real
backend kernel generation.

Verification:
The old code identifier that implied fallback execution is gone after the audit
patch. Full tests should still be run after the doc/code rename.

Links:
- `next/docs/pjrtx_architecture_vision.md`
- `next/docs/specs/implementation_workplan_v0.md`
- `next/docs/specs/compiler_pass_pipeline_v0.md`
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`
- `next/docs/specs/harness_v0.md`
- `pjrtx/compiler/mlir_state/package.zig`

## 2026-05-19 - Schedule Rows Are Derived In MLIR State

Status: accepted

Context:
Kernel codegen rows had moved onto the `mlir_state` path, but
`TracePlanBuilder` still assembled the V0 H2D/backend/D2H schedule and
serialized overlap rows. That left command inputs/outputs, dependency edges,
lowering/cost links, and overlap reason strings as compiler-local facts before
`scheduled`.

Decision:
Add `SchedulePlanFact` and `mlir_state.deriveSchedulePlan`. The compiler still
extracts parameter IDs and return value IDs from the graph, then passes those
IDs plus extracted lowering records and cost ledger entries into `mlir_state`.
`mlir_state` derives the H2D, backend execute, and D2H commands, derives
serialized transfer/compute and compute/transfer overlap rows, owns overlap
reason strings, verifies both record sets, and returns typed schedule views for
commit/extraction.

Reasoning:
Scheduling is where lowering regions and cost records become executable command
order. Even in the conservative V0 sequence, command provenance and overlap
decisions must be explainable before backend binding, allocation, streams,
events, profiling, and hardware summaries consume them.

Consequences:
The compiler trace builder no longer owns schedule command or overlap-row
construction. It still identifies graph boundary values and orchestrates the
MLIR commit/extract boundary. The next natural step is backend-binding
derivation, then moving the current attribute bridge toward real MLIR schedule,
event, stream, allocator, and overlap facts.

Verification:
`bazel test //next/pjrtx/compiler/mlir_state:unit_tests //next/pjrtx/compiler:unit_tests //next/pjrtx/vertical_slice:execution_test //next/pjrtx/vertical_slice:lowering_tests --test_output=errors` passes.

Links:
- `pjrtx/compiler/mlir_state/package.zig`
- `pjrtx/compiler/compiler.zig`
- `next/docs/specs/pjrtx_mlir_dialect_v0.md`
- `next/docs/specs/compiler_pass_pipeline_v0.md`
- `next/docs/specs/mlir_state_machine_compiler_v0.md`
