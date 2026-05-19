# PjRTx MLIR Dialect V0

This spec defines the first concrete MLIR surface for the new PjRTx compiler
middle. It refines `mlir_state_machine_compiler_v0.md` into something that can
be implemented in small steps under `//pjrtx/compiler/...`.

The goal is not to create a beautiful final dialect immediately. The goal is
to stop growing a parallel Zig compiler representation. The next slice should
make MLIR carry one real compiler decision, then extract the existing Zig report
view from that MLIR state.

## Direction

Use MLIR as the compiler state machine:

```text
StableHLO/Shardy MLIR
  -> PjRTx module state attributes
  -> PjRTx decision attributes/ops
  -> verified state transitions
  -> extracted Zig TraceReport views
```

Zig APIs should make this pleasant to use, but Zig APIs must not become the
source of truth for lowering decisions.

## V0 Slice

The first implementation slice should cover:

- module state marker
- target attachment marker
- fusion candidate marker
- fusion decision marker
- pressure delta marker
- verifier for required state transitions
- extractor from MLIR fusion facts to the current `FusionGroup` report view
- MLIR snapshot tests for each boundary

This slice is intentionally narrow. If this works, tiling, memory pressure,
collectives, codegen regions, and schedules can follow the same pattern.

## Dialect Strategy

V0 may start as attributes on existing MLIR ops before a full TableGen dialect
is wired. That keeps the first slice close to the current Bazel/MLIR C API
setup. The design should still reserve the names as dialect-owned:

```mlir
module attributes {
  pjrtx.state = "fusion_planned",
  pjrtx.target.name = "npu_v0",
  pjrtx.target.fingerprint = 1234 : i64,
  pjrtx.fusion.candidates = [
    #pjrtx.fusion_candidate<
      id = 0,
      root = "stablehlo.tanh",
      operation_count = 4,
      reason = "dot+broadcast/add/tanh epilogue candidate",
      plan_index = 0,
      decision = "rejected",
      kind = "matmul_epilogue",
      instructions = [0, 1, 2, 3],
      split_kernels = 2,
      fused_kernels = 1,
      split_peak_live_bytes = 108,
      fused_live_bytes = 188,
      additional_live_bytes = 80,
      global_bytes_saved = 72,
      decision_reason = "V0 keeps matmul boundary until epilogue codegen is explicit"
    >
  ]
} {
  func.func @main(...)
}
```

The exact textual syntax can change when a real dialect is added. The stable
requirements are:

- facts are attached to the MLIR module or relevant operations
- facts survive canonicalization boundaries that claim to preserve them
- facts are visible in MLIR snapshots
- facts can be verified before extraction
- extraction into Zig records is deterministic

`pjrtx.fusion.plan` is not part of the live V0 MLIR state machine. Fusion
planning must enrich `pjrtx.fusion.candidates` directly.

### Zig-First, MLIR-Native Dialects

This V0 spec owns the current attribute bridge. The final dialect/op/pass
architecture, including the Zig/TableGen/C++ split and package layout, is
canonical in `final_mlir_dialect_op_pass_architecture_v0.md` and
`package_boundaries_v0.md`.

For this file, the rule is narrower:

- V0 may use module attributes while bootstrapping real dialect plumbing.
- Zig external passes are allowed when the public MLIR C API is sufficient.
- Once a fact needs native dialect syntax, parser/printer integration, verifier
  hooks, or dialect-conversion support, migrate that fact to the final dialect
  path instead of growing the attribute shim.
- The C-compatible shim must not expose a second compiler model or duplicate
  PjRTx policy.

## State Attributes

Use a module-level state attribute first:

```text
pjrtx.state = imported
pjrtx.state = stablehlo_verified
pjrtx.state = canonicalized
pjrtx.state = target_attached
pjrtx.state = target_legal
pjrtx.state = fusion_planned
pjrtx.state = tiled
pjrtx.state = memory_planned
pjrtx.state = collectives_planned
pjrtx.state = collectives_lowered
pjrtx.state = lowering_planned
pjrtx.state = performance_modeled
pjrtx.state = bufferized
pjrtx.state = codegen_planned
pjrtx.state = tile_legal
pjrtx.state = scheduled
pjrtx.state = backend_bound
pjrtx.state = executable_ready
pjrtx.state = backend_executable_planned
pjrtx.state = backend_kernel_graph_planned
pjrtx.state = runtime_allocation_planned
pjrtx.state = runtime_stream_planned
pjrtx.state = runtime_profiled
pjrtx.state = runtime_profile_joined
pjrtx.state = backend_profile_joined
```

Rules:

- a pass declares required input state and produced output state
- a pass may not skip states unless the skip is represented by an explicit
  verifier-approved transition
- a failed verifier prevents executable creation
- extracted reports include the state transition records, but the MLIR state is
  authoritative

## Target Attributes

Target attachment should be module-level:

```text
pjrtx.target.name
pjrtx.target.kind
pjrtx.target.fingerprint
pjrtx.target.replicas
pjrtx.target.partitions
pjrtx.target.spec
```

`pjrtx.target.spec` is the full V0 hardware contract: devices, memory spaces,
transfer edges, execution units, dtype/op-class rates, and notes. The compact
identity attributes remain because they are useful for fast state checks and
human summaries, but performance-facing summaries must read hardware
capabilities back from `extractTargetDescription` instead of the original Zig
target object.

Verifier rules:

- `target_attached` requires target name, kind, fingerprint, replicas,
  partitions, and a dictionary `pjrtx.target.spec`
- later target-specific decisions must reference the same fingerprint
- extracting a report with mismatched target fingerprint is a compiler error

## Fusion Facts

Fusion facts should be represented before they are extracted to `FusionGroup`.

V0 fields:

```text
id
decision
kind
source_instruction_ids
root_instruction_id
bytes_saved
launches_before
launches_after
pressure_delta
reason
```

Pressure delta fields:

```text
split_kernel_count
fused_kernel_count
split_peak_live_bytes
fused_live_bytes
additional_live_bytes
global_bytes_saved
```

Verifier rules:

- fusion planning requires state `target_legal`
- fusion planning produces state `fusion_planned`
- every fusion fact references existing StableHLO/PjRTx graph source IDs or
  carries a diagnostic explaining why it is internal
- accepted fusion must not cross token, collective, side-effect, or strict math
  boundaries
- rejected fusion must carry a reason
- pressure byte fields use byte units and must not be negative
- `additional_live_bytes` must equal `fused_live_bytes - split_peak_live_bytes`
  when both sides are known

## Extraction Mapping

The current Zig report remains the public evidence surface. The first mapping
should be one-way:

```text
MLIR pjrtx fusion fact -> core.FusionGroup
```

Do not reconstruct MLIR from `FusionGroup`.

Extraction requirements:

- deterministic order by MLIR fusion ID
- copied strings and slices owned by the extracted report
- no MLIR pointers stored in `TraceReport`
- extraction validates state before producing records
- extraction failure writes a focused diagnostic through `std.Io.Writer`

## Zig API Shape

PjRTx should borrow the useful parts of ZML's MLIR handling:

- one registry initialization path
- a context/module/pass-manager owner
- explicit dialect loading
- small helpers for attributes and operation construction
- `std.Io` threaded through compile setup and diagnostics
- `std.Io.Writer.Allocating` for serialization buffers and debug dumps

ZML reference patterns:

- `/Users/hugo/Developer/zml/zml/module.zig` initializes a dialect registry,
  MLIR context, module, and pass manager.
- It registers `func`, `stablehlo`, and `sdy`, then loads available dialects.
- It attaches module attributes with `operation().setAttributeByName(...)`.
- It constructs Shardy mesh ops with `mlir.Operation.make(...)`.
- It runs canonicalization/CSE through a pass manager.
- `/Users/hugo/Developer/zml/zml/mlirx.zig` keeps type helpers thin and local.

PjRTx should adapt that shape, not copy it blindly.

### `MlirSession`

The first PjRTx wrapper should be small:

```zig
pub const MlirSession = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    registry: mlir.MlirDialectRegistry,
    context: mlir.MlirContext,
    module: mlir.MlirModule,
    pass_manager: mlir.MlirPassManager,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        options: MlirSessionOptions,
    ) !MlirSession;

    pub fn deinit(self: *MlirSession) void;
};
```

V0 can keep using raw MLIR C handles internally, but they should be owned by a
single session object. This avoids scattering context lifetime across compiler
passes.

Rules:

- registry setup may be process-global if MLIR requires it, but compiler state
  must not be global
- global registry initialization must be guarded by `std.Io.Mutex`
- session owns context, module, and pass manager lifetime
- pass helpers receive `*MlirSession` and diagnostics writer
- session methods return errors; they do not panic on invalid user input
- tests may panic on impossible fixture construction, not on compile failures

### Typed Helpers

Add helpers only where they encode PjRTx intent:

```zig
pub fn setModuleState(session: *MlirSession, state: ModuleState) !void;
pub fn requireModuleState(
    session: *const MlirSession,
    expected: ModuleState,
    diagnostics: *std.Io.Writer,
) !void;
pub fn attachTarget(
    session: *MlirSession,
    target: target_pkg.TargetDescription,
    diagnostics: *std.Io.Writer,
) !void;
pub fn planFusionFromCandidates(
    session: *MlirSession,
    diagnostics: *std.Io.Writer,
) !void;
pub fn extractFusionGroups(
    allocator: std.mem.Allocator,
    session: *const MlirSession,
    diagnostics: *std.Io.Writer,
) ![]core.FusionGroup;
pub fn extractTargetDescription(
    allocator: std.mem.Allocator,
    session: *const MlirSession,
    diagnostics: *std.Io.Writer,
) !target_pkg.TargetDescription;
```

Avoid wrapping every MLIR C API call. Use the Zig standard library and the MLIR
C API directly when no PjRTx invariant is being added.

## Pass Shape

The first MLIR-state passes can be simple functions:

```zig
pub fn attachTargetPass(
    session: *MlirSession,
    target: target_pkg.TargetDescription,
    diagnostics: *std.Io.Writer,
) !void;

pub fn fusionPlanPass(
    session: *MlirSession,
    graph: GraphModuleView,
    diagnostics: *std.Io.Writer,
) !void;
```

The contract is more important than the framework:

- check input state
- inspect MLIR and existing extracted graph view as needed
- attach/update MLIR facts
- run verifier
- set output state
- optionally extract records for V0 reports

The next step is to prove Zig-owned MLIR passes through the MLIR C API external
pass interface. A PjRTx pass may be implemented as Zig `callconv(.c)` callbacks
and inserted into an MLIR pass manager when the operation access needed by that
pass is available through the C API. The pass still follows the same contract:
stable name, declared state transition, verifier boundary, diagnostics, and
deterministic extraction.

Use this staged pass shape:

1. Straight-line Zig functions for the current V0 state shim.
2. Zig external MLIR passes through the MLIR C API for state, verification, and
   transformations that do not require advanced C++ rewrite infrastructure.
3. Minimal MLIR-native C++/TableGen pass or rewrite glue only when the pass
   needs dialect conversion, pattern rewriting, generated op adaptors, or
   parser/printer/verifier integration unavailable through the C API.

Avoid building a separate Zig pass framework that competes with MLIR's pass
manager. Zig should configure and run MLIR pass managers, implement PjRTx
policy where practical, and keep extraction/reporting deterministic.

## Boundary Tests

Add tests before expanding the dialect:

```text
//pjrtx/tests/fixtures/tanh_dot_bias.mlir
  -> after-imported.mlir
  -> after-target-attached.mlir
  -> after-fusion-planned.mlir
  -> extracted-fusion-report.txt
```

Failure tests:

- attach fusion before target legality
- missing target fingerprint
- rejected fusion without reason
- pressure delta arithmetic mismatch
- extraction from wrong state

Use runfiles for fixtures and goldens. Do not use `@embedFile`.

## Bazel Shape

The first code layout should be:

```text
//pjrtx/mlir:session
  owns MlirSession, registry setup, pass manager, and dumps

//pjrtx/compiler/mlir_state
  owns the temporary attribute bridge and extraction helpers

//pjrtx/compiler:compiler
  consumes //pjrtx/compiler/mlir_state

//pjrtx/tests/vertical_slice:mlir_state_tests
  fixture snapshots and extraction tests
```

The underlying C++/C API target should continue to depend on MLIR, StableHLO,
and Shardy:

```text
@llvm-project//mlir:CAPIIR
@llvm-project//mlir:CAPITransforms
@llvm-project//mlir:Pass
@llvm-project//mlir:FuncDialect
@stablehlo//:stablehlo_capi
@stablehlo//:stablehlo_dialect_capi
@shardy//shardy/integrations/c:sdy_capi
```

When a real PjRTx dialect is introduced, add a dedicated Bazel target for the
dialect and keep the Zig session API stable.

Expected dialect targets:

```text
//pjrtx/dialects:pjrtx_dialect_td
  TableGen declarations for PjRTx ops/types/attrs

//pjrtx/dialects:pjrtx_dialect
  generated MLIR dialect implementation and minimal hand-written verifiers

//pjrtx/dialects:pjrtx_c_api
  narrow C-compatible registration/pass construction functions consumed by Zig
```

The Zig targets should depend on these only through stable C headers and Bazel
deps. Do not include C++ dialect internals directly in Zig-facing APIs.

## What To Avoid

- Do not add another large family of Zig records before deciding where the MLIR
  fact lives.
- Do not store MLIR pointers in runtime or report objects.
- Do not let backend code infer fusion, tiling, memory, or collective choices
  from StableHLO strings.
- Do not make extraction stronger than the MLIR verifier.
- Do not use global mutable state for module, target, pass, or report facts.
- Do not hide failed verification behind fallback execution.

## Acceptance Criteria

The first dialect slice is accepted when:

- an MLIR module carries a visible PjRTx state attribute
- target attachment updates and verifies that state
- fusion planning attaches visible MLIR fusion/pressure facts
- extraction produces the existing `FusionGroup` report data from MLIR
- invalid state transitions fail with diagnostics
- golden MLIR snapshots prove the boundary
- `bazel test //pjrtx/...` passes

## Initial Implementation Note

The first historical code slice now lives in `//pjrtx/compiler/mlir_state`. The
next refactor should move session ownership to `//pjrtx/mlir:session` and keep
`//pjrtx/compiler/mlir_state` focused on the temporary attribute bridge until
real dialect attrs/ops replace it.

It intentionally uses attributes on the module until a real TableGen dialect is
introduced. This is a temporary dialect shim, not the final IR design. The
important property is already true: the fusion/pressure fact is attached to the
MLIR module as a structured array/dictionary attribute and the Zig
`FusionGroup` is extracted from that MLIR fact.

`compileV0FromReader` uses this shim on the end-to-end compile path. The old
graph-only full-Zig fusion planner compatibility path is removed: full trace
planning now requires an `MlirSession`, and tests or helpers that start after
graph import must call `planV0TraceReportFromMlirSession` with the session that
owns the MLIR compiler-state facts. The orchestrated V0 compile path discovers
candidates, makes V0 fusion decisions, and extracts final fusion groups through
MLIR state before the final trace report receives them.

`//pjrtx/tests/vertical_slice:mlir_state_test` keeps the first runfiles-backed
golden for this boundary. It summarizes the MLIR-owned state, target identity,
and structured fusion/pressure attributes. Future tests can add raw MLIR
snapshots once the dialect spelling is less temporary.

`runExternalStateProbePass` is the first Zig-owned MLIR external pass proof. It
uses MLIR's pass manager and C external-pass callback API from Zig, verifies
that PjRTx module state exists, and stamps `pjrtx.external_pass.proof = "ran"`
on the module. This keeps the next implementation step aligned with the
Zig-first policy while avoiding a separate project-local pass framework.

`markTargetLegal` now uses the same mechanism for the first meaningful state
transition pass. `runTargetLegalExternalPass` verifies `target_attached` state
and required target attributes inside an MLIR external pass before setting
`pjrtx.state = "target_legal"`. The graph-level legality check still runs in
the orchestrator before this state transition, so V0 does not claim target
legality until both the graph policy and MLIR state checks have passed.

`runFusionCandidateDiscoveryExternalPass` is the first MLIR-side discovery
slice. It runs after target legality, walks StableHLO operations through the
MLIR C API, recognizes the V0 `tanh(add(dot_general, broadcast_in_dim))`
pattern, and stamps a structured `pjrtx.fusion.candidates` array on the module.
Each candidate entry carries an index, kind, root operation, operation count,
reason, graph instruction ids, tensor-byte-derived savings, launch reduction,
and pressure delta. The compatibility count attributes
`pjrtx.fusion.candidates.matmul_epilogue` and
`pjrtx.fusion.candidates.elementwise_chain` are derived from that structured
array.

`runFusionPlanExternalPass` enriches each candidate entry with the final V0
decision: `plan_index`, `decision`, and `decision_reason`, while preserving the
candidate-side instruction, byte, launch, and pressure facts. The pass applies
the V0 decision policy directly from `pjrtx.fusion.candidates`; this is the
only live MLIR fusion-planning path. Fusion extraction reads the enriched
candidate entries, and `pjrtx.fusion.plan` should not appear in
post-`fusion_planned` MLIR snapshots.

`commitPlacementPlan` is the first placement-state bridge. The current V0 Zig
planner still computes the initial layout, logical tile shape, result memory
space, optional tile memory space, and reason string, but the trace report no
longer consumes that local object directly. The compiler commits those records
to `pjrtx.placement.records`, runs `runPlacementPlanExternalPass` to verify the
MLIR attribute shape and advance state to `placement_planned`, and then extracts
the final `PlacementRecord` report view from MLIR. This keeps placement on the
same one-way path as fusion until real dialect attrs/ops replace the temporary
attribute shim.

`commitCollectivePlan` extends the same bridge to collectives. The current V0
collective planner still computes the initial decision, selected or rejected
algorithm, checked/lowered/unsupported counts, estimated bytes, optional latency,
and reason string. The compiler commits that record to
`pjrtx.collective.records`, runs `runCollectivePlanExternalPass` to verify the
MLIR attribute shape and advance state to `collectives_planned`, and then
extracts the final `CollectivePlanRecord` report view from MLIR. Unsupported
collectives still fail before schedule build; the difference is that the
compiler state now carries the explicit rejection first.

`commitLoweringPlan` extends the bridge to lowering-region formation. The
compiler supplies compact backend capability facts for each executable graph
instruction, then `mlir_state` derives the V0 cost ledger from graph values,
graph instructions, op/type facts, formulas, byte counts, logical ops, and
expected execution-unit facts. `commitLoweringPlan` stamps that derived ledger
as `pjrtx.performance.cost_ledger` before lowering-region formation.
`mlir_state` then derives `pjrtx.lowering.records` and
`pjrtx.lowering.region_facts` from MLIR-visible fusion candidates, placement
records, and cost-ledger instruction links. The derived facts join each
lowering to graph instruction IDs, cost IDs, fusion group when known, placement
records, logical tile shape, result/tile memory spaces, rejected alternatives,
and codegen-region intent.
`runLoweringPlanExternalPass` verifies the derived lowering records and region
facts together, advances state from `collectives_planned` to
`lowering_planned`, and stamps
`pjrtx.lowering_plan.pass`, then extraction produces the final
`LoweringRecord` and `LoweringRegionFact` views from MLIR.
This makes lowering regions the verified unit that memory traffic,
performance, schedule, backend binding, kernel graph planning, and profile
joins must reference.

After lowering is planned, the compiler extracts that MLIR cost view and passes
the extracted graph, placement, lowering, and cost facts back into `mlir_state`
to derive memory traffic. Predicted ops, bytes, dtype/op-class, expected
execution unit, formulas, approximations, and memory hierarchy traffic no
longer live only in the trace builder before `performance_modeled`.

`commitPerformanceFacts` extends the bridge from cost ledger to full
performance state by adding memory traffic. `mlir_state` derives the V0
lowering-linked `MemoryTrafficRecord` rows from target memory spaces, graph
value flow, placement records, lowering records, and cost-ledger bytes. The
compiler commits those facts to `pjrtx.performance.memory_traffic`, runs
`runPerformanceFactsExternalPass` to verify IDs, non-empty metadata, op classes,
dtypes, lowering IDs, and memory-traffic references, advances state from
`lowering_planned` to `performance_modeled`, and extracts the final
`CostLedgerEntry` and
`MemoryTrafficRecord` report views from MLIR. Runtime, lowering, hardware, and
backend-call summaries must consume those extracted performance views; the old
Zig trace is no longer the hidden source of predicted bytes, ops, or traffic.

`commitKernelCodegenPlan` extends the bridge to generated-kernel and
kernel-graph planning. Compiler backend policy supplies compact codegen
capability facts: backend kind, backend operation names, fused elementwise
operation name, whole-graph execute operation name, backend execute command
ID, and expected unit IDs. `mlir_state` derives the initial lowering-to-kernel
records from those facts plus graph instructions, lowering records, placement
records, and memory-traffic records: kernel kind, operation name, value-flow
shape, logical tile, result and tile memory spaces, memory pressure, source
instruction IDs, cost IDs, memory-traffic IDs, expected unit, and reason. The
compiler commits those records to `pjrtx.codegen.records`, runs
`runKernelCodegenPlanExternalPass` to verify the MLIR attribute shape and
advance state from `performance_modeled` to `codegen_planned`, and then
extracts the final `KernelCodegenRecord` report view from MLIR before tile
legality and backend binding consume it.

`commitSchedulePlan` extends the bridge to executable command planning. The V0
schedule builder now asks `mlir_state` to derive the initial H2D/backend/D2H
command sequence and serialized overlap records from parameter IDs, return
value IDs, extracted lowering records, and extracted cost ledger entries. The
derived rows include streams, inputs, outputs, dependencies, lowering IDs, cost
IDs, and reasons. The compiler commits those records to
`pjrtx.schedule.commands` and `pjrtx.schedule.overlaps`, runs
`runSchedulePlanExternalPass` to verify the MLIR attribute shape and advance
state to `scheduled`, and then extracts the final `ScheduleCommand` and
`ScheduleOverlapRecord` report views from MLIR before backend binding and
executable view creation consume them.

Schedule facts remain valid after later backend, executable, runtime, and
profile facts are added. `extractScheduleCommands` and
`extractScheduleOverlapRecords` may therefore read schedule facts from
`scheduled` or any later V0 state. Runtime execution and hardware-utilization
summaries should receive extracted schedule commands when they join committed
stream, allocation, and profile facts back to command order.

`commitBackendBindings` extends the bridge to backend selection and executable
readiness. The current V0 Zig binder still computes the initial
command-to-backend records: command ID, backend kind, backend operation,
source instruction IDs, expected execution unit, and cost IDs. The compiler
commits those records to `pjrtx.backend.bindings`, runs
`runBackendBindingExternalPass` to verify the MLIR attribute shape and advance
state to `backend_bound`, and then extracts the final `BackendBinding` report
view from MLIR before executable view creation, Metal/MLS graph expansion, and
runtime verification consume it.

`commitExecutableReadiness` extends the bridge to the first executable
contract. The current V0 contract is intentionally small: selected target kind,
schedule command count, backend binding count, and kernel codegen record count.
The compiler commits this to `pjrtx.executable.contract`, runs
`runExecutableReadyExternalPass` to verify that the counts match the already
verified MLIR schedule, backend binding, and codegen facts, and advances state
to `executable_ready`. This is not a backend executable plan yet; it is the
compiler-owned gate that says the MLIR module has enough verified facts for
Zig executable view creation to proceed without fallback.

`//pjrtx/backend:mlir_bridge` extends the state machine to backend executable
calls without making `//pjrtx/compiler` depend on `//pjrtx/backend`. Backend
planning still expands the verified binding into concrete backend calls, then
the bridge commits the plan to `pjrtx.backend.executable`, runs
`runBackendExecutablePlanExternalPass`, verifies that the plan matches the
selected target and executable contract, and advances state to
`backend_executable_planned`. V0 records call index, command, backend kind,
primary instruction, full instruction region, feature, backend operation,
inputs, outputs, and expected execution unit.

Backend executable facts must be extractable from MLIR after verification.
`extractBackendExecutablePlan` may read `pjrtx.backend.executable` from
`backend_executable_planned` or later backend/runtime states, because later
kernel graph, allocation, stream, and profile facts refine execution without
invalidating the concrete call sequence.

The same bridge commits Metal/MLS kernel graph facts after backend executable
planning. `pjrtx.backend.kernel_graph` records backend kind, command, one node
per backend executable call, and value-flow edges between nodes. Each node
records call index, primary instruction, full instruction region, feature,
backend operation, input/output value IDs, output tensor descriptor, and a
compact attribute tag. `runBackendKernelGraphExternalPass` verifies the graph
against the committed backend executable plan and advances state to
`backend_kernel_graph_planned`.

Metal/MLS kernel graph facts must also be extractable from MLIR after
verification. `extractBackendKernelGraph` may read `pjrtx.backend.kernel_graph`
from `backend_kernel_graph_planned` or later runtime/profile states.

`//pjrtx/runtime:mlir_bridge` commits runtime allocator reservations after the
backend executable or Metal/MLS kernel graph facts exist. `pjrtx.runtime.allocation`
records buffer allocations, placement, memory-space IDs, byte sizes, command
lifetimes, command-buffer uses, and peak live device bytes. The runtime planner
still owns capacity and transfer-edge checks; the MLIR bridge records the
verified result and `runRuntimeAllocationExternalPass` advances state to
`runtime_allocation_planned`.

Runtime allocation facts must be extractable from MLIR after verification.
`extractRuntimeAllocationPlan` may read `pjrtx.runtime.allocation` from
`runtime_allocation_planned` or later runtime/profile states, because later
stream and profile facts refine execution without invalidating allocator
reservations.

The runtime bridge also commits stream/event planning after allocation.
`pjrtx.runtime.streams` records one step per schedule command with command ID,
stream ID, wait event IDs, start event, and done event.
`runRuntimeStreamExternalPass` verifies the stream records and advances state to
`runtime_stream_planned`.

Runtime stream facts must also be extractable from MLIR after verification.
`extractRuntimeStreamPlan` may read `pjrtx.runtime.streams` from
`runtime_stream_planned` or later profile states.

After streams exist, the runtime bridge commits profile observations to
`pjrtx.runtime.profile_events`. V0 synthetic events record event ID, optional
command ID, graph instruction IDs, kind, start/duration timestamps, bytes,
logical ops, status, and whether a forced synchronization occurred.
`runRuntimeProfileExternalPass` verifies the event rows and advances state to
`runtime_profiled`.

Profile events must be extractable from MLIR after verification.
`extractRuntimeProfileEvents` may read `pjrtx.runtime.profile_events` from
`runtime_profiled` or later profile states, because later profile joins refine
the explanation graph without invalidating the original observations. Later
reports should prefer these extracted events over the original runtime-owned
trace rows.

After profile events exist, the runtime bridge commits explicit join rows to
`pjrtx.runtime.profile_joins`. Each row names the explained subject kind and
ID, optional command ID, graph instruction IDs, and the profile event IDs that
measure that subject. V0 materializes schedule-command, lowering-record, and
explain-record joins so observed metrics can be followed from MLIR state back
to compiler decisions without reconstructing the relation from report text.
`runRuntimeProfileJoinExternalPass` verifies the join rows and advances state
to `runtime_profile_joined`.

After runtime profile joins exist, the backend bridge commits backend-call
profile joins to `pjrtx.backend.profile_joins`. Each row names the backend
executable call index, command ID, instruction IDs, and profile event ID that
measured the call. This is backend-owned because only the backend bridge owns
the concrete call sequence; the compiler does not import backend internals.
`runBackendProfileJoinExternalPass` verifies those rows and advances state to
`backend_profile_joined`.

Both profile-join families must be extractable from MLIR after verification.
`extractRuntimeProfileJoins` may read `pjrtx.runtime.profile_joins` from either
`runtime_profile_joined` or a later `backend_profile_joined` state, because the
runtime attribute remains valid after backend joins are added. Backend joins
are extracted from `backend_profile_joined`. Summary writers and later reports
should prefer these extracted MLIR facts over reconstructing joins from the
original Zig-side trace objects.

The report-facing bridge now includes typed summary writers for the extracted
MLIR facts. Runtime allocation, stream, runtime execution, lowering profile,
hardware utilization, backend executable, backend kernel graph, and backend
call-profile summaries are rendered from extracted `mlir_state` views in the
vertical slice. A writer may still take the compiler `TraceReport` for stable
graph topology and source context, but target hardware, lowering records,
lowering region facts, cost ledger entries, memory traffic, schedule commands,
allocation reservations, stream events, profile observations, backend calls,
kernel graph nodes, and backend-call/profile joins must come from MLIR
extraction. This keeps the report layer under pressure: a fact that matters to
explainability should be committed, verified, extracted, and then consumed from
the extracted view instead of being recovered from an old Zig planner object.

Backend call-profile reporting joins `extractTargetDescription`,
`extractCostLedgerEntries`, `extractMemoryTrafficRecords`,
`extractBackendExecutablePlan`, `extractRuntimeProfileEvents`, and
`extractBackendProfileJoins`: concrete calls come from the backend executable
fact, predicted bytes/ops and ideal roofline terms come from extracted target
and performance facts, observed bytes/ops come from extracted runtime profile
events, and the exact call-to-event relation comes from backend profile joins.
