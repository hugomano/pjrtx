# Implementation Workplan V0

This spec turns the V0 architecture docs into a concrete implementation queue
for code under `//next/pjrtx/...`.

The workplan favors small vertical proof over broad framework. Each phase must
leave tests, reports, and diagnostics behind.

## Current Implementation Status

As of 2026-05-19, code under `//next/pjrtx/...` has initial coverage for:

- Phase 0 package skeleton and coding policy
- Phase 1 core trace/target records and validators
- Phase 2 stable summary report and runfiles-backed golden comparison
- Phase 3 MLIR/StableHLO/Shardy C API text ingest
- Phase 4 typed V0 graph import and graph verification
- Phase 5 target selection for `metal_v0` and `npu_v0`
- Phase 5 first backend capability legality gate by feature and dtype
- Phase 6 first conservative V0 cost/lowering/schedule planner
- Phase 6 cost and backend binding metadata derived from backend capabilities
- Phase 6 compiler schedule verification for command order and backend
  dependencies
- Phase 6A first report-only MLIR pass pipeline artifact with Shardy metadata
  preservation status
- Phase 6A compiler pass pipeline spec added to define XLA-like pass families,
  pass contracts, invariants, and future pass ordering
- Phase 6A XLA coverage map added to document what the local XLA tree already
  does and how PjRTx should match that breadth with first-class explainability
- Phase 6B first MLIR-owned fusion candidate/decision facts with accepted
  elementwise chains and rejected matmul-boundary alternatives
- Phase 6C first placement plan artifact with layout, target-aware tile shape,
  HBM/unified result memory, and NPU local-SRAM tile staging
- Phase 6D first collective graph payload import for StableHLO all-reduce;
  unsupported algorithms now fail at compiler-middle algorithm selection before
  schedule build or runtime
- Phase 6E first trace-report join for fusion, placement, and collective
  records before cost, lowering, schedule, and backend binding
- Phase 6E first compile orchestrator that carries MLIR/Shardy pass records
  into the final V0 trace report
- Phase 6E compiler package boundary hardened: target legality uses a
  compiler-owned V0 capability view and no longer imports `//next/pjrtx/backend`
- First `//next/pjrtx/architecture:boundary_test` package guardrail exists:
  compiler/backend/runtime/plugin edges are checked, `//next/pjrtx/core:sources` is
  forbidden, and the current bootstrap `//next/pjrtx/core` import baseline is pinned
  so migration can only shrink it.
- First `//next/pjrtx/target` ownership slice is live: target hardware records,
  dtype rates, structural target validation, and target summary rendering live
  in `//next/pjrtx/target`; compiler, MLIR state, backend, runtime, and vertical
  execution tests now import target facts directly. `//next/pjrtx/core` keeps only a
  temporary compatibility bridge for report-shaped legacy callers.
- First `//next/pjrtx/compiler/facts` ownership slice is live: source IDs, graph
  value/instruction IDs, tensor types, source refs, graph instruction payloads,
  graph value/instruction records, tensor validation, and graph payload-kind
  checks live under `//next/pjrtx/compiler/facts`. `//next/pjrtx/core` temporarily
  re-exports these compiler-owned facts for legacy trace/report records.
- Compiler and backend implementation code now import those source/graph/tensor
  facts directly from `//next/pjrtx/compiler/facts`; runtime allocation/profile
  code and vertical tests also use the same compiler-facts names for immutable
  graph inputs. The old `core.Graph*`, `core.Source*`, and `core.TensorType`
  spellings are gone from live `//next/pjrtx/...` code outside the temporary
  `core` bridge.
- The next `//next/pjrtx/compiler/facts` slice is live: MLIR pass records, graph
  rewrite records, fusion groups, placement records, collective plan records,
  lowering records, cost ledger records, and memory-traffic records are defined
  in compiler facts. Live code uses `compiler_facts.*` names for these records;
  `//next/pjrtx/core` keeps only compatibility aliases for the legacy `TraceReport`
  container and report writer.
- `//next/pjrtx/compiler/facts` is now split by ownership into ID, graph, and
  compiler-middle fact families. The package root is a facade, while graph
  validation logic lives in small boundary structs such as `TensorFacts` and
  `GraphPayloadFacts`.
- Phase 6E planner now consumes fusion and placement records when emitting
  lowering records: V0 lowers the elementwise chain as one fused region instead
  of one lowering per StableHLO instruction
- Phase 6E fusion planning now records `dot+broadcast/add/tanh` as an explicit
  rejected matmul epilogue candidate before keeping the matmul boundary and
  lowering the elementwise chain separately
- Phase 6C `tile_shape_select` now keeps Metal V0 whole-tensor tiles while
  bounding NPU V0 local-SRAM tiles for large matrix and fusible elementwise work
- Phase 6E `memory_traffic_refine` now names global-memory and local-memory
  traffic explicitly, with schema slots reserved for host/device DMA and
  interconnect traffic once transfer and collective lowering records exist
- Phase 6D `collective_algorithm_select` now records `algorithm=none` for
  graphs with no collective work and rejects imported unsupported collective
  payloads with explicit direct/ring/tree/split algorithm diagnostics
- Phase 6D `collective_group_channel_verify` now stores replica-group
  participants and channel handle ID/type in the imported collective payload,
  then verifies participants against `replicas * partitions` before algorithm
  selection
- Phase 7 first backend binding verification and report edge
- Phase 8 first synthetic runtime profile path for H2D/backend/D2H commands
- Phase 8 first runtime allocation plan for host/device buffers, command
  buffer uses, lifetimes, peak live device bytes, and an MLIR-extracted summary
  path
- Phase 8 first runtime stream plan for command events and dependency waits
- Phase 8 first runtime execution summary comparing predicted and synthetic
  observed bytes/ops per command
- Phase 8 synthetic profile events now include backend lowering-region events
  alongside coarse H2D/backend/D2H command events
- Phase 8 stable trace summaries preserve profile join keys and counters while
  redacting raw timing
- Phase 8 profiled reports attach profile event IDs back to lowering and
  backend-binding explain records
- Phase 8 backend call profile summaries now include expected execution unit,
  ideal compute time, per-memory-space ideal memory time, and limiting resource
  from target dtype/op-class rates, first-class memory traffic records, and
  memory bandwidth
- Phase 8 compiler emits first-class memory traffic records for global-memory
  and local-memory bytes per lowering
- Phase 8 hardware utilization summaries print memory traffic records with
  per-memory-space bandwidth and ideal memory time
- Phase 8 hardware utilization summaries print backend-independent lowering
  roofline rows that join predicted ops/bytes, ideal compute time, ideal memory
  time, and limiting resource
- Phase 8 lowering roofline rows join back to lowering-region profile events
  with observed bytes/ops and stable profile IDs
- Phase 8 profiled reports preserve compiler-middle records so profiling views
  can still explain fusion, placement, Shardy pass, and collective context
- Phase 8 runfiles-backed golden execution report for allocation, stream, and
  runtime summaries
- Phase 8 first hardware utilization summary for memory spaces and execution
  units
- Phase 8 target dtype/op-class rate rows joined with predicted unit work
- Phase 8 ideal compute time estimates derived from target peak rates
- Phase 8 target transfer-edge bandwidth rows joined with predicted H2D/D2H
  bytes and ideal transfer time estimates
- Phase 8 runtime allocation preflight for known memory capacity and required
  H2D/D2H transfer edges
- Phase 8 runtime allocation now consumes lowering records so fused-internal
  values are not materialized as device buffers
- Phase 9 first backend executable plan that expands verified Metal
  backend bindings into concrete backend call sequences from lowering regions
- Phase 9 first Metal/MLS kernel graph plan with explicit value-flow edges
  between backend calls
- Phase 9 first Metal/MLS kernel graph descriptors for output tensor metadata
  and typed operation attributes
- Architecture naming is now `metal_v0` for generated Metal/MLS graph/kernel
  planning and TRN2-like `npu_v0` for synthetic NPU performance modeling
- Architecture correction: the current Zig compiler-middle records are schema
  prototypes and extracted views. The intended compiler truth is a PjRTx MLIR
  state machine with verifiers, as defined in
  `mlir_state_machine_compiler_v0.md`.
- First MLIR dialect/API slice is specified in `pjrtx_mlir_dialect_v0.md`:
  build `MlirSession`, module state, target attachment, fusion/pressure MLIR
  facts, and extraction to `FusionGroup`.
- First `//next/pjrtx/compiler/mlir_state` implementation exists: it parses
  StableHLO text into an owned MLIR session, attaches module state and target
  attributes, verifies target-legal before fusion planning, stores
  fusion/pressure facts on MLIR, and extracts them back to `FusionGroup`.
- `//next/pjrtx/compiler/mlir_state` is now a dedicated Bazel package with local
  `types.zig`, `limits.zig`, `lowering_policy.zig`, and `passes.zig` files. The
  parent `//next/pjrtx/compiler` target orchestrates compile policy, while the
  subpackage owns the MLIR state-machine implementation and bounded attribute
  bridge.
- MLIR external pass execution now has a single package boundary:
  `passes.zig` owns MLIR pass-manager create/add/run/destroy, verifier enable,
  and pass-run result reporting. The session facade constructs pass data and
  keeps semantic diagnostics, but does not hand-roll each pass-manager
  lifecycle.
- The first external pass family has been split out of the package facade:
  `state_target_passes.zig` owns the state-probe and target-legality callbacks,
  while `attrs.zig` provides the small MLIR C API attribute helpers needed by
  pass families. Remaining fusion/lowering/runtime/backend callbacks still
  need the same treatment.
- Fusion pass callbacks have also left the facade: `fusion_passes.zig` owns
  StableHLO fusion-candidate discovery, candidate pressure metrics, V0 fusion
  decision stamping, and the local helper functions needed for that pass
  family. The public package facade still owns diagnostics and post-pass state
  checks.
- Placement and collective verifier callbacks have left the facade:
  `placement_collective_passes.zig` owns bounded placement-record validation,
  collective-plan attribute validation, and their state transitions. The typed
  commit/extract validators remain in the package facade until those record
  families receive their own storage/extraction files.
- `compileV0FromReader` now creates an `MlirSession`; the orchestrated compile
  path discovers fusion candidates, makes V0 fusion decisions, and extracts the
  final `FusionGroup` view through MLIR state.
- The old full-Zig fusion planner compatibility path is removed. There is no
  graph-only `planFusion`, no `FusionPlan` report artifact, no graph-only
  `planV0TraceReport`, and no side-channel attachment into `MlirSession`.
  Test and helper flows that need a full trace must enter through
  `planV0TraceReportFromMlirSession` with a live MLIR session.
- `pjrtx.fusion.plan` has been removed from the live V0 MLIR state machine.
  Durable fusion truth after `fusion_planned` is the enriched
  `pjrtx.fusion.candidates` stream.
  U128 byte counters are string-encoded inside the dictionaries until the real
  dialect defines integer width policy.
- `//next/pjrtx/vertical_slice:mlir_state_test` adds a runfiles-backed golden
  boundary summary proving that target and fusion/pressure facts are visible
  from MLIR state.
- `MlirSession` now owns an MLIR pass manager slot, and
  `runExternalStateProbePass` proves a Zig `callconv(.c)` external MLIR pass can
  run through MLIR's pass manager, verify PjRTx module state, and stamp
  `pjrtx.external_pass.proof = "ran"` on the module.
- `markTargetLegal` now routes the `target_attached -> target_legal` transition
  through `runTargetLegalExternalPass`, a Zig external MLIR pass that verifies
  required target attributes and stamps `pjrtx.target_legal.pass`.
- `runFusionCandidateDiscoveryExternalPass` now runs after target legality and
  before fusion planning on the orchestrated compile path. It walks StableHLO
  from MLIR, recognizes the first `tanh(add(dot_general, broadcast_in_dim))`
  shape, records graph instruction ids, tensor-byte-derived savings, launch
  reduction, and pressure deltas, and stamps a structured
  `pjrtx.fusion.candidates` array on the module. The legacy
  `pjrtx.fusion.candidates.*` count attrs are derived compatibility markers,
  not the source of truth.
- `runFusionPlanExternalPass` now enriches each structured candidate with the
  V0 decision, plan index, and decision reason before setting
  `fusion_planned`. It applies the V0 decision policy directly from MLIR
  candidate facts; there is no planner handoff path.
- Fusion extraction now reads enriched `pjrtx.fusion.candidates` entries, not
  `pjrtx.fusion.plan`, so the final `FusionGroup` report view and boundary
  summary come from the discovery-to-decision candidate object.
- Placement planning now commits V0 layout/tile/memory-space records to MLIR as
  `pjrtx.placement.records`, verifies them through a Zig external pass, advances
  module state to `placement_planned`, and extracts the final `PlacementRecord`
  report view from MLIR. The local Zig placement planner is now the temporary
  producer for this MLIR fact, not the report source of truth.
- Collective planning now commits the V0 collective decision record to MLIR as
  `pjrtx.collective.records`, verifies it through a Zig external pass, advances
  module state to `collectives_planned`, and extracts the final
  `CollectivePlanRecord` report view from MLIR. V0 still rejects executable
  collectives before schedule build, but that rejection is now visible at the
  MLIR state boundary first.
- Lowering planning now derives V0 executable-region records inside
  `mlir_state`. Compiler target legality supplies compact backend capability
  facts, then `mlir_state` derives the cost ledger from graph op/type facts,
  logical ops, bytes, dtype/op-class, formulas, approximations, and expected
  execution units. `commitLoweringPlan` first stamps that derived ledger into
  `pjrtx.performance.cost_ledger`.
  `pjrtx.lowering.records` and `pjrtx.lowering.region_facts` are then derived
  from MLIR-visible fusion candidates, placement records, and cost-ledger
  instruction links. A Zig external pass verifies both attributes, advances
  module state from `collectives_planned` to `lowering_planned`, and extracts
  final `CostLedgerEntry`, `LoweringRecord`, and `LoweringRegionFact` views
  from MLIR. `mlir_state` then derives memory traffic from target memory
  spaces, graph value flow, placement records, lowering records, and
  cost-ledger bytes. Performance, schedule, backend binding, kernel graph
  planning, and profile joins must use extracted lowering records as the
  verified executable-region boundary.
- Performance modeling now extends the already committed
  `pjrtx.performance.cost_ledger` with `pjrtx.performance.memory_traffic`,
  verifies both through a Zig external pass, advances module state from
  `lowering_planned` to `performance_modeled`, and extracts the final
  `CostLedgerEntry` and `MemoryTrafficRecord` report views from MLIR. Runtime,
  lowering, hardware, and backend-call summaries now consume extracted
  performance facts for predicted bytes, logical ops, memory traffic, and ideal
  roofline terms.
- Kernel codegen planning now derives V0 generated-kernel region records inside
  `mlir_state` from backend capability facts, graph instructions, lowering
  records, placement records, and memory-traffic records. It commits the
  derived rows to MLIR as `pjrtx.codegen.records`, verifies them through a Zig
  external pass, advances module state from `performance_modeled` to
  `codegen_planned`, and extracts the final `KernelCodegenRecord` report view
  from MLIR before tile legality and backend binding consume those facts.
- Schedule planning now derives V0 command and overlap records inside
  `mlir_state` from parameter IDs, return value IDs, extracted lowering
  records, and extracted cost ledger entries. It commits the derived rows to
  MLIR as `pjrtx.schedule.commands` and `pjrtx.schedule.overlaps`, verifies
  them through a Zig external pass, advances module state to `scheduled`, and
  extracts the final `ScheduleCommand` and `ScheduleOverlapRecord` report views
  from MLIR before backend binding and executable view creation consume them.
- Backend binding now commits V0 command-to-backend records to MLIR as
  `pjrtx.backend.bindings`, verifies them through a Zig external pass, advances
  module state to `backend_bound`, and extracts the final `BackendBinding`
  report view from MLIR before executable view creation and backend graph
  expansion consume it.
- Executable readiness now commits a V0 executable contract to MLIR as
  `pjrtx.executable.contract`, verifies that it matches the already verified
  schedule, backend binding, and kernel codegen facts through a Zig external
  pass, and advances module state to `executable_ready` before the Zig
  executable view is created.
- Backend executable planning now has a backend-owned MLIR bridge target,
  `//next/pjrtx/backend:mlir_bridge`, which commits concrete backend calls to MLIR
  as `pjrtx.backend.executable`, verifies them through a Zig external pass, and
  advances module state to `backend_executable_planned` without adding a
  compiler dependency on `//next/pjrtx/backend`.
- Backend executable facts are now extractable from MLIR after verification.
  The NPU and Metal vertical slices check extracted call metadata, operation
  names, instruction regions, input/output values, and expected execution units.
- Metal/MLS kernel graph planning now uses the same bridge to commit graph
  nodes and value-flow edges to MLIR as `pjrtx.backend.kernel_graph`, verifies
  them through a Zig external pass, and advances module state to
  `backend_kernel_graph_planned` before future command-buffer generation,
  allocator joins, or profiling consume those facts.
- Metal/MLS kernel graph facts are now extractable from MLIR after
  verification. The Metal bridge slice checks extracted node metadata, output
  tensor descriptors, compact attributes, and value-flow edges.
- Runtime allocation planning now has a runtime-owned MLIR bridge target,
  `//next/pjrtx/runtime:mlir_bridge`, which commits verified buffer reservations,
  command uses, lifetimes, memory spaces, and peak live device bytes to MLIR as
  `pjrtx.runtime.allocation` and advances module state to
  `runtime_allocation_planned`.
- Runtime allocation facts are now extractable from MLIR after verification.
  The execution vertical slice checks extracted allocation reservations,
  buffer-use access, lifetimes, and peak device bytes from the MLIR state.
- Runtime stream planning now uses the same bridge to commit command stream
  steps, wait events, start events, and done events to MLIR as
  `pjrtx.runtime.streams` and advances module state to
  `runtime_stream_planned`.
- Runtime stream facts are now extractable from MLIR after verification. The
  execution vertical slice checks extracted command, stream, wait event, start
  event, and done event fields directly.
- Runtime profiling now uses the same bridge to commit synthetic profile event
  rows to MLIR as `pjrtx.runtime.profile_events`, preserving command,
  instruction, byte, op, status, and synchronization join keys, and advances
  module state to `runtime_profiled`.
- Runtime profile events are now extractable from MLIR after verification. The
  execution vertical slice checks the extracted `RuntimeProfileEventFact`
  payload directly, including command, instruction, byte, op, status, and
  synchronization fields.
- Runtime profile joins now use the same bridge to commit explicit
  schedule-command, lowering-record, and explain-record relations to MLIR as
  `pjrtx.runtime.profile_joins`, so observed metrics are not inferred from
  report-only text. The join pass advances module state to
  `runtime_profile_joined`.
- Backend call profile joins now use the backend-owned MLIR bridge to commit
  concrete executable-call-to-profile-event rows as
  `pjrtx.backend.profile_joins`, preserving the compiler/backend dependency
  boundary while making generated or selected kernel calls explainable from
  MLIR state. The pass advances module state to `backend_profile_joined`.
- Runtime and backend profile joins are now extractable from MLIR after their
  verifier passes run. The execution vertical slice checks the extracted
  `RuntimeProfileJoinFact` and `BackendProfileJoinFact` values directly, so
  join readback no longer depends on summary string reconstruction.
- Report summaries now consume extracted MLIR facts for allocation, stream,
  runtime execution, lowering-profile observations, hardware utilization,
  backend executable calls, Metal/MLS kernel graph nodes, and backend call
  profile joins. Runtime execution and hardware summaries now also receive
  schedule commands extracted from MLIR, even after later backend/runtime states
  have been committed. Committed target, schedule, runtime, backend, cost, and
  memory-traffic facts must be read back through `mlir_state` before they are
  printed.
- Backend call-profile summaries now use extracted runtime profile events plus
  extracted backend profile joins for observed call metrics. The old runtime
  profile array is no longer the source of observed backend-call bytes, ops, or
  event IDs in that report path.
- Target hardware specs are now committed as `pjrtx.target.spec` and extracted
  through `extractTargetDescription`. Hardware utilization summaries consume
  memory spaces, transfer edges, execution units, and dtype rates from that
  extracted MLIR target.
- Cost ledger and memory traffic facts are now committed as
  `pjrtx.performance.cost_ledger` and `pjrtx.performance.memory_traffic` and
  extracted through `extractCostLedgerEntries` and
  `extractMemoryTrafficRecords`. Runtime execution, lowering profile, hardware
  utilization, and backend call-profile summaries consume these extracted MLIR
  performance views instead of the original compiler trace arrays.
- Lowering records and region facts are now derived into
  `pjrtx.lowering.records` and `pjrtx.lowering.region_facts`, then extracted
  through `extractLoweringRecords` and `extractLoweringRegionFacts`. The
  original Zig lowering array is no longer allowed to be the hidden source of
  executable-region decisions once the MLIR lowering boundary has verified
  those records.

## Documentation Audit Checkpoint

Status: current as of 2026-05-19.

The vision and V0 specs were rechecked before declaring the plan healthy. The
main direction remains correct: PjRTx must own lowering, fusion, tiling,
memory-space, collective, schedule, codegen, allocation, profiling, and
explainability decisions; MLIR state is the compiler truth; Zig reports are
extracted views and runtime/API handoff structures; unsupported work fails at
compile boundaries.

The important caveat is also now explicit: the current implementation is still
mostly a structured MLIR attribute/state bridge with Zig external passes. It is
not yet the final PjRTx MLIR dialect with native ops, attrs, canonicalization,
dialect conversion, bufferization, and backend kernel generation. That bridge is
acceptable only because committed facts are verified, extracted, tested, and no
longer hidden behind old Zig summary/report objects.

The next work must keep converting committed MLIR facts into typed extracted
views and then move from the attribute bridge toward real compiler lowering:
backend-binding derivation in `mlir_state`, transfer and collective traffic,
bufferization/allocation facts, real dialect attrs/ops, and backend-specific
kernel generation contracts.

The planner is intentionally still a baseline: it creates an explainable
H2D/backend/D2H trace with cost and lowering edges, but it does not yet perform
stream overlap, custom kernel generation, real stream execution, device
allocation calls, or hardware profiling. The allocation plan records buffer
placement, sizes, lifetimes, and peak live device bytes, but does not yet call
a real device allocator. The stream plan records event waits, start events, and
done events, and those stream facts are now committed into MLIR, but the runtime
does not yet submit commands to a real stream executor. The
runtime execution summary compares predicted cost metrics with synthetic
profile metrics, and those profile facts are now committed into MLIR, but it
does not yet collect hardware counters. The hardware
utilization summary reports memory-space pressure, predicted execution-unit
work, target dtype/op-class rates, transfer-edge bytes, ideal compute time,
and ideal transfer time estimates from existing target/cost/allocation records.
Runtime allocation planning now rejects plans that exceed known memory capacity
or schedule a host/device transfer that cannot be matched to a target transfer
edge. Runtime allocation is lowering-aware: source graph values that are
internal to a fused lowering remain traceable graph values, but they do not get
standalone runtime buffers unless a later lowering boundary exposes them. The
verified allocation plan is now also committed into MLIR before stream planning
or profiling consumes it.
The first Metal bridge slice expands a verified backend binding into an
owned backend executable plan with one call per lowering region, but it does
not yet lower those calls into submitted Metal command buffers. The Metal/MLS
kernel graph records one node per planned backend call, including fused region
nodes, and value-flow edges between producer and consumer nodes. Those nodes
and edges are now committed into MLIR before future Metal codegen consumes
them. Kernel graph nodes also record output tensor metadata and attributes
needed by future Metal codegen.
The first compiler-middle reroute slice now adds separate pass, fusion,
placement, and collective artifacts under `//next/pjrtx/compiler`, plus a
`//next/pjrtx/vertical_slice:lowering_tests` target. Fusion, placement, and
collective records are now part of `TraceReport` and are emitted before the
cost ledger and schedule. `compileV0FromReader` now owns the full input,
StableHLO ingest, Shardy-aware pass report, graph import, target selection, and
trace planning path, so final compiled traces carry MLIR pass records too. Full
trace planning requires an `MlirSession`; tests that start after graph import
must still provide the session that owns the compiler-state facts.
The planner now emits lowering records from compiler-middle regions: the V0
matmul remains an explicit backend kernel graph boundary, while the compatible
broadcast/add/tanh chain lowers as one elementwise fusion region linked to all
three cost entries and validated placement records. Those lowering records are
committed to MLIR and read back before memory traffic and performance modeling
consume them.

## Active Reroute: MLIR Compiler Middle First

Status: active as of 2026-05-19.

The implementation started to deepen backend executable planning, runtime
allocation summaries, and Metal/MLS kernel graph descriptors before the compiler
middle was strong enough. That is useful infrastructure, but it risks repeating
the common plugin trap: XLA remains the real compiler, while a new project only
implements a StreamExecutor-like submission layer. PjRTx must own the lowering,
fusion, tiling, layout, memory-space, and collective decisions that determine
performance and correctness.

The second correction is that the compiler middle should not become a large Zig
record machine. The current records are useful because they define the evidence
the compiler must produce, but the transformed program, legality state,
pressure facts, collective facts, schedule facts, and codegen facts should live
in MLIR. Zig should orchestrate, extract, validate, print, and hand off runtime
views.

Remaining priority is now:

1. Move backend-binding derivation into `mlir_state`, so command-to-backend
   records follow the same derive/commit/verify/extract path as lowering,
   performance, codegen, and schedule.
2. Add transfer memory-traffic facts for H2D/D2H commands and keep them distinct
   from compute lowering traffic.
3. Add collective traffic, collective-engine, and overlap facts before claiming
   distributed or NPU/TPU-like performance behavior.
4. Move tile legality and bufferization/allocation pressure into MLIR facts so
   allocator planning consumes verified compiler state, not local runtime
   inference.
5. Replace the temporary structured attribute shim with real dialect attrs or
   ops using minimal TableGen/C++ dialect plumbing and a narrow C-compatible
   shim consumed from Zig.
6. Deepen Zig-owned external MLIR passes for compiler policy where the MLIR C
   API is sufficient. The pass-manager lifecycle now lives in
   `//next/pjrtx/compiler/mlir_state:passes`; next, pass callback structs should move
   behind smaller pass-family files while MLIR-native rewrite/dialect-conversion
   glue is reserved for boundaries that need it.
7. Expand runfiles-backed MLIR boundary tests from summaries to raw MLIR
   snapshots or FileCheck-style state-boundary checks where the spelling is
   stable enough.
8. Define backend-specific kernel generation contracts for Metal/MLS and
   TRN2-like `npu_v0`: kernel IR, math policy, tile/scratch requirements,
   library-call selection, generated code artifacts, and profile join keys.

XLA is a reference for hard problems and proven pass families. It must not be a
hidden dependency that silently supplies the compiler intelligence for PjRTx.
StreamExecutor-like runtime submission begins after PjRTx has produced a
verified executable schedule.

The local XLA coverage map now makes that concrete: PjRTx needs the same broad
responsibility categories as XLA, including HLO/StableHLO import, pass
pipelines, fusion, tiling, Shardy/SPMD, collective lowering, layout,
memory-space assignment, buffer assignment, scheduling, backend codegen,
runtime IO, allocation, profiling, and autotuning. The difference is that PjRTx
must expose each category through MLIR state and extracted records that join
source operations to hardware facts, generated or selected kernels, collective
commands, and profile events.

## Package Layout

Use `package_boundaries_v0.md` as the canonical package graph and dependency
contract. This workplan only names implementation phases; it does not duplicate
the full package layout.

Operational rules for this queue:

- no new public architecture code belongs under `//src/...`
- no new package may centralize shared vocabulary as `//next/pjrtx/core`
- new records move to the package that owns their invariants
- logic lives in small boundary structs, and package roots stay composition
  facades rather than implementation buckets
- package intent docs stay brief and point back to the canonical specs

## Phase 0: Build Skeleton

Create Bazel packages and empty test targets:

```text
//next/pjrtx/target:unit_tests
//next/pjrtx/mlir:unit_tests
//next/pjrtx/dialects:unit_tests
//next/pjrtx/compiler/facts:unit_tests
//next/pjrtx/compiler:unit_tests
//next/pjrtx/backend:unit_tests
//next/pjrtx/runtime:unit_tests
//next/pjrtx/report:unit_tests
//next/pjrtx/vertical_slice:report_test
```

Acceptance:

- `bazel test //next/pjrtx/...` discovers the skeleton targets
- packages have strict visibility
- `//next/pjrtx/target` imports no compiler/runtime/backend/plugin/report package
- `//next/pjrtx/compiler` imports no runtime/backend/plugin package
- no package imports `//next/pjrtx/core` after migration
- add a short `//next/pjrtx/...` coding policy note that points to the adapted
  TigerStyle rules in the architecture doc
- the coding policy says to prefer Zig `std` primitives before adding local
  utility functions
- the coding policy says comments document intent and invariants, not obvious
  mechanics
- the coding policy says domain logic belongs in small Zig structs with clear
  boundaries, and packages should be split across meaningful files rather than
  centralized implementation buckets
- the coding policy records the preferred typed initialization style:
  `const value: Type = try .init(allocator);`
- the coding policy says tests should avoid inline `@as(T, value)` acceptance
  casts when a typed expected value is clearer
- the coding policy says schema docs remain centralized under `next/docs/specs`,
  while code carries inline intent comments for implemented invariants

## Phase 1: Segregated Facts, Target Types, And Reports

Implement:

- typed IDs
- tensor type records
- source refs
- trace records
- target description records
- validators
- `std.Io.Writer` formatting
- explicit deinit paths

Initial ownership:

- target records and validators live under `//next/pjrtx/target`
- graph/compiler records live under `//next/pjrtx/compiler/facts`
- backend records live under `//next/pjrtx/backend/facts`
- runtime records live under `//next/pjrtx/runtime/facts`
- deterministic report rendering lives under `//next/pjrtx/report`

Acceptance:

- invalid ID links fail validation
- unknown target fields print as `unknown`
- report order is deterministic
- ownership/deinit smoke tests pass
- IDs and serialized counts use explicit-width integer types
- public record names avoid abbreviations except for established compiler or
  hardware terms
- formatting, sorting, hashing, and test allocation use Zig `std` primitives
  unless a PjRTx-specific abstraction is justified
- tests follow the no-inline-`@as` acceptance style
- each package includes inline intent comments for implemented contracts and
  points readers back to central specs
- `rg "//next/pjrtx/core|pjrtx/core" pjrtx` is empty when the migration is complete

## Phase 2: Report Writer And Normalizer

Implement stable text report output before execution exists.

Acceptance:

- normalized golden comparison works
- timing fields redact
- target-dependent device names normalize to `<device-name>`
- report validation runs before golden comparison

## Phase 3: StableHLO Ingest

Use Bazel MLIR, StableHLO, and Shardy C API targets. Do not write a custom
StableHLO parser.

Acceptance:

- `tanh_dot_bias.mlir` parses and verifies
- malformed StableHLO emits diagnostics through `std.Io.Writer`
- source operation order is stable
- Shardy metadata is preserved when present

## Phase 4: Typed Graph Import

Import the V0 operation set:

- parameter
- constant when needed
- `stablehlo.dot_general`
- `stablehlo.add`
- `stablehlo.tanh`
- `stablehlo.broadcast_in_dim`
- return

Acceptance:

- unsupported convolution fails during import
- dynamic shapes fail in V0
- dot shape errors fail before backend binding
- every graph instruction has source provenance

## Phase 5: Target Selection And Legality

Implement `metal_v0` and `npu_v0` target descriptions.

Acceptance:

- unknown target fails
- unknown performance rate is allowed and explicit
- unknown execution capability fails
- required memory spaces and transfer edges validate
- no graph region proceeds without a target execution path

## Phase 6: Cost, Lowering, And Schedule

Status: superseded as a single phase by Phase 6A through Phase 6E. The old
baseline planner may stay as scaffolding, but no further backend/runtime depth
should be added until the compiler-middle artifacts below exist.

## Phase 6A: MLIR Lowering Pass Pipeline

Use Bazel MLIR, StableHLO, and Shardy C API targets to build an explicit pass
pipeline artifact. Do not treat XLA as the compiler behind PjRTx. PjRTx may use
MLIR/StableHLO passes and XLA as a design reference, but every pass that affects
the executable must appear in the PjRTx report.

PjRTx pass policy should be written in Zig when possible. The immediate MLIR
pass-manager milestone is a Zig external pass created through the MLIR C API,
not a C++ optimizer hidden behind the shim. Minimal C++/TableGen is reserved
for real dialect definitions, generated op/type/attribute plumbing, verifier
integration, and rewrite/dialect-conversion helpers that MLIR does not expose
cleanly through C.

The pass family contract lives in `compiler_pass_pipeline_v0.md`. This phase
implements the first records and tests from that spec; later phases deepen the
individual pass families without changing the no-fallback contract.

Current implementation:

- `//next/pjrtx/compiler` owns a typed V0 pass catalog with stable pass names, stages,
  artifact kinds, target facts, preserved invariants, invalidated analyses, and
  emitted record names
- MLIR pass reporting uses that catalog for parse, verify, collective payload
  preservation/import, canonicalization/CSE boundary, and Shardy metadata
  reporting
- the catalog is pinned by vertical lowering tests so future fusion, tiling,
  collective, allocation, and backend-codegen passes have an existing slot
  instead of being hidden behind backend execution
- `broadcast_simplify` now runs after typed graph import and before downstream
  planning; it removes identity broadcasts and records applied/rejected graph
  rewrites through `GraphRewriteRecord`
- `reshape_transpose_fold` now imports StableHLO reshape/transpose, folds only
  identity mappings, and rejects non-identity transforms through target legality
  until explicit lowering, memory traffic, and allocation support exists
- `matmul_epilogue_fusion_select` is now a cataloged compiler pass, and the
  V0 planner records the dot-plus-bias-plus-activation epilogue candidate as
  rejected until kernel IR, math-policy, and backend epilogue support are
  explicit

Implement:

- pass pipeline configuration
- MLIR pass manager ownership in `MlirSession`
- one Zig external MLIR pass with a stable name and state-boundary test
- pass result records with before/after module fingerprints
- verifier/canonicalization/CSE records
- StableHLO-to-PjRTx graph import boundary
- Shardy metadata preservation and propagation records
- diagnostics through `std.Io.Writer`

Acceptance:

- every pass has a stable name, status, and diagnostic section
- failed pass reports `executable_created=false`
- pass output fingerprints are deterministic in tests
- unsupported dialect/op lowering fails before typed graph import or target
  legality, depending on where it is discovered
- no pass silently drops source provenance, Shardy metadata, tokens, regions, or
  side-effect information
- the harness includes a focused lowering-pass golden for the V0 fixture

## Phase 6B: Fusion Planning

Implement fusion as a compiler decision, not as a backend side effect.

Implement:

- fusion candidate discovery
- fusion group records
- accepted and rejected fusion reasons
- source instruction membership
- dtype/layout/shape compatibility checks
- register, local-memory, launch-count, and bytes-saved estimates when the
  target provides enough data

Acceptance:

- `add` + `tanh` can form an elementwise fusion record for the V0 workload
- `dot+broadcast/add/tanh` records a matmul epilogue candidate and explains why
  V0 rejects it instead of hiding the decision in backend codegen
- the rejected matmul epilogue candidate records split-versus-fused pressure
  deltas before acceptance is allowed
- rejected fusion is printable when a boundary matters
- fusion never crosses unsupported collective, token, side-effect, or
  correctness boundaries
- fusion groups link to cost ledger entries and later backend kernels or kernel
  graph nodes
- tests cover a pure elementwise chain and a dot-plus-bias-plus-activation case

## Phase 6C: Layout, Tiling, And Memory-Space Planning

Implement first-class plans for the decisions that feed real performance on
TPU-, Neuron-, and NPU-like targets.

Implement:

- layout plan records
- tile plan records
- memory-space assignment records
- HBM/SRAM/scratchpad pressure estimates
- reserved DMA chunking and prefetch/eviction facts
- target-specific constraints for `metal_v0` and TRN2-like `npu_v0`

Acceptance:

- matrix work records logical tile choices and uses bounded target-aware tiles
  for NPU V0 instead of silently inheriting full output shape
- elementwise work records vectorization or scalar fallback as an explicit
  decision; unsupported choices fail
- memory-space assignment records why values are placed in host, HBM, local
  SRAM, or unified device memory
- tile/layout/memory records link to source instructions, cost entries, and
  backend binding records
- no command schedule is built from a graph that lacks required layout or memory
  placement records

## Phase 6D: Collective Specs And Lowering Skeleton

Collectives must be represented before PjRTx claims distributed or NPU/TPU-like
performance architecture.

Implement:

- collective graph payloads for all-reduce, reduce-scatter, all-gather,
  all-to-all, collective-permute, send, and recv
- replica/partition group records
- channel/rendezvous records
- collective legality diagnostics
- collective lowering records with selected or rejected algorithms
- reserved collective engine binding facts for `npu_v0`

Acceptance:

- unsupported collective algorithms fail during compile, not runtime
- group, dtype, layout, and token constraints are verified
- collective lowering can explain why it selected direct, ring, tree, split, or
  unsupported behavior, even when V0 only supports diagnostics
- collectives form hard boundaries for unsafe fusion
- report goldens include at least one unsupported collective failure case

## Phase 6E: Cost, Lowering, And Schedule

Implement:

- cost ledger formulas
- lowering records
- rejected alternatives when meaningful
- H2D/backend/D2H schedule commands
- transfer/compute and compute/transfer schedule overlap records
- schedule verification

Acceptance:

- FLOPs and bytes appear by source instruction
- `tanh` is classified as `transcendental`
- backend commands link to lowering and cost records
- overlap planning emits explicit serialized edges before backend binding, with
  command IDs, stream IDs, dependency kind, and a stable reason
- missing dependency fails schedule validation
- unlowerable programs never reach runtime
- cost records link through fusion, tile/layout, memory-space, and collective
  plans when those plans exist
- schedule commands are built from compiler-middle records, not directly from
  raw StableHLO operations

## Phase 7: Backend Binding Records

Status: active for MLIR-state-derived codegen records plus backend binding.

Implement backend binding without requiring full kernel generation. The
compiler must still derive and extract `KernelCodegenRecord` rows through
`mlir_state` first, so backend binding is consuming named generated or selected
kernel candidates rather than raw StableHLO or opaque lowering helpers.

Acceptance:

- kernel codegen records exist before backend bindings
- each executable lowering region records backend kind, operation name, command
  ID, graph instructions, cost ledger IDs, memory traffic IDs, a small kernel
  IR shape/value-flow/tile/memory-pressure summary, and expected unit when the
  target can provide one
- tile legality verifies codegen result/tile memory requirements against target
  capacities before backend binding
- Metal binding records `metal_mls_graph_execute`
- NPU binding records expected execution units
- every backend execute command has exactly one binding
- missing backend capability fails compile
- kernel graphs are recorded as intentional codegen/lowering artifacts, not
  fallback
- backend bindings preserve the compiler-middle region boundaries that later
  backend executable planning consumes

## Phase 8: NPU Profile Path

Use NPU to prove hardware-aware reporting before generated kernels.

Acceptance:

- synthetic profile events link to schedule commands
- memory-space and execution-unit utilization is visible
- transfer-edge utilization and ideal transfer time is visible
- known memory capacity and required transfer edges are enforced before runtime
  execution
- predicted FLOPs/bytes appear beside observed/synthetic event data
- lowering-region profile rows join matmul and fused elementwise lowerings to
  exact profile events
- explain records for lowerings can be joined to the exact profile events that
  measured them
- backend call profile summaries expose per-kernel predicted ops/bytes,
  observed ops/bytes, expected unit, ideal compute time, per-memory-space
  bytes/ideal time, and limiting resource
- `mlir_state` derives memory traffic records that backend summaries consume
  through extracted performance views
- hardware utilization summaries expose memory traffic records directly, so
  HBM boundary traffic and local tile traffic are visible outside backend rows
- hardware utilization summaries expose lowering roofline rows, so performance
  limits can be reviewed before backend-specific kernel graph details
- lowering roofline rows include profile event joins so predicted and observed
  lowering work can be compared without backend-private state
- allocation planning consumes lowering regions, not raw per-op instruction
  lists, so fused intermediates do not become standalone device buffers
- report tests do not require local accelerator hardware

## Phase 9: Metal Execution Bridge

Status: active for the verified Metal/MLS bridge after Phase 6A through
Phase 6D produced compiler-middle records.

Bind the supported current backend path only after compile gates and report
validation exist.

Acceptance:

- verified Metal bindings expand lowering regions into deterministic backend
  executable calls
- Metal executable calls expand into deterministic MLS kernel graph nodes and
  value-flow edges
- MLS kernel graph summaries expose dtype, rank, and operation attributes
- elementwise fusion lowering produces one `metal_mls_elementwise_fusion_kernel`
  call with all fused instruction provenance and explicit external inputs and
  outputs
- backend call profile summaries map Metal/MLS calls to exact lowering-region
  profile events when those events exist
- backend executable planning rejects binding/report mismatch and non-backend
  operations before runtime execution
- supported V0 workload executes through verified schedule
- reference oracle compares outputs in tests only
- profile events include H2D/backend/D2H
- no production path can call reference execution

## Phase 10: PJRT/JAX Smoke

Expose the V0 path through the new PJRT adapter.

Acceptance:

- JAX smoke compiles and executes the V0 workload when local backend exists
- missing backend can skip only when the test is configured to allow skip
- diagnostics map to PJRT errors without losing internal stage labels
- explain report remains available after PJRT execution

## Stop Conditions

Stop and update docs before continuing if implementation discovers:

- a new public artifact type
- a new fallback-like behavior
- a correctness relaxation
- a new target capability category
- a report field that cannot be validated
- ownership that cannot be expressed with clear `deinit`

Architecture is part of the implementation. If code needs a new rule, the docs
must say the rule.
