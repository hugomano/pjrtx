# Compiler Pass Pipeline V0

This spec defines how PjRTx should grow an XLA-like optimization pipeline
without inheriting XLA's opacity. PjRTx should have many compiler passes, but
each pass must be explainable, validated, and connected to MLIR state plus
trace records extracted from that state.

XLA HLO passes are a useful reference because they encode years of hard
compiler work: canonicalization, algebraic simplification, fusion, layout,
collective handling, buffer assignment, scheduling, and backend lowering. The
PjRTx goal is not to avoid that structure. The goal is to make the structure
composable and inspectable from PJRT input down to generated or selected
kernels.

Use `xla_coverage_map_v0.md` as the local reference checklist for XLA-side
coverage. PjRTx should match the responsibility categories over time, but every
category must become a typed, validated, and printable PjRTx artifact instead
of an opaque backend behavior.

Use `mlir_state_machine_compiler_v0.md` as the direction for the compiler
middle. Current Zig records are V0 schema prototypes and extracted report
surfaces, but they should be moved out of the bootstrap `//next/pjrtx/core` package
by ownership. The long-term pass pipeline updates PjRTx MLIR dialect state,
verifies state transitions, and then extracts records for reports, runtime
handoff, and golden tests.

Use `pjrtx_mlir_dialect_v0.md` for the first implementation slice. The first
MLIR-owned compiler decision should be fusion plus pressure delta, extracted
back into the current `FusionGroup` report.

## Design Rule

Use many small passes with explicit contracts.

Do not build one opaque optimizer that mutates the program until it happens to
run. Do not hide lowering inside a backend or rely on XLA as an invisible
compiler. Every pass that affects correctness, performance, memory movement,
collectives, kernel selection, or scheduling must leave a stable MLIR fact and
an extractable report record.

The preferred implementation language for PjRTx pass policy is Zig. A pass can
be a straight-line Zig function during the V0 shim, or a Zig external MLIR pass
registered with MLIR's pass manager through the C API. Use minimal
MLIR-native C++/TableGen only for real dialect definitions, generated op
adaptors, parsers/printers/verifiers, and rewrite/dialect-conversion facilities
that are not available through the C API.

The split is:

```text
Zig:
  policy, cost models, target legality, diagnostics, extraction, external pass
  callbacks, harnesses, reports

MLIR-native C++/TableGen:
  dialect registration, ops/types/attrs, parser/printer/verifier integration,
  pattern-rewrite and dialect-conversion glue where needed
```

No pass may hide a meaningful decision inside the C++ shim. The shim exposes
MLIR extension points; the PjRTx pass contract still owns the explanation.

## Why Passes Matter

PjRTx needs passes because performance depends on sequences of decisions:

- canonical forms expose fusion and tiling opportunities
- correctness checks prevent unsafe rewrites
- fusion changes launch count, memory traffic, and backend kernel shape
- layout and tiling determine HBM, SRAM, scratchpad, and vector-unit pressure
- sharding decisions create collective and communication work
- collective lowering affects overlap, bandwidth, and synchronization
- cost and roofline estimates need stable units of work
- schedule and allocation depend on lowering regions, not raw operations
- profile feedback must join back to the decisions that produced kernels

A pass pipeline is good when these decisions are visible. It is bad when the
only explanation is "the optimizer did it."

## Pass Contract

Every pass should have a stable contract.

```zig
Pass
  stable_name
  stage
  input_artifact_kind
  output_artifact_kind
  required_target_facts
  preserved_invariants
  invalidated_analyses
  emitted_or_updated_mlir_state
  extracted_records
  failure_diagnostics
```

Implementation APIs can be idiomatic Zig error unions, but the conceptual shape
is:

```text
input artifact + target facts + compile options + diagnostics writer
  -> output artifact or compile error
```

Rules:

- diagnostics are written through `std.Io.Writer`
- large inputs use `std.Io.Reader`
- extracted pass records are printed through `std.Io.Writer`
- pass output is deterministic for golden tests
- pass names are stable and not derived from function pointers
- passes may be conservative, but not silent
- every pass that mutates executable semantics updates verified MLIR state and
  extracts a trace or explain record
- failed passes must stop executable creation
- no pass may drop source provenance, Shardy metadata, tokens, side effects, or
  regions unless a later explicit MLIR fact and extracted record explain why
  that information became irrelevant

## Required Invariants

Every pass must declare which invariants it preserves or establishes:

- graph values have valid shapes, dtypes, and layouts
- instruction payloads match instruction kind
- StableHLO semantics are preserved
- dtype conversions are explicit and legal
- reduction and transcendental behavior follows compile options
- Shardy metadata is preserved or intentionally consumed
- collective token and channel semantics are preserved
- source references remain joinable to frontend operations
- memory-space IDs refer to the target model
- lowering records cover all executable graph regions
- schedule commands link to lowering and cost records
- backend bindings are legal for the selected target

Correctness invariants come before performance transformations.

## Pipeline Families

The pass pipeline should be organized into families, not a flat unstructured
list.

The families below intentionally mirror XLA-scale responsibilities: import,
simplification, decomposition, sharding, collectives, fusion, layout, tiling,
memory planning, cost modeling, lowering, scheduling, backend binding, codegen,
profiling, and autotuning. V0 can implement tiny conservative slices, but it
must leave the slots visible so the architecture does not drift into a runtime
plugin shape.

### 1. Input And StableHLO Ingest

Purpose:

- parse StableHLO/VHLO artifacts
- verify MLIR module structure
- build source tables
- preserve locations and frontend operation names

V0 status:

- implemented as StableHLO text ingest through MLIR/StableHLO/Shardy C APIs
- no custom StableHLO parser

Representative passes:

- `stablehlo_parse`
- `mlir_verify`
- `stablehlo_source_table_build`
- `stablehlo_unsupported_dialect_gate`

### 2. MLIR Canonicalization And Metadata Preservation

Purpose:

- run safe canonicalization and CSE
- keep source and Shardy metadata visible
- reject unsupported regions or side effects early

Representative passes:

- `mlir_canonicalize`
- `mlir_cse`
- `stablehlo_shape_refine`
- `shardy_metadata_propagation`
- `token_side_effect_gate`

MLIR facts and extracted records:

- `MlirPassRecord`
- input and output fingerprints
- source provenance preservation flag
- Shardy metadata preservation flag

### 3. Typed Graph Import And Verification

Purpose:

- convert supported StableHLO into PjRTx graph values and graph instructions
- verify shape, dtype, layout, payload, and source references

Representative passes:

- `pjrtx_graph_import`
- `pjrtx_graph_verify`
- `pjrtx_dynamic_shape_gate`

MLIR facts and extracted records:

- `GraphValue`
- `GraphInstruction`
- `SourceRef`

### 4. Target Legality

Purpose:

- prove that every graph region has a legal path on the selected target
- fail before cost, lowering, schedule, backend binding, or runtime when support
  is missing

Representative passes:

- `target_select`
- `target_feature_legality`
- `dtype_legality`
- `layout_constraint_legality`
- `memory_space_legality`
- `collective_capability_legality`

Rules:

- no fallback
- no runtime repair
- no backend-private target interpretation

### 5. Correctness And Algebraic Normalization

Purpose:

- simplify the graph without changing mathematical meaning
- make later fusion and tiling easier

Representative passes:

- `constant_fold`
- `dead_code_eliminate`
- `broadcast_simplify`
- `reshape_fold`
- `transpose_fold`
- `algebraic_simplify_strict`

Rules:

- no unsafe reassociation unless compile options permit it
- no silent precision contraction
- no relaxation of NaN, infinity, signed-zero, or reduction semantics
- every relaxed math mode must be explicit in options and explain records

V0 status:

- mostly future work, except verification and conservative graph import

### 6. Fusion Planning

Purpose:

- discover and select groups that should lower together
- make launch count, memory traffic, and backend kernel regions explicit

Representative passes:

- `fusion_candidate_discovery`
- `elementwise_fusion_select`
- `producer_consumer_fusion_select`
- `matmul_epilogue_fusion_select`
- `reduction_fusion_select`
- `fusion_legality_verify`

MLIR facts and extracted records:

- `FusionGroup`
- accepted or rejected decision
- source instruction IDs
- bytes saved
- launch count reduction
- pressure delta for rejected or accepted candidates when memory pressure
  changes the decision
- reason and rejected alternatives

Rules:

- fusion must not cross collective, token, side-effect, or unsafe correctness
  boundaries
- fusion must account for target memory and execution-unit constraints
- fusion decisions feed cost, lowering, schedule, backend binding, and profile
  joins

V0 status:

- elementwise chain fusion is represented by `FusionGroup`
- `broadcast/add/tanh` lowers as one elementwise-fusion region
- `dot+broadcast/add/tanh` remains rejected, but records a pressure delta:
  two split kernels versus one fused kernel, split peak live bytes, fused live
  bytes, additional live bytes, and global bytes saved

### 7. Sharding And Collective Planning

Purpose:

- keep Shardy metadata meaningful
- identify collective requirements
- choose or reject collective algorithms
- expose communication and synchronization costs

Representative passes:

- `shardy_mesh_import`
- `sharding_propagate`
- `collective_detect`
- `collective_group_verify`
- `collective_group_channel_verify`
- `collective_algorithm_select`
- `collective_overlap_plan`
- `collective_lowering_verify`

MLIR facts and extracted records:

- `CollectivePlanRecord`
- future collective group/channel records
- future collective traffic records

Rules:

- collectives are hard fusion boundaries unless a pass proves otherwise
- unsupported collectives fail during compile
- group, channel, token, dtype, shape, and layout facts must remain visible

V0 status:

- `stablehlo.all_reduce` is imported as a typed graph payload before algorithm
  selection
- `collective_group_channel_verify` validates imported replica groups, channel
  handles, compile topology, and token schedulability before algorithm
  selection
- `collective_algorithm_select` records `decision=no_collectives` and
  `algorithm=none` for graphs without collective work
- unsupported collective payloads fail before schedule build with diagnostics
  that name rejected algorithms: direct, ring, tree, and split
- Shardy metadata preservation is reported

### 8. Layout, Tiling, And Memory-Space Planning

Purpose:

- decide how tensors are physically traversed and where bytes live
- expose HBM, SRAM, scratchpad, unified memory, and transfer pressure

Representative passes:

- `layout_select`
- `tile_shape_select`
- `memory_space_assign`
- `prefetch_plan`
- `eviction_plan`
- `memory_traffic_refine`
- `tile_legality_verify`

MLIR facts and extracted records:

- `PlacementRecord`
- `MemoryTrafficRecord`
- future tile/layout-specific records

Rules:

- layout and tile choices are target-aware
- memory-space choices must refer to the target model
- memory traffic is a compiler fact, not backend formatting
- later backend codegen consumes verified MLIR state and extracted views instead
  of rediscovering these facts

V0 status:

- placement records include layout, target-aware logical tile shape, result
  memory, and tile memory
- `tile_shape_select` keeps Metal V0 whole-tensor tiles, but bounds NPU V0
  local-SRAM tiles for matrix and fusible elementwise work
- `memory_traffic_refine` records global-memory and local-memory traffic for
  lowering regions inside `mlir_state` from target memory spaces, graph value
  flow, placement records, lowering records, and cost-ledger bytes;
  host/device DMA and interconnect kinds are schema-visible but wait for
  transfer and collective lowering records before they are emitted

### 9. Cost And Roofline Planning

Purpose:

- attach predicted work to graph/lowering decisions
- estimate ideal compute and memory lower bounds from target facts
- identify limiting resources before backend codegen

Representative passes:

- `logical_cost_estimate`
- `dtype_rate_join`
- `memory_bandwidth_join`
- `lowering_roofline_estimate`
- `transfer_roofline_estimate`

MLIR facts and extracted records:

- `CostLedgerEntry`
- `MemoryTrafficRecord`
- hardware utilization report rows
- explain records when a cost changes a decision

Rules:

- cost formulas must be stable and human-readable
- units must be explicit: bytes, ops, ns, ps, bytes per second
- ideal lower bounds are not observed performance
- profile rows must join back to cost and lowering records

V0 status:

- compiler target legality supplies compact backend capability facts, while
  `mlir_state` derives cost ledger rows from graph op/type facts, formulas,
  logical ops, bytes, dtype/op-class, and expected execution units
- the derived cost ledger is stamped into MLIR before lowering-region
  formation, then extracted back before `mlir_state` derives memory traffic
- memory traffic, backend call roofline, and lowering roofline rows exist
- lowering roofline rows join to profile events

### 10. Lowering Region Formation

Purpose:

- convert graph decisions into backend-executable regions
- define the unit that backend kernel generation or library selection consumes

Representative passes:

- `lowering_region_form`
- `elementwise_fusion_region_lower`
- `matmul_kernel_region_lower`
- `collective_region_lower`
- `transfer_region_lower`

MLIR facts and extracted records:

- `LoweringRecord`
- `LoweringRegionFact`
- rejected alternatives
- cost ledger references
- source graph instruction IDs
- fusion-group reference when the lowering came from accepted fusion
- placement record references, logical tile shape, result memory, tile memory,
  and codegen-region intent

V0 status:

- matmul lowers as one backend kernel graph region
- compatible elementwise chain lowers as one fused region
- lowering-region records and region facts are derived into MLIR as
  `pjrtx.lowering.records` and `pjrtx.lowering.region_facts` from MLIR fusion
  and placement facts plus cost-ledger instruction links, verified through
  `runLoweringPlanExternalPass`, extracted back to `LoweringRecord` and
  `LoweringRegionFact`, and consumed by memory traffic, performance, schedule,
  backend binding, kernel graph planning, and profile joins from that extracted
  view

### 11. Schedule, Allocation, And Runtime Command Planning

Purpose:

- build executable order, dependencies, memory lifetime, and streams
- keep runtime as schedule execution, not operation interpretation

Representative passes:

- `schedule_build`
- `schedule_overlap_plan`
- `schedule_verify`
- `buffer_lifetime_plan`
- `buffer_alias_plan`
- `device_allocation_plan`
- `stream_assign`
- `event_dependency_plan`
- `transfer_overlap_plan`

MLIR facts and extracted records:

- `ScheduleCommand`
- `ScheduleOverlapRecord`
- allocation plan
- stream plan
- explain records

Rules:

- schedule commands must link to lowering and cost records
- allocation consumes lowering regions, not raw graph instructions
- runtime must not invent unsupported execution paths

V0 status:

- H2D/backend/D2H schedule rows are derived inside `mlir_state` from
  parameter IDs, return value IDs, extracted lowering records, and extracted
  cost ledger entries
- transfer/compute and compute/transfer dependencies are derived inside
  `mlir_state` as `ScheduleOverlapRecord` rows with serialized V0 decisions
- allocation and stream summaries are deterministic
- real allocator and stream executor calls are future work

### 12. Backend Binding And Kernel Generation

Purpose:

- bind lowering regions to generated kernels, selected library calls, DMA, or
  collective engines
- keep backend-specific implementation visible through trace records

Representative passes:

- `backend_binding_select`
- `backend_binding_verify`
- `kernel_graph_build`
- `kernel_codegen_plan`
- `library_algorithm_select`
- `kernel_launch_plan`

MLIR facts and extracted records:

- `BackendBinding`
- backend executable plan
- backend kernel graph plan
- backend call profile rows

Rules:

- library calls are intentional lowerings, not fallback
- generated kernels must link to lowering, cost, memory traffic, and profile
  events
- backend code must not rediscover StableHLO semantics from strings

V0 status:

- Metal/MLS backend executable and kernel graph summaries exist
- NPU V0 backend call summaries model TRN2-like units

### 13. Profile Feedback And Autotuning

Purpose:

- join measured events back to compiler decisions
- identify prediction errors
- feed future fusion, tiling, layout, algorithm, and schedule decisions

Representative passes:

- `profile_event_join`
- `lowering_profile_compare`
- `kernel_profile_compare`
- `autotune_candidate_generate`
- `autotune_result_select`
- `profile_guided_recompile`

MLIR facts and extracted records:

- `ProfileEvent`
- explain records with `profile_event_ids`
- lowering roofline profile joins
- backend call profile joins

Rules:

- golden tests redact raw timings but preserve stable join keys
- forced synchronization must be recorded
- profile-guided changes must be reproducible or explicitly marked empirical

V0 status:

- synthetic H2D/backend/D2H and lowering-region profile events exist
- roofline and backend call rows join to lowering profile events

## V0 Pass Pipeline

The current V0 pipeline should be treated as the first executable slice:

```text
stablehlo_parse
mlir_verify
collective_graph_payload_import
shardy_metadata_propagation_report
pjrtx_graph_import
pjrtx_graph_verify
target_select
target_feature_legality
broadcast_simplify
reshape_transpose_fold
fusion_candidate_discovery
matmul_epilogue_fusion_select
elementwise_fusion_select
tile_shape_select
collective_group_channel_verify
collective_plan_v0
collective_algorithm_select
cost_ledger_build
lowering_region_form
memory_traffic_refine
kernel_codegen_plan
tile_legality_verify
schedule_build
schedule_overlap_plan
schedule_verify
backend_binding_select
backend_binding_verify
executable_view_create
report_emit
```

Some entries are currently conservative report artifacts rather than full MLIR
rewrites. That is acceptable only because the records already exist and tests
pin the behavior. The implementation should grow pass depth behind those stable
contracts.

V0 currently proposes the H2D/backend/D2H schedule rows early enough to provide
command IDs to codegen records, then commits and verifies the MLIR
`pjrtx.schedule.commands` and `pjrtx.schedule.overlaps` state after
`codegen_planned`. That two-step shape is temporary; the invariant is that
backend binding consumes verified schedule facts, not raw StableHLO operations
or backend-private command inference.

V0 implementation status:

- `//next/pjrtx/compiler` exposes a first explicit compiler pass catalog through
  `v0CompilerPassContracts`
- the catalog records stable names, pipeline stages, artifact boundaries,
  target facts, preserved invariants, emitted records, and pass effects
- the MLIR pass report now uses the same stable names for `stablehlo_parse`,
  `mlir_verify`, `collective_graph_payload_import`,
  `mlir_canonicalize_cse`, and `shardy_metadata_propagation_report`
- `broadcast_simplify` is the first operational graph normalization pass; it
  removes identity broadcasts before fusion and emits `GraphRewriteRecord`
  entries for applied or rejected broadcast rewrites
- `reshape_transpose_fold` imports typed StableHLO reshape/transpose operations,
  folds only identity shape/layout mappings, and keeps non-identity transforms
  on the no-fallback target-legality failure path
- `matmul_epilogue_fusion_select` records `dot+broadcast/add/tanh` as an
  explicit rejected epilogue candidate in V0, so matmul-boundary behavior is a
  compiler decision instead of hidden backend policy
- `collective_graph_payload_import` imports `stablehlo.all_reduce` as a typed
  collective graph payload with add reduction and replica-group metadata; V0
  still rejects executable collective lowering at `collective_algorithm_select`
  with direct/ring/tree/split named as rejected algorithms
- `collective_group_channel_verify` validates replica-group participants
  against `replicas * partitions`, parses StableHLO channel handles into
  channel ID/type fields, rejects duplicate or out-of-topology participants,
  and keeps tokenized collectives off the executable path in V0
- `logical_cost_estimate` lives in `mlir_state` for V0: compiler target
  legality contributes only per-instruction backend capability facts, and MLIR
  state derives the committed cost-ledger rows before lowering-region formation
- `memory_traffic_refine` lives in `mlir_state` for V0: it derives
  lowering-linked global-memory and local-memory traffic from extracted cost,
  lowering, placement, graph, and target-memory facts before performance facts
  are committed
- `schedule_overlap_plan` records transfer/compute and compute/transfer
  dependency edges as explicit serialized overlap decisions, preserving the
  command IDs, stream IDs, dependency kind, and reason that future async DMA,
  collective engines, and event planning must consume
- `lowering_region_form` derives backend-kernel and elementwise-fusion region
  decisions into MLIR as `pjrtx.lowering.records`, derives
  `pjrtx.lowering.region_facts` from MLIR fusion and placement facts plus
  cost-ledger links for tile, memory, and codegen-region intent, advances state
  to `lowering_planned`, and extracts the verified `LoweringRecord` and
  `LoweringRegionFact` views before memory traffic, performance, schedule,
  backend binding, or codegen consume those regions
- `kernel_codegen_plan` records one backend-neutral codegen row per executable
  lowering region inside `mlir_state`, using backend capability facts plus
  graph, lowering, placement, and memory-traffic facts to link lowering,
  command, graph instructions, cost ledger, memory traffic, selected operation
  name, backend kind, expected unit, and a tiny kernel IR
  shape/value-flow/tile/memory-pressure summary before backend binding or
  Metal/MLS graph expansion
- `tile_legality_verify` checks codegen result memory and tile memory decisions
  against target memory-space capacities, failing before backend binding when a
  generated region cannot fit the selected target
- `schedule_build` and `schedule_overlap_plan` now derive command and overlap
  rows in `mlir_state`; the compiler still supplies graph boundary values, but
  schedule row shape, dependency edges, and overlap reasons are no longer local
  trace-builder facts
- the catalog is intentionally a typed Zig data structure, not a callback-heavy
  framework; straight-line orchestration remains acceptable while contracts are
  still settling

## Next Passes To Add

The next pass work should focus on compiler-middle depth before deeper runtime
or backend submission:

1. `matmul_epilogue_fusion_accept`
   Turn the existing rejected matmul epilogue candidate into a target-specific
   accepted lowering only after kernel IR, math policy, tile pressure, and
   backend epilogue support are explicit.

2. `transfer_memory_traffic_record`
   Add host/device DMA memory traffic records once transfer commands can carry
   their own provenance without pretending they are compute lowerings.

3. `schedule_overlap_select`
   Upgrade serialized overlap records into selected or rejected async overlap
   candidates after transfer memory traffic, buffer lifetime, stream capacity,
   and target event semantics are explicit.

4. `kernel_ir_tile_contract_v0`
   Expand the current tile/memory-pressure summary into per-buffer tile
   lifetimes and scratch/workspace requirements.

5. `matmul_epilogue_codegen_contract`
   Define the exact generated-kernel contract for accepted epilogues, including
   math mode, accumulator dtype, supported activations, and backend operation.

## Pass Manager Shape

Do not overbuild a separate framework before the pass contracts are stable. V0
can use straight-line orchestration, but the code should evolve toward MLIR pass
manager execution with Zig-owned external passes where practical:

```zig
PassContext
  allocator
  diagnostics: *std.Io.Writer
  compile_options
  target
  source_table
  analysis_cache

PassResult
  status
  updated_mlir_state
  extracted_records
  invalidated_analyses
```

Implementation guidance:

- keep code under `//next/pjrtx/compiler/...`
- keep extracted compiler record types under `//next/pjrtx/compiler/facts`
- keep target model types under `//next/pjrtx/target`
- keep report rendering under `//next/pjrtx/report`
- use Bazel targets for MLIR, StableHLO, and Shardy C APIs
- add minimal `//next/pjrtx/compiler/mlir/...` C-compatible shims only when the
  public MLIR C API cannot express a required dialect or pass boundary
- add `//next/pjrtx/dialects/...` TableGen/MLIR targets for real PjRTx dialects
- use Zig `std` heavily before adding helper functions
- use `std.Io.Reader` and `std.Io.Writer` for input, diagnostics, and reports
- keep pass configuration explicit and testable
- prefer simple typed structs and MLIR pass manager integration over a
  callback-heavy project-local framework
- prefer Zig external pass callbacks for PjRTx policy passes when MLIR C API
  access is enough
- grow discovery before decision: candidate facts may be MLIR-owned before the
  full decision planner moves out of V0 Zig records
- reject decision attachment when temporary Zig planner groups disagree with
  MLIR-owned structured candidate facts
- document intent and invariants in code comments where the pass is subtle
- refactor when a pass starts hiding ownership, correctness, or performance
  logic

## Test Requirements

Every pass family needs focused tests:

- success golden for the V0 fixture
- unsupported-op or unsupported-shape failure
- source/provenance preservation check
- Shardy metadata preservation when relevant
- correctness guard for any algebraic rewrite
- MLIR state and extracted performance record checks when the pass changes
  fusion, layout, memory, collective, schedule, or backend binding
- profile join check when measured data exists

The fast loop should stay:

```text
bazel test //next/pjrtx/compiler:unit_tests
bazel test //next/pjrtx/vertical_slice:lowering_tests
bazel test //next/pjrtx/vertical_slice:execution_test
bazel test //next/pjrtx/...
```

## What Not To Do

- Do not call XLA as an opaque optimizer and only implement runtime submission.
- Do not let backend code make hidden fusion, tiling, or collective decisions.
- Do not add a pass whose only observable output is changed performance.
- Do not let pass ordering depend on incidental mutation.
- Do not hide failed legality behind runtime fallback.
- Do not optimize away correctness assumptions.
- Do not duplicate Zig `std` helpers when a standard primitive is enough.

## Acceptance For This Spec

The pass pipeline is healthy when a reviewer can ask:

```text
Why did this StableHLO operation become this generated or selected kernel?
Which passes changed it?
Which correctness assumptions were used?
Which hardware facts were used?
Which memory spaces carry its bytes?
Which profile event measured it?
What alternatives were rejected?
```

and answer those questions from PjRTx records without inspecting backend-private
state or replaying a hidden compiler.
