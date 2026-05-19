# MLIR State Machine Compiler V0

This spec defines the intended compiler architecture for PjRTx after the first
Zig scaffolding slices. The central correction is:

```text
MLIR is the compiler truth.
Zig records are extracted views, temporary scaffolding, and runtime/API support.
```

The current bootstrap Zig records are useful because they forced the vocabulary
to become explicit: fusion decisions, tile pressure, codegen regions, memory
traffic, collectives, schedule commands, and profile joins. They should not
become the permanent lowering representation, and they should not remain in a
central package. Long term, those facts must live in MLIR dialects, attributes,
analyses, and verifier state transitions, with extracted views owned by the
package that consumes their invariants.

Use `pjrtx_mlir_dialect_v0.md` for the first concrete dialect/API slice:
module state attributes, target attachment, fusion facts, pressure deltas,
Zig `MlirSession`, verifier boundaries, and extraction into the existing
`FusionGroup` report view.

## Goal

Given StableHLO/Shardy MLIR, target hardware specifications, and compile
options, PjRTx should produce an executable plan through an MLIR state machine:

```text
StableHLO/Shardy module
  -> PjRTx MLIR states
  -> verified lowering/codegen/schedule MLIR
  -> extracted reports and executable view
  -> runtime execution
```

Every important correctness, performance, pressure, allocation, collective, and
profiling decision should be visible and verifiable in MLIR before it is
extracted to Zig reports or consumed by a backend.

## Non-Goal

Do not rebuild an MLIR/HLO compiler in Zig structs.

Zig is excellent for orchestration, API boundaries, deterministic extraction,
runtime ownership, `std.Io.Reader`/`std.Io.Writer` plumbing, and validation of
public artifacts. It should not become a parallel IR that replaces MLIR.

This does not mean that PjRTx compiler passes cannot be written in Zig. Zig is
the preferred language for PjRTx compiler policy when the pass can operate
through MLIR C API handles, external pass callbacks, and narrow C-compatible
MLIR helpers. The non-goal is a large Zig-only lowering representation that
drifts away from MLIR verifiers and state snapshots.

Full dialect definitions remain MLIR-native. Operation/type/attribute
definitions, parser/printer syntax, dialect registration, and advanced rewrite
or dialect-conversion infrastructure should use TableGen and minimal C++ glue,
then expose narrow C-compatible entry points to Zig.

## Architecture Rule

Use MLIR for compiler state, transformations, and proof obligations.

Use Zig for:

- PJRT/API boundaries
- target discovery and runtime integration
- pass pipeline orchestration
- Zig-owned external MLIR passes where the MLIR C API is sufficient
- runfiles, harnesses, and deterministic test drivers
- extracting stable summaries from MLIR
- validating public reports and executable handoff structures
- `std.Io.Reader` and `std.Io.Writer` based input/output paths

The extracted Zig report is a view. It must not be the only place where a
compiler decision exists.

Use minimal MLIR-native C++/TableGen for:

- real PjRTx dialect registration
- generated op/type/attribute definitions
- parser, printer, and verifier integration
- canonicalization and dialect-conversion helpers when MLIR exposes no
  practical C API path
- thin C-compatible functions that create/register dialects and passes for Zig

The C++ layer must not become the hidden compiler. It exists to expose MLIR's
native extension points while Zig remains the owner of PjRTx target policy,
explainability, diagnostics, and V0 implementation discipline.

## Critique Of Current Zig Records

The current Zig records are a good forcing function, but they are the wrong
long-term compiler representation.

They are good because they make invisible compiler questions explicit:

- which source operation produced this region
- which pass made or rejected a decision
- which memory space and tile shape were selected
- which collective algorithm was selected or rejected
- which bytes, ops, launch counts, and profile events belong together

They are wrong as the permanent lowering architecture because:

- they sit beside MLIR instead of being part of the transformed program
- they make rewrite legality harder to verify with MLIR dialect verifiers
- they duplicate facts that should be attached to ops, regions, types, and
  attributes
- they encourage hand-built compiler state machines in Zig
- they make boundary tests report-centric instead of IR-centric
- they make it too easy for MLIR lowering and Zig reports to drift apart

The intended direction is not "more Zig records." The intended direction is a
PjRTx MLIR dialect and state machine that carries the same facts inside the
compiler IR, then extracts compact Zig records for reports, runtime handoff,
golden tests, and user-facing diagnostics.

## State Machine

Compilation should be modeled as explicit MLIR states. A state can be a module
attribute, a top-level `pjrtx.module` wrapper, or a combination of dialect ops
and verifier-enforced attributes. The exact spelling can evolve, but the state
transition must be inspectable.

```text
Imported
  -> StableHLOVerified
  -> Canonicalized
  -> TargetAttached
  -> TargetLegal
  -> FusionPlanned
  -> Tiled
  -> MemoryPlanned
  -> CollectivesPlanned
  -> CollectivesLowered
  -> LoweringPlanned
  -> PerformanceModeled
  -> CodegenPlanned
  -> Bufferized
  -> TileLegal
  -> Scheduled
  -> ExecutableReady
```

Rules:

- passes declare required input state and produced output state
- verifiers reject impossible state transitions
- no pass may silently weaken correctness assumptions
- no pass may drop source locations, Shardy metadata, tokens, channels,
  side-effects, or memory-space facts without recording why
- failed verification prevents executable creation

## MLIR-Carried Facts

These facts should live in MLIR, not only in Zig structs.

### Correctness

Represent in MLIR:

- StableHLO semantic preservation
- dtype and accumulation dtype
- math policy, including strict or relaxed modes
- NaN, infinity, signed-zero, reduction-order, and transcendental assumptions
- token and side-effect ordering
- aliasing and bufferization constraints
- source locations and frontend operation names

Correctness facts should be verified at every state boundary that can affect
semantics.

### Performance

Represent in MLIR:

- logical ops by op class and dtype
- selected execution unit
- dtype-specific rates from the target model
- predicted compute time
- launch count and launch-count deltas
- generated or selected kernel identity
- profile event join keys

Performance is not a backend log. It is part of the compiler state.

### Pressure

Represent in MLIR:

- memory spaces
- logical tile shape
- value-flow boundaries
- external inputs and outputs
- intermediate values
- global/HBM traffic
- local SRAM or scratchpad pressure
- DMA/interconnect traffic
- buffer live ranges
- scratch/workspace requirements
- split-versus-fused pressure deltas

Pressure facts must be capacity checked before backend binding.

### Collectives

Represent in MLIR:

- collective op kind
- replica and partition groups
- explicit participant IDs
- channel IDs and channel types
- token use
- selected or rejected algorithms
- collective engine requirements
- interconnect traffic
- schedule/overlap constraints

Collective lowering is not a runtime fallback and not a backend-private
operation.

## Proposed PjRTx Dialect Surface

The canonical dialect/op/pass destination is
`final_mlir_dialect_op_pass_architecture_v0.md`. This state-machine spec owns
the rule, not the exhaustive op list:

- PjRTx dialect ops and attributes should have verifiers.
- If a fact affects correctness, performance, allocation, codegen, collectives,
  or scheduling, it should be represented as MLIR IR or as MLIR
  attribute/analysis state with a verifier and extraction path.
- The current attribute bridge is temporary and must shrink as real dialect
  attrs/ops land.

## Pass Contracts

Each pass should have:

```text
stable_name
input_state
output_state
required_target_facts
preserved_invariants
invalidated_analyses
emitted_or_updated_mlir_ops
failure_diagnostics
extraction_impact
```

Example:

```text
stable_name: tile_legality_verify
input_state: CodegenPlanned
output_state: TileLegal
required_target_facts: memory spaces, capacities, tile pressure
preserved_invariants: StableHLO semantics, source locations, value flow
emitted_or_updated_mlir_ops: none, verifier-only
failure_diagnostics: exact codegen region, memory space, required bytes, capacity
extraction_impact: tile legality appears in final report
```

## Boundary Tests

Every state boundary should have MLIR tests. Prefer `mlir-opt` style tests where
possible, with focused fixtures:

```text
input.stablehlo.mlir
  -> after-canonicalize.mlir
  -> after-target-legal.mlir
  -> after-fusion-planned.mlir
  -> after-tiled.mlir
  -> after-memory-planned.mlir
  -> after-collectives-lowered.mlir
  -> after-codegen-planned.mlir
  -> after-tile-legal.mlir
  -> after-scheduled.mlir
  -> after-backend-bound.mlir
```

Test categories:

- positive state transition tests
- verifier failure tests
- golden MLIR snapshots
- extracted report tests
- pressure and capacity tests
- math correctness policy tests
- collective group/channel/token tests
- profile join tests

The harness should preserve fast iteration:

- small MLIR fixtures
- deterministic pass output
- narrow failure diagnostics
- runfiles, not embedded test data
- optional debug dumps for each state boundary

## Extraction

Reports should be extracted from MLIR, not independently invented in Zig.

Extraction rules:

- extraction is deterministic
- extracted IDs are stable for tests
- extracted records preserve MLIR locations and source names
- extraction failure is a compiler bug unless the MLIR state verifier already
  rejected the module
- extracted Zig records may be smaller than MLIR, but never stronger than MLIR

The final report remains valuable because users and runtime code should not need
to parse arbitrary MLIR to answer common questions. But the report is a view of
verified MLIR state.

## Migration Plan

The current Zig records should be treated as schema prototypes and tests for
what MLIR must eventually carry.

1. Keep current Zig records while implementing the first MLIR state shim.
2. Prove Zig external MLIR passes through the MLIR C API pass manager.
3. Add a minimal PjRTx MLIR dialect with module state and target attributes
   using TableGen plus a narrow C-compatible shim.
4. Move fusion candidates and pressure deltas into MLIR.
5. Move tile plans, memory pressure, and tile legality into MLIR. The first
   V0 slice commits placement records into MLIR and extracts the report view
   from `placement_planned`; deeper work should move tile selection and pressure
   calculation themselves into MLIR passes.
6. Move collective plan/group/channel state into MLIR. The first V0 slice
   commits the collective decision record into MLIR and extracts the report view
   from `collectives_planned`; deeper work should move group/channel facts,
   token checks, algorithm selection, and lowering commands themselves into
   MLIR passes.
7. Move lowering regions into MLIR. The current V0 slice derives
   `pjrtx.lowering.records` and `pjrtx.lowering.region_facts` inside
   `mlir_state` from MLIR fusion candidates, placement records, and cost-ledger
   instruction links. Before this derivation, compiler target legality supplies
   compact backend capability facts and `mlir_state` derives the cost-ledger
   rows from graph op/type facts, logical ops, bytes, dtype/op-class, formulas,
   and expected execution-unit facts. `commitLoweringPlan` stamps that derived
   ledger into MLIR, and extraction reads the MLIR cost view before memory
   traffic is formed. A Zig external pass verifies graph-instruction,
   cost-ledger, fusion, placement, tile, memory-space, and codegen-region
   references, advances state to `lowering_planned`, and extracts typed
   `LoweringRecord` and `LoweringRegionFact` report views. Deeper work should
   replace the attribute bridge with real MLIR region formation from fusion,
   placement, Shardy partitioning, collective legality, and backend codegen
   constraints.
8. Move cost ledger and memory traffic state into MLIR. The current V0 slice
   derives and commits `pjrtx.performance.cost_ledger` plus
   `pjrtx.performance.memory_traffic`. Memory traffic derivation now happens
   inside `mlir_state` from target memory spaces, graph value flow, placement
   records, lowering records, and cost-ledger bytes before the performance
   attrs are committed. A Zig external pass verifies them, advances state from
   `lowering_planned` to `performance_modeled`, and extracts typed
   `CostLedgerEntry` and `MemoryTrafficRecord` report views. Runtime,
   lowering, hardware, and backend-call summaries must read predicted bytes,
   logical ops, memory traffic, and roofline inputs from those extracted views.
9. Move codegen regions and kernel candidates into MLIR. The current V0 slice
   derives kernel codegen records inside `mlir_state` from backend capability
   facts, graph instructions, lowering records, placement records, and
   memory-traffic records, commits them into MLIR, and extracts the report view
   from `codegen_planned`; deeper work should make kernel IR/candidates real
   MLIR ops or attrs instead of temporary committed records.
10. Move schedule commands and overlap candidates into MLIR. The current V0
   slice derives schedule commands and overlap records inside `mlir_state` from
   parameter IDs, return value IDs, extracted lowering records, and extracted
   cost ledger entries, commits them into MLIR, and extracts the report views
   from `scheduled`; deeper work should make event, stream, allocator, and
   overlap constraints real MLIR schedule facts.
11. Move backend binding and executable-readiness facts into MLIR. The first V0
   slice commits `BackendBinding` records into MLIR and extracts the report
   view from `backend_bound`, then commits a minimal executable contract and
   advances state to `executable_ready`; deeper work should make generated
   kernel handles, backend graph calls, allocator reservations, and richer
   executable contracts real MLIR facts. The first backend-owned bridge now
   commits backend executable calls to `pjrtx.backend.executable` and advances
   state to `backend_executable_planned` without reversing the compiler/backend
   dependency boundary. The same bridge now commits Metal/MLS kernel graph
   nodes and edges to `pjrtx.backend.kernel_graph` and advances state to
   `backend_kernel_graph_planned`; deeper work should replace compact attribute
   tags with richer generated-kernel and command-buffer facts. The first
   runtime-owned bridge now commits allocator reservations to
   `pjrtx.runtime.allocation` and advances state to
   `runtime_allocation_planned`, then commits stream/event records to
   `pjrtx.runtime.streams` and advances state to `runtime_stream_planned`, then
   commits profile observations to `pjrtx.runtime.profile_events` and advances
   state to `runtime_profiled`, then commits explicit profile joins to
   `pjrtx.runtime.profile_joins` and advances state to
   `runtime_profile_joined`. The backend-owned bridge then commits executable
   call profile joins to `pjrtx.backend.profile_joins` and advances state to
   `backend_profile_joined`, keeping concrete kernel-call explainability in
   MLIR without making the compiler import backend internals.
12. Extract the existing Zig `TraceReport` from MLIR.
13. Delete or shrink Zig-only compiler-middle structs once extraction covers the
   same information.

During migration, every duplicated fact must name its source of truth. The
desired direction is:

```text
MLIR fact -> extracted Zig view
```

not:

```text
Zig fact -> reconstructed MLIR guess
```

## Hard Requirements

- no fallback path
- MLIR verifiers enforce state invariants
- performance facts are compiler state
- pressure facts are compiler state
- mathematical correctness facts are compiler state
- collectives are compiler state
- schedule/overlap decisions are compiler state
- backend binding consumes verified MLIR codegen/schedule state
- Zig reports are extracted views

This is the architecture that avoids both traps: rebuilding XLA badly in Zig,
and hiding all meaningful decisions inside an opaque MLIR pipeline.
