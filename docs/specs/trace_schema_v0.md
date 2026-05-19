# Trace Schema V0

This spec defines the minimum records needed to preserve vertical traceability
in the new PjRTx architecture under `//pjrtx/...`.

Trace records are not the long-term compiler IR. They are extracted views for
reports, runtime handoff, diagnostics, and golden tests. The intended compiler
truth is the MLIR state machine described in
`mlir_state_machine_compiler_v0.md`; when both exist, verified MLIR state owns
lowering, fusion, tiling, memory planning, collectives, codegen, scheduling,
and correctness/performance facts.

The first extraction path is specified in `pjrtx_mlir_dialect_v0.md`: MLIR
fusion and pressure facts should produce the current `FusionGroup` report view.

The purpose of V0 is simple:

```text
PJRT / StableHLO source
  -> PjRTx MLIR state transitions
  -> graph values and instructions extracted from compiler state
  -> MLIR pass, fusion, placement, and collective records/views
  -> cost ledger entries
  -> lowering records
  -> memory traffic records
  -> schedule commands
  -> kernel codegen records
  -> backend bindings
  -> profile events
  -> explain records
```

Every record should have a stable ID. A report should be able to join records
without pointer addresses, backend private state, hidden compiler pass state,
or hidden side channels.

V0 Zig records are still valuable because they define the public evidence PjRTx
owes users. They should be treated as schema prototypes until extraction from
PjRTx MLIR dialect state is implemented.

## Requirements

Trace records must be:

- stable enough for golden report tests
- explicit about ownership
- printable through `std.Io.Writer`
- constructible from compiler/runtime code without importing plugin internals
- independent of current `//src/...` implementation details
- safe to emit even when compilation fails partway through

Trace records do not own runtime buffers or backend handles. They may own
strings and small metadata slices.

## ID Types

Use typed IDs instead of raw `u32` where practical.

```zig
pub const SourceId = struct { index: u32 };
pub const GraphValueId = struct { index: u32 };
pub const GraphInstructionId = struct { index: u32 };
pub const CostLedgerId = struct { index: u32 };
pub const LoweringRecordId = struct { index: u32 };
pub const MemoryTrafficId = struct { index: u32 };
pub const ScheduleOverlapId = struct { index: u32 };
pub const ScheduleCommandId = struct { index: u32 };
pub const KernelCodegenId = struct { index: u32 };
pub const BackendBindingId = struct { index: u32 };
pub const ProfileEventId = struct { index: u32 };
pub const ExplainRecordId = struct { index: u32 };
```

IDs are local to one compiled executable report unless explicitly marked as
global. They should be assigned monotonically in deterministic order.

## SourceRef

`SourceRef` ties internal records back to frontend-visible program structure.

```zig
pub const SourceFrontend = enum {
    pjrt,
    stablehlo,
    shardy,
    internal,
};

pub const SourceRef = struct {
    id: SourceId,
    frontend: SourceFrontend,
    op_name: []const u8,
    source_index: u32,
    location: []const u8,
};
```

Rules:

- `op_name` should be stable, such as `stablehlo.dot_general`
- `source_index` is the deterministic operation walk index
- `location` may be empty when MLIR location is unavailable
- internal records created by compiler passes should use `frontend = internal`

## TensorType

```zig
pub const TensorType = struct {
    element_type: target_pkg.BufferType,
    dims: []const i64,
    layout: compiler_facts.LayoutKind,
};
```

Dynamic shape support is out of scope for V0. Unknown dimensions should fail in
V0 unless a fixture explicitly tests the diagnostic.

## GraphValue

```zig
pub const GraphValueRole = enum {
    parameter,
    constant,
    instruction_result,
    output,
};

pub const GraphValue = struct {
    id: GraphValueId,
    ty: TensorType,
    role: GraphValueRole,
    source: ?SourceRef,
};
```

Rules:

- every instruction input/output references a valid `GraphValueId`
- parameters appear before instruction results in deterministic reports
- outputs point to existing graph values rather than duplicating tensors

## GraphInstruction

```zig
pub const GraphInstructionKind = enum {
    dot_general,
    elementwise_unary,
    elementwise_binary,
    broadcast,
    reshape,
    transpose,
    collective,
    return_,
};

pub const GraphInstruction = struct {
    id: GraphInstructionId,
    kind: GraphInstructionKind,
    inputs: []const GraphValueId,
    outputs: []const GraphValueId,
    payload: GraphPayload,
    source: SourceRef,
};
```

Typed payloads make invalid metadata combinations harder to represent.

```zig
pub const CollectiveOp = enum {
    all_reduce,
};

pub const CollectiveReduction = enum {
    add,
};

pub const CollectiveSpec = struct {
    op: CollectiveOp,
    reduction: CollectiveReduction,
    replica_group_count: u32,
    replica_group_size: u32,
    replica_groups: []const u32,
    channel_id: ?u64,
    channel_type: ?u32,
    uses_token: bool,
};

pub const GraphPayload = union(enum) {
    dot_general: DotGeneralSpec,
    elementwise_unary: ElementwiseUnarySpec,
    elementwise_binary: ElementwiseBinarySpec,
    broadcast: BroadcastSpec,
    reshape: ReshapeSpec,
    transpose: TransposeSpec,
    collective: CollectiveSpec,
    return_: ReturnSpec,
};
```

V0 should reject graph instructions whose `kind` and `payload` disagree.
Collective payloads are compiler-owned records, not backend fallbacks. The
first imported collective payload is `stablehlo.all_reduce` with add reduction,
replica group count, replica group size, explicit participant IDs, optional
channel ID, optional channel type, and token-use state. Unsupported collective
algorithms still fail before schedule build.

## MlirPassRecord

`MlirPassRecord` is the V0 report surface for early MLIR and StableHLO pass
boundaries. The compiler pass catalog in `//pjrtx/compiler` gives the broader
contract for every pass family; this record is the stable trace shape currently
used for MLIR-side passes.

```zig
pub const MlirPassStatus = enum {
    ok,
    skipped,
    failed,
};

pub const MlirPassRecord = struct {
    index: u32,
    pass_name: []const u8,
    status: MlirPassStatus,
    input_fingerprint: u64,
    output_fingerprint: u64,
    preserves_source_provenance: bool,
    preserves_shardy_metadata: bool,
    reason: []const u8,
};
```

Rules:

- `pass_name` must come from the stable compiler pass catalog
- fingerprints must be deterministic in golden tests
- skipped passes must still explain why they were skipped
- failed passes must prevent executable creation
- source provenance and Shardy metadata preservation flags must be explicit
- StableHLO collectives are preserved for typed graph import; unsupported
  collective algorithms fail at `collective_algorithm_select` because there is
  no fallback path

## CollectivePlanRecord

`CollectivePlanRecord` is the V0 report surface for collective detection and
algorithm selection. V0 does not lower collectives yet, but the trace still
records the algorithm decision so unsupported communication cannot disappear
behind runtime behavior.

```zig
pub const CollectivePlanDecision = enum {
    no_collectives,
    selected,
    rejected,
    unsupported,
};

pub const CollectiveAlgorithm = enum {
    none,
    direct,
    ring,
    tree,
    split,
};

pub const CollectivePlanRecord = struct {
    index: u32,
    decision: CollectivePlanDecision,
    algorithm: CollectiveAlgorithm,
    checked_graph_instruction_count: u32,
    lowered_collective_count: u32,
    unsupported_collective_count: u32,
    estimated_bytes: u128,
    estimated_latency_ns: ?u64,
    reason: []const u8,
};
```

Rules:

- `no_collectives` must use `algorithm = .none`
- selected collective algorithms must name the chosen algorithm and record
  estimated bytes before schedule build
- rejected or unsupported collective algorithms must fail executable creation
  unless the graph contains no collective work
- V0 imports `stablehlo.all_reduce` payloads and rejects unsupported algorithms
  at `collective_algorithm_select`; the diagnostic names direct, ring, tree,
  and split instead of pretending a runtime path exists
- `collective_group_channel_verify` must validate replica-group participants
  against `replicas * partitions`, reject duplicate participants, and reject
  incomplete channel handles before algorithm selection
- future collective lowering should replace the V0 unsupported diagnostic with
  executable per-collective records that join group/channel metadata, target
  collective engines, memory traffic, schedule commands, and profile events

## GraphRewriteRecord

`GraphRewriteRecord` records typed graph rewrites after StableHLO import and
before fusion, placement, cost, and lowering. These rows make algebraic
normalization explainable instead of letting graph mutation disappear inside a
planner helper.

```zig
pub const GraphRewriteDecision = enum {
    applied,
    rejected,
};

pub const GraphRewriteRecord = struct {
    index: u32,
    pass_name: []const u8,
    decision: GraphRewriteDecision,
    input_instruction_id: GraphInstructionId,
    output_instruction_id: ?GraphInstructionId,
    replaced_value_id: ?GraphValueId,
    replacement_value_id: ?GraphValueId,
    reason: []const u8,
};
```

Rules:

- `pass_name` must come from the compiler pass catalog
- `input_instruction_id` refers to the pre-rewrite instruction ID
- `output_instruction_id` is null when the rewrite removes the instruction
- replaced and replacement value IDs must refer to graph values in the final
  report when present
- rejected rewrites are useful when they explain why a tempting optimization was
  not mathematically valid
- V0 applies `broadcast_simplify` only for identity broadcasts with identical
  input/output tensor type and dimension map
- V0 applies `reshape_transpose_fold` only for identity reshapes and identity
  transposes; non-identity shape/layout transforms must fail target legality
  until explicit lowering and allocation support exists

## FusionGroup

`FusionGroup` records accepted and rejected fusion decisions before placement,
lowering, codegen, and backend binding consume the region boundaries.

```zig
pub const FusionDecision = enum {
    accepted,
    rejected,
};

pub const FusionPressureDelta = struct {
    split_kernel_count: u32,
    fused_kernel_count: u32,
    split_peak_live_bytes: u128,
    fused_live_bytes: u128,
    additional_live_bytes: u128,
    global_bytes_saved: u128,
};

pub const FusionGroup = struct {
    index: u32,
    decision: FusionDecision,
    kind: []const u8,
    graph_instruction_ids: []const GraphInstructionId,
    bytes_saved: u128,
    launch_count_reduction: u32,
    pressure_delta: FusionPressureDelta,
    reason: []const u8,
};
```

Rules:

- rejected fusion is still a first-class compiler decision when the boundary
  affects performance, math policy, memory pressure, or backend codegen
- `pressure_delta` compares the current split lowering with a proposed fused
  lowering; V0 uses it for the rejected matmul epilogue candidate before
  enabling acceptance
- a positive `additional_live_bytes` is not automatically illegal, but it must
  be checked by tile legality and future buffer-lifetime planning before
  changing the fusion decision

## PlacementRecord

`PlacementRecord` is the V0 report surface for layout, tile shape, and memory
space decisions before cost, lowering, schedule, and backend binding consume
them.

```zig
pub const PlacementRecord = struct {
    index: u32,
    graph_instruction_id: GraphInstructionId,
    output_value_ids: []const GraphValueId,
    layout: LayoutKind,
    logical_tile_shape: []const i64,
    result_memory_space_id: u32,
    tile_memory_space_id: ?u32,
    reason: []const u8,
};
```

Rules:

- every executable graph instruction must have a placement record before
  lowering
- `logical_tile_shape` is the compiler-visible tile shape, not a backend-private
  emitter detail
- Metal V0 may keep whole-tensor tiles because unified memory is the current
  public target model
- NPU V0 uses bounded local-SRAM tile shapes for matrix and fusible elementwise
  work so large tensors do not silently become whole-tensor local-memory tiles
- `result_memory_space_id` and `tile_memory_space_id` must refer to the selected
  target description when a target is present
- later codegen must consume placement records instead of rediscovering layout
  or tile shape from StableHLO strings

## CostLedgerEntry

The cost ledger tracks logical work and bytes by source instruction.

```zig
pub const CostOpClass = enum {
    matmul,
    elementwise,
    transcendental,
    transfer,
    backend_kernel,
};

pub const CostLedgerEntry = struct {
    id: CostLedgerId,
    source: ?SourceRef,
    graph_instruction_ids: []const GraphInstructionId,
    op_class: CostOpClass,
    dtype: target_pkg.BufferType,
    accumulation_dtype: ?target_pkg.BufferType,
    logical_ops: u128,
    bytes_read: u128,
    bytes_written: u128,
    expected_unit_id: ?u32,
    formula: []const u8,
    approximation: []const u8,
};
```

Rules:

- `formula` is a stable human-readable formula, not generated prose
- `approximation` is empty when exact enough for V0
- transfer entries use `logical_ops = 0`
- `tanh` and similar operations should be `transcendental`, not generic
  elementwise

## LoweringRecord

```zig
pub const LoweringDecision = enum {
    backend_kernel_graph,
    elementwise_fusion,
    transfer,
    unsupported,
};

pub const LoweringRecord = struct {
    id: LoweringRecordId,
    graph_instruction_ids: []const GraphInstructionId,
    decision: LoweringDecision,
    reason: []const u8,
    rejected_alternatives: []const []const u8,
    cost_ledger_ids: []const CostLedgerId,
};
```

Rules:

- every scheduled backend execute command should link to at least one lowering
  record
- unsupported lowering records should be printable even when compilation fails
- `reason` should be short and stable enough for reports

## LoweringRegionFact

`LoweringRegionFact` is the MLIR-extracted compiler-middle explanation for why
a lowering region has the fusion, tile, memory, and codegen shape that it does.
It is not a separate executable unit; it decorates `LoweringRecord`.

```zig
pub const LoweringRegionFact = struct {
    lowering_record_id: LoweringRecordId,
    fusion_group_index: ?u32,
    placement_record_indices: []const u32,
    logical_tile_shape: []const i64,
    result_memory_space_id: u32,
    tile_memory_space_id: ?u32,
    codegen_region: LoweringDecision,
    reason: []const u8,
};
```

Rules:

- every fact must reference exactly one lowering record
- placement references must cover the graph instructions in that lowering
- `codegen_region` must match the lowering decision until deeper MLIR codegen
  dialect facts replace the temporary V0 bridge
- `fusion_group_index` may be absent when a test or future verifier is carrying
  an explicit unsupported/no-fallback lowering path
- later memory traffic, schedule, backend binding, kernel graph, and profile
  facts should use the verified lowering ID, while reports can use the region
  fact to explain fusion, tile, memory-space, and codegen intent

## MemoryTrafficRecord

```zig
pub const MemoryTrafficKind = enum {
    global_memory,
    local_memory,
    host_device_dma,
    interconnect,
};

pub const MemoryTrafficRecord = struct {
    id: MemoryTrafficId,
    lowering_record_id: LoweringRecordId,
    memory_space_id: u32,
    kind: MemoryTrafficKind,
    graph_instruction_ids: []const GraphInstructionId,
    cost_ledger_ids: []const CostLedgerId,
    bytes_read: u128,
    bytes_written: u128,
    reason: []const u8,
};
```

Rules:

- memory traffic records are compiler facts, not backend-local debug output
- every record must join to a lowering record, graph instructions, cost entries,
  and a target memory space when a target is present
- `global_memory` is traffic visible in device/global memory such as HBM,
  remote device memory, or unified device memory
- `local_memory` is traffic attributed to local SRAM or scratchpad placement
- `host_device_dma` is reserved for explicit host/device transfer traffic once
  transfer commands can be represented as memory traffic records without
  inventing lowering provenance
- `interconnect` is reserved for explicit collective or remote-device traffic
  once collective lowering creates executable collective records
- backend call summaries and future kernel generation should consume these
  records instead of re-deriving memory hierarchy from placement
- V0 does not yet model cache, register spills, collective traffic, or
  multi-level tile movement; those should extend this schema rather than hiding
  in backend code

## ScheduleOverlapRecord

`ScheduleOverlapRecord` makes overlap planning explainable before the runtime
has asynchronous stream execution. V0 records dependency edges conservatively as
serialized, then future passes can turn the same rows into selected or rejected
overlap plans.

```zig
pub const ScheduleOverlapDecision = enum {
    serialized,
    candidate,
    selected,
    rejected,
};

pub const ScheduleOverlapKind = enum {
    transfer_compute,
    compute_transfer,
    transfer_transfer,
    collective_compute,
};

pub const ScheduleOverlapRecord = struct {
    id: ScheduleOverlapId,
    decision: ScheduleOverlapDecision,
    kind: ScheduleOverlapKind,
    first_command_id: ScheduleCommandId,
    second_command_id: ScheduleCommandId,
    dependency_kind: DependencyKind,
    first_stream: StreamId,
    second_stream: StreamId,
    reason: []const u8,
};
```

Rules:

- command references must exist and point from an earlier command to a later
  command
- `reason` must explain why V0 serialized the edge or why a future pass selected
  or rejected overlap
- overlap records must not create execution permission; schedule commands and
  dependencies remain authoritative
- future collective overlap must join to collective lowering records, target
  collective engines, interconnect memory traffic, and profile events rather
  than becoming backend-private stream behavior

## ScheduleCommand

```zig
pub const StreamId = struct { index: u32 };

pub const CommandKind = enum {
    host_to_device,
    backend_execute,
    device_to_host,
    event_record,
    event_wait,
};

pub const DependencyKind = enum {
    data,
    stream_order,
    memory_availability,
};

pub const CommandDependency = struct {
    command_id: ScheduleCommandId,
    kind: DependencyKind,
};

pub const ScheduleCommand = struct {
    id: ScheduleCommandId,
    kind: CommandKind,
    stream: StreamId,
    inputs: []const GraphValueId,
    outputs: []const GraphValueId,
    dependencies: []const CommandDependency,
    lowering_record_ids: []const LoweringRecordId,
    cost_ledger_ids: []const CostLedgerId,
};
```

V0 may execute synchronously, but the schedule must still describe command
dependencies.

## KernelCodegenRecord

`KernelCodegenRecord` is the compiler-middle bridge between lowering regions
and backend binding. It names the generated or selected kernel candidate before
backend executable planning can hide it in Metal/MLS, NPU, or future collective
engine code.

```zig
pub const KernelCodegenKind = enum {
    backend_kernel_graph,
    elementwise_fusion_kernel,
    library_call,
    collective_engine,
};

pub const KernelCodegenShape = struct {
    operation_count: u32,
    external_input_count: u32,
    external_output_count: u32,
    intermediate_value_count: u32,
};

pub const KernelMemoryPressure = struct {
    global_bytes_read: u128,
    global_bytes_written: u128,
    local_bytes_read: u128,
    local_bytes_written: u128,
};

pub const KernelCodegenRecord = struct {
    id: KernelCodegenId,
    lowering_record_id: LoweringRecordId,
    command_id: ScheduleCommandId,
    backend_kind: BackendKind,
    kind: KernelCodegenKind,
    operation: []const u8,
    shape: KernelCodegenShape,
    logical_tile_shape: []const i64,
    result_memory_space_id: u32,
    tile_memory_space_id: ?u32,
    memory_pressure: KernelMemoryPressure,
    external_input_ids: []const GraphValueId,
    external_output_ids: []const GraphValueId,
    intermediate_value_ids: []const GraphValueId,
    graph_instruction_ids: []const GraphInstructionId,
    cost_ledger_ids: []const CostLedgerId,
    memory_traffic_ids: []const MemoryTrafficId,
    expected_unit_id: ?u32,
    reason: []const u8,
};
```

Rules:

- every executable lowering region should have one codegen record before
  backend binding
- codegen records link to lowering, schedule, graph instructions, cost ledger,
  and memory traffic records
- `operation` is a stable compiler-visible name such as `npu_matmul`,
  `npu_elementwise_fusion`, or `metal_mls_elementwise_fusion_kernel`
- `shape` is the first backend-neutral kernel IR surface: it records operation
  count, external inputs, external outputs, and internal intermediate values
  for the lowering region
- `logical_tile_shape`, `result_memory_space_id`, `tile_memory_space_id`, and
  `memory_pressure` connect codegen directly to placement and memory-traffic
  facts before backend binding
- external input, external output, and intermediate value lists are stable value
  IDs, so allocation and fusion acceptance can reason from compiler records
  rather than backend-private graph construction
- `expected_unit_id` is `null` only when the selected region spans mixed units
  or the target description does not know the unit
- backend executable planning may expand these records into backend-specific
  calls or graph nodes, but it must not rediscover source semantics from strings

## BackendBinding

```zig
pub const BackendBinding = struct {
    id: BackendBindingId,
    command_id: ScheduleCommandId,
    backend_kind: core.BackendKind,
    backend_operation: []const u8,
    graph_instruction_ids: []const GraphInstructionId,
    expected_unit_id: ?u32,
    cost_ledger_ids: []const CostLedgerId,
};
```

For Metal V0, `backend_operation` can be `metal_mls_graph_execute`.

## ProfileEvent

```zig
pub const ProfileEventKind = enum {
    compile_pass,
    h2d,
    backend_execute,
    d2h,
};

pub const ProfileStatus = enum {
    ok,
    failed,
};

pub const ProfileEvent = struct {
    id: ProfileEventId,
    command_id: ?ScheduleCommandId,
    graph_instruction_ids: []const GraphInstructionId,
    kind: ProfileEventKind,
    start_ns: u64,
    duration_ns: u64,
    bytes: u128,
    logical_ops: u128,
    status: ProfileStatus,
    forced_synchronization: bool,
};
```

Golden tests should redact `start_ns` and `duration_ns`, but preserve event
presence, command IDs, graph instruction IDs, bytes, logical ops, status, and
forced synchronization flags. Backend lowering-region events use the same
`ProfileEvent` record and identify their region through `command_id` plus the
exact `graph_instruction_ids` in that lowering.

## ExplainRecord

```zig
pub const ExplainSubject = union(enum) {
    graph_instruction: GraphInstructionId,
    lowering_record: LoweringRecordId,
    schedule_command: ScheduleCommandId,
    backend_binding: BackendBindingId,
};

pub const ExplainRecord = struct {
    id: ExplainRecordId,
    pass_name: []const u8,
    subject: ExplainSubject,
    decision: []const u8,
    reason: []const u8,
    source_refs: []const SourceRef,
    cost_ledger_ids: []const CostLedgerId,
    profile_event_ids: []const ProfileEventId,
};
```

Explain records should be concise. Long explanations belong in human docs or
debug dumps, not every report. Profiled reports should attach `profile_event_ids`
to explain records when the profile event can be matched by command ID,
lowering record, backend binding, or graph instruction provenance.

## Report Ordering

Stable report order:

```text
sources
target summary
graph values
graph instructions
mlir pass records
graph rewrite records
fusion groups
placement records
collective records
cost ledger
lowering records
memory traffic records
schedule overlap records
schedule commands
kernel codegen records
backend bindings
profile events
explain records
correctness summary
```

Within each section, sort by typed ID.

## V0 Validation

The report validator should check:

- every referenced ID exists
- every graph rewrite record has a stable pass name and valid replacement values
- every scheduled backend command has lowering provenance
- every memory traffic record references a lowering, memory space, graph
  instructions, and cost entries
- every schedule overlap record references valid ordered commands and has a
  non-empty reason
- every kernel codegen record references valid lowering, command, graph
  instruction, cost, and memory traffic IDs
- every backend binding references an existing schedule command
- every profile event with a command references an existing command
- every cost entry referenced by explain/lowering/schedule exists
- no timing field is required for golden report identity

## V0 Decisions

IDs are local to one compiled executable report. Module-global IDs can be added
later as a separate ID namespace, but V0 report joins must not depend on
process-global state.

`SourceRef` strings are owned by the report/source table lifetime, not by every
individual record. Records store slices into the owned source table. This keeps
reports explicit without copying operation names and locations into every
provenance edge.

Profile event facts live under `//pjrtx/runtime/facts`. Compiler facts,
backend facts, runtime facts, and target facts are rendered together by
`//pjrtx/report`, so reports can join compile-time and runtime observations
without turning any one package into a central schema bucket.

`formula` stays a stable string in V0. The string is not prose; it is a compact
canonical expression such as `2*M*N*K` or `numel*sizeof(dtype)`. A structured
formula enum can be added after V0 if report consumers need machine algebra.
