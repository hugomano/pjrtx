# PjRTx Vertical Slice V0

This document turns the architecture vision into the first buildable slice.
The goal is not to implement the whole PjRTx compiler. The goal is to prove
the architecture with one small workload that travels through every important
layer:

```text
PJRT / StableHLO input
  -> MLIR state machine
  -> typed graph IR
  -> target description
  -> cost ledger
  -> simple lowering decision
  -> schedule IR
  -> backend execution
  -> profiling events
  -> explain trace
  -> correctness check
```

This slice should make one thing real:

> A user can trace useful work from PJRT-visible computation to lowered command,
> backend kernel or kernel graph, hardware cost estimate, runtime profile, and
> correctness result.

## Two Versions

V0 belongs to the new architecture. It should live under `//next/pjrtx/...` and use
the docs as its source of truth.

The existing README and `//src/...` tree describe and contain the current
bootstrap implementation. That implementation should stay intact while V0 is
built beside it. V0 may call into or bridge to `//src/...` where useful, but new
architecture modules, tests, and reports should be rooted in `//next/pjrtx/...`.

## Non-Goals

V0 should stay deliberately small.

Out of scope:

- full StableHLO coverage
- collectives
- AOT packaging
- multi-host execution
- full memory-space assignment
- custom kernel generation
- real Verilog or RTL import
- advanced autotuning
- dynamic shapes
- async stream overlap beyond explicit transfer and execute events
- full replacement of the existing runtime/backend path

V0 can use the current Metal backend path where useful. It can also use a
NPU target model to prove traceability when Metal does not expose enough
hardware detail.

Compiler-middle facts should still move toward MLIR ownership inside this
small scope. Zig may implement PjRTx policy passes and extraction, including
external MLIR passes through the C API. Real PjRTx dialect syntax and verifier
integration should use minimal TableGen/C++ plumbing with a narrow C-compatible
boundary back to Zig.

## Workload

The initial workload should be small but representative:

```python
def f(x, w, b):
    return jnp.tanh(jnp.dot(x, w) + b)
```

StableHLO families required:

- `parameter`
- `dot_general`
- `broadcast_in_dim` or implicit bias broadcast
- `add`
- `tanh`
- `return`

Why this workload:

- it has a matrix op with meaningful FLOPs
- it has elementwise ops that invite fusion
- it has shape, dtype, and layout constraints
- it has parameters and outputs
- it can be checked numerically against JAX CPU
- it maps cleanly to a single backend executable today

Optional second workload:

```python
def g(x):
    return jnp.exp(x) * jnp.rsqrt(jnp.maximum(x, 0.001))
```

This is useful for validating pure elementwise fusion and byte traffic
accounting without matmul.

## Success Criteria

V0 succeeds when PjRTx can produce a report like:

```text
program: tanh(dot(x, w) + b)
target: metal_v0
dtype: f32

graph:
  stablehlo.dot_general -> graph.instruction.2
  stablehlo.add -> graph.instruction.3
  stablehlo.tanh -> graph.instruction.4

lowering:
  graph.instructions [2] lowered to backend command cmd.0
  graph.instructions [3,4] lowered to backend command cmd.1

cost estimate:
  dot_general: 2*M*N*K f32 ops
  add/tanh: M*N elementwise ops
  bytes read/write by value and memory space

execution:
  h2d copies: 3
  backend commands: 2
  d2h copies: 1
  total elapsed: X us

correctness:
  max_abs_error <= tolerance
  max_relative_error <= tolerance

explain:
  add+tanh fused because same shape/layout/dtype and no side effects
  dot+broadcast/add/tanh recognized as an epilogue candidate but rejected in V0
  because kernel IR, math policy, and backend epilogue support are not explicit
```

V0 does not need to be fast. It needs to make performance structure visible.

## Artifacts

V0 should create concrete artifacts, preferably printable as text or JSON:

```text
GraphDump
TargetDump
CostLedgerDump
ScheduleDump
ProfileDump
ExplainDump
CorrectnessDump
```

These artifacts should be stable enough for tests.

## Typed Graph IR V0

V0 needs a graph IR that is smaller than the final vision but stronger than the
current optional-field instruction bag.

Suggested records:

```zig
pub const GraphValueId = struct {
    index: u32,
};

pub const SourceRef = struct {
    frontend: enum { pjrt, stablehlo, internal },
    op_name: []const u8,
    source_index: u32,
    location: []const u8,
};

pub const TensorType = struct {
    element_type: core.BufferType,
    dims: []const i64,
    layout: core.LayoutKind,
};

pub const GraphValue = struct {
    id: GraphValueId,
    ty: TensorType,
    role: enum { parameter, constant, instruction_result, output },
    source: ?SourceRef,
};

pub const GraphInstruction = struct {
    id: u32,
    kind: GraphInstructionKind,
    inputs: []const GraphValueId,
    outputs: []const GraphValueId,
    payload: GraphPayload,
    source: SourceRef,
};

pub const GraphPayload = union(enum) {
    dot_general: DotGeneralSpec,
    elementwise_unary: ElementwiseUnarySpec,
    elementwise_binary: ElementwiseBinarySpec,
    broadcast: BroadcastSpec,
    return_: ReturnSpec,
};
```

Important V0 rule:

```text
No instruction may carry metadata that does not belong to its payload.
```

This proves the future typed-payload direction without refactoring every
existing operation at once.

## Target Model V0

V0 should define a minimal target description. For Metal, some fields may
be approximate or unknown.

```zig
pub const TargetDescription = struct {
    name: []const u8,
    devices: []const TargetDevice,
    memory_spaces: []const TargetMemorySpace,
    transfer_edges: []const TargetTransferEdge,
    execution_units: []const ExecutionUnit,
};

pub const ExecutionUnit = struct {
    id: u32,
    name: []const u8,
    kind: enum { scalar, vector, matrix, library, unknown },
    dtype_rates: []const DTypeRate,
};

pub const DTypeRate = struct {
    dtype: core.BufferType,
    op_class: enum { elementwise, matmul, transcendental, memory },
    ops_per_second: ?f64,
    source: enum { measured, vendor_spec, heuristic, unknown },
};

pub const TargetMemorySpace = struct {
    id: u32,
    name: []const u8,
    kind: core.MemoryKind,
    capacity_bytes: ?u64,
    bandwidth_bytes_per_second: ?f64,
};

pub const TargetTransferEdge = struct {
    src_memory_space: u32,
    dst_memory_space: u32,
    bandwidth_bytes_per_second: ?f64,
    latency_ns: ?u64,
    supports_async: bool,
};
```

V0 target fields can be conservative. Unknown is better than pretending.

Example:

```text
target.name = metal_v0
execution_units = [metal_shader_core, unknown_gpu]
memory_spaces = [host_pinned, host_unpinned, device_unified]
```

## Cost Ledger V0

The cost ledger is the first step toward vertical performance traceability.

```zig
pub const CostLedgerEntry = struct {
    id: u32,
    source_instruction_id: u32,
    graph_instruction_ids: []const u32,
    op_class: enum {
        matmul,
        elementwise,
        transcendental,
        transfer,
        backend_kernel,
    },
    dtype: core.BufferType,
    accumulation_dtype: ?core.BufferType,
    logical_ops: u128,
    bytes_read: u128,
    bytes_written: u128,
    expected_unit_id: ?u32,
    lowering_stage: []const u8,
};
```

Cost formulas for V0:

```text
dot_general rank-2:
  logical_ops = 2 * M * N * K
  bytes_read = sizeof(dtype) * (M*K + K*N)
  bytes_written = sizeof(dtype) * (M*N)

elementwise unary:
  logical_ops = element_count
  bytes_read = sizeof(dtype) * element_count
  bytes_written = sizeof(dtype) * element_count

elementwise binary:
  logical_ops = element_count
  bytes_read = sizeof(dtype) * element_count * 2
  bytes_written = sizeof(dtype) * element_count

host_to_device/device_to_host:
  logical_ops = 0
  bytes_read / bytes_written = transfer byte count
```

For `tanh`, V0 should mark the op class as `transcendental` rather than pretend
it is the same as add.

Correctness of the ledger matters. If the formula is approximate, the entry
must say so.

## Lowering V0

V0 lowering can be simple:

```text
dot_general -> backend kernel graph command
add + tanh -> elementwise fusion group if same shape/dtype/layout
host inputs -> h2d transfer commands
output -> d2h transfer command when requested
```

Lowering records:

```zig
pub const LoweringRecord = struct {
    id: u32,
    graph_instruction_ids: []const u32,
    decision: enum {
        backend_kernel_graph,
        elementwise_fusion,
        transfer,
        unsupported,
    },
    reason: []const u8,
    rejected_alternatives: []const []const u8,
    cost_ledger_ids: []const u32,
};
```

Example:

```text
lowering.0:
  graph_instruction_ids: [dot_general]
  decision: backend_kernel_graph
  reason: V0 treats matmul as backend kernel graph boundary

lowering.1:
  graph_instruction_ids: [add, tanh]
  decision: elementwise_fusion
  reason: same output shape, same dtype, no side effects
```

## Schedule IR V0

The schedule should be explicit even if execution remains mostly sequential.

```zig
pub const ScheduleCommand = struct {
    id: u32,
    kind: CommandKind,
    stream: StreamId,
    inputs: []const GraphValueId,
    outputs: []const GraphValueId,
    dependencies: []const CommandDependency,
    lowering_record_id: ?u32,
    cost_ledger_ids: []const u32,
};

pub const CommandKind = enum {
    host_to_device,
    backend_execute,
    device_to_host,
    event_record,
    event_wait,
};

pub const CommandDependency = struct {
    command_id: u32,
    kind: enum { data, stream_order, memory_availability },
};
```

V0 streams:

```text
stream.0 = host
stream.1 = compute
stream.2 = transfer
```

V0 may execute synchronously, but the schedule must still describe where async
execution would fit.

## Host/Device IO V0

Transfers must be visible.

```zig
pub const TransferRecord = struct {
    id: u32,
    kind: enum { h2d, d2h },
    source_value: ?GraphValueId,
    destination_value: ?GraphValueId,
    byte_size: usize,
    dtype: core.BufferType,
    dims: []const i64,
    source_memory: []const u8,
    destination_memory: []const u8,
    schedule_command_id: u32,
};
```

V0 requirements:

- all input H2D copies are counted
- all requested output D2H copies are counted
- byte size is overflow checked
- dtype and dims match graph values
- profile events include transfer time

## Backend Execution V0

V0 does not need a new backend. It can wrap the current backend executable path
and attach trace IDs around it, but this is a backend binding, not a runtime
fallback. If the selected target/backend cannot bind the V0 workload, compile
fails before runtime execution.

Minimum backend binding record:

```zig
pub const BackendBinding = struct {
    command_id: u32,
    backend_kind: core.BackendKind,
    backend_operation: []const u8,
    graph_instruction_ids: []const u32,
    expected_unit_id: ?u32,
};
```

For Metal:

```text
backend_operation = "metal_mls_graph_execute"
expected_unit_id = metal_shader_core or unknown_gpu
```

The architecture should not pretend this is full kernel generation yet. It is a
binding from graph work to a backend executable command.

The backend package should then expand that binding into a backend executable
plan:

```zig
pub const BackendExecutableCall = struct {
    graph_instruction_id: u32,
    feature: BackendFeature,
    backend_operation: []const u8,
    input_value_ids: []const u32,
    output_value_ids: []const u32,
    expected_unit_id: ?u32,
};
```

The Metal bridge should also materialize the MLS kernel graph:

```zig
pub const BackendKernelGraphNode = struct {
    call_index: u32,
    graph_instruction_id: u32,
    backend_operation: []const u8,
    input_value_ids: []const u32,
    output_value_ids: []const u32,
    output_type: BackendTensorDescriptor,
    attributes: BackendKernelAttributes,
};

pub const BackendKernelGraphEdge = struct {
    value_id: u32,
    src_node_id: u32,
    dst_node_id: u32,
};

pub const BackendTensorDescriptor = struct {
    element_type: BufferType,
    dims: []const i64,
    layout: LayoutKind,
};

pub const BackendKernelAttributes = union(enum) {
    rank2_dot_general: DotGeneralSpec,
    broadcast_in_dim: []const u32,
    add: void,
    tanh: void,
    elementwise_fusion: []const GraphInstructionId,
};
```

For Metal V0, the current call sequence is `metal_mls_matmul_kernel` followed
by `metal_mls_elementwise_fusion_kernel` for the broadcast/add/tanh lowering
region. This is still not fallback; it is the explicit bridge from verified
compiler lowering regions to generated Metal/MLS graph and kernel planning.
The graph records dataflow such as the matmul output feeding the fused
elementwise node. Each node records output dtype, rank/shape, layout, fused
instruction provenance, and operation attributes so Metal codegen does not need
to rediscover StableHLO semantics from strings.

## Profiling V0

V0 profiling should be lightweight and deterministic enough for tests.
The profile stream records coarse schedule-command events and backend
lowering-region events. For the V0 workload, the backend command therefore has
one command-level event plus one matmul lowering event and one fused
elementwise lowering event.

```zig
pub const ProfileEvent = struct {
    id: u32,
    command_id: u32,
    graph_instruction_ids: []const u32,
    kind: enum { compile_pass, h2d, backend_execute, d2h },
    start_ns: u64,
    duration_ns: u64,
    bytes: u128,
    logical_ops: u128,
    status: enum { ok, failed },
};
```

Derived report:

```zig
pub const ProfileSummary = struct {
    total_compile_ns: u64,
    total_execute_ns: u64,
    h2d_bytes: u128,
    d2h_bytes: u128,
    logical_ops: u128,
    observed_ops_per_second: ?f64,
    observed_h2d_bandwidth: ?f64,
    observed_d2h_bandwidth: ?f64,
};
```

V0 should not require hardware counters. Wall-clock timing around compile,
transfer, execute, and readback is enough.

## Explain Trace V0

Every important artifact should be linked by ID.

```zig
pub const ExplainRecord = struct {
    id: u32,
    pass_name: []const u8,
    subject: ExplainSubject,
    decision: []const u8,
    reason: []const u8,
    source_refs: []const SourceRef,
    cost_ledger_ids: []const u32,
    profile_event_ids: []const u32,
};

pub const ExplainSubject = union(enum) {
    graph_instruction: u32,
    lowering_record: u32,
    schedule_command: u32,
    backend_binding: u32,
};
```

V0 explanation examples:

```text
pass=graph-import
subject=stablehlo.dot_general
decision=created graph instruction 2
reason=rank-2 dot_general supported by V0 graph importer

pass=lowering
subject=graph instructions [3,4]
decision=fused elementwise region
reason=add and tanh have same shape/dtype/layout and no side effects

pass=schedule
subject=command 2
decision=backend execute on compute stream
reason=all inputs are resident on device
```

## Correctness V0

Correctness checks should be boring and strict.

Compile-time checks:

- all graph values have valid dtype and dims
- `dot_general` rank-2 dimensions match
- elementwise operations have compatible shapes
- no unsupported op enters schedule
- no schedule command references missing values
- transfer byte sizes match tensor descriptors

Runtime checks:

- all input buffers are live
- all backend buffers are available before execute
- D2H waits for backend execution
- output dtype and dims match expected graph value

Numerical check:

```text
Compare PjRTx result to JAX CPU or NumPy reference.
For f32 V0:
  max_abs_error <= 1e-4
  max_relative_error <= 1e-4
```

If Metal tanh or matmul differs slightly, the tolerance must be recorded in
the correctness report. No silent widening of tolerance.

## Tests

Add tests at three levels.

Unit tests:

- graph import creates typed payloads
- cost ledger formulas for dot and elementwise ops
- schedule command dependencies are valid
- transfer byte-size overflow checks
- explain records link to existing IDs

Integration tests:

- compile `tanh(dot(x, w) + b)`
- expand Metal backend binding into executable calls
- execute through Metal when available
- compare output to JAX CPU
- assert profile events include H2D, backend execute, and D2H
- assert cost ledger includes dot FLOPs and elementwise ops
- assert explain trace maps source ops to schedule commands

Golden report test:

```text
Run the workload and compare a stable normalized report.
Timing fields can be redacted.
IDs, op names, shapes, dtype, command kinds, and ledger formulas should remain.
```

## V0 Report Shape

The report can be text first and JSON later.

```text
PjRTx vertical slice report

program:
  name: tanh_dot_bias
  dtype: f32

target:
  name: metal_v0
  memory_spaces: host_unpinned, device_unified
  execution_units: metal_shader_core, unknown_gpu

graph:
  value.0 parameter f32[8,16] source=arg0
  value.1 parameter f32[16,32] source=arg1
  value.2 parameter f32[32] source=arg2
  inst.0 dot_general inputs=[0,1] outputs=[3]
  inst.1 add inputs=[3,2] outputs=[4]
  inst.2 tanh inputs=[4] outputs=[5]

cost:
  inst.0 matmul logical_ops=8192 bytes_read=... bytes_written=...
  inst.1 elementwise logical_ops=256
  inst.2 transcendental logical_ops=256

lowering:
  inst.0 -> command.3 backend_kernel_graph
  inst.1+inst.2 -> command.3 backend_execute region

schedule:
  command.0 h2d value.0
  command.1 h2d value.1
  command.2 h2d value.2
  command.3 backend_execute deps=[0,1,2]
  command.4 d2h value.5 deps=[3]

profile:
  h2d_bytes=...
  d2h_bytes=...
  backend_execute_ns=...
  observed_ops_per_second=...

correctness:
  max_abs_error=...
  max_relative_error=...
  status=ok

explain:
  inst.1+inst.2 fused: same shape/dtype/layout, no side effects
  inst.0+inst.1+inst.2+inst.3 not fused as matmul epilogue in V0:
  kernel IR, math policy, and backend epilogue support are not explicit
```

## Exit Criteria

V0 is done when:

- one workload compiles and runs through the current PJRT path
- the typed graph exists for that workload
- the target model exists, even if approximate
- FLOPs and bytes are reported by source instruction
- schedule commands are reported
- profile events are reported
- source operations map to backend execution
- correctness is checked against reference
- unsupported cases fail clearly before runtime execution

After V0, the next vertical slices should be:

1. **Memory Slice V1**: explicit memory plan, allocation lifetimes, donation,
   resident constants, and transfer overlap.
2. **Collective Slice V1**: all-reduce or reduce-scatter with participant
   groups, channel IDs, schedule records, and correctness tests.
3. **Tiling Slice V1**: NPU target with SRAM, tile plans, cost estimates,
   and explainable tile selection.
4. **AOT Slice V1**: serialize executable schedule, target requirements,
   constants, and explain/correctness manifests.
