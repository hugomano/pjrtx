# Target Model V0

This spec defines the minimum target model for the new PjRTx architecture under
`//next/pjrtx/...`.

V0 target modeling should make performance estimates honest without pretending
to know what the backend cannot expose.

Implementation ownership: concrete target records, validation, dtype-rate
records, and target-local summary rendering live in `//next/pjrtx/target`. During the
`//next/pjrtx/core` migration, old names may be re-exported only as a compatibility
bridge; new code should import `pjrtx/target` directly.

The first targets:

```text
metal_v0
  Bootstrap target for generated Metal/MLS graph and kernel planning.

npu_v0
  TRN2-like synthetic target for testing hardware-aware traceability, memory
  spaces, dtype rates, and schedule reports.
```

## Requirements

The target model must:

- describe execution units
- describe dtype-specific rates
- describe memory spaces
- describe transfer edges
- preserve unknown fields explicitly
- be printable in stable reports
- avoid backend-private handles
- be usable by cost ledger and explain records

V0 does not need exact hardware counters or full device discovery.

## TargetDescription

```zig
pub const TargetDescription = struct {
    name: []const u8,
    kind: TargetKind,
    devices: []const TargetDevice,
    memory_spaces: []const TargetMemorySpace,
    transfer_edges: []const TargetTransferEdge,
    execution_units: []const ExecutionUnit,
};

pub const TargetKind = enum {
    metal_v0,
    npu_v0,
};
```

Reports should always include target name and kind.

## TargetDevice

```zig
pub const TargetDevice = struct {
    id: u32,
    local_hardware_id: i32,
    name: []const u8,
    memory_space_ids: []const u32,
    execution_unit_ids: []const u32,
};
```

V0 is single-device by default. Multi-device topology is future work, except
that IDs should not prevent multi-device later.

## ExecutionUnit

```zig
pub const ExecutionUnitKind = enum {
    scalar,
    vector,
    matrix,
    library,
    dma,
    collective,
    unknown,
};

pub const ExecutionUnit = struct {
    id: u32,
    name: []const u8,
    kind: ExecutionUnitKind,
    dtype_rates: []const DTypeRate,
};
```

Examples:

```text
metal_shader_core
unknown_gpu
trn2_tensor_engine
trn2_vector_engine
trn2_dma_engine
trn2_collective_engine
```

## DTypeRate

```zig
pub const OpClass = enum {
    elementwise,
    matmul,
    transcendental,
    memory,
    collective,
};

pub const RateSource = enum {
    measured,
    vendor_spec,
    heuristic,
    synthetic,
    unknown,
};

pub const DTypeRate = struct {
    dtype: target_pkg.BufferType,
    op_class: OpClass,
    ops_per_second: ?f64,
    source: RateSource,
    note: []const u8,
};
```

Rules:

- unknown rates use `ops_per_second = null`
- NPU target rates use `source = synthetic`
- measured rates must say how they were measured in `note`
- dtype rates are not interchangeable across op classes

## Memory Spaces

```zig
pub const MemorySpaceKind = enum {
    host_unpinned,
    host_pinned,
    device_unified,
    device_hbm,
    local_sram,
    scratchpad,
    remote_device,
    unknown,
};

pub const TargetMemorySpace = struct {
    id: u32,
    name: []const u8,
    kind: MemorySpaceKind,
    capacity_bytes: ?u64,
    bandwidth_bytes_per_second: ?f64,
    note: []const u8,
};
```

Rules:

- unknown capacity/bandwidth is represented as `null`, not zero
- `device_unified` is valid for Metal bootstrap but should not hide transfer
  accounting
- NPU should expose at least HBM and local SRAM

## Transfer Edges

```zig
pub const TargetTransferEdge = struct {
    id: u32,
    src_memory_space: u32,
    dst_memory_space: u32,
    bandwidth_bytes_per_second: ?f64,
    latency_ns: ?u64,
    supports_async: bool,
    engine_unit_id: ?u32,
    note: []const u8,
};
```

Transfer edges should be directed. If both directions are supported, declare
both edges.

## `metal_v0`

Purpose:

```text
Bridge the new architecture to generated Metal/MLS graph and kernel planning
without claiming more hardware visibility than Metal exposes.
```

Suggested target:

```text
name: metal_v0
kind: metal_v0

memory_spaces:
  host_unpinned
  device_unified

execution_units:
  metal_shader_core
  unknown_gpu

transfer_edges:
  host_unpinned -> device_unified
  device_unified -> host_unpinned
```

Rates:

```text
metal_shader_core f32 matmul ops_per_second=unknown
metal_shader_core f32 elementwise ops_per_second=unknown
unknown_gpu f32 memory ops_per_second=unknown
```

Rules:

- report unknown rates as `unknown`
- generated Metal/MLS kernel planning is explicit in backend executable calls
- do not claim submitted Metal command-buffer execution until PjRTx owns that
  path
- profile H2D/backend/D2H even on unified memory
- backend binding may use expected unit `metal_shader_core` or `unknown_gpu`

## `npu_v0`

Purpose:

```text
Prove hardware-aware traceability before real NPU integration. The V0 target is
TRN2-like: it models HBM, local SRAM, tensor/vector execution, DMA, and a
collective engine, but all rates remain synthetic until measured or replaced by
vendor facts.
```

Suggested target:

```text
name: npu_v0
kind: npu_v0

memory_spaces:
  host_pinned
  device_hbm
  local_sram

execution_units:
  trn2_tensor_engine
  trn2_vector_engine
  trn2_dma_engine
  trn2_collective_engine
```

Suggested synthetic rates:

```text
trn2_tensor_engine bf16 matmul 100e12 ops/s
trn2_tensor_engine f32 matmul 25e12 ops/s
trn2_vector_engine f32 elementwise 5e12 ops/s
trn2_vector_engine f32 transcendental 0.5e12 ops/s
trn2_dma_engine memory 1e12 bytes/s equivalent tracked through transfer edges
```

Suggested memory spaces:

```text
host_pinned capacity=unknown bandwidth=unknown
device_hbm capacity=32 GiB bandwidth=1 TB/s
local_sram capacity=64 MiB bandwidth=20 TB/s
```

Suggested transfer edges:

```text
host_pinned -> device_hbm async=true
device_hbm -> host_pinned async=true
device_hbm -> local_sram async=true
local_sram -> device_hbm async=true
```

NPU does not need to execute real kernels. It may produce synthetic profile
events based on cost estimates.

## Stable Report Format

Target report section should include:

```text
target:
  name: npu_v0
  kind: npu_v0

memory_spaces:
  memory.0 host_pinned capacity=unknown bandwidth=unknown
  memory.1 device_hbm capacity=34359738368 bandwidth=1000000000000
  memory.2 local_sram capacity=67108864 bandwidth=20000000000000

execution_units:
  unit.0 trn2_tensor_engine kind=matrix
    rate dtype=bf16 op_class=matmul ops_per_second=100000000000000 source=synthetic

transfer_edges:
  edge.0 host_pinned -> device_hbm async=true bandwidth=64000000000
  edge.1 device_hbm -> host_pinned async=true bandwidth=64000000000
```

Unknown fields should be rendered as `unknown`, not omitted.

## Validation

Target validation should check:

- device memory space IDs exist
- device execution unit IDs exist
- transfer edge memory spaces exist
- transfer edge engine unit exists when provided
- dtype rates use valid dtypes
- target has at least one device
- target has at least one memory space
- target has at least one execution unit

Validation should write diagnostics through `std.Io.Writer`.

Runtime allocation planning must additionally check target facts against the
planned workload:

- if a memory-space capacity is known, peak live bytes for that space must not
  exceed it
- every scheduled H2D/D2H command must resolve to a target transfer edge
- missing transfer bandwidth may produce `ideal_transfer_ps=0`, but a missing
  transfer edge is invalid

## V0 Decisions

Transfer edges are authoritative for movement bandwidth and latency. Memory
space bandwidth is a local-access hint for roofline estimates. If both are
present and disagree, transfer planning uses the edge value and compute
roofline estimates use the memory-space value.

`TargetMemorySpace.kind` uses the richer target enum defined here. Boundary code
may map it to narrower runtime/core memory enums, but the target model must not
lose distinctions such as HBM, local SRAM, scratchpad, remote device memory,
and unified device memory.

NPU synthetic profile events are produced by the NPU backend implementation.
Profile event facts belong under `//next/pjrtx/runtime/facts`, while target
hardware facts remain under `//next/pjrtx/target`. Runtime tests only validate event
ordering, report joins, and no-fallback invariants.

V0 target fingerprints are serialized as canonical text plus an explicit schema
version:

```text
pjrtx-target-fingerprint-v0
name=<target-name>
kind=<target-kind>
device...
memory...
unit...
edge...
```

Golden tests compare the canonical fingerprint material directly. A hash can be
added for cache keys, but the human-readable material is the source of truth for
V0.
