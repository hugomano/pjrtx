# PjRTx Architecture Vision

PjRTx should become a target-driven compiler and runtime for heterogeneous
accelerator platforms. PJRT compatibility is the frontend contract, not the
center of the architecture.

The goal is:

> Given a program, a topology, hardware specifications, and runtime constraints,
> produce an explainable executable plan that maximizes useful end-to-end
> performance while preserving mathematical correctness at every layer.

This document describes the architecture PjRTx should grow toward. XLA is a
useful reference for vocabulary and proven compiler/runtime problems, but it is
not the full vision. Some parts of XLA exist because of history, backend
specific constraints, and accumulated technical debt. PjRTx should borrow the
hard lessons while designing cleaner interfaces around performance,
correctness, explainability, and composability.

Design decisions and reversals should be recorded in
`docs/architecture_logbook.md` so the reasoning does not get lost in chat
history.

## Current Correction

The new architecture must not become only a runtime plugin under a hidden XLA
compiler. Many accelerator plugins start by relying on XLA for lowering,
fusion, tiling, collectives, and scheduling, then only implement the
StreamExecutor-style layer. That path can run programs, but it does not satisfy
the PjRTx goal: from PJRT-level computation and hardware specification to
explainable FLOPs, bytes, memory movement, collectives, generated or selected
kernels, and profile feedback.

The reroute is compiler-middle first, and the compiler-middle should be
MLIR-first. The current Zig trace records are useful scaffolding and extracted
views, but they should not become the permanent representation for lowering,
fusion, tiling, memory planning, collectives, codegen, or scheduling. Those
facts should live in PjRTx MLIR dialect state with verifiers, then be extracted
to compact Zig reports and runtime handoff structures.

Before deepening backend submission, PjRTx must represent:

- MLIR/StableHLO pass-pipeline reports
- an explicit compiler pass pipeline with small passes, stable names,
  correctness/performance invariants, MLIR state updates, and extracted records
- fusion candidates, accepted groups, and rejected alternatives
- layout, tile, and memory-space plans
- collective specs and collective lowering records
- cost records that join those decisions to target hardware rates
- schedule and backend binding records derived from those compiler decisions

StreamExecutor-like runtime submission begins after a verified executable
schedule exists. It is not the compiler.

## Two Versions

PjRTx intentionally has two versions living side by side.

The current implementation remains the README-described bootstrap architecture
under `//src/...`. That code should keep working and should not be rewritten
accidentally while the new architecture is being designed.

The new architecture lives in `docs/` and should be implemented under
`//pjrtx/...`. It is allowed to bridge to `//src/...` during migration, but the
new public architecture should not be mixed into the legacy package namespace.

In short:

```text
README.md + //src/...      current/bootstrap implementation
docs/*.md + //pjrtx/...    new architecture and implementation
```

## Document Map

This file is the canonical long-range architecture. The V0 implementation
should use these narrower specs as executable design contracts:

- `docs/vertical_slice_v0.md`: the first vertical slice scope.
- `docs/specs/mlir_state_machine_compiler_v0.md`: the intended MLIR-first
  compiler state machine where compiler-middle facts live in MLIR dialect
  state and Zig records are extracted views.
- `docs/specs/pjrtx_mlir_dialect_v0.md`: the first concrete PjRTx MLIR
  dialect/API slice, including module state, target attachment, fusion facts,
  pressure deltas, Zig `MlirSession`, and extraction rules.
- `docs/specs/final_mlir_dialect_op_pass_architecture_v0.md`: the destination
  PjRTx dialect/op/pass architecture after the current attribute-state bridge.
- `docs/specs/package_boundaries_v0.md`: the segregated package graph and the
  no-`//pjrtx/core` ownership rule.
- `docs/specs/trace_schema_v0.md`: stable IDs, provenance records, profile
  events, and report joins extracted from compiler/runtime state.
- `docs/specs/target_model_v0.md`: target, memory, bandwidth, execution-unit,
  dtype-rate, and fingerprint records.
- `docs/specs/compile_pipeline_v0.md`: no-fallback compile gates from input to
  executable creation.
- `docs/specs/compiler_pass_pipeline_v0.md`: explainable XLA-like compiler pass
  families, pass contracts, ordering, and invariants.
- `docs/specs/xla_coverage_map_v0.md`: local XLA coverage map and the design
  lesson that PjRTx should match XLA's breadth while making the causal chain
  from PJRT program to hardware profile first-class.
- `docs/specs/harness_v0.md`: fast tests, golden reports, debug dumps, and
  hardware execution policy.
- `docs/specs/correctness_policy_v0.md`: strict math defaults and reference
  oracle policy.
- `docs/specs/implementation_workplan_v0.md`: concrete packages, build phases,
  and acceptance criteria.

When these documents disagree, fix the docs before implementing. The specs are
allowed to be smaller than the vision, but not weaker on correctness,
explainability, or no-fallback behavior.

Package ownership is part of the architecture. New public architecture code
must not grow around a central `//pjrtx/core` package. Types live with the
package that owns their invariants, and reports compose extracted facts through
explicit package contracts.

## Principles

Performance is target-specific. A compiler cannot optimize for abstract FLOPs.
It must optimize for a concrete system: dtype-specific compute units, memory
spaces, bandwidths, interconnects, collective engines, DMA engines, launch
costs, synchronization costs, and runtime allocation constraints.

Mathematical correctness is a first-class property. Lowering cannot silently
change StableHLO semantics, dtype behavior, reduction ordering guarantees,
NaN/infinity behavior, shape semantics, aliasing, or sharding semantics. Any
intentional relaxation must be represented in compile options, recorded in the
explain plan, and tested.

Explainability is part of the product. Every important compiler decision should
leave a structured trail: why a fusion happened, why a fusion was rejected, why
a tile shape was selected, why a tensor lives in one memory space instead of
another, why a collective is chunked, and what bottleneck the compiler predicts.

Vertical traceability is a design requirement. A user should be able to start
from a PJRT-visible computation and follow it down through StableHLO, compiler
IR, lowering passes, schedule commands, generated or selected kernels, memory
operations, collectives, and target hardware units. FLOPs, bytes, latency, and
correctness assumptions must remain visible across that whole stack.

Composability beats one giant backend switch. Backends should expose target
capabilities, memory spaces, communication capabilities, codegen hooks, and
runtime command execution. The compiler should choose strategies from those
capabilities instead of asking every backend to implement a flat list of
StableHLO operations.

The runtime should execute schedules, not interpret operations. Runtime should
own devices, memory, streams, events, allocators, command buffers, profiling,
and executable instances. A reference interpreter can exist as a test oracle,
but it must not be the production execution path.

There is no runtime fallback path in the new architecture. If a program cannot
be fully legalized to the selected target, compilation must fail with precise
diagnostics before execution. Reference execution is only for tests and
correctness oracles.

## XLA Coverage Lesson

The local XLA tree demonstrates the breadth PjRTx must eventually cover:
StableHLO/HLO import, HLO pass pipelines, simplification, decomposition,
fusion, tiling, Shardy/SPMD, collective lowering, layout assignment,
memory-space assignment, buffer assignment, scheduling, backend codegen,
autotuning, PJRT integration, StreamExecutor-style runtime submission,
host/device IO, allocation, and profiling.

That breadth is correct. The part PjRTx should not inherit is opacity. In XLA,
performance can depend on decisions spread across HLO transforms,
backend-specific compiler files, collective passes, memory planners, autotuning,
buffer assignment, StreamExecutor, allocators, and runtime profiling. Those
pieces can be debugged, but they are not one causal product surface.

PjRTx should make the causal chain explicit:

```text
PJRT-visible program
  -> StableHLO / Shardy metadata
  -> PjRTx MLIR state machine
  -> typed graph/extracted views
  -> compiler pass records extracted from MLIR transitions
  -> fusion, layout, tile, collective, and memory-space MLIR plans
  -> cost and roofline records from hardware facts
  -> lowering regions
  -> schedule, allocation, stream, and event records
  -> backend bindings
  -> generated kernels, library calls, kernel graphs, DMA, or collectives
  -> runtime profile events and hardware counters
```

The standard is not just "does this run?" The standard is: can a user explain
which source operations became which kernels or collectives, why the compiler
chose that lowering, what hardware unit and dtype rate apply, which bytes moved
through which memory spaces, and which profile event measured the result?

## What PjRTx Is Today

The current project is a PJRT C API producer with a compiler/runtime/backend
split. It can ingest StableHLO/MLIR, build a PjRTx executable plan, require
device-only lowering through the Metal backend, and execute supported
programs through opaque backend handles.

That is a useful bootstrap, but the current shape is still too close to:

```text
PJRT -> StableHLO ingest -> PjRTx plan -> backend executable -> Metal operation calls
```

The architecture should move toward:

```text
PJRT / future frontends
  -> StableHLO and Shardy import
  -> canonical graph IR
  -> target and topology analysis
  -> sharding and collective formation
  -> layout, fusion, tiling, and memory planning
  -> collective lowering and latency hiding
  -> kernel, library, DMA, and communication command generation
  -> executable schedule
  -> runtime streams, allocators, events, profiling
```

## Target Model

The target model should describe the hardware as a system, not as one device
with a peak FLOP number.

Core target concepts:

```zig
TargetDescription
  platform_kind
  hosts
  devices
  mesh_axes
  memory_spaces
  memory_transfer_edges
  interconnect_links
  execution_units
  dma_engines
  collective_engines
  supported_dtypes
  layout_constraints
  codegen_backends
  runtime_capabilities
```

Device performance must be dtype-specific:

```zig
DTypeRate
  dtype
  operation_class
  units_per_core
  ops_per_cycle
  clock_hz
  sparsity_multiplier
  accumulation_type
```

Transfer performance must be edge-specific. Host/device, HBM/SRAM,
inter-device, and collective fabrics have different bandwidth, latency,
asynchrony, and engine ownership. Reports should join planned bytes to the
exact transfer edge used and show an ideal transfer time derived from that edge
whenever the target specification provides bandwidth.

This matters for GPUs, but it is even more important for NPUs, TPUs, and
Neuron-like devices. A platform can have scalar cores, vector cores, matrix
cores, systolic arrays, tensor engines, collective engines, and DMA engines.
The compiler must know which unit can execute which operation, at which dtype,
with which accumulation behavior, and with which memory access pattern.

The target model should also encode topology:

```zig
DeviceTopology
  process_index
  local_device_id
  global_device_id
  coordinates
  mesh_axis_membership
  local_links
  cross_host_links
  addressable_memory_spaces
```

Performance estimates should use a roofline-like view, but not only a FLOP
roofline. Useful lower bounds include compute time, memory time, interconnect
time, collective time, DMA time, launch time, synchronization time, and
allocation pressure.

## Hardware Specification And Vertical Traceability

PjRTx should make the path from PJRT to hardware inspectable. The user should
be able to ask:

```text
For this PJRT executable, where are the FLOPs?
Which StableHLO operations produced them?
Which lowering pass fused or tiled them?
Which kernel or kernel graph performs them?
Which hardware unit is expected to execute them?
Which dtype rate is used for the estimate?
Which bytes feed those FLOPs?
Which memory spaces and interconnects constrain them?
Which measured profile events confirm or contradict the estimate?
```

This is where PjRTx should deliberately improve on systems where compilation
becomes opaque. Some computations in mature compiler stacks are hard to explain
because provenance is lost, rewritten away, or scattered across backend-specific
side channels. PjRTx should preserve provenance as data.

The hardware specification should be rich enough to support this trace. It does
not need to literally ingest Verilog on day one, but the model should be able
to line up with RTL-level or hardware-design-level concepts when those specs
exist.

Useful hardware-spec concepts:

```zig
HardwareSpec
  clock_domains
  execution_units
  pipelines
  issue_slots
  vector_lanes
  matrix_shapes
  systolic_array_shapes
  accumulator_types
  register_files
  local_memories
  memory_banks
  dma_engines
  collective_engines
  interconnect_ports
  instruction_formats
  supported_kernel_abis
  hardware_counters
```

Every generated or selected kernel should have a hardware binding record:

```zig
KernelHardwareBinding
  kernel_id
  source_instruction_ids
  fusion_group_id
  target_unit
  dtype
  accumulation_dtype
  logical_flops
  effective_ops
  bytes_read_by_space
  bytes_written_by_space
  expected_occupancy
  expected_pipeline_utilization
  expected_memory_utilization
  expected_interconnect_utilization
```

The FLOP count itself must be structured. A matmul, a transcendental op, an
integer bitwise op, and a collective reduction are not interchangeable.

```zig
OperationCostLedger
  source_value_ids
  source_instruction_ids
  op_class
  dtype
  accumulation_dtype
  logical_ops
  hardware_ops
  bytes_accessed
  expected_unit
  lowering_stage
```

The ledger should survive transformations. If five StableHLO operations become
one fused kernel, the fused kernel should still point back to all five source
operations. If one all-reduce is decomposed into reduce-scatter plus all-gather,
the new collective commands should point back to the original collective and
record the reason for decomposition.

The ideal explanation path is:

```text
PJRT executable
  -> StableHLO op and source location
  -> canonical graph value/instruction
  -> fusion group or decomposition
  -> tile/layout/memory-space decision
  -> command graph node
  -> generated kernel, kernel graph, DMA, or collective
  -> hardware unit and dtype rate
  -> profile event and hardware counters
```

If PjRTx eventually targets hardware described close to Verilog/RTL, the
compiler should be able to map high-level work to hardware features without
pretending it controls every transistor. For example:

- matrix tiles map to systolic array dimensions or tensor-engine instruction
  shapes
- vectorized elementwise work maps to vector lanes and issue slots
- scratchpad tiles map to SRAM banks and DMA engines
- collective chunks map to interconnect links and collective engines
- profile events map to hardware counters where available

This design should make it possible to view expected FLOPs and achieved FLOPs
side by side:

```zig
FlopTrace
  logical_flops
  lowered_flops
  issued_hardware_ops
  achieved_ops_per_second
  percent_of_dtype_peak
  limiting_resource
  explanation_record_id
```

The key rule: performance data must not be detached from program semantics. A
fast kernel with no source provenance is not explainable. A correct graph with
no hardware binding is not performance-engineered. PjRTx needs both.

## Memory Model

PjRTx should make memory hierarchy explicit.

Example memory spaces:

```zig
MemorySpace
  host_unpinned
  host_pinned
  device_hbm
  device_dram
  local_sram
  scratchpad
  l2_cache
  register_file
  remote_device_memory
  interconnect_visible_memory
```

Example transfer edges:

```zig
MemoryTransferEdge
  src_space
  dst_space
  bandwidth_bytes_per_sec
  latency_ns
  max_inflight_bytes
  supports_async
  requires_alignment
  engine_id
```

This enables memory space assignment, similar in spirit to XLA's alternate
memory assignment, but generalized for heterogeneous targets. The compiler
should decide when to keep a tensor in HBM, prefetch to SRAM, evict from SRAM,
slice a copy to reduce pressure, use pinned host staging, or recompute instead
of storing.

A memory plan should include:

```zig
MemoryPlan
  logical_buffer_lifetimes
  physical_allocations
  offsets
  memory_spaces
  aliases
  donated_parameter_aliases
  temporary_workspaces
  prefetches
  evictions
  sliced_copies
  peak_memory_by_space
  fragmentation_estimate
```

Runtime then needs a real device allocator:

```zig
DeviceAllocator
  allocate(space, size, alignment)
  deallocate(handle)
  suballocate(arena, size, alignment)
  reserve_workspace(size)
  import_external_allocation(...)
  export_raw_buffer(...)
  report_stats()
```

The allocator must support production concerns: alignment, aliasing, donation,
workspace reservation, asynchronous deallocation, fragmentation, multi-stream
lifetimes, and memory pressure diagnostics.

## Graph And IR Layers

PjRTx should avoid using one struct as every IR. Each layer needs its own
semantic level.

Suggested layers:

```text
StableHLO module
  Source program semantics.

Canonical graph IR
  Operation graph with typed values, shapes, layouts, side effects, tokens,
  regions, and exact mathematical semantics.

Distributed graph IR
  Sharding, placements, mesh axes, replica groups, partition groups, and
  collectives made explicit.

Platform IR
  Target-legal operation families, layouts, memory spaces, fusion groups,
  tiling candidates, library-call candidates, and collective candidates.

Schedule IR
  Ordered commands, streams, events, barriers, memory operations, collectives,
  kernel launches, kernel graphs, and host interactions.

Backend executable
  Opaque target-specific code, library descriptors, command buffers, constants,
  and runtime metadata.
```

The current `PlanInstruction` style is useful for bootstrapping, but it should
not remain the long-term center. A better core instruction has typed payloads:

```zig
InstructionPayload = union(enum) {
  elementwise: ElementwiseSpec,
  dot: DotSpec,
  convolution: ConvolutionSpec,
  gather: GatherSpec,
  scatter: ScatterSpec,
  collective: CollectiveSpec,
  control_flow: ControlFlowSpec,
  custom_call: CustomCallSpec,
  transfer: TransferSpec,
}
```

Invalid combinations should be impossible to represent. Verification should
prove shape, dtype, layout, side-effect, aliasing, and sharding invariants
before lowering continues.

## Compiler Pipeline

The compiler should be a pass pipeline with explicit artifacts between major
stages.

Recommended high-level pipeline:

```text
1. Frontend import
   StableHLO/VHLO/MLIR parse, version checks, dialect legalization.

2. Semantic verification
   Shapes, dtypes, tokens, regions, side effects, custom calls, and strict
   StableHLO semantics.

3. Canonicalization
   Algebraic simplification, CSE, reshape/transpose normalization, tuple and
   control-flow cleanup.

4. Sharding and topology analysis
   Shardy metadata, mesh axes, device assignment, placement constraints.

5. Collective formation
   all_reduce, reduce_scatter, all_gather, all_to_all, collective_permute,
   send/recv, channel IDs, replica groups, partition groups.

6. Target legalization
   Decide which regions can become generated kernels, kernel graphs,
   collective commands, DMA commands, or unsupported diagnostics.

7. Layout assignment
   Choose layouts for compute units, memory spaces, library compatibility,
   collective compatibility, and transfer efficiency.

8. Fusion planning
   Build fusion groups using target cost models, register/local-memory pressure,
   recomputation tradeoffs, and collective boundaries.

9. Tiling and partitioning
   Choose tile shapes for matrix units, vector lanes, SRAM/scratchpad capacity,
   DMA chunking, and collective chunking.

10. Memory space assignment
    Place values in HBM, SRAM, scratchpad, host memory, or remote memory.
    Insert prefetches, evictions, sliced copies, and recomputation.

11. Buffer assignment
    Compute lifetimes, aliasing, donation, heap layout, workspaces, constants,
    and peak memory.

12. Collective lowering
    Select algorithms, rings, trees, routes, chunk sizes, quantization, and
    async start/done intervals.

13. Scheduling
    Assign commands to compute, DMA, and collective streams. Hide latency by
    overlapping independent compute with communication and transfers.

14. Code generation
    Emit target kernels, library descriptors, command descriptors, constants,
    and metadata.

15. Verification and explanation
    Recheck invariants, record decisions, emit predicted bottlenecks, and
    produce an explainable executable.
```

Each stage should have correctness checks and performance records.

## Compilation Modes

PjRTx should support both JIT and AOT compilation. They share most compiler
passes, but they optimize for different constraints.

JIT compilation is for dynamic frontend workloads, interactive systems, shape
specialization, profile feedback, and fast iteration. JIT must minimize compile
latency while still producing high-quality code. It should cache aggressively,
reuse prior autotuning results, specialize only where it pays off, and emit
enough explanation to debug performance without overwhelming the hot path.

AOT compilation is for deployment, reproducibility, embedded/runtime-limited
environments, and platforms where compile toolchains are unavailable or too
expensive at runtime. AOT can spend more time on search, autotuning, exhaustive
verification, memory planning, and multi-target packaging.

The compiler should make the mode explicit:

```zig
CompilationMode
  jit
  aot
  hybrid
```

JIT artifacts:

```zig
JitExecutable
  target_fingerprint
  shape_specialization_key
  compile_options_key
  executable_schedule
  loaded_kernels
  runtime_allocations
  profiling_hooks
```

AOT artifacts:

```zig
AotPackage
  stable_program_fingerprint
  target_requirements
  supported_shape_signatures
  executable_schedules
  kernel_blobs
  constant_blobs
  memory_plan
  explain_plan
  correctness_manifest
  autotune_manifest
```

Hybrid mode should allow ahead-of-time packaging of stable kernels and runtime
JIT of shape-specialized schedules, collective plans, or platform-specific
late bindings.

Cache keys must be precise. They should include source program fingerprint,
canonical graph fingerprint, compile options, target description, backend
version, dtype policy, shape specialization, sharding, memory-space policy, and
relevant autotune/profiling inputs.

Compilation correctness differs by mode. JIT must not skip semantic
verification for speed unless it is using a proven trusted artifact. AOT must
record enough target and correctness metadata to reject execution on an
incompatible platform.

## Fusion

Fusion is not just a graph simplification. It is a target-specific scheduling
decision.

A fusion planner should consider:

```zig
FusionCandidate
  producers
  consumers
  bytes_saved
  extra_flops
  register_pressure
  local_memory_pressure
  code_size
  launch_count_reduction
  vectorization_compatibility
  layout_compatibility
  collective_boundary
  numerical_semantics
```

Fusion should not cross boundaries blindly. Some boundaries are natural:
collectives, host callbacks, side-effecting custom calls, large reductions,
layout-changing kernel graphs, and operations that would create unacceptable
register or SRAM pressure.

The explain plan should say:

```text
Fused add.12 -> tanh.13 -> multiply.14 because it removes two HBM reads and
two HBM writes, preserves dtype semantics, and fits estimated register pressure.

Did not fuse dot.20 -> all_reduce.21 because the all_reduce is scheduled
asynchronously and starts before independent compute. Fusion would delay the
collective start.
```

## Tiling And Xtiled Execution

Tiling should be a core compiler concern, especially for TPUs and Neuron-like
NPUs.

Tile plans should describe both logical and physical execution:

```zig
TilePlan
  logical_tile_shape
  physical_tile_shape
  memory_tile_shape
  core_mapping
  matrix_unit_shape
  vector_width
  unroll_factor
  double_buffering
  pipeline_stages
  halo_exchange
  reduction_split
  collective_chunk
```

The tiler must account for:

- matrix unit dimensions
- vector width
- SRAM/scratchpad capacity
- DMA granularity
- alignment
- layout
- inter-core exchange
- collective chunking
- accumulation dtype
- numerically safe reduction order

For matrix-heavy programs, maximizing FLOPs means feeding the matrix units
without starving them on memory or collectives. For bandwidth-bound programs,
the right tile may reduce bytes moved rather than maximize arithmetic density.

## Collectives

Collectives must be first-class in the IR, compiler, backend, runtime, and
explain plan.

Required collective operations:

```zig
CollectiveKind
  all_reduce
  reduce_scatter
  all_gather
  all_to_all
  collective_permute
  collective_broadcast
  send
  recv
  ragged_all_to_all
```

Collective metadata:

```zig
CollectiveSpec
  kind
  channel_id
  group_mode
  participant_groups
  reduction_kind
  source_target_pairs
  split_dimension
  concat_dimension
  scatter_dimension
  layout
  dtype
  async_kind
```

Collective plans:

```zig
CollectivePlan
  algorithm
  route
  chunk_size
  num_chunks
  stream
  start_time
  done_time
  overlap_window
  estimated_latency
  estimated_bandwidth
  temporary_buffers
```

The compiler should optimize collectives with passes such as:

- combine adjacent compatible collectives
- split large all-reduces
- rewrite all-reduce to reduce-scatter plus all-gather when beneficial
- reassociate reduce-scatter/all-reduce patterns
- move collectives across while loops when legal
- quantize collective payloads when mathematically allowed
- turn sync collectives into async start/done pairs
- schedule collectives early to overlap with compute
- linearize collectives only when required by backend/runtime constraints

Correctness constraints are strict. Participant groups, channel IDs, replica
and partition group modes, rendezvous keys, token dependencies, and layout
compatibility must be checked. A collective with the wrong group is not a
performance bug; it is a correctness bug.

## Scheduling And Runtime

The runtime should execute a command graph.

Command kinds:

```zig
CommandKind
  kernel_launch
  library_call
  collective
  dma_copy
  memset
  event_record
  event_wait
  barrier
  host_callback
```

Streams:

```zig
StreamKind
  compute
  dma
  collective
  host
  high_priority_compute
```

The scheduler should support:

- compute/collective overlap
- compute/DMA overlap
- prefetch and eviction
- double buffering
- asynchronous deallocation
- stream priorities
- command dependencies
- event-based timing
- profiling records

This is the PjRTx equivalent of a StreamExecutor-like layer, but it should not
be GPU-only. It must describe NPU runtimes with separate matrix engines, vector
engines, DMA engines, and collective engines.

## Graph Execution

The executable should be a graph of commands with explicit data, control,
memory, and stream dependencies. It should not be a linear list unless a target
requires linearization.

Graph execution concepts:

```zig
ExecutableGraph
  commands
  values
  buffers
  streams
  events
  dependencies
  memory_plan
  profiling_plan
```

Command dependencies should distinguish:

```zig
DependencyKind
  data
  token
  stream_order
  memory_availability
  aliasing
  collective_rendezvous
  host_callback
```

The graph executor should support:

- multiple streams per device
- multiple devices per executable
- cross-device collective rendezvous
- cross-host rendezvous where the backend supports it
- graph capture or command-buffer replay when the target runtime supports it
- dynamic shape metadata where explicitly allowed
- partial execution boundaries for debugging and profiling
- cancellation and async error propagation

Graph execution should preserve explainability. Runtime should be able to map a
kernel launch, DMA copy, collective, allocation, or wait event back to the
compiler decision that created it.

The runtime should also support graph-level validation before launch:

- all buffers are live and on the expected device or memory space
- donated buffers are not reused
- required constants are resident
- streams and events are created
- collective participants are ready
- memory aliases match the allocation plan
- executable target requirements match the actual device

## Host And Device IO

Host/device IO must be an explicit part of the architecture. It is often the
real bottleneck, and it is also a correctness boundary.

IO paths:

```zig
IoPath
  host_to_device
  device_to_host
  device_to_device
  host_to_host
  remote_device_to_device
  file_to_host
  file_to_device
```

Host memory kinds:

```zig
HostMemoryKind
  unpinned
  pinned
  mapped
  shared
  external
```

Device IO should be planned:

```zig
TransferPlan
  source
  destination
  byte_size
  dtype
  shape
  layout
  memory_space
  staging_space
  stream
  start_event
  done_event
  chunking
  alignment
```

The compiler/runtime should support:

- asynchronous host-to-device and device-to-host copies
- pinned host staging buffers
- direct file-to-device or memory-mapped loading where possible
- chunked transfers for large weights and activations
- overlapped copy and compute
- layout conversion during transfer when profitable and correct
- device-to-device copies across local interconnects
- remote transfers across hosts when the backend supports them
- explicit ownership for external buffers

Weights deserve special treatment. Large model weights should not be treated
like incidental host arrays. PjRTx should support resident constants, lazy
loading, sharded loading, prefetch, host staging, memory pressure decisions, and
explainable placement.

IO correctness requirements:

- shape, dtype, and layout must match the receiving value
- byte sizes must be validated with overflow checks
- host buffers must remain alive until asynchronous copies complete
- external buffers need explicit ownership and lifetime contracts
- endian and dtype packing rules must be specified
- device-to-host reads must wait for producing commands
- IO errors must propagate through PJRT events and runtime diagnostics

IO performance records should include bytes transferred, source and destination
memory spaces, stream, chunk size, overlap with compute, and effective
bandwidth.

## Backend Contract

The backend interface should shrink at the operation level and grow at the
target level.

Instead of exposing one method per operation, a backend should provide:

```zig
Backend
  describeTarget()
  validateTargetProgram(...)
  legalize(...)
  estimate(...)
  compileKernel(...)
  compileLibraryCall(...)
  compileCollective(...)
  buildExecutable(...)
  createAllocator(...)
  createStreams(...)
  execute(...)
  profile(...)
```

Backend responsibilities:

- expose hardware capabilities
- expose memory spaces and transfer edges
- expose communication capabilities
- expose supported codegen routes
- compile target kernels
- bind runtime resources
- execute command schedules
- report profiling data

Compiler responsibilities:

- choose operation decomposition
- choose fusion
- choose tiling
- choose layouts
- choose memory spaces
- choose collective algorithms
- choose schedule
- prove correctness
- explain decisions

This separation lets Metal, CUDA, ROCm, TPU, Neuron, and future NPU
backends differ naturally without forcing them all through the same per-op
vtable.

## Backend Kernel Generation

Backend-specific lowering is the bridge between the abstract schedule and real
performance. Kernel generation is not only about emitting code. It is about
preserving the trace from program semantics to hardware execution.

Backend-specific lowering should make these questions answerable:

- Which graph instructions became generated kernels?
- Which graph instructions became kernel graphs?
- Which graph instructions became collectives?
- Which graph instructions became DMA or transfer commands?
- Which backend could not lower a region, and why?
- Which hardware unit is expected to execute each command?
- Which dtype and accumulation policy is used?
- How many logical FLOPs and bytes are attached to each command?
- Which runtime profile events correspond to each generated or selected kernel?

The compiler should not lower one StableHLO operation at a time unless that is
the right unit. It should lower regions.

Region kinds:

```zig
LoweringRegionKind
  elementwise_fusion
  reduction_fusion
  matmul
  convolution
  attention
  transpose_or_layout
  collective
  transfer
  custom_call
  control_flow
```

Lowering results:

```zig
LoweringResult
  unsupported
  generated_kernel
  selected_library_call
  collective_command
  dma_command
  view_or_alias
  composite_sequence
```

Unsupported is a first-class result. It should include the blocking feature,
source instructions, dtype, shape, layout, sharding, and target capability that
was missing.

Generated kernels should be used when the backend can emit target-specific code
that is expected to beat or complement kernel graphs.

```zig
KernelIr
  source_instruction_ids
  input_values
  output_values
  dtype_policy
  layout
  tile_plan
  memory_plan
  math_semantics
  target_unit
```

Kernel compilation should produce:

```zig
CompiledKernel
  kernel_id
  backend_kind
  target_fingerprint
  code_object
  entry_point
  launch_or_dispatch_dimensions
  argument_layout
  shared_or_local_memory_bytes
  register_estimate
  hardware_binding
  cost_ledger_ids
  explanation_record_ids
```

Generated kernels must be deterministic under the selected correctness policy.
If a generated kernel uses approximate math, reassociated reductions, reduced
precision, stochastic rounding, or target-specific undefined behavior, the
compile options and explain records must say so.

Library calls are not a fallback. On many targets, they are the best lowering.
Matmul, convolution, FFT, triangular solve, attention, and collectives may be
library-backed.

```zig
LibraryCall
  library
  symbol_or_algorithm
  source_instruction_ids
  dtype
  accumulation_dtype
  input_layouts
  output_layouts
  workspace_bytes
  deterministic
  math_mode
  cost_estimate
```

The compiler should explain why a kernel graph was chosen:

```text
dot_general.4 lowered to metal_mls_matmul_kernel because V0 treats rank-2 matmul as a
library boundary and the input layouts are compatible.
broadcast.5/add.6/tanh.7 lowered to metal_mls_elementwise_fusion_kernel because
the producer-consumer chain has compatible dtype, layout, and placement, and no
collective or memory-space boundary cuts the region.
```

Some operations should lower into command sequences:

- all-reduce -> reduce-scatter + all-gather
- attention -> matmul + softmax fusion + matmul
- dynamic update -> copy + scatter kernel
- layout conversion -> generated transpose kernel + alias/view
- host input -> pinned staging + DMA copy

Composite lowering should preserve provenance:

```zig
CompositeLowering
  source_instruction_ids
  generated_commands
  reason
  correctness_constraints
  cost_before
  cost_after
```

If a source operation becomes multiple commands, every command must point back
to the original source operation.

Backend capability queries should be structured:

```zig
BackendCapabilities
  target
  codegen
  libraries
  collectives
  memory
  runtime
```

```zig
CodegenCapabilities
  supports_generated_kernels
  supported_kernel_languages
  supports_runtime_compilation
  supports_aot_compilation
  supports_specialization
  supports_dynamic_shapes
  supports_vectorization
  supports_shared_memory
  supports_tensor_memory_accelerator
```

```zig
RuntimeCapabilities
  streams
  events
  command_buffers
  graph_capture
  async_alloc
  async_copy
  host_callbacks
  hardware_counters
```

Capabilities should be exact enough to avoid pretending support exists. Unknown
is better than optimistic.

Metal is the bootstrap backend. It should not define the whole PjRTx
architecture.

V0 Metal behavior:

```text
StableHLO subset
  -> typed graph
  -> cost/explain records
  -> Metal/MLS backend executable plan
  -> backend binding records around Metal execution
```

Near-term Metal strategy:

- generate Metal/MLS graph and kernel plans for supported tensor graph
  fragments
- treat the generated Metal/MLS graph as the backend executable boundary
- expand verified backend bindings into explicit Metal call sequences before
  runtime submission
- attach PjRTx graph/source IDs to Metal backend commands
- record conservative hardware binding as `metal_shader_core` or `unknown_gpu`
- measure H2D, Metal execute, and D2H timing
- avoid lowering through library fallback paths

Future Metal strategy:

- deepen generated Metal/MLS kernels for elementwise, reduction, layout, and
  fusion fragments
- choose between generated kernel alternatives with target cost models
- use Metal command buffers and events explicitly
- model unified memory carefully instead of assuming IO is free
- keep Metal as one backend mechanism, not the backend interface

If Metal does not expose enough control to validate tiling, memory-space
assignment, or hardware binding, PjRTx should use a NPU backend or
lower-level experimental backend to test those architecture layers.

A NPU backend is valuable. It can prove architecture before real hardware
integration.

NPU should model:

- matrix unit with dtype-specific rates
- vector unit with dtype-specific rates
- local SRAM
- HBM
- DMA engine
- reserved collective engine facts
- alignment constraints
- tile-size constraints
- synthetic profile events

It does not need to execute fast. It needs to validate target descriptions,
legality checks, tile selection, memory-space assignment, schedule commands,
performance ledgers, explain records, and expected-versus-observed synthetic
profile matching.

Long-term kernel generation should follow a clear pipeline:

```text
LoweringRegion
  -> Kernel IR
  -> target legalization
  -> layout and tile binding
  -> memory access planning
  -> vectorization or matrix-unit mapping
  -> code emission
  -> backend compiler invocation
  -> code object
  -> runtime binding
```

Kernel IR should not be tied to one backend language. Possible lowerings:

- Metal Shading Language
- LLVM IR
- MLIR dialects
- PTX
- AMDGCN
- Triton-like IR
- Neuron-specific IR
- TPU/custom NPU command or HLO dialects

Generated and selected kernels must preserve semantics:

- input and output shape compatibility
- dtype compatibility
- layout compatibility
- memory aliasing correctness
- bounds safety for generated indexing
- reduction identity correctness
- accumulation dtype correctness
- NaN/infinity/signed-zero policy
- deterministic or nondeterministic behavior recorded
- approximate math recorded

Every kernel or kernel graph should have a performance ledger entry:

```zig
KernelPerformanceRecord
  kernel_id
  source_instruction_ids
  logical_flops
  hardware_ops
  bytes_read_by_space
  bytes_written_by_space
  expected_time_ns
  observed_time_ns
  expected_unit
  observed_counters
  limiting_resource
```

The first V0 form of this record may be a stable backend-call profile summary:
backend operation, lowering instruction IDs, expected execution unit,
predicted bytes and ops, observed bytes and ops, ideal compute time from target
dtype/op-class rates, memory-space bytes and ideal memory time from first-class
memory traffic records plus target bandwidth, limiting resource, and the
matching profile event.

Hardware utilization reports should print those same memory traffic records,
not only backend call summaries. A reviewer should be able to see, for each
lowering, which bytes hit global device memory and which bytes hit local tile
memory before looking at backend-specific kernel graph details.

They should also print lowering-level roofline rows. The row should be derived
from trace records: lowering cost entries provide predicted ops and ideal
compute time, memory traffic records provide ideal memory time, and the report
marks the limiting resource. This keeps the performance explanation available
even before backend-specific kernel graph construction. When profiling data is
available, the row should include observed bytes/ops and the exact profile event
ID so the estimate and measurement are joined in one stable line.

Kernel performance should be explained in terms of dtype peak, memory
bandwidth, register/local-memory pressure, occupancy, vectorization, tile
efficiency, launch overhead, library algorithm choice, and collective or
transfer overlap.

JIT backends need runtime compiler availability, cache keys, compile latency
reporting, code-object lifetime management, and shape-specialization policy.
AOT backends need serializable code objects, target compatibility checks,
stable kernel argument ABIs, constant packaging, memory-plan packaging, and
explain/correctness manifests. Hybrid backends should allow AOT kernels with
JIT schedule binding or JIT specialization.

V0 should not try to implement the final backend interface. It should add the
minimum records needed for traceability:

```zig
BackendBinding
  command_id
  backend_kind
  backend_operation
  graph_instruction_ids
  expected_unit_id
  cost_ledger_ids
```

```zig
BackendDiagnostic
  backend_kind
  pass_name
  graph_instruction_ids
  feature
  reason
```

```zig
BackendProfileEvent
  command_id
  backend_operation
  duration_ns
  bytes
  logical_ops
```

V0 can attach these records around the existing backend executable path.

Backend and codegen packages for the new architecture live under `//pjrtx/...`.
Use `docs/specs/package_boundaries_v0.md` for the canonical package graph.
Backend code may need C ABI shims for C++/Objective-C/runtime APIs; those shims
should stay private to backend implementation packages. Backend implementation
details must not leak into target, compiler fact, runtime fact, or report
packages.

## Mathematical Correctness

Correctness must be checked at every layer.

Frontend correctness:

- StableHLO version and dialect compatibility
- exact dtype and shape parsing
- region semantics
- token and side-effect semantics
- custom-call contracts
- sharding metadata correctness

Graph correctness:

- every value has a type, shape, layout, placement, and defining operation
- tuple/control-flow semantics are preserved
- dynamic shape constraints are explicit
- aliasing is represented, not inferred late by accident

Lowering correctness:

- dtype conversions are explicit
- bitcast versus numeric conversion is preserved
- reduction identities are correct
- reduction associativity assumptions are explicit
- NaN, infinity, signed zero, and overflow behavior are respected according to
  the chosen semantics
- approximate math is opt-in and recorded

Distributed correctness:

- replica groups are valid
- partition groups are valid
- channel IDs and rendezvous keys match
- collectives have the same layout and dtype where required
- async start/done pairs are balanced
- tokens enforce side-effect order

Memory correctness:

- lifetimes do not overlap unless aliasing is legal
- donated buffers are not reused by the caller
- constants are immutable
- host staging does not become accidental device state
- memory-space copies respect availability times

Runtime correctness:

- stream dependencies are complete
- events are recorded and waited on correctly
- asynchronous errors propagate to PJRT events
- deallocation does not race with execution
- profiling does not change behavior

Correctness tests should include reference comparisons, property tests for
shape/layout transforms, collective group tests, allocator lifetime tests,
schedule dependency tests, and backend conformance tests.

## Performance Correctness

Performance needs its own form of correctness. The compiler should not merely
run fast sometimes; it should know what it is trying to optimize and report
when predictions fail.

Performance records should include:

```zig
PerformanceEstimate
  flops_by_dtype
  bytes_by_memory_space
  bytes_by_transfer_edge
  collective_bytes
  collective_latency
  launch_count
  synchronization_count
  peak_memory_by_space
  predicted_bottleneck
  predicted_time_ns
```

After execution, profiling should produce:

```zig
PerformanceObservation
  actual_kernel_time
  actual_collective_time
  actual_dma_time
  actual_overlap
  actual_memory_peak
  cache_hit_or_miss
  prediction_error
```

The compiler should be able to compare predicted and observed behavior. This
creates a path to autotuning and profile-guided scheduling without making the
initial architecture depend on a giant autotuner.

## Profiling

Profiling should be designed into the runtime and compiler from the start. It
is not just logging elapsed time; it is how PjRTx validates performance models,
finds bottlenecks, improves schedules, and explains behavior.

Profiling layers:

```zig
ProfileLayer
  compile_pass
  graph_schedule
  kernel
  library_call
  collective
  dma_transfer
  allocator
  host_runtime
  end_to_end
```

Compile-time profiling should record:

- pass runtime
- IR size before and after each pass
- number of fusions formed and rejected
- number of collectives transformed
- memory plan peak by memory space
- autotune search cost
- codegen time
- cache hits and misses

Runtime profiling should record:

- command start and end times
- stream occupancy
- kernel duration
- library-call duration
- collective duration
- DMA duration
- host/device transfer bandwidth
- overlap between compute, DMA, and collectives
- allocation and deallocation events
- memory high-water marks
- async wait time
- launch overhead

Profiles should connect back to compiler records:

```zig
ProfileEvent
  command_id
  source_instruction_ids
  lowering_record_id
  backend_call_index
  pass_decision_id
  stream_id
  device_id
  memory_space
  start_time_ns
  duration_ns
  bytes
  flops
  status
```

The profiler should support several modes:

```zig
ProfileMode
  off
  lightweight_counters
  sampled
  full_timeline
  correctness_debug
  autotune
```

Profiling must not invalidate correctness. Instrumentation should preserve
stream dependencies and should not hide races by accidentally synchronizing
too much. When profiling changes scheduling or timing, that fact must be
recorded.

Profile-guided optimization should be incremental. The first goal is to compare
predicted and observed time. Later, PjRTx can use profiles to choose fusion
variants, tile sizes, collective algorithms, prefetch distances, and stream
schedules.

Profiling should also support vertical performance matching:

```zig
ProfileMatch
  source_instruction_ids
  command_id
  kernel_id
  hardware_unit
  logical_flops
  expected_hardware_ops
  observed_hardware_ops
  expected_bytes
  observed_bytes
  expected_time_ns
  observed_time_ns
  bottleneck_prediction
  bottleneck_observation
```

This is how PjRTx can answer whether lowering actually matched the hardware
specification. If a kernel should use a matrix unit but profile data shows low
matrix utilization and high memory stalls, the explain plan should expose that
mismatch directly.

## Explainability

Every pass should update verified MLIR state and expose structured explanation
records extracted from that state:

```zig
ExplainRecord
  pass_name
  entity_id
  decision
  alternatives
  chosen_reason
  correctness_constraints
  estimated_effect
  rejected_reasons
```

Example questions the system should answer:

- Why was this op not fused?
- Why did this tensor stay in HBM instead of SRAM?
- Why did the compiler pick BF16 accumulation here?
- Why did an all-reduce start before this matmul?
- Why was this all-reduce split?
- Why did the compiler choose this tile shape?
- Why is the program memory-bound?
- Which collective blocks scaling?
- Which pass changed numerical behavior, and was it allowed?

Explainability should be available programmatically and as human-readable
diagnostics.

## Autotuning And Feedback

Autotuning should refine choices, not replace architecture.

Autotunable decisions:

- tile shapes
- launch dimensions
- fusion variants
- library algorithms
- collective algorithms
- chunk sizes
- prefetch distances
- double-buffering depth
- stream assignment

Autotune keys should include:

- target fingerprint
- dtype
- shapes
- layouts
- memory spaces
- operation family
- sharding
- collective group size
- compiler version
- backend version

Autotuning results should remain separate from target description. Hardware
description says what the device is. Autotuning says what worked for a specific
program family on that target.

## Suggested Repo Direction

The canonical long-term package graph lives in
`docs/specs/package_boundaries_v0.md`. The canonical final dialect/op/pass
destination lives in
`docs/specs/final_mlir_dialect_op_pass_architecture_v0.md`.

New architecture code should live under Bazel packages rooted at `//pjrtx/...`.
The existing `//src/...` packages can remain as bootstrap and compatibility
code during migration, but new public architecture should establish the
long-term `pjrtx` namespace and must not grow around a central core package.

The implementation should be idiomatic Zig: explicit allocators and ownership,
small structs, tagged unions for typed IR payloads, `deinit` for owned
artifacts, error unions for fallible stages, `std.Io.Reader` for ingestion, and
`std.Io.Writer` for diagnostics and reports. PjRTx should not become a C++
compiler framework translated mechanically into Zig.

Logic should be contained in small Zig structs named after the boundary they
protect, such as `TensorFacts`, `GraphPayloadFacts`, `PassPipelineFacts`, or
`RuntimeAllocationFacts`. Package root files should compose and re-export
meaningful families; implementation should be split across files with names
that reveal ownership rather than across central buckets.

The MLIR compiler layer should be Zig-first but MLIR-native. PjRTx compiler
policy, target legality, cost models, diagnostics, extraction, and external
pass callbacks should be written in Zig whenever the MLIR C API gives enough
access. Real MLIR dialect definitions still need minimal TableGen/C++ plumbing
for operations, types, attributes, parser/printer syntax, verifiers, and
advanced rewrite or dialect-conversion helpers. That C++ layer must stay narrow
and C-compatible at the Zig boundary; it exposes MLIR extension points, not a
second compiler hidden from the PjRTx trace.

## Implementation Discipline

The architecture should be implemented in a way that keeps the compiler
inspectable, testable, and pleasant to change. This is part of the design, not
only code style.

PjRTx should adapt the spirit of
[TigerStyle](https://github.com/tigerbeetle/tigerbeetle/blob/main/docs/TIGER_STYLE.md)
for new `//pjrtx/...` code. The relevant principle is not cosmetic formatting;
style is a design tool for safety, performance, and developer experience, in
that order.

For PjRTx this means:

- safety first: every compiler artifact has validation, invariants, and
  failure tests
- performance from design: every hot path needs a back-of-the-envelope model
  for FLOPs, bytes, bandwidth, latency, allocation, and synchronization
- developer experience as correctness leverage: names, reports, dumps, and
  diagnostics must make the system easier to reason about
- zero hidden debt in core architecture: do not merge temporary execution paths,
  implicit fallback, unbounded queues, or opaque ownership into `//pjrtx/...`
- boundedness by default: loops, queues, caches, report sections, and compile
  worklists should have explicit limits or documented proofs of termination
- assertions document invariants: use assertions for programmer errors and
  diagnostics/error unions for expected user/program/target failures
- explicit control flow: prefer straightforward pass orchestration over clever
  callback chains when correctness or provenance is at stake
- explicit options: call sites must pass semantic options intentionally instead
  of relying on library defaults for math, layout, allocation, profiling, or
  target behavior
- standard library first: before creating a local helper, wrapper, or miniature
  framework, check whether Zig `std` already provides the right primitive for
  IO, allocation, formatting, sorting, hashing, bounds-checked slices, parsing,
  testing, or data structures
- small boundary structs: put domain operations behind small Zig structs with
  clear ownership instead of loose utility functions; file-local helpers are
  acceptable only when they serve one boundary and do not obscure intent
- meaningful files: split code by fact family, pass family, backend contract,
  or runtime contract; root package files should be facades, not dumping
  grounds for unrelated implementation
- document intent: comments and docs should explain why the code exists, what
  invariant it protects, or what compiler/runtime contract it encodes; avoid
  comments that merely restate a line of code
- refactor continuously: when a change exposes the wrong abstraction, duplicated
  policy, unclear ownership, or a confusing API, fix the structure as part of
  the work instead of layering more code around it

PjRTx does not inherit TigerBeetle's storage-engine constraints literally. For
example, compiler construction may allocate dynamically, but ownership and
lifetimes must still be explicit, bounded where practical, and visible at
artifact boundaries.

### Zig IO

Use `std.Io.Reader` and `std.Io.Writer` heavily. Avoid designing APIs that take
only owned byte slices when the data could be streamed.

Preferred API shape:

```zig
pub fn importStablehloGraph(
    allocator: std.mem.Allocator,
    format: []const u8,
    reader: *std.Io.Reader,
    diagnostics: *std.Io.Writer,
) ImportError!GraphModule;

pub fn writeGraphDump(
    graph: *const GraphModule,
    writer: *std.Io.Writer,
) std.Io.Writer.Error!void;

pub fn writeCostLedger(
    ledger: *const CostLedger,
    writer: *std.Io.Writer,
) std.Io.Writer.Error!void;

pub fn writeVerticalSliceReport(
    report: *const VerticalSliceReport,
    writer: *std.Io.Writer,
) std.Io.Writer.Error!void;
```

Rules:

- program ingestion uses `std.Io.Reader`
- compile options parsing uses `std.Io.Reader`
- diagnostics use `std.Io.Writer`
- graph dumps use `std.Io.Writer`
- cost/profiling/explain reports use `std.Io.Writer`
- tests may use `std.Io.Reader.fixed` and `std.Io.Writer.Allocating`
- large artifacts should not require copying into temporary strings unless the
  MLIR C API boundary requires it

This keeps the architecture ready for large StableHLO modules, model weight
metadata, external report streams, file IO, network IO, and future direct
device IO.

### Ownership

All compiler/runtime artifacts should own their memory explicitly or clearly
borrow it. Owned artifacts expose `deinit`.

Expected pattern:

```zig
pub const GraphModule = struct {
    allocator: std.mem.Allocator,
    values: []GraphValue,
    instructions: []GraphInstruction,

    pub fn deinit(self: *GraphModule) void {
        // free nested owned slices, then top-level slices
    }
};
```

Avoid hidden global mutable state except for unavoidable one-time MLIR/Shardy
registration guarded by a mutex. Do not put backend state into compiler
artifacts. Do not let report records own runtime buffers.

### Idiomatic Zig

The implementation should be idiomatic Zig, not C++ architecture transliterated
into Zig.

Guidelines:

- prefer plain structs, tagged unions, slices, and explicit ownership
- make invalid states unrepresentable with typed payload unions where practical
- pass allocators explicitly
- expose `init`/`deinit` or value constructors with clear ownership
- use error unions for recoverable failures
- use `std.Io.Reader` and `std.Io.Writer` for ingestion, diagnostics, dumps,
  and reports
- use explicit-width integers for IDs, counts, byte sizes, and serialized
  formats; use `usize` only for local indexing/slice interaction
- leverage Zig `std` heavily instead of creating project-local functions for
  common mechanics; local helpers should encode PjRTx domain meaning, not
  duplicate `std` behavior
- contain domain behavior in small structs that name the protected boundary;
  for example, prefer `TensorFacts.validate` over a loose
  `validateTensorType` package function
- split implementation across meaningful files before a module becomes a mixed
  bag of IDs, records, validators, writers, and orchestration
- prefer typed initialization at the binding site, for example
  `const block: Block = try .init(allocator);` instead of
  `const block = try Block.init(allocator);`
- in tests and acceptance checks, prefer typed expected values or helper values
  over inline `@as(T, value)` casts; casts should not become test noise
- keep functions short enough to review with one mental model; split large
  functions by responsibility, not by arbitrary helper extraction
- avoid compound assertions and compound boolean gates when separate checks
  produce clearer diagnostics
- prefer positive invariants and explicit negative-case handling
- place units at the end of names, for example `duration_ns` and
  `bandwidth_bytes_per_second`
- avoid abbreviations except for domain-standard terms such as HLO, MLIR, PJRT,
  DMA, HBM, SRAM, and NPU
- avoid hidden heap allocation in helpers unless the function contract says it
  returns owned memory
- avoid global registries except for backend/plugin registration or MLIR pass
  one-time initialization
- keep vtables small and capability-oriented
- avoid class-like inheritance patterns
- use `comptime` only where it simplifies real type-level structure
- write small testable functions instead of large object graphs

PjRTx should feel like Zig code that happens to build a compiler, not like a
miniature C++ compiler framework forced into Zig syntax.

### Diagnostics

Every fallible compiler stage should write structured human-readable
diagnostics before returning an error.

Diagnostic fields should include:

```text
pass
source op
instruction id
value id
dtype
rank
shape
sharding or placement
target feature
reason
```

Example:

```text
unsupported op: pass=vertical-slice-import op=stablehlo.convolution
feature=v0-stablehlo-subset reason="convolution is outside V0"
```

Diagnostics should be produced with `std.Io.Writer`, not assembled only as
returned strings. PJRT error messages can still duplicate the final buffered
diagnostic when crossing the C API boundary.

### MLIR, StableHLO, And Shardy

PjRTx should rely on the Bazel-provided MLIR, StableHLO, and Shardy C API
targets. Do not invent text parsers for StableHLO beyond tiny test fixtures.

Expected path:

```text
StableHLO/VHLO bytes
  -> MLIR C API parse or StableHLO portable artifact deserialize
  -> verifier
  -> optional Shardy-aware canonicalization
  -> operation walk
  -> typed graph
```

Boundary rules:

- `//pjrtx/compiler` may depend on `//pjrtx/target`,
  `//pjrtx/compiler/facts`, `//pjrtx/mlir`, `//pjrtx/dialects`, and
  MLIR/StableHLO/Shardy C API targets
- `//pjrtx/compiler` may depend on narrow `//pjrtx/compiler/mlir` C-compatible
  shim targets for MLIR extension points that are not exposed by the public C
  API
- real PjRTx dialect definitions live under `//pjrtx/dialects/...` and use
  TableGen/C++ only for MLIR-native dialect plumbing
- `//pjrtx/compiler` must not depend on `//pjrtx/runtime`
- `//pjrtx/compiler` must not depend on `//pjrtx/backend`
- `//pjrtx/runtime` must not import Metal C symbols
- no new package may depend on `//pjrtx/core`
- `//pjrtx/plugin` must stay a PJRT adapter

Pass rules:

- prefer Zig external MLIR passes through the MLIR C API for PjRTx policy
  passes
- use MLIR-native C++/TableGen passes or helpers only when dialect conversion,
  generated op adaptors, parser/printer integration, or pattern-rewrite
  infrastructure requires it
- every pass, regardless of implementation language, must have a stable name,
  declared state transition, diagnostics, verifier boundary, and extracted
  explanation record when it affects the executable

### C API Boundary

Do not leak new architecture internals into PJRT structs. The plugin may expose
reports through debug logging, test hooks, or internal helpers, but PJRT API
behavior should stay compatible.

At the PJRT boundary:

- convert PJRT inputs into compiler readers/options
- call compiler/runtime APIs
- translate errors into `PJRT_Error`
- expose normal PJRT executable/buffer behavior

Everything else belongs below the adapter.

## Harness And Iteration Loop

PjRTx needs a fast, debuggable harness from the beginning. The goal is a loop
where a compiler/runtime change can be understood in minutes:

```text
edit
  -> focused Bazel test
  -> stable report diff
  -> optional local JAX execution
  -> profile/explain inspection
  -> broader regression test
```

The harness should make architectural regressions obvious: lost provenance,
wrong FLOP counts, missing diagnostics, unlowerable programs reaching runtime,
unstable reports, or performance events detached from source instructions.

### Test Targets

Suggested target families:

```text
//pjrtx/target:unit_tests
//pjrtx/mlir:unit_tests
//pjrtx/dialects:unit_tests
//pjrtx/compiler/facts:unit_tests
//pjrtx/compiler:unit_tests
//pjrtx/runtime:unit_tests
//pjrtx/backend:unit_tests
//pjrtx/report:unit_tests
//pjrtx/tests/vertical_slice:report_test
//pjrtx/tests/vertical_slice:execution_test
//pjrtx/tests/architecture:boundary_test
//pjrtx/tests/performance:smoke_test
```

Unit tests should be tiny and fast. Report tests should normalize or redact
timing. Execution tests may be slower and should be fewer. Performance smoke
tests should detect shape of behavior, not enforce brittle timing thresholds.

### Fixtures

Keep fixtures small and explicit.

Recommended early fixtures:

```text
pjrtx/tests/fixtures/tanh_dot_bias.mlir
pjrtx/tests/fixtures/elementwise_chain.mlir
pjrtx/tests/fixtures/unsupported_convolution.mlir
pjrtx/tests/fixtures/bad_dot_shape.mlir
pjrtx/tests/fixtures/shardy_tiny_mesh.mlir
```

Each fixture should have a clear purpose: import, diagnostics, cost ledger,
schedule, explain trace, correctness, or unsupported feature behavior.

### Stable Reports

Report rendering must be deterministic. Golden tests need stable output.

Rules:

- sort records by stable ID
- avoid pointer addresses
- redact timing fields in golden comparisons
- include shapes, dtypes, op names, command kinds, and ledger formulas
- keep unknown target fields as `unknown`, not omitted
- use explicit IDs for graph instructions, schedule commands, profile events,
  cost entries, memory traffic records, and explain records

The first report can be text. JSON can come later once the schema stabilizes.

### Debug Modes

Debugging should not require a debugger first. The harness should support
environment flags or compile options for:

```text
PJRTX_TRACE=1
PJRTX_DUMP_GRAPH=1
PJRTX_DUMP_COST=1
PJRTX_DUMP_SCHEDULE=1
PJRTX_DUMP_PROFILE=1
PJRTX_DUMP_EXPLAIN=1
PJRTX_REQUIRE_BACKEND_EXECUTABLE=1
PJRTX_ASSERT_NO_RUNTIME_FALLBACK=1
```

These names can change, but the capabilities should exist.

### Fast Loop

The fast loop should have layers:

```text
1. Pure core tests
   No MLIR, no backend, no JAX. Validate IDs, ownership, formulas, reports.

2. Compiler import tests
   MLIR/StableHLO/Shardy C API, no runtime execution.

3. Schedule/report tests
   Build graph, cost ledger, schedule, explain records, golden report.

4. Backend binding tests
   Use Metal or NPU backend to validate backend records.

5. PJRT/JAX smoke tests
   Validate frontend integration end to end.
```

Most development should hit levels 1-3. Levels 4-5 should be available but not
required for every small edit.

### Profiling In The Harness

The harness should record lightweight profile events even in tests:

- compile/import time
- H2D time and bytes
- backend execute time
- D2H time and bytes
- total end-to-end time

Golden tests should redact timings but preserve event presence, command IDs,
bytes, and logical ops. Manual performance runs can keep real timings.

Do not introduce hidden synchronization solely for nicer profile data unless
the profile mode says so. If a measurement forces synchronization, record that
in the profile event.

### Architecture Boundary Tests

Boundary tests should grow with the new package tree:

- `//pjrtx/compiler` cannot import runtime/backend/plugin
- `//pjrtx/target` cannot import compiler/runtime/backend/plugin/report
- no package can import `//pjrtx/core` after the migration is complete
- `//pjrtx/runtime` cannot mention Metal C symbols
- `//pjrtx/plugin` cannot mention Metal backend shim symbols
- `//pjrtx/backend` implementations cannot leak into target/compiler/runtime
  fact packages
- vertical-slice graph/cost/schedule modules cannot import compiler, runtime,
  backend, or plugin

String-based tests are acceptable as smoke tests, but Bazel visibility and
dependency structure should do most of the enforcement.

### Failure Quality

The harness should test failures, not only successful execution.

Failure tests should assert:

- unsupported ops fail during compile
- diagnostics include pass, op, shape/dtype, and feature
- invalid shapes fail before backend execution
- missing backend capability fails with an explanation
- unlowerable programs never reach runtime execution
- malformed reports are not emitted as successful artifacts

This is how PjRTx stays debuggable while it becomes more ambitious.

## Immediate Refactor Path

The first refactors should create room for the architecture without trying to
build the whole compiler at once.

1. Split the current executable plan into a typed graph IR and a schedule IR.
2. Add an MLIR/StableHLO pass-pipeline artifact with stable diagnostics and
   provenance-preservation checks.
3. Add fusion, layout, tiling, memory-space, and collective lowering records
   before adding deeper backend submission.
4. Move production execution away from runtime instruction interpretation.
5. Add target description structs with dtype rates, memory spaces, and transfer
   edges.
6. Add first-class collective specs even before all collectives are supported.
7. Move fingerprinting, memory-kind policy, and compile orchestration out of
   the PJRT adapter.
8. Add structured diagnostics and explanation records to compiler passes.
9. Add a simple performance estimate record for every compiled executable.
10. Add buffer lifetime and allocation planning before backend execution.
11. Make Metal a backend that consumes a schedule, not a backend that
   defines the compiler architecture.
12. Add explicit host/device IO plans for transfers, staging, and residency.
13. Add a graph executor that runs command schedules instead of operation
    switches.
14. Add JIT/AOT artifact boundaries and cache keys.
15. Add profiling events that map runtime observations back to compiler
    decisions.
16. Add vertical provenance records from source program to generated or selected
    kernels.
17. Add a first performance ledger for FLOPs, bytes, dtype rates, and hardware
    unit binding.
18. Keep reference execution as a test oracle only; the production path has no
    fallback.

## Critique Of This Document

This document is intentionally ambitious. That is useful for setting direction,
but it also creates risks.

The biggest weakness is scope. The document names almost every serious problem
in accelerator compilation: graph IR, sharding, collectives, memory spaces,
buffer assignment, fusion, tiling, scheduling, codegen, profiling, AOT/JIT, and
explainability. A project cannot implement all of that at once. The immediate
plan must select a thin vertical slice that exercises the whole architecture:
one target description, one graph IR path, one memory plan, one schedule IR,
one profiling path, and one explanation path.

The second weakness is that many structs are aspirational rather than designed.
They are useful vocabulary, but they are not yet APIs. Before implementation,
PjRTx should turn the most important records into concrete ownership models,
serialization formats, and tests. Otherwise the architecture may look composed
on paper but still degrade into side channels in code.

The third weakness is backend realism. Metal may not expose enough low-level
control to validate the full vision. That is acceptable for bootstrap, but the
architecture should avoid mistaking Metal's abstraction boundary for PjRTx's
long-term abstraction boundary. A lower-level experimental backend, even a synthetic
NPU simulator, may be needed to test memory-space assignment, scheduling,
hardware binding, and vertical traceability.

The fourth weakness is mathematical correctness under optimization. The document
says correctness is first-class, but correctness policy needs more precision:
strict versus relaxed math, reduction reproducibility, mixed-precision
accumulation, stochastic rounding, quantized collectives, approximations, and
backend library nondeterminism. These should become explicit compile options
and test matrices.

The fifth weakness is explainability cost. Preserving provenance through every
pass can slow compilation and complicate IR rewrites. PjRTx should design
explanation levels: minimal production metadata, debug-level provenance, and
full research/profiling traces. Otherwise explainability could become too heavy
for JIT.

The sixth weakness is profiling trust. Profiling can perturb execution,
especially around async streams and collectives. The document mentions this,
but the design must make profiling modes explicit and record when measurement
changes scheduling. Otherwise profiling data may be precise-looking but
misleading.

The seventh weakness is Verilog/RTL ambition. Mapping PJRT-level computation to
hardware specs is powerful, but direct RTL understanding is a long-term goal.
The practical first step is a structured target description that can be derived
from hardware docs, runtime queries, or handwritten specs. Later, PjRTx can add
importers for more formal hardware descriptions.

The eighth weakness is no prioritization between compiler purity and product
usefulness. A perfect architecture that cannot run real JAX programs is not
useful. The roadmap should alternate between vertical infrastructure and real
workloads: elementwise chains, matmul-heavy blocks, attention-like graphs,
collective-heavy distributed graphs, and weight-loading-heavy inference.

The hard requirement after this critique is simple: every next implementation
step should prove one architectural claim with code. If a change does not make
performance, correctness, explainability, or composability more real, it should
wait.

## North Star

PjRTx should be able to say:

```text
This program is bottlenecked by reduce-scatter latency on mesh axis data.
The compiler split the matmul reduction into four chunks, started
reduce-scatter after chunk 1, overlapped chunks 2-4 with communication,
kept activations in local SRAM for 17 schedule steps, evicted one tensor to HBM
because SRAM pressure exceeded 92%, and selected BF16 matrix units with F32
accumulation because the compile options required strict enough numerical
behavior for this reduction.
```

That is the standard: not just executing StableHLO through PJRT, but producing
fast, correct, composable, explainable executables for real accelerator
systems.
