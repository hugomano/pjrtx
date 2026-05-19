# XLA Coverage Map V0

This spec records what the local XLA tree already does, and what PjRTx should
learn from it.

XLA proves that a serious accelerator stack needs a broad compiler and runtime:
StableHLO/HLO import, many HLO passes, sharding, collectives, fusion, tiling,
layout assignment, memory planning, scheduling, backend codegen, runtime
submission, host/device transfer, allocation, profiling, and PJRT integration.

PjRTx should match that breadth over time. It should not inherit the opacity.
Every important PjRTx decision must be represented as a typed artifact,
validated, printable through `std.Io.Writer`, and joinable from source
operations to generated or selected kernels and profile events.

## Local XLA Reference Points

The local XLA checkout at `/Users/hugo/Developer/xla` has these major pieces:

| Concern | XLA location | What XLA does |
| --- | --- | --- |
| StableHLO/HLO import | `xla/hlo/translate/stablehlo_to_hlo`, `xla/mlir_hlo` | Legalizes StableHLO/MHLO into XLA HLO and related MLIR forms. |
| Compiler orchestration | `xla/service/gpu/gpu_compiler.cc`, `xla/service/cpu/cpu_compiler.cc` | Builds large backend-specific HLO pass pipelines and invokes backend lowering. |
| Simplification | `xla/hlo/transforms/simplifiers` | Algebraic simplification, broadcast canonicalization, reshape movement, DCE, constant folding, memory scheduling, rematerialization, and domain-specific cleanup. |
| Decomposition | `xla/hlo/transforms/expanders` | Lowers complex HLOs into simpler operations before backend codegen. |
| Fusion | `xla/service/*fusion*`, `xla/service/gpu`, `xla/backends/gpu/codegen` | Forms fusion regions, GEMM epilogues, reductions, Triton fusions, and backend-specific emitter regions. |
| Tiling and Triton/XTile | `xla/backends/gpu/codegen/triton` | Lowers selected GPU regions through tiled IR and Triton-oriented codegen passes. |
| Collectives | `xla/hlo/transforms/collectives`, `xla/backends/*/collectives` | Combines, rewrites, pipelines, decomposes, schedules, and executes all-reduce/all-gather/reduce-scatter/all-to-all/permutation-style communication. |
| Sharding and SPMD | `xla/service/spmd`, `xla/service/spmd/shardy` | Propagates and lowers sharding, partitions programs, and integrates Shardy. |
| Layout | `xla/service/*layout*`, backend compiler pipelines | Assigns physical layouts and runs layout-sensitive normalization. |
| Memory spaces | `xla/service/memory_space_assignment`, `xla/service/heap_simulator` | Models alternate memory, heap pressure, prefetching, eviction, and placement. |
| Buffer assignment | `xla/service/buffer_assignment.*` | Computes buffer lifetimes, aliases, donations, workspaces, and compiled memory stats. |
| Scheduling | HLO scheduling passes, GPU compiler scheduling stages, collective schedule linearization | Orders compute, transfer, and communication while respecting dependencies and backend constraints. |
| Runtime execution | `xla/pjrt`, `xla/stream_executor` | Exposes PJRT clients/executables/buffers and device streams, events, kernels, memory, allocators, command buffers, and transfers. |
| Backend codegen | `xla/backends/gpu/codegen`, `xla/backends/cpu/codegen` | Emits or selects kernels, library calls, and backend command descriptors. |
| Autotuning/profiling | `xla/backends/*/autotuner`, backend profilers | Searches algorithms and compares runtime behavior for selected backends. |

This table is not a dependency list. It is a coverage map: each row names a
compiler/runtime responsibility PjRTx must eventually own or intentionally
delegate with an explicit, explainable contract.

## Main Lesson

XLA's breadth is correct. Its explainability is not enough for PjRTx.

In XLA, a performance result can depend on many places at once: HLO
simplification, SPMD partitioning, collective rewriting, layout assignment,
fusion, backend-specific normalization, autotuning, buffer assignment,
scheduling, StreamExecutor behavior, allocator choices, and runtime profiling.
Those decisions can be inspected through dumps, debug options, proto records,
and backend logs, but they are not presented as one causal product surface.

PjRTx should make the causal chain first-class:

```text
PJRT-visible program
  -> StableHLO / Shardy metadata
  -> typed graph
  -> pass records
  -> fusion, layout, tile, collective, and memory-space plans
  -> cost and roofline records from target hardware facts
  -> lowering regions
  -> schedule, allocation, stream, and event records
  -> backend bindings
  -> generated kernels, library calls, kernel graphs, DMA, or collectives
  -> runtime profile events and hardware counters
```

The user should be able to ask why a kernel exists, which source operations it
covers, what hardware unit it should use, what dtype rate applies, which bytes
move through which memory spaces, what collective route was selected, what
profile event measured it, and which pass made each decision.

## PjRTx Design Consequences

PjRTx should not implement only a StreamExecutor-like layer under an invisible
compiler. That architecture can execute programs, but it cannot explain or own
performance.

PjRTx should not copy XLA's pass count mechanically. It should copy the
responsibilities, then expose them with cleaner ownership:

- frontend import and semantic verification
- MLIR/StableHLO/Shardy pass reporting
- typed graph import
- algebraic and shape canonicalization
- target legality
- sharding and collective planning
- fusion planning
- layout, tile, and memory-space planning
- cost and roofline planning
- lowering-region formation
- buffer assignment and allocator planning
- schedule, stream, event, and overlap planning
- backend binding
- kernel graph construction
- generated-kernel or library-call selection
- collective runtime binding
- profiling and autotuning feedback

Every one of those responsibilities should produce stable records before a
later layer consumes the decision. Backend code may specialize, but it must not
rediscover frontend semantics from strings or silently choose a fallback path.

## Required PjRTx Trace Properties

The PjRTx trace must make these joins possible:

- source operation to graph instruction
- graph instruction to compiler pass decision
- compiler pass decision to accepted or rejected alternatives
- fusion group to lowering region
- collective spec to selected or rejected collective algorithm
- placement record to memory space and transfer edge
- tile record to hardware unit, local memory, and dtype rate
- cost record to target peak, bandwidth, and predicted bottleneck
- lowering region to schedule command
- schedule command to allocation, stream, event, and dependency records
- backend binding to generated kernel, kernel graph, library call, DMA, or
  collective command
- backend command to profile event and hardware counter when available

When a join is impossible, compilation should fail for executable artifacts or
the report validator should reject the trace for diagnostic artifacts.

## Anti-Goals

- Do not use XLA as a hidden compiler while PjRTx only implements runtime
  submission.
- Do not treat library calls as fallback. They are valid only when selected as
  explicit backend bindings.
- Do not let backend-specific codegen bypass target legality, correctness
  policy, Shardy metadata, collective semantics, or source provenance.
- Do not optimize for peak FLOPs alone. Use dtype rates, memory spaces,
  bandwidths, interconnects, collective engines, DMA engines, launch overhead,
  synchronization, allocation pressure, and measured profile data.
- Do not accept a fast kernel that cannot be explained from source operation to
  hardware unit.

## V0 Application

V0 is intentionally smaller than XLA, but it must grow in the same direction:

- keep the pass catalog explicit in `//next/pjrtx/compiler`
- treat `broadcast_simplify` and `reshape_transpose_fold` as the first tiny
  examples of a much broader simplification family
- make matmul epilogue opportunities explicit rejected or accepted fusion
  decisions before backend binding
- make tile-shape choices target-aware before backend codegen, even when V0
  uses conservative synthetic tile limits
- refine memory traffic records by memory hierarchy role so global memory,
  local tile memory, future DMA, and future interconnect traffic do not collapse
  into one opaque byte count
- make collective algorithm decisions explicit, even when V0 can only record
  `algorithm=none` or reject direct/ring/tree/split candidates before graph
  import
- preserve MLIR and Shardy pass records through the compile orchestrator
- reject unsupported collectives during compile
- make fusion, placement, collective, cost, lowering, schedule, backend
  binding, profile, and explain records part of one trace report
- extend the next slices toward real layout/tile selection, collective lowering,
  memory-space assignment, buffer assignment, and backend kernel generation

The expected direction is not "build a smaller XLA." The direction is "build a
compiler/runtime where XLA-scale decisions are visible, typed, testable, and
owned by PjRTx."
