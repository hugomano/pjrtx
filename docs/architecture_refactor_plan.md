# PjRTx Architecture Refactor Plan

This plan moves PjRTx from bootstrap proof of concept to a Zig-first,
backend-composable PJRT implementation. The north star is still fast and
correct LLM execution from JAX and ZML, but the architecture should make
platform ownership, compilation, placement, memory, IO, and sharding explicit
instead of implicit.

## ZML/v2 Principles To Import

ZML/v2's useful lesson for PjRTx is composability over hidden turnkey behavior.
The platform is a first-class object used for transfer, build, compile, and
execute. PjRTx should mirror that with explicit client/backend ownership and no
ambient backend state.

- Platform ownership is explicit: PJRT clients own a selected backend, topology,
  devices, memories, transfer engines, and executable lifecycle.
- Build sandboxing is non-negotiable: runtime packages are stripped to the
  minimum, vendored hermetically, and built through Bazel with the sandboxed
  macOS SDK and hermetic LLVM.
- Memory and IO are composable: pinned, pageable, device, accelerator-local, and
  interconnect-visible memory classes remain visible in the model, even when a
  backend like Metal maps most storage to unified memory.
- Transfers are streams, not incidental copies: ingestion and diagnostics use
  `std.Io.Reader` and `std.Io.Writer`; future weight loading should compose
  with DMA/pinned staging and overlapped host-to-device writes.
- Sharding is first-class: meshes, shardings, placements, and manual shard-local
  computation are compiler/runtime data, not backend side channels.
- Optimized libraries are pluggable backends: MLX is the first real backend,
  but attention kernels, future Neuron-like hardware, CUDA/ROCm, and synthetic
  tests should all slot behind PjRTx-owned interfaces.

## Current Layering Contract

- `src/core` owns backend-neutral value types and executable-plan contracts.
- `src/compiler` owns StableHLO/MLIR/Shardy ingestion and emits PjRTx IR.
- `src/runtime` owns client/device/memory/buffer/executable lifecycle and
  scheduling. It depends on backend interfaces, not backend implementations.
- `src/backend` owns only the backend interface and registry.
- `src/backend/synthetic` owns synthetic multi-device simulation.
- `src/backend/mlx_metal` owns MLX/Metal Zig, C, and C++ implementation details.
- `src/plugin` is only a PJRT C API adapter.

## IR Direction

Operations must not be modeled as buffers. A StableHLO op is an input program
operation. A PjRTx instruction is a compiler/runtime scheduling unit. A runtime
buffer is one possible materialized storage object for a value after placement
and memory planning.

The central compiler contract should keep these concepts separate:

- `Value`: typed logical tensor/token/control value with shape, layout, sharding,
  memory-space intent, and placement constraints.
- `Instruction`: computation, transfer, view, async dependency, collective, or
  backend fragment that consumes and produces `ValueId`s.
- `ExecutablePlan`: topology, compile options, sharding plans, values,
  instructions, device assignment, memory plan, and backend legalization data.
- `Buffer`: runtime storage for a placed value shard, with logical descriptor,
  placement, opaque backend handle, optional host/debug cache, and readiness.

The bootstrap linear executor may still walk instructions over buffers for
smoke tests, but the contract must evolve toward value graphs plus backend
legalization so other backends can lower to command buffers, graphs, fused
kernels, streams, DMA copies, or library calls without pretending every op is a
buffer allocation.

## Compiler Pipeline

1. StableHLO/MLIR ingestion and verification through MLIR C APIs.
2. PjRTx value/instruction graph construction.
3. Shape, dtype, layout, and token/control verification.
4. Shardy mesh/sharding/manual-computation interpretation.
5. Topology and device-assignment validation.
6. Memory-space and layout planning.
7. Tiling and shard planning.
8. Async/dependency modeling for transfer and execution overlap.
9. Fusion candidate marking.
10. Backend legalization into backend-fragment instructions.

## Runtime And Backend Direction

- Runtime dispatches executable plans through backend vtables and never imports
  MLX/Metal/C symbols directly.
- Backends live in their own package directories from the start.
- Backend APIs should grow around values/fragments/transfer streams, while
  preserving the current buffer methods as bootstrap conveniences.
- Synthetic remains equal in importance to MLX for conformance and
  multi-device/sharding tests.

## Acceptance Gates

- `bazel test //... --test_output=errors`
- `bazel build //...`
- `bazel-bin/src/plugin/libpjrtx_metal_plugin.dylib` exists.
- Architecture boundary tests reject compiler/runtime/plugin dependency leaks.
- Backend implementation files stay under backend-specific directories.
