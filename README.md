# PjRTx

PjRTx is a Zig 0.16 + Bazel bootstrap for PJRT plugins targeting JAX and ZML.
The current milestone is a standalone PJRT C API producer skeleton with a
multi-device runtime model and Shardy-aware compile planning.

## Build

```sh
bazel build //...
bazel test //...
bazel build //src/plugin:pjrtx_metal_plugin
```

The default `.bazelrc` uses:

```sh
--override_repository=xla=/Users/hugo/Developer/xla
```

Remove or override that flag to exercise the pinned fallback in
`third_party/xla/repo.bzl`.

The Bazel toolchain is configured to follow ZML's sandboxed macOS path:

- Zig downloads prefer the Cloudflare-backed `https://mirror.zml.ai/zig` mirror.
- C/C++ actions use hermetic LLVM via `@llvm//toolchain:all`.
- macOS SDK headers/libs come from the hermetic `@macos_sdk` repository created
  with `@llvm//extensions:osx.bzl`, so builds do not depend on local Xcode/CLT
  discovery. The root module explicitly adds the Metal stack framework slices
  through `osx.frameworks`.

## Architecture

PjRTx follows the ZML/v2 bias toward explicit ownership and composability. A
PJRT client owns a selected backend, topology, devices, memories, transfer
paths, compilation results, and executable lifecycle. Backend behavior is never
hidden in ambient global state.

The architecture imports these ZML/v2 principles:

- Platform ownership is explicit: clients select and own the backend they use
  for transfer, compile, and execute.
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

- `src/core`: backend-neutral value types for dtype, shape, layout, sharding,
  device ids, memory ids, topology, buffer descriptors, placements, compile
  options, value/instruction executable plans, and bootstrap plan instructions.
  These are the contracts shared across compiler, runtime, plugin, and backends.
- `src/compiler`: StableHLO/MLIR ingestion, verification, compile option
  parsing, Shardy metadata extraction, and construction of PjRTx executable
  plans. The compiler depends on `src/core` and MLIR C API targets only; it does
  not depend on runtime or backend packages.
- `src/runtime`: client/device/memory/topology/buffer/executable lifecycle and
  scheduling. Runtime consumes PjRTx executable plans and dispatches through a
  backend vtable with opaque backend buffer handles. It does not import MLX,
  Metal, or C symbols.
- `src/backend`: the PjRTx backend interface plus implementations. Synthetic is
  first-class for tests and multi-device simulation. MLX/Metal specifics live
  under `src/backend/mlx_metal` behind the private C++ C ABI shim.
- `src/plugin`: PJRT C API adapter only. It parses PJRT inputs, selects a
  backend through the backend registry, calls compiler/runtime APIs, and
  translates results back into PJRT structs. It does not import MLX symbols.

Bazel enforces these boundaries with package dependencies and the
`//:architecture_boundary_test` smoke test.

### IR Direction

Operations are not modeled as buffers. A StableHLO op is an input program
operation. A PjRTx instruction is a compiler/runtime scheduling unit. A runtime
buffer is one possible materialized storage object for a value after placement
and memory planning.

The compiler contract keeps these concepts separate:

- `Value`: typed logical tensor/token/control value with shape, layout, sharding,
  memory-space intent, and placement constraints.
- `Instruction`: computation, transfer, view, async dependency, collective, or
  backend fragment that consumes and produces `ValueId`s.
- `ExecutablePlan`: topology, compile options, sharding plans, values,
  instructions, device assignment, memory plan, and backend legalization data.
- `Buffer`: runtime storage for a placed value shard, with logical descriptor,
  placement, opaque backend handle, optional host/debug cache, and readiness.

The bootstrap linear executor still walks instructions over buffers for smoke
tests, but the contract evolves toward value graphs plus backend legalization so
other backends can lower to command buffers, graphs, fused kernels, streams,
DMA copies, or library calls without pretending every op is a buffer allocation.

## Current Status

- `src/runtime`: explicit client/device/memory/topology/buffer ownership model,
  including synthetic 2/4+ device test coverage, stable PJRT handle arrays, and
  backend-neutral buffer placement/storage split. Buffers keep a host shadow for
  bootstrap correctness and may own an opaque backend buffer for transfer-path
  coverage. Operations are not modeled as buffers in the compiler contract:
  PjRTx plans contain logical values and instructions, while runtime buffers are
  only materialized storage for placed value shards.
- `third_party/mlx`: pinned MLX vendor module at
  `ml-explore/mlx@7b7c12407f85b494e3e6d1cd3888650d224f362c`, exposing only
  core headers plus a Metal-first runtime target. The runtime target vendors
  MLX core, GPU common code, Metal backend code, no-CPU/no-CUDA stubs, and
  header-only `fmt`; full CPU and CUDA backend source trees are intentionally
  not linked by the PjRTx Bazel wrapper. MLX's Metal JIT source strings are
  generated from the vendored kernel headers by a sandboxed Bazel action instead
  of MLX's default `xcrun metal -E` CMake path, keeping the runtime on MLX while
  preserving the no-host-Xcode constraint.
- `third_party/metal_cpp`: pinned Apple `metal-cpp_26.zip`, used by the MLX
  Metal header target and linked against the sandboxed macOS SDK framework
  slices. No host Xcode SDK paths are used.
- `src/backend`: a Zig backend vtable/interface plus backend-specific package
  directories. `src/backend/synthetic` owns synthetic multi-device simulation.
  `src/backend/mlx_metal` owns the small private C ABI shim over vendored MLX
  Metal headers and MLX device APIs. It copies MLX Metal device names and
  recommended working-set sizes into plain C structs so Zig never owns
  C++/Objective-C objects. The same shim exposes opaque buffers that always keep
  bootstrap host bytes and opportunistically keep an MLX array when the vendored
  runtime can construct one, avoiding MLX's non-hermetic default `mlx.metallib`
  load during plain PJRT host-buffer ownership. The typed constructor preserves
  dtype and shape metadata for the MLX array path. Elementwise `u8` and `f32`
  arithmetic plus f32 unary math, transpose, broadcast-in-dim, slice, and
  concatenate now run through
  `mlx::core::{add,subtract,multiply,divide,floor_divide,negative,exp,tanh,sqrt,rsqrt,transpose,reshape,broadcast_to,slice,concatenate}`
  on the GPU device using MLX's runtime Metal JIT when an MLX array and Metal
  device are available. The PjRTx C shim no longer builds direct Metal arithmetic
  kernels or calls host `xcrun`.
- `src/compiler`: compile-option parsing and executable-plan construction for
  replicas, partitions, device assignment, and Shardy metadata. Program
  ingestion now flows through `std.Io.Reader`, parses and verifies MLIR through
  the MLIR C API, walks `func`, `stablehlo`, and `sdy` operations/attributes
  directly, records supported StableHLO ops, constructs parameter/output
  sharding plans from Shardy C attributes, lowers the first bootstrap execution
  ops into a PjRTx executable plan (`copy_arg0` for empty programs, arithmetic
  plan instructions for StableHLO add/subtract/multiply/divide/negate, f32
  exp/tanh/sqrt/rsqrt, shape metadata for StableHLO reshape, transpose
  permutations, broadcast dimensions, slice start/limit/stride metadata from
  MLIR DenseI64ArrayAttr, and concatenate dimensions from MLIR integer
  attributes), and emits precise diagnostics
  through `std.Io.Writer` for unsupported ops, GSPMD shardings, CHLO/shape
  interop, and unsupported StableHLO bytecode. The transform path is back on:
  Shardy propagation is gated on Shardy usage, then canonicalize/CSE/canonicalize
  runs through the MLIR pass manager.
- `src/plugin`: Zig shared library exporting `GetPjrtApi` and a PJRT API table
  with plugin attributes, errors, events, client/device/memory enumeration,
  host buffer copies, compile skeleton, loaded executable metadata, and
  per-device execute plumbing. Bootstrap execute now dispatches through runtime
  from compiled executable plans: empty bootstrap programs copy arg0, linear
  StableHLO arithmetic chains execute the bootstrap `u8` and `f32` elementwise
  paths for matching buffers, StableHLO reshape preserves bytes while updating
  buffer dimensions and typed MLX metadata, StableHLO transpose performs dense
  row-major layout permutation, StableHLO broadcast-in-dim expands dense buffers
  with explicit output dimensions, StableHLO slice performs dense strided
  slicing with explicit bounds, and StableHLO concatenate joins two dense
  buffers along the compiled dimension.
  `PJRT_Client_Create` accepts `pjrtx_backend`
  (`metal_mlx` or `synthetic`) plus `pjrtx_synthetic_device_count` as an int64
  create option for synthetic multi-device tests. `PJRT_Client_Compile` accepts
  the bootstrap text compile
  options form used by the compiler tests:
  `replicas=2; partitions=2; use_shardy=true; assignment=0,1,2,3`.

The plugin target currently produces:

```text
bazel-bin/src/plugin/libpjrtx_metal_plugin.dylib
```

## Next Implementation Steps

- Grow the staged PjRTx IR passes: shape/type/layout verification, topology
  validation, memory-space planning, tiling/shard planning, async dependency
  modeling, fusion marking, and backend legalization.
- Implement the initial StableHLO op set and JAX CPU-vs-PjRTx correctness tests.
- Add focused backend conformance tests so synthetic and MLX backends prove the
  same buffer/execution semantics through the vtable.
