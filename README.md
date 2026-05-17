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

## JAX Sandbox

The first JAX-facing sandbox is split into two targets:

```sh
bazel test //tests/pjrt:plugin_ctypes_smoke_test
bazel build //tests/jax:jax_plugin_smoke
```

`//tests/pjrt:plugin_ctypes_smoke_test` runs in the Bazel macOS sandbox, loads
the plugin dylib, and verifies `GetPjrtApi` is exported.
`//tests/jax:jax_plugin_smoke` registers the same dylib with JAX as the `pjrtx`
backend and runs a tiny `jax.jit` add program through JAX's normal PJRT plugin
path. The hermetic Python graph pins `jax` and `jaxlib`.

```sh
bazel run //tests/jax:jax_plugin_smoke
# or, as an explicit manual test:
bazel test //tests/jax:jax_plugin_smoke_test --test_output=streamed
```

The runner uses the MLX/Metal backend. PjRTx currently supports only this
backend:

```sh
PJRTX_BACKEND=metal_mlx bazel run //tests/jax:jax_plugin_smoke
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
- Optimized libraries are pluggable backends: MLX/Metal is the only supported
  backend today, while attention kernels, future Neuron-like hardware, and
  CUDA/ROCm should all slot behind PjRTx-owned interfaces later.

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
- `src/backend`: the PjRTx backend interface plus implementations. MLX/Metal
  specifics live under `src/backend/mlx_metal` behind the private C++ C ABI
  shim.
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
  placement, opaque backend handle, byte-size metadata, and readiness. Host
  memory is only an explicit transfer source/sink at PJRT boundaries, never a
  persistent mirror of device storage.

Compile now materializes a runtime `ExecutableGraph` from the PjRTx plan.
Execution enters through that graph per device, then dispatches through
backend-neutral runtime operations. The next lowering step is to legalize graph
nodes into backend command fragments so devices enqueue MLX kernels, collectives,
custom calls, DMA copies, or library calls directly without pretending every op
is a buffer allocation.

## Current Status

- `src/runtime`: explicit client/device/memory/topology/buffer/executable-graph
  ownership model, including stable PJRT handle arrays, per-device graph nodes,
  and backend-neutral buffer
  placement/storage split. Buffers are moving to backend-owned storage as the
  source of truth; host memory is an ingress/egress transfer medium, not a
  runtime cache. Operations are not modeled as buffers in the compiler contract:
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
  directories. `src/backend/mlx_metal` owns the small private C ABI shim over vendored MLX
  Metal headers and MLX device APIs. It copies MLX Metal device names and
  recommended working-set sizes into plain C structs so Zig never owns
  C++/Objective-C objects. The same shim exposes opaque buffers that keep MLX
  arrays only; host bytes are transient at `buffer_from_host` and
  `copy_to_host`. There is no persistent host shadow in the MLX backend. The
  typed constructor preserves dtype and shape metadata for the MLX array path.
  Elementwise `u8` and `f32`
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
  MLIR DenseI64ArrayAttr, concatenate dimensions from MLIR integer attributes,
  and the ZML-declared heavy/control/random/structural op shells including
  `cholesky`, `custom_call`, `partition_id`, `reduce_precision`, `rng`,
  `rng_bit_generator`, `scatter`, `tuple`, and `while`), and emits precise diagnostics
  through `std.Io.Writer` for unsupported ops, GSPMD shardings, CHLO/shape
  interop, and invalid StableHLO portable artifacts. JAX-provided VHLO/StableHLO
  portable artifacts are deserialized at the StableHLO frontend boundary before
  later PjRTx-owned compiler stages run. The transform path is back on:
  Shardy propagation is gated on Shardy usage, then canonicalize/CSE/canonicalize
  runs through the MLIR pass manager.
- `src/plugin`: Zig shared library exporting `GetPjrtApi` and a PJRT API table
  with plugin attributes, errors, events, client/device/memory enumeration,
  host buffer copies, compile skeleton, loaded executable metadata, and
  per-device execute plumbing. Compile builds a runtime executable graph from
  the compiler plan; execute calls that graph for each selected device and no
  longer owns instruction scheduling in the PJRT adapter. Bootstrap graph
  execution currently lowers through runtime buffer operations: empty bootstrap
  programs copy arg0, linear
  StableHLO arithmetic chains execute the bootstrap `u8` and `f32` elementwise
  paths for matching buffers, StableHLO reshape preserves bytes while updating
  buffer dimensions and typed MLX metadata, StableHLO transpose performs dense
  row-major layout permutation, StableHLO broadcast-in-dim expands dense buffers
  with explicit output dimensions, StableHLO slice performs dense strided
  slicing with explicit bounds, StableHLO concatenate joins two dense buffers
  along the compiled dimension, StableHLO `reduce_precision` is an identity
  device copy, `partition_id` materializes scalar partition ids, deterministic
  bootstrap `rng`/`rng_bit_generator` paths exist for tests, and f32 dense
  `cholesky` currently has a correctness implementation that must move behind a
  backend-native linear algebra hook before it is accepted as a fast path.
  Region/control/complex-heavy
  operations that need real staged lowering (`while`, `tuple`,
  `get_tuple_element`, `scatter`, `convolution`, `custom_call`,
  `triangular_solve`, `fft`, `complex`/`real`/`imag`) fail with explicit
  `UNIMPLEMENTED` feature diagnostics instead of leaking through buffer-level
  execution.
  `PJRT_Client_Create` accepts `pjrtx_backend=metal_mlx`; other backend names
  are rejected. `PJRT_Client_Compile` accepts
  the bootstrap text compile
  options form used by the compiler tests:
  `replicas=2; partitions=2; use_shardy=true; assignment=0,1,2,3`.

The plugin target currently produces:

```text
bazel-bin/src/plugin/libpjrtx_metal_plugin.dylib
```

## Next Implementation Steps

- Finish the runtime cutover to device-only storage. Runtime buffers should keep
  backend handles live across instruction chains, never cache host mirrors, and
  hand backend-native operations to MLX without synchronizing after every op.
- Expand the MLX backend implementation for the LLM hot path: f32 binary/unary
  math, compare/select, reductions, matmul-like `dot_general`, shape/view ops,
  and then backend-native gather/pad/dynamic-slice/update where MLX exposes the
  right primitive.
- Add a backend legalization/pipeline stage to turn PjRTx value graphs into
  per-device command fragments: memory placement, layout, tiling/shard planning,
  async dependencies, fusion candidate groups, and final backend kernel/library
  dispatch. CPU/host reference code remains test-only oracle code, not runtime
  execution.
- Add focused MLX backend conformance tests for buffer/execution semantics
  through the vtable, with extra tests asserting MLX chains do not copy device
  intermediates back to host.
