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
PJRTX_BACKEND=metal_mlx PJRTX_TRACE=1 bazel run //tests/jax:jax_op_suite
PJRTX_BACKEND=metal_mlx PJRTX_TRACE=1 bazel run //tests/jax:jax_llama_like_inference
# or, as an explicit manual test:
bazel test //tests/jax:jax_plugin_smoke_test --test_output=streamed
```

The runner uses the MLX/Metal backend. PjRTx currently supports only this
backend:

```sh
PJRTX_BACKEND=metal_mlx bazel run //tests/jax:jax_plugin_smoke
```

Set `PJRTX_TRACE=1` to print one-line compile, host-to-device, execute, and
device-to-host timing/byte counters while running the sandbox:

```sh
PJRTX_BACKEND=metal_mlx PJRTX_TRACE=1 bazel run //tests/jax:jax_plugin_smoke
```

`//tests/jax:jax_op_suite` compares the currently lowered PjRTx fast path
against JAX CPU for repeated `jax.jit` execution of elementwise float chains,
reshape/transpose/broadcast, reductions, matmul, clipping via min/max, integer
bitwise ops, StableHLO sort regions, descending sort, argsort, CHLO/StableHLO
top-k composites, single-axis, point-style, and batched general gather forms,
single-axis, point-style, windowed, and batched scatter
set/add forms, and constant
edge/interior padding. The suite asserts repeated
execution against the MLX Metal device fast path with no runtime fallback.

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

Compile now materializes a runtime `ExecutableGraph` from the PjRTx plan and
asks the selected backend to compile an opaque backend executable. Execution
enters through that graph per device. PJRT compile uses device-only lowering:
the whole program must legalize to an MLX backend executable, and execute only
dispatches opaque backend buffer handles through that executable. Runtime
tracks buffer lifecycle (`live`, `deleted`, `donated`), readiness events,
device-memory byte accounting, host/device transfer counters, and executable
cache hit/miss metadata. The old backend-neutral operation interpreter remains
available only as an explicit runtime test mode. New llama-facing work should
broaden backend legalization so devices enqueue MLX kernels, collectives,
custom calls, DMA copies, or library calls directly without pretending every op
is a buffer allocation.

Backend legalization is validated before the plugin accepts a compiled program.
Failures are reported through `std.Io.Writer` diagnostics with the lowering
pass, instruction index, op name, value id, dtype/rank/shape, sharding label, and
backend feature label, so unsupported forms such as batched gather metadata or
general scatter fail during compile instead of falling through to host/reference
execution.

## Current Status

- `src/runtime`: explicit client/device/memory/topology/buffer/executable-graph
  ownership model, including stable PJRT handle arrays, per-device graph nodes,
  backend-neutral buffer placement/storage split, executable cache metadata,
  strict buffer deletion/donation state, readiness events, and memory/transfer
  accounting. Buffers are moving to backend-owned storage as the source of
  truth; host memory is an ingress/egress transfer medium, not a runtime cache.
  Operations are not modeled as buffers in the compiler contract: PjRTx plans
  contain logical values and instructions, while runtime buffers are only
  materialized storage for placed value shards.
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
  directories. Backends expose opaque buffer handles and opaque executable
  handles. `src/backend/mlx_metal` owns the small private C ABI shim over vendored MLX
  Metal headers and MLX device APIs. It copies MLX Metal device names and
  recommended working-set sizes into plain C structs so Zig never owns
  C++/Objective-C objects. The same shim exposes opaque buffers that keep MLX
  arrays only; host bytes are transient at `buffer_from_host` and
  `copy_to_host`. There is no persistent host shadow in the MLX backend. The
  typed constructor preserves dtype and shape metadata for the MLX array path,
  including `s8`, `u8`, `s32`, `u32`, `f16`, `bf16`, and `f32` host imports.
  StableHLO `convert` lowers to MLX `astype` through the backend vtable, so
  dtype casts on resident buffers stay on device. `bitcast_convert` is kept out
  of the fast path until the backend has an explicit dtype reinterpretation API.
  MLX backend executables lower into an explicit backend program with nodes,
  backend value ids, liveness metadata, output-value retention, resident
  constants, fusible view/elementwise group metadata, and per-device hardware
  assignments. The execute path still walks this program in schedule order, but
  it now releases dead intermediates and enqueues MLX evaluation at fusion group
  ends, non-fusible materialization nodes, and final PJRT outputs without
  copying intermediates to host. Compile-time
  constants stay resident as device arrays and clone those handles during
  execute, which is the first weight-residency path. Elementwise arithmetic,
  extra unary/binary math (`atan2`, `expm1`,
  `is_finite`, nearest-even `round`), logical/bitwise ops, compare/select for
  MLX-supported numeric dtypes,
  f32 sum/max and pred and/or reductions, `dot_general`, dtype casts, StableHLO `iota`, StableHLO `clamp`,
  shape/view ops, StableHLO `reverse`, dynamic slice, dynamic update-slice,
  constant edge/interior padding, single-axis, point-style, and batched general gather
  forms, single-axis, point-style, windowed, and batched StableHLO scatter set/add, and ascending/descending
  StableHLO `sort` now run through MLX core operations on the GPU device when
  MLX arrays and Metal devices are available. More exotic scatter metadata
  still requires broader backend legalization before
  they enter the fast path.
  The PjRTx C shim no longer builds direct Metal arithmetic kernels or calls
  host `xcrun`.
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
  Shardy propagation is gated on Shardy usage, then inline/canonicalize/CSE/
  canonicalize runs through the MLIR pass manager.
- `src/plugin`: Zig shared library exporting `GetPjrtApi` and a PJRT API table
  with plugin attributes, errors, events, client/device/memory enumeration,
  host buffer copies, compile skeleton, loaded executable metadata, and
  per-device execute plumbing. Compile builds a runtime executable graph from
  the compiler plan; execute calls that graph for each selected device and no
  longer owns instruction scheduling in the PJRT adapter. Compile now rejects
  programs that cannot fully lower to an MLX backend executable, so llama-facing
  execution does not fall back through host/reference buffers. Bootstrap graph
  execution supports: empty bootstrap programs copy arg0, linear
  StableHLO arithmetic chains execute the bootstrap `u8` and `f32` elementwise
  paths for matching buffers, StableHLO reshape preserves bytes while updating
  buffer dimensions and typed MLX metadata, StableHLO `iota` materializes
  coordinate grids on device, StableHLO transpose performs dense row-major
  layout permutation, StableHLO broadcast-in-dim expands dense buffers with
  explicit output dimensions, StableHLO slice performs dense strided slicing
  with explicit bounds, StableHLO `reverse` flips compiled axes, StableHLO
  concatenate joins two dense buffers along the compiled dimension, StableHLO
  `sort` uses the comparator direction parsed from the StableHLO region,
  key/value sort lowers through MLX `argsort` plus device-side
  `take_along_axis`, CHLO/StableHLO top-k composites lower to a device-only
  sort/argsort/reverse/slice plan, StableHLO `reduce_precision` is an identity
  device copy, `partition_id` materializes scalar partition ids, deterministic
  bootstrap `rng`/`rng_bit_generator` paths exist for tests, and f32 dense
  `cholesky` currently has a correctness implementation that must move behind a
  backend-native linear algebra hook before it is accepted as a fast path.
  Region/control/complex-heavy
  operations that need real staged lowering (`while`, `tuple`,
  `get_tuple_element`, general scatter, `convolution`, `custom_call`,
  `triangular_solve`, `fft`, `complex`/`real`/`imag`) fail with explicit
  `UNIMPLEMENTED` feature diagnostics instead of leaking through buffer-level
  execution.
  `PJRT_Client_Create` accepts `pjrtx_backend=metal_mlx`; other backend names
  are rejected. `PJRT_Client_Compile` accepts
  the bootstrap text compile
  options form used by the compiler tests:
  `replicas=2; partitions=2; use_shardy=true; assignment=0,1,2,3`.
  Optional `PJRTX_TRACE=1` instrumentation prints `pjrtx_trace` lines for
  compile, H2D, execute, and D2H events, including byte counts, device count,
  backend-executable eligibility, and elapsed microseconds.

The plugin target currently produces:

```text
bazel-bin/src/plugin/libpjrtx_metal_plugin.dylib
```

## Next Implementation Steps

- Broaden MLX backend executable legalization. The PJRT path is now device-only
  and the runtime fallback path is test-only, so unsupported StableHLO ops must
  either lower to MLX graph fragments or fail at compile time with precise
  diagnostics. Current backend legalization diagnostics identify the blocking
  pass, op/value, shape, sharding, and MLX feature label.
- Expand the MLX backend implementation for the remaining LLM hot path:
  richer custom comparator sort semantics, top-k fast paths using MLX library
  primitives where they expose indices directly, and custom-call hooks where
  MLX exposes the right primitive.
- Add a backend legalization/pipeline stage to turn PjRTx value graphs into
  per-device command fragments: memory placement, layout, tiling/shard planning,
  async dependencies, evaluated fusion groups, and final backend kernel/library
  dispatch. CPU/host reference code remains test-only oracle code, not runtime
  execution.
- Add focused MLX backend conformance tests for buffer/execution semantics
  through the vtable, with extra tests asserting MLX chains do not copy device
  intermediates back to host.
