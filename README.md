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

## Current Status

- `src/runtime`: explicit client/device/memory/topology/buffer ownership model,
  including synthetic 2/4+ device test coverage, stable PJRT handle arrays, and
  a backend-kind entry point for the Metal/MLX backend. `metal_mlx` buffers keep
  a host shadow for bootstrap execution and own an optional opaque Metal buffer
  for real transfer-path coverage.
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
- `src/backend`: a small PjRTx C ABI shim over the vendored MLX Metal headers
  and MLX device APIs. It copies MLX Metal device names and recommended
  working-set sizes into plain C structs so Zig never owns C++/Objective-C
  objects. The same shim exposes opaque buffers that always keep bootstrap host
  bytes and opportunistically keep an MLX array when the vendored runtime can
  construct one, avoiding MLX's non-hermetic default `mlx.metallib` load during
  plain PJRT host-buffer ownership. The typed constructor preserves dtype and
  shape metadata for the MLX array path. Elementwise `u8` and `f32` arithmetic
  plus f32 unary math, transpose, broadcast-in-dim, slice, and concatenate now
  run through
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
  plan ops for StableHLO add/subtract/multiply/divide/negate, f32
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
  per-device execute plumbing. Bootstrap execute now dispatches from the
  compiled executable plan: empty bootstrap programs copy arg0, linear
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

- Extend executable-plan driven StableHLO op dispatch to shape-preserving view
  ops while preserving the typed MLX buffer metadata.
- Implement the initial StableHLO op set and JAX CPU-vs-PjRTx correctness tests.
