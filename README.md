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
bazel test //src/plugin/pjrt:plugin_ctypes_smoke_test
bazel build //src/plugin/jax:jax_plugin_smoke
```

`//src/plugin/pjrt:plugin_ctypes_smoke_test` runs in the Bazel macOS sandbox, loads
the plugin dylib, and verifies `GetPjrtApi` is exported.
`//src/plugin/jax:jax_plugin_smoke` registers the same dylib with JAX as the `pjrtx`
backend and runs a tiny `jax.jit` add program through JAX's normal PJRT plugin
path. The hermetic Python graph pins `jax` and `jaxlib`.

```sh
bazel run //src/plugin/jax:jax_plugin_smoke
PJRTX_BACKEND=metal_mlx PJRTX_TRACE=1 bazel run //src/plugin/jax:jax_op_suite
PJRTX_BACKEND=metal_mlx PJRTX_TRACE=1 bazel run //src/plugin/jax:jax_llama_like_inference
PJRTX_BACKEND=metal_mlx bazel test //src/plugin/jax:jax_upstream_subset_test --test_output=errors
PJRTX_BACKEND=metal_mlx bazel test //src/plugin/jax:jax_upstream_all_test --test_output=errors
# or, as an explicit manual test:
bazel test //src/plugin/jax:jax_plugin_smoke_test --test_output=streamed
```

The runner uses the MLX/Metal backend. PjRTx currently supports only this
backend:

```sh
PJRTX_BACKEND=metal_mlx bazel run //src/plugin/jax:jax_plugin_smoke
```

`//src/plugin/jax:jax_upstream_subset_test` runs a curated slice of the pinned
official JAX test tree through `@jax_upstream`. The pytest adapter mirrors the
JAX Bazel backend matrix by setting `jax_test_dut=pjrtx` and registering JAX
test-util tags `pjrtx,metal,gpu`, so upstream `run_on_devices`,
`skip_on_devices`, and `_CompileAndCheck` tests can be used directly against
the plugin. Upstream JAX runs are plugin-only by default:
`jax_platforms=pjrtx`, with no CPU backend admitted. To inspect coverage without
executing tests:

```sh
bazel run //src/plugin/jax:jax_upstream_subset -- --collect-only
bazel run //src/plugin/jax:jax_upstream_all -- --collect-only
```

`//src/plugin/jax:jax_upstream_all_test` points pytest at the full upstream JAX
`tests/` tree. It is tagged `manual` and `pjrtx_full_jax` with an eternal
timeout because it is a broad conformance job rather than a quick smoke test.
If a debugging run genuinely needs CPU as a secondary platform, pass
`--test_arg=--allow-cpu`; do not use that for plugin coverage.

Set `PJRTX_TRACE=1` to print one-line compile, host-to-device, execute, and
device-to-host timing/byte counters while running the sandbox:

```sh
PJRTX_BACKEND=metal_mlx PJRTX_TRACE=1 bazel run //src/plugin/jax:jax_plugin_smoke
```

Set `PJRTX_PROFILE=1` when investigating throughput. This enables the same
plugin compile/execute summaries plus lower-level `pjrtx_profile` records from
the MLX Metal backend:

- `event=mlx_from_host`: host-to-device imports, including closed-over weights
  and activations.
- `event=mlx_eval_many`: MLX lazy graph eval batches issued by materialization
  boundaries. By default this reports host enqueue time only.
- `event=mlx_copy_to_host`: explicit PJRT output readbacks.
- `event=backend_execute`: runtime schedule timing split into nodes, fusion
  groups, materialization evals, and output cloning.

Use `PJRTX_PROFILE=verbose` for one `backend_schedule_item` row per scheduled
node/fusion/materialization item:

```sh
PJRTX_BACKEND=metal_mlx PJRTX_PROFILE=1 bazel run //src/plugin/jax:jax_llama_like_inference
PJRTX_BACKEND=metal_mlx PJRTX_PROFILE=verbose bazel run //src/plugin/jax:jax_llama_like_inference
```

For device-side profiling, use MLX's Metal capture path. This writes one
`.gputrace` per MLX eval batch and synchronizes the stream so the capture is
complete; inspect the trace in Xcode/Instruments for GPU command timing. macOS
requires `MTL_CAPTURE_ENABLED=1` or MLX will report that the capture layer was
not inserted:

```sh
mkdir -p /tmp/pjrtx-metal-captures
MTL_CAPTURE_ENABLED=1 \
  PJRTX_BACKEND=metal_mlx \
  PJRTX_PROFILE=1 \
  PJRTX_METAL_CAPTURE_DIR=/tmp/pjrtx-metal-captures \
  bazel run //src/plugin/jax:jax_llama_like_inference
```

`PJRTX_PROFILE_DEVICE_SYNC=1` adds `device_sync_wait_us` to `event=mlx_eval_many`
without writing capture files. This is an intrusive host-observed wait for queued
GPU work to finish, useful for decode triage but not a substitute for Metal GPU
timestamps:

```sh
PJRTX_BACKEND=metal_mlx \
  PJRTX_PROFILE=1 \
  PJRTX_PROFILE_DEVICE_SYNC=1 \
  bazel run //src/plugin/jax:jax_llama_like_inference
```

`//src/plugin/jax:jax_op_suite` compares the currently lowered PjRTx fast path
against JAX CPU for repeated `jax.jit` execution of elementwise float chains,
reshape/transpose/broadcast, f32/f16 sum/max/min reductions, two-output
max/index reductions for ZML-style argMax, f32/f16 matmul,
attention-style `dot_general` score/context contractions, clipping via min/max, integer
bitwise ops plus integer right shifts, predicate logic, integer
`popcnt`/`count_leading_zeros`, dense
`bitcast_convert`, f32 `sin`/`cos`/`log1p`/`logistic`/`floor`/`ceil`/`sign`,
`power`/`remainder` with scalar-broadcast compare/select lowerings, f32 `cbrt`,
round-away-from-zero, StableHLO `complex`, complex `abs`, and
real/complex-input `real`/`imag`, StableHLO sort regions, descending sort, argsort, CHLO/StableHLO
top-k composites, single-axis, point-style, and batched general gather forms,
single-axis, point-style, windowed, and batched scatter
set/add forms, f32/f16 StableHLO `reduce_window` sum/max forms with static windows,
two-output max/index `reduce_window` for ZML-style max-pool,
canonical f32/f16/bf16 StableHLO `convolution` forms through MLX
`conv_general`,
StableHLO `rfft` through MLX FFT with c64 PJRT buffer roundtrips,
f32 `cholesky`/left-lower `triangular_solve` through backend-native MLX Metal
kernels,
and constant
edge/interior padding. The suite asserts repeated
execution against the MLX Metal device fast path with no runtime fallback. With
`PJRTX_TRACE=1`, it also parses per-case traces and rejects compile fallback,
non-backend execute candidates, cache pressure failures, and pending backend
completion in the MLX path.
`//src/plugin/jax:jax_custom_call` asks JAX to emit a `stablehlo.custom_call` to the
built-in `pjrtx.mlx_metal.custom_binary_add_f32` target and executes that target
through a backend-owned MLX `fast::metal_kernel` before a follow-on device
`sqrt`; the test does not drive the PJRT C API from Python.
`//src/plugin/jax:jax_rng_u64` enables JAX x64 only for an isolated Threefry
`uint64` output check so the main op suite stays on the narrower dtype surface.
`//src/plugin/jax:jax_llama_like_inference` also closes over tiny and medium
llama-shaped weight sets, runs each compiled block three times with only the
activation as a PJRT execute argument, captures `pjrtx_trace` from the plugin,
and asserts that backend execute counters advance while resident constant
count/bytes stay fixed and resident constant nodes are borrowed again instead
of re-uploaded. It checks that compile-time resident bytes cover closed-over
weights, execute-time H2D traffic contains exactly one activation upload per
run, D2H traffic contains one output readback per run, and compile traces expose
a populated backend program with values, nodes, dependency edges, schedule
items, fusion groups, materialization boundaries, planned release count, and
planned release bytes, peak live value count, and peak live bytes. Execute
traces also expose cumulative fusion-group executions, materialization eval
batches/buffers, and released backend intermediates, so the fixture verifies
that repeated execution uses the scheduled graph and liveness path rather than
retaining every transient value. Execute traces include backend completion
state counters; MLX Metal currently reports completed dispatches, while future
async backends must integrate their completion events before outputs become
ready. Runtime release counts must match the
compile-time liveness plan on every execute. The script prints a compact
repeated-execute budget covering resident constant bytes, backend program
shape, liveness byte pressure, activation upload bytes, output readback bytes,
compile latency, and first/last backend execute latency. The first execute is
reported separately from the final two steady-state executes so later latency
budgets can ignore one-time MLX warm-up. The fixture also compiles an equivalent
closed-weight JAX function a second time and requires `cache_hit=1` with stable
program, residency, and liveness metadata. Cache hits now reuse the cached
backend executable handle and resident constants, and the cache probe verifies
that backend execute and resident-constant borrow counters continue from the
original loaded executable. Set `PJRTX_EXECUTABLE_CACHE_MAX_BYTES` to cap
resident cached backend executable storage; idle cached executables are evicted
under that cap while active loaded executables remain protected. Host uploads,
resident executable compilation, and graph executes feed planned allocation
pressure into the same cache: if tracked device-memory capacity would be
exceeded, runtime trims idle resident executables before importing activations,
accounting newly compiled constants, or dispatching the backend program. The
sandbox enforces strict byte
budgets for activation uploads, output readbacks, and
resident-weight overhead, plus a loose steady execute latency ceiling. Override
latency or overhead budgets with
`PJRTX_STEADY_EXECUTE_BUDGET_US`, `PJRTX_MEDIUM_STEADY_EXECUTE_BUDGET_US`,
`PJRTX_RESIDENT_CONSTANT_OVERHEAD_BUDGET_BYTES`, or
`PJRTX_MEDIUM_RESIDENT_CONSTANT_OVERHEAD_BUDGET_BYTES`.

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
  options, value/instruction executable plans, donated-parameter metadata, and
  bootstrap plan instructions. These are the contracts shared across compiler,
  runtime, plugin, and backends.
- `src/compiler`: StableHLO/MLIR ingestion, verification, compile option
  parsing, Shardy metadata extraction, and construction of PjRTx executable
  plans. The compiler depends on `src/core` and MLIR C API targets only; it does
  not depend on runtime or backend packages.
- `src/runtime`: client/device/memory/topology/buffer/executable lifecycle and
  scheduling. Runtime consumes PjRTx executable plans and dispatches through a
  backend vtable with opaque backend buffer handles. Runtime owns execute
  completion propagation and output readiness events. It does not import MLX,
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

- `Value`: typed logical tensor/token/control value with explicit storage kind
  (`tensor`, `tuple`, `token`, or `complex_pair`), shape, layout, sharding,
  structured element references, memory-space intent, and placement constraints.
- `Instruction`: computation, transfer, view, async dependency, collective, or
  backend fragment that consumes and produces `ValueId`s. Region-bodied
  instructions reference `PlanRegion` summaries instead of requiring later
  passes to rediscover nested MLIR regions.
- `ExecutablePlan`: topology, compile options, sharding plans, values,
  instructions, regions, device assignment, memory plan, and backend
  legalization data.
- `Buffer`: runtime storage for a placed value shard, with logical descriptor,
  placement, opaque backend handle, byte-size metadata, and readiness. Host
  memory is only an explicit transfer source/sink at PJRT boundaries, never a
  persistent mirror of device storage.

Compile now materializes a runtime `ExecutableGraph` from the PjRTx plan and
asks the selected backend to compile an opaque backend executable. Execution
enters through that graph per device. PJRT compile uses device-only lowering:
the whole program must legalize to an MLX backend executable, and execute only
dispatches opaque backend buffer handles through that executable. The PJRT
adapter no longer carries the old instruction-by-instruction legacy execute
path, and the runtime no longer has an instruction interpreter fallback. Runtime
tracks buffer lifecycle (`live`, `deleted`, `donated`), readiness events,
device-memory byte accounting, host/device transfer counters, and executable
cache hit/miss metadata. Buffer consumers now honor readiness: graph execution
rejects pending or failed argument buffers, and D2H copies refuse to read from
buffers whose producer event has not completed successfully. Backend executable results are validated against the
compiled PjRTx output contract before PJRT buffers are created, including output
arity, dtype, shape, and byte size. Backend executables also reject invalid
device slots and require execute argument counts to exactly match the compiled
parameter contract. New llama-facing work should broaden backend legalization so
devices enqueue MLX kernels, collectives, custom calls, DMA copies, or library
calls directly without pretending every op is a buffer allocation.

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
  strict buffer deletion/donation state, bounded readiness events with
  pending/ready/failed transitions, dependency chaining, and callback registration, and
  memory/transfer accounting. Buffers are moving to backend-owned storage as the source of
  truth; host memory is an ingress/egress transfer medium, not a runtime cache.
  Buffer delete and donation now release backend storage and resident-byte
  accounting immediately while preserving a failed tombstone handle for PJRT
  lifecycle queries.
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
  dtype casts on resident buffers stay on device. StableHLO `bitcast_convert`
  lowers through the backend vtable to MLX `view(dtype)` plus a shape-only
  reshape when needed, so ZML byte views and same-byte dtype reinterpretation
  stay device-resident instead of becoming numeric casts or host roundtrips.
  MLX backend executables lower into an explicit backend program with value
  records, nodes, a typed schedule, dependency edges, producer/last-use
  metadata, output-value retention, resident constants, typed fusible
  view/elementwise groups with explicit node lists plus boundary inputs/outputs,
  materialization boundaries, and per-device hardware assignments. The schedule
  emits fusion-group work items for fusible node sets and
  materialization-boundary work items for forced MLX eval. The execute path
  still expands fusion groups through the existing node executor, but it builds
  the full MLX lazy graph, releases dead fusion-group values at group
  boundaries, releases other dead intermediates in schedule order, and
  asynchronously enqueues batched materialization-boundary schedule items
  without copying intermediates to host. Backend programs are validated with
  `std.Io.Writer` diagnostics before executable creation so malformed value
  references, producer/last-use metadata, edges, fusion groups, recomputed
  fusion boundary inputs/outputs, PJRT output materialization invariants,
  unscheduled nodes, duplicate scheduled nodes, dependency-order violations,
  materialization boundary coverage/order, and schedule items fail during compile with
  `pass=backend-program-verify` detail instead of device execution. Compile-time
  constants stay resident as immutable device arrays and execute borrows those
  resident handles directly, only cloning if a constant itself is returned as a
  PJRT output. The backend interface exposes executable residency stats for
  resident constant count/bytes, successful execute count, and borrowed constant
  nodes, plus backend program shape stats for values, nodes, edges, schedule
  items, control-flow subprograms/scheduler descriptors, fusion groups, and materialization boundaries. It also reports
  compile-time planned releases, planned release bytes, peak live values, and
  peak live bytes, plus cumulative fusion-group executions, materialization eval
  batches/buffers, and released intermediate values so tests can catch
  accidental graph linearization or liveness regressions. The MLX backend
  regression test proves repeated execution reuses the same resident constants
  instead of uploading weights again. Elementwise arithmetic,
  extra unary/binary math (`atan2`, `cbrt`, `expm1`,
  `is_finite`, nearest-even and away-from-zero `round`), integer
  `popcnt`/`count_leading_zeros`, StableHLO `complex`, real/complex-input
  `real`/`imag`, logical/bitwise ops, compare/select for
  MLX-supported numeric dtypes,
  f16/bf16/f32 sum/max/min and pred and/or reductions, two-output
  max/index reductions for ZML-style argMax, f16/bf16/f32 matmul-like
  `dot_general` including attention score/context contractions that reshape and
  transpose RHS on device before MLX `matmul` when required, dtype casts, StableHLO `iota`, StableHLO `clamp`,
  shape/view ops, StableHLO `reverse`, dynamic slice, dynamic update-slice,
  constant edge/interior padding, single-axis, point-style, and batched general gather
  forms, single-axis, point-style, windowed, and batched StableHLO scatter set/add,
  f16/bf16/f32 StableHLO `reduce_window` sum/max and ZML-style two-output
  max/index windows through MLX pad/as_strided/sort graph
  fragments, canonical rank-3/4/5 StableHLO `convolution` through MLX
  `conv_general` with NCW/NCHW/NCDHW inputs and OIW/OIHW/OIDHW kernels
  (rank-3 lowers through a dummy MLX 2D spatial axis to avoid MLX's
  default-metallib 1D depthwise path in hermetic builds), and ascending/descending
  StableHLO `sort`, StableHLO `complex` f32-to-c64 construction and c64-to-f32
  `real`/`imag`, plus StableHLO FFT metadata for c64 `fft`/`ifft`, f32-to-c64
  `rfft`, and c64-to-f32 `irfft`, and f32 dense StableHLO `cholesky` plus
  `triangular_solve` through backend-native MLX Metal kernels, now run through MLX-owned device operations on the GPU device when
  MLX arrays and Metal devices are available. More exotic scatter metadata
  and multi-output window reductions, non-canonical convolution dimension
  numbers, batch-grouped convolution, convolution window reversal, and broader
  tiled/batched linear-algebra specializations still
  require broader backend legalization before they enter the fast path.
  The PjRTx C shim no longer builds direct Metal arithmetic kernels or calls
  host `xcrun`.
- `src/compiler`: compile-option parsing and executable-plan construction for
  replicas, partitions, device assignment, and Shardy metadata. Program
  ingestion now flows through `std.Io.Reader`, parses and verifies MLIR through
  the MLIR C API, walks `func`, `stablehlo`, and `sdy` operations/attributes
  directly, records supported StableHLO ops, constructs parameter/output
  sharding plans from Shardy C attributes, lowers the first bootstrap execution
  ops into a PjRTx executable plan (parameter alias returns remain
  zero-instruction plans whose outputs reference parameter values directly,
  arithmetic plan instructions for StableHLO add/subtract/multiply/divide/negate, f32
  exp/tanh/sqrt/rsqrt, shape metadata for StableHLO reshape, transpose
  permutations, broadcast dimensions, slice start/limit/stride metadata from
  MLIR DenseI64ArrayAttr, concatenate dimensions from MLIR integer attributes,
  and the ZML-declared heavy/control/random/structural op shells including
  `cholesky`, `custom_call`, `optimization_barrier`, `partition_id`,
  `reduce_precision`, `rng`, `rng_bit_generator`, `scatter`, `tuple`, and
  `while`). PjRTx IR values now carry explicit structured storage for tuple and
  complex-pair values, and region-bodied ops such as reduce/sort/scatter/while
  carry `PlanRegion` subprograms with local region SSA values, operand/result
  ids, region kind, argument/return descriptors, terminator operand ids and
  descriptors, and a compact nested instruction summary.
  StableHLO `while` is normalized at the frontend boundary by splitting the
  C API's region/block representation into PjRTx cond/body regions, giving
  later staged passes a real region model without keeping MLIR C API objects
  alive. The compiler emits precise diagnostics
  through `std.Io.Writer` for unsupported ops, unsupported GSPMD custom-call
  targets, GSPMD shardings, CHLO/shape
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
  execution supports: parameter-alias returns as zero-instruction backend
  programs, linear
  StableHLO arithmetic chains execute the bootstrap `u8` and `f32` elementwise
  paths for matching buffers, StableHLO reshape preserves bytes while updating
  buffer dimensions and typed MLX metadata, StableHLO `iota` materializes
  coordinate grids on device, StableHLO transpose performs dense row-major
  layout permutation, StableHLO broadcast-in-dim expands dense buffers with
  explicit output dimensions, StableHLO slice performs dense strided slicing
  with explicit bounds, StableHLO `reverse` flips compiled axes, StableHLO
  `cbrt`, `round_nearest_afz`, `popcnt`, and `count_leading_zeros` lower to
  MLX-owned Metal custom kernels on resident device buffers, StableHLO
  `complex` materializes c64 tensors from f32 real/imaginary operands through
  MLX-owned device ops, StableHLO `real`/`imag` use MLX extraction for c64
  inputs and real-input identity/zero behavior, StableHLO
  concatenate joins two dense buffers along the compiled dimension, StableHLO
  `sort` uses the comparator direction parsed from the StableHLO region,
  key/value sort lowers through MLX `argsort` plus device-side
  `take_along_axis`, ZML-style two-output max/index reductions lower to an
  MLX max/equality/min-index graph, CHLO/StableHLO top-k composites lower to a device-only
  sort/argsort/reverse/slice plan, StableHLO `reduce_precision` is an identity
  device copy, StableHLO `optimization_barrier` clones each operand on device,
  StableHLO `tuple` is a structural IR node with no backend buffer allocation,
  and StableHLO `get_tuple_element` forwards the selected resident device value
  while preserving liveness for hidden tuple-element dependencies,
  StableHLO `reduce_window` lowers f32 one-input/one-output sum and max windows
  with static dimensions, strides, padding, unit base dilation, and MLX lazy
  graph materialization at the compiled output boundary,
  StableHLO `convolution` lowers canonical rank-3/4/5 floating-point
  NCW/NCHW/NCDHW by OIW/OIHW/OIDHW programs to MLX `conv_general`, including
  feature groups, static strides, padding, and input/kernel dilation,
  StableHLO `fft` lowers one-to-three innermost FFT dimensions for c64
  FFT/IFFT, f32-to-c64 RFFT, and c64-to-f32 IRFFT through MLX FFT,
  metadata-only `custom_call` target `annotate_device_placement` is treated as
  an identity on device, and registered PjRTx custom calls can lower to MLX
  identity/unary/binary graph fragments by target name. The built-in target
  `pjrtx.mlx_metal.custom_binary_add_f32` lowers to an MLX Metal custom kernel,
  giving the backend a real device-code custom-call path independent of Python
  registration. Custom-call
  registration is exposed through the JAX/XLA PJRT GPU custom-call extension
  (`PJRT_Gpu_Custom_Call` on `PJRT_Api.extension_start`) and currently accepts
  PjRTx MLX executable handler markers for identity, unary sqrt, and binary add.
  The shared library also exports test/embedder convenience shims
  `PjRTx_RegisterCustomCallIdentity`,
  `PjRTx_RegisterCustomCallUnary`, `PjRTx_RegisterCustomCallBinary`, and
  `PjRTx_UnregisterCustomCall`; opaque handlers, unsupported targets, or
  unregistered targets fail during backend legalization with the target name in
  the diagnostic rather than falling back to host execution. `bitcast_convert`
  uses MLX dtype views for dense byte-preserving tensor reinterpretation.
  `partition_id` materializes scalar partition ids through the MLX backend
  program path, deprecated StableHLO `rng` lowers uniform/normal distributions
  through MLX random on device, deterministic
  bootstrap `rng_bit_generator` paths use MLX-owned Metal kernels, and
  the StableHLO u64-state `THREE_FRY` path byte-matches JAX CPU for
  u8/u16/u32/u64 `jax.lax.rng_bit_generator` outputs without leaving the device execution path. f32 dense
  StableHLO `cholesky`/`triangular_solve` lower through backend-owned MLX Metal
  kernels without a runtime host fallback. These linalg kernels are unblocked
  correctness kernels first; tiled/library-grade implementations are the next
  performance step.
  Region/control
  operations that need real staged lowering (`while`, tuple-valued PJRT
  outputs, multi-output `reduce_window`, general scatter,
  unregistered or opaque custom calls) fail with explicit
  `UNIMPLEMENTED` feature diagnostics instead of leaking through buffer-level
  execution. `while` legalization now validates the loop-state descriptor
  contract against captured cond/body subprogram dataflow, and the backend
  program now owns a `while_loop` scheduler descriptor for the node. The MLX
  shim now exposes device-side bounded f32 compare/update loop primitives,
  including `<`/`+` and `>`/`-` forms, and the backend matcher now lowers the
  corresponding single-state StableHLO `while` region patterns with resident
  loop constants. Other `while` forms still fail with `mlx-while-region-pattern`;
  host-loop fallback remains disabled. GSPMD custom-call targets such as `Sharding`,
  `SPMDFullToShardShape`, and `SPMDShardToFullShape` stay unsupported by
  design; PjRTx's sharding path remains Shardy metadata.
  `PJRT_Client_Create` accepts `pjrtx_backend=metal_mlx`; other backend names
  are rejected. `PJRT_Client_Compile` accepts
  the bootstrap text compile
  options form used by the compiler tests:
  `replicas=2; partitions=2; use_shardy=true; assignment=0,1,2,3`.
  `PJRT_Device_MemoryStats` reports total device pressure from both live
  buffers and resident cached backend executables, while runtime memory stats
  keep buffer bytes, executable-cache bytes, and pressure-trim counters
  separately inspectable. PJRT lifecycle tests also cover the loaded-executable
  delete path that releases an active graph reference while keeping an idle
  resident executable cache entry reclaimable by the next compile under memory
  pressure. The PJRT stats response also fills the supported
  optional fields for peak bytes, live allocation count, largest allocation,
  memory limit, largest free block, and reservable limit when the backend
  reports a nonzero capacity.
  `PJRT_LoadedExecutable_Delete` is idempotent from the user perspective: it
  marks the handle deleted, releases the loaded executable graph/backend cache
  reference immediately, rejects future execute calls, and leaves metadata alive
  until `PJRT_LoadedExecutable_Destroy`.
  `PJRT_LoadedExecutable_Execute` validates its ABI boundary before reading C
  arrays: device count must be nonzero and within the loaded executable graph,
  argument count must exactly match executable parameters, required per-device
  argument/output lists must be non-null, and null argument buffers fail with
  `INVALID_ARGUMENT`. Donated parameters are validated before dispatch as well:
  a donated buffer cannot alias any other execute argument unless that parameter
  index is explicitly listed as non-donatable in PJRT execute options. Execute
  initializes expected output and completion-event slots to null before
  dispatch, asks runtime graph execution for both outputs and a completion
  event, and adapts that runtime event into PJRT per-device completion events.
  Execute output buffer readiness is chained to that device completion event,
  so future async backend completion can use the same PJRT buffer/event
  contract. Backends expose pending execution event status/destruction through
  the backend vtable; until runtime scheduler integration owns polling/callback
  progression, unresolved pending backend events fail closed instead of
  leaking handles or marking outputs ready. If execution fails after producing
  partial results it
  destroys those PjRTx-owned partial outputs/events before returning the PJRT
  error.
  Runtime graph execution also validates placed arguments against the selected
  device slot: argument count must exactly match executable parameters, stable
  device id must match the graph device assignment, shard index must match
  the per-device execute list index, and argument buffers must have ready
  producer events. Buffer constructors reject device/memory
  pairs where the memory is not addressable by the requested device, and PJRT
  host buffer creation derives the shard index from the device's client slot
  rather than assuming stable device ids are dense array indexes.
  Optional `PJRTX_TRACE=1` instrumentation prints `pjrtx_trace` lines for
  compile, compile-cache, H2D, execute, loaded-executable delete, and D2H events, including byte counts, device count,
  backend-executable eligibility, resident constant count/bytes, backend
  program value/node/edge/schedule/subprogram/fusion/materialization counts, backend
  planned release count/bytes, peak live value count/bytes, backend execute count,
  backend program device count, last dispatched device index/local hardware id,
  fusion-group execution count, materialization eval count,
  released-intermediate count, borrowed constant node count, executable
  fingerprint cache hit/miss totals, backend executable reuse status, resident
  executable-cache entry/byte/peak-byte counters, cache eviction count, and
  evicted resident executable bytes, executable-cache compile latency samples,
  total compile latency, peak compile latency, compile/H2D/execute
  pressure-trim bytes/eviction counts, remaining cache pressure,
  pressure-failure flags, and execute backend-completion pending counts, and
  elapsed microseconds.
  `PJRTX_EXECUTABLE_CACHE_MAX_BYTES=0` is useful for
  stress-testing eviction: fingerprint metadata still records hits, but evicted
  backend executables are rebuilt and report `backend_cache_reuse=0`.
  PJRT event callbacks bridge through runtime event registrations: completed
  events invoke immediately, pending events hold a bounded callback slot until
  the runtime marks them ready or failed, and failed events pass owned PJRT
  errors to the callback. H2D, D2H, execute completion, and buffer-ready events
  now all go through the same runtime event constructors, with buffer-ready
  events mirroring the buffer tombstone state after delete or donation and
  execute output readiness chained from the per-device completion event. Execute
  also honors explicit executable-plan donation
  metadata: successfully donated input buffers are marked unusable after
  execution, while PJRT `non_donatable_input_indices` vetoes donation for the
  listed argument indices. Executable cache keys now cover the source artifact,
  backend capabilities, resolved target device hardware/memory traits, compile
  options, device assignment, sharding metadata, output ids, donation metadata,
  value descriptors/structured storage/placement, and instruction payloads.

The plugin target currently produces:

```text
bazel-bin/src/plugin/libpjrtx_metal_plugin.dylib
```

## Next Implementation Steps

- Broaden MLX backend executable legalization. Runtime execution is device-only,
  so unsupported StableHLO ops must either lower to MLX graph fragments or fail
  at compile time with precise diagnostics. Current backend legalization
  diagnostics identify the blocking pass, op/value, shape, sharding, and MLX
  feature label.
- Expand the MLX backend implementation for the remaining LLM hot path:
  richer custom comparator sort semantics, top-k fast paths using MLX library
  primitives where they expose indices directly, and custom-call hooks where
  MLX exposes the right primitive.
- Add a backend legalization/pipeline stage to turn PjRTx value graphs into
  per-device command fragments: memory placement, layout, tiling/shard planning,
  async dependencies, forced materialization boundaries, and final backend
  kernel/library dispatch. The runtime has no CPU reference executor; tests may
  compare device results against external CPU oracles, but PjRTx buffers and
  executable paths stay backend-device backed.
- Add focused MLX backend conformance tests for buffer/execution semantics
  through the vtable, with extra tests asserting MLX chains do not copy device
  intermediates back to host.
- Start using the budget fixture to compare fusion and scheduling changes:
  every MLX graph/fusion change should preserve strict transfer/residency
  budgets and improve or hold warm steady-state latency.
- Keep improving cached backend executable policy: the runtime now has
  resident-byte accounting, an environment-configurable cap, idle-entry
  eviction, pressure-aware largest-resident-entry victim selection with LRU
  and rebuild-cost tie-breaking, plus evicted-byte and compile-latency
  telemetry. Host imports, resident executable compilation, and backend graph
  dispatch now provide allocation pressure feedback so idle resident
  executables can be reclaimed before activation imports, newly compiled
  constants, or execute output buffers are accounted. It still needs production-grade ranking across
  multiple target memories and backend-specific pressure signals beyond the
  PJRT-visible memory stats fields.
