# Agent Guide For `src/runtime`

Read the root `AGENTS.md` and `CODING_POLICY.md` first. Runtime code owns PJRTx
lifecycle and scheduling for concrete Metal backends; it is not a compiler,
backend registry, generic abstraction layer, or CPU reference interpreter.

## Runtime Boundary

Runtime concrete backend imports must stay centralized in
`backend_selection.zig`. That module may import concrete backend package facades
such as `src/backend/mlx_metal` and `src/backend/metalcpp`, and select between
them with `PJRTX_RUNTIME_BACKEND` / `PJRTX_BACKEND`. Other runtime modules must
consume the closed `Backend` union from `backend_selection.zig`; do not recreate
a vtable, registry, root backend facade, or scatter environment reads.

Runtime must not import MLX C symbols, Metal shims, PJRT C structs,
StableHLO/MLIR C APIs, or backend internals.

Keep responsibilities strict:

- `runtime.zig`: package root only. It may import domain modules and re-export
  public runtime contracts, but it should not own lifecycle, cache, residency,
  or execution implementation bodies.
- `backend_selection.zig`: closed concrete-backend selection and method
  dispatch. It owns the runtime env knob and no backend implementation details.
- `client.zig`: client lifecycle, topology ownership, compile orchestration.
- `device_memory.zig`: devices, memories, topology, memory stats.
- `event.zig`: readiness state, callbacks, dependency chaining.
- `buffer.zig`: buffer lifecycle, placement, donation, backend handle ownership.
- `executable_cache.zig`: executable fingerprints, residency, trimming, stats.
- `executable.zig`: compiled executable metadata and owned residency lifetime.
- `execution.zig`: per-device dispatch, donation aliasing, output wrapping.
- `custom_call.zig`: runtime custom-call registration API.

Do not add broad files such as `types.zig`, `state.zig`, `backend.zig`,
`common.zig`, or `helpers.zig`. If a module name does not describe an owned
runtime invariant, it is probably the wrong module.

## No Host Fallback

Runtime execution is device-driven. Do not add CPU fallback, host shadow
execution, host scalar readback for control flow, synthetic devices, or
instruction interpreters. Unsupported programs must fail at compile/lowering
time with diagnostics from the owning compiler/backend layer.

Runtime buffers own placement and backend handles, not StableHLO operation
lowering. If an operation needs lowering, put it in compiler IR planning or the
Metal/MLX backend program compiler.

## Parallel Refactor Rules

`runtime.zig` is now the package root. It should stay as imports and re-exports
only; do not add tests, fixtures, lifecycle code, or local helper functions to
it.

Parallel agents must receive disjoint write sets:

- `buffer.zig`: buffer state, placement, backend buffer handles, buffer
  constructors, and buffer tests.
- `execution.zig`: per-device dispatch, output wrapping, completion-event
  adaptation, donation aliasing, and execution tests.
- `executable.zig`: compiled executable metadata, schedule construction, backend
  residency options, backend executable handles, and residency lifetime tests.
- `client.zig`: client lifecycle, topology/device enumeration, compile
  orchestration, executable-cache locking, async transfer entrypoints, and
  client/cache tests.
- `executable_cache.zig`: cache entry accounting, residency policy, trim
  policy, and cache-only tests.

If a change seems to require editing two owner modules, first decide which module
owns the behavior and expose a narrow method from that owner. Do not create a
new shared helper file just to make a parallel edit convenient.

## Public APIs

Default to private. Every public runtime API needs a Zig doc comment explaining
intent, ownership, and invariants. Public fields are commitments; expose methods
when a caller only needs behavior.

Keep aliases temporary and explicit. Re-export compiler-owned tensor vocabulary
only while downstream users are being migrated, and document that compiler IR is
the source of truth.

## Synchronization And IO

Use Zig 0.16 primitives directly: `std.Io.Reader`, `std.Io.Writer`,
`std.Io.Timestamp`, and `std.Io.Mutex`. Do not create ad hoc C clocks or naked
global flags. Runtime-wide lazy state must be guarded by a real owner object.

Prefer passing an owning runtime/client IO handle through APIs once that exists;
do not create fresh process-global IO objects deep in helper functions.
