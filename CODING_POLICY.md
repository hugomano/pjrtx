# PjRTx Coding Policy

This policy applies to the whole PjRTx codebase. PjRTx should feel like a
Zig-first accelerator runtime/compiler project with clear ownership, not a pile
of bindings, helper files, and accidental layers.

The dependency graph should tell the architecture story. When someone opens the
repo or reads Bazel targets, they should see the flow of responsibility:

```text
PJRT ABI adapter -> runtime lifecycle/scheduling -> compiler plans -> backend execution
```

Each package and file should have a reason to exist in that story.

## Architecture Boundaries

Keep boundaries crisp:

- PJRT adapter code translates PJRT ABI calls to PjRTx runtime APIs.
- Runtime owns clients, devices, memories, buffers, events, executable lifetime,
  scheduling, donation, readiness, and memory accounting.
- Compiler owns program ingestion, PjRTx IR, verification, sharding/layout
  planning, legalization, diagnostics, and executable plans.
- Backend code owns MLX/Metal integration, device execution, backend programs,
  resident constants, custom-call execution, and backend capabilities.
- Compiler-owned IR modules own shared tensor vocabulary and executable-plan
  concepts. Do not recreate `src/core`, `shared`, or `common` packages for
  cross-layer convenience.

Do not let concerns leak upward:

- Plugin/PJRT code must not import MLX, Metal, StableHLO, MLIR, or backend
  internals.
- Runtime should not parse StableHLO or know compiler syntax.
- Compiler should not allocate runtime buffers or know backend object handles.
- Backend-specific details should not appear in compiler or PJRT adapter code
  unless they are expressed through an explicit PjRTx-owned capability or plan.

When a layer needs a new capability, add a real API at the owning layer. Do not
patch around it from the caller.

## File Shape

File names should describe domain ownership, not implementation accidents.
Avoid generic buckets such as:

- `types.zig`
- `state.zig`
- `utils.zig`
- `helpers.zig`
- `common.zig`
- `misc.zig`

These names hide ownership and become dumping grounds. Prefer files named after
the thing that owns invariants or behavior:

- `buffer.zig`
- `event.zig`
- `device_memory.zig`
- `executable_cache.zig`
- `compile_options.zig`
- `stablehlo_import.zig`
- `mlx_program.zig`
- `pjrt_abi.zig`

Small package roots may re-export or compose modules, but they should not grow
mixed implementation bodies. If a file accumulates unrelated IDs, shape facts,
passes, callbacks, and backend details, split it before adding more behavior.

Existing generic files are legacy debt, not precedent. When touching one, move
new behavior toward a domain-owned module and shrink the generic file.

## File Order And Naming Harmony

Keep every file sorted in a predictable, domain-first order. This is a coding
standard, not a formatting preference. A reader should be able to scan a file
from top to bottom and see ownership before mechanics:

- imports, grouped from standard library to foreign ABI to local owner modules
- file-level constants and handle aliases
- public owned domain types, with public methods near the owned state
- private borrowed references or decoded request structs
- operation enums and callback generators
- lifecycle/callback implementations
- exported package API tables at the end
- colocated tests after implementation

Do not interleave unrelated callback families, request decoders, tests, and
public state just because it avoids moving text. When a file has multiple
related scopes, keep each scope internally ordered the same way: fields, `Api`,
constructors/decoders, validation, writers, lifecycle. Tests live after the
implementation they verify, not in the middle of callback machinery.

Use the same vocabulary for the same role across the codebase:

- `Owned` or the domain noun for values that own storage
- `Ref` for a borrowed decoded handle/reference
- `Request` or `Call` for decoded PJRT entrypoint arguments
- `Handle` only for opaque ABI handle adapters
- `Api` only for callback groups installed into `PJRT_Api`

Avoid vague role names such as `View`, `Data`, `Info`, `Helper`, `Manager`, or
`State` unless the domain or external protocol itself uses that noun and the
type owns a precise invariant. A name should explain what owns the value, not
how convenient it was to pass around. For example, PJRT's
`AsyncHostToDeviceTransferManager` may keep the protocol word `Manager`, but a
local decoded transfer payload should be named for its domain, such as
`ByteTransferRequest`, not `Data`.

A file may only expose public APIs for the domain named by the file. Do not put
callbacks, structs, or constants in a nearby module just because it already has
the imports you need. For example:

- plugin metadata callbacks belong with plugin metadata, not error handling
- executable ownership belongs with executable lifecycle, not process state
- buffer element-type conversion belongs with buffer typing, not a generic
  `types.zig`
- opaque runtime handle aliases belong in a handle-boundary module, not the raw
  ABI mechanics module

Boundary modules should be honest about their level. A raw ABI helper such as
`pjrt_abi.zig` must not import runtime, compiler, backend, MLX, or Metal code.
If a helper needs runtime types, it is no longer raw ABI; move it to a
runtime-aware adapter such as `pjrt_handles.zig` or a domain-specific placement
module.

API table composition should also tell the truth. Install each PJRT prefix once
from one installer call. If implementation lives in multiple domain modules,
pass those owner scopes as a tuple to the generic installer. Do not hand-forward
individual callbacks in `api.zig`, and do not create facade files that exist
only to re-export callbacks from real owner modules.

## Zig Style

Prefer owning structs with methods over loose package-level helper functions.
Code should read as objects and domains doing their own work:

```zig
const Buffer = struct {
    ptr: *runtime.Buffer,

    fn ensureReady(self: Buffer) !void {
        try self.ptr.ensureReady();
    }
};
```

Use scoped namespaces when they clarify ownership:

```zig
pub const Api = struct {
    pub const Destroy = BufferCallback(c.PJRT_Buffer_Destroy_Args, .destroy).call;
};
```

Avoid helper soup:

- No scattered `toC` / `fromC` wrappers.
- No broad `makeThing` helpers when a typed `Thing.init` is clearer.
- No local clones of standard library functionality.
- No hidden global state except guarded registries or process-wide constants
  with explicit ownership.

Use Zig features deliberately:

- `comptime` for callback tables, type-driven rendering, and repeated ABI
  shapes.
- Tagged unions and enums for domain variants.
- Explicit error sets for expected failures.
- `errdefer` and `deinit` for visible ownership.
- `std.Io.Reader` for streamed program/data ingestion.
- `std.Io.Writer` for diagnostics, traces, reports, and dumps.
- Zig 0.16 standard-library primitives before local inventions:
  `std.Io`, `std.Io.Timestamp`, `std.Io.Mutex`, bounded arrays, writers,
  readers, and allocator-aware containers should be the default vocabulary.
- Process-wide mutable state must be guarded with `std.Io.Mutex` or a
  stricter Zig 0.16 synchronization primitive owned by the module. Do not use
  naked boolean init flags for API tables, plugin contexts, registries, or
  lazily initialized attributes.
- Initialize IO mutexes with `std.Io.Mutex.init` and pass the owning layer's
  `std.Io` to lock/unlock calls. For non-cancelable C ABI entrypoints, prefer
  `lockUncancelable(io)` when holding a process-lifetime guard.
- Reuse the plugin/runtime IO objects exposed by the owning layer instead of
  creating ad hoc writers, clocks, or C time shims at call sites.
- Only the process/plugin initialization root should create the first
  process-lifetime `std.Io` handle. All other plugin modules should call
  `plugin.io()` or the owning layer's IO accessor.

When Zig 0.16 has a standard feature for a job, prefer it. A local abstraction
is justified only when it gives a domain name to an invariant, not because it
hides ordinary standard-library usage.

## Public API And Visibility

Be restrictive with `pub`. Public APIs, public structs, public fields, and
public declarations are architectural commitments. Make a declaration public
only when another package or boundary is supposed to depend on it.

Prefer private by default:

- private helper functions
- private struct fields
- private nested types
- package-local aliases instead of public re-exports
- narrow public methods instead of exposing mutable fields

Every public API must have a Zig doc comment that explains intent and contract,
not implementation mechanics. The comment should answer why this API exists,
what owns the value, and what invariants or boundary expectations callers must
respect.

Good:

```zig
/// Describes a device-resident executable that may borrow cached constants.
/// Callers own the returned handle and must release it with `deinit`.
pub const LoadedExecutable = struct { ... };
```

Bad:

```zig
/// Struct with executable fields.
pub const LoadedExecutable = struct { ... };
```

Do not make a type or field public to make tests easier. Prefer testing through
the intended API, or add a narrow test-only helper in the owning module when the
invariant genuinely needs direct coverage.

## ABI And Foreign Boundaries

C, C++, Objective-C, MLIR C API, PJRT C API, and MLX/Metal details should live
at explicit boundary modules.

Inside a boundary module:

- Decode raw pointers once.
- Convert to Zig slices, enums, and typed views.
- Keep nullable C pointer handling close to the callback or FFI entrypoint.
- Return precise errors with enough context to debug the caller.

Outside the boundary module:

- Use PjRTx-owned types.
- Do not pass raw C structs through multiple layers.
- Do not expose backend handles to compiler or PJRT adapter code.

## Diagnostics And Tracing

Diagnostics are product surface. They should identify:

- pass or layer name
- operation/value when relevant
- dtype, rank, shape, layout, sharding, or memory space when relevant
- backend feature label when a backend cannot lower something
- exact PJRT API state when a lifecycle call fails

Tracing should be structured and automatic where possible. Do not scatter
one-off prints through hot paths. Add a trace hook at the boundary or scheduler
that owns the event.

Use `std.Io.Timestamp` and `std.Io.Writer` for trace rendering. Avoid direct
platform timing APIs unless the backend boundary requires them.

## Runtime And Backend Path

PjRTx is device-driven:

- No CPU fallback in runtime or plugin paths.
- No host shadow execution.
- No host predicate readback to drive control flow.
- No repeated weight upload when executable residency should own constants.
- No backend-specific shortcuts above the backend boundary.

Tests may compare device results against external CPU oracles, but PjRTx
execution itself must stay on the backend path.

## Ownership

Every allocation needs an obvious owner and release path:

- Constructors should say who owns returned memory.
- `deinit` should release all owned memory and backend/runtime handles.
- `errdefer` should clean partially initialized values.
- Borrowed slices should be shaped so their owner is obvious from context.
- Cache residency and device memory accounting must be updated where ownership
  changes, not later as an afterthought.

Prefer small structs with clear invariants over large structs where half the
fields are optional because multiple phases were mixed together.

## Tests

Keep tests close to the code they validate. Use integration tests only at real
boundaries: PJRT, runtime/backend, compiler import/lowering, or JAX/ZML.

For broad changes, run:

```sh
bazel test //src/... --test_output=errors
bazel test //... --test_output=errors
```

For PJRT/JAX-facing changes, also run:

```sh
bazel test //src/plugin:plugin_test --test_output=errors
bazel test //src/plugin/jax:jax_plugin_smoke_test //src/plugin/jax:jax_op_suite_test --test_output=errors
```

For performance or ZML behavior changes, run the specific ZML/JAX fixture named
by the task. Do not substitute CPU tests for plugin-only behavior.

## Refactor Checklist

Before adding or changing code, ask:

- What layer owns this behavior?
- Does the file name explain that ownership?
- Does the dependency edge preserve the architecture?
- Can the dependency graph still be explained in one sentence?
- Is C/MLX/MLIR/PJRT ABI handling confined to the right boundary?
- Is ownership explicit through init/deinit/errdefer?
- Are diagnostics precise enough for the next failure?
- Are tests colocated or covered by a real boundary test?

If the answer is unclear, refactor the boundary first. Do not add new behavior
on top of a shape that already hides ownership.

## Codebase Critique Standard

Critique code by naming the architectural failure mode, not by giving a
taste-only judgment. Useful critique is concrete enough to guide the next patch.

Use this vocabulary:

- Boundary leak: a layer imports or knows another layer's private concerns.
- Ownership blur: allocation, lifetime, residency, or `deinit` responsibility
  is unclear.
- Generic bucket: behavior is hidden in `types.zig`, `state.zig`, `utils.zig`,
  `helpers.zig`, or a similar catch-all file.
- Helper soup: behavior is spread across loose free functions instead of owned
  methods or scoped namespaces.
- Dependency lie: Bazel dependencies do not match the architecture story.
- Fallback smell: CPU or host behavior hides a missing backend/runtime
  capability.
- Diagnostic gap: errors do not name the failing pass, op, shape, API, or
  backend feature.
- Visibility leak: `pub` exposes data or behavior that should be private to the
  owner.
- Documentation gap: a public API lacks an intent comment or documents
  implementation details instead of contract.
- Domain mismatch: a public API lives in a file whose name describes a different
  owner.
- Boundary impurity: a low-level ABI or foreign-boundary module imports a
  higher-level runtime/compiler/backend package.
- Test mismatch: tests validate an implementation shortcut instead of the
  intended boundary.

When criticizing code, include:

- the concrete file or module involved
- the ownership or boundary being violated
- the smallest refactor that would improve the shape
- the tests needed to protect the new boundary

Avoid comments that only say code is ugly or sloppy. Say, for example:

```text
`client.zig` is mixing compile-option decoding with buffer placement. Buffer
placement belongs to a request/ref type owned by runtime-facing client code;
split that path and keep PJRT field decoding at the callback boundary.
```

Existing code is context, not permission. If current code violates this policy,
do not copy the pattern. Improve the touched path incrementally and leave a
clearer ownership edge than you found.
