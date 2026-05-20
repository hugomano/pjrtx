# Agent Guide For PjRTx

Read `CODING_POLICY.md` before editing this repository. The important idea:
write idiomatic Zig whose packages and dependency edges explain the system.
Avoid generic buckets and keep work at the layer that owns it.

Some directories add narrower agent guidance. In particular, read
`src/plugin/AGENTS.md` before editing the PJRT C API adapter.

## Project Shape

PjRTx is a PJRT runtime/compiler/backend stack for Metal/MLX. The code should
make these boundaries obvious:

- PJRT adapter: ABI translation only.
- Runtime: lifecycle, scheduling, buffers, events, memory, executable residency.
- Compiler: program import, IR, verification, sharding/layout planning,
  diagnostics, executable plans.
- Backend: MLX/Metal execution, backend programs, resident constants,
  custom-call execution, capabilities.

Do not move behavior across those boundaries for convenience.

## Non-Negotiables

- Do not add CPU fallback, host shadow execution, or host-driven control flow.
- Do not let plugin/PJRT code import MLX, Metal, MLIR, StableHLO, or backend
  internals.
- Do not let compiler code allocate runtime buffers or depend on backend object
  handles.
- Do not use `types.zig`, `state.zig`, `utils.zig`, `helpers.zig`, or similar
  generic files as a place for new behavior.
- Do not scatter `toC` / `fromC` helpers. Put ABI mechanics in explicit boundary
  modules.
- Do not add manual trace prints when a structured trace hook belongs at a
  boundary.

Existing generic files are legacy debt. When touching them, prefer moving code
toward a domain-owned file instead of expanding the bucket.

## Preferred Code

Use owning structs and scoped APIs:

```zig
const Event = struct {
    ptr: *runtime.Event,

    pub const Api = struct {
        pub const Await = EventCallback(c.PJRT_Event_Await_Args, .await).call;
    };

    fn at(raw: anytype) Event {
        return .{ .ptr = abi.Event.view(raw) };
    }
};
```

Use small request/view structs when an entrypoint has meaningful state:

```zig
const ExecuteCall = struct {
    raw: *allowzero c.PJRT_LoadedExecutable_Execute_Args,
    executable: *Executable,

    fn init(raw: *allowzero c.PJRT_LoadedExecutable_Execute_Args) ExecuteCall {
        return .{ .raw = raw, .executable = LoadedExecutableHandle.view(raw.executable) };
    }
};
```

Prefer `Thing.init`, `Thing.deinit`, `Thing.validate`, `Thing.writeTo`, and
domain-specific methods over free-floating helpers.

## Public API Discipline

Default to private. Only use `pub` when the declaration is part of a real
package or boundary contract.

Be restrictive about:

- public structs
- public fields
- public helper functions
- public aliases and re-exports
- mutable state exposed across packages

Every public API needs a Zig doc comment. The comment should explain intent,
ownership, and invariants; it should not narrate the implementation. If you add
or expose a `pub` declaration, add or update the doc comment in the same patch.

Do not make internals public just to satisfy tests. Test through the intended
API, or add a very narrow test-only path in the owning module when necessary.

## File Naming

A file should name an architectural concept or owned invariant. Good names
sound like parts of the system:

- `buffer.zig`
- `event.zig`
- `compile_options.zig`
- `executable_cache.zig`
- `stablehlo_import.zig`
- `mlx_program.zig`
- `pjrt_abi.zig`

Bad names sound like storage bins:

- `types.zig`
- `state.zig`
- `utils.zig`
- `helpers.zig`
- `common.zig`

The Bazel dependency graph should read like a design document. If a new edge is
hard to justify, the code is probably in the wrong place.

## Before Editing

1. Inspect the owning package, BUILD visibility, and nearby tests.
2. Decide which layer owns the requested behavior.
3. If the current file is a generic bucket, avoid making it bigger.
4. Preserve public ABI and target names unless the user explicitly asks for a
   breaking change.
5. Keep behavior changes separate from structural cleanup when possible.
6. Use `apply_patch` for manual edits and `zig fmt` after Zig changes.

## How To Critique Before Refactoring

Before editing, classify the problem:

1. Is this code in the wrong layer?
2. Is the file name hiding ownership?
3. Is the dependency edge backwards or too broad?
4. Is C, MLX, MLIR, StableHLO, or PJRT leaking past its boundary?
5. Is state owned by a real object, or floating globally?
6. Is `pub` exposing something that should be private?
7. Is a public API missing an intent comment?
8. Is the test protecting intended behavior or protecting an accident?

Use concrete critique categories:

- Boundary leak
- Ownership blur
- Generic bucket
- Helper soup
- Dependency lie
- Fallback smell
- Diagnostic gap
- Visibility leak
- Documentation gap
- Test mismatch

Then choose one action:

- Local cleanup: rename, re-scope, or privatize helpers inside the same owner.
- Boundary refactor: move behavior to the owner layer and expose a narrow API.
- Stop and report: if the fix needs a larger design decision.

Critique the code shape, not the previous author. Be direct and useful:
identify the file, the violated ownership boundary, the smallest improvement,
and the tests that prove it. Do not say a file is bad without naming the
specific boundary problem and next refactor.

Existing code is context, not permission. If a current file violates the policy,
do not copy the pattern.

## Verification

For normal code changes:

```sh
bazel test //src/... --test_output=errors
```

For broad or boundary changes:

```sh
bazel test //... --test_output=errors
```

For PJRT/JAX-facing changes:

```sh
bazel test //src/plugin:plugin_test --test_output=errors
bazel test //src/plugin/jax:jax_plugin_smoke_test //src/plugin/jax:jax_op_suite_test --test_output=errors
```

If you cannot run a requested or relevant command, say exactly which command was
skipped and why.
