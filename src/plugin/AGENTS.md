# Agent Guide For `src/plugin`

This directory implements the PJRT C API adapter for PjRTx. Read the root
`AGENTS.md` and `CODING_POLICY.md` first; this file adds plugin-specific PJRT
reference guidance.

## PJRT References

Use these OpenXLA/XLA documents as references for PJRT semantics, ABI shape,
lifecycle expectations, and integration vocabulary:

- [`pjrt_c_api.h`](https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api.h)
- [`pjrt_client.h`](https://github.com/openxla/xla/blob/main/xla/pjrt/pjrt_client.h)
- [`xla/pjrt/c/README.md`](https://github.com/openxla/xla/blob/main/xla/pjrt/c/README.md)
- [`cpp_api_overview.md`](https://github.com/openxla/xla/blob/main/docs/pjrt/cpp_api_overview.md)
- [`pjrt_integration.md`](https://github.com/openxla/xla/blob/main/docs/pjrt/pjrt_integration.md)

These documents are reference material only. They are not implementation
templates for PjRTx.

Be careful:

- Do not copy XLA's internal C++ architecture into this Zig plugin.
- Do not mirror XLA's implementation layering when it conflicts with PjRTx's
  runtime/compiler/backend boundaries.
- Do not add GSPMD, XLA service, or host fallback concepts just because they
  appear in the reference code.
- Do not make plugin modules depend on XLA implementation headers or C++
  classes.
- Do use the PJRT C API header as the ABI contract for struct fields, callback
  names, extension chains, ownership rules, and version compatibility.
- Do use PJRT client docs to understand semantic expectations around clients,
  devices, memories, buffers, events, executables, loading, execution, donation,
  and async behavior.

When local sources are available, prefer the checked-out XLA tree that this repo
is built against for exact ABI details. The GitHub links above are for
orientation and review.

## Plugin Boundary Reminder

The plugin is a translation layer:

```text
PJRT C ABI -> PjRTx runtime API -> backend-owned execution
```

Keep raw PJRT structs and nullable C pointer handling at the callback boundary
or in `pjrt_abi.zig`. Decode into Zig request/view structs immediately, then
call runtime APIs. If a change needs compiler or backend behavior, add that
behavior to the owning layer and keep the plugin as the adapter.

