# `//next/pjrtx/compiler/passes`

Intent: own Zig-written MLIR external passes and pass-local policy.

Owns:

- Zig external pass callbacks through the MLIR C API
- state transition checks that are expressible through MLIR C handles
- pass-local diagnostics through `std.Io.Writer`
- pass-local extraction probes used by tests

Does not own:

- whole compile orchestration
- dialect parser/printer/verifier implementation
- backend/runtime execution
- report rendering

Allowed dependencies:

- Zig `std`
- `//next/pjrtx/mlir`
- `//next/pjrtx/dialects`
- `//next/pjrtx/compiler/facts`
- `//next/pjrtx/target` when target policy is needed

Primary outputs:

- registered external passes
- pass result facts
- verifier diagnostics

