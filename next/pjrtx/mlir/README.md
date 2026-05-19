# `//next/pjrtx/mlir`

Intent: own Zig-side MLIR lifetime and IO plumbing.

Owns:

- MLIR context and dialect registry setup
- module ownership
- pass manager ownership
- deterministic MLIR dumps
- narrow Zig imports of C-compatible MLIR APIs

Does not own:

- PjRTx compiler policy
- PjRTx dialect definitions
- extracted report schemas
- backend or runtime execution

Allowed dependencies:

- Zig `std`
- MLIR/StableHLO/Shardy C API targets
- `//next/pjrtx/dialects` C-compatible registration APIs once they exist

Primary outputs:

- `MlirSession`
- pass-manager helpers
- deterministic dump/readback helpers using `std.Io.Writer`

