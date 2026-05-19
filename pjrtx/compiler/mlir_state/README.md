# MLIR State

Owns the temporary MLIR state-machine bridge for V0: module state transitions,
bounded attribute encodings, Zig external pass callbacks, and typed extraction
from committed MLIR facts.

This package does not own StableHLO import policy, backend execution, runtime
allocation calls, or report formatting. `//pjrtx/compiler` orchestrates the
compile pipeline; this package commits, verifies, and extracts MLIR-visible
facts for that pipeline.

Implementation files are split by intent:

- `types.zig`: public state-machine and extracted fact types
- `limits.zig`: bounded array/dictionary limits for temporary attributes
- `lowering_policy.zig`: V0 lowering decisions and rejected alternatives
- `attrs.zig`: small MLIR C API attribute helpers shared by pass families
- `passes.zig`: MLIR pass-manager lifecycle and pass-run result boundary
- `state_target_passes.zig`: state-probe and target-legality external pass callbacks
- `fusion_passes.zig`: fusion-candidate discovery and fusion-decision external pass callbacks
- `placement_collective_passes.zig`: placement-record and collective-plan external pass callbacks
- `package.zig`: package façade plus remaining session, pass callback,
  commit/verify/extract logic while the larger split continues

The final destination is real PjRTx MLIR dialect ops/attrs/passes. Until then,
all temporary attribute facts must stay bounded, verified, and extracted before
other packages consume them.
