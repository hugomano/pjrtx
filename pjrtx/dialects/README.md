# `//pjrtx/dialects`

Intent: expose real PjRTx MLIR dialect extension points with minimal
MLIR-native code.

Owns:

- TableGen dialect definitions
- operation, type, and attribute declarations
- parser/printer/verifier hooks
- canonicalization and dialect-conversion glue when MLIR requires C++ APIs
- narrow C-compatible registration and pass construction functions

Does not own:

- PjRTx target policy
- cost models
- schedule policy
- runtime execution
- report rendering

Allowed dependencies:

- MLIR C++/TableGen targets
- StableHLO/Shardy dialect targets only when dialect integration needs them

Primary outputs:

- PjRTx dialect library
- generated op/type/attr headers
- `pjrtx_c_api` surface consumed by Zig

