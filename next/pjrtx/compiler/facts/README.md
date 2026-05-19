# `//next/pjrtx/compiler/facts`

Intent: own extracted compiler views after MLIR verification.

Owns:

- source refs and graph views
- compiler pass records
- fusion, placement, tile, memory, collective, lowering, cost, traffic,
  codegen, schedule, backend-binding intent, executable readiness, and explain
  views
- validators for compiler-owned extracted facts
- deinit and deterministic formatting for compiler facts

Does not own:

- MLIR session lifetime
- target hardware model
- backend executable calls
- runtime allocation or profile events
- report orchestration

Allowed dependencies:

- Zig `std`
- `//next/pjrtx/target`

Primary outputs:

- compact extracted compiler fact structs
- compiler fact validators
- fact-local summaries for debugging

Live V0 slice:

- source IDs and source refs
- graph value and instruction IDs
- tensor type and layout facts
- graph value and graph instruction records
- StableHLO-shaped graph payload specs
- tensor validation and payload-kind checks
- MLIR pass and graph rewrite records
- fusion, placement, and collective plan records
- lowering, cost ledger, and memory-traffic records

Direct consumers:

- `//next/pjrtx/compiler`
- `//next/pjrtx/backend`
- `//next/pjrtx/runtime`, as immutable graph inputs for allocation/profile planning
- `//next/pjrtx/vertical_slice`

Temporary bridge:

- `//next/pjrtx/core` re-exports these names only while the legacy `TraceReport`
  container and report writer remain there.
