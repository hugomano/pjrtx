# `//next/pjrtx/runtime/facts`

Intent: own extracted runtime views after runtime planning/profiling.

Owns:

- runtime buffer IDs
- allocation reservations
- command buffer uses
- buffer lifetimes
- stream steps and event IDs
- runtime profile events
- runtime profile joins

Does not own:

- compiler legality
- backend codegen
- target hardware definitions
- report rendering

Allowed dependencies:

- Zig `std`
- `//next/pjrtx/compiler/facts`
- `//next/pjrtx/backend/facts`
- `//next/pjrtx/target`

Primary outputs:

- allocation facts
- stream/event facts
- profile event facts

