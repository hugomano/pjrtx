# `//next/pjrtx/backend/facts`

Intent: own extracted backend views after compiler/backend verification.

Owns:

- backend executable calls
- backend kernel graph nodes and edges
- backend codegen descriptors
- backend profile joins

Does not own:

- target model definitions
- compiler backend-binding intent
- compiler schedule construction
- runtime allocations or streams
- Metal/MLS or NPU implementation details

Allowed dependencies:

- Zig `std`
- `//next/pjrtx/target`
- `//next/pjrtx/compiler/facts`

Primary outputs:

- backend executable views
- kernel graph facts
- backend profile join facts
