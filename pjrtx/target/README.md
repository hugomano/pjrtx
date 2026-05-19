# `//pjrtx/target`

Intent: own hardware facts, not compiler or runtime policy.

Owns:

- target kind and fingerprint material
- memory spaces and capacities
- transfer edges and bandwidth/latency facts
- execution units and dtype/op-class rates
- target validation and target summaries

Does not own:

- graph import
- lowering, fusion, tiling, scheduling, or backend binding decisions
- runtime allocations or streams
- report rendering beyond target-local summaries

Allowed dependencies:

- Zig `std`
- narrow generated target-local helpers

Primary outputs:

- validated target description
- target fingerprint material
- target capability facts consumed by compiler and report packages

