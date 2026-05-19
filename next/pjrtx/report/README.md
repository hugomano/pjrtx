# `//next/pjrtx/report`

Intent: render deterministic reports from extracted facts without owning the
facts.

Owns:

- stable text summaries
- report normalization
- golden comparison helpers
- report-level validation that joins extracted facts

Does not own:

- compiler decisions
- target hardware definitions
- backend executable planning
- runtime execution

Allowed dependencies:

- Zig `std`
- `//next/pjrtx/target`
- `//next/pjrtx/compiler/facts`
- `//next/pjrtx/backend/facts`
- `//next/pjrtx/runtime/facts`

Primary outputs:

- deterministic human-readable reports
- normalized golden text
- report diagnostics through `std.Io.Writer`

