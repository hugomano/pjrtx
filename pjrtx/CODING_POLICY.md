# PjRTx Coding Policy

New architecture code lives under `//pjrtx/...` and follows the implementation
discipline in `docs/pjrtx_architecture_vision.md`.

Rules for the first slices:

- Prefer Zig `std` primitives before adding local utility functions.
- Local helpers should encode PjRTx domain meaning, not duplicate `std`.
- Put domain logic behind small Zig `struct` namespaces owned by one boundary,
  for example `TensorFacts.validate` or `GraphPayloadFacts.matchesKind`, instead
  of growing loose package-level utility functions.
- Split packages across meaningful files by fact family, pass family, backend
  contract, or runtime contract. Package roots should compose and re-export;
  they should not become mixed implementation buckets.
- Document intent, invariants, and contracts. Do not add comments that only
  restate the code.
- Refactor when the current shape hides ownership, correctness, performance, or
  explainability. Do not layer new code over a known-wrong abstraction.
- Keep schema docs centralized under `docs/specs`. Code should carry inline
  intent comments for invariants and contracts, but packages should not grow
  standalone schema files unless the architecture docs explicitly ask for one.
- Use `std.Io.Reader` for streamed input and `std.Io.Writer` for diagnostics,
  reports, and dumps.
- Use explicit-width integers for report IDs, serialized counts, byte sizes,
  and stable formats.
- Prefer typed initialization at the binding site:
  `const block: Block = try .init(allocator);`.
- Avoid inline `@as(T, value)` casts in test acceptance checks when a typed
  expected value or helper makes the test clearer.
- Use assertions for programmer invariants and diagnostics/error unions for
  expected program, target, and user failures.
- Keep ownership visible with explicit allocators and `deinit`.
- Do not add new dependencies on `//pjrtx/core`; new types live in the package
  that owns their invariants.
- Preserve package segregation from `docs/specs/package_boundaries_v0.md`.
- Treat a long file that mixes unrelated IDs, graph facts, pass facts, backend
  facts, validators, and writers as a design smell. Split it before adding more
  behavior.
- Prefer Zig for PjRTx compiler policy and MLIR external pass callbacks when
  the MLIR C API is sufficient.
- Keep C/C++ MLIR shim code minimal and limited to dialect registration,
  TableGen-generated ops/types/attrs, verifier/parser/printer integration, and
  rewrite helpers unavailable through the public C API.
- Do not introduce runtime fallback paths.
