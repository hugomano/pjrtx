# `//next/pjrtx/core`

`//next/pjrtx/core` is deprecated bootstrap code.

It was useful for the first vertical slice because it forced PjRTx vocabulary
to become explicit. It should no longer be treated as the stable center of the
architecture. New code must not add dependencies on this package.

## Migration Direction

Move records out by ownership:

- target descriptions -> `//next/pjrtx/target`
- source, graph, compiler pass, fusion, placement, collective, lowering, cost,
  traffic, codegen, schedule, executable readiness, and explain facts ->
  `//next/pjrtx/compiler/facts`
- backend bindings, executable calls, kernel graphs, backend profile joins ->
  `//next/pjrtx/backend/facts`
- allocations, lifetimes, streams, events, runtime profile facts ->
  `//next/pjrtx/runtime/facts`
- stable text summaries and normalization -> `//next/pjrtx/report`

`docs/specs/package_boundaries_v0.md` is the migration contract. Delete this
package when `rg "//next/pjrtx/core|pjrtx/core" pjrtx` is empty.

## Rule

Do not add new schema here. Prefer a small package-owned type over a central
shared type unless the owning spec says otherwise.

Temporary aliases may exist only as migration bridges from old `core` names to
package-owned types. They must shrink whenever callers move to the owning
package, and they must not become the source of truth.

Current bridges:

- `//next/pjrtx/target` owns target hardware facts used by the trace report.
- `//next/pjrtx/compiler/facts` owns source, tensor, graph value, and graph
  instruction facts used by the legacy `TraceReport` shape.
- `//next/pjrtx/compiler/facts` owns pass, rewrite, fusion, placement, collective,
  lowering, cost ledger, and memory-traffic facts still embedded in the legacy
  `TraceReport` container.
