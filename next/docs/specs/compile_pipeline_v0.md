# Compile Pipeline V0

This spec defines the no-fallback compile pipeline for the new PjRTx
architecture under `//next/pjrtx/...`.

The central rule:

```text
If any stage cannot prove legality, stop.
Do not emit an executable.
Do not reach runtime.
```

Reference execution is only a test oracle. The production path has no runtime
fallback.

## Pipeline Overview

V0 pipeline:

```text
1. Input setup
2. StableHLO/VHLO ingest
3. MLIR verify and canonicalize
4. StableHLO/PjRTx lowering pass pipeline
5. Typed graph import
6. Graph verification
7. Target selection
8. Target legality
9. Fusion candidate discovery
10. Fusion decision plan
11. Layout, tile, and memory-space planning
12. Collective lowering plan
13. Cost ledger construction
14. Lowering records
15. Schedule build
16. Schedule verification
17. Backend binding
18. Backend binding verification
19. Executable creation
20. Report emission
```

Every stage should either produce a valid artifact or fail with diagnostics.
The compiler middle is intentional in V0 even when a stage initially emits a
small or conservative record. PjRTx must not depend on XLA as a hidden lowering,
fusion, tiling, collective, or scheduling engine. XLA is a reference for pass
families and hard problems; the PjRTx report must describe the decisions that
create the executable.

For the internal pass families used by stages 3 through 14, see
`compiler_pass_pipeline_v0.md`. This file defines the stage gates; the pass
pipeline spec defines the small compiler passes, invariants, records, and
future ordering inside those gates.

## Stage Contract

Every stage has the same shape:

```zig
pub const StageResult = union(enum) {
    ok,
    failed,
};
```

Actual APIs should use Zig error unions, but the conceptual contract is:

```text
input artifacts + diagnostics writer -> output artifact or error
```

Rules:

- diagnostics are written through `std.Io.Writer`
- no stage should silently drop unsupported information
- every produced artifact must be internally valid
- every failure should include pass/stage name and feature label
- failed compile may emit partial diagnostic/report artifacts
- failed compile must not emit a runnable executable

## Stage 1: Input Setup

Inputs:

- PJRT program bytes or test fixture bytes
- program format string
- compile options
- selected backend or target kind
- diagnostics writer

Output:

```zig
CompileInput
  program_format
  program_reader
  compile_options
  target_request
```

Validation:

- program is present
- format is recognized or diagnosable
- compile options parse
- requested target exists

Failure examples:

```text
invalid program format
invalid compile option
unknown target
```

## Stage 2: StableHLO/VHLO Ingest

Inputs:

- `CompileInput`
- `std.Io.Reader`

Output:

```zig
MlirModuleArtifact
  module
  source_table
```

Rules:

- use MLIR/StableHLO C API targets
- support StableHLO text and portable artifacts as the current compiler does
- do not invent a custom StableHLO parser
- retain source operation order for `SourceRef`

Failure examples:

```text
portable artifact deserialization failed
MLIR parser rejected module
unsupported program encoding
```

## Stage 3: MLIR Verify And Canonicalize

Inputs:

- `MlirModuleArtifact`

Output:

```zig
CanonicalMlirArtifact
  module
  source_table
  pass_summary
```

Rules:

- run MLIR verifier
- run canonicalization/CSE as appropriate
- run Shardy propagation only when Shardy metadata is present
- preserve enough source provenance for graph import

Failure examples:

```text
MLIR verifier rejected module
Shardy propagation failed
canonicalization pipeline failed
```

## Stage 4: StableHLO/PjRTx Lowering Pass Pipeline

Inputs:

- `CanonicalMlirArtifact`
- compile options
- target-independent lowering policy

Output:

```zig
LoweredMlirArtifact
  module
  source_table
  pass_pipeline_report
```

Rules:

- use Bazel MLIR, StableHLO, and Shardy C API targets
- do not call into XLA as an opaque compiler
- run PjRTx policy passes in Zig where practical, either as straight-line V0
  shim steps or MLIR external passes registered through the C API pass manager
- use minimal TableGen/C++ only for real PjRTx dialect definitions,
  parser/printer/verifier integration, and MLIR rewrite or dialect-conversion
  hooks unavailable through the C API
- every pass that changes the executable must produce a stable report entry
- pass entries record pass name, status, input fingerprint, output fingerprint,
  diagnostics, and source-provenance preservation status
- preserve Shardy metadata, tokens, side effects, regions, and source refs until
  an explicit later stage consumes them
- fail if a lowering pass would erase information required for correctness or
  explainability

Failure examples:

```text
stablehlo legalization pipeline failed
pass erased source provenance for instruction
unsupported region form after lowering
```

## Stage 5: Typed Graph Import

Inputs:

- `LoweredMlirArtifact`

Output:

```zig
GraphModule
  values
  instructions
  sources
```

V0 supported operations:

- parameter
- constant when needed by fixture
- `stablehlo.dot_general`
- `stablehlo.add`
- `stablehlo.tanh`
- `stablehlo.broadcast_in_dim`
- return

Rules:

- create typed payloads
- no optional metadata bag
- unsupported ops fail here
- all source operations receive stable `SourceRef`

Failure examples:

```text
unsupported op: stablehlo.convolution
unsupported dynamic shape
unsupported tuple result
unsupported region form
```

## Stage 6: Graph Verification

Inputs:

- `GraphModule`

Output:

```zig
VerifiedGraph
```

Checks:

- all value IDs are valid
- all instruction inputs and outputs exist
- every value has dtype, dims, and layout
- rank-2 dot dimensions match
- elementwise shapes are compatible
- broadcast dimensions are valid
- output values exist
- no unsupported instruction kind remains

Failure examples:

```text
dot_general contracting dimensions mismatch
elementwise shape mismatch
instruction references missing value
```

## Stage 7: Target Selection

Inputs:

- target request
- available targets

Output:

```zig
SelectedTarget
  target_description
```

V0 targets:

- `metal_v0`
- `npu_v0`

Rules:

- unknown target fails
- target description validates before use
- unknown performance fields are allowed but explicit

## Stage 8: Target Legality

Inputs:

- `VerifiedGraph`
- `SelectedTarget`

Output:

```zig
LegalGraph
  graph
  target
  legality_records
```

Checks:

- target supports required dtype or reports unknown/unsupported
- target has an execution path for each graph region
- required memory spaces exist
- required transfer edges exist
- backend can bind the selected lowering strategy

Rules:

- unknown performance rate is not a legality failure
- unknown execution capability is a legality failure
- no unsupported graph region may continue

Failure examples:

```text
target does not support f32 dot_general
target has no backend execution path for tanh
missing host_to_device transfer edge
```

## Stage 9: Fusion Candidate Discovery

Inputs:

- `LegalGraph`
- `SelectedTarget`

Output:

```zig
FusionCandidateSet
  candidates
  boundaries
```

Rules:

- discover regions, not only single operations
- record producer/consumer IDs and candidate source instructions
- record hard boundaries for collectives, tokens, side effects, layout
  barriers, unsupported dtypes, and target capability gaps
- estimate bytes saved, launch reduction, and pressure when target data exists
- unknown estimates are explicit and must not become legality claims

Failure examples:

```text
fusion candidate references missing value
fusion candidate crosses token boundary
```

## Stage 10: Fusion Decision Plan

Inputs:

- `FusionCandidateSet`
- `SelectedTarget`
- compile options

Output:

```zig
FusionDecisionSet
  candidates
  accepted_groups
  rejected_candidates
```

Rules:

- accepted fusion groups link all source instruction IDs
- rejected candidates carry a reason
- mathematical semantics must be preserved unless compile options explicitly
  allow a relaxation
- fusion decisions are MLIR-owned facts and become inputs to cost, tiling,
  backend binding, and explain records

Failure examples:

```text
fusion changes strict reduction semantics
fusion group exceeds target local memory constraint
```

## Stage 11: Layout, Tile, And Memory-Space Planning

Inputs:

- `LegalGraph`
- `FusionDecisionSet`
- `SelectedTarget`

Output:

```zig
PlacementPlan
  layout_records
  tile_records
  memory_space_records
```

Rules:

- record layout for every value that reaches schedule build
- record tile shape for every generated-kernel or kernel-graph region; Metal
  V0 may use whole-tensor tiles, while NPU V0 must expose bounded local-memory
  tiles when tensors exceed the synthetic tile limit
- record memory-space assignment for values and temporaries
- join HBM/SRAM/scratchpad/unified-memory decisions to target memory spaces
- record unsupported tile/layout choices as compile failures

Failure examples:

```text
no legal layout for backend kernel graph
tile exceeds local_sram capacity
missing memory transfer edge for planned placement
```

## Stage 12: Collective Lowering Plan

Inputs:

- `LegalGraph`
- `FusionDecisionSet`
- `PlacementPlan`
- `SelectedTarget`

Output:

```zig
CollectivePlan
  decision
  algorithm
  lowered_collectives
  rejected_collectives
  estimated_bytes
  estimated_latency_ns
```

Rules:

- verify replica groups, partition groups, channel IDs, token ordering, dtype,
  layout, and rendezvous requirements
- decide direct, ring, tree, split, async start/done, or unsupported behavior
- record algorithm, route, chunking, temporary buffers, stream kind, and overlap
  intent when supported
- record `algorithm=none` for graphs without collective work
- fail before schedule build when no target collective path exists
- collectives are hard fusion boundaries unless a pass proves legality

Failure examples:

```text
collective group references missing participant
target has no collective engine for all_reduce
collective algorithm unsupported for dtype
```

## Stage 13: Cost Ledger Construction

Inputs:

- `LegalGraph`
- `FusionDecisionSet`
- `PlacementPlan`
- `CollectivePlan`

Output:

```zig
CostLedger
```

Rules:

- every graph instruction that does work gets a cost entry
- transfers get cost entries with zero logical ops
- formulas are stable strings
- approximations are explicitly marked
- unknown target rates do not prevent logical FLOP/byte accounting

Failure examples:

```text
cannot compute byte size due to invalid dimension
unsupported dot_general rank for V0 cost formula
```

## Stage 14: Lowering Records

Inputs:

- `LegalGraph`
- `FusionDecisionSet`
- `PlacementPlan`
- `CollectivePlan`
- `CostLedger`

Output:

```zig
LoweringPlan
  records
```

V0 lowering decisions:

- rank-2 `dot_general` -> backend kernel graph command
- compatible `add` + `tanh` -> elementwise fusion record
- inputs/outputs -> transfer records
- supported collectives -> collective command records
- unsupported collectives -> failed lowering diagnostics

Rules:

- every lowered region links graph instruction IDs
- every lowered region links cost ledger IDs
- rejected alternatives are recorded when meaningful
- unsupported lowering fails here, not at runtime

Failure examples:

```text
cannot lower tanh for selected target
cannot fuse elementwise region due to dtype mismatch
```

## Stage 15: Schedule Build

Inputs:

- `LegalGraph`
- `LoweringPlan`
- `CostLedger`
- `PlacementPlan`
- `CollectivePlan`

Output:

```zig
ExecutableSchedule
  commands
  streams
```

V0 commands:

- H2D for each input
- backend execute
- DMA copy when required by placement
- collective start/done when required by collective lowering
- D2H for requested output

Rules:

- commands have stable IDs
- data dependencies are explicit
- stream order is explicit
- command links to lowering/cost records
- command stream kind follows compute, DMA, or collective ownership
- schedule is built from lowering/fusion/tile/memory/collective records, not
  directly from raw StableHLO

## Stage 16: Schedule Verification

Inputs:

- `ExecutableSchedule`

Output:

```zig
VerifiedSchedule
```

Checks:

- all command dependencies exist
- command input/output values exist
- backend execute depends on required H2D commands
- D2H depends on backend execute
- every backend command has lowering provenance
- every command with work has cost provenance

Failure examples:

```text
schedule command references missing graph value
backend execute missing dependency on input transfer
command missing lowering record
```

## Stage 17: Kernel Codegen Plan

Inputs:

- `VerifiedSchedule`
- lowering records
- memory traffic records
- selected backend

Output:

```zig
KernelCodegenPlan
  records
```

V0 behavior:

- one record per executable lowering region
- matmul regions name `npu_matmul` or the matching Metal/MLS kernel operation
- fused elementwise regions name `npu_elementwise_fusion` or
  `metal_mls_elementwise_fusion_kernel`

Rules:

- records link lowering, command, graph instructions, cost ledger, memory
  traffic, backend kind, operation name, and expected unit
- result memory and tile memory requirements are checked against target
  capacities before backend binding
- backend binding consumes these records instead of rediscovering codegen shape
- no backend-private generated kernel may skip this trace row

## Stage 18: Backend Binding

Inputs:

- `KernelCodegenPlan`
- selected backend

Output:

```zig
BackendBindingPlan
  bindings
```

V0 behavior:

- Metal can bind one backend executable command
- NPU can bind synthetic backend commands
- collective commands bind to target collective engines when supported

Rules:

- every backend execute command gets a binding
- binding records backend operation string
- binding records expected execution unit when known
- binding links graph instructions and cost ledger IDs

Failure examples:

```text
backend cannot bind executable command
backend missing required capability
```

## Stage 19: Backend Binding Verification

Inputs:

- `BackendBindingPlan`
- `VerifiedSchedule`
- `SelectedTarget`

Output:

```zig
VerifiedBackendBindings
```

Checks:

- every binding references an existing command
- every backend execute command has exactly one binding
- expected unit exists when provided
- backend kind matches selected target
- cost ledger links exist
- fused, tiled, memory-space, and collective records referenced by the binding
  exist when the command depends on them

## Stage 20: Executable Creation

Inputs:

- `VerifiedGraph`
- `VerifiedSchedule`
- `KernelCodegenPlan`
- `VerifiedBackendBindings`
- target

Output:

```zig
CompiledExecutable
  graph
  schedule
  backend_bindings
  backend_executable_plans
  cost_ledger
  explain_records
```

Rules:

- only verified artifacts can create an executable
- executable owns or references artifacts with clear lifetime
- backend executable plans expand backend bindings into concrete backend calls
  before runtime submission
- Metal backend calls expand into an MLS kernel graph with one node per planned
  lowering-region kernel and explicit value-flow edges between producer and
  consumer nodes
- MLS kernel graph nodes carry output tensor descriptors and typed operation
  attributes such as dot contracting dimensions, broadcast dimensions, or fused
  instruction provenance
- backend executable planning must reject bindings that are not present in the
  validated report
- runtime receives only `CompiledExecutable`
- runtime does not re-check unsupported operation legality

No executable is produced on failed compile.

## Stage 21: Report Emission

Inputs:

- successful `CompiledExecutable`, or failed partial compile diagnostics

Output:

```text
Vertical slice report
```

Rules:

- stable order
- redacted timings in golden tests
- all IDs validate
- failed compiles may emit diagnostic reports
- failed compiles must say `executable_created=false`
- compiler-middle sections are emitted when their stages ran, including MLIR
  pass summaries, fusion plans, tile/layout/memory-space plans, and collective
  plans

## No-Fallback Invariants

These invariants must hold:

- runtime never receives an unverified schedule
- runtime never receives unsupported graph instructions
- backend execute command always has a backend binding
- unlowerable graph regions fail before schedule verification
- invalid schedules fail before executable creation
- reference execution is never used to produce PJRT outputs

## Error Mapping

Suggested error categories:

```zig
CompilePipelineError
  InvalidInput
  InvalidStablehlo
  UnsupportedProgram
  InvalidGraph
  UnsupportedTargetFeature
  InvalidCostModel
  InvalidLowering
  InvalidSchedule
  InvalidBackendBinding
  OutOfMemory
```

PJRT mapping can happen in the plugin adapter, but the compile pipeline should
preserve structured diagnostics below it.

## Tests

Required tests:

- unsupported op fails before schedule build
- invalid dot shape fails during graph verification
- missing target capability fails during target legality
- lowering pass failure reports pass name and no executable
- invalid fusion candidate fails before schedule build
- tile/layout/memory-space impossibility fails before schedule build
- unsupported collective lowering fails before schedule build
- missing lowering record fails schedule verification
- missing backend binding fails backend binding verification
- failed compile reports `executable_created=false`
- successful V0 workload reaches executable creation

## V0 Decisions

Target legality and compiler-middle planning happen before cost ledger
construction. The compiler first proves that the selected target has an
execution path for every graph region, then records fusion, layout, tile,
memory-space, and collective decisions. It then supplies compact backend
capability facts to MLIR state, where cost records are derived for the legal
planned graph and committed before lowering-region formation. Future diagnostic
modes may emit best-effort cost hints for unsupported programs, but those hints
must not feed executable creation.

The compiler-middle stages may start conservative, but they cannot be skipped
silently. If V0 chooses a whole-tensor tile, bounded local-memory tile, no
fusion, or unsupported collective behavior, that choice is still a reportable
artifact or a compile failure. This keeps PjRTx from outsourcing the real
compiler to XLA or hiding decisions inside backend submission code.

Failed compiles emit diagnostic reports as early as possible. Before graph
import succeeds, the report contains input, MLIR, and diagnostic sections only.
After graph import, every partial artifact that exists must validate its own
links. All failed reports include `executable_created=false`.

Executable creation owns verified artifacts with explicit Zig lifetimes. V0
should prefer allocator-owned slices and clear `deinit` paths over a hidden
single compile arena. Arena allocation is allowed inside a stage as an
implementation detail, but the `CompiledExecutable` boundary must make
ownership visible.
