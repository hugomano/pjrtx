#!/usr/bin/env bash
set -euo pipefail

repo="${TEST_SRCDIR:-}"
if [[ -n "${repo}" && -d "${repo}/_main" ]]; then
  cd "${repo}/_main"
fi

fail() {
  echo "architecture boundary violation: $*" >&2
  exit 1
}

if find src/compiler -name '*.zig' -print0 | xargs -0 grep -Eq 'src/runtime|src/backend'; then
  fail "compiler must not import runtime or backend"
fi

if [[ -e src/core ]]; then
  fail "src/core must not exist; compiler owns IR and tensor vocabulary"
fi

if find src -name '*.zig' -print0 | xargs -0 grep -Eq 'src/core'; then
  fail "no Zig source may import src/core"
fi

if { grep -R -Eq '//src/core' src; } || { [[ -e MODULE.bazel ]] && grep -Eq '//src/core' MODULE.bazel; }; then
  fail "no Bazel target may depend on //src/core"
fi

if [[ -e src/backend/backend.zig || -e src/backend/registry.zig ]]; then
  fail "root backend vtable/registry files must not exist"
fi

if find src -name '*.zig' -print0 | xargs -0 grep -Eq 'Backend\.VTable|backend_registry|src/backend/registry'; then
  fail "dynamic backend vtable/registry usage must not return"
fi

if find src/runtime -name '*.zig' -print0 | xargs -0 grep -Eq '@import\("c"\)|PjrtxMlx|pjrtx_mlx|mlx_metal_api'; then
  fail "runtime must not import or mention MLX/Metal C API symbols"
fi

if grep -Eq 'PjrtxMlx|pjrtx_mlx|mlx_metal_api' src/plugin/plugin.zig; then
  fail "plugin must not import or mention MLX C shim symbols"
fi

if find src/plugin -name '*.zig' -print0 | xargs -0 grep -Eq '@import\("src/(compiler|backend|core)'; then
  fail "plugin must stay a PJRT-to-runtime adapter and must not import compiler/backend/core packages"
fi

if grep -Eq '@import\("src/runtime"|@import\("src/compiler"|@import\("src/backend' src/plugin/pjrt_abi.zig; then
  fail "raw PJRT ABI helpers must not import runtime, compiler, or backend packages"
fi

if find src/runtime -name '*.zig' -print0 | xargs -0 grep -Eq 'executeInstruction|runtime_fallback|fallback_instruction_count|allow_runtime_fallback|test_only_runtime_fallback'; then
  fail "runtime must not contain an instruction-interpreter fallback"
fi

if find src/runtime -name '*.zig' -print0 | xargs -0 grep -Eq 'evalReduce|evalDotGeneral|sortDenseBytes|seedFromBytes|nextRandomU32|readScalarAsF64|writeScalarFromF64|scalarIndexAt|dense host fallback|host shadow'; then
  fail "runtime must not contain CPU operation fallback helpers"
fi

if grep -Eq 'pub fn init(Elementwise|Compare|Select|Convert|Iota|Reshape|Transpose|Broadcast|Slice|Concatenate|DotGeneral|Reduce|Dynamic|Pad|Gather|Sort|PartitionId|Cholesky|Rng|Clamp|Reverse)' src/runtime/buffer.zig; then
  fail "runtime buffers must not expose StableHLO op constructors; lowering belongs to compiler/backend programs"
fi

if grep -Eq '^const (BufferDescriptor|BufferPlacement|DeviceStorage) = struct\b' src/runtime/buffer.zig; then
  fail "runtime buffer descriptor, placement, and storage owners must stay in buffer_* owner modules"
fi

if find src/runtime src/plugin src/backend -name '*.zig' -print0 | xargs -0 grep -Eq 'host_debug|hostDebugByteCount'; then
  fail "runtime buffers must not retain host-debug storage or expose hostDebugByteCount"
fi

if find src/runtime -name '*.zig' ! -path 'src/runtime/buffer.zig' ! -path 'src/runtime/buffer_storage.zig' -print0 | xargs -0 grep -Eq '@import\("buffer_storage\.zig"\)|\bstorage\.(handle|backend|accounted_bytes)\b'; then
  fail "runtime backend buffer storage ownership must stay behind Buffer and buffer_storage.zig"
fi

if find src/plugin src/compiler src/backend -name '*.zig' -print0 | xargs -0 grep -Eq '@import\("src/runtime/buffer_(descriptor|placement|storage)"'; then
  fail "runtime buffer owner modules are runtime-internal and must not be imported by plugin, compiler, or backend"
fi

if grep -Eq 'src/compiler/ir|Elementwise(Binary|Unary)Op|parse.*CustomCallOp' src/runtime/custom_call.zig; then
  fail "runtime custom-call registration must not parse backend operation names"
fi

if grep -Eq 'pub const Elementwise(Binary|Unary)Op|pub const CompareOp' src/runtime/runtime.zig; then
  fail "runtime root must not re-export compiler op enums it does not own"
fi

if grep -Eq 'pub const (CompileOptions|PlanInstruction|ShardingPlan) = ir\.' src/runtime/runtime.zig; then
  fail "runtime root must not re-export compiler plan internals that callers can import from compiler IR"
fi

if grep -Eq 'pub const (ExecutableCacheEntry|CachedBackendExecutable|ExecutableCacheStats)' src/runtime/runtime.zig; then
  fail "runtime root must not re-export executable-cache internals"
fi

if grep -Eq '^pub const Entry\b' src/runtime/executable_cache.zig; then
  fail "executable cache entries must stay private; use opaque cache leases across runtime modules"
fi

if grep -Eq 'pub const (ExecutableCache|DonationAliasStats|GraphNodeKind|GraphNode|BackendCompileOptions|BackendResidency|ExecutableContext)\b' src/runtime/runtime.zig; then
  fail "runtime root must not re-export owner-module internals that are not plugin/runtime entrypoint contracts"
fi

if grep -Eq 'pub const Topology\b' src/runtime/runtime.zig; then
  fail "runtime root must expose device and memory handles, not topology implementation details"
fi

if find src/runtime src/plugin -name '*.zig' -print0 | xargs -0 grep -Eq '\bGraphExecute(Result|Error)\b|graphExecuteError'; then
  fail "compiled executable execution must use runtime ExecutionResult/ExecutionError vocabulary, not graph-level public names"
fi

if find src/runtime src/plugin -name '*.zig' -print0 | xargs -0 grep -Eq 'ExecutableGraph|\.graph\b|graphDeviceCount|releaseGraph'; then
  fail "CompiledExecutable must be the only runtime executable owner; graph internals must not be reachable contracts"
fi

if grep -Eq 'pub const (ExecutablePlan|ExecutableGraph)\b' src/runtime/runtime.zig; then
  fail "runtime root must not re-export executable plan or graph internals"
fi

if grep -Eq 'pub const executeDevice\b' src/runtime/runtime.zig; then
  fail "runtime root must expose compiled-executable execution, not lower-level graph dispatch"
fi

if grep -Eq '@import\("src/compiler"\)|@import\("compile_options\.zig"\)' src/runtime/client.zig; then
  fail "runtime client must call compile_pipeline instead of importing compiler and compile-options owners"
fi

if grep -Eq 'compile_pipeline\.|executable_fingerprint\.alloc' src/runtime/client.zig; then
  fail "runtime client compile orchestration must stay in client_compile.zig"
fi

if grep -Eq 'clientFromExecutableContext|acquireCachedBackendExecutableForContext|releaseCachedBackendExecutableForContext|trimExecutableCacheForContext' src/runtime/client.zig; then
  fail "runtime client executable-context callback plumbing must stay in client_executable_context.zig"
fi

if grep -Eq 'acquireBackendExecutable|release\(self\.io|trimForAllocation\(self\.io|recordCompile\(self\.io|setMaxResidentBytes\(self\.io' src/runtime/client.zig; then
  fail "runtime client executable residency policy must stay in client_residency.zig"
fi

if find src/plugin src/compiler src/backend -name '*.zig' -print0 | xargs -0 grep -Eq '@import\("src/runtime/client_(compile|executable_context|residency)"'; then
  fail "runtime client owner modules are runtime-internal and must not be imported by plugin, compiler, or backend"
fi

if grep -Eq 'fn (parseCompileOptions|analyzeProgram|makeExecutablePlan|verifyPlan|allocExecutableFingerprint|updateInstructionFingerprint|updateTargetDeviceFingerprint|updateShardingFingerprint)\(' src/runtime/client.zig; then
  fail "runtime client must not own compile-pipeline or executable-fingerprint helpers"
fi

if grep -Eq 'fn (testShardingPlan|addU8ExecutablePlanForTest|constantU8ExecutablePlanForTest|cacheEntrySnapshotForTest)\(' src/runtime/client.zig; then
  fail "runtime client test fixtures must live under ClientTestSupport, not production-looking free functions"
fi

if grep -Eq 'fn testShardingPlan\(' src/runtime/executable.zig; then
  fail "runtime executable test fixtures must live under ExecutableTestSupport, not production-looking free functions"
fi

if grep -Eq 'pub fn (acquireCachedBackendExecutable|releaseCachedBackendExecutable)\(' src/runtime/client.zig; then
  fail "runtime client cache acquire/release plumbing must stay private behind executableContext callbacks"
fi

if grep -Eq 'descriptor\.(name|debug_string)|addressable_memories = device_memories|addressable_devices = memory_devices' src/runtime/client.zig; then
  fail "runtime client must construct topology through device_memory.DeviceMemoryTopology"
fi

if grep -Eq '\.executable_residency\.cache\b|\.executable_cache\b|executable_cache_mutex' src/runtime/client.zig; then
  fail "runtime client must use ExecutableResidencyCache behavior instead of reading cache internals"
fi

if grep -Eq '\.device_memory\.(devices|memories|device_handles|memory_handles)\b' src/runtime/client.zig; then
  fail "runtime client must use DeviceMemoryTopology methods instead of reading topology storage fields"
fi

if grep -Eq '\.backend\.(beginAsyncHostToDeviceTransfer|writeAsyncHostToDeviceTransfer|finishAsyncHostToDeviceTransfer|destroyAsyncHostToDeviceTransfer)\b' src/runtime/client.zig; then
  fail "runtime client must route async H2D operations through AsyncHostTransfer"
fi

if grep -Eq 'pub fn executeDevice\(' src/runtime/execution.zig; then
  fail "runtime execution must expose compiled-executable dispatch, not graph-level executeDevice"
fi

if grep -Eq '^test "' src/runtime/execution.zig; then
  fail "runtime execution.zig must stay a facade; execution tests belong in execution_* owner modules"
fi

if grep -Eq '^fn |^const (DonationAliasDelta|ExecutionTestSupport)\b|tryExecuteBackendExecutable|runtimeEventFromBackendCompletion|argumentMatchesDevice|mapBufferError' src/runtime/execution.zig; then
  fail "runtime execution.zig must not own implementation helpers; execution work belongs in execution_* owner modules"
fi

if grep -Eq 'donatedParameterAliasForOutput|takeBackendStorageForDonationAlias|rollbackDonationAlias|recordDonationAlias' src/runtime/execution.zig src/runtime/execution_call.zig src/runtime/execution_outputs.zig; then
  fail "runtime donation alias logic must stay in execution_donation.zig"
fi

if grep -Eq 'backendOutputMatches|owned_backend_outputs|backend_outputs' src/runtime/execution.zig src/runtime/execution_call.zig src/runtime/execution_donation.zig src/runtime/execution_completion.zig; then
  fail "runtime output descriptor validation and wrapping must stay in execution_outputs.zig"
fi

if grep -Eq 'Buffer\.initBackendHandle' src/runtime/execution.zig src/runtime/execution_call.zig src/runtime/execution_completion.zig; then
  fail "runtime output descriptor validation and wrapping must stay in execution_outputs.zig"
fi

if grep -Eq 'executionEventStatus|destroyExecutionEvent|backend returned asynchronous completion|backend execution event' src/runtime/execution.zig src/runtime/execution_call.zig src/runtime/execution_outputs.zig src/runtime/execution_donation.zig; then
  fail "runtime backend completion adaptation must stay in execution_completion.zig"
fi

if find src/plugin src/compiler src/backend -name '*.zig' -print0 | xargs -0 grep -Eq '@import\("src/runtime/execution_(call|outputs|donation|completion)"'; then
  fail "runtime execution owner modules are runtime-internal and must not be imported by plugin, compiler, or backend"
fi

if grep -Eq 'allocator\.alloc\([^)]*value_(handles|owned)|var value_(handles|owned) =' src/backend/mlx_metal/execution.zig; then
  fail "MLX Metal execution value-handle table allocation must stay behind ValueBindings"
fi

if grep -Eq '^test "' src/backend/mlx_metal/execution.zig; then
  fail "MLX Metal execution.zig must stay a facade; execution tests belong in execution_* owner modules"
fi

if grep -Eq '@import\("src/compiler/ir"\)|@import\("(buffer|custom_call|device|program)\.zig"\)' src/backend/mlx_metal/execution.zig; then
  fail "MLX Metal execution.zig must not import backend internals only needed by colocated tests"
fi

if grep -Eq '^const (ExecuteCall|ValueBindings|ScheduleDispatch|OutputBindings|ProgramNodeDispatch|ControlFlowDispatch|LivenessRelease|MaterializationBoundaryEval|CustomCallDispatch)\b|^fn (executeExecutable|executeCompiledProgram|donatedProgramInputIndices|maybeCreateInitialArgumentCapturedProgram|executeArgumentCapturedProgram|updateArgumentCaptureState|traceScheduleFailure|writeExecuteProfile|writeScheduleProfile)\(' src/backend/mlx_metal/execution.zig; then
  fail "MLX Metal execution.zig must stay a facade; execution implementation belongs in execution_* owner modules"
fi

if grep -Eq 'programCreateWithCaptures|programExecuteWithDonation|donatedProgramInputIndices|ArgumentCaptureState|InitialCaptureSmallControlBytes|MinCapturedProgramStableInputs' src/backend/mlx_metal/execution.zig; then
  fail "MLX Metal compiled-program execution and argument capture must stay in execution_compiled_program.zig"
fi

if grep -Eq 'handles: \[\]\?BufferHandle|owned: \[\]bool|storeOwnedValueHandle|storeBorrowedValueHandle|destroyOwnedValueHandles' src/backend/mlx_metal/execution.zig; then
  fail "MLX Metal execution value binding ownership must stay in execution_values.zig"
fi

if grep -Eq '^const (ProgramNodeDispatch|ControlFlowDispatch|LivenessRelease|MaterializationBoundaryEval|CustomCallDispatch)\b' src/backend/mlx_metal/execution.zig; then
  fail "MLX Metal node/control/liveness/materialization/custom-call dispatch owners must not return to execution.zig"
fi

if grep -Eq '^fn (executeProgramNode|executeControlFlowNode|whilePatternOperandHandle|whileStepOperandHandle|executeLoopInvariantRegionInstruction|loopInvariantRegionOperandHandle)\(' src/backend/mlx_metal/execution.zig; then
  fail "MLX Metal execution node and control-flow dispatch must stay behind ProgramNodeDispatch and ControlFlowDispatch owners"
fi

if grep -Eq '^fn (releaseDeadInputs|releaseDeadFusionGroupValues|evalMaterializationBoundaryRange|traceMaterializationFailure)\(' src/backend/mlx_metal/execution.zig; then
  fail "MLX Metal execution liveness and materialization work must stay behind named owner structs"
fi

if grep -Eq '^fn executeRegisteredCustomCall\(' src/backend/mlx_metal/execution.zig; then
  fail "MLX Metal custom-call execution must stay behind a backend-local dispatch owner"
fi

if grep -Eq '^test "' src/backend/mlx_metal/execution_node.zig; then
  fail "MLX Metal execution_node.zig must stay a dispatch router; node tests belong in operation-family modules"
fi

if grep -Eq '@import\("(buffer|device|lowering)\.zig"\)|buffer_mod\.Opaque\.' src/backend/mlx_metal/execution_node.zig; then
  fail "MLX Metal execution_node.zig must route to operation-family owners instead of calling buffer/lowering/device internals directly"
fi

if grep -Eq '\b(sortKeyValue|topK|reduceWindowMaxWithIndices|reduceMaxWithIndices|rngBitGenerator|optimizationBarrier|getTupleElement)\b' src/backend/mlx_metal/execution_node.zig; then
  fail "MLX Metal node operation bodies must stay in execution_node_* family modules"
fi

if find src/runtime src/plugin src/compiler -name '*.zig' -print0 | xargs -0 grep -Eq 'execution_node_(elementwise|generation|indexing|linalg|reduction|structural)\.zig'; then
  fail "MLX Metal execution-node family modules are backend-internal and must not be imported by runtime, plugin, or compiler"
fi

if find src/runtime src/plugin src/compiler -name '*.zig' -print0 | xargs -0 grep -Eq 'buffer_(encoding|lifecycle|elementwise|indexing|linalg|reduction|generation|control_flow|custom_call)\.zig'; then
  fail "MLX Metal buffer owner modules are backend-internal and must not be imported by runtime, plugin, or compiler"
fi

if grep -Eq 'const (Dtype|BinaryOp|UnaryOp|ReduceOp|ScatterUpdate|CompareOp|FftKind|RngDistribution|TriangularTranspose)\b|fn (wrap|handles)\(' src/backend/mlx_metal/buffer.zig; then
  fail "MLX Metal buffer facade must keep encoding and shim handle helpers in buffer owner modules"
fi

if grep -Eq 'mlx_call\.(buffer|customCall)' src/backend/mlx_metal/buffer.zig; then
  fail "MLX Metal buffer facade must delegate MLX operations to buffer owner modules"
fi

if find src/runtime src/plugin src/compiler -name '*.zig' -print0 | xargs -0 grep -Eq 'executable_(stats|constants|compiled_program|argument_capture|compile)\.zig'; then
  fail "MLX Metal executable owner modules are backend-internal and must not be imported by runtime, plugin, or compiler"
fi

if grep -Eq 'program_build_mod|lowering_mod|loadInstructionConstants|loadWhilePatternConstants|materializeConstant|programCompileEnabled|planSupportsCompiledProgram|destroy(ConstantHandles|CompiledPrograms|ArgumentCaptureStates)' src/backend/mlx_metal/executable.zig; then
  fail "MLX Metal executable facade must delegate compile, constants, compiled-programs, and teardown to executable owner modules"
fi

if grep -Eq '\b(previous_arguments|dynamic_indices|program_handle)\b' src/backend/mlx_metal/executable.zig; then
  fail "MLX Metal argument-capture storage must stay in executable_argument_capture.zig"
fi

if grep -Eq 'mlx_call\.program(Create|Destroy)' src/backend/mlx_metal/executable.zig; then
  fail "MLX Metal compiled-program handle ownership must stay in executable_compiled_program.zig"
fi

if find src/runtime src/plugin src/compiler -name '*.zig' -print0 | xargs -0 grep -Eq 'profiling_(env|clock|stats|writer)\.zig'; then
  fail "MLX Metal profiling owner modules are backend-internal and must not be imported by runtime, plugin, or compiler"
fi

if grep -Eq 'std\.c\.getenv|PJRTX_(TRACE|PROFILE|MLX_PROGRAM_COMPILE)|std\.debug\.print|Timestamp\.now|durationTo|recordScheduleItem|recordCompiledProgram|recordOutputClone|execute_wall_us_total' src/backend/mlx_metal/profiling.zig; then
  fail "MLX Metal profiling facade must delegate env, clock, stats, and rendering to profiling owner modules"
fi

if grep -Eq 'std\.c\.getenv|PJRTX_(TRACE|PROFILE|MLX_PROGRAM_COMPILE)' src/backend/mlx_metal/profiling.zig src/backend/mlx_metal/profiling_clock.zig src/backend/mlx_metal/profiling_stats.zig src/backend/mlx_metal/profiling_writer.zig; then
  fail "MLX Metal profiling environment reads must stay in profiling_env.zig"
fi

if grep -Eq 'std\.debug\.print' src/backend/mlx_metal/profiling.zig src/backend/mlx_metal/profiling_env.zig src/backend/mlx_metal/profiling_clock.zig src/backend/mlx_metal/profiling_stats.zig; then
  fail "MLX Metal profile and trace rendering must stay in profiling_writer.zig"
fi

if grep -Eq 'global_single_threaded\.io|Timestamp\.now' src/backend/mlx_metal/profiling.zig src/backend/mlx_metal/profiling_env.zig src/backend/mlx_metal/profiling_stats.zig src/backend/mlx_metal/profiling_writer.zig; then
  fail "MLX Metal backend IO and timestamps must stay in profiling_clock.zig"
fi

if grep -Eq 'pub const (ReduceMaxWithIndicesResult|ReduceWindowMaxWithIndicesResult|RngBitGeneratorResult)\b|pub fn (iota|partitionId|complex|realPart|imagPart|convert|bitcast|binary|unary|reshape|transpose|broadcastInDim|slice|dynamicSlice|dynamicUpdateSlice|pad|reverse|concatenate|gather|gatherAxis|scatter|scatterAxis|sort|argsort|takeAlongAxis|dotGeneral|convolution|cholesky|triangularSolve|fft|rng|rngBitGenerator|reduce|reduceMaxWithIndices|reduceWindow|reduceWindowMaxWithIndices|compare|select|clamp)\(' src/backend/mlx_metal/backend.zig; then
  fail "MLX Metal backend facade must expose runtime lifecycle contracts, not direct operation forwarding wrappers"
fi

if grep -Eq '^fn (validateValues|validateNodes|validateControlFlows|validateSubprograms|validateEdges|validateFusionGroups|validateMaterializationBoundaries|validateSchedule|validateMarkedValueSet|invalidProgram|markNodeOutputsLive|releaseDeadNodeInputs|releaseDeadFusionNodeInputs|releasePlannedValue)\b' src/backend/mlx_metal/program.zig src/backend/mlx_metal/program_build.zig; then
  fail "MLX Metal backend program validation and liveness bodies must stay in program_validation/program_liveness owners"
fi

if grep -Eq '^fn (countPlanSubprograms|countPlanControlFlows|buildNodeSubprograms|buildNodeControlFlow|cloneProgramSubprogram|cloneDescriptorList|cloneRegionValueList|cloneRegionInstructionList|deinitProgramSubprograms|deinitProgramControlFlows)\b' src/backend/mlx_metal/program_build.zig; then
  fail "MLX Metal backend region cloning and control-flow metadata must stay in program_region.zig"
fi

if grep -Eq '^fn (buildFusionGroups|deinitFusionGroups|fusionGroupNodeIndices|markedValueIds)\b|groupMarkIndex' src/backend/mlx_metal/program_build.zig; then
  fail "MLX Metal backend fusion group construction must stay in program_fusion.zig"
fi

if grep -Eq 'max_schedule|program_mod\.ScheduleItem' src/backend/mlx_metal/program_build.zig; then
  fail "MLX Metal backend schedule construction must stay in program_schedule_build.zig"
fi

if grep -Eq '^test "' src/backend/mlx_metal/program_build.zig; then
  fail "MLX Metal program_build.zig must stay a build router; tests belong with the owner module"
fi

if find src/runtime src/plugin src/compiler -name '*.zig' -print0 | xargs -0 grep -Eq '@import\("src/backend/mlx_metal/|@import\("[^"]*backend/mlx_metal/'; then
  fail "runtime, plugin, and compiler must import the MLX Metal backend package facade, not backend internals"
fi

if grep -Eq 'fn (mlxMetalBackendForTest|initMlxMetalClientForTest|testShardingPlan|addU8ExecutablePlanForTest|constantU8ExecutablePlanForTest)\(' src/runtime/execution.zig; then
  fail "runtime execution test fixtures must live under a named testing owner, not production-looking free functions"
fi

if grep -Eq 'fn (backendOutputMatchesPlan|planOutputBytes|donatedParameterAliasForOutput|planDonatesParameter)\(' src/runtime/execution.zig; then
  fail "runtime execution must query CompiledExecutable for output and donation facts"
fi

if grep -Eq '^pub const (DonationAliasStats|GraphNodeKind|GraphNode|BackendResidency)\b' src/runtime/executable.zig; then
  fail "runtime executable internals must stay private behind CompiledExecutable owner methods"
fi

if grep -Eq 'residencyStatus\(' src/runtime/executable.zig src/runtime/client.zig src/runtime/execution.zig; then
  fail "runtime executable cache/residency observations must use narrow CompiledExecutable methods"
fi

if find src/runtime -name '*.zig' -print0 | xargs -0 grep -Eq 'LoweringOptions|LoweringPipeline|\.lowering'; then
  fail "runtime must describe backend executable residency, not own a lowering pipeline"
fi

if find src/runtime -name '*.zig' -print0 | xargs -0 grep -Eq '^const io = std\.Io\.Threaded\.global_single_threaded\.io\(\);'; then
  fail "runtime IO handles must be owned by lifecycle objects instead of file-level globals"
fi

if grep -Eq 'entry\.ref_count' src/runtime/executable.zig; then
  fail "compiled executable residency must release cache entries through the executable-cache owner"
fi

cache_entry_leaks="$(
  find src/runtime src/plugin -name '*.zig' ! -path 'src/runtime/executable_cache.zig' ! -path 'src/runtime/client.zig' -print0 \
    | xargs -0 grep -En 'executable_cache\.get|executable_cache\.stats\b|executable_cache\.entrySnapshot|entry\.(backend_executable|ref_count|compile_latency_us)' || true
)"
if [[ -n "${cache_entry_leaks}" ]]; then
  echo "${cache_entry_leaks}" >&2
  fail "executable cache entries and counters must be observed through cache/client snapshot APIs"
fi

graph_field_leaks="$(
  find src/runtime src/plugin -name '*.zig' ! -path 'src/runtime/executable.zig' ! -path 'src/runtime/executable_cache.zig' ! -path 'src/runtime/executable_residency.zig' ! -path 'src/runtime/executable_schedule.zig' -print0 \
    | xargs -0 grep -En '\.(backend_executable|backend_residency|device_ids|last_compile_cache_trim|last_execute_cache_trim|last_backend_completion|donation_alias_stats)\b' || true
)"
if [[ -n "${graph_field_leaks}" ]]; then
  echo "${graph_field_leaks}" >&2
  fail "executable residency bookkeeping must be observed through owner methods"
fi

runtime_context_field_leaks="$(
  grep -En 'context\.(backend|devices)\b' src/runtime/execution.zig src/runtime/executable.zig || true
)"
if [[ -n "${runtime_context_field_leaks}" ]]; then
  echo "${runtime_context_field_leaks}" >&2
  fail "runtime executable/execution code must use Context methods instead of reading owned storage fields"
fi

buffer_storage_field_leaks="$(
  find src/runtime src/plugin -name '*.zig' ! -path 'src/runtime/buffer.zig' -print0 \
    | xargs -0 grep -En '\.backend_buffer\b|\.storage\.(handle|backend|accounted_bytes)\b' || true
)"
if [[ -n "${buffer_storage_field_leaks}" ]]; then
  echo "${buffer_storage_field_leaks}" >&2
  fail "runtime/plugin code must use Buffer storage methods instead of reading backend_buffer directly"
fi

if grep -Eq 'ptr\.(element_type|byte_size|deleted|device|memory|dims|ready_event)\b' src/plugin/buffer.zig; then
  fail "plugin buffer metadata must use runtime Buffer accessors instead of reading buffer fields"
fi

if grep -Eq 'deleted: bool|deleted = (true|false)' src/runtime/buffer.zig; then
  fail "runtime Buffer deletion state must be represented by BufferState, not a duplicate boolean"
fi

if grep -Eq '\.ready_event\b|buffer\.(byte_size|memory)\b' src/plugin/async_h2d.zig; then
  fail "plugin async H2D must use runtime Buffer readiness/accounting methods instead of mutating buffer fields"
fi

if grep -Eq 'client\.(backend|devices|memories|executable_cache)\b' src/runtime/execution.zig; then
  fail "runtime execution must use client/execution context APIs instead of reading client-owned storage fields"
fi

if find src/runtime -name '*.zig' -print0 | xargs -0 grep -Eq '\bbackend_impl\b'; then
  fail "runtime must use the concrete backend noun instead of legacy backend_impl abstraction vocabulary"
fi

if find src/runtime -name '*.zig' -print0 | xargs -0 grep -Eq 'bufferFromHost\(.*\) orelse null|allocator\.dupe\(u8, src\)'; then
  fail "host imports must materialize backend device storage, not host-only buffers"
fi

if grep -Eq 'loadedExecutableExecuteLegacy|dense host fallback' src/plugin/plugin.zig; then
  fail "plugin execute path must stay graph/backend-only; no legacy or host fallback executor"
fi

if grep -Eq 'runtime\.executeDevice|\.plan\.deinit\(\)|\.graph\.deinit\(\)|plugin\.allocator\(\)\.free\(executable\.(fingerprint|optimized_program)\)' src/plugin/executable.zig; then
  fail "plugin loaded executables must own runtime CompiledExecutable as one object instead of dismantling runtime internals"
fi

plugin_client_field_leaks="$(
  find src/plugin -name '*.zig' -print0 \
    | xargs -0 grep -En 'client\.(devices|memories|device_handles|memory_handles|topology)\b|\.(device_handles|memory_handles)\b|handles\.Client\.ref\([^)]*\)\.(devices|memories|device_handles|memory_handles|topology)\b' || true
)"
if [[ -n "${plugin_client_field_leaks}" ]]; then
  echo "${plugin_client_field_leaks}" >&2
  fail "plugin must use runtime Client topology/placement accessors instead of reading client-owned storage"
fi

if grep -Eq '//src/runtime|//src/backend' src/compiler/BUILD.bazel; then
  fail "compiler BUILD deps must stay independent of runtime/backend"
fi

if grep -Eq '//src/backend(:|")|//src/backend:registry|//src/backend:backend' src/runtime/BUILD.bazel; then
  fail "runtime must depend on the concrete MLX backend package, not a root backend facade"
fi

if grep -Eq 'mlx_metal_api|mlx_metal_backend' src/runtime/BUILD.bazel; then
  fail "runtime must not depend directly on MLX C shim targets"
fi

if find src/backend -maxdepth 1 -type f -name 'mlx_*' | grep -q .; then
  fail "backend implementations must live under backend-specific directories"
fi

if [[ -e src/backend/synthetic ]]; then
  fail "synthetic backend code must not exist in //src; MLX Metal is the only production backend"
fi

mlx_c_violations="$(
  find src/backend/mlx_metal -name '*.zig' ! -name 'mlx_call.zig' -print0 \
    | xargs -0 grep -En '@import\("c"\)|extern "c"|pjrtx_mlx_metal_' || true
)"
if [[ -n "${mlx_c_violations}" ]]; then
  echo "${mlx_c_violations}" >&2
  fail "only src/backend/mlx_metal/mlx_call.zig may import translated C or call MLX/Metal C shims"
fi

if find src/backend/mlx_metal -name '*.zig' -print0 | xargs -0 grep -Eq 'fallbackDevice|synthetic backend'; then
  fail "MLX Metal backend must not grow fake or synthetic device fallback paths"
fi

if grep -Eq 'fn (executeProgramNode|executeFusionGroup|traceScheduleFailure|executeRegisteredCustomCall)\(' src/backend/mlx_metal/backend.zig; then
  fail "MLX Metal backend root must not own execution scheduling bodies"
fi

if grep -Eq 'fn (instructionIssue|validate[A-Za-z0-9_]*Lowering|supportedScatterAxis)\(' src/backend/mlx_metal/backend.zig; then
  fail "MLX Metal backend root must not own lowering validation bodies"
fi

if grep -Eq '^fn (instructionIssue|validate[A-Za-z0-9_]*Lowering|validate[A-Za-z0-9_]*CustomCall|inputDescriptor|dimsEqual|valid[A-Za-z0-9_]*Shape|isSupported(Float|Integer|Comparable)|dotGeneralIsMatmulLike|regionValueById|matchWhileF32LtAddPattern|descriptorsEqual)\(' src/backend/mlx_metal/lowering.zig; then
  fail "MLX Metal lowering facade must stay a package surface; implementation belongs in lowering owner modules"
fi

if grep -Eq '^fn validate[A-Za-z0-9_]*CustomCall|custom_call_mod\.lookup' src/backend/mlx_metal/lowering_rules.zig; then
  fail "MLX Metal custom-call lowering legality must stay in lowering_custom_call.zig"
fi

if grep -Eq '^fn (inputDescriptor|dimsEqual|valid[A-Za-z0-9_]*Shape|isSupported(Float|Integer|Comparable)|dotGeneralIsMatmulLike|supported(Gather|Scatter)Axis|scatterUpdateShapeMatchesAxis|gatherOutputShapeMatchesTake)\(' src/backend/mlx_metal/lowering_rules.zig; then
  fail "MLX Metal shape and dtype predicates must stay in lowering_shapes.zig"
fi

if grep -Eq '^fn (regionValueById|matchWhileF32LtAddPattern|descriptorsEqual|regionValueIsArgumentIndex|whileStepOperandFromRegionValue|loopInvariantProducerInstructionIndex)\(' src/backend/mlx_metal/lowering_rules.zig; then
  fail "MLX Metal while/region pattern matching must stay in lowering_control_flow.zig"
fi

if grep -Eq '^(pub )?fn validate(Bitcast|PartitionId|Rng|RngBitGenerator|OptimizationBarrier|Tuple|GetTupleElement|Binary|Unary|Complex|RealImag|Compare|Select|Clamp|Transpose|BroadcastInDim|Slice|DynamicSlice|DynamicUpdateSlice|Concatenate|Pad|Gather|Scatter|Sort|TopK|DotGeneral|Convolution|Cholesky|TriangularSolve|Fft|Reduce|ReduceWindow|While)' src/backend/mlx_metal/lowering_rules.zig; then
  fail "MLX Metal lowering_rules.zig must route validation; family validators belong to lowering_* owner modules"
fi

if grep -Eq '^fn (convMetadataLen|convReversalLen|defaultSpatialDims)\(' src/backend/mlx_metal/lowering_rules.zig; then
  fail "MLX Metal linalg helper validation must stay in lowering_linalg.zig"
fi

if grep -Eq 'fn (programNodeKind|buildFusionGroups|buildNodeSubprograms|countPlanSubprograms)\(' src/backend/mlx_metal/backend.zig; then
  fail "MLX Metal backend root must not own program graph construction bodies"
fi

if grep -Eq 'std\.c\.getenv|PJRTX_MLX_PROGRAM_COMPILE' src/backend/mlx_metal/backend.zig; then
  fail "MLX Metal backend root must not own executable compile policy; executable ownership belongs in executable.zig"
fi

if grep -Eq 'buffer_mod\.Buffer|bufferRef|bufferRefs|maybeBufferHandle' src/backend/mlx_metal/backend.zig; then
  fail "MLX Metal backend root must not own opaque buffer decoding; buffer ownership belongs in buffer.zig"
fi

if grep -Eq 'async_transfer_mod\.AsyncTransfer|\.toHandle\(' src/backend/mlx_metal/backend.zig; then
  fail "MLX Metal backend root must not own opaque async-transfer decoding; async transfer ownership belongs in async_transfer.zig"
fi

if grep -Eq '^fn ' src/backend/mlx_metal/backend.zig; then
  fail "MLX Metal backend root must stay package surface only; implementation functions belong in owner modules"
fi

backend_env_leaks="$(
  find src/backend/mlx_metal -name '*.zig' ! -name 'profiling.zig' ! -name 'profiling_env.zig' -print0 \
    | xargs -0 grep -En 'std\.c\.getenv|PJRTX_(TRACE|PROFILE|MLX_PROGRAM_COMPILE)' || true
)"
if [[ -n "${backend_env_leaks}" ]]; then
  echo "${backend_env_leaks}" >&2
  fail "MLX Metal backend environment knobs must be read through profiling.zig"
fi

if grep -Eq 'std\.atomic\.Mutex|spinLoopHint' src/backend/mlx_metal/custom_call.zig; then
  fail "MLX Metal custom-call registry must use std.Io.Mutex instead of spin-loop registry locks"
fi

if find src/backend/mlx_metal -name '*.zig' -print0 | xargs -0 grep -Eq 'std\.atomic\.Mutex|spinLoopHint'; then
  fail "MLX Metal backend synchronization must use std.Io.Mutex instead of spin-loop mutexes"
fi

if grep -Eq '@import\("execution\.zig"\)' src/backend/mlx_metal/execution.zig; then
  fail "MLX Metal execution module must not import itself to recover local types"
fi

if grep -Eq '^const backend = struct' src/backend/mlx_metal/execution.zig; then
  fail "MLX Metal execution must use owned module types directly instead of a fake local backend namespace"
fi

if grep -Eq '\bBackend\b|backend_impl|create\(\)' src/backend/mlx_metal/execution.zig; then
  fail "MLX Metal execution must not thread an empty backend facade through owned execution helpers"
fi

if grep -Eq 'TestBackend|createTestBackend' src/backend/mlx_metal/execution.zig; then
  fail "MLX Metal execution tests must call owned module APIs directly instead of local backend facades"
fi

if grep -Eq 'async_transfer_mod|AsyncHostToDeviceTransferHandle|fn (beginAsyncHostToDeviceTransfer|writeAsyncHostToDeviceTransfer|finishAsyncHostToDeviceTransfer|destroyAsyncHostToDeviceTransfer|allocateBuffer|zeroLike|profileEnabled|profileVerbose|profileStart|profileElapsedUs|lookupCustomCall|matchWhileF32LtAddPattern|regionValueById|supportedScatterAxis|executableBinaryOp|executableUnaryOp)\(' src/backend/mlx_metal/execution.zig; then
  fail "MLX Metal execution must call owned async-transfer, profiling, lowering, and custom-call modules directly instead of local forwarding wrappers"
fi

if grep -Eq 'std\.debug\.print' src/backend/mlx_metal/execution.zig; then
  fail "MLX Metal execution must pass trace/profile snapshots to profiling.zig instead of rendering text directly"
fi

if grep -Eq 'bufferRef|bufferRefs|maybeBufferHandle|fn (iota|partitionId|cloneBuffer|complex|realPart|imagPart|convert|bitcast|binary|binaryWithOutputDims|unary|reshape|transpose|broadcastInDim|slice|dynamicSlice|dynamicUpdateSlice|pad|reverse|concatenate|gatherAxis|gather|scatterAxis|scatter|sort|argsort|takeAlongAxis|dotGeneral|convolution|fft|rng|rngBitGenerator|cholesky|triangularSolve|reduce|reduceMaxWithIndices|reduceWindow|reduceWindowMaxWithIndices|compare|select|clamp|whileF32CompareAdd|copyToHost|destroyBuffer|evalBuffers|evalBuffer)\(' src/backend/mlx_metal/execution.zig; then
  fail "MLX Metal execution must use buffer_mod.Opaque for opaque buffer operations instead of local buffer forwarding wrappers"
fi

if grep -Eq 'pub fn destroy(ConstantHandles|CompiledPrograms|ArgumentCaptureStates)' src/backend/mlx_metal/execution.zig; then
  fail "MLX Metal execution must not re-export executable teardown helpers owned by executable.zig"
fi

if grep -Eq 'fn (executableStats|recordSuccessfulExecute|recordFusionGroupExecute|recordCompiledProgramExecute|recordCapturedProgramExecute|recordMaterializationEval|recordReleasedIntermediateValues|recordBorrowedConstantNode|recordExecuteProfile|lockExecutableStats|unlockExecutableStats|destroyExecutable)\(' src/backend/mlx_metal/execution.zig; then
  fail "MLX Metal execution must not own executable stats, locks, or teardown; executable ownership belongs in executable.zig"
fi

if grep -Eq 'fn (argumentCaptureMatches|rememberArgumentCaptureBaseline|resetArgumentCaptureState)\(' src/backend/mlx_metal/execution.zig; then
  fail "MLX Metal execution must mutate argument-capture state through executable-owned methods"
fi

if grep -Eq 'fn (constantIndex|whileConstantIndex)\(' src/backend/mlx_metal/execution.zig; then
  fail "MLX Metal execution must use executable-owned residency indexing helpers directly"
fi

if grep -Eq 'fn (executableSupportsInstruction|validElementwiseBroadcast|dotGeneralIsMatmulLike|validGatherShape|isSupported(Float|Integer|Comparable))\(' src/backend/mlx_metal/execution.zig; then
  fail "MLX Metal execution must not own lowering validation helpers; lowering ownership belongs in lowering.zig"
fi

backend_buffer_decode_leaks="$(
  find src/backend/mlx_metal -name '*.zig' ! -name 'buffer.zig' ! -name 'buffer_lifecycle.zig' -print0 \
    | xargs -0 grep -En 'Buffer\.fromHandle' || true
)"
if [[ -n "${backend_buffer_decode_leaks}" ]]; then
  echo "${backend_buffer_decode_leaks}" >&2
  fail "MLX Metal opaque buffer decoding must stay in buffer.zig"
fi

echo "architecture boundaries OK"
