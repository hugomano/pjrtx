const core = @import("pjrtx/core");
const compiler_facts = @import("pjrtx/compiler/facts");
const target_pkg = @import("pjrtx/target");

/// MLIR-state errors describe verifier or state-machine failures at the
/// compiler boundary. Runtime/backend packages may observe them through
/// extracted facts, but they do not own the transitions.
pub const MlirStateError = error{
    InvalidStablehlo,
    InvalidStateTransition,
    InvalidTargetAttachment,
    InvalidFusionPlan,
    InvalidPlacementPlan,
    InvalidCollectivePlan,
    InvalidLoweringPlan,
    InvalidKernelCodegenPlan,
    InvalidSchedulePlan,
    InvalidBackendBindingPlan,
    InvalidExecutableContract,
    InvalidBackendExecutablePlan,
    InvalidExtractionState,
    ExternalPassFailed,
};

pub const ModuleState = enum {
    imported,
    target_attached,
    target_legal,
    fusion_planned,
    placement_planned,
    collectives_planned,
    lowering_planned,
    performance_modeled,
    codegen_planned,
    scheduled,
    backend_bound,
    executable_ready,
    backend_executable_planned,
    backend_kernel_graph_planned,
    runtime_allocation_planned,
    runtime_stream_planned,
    runtime_profiled,
    runtime_profile_joined,
    backend_profile_joined,

    pub fn text(self: ModuleState) []const u8 {
        return @tagName(self);
    }
};

pub const MlirSessionOptions = struct {
    program_name: []const u8 = "pjrtx",
};

pub const LoweringRegionFact = struct {
    lowering_record_id: compiler_facts.LoweringRecordId,
    fusion_group_index: ?u32,
    placement_record_indices: []const u32,
    logical_tile_shape: []const i64,
    result_memory_space_id: u32,
    tile_memory_space_id: ?u32,
    codegen_region: compiler_facts.LoweringDecision,
    reason: []const u8,
};

pub const CostCapabilityFact = struct {
    graph_instruction_id: compiler_facts.GraphInstructionId,
    expected_unit_id: ?u32,
};

pub const KernelCodegenCapabilityFact = struct {
    graph_instruction_id: compiler_facts.GraphInstructionId,
    backend_operation: []const u8,
    expected_unit_id: ?u32,
};

pub const ExecutableContract = struct {
    target_kind: target_pkg.TargetKind,
    schedule_command_count: u32,
    backend_binding_count: u32,
    kernel_codegen_count: u32,
};

pub const BackendExecutableCallFact = struct {
    index: u32,
    command_id: core.ScheduleCommandId,
    backend_kind: core.BackendKind,
    graph_instruction_id: compiler_facts.GraphInstructionId,
    graph_instruction_ids: []const compiler_facts.GraphInstructionId,
    feature: []const u8,
    backend_operation: []const u8,
    input_value_ids: []const compiler_facts.GraphValueId,
    output_value_ids: []const compiler_facts.GraphValueId,
    expected_unit_id: ?u32,
};

pub const BackendExecutablePlanFact = struct {
    backend_kind: core.BackendKind,
    command_id: core.ScheduleCommandId,
    backend_operation: []const u8,
    calls: []const BackendExecutableCallFact,
};

pub const BackendProfileJoinFact = struct {
    index: u32,
    call_index: u32,
    command_id: core.ScheduleCommandId,
    graph_instruction_ids: []const compiler_facts.GraphInstructionId,
    profile_event_id: core.ProfileEventId,
};

pub const BackendProfileJoinPlanFact = struct {
    joins: []const BackendProfileJoinFact,
};

pub const SchedulePlanFact = struct {
    commands: []core.ScheduleCommand,
    overlaps: []core.ScheduleOverlapRecord,
};

pub const BackendTensorDescriptorFact = struct {
    element_type: core.BufferType,
    dims: []const i64,
    layout: compiler_facts.LayoutKind,
};

pub const BackendKernelGraphNodeFact = struct {
    index: u32,
    call_index: u32,
    graph_instruction_id: compiler_facts.GraphInstructionId,
    graph_instruction_ids: []const compiler_facts.GraphInstructionId,
    feature: []const u8,
    backend_operation: []const u8,
    input_value_ids: []const compiler_facts.GraphValueId,
    output_value_ids: []const compiler_facts.GraphValueId,
    output_type: BackendTensorDescriptorFact,
    attributes: []const u8,
};

pub const BackendKernelGraphEdgeFact = struct {
    value_id: compiler_facts.GraphValueId,
    src_node_index: u32,
    dst_node_index: u32,
};

pub const BackendKernelGraphFact = struct {
    backend_kind: core.BackendKind,
    command_id: core.ScheduleCommandId,
    nodes: []const BackendKernelGraphNodeFact,
    edges: []const BackendKernelGraphEdgeFact,
};

pub const RuntimeAllocationFact = struct {
    index: u32,
    value_id: compiler_facts.GraphValueId,
    placement: []const u8,
    memory_space_id: u32,
    size_bytes: u128,
    first_command_id: core.ScheduleCommandId,
    last_command_id: core.ScheduleCommandId,
};

pub const RuntimeBufferUseFact = struct {
    command_id: core.ScheduleCommandId,
    buffer_index: u32,
    access: []const u8,
};

pub const RuntimeAllocationPlanFact = struct {
    allocations: []const RuntimeAllocationFact,
    command_buffer_uses: []const RuntimeBufferUseFact,
    peak_device_bytes: u128,
};

pub const RuntimeStreamStepFact = struct {
    command_id: core.ScheduleCommandId,
    stream: core.StreamId,
    wait_event_ids: []const u32,
    start_event_id: u32,
    done_event_id: u32,
};

pub const RuntimeStreamPlanFact = struct {
    steps: []const RuntimeStreamStepFact,
};

pub const RuntimeProfileEventFact = struct {
    index: u32,
    command_id: ?core.ScheduleCommandId,
    graph_instruction_ids: []const compiler_facts.GraphInstructionId,
    kind: []const u8,
    start_ns: u64,
    duration_ns: u64,
    bytes: u128,
    logical_ops: u128,
    status: []const u8,
    forced_synchronization: bool,
};

pub const RuntimeProfileFact = struct {
    events: []const RuntimeProfileEventFact,
};

pub const RuntimeProfileJoinFact = struct {
    index: u32,
    subject_kind: []const u8,
    subject_id: u32,
    command_id: ?core.ScheduleCommandId,
    graph_instruction_ids: []const compiler_facts.GraphInstructionId,
    profile_event_ids: []const core.ProfileEventId,
};

pub const RuntimeProfileJoinPlanFact = struct {
    joins: []const RuntimeProfileJoinFact,
};

pub const ExternalPassProbeRecord = struct {
    construct_count: u32,
    initialize_count: u32,
    run_count: u32,
    clone_count: u32,
    destruct_count: u32,
};
