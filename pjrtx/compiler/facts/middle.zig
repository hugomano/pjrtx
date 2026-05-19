const ids = @import("ids.zig");
const graph = @import("graph.zig");
const target_pkg = @import("pjrtx/target");

/// Compiler-middle facts are extracted views of verified MLIR state. They
/// describe decisions that affect performance and correctness before backend or
/// runtime packages consume them.
pub const MlirPassStatus = enum {
    ok,
    skipped,
    failed,
};

pub const MlirPassRecord = struct {
    index: u32,
    pass_name: []const u8,
    status: MlirPassStatus,
    input_fingerprint: u64,
    output_fingerprint: u64,
    preserves_source_provenance: bool,
    preserves_shardy_metadata: bool,
    reason: []const u8,
};

pub const GraphRewriteDecision = enum {
    applied,
    rejected,
};

pub const GraphRewriteRecord = struct {
    index: u32,
    pass_name: []const u8,
    decision: GraphRewriteDecision,
    input_instruction_id: ids.GraphInstructionId,
    output_instruction_id: ?ids.GraphInstructionId,
    replaced_value_id: ?ids.GraphValueId,
    replacement_value_id: ?ids.GraphValueId,
    reason: []const u8,
};

pub const FusionDecision = enum {
    accepted,
    rejected,
};

pub const FusionPressureDelta = struct {
    split_kernel_count: u32 = 0,
    fused_kernel_count: u32 = 0,
    split_peak_live_bytes: u128 = 0,
    fused_live_bytes: u128 = 0,
    additional_live_bytes: u128 = 0,
    global_bytes_saved: u128 = 0,
};

pub const FusionGroup = struct {
    index: u32,
    decision: FusionDecision,
    kind: []const u8,
    graph_instruction_ids: []const ids.GraphInstructionId,
    bytes_saved: u128,
    launch_count_reduction: u32,
    pressure_delta: FusionPressureDelta = .{},
    reason: []const u8,
};

pub const PlacementRecord = struct {
    index: u32,
    graph_instruction_id: ids.GraphInstructionId,
    output_value_ids: []const ids.GraphValueId,
    layout: graph.LayoutKind,
    logical_tile_shape: []const i64,
    result_memory_space_id: u32,
    tile_memory_space_id: ?u32,
    reason: []const u8,
};

pub const CollectivePlanDecision = enum {
    no_collectives,
    selected,
    rejected,
    unsupported,
};

pub const CollectiveAlgorithm = enum {
    none,
    direct,
    ring,
    tree,
    split,
};

pub const CollectivePlanRecord = struct {
    index: u32,
    decision: CollectivePlanDecision,
    algorithm: CollectiveAlgorithm,
    checked_graph_instruction_count: u32,
    lowered_collective_count: u32,
    unsupported_collective_count: u32,
    estimated_bytes: u128,
    estimated_latency_ns: ?u64,
    reason: []const u8,
};

pub const CostOpClass = enum {
    matmul,
    elementwise,
    transcendental,
    transfer,
    backend_kernel,
};

pub const CostLedgerEntry = struct {
    id: ids.CostLedgerId,
    source: ?graph.SourceRef,
    graph_instruction_ids: []const ids.GraphInstructionId,
    op_class: CostOpClass,
    dtype: target_pkg.BufferType,
    accumulation_dtype: ?target_pkg.BufferType,
    logical_ops: u128,
    bytes_read: u128,
    bytes_written: u128,
    expected_unit_id: ?u32,
    formula: []const u8,
    approximation: []const u8,
};

pub const MemoryTrafficKind = enum {
    global_memory,
    local_memory,
    host_device_dma,
    interconnect,
};

pub const MemoryTrafficRecord = struct {
    id: ids.MemoryTrafficId,
    lowering_record_id: ids.LoweringRecordId,
    memory_space_id: u32,
    kind: MemoryTrafficKind,
    graph_instruction_ids: []const ids.GraphInstructionId,
    cost_ledger_ids: []const ids.CostLedgerId,
    bytes_read: u128,
    bytes_written: u128,
    reason: []const u8,
};

pub const LoweringDecision = enum {
    backend_kernel_graph,
    elementwise_fusion,
    transfer,
    unsupported,
};

pub const LoweringRecord = struct {
    id: ids.LoweringRecordId,
    graph_instruction_ids: []const ids.GraphInstructionId,
    decision: LoweringDecision,
    reason: []const u8,
    rejected_alternatives: []const []const u8,
    cost_ledger_ids: []const ids.CostLedgerId,
};
