const std = @import("std");
const ir = @import("src/compiler/ir");

const program_liveness = @import("program_liveness.zig");
const program_validation = @import("program_validation.zig");

/// Errors reported by the MLX backend program graph verifier and liveness planner.
pub const Error = error{
    InvalidDeviceCount,
    InvalidProgram,
    UnsupportedElementType,
    ShapeMismatch,
    BufferAllocationFailed,
    CommandSubmissionFailed,
    BufferCopyFailed,
    OutOfMemory,
    InvalidCustomCall,
};

/// Classifies the backend execution role of an executable-plan instruction.
pub const NodeKind = enum {
    constant,
    parameter,
    structural,
    view,
    elementwise,
    reduction,
    matmul,
    control_flow,
    library_call,
    materialize,
};

/// Captures one scheduled backend graph node and its value/subprogram relationships.
pub const Node = struct {
    /// Original executable-plan instruction index represented by this node.
    instruction_index: usize,
    /// Backend execution category used by scheduling and fusion.
    kind: NodeKind,
    /// Program values consumed by this node.
    inputs: []const ir.ValueId,
    /// Program values produced by this node.
    outputs: []const ir.ValueId,
    /// Subprogram indices owned by this node, when it represents control flow.
    subprograms: []const usize = &.{},
    /// Control-flow metadata index owned by this node, when present.
    control_flow: ?usize = null,
    /// Whether executing the node produces device materialization.
    materializes: bool = true,
    /// Fusion group index containing this node, when it is scheduled as part of a group.
    fusion_group: ?usize = null,
};

/// Classifies a group of nodes that the backend may execute as one fused graph segment.
pub const FusionGroupKind = enum {
    view_elementwise,
};

/// Owns metadata for a contiguous fusion group in the backend program schedule.
pub const FusionGroup = struct {
    /// Stable group index inside `Program.fusion_groups`.
    id: usize,
    /// Backend fusion strategy selected for the group.
    kind: FusionGroupKind,
    /// First program node index covered by this group.
    first_node: usize,
    /// Last program node index covered by this group.
    last_node: usize,
    /// Number of nodes covered by this group.
    node_count: usize,
    /// Concrete node indices covered by this group.
    node_indices: []const usize = &.{},
    /// Values entering the group from parameters or outside producers.
    input_values: []const ir.ValueId = &.{},
    /// Values leaving the group for outside consumers or materialization.
    output_values: []const ir.ValueId = &.{},

    /// Releases heap-owned lists attached to a fusion group.
    pub fn deinit(self: FusionGroup, allocator: std.mem.Allocator) void {
        if (self.node_indices.len != 0) allocator.free(self.node_indices);
        if (self.input_values.len != 0) allocator.free(self.input_values);
        if (self.output_values.len != 0) allocator.free(self.output_values);
    }
};

/// Explains why the backend must materialize a program value at a schedule boundary.
pub const MaterializationReason = enum {
    pjrt_output,
    side_effect,
    token,
    donation_alias,
    debug,
    backend_requirement,
};

/// Marks a value that must be evaluated and resident at a schedule boundary.
pub const MaterializationBoundary = struct {
    /// Program value affected by the materialization boundary.
    value_id: ir.ValueId,
    /// Backend or runtime reason for materializing the value.
    reason: MaterializationReason,
};

/// Records ownership, size, and liveness metadata for one backend program value.
pub const Value = struct {
    /// Stable value id from the executable plan.
    value_id: ir.ValueId,
    /// Dense byte size used by liveness accounting.
    byte_size: usize = 0,
    /// Producer node index, or null for external parameters.
    producer_node: ?usize = null,
    /// Last consuming node index, if the value is consumed by the program.
    last_use_node: ?usize = null,
    /// Whether this value is returned as a PJRT output.
    is_output: bool = false,
    /// Materialization boundary index associated with this value, when any.
    materialization_boundary: ?usize = null,
};

/// Describes a dependency edge from one producing node to one consuming node.
pub const Edge = struct {
    /// Value carried across this edge.
    value_id: ir.ValueId,
    /// Producing node index.
    from_node: usize,
    /// Consuming node index.
    to_node: usize,
};

/// Classifies one item in the backend execution schedule.
pub const ScheduleKind = enum {
    node,
    fusion_group,
    materialization_boundary,
};

/// Describes one node, fusion group, or materialization range in execution order.
pub const ScheduleItem = struct {
    /// Kind of scheduled item.
    kind: ScheduleKind,
    /// Node, fusion group, or materialization-boundary start index.
    index: usize,
    /// Number of entries represented by this item.
    count: usize = 1,
};

/// Owns a cloned executable-plan region used by a backend control-flow node.
pub const Subprogram = struct {
    /// Stable subprogram index inside `Program.subprograms`.
    id: usize,
    /// Program node that owns this subprogram.
    parent_node: usize,
    /// Source region id from the executable plan.
    region_id: ir.RegionId,
    /// Semantic role of the source region.
    kind: ir.RegionKind,
    /// Cloned region values, including owned descriptor dimensions and literals.
    values: []const ir.RegionValue = &.{},
    /// Cloned argument descriptors.
    argument_descriptors: []const ir.BufferDescriptor = &.{},
    /// Cloned region instructions.
    instructions: []const ir.RegionInstruction = &.{},
    /// Cloned return descriptors.
    return_descriptors: []const ir.BufferDescriptor = &.{},
    /// Region terminator operands.
    terminator_operands: []const ir.RegionValueId = &.{},
    /// Cloned descriptors for terminator operands.
    terminator_operand_descriptors: []const ir.BufferDescriptor = &.{},

    /// Releases all heap-owned clones attached to this subprogram.
    pub fn deinit(self: Subprogram, allocator: std.mem.Allocator) void {
        freeRegionValueList(allocator, self.values);
        freeDescriptorList(allocator, self.argument_descriptors);
        freeRegionInstructionList(allocator, self.instructions);
        freeDescriptorList(allocator, self.return_descriptors);
        if (self.terminator_operands.len != 0) allocator.free(self.terminator_operands);
        freeDescriptorList(allocator, self.terminator_operand_descriptors);
    }
};

/// Classifies backend-native control-flow metadata.
pub const ControlFlowKind = enum {
    while_loop,
};

/// Owns backend scheduling metadata for a control-flow node.
pub const ControlFlow = struct {
    /// Stable control-flow index inside `Program.control_flows`.
    id: usize,
    /// Program node that owns this control-flow metadata.
    parent_node: usize,
    /// Backend control-flow strategy.
    kind: ControlFlowKind,
    /// Condition subprogram index.
    condition_subprogram: usize,
    /// Body subprogram index.
    body_subprogram: usize,
    /// Loop state values consumed by the parent node.
    state_inputs: []const ir.ValueId = &.{},
    /// Loop state values produced by the parent node.
    state_outputs: []const ir.ValueId = &.{},
    /// Predicate value produced by the condition subprogram.
    predicate_output: ir.RegionValueId = ir.RegionValueId.invalid,

    /// Releases heap-owned loop state lists attached to this control-flow record.
    pub fn deinit(self: ControlFlow, allocator: std.mem.Allocator) void {
        if (self.state_inputs.len != 0) allocator.free(self.state_inputs);
        if (self.state_outputs.len != 0) allocator.free(self.state_outputs);
    }
};

/// Summarizes planned releases and peak live memory for a backend program.
pub const LivenessStats = struct {
    /// Count of intermediate values the schedule can release before completion.
    planned_release_count: usize = 0,
    /// Total bytes associated with planned intermediate releases.
    planned_release_bytes: usize = 0,
    /// Maximum number of simultaneously live values in the schedule.
    peak_live_value_count: usize = 0,
    /// Maximum live bytes in the schedule.
    peak_live_bytes: usize = 0,
};

/// Owns the MLX backend program graph derived from an executable plan.
pub const Program = struct {
    /// Allocator used for all owned program graph storage.
    allocator: std.mem.Allocator,
    /// Value table indexed by executable-plan value id.
    values: []Value,
    /// Node table indexed by executable-plan instruction index.
    nodes: []Node,
    /// Producer-to-consumer dependency edges.
    edges: []Edge,
    /// Execution schedule over nodes, fusion groups, and materialization boundaries.
    schedule: []ScheduleItem,
    /// Control-flow subprograms cloned from executable-plan regions.
    subprograms: []Subprogram = &.{},
    /// Control-flow metadata records owned by control-flow nodes.
    control_flows: []ControlFlow = &.{},
    /// Fusion groups scheduled as single backend graph segments.
    fusion_groups: []FusionGroup,
    /// Values that must be materialized at schedule boundaries.
    materialization_boundaries: []MaterializationBoundary,
    /// Number of fusion groups discovered during construction.
    fusion_group_count: usize = 0,

    /// Verifies structural invariants without producing a diagnostic string.
    pub fn validate(self: *const Program) Error!void {
        return program_validation.validate(self, null) catch |err| switch (err) {
            error.InvalidProgram => error.InvalidProgram,
            error.OutOfMemory => unreachable,
            else => unreachable,
        };
    }

    /// Computes planned release counts and peak live value/byte counts for the schedule.
    pub fn livenessStats(self: *const Program) Error!LivenessStats {
        return program_liveness.compute(self);
    }

    /// Verifies structural invariants and writes a backend-program diagnostic on failure.
    pub fn validateWithWriter(self: *const Program, writer: ?*std.Io.Writer) (Error || std.Io.Writer.Error)!void {
        return program_validation.validate(self, writer);
    }

    /// Releases all heap-owned graph storage and invalidates this program.
    pub fn deinit(self: *Program) void {
        for (self.nodes) |node| {
            self.allocator.free(node.inputs);
            self.allocator.free(node.outputs);
            if (node.subprograms.len != 0) self.allocator.free(node.subprograms);
        }
        for (self.control_flows) |control_flow| control_flow.deinit(self.allocator);
        for (self.subprograms) |subprogram| subprogram.deinit(self.allocator);
        for (self.fusion_groups) |group| group.deinit(self.allocator);
        self.allocator.free(self.materialization_boundaries);
        self.allocator.free(self.fusion_groups);
        if (self.control_flows.len != 0) self.allocator.free(self.control_flows);
        if (self.subprograms.len != 0) self.allocator.free(self.subprograms);
        self.allocator.free(self.schedule);
        self.allocator.free(self.edges);
        self.allocator.free(self.nodes);
        self.allocator.free(self.values);
        self.* = undefined;
    }
};

pub fn freeDescriptorList(allocator: std.mem.Allocator, descriptors: []const ir.BufferDescriptor) void {
    for (descriptors) |descriptor| {
        if (descriptor.dims.len != 0) allocator.free(descriptor.dims);
    }
    if (descriptors.len != 0) allocator.free(descriptors);
}

pub fn freeRegionValueList(allocator: std.mem.Allocator, values: []const ir.RegionValue) void {
    for (values) |value| {
        if (value.descriptor.dims.len != 0) allocator.free(value.descriptor.dims);
        if (value.literal) |literal| allocator.free(literal);
    }
    if (values.len != 0) allocator.free(values);
}

pub fn freeRegionInstructionList(allocator: std.mem.Allocator, instructions: []const ir.RegionInstruction) void {
    for (instructions) |instruction| {
        if (instruction.inputs.len != 0) allocator.free(instruction.inputs);
        if (instruction.outputs.len != 0) allocator.free(instruction.outputs);
        freeDescriptorList(allocator, instruction.operand_descriptors);
        freeDescriptorList(allocator, instruction.result_descriptors);
    }
    if (instructions.len != 0) allocator.free(instructions);
}
