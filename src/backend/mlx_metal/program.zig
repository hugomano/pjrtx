const std = @import("std");
const ir = @import("src/compiler/ir");

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
        return self.validateWithWriter(null) catch |err| switch (err) {
            error.InvalidProgram => error.InvalidProgram,
            error.OutOfMemory => unreachable,
            else => unreachable,
        };
    }

    /// Computes planned release counts and peak live value/byte counts for the schedule.
    pub fn livenessStats(self: *const Program) Error!LivenessStats {
        const live_values = self.allocator.alloc(bool, self.values.len) catch return error.OutOfMemory;
        defer self.allocator.free(live_values);
        @memset(live_values, false);

        var live_count: usize = 0;
        var live_bytes: usize = 0;
        for (self.values, 0..) |value, value_index| {
            if (value.producer_node != null) continue;
            if (value.last_use_node == null and !value.is_output) continue;
            live_values[value_index] = true;
            live_count += 1;
            live_bytes += value.byte_size;
        }

        var stats = LivenessStats{
            .peak_live_value_count = live_count,
            .peak_live_bytes = live_bytes,
        };
        for (self.schedule) |item| {
            switch (item.kind) {
                .node => {
                    if (item.index >= self.nodes.len) return error.InvalidProgram;
                    const node = self.nodes[item.index];
                    try self.markNodeOutputsLive(node, live_values, &live_count, &live_bytes, &stats);
                    try self.releaseDeadNodeInputs(node, item.index, live_values, &live_count, &live_bytes, &stats);
                },
                .fusion_group => {
                    if (item.index >= self.fusion_groups.len) return error.InvalidProgram;
                    const group = self.fusion_groups[item.index];
                    if (item.count != group.node_indices.len) return error.InvalidProgram;
                    for (group.node_indices) |node_index| {
                        if (node_index >= self.nodes.len) return error.InvalidProgram;
                        try self.markNodeOutputsLive(self.nodes[node_index], live_values, &live_count, &live_bytes, &stats);
                    }
                    for (group.node_indices) |node_index| {
                        if (node_index >= self.nodes.len) return error.InvalidProgram;
                        try self.releaseDeadFusionNodeInputs(self.nodes[node_index], group.last_node, live_values, &live_count, &live_bytes, &stats);
                    }
                },
                .materialization_boundary => {},
            }
        }

        return stats;
    }

    /// Verifies structural invariants and writes a backend-program diagnostic on failure.
    pub fn validateWithWriter(self: *const Program, writer: ?*std.Io.Writer) (Error || std.Io.Writer.Error)!void {
        if (self.fusion_group_count != self.fusion_groups.len) {
            return invalidProgram(writer, "fusion_group_count={} does not match fusion_groups.len={}", .{ self.fusion_group_count, self.fusion_groups.len });
        }

        for (self.values, 0..) |value, value_index| {
            if (value.value_id.index != value_index) {
                return invalidProgram(writer, "value table slot {} contains value_id={}", .{ value_index, value.value_id.index });
            }
            if (value.producer_node) |producer_node| {
                if (producer_node >= self.nodes.len) {
                    return invalidProgram(writer, "value {} producer_node={} is outside nodes.len={}", .{ value_index, producer_node, self.nodes.len });
                }
            }
            if (value.last_use_node) |last_use_node| {
                if (last_use_node >= self.nodes.len) {
                    return invalidProgram(writer, "value {} last_use_node={} is outside nodes.len={}", .{ value_index, last_use_node, self.nodes.len });
                }
            }
            if (value.materialization_boundary) |boundary_index| {
                if (boundary_index >= self.materialization_boundaries.len) {
                    return invalidProgram(writer, "value {} materialization_boundary={} is outside materialization_boundaries.len={}", .{ value_index, boundary_index, self.materialization_boundaries.len });
                }
            }
        }

        for (self.nodes, 0..) |node, node_index| {
            for (node.inputs) |input| {
                if (input.index >= self.values.len) {
                    return invalidProgram(writer, "node {} input value_id={} is outside values.len={}", .{ node_index, input.index, self.values.len });
                }
            }
            for (node.outputs) |output| {
                if (output.index >= self.values.len) {
                    return invalidProgram(writer, "node {} output value_id={} is outside values.len={}", .{ node_index, output.index, self.values.len });
                }
            }
            if (node.fusion_group) |group_index| {
                if (group_index >= self.fusion_groups.len) {
                    return invalidProgram(writer, "node {} fusion_group={} is outside fusion_groups.len={}", .{ node_index, group_index, self.fusion_groups.len });
                }
            }
            if (node.control_flow) |control_flow_index| {
                if (control_flow_index >= self.control_flows.len) {
                    return invalidProgram(writer, "node {} control_flow={} is outside control_flows.len={}", .{ node_index, control_flow_index, self.control_flows.len });
                }
                if (self.control_flows[control_flow_index].parent_node != node_index) {
                    return invalidProgram(writer, "node {} control_flow={} parent_node metadata does not point back to the node", .{ node_index, control_flow_index });
                }
                if (node.kind != .control_flow) {
                    return invalidProgram(writer, "node {} owns control_flow={} but is not a control_flow node", .{ node_index, control_flow_index });
                }
            }
            for (node.subprograms) |subprogram_index| {
                if (subprogram_index >= self.subprograms.len) {
                    return invalidProgram(writer, "node {} subprogram={} is outside subprograms.len={}", .{ node_index, subprogram_index, self.subprograms.len });
                }
                if (self.subprograms[subprogram_index].parent_node != node_index) {
                    return invalidProgram(writer, "node {} subprogram={} parent_node metadata does not point back to the node", .{ node_index, subprogram_index });
                }
                if (node.kind != .control_flow) {
                    return invalidProgram(writer, "node {} owns subprogram={} but is not a control_flow node", .{ node_index, subprogram_index });
                }
            }
        }

        for (self.control_flows, 0..) |control_flow, control_flow_index| {
            if (control_flow.id != control_flow_index) {
                return invalidProgram(writer, "control_flow slot {} contains id={}", .{ control_flow_index, control_flow.id });
            }
            if (control_flow.parent_node >= self.nodes.len) {
                return invalidProgram(writer, "control_flow {} parent_node={} is outside nodes.len={}", .{ control_flow_index, control_flow.parent_node, self.nodes.len });
            }
            const parent = self.nodes[control_flow.parent_node];
            if (parent.control_flow != control_flow_index) {
                return invalidProgram(writer, "control_flow {} is not referenced by parent node {}", .{ control_flow_index, control_flow.parent_node });
            }
            if (control_flow.condition_subprogram >= self.subprograms.len or control_flow.body_subprogram >= self.subprograms.len) {
                return invalidProgram(writer, "control_flow {} references subprograms outside subprograms.len={}", .{ control_flow_index, self.subprograms.len });
            }
            const condition = self.subprograms[control_flow.condition_subprogram];
            const body = self.subprograms[control_flow.body_subprogram];
            if (condition.parent_node != control_flow.parent_node or body.parent_node != control_flow.parent_node) {
                return invalidProgram(writer, "control_flow {} subprogram parent metadata does not match parent node", .{control_flow_index});
            }
            if (condition.kind != .while_cond or body.kind != .while_body) {
                return invalidProgram(writer, "control_flow {} while scheduler requires cond/body subprogram kinds", .{control_flow_index});
            }
            if (control_flow.state_inputs.len != parent.inputs.len or control_flow.state_outputs.len != parent.outputs.len) {
                return invalidProgram(writer, "control_flow {} state arity does not match parent node inputs/outputs", .{control_flow_index});
            }
            if (condition.argument_descriptors.len != control_flow.state_inputs.len or body.argument_descriptors.len != control_flow.state_inputs.len) {
                return invalidProgram(writer, "control_flow {} cond/body argument arity does not match loop state", .{control_flow_index});
            }
            if (body.terminator_operands.len != control_flow.state_outputs.len) {
                return invalidProgram(writer, "control_flow {} body terminator arity does not match loop outputs", .{control_flow_index});
            }
            if (condition.terminator_operands.len != 1 or control_flow.predicate_output.index >= condition.values.len) {
                return invalidProgram(writer, "control_flow {} condition must expose one predicate terminator value", .{control_flow_index});
            }
            if (condition.terminator_operands[0].index != control_flow.predicate_output.index) {
                return invalidProgram(writer, "control_flow {} predicate output does not match condition terminator", .{control_flow_index});
            }
            const predicate = condition.values[control_flow.predicate_output.index].descriptor;
            if (predicate.element_type != .pred or predicate.dims.len != 0) {
                return invalidProgram(writer, "control_flow {} condition predicate must be scalar pred", .{control_flow_index});
            }
            for (control_flow.state_inputs) |state_input| {
                if (state_input.index >= self.values.len) {
                    return invalidProgram(writer, "control_flow {} state input value_id={} is outside values.len={}", .{ control_flow_index, state_input.index, self.values.len });
                }
            }
            for (control_flow.state_outputs) |state_output| {
                if (state_output.index >= self.values.len) {
                    return invalidProgram(writer, "control_flow {} state output value_id={} is outside values.len={}", .{ control_flow_index, state_output.index, self.values.len });
                }
            }
        }

        for (self.subprograms, 0..) |subprogram, subprogram_index| {
            if (subprogram.id != subprogram_index) {
                return invalidProgram(writer, "subprogram slot {} contains id={}", .{ subprogram_index, subprogram.id });
            }
            for (subprogram.values, 0..) |value, value_index| {
                if (value.id.index != value_index) {
                    return invalidProgram(writer, "subprogram {} value slot {} contains value_id={}", .{ subprogram_index, value_index, value.id.index });
                }
            }
            if (subprogram.parent_node >= self.nodes.len) {
                return invalidProgram(writer, "subprogram {} parent_node={} is outside nodes.len={}", .{ subprogram_index, subprogram.parent_node, self.nodes.len });
            }
            if (!containsUsize(self.nodes[subprogram.parent_node].subprograms, subprogram_index)) {
                return invalidProgram(writer, "subprogram {} is not referenced by parent node {}", .{ subprogram_index, subprogram.parent_node });
            }
            if (subprogram.return_descriptors.len != subprogram.terminator_operand_descriptors.len) {
                return invalidProgram(writer, "subprogram {} return descriptor count={} does not match terminator operand count={}", .{
                    subprogram_index,
                    subprogram.return_descriptors.len,
                    subprogram.terminator_operand_descriptors.len,
                });
            }
            if (subprogram.return_descriptors.len != subprogram.terminator_operands.len) {
                return invalidProgram(writer, "subprogram {} return descriptor count={} does not match terminator operand id count={}", .{
                    subprogram_index,
                    subprogram.return_descriptors.len,
                    subprogram.terminator_operands.len,
                });
            }
            for (subprogram.terminator_operands) |operand| {
                if (operand.index >= subprogram.values.len) {
                    return invalidProgram(writer, "subprogram {} terminator operand value_id={} is outside values.len={}", .{ subprogram_index, operand.index, subprogram.values.len });
                }
            }
            for (subprogram.instructions, 0..) |instruction, instruction_index| {
                if (instruction.inputs.len != instruction.operand_descriptors.len) {
                    return invalidProgram(writer, "subprogram {} instruction {} input count={} does not match operand descriptor count={}", .{
                        subprogram_index,
                        instruction_index,
                        instruction.inputs.len,
                        instruction.operand_descriptors.len,
                    });
                }
                if (instruction.outputs.len != instruction.result_descriptors.len) {
                    return invalidProgram(writer, "subprogram {} instruction {} output count={} does not match result descriptor count={}", .{
                        subprogram_index,
                        instruction_index,
                        instruction.outputs.len,
                        instruction.result_descriptors.len,
                    });
                }
                for (instruction.inputs) |input| {
                    if (input.index >= subprogram.values.len) {
                        return invalidProgram(writer, "subprogram {} instruction {} input value_id={} is outside values.len={}", .{ subprogram_index, instruction_index, input.index, subprogram.values.len });
                    }
                }
                for (instruction.outputs) |output| {
                    if (output.index >= subprogram.values.len) {
                        return invalidProgram(writer, "subprogram {} instruction {} output value_id={} is outside values.len={}", .{ subprogram_index, instruction_index, output.index, subprogram.values.len });
                    }
                }
            }
        }

        const computed_producers = try self.allocator.alloc(?usize, self.values.len);
        defer self.allocator.free(computed_producers);
        @memset(computed_producers, null);
        const computed_last_uses = try self.allocator.alloc(?usize, self.values.len);
        defer self.allocator.free(computed_last_uses);
        @memset(computed_last_uses, null);

        for (self.nodes, 0..) |node, node_index| {
            for (node.outputs) |output| {
                if (computed_producers[output.index]) |previous_node| {
                    return invalidProgram(writer, "value {} is produced by both node {} and node {}", .{ output.index, previous_node, node_index });
                }
                computed_producers[output.index] = node_index;
            }
            for (node.inputs) |input| {
                computed_last_uses[input.index] = node_index;
            }
        }

        for (self.values, 0..) |value, value_index| {
            if (value.producer_node != computed_producers[value_index]) {
                return invalidProgram(writer, "value {} producer metadata does not match node outputs", .{value_index});
            }
            if (value.last_use_node != computed_last_uses[value_index]) {
                return invalidProgram(writer, "value {} last-use metadata does not match node inputs", .{value_index});
            }
        }

        for (self.edges, 0..) |edge, edge_index| {
            if (edge.value_id.index >= self.values.len) {
                return invalidProgram(writer, "edge {} value_id={} is outside values.len={}", .{ edge_index, edge.value_id.index, self.values.len });
            }
            if (edge.from_node >= self.nodes.len or edge.to_node >= self.nodes.len) {
                return invalidProgram(writer, "edge {} references nodes from={} to={} outside nodes.len={}", .{ edge_index, edge.from_node, edge.to_node, self.nodes.len });
            }
            if (self.values[edge.value_id.index].producer_node != edge.from_node) {
                return invalidProgram(writer, "edge {} value_id={} producer does not match from_node={}", .{ edge_index, edge.value_id.index, edge.from_node });
            }
            if (!containsValueId(self.nodes[edge.from_node].outputs, edge.value_id)) {
                return invalidProgram(writer, "edge {} value_id={} is not produced by from_node={}", .{ edge_index, edge.value_id.index, edge.from_node });
            }
            if (!containsValueId(self.nodes[edge.to_node].inputs, edge.value_id)) {
                return invalidProgram(writer, "edge {} value_id={} is not consumed by to_node={}", .{ edge_index, edge.value_id.index, edge.to_node });
            }
        }

        for (self.nodes, 0..) |node, node_index| {
            for (node.inputs) |input| {
                const producer_node = self.values[input.index].producer_node orelse continue;
                if (!self.hasEdge(input, producer_node, node_index)) {
                    return invalidProgram(writer, "node {} input value_id={} is missing producer edge from node {}", .{ node_index, input.index, producer_node });
                }
            }
        }

        for (self.fusion_groups, 0..) |group, group_index| {
            if (group.id != group_index) {
                return invalidProgram(writer, "fusion group slot {} contains id={}", .{ group_index, group.id });
            }
            if (group.node_count == 0) {
                return invalidProgram(writer, "fusion group {} is empty", .{group_index});
            }
            if (group.node_count != group.node_indices.len) {
                return invalidProgram(writer, "fusion group {} node_count={} does not match node_indices.len={}", .{ group_index, group.node_count, group.node_indices.len });
            }
            if (group.first_node >= self.nodes.len or group.last_node >= self.nodes.len) {
                return invalidProgram(writer, "fusion group {} range first={} last={} is outside nodes.len={}", .{ group_index, group.first_node, group.last_node, self.nodes.len });
            }
            if (group.last_node < group.first_node) {
                return invalidProgram(writer, "fusion group {} range is reversed first={} last={}", .{ group_index, group.first_node, group.last_node });
            }

            for (group.node_indices) |node_index| {
                if (node_index >= self.nodes.len) {
                    return invalidProgram(writer, "fusion group {} node_index={} is outside nodes.len={}", .{ group_index, node_index, self.nodes.len });
                }
                if (node_index < group.first_node or node_index > group.last_node) {
                    return invalidProgram(writer, "fusion group {} node_index={} is outside group range {}..{}", .{ group_index, node_index, group.first_node, group.last_node });
                }
                if (self.nodes[node_index].fusion_group != group_index) {
                    return invalidProgram(writer, "fusion group {} includes node {} whose node.fusion_group is not this group", .{ group_index, node_index });
                }
            }
            for (group.input_values) |value_id| {
                if (value_id.index >= self.values.len) {
                    return invalidProgram(writer, "fusion group {} input value_id={} is outside values.len={}", .{ group_index, value_id.index, self.values.len });
                }
            }
            for (group.output_values) |value_id| {
                if (value_id.index >= self.values.len) {
                    return invalidProgram(writer, "fusion group {} output value_id={} is outside values.len={}", .{ group_index, value_id.index, self.values.len });
                }
            }
        }

        for (self.materialization_boundaries, 0..) |boundary, boundary_index| {
            if (boundary.value_id.index >= self.values.len) {
                return invalidProgram(writer, "materialization boundary {} value_id={} is outside values.len={}", .{ boundary_index, boundary.value_id.index, self.values.len });
            }
            if (self.values[boundary.value_id.index].materialization_boundary != boundary_index) {
                return invalidProgram(writer, "materialization boundary {} is not referenced by value {}", .{ boundary_index, boundary.value_id.index });
            }
            if (boundary.reason == .pjrt_output and !self.values[boundary.value_id.index].is_output) {
                return invalidProgram(writer, "materialization boundary {} marks non-output value {} as pjrt_output", .{ boundary_index, boundary.value_id.index });
            }
        }

        for (self.values, 0..) |value, value_index| {
            if (!value.is_output) continue;
            const boundary_index = value.materialization_boundary orelse
                return invalidProgram(writer, "output value {} has no materialization boundary", .{value_index});
            if (self.materialization_boundaries[boundary_index].reason != .pjrt_output) {
                return invalidProgram(writer, "output value {} materialization boundary reason is not pjrt_output", .{value_index});
            }
        }

        const fusion_mark_count = self.fusion_groups.len * self.values.len;
        const computed_group_inputs = try self.allocator.alloc(bool, fusion_mark_count);
        defer self.allocator.free(computed_group_inputs);
        @memset(computed_group_inputs, false);
        const computed_group_outputs = try self.allocator.alloc(bool, fusion_mark_count);
        defer self.allocator.free(computed_group_outputs);
        @memset(computed_group_outputs, false);

        for (self.nodes) |node| {
            const group_index = node.fusion_group orelse continue;
            for (node.inputs) |input| {
                if (self.values[input.index].producer_node == null) {
                    computed_group_inputs[Program.groupMarkIndex(self.values.len, group_index, input.index)] = true;
                }
            }
        }
        for (self.edges) |edge| {
            const from_group = self.nodes[edge.from_node].fusion_group;
            const to_group = self.nodes[edge.to_node].fusion_group;
            if (to_group) |group_index| {
                if (from_group != group_index) {
                    computed_group_inputs[Program.groupMarkIndex(self.values.len, group_index, edge.value_id.index)] = true;
                }
            }
            if (from_group) |group_index| {
                if (to_group != group_index) {
                    computed_group_outputs[Program.groupMarkIndex(self.values.len, group_index, edge.value_id.index)] = true;
                }
            }
        }
        for (self.materialization_boundaries) |boundary| {
            const producer_node = self.values[boundary.value_id.index].producer_node orelse continue;
            const group_index = self.nodes[producer_node].fusion_group orelse continue;
            computed_group_outputs[Program.groupMarkIndex(self.values.len, group_index, boundary.value_id.index)] = true;
        }
        for (self.fusion_groups, 0..) |group, group_index| {
            try validateMarkedValueSet(
                writer,
                "input_values",
                group_index,
                group.input_values,
                self.values,
                computed_group_inputs[Program.groupMarkIndex(self.values.len, group_index, 0)..][0..self.values.len],
            );
            try validateMarkedValueSet(
                writer,
                "output_values",
                group_index,
                group.output_values,
                self.values,
                computed_group_outputs[Program.groupMarkIndex(self.values.len, group_index, 0)..][0..self.values.len],
            );
        }

        const scheduled_nodes = try self.allocator.alloc(bool, self.nodes.len);
        defer self.allocator.free(scheduled_nodes);
        @memset(scheduled_nodes, false);
        const schedule_ranks = try self.allocator.alloc(usize, self.nodes.len);
        defer self.allocator.free(schedule_ranks);
        @memset(schedule_ranks, std.math.maxInt(usize));
        const scheduled_boundaries = try self.allocator.alloc(bool, self.materialization_boundaries.len);
        defer self.allocator.free(scheduled_boundaries);
        @memset(scheduled_boundaries, false);
        const boundary_schedule_ranks = try self.allocator.alloc(usize, self.materialization_boundaries.len);
        defer self.allocator.free(boundary_schedule_ranks);
        @memset(boundary_schedule_ranks, std.math.maxInt(usize));
        var next_schedule_rank: usize = 0;

        for (self.schedule, 0..) |item, schedule_index| {
            if (item.count == 0) {
                return invalidProgram(writer, "schedule item {} has count=0", .{schedule_index});
            }
            switch (item.kind) {
                .node => {
                    if (item.index >= self.nodes.len or item.count != 1) {
                        return invalidProgram(writer, "schedule item {} invalid node index={} count={} nodes.len={}", .{ schedule_index, item.index, item.count, self.nodes.len });
                    }
                    if (self.nodes[item.index].fusion_group != null) {
                        return invalidProgram(writer, "schedule item {} directly schedules fused node {}", .{ schedule_index, item.index });
                    }
                    if (scheduled_nodes[item.index]) {
                        return invalidProgram(writer, "schedule item {} schedules node {} more than once", .{ schedule_index, item.index });
                    }
                    scheduled_nodes[item.index] = true;
                    schedule_ranks[item.index] = next_schedule_rank;
                    next_schedule_rank += 1;
                },
                .fusion_group => {
                    if (item.index >= self.fusion_groups.len) {
                        return invalidProgram(writer, "schedule item {} fusion_group={} is outside fusion_groups.len={}", .{ schedule_index, item.index, self.fusion_groups.len });
                    }
                    if (item.count != self.fusion_groups[item.index].node_indices.len) {
                        return invalidProgram(writer, "schedule item {} fusion_group={} count={} does not match node_indices.len={}", .{ schedule_index, item.index, item.count, self.fusion_groups[item.index].node_indices.len });
                    }
                    for (self.fusion_groups[item.index].node_indices) |node_index| {
                        if (scheduled_nodes[node_index]) {
                            return invalidProgram(writer, "schedule item {} schedules fusion group {} node {} more than once", .{ schedule_index, item.index, node_index });
                        }
                        scheduled_nodes[node_index] = true;
                        schedule_ranks[node_index] = next_schedule_rank;
                        next_schedule_rank += 1;
                    }
                },
                .materialization_boundary => {
                    if (item.index > self.materialization_boundaries.len) {
                        return invalidProgram(writer, "schedule item {} materialization index={} is outside materialization_boundaries.len={}", .{ schedule_index, item.index, self.materialization_boundaries.len });
                    }
                    if (item.count > self.materialization_boundaries.len - item.index) {
                        return invalidProgram(writer, "schedule item {} materialization range index={} count={} exceeds materialization_boundaries.len={}", .{ schedule_index, item.index, item.count, self.materialization_boundaries.len });
                    }
                    for (item.index..item.index + item.count) |boundary_index| {
                        if (scheduled_boundaries[boundary_index]) {
                            return invalidProgram(writer, "schedule item {} schedules materialization boundary {} more than once", .{ schedule_index, boundary_index });
                        }
                        scheduled_boundaries[boundary_index] = true;
                        boundary_schedule_ranks[boundary_index] = next_schedule_rank;
                        next_schedule_rank += 1;
                    }
                },
            }
        }

        for (scheduled_nodes, 0..) |scheduled, node_index| {
            if (!scheduled) {
                return invalidProgram(writer, "node {} is not covered by schedule", .{node_index});
            }
        }

        for (scheduled_boundaries, 0..) |scheduled, boundary_index| {
            if (!scheduled) {
                return invalidProgram(writer, "materialization boundary {} is not covered by schedule", .{boundary_index});
            }
            const value_id = self.materialization_boundaries[boundary_index].value_id;
            if (self.values[value_id.index].producer_node) |producer_node| {
                if (schedule_ranks[producer_node] >= boundary_schedule_ranks[boundary_index]) {
                    return invalidProgram(writer, "materialization boundary {} value_id={} is scheduled before producer node {}", .{ boundary_index, value_id.index, producer_node });
                }
            }
        }

        for (self.edges, 0..) |edge, edge_index| {
            if (schedule_ranks[edge.from_node] >= schedule_ranks[edge.to_node]) {
                return invalidProgram(writer, "edge {} value_id={} violates schedule order: producer node {} rank {} must run before consumer node {} rank {}", .{
                    edge_index,
                    edge.value_id.index,
                    edge.from_node,
                    schedule_ranks[edge.from_node],
                    edge.to_node,
                    schedule_ranks[edge.to_node],
                });
            }
        }
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

    fn hasEdge(self: *const Program, value_id: ir.ValueId, from_node: usize, to_node: usize) bool {
        for (self.edges) |edge| {
            if (edge.value_id.index == value_id.index and edge.from_node == from_node and edge.to_node == to_node) return true;
        }
        return false;
    }

    fn containsValueId(values: []const ir.ValueId, value_id: ir.ValueId) bool {
        for (values) |candidate| {
            if (candidate.index == value_id.index) return true;
        }
        return false;
    }

    fn containsUsize(values: []const usize, value: usize) bool {
        for (values) |candidate| {
            if (candidate == value) return true;
        }
        return false;
    }

    fn groupMarkIndex(value_count: usize, group_index: usize, value_index: usize) usize {
        return group_index * value_count + value_index;
    }

    fn validateMarkedValueSet(
        writer: ?*std.Io.Writer,
        comptime label: []const u8,
        group_index: usize,
        declared: []const ir.ValueId,
        values: []const Value,
        marks: []const bool,
    ) (Error || std.Io.Writer.Error)!void {
        var marked_count: usize = 0;
        for (marks) |mark| {
            if (mark) marked_count += 1;
        }
        if (declared.len != marked_count) {
            return invalidProgram(writer, "fusion group {} {s} count={} does not match computed count={}", .{ group_index, label, declared.len, marked_count });
        }
        for (declared) |value_id| {
            if (!marks[value_id.index]) {
                return invalidProgram(writer, "fusion group {} {s} contains stale value_id={}", .{ group_index, label, value_id.index });
            }
        }
        for (values, marks) |value, mark| {
            if (!mark) continue;
            if (!containsValueId(declared, value.value_id)) {
                return invalidProgram(writer, "fusion group {} {s} is missing value_id={}", .{ group_index, label, value.value_id.index });
            }
        }
    }

    fn invalidProgram(writer: ?*std.Io.Writer, comptime detail_fmt: []const u8, args: anytype) (Error || std.Io.Writer.Error)!void {
        if (writer) |w| {
            try w.writeAll("invalid backend program: pass=backend-program-verify feature=backend-program detail=\"");
            try w.print(detail_fmt, args);
            try w.writeAll("\"");
        }
        return error.InvalidProgram;
    }

    fn markNodeOutputsLive(
        self: *const Program,
        node: Node,
        live_values: []bool,
        live_count: *usize,
        live_bytes: *usize,
        stats: *LivenessStats,
    ) Error!void {
        for (node.outputs) |output| {
            if (output.index >= live_values.len or output.index >= self.values.len) return error.InvalidProgram;
            if (live_values[output.index]) continue;
            live_values[output.index] = true;
            live_count.* += 1;
            live_bytes.* += self.values[output.index].byte_size;
            stats.peak_live_value_count = @max(stats.peak_live_value_count, live_count.*);
            stats.peak_live_bytes = @max(stats.peak_live_bytes, live_bytes.*);
        }
    }

    fn releaseDeadNodeInputs(
        self: *const Program,
        node: Node,
        node_index: usize,
        live_values: []bool,
        live_count: *usize,
        live_bytes: *usize,
        stats: *LivenessStats,
    ) Error!void {
        for (node.inputs) |input| {
            if (input.index >= self.values.len or input.index >= live_values.len) return error.InvalidProgram;
            const value = self.values[input.index];
            if (value.last_use_node != @as(?usize, node_index)) continue;
            try self.releasePlannedValue(input.index, live_values, live_count, live_bytes, stats);
        }
    }

    fn releaseDeadFusionNodeInputs(
        self: *const Program,
        node: Node,
        group_last_node: usize,
        live_values: []bool,
        live_count: *usize,
        live_bytes: *usize,
        stats: *LivenessStats,
    ) Error!void {
        for (node.inputs) |input| {
            if (input.index >= self.values.len or input.index >= live_values.len) return error.InvalidProgram;
            const value = self.values[input.index];
            const last_use = value.last_use_node orelse continue;
            if (last_use > group_last_node) continue;
            try self.releasePlannedValue(input.index, live_values, live_count, live_bytes, stats);
        }
    }

    fn releasePlannedValue(
        self: *const Program,
        value_index: usize,
        live_values: []bool,
        live_count: *usize,
        live_bytes: *usize,
        stats: *LivenessStats,
    ) Error!void {
        const value = self.values[value_index];
        if (value.is_output) return;
        const producer_node = value.producer_node orelse return;
        if (producer_node >= self.nodes.len) return error.InvalidProgram;
        if (self.nodes[producer_node].kind == .constant) return;
        if (!live_values[value_index]) return;
        live_values[value_index] = false;
        live_count.* -= 1;
        live_bytes.* -= value.byte_size;
        stats.planned_release_count += 1;
        stats.planned_release_bytes += value.byte_size;
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
