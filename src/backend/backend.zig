const std = @import("std");
const core = @import("src/core");

pub const BufferHandle = *anyopaque;
pub const ExecutableHandle = *anyopaque;
pub const ExecutionEventHandle = *anyopaque;
pub const AsyncHostToDeviceTransferHandle = *anyopaque;

pub const ReduceWindowMaxWithIndicesResult = struct {
    values: BufferHandle,
    indices: BufferHandle,
};

pub const ReduceMaxWithIndicesResult = struct {
    values: BufferHandle,
    indices: BufferHandle,
};

pub const RngBitGeneratorResult = struct {
    state: BufferHandle,
    bits: BufferHandle,
};

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

pub const CustomCallKind = enum {
    identity,
    unary,
    binary,
};

pub const CustomCallRegistration = struct {
    target: []const u8,
    kind: CustomCallKind,
    unary_op: ?core.ElementwiseUnaryOp = null,
    binary_op: ?core.ElementwiseBinaryOp = null,
};

pub const ExecutableOutput = struct {
    handle: BufferHandle,
    element_type: core.BufferType,
    dims: []const i64,
    byte_size: usize,
};

pub const ExecutionCompletionKind = enum {
    completed,
    pending,
};

pub const ExecutionCompletion = struct {
    kind: ExecutionCompletionKind = .completed,
    backend_event: ?ExecutionEventHandle = null,

    pub fn completed() ExecutionCompletion {
        return .{ .kind = .completed };
    }

    pub fn pending(event: ExecutionEventHandle) ExecutionCompletion {
        return .{ .kind = .pending, .backend_event = event };
    }
};

pub const ExecutionEventState = enum {
    pending,
    ready,
    failed,
};

pub const ExecutionEventStatus = struct {
    state: ExecutionEventState,
    message: []const u8 = "",
};

pub const ExecutionResult = struct {
    outputs: []ExecutableOutput,
    completion: ExecutionCompletion = .{},
};

pub const ExecutableStats = struct {
    resident_constant_count: usize = 0,
    resident_constant_bytes: usize = 0,
    program_value_count: usize = 0,
    program_node_count: usize = 0,
    program_edge_count: usize = 0,
    program_schedule_item_count: usize = 0,
    program_subprogram_count: usize = 0,
    program_control_flow_count: usize = 0,
    program_fusion_group_count: usize = 0,
    program_materialization_boundary_count: usize = 0,
    program_planned_release_count: usize = 0,
    program_planned_release_bytes: usize = 0,
    program_peak_live_value_count: usize = 0,
    program_peak_live_bytes: usize = 0,
    program_device_count: usize = 0,
    last_execute_device_index: usize = std.math.maxInt(usize),
    last_execute_local_hardware_id: i32 = -1,
    execute_count: usize = 0,
    compiled_program_execute_count: usize = 0,
    compiled_program_output_count: usize = 0,
    captured_program_execute_count: usize = 0,
    captured_program_dynamic_input_count: usize = 0,
    captured_program_captured_input_count: usize = 0,
    donation_alias_output_count: usize = 0,
    donation_alias_output_bytes: usize = 0,
    fusion_group_execute_count: usize = 0,
    materialization_eval_count: usize = 0,
    materialization_eval_buffer_count: usize = 0,
    released_intermediate_count: usize = 0,
    borrowed_constant_nodes: usize = 0,
    execute_wall_us_total: u64 = 0,
    execute_wall_us_peak: u64 = 0,
    schedule_us_total: u64 = 0,
    schedule_us_peak: u64 = 0,
    node_us_total: u64 = 0,
    node_us_peak: u64 = 0,
    fusion_group_us_total: u64 = 0,
    fusion_group_us_peak: u64 = 0,
    materialization_eval_us_total: u64 = 0,
    materialization_eval_us_peak: u64 = 0,
    output_clone_us_total: u64 = 0,
    output_clone_us_peak: u64 = 0,
    compiled_program_us_total: u64 = 0,
    compiled_program_us_peak: u64 = 0,
};

pub const ProgramNodeKind = enum {
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

pub const ProgramNode = struct {
    instruction_index: usize,
    kind: ProgramNodeKind,
    inputs: []const core.ValueId,
    outputs: []const core.ValueId,
    subprograms: []const usize = &.{},
    control_flow: ?usize = null,
    materializes: bool = true,
    fusion_group: ?usize = null,
};

pub const FusionGroupKind = enum {
    view_elementwise,
};

pub const FusionGroup = struct {
    id: usize,
    kind: FusionGroupKind,
    first_node: usize,
    last_node: usize,
    node_count: usize,
    node_indices: []const usize = &.{},
    input_values: []const core.ValueId = &.{},
    output_values: []const core.ValueId = &.{},
};

pub const MaterializationReason = enum {
    pjrt_output,
    side_effect,
    token,
    donation_alias,
    debug,
    backend_requirement,
};

pub const MaterializationBoundary = struct {
    value_id: core.ValueId,
    reason: MaterializationReason,
};

pub const ProgramValue = struct {
    value_id: core.ValueId,
    byte_size: usize = 0,
    producer_node: ?usize = null,
    last_use_node: ?usize = null,
    is_output: bool = false,
    materialization_boundary: ?usize = null,
};

pub const ProgramEdge = struct {
    value_id: core.ValueId,
    from_node: usize,
    to_node: usize,
};

pub const ProgramScheduleKind = enum {
    node,
    fusion_group,
    materialization_boundary,
};

pub const ProgramScheduleItem = struct {
    kind: ProgramScheduleKind,
    index: usize,
    count: usize = 1,
};

pub const ProgramSubprogram = struct {
    id: usize,
    parent_node: usize,
    region_id: core.RegionId,
    kind: core.RegionKind,
    values: []const core.RegionValue = &.{},
    argument_descriptors: []const core.BufferDescriptor = &.{},
    instructions: []const core.RegionInstruction = &.{},
    return_descriptors: []const core.BufferDescriptor = &.{},
    terminator_operands: []const core.RegionValueId = &.{},
    terminator_operand_descriptors: []const core.BufferDescriptor = &.{},
};

pub const ProgramControlFlowKind = enum {
    while_loop,
};

pub const ProgramControlFlow = struct {
    id: usize,
    parent_node: usize,
    kind: ProgramControlFlowKind,
    condition_subprogram: usize,
    body_subprogram: usize,
    state_inputs: []const core.ValueId = &.{},
    state_outputs: []const core.ValueId = &.{},
    predicate_output: core.RegionValueId = core.RegionValueId.invalid,
};

pub const ProgramLivenessStats = struct {
    planned_release_count: usize = 0,
    planned_release_bytes: usize = 0,
    peak_live_value_count: usize = 0,
    peak_live_bytes: usize = 0,
};

pub const Program = struct {
    allocator: std.mem.Allocator,
    values: []ProgramValue,
    nodes: []ProgramNode,
    edges: []ProgramEdge,
    schedule: []ProgramScheduleItem,
    subprograms: []ProgramSubprogram = &.{},
    control_flows: []ProgramControlFlow = &.{},
    fusion_groups: []FusionGroup,
    materialization_boundaries: []MaterializationBoundary,
    fusion_group_count: usize = 0,

    pub fn validate(self: *const Program) Error!void {
        return self.validateWithWriter(null) catch |err| switch (err) {
            error.InvalidProgram => error.InvalidProgram,
            else => unreachable,
        };
    }

    pub fn livenessStats(self: *const Program) Error!ProgramLivenessStats {
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

        var stats = ProgramLivenessStats{
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
                    computed_group_inputs[groupMarkIndex(self.values.len, group_index, input.index)] = true;
                }
            }
        }
        for (self.edges) |edge| {
            const from_group = self.nodes[edge.from_node].fusion_group;
            const to_group = self.nodes[edge.to_node].fusion_group;
            if (to_group) |group_index| {
                if (from_group != group_index) {
                    computed_group_inputs[groupMarkIndex(self.values.len, group_index, edge.value_id.index)] = true;
                }
            }
            if (from_group) |group_index| {
                if (to_group != group_index) {
                    computed_group_outputs[groupMarkIndex(self.values.len, group_index, edge.value_id.index)] = true;
                }
            }
        }
        for (self.materialization_boundaries) |boundary| {
            const producer_node = self.values[boundary.value_id.index].producer_node orelse continue;
            const group_index = self.nodes[producer_node].fusion_group orelse continue;
            computed_group_outputs[groupMarkIndex(self.values.len, group_index, boundary.value_id.index)] = true;
        }
        for (self.fusion_groups, 0..) |group, group_index| {
            try validateMarkedValueSet(
                writer,
                "input_values",
                group_index,
                group.input_values,
                self.values,
                computed_group_inputs[groupMarkIndex(self.values.len, group_index, 0)..][0..self.values.len],
            );
            try validateMarkedValueSet(
                writer,
                "output_values",
                group_index,
                group.output_values,
                self.values,
                computed_group_outputs[groupMarkIndex(self.values.len, group_index, 0)..][0..self.values.len],
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

    fn hasEdge(self: *const Program, value_id: core.ValueId, from_node: usize, to_node: usize) bool {
        for (self.edges) |edge| {
            if (edge.value_id.index == value_id.index and edge.from_node == from_node and edge.to_node == to_node) return true;
        }
        return false;
    }

    fn containsValueId(values: []const core.ValueId, value_id: core.ValueId) bool {
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
        declared: []const core.ValueId,
        values: []const ProgramValue,
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
        node: ProgramNode,
        live_values: []bool,
        live_count: *usize,
        live_bytes: *usize,
        stats: *ProgramLivenessStats,
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
        node: ProgramNode,
        node_index: usize,
        live_values: []bool,
        live_count: *usize,
        live_bytes: *usize,
        stats: *ProgramLivenessStats,
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
        node: ProgramNode,
        group_last_node: usize,
        live_values: []bool,
        live_count: *usize,
        live_bytes: *usize,
        stats: *ProgramLivenessStats,
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
        stats: *ProgramLivenessStats,
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

    pub fn deinit(self: *Program) void {
        for (self.nodes) |node| {
            self.allocator.free(node.inputs);
            self.allocator.free(node.outputs);
            if (node.subprograms.len != 0) self.allocator.free(node.subprograms);
        }
        for (self.control_flows) |control_flow| {
            if (control_flow.state_inputs.len != 0) self.allocator.free(control_flow.state_inputs);
            if (control_flow.state_outputs.len != 0) self.allocator.free(control_flow.state_outputs);
        }
        for (self.subprograms) |subprogram| {
            freeRegionValueList(self.allocator, subprogram.values);
            freeDescriptorList(self.allocator, subprogram.argument_descriptors);
            freeRegionInstructionList(self.allocator, subprogram.instructions);
            freeDescriptorList(self.allocator, subprogram.return_descriptors);
            if (subprogram.terminator_operands.len != 0) self.allocator.free(subprogram.terminator_operands);
            freeDescriptorList(self.allocator, subprogram.terminator_operand_descriptors);
        }
        for (self.fusion_groups) |group| {
            if (group.node_indices.len != 0) self.allocator.free(group.node_indices);
            if (group.input_values.len != 0) self.allocator.free(group.input_values);
            if (group.output_values.len != 0) self.allocator.free(group.output_values);
        }
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

fn freeDescriptorList(allocator: std.mem.Allocator, descriptors: []const core.BufferDescriptor) void {
    for (descriptors) |descriptor| {
        if (descriptor.dims.len != 0) allocator.free(descriptor.dims);
    }
    if (descriptors.len != 0) allocator.free(descriptors);
}

fn freeRegionValueList(allocator: std.mem.Allocator, values: []const core.RegionValue) void {
    for (values) |value| {
        if (value.descriptor.dims.len != 0) allocator.free(value.descriptor.dims);
        if (value.literal) |literal| allocator.free(literal);
    }
    if (values.len != 0) allocator.free(values);
}

fn freeRegionInstructionList(allocator: std.mem.Allocator, instructions: []const core.RegionInstruction) void {
    for (instructions) |instruction| {
        if (instruction.inputs.len != 0) allocator.free(instruction.inputs);
        if (instruction.outputs.len != 0) allocator.free(instruction.outputs);
        freeDescriptorList(allocator, instruction.operand_descriptors);
        freeDescriptorList(allocator, instruction.result_descriptors);
    }
    if (instructions.len != 0) allocator.free(instructions);
}

pub const Capabilities = struct {
    kind: core.BackendKind,
    name: []const u8,
    supports_device_buffers: bool,
    supports_unified_memory: bool,
    supports_async_execution: bool = false,
};

pub const Backend = struct {
    ptr: ?*anyopaque = null,
    vtable: *const VTable,

    pub const VTable = struct {
        kind: *const fn (backend: Backend) core.BackendKind,
        capabilities: *const fn (backend: Backend) Capabilities,
        enumerateDevices: *const fn (backend: Backend, allocator: std.mem.Allocator, device_count_hint: usize) Error![]core.DeviceDescriptor,
        releaseDeviceDescriptors: *const fn (backend: Backend, allocator: std.mem.Allocator, descriptors: []core.DeviceDescriptor) void,
        bufferFromHost: *const fn (backend: Backend, device_local_hardware_id: i32, element_type: core.BufferType, dims: []const i64, src: []const u8) Error!?BufferHandle,
        beginAsyncHostToDeviceTransfer: *const fn (backend: Backend, device_local_hardware_id: i32, element_type: core.BufferType, dims: []const i64, byte_size: usize) Error!?AsyncHostToDeviceTransferHandle,
        writeAsyncHostToDeviceTransfer: *const fn (backend: Backend, transfer: AsyncHostToDeviceTransferHandle, offset: usize, src: []const u8) Error!void,
        finishAsyncHostToDeviceTransfer: *const fn (backend: Backend, transfer: AsyncHostToDeviceTransferHandle) Error!?BufferHandle,
        destroyAsyncHostToDeviceTransfer: *const fn (backend: Backend, transfer: AsyncHostToDeviceTransferHandle) void,
        allocateBuffer: *const fn (backend: Backend, device_local_hardware_id: i32, element_type: core.BufferType, dims: []const i64) Error!?BufferHandle,
        iota: *const fn (backend: Backend, device_local_hardware_id: i32, element_type: core.BufferType, dims: []const i64, iota_dimension: i64) Error!?BufferHandle,
        partitionId: *const fn (backend: Backend, device_local_hardware_id: i32, element_type: core.BufferType, partition_id: u32) Error!?BufferHandle,
        cloneBuffer: *const fn (backend: Backend, src: BufferHandle) Error!?BufferHandle,
        complex: *const fn (backend: Backend, real: BufferHandle, imag: BufferHandle, output_dims: []const i64) Error!?BufferHandle,
        realPart: *const fn (backend: Backend, src: BufferHandle, output_dims: []const i64) Error!?BufferHandle,
        imagPart: *const fn (backend: Backend, src: BufferHandle, output_dims: []const i64) Error!?BufferHandle,
        convert: *const fn (backend: Backend, src: BufferHandle, output_type: core.BufferType) Error!?BufferHandle,
        bitcast: *const fn (backend: Backend, src: BufferHandle, output_type: core.BufferType, output_dims: []const i64) Error!?BufferHandle,
        binary: *const fn (backend: Backend, lhs: BufferHandle, rhs: BufferHandle, op: core.ElementwiseBinaryOp) Error!?BufferHandle,
        unary: *const fn (backend: Backend, src: BufferHandle, op: core.ElementwiseUnaryOp) Error!?BufferHandle,
        reshape: *const fn (backend: Backend, src: BufferHandle, dims: []const i64) Error!?BufferHandle,
        transpose: *const fn (backend: Backend, src: BufferHandle, permutation: []const i64) Error!?BufferHandle,
        broadcastInDim: *const fn (backend: Backend, src: BufferHandle, broadcast_dimensions: []const i64, output_dims: []const i64) Error!?BufferHandle,
        slice: *const fn (backend: Backend, src: BufferHandle, start_indices: []const i64, limit_indices: []const i64, strides: []const i64, output_dims: []const i64) Error!?BufferHandle,
        dynamicSlice: *const fn (backend: Backend, src: BufferHandle, start_buffers: []const BufferHandle, slice_sizes: []const i64, output_dims: []const i64) Error!?BufferHandle,
        dynamicUpdateSlice: *const fn (backend: Backend, src: BufferHandle, update: BufferHandle, start_buffers: []const BufferHandle, output_dims: []const i64) Error!?BufferHandle,
        pad: *const fn (backend: Backend, src: BufferHandle, padding_value: BufferHandle, edge_padding_low: []const i64, edge_padding_high: []const i64, interior_padding: []const i64, output_dims: []const i64) Error!?BufferHandle,
        reverse: *const fn (backend: Backend, src: BufferHandle, dimensions: []const i64, output_dims: []const i64) Error!?BufferHandle,
        concatenate: *const fn (backend: Backend, lhs: BufferHandle, rhs: BufferHandle, dimension: i64, output_dims: []const i64) Error!?BufferHandle,
        gather: *const fn (backend: Backend, operand: BufferHandle, indices: BufferHandle, start_index_map: []const i64, collapsed_slice_dims: []const i64, operand_batching_dims: []const i64, start_indices_batching_dims: []const i64, index_vector_dim: i64, slice_sizes: []const i64, offset_dims: []const i64, output_dims: []const i64) Error!?BufferHandle,
        gatherAxis: *const fn (backend: Backend, operand: BufferHandle, indices: BufferHandle, axis: i64, index_vector_dim: i64, output_dims: []const i64) Error!?BufferHandle,
        scatter: *const fn (backend: Backend, operand: BufferHandle, indices: BufferHandle, updates: BufferHandle, scatter_dims_to_operand_dims: []const i64, inserted_window_dims: []const i64, update_window_dims: []const i64, input_batching_dims: []const i64, scatter_indices_batching_dims: []const i64, index_vector_dim: i64, update_kind: core.ScatterUpdateKind, output_dims: []const i64) Error!?BufferHandle,
        scatterAxis: *const fn (backend: Backend, operand: BufferHandle, indices: BufferHandle, updates: BufferHandle, axis: i64, index_vector_dim: i64, update_kind: core.ScatterUpdateKind, output_dims: []const i64) Error!?BufferHandle,
        sort: *const fn (backend: Backend, src: BufferHandle, dimension: i64, output_dims: []const i64) Error!?BufferHandle,
        argsort: *const fn (backend: Backend, src: BufferHandle, dimension: i64, output_type: core.BufferType, output_dims: []const i64) Error!?BufferHandle,
        takeAlongAxis: *const fn (backend: Backend, src: BufferHandle, indices: BufferHandle, dimension: i64, output_dims: []const i64) Error!?BufferHandle,
        dotGeneral: *const fn (backend: Backend, lhs: BufferHandle, rhs: BufferHandle, lhs_batch_dimensions: []const i64, rhs_batch_dimensions: []const i64, lhs_contracting_dimensions: []const i64, rhs_contracting_dimensions: []const i64, output_dims: []const i64) Error!?BufferHandle,
        convolution: *const fn (backend: Backend, lhs: BufferHandle, rhs: BufferHandle, window_strides: []const i64, padding_low: []const i64, padding_high: []const i64, lhs_dilation: []const i64, rhs_dilation: []const i64, window_reversal: []const bool, feature_group_count: i64, output_dims: []const i64) Error!?BufferHandle,
        cholesky: *const fn (backend: Backend, src: BufferHandle, lower: bool, output_dims: []const i64) Error!?BufferHandle,
        triangularSolve: *const fn (backend: Backend, a: BufferHandle, b: BufferHandle, left_side: bool, lower: bool, unit_diagonal: bool, transpose_a: core.TriangularSolveTranspose, output_dims: []const i64) Error!?BufferHandle,
        fft: *const fn (backend: Backend, src: BufferHandle, fft_kind: core.FftKind, fft_lengths: []const i64, output_dims: []const i64) Error!?BufferHandle,
        rng: *const fn (backend: Backend, a: BufferHandle, b: BufferHandle, distribution: core.RngDistribution, output_type: core.BufferType, output_dims: []const i64) Error!?BufferHandle,
        rngBitGenerator: *const fn (backend: Backend, state: BufferHandle, output_type: core.BufferType, output_dims: []const i64) Error!?RngBitGeneratorResult,
        reduce: *const fn (backend: Backend, src: BufferHandle, op: core.PlanInstructionKind, dimensions: []const i64, output_dims: []const i64) Error!?BufferHandle,
        reduceMaxWithIndices: *const fn (backend: Backend, values: BufferHandle, indices: BufferHandle, dimensions: []const i64, output_dims: []const i64) Error!?ReduceMaxWithIndicesResult,
        reduceWindow: *const fn (backend: Backend, src: BufferHandle, op: core.PlanInstructionKind, window_dimensions: []const i64, window_strides: []const i64, base_dilations: []const i64, window_dilations: []const i64, padding_low: []const i64, padding_high: []const i64, output_dims: []const i64) Error!?BufferHandle,
        reduceWindowMaxWithIndices: *const fn (backend: Backend, values: BufferHandle, indices: BufferHandle, window_dimensions: []const i64, window_strides: []const i64, base_dilations: []const i64, window_dilations: []const i64, padding_low: []const i64, padding_high: []const i64, output_dims: []const i64) Error!?ReduceWindowMaxWithIndicesResult,
        compare: *const fn (backend: Backend, lhs: BufferHandle, rhs: BufferHandle, direction: core.CompareOp, output_dims: []const i64) Error!?BufferHandle,
        select: *const fn (backend: Backend, pred: BufferHandle, on_true: BufferHandle, on_false: BufferHandle, output_dims: []const i64) Error!?BufferHandle,
        clamp: *const fn (backend: Backend, min: BufferHandle, value: BufferHandle, max: BufferHandle, output_dims: []const i64) Error!?BufferHandle,
        compileExecutable: *const fn (backend: Backend, allocator: std.mem.Allocator, plan: *const core.ExecutablePlan, device_local_hardware_ids: []const i32) Error!?ExecutableHandle,
        writeExecutableLoweringDiagnostic: *const fn (backend: Backend, plan: *const core.ExecutablePlan, device_local_hardware_ids: []const i32, writer: *std.Io.Writer) std.Io.Writer.Error!void,
        executeExecutable: *const fn (backend: Backend, allocator: std.mem.Allocator, executable: ExecutableHandle, device_index: usize, arguments: []const BufferHandle) Error!?ExecutionResult,
        executionEventStatus: *const fn (backend: Backend, event: ExecutionEventHandle) Error!ExecutionEventStatus,
        destroyExecutionEvent: *const fn (backend: Backend, event: ExecutionEventHandle) void,
        executableStats: *const fn (backend: Backend, executable: ExecutableHandle) ExecutableStats,
        destroyExecutable: *const fn (backend: Backend, executable: ExecutableHandle) void,
        registerCustomCall: *const fn (backend: Backend, registration: CustomCallRegistration) Error!void,
        unregisterCustomCall: *const fn (backend: Backend, target: []const u8) void,
        customCallRegistryVersion: *const fn (backend: Backend) u64,
        copyToHost: *const fn (backend: Backend, src: BufferHandle, dst: []u8) Error!void,
        destroyBuffer: *const fn (backend: Backend, buffer: BufferHandle) void,
    };

    pub fn kind(self: Backend) core.BackendKind {
        return self.vtable.kind(self);
    }

    pub fn capabilities(self: Backend) Capabilities {
        return self.vtable.capabilities(self);
    }

    pub fn enumerateDevices(self: Backend, allocator: std.mem.Allocator, device_count_hint: usize) Error![]core.DeviceDescriptor {
        return self.vtable.enumerateDevices(self, allocator, device_count_hint);
    }

    pub fn releaseDeviceDescriptors(self: Backend, allocator: std.mem.Allocator, descriptors: []core.DeviceDescriptor) void {
        self.vtable.releaseDeviceDescriptors(self, allocator, descriptors);
    }

    pub fn bufferFromHost(self: Backend, device_local_hardware_id: i32, element_type: core.BufferType, dims: []const i64, src: []const u8) Error!?BufferHandle {
        return self.vtable.bufferFromHost(self, device_local_hardware_id, element_type, dims, src);
    }

    pub fn beginAsyncHostToDeviceTransfer(self: Backend, device_local_hardware_id: i32, element_type: core.BufferType, dims: []const i64, byte_size: usize) Error!?AsyncHostToDeviceTransferHandle {
        return self.vtable.beginAsyncHostToDeviceTransfer(self, device_local_hardware_id, element_type, dims, byte_size);
    }

    pub fn writeAsyncHostToDeviceTransfer(self: Backend, transfer: AsyncHostToDeviceTransferHandle, offset: usize, src: []const u8) Error!void {
        return self.vtable.writeAsyncHostToDeviceTransfer(self, transfer, offset, src);
    }

    pub fn finishAsyncHostToDeviceTransfer(self: Backend, transfer: AsyncHostToDeviceTransferHandle) Error!?BufferHandle {
        return self.vtable.finishAsyncHostToDeviceTransfer(self, transfer);
    }

    pub fn destroyAsyncHostToDeviceTransfer(self: Backend, transfer: AsyncHostToDeviceTransferHandle) void {
        self.vtable.destroyAsyncHostToDeviceTransfer(self, transfer);
    }

    pub fn allocateBuffer(self: Backend, device_local_hardware_id: i32, element_type: core.BufferType, dims: []const i64) Error!?BufferHandle {
        return self.vtable.allocateBuffer(self, device_local_hardware_id, element_type, dims);
    }

    pub fn iota(self: Backend, device_local_hardware_id: i32, element_type: core.BufferType, dims: []const i64, iota_dimension: i64) Error!?BufferHandle {
        return self.vtable.iota(self, device_local_hardware_id, element_type, dims, iota_dimension);
    }

    pub fn partitionId(self: Backend, device_local_hardware_id: i32, element_type: core.BufferType, partition_id: u32) Error!?BufferHandle {
        return self.vtable.partitionId(self, device_local_hardware_id, element_type, partition_id);
    }

    pub fn cloneBuffer(self: Backend, src: BufferHandle) Error!?BufferHandle {
        return self.vtable.cloneBuffer(self, src);
    }

    pub fn complex(self: Backend, real: BufferHandle, imag: BufferHandle, output_dims: []const i64) Error!?BufferHandle {
        return self.vtable.complex(self, real, imag, output_dims);
    }

    pub fn realPart(self: Backend, src: BufferHandle, output_dims: []const i64) Error!?BufferHandle {
        return self.vtable.realPart(self, src, output_dims);
    }

    pub fn imagPart(self: Backend, src: BufferHandle, output_dims: []const i64) Error!?BufferHandle {
        return self.vtable.imagPart(self, src, output_dims);
    }

    pub fn convert(self: Backend, src: BufferHandle, output_type: core.BufferType) Error!?BufferHandle {
        return self.vtable.convert(self, src, output_type);
    }

    pub fn bitcast(self: Backend, src: BufferHandle, output_type: core.BufferType, output_dims: []const i64) Error!?BufferHandle {
        return self.vtable.bitcast(self, src, output_type, output_dims);
    }

    pub fn binary(self: Backend, lhs: BufferHandle, rhs: BufferHandle, op: core.ElementwiseBinaryOp) Error!?BufferHandle {
        return self.vtable.binary(self, lhs, rhs, op);
    }

    pub fn unary(self: Backend, src: BufferHandle, op: core.ElementwiseUnaryOp) Error!?BufferHandle {
        return self.vtable.unary(self, src, op);
    }

    pub fn reshape(self: Backend, src: BufferHandle, dims: []const i64) Error!?BufferHandle {
        return self.vtable.reshape(self, src, dims);
    }

    pub fn transpose(self: Backend, src: BufferHandle, permutation: []const i64) Error!?BufferHandle {
        return self.vtable.transpose(self, src, permutation);
    }

    pub fn broadcastInDim(self: Backend, src: BufferHandle, broadcast_dimensions: []const i64, output_dims: []const i64) Error!?BufferHandle {
        return self.vtable.broadcastInDim(self, src, broadcast_dimensions, output_dims);
    }

    pub fn slice(self: Backend, src: BufferHandle, start_indices: []const i64, limit_indices: []const i64, strides: []const i64, output_dims: []const i64) Error!?BufferHandle {
        return self.vtable.slice(self, src, start_indices, limit_indices, strides, output_dims);
    }

    pub fn dynamicSlice(self: Backend, src: BufferHandle, start_buffers: []const BufferHandle, slice_sizes: []const i64, output_dims: []const i64) Error!?BufferHandle {
        return self.vtable.dynamicSlice(self, src, start_buffers, slice_sizes, output_dims);
    }

    pub fn dynamicUpdateSlice(self: Backend, src: BufferHandle, update: BufferHandle, start_buffers: []const BufferHandle, output_dims: []const i64) Error!?BufferHandle {
        return self.vtable.dynamicUpdateSlice(self, src, update, start_buffers, output_dims);
    }

    pub fn pad(self: Backend, src: BufferHandle, padding_value: BufferHandle, edge_padding_low: []const i64, edge_padding_high: []const i64, interior_padding: []const i64, output_dims: []const i64) Error!?BufferHandle {
        return self.vtable.pad(self, src, padding_value, edge_padding_low, edge_padding_high, interior_padding, output_dims);
    }

    pub fn reverse(self: Backend, src: BufferHandle, dimensions: []const i64, output_dims: []const i64) Error!?BufferHandle {
        return self.vtable.reverse(self, src, dimensions, output_dims);
    }

    pub fn concatenate(self: Backend, lhs: BufferHandle, rhs: BufferHandle, dimension: i64, output_dims: []const i64) Error!?BufferHandle {
        return self.vtable.concatenate(self, lhs, rhs, dimension, output_dims);
    }

    pub fn gather(self: Backend, operand: BufferHandle, indices: BufferHandle, start_index_map: []const i64, collapsed_slice_dims: []const i64, operand_batching_dims: []const i64, start_indices_batching_dims: []const i64, index_vector_dim: i64, slice_sizes: []const i64, offset_dims: []const i64, output_dims: []const i64) Error!?BufferHandle {
        return self.vtable.gather(self, operand, indices, start_index_map, collapsed_slice_dims, operand_batching_dims, start_indices_batching_dims, index_vector_dim, slice_sizes, offset_dims, output_dims);
    }

    pub fn gatherAxis(self: Backend, operand: BufferHandle, indices: BufferHandle, axis: i64, index_vector_dim: i64, output_dims: []const i64) Error!?BufferHandle {
        return self.vtable.gatherAxis(self, operand, indices, axis, index_vector_dim, output_dims);
    }

    pub fn scatter(self: Backend, operand: BufferHandle, indices: BufferHandle, updates: BufferHandle, scatter_dims_to_operand_dims: []const i64, inserted_window_dims: []const i64, update_window_dims: []const i64, input_batching_dims: []const i64, scatter_indices_batching_dims: []const i64, index_vector_dim: i64, update_kind: core.ScatterUpdateKind, output_dims: []const i64) Error!?BufferHandle {
        return self.vtable.scatter(self, operand, indices, updates, scatter_dims_to_operand_dims, inserted_window_dims, update_window_dims, input_batching_dims, scatter_indices_batching_dims, index_vector_dim, update_kind, output_dims);
    }

    pub fn scatterAxis(self: Backend, operand: BufferHandle, indices: BufferHandle, updates: BufferHandle, axis: i64, index_vector_dim: i64, update_kind: core.ScatterUpdateKind, output_dims: []const i64) Error!?BufferHandle {
        return self.vtable.scatterAxis(self, operand, indices, updates, axis, index_vector_dim, update_kind, output_dims);
    }

    pub fn sort(self: Backend, src: BufferHandle, dimension: i64, output_dims: []const i64) Error!?BufferHandle {
        return self.vtable.sort(self, src, dimension, output_dims);
    }

    pub fn argsort(self: Backend, src: BufferHandle, dimension: i64, output_type: core.BufferType, output_dims: []const i64) Error!?BufferHandle {
        return self.vtable.argsort(self, src, dimension, output_type, output_dims);
    }

    pub fn takeAlongAxis(self: Backend, src: BufferHandle, indices: BufferHandle, dimension: i64, output_dims: []const i64) Error!?BufferHandle {
        return self.vtable.takeAlongAxis(self, src, indices, dimension, output_dims);
    }

    pub fn dotGeneral(self: Backend, lhs: BufferHandle, rhs: BufferHandle, lhs_batch_dimensions: []const i64, rhs_batch_dimensions: []const i64, lhs_contracting_dimensions: []const i64, rhs_contracting_dimensions: []const i64, output_dims: []const i64) Error!?BufferHandle {
        return self.vtable.dotGeneral(self, lhs, rhs, lhs_batch_dimensions, rhs_batch_dimensions, lhs_contracting_dimensions, rhs_contracting_dimensions, output_dims);
    }

    pub fn convolution(self: Backend, lhs: BufferHandle, rhs: BufferHandle, window_strides: []const i64, padding_low: []const i64, padding_high: []const i64, lhs_dilation: []const i64, rhs_dilation: []const i64, window_reversal: []const bool, feature_group_count: i64, output_dims: []const i64) Error!?BufferHandle {
        return self.vtable.convolution(self, lhs, rhs, window_strides, padding_low, padding_high, lhs_dilation, rhs_dilation, window_reversal, feature_group_count, output_dims);
    }

    pub fn cholesky(self: Backend, src: BufferHandle, lower: bool, output_dims: []const i64) Error!?BufferHandle {
        return self.vtable.cholesky(self, src, lower, output_dims);
    }

    pub fn triangularSolve(self: Backend, a: BufferHandle, b: BufferHandle, left_side: bool, lower: bool, unit_diagonal: bool, transpose_a: core.TriangularSolveTranspose, output_dims: []const i64) Error!?BufferHandle {
        return self.vtable.triangularSolve(self, a, b, left_side, lower, unit_diagonal, transpose_a, output_dims);
    }

    pub fn fft(self: Backend, src: BufferHandle, fft_kind: core.FftKind, fft_lengths: []const i64, output_dims: []const i64) Error!?BufferHandle {
        return self.vtable.fft(self, src, fft_kind, fft_lengths, output_dims);
    }

    pub fn rngBitGenerator(self: Backend, state: BufferHandle, output_type: core.BufferType, output_dims: []const i64) Error!?RngBitGeneratorResult {
        return self.vtable.rngBitGenerator(self, state, output_type, output_dims);
    }

    pub fn rng(self: Backend, a: BufferHandle, b: BufferHandle, distribution: core.RngDistribution, output_type: core.BufferType, output_dims: []const i64) Error!?BufferHandle {
        return self.vtable.rng(self, a, b, distribution, output_type, output_dims);
    }

    pub fn reduce(self: Backend, src: BufferHandle, op: core.PlanInstructionKind, dimensions: []const i64, output_dims: []const i64) Error!?BufferHandle {
        return self.vtable.reduce(self, src, op, dimensions, output_dims);
    }

    pub fn reduceMaxWithIndices(self: Backend, values: BufferHandle, indices: BufferHandle, dimensions: []const i64, output_dims: []const i64) Error!?ReduceMaxWithIndicesResult {
        return self.vtable.reduceMaxWithIndices(self, values, indices, dimensions, output_dims);
    }

    pub fn reduceWindow(self: Backend, src: BufferHandle, op: core.PlanInstructionKind, window_dimensions: []const i64, window_strides: []const i64, base_dilations: []const i64, window_dilations: []const i64, padding_low: []const i64, padding_high: []const i64, output_dims: []const i64) Error!?BufferHandle {
        return self.vtable.reduceWindow(self, src, op, window_dimensions, window_strides, base_dilations, window_dilations, padding_low, padding_high, output_dims);
    }

    pub fn reduceWindowMaxWithIndices(self: Backend, values: BufferHandle, indices: BufferHandle, window_dimensions: []const i64, window_strides: []const i64, base_dilations: []const i64, window_dilations: []const i64, padding_low: []const i64, padding_high: []const i64, output_dims: []const i64) Error!?ReduceWindowMaxWithIndicesResult {
        return self.vtable.reduceWindowMaxWithIndices(self, values, indices, window_dimensions, window_strides, base_dilations, window_dilations, padding_low, padding_high, output_dims);
    }

    pub fn compare(self: Backend, lhs: BufferHandle, rhs: BufferHandle, direction: core.CompareOp, output_dims: []const i64) Error!?BufferHandle {
        return self.vtable.compare(self, lhs, rhs, direction, output_dims);
    }

    pub fn select(self: Backend, pred: BufferHandle, on_true: BufferHandle, on_false: BufferHandle, output_dims: []const i64) Error!?BufferHandle {
        return self.vtable.select(self, pred, on_true, on_false, output_dims);
    }

    pub fn clamp(self: Backend, min: BufferHandle, value: BufferHandle, max: BufferHandle, output_dims: []const i64) Error!?BufferHandle {
        return self.vtable.clamp(self, min, value, max, output_dims);
    }

    pub fn compileExecutable(self: Backend, allocator: std.mem.Allocator, plan: *const core.ExecutablePlan, device_local_hardware_ids: []const i32) Error!?ExecutableHandle {
        return self.vtable.compileExecutable(self, allocator, plan, device_local_hardware_ids);
    }

    pub fn writeExecutableLoweringDiagnostic(self: Backend, plan: *const core.ExecutablePlan, device_local_hardware_ids: []const i32, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.vtable.writeExecutableLoweringDiagnostic(self, plan, device_local_hardware_ids, writer);
    }

    pub fn executeExecutable(self: Backend, allocator: std.mem.Allocator, executable: ExecutableHandle, device_index: usize, arguments: []const BufferHandle) Error!?ExecutionResult {
        return self.vtable.executeExecutable(self, allocator, executable, device_index, arguments);
    }

    pub fn executionEventStatus(self: Backend, event: ExecutionEventHandle) Error!ExecutionEventStatus {
        return self.vtable.executionEventStatus(self, event);
    }

    pub fn destroyExecutionEvent(self: Backend, event: ExecutionEventHandle) void {
        self.vtable.destroyExecutionEvent(self, event);
    }

    pub fn executableStats(self: Backend, executable: ExecutableHandle) ExecutableStats {
        return self.vtable.executableStats(self, executable);
    }

    pub fn destroyExecutable(self: Backend, executable: ExecutableHandle) void {
        self.vtable.destroyExecutable(self, executable);
    }

    pub fn registerCustomCall(self: Backend, registration: CustomCallRegistration) Error!void {
        return self.vtable.registerCustomCall(self, registration);
    }

    pub fn unregisterCustomCall(self: Backend, target: []const u8) void {
        self.vtable.unregisterCustomCall(self, target);
    }

    pub fn customCallRegistryVersion(self: Backend) u64 {
        return self.vtable.customCallRegistryVersion(self);
    }

    pub fn copyToHost(self: Backend, src: BufferHandle, dst: []u8) Error!void {
        return self.vtable.copyToHost(self, src, dst);
    }

    pub fn destroyBuffer(self: Backend, buffer: BufferHandle) void {
        self.vtable.destroyBuffer(self, buffer);
    }
};

test "backend capability model is backend-neutral" {
    const caps: Capabilities = .{
        .kind = .metal_mlx,
        .name = "test",
        .supports_device_buffers = false,
        .supports_unified_memory = true,
    };
    try std.testing.expectEqual(core.BackendKind.metal_mlx, caps.kind);
    try std.testing.expect(!caps.supports_device_buffers);
    try std.testing.expect(!caps.supports_async_execution);
}

test "backend program validator accepts minimal scheduled program" {
    var values = [_]ProgramValue{
        .{
            .value_id = .{ .index = 0 },
            .last_use_node = 0,
        },
        .{
            .value_id = .{ .index = 1 },
            .producer_node = 0,
            .is_output = true,
            .materialization_boundary = 0,
        },
    };
    var node_inputs = [_]core.ValueId{.{ .index = 0 }};
    var node_outputs = [_]core.ValueId{.{ .index = 1 }};
    var nodes = [_]ProgramNode{.{
        .instruction_index = 0,
        .kind = .elementwise,
        .inputs = node_inputs[0..],
        .outputs = node_outputs[0..],
    }};
    var boundaries = [_]MaterializationBoundary{.{
        .value_id = .{ .index = 1 },
        .reason = .pjrt_output,
    }};
    var schedule = [_]ProgramScheduleItem{
        .{
            .kind = .node,
            .index = 0,
        },
        .{
            .kind = .materialization_boundary,
            .index = 0,
            .count = 1,
        },
    };
    const program = Program{
        .allocator = std.testing.allocator,
        .values = values[0..],
        .nodes = nodes[0..],
        .edges = &.{},
        .schedule = schedule[0..],
        .fusion_groups = &.{},
        .materialization_boundaries = boundaries[0..],
    };

    try program.validate();
}

test "backend program validator rejects invalid schedule references" {
    var values = [_]ProgramValue{.{
        .value_id = .{ .index = 0 },
        .producer_node = 0,
        .is_output = true,
        .materialization_boundary = 0,
    }};
    var node_outputs = [_]core.ValueId{.{ .index = 0 }};
    var nodes = [_]ProgramNode{.{
        .instruction_index = 0,
        .kind = .elementwise,
        .inputs = &.{},
        .outputs = node_outputs[0..],
    }};
    var boundaries = [_]MaterializationBoundary{.{
        .value_id = .{ .index = 0 },
        .reason = .pjrt_output,
    }};
    var schedule = [_]ProgramScheduleItem{.{
        .kind = .node,
        .index = 1,
    }};
    const program = Program{
        .allocator = std.testing.allocator,
        .values = values[0..],
        .nodes = nodes[0..],
        .edges = &.{},
        .schedule = schedule[0..],
        .fusion_groups = &.{},
        .materialization_boundaries = boundaries[0..],
    };

    var diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer diagnostics.deinit();
    try std.testing.expectError(error.InvalidProgram, program.validateWithWriter(&diagnostics.writer));
    const message = diagnostics.writer.buffered();
    try std.testing.expect(std.mem.indexOf(u8, message, "pass=backend-program-verify") != null);
    try std.testing.expect(std.mem.indexOf(u8, message, "feature=backend-program") != null);
    try std.testing.expect(std.mem.indexOf(u8, message, "invalid node index=1") != null);
}

test "backend program validator rejects unscheduled nodes" {
    var values = [_]ProgramValue{.{
        .value_id = .{ .index = 0 },
        .producer_node = 0,
        .is_output = true,
        .materialization_boundary = 0,
    }};
    var node_outputs = [_]core.ValueId{.{ .index = 0 }};
    var nodes = [_]ProgramNode{.{
        .instruction_index = 0,
        .kind = .elementwise,
        .inputs = &.{},
        .outputs = node_outputs[0..],
    }};
    var boundaries = [_]MaterializationBoundary{.{
        .value_id = .{ .index = 0 },
        .reason = .pjrt_output,
    }};
    var schedule = [_]ProgramScheduleItem{.{
        .kind = .materialization_boundary,
        .index = 0,
        .count = 1,
    }};
    const program = Program{
        .allocator = std.testing.allocator,
        .values = values[0..],
        .nodes = nodes[0..],
        .edges = &.{},
        .schedule = schedule[0..],
        .fusion_groups = &.{},
        .materialization_boundaries = boundaries[0..],
    };

    var diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer diagnostics.deinit();
    try std.testing.expectError(error.InvalidProgram, program.validateWithWriter(&diagnostics.writer));
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "node 0 is not covered by schedule") != null);
}

test "backend program validator rejects unscheduled materialization boundaries" {
    var values = [_]ProgramValue{.{
        .value_id = .{ .index = 0 },
        .producer_node = 0,
        .is_output = true,
        .materialization_boundary = 0,
    }};
    var node_outputs = [_]core.ValueId{.{ .index = 0 }};
    var nodes = [_]ProgramNode{.{
        .instruction_index = 0,
        .kind = .elementwise,
        .inputs = &.{},
        .outputs = node_outputs[0..],
    }};
    var boundaries = [_]MaterializationBoundary{.{
        .value_id = .{ .index = 0 },
        .reason = .pjrt_output,
    }};
    var schedule = [_]ProgramScheduleItem{.{
        .kind = .node,
        .index = 0,
    }};
    const program = Program{
        .allocator = std.testing.allocator,
        .values = values[0..],
        .nodes = nodes[0..],
        .edges = &.{},
        .schedule = schedule[0..],
        .fusion_groups = &.{},
        .materialization_boundaries = boundaries[0..],
    };

    var diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer diagnostics.deinit();
    try std.testing.expectError(error.InvalidProgram, program.validateWithWriter(&diagnostics.writer));
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "materialization boundary 0 is not covered by schedule") != null);
}

test "backend program validator rejects duplicate materialization boundaries" {
    var values = [_]ProgramValue{.{
        .value_id = .{ .index = 0 },
        .producer_node = 0,
        .is_output = true,
        .materialization_boundary = 0,
    }};
    var node_outputs = [_]core.ValueId{.{ .index = 0 }};
    var nodes = [_]ProgramNode{.{
        .instruction_index = 0,
        .kind = .elementwise,
        .inputs = &.{},
        .outputs = node_outputs[0..],
    }};
    var boundaries = [_]MaterializationBoundary{.{
        .value_id = .{ .index = 0 },
        .reason = .pjrt_output,
    }};
    var schedule = [_]ProgramScheduleItem{
        .{ .kind = .node, .index = 0 },
        .{ .kind = .materialization_boundary, .index = 0, .count = 1 },
        .{ .kind = .materialization_boundary, .index = 0, .count = 1 },
    };
    const program = Program{
        .allocator = std.testing.allocator,
        .values = values[0..],
        .nodes = nodes[0..],
        .edges = &.{},
        .schedule = schedule[0..],
        .fusion_groups = &.{},
        .materialization_boundaries = boundaries[0..],
    };

    var diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer diagnostics.deinit();
    try std.testing.expectError(error.InvalidProgram, program.validateWithWriter(&diagnostics.writer));
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "materialization boundary 0 more than once") != null);
}

test "backend program validator rejects materialization before producer" {
    var values = [_]ProgramValue{.{
        .value_id = .{ .index = 0 },
        .producer_node = 0,
        .is_output = true,
        .materialization_boundary = 0,
    }};
    var node_outputs = [_]core.ValueId{.{ .index = 0 }};
    var nodes = [_]ProgramNode{.{
        .instruction_index = 0,
        .kind = .elementwise,
        .inputs = &.{},
        .outputs = node_outputs[0..],
    }};
    var boundaries = [_]MaterializationBoundary{.{
        .value_id = .{ .index = 0 },
        .reason = .pjrt_output,
    }};
    var schedule = [_]ProgramScheduleItem{
        .{ .kind = .materialization_boundary, .index = 0, .count = 1 },
        .{ .kind = .node, .index = 0 },
    };
    const program = Program{
        .allocator = std.testing.allocator,
        .values = values[0..],
        .nodes = nodes[0..],
        .edges = &.{},
        .schedule = schedule[0..],
        .fusion_groups = &.{},
        .materialization_boundaries = boundaries[0..],
    };

    var diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer diagnostics.deinit();
    try std.testing.expectError(error.InvalidProgram, program.validateWithWriter(&diagnostics.writer));
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "scheduled before producer node") != null);
}

test "backend program validator rejects duplicate scheduled nodes" {
    var values = [_]ProgramValue{.{
        .value_id = .{ .index = 0 },
        .producer_node = 0,
        .is_output = true,
        .materialization_boundary = 0,
    }};
    var node_outputs = [_]core.ValueId{.{ .index = 0 }};
    var nodes = [_]ProgramNode{.{
        .instruction_index = 0,
        .kind = .elementwise,
        .inputs = &.{},
        .outputs = node_outputs[0..],
    }};
    var boundaries = [_]MaterializationBoundary{.{
        .value_id = .{ .index = 0 },
        .reason = .pjrt_output,
    }};
    var schedule = [_]ProgramScheduleItem{
        .{
            .kind = .node,
            .index = 0,
        },
        .{
            .kind = .node,
            .index = 0,
        },
        .{
            .kind = .materialization_boundary,
            .index = 0,
            .count = 1,
        },
    };
    const program = Program{
        .allocator = std.testing.allocator,
        .values = values[0..],
        .nodes = nodes[0..],
        .edges = &.{},
        .schedule = schedule[0..],
        .fusion_groups = &.{},
        .materialization_boundaries = boundaries[0..],
    };

    var diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer diagnostics.deinit();
    try std.testing.expectError(error.InvalidProgram, program.validateWithWriter(&diagnostics.writer));
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "schedules node 0 more than once") != null);
}

test "backend program validator rejects dependency order violations" {
    var values = [_]ProgramValue{
        .{
            .value_id = .{ .index = 0 },
            .producer_node = 0,
            .last_use_node = 1,
        },
        .{
            .value_id = .{ .index = 1 },
            .producer_node = 1,
            .is_output = true,
            .materialization_boundary = 0,
        },
    };
    var node0_outputs = [_]core.ValueId{.{ .index = 0 }};
    var node1_inputs = [_]core.ValueId{.{ .index = 0 }};
    var node1_outputs = [_]core.ValueId{.{ .index = 1 }};
    var nodes = [_]ProgramNode{
        .{
            .instruction_index = 0,
            .kind = .elementwise,
            .inputs = &.{},
            .outputs = node0_outputs[0..],
        },
        .{
            .instruction_index = 1,
            .kind = .elementwise,
            .inputs = node1_inputs[0..],
            .outputs = node1_outputs[0..],
        },
    };
    var edges = [_]ProgramEdge{.{
        .value_id = .{ .index = 0 },
        .from_node = 0,
        .to_node = 1,
    }};
    var boundaries = [_]MaterializationBoundary{.{
        .value_id = .{ .index = 1 },
        .reason = .pjrt_output,
    }};
    var schedule = [_]ProgramScheduleItem{
        .{
            .kind = .node,
            .index = 1,
        },
        .{
            .kind = .node,
            .index = 0,
        },
        .{
            .kind = .materialization_boundary,
            .index = 0,
            .count = 1,
        },
    };
    const program = Program{
        .allocator = std.testing.allocator,
        .values = values[0..],
        .nodes = nodes[0..],
        .edges = edges[0..],
        .schedule = schedule[0..],
        .fusion_groups = &.{},
        .materialization_boundaries = boundaries[0..],
    };

    var diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer diagnostics.deinit();
    try std.testing.expectError(error.InvalidProgram, program.validateWithWriter(&diagnostics.writer));
    const message = diagnostics.writer.buffered();
    try std.testing.expect(std.mem.indexOf(u8, message, "violates schedule order") != null);
    try std.testing.expect(std.mem.indexOf(u8, message, "producer node 0") != null);
}

test "backend program validator rejects stale producer metadata" {
    var values = [_]ProgramValue{.{
        .value_id = .{ .index = 0 },
        .is_output = true,
        .materialization_boundary = 0,
    }};
    var node_outputs = [_]core.ValueId{.{ .index = 0 }};
    var nodes = [_]ProgramNode{.{
        .instruction_index = 0,
        .kind = .elementwise,
        .inputs = &.{},
        .outputs = node_outputs[0..],
    }};
    var boundaries = [_]MaterializationBoundary{.{
        .value_id = .{ .index = 0 },
        .reason = .pjrt_output,
    }};
    var schedule = [_]ProgramScheduleItem{
        .{ .kind = .node, .index = 0 },
        .{ .kind = .materialization_boundary, .index = 0, .count = 1 },
    };
    const program = Program{
        .allocator = std.testing.allocator,
        .values = values[0..],
        .nodes = nodes[0..],
        .edges = &.{},
        .schedule = schedule[0..],
        .fusion_groups = &.{},
        .materialization_boundaries = boundaries[0..],
    };

    var diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer diagnostics.deinit();
    try std.testing.expectError(error.InvalidProgram, program.validateWithWriter(&diagnostics.writer));
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "producer metadata does not match node outputs") != null);
}

test "backend program validator rejects missing producer edges" {
    var values = [_]ProgramValue{
        .{
            .value_id = .{ .index = 0 },
            .producer_node = 0,
            .last_use_node = 1,
        },
        .{
            .value_id = .{ .index = 1 },
            .producer_node = 1,
            .is_output = true,
            .materialization_boundary = 0,
        },
    };
    var node0_outputs = [_]core.ValueId{.{ .index = 0 }};
    var node1_inputs = [_]core.ValueId{.{ .index = 0 }};
    var node1_outputs = [_]core.ValueId{.{ .index = 1 }};
    var nodes = [_]ProgramNode{
        .{
            .instruction_index = 0,
            .kind = .elementwise,
            .inputs = &.{},
            .outputs = node0_outputs[0..],
        },
        .{
            .instruction_index = 1,
            .kind = .elementwise,
            .inputs = node1_inputs[0..],
            .outputs = node1_outputs[0..],
        },
    };
    var boundaries = [_]MaterializationBoundary{.{
        .value_id = .{ .index = 1 },
        .reason = .pjrt_output,
    }};
    var schedule = [_]ProgramScheduleItem{
        .{ .kind = .node, .index = 0 },
        .{ .kind = .node, .index = 1 },
        .{ .kind = .materialization_boundary, .index = 0, .count = 1 },
    };
    const program = Program{
        .allocator = std.testing.allocator,
        .values = values[0..],
        .nodes = nodes[0..],
        .edges = &.{},
        .schedule = schedule[0..],
        .fusion_groups = &.{},
        .materialization_boundaries = boundaries[0..],
    };

    var diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer diagnostics.deinit();
    try std.testing.expectError(error.InvalidProgram, program.validateWithWriter(&diagnostics.writer));
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "missing producer edge") != null);
}

test "backend program validator rejects output without pjrt boundary" {
    var values = [_]ProgramValue{.{
        .value_id = .{ .index = 0 },
        .producer_node = 0,
        .is_output = true,
    }};
    var node_outputs = [_]core.ValueId{.{ .index = 0 }};
    var nodes = [_]ProgramNode{.{
        .instruction_index = 0,
        .kind = .elementwise,
        .inputs = &.{},
        .outputs = node_outputs[0..],
    }};
    var schedule = [_]ProgramScheduleItem{.{
        .kind = .node,
        .index = 0,
    }};
    const program = Program{
        .allocator = std.testing.allocator,
        .values = values[0..],
        .nodes = nodes[0..],
        .edges = &.{},
        .schedule = schedule[0..],
        .fusion_groups = &.{},
        .materialization_boundaries = &.{},
    };

    var diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer diagnostics.deinit();
    try std.testing.expectError(error.InvalidProgram, program.validateWithWriter(&diagnostics.writer));
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "output value 0 has no materialization boundary") != null);
}

test "backend program validator rejects pjrt output boundary for non-output value" {
    var values = [_]ProgramValue{.{
        .value_id = .{ .index = 0 },
        .producer_node = 0,
        .materialization_boundary = 0,
    }};
    var node_outputs = [_]core.ValueId{.{ .index = 0 }};
    var nodes = [_]ProgramNode{.{
        .instruction_index = 0,
        .kind = .elementwise,
        .inputs = &.{},
        .outputs = node_outputs[0..],
    }};
    var boundaries = [_]MaterializationBoundary{.{
        .value_id = .{ .index = 0 },
        .reason = .pjrt_output,
    }};
    var schedule = [_]ProgramScheduleItem{
        .{ .kind = .node, .index = 0 },
        .{ .kind = .materialization_boundary, .index = 0, .count = 1 },
    };
    const program = Program{
        .allocator = std.testing.allocator,
        .values = values[0..],
        .nodes = nodes[0..],
        .edges = &.{},
        .schedule = schedule[0..],
        .fusion_groups = &.{},
        .materialization_boundaries = boundaries[0..],
    };

    var diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer diagnostics.deinit();
    try std.testing.expectError(error.InvalidProgram, program.validateWithWriter(&diagnostics.writer));
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "marks non-output value 0 as pjrt_output") != null);
}

test "backend program validator rejects output boundary reason mismatch" {
    var values = [_]ProgramValue{.{
        .value_id = .{ .index = 0 },
        .producer_node = 0,
        .is_output = true,
        .materialization_boundary = 0,
    }};
    var node_outputs = [_]core.ValueId{.{ .index = 0 }};
    var nodes = [_]ProgramNode{.{
        .instruction_index = 0,
        .kind = .elementwise,
        .inputs = &.{},
        .outputs = node_outputs[0..],
    }};
    var boundaries = [_]MaterializationBoundary{.{
        .value_id = .{ .index = 0 },
        .reason = .backend_requirement,
    }};
    var schedule = [_]ProgramScheduleItem{
        .{ .kind = .node, .index = 0 },
        .{ .kind = .materialization_boundary, .index = 0, .count = 1 },
    };
    const program = Program{
        .allocator = std.testing.allocator,
        .values = values[0..],
        .nodes = nodes[0..],
        .edges = &.{},
        .schedule = schedule[0..],
        .fusion_groups = &.{},
        .materialization_boundaries = boundaries[0..],
    };

    var diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer diagnostics.deinit();
    try std.testing.expectError(error.InvalidProgram, program.validateWithWriter(&diagnostics.writer));
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "output value 0 materialization boundary reason is not pjrt_output") != null);
}

test "backend program validator rejects stale fusion group inputs" {
    var values = [_]ProgramValue{
        .{
            .value_id = .{ .index = 0 },
            .last_use_node = 0,
        },
        .{
            .value_id = .{ .index = 1 },
            .producer_node = 0,
            .is_output = true,
            .materialization_boundary = 0,
        },
    };
    var node_inputs = [_]core.ValueId{.{ .index = 0 }};
    var node_outputs = [_]core.ValueId{.{ .index = 1 }};
    var nodes = [_]ProgramNode{.{
        .instruction_index = 0,
        .kind = .elementwise,
        .inputs = node_inputs[0..],
        .outputs = node_outputs[0..],
        .fusion_group = 0,
    }};
    var group_nodes = [_]usize{0};
    var group_outputs = [_]core.ValueId{.{ .index = 1 }};
    var groups = [_]FusionGroup{.{
        .id = 0,
        .kind = .view_elementwise,
        .first_node = 0,
        .last_node = 0,
        .node_count = 1,
        .node_indices = group_nodes[0..],
        .output_values = group_outputs[0..],
    }};
    var boundaries = [_]MaterializationBoundary{.{
        .value_id = .{ .index = 1 },
        .reason = .pjrt_output,
    }};
    var schedule = [_]ProgramScheduleItem{
        .{ .kind = .fusion_group, .index = 0, .count = 1 },
        .{ .kind = .materialization_boundary, .index = 0, .count = 1 },
    };
    const program = Program{
        .allocator = std.testing.allocator,
        .values = values[0..],
        .nodes = nodes[0..],
        .edges = &.{},
        .schedule = schedule[0..],
        .fusion_groups = groups[0..],
        .materialization_boundaries = boundaries[0..],
        .fusion_group_count = 1,
    };

    var diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer diagnostics.deinit();
    try std.testing.expectError(error.InvalidProgram, program.validateWithWriter(&diagnostics.writer));
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "fusion group 0 input_values") != null);
}

test "backend program validator rejects stale fusion group outputs" {
    var values = [_]ProgramValue{
        .{
            .value_id = .{ .index = 0 },
            .last_use_node = 0,
        },
        .{
            .value_id = .{ .index = 1 },
            .producer_node = 0,
            .is_output = true,
            .materialization_boundary = 0,
        },
    };
    var node_inputs = [_]core.ValueId{.{ .index = 0 }};
    var node_outputs = [_]core.ValueId{.{ .index = 1 }};
    var nodes = [_]ProgramNode{.{
        .instruction_index = 0,
        .kind = .elementwise,
        .inputs = node_inputs[0..],
        .outputs = node_outputs[0..],
        .fusion_group = 0,
    }};
    var group_nodes = [_]usize{0};
    var group_inputs = [_]core.ValueId{.{ .index = 0 }};
    var groups = [_]FusionGroup{.{
        .id = 0,
        .kind = .view_elementwise,
        .first_node = 0,
        .last_node = 0,
        .node_count = 1,
        .node_indices = group_nodes[0..],
        .input_values = group_inputs[0..],
    }};
    var boundaries = [_]MaterializationBoundary{.{
        .value_id = .{ .index = 1 },
        .reason = .pjrt_output,
    }};
    var schedule = [_]ProgramScheduleItem{
        .{ .kind = .fusion_group, .index = 0, .count = 1 },
        .{ .kind = .materialization_boundary, .index = 0, .count = 1 },
    };
    const program = Program{
        .allocator = std.testing.allocator,
        .values = values[0..],
        .nodes = nodes[0..],
        .edges = &.{},
        .schedule = schedule[0..],
        .fusion_groups = groups[0..],
        .materialization_boundaries = boundaries[0..],
        .fusion_group_count = 1,
    };

    var diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer diagnostics.deinit();
    try std.testing.expectError(error.InvalidProgram, program.validateWithWriter(&diagnostics.writer));
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "fusion group 0 output_values") != null);
}

test "backend program validator rejects inconsistent fusion membership" {
    var values = [_]ProgramValue{
        .{
            .value_id = .{ .index = 0 },
            .last_use_node = 0,
        },
        .{
            .value_id = .{ .index = 1 },
            .producer_node = 0,
            .is_output = true,
            .materialization_boundary = 0,
        },
    };
    var node_inputs = [_]core.ValueId{.{ .index = 0 }};
    var node_outputs = [_]core.ValueId{.{ .index = 1 }};
    var nodes = [_]ProgramNode{.{
        .instruction_index = 0,
        .kind = .elementwise,
        .inputs = node_inputs[0..],
        .outputs = node_outputs[0..],
        .fusion_group = 0,
    }};
    var group_nodes = [_]usize{0};
    var group_inputs = [_]core.ValueId{.{ .index = 0 }};
    var group_outputs = [_]core.ValueId{.{ .index = 1 }};
    var groups = [_]FusionGroup{.{
        .id = 0,
        .kind = .view_elementwise,
        .first_node = 0,
        .last_node = 0,
        .node_count = 1,
        .node_indices = group_nodes[0..],
        .input_values = group_inputs[0..],
        .output_values = group_outputs[0..],
    }};
    var boundaries = [_]MaterializationBoundary{.{
        .value_id = .{ .index = 1 },
        .reason = .pjrt_output,
    }};
    var schedule = [_]ProgramScheduleItem{.{
        .kind = .node,
        .index = 0,
    }};
    const program = Program{
        .allocator = std.testing.allocator,
        .values = values[0..],
        .nodes = nodes[0..],
        .edges = &.{},
        .schedule = schedule[0..],
        .fusion_groups = groups[0..],
        .materialization_boundaries = boundaries[0..],
        .fusion_group_count = 1,
    };

    var diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer diagnostics.deinit();
    try std.testing.expectError(error.InvalidProgram, program.validateWithWriter(&diagnostics.writer));
    const message = diagnostics.writer.buffered();
    try std.testing.expect(std.mem.indexOf(u8, message, "pass=backend-program-verify") != null);
    try std.testing.expect(std.mem.indexOf(u8, message, "directly schedules fused node 0") != null);
}
