const std = @import("std");
const ir = @import("src/compiler/ir");

const schedule = @import("program_validation_schedule.zig");
const diagnostic = @import("program_validation_diagnostic.zig");

pub const Error = diagnostic.Error;
pub const invalidProgram = diagnostic.invalidProgram;

/// Verifies backend-program structural invariants and writes diagnostics on failure.
pub fn validate(program: anytype, writer: ?*std.Io.Writer) (Error || std.Io.Writer.Error)!void {
    if (program.fusion_group_count != program.fusion_groups.len) {
        return invalidProgram(writer, "fusion_group_count={} does not match fusion_groups.len={}", .{ program.fusion_group_count, program.fusion_groups.len });
    }

    try validateValues(program, writer);
    try validateNodes(program, writer);
    try validateControlFlows(program, writer);
    try validateSubprograms(program, writer);
    try validateEdges(program, writer);
    try validateFusionGroups(program, writer);
    try validateMaterializationBoundaries(program, writer);
    try schedule.validate(program, writer);
}

fn validateValues(program: anytype, writer: ?*std.Io.Writer) (Error || std.Io.Writer.Error)!void {
    for (program.values, 0..) |value, value_index| {
        if (value.value_id.index != value_index) {
            return invalidProgram(writer, "value table slot {} contains value_id={}", .{ value_index, value.value_id.index });
        }
        if (value.producer_node) |producer_node| {
            if (producer_node >= program.nodes.len) {
                return invalidProgram(writer, "value {} producer_node={} is outside nodes.len={}", .{ value_index, producer_node, program.nodes.len });
            }
        }
        if (value.last_use_node) |last_use_node| {
            if (last_use_node >= program.nodes.len) {
                return invalidProgram(writer, "value {} last_use_node={} is outside nodes.len={}", .{ value_index, last_use_node, program.nodes.len });
            }
        }
        if (value.materialization_boundary) |boundary_index| {
            if (boundary_index >= program.materialization_boundaries.len) {
                return invalidProgram(writer, "value {} materialization_boundary={} is outside materialization_boundaries.len={}", .{ value_index, boundary_index, program.materialization_boundaries.len });
            }
        }
    }
}

fn validateNodes(program: anytype, writer: ?*std.Io.Writer) (Error || std.Io.Writer.Error)!void {
    for (program.nodes, 0..) |node, node_index| {
        for (node.inputs) |input| {
            if (input.index >= program.values.len) {
                return invalidProgram(writer, "node {} input value_id={} is outside values.len={}", .{ node_index, input.index, program.values.len });
            }
        }
        for (node.outputs) |output| {
            if (output.index >= program.values.len) {
                return invalidProgram(writer, "node {} output value_id={} is outside values.len={}", .{ node_index, output.index, program.values.len });
            }
        }
        if (node.fusion_group) |group_index| {
            if (group_index >= program.fusion_groups.len) {
                return invalidProgram(writer, "node {} fusion_group={} is outside fusion_groups.len={}", .{ node_index, group_index, program.fusion_groups.len });
            }
        }
        if (node.control_flow) |control_flow_index| {
            if (control_flow_index >= program.control_flows.len) {
                return invalidProgram(writer, "node {} control_flow={} is outside control_flows.len={}", .{ node_index, control_flow_index, program.control_flows.len });
            }
            if (program.control_flows[control_flow_index].parent_node != node_index) {
                return invalidProgram(writer, "node {} control_flow={} parent_node metadata does not point back to the node", .{ node_index, control_flow_index });
            }
            if (node.kind != .control_flow) {
                return invalidProgram(writer, "node {} owns control_flow={} but is not a control_flow node", .{ node_index, control_flow_index });
            }
        }
        for (node.subprograms) |subprogram_index| {
            if (subprogram_index >= program.subprograms.len) {
                return invalidProgram(writer, "node {} subprogram={} is outside subprograms.len={}", .{ node_index, subprogram_index, program.subprograms.len });
            }
            if (program.subprograms[subprogram_index].parent_node != node_index) {
                return invalidProgram(writer, "node {} subprogram={} parent_node metadata does not point back to the node", .{ node_index, subprogram_index });
            }
            if (node.kind != .control_flow) {
                return invalidProgram(writer, "node {} owns subprogram={} but is not a control_flow node", .{ node_index, subprogram_index });
            }
        }
    }
}

fn validateControlFlows(program: anytype, writer: ?*std.Io.Writer) (Error || std.Io.Writer.Error)!void {
    for (program.control_flows, 0..) |control_flow, control_flow_index| {
        if (control_flow.id != control_flow_index) {
            return invalidProgram(writer, "control_flow slot {} contains id={}", .{ control_flow_index, control_flow.id });
        }
        if (control_flow.parent_node >= program.nodes.len) {
            return invalidProgram(writer, "control_flow {} parent_node={} is outside nodes.len={}", .{ control_flow_index, control_flow.parent_node, program.nodes.len });
        }
        const parent = program.nodes[control_flow.parent_node];
        if (parent.control_flow != control_flow_index) {
            return invalidProgram(writer, "control_flow {} is not referenced by parent node {}", .{ control_flow_index, control_flow.parent_node });
        }
        if (control_flow.condition_subprogram >= program.subprograms.len or control_flow.body_subprogram >= program.subprograms.len) {
            return invalidProgram(writer, "control_flow {} references subprograms outside subprograms.len={}", .{ control_flow_index, program.subprograms.len });
        }
        const condition = program.subprograms[control_flow.condition_subprogram];
        const body = program.subprograms[control_flow.body_subprogram];
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
            if (state_input.index >= program.values.len) {
                return invalidProgram(writer, "control_flow {} state input value_id={} is outside values.len={}", .{ control_flow_index, state_input.index, program.values.len });
            }
        }
        for (control_flow.state_outputs) |state_output| {
            if (state_output.index >= program.values.len) {
                return invalidProgram(writer, "control_flow {} state output value_id={} is outside values.len={}", .{ control_flow_index, state_output.index, program.values.len });
            }
        }
    }
}

fn validateSubprograms(program: anytype, writer: ?*std.Io.Writer) (Error || std.Io.Writer.Error)!void {
    for (program.subprograms, 0..) |subprogram, subprogram_index| {
        if (subprogram.id != subprogram_index) {
            return invalidProgram(writer, "subprogram slot {} contains id={}", .{ subprogram_index, subprogram.id });
        }
        for (subprogram.values, 0..) |value, value_index| {
            if (value.id.index != value_index) {
                return invalidProgram(writer, "subprogram {} value slot {} contains value_id={}", .{ subprogram_index, value_index, value.id.index });
            }
        }
        if (subprogram.parent_node >= program.nodes.len) {
            return invalidProgram(writer, "subprogram {} parent_node={} is outside nodes.len={}", .{ subprogram_index, subprogram.parent_node, program.nodes.len });
        }
        if (!containsUsize(program.nodes[subprogram.parent_node].subprograms, subprogram_index)) {
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
}

fn validateEdges(program: anytype, writer: ?*std.Io.Writer) (Error || std.Io.Writer.Error)!void {
    const computed_producers = try program.allocator.alloc(?usize, program.values.len);
    defer program.allocator.free(computed_producers);
    @memset(computed_producers, null);
    const computed_last_uses = try program.allocator.alloc(?usize, program.values.len);
    defer program.allocator.free(computed_last_uses);
    @memset(computed_last_uses, null);

    for (program.nodes, 0..) |node, node_index| {
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

    for (program.values, 0..) |value, value_index| {
        if (value.producer_node != computed_producers[value_index]) {
            return invalidProgram(writer, "value {} producer metadata does not match node outputs", .{value_index});
        }
        if (value.last_use_node != computed_last_uses[value_index]) {
            return invalidProgram(writer, "value {} last-use metadata does not match node inputs", .{value_index});
        }
    }

    for (program.edges, 0..) |edge, edge_index| {
        if (edge.value_id.index >= program.values.len) {
            return invalidProgram(writer, "edge {} value_id={} is outside values.len={}", .{ edge_index, edge.value_id.index, program.values.len });
        }
        if (edge.from_node >= program.nodes.len or edge.to_node >= program.nodes.len) {
            return invalidProgram(writer, "edge {} references nodes from={} to={} outside nodes.len={}", .{ edge_index, edge.from_node, edge.to_node, program.nodes.len });
        }
        if (program.values[edge.value_id.index].producer_node != edge.from_node) {
            return invalidProgram(writer, "edge {} value_id={} producer does not match from_node={}", .{ edge_index, edge.value_id.index, edge.from_node });
        }
        if (!containsValueId(program.nodes[edge.from_node].outputs, edge.value_id)) {
            return invalidProgram(writer, "edge {} value_id={} is not produced by from_node={}", .{ edge_index, edge.value_id.index, edge.from_node });
        }
        if (!containsValueId(program.nodes[edge.to_node].inputs, edge.value_id)) {
            return invalidProgram(writer, "edge {} value_id={} is not consumed by to_node={}", .{ edge_index, edge.value_id.index, edge.to_node });
        }
    }

    for (program.nodes, 0..) |node, node_index| {
        for (node.inputs) |input| {
            const producer_node = program.values[input.index].producer_node orelse continue;
            if (!hasEdge(program, input, producer_node, node_index)) {
                return invalidProgram(writer, "node {} input value_id={} is missing producer edge from node {}", .{ node_index, input.index, producer_node });
            }
        }
    }
}

fn validateFusionGroups(program: anytype, writer: ?*std.Io.Writer) (Error || std.Io.Writer.Error)!void {
    for (program.fusion_groups, 0..) |group, group_index| {
        if (group.id != group_index) {
            return invalidProgram(writer, "fusion group slot {} contains id={}", .{ group_index, group.id });
        }
        if (group.node_count == 0) {
            return invalidProgram(writer, "fusion group {} is empty", .{group_index});
        }
        if (group.node_count != group.node_indices.len) {
            return invalidProgram(writer, "fusion group {} node_count={} does not match node_indices.len={}", .{ group_index, group.node_count, group.node_indices.len });
        }
        if (group.first_node >= program.nodes.len or group.last_node >= program.nodes.len) {
            return invalidProgram(writer, "fusion group {} range first={} last={} is outside nodes.len={}", .{ group_index, group.first_node, group.last_node, program.nodes.len });
        }
        if (group.last_node < group.first_node) {
            return invalidProgram(writer, "fusion group {} range is reversed first={} last={}", .{ group_index, group.first_node, group.last_node });
        }

        for (group.node_indices) |node_index| {
            if (node_index >= program.nodes.len) {
                return invalidProgram(writer, "fusion group {} node_index={} is outside nodes.len={}", .{ group_index, node_index, program.nodes.len });
            }
            if (node_index < group.first_node or node_index > group.last_node) {
                return invalidProgram(writer, "fusion group {} node_index={} is outside group range {}..{}", .{ group_index, node_index, group.first_node, group.last_node });
            }
            if (program.nodes[node_index].fusion_group != group_index) {
                return invalidProgram(writer, "fusion group {} includes node {} whose node.fusion_group is not this group", .{ group_index, node_index });
            }
        }
        for (group.input_values) |value_id| {
            if (value_id.index >= program.values.len) {
                return invalidProgram(writer, "fusion group {} input value_id={} is outside values.len={}", .{ group_index, value_id.index, program.values.len });
            }
        }
        for (group.output_values) |value_id| {
            if (value_id.index >= program.values.len) {
                return invalidProgram(writer, "fusion group {} output value_id={} is outside values.len={}", .{ group_index, value_id.index, program.values.len });
            }
        }
    }

    const fusion_mark_count = program.fusion_groups.len * program.values.len;
    const computed_group_inputs = try program.allocator.alloc(bool, fusion_mark_count);
    defer program.allocator.free(computed_group_inputs);
    @memset(computed_group_inputs, false);
    const computed_group_outputs = try program.allocator.alloc(bool, fusion_mark_count);
    defer program.allocator.free(computed_group_outputs);
    @memset(computed_group_outputs, false);

    for (program.nodes) |node| {
        const group_index = node.fusion_group orelse continue;
        for (node.inputs) |input| {
            if (program.values[input.index].producer_node == null) {
                computed_group_inputs[groupMarkIndex(program.values.len, group_index, input.index)] = true;
            }
        }
    }
    for (program.edges) |edge| {
        const from_group = program.nodes[edge.from_node].fusion_group;
        const to_group = program.nodes[edge.to_node].fusion_group;
        if (to_group) |group_index| {
            if (from_group != group_index) {
                computed_group_inputs[groupMarkIndex(program.values.len, group_index, edge.value_id.index)] = true;
            }
        }
        if (from_group) |group_index| {
            if (to_group != group_index) {
                computed_group_outputs[groupMarkIndex(program.values.len, group_index, edge.value_id.index)] = true;
            }
        }
    }
    for (program.materialization_boundaries) |boundary| {
        const producer_node = program.values[boundary.value_id.index].producer_node orelse continue;
        const group_index = program.nodes[producer_node].fusion_group orelse continue;
        computed_group_outputs[groupMarkIndex(program.values.len, group_index, boundary.value_id.index)] = true;
    }
    for (program.fusion_groups, 0..) |group, group_index| {
        try validateMarkedValueSet(
            writer,
            "input_values",
            group_index,
            group.input_values,
            program.values,
            computed_group_inputs[groupMarkIndex(program.values.len, group_index, 0)..][0..program.values.len],
        );
        try validateMarkedValueSet(
            writer,
            "output_values",
            group_index,
            group.output_values,
            program.values,
            computed_group_outputs[groupMarkIndex(program.values.len, group_index, 0)..][0..program.values.len],
        );
    }
}

fn validateMaterializationBoundaries(program: anytype, writer: ?*std.Io.Writer) (Error || std.Io.Writer.Error)!void {
    for (program.materialization_boundaries, 0..) |boundary, boundary_index| {
        if (boundary.value_id.index >= program.values.len) {
            return invalidProgram(writer, "materialization boundary {} value_id={} is outside values.len={}", .{ boundary_index, boundary.value_id.index, program.values.len });
        }
        if (program.values[boundary.value_id.index].materialization_boundary != boundary_index) {
            return invalidProgram(writer, "materialization boundary {} is not referenced by value {}", .{ boundary_index, boundary.value_id.index });
        }
        if (boundary.reason == .pjrt_output and !program.values[boundary.value_id.index].is_output) {
            return invalidProgram(writer, "materialization boundary {} marks non-output value {} as pjrt_output", .{ boundary_index, boundary.value_id.index });
        }
    }

    for (program.values, 0..) |value, value_index| {
        if (!value.is_output) continue;
        const boundary_index = value.materialization_boundary orelse
            return invalidProgram(writer, "output value {} has no materialization boundary", .{value_index});
        if (program.materialization_boundaries[boundary_index].reason != .pjrt_output) {
            return invalidProgram(writer, "output value {} materialization boundary reason is not pjrt_output", .{value_index});
        }
    }
}

fn hasEdge(program: anytype, value_id: ir.ValueId, from_node: usize, to_node: usize) bool {
    for (program.edges) |edge| {
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
    values: anytype,
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
