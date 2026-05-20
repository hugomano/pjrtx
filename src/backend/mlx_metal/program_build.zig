const std = @import("std");

const ir = @import("src/compiler/ir");
const program_mod = @import("program.zig");

/// Builds an owned MLX backend program graph from a compiler executable plan.
pub fn build(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, diagnostic_writer: ?*std.Io.Writer) !program_mod.Program {
    var nodes = try allocator.alloc(program_mod.Node, plan.instructions.len);
    errdefer allocator.free(nodes);
    var initialized_nodes: usize = 0;
    errdefer {
        for (nodes[0..initialized_nodes]) |node| {
            allocator.free(node.inputs);
            allocator.free(node.outputs);
            if (node.subprograms.len != 0) allocator.free(node.subprograms);
        }
    }
    const subprogram_capacity = try countPlanSubprograms(plan);
    const subprograms = try allocator.alloc(program_mod.Subprogram, subprogram_capacity);
    var initialized_subprograms: usize = 0;
    var subprograms_owned = true;
    errdefer if (subprograms_owned) deinitProgramSubprograms(allocator, subprograms[0..initialized_subprograms]);
    const control_flow_capacity = try countPlanControlFlows(plan);
    const control_flows = try allocator.alloc(program_mod.ControlFlow, control_flow_capacity);
    var initialized_control_flows: usize = 0;
    var control_flows_owned = true;
    errdefer if (control_flows_owned) deinitProgramControlFlows(allocator, control_flows[0..initialized_control_flows]);

    const last_uses = try allocator.alloc(?usize, plan.values.len);
    defer allocator.free(last_uses);
    @memset(last_uses, null);

    const value_producers = try allocator.alloc(?usize, plan.values.len);
    defer allocator.free(value_producers);
    @memset(value_producers, null);

    const output_values = try allocator.alloc(bool, plan.values.len);
    defer allocator.free(output_values);
    @memset(output_values, false);
    for (plan.output_ids) |output_id| {
        if (output_id.index < output_values.len) output_values[output_id.index] = true;
    }

    const max_fusion_groups = try allocator.alloc(program_mod.FusionGroup, plan.instructions.len);
    defer allocator.free(max_fusion_groups);

    var current_fusion_group: ?usize = null;
    var fusion_group_count: usize = 0;
    for (plan.instructions, 0..) |instruction, instruction_index| {
        for (instruction.inputs) |input_id| {
            if (input_id.index < last_uses.len) {
                last_uses[input_id.index] = instruction_index;
            }
        }
        if (instruction.kind == .get_tuple_element and instruction.inputs.len >= 1) {
            const tuple_id = instruction.inputs[0];
            if (tuple_id.index < plan.values.len) {
                const tuple_value = plan.values[tuple_id.index];
                if (tuple_value.storage == .tuple) {
                    if (instruction.tuple_index) |tuple_index| {
                        if (tuple_index >= 0 and tuple_index < @as(i64, @intCast(tuple_value.elements.len))) {
                            const element_id = tuple_value.elements[@intCast(tuple_index)];
                            if (element_id.index < last_uses.len) last_uses[element_id.index] = instruction_index;
                        }
                    } else {
                        for (tuple_value.elements) |element_id| {
                            if (element_id.index < last_uses.len) last_uses[element_id.index] = instruction_index;
                        }
                    }
                }
            }
        }
        for (instruction.outputs) |output_id| {
            if (output_id.index < value_producers.len) value_producers[output_id.index] = instruction_index;
        }
        const node_kind = programNodeKind(instruction.kind);
        const fusion_group = if (programNodeFusible(node_kind)) group: {
            if (current_fusion_group == null) {
                current_fusion_group = fusion_group_count;
                max_fusion_groups[fusion_group_count] = .{
                    .id = fusion_group_count,
                    .kind = .view_elementwise,
                    .first_node = instruction_index,
                    .last_node = instruction_index,
                    .node_count = 0,
                };
                fusion_group_count += 1;
            }
            max_fusion_groups[current_fusion_group.?].last_node = instruction_index;
            max_fusion_groups[current_fusion_group.?].node_count += 1;
            break :group current_fusion_group;
        } else group: {
            current_fusion_group = null;
            break :group null;
        };
        const node_inputs = try programNodeInputs(allocator, plan, instruction);
        var node_inputs_owned = true;
        errdefer if (node_inputs_owned) allocator.free(node_inputs);
        const node_outputs = try allocator.dupe(ir.ValueId, instruction.outputs);
        var node_outputs_owned = true;
        errdefer if (node_outputs_owned) allocator.free(node_outputs);
        const node_subprograms = try buildNodeSubprograms(allocator, plan, instruction, instruction_index, subprograms, &initialized_subprograms);
        var node_subprograms_owned = true;
        errdefer if (node_subprograms_owned and node_subprograms.len != 0) allocator.free(node_subprograms);
        const node_control_flow = try buildNodeControlFlow(
            allocator,
            instruction,
            instruction_index,
            node_subprograms,
            subprograms,
            control_flows,
            &initialized_control_flows,
        );
        nodes[instruction_index] = .{
            .instruction_index = instruction_index,
            .kind = node_kind,
            .inputs = node_inputs,
            .outputs = node_outputs,
            .subprograms = node_subprograms,
            .control_flow = node_control_flow,
            .materializes = instructionMaterializes(instruction.kind),
            .fusion_group = fusion_group,
        };
        node_inputs_owned = false;
        node_outputs_owned = false;
        node_subprograms_owned = false;
        initialized_nodes += 1;
    }

    const materialization_boundaries = try allocator.alloc(program_mod.MaterializationBoundary, plan.output_ids.len);
    errdefer allocator.free(materialization_boundaries);
    for (plan.output_ids, 0..) |output_id, index| {
        materialization_boundaries[index] = .{
            .value_id = output_id,
            .reason = .pjrt_output,
        };
    }

    const values = try allocator.alloc(program_mod.Value, plan.values.len);
    errdefer allocator.free(values);
    for (plan.values, 0..) |value, value_index| {
        values[value_index] = .{
            .value_id = value.id,
            .byte_size = ir.denseByteSize(value.descriptor.element_type, value.descriptor.dims),
            .producer_node = value_producers[value_index],
            .last_use_node = last_uses[value_index],
            .is_output = output_values[value_index],
        };
    }
    for (materialization_boundaries, 0..) |boundary, boundary_index| {
        if (boundary.value_id.index >= values.len) continue;
        if (values[boundary.value_id.index].materialization_boundary == null) {
            values[boundary.value_id.index].materialization_boundary = boundary_index;
        }
    }

    var edge_count: usize = 0;
    for (nodes) |node| {
        for (node.inputs) |input_id| {
            if (input_id.index >= values.len) continue;
            if (values[input_id.index].producer_node != null) edge_count += 1;
        }
    }

    const edges = try allocator.alloc(program_mod.Edge, edge_count);
    errdefer allocator.free(edges);
    var edge_index: usize = 0;
    for (nodes, 0..) |node, to_node| {
        for (node.inputs) |input_id| {
            if (input_id.index >= values.len) continue;
            const from_node = values[input_id.index].producer_node orelse continue;
            edges[edge_index] = .{
                .value_id = input_id,
                .from_node = from_node,
                .to_node = to_node,
            };
            edge_index += 1;
        }
    }

    const fusion_groups = try buildFusionGroups(allocator, max_fusion_groups[0..fusion_group_count], nodes, values, edges);
    errdefer deinitFusionGroups(allocator, fusion_groups);

    const max_schedule_len = nodes.len + if (materialization_boundaries.len == 0) @as(usize, 0) else 1;
    const max_schedule = try allocator.alloc(program_mod.ScheduleItem, max_schedule_len);
    defer allocator.free(max_schedule);
    var schedule_index: usize = 0;
    var node_index: usize = 0;
    while (node_index < nodes.len) {
        const node = nodes[node_index];
        if (node.fusion_group) |group_id| {
            const group = fusion_groups[group_id];
            if (group.first_node == node_index) {
                max_schedule[schedule_index] = .{
                    .kind = .fusion_group,
                    .index = group_id,
                    .count = group.node_indices.len,
                };
                schedule_index += 1;
                node_index = group.last_node + 1;
                continue;
            }
        }
        max_schedule[schedule_index] = .{
            .kind = .node,
            .index = node_index,
        };
        schedule_index += 1;
        node_index += 1;
    }
    if (materialization_boundaries.len != 0) {
        max_schedule[schedule_index] = .{
            .kind = .materialization_boundary,
            .index = 0,
            .count = materialization_boundaries.len,
        };
        schedule_index += 1;
    }
    const schedule = try allocator.dupe(program_mod.ScheduleItem, max_schedule[0..schedule_index]);
    errdefer allocator.free(schedule);

    const program = program_mod.Program{
        .allocator = allocator,
        .values = values,
        .nodes = nodes,
        .edges = edges,
        .schedule = schedule,
        .subprograms = subprograms,
        .control_flows = control_flows,
        .fusion_groups = fusion_groups,
        .materialization_boundaries = materialization_boundaries,
        .fusion_group_count = fusion_group_count,
    };
    try program.validateWithWriter(diagnostic_writer);
    subprograms_owned = false;
    control_flows_owned = false;
    return program;
}

fn countPlanSubprograms(plan: *const ir.ExecutablePlan) !usize {
    var count: usize = 0;
    for (plan.instructions) |instruction| {
        if (instruction.kind != .while_) continue;
        count += instruction.region_ids.len;
    }
    return count;
}

fn countPlanControlFlows(plan: *const ir.ExecutablePlan) !usize {
    var count: usize = 0;
    for (plan.instructions) |instruction| {
        if (instruction.kind == .while_) count += 1;
    }
    return count;
}

fn buildNodeSubprograms(
    allocator: std.mem.Allocator,
    plan: *const ir.ExecutablePlan,
    instruction: ir.PlanInstruction,
    instruction_index: usize,
    subprograms: []program_mod.Subprogram,
    initialized_subprograms: *usize,
) ![]const usize {
    if (instruction.kind != .while_ or instruction.region_ids.len == 0) return &.{};
    const indices = try allocator.alloc(usize, instruction.region_ids.len);
    errdefer allocator.free(indices);
    for (instruction.region_ids, 0..) |region_id, index| {
        if (region_id.index >= plan.regions.len or initialized_subprograms.* >= subprograms.len) return error.InvalidProgram;
        const subprogram_index = initialized_subprograms.*;
        subprograms[subprogram_index] = try cloneProgramSubprogram(
            allocator,
            plan.regions[region_id.index],
            subprogram_index,
            instruction_index,
        );
        initialized_subprograms.* += 1;
        indices[index] = subprogram_index;
    }
    return indices;
}

fn buildNodeControlFlow(
    allocator: std.mem.Allocator,
    instruction: ir.PlanInstruction,
    instruction_index: usize,
    node_subprograms: []const usize,
    subprograms: []const program_mod.Subprogram,
    control_flows: []program_mod.ControlFlow,
    initialized_control_flows: *usize,
) !?usize {
    if (instruction.kind != .while_) return null;
    if (node_subprograms.len != 2 or initialized_control_flows.* >= control_flows.len) return error.InvalidProgram;
    if (node_subprograms[0] >= subprograms.len) return error.InvalidProgram;
    const condition = subprograms[node_subprograms[0]];
    if (condition.terminator_operands.len != 1) return error.InvalidProgram;
    const state_inputs = try allocator.dupe(ir.ValueId, instruction.inputs);
    errdefer allocator.free(state_inputs);
    const state_outputs = try allocator.dupe(ir.ValueId, instruction.outputs);
    errdefer allocator.free(state_outputs);
    const control_flow_index = initialized_control_flows.*;
    control_flows[control_flow_index] = .{
        .id = control_flow_index,
        .parent_node = instruction_index,
        .kind = .while_loop,
        .condition_subprogram = node_subprograms[0],
        .body_subprogram = node_subprograms[1],
        .state_inputs = state_inputs,
        .state_outputs = state_outputs,
        .predicate_output = condition.terminator_operands[0],
    };
    initialized_control_flows.* += 1;
    return control_flow_index;
}

fn cloneProgramSubprogram(
    allocator: std.mem.Allocator,
    region: ir.PlanRegion,
    subprogram_id: usize,
    parent_node: usize,
) !program_mod.Subprogram {
    const values = try cloneRegionValueList(allocator, region.values);
    errdefer program_mod.freeRegionValueList(allocator, values);
    const arguments = try cloneDescriptorList(allocator, region.argument_descriptors);
    errdefer program_mod.freeDescriptorList(allocator, arguments);
    const instructions = try cloneRegionInstructionList(allocator, region.instructions);
    errdefer program_mod.freeRegionInstructionList(allocator, instructions);
    const returns = try cloneDescriptorList(allocator, region.return_descriptors);
    errdefer program_mod.freeDescriptorList(allocator, returns);
    const terminator_operand_ids = try allocator.dupe(ir.RegionValueId, region.terminator_operands);
    errdefer allocator.free(terminator_operand_ids);
    const terminator_operand_descriptors = try cloneDescriptorList(allocator, region.terminator_operand_descriptors);
    errdefer program_mod.freeDescriptorList(allocator, terminator_operand_descriptors);
    return .{
        .id = subprogram_id,
        .parent_node = parent_node,
        .region_id = region.id,
        .kind = region.kind,
        .values = values,
        .argument_descriptors = arguments,
        .instructions = instructions,
        .return_descriptors = returns,
        .terminator_operands = terminator_operand_ids,
        .terminator_operand_descriptors = terminator_operand_descriptors,
    };
}

fn cloneDescriptorList(allocator: std.mem.Allocator, source: []const ir.BufferDescriptor) ![]const ir.BufferDescriptor {
    const descriptors = try allocator.alloc(ir.BufferDescriptor, source.len);
    var initialized: usize = 0;
    errdefer {
        for (descriptors[0..initialized]) |descriptor| {
            if (descriptor.dims.len != 0) allocator.free(descriptor.dims);
        }
        allocator.free(descriptors);
    }
    for (source, descriptors) |src, *dst| {
        const dims = try allocator.dupe(i64, src.dims);
        dst.* = .{
            .element_type = src.element_type,
            .dims = dims,
            .layout = src.layout,
            .device_id = src.device_id,
            .memory_id = src.memory_id,
            .shard_index = src.shard_index,
        };
        initialized += 1;
    }
    return descriptors;
}

fn cloneRegionValueList(allocator: std.mem.Allocator, source: []const ir.RegionValue) ![]const ir.RegionValue {
    const values = try allocator.alloc(ir.RegionValue, source.len);
    var initialized: usize = 0;
    errdefer {
        for (values[0..initialized]) |value| {
            if (value.descriptor.dims.len != 0) allocator.free(value.descriptor.dims);
            if (value.literal) |literal| allocator.free(literal);
        }
        allocator.free(values);
    }
    for (source, values) |src, *dst| {
        const dims = try allocator.dupe(i64, src.descriptor.dims);
        var dims_owned = true;
        errdefer if (dims_owned) allocator.free(dims);
        const literal = if (src.literal) |bytes| try allocator.dupe(u8, bytes) else null;
        var literal_owned = true;
        errdefer if (literal_owned) if (literal) |bytes| allocator.free(bytes);
        dst.* = .{
            .id = src.id,
            .role = src.role,
            .descriptor = .{
                .element_type = src.descriptor.element_type,
                .dims = dims,
                .layout = src.descriptor.layout,
                .device_id = src.descriptor.device_id,
                .memory_id = src.descriptor.memory_id,
                .shard_index = src.descriptor.shard_index,
            },
            .literal = literal,
        };
        dims_owned = false;
        literal_owned = false;
        initialized += 1;
    }
    return values;
}

fn cloneRegionInstructionList(allocator: std.mem.Allocator, source: []const ir.RegionInstruction) ![]const ir.RegionInstruction {
    const instructions = try allocator.alloc(ir.RegionInstruction, source.len);
    var initialized: usize = 0;
    errdefer {
        for (instructions[0..initialized]) |instruction| {
            if (instruction.inputs.len != 0) allocator.free(instruction.inputs);
            if (instruction.outputs.len != 0) allocator.free(instruction.outputs);
            program_mod.freeDescriptorList(allocator, instruction.operand_descriptors);
            program_mod.freeDescriptorList(allocator, instruction.result_descriptors);
        }
        allocator.free(instructions);
    }
    for (source, instructions) |src, *dst| {
        const inputs = try allocator.dupe(ir.RegionValueId, src.inputs);
        errdefer allocator.free(inputs);
        const outputs = try allocator.dupe(ir.RegionValueId, src.outputs);
        errdefer allocator.free(outputs);
        const operands = try cloneDescriptorList(allocator, src.operand_descriptors);
        errdefer program_mod.freeDescriptorList(allocator, operands);
        const results = try cloneDescriptorList(allocator, src.result_descriptors);
        errdefer program_mod.freeDescriptorList(allocator, results);
        dst.* = .{
            .kind = src.kind,
            .line = src.line,
            .column = src.column,
            .inputs = inputs,
            .outputs = outputs,
            .operand_descriptors = operands,
            .result_descriptors = results,
            .compare_direction = src.compare_direction,
        };
        initialized += 1;
    }
    return instructions;
}

fn deinitProgramSubprograms(allocator: std.mem.Allocator, subprograms: []program_mod.Subprogram) void {
    for (subprograms) |subprogram| {
        program_mod.freeRegionValueList(allocator, subprogram.values);
        program_mod.freeDescriptorList(allocator, subprogram.argument_descriptors);
        program_mod.freeRegionInstructionList(allocator, subprogram.instructions);
        program_mod.freeDescriptorList(allocator, subprogram.return_descriptors);
        if (subprogram.terminator_operands.len != 0) allocator.free(subprogram.terminator_operands);
        program_mod.freeDescriptorList(allocator, subprogram.terminator_operand_descriptors);
    }
    if (subprograms.len != 0) allocator.free(subprograms);
}

fn deinitProgramControlFlows(allocator: std.mem.Allocator, control_flows: []program_mod.ControlFlow) void {
    for (control_flows) |control_flow| {
        if (control_flow.state_inputs.len != 0) allocator.free(control_flow.state_inputs);
        if (control_flow.state_outputs.len != 0) allocator.free(control_flow.state_outputs);
    }
    if (control_flows.len != 0) allocator.free(control_flows);
}

fn programNodeInputs(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction) ![]const ir.ValueId {
    if (instruction.kind != .get_tuple_element or instruction.inputs.len == 0) {
        return allocator.dupe(ir.ValueId, instruction.inputs);
    }
    const tuple_id = instruction.inputs[0];
    if (tuple_id.index >= plan.values.len) return allocator.dupe(ir.ValueId, instruction.inputs);
    const tuple_value = plan.values[tuple_id.index];
    if (tuple_value.storage != .tuple) return allocator.dupe(ir.ValueId, instruction.inputs);
    const tuple_index = instruction.tuple_index orelse return allocator.dupe(ir.ValueId, instruction.inputs);
    if (tuple_index < 0 or tuple_index >= @as(i64, @intCast(tuple_value.elements.len))) {
        return allocator.dupe(ir.ValueId, instruction.inputs);
    }
    return allocator.dupe(ir.ValueId, &.{ tuple_id, tuple_value.elements[@intCast(tuple_index)] });
}

fn buildFusionGroups(
    allocator: std.mem.Allocator,
    groups: []const program_mod.FusionGroup,
    nodes: []const program_mod.Node,
    values: []const program_mod.Value,
    edges: []const program_mod.Edge,
) ![]program_mod.FusionGroup {
    const fusion_groups = try allocator.alloc(program_mod.FusionGroup, groups.len);
    errdefer allocator.free(fusion_groups);
    @memcpy(fusion_groups, groups);
    var initialized_groups: usize = 0;
    errdefer deinitFusionGroups(allocator, fusion_groups[0..initialized_groups]);

    const mark_count = groups.len * values.len;
    const input_marks = try allocator.alloc(bool, mark_count);
    defer allocator.free(input_marks);
    @memset(input_marks, false);
    const output_marks = try allocator.alloc(bool, mark_count);
    defer allocator.free(output_marks);
    @memset(output_marks, false);

    for (nodes, 0..) |node, node_index| {
        const group_id = node.fusion_group orelse continue;
        for (node.inputs) |input_id| {
            if (input_id.index >= values.len) continue;
            if (values[input_id.index].producer_node == null) {
                input_marks[groupMarkIndex(values.len, group_id, input_id.index)] = true;
            }
        }
        for (node.outputs) |output_id| {
            if (output_id.index >= values.len) continue;
            if (values[output_id.index].materialization_boundary != null) {
                output_marks[groupMarkIndex(values.len, group_id, output_id.index)] = true;
            }
        }
        _ = node_index;
    }

    for (edges) |edge| {
        const to_group = nodes[edge.to_node].fusion_group;
        const from_group = nodes[edge.from_node].fusion_group;
        if (to_group) |group_id| {
            if (from_group != group_id) {
                input_marks[groupMarkIndex(values.len, group_id, edge.value_id.index)] = true;
            }
        }
        if (from_group) |group_id| {
            if (to_group != group_id) {
                output_marks[groupMarkIndex(values.len, group_id, edge.value_id.index)] = true;
            }
        }
    }

    for (fusion_groups) |*group| {
        group.node_indices = try fusionGroupNodeIndices(allocator, group.*);
        errdefer allocator.free(group.node_indices);
        group.input_values = try markedValueIds(allocator, values, input_marks[group.id * values.len ..][0..values.len]);
        errdefer allocator.free(group.input_values);
        group.output_values = try markedValueIds(allocator, values, output_marks[group.id * values.len ..][0..values.len]);
        initialized_groups += 1;
    }

    return fusion_groups;
}

fn deinitFusionGroups(allocator: std.mem.Allocator, groups: []program_mod.FusionGroup) void {
    for (groups) |group| {
        if (group.node_indices.len != 0) allocator.free(group.node_indices);
        if (group.input_values.len != 0) allocator.free(group.input_values);
        if (group.output_values.len != 0) allocator.free(group.output_values);
    }
    allocator.free(groups);
}

fn fusionGroupNodeIndices(allocator: std.mem.Allocator, group: program_mod.FusionGroup) ![]const usize {
    if (group.node_count == 0) return allocator.alloc(usize, 0);
    if (group.last_node < group.first_node) return error.CommandSubmissionFailed;
    if (group.last_node - group.first_node + 1 != group.node_count) return error.CommandSubmissionFailed;

    const node_indices = try allocator.alloc(usize, group.node_count);
    var index: usize = 0;
    var node_index = group.first_node;
    while (index < node_indices.len) : ({
        index += 1;
        node_index += 1;
    }) {
        node_indices[index] = node_index;
    }
    return node_indices;
}

fn groupMarkIndex(value_count: usize, group_id: usize, value_index: usize) usize {
    return group_id * value_count + value_index;
}

fn markedValueIds(allocator: std.mem.Allocator, values: []const program_mod.Value, marks: []const bool) ![]const ir.ValueId {
    var count: usize = 0;
    for (marks) |mark| {
        if (mark) count += 1;
    }
    const ids = try allocator.alloc(ir.ValueId, count);
    var index: usize = 0;
    for (values, marks) |value, mark| {
        if (!mark) continue;
        ids[index] = value.value_id;
        index += 1;
    }
    return ids;
}

fn programNodeKind(instruction_kind: ir.PlanInstructionKind) program_mod.NodeKind {
    return switch (instruction_kind) {
        .constant => .constant,
        .copy_arg0 => .parameter,
        .tuple, .get_tuple_element => .structural,
        .reshape, .transpose, .broadcast_in_dim, .slice => .view,
        .add, .subtract, .multiply, .divide, .maximum, .minimum, .power, .atan2, .remainder, .and_, .or_, .xor, .shift_left, .shift_right_arithmetic, .shift_right_logical, .negate, .exp, .expm1, .tanh, .sqrt, .rsqrt, .abs, .cbrt, .ceil, .floor, .log, .log1p, .logistic, .sine, .cosine, .not_, .sign, .is_finite, .round_nearest_afz, .round_nearest_even, .popcnt, .count_leading_zeros, .real, .imag, .compare, .select, .clamp => .elementwise,
        .reduce_sum, .reduce_max, .reduce_min, .reduce_and, .reduce_or, .reduce_window_sum, .reduce_window_max => .reduction,
        .dot_general => .matmul,
        .while_ => .control_flow,
        .sort, .top_k, .gather, .scatter, .dynamic_slice, .dynamic_update_slice, .pad, .reverse, .concatenate, .iota, .partition_id, .convert, .bitcast_convert, .reduce_precision, .rng, .rng_bit_generator, .custom_call, .optimization_barrier => .materialize,
        else => .library_call,
    };
}

fn instructionMaterializes(instruction_kind: ir.PlanInstructionKind) bool {
    return switch (instruction_kind) {
        .tuple, .get_tuple_element, .reshape, .transpose, .broadcast_in_dim, .slice => false,
        else => true,
    };
}

fn programNodeFusible(node_kind: program_mod.NodeKind) bool {
    return switch (node_kind) {
        .view, .elementwise => true,
        else => false,
    };
}

test "mlx metal backend program owns while cond body subprogram descriptors" {
    const allocator = std.testing.allocator;
    const assignment = [_]i32{0};

    const scalar_f32 = ir.BufferDescriptor{
        .element_type = .f32,
        .dims = &.{},
        .device_id = 0,
        .memory_id = 0,
        .shard_index = 0,
    };
    const scalar_pred = ir.BufferDescriptor{
        .element_type = .pred,
        .dims = &.{},
        .device_id = 0,
        .memory_id = 0,
        .shard_index = 0,
    };

    const values = try allocator.alloc(ir.Value, 2);
    errdefer allocator.free(values);
    values[0] = .{
        .id = .{ .index = 0 },
        .role = .parameter,
        .descriptor = scalar_f32,
    };
    values[1] = .{
        .id = .{ .index = 1 },
        .role = .instruction_result,
        .descriptor = scalar_f32,
    };

    var parameter_shardings = try allocator.alloc(ir.ShardingPlan, 1);
    parameter_shardings[0] = .{
        .kind = .replicated,
        .mesh_name = try allocator.dupe(u8, ""),
        .device_assignment = try allocator.dupe(i32, &assignment),
    };
    var output_shardings = try allocator.alloc(ir.ShardingPlan, 1);
    output_shardings[0] = .{
        .kind = .replicated,
        .mesh_name = try allocator.dupe(u8, ""),
        .device_assignment = try allocator.dupe(i32, &assignment),
    };

    const cond_args = try allocator.dupe(ir.BufferDescriptor, &.{scalar_f32});
    const cond_returns = try allocator.dupe(ir.BufferDescriptor, &.{scalar_pred});
    const cond_terminator_ids = try allocator.dupe(ir.RegionValueId, &.{.{ .index = 1 }});
    const cond_terminator = try allocator.dupe(ir.BufferDescriptor, &.{scalar_pred});
    const cond_instruction_operands = try allocator.dupe(ir.BufferDescriptor, &.{ scalar_f32, scalar_f32 });
    const cond_instruction_results = try allocator.dupe(ir.BufferDescriptor, &.{scalar_pred});
    const cond_instruction_inputs = try allocator.dupe(ir.RegionValueId, &.{ .{ .index = 0 }, .{ .index = 0 } });
    const cond_instruction_outputs = try allocator.dupe(ir.RegionValueId, &.{.{ .index = 1 }});
    const cond_instructions = try allocator.dupe(ir.RegionInstruction, &.{.{
        .kind = .compare,
        .inputs = cond_instruction_inputs,
        .outputs = cond_instruction_outputs,
        .operand_descriptors = cond_instruction_operands,
        .result_descriptors = cond_instruction_results,
    }});
    const cond_values = try allocator.dupe(ir.RegionValue, &.{
        .{ .id = .{ .index = 0 }, .role = .argument, .descriptor = scalar_f32 },
        .{ .id = .{ .index = 1 }, .role = .instruction_result, .descriptor = scalar_pred },
    });

    const body_args = try allocator.dupe(ir.BufferDescriptor, &.{scalar_f32});
    const body_returns = try allocator.dupe(ir.BufferDescriptor, &.{scalar_f32});
    const body_terminator_ids = try allocator.dupe(ir.RegionValueId, &.{.{ .index = 1 }});
    const body_terminator = try allocator.dupe(ir.BufferDescriptor, &.{scalar_f32});
    const body_instruction_operands = try allocator.dupe(ir.BufferDescriptor, &.{ scalar_f32, scalar_f32 });
    const body_instruction_results = try allocator.dupe(ir.BufferDescriptor, &.{scalar_f32});
    const body_instruction_inputs = try allocator.dupe(ir.RegionValueId, &.{ .{ .index = 0 }, .{ .index = 0 } });
    const body_instruction_outputs = try allocator.dupe(ir.RegionValueId, &.{.{ .index = 1 }});
    const body_instructions = try allocator.dupe(ir.RegionInstruction, &.{.{
        .kind = .add,
        .inputs = body_instruction_inputs,
        .outputs = body_instruction_outputs,
        .operand_descriptors = body_instruction_operands,
        .result_descriptors = body_instruction_results,
    }});
    const body_values = try allocator.dupe(ir.RegionValue, &.{
        .{ .id = .{ .index = 0 }, .role = .argument, .descriptor = scalar_f32 },
        .{ .id = .{ .index = 1 }, .role = .instruction_result, .descriptor = scalar_f32 },
    });

    var regions = try allocator.alloc(ir.PlanRegion, 2);
    regions[0] = .{
        .id = .{ .index = 0 },
        .parent_instruction_index = 0,
        .kind = .while_cond,
        .values = cond_values,
        .argument_descriptors = cond_args,
        .instructions = cond_instructions,
        .return_descriptors = cond_returns,
        .terminator_operands = cond_terminator_ids,
        .terminator_operand_descriptors = cond_terminator,
    };
    regions[1] = .{
        .id = .{ .index = 1 },
        .parent_instruction_index = 0,
        .kind = .while_body,
        .values = body_values,
        .argument_descriptors = body_args,
        .instructions = body_instructions,
        .return_descriptors = body_returns,
        .terminator_operands = body_terminator_ids,
        .terminator_operand_descriptors = body_terminator,
    };

    var plan = ir.ExecutablePlan{
        .allocator = allocator,
        .options = .{
            .num_replicas = 1,
            .num_partitions = 1,
            .device_assignment = try allocator.dupe(i32, &assignment),
        },
        .values = values,
        .regions = regions,
        .parameter_shardings = parameter_shardings,
        .output_shardings = output_shardings,
        .output_ids = try allocator.dupe(ir.ValueId, &.{.{ .index = 1 }}),
        .instructions = try allocator.dupe(ir.PlanInstruction, &.{.{
            .kind = .while_,
            .inputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 0 }}),
            .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 1 }}),
            .region_ids = try allocator.dupe(ir.RegionId, &.{ .{ .index = 0 }, .{ .index = 1 } }),
        }}),
    };
    defer plan.deinit();

    var program = try build(allocator, &plan, null);
    defer program.deinit();

    try std.testing.expectEqual(@as(usize, 1), program.nodes.len);
    try std.testing.expectEqual(program_mod.NodeKind.control_flow, program.nodes[0].kind);
    try std.testing.expectEqual(@as(usize, 2), program.nodes[0].subprograms.len);
    try std.testing.expectEqual(@as(usize, 2), program.subprograms.len);
    try std.testing.expectEqual(@as(usize, 1), program.control_flows.len);
    try std.testing.expectEqual(@as(?usize, 0), program.nodes[0].control_flow);
    const control_flow = program.control_flows[0];
    try std.testing.expectEqual(program_mod.ControlFlowKind.while_loop, control_flow.kind);
    try std.testing.expectEqual(@as(usize, 0), control_flow.parent_node);
    try std.testing.expectEqual(program.nodes[0].subprograms[0], control_flow.condition_subprogram);
    try std.testing.expectEqual(program.nodes[0].subprograms[1], control_flow.body_subprogram);
    try std.testing.expectEqual(@as(usize, 1), control_flow.state_inputs.len);
    try std.testing.expectEqual(@as(usize, 1), control_flow.state_outputs.len);
    try std.testing.expectEqual(@as(u32, 1), control_flow.predicate_output.index);
    const cond = program.subprograms[program.nodes[0].subprograms[0]];
    const body = program.subprograms[program.nodes[0].subprograms[1]];
    try std.testing.expectEqual(ir.RegionKind.while_cond, cond.kind);
    try std.testing.expectEqual(ir.RegionKind.while_body, body.kind);
    try std.testing.expectEqual(@as(usize, 2), cond.values.len);
    try std.testing.expectEqual(ir.RegionValueRole.argument, cond.values[0].role);
    try std.testing.expectEqual(ir.RegionValueRole.instruction_result, cond.values[1].role);
    try std.testing.expectEqual(@as(usize, 1), cond.instructions.len);
    try std.testing.expectEqual(ir.PlanInstructionKind.compare, cond.instructions[0].kind);
    try std.testing.expectEqual(@as(usize, 2), cond.instructions[0].inputs.len);
    try std.testing.expectEqual(@as(u32, 0), cond.instructions[0].inputs[0].index);
    try std.testing.expectEqual(@as(u32, 1), cond.instructions[0].outputs[0].index);
    try std.testing.expectEqual(@as(u32, 1), cond.terminator_operands[0].index);
    try std.testing.expectEqual(@as(usize, 1), body.instructions.len);
    try std.testing.expectEqual(ir.PlanInstructionKind.add, body.instructions[0].kind);
    try std.testing.expectEqual(@as(u32, 1), body.terminator_operands[0].index);
    try std.testing.expectEqual(ir.BufferType.pred, cond.return_descriptors[0].element_type);
    try std.testing.expectEqual(ir.BufferType.f32, body.return_descriptors[0].element_type);
}
