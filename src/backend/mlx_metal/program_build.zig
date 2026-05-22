const std = @import("std");

const ir = @import("src/compiler/ir");
const program_fusion = @import("program_fusion.zig");
const program_mod = @import("program.zig");
const program_region = @import("program_region.zig");
const program_schedule_build = @import("program_schedule_build.zig");

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
    const subprogram_capacity = try program_region.countSubprograms(plan);
    const subprograms = try allocator.alloc(program_mod.Subprogram, subprogram_capacity);
    var initialized_subprograms: usize = 0;
    var subprograms_owned = true;
    errdefer if (subprograms_owned) program_region.deinitSubprograms(allocator, subprograms[0..initialized_subprograms]);
    const control_flow_capacity = try program_region.countControlFlows(plan);
    const control_flows = try allocator.alloc(program_mod.ControlFlow, control_flow_capacity);
    var initialized_control_flows: usize = 0;
    var control_flows_owned = true;
    errdefer if (control_flows_owned) program_region.deinitControlFlows(allocator, control_flows[0..initialized_control_flows]);

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
        const node_materializes = instructionMaterializes(instruction.kind);
        const fusion_group = if (programNodeFusible(instruction.kind, node_kind, node_materializes)) group: {
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
        const node_subprograms = try program_region.buildNodeSubprograms(allocator, plan, instruction, instruction_index, subprograms, &initialized_subprograms);
        var node_subprograms_owned = true;
        errdefer if (node_subprograms_owned and node_subprograms.len != 0) allocator.free(node_subprograms);
        const node_control_flow = try program_region.buildNodeControlFlow(
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
            .materializes = node_materializes,
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

    const fusion_groups = try program_fusion.build(allocator, max_fusion_groups[0..fusion_group_count], nodes, values, edges);
    errdefer program_fusion.deinit(allocator, fusion_groups);

    const schedule = try program_schedule_build.build(allocator, nodes, fusion_groups, materialization_boundaries);
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

fn programNodeFusible(instruction_kind: ir.PlanInstructionKind, node_kind: program_mod.NodeKind, materializes: bool) bool {
    return switch (node_kind) {
        .view, .elementwise => true,
        .structural => !materializes,
        else => instructionFusesWithViewElementwiseChain(instruction_kind),
    };
}

fn instructionFusesWithViewElementwiseChain(instruction_kind: ir.PlanInstructionKind) bool {
    return switch (instruction_kind) {
        .copy_arg0,
        .convert,
        .bitcast_convert,
        .reduce_precision,
        => true,
        else => false,
    };
}
