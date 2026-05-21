const std = @import("std");
const ir = @import("src/compiler/ir");

const program_mod = @import("program.zig");

/// Counts executable-plan regions that must be cloned into backend subprograms.
pub fn countSubprograms(plan: *const ir.ExecutablePlan) !usize {
    var count: usize = 0;
    for (plan.instructions) |instruction| {
        if (instruction.kind != .while_) continue;
        count += instruction.region_ids.len;
    }
    return count;
}

/// Counts control-flow metadata records needed by the backend program.
pub fn countControlFlows(plan: *const ir.ExecutablePlan) !usize {
    var count: usize = 0;
    for (plan.instructions) |instruction| {
        if (instruction.kind == .while_) count += 1;
    }
    return count;
}

/// Clones backend subprograms owned by one executable-plan instruction.
pub fn buildNodeSubprograms(
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
        subprograms[subprogram_index] = try cloneSubprogram(
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

/// Builds backend control-flow metadata for one executable-plan instruction.
pub fn buildNodeControlFlow(
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

/// Releases partially initialized backend subprogram storage during failed program construction.
pub fn deinitSubprograms(allocator: std.mem.Allocator, subprograms: []program_mod.Subprogram) void {
    for (subprograms) |subprogram| {
        subprogram.deinit(allocator);
    }
    if (subprograms.len != 0) allocator.free(subprograms);
}

/// Releases partially initialized control-flow storage during failed program construction.
pub fn deinitControlFlows(allocator: std.mem.Allocator, control_flows: []program_mod.ControlFlow) void {
    for (control_flows) |control_flow| {
        control_flow.deinit(allocator);
    }
    if (control_flows.len != 0) allocator.free(control_flows);
}

fn cloneSubprogram(
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

test "mlx metal backend program owns while cond body subprogram descriptors" {
    const program_build_mod = @import("program_build.zig");
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

    var program = try program_build_mod.build(allocator, &plan, null);
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
