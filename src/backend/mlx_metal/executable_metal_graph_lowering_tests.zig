const std = @import("std");

const ir = @import("src/compiler/ir");
const lowering = @import("executable_metal_graph_lowering.zig");
const fusion = @import("executable_metal_graph_lowering_fusion.zig");
const program_mod = @import("program.zig");

test "executable metal graph lowering rejects recomputing expanded in-group broadcasts" {
    const row_dims = [_]i64{32};
    const matrix_dims = [_]i64{ 32, 4096 };
    const broadcast_dims = [_]i64{0};
    var values = [_]ir.Value{
        .{ .id = .{ .index = 0 }, .role = .parameter, .descriptor = .{ .element_type = .f32, .dims = &row_dims, .device_id = 0, .memory_id = 0, .shard_index = 0 } },
        .{ .id = .{ .index = 1 }, .role = .instruction_result, .descriptor = .{ .element_type = .f32, .dims = &row_dims, .device_id = 0, .memory_id = 0, .shard_index = 0 } },
        .{ .id = .{ .index = 2 }, .role = .instruction_result, .descriptor = .{ .element_type = .f32, .dims = &matrix_dims, .device_id = 0, .memory_id = 0, .shard_index = 0 } },
    };
    var instructions = [_]ir.PlanInstruction{
        .{ .kind = .add, .inputs = &.{ .{ .index = 0 }, .{ .index = 0 } }, .outputs = &.{.{ .index = 1 }} },
        .{ .kind = .broadcast_in_dim, .inputs = &.{.{ .index = 1 }}, .outputs = &.{.{ .index = 2 }}, .broadcast_dimensions = &broadcast_dims },
    };
    var plan = ir.ExecutablePlan{
        .allocator = std.testing.allocator,
        .options = .{},
        .values = values[0..],
        .parameter_shardings = &.{},
        .output_shardings = &.{},
        .instructions = instructions[0..],
    };
    var nodes = [_]program_mod.Node{
        .{ .kind = .elementwise, .instruction_index = 0, .inputs = &.{ .{ .index = 0 }, .{ .index = 0 } }, .outputs = &.{.{ .index = 1 }}, .fusion_group = 0 },
        .{ .kind = .view, .instruction_index = 1, .inputs = &.{.{ .index = 1 }}, .outputs = &.{.{ .index = 2 }}, .materializes = false, .fusion_group = 0 },
    };
    var program_values = [_]program_mod.Value{
        .{ .value_id = .{ .index = 0 } },
        .{ .value_id = .{ .index = 1 }, .producer_node = 0, .last_use_node = 1 },
        .{ .value_id = .{ .index = 2 }, .producer_node = 1 },
    };
    const group = program_mod.FusionGroup{
        .id = 0,
        .kind = .view_elementwise,
        .first_node = 0,
        .last_node = 1,
        .node_count = 2,
        .node_indices = &.{ 0, 1 },
    };
    var fusion_groups = [_]program_mod.FusionGroup{group};
    var program = program_mod.Program{
        .allocator = std.testing.allocator,
        .values = program_values[0..],
        .nodes = nodes[0..],
        .edges = &.{},
        .fusion_groups = fusion_groups[0..],
        .materialization_boundaries = &.{},
        .schedule = &.{},
        .subprograms = &.{},
        .control_flows = &.{},
    };

    try std.testing.expect(!fusion.FusionBoundaryPass.init(&plan, &program, group).allows());
}

test "executable metal graph lowering conservative pass accepts same-shape map chains" {
    const allocator = std.testing.allocator;
    const dims = [_]i64{4};
    var plan_values = [_]ir.Value{
        .{ .id = .{ .index = 0 }, .role = .parameter, .descriptor = .{ .element_type = .f32, .dims = &dims, .device_id = 0, .memory_id = 0, .shard_index = 0 } },
        .{ .id = .{ .index = 1 }, .role = .instruction_result, .descriptor = .{ .element_type = .f32, .dims = &dims, .device_id = 0, .memory_id = 0, .shard_index = 0 } },
        .{ .id = .{ .index = 2 }, .role = .instruction_result, .descriptor = .{ .element_type = .f32, .dims = &dims, .device_id = 0, .memory_id = 0, .shard_index = 0 } },
    };
    var instructions = [_]ir.PlanInstruction{
        .{ .kind = .add, .inputs = &.{ .{ .index = 0 }, .{ .index = 0 } }, .outputs = &.{.{ .index = 1 }} },
        .{ .kind = .multiply, .inputs = &.{ .{ .index = 1 }, .{ .index = 0 } }, .outputs = &.{.{ .index = 2 }} },
    };
    var nodes = [_]program_mod.Node{
        .{ .kind = .elementwise, .instruction_index = 0, .inputs = &.{ .{ .index = 0 }, .{ .index = 0 } }, .outputs = &.{.{ .index = 1 }}, .fusion_group = 0 },
        .{ .kind = .elementwise, .instruction_index = 1, .inputs = &.{ .{ .index = 1 }, .{ .index = 0 } }, .outputs = &.{.{ .index = 2 }}, .fusion_group = 0 },
    };
    var program_values = [_]program_mod.Value{
        .{ .value_id = .{ .index = 0 }, .last_use_node = 1 },
        .{ .value_id = .{ .index = 1 }, .producer_node = 0, .last_use_node = 1 },
        .{ .value_id = .{ .index = 2 }, .producer_node = 1, .is_output = true },
    };
    const group = program_mod.FusionGroup{
        .id = 0,
        .kind = .view_elementwise,
        .first_node = 0,
        .last_node = 1,
        .node_count = 2,
        .node_indices = &.{ 0, 1 },
        .input_values = &.{.{ .index = 0 }},
        .output_values = &.{.{ .index = 2 }},
    };
    var fusion_groups = [_]program_mod.FusionGroup{group};
    var program = program_mod.Program{
        .allocator = allocator,
        .values = program_values[0..],
        .nodes = nodes[0..],
        .edges = &.{},
        .fusion_groups = fusion_groups[0..],
        .materialization_boundaries = &.{},
        .schedule = &.{},
        .subprograms = &.{},
        .control_flows = &.{},
    };
    var plan = ir.ExecutablePlan{
        .allocator = allocator,
        .options = .{},
        .values = plan_values[0..],
        .parameter_shardings = &.{},
        .output_shardings = &.{},
        .instructions = instructions[0..],
    };
    try std.testing.expect(fusion.ConservativeFusionPass.init(&plan, &program, group).allows());
}

test "executable metal graph lowering falls back unsupported conservative fusion groups by default" {
    const allocator = std.testing.allocator;
    const dims = [_]i64{4};
    var plan_values = [_]ir.Value{
        .{ .id = .{ .index = 0 }, .role = .parameter, .descriptor = .{ .element_type = .u32, .dims = &dims, .device_id = 0, .memory_id = 0, .shard_index = 0 } },
        .{ .id = .{ .index = 1 }, .role = .instruction_result, .descriptor = .{ .element_type = .u32, .dims = &dims, .device_id = 0, .memory_id = 0, .shard_index = 0 } },
        .{ .id = .{ .index = 2 }, .role = .instruction_result, .descriptor = .{ .element_type = .u32, .dims = &dims, .device_id = 0, .memory_id = 0, .shard_index = 0 } },
    };
    var instructions = [_]ir.PlanInstruction{
        .{ .kind = .and_, .inputs = &.{ .{ .index = 0 }, .{ .index = 0 } }, .outputs = &.{.{ .index = 1 }} },
        .{ .kind = .and_, .inputs = &.{ .{ .index = 1 }, .{ .index = 0 } }, .outputs = &.{.{ .index = 2 }} },
    };
    const nodes = try allocator.alloc(program_mod.Node, 2);
    defer allocator.free(nodes);
    nodes[0] = .{
        .kind = .elementwise,
        .instruction_index = 0,
        .inputs = &.{ .{ .index = 0 }, .{ .index = 0 } },
        .outputs = &.{.{ .index = 1 }},
        .fusion_group = 0,
    };
    nodes[1] = .{
        .kind = .elementwise,
        .instruction_index = 1,
        .inputs = &.{ .{ .index = 1 }, .{ .index = 0 } },
        .outputs = &.{.{ .index = 2 }},
        .fusion_group = 0,
    };
    const values = try allocator.alloc(program_mod.Value, 3);
    defer allocator.free(values);
    values[0] = .{ .value_id = .{ .index = 0 }, .last_use_node = 1 };
    values[1] = .{ .value_id = .{ .index = 1 }, .producer_node = 0, .last_use_node = 1 };
    values[2] = .{ .value_id = .{ .index = 2 }, .producer_node = 1, .is_output = true };
    const node_indices = try allocator.dupe(usize, &.{ 0, 1 });
    defer allocator.free(node_indices);
    const fusion_groups = try allocator.alloc(program_mod.FusionGroup, 1);
    defer allocator.free(fusion_groups);
    fusion_groups[0] = .{
        .id = 0,
        .kind = .view_elementwise,
        .first_node = 0,
        .last_node = 1,
        .node_count = 2,
        .node_indices = node_indices,
        .input_values = &.{.{ .index = 0 }},
        .output_values = &.{.{ .index = 2 }},
    };
    const schedule = try allocator.dupe(program_mod.ScheduleItem, &.{.{
        .kind = .fusion_group,
        .index = 0,
        .count = 2,
    }});
    defer allocator.free(schedule);

    var program = program_mod.Program{
        .allocator = allocator,
        .values = values,
        .nodes = nodes,
        .edges = &.{},
        .fusion_groups = fusion_groups,
        .materialization_boundaries = &.{},
        .schedule = schedule,
        .subprograms = &.{},
        .control_flows = &.{},
    };
    var plan = ir.ExecutablePlan{
        .allocator = allocator,
        .options = .{},
        .values = plan_values[0..],
        .parameter_shardings = &.{},
        .output_shardings = &.{},
        .instructions = instructions[0..],
    };
    var lowered = try lowering.run(allocator, &plan, &program);
    defer lowered.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 2), lowered.steps.len);
    try std.testing.expectEqual(@as(usize, 1), lowered.metrics.fallback_group_count);
    try std.testing.expectEqual(@as(usize, 2), lowered.metrics.planned_map_step_count);
    try std.testing.expectEqual(@as(usize, 2), lowered.metrics.fallback_map_step_count);
    try std.testing.expectEqual(lowering.StepOrigin.fusion_fallback_node, lowered.steps[0].instruction.origin);
    try std.testing.expectEqual(@as(usize, 0), lowered.steps[0].instruction.index);
    try std.testing.expectEqualSlices(u64, &.{}, lowered.steps[0].instruction.release_values);
    try std.testing.expectEqual(lowering.StepOrigin.fusion_fallback_node, lowered.steps[1].instruction.origin);
    try std.testing.expectEqual(@as(usize, 1), lowered.steps[1].instruction.index);
    try std.testing.expectEqualSlices(u64, &.{1}, lowered.steps[1].instruction.release_values);
}

test "executable metal graph lowering coalesces adjacent dot nodes sharing lhs" {
    const allocator = std.testing.allocator;
    var nodes = [_]program_mod.Node{
        .{
            .kind = .matmul,
            .instruction_index = 0,
            .inputs = &.{ .{ .index = 0 }, .{ .index = 1 } },
            .outputs = &.{.{ .index = 3 }},
        },
        .{
            .kind = .matmul,
            .instruction_index = 1,
            .inputs = &.{ .{ .index = 0 }, .{ .index = 2 } },
            .outputs = &.{.{ .index = 4 }},
        },
    };
    var values = [_]program_mod.Value{
        .{ .value_id = .{ .index = 0 } },
        .{ .value_id = .{ .index = 1 } },
        .{ .value_id = .{ .index = 2 } },
        .{ .value_id = .{ .index = 3 }, .producer_node = 0 },
        .{ .value_id = .{ .index = 4 }, .producer_node = 1 },
    };
    var schedule = [_]program_mod.ScheduleItem{
        .{ .kind = .node, .index = 0 },
        .{ .kind = .node, .index = 1 },
    };
    var program = program_mod.Program{
        .allocator = allocator,
        .values = values[0..],
        .nodes = nodes[0..],
        .edges = &.{},
        .schedule = schedule[0..],
        .fusion_groups = &.{},
        .materialization_boundaries = &.{},
    };
    var plan = ir.ExecutablePlan{
        .allocator = allocator,
        .options = .{},
        .values = &.{},
        .parameter_shardings = &.{},
        .output_shardings = &.{},
        .instructions = &.{},
    };

    var lowered = try lowering.run(allocator, &plan, &program);
    defer lowered.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 1), lowered.steps.len);
    try std.testing.expectEqual(@as(usize, 1), lowered.metrics.planned_matmul_step_count);
    try std.testing.expectEqual(@as(usize, 1), lowered.metrics.dot_group_step_count);
    try std.testing.expectEqual(@as(usize, 2), lowered.metrics.dot_group_node_count);
    switch (lowered.steps[0]) {
        .dot_group => |group| {
            try std.testing.expectEqual(@as(usize, 2), group.node_indices.len);
            try std.testing.expectEqual(@as(usize, 0), group.node_indices[0]);
            try std.testing.expectEqual(@as(usize, 1), group.node_indices[1]);
        },
        else => return error.TestUnexpectedResult,
    }
}
