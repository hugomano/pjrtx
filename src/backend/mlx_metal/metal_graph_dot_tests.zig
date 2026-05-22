const std = @import("std");

const ir = @import("src/compiler/ir");
const dot = @import("metal_graph_dot.zig");
const program_mod = @import("program.zig");
const step_storage = @import("executable_metal_graph_step_storage.zig");

test "metal graph dot emits tiled step for large contracting dimension" {
    const allocator = std.testing.allocator;
    const lhs_dims = [_]i64{ 1, 4096 };
    const rhs_dims = [_]i64{ 14336, 4096 };
    const output_dims = [_]i64{ 1, 14336 };
    const lhs_contract = [_]i64{1};
    const rhs_contract = [_]i64{1};

    var values = [_]ir.Value{
        .{
            .id = .{ .index = 0 },
            .role = .parameter,
            .descriptor = .{ .element_type = .bf16, .dims = &lhs_dims, .device_id = 0, .memory_id = 0, .shard_index = 0 },
        },
        .{
            .id = .{ .index = 1 },
            .role = .parameter,
            .descriptor = .{ .element_type = .bf16, .dims = &rhs_dims, .device_id = 0, .memory_id = 0, .shard_index = 0 },
        },
        .{
            .id = .{ .index = 2 },
            .role = .instruction_result,
            .descriptor = .{ .element_type = .bf16, .dims = &output_dims, .device_id = 0, .memory_id = 0, .shard_index = 0 },
        },
    };
    var parameter_shardings = [_]ir.ShardingPlan{};
    var output_shardings = [_]ir.ShardingPlan{};
    var instructions = [_]ir.PlanInstruction{.{
        .kind = .dot_general,
        .inputs = &.{ .{ .index = 0 }, .{ .index = 1 } },
        .outputs = &.{.{ .index = 2 }},
        .lhs_contracting_dimensions = &lhs_contract,
        .rhs_contracting_dimensions = &rhs_contract,
    }};
    var plan = ir.ExecutablePlan{
        .allocator = allocator,
        .options = .{},
        .values = values[0..],
        .parameter_shardings = parameter_shardings[0..],
        .output_shardings = output_shardings[0..],
        .instructions = instructions[0..],
    };

    const step = (try dot.makeStep(allocator, &plan, instructions[0], 11)) orelse return error.TestUnexpectedResult;
    defer step_storage.StepStorage.deinitStep(allocator, step);
    try std.testing.expectEqual(@as(u32, 256), step.threads_per_threadgroup);
    try std.testing.expectEqual(@as(u64, @divTrunc(14336 + 4 - 1, 4) * 256), step.element_count);
    try std.testing.expect(std.mem.indexOf(u8, step.source, "threadgroup float partials[1024]") != null);
    try std.testing.expect(std.mem.indexOf(u8, step.source, "thread_position_in_threadgroup") != null);
    try std.testing.expect(std.mem.indexOf(u8, step.source, "threadgroup_position_in_grid") != null);
    try std.testing.expect(std.mem.indexOf(u8, step.source, "for (uint i = tid; i < k; i += 256u)") != null);
}

test "metal graph dot emits one tiled group step for adjacent projections" {
    const allocator = std.testing.allocator;
    const lhs_dims = [_]i64{ 1, 4096 };
    const rhs0_dims = [_]i64{ 14336, 4096 };
    const rhs1_dims = [_]i64{ 4096, 4096 };
    const output0_dims = [_]i64{ 1, 14336 };
    const output1_dims = [_]i64{ 1, 4096 };
    const lhs_contract = [_]i64{1};
    const rhs_contract = [_]i64{1};

    var values = [_]ir.Value{
        .{
            .id = .{ .index = 0 },
            .role = .parameter,
            .descriptor = .{ .element_type = .bf16, .dims = &lhs_dims, .device_id = 0, .memory_id = 0, .shard_index = 0 },
        },
        .{
            .id = .{ .index = 1 },
            .role = .parameter,
            .descriptor = .{ .element_type = .bf16, .dims = &rhs0_dims, .device_id = 0, .memory_id = 0, .shard_index = 0 },
        },
        .{
            .id = .{ .index = 2 },
            .role = .parameter,
            .descriptor = .{ .element_type = .bf16, .dims = &rhs1_dims, .device_id = 0, .memory_id = 0, .shard_index = 0 },
        },
        .{
            .id = .{ .index = 3 },
            .role = .instruction_result,
            .descriptor = .{ .element_type = .bf16, .dims = &output0_dims, .device_id = 0, .memory_id = 0, .shard_index = 0 },
        },
        .{
            .id = .{ .index = 4 },
            .role = .instruction_result,
            .descriptor = .{ .element_type = .bf16, .dims = &output1_dims, .device_id = 0, .memory_id = 0, .shard_index = 0 },
        },
    };
    var parameter_shardings = [_]ir.ShardingPlan{};
    var output_shardings = [_]ir.ShardingPlan{};
    var instructions = [_]ir.PlanInstruction{
        .{
            .kind = .dot_general,
            .inputs = &.{ .{ .index = 0 }, .{ .index = 1 } },
            .outputs = &.{.{ .index = 3 }},
            .lhs_contracting_dimensions = &lhs_contract,
            .rhs_contracting_dimensions = &rhs_contract,
        },
        .{
            .kind = .dot_general,
            .inputs = &.{ .{ .index = 0 }, .{ .index = 2 } },
            .outputs = &.{.{ .index = 4 }},
            .lhs_contracting_dimensions = &lhs_contract,
            .rhs_contracting_dimensions = &rhs_contract,
        },
    };
    var plan = ir.ExecutablePlan{
        .allocator = allocator,
        .options = .{},
        .values = values[0..],
        .parameter_shardings = parameter_shardings[0..],
        .output_shardings = output_shardings[0..],
        .instructions = instructions[0..],
    };
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
    var program_values = [_]program_mod.Value{
        .{ .value_id = .{ .index = 0 } },
        .{ .value_id = .{ .index = 1 } },
        .{ .value_id = .{ .index = 2 } },
        .{ .value_id = .{ .index = 3 }, .producer_node = 0, .is_output = true },
        .{ .value_id = .{ .index = 4 }, .producer_node = 1, .is_output = true },
    };
    var schedule = [_]program_mod.ScheduleItem{
        .{ .kind = .node, .index = 0 },
        .{ .kind = .node, .index = 1 },
    };
    var program = program_mod.Program{
        .allocator = allocator,
        .values = program_values[0..],
        .nodes = nodes[0..],
        .edges = &.{},
        .schedule = schedule[0..],
        .fusion_groups = &.{},
        .materialization_boundaries = &.{},
    };

    const step = (try dot.makeGroupStep(allocator, &plan, &program, &.{ 0, 1 })) orelse return error.TestUnexpectedResult;
    defer step_storage.StepStorage.deinitStep(allocator, step);
    try std.testing.expectEqual(@as(usize, 3), step.inputs.len);
    try std.testing.expectEqual(@as(usize, 2), step.outputs.len);
    try std.testing.expectEqual(@as(u32, 256), step.threads_per_threadgroup);
    try std.testing.expectEqual(@as(u64, (@divTrunc(14336 + 4 - 1, 4) + @divTrunc(4096 + 4 - 1, 4)) * 256), step.element_count);
    try std.testing.expect(std.mem.indexOf(u8, step.kernel_name, "tiled_dot_group") != null);
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, step.source, "threadgroup float partials[1024]"));
    try std.testing.expect(std.mem.indexOf(u8, step.source, "device bfloat* out0") != null);
    try std.testing.expect(std.mem.indexOf(u8, step.source, "device bfloat* out1") != null);
}
