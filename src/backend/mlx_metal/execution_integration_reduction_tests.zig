const std = @import("std");
const ir = @import("src/compiler/ir");

const buffer_mod = @import("buffer.zig");
const custom_call_mod = @import("custom_call.zig");
const device_mod = @import("device.zig");
const executable_mod = @import("executable.zig");
const execution_mod = @import("execution.zig");
const program_mod = @import("program.zig");

test "mlx metal backend executable lowers reduce_window sum on device" {
    const allocator = std.testing.allocator;
    const local_hardware_id: i32 = 0;
    const input_dims = [_]i64{3};
    const output_dims = [_]i64{3};
    const assignment = [_]i32{0};

    const values = try allocator.alloc(ir.Value, 2);
    errdefer allocator.free(values);
    values[0] = .{
        .id = .{ .index = 0 },
        .role = .parameter,
        .descriptor = .{
            .element_type = .f32,
            .dims = try allocator.dupe(i64, &input_dims),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };
    values[1] = .{
        .id = .{ .index = 1 },
        .role = .instruction_result,
        .descriptor = .{
            .element_type = .f32,
            .dims = try allocator.dupe(i64, &output_dims),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
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

    var plan = ir.ExecutablePlan{
        .allocator = allocator,
        .options = .{
            .num_replicas = 1,
            .num_partitions = 1,
            .device_assignment = try allocator.dupe(i32, &assignment),
        },
        .values = values,
        .parameter_shardings = parameter_shardings,
        .output_shardings = output_shardings,
        .output_ids = try allocator.dupe(ir.ValueId, &.{.{ .index = 1 }}),
        .instructions = try allocator.dupe(ir.PlanInstruction, &.{.{
            .kind = .reduce_window_sum,
            .inputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 0 }}),
            .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 1 }}),
            .window_dimensions = try allocator.dupe(i64, &.{2}),
            .window_strides = try allocator.dupe(i64, &.{1}),
            .base_dilations = try allocator.dupe(i64, &.{1}),
            .window_dilations = try allocator.dupe(i64, &.{1}),
            .edge_padding_low = try allocator.dupe(i64, &.{1}),
            .edge_padding_high = try allocator.dupe(i64, &.{0}),
        }}),
    };
    defer plan.deinit();

    const executable = (try executable_mod.compile(allocator, &plan, &.{local_hardware_id}, execution_mod.compiledProgramBuildCallback)) orelse return error.TestUnexpectedResult;
    defer execution_mod.destroy(executable);
    const compiled = executable_mod.Executable.fromHandle(executable);
    try std.testing.expectEqual(program_mod.NodeKind.reduction, compiled.program.nodes[0].kind);

    const input = [_]f32{ 1.5, -2.0, 4.0 };
    const arg = (try buffer_mod.Buffer.fromHost(local_hardware_id, .f32, &input_dims, std.mem.asBytes(&input))) orelse return error.TestUnexpectedResult;
    defer arg.destroy();
    const result = (try execution_mod.execute(allocator, executable, 0, &.{arg.toHandle()})) orelse return error.TestUnexpectedResult;
    defer {
        for (result.outputs) |output| buffer_mod.Opaque.destroy(output.handle);
        allocator.free(result.outputs);
    }
    var out = [_]f32{ 0, 0, 0 };
    try buffer_mod.Opaque.copyToHost(result.outputs[0].handle, std.mem.asBytes(&out));
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), out[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, -0.5), out[1], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out[2], 0.0001);
}
