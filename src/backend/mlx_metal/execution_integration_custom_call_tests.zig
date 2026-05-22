const std = @import("std");
const ir = @import("src/compiler/ir");

const buffer_mod = @import("buffer.zig");
const custom_call_mod = @import("custom_call.zig");
const device_mod = @import("device.zig");
const executable_mod = @import("executable.zig");
const execution_mod = @import("execution.zig");
const program_mod = @import("program.zig");

test "mlx metal backend lowers metadata custom call and optimization barrier on device" {
    const allocator = std.testing.allocator;
    const devices = try device_mod.DeviceList.enumerate(allocator);
    defer device_mod.DeviceList.release(allocator, devices);
    const local_hardware_id = devices[0].local_hardware_id;
    const assignment = [_]i32{local_hardware_id};
    const dims = [_]i64{4};

    const values = try allocator.alloc(ir.Value, 5);
    for (values, 0..) |*value, index| {
        value.* = .{
            .id = .{ .index = @intCast(index) },
            .role = if (index < 2) .parameter else .instruction_result,
            .descriptor = .{
                .element_type = .u8,
                .dims = try allocator.dupe(i64, &dims),
                .device_id = 0,
                .memory_id = 0,
                .shard_index = 0,
            },
        };
    }

    const parameter_shardings = try allocator.alloc(ir.ShardingPlan, 2);
    for (parameter_shardings) |*sharding| {
        sharding.* = .{
            .kind = .replicated,
            .mesh_name = try allocator.dupe(u8, "test"),
            .device_assignment = try allocator.dupe(i32, &assignment),
        };
    }

    var plan = ir.ExecutablePlan{
        .allocator = allocator,
        .options = .{
            .num_replicas = 1,
            .num_partitions = 1,
            .device_assignment = try allocator.dupe(i32, &assignment),
        },
        .values = values,
        .parameter_shardings = parameter_shardings,
        .output_shardings = try allocator.alloc(ir.ShardingPlan, 0),
        .output_ids = try allocator.dupe(ir.ValueId, &.{ .{ .index = 3 }, .{ .index = 4 } }),
        .instructions = try allocator.dupe(ir.PlanInstruction, &.{
            .{
                .kind = .optimization_barrier,
                .inputs = try allocator.dupe(ir.ValueId, &.{ .{ .index = 0 }, .{ .index = 1 } }),
                .outputs = try allocator.dupe(ir.ValueId, &.{ .{ .index = 2 }, .{ .index = 3 } }),
                .dims = try allocator.dupe(i64, &dims),
            },
            .{
                .kind = .custom_call,
                .inputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 2 }}),
                .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 4 }}),
                .dims = try allocator.dupe(i64, &dims),
                .custom_call_target = try allocator.dupe(u8, "annotate_device_placement"),
            },
        }),
    };
    defer plan.deinit();

    const executable = (try executable_mod.compile(allocator, &plan, &assignment, execution_mod.compiledProgramBuildCallback)) orelse return error.TestUnexpectedResult;
    defer execution_mod.destroy(executable);

    const lhs = (try buffer_mod.Opaque.fromHost(local_hardware_id, .u8, &dims, &.{ 1, 2, 3, 4 })) orelse return error.TestUnexpectedResult;
    defer buffer_mod.Opaque.destroy(lhs);
    const rhs = (try buffer_mod.Opaque.fromHost(local_hardware_id, .u8, &dims, &.{ 10, 20, 30, 40 })) orelse return error.TestUnexpectedResult;
    defer buffer_mod.Opaque.destroy(rhs);

    const result = (try execution_mod.execute(allocator, executable, 0, &.{ lhs, rhs })) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(execution_mod.ExecutionCompletionKind.completed, result.completion.kind);
    const outputs = result.outputs;
    defer allocator.free(outputs);
    defer {
        for (outputs) |output| buffer_mod.Opaque.destroy(output.handle);
    }

    try std.testing.expectEqual(@as(usize, 2), outputs.len);
    var barrier_rhs: [4]u8 = undefined;
    var annotated_lhs: [4]u8 = undefined;
    try buffer_mod.Opaque.copyToHost(outputs[0].handle, &barrier_rhs);
    try buffer_mod.Opaque.copyToHost(outputs[1].handle, &annotated_lhs);
    try std.testing.expectEqualSlices(u8, &.{ 10, 20, 30, 40 }, &barrier_rhs);
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4 }, &annotated_lhs);
}

test "mlx metal backend runs registered binary custom call on device buffers" {
    const allocator = std.testing.allocator;
    const local_hardware_id: i32 = 0;
    const assignment = [_]i32{local_hardware_id};
    const dims = [_]i64{2};
    const target = "pjrtx.test.binary_add";

    try custom_call_mod.register(.{
        .target = target,
        .kind = .binary,
        .binary_op = .add,
    });
    defer custom_call_mod.unregister(target);

    const values = try allocator.alloc(ir.Value, 3);
    for (values, 0..) |*value, index| {
        value.* = .{
            .id = .{ .index = @intCast(index) },
            .role = if (index < 2) .parameter else .instruction_result,
            .descriptor = .{
                .element_type = .f32,
                .dims = try allocator.dupe(i64, &dims),
                .device_id = 0,
                .memory_id = 0,
                .shard_index = 0,
            },
        };
    }

    const parameter_shardings = try allocator.alloc(ir.ShardingPlan, 2);
    for (parameter_shardings) |*sharding| {
        sharding.* = .{
            .kind = .replicated,
            .mesh_name = try allocator.dupe(u8, "test"),
            .device_assignment = try allocator.dupe(i32, &assignment),
        };
    }

    var plan = ir.ExecutablePlan{
        .allocator = allocator,
        .options = .{
            .num_replicas = 1,
            .num_partitions = 1,
            .device_assignment = try allocator.dupe(i32, &assignment),
        },
        .values = values,
        .parameter_shardings = parameter_shardings,
        .output_shardings = try allocator.alloc(ir.ShardingPlan, 0),
        .output_ids = try allocator.dupe(ir.ValueId, &.{.{ .index = 2 }}),
        .instructions = try allocator.dupe(ir.PlanInstruction, &.{.{
            .kind = .custom_call,
            .inputs = try allocator.dupe(ir.ValueId, &.{ .{ .index = 0 }, .{ .index = 1 } }),
            .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 2 }}),
            .dims = try allocator.dupe(i64, &dims),
            .custom_call_target = try allocator.dupe(u8, target),
        }}),
    };
    defer plan.deinit();

    const executable = (try executable_mod.compile(allocator, &plan, &assignment, execution_mod.compiledProgramBuildCallback)) orelse return error.TestUnexpectedResult;
    defer execution_mod.destroy(executable);

    const lhs_data = [_]f32{ 1.5, 2.25 };
    const rhs_data = [_]f32{ 4.0, -0.25 };
    const lhs = (try buffer_mod.Opaque.fromHost(local_hardware_id, .f32, &dims, std.mem.asBytes(&lhs_data))) orelse return error.TestUnexpectedResult;
    defer buffer_mod.Opaque.destroy(lhs);
    const rhs = (try buffer_mod.Opaque.fromHost(local_hardware_id, .f32, &dims, std.mem.asBytes(&rhs_data))) orelse return error.TestUnexpectedResult;
    defer buffer_mod.Opaque.destroy(rhs);

    const result = (try execution_mod.execute(allocator, executable, 0, &.{ lhs, rhs })) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(execution_mod.ExecutionCompletionKind.completed, result.completion.kind);
    const outputs = result.outputs;
    defer allocator.free(outputs);
    defer {
        for (outputs) |output| buffer_mod.Opaque.destroy(output.handle);
    }

    var output: [2]f32 = undefined;
    try buffer_mod.Opaque.copyToHost(outputs[0].handle, std.mem.asBytes(&output));
    try std.testing.expectApproxEqAbs(@as(f32, 5.5), output[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), output[1], 0.0001);
}
