const std = @import("std");
const ir = @import("src/compiler/ir");

const buffer_mod = @import("buffer.zig");
const custom_call_mod = @import("custom_call.zig");
const device_mod = @import("device.zig");
const executable_mod = @import("executable.zig");
const execution_mod = @import("execution.zig");
const program_mod = @import("program.zig");

test "mlx metal backend executable materializes iota on device" {
    const allocator = std.testing.allocator;
    const devices = try device_mod.DeviceList.enumerate(allocator);
    defer device_mod.DeviceList.release(allocator, devices);
    const local_hardware_id = devices[0].local_hardware_id;
    const dims = [_]i64{ 2, 3 };
    const assignment = [_]i32{0};

    const values = try allocator.alloc(ir.Value, 1);
    errdefer allocator.free(values);
    values[0] = .{
        .id = .{ .index = 0 },
        .role = .instruction_result,
        .descriptor = .{
            .element_type = .f32,
            .dims = try allocator.dupe(i64, &dims),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };

    const parameter_shardings = try allocator.alloc(ir.ShardingPlan, 0);
    const output_shardings = try allocator.alloc(ir.ShardingPlan, 1);
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
        .output_ids = try allocator.dupe(ir.ValueId, &.{.{ .index = 0 }}),
        .instructions = try allocator.dupe(ir.PlanInstruction, &.{.{
            .kind = .iota,
            .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 0 }}),
            .dims = try allocator.dupe(i64, &dims),
            .iota_dimension = 1,
        }}),
    };
    defer plan.deinit();

    const executable = (try executable_mod.compile(allocator, &plan, &.{local_hardware_id}, execution_mod.compiledProgramBuildCallback)) orelse return error.TestUnexpectedResult;
    defer execution_mod.destroy(executable);

    const result = (try execution_mod.execute(allocator, executable, 0, &.{})) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(execution_mod.ExecutionCompletionKind.completed, result.completion.kind);
    const outputs = result.outputs;
    defer allocator.free(outputs);
    defer {
        for (outputs) |output| buffer_mod.Opaque.destroy(output.handle);
    }

    try std.testing.expectEqual(@as(usize, 1), outputs.len);
    try std.testing.expectEqual(ir.BufferType.f32, outputs[0].element_type);
    try std.testing.expectEqualSlices(i64, &dims, outputs[0].dims);
    var actual: [6]f32 = undefined;
    try buffer_mod.Opaque.copyToHost(outputs[0].handle, std.mem.asBytes(&actual));
    try std.testing.expectEqualSlices(f32, &.{ 0.0, 1.0, 2.0, 0.0, 1.0, 2.0 }, &actual);
}

test "mlx metal backend executable materializes partition_id on device" {
    const allocator = std.testing.allocator;
    const devices = try device_mod.DeviceList.enumerate(allocator);
    defer device_mod.DeviceList.release(allocator, devices);
    const local_hardware_id = devices[0].local_hardware_id;
    const assignment = [_]i32{0};

    const values = try allocator.alloc(ir.Value, 1);
    errdefer allocator.free(values);
    values[0] = .{
        .id = .{ .index = 0 },
        .role = .instruction_result,
        .descriptor = .{
            .element_type = .u32,
            .dims = try allocator.alloc(i64, 0),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };

    const parameter_shardings = try allocator.alloc(ir.ShardingPlan, 0);
    const output_shardings = try allocator.alloc(ir.ShardingPlan, 1);
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
        .output_ids = try allocator.dupe(ir.ValueId, &.{.{ .index = 0 }}),
        .instructions = try allocator.dupe(ir.PlanInstruction, &.{.{
            .kind = .partition_id,
            .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 0 }}),
        }}),
    };
    defer plan.deinit();

    const executable = (try executable_mod.compile(allocator, &plan, &.{local_hardware_id}, execution_mod.compiledProgramBuildCallback)) orelse return error.TestUnexpectedResult;
    defer execution_mod.destroy(executable);

    const result = (try execution_mod.execute(allocator, executable, 0, &.{})) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(execution_mod.ExecutionCompletionKind.completed, result.completion.kind);
    const outputs = result.outputs;
    defer allocator.free(outputs);
    defer {
        for (outputs) |output| buffer_mod.Opaque.destroy(output.handle);
    }

    try std.testing.expectEqual(@as(usize, 1), outputs.len);
    try std.testing.expectEqual(ir.BufferType.u32, outputs[0].element_type);
    try std.testing.expectEqual(@as(usize, 0), outputs[0].dims.len);
    var actual: u32 = undefined;
    try buffer_mod.Opaque.copyToHost(outputs[0].handle, std.mem.asBytes(&actual));
    try std.testing.expectEqual(@as(u32, 0), actual);
}

test "mlx metal backend executable lowers deprecated rng on device" {
    const allocator = std.testing.allocator;
    const devices = try device_mod.DeviceList.enumerate(allocator);
    defer device_mod.DeviceList.release(allocator, devices);
    const local_hardware_id = devices[0].local_hardware_id;
    const output_dims = [_]i64{ 2, 4 };
    const assignment = [_]i32{0};

    const values = try allocator.alloc(ir.Value, 3);
    errdefer allocator.free(values);
    values[0] = .{
        .id = .{ .index = 0 },
        .role = .parameter,
        .descriptor = .{
            .element_type = .f32,
            .dims = try allocator.alloc(i64, 0),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };
    values[1] = .{
        .id = .{ .index = 1 },
        .role = .parameter,
        .descriptor = .{
            .element_type = .f32,
            .dims = try allocator.alloc(i64, 0),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };
    values[2] = .{
        .id = .{ .index = 2 },
        .role = .instruction_result,
        .descriptor = .{
            .element_type = .f32,
            .dims = try allocator.dupe(i64, &output_dims),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };

    var parameter_shardings = try allocator.alloc(ir.ShardingPlan, 2);
    parameter_shardings[0] = .{
        .kind = .replicated,
        .mesh_name = try allocator.dupe(u8, ""),
        .device_assignment = try allocator.dupe(i32, &assignment),
    };
    parameter_shardings[1] = .{
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
        .output_ids = try allocator.dupe(ir.ValueId, &.{.{ .index = 2 }}),
        .instructions = try allocator.dupe(ir.PlanInstruction, &.{.{
            .kind = .rng,
            .inputs = try allocator.dupe(ir.ValueId, &.{ .{ .index = 0 }, .{ .index = 1 } }),
            .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 2 }}),
            .dims = try allocator.dupe(i64, &output_dims),
            .rng_distribution = .normal,
        }}),
    };
    defer plan.deinit();

    const executable = (try executable_mod.compile(allocator, &plan, &.{local_hardware_id}, execution_mod.compiledProgramBuildCallback)) orelse return error.TestUnexpectedResult;
    defer execution_mod.destroy(executable);

    var mean: f32 = 0.0;
    var scale: f32 = 1.0;
    const mean_buffer = (try buffer_mod.Opaque.fromHost(local_hardware_id, .f32, &.{}, std.mem.asBytes(&mean))) orelse return error.TestUnexpectedResult;
    defer buffer_mod.Opaque.destroy(mean_buffer);
    const scale_buffer = (try buffer_mod.Opaque.fromHost(local_hardware_id, .f32, &.{}, std.mem.asBytes(&scale))) orelse return error.TestUnexpectedResult;
    defer buffer_mod.Opaque.destroy(scale_buffer);

    const result = (try execution_mod.execute(allocator, executable, 0, &.{ mean_buffer, scale_buffer })) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(execution_mod.ExecutionCompletionKind.completed, result.completion.kind);
    const outputs = result.outputs;
    defer allocator.free(outputs);
    defer {
        for (outputs) |output| buffer_mod.Opaque.destroy(output.handle);
    }

    try std.testing.expectEqual(@as(usize, 1), outputs.len);
    try std.testing.expectEqual(ir.BufferType.f32, outputs[0].element_type);
    try std.testing.expectEqualSlices(i64, &output_dims, outputs[0].dims);
    var actual: [8]f32 = undefined;
    try buffer_mod.Opaque.copyToHost(outputs[0].handle, std.mem.asBytes(&actual));
    for (actual) |value| try std.testing.expect(std.math.isFinite(value));
}
