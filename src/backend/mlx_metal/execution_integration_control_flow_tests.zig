const std = @import("std");
const ir = @import("src/compiler/ir");

const buffer_mod = @import("buffer.zig");
const custom_call_mod = @import("custom_call.zig");
const device_mod = @import("device.zig");
const executable_mod = @import("executable.zig");
const execution_mod = @import("execution.zig");
const program_mod = @import("program.zig");

test "mlx metal backend executes f32 lt/add while loop on device" {
    const allocator = std.testing.allocator;
    const local_hardware_id: i32 = 0;
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
    values[0] = .{ .id = .{ .index = 0 }, .role = .parameter, .descriptor = scalar_f32 };
    values[1] = .{ .id = .{ .index = 1 }, .role = .instruction_result, .descriptor = scalar_f32 };

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

    var limit_value: f32 = 4.0;
    var step_value: f32 = 1.0;
    const cond_values = try allocator.dupe(ir.RegionValue, &.{
        .{ .id = .{ .index = 0 }, .role = .argument, .descriptor = scalar_f32 },
        .{ .id = .{ .index = 1 }, .role = .constant, .descriptor = scalar_f32, .literal = try allocator.dupe(u8, std.mem.asBytes(&limit_value)) },
        .{ .id = .{ .index = 2 }, .role = .instruction_result, .descriptor = scalar_pred },
    });
    const body_values = try allocator.dupe(ir.RegionValue, &.{
        .{ .id = .{ .index = 0 }, .role = .argument, .descriptor = scalar_f32 },
        .{ .id = .{ .index = 1 }, .role = .constant, .descriptor = scalar_f32, .literal = try allocator.dupe(u8, std.mem.asBytes(&step_value)) },
        .{ .id = .{ .index = 2 }, .role = .instruction_result, .descriptor = scalar_f32 },
    });

    var regions = try allocator.alloc(ir.PlanRegion, 2);
    regions[0] = .{
        .id = .{ .index = 0 },
        .parent_instruction_index = 0,
        .kind = .while_cond,
        .values = cond_values,
        .argument_descriptors = try allocator.dupe(ir.BufferDescriptor, &.{scalar_f32}),
        .instructions = try allocator.dupe(ir.RegionInstruction, &.{.{
            .kind = .compare,
            .inputs = try allocator.dupe(ir.RegionValueId, &.{ .{ .index = 0 }, .{ .index = 1 } }),
            .outputs = try allocator.dupe(ir.RegionValueId, &.{.{ .index = 2 }}),
            .operand_descriptors = try allocator.dupe(ir.BufferDescriptor, &.{ scalar_f32, scalar_f32 }),
            .result_descriptors = try allocator.dupe(ir.BufferDescriptor, &.{scalar_pred}),
            .compare_direction = .lt,
        }}),
        .return_descriptors = try allocator.dupe(ir.BufferDescriptor, &.{scalar_pred}),
        .terminator_operands = try allocator.dupe(ir.RegionValueId, &.{.{ .index = 2 }}),
        .terminator_operand_descriptors = try allocator.dupe(ir.BufferDescriptor, &.{scalar_pred}),
    };
    regions[1] = .{
        .id = .{ .index = 1 },
        .parent_instruction_index = 0,
        .kind = .while_body,
        .values = body_values,
        .argument_descriptors = try allocator.dupe(ir.BufferDescriptor, &.{scalar_f32}),
        .instructions = try allocator.dupe(ir.RegionInstruction, &.{.{
            .kind = .add,
            .inputs = try allocator.dupe(ir.RegionValueId, &.{ .{ .index = 0 }, .{ .index = 1 } }),
            .outputs = try allocator.dupe(ir.RegionValueId, &.{.{ .index = 2 }}),
            .operand_descriptors = try allocator.dupe(ir.BufferDescriptor, &.{ scalar_f32, scalar_f32 }),
            .result_descriptors = try allocator.dupe(ir.BufferDescriptor, &.{scalar_f32}),
        }}),
        .return_descriptors = try allocator.dupe(ir.BufferDescriptor, &.{scalar_f32}),
        .terminator_operands = try allocator.dupe(ir.RegionValueId, &.{.{ .index = 2 }}),
        .terminator_operand_descriptors = try allocator.dupe(ir.BufferDescriptor, &.{scalar_f32}),
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

    const executable = (try executable_mod.compile(allocator, &plan, &.{local_hardware_id}, execution_mod.compiledProgramBuildCallback)) orelse return error.TestUnexpectedResult;
    defer execution_mod.destroy(executable);
    var state: f32 = 0.0;
    const state_buffer = (try buffer_mod.Buffer.fromHost(local_hardware_id, .f32, &.{}, std.mem.asBytes(&state))) orelse return error.TestUnexpectedResult;
    defer state_buffer.destroy();
    const result = (try execution_mod.execute(allocator, executable, 0, &.{state_buffer.toHandle()})) orelse return error.TestUnexpectedResult;
    defer {
        for (result.outputs) |output| buffer_mod.Opaque.destroy(output.handle);
        allocator.free(result.outputs);
    }
    var actual: f32 = 0.0;
    try buffer_mod.Opaque.copyToHost(result.outputs[0].handle, std.mem.asBytes(&actual));
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), actual, 0.0001);
}
