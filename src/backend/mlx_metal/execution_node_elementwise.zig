const std = @import("std");
const ir = @import("src/compiler/ir");

const buffer_mod = @import("buffer.zig");
const device_mod = @import("device.zig");
const executable_mod = @import("executable.zig");
const lowering_mod = @import("lowering.zig");
const program_mod = @import("program.zig");
const types = @import("execution_types.zig");
const values_mod = @import("execution_values.zig");

const BufferHandle = types.BufferHandle;
const Error = types.Error;
const ValueBindings = values_mod.ValueBindings;

/// Executes elementwise and dtype-transform node forms.
pub const Context = struct {
    values: *ValueBindings,

    /// Returns the device buffer produced by one elementwise instruction.
    pub fn run(self: Context, instruction: ir.PlanInstruction, output_descriptor: ir.BufferDescriptor, output_dims: []const i64) Error!?BufferHandle {
        return switch (instruction.kind) {
            .complex => blk: {
                const real = try self.handle(instruction.inputs[0]);
                const imag = try self.handle(instruction.inputs[1]);
                break :blk (try buffer_mod.Opaque.complex(real, imag, output_dims)) orelse return null;
            },
            .real => blk: {
                const input = try self.handle(instruction.inputs[0]);
                break :blk (try buffer_mod.Opaque.realPart(input, output_dims)) orelse return null;
            },
            .imag => blk: {
                const input = try self.handle(instruction.inputs[0]);
                break :blk (try buffer_mod.Opaque.imagPart(input, output_dims)) orelse return null;
            },
            .convert => blk: {
                const input = try self.handle(instruction.inputs[0]);
                break :blk (try buffer_mod.Opaque.convert(input, output_descriptor.element_type)) orelse return null;
            },
            .bitcast_convert => blk: {
                const input = try self.handle(instruction.inputs[0]);
                break :blk (try buffer_mod.Opaque.bitcast(input, output_descriptor.element_type, output_dims)) orelse return null;
            },
            .add, .subtract, .multiply, .divide, .maximum, .minimum, .power, .atan2, .remainder, .and_, .or_, .xor, .shift_left, .shift_right_arithmetic, .shift_right_logical => blk: {
                const op = lowering_mod.executableBinaryOp(instruction.kind) orelse return null;
                const lhs = try self.handle(instruction.inputs[0]);
                const rhs = try self.handle(instruction.inputs[1]);
                break :blk (try buffer_mod.Opaque.binaryWithOutputDims(lhs, rhs, op, output_dims)) orelse return null;
            },
            .negate, .exp, .expm1, .tanh, .sqrt, .rsqrt, .abs, .cbrt, .ceil, .floor, .log, .log1p, .logistic, .sine, .cosine, .not_, .sign, .is_finite, .round_nearest_afz, .round_nearest_even, .popcnt, .count_leading_zeros => blk: {
                const op = lowering_mod.executableUnaryOp(instruction.kind) orelse return null;
                const input = try self.handle(instruction.inputs[0]);
                break :blk (try buffer_mod.Opaque.unary(input, op)) orelse return null;
            },
            .compare => blk: {
                const lhs = try self.handle(instruction.inputs[0]);
                const rhs = try self.handle(instruction.inputs[1]);
                break :blk (try buffer_mod.Opaque.compare(lhs, rhs, instruction.compare_direction orelse .eq, output_dims)) orelse return null;
            },
            .select => blk: {
                const pred = try self.handle(instruction.inputs[0]);
                const on_true = try self.handle(instruction.inputs[1]);
                const on_false = try self.handle(instruction.inputs[2]);
                break :blk (try buffer_mod.Opaque.select(pred, on_true, on_false, output_dims)) orelse return null;
            },
            .clamp => blk: {
                const min = try self.handle(instruction.inputs[0]);
                const value = try self.handle(instruction.inputs[1]);
                const max = try self.handle(instruction.inputs[2]);
                break :blk (try buffer_mod.Opaque.clamp(min, value, max, output_dims)) orelse return null;
            },
            else => null,
        };
    }

    fn handle(self: Context, value_id: ir.ValueId) Error!BufferHandle {
        if (value_id.index >= self.values.handles.len) return error.CommandSubmissionFailed;
        return self.values.handles[value_id.index] orelse error.CommandSubmissionFailed;
    }
};

test "mlx metal backend executable bitcast_convert reinterprets resident bytes" {
    const execution_mod = @import("execution.zig");
    const allocator = std.testing.allocator;
    const dims = [_]i64{2};
    const assignment = [_]i32{0};
    const local_hardware_id: i32 = 0;

    const values = try allocator.alloc(ir.Value, 2);
    errdefer allocator.free(values);
    values[0] = .{
        .id = .{ .index = 0 },
        .role = .parameter,
        .descriptor = .{
            .element_type = .u32,
            .dims = try allocator.dupe(i64, &dims),
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
            .dims = try allocator.dupe(i64, &dims),
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
        .instructions = try allocator.dupe(ir.PlanInstruction, &.{
            .{
                .kind = .bitcast_convert,
                .inputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 0 }}),
                .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 1 }}),
            },
        }),
    };
    defer plan.deinit();

    const executable = (try executable_mod.compile(allocator, &plan, &.{local_hardware_id}, execution_mod.compiledProgramBuildCallback)) orelse return error.TestUnexpectedResult;
    defer execution_mod.destroy(executable);
    const compiled = executable_mod.Executable.fromHandle(executable);
    try std.testing.expectEqual(program_mod.NodeKind.materialize, compiled.program.nodes[0].kind);

    const input_bits = [_]u32{ 0x3f800000, 0xc0000000 };
    const input_bytes = std.mem.sliceAsBytes(&input_bits);
    const arg = (try buffer_mod.Buffer.fromHost(local_hardware_id, .u32, &dims, input_bytes)) orelse return error.TestUnexpectedResult;
    defer arg.destroy();
    const result = (try execution_mod.execute(allocator, executable, 0, &.{arg.toHandle()})) orelse return error.TestUnexpectedResult;
    defer {
        for (result.outputs) |output| buffer_mod.Opaque.destroy(output.handle);
        allocator.free(result.outputs);
    }
    try std.testing.expectEqual(@as(usize, 1), result.outputs.len);

    var out = [_]f32{ 0, 0 };
    try buffer_mod.Opaque.copyToHost(result.outputs[0].handle, std.mem.sliceAsBytes(&out));
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, -2.0), out[1], 0.0001);
}

test "mlx metal backend executable lowers clamp with scalar bounds" {
    const execution_mod = @import("execution.zig");
    const allocator = std.testing.allocator;
    const devices = try device_mod.DeviceList.enumerate(allocator);
    defer device_mod.DeviceList.release(allocator, devices);
    const local_hardware_id = devices[0].local_hardware_id;
    const dims = [_]i64{3};
    const assignment = [_]i32{0};
    const min_literal_value: f32 = -1.0;
    const max_literal_value: f32 = 2.0;

    const values = try allocator.alloc(ir.Value, 4);
    errdefer allocator.free(values);
    values[0] = .{
        .id = .{ .index = 0 },
        .role = .constant,
        .descriptor = .{
            .element_type = .f32,
            .dims = try allocator.dupe(i64, &.{}),
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
            .dims = try allocator.dupe(i64, &dims),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };
    values[2] = .{
        .id = .{ .index = 2 },
        .role = .constant,
        .descriptor = .{
            .element_type = .f32,
            .dims = try allocator.dupe(i64, &.{}),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };
    values[3] = .{
        .id = .{ .index = 3 },
        .role = .instruction_result,
        .descriptor = .{
            .element_type = .f32,
            .dims = try allocator.dupe(i64, &dims),
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
        .output_ids = try allocator.dupe(ir.ValueId, &.{.{ .index = 3 }}),
        .instructions = try allocator.dupe(ir.PlanInstruction, &.{
            .{
                .kind = .constant,
                .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 0 }}),
                .literal = try allocator.dupe(u8, std.mem.asBytes(&min_literal_value)),
            },
            .{
                .kind = .constant,
                .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 2 }}),
                .literal = try allocator.dupe(u8, std.mem.asBytes(&max_literal_value)),
            },
            .{
                .kind = .clamp,
                .inputs = try allocator.dupe(ir.ValueId, &.{ .{ .index = 0 }, .{ .index = 1 }, .{ .index = 2 } }),
                .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 3 }}),
                .dims = try allocator.dupe(i64, &dims),
            },
        }),
    };
    defer plan.deinit();

    const executable = (try executable_mod.compile(allocator, &plan, &.{local_hardware_id}, execution_mod.compiledProgramBuildCallback)) orelse return error.TestUnexpectedResult;
    defer execution_mod.destroy(executable);

    const input = [_]f32{ -2.0, 0.5, 3.0 };
    const input_buffer = (try buffer_mod.Opaque.fromHost(local_hardware_id, .f32, &dims, std.mem.asBytes(&input))) orelse return error.TestUnexpectedResult;
    defer buffer_mod.Opaque.destroy(input_buffer);

    const result = (try execution_mod.execute(allocator, executable, 0, &.{input_buffer})) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(execution_mod.ExecutionCompletionKind.completed, result.completion.kind);
    const outputs = result.outputs;
    defer allocator.free(outputs);
    defer {
        for (outputs) |output| buffer_mod.Opaque.destroy(output.handle);
    }

    var actual: [3]f32 = undefined;
    try buffer_mod.Opaque.copyToHost(outputs[0].handle, std.mem.asBytes(&actual));
    try std.testing.expectEqualSlices(f32, &.{ -1.0, 0.5, 2.0 }, &actual);
}
