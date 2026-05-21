const std = @import("std");
const ir = @import("src/compiler/ir");

const buffer_mod = @import("buffer.zig");
const executable_mod = @import("executable.zig");
const program_mod = @import("program.zig");
const types = @import("execution_types.zig");
const values_mod = @import("execution_values.zig");

const BufferHandle = types.BufferHandle;
const Error = types.Error;
const ValueBindings = values_mod.ValueBindings;

/// Executes reduction node forms, including multi-output index variants.
pub const Context = struct {
    plan: *const ir.ExecutablePlan,
    values: *ValueBindings,

    /// Stores multi-output reduction results into the active value table.
    pub fn runStored(self: Context, instruction: ir.PlanInstruction) Error!?void {
        return switch (instruction.kind) {
            .reduce_window_max => if (instruction.inputs.len == 2 and instruction.outputs.len == 2) self.reduceWindowMaxWithIndices(instruction) else null,
            .reduce_max => if (instruction.inputs.len == 2 and instruction.outputs.len == 2) self.reduceMaxWithIndices(instruction) else null,
            else => null,
        };
    }

    /// Returns the device buffer produced by one single-output reduction instruction.
    pub fn run(self: Context, instruction: ir.PlanInstruction, output_dims: []const i64) Error!?BufferHandle {
        return switch (instruction.kind) {
            .reduce_sum, .reduce_max, .reduce_min, .reduce_and, .reduce_or => blk: {
                const input = try self.handle(instruction.inputs[0]);
                break :blk (try buffer_mod.Opaque.reduce(input, instruction.kind, instruction.reduce_dimensions orelse &.{}, output_dims)) orelse return null;
            },
            .reduce_window_sum, .reduce_window_max => blk: {
                const input = try self.handle(instruction.inputs[0]);
                break :blk (try buffer_mod.Opaque.reduceWindow(
                    input,
                    instruction.kind,
                    instruction.window_dimensions orelse return null,
                    instruction.window_strides orelse return null,
                    instruction.base_dilations orelse return null,
                    instruction.window_dilations orelse return null,
                    instruction.edge_padding_low orelse return null,
                    instruction.edge_padding_high orelse return null,
                    output_dims,
                )) orelse return null;
            },
            else => null,
        };
    }

    fn reduceWindowMaxWithIndices(self: Context, instruction: ir.PlanInstruction) Error!?void {
        const values_id = instruction.inputs[0];
        const indices_id = instruction.inputs[1];
        const values = try self.handle(values_id);
        const indices = try self.handle(indices_id);
        const values_output_id = instruction.outputs[0];
        const indices_output_id = instruction.outputs[1];
        if (values_output_id.index >= self.plan.values.len or indices_output_id.index >= self.plan.values.len) return error.CommandSubmissionFailed;
        const output_dims = self.plan.values[values_output_id.index].descriptor.dims;
        const result = (try buffer_mod.Opaque.reduceWindowMaxWithIndices(
            values,
            indices,
            instruction.window_dimensions orelse return null,
            instruction.window_strides orelse return null,
            instruction.base_dilations orelse return null,
            instruction.window_dilations orelse return null,
            instruction.edge_padding_low orelse return null,
            instruction.edge_padding_high orelse return null,
            output_dims,
        )) orelse return null;
        errdefer buffer_mod.Opaque.destroy(result.values);
        errdefer buffer_mod.Opaque.destroy(result.indices);
        try values_mod.storeOwnedValueHandle(self.values.handles, self.values.owned, values_output_id, result.values);
        errdefer self.values.owned[values_output_id.index] = false;
        try values_mod.storeOwnedValueHandle(self.values.handles, self.values.owned, indices_output_id, result.indices);
        return {};
    }

    fn reduceMaxWithIndices(self: Context, instruction: ir.PlanInstruction) Error!?void {
        const values_id = instruction.inputs[0];
        const indices_id = instruction.inputs[1];
        const values = try self.handle(values_id);
        const indices = try self.handle(indices_id);
        const values_output_id = instruction.outputs[0];
        const indices_output_id = instruction.outputs[1];
        if (values_output_id.index >= self.plan.values.len or indices_output_id.index >= self.plan.values.len) return error.CommandSubmissionFailed;
        const output_dims = self.plan.values[values_output_id.index].descriptor.dims;
        const result = (try buffer_mod.Opaque.reduceMaxWithIndices(
            values,
            indices,
            instruction.reduce_dimensions orelse return null,
            output_dims,
        )) orelse return null;
        errdefer buffer_mod.Opaque.destroy(result.values);
        errdefer buffer_mod.Opaque.destroy(result.indices);
        try values_mod.storeOwnedValueHandle(self.values.handles, self.values.owned, values_output_id, result.values);
        errdefer self.values.owned[values_output_id.index] = false;
        try values_mod.storeOwnedValueHandle(self.values.handles, self.values.owned, indices_output_id, result.indices);
        return {};
    }

    fn handle(self: Context, value_id: ir.ValueId) Error!BufferHandle {
        if (value_id.index >= self.values.handles.len) return error.CommandSubmissionFailed;
        return self.values.handles[value_id.index] orelse error.CommandSubmissionFailed;
    }
};

test "mlx metal backend executable lowers reduce_window sum on device" {
    const execution_mod = @import("execution.zig");
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
