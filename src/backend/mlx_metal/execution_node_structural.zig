const std = @import("std");
const ir = @import("src/compiler/ir");

const buffer_mod = @import("buffer.zig");
const executable_mod = @import("executable.zig");
const program_mod = @import("program.zig");
const types = @import("execution_types.zig");
const values_mod = @import("execution_values.zig");

const BufferHandle = types.BufferHandle;
const Error = types.Error;
const CompiledExecutable = executable_mod.Executable;
const ValueBindings = values_mod.ValueBindings;

/// Executes structural node forms that do not lower to arithmetic kernels.
pub const Context = struct {
    executable: *CompiledExecutable,
    device_index: usize,
    values: *ValueBindings,

    /// Stores structural outputs directly into the active value table.
    pub fn runStored(self: Context, instruction_index: usize, instruction: ir.PlanInstruction) Error!?void {
        return switch (instruction.kind) {
            .constant => self.constant(instruction_index, instruction),
            .optimization_barrier => self.optimizationBarrier(instruction),
            .tuple => self.tuple(instruction),
            else => null,
        };
    }

    /// Returns the device buffer produced by a single-output structural node.
    pub fn run(self: Context, instruction: ir.PlanInstruction) Error!?BufferHandle {
        return switch (instruction.kind) {
            .copy_arg0, .reduce_precision => self.cloneArg0(instruction),
            .get_tuple_element => self.getTupleElement(instruction),
            else => null,
        };
    }

    fn constant(self: Context, instruction_index: usize, instruction: ir.PlanInstruction) Error!?void {
        if (instruction.outputs.len != 1) return null;
        const output_id = instruction.outputs[0];
        const plan = self.executable.plan;
        const cached = self.executable.constant_handles[executable_mod.constantIndex(plan.instructions.len, self.device_index, instruction_index)] orelse return null;
        try values_mod.storeBorrowedValueHandle(self.values.handles, self.values.owned, output_id, cached);
        self.executable.recordBorrowedConstantNode();
        return {};
    }

    fn optimizationBarrier(self: Context, instruction: ir.PlanInstruction) Error!?void {
        if (instruction.inputs.len != instruction.outputs.len) return null;
        var stored_outputs: usize = 0;
        errdefer {
            for (instruction.outputs[0..stored_outputs]) |output_id| {
                if (output_id.index < self.values.handles.len and self.values.owned[output_id.index]) {
                    if (self.values.handles[output_id.index]) |owned_handle| buffer_mod.Opaque.destroy(owned_handle);
                    self.values.handles[output_id.index] = null;
                    self.values.owned[output_id.index] = false;
                }
            }
        }
        for (instruction.inputs, instruction.outputs) |input_id, output_id| {
            const input = try self.handle(input_id);
            const cloned = (try buffer_mod.Opaque.clone(input)) orelse return null;
            try values_mod.storeOwnedValueHandle(self.values.handles, self.values.owned, output_id, cloned);
            stored_outputs += 1;
        }
        return {};
    }

    fn tuple(self: Context, instruction: ir.PlanInstruction) Error!?void {
        if (instruction.outputs.len != 1) return null;
        const output_id = instruction.outputs[0];
        const plan = self.executable.plan;
        if (output_id.index >= plan.values.len or plan.values[output_id.index].storage != .tuple) return null;
        return {};
    }

    fn cloneArg0(self: Context, instruction: ir.PlanInstruction) Error!?BufferHandle {
        const input = try self.handle(instruction.inputs[0]);
        return (try buffer_mod.Opaque.clone(input)) orelse null;
    }

    fn getTupleElement(self: Context, instruction: ir.PlanInstruction) Error!?BufferHandle {
        if (instruction.inputs.len != 1) return null;
        const plan = self.executable.plan;
        const tuple_id = instruction.inputs[0];
        if (tuple_id.index >= plan.values.len) return error.CommandSubmissionFailed;
        const tuple_value = plan.values[tuple_id.index];
        if (tuple_value.storage != .tuple) return null;
        const tuple_index = instruction.tuple_index orelse return null;
        if (tuple_index < 0 or tuple_index >= @as(i64, @intCast(tuple_value.elements.len))) return null;
        const element_id = tuple_value.elements[@intCast(tuple_index)];
        const element = try self.handle(element_id);
        return (try buffer_mod.Opaque.clone(element)) orelse null;
    }

    fn handle(self: Context, value_id: ir.ValueId) Error!BufferHandle {
        if (value_id.index >= self.values.handles.len) return error.CommandSubmissionFailed;
        return self.values.handles[value_id.index] orelse error.CommandSubmissionFailed;
    }
};

test "mlx metal backend executable lowers tuple get_tuple_element without materializing tuple buffers" {
    const execution_mod = @import("execution.zig");
    const allocator = std.testing.allocator;
    const local_hardware_id: i32 = 0;
    const dims = [_]i64{2};
    const assignment = [_]i32{0};

    const values = try allocator.alloc(ir.Value, 4);
    errdefer allocator.free(values);
    for (values[0..2], 0..) |*value, i| {
        value.* = .{
            .id = .{ .index = @intCast(i) },
            .role = .parameter,
            .descriptor = .{
                .element_type = .f32,
                .dims = try allocator.dupe(i64, &dims),
                .device_id = 0,
                .memory_id = 0,
                .shard_index = 0,
            },
        };
    }
    values[2] = .{
        .id = .{ .index = 2 },
        .role = .instruction_result,
        .descriptor = .{
            .element_type = .invalid,
            .dims = try allocator.dupe(i64, &.{}),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
        .storage = .tuple,
        .elements = try allocator.dupe(ir.ValueId, &.{ .{ .index = 0 }, .{ .index = 1 } }),
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

    const parameter_shardings = try allocator.alloc(ir.ShardingPlan, 2);
    for (parameter_shardings) |*sharding| {
        sharding.* = .{
            .kind = .replicated,
            .mesh_name = try allocator.dupe(u8, ""),
            .device_assignment = try allocator.dupe(i32, &assignment),
        };
    }
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
        .output_ids = try allocator.dupe(ir.ValueId, &.{.{ .index = 3 }}),
        .instructions = try allocator.dupe(ir.PlanInstruction, &.{
            .{
                .kind = .tuple,
                .inputs = try allocator.dupe(ir.ValueId, &.{ .{ .index = 0 }, .{ .index = 1 } }),
                .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 2 }}),
            },
            .{
                .kind = .get_tuple_element,
                .inputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 2 }}),
                .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 3 }}),
                .tuple_index = 1,
            },
        }),
    };
    defer plan.deinit();

    const executable = (try executable_mod.compile(allocator, &plan, &.{local_hardware_id}, execution_mod.compiledProgramBuildCallback)) orelse return error.TestUnexpectedResult;
    defer execution_mod.destroy(executable);
    const compiled = executable_mod.Executable.fromHandle(executable);
    try std.testing.expectEqual(program_mod.NodeKind.structural, compiled.program.nodes[0].kind);
    try std.testing.expectEqual(program_mod.NodeKind.structural, compiled.program.nodes[1].kind);
    try std.testing.expectEqual(@as(usize, 2), compiled.program.nodes[1].inputs.len);
    try std.testing.expectEqual(@as(usize, 1), compiled.program.nodes[1].inputs[1].index);

    const lhs_data = [_]f32{ 1.0, 2.0 };
    const rhs_data = [_]f32{ 3.0, 4.0 };
    const lhs = (try buffer_mod.Buffer.fromHost(local_hardware_id, .f32, &dims, std.mem.asBytes(&lhs_data))) orelse return error.TestUnexpectedResult;
    defer lhs.destroy();
    const rhs = (try buffer_mod.Buffer.fromHost(local_hardware_id, .f32, &dims, std.mem.asBytes(&rhs_data))) orelse return error.TestUnexpectedResult;
    defer rhs.destroy();

    const result = (try execution_mod.execute(allocator, executable, 0, &.{ lhs.toHandle(), rhs.toHandle() })) orelse return error.TestUnexpectedResult;
    const outputs = result.outputs;
    defer allocator.free(outputs);
    defer for (outputs) |output| buffer_mod.Opaque.destroy(output.handle);

    try std.testing.expectEqual(@as(usize, 1), outputs.len);
    var actual: [2]f32 = undefined;
    try buffer_mod.Opaque.copyToHost(outputs[0].handle, std.mem.asBytes(&actual));
    try std.testing.expectEqualSlices(f32, &rhs_data, &actual);
}
