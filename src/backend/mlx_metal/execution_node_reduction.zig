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
