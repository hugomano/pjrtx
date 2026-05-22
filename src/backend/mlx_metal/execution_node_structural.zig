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
