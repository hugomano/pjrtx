const std = @import("std");
const ir = @import("src/compiler/ir");

const buffer_mod = @import("buffer.zig");
const device_mod = @import("device.zig");
const executable_mod = @import("executable.zig");
const types = @import("execution_types.zig");
const values_mod = @import("execution_values.zig");

const BufferHandle = types.BufferHandle;
const Error = types.Error;
const CompiledExecutable = executable_mod.Executable;
const ValueBindings = values_mod.ValueBindings;

/// Executes device-side value generation node forms.
pub const Context = struct {
    executable: *CompiledExecutable,
    device_index: usize,
    values: *ValueBindings,

    /// Stores multi-output generated values into the active value table.
    pub fn runStored(self: Context, instruction: ir.PlanInstruction) Error!?void {
        if (instruction.kind == .rng_bit_generator and instruction.inputs.len == 1 and instruction.outputs.len == 2) {
            return self.rngBitGenerator(instruction);
        }
        return null;
    }

    /// Returns the device buffer produced by one generation instruction.
    pub fn run(self: Context, instruction: ir.PlanInstruction, output_descriptor: ir.BufferDescriptor, output_dims: []const i64) Error!?BufferHandle {
        return switch (instruction.kind) {
            .iota => (try buffer_mod.Opaque.iota(
                self.executable.device_local_hardware_ids[self.device_index],
                output_descriptor.element_type,
                output_dims,
                instruction.iota_dimension orelse return null,
            )) orelse null,
            .partition_id => blk: {
                const plan = self.executable.plan;
                const partition_count = if (plan.options.num_partitions <= 0) 1 else plan.options.num_partitions;
                const partition_id_value: u32 = @intCast(self.device_index % @as(usize, @intCast(partition_count)));
                break :blk (try buffer_mod.Opaque.partitionId(
                    self.executable.device_local_hardware_ids[self.device_index],
                    output_descriptor.element_type,
                    partition_id_value,
                )) orelse return null;
            },
            .rng => blk: {
                const a = try self.handle(instruction.inputs[0]);
                const b = try self.handle(instruction.inputs[1]);
                break :blk (try buffer_mod.Opaque.rng(
                    a,
                    b,
                    instruction.rng_distribution orelse return null,
                    output_descriptor.element_type,
                    output_dims,
                )) orelse return null;
            },
            else => null,
        };
    }

    fn rngBitGenerator(self: Context, instruction: ir.PlanInstruction) Error!?void {
        const state_id = instruction.inputs[0];
        const state = try self.handle(state_id);
        const state_output_id = instruction.outputs[0];
        const bits_output_id = instruction.outputs[1];
        const plan = self.executable.plan;
        if (state_output_id.index >= plan.values.len or bits_output_id.index >= plan.values.len) return error.CommandSubmissionFailed;
        const bits_descriptor = plan.values[bits_output_id.index].descriptor;
        const result = (try buffer_mod.Opaque.rngBitGenerator(
            state,
            bits_descriptor.element_type,
            bits_descriptor.dims,
        )) orelse return null;
        errdefer buffer_mod.Opaque.destroy(result.state);
        errdefer buffer_mod.Opaque.destroy(result.bits);
        try values_mod.storeOwnedValueHandle(self.values.handles, self.values.owned, state_output_id, result.state);
        errdefer self.values.owned[state_output_id.index] = false;
        try values_mod.storeOwnedValueHandle(self.values.handles, self.values.owned, bits_output_id, result.bits);
        return {};
    }

    fn handle(self: Context, value_id: ir.ValueId) Error!BufferHandle {
        if (value_id.index >= self.values.handles.len) return error.CommandSubmissionFailed;
        return self.values.handles[value_id.index] orelse error.CommandSubmissionFailed;
    }
};
