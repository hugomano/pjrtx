const std = @import("std");
const ir = @import("src/compiler/ir");

const buffer_mod = @import("buffer.zig");
const custom_call_mod = @import("custom_call.zig");
const device_mod = @import("device.zig");
const executable_mod = @import("executable.zig");
const types = @import("execution_types.zig");
const values_mod = @import("execution_values.zig");

const BufferHandle = types.BufferHandle;
const Error = types.Error;
const ScaledDotProductAttention = custom_call_mod.ScaledDotProductAttention;
const ValueBindings = values_mod.ValueBindings;

/// Executes registered backend custom calls against device-resident arguments.
pub const CustomCallDispatch = struct {
    values: *ValueBindings,

    /// Dispatches one custom-call instruction through the backend registry contract.
    pub fn run(self: CustomCallDispatch, instruction: ir.PlanInstruction) Error!?BufferHandle {
        const value_handles = self.values.handles;
        const target = instruction.custom_call_target orelse return null;
        const spec = custom_call_mod.lookup(target) orelse return null;
        return switch (spec.kind) {
            .identity => blk: {
                if (instruction.inputs.len != 1) return null;
                const input_id = instruction.inputs[0];
                if (input_id.index >= value_handles.len) return error.CommandSubmissionFailed;
                const input = value_handles[input_id.index] orelse return error.CommandSubmissionFailed;
                break :blk (try buffer_mod.Opaque.clone(input)) orelse return null;
            },
            .unary => blk: {
                if (instruction.inputs.len != 1) return null;
                const input_id = instruction.inputs[0];
                if (input_id.index >= value_handles.len) return error.CommandSubmissionFailed;
                const input = value_handles[input_id.index] orelse return error.CommandSubmissionFailed;
                break :blk (try buffer_mod.Opaque.unary(input, spec.unary_op.?)) orelse return null;
            },
            .binary => blk: {
                if (instruction.inputs.len != 2) return null;
                const lhs_id = instruction.inputs[0];
                const rhs_id = instruction.inputs[1];
                if (lhs_id.index >= value_handles.len or rhs_id.index >= value_handles.len) return error.CommandSubmissionFailed;
                const lhs = value_handles[lhs_id.index] orelse return error.CommandSubmissionFailed;
                const rhs = value_handles[rhs_id.index] orelse return error.CommandSubmissionFailed;
                break :blk (try buffer_mod.Opaque.binary(lhs, rhs, spec.binary_op.?)) orelse return null;
            },
            .metal_kernel_binary_add_f32 => blk: {
                if (instruction.inputs.len != 2) return null;
                const lhs_id = instruction.inputs[0];
                const rhs_id = instruction.inputs[1];
                if (lhs_id.index >= value_handles.len or rhs_id.index >= value_handles.len) return error.CommandSubmissionFailed;
                const lhs = value_handles[lhs_id.index] orelse return error.CommandSubmissionFailed;
                const rhs = value_handles[rhs_id.index] orelse return error.CommandSubmissionFailed;
                break :blk (try buffer_mod.Opaque.customBinaryAddF32(lhs, rhs)) orelse return error.CommandSubmissionFailed;
            },
            .scaled_dot_product_attention => blk: {
                if (instruction.inputs.len != ScaledDotProductAttention.InputCount) return null;
                const q_id = instruction.inputs[ScaledDotProductAttention.Input.q.index()];
                const k_id = instruction.inputs[ScaledDotProductAttention.Input.k.index()];
                const v_id = instruction.inputs[ScaledDotProductAttention.Input.v.index()];
                const token_index_id = instruction.inputs[ScaledDotProductAttention.Input.token_index.index()];
                if (q_id.index >= value_handles.len or k_id.index >= value_handles.len or v_id.index >= value_handles.len or token_index_id.index >= value_handles.len) return error.CommandSubmissionFailed;
                const q = value_handles[q_id.index] orelse return error.CommandSubmissionFailed;
                const k = value_handles[k_id.index] orelse return error.CommandSubmissionFailed;
                const v = value_handles[v_id.index] orelse return error.CommandSubmissionFailed;
                const token_index = value_handles[token_index_id.index] orelse return error.CommandSubmissionFailed;
                break :blk (try buffer_mod.Opaque.customScaledDotProductAttention(q, k, v, token_index)) orelse return error.CommandSubmissionFailed;
            },
        };
    }
};
