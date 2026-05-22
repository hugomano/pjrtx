const ir = @import("src/compiler/ir");

const buffer_mod = @import("buffer.zig");
const types = @import("execution_types.zig");
const values_mod = @import("execution_values.zig");

const BufferHandle = types.BufferHandle;
const Error = types.Error;
const ValueBindings = values_mod.ValueBindings;

/// Runs gather-style indexing operations over device-resident buffers.
pub fn gather(values: *ValueBindings, instruction: ir.PlanInstruction, output_dims: []const i64) Error!?BufferHandle {
    const operand = try handle(values, instruction.inputs[0]);
    const indices = try handle(values, instruction.inputs[1]);
    return (try buffer_mod.Opaque.gather(
        operand,
        indices,
        instruction.start_index_map orelse return null,
        instruction.collapsed_slice_dims orelse &.{},
        instruction.operand_batching_dims orelse &.{},
        instruction.start_indices_batching_dims orelse &.{},
        instruction.index_vector_dim orelse 0,
        instruction.slice_sizes orelse return null,
        instruction.offset_dims orelse &.{},
        output_dims,
    )) orelse return null;
}

fn handle(values: *ValueBindings, value_id: ir.ValueId) Error!BufferHandle {
    if (value_id.index >= values.handles.len) return error.CommandSubmissionFailed;
    return values.handles[value_id.index] orelse error.CommandSubmissionFailed;
}
