const ir = @import("src/compiler/ir");

const buffer_mod = @import("buffer.zig");
const lowering_mod = @import("lowering.zig");
const types = @import("execution_types.zig");
const values_mod = @import("execution_values.zig");

const BufferHandle = types.BufferHandle;
const Error = types.Error;
const ValueBindings = values_mod.ValueBindings;

/// Executes StableHLO scatter forms over device-resident buffers.
pub fn run(values: *ValueBindings, instruction: ir.PlanInstruction, output_dims: []const i64) Error!?BufferHandle {
    const operand = try handle(values, instruction.inputs[0]);
    const indices = try handle(values, instruction.inputs[1]);
    const updates = try handle(values, instruction.inputs[2]);
    if (lowering_mod.supportedScatterAxis(instruction)) |scatter_axis| {
        return (try buffer_mod.Opaque.scatterAxis(
            operand,
            indices,
            updates,
            scatter_axis,
            instruction.index_vector_dim orelse 0,
            instruction.scatter_update_kind orelse .set,
            output_dims,
        )) orelse null;
    }
    return (try buffer_mod.Opaque.scatter(
        operand,
        indices,
        updates,
        instruction.scatter_dims_to_operand_dims orelse return null,
        instruction.inserted_window_dims orelse return null,
        instruction.update_window_dims orelse &.{},
        instruction.input_batching_dims orelse &.{},
        instruction.scatter_indices_batching_dims orelse &.{},
        instruction.index_vector_dim orelse 0,
        instruction.scatter_update_kind orelse .set,
        output_dims,
    )) orelse null;
}

fn handle(values: *ValueBindings, value_id: ir.ValueId) Error!BufferHandle {
    if (value_id.index >= values.handles.len) return error.CommandSubmissionFailed;
    return values.handles[value_id.index] orelse error.CommandSubmissionFailed;
}
