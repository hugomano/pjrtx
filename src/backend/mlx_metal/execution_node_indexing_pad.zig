const ir = @import("src/compiler/ir");

const buffer_mod = @import("buffer.zig");
const types = @import("execution_types.zig");
const values_mod = @import("execution_values.zig");

const BufferHandle = types.BufferHandle;
const Error = types.Error;
const ValueBindings = values_mod.ValueBindings;

/// Runs pad indexing operations over device-resident buffers.
pub fn pad(values: *ValueBindings, instruction: ir.PlanInstruction, output_dims: []const i64) Error!?BufferHandle {
    const input = try handle(values, instruction.inputs[0]);
    const padding_value = try handle(values, instruction.inputs[1]);
    return (try buffer_mod.Opaque.pad(
        input,
        padding_value,
        instruction.edge_padding_low orelse return null,
        instruction.edge_padding_high orelse return null,
        instruction.interior_padding orelse return null,
        output_dims,
    )) orelse return null;
}

fn handle(values: *ValueBindings, value_id: ir.ValueId) Error!BufferHandle {
    if (value_id.index >= values.handles.len) return error.CommandSubmissionFailed;
    return values.handles[value_id.index] orelse error.CommandSubmissionFailed;
}
