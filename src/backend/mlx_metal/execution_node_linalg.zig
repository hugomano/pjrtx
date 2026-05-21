const ir = @import("src/compiler/ir");

const buffer_mod = @import("buffer.zig");
const types = @import("execution_types.zig");
const values_mod = @import("execution_values.zig");

const BufferHandle = types.BufferHandle;
const Error = types.Error;
const ValueBindings = values_mod.ValueBindings;

/// Executes linear algebra and transform node forms.
pub const Context = struct {
    values: *ValueBindings,

    /// Returns the device buffer produced by one linalg instruction.
    pub fn run(self: Context, instruction: ir.PlanInstruction, output_dims: []const i64) Error!?BufferHandle {
        return switch (instruction.kind) {
            .dot_general => blk: {
                const lhs = try self.handle(instruction.inputs[0]);
                const rhs = try self.handle(instruction.inputs[1]);
                break :blk (try buffer_mod.Opaque.dotGeneral(
                    lhs,
                    rhs,
                    instruction.lhs_batch_dimensions orelse &.{},
                    instruction.rhs_batch_dimensions orelse &.{},
                    instruction.lhs_contracting_dimensions orelse &.{},
                    instruction.rhs_contracting_dimensions orelse &.{},
                    output_dims,
                )) orelse return null;
            },
            .convolution => blk: {
                const lhs = try self.handle(instruction.inputs[0]);
                const rhs = try self.handle(instruction.inputs[1]);
                break :blk (try buffer_mod.Opaque.convolution(
                    lhs,
                    rhs,
                    instruction.window_strides orelse return null,
                    instruction.edge_padding_low orelse return null,
                    instruction.edge_padding_high orelse return null,
                    instruction.base_dilations orelse return null,
                    instruction.window_dilations orelse return null,
                    instruction.window_reversal orelse return null,
                    instruction.feature_group_count orelse 1,
                    output_dims,
                )) orelse return null;
            },
            .cholesky => blk: {
                const input = try self.handle(instruction.inputs[0]);
                break :blk (try buffer_mod.Opaque.cholesky(input, instruction.lower orelse true, output_dims)) orelse return null;
            },
            .triangular_solve => blk: {
                const a = try self.handle(instruction.inputs[0]);
                const b = try self.handle(instruction.inputs[1]);
                break :blk (try buffer_mod.Opaque.triangularSolve(
                    a,
                    b,
                    instruction.triangular_left_side orelse true,
                    instruction.triangular_lower orelse true,
                    instruction.triangular_unit_diagonal orelse false,
                    instruction.triangular_transpose orelse .no_transpose,
                    output_dims,
                )) orelse return null;
            },
            .fft => blk: {
                const input = try self.handle(instruction.inputs[0]);
                break :blk (try buffer_mod.Opaque.fft(input, instruction.fft_kind orelse return null, instruction.dimensions orelse return null, output_dims)) orelse return null;
            },
            else => null,
        };
    }

    fn handle(self: Context, value_id: ir.ValueId) Error!BufferHandle {
        if (value_id.index >= self.values.handles.len) return error.CommandSubmissionFailed;
        return self.values.handles[value_id.index] orelse error.CommandSubmissionFailed;
    }
};
