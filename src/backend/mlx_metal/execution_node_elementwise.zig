const std = @import("std");
const ir = @import("src/compiler/ir");

const buffer_mod = @import("buffer.zig");
const device_mod = @import("device.zig");
const executable_mod = @import("executable.zig");
const lowering_mod = @import("lowering.zig");
const program_mod = @import("program.zig");
const types = @import("execution_types.zig");
const values_mod = @import("execution_values.zig");

const BufferHandle = types.BufferHandle;
const Error = types.Error;
const ValueBindings = values_mod.ValueBindings;

/// Executes elementwise and dtype-transform node forms.
pub const Context = struct {
    values: *ValueBindings,

    /// Returns the device buffer produced by one elementwise instruction.
    pub fn run(self: Context, instruction: ir.PlanInstruction, output_descriptor: ir.BufferDescriptor, output_dims: []const i64) Error!?BufferHandle {
        return switch (instruction.kind) {
            .complex => blk: {
                const real = try self.handle(instruction.inputs[0]);
                const imag = try self.handle(instruction.inputs[1]);
                break :blk (try buffer_mod.Opaque.complex(real, imag, output_dims)) orelse return null;
            },
            .real => blk: {
                const input = try self.handle(instruction.inputs[0]);
                break :blk (try buffer_mod.Opaque.realPart(input, output_dims)) orelse return null;
            },
            .imag => blk: {
                const input = try self.handle(instruction.inputs[0]);
                break :blk (try buffer_mod.Opaque.imagPart(input, output_dims)) orelse return null;
            },
            .convert => blk: {
                const input = try self.handle(instruction.inputs[0]);
                break :blk (try buffer_mod.Opaque.convert(input, output_descriptor.element_type)) orelse return null;
            },
            .bitcast_convert => blk: {
                const input = try self.handle(instruction.inputs[0]);
                break :blk (try buffer_mod.Opaque.bitcast(input, output_descriptor.element_type, output_dims)) orelse return null;
            },
            .add, .subtract, .multiply, .divide, .maximum, .minimum, .power, .atan2, .remainder, .and_, .or_, .xor, .shift_left, .shift_right_arithmetic, .shift_right_logical => blk: {
                const op = lowering_mod.executableBinaryOp(instruction.kind) orelse return null;
                const lhs = try self.handle(instruction.inputs[0]);
                const rhs = try self.handle(instruction.inputs[1]);
                break :blk (try buffer_mod.Opaque.binaryWithOutputDims(lhs, rhs, op, output_dims)) orelse return null;
            },
            .negate, .exp, .expm1, .tanh, .sqrt, .rsqrt, .abs, .cbrt, .ceil, .floor, .log, .log1p, .logistic, .sine, .cosine, .not_, .sign, .is_finite, .round_nearest_afz, .round_nearest_even, .popcnt, .count_leading_zeros => blk: {
                const op = lowering_mod.executableUnaryOp(instruction.kind) orelse return null;
                const input = try self.handle(instruction.inputs[0]);
                break :blk (try buffer_mod.Opaque.unary(input, op)) orelse return null;
            },
            .compare => blk: {
                const lhs = try self.handle(instruction.inputs[0]);
                const rhs = try self.handle(instruction.inputs[1]);
                break :blk (try buffer_mod.Opaque.compare(lhs, rhs, instruction.compare_direction orelse .eq, output_dims)) orelse return null;
            },
            .select => blk: {
                const pred = try self.handle(instruction.inputs[0]);
                const on_true = try self.handle(instruction.inputs[1]);
                const on_false = try self.handle(instruction.inputs[2]);
                break :blk (try buffer_mod.Opaque.select(pred, on_true, on_false, output_dims)) orelse return null;
            },
            .clamp => blk: {
                const min = try self.handle(instruction.inputs[0]);
                const value = try self.handle(instruction.inputs[1]);
                const max = try self.handle(instruction.inputs[2]);
                break :blk (try buffer_mod.Opaque.clamp(min, value, max, output_dims)) orelse return null;
            },
            else => null,
        };
    }

    fn handle(self: Context, value_id: ir.ValueId) Error!BufferHandle {
        if (value_id.index >= self.values.handles.len) return error.CommandSubmissionFailed;
        return self.values.handles[value_id.index] orelse error.CommandSubmissionFailed;
    }
};
