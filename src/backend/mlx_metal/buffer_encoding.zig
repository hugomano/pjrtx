const ir = @import("src/compiler/ir");
const mlx_call = @import("mlx_call.zig");

/// Returns the MLX dtype matching a compiler IR element type, when supported.
pub fn dtype(element_type: ir.BufferType) ?mlx_call.Dtype {
    return switch (element_type) {
        .pred => .pred,
        .s8 => .s8,
        .s32 => .s32,
        .u8 => .u8,
        .u16 => .u16,
        .u32 => .u32,
        .u64 => .u64,
        .f16 => .f16,
        .f32 => .f32,
        .bf16 => .bf16,
        .c64 => .c64,
        else => null,
    };
}

/// Returns the MLX binary opcode matching a compiler IR elementwise operation.
pub fn binaryOp(op: ir.ElementwiseBinaryOp) ?mlx_call.BinaryOp {
    return switch (op) {
        .add => .add,
        .subtract => .subtract,
        .multiply => .multiply,
        .divide => .divide,
        .maximum => .maximum,
        .minimum => .minimum,
        .power => .power,
        .atan2 => .atan2,
        .remainder => .remainder,
        .and_ => .and_,
        .or_ => .or_,
        .xor => .xor,
        .shift_left => .shift_left,
        .shift_right_arithmetic, .shift_right_logical => .shift_right,
    };
}

/// Returns the MLX unary opcode matching a compiler IR elementwise operation.
pub fn unaryOp(op: ir.ElementwiseUnaryOp) ?mlx_call.UnaryOp {
    return switch (op) {
        .negate => .negate,
        .exp => .exp,
        .expm1 => .expm1,
        .tanh => .tanh,
        .sqrt => .sqrt,
        .rsqrt => .rsqrt,
        .abs => .abs,
        .cbrt => .cbrt,
        .ceil => .ceil,
        .floor => .floor,
        .log => .log,
        .log1p => .log1p,
        .logistic => .logistic,
        .sine => .sine,
        .cosine => .cosine,
        .not_ => .not_,
        .sign => .sign,
        .is_finite => .is_finite,
        .round_nearest_afz => .round_nearest_afz,
        .round_nearest_even => .round_nearest_even,
        .popcnt => .popcnt,
        .count_leading_zeros => .count_leading_zeros,
    };
}

/// Returns the MLX reduction opcode matching a compiler IR reduction instruction.
pub fn reduceOp(op: ir.PlanInstructionKind) ?mlx_call.ReduceOp {
    return switch (op) {
        .reduce_sum => .sum,
        .reduce_max => .max,
        .reduce_min => .min,
        .reduce_and => .and_,
        .reduce_or => .or_,
        else => null,
    };
}

/// Returns the MLX reduction opcode matching a compiler IR reduce-window instruction.
pub fn reduceWindowOp(op: ir.PlanInstructionKind) ?mlx_call.ReduceOp {
    return switch (op) {
        .reduce_window_sum => .sum,
        .reduce_window_max => .max,
        else => null,
    };
}

/// Returns the MLX scatter update code matching a compiler IR scatter update kind.
pub fn scatterUpdate(update_kind: ir.ScatterUpdateKind) mlx_call.ScatterUpdate {
    return switch (update_kind) {
        .set => .set,
        .add => .add,
    };
}

/// Returns the MLX compare opcode matching a compiler IR compare direction.
pub fn compareOp(op: ir.CompareOp) mlx_call.CompareOp {
    return switch (op) {
        .eq => .eq,
        .ne => .ne,
        .ge => .ge,
        .gt => .gt,
        .le => .le,
        .lt => .lt,
    };
}

/// Returns the MLX FFT opcode matching a compiler IR FFT kind.
pub fn fftKind(kind: ir.FftKind) mlx_call.FftKind {
    return switch (kind) {
        .fft => .fft,
        .ifft => .ifft,
        .rfft => .rfft,
        .irfft => .irfft,
    };
}

/// Returns the MLX random distribution code matching compiler IR.
pub fn rngDistribution(distribution: ir.RngDistribution) mlx_call.RngDistribution {
    return switch (distribution) {
        .uniform => .uniform,
        .normal => .normal,
    };
}

/// Returns the MLX triangular-solve transpose code matching compiler IR.
pub fn triangularTranspose(transpose_kind: ir.TriangularSolveTranspose) mlx_call.TriangularTranspose {
    return switch (transpose_kind) {
        .no_transpose => .none,
        .transpose, .adjoint => .transpose,
    };
}
