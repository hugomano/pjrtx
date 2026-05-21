const ir = @import("src/compiler/ir");

const buffer_mod = @import("buffer.zig");
const encoding = @import("buffer_encoding.zig");
const mlx_call = @import("mlx_call.zig");

const Buffer = buffer_mod.Buffer;
const Error = buffer_mod.Error;

/// Runs dot-general with explicit batch and contracting dimensions.
pub fn dotGeneral(lhs: Buffer, rhs: Buffer, lhs_batch_dimensions: []const i64, rhs_batch_dimensions: []const i64, lhs_contracting_dimensions: []const i64, rhs_contracting_dimensions: []const i64, output_dims: []const i64) Error!?Buffer {
    return wrap(mlx_call.bufferDotGeneral(lhs.handle, rhs.handle, lhs_batch_dimensions, rhs_batch_dimensions, lhs_contracting_dimensions, rhs_contracting_dimensions, output_dims), error.CommandSubmissionFailed);
}

/// Runs convolution with explicit window metadata.
pub fn convolution(lhs: Buffer, rhs: Buffer, window_strides: []const i64, padding_low: []const i64, padding_high: []const i64, lhs_dilation: []const i64, rhs_dilation: []const i64, window_reversal: []const bool, feature_group_count: i64, output_dims: []const i64) Error!?Buffer {
    return wrap(mlx_call.bufferConvolution(lhs.handle, rhs.handle, window_strides, padding_low, padding_high, lhs_dilation, rhs_dilation, window_reversal, feature_group_count, output_dims), error.CommandSubmissionFailed);
}

/// Runs Cholesky decomposition on this MLX/Metal buffer.
pub fn cholesky(src: Buffer, lower: bool, output_dims: []const i64) Error!?Buffer {
    return wrap(mlx_call.bufferCholesky(src.handle, lower, output_dims), error.CommandSubmissionFailed);
}

/// Runs triangular solve with this buffer as the coefficient matrix.
pub fn triangularSolve(a: Buffer, b: Buffer, left_side: bool, lower: bool, unit_diagonal: bool, transpose_a: ir.TriangularSolveTranspose, output_dims: []const i64) Error!?Buffer {
    return wrap(mlx_call.bufferTriangularSolve(a.handle, b.handle, left_side, lower, unit_diagonal, encoding.triangularTranspose(transpose_a), output_dims), error.CommandSubmissionFailed);
}

/// Runs an FFT operation on this MLX/Metal buffer.
pub fn fft(src: Buffer, fft_kind: ir.FftKind, fft_lengths: []const i64, output_dims: []const i64) Error!?Buffer {
    return wrap(mlx_call.bufferFft(src.handle, encoding.fftKind(fft_kind), fft_lengths, output_dims), error.CommandSubmissionFailed);
}

fn wrap(handle: ?mlx_call.BufferHandle, err: Error) Error!?Buffer {
    return if (handle) |ptr| Buffer{ .handle = ptr } else err;
}
