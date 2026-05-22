const ir = @import("src/compiler/ir");

const linalg = @import("buffer_linalg.zig");

/// Builds linear-algebra methods for the typed backend buffer owner.
pub fn Typed(comptime Buffer: type) type {
    return struct {
        /// Runs dot-general with explicit batch and contracting dimensions.
        pub fn dotGeneral(lhs: Buffer, rhs: Buffer, lhs_batch_dimensions: []const i64, rhs_batch_dimensions: []const i64, lhs_contracting_dimensions: []const i64, rhs_contracting_dimensions: []const i64, output_dims: []const i64) linalg.Error!?Buffer { return linalg.dotGeneral(lhs, rhs, lhs_batch_dimensions, rhs_batch_dimensions, lhs_contracting_dimensions, rhs_contracting_dimensions, output_dims); }
        /// Runs convolution with explicit window metadata.
        pub fn convolution(lhs: Buffer, rhs: Buffer, window_strides: []const i64, padding_low: []const i64, padding_high: []const i64, lhs_dilation: []const i64, rhs_dilation: []const i64, window_reversal: []const bool, feature_group_count: i64, output_dims: []const i64) linalg.Error!?Buffer { return linalg.convolution(lhs, rhs, window_strides, padding_low, padding_high, lhs_dilation, rhs_dilation, window_reversal, feature_group_count, output_dims); }
        /// Runs Cholesky decomposition on this MLX/Metal buffer.
        pub fn cholesky(self: Buffer, lower: bool, output_dims: []const i64) linalg.Error!?Buffer { return linalg.cholesky(self, lower, output_dims); }
        /// Runs triangular solve with this buffer as the coefficient matrix.
        pub fn triangularSolve(a: Buffer, b: Buffer, left_side: bool, lower: bool, unit_diagonal: bool, transpose_a: ir.TriangularSolveTranspose, output_dims: []const i64) linalg.Error!?Buffer { return linalg.triangularSolve(a, b, left_side, lower, unit_diagonal, transpose_a, output_dims); }
        /// Runs an FFT operation on this MLX/Metal buffer.
        pub fn fft(self: Buffer, fft_kind: ir.FftKind, fft_lengths: []const i64, output_dims: []const i64) linalg.Error!?Buffer { return linalg.fft(self, fft_kind, fft_lengths, output_dims); }
    };
}
