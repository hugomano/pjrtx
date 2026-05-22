const ir = @import("src/compiler/ir");

const opaque_ref = @import("buffer_opaque_ref.zig");

const Error = opaque_ref.Error;
const maybeHandle = opaque_ref.maybeHandle;
const ref = opaque_ref.ref;
const refs = opaque_ref.refs;

/// Runs dot-general with opaque buffer handles.
pub fn dotGeneral(lhs: *anyopaque, rhs: *anyopaque, lhs_batch_dimensions: []const i64, rhs_batch_dimensions: []const i64, lhs_contracting_dimensions: []const i64, rhs_contracting_dimensions: []const i64, output_dims: []const i64) Error!?*anyopaque { return maybeHandle(try opaque_ref.Ref.dotGeneral(ref(lhs), ref(rhs), lhs_batch_dimensions, rhs_batch_dimensions, lhs_contracting_dimensions, rhs_contracting_dimensions, output_dims)); }
/// Runs convolution with opaque buffer handles.
pub fn convolution(lhs: *anyopaque, rhs: *anyopaque, window_strides: []const i64, padding_low: []const i64, padding_high: []const i64, lhs_dilation: []const i64, rhs_dilation: []const i64, window_reversal: []const bool, feature_group_count: i64, output_dims: []const i64) Error!?*anyopaque { return maybeHandle(try opaque_ref.Ref.convolution(ref(lhs), ref(rhs), window_strides, padding_low, padding_high, lhs_dilation, rhs_dilation, window_reversal, feature_group_count, output_dims)); }
/// Runs Cholesky decomposition on an opaque buffer handle.
pub fn cholesky(src: *anyopaque, lower: bool, output_dims: []const i64) Error!?*anyopaque { return maybeHandle(try ref(src).cholesky(lower, output_dims)); }
/// Runs triangular solve with opaque buffer handles.
pub fn triangularSolve(a: *anyopaque, b: *anyopaque, left_side: bool, lower: bool, unit_diagonal: bool, transpose_a: ir.TriangularSolveTranspose, output_dims: []const i64) Error!?*anyopaque { return maybeHandle(try opaque_ref.Ref.triangularSolve(ref(a), ref(b), left_side, lower, unit_diagonal, transpose_a, output_dims)); }
/// Runs an FFT operation on an opaque buffer handle.
pub fn fft(src: *anyopaque, kind: ir.FftKind, fft_lengths: []const i64, output_dims: []const i64) Error!?*anyopaque { return maybeHandle(try ref(src).fft(kind, fft_lengths, output_dims)); }
