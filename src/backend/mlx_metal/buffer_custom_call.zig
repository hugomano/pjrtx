const buffer_handle = @import("buffer_handle.zig");
const mlx_call = @import("mlx_call.zig");

pub const Error = buffer_handle.Error;

/// Runs the built-in MLX/Metal binary-add custom call.
pub fn customBinaryAddF32(lhs: anytype, rhs: @TypeOf(lhs)) Error!?@TypeOf(lhs) {
    return buffer_handle.wrap(@TypeOf(lhs), mlx_call.customCallBinaryAddF32(lhs.handle, rhs.handle), error.CommandSubmissionFailed);
}

/// Runs the built-in MLX/Metal scaled dot-product attention custom call.
pub fn customScaledDotProductAttention(q: anytype, k: @TypeOf(q), v: @TypeOf(q), token_index: @TypeOf(q)) Error!?@TypeOf(q) {
    return buffer_handle.wrap(@TypeOf(q), mlx_call.customCallScaledDotProductAttention(q.handle, k.handle, v.handle, token_index.handle), error.CommandSubmissionFailed);
}
