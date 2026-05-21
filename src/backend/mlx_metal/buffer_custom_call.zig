const buffer_mod = @import("buffer.zig");
const mlx_call = @import("mlx_call.zig");

const Buffer = buffer_mod.Buffer;
const Error = buffer_mod.Error;

/// Runs the built-in MLX/Metal binary-add custom call.
pub fn customBinaryAddF32(lhs: Buffer, rhs: Buffer) Error!?Buffer {
    return wrap(mlx_call.customCallBinaryAddF32(lhs.handle, rhs.handle), error.CommandSubmissionFailed);
}

/// Runs the built-in MLX/Metal scaled dot-product attention custom call.
pub fn customScaledDotProductAttention(q: Buffer, k: Buffer, v: Buffer, token_index: Buffer) Error!?Buffer {
    return wrap(mlx_call.customCallScaledDotProductAttention(q.handle, k.handle, v.handle, token_index.handle), error.CommandSubmissionFailed);
}

fn wrap(handle: ?mlx_call.BufferHandle, err: Error) Error!?Buffer {
    return if (handle) |ptr| Buffer{ .handle = ptr } else err;
}
