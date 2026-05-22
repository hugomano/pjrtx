const mlx_call = @import("mlx_call.zig");

/// Error set produced by typed MLX/Metal buffer operations.
pub const Error = error{
    UnsupportedElementType,
    ShapeMismatch,
    BufferAllocationFailed,
    CommandSubmissionFailed,
    BufferCopyFailed,
};

/// Pair of typed MLX/Metal buffers returned by two-output operations.
pub fn Pair(comptime Buffer: type) type {
    return struct {
        /// First returned typed buffer.
        first: Buffer,
        /// Second returned typed buffer.
        second: Buffer,
    };
}

/// Wraps a nullable raw MLX/Metal handle in the caller-owned typed buffer.
pub fn wrap(comptime Buffer: type, raw: ?mlx_call.BufferHandle, err: Error) Error!?Buffer {
    return if (raw) |ptr| Buffer{ .handle = ptr } else err;
}

/// Reinterprets a packed slice of typed buffer wrappers as raw MLX handles.
pub fn rawHandles(buffers: anytype) []const mlx_call.BufferHandle {
    return @ptrCast(buffers);
}
