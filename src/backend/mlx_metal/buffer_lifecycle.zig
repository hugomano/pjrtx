const std = @import("std");
const ir = @import("src/compiler/ir");

const buffer_mod = @import("buffer.zig");
const encoding = @import("buffer_encoding.zig");
const mlx_call = @import("mlx_call.zig");

const Buffer = buffer_mod.Buffer;
const Error = buffer_mod.Error;

/// Copies host bytes into a typed MLX/Metal buffer.
pub fn fromHost(device_local_hardware_id: i32, element_type: ir.BufferType, dims: []const i64, src: []const u8) Error!?Buffer {
    if (src.len == 0) return null;
    const dtype = encoding.dtype(element_type) orelse return error.UnsupportedElementType;
    return wrap(mlx_call.bufferFromHost(device_local_hardware_id, dtype, dims, src), error.BufferAllocationFailed);
}

/// Allocates a zero-initialized MLX/Metal buffer.
pub fn zeros(device_local_hardware_id: i32, element_type: ir.BufferType, dims: []const i64) Error!?Buffer {
    const dtype = encoding.dtype(element_type) orelse return error.UnsupportedElementType;
    return wrap(mlx_call.bufferZeros(device_local_hardware_id, dtype, dims), error.BufferAllocationFailed);
}

/// Clones this buffer into a new MLX/Metal buffer.
pub fn clone(src: Buffer) Error!?Buffer {
    return wrap(mlx_call.bufferClone(src.handle), error.CommandSubmissionFailed);
}

/// Creates a zero buffer with the same MLX/Metal shape and type as this buffer.
pub fn zeroLike(src: Buffer) Error!?Buffer {
    return wrap(mlx_call.bufferZeroLike(src.handle), error.CommandSubmissionFailed);
}

/// Forces this MLX/Metal buffer to evaluate.
pub fn eval(buffer: Buffer) Error!void {
    if (!mlx_call.bufferEval(buffer.handle)) return error.CommandSubmissionFailed;
}

/// Forces all supplied MLX/Metal buffers to evaluate as a single backend call.
pub fn evalMany(buffers: []const Buffer) Error!void {
    if (!mlx_call.bufferEvalMany(handles(buffers))) return error.CommandSubmissionFailed;
}

/// Copies this MLX/Metal buffer into host memory.
pub fn copyToHost(buffer: Buffer, dst: []u8) Error!void {
    if (!mlx_call.bufferCopyToHost(buffer.handle, dst)) return error.BufferCopyFailed;
}

/// Reports whether this buffer still owns a host shadow allocation.
pub fn hasHostShadow(buffer: Buffer) bool {
    return mlx_call.bufferHasHostShadow(buffer.handle);
}

/// Destroys this MLX/Metal buffer handle.
pub fn destroy(buffer: Buffer) void {
    mlx_call.bufferDestroy(buffer.handle);
}

fn wrap(handle: ?mlx_call.BufferHandle, err: Error) Error!?Buffer {
    return if (handle) |ptr| Buffer{ .handle = ptr } else err;
}

fn handles(buffers: []const Buffer) []const mlx_call.BufferHandle {
    return @ptrCast(buffers);
}

test "buffer handle wrapper preserves opaque identity" {
    const raw: *anyopaque = @ptrFromInt(@alignOf(usize));
    try std.testing.expectEqual(raw, Buffer.fromHandle(raw).toHandle());
}
