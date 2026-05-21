const ir = @import("src/compiler/ir");
const mlx_call = @import("mlx_call.zig");

/// Typed owner-side wrapper for an MLX/Metal async host-to-device transfer.
pub const AsyncTransfer = struct {
    handle: mlx_call.AsyncTransferHandle,

    /// Wraps a backend-opaque async transfer handle without taking ownership.
    pub fn fromHandle(handle: *anyopaque) AsyncTransfer {
        return .{ .handle = @ptrCast(handle) };
    }

    /// Returns the backend-opaque representation used by the runtime boundary.
    pub fn toHandle(self: AsyncTransfer) *anyopaque {
        return @ptrCast(self.handle);
    }

    /// Begins a typed MLX/Metal async host-to-device transfer.
    pub fn begin(device_local_hardware_id: i32, element_type: ir.BufferType, dims: []const i64, byte_size: usize) Error!?AsyncTransfer {
        if (byte_size == 0) return null;
        const dtype = dtypeFromIr(element_type) orelse return error.UnsupportedElementType;
        const handle = mlx_call.asyncH2DCreate(device_local_hardware_id, dtype, dims, byte_size) orelse return error.BufferAllocationFailed;
        return .{ .handle = handle };
    }

    /// Writes one byte range into this transfer.
    pub fn write(self: AsyncTransfer, offset: usize, src: []const u8) Error!void {
        if (!mlx_call.asyncH2DWrite(self.handle, offset, src)) return error.BufferCopyFailed;
    }

    /// Finishes this transfer and returns the completed opaque MLX/Metal buffer.
    pub fn finish(self: AsyncTransfer) Error!?*anyopaque {
        const handle = mlx_call.asyncH2DFinish(self.handle) orelse return error.BufferAllocationFailed;
        return @ptrCast(handle);
    }

    /// Destroys this transfer without producing a completed buffer.
    pub fn destroy(self: AsyncTransfer) void {
        mlx_call.asyncH2DDestroy(self.handle);
    }
};

/// Runtime-facing opaque async transfer operations owned by the transfer layer.
pub const Opaque = struct {
    /// Begins an opaque MLX/Metal async host-to-device transfer.
    pub fn begin(device_local_hardware_id: i32, element_type: ir.BufferType, dims: []const i64, byte_size: usize) Error!?*anyopaque {
        const transfer = try AsyncTransfer.begin(device_local_hardware_id, element_type, dims, byte_size);
        return if (transfer) |value| value.toHandle() else null;
    }

    /// Writes one byte range into an opaque async transfer handle.
    pub fn write(transfer: *anyopaque, offset: usize, src: []const u8) Error!void {
        try AsyncTransfer.fromHandle(transfer).write(offset, src);
    }

    /// Finishes an opaque async transfer handle and returns the completed opaque buffer handle.
    pub fn finish(transfer: *anyopaque) Error!?*anyopaque {
        return try AsyncTransfer.fromHandle(transfer).finish();
    }

    /// Destroys an opaque async transfer handle without producing a buffer.
    pub fn destroy(transfer: *anyopaque) void {
        AsyncTransfer.fromHandle(transfer).destroy();
    }
};

/// Error set produced by typed MLX/Metal async transfer operations.
pub const Error = error{
    UnsupportedElementType,
    BufferAllocationFailed,
    BufferCopyFailed,
};

fn dtypeFromIr(element_type: ir.BufferType) ?mlx_call.Dtype {
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

test "async transfer handle wrapper preserves opaque identity" {
    const raw: *anyopaque = @ptrFromInt(@alignOf(usize));
    try @import("std").testing.expectEqual(raw, AsyncTransfer.fromHandle(raw).toHandle());
}
