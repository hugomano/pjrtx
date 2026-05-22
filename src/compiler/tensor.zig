const std = @import("std");

/// Scalar element type for compiler IR tensor values.
pub const BufferType = enum {
    invalid,
    pred,
    s8,
    s16,
    s32,
    s64,
    u8,
    u16,
    u32,
    u64,
    f16,
    f32,
    f64,
    bf16,
    c64,
    c128,

    /// Returns the fixed byte width of one scalar element, or zero for invalid types.
    pub fn byteSize(self: BufferType) usize {
        return switch (self) {
            .invalid => 0,
            .pred, .s8, .u8 => 1,
            .s16, .u16, .f16, .bf16 => 2,
            .s32, .u32, .f32 => 4,
            .s64, .u64, .f64, .c64 => 8,
            .c128 => 16,
        };
    }
};

/// Layout class for a tensor buffer descriptor at the compiler IR boundary.
pub const LayoutKind = enum {
    dense_row_major,
    opaque_backend,
};

/// Compiler-owned tensor buffer facts shared across plan values and placements.
pub const BufferDescriptor = struct {
    element_type: BufferType,
    dims: []const i64,
    layout: LayoutKind = .dense_row_major,
    device_id: i32,
    memory_id: i32,
    shard_index: usize,
};

/// Returns the byte size of a dense tensor, or zero for invalid/overflowing shapes.
pub fn denseByteSize(element_type: BufferType, dims: []const i64) usize {
    const element_size = element_type.byteSize();
    if (element_size == 0) return 0;
    var elements: usize = 1;
    for (dims) |dim| {
        if (dim < 0) return 0;
        elements = std.math.mul(usize, elements, @intCast(dim)) catch return 0;
    }
    return std.math.mul(usize, elements, element_size) catch 0;
}

test "core dense byte size rejects invalid dimensions" {
    try std.testing.expectEqual(@as(usize, 16), denseByteSize(.f32, &.{ 2, 2 }));
    try std.testing.expectEqual(@as(usize, 0), denseByteSize(.f32, &.{ -1, 2 }));
}
