const std = @import("std");

const runtime = @import("src/runtime");
const c = @import("c");
const abi = @import("pjrt_abi.zig");
const state = @import("state.zig");

const default_memory_kind = state.default_memory_kind;

pub fn copyBytesWithIo(dst: []u8, src: []const u8) !void {
    if (dst.len != src.len) return error.InvalidArgument;
    var reader: std.Io.Reader = .fixed(src);
    var writer = std.Io.Writer.fixed(dst);
    const copied = try reader.streamRemaining(&writer);
    if (copied != src.len) return error.InvalidArgument;
}
pub fn fillMemoryKindArrays(kinds: [][*c]const u8, sizes: []usize) void {
    for (kinds, sizes) |*kind, *size| {
        kind.* = default_memory_kind.ptr;
        size.* = default_memory_kind.len;
    }
}
pub fn cstrLen(ptr: [*c]const u8) usize {
    return std.mem.len(ptr);
}
pub fn elementSize(t: c.PJRT_Buffer_Type) usize {
    return switch (t) {
        c.PJRT_Buffer_Type_PRED, c.PJRT_Buffer_Type_S8, c.PJRT_Buffer_Type_U8 => 1,
        c.PJRT_Buffer_Type_S16, c.PJRT_Buffer_Type_U16, c.PJRT_Buffer_Type_F16, c.PJRT_Buffer_Type_BF16 => 2,
        c.PJRT_Buffer_Type_S32, c.PJRT_Buffer_Type_U32, c.PJRT_Buffer_Type_F32 => 4,
        c.PJRT_Buffer_Type_S64, c.PJRT_Buffer_Type_U64, c.PJRT_Buffer_Type_F64, c.PJRT_Buffer_Type_C64 => 8,
        c.PJRT_Buffer_Type_C128 => 16,
        else => 0,
    };
}
pub fn runtimeTypeFromPjrt(t: c.PJRT_Buffer_Type) runtime.BufferType {
    return switch (t) {
        c.PJRT_Buffer_Type_PRED => .pred,
        c.PJRT_Buffer_Type_S8 => .s8,
        c.PJRT_Buffer_Type_S16 => .s16,
        c.PJRT_Buffer_Type_S32 => .s32,
        c.PJRT_Buffer_Type_S64 => .s64,
        c.PJRT_Buffer_Type_U8 => .u8,
        c.PJRT_Buffer_Type_U16 => .u16,
        c.PJRT_Buffer_Type_U32 => .u32,
        c.PJRT_Buffer_Type_U64 => .u64,
        c.PJRT_Buffer_Type_F16 => .f16,
        c.PJRT_Buffer_Type_F32 => .f32,
        c.PJRT_Buffer_Type_F64 => .f64,
        c.PJRT_Buffer_Type_BF16 => .bf16,
        c.PJRT_Buffer_Type_C64 => .c64,
        c.PJRT_Buffer_Type_C128 => .c128,
        else => .invalid,
    };
}
pub fn pjrtTypeFromRuntime(t: runtime.BufferType) c.PJRT_Buffer_Type {
    return switch (t) {
        .invalid => c.PJRT_Buffer_Type_INVALID,
        .pred => c.PJRT_Buffer_Type_PRED,
        .s8 => c.PJRT_Buffer_Type_S8,
        .s16 => c.PJRT_Buffer_Type_S16,
        .s32 => c.PJRT_Buffer_Type_S32,
        .s64 => c.PJRT_Buffer_Type_S64,
        .u8 => c.PJRT_Buffer_Type_U8,
        .u16 => c.PJRT_Buffer_Type_U16,
        .u32 => c.PJRT_Buffer_Type_U32,
        .u64 => c.PJRT_Buffer_Type_U64,
        .f16 => c.PJRT_Buffer_Type_F16,
        .f32 => c.PJRT_Buffer_Type_F32,
        .f64 => c.PJRT_Buffer_Type_F64,
        .bf16 => c.PJRT_Buffer_Type_BF16,
        .c64 => c.PJRT_Buffer_Type_C64,
        .c128 => c.PJRT_Buffer_Type_C128,
    };
}
pub fn denseByteSize(t: c.PJRT_Buffer_Type, dims: []const i64) usize {
    var elems: usize = 1;
    for (dims) |dim| elems *= @intCast(dim);
    return elems * elementSize(t);
}
