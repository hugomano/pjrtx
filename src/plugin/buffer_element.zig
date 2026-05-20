const c = @import("c");
const runtime = @import("src/runtime");

/// Converts between PJRT buffer element types and PjRTx runtime buffer types.
pub const ElementType = struct {
    /// Returns the dense byte width for one element of a PJRT buffer type.
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

    /// Decodes a PJRT element type into the runtime type used for buffers.
    pub fn fromPjrt(t: c.PJRT_Buffer_Type) runtime.BufferType {
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

    /// Encodes a runtime buffer element type for the PJRT C ABI.
    pub fn toPjrt(t: runtime.BufferType) c.PJRT_Buffer_Type {
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

    /// Computes the dense storage byte size for a PJRT shape.
    pub fn denseByteSize(t: c.PJRT_Buffer_Type, dims: []const i64) usize {
        var elems: usize = 1;
        for (dims) |dim| elems *= @intCast(dim);
        return elems * elementSize(t);
    }
};
