const custom_call = @import("buffer_custom_call.zig");

/// Builds custom-call methods for the typed backend buffer owner.
pub fn Typed(comptime Buffer: type) type {
    return struct {
        /// Runs the built-in MLX/Metal binary-add custom call.
        pub fn customBinaryAddF32(lhs: Buffer, rhs: Buffer) custom_call.Error!?Buffer { return custom_call.customBinaryAddF32(lhs, rhs); }
        /// Runs the built-in MLX/Metal scaled dot-product attention custom call.
        pub fn customScaledDotProductAttention(q: Buffer, k: Buffer, v: Buffer, token_index: Buffer) custom_call.Error!?Buffer { return custom_call.customScaledDotProductAttention(q, k, v, token_index); }
    };
}
