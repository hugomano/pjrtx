const ir = @import("src/compiler/ir");

const opaque_ref = @import("buffer_opaque_ref.zig");

const Error = opaque_ref.Error;
const maybeHandle = opaque_ref.maybeHandle;
const ref = opaque_ref.ref;
const refs = opaque_ref.refs;

/// Runs the built-in binary-add custom call over opaque handles.
pub fn customBinaryAddF32(lhs: *anyopaque, rhs: *anyopaque) Error!?*anyopaque { return maybeHandle(try opaque_ref.Ref.customBinaryAddF32(ref(lhs), ref(rhs))); }
/// Runs the built-in scaled dot-product attention custom call over opaque handles.
pub fn customScaledDotProductAttention(q: *anyopaque, k: *anyopaque, v: *anyopaque, token_index: *anyopaque) Error!?*anyopaque { return maybeHandle(try opaque_ref.Ref.customScaledDotProductAttention(ref(q), ref(k), ref(v), ref(token_index))); }
