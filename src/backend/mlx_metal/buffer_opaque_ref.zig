const api_control_flow = @import("buffer_api_control_flow.zig");
const api_custom_call = @import("buffer_api_custom_call.zig");
const api_elementwise = @import("buffer_api_elementwise.zig");
const api_generation = @import("buffer_api_generation.zig");
const api_indexing = @import("buffer_api_indexing.zig");
const api_lifecycle = @import("buffer_api_lifecycle.zig");
const api_linalg = @import("buffer_api_linalg.zig");
const api_reduction = @import("buffer_api_reduction.zig");
const handle_mod = @import("buffer_handle.zig");
const lifecycle = @import("buffer_lifecycle.zig");
const mlx_call = @import("mlx_call.zig");

/// Error set produced by opaque MLX/Metal buffer operations.
pub const Error = handle_mod.Error;

/// Borrowed typed view over an opaque runtime-facing backend buffer handle.
pub const Ref = struct {
    /// Backend-owned device buffer handle.
    handle: mlx_call.BufferHandle,

    /// Returns the runtime-facing opaque handle.
    pub fn toHandle(self: Ref) *anyopaque {
        return @ptrCast(self.handle);
    }

    pub const argsort = api_indexing.Typed(Ref).argsort;
    pub const bitcast = api_elementwise.Typed(Ref).bitcast;
    pub const binary = api_elementwise.Typed(Ref).binary;
    pub const binaryWithOutputDims = api_elementwise.Typed(Ref).binaryWithOutputDims;
    pub const broadcastInDim = api_indexing.Typed(Ref).broadcastInDim;
    pub const cholesky = api_linalg.Typed(Ref).cholesky;
    pub const clamp = api_elementwise.Typed(Ref).clamp;
    pub const clone = api_lifecycle.Typed(Ref).clone;
    pub const compare = api_elementwise.Typed(Ref).compare;
    pub const complex = api_elementwise.Typed(Ref).complex;
    pub const concatenate = api_indexing.Typed(Ref).concatenate;
    pub const convolution = api_linalg.Typed(Ref).convolution;
    pub const convert = api_elementwise.Typed(Ref).convert;
    pub const copyToHost = api_lifecycle.Typed(Ref).copyToHost;
    pub const customBinaryAddF32 = api_custom_call.Typed(Ref).customBinaryAddF32;
    pub const customScaledDotProductAttention = api_custom_call.Typed(Ref).customScaledDotProductAttention;
    pub const destroy = api_lifecycle.Typed(Ref).destroy;
    pub const dotGeneral = api_linalg.Typed(Ref).dotGeneral;
    pub const dynamicSlice = api_indexing.Typed(Ref).dynamicSlice;
    pub const dynamicUpdateSlice = api_indexing.Typed(Ref).dynamicUpdateSlice;
    pub const eval = api_lifecycle.Typed(Ref).eval;
    pub const fft = api_linalg.Typed(Ref).fft;
    pub const gather = api_indexing.Typed(Ref).gather;
    pub const gatherAxis = api_indexing.Typed(Ref).gatherAxis;
    pub const hasHostShadow = api_lifecycle.Typed(Ref).hasHostShadow;
    pub const imagPart = api_elementwise.Typed(Ref).imagPart;
    pub const pad = api_indexing.Typed(Ref).pad;
    pub const realPart = api_elementwise.Typed(Ref).realPart;
    pub const reduce = api_reduction.Typed(Ref).reduce;
    pub const reduceMaxWithIndices = api_reduction.Typed(Ref).reduceMaxWithIndices;
    pub const reduceWindow = api_reduction.Typed(Ref).reduceWindow;
    pub const reduceWindowMaxWithIndices = api_reduction.Typed(Ref).reduceWindowMaxWithIndices;
    pub const reshape = api_indexing.Typed(Ref).reshape;
    pub const reverse = api_indexing.Typed(Ref).reverse;
    pub const rng = api_generation.Typed(Ref).rng;
    pub const rngBitGenerator = api_generation.Typed(Ref).rngBitGenerator;
    pub const scatter = api_indexing.Typed(Ref).scatter;
    pub const scatterAxis = api_indexing.Typed(Ref).scatterAxis;
    pub const select = api_elementwise.Typed(Ref).select;
    pub const slice = api_indexing.Typed(Ref).slice;
    pub const sort = api_indexing.Typed(Ref).sort;
    pub const takeAlongAxis = api_indexing.Typed(Ref).takeAlongAxis;
    pub const transpose = api_indexing.Typed(Ref).transpose;
    pub const triangularSolve = api_linalg.Typed(Ref).triangularSolve;
    pub const unary = api_elementwise.Typed(Ref).unary;
    pub const whileF32CompareAdd = api_control_flow.Typed(Ref).whileF32CompareAdd;
    pub const zeroLike = api_lifecycle.Typed(Ref).zeroLike;
};

/// Decodes a runtime-facing opaque buffer handle for backend buffer operations.
pub fn ref(raw_handle: *anyopaque) Ref {
    return .{ .handle = @ptrCast(raw_handle) };
}

/// Decodes an array of runtime-facing opaque handles for backend calls.
pub fn refs(handles: []const *anyopaque) []const Ref {
    return @ptrCast(handles);
}

/// Converts an optional decoded buffer to the runtime-facing opaque handle.
pub fn maybeHandle(buffer: ?Ref) ?*anyopaque {
    return if (buffer) |value| value.toHandle() else null;
}

/// Forces all decoded opaque buffers to evaluate as one backend call.
pub fn evalMany(buffers: []const Ref) Error!void {
    try lifecycle.evalMany(buffers);
}
