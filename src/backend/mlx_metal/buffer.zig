const ir = @import("src/compiler/ir");

const api_control_flow = @import("buffer_api_control_flow.zig");
const api_custom_call = @import("buffer_api_custom_call.zig");
const api_elementwise = @import("buffer_api_elementwise.zig");
const api_generation = @import("buffer_api_generation.zig");
const api_indexing = @import("buffer_api_indexing.zig");
const api_lifecycle = @import("buffer_api_lifecycle.zig");
const api_linalg = @import("buffer_api_linalg.zig");
const api_reduction = @import("buffer_api_reduction.zig");
const encoding = @import("buffer_encoding.zig");
const handle_mod = @import("buffer_handle.zig");
const lifecycle = @import("buffer_lifecycle.zig");
const mlx_call = @import("mlx_call.zig");
const opaque_control_flow = @import("buffer_opaque_control_flow.zig");
const opaque_custom_call = @import("buffer_opaque_custom_call.zig");
const opaque_elementwise = @import("buffer_opaque_elementwise.zig");
const opaque_generation = @import("buffer_opaque_generation.zig");
const opaque_indexing = @import("buffer_opaque_indexing.zig");
const opaque_lifecycle = @import("buffer_opaque_lifecycle.zig");
const opaque_linalg = @import("buffer_opaque_linalg.zig");
const opaque_reduction = @import("buffer_opaque_reduction.zig");
const pair_mod = @import("buffer_pair.zig");
const results = @import("buffer_results.zig");

/// Typed owner-side wrapper for an MLX/Metal buffer handle.
pub const Buffer = struct {
    /// Backend-owned device buffer handle.
    handle: mlx_call.BufferHandle,

    /// Wraps a backend-opaque buffer handle without taking ownership.
    pub fn fromHandle(raw_handle: *anyopaque) Buffer { return .{ .handle = @ptrCast(raw_handle) }; }

    /// Returns the backend-opaque representation used by the runtime boundary.
    pub fn toHandle(self: Buffer) *anyopaque { return @ptrCast(self.handle); }

    pub const bitcast = api_elementwise.Typed(Buffer).bitcast;
    pub const argsort = api_indexing.Typed(Buffer).argsort;
    pub const binary = api_elementwise.Typed(Buffer).binary;
    pub const binaryWithOutputDims = api_elementwise.Typed(Buffer).binaryWithOutputDims;
    pub const broadcastInDim = api_indexing.Typed(Buffer).broadcastInDim;
    pub const cholesky = api_linalg.Typed(Buffer).cholesky;
    pub const clamp = api_elementwise.Typed(Buffer).clamp;
    pub const clone = api_lifecycle.Typed(Buffer).clone;
    pub const compare = api_elementwise.Typed(Buffer).compare;
    pub const complex = api_elementwise.Typed(Buffer).complex;
    pub const concatenate = api_indexing.Typed(Buffer).concatenate;
    pub const convolution = api_linalg.Typed(Buffer).convolution;
    pub const convert = api_elementwise.Typed(Buffer).convert;
    pub const copyToHost = api_lifecycle.Typed(Buffer).copyToHost;
    pub const customBinaryAddF32 = api_custom_call.Typed(Buffer).customBinaryAddF32;
    pub const customScaledDotProductAttention = api_custom_call.Typed(Buffer).customScaledDotProductAttention;
    pub const destroy = api_lifecycle.Typed(Buffer).destroy;
    pub const dotGeneral = api_linalg.Typed(Buffer).dotGeneral;
    pub const dynamicSlice = api_indexing.Typed(Buffer).dynamicSlice;
    pub const dynamicUpdateSlice = api_indexing.Typed(Buffer).dynamicUpdateSlice;
    pub const eval = api_lifecycle.Typed(Buffer).eval;
    pub const fft = api_linalg.Typed(Buffer).fft;
    pub const fromHost = api_lifecycle.Typed(Buffer).fromHost;
    pub const gather = api_indexing.Typed(Buffer).gather;
    pub const gatherAxis = api_indexing.Typed(Buffer).gatherAxis;
    pub const hasHostShadow = api_lifecycle.Typed(Buffer).hasHostShadow;
    pub const imagPart = api_elementwise.Typed(Buffer).imagPart;
    pub const iota = api_generation.Typed(Buffer).iota;
    pub const pad = api_indexing.Typed(Buffer).pad;
    pub const partitionId = api_generation.Typed(Buffer).partitionId;
    pub const realPart = api_elementwise.Typed(Buffer).realPart;
    pub const reduce = api_reduction.Typed(Buffer).reduce;
    pub const reduceMaxWithIndices = api_reduction.Typed(Buffer).reduceMaxWithIndices;
    pub const reduceWindow = api_reduction.Typed(Buffer).reduceWindow;
    pub const reduceWindowMaxWithIndices = api_reduction.Typed(Buffer).reduceWindowMaxWithIndices;
    pub const reshape = api_indexing.Typed(Buffer).reshape;
    pub const reverse = api_indexing.Typed(Buffer).reverse;
    pub const rng = api_generation.Typed(Buffer).rng;
    pub const rngBitGenerator = api_generation.Typed(Buffer).rngBitGenerator;
    pub const scatter = api_indexing.Typed(Buffer).scatter;
    pub const scatterAxis = api_indexing.Typed(Buffer).scatterAxis;
    pub const select = api_elementwise.Typed(Buffer).select;
    pub const slice = api_indexing.Typed(Buffer).slice;
    pub const sort = api_indexing.Typed(Buffer).sort;
    pub const takeAlongAxis = api_indexing.Typed(Buffer).takeAlongAxis;
    pub const transpose = api_indexing.Typed(Buffer).transpose;
    pub const triangularSolve = api_linalg.Typed(Buffer).triangularSolve;
    pub const unary = api_elementwise.Typed(Buffer).unary;
    pub const whileF32CompareAdd = api_control_flow.Typed(Buffer).whileF32CompareAdd;
    pub const zeroLike = api_lifecycle.Typed(Buffer).zeroLike;
    pub const zeros = api_lifecycle.Typed(Buffer).zeros;
};

/// Returns whether the MLX/Metal shim accepts this compiler IR element type.
pub fn supportsElementType(element_type: ir.BufferType) bool { return encoding.dtype(element_type) != null; }

/// Returns whether the MLX/Metal shim accepts this elementwise unary operation.
pub fn supportsUnaryOp(op: ir.ElementwiseUnaryOp) bool { return encoding.unaryOp(op) != null; }

/// Pair of typed MLX/Metal buffers returned by two-output operations.
pub const Pair = pair_mod.Pair(Buffer);

/// Opaque backend-buffer pair returned by max-reduction-with-indices.
pub const ReduceMaxWithIndicesResult = results.ReduceMaxWithIndicesResult;

/// Opaque backend-buffer pair returned by windowed max-reduction-with-indices.
pub const ReduceWindowMaxWithIndicesResult = results.ReduceWindowMaxWithIndicesResult;

/// Opaque backend-buffer pair returned by random bit generation.
pub const RngBitGeneratorResult = results.RngBitGeneratorResult;

/// Runtime-facing opaque buffer operations owned by the MLX/Metal buffer layer.
pub const Opaque = struct {
    pub const argsort = opaque_indexing.argsort;
    pub const bitcast = opaque_elementwise.bitcast;
    pub const binary = opaque_elementwise.binary;
    pub const binaryWithOutputDims = opaque_elementwise.binaryWithOutputDims;
    pub const broadcastInDim = opaque_indexing.broadcastInDim;
    pub const cholesky = opaque_linalg.cholesky;
    pub const clamp = opaque_elementwise.clamp;
    pub const clone = opaque_lifecycle.clone;
    pub const compare = opaque_elementwise.compare;
    pub const complex = opaque_elementwise.complex;
    pub const concatenate = opaque_indexing.concatenate;
    pub const convolution = opaque_linalg.convolution;
    pub const convert = opaque_elementwise.convert;
    pub const copyToHost = opaque_lifecycle.copyToHost;
    pub const customBinaryAddF32 = opaque_custom_call.customBinaryAddF32;
    pub const customScaledDotProductAttention = opaque_custom_call.customScaledDotProductAttention;
    pub const destroy = opaque_lifecycle.destroy;
    pub const dotGeneral = opaque_linalg.dotGeneral;
    pub const dynamicSlice = opaque_indexing.dynamicSlice;
    pub const dynamicUpdateSlice = opaque_indexing.dynamicUpdateSlice;
    pub const eval = opaque_lifecycle.eval;
    pub const evalMany = opaque_lifecycle.evalMany;
    pub const fft = opaque_linalg.fft;
    pub const fromHost = opaque_lifecycle.fromHost;
    pub const gather = opaque_indexing.gather;
    pub const gatherAxis = opaque_indexing.gatherAxis;
    pub const hasHostShadow = opaque_lifecycle.hasHostShadow;
    pub const imagPart = opaque_elementwise.imagPart;
    pub const iota = opaque_generation.iota;
    pub const pad = opaque_indexing.pad;
    pub const partitionId = opaque_generation.partitionId;
    pub const realPart = opaque_elementwise.realPart;
    pub const reduce = opaque_reduction.reduce;
    pub const reduceMaxWithIndices = opaque_reduction.reduceMaxWithIndices;
    pub const reduceWindow = opaque_reduction.reduceWindow;
    pub const reduceWindowMaxWithIndices = opaque_reduction.reduceWindowMaxWithIndices;
    pub const reshape = opaque_indexing.reshape;
    pub const reverse = opaque_indexing.reverse;
    pub const rng = opaque_generation.rng;
    pub const rngBitGenerator = opaque_generation.rngBitGenerator;
    pub const scatter = opaque_indexing.scatter;
    pub const scatterAxis = opaque_indexing.scatterAxis;
    pub const select = opaque_elementwise.select;
    pub const slice = opaque_indexing.slice;
    pub const sort = opaque_indexing.sort;
    pub const takeAlongAxis = opaque_indexing.takeAlongAxis;
    pub const transpose = opaque_indexing.transpose;
    pub const triangularSolve = opaque_linalg.triangularSolve;
    pub const unary = opaque_elementwise.unary;
    pub const whileF32CompareAdd = opaque_control_flow.whileF32CompareAdd;
    pub const zeroLike = opaque_lifecycle.zeroLike;
    pub const zeros = opaque_lifecycle.zeros;
};

/// Error set produced by typed MLX/Metal buffer operations.
pub const Error = handle_mod.Error;

/// Forces all supplied MLX/Metal buffers to evaluate as a single backend call.
pub fn evalMany(buffers: []const Buffer) Error!void { try lifecycle.evalMany(buffers); }
