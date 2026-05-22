const c = @import("c");

const raw = @import("mlx_call_raw.zig");

const BinaryOp = raw.BinaryOp;
const BufferHandle = raw.BufferHandle;
const BufferPair = raw.BufferPair;
const CompareOp = raw.CompareOp;
const Dtype = raw.Dtype;
const FftKind = raw.FftKind;
const ReduceOp = raw.ReduceOp;
const RngDistribution = raw.RngDistribution;
const ScatterUpdate = raw.ScatterUpdate;
const TriangularTranspose = raw.TriangularTranspose;
const UnaryOp = raw.UnaryOp;
const code = raw.code;
const flag = raw.flag;
const fromRawBuffer = raw.fromRawBuffer;
const rawBuffer = raw.rawBuffer;
const rawBufferList = raw.rawBufferList;

/// Creates a typed MLX/Metal buffer by copying host bytes to a device.
pub fn bufferFromHost(device_ordinal: i32, dtype: Dtype, dims: []const i64, src: []const u8) ?BufferHandle {
    return fromRawBuffer(c.pjrtx_mlx_metal_buffer_from_host_typed(
        device_ordinal,
        src.ptr,
        src.len,
        code(dtype),
        dims.ptr,
        dims.len,
    ));
}

/// Allocates a zero-filled MLX/Metal buffer.
pub fn bufferZeros(device_ordinal: i32, dtype: Dtype, dims: []const i64) ?BufferHandle {
    return fromRawBuffer(c.pjrtx_mlx_metal_buffer_zeros(device_ordinal, code(dtype), dims.ptr, dims.len));
}

/// Creates an MLX/Metal iota buffer.
pub fn bufferIota(device_ordinal: i32, dtype: Dtype, dims: []const i64, iota_dimension: i64) ?BufferHandle {
    return fromRawBuffer(c.pjrtx_mlx_metal_buffer_iota(device_ordinal, code(dtype), dims.ptr, dims.len, iota_dimension));
}

/// Creates an MLX/Metal scalar partition-id buffer.
pub fn bufferPartitionId(device_ordinal: i32, dtype: Dtype, partition_id: u32) ?BufferHandle {
    return fromRawBuffer(c.pjrtx_mlx_metal_buffer_partition_id(device_ordinal, code(dtype), partition_id));
}

/// Clones an MLX/Metal buffer into a new device buffer.
pub fn bufferClone(src: BufferHandle) ?BufferHandle {
    return fromRawBuffer(c.pjrtx_mlx_metal_buffer_clone(rawBuffer(src)));
}

/// Creates a zero buffer with the same type and shape as `src`.
pub fn bufferZeroLike(src: BufferHandle) ?BufferHandle {
    return fromRawBuffer(c.pjrtx_mlx_metal_buffer_zero_like(rawBuffer(src)));
}

/// Creates a complex MLX/Metal buffer from real and imaginary inputs.
pub fn bufferComplex(real: BufferHandle, imag: BufferHandle, output_dims: []const i64) ?BufferHandle {
    return fromRawBuffer(c.pjrtx_mlx_metal_buffer_complex(rawBuffer(real), rawBuffer(imag), output_dims.ptr, output_dims.len));
}

/// Extracts the real component from an MLX/Metal complex buffer.
pub fn bufferReal(src: BufferHandle, output_dims: []const i64) ?BufferHandle {
    return fromRawBuffer(c.pjrtx_mlx_metal_buffer_real(rawBuffer(src), output_dims.ptr, output_dims.len));
}

/// Extracts the imaginary component from an MLX/Metal complex buffer.
pub fn bufferImag(src: BufferHandle, output_dims: []const i64) ?BufferHandle {
    return fromRawBuffer(c.pjrtx_mlx_metal_buffer_imag(rawBuffer(src), output_dims.ptr, output_dims.len));
}

/// Converts an MLX/Metal buffer to another dtype.
pub fn bufferAstype(src: BufferHandle, dtype: Dtype) ?BufferHandle {
    return fromRawBuffer(c.pjrtx_mlx_metal_buffer_astype(rawBuffer(src), code(dtype)));
}

/// Reinterprets an MLX/Metal buffer with a new dtype and shape.
pub fn bufferViewDtype(src: BufferHandle, dtype: Dtype, dims: []const i64) ?BufferHandle {
    return fromRawBuffer(c.pjrtx_mlx_metal_buffer_view_dtype(rawBuffer(src), code(dtype), dims.ptr, dims.len));
}

/// Applies an MLX/Metal binary operation.
pub fn bufferBinary(lhs: BufferHandle, rhs: BufferHandle, op: BinaryOp) ?BufferHandle {
    return fromRawBuffer(c.pjrtx_mlx_metal_buffer_binary(rawBuffer(lhs), rawBuffer(rhs), code(op)));
}

/// Applies an MLX/Metal binary operation with explicit output dimensions.
pub fn bufferBinaryOut(lhs: BufferHandle, rhs: BufferHandle, op: BinaryOp, output_dims: []const i64) ?BufferHandle {
    return fromRawBuffer(c.pjrtx_mlx_metal_buffer_binary_out(rawBuffer(lhs), rawBuffer(rhs), code(op), output_dims.ptr, output_dims.len));
}

/// Applies an MLX/Metal unary operation.
pub fn bufferUnary(src: BufferHandle, op: UnaryOp) ?BufferHandle {
    return fromRawBuffer(c.pjrtx_mlx_metal_buffer_unary(rawBuffer(src), code(op)));
}

/// Reshapes an MLX/Metal buffer.
pub fn bufferReshape(src: BufferHandle, dims: []const i64) ?BufferHandle {
    return fromRawBuffer(c.pjrtx_mlx_metal_buffer_reshape(rawBuffer(src), dims.ptr, dims.len));
}

/// Transposes an MLX/Metal buffer.
pub fn bufferTranspose(src: BufferHandle, permutation: []const i64) ?BufferHandle {
    return fromRawBuffer(c.pjrtx_mlx_metal_buffer_transpose(rawBuffer(src), permutation.ptr, permutation.len));
}

/// Broadcasts an MLX/Metal buffer into explicit output dimensions.
pub fn bufferBroadcastInDim(src: BufferHandle, broadcast_dimensions: []const i64, output_dims: []const i64) ?BufferHandle {
    return fromRawBuffer(c.pjrtx_mlx_metal_buffer_broadcast_in_dim(rawBuffer(src), broadcast_dimensions.ptr, broadcast_dimensions.len, output_dims.ptr, output_dims.len));
}

/// Slices an MLX/Metal buffer.
pub fn bufferSlice(src: BufferHandle, start_indices: []const i64, limit_indices: []const i64, strides: []const i64, output_dims: []const i64) ?BufferHandle {
    return fromRawBuffer(c.pjrtx_mlx_metal_buffer_slice(rawBuffer(src), start_indices.ptr, limit_indices.ptr, strides.ptr, start_indices.len, output_dims.ptr, output_dims.len));
}

/// Dynamically slices an MLX/Metal buffer using scalar start buffers.
pub fn bufferDynamicSlice(src: BufferHandle, start_buffers: []const BufferHandle, slice_sizes: []const i64, output_dims: []const i64) ?BufferHandle {
    return fromRawBuffer(c.pjrtx_mlx_metal_buffer_dynamic_slice(rawBuffer(src), rawBufferList(start_buffers), start_buffers.len, slice_sizes.ptr, slice_sizes.len, output_dims.ptr, output_dims.len));
}

/// Dynamically updates an MLX/Metal buffer using scalar start buffers.
pub fn bufferDynamicUpdateSlice(src: BufferHandle, update: BufferHandle, start_buffers: []const BufferHandle, output_dims: []const i64) ?BufferHandle {
    return fromRawBuffer(c.pjrtx_mlx_metal_buffer_dynamic_update_slice(rawBuffer(src), rawBuffer(update), rawBufferList(start_buffers), start_buffers.len, output_dims.ptr, output_dims.len));
}

/// Pads an MLX/Metal buffer with an MLX/Metal scalar value.
pub fn bufferPad(src: BufferHandle, padding_value: BufferHandle, edge_padding_low: []const i64, edge_padding_high: []const i64, interior_padding: []const i64, output_dims: []const i64) ?BufferHandle {
    return fromRawBuffer(c.pjrtx_mlx_metal_buffer_pad(rawBuffer(src), rawBuffer(padding_value), edge_padding_low.ptr, edge_padding_high.ptr, interior_padding.ptr, edge_padding_low.len, output_dims.ptr, output_dims.len));
}

/// Reverses dimensions of an MLX/Metal buffer.
pub fn bufferReverse(src: BufferHandle, dimensions: []const i64, output_dims: []const i64) ?BufferHandle {
    return fromRawBuffer(c.pjrtx_mlx_metal_buffer_reverse(rawBuffer(src), dimensions.ptr, dimensions.len, output_dims.ptr, output_dims.len));
}

/// Concatenates two MLX/Metal buffers along one dimension.
pub fn bufferConcatenate(lhs: BufferHandle, rhs: BufferHandle, dimension: i64, output_dims: []const i64) ?BufferHandle {
    return fromRawBuffer(c.pjrtx_mlx_metal_buffer_concatenate(rawBuffer(lhs), rawBuffer(rhs), dimension, output_dims.ptr, output_dims.len));
}

/// Gathers from an MLX/Metal buffer along one axis.
pub fn bufferGatherAxis(operand: BufferHandle, indices: BufferHandle, axis: i64, index_vector_dim: i64, output_dims: []const i64) ?BufferHandle {
    return fromRawBuffer(c.pjrtx_mlx_metal_buffer_gather_axis(rawBuffer(operand), rawBuffer(indices), axis, index_vector_dim, output_dims.ptr, output_dims.len));
}

/// Gathers from an MLX/Metal buffer using explicit dimension-number metadata.
pub fn bufferGather(operand: BufferHandle, indices: BufferHandle, start_index_map: []const i64, collapsed_slice_dims: []const i64, operand_batching_dims: []const i64, start_indices_batching_dims: []const i64, index_vector_dim: i64, slice_sizes: []const i64, offset_dims: []const i64, output_dims: []const i64) ?BufferHandle {
    return fromRawBuffer(c.pjrtx_mlx_metal_buffer_gather(rawBuffer(operand), rawBuffer(indices), start_index_map.ptr, start_index_map.len, collapsed_slice_dims.ptr, collapsed_slice_dims.len, operand_batching_dims.ptr, operand_batching_dims.len, start_indices_batching_dims.ptr, start_indices_batching_dims.len, index_vector_dim, slice_sizes.ptr, slice_sizes.len, offset_dims.ptr, offset_dims.len, output_dims.ptr, output_dims.len));
}

/// Scatters updates into an MLX/Metal buffer along one axis.
pub fn bufferScatterAxis(operand: BufferHandle, indices: BufferHandle, updates: BufferHandle, axis: i64, index_vector_dim: i64, update_kind: ScatterUpdate, output_dims: []const i64) ?BufferHandle {
    return fromRawBuffer(c.pjrtx_mlx_metal_buffer_scatter_axis(rawBuffer(operand), rawBuffer(indices), rawBuffer(updates), axis, index_vector_dim, code(update_kind), output_dims.ptr, output_dims.len));
}

/// Scatters updates into an MLX/Metal buffer using explicit dimension-number metadata.
pub fn bufferScatter(operand: BufferHandle, indices: BufferHandle, updates: BufferHandle, scatter_dims_to_operand_dims: []const i64, inserted_window_dims: []const i64, update_window_dims: []const i64, input_batching_dims: []const i64, scatter_indices_batching_dims: []const i64, index_vector_dim: i64, update_kind: ScatterUpdate, output_dims: []const i64) ?BufferHandle {
    return fromRawBuffer(c.pjrtx_mlx_metal_buffer_scatter(rawBuffer(operand), rawBuffer(indices), rawBuffer(updates), scatter_dims_to_operand_dims.ptr, scatter_dims_to_operand_dims.len, inserted_window_dims.ptr, inserted_window_dims.len, update_window_dims.ptr, update_window_dims.len, input_batching_dims.ptr, input_batching_dims.len, scatter_indices_batching_dims.ptr, scatter_indices_batching_dims.len, index_vector_dim, code(update_kind), output_dims.ptr, output_dims.len));
}

/// Sorts an MLX/Metal buffer along one dimension.
pub fn bufferSort(src: BufferHandle, dimension: i64, output_dims: []const i64) ?BufferHandle {
    return fromRawBuffer(c.pjrtx_mlx_metal_buffer_sort(rawBuffer(src), dimension, output_dims.ptr, output_dims.len));
}

/// Returns sorted indices for an MLX/Metal buffer.
pub fn bufferArgsort(src: BufferHandle, dimension: i64, output_dtype: Dtype, output_dims: []const i64) ?BufferHandle {
    return fromRawBuffer(c.pjrtx_mlx_metal_buffer_argsort(rawBuffer(src), dimension, code(output_dtype), output_dims.ptr, output_dims.len));
}

/// Takes values from an MLX/Metal buffer using indices along one axis.
pub fn bufferTakeAlongAxis(src: BufferHandle, indices: BufferHandle, dimension: i64, output_dims: []const i64) ?BufferHandle {
    return fromRawBuffer(c.pjrtx_mlx_metal_buffer_take_along_axis(rawBuffer(src), rawBuffer(indices), dimension, output_dims.ptr, output_dims.len));
}

/// Runs MLX/Metal dot-general with explicit batch and contracting dimensions.
pub fn bufferDotGeneral(lhs: BufferHandle, rhs: BufferHandle, lhs_batch_dimensions: []const i64, rhs_batch_dimensions: []const i64, lhs_contracting_dimensions: []const i64, rhs_contracting_dimensions: []const i64, output_dims: []const i64) ?BufferHandle {
    return fromRawBuffer(c.pjrtx_mlx_metal_buffer_dot_general(rawBuffer(lhs), rawBuffer(rhs), lhs_batch_dimensions.ptr, lhs_batch_dimensions.len, rhs_batch_dimensions.ptr, rhs_batch_dimensions.len, lhs_contracting_dimensions.ptr, lhs_contracting_dimensions.len, rhs_contracting_dimensions.ptr, rhs_contracting_dimensions.len, output_dims.ptr, output_dims.len));
}

/// Runs MLX/Metal convolution with explicit window metadata.
pub fn bufferConvolution(lhs: BufferHandle, rhs: BufferHandle, window_strides: []const i64, padding_low: []const i64, padding_high: []const i64, lhs_dilation: []const i64, rhs_dilation: []const i64, window_reversal: []const bool, feature_group_count: i64, output_dims: []const i64) ?BufferHandle {
    const reversal_bytes: [*]const u8 = @ptrCast(window_reversal.ptr);
    return fromRawBuffer(c.pjrtx_mlx_metal_buffer_convolution(rawBuffer(lhs), rawBuffer(rhs), window_strides.ptr, padding_low.ptr, padding_high.ptr, lhs_dilation.ptr, rhs_dilation.ptr, reversal_bytes, window_strides.len, feature_group_count, output_dims.ptr, output_dims.len));
}

/// Runs MLX/Metal Cholesky decomposition.
pub fn bufferCholesky(src: BufferHandle, lower: bool, output_dims: []const i64) ?BufferHandle {
    return fromRawBuffer(c.pjrtx_mlx_metal_buffer_cholesky(rawBuffer(src), flag(lower), output_dims.ptr, output_dims.len));
}

/// Runs MLX/Metal triangular solve.
pub fn bufferTriangularSolve(a: BufferHandle, b: BufferHandle, left_side: bool, lower: bool, unit_diagonal: bool, transpose_a: TriangularTranspose, output_dims: []const i64) ?BufferHandle {
    return fromRawBuffer(c.pjrtx_mlx_metal_buffer_triangular_solve(rawBuffer(a), rawBuffer(b), flag(left_side), flag(lower), flag(unit_diagonal), code(transpose_a), output_dims.ptr, output_dims.len));
}

/// Runs an MLX/Metal FFT operation.
pub fn bufferFft(src: BufferHandle, fft_kind: FftKind, fft_lengths: []const i64, output_dims: []const i64) ?BufferHandle {
    return fromRawBuffer(c.pjrtx_mlx_metal_buffer_fft(rawBuffer(src), code(fft_kind), fft_lengths.ptr, fft_lengths.len, output_dims.ptr, output_dims.len));
}

/// Runs an MLX/Metal random distribution operation.
pub fn bufferRng(a: BufferHandle, b: BufferHandle, distribution: RngDistribution, output_dtype: Dtype, output_dims: []const i64) ?BufferHandle {
    return fromRawBuffer(c.pjrtx_mlx_metal_buffer_rng(rawBuffer(a), rawBuffer(b), code(distribution), code(output_dtype), output_dims.ptr, output_dims.len));
}

/// Runs an MLX/Metal random bit-generator operation.
pub fn bufferRngBitGenerator(state: BufferHandle, output_dtype: Dtype, output_dims: []const i64) ?BufferPair {
    var out_state: ?*c.PjrtxMlxMetalBuffer = null;
    var out_bits: ?*c.PjrtxMlxMetalBuffer = null;
    if (c.pjrtx_mlx_metal_buffer_rng_bit_generator(rawBuffer(state), code(output_dtype), output_dims.ptr, output_dims.len, &out_state, &out_bits) == 0) return null;
    return .{
        .first = fromRawBuffer(out_state) orelse return null,
        .second = fromRawBuffer(out_bits) orelse return null,
    };
}

/// Runs an MLX/Metal reduction operation.
pub fn bufferReduce(src: BufferHandle, op: ReduceOp, dimensions: []const i64, output_dims: []const i64) ?BufferHandle {
    return fromRawBuffer(c.pjrtx_mlx_metal_buffer_reduce(rawBuffer(src), code(op), dimensions.ptr, dimensions.len, output_dims.ptr, output_dims.len));
}

/// Runs MLX/Metal max-reduction and returns values plus indices.
pub fn bufferReduceMaxWithIndices(values: BufferHandle, indices: BufferHandle, dimensions: []const i64, output_dims: []const i64) ?BufferPair {
    var out_values: ?*c.PjrtxMlxMetalBuffer = null;
    var out_indices: ?*c.PjrtxMlxMetalBuffer = null;
    if (c.pjrtx_mlx_metal_buffer_reduce_max_with_indices(rawBuffer(values), rawBuffer(indices), dimensions.ptr, dimensions.len, output_dims.ptr, output_dims.len, &out_values, &out_indices) == 0) return null;
    return .{
        .first = fromRawBuffer(out_values) orelse return null,
        .second = fromRawBuffer(out_indices) orelse return null,
    };
}

/// Runs an MLX/Metal windowed reduction operation.
pub fn bufferReduceWindow(src: BufferHandle, op: ReduceOp, window_dimensions: []const i64, window_strides: []const i64, base_dilations: []const i64, window_dilations: []const i64, padding_low: []const i64, padding_high: []const i64, output_dims: []const i64) ?BufferHandle {
    return fromRawBuffer(c.pjrtx_mlx_metal_buffer_reduce_window(rawBuffer(src), code(op), window_dimensions.ptr, window_strides.ptr, base_dilations.ptr, window_dilations.ptr, padding_low.ptr, padding_high.ptr, output_dims.len, output_dims.ptr, output_dims.len));
}

/// Runs MLX/Metal windowed max-reduction and returns values plus indices.
pub fn bufferReduceWindowMaxWithIndices(values: BufferHandle, indices: BufferHandle, window_dimensions: []const i64, window_strides: []const i64, base_dilations: []const i64, window_dilations: []const i64, padding_low: []const i64, padding_high: []const i64, output_dims: []const i64) ?BufferPair {
    var out_values: ?*c.PjrtxMlxMetalBuffer = null;
    var out_indices: ?*c.PjrtxMlxMetalBuffer = null;
    if (c.pjrtx_mlx_metal_buffer_reduce_window_max_with_indices(rawBuffer(values), rawBuffer(indices), window_dimensions.ptr, window_strides.ptr, base_dilations.ptr, window_dilations.ptr, padding_low.ptr, padding_high.ptr, output_dims.len, output_dims.ptr, output_dims.len, &out_values, &out_indices) == 0) return null;
    return .{
        .first = fromRawBuffer(out_values) orelse return null,
        .second = fromRawBuffer(out_indices) orelse return null,
    };
}

/// Compares two MLX/Metal buffers.
pub fn bufferCompare(lhs: BufferHandle, rhs: BufferHandle, direction: CompareOp, output_dims: []const i64) ?BufferHandle {
    return fromRawBuffer(c.pjrtx_mlx_metal_buffer_compare(rawBuffer(lhs), rawBuffer(rhs), code(direction), output_dims.ptr, output_dims.len));
}

/// Selects between two MLX/Metal buffers using a predicate buffer.
pub fn bufferSelect(pred: BufferHandle, on_true: BufferHandle, on_false: BufferHandle, output_dims: []const i64) ?BufferHandle {
    return fromRawBuffer(c.pjrtx_mlx_metal_buffer_select(rawBuffer(pred), rawBuffer(on_true), rawBuffer(on_false), output_dims.ptr, output_dims.len));
}

/// Clamps an MLX/Metal buffer between minimum and maximum buffers.
pub fn bufferClamp(min: BufferHandle, value: BufferHandle, max: BufferHandle, output_dims: []const i64) ?BufferHandle {
    return fromRawBuffer(c.pjrtx_mlx_metal_buffer_clamp(rawBuffer(min), rawBuffer(value), rawBuffer(max), output_dims.ptr, output_dims.len));
}

/// Runs the MLX/Metal while-compare-add fast path.
pub fn bufferWhileF32CompareAdd(state: BufferHandle, limit: BufferHandle, step: BufferHandle, compare_direction: CompareOp, update_op: BinaryOp, output_dims: []const i64, max_iterations: u64) ?BufferHandle {
    return fromRawBuffer(c.pjrtx_mlx_metal_buffer_while_f32_compare_add(rawBuffer(state), rawBuffer(limit), rawBuffer(step), code(compare_direction), code(update_op), output_dims.ptr, output_dims.len, max_iterations));
}

/// Runs the MLX/Metal custom binary add kernel.
pub fn customCallBinaryAddF32(lhs: BufferHandle, rhs: BufferHandle) ?BufferHandle {
    return fromRawBuffer(c.pjrtx_mlx_metal_custom_call_binary_add_f32(rawBuffer(lhs), rawBuffer(rhs)));
}

/// Runs the MLX/Metal scaled dot-product attention custom call.
pub fn customCallScaledDotProductAttention(q: BufferHandle, k: BufferHandle, v: BufferHandle, token_index: BufferHandle) ?BufferHandle {
    return fromRawBuffer(c.pjrtx_mlx_metal_custom_call_scaled_dot_product_attention(rawBuffer(q), rawBuffer(k), rawBuffer(v), rawBuffer(token_index)));
}

/// Forces MLX/Metal evaluation of one buffer.
pub fn bufferEval(buffer: BufferHandle) bool {
    return c.pjrtx_mlx_metal_buffer_eval(rawBuffer(buffer)) != 0;
}

/// Forces MLX/Metal evaluation of several buffers.
pub fn bufferEvalMany(buffers: []const BufferHandle) bool {
    if (buffers.len == 0) return true;
    return c.pjrtx_mlx_metal_buffer_eval_many(rawBufferList(buffers), buffers.len) != 0;
}

/// Copies MLX/Metal buffer bytes into host memory.
pub fn bufferCopyToHost(buffer: BufferHandle, dst: []u8) bool {
    return c.pjrtx_mlx_metal_buffer_copy_to_host(rawBuffer(buffer), dst.ptr, dst.len) != 0;
}

/// Reports whether a buffer still has a host shadow allocation.
pub fn bufferHasHostShadow(buffer: BufferHandle) bool {
    return c.pjrtx_mlx_metal_buffer_has_host_shadow(rawBuffer(buffer)) != 0;
}

/// Destroys an MLX/Metal buffer handle.
pub fn bufferDestroy(buffer: BufferHandle) void {
    c.pjrtx_mlx_metal_buffer_destroy(rawBuffer(buffer));
}

