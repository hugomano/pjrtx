const ir = @import("src/compiler/ir");

const buffer_handle = @import("buffer_handle.zig");
const encoding = @import("buffer_encoding.zig");
const mlx_call = @import("mlx_call.zig");

pub const Error = buffer_handle.Error;

/// Reshapes this MLX/Metal buffer.
pub fn reshape(src: anytype, dims: []const i64) Error!?@TypeOf(src) {
    return buffer_handle.wrap(@TypeOf(src), mlx_call.bufferReshape(src.handle, dims), error.CommandSubmissionFailed);
}

/// Transposes this MLX/Metal buffer.
pub fn transpose(src: anytype, permutation: []const i64) Error!?@TypeOf(src) {
    return buffer_handle.wrap(@TypeOf(src), mlx_call.bufferTranspose(src.handle, permutation), error.CommandSubmissionFailed);
}

/// Broadcasts this MLX/Metal buffer into explicit output dimensions.
pub fn broadcastInDim(src: anytype, broadcast_dimensions: []const i64, output_dims: []const i64) Error!?@TypeOf(src) {
    return buffer_handle.wrap(@TypeOf(src), mlx_call.bufferBroadcastInDim(src.handle, broadcast_dimensions, output_dims), error.CommandSubmissionFailed);
}

/// Slices this MLX/Metal buffer.
pub fn slice(src: anytype, start_indices: []const i64, limit_indices: []const i64, strides: []const i64, output_dims: []const i64) Error!?@TypeOf(src) {
    return buffer_handle.wrap(@TypeOf(src), mlx_call.bufferSlice(src.handle, start_indices, limit_indices, strides, output_dims), error.CommandSubmissionFailed);
}

/// Dynamically slices this MLX/Metal buffer.
pub fn dynamicSlice(src: anytype, start_buffers: []const @TypeOf(src), slice_sizes: []const i64, output_dims: []const i64) Error!?@TypeOf(src) {
    if (start_buffers.len != slice_sizes.len) return error.ShapeMismatch;
    return buffer_handle.wrap(@TypeOf(src), mlx_call.bufferDynamicSlice(src.handle, buffer_handle.rawHandles(start_buffers), slice_sizes, output_dims), error.CommandSubmissionFailed);
}

/// Dynamically updates this MLX/Metal buffer.
pub fn dynamicUpdateSlice(src: anytype, update: @TypeOf(src), start_buffers: []const @TypeOf(src), output_dims: []const i64) Error!?@TypeOf(src) {
    return buffer_handle.wrap(@TypeOf(src), mlx_call.bufferDynamicUpdateSlice(src.handle, update.handle, buffer_handle.rawHandles(start_buffers), output_dims), error.CommandSubmissionFailed);
}

/// Pads this MLX/Metal buffer.
pub fn pad(src: anytype, padding_value: @TypeOf(src), edge_padding_low: []const i64, edge_padding_high: []const i64, interior_padding: []const i64, output_dims: []const i64) Error!?@TypeOf(src) {
    return buffer_handle.wrap(@TypeOf(src), mlx_call.bufferPad(src.handle, padding_value.handle, edge_padding_low, edge_padding_high, interior_padding, output_dims), error.CommandSubmissionFailed);
}

/// Reverses dimensions of this MLX/Metal buffer.
pub fn reverse(src: anytype, dimensions: []const i64, output_dims: []const i64) Error!?@TypeOf(src) {
    return buffer_handle.wrap(@TypeOf(src), mlx_call.bufferReverse(src.handle, dimensions, output_dims), error.CommandSubmissionFailed);
}

/// Concatenates two MLX/Metal buffers.
pub fn concatenate(lhs: anytype, rhs: @TypeOf(lhs), dimension: i64, output_dims: []const i64) Error!?@TypeOf(lhs) {
    return buffer_handle.wrap(@TypeOf(lhs), mlx_call.bufferConcatenate(lhs.handle, rhs.handle, dimension, output_dims), error.CommandSubmissionFailed);
}

/// Gathers from this MLX/Metal buffer along one axis.
pub fn gatherAxis(operand: anytype, indices: @TypeOf(operand), axis: i64, index_vector_dim: i64, output_dims: []const i64) Error!?@TypeOf(operand) {
    return buffer_handle.wrap(@TypeOf(operand), mlx_call.bufferGatherAxis(operand.handle, indices.handle, axis, index_vector_dim, output_dims), error.CommandSubmissionFailed);
}

/// Gathers from this MLX/Metal buffer using explicit dimension-number metadata.
pub fn gather(operand: anytype, indices: @TypeOf(operand), start_index_map: []const i64, collapsed_slice_dims: []const i64, operand_batching_dims: []const i64, start_indices_batching_dims: []const i64, index_vector_dim: i64, slice_sizes: []const i64, offset_dims: []const i64, output_dims: []const i64) Error!?@TypeOf(operand) {
    return buffer_handle.wrap(@TypeOf(operand), mlx_call.bufferGather(operand.handle, indices.handle, start_index_map, collapsed_slice_dims, operand_batching_dims, start_indices_batching_dims, index_vector_dim, slice_sizes, offset_dims, output_dims), error.CommandSubmissionFailed);
}

/// Scatters updates into this MLX/Metal buffer along one axis.
pub fn scatterAxis(operand: anytype, indices: @TypeOf(operand), updates: @TypeOf(operand), axis: i64, index_vector_dim: i64, update_kind: ir.ScatterUpdateKind, output_dims: []const i64) Error!?@TypeOf(operand) {
    return buffer_handle.wrap(@TypeOf(operand), mlx_call.bufferScatterAxis(operand.handle, indices.handle, updates.handle, axis, index_vector_dim, encoding.scatterUpdate(update_kind), output_dims), error.CommandSubmissionFailed);
}

/// Scatters updates into this MLX/Metal buffer using explicit dimension-number metadata.
pub fn scatter(operand: anytype, indices: @TypeOf(operand), updates: @TypeOf(operand), scatter_dims_to_operand_dims: []const i64, inserted_window_dims: []const i64, update_window_dims: []const i64, input_batching_dims: []const i64, scatter_indices_batching_dims: []const i64, index_vector_dim: i64, update_kind: ir.ScatterUpdateKind, output_dims: []const i64) Error!?@TypeOf(operand) {
    return buffer_handle.wrap(@TypeOf(operand), mlx_call.bufferScatter(operand.handle, indices.handle, updates.handle, scatter_dims_to_operand_dims, inserted_window_dims, update_window_dims, input_batching_dims, scatter_indices_batching_dims, index_vector_dim, encoding.scatterUpdate(update_kind), output_dims), error.CommandSubmissionFailed);
}

/// Sorts this MLX/Metal buffer along one dimension.
pub fn sort(src: anytype, dimension: i64, output_dims: []const i64) Error!?@TypeOf(src) {
    return buffer_handle.wrap(@TypeOf(src), mlx_call.bufferSort(src.handle, dimension, output_dims), error.CommandSubmissionFailed);
}

/// Returns sorted indices for this MLX/Metal buffer.
pub fn argsort(src: anytype, dimension: i64, output_type: ir.BufferType, output_dims: []const i64) Error!?@TypeOf(src) {
    const dtype = encoding.dtype(output_type) orelse return error.UnsupportedElementType;
    return buffer_handle.wrap(@TypeOf(src), mlx_call.bufferArgsort(src.handle, dimension, dtype, output_dims), error.CommandSubmissionFailed);
}

/// Takes values from this MLX/Metal buffer using indices along one axis.
pub fn takeAlongAxis(src: anytype, indices: @TypeOf(src), dimension: i64, output_dims: []const i64) Error!?@TypeOf(src) {
    return buffer_handle.wrap(@TypeOf(src), mlx_call.bufferTakeAlongAxis(src.handle, indices.handle, dimension, output_dims), error.CommandSubmissionFailed);
}
