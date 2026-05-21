const ir = @import("src/compiler/ir");

const buffer_mod = @import("buffer.zig");
const encoding = @import("buffer_encoding.zig");
const mlx_call = @import("mlx_call.zig");

const Buffer = buffer_mod.Buffer;
const Error = buffer_mod.Error;

/// Reshapes this MLX/Metal buffer.
pub fn reshape(src: Buffer, dims: []const i64) Error!?Buffer {
    return wrap(mlx_call.bufferReshape(src.handle, dims), error.CommandSubmissionFailed);
}

/// Transposes this MLX/Metal buffer.
pub fn transpose(src: Buffer, permutation: []const i64) Error!?Buffer {
    return wrap(mlx_call.bufferTranspose(src.handle, permutation), error.CommandSubmissionFailed);
}

/// Broadcasts this MLX/Metal buffer into explicit output dimensions.
pub fn broadcastInDim(src: Buffer, broadcast_dimensions: []const i64, output_dims: []const i64) Error!?Buffer {
    return wrap(mlx_call.bufferBroadcastInDim(src.handle, broadcast_dimensions, output_dims), error.CommandSubmissionFailed);
}

/// Slices this MLX/Metal buffer.
pub fn slice(src: Buffer, start_indices: []const i64, limit_indices: []const i64, strides: []const i64, output_dims: []const i64) Error!?Buffer {
    return wrap(mlx_call.bufferSlice(src.handle, start_indices, limit_indices, strides, output_dims), error.CommandSubmissionFailed);
}

/// Dynamically slices this MLX/Metal buffer.
pub fn dynamicSlice(src: Buffer, start_buffers: []const Buffer, slice_sizes: []const i64, output_dims: []const i64) Error!?Buffer {
    if (start_buffers.len != slice_sizes.len) return error.ShapeMismatch;
    return wrap(mlx_call.bufferDynamicSlice(src.handle, handles(start_buffers), slice_sizes, output_dims), error.CommandSubmissionFailed);
}

/// Dynamically updates this MLX/Metal buffer.
pub fn dynamicUpdateSlice(src: Buffer, update: Buffer, start_buffers: []const Buffer, output_dims: []const i64) Error!?Buffer {
    return wrap(mlx_call.bufferDynamicUpdateSlice(src.handle, update.handle, handles(start_buffers), output_dims), error.CommandSubmissionFailed);
}

/// Pads this MLX/Metal buffer.
pub fn pad(src: Buffer, padding_value: Buffer, edge_padding_low: []const i64, edge_padding_high: []const i64, interior_padding: []const i64, output_dims: []const i64) Error!?Buffer {
    return wrap(mlx_call.bufferPad(src.handle, padding_value.handle, edge_padding_low, edge_padding_high, interior_padding, output_dims), error.CommandSubmissionFailed);
}

/// Reverses dimensions of this MLX/Metal buffer.
pub fn reverse(src: Buffer, dimensions: []const i64, output_dims: []const i64) Error!?Buffer {
    return wrap(mlx_call.bufferReverse(src.handle, dimensions, output_dims), error.CommandSubmissionFailed);
}

/// Concatenates two MLX/Metal buffers.
pub fn concatenate(lhs: Buffer, rhs: Buffer, dimension: i64, output_dims: []const i64) Error!?Buffer {
    return wrap(mlx_call.bufferConcatenate(lhs.handle, rhs.handle, dimension, output_dims), error.CommandSubmissionFailed);
}

/// Gathers from this MLX/Metal buffer along one axis.
pub fn gatherAxis(operand: Buffer, indices: Buffer, axis: i64, index_vector_dim: i64, output_dims: []const i64) Error!?Buffer {
    return wrap(mlx_call.bufferGatherAxis(operand.handle, indices.handle, axis, index_vector_dim, output_dims), error.CommandSubmissionFailed);
}

/// Gathers from this MLX/Metal buffer using explicit dimension-number metadata.
pub fn gather(operand: Buffer, indices: Buffer, start_index_map: []const i64, collapsed_slice_dims: []const i64, operand_batching_dims: []const i64, start_indices_batching_dims: []const i64, index_vector_dim: i64, slice_sizes: []const i64, offset_dims: []const i64, output_dims: []const i64) Error!?Buffer {
    return wrap(mlx_call.bufferGather(operand.handle, indices.handle, start_index_map, collapsed_slice_dims, operand_batching_dims, start_indices_batching_dims, index_vector_dim, slice_sizes, offset_dims, output_dims), error.CommandSubmissionFailed);
}

/// Scatters updates into this MLX/Metal buffer along one axis.
pub fn scatterAxis(operand: Buffer, indices: Buffer, updates: Buffer, axis: i64, index_vector_dim: i64, update_kind: ir.ScatterUpdateKind, output_dims: []const i64) Error!?Buffer {
    return wrap(mlx_call.bufferScatterAxis(operand.handle, indices.handle, updates.handle, axis, index_vector_dim, encoding.scatterUpdate(update_kind), output_dims), error.CommandSubmissionFailed);
}

/// Scatters updates into this MLX/Metal buffer using explicit dimension-number metadata.
pub fn scatter(operand: Buffer, indices: Buffer, updates: Buffer, scatter_dims_to_operand_dims: []const i64, inserted_window_dims: []const i64, update_window_dims: []const i64, input_batching_dims: []const i64, scatter_indices_batching_dims: []const i64, index_vector_dim: i64, update_kind: ir.ScatterUpdateKind, output_dims: []const i64) Error!?Buffer {
    return wrap(mlx_call.bufferScatter(operand.handle, indices.handle, updates.handle, scatter_dims_to_operand_dims, inserted_window_dims, update_window_dims, input_batching_dims, scatter_indices_batching_dims, index_vector_dim, encoding.scatterUpdate(update_kind), output_dims), error.CommandSubmissionFailed);
}

/// Sorts this MLX/Metal buffer along one dimension.
pub fn sort(src: Buffer, dimension: i64, output_dims: []const i64) Error!?Buffer {
    return wrap(mlx_call.bufferSort(src.handle, dimension, output_dims), error.CommandSubmissionFailed);
}

/// Returns sorted indices for this MLX/Metal buffer.
pub fn argsort(src: Buffer, dimension: i64, output_type: ir.BufferType, output_dims: []const i64) Error!?Buffer {
    const dtype = encoding.dtype(output_type) orelse return error.UnsupportedElementType;
    return wrap(mlx_call.bufferArgsort(src.handle, dimension, dtype, output_dims), error.CommandSubmissionFailed);
}

/// Takes values from this MLX/Metal buffer using indices along one axis.
pub fn takeAlongAxis(src: Buffer, indices: Buffer, dimension: i64, output_dims: []const i64) Error!?Buffer {
    return wrap(mlx_call.bufferTakeAlongAxis(src.handle, indices.handle, dimension, output_dims), error.CommandSubmissionFailed);
}

fn wrap(handle: ?mlx_call.BufferHandle, err: Error) Error!?Buffer {
    return if (handle) |ptr| Buffer{ .handle = ptr } else err;
}

fn handles(buffers: []const Buffer) []const mlx_call.BufferHandle {
    return @ptrCast(buffers);
}
