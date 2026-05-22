const ir = @import("src/compiler/ir");

const opaque_ref = @import("buffer_opaque_ref.zig");

const Error = opaque_ref.Error;
const maybeHandle = opaque_ref.maybeHandle;
const ref = opaque_ref.ref;
const refs = opaque_ref.refs;

/// Reshapes an opaque buffer handle.
pub fn reshape(src: *anyopaque, dims: []const i64) Error!?*anyopaque { return maybeHandle(try ref(src).reshape(dims)); }
/// Transposes an opaque buffer handle.
pub fn transpose(src: *anyopaque, permutation: []const i64) Error!?*anyopaque { return maybeHandle(try ref(src).transpose(permutation)); }
/// Broadcasts an opaque buffer handle into explicit output dimensions.
pub fn broadcastInDim(src: *anyopaque, broadcast_dimensions: []const i64, output_dims: []const i64) Error!?*anyopaque { return maybeHandle(try ref(src).broadcastInDim(broadcast_dimensions, output_dims)); }
/// Slices an opaque buffer handle.
pub fn slice(src: *anyopaque, start_indices: []const i64, limit_indices: []const i64, strides: []const i64, output_dims: []const i64) Error!?*anyopaque { return maybeHandle(try ref(src).slice(start_indices, limit_indices, strides, output_dims)); }
/// Dynamically slices an opaque buffer handle.
pub fn dynamicSlice(src: *anyopaque, start_buffers: []const *anyopaque, slice_sizes: []const i64, output_dims: []const i64) Error!?*anyopaque { return maybeHandle(try ref(src).dynamicSlice(refs(start_buffers), slice_sizes, output_dims)); }
/// Dynamically updates an opaque buffer handle.
pub fn dynamicUpdateSlice(src: *anyopaque, update: *anyopaque, start_buffers: []const *anyopaque, output_dims: []const i64) Error!?*anyopaque { return maybeHandle(try ref(src).dynamicUpdateSlice(ref(update), refs(start_buffers), output_dims)); }
/// Pads an opaque buffer handle.
pub fn pad(src: *anyopaque, padding_value: *anyopaque, edge_padding_low: []const i64, edge_padding_high: []const i64, interior_padding: []const i64, output_dims: []const i64) Error!?*anyopaque { return maybeHandle(try ref(src).pad(ref(padding_value), edge_padding_low, edge_padding_high, interior_padding, output_dims)); }
/// Reverses dimensions of an opaque buffer handle.
pub fn reverse(src: *anyopaque, dimensions: []const i64, output_dims: []const i64) Error!?*anyopaque { return maybeHandle(try ref(src).reverse(dimensions, output_dims)); }
/// Concatenates two opaque buffer handles.
pub fn concatenate(lhs: *anyopaque, rhs: *anyopaque, dimension: i64, output_dims: []const i64) Error!?*anyopaque { return maybeHandle(try opaque_ref.Ref.concatenate(ref(lhs), ref(rhs), dimension, output_dims)); }
/// Gathers from an opaque operand handle along one axis.
pub fn gatherAxis(operand: *anyopaque, indices: *anyopaque, axis: i64, index_vector_dim: i64, output_dims: []const i64) Error!?*anyopaque { return maybeHandle(try opaque_ref.Ref.gatherAxis(ref(operand), ref(indices), axis, index_vector_dim, output_dims)); }
/// Gathers from an opaque operand handle using explicit dimension-number metadata.
pub fn gather(operand: *anyopaque, indices: *anyopaque, start_index_map: []const i64, collapsed_slice_dims: []const i64, operand_batching_dims: []const i64, start_indices_batching_dims: []const i64, index_vector_dim: i64, slice_sizes: []const i64, offset_dims: []const i64, output_dims: []const i64) Error!?*anyopaque { return maybeHandle(try opaque_ref.Ref.gather(ref(operand), ref(indices), start_index_map, collapsed_slice_dims, operand_batching_dims, start_indices_batching_dims, index_vector_dim, slice_sizes, offset_dims, output_dims)); }
/// Scatters updates into an opaque operand handle along one axis.
pub fn scatterAxis(operand: *anyopaque, indices: *anyopaque, updates: *anyopaque, axis: i64, index_vector_dim: i64, update_kind: ir.ScatterUpdateKind, output_dims: []const i64) Error!?*anyopaque { return maybeHandle(try opaque_ref.Ref.scatterAxis(ref(operand), ref(indices), ref(updates), axis, index_vector_dim, update_kind, output_dims)); }
/// Scatters updates into an opaque operand handle using explicit dimension-number metadata.
pub fn scatter(operand: *anyopaque, indices: *anyopaque, updates: *anyopaque, scatter_dims_to_operand_dims: []const i64, inserted_window_dims: []const i64, update_window_dims: []const i64, input_batching_dims: []const i64, scatter_indices_batching_dims: []const i64, index_vector_dim: i64, update_kind: ir.ScatterUpdateKind, output_dims: []const i64) Error!?*anyopaque { return maybeHandle(try opaque_ref.Ref.scatter(ref(operand), ref(indices), ref(updates), scatter_dims_to_operand_dims, inserted_window_dims, update_window_dims, input_batching_dims, scatter_indices_batching_dims, index_vector_dim, update_kind, output_dims)); }
/// Sorts an opaque buffer handle along one dimension.
pub fn sort(src: *anyopaque, dimension: i64, output_dims: []const i64) Error!?*anyopaque { return maybeHandle(try ref(src).sort(dimension, output_dims)); }
/// Returns sorted indices for an opaque buffer handle.
pub fn argsort(src: *anyopaque, dimension: i64, output_type: ir.BufferType, output_dims: []const i64) Error!?*anyopaque { return maybeHandle(try ref(src).argsort(dimension, output_type, output_dims)); }
/// Takes values from an opaque buffer handle using indices along one axis.
pub fn takeAlongAxis(src: *anyopaque, indices: *anyopaque, dimension: i64, output_dims: []const i64) Error!?*anyopaque { return maybeHandle(try ref(src).takeAlongAxis(ref(indices), dimension, output_dims)); }
