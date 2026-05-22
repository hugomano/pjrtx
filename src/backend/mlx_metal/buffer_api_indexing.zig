const ir = @import("src/compiler/ir");

const indexing = @import("buffer_indexing.zig");

/// Builds indexing methods for the typed backend buffer owner.
pub fn Typed(comptime Buffer: type) type {
    return struct {
        /// Reshapes this MLX/Metal buffer.
        pub fn reshape(self: Buffer, dims: []const i64) indexing.Error!?Buffer { return indexing.reshape(self, dims); }
        /// Transposes this MLX/Metal buffer.
        pub fn transpose(self: Buffer, permutation: []const i64) indexing.Error!?Buffer { return indexing.transpose(self, permutation); }
        /// Broadcasts this MLX/Metal buffer into explicit output dimensions.
        pub fn broadcastInDim(self: Buffer, broadcast_dimensions: []const i64, output_dims: []const i64) indexing.Error!?Buffer { return indexing.broadcastInDim(self, broadcast_dimensions, output_dims); }
        /// Slices this MLX/Metal buffer.
        pub fn slice(self: Buffer, start_indices: []const i64, limit_indices: []const i64, strides: []const i64, output_dims: []const i64) indexing.Error!?Buffer { return indexing.slice(self, start_indices, limit_indices, strides, output_dims); }
        /// Dynamically slices this MLX/Metal buffer.
        pub fn dynamicSlice(self: Buffer, start_buffers: []const Buffer, slice_sizes: []const i64, output_dims: []const i64) indexing.Error!?Buffer { return indexing.dynamicSlice(self, start_buffers, slice_sizes, output_dims); }
        /// Dynamically updates this MLX/Metal buffer.
        pub fn dynamicUpdateSlice(self: Buffer, update: Buffer, start_buffers: []const Buffer, output_dims: []const i64) indexing.Error!?Buffer { return indexing.dynamicUpdateSlice(self, update, start_buffers, output_dims); }
        /// Pads this MLX/Metal buffer.
        pub fn pad(self: Buffer, padding_value: Buffer, edge_padding_low: []const i64, edge_padding_high: []const i64, interior_padding: []const i64, output_dims: []const i64) indexing.Error!?Buffer { return indexing.pad(self, padding_value, edge_padding_low, edge_padding_high, interior_padding, output_dims); }
        /// Reverses dimensions of this MLX/Metal buffer.
        pub fn reverse(self: Buffer, dimensions: []const i64, output_dims: []const i64) indexing.Error!?Buffer { return indexing.reverse(self, dimensions, output_dims); }
        /// Concatenates this buffer with another MLX/Metal buffer.
        pub fn concatenate(lhs: Buffer, rhs: Buffer, dimension: i64, output_dims: []const i64) indexing.Error!?Buffer { return indexing.concatenate(lhs, rhs, dimension, output_dims); }
        /// Gathers from this MLX/Metal buffer along one axis.
        pub fn gatherAxis(operand: Buffer, indices: Buffer, axis: i64, index_vector_dim: i64, output_dims: []const i64) indexing.Error!?Buffer { return indexing.gatherAxis(operand, indices, axis, index_vector_dim, output_dims); }
        /// Gathers from this MLX/Metal buffer using explicit dimension-number metadata.
        pub fn gather(operand: Buffer, indices: Buffer, start_index_map: []const i64, collapsed_slice_dims: []const i64, operand_batching_dims: []const i64, start_indices_batching_dims: []const i64, index_vector_dim: i64, slice_sizes: []const i64, offset_dims: []const i64, output_dims: []const i64) indexing.Error!?Buffer { return indexing.gather(operand, indices, start_index_map, collapsed_slice_dims, operand_batching_dims, start_indices_batching_dims, index_vector_dim, slice_sizes, offset_dims, output_dims); }
        /// Scatters updates into this MLX/Metal buffer along one axis.
        pub fn scatterAxis(operand: Buffer, indices: Buffer, updates: Buffer, axis: i64, index_vector_dim: i64, update_kind: ir.ScatterUpdateKind, output_dims: []const i64) indexing.Error!?Buffer { return indexing.scatterAxis(operand, indices, updates, axis, index_vector_dim, update_kind, output_dims); }
        /// Scatters updates into this MLX/Metal buffer using explicit dimension-number metadata.
        pub fn scatter(operand: Buffer, indices: Buffer, updates: Buffer, scatter_dims_to_operand_dims: []const i64, inserted_window_dims: []const i64, update_window_dims: []const i64, input_batching_dims: []const i64, scatter_indices_batching_dims: []const i64, index_vector_dim: i64, update_kind: ir.ScatterUpdateKind, output_dims: []const i64) indexing.Error!?Buffer { return indexing.scatter(operand, indices, updates, scatter_dims_to_operand_dims, inserted_window_dims, update_window_dims, input_batching_dims, scatter_indices_batching_dims, index_vector_dim, update_kind, output_dims); }
        /// Sorts this MLX/Metal buffer along one dimension.
        pub fn sort(self: Buffer, dimension: i64, output_dims: []const i64) indexing.Error!?Buffer { return indexing.sort(self, dimension, output_dims); }
        /// Returns sorted indices for this MLX/Metal buffer.
        pub fn argsort(self: Buffer, dimension: i64, output_type: ir.BufferType, output_dims: []const i64) indexing.Error!?Buffer { return indexing.argsort(self, dimension, output_type, output_dims); }
        /// Takes values from this MLX/Metal buffer using indices along one axis.
        pub fn takeAlongAxis(self: Buffer, indices: Buffer, dimension: i64, output_dims: []const i64) indexing.Error!?Buffer { return indexing.takeAlongAxis(self, indices, dimension, output_dims); }
    };
}
