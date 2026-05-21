const ir = @import("src/compiler/ir");
const mlx_call = @import("mlx_call.zig");

/// Typed owner-side wrapper for an MLX/Metal buffer handle.
pub const Buffer = struct {
    handle: mlx_call.BufferHandle,

    /// Wraps a backend-opaque buffer handle without taking ownership.
    pub fn fromHandle(handle: *anyopaque) Buffer {
        return .{ .handle = @ptrCast(handle) };
    }

    /// Returns the backend-opaque representation used by the runtime boundary.
    pub fn toHandle(self: Buffer) *anyopaque {
        return @ptrCast(self.handle);
    }

    /// Copies host bytes into a typed MLX/Metal buffer.
    pub fn fromHost(device_local_hardware_id: i32, element_type: ir.BufferType, dims: []const i64, src: []const u8) Error!?Buffer {
        if (src.len == 0) return null;
        const dtype = Dtype.fromIr(element_type) orelse return error.UnsupportedElementType;
        return wrap(mlx_call.bufferFromHost(device_local_hardware_id, dtype, dims, src), error.BufferAllocationFailed);
    }

    /// Allocates a zero-initialized MLX/Metal buffer.
    pub fn zeros(device_local_hardware_id: i32, element_type: ir.BufferType, dims: []const i64) Error!?Buffer {
        const dtype = Dtype.fromIr(element_type) orelse return error.UnsupportedElementType;
        return wrap(mlx_call.bufferZeros(device_local_hardware_id, dtype, dims), error.BufferAllocationFailed);
    }

    /// Creates an MLX/Metal iota buffer.
    pub fn iota(device_local_hardware_id: i32, element_type: ir.BufferType, dims: []const i64, iota_dimension: i64) Error!?Buffer {
        const dtype = Dtype.fromIr(element_type) orelse return error.UnsupportedElementType;
        return wrap(mlx_call.bufferIota(device_local_hardware_id, dtype, dims, iota_dimension), error.CommandSubmissionFailed);
    }

    /// Creates an MLX/Metal partition-id scalar buffer.
    pub fn partitionId(device_local_hardware_id: i32, element_type: ir.BufferType, partition_id: u32) Error!?Buffer {
        const dtype = Dtype.fromIr(element_type) orelse return error.UnsupportedElementType;
        return wrap(mlx_call.bufferPartitionId(device_local_hardware_id, dtype, partition_id), error.CommandSubmissionFailed);
    }

    /// Clones this buffer into a new MLX/Metal buffer.
    pub fn clone(self: Buffer) Error!?Buffer {
        return wrap(mlx_call.bufferClone(self.handle), error.CommandSubmissionFailed);
    }

    /// Creates a zero buffer with the same MLX/Metal shape and type as this buffer.
    pub fn zeroLike(self: Buffer) Error!?Buffer {
        return wrap(mlx_call.bufferZeroLike(self.handle), error.CommandSubmissionFailed);
    }

    /// Combines real and imaginary buffers into a complex MLX/Metal buffer.
    pub fn complex(real: Buffer, imag: Buffer, output_dims: []const i64) Error!?Buffer {
        return wrap(mlx_call.bufferComplex(real.handle, imag.handle, output_dims), error.CommandSubmissionFailed);
    }

    /// Extracts the real part of this complex MLX/Metal buffer.
    pub fn realPart(self: Buffer, output_dims: []const i64) Error!?Buffer {
        return wrap(mlx_call.bufferReal(self.handle, output_dims), error.CommandSubmissionFailed);
    }

    /// Extracts the imaginary part of this complex MLX/Metal buffer.
    pub fn imagPart(self: Buffer, output_dims: []const i64) Error!?Buffer {
        return wrap(mlx_call.bufferImag(self.handle, output_dims), error.CommandSubmissionFailed);
    }

    /// Converts this MLX/Metal buffer to another element type.
    pub fn convert(self: Buffer, output_type: ir.BufferType) Error!?Buffer {
        const dtype = Dtype.fromIr(output_type) orelse return error.UnsupportedElementType;
        return wrap(mlx_call.bufferAstype(self.handle, dtype), error.CommandSubmissionFailed);
    }

    /// Reinterprets this MLX/Metal buffer with a new element type and shape.
    pub fn bitcast(self: Buffer, output_type: ir.BufferType, output_dims: []const i64) Error!?Buffer {
        const dtype = Dtype.fromIr(output_type) orelse return error.UnsupportedElementType;
        return wrap(mlx_call.bufferViewDtype(self.handle, dtype, output_dims), error.CommandSubmissionFailed);
    }

    /// Applies an elementwise binary operation to two MLX/Metal buffers.
    pub fn binary(lhs: Buffer, rhs: Buffer, op: ir.ElementwiseBinaryOp) Error!?Buffer {
        const code = BinaryOp.fromIr(op) orelse return error.CommandSubmissionFailed;
        return wrap(mlx_call.bufferBinary(lhs.handle, rhs.handle, code), error.CommandSubmissionFailed);
    }

    /// Applies an elementwise binary operation with explicit output dimensions.
    pub fn binaryWithOutputDims(lhs: Buffer, rhs: Buffer, op: ir.ElementwiseBinaryOp, output_dims: []const i64) Error!?Buffer {
        const code = BinaryOp.fromIr(op) orelse return error.CommandSubmissionFailed;
        return wrap(mlx_call.bufferBinaryOut(lhs.handle, rhs.handle, code, output_dims), error.CommandSubmissionFailed);
    }

    /// Applies an elementwise unary operation to this MLX/Metal buffer.
    pub fn unary(self: Buffer, op: ir.ElementwiseUnaryOp) Error!?Buffer {
        const code = UnaryOp.fromIr(op) orelse return error.CommandSubmissionFailed;
        return wrap(mlx_call.bufferUnary(self.handle, code), error.CommandSubmissionFailed);
    }

    /// Reshapes this MLX/Metal buffer.
    pub fn reshape(self: Buffer, dims: []const i64) Error!?Buffer {
        return wrap(mlx_call.bufferReshape(self.handle, dims), error.CommandSubmissionFailed);
    }

    /// Transposes this MLX/Metal buffer.
    pub fn transpose(self: Buffer, permutation: []const i64) Error!?Buffer {
        return wrap(mlx_call.bufferTranspose(self.handle, permutation), error.CommandSubmissionFailed);
    }

    /// Broadcasts this MLX/Metal buffer into explicit output dimensions.
    pub fn broadcastInDim(self: Buffer, broadcast_dimensions: []const i64, output_dims: []const i64) Error!?Buffer {
        return wrap(mlx_call.bufferBroadcastInDim(self.handle, broadcast_dimensions, output_dims), error.CommandSubmissionFailed);
    }

    /// Slices this MLX/Metal buffer.
    pub fn slice(self: Buffer, start_indices: []const i64, limit_indices: []const i64, strides: []const i64, output_dims: []const i64) Error!?Buffer {
        return wrap(mlx_call.bufferSlice(self.handle, start_indices, limit_indices, strides, output_dims), error.CommandSubmissionFailed);
    }

    /// Dynamically slices this MLX/Metal buffer.
    pub fn dynamicSlice(self: Buffer, start_buffers: []const Buffer, slice_sizes: []const i64, output_dims: []const i64) Error!?Buffer {
        if (start_buffers.len != slice_sizes.len) return error.ShapeMismatch;
        return wrap(mlx_call.bufferDynamicSlice(self.handle, handles(start_buffers), slice_sizes, output_dims), error.CommandSubmissionFailed);
    }

    /// Dynamically updates this MLX/Metal buffer.
    pub fn dynamicUpdateSlice(self: Buffer, update: Buffer, start_buffers: []const Buffer, output_dims: []const i64) Error!?Buffer {
        return wrap(mlx_call.bufferDynamicUpdateSlice(self.handle, update.handle, handles(start_buffers), output_dims), error.CommandSubmissionFailed);
    }

    /// Pads this MLX/Metal buffer.
    pub fn pad(self: Buffer, padding_value: Buffer, edge_padding_low: []const i64, edge_padding_high: []const i64, interior_padding: []const i64, output_dims: []const i64) Error!?Buffer {
        return wrap(mlx_call.bufferPad(self.handle, padding_value.handle, edge_padding_low, edge_padding_high, interior_padding, output_dims), error.CommandSubmissionFailed);
    }

    /// Reverses dimensions of this MLX/Metal buffer.
    pub fn reverse(self: Buffer, dimensions: []const i64, output_dims: []const i64) Error!?Buffer {
        return wrap(mlx_call.bufferReverse(self.handle, dimensions, output_dims), error.CommandSubmissionFailed);
    }

    /// Concatenates this buffer with another MLX/Metal buffer.
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
        return wrap(mlx_call.bufferScatterAxis(operand.handle, indices.handle, updates.handle, axis, index_vector_dim, ScatterUpdate.fromIr(update_kind), output_dims), error.CommandSubmissionFailed);
    }

    /// Scatters updates into this MLX/Metal buffer using explicit dimension-number metadata.
    pub fn scatter(operand: Buffer, indices: Buffer, updates: Buffer, scatter_dims_to_operand_dims: []const i64, inserted_window_dims: []const i64, update_window_dims: []const i64, input_batching_dims: []const i64, scatter_indices_batching_dims: []const i64, index_vector_dim: i64, update_kind: ir.ScatterUpdateKind, output_dims: []const i64) Error!?Buffer {
        return wrap(mlx_call.bufferScatter(operand.handle, indices.handle, updates.handle, scatter_dims_to_operand_dims, inserted_window_dims, update_window_dims, input_batching_dims, scatter_indices_batching_dims, index_vector_dim, ScatterUpdate.fromIr(update_kind), output_dims), error.CommandSubmissionFailed);
    }

    /// Sorts this MLX/Metal buffer along one dimension.
    pub fn sort(self: Buffer, dimension: i64, output_dims: []const i64) Error!?Buffer {
        return wrap(mlx_call.bufferSort(self.handle, dimension, output_dims), error.CommandSubmissionFailed);
    }

    /// Returns sorted indices for this MLX/Metal buffer.
    pub fn argsort(self: Buffer, dimension: i64, output_type: ir.BufferType, output_dims: []const i64) Error!?Buffer {
        const dtype = Dtype.fromIr(output_type) orelse return error.UnsupportedElementType;
        return wrap(mlx_call.bufferArgsort(self.handle, dimension, dtype, output_dims), error.CommandSubmissionFailed);
    }

    /// Takes values from this MLX/Metal buffer using indices along one axis.
    pub fn takeAlongAxis(self: Buffer, indices: Buffer, dimension: i64, output_dims: []const i64) Error!?Buffer {
        return wrap(mlx_call.bufferTakeAlongAxis(self.handle, indices.handle, dimension, output_dims), error.CommandSubmissionFailed);
    }

    /// Runs dot-general with explicit batch and contracting dimensions.
    pub fn dotGeneral(lhs: Buffer, rhs: Buffer, lhs_batch_dimensions: []const i64, rhs_batch_dimensions: []const i64, lhs_contracting_dimensions: []const i64, rhs_contracting_dimensions: []const i64, output_dims: []const i64) Error!?Buffer {
        return wrap(mlx_call.bufferDotGeneral(lhs.handle, rhs.handle, lhs_batch_dimensions, rhs_batch_dimensions, lhs_contracting_dimensions, rhs_contracting_dimensions, output_dims), error.CommandSubmissionFailed);
    }

    /// Runs convolution with explicit window metadata.
    pub fn convolution(lhs: Buffer, rhs: Buffer, window_strides: []const i64, padding_low: []const i64, padding_high: []const i64, lhs_dilation: []const i64, rhs_dilation: []const i64, window_reversal: []const bool, feature_group_count: i64, output_dims: []const i64) Error!?Buffer {
        return wrap(mlx_call.bufferConvolution(lhs.handle, rhs.handle, window_strides, padding_low, padding_high, lhs_dilation, rhs_dilation, window_reversal, feature_group_count, output_dims), error.CommandSubmissionFailed);
    }

    /// Runs Cholesky decomposition on this MLX/Metal buffer.
    pub fn cholesky(self: Buffer, lower: bool, output_dims: []const i64) Error!?Buffer {
        return wrap(mlx_call.bufferCholesky(self.handle, lower, output_dims), error.CommandSubmissionFailed);
    }

    /// Runs triangular solve with this buffer as the coefficient matrix.
    pub fn triangularSolve(a: Buffer, b: Buffer, left_side: bool, lower: bool, unit_diagonal: bool, transpose_a: ir.TriangularSolveTranspose, output_dims: []const i64) Error!?Buffer {
        return wrap(mlx_call.bufferTriangularSolve(a.handle, b.handle, left_side, lower, unit_diagonal, TriangularTranspose.fromIr(transpose_a), output_dims), error.CommandSubmissionFailed);
    }

    /// Runs an FFT operation on this MLX/Metal buffer.
    pub fn fft(self: Buffer, fft_kind: ir.FftKind, fft_lengths: []const i64, output_dims: []const i64) Error!?Buffer {
        return wrap(mlx_call.bufferFft(self.handle, FftKind.fromIr(fft_kind), fft_lengths, output_dims), error.CommandSubmissionFailed);
    }

    /// Runs a random distribution operation using this buffer and another bound buffer.
    pub fn rng(a: Buffer, b: Buffer, distribution: ir.RngDistribution, output_type: ir.BufferType, output_dims: []const i64) Error!?Buffer {
        const dtype = Dtype.fromIr(output_type) orelse return error.UnsupportedElementType;
        return wrap(mlx_call.bufferRng(a.handle, b.handle, RngDistribution.fromIr(distribution), dtype, output_dims), error.CommandSubmissionFailed);
    }

    /// Runs random bit generation and returns updated state plus bits.
    pub fn rngBitGenerator(state: Buffer, output_type: ir.BufferType, output_dims: []const i64) Error!?Pair {
        const dtype = Dtype.fromIr(output_type) orelse return error.UnsupportedElementType;
        const pair = mlx_call.bufferRngBitGenerator(state.handle, dtype, output_dims) orelse return error.CommandSubmissionFailed;
        return .{
            .first = .{ .handle = pair.first },
            .second = .{ .handle = pair.second },
        };
    }

    /// Runs a reduction operation on this MLX/Metal buffer.
    pub fn reduce(self: Buffer, op: ir.PlanInstructionKind, dimensions: []const i64, output_dims: []const i64) Error!?Buffer {
        const code = ReduceOp.fromPlan(op) orelse return error.CommandSubmissionFailed;
        return wrap(mlx_call.bufferReduce(self.handle, code, dimensions, output_dims), error.CommandSubmissionFailed);
    }

    /// Runs max-reduction and returns values plus indices.
    pub fn reduceMaxWithIndices(values: Buffer, indices: Buffer, dimensions: []const i64, output_dims: []const i64) Error!?Pair {
        const pair = mlx_call.bufferReduceMaxWithIndices(values.handle, indices.handle, dimensions, output_dims) orelse return error.CommandSubmissionFailed;
        return .{
            .first = .{ .handle = pair.first },
            .second = .{ .handle = pair.second },
        };
    }

    /// Runs a windowed reduction operation on this MLX/Metal buffer.
    pub fn reduceWindow(self: Buffer, op: ir.PlanInstructionKind, window_dimensions: []const i64, window_strides: []const i64, base_dilations: []const i64, window_dilations: []const i64, padding_low: []const i64, padding_high: []const i64, output_dims: []const i64) Error!?Buffer {
        const code = ReduceOp.fromWindowPlan(op) orelse return error.CommandSubmissionFailed;
        return wrap(mlx_call.bufferReduceWindow(self.handle, code, window_dimensions, window_strides, base_dilations, window_dilations, padding_low, padding_high, output_dims), error.CommandSubmissionFailed);
    }

    /// Runs windowed max-reduction and returns values plus indices.
    pub fn reduceWindowMaxWithIndices(values: Buffer, indices: Buffer, window_dimensions: []const i64, window_strides: []const i64, base_dilations: []const i64, window_dilations: []const i64, padding_low: []const i64, padding_high: []const i64, output_dims: []const i64) Error!?Pair {
        const pair = mlx_call.bufferReduceWindowMaxWithIndices(values.handle, indices.handle, window_dimensions, window_strides, base_dilations, window_dilations, padding_low, padding_high, output_dims) orelse return error.CommandSubmissionFailed;
        return .{
            .first = .{ .handle = pair.first },
            .second = .{ .handle = pair.second },
        };
    }

    /// Compares this MLX/Metal buffer with another buffer.
    pub fn compare(lhs: Buffer, rhs: Buffer, direction: ir.CompareOp, output_dims: []const i64) Error!?Buffer {
        return wrap(mlx_call.bufferCompare(lhs.handle, rhs.handle, CompareOp.fromIr(direction), output_dims), error.CommandSubmissionFailed);
    }

    /// Selects between two MLX/Metal buffers using this predicate buffer.
    pub fn select(pred: Buffer, on_true: Buffer, on_false: Buffer, output_dims: []const i64) Error!?Buffer {
        return wrap(mlx_call.bufferSelect(pred.handle, on_true.handle, on_false.handle, output_dims), error.CommandSubmissionFailed);
    }

    /// Clamps this value buffer between minimum and maximum buffers.
    pub fn clamp(min: Buffer, value: Buffer, max: Buffer, output_dims: []const i64) Error!?Buffer {
        return wrap(mlx_call.bufferClamp(min.handle, value.handle, max.handle, output_dims), error.CommandSubmissionFailed);
    }

    /// Runs the while-compare-add backend fast path.
    pub fn whileF32CompareAdd(state: Buffer, limit: Buffer, step: Buffer, compare_direction: ir.CompareOp, update_op: ir.ElementwiseBinaryOp, output_dims: []const i64, max_iterations: u64) Error!?Buffer {
        const update_code = BinaryOp.fromIr(update_op) orelse return error.CommandSubmissionFailed;
        return wrap(mlx_call.bufferWhileF32CompareAdd(state.handle, limit.handle, step.handle, CompareOp.fromIr(compare_direction), update_code, output_dims, max_iterations), error.CommandSubmissionFailed);
    }

    /// Runs the built-in MLX/Metal binary-add custom call.
    pub fn customBinaryAddF32(lhs: Buffer, rhs: Buffer) Error!?Buffer {
        return wrap(mlx_call.customCallBinaryAddF32(lhs.handle, rhs.handle), error.CommandSubmissionFailed);
    }

    /// Runs the built-in MLX/Metal scaled dot-product attention custom call.
    pub fn customScaledDotProductAttention(q: Buffer, k: Buffer, v: Buffer, token_index: Buffer) Error!?Buffer {
        return wrap(mlx_call.customCallScaledDotProductAttention(q.handle, k.handle, v.handle, token_index.handle), error.CommandSubmissionFailed);
    }

    /// Forces this MLX/Metal buffer to evaluate.
    pub fn eval(self: Buffer) Error!void {
        if (!mlx_call.bufferEval(self.handle)) return error.CommandSubmissionFailed;
    }

    /// Copies this MLX/Metal buffer into host memory.
    pub fn copyToHost(self: Buffer, dst: []u8) Error!void {
        if (!mlx_call.bufferCopyToHost(self.handle, dst)) return error.BufferCopyFailed;
    }

    /// Reports whether this buffer still owns a host shadow allocation.
    pub fn hasHostShadow(self: Buffer) bool {
        return mlx_call.bufferHasHostShadow(self.handle);
    }

    /// Destroys this MLX/Metal buffer handle.
    pub fn destroy(self: Buffer) void {
        mlx_call.bufferDestroy(self.handle);
    }
};


/// Returns whether the MLX/Metal shim accepts this compiler IR element type.
pub fn supportsElementType(element_type: ir.BufferType) bool {
    return Dtype.fromIr(element_type) != null;
}

/// Returns whether the MLX/Metal shim accepts this elementwise unary operation.
pub fn supportsUnaryOp(op: ir.ElementwiseUnaryOp) bool {
    return UnaryOp.fromIr(op) != null;
}

/// Pair of typed MLX/Metal buffers returned by two-output operations.
pub const Pair = struct {
    /// First returned typed buffer.
    first: Buffer,
    /// Second returned typed buffer.
    second: Buffer,
};

/// Opaque backend-buffer pair returned by max-reduction-with-indices.
pub const ReduceMaxWithIndicesResult = struct {
    /// Device buffer containing reduced values.
    values: *anyopaque,
    /// Device buffer containing selected indices.
    indices: *anyopaque,
};

/// Opaque backend-buffer pair returned by windowed max-reduction-with-indices.
pub const ReduceWindowMaxWithIndicesResult = struct {
    /// Device buffer containing reduced values.
    values: *anyopaque,
    /// Device buffer containing selected indices.
    indices: *anyopaque,
};

/// Opaque backend-buffer pair returned by random bit generation.
pub const RngBitGeneratorResult = struct {
    /// Device buffer containing the updated RNG state.
    state: *anyopaque,
    /// Device buffer containing generated random bits.
    bits: *anyopaque,
};

/// Runtime-facing opaque buffer operations owned by the MLX/Metal buffer layer.
pub const Opaque = struct {
    /// Copies host bytes into a device-resident opaque buffer handle.
    pub fn fromHost(device_local_hardware_id: i32, element_type: ir.BufferType, dims: []const i64, src: []const u8) Error!?*anyopaque {
        return maybeHandle(try Buffer.fromHost(device_local_hardware_id, element_type, dims, src));
    }

    /// Allocates a zero-initialized device-resident opaque buffer handle.
    pub fn zeros(device_local_hardware_id: i32, element_type: ir.BufferType, dims: []const i64) Error!?*anyopaque {
        return maybeHandle(try Buffer.zeros(device_local_hardware_id, element_type, dims));
    }

    /// Creates an opaque iota buffer handle.
    pub fn iota(device_local_hardware_id: i32, element_type: ir.BufferType, dims: []const i64, iota_dimension: i64) Error!?*anyopaque {
        return maybeHandle(try Buffer.iota(device_local_hardware_id, element_type, dims, iota_dimension));
    }

    /// Creates an opaque partition-id scalar buffer handle.
    pub fn partitionId(device_local_hardware_id: i32, element_type: ir.BufferType, partition_id: u32) Error!?*anyopaque {
        return maybeHandle(try Buffer.partitionId(device_local_hardware_id, element_type, partition_id));
    }

    /// Clones an opaque buffer handle.
    pub fn clone(src: *anyopaque) Error!?*anyopaque {
        return maybeHandle(try ref(src).clone());
    }

    /// Creates an opaque zero buffer with the same shape and type as another handle.
    pub fn zeroLike(src: *anyopaque) Error!?*anyopaque {
        return maybeHandle(try ref(src).zeroLike());
    }

    /// Combines real and imaginary opaque handles into a complex buffer.
    pub fn complex(real: *anyopaque, imag: *anyopaque, output_dims: []const i64) Error!?*anyopaque {
        return maybeHandle(try Buffer.complex(ref(real), ref(imag), output_dims));
    }

    /// Extracts the real part from an opaque complex buffer handle.
    pub fn realPart(src: *anyopaque, output_dims: []const i64) Error!?*anyopaque {
        return maybeHandle(try ref(src).realPart(output_dims));
    }

    /// Extracts the imaginary part from an opaque complex buffer handle.
    pub fn imagPart(src: *anyopaque, output_dims: []const i64) Error!?*anyopaque {
        return maybeHandle(try ref(src).imagPart(output_dims));
    }

    /// Converts an opaque buffer handle to another element type.
    pub fn convert(src: *anyopaque, output_type: ir.BufferType) Error!?*anyopaque {
        return maybeHandle(try ref(src).convert(output_type));
    }

    /// Reinterprets an opaque buffer handle with a new element type and shape.
    pub fn bitcast(src: *anyopaque, output_type: ir.BufferType, output_dims: []const i64) Error!?*anyopaque {
        return maybeHandle(try ref(src).bitcast(output_type, output_dims));
    }

    /// Applies an elementwise binary operation to two opaque buffer handles.
    pub fn binary(lhs: *anyopaque, rhs: *anyopaque, op: ir.ElementwiseBinaryOp) Error!?*anyopaque {
        return maybeHandle(try Buffer.binary(ref(lhs), ref(rhs), op));
    }

    /// Applies an elementwise binary operation with explicit output dimensions.
    pub fn binaryWithOutputDims(lhs: *anyopaque, rhs: *anyopaque, op: ir.ElementwiseBinaryOp, output_dims: []const i64) Error!?*anyopaque {
        return maybeHandle(try Buffer.binaryWithOutputDims(ref(lhs), ref(rhs), op, output_dims));
    }

    /// Applies an elementwise unary operation to an opaque buffer handle.
    pub fn unary(src: *anyopaque, op: ir.ElementwiseUnaryOp) Error!?*anyopaque {
        return maybeHandle(try ref(src).unary(op));
    }

    /// Reshapes an opaque buffer handle.
    pub fn reshape(src: *anyopaque, dims: []const i64) Error!?*anyopaque {
        return maybeHandle(try ref(src).reshape(dims));
    }

    /// Transposes an opaque buffer handle.
    pub fn transpose(src: *anyopaque, permutation: []const i64) Error!?*anyopaque {
        return maybeHandle(try ref(src).transpose(permutation));
    }

    /// Broadcasts an opaque buffer handle into explicit output dimensions.
    pub fn broadcastInDim(src: *anyopaque, broadcast_dimensions: []const i64, output_dims: []const i64) Error!?*anyopaque {
        return maybeHandle(try ref(src).broadcastInDim(broadcast_dimensions, output_dims));
    }

    /// Slices an opaque buffer handle.
    pub fn slice(src: *anyopaque, start_indices: []const i64, limit_indices: []const i64, strides: []const i64, output_dims: []const i64) Error!?*anyopaque {
        return maybeHandle(try ref(src).slice(start_indices, limit_indices, strides, output_dims));
    }

    /// Dynamically slices an opaque buffer handle.
    pub fn dynamicSlice(src: *anyopaque, start_buffers: []const *anyopaque, slice_sizes: []const i64, output_dims: []const i64) Error!?*anyopaque {
        return maybeHandle(try ref(src).dynamicSlice(refs(start_buffers), slice_sizes, output_dims));
    }

    /// Dynamically updates an opaque buffer handle.
    pub fn dynamicUpdateSlice(src: *anyopaque, update: *anyopaque, start_buffers: []const *anyopaque, output_dims: []const i64) Error!?*anyopaque {
        return maybeHandle(try ref(src).dynamicUpdateSlice(ref(update), refs(start_buffers), output_dims));
    }

    /// Pads an opaque buffer handle.
    pub fn pad(src: *anyopaque, padding_value: *anyopaque, edge_padding_low: []const i64, edge_padding_high: []const i64, interior_padding: []const i64, output_dims: []const i64) Error!?*anyopaque {
        return maybeHandle(try ref(src).pad(ref(padding_value), edge_padding_low, edge_padding_high, interior_padding, output_dims));
    }

    /// Reverses dimensions of an opaque buffer handle.
    pub fn reverse(src: *anyopaque, dimensions: []const i64, output_dims: []const i64) Error!?*anyopaque {
        return maybeHandle(try ref(src).reverse(dimensions, output_dims));
    }

    /// Concatenates two opaque buffer handles.
    pub fn concatenate(lhs: *anyopaque, rhs: *anyopaque, dimension: i64, output_dims: []const i64) Error!?*anyopaque {
        return maybeHandle(try Buffer.concatenate(ref(lhs), ref(rhs), dimension, output_dims));
    }

    /// Gathers from an opaque operand handle along one axis.
    pub fn gatherAxis(operand: *anyopaque, indices: *anyopaque, axis: i64, index_vector_dim: i64, output_dims: []const i64) Error!?*anyopaque {
        return maybeHandle(try Buffer.gatherAxis(ref(operand), ref(indices), axis, index_vector_dim, output_dims));
    }

    /// Gathers from an opaque operand handle using explicit dimension-number metadata.
    pub fn gather(operand: *anyopaque, indices: *anyopaque, start_index_map: []const i64, collapsed_slice_dims: []const i64, operand_batching_dims: []const i64, start_indices_batching_dims: []const i64, index_vector_dim: i64, slice_sizes: []const i64, offset_dims: []const i64, output_dims: []const i64) Error!?*anyopaque {
        return maybeHandle(try Buffer.gather(ref(operand), ref(indices), start_index_map, collapsed_slice_dims, operand_batching_dims, start_indices_batching_dims, index_vector_dim, slice_sizes, offset_dims, output_dims));
    }

    /// Scatters updates into an opaque operand handle along one axis.
    pub fn scatterAxis(operand: *anyopaque, indices: *anyopaque, updates: *anyopaque, axis: i64, index_vector_dim: i64, update_kind: ir.ScatterUpdateKind, output_dims: []const i64) Error!?*anyopaque {
        return maybeHandle(try Buffer.scatterAxis(ref(operand), ref(indices), ref(updates), axis, index_vector_dim, update_kind, output_dims));
    }

    /// Scatters updates into an opaque operand handle using explicit dimension-number metadata.
    pub fn scatter(operand: *anyopaque, indices: *anyopaque, updates: *anyopaque, scatter_dims_to_operand_dims: []const i64, inserted_window_dims: []const i64, update_window_dims: []const i64, input_batching_dims: []const i64, scatter_indices_batching_dims: []const i64, index_vector_dim: i64, update_kind: ir.ScatterUpdateKind, output_dims: []const i64) Error!?*anyopaque {
        return maybeHandle(try Buffer.scatter(ref(operand), ref(indices), ref(updates), scatter_dims_to_operand_dims, inserted_window_dims, update_window_dims, input_batching_dims, scatter_indices_batching_dims, index_vector_dim, update_kind, output_dims));
    }

    /// Sorts an opaque buffer handle along one dimension.
    pub fn sort(src: *anyopaque, dimension: i64, output_dims: []const i64) Error!?*anyopaque {
        return maybeHandle(try ref(src).sort(dimension, output_dims));
    }

    /// Returns sorted indices for an opaque buffer handle.
    pub fn argsort(src: *anyopaque, dimension: i64, output_type: ir.BufferType, output_dims: []const i64) Error!?*anyopaque {
        return maybeHandle(try ref(src).argsort(dimension, output_type, output_dims));
    }

    /// Takes values from an opaque buffer handle using indices along one axis.
    pub fn takeAlongAxis(src: *anyopaque, indices: *anyopaque, dimension: i64, output_dims: []const i64) Error!?*anyopaque {
        return maybeHandle(try ref(src).takeAlongAxis(ref(indices), dimension, output_dims));
    }

    /// Runs dot-general with opaque buffer handles.
    pub fn dotGeneral(lhs: *anyopaque, rhs: *anyopaque, lhs_batch_dimensions: []const i64, rhs_batch_dimensions: []const i64, lhs_contracting_dimensions: []const i64, rhs_contracting_dimensions: []const i64, output_dims: []const i64) Error!?*anyopaque {
        return maybeHandle(try Buffer.dotGeneral(ref(lhs), ref(rhs), lhs_batch_dimensions, rhs_batch_dimensions, lhs_contracting_dimensions, rhs_contracting_dimensions, output_dims));
    }

    /// Runs convolution with opaque buffer handles.
    pub fn convolution(lhs: *anyopaque, rhs: *anyopaque, window_strides: []const i64, padding_low: []const i64, padding_high: []const i64, lhs_dilation: []const i64, rhs_dilation: []const i64, window_reversal: []const bool, feature_group_count: i64, output_dims: []const i64) Error!?*anyopaque {
        return maybeHandle(try Buffer.convolution(ref(lhs), ref(rhs), window_strides, padding_low, padding_high, lhs_dilation, rhs_dilation, window_reversal, feature_group_count, output_dims));
    }

    /// Runs an FFT operation on an opaque buffer handle.
    pub fn fft(src: *anyopaque, kind: ir.FftKind, fft_lengths: []const i64, output_dims: []const i64) Error!?*anyopaque {
        return maybeHandle(try ref(src).fft(kind, fft_lengths, output_dims));
    }

    /// Runs a random distribution operation using opaque bound handles.
    pub fn rng(a: *anyopaque, b: *anyopaque, distribution: ir.RngDistribution, output_type: ir.BufferType, output_dims: []const i64) Error!?*anyopaque {
        return maybeHandle(try Buffer.rng(ref(a), ref(b), distribution, output_type, output_dims));
    }

    /// Runs random bit generation and returns opaque state plus bits handles.
    pub fn rngBitGenerator(state: *anyopaque, output_type: ir.BufferType, output_dims: []const i64) Error!?RngBitGeneratorResult {
        const pair = (try Buffer.rngBitGenerator(ref(state), output_type, output_dims)) orelse return null;
        return .{ .state = pair.first.toHandle(), .bits = pair.second.toHandle() };
    }

    /// Runs Cholesky decomposition on an opaque buffer handle.
    pub fn cholesky(src: *anyopaque, lower: bool, output_dims: []const i64) Error!?*anyopaque {
        return maybeHandle(try ref(src).cholesky(lower, output_dims));
    }

    /// Runs triangular solve with opaque buffer handles.
    pub fn triangularSolve(a: *anyopaque, b: *anyopaque, left_side: bool, lower: bool, unit_diagonal: bool, transpose_a: ir.TriangularSolveTranspose, output_dims: []const i64) Error!?*anyopaque {
        return maybeHandle(try Buffer.triangularSolve(ref(a), ref(b), left_side, lower, unit_diagonal, transpose_a, output_dims));
    }

    /// Runs a reduction operation on an opaque buffer handle.
    pub fn reduce(src: *anyopaque, op: ir.PlanInstructionKind, dimensions: []const i64, output_dims: []const i64) Error!?*anyopaque {
        return maybeHandle(try ref(src).reduce(op, dimensions, output_dims));
    }

    /// Runs max-reduction and returns opaque values plus indices handles.
    pub fn reduceMaxWithIndices(values: *anyopaque, indices: *anyopaque, dimensions: []const i64, output_dims: []const i64) Error!?ReduceMaxWithIndicesResult {
        const pair = (try Buffer.reduceMaxWithIndices(ref(values), ref(indices), dimensions, output_dims)) orelse return null;
        return .{ .values = pair.first.toHandle(), .indices = pair.second.toHandle() };
    }

    /// Runs a windowed reduction operation on an opaque buffer handle.
    pub fn reduceWindow(src: *anyopaque, op: ir.PlanInstructionKind, window_dimensions: []const i64, window_strides: []const i64, base_dilations: []const i64, window_dilations: []const i64, padding_low: []const i64, padding_high: []const i64, output_dims: []const i64) Error!?*anyopaque {
        return maybeHandle(try ref(src).reduceWindow(op, window_dimensions, window_strides, base_dilations, window_dilations, padding_low, padding_high, output_dims));
    }

    /// Runs windowed max-reduction and returns opaque values plus indices handles.
    pub fn reduceWindowMaxWithIndices(values: *anyopaque, indices: *anyopaque, window_dimensions: []const i64, window_strides: []const i64, base_dilations: []const i64, window_dilations: []const i64, padding_low: []const i64, padding_high: []const i64, output_dims: []const i64) Error!?ReduceWindowMaxWithIndicesResult {
        const pair = (try Buffer.reduceWindowMaxWithIndices(ref(values), ref(indices), window_dimensions, window_strides, base_dilations, window_dilations, padding_low, padding_high, output_dims)) orelse return null;
        return .{ .values = pair.first.toHandle(), .indices = pair.second.toHandle() };
    }

    /// Compares two opaque buffer handles.
    pub fn compare(lhs: *anyopaque, rhs: *anyopaque, direction: ir.CompareOp, output_dims: []const i64) Error!?*anyopaque {
        return maybeHandle(try Buffer.compare(ref(lhs), ref(rhs), direction, output_dims));
    }

    /// Selects between opaque buffer handles using an opaque predicate handle.
    pub fn select(pred: *anyopaque, on_true: *anyopaque, on_false: *anyopaque, output_dims: []const i64) Error!?*anyopaque {
        return maybeHandle(try Buffer.select(ref(pred), ref(on_true), ref(on_false), output_dims));
    }

    /// Clamps an opaque value handle between minimum and maximum handles.
    pub fn clamp(min: *anyopaque, value: *anyopaque, max: *anyopaque, output_dims: []const i64) Error!?*anyopaque {
        return maybeHandle(try Buffer.clamp(ref(min), ref(value), ref(max), output_dims));
    }

    /// Runs the while-compare-add backend fast path over opaque handles.
    pub fn whileF32CompareAdd(state: *anyopaque, limit: *anyopaque, step: *anyopaque, compare_direction: ir.CompareOp, update_op: ir.ElementwiseBinaryOp, output_dims: []const i64, max_iterations: u64) Error!?*anyopaque {
        return maybeHandle(try Buffer.whileF32CompareAdd(ref(state), ref(limit), ref(step), compare_direction, update_op, output_dims, max_iterations));
    }

    /// Runs the built-in binary-add custom call over opaque handles.
    pub fn customBinaryAddF32(lhs: *anyopaque, rhs: *anyopaque) Error!?*anyopaque {
        return maybeHandle(try Buffer.customBinaryAddF32(ref(lhs), ref(rhs)));
    }

    /// Runs the built-in scaled dot-product attention custom call over opaque handles.
    pub fn customScaledDotProductAttention(q: *anyopaque, k: *anyopaque, v: *anyopaque, token_index: *anyopaque) Error!?*anyopaque {
        return maybeHandle(try Buffer.customScaledDotProductAttention(ref(q), ref(k), ref(v), ref(token_index)));
    }

    /// Forces an opaque buffer handle to evaluate.
    pub fn eval(buffer: *anyopaque) Error!void {
        try ref(buffer).eval();
    }

    /// Forces opaque buffer handles to evaluate as a single backend call.
    pub fn evalMany(buffers: []const *anyopaque) Error!void {
        if (!mlx_call.bufferEvalMany(handles(refs(buffers)))) return error.CommandSubmissionFailed;
    }

    /// Copies an opaque buffer handle into host memory.
    pub fn copyToHost(src: *anyopaque, dst: []u8) Error!void {
        try ref(src).copyToHost(dst);
    }

    /// Reports whether an opaque buffer still owns a host shadow allocation.
    pub fn hasHostShadow(src: *anyopaque) bool {
        return ref(src).hasHostShadow();
    }

    /// Destroys an opaque buffer handle.
    pub fn destroy(buffer: *anyopaque) void {
        ref(buffer).destroy();
    }

    fn ref(handle: *anyopaque) Buffer {
        return Buffer.fromHandle(handle);
    }

    fn refs(handles_: []const *anyopaque) []const Buffer {
        return @ptrCast(handles_);
    }

    fn maybeHandle(buffer: ?Buffer) ?*anyopaque {
        return if (buffer) |value| value.toHandle() else null;
    }
};

/// Error set produced by typed MLX/Metal buffer operations.
pub const Error = error{
    UnsupportedElementType,
    ShapeMismatch,
    BufferAllocationFailed,
    CommandSubmissionFailed,
    BufferCopyFailed,
};

/// Forces all supplied MLX/Metal buffers to evaluate as a single backend call.
pub fn evalMany(buffers: []const Buffer) Error!void {
    if (!mlx_call.bufferEvalMany(handles(buffers))) return error.CommandSubmissionFailed;
}

const Dtype = struct {
    fn fromIr(element_type: ir.BufferType) ?mlx_call.Dtype {
        return switch (element_type) {
            .pred => .pred,
            .s8 => .s8,
            .s32 => .s32,
            .u8 => .u8,
            .u16 => .u16,
            .u32 => .u32,
            .u64 => .u64,
            .f16 => .f16,
            .f32 => .f32,
            .bf16 => .bf16,
            .c64 => .c64,
            else => null,
        };
    }
};

const BinaryOp = struct {
    fn fromIr(op: ir.ElementwiseBinaryOp) ?mlx_call.BinaryOp {
        return switch (op) {
            .add => .add,
            .subtract => .subtract,
            .multiply => .multiply,
            .divide => .divide,
            .maximum => .maximum,
            .minimum => .minimum,
            .power => .power,
            .atan2 => .atan2,
            .remainder => .remainder,
            .and_ => .and_,
            .or_ => .or_,
            .xor => .xor,
            .shift_left => .shift_left,
            .shift_right_arithmetic, .shift_right_logical => .shift_right,
        };
    }
};

const UnaryOp = struct {
    fn fromIr(op: ir.ElementwiseUnaryOp) ?mlx_call.UnaryOp {
        return switch (op) {
            .negate => .negate,
            .exp => .exp,
            .expm1 => .expm1,
            .tanh => .tanh,
            .sqrt => .sqrt,
            .rsqrt => .rsqrt,
            .abs => .abs,
            .cbrt => .cbrt,
            .ceil => .ceil,
            .floor => .floor,
            .log => .log,
            .log1p => .log1p,
            .logistic => .logistic,
            .sine => .sine,
            .cosine => .cosine,
            .not_ => .not_,
            .sign => .sign,
            .is_finite => .is_finite,
            .round_nearest_afz => .round_nearest_afz,
            .round_nearest_even => .round_nearest_even,
            .popcnt => .popcnt,
            .count_leading_zeros => .count_leading_zeros,
        };
    }
};

const ReduceOp = struct {
    fn fromPlan(op: ir.PlanInstructionKind) ?mlx_call.ReduceOp {
        return switch (op) {
            .reduce_sum => .sum,
            .reduce_max => .max,
            .reduce_min => .min,
            .reduce_and => .and_,
            .reduce_or => .or_,
            else => null,
        };
    }

    fn fromWindowPlan(op: ir.PlanInstructionKind) ?mlx_call.ReduceOp {
        return switch (op) {
            .reduce_window_sum => .sum,
            .reduce_window_max => .max,
            else => null,
        };
    }
};

const ScatterUpdate = struct {
    fn fromIr(update_kind: ir.ScatterUpdateKind) mlx_call.ScatterUpdate {
        return switch (update_kind) {
            .set => .set,
            .add => .add,
        };
    }
};

const CompareOp = struct {
    fn fromIr(op: ir.CompareOp) mlx_call.CompareOp {
        return switch (op) {
            .eq => .eq,
            .ne => .ne,
            .ge => .ge,
            .gt => .gt,
            .le => .le,
            .lt => .lt,
        };
    }
};

const FftKind = struct {
    fn fromIr(kind: ir.FftKind) mlx_call.FftKind {
        return switch (kind) {
            .fft => .fft,
            .ifft => .ifft,
            .rfft => .rfft,
            .irfft => .irfft,
        };
    }
};

const RngDistribution = struct {
    fn fromIr(distribution: ir.RngDistribution) mlx_call.RngDistribution {
        return switch (distribution) {
            .uniform => .uniform,
            .normal => .normal,
        };
    }
};

const TriangularTranspose = struct {
    fn fromIr(transpose_kind: ir.TriangularSolveTranspose) mlx_call.TriangularTranspose {
        return switch (transpose_kind) {
            .no_transpose => .none,
            .transpose, .adjoint => .transpose,
        };
    }
};

fn wrap(handle: ?mlx_call.BufferHandle, err: Error) Error!?Buffer {
    return if (handle) |ptr| Buffer{ .handle = ptr } else err;
}

fn handles(buffers: []const Buffer) []const mlx_call.BufferHandle {
    return @ptrCast(buffers);
}

test "buffer handle wrapper preserves opaque identity" {
    const raw: *anyopaque = @ptrFromInt(@alignOf(usize));
    try @import("std").testing.expectEqual(raw, Buffer.fromHandle(raw).toHandle());
}
