const std = @import("std");
const c = @import("c");

/// Opaque MLX/Metal buffer owned by the foreign backend runtime.
pub const BufferHandle = *anyopaque;

/// Opaque MLX/Metal async host-to-device transfer owned by the foreign backend runtime.
pub const AsyncTransferHandle = *anyopaque;

/// Opaque compiled MLX/Metal program owned by the foreign backend runtime.
pub const ProgramHandle = *anyopaque;

/// Device-buffer callback frame used while MLX records a compiled program.
pub const ProgramBuildCall = struct {
    raw_inputs: [*c]const ?*c.PjrtxMlxMetalBuffer,
    input_count: usize,
    raw_outputs: [*c]?*c.PjrtxMlxMetalBuffer,
    output_count: usize,

    /// Returns the number of dynamic inputs presented by the compiled program.
    pub fn inputCount(self: ProgramBuildCall) usize {
        return self.input_count;
    }

    /// Returns the number of outputs expected from the compiled program builder.
    pub fn outputCount(self: ProgramBuildCall) usize {
        return self.output_count;
    }

    /// Borrows one input buffer from the compiled program builder.
    pub fn input(self: ProgramBuildCall, index: usize) ?BufferHandle {
        if (index >= self.input_count) return null;
        return fromRawBuffer(self.raw_inputs[index]);
    }

    /// Publishes one output buffer produced by the compiled program builder.
    pub fn setOutput(self: ProgramBuildCall, index: usize, handle: BufferHandle) bool {
        if (index >= self.output_count) return false;
        self.raw_outputs[index] = rawBuffer(handle);
        return true;
    }
};

/// Callback invoked while MLX records the body of a compiled program.
pub const ProgramBuildCallback = fn (?*anyopaque, ProgramBuildCall) bool;

/// Outputs returned from an executed compiled MLX/Metal program.
pub const ProgramOutputs = struct {
    raw_outputs: [*c]?*c.PjrtxMlxMetalBuffer,
    count: usize,

    /// Releases any output array storage still owned by the foreign runtime.
    pub fn deinit(self: ProgramOutputs) void {
        c.pjrtx_mlx_metal_program_output_array_destroy(self.raw_outputs);
    }

    /// Returns the number of buffers produced by the compiled program.
    pub fn len(self: ProgramOutputs) usize {
        return self.count;
    }

    /// Transfers one output buffer handle to the caller.
    pub fn take(self: ProgramOutputs, index: usize) ?BufferHandle {
        if (index >= self.count) return null;
        const raw = self.raw_outputs[index] orelse return null;
        self.raw_outputs[index] = null;
        return fromRawBuffer(raw);
    }
};

/// Device metadata copied out of the MLX/Metal runtime without exposing C layout.
pub const DeviceInfo = struct {
    /// MLX/Metal device ordinal used for subsequent backend calls.
    ordinal: i32,
    /// Metal registry identifier when the runtime can provide it.
    registry_id: u64,
    /// Runtime-provided memory working-set recommendation.
    recommended_max_working_set_size: u64,
    /// Whether Metal reports unified memory for this device.
    has_unified_memory: bool,
    /// NUL-padded device name bytes copied from the C boundary.
    name: [DeviceNameBytes]u8,
};

/// MLX/Metal dtype code accepted only by the raw C boundary calls.
pub const Dtype = enum(c_int) {
    pred = c.PJRTX_MLX_METAL_DTYPE_PRED,
    s8 = c.PJRTX_MLX_METAL_DTYPE_S8,
    s32 = c.PJRTX_MLX_METAL_DTYPE_S32,
    u8 = c.PJRTX_MLX_METAL_DTYPE_U8,
    u16 = c.PJRTX_MLX_METAL_DTYPE_U16,
    u32 = c.PJRTX_MLX_METAL_DTYPE_U32,
    u64 = c.PJRTX_MLX_METAL_DTYPE_U64,
    f16 = c.PJRTX_MLX_METAL_DTYPE_F16,
    f32 = c.PJRTX_MLX_METAL_DTYPE_F32,
    bf16 = c.PJRTX_MLX_METAL_DTYPE_BF16,
    c64 = c.PJRTX_MLX_METAL_DTYPE_C64,
};

/// MLX/Metal binary operation code accepted only by the raw C boundary calls.
pub const BinaryOp = enum(c_int) {
    add = c.PJRTX_MLX_METAL_U8_BINARY_ADD,
    subtract = c.PJRTX_MLX_METAL_U8_BINARY_SUBTRACT,
    multiply = c.PJRTX_MLX_METAL_U8_BINARY_MULTIPLY,
    divide = c.PJRTX_MLX_METAL_U8_BINARY_DIVIDE,
    maximum = c.PJRTX_MLX_METAL_BINARY_MAXIMUM,
    minimum = c.PJRTX_MLX_METAL_BINARY_MINIMUM,
    power = c.PJRTX_MLX_METAL_BINARY_POWER,
    remainder = c.PJRTX_MLX_METAL_BINARY_REMAINDER,
    atan2 = c.PJRTX_MLX_METAL_BINARY_ATAN2,
    and_ = c.PJRTX_MLX_METAL_BINARY_AND,
    or_ = c.PJRTX_MLX_METAL_BINARY_OR,
    xor = c.PJRTX_MLX_METAL_BINARY_XOR,
    shift_left = c.PJRTX_MLX_METAL_BINARY_SHIFT_LEFT,
    shift_right = c.PJRTX_MLX_METAL_BINARY_SHIFT_RIGHT,
};

/// MLX/Metal unary operation code accepted only by the raw C boundary calls.
pub const UnaryOp = enum(c_int) {
    negate = c.PJRTX_MLX_METAL_U8_UNARY_NEGATE,
    exp = c.PJRTX_MLX_METAL_UNARY_EXP,
    tanh = c.PJRTX_MLX_METAL_UNARY_TANH,
    sqrt = c.PJRTX_MLX_METAL_UNARY_SQRT,
    rsqrt = c.PJRTX_MLX_METAL_UNARY_RSQRT,
    abs = c.PJRTX_MLX_METAL_UNARY_ABS,
    ceil = c.PJRTX_MLX_METAL_UNARY_CEIL,
    floor = c.PJRTX_MLX_METAL_UNARY_FLOOR,
    log = c.PJRTX_MLX_METAL_UNARY_LOG,
    log1p = c.PJRTX_MLX_METAL_UNARY_LOG1P,
    logistic = c.PJRTX_MLX_METAL_UNARY_LOGISTIC,
    sine = c.PJRTX_MLX_METAL_UNARY_SIN,
    cosine = c.PJRTX_MLX_METAL_UNARY_COS,
    sign = c.PJRTX_MLX_METAL_UNARY_SIGN,
    expm1 = c.PJRTX_MLX_METAL_UNARY_EXPM1,
    not_ = c.PJRTX_MLX_METAL_UNARY_NOT,
    is_finite = c.PJRTX_MLX_METAL_UNARY_ISFINITE,
    round_nearest_even = c.PJRTX_MLX_METAL_UNARY_ROUND,
    cbrt = c.PJRTX_MLX_METAL_UNARY_CBRT,
    round_nearest_afz = c.PJRTX_MLX_METAL_UNARY_ROUND_AFZ,
    popcnt = c.PJRTX_MLX_METAL_UNARY_POPCNT,
    count_leading_zeros = c.PJRTX_MLX_METAL_UNARY_CLZ,
};

/// MLX/Metal reduction operation code accepted only by the raw C boundary calls.
pub const ReduceOp = enum(c_int) {
    sum = c.PJRTX_MLX_METAL_REDUCE_SUM,
    max = c.PJRTX_MLX_METAL_REDUCE_MAX,
    min = c.PJRTX_MLX_METAL_REDUCE_MIN,
    and_ = c.PJRTX_MLX_METAL_REDUCE_AND,
    or_ = c.PJRTX_MLX_METAL_REDUCE_OR,
};

/// MLX/Metal scatter update code accepted only by the raw C boundary calls.
pub const ScatterUpdate = enum(c_int) {
    set = c.PJRTX_MLX_METAL_SCATTER_SET,
    add = c.PJRTX_MLX_METAL_SCATTER_ADD,
};

/// MLX/Metal comparison code accepted only by the raw C boundary calls.
pub const CompareOp = enum(c_int) {
    eq = c.PJRTX_MLX_METAL_COMPARE_EQ,
    ne = c.PJRTX_MLX_METAL_COMPARE_NE,
    ge = c.PJRTX_MLX_METAL_COMPARE_GE,
    gt = c.PJRTX_MLX_METAL_COMPARE_GT,
    le = c.PJRTX_MLX_METAL_COMPARE_LE,
    lt = c.PJRTX_MLX_METAL_COMPARE_LT,
};

/// MLX/Metal FFT code accepted only by the raw C boundary calls.
pub const FftKind = enum(c_int) {
    fft = c.PJRTX_MLX_METAL_FFT,
    ifft = c.PJRTX_MLX_METAL_IFFT,
    rfft = c.PJRTX_MLX_METAL_RFFT,
    irfft = c.PJRTX_MLX_METAL_IRFFT,
};

/// MLX/Metal random distribution code accepted only by the raw C boundary calls.
pub const RngDistribution = enum(c_int) {
    uniform = c.PJRTX_MLX_METAL_RNG_UNIFORM,
    normal = c.PJRTX_MLX_METAL_RNG_NORMAL,
};

/// MLX/Metal triangular solve transpose mode accepted only by the raw C boundary calls.
pub const TriangularTranspose = enum(c_int) {
    none = 0,
    transpose = 1,
};

/// Pair of MLX/Metal buffers produced by an operation with two outputs.
pub const BufferPair = struct {
    /// First returned buffer handle.
    first: BufferHandle,
    /// Second returned buffer handle.
    second: BufferHandle,
};

/// Fixed byte capacity of a copied MLX/Metal device name.
pub const DeviceNameBytes = c.PJRTX_MLX_METAL_DEVICE_NAME_BYTES;

/// Copies MLX/Metal device metadata into Zig-owned boundary structs.
pub fn copyDevices(out_devices: []DeviceInfo) i32 {
    var raw_devices: [256]c.PjrtxMlxMetalDeviceInfo = undefined;
    const max_devices = @min(out_devices.len, raw_devices.len);
    const copied = c.pjrtx_mlx_metal_copy_devices(&raw_devices, @intCast(max_devices));
    if (copied <= 0) return copied;
    const count: usize = @min(@as(usize, @intCast(copied)), max_devices);
    for (out_devices[0..count], raw_devices[0..count]) |*out_device, raw_device| {
        out_device.* = .{
            .ordinal = raw_device.ordinal,
            .registry_id = raw_device.registry_id,
            .recommended_max_working_set_size = raw_device.recommended_max_working_set_size,
            .has_unified_memory = raw_device.has_unified_memory != 0,
            .name = raw_device.name,
        };
    }
    return copied;
}

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

/// Creates an MLX/Metal async host-to-device transfer.
pub fn asyncH2DCreate(device_ordinal: i32, dtype: Dtype, dims: []const i64, byte_size: usize) ?AsyncTransferHandle {
    return fromRawTransfer(c.pjrtx_mlx_metal_async_h2d_create(device_ordinal, code(dtype), dims.ptr, dims.len, byte_size));
}

/// Writes one byte range into an MLX/Metal async host-to-device transfer.
pub fn asyncH2DWrite(transfer: AsyncTransferHandle, offset: usize, src: []const u8) bool {
    return c.pjrtx_mlx_metal_async_h2d_write(rawTransfer(transfer), offset, src.ptr, src.len) != 0;
}

/// Finishes an MLX/Metal async host-to-device transfer and returns its buffer.
pub fn asyncH2DFinish(transfer: AsyncTransferHandle) ?BufferHandle {
    return fromRawBuffer(c.pjrtx_mlx_metal_async_h2d_finish(rawTransfer(transfer)));
}

/// Destroys an MLX/Metal async host-to-device transfer.
pub fn asyncH2DDestroy(transfer: AsyncTransferHandle) void {
    c.pjrtx_mlx_metal_async_h2d_destroy(rawTransfer(transfer));
}

/// Creates an MLX compiled program with every argument dynamic.
pub fn programCreate(
    user_data: ?*anyopaque,
    input_count: usize,
    output_count: usize,
    comptime callback: ProgramBuildCallback,
) ?ProgramHandle {
    return fromRawProgram(c.pjrtx_mlx_metal_program_create(
        user_data,
        input_count,
        output_count,
        ProgramBuildAdapter(callback).call,
    ));
}

/// Creates an MLX compiled program with stable captured inputs and dynamic indices.
pub fn programCreateWithCaptures(
    user_data: ?*anyopaque,
    input_count: usize,
    output_count: usize,
    comptime callback: ProgramBuildCallback,
    arguments: []const BufferHandle,
    dynamic_indices: []const u64,
) ?ProgramHandle {
    return fromRawProgram(c.pjrtx_mlx_metal_program_create_with_captures(
        user_data,
        input_count,
        output_count,
        ProgramBuildAdapter(callback).call,
        rawBufferList(arguments),
        dynamic_indices.ptr,
        dynamic_indices.len,
    ));
}

/// Executes an MLX compiled program with optional donated input indices.
pub fn programExecuteWithDonation(program: ProgramHandle, arguments: []const BufferHandle, donated_input_indices: []const u64) ?ProgramOutputs {
    var raw_outputs: [*c]?*c.PjrtxMlxMetalBuffer = null;
    var raw_output_count: u64 = 0;
    const ok = c.pjrtx_mlx_metal_program_execute_with_donation(
        rawProgram(program),
        rawBufferList(arguments),
        arguments.len,
        if (donated_input_indices.len == 0) null else donated_input_indices.ptr,
        donated_input_indices.len,
        &raw_outputs,
        &raw_output_count,
    );
    if (ok == 0 or raw_outputs == null) {
        if (raw_outputs != null) c.pjrtx_mlx_metal_program_output_array_destroy(raw_outputs);
        return null;
    }
    return .{ .raw_outputs = raw_outputs, .count = @intCast(raw_output_count) };
}

/// Destroys an MLX compiled program handle.
pub fn programDestroy(program: ProgramHandle) void {
    c.pjrtx_mlx_metal_program_destroy(rawProgram(program));
}

fn ProgramBuildAdapter(comptime callback: ProgramBuildCallback) type {
    return struct {
        fn call(
            user_data: ?*anyopaque,
            raw_inputs: [*c]const ?*c.PjrtxMlxMetalBuffer,
            input_count: u64,
            raw_outputs: [*c]?*c.PjrtxMlxMetalBuffer,
            output_count: u64,
        ) callconv(.c) c_int {
            const frame: ProgramBuildCall = .{
                .raw_inputs = raw_inputs,
                .input_count = @intCast(input_count),
                .raw_outputs = raw_outputs,
                .output_count = @intCast(output_count),
            };
            return flag(callback(user_data, frame));
        }
    };
}

fn code(value: anytype) c_int {
    return @intFromEnum(value);
}

fn flag(value: bool) c_int {
    return if (value) 1 else 0;
}

fn fromRawBuffer(handle: ?*c.PjrtxMlxMetalBuffer) ?BufferHandle {
    return if (handle) |ptr| @ptrCast(ptr) else null;
}

fn rawBuffer(handle: BufferHandle) *c.PjrtxMlxMetalBuffer {
    return @ptrCast(@alignCast(handle));
}

fn rawBufferList(handles: []const BufferHandle) [*c]?*c.PjrtxMlxMetalBuffer {
    return @ptrCast(@constCast(handles.ptr));
}

fn fromRawTransfer(handle: ?*c.PjrtxMlxMetalAsyncHostToDeviceTransfer) ?AsyncTransferHandle {
    return if (handle) |ptr| @ptrCast(ptr) else null;
}

fn rawTransfer(handle: AsyncTransferHandle) *c.PjrtxMlxMetalAsyncHostToDeviceTransfer {
    return @ptrCast(@alignCast(handle));
}

fn fromRawProgram(handle: ?*c.PjrtxMlxMetalProgram) ?ProgramHandle {
    return if (handle) |ptr| @ptrCast(ptr) else null;
}

fn rawProgram(handle: ProgramHandle) *c.PjrtxMlxMetalProgram {
    return @ptrCast(@alignCast(handle));
}

test "device name bytes are copied without exposing C callers" {
    var devices = [_]DeviceInfo{std.mem.zeroes(DeviceInfo)} ** 1;
    _ = copyDevices(&devices);
}
