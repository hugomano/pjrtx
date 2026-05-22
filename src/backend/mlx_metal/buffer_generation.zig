const ir = @import("src/compiler/ir");

const buffer_handle = @import("buffer_handle.zig");
const encoding = @import("buffer_encoding.zig");
const mlx_call = @import("mlx_call.zig");

pub const Error = buffer_handle.Error;

/// Creates an MLX/Metal iota buffer.
pub fn iota(comptime Buffer: type, device_local_hardware_id: i32, element_type: ir.BufferType, dims: []const i64, iota_dimension: i64) Error!?Buffer {
    const dtype = encoding.dtype(element_type) orelse return error.UnsupportedElementType;
    return buffer_handle.wrap(Buffer, mlx_call.bufferIota(device_local_hardware_id, dtype, dims, iota_dimension), error.CommandSubmissionFailed);
}

/// Creates an MLX/Metal partition-id scalar buffer.
pub fn partitionId(comptime Buffer: type, device_local_hardware_id: i32, element_type: ir.BufferType, partition_id: u32) Error!?Buffer {
    const dtype = encoding.dtype(element_type) orelse return error.UnsupportedElementType;
    return buffer_handle.wrap(Buffer, mlx_call.bufferPartitionId(device_local_hardware_id, dtype, partition_id), error.CommandSubmissionFailed);
}

/// Runs a random distribution operation using two bound buffers.
pub fn rng(a: anytype, b: @TypeOf(a), distribution: ir.RngDistribution, output_type: ir.BufferType, output_dims: []const i64) Error!?@TypeOf(a) {
    const dtype = encoding.dtype(output_type) orelse return error.UnsupportedElementType;
    return buffer_handle.wrap(@TypeOf(a), mlx_call.bufferRng(a.handle, b.handle, encoding.rngDistribution(distribution), dtype, output_dims), error.CommandSubmissionFailed);
}

/// Runs random bit generation and returns updated state plus bits.
pub fn rngBitGenerator(state: anytype, output_type: ir.BufferType, output_dims: []const i64) Error!?buffer_handle.Pair(@TypeOf(state)) {
    const dtype = encoding.dtype(output_type) orelse return error.UnsupportedElementType;
    const pair = mlx_call.bufferRngBitGenerator(state.handle, dtype, output_dims) orelse return error.CommandSubmissionFailed;
    return .{
        .first = .{ .handle = pair.first },
        .second = .{ .handle = pair.second },
    };
}
