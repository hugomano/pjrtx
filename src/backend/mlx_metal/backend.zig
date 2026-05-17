const std = @import("std");
const c = @import("c");
const backend = @import("src/backend");
const core = @import("src/core");

pub fn create() backend.Backend {
    return .{ .vtable = &vtable };
}

fn kind(_: backend.Backend) core.BackendKind {
    return .metal_mlx;
}

fn capabilities(_: backend.Backend) backend.Capabilities {
    return .{
        .kind = .metal_mlx,
        .name = "metal_mlx",
        .supports_device_buffers = true,
        .supports_unified_memory = true,
    };
}

fn enumerateDevices(_: backend.Backend, allocator: std.mem.Allocator, _: usize) backend.Error![]core.DeviceDescriptor {
    var metal_devices: [core.MAX_DEVICES]c.PjrtxMlxMetalDeviceInfo = undefined;
    const copied = c.pjrtx_mlx_metal_copy_devices(&metal_devices, core.MAX_DEVICES);
    const count: usize = if (copied <= 0) 1 else @intCast(copied);
    const devices = try allocator.alloc(core.DeviceDescriptor, count);
    errdefer allocator.free(devices);

    for (devices, 0..) |*device, i| {
        const metal_device = if (copied > 0) metal_devices[i] else std.mem.zeroes(c.PjrtxMlxMetalDeviceInfo);
        const name_bytes = if (copied > 0) cNameBytes(&metal_device.name) else "Metal/MLX device";
        const name = try allocator.dupe(u8, name_bytes);
        errdefer allocator.free(name);

        var debug_buffer: [256]u8 = undefined;
        var debug_writer = std.Io.Writer.fixed(&debug_buffer);
        debug_writer.print("PjRTx Metal/MLX device {d}: {s}", .{ i, name }) catch return error.BufferAllocationFailed;
        const debug_string = try allocator.dupe(u8, debug_writer.buffered());
        errdefer allocator.free(debug_string);

        const id: i32 = @intCast(i);
        device.* = .{
            .id = id,
            .local_hardware_id = if (copied > 0) metal_device.ordinal else id,
            .registry_id = if (copied > 0) metal_device.registry_id else 0,
            .name = name,
            .debug_string = debug_string,
            .memory_bytes = if (copied > 0) metal_device.recommended_max_working_set_size else 0,
            .has_unified_memory = copied <= 0 or metal_device.has_unified_memory != 0,
            .default_memory_id = id,
        };
    }
    return devices;
}

fn releaseDeviceDescriptors(_: backend.Backend, allocator: std.mem.Allocator, descriptors: []core.DeviceDescriptor) void {
    for (descriptors) |descriptor| {
        allocator.free(descriptor.name);
        allocator.free(descriptor.debug_string);
    }
    allocator.free(descriptors);
}

fn bufferFromHost(_: backend.Backend, device_local_hardware_id: i32, element_type: core.BufferType, dims: []const i64, src: []const u8) backend.Error!?backend.BufferHandle {
    if (src.len == 0) return null;
    const dtype = mlxDtype(element_type) orelse return error.UnsupportedElementType;
    const handle = c.pjrtx_mlx_metal_buffer_from_host_typed(device_local_hardware_id, src.ptr, src.len, dtype, dims.ptr, dims.len) orelse return error.BufferAllocationFailed;
    return @ptrCast(handle);
}

fn cloneBuffer(_: backend.Backend, src: backend.BufferHandle) backend.Error!?backend.BufferHandle {
    const handle = c.pjrtx_mlx_metal_buffer_clone(@ptrCast(@alignCast(src))) orelse return error.CommandSubmissionFailed;
    return @ptrCast(handle);
}

fn binary(_: backend.Backend, lhs: backend.BufferHandle, rhs: backend.BufferHandle, op: core.ElementwiseBinaryOp) backend.Error!?backend.BufferHandle {
    const op_code = mlxBinaryOpCode(op) orelse return error.CommandSubmissionFailed;
    const handle = c.pjrtx_mlx_metal_buffer_binary(@ptrCast(@alignCast(lhs)), @ptrCast(@alignCast(rhs)), op_code) orelse return error.CommandSubmissionFailed;
    return @ptrCast(handle);
}

fn unary(_: backend.Backend, src: backend.BufferHandle, op: core.ElementwiseUnaryOp) backend.Error!?backend.BufferHandle {
    const op_code = mlxUnaryOpCode(op) orelse return error.CommandSubmissionFailed;
    const handle = c.pjrtx_mlx_metal_buffer_unary(@ptrCast(@alignCast(src)), op_code) orelse return error.CommandSubmissionFailed;
    return @ptrCast(handle);
}

fn reshape(_: backend.Backend, src: backend.BufferHandle, dims: []const i64) backend.Error!?backend.BufferHandle {
    const handle = c.pjrtx_mlx_metal_buffer_reshape(@ptrCast(@alignCast(src)), dims.ptr, dims.len) orelse return error.CommandSubmissionFailed;
    return @ptrCast(handle);
}

fn transpose(_: backend.Backend, src: backend.BufferHandle, permutation: []const i64) backend.Error!?backend.BufferHandle {
    const handle = c.pjrtx_mlx_metal_buffer_transpose(@ptrCast(@alignCast(src)), permutation.ptr, permutation.len) orelse return error.CommandSubmissionFailed;
    return @ptrCast(handle);
}

fn broadcastInDim(_: backend.Backend, src: backend.BufferHandle, broadcast_dimensions: []const i64, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    const handle = c.pjrtx_mlx_metal_buffer_broadcast_in_dim(@ptrCast(@alignCast(src)), broadcast_dimensions.ptr, broadcast_dimensions.len, output_dims.ptr, output_dims.len) orelse return error.CommandSubmissionFailed;
    return @ptrCast(handle);
}

fn slice(_: backend.Backend, src: backend.BufferHandle, start_indices: []const i64, limit_indices: []const i64, strides: []const i64, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    const handle = c.pjrtx_mlx_metal_buffer_slice(@ptrCast(@alignCast(src)), start_indices.ptr, limit_indices.ptr, strides.ptr, start_indices.len, output_dims.ptr, output_dims.len) orelse return error.CommandSubmissionFailed;
    return @ptrCast(handle);
}

fn concatenate(_: backend.Backend, lhs: backend.BufferHandle, rhs: backend.BufferHandle, dimension: i64, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    const handle = c.pjrtx_mlx_metal_buffer_concatenate(@ptrCast(@alignCast(lhs)), @ptrCast(@alignCast(rhs)), dimension, output_dims.ptr, output_dims.len) orelse return error.CommandSubmissionFailed;
    return @ptrCast(handle);
}

fn dotGeneral(_: backend.Backend, lhs: backend.BufferHandle, rhs: backend.BufferHandle, lhs_batch_dimensions: []const i64, rhs_batch_dimensions: []const i64, lhs_contracting_dimensions: []const i64, rhs_contracting_dimensions: []const i64, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    const handle = c.pjrtx_mlx_metal_buffer_dot_general(
        @ptrCast(@alignCast(lhs)),
        @ptrCast(@alignCast(rhs)),
        lhs_batch_dimensions.ptr,
        lhs_batch_dimensions.len,
        rhs_batch_dimensions.ptr,
        rhs_batch_dimensions.len,
        lhs_contracting_dimensions.ptr,
        lhs_contracting_dimensions.len,
        rhs_contracting_dimensions.ptr,
        rhs_contracting_dimensions.len,
        output_dims.ptr,
        output_dims.len,
    ) orelse return error.CommandSubmissionFailed;
    return @ptrCast(handle);
}

fn reduce(_: backend.Backend, src: backend.BufferHandle, op: core.PlanInstructionKind, dimensions: []const i64, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    const code: c_int = switch (op) {
        .reduce_sum => c.PJRTX_MLX_METAL_REDUCE_SUM,
        .reduce_max => c.PJRTX_MLX_METAL_REDUCE_MAX,
        else => return error.CommandSubmissionFailed,
    };
    const handle = c.pjrtx_mlx_metal_buffer_reduce(@ptrCast(@alignCast(src)), code, dimensions.ptr, dimensions.len, output_dims.ptr, output_dims.len) orelse return error.CommandSubmissionFailed;
    return @ptrCast(handle);
}

fn compare(_: backend.Backend, lhs: backend.BufferHandle, rhs: backend.BufferHandle, direction: core.CompareOp, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    const handle = c.pjrtx_mlx_metal_buffer_compare(@ptrCast(@alignCast(lhs)), @ptrCast(@alignCast(rhs)), mlxCompareOpCode(direction), output_dims.ptr, output_dims.len) orelse return error.CommandSubmissionFailed;
    return @ptrCast(handle);
}

fn select(_: backend.Backend, pred: backend.BufferHandle, on_true: backend.BufferHandle, on_false: backend.BufferHandle, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    const handle = c.pjrtx_mlx_metal_buffer_select(@ptrCast(@alignCast(pred)), @ptrCast(@alignCast(on_true)), @ptrCast(@alignCast(on_false)), output_dims.ptr, output_dims.len) orelse return error.CommandSubmissionFailed;
    return @ptrCast(handle);
}

fn copyToHost(_: backend.Backend, src: backend.BufferHandle, dst: []u8) backend.Error!void {
    const ok = c.pjrtx_mlx_metal_buffer_copy_to_host(@ptrCast(@alignCast(src)), dst.ptr, dst.len);
    if (ok == 0) return error.BufferCopyFailed;
}

fn destroyBuffer(_: backend.Backend, buffer: backend.BufferHandle) void {
    c.pjrtx_mlx_metal_buffer_destroy(@ptrCast(@alignCast(buffer)));
}

fn cNameBytes(name: *const [128]u8) []const u8 {
    const end = std.mem.indexOfScalar(u8, name, 0) orelse name.len;
    return name[0..end];
}

fn mlxDtype(element_type: core.BufferType) ?c_int {
    return switch (element_type) {
        .pred => c.PJRTX_MLX_METAL_DTYPE_PRED,
        .s8 => c.PJRTX_MLX_METAL_DTYPE_S8,
        .s32 => c.PJRTX_MLX_METAL_DTYPE_S32,
        .u8 => c.PJRTX_MLX_METAL_DTYPE_U8,
        .u32 => c.PJRTX_MLX_METAL_DTYPE_U32,
        .f32 => c.PJRTX_MLX_METAL_DTYPE_F32,
        else => null,
    };
}

fn mlxBinaryOpCode(op: core.ElementwiseBinaryOp) ?c_int {
    return switch (op) {
        .add => c.PJRTX_MLX_METAL_U8_BINARY_ADD,
        .subtract => c.PJRTX_MLX_METAL_U8_BINARY_SUBTRACT,
        .multiply => c.PJRTX_MLX_METAL_U8_BINARY_MULTIPLY,
        .divide => c.PJRTX_MLX_METAL_U8_BINARY_DIVIDE,
        .maximum => c.PJRTX_MLX_METAL_BINARY_MAXIMUM,
        .minimum => c.PJRTX_MLX_METAL_BINARY_MINIMUM,
        .power => c.PJRTX_MLX_METAL_BINARY_POWER,
        .remainder => c.PJRTX_MLX_METAL_BINARY_REMAINDER,
        else => null,
    };
}

fn mlxUnaryOpCode(op: core.ElementwiseUnaryOp) ?c_int {
    return switch (op) {
        .negate => c.PJRTX_MLX_METAL_U8_UNARY_NEGATE,
        .exp => c.PJRTX_MLX_METAL_UNARY_EXP,
        .tanh => c.PJRTX_MLX_METAL_UNARY_TANH,
        .sqrt => c.PJRTX_MLX_METAL_UNARY_SQRT,
        .rsqrt => c.PJRTX_MLX_METAL_UNARY_RSQRT,
        .abs => c.PJRTX_MLX_METAL_UNARY_ABS,
        .ceil => c.PJRTX_MLX_METAL_UNARY_CEIL,
        .floor => c.PJRTX_MLX_METAL_UNARY_FLOOR,
        .log => c.PJRTX_MLX_METAL_UNARY_LOG,
        .log1p => c.PJRTX_MLX_METAL_UNARY_LOG1P,
        .logistic => c.PJRTX_MLX_METAL_UNARY_LOGISTIC,
        .sine => c.PJRTX_MLX_METAL_UNARY_SIN,
        .cosine => c.PJRTX_MLX_METAL_UNARY_COS,
        .sign => c.PJRTX_MLX_METAL_UNARY_SIGN,
        else => null,
    };
}

fn mlxCompareOpCode(op: core.CompareOp) c_int {
    return switch (op) {
        .eq => c.PJRTX_MLX_METAL_COMPARE_EQ,
        .ne => c.PJRTX_MLX_METAL_COMPARE_NE,
        .ge => c.PJRTX_MLX_METAL_COMPARE_GE,
        .gt => c.PJRTX_MLX_METAL_COMPARE_GT,
        .le => c.PJRTX_MLX_METAL_COMPARE_LE,
        .lt => c.PJRTX_MLX_METAL_COMPARE_LT,
    };
}

const vtable: backend.Backend.VTable = .{
    .kind = kind,
    .capabilities = capabilities,
    .enumerateDevices = enumerateDevices,
    .releaseDeviceDescriptors = releaseDeviceDescriptors,
    .bufferFromHost = bufferFromHost,
    .cloneBuffer = cloneBuffer,
    .binary = binary,
    .unary = unary,
    .reshape = reshape,
    .transpose = transpose,
    .broadcastInDim = broadcastInDim,
    .slice = slice,
    .concatenate = concatenate,
    .dotGeneral = dotGeneral,
    .reduce = reduce,
    .compare = compare,
    .select = select,
    .copyToHost = copyToHost,
    .destroyBuffer = destroyBuffer,
};

test "mlx metal backend exposes opaque backend interface" {
    const b = create();
    try std.testing.expectEqual(core.BackendKind.metal_mlx, b.kind());
    const devices = try b.enumerateDevices(std.testing.allocator, 1);
    defer b.releaseDeviceDescriptors(std.testing.allocator, devices);
    try std.testing.expect(devices.len >= 1);
}
