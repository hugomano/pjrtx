const std = @import("std");
const backend_api = @import("src/backend/mlx_metal");
const ir = @import("src/compiler/ir");

const device_memory = @import("device_memory.zig");
const event_mod = @import("event.zig");

pub const BufferType = ir.BufferType;
pub const ElementwiseBinaryOp = ir.ElementwiseBinaryOp;
pub const ElementwiseUnaryOp = ir.ElementwiseUnaryOp;
pub const CompareOp = ir.CompareOp;
pub const Device = device_memory.Device;
pub const Memory = device_memory.Memory;
pub const Event = event_mod.Event;

/// Errors returned while creating device-resident runtime buffers.
pub const BufferCreateError = std.mem.Allocator.Error || backend_api.Error || error{ InvalidArgument, UnsupportedRuntimeFeature };

/// Errors returned while reserving a buffer for async backend transfer completion.
pub const PendingBackendTransferBufferError = std.mem.Allocator.Error || error{InvalidArgument};

/// Errors returned while copying device-resident buffer contents to host memory.
pub const HostReadbackError = error{
    DestinationTooSmall,
    BufferDeleted,
    BufferDonated,
    BufferNotReady,
    BufferReadinessFailed,
    UnsupportedRuntimeFeature,
    BackendBufferCopyFailed,
};

/// Tracks whether a runtime buffer may still be used by execute/copy paths.
pub const BufferState = enum {
    live,
    deleted,
    donated,
};

pub const Buffer = struct {
    allocator: std.mem.Allocator,
    backend: backend_api.Backend,
    element_type: BufferType,
    dims: []i64,
    device_id: i32,
    memory_id: i32,
    device: *Device,
    memory: *Memory,
    shard_index: usize,
    byte_size: usize,
    bytes: []u8,
    backend_buffer: ?backend_api.BufferHandle = null,
    state: BufferState = .live,
    deleted: bool = false,
    accounted_bytes: usize = 0,
    ready_event: Event = Event.ready(),

    fn initBackendOnly(
        allocator: std.mem.Allocator,
        src: *Buffer,
        element_type: BufferType,
        dims_: []const i64,
        byte_size: usize,
        backend_buffer: backend_api.BufferHandle,
        shard_index: usize,
    ) BufferCreateError!*Buffer {
        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, dims_);
        errdefer allocator.free(dims);

        const bytes = try allocator.alloc(u8, 0);
        errdefer allocator.free(bytes);

        buffer.* = .{
            .allocator = allocator,
            .backend = src.backend,
            .element_type = element_type,
            .dims = dims,
            .device_id = src.device_id,
            .memory_id = src.memory_id,
            .device = src.device,
            .memory = src.memory,
            .shard_index = shard_index,
            .byte_size = byte_size,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
        };
        return buffer.accountDeviceBytes();
    }

    pub fn initHostCopyForBackend(
        allocator: std.mem.Allocator,
        backend_impl: backend_api.Backend,
        element_type: BufferType,
        dims_: []const i64,
        device: *Device,
        memory: *Memory,
        shard_index: usize,
        src: []const u8,
    ) BufferCreateError!*Buffer {
        if (!memory.isAddressableBy(device)) return error.InvalidArgument;

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, dims_);
        errdefer allocator.free(dims);

        const backend_buffer = try backend_impl.bufferFromHost(device.local_hardware_id, element_type, dims, src) orelse
            return error.UnsupportedRuntimeFeature;
        errdefer backend_impl.destroyBuffer(backend_buffer);

        const bytes = try allocator.alloc(u8, 0);
        errdefer allocator.free(bytes);

        buffer.* = .{
            .allocator = allocator,
            .backend = backend_impl,
            .element_type = element_type,
            .dims = dims,
            .device_id = device.id,
            .memory_id = memory.id,
            .device = device,
            .memory = memory,
            .shard_index = shard_index,
            .byte_size = src.len,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
        };
        memory.stats.host_to_device_bytes += @intCast(src.len);
        return buffer.accountDeviceBytes();
    }

    pub fn initDeviceAllocationForBackend(
        allocator: std.mem.Allocator,
        backend_impl: backend_api.Backend,
        element_type: BufferType,
        dims_: []const i64,
        device: *Device,
        memory: *Memory,
        shard_index: usize,
    ) BufferCreateError!*Buffer {
        if (!memory.isAddressableBy(device)) return error.InvalidArgument;

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, dims_);
        errdefer allocator.free(dims);

        const byte_size = ir.denseByteSize(element_type, dims);
        const backend_buffer = try backend_impl.allocateBuffer(device.local_hardware_id, element_type, dims) orelse
            return error.UnsupportedRuntimeFeature;
        errdefer backend_impl.destroyBuffer(backend_buffer);

        const bytes = try allocator.alloc(u8, 0);
        errdefer allocator.free(bytes);

        buffer.* = .{
            .allocator = allocator,
            .backend = backend_impl,
            .element_type = element_type,
            .dims = dims,
            .device_id = device.id,
            .memory_id = memory.id,
            .device = device,
            .memory = memory,
            .shard_index = shard_index,
            .byte_size = byte_size,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
        };
        return buffer.accountDeviceBytes();
    }

    pub fn initPendingBackendTransfer(
        allocator: std.mem.Allocator,
        backend_impl: backend_api.Backend,
        element_type: BufferType,
        dims_: []const i64,
        device: *Device,
        memory: *Memory,
        shard_index: usize,
    ) PendingBackendTransferBufferError!*Buffer {
        if (!memory.isAddressableBy(device)) return error.InvalidArgument;

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, dims_);
        errdefer allocator.free(dims);

        const bytes = try allocator.alloc(u8, 0);
        errdefer allocator.free(bytes);

        buffer.* = .{
            .allocator = allocator,
            .backend = backend_impl,
            .element_type = element_type,
            .dims = dims,
            .device_id = device.id,
            .memory_id = memory.id,
            .device = device,
            .memory = memory,
            .shard_index = shard_index,
            .byte_size = ir.denseByteSize(element_type, dims),
            .bytes = bytes,
            .backend_buffer = null,
            .ready_event = Event.pending(),
        };
        return buffer;
    }

    pub fn initDeviceCopy(
        allocator: std.mem.Allocator,
        src: *Buffer,
        shard_index: usize,
    ) !*Buffer {
        try src.ensureUsable();
        const src_backend = src.backend_buffer orelse return error.UnsupportedRuntimeFeature;
        const backend_buffer = try src.backend.cloneBuffer(src_backend) orelse return error.UnsupportedRuntimeFeature;
        errdefer src.backend.destroyBuffer(backend_buffer);

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, src.dims);
        errdefer allocator.free(dims);

        const bytes = try allocator.alloc(u8, 0);
        errdefer allocator.free(bytes);

        buffer.* = .{
            .allocator = allocator,
            .backend = src.backend,
            .element_type = src.element_type,
            .dims = dims,
            .device_id = src.device_id,
            .memory_id = src.memory_id,
            .device = src.device,
            .memory = src.memory,
            .shard_index = shard_index,
            .byte_size = src.byte_size,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
        };
        return buffer.accountDeviceBytes();
    }

    pub fn initPartitionId(
        allocator: std.mem.Allocator,
        backend_impl: backend_api.Backend,
        output_type: BufferType,
        output_dims: []const i64,
        device: *Device,
        memory: *Memory,
        partition_id: u32,
        shard_index: usize,
    ) !*Buffer {
        if (output_dims.len != 0) return error.ShapeMismatch;
        if (output_type != .u32 and output_type != .s32) return error.UnsupportedElementType;
        if (!memory.isAddressableBy(device)) return error.InvalidArgument;

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, output_dims);
        errdefer allocator.free(dims);

        const backend_buffer = try backend_impl.partitionId(device.local_hardware_id, output_type, partition_id) orelse return error.UnsupportedRuntimeFeature;
        errdefer backend_impl.destroyBuffer(backend_buffer);
        const storage_bytes = try allocator.alloc(u8, 0);
        errdefer allocator.free(storage_bytes);

        buffer.* = .{
            .allocator = allocator,
            .backend = backend_impl,
            .element_type = output_type,
            .dims = dims,
            .device_id = device.id,
            .memory_id = memory.id,
            .device = device,
            .memory = memory,
            .shard_index = shard_index,
            .byte_size = output_type.byteSize(),
            .bytes = storage_bytes,
            .backend_buffer = backend_buffer,
        };
        return buffer.accountDeviceBytes();
    }

    pub fn initCholesky(
        allocator: std.mem.Allocator,
        src: *Buffer,
        lower: bool,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (src.element_type != .f32) return error.UnsupportedElementType;
        if (!std.mem.eql(i64, src.dims, output_dims) or output_dims.len < 2) return error.ShapeMismatch;
        const n_i64 = output_dims[output_dims.len - 1];
        const rows_i64 = output_dims[output_dims.len - 2];
        if (n_i64 <= 0 or rows_i64 != n_i64) return error.ShapeMismatch;
        const backend_src = src.backend_buffer orelse return error.UnsupportedRuntimeFeature;
        const output_byte_size = denseByteSize(.f32, output_dims);
        if (src.backend.cholesky(backend_src, lower, output_dims) catch null) |backend_buffer| {
            return initBackendOnly(allocator, src, .f32, output_dims, output_byte_size, backend_buffer, shard_index);
        }
        return error.UnsupportedRuntimeFeature;
    }

    pub fn initRngUniform(
        allocator: std.mem.Allocator,
        min: *Buffer,
        max: *Buffer,
        output_type: BufferType,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        _ = allocator;
        _ = output_dims;
        _ = shard_index;
        if (output_type != .f32 and output_type != .u32 and output_type != .s32) return error.UnsupportedElementType;
        if (min.element_type != max.element_type or min.element_type != output_type) return error.UnsupportedElementType;
        return error.UnsupportedRuntimeFeature;
    }

    pub fn initRngBits(
        allocator: std.mem.Allocator,
        state: *Buffer,
        output_type: BufferType,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        _ = allocator;
        _ = state;
        _ = output_dims;
        _ = shard_index;
        if (output_type != .u32 and output_type != .s32) return error.UnsupportedElementType;
        return error.UnsupportedRuntimeFeature;
    }

    pub fn initRngStateUpdate(
        allocator: std.mem.Allocator,
        state: *Buffer,
        shard_index: usize,
    ) !*Buffer {
        _ = allocator;
        _ = shard_index;
        if (state.element_type.byteSize() == 0) return error.UnsupportedElementType;
        return error.UnsupportedRuntimeFeature;
    }

    pub fn initElementwiseBinary(
        allocator: std.mem.Allocator,
        op: ElementwiseBinaryOp,
        lhs: *Buffer,
        rhs: *Buffer,
        shard_index: usize,
    ) !*Buffer {
        if (lhs.element_type != rhs.element_type) return error.UnsupportedElementType;
        if (lhs.element_type.byteSize() == 0) return error.UnsupportedElementType;
        if (!std.mem.eql(i64, lhs.dims, rhs.dims) or lhs.byte_size != rhs.byte_size) return error.ShapeMismatch;
        if (lhs.backend_buffer != null and rhs.backend_buffer != null) {
            if (lhs.backend.binary(lhs.backend_buffer.?, rhs.backend_buffer.?, op) catch null) |backend_buffer| {
                return initBackendOnly(allocator, lhs, lhs.element_type, lhs.dims, lhs.byte_size, backend_buffer, shard_index);
            }
        }
        return error.UnsupportedRuntimeFeature;
    }

    pub fn initU8Add(
        allocator: std.mem.Allocator,
        lhs: *Buffer,
        rhs: *Buffer,
        shard_index: usize,
    ) !*Buffer {
        return initElementwiseBinary(allocator, .add, lhs, rhs, shard_index);
    }

    pub fn initU8Binary(
        allocator: std.mem.Allocator,
        op: ElementwiseBinaryOp,
        lhs: *Buffer,
        rhs: *Buffer,
        shard_index: usize,
    ) !*Buffer {
        if (lhs.element_type != .u8 or rhs.element_type != .u8) return error.UnsupportedElementType;
        return initElementwiseBinary(allocator, op, lhs, rhs, shard_index);
    }

    pub fn initElementwiseUnary(
        allocator: std.mem.Allocator,
        op: ElementwiseUnaryOp,
        src: *Buffer,
        shard_index: usize,
    ) !*Buffer {
        return initElementwiseUnaryTyped(allocator, op, src, src.element_type, src.dims, shard_index);
    }

    pub fn initElementwiseUnaryTyped(
        allocator: std.mem.Allocator,
        op: ElementwiseUnaryOp,
        src: *Buffer,
        output_type: BufferType,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (!std.mem.eql(i64, src.dims, output_dims)) return error.ShapeMismatch;
        if (src.element_type.byteSize() == 0 or output_type.byteSize() == 0) return error.UnsupportedElementType;
        const output_byte_size = denseByteSize(output_type, output_dims);
        if (src.backend_buffer) |src_backend| {
            if (src.backend.unary(src_backend, op) catch null) |backend_buffer| {
                return initBackendOnly(allocator, src, output_type, output_dims, output_byte_size, backend_buffer, shard_index);
            }
        }
        return error.UnsupportedRuntimeFeature;
    }

    pub fn initReshape(
        allocator: std.mem.Allocator,
        src: *Buffer,
        new_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (src.element_type.byteSize() == 0) return error.UnsupportedElementType;
        if (denseByteSize(src.element_type, new_dims) != src.byte_size) return error.ShapeMismatch;
        if (src.backend_buffer) |src_backend| {
            if (src.backend.reshape(src_backend, new_dims) catch null) |backend_buffer| {
                return initBackendOnly(allocator, src, src.element_type, new_dims, src.byte_size, backend_buffer, shard_index);
            }
        }
        return error.UnsupportedRuntimeFeature;
    }

    pub fn initTranspose(
        allocator: std.mem.Allocator,
        src: *Buffer,
        permutation: []const i64,
        new_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (!validPermutation(permutation, src.dims.len)) return error.ShapeMismatch;
        if (new_dims.len != permutation.len) return error.ShapeMismatch;
        if (denseByteSize(src.element_type, new_dims) != src.byte_size) return error.ShapeMismatch;
        const element_size = src.element_type.byteSize();
        if (element_size == 0) return error.UnsupportedElementType;
        if (src.backend_buffer) |src_backend| {
            if (src.backend.transpose(src_backend, permutation) catch null) |backend_buffer| {
                return initBackendOnly(allocator, src, src.element_type, new_dims, src.byte_size, backend_buffer, shard_index);
            }
        }
        return error.UnsupportedRuntimeFeature;
    }

    pub fn initBroadcastInDim(
        allocator: std.mem.Allocator,
        src: *Buffer,
        broadcast_dimensions: []const i64,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (!validBroadcastDimensions(broadcast_dimensions, src.dims, output_dims)) return error.ShapeMismatch;
        const element_size = src.element_type.byteSize();
        if (element_size == 0) return error.UnsupportedElementType;
        const output_byte_size = denseByteSize(src.element_type, output_dims);
        if (src.backend_buffer) |src_backend| {
            if (src.backend.broadcastInDim(src_backend, broadcast_dimensions, output_dims) catch null) |backend_buffer| {
                return initBackendOnly(allocator, src, src.element_type, output_dims, output_byte_size, backend_buffer, shard_index);
            }
        }
        return error.UnsupportedRuntimeFeature;
    }

    pub fn initSlice(
        allocator: std.mem.Allocator,
        src: *Buffer,
        start_indices: []const i64,
        limit_indices: []const i64,
        strides_: []const i64,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (!validSlice(start_indices, limit_indices, strides_, src.dims, output_dims)) return error.ShapeMismatch;
        const element_size = src.element_type.byteSize();
        if (element_size == 0) return error.UnsupportedElementType;
        const output_byte_size = denseByteSize(src.element_type, output_dims);
        if (src.backend_buffer) |src_backend| {
            if (src.backend.slice(src_backend, start_indices, limit_indices, strides_, output_dims) catch null) |backend_buffer| {
                return initBackendOnly(allocator, src, src.element_type, output_dims, output_byte_size, backend_buffer, shard_index);
            }
        }
        return error.UnsupportedRuntimeFeature;
    }

    pub fn initConcatenate(
        allocator: std.mem.Allocator,
        lhs: *Buffer,
        rhs: *Buffer,
        dimension: i64,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (lhs.element_type != rhs.element_type) return error.UnsupportedElementType;
        if (!validConcatenate(lhs.dims, rhs.dims, dimension, output_dims)) return error.ShapeMismatch;
        const element_size = lhs.element_type.byteSize();
        if (element_size == 0) return error.UnsupportedElementType;
        const output_byte_size = denseByteSize(lhs.element_type, output_dims);
        if (lhs.backend_buffer != null and rhs.backend_buffer != null) {
            if (lhs.backend.concatenate(lhs.backend_buffer.?, rhs.backend_buffer.?, dimension, output_dims) catch null) |backend_buffer| {
                return initBackendOnly(allocator, lhs, lhs.element_type, output_dims, output_byte_size, backend_buffer, shard_index);
            }
        }
        return error.UnsupportedRuntimeFeature;
    }

    pub fn initDotGeneral(
        allocator: std.mem.Allocator,
        lhs: *Buffer,
        rhs: *Buffer,
        lhs_batch_dimensions: []const i64,
        rhs_batch_dimensions: []const i64,
        lhs_contracting_dimensions: []const i64,
        rhs_contracting_dimensions: []const i64,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (lhs.element_type != rhs.element_type) return error.UnsupportedElementType;
        switch (lhs.element_type) {
            .f16, .bf16, .f32 => {},
            else => return error.UnsupportedElementType,
        }
        if (!validDotGeneral(lhs.dims, rhs.dims, lhs_batch_dimensions, rhs_batch_dimensions, lhs_contracting_dimensions, rhs_contracting_dimensions, output_dims)) return error.ShapeMismatch;
        const output_byte_size = denseByteSize(lhs.element_type, output_dims);
        if (lhs.backend_buffer != null and rhs.backend_buffer != null) {
            if (lhs.backend.dotGeneral(lhs.backend_buffer.?, rhs.backend_buffer.?, lhs_batch_dimensions, rhs_batch_dimensions, lhs_contracting_dimensions, rhs_contracting_dimensions, output_dims) catch null) |backend_buffer| {
                return initBackendOnly(allocator, lhs, lhs.element_type, output_dims, output_byte_size, backend_buffer, shard_index);
            }
        }
        return error.UnsupportedRuntimeFeature;
    }

    pub fn initReduce(
        allocator: std.mem.Allocator,
        kind: ir.PlanInstructionKind,
        src: *Buffer,
        dimensions: []const i64,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        const output_type: BufferType = switch (kind) {
            .reduce_sum, .reduce_max, .reduce_min => switch (src.element_type) {
                .s8, .s32, .u8, .u16, .u32, .u64, .f16, .f32, .bf16 => src.element_type,
                else => return error.UnsupportedElementType,
            },
            .reduce_and, .reduce_or => if (src.element_type == .pred) .pred else return error.UnsupportedElementType,
            else => return error.UnsupportedElementType,
        };
        if (!validReduce(src.dims, dimensions, output_dims)) return error.ShapeMismatch;
        const output_byte_size = denseByteSize(output_type, output_dims);
        if (src.backend_buffer) |src_backend| {
            if (src.backend.reduce(src_backend, kind, dimensions, output_dims) catch null) |backend_buffer| {
                return initBackendOnly(allocator, src, output_type, output_dims, output_byte_size, backend_buffer, shard_index);
            }
        }
        return error.UnsupportedRuntimeFeature;
    }

    pub fn initCompare(
        allocator: std.mem.Allocator,
        lhs: *Buffer,
        rhs: *Buffer,
        direction: CompareOp,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (lhs.element_type != rhs.element_type) return error.UnsupportedElementType;
        if (!std.mem.eql(i64, lhs.dims, rhs.dims) or !std.mem.eql(i64, lhs.dims, output_dims)) return error.ShapeMismatch;
        const output_byte_size = denseByteSize(.pred, output_dims);
        if (lhs.backend_buffer != null and rhs.backend_buffer != null) {
            if (lhs.backend.compare(lhs.backend_buffer.?, rhs.backend_buffer.?, direction, output_dims) catch null) |backend_buffer| {
                return initBackendOnly(allocator, lhs, .pred, output_dims, output_byte_size, backend_buffer, shard_index);
            }
        }
        return error.UnsupportedRuntimeFeature;
    }

    pub fn initSelect(
        allocator: std.mem.Allocator,
        pred: *Buffer,
        on_true: *Buffer,
        on_false: *Buffer,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (pred.element_type != .pred) return error.UnsupportedElementType;
        if (on_true.element_type != on_false.element_type) return error.UnsupportedElementType;
        if (!std.mem.eql(i64, on_true.dims, on_false.dims) or !std.mem.eql(i64, on_true.dims, output_dims) or !std.mem.eql(i64, pred.dims, output_dims)) return error.ShapeMismatch;
        const element_size = on_true.element_type.byteSize();
        if (element_size == 0) return error.UnsupportedElementType;
        if (pred.backend_buffer != null and on_true.backend_buffer != null and on_false.backend_buffer != null) {
            if (on_true.backend.select(pred.backend_buffer.?, on_true.backend_buffer.?, on_false.backend_buffer.?, output_dims) catch null) |backend_buffer| {
                return initBackendOnly(allocator, on_true, on_true.element_type, output_dims, denseByteSize(on_true.element_type, output_dims), backend_buffer, shard_index);
            }
        }
        return error.UnsupportedRuntimeFeature;
    }

    pub fn initConvert(
        allocator: std.mem.Allocator,
        src: *Buffer,
        output_type: BufferType,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (!std.mem.eql(i64, src.dims, output_dims)) return error.ShapeMismatch;
        if (src.element_type.byteSize() == 0 or output_type.byteSize() == 0) return error.UnsupportedElementType;
        const output_size = denseByteSize(output_type, output_dims);
        if (src.backend_buffer) |src_backend| {
            if (src.backend.convert(src_backend, output_type) catch null) |backend_buffer| {
                return initBackendOnly(allocator, src, output_type, output_dims, output_size, backend_buffer, shard_index);
            }
        }
        return error.UnsupportedRuntimeFeature;
    }

    pub fn initIota(
        allocator: std.mem.Allocator,
        backend_impl: backend_api.Backend,
        element_type: BufferType,
        output_dims: []const i64,
        device: *Device,
        memory: *Memory,
        iota_dimension: i64,
        shard_index: usize,
    ) !*Buffer {
        if (iota_dimension < 0 or iota_dimension >= @as(i64, @intCast(output_dims.len))) return error.ShapeMismatch;
        if (element_type.byteSize() == 0) return error.UnsupportedElementType;
        if (backend_impl.iota(device.local_hardware_id, element_type, output_dims, iota_dimension) catch null) |backend_buffer| {
            return initBackendHandle(
                allocator,
                backend_impl,
                element_type,
                output_dims,
                device,
                memory,
                shard_index,
                denseByteSize(element_type, output_dims),
                backend_buffer,
            ) catch |err| {
                backend_impl.destroyBuffer(backend_buffer);
                return err;
            };
        }
        return error.UnsupportedRuntimeFeature;
    }

    pub fn initReverse(
        allocator: std.mem.Allocator,
        src: *Buffer,
        dimensions: []const i64,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (!std.mem.eql(i64, src.dims, output_dims)) return error.ShapeMismatch;
        const element_size = src.element_type.byteSize();
        if (element_size == 0) return error.UnsupportedElementType;
        if (src.backend_buffer) |src_backend| {
            if (src.backend.reverse(src_backend, dimensions, output_dims) catch null) |backend_buffer| {
                return initBackendOnly(allocator, src, src.element_type, output_dims, src.byte_size, backend_buffer, shard_index);
            }
        }
        return error.UnsupportedRuntimeFeature;
    }

    pub fn initClamp(
        allocator: std.mem.Allocator,
        min_buffer: *Buffer,
        value_buffer: *Buffer,
        max_buffer: *Buffer,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (value_buffer.element_type != min_buffer.element_type or value_buffer.element_type != max_buffer.element_type) return error.UnsupportedElementType;
        if (!std.mem.eql(i64, value_buffer.dims, output_dims)) return error.ShapeMismatch;
        const element_size = value_buffer.element_type.byteSize();
        if (element_size == 0) return error.UnsupportedElementType;
        const min_scalar = min_buffer.dims.len == 0;
        const max_scalar = max_buffer.dims.len == 0;
        if (!min_scalar and !std.mem.eql(i64, min_buffer.dims, output_dims)) return error.ShapeMismatch;
        if (!max_scalar and !std.mem.eql(i64, max_buffer.dims, output_dims)) return error.ShapeMismatch;
        if (min_buffer.backend_buffer) |min_backend| {
            if (value_buffer.backend_buffer) |value_backend| {
                if (max_buffer.backend_buffer) |max_backend| {
                    if (value_buffer.backend.clamp(min_backend, value_backend, max_backend, output_dims) catch null) |backend_buffer| {
                        return initBackendOnly(allocator, value_buffer, value_buffer.element_type, output_dims, denseByteSize(value_buffer.element_type, output_dims), backend_buffer, shard_index);
                    }
                }
            }
        }
        return error.UnsupportedRuntimeFeature;
    }

    pub fn initDynamicSlice(
        allocator: std.mem.Allocator,
        src: *Buffer,
        start_buffers: []const *Buffer,
        slice_sizes: []const i64,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (start_buffers.len != src.dims.len or slice_sizes.len != src.dims.len) return error.ShapeMismatch;
        if (src.backend_buffer) |src_backend| {
            const start_handles = try backendStartHandles(allocator, start_buffers);
            defer allocator.free(start_handles);
            if (start_handles.len == start_buffers.len) {
                if (src.backend.dynamicSlice(src_backend, start_handles, slice_sizes, output_dims) catch null) |backend_buffer| {
                    return initBackendOnly(allocator, src, src.element_type, output_dims, denseByteSize(src.element_type, output_dims), backend_buffer, shard_index);
                }
            }
        }
        return error.UnsupportedRuntimeFeature;
    }

    pub fn initDynamicUpdateSlice(
        allocator: std.mem.Allocator,
        src: *Buffer,
        update: *Buffer,
        start_buffers: []const *Buffer,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (src.element_type != update.element_type) return error.UnsupportedElementType;
        if (!std.mem.eql(i64, src.dims, output_dims) or src.dims.len != update.dims.len or start_buffers.len != src.dims.len) return error.ShapeMismatch;
        const element_size = src.element_type.byteSize();
        if (element_size == 0) return error.UnsupportedElementType;
        if (src.backend_buffer != null and update.backend_buffer != null) {
            const start_handles = try backendStartHandles(allocator, start_buffers);
            defer allocator.free(start_handles);
            if (start_handles.len == start_buffers.len) {
                if (src.backend.dynamicUpdateSlice(src.backend_buffer.?, update.backend_buffer.?, start_handles, output_dims) catch null) |backend_buffer| {
                    return initBackendOnly(allocator, src, src.element_type, output_dims, denseByteSize(src.element_type, output_dims), backend_buffer, shard_index);
                }
            }
        }
        return error.UnsupportedRuntimeFeature;
    }

    pub fn initPad(
        allocator: std.mem.Allocator,
        src: *Buffer,
        padding_value: *Buffer,
        edge_padding_low: []const i64,
        edge_padding_high: []const i64,
        interior_padding: []const i64,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (src.element_type != padding_value.element_type or padding_value.dims.len != 0) return error.UnsupportedElementType;
        const rank = src.dims.len;
        if (edge_padding_low.len != rank or edge_padding_high.len != rank or interior_padding.len != rank or output_dims.len != rank) return error.ShapeMismatch;
        const element_size = src.element_type.byteSize();
        if (element_size == 0) return error.UnsupportedElementType;
        if (src.backend_buffer != null and padding_value.backend_buffer != null) {
            if (src.backend.pad(src.backend_buffer.?, padding_value.backend_buffer.?, edge_padding_low, edge_padding_high, interior_padding, output_dims) catch null) |backend_buffer| {
                return initBackendOnly(allocator, src, src.element_type, output_dims, denseByteSize(src.element_type, output_dims), backend_buffer, shard_index);
            }
        }
        return error.UnsupportedRuntimeFeature;
    }

    pub fn initGather(
        allocator: std.mem.Allocator,
        operand: *Buffer,
        indices: *Buffer,
        offset_dims: []const i64,
        collapsed_slice_dims: []const i64,
        operand_batching_dims: []const i64,
        start_indices_batching_dims: []const i64,
        start_index_map: []const i64,
        index_vector_dim: i64,
        slice_sizes: []const i64,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (operand.element_type.byteSize() == 0) return error.UnsupportedElementType;
        if (start_index_map.len == 0 or slice_sizes.len != operand.dims.len) return error.ShapeMismatch;
        if (index_vector_dim < 0) return error.ShapeMismatch;

        if (operand.backend_buffer != null and indices.backend_buffer != null) {
            if (supportedGatherAxisForRuntime(collapsed_slice_dims, start_index_map, slice_sizes)) |gather_axis| {
                if (operand.backend.gatherAxis(operand.backend_buffer.?, indices.backend_buffer.?, gather_axis, index_vector_dim, output_dims) catch null) |backend_buffer| {
                    return initBackendOnly(allocator, operand, operand.element_type, output_dims, denseByteSize(operand.element_type, output_dims), backend_buffer, shard_index);
                }
            }
            if (operand.backend.gather(
                operand.backend_buffer.?,
                indices.backend_buffer.?,
                start_index_map,
                collapsed_slice_dims,
                operand_batching_dims,
                start_indices_batching_dims,
                index_vector_dim,
                slice_sizes,
                offset_dims,
                output_dims,
            ) catch null) |backend_buffer| {
                return initBackendOnly(allocator, operand, operand.element_type, output_dims, denseByteSize(operand.element_type, output_dims), backend_buffer, shard_index);
            }
        }
        return error.UnsupportedRuntimeFeature;
    }

    pub fn initSort(
        allocator: std.mem.Allocator,
        src: *Buffer,
        dimension: i64,
        output_dims: []const i64,
        direction: CompareOp,
        shard_index: usize,
    ) !*Buffer {
        if (!std.mem.eql(i64, src.dims, output_dims)) return error.ShapeMismatch;
        if (dimension < 0 or dimension >= @as(i64, @intCast(src.dims.len))) return error.ShapeMismatch;
        const element_size = src.element_type.byteSize();
        if (element_size == 0) return error.UnsupportedElementType;
        const ascending = switch (direction) {
            .lt, .le => true,
            .gt, .ge => false,
            else => return error.UnsupportedElementType,
        };
        if (src.backend_buffer) |src_backend| {
            if (src.backend.sort(src_backend, dimension, output_dims) catch null) |sorted_backend| {
                if (ascending) {
                    return initBackendOnly(allocator, src, src.element_type, output_dims, src.byte_size, sorted_backend, shard_index);
                }
                const reverse_dimensions = [_]i64{dimension};
                if (src.backend.reverse(sorted_backend, &reverse_dimensions, output_dims) catch null) |backend_buffer| {
                    src.backend.destroyBuffer(sorted_backend);
                    return initBackendOnly(allocator, src, src.element_type, output_dims, src.byte_size, backend_buffer, shard_index);
                }
                src.backend.destroyBuffer(sorted_backend);
            }
        }
        return error.UnsupportedRuntimeFeature;
    }

    pub fn initU8Unary(
        allocator: std.mem.Allocator,
        op: ElementwiseUnaryOp,
        src: *Buffer,
        shard_index: usize,
    ) !*Buffer {
        if (src.element_type != .u8) return error.UnsupportedElementType;
        return initElementwiseUnary(allocator, op, src, shard_index);
    }

    pub fn deinit(self: *Buffer) void {
        self.releaseStorage();
        self.allocator.free(self.bytes);
        self.allocator.free(self.dims);
        self.allocator.destroy(self);
    }

    pub fn copyToHost(self: *Buffer, dst: []u8) HostReadbackError!void {
        try self.ensureUsable();
        try self.ensureReady();
        if (dst.len < self.byte_size) return error.DestinationTooSmall;
        const backend_buffer = self.backend_buffer orelse return error.UnsupportedRuntimeFeature;
        self.backend.copyToHost(backend_buffer, dst) catch return error.BackendBufferCopyFailed;
        self.memory.stats.device_to_host_bytes += @intCast(self.byte_size);
    }

    pub fn hasBackendStorage(self: *const Buffer) bool {
        return self.backend_buffer != null;
    }

    pub fn markDeleted(self: *Buffer) void {
        self.releaseStorage();
        self.state = .deleted;
        self.deleted = true;
        self.ready_event.setFailed("buffer has been deleted");
    }

    pub fn markDonated(self: *Buffer) void {
        self.releaseStorage();
        self.state = .donated;
        self.deleted = true;
        self.ready_event.setFailed("buffer has been donated");
    }

    pub fn takeBackendStorageForDonationAlias(self: *Buffer) !backend_api.BufferHandle {
        try self.ensureUsable();
        const backend_buffer = self.backend_buffer orelse return error.UnsupportedRuntimeFeature;
        self.backend_buffer = null;
        if (self.accounted_bytes != 0) {
            self.memory.stats.release(self.accounted_bytes);
            self.accounted_bytes = 0;
        }
        return backend_buffer;
    }

    pub fn replaceBackendStorage(self: *Buffer, backend_buffer: backend_api.BufferHandle) error{ BufferDeleted, BufferDonated }!void {
        try self.ensureUsable();
        self.releaseStorage();
        self.backend_buffer = backend_buffer;
        _ = self.accountDeviceBytes();
    }

    pub fn ensureUsable(self: *const Buffer) !void {
        return switch (self.state) {
            .live => {},
            .deleted => error.BufferDeleted,
            .donated => error.BufferDonated,
        };
    }

    pub fn ensureReady(self: *const Buffer) !void {
        self.ready_event.awaitReady() catch |err| return switch (err) {
            error.EventPending => error.BufferNotReady,
            error.EventFailed => error.BufferReadinessFailed,
        };
    }

    pub fn chainReadyAfter(self: *Buffer, dependency: *Event) !void {
        try self.ensureUsable();
        self.ready_event = Event.pending();
        try dependency.chainTo(&self.ready_event);
    }

    fn accountDeviceBytes(self: *Buffer) *Buffer {
        self.accounted_bytes = self.byte_size;
        self.memory.stats.retain(self.accounted_bytes);
        return self;
    }

    fn releaseStorage(self: *Buffer) void {
        if (self.accounted_bytes != 0) {
            self.memory.stats.release(self.accounted_bytes);
            self.accounted_bytes = 0;
        }
        if (self.backend_buffer) |backend_buffer| {
            self.backend.destroyBuffer(backend_buffer);
            self.backend_buffer = null;
        }
    }

    pub fn initBackendHandle(
        allocator: std.mem.Allocator,
        backend_impl: backend_api.Backend,
        element_type: BufferType,
        dims_: []const i64,
        device: *Device,
        memory: *Memory,
        shard_index: usize,
        byte_size: usize,
        backend_buffer: backend_api.BufferHandle,
    ) !*Buffer {
        if (!memory.isAddressableBy(device)) return error.InvalidArgument;

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, dims_);
        errdefer allocator.free(dims);

        const bytes = try allocator.alloc(u8, 0);
        errdefer allocator.free(bytes);

        buffer.* = .{
            .allocator = allocator,
            .backend = backend_impl,
            .element_type = element_type,
            .dims = dims,
            .device_id = device.id,
            .memory_id = memory.id,
            .device = device,
            .memory = memory,
            .shard_index = shard_index,
            .byte_size = byte_size,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
        };
        return buffer.accountDeviceBytes();
    }
};

fn denseByteSize(element_type: BufferType, dims: []const i64) usize {
    return ir.denseByteSize(element_type, dims);
}

fn validPermutation(permutation: []const i64, rank: usize) bool {
    if (permutation.len != rank) return false;
    var seen_storage = [_]bool{false} ** 16;
    if (rank > seen_storage.len) return false;
    const seen = seen_storage[0..rank];
    for (permutation) |axis| {
        if (axis < 0 or axis >= @as(i64, @intCast(rank))) return false;
        const axis_index: usize = @intCast(axis);
        if (seen[axis_index]) return false;
        seen[axis_index] = true;
    }
    return true;
}

fn validBroadcastDimensions(broadcast_dimensions: []const i64, input_dims: []const i64, output_dims: []const i64) bool {
    if (broadcast_dimensions.len != input_dims.len) return false;
    var seen_storage = [_]bool{false} ** 64;
    if (output_dims.len > seen_storage.len) return false;
    const seen = seen_storage[0..output_dims.len];
    for (broadcast_dimensions, 0..) |axis, input_axis| {
        if (axis < 0 or axis >= @as(i64, @intCast(output_dims.len))) return false;
        const output_axis: usize = @intCast(axis);
        if (seen[output_axis]) return false;
        seen[output_axis] = true;
        if (input_dims[input_axis] != 1 and input_dims[input_axis] != output_dims[output_axis]) return false;
    }
    return true;
}

fn validSlice(start_indices: []const i64, limit_indices: []const i64, strides: []const i64, input_dims: []const i64, output_dims: []const i64) bool {
    const rank = input_dims.len;
    if (start_indices.len != rank or limit_indices.len != rank or strides.len != rank or output_dims.len != rank) return false;
    for (0..rank) |axis| {
        const input_dim = input_dims[axis];
        const start = start_indices[axis];
        const limit = limit_indices[axis];
        const stride = strides[axis];
        const output_dim = output_dims[axis];
        if (input_dim < 0 or output_dim < 0) return false;
        if (start < 0 or limit < start or limit > input_dim or stride <= 0) return false;
        const span = limit - start;
        const expected = if (span == 0) 0 else @divTrunc(span + stride - 1, stride);
        if (output_dim != expected) return false;
    }
    return true;
}

fn validConcatenate(lhs_dims: []const i64, rhs_dims: []const i64, dimension: i64, output_dims: []const i64) bool {
    const rank = lhs_dims.len;
    if (rank == 0 or rhs_dims.len != rank or output_dims.len != rank) return false;
    if (dimension < 0 or dimension >= @as(i64, @intCast(rank))) return false;
    const concat_axis: usize = @intCast(dimension);
    for (0..rank) |axis| {
        if (lhs_dims[axis] < 0 or rhs_dims[axis] < 0 or output_dims[axis] < 0) return false;
        const expected = if (axis == concat_axis) lhs_dims[axis] + rhs_dims[axis] else lhs_dims[axis];
        if (output_dims[axis] != expected) return false;
        if (axis != concat_axis and lhs_dims[axis] != rhs_dims[axis]) return false;
    }
    return true;
}

fn validDotGeneral(
    lhs_dims: []const i64,
    rhs_dims: []const i64,
    lhs_batch_dimensions: []const i64,
    rhs_batch_dimensions: []const i64,
    lhs_contracting_dimensions: []const i64,
    rhs_contracting_dimensions: []const i64,
    output_dims: []const i64,
) bool {
    if (lhs_contracting_dimensions.len != 1 or rhs_contracting_dimensions.len != 1) return false;
    if (lhs_batch_dimensions.len != rhs_batch_dimensions.len) return false;
    const lhs_contract: usize = if (lhs_contracting_dimensions[0] < 0) return false else @intCast(lhs_contracting_dimensions[0]);
    const rhs_contract: usize = if (rhs_contracting_dimensions[0] < 0) return false else @intCast(rhs_contracting_dimensions[0]);
    if (lhs_contract >= lhs_dims.len or rhs_contract >= rhs_dims.len) return false;
    if (lhs_dims[lhs_contract] != rhs_dims[rhs_contract]) return false;
    for (lhs_batch_dimensions, rhs_batch_dimensions) |lhs_axis_i64, rhs_axis_i64| {
        if (lhs_axis_i64 < 0 or rhs_axis_i64 < 0) return false;
        const lhs_axis: usize = @intCast(lhs_axis_i64);
        const rhs_axis: usize = @intCast(rhs_axis_i64);
        if (lhs_axis >= lhs_dims.len or rhs_axis >= rhs_dims.len or lhs_dims[lhs_axis] != rhs_dims[rhs_axis]) return false;
    }
    return denseByteSize(.f32, output_dims) != 0;
}

fn validReduce(input_dims: []const i64, dimensions: []const i64, output_dims: []const i64) bool {
    var reduced = [_]bool{false} ** 64;
    if (input_dims.len > reduced.len) return false;
    for (dimensions) |dim_i64| {
        if (dim_i64 < 0 or dim_i64 >= @as(i64, @intCast(input_dims.len))) return false;
        const dim: usize = @intCast(dim_i64);
        if (reduced[dim]) return false;
        reduced[dim] = true;
    }
    var expected_rank: usize = 0;
    for (0..input_dims.len) |axis| {
        if (!reduced[axis]) expected_rank += 1;
    }
    if (output_dims.len != expected_rank) return false;
    var out_axis: usize = 0;
    for (input_dims, 0..) |dim, axis| {
        if (!reduced[axis]) {
            if (output_dims[out_axis] != dim) return false;
            out_axis += 1;
        }
    }
    return true;
}

fn readU32LE(bytes: []const u8, index: usize) u32 {
    return std.mem.readInt(u32, bytes[index * 4 ..][0..4], .little);
}

fn backendStartHandles(allocator: std.mem.Allocator, start_buffers: []const *Buffer) ![]backend_api.BufferHandle {
    for (start_buffers) |start_buffer| {
        if (start_buffer.backend_buffer == null) return allocator.alloc(backend_api.BufferHandle, 0);
    }
    const handles = try allocator.alloc(backend_api.BufferHandle, start_buffers.len);
    errdefer allocator.free(handles);
    for (start_buffers, 0..) |start_buffer, index| {
        handles[index] = start_buffer.backend_buffer.?;
    }
    return handles;
}

fn supportedGatherAxisForRuntime(collapsed_slice_dims: []const i64, start_index_map: []const i64, slice_sizes: []const i64) ?i64 {
    if (collapsed_slice_dims.len != 1 or start_index_map.len != 1) return null;
    const axis = start_index_map[0];
    if (axis < 0 or collapsed_slice_dims[0] != axis) return null;
    if (slice_sizes.len == 0 or axis >= @as(i64, @intCast(slice_sizes.len)) or slice_sizes[@intCast(axis)] != 1) return null;
    return axis;
}

const BufferTestContext = struct {
    allocator: std.mem.Allocator,
    backend: backend_api.Backend,
    descriptors: []ir.DeviceDescriptor,
    device: *Device,
    memory: *Memory,
    addressable_device_ids: []i32,
    addressable_devices: []*Device,
    addressable_memories: []*Memory,

    fn init(allocator: std.mem.Allocator) !BufferTestContext {
        const backend = backend_api.create();
        const descriptors = try backend.enumerateDevices(allocator, 1);
        errdefer backend.releaseDeviceDescriptors(allocator, descriptors);
        if (descriptors.len == 0) return error.InvalidDeviceCount;

        const device = try allocator.create(Device);
        errdefer allocator.destroy(device);

        const memory = try allocator.create(Memory);
        errdefer allocator.destroy(memory);

        const addressable_device_ids = try allocator.alloc(i32, 1);
        errdefer allocator.free(addressable_device_ids);

        const addressable_devices = try allocator.alloc(*Device, 1);
        errdefer allocator.free(addressable_devices);

        const addressable_memories = try allocator.alloc(*Memory, 1);
        errdefer allocator.free(addressable_memories);

        const descriptor = descriptors[0];
        addressable_device_ids[0] = descriptor.id;
        memory.* = .{
            .id = descriptor.default_memory_id,
            .kind = .device,
            .debug_string = "device",
            .addressable_device_ids = addressable_device_ids,
            .addressable_devices = addressable_devices,
            .stats = .{ .capacity_bytes = descriptor.memory_bytes },
        };
        device.* = .{
            .id = descriptor.id,
            .local_hardware_id = descriptor.local_hardware_id,
            .registry_id = descriptor.registry_id,
            .process_index = descriptor.process_index,
            .addressable = descriptor.addressable,
            .name = descriptor.name,
            .debug_string = descriptor.debug_string,
            .memory_bytes = descriptor.memory_bytes,
            .has_unified_memory = descriptor.has_unified_memory,
            .default_memory_id = descriptor.default_memory_id,
            .default_memory = memory,
            .addressable_memories = addressable_memories,
        };
        addressable_devices[0] = device;
        addressable_memories[0] = memory;

        return .{
            .allocator = allocator,
            .backend = backend,
            .descriptors = descriptors,
            .device = device,
            .memory = memory,
            .addressable_device_ids = addressable_device_ids,
            .addressable_devices = addressable_devices,
            .addressable_memories = addressable_memories,
        };
    }

    fn deinit(self: *BufferTestContext) void {
        self.allocator.free(self.addressable_memories);
        self.allocator.free(self.addressable_devices);
        self.allocator.free(self.addressable_device_ids);
        self.allocator.destroy(self.memory);
        self.allocator.destroy(self.device);
        self.backend.releaseDeviceDescriptors(self.allocator, self.descriptors);
        self.* = undefined;
    }

    fn hostCopy(
        self: *BufferTestContext,
        element_type: BufferType,
        dims: []const i64,
        shard_index: usize,
        src: []const u8,
    ) !*Buffer {
        return Buffer.initHostCopyForBackend(self.allocator, self.backend, element_type, dims, self.device, self.memory, shard_index, src);
    }
};

fn expectBufferBytes(buffer: *Buffer, expected: []const u8) !void {
    const actual = try std.testing.allocator.alloc(u8, expected.len);
    defer std.testing.allocator.free(actual);
    try buffer.copyToHost(actual);
    try std.testing.expectEqualSlices(u8, expected, actual);
}

fn expectBufferF32(buffer: *Buffer, expected: []const f32) !void {
    const actual = try std.testing.allocator.alloc(u8, expected.len * @sizeOf(f32));
    defer std.testing.allocator.free(actual);
    try buffer.copyToHost(actual);
    const floats = std.mem.bytesAsSlice(f32, actual);
    for (expected, floats) |want, got| {
        try std.testing.expectApproxEqAbs(want, got, 0.0001);
    }
}
test "buffer keeps shard/device/memory ownership metadata" {
    var ctx = try BufferTestContext.init(std.testing.allocator);
    defer ctx.deinit();

    const dims = [_]i64{ 2, 2 };
    const data = [_]u8{ 1, 2, 3, 4 };
    const buffer = try ctx.hostCopy(.u8, &dims, 0, &data);
    defer buffer.deinit();

    try std.testing.expectEqual(@as(i32, 0), buffer.device_id);
    try std.testing.expectEqual(@as(i32, 0), buffer.memory_id);
    try std.testing.expectEqual(@as(i32, 0), buffer.device.id);
    try std.testing.expectEqual(@as(i32, 0), buffer.memory.id);
    try std.testing.expectEqual(@as(usize, 0), buffer.shard_index);
    try expectBufferBytes(buffer, &data);

    const rhs_data = [_]u8{ 10, 20, 30, 40 };
    const rhs = try ctx.hostCopy(.u8, &dims, 0, &rhs_data);
    defer rhs.deinit();
    const sum = try Buffer.initU8Add(std.testing.allocator, buffer, rhs, 0);
    defer sum.deinit();
    try expectBufferBytes(sum, &.{ 11, 22, 33, 44 });

    const difference = try Buffer.initU8Binary(std.testing.allocator, .subtract, rhs, buffer, 0);
    defer difference.deinit();
    try expectBufferBytes(difference, &.{ 9, 18, 27, 36 });

    const product = try Buffer.initU8Binary(std.testing.allocator, .multiply, buffer, rhs, 0);
    defer product.deinit();
    try expectBufferBytes(product, &.{ 10, 40, 90, 160 });

    const quotient = try Buffer.initU8Binary(std.testing.allocator, .divide, rhs, buffer, 0);
    defer quotient.deinit();
    try expectBufferBytes(quotient, &.{ 10, 10, 10, 10 });

    const negated = try Buffer.initU8Unary(std.testing.allocator, .negate, buffer, 0);
    defer negated.deinit();
    try expectBufferBytes(negated, &.{ 255, 254, 253, 252 });
}

test "buffer constructors reject memory not addressable by device" {
    var ctx = try BufferTestContext.init(std.testing.allocator);
    defer ctx.deinit();

    var invalid_device_ids = [_]i32{1234};
    var no_devices = [_]*Device{};
    var invalid_memory = Memory{
        .id = 99,
        .kind = .device,
        .debug_string = "unreachable test memory",
        .addressable_device_ids = invalid_device_ids[0..],
        .addressable_devices = no_devices[0..],
    };

    const dims = [_]i64{4};
    const data = [_]u8{ 1, 2, 3, 4 };
    try std.testing.expect(!invalid_memory.isAddressableBy(ctx.device));
    try std.testing.expectError(
        error.InvalidArgument,
        Buffer.initHostCopyForBackend(std.testing.allocator, ctx.backend, .u8, &dims, ctx.device, &invalid_memory, 0, &data),
    );
    try std.testing.expectError(
        error.InvalidArgument,
        Buffer.initPartitionId(std.testing.allocator, ctx.backend, .u32, &.{}, ctx.device, &invalid_memory, 0, 0),
    );

    if (try ctx.backend.bufferFromHost(ctx.device.local_hardware_id, .u8, &dims, &data)) |backend_buffer| {
        try std.testing.expectError(
            error.InvalidArgument,
            Buffer.initBackendHandle(std.testing.allocator, ctx.backend, .u8, &dims, ctx.device, &invalid_memory, 0, data.len, backend_buffer),
        );
        ctx.backend.destroyBuffer(backend_buffer);
    }
}

test "buffer lifecycle rejects deleted and donated buffers" {
    var ctx = try BufferTestContext.init(std.testing.allocator);
    defer ctx.deinit();

    const dims = [_]i64{4};
    const data = [_]u8{ 1, 2, 3, 4 };
    const before_deleted = ctx.memory.stats.bytes_in_use;
    const deleted = try ctx.hostCopy(.u8, &dims, 0, &data);
    defer deleted.deinit();
    try std.testing.expectEqual(before_deleted + data.len, ctx.memory.stats.bytes_in_use);
    try std.testing.expect(deleted.hasBackendStorage());
    deleted.markDeleted();
    try std.testing.expectEqual(BufferState.deleted, deleted.state);
    try std.testing.expect(deleted.deleted);
    try std.testing.expect(!deleted.hasBackendStorage());
    try std.testing.expectEqual(before_deleted, ctx.memory.stats.bytes_in_use);
    try std.testing.expectError(error.BufferDeleted, deleted.ensureUsable());

    const before_donated = ctx.memory.stats.bytes_in_use;
    const donated = try ctx.hostCopy(.u8, &dims, 0, &data);
    defer donated.deinit();
    try std.testing.expectEqual(before_donated + data.len, ctx.memory.stats.bytes_in_use);
    try std.testing.expect(donated.hasBackendStorage());
    donated.markDonated();
    try std.testing.expectEqual(BufferState.donated, donated.state);
    try std.testing.expect(donated.deleted);
    try std.testing.expect(!donated.hasBackendStorage());
    try std.testing.expectEqual(before_donated, ctx.memory.stats.bytes_in_use);
    try std.testing.expectError(error.BufferDonated, donated.ensureUsable());
}

test "buffer copies respect readiness events" {
    var ctx = try BufferTestContext.init(std.testing.allocator);
    defer ctx.deinit();

    const dims = [_]i64{4};
    const data = [_]u8{ 1, 2, 3, 4 };
    const buffer = try ctx.hostCopy(.u8, &dims, 0, &data);
    defer buffer.deinit();

    var out: [4]u8 = undefined;
    buffer.ready_event = Event.pending();
    try std.testing.expectError(error.BufferNotReady, buffer.ensureReady());
    try std.testing.expectError(error.BufferNotReady, buffer.copyToHost(&out));

    buffer.ready_event.setReady();
    try buffer.ensureReady();
    try buffer.copyToHost(&out);
    try std.testing.expectEqualSlices(u8, &data, &out);

    buffer.ready_event = Event.failed("producer failed");
    try std.testing.expectError(error.BufferReadinessFailed, buffer.ensureReady());
    try std.testing.expectError(error.BufferReadinessFailed, buffer.copyToHost(&out));
}

test "buffer elementwise arithmetic supports f32 execution" {
    var ctx = try BufferTestContext.init(std.testing.allocator);
    defer ctx.deinit();

    const dims = [_]i64{2};
    const lhs_values = [_]f32{ 1.5, -2.0 };
    const rhs_values = [_]f32{ 2.25, 4.0 };
    const lhs = try ctx.hostCopy(.f32, &dims, 0, std.mem.asBytes(&lhs_values));
    defer lhs.deinit();
    const rhs = try ctx.hostCopy(.f32, &dims, 0, std.mem.asBytes(&rhs_values));
    defer rhs.deinit();

    const sum = try Buffer.initElementwiseBinary(std.testing.allocator, .add, lhs, rhs, 0);
    defer sum.deinit();
    try expectBufferF32(sum, &.{ 3.75, 2.0 });

    const quotient = try Buffer.initElementwiseBinary(std.testing.allocator, .divide, rhs, lhs, 0);
    defer quotient.deinit();
    try expectBufferF32(quotient, &.{ 1.5, -2.0 });

    const negated = try Buffer.initElementwiseUnary(std.testing.allocator, .negate, lhs, 0);
    defer negated.deinit();
    try expectBufferF32(negated, &.{ -1.5, 2.0 });
}

test "buffer convert uses backend astype for resident buffers" {
    var ctx = try BufferTestContext.init(std.testing.allocator);
    defer ctx.deinit();

    const dims = [_]i64{2};
    const input = [_]u8{ 1, 255 };
    const source = try ctx.hostCopy(.u8, &dims, 0, &input);
    defer source.deinit();

    const converted = try Buffer.initConvert(std.testing.allocator, source, .f32, &dims, 0);
    defer converted.deinit();
    try std.testing.expect(converted.backend_buffer != null);
    try std.testing.expectEqual(@as(usize, 0), converted.bytes.len);
    try expectBufferF32(converted, &.{ 1.0, 255.0 });
}

test "buffer iota uses resident mlx backend path" {
    var ctx = try BufferTestContext.init(std.testing.allocator);
    defer ctx.deinit();

    const dims = [_]i64{ 2, 3 };
    const buffer = try Buffer.initIota(
        std.testing.allocator,
        ctx.backend,
        .f32,
        &dims,
        ctx.device,
        ctx.memory,
        1,
        0,
    );
    defer buffer.deinit();
    try std.testing.expect(buffer.backend_buffer != null);
    try std.testing.expectEqual(@as(usize, 0), buffer.bytes.len);
    try expectBufferF32(buffer, &.{ 0.0, 1.0, 2.0, 0.0, 1.0, 2.0 });
}

test "buffer movement ops use resident mlx backend paths" {
    var ctx = try BufferTestContext.init(std.testing.allocator);
    defer ctx.deinit();

    const dims = [_]i64{ 3, 4 };
    const values = [_]f32{
        1.0, 2.0,  3.0,  4.0,
        5.0, 6.0,  7.0,  8.0,
        9.0, 10.0, 11.0, 12.0,
    };
    const source = try ctx.hostCopy(.f32, &dims, 0, std.mem.asBytes(&values));
    defer source.deinit();

    const start0: i32 = 1;
    const start1: i32 = 1;
    const start0_buffer = try ctx.hostCopy(.s32, &.{}, 0, std.mem.asBytes(&start0));
    defer start0_buffer.deinit();
    const start1_buffer = try ctx.hostCopy(.s32, &.{}, 0, std.mem.asBytes(&start1));
    defer start1_buffer.deinit();

    const slice_dims = [_]i64{ 2, 2 };
    const sliced = try Buffer.initDynamicSlice(std.testing.allocator, source, &.{ start0_buffer, start1_buffer }, &slice_dims, &slice_dims, 0);
    defer sliced.deinit();
    try std.testing.expect(sliced.backend_buffer != null);
    try std.testing.expectEqual(@as(usize, 0), sliced.bytes.len);
    try expectBufferF32(sliced, &.{ 6.0, 7.0, 10.0, 11.0 });

    const update_values = [_]f32{ 100.0, 101.0, 102.0, 103.0 };
    const update = try ctx.hostCopy(.f32, &slice_dims, 0, std.mem.asBytes(&update_values));
    defer update.deinit();
    const updated = try Buffer.initDynamicUpdateSlice(std.testing.allocator, source, update, &.{ start0_buffer, start1_buffer }, &dims, 0);
    defer updated.deinit();
    try std.testing.expect(updated.backend_buffer != null);
    try std.testing.expectEqual(@as(usize, 0), updated.bytes.len);
    try expectBufferF32(updated, &.{
        1.0, 2.0,   3.0,   4.0,
        5.0, 100.0, 101.0, 8.0,
        9.0, 102.0, 103.0, 12.0,
    });

    const pad_input_dims = [_]i64{2};
    const pad_output_dims = [_]i64{5};
    const pad_input_values = [_]f32{ 2.0, 3.0 };
    const pad_value: f32 = 0.0;
    const pad_input = try ctx.hostCopy(.f32, &pad_input_dims, 0, std.mem.asBytes(&pad_input_values));
    defer pad_input.deinit();
    const pad_value_buffer = try ctx.hostCopy(.f32, &.{}, 0, std.mem.asBytes(&pad_value));
    defer pad_value_buffer.deinit();
    const padded = try Buffer.initPad(std.testing.allocator, pad_input, pad_value_buffer, &.{1}, &.{2}, &.{0}, &pad_output_dims, 0);
    defer padded.deinit();
    try std.testing.expect(padded.backend_buffer != null);
    try std.testing.expectEqual(@as(usize, 0), padded.bytes.len);
    try expectBufferF32(padded, &.{ 0.0, 2.0, 3.0, 0.0, 0.0 });

    const reverse_dims = [_]i64{ 2, 3 };
    const reverse_values = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const reverse_source = try ctx.hostCopy(.f32, &reverse_dims, 0, std.mem.asBytes(&reverse_values));
    defer reverse_source.deinit();
    const reversed = try Buffer.initReverse(std.testing.allocator, reverse_source, &.{1}, &reverse_dims, 0);
    defer reversed.deinit();
    try std.testing.expect(reversed.backend_buffer != null);
    try std.testing.expectEqual(@as(usize, 0), reversed.bytes.len);
    try expectBufferF32(reversed, &.{ 3.0, 2.0, 1.0, 6.0, 5.0, 4.0 });

    const gather_operand_dims = [_]i64{ 3, 2 };
    const gather_output_dims = [_]i64{ 2, 2 };
    const gather_operand_values = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const gather_indices_values = [_]i32{ 2, 0 };
    const gather_operand = try ctx.hostCopy(.f32, &gather_operand_dims, 0, std.mem.asBytes(&gather_operand_values));
    defer gather_operand.deinit();
    const gather_indices = try ctx.hostCopy(.s32, &.{2}, 0, std.mem.asBytes(&gather_indices_values));
    defer gather_indices.deinit();
    const gathered = try Buffer.initGather(
        std.testing.allocator,
        gather_operand,
        gather_indices,
        &.{1},
        &.{0},
        &.{},
        &.{},
        &.{0},
        1,
        &.{ 1, 2 },
        &gather_output_dims,
        0,
    );
    defer gathered.deinit();
    try std.testing.expect(gathered.backend_buffer != null);
    try std.testing.expectEqual(@as(usize, 0), gathered.bytes.len);
    try expectBufferF32(gathered, &.{ 5.0, 6.0, 1.0, 2.0 });

    const sort_dims = [_]i64{ 2, 3 };
    const sort_values = [_]f32{ 3.0, 1.0, 2.0, 6.0, 4.0, 5.0 };
    const sort_source = try ctx.hostCopy(.f32, &sort_dims, 0, std.mem.asBytes(&sort_values));
    defer sort_source.deinit();
    const sorted = try Buffer.initSort(std.testing.allocator, sort_source, 1, &sort_dims, .lt, 0);
    defer sorted.deinit();
    try std.testing.expect(sorted.backend_buffer != null);
    try std.testing.expectEqual(@as(usize, 0), sorted.bytes.len);
    try expectBufferF32(sorted, &.{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });

    const reverse_sorted = try Buffer.initSort(std.testing.allocator, sort_source, 1, &sort_dims, .gt, 0);
    defer reverse_sorted.deinit();
    try std.testing.expect(reverse_sorted.backend_buffer != null);
    try std.testing.expectEqual(@as(usize, 0), reverse_sorted.bytes.len);
    try expectBufferF32(reverse_sorted, &.{ 3.0, 2.0, 1.0, 6.0, 5.0, 4.0 });
}

test "buffer elementwise unary math supports f32 execution" {
    var ctx = try BufferTestContext.init(std.testing.allocator);
    defer ctx.deinit();

    const dims = [_]i64{2};
    const values = [_]f32{ 1.0, 4.0 };
    const input = try ctx.hostCopy(.f32, &dims, 0, std.mem.asBytes(&values));
    defer input.deinit();

    const exp = try Buffer.initElementwiseUnary(std.testing.allocator, .exp, input, 0);
    defer exp.deinit();
    try expectBufferF32(exp, &.{ std.math.exp(@as(f32, 1.0)), std.math.exp(@as(f32, 4.0)) });

    const tanh = try Buffer.initElementwiseUnary(std.testing.allocator, .tanh, input, 0);
    defer tanh.deinit();
    try expectBufferF32(tanh, &.{ std.math.tanh(@as(f32, 1.0)), std.math.tanh(@as(f32, 4.0)) });

    const sqrt = try Buffer.initElementwiseUnary(std.testing.allocator, .sqrt, input, 0);
    defer sqrt.deinit();
    try expectBufferF32(sqrt, &.{ 1.0, 2.0 });

    const rsqrt = try Buffer.initElementwiseUnary(std.testing.allocator, .rsqrt, input, 0);
    defer rsqrt.deinit();
    try expectBufferF32(rsqrt, &.{ 1.0, 0.5 });
}

test "buffer reshape preserves typed bytes and updates dimensions" {
    var ctx = try BufferTestContext.init(std.testing.allocator);
    defer ctx.deinit();

    const dims = [_]i64{ 2, 2 };
    const values = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const input = try ctx.hostCopy(.f32, &dims, 0, std.mem.asBytes(&values));
    defer input.deinit();

    const reshaped = try Buffer.initReshape(std.testing.allocator, input, &.{4}, 0);
    defer reshaped.deinit();

    try std.testing.expectEqualSlices(i64, &.{4}, reshaped.dims);
    try expectBufferF32(reshaped, &values);
}

test "buffer transpose permutes resident buffer values and updates dimensions" {
    var ctx = try BufferTestContext.init(std.testing.allocator);
    defer ctx.deinit();

    const dims = [_]i64{ 2, 3 };
    const values = [_]u8{ 1, 2, 3, 4, 5, 6 };
    const input = try ctx.hostCopy(.u8, &dims, 0, &values);
    defer input.deinit();

    const transposed = try Buffer.initTranspose(std.testing.allocator, input, &.{ 1, 0 }, &.{ 3, 2 }, 0);
    defer transposed.deinit();

    try std.testing.expectEqualSlices(i64, &.{ 3, 2 }, transposed.dims);
    try expectBufferBytes(transposed, &.{ 1, 4, 2, 5, 3, 6 });
}

test "buffer broadcast_in_dim expands resident buffer values and updates dimensions" {
    var ctx = try BufferTestContext.init(std.testing.allocator);
    defer ctx.deinit();

    const dims = [_]i64{3};
    const values = [_]u8{ 7, 8, 9 };
    const input = try ctx.hostCopy(.u8, &dims, 0, &values);
    defer input.deinit();

    const broadcasted = try Buffer.initBroadcastInDim(std.testing.allocator, input, &.{1}, &.{ 2, 3 }, 0);
    defer broadcasted.deinit();

    try std.testing.expectEqualSlices(i64, &.{ 2, 3 }, broadcasted.dims);
    try expectBufferBytes(broadcasted, &.{ 7, 8, 9, 7, 8, 9 });
}

test "buffer slice copies strided resident buffer values and updates dimensions" {
    var ctx = try BufferTestContext.init(std.testing.allocator);
    defer ctx.deinit();

    const dims = [_]i64{ 3, 4 };
    const values = [_]u8{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const input = try ctx.hostCopy(.u8, &dims, 0, &values);
    defer input.deinit();

    const sliced = try Buffer.initSlice(
        std.testing.allocator,
        input,
        &.{ 1, 0 },
        &.{ 3, 4 },
        &.{ 1, 2 },
        &.{ 2, 2 },
        0,
    );
    defer sliced.deinit();

    try std.testing.expectEqualSlices(i64, &.{ 2, 2 }, sliced.dims);
    try expectBufferBytes(sliced, &.{ 5, 7, 9, 11 });
}

test "buffer concatenate joins resident buffer values along an axis" {
    var ctx = try BufferTestContext.init(std.testing.allocator);
    defer ctx.deinit();

    const lhs_dims = [_]i64{ 2, 2 };
    const rhs_dims = [_]i64{ 2, 3 };
    const lhs_values = [_]u8{ 1, 2, 3, 4 };
    const rhs_values = [_]u8{ 5, 6, 7, 8, 9, 10 };
    const lhs = try ctx.hostCopy(.u8, &lhs_dims, 0, &lhs_values);
    defer lhs.deinit();
    const rhs = try ctx.hostCopy(.u8, &rhs_dims, 0, &rhs_values);
    defer rhs.deinit();

    const concatenated = try Buffer.initConcatenate(std.testing.allocator, lhs, rhs, 1, &.{ 2, 5 }, 0);
    defer concatenated.deinit();

    try std.testing.expectEqualSlices(i64, &.{ 2, 5 }, concatenated.dims);
    try expectBufferBytes(concatenated, &.{ 1, 2, 5, 6, 7, 3, 4, 8, 9, 10 });
}

test "buffer partition_id materializes scalar device partition" {
    var ctx = try BufferTestContext.init(std.testing.allocator);
    defer ctx.deinit();

    const partition = try Buffer.initPartitionId(std.testing.allocator, ctx.backend, .u32, &.{}, ctx.device, ctx.memory, 0, 0);
    defer partition.deinit();

    try std.testing.expectEqual(@as(i32, 0), partition.device_id);
    var bytes: [4]u8 = undefined;
    try partition.copyToHost(&bytes);
    try std.testing.expectEqual(@as(u32, 0), readU32LE(&bytes, 0));
}

test "buffer cholesky lowers through backend-native MLX buffers" {
    var ctx = try BufferTestContext.init(std.testing.allocator);
    defer ctx.deinit();

    const dims = [_]i64{ 2, 2 };
    const values = [_]f32{ 4.0, 2.0, 2.0, 3.0 };
    const input = try ctx.hostCopy(.f32, &dims, 0, std.mem.asBytes(&values));
    defer input.deinit();

    const factor = try Buffer.initCholesky(std.testing.allocator, input, true, &dims, 0);
    defer factor.deinit();

    try expectBufferF32(factor, &.{ 2.0, 0.0, 1.0, 1.4142135 });
}

test "buffer rng requires backend-native lowering on MLX buffers" {
    var ctx = try BufferTestContext.init(std.testing.allocator);
    defer ctx.deinit();

    const scalar = [_]i64{};
    const min_value = [_]f32{0.0};
    const max_value = [_]f32{1.0};
    const min = try ctx.hostCopy(.f32, &scalar, 0, std.mem.asBytes(&min_value));
    defer min.deinit();
    const max = try ctx.hostCopy(.f32, &scalar, 0, std.mem.asBytes(&max_value));
    defer max.deinit();

    try std.testing.expectError(error.UnsupportedRuntimeFeature, Buffer.initRngUniform(std.testing.allocator, min, max, .f32, &.{4}, 0));

    const state_words = [_]u32{ 1, 2 };
    const state = try ctx.hostCopy(.u32, &.{2}, 0, std.mem.asBytes(&state_words));
    defer state.deinit();
    try std.testing.expectError(error.UnsupportedRuntimeFeature, Buffer.initRngBits(std.testing.allocator, state, .u32, &.{4}, 0));
    try std.testing.expectError(error.UnsupportedRuntimeFeature, Buffer.initRngStateUpdate(std.testing.allocator, state, 0));
}
