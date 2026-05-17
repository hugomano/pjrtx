const std = @import("std");
const core = @import("src/core");

pub const BufferHandle = *anyopaque;

pub const Error = error{
    InvalidDeviceCount,
    UnsupportedElementType,
    ShapeMismatch,
    BufferAllocationFailed,
    CommandSubmissionFailed,
    BufferCopyFailed,
    OutOfMemory,
};

pub const Capabilities = struct {
    kind: core.BackendKind,
    name: []const u8,
    supports_device_buffers: bool,
    supports_unified_memory: bool,
    supports_async_execution: bool = false,
};

pub const Backend = struct {
    ptr: ?*anyopaque = null,
    vtable: *const VTable,

    pub const VTable = struct {
        kind: *const fn (backend: Backend) core.BackendKind,
        capabilities: *const fn (backend: Backend) Capabilities,
        enumerateDevices: *const fn (backend: Backend, allocator: std.mem.Allocator, device_count_hint: usize) Error![]core.DeviceDescriptor,
        releaseDeviceDescriptors: *const fn (backend: Backend, allocator: std.mem.Allocator, descriptors: []core.DeviceDescriptor) void,
        bufferFromHost: *const fn (backend: Backend, device_local_hardware_id: i32, element_type: core.BufferType, dims: []const i64, src: []const u8) Error!?BufferHandle,
        cloneBuffer: *const fn (backend: Backend, src: BufferHandle) Error!?BufferHandle,
        binary: *const fn (backend: Backend, lhs: BufferHandle, rhs: BufferHandle, op: core.ElementwiseBinaryOp) Error!?BufferHandle,
        unary: *const fn (backend: Backend, src: BufferHandle, op: core.ElementwiseUnaryOp) Error!?BufferHandle,
        reshape: *const fn (backend: Backend, device_local_hardware_id: i32, element_type: core.BufferType, src_bytes: []const u8, dims: []const i64) Error!?BufferHandle,
        transpose: *const fn (backend: Backend, src: BufferHandle, permutation: []const i64) Error!?BufferHandle,
        broadcastInDim: *const fn (backend: Backend, src: BufferHandle, broadcast_dimensions: []const i64, output_dims: []const i64) Error!?BufferHandle,
        slice: *const fn (backend: Backend, src: BufferHandle, start_indices: []const i64, limit_indices: []const i64, strides: []const i64, output_dims: []const i64) Error!?BufferHandle,
        concatenate: *const fn (backend: Backend, lhs: BufferHandle, rhs: BufferHandle, dimension: i64, output_dims: []const i64) Error!?BufferHandle,
        dotGeneral: *const fn (backend: Backend, lhs: BufferHandle, rhs: BufferHandle, lhs_batch_dimensions: []const i64, rhs_batch_dimensions: []const i64, lhs_contracting_dimensions: []const i64, rhs_contracting_dimensions: []const i64, output_dims: []const i64) Error!?BufferHandle,
        reduce: *const fn (backend: Backend, src: BufferHandle, op: core.PlanInstructionKind, dimensions: []const i64, output_dims: []const i64) Error!?BufferHandle,
        compare: *const fn (backend: Backend, lhs: BufferHandle, rhs: BufferHandle, direction: core.CompareOp, output_dims: []const i64) Error!?BufferHandle,
        select: *const fn (backend: Backend, pred: BufferHandle, on_true: BufferHandle, on_false: BufferHandle, output_dims: []const i64) Error!?BufferHandle,
        copyToHost: *const fn (backend: Backend, src: BufferHandle, dst: []u8) Error!void,
        destroyBuffer: *const fn (backend: Backend, buffer: BufferHandle) void,
    };

    pub fn kind(self: Backend) core.BackendKind {
        return self.vtable.kind(self);
    }

    pub fn capabilities(self: Backend) Capabilities {
        return self.vtable.capabilities(self);
    }

    pub fn enumerateDevices(self: Backend, allocator: std.mem.Allocator, device_count_hint: usize) Error![]core.DeviceDescriptor {
        return self.vtable.enumerateDevices(self, allocator, device_count_hint);
    }

    pub fn releaseDeviceDescriptors(self: Backend, allocator: std.mem.Allocator, descriptors: []core.DeviceDescriptor) void {
        self.vtable.releaseDeviceDescriptors(self, allocator, descriptors);
    }

    pub fn bufferFromHost(self: Backend, device_local_hardware_id: i32, element_type: core.BufferType, dims: []const i64, src: []const u8) Error!?BufferHandle {
        return self.vtable.bufferFromHost(self, device_local_hardware_id, element_type, dims, src);
    }

    pub fn cloneBuffer(self: Backend, src: BufferHandle) Error!?BufferHandle {
        return self.vtable.cloneBuffer(self, src);
    }

    pub fn binary(self: Backend, lhs: BufferHandle, rhs: BufferHandle, op: core.ElementwiseBinaryOp) Error!?BufferHandle {
        return self.vtable.binary(self, lhs, rhs, op);
    }

    pub fn unary(self: Backend, src: BufferHandle, op: core.ElementwiseUnaryOp) Error!?BufferHandle {
        return self.vtable.unary(self, src, op);
    }

    pub fn reshape(self: Backend, device_local_hardware_id: i32, element_type: core.BufferType, src_bytes: []const u8, dims: []const i64) Error!?BufferHandle {
        return self.vtable.reshape(self, device_local_hardware_id, element_type, src_bytes, dims);
    }

    pub fn transpose(self: Backend, src: BufferHandle, permutation: []const i64) Error!?BufferHandle {
        return self.vtable.transpose(self, src, permutation);
    }

    pub fn broadcastInDim(self: Backend, src: BufferHandle, broadcast_dimensions: []const i64, output_dims: []const i64) Error!?BufferHandle {
        return self.vtable.broadcastInDim(self, src, broadcast_dimensions, output_dims);
    }

    pub fn slice(self: Backend, src: BufferHandle, start_indices: []const i64, limit_indices: []const i64, strides: []const i64, output_dims: []const i64) Error!?BufferHandle {
        return self.vtable.slice(self, src, start_indices, limit_indices, strides, output_dims);
    }

    pub fn concatenate(self: Backend, lhs: BufferHandle, rhs: BufferHandle, dimension: i64, output_dims: []const i64) Error!?BufferHandle {
        return self.vtable.concatenate(self, lhs, rhs, dimension, output_dims);
    }

    pub fn dotGeneral(self: Backend, lhs: BufferHandle, rhs: BufferHandle, lhs_batch_dimensions: []const i64, rhs_batch_dimensions: []const i64, lhs_contracting_dimensions: []const i64, rhs_contracting_dimensions: []const i64, output_dims: []const i64) Error!?BufferHandle {
        return self.vtable.dotGeneral(self, lhs, rhs, lhs_batch_dimensions, rhs_batch_dimensions, lhs_contracting_dimensions, rhs_contracting_dimensions, output_dims);
    }

    pub fn reduce(self: Backend, src: BufferHandle, op: core.PlanInstructionKind, dimensions: []const i64, output_dims: []const i64) Error!?BufferHandle {
        return self.vtable.reduce(self, src, op, dimensions, output_dims);
    }

    pub fn compare(self: Backend, lhs: BufferHandle, rhs: BufferHandle, direction: core.CompareOp, output_dims: []const i64) Error!?BufferHandle {
        return self.vtable.compare(self, lhs, rhs, direction, output_dims);
    }

    pub fn select(self: Backend, pred: BufferHandle, on_true: BufferHandle, on_false: BufferHandle, output_dims: []const i64) Error!?BufferHandle {
        return self.vtable.select(self, pred, on_true, on_false, output_dims);
    }

    pub fn copyToHost(self: Backend, src: BufferHandle, dst: []u8) Error!void {
        return self.vtable.copyToHost(self, src, dst);
    }

    pub fn destroyBuffer(self: Backend, buffer: BufferHandle) void {
        self.vtable.destroyBuffer(self, buffer);
    }
};

test "backend capability model is backend-neutral" {
    const caps: Capabilities = .{
        .kind = .synthetic,
        .name = "test",
        .supports_device_buffers = false,
        .supports_unified_memory = true,
    };
    try std.testing.expectEqual(core.BackendKind.synthetic, caps.kind);
    try std.testing.expect(!caps.supports_device_buffers);
}
