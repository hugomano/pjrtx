const std = @import("std");
const core = @import("src/core");

pub const BufferHandle = *anyopaque;
pub const ExecutableHandle = *anyopaque;

pub const Error = error{
    InvalidDeviceCount,
    UnsupportedElementType,
    ShapeMismatch,
    BufferAllocationFailed,
    CommandSubmissionFailed,
    BufferCopyFailed,
    OutOfMemory,
};

pub const ExecutableOutput = struct {
    handle: BufferHandle,
    element_type: core.BufferType,
    dims: []const i64,
    byte_size: usize,
};

pub const ProgramNodeKind = enum {
    constant,
    parameter,
    view,
    elementwise,
    reduction,
    matmul,
    library_call,
    materialize,
};

pub const ProgramNode = struct {
    instruction_index: usize,
    kind: ProgramNodeKind,
    inputs: []const core.ValueId,
    outputs: []const core.ValueId,
    materializes: bool = true,
};

pub const Program = struct {
    allocator: std.mem.Allocator,
    nodes: []ProgramNode,
    last_uses: []usize,
    output_values: []bool,

    pub fn deinit(self: *Program) void {
        for (self.nodes) |node| {
            self.allocator.free(node.inputs);
            self.allocator.free(node.outputs);
        }
        self.allocator.free(self.output_values);
        self.allocator.free(self.last_uses);
        self.allocator.free(self.nodes);
        self.* = undefined;
    }
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
        iota: *const fn (backend: Backend, device_local_hardware_id: i32, element_type: core.BufferType, dims: []const i64, iota_dimension: i64) Error!?BufferHandle,
        cloneBuffer: *const fn (backend: Backend, src: BufferHandle) Error!?BufferHandle,
        convert: *const fn (backend: Backend, src: BufferHandle, output_type: core.BufferType) Error!?BufferHandle,
        binary: *const fn (backend: Backend, lhs: BufferHandle, rhs: BufferHandle, op: core.ElementwiseBinaryOp) Error!?BufferHandle,
        unary: *const fn (backend: Backend, src: BufferHandle, op: core.ElementwiseUnaryOp) Error!?BufferHandle,
        reshape: *const fn (backend: Backend, src: BufferHandle, dims: []const i64) Error!?BufferHandle,
        transpose: *const fn (backend: Backend, src: BufferHandle, permutation: []const i64) Error!?BufferHandle,
        broadcastInDim: *const fn (backend: Backend, src: BufferHandle, broadcast_dimensions: []const i64, output_dims: []const i64) Error!?BufferHandle,
        slice: *const fn (backend: Backend, src: BufferHandle, start_indices: []const i64, limit_indices: []const i64, strides: []const i64, output_dims: []const i64) Error!?BufferHandle,
        dynamicSlice: *const fn (backend: Backend, src: BufferHandle, start_buffers: []const BufferHandle, slice_sizes: []const i64, output_dims: []const i64) Error!?BufferHandle,
        dynamicUpdateSlice: *const fn (backend: Backend, src: BufferHandle, update: BufferHandle, start_buffers: []const BufferHandle, output_dims: []const i64) Error!?BufferHandle,
        pad: *const fn (backend: Backend, src: BufferHandle, padding_value: BufferHandle, edge_padding_low: []const i64, edge_padding_high: []const i64, interior_padding: []const i64, output_dims: []const i64) Error!?BufferHandle,
        reverse: *const fn (backend: Backend, src: BufferHandle, dimensions: []const i64, output_dims: []const i64) Error!?BufferHandle,
        concatenate: *const fn (backend: Backend, lhs: BufferHandle, rhs: BufferHandle, dimension: i64, output_dims: []const i64) Error!?BufferHandle,
        gather: *const fn (backend: Backend, operand: BufferHandle, indices: BufferHandle, start_index_map: []const i64, collapsed_slice_dims: []const i64, operand_batching_dims: []const i64, start_indices_batching_dims: []const i64, index_vector_dim: i64, slice_sizes: []const i64, offset_dims: []const i64, output_dims: []const i64) Error!?BufferHandle,
        gatherAxis: *const fn (backend: Backend, operand: BufferHandle, indices: BufferHandle, axis: i64, index_vector_dim: i64, output_dims: []const i64) Error!?BufferHandle,
        scatter: *const fn (backend: Backend, operand: BufferHandle, indices: BufferHandle, updates: BufferHandle, scatter_dims_to_operand_dims: []const i64, inserted_window_dims: []const i64, update_window_dims: []const i64, input_batching_dims: []const i64, scatter_indices_batching_dims: []const i64, index_vector_dim: i64, update_kind: core.ScatterUpdateKind, output_dims: []const i64) Error!?BufferHandle,
        scatterAxis: *const fn (backend: Backend, operand: BufferHandle, indices: BufferHandle, updates: BufferHandle, axis: i64, index_vector_dim: i64, update_kind: core.ScatterUpdateKind, output_dims: []const i64) Error!?BufferHandle,
        sort: *const fn (backend: Backend, src: BufferHandle, dimension: i64, output_dims: []const i64) Error!?BufferHandle,
        argsort: *const fn (backend: Backend, src: BufferHandle, dimension: i64, output_type: core.BufferType, output_dims: []const i64) Error!?BufferHandle,
        takeAlongAxis: *const fn (backend: Backend, src: BufferHandle, indices: BufferHandle, dimension: i64, output_dims: []const i64) Error!?BufferHandle,
        dotGeneral: *const fn (backend: Backend, lhs: BufferHandle, rhs: BufferHandle, lhs_batch_dimensions: []const i64, rhs_batch_dimensions: []const i64, lhs_contracting_dimensions: []const i64, rhs_contracting_dimensions: []const i64, output_dims: []const i64) Error!?BufferHandle,
        reduce: *const fn (backend: Backend, src: BufferHandle, op: core.PlanInstructionKind, dimensions: []const i64, output_dims: []const i64) Error!?BufferHandle,
        compare: *const fn (backend: Backend, lhs: BufferHandle, rhs: BufferHandle, direction: core.CompareOp, output_dims: []const i64) Error!?BufferHandle,
        select: *const fn (backend: Backend, pred: BufferHandle, on_true: BufferHandle, on_false: BufferHandle, output_dims: []const i64) Error!?BufferHandle,
        clamp: *const fn (backend: Backend, min: BufferHandle, value: BufferHandle, max: BufferHandle, output_dims: []const i64) Error!?BufferHandle,
        compileExecutable: *const fn (backend: Backend, allocator: std.mem.Allocator, plan: *const core.ExecutablePlan, device_local_hardware_ids: []const i32) Error!?ExecutableHandle,
        writeExecutableLoweringDiagnostic: *const fn (backend: Backend, plan: *const core.ExecutablePlan, device_local_hardware_ids: []const i32, writer: *std.Io.Writer) std.Io.Writer.Error!void,
        executeExecutable: *const fn (backend: Backend, allocator: std.mem.Allocator, executable: ExecutableHandle, device_index: usize, arguments: []const BufferHandle) Error!?[]ExecutableOutput,
        destroyExecutable: *const fn (backend: Backend, executable: ExecutableHandle) void,
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

    pub fn iota(self: Backend, device_local_hardware_id: i32, element_type: core.BufferType, dims: []const i64, iota_dimension: i64) Error!?BufferHandle {
        return self.vtable.iota(self, device_local_hardware_id, element_type, dims, iota_dimension);
    }

    pub fn cloneBuffer(self: Backend, src: BufferHandle) Error!?BufferHandle {
        return self.vtable.cloneBuffer(self, src);
    }

    pub fn convert(self: Backend, src: BufferHandle, output_type: core.BufferType) Error!?BufferHandle {
        return self.vtable.convert(self, src, output_type);
    }

    pub fn binary(self: Backend, lhs: BufferHandle, rhs: BufferHandle, op: core.ElementwiseBinaryOp) Error!?BufferHandle {
        return self.vtable.binary(self, lhs, rhs, op);
    }

    pub fn unary(self: Backend, src: BufferHandle, op: core.ElementwiseUnaryOp) Error!?BufferHandle {
        return self.vtable.unary(self, src, op);
    }

    pub fn reshape(self: Backend, src: BufferHandle, dims: []const i64) Error!?BufferHandle {
        return self.vtable.reshape(self, src, dims);
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

    pub fn dynamicSlice(self: Backend, src: BufferHandle, start_buffers: []const BufferHandle, slice_sizes: []const i64, output_dims: []const i64) Error!?BufferHandle {
        return self.vtable.dynamicSlice(self, src, start_buffers, slice_sizes, output_dims);
    }

    pub fn dynamicUpdateSlice(self: Backend, src: BufferHandle, update: BufferHandle, start_buffers: []const BufferHandle, output_dims: []const i64) Error!?BufferHandle {
        return self.vtable.dynamicUpdateSlice(self, src, update, start_buffers, output_dims);
    }

    pub fn pad(self: Backend, src: BufferHandle, padding_value: BufferHandle, edge_padding_low: []const i64, edge_padding_high: []const i64, interior_padding: []const i64, output_dims: []const i64) Error!?BufferHandle {
        return self.vtable.pad(self, src, padding_value, edge_padding_low, edge_padding_high, interior_padding, output_dims);
    }

    pub fn reverse(self: Backend, src: BufferHandle, dimensions: []const i64, output_dims: []const i64) Error!?BufferHandle {
        return self.vtable.reverse(self, src, dimensions, output_dims);
    }

    pub fn concatenate(self: Backend, lhs: BufferHandle, rhs: BufferHandle, dimension: i64, output_dims: []const i64) Error!?BufferHandle {
        return self.vtable.concatenate(self, lhs, rhs, dimension, output_dims);
    }

    pub fn gather(self: Backend, operand: BufferHandle, indices: BufferHandle, start_index_map: []const i64, collapsed_slice_dims: []const i64, operand_batching_dims: []const i64, start_indices_batching_dims: []const i64, index_vector_dim: i64, slice_sizes: []const i64, offset_dims: []const i64, output_dims: []const i64) Error!?BufferHandle {
        return self.vtable.gather(self, operand, indices, start_index_map, collapsed_slice_dims, operand_batching_dims, start_indices_batching_dims, index_vector_dim, slice_sizes, offset_dims, output_dims);
    }

    pub fn gatherAxis(self: Backend, operand: BufferHandle, indices: BufferHandle, axis: i64, index_vector_dim: i64, output_dims: []const i64) Error!?BufferHandle {
        return self.vtable.gatherAxis(self, operand, indices, axis, index_vector_dim, output_dims);
    }

    pub fn scatter(self: Backend, operand: BufferHandle, indices: BufferHandle, updates: BufferHandle, scatter_dims_to_operand_dims: []const i64, inserted_window_dims: []const i64, update_window_dims: []const i64, input_batching_dims: []const i64, scatter_indices_batching_dims: []const i64, index_vector_dim: i64, update_kind: core.ScatterUpdateKind, output_dims: []const i64) Error!?BufferHandle {
        return self.vtable.scatter(self, operand, indices, updates, scatter_dims_to_operand_dims, inserted_window_dims, update_window_dims, input_batching_dims, scatter_indices_batching_dims, index_vector_dim, update_kind, output_dims);
    }

    pub fn scatterAxis(self: Backend, operand: BufferHandle, indices: BufferHandle, updates: BufferHandle, axis: i64, index_vector_dim: i64, update_kind: core.ScatterUpdateKind, output_dims: []const i64) Error!?BufferHandle {
        return self.vtable.scatterAxis(self, operand, indices, updates, axis, index_vector_dim, update_kind, output_dims);
    }

    pub fn sort(self: Backend, src: BufferHandle, dimension: i64, output_dims: []const i64) Error!?BufferHandle {
        return self.vtable.sort(self, src, dimension, output_dims);
    }

    pub fn argsort(self: Backend, src: BufferHandle, dimension: i64, output_type: core.BufferType, output_dims: []const i64) Error!?BufferHandle {
        return self.vtable.argsort(self, src, dimension, output_type, output_dims);
    }

    pub fn takeAlongAxis(self: Backend, src: BufferHandle, indices: BufferHandle, dimension: i64, output_dims: []const i64) Error!?BufferHandle {
        return self.vtable.takeAlongAxis(self, src, indices, dimension, output_dims);
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

    pub fn clamp(self: Backend, min: BufferHandle, value: BufferHandle, max: BufferHandle, output_dims: []const i64) Error!?BufferHandle {
        return self.vtable.clamp(self, min, value, max, output_dims);
    }

    pub fn compileExecutable(self: Backend, allocator: std.mem.Allocator, plan: *const core.ExecutablePlan, device_local_hardware_ids: []const i32) Error!?ExecutableHandle {
        return self.vtable.compileExecutable(self, allocator, plan, device_local_hardware_ids);
    }

    pub fn writeExecutableLoweringDiagnostic(self: Backend, plan: *const core.ExecutablePlan, device_local_hardware_ids: []const i32, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return self.vtable.writeExecutableLoweringDiagnostic(self, plan, device_local_hardware_ids, writer);
    }

    pub fn executeExecutable(self: Backend, allocator: std.mem.Allocator, executable: ExecutableHandle, device_index: usize, arguments: []const BufferHandle) Error!?[]ExecutableOutput {
        return self.vtable.executeExecutable(self, allocator, executable, device_index, arguments);
    }

    pub fn destroyExecutable(self: Backend, executable: ExecutableHandle) void {
        self.vtable.destroyExecutable(self, executable);
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
        .kind = .metal_mlx,
        .name = "test",
        .supports_device_buffers = false,
        .supports_unified_memory = true,
    };
    try std.testing.expectEqual(core.BackendKind.metal_mlx, caps.kind);
    try std.testing.expect(!caps.supports_device_buffers);
}
