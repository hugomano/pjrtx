const std = @import("std");
const backend = @This();
const ir = @import("src/compiler/ir");
const program_mod = @import("program.zig");
const device_mod = @import("device.zig");
const buffer_mod = @import("buffer.zig");
const async_transfer_mod = @import("async_transfer.zig");
const mlx_call = @import("mlx_call.zig");
const custom_call_mod = @import("custom_call.zig");
const profiling_mod = @import("profiling.zig");
const executable_mod = @import("executable.zig");

pub const BufferHandle = *anyopaque;
pub const ExecutableHandle = *anyopaque;
pub const ExecutionEventHandle = *anyopaque;
pub const AsyncHostToDeviceTransferHandle = *anyopaque;

pub const ReduceWindowMaxWithIndicesResult = struct {
    values: BufferHandle,
    indices: BufferHandle,
};

pub const ReduceMaxWithIndicesResult = struct {
    values: BufferHandle,
    indices: BufferHandle,
};

pub const RngBitGeneratorResult = struct {
    state: BufferHandle,
    bits: BufferHandle,
};

pub const Error = error{
    InvalidDeviceCount,
    InvalidProgram,
    UnsupportedElementType,
    ShapeMismatch,
    BufferAllocationFailed,
    CommandSubmissionFailed,
    BufferCopyFailed,
    OutOfMemory,
    InvalidCustomCall,
};

/// Public custom-call registration kind accepted by the MLX/Metal backend.
pub const CustomCallKind = custom_call_mod.Kind;

/// Public custom-call registration record copied into the backend registry.
pub const CustomCallRegistration = custom_call_mod.Registration;

pub const ExecutableOutput = struct {
    handle: BufferHandle,
    element_type: ir.BufferType,
    dims: []const i64,
    byte_size: usize,
};

pub const ExecutionCompletionKind = enum {
    completed,
    pending,
};

pub const ExecutionCompletion = struct {
    kind: ExecutionCompletionKind = .completed,
    backend_event: ?ExecutionEventHandle = null,

    pub fn completed() ExecutionCompletion {
        return .{ .kind = .completed };
    }

    pub fn pending(event: ExecutionEventHandle) ExecutionCompletion {
        return .{ .kind = .pending, .backend_event = event };
    }
};

pub const ExecutionEventState = enum {
    pending,
    ready,
    failed,
};

pub const ExecutionEventStatus = struct {
    state: ExecutionEventState,
    message: []const u8 = "",
};

pub const ExecutionResult = struct {
    outputs: []ExecutableOutput,
    completion: ExecutionCompletion = .{},
};

/// Runtime-visible execution counters for one compiled MLX/Metal executable.
pub const ExecutableStats = executable_mod.Stats;

/// Classifies the backend execution role of an executable-plan instruction.
pub const ProgramNodeKind = program_mod.NodeKind;
/// Captures one scheduled backend graph node and its value/subprogram relationships.
pub const ProgramNode = program_mod.Node;
/// Classifies a group of nodes that the backend may execute as one fused graph segment.
pub const FusionGroupKind = program_mod.FusionGroupKind;
/// Owns metadata for a contiguous fusion group in the backend program schedule.
pub const FusionGroup = program_mod.FusionGroup;
/// Explains why the backend must materialize a program value at a schedule boundary.
pub const MaterializationReason = program_mod.MaterializationReason;
/// Marks one value that must be materialized during backend execution.
pub const MaterializationBoundary = program_mod.MaterializationBoundary;
/// Records producer, last-use, size, and materialization metadata for one program value.
pub const ProgramValue = program_mod.Value;
/// Represents one producer-to-consumer value dependency in the backend graph.
pub const ProgramEdge = program_mod.Edge;
/// Describes the kind of scheduled backend work item.
pub const ProgramScheduleKind = program_mod.ScheduleKind;
/// Describes one scheduled backend work item.
pub const ProgramScheduleItem = program_mod.ScheduleItem;
/// Owns cloned region metadata for backend control-flow execution.
pub const ProgramSubprogram = program_mod.Subprogram;
/// Describes the supported backend control-flow family.
pub const ProgramControlFlowKind = program_mod.ControlFlowKind;
/// Owns backend while-loop metadata and state mapping.
pub const ProgramControlFlow = program_mod.ControlFlow;
/// Reports liveness planning results for a backend program.
pub const ProgramLivenessStats = program_mod.LivenessStats;
/// Owns the backend graph, schedule, regions, fusion groups, and materialization boundaries.
pub const Program = program_mod.Program;

pub const Capabilities = struct {
    name: []const u8,
    supports_device_buffers: bool,
    supports_unified_memory: bool,
    supports_async_execution: bool = false,
};

pub const Backend = struct {
    pub fn capabilities(self: Backend) Capabilities {
        return backend.capabilities(self);
    }

    pub fn enumerateDevices(self: Backend, allocator: std.mem.Allocator, device_count_hint: usize) Error![]ir.DeviceDescriptor {
        return backend.enumerateDevices(self, allocator, device_count_hint);
    }

    pub fn releaseDeviceDescriptors(self: Backend, allocator: std.mem.Allocator, descriptors: []ir.DeviceDescriptor) void {
        backend.releaseDeviceDescriptors(self, allocator, descriptors);
    }

    pub fn bufferFromHost(self: Backend, device_local_hardware_id: i32, element_type: ir.BufferType, dims: []const i64, src: []const u8) Error!?BufferHandle {
        return backend.bufferFromHost(self, device_local_hardware_id, element_type, dims, src);
    }

    pub fn beginAsyncHostToDeviceTransfer(self: Backend, device_local_hardware_id: i32, element_type: ir.BufferType, dims: []const i64, byte_size: usize) Error!?AsyncHostToDeviceTransferHandle {
        return backend.beginAsyncHostToDeviceTransfer(self, device_local_hardware_id, element_type, dims, byte_size);
    }

    pub fn writeAsyncHostToDeviceTransfer(self: Backend, transfer: AsyncHostToDeviceTransferHandle, offset: usize, src: []const u8) Error!void {
        return backend.writeAsyncHostToDeviceTransfer(self, transfer, offset, src);
    }

    pub fn finishAsyncHostToDeviceTransfer(self: Backend, transfer: AsyncHostToDeviceTransferHandle) Error!?BufferHandle {
        return backend.finishAsyncHostToDeviceTransfer(self, transfer);
    }

    pub fn destroyAsyncHostToDeviceTransfer(self: Backend, transfer: AsyncHostToDeviceTransferHandle) void {
        backend.destroyAsyncHostToDeviceTransfer(self, transfer);
    }

    pub fn allocateBuffer(self: Backend, device_local_hardware_id: i32, element_type: ir.BufferType, dims: []const i64) Error!?BufferHandle {
        return backend.allocateBuffer(self, device_local_hardware_id, element_type, dims);
    }

    pub fn iota(self: Backend, device_local_hardware_id: i32, element_type: ir.BufferType, dims: []const i64, iota_dimension: i64) Error!?BufferHandle {
        return backend.iota(self, device_local_hardware_id, element_type, dims, iota_dimension);
    }

    pub fn partitionId(self: Backend, device_local_hardware_id: i32, element_type: ir.BufferType, partition_id: u32) Error!?BufferHandle {
        return backend.partitionId(self, device_local_hardware_id, element_type, partition_id);
    }

    pub fn cloneBuffer(self: Backend, src: BufferHandle) Error!?BufferHandle {
        return backend.cloneBuffer(self, src);
    }

    pub fn complex(self: Backend, real: BufferHandle, imag: BufferHandle, output_dims: []const i64) Error!?BufferHandle {
        return backend.complex(self, real, imag, output_dims);
    }

    pub fn realPart(self: Backend, src: BufferHandle, output_dims: []const i64) Error!?BufferHandle {
        return backend.realPart(self, src, output_dims);
    }

    pub fn imagPart(self: Backend, src: BufferHandle, output_dims: []const i64) Error!?BufferHandle {
        return backend.imagPart(self, src, output_dims);
    }

    pub fn convert(self: Backend, src: BufferHandle, output_type: ir.BufferType) Error!?BufferHandle {
        return backend.convert(self, src, output_type);
    }

    pub fn bitcast(self: Backend, src: BufferHandle, output_type: ir.BufferType, output_dims: []const i64) Error!?BufferHandle {
        return backend.bitcast(self, src, output_type, output_dims);
    }

    pub fn binary(self: Backend, lhs: BufferHandle, rhs: BufferHandle, op: ir.ElementwiseBinaryOp) Error!?BufferHandle {
        return backend.binary(self, lhs, rhs, op);
    }

    pub fn unary(self: Backend, src: BufferHandle, op: ir.ElementwiseUnaryOp) Error!?BufferHandle {
        return backend.unary(self, src, op);
    }

    pub fn reshape(self: Backend, src: BufferHandle, dims: []const i64) Error!?BufferHandle {
        return backend.reshape(self, src, dims);
    }

    pub fn transpose(self: Backend, src: BufferHandle, permutation: []const i64) Error!?BufferHandle {
        return backend.transpose(self, src, permutation);
    }

    pub fn broadcastInDim(self: Backend, src: BufferHandle, broadcast_dimensions: []const i64, output_dims: []const i64) Error!?BufferHandle {
        return backend.broadcastInDim(self, src, broadcast_dimensions, output_dims);
    }

    pub fn slice(self: Backend, src: BufferHandle, start_indices: []const i64, limit_indices: []const i64, strides: []const i64, output_dims: []const i64) Error!?BufferHandle {
        return backend.slice(self, src, start_indices, limit_indices, strides, output_dims);
    }

    pub fn dynamicSlice(self: Backend, src: BufferHandle, start_buffers: []const BufferHandle, slice_sizes: []const i64, output_dims: []const i64) Error!?BufferHandle {
        return backend.dynamicSlice(self, src, start_buffers, slice_sizes, output_dims);
    }

    pub fn dynamicUpdateSlice(self: Backend, src: BufferHandle, update: BufferHandle, start_buffers: []const BufferHandle, output_dims: []const i64) Error!?BufferHandle {
        return backend.dynamicUpdateSlice(self, src, update, start_buffers, output_dims);
    }

    pub fn pad(self: Backend, src: BufferHandle, padding_value: BufferHandle, edge_padding_low: []const i64, edge_padding_high: []const i64, interior_padding: []const i64, output_dims: []const i64) Error!?BufferHandle {
        return backend.pad(self, src, padding_value, edge_padding_low, edge_padding_high, interior_padding, output_dims);
    }

    pub fn reverse(self: Backend, src: BufferHandle, dimensions: []const i64, output_dims: []const i64) Error!?BufferHandle {
        return backend.reverse(self, src, dimensions, output_dims);
    }

    pub fn concatenate(self: Backend, lhs: BufferHandle, rhs: BufferHandle, dimension: i64, output_dims: []const i64) Error!?BufferHandle {
        return backend.concatenate(self, lhs, rhs, dimension, output_dims);
    }

    pub fn gather(self: Backend, operand: BufferHandle, indices: BufferHandle, start_index_map: []const i64, collapsed_slice_dims: []const i64, operand_batching_dims: []const i64, start_indices_batching_dims: []const i64, index_vector_dim: i64, slice_sizes: []const i64, offset_dims: []const i64, output_dims: []const i64) Error!?BufferHandle {
        return backend.gather(self, operand, indices, start_index_map, collapsed_slice_dims, operand_batching_dims, start_indices_batching_dims, index_vector_dim, slice_sizes, offset_dims, output_dims);
    }

    pub fn gatherAxis(self: Backend, operand: BufferHandle, indices: BufferHandle, axis: i64, index_vector_dim: i64, output_dims: []const i64) Error!?BufferHandle {
        return backend.gatherAxis(self, operand, indices, axis, index_vector_dim, output_dims);
    }

    pub fn scatter(self: Backend, operand: BufferHandle, indices: BufferHandle, updates: BufferHandle, scatter_dims_to_operand_dims: []const i64, inserted_window_dims: []const i64, update_window_dims: []const i64, input_batching_dims: []const i64, scatter_indices_batching_dims: []const i64, index_vector_dim: i64, update_kind: ir.ScatterUpdateKind, output_dims: []const i64) Error!?BufferHandle {
        return backend.scatter(self, operand, indices, updates, scatter_dims_to_operand_dims, inserted_window_dims, update_window_dims, input_batching_dims, scatter_indices_batching_dims, index_vector_dim, update_kind, output_dims);
    }

    pub fn scatterAxis(self: Backend, operand: BufferHandle, indices: BufferHandle, updates: BufferHandle, axis: i64, index_vector_dim: i64, update_kind: ir.ScatterUpdateKind, output_dims: []const i64) Error!?BufferHandle {
        return backend.scatterAxis(self, operand, indices, updates, axis, index_vector_dim, update_kind, output_dims);
    }

    pub fn sort(self: Backend, src: BufferHandle, dimension: i64, output_dims: []const i64) Error!?BufferHandle {
        return backend.sort(self, src, dimension, output_dims);
    }

    pub fn argsort(self: Backend, src: BufferHandle, dimension: i64, output_type: ir.BufferType, output_dims: []const i64) Error!?BufferHandle {
        return backend.argsort(self, src, dimension, output_type, output_dims);
    }

    pub fn takeAlongAxis(self: Backend, src: BufferHandle, indices: BufferHandle, dimension: i64, output_dims: []const i64) Error!?BufferHandle {
        return backend.takeAlongAxis(self, src, indices, dimension, output_dims);
    }

    pub fn dotGeneral(self: Backend, lhs: BufferHandle, rhs: BufferHandle, lhs_batch_dimensions: []const i64, rhs_batch_dimensions: []const i64, lhs_contracting_dimensions: []const i64, rhs_contracting_dimensions: []const i64, output_dims: []const i64) Error!?BufferHandle {
        return backend.dotGeneral(self, lhs, rhs, lhs_batch_dimensions, rhs_batch_dimensions, lhs_contracting_dimensions, rhs_contracting_dimensions, output_dims);
    }

    pub fn convolution(self: Backend, lhs: BufferHandle, rhs: BufferHandle, window_strides: []const i64, padding_low: []const i64, padding_high: []const i64, lhs_dilation: []const i64, rhs_dilation: []const i64, window_reversal: []const bool, feature_group_count: i64, output_dims: []const i64) Error!?BufferHandle {
        return backend.convolution(self, lhs, rhs, window_strides, padding_low, padding_high, lhs_dilation, rhs_dilation, window_reversal, feature_group_count, output_dims);
    }

    pub fn cholesky(self: Backend, src: BufferHandle, lower: bool, output_dims: []const i64) Error!?BufferHandle {
        return backend.cholesky(self, src, lower, output_dims);
    }

    pub fn triangularSolve(self: Backend, a: BufferHandle, b: BufferHandle, left_side: bool, lower: bool, unit_diagonal: bool, transpose_a: ir.TriangularSolveTranspose, output_dims: []const i64) Error!?BufferHandle {
        return backend.triangularSolve(self, a, b, left_side, lower, unit_diagonal, transpose_a, output_dims);
    }

    pub fn fft(self: Backend, src: BufferHandle, fft_kind: ir.FftKind, fft_lengths: []const i64, output_dims: []const i64) Error!?BufferHandle {
        return backend.fft(self, src, fft_kind, fft_lengths, output_dims);
    }

    pub fn rngBitGenerator(self: Backend, state: BufferHandle, output_type: ir.BufferType, output_dims: []const i64) Error!?RngBitGeneratorResult {
        return backend.rngBitGenerator(self, state, output_type, output_dims);
    }

    pub fn rng(self: Backend, a: BufferHandle, b: BufferHandle, distribution: ir.RngDistribution, output_type: ir.BufferType, output_dims: []const i64) Error!?BufferHandle {
        return backend.rng(self, a, b, distribution, output_type, output_dims);
    }

    pub fn reduce(self: Backend, src: BufferHandle, op: ir.PlanInstructionKind, dimensions: []const i64, output_dims: []const i64) Error!?BufferHandle {
        return backend.reduce(self, src, op, dimensions, output_dims);
    }

    pub fn reduceMaxWithIndices(self: Backend, values: BufferHandle, indices: BufferHandle, dimensions: []const i64, output_dims: []const i64) Error!?ReduceMaxWithIndicesResult {
        return backend.reduceMaxWithIndices(self, values, indices, dimensions, output_dims);
    }

    pub fn reduceWindow(self: Backend, src: BufferHandle, op: ir.PlanInstructionKind, window_dimensions: []const i64, window_strides: []const i64, base_dilations: []const i64, window_dilations: []const i64, padding_low: []const i64, padding_high: []const i64, output_dims: []const i64) Error!?BufferHandle {
        return backend.reduceWindow(self, src, op, window_dimensions, window_strides, base_dilations, window_dilations, padding_low, padding_high, output_dims);
    }

    pub fn reduceWindowMaxWithIndices(self: Backend, values: BufferHandle, indices: BufferHandle, window_dimensions: []const i64, window_strides: []const i64, base_dilations: []const i64, window_dilations: []const i64, padding_low: []const i64, padding_high: []const i64, output_dims: []const i64) Error!?ReduceWindowMaxWithIndicesResult {
        return backend.reduceWindowMaxWithIndices(self, values, indices, window_dimensions, window_strides, base_dilations, window_dilations, padding_low, padding_high, output_dims);
    }

    pub fn compare(self: Backend, lhs: BufferHandle, rhs: BufferHandle, direction: ir.CompareOp, output_dims: []const i64) Error!?BufferHandle {
        return backend.compare(self, lhs, rhs, direction, output_dims);
    }

    pub fn select(self: Backend, pred: BufferHandle, on_true: BufferHandle, on_false: BufferHandle, output_dims: []const i64) Error!?BufferHandle {
        return backend.select(self, pred, on_true, on_false, output_dims);
    }

    pub fn clamp(self: Backend, min: BufferHandle, value: BufferHandle, max: BufferHandle, output_dims: []const i64) Error!?BufferHandle {
        return backend.clamp(self, min, value, max, output_dims);
    }

    pub fn compileExecutable(self: Backend, allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, device_local_hardware_ids: []const i32) Error!?ExecutableHandle {
        return backend.compileExecutable(self, allocator, plan, device_local_hardware_ids);
    }

    pub fn writeExecutableLoweringDiagnostic(self: Backend, plan: *const ir.ExecutablePlan, device_local_hardware_ids: []const i32, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return backend.writeExecutableLoweringDiagnostic(self, plan, device_local_hardware_ids, writer);
    }

    pub fn executeExecutable(self: Backend, allocator: std.mem.Allocator, executable: ExecutableHandle, device_index: usize, arguments: []const BufferHandle) Error!?ExecutionResult {
        return backend.executeExecutable(self, allocator, executable, device_index, arguments);
    }

    pub fn executionEventStatus(self: Backend, event: ExecutionEventHandle) Error!ExecutionEventStatus {
        return backend.executionEventStatus(self, event);
    }

    pub fn destroyExecutionEvent(self: Backend, event: ExecutionEventHandle) void {
        backend.destroyExecutionEvent(self, event);
    }

    pub fn executableStats(self: Backend, executable: ExecutableHandle) ExecutableStats {
        return backend.executableStats(self, executable);
    }

    pub fn destroyExecutable(self: Backend, executable: ExecutableHandle) void {
        backend.destroyExecutable(self, executable);
    }

    pub fn registerCustomCall(self: Backend, registration: CustomCallRegistration) Error!void {
        return backend.registerCustomCall(self, registration);
    }

    pub fn unregisterCustomCall(self: Backend, target: []const u8) void {
        backend.unregisterCustomCall(self, target);
    }

    pub fn customCallRegistryVersion(self: Backend) u64 {
        return backend.customCallRegistryVersion(self);
    }

    pub fn copyToHost(self: Backend, src: BufferHandle, dst: []u8) Error!void {
        return backend.copyToHost(self, src, dst);
    }

    pub fn destroyBuffer(self: Backend, buffer: BufferHandle) void {
        backend.destroyBuffer(self, buffer);
    }
};

pub fn create() Backend {
    return .{};
}

fn capabilities(_: backend.Backend) backend.Capabilities {
    return .{
        .name = "metal_mlx",
        .supports_device_buffers = true,
        .supports_unified_memory = true,
    };
}

fn enumerateDevices(_: backend.Backend, allocator: std.mem.Allocator, _: usize) backend.Error![]ir.DeviceDescriptor {
    return device_mod.DeviceList.enumerate(allocator);
}

fn releaseDeviceDescriptors(_: backend.Backend, allocator: std.mem.Allocator, descriptors: []ir.DeviceDescriptor) void {
    device_mod.DeviceList.release(allocator, descriptors);
}

fn bufferFromHost(_: backend.Backend, device_local_hardware_id: i32, element_type: ir.BufferType, dims: []const i64, src: []const u8) backend.Error!?backend.BufferHandle {
    return maybeBufferHandle(try buffer_mod.Buffer.fromHost(device_local_hardware_id, element_type, dims, src));
}

fn beginAsyncHostToDeviceTransfer(_: backend.Backend, device_local_hardware_id: i32, element_type: ir.BufferType, dims: []const i64, byte_size: usize) backend.Error!?backend.AsyncHostToDeviceTransferHandle {
    const transfer = try async_transfer_mod.AsyncTransfer.begin(device_local_hardware_id, element_type, dims, byte_size);
    return if (transfer) |value| value.toHandle() else null;
}

fn writeAsyncHostToDeviceTransfer(_: backend.Backend, transfer: backend.AsyncHostToDeviceTransferHandle, offset: usize, src: []const u8) backend.Error!void {
    try async_transfer_mod.AsyncTransfer.fromHandle(transfer).write(offset, src);
}

fn finishAsyncHostToDeviceTransfer(_: backend.Backend, transfer: backend.AsyncHostToDeviceTransferHandle) backend.Error!?backend.BufferHandle {
    return maybeBufferHandle(try async_transfer_mod.AsyncTransfer.fromHandle(transfer).finish());
}

fn destroyAsyncHostToDeviceTransfer(_: backend.Backend, transfer: backend.AsyncHostToDeviceTransferHandle) void {
    async_transfer_mod.AsyncTransfer.fromHandle(transfer).destroy();
}

fn allocateBuffer(_: backend.Backend, device_local_hardware_id: i32, element_type: ir.BufferType, dims: []const i64) backend.Error!?backend.BufferHandle {
    return maybeBufferHandle(try buffer_mod.Buffer.zeros(device_local_hardware_id, element_type, dims));
}

fn iota(_: backend.Backend, device_local_hardware_id: i32, element_type: ir.BufferType, dims: []const i64, iota_dimension: i64) backend.Error!?backend.BufferHandle {
    return maybeBufferHandle(try buffer_mod.Buffer.iota(device_local_hardware_id, element_type, dims, iota_dimension));
}

fn partitionId(_: backend.Backend, device_local_hardware_id: i32, element_type: ir.BufferType, partition_id: u32) backend.Error!?backend.BufferHandle {
    return maybeBufferHandle(try buffer_mod.Buffer.partitionId(device_local_hardware_id, element_type, partition_id));
}

fn cloneBuffer(_: backend.Backend, src: backend.BufferHandle) backend.Error!?backend.BufferHandle {
    return maybeBufferHandle(try bufferRef(src).clone());
}

fn zeroLike(_: backend.Backend, src: backend.BufferHandle) backend.Error!?backend.BufferHandle {
    return maybeBufferHandle(try bufferRef(src).zeroLike());
}

fn complex(_: backend.Backend, real: backend.BufferHandle, imag: backend.BufferHandle, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    return maybeBufferHandle(try buffer_mod.Buffer.complex(bufferRef(real), bufferRef(imag), output_dims));
}

fn realPart(_: backend.Backend, src: backend.BufferHandle, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    return maybeBufferHandle(try bufferRef(src).realPart(output_dims));
}

fn imagPart(_: backend.Backend, src: backend.BufferHandle, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    return maybeBufferHandle(try bufferRef(src).imagPart(output_dims));
}

fn convert(_: backend.Backend, src: backend.BufferHandle, output_type: ir.BufferType) backend.Error!?backend.BufferHandle {
    return maybeBufferHandle(try bufferRef(src).convert(output_type));
}

fn bitcast(_: backend.Backend, src: backend.BufferHandle, output_type: ir.BufferType, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    return maybeBufferHandle(try bufferRef(src).bitcast(output_type, output_dims));
}

fn binary(_: backend.Backend, lhs: backend.BufferHandle, rhs: backend.BufferHandle, op: ir.ElementwiseBinaryOp) backend.Error!?backend.BufferHandle {
    return maybeBufferHandle(try buffer_mod.Buffer.binary(bufferRef(lhs), bufferRef(rhs), op));
}

fn binaryWithOutputDims(_: backend.Backend, lhs: backend.BufferHandle, rhs: backend.BufferHandle, op: ir.ElementwiseBinaryOp, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    return maybeBufferHandle(try buffer_mod.Buffer.binaryWithOutputDims(bufferRef(lhs), bufferRef(rhs), op, output_dims));
}

fn unary(_: backend.Backend, src: backend.BufferHandle, op: ir.ElementwiseUnaryOp) backend.Error!?backend.BufferHandle {
    return maybeBufferHandle(try bufferRef(src).unary(op));
}

fn reshape(_: backend.Backend, src: backend.BufferHandle, dims: []const i64) backend.Error!?backend.BufferHandle {
    return maybeBufferHandle(try bufferRef(src).reshape(dims));
}

fn transpose(_: backend.Backend, src: backend.BufferHandle, permutation: []const i64) backend.Error!?backend.BufferHandle {
    return maybeBufferHandle(try bufferRef(src).transpose(permutation));
}

fn broadcastInDim(_: backend.Backend, src: backend.BufferHandle, broadcast_dimensions: []const i64, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    return maybeBufferHandle(try bufferRef(src).broadcastInDim(broadcast_dimensions, output_dims));
}

fn slice(_: backend.Backend, src: backend.BufferHandle, start_indices: []const i64, limit_indices: []const i64, strides: []const i64, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    return maybeBufferHandle(try bufferRef(src).slice(start_indices, limit_indices, strides, output_dims));
}

fn dynamicSlice(_: backend.Backend, src: backend.BufferHandle, start_buffers: []const backend.BufferHandle, slice_sizes: []const i64, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    return maybeBufferHandle(try bufferRef(src).dynamicSlice(bufferRefs(start_buffers), slice_sizes, output_dims));
}

fn dynamicUpdateSlice(_: backend.Backend, src: backend.BufferHandle, update: backend.BufferHandle, start_buffers: []const backend.BufferHandle, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    return maybeBufferHandle(try bufferRef(src).dynamicUpdateSlice(bufferRef(update), bufferRefs(start_buffers), output_dims));
}

fn pad(_: backend.Backend, src: backend.BufferHandle, padding_value: backend.BufferHandle, edge_padding_low: []const i64, edge_padding_high: []const i64, interior_padding: []const i64, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    return maybeBufferHandle(try bufferRef(src).pad(bufferRef(padding_value), edge_padding_low, edge_padding_high, interior_padding, output_dims));
}

fn reverse(_: backend.Backend, src: backend.BufferHandle, dimensions: []const i64, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    return maybeBufferHandle(try bufferRef(src).reverse(dimensions, output_dims));
}

fn concatenate(_: backend.Backend, lhs: backend.BufferHandle, rhs: backend.BufferHandle, dimension: i64, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    return maybeBufferHandle(try buffer_mod.Buffer.concatenate(bufferRef(lhs), bufferRef(rhs), dimension, output_dims));
}

fn gatherAxis(_: backend.Backend, operand: backend.BufferHandle, indices: backend.BufferHandle, axis: i64, index_vector_dim: i64, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    return maybeBufferHandle(try buffer_mod.Buffer.gatherAxis(bufferRef(operand), bufferRef(indices), axis, index_vector_dim, output_dims));
}

fn gather(_: backend.Backend, operand: backend.BufferHandle, indices: backend.BufferHandle, start_index_map: []const i64, collapsed_slice_dims: []const i64, operand_batching_dims: []const i64, start_indices_batching_dims: []const i64, index_vector_dim: i64, slice_sizes: []const i64, offset_dims: []const i64, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    return maybeBufferHandle(try buffer_mod.Buffer.gather(bufferRef(operand), bufferRef(indices), start_index_map, collapsed_slice_dims, operand_batching_dims, start_indices_batching_dims, index_vector_dim, slice_sizes, offset_dims, output_dims));
}

fn scatterAxis(_: backend.Backend, operand: backend.BufferHandle, indices: backend.BufferHandle, updates: backend.BufferHandle, axis: i64, index_vector_dim: i64, update_kind: ir.ScatterUpdateKind, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    return maybeBufferHandle(try buffer_mod.Buffer.scatterAxis(bufferRef(operand), bufferRef(indices), bufferRef(updates), axis, index_vector_dim, update_kind, output_dims));
}

fn scatter(_: backend.Backend, operand: backend.BufferHandle, indices: backend.BufferHandle, updates: backend.BufferHandle, scatter_dims_to_operand_dims: []const i64, inserted_window_dims: []const i64, update_window_dims: []const i64, input_batching_dims: []const i64, scatter_indices_batching_dims: []const i64, index_vector_dim: i64, update_kind: ir.ScatterUpdateKind, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    return maybeBufferHandle(try buffer_mod.Buffer.scatter(bufferRef(operand), bufferRef(indices), bufferRef(updates), scatter_dims_to_operand_dims, inserted_window_dims, update_window_dims, input_batching_dims, scatter_indices_batching_dims, index_vector_dim, update_kind, output_dims));
}

fn sort(_: backend.Backend, src: backend.BufferHandle, dimension: i64, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    return maybeBufferHandle(try bufferRef(src).sort(dimension, output_dims));
}

fn argsort(_: backend.Backend, src: backend.BufferHandle, dimension: i64, output_type: ir.BufferType, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    return maybeBufferHandle(try bufferRef(src).argsort(dimension, output_type, output_dims));
}

fn takeAlongAxis(_: backend.Backend, src: backend.BufferHandle, indices: backend.BufferHandle, dimension: i64, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    return maybeBufferHandle(try bufferRef(src).takeAlongAxis(bufferRef(indices), dimension, output_dims));
}

fn dotGeneral(_: backend.Backend, lhs: backend.BufferHandle, rhs: backend.BufferHandle, lhs_batch_dimensions: []const i64, rhs_batch_dimensions: []const i64, lhs_contracting_dimensions: []const i64, rhs_contracting_dimensions: []const i64, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    return maybeBufferHandle(try buffer_mod.Buffer.dotGeneral(bufferRef(lhs), bufferRef(rhs), lhs_batch_dimensions, rhs_batch_dimensions, lhs_contracting_dimensions, rhs_contracting_dimensions, output_dims));
}

fn convolution(_: backend.Backend, lhs: backend.BufferHandle, rhs: backend.BufferHandle, window_strides: []const i64, padding_low: []const i64, padding_high: []const i64, lhs_dilation: []const i64, rhs_dilation: []const i64, window_reversal: []const bool, feature_group_count: i64, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    return maybeBufferHandle(try buffer_mod.Buffer.convolution(bufferRef(lhs), bufferRef(rhs), window_strides, padding_low, padding_high, lhs_dilation, rhs_dilation, window_reversal, feature_group_count, output_dims));
}

fn fft(_: backend.Backend, src: backend.BufferHandle, kind_: ir.FftKind, fft_lengths: []const i64, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    return maybeBufferHandle(try bufferRef(src).fft(kind_, fft_lengths, output_dims));
}

fn rng(_: backend.Backend, a: backend.BufferHandle, b: backend.BufferHandle, distribution: ir.RngDistribution, output_type: ir.BufferType, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    return maybeBufferHandle(try buffer_mod.Buffer.rng(bufferRef(a), bufferRef(b), distribution, output_type, output_dims));
}

fn rngBitGenerator(_: backend.Backend, state: backend.BufferHandle, output_type: ir.BufferType, output_dims: []const i64) backend.Error!?backend.RngBitGeneratorResult {
    const pair = (try buffer_mod.Buffer.rngBitGenerator(bufferRef(state), output_type, output_dims)) orelse return null;
    return .{ .state = pair.first.toHandle(), .bits = pair.second.toHandle() };
}

fn cholesky(_: backend.Backend, src: backend.BufferHandle, lower: bool, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    return maybeBufferHandle(try bufferRef(src).cholesky(lower, output_dims));
}

fn triangularSolve(_: backend.Backend, a: backend.BufferHandle, b: backend.BufferHandle, left_side: bool, lower: bool, unit_diagonal: bool, transpose_a: ir.TriangularSolveTranspose, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    return maybeBufferHandle(try buffer_mod.Buffer.triangularSolve(bufferRef(a), bufferRef(b), left_side, lower, unit_diagonal, transpose_a, output_dims));
}

fn reduce(_: backend.Backend, src: backend.BufferHandle, op: ir.PlanInstructionKind, dimensions: []const i64, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    return maybeBufferHandle(try bufferRef(src).reduce(op, dimensions, output_dims));
}

fn reduceMaxWithIndices(_: backend.Backend, values: backend.BufferHandle, indices: backend.BufferHandle, dimensions: []const i64, output_dims: []const i64) backend.Error!?backend.ReduceMaxWithIndicesResult {
    const pair = (try buffer_mod.Buffer.reduceMaxWithIndices(bufferRef(values), bufferRef(indices), dimensions, output_dims)) orelse return null;
    return .{ .values = pair.first.toHandle(), .indices = pair.second.toHandle() };
}

fn reduceWindow(_: backend.Backend, src: backend.BufferHandle, op: ir.PlanInstructionKind, window_dimensions: []const i64, window_strides: []const i64, base_dilations: []const i64, window_dilations: []const i64, padding_low: []const i64, padding_high: []const i64, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    return maybeBufferHandle(try bufferRef(src).reduceWindow(op, window_dimensions, window_strides, base_dilations, window_dilations, padding_low, padding_high, output_dims));
}

fn reduceWindowMaxWithIndices(_: backend.Backend, values: backend.BufferHandle, indices: backend.BufferHandle, window_dimensions: []const i64, window_strides: []const i64, base_dilations: []const i64, window_dilations: []const i64, padding_low: []const i64, padding_high: []const i64, output_dims: []const i64) backend.Error!?backend.ReduceWindowMaxWithIndicesResult {
    const pair = (try buffer_mod.Buffer.reduceWindowMaxWithIndices(bufferRef(values), bufferRef(indices), window_dimensions, window_strides, base_dilations, window_dilations, padding_low, padding_high, output_dims)) orelse return null;
    return .{ .values = pair.first.toHandle(), .indices = pair.second.toHandle() };
}

fn compare(_: backend.Backend, lhs: backend.BufferHandle, rhs: backend.BufferHandle, direction: ir.CompareOp, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    return maybeBufferHandle(try buffer_mod.Buffer.compare(bufferRef(lhs), bufferRef(rhs), direction, output_dims));
}

fn select(_: backend.Backend, pred: backend.BufferHandle, on_true: backend.BufferHandle, on_false: backend.BufferHandle, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    return maybeBufferHandle(try buffer_mod.Buffer.select(bufferRef(pred), bufferRef(on_true), bufferRef(on_false), output_dims));
}

fn clamp(_: backend.Backend, min: backend.BufferHandle, value: backend.BufferHandle, max: backend.BufferHandle, output_dims: []const i64) backend.Error!?backend.BufferHandle {
    return maybeBufferHandle(try buffer_mod.Buffer.clamp(bufferRef(min), bufferRef(value), bufferRef(max), output_dims));
}

fn whileF32CompareAdd(_: backend.Backend, state: backend.BufferHandle, limit: backend.BufferHandle, step: backend.BufferHandle, compare_direction: ir.CompareOp, update_op: ir.ElementwiseBinaryOp, output_dims: []const i64, max_iterations: u64) backend.Error!?backend.BufferHandle {
    return maybeBufferHandle(try buffer_mod.Buffer.whileF32CompareAdd(bufferRef(state), bufferRef(limit), bufferRef(step), compare_direction, update_op, output_dims, max_iterations));
}

fn evalBuffer(buffer: backend.BufferHandle) backend.Error!void {
    try bufferRef(buffer).eval();
}

fn evalBuffers(buffers: []const backend.BufferHandle) backend.Error!void {
    try buffer_mod.evalMany(bufferRefs(buffers));
}

fn copyToHost(_: backend.Backend, src: backend.BufferHandle, dst: []u8) backend.Error!void {
    try bufferRef(src).copyToHost(dst);
}

fn destroyBuffer(_: backend.Backend, buffer: backend.BufferHandle) void {
    bufferRef(buffer).destroy();
}

fn bufferRef(handle: backend.BufferHandle) buffer_mod.Buffer {
    return buffer_mod.Buffer.fromHandle(handle);
}

fn bufferRefs(handles: []const backend.BufferHandle) []const buffer_mod.Buffer {
    return @ptrCast(handles);
}

fn maybeBufferHandle(buffer: ?buffer_mod.Buffer) ?backend.BufferHandle {
    return if (buffer) |value| value.toHandle() else null;
}

fn envFlag(comptime name: [:0]const u8) bool {
    const value = std.c.getenv(name) orelse return false;
    const text = std.mem.span(value);
    return text.len != 0 and !std.mem.eql(u8, text, "0") and !std.ascii.eqlIgnoreCase(text, "false");
}

fn profileEnabled() bool {
    return profiling_mod.enabled();
}

fn profileVerbose() bool {
    return profiling_mod.verbose();
}

fn programCompileEnabled() bool {
    const value = std.c.getenv("PJRTX_MLX_PROGRAM_COMPILE") orelse return true;
    const text = std.mem.span(value);
    return text.len == 0 or (!std.mem.eql(u8, text, "0") and !std.ascii.eqlIgnoreCase(text, "false"));
}

fn planSupportsCompiledProgram(plan: *const ir.ExecutablePlan) bool {
    for (plan.instructions) |instruction| {
        switch (instruction.kind) {
            // The RNG path is device-native, but it uses a custom Metal kernel under MLX
            // and is not currently valid inside an mlx::compile graph builder.
            .rng_bit_generator => return false,
            else => {},
        }
    }
    return true;
}

fn profileStart(enabled: bool) std.Io.Timestamp {
    return profiling_mod.start(enabled);
}

fn profileElapsedUs(start: std.Io.Timestamp) u64 {
    return profiling_mod.elapsedUs(start);
}

const ExecuteProfile = profiling_mod.Execute;

const CompiledExecutable = executable_mod.Executable;

const LoweringIssue = struct {
    instruction_index: ?usize = null,
    value_id: ?ir.ValueId = null,
    op: ?ir.PlanInstructionKind = null,
    detail: []const u8,
    feature: []const u8 = "mlx-backend-executable",
};

const DefaultWhileMaxIterations: u64 = 1_000_000;
const InitialCaptureSmallControlBytes: usize = 4096;

const WhileF32LtAddPattern = struct {
    limit: ir.RegionValue,
    step: WhilePatternOperand,
    state_index: usize = 0,
    compare_direction: ir.CompareOp = .lt,
    update_op: ir.ElementwiseBinaryOp = .add,
    state_count: usize = 1,
    max_iterations: u64 = DefaultWhileMaxIterations,
};

const WhilePatternOperand = struct {
    value: ir.RegionValue,
    producer_instruction_index: ?usize = null,
};

const WhileOperandHandle = struct {
    handle: backend.BufferHandle,
    owned: bool = false,
};

const CompiledProgramContext = executable_mod.CompiledProgramContext;
const ArgumentCaptureState = executable_mod.ArgumentCaptureState;

fn registerCustomCall(_: backend.Backend, registration: backend.CustomCallRegistration) backend.Error!void {
    return custom_call_mod.register(registration);
}

fn unregisterCustomCall(_: backend.Backend, target: []const u8) void {
    custom_call_mod.unregister(target);
}

fn customCallRegistryVersion(_: backend.Backend) u64 {
    return custom_call_mod.version();
}

fn lookupCustomCall(target: []const u8) ?custom_call_mod.Spec {
    return custom_call_mod.lookup(target);
}

fn buildBackendProgram(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, diagnostic_writer: ?*std.Io.Writer) !backend.Program {
    var nodes = try allocator.alloc(backend.ProgramNode, plan.instructions.len);
    errdefer allocator.free(nodes);
    var initialized_nodes: usize = 0;
    errdefer {
        for (nodes[0..initialized_nodes]) |node| {
            allocator.free(node.inputs);
            allocator.free(node.outputs);
            if (node.subprograms.len != 0) allocator.free(node.subprograms);
        }
    }
    const subprogram_capacity = try countPlanSubprograms(plan);
    const subprograms = try allocator.alloc(backend.ProgramSubprogram, subprogram_capacity);
    var initialized_subprograms: usize = 0;
    var subprograms_owned = true;
    errdefer if (subprograms_owned) deinitProgramSubprograms(allocator, subprograms[0..initialized_subprograms]);
    const control_flow_capacity = try countPlanControlFlows(plan);
    const control_flows = try allocator.alloc(backend.ProgramControlFlow, control_flow_capacity);
    var initialized_control_flows: usize = 0;
    var control_flows_owned = true;
    errdefer if (control_flows_owned) deinitProgramControlFlows(allocator, control_flows[0..initialized_control_flows]);

    const last_uses = try allocator.alloc(?usize, plan.values.len);
    defer allocator.free(last_uses);
    @memset(last_uses, null);

    const value_producers = try allocator.alloc(?usize, plan.values.len);
    defer allocator.free(value_producers);
    @memset(value_producers, null);

    const output_values = try allocator.alloc(bool, plan.values.len);
    defer allocator.free(output_values);
    @memset(output_values, false);
    for (plan.output_ids) |output_id| {
        if (output_id.index < output_values.len) output_values[output_id.index] = true;
    }

    const max_fusion_groups = try allocator.alloc(backend.FusionGroup, plan.instructions.len);
    defer allocator.free(max_fusion_groups);

    var current_fusion_group: ?usize = null;
    var fusion_group_count: usize = 0;
    for (plan.instructions, 0..) |instruction, instruction_index| {
        for (instruction.inputs) |input_id| {
            if (input_id.index < last_uses.len) {
                last_uses[input_id.index] = instruction_index;
            }
        }
        if (instruction.kind == .get_tuple_element and instruction.inputs.len >= 1) {
            const tuple_id = instruction.inputs[0];
            if (tuple_id.index < plan.values.len) {
                const tuple_value = plan.values[tuple_id.index];
                if (tuple_value.storage == .tuple) {
                    if (instruction.tuple_index) |tuple_index| {
                        if (tuple_index >= 0 and tuple_index < @as(i64, @intCast(tuple_value.elements.len))) {
                            const element_id = tuple_value.elements[@intCast(tuple_index)];
                            if (element_id.index < last_uses.len) last_uses[element_id.index] = instruction_index;
                        }
                    } else {
                        for (tuple_value.elements) |element_id| {
                            if (element_id.index < last_uses.len) last_uses[element_id.index] = instruction_index;
                        }
                    }
                }
            }
        }
        for (instruction.outputs) |output_id| {
            if (output_id.index < value_producers.len) value_producers[output_id.index] = instruction_index;
        }
        const node_kind = programNodeKind(instruction.kind);
        const fusion_group = if (programNodeFusible(node_kind)) group: {
            if (current_fusion_group == null) {
                current_fusion_group = fusion_group_count;
                max_fusion_groups[fusion_group_count] = .{
                    .id = fusion_group_count,
                    .kind = .view_elementwise,
                    .first_node = instruction_index,
                    .last_node = instruction_index,
                    .node_count = 0,
                };
                fusion_group_count += 1;
            }
            max_fusion_groups[current_fusion_group.?].last_node = instruction_index;
            max_fusion_groups[current_fusion_group.?].node_count += 1;
            break :group current_fusion_group;
        } else group: {
            current_fusion_group = null;
            break :group null;
        };
        const node_inputs = try programNodeInputs(allocator, plan, instruction);
        var node_inputs_owned = true;
        errdefer if (node_inputs_owned) allocator.free(node_inputs);
        const node_outputs = try allocator.dupe(ir.ValueId, instruction.outputs);
        var node_outputs_owned = true;
        errdefer if (node_outputs_owned) allocator.free(node_outputs);
        const node_subprograms = try buildNodeSubprograms(allocator, plan, instruction, instruction_index, subprograms, &initialized_subprograms);
        var node_subprograms_owned = true;
        errdefer if (node_subprograms_owned and node_subprograms.len != 0) allocator.free(node_subprograms);
        const node_control_flow = try buildNodeControlFlow(
            allocator,
            instruction,
            instruction_index,
            node_subprograms,
            subprograms,
            control_flows,
            &initialized_control_flows,
        );
        nodes[instruction_index] = .{
            .instruction_index = instruction_index,
            .kind = node_kind,
            .inputs = node_inputs,
            .outputs = node_outputs,
            .subprograms = node_subprograms,
            .control_flow = node_control_flow,
            .materializes = instructionMaterializes(instruction.kind),
            .fusion_group = fusion_group,
        };
        node_inputs_owned = false;
        node_outputs_owned = false;
        node_subprograms_owned = false;
        initialized_nodes += 1;
    }

    const materialization_boundaries = try allocator.alloc(backend.MaterializationBoundary, plan.output_ids.len);
    errdefer allocator.free(materialization_boundaries);
    for (plan.output_ids, 0..) |output_id, index| {
        materialization_boundaries[index] = .{
            .value_id = output_id,
            .reason = .pjrt_output,
        };
    }

    const values = try allocator.alloc(backend.ProgramValue, plan.values.len);
    errdefer allocator.free(values);
    for (plan.values, 0..) |value, value_index| {
        values[value_index] = .{
            .value_id = value.id,
            .byte_size = ir.denseByteSize(value.descriptor.element_type, value.descriptor.dims),
            .producer_node = value_producers[value_index],
            .last_use_node = last_uses[value_index],
            .is_output = output_values[value_index],
        };
    }
    for (materialization_boundaries, 0..) |boundary, boundary_index| {
        if (boundary.value_id.index >= values.len) continue;
        if (values[boundary.value_id.index].materialization_boundary == null) {
            values[boundary.value_id.index].materialization_boundary = boundary_index;
        }
    }

    var edge_count: usize = 0;
    for (nodes) |node| {
        for (node.inputs) |input_id| {
            if (input_id.index >= values.len) continue;
            if (values[input_id.index].producer_node != null) edge_count += 1;
        }
    }

    const edges = try allocator.alloc(backend.ProgramEdge, edge_count);
    errdefer allocator.free(edges);
    var edge_index: usize = 0;
    for (nodes, 0..) |node, to_node| {
        for (node.inputs) |input_id| {
            if (input_id.index >= values.len) continue;
            const from_node = values[input_id.index].producer_node orelse continue;
            edges[edge_index] = .{
                .value_id = input_id,
                .from_node = from_node,
                .to_node = to_node,
            };
            edge_index += 1;
        }
    }

    const fusion_groups = try buildFusionGroups(allocator, max_fusion_groups[0..fusion_group_count], nodes, values, edges);
    errdefer deinitFusionGroups(allocator, fusion_groups);

    const max_schedule_len = nodes.len + if (materialization_boundaries.len == 0) @as(usize, 0) else 1;
    const max_schedule = try allocator.alloc(backend.ProgramScheduleItem, max_schedule_len);
    defer allocator.free(max_schedule);
    var schedule_index: usize = 0;
    var node_index: usize = 0;
    while (node_index < nodes.len) {
        const node = nodes[node_index];
        if (node.fusion_group) |group_id| {
            const group = fusion_groups[group_id];
            if (group.first_node == node_index) {
                max_schedule[schedule_index] = .{
                    .kind = .fusion_group,
                    .index = group_id,
                    .count = group.node_indices.len,
                };
                schedule_index += 1;
                node_index = group.last_node + 1;
                continue;
            }
        }
        max_schedule[schedule_index] = .{
            .kind = .node,
            .index = node_index,
        };
        schedule_index += 1;
        node_index += 1;
    }
    if (materialization_boundaries.len != 0) {
        max_schedule[schedule_index] = .{
            .kind = .materialization_boundary,
            .index = 0,
            .count = materialization_boundaries.len,
        };
        schedule_index += 1;
    }
    const schedule = try allocator.dupe(backend.ProgramScheduleItem, max_schedule[0..schedule_index]);
    errdefer allocator.free(schedule);

    const program = backend.Program{
        .allocator = allocator,
        .values = values,
        .nodes = nodes,
        .edges = edges,
        .schedule = schedule,
        .subprograms = subprograms,
        .control_flows = control_flows,
        .fusion_groups = fusion_groups,
        .materialization_boundaries = materialization_boundaries,
        .fusion_group_count = fusion_group_count,
    };
    try program.validateWithWriter(diagnostic_writer);
    subprograms_owned = false;
    control_flows_owned = false;
    return program;
}

fn countPlanSubprograms(plan: *const ir.ExecutablePlan) !usize {
    var count: usize = 0;
    for (plan.instructions) |instruction| {
        if (instruction.kind != .while_) continue;
        count += instruction.region_ids.len;
    }
    return count;
}

fn countPlanControlFlows(plan: *const ir.ExecutablePlan) !usize {
    var count: usize = 0;
    for (plan.instructions) |instruction| {
        if (instruction.kind == .while_) count += 1;
    }
    return count;
}

fn buildNodeSubprograms(
    allocator: std.mem.Allocator,
    plan: *const ir.ExecutablePlan,
    instruction: ir.PlanInstruction,
    instruction_index: usize,
    subprograms: []backend.ProgramSubprogram,
    initialized_subprograms: *usize,
) ![]const usize {
    if (instruction.kind != .while_ or instruction.region_ids.len == 0) return &.{};
    const indices = try allocator.alloc(usize, instruction.region_ids.len);
    errdefer allocator.free(indices);
    for (instruction.region_ids, 0..) |region_id, index| {
        if (region_id.index >= plan.regions.len or initialized_subprograms.* >= subprograms.len) return error.InvalidProgram;
        const subprogram_index = initialized_subprograms.*;
        subprograms[subprogram_index] = try cloneProgramSubprogram(
            allocator,
            plan.regions[region_id.index],
            subprogram_index,
            instruction_index,
        );
        initialized_subprograms.* += 1;
        indices[index] = subprogram_index;
    }
    return indices;
}

fn buildNodeControlFlow(
    allocator: std.mem.Allocator,
    instruction: ir.PlanInstruction,
    instruction_index: usize,
    node_subprograms: []const usize,
    subprograms: []const backend.ProgramSubprogram,
    control_flows: []backend.ProgramControlFlow,
    initialized_control_flows: *usize,
) !?usize {
    if (instruction.kind != .while_) return null;
    if (node_subprograms.len != 2 or initialized_control_flows.* >= control_flows.len) return error.InvalidProgram;
    if (node_subprograms[0] >= subprograms.len) return error.InvalidProgram;
    const condition = subprograms[node_subprograms[0]];
    if (condition.terminator_operands.len != 1) return error.InvalidProgram;
    const state_inputs = try allocator.dupe(ir.ValueId, instruction.inputs);
    errdefer allocator.free(state_inputs);
    const state_outputs = try allocator.dupe(ir.ValueId, instruction.outputs);
    errdefer allocator.free(state_outputs);
    const control_flow_index = initialized_control_flows.*;
    control_flows[control_flow_index] = .{
        .id = control_flow_index,
        .parent_node = instruction_index,
        .kind = .while_loop,
        .condition_subprogram = node_subprograms[0],
        .body_subprogram = node_subprograms[1],
        .state_inputs = state_inputs,
        .state_outputs = state_outputs,
        .predicate_output = condition.terminator_operands[0],
    };
    initialized_control_flows.* += 1;
    return control_flow_index;
}

fn cloneProgramSubprogram(
    allocator: std.mem.Allocator,
    region: ir.PlanRegion,
    subprogram_id: usize,
    parent_node: usize,
) !backend.ProgramSubprogram {
    const values = try cloneRegionValueList(allocator, region.values);
    errdefer program_mod.freeRegionValueList(allocator, values);
    const arguments = try cloneDescriptorList(allocator, region.argument_descriptors);
    errdefer program_mod.freeDescriptorList(allocator, arguments);
    const instructions = try cloneRegionInstructionList(allocator, region.instructions);
    errdefer program_mod.freeRegionInstructionList(allocator, instructions);
    const returns = try cloneDescriptorList(allocator, region.return_descriptors);
    errdefer program_mod.freeDescriptorList(allocator, returns);
    const terminator_operand_ids = try allocator.dupe(ir.RegionValueId, region.terminator_operands);
    errdefer allocator.free(terminator_operand_ids);
    const terminator_operand_descriptors = try cloneDescriptorList(allocator, region.terminator_operand_descriptors);
    errdefer program_mod.freeDescriptorList(allocator, terminator_operand_descriptors);
    return .{
        .id = subprogram_id,
        .parent_node = parent_node,
        .region_id = region.id,
        .kind = region.kind,
        .values = values,
        .argument_descriptors = arguments,
        .instructions = instructions,
        .return_descriptors = returns,
        .terminator_operands = terminator_operand_ids,
        .terminator_operand_descriptors = terminator_operand_descriptors,
    };
}

fn cloneDescriptorList(allocator: std.mem.Allocator, source: []const ir.BufferDescriptor) ![]const ir.BufferDescriptor {
    const descriptors = try allocator.alloc(ir.BufferDescriptor, source.len);
    var initialized: usize = 0;
    errdefer {
        for (descriptors[0..initialized]) |descriptor| {
            if (descriptor.dims.len != 0) allocator.free(descriptor.dims);
        }
        allocator.free(descriptors);
    }
    for (source, descriptors) |src, *dst| {
        const dims = try allocator.dupe(i64, src.dims);
        dst.* = .{
            .element_type = src.element_type,
            .dims = dims,
            .layout = src.layout,
            .device_id = src.device_id,
            .memory_id = src.memory_id,
            .shard_index = src.shard_index,
        };
        initialized += 1;
    }
    return descriptors;
}

fn cloneRegionValueList(allocator: std.mem.Allocator, source: []const ir.RegionValue) ![]const ir.RegionValue {
    const values = try allocator.alloc(ir.RegionValue, source.len);
    var initialized: usize = 0;
    errdefer {
        for (values[0..initialized]) |value| {
            if (value.descriptor.dims.len != 0) allocator.free(value.descriptor.dims);
            if (value.literal) |literal| allocator.free(literal);
        }
        allocator.free(values);
    }
    for (source, values) |src, *dst| {
        const dims = try allocator.dupe(i64, src.descriptor.dims);
        var dims_owned = true;
        errdefer if (dims_owned) allocator.free(dims);
        const literal = if (src.literal) |bytes| try allocator.dupe(u8, bytes) else null;
        var literal_owned = true;
        errdefer if (literal_owned) if (literal) |bytes| allocator.free(bytes);
        dst.* = .{
            .id = src.id,
            .role = src.role,
            .descriptor = .{
                .element_type = src.descriptor.element_type,
                .dims = dims,
                .layout = src.descriptor.layout,
                .device_id = src.descriptor.device_id,
                .memory_id = src.descriptor.memory_id,
                .shard_index = src.descriptor.shard_index,
            },
            .literal = literal,
        };
        dims_owned = false;
        literal_owned = false;
        initialized += 1;
    }
    return values;
}

fn cloneRegionInstructionList(allocator: std.mem.Allocator, source: []const ir.RegionInstruction) ![]const ir.RegionInstruction {
    const instructions = try allocator.alloc(ir.RegionInstruction, source.len);
    var initialized: usize = 0;
    errdefer {
        for (instructions[0..initialized]) |instruction| {
            if (instruction.inputs.len != 0) allocator.free(instruction.inputs);
            if (instruction.outputs.len != 0) allocator.free(instruction.outputs);
            program_mod.freeDescriptorList(allocator, instruction.operand_descriptors);
            program_mod.freeDescriptorList(allocator, instruction.result_descriptors);
        }
        allocator.free(instructions);
    }
    for (source, instructions) |src, *dst| {
        const inputs = try allocator.dupe(ir.RegionValueId, src.inputs);
        errdefer allocator.free(inputs);
        const outputs = try allocator.dupe(ir.RegionValueId, src.outputs);
        errdefer allocator.free(outputs);
        const operands = try cloneDescriptorList(allocator, src.operand_descriptors);
        errdefer program_mod.freeDescriptorList(allocator, operands);
        const results = try cloneDescriptorList(allocator, src.result_descriptors);
        errdefer program_mod.freeDescriptorList(allocator, results);
        dst.* = .{
            .kind = src.kind,
            .line = src.line,
            .column = src.column,
            .inputs = inputs,
            .outputs = outputs,
            .operand_descriptors = operands,
            .result_descriptors = results,
            .compare_direction = src.compare_direction,
        };
        initialized += 1;
    }
    return instructions;
}

fn deinitProgramSubprograms(allocator: std.mem.Allocator, subprograms: []backend.ProgramSubprogram) void {
    for (subprograms) |subprogram| {
        program_mod.freeRegionValueList(allocator, subprogram.values);
        program_mod.freeDescriptorList(allocator, subprogram.argument_descriptors);
        program_mod.freeRegionInstructionList(allocator, subprogram.instructions);
        program_mod.freeDescriptorList(allocator, subprogram.return_descriptors);
        if (subprogram.terminator_operands.len != 0) allocator.free(subprogram.terminator_operands);
        program_mod.freeDescriptorList(allocator, subprogram.terminator_operand_descriptors);
    }
    if (subprograms.len != 0) allocator.free(subprograms);
}

fn deinitProgramControlFlows(allocator: std.mem.Allocator, control_flows: []backend.ProgramControlFlow) void {
    for (control_flows) |control_flow| {
        if (control_flow.state_inputs.len != 0) allocator.free(control_flow.state_inputs);
        if (control_flow.state_outputs.len != 0) allocator.free(control_flow.state_outputs);
    }
    if (control_flows.len != 0) allocator.free(control_flows);
}

fn programNodeInputs(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction) ![]const ir.ValueId {
    if (instruction.kind != .get_tuple_element or instruction.inputs.len == 0) {
        return allocator.dupe(ir.ValueId, instruction.inputs);
    }
    const tuple_id = instruction.inputs[0];
    if (tuple_id.index >= plan.values.len) return allocator.dupe(ir.ValueId, instruction.inputs);
    const tuple_value = plan.values[tuple_id.index];
    if (tuple_value.storage != .tuple) return allocator.dupe(ir.ValueId, instruction.inputs);
    const tuple_index = instruction.tuple_index orelse return allocator.dupe(ir.ValueId, instruction.inputs);
    if (tuple_index < 0 or tuple_index >= @as(i64, @intCast(tuple_value.elements.len))) {
        return allocator.dupe(ir.ValueId, instruction.inputs);
    }
    return allocator.dupe(ir.ValueId, &.{ tuple_id, tuple_value.elements[@intCast(tuple_index)] });
}

fn buildFusionGroups(
    allocator: std.mem.Allocator,
    groups: []const backend.FusionGroup,
    nodes: []const backend.ProgramNode,
    values: []const backend.ProgramValue,
    edges: []const backend.ProgramEdge,
) ![]backend.FusionGroup {
    const fusion_groups = try allocator.alloc(backend.FusionGroup, groups.len);
    errdefer allocator.free(fusion_groups);
    @memcpy(fusion_groups, groups);
    var initialized_groups: usize = 0;
    errdefer deinitFusionGroups(allocator, fusion_groups[0..initialized_groups]);

    const mark_count = groups.len * values.len;
    const input_marks = try allocator.alloc(bool, mark_count);
    defer allocator.free(input_marks);
    @memset(input_marks, false);
    const output_marks = try allocator.alloc(bool, mark_count);
    defer allocator.free(output_marks);
    @memset(output_marks, false);

    for (nodes, 0..) |node, node_index| {
        const group_id = node.fusion_group orelse continue;
        for (node.inputs) |input_id| {
            if (input_id.index >= values.len) continue;
            if (values[input_id.index].producer_node == null) {
                input_marks[groupMarkIndex(values.len, group_id, input_id.index)] = true;
            }
        }
        for (node.outputs) |output_id| {
            if (output_id.index >= values.len) continue;
            if (values[output_id.index].materialization_boundary != null) {
                output_marks[groupMarkIndex(values.len, group_id, output_id.index)] = true;
            }
        }
        _ = node_index;
    }

    for (edges) |edge| {
        const to_group = nodes[edge.to_node].fusion_group;
        const from_group = nodes[edge.from_node].fusion_group;
        if (to_group) |group_id| {
            if (from_group != group_id) {
                input_marks[groupMarkIndex(values.len, group_id, edge.value_id.index)] = true;
            }
        }
        if (from_group) |group_id| {
            if (to_group != group_id) {
                output_marks[groupMarkIndex(values.len, group_id, edge.value_id.index)] = true;
            }
        }
    }

    for (fusion_groups) |*group| {
        group.node_indices = try fusionGroupNodeIndices(allocator, group.*);
        errdefer allocator.free(group.node_indices);
        group.input_values = try markedValueIds(allocator, values, input_marks[group.id * values.len ..][0..values.len]);
        errdefer allocator.free(group.input_values);
        group.output_values = try markedValueIds(allocator, values, output_marks[group.id * values.len ..][0..values.len]);
        initialized_groups += 1;
    }

    return fusion_groups;
}

fn deinitFusionGroups(allocator: std.mem.Allocator, groups: []backend.FusionGroup) void {
    for (groups) |group| {
        if (group.node_indices.len != 0) allocator.free(group.node_indices);
        if (group.input_values.len != 0) allocator.free(group.input_values);
        if (group.output_values.len != 0) allocator.free(group.output_values);
    }
    allocator.free(groups);
}

fn fusionGroupNodeIndices(allocator: std.mem.Allocator, group: backend.FusionGroup) ![]const usize {
    if (group.node_count == 0) return allocator.alloc(usize, 0);
    if (group.last_node < group.first_node) return error.CommandSubmissionFailed;
    if (group.last_node - group.first_node + 1 != group.node_count) return error.CommandSubmissionFailed;

    const node_indices = try allocator.alloc(usize, group.node_count);
    var index: usize = 0;
    var node_index = group.first_node;
    while (index < node_indices.len) : ({
        index += 1;
        node_index += 1;
    }) {
        node_indices[index] = node_index;
    }
    return node_indices;
}

fn groupMarkIndex(value_count: usize, group_id: usize, value_index: usize) usize {
    return group_id * value_count + value_index;
}

fn markedValueIds(allocator: std.mem.Allocator, values: []const backend.ProgramValue, marks: []const bool) ![]const ir.ValueId {
    var count: usize = 0;
    for (marks) |mark| {
        if (mark) count += 1;
    }
    const ids = try allocator.alloc(ir.ValueId, count);
    var index: usize = 0;
    for (values, marks) |value, mark| {
        if (!mark) continue;
        ids[index] = value.value_id;
        index += 1;
    }
    return ids;
}

fn programNodeKind(instruction_kind: ir.PlanInstructionKind) backend.ProgramNodeKind {
    return switch (instruction_kind) {
        .constant => .constant,
        .copy_arg0 => .parameter,
        .tuple, .get_tuple_element => .structural,
        .reshape, .transpose, .broadcast_in_dim, .slice => .view,
        .add, .subtract, .multiply, .divide, .maximum, .minimum, .power, .atan2, .remainder, .and_, .or_, .xor, .shift_left, .shift_right_arithmetic, .shift_right_logical, .negate, .exp, .expm1, .tanh, .sqrt, .rsqrt, .abs, .cbrt, .ceil, .floor, .log, .log1p, .logistic, .sine, .cosine, .not_, .sign, .is_finite, .round_nearest_afz, .round_nearest_even, .popcnt, .count_leading_zeros, .real, .imag, .compare, .select, .clamp => .elementwise,
        .reduce_sum, .reduce_max, .reduce_min, .reduce_and, .reduce_or, .reduce_window_sum, .reduce_window_max => .reduction,
        .dot_general => .matmul,
        .while_ => .control_flow,
        .sort, .top_k, .gather, .scatter, .dynamic_slice, .dynamic_update_slice, .pad, .reverse, .concatenate, .iota, .partition_id, .convert, .bitcast_convert, .reduce_precision, .rng, .rng_bit_generator, .custom_call, .optimization_barrier => .materialize,
        else => .library_call,
    };
}

fn instructionMaterializes(instruction_kind: ir.PlanInstructionKind) bool {
    return switch (instruction_kind) {
        .tuple, .get_tuple_element, .reshape, .transpose, .broadcast_in_dim, .slice => false,
        else => true,
    };
}

fn programNodeFusible(node_kind: backend.ProgramNodeKind) bool {
    return switch (node_kind) {
        .view, .elementwise => true,
        else => false,
    };
}

fn compileExecutable(backend_impl: backend.Backend, allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, device_local_hardware_ids: []const i32) backend.Error!?backend.ExecutableHandle {
    if (executableLoweringIssue(plan, device_local_hardware_ids)) |_| return null;

    const executable = try allocator.create(CompiledExecutable);
    errdefer allocator.destroy(executable);
    const ids = try allocator.dupe(i32, device_local_hardware_ids);
    errdefer allocator.free(ids);
    const constant_handles = try allocator.alloc(?backend.BufferHandle, plan.instructions.len * device_local_hardware_ids.len);
    errdefer allocator.free(constant_handles);
    @memset(constant_handles, null);
    errdefer destroyConstantHandles(backend_impl, constant_handles);
    var program = buildBackendProgram(allocator, plan, null) catch |err| switch (err) {
        error.WriteFailed => unreachable,
        error.InvalidDeviceCount => return error.InvalidDeviceCount,
        error.InvalidProgram => return error.InvalidProgram,
        error.UnsupportedElementType => return error.UnsupportedElementType,
        error.ShapeMismatch => return error.ShapeMismatch,
        error.BufferAllocationFailed => return error.BufferAllocationFailed,
        error.CommandSubmissionFailed => return error.CommandSubmissionFailed,
        error.BufferCopyFailed => return error.BufferCopyFailed,
        error.OutOfMemory => return error.OutOfMemory,
        error.InvalidCustomCall => return error.InvalidCustomCall,
    };
    errdefer program.deinit();
    const while_constant_handles = try allocator.alloc(?backend.BufferHandle, program.control_flows.len * device_local_hardware_ids.len * 2);
    errdefer allocator.free(while_constant_handles);
    @memset(while_constant_handles, null);
    errdefer destroyConstantHandles(backend_impl, while_constant_handles);
    const compiled_program_contexts = try allocator.alloc(CompiledProgramContext, device_local_hardware_ids.len);
    errdefer allocator.free(compiled_program_contexts);
    const compiled_program_handles = try allocator.alloc(?mlx_call.ProgramHandle, device_local_hardware_ids.len);
    errdefer allocator.free(compiled_program_handles);
    @memset(compiled_program_handles, null);
    errdefer destroyCompiledPrograms(compiled_program_handles);
    const argument_capture_states = try allocator.alloc(ArgumentCaptureState, device_local_hardware_ids.len);
    errdefer allocator.free(argument_capture_states);
    for (argument_capture_states) |*state| state.* = .{};
    errdefer destroyArgumentCaptureStates(allocator, argument_capture_states);
    const liveness_stats = try program.livenessStats();

    var resident_constant_count: usize = 0;
    var resident_constant_bytes: usize = 0;
    for (device_local_hardware_ids, 0..) |local_hardware_id, device_index| {
        for (plan.instructions, 0..) |instruction, instruction_index| {
            if (instruction.kind != .constant) continue;
            const output_id = instruction.outputs[0];
            const descriptor = plan.values[output_id.index].descriptor;
            const literal = instruction.literal.?;
            constant_handles[constantIndex(plan.instructions.len, device_index, instruction_index)] = (try bufferFromHost(
                backend_impl,
                local_hardware_id,
                descriptor.element_type,
                descriptor.dims,
                literal,
            )) orelse return error.CommandSubmissionFailed;
            resident_constant_count += 1;
            resident_constant_bytes += literal.len;
        }
    }
    for (program.control_flows, 0..) |control_flow, control_flow_index| {
        if (control_flow.condition_subprogram >= program.subprograms.len or control_flow.body_subprogram >= program.subprograms.len) return error.InvalidProgram;
        const pattern = matchWhileF32LtAddPattern(program.subprograms[control_flow.condition_subprogram], program.subprograms[control_flow.body_subprogram]) orelse continue;
        for (device_local_hardware_ids, 0..) |local_hardware_id, device_index| {
            if (pattern.limit.role == .constant) {
                const limit_literal = pattern.limit.literal orelse return error.InvalidProgram;
                while_constant_handles[whileConstantIndex(program.control_flows.len, device_index, control_flow_index, 0)] = (try bufferFromHost(
                    backend_impl,
                    local_hardware_id,
                    pattern.limit.descriptor.element_type,
                    pattern.limit.descriptor.dims,
                    limit_literal,
                )) orelse return error.CommandSubmissionFailed;
                resident_constant_count += 1;
                resident_constant_bytes += limit_literal.len;
            }
            if (pattern.step.value.role == .constant) {
                const step_literal = pattern.step.value.literal orelse return error.InvalidProgram;
                while_constant_handles[whileConstantIndex(program.control_flows.len, device_index, control_flow_index, 1)] = (try bufferFromHost(
                    backend_impl,
                    local_hardware_id,
                    pattern.step.value.descriptor.element_type,
                    pattern.step.value.descriptor.dims,
                    step_literal,
                )) orelse return error.CommandSubmissionFailed;
                resident_constant_count += 1;
                resident_constant_bytes += step_literal.len;
            }
        }
    }

    executable.* = .{
        .allocator = allocator,
        .plan = plan,
        .device_local_hardware_ids = ids,
        .constant_handles = constant_handles,
        .while_constant_handles = while_constant_handles,
        .compiled_program_contexts = compiled_program_contexts,
        .compiled_program_handles = compiled_program_handles,
        .argument_capture_states = argument_capture_states,
        .program = program,
        .stats = .{
            .resident_constant_count = resident_constant_count,
            .resident_constant_bytes = resident_constant_bytes,
            .program_value_count = program.values.len,
            .program_node_count = program.nodes.len,
            .program_edge_count = program.edges.len,
            .program_schedule_item_count = program.schedule.len,
            .program_subprogram_count = program.subprograms.len,
            .program_control_flow_count = program.control_flows.len,
            .program_fusion_group_count = program.fusion_groups.len,
            .program_materialization_boundary_count = program.materialization_boundaries.len,
            .program_planned_release_count = liveness_stats.planned_release_count,
            .program_planned_release_bytes = liveness_stats.planned_release_bytes,
            .program_peak_live_value_count = liveness_stats.peak_live_value_count,
            .program_peak_live_bytes = liveness_stats.peak_live_bytes,
            .program_device_count = device_local_hardware_ids.len,
        },
    };
    for (compiled_program_contexts, 0..) |*context, device_index| {
        context.* = .{
            .executable = executable,
            .device_index = device_index,
        };
    }
    if (programCompileEnabled() and planSupportsCompiledProgram(plan)) {
        for (compiled_program_handles, 0..) |*handle_slot, device_index| {
            handle_slot.* = mlx_call.programCreate(
                &compiled_program_contexts[device_index],
                plan.parameter_shardings.len,
                plan.output_ids.len,
                compiledProgramBuildCallback,
            ) orelse return error.CommandSubmissionFailed;
        }
    }
    return @ptrCast(executable);
}

fn writeExecutableLoweringDiagnostic(_: backend.Backend, plan: *const ir.ExecutablePlan, device_local_hardware_ids: []const i32, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    if (executableLoweringIssue(plan, device_local_hardware_ids)) |issue| {
        try writeLoweringIssue(plan, issue, writer);
        return;
    }
    var program = buildBackendProgram(std.heap.page_allocator, plan, writer) catch |err| {
        if (err == error.InvalidProgram) return;
        if (err == error.OutOfMemory) {
            try writer.writeAll("invalid executable plan: pass=mlx-backend-legalization detail=\"failed to allocate backend program diagnostic\" feature=mlx-backend-executable");
            return;
        }
        try writer.writeAll("invalid executable plan: pass=mlx-backend-legalization detail=\"failed to build backend program diagnostic\" feature=mlx-backend-executable");
        return;
    };
    defer program.deinit();
    try writer.writeAll("invalid executable plan: pass=mlx-backend-legalization detail=\"MLX backend rejected executable plan without a specific issue\" feature=mlx-backend-executable");
}

fn traceScheduleFailure(executable: *const CompiledExecutable, item: backend.ProgramScheduleItem, err: backend.Error) void {
    if (std.c.getenv("PJRTX_TRACE") == null) return;
    std.debug.print(
        "pjrtx_trace event=backend_execute_error schedule_kind={s} schedule_index={d} schedule_count={d} err={s}",
        .{ @tagName(item.kind), item.index, item.count, @errorName(err) },
    );
    switch (item.kind) {
        .node => {
            if (item.index < executable.program.nodes.len) {
                const node = executable.program.nodes[item.index];
                std.debug.print(" node_kind={s} instruction={d}", .{ @tagName(node.kind), node.instruction_index });
                if (node.instruction_index < executable.plan.instructions.len) {
                    const instruction = executable.plan.instructions[node.instruction_index];
                    std.debug.print(" op={s}", .{@tagName(instruction.kind)});
                    if (instruction.outputs.len > 0 and instruction.outputs[0].index < executable.plan.values.len) {
                        const descriptor = executable.plan.values[instruction.outputs[0].index].descriptor;
                        std.debug.print(" output_value={d} dtype={s} rank={d}", .{ instruction.outputs[0].index, @tagName(descriptor.element_type), descriptor.dims.len });
                    }
                }
            }
        },
        .fusion_group => {
            if (item.index < executable.program.fusion_groups.len) {
                const group = executable.program.fusion_groups[item.index];
                std.debug.print(" group_first_node={d} group_last_node={d} group_nodes={d}", .{ group.first_node, group.last_node, group.node_count });
            }
        },
        .materialization_boundary => {},
    }
    std.debug.print("\n", .{});
}

fn executeExecutable(backend_impl: backend.Backend, allocator: std.mem.Allocator, executable_handle: backend.ExecutableHandle, device_index: usize, arguments: []const backend.BufferHandle) backend.Error!?backend.ExecutionResult {
    const executable: *CompiledExecutable = @ptrCast(@alignCast(executable_handle));
    const profile_enabled = profileEnabled();
    const profile_verbose = profile_enabled and profileVerbose();
    const execute_start_ns = profileStart(profile_enabled);
    var profile: ExecuteProfile = .{};
    const plan = executable.plan;
    if (device_index >= executable.device_local_hardware_ids.len) return error.CommandSubmissionFailed;
    if (arguments.len != plan.parameter_shardings.len) return error.CommandSubmissionFailed;
    if (device_index < executable.compiled_program_handles.len) {
        if (executable.compiled_program_handles[device_index]) |compiled_program| {
            try maybeCreateInitialArgumentCapturedProgram(executable, device_index, arguments);
            if (try executeArgumentCapturedProgram(backend_impl, allocator, executable, device_index, arguments, &profile)) |outputs| {
                if (profile_enabled) profile.wall_us = profileElapsedUs(execute_start_ns);
                recordSuccessfulExecute(executable, device_index, executable.device_local_hardware_ids[device_index]);
                if (profile_enabled) {
                    recordExecuteProfile(executable, profile);
                    writeExecuteProfile(executable, device_index, arguments.len, outputs.len, profile);
                }
                return .{
                    .outputs = outputs,
                    .completion = backend.ExecutionCompletion.completed(),
                };
            }
            const donated_input_indices = try donatedProgramInputIndices(allocator, executable.plan, null);
            defer allocator.free(donated_input_indices);
            const outputs = try executeCompiledProgram(backend_impl, allocator, executable, compiled_program, arguments, donated_input_indices, &profile);
            try updateArgumentCaptureState(executable, device_index, arguments);
            if (profile_enabled) profile.wall_us = profileElapsedUs(execute_start_ns);
            recordSuccessfulExecute(executable, device_index, executable.device_local_hardware_ids[device_index]);
            if (profile_enabled) {
                recordExecuteProfile(executable, profile);
                writeExecuteProfile(executable, device_index, arguments.len, outputs.len, profile);
            }
            return .{
                .outputs = outputs,
                .completion = backend.ExecutionCompletion.completed(),
            };
        }
    }

    var value_handles = try allocator.alloc(?backend.BufferHandle, plan.values.len);
    defer allocator.free(value_handles);
    @memset(value_handles, null);

    var value_owned = try allocator.alloc(bool, plan.values.len);
    defer allocator.free(value_owned);
    @memset(value_owned, false);
    defer destroyOwnedValueHandles(backend_impl, value_handles, value_owned);

    var parameter_index: usize = 0;
    for (plan.values) |value| {
        if (value.role != .parameter) continue;
        if (parameter_index >= arguments.len or value.id.index >= value_handles.len) return error.CommandSubmissionFailed;
        value_handles[value.id.index] = arguments[parameter_index];
        parameter_index += 1;
    }

    for (executable.program.schedule, 0..) |schedule_item, schedule_index| {
        const schedule_start_ns = profileStart(profile_enabled);
        switch (schedule_item.kind) {
            .node => {
                (executeProgramNode(backend_impl, allocator, executable, device_index, value_handles, value_owned, schedule_item.index, true) catch |err| {
                    traceScheduleFailure(executable, schedule_item, err);
                    return err;
                }) orelse return null;
            },
            .fusion_group => {
                (executeFusionGroup(backend_impl, allocator, executable, device_index, value_handles, value_owned, schedule_item.index, schedule_item.count) catch |err| {
                    traceScheduleFailure(executable, schedule_item, err);
                    return err;
                }) orelse return null;
            },
            .materialization_boundary => {
                evalMaterializationBoundaryRange(&executable.program, value_handles, schedule_item.index, schedule_item.count) catch |err| {
                    traceScheduleFailure(executable, schedule_item, err);
                    return err;
                };
                recordMaterializationEval(executable, schedule_item.count);
            },
        }
        const schedule_us = profileElapsedUs(schedule_start_ns);
        if (profile_enabled) {
            profile.schedule_us +|= schedule_us;
            profile.schedule_peak_us = @max(profile.schedule_peak_us, schedule_us);
            switch (schedule_item.kind) {
                .node => {
                    profile.node_us +|= schedule_us;
                    profile.node_peak_us = @max(profile.node_peak_us, schedule_us);
                },
                .fusion_group => {
                    profile.fusion_group_us +|= schedule_us;
                    profile.fusion_group_peak_us = @max(profile.fusion_group_peak_us, schedule_us);
                },
                .materialization_boundary => {
                    profile.materialization_eval_us +|= schedule_us;
                    profile.materialization_eval_peak_us = @max(profile.materialization_eval_peak_us, schedule_us);
                },
            }
            if (profile_verbose) {
                writeScheduleProfile(executable, schedule_index, schedule_item, schedule_us);
            }
        }
    }

    const output_clone_start_ns = profileStart(profile_enabled);
    const outputs = try allocator.alloc(backend.ExecutableOutput, plan.output_ids.len);
    errdefer allocator.free(outputs);
    var initialized: usize = 0;
    errdefer {
        for (outputs[0..initialized]) |output| destroyBuffer(backend_impl, output.handle);
    }

    for (plan.output_ids, 0..) |output_id, output_index| {
        if (output_id.index >= value_handles.len) return error.CommandSubmissionFailed;
        const value = value_handles[output_id.index] orelse return error.CommandSubmissionFailed;
        const handle = if (value_owned[output_id.index]) blk: {
            value_owned[output_id.index] = false;
            break :blk value;
        } else (try cloneBuffer(backend_impl, value)) orelse return null;
        const descriptor = plan.values[output_id.index].descriptor;
        outputs[output_index] = .{
            .handle = handle,
            .element_type = descriptor.element_type,
            .dims = descriptor.dims,
            .byte_size = ir.denseByteSize(descriptor.element_type, descriptor.dims),
        };
        initialized += 1;
    }
    const output_clone_us = profileElapsedUs(output_clone_start_ns);
    if (profile_enabled) {
        profile.output_clone_us = output_clone_us;
        profile.output_clone_peak_us = output_clone_us;
        profile.wall_us = profileElapsedUs(execute_start_ns);
    }

    recordSuccessfulExecute(executable, device_index, executable.device_local_hardware_ids[device_index]);
    if (profile_enabled) {
        recordExecuteProfile(executable, profile);
        writeExecuteProfile(executable, device_index, arguments.len, outputs.len, profile);
    }
    return .{
        .outputs = outputs,
        .completion = backend.ExecutionCompletion.completed(),
    };
}

fn executeCompiledProgram(
    backend_impl: backend.Backend,
    allocator: std.mem.Allocator,
    executable: *CompiledExecutable,
    compiled_program: mlx_call.ProgramHandle,
    arguments: []const backend.BufferHandle,
    donated_input_indices: []const u64,
    profile: *ExecuteProfile,
) backend.Error![]backend.ExecutableOutput {
    const profile_enabled = profileEnabled();
    const execute_start_ns = profileStart(profile_enabled);
    const program_outputs = mlx_call.programExecuteWithDonation(compiled_program, arguments, donated_input_indices) orelse return error.CommandSubmissionFailed;
    defer program_outputs.deinit();
    if (program_outputs.len() != executable.plan.output_ids.len) {
        return error.CommandSubmissionFailed;
    }

    const outputs = try allocator.alloc(backend.ExecutableOutput, executable.plan.output_ids.len);
    errdefer allocator.free(outputs);
    var initialized: usize = 0;
    errdefer {
        for (outputs[0..initialized]) |output| destroyBuffer(backend_impl, output.handle);
    }

    for (outputs, 0..) |*output, output_index| {
        const handle = program_outputs.take(output_index) orelse return error.CommandSubmissionFailed;
        const output_id = executable.plan.output_ids[output_index];
        if (output_id.index >= executable.plan.values.len) return error.CommandSubmissionFailed;
        const descriptor = executable.plan.values[output_id.index].descriptor;
        output.* = .{
            .handle = @ptrCast(handle),
            .element_type = descriptor.element_type,
            .dims = descriptor.dims,
            .byte_size = ir.denseByteSize(descriptor.element_type, descriptor.dims),
        };
        initialized += 1;
    }
    if (profile_enabled) {
        const elapsed = profileElapsedUs(execute_start_ns);
        profile.schedule_us +|= elapsed;
        profile.schedule_peak_us = @max(profile.schedule_peak_us, elapsed);
        profile.compiled_program_us +|= elapsed;
        profile.compiled_program_peak_us = @max(profile.compiled_program_peak_us, elapsed);
    }
    recordCompiledProgramExecute(executable, outputs.len);
    return outputs;
}

fn donatedProgramInputIndices(
    allocator: std.mem.Allocator,
    plan: *const ir.ExecutablePlan,
    dynamic_indices: ?[]const u64,
) backend.Error![]const u64 {
    var donated: std.ArrayList(u64) = .empty;
    errdefer donated.deinit(allocator);
    for (plan.donated_parameter_indices) |parameter_index| {
        if (parameterFeedsIdentityOutput(plan, parameter_index)) continue;
        if (dynamic_indices) |indices| {
            for (indices, 0..) |full_index, dynamic_index| {
                if (full_index == parameter_index) {
                    try donated.append(allocator, @intCast(dynamic_index));
                    break;
                }
            }
        } else {
            try donated.append(allocator, parameter_index);
        }
    }
    return try donated.toOwnedSlice(allocator);
}

fn parameterFeedsIdentityOutput(plan: *const ir.ExecutablePlan, parameter_index: u32) bool {
    for (plan.output_aliases) |alias| {
        if (alias.parameter_index == parameter_index and alias.kind == .identity) return true;
    }
    var seen_parameter_index: u32 = 0;
    for (plan.values) |value| {
        if (value.role != .parameter) continue;
        if (seen_parameter_index == parameter_index) {
            for (plan.output_ids) |output_id| {
                if (output_id.index == value.id.index) return true;
            }
            return false;
        }
        seen_parameter_index += 1;
    }
    return false;
}

const MinCapturedProgramStableInputs = 8;

fn maybeCreateInitialArgumentCapturedProgram(
    executable: *CompiledExecutable,
    device_index: usize,
    arguments: []const backend.BufferHandle,
) backend.Error!void {
    if (device_index >= executable.argument_capture_states.len) return;

    lockArgumentCapture(executable);
    defer executable.argument_capture_mutex.unlock();

    const state = &executable.argument_capture_states[device_index];
    if (state.program_handle != null or state.previous_arguments.len != 0) return;

    const dynamic_indices = try initialArgumentCaptureDynamicIndices(executable.allocator, executable.plan);
    errdefer executable.allocator.free(dynamic_indices);
    if (dynamic_indices.len >= arguments.len or arguments.len - dynamic_indices.len < MinCapturedProgramStableInputs) {
        executable.allocator.free(dynamic_indices);
        return;
    }

    const program = mlx_call.programCreateWithCaptures(
        &executable.compiled_program_contexts[device_index],
        arguments.len,
        executable.plan.output_ids.len,
        compiledProgramBuildCallback,
        arguments,
        dynamic_indices,
    ) orelse {
        executable.allocator.free(dynamic_indices);
        return;
    };

    state.dynamic_indices = dynamic_indices;
    state.program_handle = program;
    rememberArgumentCaptureBaseline(executable.allocator, state, arguments) catch |err| {
        resetArgumentCaptureState(executable.allocator, state);
        return err;
    };
}

fn initialArgumentCaptureDynamicIndices(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan) ![]u64 {
    var dynamic: std.ArrayList(u64) = .empty;
    errdefer dynamic.deinit(allocator);

    var parameter_index: u32 = 0;
    for (plan.values) |value| {
        if (value.role != .parameter) continue;
        if (initiallyDynamicParameter(plan, parameter_index, value.descriptor)) {
            try dynamic.append(allocator, parameter_index);
        }
        parameter_index += 1;
    }
    return try dynamic.toOwnedSlice(allocator);
}

fn initiallyDynamicParameter(plan: *const ir.ExecutablePlan, parameter_index: u32, descriptor: ir.BufferDescriptor) bool {
    for (plan.donated_parameter_indices) |donated_index| {
        if (donated_index == parameter_index) return true;
    }

    const byte_size = ir.denseByteSize(descriptor.element_type, descriptor.dims);
    if (byte_size == 0 or byte_size > InitialCaptureSmallControlBytes) return false;
    return switch (descriptor.element_type) {
        .pred, .s8, .s16, .s32, .s64, .u8, .u16, .u32, .u64 => true,
        else => false,
    };
}

fn executeArgumentCapturedProgram(
    backend_impl: backend.Backend,
    allocator: std.mem.Allocator,
    executable: *CompiledExecutable,
    device_index: usize,
    arguments: []const backend.BufferHandle,
    profile: *ExecuteProfile,
) backend.Error!?[]backend.ExecutableOutput {
    if (device_index >= executable.argument_capture_states.len) return null;
    lockArgumentCapture(executable);
    defer executable.argument_capture_mutex.unlock();

    const state = &executable.argument_capture_states[device_index];
    const program = state.program_handle orelse return null;
    if (!argumentCaptureMatches(state.*, arguments)) {
        resetArgumentCaptureState(executable.allocator, state);
        return null;
    }

    const dynamic_arguments = try allocator.alloc(backend.BufferHandle, state.dynamic_indices.len);
    defer allocator.free(dynamic_arguments);
    for (state.dynamic_indices, 0..) |dynamic_index, out_index| {
        if (dynamic_index >= arguments.len) return error.CommandSubmissionFailed;
        dynamic_arguments[out_index] = arguments[@intCast(dynamic_index)];
    }
    const donated_input_indices = try donatedProgramInputIndices(allocator, executable.plan, state.dynamic_indices);
    defer allocator.free(donated_input_indices);
    const outputs = try executeCompiledProgram(backend_impl, allocator, executable, program, dynamic_arguments, donated_input_indices, profile);
    recordCapturedProgramExecute(executable, dynamic_arguments.len, arguments.len - dynamic_arguments.len);
    return outputs;
}

fn argumentCaptureMatches(state: ArgumentCaptureState, arguments: []const backend.BufferHandle) bool {
    if (state.previous_arguments.len != arguments.len) return false;
    var dynamic_index_cursor: usize = 0;
    for (arguments, 0..) |argument, index| {
        if (dynamic_index_cursor < state.dynamic_indices.len and state.dynamic_indices[dynamic_index_cursor] == index) {
            dynamic_index_cursor += 1;
            continue;
        }
        if (state.previous_arguments[index] != argument) return false;
    }
    return true;
}

fn updateArgumentCaptureState(
    executable: *CompiledExecutable,
    device_index: usize,
    arguments: []const backend.BufferHandle,
) backend.Error!void {
    if (device_index >= executable.argument_capture_states.len) return;
    lockArgumentCapture(executable);
    defer executable.argument_capture_mutex.unlock();

    const state = &executable.argument_capture_states[device_index];
    if (state.program_handle != null) return;
    if (state.previous_arguments.len == 0) {
        try rememberArgumentCaptureBaseline(executable.allocator, state, arguments);
        return;
    }
    if (state.previous_arguments.len != arguments.len) {
        resetArgumentCaptureState(executable.allocator, state);
        try rememberArgumentCaptureBaseline(executable.allocator, state, arguments);
        return;
    }

    var dynamic_count: usize = 0;
    for (arguments, 0..) |argument, index| {
        if (state.previous_arguments[index] != argument) dynamic_count += 1;
    }
    const captured_count = arguments.len - dynamic_count;
    if (captured_count < MinCapturedProgramStableInputs or dynamic_count == arguments.len) {
        try rememberArgumentCaptureBaseline(executable.allocator, state, arguments);
        return;
    }

    const dynamic_indices = try executable.allocator.alloc(u64, dynamic_count);
    errdefer executable.allocator.free(dynamic_indices);
    var out_index: usize = 0;
    for (arguments, 0..) |argument, index| {
        if (state.previous_arguments[index] != argument) {
            dynamic_indices[out_index] = @intCast(index);
            out_index += 1;
        }
    }

    const program = mlx_call.programCreateWithCaptures(
        &executable.compiled_program_contexts[device_index],
        arguments.len,
        executable.plan.output_ids.len,
        compiledProgramBuildCallback,
        arguments,
        dynamic_indices,
    ) orelse {
        executable.allocator.free(dynamic_indices);
        try rememberArgumentCaptureBaseline(executable.allocator, state, arguments);
        return;
    };

    executable.allocator.free(state.dynamic_indices);
    state.dynamic_indices = dynamic_indices;
    state.program_handle = program;
    try rememberArgumentCaptureBaseline(executable.allocator, state, arguments);
}

fn rememberArgumentCaptureBaseline(
    allocator: std.mem.Allocator,
    state: *ArgumentCaptureState,
    arguments: []const backend.BufferHandle,
) !void {
    if (state.previous_arguments.len != arguments.len) {
        allocator.free(state.previous_arguments);
        state.previous_arguments = try allocator.alloc(?backend.BufferHandle, arguments.len);
    }
    for (arguments, 0..) |argument, index| {
        state.previous_arguments[index] = argument;
    }
}

fn resetArgumentCaptureState(allocator: std.mem.Allocator, state: *ArgumentCaptureState) void {
    if (state.program_handle) |program| mlx_call.programDestroy(program);
    state.program_handle = null;
    allocator.free(state.previous_arguments);
    state.previous_arguments = &.{};
    allocator.free(state.dynamic_indices);
    state.dynamic_indices = &.{};
}

fn compiledProgramBuildCallback(
    user_data: ?*anyopaque,
    call: mlx_call.ProgramBuildCall,
) bool {
    const context: *CompiledProgramContext = @ptrCast(@alignCast(user_data orelse return false));
    const executable = context.executable;
    const allocator = executable.allocator;
    if (call.inputCount() != executable.plan.parameter_shardings.len or call.outputCount() != executable.plan.output_ids.len) return false;

    const arguments = allocator.alloc(backend.BufferHandle, call.inputCount()) catch return false;
    defer allocator.free(arguments);
    for (arguments, 0..) |*argument, index| {
        argument.* = call.input(index) orelse return false;
    }

    const outputs = buildExecutableOutputHandlesForCompiledTrace(
        create(),
        allocator,
        executable,
        context.device_index,
        arguments,
    ) catch return false;
    defer allocator.free(outputs);
    if (outputs.len != call.outputCount()) {
        for (outputs) |output| destroyBuffer(create(), output.handle);
        return false;
    }
    for (outputs, 0..) |output, index| {
        if (!call.setOutput(index, output.handle)) return false;
    }
    return true;
}

fn buildExecutableOutputHandlesForCompiledTrace(
    backend_impl: backend.Backend,
    allocator: std.mem.Allocator,
    executable: *CompiledExecutable,
    device_index: usize,
    arguments: []const backend.BufferHandle,
) backend.Error![]backend.ExecutableOutput {
    const plan = executable.plan;
    if (device_index >= executable.device_local_hardware_ids.len) return error.CommandSubmissionFailed;
    if (arguments.len != plan.parameter_shardings.len) return error.CommandSubmissionFailed;

    var value_handles = try allocator.alloc(?backend.BufferHandle, plan.values.len);
    defer allocator.free(value_handles);
    @memset(value_handles, null);

    var value_owned = try allocator.alloc(bool, plan.values.len);
    defer allocator.free(value_owned);
    @memset(value_owned, false);
    defer destroyOwnedValueHandles(backend_impl, value_handles, value_owned);

    var parameter_index: usize = 0;
    for (plan.values) |value| {
        if (value.role != .parameter) continue;
        if (parameter_index >= arguments.len or value.id.index >= value_handles.len) return error.CommandSubmissionFailed;
        value_handles[value.id.index] = arguments[parameter_index];
        parameter_index += 1;
    }

    for (executable.program.schedule) |schedule_item| {
        switch (schedule_item.kind) {
            .node => {
                (executeProgramNode(backend_impl, allocator, executable, device_index, value_handles, value_owned, schedule_item.index, true) catch |err| {
                    traceScheduleFailure(executable, schedule_item, err);
                    return err;
                }) orelse return error.CommandSubmissionFailed;
            },
            .fusion_group => {
                (executeFusionGroup(backend_impl, allocator, executable, device_index, value_handles, value_owned, schedule_item.index, schedule_item.count) catch |err| {
                    traceScheduleFailure(executable, schedule_item, err);
                    return err;
                }) orelse return error.CommandSubmissionFailed;
            },
            .materialization_boundary => {},
        }
    }

    const outputs = try allocator.alloc(backend.ExecutableOutput, plan.output_ids.len);
    errdefer allocator.free(outputs);
    var initialized: usize = 0;
    errdefer {
        for (outputs[0..initialized]) |output| destroyBuffer(backend_impl, output.handle);
    }

    for (plan.output_ids, 0..) |output_id, output_index| {
        if (output_id.index >= value_handles.len) return error.CommandSubmissionFailed;
        const value = value_handles[output_id.index] orelse return error.CommandSubmissionFailed;
        const handle = if (value_owned[output_id.index]) blk: {
            value_owned[output_id.index] = false;
            break :blk value;
        } else (try cloneBuffer(backend_impl, value)) orelse return error.CommandSubmissionFailed;
        const descriptor = plan.values[output_id.index].descriptor;
        outputs[output_index] = .{
            .handle = handle,
            .element_type = descriptor.element_type,
            .dims = descriptor.dims,
            .byte_size = ir.denseByteSize(descriptor.element_type, descriptor.dims),
        };
        initialized += 1;
    }
    return outputs;
}

fn executionEventStatus(_: backend.Backend, _: backend.ExecutionEventHandle) backend.Error!backend.ExecutionEventStatus {
    return .{
        .state = .failed,
        .message = "MLX Metal backend does not expose asynchronous execution event handles",
    };
}

fn destroyExecutionEvent(_: backend.Backend, _: backend.ExecutionEventHandle) void {}

fn executeFusionGroup(
    backend_impl: backend.Backend,
    allocator: std.mem.Allocator,
    executable: *CompiledExecutable,
    device_index: usize,
    value_handles: []?backend.BufferHandle,
    value_owned: []bool,
    group_index: usize,
    scheduled_node_count: usize,
) backend.Error!?void {
    if (group_index >= executable.program.fusion_groups.len) return error.CommandSubmissionFailed;
    const group = executable.program.fusion_groups[group_index];
    if (scheduled_node_count != group.node_indices.len) return error.CommandSubmissionFailed;
    switch (group.kind) {
        .view_elementwise => {},
    }
    recordFusionGroupExecute(executable);
    for (group.node_indices) |group_node_index| {
        if (group_node_index >= executable.program.nodes.len) return error.CommandSubmissionFailed;
        const node = executable.program.nodes[group_node_index];
        if (node.fusion_group != group_index) return error.CommandSubmissionFailed;
        (try executeProgramNode(backend_impl, allocator, executable, device_index, value_handles, value_owned, group_node_index, false)) orelse return null;
    }
    const released = releaseDeadFusionGroupValues(backend_impl, &executable.program, value_handles, value_owned, group);
    if (released != 0) recordReleasedIntermediateValues(executable, released);
    return {};
}

fn executeProgramNode(
    backend_impl: backend.Backend,
    allocator: std.mem.Allocator,
    executable: *CompiledExecutable,
    device_index: usize,
    value_handles: []?backend.BufferHandle,
    value_owned: []bool,
    node_index: usize,
    release_inputs: bool,
) backend.Error!?void {
    const plan = executable.plan;
    if (node_index >= executable.program.nodes.len) return error.CommandSubmissionFailed;
    const node = executable.program.nodes[node_index];
    const instruction_index = node.instruction_index;
    const instruction = plan.instructions[instruction_index];
    if (node.kind == .control_flow) {
        (try executeControlFlowNode(backend_impl, executable, device_index, value_handles, value_owned, node, instruction, release_inputs)) orelse return null;
        return {};
    }
    if (instruction.kind == .sort and instruction.inputs.len == 2 and instruction.outputs.len == 2) {
        const dimension = instruction.dimension orelse return null;
        const direction = instruction.compare_direction orelse .lt;
        const keys = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
        const values = value_handles[instruction.inputs[1].index] orelse return error.CommandSubmissionFailed;
        const key_output_id = instruction.outputs[0];
        const value_output_id = instruction.outputs[1];
        if (key_output_id.index >= plan.values.len or value_output_id.index >= plan.values.len) return error.CommandSubmissionFailed;
        const key_descriptor = plan.values[key_output_id.index].descriptor;
        const value_descriptor = plan.values[value_output_id.index].descriptor;
        const key_dims = instruction.dims orelse key_descriptor.dims;
        const value_dims = value_descriptor.dims;
        const sorted_keys = (try sort(backend_impl, keys, dimension, key_dims)) orelse return null;
        const directed_keys = (try reverseIfDescending(backend_impl, sorted_keys, dimension, key_dims, direction)) orelse return null;
        errdefer destroyBuffer(backend_impl, directed_keys);
        const order = (try argsort(backend_impl, keys, dimension, value_descriptor.element_type, value_dims)) orelse return null;
        const directed_order = (try reverseIfDescending(backend_impl, order, dimension, value_dims, direction)) orelse return null;
        errdefer destroyBuffer(backend_impl, directed_order);
        const sorted_values = (try takeAlongAxis(backend_impl, values, directed_order, dimension, value_dims)) orelse return null;

        try storeOwnedValueHandle(backend_impl, value_handles, value_owned, key_output_id, directed_keys);
        errdefer value_owned[key_output_id.index] = false;
        try storeOwnedValueHandle(backend_impl, value_handles, value_owned, value_output_id, sorted_values);
        destroyBuffer(backend_impl, directed_order);
        if (release_inputs) {
            const released = releaseDeadInputs(backend_impl, &executable.program, value_handles, value_owned, node.inputs, instruction_index);
            if (released != 0) recordReleasedIntermediateValues(executable, released);
        }
        return {};
    }
    if (instruction.kind == .top_k and instruction.inputs.len == 1 and instruction.outputs.len == 2) {
        const input_id = instruction.inputs[0];
        const input = value_handles[input_id.index] orelse return error.CommandSubmissionFailed;
        const input_descriptor = plan.values[input_id.index].descriptor;
        if (input_descriptor.dims.len == 0) return null;
        const axis: i64 = @intCast(input_descriptor.dims.len - 1);
        const k = instruction.top_k_k orelse return null;
        const values_id = instruction.outputs[0];
        const indices_id = instruction.outputs[1];
        if (values_id.index >= plan.values.len or indices_id.index >= plan.values.len) return error.CommandSubmissionFailed;
        const values_descriptor = plan.values[values_id.index].descriptor;
        const indices_descriptor = plan.values[indices_id.index].descriptor;

        const starts = try allocator.alloc(i64, input_descriptor.dims.len);
        defer allocator.free(starts);
        const limits = try allocator.dupe(i64, input_descriptor.dims);
        defer allocator.free(limits);
        const strides = try allocator.alloc(i64, input_descriptor.dims.len);
        defer allocator.free(strides);
        @memset(starts, 0);
        @memset(strides, 1);
        limits[limits.len - 1] = k;

        const sorted_values = (try sort(backend_impl, input, axis, input_descriptor.dims)) orelse return null;
        const descending_values = (try reverseIfDescending(backend_impl, sorted_values, axis, input_descriptor.dims, .gt)) orelse return null;
        errdefer destroyBuffer(backend_impl, descending_values);
        const top_values = (try slice(backend_impl, descending_values, starts, limits, strides, values_descriptor.dims)) orelse return null;
        destroyBuffer(backend_impl, descending_values);

        const sorted_indices = (try argsort(backend_impl, input, axis, indices_descriptor.element_type, input_descriptor.dims)) orelse return null;
        const descending_indices = (try reverseIfDescending(backend_impl, sorted_indices, axis, input_descriptor.dims, .gt)) orelse return null;
        errdefer destroyBuffer(backend_impl, descending_indices);
        const top_indices = (try slice(backend_impl, descending_indices, starts, limits, strides, indices_descriptor.dims)) orelse return null;
        destroyBuffer(backend_impl, descending_indices);

        try storeOwnedValueHandle(backend_impl, value_handles, value_owned, values_id, top_values);
        errdefer value_owned[values_id.index] = false;
        try storeOwnedValueHandle(backend_impl, value_handles, value_owned, indices_id, top_indices);
        if (release_inputs) {
            const released = releaseDeadInputs(backend_impl, &executable.program, value_handles, value_owned, node.inputs, instruction_index);
            if (released != 0) recordReleasedIntermediateValues(executable, released);
        }
        return {};
    }
    if (instruction.kind == .reduce_window_max and instruction.inputs.len == 2 and instruction.outputs.len == 2) {
        const values_id = instruction.inputs[0];
        const indices_id = instruction.inputs[1];
        const values = value_handles[values_id.index] orelse return error.CommandSubmissionFailed;
        const indices = value_handles[indices_id.index] orelse return error.CommandSubmissionFailed;
        const values_output_id = instruction.outputs[0];
        const indices_output_id = instruction.outputs[1];
        if (values_output_id.index >= plan.values.len or indices_output_id.index >= plan.values.len) return error.CommandSubmissionFailed;
        const output_dims = plan.values[values_output_id.index].descriptor.dims;
        const result = (try backend_impl.reduceWindowMaxWithIndices(
            values,
            indices,
            instruction.window_dimensions orelse return null,
            instruction.window_strides orelse return null,
            instruction.base_dilations orelse return null,
            instruction.window_dilations orelse return null,
            instruction.edge_padding_low orelse return null,
            instruction.edge_padding_high orelse return null,
            output_dims,
        )) orelse return null;
        errdefer destroyBuffer(backend_impl, result.values);
        errdefer destroyBuffer(backend_impl, result.indices);
        try storeOwnedValueHandle(backend_impl, value_handles, value_owned, values_output_id, result.values);
        errdefer value_owned[values_output_id.index] = false;
        try storeOwnedValueHandle(backend_impl, value_handles, value_owned, indices_output_id, result.indices);
        if (release_inputs) {
            const released = releaseDeadInputs(backend_impl, &executable.program, value_handles, value_owned, node.inputs, instruction_index);
            if (released != 0) recordReleasedIntermediateValues(executable, released);
        }
        return {};
    }
    if (instruction.kind == .reduce_max and instruction.inputs.len == 2 and instruction.outputs.len == 2) {
        const values_id = instruction.inputs[0];
        const indices_id = instruction.inputs[1];
        const values = value_handles[values_id.index] orelse return error.CommandSubmissionFailed;
        const indices = value_handles[indices_id.index] orelse return error.CommandSubmissionFailed;
        const values_output_id = instruction.outputs[0];
        const indices_output_id = instruction.outputs[1];
        if (values_output_id.index >= plan.values.len or indices_output_id.index >= plan.values.len) return error.CommandSubmissionFailed;
        const output_dims = plan.values[values_output_id.index].descriptor.dims;
        const result = (try backend_impl.reduceMaxWithIndices(
            values,
            indices,
            instruction.reduce_dimensions orelse return null,
            output_dims,
        )) orelse return null;
        errdefer destroyBuffer(backend_impl, result.values);
        errdefer destroyBuffer(backend_impl, result.indices);
        try storeOwnedValueHandle(backend_impl, value_handles, value_owned, values_output_id, result.values);
        errdefer value_owned[values_output_id.index] = false;
        try storeOwnedValueHandle(backend_impl, value_handles, value_owned, indices_output_id, result.indices);
        if (release_inputs) {
            const released = releaseDeadInputs(backend_impl, &executable.program, value_handles, value_owned, node.inputs, instruction_index);
            if (released != 0) recordReleasedIntermediateValues(executable, released);
        }
        return {};
    }
    if (instruction.kind == .rng_bit_generator and instruction.inputs.len == 1 and instruction.outputs.len == 2) {
        const state_id = instruction.inputs[0];
        const state = value_handles[state_id.index] orelse return error.CommandSubmissionFailed;
        const state_output_id = instruction.outputs[0];
        const bits_output_id = instruction.outputs[1];
        if (state_output_id.index >= plan.values.len or bits_output_id.index >= plan.values.len) return error.CommandSubmissionFailed;
        const bits_descriptor = plan.values[bits_output_id.index].descriptor;
        const result = (try backend_impl.rngBitGenerator(
            state,
            bits_descriptor.element_type,
            bits_descriptor.dims,
        )) orelse return null;
        errdefer destroyBuffer(backend_impl, result.state);
        errdefer destroyBuffer(backend_impl, result.bits);
        try storeOwnedValueHandle(backend_impl, value_handles, value_owned, state_output_id, result.state);
        errdefer value_owned[state_output_id.index] = false;
        try storeOwnedValueHandle(backend_impl, value_handles, value_owned, bits_output_id, result.bits);
        if (release_inputs) {
            const released = releaseDeadInputs(backend_impl, &executable.program, value_handles, value_owned, node.inputs, instruction_index);
            if (released != 0) recordReleasedIntermediateValues(executable, released);
        }
        return {};
    }
    if (instruction.kind == .constant) {
        if (instruction.outputs.len != 1) return null;
        const output_id = instruction.outputs[0];
        const cached = executable.constant_handles[constantIndex(plan.instructions.len, device_index, instruction_index)] orelse return null;
        try storeBorrowedValueHandle(backend_impl, value_handles, value_owned, output_id, cached);
        recordBorrowedConstantNode(executable);
        return {};
    }
    if (instruction.kind == .optimization_barrier) {
        if (instruction.inputs.len != instruction.outputs.len) return null;
        var stored_outputs: usize = 0;
        errdefer {
            for (instruction.outputs[0..stored_outputs]) |output_id| {
                if (output_id.index < value_handles.len and value_owned[output_id.index]) {
                    if (value_handles[output_id.index]) |handle| destroyBuffer(backend_impl, handle);
                    value_handles[output_id.index] = null;
                    value_owned[output_id.index] = false;
                }
            }
        }
        for (instruction.inputs, instruction.outputs) |input_id, output_id| {
            const input = value_handles[input_id.index] orelse return error.CommandSubmissionFailed;
            const cloned = (try cloneBuffer(backend_impl, input)) orelse return null;
            try storeOwnedValueHandle(backend_impl, value_handles, value_owned, output_id, cloned);
            stored_outputs += 1;
        }
        if (release_inputs) {
            const released = releaseDeadInputs(backend_impl, &executable.program, value_handles, value_owned, node.inputs, instruction_index);
            if (released != 0) recordReleasedIntermediateValues(executable, released);
        }
        return {};
    }
    if (instruction.kind == .tuple) {
        if (instruction.outputs.len != 1) return null;
        const output_id = instruction.outputs[0];
        if (output_id.index >= plan.values.len or plan.values[output_id.index].storage != .tuple) return null;
        if (release_inputs) {
            const released = releaseDeadInputs(backend_impl, &executable.program, value_handles, value_owned, node.inputs, instruction_index);
            if (released != 0) recordReleasedIntermediateValues(executable, released);
        }
        return {};
    }
    if (instruction.outputs.len != 1) return null;
    const output_id = instruction.outputs[0];
    if (output_id.index >= value_handles.len) return error.CommandSubmissionFailed;
    const output_descriptor = plan.values[output_id.index].descriptor;
    const output_dims = instruction.dims orelse output_descriptor.dims;
    const next = switch (instruction.kind) {
        .copy_arg0, .reduce_precision => blk: {
            const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
            break :blk (try cloneBuffer(backend_impl, input)) orelse return null;
        },
        .complex => blk: {
            const real = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
            const imag = value_handles[instruction.inputs[1].index] orelse return error.CommandSubmissionFailed;
            break :blk (try complex(backend_impl, real, imag, output_dims)) orelse return null;
        },
        .real => blk: {
            const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
            break :blk (try realPart(backend_impl, input, output_dims)) orelse return null;
        },
        .imag => blk: {
            const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
            break :blk (try imagPart(backend_impl, input, output_dims)) orelse return null;
        },
        .custom_call => (try executeRegisteredCustomCall(backend_impl, instruction, value_handles)) orelse return null,
        .get_tuple_element => blk: {
            if (instruction.inputs.len != 1) return null;
            const tuple_id = instruction.inputs[0];
            if (tuple_id.index >= plan.values.len) return error.CommandSubmissionFailed;
            const tuple_value = plan.values[tuple_id.index];
            if (tuple_value.storage != .tuple) return null;
            const tuple_index = instruction.tuple_index orelse return null;
            if (tuple_index < 0 or tuple_index >= @as(i64, @intCast(tuple_value.elements.len))) return null;
            const element_id = tuple_value.elements[@intCast(tuple_index)];
            if (element_id.index >= value_handles.len) return error.CommandSubmissionFailed;
            const element = value_handles[element_id.index] orelse return error.CommandSubmissionFailed;
            break :blk (try cloneBuffer(backend_impl, element)) orelse return null;
        },
        .iota => blk: {
            break :blk (try iota(
                backend_impl,
                executable.device_local_hardware_ids[device_index],
                output_descriptor.element_type,
                output_dims,
                instruction.iota_dimension orelse return null,
            )) orelse return null;
        },
        .partition_id => blk: {
            const partition_count = if (plan.options.num_partitions <= 0) 1 else plan.options.num_partitions;
            const partition_id_value: u32 = @intCast(device_index % @as(usize, @intCast(partition_count)));
            break :blk (try partitionId(
                backend_impl,
                executable.device_local_hardware_ids[device_index],
                output_descriptor.element_type,
                partition_id_value,
            )) orelse return null;
        },
        .rng => blk: {
            const a = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
            const b = value_handles[instruction.inputs[1].index] orelse return error.CommandSubmissionFailed;
            break :blk (try backend_impl.rng(
                a,
                b,
                instruction.rng_distribution orelse return null,
                output_descriptor.element_type,
                output_dims,
            )) orelse return null;
        },
        .convert => blk: {
            const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
            break :blk (try convert(backend_impl, input, output_descriptor.element_type)) orelse return null;
        },
        .bitcast_convert => blk: {
            const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
            break :blk (try bitcast(backend_impl, input, output_descriptor.element_type, output_dims)) orelse return null;
        },
        .add, .subtract, .multiply, .divide, .maximum, .minimum, .power, .atan2, .remainder, .and_, .or_, .xor, .shift_left, .shift_right_arithmetic, .shift_right_logical => blk: {
            const op = executableBinaryOp(instruction.kind) orelse return null;
            const lhs = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
            const rhs = value_handles[instruction.inputs[1].index] orelse return error.CommandSubmissionFailed;
            break :blk (try binaryWithOutputDims(backend_impl, lhs, rhs, op, output_dims)) orelse return null;
        },
        .negate, .exp, .expm1, .tanh, .sqrt, .rsqrt, .abs, .cbrt, .ceil, .floor, .log, .log1p, .logistic, .sine, .cosine, .not_, .sign, .is_finite, .round_nearest_afz, .round_nearest_even, .popcnt, .count_leading_zeros => blk: {
            const op = executableUnaryOp(instruction.kind) orelse return null;
            const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
            break :blk (try unary(backend_impl, input, op)) orelse return null;
        },
        .reshape => blk: {
            const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
            break :blk (try reshape(backend_impl, input, output_dims)) orelse return null;
        },
        .transpose => blk: {
            const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
            break :blk (try transpose(backend_impl, input, instruction.permutation orelse return null)) orelse return null;
        },
        .broadcast_in_dim => blk: {
            const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
            break :blk (try broadcastInDim(backend_impl, input, instruction.broadcast_dimensions orelse return null, output_dims)) orelse return null;
        },
        .slice => blk: {
            const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
            break :blk (try slice(
                backend_impl,
                input,
                instruction.start_indices orelse return null,
                instruction.limit_indices orelse return null,
                instruction.strides orelse return null,
                output_dims,
            )) orelse return null;
        },
        .dynamic_slice => blk: {
            const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
            const starts = try startHandles(allocator, value_handles, instruction.inputs[1..]);
            defer allocator.free(starts);
            break :blk (try dynamicSlice(backend_impl, input, starts, instruction.slice_sizes orelse return null, output_dims)) orelse return null;
        },
        .dynamic_update_slice => blk: {
            const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
            const update = value_handles[instruction.inputs[1].index] orelse return error.CommandSubmissionFailed;
            const starts = try startHandles(allocator, value_handles, instruction.inputs[2..]);
            defer allocator.free(starts);
            break :blk (try dynamicUpdateSlice(backend_impl, input, update, starts, output_dims)) orelse return null;
        },
        .pad => blk: {
            const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
            const padding_value = value_handles[instruction.inputs[1].index] orelse return error.CommandSubmissionFailed;
            break :blk (try pad(
                backend_impl,
                input,
                padding_value,
                instruction.edge_padding_low orelse return null,
                instruction.edge_padding_high orelse return null,
                instruction.interior_padding orelse return null,
                output_dims,
            )) orelse return null;
        },
        .reverse => blk: {
            const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
            break :blk (try reverse(backend_impl, input, instruction.dimensions orelse &.{}, output_dims)) orelse return null;
        },
        .concatenate => blk: {
            const lhs = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
            const rhs = value_handles[instruction.inputs[1].index] orelse return error.CommandSubmissionFailed;
            break :blk (try concatenate(backend_impl, lhs, rhs, instruction.dimension orelse return null, output_dims)) orelse return null;
        },
        .gather => blk: {
            const operand = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
            const indices = value_handles[instruction.inputs[1].index] orelse return error.CommandSubmissionFailed;
            break :blk (try gather(
                backend_impl,
                operand,
                indices,
                instruction.start_index_map orelse return null,
                instruction.collapsed_slice_dims orelse &.{},
                instruction.operand_batching_dims orelse &.{},
                instruction.start_indices_batching_dims orelse &.{},
                instruction.index_vector_dim orelse 0,
                instruction.slice_sizes orelse return null,
                instruction.offset_dims orelse &.{},
                output_dims,
            )) orelse return null;
        },
        .scatter => blk: {
            const operand = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
            const indices = value_handles[instruction.inputs[1].index] orelse return error.CommandSubmissionFailed;
            const updates = value_handles[instruction.inputs[2].index] orelse return error.CommandSubmissionFailed;
            if (supportedScatterAxis(instruction)) |scatter_axis| {
                break :blk (try scatterAxis(
                    backend_impl,
                    operand,
                    indices,
                    updates,
                    scatter_axis,
                    instruction.index_vector_dim orelse 0,
                    instruction.scatter_update_kind orelse .set,
                    output_dims,
                )) orelse return null;
            }
            break :blk (try scatter(
                backend_impl,
                operand,
                indices,
                updates,
                instruction.scatter_dims_to_operand_dims orelse return null,
                instruction.inserted_window_dims orelse return null,
                instruction.update_window_dims orelse &.{},
                instruction.input_batching_dims orelse &.{},
                instruction.scatter_indices_batching_dims orelse &.{},
                instruction.index_vector_dim orelse 0,
                instruction.scatter_update_kind orelse .set,
                output_dims,
            )) orelse return null;
        },
        .sort => blk: {
            const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
            const sorted = (try sort(backend_impl, input, instruction.dimension orelse return null, output_dims)) orelse return null;
            break :blk (try reverseIfDescending(backend_impl, sorted, instruction.dimension.?, output_dims, instruction.compare_direction orelse .lt)) orelse return null;
        },
        .dot_general => blk: {
            const lhs = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
            const rhs = value_handles[instruction.inputs[1].index] orelse return error.CommandSubmissionFailed;
            break :blk (try dotGeneral(
                backend_impl,
                lhs,
                rhs,
                instruction.lhs_batch_dimensions orelse &.{},
                instruction.rhs_batch_dimensions orelse &.{},
                instruction.lhs_contracting_dimensions orelse &.{},
                instruction.rhs_contracting_dimensions orelse &.{},
                output_dims,
            )) orelse return null;
        },
        .convolution => blk: {
            const lhs = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
            const rhs = value_handles[instruction.inputs[1].index] orelse return error.CommandSubmissionFailed;
            break :blk (try convolution(
                backend_impl,
                lhs,
                rhs,
                instruction.window_strides orelse return null,
                instruction.edge_padding_low orelse return null,
                instruction.edge_padding_high orelse return null,
                instruction.base_dilations orelse return null,
                instruction.window_dilations orelse return null,
                instruction.window_reversal orelse return null,
                instruction.feature_group_count orelse 1,
                output_dims,
            )) orelse return null;
        },
        .cholesky => blk: {
            const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
            break :blk (try cholesky(backend_impl, input, instruction.lower orelse true, output_dims)) orelse return null;
        },
        .triangular_solve => blk: {
            const a = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
            const b = value_handles[instruction.inputs[1].index] orelse return error.CommandSubmissionFailed;
            break :blk (try triangularSolve(
                backend_impl,
                a,
                b,
                instruction.triangular_left_side orelse true,
                instruction.triangular_lower orelse true,
                instruction.triangular_unit_diagonal orelse false,
                instruction.triangular_transpose orelse .no_transpose,
                output_dims,
            )) orelse return null;
        },
        .fft => blk: {
            const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
            break :blk (try fft(backend_impl, input, instruction.fft_kind orelse return null, instruction.dimensions orelse return null, output_dims)) orelse return null;
        },
        .reduce_sum, .reduce_max, .reduce_min, .reduce_and, .reduce_or => blk: {
            const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
            break :blk (try reduce(backend_impl, input, instruction.kind, instruction.reduce_dimensions orelse &.{}, output_dims)) orelse return null;
        },
        .reduce_window_sum, .reduce_window_max => blk: {
            const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
            break :blk (try reduceWindow(
                backend_impl,
                input,
                instruction.kind,
                instruction.window_dimensions orelse return null,
                instruction.window_strides orelse return null,
                instruction.base_dilations orelse return null,
                instruction.window_dilations orelse return null,
                instruction.edge_padding_low orelse return null,
                instruction.edge_padding_high orelse return null,
                output_dims,
            )) orelse return null;
        },
        .compare => blk: {
            const lhs = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
            const rhs = value_handles[instruction.inputs[1].index] orelse return error.CommandSubmissionFailed;
            break :blk (try compare(backend_impl, lhs, rhs, instruction.compare_direction orelse .eq, output_dims)) orelse return null;
        },
        .select => blk: {
            const pred = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
            const on_true = value_handles[instruction.inputs[1].index] orelse return error.CommandSubmissionFailed;
            const on_false = value_handles[instruction.inputs[2].index] orelse return error.CommandSubmissionFailed;
            break :blk (try select(backend_impl, pred, on_true, on_false, output_dims)) orelse return null;
        },
        .clamp => blk: {
            const min = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
            const value = value_handles[instruction.inputs[1].index] orelse return error.CommandSubmissionFailed;
            const max = value_handles[instruction.inputs[2].index] orelse return error.CommandSubmissionFailed;
            break :blk (try clamp(backend_impl, min, value, max, output_dims)) orelse return null;
        },
        else => return null,
    };

    try storeOwnedValueHandle(backend_impl, value_handles, value_owned, output_id, next);
    if (release_inputs) {
        const released = releaseDeadInputs(backend_impl, &executable.program, value_handles, value_owned, node.inputs, instruction_index);
        if (released != 0) recordReleasedIntermediateValues(executable, released);
    }
    return {};
}

fn executeControlFlowNode(
    backend_impl: backend.Backend,
    executable: *CompiledExecutable,
    device_index: usize,
    value_handles: []?backend.BufferHandle,
    value_owned: []bool,
    node: backend.ProgramNode,
    instruction: ir.PlanInstruction,
    release_inputs: bool,
) backend.Error!?void {
    const control_flow_index = node.control_flow orelse return error.CommandSubmissionFailed;
    if (control_flow_index >= executable.program.control_flows.len) return error.CommandSubmissionFailed;
    const control_flow = executable.program.control_flows[control_flow_index];
    switch (control_flow.kind) {
        .while_loop => {
            if (instruction.kind != .while_) return error.CommandSubmissionFailed;
            if (control_flow.condition_subprogram >= executable.program.subprograms.len or
                control_flow.body_subprogram >= executable.program.subprograms.len)
            {
                return error.CommandSubmissionFailed;
            }
            const pattern = matchWhileF32LtAddPattern(
                executable.program.subprograms[control_flow.condition_subprogram],
                executable.program.subprograms[control_flow.body_subprogram],
            ) orelse return null;
            if (instruction.inputs.len != pattern.state_count or instruction.outputs.len != pattern.state_count) return null;
            if (pattern.state_index >= instruction.inputs.len) return error.CommandSubmissionFailed;
            const state_id = instruction.inputs[pattern.state_index];
            if (state_id.index >= value_handles.len) return error.CommandSubmissionFailed;
            const state = value_handles[state_id.index] orelse return error.CommandSubmissionFailed;
            const output_id = instruction.outputs[pattern.state_index];
            if (output_id.index >= executable.plan.values.len) return error.CommandSubmissionFailed;
            const output_descriptor = executable.plan.values[output_id.index].descriptor;
            const limit = try whilePatternOperandHandle(executable, value_handles, instruction, device_index, control_flow_index, pattern.limit, 0);
            const step = try whileStepOperandHandle(
                backend_impl,
                executable,
                value_handles,
                instruction,
                device_index,
                control_flow_index,
                executable.program.subprograms[control_flow.body_subprogram],
                pattern.step,
            );
            defer if (step.owned) destroyBuffer(backend_impl, step.handle);
            const next = (try whileF32CompareAdd(
                backend_impl,
                state,
                limit,
                step.handle,
                pattern.compare_direction,
                pattern.update_op,
                output_descriptor.dims,
                pattern.max_iterations,
            )) orelse return null;
            try storeOwnedValueHandle(backend_impl, value_handles, value_owned, output_id, next);
            var invariant_index: usize = 1;
            invariant_index = 0;
            while (invariant_index < pattern.state_count) : (invariant_index += 1) {
                if (invariant_index == pattern.state_index) continue;
                const input_id = instruction.inputs[invariant_index];
                const invariant_output_id = instruction.outputs[invariant_index];
                if (input_id.index >= value_handles.len) return error.CommandSubmissionFailed;
                const input = value_handles[input_id.index] orelse return error.CommandSubmissionFailed;
                const cloned = (try cloneBuffer(backend_impl, input)) orelse return null;
                try storeOwnedValueHandle(backend_impl, value_handles, value_owned, invariant_output_id, cloned);
            }
            if (release_inputs) {
                const released = releaseDeadInputs(backend_impl, &executable.program, value_handles, value_owned, node.inputs, node.instruction_index);
                if (released != 0) recordReleasedIntermediateValues(executable, released);
            }
            return {};
        },
    }
}

fn whilePatternOperandHandle(
    executable: *CompiledExecutable,
    value_handles: []const ?backend.BufferHandle,
    instruction: ir.PlanInstruction,
    device_index: usize,
    control_flow_index: usize,
    value: ir.RegionValue,
    constant_slot: usize,
) backend.Error!backend.BufferHandle {
    return switch (value.role) {
        .constant => executable.while_constant_handles[whileConstantIndex(executable.program.control_flows.len, device_index, control_flow_index, constant_slot)] orelse error.CommandSubmissionFailed,
        .argument => blk: {
            const argument_index: usize = value.id.index;
            if (argument_index >= instruction.inputs.len) return error.CommandSubmissionFailed;
            const input_id = instruction.inputs[argument_index];
            if (input_id.index >= value_handles.len) return error.CommandSubmissionFailed;
            break :blk value_handles[input_id.index] orelse error.CommandSubmissionFailed;
        },
        else => error.CommandSubmissionFailed,
    };
}

fn whileStepOperandHandle(
    backend_impl: backend.Backend,
    executable: *CompiledExecutable,
    value_handles: []const ?backend.BufferHandle,
    instruction: ir.PlanInstruction,
    device_index: usize,
    control_flow_index: usize,
    body: backend.ProgramSubprogram,
    operand: WhilePatternOperand,
) backend.Error!WhileOperandHandle {
    if (operand.producer_instruction_index == null) {
        return .{
            .handle = try whilePatternOperandHandle(executable, value_handles, instruction, device_index, control_flow_index, operand.value, 1),
            .owned = false,
        };
    }
    const producer_index = operand.producer_instruction_index.?;
    if (producer_index >= body.instructions.len) return error.CommandSubmissionFailed;
    const producer = body.instructions[producer_index];
    const handle = (try executeLoopInvariantRegionInstruction(
        backend_impl,
        value_handles,
        instruction,
        body,
        producer,
    )) orelse return error.CommandSubmissionFailed;
    return .{ .handle = handle, .owned = true };
}

fn executeLoopInvariantRegionInstruction(
    backend_impl: backend.Backend,
    value_handles: []const ?backend.BufferHandle,
    parent_instruction: ir.PlanInstruction,
    subprogram: backend.ProgramSubprogram,
    instruction: ir.RegionInstruction,
) backend.Error!?backend.BufferHandle {
    if (instruction.outputs.len != 1 or instruction.result_descriptors.len != 1) return null;
    const output_dims = instruction.result_descriptors[0].dims;
    switch (instruction.kind) {
        .add, .subtract, .multiply, .divide, .maximum, .minimum => {
            if (instruction.inputs.len != 2) return null;
            const op = executableBinaryOp(instruction.kind) orelse return null;
            const lhs = try loopInvariantRegionOperandHandle(value_handles, parent_instruction, subprogram, instruction.inputs[0]);
            const rhs = try loopInvariantRegionOperandHandle(value_handles, parent_instruction, subprogram, instruction.inputs[1]);
            return (try binaryWithOutputDims(backend_impl, lhs, rhs, op, output_dims)) orelse return null;
        },
        else => return null,
    }
}

fn loopInvariantRegionOperandHandle(
    value_handles: []const ?backend.BufferHandle,
    parent_instruction: ir.PlanInstruction,
    subprogram: backend.ProgramSubprogram,
    value_id: ir.RegionValueId,
) backend.Error!backend.BufferHandle {
    const value = regionValueById(subprogram, value_id) orelse return error.CommandSubmissionFailed;
    if (value.role != .argument) return error.CommandSubmissionFailed;
    const argument_index: usize = value.id.index;
    if (argument_index >= parent_instruction.inputs.len) return error.CommandSubmissionFailed;
    const input_id = parent_instruction.inputs[argument_index];
    if (input_id.index >= value_handles.len) return error.CommandSubmissionFailed;
    return value_handles[input_id.index] orelse error.CommandSubmissionFailed;
}

fn executableStats(_: backend.Backend, executable_handle: backend.ExecutableHandle) backend.ExecutableStats {
    const executable: *CompiledExecutable = @ptrCast(@alignCast(executable_handle));
    lockExecutableStats(executable);
    defer executable.stats_mutex.unlock();
    return executable.stats;
}

fn recordSuccessfulExecute(executable: *CompiledExecutable, device_index: usize, local_hardware_id: i32) void {
    lockExecutableStats(executable);
    defer executable.stats_mutex.unlock();
    executable.stats.execute_count += 1;
    executable.stats.last_execute_device_index = device_index;
    executable.stats.last_execute_local_hardware_id = local_hardware_id;
}

fn recordFusionGroupExecute(executable: *CompiledExecutable) void {
    lockExecutableStats(executable);
    defer executable.stats_mutex.unlock();
    executable.stats.fusion_group_execute_count += 1;
}

fn recordCompiledProgramExecute(executable: *CompiledExecutable, output_count: usize) void {
    lockExecutableStats(executable);
    defer executable.stats_mutex.unlock();
    executable.stats.compiled_program_execute_count += 1;
    executable.stats.compiled_program_output_count += output_count;
}

fn recordCapturedProgramExecute(executable: *CompiledExecutable, dynamic_count: usize, captured_count: usize) void {
    lockExecutableStats(executable);
    defer executable.stats_mutex.unlock();
    executable.stats.captured_program_execute_count += 1;
    executable.stats.captured_program_dynamic_input_count += dynamic_count;
    executable.stats.captured_program_captured_input_count += captured_count;
}

fn recordMaterializationEval(executable: *CompiledExecutable, buffer_count: usize) void {
    lockExecutableStats(executable);
    defer executable.stats_mutex.unlock();
    executable.stats.materialization_eval_count += 1;
    executable.stats.materialization_eval_buffer_count += buffer_count;
}

fn recordReleasedIntermediateValues(executable: *CompiledExecutable, count: usize) void {
    lockExecutableStats(executable);
    defer executable.stats_mutex.unlock();
    executable.stats.released_intermediate_count += count;
}

fn recordBorrowedConstantNode(executable: *CompiledExecutable) void {
    lockExecutableStats(executable);
    defer executable.stats_mutex.unlock();
    executable.stats.borrowed_constant_nodes += 1;
}

fn recordExecuteProfile(executable: *CompiledExecutable, profile: ExecuteProfile) void {
    lockExecutableStats(executable);
    defer executable.stats_mutex.unlock();
    executable.stats.execute_wall_us_total +|= profile.wall_us;
    executable.stats.execute_wall_us_peak = @max(executable.stats.execute_wall_us_peak, profile.wall_us);
    executable.stats.schedule_us_total +|= profile.schedule_us;
    executable.stats.schedule_us_peak = @max(executable.stats.schedule_us_peak, profile.schedule_peak_us);
    executable.stats.node_us_total +|= profile.node_us;
    executable.stats.node_us_peak = @max(executable.stats.node_us_peak, profile.node_peak_us);
    executable.stats.fusion_group_us_total +|= profile.fusion_group_us;
    executable.stats.fusion_group_us_peak = @max(executable.stats.fusion_group_us_peak, profile.fusion_group_peak_us);
    executable.stats.materialization_eval_us_total +|= profile.materialization_eval_us;
    executable.stats.materialization_eval_us_peak = @max(executable.stats.materialization_eval_us_peak, profile.materialization_eval_peak_us);
    executable.stats.output_clone_us_total +|= profile.output_clone_us;
    executable.stats.output_clone_us_peak = @max(executable.stats.output_clone_us_peak, profile.output_clone_peak_us);
    executable.stats.compiled_program_us_total +|= profile.compiled_program_us;
    executable.stats.compiled_program_us_peak = @max(executable.stats.compiled_program_us_peak, profile.compiled_program_peak_us);
}

fn writeExecuteProfile(executable: *const CompiledExecutable, device_index: usize, argument_count: usize, output_count: usize, profile: ExecuteProfile) void {
    std.debug.print(
        "pjrtx_profile event=backend_execute executable=0x{x} device={d} args={d} outputs={d} schedule_items={d} nodes={d} fusion_groups={d} materialization_boundaries={d} wall_us={d} schedule_us={d} schedule_peak_us={d} node_us={d} node_peak_us={d} fusion_us={d} fusion_peak_us={d} materialization_us={d} materialization_peak_us={d} compiled_program_us={d} compiled_program_peak_us={d} output_clone_us={d}\n",
        .{
            @intFromPtr(executable),
            device_index,
            argument_count,
            output_count,
            executable.program.schedule.len,
            executable.program.nodes.len,
            executable.program.fusion_groups.len,
            executable.program.materialization_boundaries.len,
            profile.wall_us,
            profile.schedule_us,
            profile.schedule_peak_us,
            profile.node_us,
            profile.node_peak_us,
            profile.fusion_group_us,
            profile.fusion_group_peak_us,
            profile.materialization_eval_us,
            profile.materialization_eval_peak_us,
            profile.compiled_program_us,
            profile.compiled_program_peak_us,
            profile.output_clone_us,
        },
    );
}

fn writeScheduleProfile(executable: *const CompiledExecutable, schedule_index: usize, item: backend.ProgramScheduleItem, elapsed_us: u64) void {
    std.debug.print(
        "pjrtx_profile event=backend_schedule_item executable=0x{x} schedule_index={d} kind={s} index={d} count={d} elapsed_us={d}",
        .{ @intFromPtr(executable), schedule_index, @tagName(item.kind), item.index, item.count, elapsed_us },
    );
    switch (item.kind) {
        .node => {
            if (item.index < executable.program.nodes.len) {
                const node = executable.program.nodes[item.index];
                std.debug.print(" node_kind={s} instruction={d}", .{ @tagName(node.kind), node.instruction_index });
                if (node.instruction_index < executable.plan.instructions.len) {
                    const instruction = executable.plan.instructions[node.instruction_index];
                    std.debug.print(" op={s}", .{@tagName(instruction.kind)});
                }
            }
        },
        .fusion_group => {
            if (item.index < executable.program.fusion_groups.len) {
                const group = executable.program.fusion_groups[item.index];
                std.debug.print(" first_node={d} last_node={d} node_count={d}", .{ group.first_node, group.last_node, group.node_count });
                std.debug.print(" ops=\"", .{});
                for (group.node_indices, 0..) |group_node_index, op_index| {
                    if (op_index != 0) std.debug.print(",", .{});
                    if (group_node_index >= executable.program.nodes.len) {
                        std.debug.print("?", .{});
                        continue;
                    }
                    const group_node = executable.program.nodes[group_node_index];
                    if (group_node.instruction_index >= executable.plan.instructions.len) {
                        std.debug.print("?", .{});
                        continue;
                    }
                    std.debug.print("{s}", .{@tagName(executable.plan.instructions[group_node.instruction_index].kind)});
                }
                std.debug.print("\"", .{});
            }
        },
        .materialization_boundary => {},
    }
    std.debug.print("\n", .{});
}

fn lockExecutableStats(executable: *CompiledExecutable) void {
    while (!executable.stats_mutex.tryLock()) std.atomic.spinLoopHint();
}

fn lockArgumentCapture(executable: *CompiledExecutable) void {
    while (!executable.argument_capture_mutex.tryLock()) std.atomic.spinLoopHint();
}

fn destroyExecutable(backend_impl: backend.Backend, executable_handle: backend.ExecutableHandle) void {
    const executable: *CompiledExecutable = @ptrCast(@alignCast(executable_handle));
    destroyCompiledPrograms(executable.compiled_program_handles);
    destroyArgumentCaptureStates(executable.allocator, executable.argument_capture_states);
    executable.program.deinit();
    destroyConstantHandles(backend_impl, executable.constant_handles);
    destroyConstantHandles(backend_impl, executable.while_constant_handles);
    executable.allocator.free(executable.compiled_program_handles);
    executable.allocator.free(executable.compiled_program_contexts);
    executable.allocator.free(executable.argument_capture_states);
    executable.allocator.free(executable.constant_handles);
    executable.allocator.free(executable.while_constant_handles);
    executable.allocator.free(executable.device_local_hardware_ids);
    executable.allocator.destroy(executable);
}

fn releaseDeadInputs(
    backend_impl: backend.Backend,
    program: *const backend.Program,
    value_handles: []?backend.BufferHandle,
    value_owned: []bool,
    input_ids: []const ir.ValueId,
    instruction_index: usize,
) usize {
    var released: usize = 0;
    for (input_ids) |input_id| {
        if (input_id.index >= value_handles.len or input_id.index >= program.values.len) continue;
        const value = program.values[input_id.index];
        if (value.is_output) continue;
        if (value.last_use_node != @as(?usize, instruction_index)) continue;
        if (!value_owned[input_id.index]) continue;
        if (value_handles[input_id.index]) |old| destroyBuffer(backend_impl, old);
        value_handles[input_id.index] = null;
        value_owned[input_id.index] = false;
        released += 1;
    }
    return released;
}

fn releaseDeadFusionGroupValues(
    backend_impl: backend.Backend,
    program: *const backend.Program,
    value_handles: []?backend.BufferHandle,
    value_owned: []bool,
    group: backend.FusionGroup,
) usize {
    var released: usize = 0;
    for (group.node_indices) |node_index| {
        if (node_index >= program.nodes.len) continue;
        const node = program.nodes[node_index];
        for (node.inputs) |input_id| {
            if (input_id.index >= value_handles.len or input_id.index >= program.values.len) continue;
            const value = program.values[input_id.index];
            if (value.is_output) continue;
            const last_use = value.last_use_node orelse continue;
            if (last_use > group.last_node) continue;
            if (!value_owned[input_id.index]) continue;
            if (value_handles[input_id.index]) |old| destroyBuffer(backend_impl, old);
            value_handles[input_id.index] = null;
            value_owned[input_id.index] = false;
            released += 1;
        }
    }
    return released;
}

fn evalMaterializationBoundaryRange(program: *const backend.Program, value_handles: []const ?backend.BufferHandle, first_boundary: usize, boundary_count: usize) backend.Error!void {
    if (first_boundary > program.materialization_boundaries.len) return error.CommandSubmissionFailed;
    if (boundary_count > program.materialization_boundaries.len - first_boundary) return error.CommandSubmissionFailed;

    var stack_handles: [8]backend.BufferHandle = undefined;
    if (boundary_count <= stack_handles.len) {
        var count: usize = 0;
        for (program.materialization_boundaries[first_boundary..][0..boundary_count]) |boundary| {
            const value_index = boundary.value_id.index;
            if (value_index >= value_handles.len) {
                traceMaterializationFailure("out_of_range", value_index, boundary.reason);
                return error.CommandSubmissionFailed;
            }
            stack_handles[count] = value_handles[value_index] orelse {
                traceMaterializationFailure("missing_handle", value_index, boundary.reason);
                return error.CommandSubmissionFailed;
            };
            count += 1;
        }
        evalBuffers(stack_handles[0..count]) catch {
            for (stack_handles[0..count]) |handle| {
                evalBuffer(handle) catch |err| {
                    traceMaterializationFailure(@errorName(err), std.math.maxInt(usize), .backend_requirement);
                    return err;
                };
            }
        };
        return;
    }

    for (program.materialization_boundaries[first_boundary..][0..boundary_count]) |boundary| {
        const value_index = boundary.value_id.index;
        if (value_index >= value_handles.len) {
            traceMaterializationFailure("out_of_range", value_index, boundary.reason);
            return error.CommandSubmissionFailed;
        }
        const handle = value_handles[value_index] orelse {
            traceMaterializationFailure("missing_handle", value_index, boundary.reason);
            return error.CommandSubmissionFailed;
        };
        evalBuffer(handle) catch |err| {
            traceMaterializationFailure(@errorName(err), value_index, boundary.reason);
            return err;
        };
    }
}

fn traceMaterializationFailure(detail: []const u8, value_index: usize, reason: backend.MaterializationReason) void {
    if (std.c.getenv("PJRTX_TRACE") == null) return;
    std.debug.print("pjrtx_trace event=materialization_error detail={s} value={d} reason={s}\n", .{ detail, value_index, @tagName(reason) });
}

fn destroyConstantHandles(backend_impl: backend.Backend, constant_handles: []?backend.BufferHandle) void {
    for (constant_handles) |maybe_handle| {
        if (maybe_handle) |handle| destroyBuffer(backend_impl, handle);
    }
}

fn destroyCompiledPrograms(handles: []?mlx_call.ProgramHandle) void {
    for (handles) |maybe_handle| {
        if (maybe_handle) |handle| mlx_call.programDestroy(handle);
    }
}

fn destroyArgumentCaptureStates(allocator: std.mem.Allocator, states: []ArgumentCaptureState) void {
    for (states) |*state| resetArgumentCaptureState(allocator, state);
}

fn constantIndex(instruction_count: usize, device_index: usize, instruction_index: usize) usize {
    return executable_mod.constantIndex(instruction_count, device_index, instruction_index);
}

fn whileConstantIndex(control_flow_count: usize, device_index: usize, control_flow_index: usize, constant_index: usize) usize {
    return executable_mod.whileConstantIndex(control_flow_count, device_index, control_flow_index, constant_index);
}

fn executableSupportsInstruction(kind_: ir.PlanInstructionKind) bool {
    return switch (kind_) {
        .constant,
        .iota,
        .partition_id,
        .copy_arg0,
        .custom_call,
        .optimization_barrier,
        .reduce_precision,
        .convert,
        .bitcast_convert,
        .add,
        .subtract,
        .multiply,
        .divide,
        .maximum,
        .minimum,
        .power,
        .atan2,
        .remainder,
        .and_,
        .or_,
        .xor,
        .shift_left,
        .shift_right_arithmetic,
        .shift_right_logical,
        .negate,
        .exp,
        .expm1,
        .tanh,
        .sqrt,
        .rsqrt,
        .abs,
        .cbrt,
        .ceil,
        .floor,
        .log,
        .log1p,
        .logistic,
        .sine,
        .cosine,
        .not_,
        .sign,
        .is_finite,
        .round_nearest_afz,
        .round_nearest_even,
        .popcnt,
        .count_leading_zeros,
        .complex,
        .real,
        .imag,
        .reshape,
        .transpose,
        .broadcast_in_dim,
        .slice,
        .dynamic_slice,
        .dynamic_update_slice,
        .pad,
        .reverse,
        .concatenate,
        .gather,
        .scatter,
        .tuple,
        .get_tuple_element,
        .sort,
        .top_k,
        .dot_general,
        .convolution,
        .cholesky,
        .triangular_solve,
        .fft,
        .rng,
        .rng_bit_generator,
        .while_,
        .reduce_sum,
        .reduce_max,
        .reduce_min,
        .reduce_and,
        .reduce_or,
        .reduce_window_sum,
        .reduce_window_max,
        .compare,
        .select,
        .clamp,
        => true,
        else => false,
    };
}

fn executableLoweringIssue(plan: *const ir.ExecutablePlan, device_local_hardware_ids: []const i32) ?LoweringIssue {
    if (device_local_hardware_ids.len == 0) return .{
        .detail = "backend executable requires at least one device",
        .feature = "mlx-device-assignment",
    };
    for (plan.output_ids) |output_id| {
        if (output_id.index >= plan.values.len) return .{
            .value_id = output_id,
            .detail = "plan output value is outside the executable value table",
            .feature = "mlx-executable-values",
        };
        if (plan.values[output_id.index].storage != .tensor and
            !(plan.values[output_id.index].storage == .complex_pair and plan.values[output_id.index].descriptor.element_type == .c64))
            return .{
                .value_id = output_id,
                .detail = "MLX executable PJRT outputs must be tensor values",
                .feature = "mlx-structured-output",
            };
    }
    for (plan.instructions, 0..) |instruction, instruction_index| {
        if (!executableSupportsInstruction(instruction.kind)) return .{
            .instruction_index = instruction_index,
            .op = instruction.kind,
            .detail = "operation is not supported by the MLX backend executable",
        };
        const valid_output_count = instruction.outputs.len == 1 or
            ((instruction.kind == .sort or instruction.kind == .top_k) and instruction.outputs.len == 2) or
            (instruction.kind == .reduce_max and instruction.inputs.len == 2 and instruction.outputs.len == 2) or
            (instruction.kind == .reduce_window_max and instruction.inputs.len == 2 and instruction.outputs.len == 2) or
            (instruction.kind == .rng_bit_generator and instruction.inputs.len == 1 and instruction.outputs.len == 2) or
            (instruction.kind == .optimization_barrier and instruction.outputs.len == instruction.inputs.len) or
            (instruction.kind == .while_ and instruction.outputs.len != 0 and instruction.outputs.len == instruction.inputs.len);
        if (!valid_output_count) return .{
            .instruction_index = instruction_index,
            .op = instruction.kind,
            .detail = "MLX executable lowering requires one output per instruction except two-output sort/top_k/reduce_max/reduce_window_max/rng_bit_generator",
            .feature = "mlx-executable-values",
        };
        for (instruction.outputs) |output_id| {
            if (output_id.index >= plan.values.len) return .{
                .instruction_index = instruction_index,
                .value_id = output_id,
                .op = instruction.kind,
                .detail = "instruction output value is outside the executable value table",
                .feature = "mlx-executable-values",
            };
        }
        if (instructionLoweringIssue(plan, instruction, instruction_index, instruction.outputs[0])) |issue| return issue;
    }
    return null;
}

fn instructionLoweringIssue(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?LoweringIssue {
    const output_descriptor = plan.values[output_id.index].descriptor;
    return switch (instruction.kind) {
        .constant => if (instruction.literal == null) .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "constant lowering requires an embedded literal",
            .feature = "mlx-constant-literal",
        } else null,
        .custom_call => validateCustomCallLowering(plan, instruction, instruction_index, output_id),
        .fft => validateFftLowering(plan, instruction, instruction_index, output_id),
        .optimization_barrier => validateOptimizationBarrierLowering(plan, instruction, instruction_index),
        .iota => blk: {
            const dim = instruction.iota_dimension orelse break :blk .{
                .instruction_index = instruction_index,
                .value_id = output_id,
                .op = instruction.kind,
                .detail = "iota lowering requires an iota dimension",
                .feature = "mlx-iota",
            };
            if (dim < 0 or dim >= @as(i64, @intCast(output_descriptor.dims.len))) break :blk .{
                .instruction_index = instruction_index,
                .value_id = output_id,
                .op = instruction.kind,
                .detail = "iota dimension is outside the output rank",
                .feature = "mlx-iota",
            };
            break :blk null;
        },
        .partition_id => validatePartitionIdLowering(plan, instruction, instruction_index, output_id),
        .rng => validateRngLowering(plan, instruction, instruction_index, output_id),
        .bitcast_convert => validateBitcastConvertLowering(plan, instruction, instruction_index, output_id),
        .atan2, .and_, .or_, .xor, .shift_left, .shift_right_arithmetic, .shift_right_logical => validateBinaryElementwiseLowering(plan, instruction, instruction_index, output_id),
        .complex => validateComplexLowering(plan, instruction, instruction_index, output_id),
        .real, .imag => validateRealImagLowering(plan, instruction, instruction_index, output_id),
        .expm1, .cbrt, .not_, .is_finite, .round_nearest_afz, .round_nearest_even, .popcnt, .count_leading_zeros => validateUnaryElementwiseLowering(plan, instruction, instruction_index, output_id),
        .transpose => if (instruction.permutation == null) .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "transpose lowering requires a permutation",
            .feature = "mlx-layout",
        } else null,
        .broadcast_in_dim => if (instruction.broadcast_dimensions == null) .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "broadcast_in_dim lowering requires broadcast dimensions",
            .feature = "mlx-shape",
        } else null,
        .slice => if (instruction.start_indices == null or instruction.limit_indices == null or instruction.strides == null) .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "slice lowering requires static starts, limits, and strides",
            .feature = "mlx-slice",
        } else null,
        .dynamic_slice => if (instruction.slice_sizes == null) .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "dynamic_slice lowering requires static slice sizes",
            .feature = "mlx-dynamic-slice",
        } else null,
        .dynamic_update_slice => if (instruction.inputs.len < 2) .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "dynamic_update_slice lowering requires operand and update inputs",
            .feature = "mlx-dynamic-update-slice",
        } else null,
        .pad => validatePadLowering(plan, instruction, instruction_index, output_id),
        .concatenate => if (instruction.dimension == null) .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "concatenate lowering requires a dimension",
            .feature = "mlx-concatenate",
        } else null,
        .gather => validateGatherLowering(plan, instruction, instruction_index, output_id),
        .scatter => validateScatterLowering(plan, instruction, instruction_index, output_id),
        .tuple => validateTupleLowering(plan, instruction, instruction_index, output_id),
        .get_tuple_element => validateGetTupleElementLowering(plan, instruction, instruction_index, output_id),
        .sort => validateSortLowering(instruction, instruction_index, output_id),
        .top_k => validateTopKLowering(plan, instruction, instruction_index, output_id),
        .dot_general => validateDotGeneralLowering(plan, instruction, instruction_index, output_id),
        .convolution => validateConvolutionLowering(plan, instruction, instruction_index, output_id),
        .cholesky => validateCholeskyLowering(plan, instruction, instruction_index, output_id),
        .triangular_solve => validateTriangularSolveLowering(plan, instruction, instruction_index, output_id),
        .reduce_sum, .reduce_max, .reduce_min, .reduce_and, .reduce_or => validateReduceLowering(plan, instruction, instruction_index, output_id),
        .reduce_window_sum, .reduce_window_max => validateReduceWindowLowering(plan, instruction, instruction_index, output_id),
        .rng_bit_generator => validateRngBitGeneratorLowering(plan, instruction, instruction_index, output_id),
        .while_ => validateWhileLowering(plan, instruction, instruction_index, output_id),
        .compare => validateCompareLowering(plan, instruction, instruction_index, output_id),
        .select => validateSelectLowering(plan, instruction, instruction_index, output_id),
        .clamp => validateClampLowering(plan, instruction, instruction_index, output_id),
        else => null,
    };
}

fn validateBitcastConvertLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?LoweringIssue {
    if (instruction.inputs.len != 1 or instruction.outputs.len != 1) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "bitcast_convert lowering requires exactly one input and one output",
        .feature = "mlx-bitcast",
    };
    const input = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[0],
        .op = instruction.kind,
        .detail = "bitcast_convert input is outside the executable value table",
        .feature = "mlx-bitcast",
    };
    const output = plan.values[output_id.index].descriptor;
    if (!buffer_mod.supportsElementType(input.element_type) or !buffer_mod.supportsElementType(output.element_type)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "bitcast_convert lowering requires MLX-supported input and output dtypes",
        .feature = "mlx-bitcast-dtype",
    };
    if (ir.denseByteSize(input.element_type, input.dims) != ir.denseByteSize(output.element_type, output.dims)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "bitcast_convert must preserve dense byte size",
        .feature = "mlx-bitcast-shape",
    };
    return null;
}

fn validatePartitionIdLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?LoweringIssue {
    const output = plan.values[output_id.index].descriptor;
    if (instruction.inputs.len != 0 or instruction.outputs.len != 1) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "partition_id lowering requires no inputs and exactly one output",
        .feature = "mlx-partition-id-arity",
    };
    if (output.dims.len != 0) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "partition_id lowering requires a scalar output",
        .feature = "mlx-partition-id-shape",
    };
    if (output.element_type != .u32 and output.element_type != .s32) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "partition_id lowering supports u32 or s32 scalar outputs",
        .feature = "mlx-partition-id-dtype",
    };
    return null;
}

fn validateRngLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?LoweringIssue {
    if (instruction.inputs.len != 2 or instruction.outputs.len != 1) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "rng lowering requires low/scale inputs and one output",
        .feature = "mlx-rng",
    };
    if (instruction.rng_distribution == null) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "rng lowering requires rng_distribution metadata",
        .feature = "mlx-rng-distribution",
    };
    const a = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[0],
        .op = instruction.kind,
        .detail = "rng first input is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const b = inputDescriptor(plan, instruction, 1) orelse return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[1],
        .op = instruction.kind,
        .detail = "rng second input is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const output = plan.values[output_id.index].descriptor;
    if (a.dims.len != 0 or b.dims.len != 0) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "rng lowering currently requires scalar distribution parameters",
        .feature = "mlx-rng-params",
    };
    if (a.element_type != b.element_type or a.element_type != output.element_type) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "rng distribution parameters and output must use the same dtype",
        .feature = "mlx-rng-dtype",
    };
    if (!isSupportedFloat(output.element_type)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "rng lowering supports MLX floating output dtypes only",
        .feature = "mlx-rng-dtype",
    };
    return null;
}

fn validateCustomCallLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?LoweringIssue {
    const target = instruction.custom_call_target orelse return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "custom_call lowering requires call_target_name",
        .feature = "mlx-custom-call",
    };
    const spec = lookupCustomCall(target) orelse return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "custom_call target has no registered MLX backend implementation",
        .feature = target,
    };
    return switch (spec.kind) {
        .identity => validateIdentityCustomCallLowering(plan, instruction, instruction_index, output_id, target),
        .unary => validateUnaryCustomCallLowering(plan, instruction, instruction_index, output_id, target, spec.unary_op.?),
        .binary => validateBinaryCustomCallLowering(plan, instruction, instruction_index, output_id, target, spec.binary_op.?),
        .metal_kernel_binary_add_f32 => validateMetalKernelBinaryAddF32CustomCallLowering(plan, instruction, instruction_index, output_id, target),
        .scaled_dot_product_attention => validateScaledDotProductAttentionCustomCallLowering(plan, instruction, instruction_index, output_id, target),
    };
}

fn validateIdentityCustomCallLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId, target: []const u8) ?LoweringIssue {
    if (instruction.inputs.len != 1 or instruction.outputs.len != 1) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "identity custom_call lowering requires exactly one input and one output",
        .feature = target,
    };
    const input = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[0],
        .op = instruction.kind,
        .detail = "custom_call input is outside the executable value table",
        .feature = target,
    };
    const output = plan.values[output_id.index].descriptor;
    if (input.element_type != output.element_type or !dimsEqual(input.dims, output.dims)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "identity custom_call output must match input dtype and shape",
        .feature = target,
    };
    return null;
}

fn validateUnaryCustomCallLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId, target: []const u8, op: ir.ElementwiseUnaryOp) ?LoweringIssue {
    if (instruction.inputs.len != 1 or instruction.outputs.len != 1) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "unary custom_call lowering requires exactly one input and one output",
        .feature = target,
    };
    if (!buffer_mod.supportsUnaryOp(op)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "registered unary custom_call uses an MLX unsupported unary op",
        .feature = target,
    };
    const input = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[0],
        .op = instruction.kind,
        .detail = "custom_call input is outside the executable value table",
        .feature = target,
    };
    const output = plan.values[output_id.index].descriptor;
    if (!dimsEqual(input.dims, output.dims)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "unary custom_call lowering requires matching input/output shapes",
        .feature = target,
    };
    switch (op) {
        .is_finite => if (!isSupportedFloat(input.element_type) or output.element_type != .pred) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "is_finite custom_call lowering requires floating input and pred output",
            .feature = target,
        },
        .not_ => if ((!isSupportedInteger(input.element_type) and input.element_type != .pred) or output.element_type != input.element_type) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "not custom_call lowering requires pred or MLX-supported integer dtype",
            .feature = target,
        },
        .expm1, .round_nearest_even => if (!isSupportedFloat(input.element_type) or output.element_type != input.element_type) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "floating unary custom_call lowering requires matching MLX-supported floating dtype",
            .feature = target,
        },
        .cbrt, .round_nearest_afz => if (input.element_type != .f32 or output.element_type != .f32) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "custom Metal unary custom_call lowering currently supports f32 tensors only",
            .feature = target,
        },
        .popcnt, .count_leading_zeros => if (!isSupportedInteger(input.element_type) or output.element_type != input.element_type) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "integer unary custom_call lowering requires matching MLX-supported integer dtype",
            .feature = target,
        },
        else => if (output.element_type != input.element_type) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "unary custom_call output dtype must match input dtype",
            .feature = target,
        },
    }
    return null;
}

fn validateBinaryCustomCallLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId, target: []const u8, op: ir.ElementwiseBinaryOp) ?LoweringIssue {
    if (instruction.inputs.len != 2 or instruction.outputs.len != 1) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "binary custom_call lowering requires exactly two inputs and one output",
        .feature = target,
    };
    const lhs = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[0],
        .op = instruction.kind,
        .detail = "custom_call lhs input is outside the executable value table",
        .feature = target,
    };
    const rhs = inputDescriptor(plan, instruction, 1) orelse return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[1],
        .op = instruction.kind,
        .detail = "custom_call rhs input is outside the executable value table",
        .feature = target,
    };
    const output = plan.values[output_id.index].descriptor;
    if (lhs.element_type != rhs.element_type or lhs.element_type != output.element_type or !dimsEqual(lhs.dims, rhs.dims) or !dimsEqual(lhs.dims, output.dims)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "binary custom_call lowering requires matching input/output dtypes and shapes",
        .feature = target,
    };
    switch (op) {
        .atan2 => if (!isSupportedFloat(lhs.element_type)) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "atan2 custom_call lowering requires an MLX-supported floating dtype",
            .feature = target,
        },
        .and_, .or_, .xor => if (!isSupportedInteger(lhs.element_type) and lhs.element_type != .pred) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "logical/bitwise custom_call lowering requires pred or MLX-supported integer dtype",
            .feature = target,
        },
        .shift_left, .shift_right_arithmetic, .shift_right_logical => if (!isSupportedInteger(lhs.element_type)) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "shift custom_call lowering requires an MLX-supported integer dtype",
            .feature = target,
        },
        else => {},
    }
    return null;
}

fn validateMetalKernelBinaryAddF32CustomCallLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId, target: []const u8) ?LoweringIssue {
    const issue = validateBinaryCustomCallLowering(plan, instruction, instruction_index, output_id, target, .add);
    if (issue) |found| return found;
    const lhs = inputDescriptor(plan, instruction, 0).?;
    const output = plan.values[output_id.index].descriptor;
    if (lhs.element_type != .f32 or output.element_type != .f32) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "built-in Metal custom_call binary add currently requires f32 tensors",
        .feature = target,
    };
    return null;
}

fn validateScaledDotProductAttentionCustomCallLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId, target: []const u8) ?LoweringIssue {
    if (instruction.inputs.len != 4 or instruction.outputs.len != 1) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "scaled-dot-product-attention custom_call requires q, k, v, and token_index inputs with one output",
        .feature = target,
    };
    const q = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[0],
        .op = instruction.kind,
        .detail = "attention q input is outside the executable value table",
        .feature = target,
    };
    const k = inputDescriptor(plan, instruction, 1) orelse return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[1],
        .op = instruction.kind,
        .detail = "attention k input is outside the executable value table",
        .feature = target,
    };
    const v = inputDescriptor(plan, instruction, 2) orelse return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[2],
        .op = instruction.kind,
        .detail = "attention v input is outside the executable value table",
        .feature = target,
    };
    const token_index = inputDescriptor(plan, instruction, 3) orelse return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[3],
        .op = instruction.kind,
        .detail = "attention token_index input is outside the executable value table",
        .feature = target,
    };
    const output = plan.values[output_id.index].descriptor;
    if (!isSupportedFloat(q.element_type) or q.element_type != k.element_type or q.element_type != v.element_type or output.element_type != q.element_type) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "attention custom_call requires matching f16/bf16/f32 q/k/v/output dtypes",
        .feature = target,
    };
    if (q.dims.len != 3 and q.dims.len != 4) return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[0],
        .op = instruction.kind,
        .detail = "attention custom_call supports rank-3 [q,h,hd] or rank-4 [b,q,h,hd] q tensors",
        .feature = target,
    };
    if (k.dims.len != q.dims.len or v.dims.len != q.dims.len or !dimsEqual(k.dims, v.dims) or !dimsEqual(q.dims, output.dims)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "attention custom_call requires k/v matching shape and output shape matching q",
        .feature = target,
    };
    const query_axis: usize = if (q.dims.len == 4) 1 else 0;
    const head_axis: usize = if (q.dims.len == 4) 2 else 1;
    const dim_axis: usize = if (q.dims.len == 4) 3 else 2;
    if (q.dims[dim_axis] != k.dims[dim_axis] or q.dims[head_axis] <= 0 or k.dims[head_axis] <= 0 or @mod(q.dims[head_axis], k.dims[head_axis]) != 0 or q.dims[query_axis] <= 0 or k.dims[query_axis] <= 0) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "attention custom_call requires positive q/k lengths, equal head_dim, and q_heads divisible by kv_heads",
        .feature = target,
    };
    if (q.dims.len == 4 and q.dims[0] != k.dims[0]) return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[1],
        .op = instruction.kind,
        .detail = "attention custom_call requires q and k batch dimensions to match",
        .feature = target,
    };
    if (!isSupportedInteger(token_index.element_type) or token_index.dims.len > 1 or (token_index.dims.len == 1 and token_index.dims[0] != 1)) return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[3],
        .op = instruction.kind,
        .detail = "attention custom_call token_index must be an integer scalar or length-1 tensor",
        .feature = target,
    };
    return null;
}

fn validateOptimizationBarrierLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize) ?LoweringIssue {
    if (instruction.inputs.len == 0 or instruction.inputs.len != instruction.outputs.len) return .{
        .instruction_index = instruction_index,
        .op = instruction.kind,
        .detail = "optimization_barrier lowering requires one output for each input",
        .feature = "mlx-optimization-barrier",
    };
    for (instruction.inputs, instruction.outputs) |input_id, output_id| {
        if (input_id.index >= plan.values.len) return .{
            .instruction_index = instruction_index,
            .value_id = input_id,
            .op = instruction.kind,
            .detail = "optimization_barrier input is outside the executable value table",
            .feature = "mlx-optimization-barrier",
        };
        if (output_id.index >= plan.values.len) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "optimization_barrier output is outside the executable value table",
            .feature = "mlx-optimization-barrier",
        };
        const input = plan.values[input_id.index].descriptor;
        const output = plan.values[output_id.index].descriptor;
        if (input.element_type != output.element_type or !dimsEqual(input.dims, output.dims)) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "optimization_barrier output must match corresponding input dtype and shape",
            .feature = "mlx-optimization-barrier",
        };
    }
    return null;
}

fn validateBinaryElementwiseLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?LoweringIssue {
    const lhs = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 0) instruction.inputs[0] else output_id,
        .op = instruction.kind,
        .detail = "binary operand lhs is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const rhs = inputDescriptor(plan, instruction, 1) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 1) instruction.inputs[1] else output_id,
        .op = instruction.kind,
        .detail = "binary operand rhs is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const output = plan.values[output_id.index].descriptor;
    if (lhs.element_type != rhs.element_type or lhs.element_type != output.element_type or !validElementwiseBroadcast(lhs.dims, rhs.dims, output.dims)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "binary lowering requires matching dtypes and broadcast-compatible input/output shapes",
        .feature = "mlx-elementwise-binary",
    };
    switch (instruction.kind) {
        .atan2 => if (!isSupportedFloat(lhs.element_type)) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "atan2 lowering requires an MLX-supported floating dtype",
            .feature = "mlx-atan2",
        },
        .and_, .or_, .xor => if (!isSupportedInteger(lhs.element_type) and lhs.element_type != .pred) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "logical/bitwise lowering requires pred or MLX-supported integer dtype",
            .feature = "mlx-bitwise",
        },
        .shift_left, .shift_right_arithmetic, .shift_right_logical => if (!isSupportedInteger(lhs.element_type)) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "shift lowering requires an MLX-supported integer dtype",
            .feature = "mlx-shift",
        },
        else => {},
    }
    return null;
}

fn validateUnaryElementwiseLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?LoweringIssue {
    const input = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 0) instruction.inputs[0] else output_id,
        .op = instruction.kind,
        .detail = "unary operand is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const output = plan.values[output_id.index].descriptor;
    if (!dimsEqual(input.dims, output.dims)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "unary lowering requires matching input/output shapes",
        .feature = "mlx-elementwise-unary",
    };
    switch (instruction.kind) {
        .abs => {
            const valid_abs = if (input.element_type == .c64)
                output.element_type == .f32
            else
                output.element_type == input.element_type and (isSupportedFloat(input.element_type) or isSupportedInteger(input.element_type));
            if (!valid_abs) return .{
                .instruction_index = instruction_index,
                .value_id = output_id,
                .op = instruction.kind,
                .detail = "abs lowering requires MLX-supported real/integer input with matching output or c64 input with f32 output",
                .feature = "mlx-abs",
            };
        },
        .expm1, .round_nearest_even => if (!isSupportedFloat(input.element_type) or output.element_type != input.element_type) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "floating unary lowering requires matching MLX-supported floating dtype",
            .feature = "mlx-unary-float",
        },
        .cbrt, .round_nearest_afz => if (input.element_type != .f32 or output.element_type != .f32) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "custom Metal unary lowering currently supports f32 tensors only",
            .feature = "mlx-metal-unary-f32",
        },
        .popcnt, .count_leading_zeros => if (!isSupportedInteger(input.element_type) or output.element_type != input.element_type) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "integer unary lowering requires matching MLX-supported integer dtype",
            .feature = "mlx-integer-unary",
        },
        .is_finite => if (!isSupportedFloat(input.element_type) or output.element_type != .pred) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "is_finite lowering requires floating input and pred output",
            .feature = "mlx-is-finite",
        },
        .not_ => if ((!isSupportedInteger(input.element_type) and input.element_type != .pred) or output.element_type != input.element_type) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "not lowering requires pred or MLX-supported integer dtype",
            .feature = "mlx-not",
        },
        else => {},
    }
    return null;
}

fn validateComplexLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?LoweringIssue {
    const real = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 0) instruction.inputs[0] else output_id,
        .op = instruction.kind,
        .detail = "complex real operand is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const imag = inputDescriptor(plan, instruction, 1) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 1) instruction.inputs[1] else output_id,
        .op = instruction.kind,
        .detail = "complex imaginary operand is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const output = plan.values[output_id.index].descriptor;
    if (real.element_type != .f32 or imag.element_type != .f32 or output.element_type != .c64) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "complex lowering currently supports f32 operands producing c64 tensors",
        .feature = "mlx-complex-dtype",
    };
    if (!dimsEqual(real.dims, imag.dims) or !dimsEqual(real.dims, output.dims)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "complex lowering requires matching real, imaginary, and output shapes",
        .feature = "mlx-complex-shape",
    };
    return null;
}

fn validateRealImagLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?LoweringIssue {
    const input = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 0) instruction.inputs[0] else output_id,
        .op = instruction.kind,
        .detail = "real/imag operand is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const output = plan.values[output_id.index].descriptor;
    if (!dimsEqual(input.dims, output.dims)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "real/imag lowering requires matching input/output shapes",
        .feature = "mlx-real-imag-shape",
    };
    if (input.element_type == .c64 and output.element_type == .f32) return null;
    if (isSupportedFloat(input.element_type) and output.element_type == input.element_type) return null;
    return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "real/imag lowering supports real floating tensors or c64-to-f32 extraction",
        .feature = "mlx-real-imag-dtype",
    };
}

fn validatePadLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?LoweringIssue {
    const low = instruction.edge_padding_low orelse return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "pad lowering requires low edge padding",
        .feature = "mlx-pad",
    };
    const high = instruction.edge_padding_high orelse return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "pad lowering requires high edge padding",
        .feature = "mlx-pad",
    };
    const interior = instruction.interior_padding orelse return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "pad lowering requires interior padding metadata",
        .feature = "mlx-pad",
    };
    if (instruction.inputs.len < 2) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "pad lowering requires operand and scalar padding value inputs",
        .feature = "mlx-pad",
    };
    const input_descriptor = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[0],
        .op = instruction.kind,
        .detail = "pad operand is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const padding_descriptor = inputDescriptor(plan, instruction, 1) orelse return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[1],
        .op = instruction.kind,
        .detail = "pad scalar value is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const output_descriptor = plan.values[output_id.index].descriptor;
    if (padding_descriptor.element_type != input_descriptor.element_type or padding_descriptor.dims.len != 0) return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[1],
        .op = instruction.kind,
        .detail = "pad lowering requires a scalar padding value with the operand dtype",
        .feature = "mlx-pad-scalar",
    };
    if (low.len != input_descriptor.dims.len or high.len != input_descriptor.dims.len or interior.len != input_descriptor.dims.len or output_descriptor.dims.len != input_descriptor.dims.len) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "pad metadata rank does not match operand/output rank",
        .feature = "mlx-pad",
    };
    for (input_descriptor.dims, 0..) |dim, axis| {
        if (low[axis] < 0 or high[axis] < 0 or interior[axis] < 0) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "pad lowering requires non-negative edge and interior padding",
            .feature = "mlx-pad",
        };
        const interior_slots = if (dim > 0) (dim - 1) * interior[axis] else 0;
        if (output_descriptor.dims[axis] != dim + low[axis] + high[axis] + interior_slots) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "pad output shape must equal StableHLO edge plus interior padding formula",
            .feature = "mlx-pad",
        };
    }
    return null;
}

fn validateGatherLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?LoweringIssue {
    const operand = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 0) instruction.inputs[0] else output_id,
        .op = instruction.kind,
        .detail = "gather operand is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const indices = inputDescriptor(plan, instruction, 1) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 1) instruction.inputs[1] else output_id,
        .op = instruction.kind,
        .detail = "gather indices are outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const output = plan.values[output_id.index].descriptor;
    if (operand.element_type != output.element_type or !isSupportedInteger(indices.element_type)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "gather lowering requires matching operand/output dtypes and integer indices",
        .feature = "mlx-gather-types",
    };
    const slice_sizes = instruction.slice_sizes orelse return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "gather lowering requires static slice sizes",
        .feature = "mlx-gather-slice-sizes",
    };
    if (slice_sizes.len != operand.dims.len) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "gather slice size rank must match operand rank",
        .feature = "mlx-gather-slice-sizes",
    };
    if (!validGatherShape(
        operand.dims,
        indices.dims,
        instruction.start_index_map orelse &.{},
        instruction.collapsed_slice_dims orelse &.{},
        instruction.operand_batching_dims orelse &.{},
        instruction.start_indices_batching_dims orelse &.{},
        instruction.index_vector_dim orelse 0,
        slice_sizes,
        instruction.offset_dims orelse &.{},
        output.dims,
    )) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "gather metadata and output shape must match MLX general gather semantics",
        .feature = "mlx-gather-general-shape",
    };
    return null;
}

fn validateScatterLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?LoweringIssue {
    const operand = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 0) instruction.inputs[0] else output_id,
        .op = instruction.kind,
        .detail = "scatter operand is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const indices = inputDescriptor(plan, instruction, 1) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 1) instruction.inputs[1] else output_id,
        .op = instruction.kind,
        .detail = "scatter indices are outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const updates = inputDescriptor(plan, instruction, 2) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 2) instruction.inputs[2] else output_id,
        .op = instruction.kind,
        .detail = "scatter updates are outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const output = plan.values[output_id.index].descriptor;
    if (operand.element_type != updates.element_type or operand.element_type != output.element_type or !dimsEqual(operand.dims, output.dims) or !isSupportedInteger(indices.element_type)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "scatter lowering requires matching operand/update/output dtypes and integer indices",
        .feature = "mlx-scatter-types",
    };
    if (instruction.scatter_update_kind == null) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "scatter combiner must be set or add",
        .feature = "mlx-scatter-combiner",
    };
    if (supportedScatterAxis(instruction)) |axis| {
        if (axis < 0 or axis >= @as(i64, @intCast(operand.dims.len))) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "scatter axis is outside the operand rank",
            .feature = "mlx-scatter-axis",
        };
        if (!scatterUpdateShapeMatchesAxis(operand.dims, indices.dims, updates.dims, instruction.index_vector_dim orelse 0, axis)) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "scatter update shape must match MLX scatter semantics for the scattered axis",
            .feature = "mlx-scatter-update-shape",
        };
        return null;
    }
    if (!validScatterShape(
        operand.dims,
        indices.dims,
        updates.dims,
        instruction.scatter_dims_to_operand_dims orelse &.{},
        instruction.inserted_window_dims orelse &.{},
        instruction.update_window_dims orelse &.{},
        instruction.input_batching_dims orelse &.{},
        instruction.scatter_indices_batching_dims orelse &.{},
        instruction.index_vector_dim orelse 0,
        output.dims,
    )) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "scatter metadata and update shape must match MLX scatter semantics",
        .feature = "mlx-scatter-shape",
    };
    return null;
}

fn validateTupleLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?LoweringIssue {
    const output = plan.values[output_id.index];
    if (output.storage != .tuple) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "tuple lowering requires a tuple storage output value",
        .feature = "mlx-tuple-structural",
    };
    if (output.elements.len != instruction.inputs.len) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "tuple output element list must match tuple operands",
        .feature = "mlx-tuple-structural",
    };
    for (output.elements, instruction.inputs) |element_id, input_id| {
        if (element_id.index >= plan.values.len or input_id.index >= plan.values.len or element_id.index != input_id.index) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "tuple output element list must reference tuple operands in order",
            .feature = "mlx-tuple-structural",
        };
    }
    return null;
}

fn validateGetTupleElementLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?LoweringIssue {
    if (instruction.inputs.len != 1 or instruction.outputs.len != 1) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "get_tuple_element lowering requires exactly one tuple input and one output",
        .feature = "mlx-get-tuple-element",
    };
    const tuple_index = instruction.tuple_index orelse return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "get_tuple_element lowering requires a tuple index",
        .feature = "mlx-get-tuple-element",
    };
    if (tuple_index < 0) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "get_tuple_element tuple index must be non-negative",
        .feature = "mlx-get-tuple-element",
    };
    const tuple_id = instruction.inputs[0];
    if (tuple_id.index >= plan.values.len) return .{
        .instruction_index = instruction_index,
        .value_id = tuple_id,
        .op = instruction.kind,
        .detail = "get_tuple_element input is outside the executable value table",
        .feature = "mlx-get-tuple-element",
    };
    const tuple_value = plan.values[tuple_id.index];
    if (tuple_value.storage != .tuple or tuple_index >= @as(i64, @intCast(tuple_value.elements.len))) return .{
        .instruction_index = instruction_index,
        .value_id = tuple_id,
        .op = instruction.kind,
        .detail = "get_tuple_element input must be a tuple with the requested element",
        .feature = "mlx-get-tuple-element",
    };
    const element_id = tuple_value.elements[@intCast(tuple_index)];
    if (element_id.index >= plan.values.len) return .{
        .instruction_index = instruction_index,
        .value_id = element_id,
        .op = instruction.kind,
        .detail = "get_tuple_element selected element is outside the executable value table",
        .feature = "mlx-get-tuple-element",
    };
    if (plan.values[output_id.index].storage != .tensor or plan.values[element_id.index].storage != .tensor) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "get_tuple_element currently lowers tensor tuple elements only",
        .feature = "mlx-get-tuple-element",
    };
    const element = plan.values[element_id.index].descriptor;
    const output = plan.values[output_id.index].descriptor;
    if (element.element_type != output.element_type or !dimsEqual(element.dims, output.dims)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "get_tuple_element output descriptor must match the selected tuple element",
        .feature = "mlx-get-tuple-element",
    };
    return null;
}

fn validateSortLowering(instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?LoweringIssue {
    if (instruction.dimension == null) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "sort lowering requires a sort dimension",
        .feature = "mlx-sort",
    };
    if (!(instruction.inputs.len == 1 and instruction.outputs.len == 1) and !(instruction.inputs.len == 2 and instruction.outputs.len == 2)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "sort lowering supports single-value sort or key/value sort with two operands and two outputs",
        .feature = "mlx-sort-arity",
    };
    switch (instruction.compare_direction orelse .lt) {
        .lt, .le, .gt, .ge => return null,
        else => return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "sort lowering supports lt/le/gt/ge comparator directions only",
            .feature = "mlx-sort-comparator",
        },
    }
}

fn validateTopKLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?LoweringIssue {
    if (instruction.inputs.len != 1 or instruction.outputs.len != 2) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "top_k lowering requires one input and values/indices outputs",
        .feature = "mlx-top-k-arity",
    };
    const k = instruction.top_k_k orelse return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "top_k lowering requires static k",
        .feature = "mlx-top-k",
    };
    const input = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[0],
        .op = instruction.kind,
        .detail = "top_k input is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    if (input.dims.len == 0 or k < 0 or k > input.dims[input.dims.len - 1]) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "top_k supports static k along the last non-scalar dimension",
        .feature = "mlx-top-k-shape",
    };
    return null;
}

fn validateDotGeneralLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?LoweringIssue {
    const lhs = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 0) instruction.inputs[0] else output_id,
        .op = instruction.kind,
        .detail = "dot_general lhs is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const rhs = inputDescriptor(plan, instruction, 1) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 1) instruction.inputs[1] else output_id,
        .op = instruction.kind,
        .detail = "dot_general rhs is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const output = plan.values[output_id.index].descriptor;
    if (lhs.element_type != rhs.element_type or lhs.element_type != output.element_type or !isSupportedFloat(lhs.element_type)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "dot_general lowering currently supports same-dtype MLX floating matmul-like tensors only",
        .feature = "mlx-dot-general-float",
    };
    if (!dotGeneralIsMatmulLike(
        lhs.dims,
        rhs.dims,
        instruction.lhs_batch_dimensions orelse &.{},
        instruction.rhs_batch_dimensions orelse &.{},
        instruction.lhs_contracting_dimensions orelse &.{},
        instruction.rhs_contracting_dimensions orelse &.{},
        output.dims,
    )) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "dot_general lowering currently supports matmul-like contracting dimensions only",
        .feature = "mlx-dot-general-matmul",
    };
    return null;
}

fn validateConvolutionLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?LoweringIssue {
    const lhs = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 0) instruction.inputs[0] else output_id,
        .op = instruction.kind,
        .detail = "convolution lhs is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const rhs = inputDescriptor(plan, instruction, 1) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 1) instruction.inputs[1] else output_id,
        .op = instruction.kind,
        .detail = "convolution rhs is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const output = plan.values[output_id.index].descriptor;
    if (lhs.element_type != rhs.element_type or lhs.element_type != output.element_type or !isSupportedFloat(lhs.element_type)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "convolution lowering requires matching MLX-supported floating dtypes",
        .feature = "mlx-convolution-dtype",
    };
    if (lhs.dims.len != rhs.dims.len or lhs.dims.len != output.dims.len or lhs.dims.len < 3 or lhs.dims.len > 5) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "convolution lowering supports rank 3, 4, or 5 tensors only",
        .feature = "mlx-convolution-rank",
    };
    const spatial_rank = lhs.dims.len - 2;
    if (!std.mem.eql(i64, instruction.input_spatial_dimensions orelse &.{}, defaultSpatialDims(spatial_rank)) or
        !std.mem.eql(i64, instruction.kernel_spatial_dimensions orelse &.{}, defaultSpatialDims(spatial_rank)) or
        !std.mem.eql(i64, instruction.output_spatial_dimensions orelse &.{}, defaultSpatialDims(spatial_rank)) or
        instruction.input_batch_dimension != 0 or instruction.input_feature_dimension != 1 or
        instruction.kernel_output_feature_dimension != 0 or instruction.kernel_input_feature_dimension != 1 or
        instruction.output_batch_dimension != 0 or instruction.output_feature_dimension != 1)
    {
        return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "convolution lowering currently supports ZML NCHW/OIHW-style dimension numbers only",
            .feature = "mlx-convolution-layout",
        };
    }
    if ((instruction.batch_group_count orelse 1) != 1) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "convolution lowering does not support batch_group_count yet",
        .feature = "mlx-convolution-batch-groups",
    };
    if (!convMetadataLen(instruction.window_strides, spatial_rank) or
        !convMetadataLen(instruction.edge_padding_low, spatial_rank) or
        !convMetadataLen(instruction.edge_padding_high, spatial_rank) or
        !convMetadataLen(instruction.base_dilations, spatial_rank) or
        !convMetadataLen(instruction.window_dilations, spatial_rank) or
        !convReversalLen(instruction.window_reversal, spatial_rank))
    {
        return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "convolution lowering requires static spatial window metadata",
            .feature = "mlx-convolution-window",
        };
    }
    for (instruction.window_reversal orelse &.{}) |reversed| {
        if (reversed) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "convolution lowering does not support window_reversal yet",
            .feature = "mlx-convolution-window-reversal",
        };
    }
    const groups = instruction.feature_group_count orelse 1;
    if (groups <= 0 or @rem(lhs.dims[1], groups) != 0 or rhs.dims[1] * groups != lhs.dims[1]) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "convolution feature groups must divide input channels and match kernel input channels",
        .feature = "mlx-convolution-feature-groups",
    };
    if (output.dims[0] != lhs.dims[0] or output.dims[1] != rhs.dims[0]) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "convolution output batch/features must match lhs batch and rhs output features",
        .feature = "mlx-convolution-shape",
    };
    return null;
}

fn convMetadataLen(maybe_values: ?[]const i64, expected: usize) bool {
    const values = maybe_values orelse return false;
    return values.len == expected;
}

fn convReversalLen(maybe_values: ?[]const bool, expected: usize) bool {
    const values = maybe_values orelse return false;
    return values.len == expected;
}

fn defaultSpatialDims(rank: usize) []const i64 {
    return switch (rank) {
        1 => &.{2},
        2 => &.{ 2, 3 },
        3 => &.{ 2, 3, 4 },
        else => &.{},
    };
}

fn validateCholeskyLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?LoweringIssue {
    const input = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 0) instruction.inputs[0] else output_id,
        .op = instruction.kind,
        .detail = "cholesky input is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const output = plan.values[output_id.index].descriptor;
    if (input.element_type != .f32 or output.element_type != .f32) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "cholesky lowering currently supports f32 tensors only",
        .feature = "mlx-cholesky-dtype",
    };
    if (!std.mem.eql(i64, input.dims, output.dims) or output.dims.len < 2) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "cholesky output shape must match a rank >= 2 input",
        .feature = "mlx-cholesky-shape",
    };
    const n = output.dims[output.dims.len - 1];
    if (n <= 0 or output.dims[output.dims.len - 2] != n) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "cholesky lowering requires square minor dimensions",
        .feature = "mlx-cholesky-shape",
    };
    return null;
}

fn validateTriangularSolveLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?LoweringIssue {
    const a = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 0) instruction.inputs[0] else output_id,
        .op = instruction.kind,
        .detail = "triangular_solve matrix input is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const b = inputDescriptor(plan, instruction, 1) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 1) instruction.inputs[1] else output_id,
        .op = instruction.kind,
        .detail = "triangular_solve rhs input is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const output = plan.values[output_id.index].descriptor;
    if (a.element_type != .f32 or b.element_type != .f32 or output.element_type != .f32) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "triangular_solve lowering currently supports f32 tensors only",
        .feature = "mlx-triangular-solve-dtype",
    };
    if (!std.mem.eql(i64, b.dims, output.dims) or a.dims.len != b.dims.len or b.dims.len < 2) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "triangular_solve output shape must match rhs and ranks must match",
        .feature = "mlx-triangular-solve-shape",
    };
    const n = a.dims[a.dims.len - 1];
    if (n <= 0 or a.dims[a.dims.len - 2] != n) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "triangular_solve matrix input must have square minor dimensions",
        .feature = "mlx-triangular-solve-shape",
    };
    for (a.dims[0 .. a.dims.len - 2], b.dims[0 .. b.dims.len - 2]) |a_dim, b_dim| {
        if (a_dim != b_dim) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "triangular_solve lowering currently requires identical batch dimensions",
            .feature = "mlx-triangular-solve-batch",
        };
    }
    if (instruction.triangular_left_side orelse true) {
        if (b.dims[b.dims.len - 2] != n) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "left-side triangular_solve requires rhs row dimension to match matrix size",
            .feature = "mlx-triangular-solve-shape",
        };
    } else if (b.dims[b.dims.len - 1] != n) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "right-side triangular_solve requires rhs column dimension to match matrix size",
        .feature = "mlx-triangular-solve-shape",
    };
    return null;
}

fn validateFftLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?LoweringIssue {
    const input = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 0) instruction.inputs[0] else output_id,
        .op = instruction.kind,
        .detail = "fft input is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const output = plan.values[output_id.index].descriptor;
    const lengths = instruction.dimensions orelse return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "fft lowering requires StableHLO fft_length metadata",
        .feature = "mlx-fft-metadata",
    };
    const fft_kind = instruction.fft_kind orelse return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "fft lowering requires StableHLO fft_type metadata",
        .feature = "mlx-fft-metadata",
    };
    if (lengths.len == 0 or lengths.len > 3 or lengths.len > input.dims.len) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "fft lowering supports one to three innermost FFT dimensions",
        .feature = "mlx-fft-rank",
    };
    if (input.dims.len != output.dims.len) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "fft lowering requires input and output tensors to have the same rank",
        .feature = "mlx-fft-shape",
    };
    for (lengths) |length| {
        if (length <= 0) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "fft_length values must be positive",
            .feature = "mlx-fft-shape",
        };
    }
    switch (fft_kind) {
        .fft, .ifft => if (input.element_type != .c64 or output.element_type != .c64) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "FFT/IFFT lowering currently supports c64 tensors only",
            .feature = "mlx-fft-dtype",
        },
        .rfft => if (input.element_type != .f32 or output.element_type != .c64) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "RFFT lowering currently supports f32 input and c64 output only",
            .feature = "mlx-fft-dtype",
        },
        .irfft => if (input.element_type != .c64 or output.element_type != .f32) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "IRFFT lowering currently supports c64 input and f32 output only",
            .feature = "mlx-fft-dtype",
        },
    }
    const first_fft_axis = input.dims.len - lengths.len;
    for (lengths, 0..) |length, index| {
        const axis = first_fft_axis + index;
        const input_dim = input.dims[axis];
        const output_dim = output.dims[axis];
        switch (fft_kind) {
            .fft, .ifft => if (input_dim != length or output_dim != length) return .{
                .instruction_index = instruction_index,
                .value_id = output_id,
                .op = instruction.kind,
                .detail = "FFT/IFFT lowering requires innermost input/output dimensions to match fft_length",
                .feature = "mlx-fft-shape",
            },
            .rfft => {
                const expected_output = if (index == lengths.len - 1) @divFloor(length, 2) + 1 else length;
                if (input_dim != length or output_dim != expected_output) return .{
                    .instruction_index = instruction_index,
                    .value_id = output_id,
                    .op = instruction.kind,
                    .detail = "RFFT lowering requires innermost input dimensions to match fft_length and final output dimension length/2+1",
                    .feature = "mlx-fft-shape",
                };
            },
            .irfft => {
                const expected_input = if (index == lengths.len - 1) @divFloor(length, 2) + 1 else length;
                if (input_dim != expected_input or output_dim != length) return .{
                    .instruction_index = instruction_index,
                    .value_id = output_id,
                    .op = instruction.kind,
                    .detail = "IRFFT lowering requires final input dimension fft_length/2+1 and output dimensions to match fft_length",
                    .feature = "mlx-fft-shape",
                };
            },
        }
    }
    return null;
}

fn validateRngBitGeneratorLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?LoweringIssue {
    if (instruction.inputs.len != 1 or instruction.outputs.len != 2) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "rng_bit_generator lowering requires one state input and state/bits outputs",
        .feature = "mlx-rng-arity",
    };
    const state = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[0],
        .op = instruction.kind,
        .detail = "rng_bit_generator state input is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const state_output = plan.values[instruction.outputs[0].index].descriptor;
    const bits_output = plan.values[instruction.outputs[1].index].descriptor;
    if (!std.mem.eql(i64, state.dims, &.{2}) or (state.element_type != .u32 and state.element_type != .u64)) return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[0],
        .op = instruction.kind,
        .detail = "rng_bit_generator state must be u32[2] or u64[2]",
        .feature = "mlx-rng-state",
    };
    if (state.element_type != state_output.element_type or !std.mem.eql(i64, state.dims, state_output.dims)) return .{
        .instruction_index = instruction_index,
        .value_id = instruction.outputs[0],
        .op = instruction.kind,
        .detail = "rng_bit_generator state output must preserve state dtype and shape",
        .feature = "mlx-rng-state",
    };
    if (bits_output.element_type != .u8 and bits_output.element_type != .u16 and bits_output.element_type != .u32 and bits_output.element_type != .u64) return .{
        .instruction_index = instruction_index,
        .value_id = instruction.outputs[1],
        .op = instruction.kind,
        .detail = "rng_bit_generator bits output supports u8/u16/u32/u64 only",
        .feature = "mlx-rng-bits",
    };
    if (bits_output.element_type == .u64 and state.element_type != .u64) return .{
        .instruction_index = instruction_index,
        .value_id = instruction.outputs[1],
        .op = instruction.kind,
        .detail = "rng_bit_generator u64 bits require the StableHLO u64 state path",
        .feature = "mlx-rng-u64-state",
    };
    return null;
}

fn descriptorsEqual(a: ir.BufferDescriptor, b: ir.BufferDescriptor) bool {
    return a.element_type == b.element_type and std.mem.eql(i64, a.dims, b.dims);
}

fn regionValueById(subprogram: backend.ProgramSubprogram, id: ir.RegionValueId) ?ir.RegionValue {
    if (id.index >= subprogram.values.len) return null;
    return subprogram.values[id.index];
}

fn regionValueIsArgumentIndex(subprogram: backend.ProgramSubprogram, id: ir.RegionValueId, index: usize) bool {
    const value = regionValueById(subprogram, id) orelse return false;
    return value.role == .argument and id.index == index;
}

fn constantCompatibleWithState(value: ir.RegionValue, state: ir.BufferDescriptor) bool {
    if (value.role != .constant or value.literal == null) return false;
    if (value.descriptor.element_type != state.element_type) return false;
    if (value.descriptor.element_type != .f32 and value.descriptor.element_type != .bf16) return false;
    if (value.descriptor.dims.len == 0) return true;
    return std.mem.eql(i64, value.descriptor.dims, state.dims);
}

fn whileOperandCompatibleWithState(value: ir.RegionValue, state: ir.BufferDescriptor) bool {
    if (value.role != .constant and value.role != .argument and value.role != .instruction_result) return false;
    if (value.role == .constant and value.literal == null) return false;
    if (value.descriptor.element_type != state.element_type) return false;
    if (value.descriptor.element_type != .f32 and value.descriptor.element_type != .bf16) return false;
    if (value.descriptor.dims.len == 0) return true;
    return std.mem.eql(i64, value.descriptor.dims, state.dims);
}

fn addInstructionStepOperand(subprogram: backend.ProgramSubprogram, instruction: ir.RegionInstruction, state: ir.BufferDescriptor, state_index: usize, update_instruction_index: usize) ?WhilePatternOperand {
    if (instruction.inputs.len != 2) return null;
    if (regionValueIsArgumentIndex(subprogram, instruction.inputs[0], state_index)) {
        const rhs = regionValueById(subprogram, instruction.inputs[1]) orelse return null;
        return whileStepOperandFromRegionValue(subprogram, rhs, state, state_index, update_instruction_index);
    }
    if (instruction.kind == .add and regionValueIsArgumentIndex(subprogram, instruction.inputs[1], state_index)) {
        const lhs = regionValueById(subprogram, instruction.inputs[0]) orelse return null;
        return whileStepOperandFromRegionValue(subprogram, lhs, state, state_index, update_instruction_index);
    }
    return null;
}

fn whileStepOperandFromRegionValue(subprogram: backend.ProgramSubprogram, value: ir.RegionValue, state: ir.BufferDescriptor, state_index: usize, update_instruction_index: usize) ?WhilePatternOperand {
    if (!whileOperandCompatibleWithState(value, state)) return null;
    switch (value.role) {
        .constant, .argument => return .{ .value = value },
        .instruction_result => {
            const producer_index = loopInvariantProducerInstructionIndex(subprogram, value.id, state, state_index, update_instruction_index) orelse return null;
            return .{ .value = value, .producer_instruction_index = producer_index };
        },
    }
}

fn loopInvariantProducerInstructionIndex(subprogram: backend.ProgramSubprogram, output_id: ir.RegionValueId, state: ir.BufferDescriptor, state_index: usize, update_instruction_index: usize) ?usize {
    var instruction_index: usize = 0;
    while (instruction_index < update_instruction_index) : (instruction_index += 1) {
        const instruction = subprogram.instructions[instruction_index];
        if (instruction.outputs.len != 1 or instruction.outputs[0].index != output_id.index) continue;
        if (instruction.result_descriptors.len != 1 or !descriptorsEqual(instruction.result_descriptors[0], state)) return null;
        if (executableBinaryOp(instruction.kind) == null) return null;
        if (instruction.inputs.len != 2) return null;
        for (instruction.inputs) |input_id| {
            const input = regionValueById(subprogram, input_id) orelse return null;
            if (!whileOperandCompatibleWithState(input, state)) return null;
            if (input.role != .argument) return null;
            if (input.id.index == state_index) return null;
        }
        return instruction_index;
    }
    return null;
}

fn compareDirectionSupportedInWhile(direction: ir.CompareOp) bool {
    return switch (direction) {
        .lt, .le, .gt, .ge => true,
        .eq, .ne => false,
    };
}

fn matchWhileF32LtAddPattern(cond: backend.ProgramSubprogram, body: backend.ProgramSubprogram) ?WhileF32LtAddPattern {
    if (cond.kind != .while_cond or body.kind != .while_body) return null;
    if (cond.argument_descriptors.len == 0 or body.argument_descriptors.len != cond.argument_descriptors.len) return null;
    if (cond.instructions.len != 1 or body.instructions.len == 0 or body.instructions.len > 3) return null;
    if (cond.terminator_operands.len != 1 or body.terminator_operands.len != body.argument_descriptors.len) return null;
    for (cond.argument_descriptors, 0..) |descriptor, argument_index| {
        if (!descriptorsEqual(descriptor, body.argument_descriptors[argument_index])) return null;
    }

    const compare_instruction = cond.instructions[0];
    if (compare_instruction.kind != .compare) return null;
    const compare_direction = compare_instruction.compare_direction orelse return null;
    if (!compareDirectionSupportedInWhile(compare_direction)) return null;
    if (compare_instruction.inputs.len != 2 or compare_instruction.outputs.len != 1) return null;
    if (compare_instruction.outputs[0].index != cond.terminator_operands[0].index) return null;

    var loop_state_index: ?usize = null;
    var argument_index: usize = 0;
    while (argument_index < cond.argument_descriptors.len) : (argument_index += 1) {
        if (regionValueIsArgumentIndex(cond, compare_instruction.inputs[0], argument_index)) {
            loop_state_index = argument_index;
            break;
        }
    }
    const state_index = loop_state_index orelse return null;
    const state = cond.argument_descriptors[state_index];
    if (state.element_type != .f32 and state.element_type != .bf16) return null;
    const limit = regionValueById(cond, compare_instruction.inputs[1]) orelse return null;
    if (!whileOperandCompatibleWithState(limit, state)) return null;

    var update_instruction_index: ?usize = null;
    var body_instruction_index: usize = 0;
    while (body_instruction_index < body.instructions.len) : (body_instruction_index += 1) {
        const candidate = body.instructions[body_instruction_index];
        if (candidate.kind != .add and candidate.kind != .subtract) continue;
        if (candidate.outputs.len != 1) continue;
        if (addInstructionStepOperand(body, candidate, state, state_index, body_instruction_index) == null) continue;
        update_instruction_index = body_instruction_index;
        break;
    }
    const update_index = update_instruction_index orelse return null;
    const update_instruction = body.instructions[update_index];
    if ((update_instruction.kind != .add and update_instruction.kind != .subtract) or update_instruction.outputs.len != 1) return null;
    const step = addInstructionStepOperand(body, update_instruction, state, state_index, update_index) orelse return null;
    if (update_index + 1 == body.instructions.len) {
        if (update_instruction.outputs[0].index != body.terminator_operands[state_index].index) return null;
    } else {
        if (update_index + 2 != body.instructions.len) return null;
        const cast_instruction = body.instructions[update_index + 1];
        if (cast_instruction.kind != .convert or cast_instruction.inputs.len != 1 or cast_instruction.outputs.len != 1) return null;
        if (cast_instruction.inputs[0].index != update_instruction.outputs[0].index) return null;
        if (cast_instruction.outputs[0].index != body.terminator_operands[state_index].index) return null;
        if (cast_instruction.result_descriptors.len != 1 or !descriptorsEqual(cast_instruction.result_descriptors[0], state)) return null;
    }
    var invariant_index: usize = 0;
    while (invariant_index < cond.argument_descriptors.len) : (invariant_index += 1) {
        if (invariant_index == state_index) continue;
        if (!regionValueIsArgumentIndex(body, body.terminator_operands[invariant_index], invariant_index)) return null;
    }
    const update_op: ir.ElementwiseBinaryOp = if (update_instruction.kind == .subtract) .subtract else .add;
    return .{ .limit = limit, .step = step, .state_index = state_index, .compare_direction = compare_direction, .update_op = update_op, .state_count = cond.argument_descriptors.len };
}

fn validateWhileLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?LoweringIssue {
    if (instruction.inputs.len == 0 or instruction.outputs.len != instruction.inputs.len or instruction.region_ids.len != 2) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "while lowering requires state inputs, matching state outputs, and cond/body regions",
        .feature = "mlx-while-region-contract",
    };
    const cond_id = instruction.region_ids[0];
    const body_id = instruction.region_ids[1];
    if (cond_id.index >= plan.regions.len or body_id.index >= plan.regions.len) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "while region ids must reference captured PjRTx region summaries",
        .feature = "mlx-while-region-contract",
    };
    const cond = plan.regions[cond_id.index];
    const body = plan.regions[body_id.index];
    if (cond.kind != .while_cond or body.kind != .while_body) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "while lowering requires cond region followed by body region",
        .feature = "mlx-while-region-contract",
    };
    if (cond.argument_descriptors.len != instruction.inputs.len or body.argument_descriptors.len != instruction.inputs.len) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "while cond/body region arguments must match the loop state arity",
        .feature = "mlx-while-region-contract",
    };
    if (cond.return_descriptors.len != 1 or cond.return_descriptors[0].element_type != .pred or cond.return_descriptors[0].dims.len != 0) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "while cond region must return a scalar predicate",
        .feature = "mlx-while-region-contract",
    };
    if (body.return_descriptors.len != instruction.outputs.len) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "while body region return arity must match the loop state arity",
        .feature = "mlx-while-region-contract",
    };
    for (instruction.inputs, instruction.outputs, 0..) |input_id, state_output_id, state_index| {
        if (input_id.index >= plan.values.len or state_output_id.index >= plan.values.len) return .{
            .instruction_index = instruction_index,
            .value_id = state_output_id,
            .op = instruction.kind,
            .detail = "while state values must reference executable value descriptors",
            .feature = "mlx-executable-values",
        };
        const input = plan.values[input_id.index].descriptor;
        const output = plan.values[state_output_id.index].descriptor;
        if (!descriptorsEqual(input, output) or
            !descriptorsEqual(input, cond.argument_descriptors[state_index]) or
            !descriptorsEqual(input, body.argument_descriptors[state_index]) or
            !descriptorsEqual(input, body.return_descriptors[state_index]))
        {
            return .{
                .instruction_index = instruction_index,
                .value_id = state_output_id,
                .op = instruction.kind,
                .detail = "while loop state descriptors must be invariant across inputs, outputs, body args, and body returns",
                .feature = "mlx-while-region-contract",
            };
        }
    }
    const matched = matchWhileF32LtAddPattern(.{
        .id = 0,
        .parent_node = instruction_index,
        .region_id = cond.id,
        .kind = cond.kind,
        .values = cond.values,
        .argument_descriptors = cond.argument_descriptors,
        .instructions = cond.instructions,
        .return_descriptors = cond.return_descriptors,
        .terminator_operands = cond.terminator_operands,
        .terminator_operand_descriptors = cond.terminator_operand_descriptors,
    }, .{
        .id = 1,
        .parent_node = instruction_index,
        .region_id = body.id,
        .kind = body.kind,
        .values = body.values,
        .argument_descriptors = body.argument_descriptors,
        .instructions = body.instructions,
        .return_descriptors = body.return_descriptors,
        .terminator_operands = body.terminator_operands,
        .terminator_operand_descriptors = body.terminator_operand_descriptors,
    }) != null;
    if (matched) return null;
    return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "while region subprogram lowering requires a supported device-side loop pattern; host-loop execution is disabled",
        .feature = "mlx-while-region-pattern",
    };
}

fn validateReduceLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?LoweringIssue {
    if (instruction.inputs.len == 2 and instruction.outputs.len == 2 and instruction.kind == .reduce_max) {
        const values = inputDescriptor(plan, instruction, 0) orelse return .{
            .instruction_index = instruction_index,
            .value_id = instruction.inputs[0],
            .op = instruction.kind,
            .detail = "reduce argmax values input is outside the executable value table",
            .feature = "mlx-executable-values",
        };
        const indices = inputDescriptor(plan, instruction, 1) orelse return .{
            .instruction_index = instruction_index,
            .value_id = instruction.inputs[1],
            .op = instruction.kind,
            .detail = "reduce argmax indices input is outside the executable value table",
            .feature = "mlx-executable-values",
        };
        const values_output = plan.values[instruction.outputs[0].index].descriptor;
        const indices_output = plan.values[instruction.outputs[1].index].descriptor;
        if (!isSupportedFloat(values.element_type) or values.element_type != values_output.element_type) return .{
            .instruction_index = instruction_index,
            .value_id = instruction.outputs[0],
            .op = instruction.kind,
            .detail = "reduce argmax values must preserve f16/bf16/f32 dtype",
            .feature = "mlx-reduce-types",
        };
        if ((indices.element_type != .s32 and indices.element_type != .u32) or indices.element_type != indices_output.element_type) return .{
            .instruction_index = instruction_index,
            .value_id = instruction.outputs[1],
            .op = instruction.kind,
            .detail = "reduce argmax indices must preserve s32/u32 dtype",
            .feature = "mlx-reduce-types",
        };
        if (!std.mem.eql(i64, values.dims, indices.dims) or !std.mem.eql(i64, values_output.dims, indices_output.dims)) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "reduce argmax values and indices must have matching input and output shapes",
            .feature = "mlx-reduce-shape",
        };
        if (!validReduceShape(values.dims, instruction.reduce_dimensions orelse &.{}, values_output.dims)) return .{
            .instruction_index = instruction_index,
            .value_id = instruction.outputs[0],
            .op = instruction.kind,
            .detail = "reduce argmax output shape must equal input shape with reduced axes removed",
            .feature = "mlx-reduce-shape",
        };
        return null;
    }
    const input = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 0) instruction.inputs[0] else output_id,
        .op = instruction.kind,
        .detail = "reduce operand is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const output = plan.values[output_id.index].descriptor;
    const supported_reduce_type = switch (instruction.kind) {
        .reduce_sum, .reduce_max, .reduce_min => input.element_type == output.element_type and (isSupportedFloat(input.element_type) or isSupportedInteger(input.element_type)),
        .reduce_and, .reduce_or => input.element_type == .pred and output.element_type == .pred,
        else => false,
    };
    if (!supported_reduce_type) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "reduce lowering supports integer and f16/bf16/f32 sum/max/min plus pred and/or only",
        .feature = "mlx-reduce-types",
    };
    if (!validReduceShape(input.dims, instruction.reduce_dimensions orelse &.{}, output.dims)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "reduce output shape must equal input shape with reduced axes removed",
        .feature = "mlx-reduce-shape",
    };
    return null;
}

fn validateReduceWindowLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?LoweringIssue {
    if (instruction.inputs.len == 2 and instruction.outputs.len == 2 and instruction.kind == .reduce_window_max) {
        const values = inputDescriptor(plan, instruction, 0) orelse return .{
            .instruction_index = instruction_index,
            .value_id = instruction.inputs[0],
            .op = instruction.kind,
            .detail = "reduce_window max-pool values input is outside the executable value table",
            .feature = "mlx-executable-values",
        };
        const indices = inputDescriptor(plan, instruction, 1) orelse return .{
            .instruction_index = instruction_index,
            .value_id = instruction.inputs[1],
            .op = instruction.kind,
            .detail = "reduce_window max-pool indices input is outside the executable value table",
            .feature = "mlx-executable-values",
        };
        const values_output = plan.values[instruction.outputs[0].index].descriptor;
        const indices_output = plan.values[instruction.outputs[1].index].descriptor;
        if (!isSupportedFloat(values.element_type) or values.element_type != values_output.element_type) return .{
            .instruction_index = instruction_index,
            .value_id = instruction.outputs[0],
            .op = instruction.kind,
            .detail = "reduce_window max-pool values must preserve f16/bf16/f32 dtype",
            .feature = "mlx-reduce-window-types",
        };
        if ((indices.element_type != .s32 and indices.element_type != .u32) or indices.element_type != indices_output.element_type) return .{
            .instruction_index = instruction_index,
            .value_id = instruction.outputs[1],
            .op = instruction.kind,
            .detail = "reduce_window max-pool indices must preserve s32/u32 dtype",
            .feature = "mlx-reduce-window-types",
        };
        if (!std.mem.eql(i64, values.dims, indices.dims) or !std.mem.eql(i64, values_output.dims, indices_output.dims)) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "reduce_window max-pool values and indices must have matching input and output shapes",
            .feature = "mlx-reduce-window-shape",
        };
        if (!validReduceWindowShape(
            values.dims,
            instruction.window_dimensions orelse &.{},
            instruction.window_strides orelse &.{},
            instruction.base_dilations orelse &.{},
            instruction.window_dilations orelse &.{},
            instruction.edge_padding_low orelse &.{},
            instruction.edge_padding_high orelse &.{},
            values_output.dims,
        )) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "reduce_window max-pool shape metadata must match StableHLO static window shape formula with unit base dilation",
            .feature = "mlx-reduce-window-shape",
        };
        return null;
    }
    if (instruction.inputs.len != 1 or instruction.outputs.len != 1) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "reduce_window lowering currently supports exactly one operand/result",
        .feature = "mlx-reduce-window-arity",
    };
    const input = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[0],
        .op = instruction.kind,
        .detail = "reduce_window operand is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const output = plan.values[output_id.index].descriptor;
    if (input.element_type != output.element_type or !isSupportedFloat(input.element_type)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "reduce_window lowering currently supports f16/bf16/f32 sum/max only",
        .feature = "mlx-reduce-window-types",
    };
    if (!validReduceWindowShape(
        input.dims,
        instruction.window_dimensions orelse &.{},
        instruction.window_strides orelse &.{},
        instruction.base_dilations orelse &.{},
        instruction.window_dilations orelse &.{},
        instruction.edge_padding_low orelse &.{},
        instruction.edge_padding_high orelse &.{},
        output.dims,
    )) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "reduce_window shape metadata must match StableHLO static window shape formula with unit base dilation",
        .feature = "mlx-reduce-window-shape",
    };
    return null;
}

fn validateCompareLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?LoweringIssue {
    const lhs = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 0) instruction.inputs[0] else output_id,
        .op = instruction.kind,
        .detail = "compare lhs is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const rhs = inputDescriptor(plan, instruction, 1) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 1) instruction.inputs[1] else output_id,
        .op = instruction.kind,
        .detail = "compare rhs is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const output = plan.values[output_id.index].descriptor;
    if (lhs.element_type != rhs.element_type or !isSupportedComparable(lhs.element_type) or output.element_type != .pred or !validElementwiseBroadcast(lhs.dims, rhs.dims, output.dims)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "compare lowering requires MLX-supported comparable inputs, pred output, and broadcast-compatible shapes",
        .feature = "mlx-compare",
    };
    return null;
}

fn validateSelectLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?LoweringIssue {
    const pred = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 0) instruction.inputs[0] else output_id,
        .op = instruction.kind,
        .detail = "select predicate is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const on_true = inputDescriptor(plan, instruction, 1) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 1) instruction.inputs[1] else output_id,
        .op = instruction.kind,
        .detail = "select true operand is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const on_false = inputDescriptor(plan, instruction, 2) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 2) instruction.inputs[2] else output_id,
        .op = instruction.kind,
        .detail = "select false operand is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const output = plan.values[output_id.index].descriptor;
    if (pred.element_type != .pred or on_true.element_type != on_false.element_type or !dimsEqual(pred.dims, on_true.dims) or !dimsEqual(on_true.dims, on_false.dims) or !dimsEqual(on_true.dims, output.dims)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "select lowering requires same-shape pred/true/false tensors",
        .feature = "mlx-select",
    };
    return null;
}

fn validateClampLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?LoweringIssue {
    const min = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 0) instruction.inputs[0] else output_id,
        .op = instruction.kind,
        .detail = "clamp min operand is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const value = inputDescriptor(plan, instruction, 1) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 1) instruction.inputs[1] else output_id,
        .op = instruction.kind,
        .detail = "clamp value operand is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const max = inputDescriptor(plan, instruction, 2) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 2) instruction.inputs[2] else output_id,
        .op = instruction.kind,
        .detail = "clamp max operand is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const output = plan.values[output_id.index].descriptor;
    if (min.element_type != value.element_type or max.element_type != value.element_type or output.element_type != value.element_type or !dimsEqual(value.dims, output.dims)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "clamp lowering requires matching min/value/max/output dtypes and value/output shapes",
        .feature = "mlx-clamp",
    };
    if (min.dims.len != 0 and !dimsEqual(min.dims, value.dims)) return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[0],
        .op = instruction.kind,
        .detail = "clamp min must be scalar or match the value shape",
        .feature = "mlx-clamp-bounds",
    };
    if (max.dims.len != 0 and !dimsEqual(max.dims, value.dims)) return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[2],
        .op = instruction.kind,
        .detail = "clamp max must be scalar or match the value shape",
        .feature = "mlx-clamp-bounds",
    };
    return null;
}

fn inputDescriptor(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, input_index: usize) ?ir.BufferDescriptor {
    if (input_index >= instruction.inputs.len) return null;
    const id = instruction.inputs[input_index];
    if (id.index >= plan.values.len) return null;
    return plan.values[id.index].descriptor;
}

fn dimsEqual(lhs: []const i64, rhs: []const i64) bool {
    if (lhs.len != rhs.len) return false;
    for (lhs, rhs) |a, b| if (a != b) return false;
    return true;
}

fn validReduceShape(input_dims: []const i64, dimensions: []const i64, output_dims: []const i64) bool {
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
        if (reduced[axis]) continue;
        if (output_dims[out_axis] != dim) return false;
        out_axis += 1;
    }
    return true;
}

fn validReduceWindowShape(input_dims: []const i64, window_dimensions: []const i64, window_strides: []const i64, base_dilations: []const i64, window_dilations: []const i64, padding_low: []const i64, padding_high: []const i64, output_dims: []const i64) bool {
    const rank = input_dims.len;
    if (rank == 0 or output_dims.len != rank or window_dimensions.len != rank or window_strides.len != rank or base_dilations.len != rank or window_dilations.len != rank or padding_low.len != rank or padding_high.len != rank) return false;
    for (0..rank) |axis| {
        if (input_dims[axis] < 0 or window_dimensions[axis] <= 0 or window_strides[axis] <= 0 or base_dilations[axis] != 1 or window_dilations[axis] <= 0 or padding_low[axis] < 0 or padding_high[axis] < 0) return false;
        const padded = padding_low[axis] + input_dims[axis] + padding_high[axis];
        const window = (window_dimensions[axis] - 1) * window_dilations[axis] + 1;
        const expected = if (padded < window) 0 else @divFloor(padded - window, window_strides[axis]) + 1;
        if (output_dims[axis] != expected) return false;
    }
    return true;
}

fn executeRegisteredCustomCall(
    backend_impl: backend.Backend,
    instruction: ir.PlanInstruction,
    value_handles: []?backend.BufferHandle,
) backend.Error!?backend.BufferHandle {
    const target = instruction.custom_call_target orelse return null;
    const spec = lookupCustomCall(target) orelse return null;
    return switch (spec.kind) {
        .identity => blk: {
            if (instruction.inputs.len != 1) return null;
            const input_id = instruction.inputs[0];
            if (input_id.index >= value_handles.len) return error.CommandSubmissionFailed;
            const input = value_handles[input_id.index] orelse return error.CommandSubmissionFailed;
            break :blk (try cloneBuffer(backend_impl, input)) orelse return null;
        },
        .unary => blk: {
            if (instruction.inputs.len != 1) return null;
            const input_id = instruction.inputs[0];
            if (input_id.index >= value_handles.len) return error.CommandSubmissionFailed;
            const input = value_handles[input_id.index] orelse return error.CommandSubmissionFailed;
            break :blk (try unary(backend_impl, input, spec.unary_op.?)) orelse return null;
        },
        .binary => blk: {
            if (instruction.inputs.len != 2) return null;
            const lhs_id = instruction.inputs[0];
            const rhs_id = instruction.inputs[1];
            if (lhs_id.index >= value_handles.len or rhs_id.index >= value_handles.len) return error.CommandSubmissionFailed;
            const lhs = value_handles[lhs_id.index] orelse return error.CommandSubmissionFailed;
            const rhs = value_handles[rhs_id.index] orelse return error.CommandSubmissionFailed;
            break :blk (try binary(backend_impl, lhs, rhs, spec.binary_op.?)) orelse return null;
        },
        .metal_kernel_binary_add_f32 => blk: {
            if (instruction.inputs.len != 2) return null;
            const lhs_id = instruction.inputs[0];
            const rhs_id = instruction.inputs[1];
            if (lhs_id.index >= value_handles.len or rhs_id.index >= value_handles.len) return error.CommandSubmissionFailed;
            const lhs = value_handles[lhs_id.index] orelse return error.CommandSubmissionFailed;
            const rhs = value_handles[rhs_id.index] orelse return error.CommandSubmissionFailed;
            break :blk maybeBufferHandle((try buffer_mod.Buffer.customBinaryAddF32(bufferRef(lhs), bufferRef(rhs))) orelse return error.CommandSubmissionFailed).?;
        },
        .scaled_dot_product_attention => blk: {
            if (instruction.inputs.len != 4) return null;
            const q_id = instruction.inputs[0];
            const k_id = instruction.inputs[1];
            const v_id = instruction.inputs[2];
            const token_index_id = instruction.inputs[3];
            if (q_id.index >= value_handles.len or k_id.index >= value_handles.len or v_id.index >= value_handles.len or token_index_id.index >= value_handles.len) return error.CommandSubmissionFailed;
            const q = value_handles[q_id.index] orelse return error.CommandSubmissionFailed;
            const k = value_handles[k_id.index] orelse return error.CommandSubmissionFailed;
            const v = value_handles[v_id.index] orelse return error.CommandSubmissionFailed;
            const token_index = value_handles[token_index_id.index] orelse return error.CommandSubmissionFailed;
            break :blk maybeBufferHandle((try buffer_mod.Buffer.customScaledDotProductAttention(bufferRef(q), bufferRef(k), bufferRef(v), bufferRef(token_index))) orelse return error.CommandSubmissionFailed).?;
        },
    };
}

fn isSupportedFloat(element_type: ir.BufferType) bool {
    return switch (element_type) {
        .f16, .f32, .bf16 => true,
        else => false,
    };
}

fn isSupportedInteger(element_type: ir.BufferType) bool {
    return switch (element_type) {
        .s8, .s32, .u8, .u16, .u32, .u64 => true,
        else => false,
    };
}

fn isSupportedComparable(element_type: ir.BufferType) bool {
    return element_type == .pred or isSupportedFloat(element_type) or isSupportedInteger(element_type);
}

fn dimsBroadcastTo(input_dims: []const i64, output_dims: []const i64) bool {
    if (input_dims.len > output_dims.len) return false;
    const offset = output_dims.len - input_dims.len;
    for (input_dims, 0..) |dim, index| {
        const output_dim = output_dims[offset + index];
        if (dim != 1 and dim != output_dim) return false;
    }
    return true;
}

fn validElementwiseBroadcast(lhs_dims: []const i64, rhs_dims: []const i64, output_dims: []const i64) bool {
    return dimsBroadcastTo(lhs_dims, output_dims) and dimsBroadcastTo(rhs_dims, output_dims);
}

fn dotGeneralIsMatmulLike(lhs_dims: []const i64, rhs_dims: []const i64, lhs_batch: []const i64, rhs_batch: []const i64, lhs_contract: []const i64, rhs_contract: []const i64, output_dims: []const i64) bool {
    if (lhs_contract.len != 1 or rhs_contract.len != 1 or lhs_batch.len != rhs_batch.len or lhs_dims.len == 0 or rhs_dims.len < 2 or output_dims.len == 0) return false;
    const lhs_k = lhs_contract[0];
    const rhs_k = rhs_contract[0];
    if (lhs_k < 0 or rhs_k < 0) return false;
    if (@as(usize, @intCast(lhs_k)) >= lhs_dims.len or @as(usize, @intCast(rhs_k)) >= rhs_dims.len) return false;
    if (lhs_dims[@intCast(lhs_k)] != rhs_dims[@intCast(rhs_k)]) return false;
    var lhs_used_buf: [16]bool = [_]bool{false} ** 16;
    var rhs_used_buf: [16]bool = [_]bool{false} ** 16;
    if (lhs_dims.len > lhs_used_buf.len or rhs_dims.len > rhs_used_buf.len) return false;
    const lhs_used = lhs_used_buf[0..lhs_dims.len];
    const rhs_used = rhs_used_buf[0..rhs_dims.len];
    lhs_used[@intCast(lhs_k)] = true;
    rhs_used[@intCast(rhs_k)] = true;
    for (lhs_batch, rhs_batch) |lhs_axis, rhs_axis| {
        if (lhs_axis < 0 or rhs_axis < 0) return false;
        if (@as(usize, @intCast(lhs_axis)) >= lhs_dims.len or @as(usize, @intCast(rhs_axis)) >= rhs_dims.len) return false;
        if (lhs_used[@intCast(lhs_axis)] or rhs_used[@intCast(rhs_axis)]) return false;
        if (lhs_dims[@intCast(lhs_axis)] != rhs_dims[@intCast(rhs_axis)]) return false;
        lhs_used[@intCast(lhs_axis)] = true;
        rhs_used[@intCast(rhs_axis)] = true;
    }
    var expected_buf: [32]i64 = undefined;
    var expected: std.ArrayListUnmanaged(i64) = .initBuffer(&expected_buf);
    for (lhs_batch) |axis| expected.appendBounded(lhs_dims[@intCast(axis)]) catch return false;
    for (lhs_dims, 0..) |dim, axis| if (!lhs_used[axis]) expected.appendBounded(dim) catch return false;
    for (rhs_dims, 0..) |dim, axis| if (!rhs_used[axis]) expected.appendBounded(dim) catch return false;
    return std.mem.eql(i64, expected.items, output_dims);
}

fn writeLoweringIssue(plan: *const ir.ExecutablePlan, issue: LoweringIssue, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    try writer.writeAll("invalid executable plan: pass=mlx-backend-legalization");
    if (issue.instruction_index) |index| try writer.print(" instruction={d}", .{index});
    if (issue.op) |op| try writer.print(" op={s}", .{@tagName(op)});
    if (issue.value_id) |value_id| {
        try writer.print(" value={d}", .{value_id.index});
        if (value_id.index < plan.values.len) {
            const descriptor = plan.values[value_id.index].descriptor;
            try writer.print(" dtype={s} rank={d} shape=", .{ @tagName(descriptor.element_type), descriptor.dims.len });
            try writeDims(writer, descriptor.dims);
            try writer.print(" sharding={s}", .{shardingLabel(plan, value_id)});
        }
    }
    try writer.print(" detail=\"{s}\" feature={s}", .{ issue.detail, issue.feature });
}

fn writeDims(writer: *std.Io.Writer, dims: []const i64) std.Io.Writer.Error!void {
    try writer.writeAll("[");
    for (dims, 0..) |dim, index| {
        if (index != 0) try writer.writeAll(",");
        try writer.print("{d}", .{dim});
    }
    try writer.writeAll("]");
}

fn shardingLabel(plan: *const ir.ExecutablePlan, value_id: ir.ValueId) []const u8 {
    for (plan.output_ids, 0..) |output_id, index| {
        if (output_id.index == value_id.index and index < plan.output_shardings.len) return @tagName(plan.output_shardings[index].kind);
    }
    if (value_id.index < plan.values.len and plan.values[value_id.index].role == .parameter) {
        var parameter_index: usize = 0;
        for (plan.values) |value| {
            if (value.role != .parameter) continue;
            if (value.id.index == value_id.index and parameter_index < plan.parameter_shardings.len) return @tagName(plan.parameter_shardings[parameter_index].kind);
            parameter_index += 1;
        }
    }
    return "internal";
}

fn destroyOwnedValueHandles(backend_impl: backend.Backend, value_handles: []?backend.BufferHandle, value_owned: []const bool) void {
    for (value_handles, value_owned) |maybe_handle, owned| {
        if (owned) {
            if (maybe_handle) |handle| destroyBuffer(backend_impl, handle);
        }
    }
}

fn startHandles(allocator: std.mem.Allocator, value_handles: []const ?backend.BufferHandle, ids: []const ir.ValueId) ![]backend.BufferHandle {
    const handles = try allocator.alloc(backend.BufferHandle, ids.len);
    errdefer allocator.free(handles);
    for (ids, 0..) |id, index| {
        if (id.index >= value_handles.len) return error.CommandSubmissionFailed;
        handles[index] = value_handles[id.index] orelse return error.CommandSubmissionFailed;
    }
    return handles;
}

fn storeOwnedValueHandle(
    backend_impl: backend.Backend,
    value_handles: []?backend.BufferHandle,
    value_owned: []bool,
    value_id: ir.ValueId,
    handle: backend.BufferHandle,
) backend.Error!void {
    if (value_id.index >= value_handles.len) return error.CommandSubmissionFailed;
    if (value_owned[value_id.index]) {
        if (value_handles[value_id.index]) |old| destroyBuffer(backend_impl, old);
    }
    value_handles[value_id.index] = handle;
    value_owned[value_id.index] = true;
}

fn storeBorrowedValueHandle(
    backend_impl: backend.Backend,
    value_handles: []?backend.BufferHandle,
    value_owned: []bool,
    value_id: ir.ValueId,
    handle: backend.BufferHandle,
) backend.Error!void {
    if (value_id.index >= value_handles.len) return error.CommandSubmissionFailed;
    if (value_owned[value_id.index]) {
        if (value_handles[value_id.index]) |old| destroyBuffer(backend_impl, old);
    }
    value_handles[value_id.index] = handle;
    value_owned[value_id.index] = false;
}

fn reverseIfDescending(
    backend_impl: backend.Backend,
    handle: backend.BufferHandle,
    dimension: i64,
    output_dims: []const i64,
    direction: ir.CompareOp,
) backend.Error!?backend.BufferHandle {
    return switch (direction) {
        .lt, .le => handle,
        .gt, .ge => blk: {
            const dimensions = [_]i64{dimension};
            const reversed = reverse(backend_impl, handle, &dimensions, output_dims) catch |err| {
                destroyBuffer(backend_impl, handle);
                return err;
            };
            destroyBuffer(backend_impl, handle);
            break :blk reversed;
        },
        else => blk: {
            destroyBuffer(backend_impl, handle);
            break :blk null;
        },
    };
}

fn supportedGatherAxis(instruction: ir.PlanInstruction) ?i64 {
    const start_index_map = instruction.start_index_map orelse return null;
    const slice_sizes = instruction.slice_sizes orelse return null;
    const collapsed_slice_dims = instruction.collapsed_slice_dims orelse return null;
    if (start_index_map.len != 1 or collapsed_slice_dims.len != 1) return null;
    const axis = start_index_map[0];
    if (axis < 0 or collapsed_slice_dims[0] != axis) return null;
    if (slice_sizes.len == 0 or axis >= @as(i64, @intCast(slice_sizes.len)) or slice_sizes[@intCast(axis)] != 1) return null;
    return axis;
}

fn gatherHasExplicitIndexVector(indices_dims: []const i64, start_axis_count: usize, index_vector_dim: i64) bool {
    if (index_vector_dim < 0 or index_vector_dim >= @as(i64, @intCast(indices_dims.len))) return false;
    return indices_dims[@intCast(index_vector_dim)] == @as(i64, @intCast(start_axis_count));
}

fn markUniqueAxis(seen: []bool, axis: i64) bool {
    if (axis < 0 or axis >= @as(i64, @intCast(seen.len))) return false;
    const index: usize = @intCast(axis);
    if (seen[index]) return false;
    seen[index] = true;
    return true;
}

fn validGatherShape(operand_dims: []const i64, indices_dims: []const i64, start_index_map: []const i64, collapsed_slice_dims: []const i64, operand_batching_dims: []const i64, start_indices_batching_dims: []const i64, index_vector_dim: i64, slice_sizes: []const i64, offset_dims: []const i64, output_dims: []const i64) bool {
    if (operand_dims.len > 64 or output_dims.len > 64 or start_index_map.len == 0 or slice_sizes.len != operand_dims.len) return false;
    if (start_index_map.len > 1 and !gatherHasExplicitIndexVector(indices_dims, start_index_map.len, index_vector_dim)) return false;
    if (operand_batching_dims.len != start_indices_batching_dims.len) return false;

    var gathered = [_]bool{false} ** 64;
    for (start_index_map) |axis| {
        if (!markUniqueAxis(gathered[0..operand_dims.len], axis)) return false;
    }
    for (operand_batching_dims, start_indices_batching_dims) |operand_axis, indices_axis| {
        if (!markUniqueAxis(gathered[0..operand_dims.len], operand_axis)) return false;
        if (indices_axis < 0 or indices_axis >= @as(i64, @intCast(indices_dims.len)) or indices_axis == index_vector_dim) return false;
        if (operand_dims[@intCast(operand_axis)] != indices_dims[@intCast(indices_axis)]) return false;
        if (slice_sizes[@intCast(operand_axis)] != 1) return false;
    }

    var collapsed = [_]bool{false} ** 64;
    for (collapsed_slice_dims) |axis| {
        if (!markUniqueAxis(collapsed[0..operand_dims.len], axis)) return false;
        if (slice_sizes[@intCast(axis)] != 1) return false;
    }
    for (operand_batching_dims) |axis| {
        if (!markUniqueAxis(collapsed[0..operand_dims.len], axis)) return false;
    }

    var non_collapsed_slice_rank: usize = 0;
    for (operand_dims, 0..) |dim, axis| {
        if (dim < 0 or slice_sizes[axis] < 0 or slice_sizes[axis] > dim) return false;
        if (!collapsed[axis]) non_collapsed_slice_rank += 1;
    }
    if (offset_dims.len != non_collapsed_slice_rank) return false;

    const explicit_vector = gatherHasExplicitIndexVector(indices_dims, start_index_map.len, index_vector_dim);
    const index_prefix_rank = indices_dims.len - @as(usize, if (explicit_vector) 1 else 0);
    if (output_dims.len != index_prefix_rank + non_collapsed_slice_rank) return false;

    var output_is_offset = [_]bool{false} ** 64;
    for (offset_dims) |axis| {
        if (!markUniqueAxis(output_is_offset[0..output_dims.len], axis)) return false;
    }

    var index_axis: usize = 0;
    var slice_axis: usize = 0;
    for (output_dims, 0..) |output_dim, output_axis| {
        if (output_is_offset[output_axis]) {
            while (slice_axis < operand_dims.len and collapsed[slice_axis]) slice_axis += 1;
            if (slice_axis >= slice_sizes.len or output_dim != slice_sizes[slice_axis]) return false;
            slice_axis += 1;
        } else {
            while (explicit_vector and index_axis == @as(usize, @intCast(index_vector_dim))) index_axis += 1;
            if (index_axis >= indices_dims.len or output_dim != indices_dims[index_axis]) return false;
            index_axis += 1;
        }
    }
    return true;
}

fn supportedScatterAxis(instruction: ir.PlanInstruction) ?i64 {
    const scatter_dims_to_operand_dims = instruction.scatter_dims_to_operand_dims orelse return null;
    const inserted_window_dims = instruction.inserted_window_dims orelse return null;
    const input_batching_dims = instruction.input_batching_dims orelse &.{};
    const scatter_indices_batching_dims = instruction.scatter_indices_batching_dims orelse &.{};
    if (scatter_dims_to_operand_dims.len != 1 or inserted_window_dims.len != 1) return null;
    if (input_batching_dims.len != 0 or scatter_indices_batching_dims.len != 0) return null;
    const axis = scatter_dims_to_operand_dims[0];
    if (axis < 0 or inserted_window_dims[0] != axis) return null;
    return axis;
}

fn validScatterShape(operand_dims: []const i64, indices_dims: []const i64, update_dims: []const i64, scatter_dims_to_operand_dims: []const i64, inserted_window_dims: []const i64, update_window_dims: []const i64, input_batching_dims: []const i64, scatter_indices_batching_dims: []const i64, index_vector_dim: i64, output_dims: []const i64) bool {
    if (operand_dims.len > 64 or !dimsEqual(operand_dims, output_dims)) return false;
    if (scatter_dims_to_operand_dims.len == 0 or inserted_window_dims.len + update_window_dims.len + input_batching_dims.len != operand_dims.len or input_batching_dims.len != scatter_indices_batching_dims.len) return false;
    const explicit_vector = gatherHasExplicitIndexVector(indices_dims, scatter_dims_to_operand_dims.len, index_vector_dim);
    if (scatter_dims_to_operand_dims.len > 1 and !explicit_vector) return false;

    var scatter_axes = [_]bool{false} ** 64;
    for (scatter_dims_to_operand_dims) |axis| {
        if (!markUniqueAxis(scatter_axes[0..operand_dims.len], axis)) return false;
    }
    for (input_batching_dims, scatter_indices_batching_dims) |operand_axis, indices_axis| {
        if (!markUniqueAxis(scatter_axes[0..operand_dims.len], operand_axis)) return false;
        if (indices_axis < 0 or indices_axis >= @as(i64, @intCast(indices_dims.len)) or indices_axis == index_vector_dim) return false;
        if (operand_dims[@intCast(operand_axis)] != indices_dims[@intCast(indices_axis)]) return false;
    }

    var window_axes = [_]bool{false} ** 64;
    for (inserted_window_dims) |axis| {
        if (!markUniqueAxis(window_axes[0..operand_dims.len], axis)) return false;
    }
    for (update_window_dims) |axis| {
        if (!markUniqueAxis(window_axes[0..operand_dims.len], axis)) return false;
    }
    for (input_batching_dims) |axis| {
        if (axis < 0 or axis >= @as(i64, @intCast(operand_dims.len)) or window_axes[@intCast(axis)]) return false;
    }

    const index_prefix_rank = indices_dims.len - @as(usize, if (explicit_vector) 1 else 0);
    if (update_dims.len != index_prefix_rank + update_window_dims.len) return false;
    var update_axis: usize = 0;
    for (indices_dims, 0..) |dim, axis| {
        if (explicit_vector and axis == @as(usize, @intCast(index_vector_dim))) continue;
        if (update_axis >= update_dims.len or update_dims[update_axis] != dim) return false;
        update_axis += 1;
    }
    if (update_axis != index_prefix_rank) return false;
    for (update_window_dims, 0..) |operand_axis, window_axis| {
        if (operand_axis < 0 or operand_axis >= @as(i64, @intCast(operand_dims.len))) return false;
        const dim = update_dims[index_prefix_rank + window_axis];
        if (dim < 0 or dim > operand_dims[@intCast(operand_axis)]) return false;
    }
    return true;
}

fn scatterUpdateShapeMatchesAxis(operand_dims: []const i64, indices_dims: []const i64, update_dims: []const i64, index_vector_dim: i64, axis: i64) bool {
    if (axis < 0 or axis >= @as(i64, @intCast(operand_dims.len))) return false;
    var expected_rank: usize = operand_dims.len - 1;
    for (indices_dims, 0..) |dim, dim_index| {
        if (index_vector_dim >= 0 and dim_index == @as(usize, @intCast(index_vector_dim)) and dim == 1) continue;
        expected_rank += 1;
    }
    if (update_dims.len != expected_rank) return false;

    var update_index: usize = 0;
    for (operand_dims[0..@intCast(axis)]) |dim| {
        if (update_dims[update_index] != dim) return false;
        update_index += 1;
    }
    for (indices_dims, 0..) |dim, dim_index| {
        if (index_vector_dim >= 0 and dim_index == @as(usize, @intCast(index_vector_dim)) and dim == 1) continue;
        if (update_dims[update_index] != dim) return false;
        update_index += 1;
    }
    const axis_index: usize = @intCast(axis);
    for (operand_dims[axis_index + 1 ..]) |dim| {
        if (update_dims[update_index] != dim) return false;
        update_index += 1;
    }
    return update_index == update_dims.len;
}

fn gatherOutputShapeMatchesTake(operand_dims: []const i64, indices_dims: []const i64, index_vector_dim: i64, axis: i64, output_dims: []const i64) bool {
    if (axis < 0 or axis >= @as(i64, @intCast(operand_dims.len))) return false;
    var expected_rank: usize = operand_dims.len - 1;
    for (indices_dims, 0..) |dim, dim_index| {
        if (index_vector_dim >= 0 and dim_index == @as(usize, @intCast(index_vector_dim)) and dim == 1) continue;
        expected_rank += 1;
    }
    if (output_dims.len != expected_rank) return false;

    var out_index: usize = 0;
    for (operand_dims[0..@intCast(axis)]) |dim| {
        if (output_dims[out_index] != dim) return false;
        out_index += 1;
    }
    for (indices_dims, 0..) |dim, dim_index| {
        if (index_vector_dim >= 0 and dim_index == @as(usize, @intCast(index_vector_dim)) and dim == 1) continue;
        if (output_dims[out_index] != dim) return false;
        out_index += 1;
    }
    const axis_index: usize = @intCast(axis);
    for (operand_dims[axis_index + 1 ..]) |dim| {
        if (output_dims[out_index] != dim) return false;
        out_index += 1;
    }
    return out_index == output_dims.len;
}

fn executableBinaryOp(instruction_kind: ir.PlanInstructionKind) ?ir.ElementwiseBinaryOp {
    return switch (instruction_kind) {
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
        .shift_right_arithmetic, .shift_right_logical => .shift_right_logical,
        else => null,
    };
}

fn executableUnaryOp(instruction_kind: ir.PlanInstructionKind) ?ir.ElementwiseUnaryOp {
    return switch (instruction_kind) {
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
        else => null,
    };
}

test "mlx metal backend exposes opaque backend interface" {
    const b = create();
    try std.testing.expectEqualStrings("metal_mlx", b.capabilities().name);
    const devices = try b.enumerateDevices(std.testing.allocator, 1);
    defer b.releaseDeviceDescriptors(std.testing.allocator, devices);
    try std.testing.expect(devices.len >= 1);
}

test "mlx metal backend executable bitcast_convert reinterprets resident bytes" {
    const b = create();
    const allocator = std.testing.allocator;
    const devices = try b.enumerateDevices(allocator, 1);
    defer b.releaseDeviceDescriptors(allocator, devices);
    const local_hardware_id = devices[0].local_hardware_id;
    const dims = [_]i64{2};
    const assignment = [_]i32{0};

    const values = try allocator.alloc(ir.Value, 2);
    errdefer allocator.free(values);
    values[0] = .{
        .id = .{ .index = 0 },
        .role = .parameter,
        .descriptor = .{
            .element_type = .u32,
            .dims = try allocator.dupe(i64, &dims),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };
    values[1] = .{
        .id = .{ .index = 1 },
        .role = .instruction_result,
        .descriptor = .{
            .element_type = .f32,
            .dims = try allocator.dupe(i64, &dims),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };

    var parameter_shardings = try allocator.alloc(ir.ShardingPlan, 1);
    parameter_shardings[0] = .{
        .kind = .replicated,
        .mesh_name = try allocator.dupe(u8, ""),
        .device_assignment = try allocator.dupe(i32, &assignment),
    };
    var output_shardings = try allocator.alloc(ir.ShardingPlan, 1);
    output_shardings[0] = .{
        .kind = .replicated,
        .mesh_name = try allocator.dupe(u8, ""),
        .device_assignment = try allocator.dupe(i32, &assignment),
    };

    var plan = ir.ExecutablePlan{
        .allocator = allocator,
        .options = .{
            .num_replicas = 1,
            .num_partitions = 1,
            .device_assignment = try allocator.dupe(i32, &assignment),
        },
        .values = values,
        .parameter_shardings = parameter_shardings,
        .output_shardings = output_shardings,
        .output_ids = try allocator.dupe(ir.ValueId, &.{.{ .index = 1 }}),
        .instructions = try allocator.dupe(ir.PlanInstruction, &.{
            .{
                .kind = .bitcast_convert,
                .inputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 0 }}),
                .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 1 }}),
            },
        }),
    };
    defer plan.deinit();

    const executable = (try b.compileExecutable(allocator, &plan, &.{local_hardware_id})) orelse return error.TestUnexpectedResult;
    defer b.destroyExecutable(executable);
    const compiled: *CompiledExecutable = @ptrCast(@alignCast(executable));
    try std.testing.expectEqual(backend.ProgramNodeKind.materialize, compiled.program.nodes[0].kind);

    const input_bits = [_]u32{ 0x3f800000, 0xc0000000 };
    const input_bytes = std.mem.sliceAsBytes(&input_bits);
    const arg = (try b.bufferFromHost(local_hardware_id, .u32, &dims, input_bytes)) orelse return error.TestUnexpectedResult;
    defer b.destroyBuffer(arg);
    const result = (try b.executeExecutable(allocator, executable, 0, &.{arg})) orelse return error.TestUnexpectedResult;
    defer {
        for (result.outputs) |output| b.destroyBuffer(output.handle);
        allocator.free(result.outputs);
    }
    try std.testing.expectEqual(@as(usize, 1), result.outputs.len);

    var out = [_]f32{ 0, 0 };
    try b.copyToHost(result.outputs[0].handle, std.mem.sliceAsBytes(&out));
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, -2.0), out[1], 0.0001);
}

test "mlx metal backend executable lowers reduce_window sum on device" {
    const b = create();
    const allocator = std.testing.allocator;
    const devices = try b.enumerateDevices(allocator, 1);
    defer b.releaseDeviceDescriptors(allocator, devices);
    const local_hardware_id = devices[0].local_hardware_id;
    const input_dims = [_]i64{3};
    const output_dims = [_]i64{3};
    const assignment = [_]i32{0};

    const values = try allocator.alloc(ir.Value, 2);
    errdefer allocator.free(values);
    values[0] = .{
        .id = .{ .index = 0 },
        .role = .parameter,
        .descriptor = .{
            .element_type = .f32,
            .dims = try allocator.dupe(i64, &input_dims),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };
    values[1] = .{
        .id = .{ .index = 1 },
        .role = .instruction_result,
        .descriptor = .{
            .element_type = .f32,
            .dims = try allocator.dupe(i64, &output_dims),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };

    var parameter_shardings = try allocator.alloc(ir.ShardingPlan, 1);
    parameter_shardings[0] = .{
        .kind = .replicated,
        .mesh_name = try allocator.dupe(u8, ""),
        .device_assignment = try allocator.dupe(i32, &assignment),
    };
    var output_shardings = try allocator.alloc(ir.ShardingPlan, 1);
    output_shardings[0] = .{
        .kind = .replicated,
        .mesh_name = try allocator.dupe(u8, ""),
        .device_assignment = try allocator.dupe(i32, &assignment),
    };

    var plan = ir.ExecutablePlan{
        .allocator = allocator,
        .options = .{
            .num_replicas = 1,
            .num_partitions = 1,
            .device_assignment = try allocator.dupe(i32, &assignment),
        },
        .values = values,
        .parameter_shardings = parameter_shardings,
        .output_shardings = output_shardings,
        .output_ids = try allocator.dupe(ir.ValueId, &.{.{ .index = 1 }}),
        .instructions = try allocator.dupe(ir.PlanInstruction, &.{.{
            .kind = .reduce_window_sum,
            .inputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 0 }}),
            .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 1 }}),
            .window_dimensions = try allocator.dupe(i64, &.{2}),
            .window_strides = try allocator.dupe(i64, &.{1}),
            .base_dilations = try allocator.dupe(i64, &.{1}),
            .window_dilations = try allocator.dupe(i64, &.{1}),
            .edge_padding_low = try allocator.dupe(i64, &.{1}),
            .edge_padding_high = try allocator.dupe(i64, &.{0}),
        }}),
    };
    defer plan.deinit();

    const executable = (try b.compileExecutable(allocator, &plan, &.{local_hardware_id})) orelse return error.TestUnexpectedResult;
    defer b.destroyExecutable(executable);
    const compiled: *CompiledExecutable = @ptrCast(@alignCast(executable));
    try std.testing.expectEqual(backend.ProgramNodeKind.reduction, compiled.program.nodes[0].kind);

    const input = [_]f32{ 1.5, -2.0, 4.0 };
    const arg = (try b.bufferFromHost(local_hardware_id, .f32, &input_dims, std.mem.asBytes(&input))) orelse return error.TestUnexpectedResult;
    defer b.destroyBuffer(arg);
    const result = (try b.executeExecutable(allocator, executable, 0, &.{arg})) orelse return error.TestUnexpectedResult;
    defer {
        for (result.outputs) |output| b.destroyBuffer(output.handle);
        allocator.free(result.outputs);
    }
    var out = [_]f32{ 0, 0, 0 };
    try b.copyToHost(result.outputs[0].handle, std.mem.asBytes(&out));
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), out[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, -0.5), out[1], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out[2], 0.0001);
}

test "mlx metal backend program owns while cond body subprogram descriptors" {
    const allocator = std.testing.allocator;
    const assignment = [_]i32{0};

    const scalar_f32 = ir.BufferDescriptor{
        .element_type = .f32,
        .dims = &.{},
        .device_id = 0,
        .memory_id = 0,
        .shard_index = 0,
    };
    const scalar_pred = ir.BufferDescriptor{
        .element_type = .pred,
        .dims = &.{},
        .device_id = 0,
        .memory_id = 0,
        .shard_index = 0,
    };

    const values = try allocator.alloc(ir.Value, 2);
    errdefer allocator.free(values);
    values[0] = .{
        .id = .{ .index = 0 },
        .role = .parameter,
        .descriptor = scalar_f32,
    };
    values[1] = .{
        .id = .{ .index = 1 },
        .role = .instruction_result,
        .descriptor = scalar_f32,
    };

    var parameter_shardings = try allocator.alloc(ir.ShardingPlan, 1);
    parameter_shardings[0] = .{
        .kind = .replicated,
        .mesh_name = try allocator.dupe(u8, ""),
        .device_assignment = try allocator.dupe(i32, &assignment),
    };
    var output_shardings = try allocator.alloc(ir.ShardingPlan, 1);
    output_shardings[0] = .{
        .kind = .replicated,
        .mesh_name = try allocator.dupe(u8, ""),
        .device_assignment = try allocator.dupe(i32, &assignment),
    };

    const cond_args = try allocator.dupe(ir.BufferDescriptor, &.{scalar_f32});
    const cond_returns = try allocator.dupe(ir.BufferDescriptor, &.{scalar_pred});
    const cond_terminator_ids = try allocator.dupe(ir.RegionValueId, &.{.{ .index = 1 }});
    const cond_terminator = try allocator.dupe(ir.BufferDescriptor, &.{scalar_pred});
    const cond_instruction_operands = try allocator.dupe(ir.BufferDescriptor, &.{ scalar_f32, scalar_f32 });
    const cond_instruction_results = try allocator.dupe(ir.BufferDescriptor, &.{scalar_pred});
    const cond_instruction_inputs = try allocator.dupe(ir.RegionValueId, &.{ .{ .index = 0 }, .{ .index = 0 } });
    const cond_instruction_outputs = try allocator.dupe(ir.RegionValueId, &.{.{ .index = 1 }});
    const cond_instructions = try allocator.dupe(ir.RegionInstruction, &.{.{
        .kind = .compare,
        .inputs = cond_instruction_inputs,
        .outputs = cond_instruction_outputs,
        .operand_descriptors = cond_instruction_operands,
        .result_descriptors = cond_instruction_results,
    }});
    const cond_values = try allocator.dupe(ir.RegionValue, &.{
        .{ .id = .{ .index = 0 }, .role = .argument, .descriptor = scalar_f32 },
        .{ .id = .{ .index = 1 }, .role = .instruction_result, .descriptor = scalar_pred },
    });

    const body_args = try allocator.dupe(ir.BufferDescriptor, &.{scalar_f32});
    const body_returns = try allocator.dupe(ir.BufferDescriptor, &.{scalar_f32});
    const body_terminator_ids = try allocator.dupe(ir.RegionValueId, &.{.{ .index = 1 }});
    const body_terminator = try allocator.dupe(ir.BufferDescriptor, &.{scalar_f32});
    const body_instruction_operands = try allocator.dupe(ir.BufferDescriptor, &.{ scalar_f32, scalar_f32 });
    const body_instruction_results = try allocator.dupe(ir.BufferDescriptor, &.{scalar_f32});
    const body_instruction_inputs = try allocator.dupe(ir.RegionValueId, &.{ .{ .index = 0 }, .{ .index = 0 } });
    const body_instruction_outputs = try allocator.dupe(ir.RegionValueId, &.{.{ .index = 1 }});
    const body_instructions = try allocator.dupe(ir.RegionInstruction, &.{.{
        .kind = .add,
        .inputs = body_instruction_inputs,
        .outputs = body_instruction_outputs,
        .operand_descriptors = body_instruction_operands,
        .result_descriptors = body_instruction_results,
    }});
    const body_values = try allocator.dupe(ir.RegionValue, &.{
        .{ .id = .{ .index = 0 }, .role = .argument, .descriptor = scalar_f32 },
        .{ .id = .{ .index = 1 }, .role = .instruction_result, .descriptor = scalar_f32 },
    });

    var regions = try allocator.alloc(ir.PlanRegion, 2);
    regions[0] = .{
        .id = .{ .index = 0 },
        .parent_instruction_index = 0,
        .kind = .while_cond,
        .values = cond_values,
        .argument_descriptors = cond_args,
        .instructions = cond_instructions,
        .return_descriptors = cond_returns,
        .terminator_operands = cond_terminator_ids,
        .terminator_operand_descriptors = cond_terminator,
    };
    regions[1] = .{
        .id = .{ .index = 1 },
        .parent_instruction_index = 0,
        .kind = .while_body,
        .values = body_values,
        .argument_descriptors = body_args,
        .instructions = body_instructions,
        .return_descriptors = body_returns,
        .terminator_operands = body_terminator_ids,
        .terminator_operand_descriptors = body_terminator,
    };

    var plan = ir.ExecutablePlan{
        .allocator = allocator,
        .options = .{
            .num_replicas = 1,
            .num_partitions = 1,
            .device_assignment = try allocator.dupe(i32, &assignment),
        },
        .values = values,
        .regions = regions,
        .parameter_shardings = parameter_shardings,
        .output_shardings = output_shardings,
        .output_ids = try allocator.dupe(ir.ValueId, &.{.{ .index = 1 }}),
        .instructions = try allocator.dupe(ir.PlanInstruction, &.{.{
            .kind = .while_,
            .inputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 0 }}),
            .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 1 }}),
            .region_ids = try allocator.dupe(ir.RegionId, &.{ .{ .index = 0 }, .{ .index = 1 } }),
        }}),
    };
    defer plan.deinit();

    var program = try buildBackendProgram(allocator, &plan, null);
    defer program.deinit();

    try std.testing.expectEqual(@as(usize, 1), program.nodes.len);
    try std.testing.expectEqual(backend.ProgramNodeKind.control_flow, program.nodes[0].kind);
    try std.testing.expectEqual(@as(usize, 2), program.nodes[0].subprograms.len);
    try std.testing.expectEqual(@as(usize, 2), program.subprograms.len);
    try std.testing.expectEqual(@as(usize, 1), program.control_flows.len);
    try std.testing.expectEqual(@as(?usize, 0), program.nodes[0].control_flow);
    const control_flow = program.control_flows[0];
    try std.testing.expectEqual(backend.ProgramControlFlowKind.while_loop, control_flow.kind);
    try std.testing.expectEqual(@as(usize, 0), control_flow.parent_node);
    try std.testing.expectEqual(program.nodes[0].subprograms[0], control_flow.condition_subprogram);
    try std.testing.expectEqual(program.nodes[0].subprograms[1], control_flow.body_subprogram);
    try std.testing.expectEqual(@as(usize, 1), control_flow.state_inputs.len);
    try std.testing.expectEqual(@as(usize, 1), control_flow.state_outputs.len);
    try std.testing.expectEqual(@as(u32, 1), control_flow.predicate_output.index);
    const cond = program.subprograms[program.nodes[0].subprograms[0]];
    const body = program.subprograms[program.nodes[0].subprograms[1]];
    try std.testing.expectEqual(ir.RegionKind.while_cond, cond.kind);
    try std.testing.expectEqual(ir.RegionKind.while_body, body.kind);
    try std.testing.expectEqual(@as(usize, 2), cond.values.len);
    try std.testing.expectEqual(ir.RegionValueRole.argument, cond.values[0].role);
    try std.testing.expectEqual(ir.RegionValueRole.instruction_result, cond.values[1].role);
    try std.testing.expectEqual(@as(usize, 1), cond.instructions.len);
    try std.testing.expectEqual(ir.PlanInstructionKind.compare, cond.instructions[0].kind);
    try std.testing.expectEqual(@as(usize, 2), cond.instructions[0].inputs.len);
    try std.testing.expectEqual(@as(u32, 0), cond.instructions[0].inputs[0].index);
    try std.testing.expectEqual(@as(u32, 1), cond.instructions[0].outputs[0].index);
    try std.testing.expectEqual(@as(u32, 1), cond.terminator_operands[0].index);
    try std.testing.expectEqual(@as(usize, 1), body.instructions.len);
    try std.testing.expectEqual(ir.PlanInstructionKind.add, body.instructions[0].kind);
    try std.testing.expectEqual(@as(u32, 1), body.terminator_operands[0].index);
    try std.testing.expectEqual(ir.BufferType.pred, cond.return_descriptors[0].element_type);
    try std.testing.expectEqual(ir.BufferType.f32, body.return_descriptors[0].element_type);
}

test "mlx metal backend executes f32 lt/add while loop on device" {
    const b = create();
    const allocator = std.testing.allocator;
    const devices = try b.enumerateDevices(allocator, 1);
    defer b.releaseDeviceDescriptors(allocator, devices);
    const local_hardware_id = devices[0].local_hardware_id;
    const assignment = [_]i32{0};

    const scalar_f32 = ir.BufferDescriptor{
        .element_type = .f32,
        .dims = &.{},
        .device_id = 0,
        .memory_id = 0,
        .shard_index = 0,
    };
    const scalar_pred = ir.BufferDescriptor{
        .element_type = .pred,
        .dims = &.{},
        .device_id = 0,
        .memory_id = 0,
        .shard_index = 0,
    };

    const values = try allocator.alloc(ir.Value, 2);
    errdefer allocator.free(values);
    values[0] = .{ .id = .{ .index = 0 }, .role = .parameter, .descriptor = scalar_f32 };
    values[1] = .{ .id = .{ .index = 1 }, .role = .instruction_result, .descriptor = scalar_f32 };

    var parameter_shardings = try allocator.alloc(ir.ShardingPlan, 1);
    parameter_shardings[0] = .{
        .kind = .replicated,
        .mesh_name = try allocator.dupe(u8, ""),
        .device_assignment = try allocator.dupe(i32, &assignment),
    };
    var output_shardings = try allocator.alloc(ir.ShardingPlan, 1);
    output_shardings[0] = .{
        .kind = .replicated,
        .mesh_name = try allocator.dupe(u8, ""),
        .device_assignment = try allocator.dupe(i32, &assignment),
    };

    var limit_value: f32 = 4.0;
    var step_value: f32 = 1.0;
    const cond_values = try allocator.dupe(ir.RegionValue, &.{
        .{ .id = .{ .index = 0 }, .role = .argument, .descriptor = scalar_f32 },
        .{ .id = .{ .index = 1 }, .role = .constant, .descriptor = scalar_f32, .literal = try allocator.dupe(u8, std.mem.asBytes(&limit_value)) },
        .{ .id = .{ .index = 2 }, .role = .instruction_result, .descriptor = scalar_pred },
    });
    const body_values = try allocator.dupe(ir.RegionValue, &.{
        .{ .id = .{ .index = 0 }, .role = .argument, .descriptor = scalar_f32 },
        .{ .id = .{ .index = 1 }, .role = .constant, .descriptor = scalar_f32, .literal = try allocator.dupe(u8, std.mem.asBytes(&step_value)) },
        .{ .id = .{ .index = 2 }, .role = .instruction_result, .descriptor = scalar_f32 },
    });

    var regions = try allocator.alloc(ir.PlanRegion, 2);
    regions[0] = .{
        .id = .{ .index = 0 },
        .parent_instruction_index = 0,
        .kind = .while_cond,
        .values = cond_values,
        .argument_descriptors = try allocator.dupe(ir.BufferDescriptor, &.{scalar_f32}),
        .instructions = try allocator.dupe(ir.RegionInstruction, &.{.{
            .kind = .compare,
            .inputs = try allocator.dupe(ir.RegionValueId, &.{ .{ .index = 0 }, .{ .index = 1 } }),
            .outputs = try allocator.dupe(ir.RegionValueId, &.{.{ .index = 2 }}),
            .operand_descriptors = try allocator.dupe(ir.BufferDescriptor, &.{ scalar_f32, scalar_f32 }),
            .result_descriptors = try allocator.dupe(ir.BufferDescriptor, &.{scalar_pred}),
            .compare_direction = .lt,
        }}),
        .return_descriptors = try allocator.dupe(ir.BufferDescriptor, &.{scalar_pred}),
        .terminator_operands = try allocator.dupe(ir.RegionValueId, &.{.{ .index = 2 }}),
        .terminator_operand_descriptors = try allocator.dupe(ir.BufferDescriptor, &.{scalar_pred}),
    };
    regions[1] = .{
        .id = .{ .index = 1 },
        .parent_instruction_index = 0,
        .kind = .while_body,
        .values = body_values,
        .argument_descriptors = try allocator.dupe(ir.BufferDescriptor, &.{scalar_f32}),
        .instructions = try allocator.dupe(ir.RegionInstruction, &.{.{
            .kind = .add,
            .inputs = try allocator.dupe(ir.RegionValueId, &.{ .{ .index = 0 }, .{ .index = 1 } }),
            .outputs = try allocator.dupe(ir.RegionValueId, &.{.{ .index = 2 }}),
            .operand_descriptors = try allocator.dupe(ir.BufferDescriptor, &.{ scalar_f32, scalar_f32 }),
            .result_descriptors = try allocator.dupe(ir.BufferDescriptor, &.{scalar_f32}),
        }}),
        .return_descriptors = try allocator.dupe(ir.BufferDescriptor, &.{scalar_f32}),
        .terminator_operands = try allocator.dupe(ir.RegionValueId, &.{.{ .index = 2 }}),
        .terminator_operand_descriptors = try allocator.dupe(ir.BufferDescriptor, &.{scalar_f32}),
    };

    var plan = ir.ExecutablePlan{
        .allocator = allocator,
        .options = .{
            .num_replicas = 1,
            .num_partitions = 1,
            .device_assignment = try allocator.dupe(i32, &assignment),
        },
        .values = values,
        .regions = regions,
        .parameter_shardings = parameter_shardings,
        .output_shardings = output_shardings,
        .output_ids = try allocator.dupe(ir.ValueId, &.{.{ .index = 1 }}),
        .instructions = try allocator.dupe(ir.PlanInstruction, &.{.{
            .kind = .while_,
            .inputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 0 }}),
            .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 1 }}),
            .region_ids = try allocator.dupe(ir.RegionId, &.{ .{ .index = 0 }, .{ .index = 1 } }),
        }}),
    };
    defer plan.deinit();

    const executable = (try b.compileExecutable(allocator, &plan, &.{local_hardware_id})) orelse return error.TestUnexpectedResult;
    defer b.destroyExecutable(executable);
    var state: f32 = 0.0;
    const state_buffer = (try b.bufferFromHost(local_hardware_id, .f32, &.{}, std.mem.asBytes(&state))) orelse return error.TestUnexpectedResult;
    defer b.destroyBuffer(state_buffer);
    const result = (try b.executeExecutable(allocator, executable, 0, &.{state_buffer})) orelse return error.TestUnexpectedResult;
    defer {
        for (result.outputs) |output| b.destroyBuffer(output.handle);
        allocator.free(result.outputs);
    }
    var actual: f32 = 0.0;
    try b.copyToHost(result.outputs[0].handle, std.mem.asBytes(&actual));
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), actual, 0.0001);
}

test "mlx metal backend executable runs resident device buffers" {
    const b = create();
    const allocator = std.testing.allocator;
    const devices = try b.enumerateDevices(allocator, 1);
    defer b.releaseDeviceDescriptors(allocator, devices);
    const local_hardware_id = devices[0].local_hardware_id;
    const dims = [_]i64{4};
    const assignment = [_]i32{0};

    const values = try allocator.alloc(ir.Value, 4);
    errdefer allocator.free(values);
    for (values, 0..) |*value, i| {
        value.* = .{
            .id = .{ .index = @intCast(i) },
            .role = if (i == 0) .parameter else .instruction_result,
            .descriptor = .{
                .element_type = .u8,
                .dims = try allocator.dupe(i64, &dims),
                .device_id = 0,
                .memory_id = 0,
                .shard_index = 0,
            },
        };
    }

    var parameter_shardings = try allocator.alloc(ir.ShardingPlan, 1);
    parameter_shardings[0] = .{
        .kind = .replicated,
        .mesh_name = try allocator.dupe(u8, ""),
        .device_assignment = try allocator.dupe(i32, &assignment),
    };
    var output_shardings = try allocator.alloc(ir.ShardingPlan, 1);
    output_shardings[0] = .{
        .kind = .replicated,
        .mesh_name = try allocator.dupe(u8, ""),
        .device_assignment = try allocator.dupe(i32, &assignment),
    };

    var plan = ir.ExecutablePlan{
        .allocator = allocator,
        .options = .{
            .num_replicas = 1,
            .num_partitions = 1,
            .device_assignment = try allocator.dupe(i32, &assignment),
        },
        .values = values,
        .parameter_shardings = parameter_shardings,
        .output_shardings = output_shardings,
        .output_ids = try allocator.dupe(ir.ValueId, &.{.{ .index = 3 }}),
        .instructions = try allocator.dupe(ir.PlanInstruction, &.{
            .{
                .kind = .constant,
                .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 1 }}),
                .literal = try allocator.dupe(u8, &.{ 2, 3, 4, 5 }),
            },
            .{
                .kind = .add,
                .inputs = try allocator.dupe(ir.ValueId, &.{ .{ .index = 0 }, .{ .index = 1 } }),
                .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 2 }}),
            },
            .{
                .kind = .multiply,
                .inputs = try allocator.dupe(ir.ValueId, &.{ .{ .index = 2 }, .{ .index = 1 } }),
                .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 3 }}),
            },
        }),
    };
    defer plan.deinit();

    const executable = (try b.compileExecutable(allocator, &plan, &.{local_hardware_id})) orelse return error.TestUnexpectedResult;
    defer b.destroyExecutable(executable);
    const compiled_programs_enabled = programCompileEnabled();
    const compiled: *CompiledExecutable = @ptrCast(@alignCast(executable));
    try std.testing.expectEqual(@as(usize, 3), compiled.program.nodes.len);
    try std.testing.expectEqual(@as(usize, 3), compiled.program.schedule.len);
    try std.testing.expectEqual(backend.ProgramScheduleKind.node, compiled.program.schedule[0].kind);
    try std.testing.expectEqual(@as(usize, 0), compiled.program.schedule[0].index);
    try std.testing.expectEqual(backend.ProgramScheduleKind.fusion_group, compiled.program.schedule[1].kind);
    try std.testing.expectEqual(@as(usize, 0), compiled.program.schedule[1].index);
    try std.testing.expectEqual(@as(usize, 2), compiled.program.schedule[1].count);
    try std.testing.expectEqual(backend.ProgramScheduleKind.materialization_boundary, compiled.program.schedule[2].kind);
    try std.testing.expectEqual(@as(usize, 0), compiled.program.schedule[2].index);
    try std.testing.expectEqual(@as(usize, 1), compiled.program.schedule[2].count);
    try std.testing.expectEqual(backend.ProgramNodeKind.constant, compiled.program.nodes[0].kind);
    try std.testing.expectEqual(backend.ProgramNodeKind.elementwise, compiled.program.nodes[1].kind);
    try std.testing.expectEqual(backend.ProgramNodeKind.elementwise, compiled.program.nodes[2].kind);
    try std.testing.expect(compiled.program.nodes[0].materializes);
    try std.testing.expectEqual(@as(?usize, null), compiled.program.nodes[0].fusion_group);
    try std.testing.expectEqual(@as(?usize, 0), compiled.program.nodes[1].fusion_group);
    try std.testing.expectEqual(@as(?usize, 0), compiled.program.nodes[2].fusion_group);
    try std.testing.expectEqual(@as(usize, 1), compiled.program.fusion_group_count);
    try std.testing.expectEqual(@as(usize, 1), compiled.program.fusion_groups.len);
    try std.testing.expectEqual(@as(usize, 0), compiled.program.fusion_groups[0].id);
    try std.testing.expectEqual(backend.FusionGroupKind.view_elementwise, compiled.program.fusion_groups[0].kind);
    try std.testing.expectEqual(@as(usize, 1), compiled.program.fusion_groups[0].first_node);
    try std.testing.expectEqual(@as(usize, 2), compiled.program.fusion_groups[0].last_node);
    try std.testing.expectEqual(@as(usize, 2), compiled.program.fusion_groups[0].node_count);
    try std.testing.expectEqual(@as(usize, 2), compiled.program.fusion_groups[0].node_indices.len);
    try std.testing.expectEqual(@as(usize, 1), compiled.program.fusion_groups[0].node_indices[0]);
    try std.testing.expectEqual(@as(usize, 2), compiled.program.fusion_groups[0].node_indices[1]);
    try std.testing.expectEqual(@as(usize, 2), compiled.program.fusion_groups[0].input_values.len);
    try std.testing.expectEqual(@as(usize, 0), compiled.program.fusion_groups[0].input_values[0].index);
    try std.testing.expectEqual(@as(usize, 1), compiled.program.fusion_groups[0].input_values[1].index);
    try std.testing.expectEqual(@as(usize, 1), compiled.program.fusion_groups[0].output_values.len);
    try std.testing.expectEqual(@as(usize, 3), compiled.program.fusion_groups[0].output_values[0].index);
    try std.testing.expectEqual(@as(usize, 1), compiled.program.materialization_boundaries.len);
    try std.testing.expectEqual(@as(usize, 3), compiled.program.materialization_boundaries[0].value_id.index);
    try std.testing.expectEqual(backend.MaterializationReason.pjrt_output, compiled.program.materialization_boundaries[0].reason);
    try std.testing.expectEqual(@as(usize, 3), compiled.program.edges.len);
    try std.testing.expectEqual(@as(usize, 1), compiled.program.edges[0].value_id.index);
    try std.testing.expectEqual(@as(usize, 0), compiled.program.edges[0].from_node);
    try std.testing.expectEqual(@as(usize, 1), compiled.program.edges[0].to_node);
    try std.testing.expectEqual(@as(usize, 2), compiled.program.edges[1].value_id.index);
    try std.testing.expectEqual(@as(usize, 1), compiled.program.edges[1].from_node);
    try std.testing.expectEqual(@as(usize, 2), compiled.program.edges[1].to_node);
    try std.testing.expectEqual(@as(usize, 1), compiled.program.edges[2].value_id.index);
    try std.testing.expectEqual(@as(usize, 0), compiled.program.edges[2].from_node);
    try std.testing.expectEqual(@as(usize, 2), compiled.program.edges[2].to_node);
    try std.testing.expectEqual(@as(usize, 4), compiled.program.values.len);
    try std.testing.expectEqual(@as(usize, 0), compiled.program.values[0].value_id.index);
    try std.testing.expectEqual(@as(?usize, null), compiled.program.values[0].producer_node);
    try std.testing.expectEqual(@as(?usize, 1), compiled.program.values[0].last_use_node);
    try std.testing.expect(!compiled.program.values[0].is_output);
    try std.testing.expectEqual(@as(?usize, null), compiled.program.values[0].materialization_boundary);
    try std.testing.expectEqual(@as(?usize, 0), compiled.program.values[1].producer_node);
    try std.testing.expectEqual(@as(?usize, 2), compiled.program.values[1].last_use_node);
    try std.testing.expectEqual(@as(?usize, 1), compiled.program.values[2].producer_node);
    try std.testing.expectEqual(@as(?usize, 2), compiled.program.values[2].last_use_node);
    try std.testing.expectEqual(@as(?usize, 2), compiled.program.values[3].producer_node);
    try std.testing.expect(compiled.program.values[3].is_output);
    try std.testing.expectEqual(@as(?usize, 0), compiled.program.values[3].materialization_boundary);

    var stats = b.executableStats(executable);
    try std.testing.expectEqual(@as(usize, 1), stats.resident_constant_count);
    try std.testing.expectEqual(@as(usize, 4), stats.resident_constant_bytes);
    try std.testing.expectEqual(@as(usize, 4), stats.program_value_count);
    try std.testing.expectEqual(@as(usize, 3), stats.program_node_count);
    try std.testing.expectEqual(@as(usize, 3), stats.program_edge_count);
    try std.testing.expectEqual(@as(usize, 3), stats.program_schedule_item_count);
    try std.testing.expectEqual(@as(usize, 1), stats.program_fusion_group_count);
    try std.testing.expectEqual(@as(usize, 1), stats.program_materialization_boundary_count);
    try std.testing.expectEqual(@as(usize, 1), stats.program_planned_release_count);
    try std.testing.expectEqual(@as(usize, 4), stats.program_planned_release_bytes);
    try std.testing.expectEqual(@as(usize, 4), stats.program_peak_live_value_count);
    try std.testing.expectEqual(@as(usize, 16), stats.program_peak_live_bytes);
    try std.testing.expectEqual(@as(usize, 1), stats.program_device_count);
    try std.testing.expectEqual(@as(usize, std.math.maxInt(usize)), stats.last_execute_device_index);
    try std.testing.expectEqual(@as(i32, -1), stats.last_execute_local_hardware_id);
    try std.testing.expectEqual(@as(usize, 0), stats.execute_count);
    try std.testing.expectEqual(@as(usize, 0), stats.compiled_program_execute_count);
    try std.testing.expectEqual(@as(usize, 0), stats.compiled_program_output_count);
    try std.testing.expectEqual(@as(usize, 0), stats.fusion_group_execute_count);
    try std.testing.expectEqual(@as(usize, 0), stats.materialization_eval_count);
    try std.testing.expectEqual(@as(usize, 0), stats.materialization_eval_buffer_count);
    try std.testing.expectEqual(@as(usize, 0), stats.released_intermediate_count);
    try std.testing.expectEqual(@as(usize, 0), stats.borrowed_constant_nodes);

    const lhs = (try b.bufferFromHost(local_hardware_id, .u8, &dims, &.{ 1, 2, 3, 4 })) orelse return error.TestUnexpectedResult;
    defer b.destroyBuffer(lhs);

    try std.testing.expectError(error.CommandSubmissionFailed, b.executeExecutable(allocator, executable, 1, &.{lhs}));
    try std.testing.expectError(error.CommandSubmissionFailed, b.executeExecutable(allocator, executable, 0, &.{}));
    try std.testing.expectError(error.CommandSubmissionFailed, b.executeExecutable(allocator, executable, 0, &.{ lhs, lhs }));
    stats = b.executableStats(executable);
    try std.testing.expectEqual(@as(usize, 0), stats.execute_count);
    try std.testing.expectEqual(@as(usize, std.math.maxInt(usize)), stats.last_execute_device_index);
    try std.testing.expectEqual(@as(i32, -1), stats.last_execute_local_hardware_id);

    const result = (try b.executeExecutable(allocator, executable, 0, &.{lhs})) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(backend.ExecutionCompletionKind.completed, result.completion.kind);
    const outputs = result.outputs;
    defer allocator.free(outputs);
    defer {
        for (outputs) |output| b.destroyBuffer(output.handle);
    }

    try std.testing.expectEqual(@as(usize, 1), outputs.len);
    try std.testing.expectEqual(ir.BufferType.u8, outputs[0].element_type);
    try std.testing.expectEqualSlices(i64, &dims, outputs[0].dims);
    try std.testing.expectEqual(@as(u1, 0), @intFromBool(bufferRef(outputs[0].handle).hasHostShadow()));
    var actual: [4]u8 = undefined;
    try b.copyToHost(outputs[0].handle, &actual);
    try std.testing.expectEqualSlices(u8, &.{ 6, 15, 28, 45 }, &actual);

    stats = b.executableStats(executable);
    try std.testing.expectEqual(@as(usize, 1), stats.resident_constant_count);
    try std.testing.expectEqual(@as(usize, 4), stats.resident_constant_bytes);
    try std.testing.expectEqual(@as(usize, 1), stats.execute_count);
    try std.testing.expectEqual(@as(usize, 0), stats.last_execute_device_index);
    try std.testing.expectEqual(local_hardware_id, stats.last_execute_local_hardware_id);
    if (compiled_programs_enabled) {
        try std.testing.expectEqual(@as(usize, 1), stats.compiled_program_execute_count);
        try std.testing.expectEqual(@as(usize, 1), stats.compiled_program_output_count);
        try std.testing.expect(stats.fusion_group_execute_count <= 1);
        try std.testing.expectEqual(@as(usize, 0), stats.materialization_eval_count);
        try std.testing.expectEqual(@as(usize, 0), stats.materialization_eval_buffer_count);
        try std.testing.expect(stats.released_intermediate_count <= 1);
        try std.testing.expect(stats.borrowed_constant_nodes <= 1);
    } else {
        try std.testing.expectEqual(@as(usize, 0), stats.compiled_program_execute_count);
        try std.testing.expectEqual(@as(usize, 0), stats.compiled_program_output_count);
        try std.testing.expectEqual(@as(usize, 1), stats.fusion_group_execute_count);
        try std.testing.expectEqual(@as(usize, 1), stats.materialization_eval_count);
        try std.testing.expectEqual(@as(usize, 1), stats.materialization_eval_buffer_count);
        try std.testing.expectEqual(@as(usize, 1), stats.released_intermediate_count);
        try std.testing.expectEqual(@as(usize, 1), stats.borrowed_constant_nodes);
    }

    const second_lhs = (try b.bufferFromHost(local_hardware_id, .u8, &dims, &.{ 2, 4, 6, 8 })) orelse return error.TestUnexpectedResult;
    defer b.destroyBuffer(second_lhs);
    const second_result = (try b.executeExecutable(allocator, executable, 0, &.{second_lhs})) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(backend.ExecutionCompletionKind.completed, second_result.completion.kind);
    const second_outputs = second_result.outputs;
    defer allocator.free(second_outputs);
    defer {
        for (second_outputs) |output| b.destroyBuffer(output.handle);
    }

    try std.testing.expectEqual(@as(usize, 1), second_outputs.len);
    try std.testing.expectEqual(@as(u1, 0), @intFromBool(bufferRef(second_outputs[0].handle).hasHostShadow()));
    try b.copyToHost(second_outputs[0].handle, &actual);
    try std.testing.expectEqualSlices(u8, &.{ 8, 21, 40, 65 }, &actual);

    stats = b.executableStats(executable);
    try std.testing.expectEqual(@as(usize, 1), stats.resident_constant_count);
    try std.testing.expectEqual(@as(usize, 4), stats.resident_constant_bytes);
    try std.testing.expectEqual(@as(usize, 2), stats.execute_count);
    try std.testing.expectEqual(@as(usize, 0), stats.last_execute_device_index);
    try std.testing.expectEqual(local_hardware_id, stats.last_execute_local_hardware_id);
    if (compiled_programs_enabled) {
        try std.testing.expectEqual(@as(usize, 2), stats.compiled_program_execute_count);
        try std.testing.expectEqual(@as(usize, 2), stats.compiled_program_output_count);
        try std.testing.expect(stats.fusion_group_execute_count <= 1);
        try std.testing.expectEqual(@as(usize, 0), stats.materialization_eval_count);
        try std.testing.expectEqual(@as(usize, 0), stats.materialization_eval_buffer_count);
        try std.testing.expect(stats.released_intermediate_count <= 1);
        try std.testing.expect(stats.borrowed_constant_nodes <= 1);
    } else {
        try std.testing.expectEqual(@as(usize, 0), stats.compiled_program_execute_count);
        try std.testing.expectEqual(@as(usize, 0), stats.compiled_program_output_count);
        try std.testing.expectEqual(@as(usize, 2), stats.fusion_group_execute_count);
        try std.testing.expectEqual(@as(usize, 2), stats.materialization_eval_count);
        try std.testing.expectEqual(@as(usize, 2), stats.materialization_eval_buffer_count);
        try std.testing.expectEqual(@as(usize, 2), stats.released_intermediate_count);
        try std.testing.expectEqual(@as(usize, 2), stats.borrowed_constant_nodes);
    }
}

test "mlx metal backend executable lowers tuple get_tuple_element without materializing tuple buffers" {
    const b = create();
    const allocator = std.testing.allocator;
    const devices = try b.enumerateDevices(allocator, 1);
    defer b.releaseDeviceDescriptors(allocator, devices);
    const local_hardware_id = devices[0].local_hardware_id;
    const dims = [_]i64{2};
    const assignment = [_]i32{0};

    const values = try allocator.alloc(ir.Value, 4);
    errdefer allocator.free(values);
    for (values[0..2], 0..) |*value, i| {
        value.* = .{
            .id = .{ .index = @intCast(i) },
            .role = .parameter,
            .descriptor = .{
                .element_type = .f32,
                .dims = try allocator.dupe(i64, &dims),
                .device_id = 0,
                .memory_id = 0,
                .shard_index = 0,
            },
        };
    }
    values[2] = .{
        .id = .{ .index = 2 },
        .role = .instruction_result,
        .descriptor = .{
            .element_type = .invalid,
            .dims = try allocator.dupe(i64, &.{}),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
        .storage = .tuple,
        .elements = try allocator.dupe(ir.ValueId, &.{ .{ .index = 0 }, .{ .index = 1 } }),
    };
    values[3] = .{
        .id = .{ .index = 3 },
        .role = .instruction_result,
        .descriptor = .{
            .element_type = .f32,
            .dims = try allocator.dupe(i64, &dims),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };

    const parameter_shardings = try allocator.alloc(ir.ShardingPlan, 2);
    for (parameter_shardings) |*sharding| {
        sharding.* = .{
            .kind = .replicated,
            .mesh_name = try allocator.dupe(u8, ""),
            .device_assignment = try allocator.dupe(i32, &assignment),
        };
    }
    const output_shardings = try allocator.alloc(ir.ShardingPlan, 1);
    output_shardings[0] = .{
        .kind = .replicated,
        .mesh_name = try allocator.dupe(u8, ""),
        .device_assignment = try allocator.dupe(i32, &assignment),
    };

    var plan = ir.ExecutablePlan{
        .allocator = allocator,
        .options = .{
            .num_replicas = 1,
            .num_partitions = 1,
            .device_assignment = try allocator.dupe(i32, &assignment),
        },
        .values = values,
        .parameter_shardings = parameter_shardings,
        .output_shardings = output_shardings,
        .output_ids = try allocator.dupe(ir.ValueId, &.{.{ .index = 3 }}),
        .instructions = try allocator.dupe(ir.PlanInstruction, &.{
            .{
                .kind = .tuple,
                .inputs = try allocator.dupe(ir.ValueId, &.{ .{ .index = 0 }, .{ .index = 1 } }),
                .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 2 }}),
            },
            .{
                .kind = .get_tuple_element,
                .inputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 2 }}),
                .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 3 }}),
                .tuple_index = 1,
            },
        }),
    };
    defer plan.deinit();

    const executable = (try b.compileExecutable(allocator, &plan, &.{local_hardware_id})) orelse return error.TestUnexpectedResult;
    defer b.destroyExecutable(executable);
    const compiled: *CompiledExecutable = @ptrCast(@alignCast(executable));
    try std.testing.expectEqual(backend.ProgramNodeKind.structural, compiled.program.nodes[0].kind);
    try std.testing.expectEqual(backend.ProgramNodeKind.structural, compiled.program.nodes[1].kind);
    try std.testing.expectEqual(@as(usize, 2), compiled.program.nodes[1].inputs.len);
    try std.testing.expectEqual(@as(usize, 1), compiled.program.nodes[1].inputs[1].index);

    const lhs_data = [_]f32{ 1.0, 2.0 };
    const rhs_data = [_]f32{ 3.0, 4.0 };
    const lhs = (try b.bufferFromHost(local_hardware_id, .f32, &dims, std.mem.asBytes(&lhs_data))) orelse return error.TestUnexpectedResult;
    defer b.destroyBuffer(lhs);
    const rhs = (try b.bufferFromHost(local_hardware_id, .f32, &dims, std.mem.asBytes(&rhs_data))) orelse return error.TestUnexpectedResult;
    defer b.destroyBuffer(rhs);

    const result = (try b.executeExecutable(allocator, executable, 0, &.{ lhs, rhs })) orelse return error.TestUnexpectedResult;
    const outputs = result.outputs;
    defer allocator.free(outputs);
    defer for (outputs) |output| b.destroyBuffer(output.handle);

    try std.testing.expectEqual(@as(usize, 1), outputs.len);
    var actual: [2]f32 = undefined;
    try b.copyToHost(outputs[0].handle, std.mem.asBytes(&actual));
    try std.testing.expectEqualSlices(f32, &rhs_data, &actual);
}

test "mlx metal backend lowers metadata custom call and optimization barrier on device" {
    const b = create();
    const allocator = std.testing.allocator;
    const devices = try b.enumerateDevices(allocator, 1);
    defer b.releaseDeviceDescriptors(allocator, devices);
    const local_hardware_id = devices[0].local_hardware_id;
    const assignment = [_]i32{local_hardware_id};
    const dims = [_]i64{4};

    const values = try allocator.alloc(ir.Value, 5);
    for (values, 0..) |*value, index| {
        value.* = .{
            .id = .{ .index = @intCast(index) },
            .role = if (index < 2) .parameter else .instruction_result,
            .descriptor = .{
                .element_type = .u8,
                .dims = try allocator.dupe(i64, &dims),
                .device_id = 0,
                .memory_id = 0,
                .shard_index = 0,
            },
        };
    }

    const parameter_shardings = try allocator.alloc(ir.ShardingPlan, 2);
    for (parameter_shardings) |*sharding| {
        sharding.* = .{
            .kind = .replicated,
            .mesh_name = try allocator.dupe(u8, "test"),
            .device_assignment = try allocator.dupe(i32, &assignment),
        };
    }

    var plan = ir.ExecutablePlan{
        .allocator = allocator,
        .options = .{
            .num_replicas = 1,
            .num_partitions = 1,
            .device_assignment = try allocator.dupe(i32, &assignment),
        },
        .values = values,
        .parameter_shardings = parameter_shardings,
        .output_shardings = try allocator.alloc(ir.ShardingPlan, 0),
        .output_ids = try allocator.dupe(ir.ValueId, &.{ .{ .index = 3 }, .{ .index = 4 } }),
        .instructions = try allocator.dupe(ir.PlanInstruction, &.{
            .{
                .kind = .optimization_barrier,
                .inputs = try allocator.dupe(ir.ValueId, &.{ .{ .index = 0 }, .{ .index = 1 } }),
                .outputs = try allocator.dupe(ir.ValueId, &.{ .{ .index = 2 }, .{ .index = 3 } }),
                .dims = try allocator.dupe(i64, &dims),
            },
            .{
                .kind = .custom_call,
                .inputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 2 }}),
                .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 4 }}),
                .dims = try allocator.dupe(i64, &dims),
                .custom_call_target = try allocator.dupe(u8, "annotate_device_placement"),
            },
        }),
    };
    defer plan.deinit();

    const executable = (try b.compileExecutable(allocator, &plan, &assignment)) orelse return error.TestUnexpectedResult;
    defer b.destroyExecutable(executable);

    const lhs = (try b.bufferFromHost(local_hardware_id, .u8, &dims, &.{ 1, 2, 3, 4 })) orelse return error.TestUnexpectedResult;
    defer b.destroyBuffer(lhs);
    const rhs = (try b.bufferFromHost(local_hardware_id, .u8, &dims, &.{ 10, 20, 30, 40 })) orelse return error.TestUnexpectedResult;
    defer b.destroyBuffer(rhs);

    const result = (try b.executeExecutable(allocator, executable, 0, &.{ lhs, rhs })) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(backend.ExecutionCompletionKind.completed, result.completion.kind);
    const outputs = result.outputs;
    defer allocator.free(outputs);
    defer {
        for (outputs) |output| b.destroyBuffer(output.handle);
    }

    try std.testing.expectEqual(@as(usize, 2), outputs.len);
    var barrier_rhs: [4]u8 = undefined;
    var annotated_lhs: [4]u8 = undefined;
    try b.copyToHost(outputs[0].handle, &barrier_rhs);
    try b.copyToHost(outputs[1].handle, &annotated_lhs);
    try std.testing.expectEqualSlices(u8, &.{ 10, 20, 30, 40 }, &barrier_rhs);
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4 }, &annotated_lhs);
}

test "mlx metal backend rejects gspmd custom call targets precisely" {
    const b = create();
    const allocator = std.testing.allocator;
    const assignment = [_]i32{0};
    const dims = [_]i64{4};

    const values = try allocator.alloc(ir.Value, 2);
    for (values, 0..) |*value, index| {
        value.* = .{
            .id = .{ .index = @intCast(index) },
            .role = if (index == 0) .parameter else .instruction_result,
            .descriptor = .{
                .element_type = .u8,
                .dims = try allocator.dupe(i64, &dims),
                .device_id = 0,
                .memory_id = 0,
                .shard_index = 0,
            },
        };
    }

    const parameter_shardings = try allocator.alloc(ir.ShardingPlan, 1);
    parameter_shardings[0] = .{
        .kind = .replicated,
        .mesh_name = try allocator.dupe(u8, "test"),
        .device_assignment = try allocator.dupe(i32, &assignment),
    };

    var plan = ir.ExecutablePlan{
        .allocator = allocator,
        .options = .{
            .num_replicas = 1,
            .num_partitions = 1,
            .device_assignment = try allocator.dupe(i32, &assignment),
        },
        .values = values,
        .parameter_shardings = parameter_shardings,
        .output_shardings = try allocator.alloc(ir.ShardingPlan, 0),
        .output_ids = try allocator.dupe(ir.ValueId, &.{.{ .index = 1 }}),
        .instructions = try allocator.dupe(ir.PlanInstruction, &.{.{
            .kind = .custom_call,
            .inputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 0 }}),
            .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 1 }}),
            .dims = try allocator.dupe(i64, &dims),
            .custom_call_target = try allocator.dupe(u8, "Sharding"),
        }}),
    };
    defer plan.deinit();

    const maybe_executable = try b.compileExecutable(allocator, &plan, &assignment);
    if (maybe_executable) |executable| b.destroyExecutable(executable);
    try std.testing.expect(maybe_executable == null);

    var diagnostics = std.Io.Writer.Allocating.init(allocator);
    defer diagnostics.deinit();
    try b.writeExecutableLoweringDiagnostic(&plan, &assignment, &diagnostics.writer);
    const message = diagnostics.writer.buffered();
    try std.testing.expect(std.mem.indexOf(u8, message, "op=custom_call") != null);
    try std.testing.expect(std.mem.indexOf(u8, message, "feature=Sharding") != null);
}

test "mlx metal backend runs registered binary custom call on device buffers" {
    const b = create();
    const allocator = std.testing.allocator;
    const local_hardware_id: i32 = 0;
    const assignment = [_]i32{local_hardware_id};
    const dims = [_]i64{2};
    const target = "pjrtx.test.binary_add";

    try b.registerCustomCall(.{
        .target = target,
        .kind = .binary,
        .binary_op = .add,
    });
    defer b.unregisterCustomCall(target);

    const values = try allocator.alloc(ir.Value, 3);
    for (values, 0..) |*value, index| {
        value.* = .{
            .id = .{ .index = @intCast(index) },
            .role = if (index < 2) .parameter else .instruction_result,
            .descriptor = .{
                .element_type = .f32,
                .dims = try allocator.dupe(i64, &dims),
                .device_id = 0,
                .memory_id = 0,
                .shard_index = 0,
            },
        };
    }

    const parameter_shardings = try allocator.alloc(ir.ShardingPlan, 2);
    for (parameter_shardings) |*sharding| {
        sharding.* = .{
            .kind = .replicated,
            .mesh_name = try allocator.dupe(u8, "test"),
            .device_assignment = try allocator.dupe(i32, &assignment),
        };
    }

    var plan = ir.ExecutablePlan{
        .allocator = allocator,
        .options = .{
            .num_replicas = 1,
            .num_partitions = 1,
            .device_assignment = try allocator.dupe(i32, &assignment),
        },
        .values = values,
        .parameter_shardings = parameter_shardings,
        .output_shardings = try allocator.alloc(ir.ShardingPlan, 0),
        .output_ids = try allocator.dupe(ir.ValueId, &.{.{ .index = 2 }}),
        .instructions = try allocator.dupe(ir.PlanInstruction, &.{.{
            .kind = .custom_call,
            .inputs = try allocator.dupe(ir.ValueId, &.{ .{ .index = 0 }, .{ .index = 1 } }),
            .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 2 }}),
            .dims = try allocator.dupe(i64, &dims),
            .custom_call_target = try allocator.dupe(u8, target),
        }}),
    };
    defer plan.deinit();

    const executable = (try b.compileExecutable(allocator, &plan, &assignment)) orelse return error.TestUnexpectedResult;
    defer b.destroyExecutable(executable);

    const lhs_data = [_]f32{ 1.5, 2.25 };
    const rhs_data = [_]f32{ 4.0, -0.25 };
    const lhs = (try b.bufferFromHost(local_hardware_id, .f32, &dims, std.mem.asBytes(&lhs_data))) orelse return error.TestUnexpectedResult;
    defer b.destroyBuffer(lhs);
    const rhs = (try b.bufferFromHost(local_hardware_id, .f32, &dims, std.mem.asBytes(&rhs_data))) orelse return error.TestUnexpectedResult;
    defer b.destroyBuffer(rhs);

    const result = (try b.executeExecutable(allocator, executable, 0, &.{ lhs, rhs })) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(backend.ExecutionCompletionKind.completed, result.completion.kind);
    const outputs = result.outputs;
    defer allocator.free(outputs);
    defer {
        for (outputs) |output| b.destroyBuffer(output.handle);
    }

    var output: [2]f32 = undefined;
    try b.copyToHost(outputs[0].handle, std.mem.asBytes(&output));
    try std.testing.expectApproxEqAbs(@as(f32, 5.5), output[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), output[1], 0.0001);
}

test "mlx metal backend executable materializes iota on device" {
    const b = create();
    const allocator = std.testing.allocator;
    const devices = try b.enumerateDevices(allocator, 1);
    defer b.releaseDeviceDescriptors(allocator, devices);
    const local_hardware_id = devices[0].local_hardware_id;
    const dims = [_]i64{ 2, 3 };
    const assignment = [_]i32{0};

    const values = try allocator.alloc(ir.Value, 1);
    errdefer allocator.free(values);
    values[0] = .{
        .id = .{ .index = 0 },
        .role = .instruction_result,
        .descriptor = .{
            .element_type = .f32,
            .dims = try allocator.dupe(i64, &dims),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };

    const parameter_shardings = try allocator.alloc(ir.ShardingPlan, 0);
    const output_shardings = try allocator.alloc(ir.ShardingPlan, 1);
    output_shardings[0] = .{
        .kind = .replicated,
        .mesh_name = try allocator.dupe(u8, ""),
        .device_assignment = try allocator.dupe(i32, &assignment),
    };

    var plan = ir.ExecutablePlan{
        .allocator = allocator,
        .options = .{
            .num_replicas = 1,
            .num_partitions = 1,
            .device_assignment = try allocator.dupe(i32, &assignment),
        },
        .values = values,
        .parameter_shardings = parameter_shardings,
        .output_shardings = output_shardings,
        .output_ids = try allocator.dupe(ir.ValueId, &.{.{ .index = 0 }}),
        .instructions = try allocator.dupe(ir.PlanInstruction, &.{.{
            .kind = .iota,
            .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 0 }}),
            .dims = try allocator.dupe(i64, &dims),
            .iota_dimension = 1,
        }}),
    };
    defer plan.deinit();

    const executable = (try b.compileExecutable(allocator, &plan, &.{local_hardware_id})) orelse return error.TestUnexpectedResult;
    defer b.destroyExecutable(executable);

    const result = (try b.executeExecutable(allocator, executable, 0, &.{})) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(backend.ExecutionCompletionKind.completed, result.completion.kind);
    const outputs = result.outputs;
    defer allocator.free(outputs);
    defer {
        for (outputs) |output| b.destroyBuffer(output.handle);
    }

    try std.testing.expectEqual(@as(usize, 1), outputs.len);
    try std.testing.expectEqual(ir.BufferType.f32, outputs[0].element_type);
    try std.testing.expectEqualSlices(i64, &dims, outputs[0].dims);
    var actual: [6]f32 = undefined;
    try b.copyToHost(outputs[0].handle, std.mem.asBytes(&actual));
    try std.testing.expectEqualSlices(f32, &.{ 0.0, 1.0, 2.0, 0.0, 1.0, 2.0 }, &actual);
}

test "mlx metal backend executable materializes partition_id on device" {
    const b = create();
    const allocator = std.testing.allocator;
    const devices = try b.enumerateDevices(allocator, 1);
    defer b.releaseDeviceDescriptors(allocator, devices);
    const local_hardware_id = devices[0].local_hardware_id;
    const assignment = [_]i32{0};

    const values = try allocator.alloc(ir.Value, 1);
    errdefer allocator.free(values);
    values[0] = .{
        .id = .{ .index = 0 },
        .role = .instruction_result,
        .descriptor = .{
            .element_type = .u32,
            .dims = try allocator.alloc(i64, 0),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };

    const parameter_shardings = try allocator.alloc(ir.ShardingPlan, 0);
    const output_shardings = try allocator.alloc(ir.ShardingPlan, 1);
    output_shardings[0] = .{
        .kind = .replicated,
        .mesh_name = try allocator.dupe(u8, ""),
        .device_assignment = try allocator.dupe(i32, &assignment),
    };

    var plan = ir.ExecutablePlan{
        .allocator = allocator,
        .options = .{
            .num_replicas = 1,
            .num_partitions = 1,
            .device_assignment = try allocator.dupe(i32, &assignment),
        },
        .values = values,
        .parameter_shardings = parameter_shardings,
        .output_shardings = output_shardings,
        .output_ids = try allocator.dupe(ir.ValueId, &.{.{ .index = 0 }}),
        .instructions = try allocator.dupe(ir.PlanInstruction, &.{.{
            .kind = .partition_id,
            .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 0 }}),
        }}),
    };
    defer plan.deinit();

    const executable = (try b.compileExecutable(allocator, &plan, &.{local_hardware_id})) orelse return error.TestUnexpectedResult;
    defer b.destroyExecutable(executable);

    const result = (try b.executeExecutable(allocator, executable, 0, &.{})) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(backend.ExecutionCompletionKind.completed, result.completion.kind);
    const outputs = result.outputs;
    defer allocator.free(outputs);
    defer {
        for (outputs) |output| b.destroyBuffer(output.handle);
    }

    try std.testing.expectEqual(@as(usize, 1), outputs.len);
    try std.testing.expectEqual(ir.BufferType.u32, outputs[0].element_type);
    try std.testing.expectEqual(@as(usize, 0), outputs[0].dims.len);
    var actual: u32 = undefined;
    try b.copyToHost(outputs[0].handle, std.mem.asBytes(&actual));
    try std.testing.expectEqual(@as(u32, 0), actual);
}

test "mlx metal backend executable lowers deprecated rng on device" {
    const b = create();
    const allocator = std.testing.allocator;
    const devices = try b.enumerateDevices(allocator, 1);
    defer b.releaseDeviceDescriptors(allocator, devices);
    const local_hardware_id = devices[0].local_hardware_id;
    const output_dims = [_]i64{ 2, 4 };
    const assignment = [_]i32{0};

    const values = try allocator.alloc(ir.Value, 3);
    errdefer allocator.free(values);
    values[0] = .{
        .id = .{ .index = 0 },
        .role = .parameter,
        .descriptor = .{
            .element_type = .f32,
            .dims = try allocator.alloc(i64, 0),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };
    values[1] = .{
        .id = .{ .index = 1 },
        .role = .parameter,
        .descriptor = .{
            .element_type = .f32,
            .dims = try allocator.alloc(i64, 0),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };
    values[2] = .{
        .id = .{ .index = 2 },
        .role = .instruction_result,
        .descriptor = .{
            .element_type = .f32,
            .dims = try allocator.dupe(i64, &output_dims),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };

    var parameter_shardings = try allocator.alloc(ir.ShardingPlan, 2);
    parameter_shardings[0] = .{
        .kind = .replicated,
        .mesh_name = try allocator.dupe(u8, ""),
        .device_assignment = try allocator.dupe(i32, &assignment),
    };
    parameter_shardings[1] = .{
        .kind = .replicated,
        .mesh_name = try allocator.dupe(u8, ""),
        .device_assignment = try allocator.dupe(i32, &assignment),
    };
    var output_shardings = try allocator.alloc(ir.ShardingPlan, 1);
    output_shardings[0] = .{
        .kind = .replicated,
        .mesh_name = try allocator.dupe(u8, ""),
        .device_assignment = try allocator.dupe(i32, &assignment),
    };

    var plan = ir.ExecutablePlan{
        .allocator = allocator,
        .options = .{
            .num_replicas = 1,
            .num_partitions = 1,
            .device_assignment = try allocator.dupe(i32, &assignment),
        },
        .values = values,
        .parameter_shardings = parameter_shardings,
        .output_shardings = output_shardings,
        .output_ids = try allocator.dupe(ir.ValueId, &.{.{ .index = 2 }}),
        .instructions = try allocator.dupe(ir.PlanInstruction, &.{.{
            .kind = .rng,
            .inputs = try allocator.dupe(ir.ValueId, &.{ .{ .index = 0 }, .{ .index = 1 } }),
            .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 2 }}),
            .dims = try allocator.dupe(i64, &output_dims),
            .rng_distribution = .normal,
        }}),
    };
    defer plan.deinit();

    const executable = (try b.compileExecutable(allocator, &plan, &.{local_hardware_id})) orelse return error.TestUnexpectedResult;
    defer b.destroyExecutable(executable);

    var mean: f32 = 0.0;
    var scale: f32 = 1.0;
    const mean_buffer = (try b.bufferFromHost(local_hardware_id, .f32, &.{}, std.mem.asBytes(&mean))) orelse return error.TestUnexpectedResult;
    defer b.destroyBuffer(mean_buffer);
    const scale_buffer = (try b.bufferFromHost(local_hardware_id, .f32, &.{}, std.mem.asBytes(&scale))) orelse return error.TestUnexpectedResult;
    defer b.destroyBuffer(scale_buffer);

    const result = (try b.executeExecutable(allocator, executable, 0, &.{ mean_buffer, scale_buffer })) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(backend.ExecutionCompletionKind.completed, result.completion.kind);
    const outputs = result.outputs;
    defer allocator.free(outputs);
    defer {
        for (outputs) |output| b.destroyBuffer(output.handle);
    }

    try std.testing.expectEqual(@as(usize, 1), outputs.len);
    try std.testing.expectEqual(ir.BufferType.f32, outputs[0].element_type);
    try std.testing.expectEqualSlices(i64, &output_dims, outputs[0].dims);
    var actual: [8]f32 = undefined;
    try b.copyToHost(outputs[0].handle, std.mem.asBytes(&actual));
    for (actual) |value| try std.testing.expect(std.math.isFinite(value));
}

test "mlx metal backend executable lowers clamp with scalar bounds" {
    const b = create();
    const allocator = std.testing.allocator;
    const devices = try b.enumerateDevices(allocator, 1);
    defer b.releaseDeviceDescriptors(allocator, devices);
    const local_hardware_id = devices[0].local_hardware_id;
    const dims = [_]i64{3};
    const assignment = [_]i32{0};
    const min_literal_value: f32 = -1.0;
    const max_literal_value: f32 = 2.0;

    const values = try allocator.alloc(ir.Value, 4);
    errdefer allocator.free(values);
    values[0] = .{
        .id = .{ .index = 0 },
        .role = .constant,
        .descriptor = .{
            .element_type = .f32,
            .dims = try allocator.dupe(i64, &.{}),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };
    values[1] = .{
        .id = .{ .index = 1 },
        .role = .parameter,
        .descriptor = .{
            .element_type = .f32,
            .dims = try allocator.dupe(i64, &dims),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };
    values[2] = .{
        .id = .{ .index = 2 },
        .role = .constant,
        .descriptor = .{
            .element_type = .f32,
            .dims = try allocator.dupe(i64, &.{}),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };
    values[3] = .{
        .id = .{ .index = 3 },
        .role = .instruction_result,
        .descriptor = .{
            .element_type = .f32,
            .dims = try allocator.dupe(i64, &dims),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };

    var parameter_shardings = try allocator.alloc(ir.ShardingPlan, 1);
    parameter_shardings[0] = .{
        .kind = .replicated,
        .mesh_name = try allocator.dupe(u8, ""),
        .device_assignment = try allocator.dupe(i32, &assignment),
    };
    var output_shardings = try allocator.alloc(ir.ShardingPlan, 1);
    output_shardings[0] = .{
        .kind = .replicated,
        .mesh_name = try allocator.dupe(u8, ""),
        .device_assignment = try allocator.dupe(i32, &assignment),
    };

    var plan = ir.ExecutablePlan{
        .allocator = allocator,
        .options = .{
            .num_replicas = 1,
            .num_partitions = 1,
            .device_assignment = try allocator.dupe(i32, &assignment),
        },
        .values = values,
        .parameter_shardings = parameter_shardings,
        .output_shardings = output_shardings,
        .output_ids = try allocator.dupe(ir.ValueId, &.{.{ .index = 3 }}),
        .instructions = try allocator.dupe(ir.PlanInstruction, &.{
            .{
                .kind = .constant,
                .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 0 }}),
                .literal = try allocator.dupe(u8, std.mem.asBytes(&min_literal_value)),
            },
            .{
                .kind = .constant,
                .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 2 }}),
                .literal = try allocator.dupe(u8, std.mem.asBytes(&max_literal_value)),
            },
            .{
                .kind = .clamp,
                .inputs = try allocator.dupe(ir.ValueId, &.{ .{ .index = 0 }, .{ .index = 1 }, .{ .index = 2 } }),
                .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 3 }}),
                .dims = try allocator.dupe(i64, &dims),
            },
        }),
    };
    defer plan.deinit();

    const executable = (try b.compileExecutable(allocator, &plan, &.{local_hardware_id})) orelse return error.TestUnexpectedResult;
    defer b.destroyExecutable(executable);

    const input = [_]f32{ -2.0, 0.5, 3.0 };
    const input_buffer = (try b.bufferFromHost(local_hardware_id, .f32, &dims, std.mem.asBytes(&input))) orelse return error.TestUnexpectedResult;
    defer b.destroyBuffer(input_buffer);

    const result = (try b.executeExecutable(allocator, executable, 0, &.{input_buffer})) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(backend.ExecutionCompletionKind.completed, result.completion.kind);
    const outputs = result.outputs;
    defer allocator.free(outputs);
    defer {
        for (outputs) |output| b.destroyBuffer(output.handle);
    }

    var actual: [3]f32 = undefined;
    try b.copyToHost(outputs[0].handle, std.mem.asBytes(&actual));
    try std.testing.expectEqualSlices(f32, &.{ -1.0, 0.5, 2.0 }, &actual);
}

test "mlx metal backend executable rejects unsupported gather form during lowering" {
    const b = create();
    const allocator = std.testing.allocator;
    const assignment = [_]i32{0};
    const operand_dims = [_]i64{ 4, 2 };
    const index_dims = [_]i64{2};
    const output_dims = [_]i64{ 2, 2 };

    const values = try allocator.alloc(ir.Value, 3);
    errdefer allocator.free(values);
    values[0] = .{
        .id = .{ .index = 0 },
        .role = .parameter,
        .descriptor = .{
            .element_type = .f32,
            .dims = try allocator.dupe(i64, &operand_dims),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };
    values[1] = .{
        .id = .{ .index = 1 },
        .role = .parameter,
        .descriptor = .{
            .element_type = .s32,
            .dims = try allocator.dupe(i64, &index_dims),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };
    values[2] = .{
        .id = .{ .index = 2 },
        .role = .instruction_result,
        .descriptor = .{
            .element_type = .f32,
            .dims = try allocator.dupe(i64, &output_dims),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };

    var parameter_shardings = try allocator.alloc(ir.ShardingPlan, 2);
    parameter_shardings[0] = .{
        .kind = .replicated,
        .mesh_name = try allocator.dupe(u8, ""),
        .device_assignment = try allocator.dupe(i32, &assignment),
    };
    parameter_shardings[1] = .{
        .kind = .replicated,
        .mesh_name = try allocator.dupe(u8, ""),
        .device_assignment = try allocator.dupe(i32, &assignment),
    };
    var output_shardings = try allocator.alloc(ir.ShardingPlan, 1);
    output_shardings[0] = .{
        .kind = .replicated,
        .mesh_name = try allocator.dupe(u8, ""),
        .device_assignment = try allocator.dupe(i32, &assignment),
    };

    var plan = ir.ExecutablePlan{
        .allocator = allocator,
        .options = .{
            .num_replicas = 1,
            .num_partitions = 1,
            .device_assignment = try allocator.dupe(i32, &assignment),
        },
        .values = values,
        .parameter_shardings = parameter_shardings,
        .output_shardings = output_shardings,
        .output_ids = try allocator.dupe(ir.ValueId, &.{.{ .index = 2 }}),
        .instructions = try allocator.dupe(ir.PlanInstruction, &.{.{
            .kind = .gather,
            .inputs = try allocator.dupe(ir.ValueId, &.{ .{ .index = 0 }, .{ .index = 1 } }),
            .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 2 }}),
            .dims = try allocator.dupe(i64, &output_dims),
            .start_index_map = try allocator.dupe(i64, &.{ 0, 1 }),
            .collapsed_slice_dims = try allocator.dupe(i64, &.{1}),
            .slice_sizes = try allocator.dupe(i64, &.{ 4, 1 }),
            .index_vector_dim = 1,
        }}),
    };
    defer plan.deinit();

    const maybe_executable = try b.compileExecutable(allocator, &plan, &assignment);
    if (maybe_executable) |executable| b.destroyExecutable(executable);
    try std.testing.expect(maybe_executable == null);
    var diagnostics = std.Io.Writer.Allocating.init(allocator);
    defer diagnostics.deinit();
    try b.writeExecutableLoweringDiagnostic(&plan, &assignment, &diagnostics.writer);
    const message = diagnostics.writer.buffered();
    try std.testing.expect(std.mem.indexOf(u8, message, "op=gather") != null);
    try std.testing.expect(std.mem.indexOf(u8, message, "feature=mlx-gather-general-shape") != null);
}
