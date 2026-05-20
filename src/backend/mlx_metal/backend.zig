const std = @import("std");
const backend = @This();
const ir = @import("src/compiler/ir");
const program_mod = @import("program.zig");
const program_build_mod = @import("program_build.zig");
const device_mod = @import("device.zig");
const buffer_mod = @import("buffer.zig");
const async_transfer_mod = @import("async_transfer.zig");
const mlx_call = @import("mlx_call.zig");
const custom_call_mod = @import("custom_call.zig");
const lowering_mod = @import("lowering.zig");
const profiling_mod = @import("profiling.zig");
const executable_mod = @import("executable.zig");
const execution_mod = @import("execution.zig");

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

/// Device buffer returned for one executable output.
pub const ExecutableOutput = execution_mod.ExecutableOutput;
/// Completion mode for an MLX/Metal execute call.
pub const ExecutionCompletionKind = execution_mod.ExecutionCompletionKind;
/// Completion token returned with execution outputs.
pub const ExecutionCompletion = execution_mod.ExecutionCompletion;
/// State reported for an asynchronous backend execution event.
pub const ExecutionEventState = execution_mod.ExecutionEventState;
/// Status payload for an asynchronous backend execution event.
pub const ExecutionEventStatus = execution_mod.ExecutionEventStatus;
/// Result of executing a compiled MLX/Metal executable on one device.
pub const ExecutionResult = execution_mod.ExecutionResult;

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

    /// Registers a named binary custom-call target in the MLX/Metal backend registry.
    pub fn registerBinaryCustomCall(self: Backend, target: []const u8, op_name: []const u8) Error!void {
        return backend.registerBinaryCustomCall(self, target, op_name);
    }

    /// Registers an identity custom-call target in the MLX/Metal backend registry.
    pub fn registerIdentityCustomCall(self: Backend, target: []const u8) Error!void {
        return backend.registerIdentityCustomCall(self, target);
    }

    /// Registers a named unary custom-call target in the MLX/Metal backend registry.
    pub fn registerUnaryCustomCall(self: Backend, target: []const u8, op_name: []const u8) Error!void {
        return backend.registerUnaryCustomCall(self, target, op_name);
    }

    /// Registers the built-in unary square-root custom-call marker.
    pub fn registerUnarySqrtCustomCall(self: Backend, target: []const u8) Error!void {
        return backend.registerUnarySqrtCustomCall(self, target);
    }

    /// Registers the built-in binary add custom-call marker.
    pub fn registerBinaryAddCustomCall(self: Backend, target: []const u8) Error!void {
        return backend.registerBinaryAddCustomCall(self, target);
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

const CompiledExecutable = executable_mod.Executable;

const CompiledProgramContext = executable_mod.CompiledProgramContext;
const ArgumentCaptureState = executable_mod.ArgumentCaptureState;

fn registerCustomCall(_: backend.Backend, registration: backend.CustomCallRegistration) backend.Error!void {
    return custom_call_mod.register(registration);
}

fn registerBinaryCustomCall(_: backend.Backend, target: []const u8, op_name: []const u8) backend.Error!void {
    return custom_call_mod.registerBinary(target, op_name);
}

fn registerIdentityCustomCall(_: backend.Backend, target: []const u8) backend.Error!void {
    return custom_call_mod.registerIdentity(target);
}

fn registerUnaryCustomCall(_: backend.Backend, target: []const u8, op_name: []const u8) backend.Error!void {
    return custom_call_mod.registerUnary(target, op_name);
}

fn registerUnarySqrtCustomCall(_: backend.Backend, target: []const u8) backend.Error!void {
    return custom_call_mod.registerUnarySqrt(target);
}

fn registerBinaryAddCustomCall(_: backend.Backend, target: []const u8) backend.Error!void {
    return custom_call_mod.registerBinaryAdd(target);
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
    return program_build_mod.build(allocator, plan, diagnostic_writer);
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
    errdefer execution_mod.destroyConstantHandles(constant_handles);
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
    errdefer execution_mod.destroyConstantHandles(while_constant_handles);
    const compiled_program_contexts = try allocator.alloc(CompiledProgramContext, device_local_hardware_ids.len);
    errdefer allocator.free(compiled_program_contexts);
    const compiled_program_handles = try allocator.alloc(?mlx_call.ProgramHandle, device_local_hardware_ids.len);
    errdefer allocator.free(compiled_program_handles);
    @memset(compiled_program_handles, null);
    errdefer execution_mod.destroyCompiledPrograms(compiled_program_handles);
    const argument_capture_states = try allocator.alloc(ArgumentCaptureState, device_local_hardware_ids.len);
    errdefer allocator.free(argument_capture_states);
    for (argument_capture_states) |*state| state.* = .{};
    errdefer execution_mod.destroyArgumentCaptureStates(allocator, argument_capture_states);
    const liveness_stats = try program.livenessStats();

    var resident_constant_count: usize = 0;
    var resident_constant_bytes: usize = 0;
    for (device_local_hardware_ids, 0..) |local_hardware_id, device_index| {
        for (plan.instructions, 0..) |instruction, instruction_index| {
            if (instruction.kind != .constant) continue;
            const output_id = instruction.outputs[0];
            const descriptor = plan.values[output_id.index].descriptor;
            const literal = instruction.literal.?;
            constant_handles[executable_mod.constantIndex(plan.instructions.len, device_index, instruction_index)] = (try bufferFromHost(
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
        const pattern = lowering_mod.matchWhileF32LtAddPattern(program.subprograms[control_flow.condition_subprogram], program.subprograms[control_flow.body_subprogram]) orelse continue;
        for (device_local_hardware_ids, 0..) |local_hardware_id, device_index| {
            if (pattern.limit.role == .constant) {
                const limit_literal = pattern.limit.literal orelse return error.InvalidProgram;
                while_constant_handles[executable_mod.whileConstantIndex(program.control_flows.len, device_index, control_flow_index, 0)] = (try bufferFromHost(
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
                while_constant_handles[executable_mod.whileConstantIndex(program.control_flows.len, device_index, control_flow_index, 1)] = (try bufferFromHost(
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
                execution_mod.compiledProgramBuildCallback,
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

fn executeExecutable(_: backend.Backend, allocator: std.mem.Allocator, executable_handle: backend.ExecutableHandle, device_index: usize, arguments: []const backend.BufferHandle) backend.Error!?backend.ExecutionResult {
    return execution_mod.execute(allocator, executable_handle, device_index, arguments);
}

fn executionEventStatus(_: backend.Backend, event: backend.ExecutionEventHandle) backend.Error!backend.ExecutionEventStatus {
    return execution_mod.eventStatus(event);
}

fn destroyExecutionEvent(_: backend.Backend, event: backend.ExecutionEventHandle) void {
    execution_mod.destroyEvent(event);
}

fn executableStats(_: backend.Backend, executable_handle: backend.ExecutableHandle) backend.ExecutableStats {
    return execution_mod.stats(executable_handle);
}

fn destroyExecutable(_: backend.Backend, executable_handle: backend.ExecutableHandle) void {
    execution_mod.destroy(executable_handle);
}

fn executableLoweringIssue(plan: *const ir.ExecutablePlan, device_local_hardware_ids: []const i32) ?lowering_mod.Issue {
    return lowering_mod.executableIssue(plan, device_local_hardware_ids);
}

fn writeLoweringIssue(plan: *const ir.ExecutablePlan, issue: lowering_mod.Issue, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    return lowering_mod.writeIssue(plan, issue, writer);
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
