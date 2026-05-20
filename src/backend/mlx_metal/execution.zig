const std = @import("std");
const ir = @import("src/compiler/ir");

const async_transfer_mod = @import("async_transfer.zig");
const buffer_mod = @import("buffer.zig");
const custom_call_mod = @import("custom_call.zig");
const executable_mod = @import("executable.zig");
const lowering_mod = @import("lowering.zig");
const mlx_call = @import("mlx_call.zig");
const profiling_mod = @import("profiling.zig");
const program_mod = @import("program.zig");

/// Opaque MLX/Metal buffer handle accepted by backend execution.
pub const BufferHandle = *anyopaque;
/// Opaque compiled executable handle owned by the MLX backend.
pub const ExecutableHandle = *anyopaque;
/// Opaque asynchronous execution event handle reserved for future MLX support.
pub const ExecutionEventHandle = *anyopaque;
/// Opaque async host-to-device transfer handle used by private execution helpers.
pub const AsyncHostToDeviceTransferHandle = *anyopaque;
/// Errors reported by MLX/Metal execution and executable teardown.
pub const Error = program_mod.Error;

const RootBufferHandle = BufferHandle;
const RootExecutableHandle = ExecutableHandle;
const RootExecutionEventHandle = ExecutionEventHandle;
const RootAsyncHostToDeviceTransferHandle = AsyncHostToDeviceTransferHandle;

/// Device buffer returned for one executable output.
pub const ExecutableOutput = struct {
    handle: BufferHandle,
    element_type: ir.BufferType,
    dims: []const i64,
    byte_size: usize,
};

/// Completion mode for an MLX/Metal execute call.
pub const ExecutionCompletionKind = enum {
    completed,
    pending,
};

/// Completion token returned with execution outputs.
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

/// State reported for an asynchronous backend execution event.
pub const ExecutionEventState = enum {
    pending,
    ready,
    failed,
};

/// Status payload for an asynchronous backend execution event.
pub const ExecutionEventStatus = struct {
    state: ExecutionEventState,
    message: []const u8 = "",
};

/// Result of executing a compiled MLX/Metal executable on one device.
pub const ExecutionResult = struct {
    outputs: []ExecutableOutput,
    completion: ExecutionCompletion = .{},
};

const backend = struct {
    const Backend = struct {};
    const BufferHandle = RootBufferHandle;
    const ExecutableHandle = RootExecutableHandle;
    const ExecutionEventHandle = RootExecutionEventHandle;
    const AsyncHostToDeviceTransferHandle = RootAsyncHostToDeviceTransferHandle;
    const Error = program_mod.Error;
    const ExecutableOutput = @import("execution.zig").ExecutableOutput;
    const ExecutionResult = @import("execution.zig").ExecutionResult;
    const ExecutionCompletion = @import("execution.zig").ExecutionCompletion;
    const ExecutionEventStatus = @import("execution.zig").ExecutionEventStatus;
    const ExecutableStats = executable_mod.Stats;
    const ReduceWindowMaxWithIndicesResult = struct { values: RootBufferHandle, indices: RootBufferHandle };
    const ReduceMaxWithIndicesResult = struct { values: RootBufferHandle, indices: RootBufferHandle };
    const RngBitGeneratorResult = struct { state: RootBufferHandle, bits: RootBufferHandle };
    const Program = program_mod.Program;
    const ProgramNode = program_mod.Node;
    const ProgramNodeKind = program_mod.NodeKind;
    const ProgramScheduleItem = program_mod.ScheduleItem;
    const ProgramScheduleKind = program_mod.ScheduleKind;
    const MaterializationReason = program_mod.MaterializationReason;
    const FusionGroup = program_mod.FusionGroup;
    const ProgramSubprogram = program_mod.Subprogram;
};

const CompiledExecutable = executable_mod.Executable;
const CompiledProgramContext = executable_mod.CompiledProgramContext;
const ArgumentCaptureState = executable_mod.ArgumentCaptureState;
const ExecuteProfile = profiling_mod.Execute;
const WhileF32LtAddPattern = lowering_mod.WhileF32LtAddPattern;
const WhilePatternOperand = lowering_mod.WhilePatternOperand;

const DefaultWhileMaxIterations: u64 = 1_000_000;
const InitialCaptureSmallControlBytes: usize = 4096;
const MinCapturedProgramStableInputs = 8;

const WhileOperandHandle = struct {
    handle: BufferHandle,
    owned: bool = false,
};

/// Executes a compiled MLX/Metal executable on one device.
pub fn execute(allocator: std.mem.Allocator, executable_handle: ExecutableHandle, device_index: usize, arguments: []const BufferHandle) Error!?ExecutionResult {
    return executeExecutable(create(), allocator, executable_handle, device_index, arguments);
}

/// Reports the status of a backend execution event.
pub fn eventStatus(event: ExecutionEventHandle) Error!ExecutionEventStatus {
    return executionEventStatus(create(), event);
}

/// Releases a backend execution event handle.
pub fn destroyEvent(event: ExecutionEventHandle) void {
    destroyExecutionEvent(create(), event);
}

/// Returns accumulated execution statistics for a compiled executable.
pub fn stats(executable_handle: ExecutableHandle) executable_mod.Stats {
    return executableStats(create(), executable_handle);
}

/// Destroys a compiled executable and all resident backend resources.
pub fn destroy(executable_handle: ExecutableHandle) void {
    destroyExecutable(create(), executable_handle);
}

/// Releases resident constant buffer handles after a compile failure or executable destroy.
pub fn destroyConstantHandles(constant_handles: []?BufferHandle) void {
    for (constant_handles) |maybe_handle| {
        if (maybe_handle) |handle| destroyBuffer(create(), handle);
    }
}

/// Releases compiled MLX program handles after a compile failure or executable destroy.
pub fn destroyCompiledPrograms(handles: []?mlx_call.ProgramHandle) void {
    for (handles) |maybe_handle| {
        if (maybe_handle) |handle| mlx_call.programDestroy(handle);
    }
}

/// Releases argument-capture state owned by a compiled executable.
pub fn destroyArgumentCaptureStates(allocator: std.mem.Allocator, states: []executable_mod.ArgumentCaptureState) void {
    for (states) |*state| resetArgumentCaptureState(allocator, state);
}

fn create() backend.Backend {
    return .{};
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
fn profileEnabled() bool {
    return profiling_mod.enabled();
}

fn profileVerbose() bool {
    return profiling_mod.verbose();
}

fn profileStart(enabled: bool) std.Io.Timestamp {
    return profiling_mod.start(enabled);
}

fn profileElapsedUs(start: std.Io.Timestamp) u64 {
    return profiling_mod.elapsedUs(start);
}

fn lookupCustomCall(target: []const u8) ?custom_call_mod.Spec {
    return custom_call_mod.lookup(target);
}

fn matchWhileF32LtAddPattern(cond: backend.ProgramSubprogram, body: backend.ProgramSubprogram) ?lowering_mod.WhileF32LtAddPattern {
    return lowering_mod.matchWhileF32LtAddPattern(cond, body);
}

fn regionValueById(subprogram: backend.ProgramSubprogram, id: ir.RegionValueId) ?ir.RegionValue {
    return lowering_mod.regionValueById(subprogram, id);
}

fn supportedScatterAxis(instruction: ir.PlanInstruction) ?i64 {
    return lowering_mod.supportedScatterAxis(instruction);
}

fn executableBinaryOp(instruction_kind: ir.PlanInstructionKind) ?ir.ElementwiseBinaryOp {
    return lowering_mod.executableBinaryOp(instruction_kind);
}

fn executableUnaryOp(instruction_kind: ir.PlanInstructionKind) ?ir.ElementwiseUnaryOp {
    return lowering_mod.executableUnaryOp(instruction_kind);
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

pub fn compiledProgramBuildCallback(
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
        const result = (try reduceWindowMaxWithIndices(
            backend_impl,
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
        const result = (try reduceMaxWithIndices(
            backend_impl,
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
        const result = (try rngBitGenerator(
            backend_impl,
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
            break :blk (try rng(
                backend_impl,
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
    _ = backend_impl;
    destroyConstantHandles(executable.constant_handles);
    destroyConstantHandles(executable.while_constant_handles);
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
