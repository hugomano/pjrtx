const std = @import("std");
const ir = @import("src/compiler/ir");
const program_mod = @import("program.zig");
const device_mod = @import("device.zig");
const buffer_mod = @import("buffer.zig");
const async_transfer_mod = @import("async_transfer.zig");
const custom_call_mod = @import("custom_call.zig");
const lowering_mod = @import("lowering.zig");
const executable_mod = @import("executable.zig");
const execution_mod = @import("execution.zig");

/// Opaque MLX/Metal device-buffer handle owned by runtime buffers or executable residency.
pub const BufferHandle = *anyopaque;
/// Opaque resident executable handle owned by runtime executable graphs.
pub const ExecutableHandle = *anyopaque;
/// Opaque backend execution-event handle reserved for asynchronous completion.
pub const ExecutionEventHandle = *anyopaque;
/// Opaque async host-to-device transfer handle owned by runtime transfer managers.
pub const AsyncHostToDeviceTransferHandle = *anyopaque;

/// Errors produced by the concrete MLX/Metal backend boundary.
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

/// Static capabilities reported by the concrete MLX/Metal backend.
pub const Capabilities = device_mod.Capabilities;

/// Concrete Metal/MLX backend facade used by runtime lifecycle and execution code.
pub const Backend = struct {
    pub fn capabilities(self: Backend) Capabilities { _ = self; return device_mod.capabilities(); }

    pub fn enumerateDevices(self: Backend, allocator: std.mem.Allocator, device_count_hint: usize) Error![]ir.DeviceDescriptor {
        _ = self;
        _ = device_count_hint;
        return device_mod.DeviceList.enumerate(allocator);
    }

    pub fn releaseDeviceDescriptors(self: Backend, allocator: std.mem.Allocator, descriptors: []ir.DeviceDescriptor) void {
        _ = self; device_mod.DeviceList.release(allocator, descriptors);
    }

    pub fn bufferFromHost(self: Backend, device_local_hardware_id: i32, element_type: ir.BufferType, dims: []const i64, src: []const u8) Error!?BufferHandle {
        _ = self; return buffer_mod.Opaque.fromHost(device_local_hardware_id, element_type, dims, src);
    }

    pub fn beginAsyncHostToDeviceTransfer(self: Backend, device_local_hardware_id: i32, element_type: ir.BufferType, dims: []const i64, byte_size: usize) Error!?AsyncHostToDeviceTransferHandle {
        _ = self; return async_transfer_mod.Opaque.begin(device_local_hardware_id, element_type, dims, byte_size);
    }

    pub fn writeAsyncHostToDeviceTransfer(self: Backend, transfer: AsyncHostToDeviceTransferHandle, offset: usize, src: []const u8) Error!void {
        _ = self; return async_transfer_mod.Opaque.write(transfer, offset, src);
    }

    pub fn finishAsyncHostToDeviceTransfer(self: Backend, transfer: AsyncHostToDeviceTransferHandle) Error!?BufferHandle {
        _ = self; return async_transfer_mod.Opaque.finish(transfer);
    }

    pub fn destroyAsyncHostToDeviceTransfer(self: Backend, transfer: AsyncHostToDeviceTransferHandle) void {
        _ = self; async_transfer_mod.Opaque.destroy(transfer);
    }

    pub fn allocateBuffer(self: Backend, device_local_hardware_id: i32, element_type: ir.BufferType, dims: []const i64) Error!?BufferHandle {
        _ = self; return buffer_mod.Opaque.zeros(device_local_hardware_id, element_type, dims);
    }

    pub fn cloneBuffer(self: Backend, src: BufferHandle) Error!?BufferHandle { _ = self; return buffer_mod.Opaque.clone(src); }

    pub fn compileExecutable(self: Backend, allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, device_local_hardware_ids: []const i32) Error!?ExecutableHandle {
        _ = self; return executable_mod.compile(allocator, plan, device_local_hardware_ids, execution_mod.compiledProgramBuildCallback);
    }

    pub fn writeExecutableLoweringDiagnostic(self: Backend, plan: *const ir.ExecutablePlan, device_local_hardware_ids: []const i32, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        _ = self; return lowering_mod.writeExecutableDiagnostic(plan, device_local_hardware_ids, writer);
    }

    pub fn executeExecutable(self: Backend, allocator: std.mem.Allocator, executable: ExecutableHandle, device_index: usize, arguments: []const BufferHandle) Error!?ExecutionResult {
        _ = self; return execution_mod.execute(allocator, executable, device_index, arguments);
    }

    pub fn executionEventStatus(self: Backend, event: ExecutionEventHandle) Error!ExecutionEventStatus { _ = self; return execution_mod.eventStatus(event); }

    pub fn destroyExecutionEvent(self: Backend, event: ExecutionEventHandle) void { _ = self; execution_mod.destroyEvent(event); }

    pub fn executableStats(self: Backend, executable: ExecutableHandle) ExecutableStats { _ = self; return execution_mod.stats(executable); }

    pub fn destroyExecutable(self: Backend, executable: ExecutableHandle) void { _ = self; execution_mod.destroy(executable); }

    pub fn registerCustomCall(self: Backend, registration: CustomCallRegistration) Error!void { _ = self; return custom_call_mod.register(registration); }

    /// Registers a named binary custom-call target in the MLX/Metal backend registry.
    pub fn registerBinaryCustomCall(self: Backend, target: []const u8, op_name: []const u8) Error!void {
        _ = self; return custom_call_mod.registerBinary(target, op_name);
    }

    /// Registers an identity custom-call target in the MLX/Metal backend registry.
    pub fn registerIdentityCustomCall(self: Backend, target: []const u8) Error!void { _ = self; return custom_call_mod.registerIdentity(target); }

    /// Registers a named unary custom-call target in the MLX/Metal backend registry.
    pub fn registerUnaryCustomCall(self: Backend, target: []const u8, op_name: []const u8) Error!void {
        _ = self; return custom_call_mod.registerUnary(target, op_name);
    }

    /// Registers the built-in unary square-root custom-call marker.
    pub fn registerUnarySqrtCustomCall(self: Backend, target: []const u8) Error!void { _ = self; return custom_call_mod.registerUnarySqrt(target); }

    /// Registers the built-in binary add custom-call marker.
    pub fn registerBinaryAddCustomCall(self: Backend, target: []const u8) Error!void { _ = self; return custom_call_mod.registerBinaryAdd(target); }

    pub fn unregisterCustomCall(self: Backend, target: []const u8) void {
        _ = self; custom_call_mod.unregister(target);
    }

    pub fn customCallRegistryVersion(self: Backend) u64 { _ = self; return custom_call_mod.version(); }

    pub fn copyToHost(self: Backend, src: BufferHandle, dst: []u8) Error!void { _ = self; return buffer_mod.Opaque.copyToHost(src, dst); }

    pub fn destroyBuffer(self: Backend, buffer: BufferHandle) void { _ = self; buffer_mod.Opaque.destroy(buffer); }
};

/// Returns the stateless concrete MLX/Metal backend facade.
pub fn create() Backend {
    return .{};
}

test "mlx metal backend exposes opaque backend interface" {
    _ = @import("execution_integration.zig");

    const b = create();
    try std.testing.expectEqualStrings("metal_mlx", b.capabilities().name);
    const devices = try b.enumerateDevices(std.testing.allocator, 1);
    defer b.releaseDeviceDescriptors(std.testing.allocator, devices);
    try std.testing.expect(devices.len >= 1);
}
