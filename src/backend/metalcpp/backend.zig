const std = @import("std");

const ir = @import("src/compiler/ir");
const mlx_metal = @import("src/backend/mlx_metal");
const msl_artifacts = @import("msl_artifacts.zig");

/// Opaque Metal device-buffer handle shared with the transitional MLX storage owner.
pub const BufferHandle = mlx_metal.BufferHandle;
/// Opaque resident executable handle used by runtime executable residency.
pub const ExecutableHandle = mlx_metal.ExecutableHandle;
/// Opaque execution event handle returned by asynchronous backend execution.
pub const ExecutionEventHandle = mlx_metal.ExecutionEventHandle;
/// Opaque async host-to-device transfer handle owned by the backend.
pub const AsyncHostToDeviceTransferHandle = mlx_metal.AsyncHostToDeviceTransferHandle;

/// Errors produced by the concrete metal-cpp backend boundary.
pub const Error = mlx_metal.Error;

/// Custom-call registration kind accepted by the metal-cpp backend.
pub const CustomCallKind = mlx_metal.CustomCallKind;
/// Custom-call registration record copied into the backend registry.
pub const CustomCallRegistration = mlx_metal.CustomCallRegistration;

/// Device buffer returned for one executable output.
pub const ExecutableOutput = mlx_metal.ExecutableOutput;
/// Completion mode for a metal-cpp execute call.
pub const ExecutionCompletionKind = mlx_metal.ExecutionCompletionKind;
/// Completion token returned with execution outputs.
pub const ExecutionCompletion = mlx_metal.ExecutionCompletion;
/// State reported for an asynchronous backend execution event.
pub const ExecutionEventState = mlx_metal.ExecutionEventState;
/// Status payload for an asynchronous backend execution event.
pub const ExecutionEventStatus = mlx_metal.ExecutionEventStatus;
/// Result of executing a compiled metal-cpp executable on one device.
pub const ExecutionResult = mlx_metal.ExecutionResult;

/// Runtime-visible execution counters for one compiled executable.
pub const ExecutableStats = mlx_metal.ExecutableStats;

/// Backend graph node classification re-exported for runtime tests and metadata.
pub const ProgramNodeKind = mlx_metal.ProgramNodeKind;
/// Backend graph node metadata re-exported for runtime tests and metadata.
pub const ProgramNode = mlx_metal.ProgramNode;
/// Backend fusion group classification re-exported for runtime tests and metadata.
pub const FusionGroupKind = mlx_metal.FusionGroupKind;
/// Backend fusion group metadata re-exported for runtime tests and metadata.
pub const FusionGroup = mlx_metal.FusionGroup;
/// Backend materialization reason re-exported for runtime tests and metadata.
pub const MaterializationReason = mlx_metal.MaterializationReason;
/// Backend materialization boundary re-exported for runtime tests and metadata.
pub const MaterializationBoundary = mlx_metal.MaterializationBoundary;
/// Backend program value metadata re-exported for runtime tests and metadata.
pub const ProgramValue = mlx_metal.ProgramValue;
/// Backend program edge metadata re-exported for runtime tests and metadata.
pub const ProgramEdge = mlx_metal.ProgramEdge;
/// Backend schedule kind re-exported for runtime tests and metadata.
pub const ProgramScheduleKind = mlx_metal.ProgramScheduleKind;
/// Backend schedule item re-exported for runtime tests and metadata.
pub const ProgramScheduleItem = mlx_metal.ProgramScheduleItem;
/// Backend subprogram metadata re-exported for runtime tests and metadata.
pub const ProgramSubprogram = mlx_metal.ProgramSubprogram;
/// Backend control-flow kind re-exported for runtime tests and metadata.
pub const ProgramControlFlowKind = mlx_metal.ProgramControlFlowKind;
/// Backend control-flow metadata re-exported for runtime tests and metadata.
pub const ProgramControlFlow = mlx_metal.ProgramControlFlow;
/// Backend liveness stats re-exported for runtime tests and metadata.
pub const ProgramLivenessStats = mlx_metal.ProgramLivenessStats;
/// Backend program graph re-exported for runtime tests and metadata.
pub const Program = mlx_metal.Program;

/// Static capabilities reported by the concrete metal-cpp backend.
pub const Capabilities = mlx_metal.Capabilities;

/// Concrete metal-cpp backend facade selected by runtime when requested by env.
pub const Backend = struct {
    inner: mlx_metal.Backend = mlx_metal.create(),

    /// Reports the backend identity used by compile-cache keys and diagnostics.
    pub fn capabilities(self: Backend) Capabilities {
        var caps = self.inner.capabilities();
        caps.name = "metalcpp";
        return caps;
    }

    /// Enumerates devices through the current Metal device owner.
    pub fn enumerateDevices(self: Backend, allocator: std.mem.Allocator, device_count_hint: usize) Error![]ir.DeviceDescriptor { return self.inner.enumerateDevices(allocator, device_count_hint); }

    /// Releases device descriptors allocated by `enumerateDevices`.
    pub fn releaseDeviceDescriptors(self: Backend, allocator: std.mem.Allocator, descriptors: []ir.DeviceDescriptor) void { self.inner.releaseDeviceDescriptors(allocator, descriptors); }

    /// Imports host bytes into backend-owned device storage.
    pub fn bufferFromHost(self: Backend, device_local_hardware_id: i32, element_type: ir.BufferType, dims: []const i64, src: []const u8) Error!?BufferHandle { return self.inner.bufferFromHost(device_local_hardware_id, element_type, dims, src); }

    /// Starts an async host-to-device transfer.
    pub fn beginAsyncHostToDeviceTransfer(self: Backend, device_local_hardware_id: i32, element_type: ir.BufferType, dims: []const i64, byte_size: usize) Error!?AsyncHostToDeviceTransferHandle { return self.inner.beginAsyncHostToDeviceTransfer(device_local_hardware_id, element_type, dims, byte_size); }

    /// Writes bytes into an active async transfer.
    pub fn writeAsyncHostToDeviceTransfer(self: Backend, transfer: AsyncHostToDeviceTransferHandle, offset: usize, src: []const u8) Error!void { return self.inner.writeAsyncHostToDeviceTransfer(transfer, offset, src); }

    /// Finishes an async transfer and returns device storage.
    pub fn finishAsyncHostToDeviceTransfer(self: Backend, transfer: AsyncHostToDeviceTransferHandle) Error!?BufferHandle { return self.inner.finishAsyncHostToDeviceTransfer(transfer); }

    /// Destroys an async host-to-device transfer handle.
    pub fn destroyAsyncHostToDeviceTransfer(self: Backend, transfer: AsyncHostToDeviceTransferHandle) void { self.inner.destroyAsyncHostToDeviceTransfer(transfer); }

    /// Allocates device storage for a runtime buffer.
    pub fn allocateBuffer(self: Backend, device_local_hardware_id: i32, element_type: ir.BufferType, dims: []const i64) Error!?BufferHandle { return self.inner.allocateBuffer(device_local_hardware_id, element_type, dims); }

    /// Clones backend-owned device storage.
    pub fn cloneBuffer(self: Backend, src: BufferHandle) Error!?BufferHandle { return self.inner.cloneBuffer(src); }

    /// Compiles a backend executable from compiler-owned IR.
    pub fn compileExecutable(self: Backend, allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, device_local_hardware_ids: []const i32) Error!?ExecutableHandle { return self.inner.compileExecutable(allocator, plan, device_local_hardware_ids); }

    /// Writes a lowering diagnostic for unsupported executable forms.
    pub fn writeExecutableLoweringDiagnostic(self: Backend, plan: *const ir.ExecutablePlan, device_local_hardware_ids: []const i32, writer: *std.Io.Writer) std.Io.Writer.Error!void { return self.inner.writeExecutableLoweringDiagnostic(plan, device_local_hardware_ids, writer); }

    /// Executes one device slice of a compiled backend executable.
    pub fn executeExecutable(self: Backend, allocator: std.mem.Allocator, executable: ExecutableHandle, device_index: usize, arguments: []const BufferHandle) Error!?ExecutionResult { return self.inner.executeExecutable(allocator, executable, device_index, arguments); }

    /// Queries an execution event status.
    pub fn executionEventStatus(self: Backend, event: ExecutionEventHandle) Error!ExecutionEventStatus { return self.inner.executionEventStatus(event); }

    /// Destroys an execution event handle.
    pub fn destroyExecutionEvent(self: Backend, event: ExecutionEventHandle) void { self.inner.destroyExecutionEvent(event); }

    /// Returns execution stats for a resident executable.
    pub fn executableStats(self: Backend, executable: ExecutableHandle) ExecutableStats { return self.inner.executableStats(executable); }

    /// Destroys a resident executable.
    pub fn destroyExecutable(self: Backend, executable: ExecutableHandle) void { self.inner.destroyExecutable(executable); }

    /// Registers a typed custom-call target.
    pub fn registerCustomCall(self: Backend, registration: CustomCallRegistration) Error!void { return self.inner.registerCustomCall(registration); }

    /// Registers a named binary custom-call target.
    pub fn registerBinaryCustomCall(self: Backend, target: []const u8, op_name: []const u8) Error!void {
        return self.inner.registerBinaryCustomCall(target, op_name);
    }

    /// Registers an identity custom-call target.
    pub fn registerIdentityCustomCall(self: Backend, target: []const u8) Error!void { return self.inner.registerIdentityCustomCall(target); }

    /// Registers a named unary custom-call target.
    pub fn registerUnaryCustomCall(self: Backend, target: []const u8, op_name: []const u8) Error!void {
        return self.inner.registerUnaryCustomCall(target, op_name);
    }

    /// Registers the built-in square-root custom-call marker.
    pub fn registerUnarySqrtCustomCall(self: Backend, target: []const u8) Error!void { return self.inner.registerUnarySqrtCustomCall(target); }

    /// Registers the built-in binary-add custom-call marker.
    pub fn registerBinaryAddCustomCall(self: Backend, target: []const u8) Error!void { return self.inner.registerBinaryAddCustomCall(target); }

    /// Removes a custom-call target from the backend registry.
    pub fn unregisterCustomCall(self: Backend, target: []const u8) void { self.inner.unregisterCustomCall(target); }

    /// Returns the custom-call registry version for compile-cache keys.
    pub fn customCallRegistryVersion(self: Backend) u64 { return self.inner.customCallRegistryVersion(); }

    /// Copies backend-owned device storage to host bytes.
    pub fn copyToHost(self: Backend, src: BufferHandle, dst: []u8) Error!void {
        return self.inner.copyToHost(src, dst);
    }

    /// Destroys backend-owned device storage.
    pub fn destroyBuffer(self: Backend, buffer: BufferHandle) void { self.inner.destroyBuffer(buffer); }
};

/// Returns the concrete direct metal-cpp backend facade.
pub fn create() Backend {
    return .{};
}

test "metalcpp backend reports distinct runtime identity" {
    try std.testing.expectEqualStrings("metalcpp", create().capabilities().name);
}

test {
    _ = msl_artifacts;
}
