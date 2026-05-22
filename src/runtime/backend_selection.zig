const std = @import("std");

const ir = @import("src/compiler/ir");
const metalcpp = @import("src/backend/metalcpp");
const mlx_metal = @import("src/backend/mlx_metal");

/// Opaque backend buffer handle shared by runtime buffer ownership.
pub const BufferHandle = mlx_metal.BufferHandle;
/// Opaque resident executable handle shared by runtime executable residency.
pub const ExecutableHandle = mlx_metal.ExecutableHandle;
/// Opaque backend execution event handle shared by runtime events.
pub const ExecutionEventHandle = mlx_metal.ExecutionEventHandle;
/// Opaque async host-to-device transfer handle shared by runtime transfer ownership.
pub const AsyncHostToDeviceTransferHandle = mlx_metal.AsyncHostToDeviceTransferHandle;

/// Errors returned by the selected concrete backend.
pub const Error = mlx_metal.Error;

/// Custom-call registration kind accepted by the selected concrete backend.
pub const CustomCallKind = mlx_metal.CustomCallKind;
/// Custom-call registration payload forwarded to the selected concrete backend.
pub const CustomCallRegistration = mlx_metal.CustomCallRegistration;

/// Device buffer returned for one executable output.
pub const ExecutableOutput = mlx_metal.ExecutableOutput;
/// Completion mode for an execute call.
pub const ExecutionCompletionKind = mlx_metal.ExecutionCompletionKind;
/// Completion token returned with execution outputs.
pub const ExecutionCompletion = mlx_metal.ExecutionCompletion;
/// State reported for an asynchronous backend execution event.
pub const ExecutionEventState = mlx_metal.ExecutionEventState;
/// Status payload for an asynchronous backend execution event.
pub const ExecutionEventStatus = mlx_metal.ExecutionEventStatus;
/// Result of executing a compiled executable on one device.
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

/// Static capabilities reported by the selected concrete backend.
pub const Capabilities = mlx_metal.Capabilities;

/// Runtime-selected concrete backend. This is a closed union, not a registry or vtable.
pub const Backend = union(enum) {
    mlx_metal: mlx_metal.Backend,
    metalcpp: metalcpp.Backend,

    /// Reports backend capabilities used by diagnostics and compile-cache keys.
    pub fn capabilities(self: Backend) Capabilities {
        return switch (self) {
            .mlx_metal => |backend| backend.capabilities(),
            .metalcpp => |backend| backend.capabilities(),
        };
    }

    /// Enumerates concrete backend devices.
    pub fn enumerateDevices(self: Backend, allocator: std.mem.Allocator, device_count_hint: usize) Error![]ir.DeviceDescriptor {
        return switch (self) {
            .mlx_metal => |backend| backend.enumerateDevices(allocator, device_count_hint),
            .metalcpp => |backend| backend.enumerateDevices(allocator, device_count_hint),
        };
    }

    /// Releases device descriptors allocated by `enumerateDevices`.
    pub fn releaseDeviceDescriptors(self: Backend, allocator: std.mem.Allocator, descriptors: []ir.DeviceDescriptor) void {
        return switch (self) {
            .mlx_metal => |backend| backend.releaseDeviceDescriptors(allocator, descriptors),
            .metalcpp => |backend| backend.releaseDeviceDescriptors(allocator, descriptors),
        };
    }

    /// Imports host bytes into device storage.
    pub fn bufferFromHost(self: Backend, device_local_hardware_id: i32, element_type: ir.BufferType, dims: []const i64, src: []const u8) Error!?BufferHandle {
        return switch (self) {
            .mlx_metal => |backend| backend.bufferFromHost(device_local_hardware_id, element_type, dims, src),
            .metalcpp => |backend| backend.bufferFromHost(device_local_hardware_id, element_type, dims, src),
        };
    }

    /// Starts an async host-to-device transfer.
    pub fn beginAsyncHostToDeviceTransfer(self: Backend, device_local_hardware_id: i32, element_type: ir.BufferType, dims: []const i64, byte_size: usize) Error!?AsyncHostToDeviceTransferHandle {
        return switch (self) {
            .mlx_metal => |backend| backend.beginAsyncHostToDeviceTransfer(device_local_hardware_id, element_type, dims, byte_size),
            .metalcpp => |backend| backend.beginAsyncHostToDeviceTransfer(device_local_hardware_id, element_type, dims, byte_size),
        };
    }

    /// Writes bytes into an async transfer.
    pub fn writeAsyncHostToDeviceTransfer(self: Backend, transfer: AsyncHostToDeviceTransferHandle, offset: usize, src: []const u8) Error!void {
        return switch (self) {
            .mlx_metal => |backend| backend.writeAsyncHostToDeviceTransfer(transfer, offset, src),
            .metalcpp => |backend| backend.writeAsyncHostToDeviceTransfer(transfer, offset, src),
        };
    }

    /// Finishes an async transfer and returns device storage.
    pub fn finishAsyncHostToDeviceTransfer(self: Backend, transfer: AsyncHostToDeviceTransferHandle) Error!?BufferHandle {
        return switch (self) {
            .mlx_metal => |backend| backend.finishAsyncHostToDeviceTransfer(transfer),
            .metalcpp => |backend| backend.finishAsyncHostToDeviceTransfer(transfer),
        };
    }

    /// Destroys an async transfer handle.
    pub fn destroyAsyncHostToDeviceTransfer(self: Backend, transfer: AsyncHostToDeviceTransferHandle) void {
        return switch (self) {
            .mlx_metal => |backend| backend.destroyAsyncHostToDeviceTransfer(transfer),
            .metalcpp => |backend| backend.destroyAsyncHostToDeviceTransfer(transfer),
        };
    }

    /// Allocates device storage for a runtime buffer.
    pub fn allocateBuffer(self: Backend, device_local_hardware_id: i32, element_type: ir.BufferType, dims: []const i64) Error!?BufferHandle {
        return switch (self) {
            .mlx_metal => |backend| backend.allocateBuffer(device_local_hardware_id, element_type, dims),
            .metalcpp => |backend| backend.allocateBuffer(device_local_hardware_id, element_type, dims),
        };
    }

    /// Clones backend-owned device storage.
    pub fn cloneBuffer(self: Backend, src: BufferHandle) Error!?BufferHandle {
        return switch (self) {
            .mlx_metal => |backend| backend.cloneBuffer(src),
            .metalcpp => |backend| backend.cloneBuffer(src),
        };
    }

    /// Compiles a backend executable from compiler-owned IR.
    pub fn compileExecutable(self: Backend, allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, device_local_hardware_ids: []const i32) Error!?ExecutableHandle {
        return switch (self) {
            .mlx_metal => |backend| backend.compileExecutable(allocator, plan, device_local_hardware_ids),
            .metalcpp => |backend| backend.compileExecutable(allocator, plan, device_local_hardware_ids),
        };
    }

    /// Writes lowering diagnostics for unsupported executable forms.
    pub fn writeExecutableLoweringDiagnostic(self: Backend, plan: *const ir.ExecutablePlan, device_local_hardware_ids: []const i32, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        return switch (self) {
            .mlx_metal => |backend| backend.writeExecutableLoweringDiagnostic(plan, device_local_hardware_ids, writer),
            .metalcpp => |backend| backend.writeExecutableLoweringDiagnostic(plan, device_local_hardware_ids, writer),
        };
    }

    /// Executes one device slice of a compiled executable.
    pub fn executeExecutable(self: Backend, allocator: std.mem.Allocator, executable: ExecutableHandle, device_index: usize, arguments: []const BufferHandle) Error!?ExecutionResult {
        return switch (self) {
            .mlx_metal => |backend| backend.executeExecutable(allocator, executable, device_index, arguments),
            .metalcpp => |backend| backend.executeExecutable(allocator, executable, device_index, arguments),
        };
    }

    /// Queries an execution event status.
    pub fn executionEventStatus(self: Backend, event: ExecutionEventHandle) Error!ExecutionEventStatus {
        return switch (self) {
            .mlx_metal => |backend| backend.executionEventStatus(event),
            .metalcpp => |backend| backend.executionEventStatus(event),
        };
    }

    /// Destroys an execution event handle.
    pub fn destroyExecutionEvent(self: Backend, event: ExecutionEventHandle) void {
        return switch (self) {
            .mlx_metal => |backend| backend.destroyExecutionEvent(event),
            .metalcpp => |backend| backend.destroyExecutionEvent(event),
        };
    }

    /// Returns executable stats used by runtime metadata and cache policy.
    pub fn executableStats(self: Backend, executable: ExecutableHandle) ExecutableStats {
        return switch (self) {
            .mlx_metal => |backend| backend.executableStats(executable),
            .metalcpp => |backend| backend.executableStats(executable),
        };
    }

    /// Destroys a resident executable.
    pub fn destroyExecutable(self: Backend, executable: ExecutableHandle) void {
        return switch (self) {
            .mlx_metal => |backend| backend.destroyExecutable(executable),
            .metalcpp => |backend| backend.destroyExecutable(executable),
        };
    }

    /// Registers a typed custom-call target.
    pub fn registerCustomCall(self: Backend, registration: CustomCallRegistration) Error!void {
        return switch (self) {
            .mlx_metal => |backend| backend.registerCustomCall(registration),
            .metalcpp => |backend| backend.registerCustomCall(registration),
        };
    }

    /// Registers a named binary custom-call target.
    pub fn registerBinaryCustomCall(self: Backend, target: []const u8, op_name: []const u8) Error!void {
        return switch (self) {
            .mlx_metal => |backend| backend.registerBinaryCustomCall(target, op_name),
            .metalcpp => |backend| backend.registerBinaryCustomCall(target, op_name),
        };
    }

    /// Registers an identity custom-call target.
    pub fn registerIdentityCustomCall(self: Backend, target: []const u8) Error!void {
        return switch (self) {
            .mlx_metal => |backend| backend.registerIdentityCustomCall(target),
            .metalcpp => |backend| backend.registerIdentityCustomCall(target),
        };
    }

    /// Registers a named unary custom-call target.
    pub fn registerUnaryCustomCall(self: Backend, target: []const u8, op_name: []const u8) Error!void {
        return switch (self) {
            .mlx_metal => |backend| backend.registerUnaryCustomCall(target, op_name),
            .metalcpp => |backend| backend.registerUnaryCustomCall(target, op_name),
        };
    }

    /// Registers the built-in square-root custom-call marker.
    pub fn registerUnarySqrtCustomCall(self: Backend, target: []const u8) Error!void {
        return switch (self) {
            .mlx_metal => |backend| backend.registerUnarySqrtCustomCall(target),
            .metalcpp => |backend| backend.registerUnarySqrtCustomCall(target),
        };
    }

    /// Registers the built-in binary-add custom-call marker.
    pub fn registerBinaryAddCustomCall(self: Backend, target: []const u8) Error!void {
        return switch (self) {
            .mlx_metal => |backend| backend.registerBinaryAddCustomCall(target),
            .metalcpp => |backend| backend.registerBinaryAddCustomCall(target),
        };
    }

    /// Removes a custom-call target.
    pub fn unregisterCustomCall(self: Backend, target: []const u8) void {
        return switch (self) {
            .mlx_metal => |backend| backend.unregisterCustomCall(target),
            .metalcpp => |backend| backend.unregisterCustomCall(target),
        };
    }

    /// Returns the custom-call registry version for compile-cache keys.
    pub fn customCallRegistryVersion(self: Backend) u64 {
        return switch (self) {
            .mlx_metal => |backend| backend.customCallRegistryVersion(),
            .metalcpp => |backend| backend.customCallRegistryVersion(),
        };
    }

    /// Copies backend-owned storage to host bytes.
    pub fn copyToHost(self: Backend, src: BufferHandle, dst: []u8) Error!void {
        return switch (self) {
            .mlx_metal => |backend| backend.copyToHost(src, dst),
            .metalcpp => |backend| backend.copyToHost(src, dst),
        };
    }

    /// Destroys backend-owned device storage.
    pub fn destroyBuffer(self: Backend, buffer: BufferHandle) void {
        return switch (self) {
            .mlx_metal => |backend| backend.destroyBuffer(buffer),
            .metalcpp => |backend| backend.destroyBuffer(buffer),
        };
    }
};

/// Creates the backend selected by `PJRTX_RUNTIME_BACKEND`, defaulting to MLX/Metal.
pub fn createFromEnv() Backend {
    const name = envBackendName() orelse return .{ .mlx_metal = mlx_metal.create() };
    if (std.mem.eql(u8, name, "metalcpp") or std.mem.eql(u8, name, "metal_cpp")) {
        return .{ .metalcpp = metalcpp.create() };
    }
    return .{ .mlx_metal = mlx_metal.create() };
}

/// Creates the default MLX/Metal backend for tests and explicit runtime construction.
pub fn create() Backend {
    return .{ .mlx_metal = mlx_metal.create() };
}

fn envBackendName() ?[]const u8 {
    const raw = std.c.getenv("PJRTX_RUNTIME_BACKEND") orelse std.c.getenv("PJRTX_BACKEND") orelse return null;
    const text = std.mem.span(raw);
    if (text.len == 0) return null;
    return text;
}

test "runtime backend selection defaults to mlx metal" {
    try std.testing.expectEqualStrings("metal_mlx", createFromEnv().capabilities().name);
}
