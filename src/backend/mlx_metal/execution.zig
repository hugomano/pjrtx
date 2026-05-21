const std = @import("std");

const executable_mod = @import("executable.zig");

const call_mod = @import("execution_call.zig");
const compiled_program_mod = @import("execution_compiled_program.zig");
const types = @import("execution_types.zig");

/// Opaque MLX/Metal buffer handle accepted by backend execution.
pub const BufferHandle = types.BufferHandle;
/// Opaque compiled executable handle owned by the MLX backend.
pub const ExecutableHandle = types.ExecutableHandle;
/// Opaque asynchronous execution event handle reserved for future MLX support.
pub const ExecutionEventHandle = types.ExecutionEventHandle;
/// Errors reported by MLX/Metal execution and executable teardown.
pub const Error = types.Error;
/// Device buffer returned for one executable output.
pub const ExecutableOutput = types.ExecutableOutput;
/// Completion mode for an MLX/Metal execute call.
pub const ExecutionCompletionKind = types.ExecutionCompletionKind;
/// Completion token returned with execution outputs.
pub const ExecutionCompletion = types.ExecutionCompletion;
/// State reported for an asynchronous backend execution event.
pub const ExecutionEventState = types.ExecutionEventState;
/// Status payload for an asynchronous backend execution event.
pub const ExecutionEventStatus = types.ExecutionEventStatus;
/// Result of executing a compiled MLX/Metal executable on one device.
pub const ExecutionResult = types.ExecutionResult;

/// Executes a compiled MLX/Metal executable on one device.
pub fn execute(allocator: std.mem.Allocator, executable_handle: ExecutableHandle, device_index: usize, arguments: []const BufferHandle) Error!?ExecutionResult {
    return call_mod.executeExecutable(allocator, executable_handle, device_index, arguments);
}

/// Reports the status of a backend execution event.
pub fn eventStatus(event: ExecutionEventHandle) Error!ExecutionEventStatus {
    _ = event;
    return .{
        .state = .failed,
        .message = "MLX Metal backend does not expose asynchronous execution event handles",
    };
}

/// Releases a backend execution event handle.
pub fn destroyEvent(event: ExecutionEventHandle) void {
    _ = event;
}

/// Returns accumulated execution statistics for a compiled executable.
pub fn stats(executable_handle: ExecutableHandle) executable_mod.Stats {
    return executable_mod.Executable.fromHandle(executable_handle).snapshotStats();
}

/// Destroys a compiled executable and all resident backend resources.
pub fn destroy(executable_handle: ExecutableHandle) void {
    executable_mod.Executable.fromHandle(executable_handle).deinit();
}

/// Builds one MLX compiled-program trace from captured executable context.
pub const compiledProgramBuildCallback = compiled_program_mod.compiledProgramBuildCallback;
