const ir = @import("src/compiler/ir");

const program_mod = @import("program.zig");

/// Opaque MLX/Metal buffer handle accepted by backend execution.
pub const BufferHandle = *anyopaque;
/// Opaque compiled executable handle owned by the MLX backend.
pub const ExecutableHandle = *anyopaque;
/// Opaque asynchronous execution event handle reserved for future MLX support.
pub const ExecutionEventHandle = *anyopaque;
/// Errors reported by MLX/Metal execution and executable teardown.
pub const Error = program_mod.Error;

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
