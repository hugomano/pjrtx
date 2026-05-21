const std = @import("std");

const execution_call = @import("execution_call.zig");
const executable_mod = @import("executable.zig");

const Buffer = @import("buffer.zig").Buffer;
const CompiledExecutable = executable_mod.CompiledExecutable;
const ExecutionContext = executable_mod.Context;

/// Result of one per-device executable dispatch.
pub const ExecutionResult = execution_call.Result;

/// Errors produced while validating and dispatching a compiled executable.
pub const ExecutionError = execution_call.Error;

/// Executes one device slice of a compiled executable through resident backend storage.
pub fn executeCompiledExecutable(
    executable: *CompiledExecutable,
    allocator: std.mem.Allocator,
    context: ExecutionContext,
    device_index: usize,
    arguments: []const *Buffer,
) ExecutionError!ExecutionResult {
    return execution_call.execute(executable, allocator, context, device_index, arguments);
}
