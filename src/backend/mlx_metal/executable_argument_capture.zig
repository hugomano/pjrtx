const std = @import("std");

const mlx_call = @import("mlx_call.zig");
const profiling_mod = @import("profiling.zig");

/// Opaque MLX/Metal buffer handle tracked by argument-capture state.
pub const BufferHandle = *anyopaque;

/// Tracks stable captured arguments and dynamic argument indices for MLX compile reuse.
pub const State = struct {
    previous_arguments: []?BufferHandle = &.{},
    dynamic_indices: []u64 = &.{},
    program_handle: ?mlx_call.ProgramHandle = null,

    /// Returns whether the current argument set matches the captured baseline.
    pub fn matches(self: State, arguments: []const BufferHandle) bool {
        if (self.previous_arguments.len != arguments.len) return false;
        var dynamic_index_cursor: usize = 0;
        for (arguments, 0..) |argument, index| {
            if (dynamic_index_cursor < self.dynamic_indices.len and self.dynamic_indices[dynamic_index_cursor] == index) {
                dynamic_index_cursor += 1;
                continue;
            }
            if (self.previous_arguments[index] != argument) return false;
        }
        return true;
    }

    /// Records the stable argument baseline used by captured-program reuse.
    pub fn rememberBaseline(self: *State, allocator: std.mem.Allocator, arguments: []const BufferHandle) !void {
        if (self.previous_arguments.len != arguments.len) {
            allocator.free(self.previous_arguments);
            self.previous_arguments = try allocator.alloc(?BufferHandle, arguments.len);
        }
        for (arguments, 0..) |argument, index| {
            self.previous_arguments[index] = argument;
        }
    }

    /// Releases any captured MLX program and argument baseline storage.
    pub fn reset(self: *State, allocator: std.mem.Allocator) void {
        if (self.program_handle) |program| mlx_call.programDestroy(program);
        self.program_handle = null;
        allocator.free(self.previous_arguments);
        self.previous_arguments = &.{};
        allocator.free(self.dynamic_indices);
        self.dynamic_indices = &.{};
    }

    /// Releases storage held by this capture state during executable teardown.
    pub fn deinit(self: *State, allocator: std.mem.Allocator) void {
        self.reset(allocator);
    }
};

/// Allocates initialized argument-capture slots for each executable device.
pub fn allocStates(allocator: std.mem.Allocator, device_count: usize) ![]State {
    const states = try allocator.alloc(State, device_count);
    for (states) |*state| state.* = .{};
    return states;
}

/// Releases all argument-capture slots.
pub fn destroyStates(allocator: std.mem.Allocator, states: []State) void {
    for (states) |*state| {
        state.deinit(allocator);
    }
}

/// Locks the executable argument-capture state for mutation by execution.
pub fn lock(executable: anytype) void {
    executable.argument_capture_mutex.lockUncancelable(profiling_mod.backendIo());
}

/// Unlocks the executable argument-capture state after mutation.
pub fn unlock(executable: anytype) void {
    executable.argument_capture_mutex.unlock(profiling_mod.backendIo());
}
