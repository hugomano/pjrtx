const std = @import("std");

const mlx_call = @import("mlx_call.zig");
const profiling_mod = @import("profiling.zig");

const DefaultSmallControlBytes: usize = 4096;
const DefaultMinStableInputs: usize = 8;

/// Opaque MLX/Metal buffer handle tracked by argument-capture state.
pub const BufferHandle = *anyopaque;

/// Describes when MLX argument-captured programs may retain stable inputs.
/// The default policy preserves the normal fast path; environment overrides are
/// intended for capture-pressure experiments without changing executable shape.
pub const Policy = struct {
    /// Maximum byte size for scalar/control-like integer parameters kept dynamic during initial capture.
    small_control_bytes: usize = DefaultSmallControlBytes,
    /// Minimum stable full-width inputs required before creating a captured-input program.
    min_stable_inputs: usize = DefaultMinStableInputs,
    /// Optional experiment limit that disables captured-input programs above this stable-input count.
    max_stable_inputs: ?usize = null,

    /// Returns the current process policy for MLX argument-captured programs.
    pub fn current() Policy {
        return .{
            .small_control_bytes = profiling_mod.argumentCaptureSmallControlBytes(DefaultSmallControlBytes),
            .min_stable_inputs = profiling_mod.argumentCaptureMinStableInputs(DefaultMinStableInputs),
            .max_stable_inputs = profiling_mod.argumentCaptureMaxStableInputs(),
        };
    }

    /// Returns whether a full program input set should be replaced by a
    /// captured-input program with the supplied number of dynamic inputs.
    pub fn allowsCapture(self: Policy, full_input_count: usize, dynamic_input_count: usize) bool {
        if (dynamic_input_count >= full_input_count) return false;
        const stable_input_count = full_input_count - dynamic_input_count;
        if (stable_input_count < self.min_stable_inputs) return false;
        if (self.max_stable_inputs) |limit| {
            if (stable_input_count > limit) return false;
        }
        return true;
    }

    /// Returns whether a parameter should stay dynamic in an initial capture.
    pub fn treatsSmallControlAsDynamic(self: Policy, element_type: anytype, dense_byte_size: usize) bool {
        if (dense_byte_size == 0 or dense_byte_size > self.small_control_bytes) return false;
        return switch (element_type) {
            .pred, .s8, .s16, .s32, .s64, .u8, .u16, .u32, .u64 => true,
            else => false,
        };
    }
};

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

    /// Returns how many full-width inputs are captured rather than supplied dynamically.
    pub fn capturedArgumentCount(self: State, argument_count: usize) usize {
        var count: usize = 0;
        var dynamic_index_cursor: usize = 0;
        var index: usize = 0;
        while (index < argument_count) : (index += 1) {
            if (dynamic_index_cursor < self.dynamic_indices.len and self.dynamic_indices[dynamic_index_cursor] == index) {
                dynamic_index_cursor += 1;
                continue;
            }
            count += 1;
        }
        return count;
    }

    /// Returns how many recorded baseline inputs are captured by the MLX program.
    pub fn capturedBaselineCount(self: State) usize {
        return self.capturedArgumentCount(self.previous_arguments.len);
    }

    /// Records the stable argument baseline used by captured-program reuse.
    pub fn rememberBaseline(self: *State, allocator: std.mem.Allocator, arguments: []const BufferHandle) !void {
        if (self.previous_arguments.len != arguments.len) {
            const previous_arguments = try allocator.alloc(?BufferHandle, arguments.len);
            allocator.free(self.previous_arguments);
            self.previous_arguments = previous_arguments;
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

fn fakeHandle(index: usize) BufferHandle {
    return @ptrFromInt(0x1000 + index * 0x10);
}

test "argument capture pressure counts captured inputs" {
    const allocator = std.testing.allocator;
    var state: State = .{
        .dynamic_indices = try allocator.dupe(u64, &.{ 1, 3 }),
    };
    defer state.deinit(allocator);

    const arguments = [_]BufferHandle{
        fakeHandle(0),
        fakeHandle(1),
        fakeHandle(2),
        fakeHandle(3),
        fakeHandle(4),
    };
    try state.rememberBaseline(allocator, &arguments);

    try std.testing.expectEqual(@as(usize, arguments.len), state.previous_arguments.len);
    try std.testing.expectEqual(@as(usize, 3), state.capturedBaselineCount());
    try std.testing.expectEqual(@as(usize, 3), state.capturedArgumentCount(arguments.len));

    const dynamic_changed = [_]BufferHandle{
        fakeHandle(0),
        fakeHandle(10),
        fakeHandle(2),
        fakeHandle(30),
        fakeHandle(4),
    };
    try std.testing.expect(state.matches(&dynamic_changed));

    const stable_changed = [_]BufferHandle{
        fakeHandle(0),
        fakeHandle(10),
        fakeHandle(20),
        fakeHandle(30),
        fakeHandle(4),
    };
    try std.testing.expect(!state.matches(&stable_changed));
}

test "argument capture policy gates stable input pressure" {
    const default_policy: Policy = .{};
    try std.testing.expect(default_policy.allowsCapture(296, 5));
    try std.testing.expect(!default_policy.allowsCapture(7, 0));
    try std.testing.expect(!default_policy.allowsCapture(5, 5));

    const limited_policy: Policy = .{ .max_stable_inputs = 64 };
    try std.testing.expect(!limited_policy.allowsCapture(296, 5));
    try std.testing.expect(limited_policy.allowsCapture(69, 5));
}
