const std = @import("std");

const io = std.Io.Threaded.global_single_threaded.io();

/// Returns the backend IO handle used for timestamps and non-cancelable locks.
pub fn backendIo() std.Io {
    return io;
}

/// Aggregates one backend executable invocation profile.
/// Times are recorded in microseconds and saturating-add into executable stats.
pub const Execute = struct {
    wall_us: u64 = 0,
    schedule_us: u64 = 0,
    schedule_peak_us: u64 = 0,
    node_us: u64 = 0,
    node_peak_us: u64 = 0,
    fusion_group_us: u64 = 0,
    fusion_group_peak_us: u64 = 0,
    materialization_eval_us: u64 = 0,
    materialization_eval_peak_us: u64 = 0,
    output_clone_us: u64 = 0,
    output_clone_peak_us: u64 = 0,
    compiled_program_us: u64 = 0,
    compiled_program_peak_us: u64 = 0,

    /// Records elapsed time for a schedule item kind.
    pub fn recordScheduleItem(self: *Execute, kind: ScheduleKind, elapsed_us: u64) void {
        self.schedule_us +|= elapsed_us;
        self.schedule_peak_us = @max(self.schedule_peak_us, elapsed_us);
        switch (kind) {
            .node => {
                self.node_us +|= elapsed_us;
                self.node_peak_us = @max(self.node_peak_us, elapsed_us);
            },
            .fusion_group => {
                self.fusion_group_us +|= elapsed_us;
                self.fusion_group_peak_us = @max(self.fusion_group_peak_us, elapsed_us);
            },
            .materialization_boundary => {
                self.materialization_eval_us +|= elapsed_us;
                self.materialization_eval_peak_us = @max(self.materialization_eval_peak_us, elapsed_us);
            },
        }
    }

    /// Records elapsed time for a compiled MLX program execution.
    pub fn recordCompiledProgram(self: *Execute, elapsed_us: u64) void {
        self.schedule_us +|= elapsed_us;
        self.schedule_peak_us = @max(self.schedule_peak_us, elapsed_us);
        self.compiled_program_us +|= elapsed_us;
        self.compiled_program_peak_us = @max(self.compiled_program_peak_us, elapsed_us);
    }

    /// Records elapsed time for output ownership cloning.
    pub fn recordOutputClone(self: *Execute, elapsed_us: u64) void {
        self.output_clone_us = elapsed_us;
        self.output_clone_peak_us = elapsed_us;
    }
};

/// Schedule item categories used by MLX backend profile aggregation.
pub const ScheduleKind = enum {
    node,
    fusion_group,
    materialization_boundary,
};

/// Summary values required to render executable-level profile events.
/// The struct avoids coupling this module to the concrete executable owner.
pub const ExecutableSnapshot = struct {
    address: usize,
    schedule_count: usize,
    node_count: usize,
    fusion_group_count: usize,
    materialization_boundary_count: usize,
};

/// Summary values required to render one schedule item profile event.
/// Callers populate optional labels from the backend program graph.
pub const ScheduleItemSnapshot = struct {
    executable_address: usize,
    schedule_index: usize,
    kind: ScheduleKind,
    index: usize,
    count: usize = 1,
    node_kind: ?[]const u8 = null,
    instruction_index: ?usize = null,
    op: ?[]const u8 = null,
    group_first_node: ?usize = null,
    group_last_node: ?usize = null,
    group_node_count: ?usize = null,
    group_ops: ?[]const []const u8 = null,
};

/// Summary values required to render a backend schedule failure trace.
pub const ScheduleFailureSnapshot = struct {
    schedule_kind: []const u8,
    schedule_index: usize,
    schedule_count: usize,
    err: []const u8,
    node_kind: ?[]const u8 = null,
    instruction_index: ?usize = null,
    op: ?[]const u8 = null,
    output_value: ?usize = null,
    dtype: ?[]const u8 = null,
    rank: ?usize = null,
    group_first_node: ?usize = null,
    group_last_node: ?usize = null,
    group_node_count: ?usize = null,
};

/// Summary values required to render a materialization failure trace.
pub const MaterializationFailureSnapshot = struct {
    detail: []const u8,
    value_index: usize,
    reason: []const u8,
};

/// Returns whether backend profile events should be collected.
pub fn enabled() bool {
    return envFlag("PJRTX_PROFILE");
}

/// Returns whether per-schedule-item profile events should be emitted.
pub fn verbose() bool {
    const text = envText("PJRTX_PROFILE") orelse return false;
    return std.ascii.eqlIgnoreCase(text, "verbose") or std.ascii.eqlIgnoreCase(text, "2");
}

/// Returns whether backend trace diagnostics should be emitted.
pub fn traceEnabled() bool {
    return envText("PJRTX_TRACE") != null;
}

/// Returns whether MLX compiled-program creation is enabled for executables.
pub fn programCompileEnabled() bool {
    const text = envText("PJRTX_MLX_PROGRAM_COMPILE") orelse return true;
    return text.len == 0 or (!std.mem.eql(u8, text, "0") and !std.ascii.eqlIgnoreCase(text, "false"));
}

/// Starts a monotonic timer when profiling is enabled.
/// A zero return value is a disabled timer and always has zero elapsed time.
pub fn start(is_enabled: bool) std.Io.Timestamp {
    return if (is_enabled) nowTimestamp() else .zero;
}

/// Returns the elapsed microseconds since `start`.
pub fn elapsedUs(start_timestamp: std.Io.Timestamp) u64 {
    if (start_timestamp.nanoseconds == 0) return 0;
    return @intCast(@max(start_timestamp.durationTo(nowTimestamp()).toMicroseconds(), 0));
}

/// Accumulates one execute profile into an executable stats struct.
/// The pointed-to value must expose the backend executable stats fields.
pub fn recordExecute(stats: anytype, profile: Execute) void {
    stats.execute_wall_us_total +|= profile.wall_us;
    stats.execute_wall_us_peak = @max(stats.execute_wall_us_peak, profile.wall_us);
    stats.schedule_us_total +|= profile.schedule_us;
    stats.schedule_us_peak = @max(stats.schedule_us_peak, profile.schedule_peak_us);
    stats.node_us_total +|= profile.node_us;
    stats.node_us_peak = @max(stats.node_us_peak, profile.node_peak_us);
    stats.fusion_group_us_total +|= profile.fusion_group_us;
    stats.fusion_group_us_peak = @max(stats.fusion_group_us_peak, profile.fusion_group_peak_us);
    stats.materialization_eval_us_total +|= profile.materialization_eval_us;
    stats.materialization_eval_us_peak = @max(stats.materialization_eval_us_peak, profile.materialization_eval_peak_us);
    stats.output_clone_us_total +|= profile.output_clone_us;
    stats.output_clone_us_peak = @max(stats.output_clone_us_peak, profile.output_clone_peak_us);
    stats.compiled_program_us_total +|= profile.compiled_program_us;
    stats.compiled_program_us_peak = @max(stats.compiled_program_us_peak, profile.compiled_program_peak_us);
}

/// Emits the executable-level backend profile event.
pub fn writeExecute(snapshot: ExecutableSnapshot, device_index: usize, argument_count: usize, output_count: usize, profile: Execute) void {
    std.debug.print(
        "pjrtx_profile event=backend_execute executable=0x{x} device={d} args={d} outputs={d} schedule_items={d} nodes={d} fusion_groups={d} materialization_boundaries={d} wall_us={d} schedule_us={d} schedule_peak_us={d} node_us={d} node_peak_us={d} fusion_us={d} fusion_peak_us={d} materialization_us={d} materialization_peak_us={d} compiled_program_us={d} compiled_program_peak_us={d} output_clone_us={d}\n",
        .{
            snapshot.address,
            device_index,
            argument_count,
            output_count,
            snapshot.schedule_count,
            snapshot.node_count,
            snapshot.fusion_group_count,
            snapshot.materialization_boundary_count,
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

/// Emits one schedule item profile event.
pub fn writeScheduleItem(snapshot: ScheduleItemSnapshot, elapsed_us: u64) void {
    std.debug.print(
        "pjrtx_profile event=backend_schedule_item executable=0x{x} schedule_index={d} kind={s} index={d} count={d} elapsed_us={d}",
        .{ snapshot.executable_address, snapshot.schedule_index, @tagName(snapshot.kind), snapshot.index, snapshot.count, elapsed_us },
    );
    switch (snapshot.kind) {
        .node => {
            if (snapshot.node_kind) |node_kind| {
                std.debug.print(" node_kind={s}", .{node_kind});
            }
            if (snapshot.instruction_index) |instruction_index| {
                std.debug.print(" instruction={d}", .{instruction_index});
            }
            if (snapshot.op) |op| {
                std.debug.print(" op={s}", .{op});
            }
        },
        .fusion_group => {
            if (snapshot.group_first_node) |first_node| {
                std.debug.print(" first_node={d}", .{first_node});
            }
            if (snapshot.group_last_node) |last_node| {
                std.debug.print(" last_node={d}", .{last_node});
            }
            if (snapshot.group_node_count) |node_count| {
                std.debug.print(" node_count={d}", .{node_count});
            }
            if (snapshot.group_ops) |ops| {
                std.debug.print(" ops=\"", .{});
                for (ops, 0..) |op, index| {
                    if (index != 0) std.debug.print(",", .{});
                    std.debug.print("{s}", .{op});
                }
                std.debug.print("\"", .{});
            }
        },
        .materialization_boundary => {},
    }
    std.debug.print("\n", .{});
}

/// Emits a backend schedule failure trace event.
pub fn writeScheduleFailure(snapshot: ScheduleFailureSnapshot) void {
    if (!traceEnabled()) return;
    std.debug.print(
        "pjrtx_trace event=backend_execute_error schedule_kind={s} schedule_index={d} schedule_count={d} err={s}",
        .{ snapshot.schedule_kind, snapshot.schedule_index, snapshot.schedule_count, snapshot.err },
    );
    if (snapshot.node_kind) |node_kind| {
        std.debug.print(" node_kind={s}", .{node_kind});
    }
    if (snapshot.instruction_index) |instruction_index| {
        std.debug.print(" instruction={d}", .{instruction_index});
    }
    if (snapshot.op) |op| {
        std.debug.print(" op={s}", .{op});
    }
    if (snapshot.output_value) |output_value| {
        std.debug.print(" output_value={d}", .{output_value});
    }
    if (snapshot.dtype) |dtype| {
        std.debug.print(" dtype={s}", .{dtype});
    }
    if (snapshot.rank) |rank| {
        std.debug.print(" rank={d}", .{rank});
    }
    if (snapshot.group_first_node) |first_node| {
        std.debug.print(" group_first_node={d}", .{first_node});
    }
    if (snapshot.group_last_node) |last_node| {
        std.debug.print(" group_last_node={d}", .{last_node});
    }
    if (snapshot.group_node_count) |node_count| {
        std.debug.print(" group_nodes={d}", .{node_count});
    }
    std.debug.print("\n", .{});
}

/// Emits a materialization failure trace event.
pub fn writeMaterializationFailure(snapshot: MaterializationFailureSnapshot) void {
    if (!traceEnabled()) return;
    std.debug.print(
        "pjrtx_trace event=materialization_error detail={s} value={d} reason={s}\n",
        .{ snapshot.detail, snapshot.value_index, snapshot.reason },
    );
}

fn envFlag(comptime name: [:0]const u8) bool {
    const text = envText(name) orelse return false;
    return text.len != 0 and !std.mem.eql(u8, text, "0") and !std.ascii.eqlIgnoreCase(text, "false");
}

fn envText(comptime name: [:0]const u8) ?[]const u8 {
    const value = std.c.getenv(name) orelse return null;
    return std.mem.span(value);
}

fn nowTimestamp() std.Io.Timestamp {
    return std.Io.Timestamp.now(io, .awake);
}

test "execute profile records schedule buckets" {
    var profile: Execute = .{};
    profile.recordScheduleItem(.node, 7);
    profile.recordScheduleItem(.fusion_group, 11);
    profile.recordCompiledProgram(13);
    profile.recordOutputClone(17);

    try std.testing.expectEqual(@as(u64, 31), profile.schedule_us);
    try std.testing.expectEqual(@as(u64, 13), profile.schedule_peak_us);
    try std.testing.expectEqual(@as(u64, 7), profile.node_us);
    try std.testing.expectEqual(@as(u64, 11), profile.fusion_group_us);
    try std.testing.expectEqual(@as(u64, 13), profile.compiled_program_us);
    try std.testing.expectEqual(@as(u64, 17), profile.output_clone_us);
}
