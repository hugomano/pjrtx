const std = @import("std");

const env = @import("profiling_env.zig");
const stats_mod = @import("profiling_stats.zig");

const Execute = stats_mod.Execute;
const ScheduleKind = stats_mod.ScheduleKind;

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
    if (!env.traceEnabled()) return;
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
    if (!env.traceEnabled()) return;
    std.debug.print(
        "pjrtx_trace event=materialization_error detail={s} value={d} reason={s}\n",
        .{ snapshot.detail, snapshot.value_index, snapshot.reason },
    );
}
