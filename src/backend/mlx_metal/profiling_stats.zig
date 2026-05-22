const std = @import("std");

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
    compiled_program_host_enqueue_us: u64 = 0,
    compiled_program_device_sync_wait_us: u64 = 0,
    compiled_program_device_sync_measured: bool = false,
    metal_graph_us: u64 = 0,
    metal_graph_peak_us: u64 = 0,

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

    /// Records counters reported by the compiled MLX program boundary.
    pub fn recordCompiledProgramBoundary(self: *Execute, host_enqueue_us: u64, device_sync_wait_us: u64, measured_device_sync: bool) void {
        self.compiled_program_host_enqueue_us +|= host_enqueue_us;
        self.compiled_program_device_sync_wait_us +|= device_sync_wait_us;
        self.compiled_program_device_sync_measured = self.compiled_program_device_sync_measured or measured_device_sync;
    }

    /// Records elapsed time for an executable-level generated Metal graph.
    pub fn recordMetalGraph(self: *Execute, elapsed_us: u64) void {
        self.schedule_us +|= elapsed_us;
        self.schedule_peak_us = @max(self.schedule_peak_us, elapsed_us);
        self.metal_graph_us +|= elapsed_us;
        self.metal_graph_peak_us = @max(self.metal_graph_peak_us, elapsed_us);
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
    stats.metal_graph_us_total +|= profile.metal_graph_us;
    stats.metal_graph_us_peak = @max(stats.metal_graph_us_peak, profile.metal_graph_peak_us);
}

test "execute profile records schedule buckets" {
    var profile: Execute = .{};
    profile.recordScheduleItem(.node, 7);
    profile.recordScheduleItem(.fusion_group, 11);
    profile.recordCompiledProgram(13);
    profile.recordMetalGraph(19);
    profile.recordOutputClone(17);

    try std.testing.expectEqual(@as(u64, 50), profile.schedule_us);
    try std.testing.expectEqual(@as(u64, 19), profile.schedule_peak_us);
    try std.testing.expectEqual(@as(u64, 7), profile.node_us);
    try std.testing.expectEqual(@as(u64, 11), profile.fusion_group_us);
    try std.testing.expectEqual(@as(u64, 13), profile.compiled_program_us);
    try std.testing.expectEqual(@as(u64, 19), profile.metal_graph_us);
    try std.testing.expectEqual(@as(u64, 17), profile.output_clone_us);
}
