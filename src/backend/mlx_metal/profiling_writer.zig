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

/// Summary values required to render executable-level Metal graph compile coverage.
pub const MetalGraphCompileSnapshot = struct {
    plan_address: usize,
    value_count: usize,
    instruction_count: usize,
    output_count: usize,
    device_count: usize,
    compiled_device_count: usize,
    schedule_item_count: usize = 0,
    schedule_node_count: usize = 0,
    schedule_fusion_group_count: usize = 0,
    schedule_materialization_boundary_count: usize = 0,
    input_value_count: usize = 0,
    constant_input_count: usize = 0,
    step_count: usize = 0,
    node_step_count: usize = 0,
    fusion_step_count: usize = 0,
    alias_step_count: usize = 0,
    fallback_group_count: usize = 0,
    fallback_node_step_count: usize = 0,
    planned_structural_step_count: usize = 0,
    planned_view_step_count: usize = 0,
    planned_map_step_count: usize = 0,
    planned_reduction_step_count: usize = 0,
    planned_matmul_step_count: usize = 0,
    planned_control_flow_step_count: usize = 0,
    planned_materialize_step_count: usize = 0,
    planned_library_call_step_count: usize = 0,
    planned_other_step_count: usize = 0,
    fallback_structural_step_count: usize = 0,
    fallback_view_step_count: usize = 0,
    fallback_map_step_count: usize = 0,
    fallback_reduction_step_count: usize = 0,
    fallback_matmul_step_count: usize = 0,
    fallback_control_flow_step_count: usize = 0,
    fallback_materialize_step_count: usize = 0,
    fallback_library_call_step_count: usize = 0,
    fallback_other_step_count: usize = 0,
    release_step_count: usize = 0,
    release_value_count: usize = 0,
    dot_step_count: usize = 0,
    tiled_dot_step_count: usize = 0,
    dot_group_step_count: usize = 0,
    dot_group_node_count: usize = 0,
    reduce_step_count: usize = 0,
    tiled_reduce_step_count: usize = 0,
    reduce_max_index_step_count: usize = 0,
    tiled_reduce_max_index_step_count: usize = 0,
    reason: []const u8,
    first_unsupported_op: ?[]const u8 = null,
    compile_error: ?[]const u8 = null,
};

/// Summary values required to render executable-level Metal graph execution coverage.
pub const MetalGraphExecuteSnapshot = struct {
    executable_address: usize,
    device_index: usize,
    status: []const u8,
    reason: []const u8,
    output_count: usize = 0,
    elapsed_us: u64 = 0,
};

/// Emits the executable-level backend profile event.
pub fn writeExecute(snapshot: ExecutableSnapshot, device_index: usize, argument_count: usize, output_count: usize, profile: Execute) void {
    std.debug.print(
        "pjrtx_profile event=backend_execute executable=0x{x} device={d} args={d} outputs={d} schedule_items={d} nodes={d} fusion_groups={d} materialization_boundaries={d} wall_us={d} schedule_us={d} schedule_peak_us={d} node_us={d} node_peak_us={d} fusion_us={d} fusion_peak_us={d} materialization_us={d} materialization_peak_us={d} compiled_program_us={d} compiled_program_peak_us={d} compiled_program_host_enqueue_us={d} compiled_program_device_sync_wait_us={d} compiled_program_device_sync_measured={d} metal_graph_us={d} metal_graph_peak_us={d} output_clone_us={d}\n",
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
            profile.compiled_program_host_enqueue_us,
            profile.compiled_program_device_sync_wait_us,
            @intFromBool(profile.compiled_program_device_sync_measured),
            profile.metal_graph_us,
            profile.metal_graph_peak_us,
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

const StepFamilyCounts = struct {
    structural: usize = 0,
    view: usize = 0,
    map: usize = 0,
    reduction: usize = 0,
    matmul: usize = 0,
    control_flow: usize = 0,
    materialize: usize = 0,
    library_call: usize = 0,
    other: usize = 0,

    fn any(self: StepFamilyCounts) bool {
        return self.structural != 0 or
            self.view != 0 or
            self.map != 0 or
            self.reduction != 0 or
            self.matmul != 0 or
            self.control_flow != 0 or
            self.materialize != 0 or
            self.library_call != 0 or
            self.other != 0;
    }
};

fn plannedStepFamilyCounts(snapshot: MetalGraphCompileSnapshot) StepFamilyCounts {
    var counts = StepFamilyCounts{
        .structural = snapshot.planned_structural_step_count,
        .view = snapshot.planned_view_step_count,
        .map = snapshot.planned_map_step_count,
        .reduction = snapshot.planned_reduction_step_count,
        .matmul = snapshot.planned_matmul_step_count,
        .control_flow = snapshot.planned_control_flow_step_count,
        .materialize = snapshot.planned_materialize_step_count,
        .library_call = snapshot.planned_library_call_step_count,
        .other = snapshot.planned_other_step_count,
    };
    if (!counts.any()) {
        counts.matmul = snapshot.dot_step_count;
        counts.reduction = snapshot.reduce_step_count + snapshot.reduce_max_index_step_count;
    }
    return counts;
}

fn fallbackStepFamilyCounts(snapshot: MetalGraphCompileSnapshot) StepFamilyCounts {
    var counts = StepFamilyCounts{
        .structural = snapshot.fallback_structural_step_count,
        .view = snapshot.fallback_view_step_count,
        .map = snapshot.fallback_map_step_count,
        .reduction = snapshot.fallback_reduction_step_count,
        .matmul = snapshot.fallback_matmul_step_count,
        .control_flow = snapshot.fallback_control_flow_step_count,
        .materialize = snapshot.fallback_materialize_step_count,
        .library_call = snapshot.fallback_library_call_step_count,
        .other = snapshot.fallback_other_step_count,
    };
    if (!counts.any()) counts.other = snapshot.fallback_node_step_count;
    return counts;
}

/// Emits executable-level Metal graph compile coverage.
pub fn writeMetalGraphCompile(snapshot: MetalGraphCompileSnapshot) void {
    const planned_counts = plannedStepFamilyCounts(snapshot);
    const fallback_counts = fallbackStepFamilyCounts(snapshot);
    std.debug.print(
        "pjrtx_profile event=metal_graph_compile plan=0x{x} values={d} instructions={d} outputs={d} devices={d} compiled_devices={d} schedule_items={d} schedule_nodes={d} schedule_fusion_groups={d} schedule_materialization_boundaries={d} input_values={d} constant_inputs={d} steps={d} node_steps={d} fusion_steps={d} alias_steps={d} fallback_groups={d} fallback_node_steps={d}",
        .{
            snapshot.plan_address,
            snapshot.value_count,
            snapshot.instruction_count,
            snapshot.output_count,
            snapshot.device_count,
            snapshot.compiled_device_count,
            snapshot.schedule_item_count,
            snapshot.schedule_node_count,
            snapshot.schedule_fusion_group_count,
            snapshot.schedule_materialization_boundary_count,
            snapshot.input_value_count,
            snapshot.constant_input_count,
            snapshot.step_count,
            snapshot.node_step_count,
            snapshot.fusion_step_count,
            snapshot.alias_step_count,
            snapshot.fallback_group_count,
            snapshot.fallback_node_step_count,
        },
    );
    std.debug.print(
        " planned_structural_steps={d} planned_view_steps={d} planned_map_steps={d} planned_reduction_steps={d} planned_matmul_steps={d} planned_control_flow_steps={d} planned_materialize_steps={d} planned_library_call_steps={d} planned_other_steps={d} fallback_structural_steps={d} fallback_view_steps={d} fallback_map_steps={d} fallback_reduction_steps={d} fallback_matmul_steps={d} fallback_control_flow_steps={d} fallback_materialize_steps={d} fallback_library_call_steps={d} fallback_other_steps={d}",
        .{
            planned_counts.structural,
            planned_counts.view,
            planned_counts.map,
            planned_counts.reduction,
            planned_counts.matmul,
            planned_counts.control_flow,
            planned_counts.materialize,
            planned_counts.library_call,
            planned_counts.other,
            fallback_counts.structural,
            fallback_counts.view,
            fallback_counts.map,
            fallback_counts.reduction,
            fallback_counts.matmul,
            fallback_counts.control_flow,
            fallback_counts.materialize,
            fallback_counts.library_call,
            fallback_counts.other,
        },
    );
    std.debug.print(
        " release_steps={d} release_values={d} dot_steps={d} tiled_dot_steps={d} dot_group_steps={d} dot_group_nodes={d} reduce_steps={d} tiled_reduce_steps={d} reduce_max_index_steps={d} tiled_reduce_max_index_steps={d} reason={s}",
        .{
            snapshot.release_step_count,
            snapshot.release_value_count,
            snapshot.dot_step_count,
            snapshot.tiled_dot_step_count,
            snapshot.dot_group_step_count,
            snapshot.dot_group_node_count,
            snapshot.reduce_step_count,
            snapshot.tiled_reduce_step_count,
            snapshot.reduce_max_index_step_count,
            snapshot.tiled_reduce_max_index_step_count,
            snapshot.reason,
        },
    );
    if (snapshot.first_unsupported_op) |op| {
        std.debug.print(" first_unsupported_op={s}", .{op});
    }
    if (snapshot.compile_error) |message| {
        std.debug.print(" compile_error=\"{s}\"", .{message});
    }
    std.debug.print("\n", .{});
}

/// Emits executable-level Metal graph execution coverage.
pub fn writeMetalGraphExecute(snapshot: MetalGraphExecuteSnapshot) void {
    std.debug.print(
        "pjrtx_profile event=metal_graph_execute executable=0x{x} device={d} status={s} reason={s} outputs={d} elapsed_us={d}\n",
        .{
            snapshot.executable_address,
            snapshot.device_index,
            snapshot.status,
            snapshot.reason,
            snapshot.output_count,
            snapshot.elapsed_us,
        },
    );
}
