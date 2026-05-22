const std = @import("std");

const profiling_mod = @import("profiling.zig");

/// Runtime-visible execution counters for one compiled MLX/Metal executable.
pub const Stats = struct {
    resident_constant_count: usize = 0,
    resident_constant_bytes: usize = 0,
    program_value_count: usize = 0,
    program_node_count: usize = 0,
    program_edge_count: usize = 0,
    program_schedule_item_count: usize = 0,
    program_subprogram_count: usize = 0,
    program_control_flow_count: usize = 0,
    program_fusion_group_count: usize = 0,
    program_materialization_boundary_count: usize = 0,
    program_planned_release_count: usize = 0,
    program_planned_release_bytes: usize = 0,
    program_peak_live_value_count: usize = 0,
    program_peak_live_bytes: usize = 0,
    program_device_count: usize = 0,
    last_execute_device_index: usize = std.math.maxInt(usize),
    last_execute_local_hardware_id: i32 = -1,
    execute_count: usize = 0,
    compiled_program_execute_count: usize = 0,
    compiled_program_output_count: usize = 0,
    metal_graph_compile_count: usize = 0,
    metal_graph_execute_count: usize = 0,
    metal_graph_output_count: usize = 0,
    captured_program_execute_count: usize = 0,
    captured_program_dynamic_input_count: usize = 0,
    captured_program_captured_input_count: usize = 0,
    donation_alias_output_count: usize = 0,
    donation_alias_output_bytes: usize = 0,
    fusion_group_execute_count: usize = 0,
    materialization_eval_count: usize = 0,
    materialization_eval_buffer_count: usize = 0,
    released_intermediate_count: usize = 0,
    borrowed_constant_nodes: usize = 0,
    execute_wall_us_total: u64 = 0,
    execute_wall_us_peak: u64 = 0,
    schedule_us_total: u64 = 0,
    schedule_us_peak: u64 = 0,
    node_us_total: u64 = 0,
    node_us_peak: u64 = 0,
    fusion_group_us_total: u64 = 0,
    fusion_group_us_peak: u64 = 0,
    materialization_eval_us_total: u64 = 0,
    materialization_eval_us_peak: u64 = 0,
    output_clone_us_total: u64 = 0,
    output_clone_us_peak: u64 = 0,
    compiled_program_us_total: u64 = 0,
    compiled_program_us_peak: u64 = 0,
    metal_graph_us_total: u64 = 0,
    metal_graph_us_peak: u64 = 0,
};

/// Builds the initial stats snapshot for a newly resident executable.
pub fn init(program: anytype, liveness_stats: anytype, resident_constant_count: usize, resident_constant_bytes: usize, device_count: usize) Stats {
    return .{
        .resident_constant_count = resident_constant_count,
        .resident_constant_bytes = resident_constant_bytes,
        .program_value_count = program.values.len,
        .program_node_count = program.nodes.len,
        .program_edge_count = program.edges.len,
        .program_schedule_item_count = program.schedule.len,
        .program_subprogram_count = program.subprograms.len,
        .program_control_flow_count = program.control_flows.len,
        .program_fusion_group_count = program.fusion_groups.len,
        .program_materialization_boundary_count = program.materialization_boundaries.len,
        .program_planned_release_count = liveness_stats.planned_release_count,
        .program_planned_release_bytes = liveness_stats.planned_release_bytes,
        .program_peak_live_value_count = liveness_stats.peak_live_value_count,
        .program_peak_live_bytes = liveness_stats.peak_live_bytes,
        .program_device_count = device_count,
    };
}

/// Returns a synchronized copy of the executable counters.
pub fn snapshot(executable: anytype) Stats {
    lock(executable);
    defer unlock(executable);
    return executable.stats;
}

/// Records a successful device execution and the concrete hardware target.
pub fn recordExecute(executable: anytype, device_index: usize, local_hardware_id: i32) void {
    lock(executable);
    defer unlock(executable);
    executable.stats.execute_count += 1;
    executable.stats.last_execute_device_index = device_index;
    executable.stats.last_execute_local_hardware_id = local_hardware_id;
}

/// Records one fused schedule group execution.
pub fn recordFusionGroupExecute(executable: anytype) void {
    lock(executable);
    defer unlock(executable);
    executable.stats.fusion_group_execute_count += 1;
}

/// Records one MLX compiled-program execution and its output count.
pub fn recordCompiledProgramExecute(executable: anytype, output_count: usize) void {
    lock(executable);
    defer unlock(executable);
    executable.stats.compiled_program_execute_count += 1;
    executable.stats.compiled_program_output_count += output_count;
}

/// Records resident executable-level generated Metal graph compilation.
pub fn recordMetalGraphCompile(executable: anytype, compiled_device_count: usize) void {
    lock(executable);
    defer unlock(executable);
    executable.stats.metal_graph_compile_count += compiled_device_count;
}

/// Records one executable-level generated Metal graph execution and its output count.
pub fn recordMetalGraphExecute(executable: anytype, output_count: usize) void {
    lock(executable);
    defer unlock(executable);
    executable.stats.metal_graph_execute_count += 1;
    executable.stats.metal_graph_output_count += output_count;
}

/// Records one argument-captured MLX program execution.
pub fn recordCapturedProgramExecute(executable: anytype, dynamic_count: usize, captured_count: usize) void {
    lock(executable);
    defer unlock(executable);
    executable.stats.captured_program_execute_count += 1;
    executable.stats.captured_program_dynamic_input_count += dynamic_count;
    executable.stats.captured_program_captured_input_count += captured_count;
}

/// Records an explicit materialization boundary evaluation.
pub fn recordMaterializationEval(executable: anytype, buffer_count: usize) void {
    lock(executable);
    defer unlock(executable);
    executable.stats.materialization_eval_count += 1;
    executable.stats.materialization_eval_buffer_count += buffer_count;
}

/// Records planned intermediate value releases after their final use.
pub fn recordReleasedIntermediateValues(executable: anytype, count: usize) void {
    lock(executable);
    defer unlock(executable);
    executable.stats.released_intermediate_count += count;
}

/// Records a schedule node that borrowed a resident constant buffer.
pub fn recordBorrowedConstantNode(executable: anytype) void {
    lock(executable);
    defer unlock(executable);
    executable.stats.borrowed_constant_nodes += 1;
}

/// Accumulates one backend execution profile into executable counters.
pub fn recordExecuteProfile(executable: anytype, profile: profiling_mod.Execute) void {
    lock(executable);
    defer unlock(executable);
    profiling_mod.recordExecute(&executable.stats, profile);
}

fn lock(executable: anytype) void {
    executable.stats_mutex.lockUncancelable(profiling_mod.backendIo());
}

fn unlock(executable: anytype) void {
    executable.stats_mutex.unlock(profiling_mod.backendIo());
}
