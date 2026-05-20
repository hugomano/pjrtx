const std = @import("std");

const ir = @import("src/compiler/ir");
const mlx_call = @import("mlx_call.zig");
const program_mod = @import("program.zig");

/// Opaque MLX/Metal buffer handle owned by runtime buffer/executable storage.
pub const BufferHandle = *anyopaque;

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
};

/// Owns the compiled backend program, resident constants, and execution counters.
pub const Executable = struct {
    allocator: std.mem.Allocator,
    plan: *const ir.ExecutablePlan,
    device_local_hardware_ids: []i32,
    constant_handles: []?BufferHandle,
    while_constant_handles: []?BufferHandle,
    compiled_program_contexts: []CompiledProgramContext,
    compiled_program_handles: []?mlx_call.ProgramHandle,
    argument_capture_states: []ArgumentCaptureState,
    argument_capture_mutex: std.atomic.Mutex = .unlocked,
    program: program_mod.Program,
    stats_mutex: std.atomic.Mutex = .unlocked,
    stats: Stats = .{},
};

/// Device-local callback context passed to MLX compiled-program builders.
pub const CompiledProgramContext = struct {
    executable: *Executable,
    device_index: usize,
};

/// Tracks stable captured arguments and dynamic argument indices for MLX compile reuse.
pub const ArgumentCaptureState = struct {
    previous_arguments: []?BufferHandle = &.{},
    dynamic_indices: []u64 = &.{},
    program_handle: ?mlx_call.ProgramHandle = null,
};

/// Returns the resident constant slot for a plan instruction on one device.
pub fn constantIndex(instruction_count: usize, device_index: usize, instruction_index: usize) usize {
    return device_index * instruction_count + instruction_index;
}

/// Returns the resident while-pattern constant slot for one control-flow node.
pub fn whileConstantIndex(control_flow_count: usize, device_index: usize, control_flow_index: usize, constant_index: usize) usize {
    return ((device_index * control_flow_count) + control_flow_index) * 2 + constant_index;
}

test "resident constant indices are device-major" {
    try std.testing.expectEqual(@as(usize, 7), constantIndex(4, 1, 3));
    try std.testing.expectEqual(@as(usize, 10), whileConstantIndex(3, 1, 2, 0));
    try std.testing.expectEqual(@as(usize, 11), whileConstantIndex(3, 1, 2, 1));
}
