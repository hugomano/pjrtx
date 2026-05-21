const std = @import("std");

const ir = @import("src/compiler/ir");
const buffer_mod = @import("buffer.zig");
const lowering_mod = @import("lowering.zig");
const mlx_call = @import("mlx_call.zig");
const profiling_mod = @import("profiling.zig");
const program_mod = @import("program.zig");
const program_build_mod = @import("program_build.zig");

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
    argument_capture_mutex: std.Io.Mutex = .init,
    program: program_mod.Program,
    stats_mutex: std.Io.Mutex = .init,
    stats: Stats = .{},

    /// Decodes an opaque backend executable handle into its owning executable.
    pub fn fromHandle(handle: *anyopaque) *Executable {
        return @ptrCast(@alignCast(handle));
    }

    /// Returns a synchronized copy of the runtime-visible executable counters.
    pub fn snapshotStats(self: *Executable) Stats {
        self.lockStats();
        defer self.unlockStats();
        return self.stats;
    }

    /// Records a successful device execution and the concrete hardware target.
    pub fn recordExecute(self: *Executable, device_index: usize, local_hardware_id: i32) void {
        self.lockStats();
        defer self.unlockStats();
        self.stats.execute_count += 1;
        self.stats.last_execute_device_index = device_index;
        self.stats.last_execute_local_hardware_id = local_hardware_id;
    }

    /// Records one fused schedule group execution.
    pub fn recordFusionGroupExecute(self: *Executable) void {
        self.lockStats();
        defer self.unlockStats();
        self.stats.fusion_group_execute_count += 1;
    }

    /// Records one MLX compiled-program execution and its output count.
    pub fn recordCompiledProgramExecute(self: *Executable, output_count: usize) void {
        self.lockStats();
        defer self.unlockStats();
        self.stats.compiled_program_execute_count += 1;
        self.stats.compiled_program_output_count += output_count;
    }

    /// Records one argument-captured MLX program execution.
    pub fn recordCapturedProgramExecute(self: *Executable, dynamic_count: usize, captured_count: usize) void {
        self.lockStats();
        defer self.unlockStats();
        self.stats.captured_program_execute_count += 1;
        self.stats.captured_program_dynamic_input_count += dynamic_count;
        self.stats.captured_program_captured_input_count += captured_count;
    }

    /// Records an explicit materialization boundary evaluation.
    pub fn recordMaterializationEval(self: *Executable, buffer_count: usize) void {
        self.lockStats();
        defer self.unlockStats();
        self.stats.materialization_eval_count += 1;
        self.stats.materialization_eval_buffer_count += buffer_count;
    }

    /// Records planned intermediate value releases after their final use.
    pub fn recordReleasedIntermediateValues(self: *Executable, count: usize) void {
        self.lockStats();
        defer self.unlockStats();
        self.stats.released_intermediate_count += count;
    }

    /// Records a schedule node that borrowed a resident constant buffer.
    pub fn recordBorrowedConstantNode(self: *Executable) void {
        self.lockStats();
        defer self.unlockStats();
        self.stats.borrowed_constant_nodes += 1;
    }

    /// Accumulates one backend execution profile into executable counters.
    pub fn recordExecuteProfile(self: *Executable, profile: profiling_mod.Execute) void {
        self.lockStats();
        defer self.unlockStats();
        profiling_mod.recordExecute(&self.stats, profile);
    }

    /// Locks the executable argument-capture state for mutation by execution.
    pub fn lockArgumentCapture(self: *Executable) void {
        self.argument_capture_mutex.lockUncancelable(profiling_mod.backendIo());
    }

    /// Unlocks the executable argument-capture state after mutation.
    pub fn unlockArgumentCapture(self: *Executable) void {
        self.argument_capture_mutex.unlock(profiling_mod.backendIo());
    }

    /// Releases all resident backend storage owned by this executable.
    pub fn deinit(self: *Executable) void {
        const allocator = self.allocator;
        destroyCompiledPrograms(self.compiled_program_handles);
        destroyArgumentCaptureStates(allocator, self.argument_capture_states);
        self.program.deinit();
        destroyConstantHandles(self.constant_handles);
        destroyConstantHandles(self.while_constant_handles);
        allocator.free(self.compiled_program_handles);
        allocator.free(self.compiled_program_contexts);
        allocator.free(self.argument_capture_states);
        allocator.free(self.constant_handles);
        allocator.free(self.while_constant_handles);
        allocator.free(self.device_local_hardware_ids);
        allocator.destroy(self);
    }

    fn lockStats(self: *Executable) void {
        self.stats_mutex.lockUncancelable(profiling_mod.backendIo());
    }

    fn unlockStats(self: *Executable) void {
        self.stats_mutex.unlock(profiling_mod.backendIo());
    }
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

    /// Returns whether the current argument set matches the captured baseline.
    pub fn matches(self: ArgumentCaptureState, arguments: []const BufferHandle) bool {
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
    pub fn rememberBaseline(self: *ArgumentCaptureState, allocator: std.mem.Allocator, arguments: []const BufferHandle) !void {
        if (self.previous_arguments.len != arguments.len) {
            allocator.free(self.previous_arguments);
            self.previous_arguments = try allocator.alloc(?BufferHandle, arguments.len);
        }
        for (arguments, 0..) |argument, index| {
            self.previous_arguments[index] = argument;
        }
    }

    /// Releases any captured MLX program and argument baseline storage.
    pub fn reset(self: *ArgumentCaptureState, allocator: std.mem.Allocator) void {
        if (self.program_handle) |program| mlx_call.programDestroy(program);
        self.program_handle = null;
        allocator.free(self.previous_arguments);
        self.previous_arguments = &.{};
        allocator.free(self.dynamic_indices);
        self.dynamic_indices = &.{};
    }

    /// Releases storage held by this capture state during executable teardown.
    pub fn deinit(self: *ArgumentCaptureState, allocator: std.mem.Allocator) void {
        self.reset(allocator);
    }
};

/// Compiles an executable plan into a resident MLX/Metal executable.
pub fn compile(
    allocator: std.mem.Allocator,
    plan: *const ir.ExecutablePlan,
    device_local_hardware_ids: []const i32,
    comptime build_callback: mlx_call.ProgramBuildCallback,
) program_mod.Error!?BufferHandle {
    if (lowering_mod.executableIssue(plan, device_local_hardware_ids)) |_| return null;

    const executable = try allocator.create(Executable);
    errdefer allocator.destroy(executable);
    const ids = try allocator.dupe(i32, device_local_hardware_ids);
    errdefer allocator.free(ids);
    const constant_handles = try allocator.alloc(?BufferHandle, plan.instructions.len * device_local_hardware_ids.len);
    errdefer allocator.free(constant_handles);
    @memset(constant_handles, null);
    errdefer destroyConstantHandles(constant_handles);
    var program = buildProgram(allocator, plan, null) catch |err| switch (err) {
        error.WriteFailed => unreachable,
        error.InvalidDeviceCount => return error.InvalidDeviceCount,
        error.InvalidProgram => return error.InvalidProgram,
        error.UnsupportedElementType => return error.UnsupportedElementType,
        error.ShapeMismatch => return error.ShapeMismatch,
        error.BufferAllocationFailed => return error.BufferAllocationFailed,
        error.CommandSubmissionFailed => return error.CommandSubmissionFailed,
        error.BufferCopyFailed => return error.BufferCopyFailed,
        error.OutOfMemory => return error.OutOfMemory,
        error.InvalidCustomCall => return error.InvalidCustomCall,
    };
    errdefer program.deinit();
    const while_constant_handles = try allocator.alloc(?BufferHandle, program.control_flows.len * device_local_hardware_ids.len * 2);
    errdefer allocator.free(while_constant_handles);
    @memset(while_constant_handles, null);
    errdefer destroyConstantHandles(while_constant_handles);
    const compiled_program_contexts = try allocator.alloc(CompiledProgramContext, device_local_hardware_ids.len);
    errdefer allocator.free(compiled_program_contexts);
    const compiled_program_handles = try allocator.alloc(?mlx_call.ProgramHandle, device_local_hardware_ids.len);
    errdefer allocator.free(compiled_program_handles);
    @memset(compiled_program_handles, null);
    errdefer destroyCompiledPrograms(compiled_program_handles);
    const argument_capture_states = try allocator.alloc(ArgumentCaptureState, device_local_hardware_ids.len);
    errdefer allocator.free(argument_capture_states);
    for (argument_capture_states) |*state| state.* = .{};
    errdefer destroyArgumentCaptureStates(allocator, argument_capture_states);
    const liveness_stats = try program.livenessStats();

    var resident_constant_count: usize = 0;
    var resident_constant_bytes: usize = 0;
    try loadInstructionConstants(
        plan,
        device_local_hardware_ids,
        constant_handles,
        &resident_constant_count,
        &resident_constant_bytes,
    );
    try loadWhilePatternConstants(
        program,
        device_local_hardware_ids,
        while_constant_handles,
        &resident_constant_count,
        &resident_constant_bytes,
    );

    executable.* = .{
        .allocator = allocator,
        .plan = plan,
        .device_local_hardware_ids = ids,
        .constant_handles = constant_handles,
        .while_constant_handles = while_constant_handles,
        .compiled_program_contexts = compiled_program_contexts,
        .compiled_program_handles = compiled_program_handles,
        .argument_capture_states = argument_capture_states,
        .program = program,
        .stats = .{
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
            .program_device_count = device_local_hardware_ids.len,
        },
    };
    for (compiled_program_contexts, 0..) |*context, device_index| {
        context.* = .{
            .executable = executable,
            .device_index = device_index,
        };
    }
    if (programCompileEnabled() and planSupportsCompiledProgram(plan)) {
        for (compiled_program_handles, 0..) |*handle_slot, device_index| {
            handle_slot.* = mlx_call.programCreate(
                &compiled_program_contexts[device_index],
                plan.parameter_shardings.len,
                plan.output_ids.len,
                build_callback,
            ) orelse return error.CommandSubmissionFailed;
        }
    }
    return @ptrCast(executable);
}

/// Returns the resident constant slot for a plan instruction on one device.
pub fn constantIndex(instruction_count: usize, device_index: usize, instruction_index: usize) usize {
    return device_index * instruction_count + instruction_index;
}

/// Returns the resident while-pattern constant slot for one control-flow node.
pub fn whileConstantIndex(control_flow_count: usize, device_index: usize, control_flow_index: usize, constant_index: usize) usize {
    return ((device_index * control_flow_count) + control_flow_index) * 2 + constant_index;
}

fn destroyConstantHandles(constant_handles: []?BufferHandle) void {
    for (constant_handles) |maybe_handle| {
        if (maybe_handle) |handle| buffer_mod.Opaque.destroy(handle);
    }
}

fn destroyCompiledPrograms(handles: []?mlx_call.ProgramHandle) void {
    for (handles) |maybe_handle| {
        if (maybe_handle) |handle| mlx_call.programDestroy(handle);
    }
}

fn destroyArgumentCaptureStates(allocator: std.mem.Allocator, states: []ArgumentCaptureState) void {
    for (states) |*state| {
        state.deinit(allocator);
    }
}

fn buildProgram(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, diagnostic_writer: ?*std.Io.Writer) !program_mod.Program {
    return program_build_mod.build(allocator, plan, diagnostic_writer);
}

fn loadInstructionConstants(
    plan: *const ir.ExecutablePlan,
    device_local_hardware_ids: []const i32,
    constant_handles: []?BufferHandle,
    resident_constant_count: *usize,
    resident_constant_bytes: *usize,
) program_mod.Error!void {
    for (device_local_hardware_ids, 0..) |local_hardware_id, device_index| {
        for (plan.instructions, 0..) |instruction, instruction_index| {
            if (instruction.kind != .constant) continue;
            const output_id = instruction.outputs[0];
            const descriptor = plan.values[output_id.index].descriptor;
            const literal = instruction.literal.?;
            constant_handles[constantIndex(plan.instructions.len, device_index, instruction_index)] = try materializeConstant(
                local_hardware_id,
                descriptor.element_type,
                descriptor.dims,
                literal,
            );
            resident_constant_count.* += 1;
            resident_constant_bytes.* += literal.len;
        }
    }
}

fn loadWhilePatternConstants(
    program: program_mod.Program,
    device_local_hardware_ids: []const i32,
    while_constant_handles: []?BufferHandle,
    resident_constant_count: *usize,
    resident_constant_bytes: *usize,
) program_mod.Error!void {
    for (program.control_flows, 0..) |control_flow, control_flow_index| {
        if (control_flow.condition_subprogram >= program.subprograms.len or control_flow.body_subprogram >= program.subprograms.len) return error.InvalidProgram;
        const pattern = lowering_mod.matchWhileF32LtAddPattern(program.subprograms[control_flow.condition_subprogram], program.subprograms[control_flow.body_subprogram]) orelse continue;
        for (device_local_hardware_ids, 0..) |local_hardware_id, device_index| {
            if (pattern.limit.role == .constant) {
                const limit_literal = pattern.limit.literal orelse return error.InvalidProgram;
                while_constant_handles[whileConstantIndex(program.control_flows.len, device_index, control_flow_index, 0)] = try materializeConstant(
                    local_hardware_id,
                    pattern.limit.descriptor.element_type,
                    pattern.limit.descriptor.dims,
                    limit_literal,
                );
                resident_constant_count.* += 1;
                resident_constant_bytes.* += limit_literal.len;
            }
            if (pattern.step.value.role == .constant) {
                const step_literal = pattern.step.value.literal orelse return error.InvalidProgram;
                while_constant_handles[whileConstantIndex(program.control_flows.len, device_index, control_flow_index, 1)] = try materializeConstant(
                    local_hardware_id,
                    pattern.step.value.descriptor.element_type,
                    pattern.step.value.descriptor.dims,
                    step_literal,
                );
                resident_constant_count.* += 1;
                resident_constant_bytes.* += step_literal.len;
            }
        }
    }
}

fn materializeConstant(local_hardware_id: i32, element_type: ir.BufferType, dims: []const i64, literal: []const u8) program_mod.Error!BufferHandle {
    const buffer = (try buffer_mod.Buffer.fromHost(local_hardware_id, element_type, dims, literal)) orelse return error.CommandSubmissionFailed;
    return buffer.toHandle();
}

fn programCompileEnabled() bool {
    return profiling_mod.programCompileEnabled();
}

fn planSupportsCompiledProgram(plan: *const ir.ExecutablePlan) bool {
    for (plan.instructions) |instruction| {
        switch (instruction.kind) {
            // The RNG path is device-native, but it uses a custom Metal kernel under MLX
            // and is not currently valid inside an mlx::compile graph builder.
            .rng_bit_generator => return false,
            else => {},
        }
    }
    return true;
}

test "resident constant indices are device-major" {
    try std.testing.expectEqual(@as(usize, 7), constantIndex(4, 1, 3));
    try std.testing.expectEqual(@as(usize, 10), whileConstantIndex(3, 1, 2, 0));
    try std.testing.expectEqual(@as(usize, 11), whileConstantIndex(3, 1, 2, 1));
}
