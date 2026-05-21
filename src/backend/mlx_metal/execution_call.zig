const std = @import("std");
const ir = @import("src/compiler/ir");

const buffer_mod = @import("buffer.zig");
const executable_mod = @import("executable.zig");
const mlx_call = @import("mlx_call.zig");
const profiling_mod = @import("profiling.zig");
const program_mod = @import("program.zig");
const compiled_program_mod = @import("execution_compiled_program.zig");
const schedule_mod = @import("execution_schedule.zig");
const types = @import("execution_types.zig");
const values_mod = @import("execution_values.zig");

const BufferHandle = types.BufferHandle;
const Error = types.Error;
const ExecutableHandle = types.ExecutableHandle;
const ExecutionCompletion = types.ExecutionCompletion;
const ExecutableOutput = types.ExecutableOutput;
const ExecutionResult = types.ExecutionResult;
const CompiledExecutable = executable_mod.Executable;
const ExecuteProfile = profiling_mod.Execute;

const ExecuteCall = struct {
    executable: *CompiledExecutable,
    device_index: usize,
    arguments: []const BufferHandle,

    fn init(executable_handle: ExecutableHandle, device_index: usize, arguments: []const BufferHandle) Error!ExecuteCall {
        const executable = executable_mod.Executable.fromHandle(executable_handle);
        if (device_index >= executable.device_local_hardware_ids.len) return error.CommandSubmissionFailed;
        if (arguments.len != executable.plan.parameter_shardings.len) return error.CommandSubmissionFailed;
        return .{
            .executable = executable,
            .device_index = device_index,
            .arguments = arguments,
        };
    }

    fn compiledProgram(self: ExecuteCall) ?mlx_call.ProgramHandle {
        if (self.device_index >= self.executable.compiled_program_handles.len) return null;
        return self.executable.compiled_program_handles[self.device_index];
    }

    fn recordComplete(self: ExecuteCall, profile: ExecuteProfile, profile_enabled: bool, output_count: usize) void {
        self.executable.recordExecute(self.device_index, self.executable.device_local_hardware_ids[self.device_index]);
        if (profile_enabled) {
            self.executable.recordExecuteProfile(profile);
            schedule_mod.writeExecuteProfile(self.executable, self.device_index, self.arguments.len, output_count, profile);
        }
    }

    fn result(outputs: []ExecutableOutput) ExecutionResult {
        return .{
            .outputs = outputs,
            .completion = ExecutionCompletion.completed(),
        };
    }
};

/// Executes a compiled MLX/Metal executable and returns device-resident outputs.
pub fn executeExecutable(allocator: std.mem.Allocator, executable_handle: ExecutableHandle, device_index: usize, arguments: []const BufferHandle) Error!?ExecutionResult {
    const call = try ExecuteCall.init(executable_handle, device_index, arguments);
    const profile_enabled = profiling_mod.enabled();
    const profile_verbose = profile_enabled and profiling_mod.verbose();
    const execute_start_ns = profiling_mod.start(profile_enabled);
    var profile: ExecuteProfile = .{};

    if (call.compiledProgram()) |compiled_program| {
        try compiled_program_mod.maybeCreateInitialArgumentCapturedProgram(call.executable, call.device_index, call.arguments);
        if (try compiled_program_mod.executeArgumentCapturedProgram(allocator, call.executable, call.device_index, call.arguments, &profile)) |outputs| {
            if (profile_enabled) profile.wall_us = profiling_mod.elapsedUs(execute_start_ns);
            call.recordComplete(profile, profile_enabled, outputs.len);
            return ExecuteCall.result(outputs);
        }
        const donated_input_indices = try compiled_program_mod.donatedProgramInputIndices(allocator, call.executable.plan, null);
        defer allocator.free(donated_input_indices);
        const outputs = try compiled_program_mod.executeCompiledProgram(allocator, call.executable, compiled_program, call.arguments, donated_input_indices, &profile);
        try compiled_program_mod.updateArgumentCaptureState(call.executable, call.device_index, call.arguments);
        if (profile_enabled) profile.wall_us = profiling_mod.elapsedUs(execute_start_ns);
        call.recordComplete(profile, profile_enabled, outputs.len);
        return ExecuteCall.result(outputs);
    }

    var bindings = try values_mod.ValueBindings.init(allocator, call.executable.plan, call.arguments);
    defer bindings.deinit();

    const dispatch = schedule_mod.ScheduleDispatch{
        .allocator = allocator,
        .executable = call.executable,
        .device_index = call.device_index,
        .profile = &profile,
        .profile_enabled = profile_enabled,
        .profile_verbose = profile_verbose,
    };
    if (!try dispatch.run(&bindings)) return null;

    const output_clone_start_ns = profiling_mod.start(profile_enabled);
    const outputs = (try (values_mod.OutputBindings{
        .allocator = allocator,
        .executable = call.executable,
        .values = &bindings,
    }).cloneOrNull()) orelse return null;
    if (profile_enabled) {
        const output_clone_us = profiling_mod.elapsedUs(output_clone_start_ns);
        profile.output_clone_us = output_clone_us;
        profile.output_clone_peak_us = output_clone_us;
        profile.wall_us = profiling_mod.elapsedUs(execute_start_ns);
    }

    call.recordComplete(profile, profile_enabled, outputs.len);
    return ExecuteCall.result(outputs);
}

test "mlx metal backend executable runs resident device buffers" {
    const execution_mod = @import("execution.zig");
    const allocator = std.testing.allocator;
    const local_hardware_id: i32 = 0;
    const dims = [_]i64{4};
    const assignment = [_]i32{0};

    const values = try allocator.alloc(ir.Value, 4);
    errdefer allocator.free(values);
    for (values, 0..) |*value, i| {
        value.* = .{
            .id = .{ .index = @intCast(i) },
            .role = if (i == 0) .parameter else .instruction_result,
            .descriptor = .{
                .element_type = .u8,
                .dims = try allocator.dupe(i64, &dims),
                .device_id = 0,
                .memory_id = 0,
                .shard_index = 0,
            },
        };
    }

    var parameter_shardings = try allocator.alloc(ir.ShardingPlan, 1);
    parameter_shardings[0] = .{
        .kind = .replicated,
        .mesh_name = try allocator.dupe(u8, ""),
        .device_assignment = try allocator.dupe(i32, &assignment),
    };
    var output_shardings = try allocator.alloc(ir.ShardingPlan, 1);
    output_shardings[0] = .{
        .kind = .replicated,
        .mesh_name = try allocator.dupe(u8, ""),
        .device_assignment = try allocator.dupe(i32, &assignment),
    };

    var plan = ir.ExecutablePlan{
        .allocator = allocator,
        .options = .{
            .num_replicas = 1,
            .num_partitions = 1,
            .device_assignment = try allocator.dupe(i32, &assignment),
        },
        .values = values,
        .parameter_shardings = parameter_shardings,
        .output_shardings = output_shardings,
        .output_ids = try allocator.dupe(ir.ValueId, &.{.{ .index = 3 }}),
        .instructions = try allocator.dupe(ir.PlanInstruction, &.{
            .{
                .kind = .constant,
                .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 1 }}),
                .literal = try allocator.dupe(u8, &.{ 2, 3, 4, 5 }),
            },
            .{
                .kind = .add,
                .inputs = try allocator.dupe(ir.ValueId, &.{ .{ .index = 0 }, .{ .index = 1 } }),
                .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 2 }}),
            },
            .{
                .kind = .multiply,
                .inputs = try allocator.dupe(ir.ValueId, &.{ .{ .index = 2 }, .{ .index = 1 } }),
                .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 3 }}),
            },
        }),
    };
    defer plan.deinit();

    const executable = (try executable_mod.compile(allocator, &plan, &.{local_hardware_id}, execution_mod.compiledProgramBuildCallback)) orelse return error.TestUnexpectedResult;
    defer execution_mod.destroy(executable);
    const compiled = executable_mod.Executable.fromHandle(executable);
    try std.testing.expectEqual(@as(usize, 3), compiled.program.nodes.len);
    try std.testing.expectEqual(@as(usize, 3), compiled.program.schedule.len);
    try std.testing.expectEqual(program_mod.ScheduleKind.node, compiled.program.schedule[0].kind);
    try std.testing.expectEqual(@as(usize, 0), compiled.program.schedule[0].index);
    try std.testing.expectEqual(program_mod.ScheduleKind.fusion_group, compiled.program.schedule[1].kind);
    try std.testing.expectEqual(@as(usize, 0), compiled.program.schedule[1].index);
    try std.testing.expectEqual(@as(usize, 2), compiled.program.schedule[1].count);
    try std.testing.expectEqual(program_mod.ScheduleKind.materialization_boundary, compiled.program.schedule[2].kind);
    try std.testing.expectEqual(@as(usize, 0), compiled.program.schedule[2].index);
    try std.testing.expectEqual(@as(usize, 1), compiled.program.schedule[2].count);
    try std.testing.expectEqual(program_mod.NodeKind.constant, compiled.program.nodes[0].kind);
    try std.testing.expectEqual(program_mod.NodeKind.elementwise, compiled.program.nodes[1].kind);
    try std.testing.expectEqual(program_mod.NodeKind.elementwise, compiled.program.nodes[2].kind);
    try std.testing.expect(compiled.program.nodes[0].materializes);
    try std.testing.expectEqual(@as(?usize, null), compiled.program.nodes[0].fusion_group);
    try std.testing.expectEqual(@as(?usize, 0), compiled.program.nodes[1].fusion_group);
    try std.testing.expectEqual(@as(?usize, 0), compiled.program.nodes[2].fusion_group);
    try std.testing.expectEqual(@as(usize, 1), compiled.program.fusion_group_count);
    try std.testing.expectEqual(@as(usize, 1), compiled.program.fusion_groups.len);
    try std.testing.expectEqual(@as(usize, 0), compiled.program.fusion_groups[0].id);
    try std.testing.expectEqual(program_mod.FusionGroupKind.view_elementwise, compiled.program.fusion_groups[0].kind);
    try std.testing.expectEqual(@as(usize, 1), compiled.program.fusion_groups[0].first_node);
    try std.testing.expectEqual(@as(usize, 2), compiled.program.fusion_groups[0].last_node);
    try std.testing.expectEqual(@as(usize, 2), compiled.program.fusion_groups[0].node_count);
    try std.testing.expectEqual(@as(usize, 2), compiled.program.fusion_groups[0].node_indices.len);
    try std.testing.expectEqual(@as(usize, 1), compiled.program.fusion_groups[0].node_indices[0]);
    try std.testing.expectEqual(@as(usize, 2), compiled.program.fusion_groups[0].node_indices[1]);
    try std.testing.expectEqual(@as(usize, 2), compiled.program.fusion_groups[0].input_values.len);
    try std.testing.expectEqual(@as(usize, 0), compiled.program.fusion_groups[0].input_values[0].index);
    try std.testing.expectEqual(@as(usize, 1), compiled.program.fusion_groups[0].input_values[1].index);
    try std.testing.expectEqual(@as(usize, 1), compiled.program.fusion_groups[0].output_values.len);
    try std.testing.expectEqual(@as(usize, 3), compiled.program.fusion_groups[0].output_values[0].index);
    try std.testing.expectEqual(@as(usize, 1), compiled.program.materialization_boundaries.len);
    try std.testing.expectEqual(@as(usize, 3), compiled.program.materialization_boundaries[0].value_id.index);
    try std.testing.expectEqual(program_mod.MaterializationReason.pjrt_output, compiled.program.materialization_boundaries[0].reason);
    try std.testing.expectEqual(@as(usize, 3), compiled.program.edges.len);
    try std.testing.expectEqual(@as(usize, 1), compiled.program.edges[0].value_id.index);
    try std.testing.expectEqual(@as(usize, 0), compiled.program.edges[0].from_node);
    try std.testing.expectEqual(@as(usize, 1), compiled.program.edges[0].to_node);
    try std.testing.expectEqual(@as(usize, 2), compiled.program.edges[1].value_id.index);
    try std.testing.expectEqual(@as(usize, 1), compiled.program.edges[1].from_node);
    try std.testing.expectEqual(@as(usize, 2), compiled.program.edges[1].to_node);
    try std.testing.expectEqual(@as(usize, 1), compiled.program.edges[2].value_id.index);
    try std.testing.expectEqual(@as(usize, 0), compiled.program.edges[2].from_node);
    try std.testing.expectEqual(@as(usize, 2), compiled.program.edges[2].to_node);
    try std.testing.expectEqual(@as(usize, 4), compiled.program.values.len);
    try std.testing.expectEqual(@as(usize, 0), compiled.program.values[0].value_id.index);
    try std.testing.expectEqual(@as(?usize, null), compiled.program.values[0].producer_node);
    try std.testing.expectEqual(@as(?usize, 1), compiled.program.values[0].last_use_node);
    try std.testing.expect(!compiled.program.values[0].is_output);
    try std.testing.expectEqual(@as(?usize, null), compiled.program.values[0].materialization_boundary);
    try std.testing.expectEqual(@as(?usize, 0), compiled.program.values[1].producer_node);
    try std.testing.expectEqual(@as(?usize, 2), compiled.program.values[1].last_use_node);
    try std.testing.expectEqual(@as(?usize, 1), compiled.program.values[2].producer_node);
    try std.testing.expectEqual(@as(?usize, 2), compiled.program.values[2].last_use_node);
    try std.testing.expectEqual(@as(?usize, 2), compiled.program.values[3].producer_node);
    try std.testing.expect(compiled.program.values[3].is_output);
    try std.testing.expectEqual(@as(?usize, 0), compiled.program.values[3].materialization_boundary);

    var executable_stats = execution_mod.stats(executable);
    try std.testing.expectEqual(@as(usize, 1), executable_stats.resident_constant_count);
    try std.testing.expectEqual(@as(usize, 4), executable_stats.resident_constant_bytes);
    try std.testing.expectEqual(@as(usize, 4), executable_stats.program_value_count);
    try std.testing.expectEqual(@as(usize, 3), executable_stats.program_node_count);
    try std.testing.expectEqual(@as(usize, 3), executable_stats.program_edge_count);
    try std.testing.expectEqual(@as(usize, 3), executable_stats.program_schedule_item_count);
    try std.testing.expectEqual(@as(usize, 1), executable_stats.program_fusion_group_count);
    try std.testing.expectEqual(@as(usize, 1), executable_stats.program_materialization_boundary_count);
    try std.testing.expectEqual(@as(usize, 1), executable_stats.program_planned_release_count);
    try std.testing.expectEqual(@as(usize, 4), executable_stats.program_planned_release_bytes);
    try std.testing.expectEqual(@as(usize, 4), executable_stats.program_peak_live_value_count);
    try std.testing.expectEqual(@as(usize, 16), executable_stats.program_peak_live_bytes);
    try std.testing.expectEqual(@as(usize, 1), executable_stats.program_device_count);
    try std.testing.expectEqual(@as(usize, std.math.maxInt(usize)), executable_stats.last_execute_device_index);
    try std.testing.expectEqual(@as(i32, -1), executable_stats.last_execute_local_hardware_id);
    try std.testing.expectEqual(@as(usize, 0), executable_stats.execute_count);
    try std.testing.expectEqual(@as(usize, 0), executable_stats.compiled_program_execute_count);
    try std.testing.expectEqual(@as(usize, 0), executable_stats.compiled_program_output_count);
    try std.testing.expectEqual(@as(usize, 0), executable_stats.fusion_group_execute_count);
    try std.testing.expectEqual(@as(usize, 0), executable_stats.materialization_eval_count);
    try std.testing.expectEqual(@as(usize, 0), executable_stats.materialization_eval_buffer_count);
    try std.testing.expectEqual(@as(usize, 0), executable_stats.released_intermediate_count);
    try std.testing.expectEqual(@as(usize, 0), executable_stats.borrowed_constant_nodes);

    const lhs = (try buffer_mod.Buffer.fromHost(local_hardware_id, .u8, &dims, &.{ 1, 2, 3, 4 })) orelse return error.TestUnexpectedResult;
    defer lhs.destroy();

    try std.testing.expectError(error.CommandSubmissionFailed, execution_mod.execute(allocator, executable, 1, &.{lhs.toHandle()}));
    try std.testing.expectError(error.CommandSubmissionFailed, execution_mod.execute(allocator, executable, 0, &.{}));
    try std.testing.expectError(error.CommandSubmissionFailed, execution_mod.execute(allocator, executable, 0, &.{ lhs.toHandle(), lhs.toHandle() }));
    executable_stats = execution_mod.stats(executable);
    try std.testing.expectEqual(@as(usize, 0), executable_stats.execute_count);
    try std.testing.expectEqual(@as(usize, std.math.maxInt(usize)), executable_stats.last_execute_device_index);
    try std.testing.expectEqual(@as(i32, -1), executable_stats.last_execute_local_hardware_id);

    const result = (try execution_mod.execute(allocator, executable, 0, &.{lhs.toHandle()})) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(execution_mod.ExecutionCompletionKind.completed, result.completion.kind);
    const outputs = result.outputs;
    defer allocator.free(outputs);
    defer {
        for (outputs) |output| buffer_mod.Opaque.destroy(output.handle);
    }

    try std.testing.expectEqual(@as(usize, 1), outputs.len);
    try std.testing.expectEqual(ir.BufferType.u8, outputs[0].element_type);
    try std.testing.expectEqualSlices(i64, &dims, outputs[0].dims);
    try std.testing.expectEqual(@as(u1, 0), @intFromBool(buffer_mod.Opaque.hasHostShadow(outputs[0].handle)));
    var actual: [4]u8 = undefined;
    try buffer_mod.Opaque.copyToHost(outputs[0].handle, &actual);
    try std.testing.expectEqualSlices(u8, &.{ 6, 15, 28, 45 }, &actual);

    executable_stats = execution_mod.stats(executable);
    const compiled_programs_enabled = executable_stats.compiled_program_execute_count != 0;
    try std.testing.expectEqual(@as(usize, 1), executable_stats.resident_constant_count);
    try std.testing.expectEqual(@as(usize, 4), executable_stats.resident_constant_bytes);
    try std.testing.expectEqual(@as(usize, 1), executable_stats.execute_count);
    try std.testing.expectEqual(@as(usize, 0), executable_stats.last_execute_device_index);
    try std.testing.expectEqual(local_hardware_id, executable_stats.last_execute_local_hardware_id);
    if (compiled_programs_enabled) {
        try std.testing.expectEqual(@as(usize, 1), executable_stats.compiled_program_execute_count);
        try std.testing.expectEqual(@as(usize, 1), executable_stats.compiled_program_output_count);
        try std.testing.expect(executable_stats.fusion_group_execute_count <= 1);
        try std.testing.expectEqual(@as(usize, 0), executable_stats.materialization_eval_count);
        try std.testing.expectEqual(@as(usize, 0), executable_stats.materialization_eval_buffer_count);
        try std.testing.expect(executable_stats.released_intermediate_count <= 1);
        try std.testing.expect(executable_stats.borrowed_constant_nodes <= 1);
    } else {
        try std.testing.expectEqual(@as(usize, 0), executable_stats.compiled_program_execute_count);
        try std.testing.expectEqual(@as(usize, 0), executable_stats.compiled_program_output_count);
        try std.testing.expectEqual(@as(usize, 1), executable_stats.fusion_group_execute_count);
        try std.testing.expectEqual(@as(usize, 1), executable_stats.materialization_eval_count);
        try std.testing.expectEqual(@as(usize, 1), executable_stats.materialization_eval_buffer_count);
        try std.testing.expectEqual(@as(usize, 1), executable_stats.released_intermediate_count);
        try std.testing.expectEqual(@as(usize, 1), executable_stats.borrowed_constant_nodes);
    }

    const second_lhs = (try buffer_mod.Buffer.fromHost(local_hardware_id, .u8, &dims, &.{ 2, 4, 6, 8 })) orelse return error.TestUnexpectedResult;
    defer second_lhs.destroy();
    const second_result = (try execution_mod.execute(allocator, executable, 0, &.{second_lhs.toHandle()})) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(execution_mod.ExecutionCompletionKind.completed, second_result.completion.kind);
    const second_outputs = second_result.outputs;
    defer allocator.free(second_outputs);
    defer {
        for (second_outputs) |output| buffer_mod.Opaque.destroy(output.handle);
    }

    try std.testing.expectEqual(@as(usize, 1), second_outputs.len);
    try std.testing.expectEqual(@as(u1, 0), @intFromBool(buffer_mod.Opaque.hasHostShadow(second_outputs[0].handle)));
    try buffer_mod.Opaque.copyToHost(second_outputs[0].handle, &actual);
    try std.testing.expectEqualSlices(u8, &.{ 8, 21, 40, 65 }, &actual);

    executable_stats = execution_mod.stats(executable);
    try std.testing.expectEqual(@as(usize, 1), executable_stats.resident_constant_count);
    try std.testing.expectEqual(@as(usize, 4), executable_stats.resident_constant_bytes);
    try std.testing.expectEqual(@as(usize, 2), executable_stats.execute_count);
    try std.testing.expectEqual(@as(usize, 0), executable_stats.last_execute_device_index);
    try std.testing.expectEqual(local_hardware_id, executable_stats.last_execute_local_hardware_id);
    if (compiled_programs_enabled) {
        try std.testing.expectEqual(@as(usize, 2), executable_stats.compiled_program_execute_count);
        try std.testing.expectEqual(@as(usize, 2), executable_stats.compiled_program_output_count);
        try std.testing.expect(executable_stats.fusion_group_execute_count <= 1);
        try std.testing.expectEqual(@as(usize, 0), executable_stats.materialization_eval_count);
        try std.testing.expectEqual(@as(usize, 0), executable_stats.materialization_eval_buffer_count);
        try std.testing.expect(executable_stats.released_intermediate_count <= 1);
        try std.testing.expect(executable_stats.borrowed_constant_nodes <= 1);
    } else {
        try std.testing.expectEqual(@as(usize, 0), executable_stats.compiled_program_execute_count);
        try std.testing.expectEqual(@as(usize, 0), executable_stats.compiled_program_output_count);
        try std.testing.expectEqual(@as(usize, 2), executable_stats.fusion_group_execute_count);
        try std.testing.expectEqual(@as(usize, 2), executable_stats.materialization_eval_count);
        try std.testing.expectEqual(@as(usize, 2), executable_stats.materialization_eval_buffer_count);
        try std.testing.expectEqual(@as(usize, 2), executable_stats.released_intermediate_count);
        try std.testing.expectEqual(@as(usize, 2), executable_stats.borrowed_constant_nodes);
    }
}
