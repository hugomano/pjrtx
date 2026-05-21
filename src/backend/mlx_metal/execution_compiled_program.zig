const std = @import("std");
const ir = @import("src/compiler/ir");

const buffer_mod = @import("buffer.zig");
const executable_mod = @import("executable.zig");
const mlx_call = @import("mlx_call.zig");
const profiling_mod = @import("profiling.zig");
const schedule_mod = @import("execution_schedule.zig");
const types = @import("execution_types.zig");
const values_mod = @import("execution_values.zig");

const CompiledExecutable = executable_mod.Executable;
const CompiledProgramContext = executable_mod.CompiledProgramContext;
const ArgumentCaptureState = executable_mod.ArgumentCaptureState;

const InitialCaptureSmallControlBytes: usize = 4096;
const MinCapturedProgramStableInputs = 8;

/// Executes an MLX compiled program and wraps its device-resident outputs.
pub fn executeCompiledProgram(
    allocator: std.mem.Allocator,
    executable: *CompiledExecutable,
    compiled_program: mlx_call.ProgramHandle,
    arguments: []const types.BufferHandle,
    donated_input_indices: []const u64,
    profile: *profiling_mod.Execute,
) types.Error![]types.ExecutableOutput {
    const profile_enabled = profiling_mod.enabled();
    const execute_start_ns = profiling_mod.start(profile_enabled);
    const program_outputs = mlx_call.programExecuteWithDonation(compiled_program, arguments, donated_input_indices) orelse return error.CommandSubmissionFailed;
    defer program_outputs.deinit();
    if (program_outputs.len() != executable.plan.output_ids.len) {
        return error.CommandSubmissionFailed;
    }

    const outputs = try allocator.alloc(types.ExecutableOutput, executable.plan.output_ids.len);
    errdefer allocator.free(outputs);
    var initialized: usize = 0;
    errdefer {
        for (outputs[0..initialized]) |output| buffer_mod.Opaque.destroy(output.handle);
    }

    for (outputs, 0..) |*output, output_index| {
        const handle = program_outputs.take(output_index) orelse return error.CommandSubmissionFailed;
        const output_id = executable.plan.output_ids[output_index];
        if (output_id.index >= executable.plan.values.len) return error.CommandSubmissionFailed;
        const descriptor = executable.plan.values[output_id.index].descriptor;
        output.* = .{
            .handle = @ptrCast(handle),
            .element_type = descriptor.element_type,
            .dims = descriptor.dims,
            .byte_size = ir.denseByteSize(descriptor.element_type, descriptor.dims),
        };
        initialized += 1;
    }
    if (profile_enabled) {
        const elapsed = profiling_mod.elapsedUs(execute_start_ns);
        profile.schedule_us +|= elapsed;
        profile.schedule_peak_us = @max(profile.schedule_peak_us, elapsed);
        profile.compiled_program_us +|= elapsed;
        profile.compiled_program_peak_us = @max(profile.compiled_program_peak_us, elapsed);
    }
    executable.recordCompiledProgramExecute(outputs.len);
    return outputs;
}

/// Returns donated argument positions for the full compiled program or dynamic capture inputs.
pub fn donatedProgramInputIndices(
    allocator: std.mem.Allocator,
    plan: *const ir.ExecutablePlan,
    dynamic_indices: ?[]const u64,
) types.Error![]const u64 {
    var donated: std.ArrayList(u64) = .empty;
    errdefer donated.deinit(allocator);
    for (plan.donated_parameter_indices) |parameter_index| {
        if (parameterFeedsIdentityOutput(plan, parameter_index)) continue;
        if (dynamic_indices) |indices| {
            for (indices, 0..) |full_index, dynamic_index| {
                if (full_index == parameter_index) {
                    try donated.append(allocator, @intCast(dynamic_index));
                    break;
                }
            }
        } else {
            try donated.append(allocator, parameter_index);
        }
    }
    return try donated.toOwnedSlice(allocator);
}

fn parameterFeedsIdentityOutput(plan: *const ir.ExecutablePlan, parameter_index: u32) bool {
    for (plan.output_aliases) |alias| {
        if (alias.parameter_index == parameter_index and alias.kind == .identity) return true;
    }
    var seen_parameter_index: u32 = 0;
    for (plan.values) |value| {
        if (value.role != .parameter) continue;
        if (seen_parameter_index == parameter_index) {
            for (plan.output_ids) |output_id| {
                if (output_id.index == value.id.index) return true;
            }
            return false;
        }
        seen_parameter_index += 1;
    }
    return false;
}

/// Creates an argument-captured MLX program when enough stable inputs exist.
pub fn maybeCreateInitialArgumentCapturedProgram(
    executable: *CompiledExecutable,
    device_index: usize,
    arguments: []const types.BufferHandle,
) types.Error!void {
    if (device_index >= executable.argument_capture_states.len) return;

    executable.lockArgumentCapture();
    defer executable.unlockArgumentCapture();

    const state = &executable.argument_capture_states[device_index];
    if (state.program_handle != null or state.previous_arguments.len != 0) return;

    const dynamic_indices = try initialArgumentCaptureDynamicIndices(executable.allocator, executable.plan);
    errdefer executable.allocator.free(dynamic_indices);
    if (dynamic_indices.len >= arguments.len or arguments.len - dynamic_indices.len < MinCapturedProgramStableInputs) {
        executable.allocator.free(dynamic_indices);
        return;
    }

    const program = mlx_call.programCreateWithCaptures(
        &executable.compiled_program_contexts[device_index],
        arguments.len,
        executable.plan.output_ids.len,
        compiledProgramBuildCallback,
        arguments,
        dynamic_indices,
    ) orelse {
        executable.allocator.free(dynamic_indices);
        return;
    };

    state.dynamic_indices = dynamic_indices;
    state.program_handle = program;
    state.rememberBaseline(executable.allocator, arguments) catch |err| {
        state.reset(executable.allocator);
        return err;
    };
}

fn initialArgumentCaptureDynamicIndices(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan) ![]u64 {
    var dynamic: std.ArrayList(u64) = .empty;
    errdefer dynamic.deinit(allocator);

    var parameter_index: u32 = 0;
    for (plan.values) |value| {
        if (value.role != .parameter) continue;
        if (initiallyDynamicParameter(plan, parameter_index, value.descriptor)) {
            try dynamic.append(allocator, parameter_index);
        }
        parameter_index += 1;
    }
    return try dynamic.toOwnedSlice(allocator);
}

fn initiallyDynamicParameter(plan: *const ir.ExecutablePlan, parameter_index: u32, descriptor: ir.BufferDescriptor) bool {
    for (plan.donated_parameter_indices) |donated_index| {
        if (donated_index == parameter_index) return true;
    }

    const byte_size = ir.denseByteSize(descriptor.element_type, descriptor.dims);
    if (byte_size == 0 or byte_size > InitialCaptureSmallControlBytes) return false;
    return switch (descriptor.element_type) {
        .pred, .s8, .s16, .s32, .s64, .u8, .u16, .u32, .u64 => true,
        else => false,
    };
}

/// Executes a previously captured MLX program when its stable arguments still match.
pub fn executeArgumentCapturedProgram(
    allocator: std.mem.Allocator,
    executable: *CompiledExecutable,
    device_index: usize,
    arguments: []const types.BufferHandle,
    profile: *profiling_mod.Execute,
) types.Error!?[]types.ExecutableOutput {
    if (device_index >= executable.argument_capture_states.len) return null;
    executable.lockArgumentCapture();
    defer executable.unlockArgumentCapture();

    const state = &executable.argument_capture_states[device_index];
    const program = state.program_handle orelse return null;
    if (!state.matches(arguments)) {
        state.reset(executable.allocator);
        return null;
    }

    const dynamic_arguments = try allocator.alloc(types.BufferHandle, state.dynamic_indices.len);
    defer allocator.free(dynamic_arguments);
    for (state.dynamic_indices, 0..) |dynamic_index, out_index| {
        if (dynamic_index >= arguments.len) return error.CommandSubmissionFailed;
        dynamic_arguments[out_index] = arguments[@intCast(dynamic_index)];
    }
    const donated_input_indices = try donatedProgramInputIndices(allocator, executable.plan, state.dynamic_indices);
    defer allocator.free(donated_input_indices);
    const outputs = try executeCompiledProgram(allocator, executable, program, dynamic_arguments, donated_input_indices, profile);
    executable.recordCapturedProgramExecute(dynamic_arguments.len, arguments.len - dynamic_arguments.len);
    return outputs;
}

/// Updates capture tracking after a normal compiled-program execution.
pub fn updateArgumentCaptureState(
    executable: *CompiledExecutable,
    device_index: usize,
    arguments: []const types.BufferHandle,
) types.Error!void {
    if (device_index >= executable.argument_capture_states.len) return;
    executable.lockArgumentCapture();
    defer executable.unlockArgumentCapture();

    const state = &executable.argument_capture_states[device_index];
    if (state.program_handle != null) return;
    if (state.previous_arguments.len == 0) {
        try state.rememberBaseline(executable.allocator, arguments);
        return;
    }
    if (state.previous_arguments.len != arguments.len) {
        state.reset(executable.allocator);
        try state.rememberBaseline(executable.allocator, arguments);
        return;
    }

    var dynamic_count: usize = 0;
    for (arguments, 0..) |argument, index| {
        if (state.previous_arguments[index] != argument) dynamic_count += 1;
    }
    const captured_count = arguments.len - dynamic_count;
    if (captured_count < MinCapturedProgramStableInputs or dynamic_count == arguments.len) {
        try state.rememberBaseline(executable.allocator, arguments);
        return;
    }

    const dynamic_indices = try executable.allocator.alloc(u64, dynamic_count);
    errdefer executable.allocator.free(dynamic_indices);
    var out_index: usize = 0;
    for (arguments, 0..) |argument, index| {
        if (state.previous_arguments[index] != argument) {
            dynamic_indices[out_index] = @intCast(index);
            out_index += 1;
        }
    }

    const program = mlx_call.programCreateWithCaptures(
        &executable.compiled_program_contexts[device_index],
        arguments.len,
        executable.plan.output_ids.len,
        compiledProgramBuildCallback,
        arguments,
        dynamic_indices,
    ) orelse {
        executable.allocator.free(dynamic_indices);
        try state.rememberBaseline(executable.allocator, arguments);
        return;
    };

    executable.allocator.free(state.dynamic_indices);
    state.dynamic_indices = dynamic_indices;
    state.program_handle = program;
    try state.rememberBaseline(executable.allocator, arguments);
}

/// Builds a traced MLX program body for compiled-program execution.
pub fn compiledProgramBuildCallback(
    user_data: ?*anyopaque,
    call: mlx_call.ProgramBuildCall,
) bool {
    const context: *CompiledProgramContext = @ptrCast(@alignCast(user_data orelse return false));
    const executable = context.executable;
    const allocator = executable.allocator;
    if (call.inputCount() != executable.plan.parameter_shardings.len or call.outputCount() != executable.plan.output_ids.len) return false;

    const arguments = allocator.alloc(types.BufferHandle, call.inputCount()) catch return false;
    defer allocator.free(arguments);
    for (arguments, 0..) |*argument, index| {
        argument.* = call.input(index) orelse return false;
    }

    const outputs = buildExecutableOutputHandlesForCompiledTrace(
        allocator,
        executable,
        context.device_index,
        arguments,
    ) catch return false;
    defer allocator.free(outputs);
    if (outputs.len != call.outputCount()) {
        for (outputs) |output| buffer_mod.Opaque.destroy(output.handle);
        return false;
    }
    for (outputs, 0..) |output, index| {
        if (!call.setOutput(index, output.handle)) return false;
    }
    return true;
}

fn buildExecutableOutputHandlesForCompiledTrace(
    allocator: std.mem.Allocator,
    executable: *CompiledExecutable,
    device_index: usize,
    arguments: []const types.BufferHandle,
) types.Error![]types.ExecutableOutput {
    const plan = executable.plan;
    if (device_index >= executable.device_local_hardware_ids.len) return error.CommandSubmissionFailed;
    if (arguments.len != plan.parameter_shardings.len) return error.CommandSubmissionFailed;

    var bindings = try values_mod.ValueBindings.init(allocator, plan, arguments);
    defer bindings.deinit();

    try (schedule_mod.ScheduleDispatch{
        .allocator = allocator,
        .executable = executable,
        .device_index = device_index,
    }).runForCompiledTrace(&bindings);

    return try (values_mod.OutputBindings{
        .allocator = allocator,
        .executable = executable,
        .values = &bindings,
    }).cloneOrFail();
}
