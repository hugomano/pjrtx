const std = @import("std");
const ir = @import("src/compiler/ir");

const buffer_mod = @import("buffer.zig");
const custom_call_mod = @import("custom_call.zig");
const device_mod = @import("device.zig");
const executable_mod = @import("executable.zig");
const lowering_mod = @import("lowering.zig");
const mlx_call = @import("mlx_call.zig");
const profiling_mod = @import("profiling.zig");
const program_mod = @import("program.zig");

/// Opaque MLX/Metal buffer handle accepted by backend execution.
pub const BufferHandle = *anyopaque;
/// Opaque compiled executable handle owned by the MLX backend.
pub const ExecutableHandle = *anyopaque;
/// Opaque asynchronous execution event handle reserved for future MLX support.
pub const ExecutionEventHandle = *anyopaque;
/// Errors reported by MLX/Metal execution and executable teardown.
pub const Error = program_mod.Error;

/// Device buffer returned for one executable output.
pub const ExecutableOutput = struct {
    handle: BufferHandle,
    element_type: ir.BufferType,
    dims: []const i64,
    byte_size: usize,
};

/// Completion mode for an MLX/Metal execute call.
pub const ExecutionCompletionKind = enum {
    completed,
    pending,
};

/// Completion token returned with execution outputs.
pub const ExecutionCompletion = struct {
    kind: ExecutionCompletionKind = .completed,
    backend_event: ?ExecutionEventHandle = null,

    pub fn completed() ExecutionCompletion {
        return .{ .kind = .completed };
    }

    pub fn pending(event: ExecutionEventHandle) ExecutionCompletion {
        return .{ .kind = .pending, .backend_event = event };
    }
};

/// State reported for an asynchronous backend execution event.
pub const ExecutionEventState = enum {
    pending,
    ready,
    failed,
};

/// Status payload for an asynchronous backend execution event.
pub const ExecutionEventStatus = struct {
    state: ExecutionEventState,
    message: []const u8 = "",
};

/// Result of executing a compiled MLX/Metal executable on one device.
pub const ExecutionResult = struct {
    outputs: []ExecutableOutput,
    completion: ExecutionCompletion = .{},
};

const CompiledExecutable = executable_mod.Executable;
const CompiledProgramContext = executable_mod.CompiledProgramContext;
const ArgumentCaptureState = executable_mod.ArgumentCaptureState;
const ExecuteProfile = profiling_mod.Execute;
const WhileF32LtAddPattern = lowering_mod.WhileF32LtAddPattern;
const WhilePatternOperand = lowering_mod.WhilePatternOperand;

const DefaultWhileMaxIterations: u64 = 1_000_000;
const InitialCaptureSmallControlBytes: usize = 4096;
const MinCapturedProgramStableInputs = 8;

const WhileOperandHandle = struct {
    handle: BufferHandle,
    owned: bool = false,
};

/// Executes a compiled MLX/Metal executable on one device.
pub fn execute(allocator: std.mem.Allocator, executable_handle: ExecutableHandle, device_index: usize, arguments: []const BufferHandle) Error!?ExecutionResult {
    return executeExecutable(allocator, executable_handle, device_index, arguments);
}

/// Reports the status of a backend execution event.
pub fn eventStatus(event: ExecutionEventHandle) Error!ExecutionEventStatus {
    return executionEventStatus(event);
}

/// Releases a backend execution event handle.
pub fn destroyEvent(event: ExecutionEventHandle) void {
    destroyExecutionEvent(event);
}

/// Returns accumulated execution statistics for a compiled executable.
pub fn stats(executable_handle: ExecutableHandle) executable_mod.Stats {
    return executable_mod.Executable.fromHandle(executable_handle).snapshotStats();
}

/// Destroys a compiled executable and all resident backend resources.
pub fn destroy(executable_handle: ExecutableHandle) void {
    executable_mod.Executable.fromHandle(executable_handle).deinit();
}

fn traceScheduleFailure(executable: *const CompiledExecutable, item: program_mod.ScheduleItem, err: Error) void {
    var snapshot = profiling_mod.ScheduleFailureSnapshot{
        .schedule_kind = @tagName(item.kind),
        .schedule_index = item.index,
        .schedule_count = item.count,
        .err = @errorName(err),
    };
    switch (item.kind) {
        .node => {
            if (item.index < executable.program.nodes.len) {
                const node = executable.program.nodes[item.index];
                snapshot.node_kind = @tagName(node.kind);
                snapshot.instruction_index = node.instruction_index;
                if (node.instruction_index < executable.plan.instructions.len) {
                    const instruction = executable.plan.instructions[node.instruction_index];
                    snapshot.op = @tagName(instruction.kind);
                    if (instruction.outputs.len > 0 and instruction.outputs[0].index < executable.plan.values.len) {
                        const descriptor = executable.plan.values[instruction.outputs[0].index].descriptor;
                        snapshot.output_value = instruction.outputs[0].index;
                        snapshot.dtype = @tagName(descriptor.element_type);
                        snapshot.rank = descriptor.dims.len;
                    }
                }
            }
        },
        .fusion_group => {
            if (item.index < executable.program.fusion_groups.len) {
                const group = executable.program.fusion_groups[item.index];
                snapshot.group_first_node = group.first_node;
                snapshot.group_last_node = group.last_node;
                snapshot.group_node_count = group.node_count;
            }
        },
        .materialization_boundary => {},
    }
    profiling_mod.writeScheduleFailure(snapshot);
}

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
            writeExecuteProfile(self.executable, self.device_index, self.arguments.len, output_count, profile);
        }
    }

    fn result(outputs: []ExecutableOutput) ExecutionResult {
        return .{
            .outputs = outputs,
            .completion = ExecutionCompletion.completed(),
        };
    }
};

const ValueBindings = struct {
    allocator: std.mem.Allocator,
    handles: []?BufferHandle,
    owned: []bool,

    fn init(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, arguments: []const BufferHandle) Error!ValueBindings {
        const handles = try allocator.alloc(?BufferHandle, plan.values.len);
        errdefer allocator.free(handles);
        @memset(handles, null);

        const owned = try allocator.alloc(bool, plan.values.len);
        errdefer allocator.free(owned);
        @memset(owned, false);

        var parameter_index: usize = 0;
        for (plan.values) |value| {
            if (value.role != .parameter) continue;
            if (parameter_index >= arguments.len or value.id.index >= handles.len) return error.CommandSubmissionFailed;
            handles[value.id.index] = arguments[parameter_index];
            parameter_index += 1;
        }

        return .{
            .allocator = allocator,
            .handles = handles,
            .owned = owned,
        };
    }

    fn deinit(self: *ValueBindings) void {
        destroyOwnedValueHandles(self.handles, self.owned);
        self.allocator.free(self.owned);
        self.allocator.free(self.handles);
        self.* = undefined;
    }
};

const ScheduleDispatch = struct {
    allocator: std.mem.Allocator,
    executable: *CompiledExecutable,
    device_index: usize,
    profile: ?*ExecuteProfile = null,
    profile_enabled: bool = false,
    profile_verbose: bool = false,

    fn run(self: ScheduleDispatch, bindings: *ValueBindings) Error!bool {
        for (self.executable.program.schedule, 0..) |schedule_item, schedule_index| {
            const schedule_start_ns = profiling_mod.start(self.profile_enabled);
            if (!try self.dispatch(schedule_item, bindings, true)) return false;
            const schedule_us = profiling_mod.elapsedUs(schedule_start_ns);
            self.recordProfile(schedule_index, schedule_item, schedule_us);
        }
        return true;
    }

    fn runForCompiledTrace(self: ScheduleDispatch, bindings: *ValueBindings) Error!void {
        for (self.executable.program.schedule) |schedule_item| {
            if (!try self.dispatch(schedule_item, bindings, false)) return error.CommandSubmissionFailed;
        }
    }

    fn dispatch(self: ScheduleDispatch, schedule_item: program_mod.ScheduleItem, bindings: *ValueBindings, materialize_boundaries: bool) Error!bool {
        switch (schedule_item.kind) {
            .node => {
                return ((ProgramNodeDispatch{
                    .allocator = self.allocator,
                    .executable = self.executable,
                    .device_index = self.device_index,
                    .values = bindings,
                    .release_inputs = true,
                }).run(schedule_item.index) catch |err| {
                    traceScheduleFailure(self.executable, schedule_item, err);
                    return err;
                }) != null;
            },
            .fusion_group => {
                return ((FusionGroupDispatch{
                    .allocator = self.allocator,
                    .executable = self.executable,
                    .device_index = self.device_index,
                    .values = bindings,
                }).run(schedule_item.index, schedule_item.count) catch |err| {
                    traceScheduleFailure(self.executable, schedule_item, err);
                    return err;
                }) != null;
            },
            .materialization_boundary => {
                if (!materialize_boundaries) return true;
                (MaterializationBoundaryEval{
                    .program = &self.executable.program,
                    .values = bindings,
                }).run(schedule_item.index, schedule_item.count) catch |err| {
                    traceScheduleFailure(self.executable, schedule_item, err);
                    return err;
                };
                self.executable.recordMaterializationEval(schedule_item.count);
                return true;
            },
        }
    }

    fn recordProfile(self: ScheduleDispatch, schedule_index: usize, schedule_item: program_mod.ScheduleItem, schedule_us: u64) void {
        if (!self.profile_enabled) return;
        const profile = self.profile orelse return;
        profile.schedule_us +|= schedule_us;
        profile.schedule_peak_us = @max(profile.schedule_peak_us, schedule_us);
        switch (schedule_item.kind) {
            .node => {
                profile.node_us +|= schedule_us;
                profile.node_peak_us = @max(profile.node_peak_us, schedule_us);
            },
            .fusion_group => {
                profile.fusion_group_us +|= schedule_us;
                profile.fusion_group_peak_us = @max(profile.fusion_group_peak_us, schedule_us);
            },
            .materialization_boundary => {
                profile.materialization_eval_us +|= schedule_us;
                profile.materialization_eval_peak_us = @max(profile.materialization_eval_peak_us, schedule_us);
            },
        }
        if (self.profile_verbose) {
            writeScheduleProfile(self.executable, schedule_index, schedule_item, schedule_us);
        }
    }
};

const OutputBindings = struct {
    allocator: std.mem.Allocator,
    executable: *CompiledExecutable,
    values: *ValueBindings,

    fn cloneOrNull(self: OutputBindings) Error!?[]ExecutableOutput {
        return self.clone(.return_null);
    }

    fn cloneOrFail(self: OutputBindings) Error![]ExecutableOutput {
        return (try self.clone(.fail)) orelse error.CommandSubmissionFailed;
    }

    const MissingOutput = enum {
        return_null,
        fail,
    };

    fn clone(self: OutputBindings, missing_output: MissingOutput) Error!?[]ExecutableOutput {
        const plan = self.executable.plan;
        const outputs = try self.allocator.alloc(ExecutableOutput, plan.output_ids.len);
        errdefer self.allocator.free(outputs);
        var initialized: usize = 0;
        errdefer {
            for (outputs[0..initialized]) |output| buffer_mod.Opaque.destroy(output.handle);
        }

        for (plan.output_ids, 0..) |output_id, output_index| {
            if (output_id.index >= self.values.handles.len) return error.CommandSubmissionFailed;
            const value = self.values.handles[output_id.index] orelse return error.CommandSubmissionFailed;
            const handle = if (self.values.owned[output_id.index]) blk: {
                self.values.owned[output_id.index] = false;
                break :blk value;
            } else (try buffer_mod.Opaque.clone(value)) orelse switch (missing_output) {
                .return_null => return null,
                .fail => return error.CommandSubmissionFailed,
            };
            const descriptor = plan.values[output_id.index].descriptor;
            outputs[output_index] = .{
                .handle = handle,
                .element_type = descriptor.element_type,
                .dims = descriptor.dims,
                .byte_size = ir.denseByteSize(descriptor.element_type, descriptor.dims),
            };
            initialized += 1;
        }
        return outputs;
    }
};

fn executeExecutable(allocator: std.mem.Allocator, executable_handle: ExecutableHandle, device_index: usize, arguments: []const BufferHandle) Error!?ExecutionResult {
    const call = try ExecuteCall.init(executable_handle, device_index, arguments);
    const profile_enabled = profiling_mod.enabled();
    const profile_verbose = profile_enabled and profiling_mod.verbose();
    const execute_start_ns = profiling_mod.start(profile_enabled);
    var profile: ExecuteProfile = .{};

    if (call.compiledProgram()) |compiled_program| {
        try maybeCreateInitialArgumentCapturedProgram(call.executable, call.device_index, call.arguments);
        if (try executeArgumentCapturedProgram(allocator, call.executable, call.device_index, call.arguments, &profile)) |outputs| {
            if (profile_enabled) profile.wall_us = profiling_mod.elapsedUs(execute_start_ns);
            call.recordComplete(profile, profile_enabled, outputs.len);
            return ExecuteCall.result(outputs);
        }
        const donated_input_indices = try donatedProgramInputIndices(allocator, call.executable.plan, null);
        defer allocator.free(donated_input_indices);
        const outputs = try executeCompiledProgram(allocator, call.executable, compiled_program, call.arguments, donated_input_indices, &profile);
        try updateArgumentCaptureState(call.executable, call.device_index, call.arguments);
        if (profile_enabled) profile.wall_us = profiling_mod.elapsedUs(execute_start_ns);
        call.recordComplete(profile, profile_enabled, outputs.len);
        return ExecuteCall.result(outputs);
    }

    var bindings = try ValueBindings.init(allocator, call.executable.plan, call.arguments);
    defer bindings.deinit();

    const dispatch = ScheduleDispatch{
        .allocator = allocator,
        .executable = call.executable,
        .device_index = call.device_index,
        .profile = &profile,
        .profile_enabled = profile_enabled,
        .profile_verbose = profile_verbose,
    };
    if (!try dispatch.run(&bindings)) return null;

    const output_clone_start_ns = profiling_mod.start(profile_enabled);
    const outputs = (try (OutputBindings{
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

fn executeCompiledProgram(
    allocator: std.mem.Allocator,
    executable: *CompiledExecutable,
    compiled_program: mlx_call.ProgramHandle,
    arguments: []const BufferHandle,
    donated_input_indices: []const u64,
    profile: *ExecuteProfile,
) Error![]ExecutableOutput {
    const profile_enabled = profiling_mod.enabled();
    const execute_start_ns = profiling_mod.start(profile_enabled);
    const program_outputs = mlx_call.programExecuteWithDonation(compiled_program, arguments, donated_input_indices) orelse return error.CommandSubmissionFailed;
    defer program_outputs.deinit();
    if (program_outputs.len() != executable.plan.output_ids.len) {
        return error.CommandSubmissionFailed;
    }

    const outputs = try allocator.alloc(ExecutableOutput, executable.plan.output_ids.len);
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
fn donatedProgramInputIndices(
    allocator: std.mem.Allocator,
    plan: *const ir.ExecutablePlan,
    dynamic_indices: ?[]const u64,
) Error![]const u64 {
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

fn maybeCreateInitialArgumentCapturedProgram(
    executable: *CompiledExecutable,
    device_index: usize,
    arguments: []const BufferHandle,
) Error!void {
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

fn executeArgumentCapturedProgram(
    allocator: std.mem.Allocator,
    executable: *CompiledExecutable,
    device_index: usize,
    arguments: []const BufferHandle,
    profile: *ExecuteProfile,
) Error!?[]ExecutableOutput {
    if (device_index >= executable.argument_capture_states.len) return null;
    executable.lockArgumentCapture();
    defer executable.unlockArgumentCapture();

    const state = &executable.argument_capture_states[device_index];
    const program = state.program_handle orelse return null;
    if (!state.matches(arguments)) {
        state.reset(executable.allocator);
        return null;
    }

    const dynamic_arguments = try allocator.alloc(BufferHandle, state.dynamic_indices.len);
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

fn updateArgumentCaptureState(
    executable: *CompiledExecutable,
    device_index: usize,
    arguments: []const BufferHandle,
) Error!void {
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

pub fn compiledProgramBuildCallback(
    user_data: ?*anyopaque,
    call: mlx_call.ProgramBuildCall,
) bool {
    const context: *CompiledProgramContext = @ptrCast(@alignCast(user_data orelse return false));
    const executable = context.executable;
    const allocator = executable.allocator;
    if (call.inputCount() != executable.plan.parameter_shardings.len or call.outputCount() != executable.plan.output_ids.len) return false;

    const arguments = allocator.alloc(BufferHandle, call.inputCount()) catch return false;
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
    arguments: []const BufferHandle,
) Error![]ExecutableOutput {
    const plan = executable.plan;
    if (device_index >= executable.device_local_hardware_ids.len) return error.CommandSubmissionFailed;
    if (arguments.len != plan.parameter_shardings.len) return error.CommandSubmissionFailed;

    var bindings = try ValueBindings.init(allocator, plan, arguments);
    defer bindings.deinit();

    try (ScheduleDispatch{
        .allocator = allocator,
        .executable = executable,
        .device_index = device_index,
    }).runForCompiledTrace(&bindings);

    return try (OutputBindings{
        .allocator = allocator,
        .executable = executable,
        .values = &bindings,
    }).cloneOrFail();
}

fn executionEventStatus(_: ExecutionEventHandle) Error!ExecutionEventStatus {
    return .{
        .state = .failed,
        .message = "MLX Metal backend does not expose asynchronous execution event handles",
    };
}

fn destroyExecutionEvent(_: ExecutionEventHandle) void {}

const FusionGroupDispatch = struct {
    allocator: std.mem.Allocator,
    executable: *CompiledExecutable,
    device_index: usize,
    values: *ValueBindings,

    fn run(self: FusionGroupDispatch, group_index: usize, scheduled_node_count: usize) Error!?void {
        if (group_index >= self.executable.program.fusion_groups.len) return error.CommandSubmissionFailed;
        const group = self.executable.program.fusion_groups[group_index];
        if (scheduled_node_count != group.node_indices.len) return error.CommandSubmissionFailed;
        switch (group.kind) {
            .view_elementwise => {},
        }
        self.executable.recordFusionGroupExecute();
        for (group.node_indices) |group_node_index| {
            if (group_node_index >= self.executable.program.nodes.len) return error.CommandSubmissionFailed;
            const node = self.executable.program.nodes[group_node_index];
            if (node.fusion_group != group_index) return error.CommandSubmissionFailed;
            (try (ProgramNodeDispatch{
                .allocator = self.allocator,
                .executable = self.executable,
                .device_index = self.device_index,
                .values = self.values,
                .release_inputs = false,
            }).run(group_node_index)) orelse return null;
        }
        const released = (LivenessRelease{
            .program = &self.executable.program,
            .values = self.values,
        }).fusionGroup(group);
        if (released != 0) self.executable.recordReleasedIntermediateValues(released);
        return {};
    }
};

const ProgramNodeDispatch = struct {
    allocator: std.mem.Allocator,
    executable: *CompiledExecutable,
    device_index: usize,
    values: *ValueBindings,
    release_inputs: bool,

    fn run(self: ProgramNodeDispatch, node_index: usize) Error!?void {
        const allocator = self.allocator;
        const executable = self.executable;
        const device_index = self.device_index;
        const value_handles = self.values.handles;
        const value_owned = self.values.owned;
        const release_inputs = self.release_inputs;
        const plan = executable.plan;
        if (node_index >= executable.program.nodes.len) return error.CommandSubmissionFailed;
        const node = executable.program.nodes[node_index];
        const instruction_index = node.instruction_index;
        const instruction = plan.instructions[instruction_index];
        if (node.kind == .control_flow) {
            (try (ControlFlowDispatch{
                .executable = executable,
                .device_index = device_index,
                .values = self.values,
                .release_inputs = release_inputs,
            }).run(node, instruction)) orelse return null;
            return {};
        }
        if (instruction.kind == .sort and instruction.inputs.len == 2 and instruction.outputs.len == 2) {
            const dimension = instruction.dimension orelse return null;
            const direction = instruction.compare_direction orelse .lt;
            const keys = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
            const values = value_handles[instruction.inputs[1].index] orelse return error.CommandSubmissionFailed;
            const key_output_id = instruction.outputs[0];
            const value_output_id = instruction.outputs[1];
            if (key_output_id.index >= plan.values.len or value_output_id.index >= plan.values.len) return error.CommandSubmissionFailed;
            const key_descriptor = plan.values[key_output_id.index].descriptor;
            const value_descriptor = plan.values[value_output_id.index].descriptor;
            const key_dims = instruction.dims orelse key_descriptor.dims;
            const value_dims = value_descriptor.dims;
            const sorted_keys = (try buffer_mod.Opaque.sort(keys, dimension, key_dims)) orelse return null;
            const directed_keys = (try reverseIfDescending(sorted_keys, dimension, key_dims, direction)) orelse return null;
            errdefer buffer_mod.Opaque.destroy(directed_keys);
            const order = (try buffer_mod.Opaque.argsort(keys, dimension, value_descriptor.element_type, value_dims)) orelse return null;
            const directed_order = (try reverseIfDescending(order, dimension, value_dims, direction)) orelse return null;
            errdefer buffer_mod.Opaque.destroy(directed_order);
            const sorted_values = (try buffer_mod.Opaque.takeAlongAxis(values, directed_order, dimension, value_dims)) orelse return null;

            try storeOwnedValueHandle(value_handles, value_owned, key_output_id, directed_keys);
            errdefer value_owned[key_output_id.index] = false;
            try storeOwnedValueHandle(value_handles, value_owned, value_output_id, sorted_values);
            buffer_mod.Opaque.destroy(directed_order);
            if (release_inputs) {
                self.releaseNodeInputs(node.inputs, instruction_index);
            }
            return {};
        }
        if (instruction.kind == .top_k and instruction.inputs.len == 1 and instruction.outputs.len == 2) {
            const input_id = instruction.inputs[0];
            const input = value_handles[input_id.index] orelse return error.CommandSubmissionFailed;
            const input_descriptor = plan.values[input_id.index].descriptor;
            if (input_descriptor.dims.len == 0) return null;
            const axis: i64 = @intCast(input_descriptor.dims.len - 1);
            const k = instruction.top_k_k orelse return null;
            const values_id = instruction.outputs[0];
            const indices_id = instruction.outputs[1];
            if (values_id.index >= plan.values.len or indices_id.index >= plan.values.len) return error.CommandSubmissionFailed;
            const values_descriptor = plan.values[values_id.index].descriptor;
            const indices_descriptor = plan.values[indices_id.index].descriptor;

            const starts = try allocator.alloc(i64, input_descriptor.dims.len);
            defer allocator.free(starts);
            const limits = try allocator.dupe(i64, input_descriptor.dims);
            defer allocator.free(limits);
            const strides = try allocator.alloc(i64, input_descriptor.dims.len);
            defer allocator.free(strides);
            @memset(starts, 0);
            @memset(strides, 1);
            limits[limits.len - 1] = k;

            const sorted_values = (try buffer_mod.Opaque.sort(input, axis, input_descriptor.dims)) orelse return null;
            const descending_values = (try reverseIfDescending(sorted_values, axis, input_descriptor.dims, .gt)) orelse return null;
            errdefer buffer_mod.Opaque.destroy(descending_values);
            const top_values = (try buffer_mod.Opaque.slice(descending_values, starts, limits, strides, values_descriptor.dims)) orelse return null;
            buffer_mod.Opaque.destroy(descending_values);

            const sorted_indices = (try buffer_mod.Opaque.argsort(input, axis, indices_descriptor.element_type, input_descriptor.dims)) orelse return null;
            const descending_indices = (try reverseIfDescending(sorted_indices, axis, input_descriptor.dims, .gt)) orelse return null;
            errdefer buffer_mod.Opaque.destroy(descending_indices);
            const top_indices = (try buffer_mod.Opaque.slice(descending_indices, starts, limits, strides, indices_descriptor.dims)) orelse return null;
            buffer_mod.Opaque.destroy(descending_indices);

            try storeOwnedValueHandle(value_handles, value_owned, values_id, top_values);
            errdefer value_owned[values_id.index] = false;
            try storeOwnedValueHandle(value_handles, value_owned, indices_id, top_indices);
            if (release_inputs) {
                self.releaseNodeInputs(node.inputs, instruction_index);
            }
            return {};
        }
        if (instruction.kind == .reduce_window_max and instruction.inputs.len == 2 and instruction.outputs.len == 2) {
            const values_id = instruction.inputs[0];
            const indices_id = instruction.inputs[1];
            const values = value_handles[values_id.index] orelse return error.CommandSubmissionFailed;
            const indices = value_handles[indices_id.index] orelse return error.CommandSubmissionFailed;
            const values_output_id = instruction.outputs[0];
            const indices_output_id = instruction.outputs[1];
            if (values_output_id.index >= plan.values.len or indices_output_id.index >= plan.values.len) return error.CommandSubmissionFailed;
            const output_dims = plan.values[values_output_id.index].descriptor.dims;
            const result = (try buffer_mod.Opaque.reduceWindowMaxWithIndices(
                values,
                indices,
                instruction.window_dimensions orelse return null,
                instruction.window_strides orelse return null,
                instruction.base_dilations orelse return null,
                instruction.window_dilations orelse return null,
                instruction.edge_padding_low orelse return null,
                instruction.edge_padding_high orelse return null,
                output_dims,
            )) orelse return null;
            errdefer buffer_mod.Opaque.destroy(result.values);
            errdefer buffer_mod.Opaque.destroy(result.indices);
            try storeOwnedValueHandle(value_handles, value_owned, values_output_id, result.values);
            errdefer value_owned[values_output_id.index] = false;
            try storeOwnedValueHandle(value_handles, value_owned, indices_output_id, result.indices);
            if (release_inputs) {
                self.releaseNodeInputs(node.inputs, instruction_index);
            }
            return {};
        }
        if (instruction.kind == .reduce_max and instruction.inputs.len == 2 and instruction.outputs.len == 2) {
            const values_id = instruction.inputs[0];
            const indices_id = instruction.inputs[1];
            const values = value_handles[values_id.index] orelse return error.CommandSubmissionFailed;
            const indices = value_handles[indices_id.index] orelse return error.CommandSubmissionFailed;
            const values_output_id = instruction.outputs[0];
            const indices_output_id = instruction.outputs[1];
            if (values_output_id.index >= plan.values.len or indices_output_id.index >= plan.values.len) return error.CommandSubmissionFailed;
            const output_dims = plan.values[values_output_id.index].descriptor.dims;
            const result = (try buffer_mod.Opaque.reduceMaxWithIndices(
                values,
                indices,
                instruction.reduce_dimensions orelse return null,
                output_dims,
            )) orelse return null;
            errdefer buffer_mod.Opaque.destroy(result.values);
            errdefer buffer_mod.Opaque.destroy(result.indices);
            try storeOwnedValueHandle(value_handles, value_owned, values_output_id, result.values);
            errdefer value_owned[values_output_id.index] = false;
            try storeOwnedValueHandle(value_handles, value_owned, indices_output_id, result.indices);
            if (release_inputs) {
                self.releaseNodeInputs(node.inputs, instruction_index);
            }
            return {};
        }
        if (instruction.kind == .rng_bit_generator and instruction.inputs.len == 1 and instruction.outputs.len == 2) {
            const state_id = instruction.inputs[0];
            const state = value_handles[state_id.index] orelse return error.CommandSubmissionFailed;
            const state_output_id = instruction.outputs[0];
            const bits_output_id = instruction.outputs[1];
            if (state_output_id.index >= plan.values.len or bits_output_id.index >= plan.values.len) return error.CommandSubmissionFailed;
            const bits_descriptor = plan.values[bits_output_id.index].descriptor;
            const result = (try buffer_mod.Opaque.rngBitGenerator(
                state,
                bits_descriptor.element_type,
                bits_descriptor.dims,
            )) orelse return null;
            errdefer buffer_mod.Opaque.destroy(result.state);
            errdefer buffer_mod.Opaque.destroy(result.bits);
            try storeOwnedValueHandle(value_handles, value_owned, state_output_id, result.state);
            errdefer value_owned[state_output_id.index] = false;
            try storeOwnedValueHandle(value_handles, value_owned, bits_output_id, result.bits);
            if (release_inputs) {
                self.releaseNodeInputs(node.inputs, instruction_index);
            }
            return {};
        }
        if (instruction.kind == .constant) {
            if (instruction.outputs.len != 1) return null;
            const output_id = instruction.outputs[0];
            const cached = executable.constant_handles[executable_mod.constantIndex(plan.instructions.len, device_index, instruction_index)] orelse return null;
            try storeBorrowedValueHandle(value_handles, value_owned, output_id, cached);
            executable.recordBorrowedConstantNode();
            return {};
        }
        if (instruction.kind == .optimization_barrier) {
            if (instruction.inputs.len != instruction.outputs.len) return null;
            var stored_outputs: usize = 0;
            errdefer {
                for (instruction.outputs[0..stored_outputs]) |output_id| {
                    if (output_id.index < value_handles.len and value_owned[output_id.index]) {
                        if (value_handles[output_id.index]) |handle| buffer_mod.Opaque.destroy(handle);
                        value_handles[output_id.index] = null;
                        value_owned[output_id.index] = false;
                    }
                }
            }
            for (instruction.inputs, instruction.outputs) |input_id, output_id| {
                const input = value_handles[input_id.index] orelse return error.CommandSubmissionFailed;
                const cloned = (try buffer_mod.Opaque.clone(input)) orelse return null;
                try storeOwnedValueHandle(value_handles, value_owned, output_id, cloned);
                stored_outputs += 1;
            }
            if (release_inputs) {
                self.releaseNodeInputs(node.inputs, instruction_index);
            }
            return {};
        }
        if (instruction.kind == .tuple) {
            if (instruction.outputs.len != 1) return null;
            const output_id = instruction.outputs[0];
            if (output_id.index >= plan.values.len or plan.values[output_id.index].storage != .tuple) return null;
            if (release_inputs) {
                self.releaseNodeInputs(node.inputs, instruction_index);
            }
            return {};
        }
        if (instruction.outputs.len != 1) return null;
        const output_id = instruction.outputs[0];
        if (output_id.index >= value_handles.len) return error.CommandSubmissionFailed;
        const output_descriptor = plan.values[output_id.index].descriptor;
        const output_dims = instruction.dims orelse output_descriptor.dims;
        const next = switch (instruction.kind) {
            .copy_arg0, .reduce_precision => blk: {
                const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                break :blk (try buffer_mod.Opaque.clone(input)) orelse return null;
            },
            .complex => blk: {
                const real = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                const imag = value_handles[instruction.inputs[1].index] orelse return error.CommandSubmissionFailed;
                break :blk (try buffer_mod.Opaque.complex(real, imag, output_dims)) orelse return null;
            },
            .real => blk: {
                const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                break :blk (try buffer_mod.Opaque.realPart(input, output_dims)) orelse return null;
            },
            .imag => blk: {
                const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                break :blk (try buffer_mod.Opaque.imagPart(input, output_dims)) orelse return null;
            },
            .custom_call => (try (CustomCallDispatch{ .values = self.values }).run(instruction)) orelse return null,
            .get_tuple_element => blk: {
                if (instruction.inputs.len != 1) return null;
                const tuple_id = instruction.inputs[0];
                if (tuple_id.index >= plan.values.len) return error.CommandSubmissionFailed;
                const tuple_value = plan.values[tuple_id.index];
                if (tuple_value.storage != .tuple) return null;
                const tuple_index = instruction.tuple_index orelse return null;
                if (tuple_index < 0 or tuple_index >= @as(i64, @intCast(tuple_value.elements.len))) return null;
                const element_id = tuple_value.elements[@intCast(tuple_index)];
                if (element_id.index >= value_handles.len) return error.CommandSubmissionFailed;
                const element = value_handles[element_id.index] orelse return error.CommandSubmissionFailed;
                break :blk (try buffer_mod.Opaque.clone(element)) orelse return null;
            },
            .iota => blk: {
                break :blk (try buffer_mod.Opaque.iota(
                    executable.device_local_hardware_ids[device_index],
                    output_descriptor.element_type,
                    output_dims,
                    instruction.iota_dimension orelse return null,
                )) orelse return null;
            },
            .partition_id => blk: {
                const partition_count = if (plan.options.num_partitions <= 0) 1 else plan.options.num_partitions;
                const partition_id_value: u32 = @intCast(device_index % @as(usize, @intCast(partition_count)));
                break :blk (try buffer_mod.Opaque.partitionId(
                    executable.device_local_hardware_ids[device_index],
                    output_descriptor.element_type,
                    partition_id_value,
                )) orelse return null;
            },
            .rng => blk: {
                const a = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                const b = value_handles[instruction.inputs[1].index] orelse return error.CommandSubmissionFailed;
                break :blk (try buffer_mod.Opaque.rng(
                    a,
                    b,
                    instruction.rng_distribution orelse return null,
                    output_descriptor.element_type,
                    output_dims,
                )) orelse return null;
            },
            .convert => blk: {
                const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                break :blk (try buffer_mod.Opaque.convert(input, output_descriptor.element_type)) orelse return null;
            },
            .bitcast_convert => blk: {
                const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                break :blk (try buffer_mod.Opaque.bitcast(input, output_descriptor.element_type, output_dims)) orelse return null;
            },
            .add, .subtract, .multiply, .divide, .maximum, .minimum, .power, .atan2, .remainder, .and_, .or_, .xor, .shift_left, .shift_right_arithmetic, .shift_right_logical => blk: {
                const op = lowering_mod.executableBinaryOp(instruction.kind) orelse return null;
                const lhs = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                const rhs = value_handles[instruction.inputs[1].index] orelse return error.CommandSubmissionFailed;
                break :blk (try buffer_mod.Opaque.binaryWithOutputDims(lhs, rhs, op, output_dims)) orelse return null;
            },
            .negate, .exp, .expm1, .tanh, .sqrt, .rsqrt, .abs, .cbrt, .ceil, .floor, .log, .log1p, .logistic, .sine, .cosine, .not_, .sign, .is_finite, .round_nearest_afz, .round_nearest_even, .popcnt, .count_leading_zeros => blk: {
                const op = lowering_mod.executableUnaryOp(instruction.kind) orelse return null;
                const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                break :blk (try buffer_mod.Opaque.unary(input, op)) orelse return null;
            },
            .reshape => blk: {
                const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                break :blk (try buffer_mod.Opaque.reshape(input, output_dims)) orelse return null;
            },
            .transpose => blk: {
                const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                break :blk (try buffer_mod.Opaque.transpose(input, instruction.permutation orelse return null)) orelse return null;
            },
            .broadcast_in_dim => blk: {
                const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                break :blk (try buffer_mod.Opaque.broadcastInDim(input, instruction.broadcast_dimensions orelse return null, output_dims)) orelse return null;
            },
            .slice => blk: {
                const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                break :blk (try buffer_mod.Opaque.slice(
                    input,
                    instruction.start_indices orelse return null,
                    instruction.limit_indices orelse return null,
                    instruction.strides orelse return null,
                    output_dims,
                )) orelse return null;
            },
            .dynamic_slice => blk: {
                const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                const starts = try startHandles(allocator, value_handles, instruction.inputs[1..]);
                defer allocator.free(starts);
                break :blk (try buffer_mod.Opaque.dynamicSlice(input, starts, instruction.slice_sizes orelse return null, output_dims)) orelse return null;
            },
            .dynamic_update_slice => blk: {
                const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                const update = value_handles[instruction.inputs[1].index] orelse return error.CommandSubmissionFailed;
                const starts = try startHandles(allocator, value_handles, instruction.inputs[2..]);
                defer allocator.free(starts);
                break :blk (try buffer_mod.Opaque.dynamicUpdateSlice(input, update, starts, output_dims)) orelse return null;
            },
            .pad => blk: {
                const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                const padding_value = value_handles[instruction.inputs[1].index] orelse return error.CommandSubmissionFailed;
                break :blk (try buffer_mod.Opaque.pad(
                    input,
                    padding_value,
                    instruction.edge_padding_low orelse return null,
                    instruction.edge_padding_high orelse return null,
                    instruction.interior_padding orelse return null,
                    output_dims,
                )) orelse return null;
            },
            .reverse => blk: {
                const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                break :blk (try buffer_mod.Opaque.reverse(input, instruction.dimensions orelse &.{}, output_dims)) orelse return null;
            },
            .concatenate => blk: {
                const lhs = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                const rhs = value_handles[instruction.inputs[1].index] orelse return error.CommandSubmissionFailed;
                break :blk (try buffer_mod.Opaque.concatenate(lhs, rhs, instruction.dimension orelse return null, output_dims)) orelse return null;
            },
            .gather => blk: {
                const operand = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                const indices = value_handles[instruction.inputs[1].index] orelse return error.CommandSubmissionFailed;
                break :blk (try buffer_mod.Opaque.gather(
                    operand,
                    indices,
                    instruction.start_index_map orelse return null,
                    instruction.collapsed_slice_dims orelse &.{},
                    instruction.operand_batching_dims orelse &.{},
                    instruction.start_indices_batching_dims orelse &.{},
                    instruction.index_vector_dim orelse 0,
                    instruction.slice_sizes orelse return null,
                    instruction.offset_dims orelse &.{},
                    output_dims,
                )) orelse return null;
            },
            .scatter => blk: {
                const operand = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                const indices = value_handles[instruction.inputs[1].index] orelse return error.CommandSubmissionFailed;
                const updates = value_handles[instruction.inputs[2].index] orelse return error.CommandSubmissionFailed;
                if (lowering_mod.supportedScatterAxis(instruction)) |scatter_axis| {
                    break :blk (try buffer_mod.Opaque.scatterAxis(
                        operand,
                        indices,
                        updates,
                        scatter_axis,
                        instruction.index_vector_dim orelse 0,
                        instruction.scatter_update_kind orelse .set,
                        output_dims,
                    )) orelse return null;
                }
                break :blk (try buffer_mod.Opaque.scatter(
                    operand,
                    indices,
                    updates,
                    instruction.scatter_dims_to_operand_dims orelse return null,
                    instruction.inserted_window_dims orelse return null,
                    instruction.update_window_dims orelse &.{},
                    instruction.input_batching_dims orelse &.{},
                    instruction.scatter_indices_batching_dims orelse &.{},
                    instruction.index_vector_dim orelse 0,
                    instruction.scatter_update_kind orelse .set,
                    output_dims,
                )) orelse return null;
            },
            .sort => blk: {
                const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                const sorted = (try buffer_mod.Opaque.sort(input, instruction.dimension orelse return null, output_dims)) orelse return null;
                break :blk (try reverseIfDescending(sorted, instruction.dimension.?, output_dims, instruction.compare_direction orelse .lt)) orelse return null;
            },
            .dot_general => blk: {
                const lhs = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                const rhs = value_handles[instruction.inputs[1].index] orelse return error.CommandSubmissionFailed;
                break :blk (try buffer_mod.Opaque.dotGeneral(
                    lhs,
                    rhs,
                    instruction.lhs_batch_dimensions orelse &.{},
                    instruction.rhs_batch_dimensions orelse &.{},
                    instruction.lhs_contracting_dimensions orelse &.{},
                    instruction.rhs_contracting_dimensions orelse &.{},
                    output_dims,
                )) orelse return null;
            },
            .convolution => blk: {
                const lhs = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                const rhs = value_handles[instruction.inputs[1].index] orelse return error.CommandSubmissionFailed;
                break :blk (try buffer_mod.Opaque.convolution(
                    lhs,
                    rhs,
                    instruction.window_strides orelse return null,
                    instruction.edge_padding_low orelse return null,
                    instruction.edge_padding_high orelse return null,
                    instruction.base_dilations orelse return null,
                    instruction.window_dilations orelse return null,
                    instruction.window_reversal orelse return null,
                    instruction.feature_group_count orelse 1,
                    output_dims,
                )) orelse return null;
            },
            .cholesky => blk: {
                const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                break :blk (try buffer_mod.Opaque.cholesky(input, instruction.lower orelse true, output_dims)) orelse return null;
            },
            .triangular_solve => blk: {
                const a = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                const b = value_handles[instruction.inputs[1].index] orelse return error.CommandSubmissionFailed;
                break :blk (try buffer_mod.Opaque.triangularSolve(
                    a,
                    b,
                    instruction.triangular_left_side orelse true,
                    instruction.triangular_lower orelse true,
                    instruction.triangular_unit_diagonal orelse false,
                    instruction.triangular_transpose orelse .no_transpose,
                    output_dims,
                )) orelse return null;
            },
            .fft => blk: {
                const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                break :blk (try buffer_mod.Opaque.fft(input, instruction.fft_kind orelse return null, instruction.dimensions orelse return null, output_dims)) orelse return null;
            },
            .reduce_sum, .reduce_max, .reduce_min, .reduce_and, .reduce_or => blk: {
                const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                break :blk (try buffer_mod.Opaque.reduce(input, instruction.kind, instruction.reduce_dimensions orelse &.{}, output_dims)) orelse return null;
            },
            .reduce_window_sum, .reduce_window_max => blk: {
                const input = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                break :blk (try buffer_mod.Opaque.reduceWindow(
                    input,
                    instruction.kind,
                    instruction.window_dimensions orelse return null,
                    instruction.window_strides orelse return null,
                    instruction.base_dilations orelse return null,
                    instruction.window_dilations orelse return null,
                    instruction.edge_padding_low orelse return null,
                    instruction.edge_padding_high orelse return null,
                    output_dims,
                )) orelse return null;
            },
            .compare => blk: {
                const lhs = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                const rhs = value_handles[instruction.inputs[1].index] orelse return error.CommandSubmissionFailed;
                break :blk (try buffer_mod.Opaque.compare(lhs, rhs, instruction.compare_direction orelse .eq, output_dims)) orelse return null;
            },
            .select => blk: {
                const pred = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                const on_true = value_handles[instruction.inputs[1].index] orelse return error.CommandSubmissionFailed;
                const on_false = value_handles[instruction.inputs[2].index] orelse return error.CommandSubmissionFailed;
                break :blk (try buffer_mod.Opaque.select(pred, on_true, on_false, output_dims)) orelse return null;
            },
            .clamp => blk: {
                const min = value_handles[instruction.inputs[0].index] orelse return error.CommandSubmissionFailed;
                const value = value_handles[instruction.inputs[1].index] orelse return error.CommandSubmissionFailed;
                const max = value_handles[instruction.inputs[2].index] orelse return error.CommandSubmissionFailed;
                break :blk (try buffer_mod.Opaque.clamp(min, value, max, output_dims)) orelse return null;
            },
            else => return null,
        };

        try storeOwnedValueHandle(value_handles, value_owned, output_id, next);
        if (release_inputs) {
            self.releaseNodeInputs(node.inputs, instruction_index);
        }
        return {};
    }

    fn releaseNodeInputs(self: ProgramNodeDispatch, inputs: []const ir.ValueId, instruction_index: usize) void {
        const released = (LivenessRelease{
            .program = &self.executable.program,
            .values = self.values,
        }).afterNode(inputs, instruction_index);
        if (released != 0) self.executable.recordReleasedIntermediateValues(released);
    }
};

const ControlFlowDispatch = struct {
    executable: *CompiledExecutable,
    device_index: usize,
    values: *ValueBindings,
    release_inputs: bool,

    fn run(self: ControlFlowDispatch, node: program_mod.Node, instruction: ir.PlanInstruction) Error!?void {
        const executable = self.executable;
        const value_handles = self.values.handles;
        const control_flow_index = node.control_flow orelse return error.CommandSubmissionFailed;
        if (control_flow_index >= executable.program.control_flows.len) return error.CommandSubmissionFailed;
        const control_flow = executable.program.control_flows[control_flow_index];
        switch (control_flow.kind) {
            .while_loop => {
                if (instruction.kind != .while_) return error.CommandSubmissionFailed;
                if (control_flow.condition_subprogram >= executable.program.subprograms.len or
                    control_flow.body_subprogram >= executable.program.subprograms.len)
                {
                    return error.CommandSubmissionFailed;
                }
                const pattern = lowering_mod.matchWhileF32LtAddPattern(
                    executable.program.subprograms[control_flow.condition_subprogram],
                    executable.program.subprograms[control_flow.body_subprogram],
                ) orelse return null;
                if (instruction.inputs.len != pattern.state_count or instruction.outputs.len != pattern.state_count) return null;
                if (pattern.state_index >= instruction.inputs.len) return error.CommandSubmissionFailed;
                const state_id = instruction.inputs[pattern.state_index];
                if (state_id.index >= value_handles.len) return error.CommandSubmissionFailed;
                const state = value_handles[state_id.index] orelse return error.CommandSubmissionFailed;
                const output_id = instruction.outputs[pattern.state_index];
                if (output_id.index >= executable.plan.values.len) return error.CommandSubmissionFailed;
                const output_descriptor = executable.plan.values[output_id.index].descriptor;
                const limit = try self.whilePatternOperandHandle(instruction, control_flow_index, pattern.limit, 0);
                const step = try self.whileStepOperandHandle(
                    instruction,
                    control_flow_index,
                    executable.program.subprograms[control_flow.body_subprogram],
                    pattern.step,
                );
                defer if (step.owned) buffer_mod.Opaque.destroy(step.handle);
                const next = (try buffer_mod.Opaque.whileF32CompareAdd(
                    state,
                    limit,
                    step.handle,
                    pattern.compare_direction,
                    pattern.update_op,
                    output_descriptor.dims,
                    pattern.max_iterations,
                )) orelse return null;
                try storeOwnedValueHandle(value_handles, self.values.owned, output_id, next);
                var invariant_index: usize = 1;
                invariant_index = 0;
                while (invariant_index < pattern.state_count) : (invariant_index += 1) {
                    if (invariant_index == pattern.state_index) continue;
                    const input_id = instruction.inputs[invariant_index];
                    const invariant_output_id = instruction.outputs[invariant_index];
                    if (input_id.index >= value_handles.len) return error.CommandSubmissionFailed;
                    const input = value_handles[input_id.index] orelse return error.CommandSubmissionFailed;
                    const cloned = (try buffer_mod.Opaque.clone(input)) orelse return null;
                    try storeOwnedValueHandle(value_handles, self.values.owned, invariant_output_id, cloned);
                }
                if (self.release_inputs) {
                    const released = (LivenessRelease{ .program = &executable.program, .values = self.values }).afterNode(node.inputs, node.instruction_index);
                    if (released != 0) executable.recordReleasedIntermediateValues(released);
                }
                return {};
            },
        }
    }

    fn whilePatternOperandHandle(
        self: ControlFlowDispatch,
        instruction: ir.PlanInstruction,
        control_flow_index: usize,
        value: ir.RegionValue,
        constant_slot: usize,
    ) Error!BufferHandle {
        const executable = self.executable;
        const value_handles = self.values.handles;
        return switch (value.role) {
            .constant => executable.while_constant_handles[executable_mod.whileConstantIndex(executable.program.control_flows.len, self.device_index, control_flow_index, constant_slot)] orelse error.CommandSubmissionFailed,
            .argument => blk: {
                const argument_index: usize = value.id.index;
                if (argument_index >= instruction.inputs.len) return error.CommandSubmissionFailed;
                const input_id = instruction.inputs[argument_index];
                if (input_id.index >= value_handles.len) return error.CommandSubmissionFailed;
                break :blk value_handles[input_id.index] orelse error.CommandSubmissionFailed;
            },
            else => error.CommandSubmissionFailed,
        };
    }

    fn whileStepOperandHandle(
        self: ControlFlowDispatch,
        instruction: ir.PlanInstruction,
        control_flow_index: usize,
        body: program_mod.Subprogram,
        operand: WhilePatternOperand,
    ) Error!WhileOperandHandle {
        if (operand.producer_instruction_index == null) {
            return .{
                .handle = try self.whilePatternOperandHandle(instruction, control_flow_index, operand.value, 1),
                .owned = false,
            };
        }
        const producer_index = operand.producer_instruction_index.?;
        if (producer_index >= body.instructions.len) return error.CommandSubmissionFailed;
        const producer = body.instructions[producer_index];
        const handle = (try self.executeLoopInvariantRegionInstruction(
            instruction,
            body,
            producer,
        )) orelse return error.CommandSubmissionFailed;
        return .{ .handle = handle, .owned = true };
    }

    fn executeLoopInvariantRegionInstruction(
        self: ControlFlowDispatch,
        parent_instruction: ir.PlanInstruction,
        subprogram: program_mod.Subprogram,
        instruction: ir.RegionInstruction,
    ) Error!?BufferHandle {
        if (instruction.outputs.len != 1 or instruction.result_descriptors.len != 1) return null;
        const output_dims = instruction.result_descriptors[0].dims;
        switch (instruction.kind) {
            .add, .subtract, .multiply, .divide, .maximum, .minimum => {
                if (instruction.inputs.len != 2) return null;
                const op = lowering_mod.executableBinaryOp(instruction.kind) orelse return null;
                const lhs = try self.loopInvariantRegionOperandHandle(parent_instruction, subprogram, instruction.inputs[0]);
                const rhs = try self.loopInvariantRegionOperandHandle(parent_instruction, subprogram, instruction.inputs[1]);
                return (try buffer_mod.Opaque.binaryWithOutputDims(lhs, rhs, op, output_dims)) orelse return null;
            },
            else => return null,
        }
    }

    fn loopInvariantRegionOperandHandle(
        self: ControlFlowDispatch,
        parent_instruction: ir.PlanInstruction,
        subprogram: program_mod.Subprogram,
        value_id: ir.RegionValueId,
    ) Error!BufferHandle {
        const value_handles = self.values.handles;
        const value = lowering_mod.regionValueById(subprogram, value_id) orelse return error.CommandSubmissionFailed;
        if (value.role != .argument) return error.CommandSubmissionFailed;
        const argument_index: usize = value.id.index;
        if (argument_index >= parent_instruction.inputs.len) return error.CommandSubmissionFailed;
        const input_id = parent_instruction.inputs[argument_index];
        if (input_id.index >= value_handles.len) return error.CommandSubmissionFailed;
        return value_handles[input_id.index] orelse error.CommandSubmissionFailed;
    }
};

fn writeExecuteProfile(executable: *const CompiledExecutable, device_index: usize, argument_count: usize, output_count: usize, profile: ExecuteProfile) void {
    profiling_mod.writeExecute(.{
        .address = @intFromPtr(executable),
        .schedule_count = executable.program.schedule.len,
        .node_count = executable.program.nodes.len,
        .fusion_group_count = executable.program.fusion_groups.len,
        .materialization_boundary_count = executable.program.materialization_boundaries.len,
    }, device_index, argument_count, output_count, profile);
}

fn writeScheduleProfile(executable: *const CompiledExecutable, schedule_index: usize, item: program_mod.ScheduleItem, elapsed_us: u64) void {
    var snapshot = profiling_mod.ScheduleItemSnapshot{
        .executable_address = @intFromPtr(executable),
        .schedule_index = schedule_index,
        .kind = switch (item.kind) {
            .node => .node,
            .fusion_group => .fusion_group,
            .materialization_boundary => .materialization_boundary,
        },
        .index = item.index,
        .count = item.count,
    };
    switch (item.kind) {
        .node => {
            if (item.index < executable.program.nodes.len) {
                const node = executable.program.nodes[item.index];
                snapshot.node_kind = @tagName(node.kind);
                snapshot.instruction_index = node.instruction_index;
                if (node.instruction_index < executable.plan.instructions.len) {
                    const instruction = executable.plan.instructions[node.instruction_index];
                    snapshot.op = @tagName(instruction.kind);
                }
            }
        },
        .fusion_group => {
            var group_ops_buffer: [128][]const u8 = undefined;
            if (item.index < executable.program.fusion_groups.len) {
                const group = executable.program.fusion_groups[item.index];
                snapshot.group_first_node = group.first_node;
                snapshot.group_last_node = group.last_node;
                snapshot.group_node_count = group.node_count;
                var op_count: usize = 0;
                for (group.node_indices) |group_node_index| {
                    if (op_count >= group_ops_buffer.len) break;
                    if (group_node_index >= executable.program.nodes.len) {
                        group_ops_buffer[op_count] = "?";
                    } else {
                        const group_node = executable.program.nodes[group_node_index];
                        group_ops_buffer[op_count] = if (group_node.instruction_index < executable.plan.instructions.len)
                            @tagName(executable.plan.instructions[group_node.instruction_index].kind)
                        else
                            "?";
                    }
                    op_count += 1;
                }
                snapshot.group_ops = group_ops_buffer[0..op_count];
            }
        },
        .materialization_boundary => {},
    }
    profiling_mod.writeScheduleItem(snapshot, elapsed_us);
}

const LivenessRelease = struct {
    program: *const program_mod.Program,
    values: *ValueBindings,

    fn afterNode(self: LivenessRelease, input_ids: []const ir.ValueId, instruction_index: usize) usize {
        var released: usize = 0;
        for (input_ids) |input_id| {
            if (input_id.index >= self.values.handles.len or input_id.index >= self.program.values.len) continue;
            const value = self.program.values[input_id.index];
            if (value.is_output) continue;
            if (value.last_use_node != @as(?usize, instruction_index)) continue;
            if (!self.values.owned[input_id.index]) continue;
            if (self.values.handles[input_id.index]) |old| buffer_mod.Opaque.destroy(old);
            self.values.handles[input_id.index] = null;
            self.values.owned[input_id.index] = false;
            released += 1;
        }
        return released;
    }

    fn fusionGroup(self: LivenessRelease, group: program_mod.FusionGroup) usize {
        var released: usize = 0;
        for (group.node_indices) |node_index| {
            if (node_index >= self.program.nodes.len) continue;
            const node = self.program.nodes[node_index];
            for (node.inputs) |input_id| {
                if (input_id.index >= self.values.handles.len or input_id.index >= self.program.values.len) continue;
                const value = self.program.values[input_id.index];
                if (value.is_output) continue;
                const last_use = value.last_use_node orelse continue;
                if (last_use > group.last_node) continue;
                if (!self.values.owned[input_id.index]) continue;
                if (self.values.handles[input_id.index]) |old| buffer_mod.Opaque.destroy(old);
                self.values.handles[input_id.index] = null;
                self.values.owned[input_id.index] = false;
                released += 1;
            }
        }
        return released;
    }
};

const MaterializationBoundaryEval = struct {
    program: *const program_mod.Program,
    values: *ValueBindings,

    fn run(self: MaterializationBoundaryEval, first_boundary: usize, boundary_count: usize) Error!void {
        const program = self.program;
        const value_handles = self.values.handles;
        if (first_boundary > program.materialization_boundaries.len) return error.CommandSubmissionFailed;
        if (boundary_count > program.materialization_boundaries.len - first_boundary) return error.CommandSubmissionFailed;

        var stack_handles: [8]BufferHandle = undefined;
        if (boundary_count <= stack_handles.len) {
            var count: usize = 0;
            for (program.materialization_boundaries[first_boundary..][0..boundary_count]) |boundary| {
                const value_index = boundary.value_id.index;
                if (value_index >= value_handles.len) {
                    self.traceFailure("out_of_range", value_index, boundary.reason);
                    return error.CommandSubmissionFailed;
                }
                stack_handles[count] = value_handles[value_index] orelse {
                    self.traceFailure("missing_handle", value_index, boundary.reason);
                    return error.CommandSubmissionFailed;
                };
                count += 1;
            }
            buffer_mod.Opaque.evalMany(stack_handles[0..count]) catch {
                for (stack_handles[0..count]) |handle| {
                    buffer_mod.Opaque.eval(handle) catch |err| {
                        self.traceFailure(@errorName(err), std.math.maxInt(usize), .backend_requirement);
                        return err;
                    };
                }
            };
            return;
        }

        for (program.materialization_boundaries[first_boundary..][0..boundary_count]) |boundary| {
            const value_index = boundary.value_id.index;
            if (value_index >= value_handles.len) {
                self.traceFailure("out_of_range", value_index, boundary.reason);
                return error.CommandSubmissionFailed;
            }
            const handle = value_handles[value_index] orelse {
                self.traceFailure("missing_handle", value_index, boundary.reason);
                return error.CommandSubmissionFailed;
            };
            buffer_mod.Opaque.eval(handle) catch |err| {
                self.traceFailure(@errorName(err), value_index, boundary.reason);
                return err;
            };
        }
    }

    fn traceFailure(_: MaterializationBoundaryEval, detail: []const u8, value_index: usize, reason: program_mod.MaterializationReason) void {
        profiling_mod.writeMaterializationFailure(.{
            .detail = detail,
            .value_index = value_index,
            .reason = @tagName(reason),
        });
    }
};

const CustomCallDispatch = struct {
    values: *ValueBindings,

    fn run(self: CustomCallDispatch, instruction: ir.PlanInstruction) Error!?BufferHandle {
        const value_handles = self.values.handles;
        const target = instruction.custom_call_target orelse return null;
        const spec = custom_call_mod.lookup(target) orelse return null;
        return switch (spec.kind) {
            .identity => blk: {
                if (instruction.inputs.len != 1) return null;
                const input_id = instruction.inputs[0];
                if (input_id.index >= value_handles.len) return error.CommandSubmissionFailed;
                const input = value_handles[input_id.index] orelse return error.CommandSubmissionFailed;
                break :blk (try buffer_mod.Opaque.clone(input)) orelse return null;
            },
            .unary => blk: {
                if (instruction.inputs.len != 1) return null;
                const input_id = instruction.inputs[0];
                if (input_id.index >= value_handles.len) return error.CommandSubmissionFailed;
                const input = value_handles[input_id.index] orelse return error.CommandSubmissionFailed;
                break :blk (try buffer_mod.Opaque.unary(input, spec.unary_op.?)) orelse return null;
            },
            .binary => blk: {
                if (instruction.inputs.len != 2) return null;
                const lhs_id = instruction.inputs[0];
                const rhs_id = instruction.inputs[1];
                if (lhs_id.index >= value_handles.len or rhs_id.index >= value_handles.len) return error.CommandSubmissionFailed;
                const lhs = value_handles[lhs_id.index] orelse return error.CommandSubmissionFailed;
                const rhs = value_handles[rhs_id.index] orelse return error.CommandSubmissionFailed;
                break :blk (try buffer_mod.Opaque.binary(lhs, rhs, spec.binary_op.?)) orelse return null;
            },
            .metal_kernel_binary_add_f32 => blk: {
                if (instruction.inputs.len != 2) return null;
                const lhs_id = instruction.inputs[0];
                const rhs_id = instruction.inputs[1];
                if (lhs_id.index >= value_handles.len or rhs_id.index >= value_handles.len) return error.CommandSubmissionFailed;
                const lhs = value_handles[lhs_id.index] orelse return error.CommandSubmissionFailed;
                const rhs = value_handles[rhs_id.index] orelse return error.CommandSubmissionFailed;
                break :blk (try buffer_mod.Opaque.customBinaryAddF32(lhs, rhs)) orelse return error.CommandSubmissionFailed;
            },
            .scaled_dot_product_attention => blk: {
                if (instruction.inputs.len != 4) return null;
                const q_id = instruction.inputs[0];
                const k_id = instruction.inputs[1];
                const v_id = instruction.inputs[2];
                const token_index_id = instruction.inputs[3];
                if (q_id.index >= value_handles.len or k_id.index >= value_handles.len or v_id.index >= value_handles.len or token_index_id.index >= value_handles.len) return error.CommandSubmissionFailed;
                const q = value_handles[q_id.index] orelse return error.CommandSubmissionFailed;
                const k = value_handles[k_id.index] orelse return error.CommandSubmissionFailed;
                const v = value_handles[v_id.index] orelse return error.CommandSubmissionFailed;
                const token_index = value_handles[token_index_id.index] orelse return error.CommandSubmissionFailed;
                break :blk (try buffer_mod.Opaque.customScaledDotProductAttention(q, k, v, token_index)) orelse return error.CommandSubmissionFailed;
            },
        };
    }
};

fn destroyOwnedValueHandles(value_handles: []?BufferHandle, value_owned: []const bool) void {
    for (value_handles, value_owned) |maybe_handle, owned| {
        if (owned) {
            if (maybe_handle) |handle| buffer_mod.Opaque.destroy(handle);
        }
    }
}

fn startHandles(allocator: std.mem.Allocator, value_handles: []const ?BufferHandle, ids: []const ir.ValueId) ![]BufferHandle {
    const handles = try allocator.alloc(BufferHandle, ids.len);
    errdefer allocator.free(handles);
    for (ids, 0..) |id, index| {
        if (id.index >= value_handles.len) return error.CommandSubmissionFailed;
        handles[index] = value_handles[id.index] orelse return error.CommandSubmissionFailed;
    }
    return handles;
}

fn storeOwnedValueHandle(
    value_handles: []?BufferHandle,
    value_owned: []bool,
    value_id: ir.ValueId,
    handle: BufferHandle,
) Error!void {
    if (value_id.index >= value_handles.len) return error.CommandSubmissionFailed;
    if (value_owned[value_id.index]) {
        if (value_handles[value_id.index]) |old| buffer_mod.Opaque.destroy(old);
    }
    value_handles[value_id.index] = handle;
    value_owned[value_id.index] = true;
}

fn storeBorrowedValueHandle(
    value_handles: []?BufferHandle,
    value_owned: []bool,
    value_id: ir.ValueId,
    handle: BufferHandle,
) Error!void {
    if (value_id.index >= value_handles.len) return error.CommandSubmissionFailed;
    if (value_owned[value_id.index]) {
        if (value_handles[value_id.index]) |old| buffer_mod.Opaque.destroy(old);
    }
    value_handles[value_id.index] = handle;
    value_owned[value_id.index] = false;
}

fn reverseIfDescending(
    handle: BufferHandle,
    dimension: i64,
    output_dims: []const i64,
    direction: ir.CompareOp,
) Error!?BufferHandle {
    return switch (direction) {
        .lt, .le => handle,
        .gt, .ge => blk: {
            const dimensions = [_]i64{dimension};
            const reversed = buffer_mod.Opaque.reverse(handle, &dimensions, output_dims) catch |err| {
                buffer_mod.Opaque.destroy(handle);
                return err;
            };
            buffer_mod.Opaque.destroy(handle);
            break :blk reversed;
        },
        else => blk: {
            buffer_mod.Opaque.destroy(handle);
            break :blk null;
        },
    };
}

test "mlx metal backend executable bitcast_convert reinterprets resident bytes" {
    const allocator = std.testing.allocator;
    const dims = [_]i64{2};
    const assignment = [_]i32{0};
    const local_hardware_id: i32 = 0;

    const values = try allocator.alloc(ir.Value, 2);
    errdefer allocator.free(values);
    values[0] = .{
        .id = .{ .index = 0 },
        .role = .parameter,
        .descriptor = .{
            .element_type = .u32,
            .dims = try allocator.dupe(i64, &dims),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };
    values[1] = .{
        .id = .{ .index = 1 },
        .role = .instruction_result,
        .descriptor = .{
            .element_type = .f32,
            .dims = try allocator.dupe(i64, &dims),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };

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
        .output_ids = try allocator.dupe(ir.ValueId, &.{.{ .index = 1 }}),
        .instructions = try allocator.dupe(ir.PlanInstruction, &.{
            .{
                .kind = .bitcast_convert,
                .inputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 0 }}),
                .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 1 }}),
            },
        }),
    };
    defer plan.deinit();

    const executable = (try executable_mod.compile(allocator, &plan, &.{local_hardware_id}, compiledProgramBuildCallback)) orelse return error.TestUnexpectedResult;
    defer destroy(executable);
    const compiled = executable_mod.Executable.fromHandle(executable);
    try std.testing.expectEqual(program_mod.NodeKind.materialize, compiled.program.nodes[0].kind);

    const input_bits = [_]u32{ 0x3f800000, 0xc0000000 };
    const input_bytes = std.mem.sliceAsBytes(&input_bits);
    const arg = (try buffer_mod.Buffer.fromHost(local_hardware_id, .u32, &dims, input_bytes)) orelse return error.TestUnexpectedResult;
    defer arg.destroy();
    const result = (try execute(allocator, executable, 0, &.{arg.toHandle()})) orelse return error.TestUnexpectedResult;
    defer {
        for (result.outputs) |output| buffer_mod.Opaque.destroy(output.handle);
        allocator.free(result.outputs);
    }
    try std.testing.expectEqual(@as(usize, 1), result.outputs.len);

    var out = [_]f32{ 0, 0 };
    try buffer_mod.Opaque.copyToHost(result.outputs[0].handle, std.mem.sliceAsBytes(&out));
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, -2.0), out[1], 0.0001);
}

test "mlx metal backend executable lowers reduce_window sum on device" {
    const allocator = std.testing.allocator;
    const local_hardware_id: i32 = 0;
    const input_dims = [_]i64{3};
    const output_dims = [_]i64{3};
    const assignment = [_]i32{0};

    const values = try allocator.alloc(ir.Value, 2);
    errdefer allocator.free(values);
    values[0] = .{
        .id = .{ .index = 0 },
        .role = .parameter,
        .descriptor = .{
            .element_type = .f32,
            .dims = try allocator.dupe(i64, &input_dims),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };
    values[1] = .{
        .id = .{ .index = 1 },
        .role = .instruction_result,
        .descriptor = .{
            .element_type = .f32,
            .dims = try allocator.dupe(i64, &output_dims),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };

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
        .output_ids = try allocator.dupe(ir.ValueId, &.{.{ .index = 1 }}),
        .instructions = try allocator.dupe(ir.PlanInstruction, &.{.{
            .kind = .reduce_window_sum,
            .inputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 0 }}),
            .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 1 }}),
            .window_dimensions = try allocator.dupe(i64, &.{2}),
            .window_strides = try allocator.dupe(i64, &.{1}),
            .base_dilations = try allocator.dupe(i64, &.{1}),
            .window_dilations = try allocator.dupe(i64, &.{1}),
            .edge_padding_low = try allocator.dupe(i64, &.{1}),
            .edge_padding_high = try allocator.dupe(i64, &.{0}),
        }}),
    };
    defer plan.deinit();

    const executable = (try executable_mod.compile(allocator, &plan, &.{local_hardware_id}, compiledProgramBuildCallback)) orelse return error.TestUnexpectedResult;
    defer destroy(executable);
    const compiled = executable_mod.Executable.fromHandle(executable);
    try std.testing.expectEqual(program_mod.NodeKind.reduction, compiled.program.nodes[0].kind);

    const input = [_]f32{ 1.5, -2.0, 4.0 };
    const arg = (try buffer_mod.Buffer.fromHost(local_hardware_id, .f32, &input_dims, std.mem.asBytes(&input))) orelse return error.TestUnexpectedResult;
    defer arg.destroy();
    const result = (try execute(allocator, executable, 0, &.{arg.toHandle()})) orelse return error.TestUnexpectedResult;
    defer {
        for (result.outputs) |output| buffer_mod.Opaque.destroy(output.handle);
        allocator.free(result.outputs);
    }
    var out = [_]f32{ 0, 0, 0 };
    try buffer_mod.Opaque.copyToHost(result.outputs[0].handle, std.mem.asBytes(&out));
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), out[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, -0.5), out[1], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), out[2], 0.0001);
}

test "mlx metal backend executable lowers tuple get_tuple_element without materializing tuple buffers" {
    const allocator = std.testing.allocator;
    const local_hardware_id: i32 = 0;
    const dims = [_]i64{2};
    const assignment = [_]i32{0};

    const values = try allocator.alloc(ir.Value, 4);
    errdefer allocator.free(values);
    for (values[0..2], 0..) |*value, i| {
        value.* = .{
            .id = .{ .index = @intCast(i) },
            .role = .parameter,
            .descriptor = .{
                .element_type = .f32,
                .dims = try allocator.dupe(i64, &dims),
                .device_id = 0,
                .memory_id = 0,
                .shard_index = 0,
            },
        };
    }
    values[2] = .{
        .id = .{ .index = 2 },
        .role = .instruction_result,
        .descriptor = .{
            .element_type = .invalid,
            .dims = try allocator.dupe(i64, &.{}),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
        .storage = .tuple,
        .elements = try allocator.dupe(ir.ValueId, &.{ .{ .index = 0 }, .{ .index = 1 } }),
    };
    values[3] = .{
        .id = .{ .index = 3 },
        .role = .instruction_result,
        .descriptor = .{
            .element_type = .f32,
            .dims = try allocator.dupe(i64, &dims),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };

    const parameter_shardings = try allocator.alloc(ir.ShardingPlan, 2);
    for (parameter_shardings) |*sharding| {
        sharding.* = .{
            .kind = .replicated,
            .mesh_name = try allocator.dupe(u8, ""),
            .device_assignment = try allocator.dupe(i32, &assignment),
        };
    }
    const output_shardings = try allocator.alloc(ir.ShardingPlan, 1);
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
                .kind = .tuple,
                .inputs = try allocator.dupe(ir.ValueId, &.{ .{ .index = 0 }, .{ .index = 1 } }),
                .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 2 }}),
            },
            .{
                .kind = .get_tuple_element,
                .inputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 2 }}),
                .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 3 }}),
                .tuple_index = 1,
            },
        }),
    };
    defer plan.deinit();

    const executable = (try executable_mod.compile(allocator, &plan, &.{local_hardware_id}, compiledProgramBuildCallback)) orelse return error.TestUnexpectedResult;
    defer destroy(executable);
    const compiled = executable_mod.Executable.fromHandle(executable);
    try std.testing.expectEqual(program_mod.NodeKind.structural, compiled.program.nodes[0].kind);
    try std.testing.expectEqual(program_mod.NodeKind.structural, compiled.program.nodes[1].kind);
    try std.testing.expectEqual(@as(usize, 2), compiled.program.nodes[1].inputs.len);
    try std.testing.expectEqual(@as(usize, 1), compiled.program.nodes[1].inputs[1].index);

    const lhs_data = [_]f32{ 1.0, 2.0 };
    const rhs_data = [_]f32{ 3.0, 4.0 };
    const lhs = (try buffer_mod.Buffer.fromHost(local_hardware_id, .f32, &dims, std.mem.asBytes(&lhs_data))) orelse return error.TestUnexpectedResult;
    defer lhs.destroy();
    const rhs = (try buffer_mod.Buffer.fromHost(local_hardware_id, .f32, &dims, std.mem.asBytes(&rhs_data))) orelse return error.TestUnexpectedResult;
    defer rhs.destroy();

    const result = (try execute(allocator, executable, 0, &.{ lhs.toHandle(), rhs.toHandle() })) orelse return error.TestUnexpectedResult;
    const outputs = result.outputs;
    defer allocator.free(outputs);
    defer for (outputs) |output| buffer_mod.Opaque.destroy(output.handle);

    try std.testing.expectEqual(@as(usize, 1), outputs.len);
    var actual: [2]f32 = undefined;
    try buffer_mod.Opaque.copyToHost(outputs[0].handle, std.mem.asBytes(&actual));
    try std.testing.expectEqualSlices(f32, &rhs_data, &actual);
}

test "mlx metal backend executable runs resident device buffers" {
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

    const executable = (try executable_mod.compile(allocator, &plan, &.{local_hardware_id}, compiledProgramBuildCallback)) orelse return error.TestUnexpectedResult;
    defer destroy(executable);
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

    var executable_stats = stats(executable);
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

    try std.testing.expectError(error.CommandSubmissionFailed, execute(allocator, executable, 1, &.{lhs.toHandle()}));
    try std.testing.expectError(error.CommandSubmissionFailed, execute(allocator, executable, 0, &.{}));
    try std.testing.expectError(error.CommandSubmissionFailed, execute(allocator, executable, 0, &.{ lhs.toHandle(), lhs.toHandle() }));
    executable_stats = stats(executable);
    try std.testing.expectEqual(@as(usize, 0), executable_stats.execute_count);
    try std.testing.expectEqual(@as(usize, std.math.maxInt(usize)), executable_stats.last_execute_device_index);
    try std.testing.expectEqual(@as(i32, -1), executable_stats.last_execute_local_hardware_id);

    const result = (try execute(allocator, executable, 0, &.{lhs.toHandle()})) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(ExecutionCompletionKind.completed, result.completion.kind);
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

    executable_stats = stats(executable);
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
    const second_result = (try execute(allocator, executable, 0, &.{second_lhs.toHandle()})) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(ExecutionCompletionKind.completed, second_result.completion.kind);
    const second_outputs = second_result.outputs;
    defer allocator.free(second_outputs);
    defer {
        for (second_outputs) |output| buffer_mod.Opaque.destroy(output.handle);
    }

    try std.testing.expectEqual(@as(usize, 1), second_outputs.len);
    try std.testing.expectEqual(@as(u1, 0), @intFromBool(buffer_mod.Opaque.hasHostShadow(second_outputs[0].handle)));
    try buffer_mod.Opaque.copyToHost(second_outputs[0].handle, &actual);
    try std.testing.expectEqualSlices(u8, &.{ 8, 21, 40, 65 }, &actual);

    executable_stats = stats(executable);
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

test "mlx metal backend executes f32 lt/add while loop on device" {
    const allocator = std.testing.allocator;
    const local_hardware_id: i32 = 0;
    const assignment = [_]i32{0};

    const scalar_f32 = ir.BufferDescriptor{
        .element_type = .f32,
        .dims = &.{},
        .device_id = 0,
        .memory_id = 0,
        .shard_index = 0,
    };
    const scalar_pred = ir.BufferDescriptor{
        .element_type = .pred,
        .dims = &.{},
        .device_id = 0,
        .memory_id = 0,
        .shard_index = 0,
    };

    const values = try allocator.alloc(ir.Value, 2);
    errdefer allocator.free(values);
    values[0] = .{ .id = .{ .index = 0 }, .role = .parameter, .descriptor = scalar_f32 };
    values[1] = .{ .id = .{ .index = 1 }, .role = .instruction_result, .descriptor = scalar_f32 };

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

    var limit_value: f32 = 4.0;
    var step_value: f32 = 1.0;
    const cond_values = try allocator.dupe(ir.RegionValue, &.{
        .{ .id = .{ .index = 0 }, .role = .argument, .descriptor = scalar_f32 },
        .{ .id = .{ .index = 1 }, .role = .constant, .descriptor = scalar_f32, .literal = try allocator.dupe(u8, std.mem.asBytes(&limit_value)) },
        .{ .id = .{ .index = 2 }, .role = .instruction_result, .descriptor = scalar_pred },
    });
    const body_values = try allocator.dupe(ir.RegionValue, &.{
        .{ .id = .{ .index = 0 }, .role = .argument, .descriptor = scalar_f32 },
        .{ .id = .{ .index = 1 }, .role = .constant, .descriptor = scalar_f32, .literal = try allocator.dupe(u8, std.mem.asBytes(&step_value)) },
        .{ .id = .{ .index = 2 }, .role = .instruction_result, .descriptor = scalar_f32 },
    });

    var regions = try allocator.alloc(ir.PlanRegion, 2);
    regions[0] = .{
        .id = .{ .index = 0 },
        .parent_instruction_index = 0,
        .kind = .while_cond,
        .values = cond_values,
        .argument_descriptors = try allocator.dupe(ir.BufferDescriptor, &.{scalar_f32}),
        .instructions = try allocator.dupe(ir.RegionInstruction, &.{.{
            .kind = .compare,
            .inputs = try allocator.dupe(ir.RegionValueId, &.{ .{ .index = 0 }, .{ .index = 1 } }),
            .outputs = try allocator.dupe(ir.RegionValueId, &.{.{ .index = 2 }}),
            .operand_descriptors = try allocator.dupe(ir.BufferDescriptor, &.{ scalar_f32, scalar_f32 }),
            .result_descriptors = try allocator.dupe(ir.BufferDescriptor, &.{scalar_pred}),
            .compare_direction = .lt,
        }}),
        .return_descriptors = try allocator.dupe(ir.BufferDescriptor, &.{scalar_pred}),
        .terminator_operands = try allocator.dupe(ir.RegionValueId, &.{.{ .index = 2 }}),
        .terminator_operand_descriptors = try allocator.dupe(ir.BufferDescriptor, &.{scalar_pred}),
    };
    regions[1] = .{
        .id = .{ .index = 1 },
        .parent_instruction_index = 0,
        .kind = .while_body,
        .values = body_values,
        .argument_descriptors = try allocator.dupe(ir.BufferDescriptor, &.{scalar_f32}),
        .instructions = try allocator.dupe(ir.RegionInstruction, &.{.{
            .kind = .add,
            .inputs = try allocator.dupe(ir.RegionValueId, &.{ .{ .index = 0 }, .{ .index = 1 } }),
            .outputs = try allocator.dupe(ir.RegionValueId, &.{.{ .index = 2 }}),
            .operand_descriptors = try allocator.dupe(ir.BufferDescriptor, &.{ scalar_f32, scalar_f32 }),
            .result_descriptors = try allocator.dupe(ir.BufferDescriptor, &.{scalar_f32}),
        }}),
        .return_descriptors = try allocator.dupe(ir.BufferDescriptor, &.{scalar_f32}),
        .terminator_operands = try allocator.dupe(ir.RegionValueId, &.{.{ .index = 2 }}),
        .terminator_operand_descriptors = try allocator.dupe(ir.BufferDescriptor, &.{scalar_f32}),
    };

    var plan = ir.ExecutablePlan{
        .allocator = allocator,
        .options = .{
            .num_replicas = 1,
            .num_partitions = 1,
            .device_assignment = try allocator.dupe(i32, &assignment),
        },
        .values = values,
        .regions = regions,
        .parameter_shardings = parameter_shardings,
        .output_shardings = output_shardings,
        .output_ids = try allocator.dupe(ir.ValueId, &.{.{ .index = 1 }}),
        .instructions = try allocator.dupe(ir.PlanInstruction, &.{.{
            .kind = .while_,
            .inputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 0 }}),
            .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 1 }}),
            .region_ids = try allocator.dupe(ir.RegionId, &.{ .{ .index = 0 }, .{ .index = 1 } }),
        }}),
    };
    defer plan.deinit();

    const executable = (try executable_mod.compile(allocator, &plan, &.{local_hardware_id}, compiledProgramBuildCallback)) orelse return error.TestUnexpectedResult;
    defer destroy(executable);
    var state: f32 = 0.0;
    const state_buffer = (try buffer_mod.Buffer.fromHost(local_hardware_id, .f32, &.{}, std.mem.asBytes(&state))) orelse return error.TestUnexpectedResult;
    defer state_buffer.destroy();
    const result = (try execute(allocator, executable, 0, &.{state_buffer.toHandle()})) orelse return error.TestUnexpectedResult;
    defer {
        for (result.outputs) |output| buffer_mod.Opaque.destroy(output.handle);
        allocator.free(result.outputs);
    }
    var actual: f32 = 0.0;
    try buffer_mod.Opaque.copyToHost(result.outputs[0].handle, std.mem.asBytes(&actual));
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), actual, 0.0001);
}

test "mlx metal backend lowers metadata custom call and optimization barrier on device" {
    const allocator = std.testing.allocator;
    const devices = try device_mod.DeviceList.enumerate(allocator);
    defer device_mod.DeviceList.release(allocator, devices);
    const local_hardware_id = devices[0].local_hardware_id;
    const assignment = [_]i32{local_hardware_id};
    const dims = [_]i64{4};

    const values = try allocator.alloc(ir.Value, 5);
    for (values, 0..) |*value, index| {
        value.* = .{
            .id = .{ .index = @intCast(index) },
            .role = if (index < 2) .parameter else .instruction_result,
            .descriptor = .{
                .element_type = .u8,
                .dims = try allocator.dupe(i64, &dims),
                .device_id = 0,
                .memory_id = 0,
                .shard_index = 0,
            },
        };
    }

    const parameter_shardings = try allocator.alloc(ir.ShardingPlan, 2);
    for (parameter_shardings) |*sharding| {
        sharding.* = .{
            .kind = .replicated,
            .mesh_name = try allocator.dupe(u8, "test"),
            .device_assignment = try allocator.dupe(i32, &assignment),
        };
    }

    var plan = ir.ExecutablePlan{
        .allocator = allocator,
        .options = .{
            .num_replicas = 1,
            .num_partitions = 1,
            .device_assignment = try allocator.dupe(i32, &assignment),
        },
        .values = values,
        .parameter_shardings = parameter_shardings,
        .output_shardings = try allocator.alloc(ir.ShardingPlan, 0),
        .output_ids = try allocator.dupe(ir.ValueId, &.{ .{ .index = 3 }, .{ .index = 4 } }),
        .instructions = try allocator.dupe(ir.PlanInstruction, &.{
            .{
                .kind = .optimization_barrier,
                .inputs = try allocator.dupe(ir.ValueId, &.{ .{ .index = 0 }, .{ .index = 1 } }),
                .outputs = try allocator.dupe(ir.ValueId, &.{ .{ .index = 2 }, .{ .index = 3 } }),
                .dims = try allocator.dupe(i64, &dims),
            },
            .{
                .kind = .custom_call,
                .inputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 2 }}),
                .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 4 }}),
                .dims = try allocator.dupe(i64, &dims),
                .custom_call_target = try allocator.dupe(u8, "annotate_device_placement"),
            },
        }),
    };
    defer plan.deinit();

    const executable = (try executable_mod.compile(allocator, &plan, &assignment, compiledProgramBuildCallback)) orelse return error.TestUnexpectedResult;
    defer destroy(executable);

    const lhs = (try buffer_mod.Opaque.fromHost(local_hardware_id, .u8, &dims, &.{ 1, 2, 3, 4 })) orelse return error.TestUnexpectedResult;
    defer buffer_mod.Opaque.destroy(lhs);
    const rhs = (try buffer_mod.Opaque.fromHost(local_hardware_id, .u8, &dims, &.{ 10, 20, 30, 40 })) orelse return error.TestUnexpectedResult;
    defer buffer_mod.Opaque.destroy(rhs);

    const result = (try execute(allocator, executable, 0, &.{ lhs, rhs })) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(ExecutionCompletionKind.completed, result.completion.kind);
    const outputs = result.outputs;
    defer allocator.free(outputs);
    defer {
        for (outputs) |output| buffer_mod.Opaque.destroy(output.handle);
    }

    try std.testing.expectEqual(@as(usize, 2), outputs.len);
    var barrier_rhs: [4]u8 = undefined;
    var annotated_lhs: [4]u8 = undefined;
    try buffer_mod.Opaque.copyToHost(outputs[0].handle, &barrier_rhs);
    try buffer_mod.Opaque.copyToHost(outputs[1].handle, &annotated_lhs);
    try std.testing.expectEqualSlices(u8, &.{ 10, 20, 30, 40 }, &barrier_rhs);
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 3, 4 }, &annotated_lhs);
}

test "mlx metal backend runs registered binary custom call on device buffers" {
    const allocator = std.testing.allocator;
    const local_hardware_id: i32 = 0;
    const assignment = [_]i32{local_hardware_id};
    const dims = [_]i64{2};
    const target = "pjrtx.test.binary_add";

    try custom_call_mod.register(.{
        .target = target,
        .kind = .binary,
        .binary_op = .add,
    });
    defer custom_call_mod.unregister(target);

    const values = try allocator.alloc(ir.Value, 3);
    for (values, 0..) |*value, index| {
        value.* = .{
            .id = .{ .index = @intCast(index) },
            .role = if (index < 2) .parameter else .instruction_result,
            .descriptor = .{
                .element_type = .f32,
                .dims = try allocator.dupe(i64, &dims),
                .device_id = 0,
                .memory_id = 0,
                .shard_index = 0,
            },
        };
    }

    const parameter_shardings = try allocator.alloc(ir.ShardingPlan, 2);
    for (parameter_shardings) |*sharding| {
        sharding.* = .{
            .kind = .replicated,
            .mesh_name = try allocator.dupe(u8, "test"),
            .device_assignment = try allocator.dupe(i32, &assignment),
        };
    }

    var plan = ir.ExecutablePlan{
        .allocator = allocator,
        .options = .{
            .num_replicas = 1,
            .num_partitions = 1,
            .device_assignment = try allocator.dupe(i32, &assignment),
        },
        .values = values,
        .parameter_shardings = parameter_shardings,
        .output_shardings = try allocator.alloc(ir.ShardingPlan, 0),
        .output_ids = try allocator.dupe(ir.ValueId, &.{.{ .index = 2 }}),
        .instructions = try allocator.dupe(ir.PlanInstruction, &.{.{
            .kind = .custom_call,
            .inputs = try allocator.dupe(ir.ValueId, &.{ .{ .index = 0 }, .{ .index = 1 } }),
            .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 2 }}),
            .dims = try allocator.dupe(i64, &dims),
            .custom_call_target = try allocator.dupe(u8, target),
        }}),
    };
    defer plan.deinit();

    const executable = (try executable_mod.compile(allocator, &plan, &assignment, compiledProgramBuildCallback)) orelse return error.TestUnexpectedResult;
    defer destroy(executable);

    const lhs_data = [_]f32{ 1.5, 2.25 };
    const rhs_data = [_]f32{ 4.0, -0.25 };
    const lhs = (try buffer_mod.Opaque.fromHost(local_hardware_id, .f32, &dims, std.mem.asBytes(&lhs_data))) orelse return error.TestUnexpectedResult;
    defer buffer_mod.Opaque.destroy(lhs);
    const rhs = (try buffer_mod.Opaque.fromHost(local_hardware_id, .f32, &dims, std.mem.asBytes(&rhs_data))) orelse return error.TestUnexpectedResult;
    defer buffer_mod.Opaque.destroy(rhs);

    const result = (try execute(allocator, executable, 0, &.{ lhs, rhs })) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(ExecutionCompletionKind.completed, result.completion.kind);
    const outputs = result.outputs;
    defer allocator.free(outputs);
    defer {
        for (outputs) |output| buffer_mod.Opaque.destroy(output.handle);
    }

    var output: [2]f32 = undefined;
    try buffer_mod.Opaque.copyToHost(outputs[0].handle, std.mem.asBytes(&output));
    try std.testing.expectApproxEqAbs(@as(f32, 5.5), output[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), output[1], 0.0001);
}

test "mlx metal backend executable materializes iota on device" {
    const allocator = std.testing.allocator;
    const devices = try device_mod.DeviceList.enumerate(allocator);
    defer device_mod.DeviceList.release(allocator, devices);
    const local_hardware_id = devices[0].local_hardware_id;
    const dims = [_]i64{ 2, 3 };
    const assignment = [_]i32{0};

    const values = try allocator.alloc(ir.Value, 1);
    errdefer allocator.free(values);
    values[0] = .{
        .id = .{ .index = 0 },
        .role = .instruction_result,
        .descriptor = .{
            .element_type = .f32,
            .dims = try allocator.dupe(i64, &dims),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };

    const parameter_shardings = try allocator.alloc(ir.ShardingPlan, 0);
    const output_shardings = try allocator.alloc(ir.ShardingPlan, 1);
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
        .output_ids = try allocator.dupe(ir.ValueId, &.{.{ .index = 0 }}),
        .instructions = try allocator.dupe(ir.PlanInstruction, &.{.{
            .kind = .iota,
            .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 0 }}),
            .dims = try allocator.dupe(i64, &dims),
            .iota_dimension = 1,
        }}),
    };
    defer plan.deinit();

    const executable = (try executable_mod.compile(allocator, &plan, &.{local_hardware_id}, compiledProgramBuildCallback)) orelse return error.TestUnexpectedResult;
    defer destroy(executable);

    const result = (try execute(allocator, executable, 0, &.{})) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(ExecutionCompletionKind.completed, result.completion.kind);
    const outputs = result.outputs;
    defer allocator.free(outputs);
    defer {
        for (outputs) |output| buffer_mod.Opaque.destroy(output.handle);
    }

    try std.testing.expectEqual(@as(usize, 1), outputs.len);
    try std.testing.expectEqual(ir.BufferType.f32, outputs[0].element_type);
    try std.testing.expectEqualSlices(i64, &dims, outputs[0].dims);
    var actual: [6]f32 = undefined;
    try buffer_mod.Opaque.copyToHost(outputs[0].handle, std.mem.asBytes(&actual));
    try std.testing.expectEqualSlices(f32, &.{ 0.0, 1.0, 2.0, 0.0, 1.0, 2.0 }, &actual);
}

test "mlx metal backend executable materializes partition_id on device" {
    const allocator = std.testing.allocator;
    const devices = try device_mod.DeviceList.enumerate(allocator);
    defer device_mod.DeviceList.release(allocator, devices);
    const local_hardware_id = devices[0].local_hardware_id;
    const assignment = [_]i32{0};

    const values = try allocator.alloc(ir.Value, 1);
    errdefer allocator.free(values);
    values[0] = .{
        .id = .{ .index = 0 },
        .role = .instruction_result,
        .descriptor = .{
            .element_type = .u32,
            .dims = try allocator.alloc(i64, 0),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };

    const parameter_shardings = try allocator.alloc(ir.ShardingPlan, 0);
    const output_shardings = try allocator.alloc(ir.ShardingPlan, 1);
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
        .output_ids = try allocator.dupe(ir.ValueId, &.{.{ .index = 0 }}),
        .instructions = try allocator.dupe(ir.PlanInstruction, &.{.{
            .kind = .partition_id,
            .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 0 }}),
        }}),
    };
    defer plan.deinit();

    const executable = (try executable_mod.compile(allocator, &plan, &.{local_hardware_id}, compiledProgramBuildCallback)) orelse return error.TestUnexpectedResult;
    defer destroy(executable);

    const result = (try execute(allocator, executable, 0, &.{})) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(ExecutionCompletionKind.completed, result.completion.kind);
    const outputs = result.outputs;
    defer allocator.free(outputs);
    defer {
        for (outputs) |output| buffer_mod.Opaque.destroy(output.handle);
    }

    try std.testing.expectEqual(@as(usize, 1), outputs.len);
    try std.testing.expectEqual(ir.BufferType.u32, outputs[0].element_type);
    try std.testing.expectEqual(@as(usize, 0), outputs[0].dims.len);
    var actual: u32 = undefined;
    try buffer_mod.Opaque.copyToHost(outputs[0].handle, std.mem.asBytes(&actual));
    try std.testing.expectEqual(@as(u32, 0), actual);
}

test "mlx metal backend executable lowers deprecated rng on device" {
    const allocator = std.testing.allocator;
    const devices = try device_mod.DeviceList.enumerate(allocator);
    defer device_mod.DeviceList.release(allocator, devices);
    const local_hardware_id = devices[0].local_hardware_id;
    const output_dims = [_]i64{ 2, 4 };
    const assignment = [_]i32{0};

    const values = try allocator.alloc(ir.Value, 3);
    errdefer allocator.free(values);
    values[0] = .{
        .id = .{ .index = 0 },
        .role = .parameter,
        .descriptor = .{
            .element_type = .f32,
            .dims = try allocator.alloc(i64, 0),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };
    values[1] = .{
        .id = .{ .index = 1 },
        .role = .parameter,
        .descriptor = .{
            .element_type = .f32,
            .dims = try allocator.alloc(i64, 0),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };
    values[2] = .{
        .id = .{ .index = 2 },
        .role = .instruction_result,
        .descriptor = .{
            .element_type = .f32,
            .dims = try allocator.dupe(i64, &output_dims),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };

    var parameter_shardings = try allocator.alloc(ir.ShardingPlan, 2);
    parameter_shardings[0] = .{
        .kind = .replicated,
        .mesh_name = try allocator.dupe(u8, ""),
        .device_assignment = try allocator.dupe(i32, &assignment),
    };
    parameter_shardings[1] = .{
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
        .output_ids = try allocator.dupe(ir.ValueId, &.{.{ .index = 2 }}),
        .instructions = try allocator.dupe(ir.PlanInstruction, &.{.{
            .kind = .rng,
            .inputs = try allocator.dupe(ir.ValueId, &.{ .{ .index = 0 }, .{ .index = 1 } }),
            .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 2 }}),
            .dims = try allocator.dupe(i64, &output_dims),
            .rng_distribution = .normal,
        }}),
    };
    defer plan.deinit();

    const executable = (try executable_mod.compile(allocator, &plan, &.{local_hardware_id}, compiledProgramBuildCallback)) orelse return error.TestUnexpectedResult;
    defer destroy(executable);

    var mean: f32 = 0.0;
    var scale: f32 = 1.0;
    const mean_buffer = (try buffer_mod.Opaque.fromHost(local_hardware_id, .f32, &.{}, std.mem.asBytes(&mean))) orelse return error.TestUnexpectedResult;
    defer buffer_mod.Opaque.destroy(mean_buffer);
    const scale_buffer = (try buffer_mod.Opaque.fromHost(local_hardware_id, .f32, &.{}, std.mem.asBytes(&scale))) orelse return error.TestUnexpectedResult;
    defer buffer_mod.Opaque.destroy(scale_buffer);

    const result = (try execute(allocator, executable, 0, &.{ mean_buffer, scale_buffer })) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(ExecutionCompletionKind.completed, result.completion.kind);
    const outputs = result.outputs;
    defer allocator.free(outputs);
    defer {
        for (outputs) |output| buffer_mod.Opaque.destroy(output.handle);
    }

    try std.testing.expectEqual(@as(usize, 1), outputs.len);
    try std.testing.expectEqual(ir.BufferType.f32, outputs[0].element_type);
    try std.testing.expectEqualSlices(i64, &output_dims, outputs[0].dims);
    var actual: [8]f32 = undefined;
    try buffer_mod.Opaque.copyToHost(outputs[0].handle, std.mem.asBytes(&actual));
    for (actual) |value| try std.testing.expect(std.math.isFinite(value));
}

test "mlx metal backend executable lowers clamp with scalar bounds" {
    const allocator = std.testing.allocator;
    const devices = try device_mod.DeviceList.enumerate(allocator);
    defer device_mod.DeviceList.release(allocator, devices);
    const local_hardware_id = devices[0].local_hardware_id;
    const dims = [_]i64{3};
    const assignment = [_]i32{0};
    const min_literal_value: f32 = -1.0;
    const max_literal_value: f32 = 2.0;

    const values = try allocator.alloc(ir.Value, 4);
    errdefer allocator.free(values);
    values[0] = .{
        .id = .{ .index = 0 },
        .role = .constant,
        .descriptor = .{
            .element_type = .f32,
            .dims = try allocator.dupe(i64, &.{}),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };
    values[1] = .{
        .id = .{ .index = 1 },
        .role = .parameter,
        .descriptor = .{
            .element_type = .f32,
            .dims = try allocator.dupe(i64, &dims),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };
    values[2] = .{
        .id = .{ .index = 2 },
        .role = .constant,
        .descriptor = .{
            .element_type = .f32,
            .dims = try allocator.dupe(i64, &.{}),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };
    values[3] = .{
        .id = .{ .index = 3 },
        .role = .instruction_result,
        .descriptor = .{
            .element_type = .f32,
            .dims = try allocator.dupe(i64, &dims),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };

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
                .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 0 }}),
                .literal = try allocator.dupe(u8, std.mem.asBytes(&min_literal_value)),
            },
            .{
                .kind = .constant,
                .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 2 }}),
                .literal = try allocator.dupe(u8, std.mem.asBytes(&max_literal_value)),
            },
            .{
                .kind = .clamp,
                .inputs = try allocator.dupe(ir.ValueId, &.{ .{ .index = 0 }, .{ .index = 1 }, .{ .index = 2 } }),
                .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 3 }}),
                .dims = try allocator.dupe(i64, &dims),
            },
        }),
    };
    defer plan.deinit();

    const executable = (try executable_mod.compile(allocator, &plan, &.{local_hardware_id}, compiledProgramBuildCallback)) orelse return error.TestUnexpectedResult;
    defer destroy(executable);

    const input = [_]f32{ -2.0, 0.5, 3.0 };
    const input_buffer = (try buffer_mod.Opaque.fromHost(local_hardware_id, .f32, &dims, std.mem.asBytes(&input))) orelse return error.TestUnexpectedResult;
    defer buffer_mod.Opaque.destroy(input_buffer);

    const result = (try execute(allocator, executable, 0, &.{input_buffer})) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(ExecutionCompletionKind.completed, result.completion.kind);
    const outputs = result.outputs;
    defer allocator.free(outputs);
    defer {
        for (outputs) |output| buffer_mod.Opaque.destroy(output.handle);
    }

    var actual: [3]f32 = undefined;
    try buffer_mod.Opaque.copyToHost(outputs[0].handle, std.mem.asBytes(&actual));
    try std.testing.expectEqualSlices(f32, &.{ -1.0, 0.5, 2.0 }, &actual);
}
