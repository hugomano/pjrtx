const std = @import("std");

const runtime = @import("src/runtime");
const c = @import("c");
const abi = @import("pjrt_abi.zig");
const errors = @import("errors.zig");
const events_mod = @import("events.zig");
const state = @import("state.zig");

const allocator = state.allocator;
const Executable = state.Executable;
const LoadedExecutableHandle = abi.LoadedExecutable(Executable);
const failedPrecondition = errors.failedPrecondition;
const internal = errors.internal;
const invalidArgument = errors.invalidArgument;
const resourceExhausted = errors.resourceExhausted;
const unimplemented = errors.unimplemented;
const eventCreateFromRuntime = events_mod.eventCreateFromRuntime;
const runtimeEvent = events_mod.runtimeEvent;

pub fn graphExecuteError(err: runtime.GraphExecuteError) ?*c.PJRT_Error {
    return switch (err) {
        error.OutOfMemory => internal("failed to allocate executable graph execution state"),
        error.InvalidArgument => invalidArgument("invalid executable graph arguments or device assignment"),
        error.UnsupportedElementType => unimplemented("executable graph contains an operation unsupported for this element type"),
        error.ShapeMismatch => invalidArgument("executable graph shape validation failed during execution"),
        error.UnsupportedRuntimeFeature => unimplemented("executable graph is not fully lowered to the MLX backend executable"),
        error.BufferDeleted => failedPrecondition("execute attempted to use a deleted buffer"),
        error.BufferDonated => failedPrecondition("execute attempted to use a donated buffer"),
        error.BufferNotReady => failedPrecondition("execute attempted to use a buffer that is not ready"),
        error.BufferReadinessFailed => failedPrecondition("execute attempted to use a buffer with failed readiness"),
        error.Internal => internal("failed to execute executable graph"),
    };
}

const ExecuteOptions = struct {
    non_donatable_input_indices: []const i64 = &.{},

    fn init(options: ?*c.PJRT_ExecuteOptions) ExecuteOptions {
        const raw = options orelse return .{};
        if (raw.non_donatable_input_indices == null) return .{};
        return .{
            .non_donatable_input_indices = raw.non_donatable_input_indices[0..raw.num_non_donatable_input_indices],
        };
    }

    fn keepsParameter(self: ExecuteOptions, parameter_index: usize) bool {
        for (self.non_donatable_input_indices) |non_donatable| {
            if (non_donatable >= 0 and @as(usize, @intCast(non_donatable)) == parameter_index) return true;
        }
        return false;
    }

    fn validate(self: ExecuteOptions, raw_options: ?*c.PJRT_ExecuteOptions, num_args: usize) ?*c.PJRT_Error {
        const raw = raw_options orelse return null;
        if (raw.num_non_donatable_input_indices != 0 and raw.non_donatable_input_indices == null) {
            return invalidArgument("non_donatable_input_indices count requires a non-null index list");
        }
        for (self.non_donatable_input_indices) |non_donatable| {
            if (non_donatable < 0 or @as(usize, @intCast(non_donatable)) >= num_args) {
                return invalidArgument("non_donatable_input_indices contains an out-of-range argument index");
            }
        }
        return null;
    }
};

const ExecuteCall = struct {
    raw: *allowzero c.PJRT_LoadedExecutable_Execute_Args,
    executable: *Executable,
    options: ExecuteOptions,

    fn init(raw: *allowzero c.PJRT_LoadedExecutable_Execute_Args) ExecuteCall {
        return .{
            .raw = raw,
            .executable = LoadedExecutableHandle.view(raw.executable),
            .options = ExecuteOptions.init(raw.options),
        };
    }

    fn numDevices(self: ExecuteCall) usize {
        return self.raw.num_devices;
    }

    fn numArgs(self: ExecuteCall) usize {
        return self.raw.num_args;
    }

    fn expectedArgs(self: ExecuteCall) usize {
        return self.executable.plan.parameter_shardings.len;
    }

    fn expectedOutputs(self: ExecuteCall) usize {
        return self.executable.plan.output_ids.len;
    }

    fn argument(self: ExecuteCall, device_index: usize, argument_index: usize) *runtime.Buffer {
        return abi.Buffer.view(self.raw.argument_lists[device_index][argument_index]);
    }

    fn rawArgument(self: ExecuteCall, device_index: usize, argument_index: usize) ?*c.PJRT_Buffer {
        return self.raw.argument_lists[device_index][argument_index];
    }

    fn clearResults(self: ExecuteCall, output_count: usize) void {
        for (0..self.numDevices()) |device_index| {
            if (self.raw.output_lists != null) {
                const outputs = self.raw.output_lists[device_index];
                if (outputs != null) {
                    for (0..output_count) |output_index| outputs[output_index] = null;
                }
            }
            if (self.raw.device_complete_events) |events| events[device_index] = null;
        }
    }

    fn cleanupResults(self: ExecuteCall, output_count: usize) void {
        for (0..self.numDevices()) |device_index| {
            if (self.raw.output_lists != null) {
                const outputs = self.raw.output_lists[device_index];
                if (outputs != null) {
                    for (0..output_count) |output_index| {
                        if (outputs[output_index]) |output| {
                            abi.Buffer.view(output).deinit();
                            outputs[output_index] = null;
                        }
                    }
                }
            }
            if (self.raw.device_complete_events) |events| {
                if (events[device_index]) |event| {
                    var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
                    event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
                    event_destroy_args.event = event;
                    _ = events_mod.Api.Destroy(&event_destroy_args);
                    events[device_index] = null;
                }
            }
        }
    }

    fn completionEventSlot(self: ExecuteCall, device_index: usize, source: runtime.Event) ?*runtime.Event {
        if (self.raw.device_complete_events) |events| {
            events[device_index] = eventCreateFromRuntime(source) orelse return null;
            return runtimeEvent(events[device_index].?);
        }
        return null;
    }

    fn assignOutput(self: ExecuteCall, device_index: usize, output_index: usize, output: *runtime.Buffer) void {
        self.raw.output_lists[device_index][output_index] = abi.Buffer.handle(output);
    }

    fn validate(self: ExecuteCall) ?*c.PJRT_Error {
        if (self.executable.deleted) return failedPrecondition("loaded executable has been deleted");
        if (self.validateLists()) |err| return err;
        if (self.options.validate(self.raw.options, self.numArgs())) |err| return err;
        return self.validateDonationAliasHazards();
    }

    fn validateLists(self: ExecuteCall) ?*c.PJRT_Error {
        const expected_args = self.expectedArgs();
        const expected_outputs = self.expectedOutputs();
        if (self.numDevices() == 0) return invalidArgument("PjRTx execute requires at least one device");
        if (self.numDevices() > self.executable.graph.device_ids.len) return invalidArgument("PjRTx execute requested more devices than the executable graph contains");
        if (self.numArgs() != expected_args) return invalidArgument("PjRTx execute argument count does not match executable parameters");
        if (expected_args != 0 and self.raw.argument_lists == null) return invalidArgument("PjRTx execute requires non-null argument_lists for executable parameters");
        if (expected_outputs != 0 and self.raw.output_lists == null) return invalidArgument("PjRTx execute requires non-null output_lists for executable outputs");
        for (0..self.numDevices()) |device_index| {
            if (expected_args != 0 and self.raw.argument_lists[device_index] == null) {
                return invalidArgument("PjRTx execute requires a non-null argument list for every device");
            }
            if (expected_outputs != 0 and self.raw.output_lists[device_index] == null) {
                return invalidArgument("PjRTx execute requires a non-null output list for every device");
            }
            for (0..expected_args) |argument_index| {
                if (self.raw.argument_lists[device_index][argument_index] == null) {
                    return invalidArgument("PjRTx execute argument list contains a null buffer");
                }
            }
        }
        return null;
    }

    fn validateDonationAliasHazards(self: ExecuteCall) ?*c.PJRT_Error {
        for (0..self.numDevices()) |donor_device_index| {
            for (0..self.numArgs()) |donor_argument_index| {
                if (!executableDonatesParameter(self.executable, donor_argument_index)) continue;
                if (self.options.keepsParameter(donor_argument_index)) continue;
                const donor_buffer = self.rawArgument(donor_device_index, donor_argument_index).?;
                for (0..self.numDevices()) |other_device_index| {
                    for (0..self.numArgs()) |other_argument_index| {
                        if (donor_device_index == other_device_index and donor_argument_index == other_argument_index) continue;
                        if (self.rawArgument(other_device_index, other_argument_index) == donor_buffer) {
                            return invalidArgument("donated execute argument aliases another argument");
                        }
                    }
                }
            }
        }
        return null;
    }
};

const DonationSet = struct {
    list: std.ArrayList(*runtime.Buffer) = .empty,

    fn deinit(self: *DonationSet) void {
        self.list.deinit(allocator);
    }

    fn record(self: *DonationSet, buffer: *runtime.Buffer) !void {
        for (self.list.items) |existing| {
            if (existing == buffer) return;
        }
        try self.list.append(allocator, buffer);
    }

    fn commit(self: DonationSet) void {
        for (self.list.items) |argument| argument.markDonated();
    }
};

fn executableDonatesParameter(executable: *const Executable, parameter_index: usize) bool {
    for (executable.plan.donated_parameter_indices) |candidate| {
        if (candidate == parameter_index) return true;
    }
    return false;
}

pub fn cleanupUnassignedExecuteOutputs(outputs: []const *runtime.Buffer, assigned_count: usize) void {
    if (assigned_count >= outputs.len) return;
    for (outputs[assigned_count..]) |output| output.deinit();
}

test "execute cleanup destroys only unassigned runtime outputs" {
    const test_allocator = std.testing.allocator;
    const client = try runtime.createClient(test_allocator, .{ .backend_kind = .metal_mlx, .device_count = 1 });
    defer client.deinit();

    const dims = [_]i64{4};
    const first_data = [_]u8{ 1, 2, 3, 4 };
    const second_data = [_]u8{ 5, 6, 7, 8 };
    const first = try client.createHostBufferFromBytes(test_allocator, .u8, &dims, &client.devices[0], &client.memories[0], 0, &first_data);
    defer first.deinit();
    const second = try client.createHostBufferFromBytes(test_allocator, .u8, &dims, &client.devices[0], &client.memories[0], 0, &second_data);

    const bytes_before_cleanup = client.memories[0].stats.bytes_in_use;
    const second_bytes: u64 = @intCast(second.byte_size);
    var outputs = [_]*runtime.Buffer{ first, second };
    cleanupUnassignedExecuteOutputs(&outputs, 1);

    try first.ensureUsable();
    try std.testing.expectEqual(bytes_before_cleanup - second_bytes, client.memories[0].stats.bytes_in_use);
}

pub fn loadedExecutableExecute(args: [*c]c.PJRT_LoadedExecutable_Execute_Args) callconv(.c) ?*c.PJRT_Error {
    const call = ExecuteCall.init(&args[0]);
    const executable = call.executable;
    if (call.validate()) |err| return err;
    const output_count = call.expectedOutputs();
    call.clearResults(output_count);
    var donated_arguments: DonationSet = .{};
    defer donated_arguments.deinit();
    for (0..call.numDevices()) |device_index| {
        const arguments = allocator.alloc(*runtime.Buffer, call.numArgs()) catch {
            call.cleanupResults(output_count);
            return internal("failed to allocate executable graph argument list");
        };
        defer allocator.free(arguments);
        for (arguments, 0..) |*argument, argument_index| {
            argument.* = call.argument(device_index, argument_index);
            if (executableDonatesParameter(executable, argument_index) and !call.options.keepsParameter(argument_index)) {
                donated_arguments.record(argument.*) catch {
                    call.cleanupResults(output_count);
                    return internal("failed to record donated executable argument");
                };
            }
        }

        const execute_result = executable.graph.executeDevice(allocator, executable.client, executable.plan, device_index, arguments) catch |err| {
            call.cleanupResults(output_count);
            return graphExecuteError(err);
        };
        const outputs = execute_result.outputs;
        defer allocator.free(outputs);
        var assigned_outputs: usize = 0;
        var outputs_transferred = false;
        defer if (!outputs_transferred) cleanupUnassignedExecuteOutputs(outputs, assigned_outputs);
        var stack_completion_event = execute_result.completion_event;
        const completion_event = call.completionEventSlot(device_index, execute_result.completion_event) orelse &stack_completion_event;
        if (call.raw.device_complete_events != null and completion_event == &stack_completion_event) {
            call.cleanupResults(output_count);
            return internal("failed to allocate execute completion event");
        }
        for (outputs, 0..) |output, output_index| {
            call.assignOutput(device_index, output_index, output);
            assigned_outputs = output_index + 1;
            output.chainReadyAfter(completion_event) catch {
                call.cleanupResults(output_count);
                return resourceExhausted("too many output readiness dependencies for execute completion event");
            };
        }
        outputs_transferred = true;
    }
    donated_arguments.commit();
    return null;
}

pub const LoadedExecutableApi = struct {
    pub const Execute = loadedExecutableExecute;
};
