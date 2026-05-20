const std = @import("std");

const c = @import("c");
const runtime = @import("src/runtime");

const abi = @import("pjrt_abi.zig");
const errors = @import("errors.zig");
const events_mod = @import("events.zig");
const executable_mod = @import("executable.zig");
const handles = @import("pjrt_handles.zig");
const plugin = @import("plugin.zig");

const Executable = executable_mod.Executable;
const LoadedExecutableHandle = handles.LoadedExecutable(Executable);
const PjrtError = errors.Error;
const PjrtEvent = events_mod.Event;

/// PJRT loaded-executable execution callbacks.
pub const Execution = struct {
    pub const Api = struct {
        pub const Execute = ExecuteApiCallback.call;
    };
};

fn graphExecuteError(err: runtime.GraphExecuteError) ?*c.PJRT_Error {
    return switch (err) {
        error.OutOfMemory => PjrtError.internal("failed to allocate executable graph execution state"),
        error.InvalidArgument => PjrtError.invalidArgument("invalid executable graph arguments or device assignment"),
        error.UnsupportedElementType => PjrtError.unimplemented("executable graph contains an operation unsupported for this element type"),
        error.ShapeMismatch => PjrtError.invalidArgument("executable graph shape validation failed during execution"),
        error.UnsupportedRuntimeFeature => PjrtError.unimplemented("executable graph is not fully lowered to the MLX backend executable"),
        error.BufferDeleted => PjrtError.failedPrecondition("execute attempted to use a deleted buffer"),
        error.BufferDonated => PjrtError.failedPrecondition("execute attempted to use a donated buffer"),
        error.BufferNotReady => PjrtError.failedPrecondition("execute attempted to use a buffer that is not ready"),
        error.BufferReadinessFailed => PjrtError.failedPrecondition("execute attempted to use a buffer with failed readiness"),
        error.Internal => PjrtError.internal("failed to execute executable graph"),
    };
}

const ExecuteOptions = struct {
    non_donatable_input_indices: []const i64 = &.{},

    fn init(options: ?*c.PJRT_ExecuteOptions) ExecuteOptions {
        const raw = options orelse return .{};
        if (raw.non_donatable_input_indices == null) return .{};
        return .{
            .non_donatable_input_indices = abi.Slice.constList(i64, raw.non_donatable_input_indices, raw.num_non_donatable_input_indices) orelse &.{},
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
            return PjrtError.invalidArgument("non_donatable_input_indices count requires a non-null index list");
        }
        for (self.non_donatable_input_indices) |non_donatable| {
            if (non_donatable < 0 or @as(usize, @intCast(non_donatable)) >= num_args) {
                return PjrtError.invalidArgument("non_donatable_input_indices contains an out-of-range argument index");
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
            .executable = LoadedExecutableHandle.ref(raw.executable),
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
        return self.executable.parameterCount();
    }

    fn expectedOutputs(self: ExecuteCall) usize {
        return self.executable.outputCount();
    }

    fn argument(self: ExecuteCall, device_index: usize, argument_index: usize) *runtime.Buffer {
        return handles.Buffer.ref(self.raw.argument_lists[device_index][argument_index]);
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
                            handles.Buffer.ref(output).deinit();
                            outputs[output_index] = null;
                        }
                    }
                }
            }
            if (self.raw.device_complete_events) |events| {
                if (events[device_index]) |event| {
                    PjrtEvent.destroy(event);
                    events[device_index] = null;
                }
            }
        }
    }

    fn completionEventSlot(self: ExecuteCall, device_index: usize, source: runtime.Event) ?*runtime.Event {
        if (self.raw.device_complete_events) |events| {
            const completion = PjrtEvent.Completion.fromRuntime(&events[device_index], source) orelse return null;
            return completion.runtime();
        }
        return null;
    }

    fn assignOutput(self: ExecuteCall, device_index: usize, output_index: usize, output: *runtime.Buffer) void {
        self.raw.output_lists[device_index][output_index] = handles.Buffer.handle(output);
    }

    fn validate(self: ExecuteCall) ?*c.PJRT_Error {
        if (self.executable.isDeleted()) return PjrtError.failedPrecondition("loaded executable has been deleted");
        if (self.validateLists()) |err| return err;
        if (self.options.validate(self.raw.options, self.numArgs())) |err| return err;
        return self.validateDonationAliasHazards();
    }

    fn validateLists(self: ExecuteCall) ?*c.PJRT_Error {
        const expected_args = self.expectedArgs();
        const expected_outputs = self.expectedOutputs();
        if (self.numDevices() == 0) return PjrtError.invalidArgument("PjRTx execute requires at least one device");
        if (self.numDevices() > self.executable.graphDeviceCount()) return PjrtError.invalidArgument("PjRTx execute requested more devices than the executable graph contains");
        if (self.numArgs() != expected_args) return PjrtError.invalidArgument("PjRTx execute argument count does not match executable parameters");
        if (expected_args != 0 and self.raw.argument_lists == null) return PjrtError.invalidArgument("PjRTx execute requires non-null argument_lists for executable parameters");
        if (expected_outputs != 0 and self.raw.output_lists == null) return PjrtError.invalidArgument("PjRTx execute requires non-null output_lists for executable outputs");
        for (0..self.numDevices()) |device_index| {
            if (expected_args != 0 and self.raw.argument_lists[device_index] == null) {
                return PjrtError.invalidArgument("PjRTx execute requires a non-null argument list for every device");
            }
            if (expected_outputs != 0 and self.raw.output_lists[device_index] == null) {
                return PjrtError.invalidArgument("PjRTx execute requires a non-null output list for every device");
            }
            for (0..expected_args) |argument_index| {
                if (self.raw.argument_lists[device_index][argument_index] == null) {
                    return PjrtError.invalidArgument("PjRTx execute argument list contains a null buffer");
                }
            }
        }
        return null;
    }

    fn validateDonationAliasHazards(self: ExecuteCall) ?*c.PJRT_Error {
        for (0..self.numDevices()) |donor_device_index| {
            for (0..self.numArgs()) |donor_argument_index| {
                if (!self.executable.donatesParameter(donor_argument_index)) continue;
                if (self.options.keepsParameter(donor_argument_index)) continue;
                const donor_buffer = self.rawArgument(donor_device_index, donor_argument_index).?;
                for (0..self.numDevices()) |other_device_index| {
                    for (0..self.numArgs()) |other_argument_index| {
                        if (donor_device_index == other_device_index and donor_argument_index == other_argument_index) continue;
                        if (self.rawArgument(other_device_index, other_argument_index) == donor_buffer) {
                            return PjrtError.invalidArgument("donated execute argument aliases another argument");
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
        self.list.deinit(plugin.allocator());
    }

    fn record(self: *DonationSet, buffer: *runtime.Buffer) !void {
        for (self.list.items) |existing| {
            if (existing == buffer) return;
        }
        try self.list.append(plugin.allocator(), buffer);
    }

    fn commit(self: DonationSet) void {
        for (self.list.items) |argument| argument.markDonated();
    }
};

fn cleanupUnassignedExecuteOutputs(outputs: []const *runtime.Buffer, assigned_count: usize) void {
    if (assigned_count >= outputs.len) return;
    for (outputs[assigned_count..]) |output| output.deinit();
}

const ExecuteApiCallback = struct {
    fn call(args: [*c]c.PJRT_LoadedExecutable_Execute_Args) callconv(.c) ?*c.PJRT_Error {
        const request = ExecuteCall.init(&args[0]);
        return run(request);
    }

    fn run(request: ExecuteCall) ?*c.PJRT_Error {
        const executable = request.executable;
        if (request.validate()) |err| return err;
        const output_count = request.expectedOutputs();
        request.clearResults(output_count);
        var donated_arguments: DonationSet = .{};
        defer donated_arguments.deinit();
        for (0..request.numDevices()) |device_index| {
            if (runDevice(request, executable, &donated_arguments, device_index, output_count)) |err| return err;
        }
        donated_arguments.commit();
        return null;
    }

    fn runDevice(request: ExecuteCall, executable: *Executable, donated_arguments: *DonationSet, device_index: usize, output_count: usize) ?*c.PJRT_Error {
        const arguments = plugin.allocator().alloc(*runtime.Buffer, request.numArgs()) catch {
            request.cleanupResults(output_count);
            return PjrtError.internal("failed to allocate executable graph argument list");
        };
        defer plugin.allocator().free(arguments);
        for (arguments, 0..) |*argument, argument_index| {
            argument.* = request.argument(device_index, argument_index);
            if (executable.donatesParameter(argument_index) and !request.options.keepsParameter(argument_index)) {
                donated_arguments.record(argument.*) catch {
                    request.cleanupResults(output_count);
                    return PjrtError.internal("failed to record donated executable argument");
                };
            }
        }

        const execute_result = executable.executeDevice(device_index, arguments) catch |err| {
            request.cleanupResults(output_count);
            return graphExecuteError(err);
        };
        return assignDeviceOutputs(request, device_index, output_count, execute_result);
    }

    fn assignDeviceOutputs(request: ExecuteCall, device_index: usize, output_count: usize, execute_result: runtime.GraphExecuteResult) ?*c.PJRT_Error {
        const outputs = execute_result.outputs;
        defer plugin.allocator().free(outputs);
        var assigned_outputs: usize = 0;
        var outputs_transferred = false;
        defer if (!outputs_transferred) cleanupUnassignedExecuteOutputs(outputs, assigned_outputs);
        var stack_completion_event = execute_result.completion_event;
        const completion_event = request.completionEventSlot(device_index, execute_result.completion_event) orelse &stack_completion_event;
        if (request.raw.device_complete_events != null and completion_event == &stack_completion_event) {
            request.cleanupResults(output_count);
            return PjrtError.internal("failed to allocate execute completion event");
        }
        for (outputs, 0..) |output, output_index| {
            request.assignOutput(device_index, output_index, output);
            assigned_outputs = output_index + 1;
            output.chainReadyAfter(completion_event) catch {
                request.cleanupResults(output_count);
                return PjrtError.resourceExhausted("too many output readiness dependencies for execute completion event");
            };
        }
        outputs_transferred = true;
        return null;
    }
};

test "execute cleanup destroys only unassigned runtime outputs" {
    const test_allocator = std.testing.allocator;
    const client = try runtime.createClient(test_allocator, .{ .device_count = 1 });
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
