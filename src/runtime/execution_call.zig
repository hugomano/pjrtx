const std = @import("std");
const backend_api = @import("backend_selection.zig");
const ir = @import("src/compiler/ir");

const buffer_mod = @import("buffer.zig");
const client_mod = @import("client.zig");
const event_mod = @import("event.zig");
const executable_mod = @import("executable.zig");
const execution_completion = @import("execution_completion.zig");
const execution_outputs = @import("execution_outputs.zig");

const Buffer = buffer_mod.Buffer;
const Client = client_mod.Client;
const Event = event_mod.Event;
const CompiledExecutable = executable_mod.CompiledExecutable;
const ExecutionContext = executable_mod.Context;

/// Result of one per-device executable dispatch.
pub const Result = struct {
    outputs: []*Buffer,
    completion_event: Event = Event.ready(),
};

/// Errors produced while validating and dispatching a compiled executable.
pub const Error = error{
    OutOfMemory,
    InvalidArgument,
    UnsupportedElementType,
    ShapeMismatch,
    UnsupportedRuntimeFeature,
    BufferDeleted,
    BufferDonated,
    BufferNotReady,
    BufferReadinessFailed,
    Internal,
};

/// Validates runtime arguments and dispatches one device slice.
pub fn execute(
    executable: *CompiledExecutable,
    allocator: std.mem.Allocator,
    context: ExecutionContext,
    device_index: usize,
    arguments: []const *Buffer,
) Error!Result {
    if (device_index >= executable.deviceCount() or device_index >= context.deviceCount()) return error.InvalidArgument;
    if (arguments.len != executable.parameterCount()) return error.InvalidArgument;
    for (arguments) |argument| {
        argument.ensureUsable() catch |err| return mapBufferError(err);
        argument.ensureReady() catch |err| return mapBufferError(err);
        if (!argumentMatchesDevice(executable, device_index, argument)) return error.InvalidArgument;
    }
    if (tryExecuteBackendExecutable(executable, allocator, context, device_index, arguments) catch |err| return err) |result| {
        return result;
    }
    return error.UnsupportedRuntimeFeature;
}

fn argumentMatchesDevice(executable: *const CompiledExecutable, device_index: usize, argument: *const Buffer) bool {
    const device_id = executable.deviceIdAt(device_index) orelse return false;
    return argument.matchesExecutionSlot(device_id, device_index);
}

fn tryExecuteBackendExecutable(
    executable: *CompiledExecutable,
    allocator: std.mem.Allocator,
    context: ExecutionContext,
    device_index: usize,
    arguments: []const *Buffer,
) Error!?Result {
    const backend_executable = executable.backendExecutableForDispatch() orelse return null;
    var argument_handles = try allocator.alloc(backend_api.BufferHandle, arguments.len);
    defer allocator.free(argument_handles);
    for (arguments, 0..) |argument, i| {
        argument.ensureUsable() catch |err| return mapBufferError(err);
        argument.ensureReady() catch |err| return mapBufferError(err);
        argument_handles[i] = argument.backendHandleForDispatch() orelse return null;
    }

    const device = context.deviceAt(device_index) orelse return error.InvalidArgument;
    const memory = device.default_memory;
    executable.recordExecuteCacheTrim(context.trimExecutableCacheForAllocation(memory, executable.outputByteSize() catch return error.Internal));

    const backend_result = context.executeBackendExecutable(allocator, backend_executable, device_index, argument_handles) catch |err| return mapBufferError(err);
    const owned_backend_result = backend_result orelse return null;
    const completion_event = execution_completion.eventFromBackendCompletion(context, owned_backend_result.completion);
    executable.recordBackendCompletion(owned_backend_result.completion);

    const outputs = execution_outputs.wrapBackendOutputs(
        allocator,
        context,
        executable,
        device_index,
        arguments,
        owned_backend_result.outputs,
    ) catch |err| return mapBufferError(err);

    return .{
        .outputs = outputs,
        .completion_event = completion_event,
    };
}

fn mapBufferError(err: anyerror) Error {
    return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.UnsupportedElementType => error.UnsupportedElementType,
        error.ShapeMismatch => error.ShapeMismatch,
        error.InvalidArgument => error.InvalidArgument,
        error.UnsupportedRuntimeFeature => error.UnsupportedRuntimeFeature,
        error.BufferDeleted => error.BufferDeleted,
        error.BufferDonated => error.BufferDonated,
        error.BufferNotReady => error.BufferNotReady,
        error.BufferReadinessFailed => error.BufferReadinessFailed,
        else => error.Internal,
    };
}

const ExecutionCallTestSupport = struct {
    fn mlxMetalBackend() backend_api.Backend {
        return backend_api.create();
    }

    fn initMlxMetalClient() !*Client {
        return Client.init(std.testing.allocator, mlxMetalBackend(), 1);
    }

    fn shardingPlan(allocator: std.mem.Allocator, assignment: []const i32) !ir.ShardingPlan {
        return ExecutionShardingFixture.plan(allocator, assignment);
    }

    fn addU8ExecutablePlan(allocator: std.mem.Allocator, assignment: []const i32, dims: []const i64) !ir.ExecutablePlan {
        return ExecutionPlanFixture.addU8(allocator, assignment, dims);
    }

    fn constantU8ExecutablePlan(allocator: std.mem.Allocator, assignment: []const i32, literal: []const u8) !ir.ExecutablePlan {
        return ExecutionPlanFixture.constantU8(allocator, assignment, literal);
    }

    fn expectBufferBytes(buffer: *Buffer, expected: []const u8) !void {
        return ExecutionBufferExpect.bytes(buffer, expected);
    }
};

const ExecutionShardingFixture = struct {
    fn plan(allocator: std.mem.Allocator, assignment: []const i32) !ir.ShardingPlan {
        return .{
            .kind = .replicated,
            .mesh_name = try allocator.dupe(u8, ""),
            .device_assignment = try allocator.dupe(i32, assignment),
        };
    }
};

const ExecutionPlanFixture = struct {
    fn addU8(allocator: std.mem.Allocator, assignment: []const i32, dims: []const i64) !ir.ExecutablePlan {
        return AddU8ExecutionPlanFixture.create(allocator, assignment, dims);
    }

    fn constantU8(allocator: std.mem.Allocator, assignment: []const i32, literal: []const u8) !ir.ExecutablePlan {
        return ConstantU8ExecutionPlanFixture.create(allocator, assignment, literal);
    }
};

const AddU8ExecutionPlanFixture = struct {
    fn create(allocator: std.mem.Allocator, assignment: []const i32, dims: []const i64) !ir.ExecutablePlan {
        const values = try allocator.alloc(ir.Value, 3);
        errdefer allocator.free(values);

        for (values, 0..) |*value, i| {
            value.* = .{
                .id = .{ .index = @intCast(i) },
                .role = if (i < 2) .parameter else .instruction_result,
                .descriptor = .{
                    .element_type = .u8,
                    .dims = try allocator.dupe(i64, dims),
                    .device_id = assignment[0],
                    .memory_id = assignment[0],
                    .shard_index = 0,
                },
            };
        }

        var parameter_shardings = try allocator.alloc(ir.ShardingPlan, 2);
        errdefer allocator.free(parameter_shardings);
        parameter_shardings[0] = try ExecutionShardingFixture.plan(allocator, assignment);
        errdefer {
            allocator.free(parameter_shardings[0].mesh_name);
            allocator.free(parameter_shardings[0].device_assignment);
        }
        parameter_shardings[1] = try ExecutionShardingFixture.plan(allocator, assignment);
        errdefer {
            allocator.free(parameter_shardings[1].mesh_name);
            allocator.free(parameter_shardings[1].device_assignment);
        }

        var output_shardings = try allocator.alloc(ir.ShardingPlan, 1);
        errdefer allocator.free(output_shardings);
        output_shardings[0] = try ExecutionShardingFixture.plan(allocator, assignment);
        errdefer {
            allocator.free(output_shardings[0].mesh_name);
            allocator.free(output_shardings[0].device_assignment);
        }

        const output_ids = try allocator.dupe(ir.ValueId, &.{.{ .index = 2 }});
        errdefer allocator.free(output_ids);

        const inputs = try allocator.dupe(ir.ValueId, &.{ .{ .index = 0 }, .{ .index = 1 } });
        errdefer allocator.free(inputs);

        const outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 2 }});
        errdefer allocator.free(outputs);

        const instructions = try allocator.dupe(ir.PlanInstruction, &.{.{
            .kind = .add,
            .inputs = inputs,
            .outputs = outputs,
        }});
        errdefer allocator.free(instructions);

        return .{
            .allocator = allocator,
            .options = .{
                .num_replicas = 1,
                .num_partitions = 1,
                .device_assignment = try allocator.dupe(i32, assignment),
            },
            .values = values,
            .parameter_shardings = parameter_shardings,
            .output_shardings = output_shardings,
            .output_ids = output_ids,
            .instructions = instructions,
        };
    }
};

const ConstantU8ExecutionPlanFixture = struct {
    fn create(allocator: std.mem.Allocator, assignment: []const i32, literal: []const u8) !ir.ExecutablePlan {
        var values = try allocator.alloc(ir.Value, 1);
        errdefer allocator.free(values);

        const dims = [_]i64{@intCast(literal.len)};
        const owned_dims = try allocator.dupe(i64, &dims);
        errdefer allocator.free(owned_dims);
        values[0] = .{
            .id = .{ .index = 0 },
            .role = .instruction_result,
            .descriptor = .{
                .element_type = .u8,
                .dims = owned_dims,
                .device_id = assignment[0],
                .memory_id = assignment[0],
                .shard_index = 0,
            },
        };

        var output_shardings = try allocator.alloc(ir.ShardingPlan, 1);
        errdefer allocator.free(output_shardings);
        output_shardings[0] = try ExecutionShardingFixture.plan(allocator, assignment);
        errdefer {
            allocator.free(output_shardings[0].mesh_name);
            allocator.free(output_shardings[0].device_assignment);
        }

        const output_ids = try allocator.dupe(ir.ValueId, &.{.{ .index = 0 }});
        errdefer allocator.free(output_ids);

        const literal_copy = try allocator.dupe(u8, literal);
        errdefer allocator.free(literal_copy);

        const instruction_outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 0 }});
        errdefer allocator.free(instruction_outputs);

        const instructions = try allocator.dupe(ir.PlanInstruction, &.{.{
            .kind = .constant,
            .outputs = instruction_outputs,
            .literal = literal_copy,
        }});
        errdefer allocator.free(instructions);

        return .{
            .allocator = allocator,
            .options = .{
                .num_replicas = 1,
                .num_partitions = 1,
                .device_assignment = try allocator.dupe(i32, assignment),
            },
            .values = values,
            .parameter_shardings = try allocator.alloc(ir.ShardingPlan, 0),
            .output_shardings = output_shardings,
            .output_ids = output_ids,
            .instructions = instructions,
        };
    }
};

const ExecutionBufferExpect = struct {
    fn bytes(buffer: *Buffer, expected: []const u8) !void {
        const actual = try std.testing.allocator.alloc(u8, expected.len);
        defer std.testing.allocator.free(actual);
        try buffer.copyToHost(actual);
        try std.testing.expectEqualSlices(u8, expected, actual);
    }
};

test "compiled executable executes through runtime buffers" {
    const client = try ExecutionCallTestSupport.initMlxMetalClient();
    defer client.deinit();
    const allocator = std.testing.allocator;
    const assignment = [_]i32{0};
    const dims = [_]i64{4};

    var plan = try ExecutionCallTestSupport.addU8ExecutablePlan(allocator, &assignment, &dims);
    defer plan.deinit();

    var compiled = try CompiledExecutable.Testing.initBorrowedResident(allocator, client.executableContext(), &plan, .{});
    defer CompiledExecutable.Testing.deinitBorrowedResident(&compiled);
    try std.testing.expect(compiled.hasResidentBackendExecutable());
    try std.testing.expectEqual(@as(usize, 1), compiled.residentProgramInstructionCount());

    const lhs_data = [_]u8{ 1, 2, 3, 4 };
    const rhs_data = [_]u8{ 10, 20, 30, 40 };
    const lhs = try client.createHostBufferFromBytes(allocator, .u8, &dims, client.defaultDevice(), client.defaultMemory(), 0, &lhs_data);
    defer lhs.deinit();
    const rhs = try client.createHostBufferFromBytes(allocator, .u8, &dims, client.defaultDevice(), client.defaultMemory(), 0, &rhs_data);
    defer rhs.deinit();

    client.defaultMemory().stats.capacity_bytes = client.defaultMemory().stats.bytes_in_use + 4;
    const h2d_before_execute = client.defaultMemory().stats.host_to_device_bytes;
    const d2h_before_execute = client.defaultMemory().stats.device_to_host_bytes;
    const result = try execute(&compiled, allocator, client.executableContext(), 0, &.{ lhs, rhs });
    const outputs = result.outputs;
    defer allocator.free(outputs);
    defer {
        for (outputs) |output| output.deinit();
    }

    try std.testing.expectEqual(@as(usize, 1), outputs.len);
    try std.testing.expectEqual(@as(usize, 0), outputs[0].shardIndex());
    try std.testing.expect(outputs[0].hasBackendStorage());
    try std.testing.expectEqual(@as(u64, 4), compiled.executeCacheTrim().requested_bytes);
    try std.testing.expectEqual(@as(u64, 0), compiled.executeCacheTrim().freed_bytes);
    try std.testing.expect(!compiled.executeCacheTrim().still_over_capacity);
    try std.testing.expectEqual(backend_api.ExecutionCompletionKind.completed, compiled.backendCompletion().kind);
    try std.testing.expect(result.completion_event.isReady());
    try std.testing.expectEqual(h2d_before_execute, client.defaultMemory().stats.host_to_device_bytes);
    try std.testing.expectEqual(d2h_before_execute, client.defaultMemory().stats.device_to_host_bytes);
    try ExecutionCallTestSupport.expectBufferBytes(outputs[0], &.{ 11, 22, 33, 44 });
    try std.testing.expectEqual(d2h_before_execute + outputs[0].onDeviceSizeInBytes(), client.defaultMemory().stats.device_to_host_bytes);

    const wrong_shard = try client.createHostBufferFromBytes(allocator, .u8, &dims, client.defaultDevice(), client.defaultMemory(), 7, &lhs_data);
    defer wrong_shard.deinit();
    try std.testing.expectError(error.InvalidArgument, execute(&compiled, allocator, client.executableContext(), 0, &.{ wrong_shard, rhs }));

    Buffer.Testing.setReadiness(lhs, Event.pending());
    try std.testing.expectError(error.BufferNotReady, execute(&compiled, allocator, client.executableContext(), 0, &.{ lhs, rhs }));
    lhs.markReady();
    Buffer.Testing.setReadiness(rhs, Event.failed("argument upload failed"));
    try std.testing.expectError(error.BufferReadinessFailed, execute(&compiled, allocator, client.executableContext(), 0, &.{ lhs, rhs }));
    Buffer.Testing.setReadiness(rhs, Event.ready());

    Buffer.Testing.setDeviceId(lhs, 1234);
    try std.testing.expectError(error.InvalidArgument, execute(&compiled, allocator, client.executableContext(), 0, &.{ lhs, rhs }));
}

test "compiled executable evicts idle executable cache before output allocation" {
    const client = try ExecutionCallTestSupport.initMlxMetalClient();
    defer client.deinit();
    const allocator = std.testing.allocator;
    const assignment = [_]i32{0};
    const dims = [_]i64{4};

    var idle_plan = try ExecutionCallTestSupport.constantU8ExecutablePlan(allocator, &assignment, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer idle_plan.deinit();

    try std.testing.expect(!try client.recordExecutableCompile("execute-pressure-idle-constant"));
    var idle_executable = try CompiledExecutable.Testing.initBorrowedResident(allocator, client.executableContext(), &idle_plan, .{
        .cache_fingerprint = "execute-pressure-idle-constant",
    });
    var idle_entry = Client.Testing.executableCacheEntrySnapshot(client, "execute-pressure-idle-constant") orelse return error.TestUnexpectedResult;
    const idle_resident_bytes = idle_entry.resident_bytes;
    try std.testing.expect(idle_resident_bytes >= 8);
    CompiledExecutable.Testing.deinitBorrowedResident(&idle_executable);
    idle_entry = Client.Testing.executableCacheEntrySnapshot(client, "execute-pressure-idle-constant") orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(usize, 0), idle_entry.reference_count);
    try std.testing.expect(idle_entry.resident);

    var add_plan = try ExecutionCallTestSupport.addU8ExecutablePlan(allocator, &assignment, &dims);
    defer add_plan.deinit();
    var add_compiled = try CompiledExecutable.Testing.initBorrowedResident(allocator, client.executableContext(), &add_plan, .{});
    defer CompiledExecutable.Testing.deinitBorrowedResident(&add_compiled);

    const lhs_data = [_]u8{ 1, 2, 3, 4 };
    const rhs_data = [_]u8{ 10, 20, 30, 40 };
    const lhs = try client.createHostBufferFromBytes(allocator, .u8, &dims, client.defaultDevice(), client.defaultMemory(), 0, &lhs_data);
    defer lhs.deinit();
    const rhs = try client.createHostBufferFromBytes(allocator, .u8, &dims, client.defaultDevice(), client.defaultMemory(), 0, &rhs_data);
    defer rhs.deinit();

    const output_bytes = try add_compiled.outputByteSize();
    client.defaultMemory().stats.capacity_bytes = client.defaultMemory().stats.bytes_in_use + output_bytes;

    const result = try execute(&add_compiled, allocator, client.executableContext(), 0, &.{ lhs, rhs });
    const outputs = result.outputs;
    defer allocator.free(outputs);
    defer {
        for (outputs) |output| output.deinit();
    }

    try std.testing.expectEqual(@as(usize, 1), outputs.len);
    try std.testing.expect(result.completion_event.isReady());
    try ExecutionCallTestSupport.expectBufferBytes(outputs[0], &.{ 11, 22, 33, 44 });
    try std.testing.expectEqual(@as(u64, @intCast(output_bytes)), add_compiled.executeCacheTrim().requested_bytes);
    try std.testing.expectEqual(@as(u64, 0), add_compiled.executeCacheTrim().target_resident_bytes);
    try std.testing.expectEqual(idle_resident_bytes, add_compiled.executeCacheTrim().freed_bytes);
    try std.testing.expectEqual(@as(u64, 1), add_compiled.executeCacheTrim().evicted_entries);
    try std.testing.expectEqual(@as(u64, 0), add_compiled.executeCacheTrim().remaining_resident_bytes);
    try std.testing.expect(!add_compiled.executeCacheTrim().still_over_capacity);
    idle_entry = Client.Testing.executableCacheEntrySnapshot(client, "execute-pressure-idle-constant") orelse return error.TestUnexpectedResult;
    try std.testing.expect(!idle_entry.resident);
    try std.testing.expectEqual(@as(u64, 1), client.executableCacheStats().pressure_trim_requests);
    try std.testing.expectEqual(idle_resident_bytes, client.executableCacheStats().pressure_trimmed_bytes);
    try std.testing.expectEqual(@as(u64, 0), client.executableCacheStats().pressure_trim_failures);
}
