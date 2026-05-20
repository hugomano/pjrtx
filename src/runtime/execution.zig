const std = @import("std");
const backend_api = @import("src/backend/mlx_metal");
const ir = @import("src/compiler/ir");

const buffer_mod = @import("buffer.zig");
const client_mod = @import("client.zig");
const device_memory = @import("device_memory.zig");
const event_mod = @import("event.zig");
const executable_mod = @import("executable.zig");

const Buffer = buffer_mod.Buffer;
const BufferState = buffer_mod.BufferState;
const Client = client_mod.Client;
const Event = event_mod.Event;
const ExecutableGraph = executable_mod.ExecutableGraph;
const ExecutionContext = executable_mod.Context;
const Memory = device_memory.Memory;

/// Result of one per-device executable dispatch.
pub const GraphExecuteResult = struct {
    outputs: []*Buffer,
    completion_event: Event = Event.ready(),
};

/// Errors produced while validating and dispatching a compiled graph.
pub const GraphExecuteError = error{
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

/// Executes one device slice of a resident executable graph.
pub fn executeDevice(
    graph: *ExecutableGraph,
    allocator: std.mem.Allocator,
    context: ExecutionContext,
    plan: *const ir.ExecutablePlan,
    device_index: usize,
    arguments: []const *Buffer,
) GraphExecuteError!GraphExecuteResult {
    if (device_index >= graph.device_ids.len or device_index >= context.devices.len) return error.InvalidArgument;
    if (arguments.len != plan.parameter_shardings.len) return error.InvalidArgument;
    for (arguments) |argument| {
        argument.ensureUsable() catch |err| return mapBufferError(err);
        argument.ensureReady() catch |err| return mapBufferError(err);
        if (!argumentMatchesDevice(graph, device_index, argument)) return error.InvalidArgument;
    }
    if (tryExecuteBackendExecutable(graph, allocator, context, plan, device_index, arguments) catch |err| return err) |result| {
        return result;
    }
    return error.UnsupportedRuntimeFeature;
}

fn argumentMatchesDevice(graph: *const ExecutableGraph, device_index: usize, argument: *const Buffer) bool {
    if (device_index >= graph.device_ids.len) return false;
    if (argument.device_id != graph.device_ids[device_index]) return false;
    if (argument.shard_index != device_index) return false;
    return true;
}

fn tryExecuteBackendExecutable(
    graph: *ExecutableGraph,
    allocator: std.mem.Allocator,
    context: ExecutionContext,
    plan: *const ir.ExecutablePlan,
    device_index: usize,
    arguments: []const *Buffer,
) GraphExecuteError!?GraphExecuteResult {
    const backend_executable = graph.backend_executable orelse return null;
    var argument_handles = try allocator.alloc(backend_api.BufferHandle, arguments.len);
    defer allocator.free(argument_handles);
    for (arguments, 0..) |argument, i| {
        argument.ensureUsable() catch |err| return mapBufferError(err);
        argument.ensureReady() catch |err| return mapBufferError(err);
        argument_handles[i] = argument.backend_buffer orelse return null;
    }

    const device = &context.devices[device_index];
    const memory = device.default_memory;
    graph.last_execute_cache_trim = context.trimExecutableCacheForAllocation(memory, planOutputBytes(plan) catch return error.Internal);

    const backend_result = context.backend.executeExecutable(allocator, backend_executable, device_index, argument_handles) catch |err| return mapBufferError(err);
    const owned_backend_result = backend_result orelse return null;
    const owned_backend_outputs = owned_backend_result.outputs;
    graph.last_backend_completion = owned_backend_result.completion;
    const completion_event = runtimeEventFromBackendCompletion(context.backend, owned_backend_result.completion);
    defer allocator.free(owned_backend_outputs);

    var wrapped_backend_outputs: usize = 0;
    errdefer {
        for (owned_backend_outputs[wrapped_backend_outputs..]) |output| context.backend.destroyBuffer(output.handle);
    }

    if (owned_backend_outputs.len != plan.output_ids.len) return error.Internal;
    for (owned_backend_outputs, 0..) |output, output_index| {
        if (!backendOutputMatchesPlan(plan, output_index, output)) return error.Internal;
    }

    const outputs = try allocator.alloc(*Buffer, owned_backend_outputs.len);
    errdefer allocator.free(outputs);
    var initialized: usize = 0;
    errdefer {
        for (outputs[0..initialized]) |buffer| buffer.deinit();
    }

    var donation_alias_delta: executable_mod.DonationAliasStats = .{};
    errdefer graph.donation_alias_stats.output_count -|= donation_alias_delta.output_count;
    errdefer graph.donation_alias_stats.output_bytes -|= donation_alias_delta.output_bytes;

    for (owned_backend_outputs, 0..) |output, i| {
        const alias = donatedParameterAliasForOutput(plan, i);
        const backend_handle = if (alias) |alias_info| blk: {
            if (alias_info.parameter_index >= arguments.len) return error.Internal;
            const donated_argument = arguments[alias_info.parameter_index];
            if (donated_argument.element_type != output.element_type or !std.mem.eql(i64, donated_argument.dims, output.dims)) return error.Internal;
            const donated_handle = donated_argument.backend_buffer orelse return error.Internal;
            if (alias_info.kind == .donation and output.handle != donated_handle) break :blk output.handle;
            if (output.handle != donated_handle) context.backend.destroyBuffer(output.handle);
            const transferred = donated_argument.takeBackendStorageForDonationAlias() catch |err| return mapBufferError(err);
            donation_alias_delta.output_count += 1;
            donation_alias_delta.output_bytes += output.byte_size;
            graph.donation_alias_stats.output_count += 1;
            graph.donation_alias_stats.output_bytes += output.byte_size;
            break :blk transferred;
        } else output.handle;

        outputs[i] = Buffer.initBackendHandle(
            allocator,
            context.backend,
            output.element_type,
            output.dims,
            device,
            memory,
            device_index,
            output.byte_size,
            backend_handle,
        ) catch |err| {
            if (alias != null) context.backend.destroyBuffer(backend_handle);
            return mapBufferError(err);
        };
        initialized += 1;
        wrapped_backend_outputs = initialized;
    }
    return .{
        .outputs = outputs,
        .completion_event = completion_event,
    };
}

fn runtimeEventFromBackendCompletion(backend_impl: backend_api.Backend, completion: backend_api.ExecutionCompletion) Event {
    return switch (completion.kind) {
        .completed => Event.ready(),
        .pending => blk: {
            const backend_event = completion.backend_event orelse break :blk Event.failed("backend returned asynchronous completion without an event handle");
            defer backend_impl.destroyExecutionEvent(backend_event);
            const status = backend_impl.executionEventStatus(backend_event) catch break :blk Event.failed("backend execution event status query failed");
            break :blk switch (status.state) {
                .ready => Event.ready(),
                .failed => Event.failed(if (status.message.len == 0) "backend execution event failed" else status.message),
                .pending => Event.failed("backend execution event is pending without runtime scheduler integration"),
            };
        },
    };
}

fn backendOutputMatchesPlan(plan: *const ir.ExecutablePlan, output_index: usize, output: backend_api.ExecutableOutput) bool {
    if (output_index >= plan.output_ids.len) return false;
    const value_id = plan.output_ids[output_index];
    if (value_id.index >= plan.values.len) return false;
    const descriptor = plan.values[value_id.index].descriptor;
    if (output.element_type != descriptor.element_type) return false;
    if (!std.mem.eql(i64, output.dims, descriptor.dims)) return false;
    if (output.byte_size != ir.denseByteSize(descriptor.element_type, descriptor.dims)) return false;
    return true;
}

const DonationAlias = struct {
    parameter_index: usize,
    kind: ir.OutputAliasKind,
};

fn donatedParameterAliasForOutput(plan: *const ir.ExecutablePlan, output_index: usize) ?DonationAlias {
    for (plan.output_aliases) |alias| {
        if (alias.output_index == output_index and planDonatesParameter(plan, alias.parameter_index)) {
            return .{ .parameter_index = alias.parameter_index, .kind = alias.kind };
        }
    }
    if (output_index >= plan.output_ids.len) return null;
    const output_id = plan.output_ids[output_index];
    var parameter_index: usize = 0;
    for (plan.values) |value| {
        if (value.role != .parameter) continue;
        if (value.id.index == output_id.index) {
            return if (planDonatesParameter(plan, parameter_index)) .{ .parameter_index = parameter_index, .kind = .identity } else null;
        }
        parameter_index += 1;
    }
    return null;
}

fn planDonatesParameter(plan: *const ir.ExecutablePlan, parameter_index: usize) bool {
    for (plan.donated_parameter_indices) |candidate| {
        if (candidate == parameter_index) return true;
    }
    return false;
}

fn planOutputBytes(plan: *const ir.ExecutablePlan) !usize {
    var total: usize = 0;
    for (plan.output_ids) |value_id| {
        if (value_id.index >= plan.values.len) return error.InvalidGraph;
        const descriptor = plan.values[value_id.index].descriptor;
        total = try std.math.add(usize, total, ir.denseByteSize(descriptor.element_type, descriptor.dims));
    }
    return total;
}

fn mapBufferError(err: anyerror) GraphExecuteError {
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

fn mlxMetalBackendForTest() backend_api.Backend {
    return backend_api.create();
}

fn initMlxMetalClientForTest() !*Client {
    return Client.init(std.testing.allocator, mlxMetalBackendForTest(), 1);
}

fn testShardingPlan(allocator: std.mem.Allocator, assignment: []const i32) !ir.ShardingPlan {
    return .{
        .kind = .replicated,
        .mesh_name = try allocator.dupe(u8, ""),
        .device_assignment = try allocator.dupe(i32, assignment),
    };
}

fn addU8ExecutablePlanForTest(allocator: std.mem.Allocator, assignment: []const i32, dims: []const i64) !ir.ExecutablePlan {
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
    parameter_shardings[0] = try testShardingPlan(allocator, assignment);
    errdefer {
        allocator.free(parameter_shardings[0].mesh_name);
        allocator.free(parameter_shardings[0].device_assignment);
    }
    parameter_shardings[1] = try testShardingPlan(allocator, assignment);
    errdefer {
        allocator.free(parameter_shardings[1].mesh_name);
        allocator.free(parameter_shardings[1].device_assignment);
    }

    var output_shardings = try allocator.alloc(ir.ShardingPlan, 1);
    errdefer allocator.free(output_shardings);
    output_shardings[0] = try testShardingPlan(allocator, assignment);
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

fn constantU8ExecutablePlanForTest(allocator: std.mem.Allocator, assignment: []const i32, literal: []const u8) !ir.ExecutablePlan {
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
    output_shardings[0] = try testShardingPlan(allocator, assignment);
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

fn expectBufferBytes(buffer: *Buffer, expected: []const u8) !void {
    const actual = try std.testing.allocator.alloc(u8, expected.len);
    defer std.testing.allocator.free(actual);
    try buffer.copyToHost(actual);
    try std.testing.expectEqualSlices(u8, expected, actual);
}

test "backend output descriptor validation matches executable plan outputs" {
    const allocator = std.testing.allocator;
    const assignment = [_]i32{0};
    const dims = [_]i64{ 2, 2 };

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

    var output_shardings = try allocator.alloc(ir.ShardingPlan, 1);
    output_shardings[0] = try testShardingPlan(allocator, &assignment);

    var plan = ir.ExecutablePlan{
        .allocator = allocator,
        .options = .{
            .num_replicas = 1,
            .num_partitions = 1,
            .device_assignment = try allocator.dupe(i32, &assignment),
        },
        .values = values,
        .parameter_shardings = try allocator.alloc(ir.ShardingPlan, 0),
        .output_shardings = output_shardings,
        .output_ids = try allocator.dupe(ir.ValueId, &.{.{ .index = 0 }}),
        .instructions = try allocator.alloc(ir.PlanInstruction, 0),
    };
    defer plan.deinit();

    const handle: backend_api.BufferHandle = @ptrFromInt(0x1000);
    try std.testing.expect(backendOutputMatchesPlan(&plan, 0, .{
        .handle = handle,
        .element_type = .f32,
        .dims = &dims,
        .byte_size = 16,
    }));
    try std.testing.expect(!backendOutputMatchesPlan(&plan, 0, .{
        .handle = handle,
        .element_type = .u32,
        .dims = &dims,
        .byte_size = 16,
    }));
    try std.testing.expect(!backendOutputMatchesPlan(&plan, 0, .{
        .handle = handle,
        .element_type = .f32,
        .dims = &.{4},
        .byte_size = 16,
    }));
    try std.testing.expect(!backendOutputMatchesPlan(&plan, 0, .{
        .handle = handle,
        .element_type = .f32,
        .dims = &dims,
        .byte_size = 12,
    }));
    try std.testing.expect(!backendOutputMatchesPlan(&plan, 1, .{
        .handle = handle,
        .element_type = .f32,
        .dims = &dims,
        .byte_size = 16,
    }));
}

test "executable graph executes through runtime buffers" {
    const client = try initMlxMetalClientForTest();
    defer client.deinit();
    const allocator = std.testing.allocator;
    const assignment = [_]i32{0};
    const dims = [_]i64{4};

    var plan = try addU8ExecutablePlanForTest(allocator, &assignment, &dims);
    defer plan.deinit();

    var graph = try ExecutableGraph.init(allocator, client.executableContext(), &plan);
    defer graph.deinit();
    try std.testing.expect(graph.backend_executable != null);
    try std.testing.expectEqual(true, graph.lowering.backend_executable_ready);
    try std.testing.expectEqual(@as(usize, 1), graph.lowering.lowered_instruction_count);

    const lhs_data = [_]u8{ 1, 2, 3, 4 };
    const rhs_data = [_]u8{ 10, 20, 30, 40 };
    const lhs = try Buffer.initHostCopyForBackend(allocator, client.backend, .u8, &dims, &client.devices[0], &client.memories[0], 0, &lhs_data);
    defer lhs.deinit();
    const rhs = try Buffer.initHostCopyForBackend(allocator, client.backend, .u8, &dims, &client.devices[0], &client.memories[0], 0, &rhs_data);
    defer rhs.deinit();

    client.memories[0].stats.capacity_bytes = client.memories[0].stats.bytes_in_use + 4;
    const h2d_before_execute = client.memories[0].stats.host_to_device_bytes;
    const d2h_before_execute = client.memories[0].stats.device_to_host_bytes;
    const result = try executeDevice(&graph, allocator, client.executableContext(), &plan, 0, &.{ lhs, rhs });
    const outputs = result.outputs;
    defer allocator.free(outputs);
    defer {
        for (outputs) |output| output.deinit();
    }

    try std.testing.expectEqual(@as(usize, 1), outputs.len);
    try std.testing.expectEqual(@as(usize, 0), outputs[0].shard_index);
    try std.testing.expect(outputs[0].backend_buffer != null);
    try std.testing.expectEqual(@as(usize, 0), outputs[0].bytes.len);
    try std.testing.expectEqual(@as(u64, 4), graph.last_execute_cache_trim.requested_bytes);
    try std.testing.expectEqual(@as(u64, 0), graph.last_execute_cache_trim.freed_bytes);
    try std.testing.expect(!graph.last_execute_cache_trim.still_over_capacity);
    try std.testing.expectEqual(backend_api.ExecutionCompletionKind.completed, graph.last_backend_completion.kind);
    try std.testing.expect(result.completion_event.isReady());
    try std.testing.expectEqual(h2d_before_execute, client.memories[0].stats.host_to_device_bytes);
    try std.testing.expectEqual(d2h_before_execute, client.memories[0].stats.device_to_host_bytes);
    try expectBufferBytes(outputs[0], &.{ 11, 22, 33, 44 });
    try std.testing.expectEqual(d2h_before_execute + outputs[0].byte_size, client.memories[0].stats.device_to_host_bytes);

    const wrong_shard = try Buffer.initHostCopyForBackend(allocator, client.backend, .u8, &dims, &client.devices[0], &client.memories[0], 7, &lhs_data);
    defer wrong_shard.deinit();
    try std.testing.expectError(error.InvalidArgument, executeDevice(&graph, allocator, client.executableContext(), &plan, 0, &.{ wrong_shard, rhs }));

    lhs.ready_event = Event.pending();
    try std.testing.expectError(error.BufferNotReady, executeDevice(&graph, allocator, client.executableContext(), &plan, 0, &.{ lhs, rhs }));
    lhs.ready_event.setReady();
    rhs.ready_event = Event.failed("argument upload failed");
    try std.testing.expectError(error.BufferReadinessFailed, executeDevice(&graph, allocator, client.executableContext(), &plan, 0, &.{ lhs, rhs }));
    rhs.ready_event = Event.ready();

    lhs.device_id = 1234;
    try std.testing.expectError(error.InvalidArgument, executeDevice(&graph, allocator, client.executableContext(), &plan, 0, &.{ lhs, rhs }));
}

test "executable graph transfers donated parameter alias outputs without copying" {
    const client = try initMlxMetalClientForTest();
    defer client.deinit();
    const allocator = std.testing.allocator;
    const assignment = [_]i32{0};
    const dims = [_]i64{4};

    const values = try allocator.alloc(ir.Value, 1);
    errdefer allocator.free(values);
    values[0] = .{
        .id = .{ .index = 0 },
        .role = .parameter,
        .descriptor = .{
            .element_type = .u8,
            .dims = try allocator.dupe(i64, &dims),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };

    var parameter_shardings = try allocator.alloc(ir.ShardingPlan, 1);
    parameter_shardings[0] = try testShardingPlan(allocator, &assignment);
    var output_shardings = try allocator.alloc(ir.ShardingPlan, 1);
    output_shardings[0] = try testShardingPlan(allocator, &assignment);

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
        .donated_parameter_indices = try allocator.dupe(u32, &.{0}),
        .instructions = try allocator.alloc(ir.PlanInstruction, 0),
    };
    defer plan.deinit();

    var graph = try ExecutableGraph.init(allocator, client.executableContext(), &plan);
    defer graph.deinit();

    const data = [_]u8{ 9, 8, 7, 6 };
    const input = try Buffer.initHostCopyForBackend(allocator, client.backend, .u8, &dims, &client.devices[0], &client.memories[0], 0, &data);
    defer input.deinit();
    const before_execute_bytes = client.memories[0].stats.bytes_in_use;
    const before_execute_handle = input.backend_buffer;

    const result = try executeDevice(&graph, allocator, client.executableContext(), &plan, 0, &.{input});
    defer allocator.free(result.outputs);
    defer {
        for (result.outputs) |output| output.deinit();
    }

    try std.testing.expectEqual(@as(usize, 1), result.outputs.len);
    try std.testing.expectEqual(before_execute_handle, result.outputs[0].backend_buffer);
    try std.testing.expectEqual(@as(?backend_api.BufferHandle, null), input.backend_buffer);
    input.markDonated();
    try std.testing.expectEqual(BufferState.donated, input.state);
    try std.testing.expect(result.outputs[0].hasBackendStorage());
    try expectBufferBytes(result.outputs[0], &data);
    try std.testing.expectEqual(before_execute_bytes, client.memories[0].stats.bytes_in_use);

    const stats = graph.backendExecutableStats().?;
    try std.testing.expectEqual(@as(usize, 1), stats.donation_alias_output_count);
    try std.testing.expectEqual(@as(usize, data.len), stats.donation_alias_output_bytes);
}

test "backend pending completion without event handle fails closed" {
    const event = runtimeEventFromBackendCompletion(mlxMetalBackendForTest(), .{ .kind = .pending });
    try std.testing.expect(event.isReady());
    try std.testing.expectError(error.EventFailed, event.awaitReady());
}
test "executable graph evicts idle executable cache before output allocation" {
    const client = try initMlxMetalClientForTest();
    defer client.deinit();
    const allocator = std.testing.allocator;
    const assignment = [_]i32{0};
    const dims = [_]i64{4};

    var idle_plan = try constantU8ExecutablePlanForTest(allocator, &assignment, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer idle_plan.deinit();

    try std.testing.expect(!try client.recordExecutableCompile("execute-pressure-idle-constant"));
    var idle_graph = try ExecutableGraph.initWithOptions(allocator, client.executableContext(), &idle_plan, .{
        .cache_fingerprint = "execute-pressure-idle-constant",
    });
    const idle_entry = client.executable_cache.get("execute-pressure-idle-constant") orelse return error.TestUnexpectedResult;
    const idle_resident_bytes = idle_entry.resident_bytes;
    try std.testing.expect(idle_resident_bytes >= 8);
    idle_graph.deinit();
    try std.testing.expectEqual(@as(usize, 0), idle_entry.ref_count);
    try std.testing.expect(idle_entry.backend_executable != null);

    var add_plan = try addU8ExecutablePlanForTest(allocator, &assignment, &dims);
    defer add_plan.deinit();
    var add_graph = try ExecutableGraph.init(allocator, client.executableContext(), &add_plan);
    defer add_graph.deinit();

    const lhs_data = [_]u8{ 1, 2, 3, 4 };
    const rhs_data = [_]u8{ 10, 20, 30, 40 };
    const lhs = try Buffer.initHostCopyForBackend(allocator, client.backend, .u8, &dims, &client.devices[0], &client.memories[0], 0, &lhs_data);
    defer lhs.deinit();
    const rhs = try Buffer.initHostCopyForBackend(allocator, client.backend, .u8, &dims, &client.devices[0], &client.memories[0], 0, &rhs_data);
    defer rhs.deinit();

    const output_bytes = try planOutputBytes(&add_plan);
    client.memories[0].stats.capacity_bytes = client.memories[0].stats.bytes_in_use + output_bytes;

    const result = try executeDevice(&add_graph, allocator, client.executableContext(), &add_plan, 0, &.{ lhs, rhs });
    const outputs = result.outputs;
    defer allocator.free(outputs);
    defer {
        for (outputs) |output| output.deinit();
    }

    try std.testing.expectEqual(@as(usize, 1), outputs.len);
    try std.testing.expect(result.completion_event.isReady());
    try expectBufferBytes(outputs[0], &.{ 11, 22, 33, 44 });
    try std.testing.expectEqual(@as(u64, @intCast(output_bytes)), add_graph.last_execute_cache_trim.requested_bytes);
    try std.testing.expectEqual(@as(u64, 0), add_graph.last_execute_cache_trim.target_resident_bytes);
    try std.testing.expectEqual(idle_resident_bytes, add_graph.last_execute_cache_trim.freed_bytes);
    try std.testing.expectEqual(@as(u64, 1), add_graph.last_execute_cache_trim.evicted_entries);
    try std.testing.expectEqual(@as(u64, 0), add_graph.last_execute_cache_trim.remaining_resident_bytes);
    try std.testing.expect(!add_graph.last_execute_cache_trim.still_over_capacity);
    try std.testing.expect(idle_entry.backend_executable == null);
    try std.testing.expectEqual(@as(u64, 1), client.executable_cache.stats.pressure_trim_requests);
    try std.testing.expectEqual(idle_resident_bytes, client.executable_cache.stats.pressure_trimmed_bytes);
    try std.testing.expectEqual(@as(u64, 0), client.executable_cache.stats.pressure_trim_failures);
}
