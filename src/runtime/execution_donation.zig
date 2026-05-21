const std = @import("std");
const backend_api = @import("src/backend/mlx_metal");
const ir = @import("src/compiler/ir");

const buffer_mod = @import("buffer.zig");
const client_mod = @import("client.zig");
const executable_mod = @import("executable.zig");

const Buffer = buffer_mod.Buffer;
const BufferState = buffer_mod.BufferState;
const Client = client_mod.Client;
const CompiledExecutable = executable_mod.CompiledExecutable;
const ExecutionContext = executable_mod.Context;

/// Tracks executable statistics that must be rolled back if output wrapping fails.
pub const Delta = struct {
    output_count: usize = 0,
    output_bytes: usize = 0,
};

/// Adopts a backend output handle, transferring donated input storage when required.
pub fn takeOutputHandle(
    context: ExecutionContext,
    executable: *CompiledExecutable,
    output_index: usize,
    output: backend_api.ExecutableOutput,
    arguments: []const *Buffer,
    delta: *Delta,
) !backend_api.BufferHandle {
    const alias = executable.donatedParameterAliasForOutput(output_index) orelse return output.handle;
    if (alias.parameter_index >= arguments.len) return error.Internal;

    const donated_argument = arguments[alias.parameter_index];
    if (donated_argument.elementType() != output.element_type or !std.mem.eql(i64, donated_argument.dimensions(), output.dims)) {
        return error.Internal;
    }

    const donated_handle = donated_argument.backendHandleForDispatch() orelse return error.Internal;
    if (alias.kind == .donation and output.handle != donated_handle) return output.handle;
    if (output.handle != donated_handle) context.destroyBackendBuffer(output.handle);

    const transferred = try donated_argument.takeBackendStorageForDonationAlias();
    delta.output_count += 1;
    delta.output_bytes += output.byte_size;
    executable.recordDonationAlias(output.byte_size);
    return transferred;
}

/// Rolls back executable donation-alias counters after a wrapping failure.
pub fn rollback(executable: *CompiledExecutable, delta: Delta) void {
    executable.rollbackDonationAlias(delta.output_count, delta.output_bytes);
}

const DonationTestSupport = struct {
    fn mlxMetalBackend() backend_api.Backend {
        return backend_api.create();
    }

    fn initMlxMetalClient() !*Client {
        return Client.init(std.testing.allocator, mlxMetalBackend(), 1);
    }

    fn shardingPlan(allocator: std.mem.Allocator, assignment: []const i32) !ir.ShardingPlan {
        return .{
            .kind = .replicated,
            .mesh_name = try allocator.dupe(u8, ""),
            .device_assignment = try allocator.dupe(i32, assignment),
        };
    }

    fn expectBufferBytes(buffer: *Buffer, expected: []const u8) !void {
        const actual = try std.testing.allocator.alloc(u8, expected.len);
        defer std.testing.allocator.free(actual);
        try buffer.copyToHost(actual);
        try std.testing.expectEqualSlices(u8, expected, actual);
    }
};

test "compiled executable transfers donated parameter alias outputs without copying" {
    const client = try DonationTestSupport.initMlxMetalClient();
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
    parameter_shardings[0] = try DonationTestSupport.shardingPlan(allocator, &assignment);
    var output_shardings = try allocator.alloc(ir.ShardingPlan, 1);
    output_shardings[0] = try DonationTestSupport.shardingPlan(allocator, &assignment);

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

    var compiled = try CompiledExecutable.Testing.initBorrowedResident(allocator, client.executableContext(), &plan, .{});
    defer CompiledExecutable.Testing.deinitBorrowedResident(&compiled);

    const data = [_]u8{ 9, 8, 7, 6 };
    const input = try client.createHostBufferFromBytes(allocator, .u8, &dims, client.defaultDevice(), client.defaultMemory(), 0, &data);
    defer input.deinit();
    const before_execute_bytes = client.defaultMemory().stats.bytes_in_use;
    const before_execute_handle = input.backendHandleForDispatch().?;

    var delta = Delta{};
    const transferred = try takeOutputHandle(
        client.executableContext(),
        &compiled,
        0,
        .{
            .handle = before_execute_handle,
            .element_type = .u8,
            .dims = &dims,
            .byte_size = data.len,
        },
        &.{input},
        &delta,
    );

    const output = try Buffer.initBackendHandle(
        allocator,
        client.executableContext().bufferStorageBackend(),
        .u8,
        &dims,
        client.defaultDevice(),
        client.defaultMemory(),
        0,
        data.len,
        transferred,
    );
    defer output.deinit();

    try std.testing.expectEqual(before_execute_handle, output.backendHandleForDispatch());
    try std.testing.expectEqual(@as(?backend_api.BufferHandle, null), input.backendHandleForDispatch());
    input.markDonated();
    try std.testing.expectEqual(BufferState.donated, input.lifecycleState());
    try std.testing.expect(output.hasBackendStorage());
    try DonationTestSupport.expectBufferBytes(output, &data);
    try std.testing.expectEqual(before_execute_bytes, client.defaultMemory().stats.bytes_in_use);
    try std.testing.expectEqual(@as(usize, 1), delta.output_count);
    try std.testing.expectEqual(@as(usize, data.len), delta.output_bytes);

    const stats = compiled.backendExecutableStats().?;
    try std.testing.expectEqual(@as(usize, 1), stats.donation_alias_output_count);
    try std.testing.expectEqual(@as(usize, data.len), stats.donation_alias_output_bytes);
}
