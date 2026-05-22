const std = @import("std");
const backend_api = @import("backend_selection.zig");
const ir = @import("src/compiler/ir");

const buffer_mod = @import("buffer.zig");
const device_memory = @import("device_memory.zig");
const event_mod = @import("event.zig");

const Buffer = buffer_mod.Buffer;
const BufferState = buffer_mod.BufferState;
const BufferType = ir.BufferType;
const Device = device_memory.Device;
const DeviceMemoryTopology = device_memory.DeviceMemoryTopology;
const Event = event_mod.Event;
const Memory = device_memory.Memory;

const BufferTestContext = struct {
    allocator: std.mem.Allocator,
    backend: backend_api.Backend,
    descriptors: []ir.DeviceDescriptor,
    topology: DeviceMemoryTopology,

    fn init(allocator: std.mem.Allocator) !BufferTestContext {
        const backend = backend_api.create();
        const descriptors = try backend.enumerateDevices(allocator, 1);
        errdefer backend.releaseDeviceDescriptors(allocator, descriptors);
        if (descriptors.len == 0) return error.InvalidDeviceCount;

        const topology = try DeviceMemoryTopology.initFromDescriptors(allocator, descriptors);
        errdefer topology.deinit(allocator);

        return .{
            .allocator = allocator,
            .backend = backend,
            .descriptors = descriptors,
            .topology = topology,
        };
    }

    fn deinit(self: *BufferTestContext) void {
        self.topology.deinit(self.allocator);
        self.backend.releaseDeviceDescriptors(self.allocator, self.descriptors);
        self.* = undefined;
    }

    fn device(self: *BufferTestContext) *Device {
        return self.topology.defaultDevice();
    }

    fn memory(self: *BufferTestContext) *Memory {
        return self.topology.defaultMemory();
    }

    fn hostCopy(self: *BufferTestContext, element_type: BufferType, dims: []const i64, shard_index: usize, src: []const u8) !*Buffer {
        return Buffer.initHostCopyForBackend(self.allocator, self.backend, element_type, dims, self.device(), self.memory(), shard_index, src);
    }
};

fn expectBufferBytes(buffer: *Buffer, expected: []const u8) !void {
    const actual = try std.testing.allocator.alloc(u8, expected.len);
    defer std.testing.allocator.free(actual);
    try buffer.copyToHost(actual);
    try std.testing.expectEqualSlices(u8, expected, actual);
}

test "buffer keeps shard/device/memory ownership metadata" {
    var ctx = try BufferTestContext.init(std.testing.allocator);
    defer ctx.deinit();

    const dims = [_]i64{ 2, 2 };
    const data = [_]u8{ 1, 2, 3, 4 };
    const buffer = try ctx.hostCopy(.u8, &dims, 0, &data);
    defer buffer.deinit();

    try std.testing.expect(buffer.matchesExecutionSlot(0, 0));
    try std.testing.expectEqual(@as(i32, 0), buffer.devicePlacement().id);
    try std.testing.expectEqual(@as(i32, 0), buffer.memoryPlacement().id);
    try std.testing.expectEqual(@as(usize, 0), buffer.shardIndex());
    try expectBufferBytes(buffer, &data);
}

test "buffer constructors reject memory not addressable by device" {
    var ctx = try BufferTestContext.init(std.testing.allocator);
    defer ctx.deinit();

    var invalid_device_ids = [_]i32{1234};
    var no_devices = [_]*Device{};
    var invalid_memory = Memory{
        .id = 99,
        .kind = .device,
        .debug_string = "unreachable test memory",
        .addressable_device_ids = invalid_device_ids[0..],
        .addressable_devices = no_devices[0..],
    };

    const dims = [_]i64{4};
    const data = [_]u8{ 1, 2, 3, 4 };
    try std.testing.expect(!invalid_memory.isAddressableBy(ctx.device()));
    try std.testing.expectError(error.InvalidArgument, Buffer.initHostCopyForBackend(std.testing.allocator, ctx.backend, .u8, &dims, ctx.device(), &invalid_memory, 0, &data));

    if (try ctx.backend.bufferFromHost(ctx.device().local_hardware_id, .u8, &dims, &data)) |backend_buffer| {
        try std.testing.expectError(error.InvalidArgument, Buffer.initBackendHandle(std.testing.allocator, ctx.backend, .u8, &dims, ctx.device(), &invalid_memory, 0, data.len, backend_buffer));
        ctx.backend.destroyBuffer(backend_buffer);
    }
}

test "buffer lifecycle rejects deleted and donated buffers" {
    var ctx = try BufferTestContext.init(std.testing.allocator);
    defer ctx.deinit();

    const dims = [_]i64{4};
    const data = [_]u8{ 1, 2, 3, 4 };
    const before_deleted = ctx.memory().stats.bytes_in_use;
    const deleted = try ctx.hostCopy(.u8, &dims, 0, &data);
    defer deleted.deinit();
    try std.testing.expectEqual(before_deleted + data.len, ctx.memory().stats.bytes_in_use);
    try std.testing.expect(deleted.hasBackendStorage());
    deleted.markDeleted();
    try std.testing.expectEqual(BufferState.deleted, deleted.lifecycleState());
    try std.testing.expect(deleted.isDeleted());
    try std.testing.expect(!deleted.hasBackendStorage());
    try std.testing.expectEqual(before_deleted, ctx.memory().stats.bytes_in_use);
    try std.testing.expectError(error.BufferDeleted, deleted.ensureUsable());

    const before_donated = ctx.memory().stats.bytes_in_use;
    const donated = try ctx.hostCopy(.u8, &dims, 0, &data);
    defer donated.deinit();
    try std.testing.expectEqual(before_donated + data.len, ctx.memory().stats.bytes_in_use);
    try std.testing.expect(donated.hasBackendStorage());
    donated.markDonated();
    try std.testing.expectEqual(BufferState.donated, donated.lifecycleState());
    try std.testing.expect(donated.isDeleted());
    try std.testing.expect(!donated.hasBackendStorage());
    try std.testing.expectEqual(before_donated, ctx.memory().stats.bytes_in_use);
    try std.testing.expectError(error.BufferDonated, donated.ensureUsable());
}

test "buffer copies respect readiness events" {
    var ctx = try BufferTestContext.init(std.testing.allocator);
    defer ctx.deinit();

    const dims = [_]i64{4};
    const data = [_]u8{ 1, 2, 3, 4 };
    const buffer = try ctx.hostCopy(.u8, &dims, 0, &data);
    defer buffer.deinit();

    var out: [4]u8 = undefined;
    Buffer.Testing.setReadiness(buffer, Event.pending());
    try std.testing.expectError(error.BufferNotReady, buffer.ensureReady());
    try std.testing.expectError(error.BufferNotReady, buffer.copyToHost(&out));

    buffer.markReady();
    try buffer.ensureReady();
    try buffer.copyToHost(&out);
    try std.testing.expectEqualSlices(u8, &data, &out);

    buffer.failReadiness("producer failed");
    try std.testing.expectError(error.BufferReadinessFailed, buffer.ensureReady());
    try std.testing.expectError(error.BufferReadinessFailed, buffer.copyToHost(&out));
}
