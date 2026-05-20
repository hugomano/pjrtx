const std = @import("std");
const backend_api = @import("src/backend/mlx_metal");
const ir = @import("src/compiler/ir");

const device_memory = @import("device_memory.zig");
const event_mod = @import("event.zig");

const BufferType = ir.BufferType;
const Device = device_memory.Device;
const Event = event_mod.Event;
const Memory = device_memory.Memory;

/// Errors returned while creating device-resident runtime buffers.
pub const BufferCreateError = std.mem.Allocator.Error || backend_api.Error || error{ InvalidArgument, UnsupportedRuntimeFeature };

/// Errors returned while reserving a buffer for async backend transfer completion.
pub const PendingBackendTransferBufferError = std.mem.Allocator.Error || error{InvalidArgument};

/// Errors returned while copying device-resident buffer contents to host memory.
pub const HostReadbackError = error{
    DestinationTooSmall,
    BufferDeleted,
    BufferDonated,
    BufferNotReady,
    BufferReadinessFailed,
    UnsupportedRuntimeFeature,
    BackendBufferCopyFailed,
};

/// Tracks whether a runtime buffer may still be used by execute/copy paths.
pub const BufferState = enum {
    live,
    deleted,
    donated,
};

pub const Buffer = struct {
    allocator: std.mem.Allocator,
    backend: backend_api.Backend,
    element_type: BufferType,
    dims: []i64,
    device_id: i32,
    memory_id: i32,
    device: *Device,
    memory: *Memory,
    shard_index: usize,
    byte_size: usize,
    bytes: []u8,
    backend_buffer: ?backend_api.BufferHandle = null,
    state: BufferState = .live,
    deleted: bool = false,
    accounted_bytes: usize = 0,
    ready_event: Event = Event.ready(),

    fn initBackendOnly(
        allocator: std.mem.Allocator,
        src: *Buffer,
        element_type: BufferType,
        dims_: []const i64,
        byte_size: usize,
        backend_buffer: backend_api.BufferHandle,
        shard_index: usize,
    ) BufferCreateError!*Buffer {
        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, dims_);
        errdefer allocator.free(dims);

        const bytes = try allocator.alloc(u8, 0);
        errdefer allocator.free(bytes);

        buffer.* = .{
            .allocator = allocator,
            .backend = src.backend,
            .element_type = element_type,
            .dims = dims,
            .device_id = src.device_id,
            .memory_id = src.memory_id,
            .device = src.device,
            .memory = src.memory,
            .shard_index = shard_index,
            .byte_size = byte_size,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
        };
        return buffer.accountDeviceBytes();
    }

    pub fn initHostCopyForBackend(
        allocator: std.mem.Allocator,
        backend_impl: backend_api.Backend,
        element_type: BufferType,
        dims_: []const i64,
        device: *Device,
        memory: *Memory,
        shard_index: usize,
        src: []const u8,
    ) BufferCreateError!*Buffer {
        if (!memory.isAddressableBy(device)) return error.InvalidArgument;

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, dims_);
        errdefer allocator.free(dims);

        const backend_buffer = try backend_impl.bufferFromHost(device.local_hardware_id, element_type, dims, src) orelse
            return error.UnsupportedRuntimeFeature;
        errdefer backend_impl.destroyBuffer(backend_buffer);

        const bytes = try allocator.alloc(u8, 0);
        errdefer allocator.free(bytes);

        buffer.* = .{
            .allocator = allocator,
            .backend = backend_impl,
            .element_type = element_type,
            .dims = dims,
            .device_id = device.id,
            .memory_id = memory.id,
            .device = device,
            .memory = memory,
            .shard_index = shard_index,
            .byte_size = src.len,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
        };
        memory.stats.host_to_device_bytes += @intCast(src.len);
        return buffer.accountDeviceBytes();
    }

    pub fn initDeviceAllocationForBackend(
        allocator: std.mem.Allocator,
        backend_impl: backend_api.Backend,
        element_type: BufferType,
        dims_: []const i64,
        device: *Device,
        memory: *Memory,
        shard_index: usize,
    ) BufferCreateError!*Buffer {
        if (!memory.isAddressableBy(device)) return error.InvalidArgument;

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, dims_);
        errdefer allocator.free(dims);

        const byte_size = ir.denseByteSize(element_type, dims);
        const backend_buffer = try backend_impl.allocateBuffer(device.local_hardware_id, element_type, dims) orelse
            return error.UnsupportedRuntimeFeature;
        errdefer backend_impl.destroyBuffer(backend_buffer);

        const bytes = try allocator.alloc(u8, 0);
        errdefer allocator.free(bytes);

        buffer.* = .{
            .allocator = allocator,
            .backend = backend_impl,
            .element_type = element_type,
            .dims = dims,
            .device_id = device.id,
            .memory_id = memory.id,
            .device = device,
            .memory = memory,
            .shard_index = shard_index,
            .byte_size = byte_size,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
        };
        return buffer.accountDeviceBytes();
    }

    pub fn initPendingBackendTransfer(
        allocator: std.mem.Allocator,
        backend_impl: backend_api.Backend,
        element_type: BufferType,
        dims_: []const i64,
        device: *Device,
        memory: *Memory,
        shard_index: usize,
    ) PendingBackendTransferBufferError!*Buffer {
        if (!memory.isAddressableBy(device)) return error.InvalidArgument;

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, dims_);
        errdefer allocator.free(dims);

        const bytes = try allocator.alloc(u8, 0);
        errdefer allocator.free(bytes);

        buffer.* = .{
            .allocator = allocator,
            .backend = backend_impl,
            .element_type = element_type,
            .dims = dims,
            .device_id = device.id,
            .memory_id = memory.id,
            .device = device,
            .memory = memory,
            .shard_index = shard_index,
            .byte_size = ir.denseByteSize(element_type, dims),
            .bytes = bytes,
            .backend_buffer = null,
            .ready_event = Event.pending(),
        };
        return buffer;
    }

    pub fn initDeviceCopy(
        allocator: std.mem.Allocator,
        src: *Buffer,
        shard_index: usize,
    ) !*Buffer {
        try src.ensureUsable();
        const src_backend = src.backend_buffer orelse return error.UnsupportedRuntimeFeature;
        const backend_buffer = try src.backend.cloneBuffer(src_backend) orelse return error.UnsupportedRuntimeFeature;
        errdefer src.backend.destroyBuffer(backend_buffer);

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, src.dims);
        errdefer allocator.free(dims);

        const bytes = try allocator.alloc(u8, 0);
        errdefer allocator.free(bytes);

        buffer.* = .{
            .allocator = allocator,
            .backend = src.backend,
            .element_type = src.element_type,
            .dims = dims,
            .device_id = src.device_id,
            .memory_id = src.memory_id,
            .device = src.device,
            .memory = src.memory,
            .shard_index = shard_index,
            .byte_size = src.byte_size,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
        };
        return buffer.accountDeviceBytes();
    }

    pub fn deinit(self: *Buffer) void {
        self.releaseStorage();
        self.allocator.free(self.bytes);
        self.allocator.free(self.dims);
        self.allocator.destroy(self);
    }

    pub fn copyToHost(self: *Buffer, dst: []u8) HostReadbackError!void {
        try self.ensureUsable();
        try self.ensureReady();
        if (dst.len < self.byte_size) return error.DestinationTooSmall;
        const backend_buffer = self.backend_buffer orelse return error.UnsupportedRuntimeFeature;
        self.backend.copyToHost(backend_buffer, dst) catch return error.BackendBufferCopyFailed;
        self.memory.stats.device_to_host_bytes += @intCast(self.byte_size);
    }

    pub fn hasBackendStorage(self: *const Buffer) bool {
        return self.backend_buffer != null;
    }

    pub fn markDeleted(self: *Buffer) void {
        self.releaseStorage();
        self.state = .deleted;
        self.deleted = true;
        self.ready_event.setFailed("buffer has been deleted");
    }

    pub fn markDonated(self: *Buffer) void {
        self.releaseStorage();
        self.state = .donated;
        self.deleted = true;
        self.ready_event.setFailed("buffer has been donated");
    }

    pub fn takeBackendStorageForDonationAlias(self: *Buffer) !backend_api.BufferHandle {
        try self.ensureUsable();
        const backend_buffer = self.backend_buffer orelse return error.UnsupportedRuntimeFeature;
        self.backend_buffer = null;
        if (self.accounted_bytes != 0) {
            self.memory.stats.release(self.accounted_bytes);
            self.accounted_bytes = 0;
        }
        return backend_buffer;
    }

    pub fn replaceBackendStorage(self: *Buffer, backend_buffer: backend_api.BufferHandle) error{ BufferDeleted, BufferDonated }!void {
        try self.ensureUsable();
        self.releaseStorage();
        self.backend_buffer = backend_buffer;
        _ = self.accountDeviceBytes();
    }

    pub fn ensureUsable(self: *const Buffer) !void {
        return switch (self.state) {
            .live => {},
            .deleted => error.BufferDeleted,
            .donated => error.BufferDonated,
        };
    }

    pub fn ensureReady(self: *const Buffer) !void {
        self.ready_event.awaitReady() catch |err| return switch (err) {
            error.EventPending => error.BufferNotReady,
            error.EventFailed => error.BufferReadinessFailed,
        };
    }

    pub fn chainReadyAfter(self: *Buffer, dependency: *Event) !void {
        try self.ensureUsable();
        self.ready_event = Event.pending();
        try dependency.chainTo(&self.ready_event);
    }

    fn accountDeviceBytes(self: *Buffer) *Buffer {
        self.accounted_bytes = self.byte_size;
        self.memory.stats.retain(self.accounted_bytes);
        return self;
    }

    fn releaseStorage(self: *Buffer) void {
        if (self.accounted_bytes != 0) {
            self.memory.stats.release(self.accounted_bytes);
            self.accounted_bytes = 0;
        }
        if (self.backend_buffer) |backend_buffer| {
            self.backend.destroyBuffer(backend_buffer);
            self.backend_buffer = null;
        }
    }

    pub fn initBackendHandle(
        allocator: std.mem.Allocator,
        backend_impl: backend_api.Backend,
        element_type: BufferType,
        dims_: []const i64,
        device: *Device,
        memory: *Memory,
        shard_index: usize,
        byte_size: usize,
        backend_buffer: backend_api.BufferHandle,
    ) !*Buffer {
        if (!memory.isAddressableBy(device)) return error.InvalidArgument;

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, dims_);
        errdefer allocator.free(dims);

        const bytes = try allocator.alloc(u8, 0);
        errdefer allocator.free(bytes);

        buffer.* = .{
            .allocator = allocator,
            .backend = backend_impl,
            .element_type = element_type,
            .dims = dims,
            .device_id = device.id,
            .memory_id = memory.id,
            .device = device,
            .memory = memory,
            .shard_index = shard_index,
            .byte_size = byte_size,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
        };
        return buffer.accountDeviceBytes();
    }
};

const BufferTestContext = struct {
    allocator: std.mem.Allocator,
    backend: backend_api.Backend,
    descriptors: []ir.DeviceDescriptor,
    device: *Device,
    memory: *Memory,
    addressable_device_ids: []i32,
    addressable_devices: []*Device,
    addressable_memories: []*Memory,

    fn init(allocator: std.mem.Allocator) !BufferTestContext {
        const backend = backend_api.create();
        const descriptors = try backend.enumerateDevices(allocator, 1);
        errdefer backend.releaseDeviceDescriptors(allocator, descriptors);
        if (descriptors.len == 0) return error.InvalidDeviceCount;

        const device = try allocator.create(Device);
        errdefer allocator.destroy(device);

        const memory = try allocator.create(Memory);
        errdefer allocator.destroy(memory);

        const addressable_device_ids = try allocator.alloc(i32, 1);
        errdefer allocator.free(addressable_device_ids);

        const addressable_devices = try allocator.alloc(*Device, 1);
        errdefer allocator.free(addressable_devices);

        const addressable_memories = try allocator.alloc(*Memory, 1);
        errdefer allocator.free(addressable_memories);

        const descriptor = descriptors[0];
        addressable_device_ids[0] = descriptor.id;
        memory.* = .{
            .id = descriptor.default_memory_id,
            .kind = .device,
            .debug_string = "device",
            .addressable_device_ids = addressable_device_ids,
            .addressable_devices = addressable_devices,
            .stats = .{ .capacity_bytes = descriptor.memory_bytes },
        };
        device.* = .{
            .id = descriptor.id,
            .local_hardware_id = descriptor.local_hardware_id,
            .registry_id = descriptor.registry_id,
            .process_index = descriptor.process_index,
            .addressable = descriptor.addressable,
            .name = descriptor.name,
            .debug_string = descriptor.debug_string,
            .memory_bytes = descriptor.memory_bytes,
            .has_unified_memory = descriptor.has_unified_memory,
            .default_memory_id = descriptor.default_memory_id,
            .default_memory = memory,
            .addressable_memories = addressable_memories,
        };
        addressable_devices[0] = device;
        addressable_memories[0] = memory;

        return .{
            .allocator = allocator,
            .backend = backend,
            .descriptors = descriptors,
            .device = device,
            .memory = memory,
            .addressable_device_ids = addressable_device_ids,
            .addressable_devices = addressable_devices,
            .addressable_memories = addressable_memories,
        };
    }

    fn deinit(self: *BufferTestContext) void {
        self.allocator.free(self.addressable_memories);
        self.allocator.free(self.addressable_devices);
        self.allocator.free(self.addressable_device_ids);
        self.allocator.destroy(self.memory);
        self.allocator.destroy(self.device);
        self.backend.releaseDeviceDescriptors(self.allocator, self.descriptors);
        self.* = undefined;
    }

    fn hostCopy(
        self: *BufferTestContext,
        element_type: BufferType,
        dims: []const i64,
        shard_index: usize,
        src: []const u8,
    ) !*Buffer {
        return Buffer.initHostCopyForBackend(self.allocator, self.backend, element_type, dims, self.device, self.memory, shard_index, src);
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

    try std.testing.expectEqual(@as(i32, 0), buffer.device_id);
    try std.testing.expectEqual(@as(i32, 0), buffer.memory_id);
    try std.testing.expectEqual(@as(i32, 0), buffer.device.id);
    try std.testing.expectEqual(@as(i32, 0), buffer.memory.id);
    try std.testing.expectEqual(@as(usize, 0), buffer.shard_index);
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
    try std.testing.expect(!invalid_memory.isAddressableBy(ctx.device));
    try std.testing.expectError(
        error.InvalidArgument,
        Buffer.initHostCopyForBackend(std.testing.allocator, ctx.backend, .u8, &dims, ctx.device, &invalid_memory, 0, &data),
    );

    if (try ctx.backend.bufferFromHost(ctx.device.local_hardware_id, .u8, &dims, &data)) |backend_buffer| {
        try std.testing.expectError(
            error.InvalidArgument,
            Buffer.initBackendHandle(std.testing.allocator, ctx.backend, .u8, &dims, ctx.device, &invalid_memory, 0, data.len, backend_buffer),
        );
        ctx.backend.destroyBuffer(backend_buffer);
    }
}

test "buffer lifecycle rejects deleted and donated buffers" {
    var ctx = try BufferTestContext.init(std.testing.allocator);
    defer ctx.deinit();

    const dims = [_]i64{4};
    const data = [_]u8{ 1, 2, 3, 4 };
    const before_deleted = ctx.memory.stats.bytes_in_use;
    const deleted = try ctx.hostCopy(.u8, &dims, 0, &data);
    defer deleted.deinit();
    try std.testing.expectEqual(before_deleted + data.len, ctx.memory.stats.bytes_in_use);
    try std.testing.expect(deleted.hasBackendStorage());
    deleted.markDeleted();
    try std.testing.expectEqual(BufferState.deleted, deleted.state);
    try std.testing.expect(deleted.deleted);
    try std.testing.expect(!deleted.hasBackendStorage());
    try std.testing.expectEqual(before_deleted, ctx.memory.stats.bytes_in_use);
    try std.testing.expectError(error.BufferDeleted, deleted.ensureUsable());

    const before_donated = ctx.memory.stats.bytes_in_use;
    const donated = try ctx.hostCopy(.u8, &dims, 0, &data);
    defer donated.deinit();
    try std.testing.expectEqual(before_donated + data.len, ctx.memory.stats.bytes_in_use);
    try std.testing.expect(donated.hasBackendStorage());
    donated.markDonated();
    try std.testing.expectEqual(BufferState.donated, donated.state);
    try std.testing.expect(donated.deleted);
    try std.testing.expect(!donated.hasBackendStorage());
    try std.testing.expectEqual(before_donated, ctx.memory.stats.bytes_in_use);
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
    buffer.ready_event = Event.pending();
    try std.testing.expectError(error.BufferNotReady, buffer.ensureReady());
    try std.testing.expectError(error.BufferNotReady, buffer.copyToHost(&out));

    buffer.ready_event.setReady();
    try buffer.ensureReady();
    try buffer.copyToHost(&out);
    try std.testing.expectEqualSlices(u8, &data, &out);

    buffer.ready_event = Event.failed("producer failed");
    try std.testing.expectError(error.BufferReadinessFailed, buffer.ensureReady());
    try std.testing.expectError(error.BufferReadinessFailed, buffer.copyToHost(&out));
}
