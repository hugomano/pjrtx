const std = @import("std");
const backend_api = @import("src/backend/mlx_metal");
const ir = @import("src/compiler/ir");

const device_memory = @import("device_memory.zig");
const event_mod = @import("event.zig");

const BufferType = ir.BufferType;
const Device = device_memory.Device;
const DeviceMemoryTopology = device_memory.DeviceMemoryTopology;
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

const InitialReadiness = enum {
    ready,
    pending,
};

/// Owns immutable logical tensor metadata for a runtime buffer.
const BufferDescriptor = struct {
    element_type: BufferType,
    dims: []i64,
    byte_size: usize,
    shard_index: usize,

    /// Duplicates shape metadata and records the device-resident byte size.
    pub fn init(allocator: std.mem.Allocator, element_type: BufferType, dims_: []const i64, shard_index: usize, byte_size: usize) !BufferDescriptor {
        return .{
            .element_type = element_type,
            .dims = try allocator.dupe(i64, dims_),
            .byte_size = byte_size,
            .shard_index = shard_index,
        };
    }

    /// Releases owned shape metadata.
    pub fn deinit(self: BufferDescriptor, allocator: std.mem.Allocator) void {
        allocator.free(self.dims);
    }
};

/// Owns device and memory placement metadata for a runtime buffer.
const BufferPlacement = struct {
    device_id: i32,
    memory_id: i32,
    device: *const Device,
    memory: *Memory,

    /// Validates that the requested memory can be addressed by the device.
    pub fn init(device: *const Device, memory: *Memory) error{InvalidArgument}!BufferPlacement {
        if (!memory.isAddressableBy(device)) return error.InvalidArgument;
        return .{
            .device_id = device.id,
            .memory_id = memory.id,
            .device = device,
            .memory = memory,
        };
    }

    /// Returns true when this placement belongs to an execution device slot.
    pub fn matchesExecutionSlot(self: BufferPlacement, device_id: i32) bool {
        return self.device_id == device_id;
    }

    /// Accounts bytes transferred from host into this memory placement.
    pub fn recordHostToDeviceTransfer(self: BufferPlacement, byte_size: usize) void {
        self.memory.stats.host_to_device_bytes += @intCast(byte_size);
    }

    /// Accounts bytes transferred from this memory placement back to host.
    pub fn recordDeviceToHostTransfer(self: BufferPlacement, byte_size: usize) void {
        self.memory.stats.device_to_host_bytes += @intCast(byte_size);
    }
};

/// Owns backend buffer storage and memory accounting for a runtime buffer.
const DeviceStorage = struct {
    backend: backend_api.Backend,
    handle: ?backend_api.BufferHandle = null,
    accounted_bytes: usize = 0,

    /// Creates storage from an optional backend handle without taking accounting yet.
    pub fn init(backend: backend_api.Backend, handle: ?backend_api.BufferHandle) DeviceStorage {
        return .{ .backend = backend, .handle = handle };
    }

    /// Returns whether a backend device allocation is currently attached.
    pub fn hasBackendStorage(self: DeviceStorage) bool {
        return self.handle != null;
    }

    /// Returns the backend handle for runtime dispatch without transferring ownership.
    pub fn handleForDispatch(self: DeviceStorage) ?backend_api.BufferHandle {
        return self.handle;
    }

    /// Accounts this storage against its memory placement.
    pub fn account(self: *DeviceStorage, memory: *Memory, byte_size: usize) void {
        self.accounted_bytes = byte_size;
        memory.stats.retain(self.accounted_bytes);
    }

    /// Releases backend storage and memory accounting.
    pub fn release(self: *DeviceStorage, memory: *Memory) void {
        if (self.accounted_bytes != 0) {
            memory.stats.release(self.accounted_bytes);
            self.accounted_bytes = 0;
        }
        if (self.handle) |backend_buffer| {
            self.backend.destroyBuffer(backend_buffer);
            self.handle = null;
        }
    }

    /// Transfers backend storage ownership for a donation alias.
    pub fn takeForDonationAlias(self: *DeviceStorage, memory: *Memory) !backend_api.BufferHandle {
        const backend_buffer = self.handle orelse return error.UnsupportedRuntimeFeature;
        self.handle = null;
        if (self.accounted_bytes != 0) {
            memory.stats.release(self.accounted_bytes);
            self.accounted_bytes = 0;
        }
        return backend_buffer;
    }

    /// Replaces backend storage and refreshes memory accounting.
    pub fn replace(self: *DeviceStorage, memory: *Memory, byte_size: usize, backend_buffer: backend_api.BufferHandle) void {
        self.release(memory);
        self.handle = backend_buffer;
        self.account(memory, byte_size);
    }
};

pub const Buffer = struct {
    allocator: std.mem.Allocator,
    descriptor: BufferDescriptor,
    placement: BufferPlacement,
    storage: DeviceStorage,
    host_debug_bytes: []u8,
    state: BufferState = .live,
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
        return initOwned(
            allocator,
            src.storage.backend,
            element_type,
            dims_,
            src.placement.device,
            src.placement.memory,
            shard_index,
            byte_size,
            backend_buffer,
            .ready,
        );
    }

    pub fn initHostCopyForBackend(
        allocator: std.mem.Allocator,
        backend: backend_api.Backend,
        element_type: BufferType,
        dims_: []const i64,
        device: *const Device,
        memory: *Memory,
        shard_index: usize,
        src: []const u8,
    ) BufferCreateError!*Buffer {
        const placement = try BufferPlacement.init(device, memory);
        const backend_buffer = try backend.bufferFromHost(device.local_hardware_id, element_type, dims_, src) orelse
            return error.UnsupportedRuntimeFeature;
        errdefer backend.destroyBuffer(backend_buffer);

        const buffer = try initOwnedWithPlacement(allocator, backend, element_type, dims_, placement, shard_index, src.len, backend_buffer, .ready);
        placement.recordHostToDeviceTransfer(src.len);
        return buffer;
    }

    pub fn initDeviceAllocationForBackend(
        allocator: std.mem.Allocator,
        backend: backend_api.Backend,
        element_type: BufferType,
        dims_: []const i64,
        device: *const Device,
        memory: *Memory,
        shard_index: usize,
    ) BufferCreateError!*Buffer {
        const placement = try BufferPlacement.init(device, memory);

        const byte_size = ir.denseByteSize(element_type, dims_);
        const backend_buffer = try backend.allocateBuffer(device.local_hardware_id, element_type, dims_) orelse
            return error.UnsupportedRuntimeFeature;
        errdefer backend.destroyBuffer(backend_buffer);

        return initOwnedWithPlacement(allocator, backend, element_type, dims_, placement, shard_index, byte_size, backend_buffer, .ready);
    }

    pub fn initPendingBackendTransfer(
        allocator: std.mem.Allocator,
        backend: backend_api.Backend,
        element_type: BufferType,
        dims_: []const i64,
        device: *const Device,
        memory: *Memory,
        shard_index: usize,
    ) PendingBackendTransferBufferError!*Buffer {
        const placement = try BufferPlacement.init(device, memory);
        return initOwnedWithPlacement(allocator, backend, element_type, dims_, placement, shard_index, ir.denseByteSize(element_type, dims_), null, .pending);
    }

    pub fn initDeviceCopy(
        allocator: std.mem.Allocator,
        src: *Buffer,
        shard_index: usize,
    ) !*Buffer {
        try src.ensureUsable();
        const src_backend = src.storage.handleForDispatch() orelse return error.UnsupportedRuntimeFeature;
        const backend_buffer = try src.storage.backend.cloneBuffer(src_backend) orelse return error.UnsupportedRuntimeFeature;
        errdefer src.storage.backend.destroyBuffer(backend_buffer);

        return initOwned(
            allocator,
            src.storage.backend,
            src.descriptor.element_type,
            src.descriptor.dims,
            src.placement.device,
            src.placement.memory,
            shard_index,
            src.descriptor.byte_size,
            backend_buffer,
            .ready,
        );
    }

    pub fn deinit(self: *Buffer) void {
        self.releaseStorage();
        self.allocator.free(self.host_debug_bytes);
        self.descriptor.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    pub fn copyToHost(self: *Buffer, dst: []u8) HostReadbackError!void {
        try self.ensureUsable();
        try self.ensureReady();
        if (dst.len < self.descriptor.byte_size) return error.DestinationTooSmall;
        const backend_buffer = self.storage.handleForDispatch() orelse return error.UnsupportedRuntimeFeature;
        self.storage.backend.copyToHost(backend_buffer, dst) catch return error.BackendBufferCopyFailed;
        self.placement.recordDeviceToHostTransfer(self.descriptor.byte_size);
    }

    pub fn hasBackendStorage(self: *const Buffer) bool {
        return self.storage.hasBackendStorage();
    }

    /// Returns the compiler-owned element type for PJRT metadata.
    pub fn elementType(self: *const Buffer) BufferType {
        return self.descriptor.element_type;
    }

    /// Returns immutable runtime dimensions for PJRT metadata.
    pub fn dimensions(self: *const Buffer) []const i64 {
        return self.descriptor.dims;
    }

    /// Returns device-resident byte size tracked by runtime accounting.
    pub fn onDeviceSizeInBytes(self: *const Buffer) usize {
        return self.descriptor.byte_size;
    }

    /// Returns the logical shard index for this buffer placement.
    pub fn shardIndex(self: *const Buffer) usize {
        return self.descriptor.shard_index;
    }

    /// Returns true when this buffer belongs to the given device/shard execution slot.
    pub fn matchesExecutionSlot(self: *const Buffer, device_id: i32, shard_index: usize) bool {
        return self.placement.matchesExecutionSlot(device_id) and self.descriptor.shard_index == shard_index;
    }

    /// Returns retained host/debug bytes; device-only fast paths should keep this zero.
    pub fn hostDebugByteCount(self: *const Buffer) usize {
        return self.host_debug_bytes.len;
    }

    /// Returns whether this buffer has been deleted or donated from PJRT's view.
    pub fn isDeleted(self: *const Buffer) bool {
        return self.state != .live;
    }

    /// Returns the lifecycle state for runtime validation and focused tests.
    pub fn lifecycleState(self: *const Buffer) BufferState {
        return self.state;
    }

    /// Returns the runtime device placement for PJRT metadata.
    pub fn devicePlacement(self: *const Buffer) *const Device {
        return self.placement.device;
    }

    /// Returns the runtime memory placement for PJRT metadata.
    pub fn memoryPlacement(self: *const Buffer) *Memory {
        return self.placement.memory;
    }

    /// Returns this buffer's readiness event by value for PJRT event bridging.
    pub fn readinessEvent(self: *const Buffer) Event {
        return self.ready_event;
    }

    /// Marks this buffer as ready after backend storage is installed.
    pub fn markReady(self: *Buffer) void {
        self.ready_event.setReady();
    }

    /// Marks this buffer readiness as failed with a stable runtime message.
    pub fn failReadiness(self: *Buffer, message: []const u8) void {
        self.ready_event.setFailed(message);
    }

    /// Records bytes transferred from host into this buffer's memory placement.
    pub fn recordHostToDeviceTransfer(self: *Buffer) void {
        self.placement.recordHostToDeviceTransfer(self.descriptor.byte_size);
    }

    /// Returns the backend storage handle for runtime dispatch without transferring ownership.
    pub fn backendHandleForDispatch(self: *const Buffer) ?backend_api.BufferHandle {
        return self.storage.handleForDispatch();
    }

    pub fn markDeleted(self: *Buffer) void {
        self.releaseStorage();
        self.state = .deleted;
        self.ready_event.setFailed("buffer has been deleted");
    }

    pub fn markDonated(self: *Buffer) void {
        self.releaseStorage();
        self.state = .donated;
        self.ready_event.setFailed("buffer has been donated");
    }

    pub fn takeBackendStorageForDonationAlias(self: *Buffer) !backend_api.BufferHandle {
        try self.ensureUsable();
        return self.storage.takeForDonationAlias(self.placement.memory);
    }

    pub fn replaceBackendStorage(self: *Buffer, backend_buffer: backend_api.BufferHandle) error{ BufferDeleted, BufferDonated }!void {
        try self.ensureUsable();
        self.storage.replace(self.placement.memory, self.descriptor.byte_size, backend_buffer);
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

    /// Test-only hooks for validating runtime buffer invariants.
    pub const Testing = struct {
        /// Replaces readiness state for focused execution tests.
        pub fn setReadiness(buffer: *Buffer, event: Event) void {
            buffer.ready_event = event;
        }

        /// Replaces stable device id for focused placement validation tests.
        pub fn setDeviceId(buffer: *Buffer, device_id: i32) void {
            buffer.placement.device_id = device_id;
        }
    };

    fn accountDeviceBytes(self: *Buffer) *Buffer {
        self.storage.account(self.placement.memory, self.descriptor.byte_size);
        return self;
    }

    fn releaseStorage(self: *Buffer) void {
        self.storage.release(self.placement.memory);
    }

    fn initOwned(
        allocator: std.mem.Allocator,
        backend: backend_api.Backend,
        element_type: BufferType,
        dims_: []const i64,
        device: *const Device,
        memory: *Memory,
        shard_index: usize,
        byte_size: usize,
        backend_buffer: ?backend_api.BufferHandle,
        readiness: InitialReadiness,
    ) !*Buffer {
        const placement = try BufferPlacement.init(device, memory);
        return initOwnedWithPlacement(allocator, backend, element_type, dims_, placement, shard_index, byte_size, backend_buffer, readiness);
    }

    fn initOwnedWithPlacement(
        allocator: std.mem.Allocator,
        backend: backend_api.Backend,
        element_type: BufferType,
        dims_: []const i64,
        placement: BufferPlacement,
        shard_index: usize,
        byte_size: usize,
        backend_buffer: ?backend_api.BufferHandle,
        readiness: InitialReadiness,
    ) !*Buffer {
        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const descriptor = try BufferDescriptor.init(allocator, element_type, dims_, shard_index, byte_size);
        errdefer descriptor.deinit(allocator);

        const host_debug_bytes = try allocator.alloc(u8, 0);
        errdefer allocator.free(host_debug_bytes);

        buffer.* = .{
            .allocator = allocator,
            .descriptor = descriptor,
            .placement = placement,
            .storage = DeviceStorage.init(backend, backend_buffer),
            .host_debug_bytes = host_debug_bytes,
            .ready_event = switch (readiness) {
                .ready => Event.ready(),
                .pending => Event.pending(),
            },
        };
        if (backend_buffer != null) return buffer.accountDeviceBytes();
        return buffer;
    }

    pub fn initBackendHandle(
        allocator: std.mem.Allocator,
        backend: backend_api.Backend,
        element_type: BufferType,
        dims_: []const i64,
        device: *const Device,
        memory: *Memory,
        shard_index: usize,
        byte_size: usize,
        backend_buffer: backend_api.BufferHandle,
    ) !*Buffer {
        return initOwned(allocator, backend, element_type, dims_, device, memory, shard_index, byte_size, backend_buffer, .ready);
    }
};

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

    fn hostCopy(
        self: *BufferTestContext,
        element_type: BufferType,
        dims: []const i64,
        shard_index: usize,
        src: []const u8,
    ) !*Buffer {
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

    try std.testing.expectEqual(@as(i32, 0), buffer.placement.device_id);
    try std.testing.expectEqual(@as(i32, 0), buffer.placement.memory_id);
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
    try std.testing.expectError(
        error.InvalidArgument,
        Buffer.initHostCopyForBackend(std.testing.allocator, ctx.backend, .u8, &dims, ctx.device(), &invalid_memory, 0, &data),
    );

    if (try ctx.backend.bufferFromHost(ctx.device().local_hardware_id, .u8, &dims, &data)) |backend_buffer| {
        try std.testing.expectError(
            error.InvalidArgument,
            Buffer.initBackendHandle(std.testing.allocator, ctx.backend, .u8, &dims, ctx.device(), &invalid_memory, 0, data.len, backend_buffer),
        );
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
    try std.testing.expectEqual(BufferState.deleted, deleted.state);
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
    try std.testing.expectEqual(BufferState.donated, donated.state);
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
