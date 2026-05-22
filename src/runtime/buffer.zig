const std = @import("std");
const backend_api = @import("backend_selection.zig");
const ir = @import("src/compiler/ir");

const buffer_descriptor = @import("buffer_descriptor.zig");
const buffer_placement = @import("buffer_placement.zig");
const buffer_storage = @import("buffer_storage.zig");
const device_memory = @import("device_memory.zig");
const event_mod = @import("event.zig");

const BufferType = ir.BufferType;
const Descriptor = buffer_descriptor.Descriptor;
const Device = device_memory.Device;
const Event = event_mod.Event;
const Memory = device_memory.Memory;
const Placement = buffer_placement.Placement;
const Storage = buffer_storage.Storage;

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

/// Owns runtime buffer lifecycle, readiness, placement, and backend storage.
pub const Buffer = struct {
    allocator: std.mem.Allocator,
    descriptor: Descriptor,
    placement: Placement,
    storage: Storage,
    state: BufferState = .live,
    ready_event: Event = Event.ready(),

    /// Imports host bytes through the selected backend into device-resident storage.
    pub const initHostCopyForBackend = BufferLifecycle.initHostCopyForBackend;
    /// Allocates uninitialized backend storage for this runtime buffer.
    pub const initDeviceAllocationForBackend = BufferLifecycle.initDeviceAllocationForBackend;
    /// Reserves runtime buffer metadata before async backend storage is completed.
    pub const initPendingBackendTransfer = BufferLifecycle.initPendingBackendTransfer;
    /// Creates a same-device backend clone while preserving logical metadata.
    pub const initDeviceCopy = BufferLifecycle.initDeviceCopy;
    /// Adopts an existing backend handle as owned runtime buffer storage.
    pub const initBackendHandle = BufferLifecycle.initBackendHandle;
    /// Releases backend storage, descriptor allocations, and the buffer object.
    pub const deinit = BufferLifecycle.deinit;
    /// Copies explicit device-resident contents to caller-owned host memory.
    pub const copyToHost = BufferLifecycle.copyToHost;
    /// Returns true when this buffer currently owns backend device storage.
    pub const hasBackendStorage = BufferMetadata.hasBackendStorage;
    /// Returns the compiler-owned element type for PJRT metadata.
    pub const elementType = BufferMetadata.elementType;
    /// Returns immutable runtime dimensions for PJRT metadata.
    pub const dimensions = BufferMetadata.dimensions;
    /// Returns device-resident byte size tracked by runtime accounting.
    pub const onDeviceSizeInBytes = BufferMetadata.onDeviceSizeInBytes;
    /// Returns the logical shard index for this buffer placement.
    pub const shardIndex = BufferMetadata.shardIndex;
    /// Returns true when this buffer belongs to the given device/shard execution slot.
    pub const matchesExecutionSlot = BufferMetadata.matchesExecutionSlot;
    /// Returns whether this buffer has been deleted or donated from PJRT's view.
    pub const isDeleted = BufferMetadata.isDeleted;
    /// Returns the lifecycle state for runtime validation and focused tests.
    pub const lifecycleState = BufferMetadata.lifecycleState;
    /// Returns the runtime device placement for PJRT metadata.
    pub const devicePlacement = BufferMetadata.devicePlacement;
    /// Returns the runtime memory placement for PJRT metadata.
    pub const memoryPlacement = BufferMetadata.memoryPlacement;
    /// Returns this buffer's readiness event by value for PJRT event bridging.
    pub const readinessEvent = BufferReadiness.readinessEvent;
    /// Marks this buffer as ready after backend storage is installed.
    pub const markReady = BufferReadiness.markReady;
    /// Marks this buffer readiness as failed with a stable runtime message.
    pub const failReadiness = BufferReadiness.failReadiness;
    /// Records bytes transferred from host into this buffer's memory placement.
    pub const recordHostToDeviceTransfer = BufferMetadata.recordHostToDeviceTransfer;
    /// Returns the backend storage handle for runtime dispatch without transferring ownership.
    pub const backendHandleForDispatch = BufferMetadata.backendHandleForDispatch;
    /// Deletes this buffer and releases its backend storage.
    pub const markDeleted = BufferLifecycle.markDeleted;
    /// Donates this buffer and releases its backend storage.
    pub const markDonated = BufferLifecycle.markDonated;
    /// Transfers backend storage ownership into a donation alias output.
    pub const takeBackendStorageForDonationAlias = BufferLifecycle.takeBackendStorageForDonationAlias;
    /// Replaces backend storage after alias rollback or async transfer completion.
    pub const replaceBackendStorage = BufferLifecycle.replaceBackendStorage;
    /// Rejects use-after-delete and use-after-donation at runtime boundaries.
    pub const ensureUsable = BufferLifecycle.ensureUsable;
    /// Rejects consumption before producer readiness has completed.
    pub const ensureReady = BufferLifecycle.ensureReady;
    /// Makes this buffer ready only after the dependency event completes.
    pub const chainReadyAfter = BufferReadiness.chainReadyAfter;

    /// Test-only hooks for validating runtime buffer invariants.
    pub const Testing = struct {
        /// Replaces readiness state for focused execution tests.
        pub fn setReadiness(buffer: *Buffer, event: Event) void {
            buffer.ready_event = event;
        }

        /// Replaces stable device id for focused placement validation tests.
        pub fn setDeviceId(buffer: *Buffer, device_id: i32) void {
            buffer.placement.setDeviceIdForTest(device_id);
        }
    };
};

const BufferLifecycle = struct {
    fn initHostCopyForBackend(allocator: std.mem.Allocator, backend: backend_api.Backend, element_type: BufferType, dims: []const i64, device: *const Device, memory: *Memory, shard_index: usize, src: []const u8) BufferCreateError!*Buffer {
        const placement = try Placement.init(device, memory);
        const backend_buffer = try backend.bufferFromHost(device.local_hardware_id, element_type, dims, src) orelse return error.UnsupportedRuntimeFeature;
        errdefer backend.destroyBuffer(backend_buffer);

        const buffer = try BufferAdoption.initOwnedWithPlacement(allocator, backend, element_type, dims, placement, shard_index, src.len, backend_buffer, .ready);
        placement.recordHostToDeviceTransfer(src.len);
        return buffer;
    }

    fn initDeviceAllocationForBackend(allocator: std.mem.Allocator, backend: backend_api.Backend, element_type: BufferType, dims: []const i64, device: *const Device, memory: *Memory, shard_index: usize) BufferCreateError!*Buffer {
        const placement = try Placement.init(device, memory);
        const byte_size = ir.denseByteSize(element_type, dims);
        const backend_buffer = try backend.allocateBuffer(device.local_hardware_id, element_type, dims) orelse return error.UnsupportedRuntimeFeature;
        errdefer backend.destroyBuffer(backend_buffer);
        return BufferAdoption.initOwnedWithPlacement(allocator, backend, element_type, dims, placement, shard_index, byte_size, backend_buffer, .ready);
    }

    fn initPendingBackendTransfer(allocator: std.mem.Allocator, backend: backend_api.Backend, element_type: BufferType, dims: []const i64, device: *const Device, memory: *Memory, shard_index: usize) PendingBackendTransferBufferError!*Buffer {
        const placement = try Placement.init(device, memory);
        return BufferAdoption.initOwnedWithPlacement(allocator, backend, element_type, dims, placement, shard_index, ir.denseByteSize(element_type, dims), null, .pending);
    }

    fn initDeviceCopy(allocator: std.mem.Allocator, src: *Buffer, shard_index: usize) !*Buffer {
        try src.ensureUsable();
        const src_backend = src.storage.handleForDispatch() orelse return error.UnsupportedRuntimeFeature;
        const backend_buffer = try src.storage.backend.cloneBuffer(src_backend) orelse return error.UnsupportedRuntimeFeature;
        errdefer src.storage.backend.destroyBuffer(backend_buffer);

        return BufferAdoption.initOwned(allocator, src.storage.backend, src.descriptor.element_type, src.descriptor.dims, src.placement.device, src.placement.memory, shard_index, src.descriptor.byte_size, backend_buffer, .ready);
    }

    fn initBackendHandle(allocator: std.mem.Allocator, backend: backend_api.Backend, element_type: BufferType, dims: []const i64, device: *const Device, memory: *Memory, shard_index: usize, byte_size: usize, backend_buffer: backend_api.BufferHandle) !*Buffer {
        return BufferAdoption.initOwned(allocator, backend, element_type, dims, device, memory, shard_index, byte_size, backend_buffer, .ready);
    }

    fn deinit(buffer: *Buffer) void {
        BufferAdoption.releaseStorage(buffer);
        buffer.descriptor.deinit(buffer.allocator);
        buffer.allocator.destroy(buffer);
    }

    fn copyToHost(buffer: *Buffer, dst: []u8) HostReadbackError!void {
        try buffer.ensureUsable();
        try buffer.ensureReady();
        if (dst.len < buffer.descriptor.byte_size) return error.DestinationTooSmall;
        const backend_buffer = buffer.storage.handleForDispatch() orelse return error.UnsupportedRuntimeFeature;
        buffer.storage.backend.copyToHost(backend_buffer, dst) catch return error.BackendBufferCopyFailed;
        buffer.placement.recordDeviceToHostTransfer(buffer.descriptor.byte_size);
    }

    fn markDeleted(buffer: *Buffer) void {
        transitionOutOfLiveState(buffer, .deleted, "buffer has been deleted");
    }

    fn markDonated(buffer: *Buffer) void {
        transitionOutOfLiveState(buffer, .donated, "buffer has been donated");
    }

    fn takeBackendStorageForDonationAlias(buffer: *Buffer) !backend_api.BufferHandle {
        try buffer.ensureUsable();
        return buffer.storage.takeForDonationAlias(buffer.placement.memory);
    }

    fn replaceBackendStorage(buffer: *Buffer, backend_buffer: backend_api.BufferHandle) error{ BufferDeleted, BufferDonated }!void {
        try buffer.ensureUsable();
        buffer.storage.replace(buffer.placement.memory, buffer.descriptor.byte_size, backend_buffer);
    }

    fn transitionOutOfLiveState(buffer: *Buffer, state: BufferState, message: []const u8) void {
        BufferAdoption.releaseStorage(buffer);
        buffer.state = state;
        buffer.ready_event.setFailed(message);
    }

    fn ensureUsable(buffer: *const Buffer) !void {
        return switch (buffer.state) {
            .live => {},
            .deleted => error.BufferDeleted,
            .donated => error.BufferDonated,
        };
    }

    fn ensureReady(buffer: *const Buffer) !void {
        buffer.ready_event.awaitReady() catch |err| return switch (err) {
            error.EventPending => error.BufferNotReady,
            error.EventFailed => error.BufferReadinessFailed,
        };
    }
};

const BufferMetadata = struct {
    fn hasBackendStorage(buffer: *const Buffer) bool {
        return buffer.storage.hasBackendStorage();
    }

    fn elementType(buffer: *const Buffer) BufferType {
        return buffer.descriptor.element_type;
    }

    fn dimensions(buffer: *const Buffer) []const i64 {
        return buffer.descriptor.dims;
    }

    fn onDeviceSizeInBytes(buffer: *const Buffer) usize {
        return buffer.descriptor.byte_size;
    }

    fn shardIndex(buffer: *const Buffer) usize {
        return buffer.descriptor.shard_index;
    }

    fn matchesExecutionSlot(buffer: *const Buffer, device_id: i32, shard_index: usize) bool {
        return buffer.placement.matchesExecutionSlot(device_id) and buffer.descriptor.shard_index == shard_index;
    }

    fn isDeleted(buffer: *const Buffer) bool {
        return buffer.state != .live;
    }

    fn lifecycleState(buffer: *const Buffer) BufferState {
        return buffer.state;
    }

    fn devicePlacement(buffer: *const Buffer) *const Device {
        return buffer.placement.device;
    }

    fn memoryPlacement(buffer: *const Buffer) *Memory {
        return buffer.placement.memory;
    }

    fn recordHostToDeviceTransfer(buffer: *Buffer) void {
        buffer.placement.recordHostToDeviceTransfer(buffer.descriptor.byte_size);
    }

    fn backendHandleForDispatch(buffer: *const Buffer) ?backend_api.BufferHandle {
        return buffer.storage.handleForDispatch();
    }
};

const BufferReadiness = struct {
    fn readinessEvent(buffer: *const Buffer) Event {
        return buffer.ready_event;
    }

    fn markReady(buffer: *Buffer) void {
        buffer.ready_event.setReady();
    }

    fn failReadiness(buffer: *Buffer, message: []const u8) void {
        buffer.ready_event.setFailed(message);
    }

    fn chainReadyAfter(buffer: *Buffer, dependency: *Event) !void {
        try buffer.ensureUsable();
        buffer.ready_event = Event.pending();
        try dependency.chainTo(&buffer.ready_event);
    }
};

const BufferAdoption = struct {
    fn initOwned(allocator: std.mem.Allocator, backend: backend_api.Backend, element_type: BufferType, dims: []const i64, device: *const Device, memory: *Memory, shard_index: usize, byte_size: usize, backend_buffer: ?backend_api.BufferHandle, readiness: InitialReadiness) !*Buffer {
        const placement = try Placement.init(device, memory);
        return initOwnedWithPlacement(allocator, backend, element_type, dims, placement, shard_index, byte_size, backend_buffer, readiness);
    }

    fn initOwnedWithPlacement(allocator: std.mem.Allocator, backend: backend_api.Backend, element_type: BufferType, dims: []const i64, placement: Placement, shard_index: usize, byte_size: usize, backend_buffer: ?backend_api.BufferHandle, readiness: InitialReadiness) !*Buffer {
        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const descriptor = try Descriptor.init(allocator, element_type, dims, shard_index, byte_size);
        errdefer descriptor.deinit(allocator);

        buffer.* = .{
            .allocator = allocator,
            .descriptor = descriptor,
            .placement = placement,
            .storage = Storage.init(backend, backend_buffer),
            .ready_event = switch (readiness) {
                .ready => Event.ready(),
                .pending => Event.pending(),
            },
        };
        if (backend_buffer != null) return accountDeviceBytes(buffer);
        return buffer;
    }

    fn accountDeviceBytes(buffer: *Buffer) *Buffer {
        buffer.storage.account(buffer.placement.memory, buffer.descriptor.byte_size);
        return buffer;
    }

    fn releaseStorage(buffer: *Buffer) void {
        buffer.storage.release(buffer.placement.memory);
    }
};
