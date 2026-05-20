const std = @import("std");

const runtime = @import("src/runtime");
const c = @import("c");
const abi = @import("pjrt_abi.zig");
const errors = @import("errors.zig");
const PjrtError = errors.Error;
const events = @import("events.zig");
const state = @import("state.zig");
const types = @import("types.zig");

const allocator = state.allocator;
const PjrtEvent = events.Event;
const ManagerHandle = abi.AsyncHostToDeviceTransferManager(AsyncHostToDeviceTransferManager);

const TransferIndex = struct {
    value: usize,

    fn fromPjrt(buffer_index: c_int, buffer_count: usize) !TransferIndex {
        if (buffer_index < 0) return error.InvalidArgument;
        const value: usize = @intCast(buffer_index);
        if (value >= buffer_count) return error.InvalidArgument;
        return .{ .value = value };
    }
};

const DataTransferRequest = struct {
    manager: *AsyncHostToDeviceTransferManager,
    index: usize,
    offset: usize,
    bytes: []const u8,
    is_last: bool,

    fn init(raw: *allowzero c.PJRT_AsyncHostToDeviceTransferManager_TransferData_Args) DecodeError!DataTransferRequest {
        const manager = AsyncHostToDeviceTransferManager.at(raw.transfer_manager);
        const index = manager.index(raw.buffer_index) catch return error.InvalidIndex;
        if (raw.offset < 0 or raw.transfer_size < 0) return error.NegativeRange;
        const offset: usize = @intCast(raw.offset);
        const transfer_size: usize = @intCast(raw.transfer_size);
        if (raw.data == null and transfer_size != 0) return error.NullData;
        manager.validateRange(index, offset, transfer_size) catch return error.RangeTooLarge;
        return .{
            .manager = manager,
            .index = index,
            .offset = offset,
            .bytes = abi.Slice.constBytes(raw.data, transfer_size) orelse return error.NullData,
            .is_last = raw.is_last_transfer,
        };
    }

    fn decodeError(err: DecodeError) ?*c.PJRT_Error {
        return switch (err) {
            error.InvalidIndex => PjrtError.invalidArgument("async transfer buffer index is out of range"),
            error.NegativeRange => PjrtError.invalidArgument("async transfer offset and size must be non-negative"),
            error.NullData => PjrtError.invalidArgument("async transfer data is null"),
            error.RangeTooLarge => PjrtError.invalidArgument("async transfer range exceeds buffer size"),
            error.SizeMismatch => PjrtError.invalidArgument("async literal size does not match target buffer"),
        };
    }

    const DecodeError = error{ InvalidIndex, NegativeRange, NullData, RangeTooLarge, SizeMismatch };
};

const LiteralTransferRequest = struct {
    manager: *AsyncHostToDeviceTransferManager,
    index: usize,
    bytes: []const u8,

    fn init(raw: *allowzero c.PJRT_AsyncHostToDeviceTransferManager_TransferLiteral_Args) DataTransferRequest.DecodeError!LiteralTransferRequest {
        const manager = AsyncHostToDeviceTransferManager.at(raw.transfer_manager);
        const index = manager.index(raw.buffer_index) catch return error.InvalidIndex;
        const dims = raw.shape_dims[0..raw.shape_num_dims];
        const byte_size = types.BufferType.denseByteSize(raw.shape_element_type, dims);
        if (byte_size != manager.bufferByteSize(index)) return error.SizeMismatch;
        if (raw.data == null and byte_size != 0) return error.NullData;
        return .{
            .manager = manager,
            .index = index,
            .bytes = abi.Slice.constBytes(raw.data, byte_size) orelse return error.NullData,
        };
    }
};

const CompletionEvent = struct {
    slot: *allowzero ?*c.PJRT_Event,

    fn create(slot: *allowzero ?*c.PJRT_Event) ?*c.PJRT_Error {
        slot.* = PjrtEvent.pending();
        if (slot.* == null) return PjrtError.resourceExhausted("failed to allocate async transfer event");
        return null;
    }

    fn at(slot: *allowzero ?*c.PJRT_Event) CompletionEvent {
        return .{ .slot = slot };
    }

    fn fail(self: CompletionEvent, message: []const u8) void {
        PjrtEvent.setFailed(self.slot.*, message);
    }

    fn ready(self: CompletionEvent) void {
        PjrtEvent.setReady(self.slot.*);
    }
};

pub const ShapeSpec = struct {
    element_type: runtime.BufferType,
    dims: []const i64,
    byte_size: usize,

    pub fn fromPjrt(raw: c.PJRT_ShapeSpec) ShapeSpec {
        const dims = raw.dims[0..raw.num_dims];
        return .{
            .element_type = types.BufferType.fromPjrt(raw.element_type),
            .dims = dims,
            .byte_size = types.BufferType.denseByteSize(raw.element_type, dims),
        };
    }
};

pub const AsyncHostToDeviceTransferManager = struct {
    allocator: std.mem.Allocator,
    client: *runtime.Client,
    device: *runtime.Device,
    memory: *runtime.Memory,
    buffers: []*runtime.Buffer,
    staging: [][]u8,
    backend_transfers: []?runtime.AsyncHostToDeviceTransferHandle,
    written: []usize,
    retrieved: []bool,
    completed: []bool,

    pub const Api = struct {
        pub const Destroy = ManagerDestroy.call;
        pub const TransferData = AsyncTransferData.call;
        pub const RetrieveBuffer = AsyncRetrieveBuffer.call;
        pub const Device = AsyncDevice.call;
        pub const BufferCount = AsyncBufferCount.call;
        pub const BufferSize = AsyncBufferSize.call;
        pub const SetBufferError = AsyncSetBufferError.call;
        pub const AddMetadata = AsyncAddMetadata.call;
        pub const TransferLiteral = AsyncTransferLiteral.call;
    };

    pub fn at(manager: ?*c.PJRT_AsyncHostToDeviceTransferManager) *AsyncHostToDeviceTransferManager {
        return ManagerHandle.view(manager);
    }

    pub fn handle(manager: *AsyncHostToDeviceTransferManager) *c.PJRT_AsyncHostToDeviceTransferManager {
        return ManagerHandle.handle(manager);
    }

    pub fn create(client: *runtime.Client, memory: *runtime.Memory, shape_specs: []const ShapeSpec) !*AsyncHostToDeviceTransferManager {
        if (memory.addressable_devices.len == 0) return error.InvalidArgument;
        const device = memory.addressable_devices[0];
        const shard_index = abi.Placement.deviceIndex(client, device) orelse return error.InvalidArgument;

        const manager = try allocator.create(AsyncHostToDeviceTransferManager);
        errdefer allocator.destroy(manager);

        const buffers = try allocator.alloc(*runtime.Buffer, shape_specs.len);
        errdefer allocator.free(buffers);
        const staging = try allocator.alloc([]u8, shape_specs.len);
        errdefer allocator.free(staging);
        const backend_transfers = try allocator.alloc(?runtime.AsyncHostToDeviceTransferHandle, shape_specs.len);
        errdefer allocator.free(backend_transfers);
        const written = try allocator.alloc(usize, shape_specs.len);
        errdefer allocator.free(written);
        const retrieved = try allocator.alloc(bool, shape_specs.len);
        errdefer allocator.free(retrieved);
        const completed = try allocator.alloc(bool, shape_specs.len);
        errdefer allocator.free(completed);

        @memset(written, 0);
        @memset(retrieved, false);
        @memset(completed, false);
        @memset(backend_transfers, null);

        var initialized: usize = 0;
        errdefer {
            for (staging[0..initialized]) |bytes| if (bytes.len != 0) allocator.free(bytes);
            for (backend_transfers[0..initialized]) |maybe_transfer| {
                if (maybe_transfer) |transfer| client.destroyAsyncHostToDeviceTransfer(transfer);
            }
            for (buffers[0..initialized]) |buffer| buffer.deinit();
        }

        for (shape_specs, 0..) |shape_spec, i| {
            backend_transfers[i] = client.beginAsyncHostToDeviceTransfer(device, shape_spec.element_type, shape_spec.dims, shape_spec.byte_size);
            if (backend_transfers[i] != null) {
                staging[i] = &.{};
                buffers[i] = try client.createPendingBackendTransferBuffer(
                    allocator,
                    shape_spec.element_type,
                    shape_spec.dims,
                    device,
                    memory,
                    shard_index,
                );
            } else {
                staging[i] = try allocator.alloc(u8, shape_spec.byte_size);
                @memset(staging[i], 0);
                buffers[i] = try client.createDeviceBuffer(
                    allocator,
                    shape_spec.element_type,
                    shape_spec.dims,
                    device,
                    memory,
                    shard_index,
                );
                buffers[i].ready_event = runtime.Event.pending();
            }
            initialized += 1;
        }

        manager.* = .{
            .allocator = allocator,
            .client = client,
            .device = device,
            .memory = memory,
            .buffers = buffers,
            .staging = staging,
            .backend_transfers = backend_transfers,
            .written = written,
            .retrieved = retrieved,
            .completed = completed,
        };
        return manager;
    }

    pub fn deinit(self: *AsyncHostToDeviceTransferManager) void {
        for (self.staging) |bytes| if (bytes.len != 0) self.allocator.free(bytes);
        for (self.backend_transfers) |maybe_transfer| {
            if (maybe_transfer) |transfer| self.client.destroyAsyncHostToDeviceTransfer(transfer);
        }
        for (self.buffers, self.retrieved) |buffer, was_retrieved| {
            if (!was_retrieved) buffer.deinit();
        }
        self.allocator.free(self.completed);
        self.allocator.free(self.retrieved);
        self.allocator.free(self.written);
        self.allocator.free(self.backend_transfers);
        self.allocator.free(self.staging);
        self.allocator.free(self.buffers);
        self.allocator.destroy(self);
    }

    pub fn index(self: *AsyncHostToDeviceTransferManager, buffer_index: c_int) !usize {
        return (try TransferIndex.fromPjrt(buffer_index, self.buffers.len)).value;
    }

    pub fn bufferByteSize(self: *const AsyncHostToDeviceTransferManager, i: usize) usize {
        return self.buffers[i].byte_size;
    }

    pub fn writeBytes(self: *AsyncHostToDeviceTransferManager, i: usize, offset: usize, bytes: []const u8) !void {
        if (bytes.len == 0) return;
        if (self.backend_transfers[i]) |transfer| {
            try self.client.writeAsyncHostToDeviceTransfer(transfer, offset, bytes);
        } else {
            try types.Bytes.copy(self.staging[i][offset .. offset + bytes.len], bytes);
        }
        self.written[i] = @max(self.written[i], offset + bytes.len);
    }

    pub fn validateRange(self: *const AsyncHostToDeviceTransferManager, i: usize, offset: usize, transfer_size: usize) !void {
        const buffer_size = self.bufferByteSize(i);
        if (offset > buffer_size or transfer_size > buffer_size - offset) return error.InvalidArgument;
    }

    pub fn finishIfComplete(self: *AsyncHostToDeviceTransferManager, i: usize) !void {
        if (self.written[i] < self.bufferByteSize(i)) return error.IncompleteTransfer;
        try self.finishBuffer(i);
    }

    pub fn failBuffer(self: *AsyncHostToDeviceTransferManager, i: usize, message: []const u8) void {
        self.buffers[i].ready_event.setFailed(message);
    }

    pub fn retrieve(self: *AsyncHostToDeviceTransferManager, i: usize) *runtime.Buffer {
        self.retrieved[i] = true;
        return self.buffers[i];
    }

    pub fn finishBuffer(self: *AsyncHostToDeviceTransferManager, i: usize) !void {
        if (self.completed[i]) return error.InvalidArgument;
        const buffer = self.buffers[i];
        _ = self.client.trimExecutableCacheForAllocation(self.memory, buffer.byte_size);
        if (self.backend_transfers[i]) |transfer| {
            self.backend_transfers[i] = null;
            try self.client.finishAsyncHostToDeviceTransfer(buffer, transfer);
            buffer.memory.stats.host_to_device_bytes += @intCast(buffer.byte_size);
        } else {
            try self.client.installStagedHostBuffer(buffer, self.staging[i]);
        }
        buffer.ready_event.setReady();
        self.completed[i] = true;
    }
};

const ManagerDestroy = struct {
    fn call(args: [*c]c.PJRT_AsyncHostToDeviceTransferManager_Destroy_Args) callconv(.c) ?*c.PJRT_Error {
        if (args[0].transfer_manager) |manager| AsyncHostToDeviceTransferManager.at(manager).deinit();
        return null;
    }
};

const AsyncTransferData = struct {
    fn call(args: [*c]c.PJRT_AsyncHostToDeviceTransferManager_TransferData_Args) callconv(.c) ?*c.PJRT_Error {
        const request = DataTransferRequest.init(&args[0]) catch |err| return DataTransferRequest.decodeError(err);
        const manager = request.manager;

        manager.writeBytes(request.index, request.offset, request.bytes) catch |err| {
            return switch (err) {
                error.BufferCopyFailed => PjrtError.internal("backend async transfer write failed"),
                else => PjrtError.internal("failed to write backend async transfer"),
            };
        };

        if (CompletionEvent.create(&args[0].done_with_h2d_transfer)) |err| return err;
        const completion = CompletionEvent.at(&args[0].done_with_h2d_transfer);

        if (request.is_last) {
            manager.finishIfComplete(request.index) catch |err| {
                const message = switch (err) {
                    error.IncompleteTransfer => "async transfer completed before full buffer was written",
                    else => "failed to install async transfer buffer",
                };
                completion.fail(message);
                manager.failBuffer(request.index, message);
                return switch (err) {
                    error.IncompleteTransfer => PjrtError.invalidArgument("async transfer completed before full buffer was written"),
                    error.ShapeMismatch => PjrtError.invalidArgument("async transfer data does not match buffer shape"),
                    error.BufferDeleted, error.BufferDonated => PjrtError.failedPrecondition("async transfer buffer is deleted or donated"),
                    error.UnsupportedRuntimeFeature, error.UnsupportedElementType => PjrtError.unimplemented("backend cannot import async transfer buffer"),
                    else => PjrtError.internal("failed to install async transfer buffer"),
                };
            };
        }

        completion.ready();
        return null;
    }
};

const AsyncTransferLiteral = struct {
    fn call(args: [*c]c.PJRT_AsyncHostToDeviceTransferManager_TransferLiteral_Args) callconv(.c) ?*c.PJRT_Error {
        const request = LiteralTransferRequest.init(&args[0]) catch |err| return DataTransferRequest.decodeError(err);
        const manager = request.manager;
        manager.writeBytes(request.index, 0, request.bytes) catch {
            return PjrtError.internal("failed to write backend async transfer literal");
        };
        if (CompletionEvent.create(&args[0].done_with_h2d_transfer)) |err| return err;
        const completion = CompletionEvent.at(&args[0].done_with_h2d_transfer);
        manager.finishBuffer(request.index) catch |err| {
            completion.fail("failed to install async transfer literal");
            manager.failBuffer(request.index, "failed to install async transfer literal");
            return switch (err) {
                error.ShapeMismatch => PjrtError.invalidArgument("async literal data does not match buffer shape"),
                error.BufferDeleted, error.BufferDonated => PjrtError.failedPrecondition("async literal buffer is deleted or donated"),
                error.UnsupportedRuntimeFeature, error.UnsupportedElementType => PjrtError.unimplemented("backend cannot import async transfer literal"),
                else => PjrtError.internal("failed to install async transfer literal"),
            };
        };
        completion.ready();
        return null;
    }
};

const AsyncRetrieveBuffer = struct {
    fn call(args: [*c]c.PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args) callconv(.c) ?*c.PJRT_Error {
        const manager = AsyncHostToDeviceTransferManager.at(args[0].transfer_manager);
        const i = manager.index(args[0].buffer_index) catch {
            return PjrtError.invalidArgument("async transfer buffer index is out of range");
        };
        args[0].buffer_out = abi.Buffer.handle(manager.retrieve(i));
        return null;
    }
};

const AsyncDevice = struct {
    fn call(args: [*c]c.PJRT_AsyncHostToDeviceTransferManager_Device_Args) callconv(.c) ?*c.PJRT_Error {
        const manager = AsyncHostToDeviceTransferManager.at(args[0].transfer_manager);
        args[0].device_out = abi.Device.handle(manager.device);
        return null;
    }
};

const AsyncBufferCount = struct {
    fn call(args: [*c]c.PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args) callconv(.c) ?*c.PJRT_Error {
        const manager = AsyncHostToDeviceTransferManager.at(args[0].transfer_manager);
        args[0].buffer_count = manager.buffers.len;
        return null;
    }
};

const AsyncBufferSize = struct {
    fn call(args: [*c]c.PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args) callconv(.c) ?*c.PJRT_Error {
        const manager = AsyncHostToDeviceTransferManager.at(args[0].transfer_manager);
        const i = manager.index(args[0].buffer_index) catch {
            return PjrtError.invalidArgument("async transfer buffer index is out of range");
        };
        args[0].buffer_size = manager.bufferByteSize(i);
        return null;
    }
};

const AsyncSetBufferError = struct {
    fn call(args: [*c]c.PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args) callconv(.c) ?*c.PJRT_Error {
        const manager = AsyncHostToDeviceTransferManager.at(args[0].transfer_manager);
        const i = manager.index(args[0].buffer_index) catch {
            return PjrtError.invalidArgument("async transfer buffer index is out of range");
        };
        const message = abi.Slice.bytes(args[0].error_message, args[0].error_message_size) orelse "async transfer buffer failed";
        manager.buffers[i].ready_event.setFailed(message);
        return null;
    }
};

const AsyncAddMetadata = struct {
    fn call(_: [*c]c.PJRT_AsyncHostToDeviceTransferManager_AddMetadata_Args) callconv(.c) ?*c.PJRT_Error {
        return null;
    }
};

pub const Api = AsyncHostToDeviceTransferManager.Api;
