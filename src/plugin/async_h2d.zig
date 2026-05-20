const std = @import("std");

const c = @import("c");
const runtime = @import("src/runtime");

const abi = @import("pjrt_abi.zig");
const buffer_element = @import("buffer_element.zig");
const buffer_placement = @import("buffer_placement.zig");
const errors = @import("errors.zig");
const events = @import("events.zig");
const handles = @import("pjrt_handles.zig");
const plugin = @import("plugin.zig");

const TransferHandle = handles.AsyncHostToDeviceTransferManager(TransferManager);
const PjrtError = errors.Error;
const PjrtEvent = events.Event;

const TransferIndex = struct {
    value: usize,

    fn fromPjrt(buffer_index: c_int, buffer_count: usize) !TransferIndex {
        if (buffer_index < 0) return error.InvalidArgument;
        const value: usize = @intCast(buffer_index);
        if (value >= buffer_count) return error.InvalidArgument;
        return .{ .value = value };
    }
};

const ByteTransferRequest = struct {
    manager: *TransferManager,
    index: usize,
    offset: usize,
    bytes: []const u8,
    is_last: bool,

    fn init(raw: *allowzero c.PJRT_AsyncHostToDeviceTransferManager_TransferData_Args) DecodeError!ByteTransferRequest {
        const manager = TransferManager.at(raw.transfer_manager);
        const index = manager.index(raw.buffer_index) catch return error.InvalidIndex;
        if (raw.offset < 0 or raw.transfer_size < 0) return error.NegativeRange;
        const offset: usize = @intCast(raw.offset);
        const transfer_size: usize = @intCast(raw.transfer_size);
        if (raw.data == null and transfer_size != 0) return error.NullBytes;
        manager.validateRange(index, offset, transfer_size) catch return error.RangeTooLarge;
        return .{
            .manager = manager,
            .index = index,
            .offset = offset,
            .bytes = abi.Slice.constBytes(raw.data, transfer_size) orelse return error.NullBytes,
            .is_last = raw.is_last_transfer,
        };
    }

    fn decodeError(err: DecodeError) ?*c.PJRT_Error {
        return switch (err) {
            error.InvalidIndex => PjrtError.invalidArgument("async transfer buffer index is out of range"),
            error.NegativeRange => PjrtError.invalidArgument("async transfer offset and size must be non-negative"),
            error.NullBytes => PjrtError.invalidArgument("async transfer data is null"),
            error.NullDims => PjrtError.invalidArgument("async transfer shape dimensions are null"),
            error.RangeTooLarge => PjrtError.invalidArgument("async transfer range exceeds buffer size"),
            error.SizeMismatch => PjrtError.invalidArgument("async literal size does not match target buffer"),
        };
    }

    const DecodeError = error{ InvalidIndex, NegativeRange, NullBytes, NullDims, RangeTooLarge, SizeMismatch };
};

const LiteralTransferRequest = struct {
    manager: *TransferManager,
    index: usize,
    bytes: []const u8,

    fn init(raw: *allowzero c.PJRT_AsyncHostToDeviceTransferManager_TransferLiteral_Args) ByteTransferRequest.DecodeError!LiteralTransferRequest {
        const manager = TransferManager.at(raw.transfer_manager);
        const index = manager.index(raw.buffer_index) catch return error.InvalidIndex;
        const dims = abi.Slice.constList(i64, raw.shape_dims, raw.shape_num_dims) orelse return error.NullDims;
        const byte_size = buffer_element.ElementType.denseByteSize(raw.shape_element_type, dims);
        if (byte_size != manager.bufferByteSize(index)) return error.SizeMismatch;
        if (raw.data == null and byte_size != 0) return error.NullBytes;
        return .{
            .manager = manager,
            .index = index,
            .bytes = abi.Slice.constBytes(raw.data, byte_size) orelse return error.NullBytes,
        };
    }
};

const CompletionEvent = struct {
    slot: *allowzero ?*c.PJRT_Event,

    fn create(slot: *allowzero ?*c.PJRT_Event) ?*c.PJRT_Error {
        return PjrtEvent.Completion.pending(slot);
    }

    fn at(slot: *allowzero ?*c.PJRT_Event) CompletionEvent {
        return .{ .slot = slot };
    }

    fn fail(self: CompletionEvent, message: []const u8) void {
        PjrtEvent.Completion.at(self.slot).setFailed(message);
    }

    fn ready(self: CompletionEvent) void {
        PjrtEvent.Completion.at(self.slot).setReady();
    }
};

const AsyncTransferFailure = struct {
    fn write(err: runtime.AsyncTransferWriteError) ?*c.PJRT_Error {
        return switch (err) {
            error.BufferCopyFailed => PjrtError.internal("backend async transfer write failed"),
            error.InvalidDeviceCount,
            error.InvalidProgram,
            error.UnsupportedElementType,
            error.ShapeMismatch,
            error.BufferAllocationFailed,
            error.CommandSubmissionFailed,
            error.OutOfMemory,
            error.InvalidCustomCall,
            => PjrtError.internal("failed to write backend async transfer"),
        };
    }

    fn install(err: InstallError) ?*c.PJRT_Error {
        return switch (err) {
            error.IncompleteTransfer => PjrtError.invalidArgument("async transfer completed before full buffer was written"),
            error.InvalidArgument => PjrtError.invalidArgument("async transfer buffer was already completed"),
            error.ShapeMismatch => PjrtError.invalidArgument("async transfer data does not match buffer shape"),
            error.BufferDeleted, error.BufferDonated => PjrtError.failedPrecondition("async transfer buffer is deleted or donated"),
            error.UnsupportedElementType, error.UnsupportedRuntimeFeature => PjrtError.unimplemented("backend cannot import async transfer buffer"),
            error.BufferAllocationFailed, error.OutOfMemory => PjrtError.resourceExhausted("backend failed to allocate async transfer buffer"),
            error.InvalidDeviceCount,
            error.InvalidProgram,
            error.CommandSubmissionFailed,
            error.BufferCopyFailed,
            error.InvalidCustomCall,
            => PjrtError.internal("failed to install async transfer buffer"),
        };
    }

    fn installMessage(err: InstallError) []const u8 {
        if (err == error.IncompleteTransfer) return "async transfer completed before full buffer was written";
        return "failed to install async transfer buffer";
    }

    const InstallError = runtime.AsyncTransferFinishError || error{ IncompleteTransfer, InvalidArgument };
};

const ShapeSpec = struct {
    element_type: runtime.BufferType,
    dims: []const i64,
    byte_size: usize,

    fn fromPjrt(raw: c.PJRT_ShapeSpec) !ShapeSpec {
        const dims = abi.Slice.constList(i64, raw.dims, raw.num_dims) orelse return error.InvalidArgument;
        return .{
            .element_type = buffer_element.ElementType.fromPjrt(raw.element_type),
            .dims = dims,
            .byte_size = buffer_element.ElementType.denseByteSize(raw.element_type, dims),
        };
    }
};

/// PJRT async host-to-device transfer manager backed by MLX async device transfer.
pub const TransferManager = struct {
    allocator: std.mem.Allocator,
    client: *runtime.Client,
    device: *runtime.Device,
    memory: *runtime.Memory,
    buffers: []*runtime.Buffer,
    backend_transfers: []runtime.AsyncHostToDeviceTransferHandle,
    written: []usize,
    retrieved: []bool,
    completed: []bool,

    pub const Api = struct {
        pub const Destroy = TransferDestroy.call;
        pub const TransferData = TransferBytes.call;
        pub const RetrieveBuffer = AsyncRetrieveBuffer.call;
        pub const Device = AsyncDevice.call;
        pub const BufferCount = AsyncBufferCount.call;
        pub const BufferSize = AsyncBufferSize.call;
        pub const SetBufferError = AsyncSetBufferError.call;
        pub const AddMetadata = AsyncAddMetadata.call;
        pub const TransferLiteral = AsyncTransferLiteral.call;
    };

    fn at(manager: ?*c.PJRT_AsyncHostToDeviceTransferManager) *TransferManager {
        return TransferHandle.ref(manager);
    }

    fn handle(manager: *TransferManager) *c.PJRT_AsyncHostToDeviceTransferManager {
        return TransferHandle.handle(manager);
    }

    fn create(client: *runtime.Client, memory: *runtime.Memory, shape_specs: []const ShapeSpec) CreateError!*TransferManager {
        if (memory.addressable_devices.len == 0) return error.InvalidArgument;
        const device = memory.addressable_devices[0];
        const shard_index = buffer_placement.Placement.index(client, device) orelse return error.InvalidArgument;

        const manager = try plugin.allocator().create(TransferManager);
        errdefer plugin.allocator().destroy(manager);

        const buffers = try plugin.allocator().alloc(*runtime.Buffer, shape_specs.len);
        errdefer plugin.allocator().free(buffers);
        const backend_transfers = try plugin.allocator().alloc(runtime.AsyncHostToDeviceTransferHandle, shape_specs.len);
        errdefer plugin.allocator().free(backend_transfers);
        const written = try plugin.allocator().alloc(usize, shape_specs.len);
        errdefer plugin.allocator().free(written);
        const retrieved = try plugin.allocator().alloc(bool, shape_specs.len);
        errdefer plugin.allocator().free(retrieved);
        const completed = try plugin.allocator().alloc(bool, shape_specs.len);
        errdefer plugin.allocator().free(completed);

        @memset(written, 0);
        @memset(retrieved, false);
        @memset(completed, false);

        var initialized: usize = 0;
        errdefer {
            for (backend_transfers[0..initialized], completed[0..initialized]) |transfer, done| {
                if (!done) client.destroyAsyncHostToDeviceTransfer(transfer);
            }
            for (buffers[0..initialized]) |buffer| buffer.deinit();
        }

        for (shape_specs, 0..) |shape_spec, i| {
            const transfer = try client.beginAsyncHostToDeviceTransfer(device, shape_spec.element_type, shape_spec.dims, shape_spec.byte_size);
            backend_transfers[i] = transfer;
            buffers[i] = client.createPendingBackendTransferBuffer(
                plugin.allocator(),
                shape_spec.element_type,
                shape_spec.dims,
                device,
                memory,
                shard_index,
            ) catch |err| {
                client.destroyAsyncHostToDeviceTransfer(transfer);
                return err;
            };
            initialized += 1;
        }

        manager.* = .{
            .allocator = plugin.allocator(),
            .client = client,
            .device = device,
            .memory = memory,
            .buffers = buffers,
            .backend_transfers = backend_transfers,
            .written = written,
            .retrieved = retrieved,
            .completed = completed,
        };
        return manager;
    }

    fn deinit(self: *TransferManager) void {
        for (self.backend_transfers, self.completed, self.buffers) |transfer, done, buffer| {
            if (!done) {
                buffer.ready_event.setFailed("async transfer manager destroyed before completion");
                self.client.destroyAsyncHostToDeviceTransfer(transfer);
            }
        }
        for (self.buffers, self.retrieved) |buffer, was_retrieved| {
            if (!was_retrieved) buffer.deinit();
        }
        self.allocator.free(self.completed);
        self.allocator.free(self.retrieved);
        self.allocator.free(self.written);
        self.allocator.free(self.backend_transfers);
        self.allocator.free(self.buffers);
        self.allocator.destroy(self);
    }

    fn index(self: *TransferManager, buffer_index: c_int) !usize {
        return (try TransferIndex.fromPjrt(buffer_index, self.buffers.len)).value;
    }

    fn bufferByteSize(self: *const TransferManager, i: usize) usize {
        return self.buffers[i].byte_size;
    }

    fn writeBytes(self: *TransferManager, i: usize, offset: usize, bytes: []const u8) runtime.AsyncTransferWriteError!void {
        if (bytes.len == 0) return;
        try self.client.writeAsyncHostToDeviceTransfer(self.backend_transfers[i], offset, bytes);
        self.written[i] = @max(self.written[i], offset + bytes.len);
    }

    fn validateRange(self: *const TransferManager, i: usize, offset: usize, transfer_size: usize) !void {
        const buffer_size = self.bufferByteSize(i);
        if (offset > buffer_size or transfer_size > buffer_size - offset) return error.InvalidArgument;
    }

    fn finishIfComplete(self: *TransferManager, i: usize) AsyncTransferFailure.InstallError!void {
        if (self.written[i] < self.bufferByteSize(i)) return error.IncompleteTransfer;
        try self.finishBuffer(i);
    }

    fn failBuffer(self: *TransferManager, i: usize, message: []const u8) void {
        self.buffers[i].ready_event.setFailed(message);
    }

    fn retrieve(self: *TransferManager, i: usize) *runtime.Buffer {
        self.retrieved[i] = true;
        return self.buffers[i];
    }

    fn finishBuffer(self: *TransferManager, i: usize) AsyncTransferFailure.InstallError!void {
        if (self.completed[i]) return error.InvalidArgument;
        const buffer = self.buffers[i];
        _ = self.client.trimExecutableCacheForAllocation(self.memory, buffer.byte_size);
        try self.client.finishAsyncHostToDeviceTransfer(buffer, self.backend_transfers[i]);
        buffer.memory.stats.host_to_device_bytes += @intCast(buffer.byte_size);
        buffer.ready_event.setReady();
        self.completed[i] = true;
    }

    const CreateError = std.mem.Allocator.Error || runtime.PendingBackendTransferBufferError || runtime.AsyncTransferBeginError;
};

const TransferDestroy = struct {
    fn call(args: [*c]c.PJRT_AsyncHostToDeviceTransferManager_Destroy_Args) callconv(.c) ?*c.PJRT_Error {
        if (args[0].transfer_manager) |manager| TransferManager.at(manager).deinit();
        return null;
    }
};

const TransferBytes = struct {
    fn call(args: [*c]c.PJRT_AsyncHostToDeviceTransferManager_TransferData_Args) callconv(.c) ?*c.PJRT_Error {
        const request = ByteTransferRequest.init(&args[0]) catch |err| return ByteTransferRequest.decodeError(err);
        const manager = request.manager;

        manager.writeBytes(request.index, request.offset, request.bytes) catch |err| {
            return AsyncTransferFailure.write(err);
        };

        if (CompletionEvent.create(&args[0].done_with_h2d_transfer)) |err| return err;
        const completion = CompletionEvent.at(&args[0].done_with_h2d_transfer);

        if (request.is_last) {
            manager.finishIfComplete(request.index) catch |err| {
                const message = AsyncTransferFailure.installMessage(err);
                completion.fail(message);
                manager.failBuffer(request.index, message);
                return AsyncTransferFailure.install(err);
            };
        }

        completion.ready();
        return null;
    }
};

const AsyncTransferLiteral = struct {
    fn call(args: [*c]c.PJRT_AsyncHostToDeviceTransferManager_TransferLiteral_Args) callconv(.c) ?*c.PJRT_Error {
        const request = LiteralTransferRequest.init(&args[0]) catch |err| return ByteTransferRequest.decodeError(err);
        const manager = request.manager;
        manager.writeBytes(request.index, 0, request.bytes) catch {
            return PjrtError.internal("failed to write backend async transfer literal");
        };
        if (CompletionEvent.create(&args[0].done_with_h2d_transfer)) |err| return err;
        const completion = CompletionEvent.at(&args[0].done_with_h2d_transfer);
        manager.finishBuffer(request.index) catch |err| {
            completion.fail("failed to install async transfer literal");
            manager.failBuffer(request.index, "failed to install async transfer literal");
            return AsyncTransferFailure.install(err);
        };
        completion.ready();
        return null;
    }
};

const AsyncRetrieveBuffer = struct {
    fn call(args: [*c]c.PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args) callconv(.c) ?*c.PJRT_Error {
        const manager = TransferManager.at(args[0].transfer_manager);
        const i = manager.index(args[0].buffer_index) catch {
            return PjrtError.invalidArgument("async transfer buffer index is out of range");
        };
        args[0].buffer_out = handles.Buffer.handle(manager.retrieve(i));
        return null;
    }
};

const AsyncDevice = struct {
    fn call(args: [*c]c.PJRT_AsyncHostToDeviceTransferManager_Device_Args) callconv(.c) ?*c.PJRT_Error {
        const manager = TransferManager.at(args[0].transfer_manager);
        args[0].device_out = handles.Device.handle(manager.device);
        return null;
    }
};

const AsyncBufferCount = struct {
    fn call(args: [*c]c.PJRT_AsyncHostToDeviceTransferManager_BufferCount_Args) callconv(.c) ?*c.PJRT_Error {
        const manager = TransferManager.at(args[0].transfer_manager);
        args[0].buffer_count = manager.buffers.len;
        return null;
    }
};

const AsyncBufferSize = struct {
    fn call(args: [*c]c.PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args) callconv(.c) ?*c.PJRT_Error {
        const manager = TransferManager.at(args[0].transfer_manager);
        const i = manager.index(args[0].buffer_index) catch {
            return PjrtError.invalidArgument("async transfer buffer index is out of range");
        };
        args[0].buffer_size = manager.bufferByteSize(i);
        return null;
    }
};

const AsyncSetBufferError = struct {
    fn call(args: [*c]c.PJRT_AsyncHostToDeviceTransferManager_SetBufferError_Args) callconv(.c) ?*c.PJRT_Error {
        const manager = TransferManager.at(args[0].transfer_manager);
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

/// Creates a PJRT async host-to-device transfer manager handle from PJRT shapes.
pub fn create(client: *runtime.Client, memory: *runtime.Memory, raw_shape_specs: []const c.PJRT_ShapeSpec) (TransferManager.CreateError || error{InvalidShapeSpec})!*c.PJRT_AsyncHostToDeviceTransferManager {
    const shape_specs = try plugin.allocator().alloc(ShapeSpec, raw_shape_specs.len);
    defer plugin.allocator().free(shape_specs);
    for (raw_shape_specs, shape_specs) |raw, *shape_spec| {
        shape_spec.* = ShapeSpec.fromPjrt(raw) catch return error.InvalidShapeSpec;
    }
    return TransferManager.handle(try TransferManager.create(client, memory, shape_specs));
}
