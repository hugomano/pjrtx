const backend_api = @import("backend_selection.zig");
const ir = @import("src/compiler/ir");

const buffer_mod = @import("buffer.zig");
const device_memory = @import("device_memory.zig");

const Buffer = buffer_mod.Buffer;
const BufferType = ir.BufferType;
const Device = device_memory.Device;

/// Opaque handle for an in-flight MLX async host-to-device transfer.
pub const Handle = backend_api.AsyncHostToDeviceTransferHandle;

/// Errors returned when the backend starts an async host-to-device transfer.
pub const BeginError = backend_api.Error || error{UnsupportedRuntimeFeature};

/// Errors returned when writing host bytes into an async transfer.
pub const WriteError = backend_api.Error;

/// Errors returned when finishing an async transfer into a runtime buffer.
pub const FinishError = backend_api.Error || error{ UnsupportedRuntimeFeature, BufferDeleted, BufferDonated };

/// Owns runtime async host-to-device transfer operations for the concrete backend.
pub const AsyncHostTransfer = struct {
    backend: backend_api.Backend,

    /// Creates an async-transfer owner bound to the concrete Metal/MLX backend.
    pub fn init(backend: backend_api.Backend) AsyncHostTransfer {
        return .{ .backend = backend };
    }

    /// Starts backend-managed async host-to-device transfer for a future runtime buffer.
    pub fn begin(
        self: AsyncHostTransfer,
        device: *const Device,
        element_type: BufferType,
        dims: []const i64,
        byte_size: usize,
    ) BeginError!Handle {
        return try self.backend.beginAsyncHostToDeviceTransfer(device.local_hardware_id, element_type, dims, byte_size) orelse error.UnsupportedRuntimeFeature;
    }

    /// Destroys an async transfer handle that did not become buffer storage.
    pub fn destroy(self: AsyncHostTransfer, transfer: Handle) void {
        self.backend.destroyAsyncHostToDeviceTransfer(transfer);
    }

    /// Writes one byte segment into an in-flight async transfer.
    pub fn write(self: AsyncHostTransfer, transfer: Handle, offset: usize, bytes: []const u8) WriteError!void {
        try self.backend.writeAsyncHostToDeviceTransfer(transfer, offset, bytes);
    }

    /// Installs completed async transfer storage into a pending runtime buffer.
    pub fn finish(self: AsyncHostTransfer, buffer: *Buffer, transfer: Handle) FinishError!void {
        const backend_buffer = try self.backend.finishAsyncHostToDeviceTransfer(transfer) orelse return error.UnsupportedRuntimeFeature;
        errdefer self.backend.destroyBuffer(backend_buffer);
        try buffer.replaceBackendStorage(backend_buffer);
    }
};
