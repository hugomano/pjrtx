const backend_api = @import("src/backend/mlx_metal");

const device_memory = @import("device_memory.zig");

const Memory = device_memory.Memory;

/// Owns backend buffer storage and memory accounting for a runtime buffer.
pub const Storage = struct {
    backend: backend_api.Backend,
    handle: ?backend_api.BufferHandle = null,
    accounted_bytes: usize = 0,

    /// Creates storage from an optional backend handle without taking accounting yet.
    pub fn init(backend: backend_api.Backend, handle: ?backend_api.BufferHandle) Storage {
        return .{ .backend = backend, .handle = handle };
    }

    /// Returns whether a backend device allocation is currently attached.
    pub fn hasBackendStorage(self: Storage) bool {
        return self.handle != null;
    }

    /// Returns the backend handle for runtime dispatch without transferring ownership.
    pub fn handleForDispatch(self: Storage) ?backend_api.BufferHandle {
        return self.handle;
    }

    /// Accounts this storage against its memory placement.
    pub fn account(self: *Storage, memory: *Memory, byte_size: usize) void {
        self.accounted_bytes = byte_size;
        memory.stats.retain(self.accounted_bytes);
    }

    /// Releases backend storage and memory accounting.
    pub fn release(self: *Storage, memory: *Memory) void {
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
    pub fn takeForDonationAlias(self: *Storage, memory: *Memory) !backend_api.BufferHandle {
        const backend_buffer = self.handle orelse return error.UnsupportedRuntimeFeature;
        self.handle = null;
        if (self.accounted_bytes != 0) {
            memory.stats.release(self.accounted_bytes);
            self.accounted_bytes = 0;
        }
        return backend_buffer;
    }

    /// Replaces backend storage and refreshes memory accounting.
    pub fn replace(self: *Storage, memory: *Memory, byte_size: usize, backend_buffer: backend_api.BufferHandle) void {
        self.release(memory);
        self.handle = backend_buffer;
        self.account(memory, byte_size);
    }
};
