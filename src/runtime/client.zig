const std = @import("std");
const backend_api = @import("src/backend/mlx_metal");
const ir = @import("src/compiler/ir");

const async_host_transfer = @import("async_host_transfer.zig");
const buffer_mod = @import("buffer.zig");
const client_compile = @import("client_compile.zig");
const client_executable_context = @import("client_executable_context.zig");
const client_residency = @import("client_residency.zig");
const device_memory = @import("device_memory.zig");
const executable_mod = @import("executable.zig");
const executable_cache = @import("executable_cache.zig");

const Buffer = buffer_mod.Buffer;
const BufferCreateError = buffer_mod.BufferCreateError;
const BufferType = ir.BufferType;
const CompiledExecutable = executable_mod.CompiledExecutable;
const DeviceMemoryTopology = device_memory.DeviceMemoryTopology;
const Device = device_memory.Device;
const ExecutableResidencyCache = executable_cache.Residency;
const ExecutableCacheTrim = executable_cache.Trim;
const ExecutableContext = executable_mod.Context;
const MAX_DEVICES = device_memory.MAX_DEVICES;
const Memory = device_memory.Memory;
const AsyncHostTransfer = async_host_transfer.AsyncHostTransfer;

/// Opaque handle for an in-flight MLX async host-to-device transfer.
pub const AsyncHostToDeviceTransferHandle = async_host_transfer.Handle;

/// Errors returned when the backend starts an async host-to-device transfer.
pub const AsyncTransferBeginError = async_host_transfer.BeginError;

/// Errors returned when writing host bytes into an async transfer.
pub const AsyncTransferWriteError = async_host_transfer.WriteError;

/// Errors returned when finishing an async transfer into a runtime buffer.
pub const AsyncTransferFinishError = async_host_transfer.FinishError;

/// Options used when creating a Metal/MLX runtime client.
pub const ClientCreateOptions = struct {
    device_count: usize = 1,
    executable_cache_max_resident_bytes: ?u64 = null,
};

/// Creates a runtime client backed by the process-wide Metal/MLX backend.
pub fn createClient(allocator: std.mem.Allocator, options: ClientCreateOptions) !*Client {
    const client = try Client.init(allocator, backend_api.create(), options.device_count);
    if (options.executable_cache_max_resident_bytes) |max_resident_bytes| {
        client.setExecutableCacheMaxResidentBytes(max_resident_bytes);
    }
    return client;
}

/// Program bytes and compile options accepted by the runtime compile boundary.
pub const CompileProgram = client_compile.Program;

/// Errors surfaced by runtime compilation before PJRT ABI translation.
pub const CompileProgramError = client_compile.Error;

/// Owns runtime topology, buffer factories, executable cache, and compile scheduling.
pub const Client = struct {
    allocator: std.mem.Allocator,
    backend: backend_api.Backend,
    device_memory: DeviceMemoryTopology,
    executable_residency: ExecutableResidencyCache,
    async_host_transfer: AsyncHostTransfer,
    io: std.Io = std.Io.Threaded.global_single_threaded.io(),

    /// Initializes a client from concrete Metal/MLX device descriptors.
    pub fn init(allocator: std.mem.Allocator, backend: backend_api.Backend, device_count: usize) !*Client {
        if (device_count == 0 or device_count > MAX_DEVICES) return error.InvalidDeviceCount;

        const client = try allocator.create(Client);
        errdefer allocator.destroy(client);

        const descriptors = try backend.enumerateDevices(allocator, device_count);
        defer backend.releaseDeviceDescriptors(allocator, descriptors);
        const device_memory_topology = try DeviceMemoryTopology.initFromDescriptors(allocator, descriptors);
        errdefer device_memory_topology.deinit(allocator);

        client.* = .{
            .allocator = allocator,
            .backend = backend,
            .device_memory = device_memory_topology,
            .executable_residency = ExecutableResidencyCache.init(allocator),
            .async_host_transfer = AsyncHostTransfer.init(backend),
        };
        return client;
    }

    /// Releases all devices, memories, cached executables, and topology storage.
    pub fn deinit(self: *Client) void {
        self.executable_residency.deinit(self.backend);
        self.device_memory.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    /// Finds a device by stable PJRT device id.
    pub fn lookupDevice(self: *const Client, id: i32) ?*const Device {
        return self.device_memory.lookupDevice(id);
    }

    /// Finds an addressable device by backend-local hardware id.
    pub fn lookupAddressableDeviceByLocalHardwareId(self: *const Client, local_hardware_id: i32) ?*const Device {
        return self.device_memory.lookupAddressableDeviceByLocalHardwareId(local_hardware_id);
    }

    /// Finds a runtime memory by stable PJRT memory id.
    pub fn lookupMemory(self: *const Client, id: i32) ?*const Memory {
        return self.device_memory.lookupMemory(id);
    }

    /// Returns the number of addressable devices exposed by this client.
    pub fn deviceCount(self: *const Client) usize {
        return self.device_memory.deviceCount();
    }

    /// Returns runtime devices for read-only topology descriptions.
    pub fn topologyDevices(self: *const Client) []const Device {
        return self.device_memory.deviceSlice();
    }

    /// Returns runtime device handles in topology order for PJRT adapter lists.
    pub fn deviceHandles(self: *const Client) []const *Device {
        return self.device_memory.deviceHandleSlice();
    }

    /// Returns runtime memory handles in topology order for PJRT adapter lists.
    pub fn memoryHandles(self: *const Client) []const *Memory {
        return self.device_memory.memoryHandleSlice();
    }

    /// Returns the default addressable device for placement defaults.
    pub fn defaultDevice(self: *Client) *Device {
        return self.device_memory.defaultDevice();
    }

    /// Returns the default addressable memory for placement defaults.
    pub fn defaultMemory(self: *Client) *Memory {
        return self.device_memory.defaultMemory();
    }

    /// Returns a device's logical index in this client topology.
    pub fn deviceIndex(self: *const Client, device: *const Device) ?usize {
        return self.device_memory.deviceIndex(device);
    }

    /// Returns addressable device handles for a loaded executable.
    pub fn addressableDeviceHandlesForCount(self: *const Client, count: usize) []const *Device {
        return self.device_memory.addressableDeviceHandlesForCount(count);
    }

    /// Returns executable-cache counters without exposing cache entries.
    pub fn executableCacheStats(self: *const Client) executable_cache.Stats {
        return client_residency.statsSnapshot(self);
    }

    /// Test-only observation hooks for cache behavior owned by the client.
    pub const Testing = struct {
        /// Returns a snapshot of one executable-cache entry without exposing mutable cache storage.
        pub fn executableCacheEntrySnapshot(client: *const Client, fingerprint: []const u8) ?executable_cache.EntrySnapshot {
            return client_residency.entrySnapshot(client, fingerprint);
        }

        /// Overrides compile latency for one cache entry in focused cache-policy tests.
        pub fn setExecutableCompileLatency(client: *Client, fingerprint: []const u8, latency_us: u64) void {
            client_residency.setCompileLatencyForTest(client, fingerprint, latency_us);
        }
    };

    /// Sets the resident executable-cache budget and evicts idle entries as needed.
    pub fn setExecutableCacheMaxResidentBytes(self: *Client, max_resident_bytes: u64) void {
        client_residency.setMaxResidentBytes(self, max_resident_bytes);
    }

    /// Trims idle resident executables to make room for a device allocation.
    pub fn trimExecutableCacheForAllocation(self: *Client, memory: *Memory, allocation_bytes: usize) ExecutableCacheTrim {
        return client_residency.trimForAllocation(self, memory, allocation_bytes);
    }

    /// Returns the narrow execution context consumed by compiled compiled executables.
    pub fn executableContext(self: *Client) ExecutableContext {
        return client_executable_context.fromClient(self);
    }

    /// Imports host bytes into a device-resident runtime buffer.
    pub fn createHostBufferFromBytes(
        self: *Client,
        allocator: std.mem.Allocator,
        element_type: BufferType,
        dims: []const i64,
        device: *Device,
        memory: *Memory,
        shard_index: usize,
        src: []const u8,
    ) BufferCreateError!*Buffer {
        return Buffer.initHostCopyForBackend(allocator, self.backend, element_type, dims, device, memory, shard_index, src);
    }

    /// Allocates an uninitialized device-resident runtime buffer.
    pub fn createDeviceBuffer(
        self: *Client,
        allocator: std.mem.Allocator,
        element_type: BufferType,
        dims: []const i64,
        device: *Device,
        memory: *Memory,
        shard_index: usize,
    ) BufferCreateError!*Buffer {
        return Buffer.initDeviceAllocationForBackend(allocator, self.backend, element_type, dims, device, memory, shard_index);
    }

    /// Reserves a buffer whose backend storage will arrive from async transfer completion.
    pub fn createPendingBackendTransferBuffer(
        self: *Client,
        allocator: std.mem.Allocator,
        element_type: BufferType,
        dims: []const i64,
        device: *Device,
        memory: *Memory,
        shard_index: usize,
    ) BufferCreateError!*Buffer {
        return Buffer.initPendingBackendTransfer(allocator, self.backend, element_type, dims, device, memory, shard_index);
    }

    /// Starts backend-managed async host-to-device transfer for a future runtime buffer.
    pub fn beginAsyncHostToDeviceTransfer(
        self: *Client,
        device: *const Device,
        element_type: BufferType,
        dims: []const i64,
        byte_size: usize,
    ) AsyncTransferBeginError!AsyncHostToDeviceTransferHandle {
        return self.async_host_transfer.begin(device, element_type, dims, byte_size);
    }

    /// Destroys an async transfer handle that did not become buffer storage.
    pub fn destroyAsyncHostToDeviceTransfer(self: *Client, transfer: AsyncHostToDeviceTransferHandle) void {
        self.async_host_transfer.destroy(transfer);
    }

    /// Writes one byte segment into an in-flight async transfer.
    pub fn writeAsyncHostToDeviceTransfer(self: *Client, transfer: AsyncHostToDeviceTransferHandle, offset: usize, bytes: []const u8) AsyncTransferWriteError!void {
        try self.async_host_transfer.write(transfer, offset, bytes);
    }

    /// Installs completed async transfer storage into a pending runtime buffer.
    pub fn finishAsyncHostToDeviceTransfer(self: *Client, buffer: *Buffer, transfer: AsyncHostToDeviceTransferHandle) AsyncTransferFinishError!void {
        try self.async_host_transfer.finish(buffer, transfer);
    }

    /// Compiles program bytes into a resident executable plan and backend residency.
    pub fn compileProgram(
        self: *Client,
        allocator: std.mem.Allocator,
        program: CompileProgram,
        diagnostics: *std.Io.Writer,
    ) CompileProgramError!CompiledExecutable {
        return client_compile.compile(self, allocator, program, diagnostics);
    }
};

test "client records executable cache hits and buffer memory accounting" {
    const allocator = std.testing.allocator;
    const client = try Client.init(allocator, backend_api.create(), 1);
    defer client.deinit();

    try std.testing.expect(!try client_residency.recordCompile(client, "same-program"));
    try std.testing.expect(try client_residency.recordCompile(client, "same-program"));
    try std.testing.expectEqual(@as(u64, 1), client.executableCacheStats().misses);
    try std.testing.expectEqual(@as(u64, 1), client.executableCacheStats().hits);

    const dims = [_]i64{4};
    const data = [_]u8{ 1, 2, 3, 4 };
    const before = client.defaultMemory().stats.bytes_in_use;
    {
        const buffer = try client.createHostBufferFromBytes(allocator, .u8, &dims, client.defaultDevice(), client.defaultMemory(), 0, &data);
        defer buffer.deinit();
        try std.testing.expectEqual(before + data.len, client.defaultMemory().stats.bytes_in_use);
        try std.testing.expect(client.defaultMemory().stats.peak_bytes_in_use >= client.defaultMemory().stats.bytes_in_use);
        try std.testing.expect(client.defaultMemory().stats.host_to_device_bytes >= data.len);
    }
    try std.testing.expectEqual(before, client.defaultMemory().stats.bytes_in_use);
}
