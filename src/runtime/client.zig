const std = @import("std");
const backend_api = @import("backend_selection.zig");
const ir = @import("src/compiler/ir");

const async_host_transfer = @import("async_host_transfer.zig");
const buffer_mod = @import("buffer.zig");
const client_compile = @import("client_compile.zig");
const client_executable_context = @import("client_executable_context.zig");
const client_residency = @import("client_residency.zig");
const device_memory = @import("device_memory.zig");
const executable_cache = @import("executable_cache.zig");
const executable_mod = @import("executable.zig");

const AsyncHostTransfer = async_host_transfer.AsyncHostTransfer;
const Buffer = buffer_mod.Buffer;
const BufferCreateError = buffer_mod.BufferCreateError;
const BufferType = ir.BufferType;
const CompiledExecutable = executable_mod.CompiledExecutable;
const Device = device_memory.Device;
const DeviceMemoryTopology = device_memory.DeviceMemoryTopology;
const ExecutableCacheTrim = executable_cache.Trim;
const ExecutableContext = executable_mod.Context;
const ExecutableResidencyCache = executable_cache.Residency;
const ExecutableResidencyContext = client_residency.Context;
const MAX_DEVICES = device_memory.MAX_DEVICES;
const Memory = device_memory.Memory;

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
    const client = try Client.init(allocator, backend_api.createFromEnv(), options.device_count);
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
    executable_residency_context: ExecutableResidencyContext,
    async_host_transfer: AsyncHostTransfer,
    io: std.Io = std.Io.Threaded.global_single_threaded.io(),

    /// Initializes a client from concrete Metal/MLX device descriptors.
    pub const init = ClientLifecycle.init;
    /// Releases all devices, memories, cached executables, and topology storage.
    pub const deinit = ClientLifecycle.deinit;
    /// Finds a device by stable PJRT device id.
    pub const lookupDevice = ClientTopology.lookupDevice;
    /// Finds an addressable device by backend-local hardware id.
    pub const lookupAddressableDeviceByLocalHardwareId = ClientTopology.lookupAddressableDeviceByLocalHardwareId;
    /// Finds a runtime memory by stable PJRT memory id.
    pub const lookupMemory = ClientTopology.lookupMemory;
    /// Returns the number of addressable devices exposed by this client.
    pub const deviceCount = ClientTopology.deviceCount;
    /// Returns runtime devices for read-only topology descriptions.
    pub const topologyDevices = ClientTopology.topologyDevices;
    /// Returns runtime device handles in topology order for PJRT adapter lists.
    pub const deviceHandles = ClientTopology.deviceHandles;
    /// Returns runtime memory handles in topology order for PJRT adapter lists.
    pub const memoryHandles = ClientTopology.memoryHandles;
    /// Returns the default addressable device for placement defaults.
    pub const defaultDevice = ClientTopology.defaultDevice;
    /// Returns the default addressable memory for placement defaults.
    pub const defaultMemory = ClientTopology.defaultMemory;
    /// Returns a device's logical index in this client topology.
    pub const deviceIndex = ClientTopology.deviceIndex;
    /// Returns addressable device handles for a loaded executable.
    pub const addressableDeviceHandlesForCount = ClientTopology.addressableDeviceHandlesForCount;
    /// Returns executable-cache counters without exposing cache entries.
    pub const executableCacheStats = ClientResidency.executableCacheStats;
    /// Sets the resident executable-cache budget and evicts idle entries as needed.
    pub const setExecutableCacheMaxResidentBytes = ClientResidency.setExecutableCacheMaxResidentBytes;
    /// Trims idle resident executables to make room for a device allocation.
    pub const trimExecutableCacheForAllocation = ClientResidency.trimExecutableCacheForAllocation;
    /// Returns the narrow execution context consumed by compiled compiled executables.
    pub const executableContext = ClientExecutableContext.executableContext;
    /// Imports host bytes into a device-resident runtime buffer.
    pub const createHostBufferFromBytes = ClientBufferFactory.createHostBufferFromBytes;
    /// Allocates an uninitialized device-resident runtime buffer.
    pub const createDeviceBuffer = ClientBufferFactory.createDeviceBuffer;
    /// Reserves a buffer whose backend storage will arrive from async transfer completion.
    pub const createPendingBackendTransferBuffer = ClientBufferFactory.createPendingBackendTransferBuffer;
    /// Starts backend-managed async host-to-device transfer for a future runtime buffer.
    pub const beginAsyncHostToDeviceTransfer = ClientAsyncTransfers.beginAsyncHostToDeviceTransfer;
    /// Destroys an async transfer handle that did not become buffer storage.
    pub const destroyAsyncHostToDeviceTransfer = ClientAsyncTransfers.destroyAsyncHostToDeviceTransfer;
    /// Writes one byte segment into an in-flight async transfer.
    pub const writeAsyncHostToDeviceTransfer = ClientAsyncTransfers.writeAsyncHostToDeviceTransfer;
    /// Installs completed async transfer storage into a pending runtime buffer.
    pub const finishAsyncHostToDeviceTransfer = ClientAsyncTransfers.finishAsyncHostToDeviceTransfer;
    /// Compiles program bytes into a resident executable plan and backend residency.
    pub const compileProgram = ClientCompiler.compileProgram;

    /// Test-only observation hooks for cache behavior owned by the client.
    pub const Testing = struct {
        /// Returns a snapshot of one executable-cache entry without exposing mutable cache storage.
        pub fn executableCacheEntrySnapshot(client: *const Client, fingerprint: []const u8) ?executable_cache.EntrySnapshot {
            return client_residency.entrySnapshot(&client.executable_residency_context, fingerprint);
        }

        /// Overrides compile latency for one cache entry in focused cache-policy tests.
        pub fn setExecutableCompileLatency(client: *Client, fingerprint: []const u8, latency_us: u64) void {
            client_residency.setCompileLatencyForTest(&client.executable_residency_context, fingerprint, latency_us);
        }
    };
};

const ClientLifecycle = struct {
    fn init(allocator: std.mem.Allocator, backend: backend_api.Backend, device_count: usize) !*Client {
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
            .executable_residency_context = undefined,
            .async_host_transfer = AsyncHostTransfer.init(backend),
        };
        client.executable_residency_context = ExecutableResidencyContext.init(&client.backend, &client.device_memory, &client.executable_residency, &client.io);
        return client;
    }

    fn deinit(client: *Client) void {
        client.executable_residency.deinit(client.backend);
        client.device_memory.deinit(client.allocator);
        client.allocator.destroy(client);
    }
};

const ClientTopology = struct {
    fn lookupDevice(client: *const Client, id: i32) ?*const Device {
        return client.device_memory.lookupDevice(id);
    }

    fn lookupAddressableDeviceByLocalHardwareId(client: *const Client, local_hardware_id: i32) ?*const Device {
        return client.device_memory.lookupAddressableDeviceByLocalHardwareId(local_hardware_id);
    }

    fn lookupMemory(client: *const Client, id: i32) ?*const Memory {
        return client.device_memory.lookupMemory(id);
    }

    fn deviceCount(client: *const Client) usize {
        return client.device_memory.deviceCount();
    }

    fn topologyDevices(client: *const Client) []const Device {
        return client.device_memory.deviceSlice();
    }

    fn deviceHandles(client: *const Client) []const *Device {
        return client.device_memory.deviceHandleSlice();
    }

    fn memoryHandles(client: *const Client) []const *Memory {
        return client.device_memory.memoryHandleSlice();
    }

    fn defaultDevice(client: *Client) *Device {
        return client.device_memory.defaultDevice();
    }

    fn defaultMemory(client: *Client) *Memory {
        return client.device_memory.defaultMemory();
    }

    fn deviceIndex(client: *const Client, device: *const Device) ?usize {
        return client.device_memory.deviceIndex(device);
    }

    fn addressableDeviceHandlesForCount(client: *const Client, count: usize) []const *Device {
        return client.device_memory.addressableDeviceHandlesForCount(count);
    }
};

const ClientResidency = struct {
    fn executableCacheStats(client: *const Client) executable_cache.Stats {
        return client_residency.statsSnapshot(&client.executable_residency_context);
    }

    fn setExecutableCacheMaxResidentBytes(client: *Client, max_resident_bytes: u64) void {
        client_residency.setMaxResidentBytes(&client.executable_residency_context, max_resident_bytes);
    }

    fn trimExecutableCacheForAllocation(client: *Client, memory: *Memory, allocation_bytes: usize) ExecutableCacheTrim {
        return client_residency.trimForAllocation(&client.executable_residency_context, memory, allocation_bytes);
    }
};

const ClientExecutableContext = struct {
    fn executableContext(client: *Client) ExecutableContext {
        return client_executable_context.fromClient(client);
    }
};

const ClientBufferFactory = struct {
    fn createHostBufferFromBytes(client: *Client, allocator: std.mem.Allocator, element_type: BufferType, dims: []const i64, device: *Device, memory: *Memory, shard_index: usize, src: []const u8) BufferCreateError!*Buffer {
        return Buffer.initHostCopyForBackend(allocator, client.backend, element_type, dims, device, memory, shard_index, src);
    }

    fn createDeviceBuffer(client: *Client, allocator: std.mem.Allocator, element_type: BufferType, dims: []const i64, device: *Device, memory: *Memory, shard_index: usize) BufferCreateError!*Buffer {
        return Buffer.initDeviceAllocationForBackend(allocator, client.backend, element_type, dims, device, memory, shard_index);
    }

    fn createPendingBackendTransferBuffer(client: *Client, allocator: std.mem.Allocator, element_type: BufferType, dims: []const i64, device: *Device, memory: *Memory, shard_index: usize) BufferCreateError!*Buffer {
        return Buffer.initPendingBackendTransfer(allocator, client.backend, element_type, dims, device, memory, shard_index);
    }
};

const ClientAsyncTransfers = struct {
    fn beginAsyncHostToDeviceTransfer(client: *Client, device: *const Device, element_type: BufferType, dims: []const i64, byte_size: usize) AsyncTransferBeginError!AsyncHostToDeviceTransferHandle {
        return client.async_host_transfer.begin(device, element_type, dims, byte_size);
    }

    fn destroyAsyncHostToDeviceTransfer(client: *Client, transfer: AsyncHostToDeviceTransferHandle) void {
        client.async_host_transfer.destroy(transfer);
    }

    fn writeAsyncHostToDeviceTransfer(client: *Client, transfer: AsyncHostToDeviceTransferHandle, offset: usize, bytes: []const u8) AsyncTransferWriteError!void {
        try client.async_host_transfer.write(transfer, offset, bytes);
    }

    fn finishAsyncHostToDeviceTransfer(client: *Client, buffer: *Buffer, transfer: AsyncHostToDeviceTransferHandle) AsyncTransferFinishError!void {
        try client.async_host_transfer.finish(buffer, transfer);
    }
};

const ClientCompiler = struct {
    fn compileProgram(client: *Client, allocator: std.mem.Allocator, program: CompileProgram, diagnostics: *std.Io.Writer) CompileProgramError!CompiledExecutable {
        return client_compile.compile(client, allocator, program, diagnostics);
    }
};
