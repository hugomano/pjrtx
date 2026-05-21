const std = @import("std");
const backend_api = @import("src/backend/mlx_metal");
const ir = @import("src/compiler/ir");

const async_host_transfer = @import("async_host_transfer.zig");
const buffer_mod = @import("buffer.zig");
const compile_pipeline = @import("compile_pipeline.zig");
const device_memory = @import("device_memory.zig");
const executable_mod = @import("executable.zig");
const executable_cache = @import("executable_cache.zig");
const executable_fingerprint = @import("executable_fingerprint.zig");

const Buffer = buffer_mod.Buffer;
const BufferCreateError = buffer_mod.BufferCreateError;
const BufferType = ir.BufferType;
const CachedBackendExecutable = executable_cache.Retained;
const CompiledExecutable = executable_mod.CompiledExecutable;
const DeviceMemoryTopology = device_memory.DeviceMemoryTopology;
const Device = device_memory.Device;
const ExecutableResidencyCache = executable_cache.Residency;
const ExecutableCacheLease = executable_cache.Lease;
const ExecutableCacheTrim = executable_cache.Trim;
const ExecutableContext = executable_mod.Context;
const ExecutablePlan = ir.ExecutablePlan;
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
pub const CompileProgram = compile_pipeline.Program;

/// Errors surfaced by runtime compilation before PJRT ABI translation.
pub const CompileProgramError = compile_pipeline.Error;

fn clientFromExecutableContext(user_context: *anyopaque) *Client {
    return @ptrCast(@alignCast(user_context));
}

fn acquireCachedBackendExecutableForContext(
    user_context: *anyopaque,
    allocator: std.mem.Allocator,
    fingerprint: []const u8,
    plan: *const ir.ExecutablePlan,
    device_local_hardware_ids: []const i32,
) executable_mod.CacheAcquireError!?CachedBackendExecutable {
    return clientFromExecutableContext(user_context).acquireCachedBackendExecutable(allocator, fingerprint, plan, device_local_hardware_ids);
}

fn releaseCachedBackendExecutableForContext(user_context: *anyopaque, lease: ExecutableCacheLease) void {
    clientFromExecutableContext(user_context).releaseCachedBackendExecutable(lease);
}

fn trimExecutableCacheForContext(user_context: *anyopaque, memory: *Memory, allocation_bytes: usize) ExecutableCacheTrim {
    return clientFromExecutableContext(user_context).trimExecutableCacheForAllocation(memory, allocation_bytes);
}

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

    /// Records a compile request and returns whether the fingerprint was seen before.
    fn recordExecutableCompile(self: *Client, fingerprint: []const u8) !bool {
        return self.executable_residency.recordCompile(self.io, self.device_memory.memorySlice(), fingerprint);
    }

    /// Returns executable-cache counters without exposing cache entries.
    pub fn executableCacheStats(self: *const Client) executable_cache.Stats {
        return self.executable_residency.statsSnapshot();
    }

    /// Test-only observation hooks for cache behavior owned by the client.
    pub const Testing = struct {
        /// Returns a snapshot of one executable-cache entry without exposing mutable cache storage.
        pub fn executableCacheEntrySnapshot(client: *const Client, fingerprint: []const u8) ?executable_cache.EntrySnapshot {
            return client.executable_residency.entrySnapshot(fingerprint);
        }

        /// Overrides compile latency for one cache entry in focused cache-policy tests.
        pub fn setExecutableCompileLatency(client: *Client, fingerprint: []const u8, latency_us: u64) void {
            client.executable_residency.setCompileLatencyForTest(fingerprint, latency_us);
        }
    };

    /// Sets the resident executable-cache budget and evicts idle entries as needed.
    pub fn setExecutableCacheMaxResidentBytes(self: *Client, max_resident_bytes: u64) void {
        self.executable_residency.setMaxResidentBytes(self.io, self.backend, self.device_memory.memorySlice(), max_resident_bytes);
    }

    /// Trims idle resident executables to make room for a device allocation.
    pub fn trimExecutableCacheForAllocation(self: *Client, memory: *Memory, allocation_bytes: usize) ExecutableCacheTrim {
        return self.executable_residency.trimForAllocation(self.io, self.backend, self.device_memory.memorySlice(), memory, allocation_bytes);
    }

    /// Acquires a retained resident backend executable for a compiled executable.
    fn acquireCachedBackendExecutable(
        self: *Client,
        allocator: std.mem.Allocator,
        fingerprint: []const u8,
        plan: *const ir.ExecutablePlan,
        device_local_hardware_ids: []const i32,
    ) !?CachedBackendExecutable {
        return self.executable_residency.acquireBackendExecutable(
            self.io,
            self.backend,
            allocator,
            self.device_memory.memorySlice(),
            self.device_memory.executableResidencyMemory(device_local_hardware_ids),
            fingerprint,
            plan,
            device_local_hardware_ids,
        );
    }

    /// Releases a retained executable-cache entry after residency teardown.
    fn releaseCachedBackendExecutable(self: *Client, lease: ExecutableCacheLease) void {
        self.executable_residency.release(self.io, self.backend, self.device_memory.memorySlice(), lease);
    }

    /// Returns the narrow execution context consumed by compiled compiled executables.
    pub fn executableContext(self: *Client) ExecutableContext {
        return .{
            .backend = self.backend,
            .devices = self.device_memory.deviceSlice(),
            .user_context = self,
            .acquire_cached_executable = acquireCachedBackendExecutableForContext,
            .release_cached_executable = releaseCachedBackendExecutableForContext,
            .trim_executable_cache = trimExecutableCacheForContext,
        };
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
        var parsed_options = try compile_pipeline.parseOptions(allocator, self.device_memory.deviceSlice(), program.compile_options, diagnostics);
        defer parsed_options.deinit(allocator);

        var analysis = try compile_pipeline.analyzeProgram(allocator, program, diagnostics);
        defer if (analysis) |*owned_analysis| owned_analysis.deinit();

        var plan = try compile_pipeline.makeExecutablePlan(allocator, parsed_options.options, analysis);
        var plan_moved = false;
        errdefer if (!plan_moved) plan.deinit();

        try compile_pipeline.verifyPlan(allocator, plan, diagnostics);

        const optimized_program = try compile_pipeline.retainOptimizedProgram(allocator, analysis);
        errdefer allocator.free(optimized_program);

        const fingerprint = executable_fingerprint.alloc(allocator, self.backend, &self.device_memory, optimized_program, &plan) catch return error.OutOfMemory;
        errdefer allocator.free(fingerprint);

        const cache_hit = self.recordExecutableCompile(fingerprint) catch return error.Internal;

        const plan_ptr = allocator.create(ExecutablePlan) catch return error.OutOfMemory;
        plan_ptr.* = plan;
        plan_moved = true;
        errdefer {
            plan_ptr.deinit();
            allocator.destroy(plan_ptr);
        }

        return CompiledExecutable.initResident(allocator, self.executableContext(), plan_ptr, optimized_program, fingerprint, cache_hit, diagnostics) catch |err| switch (err) {
            error.UnsupportedRuntimeFeature => return error.UnsupportedRuntimeFeature,
            error.OutOfMemory => return error.OutOfMemory,
            error.Internal => return error.Internal,
        };
    }
};

test "client records executable cache hits and buffer memory accounting" {
    const allocator = std.testing.allocator;
    const client = try Client.init(allocator, backend_api.create(), 1);
    defer client.deinit();

    try std.testing.expect(!try client.recordExecutableCompile("same-program"));
    try std.testing.expect(try client.recordExecutableCompile("same-program"));
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
const ClientTestSupport = struct {
    fn shardingPlan(allocator: std.mem.Allocator, assignment: []const i32) !ir.ShardingPlan {
        return .{
            .kind = .replicated,
            .mesh_name = try allocator.dupe(u8, ""),
            .device_assignment = try allocator.dupe(i32, assignment),
        };
    }

    fn constantU8ExecutablePlan(allocator: std.mem.Allocator, assignment: []const i32, literal: []const u8) !ir.ExecutablePlan {
        var values = try allocator.alloc(ir.Value, 1);
        errdefer allocator.free(values);

        const dims = [_]i64{@intCast(literal.len)};
        const owned_dims = try allocator.dupe(i64, &dims);
        errdefer allocator.free(owned_dims);
        values[0] = .{
            .id = .{ .index = 0 },
            .role = .instruction_result,
            .descriptor = .{
                .element_type = .u8,
                .dims = owned_dims,
                .device_id = assignment[0],
                .memory_id = assignment[0],
                .shard_index = 0,
            },
        };

        var output_shardings = try allocator.alloc(ir.ShardingPlan, 1);
        errdefer allocator.free(output_shardings);
        output_shardings[0] = try shardingPlan(allocator, assignment);
        errdefer {
            allocator.free(output_shardings[0].mesh_name);
            allocator.free(output_shardings[0].device_assignment);
        }

        const output_ids = try allocator.dupe(ir.ValueId, &.{.{ .index = 0 }});
        errdefer allocator.free(output_ids);

        const literal_copy = try allocator.dupe(u8, literal);
        errdefer allocator.free(literal_copy);

        const instruction_outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 0 }});
        errdefer allocator.free(instruction_outputs);

        const instructions = try allocator.dupe(ir.PlanInstruction, &.{.{
            .kind = .constant,
            .outputs = instruction_outputs,
            .literal = literal_copy,
        }});
        errdefer allocator.free(instructions);

        return .{
            .allocator = allocator,
            .options = .{
                .num_replicas = 1,
                .num_partitions = 1,
                .device_assignment = try allocator.dupe(i32, assignment),
            },
            .values = values,
            .parameter_shardings = try allocator.alloc(ir.ShardingPlan, 0),
            .output_shardings = output_shardings,
            .output_ids = output_ids,
            .instructions = instructions,
        };
    }

    fn addU8ExecutablePlan(allocator: std.mem.Allocator, assignment: []const i32, dims: []const i64) !ir.ExecutablePlan {
        const values = try allocator.alloc(ir.Value, 3);
        errdefer allocator.free(values);

        for (values, 0..) |*value, i| {
            value.* = .{
                .id = .{ .index = @intCast(i) },
                .role = if (i < 2) .parameter else .instruction_result,
                .descriptor = .{
                    .element_type = .u8,
                    .dims = try allocator.dupe(i64, dims),
                    .device_id = assignment[0],
                    .memory_id = assignment[0],
                    .shard_index = 0,
                },
            };
        }

        var parameter_shardings = try allocator.alloc(ir.ShardingPlan, 2);
        errdefer allocator.free(parameter_shardings);
        parameter_shardings[0] = try shardingPlan(allocator, assignment);
        errdefer {
            allocator.free(parameter_shardings[0].mesh_name);
            allocator.free(parameter_shardings[0].device_assignment);
        }
        parameter_shardings[1] = try shardingPlan(allocator, assignment);
        errdefer {
            allocator.free(parameter_shardings[1].mesh_name);
            allocator.free(parameter_shardings[1].device_assignment);
        }

        var output_shardings = try allocator.alloc(ir.ShardingPlan, 1);
        errdefer allocator.free(output_shardings);
        output_shardings[0] = try shardingPlan(allocator, assignment);
        errdefer {
            allocator.free(output_shardings[0].mesh_name);
            allocator.free(output_shardings[0].device_assignment);
        }

        const output_ids = try allocator.dupe(ir.ValueId, &.{.{ .index = 2 }});
        errdefer allocator.free(output_ids);

        const inputs = try allocator.dupe(ir.ValueId, &.{ .{ .index = 0 }, .{ .index = 1 } });
        errdefer allocator.free(inputs);

        const outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 2 }});
        errdefer allocator.free(outputs);

        const instructions = try allocator.dupe(ir.PlanInstruction, &.{.{
            .kind = .add,
            .inputs = inputs,
            .outputs = outputs,
        }});
        errdefer allocator.free(instructions);

        return .{
            .allocator = allocator,
            .options = .{
                .num_replicas = 1,
                .num_partitions = 1,
                .device_assignment = try allocator.dupe(i32, assignment),
            },
            .values = values,
            .parameter_shardings = parameter_shardings,
            .output_shardings = output_shardings,
            .output_ids = output_ids,
            .instructions = instructions,
        };
    }

    fn cacheEntrySnapshot(client: *const Client, fingerprint: []const u8) !executable_cache.EntrySnapshot {
        return Client.Testing.executableCacheEntrySnapshot(client, fingerprint) orelse error.TestUnexpectedResult;
    }
};

test "client executable cache reuses backend executable handles" {
    const client = try Client.init(std.testing.allocator, backend_api.create(), 1);
    defer client.deinit();
    const allocator = std.testing.allocator;
    const assignment = [_]i32{0};
    const dims = [_]i64{4};

    var plan = try ClientTestSupport.addU8ExecutablePlan(allocator, &assignment, &dims);
    defer plan.deinit();

    const fingerprint = "cached-add";
    try std.testing.expect(!try client.recordExecutableCompile(fingerprint));
    var first = try CompiledExecutable.Testing.initBorrowedResident(allocator, client.executableContext(), &plan, .{
        .cache_fingerprint = fingerprint,
    });
    try std.testing.expect(first.hasResidentBackendExecutable());
    try std.testing.expect(!first.backendExecutableCacheReused());

    var entry = try ClientTestSupport.cacheEntrySnapshot(client, fingerprint);
    try std.testing.expect(entry.resident);
    try std.testing.expectEqual(@as(usize, 1), entry.reference_count);

    try std.testing.expect(try client.recordExecutableCompile(fingerprint));
    var second = try CompiledExecutable.Testing.initBorrowedResident(allocator, client.executableContext(), &plan, .{
        .cache_fingerprint = fingerprint,
    });
    try std.testing.expect(second.hasResidentBackendExecutable());
    try std.testing.expect(second.backendExecutableCacheReused());
    try std.testing.expectEqual(first.backendExecutableForDispatch().?, second.backendExecutableForDispatch().?);
    try std.testing.expectEqual(@as(u64, 1), client.executableCacheStats().misses);
    try std.testing.expectEqual(@as(u64, 1), client.executableCacheStats().hits);
    entry = try ClientTestSupport.cacheEntrySnapshot(client, fingerprint);
    try std.testing.expectEqual(@as(usize, 2), entry.reference_count);

    CompiledExecutable.Testing.deinitBorrowedResident(&second);
    entry = try ClientTestSupport.cacheEntrySnapshot(client, fingerprint);
    try std.testing.expectEqual(@as(usize, 1), entry.reference_count);
    CompiledExecutable.Testing.deinitBorrowedResident(&first);
    entry = try ClientTestSupport.cacheEntrySnapshot(client, fingerprint);
    try std.testing.expectEqual(@as(usize, 0), entry.reference_count);
    try std.testing.expect(entry.resident);
}

test "executable cache evicts idle resident backend executables under byte limit" {
    const client = try Client.init(std.testing.allocator, backend_api.create(), 1);
    defer client.deinit();
    const allocator = std.testing.allocator;
    const assignment = [_]i32{0};
    const literal = [_]u8{ 1, 2, 3, 4 };

    var plan = try ClientTestSupport.constantU8ExecutablePlan(allocator, &assignment, &literal);
    defer plan.deinit();

    client.setExecutableCacheMaxResidentBytes(0);

    const fingerprint = "evict-idle-constant";
    try std.testing.expect(!try client.recordExecutableCompile(fingerprint));
    var first = try CompiledExecutable.Testing.initBorrowedResident(allocator, client.executableContext(), &plan, .{
        .cache_fingerprint = fingerprint,
    });
    var entry = try ClientTestSupport.cacheEntrySnapshot(client, fingerprint);
    try std.testing.expect(first.hasResidentBackendExecutable());
    try std.testing.expect(entry.resident);
    try std.testing.expect(entry.resident_bytes >= literal.len);
    try std.testing.expectEqual(@as(u64, 1), client.executableCacheStats().resident_entries);
    try std.testing.expect(client.executableCacheStats().resident_bytes >= literal.len);
    try std.testing.expect(client.defaultMemory().stats.totalBytesInUse() >= literal.len);
    try std.testing.expect(client.defaultMemory().stats.peakTotalBytesInUse() >= client.defaultMemory().stats.totalBytesInUse());
    try std.testing.expectEqual(@as(usize, 1), entry.reference_count);

    CompiledExecutable.Testing.deinitBorrowedResident(&first);
    entry = try ClientTestSupport.cacheEntrySnapshot(client, fingerprint);
    try std.testing.expectEqual(@as(usize, 0), entry.reference_count);
    try std.testing.expect(!entry.resident);
    try std.testing.expectEqual(@as(u64, 0), client.executableCacheStats().resident_entries);
    try std.testing.expectEqual(@as(u64, 0), client.executableCacheStats().resident_bytes);
    try std.testing.expectEqual(@as(u64, 0), client.defaultMemory().stats.totalBytesInUse());
    try std.testing.expectEqual(@as(u64, 1), client.executableCacheStats().evictions);
    try std.testing.expectEqual(@as(u64, 1), client.defaultMemory().stats.executable_cache_evictions);

    try std.testing.expect(try client.recordExecutableCompile(fingerprint));
    var second = try CompiledExecutable.Testing.initBorrowedResident(allocator, client.executableContext(), &plan, .{
        .cache_fingerprint = fingerprint,
    });
    defer CompiledExecutable.Testing.deinitBorrowedResident(&second);
    try std.testing.expect(second.hasResidentBackendExecutable());
    try std.testing.expect(!second.backendExecutableCacheReused());
    entry = try ClientTestSupport.cacheEntrySnapshot(client, fingerprint);
    try std.testing.expectEqual(@as(usize, 1), entry.reference_count);
    try std.testing.expect(entry.resident);
    try std.testing.expectEqual(@as(u64, 1), client.executableCacheStats().hits);
}

test "executable cache evicts largest idle resident executable before older smaller entries" {
    const client = try Client.init(std.testing.allocator, backend_api.create(), 1);
    defer client.deinit();
    const allocator = std.testing.allocator;
    const assignment = [_]i32{0};

    var small_plan = try ClientTestSupport.constantU8ExecutablePlan(allocator, &assignment, &.{ 1, 2, 3, 4 });
    defer small_plan.deinit();
    var large_plan = try ClientTestSupport.constantU8ExecutablePlan(allocator, &assignment, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer large_plan.deinit();

    try std.testing.expect(!try client.recordExecutableCompile("small-constant"));
    var small_executable = try CompiledExecutable.Testing.initBorrowedResident(allocator, client.executableContext(), &small_plan, .{
        .cache_fingerprint = "small-constant",
    });
    var small_entry = try ClientTestSupport.cacheEntrySnapshot(client, "small-constant");
    const small_resident_bytes = small_entry.resident_bytes;
    try std.testing.expect(small_resident_bytes >= 4);
    CompiledExecutable.Testing.deinitBorrowedResident(&small_executable);
    small_entry = try ClientTestSupport.cacheEntrySnapshot(client, "small-constant");
    try std.testing.expectEqual(@as(usize, 0), small_entry.reference_count);
    try std.testing.expect(small_entry.resident);

    try std.testing.expect(!try client.recordExecutableCompile("large-constant"));
    var large_executable = try CompiledExecutable.Testing.initBorrowedResident(allocator, client.executableContext(), &large_plan, .{
        .cache_fingerprint = "large-constant",
    });
    var large_entry = try ClientTestSupport.cacheEntrySnapshot(client, "large-constant");
    const large_resident_bytes = large_entry.resident_bytes;
    try std.testing.expect(large_resident_bytes > small_resident_bytes);
    CompiledExecutable.Testing.deinitBorrowedResident(&large_executable);
    large_entry = try ClientTestSupport.cacheEntrySnapshot(client, "large-constant");
    try std.testing.expectEqual(@as(usize, 0), large_entry.reference_count);
    try std.testing.expect(large_entry.resident);

    client.setExecutableCacheMaxResidentBytes(small_resident_bytes);
    small_entry = try ClientTestSupport.cacheEntrySnapshot(client, "small-constant");
    large_entry = try ClientTestSupport.cacheEntrySnapshot(client, "large-constant");
    try std.testing.expect(small_entry.resident);
    try std.testing.expect(!large_entry.resident);
    try std.testing.expectEqual(@as(u64, 1), client.executableCacheStats().resident_entries);
    try std.testing.expectEqual(small_resident_bytes, client.executableCacheStats().resident_bytes);
    try std.testing.expectEqual(@as(u64, 1), client.executableCacheStats().evictions);
    try std.testing.expectEqual(large_resident_bytes, client.executableCacheStats().evicted_resident_bytes);
}

test "client trims idle executable cache for tracked memory pressure" {
    const client = try Client.init(std.testing.allocator, backend_api.create(), 1);
    defer client.deinit();
    const allocator = std.testing.allocator;
    const assignment = [_]i32{0};

    var small_plan = try ClientTestSupport.constantU8ExecutablePlan(allocator, &assignment, &.{ 1, 2, 3, 4 });
    defer small_plan.deinit();
    var large_plan = try ClientTestSupport.constantU8ExecutablePlan(allocator, &assignment, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer large_plan.deinit();

    try std.testing.expect(!try client.recordExecutableCompile("pressure-small-constant"));
    var small_executable = try CompiledExecutable.Testing.initBorrowedResident(allocator, client.executableContext(), &small_plan, .{
        .cache_fingerprint = "pressure-small-constant",
    });
    var small_entry = try ClientTestSupport.cacheEntrySnapshot(client, "pressure-small-constant");
    const small_resident_bytes = small_entry.resident_bytes;
    CompiledExecutable.Testing.deinitBorrowedResident(&small_executable);

    try std.testing.expect(!try client.recordExecutableCompile("pressure-large-constant"));
    var large_executable = try CompiledExecutable.Testing.initBorrowedResident(allocator, client.executableContext(), &large_plan, .{
        .cache_fingerprint = "pressure-large-constant",
    });
    var large_entry = try ClientTestSupport.cacheEntrySnapshot(client, "pressure-large-constant");
    const large_resident_bytes = large_entry.resident_bytes;
    try std.testing.expect(large_resident_bytes > small_resident_bytes);
    CompiledExecutable.Testing.deinitBorrowedResident(&large_executable);

    client.defaultMemory().stats.capacity_bytes = small_resident_bytes;
    const trim = client.trimExecutableCacheForAllocation(client.defaultMemory(), 0);
    try std.testing.expectEqual(@as(u64, 0), trim.requested_bytes);
    try std.testing.expectEqual(small_resident_bytes, trim.target_resident_bytes);
    try std.testing.expectEqual(large_resident_bytes, trim.freed_bytes);
    try std.testing.expectEqual(@as(u64, 1), trim.evicted_entries);
    try std.testing.expect(!trim.still_over_capacity);
    small_entry = try ClientTestSupport.cacheEntrySnapshot(client, "pressure-small-constant");
    large_entry = try ClientTestSupport.cacheEntrySnapshot(client, "pressure-large-constant");
    try std.testing.expect(small_entry.resident);
    try std.testing.expect(!large_entry.resident);
    try std.testing.expectEqual(small_resident_bytes, client.executableCacheStats().resident_bytes);
    try std.testing.expectEqual(@as(u64, 1), client.executableCacheStats().pressure_trim_requests);
    try std.testing.expectEqual(large_resident_bytes, client.executableCacheStats().pressure_trimmed_bytes);
    try std.testing.expectEqual(@as(u64, 0), client.executableCacheStats().pressure_trim_failures);
    try std.testing.expectEqual(@as(u64, 1), client.defaultMemory().stats.executable_cache_pressure_trims);
    try std.testing.expectEqual(large_resident_bytes, client.defaultMemory().stats.executable_cache_pressure_trimmed_bytes);
    try std.testing.expectEqual(@as(u64, 0), client.defaultMemory().stats.executable_cache_pressure_trim_failures);
}

test "compiling resident executable trims idle cache under memory pressure" {
    const client = try Client.init(std.testing.allocator, backend_api.create(), 1);
    defer client.deinit();
    const allocator = std.testing.allocator;
    const assignment = [_]i32{0};

    var idle_plan = try ClientTestSupport.constantU8ExecutablePlan(allocator, &assignment, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer idle_plan.deinit();
    var new_plan = try ClientTestSupport.constantU8ExecutablePlan(allocator, &assignment, &.{ 9, 10, 11, 12 });
    defer new_plan.deinit();

    try std.testing.expect(!try client.recordExecutableCompile("compile-pressure-idle-constant"));
    var idle_executable = try CompiledExecutable.Testing.initBorrowedResident(allocator, client.executableContext(), &idle_plan, .{
        .cache_fingerprint = "compile-pressure-idle-constant",
    });
    var idle_entry = try ClientTestSupport.cacheEntrySnapshot(client, "compile-pressure-idle-constant");
    const idle_resident_bytes = idle_entry.resident_bytes;
    try std.testing.expect(idle_resident_bytes >= 8);
    CompiledExecutable.Testing.deinitBorrowedResident(&idle_executable);
    idle_entry = try ClientTestSupport.cacheEntrySnapshot(client, "compile-pressure-idle-constant");
    try std.testing.expectEqual(@as(usize, 0), idle_entry.reference_count);
    try std.testing.expect(idle_entry.resident);

    client.defaultMemory().stats.capacity_bytes = idle_resident_bytes;

    try std.testing.expect(!try client.recordExecutableCompile("compile-pressure-new-constant"));
    var new_executable = try CompiledExecutable.Testing.initBorrowedResident(allocator, client.executableContext(), &new_plan, .{
        .cache_fingerprint = "compile-pressure-new-constant",
    });
    defer CompiledExecutable.Testing.deinitBorrowedResident(&new_executable);
    const new_entry = try ClientTestSupport.cacheEntrySnapshot(client, "compile-pressure-new-constant");
    try std.testing.expect(new_executable.hasResidentBackendExecutable());
    try std.testing.expect(new_entry.resident);
    try std.testing.expect(new_entry.resident_bytes >= 4);
    try std.testing.expect(new_entry.resident_bytes <= idle_resident_bytes);
    idle_entry = try ClientTestSupport.cacheEntrySnapshot(client, "compile-pressure-idle-constant");
    try std.testing.expect(!idle_entry.resident);
    try std.testing.expectEqual(new_entry.resident_bytes, new_executable.compileCacheTrim().requested_bytes);
    try std.testing.expectEqual(idle_resident_bytes - new_entry.resident_bytes, new_executable.compileCacheTrim().target_resident_bytes);
    try std.testing.expectEqual(idle_resident_bytes, new_executable.compileCacheTrim().freed_bytes);
    try std.testing.expectEqual(@as(u64, 1), new_executable.compileCacheTrim().evicted_entries);
    try std.testing.expectEqual(@as(u64, 0), new_executable.compileCacheTrim().remaining_resident_bytes);
    try std.testing.expect(!new_executable.compileCacheTrim().still_over_capacity);
    try std.testing.expectEqual(@as(u64, 1), client.executableCacheStats().pressure_trim_requests);
    try std.testing.expectEqual(idle_resident_bytes, client.executableCacheStats().pressure_trimmed_bytes);
    try std.testing.expectEqual(@as(u64, 0), client.executableCacheStats().pressure_trim_failures);
    try std.testing.expectEqual(new_entry.resident_bytes, client.executableCacheStats().resident_bytes);
    try std.testing.expectEqual(@as(u64, 1), client.defaultMemory().stats.executable_cache_pressure_trims);
    try std.testing.expectEqual(idle_resident_bytes, client.defaultMemory().stats.executable_cache_pressure_trimmed_bytes);
}

test "executable cache preserves more expensive equal-size resident executable" {
    const client = try Client.init(std.testing.allocator, backend_api.create(), 1);
    defer client.deinit();
    const allocator = std.testing.allocator;
    const assignment = [_]i32{0};

    var cheap_plan = try ClientTestSupport.constantU8ExecutablePlan(allocator, &assignment, &.{ 1, 2, 3, 4 });
    defer cheap_plan.deinit();
    var expensive_plan = try ClientTestSupport.constantU8ExecutablePlan(allocator, &assignment, &.{ 5, 6, 7, 8 });
    defer expensive_plan.deinit();

    try std.testing.expect(!try client.recordExecutableCompile("cheap-constant"));
    var cheap_executable = try CompiledExecutable.Testing.initBorrowedResident(allocator, client.executableContext(), &cheap_plan, .{
        .cache_fingerprint = "cheap-constant",
    });
    var cheap_entry = try ClientTestSupport.cacheEntrySnapshot(client, "cheap-constant");
    const resident_bytes = cheap_entry.resident_bytes;
    CompiledExecutable.Testing.deinitBorrowedResident(&cheap_executable);
    Client.Testing.setExecutableCompileLatency(client, "cheap-constant", 10);

    try std.testing.expect(!try client.recordExecutableCompile("expensive-constant"));
    var expensive_executable = try CompiledExecutable.Testing.initBorrowedResident(allocator, client.executableContext(), &expensive_plan, .{
        .cache_fingerprint = "expensive-constant",
    });
    var expensive_entry = try ClientTestSupport.cacheEntrySnapshot(client, "expensive-constant");
    try std.testing.expectEqual(resident_bytes, expensive_entry.resident_bytes);
    CompiledExecutable.Testing.deinitBorrowedResident(&expensive_executable);
    Client.Testing.setExecutableCompileLatency(client, "expensive-constant", 1000);

    try std.testing.expectEqual(@as(u64, 2), client.executableCacheStats().compile_latency_samples);
    try std.testing.expect(client.executableCacheStats().compile_latency_us_total > 0);
    try std.testing.expect(client.executableCacheStats().compile_latency_us_peak > 0);

    client.setExecutableCacheMaxResidentBytes(resident_bytes);
    cheap_entry = try ClientTestSupport.cacheEntrySnapshot(client, "cheap-constant");
    expensive_entry = try ClientTestSupport.cacheEntrySnapshot(client, "expensive-constant");
    try std.testing.expect(!cheap_entry.resident);
    try std.testing.expect(expensive_entry.resident);
    try std.testing.expectEqual(@as(u64, 1), client.executableCacheStats().resident_entries);
    try std.testing.expectEqual(resident_bytes, client.executableCacheStats().resident_bytes);
    try std.testing.expectEqual(@as(u64, 1), client.executableCacheStats().evictions);
    try std.testing.expectEqual(resident_bytes, client.executableCacheStats().evicted_resident_bytes);
}
