const std = @import("std");
const backend_api = @import("src/backend/mlx_metal");
const compiler = @import("src/compiler");
const ir = @import("src/compiler/ir");

const io = std.Io.Threaded.global_single_threaded.io();

const buffer_mod = @import("buffer.zig");
const compile_options = @import("compile_options.zig");
const device_memory = @import("device_memory.zig");
const executable_mod = @import("executable.zig");
const executable_cache = @import("executable_cache.zig");

const Buffer = buffer_mod.Buffer;
const BufferCreateError = buffer_mod.BufferCreateError;
const BufferType = ir.BufferType;
const CachedBackendExecutable = executable_cache.Retained;
const CompileOptions = ir.CompileOptions;
const CompiledExecutable = executable_mod.CompiledExecutable;
const Device = device_memory.Device;
const ExecutableCache = executable_cache.Cache;
const ExecutableCacheEntry = executable_cache.Entry;
const ExecutableCacheTrim = executable_cache.Trim;
const ExecutableContext = executable_mod.Context;
const ExecutableGraph = executable_mod.ExecutableGraph;
const ExecutablePlan = ir.ExecutablePlan;
const MAX_DEVICES = device_memory.MAX_DEVICES;
const Memory = device_memory.Memory;
const PlanInstruction = ir.PlanInstruction;
const ShardingPlan = ir.ShardingPlan;
const Topology = device_memory.Topology;

/// Opaque handle for an in-flight MLX async host-to-device transfer.
pub const AsyncHostToDeviceTransferHandle = backend_api.AsyncHostToDeviceTransferHandle;

/// Errors returned when the backend starts an async host-to-device transfer.
pub const AsyncTransferBeginError = backend_api.Error || error{UnsupportedRuntimeFeature};

/// Errors returned when writing host bytes into an async transfer.
pub const AsyncTransferWriteError = backend_api.Error;

/// Errors returned when finishing an async transfer into a runtime buffer.
pub const AsyncTransferFinishError = backend_api.Error || error{ UnsupportedRuntimeFeature, BufferDeleted, BufferDonated };

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
pub const CompileProgram = struct {
    format: []const u8 = "",
    code: []const u8 = &.{},
    compile_options: []const u8 = &.{},
};

/// Errors surfaced by runtime compilation before PJRT ABI translation.
pub const CompileProgramError = error{
    InvalidOptions,
    OptionsRequireMoreDevices,
    UnknownDevice,
    UnsupportedProgram,
    InvalidProgram,
    InvalidExecutablePlan,
    UnsupportedRuntimeFeature,
    OutOfMemory,
    Internal,
};

fn writeDiagnostic(writer: *std.Io.Writer, comptime fmt: []const u8, args: anytype) void {
    writer.print(fmt, args) catch {};
}

fn parseCompileOptions(
    allocator: std.mem.Allocator,
    client: *const Client,
    text: []const u8,
    parsed_options: *bool,
    diagnostics: *std.Io.Writer,
) CompileProgramError!CompileOptions {
    var options: CompileOptions = .{
        .num_partitions = @intCast(client.devices.len),
    };
    if (text.len != 0) {
        if (compile_options.isPjrtxText(text)) {
            var options_reader: std.Io.Reader = .fixed(text);
            options = compiler.parseTextCompileOptionsFromReader(allocator, &options_reader) catch {
                writeDiagnostic(diagnostics, "invalid PjRTx text compile options", .{});
                return error.InvalidOptions;
            };
        } else {
            options = compile_options.parseXlaProto(allocator, text) catch {
                writeDiagnostic(diagnostics, "invalid XLA CompileOptionsProto", .{});
                return error.InvalidOptions;
            };
        }
        parsed_options.* = true;
    }
    if (options.numDevices() > client.devices.len) {
        writeDiagnostic(diagnostics, "compile options require more devices than the client exposes", .{});
        return error.OptionsRequireMoreDevices;
    }
    for (options.device_assignment) |device_id| {
        if (client.lookupDevice(device_id) == null) {
            writeDiagnostic(diagnostics, "compile options reference an unknown device id", .{});
            return error.UnknownDevice;
        }
    }
    return options;
}

fn analyzeProgram(
    allocator: std.mem.Allocator,
    program: CompileProgram,
    diagnostics: *std.Io.Writer,
) CompileProgramError!?compiler.ModuleAnalysis {
    if (program.code.len == 0) return null;
    var module_reader: std.Io.Reader = .fixed(program.code);
    return compiler.analyzeProgramFromReader(allocator, program.format, &module_reader, diagnostics) catch |err| switch (err) {
        error.UnsupportedOp, error.UnsupportedSharding, error.UnsupportedProgramEncoding, error.GspmdNotEnabled, error.InvalidManualComputation => error.UnsupportedProgram,
        error.UnsupportedProgramFormat, error.InvalidStablehloModule => error.InvalidProgram,
        error.OutOfMemory => error.OutOfMemory,
        else => error.Internal,
    };
}

fn makeExecutablePlan(
    allocator: std.mem.Allocator,
    options: CompileOptions,
    analysis: ?compiler.ModuleAnalysis,
) CompileProgramError!ExecutablePlan {
    return if (analysis) |owned_analysis|
        compiler.makeExecutablePlan(allocator, options, owned_analysis) catch |err| switch (err) {
            error.OutOfMemory => error.OutOfMemory,
        }
    else
        compiler.makeReplicatedPlan(allocator, options, 1, 1) catch |err| switch (err) {
            error.OutOfMemory => error.OutOfMemory,
        };
}

fn verifyPlan(
    allocator: std.mem.Allocator,
    plan: ExecutablePlan,
    diagnostics: *std.Io.Writer,
) CompileProgramError!void {
    compiler.verifyExecutablePlan(allocator, plan, diagnostics) catch |err| switch (err) {
        error.InvalidExecutablePlan => return error.InvalidExecutablePlan,
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.Internal,
    };
}

fn updateOptionalI64Slice(hasher: *std.hash.Wyhash, values: ?[]const i64) void {
    const present = values != null;
    hasher.update(std.mem.asBytes(&present));
    if (values) |slice| hasher.update(std.mem.sliceAsBytes(slice));
}

fn updateOptionalBytes(hasher: *std.hash.Wyhash, bytes: ?[]const u8) void {
    const present = bytes != null;
    hasher.update(std.mem.asBytes(&present));
    if (bytes) |slice| hasher.update(slice);
}

fn updateShardingFingerprint(hasher: *std.hash.Wyhash, shardings: []const ShardingPlan) void {
    hasher.update(std.mem.asBytes(&shardings.len));
    for (shardings) |sharding| {
        hasher.update(std.mem.asBytes(&sharding.kind));
        hasher.update(sharding.mesh_name);
        hasher.update(std.mem.sliceAsBytes(sharding.device_assignment));
    }
}

fn updateInstructionFingerprint(hasher: *std.hash.Wyhash, instruction: PlanInstruction) void {
    hasher.update(std.mem.asBytes(&instruction.kind));
    hasher.update(std.mem.sliceAsBytes(instruction.inputs));
    hasher.update(std.mem.sliceAsBytes(instruction.outputs));
    hasher.update(std.mem.sliceAsBytes(instruction.region_ids));
    updateOptionalI64Slice(hasher, instruction.dims);
    updateOptionalI64Slice(hasher, instruction.permutation);
    updateOptionalI64Slice(hasher, instruction.broadcast_dimensions);
    updateOptionalI64Slice(hasher, instruction.start_indices);
    updateOptionalI64Slice(hasher, instruction.limit_indices);
    updateOptionalI64Slice(hasher, instruction.strides);
    updateOptionalI64Slice(hasher, instruction.slice_sizes);
    updateOptionalI64Slice(hasher, instruction.edge_padding_low);
    updateOptionalI64Slice(hasher, instruction.edge_padding_high);
    updateOptionalI64Slice(hasher, instruction.interior_padding);
    updateOptionalI64Slice(hasher, instruction.offset_dims);
    updateOptionalI64Slice(hasher, instruction.collapsed_slice_dims);
    updateOptionalI64Slice(hasher, instruction.operand_batching_dims);
    updateOptionalI64Slice(hasher, instruction.start_indices_batching_dims);
    updateOptionalI64Slice(hasher, instruction.start_index_map);
    updateOptionalI64Slice(hasher, instruction.update_window_dims);
    updateOptionalI64Slice(hasher, instruction.inserted_window_dims);
    updateOptionalI64Slice(hasher, instruction.input_batching_dims);
    updateOptionalI64Slice(hasher, instruction.scatter_indices_batching_dims);
    updateOptionalI64Slice(hasher, instruction.scatter_dims_to_operand_dims);
    hasher.update(std.mem.asBytes(&instruction.index_vector_dim));
    hasher.update(std.mem.asBytes(&instruction.scatter_update_kind));
    hasher.update(std.mem.asBytes(&instruction.dimension));
    hasher.update(std.mem.asBytes(&instruction.top_k_k));
    hasher.update(std.mem.asBytes(&instruction.iota_dimension));
    hasher.update(std.mem.asBytes(&instruction.fft_kind));
    updateOptionalI64Slice(hasher, instruction.dimensions);
    hasher.update(std.mem.asBytes(&instruction.tuple_index));
    hasher.update(std.mem.asBytes(&instruction.lower));
    hasher.update(std.mem.asBytes(&instruction.triangular_left_side));
    hasher.update(std.mem.asBytes(&instruction.triangular_lower));
    hasher.update(std.mem.asBytes(&instruction.triangular_unit_diagonal));
    hasher.update(std.mem.asBytes(&instruction.triangular_transpose));
    updateOptionalBytes(hasher, instruction.custom_call_target);
    updateOptionalI64Slice(hasher, instruction.reduce_dimensions);
    updateOptionalI64Slice(hasher, instruction.lhs_batch_dimensions);
    updateOptionalI64Slice(hasher, instruction.rhs_batch_dimensions);
    updateOptionalI64Slice(hasher, instruction.lhs_contracting_dimensions);
    updateOptionalI64Slice(hasher, instruction.rhs_contracting_dimensions);
    hasher.update(std.mem.asBytes(&instruction.compare_direction));
    updateOptionalBytes(hasher, instruction.literal);
}

fn updateTargetDeviceFingerprint(hasher: *std.hash.Wyhash, client: *const Client, plan: *const ExecutablePlan) void {
    const device_count = plan.options.numDevices();
    hasher.update(std.mem.asBytes(&device_count));
    for (0..device_count) |i| {
        const device_id = if (plan.options.device_assignment.len != 0) plan.options.device_assignment[i] else client.devices[i].id;
        hasher.update(std.mem.asBytes(&device_id));
        const device = client.lookupDevice(device_id) orelse {
            const missing_device: i32 = -1;
            hasher.update(std.mem.asBytes(&missing_device));
            continue;
        };
        hasher.update(std.mem.asBytes(&device.local_hardware_id));
        hasher.update(std.mem.asBytes(&device.registry_id));
        hasher.update(std.mem.asBytes(&device.process_index));
        hasher.update(std.mem.asBytes(&device.addressable));
        hasher.update(std.mem.asBytes(&device.memory_bytes));
        hasher.update(std.mem.asBytes(&device.has_unified_memory));
        hasher.update(std.mem.asBytes(&device.default_memory_id));
        hasher.update(device.name);
        hasher.update(device.debug_string);
        const memory = client.lookupMemory(device.default_memory_id) orelse continue;
        hasher.update(std.mem.asBytes(&memory.id));
        hasher.update(std.mem.asBytes(&memory.kind));
        hasher.update(std.mem.sliceAsBytes(memory.addressable_device_ids));
    }
}

fn allocExecutableFingerprint(
    allocator: std.mem.Allocator,
    client: *const Client,
    optimized_program: []const u8,
    plan: *const ExecutablePlan,
) ![]u8 {
    var hasher = std.hash.Wyhash.init(0);
    hasher.update("pjrtx-executable-cache-v4");
    hasher.update(optimized_program);
    const caps = client.backend.capabilities();
    hasher.update(caps.name);
    hasher.update(std.mem.asBytes(&caps.supports_device_buffers));
    hasher.update(std.mem.asBytes(&caps.supports_unified_memory));
    hasher.update(std.mem.asBytes(&caps.supports_async_execution));
    const custom_call_registry_version = client.backend.customCallRegistryVersion();
    hasher.update(std.mem.asBytes(&custom_call_registry_version));
    updateTargetDeviceFingerprint(&hasher, client, plan);
    hasher.update(std.mem.asBytes(&plan.options.num_replicas));
    hasher.update(std.mem.asBytes(&plan.options.num_partitions));
    hasher.update(std.mem.asBytes(&plan.options.use_shardy_partitioner));
    hasher.update(std.mem.sliceAsBytes(plan.options.device_assignment));
    updateShardingFingerprint(&hasher, plan.parameter_shardings);
    updateShardingFingerprint(&hasher, plan.output_shardings);
    hasher.update(std.mem.sliceAsBytes(plan.output_ids));
    hasher.update(std.mem.sliceAsBytes(plan.donated_parameter_indices));
    for (plan.output_aliases) |alias| {
        hasher.update(std.mem.asBytes(&alias.output_index));
        hasher.update(std.mem.asBytes(&alias.parameter_index));
        hasher.update(std.mem.asBytes(&alias.kind));
    }
    for (plan.values) |value| {
        hasher.update(std.mem.asBytes(&value.role));
        hasher.update(std.mem.asBytes(&value.descriptor.element_type));
        hasher.update(std.mem.sliceAsBytes(value.descriptor.dims));
        hasher.update(std.mem.asBytes(&value.descriptor.layout));
        hasher.update(std.mem.asBytes(&value.descriptor.device_id));
        hasher.update(std.mem.asBytes(&value.descriptor.memory_id));
        hasher.update(std.mem.asBytes(&value.descriptor.shard_index));
        hasher.update(std.mem.asBytes(&value.storage));
        hasher.update(std.mem.sliceAsBytes(value.elements));
    }
    for (plan.regions) |region| {
        hasher.update(std.mem.asBytes(&region.id));
        hasher.update(std.mem.asBytes(&region.parent_instruction_index));
        hasher.update(std.mem.asBytes(&region.kind));
        for (region.argument_descriptors) |descriptor| {
            hasher.update(std.mem.asBytes(&descriptor.element_type));
            hasher.update(std.mem.sliceAsBytes(descriptor.dims));
            hasher.update(std.mem.asBytes(&descriptor.layout));
        }
        for (region.return_descriptors) |descriptor| {
            hasher.update(std.mem.asBytes(&descriptor.element_type));
            hasher.update(std.mem.sliceAsBytes(descriptor.dims));
            hasher.update(std.mem.asBytes(&descriptor.layout));
        }
    }
    for (plan.instructions) |instruction| updateInstructionFingerprint(&hasher, instruction);
    return std.fmt.allocPrint(allocator, "pjrtx-{x}", .{hasher.final()});
}

fn nowTimestamp() std.Io.Timestamp {
    return std.Io.Timestamp.now(io, .awake);
}

fn elapsedMicrosSince(start: std.Io.Timestamp) u64 {
    return @intCast(@max(start.durationTo(nowTimestamp()).toMicroseconds(), 0));
}

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

fn releaseCachedBackendExecutableForContext(user_context: *anyopaque, entry: *ExecutableCacheEntry) void {
    clientFromExecutableContext(user_context).releaseCachedBackendExecutable(entry);
}

fn trimExecutableCacheForContext(user_context: *anyopaque, memory: *Memory, allocation_bytes: usize) ExecutableCacheTrim {
    return clientFromExecutableContext(user_context).trimExecutableCacheForAllocation(memory, allocation_bytes);
}

/// Owns runtime topology, buffer factories, executable cache, and compile scheduling.
pub const Client = struct {
    allocator: std.mem.Allocator,
    backend: backend_api.Backend,
    devices: []Device,
    memories: []Memory,
    device_handles: []*Device,
    memory_handles: []*Memory,
    topology: Topology,
    executable_cache: ExecutableCache,
    executable_cache_mutex: std.Io.Mutex = .init,

    /// Initializes a client from concrete Metal/MLX device descriptors.
    pub fn init(allocator: std.mem.Allocator, backend_impl: backend_api.Backend, device_count: usize) !*Client {
        if (device_count == 0 or device_count > MAX_DEVICES) return error.InvalidDeviceCount;

        const client = try allocator.create(Client);
        errdefer allocator.destroy(client);

        const descriptors = try backend_impl.enumerateDevices(allocator, device_count);
        defer backend_impl.releaseDeviceDescriptors(allocator, descriptors);
        if (descriptors.len == 0 or descriptors.len > MAX_DEVICES) return error.InvalidDeviceCount;

        const devices = try allocator.alloc(Device, descriptors.len);
        errdefer allocator.free(devices);

        const memories = try allocator.alloc(Memory, descriptors.len);
        errdefer allocator.free(memories);

        const device_handles = try allocator.alloc(*Device, descriptors.len);
        errdefer allocator.free(device_handles);

        const memory_handles = try allocator.alloc(*Memory, descriptors.len);
        errdefer allocator.free(memory_handles);

        const assignment = try allocator.alloc(i32, descriptors.len);
        errdefer allocator.free(assignment);

        for (descriptors, 0..) |descriptor, i| {
            const name = try allocator.dupe(u8, descriptor.name);
            errdefer allocator.free(name);

            const debug_string = try allocator.dupe(u8, descriptor.debug_string);
            errdefer allocator.free(debug_string);

            const memory_debug_string = try allocator.dupe(u8, "device");
            errdefer allocator.free(memory_debug_string);

            const device_memories = try allocator.alloc(*Memory, 1);
            errdefer allocator.free(device_memories);

            const memory_devices = try allocator.alloc(*Device, 1);
            errdefer allocator.free(memory_devices);

            const ids = try allocator.alloc(i32, 1);
            errdefer allocator.free(ids);

            devices[i] = .{
                .id = descriptor.id,
                .local_hardware_id = descriptor.local_hardware_id,
                .registry_id = descriptor.registry_id,
                .process_index = descriptor.process_index,
                .addressable = descriptor.addressable,
                .name = name,
                .debug_string = debug_string,
                .memory_bytes = descriptor.memory_bytes,
                .has_unified_memory = descriptor.has_unified_memory,
                .default_memory_id = descriptor.default_memory_id,
                .default_memory = &memories[i],
                .addressable_memories = device_memories,
            };
            ids[0] = descriptor.id;
            memories[i] = .{
                .id = descriptor.default_memory_id,
                .kind = .device,
                .debug_string = memory_debug_string,
                .addressable_device_ids = ids,
                .addressable_devices = memory_devices,
                .stats = .{ .capacity_bytes = descriptor.memory_bytes },
            };
            devices[i].addressable_memories[0] = &memories[i];
            memories[i].addressable_devices[0] = &devices[i];
            device_handles[i] = &devices[i];
            memory_handles[i] = &memories[i];
            assignment[i] = descriptor.id;
        }

        client.* = .{
            .allocator = allocator,
            .backend = backend_impl,
            .devices = devices,
            .memories = memories,
            .device_handles = device_handles,
            .memory_handles = memory_handles,
            .topology = .{
                .device_assignment = assignment,
                .num_replicas = 1,
                .num_partitions = @intCast(descriptors.len),
            },
            .executable_cache = ExecutableCache.init(allocator),
        };
        return client;
    }

    /// Releases all devices, memories, cached executables, and topology storage.
    pub fn deinit(self: *Client) void {
        self.executable_cache.deinit(self.backend);
        for (self.devices) |device| {
            self.allocator.free(device.addressable_memories);
            self.allocator.free(device.name);
            self.allocator.free(device.debug_string);
        }
        for (self.memories) |memory| {
            self.allocator.free(memory.addressable_devices);
            self.allocator.free(memory.addressable_device_ids);
            self.allocator.free(memory.debug_string);
        }
        self.allocator.free(self.topology.device_assignment);
        self.allocator.free(self.memory_handles);
        self.allocator.free(self.device_handles);
        self.allocator.free(self.memories);
        self.allocator.free(self.devices);
        self.allocator.destroy(self);
    }

    /// Finds a device by stable PJRT device id.
    pub fn lookupDevice(self: *const Client, id: i32) ?*const Device {
        for (self.devices) |*device| {
            if (device.id == id) return device;
        }
        return null;
    }

    /// Finds an addressable device by backend-local hardware id.
    pub fn lookupAddressableDeviceByLocalHardwareId(self: *const Client, local_hardware_id: i32) ?*const Device {
        for (self.devices) |*device| {
            if (device.addressable and device.local_hardware_id == local_hardware_id) return device;
        }
        return null;
    }

    /// Finds a runtime memory by stable PJRT memory id.
    pub fn lookupMemory(self: *const Client, id: i32) ?*const Memory {
        for (self.memories) |*memory| {
            if (memory.id == id) return memory;
        }
        return null;
    }

    /// Records a compile request and returns whether the fingerprint was seen before.
    pub fn recordExecutableCompile(self: *Client, fingerprint: []const u8) !bool {
        lockExecutableCache(self);
        defer self.executable_cache_mutex.unlock(io);
        const hit = try self.executable_cache.recordCompile(fingerprint);
        self.syncExecutableCacheMemoryStats();
        return hit;
    }

    /// Sets the resident executable-cache budget and evicts idle entries as needed.
    pub fn setExecutableCacheMaxResidentBytes(self: *Client, max_resident_bytes: u64) void {
        lockExecutableCache(self);
        defer self.executable_cache_mutex.unlock(io);
        self.executable_cache.setMaxResidentBytes(self.backend, max_resident_bytes);
        self.syncExecutableCacheMemoryStats();
    }

    fn lockExecutableCache(self: *Client) void {
        self.executable_cache_mutex.lockUncancelable(io);
    }

    fn syncExecutableCacheMemoryStats(self: *Client) void {
        for (self.memories) |*memory| {
            memory.stats.executable_cache_hits = self.executable_cache.stats.hits;
            memory.stats.executable_cache_misses = self.executable_cache.stats.misses;
            memory.stats.executable_cache_evictions = self.executable_cache.stats.evictions;
            memory.stats.executable_cache_resident_entries = self.executable_cache.stats.resident_entries;
            memory.stats.executable_cache_resident_bytes = self.executable_cache.stats.resident_bytes;
            memory.stats.executable_cache_peak_resident_bytes = self.executable_cache.stats.peak_resident_bytes;
            memory.stats.executable_cache_largest_resident_bytes = self.executable_cache.stats.largest_resident_bytes;
            memory.stats.executable_cache_pressure_trims = self.executable_cache.stats.pressure_trim_requests;
            memory.stats.executable_cache_pressure_trimmed_bytes = self.executable_cache.stats.pressure_trimmed_bytes;
            memory.stats.executable_cache_pressure_trim_failures = self.executable_cache.stats.pressure_trim_failures;
        }
    }

    /// Trims idle resident executables to make room for a device allocation.
    pub fn trimExecutableCacheForAllocation(self: *Client, memory: *Memory, allocation_bytes: usize) ExecutableCacheTrim {
        lockExecutableCache(self);
        defer self.executable_cache_mutex.unlock(io);
        if (memory.stats.capacity_bytes == 0) return .{};

        const allocation: u64 = @intCast(allocation_bytes);
        const bytes_without_cache = memory.stats.bytes_in_use +| allocation;
        const resident_bytes = self.executable_cache.stats.resident_bytes;
        if (bytes_without_cache +| resident_bytes <= memory.stats.capacity_bytes) return .{
            .requested_bytes = allocation,
            .target_resident_bytes = resident_bytes,
            .remaining_resident_bytes = resident_bytes,
        };

        const target_resident_bytes = if (bytes_without_cache >= memory.stats.capacity_bytes)
            0
        else
            memory.stats.capacity_bytes - bytes_without_cache;
        var trim = self.executable_cache.trimIdleToResidentBytes(self.backend, target_resident_bytes);
        trim.requested_bytes = allocation;
        trim.still_over_capacity = bytes_without_cache +| trim.remaining_resident_bytes > memory.stats.capacity_bytes;

        self.executable_cache.stats.pressure_trim_requests += 1;
        self.executable_cache.stats.pressure_trimmed_bytes +|= trim.freed_bytes;
        if (trim.still_over_capacity) self.executable_cache.stats.pressure_trim_failures += 1;
        self.syncExecutableCacheMemoryStats();
        return trim;
    }

    fn executableResidencyMemory(self: *Client, device_local_hardware_ids: []const i32) ?*Memory {
        if (device_local_hardware_ids.len != 0) {
            const first_local_hardware_id = device_local_hardware_ids[0];
            for (self.devices) |*device| {
                if (device.local_hardware_id == first_local_hardware_id) return device.default_memory;
            }
        }
        if (self.memories.len == 0) return null;
        return &self.memories[0];
    }

    /// Acquires a retained resident backend executable for an executable graph.
    pub fn acquireCachedBackendExecutable(
        self: *Client,
        allocator: std.mem.Allocator,
        fingerprint: []const u8,
        plan: *const ir.ExecutablePlan,
        device_local_hardware_ids: []const i32,
    ) !?CachedBackendExecutable {
        const entry = blk: {
            lockExecutableCache(self);
            defer self.executable_cache_mutex.unlock(io);
            const entry = self.executable_cache.get(fingerprint) orelse return null;
            if (entry.backend_executable) |handle| {
                self.executable_cache.retain(entry);
                self.syncExecutableCacheMemoryStats();
                return .{
                    .entry = entry,
                    .handle = handle,
                    .reused = true,
                };
            }
            break :blk entry;
        };

        // Backend compilation can be expensive and may perform allocations that
        // trim the executable cache, so it intentionally happens outside the
        // cache mutex. The entry pointer remains stable until client teardown.
        {
            lockExecutableCache(self);
            defer self.executable_cache_mutex.unlock(io);
            if (entry.backend_executable) |handle| {
                self.executable_cache.retain(entry);
                self.syncExecutableCacheMemoryStats();
                return .{
                    .entry = entry,
                    .handle = handle,
                    .reused = true,
                };
            }
        }

        const compile_start = nowTimestamp();
        const handle = self.backend.compileExecutable(allocator, plan, device_local_hardware_ids) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => null,
        };
        const compile_latency_us = elapsedMicrosSince(compile_start);
        const owned_handle = handle orelse return null;
        const stats = self.backend.executableStats(owned_handle);
        const resident_constant_bytes: usize = @intCast(stats.resident_constant_bytes);
        var compile_trim = ExecutableCacheTrim{};
        if (resident_constant_bytes != 0) {
            if (self.executableResidencyMemory(device_local_hardware_ids)) |memory| {
                compile_trim = self.trimExecutableCacheForAllocation(memory, resident_constant_bytes);
            }
        }
        lockExecutableCache(self);
        defer self.executable_cache_mutex.unlock(io);
        if (entry.backend_executable) |existing| {
            self.backend.destroyExecutable(owned_handle);
            self.executable_cache.retain(entry);
            self.syncExecutableCacheMemoryStats();
            return .{
                .entry = entry,
                .handle = existing,
                .reused = true,
                .compile_trim = compile_trim,
            };
        }
        self.executable_cache.acquireResident(self.backend, entry, owned_handle, resident_constant_bytes);
        self.executable_cache.recordCompileLatency(fingerprint, compile_latency_us);
        self.syncExecutableCacheMemoryStats();
        return .{
            .entry = entry,
            .handle = owned_handle,
            .reused = false,
            .compile_trim = compile_trim,
        };
    }

    /// Releases a retained executable-cache entry after graph teardown.
    pub fn releaseCachedBackendExecutable(self: *Client, entry: *ExecutableCacheEntry) void {
        lockExecutableCache(self);
        defer self.executable_cache_mutex.unlock(io);
        self.executable_cache.release(self.backend, entry);
        self.syncExecutableCacheMemoryStats();
    }

    /// Returns the narrow execution context consumed by compiled executable graphs.
    pub fn executableContext(self: *Client) ExecutableContext {
        return .{
            .backend = self.backend,
            .devices = self.devices,
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
        return try self.backend.beginAsyncHostToDeviceTransfer(device.local_hardware_id, element_type, dims, byte_size) orelse error.UnsupportedRuntimeFeature;
    }

    /// Destroys an async transfer handle that did not become buffer storage.
    pub fn destroyAsyncHostToDeviceTransfer(self: *Client, transfer: AsyncHostToDeviceTransferHandle) void {
        self.backend.destroyAsyncHostToDeviceTransfer(transfer);
    }

    /// Writes one byte segment into an in-flight async transfer.
    pub fn writeAsyncHostToDeviceTransfer(self: *Client, transfer: AsyncHostToDeviceTransferHandle, offset: usize, bytes: []const u8) AsyncTransferWriteError!void {
        try self.backend.writeAsyncHostToDeviceTransfer(transfer, offset, bytes);
    }

    /// Installs completed async transfer storage into a pending runtime buffer.
    pub fn finishAsyncHostToDeviceTransfer(self: *Client, buffer: *Buffer, transfer: AsyncHostToDeviceTransferHandle) AsyncTransferFinishError!void {
        const backend_buffer = try self.backend.finishAsyncHostToDeviceTransfer(transfer) orelse return error.UnsupportedRuntimeFeature;
        errdefer self.backend.destroyBuffer(backend_buffer);
        try buffer.replaceBackendStorage(backend_buffer);
    }

    /// Compiles program bytes into a resident executable plan and backend graph.
    pub fn compileProgram(
        self: *Client,
        allocator: std.mem.Allocator,
        program: CompileProgram,
        diagnostics: *std.Io.Writer,
    ) CompileProgramError!CompiledExecutable {
        var parsed_options = false;
        const options = try parseCompileOptions(allocator, self, program.compile_options, &parsed_options, diagnostics);
        defer if (parsed_options) allocator.free(options.device_assignment);

        var analysis = try analyzeProgram(allocator, program, diagnostics);
        defer if (analysis) |*owned_analysis| owned_analysis.deinit();

        var plan = try makeExecutablePlan(allocator, options, analysis);
        var plan_moved = false;
        errdefer if (!plan_moved) plan.deinit();

        try verifyPlan(allocator, plan, diagnostics);

        const optimized_program = if (analysis) |owned_analysis|
            allocator.dupe(u8, owned_analysis.source) catch return error.OutOfMemory
        else
            allocator.dupe(u8, "module {}\n") catch return error.OutOfMemory;
        errdefer allocator.free(optimized_program);

        const fingerprint = allocExecutableFingerprint(allocator, self, optimized_program, &plan) catch return error.OutOfMemory;
        errdefer allocator.free(fingerprint);

        const cache_hit = self.recordExecutableCompile(fingerprint) catch return error.Internal;

        const plan_ptr = allocator.create(ExecutablePlan) catch return error.OutOfMemory;
        plan_ptr.* = plan;
        plan_moved = true;
        errdefer {
            plan_ptr.deinit();
            allocator.destroy(plan_ptr);
        }

        var graph = ExecutableGraph.initWithOptions(allocator, self.executableContext(), plan_ptr, .{
            .diagnostic_writer = diagnostics,
            .cache_fingerprint = fingerprint,
        }) catch |err| switch (err) {
            error.UnsupportedRuntimeFeature => return error.UnsupportedRuntimeFeature,
            error.OutOfMemory => return error.OutOfMemory,
            else => return error.Internal,
        };
        errdefer graph.deinit();

        return .{
            .plan = plan_ptr,
            .graph = graph,
            .optimized_program = optimized_program,
            .fingerprint = fingerprint,
            .cache_hit = cache_hit,
            .backend_stats = graph.backendExecutableStats() orelse .{},
        };
    }
};

test "client records executable cache hits and buffer memory accounting" {
    const allocator = std.testing.allocator;
    const client = try Client.init(allocator, backend_api.create(), 1);
    defer client.deinit();

    try std.testing.expect(!try client.recordExecutableCompile("same-program"));
    try std.testing.expect(try client.recordExecutableCompile("same-program"));
    try std.testing.expectEqual(@as(u64, 1), client.executable_cache.stats.misses);
    try std.testing.expectEqual(@as(u64, 1), client.executable_cache.stats.hits);

    const dims = [_]i64{4};
    const data = [_]u8{ 1, 2, 3, 4 };
    const before = client.memories[0].stats.bytes_in_use;
    {
        const buffer = try client.createHostBufferFromBytes(allocator, .u8, &dims, &client.devices[0], &client.memories[0], 0, &data);
        defer buffer.deinit();
        try std.testing.expectEqual(before + data.len, client.memories[0].stats.bytes_in_use);
        try std.testing.expect(client.memories[0].stats.peak_bytes_in_use >= client.memories[0].stats.bytes_in_use);
        try std.testing.expect(client.memories[0].stats.host_to_device_bytes >= data.len);
    }
    try std.testing.expectEqual(before, client.memories[0].stats.bytes_in_use);
}
fn testShardingPlan(allocator: std.mem.Allocator, assignment: []const i32) !ir.ShardingPlan {
    return .{
        .kind = .replicated,
        .mesh_name = try allocator.dupe(u8, ""),
        .device_assignment = try allocator.dupe(i32, assignment),
    };
}

fn constantU8ExecutablePlanForTest(allocator: std.mem.Allocator, assignment: []const i32, literal: []const u8) !ir.ExecutablePlan {
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
    output_shardings[0] = try testShardingPlan(allocator, assignment);
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

fn addU8ExecutablePlanForTest(allocator: std.mem.Allocator, assignment: []const i32, dims: []const i64) !ir.ExecutablePlan {
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
    parameter_shardings[0] = try testShardingPlan(allocator, assignment);
    errdefer {
        allocator.free(parameter_shardings[0].mesh_name);
        allocator.free(parameter_shardings[0].device_assignment);
    }
    parameter_shardings[1] = try testShardingPlan(allocator, assignment);
    errdefer {
        allocator.free(parameter_shardings[1].mesh_name);
        allocator.free(parameter_shardings[1].device_assignment);
    }

    var output_shardings = try allocator.alloc(ir.ShardingPlan, 1);
    errdefer allocator.free(output_shardings);
    output_shardings[0] = try testShardingPlan(allocator, assignment);
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

test "client executable cache reuses backend executable handles" {
    const client = try Client.init(std.testing.allocator, backend_api.create(), 1);
    defer client.deinit();
    const allocator = std.testing.allocator;
    const assignment = [_]i32{0};
    const dims = [_]i64{4};

    var plan = try addU8ExecutablePlanForTest(allocator, &assignment, &dims);
    defer plan.deinit();

    const fingerprint = "cached-add";
    try std.testing.expect(!try client.recordExecutableCompile(fingerprint));
    var first = try ExecutableGraph.initWithOptions(allocator, client.executableContext(), &plan, .{
        .cache_fingerprint = fingerprint,
    });
    try std.testing.expect(first.backend_executable != null);
    try std.testing.expect(!first.lowering.backend_executable_cache_reused);

    const entry = client.executable_cache.get(fingerprint) orelse return error.TestUnexpectedResult;
    try std.testing.expect(entry.backend_executable != null);
    try std.testing.expectEqual(@as(usize, 1), entry.ref_count);

    try std.testing.expect(try client.recordExecutableCompile(fingerprint));
    var second = try ExecutableGraph.initWithOptions(allocator, client.executableContext(), &plan, .{
        .cache_fingerprint = fingerprint,
    });
    try std.testing.expect(second.backend_executable != null);
    try std.testing.expect(second.lowering.backend_executable_cache_reused);
    try std.testing.expectEqual(first.backend_executable.?, second.backend_executable.?);
    try std.testing.expectEqual(@as(u64, 1), client.executable_cache.stats.misses);
    try std.testing.expectEqual(@as(u64, 1), client.executable_cache.stats.hits);
    try std.testing.expectEqual(@as(usize, 2), entry.ref_count);

    second.deinit();
    try std.testing.expectEqual(@as(usize, 1), entry.ref_count);
    first.deinit();
    try std.testing.expectEqual(@as(usize, 0), entry.ref_count);
    try std.testing.expect(entry.backend_executable != null);
}

test "executable cache evicts idle resident backend executables under byte limit" {
    const client = try Client.init(std.testing.allocator, backend_api.create(), 1);
    defer client.deinit();
    const allocator = std.testing.allocator;
    const assignment = [_]i32{0};
    const literal = [_]u8{ 1, 2, 3, 4 };

    var plan = try constantU8ExecutablePlanForTest(allocator, &assignment, &literal);
    defer plan.deinit();

    client.setExecutableCacheMaxResidentBytes(0);

    const fingerprint = "evict-idle-constant";
    try std.testing.expect(!try client.recordExecutableCompile(fingerprint));
    var first = try ExecutableGraph.initWithOptions(allocator, client.executableContext(), &plan, .{
        .cache_fingerprint = fingerprint,
    });
    const entry = client.executable_cache.get(fingerprint) orelse return error.TestUnexpectedResult;
    try std.testing.expect(first.backend_executable != null);
    try std.testing.expect(entry.backend_executable != null);
    try std.testing.expect(entry.resident_bytes >= literal.len);
    try std.testing.expectEqual(@as(u64, 1), client.executable_cache.stats.resident_entries);
    try std.testing.expect(client.executable_cache.stats.resident_bytes >= literal.len);
    try std.testing.expect(client.memories[0].stats.totalBytesInUse() >= literal.len);
    try std.testing.expect(client.memories[0].stats.peakTotalBytesInUse() >= client.memories[0].stats.totalBytesInUse());
    try std.testing.expectEqual(@as(usize, 1), entry.ref_count);

    first.deinit();
    try std.testing.expectEqual(@as(usize, 0), entry.ref_count);
    try std.testing.expect(entry.backend_executable == null);
    try std.testing.expectEqual(@as(u64, 0), client.executable_cache.stats.resident_entries);
    try std.testing.expectEqual(@as(u64, 0), client.executable_cache.stats.resident_bytes);
    try std.testing.expectEqual(@as(u64, 0), client.memories[0].stats.totalBytesInUse());
    try std.testing.expectEqual(@as(u64, 1), client.executable_cache.stats.evictions);
    try std.testing.expectEqual(@as(u64, 1), client.memories[0].stats.executable_cache_evictions);

    try std.testing.expect(try client.recordExecutableCompile(fingerprint));
    var second = try ExecutableGraph.initWithOptions(allocator, client.executableContext(), &plan, .{
        .cache_fingerprint = fingerprint,
    });
    defer second.deinit();
    try std.testing.expect(second.backend_executable != null);
    try std.testing.expect(!second.lowering.backend_executable_cache_reused);
    try std.testing.expectEqual(@as(usize, 1), entry.ref_count);
    try std.testing.expect(entry.backend_executable != null);
    try std.testing.expectEqual(@as(u64, 1), client.executable_cache.stats.hits);
}

test "executable cache evicts largest idle resident executable before older smaller entries" {
    const client = try Client.init(std.testing.allocator, backend_api.create(), 1);
    defer client.deinit();
    const allocator = std.testing.allocator;
    const assignment = [_]i32{0};

    var small_plan = try constantU8ExecutablePlanForTest(allocator, &assignment, &.{ 1, 2, 3, 4 });
    defer small_plan.deinit();
    var large_plan = try constantU8ExecutablePlanForTest(allocator, &assignment, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer large_plan.deinit();

    try std.testing.expect(!try client.recordExecutableCompile("small-constant"));
    var small_graph = try ExecutableGraph.initWithOptions(allocator, client.executableContext(), &small_plan, .{
        .cache_fingerprint = "small-constant",
    });
    const small_entry = client.executable_cache.get("small-constant") orelse return error.TestUnexpectedResult;
    const small_resident_bytes = small_entry.resident_bytes;
    try std.testing.expect(small_resident_bytes >= 4);
    small_graph.deinit();
    try std.testing.expectEqual(@as(usize, 0), small_entry.ref_count);
    try std.testing.expect(small_entry.backend_executable != null);

    try std.testing.expect(!try client.recordExecutableCompile("large-constant"));
    var large_graph = try ExecutableGraph.initWithOptions(allocator, client.executableContext(), &large_plan, .{
        .cache_fingerprint = "large-constant",
    });
    const large_entry = client.executable_cache.get("large-constant") orelse return error.TestUnexpectedResult;
    const large_resident_bytes = large_entry.resident_bytes;
    try std.testing.expect(large_resident_bytes > small_resident_bytes);
    large_graph.deinit();
    try std.testing.expectEqual(@as(usize, 0), large_entry.ref_count);
    try std.testing.expect(large_entry.backend_executable != null);

    client.setExecutableCacheMaxResidentBytes(small_resident_bytes);
    try std.testing.expect(small_entry.backend_executable != null);
    try std.testing.expect(large_entry.backend_executable == null);
    try std.testing.expectEqual(@as(u64, 1), client.executable_cache.stats.resident_entries);
    try std.testing.expectEqual(small_resident_bytes, client.executable_cache.stats.resident_bytes);
    try std.testing.expectEqual(@as(u64, 1), client.executable_cache.stats.evictions);
    try std.testing.expectEqual(large_resident_bytes, client.executable_cache.stats.evicted_resident_bytes);
}

test "client trims idle executable cache for tracked memory pressure" {
    const client = try Client.init(std.testing.allocator, backend_api.create(), 1);
    defer client.deinit();
    const allocator = std.testing.allocator;
    const assignment = [_]i32{0};

    var small_plan = try constantU8ExecutablePlanForTest(allocator, &assignment, &.{ 1, 2, 3, 4 });
    defer small_plan.deinit();
    var large_plan = try constantU8ExecutablePlanForTest(allocator, &assignment, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer large_plan.deinit();

    try std.testing.expect(!try client.recordExecutableCompile("pressure-small-constant"));
    var small_graph = try ExecutableGraph.initWithOptions(allocator, client.executableContext(), &small_plan, .{
        .cache_fingerprint = "pressure-small-constant",
    });
    const small_entry = client.executable_cache.get("pressure-small-constant") orelse return error.TestUnexpectedResult;
    const small_resident_bytes = small_entry.resident_bytes;
    small_graph.deinit();

    try std.testing.expect(!try client.recordExecutableCompile("pressure-large-constant"));
    var large_graph = try ExecutableGraph.initWithOptions(allocator, client.executableContext(), &large_plan, .{
        .cache_fingerprint = "pressure-large-constant",
    });
    const large_entry = client.executable_cache.get("pressure-large-constant") orelse return error.TestUnexpectedResult;
    const large_resident_bytes = large_entry.resident_bytes;
    try std.testing.expect(large_resident_bytes > small_resident_bytes);
    large_graph.deinit();

    client.memories[0].stats.capacity_bytes = small_resident_bytes;
    const trim = client.trimExecutableCacheForAllocation(&client.memories[0], 0);
    try std.testing.expectEqual(@as(u64, 0), trim.requested_bytes);
    try std.testing.expectEqual(small_resident_bytes, trim.target_resident_bytes);
    try std.testing.expectEqual(large_resident_bytes, trim.freed_bytes);
    try std.testing.expectEqual(@as(u64, 1), trim.evicted_entries);
    try std.testing.expect(!trim.still_over_capacity);
    try std.testing.expect(small_entry.backend_executable != null);
    try std.testing.expect(large_entry.backend_executable == null);
    try std.testing.expectEqual(small_resident_bytes, client.executable_cache.stats.resident_bytes);
    try std.testing.expectEqual(@as(u64, 1), client.executable_cache.stats.pressure_trim_requests);
    try std.testing.expectEqual(large_resident_bytes, client.executable_cache.stats.pressure_trimmed_bytes);
    try std.testing.expectEqual(@as(u64, 0), client.executable_cache.stats.pressure_trim_failures);
    try std.testing.expectEqual(@as(u64, 1), client.memories[0].stats.executable_cache_pressure_trims);
    try std.testing.expectEqual(large_resident_bytes, client.memories[0].stats.executable_cache_pressure_trimmed_bytes);
    try std.testing.expectEqual(@as(u64, 0), client.memories[0].stats.executable_cache_pressure_trim_failures);
}

test "compiling resident executable trims idle cache under memory pressure" {
    const client = try Client.init(std.testing.allocator, backend_api.create(), 1);
    defer client.deinit();
    const allocator = std.testing.allocator;
    const assignment = [_]i32{0};

    var idle_plan = try constantU8ExecutablePlanForTest(allocator, &assignment, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer idle_plan.deinit();
    var new_plan = try constantU8ExecutablePlanForTest(allocator, &assignment, &.{ 9, 10, 11, 12 });
    defer new_plan.deinit();

    try std.testing.expect(!try client.recordExecutableCompile("compile-pressure-idle-constant"));
    var idle_graph = try ExecutableGraph.initWithOptions(allocator, client.executableContext(), &idle_plan, .{
        .cache_fingerprint = "compile-pressure-idle-constant",
    });
    const idle_entry = client.executable_cache.get("compile-pressure-idle-constant") orelse return error.TestUnexpectedResult;
    const idle_resident_bytes = idle_entry.resident_bytes;
    try std.testing.expect(idle_resident_bytes >= 8);
    idle_graph.deinit();
    try std.testing.expectEqual(@as(usize, 0), idle_entry.ref_count);
    try std.testing.expect(idle_entry.backend_executable != null);

    client.memories[0].stats.capacity_bytes = idle_resident_bytes;

    try std.testing.expect(!try client.recordExecutableCompile("compile-pressure-new-constant"));
    var new_graph = try ExecutableGraph.initWithOptions(allocator, client.executableContext(), &new_plan, .{
        .cache_fingerprint = "compile-pressure-new-constant",
    });
    defer new_graph.deinit();
    const new_entry = client.executable_cache.get("compile-pressure-new-constant") orelse return error.TestUnexpectedResult;
    try std.testing.expect(new_graph.backend_executable != null);
    try std.testing.expect(new_entry.backend_executable != null);
    try std.testing.expect(new_entry.resident_bytes >= 4);
    try std.testing.expect(new_entry.resident_bytes <= idle_resident_bytes);
    try std.testing.expect(idle_entry.backend_executable == null);
    try std.testing.expectEqual(new_entry.resident_bytes, new_graph.last_compile_cache_trim.requested_bytes);
    try std.testing.expectEqual(idle_resident_bytes - new_entry.resident_bytes, new_graph.last_compile_cache_trim.target_resident_bytes);
    try std.testing.expectEqual(idle_resident_bytes, new_graph.last_compile_cache_trim.freed_bytes);
    try std.testing.expectEqual(@as(u64, 1), new_graph.last_compile_cache_trim.evicted_entries);
    try std.testing.expectEqual(@as(u64, 0), new_graph.last_compile_cache_trim.remaining_resident_bytes);
    try std.testing.expect(!new_graph.last_compile_cache_trim.still_over_capacity);
    try std.testing.expectEqual(@as(u64, 1), client.executable_cache.stats.pressure_trim_requests);
    try std.testing.expectEqual(idle_resident_bytes, client.executable_cache.stats.pressure_trimmed_bytes);
    try std.testing.expectEqual(@as(u64, 0), client.executable_cache.stats.pressure_trim_failures);
    try std.testing.expectEqual(new_entry.resident_bytes, client.executable_cache.stats.resident_bytes);
    try std.testing.expectEqual(@as(u64, 1), client.memories[0].stats.executable_cache_pressure_trims);
    try std.testing.expectEqual(idle_resident_bytes, client.memories[0].stats.executable_cache_pressure_trimmed_bytes);
}

test "executable cache preserves more expensive equal-size resident executable" {
    const client = try Client.init(std.testing.allocator, backend_api.create(), 1);
    defer client.deinit();
    const allocator = std.testing.allocator;
    const assignment = [_]i32{0};

    var cheap_plan = try constantU8ExecutablePlanForTest(allocator, &assignment, &.{ 1, 2, 3, 4 });
    defer cheap_plan.deinit();
    var expensive_plan = try constantU8ExecutablePlanForTest(allocator, &assignment, &.{ 5, 6, 7, 8 });
    defer expensive_plan.deinit();

    try std.testing.expect(!try client.recordExecutableCompile("cheap-constant"));
    var cheap_graph = try ExecutableGraph.initWithOptions(allocator, client.executableContext(), &cheap_plan, .{
        .cache_fingerprint = "cheap-constant",
    });
    const cheap_entry = client.executable_cache.get("cheap-constant") orelse return error.TestUnexpectedResult;
    const resident_bytes = cheap_entry.resident_bytes;
    cheap_graph.deinit();
    cheap_entry.compile_latency_us = 10;

    try std.testing.expect(!try client.recordExecutableCompile("expensive-constant"));
    var expensive_graph = try ExecutableGraph.initWithOptions(allocator, client.executableContext(), &expensive_plan, .{
        .cache_fingerprint = "expensive-constant",
    });
    const expensive_entry = client.executable_cache.get("expensive-constant") orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(resident_bytes, expensive_entry.resident_bytes);
    expensive_graph.deinit();
    expensive_entry.compile_latency_us = 1000;

    try std.testing.expectEqual(@as(u64, 2), client.executable_cache.stats.compile_latency_samples);
    try std.testing.expect(client.executable_cache.stats.compile_latency_us_total > 0);
    try std.testing.expect(client.executable_cache.stats.compile_latency_us_peak > 0);

    client.setExecutableCacheMaxResidentBytes(resident_bytes);
    try std.testing.expect(cheap_entry.backend_executable == null);
    try std.testing.expect(expensive_entry.backend_executable != null);
    try std.testing.expectEqual(@as(u64, 1), client.executable_cache.stats.resident_entries);
    try std.testing.expectEqual(resident_bytes, client.executable_cache.stats.resident_bytes);
    try std.testing.expectEqual(@as(u64, 1), client.executable_cache.stats.evictions);
    try std.testing.expectEqual(resident_bytes, client.executable_cache.stats.evicted_resident_bytes);
}
