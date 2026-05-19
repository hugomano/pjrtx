const std = @import("std");
const backend_api = @import("src/backend");
const core = @import("src/core");

pub const MAX_DEVICES = core.MAX_DEVICES;
pub const MemoryKind = core.MemoryKind;
pub const BackendKind = core.BackendKind;
pub const BufferType = core.BufferType;
pub const ElementwiseBinaryOp = core.ElementwiseBinaryOp;
pub const ElementwiseUnaryOp = core.ElementwiseUnaryOp;
pub const CompareOp = core.CompareOp;

const MachTimebaseInfo = extern struct {
    numer: u32,
    denom: u32,
};

extern "c" fn mach_absolute_time() u64;
extern "c" fn mach_timebase_info(info: *MachTimebaseInfo) c_int;

pub const Device = struct {
    id: i32,
    local_hardware_id: i32,
    registry_id: u64 = 0,
    process_index: i32 = 0,
    addressable: bool = true,
    name: []const u8,
    debug_string: []const u8,
    memory_bytes: u64 = 0,
    has_unified_memory: bool = false,
    default_memory_id: i32,
    default_memory: *Memory,
    addressable_memories: []*Memory,
};

pub const Memory = struct {
    id: i32,
    kind: MemoryKind,
    debug_string: []const u8,
    addressable_device_ids: []const i32,
    addressable_devices: []*Device,
    stats: MemoryStats = .{},

    pub fn isAddressableBy(self: Memory, device: *const Device) bool {
        for (self.addressable_device_ids) |device_id| {
            if (device_id == device.id) return true;
        }
        return false;
    }
};

pub const Topology = struct {
    device_assignment: []const i32,
    num_replicas: i32,
    num_partitions: i32,

    pub fn numDevices(self: Topology) usize {
        return @intCast(self.num_replicas * self.num_partitions);
    }
};

pub const MemoryStats = struct {
    capacity_bytes: u64 = 0,
    bytes_in_use: u64 = 0,
    peak_bytes_in_use: u64 = 0,
    live_allocs: u64 = 0,
    total_allocs: u64 = 0,
    largest_alloc_size: u64 = 0,
    host_to_device_bytes: u64 = 0,
    device_to_host_bytes: u64 = 0,
    executable_cache_hits: u64 = 0,
    executable_cache_misses: u64 = 0,
    executable_cache_evictions: u64 = 0,
    executable_cache_resident_entries: u64 = 0,
    executable_cache_resident_bytes: u64 = 0,
    executable_cache_peak_resident_bytes: u64 = 0,
    executable_cache_largest_resident_bytes: u64 = 0,
    executable_cache_pressure_trims: u64 = 0,
    executable_cache_pressure_trimmed_bytes: u64 = 0,
    executable_cache_pressure_trim_failures: u64 = 0,

    pub fn retain(self: *MemoryStats, bytes: usize) void {
        const amount: u64 = @intCast(bytes);
        self.bytes_in_use +|= amount;
        self.peak_bytes_in_use = @max(self.peak_bytes_in_use, self.bytes_in_use);
        self.live_allocs += 1;
        self.total_allocs += 1;
        self.largest_alloc_size = @max(self.largest_alloc_size, amount);
    }

    pub fn release(self: *MemoryStats, bytes: usize) void {
        const amount: u64 = @intCast(bytes);
        self.bytes_in_use = if (amount > self.bytes_in_use) 0 else self.bytes_in_use - amount;
        if (self.live_allocs != 0) self.live_allocs -= 1;
    }

    pub fn totalBytesInUse(self: MemoryStats) u64 {
        return self.bytes_in_use +| self.executable_cache_resident_bytes;
    }

    pub fn peakTotalBytesInUse(self: MemoryStats) u64 {
        return @max(self.totalBytesInUse(), self.peak_bytes_in_use +| self.executable_cache_peak_resident_bytes);
    }
};

pub const EventState = enum {
    pending,
    ready,
    failed,
};

pub const MAX_EVENT_CALLBACKS = 256;
pub const EventCallback = *const fn (message: ?[]const u8, user_arg: ?*anyopaque) void;

pub const EventCallbackRegistration = struct {
    callback: EventCallback,
    user_arg: ?*anyopaque = null,
};

pub const Event = struct {
    state: EventState = .ready,
    message: []const u8 = "",
    callbacks: [MAX_EVENT_CALLBACKS]EventCallbackRegistration = undefined,
    callback_count: usize = 0,

    pub fn pending() Event {
        return .{ .state = .pending };
    }

    pub fn ready() Event {
        return .{ .state = .ready };
    }

    pub fn failed(message: []const u8) Event {
        return .{ .state = .failed, .message = message };
    }

    pub fn isReady(self: Event) bool {
        return self.state != .pending;
    }

    pub fn onReady(self: *Event, callback: EventCallback, user_arg: ?*anyopaque) !void {
        if (self.state != .pending) {
            callback(self.errorMessage(), user_arg);
            return;
        }
        if (self.callback_count >= MAX_EVENT_CALLBACKS) return error.TooManyEventCallbacks;
        self.callbacks[self.callback_count] = .{
            .callback = callback,
            .user_arg = user_arg,
        };
        self.callback_count += 1;
    }

    pub fn chainTo(self: *Event, dependent: *Event) !void {
        try self.onReady(resolveChainedEvent, dependent);
    }

    pub fn setReady(self: *Event) void {
        if (self.state != .pending) return;
        self.state = .ready;
        self.message = "";
        self.invokeCallbacks();
    }

    pub fn setFailed(self: *Event, message: []const u8) void {
        if (self.state != .pending) {
            self.state = .failed;
            self.message = message;
            return;
        }
        self.state = .failed;
        self.message = message;
        self.invokeCallbacks();
    }

    pub fn awaitReady(self: Event) !void {
        return switch (self.state) {
            .ready => {},
            .failed => error.EventFailed,
            .pending => error.EventPending,
        };
    }

    pub fn deinit(self: *Event) void {
        if (self.state == .pending) {
            self.state = .failed;
            self.message = "event destroyed before completion";
            self.invokeCallbacks();
        }
        self.* = undefined;
    }

    fn errorMessage(self: Event) ?[]const u8 {
        return switch (self.state) {
            .failed => self.message,
            .pending, .ready => null,
        };
    }

    fn invokeCallbacks(self: *Event) void {
        const callback_count = self.callback_count;
        self.callback_count = 0;
        const message = self.errorMessage();
        for (self.callbacks[0..callback_count]) |registration| {
            registration.callback(message, registration.user_arg);
        }
    }

    fn resolveChainedEvent(message: ?[]const u8, user_arg: ?*anyopaque) void {
        const dependent: *Event = @ptrCast(@alignCast(user_arg.?));
        if (message) |msg| {
            dependent.setFailed(msg);
        } else {
            dependent.setReady();
        }
    }
};

pub const BufferState = enum {
    live,
    deleted,
    donated,
};

pub const DonationAliasStats = struct {
    output_count: usize = 0,
    output_bytes: usize = 0,
};

fn nowNs() u64 {
    var info: MachTimebaseInfo = undefined;
    if (mach_timebase_info(&info) != 0 or info.denom == 0) return mach_absolute_time();
    const ticks: u128 = mach_absolute_time();
    return @intCast((ticks * info.numer) / info.denom);
}

fn elapsedMicrosSince(start_ns: u64) u64 {
    return (nowNs() -| start_ns) / std.time.ns_per_us;
}

pub const ExecutableCacheStats = struct {
    hits: u64 = 0,
    misses: u64 = 0,
    evictions: u64 = 0,
    evicted_resident_bytes: u64 = 0,
    resident_entries: u64 = 0,
    resident_bytes: u64 = 0,
    peak_resident_bytes: u64 = 0,
    largest_resident_bytes: u64 = 0,
    compile_latency_samples: u64 = 0,
    compile_latency_us_total: u64 = 0,
    compile_latency_us_peak: u64 = 0,
    pressure_trim_requests: u64 = 0,
    pressure_trimmed_bytes: u64 = 0,
    pressure_trim_failures: u64 = 0,
};

pub const ExecutableCacheEntry = struct {
    fingerprint: []u8,
    backend_executable: ?backend_api.ExecutableHandle = null,
    resident_bytes: u64 = 0,
    compile_latency_us: u64 = 0,
    compile_latency_samples: u64 = 0,
    ref_count: usize = 0,
    last_use: u64 = 0,
};

pub const CachedBackendExecutable = struct {
    entry: *ExecutableCacheEntry,
    handle: backend_api.ExecutableHandle,
    reused: bool,
    compile_trim: ExecutableCacheTrim = .{},
};

pub const ExecutableCacheTrim = struct {
    requested_bytes: u64 = 0,
    target_resident_bytes: u64 = 0,
    freed_bytes: u64 = 0,
    evicted_entries: u64 = 0,
    remaining_resident_bytes: u64 = 0,
    still_over_capacity: bool = false,
};

pub const ExecutableCache = struct {
    allocator: std.mem.Allocator,
    entries: std.StringHashMapUnmanaged(*ExecutableCacheEntry) = .empty,
    stats: ExecutableCacheStats = .{},
    max_resident_bytes: u64 = std.math.maxInt(u64),
    next_use: u64 = 0,

    pub fn init(allocator: std.mem.Allocator) ExecutableCache {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *ExecutableCache, backend: backend_api.Backend) void {
        var it = self.entries.valueIterator();
        while (it.next()) |entry_ptr| {
            const entry = entry_ptr.*;
            if (entry.backend_executable) |handle| backend.destroyExecutable(handle);
            self.allocator.free(entry.fingerprint);
            self.allocator.destroy(entry);
        }
        self.entries.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn recordCompile(self: *ExecutableCache, fingerprint: []const u8) !bool {
        if (self.entries.contains(fingerprint)) {
            self.stats.hits += 1;
            return true;
        }
        const entry = try self.allocator.create(ExecutableCacheEntry);
        errdefer self.allocator.destroy(entry);
        entry.* = .{
            .fingerprint = try self.allocator.dupe(u8, fingerprint),
        };
        errdefer self.allocator.free(entry.fingerprint);
        try self.entries.put(self.allocator, entry.fingerprint, entry);
        self.stats.misses += 1;
        return false;
    }

    pub fn recordCompileLatency(self: *ExecutableCache, fingerprint: []const u8, latency_us: u64) void {
        const entry = self.get(fingerprint) orelse return;
        entry.compile_latency_us = latency_us;
        entry.compile_latency_samples += 1;
        self.stats.compile_latency_samples += 1;
        self.stats.compile_latency_us_total +|= latency_us;
        self.stats.compile_latency_us_peak = @max(self.stats.compile_latency_us_peak, latency_us);
    }

    pub fn setMaxResidentBytes(self: *ExecutableCache, backend: backend_api.Backend, max_resident_bytes: u64) void {
        self.max_resident_bytes = max_resident_bytes;
        self.evictIdleUntilUnderLimit(backend);
    }

    pub fn trimIdleToResidentBytes(self: *ExecutableCache, backend: backend_api.Backend, target_resident_bytes: u64) ExecutableCacheTrim {
        const before_bytes = self.stats.resident_bytes;
        const before_evictions = self.stats.evictions;
        while (self.stats.resident_bytes > target_resident_bytes) {
            const victim = self.pressureEvictionVictim() orelse break;
            self.evictIdleEntry(backend, victim);
        }
        return .{
            .target_resident_bytes = target_resident_bytes,
            .freed_bytes = before_bytes - self.stats.resident_bytes,
            .evicted_entries = self.stats.evictions - before_evictions,
            .remaining_resident_bytes = self.stats.resident_bytes,
            .still_over_capacity = self.stats.resident_bytes > target_resident_bytes,
        };
    }

    pub fn acquireResident(
        self: *ExecutableCache,
        backend: backend_api.Backend,
        entry: *ExecutableCacheEntry,
        handle: backend_api.ExecutableHandle,
        resident_bytes: u64,
    ) void {
        entry.backend_executable = handle;
        entry.resident_bytes = resident_bytes;
        entry.ref_count = 1;
        self.touch(entry);
        self.stats.resident_entries += 1;
        self.stats.resident_bytes +|= resident_bytes;
        self.stats.peak_resident_bytes = @max(self.stats.peak_resident_bytes, self.stats.resident_bytes);
        self.stats.largest_resident_bytes = @max(self.stats.largest_resident_bytes, resident_bytes);
        self.evictIdleUntilUnderLimit(backend);
    }

    pub fn retain(self: *ExecutableCache, entry: *ExecutableCacheEntry) void {
        entry.ref_count += 1;
        self.touch(entry);
    }

    pub fn release(self: *ExecutableCache, backend: backend_api.Backend, entry: *ExecutableCacheEntry) void {
        if (entry.ref_count != 0) entry.ref_count -= 1;
        self.touch(entry);
        self.evictIdleUntilUnderLimit(backend);
    }

    fn get(self: *ExecutableCache, fingerprint: []const u8) ?*ExecutableCacheEntry {
        return self.entries.get(fingerprint);
    }

    fn touch(self: *ExecutableCache, entry: *ExecutableCacheEntry) void {
        self.next_use +|= 1;
        entry.last_use = self.next_use;
    }

    fn evictIdleUntilUnderLimit(self: *ExecutableCache, backend: backend_api.Backend) void {
        while (self.stats.resident_bytes > self.max_resident_bytes) {
            const victim = self.pressureEvictionVictim() orelse return;
            self.evictIdleEntry(backend, victim);
        }
    }

    fn pressureEvictionVictim(self: *ExecutableCache) ?*ExecutableCacheEntry {
        var victim: ?*ExecutableCacheEntry = null;
        var it = self.entries.valueIterator();
        while (it.next()) |entry_ptr| {
            const entry = entry_ptr.*;
            if (entry.backend_executable == null or entry.ref_count != 0) continue;
            const current_victim = victim orelse {
                victim = entry;
                continue;
            };
            if (entry.resident_bytes > current_victim.resident_bytes) {
                victim = entry;
                continue;
            }
            if (entry.resident_bytes == current_victim.resident_bytes and entry.compile_latency_us < current_victim.compile_latency_us) {
                victim = entry;
                continue;
            }
            if (entry.resident_bytes == current_victim.resident_bytes and entry.compile_latency_us == current_victim.compile_latency_us and entry.last_use < current_victim.last_use) {
                victim = entry;
                continue;
            }
            if (entry.resident_bytes == current_victim.resident_bytes and entry.compile_latency_us == current_victim.compile_latency_us and entry.last_use == current_victim.last_use and std.mem.lessThan(u8, entry.fingerprint, current_victim.fingerprint)) {
                victim = entry;
            }
        }
        return victim;
    }

    fn evictIdleEntry(self: *ExecutableCache, backend: backend_api.Backend, entry: *ExecutableCacheEntry) void {
        const handle = entry.backend_executable orelse return;
        if (entry.ref_count != 0) return;
        backend.destroyExecutable(handle);
        entry.backend_executable = null;
        self.stats.evicted_resident_bytes +|= entry.resident_bytes;
        self.stats.resident_bytes = if (entry.resident_bytes > self.stats.resident_bytes) 0 else self.stats.resident_bytes - entry.resident_bytes;
        entry.resident_bytes = 0;
        if (self.stats.resident_entries != 0) self.stats.resident_entries -= 1;
        self.stats.evictions += 1;
    }
};

pub const Client = struct {
    allocator: std.mem.Allocator,
    backend: backend_api.Backend,
    backend_kind: BackendKind,
    devices: []Device,
    memories: []Memory,
    device_handles: []*Device,
    memory_handles: []*Memory,
    topology: Topology,
    executable_cache: ExecutableCache,
    executable_cache_mutex: std.atomic.Mutex = .unlocked,

    pub fn init(allocator: std.mem.Allocator, backend_impl: backend_api.Backend, device_count: usize) !*Client {
        if (device_count == 0 or device_count > MAX_DEVICES) return error.InvalidDeviceCount;
        const backend_kind = backend_impl.kind();

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
            .backend_kind = backend_kind,
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

    pub fn lookupDevice(self: *const Client, id: i32) ?*const Device {
        for (self.devices) |*device| {
            if (device.id == id) return device;
        }
        return null;
    }

    pub fn lookupMemory(self: *const Client, id: i32) ?*const Memory {
        for (self.memories) |*memory| {
            if (memory.id == id) return memory;
        }
        return null;
    }

    pub fn recordExecutableCompile(self: *Client, fingerprint: []const u8) !bool {
        lockExecutableCache(self);
        defer self.executable_cache_mutex.unlock();
        const hit = try self.executable_cache.recordCompile(fingerprint);
        self.syncExecutableCacheMemoryStats();
        return hit;
    }

    pub fn setExecutableCacheMaxResidentBytes(self: *Client, max_resident_bytes: u64) void {
        lockExecutableCache(self);
        defer self.executable_cache_mutex.unlock();
        self.executable_cache.setMaxResidentBytes(self.backend, max_resident_bytes);
        self.syncExecutableCacheMemoryStats();
    }

    fn lockExecutableCache(self: *Client) void {
        while (!self.executable_cache_mutex.tryLock()) std.atomic.spinLoopHint();
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

    pub fn trimExecutableCacheForAllocation(self: *Client, memory: *Memory, allocation_bytes: usize) ExecutableCacheTrim {
        lockExecutableCache(self);
        defer self.executable_cache_mutex.unlock();
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

    pub fn acquireCachedBackendExecutable(
        self: *Client,
        allocator: std.mem.Allocator,
        fingerprint: []const u8,
        plan: *const core.ExecutablePlan,
        device_local_hardware_ids: []const i32,
    ) !?CachedBackendExecutable {
        const entry = blk: {
            lockExecutableCache(self);
            defer self.executable_cache_mutex.unlock();
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
            defer self.executable_cache_mutex.unlock();
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

        const compile_start_ns = nowNs();
        const handle = self.backend.compileExecutable(allocator, plan, device_local_hardware_ids) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => null,
        };
        const compile_latency_us = elapsedMicrosSince(compile_start_ns);
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
        defer self.executable_cache_mutex.unlock();
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

    pub fn releaseCachedBackendExecutable(self: *Client, entry: *ExecutableCacheEntry) void {
        lockExecutableCache(self);
        defer self.executable_cache_mutex.unlock();
        self.executable_cache.release(self.backend, entry);
        self.syncExecutableCacheMemoryStats();
    }
};

pub const GraphNodeKind = enum {
    constant,
    parameter,
    compute,
    collective,
    custom_call,
    control_flow,
    structural,
};

pub const GraphNode = struct {
    instruction_index: usize,
    device_index: usize,
    device_id: i32,
    kind: GraphNodeKind,
};

pub const GraphExecuteResult = struct {
    outputs: []*Buffer,
    completion_event: Event = Event.ready(),
};

pub const LoweringOptions = struct {
    diagnostic_writer: ?*std.Io.Writer = null,
    cache_fingerprint: ?[]const u8 = null,
};

pub const LoweringPipeline = struct {
    backend_executable_ready: bool,
    backend_executable_cache_reused: bool = false,
    lowered_instruction_count: usize,
};

pub const ExecutableGraph = struct {
    allocator: std.mem.Allocator,
    backend: backend_api.Backend,
    device_ids: []i32,
    device_local_hardware_ids: []i32,
    nodes: []GraphNode,
    backend_executable: ?backend_api.ExecutableHandle = null,
    backend_executable_cache_entry: ?*ExecutableCacheEntry = null,
    backend_executable_cache_owner: ?*Client = null,
    lowering: LoweringPipeline,
    last_compile_cache_trim: ExecutableCacheTrim = .{},
    last_execute_cache_trim: ExecutableCacheTrim = .{},
    last_backend_completion: backend_api.ExecutionCompletion = .{},
    donation_alias_stats: DonationAliasStats = .{},

    pub fn init(allocator: std.mem.Allocator, client: *const Client, plan: *const core.ExecutablePlan) !ExecutableGraph {
        return initWithOptions(allocator, client, plan, .{});
    }

    pub fn initWithOptions(allocator: std.mem.Allocator, client: *const Client, plan: *const core.ExecutablePlan, options: LoweringOptions) !ExecutableGraph {
        const device_count = plan.options.numDevices();
        if (device_count == 0 or device_count > client.devices.len) return error.InvalidGraph;

        const device_ids = try allocator.alloc(i32, device_count);
        errdefer allocator.free(device_ids);
        const device_local_hardware_ids = try allocator.alloc(i32, device_count);
        errdefer allocator.free(device_local_hardware_ids);
        for (device_ids, 0..) |*device_id, i| {
            device_id.* = if (plan.options.device_assignment.len != 0)
                plan.options.device_assignment[i]
            else
                client.devices[i].id;
            const device = client.lookupDevice(device_id.*) orelse return error.InvalidGraph;
            device_local_hardware_ids[i] = device.local_hardware_id;
        }

        const node_count = std.math.mul(usize, plan.instructions.len, device_count) catch return error.InvalidGraph;
        const nodes = try allocator.alloc(GraphNode, node_count);
        errdefer allocator.free(nodes);

        var out: usize = 0;
        for (0..device_count) |device_index| {
            for (plan.instructions, 0..) |instruction, instruction_index| {
                nodes[out] = .{
                    .instruction_index = instruction_index,
                    .device_index = device_index,
                    .device_id = device_ids[device_index],
                    .kind = graphNodeKind(instruction.kind),
                };
                out += 1;
            }
        }

        var backend_executable_cache_entry: ?*ExecutableCacheEntry = null;
        var backend_executable_cache_owner: ?*Client = null;
        var backend_executable_cache_reused = false;
        var last_compile_cache_trim = ExecutableCacheTrim{};
        const backend_executable = if (options.cache_fingerprint) |fingerprint| blk: {
            const mutable_client = @constCast(client);
            const cached = try mutable_client.acquireCachedBackendExecutable(allocator, fingerprint, plan, device_local_hardware_ids);
            if (cached) |entry| {
                backend_executable_cache_entry = entry.entry;
                backend_executable_cache_owner = mutable_client;
                backend_executable_cache_reused = entry.reused;
                last_compile_cache_trim = entry.compile_trim;
                break :blk entry.handle;
            }
            break :blk null;
        } else client.backend.compileExecutable(allocator, plan, device_local_hardware_ids) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => null,
        };
        errdefer if (backend_executable) |handle| {
            if (backend_executable_cache_entry) |entry| @constCast(client).releaseCachedBackendExecutable(entry) else client.backend.destroyExecutable(handle);
        };
        if (backend_executable == null) {
            if (options.diagnostic_writer) |writer| {
                client.backend.writeExecutableLoweringDiagnostic(plan, device_local_hardware_ids, writer) catch {};
            }
            return error.UnsupportedRuntimeFeature;
        }

        return .{
            .allocator = allocator,
            .backend = client.backend,
            .device_ids = device_ids,
            .device_local_hardware_ids = device_local_hardware_ids,
            .nodes = nodes,
            .backend_executable = backend_executable,
            .backend_executable_cache_entry = backend_executable_cache_entry,
            .backend_executable_cache_owner = backend_executable_cache_owner,
            .lowering = .{
                .backend_executable_ready = true,
                .backend_executable_cache_reused = backend_executable_cache_reused,
                .lowered_instruction_count = plan.instructions.len,
            },
            .last_compile_cache_trim = last_compile_cache_trim,
        };
    }

    pub fn deinit(self: *ExecutableGraph) void {
        if (self.backend_executable) |handle| {
            if (self.backend_executable_cache_entry) |entry| {
                if (self.backend_executable_cache_owner) |client| {
                    client.releaseCachedBackendExecutable(entry);
                } else if (entry.ref_count != 0) {
                    entry.ref_count -= 1;
                }
            } else {
                self.backend.destroyExecutable(handle);
            }
        }
        self.allocator.free(self.device_local_hardware_ids);
        self.allocator.free(self.nodes);
        self.allocator.free(self.device_ids);
        self.* = undefined;
    }

    pub fn backendExecutableStats(self: *const ExecutableGraph) ?backend_api.ExecutableStats {
        const handle = self.backend_executable orelse return null;
        var stats = self.backend.executableStats(handle);
        stats.donation_alias_output_count += self.donation_alias_stats.output_count;
        stats.donation_alias_output_bytes += self.donation_alias_stats.output_bytes;
        return stats;
    }

    pub fn executeDevice(
        self: *ExecutableGraph,
        allocator: std.mem.Allocator,
        client: *Client,
        plan: *const core.ExecutablePlan,
        device_index: usize,
        arguments: []const *Buffer,
    ) GraphExecuteError!GraphExecuteResult {
        if (device_index >= self.device_ids.len or device_index >= client.devices.len) return error.InvalidArgument;
        if (arguments.len != plan.parameter_shardings.len) return error.InvalidArgument;
        for (arguments) |argument| {
            argument.ensureUsable() catch |err| return mapBufferError(err);
            argument.ensureReady() catch |err| return mapBufferError(err);
            if (!self.argumentMatchesDevice(device_index, argument)) return error.InvalidArgument;
        }
        if (self.tryExecuteBackendExecutable(allocator, client, plan, device_index, arguments) catch |err| return err) |result| {
            return result;
        }
        return error.UnsupportedRuntimeFeature;
    }

    fn argumentMatchesDevice(self: *const ExecutableGraph, device_index: usize, argument: *const Buffer) bool {
        if (device_index >= self.device_ids.len) return false;
        if (argument.device_id != self.device_ids[device_index]) return false;
        if (argument.shard_index != device_index) return false;
        return true;
    }

    fn tryExecuteBackendExecutable(
        self: *ExecutableGraph,
        allocator: std.mem.Allocator,
        client: *Client,
        plan: *const core.ExecutablePlan,
        device_index: usize,
        arguments: []const *Buffer,
    ) GraphExecuteError!?GraphExecuteResult {
        const backend_executable = self.backend_executable orelse return null;
        var argument_handles = try allocator.alloc(backend_api.BufferHandle, arguments.len);
        defer allocator.free(argument_handles);
        for (arguments, 0..) |argument, i| {
            argument.ensureUsable() catch |err| return mapBufferError(err);
            argument.ensureReady() catch |err| return mapBufferError(err);
            argument_handles[i] = argument.backend_buffer orelse return null;
        }

        const device = &client.devices[device_index];
        const memory = device.default_memory;
        self.last_execute_cache_trim = client.trimExecutableCacheForAllocation(memory, planOutputBytes(plan) catch return error.Internal);

        const backend_result = client.backend.executeExecutable(allocator, backend_executable, device_index, argument_handles) catch |err| return mapBufferError(err);
        const owned_backend_result = backend_result orelse return null;
        const owned_backend_outputs = owned_backend_result.outputs;
        self.last_backend_completion = owned_backend_result.completion;
        const completion_event = runtimeEventFromBackendCompletion(client.backend, owned_backend_result.completion);
        defer allocator.free(owned_backend_outputs);

        var wrapped_backend_outputs: usize = 0;
        errdefer {
            for (owned_backend_outputs[wrapped_backend_outputs..]) |output| client.backend.destroyBuffer(output.handle);
        }

        if (owned_backend_outputs.len != plan.output_ids.len) return error.Internal;
        for (owned_backend_outputs, 0..) |output, output_index| {
            if (!backendOutputMatchesPlan(plan, output_index, output)) return error.Internal;
        }

        const outputs = try allocator.alloc(*Buffer, owned_backend_outputs.len);
        errdefer allocator.free(outputs);
        var initialized: usize = 0;
        errdefer {
            for (outputs[0..initialized]) |buffer| buffer.deinit();
        }

        var donation_alias_delta: DonationAliasStats = .{};
        errdefer self.donation_alias_stats.output_count -|= donation_alias_delta.output_count;
        errdefer self.donation_alias_stats.output_bytes -|= donation_alias_delta.output_bytes;

        for (owned_backend_outputs, 0..) |output, i| {
            const alias = donatedParameterAliasForOutput(plan, i);
            const backend_handle = if (alias) |alias_info| blk: {
                if (alias_info.parameter_index >= arguments.len) return error.Internal;
                const donated_argument = arguments[alias_info.parameter_index];
                if (donated_argument.element_type != output.element_type or !std.mem.eql(i64, donated_argument.dims, output.dims)) return error.Internal;
                const donated_handle = donated_argument.backend_buffer orelse return error.Internal;
                if (alias_info.kind == .donation and output.handle != donated_handle) break :blk output.handle;
                if (output.handle != donated_handle) client.backend.destroyBuffer(output.handle);
                const transferred = donated_argument.takeBackendStorageForDonationAlias() catch |err| return mapBufferError(err);
                donation_alias_delta.output_count += 1;
                donation_alias_delta.output_bytes += output.byte_size;
                self.donation_alias_stats.output_count += 1;
                self.donation_alias_stats.output_bytes += output.byte_size;
                break :blk transferred;
            } else output.handle;

            outputs[i] = Buffer.initBackendHandle(
                allocator,
                client.backend,
                output.element_type,
                output.dims,
                device,
                memory,
                device_index,
                output.byte_size,
                backend_handle,
            ) catch |err| {
                if (alias != null) client.backend.destroyBuffer(backend_handle);
                return mapBufferError(err);
            };
            initialized += 1;
            wrapped_backend_outputs = initialized;
        }
        return .{
            .outputs = outputs,
            .completion_event = completion_event,
        };
    }
};

fn runtimeEventFromBackendCompletion(backend_impl: backend_api.Backend, completion: backend_api.ExecutionCompletion) Event {
    return switch (completion.kind) {
        .completed => Event.ready(),
        .pending => blk: {
            const backend_event = completion.backend_event orelse break :blk Event.failed("backend returned asynchronous completion without an event handle");
            defer backend_impl.destroyExecutionEvent(backend_event);
            const status = backend_impl.executionEventStatus(backend_event) catch break :blk Event.failed("backend execution event status query failed");
            break :blk switch (status.state) {
                .ready => Event.ready(),
                .failed => Event.failed(if (status.message.len == 0) "backend execution event failed" else status.message),
                .pending => Event.failed("backend execution event is pending without runtime scheduler integration"),
            };
        },
    };
}

fn backendOutputMatchesPlan(plan: *const core.ExecutablePlan, output_index: usize, output: backend_api.ExecutableOutput) bool {
    if (output_index >= plan.output_ids.len) return false;
    const value_id = plan.output_ids[output_index];
    if (value_id.index >= plan.values.len) return false;
    const descriptor = plan.values[value_id.index].descriptor;
    if (output.element_type != descriptor.element_type) return false;
    if (!std.mem.eql(i64, output.dims, descriptor.dims)) return false;
    if (output.byte_size != core.denseByteSize(descriptor.element_type, descriptor.dims)) return false;
    return true;
}

const DonationAlias = struct {
    parameter_index: usize,
    kind: core.OutputAliasKind,
};

fn donatedParameterAliasForOutput(plan: *const core.ExecutablePlan, output_index: usize) ?DonationAlias {
    for (plan.output_aliases) |alias| {
        if (alias.output_index == output_index and planDonatesParameter(plan, alias.parameter_index)) {
            return .{ .parameter_index = alias.parameter_index, .kind = alias.kind };
        }
    }
    if (output_index >= plan.output_ids.len) return null;
    const output_id = plan.output_ids[output_index];
    var parameter_index: usize = 0;
    for (plan.values) |value| {
        if (value.role != .parameter) continue;
        if (value.id.index == output_id.index) {
            return if (planDonatesParameter(plan, parameter_index)) .{ .parameter_index = parameter_index, .kind = .identity } else null;
        }
        parameter_index += 1;
    }
    return null;
}

fn planDonatesParameter(plan: *const core.ExecutablePlan, parameter_index: usize) bool {
    for (plan.donated_parameter_indices) |candidate| {
        if (candidate == parameter_index) return true;
    }
    return false;
}

fn planOutputBytes(plan: *const core.ExecutablePlan) !usize {
    var total: usize = 0;
    for (plan.output_ids) |value_id| {
        if (value_id.index >= plan.values.len) return error.InvalidGraph;
        const descriptor = plan.values[value_id.index].descriptor;
        total = try std.math.add(usize, total, core.denseByteSize(descriptor.element_type, descriptor.dims));
    }
    return total;
}

pub const GraphExecuteError = error{
    OutOfMemory,
    InvalidArgument,
    UnsupportedElementType,
    ShapeMismatch,
    UnsupportedRuntimeFeature,
    BufferDeleted,
    BufferDonated,
    BufferNotReady,
    BufferReadinessFailed,
    Internal,
};

fn graphNodeKind(kind: core.PlanInstructionKind) GraphNodeKind {
    return switch (kind) {
        .constant => .constant,
        .custom_call => .custom_call,
        .while_ => .control_flow,
        .tuple, .get_tuple_element, .optimization_barrier => .structural,
        else => .compute,
    };
}

fn mapBufferError(err: anyerror) GraphExecuteError {
    return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.UnsupportedElementType => error.UnsupportedElementType,
        error.ShapeMismatch => error.ShapeMismatch,
        error.InvalidArgument => error.InvalidArgument,
        error.UnsupportedRuntimeFeature => error.UnsupportedRuntimeFeature,
        error.BufferDeleted => error.BufferDeleted,
        error.BufferDonated => error.BufferDonated,
        error.BufferNotReady => error.BufferNotReady,
        error.BufferReadinessFailed => error.BufferReadinessFailed,
        else => error.Internal,
    };
}

pub const Buffer = struct {
    allocator: std.mem.Allocator,
    backend: backend_api.Backend,
    backend_kind: BackendKind,
    element_type: BufferType,
    dims: []i64,
    device_id: i32,
    memory_id: i32,
    device: *Device,
    memory: *Memory,
    shard_index: usize,
    byte_size: usize,
    bytes: []u8,
    backend_buffer: ?backend_api.BufferHandle = null,
    state: BufferState = .live,
    deleted: bool = false,
    accounted_bytes: usize = 0,
    ready_event: Event = Event.ready(),

    fn initBackendOnly(
        allocator: std.mem.Allocator,
        src: *Buffer,
        element_type: BufferType,
        dims_: []const i64,
        byte_size: usize,
        backend_buffer: backend_api.BufferHandle,
        shard_index: usize,
    ) !*Buffer {
        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, dims_);
        errdefer allocator.free(dims);

        const bytes = try allocator.alloc(u8, 0);
        errdefer allocator.free(bytes);

        buffer.* = .{
            .allocator = allocator,
            .backend = src.backend,
            .backend_kind = src.backend_kind,
            .element_type = element_type,
            .dims = dims,
            .device_id = src.device_id,
            .memory_id = src.memory_id,
            .device = src.device,
            .memory = src.memory,
            .shard_index = shard_index,
            .byte_size = byte_size,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
        };
        return buffer.accountDeviceBytes();
    }

    pub fn initHostCopyForBackend(
        allocator: std.mem.Allocator,
        backend_impl: backend_api.Backend,
        element_type: BufferType,
        dims_: []const i64,
        device: *Device,
        memory: *Memory,
        shard_index: usize,
        src: []const u8,
    ) !*Buffer {
        if (!memory.isAddressableBy(device)) return error.InvalidArgument;

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, dims_);
        errdefer allocator.free(dims);

        const backend_buffer = try backend_impl.bufferFromHost(device.local_hardware_id, element_type, dims, src) orelse
            return error.UnsupportedRuntimeFeature;
        errdefer backend_impl.destroyBuffer(backend_buffer);

        const bytes = try allocator.alloc(u8, 0);
        errdefer allocator.free(bytes);

        buffer.* = .{
            .allocator = allocator,
            .backend = backend_impl,
            .backend_kind = backend_impl.kind(),
            .element_type = element_type,
            .dims = dims,
            .device_id = device.id,
            .memory_id = memory.id,
            .device = device,
            .memory = memory,
            .shard_index = shard_index,
            .byte_size = src.len,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
        };
        memory.stats.host_to_device_bytes += @intCast(src.len);
        return buffer.accountDeviceBytes();
    }

    pub fn initDeviceAllocationForBackend(
        allocator: std.mem.Allocator,
        backend_impl: backend_api.Backend,
        element_type: BufferType,
        dims_: []const i64,
        device: *Device,
        memory: *Memory,
        shard_index: usize,
    ) !*Buffer {
        if (!memory.isAddressableBy(device)) return error.InvalidArgument;

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, dims_);
        errdefer allocator.free(dims);

        const byte_size = core.denseByteSize(element_type, dims);
        const backend_buffer = try backend_impl.allocateBuffer(device.local_hardware_id, element_type, dims) orelse
            return error.UnsupportedRuntimeFeature;
        errdefer backend_impl.destroyBuffer(backend_buffer);

        const bytes = try allocator.alloc(u8, 0);
        errdefer allocator.free(bytes);

        buffer.* = .{
            .allocator = allocator,
            .backend = backend_impl,
            .backend_kind = backend_impl.kind(),
            .element_type = element_type,
            .dims = dims,
            .device_id = device.id,
            .memory_id = memory.id,
            .device = device,
            .memory = memory,
            .shard_index = shard_index,
            .byte_size = byte_size,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
        };
        return buffer.accountDeviceBytes();
    }

    pub fn initPendingBackendTransfer(
        allocator: std.mem.Allocator,
        backend_impl: backend_api.Backend,
        element_type: BufferType,
        dims_: []const i64,
        device: *Device,
        memory: *Memory,
        shard_index: usize,
    ) !*Buffer {
        if (!memory.isAddressableBy(device)) return error.InvalidArgument;

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, dims_);
        errdefer allocator.free(dims);

        const bytes = try allocator.alloc(u8, 0);
        errdefer allocator.free(bytes);

        buffer.* = .{
            .allocator = allocator,
            .backend = backend_impl,
            .backend_kind = backend_impl.kind(),
            .element_type = element_type,
            .dims = dims,
            .device_id = device.id,
            .memory_id = memory.id,
            .device = device,
            .memory = memory,
            .shard_index = shard_index,
            .byte_size = core.denseByteSize(element_type, dims),
            .bytes = bytes,
            .backend_buffer = null,
            .ready_event = Event.pending(),
        };
        return buffer;
    }

    pub fn initDeviceCopy(
        allocator: std.mem.Allocator,
        src: *Buffer,
        shard_index: usize,
    ) !*Buffer {
        try src.ensureUsable();
        const src_backend = src.backend_buffer orelse return error.UnsupportedRuntimeFeature;
        const backend_buffer = try src.backend.cloneBuffer(src_backend) orelse return error.UnsupportedRuntimeFeature;
        errdefer src.backend.destroyBuffer(backend_buffer);

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, src.dims);
        errdefer allocator.free(dims);

        const bytes = try allocator.alloc(u8, 0);
        errdefer allocator.free(bytes);

        buffer.* = .{
            .allocator = allocator,
            .backend = src.backend,
            .backend_kind = src.backend_kind,
            .element_type = src.element_type,
            .dims = dims,
            .device_id = src.device_id,
            .memory_id = src.memory_id,
            .device = src.device,
            .memory = src.memory,
            .shard_index = shard_index,
            .byte_size = src.byte_size,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
        };
        return buffer.accountDeviceBytes();
    }

    pub fn initPartitionId(
        allocator: std.mem.Allocator,
        backend_impl: backend_api.Backend,
        output_type: BufferType,
        output_dims: []const i64,
        device: *Device,
        memory: *Memory,
        partition_id: u32,
        shard_index: usize,
    ) !*Buffer {
        if (output_dims.len != 0) return error.ShapeMismatch;
        if (output_type != .u32 and output_type != .s32) return error.UnsupportedElementType;
        if (!memory.isAddressableBy(device)) return error.InvalidArgument;

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, output_dims);
        errdefer allocator.free(dims);

        const backend_buffer = try backend_impl.partitionId(device.local_hardware_id, output_type, partition_id) orelse return error.UnsupportedRuntimeFeature;
        errdefer backend_impl.destroyBuffer(backend_buffer);
        const storage_bytes = try allocator.alloc(u8, 0);
        errdefer allocator.free(storage_bytes);

        buffer.* = .{
            .allocator = allocator,
            .backend = backend_impl,
            .backend_kind = backend_impl.kind(),
            .element_type = output_type,
            .dims = dims,
            .device_id = device.id,
            .memory_id = memory.id,
            .device = device,
            .memory = memory,
            .shard_index = shard_index,
            .byte_size = output_type.byteSize(),
            .bytes = storage_bytes,
            .backend_buffer = backend_buffer,
        };
        return buffer.accountDeviceBytes();
    }

    pub fn initCholesky(
        allocator: std.mem.Allocator,
        src: *Buffer,
        lower: bool,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (src.element_type != .f32) return error.UnsupportedElementType;
        if (!std.mem.eql(i64, src.dims, output_dims) or output_dims.len < 2) return error.ShapeMismatch;
        const n_i64 = output_dims[output_dims.len - 1];
        const rows_i64 = output_dims[output_dims.len - 2];
        if (n_i64 <= 0 or rows_i64 != n_i64) return error.ShapeMismatch;
        const backend_src = src.backend_buffer orelse return error.UnsupportedRuntimeFeature;
        const output_byte_size = denseByteSize(.f32, output_dims);
        if (src.backend.cholesky(backend_src, lower, output_dims) catch null) |backend_buffer| {
            return initBackendOnly(allocator, src, .f32, output_dims, output_byte_size, backend_buffer, shard_index);
        }
        return error.UnsupportedRuntimeFeature;
    }

    pub fn initRngUniform(
        allocator: std.mem.Allocator,
        min: *Buffer,
        max: *Buffer,
        output_type: BufferType,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        _ = allocator;
        _ = output_dims;
        _ = shard_index;
        if (output_type != .f32 and output_type != .u32 and output_type != .s32) return error.UnsupportedElementType;
        if (min.element_type != max.element_type or min.element_type != output_type) return error.UnsupportedElementType;
        return error.UnsupportedRuntimeFeature;
    }

    pub fn initRngBits(
        allocator: std.mem.Allocator,
        state: *Buffer,
        output_type: BufferType,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        _ = allocator;
        _ = state;
        _ = output_dims;
        _ = shard_index;
        if (output_type != .u32 and output_type != .s32) return error.UnsupportedElementType;
        return error.UnsupportedRuntimeFeature;
    }

    pub fn initRngStateUpdate(
        allocator: std.mem.Allocator,
        state: *Buffer,
        shard_index: usize,
    ) !*Buffer {
        _ = allocator;
        _ = shard_index;
        if (state.element_type.byteSize() == 0) return error.UnsupportedElementType;
        return error.UnsupportedRuntimeFeature;
    }

    pub fn initElementwiseBinary(
        allocator: std.mem.Allocator,
        op: ElementwiseBinaryOp,
        lhs: *Buffer,
        rhs: *Buffer,
        shard_index: usize,
    ) !*Buffer {
        if (lhs.element_type != rhs.element_type) return error.UnsupportedElementType;
        if (lhs.element_type.byteSize() == 0) return error.UnsupportedElementType;
        if (!std.mem.eql(i64, lhs.dims, rhs.dims) or lhs.byte_size != rhs.byte_size) return error.ShapeMismatch;
        if (lhs.backend_buffer != null and rhs.backend_buffer != null) {
            if (lhs.backend.binary(lhs.backend_buffer.?, rhs.backend_buffer.?, op) catch null) |backend_buffer| {
                return initBackendOnly(allocator, lhs, lhs.element_type, lhs.dims, lhs.byte_size, backend_buffer, shard_index);
            }
        }
        return error.UnsupportedRuntimeFeature;
    }

    pub fn initU8Add(
        allocator: std.mem.Allocator,
        lhs: *Buffer,
        rhs: *Buffer,
        shard_index: usize,
    ) !*Buffer {
        return initElementwiseBinary(allocator, .add, lhs, rhs, shard_index);
    }

    pub fn initU8Binary(
        allocator: std.mem.Allocator,
        op: ElementwiseBinaryOp,
        lhs: *Buffer,
        rhs: *Buffer,
        shard_index: usize,
    ) !*Buffer {
        if (lhs.element_type != .u8 or rhs.element_type != .u8) return error.UnsupportedElementType;
        return initElementwiseBinary(allocator, op, lhs, rhs, shard_index);
    }

    pub fn initElementwiseUnary(
        allocator: std.mem.Allocator,
        op: ElementwiseUnaryOp,
        src: *Buffer,
        shard_index: usize,
    ) !*Buffer {
        return initElementwiseUnaryTyped(allocator, op, src, src.element_type, src.dims, shard_index);
    }

    pub fn initElementwiseUnaryTyped(
        allocator: std.mem.Allocator,
        op: ElementwiseUnaryOp,
        src: *Buffer,
        output_type: BufferType,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (!std.mem.eql(i64, src.dims, output_dims)) return error.ShapeMismatch;
        if (src.element_type.byteSize() == 0 or output_type.byteSize() == 0) return error.UnsupportedElementType;
        const output_byte_size = denseByteSize(output_type, output_dims);
        if (src.backend_buffer) |src_backend| {
            if (src.backend.unary(src_backend, op) catch null) |backend_buffer| {
                return initBackendOnly(allocator, src, output_type, output_dims, output_byte_size, backend_buffer, shard_index);
            }
        }
        return error.UnsupportedRuntimeFeature;
    }

    pub fn initReshape(
        allocator: std.mem.Allocator,
        src: *Buffer,
        new_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (src.element_type.byteSize() == 0) return error.UnsupportedElementType;
        if (denseByteSize(src.element_type, new_dims) != src.byte_size) return error.ShapeMismatch;
        if (src.backend_buffer) |src_backend| {
            if (src.backend.reshape(src_backend, new_dims) catch null) |backend_buffer| {
                return initBackendOnly(allocator, src, src.element_type, new_dims, src.byte_size, backend_buffer, shard_index);
            }
        }
        return error.UnsupportedRuntimeFeature;
    }

    pub fn initTranspose(
        allocator: std.mem.Allocator,
        src: *Buffer,
        permutation: []const i64,
        new_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (!validPermutation(permutation, src.dims.len)) return error.ShapeMismatch;
        if (new_dims.len != permutation.len) return error.ShapeMismatch;
        if (denseByteSize(src.element_type, new_dims) != src.byte_size) return error.ShapeMismatch;
        const element_size = src.element_type.byteSize();
        if (element_size == 0) return error.UnsupportedElementType;
        if (src.backend_buffer) |src_backend| {
            if (src.backend.transpose(src_backend, permutation) catch null) |backend_buffer| {
                return initBackendOnly(allocator, src, src.element_type, new_dims, src.byte_size, backend_buffer, shard_index);
            }
        }
        return error.UnsupportedRuntimeFeature;
    }

    pub fn initBroadcastInDim(
        allocator: std.mem.Allocator,
        src: *Buffer,
        broadcast_dimensions: []const i64,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (!validBroadcastDimensions(broadcast_dimensions, src.dims, output_dims)) return error.ShapeMismatch;
        const element_size = src.element_type.byteSize();
        if (element_size == 0) return error.UnsupportedElementType;
        const output_byte_size = denseByteSize(src.element_type, output_dims);
        if (src.backend_buffer) |src_backend| {
            if (src.backend.broadcastInDim(src_backend, broadcast_dimensions, output_dims) catch null) |backend_buffer| {
                return initBackendOnly(allocator, src, src.element_type, output_dims, output_byte_size, backend_buffer, shard_index);
            }
        }
        return error.UnsupportedRuntimeFeature;
    }

    pub fn initSlice(
        allocator: std.mem.Allocator,
        src: *Buffer,
        start_indices: []const i64,
        limit_indices: []const i64,
        strides_: []const i64,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (!validSlice(start_indices, limit_indices, strides_, src.dims, output_dims)) return error.ShapeMismatch;
        const element_size = src.element_type.byteSize();
        if (element_size == 0) return error.UnsupportedElementType;
        const output_byte_size = denseByteSize(src.element_type, output_dims);
        if (src.backend_buffer) |src_backend| {
            if (src.backend.slice(src_backend, start_indices, limit_indices, strides_, output_dims) catch null) |backend_buffer| {
                return initBackendOnly(allocator, src, src.element_type, output_dims, output_byte_size, backend_buffer, shard_index);
            }
        }
        return error.UnsupportedRuntimeFeature;
    }

    pub fn initConcatenate(
        allocator: std.mem.Allocator,
        lhs: *Buffer,
        rhs: *Buffer,
        dimension: i64,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (lhs.element_type != rhs.element_type) return error.UnsupportedElementType;
        if (!validConcatenate(lhs.dims, rhs.dims, dimension, output_dims)) return error.ShapeMismatch;
        const element_size = lhs.element_type.byteSize();
        if (element_size == 0) return error.UnsupportedElementType;
        const output_byte_size = denseByteSize(lhs.element_type, output_dims);
        if (lhs.backend_buffer != null and rhs.backend_buffer != null) {
            if (lhs.backend.concatenate(lhs.backend_buffer.?, rhs.backend_buffer.?, dimension, output_dims) catch null) |backend_buffer| {
                return initBackendOnly(allocator, lhs, lhs.element_type, output_dims, output_byte_size, backend_buffer, shard_index);
            }
        }
        return error.UnsupportedRuntimeFeature;
    }

    pub fn initDotGeneral(
        allocator: std.mem.Allocator,
        lhs: *Buffer,
        rhs: *Buffer,
        lhs_batch_dimensions: []const i64,
        rhs_batch_dimensions: []const i64,
        lhs_contracting_dimensions: []const i64,
        rhs_contracting_dimensions: []const i64,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (lhs.element_type != rhs.element_type) return error.UnsupportedElementType;
        switch (lhs.element_type) {
            .f16, .bf16, .f32 => {},
            else => return error.UnsupportedElementType,
        }
        if (!validDotGeneral(lhs.dims, rhs.dims, lhs_batch_dimensions, rhs_batch_dimensions, lhs_contracting_dimensions, rhs_contracting_dimensions, output_dims)) return error.ShapeMismatch;
        const output_byte_size = denseByteSize(lhs.element_type, output_dims);
        if (lhs.backend_buffer != null and rhs.backend_buffer != null) {
            if (lhs.backend.dotGeneral(lhs.backend_buffer.?, rhs.backend_buffer.?, lhs_batch_dimensions, rhs_batch_dimensions, lhs_contracting_dimensions, rhs_contracting_dimensions, output_dims) catch null) |backend_buffer| {
                return initBackendOnly(allocator, lhs, lhs.element_type, output_dims, output_byte_size, backend_buffer, shard_index);
            }
        }
        return error.UnsupportedRuntimeFeature;
    }

    pub fn initReduce(
        allocator: std.mem.Allocator,
        kind: core.PlanInstructionKind,
        src: *Buffer,
        dimensions: []const i64,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        const output_type: BufferType = switch (kind) {
            .reduce_sum, .reduce_max, .reduce_min => switch (src.element_type) {
                .s8, .s32, .u8, .u16, .u32, .u64, .f16, .f32, .bf16 => src.element_type,
                else => return error.UnsupportedElementType,
            },
            .reduce_and, .reduce_or => if (src.element_type == .pred) .pred else return error.UnsupportedElementType,
            else => return error.UnsupportedElementType,
        };
        if (!validReduce(src.dims, dimensions, output_dims)) return error.ShapeMismatch;
        const output_byte_size = denseByteSize(output_type, output_dims);
        if (src.backend_buffer) |src_backend| {
            if (src.backend.reduce(src_backend, kind, dimensions, output_dims) catch null) |backend_buffer| {
                return initBackendOnly(allocator, src, output_type, output_dims, output_byte_size, backend_buffer, shard_index);
            }
        }
        return error.UnsupportedRuntimeFeature;
    }

    pub fn initCompare(
        allocator: std.mem.Allocator,
        lhs: *Buffer,
        rhs: *Buffer,
        direction: CompareOp,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (lhs.element_type != rhs.element_type) return error.UnsupportedElementType;
        if (!std.mem.eql(i64, lhs.dims, rhs.dims) or !std.mem.eql(i64, lhs.dims, output_dims)) return error.ShapeMismatch;
        const output_byte_size = denseByteSize(.pred, output_dims);
        if (lhs.backend_buffer != null and rhs.backend_buffer != null) {
            if (lhs.backend.compare(lhs.backend_buffer.?, rhs.backend_buffer.?, direction, output_dims) catch null) |backend_buffer| {
                return initBackendOnly(allocator, lhs, .pred, output_dims, output_byte_size, backend_buffer, shard_index);
            }
        }
        return error.UnsupportedRuntimeFeature;
    }

    pub fn initSelect(
        allocator: std.mem.Allocator,
        pred: *Buffer,
        on_true: *Buffer,
        on_false: *Buffer,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (pred.element_type != .pred) return error.UnsupportedElementType;
        if (on_true.element_type != on_false.element_type) return error.UnsupportedElementType;
        if (!std.mem.eql(i64, on_true.dims, on_false.dims) or !std.mem.eql(i64, on_true.dims, output_dims) or !std.mem.eql(i64, pred.dims, output_dims)) return error.ShapeMismatch;
        const element_size = on_true.element_type.byteSize();
        if (element_size == 0) return error.UnsupportedElementType;
        if (pred.backend_buffer != null and on_true.backend_buffer != null and on_false.backend_buffer != null) {
            if (on_true.backend.select(pred.backend_buffer.?, on_true.backend_buffer.?, on_false.backend_buffer.?, output_dims) catch null) |backend_buffer| {
                return initBackendOnly(allocator, on_true, on_true.element_type, output_dims, denseByteSize(on_true.element_type, output_dims), backend_buffer, shard_index);
            }
        }
        return error.UnsupportedRuntimeFeature;
    }

    pub fn initConvert(
        allocator: std.mem.Allocator,
        src: *Buffer,
        output_type: BufferType,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (!std.mem.eql(i64, src.dims, output_dims)) return error.ShapeMismatch;
        if (src.element_type.byteSize() == 0 or output_type.byteSize() == 0) return error.UnsupportedElementType;
        const output_size = denseByteSize(output_type, output_dims);
        if (src.backend_buffer) |src_backend| {
            if (src.backend.convert(src_backend, output_type) catch null) |backend_buffer| {
                return initBackendOnly(allocator, src, output_type, output_dims, output_size, backend_buffer, shard_index);
            }
        }
        return error.UnsupportedRuntimeFeature;
    }

    pub fn initIota(
        allocator: std.mem.Allocator,
        backend_impl: backend_api.Backend,
        element_type: BufferType,
        output_dims: []const i64,
        device: *Device,
        memory: *Memory,
        iota_dimension: i64,
        shard_index: usize,
    ) !*Buffer {
        if (iota_dimension < 0 or iota_dimension >= @as(i64, @intCast(output_dims.len))) return error.ShapeMismatch;
        if (element_type.byteSize() == 0) return error.UnsupportedElementType;
        if (backend_impl.iota(device.local_hardware_id, element_type, output_dims, iota_dimension) catch null) |backend_buffer| {
            return initBackendHandle(
                allocator,
                backend_impl,
                element_type,
                output_dims,
                device,
                memory,
                shard_index,
                denseByteSize(element_type, output_dims),
                backend_buffer,
            ) catch |err| {
                backend_impl.destroyBuffer(backend_buffer);
                return err;
            };
        }
        return error.UnsupportedRuntimeFeature;
    }

    pub fn initReverse(
        allocator: std.mem.Allocator,
        src: *Buffer,
        dimensions: []const i64,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (!std.mem.eql(i64, src.dims, output_dims)) return error.ShapeMismatch;
        const element_size = src.element_type.byteSize();
        if (element_size == 0) return error.UnsupportedElementType;
        if (src.backend_buffer) |src_backend| {
            if (src.backend.reverse(src_backend, dimensions, output_dims) catch null) |backend_buffer| {
                return initBackendOnly(allocator, src, src.element_type, output_dims, src.byte_size, backend_buffer, shard_index);
            }
        }
        return error.UnsupportedRuntimeFeature;
    }

    pub fn initClamp(
        allocator: std.mem.Allocator,
        min_buffer: *Buffer,
        value_buffer: *Buffer,
        max_buffer: *Buffer,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (value_buffer.element_type != min_buffer.element_type or value_buffer.element_type != max_buffer.element_type) return error.UnsupportedElementType;
        if (!std.mem.eql(i64, value_buffer.dims, output_dims)) return error.ShapeMismatch;
        const element_size = value_buffer.element_type.byteSize();
        if (element_size == 0) return error.UnsupportedElementType;
        const min_scalar = min_buffer.dims.len == 0;
        const max_scalar = max_buffer.dims.len == 0;
        if (!min_scalar and !std.mem.eql(i64, min_buffer.dims, output_dims)) return error.ShapeMismatch;
        if (!max_scalar and !std.mem.eql(i64, max_buffer.dims, output_dims)) return error.ShapeMismatch;
        if (min_buffer.backend_buffer) |min_backend| {
            if (value_buffer.backend_buffer) |value_backend| {
                if (max_buffer.backend_buffer) |max_backend| {
                    if (value_buffer.backend.clamp(min_backend, value_backend, max_backend, output_dims) catch null) |backend_buffer| {
                        return initBackendOnly(allocator, value_buffer, value_buffer.element_type, output_dims, denseByteSize(value_buffer.element_type, output_dims), backend_buffer, shard_index);
                    }
                }
            }
        }
        return error.UnsupportedRuntimeFeature;
    }

    pub fn initDynamicSlice(
        allocator: std.mem.Allocator,
        src: *Buffer,
        start_buffers: []const *Buffer,
        slice_sizes: []const i64,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (start_buffers.len != src.dims.len or slice_sizes.len != src.dims.len) return error.ShapeMismatch;
        if (src.backend_buffer) |src_backend| {
            const start_handles = try backendStartHandles(allocator, start_buffers);
            defer allocator.free(start_handles);
            if (start_handles.len == start_buffers.len) {
                if (src.backend.dynamicSlice(src_backend, start_handles, slice_sizes, output_dims) catch null) |backend_buffer| {
                    return initBackendOnly(allocator, src, src.element_type, output_dims, denseByteSize(src.element_type, output_dims), backend_buffer, shard_index);
                }
            }
        }
        return error.UnsupportedRuntimeFeature;
    }

    pub fn initDynamicUpdateSlice(
        allocator: std.mem.Allocator,
        src: *Buffer,
        update: *Buffer,
        start_buffers: []const *Buffer,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (src.element_type != update.element_type) return error.UnsupportedElementType;
        if (!std.mem.eql(i64, src.dims, output_dims) or src.dims.len != update.dims.len or start_buffers.len != src.dims.len) return error.ShapeMismatch;
        const element_size = src.element_type.byteSize();
        if (element_size == 0) return error.UnsupportedElementType;
        if (src.backend_buffer != null and update.backend_buffer != null) {
            const start_handles = try backendStartHandles(allocator, start_buffers);
            defer allocator.free(start_handles);
            if (start_handles.len == start_buffers.len) {
                if (src.backend.dynamicUpdateSlice(src.backend_buffer.?, update.backend_buffer.?, start_handles, output_dims) catch null) |backend_buffer| {
                    return initBackendOnly(allocator, src, src.element_type, output_dims, denseByteSize(src.element_type, output_dims), backend_buffer, shard_index);
                }
            }
        }
        return error.UnsupportedRuntimeFeature;
    }

    pub fn initPad(
        allocator: std.mem.Allocator,
        src: *Buffer,
        padding_value: *Buffer,
        edge_padding_low: []const i64,
        edge_padding_high: []const i64,
        interior_padding: []const i64,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (src.element_type != padding_value.element_type or padding_value.dims.len != 0) return error.UnsupportedElementType;
        const rank = src.dims.len;
        if (edge_padding_low.len != rank or edge_padding_high.len != rank or interior_padding.len != rank or output_dims.len != rank) return error.ShapeMismatch;
        const element_size = src.element_type.byteSize();
        if (element_size == 0) return error.UnsupportedElementType;
        if (src.backend_buffer != null and padding_value.backend_buffer != null) {
            if (src.backend.pad(src.backend_buffer.?, padding_value.backend_buffer.?, edge_padding_low, edge_padding_high, interior_padding, output_dims) catch null) |backend_buffer| {
                return initBackendOnly(allocator, src, src.element_type, output_dims, denseByteSize(src.element_type, output_dims), backend_buffer, shard_index);
            }
        }
        return error.UnsupportedRuntimeFeature;
    }

    pub fn initGather(
        allocator: std.mem.Allocator,
        operand: *Buffer,
        indices: *Buffer,
        offset_dims: []const i64,
        collapsed_slice_dims: []const i64,
        operand_batching_dims: []const i64,
        start_indices_batching_dims: []const i64,
        start_index_map: []const i64,
        index_vector_dim: i64,
        slice_sizes: []const i64,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (operand.element_type.byteSize() == 0) return error.UnsupportedElementType;
        if (start_index_map.len == 0 or slice_sizes.len != operand.dims.len) return error.ShapeMismatch;
        if (index_vector_dim < 0) return error.ShapeMismatch;

        if (operand.backend_buffer != null and indices.backend_buffer != null) {
            if (supportedGatherAxisForRuntime(collapsed_slice_dims, start_index_map, slice_sizes)) |gather_axis| {
                if (operand.backend.gatherAxis(operand.backend_buffer.?, indices.backend_buffer.?, gather_axis, index_vector_dim, output_dims) catch null) |backend_buffer| {
                    return initBackendOnly(allocator, operand, operand.element_type, output_dims, denseByteSize(operand.element_type, output_dims), backend_buffer, shard_index);
                }
            }
            if (operand.backend.gather(
                operand.backend_buffer.?,
                indices.backend_buffer.?,
                start_index_map,
                collapsed_slice_dims,
                operand_batching_dims,
                start_indices_batching_dims,
                index_vector_dim,
                slice_sizes,
                offset_dims,
                output_dims,
            ) catch null) |backend_buffer| {
                return initBackendOnly(allocator, operand, operand.element_type, output_dims, denseByteSize(operand.element_type, output_dims), backend_buffer, shard_index);
            }
        }
        return error.UnsupportedRuntimeFeature;
    }

    pub fn initSort(
        allocator: std.mem.Allocator,
        src: *Buffer,
        dimension: i64,
        output_dims: []const i64,
        direction: CompareOp,
        shard_index: usize,
    ) !*Buffer {
        if (!std.mem.eql(i64, src.dims, output_dims)) return error.ShapeMismatch;
        if (dimension < 0 or dimension >= @as(i64, @intCast(src.dims.len))) return error.ShapeMismatch;
        const element_size = src.element_type.byteSize();
        if (element_size == 0) return error.UnsupportedElementType;
        const ascending = switch (direction) {
            .lt, .le => true,
            .gt, .ge => false,
            else => return error.UnsupportedElementType,
        };
        if (src.backend_buffer) |src_backend| {
            if (src.backend.sort(src_backend, dimension, output_dims) catch null) |sorted_backend| {
                if (ascending) {
                    return initBackendOnly(allocator, src, src.element_type, output_dims, src.byte_size, sorted_backend, shard_index);
                }
                const reverse_dimensions = [_]i64{dimension};
                if (src.backend.reverse(sorted_backend, &reverse_dimensions, output_dims) catch null) |backend_buffer| {
                    src.backend.destroyBuffer(sorted_backend);
                    return initBackendOnly(allocator, src, src.element_type, output_dims, src.byte_size, backend_buffer, shard_index);
                }
                src.backend.destroyBuffer(sorted_backend);
            }
        }
        return error.UnsupportedRuntimeFeature;
    }

    pub fn initU8Unary(
        allocator: std.mem.Allocator,
        op: ElementwiseUnaryOp,
        src: *Buffer,
        shard_index: usize,
    ) !*Buffer {
        if (src.element_type != .u8) return error.UnsupportedElementType;
        return initElementwiseUnary(allocator, op, src, shard_index);
    }

    pub fn deinit(self: *Buffer) void {
        self.releaseStorage();
        self.allocator.free(self.bytes);
        self.allocator.free(self.dims);
        self.allocator.destroy(self);
    }

    pub fn copyToHost(self: *Buffer, dst: []u8) !void {
        try self.ensureUsable();
        try self.ensureReady();
        if (dst.len < self.byte_size) return error.DestinationTooSmall;
        const backend_buffer = self.backend_buffer orelse return error.UnsupportedRuntimeFeature;
        self.backend.copyToHost(backend_buffer, dst) catch return error.BackendBufferCopyFailed;
        self.memory.stats.device_to_host_bytes += @intCast(self.byte_size);
    }

    pub fn hasBackendStorage(self: *const Buffer) bool {
        return self.backend_buffer != null;
    }

    pub fn markDeleted(self: *Buffer) void {
        self.releaseStorage();
        self.state = .deleted;
        self.deleted = true;
        self.ready_event.setFailed("buffer has been deleted");
    }

    pub fn markDonated(self: *Buffer) void {
        self.releaseStorage();
        self.state = .donated;
        self.deleted = true;
        self.ready_event.setFailed("buffer has been donated");
    }

    pub fn takeBackendStorageForDonationAlias(self: *Buffer) !backend_api.BufferHandle {
        try self.ensureUsable();
        const backend_buffer = self.backend_buffer orelse return error.UnsupportedRuntimeFeature;
        self.backend_buffer = null;
        if (self.accounted_bytes != 0) {
            self.memory.stats.release(self.accounted_bytes);
            self.accounted_bytes = 0;
        }
        return backend_buffer;
    }

    pub fn replaceBackendStorageFromHost(self: *Buffer, src: []const u8) !void {
        try self.ensureUsable();
        if (src.len != self.byte_size) return error.ShapeMismatch;
        const backend_buffer = try self.backend.bufferFromHost(self.device.local_hardware_id, self.element_type, self.dims, src) orelse
            return error.UnsupportedRuntimeFeature;
        errdefer self.backend.destroyBuffer(backend_buffer);
        try self.replaceBackendStorage(backend_buffer);
        self.memory.stats.host_to_device_bytes += @intCast(src.len);
    }

    pub fn replaceBackendStorage(self: *Buffer, backend_buffer: backend_api.BufferHandle) !void {
        try self.ensureUsable();
        self.releaseStorage();
        self.backend_buffer = backend_buffer;
        _ = self.accountDeviceBytes();
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

    fn accountDeviceBytes(self: *Buffer) *Buffer {
        self.accounted_bytes = self.byte_size;
        self.memory.stats.retain(self.accounted_bytes);
        return self;
    }

    fn releaseStorage(self: *Buffer) void {
        if (self.accounted_bytes != 0) {
            self.memory.stats.release(self.accounted_bytes);
            self.accounted_bytes = 0;
        }
        if (self.backend_buffer) |backend_buffer| {
            self.backend.destroyBuffer(backend_buffer);
            self.backend_buffer = null;
        }
    }

    pub fn initBackendHandle(
        allocator: std.mem.Allocator,
        backend_impl: backend_api.Backend,
        element_type: BufferType,
        dims_: []const i64,
        device: *Device,
        memory: *Memory,
        shard_index: usize,
        byte_size: usize,
        backend_buffer: backend_api.BufferHandle,
    ) !*Buffer {
        if (!memory.isAddressableBy(device)) return error.InvalidArgument;

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, dims_);
        errdefer allocator.free(dims);

        const bytes = try allocator.alloc(u8, 0);
        errdefer allocator.free(bytes);

        buffer.* = .{
            .allocator = allocator,
            .backend = backend_impl,
            .backend_kind = backend_impl.kind(),
            .element_type = element_type,
            .dims = dims,
            .device_id = device.id,
            .memory_id = memory.id,
            .device = device,
            .memory = memory,
            .shard_index = shard_index,
            .byte_size = byte_size,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
        };
        return buffer.accountDeviceBytes();
    }
};

fn denseByteSize(element_type: BufferType, dims: []const i64) usize {
    return core.denseByteSize(element_type, dims);
}

fn validPermutation(permutation: []const i64, rank: usize) bool {
    if (permutation.len != rank) return false;
    var seen_storage = [_]bool{false} ** 16;
    if (rank > seen_storage.len) return false;
    const seen = seen_storage[0..rank];
    for (permutation) |axis| {
        if (axis < 0 or axis >= @as(i64, @intCast(rank))) return false;
        const axis_index: usize = @intCast(axis);
        if (seen[axis_index]) return false;
        seen[axis_index] = true;
    }
    return true;
}

fn validBroadcastDimensions(broadcast_dimensions: []const i64, input_dims: []const i64, output_dims: []const i64) bool {
    if (broadcast_dimensions.len != input_dims.len) return false;
    var seen_storage = [_]bool{false} ** 64;
    if (output_dims.len > seen_storage.len) return false;
    const seen = seen_storage[0..output_dims.len];
    for (broadcast_dimensions, 0..) |axis, input_axis| {
        if (axis < 0 or axis >= @as(i64, @intCast(output_dims.len))) return false;
        const output_axis: usize = @intCast(axis);
        if (seen[output_axis]) return false;
        seen[output_axis] = true;
        if (input_dims[input_axis] != 1 and input_dims[input_axis] != output_dims[output_axis]) return false;
    }
    return true;
}

fn validSlice(start_indices: []const i64, limit_indices: []const i64, strides: []const i64, input_dims: []const i64, output_dims: []const i64) bool {
    const rank = input_dims.len;
    if (start_indices.len != rank or limit_indices.len != rank or strides.len != rank or output_dims.len != rank) return false;
    for (0..rank) |axis| {
        const input_dim = input_dims[axis];
        const start = start_indices[axis];
        const limit = limit_indices[axis];
        const stride = strides[axis];
        const output_dim = output_dims[axis];
        if (input_dim < 0 or output_dim < 0) return false;
        if (start < 0 or limit < start or limit > input_dim or stride <= 0) return false;
        const span = limit - start;
        const expected = if (span == 0) 0 else @divTrunc(span + stride - 1, stride);
        if (output_dim != expected) return false;
    }
    return true;
}

fn validConcatenate(lhs_dims: []const i64, rhs_dims: []const i64, dimension: i64, output_dims: []const i64) bool {
    const rank = lhs_dims.len;
    if (rank == 0 or rhs_dims.len != rank or output_dims.len != rank) return false;
    if (dimension < 0 or dimension >= @as(i64, @intCast(rank))) return false;
    const concat_axis: usize = @intCast(dimension);
    for (0..rank) |axis| {
        if (lhs_dims[axis] < 0 or rhs_dims[axis] < 0 or output_dims[axis] < 0) return false;
        const expected = if (axis == concat_axis) lhs_dims[axis] + rhs_dims[axis] else lhs_dims[axis];
        if (output_dims[axis] != expected) return false;
        if (axis != concat_axis and lhs_dims[axis] != rhs_dims[axis]) return false;
    }
    return true;
}

fn validDotGeneral(
    lhs_dims: []const i64,
    rhs_dims: []const i64,
    lhs_batch_dimensions: []const i64,
    rhs_batch_dimensions: []const i64,
    lhs_contracting_dimensions: []const i64,
    rhs_contracting_dimensions: []const i64,
    output_dims: []const i64,
) bool {
    if (lhs_contracting_dimensions.len != 1 or rhs_contracting_dimensions.len != 1) return false;
    if (lhs_batch_dimensions.len != rhs_batch_dimensions.len) return false;
    const lhs_contract: usize = if (lhs_contracting_dimensions[0] < 0) return false else @intCast(lhs_contracting_dimensions[0]);
    const rhs_contract: usize = if (rhs_contracting_dimensions[0] < 0) return false else @intCast(rhs_contracting_dimensions[0]);
    if (lhs_contract >= lhs_dims.len or rhs_contract >= rhs_dims.len) return false;
    if (lhs_dims[lhs_contract] != rhs_dims[rhs_contract]) return false;
    for (lhs_batch_dimensions, rhs_batch_dimensions) |lhs_axis_i64, rhs_axis_i64| {
        if (lhs_axis_i64 < 0 or rhs_axis_i64 < 0) return false;
        const lhs_axis: usize = @intCast(lhs_axis_i64);
        const rhs_axis: usize = @intCast(rhs_axis_i64);
        if (lhs_axis >= lhs_dims.len or rhs_axis >= rhs_dims.len or lhs_dims[lhs_axis] != rhs_dims[rhs_axis]) return false;
    }
    return denseByteSize(.f32, output_dims) != 0;
}

fn validReduce(input_dims: []const i64, dimensions: []const i64, output_dims: []const i64) bool {
    var reduced = [_]bool{false} ** 64;
    if (input_dims.len > reduced.len) return false;
    for (dimensions) |dim_i64| {
        if (dim_i64 < 0 or dim_i64 >= @as(i64, @intCast(input_dims.len))) return false;
        const dim: usize = @intCast(dim_i64);
        if (reduced[dim]) return false;
        reduced[dim] = true;
    }
    var expected_rank: usize = 0;
    for (0..input_dims.len) |axis| {
        if (!reduced[axis]) expected_rank += 1;
    }
    if (output_dims.len != expected_rank) return false;
    var out_axis: usize = 0;
    for (input_dims, 0..) |dim, axis| {
        if (!reduced[axis]) {
            if (output_dims[out_axis] != dim) return false;
            out_axis += 1;
        }
    }
    return true;
}

fn readU32LE(bytes: []const u8, index: usize) u32 {
    return std.mem.readInt(u32, bytes[index * 4 ..][0..4], .little);
}

fn backendStartHandles(allocator: std.mem.Allocator, start_buffers: []const *Buffer) ![]backend_api.BufferHandle {
    for (start_buffers) |start_buffer| {
        if (start_buffer.backend_buffer == null) return allocator.alloc(backend_api.BufferHandle, 0);
    }
    const handles = try allocator.alloc(backend_api.BufferHandle, start_buffers.len);
    errdefer allocator.free(handles);
    for (start_buffers, 0..) |start_buffer, index| {
        handles[index] = start_buffer.backend_buffer.?;
    }
    return handles;
}

fn supportedGatherAxisForRuntime(collapsed_slice_dims: []const i64, start_index_map: []const i64, slice_sizes: []const i64) ?i64 {
    if (collapsed_slice_dims.len != 1 or start_index_map.len != 1) return null;
    const axis = start_index_map[0];
    if (axis < 0 or collapsed_slice_dims[0] != axis) return null;
    if (slice_sizes.len == 0 or axis >= @as(i64, @intCast(slice_sizes.len)) or slice_sizes[@intCast(axis)] != 1) return null;
    return axis;
}

fn mlxMetalBackendForTest() backend_api.Backend {
    return @import("src/backend/registry").create(.metal_mlx);
}

fn initMlxMetalClientForTest() !*Client {
    return Client.init(std.testing.allocator, mlxMetalBackendForTest(), 1);
}

fn initHostCopyForTest(
    element_type: BufferType,
    dims: []const i64,
    device: *Device,
    memory: *Memory,
    shard_index: usize,
    src: []const u8,
) !*Buffer {
    return Buffer.initHostCopyForBackend(std.testing.allocator, mlxMetalBackendForTest(), element_type, dims, device, memory, shard_index, src);
}

fn expectBufferBytes(buffer: *Buffer, expected: []const u8) !void {
    const actual = try std.testing.allocator.alloc(u8, expected.len);
    defer std.testing.allocator.free(actual);
    try buffer.copyToHost(actual);
    try std.testing.expectEqualSlices(u8, expected, actual);
}

fn expectBufferF32(buffer: *Buffer, expected: []const f32) !void {
    const actual = try std.testing.allocator.alloc(u8, expected.len * @sizeOf(f32));
    defer std.testing.allocator.free(actual);
    try buffer.copyToHost(actual);
    const floats = std.mem.bytesAsSlice(f32, actual);
    for (expected, floats) |want, got| {
        try std.testing.expectApproxEqAbs(want, got, 0.0001);
    }
}

fn testShardingPlan(allocator: std.mem.Allocator, assignment: []const i32) !core.ShardingPlan {
    return .{
        .kind = .replicated,
        .mesh_name = try allocator.dupe(u8, ""),
        .device_assignment = try allocator.dupe(i32, assignment),
    };
}

fn constantU8ExecutablePlanForTest(allocator: std.mem.Allocator, assignment: []const i32, literal: []const u8) !core.ExecutablePlan {
    var values = try allocator.alloc(core.Value, 1);
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

    var output_shardings = try allocator.alloc(core.ShardingPlan, 1);
    errdefer allocator.free(output_shardings);
    output_shardings[0] = try testShardingPlan(allocator, assignment);
    errdefer {
        allocator.free(output_shardings[0].mesh_name);
        allocator.free(output_shardings[0].device_assignment);
    }

    const output_ids = try allocator.dupe(core.ValueId, &.{.{ .index = 0 }});
    errdefer allocator.free(output_ids);

    const literal_copy = try allocator.dupe(u8, literal);
    errdefer allocator.free(literal_copy);

    const instruction_outputs = try allocator.dupe(core.ValueId, &.{.{ .index = 0 }});
    errdefer allocator.free(instruction_outputs);

    const instructions = try allocator.dupe(core.PlanInstruction, &.{.{
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
        .parameter_shardings = try allocator.alloc(core.ShardingPlan, 0),
        .output_shardings = output_shardings,
        .output_ids = output_ids,
        .instructions = instructions,
    };
}

fn addU8ExecutablePlanForTest(allocator: std.mem.Allocator, assignment: []const i32, dims: []const i64) !core.ExecutablePlan {
    const values = try allocator.alloc(core.Value, 3);
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

    var parameter_shardings = try allocator.alloc(core.ShardingPlan, 2);
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

    var output_shardings = try allocator.alloc(core.ShardingPlan, 1);
    errdefer allocator.free(output_shardings);
    output_shardings[0] = try testShardingPlan(allocator, assignment);
    errdefer {
        allocator.free(output_shardings[0].mesh_name);
        allocator.free(output_shardings[0].device_assignment);
    }

    const output_ids = try allocator.dupe(core.ValueId, &.{.{ .index = 2 }});
    errdefer allocator.free(output_ids);

    const inputs = try allocator.dupe(core.ValueId, &.{ .{ .index = 0 }, .{ .index = 1 } });
    errdefer allocator.free(inputs);

    const outputs = try allocator.dupe(core.ValueId, &.{.{ .index = 2 }});
    errdefer allocator.free(outputs);

    const instructions = try allocator.dupe(core.PlanInstruction, &.{.{
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

test "backend output descriptor validation matches executable plan outputs" {
    const allocator = std.testing.allocator;
    const assignment = [_]i32{0};
    const dims = [_]i64{ 2, 2 };

    const values = try allocator.alloc(core.Value, 1);
    errdefer allocator.free(values);
    values[0] = .{
        .id = .{ .index = 0 },
        .role = .instruction_result,
        .descriptor = .{
            .element_type = .f32,
            .dims = try allocator.dupe(i64, &dims),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };

    var output_shardings = try allocator.alloc(core.ShardingPlan, 1);
    output_shardings[0] = try testShardingPlan(allocator, &assignment);

    var plan = core.ExecutablePlan{
        .allocator = allocator,
        .options = .{
            .num_replicas = 1,
            .num_partitions = 1,
            .device_assignment = try allocator.dupe(i32, &assignment),
        },
        .values = values,
        .parameter_shardings = try allocator.alloc(core.ShardingPlan, 0),
        .output_shardings = output_shardings,
        .output_ids = try allocator.dupe(core.ValueId, &.{.{ .index = 0 }}),
        .instructions = try allocator.alloc(core.PlanInstruction, 0),
    };
    defer plan.deinit();

    const handle: backend_api.BufferHandle = @ptrFromInt(0x1000);
    try std.testing.expect(backendOutputMatchesPlan(&plan, 0, .{
        .handle = handle,
        .element_type = .f32,
        .dims = &dims,
        .byte_size = 16,
    }));
    try std.testing.expect(!backendOutputMatchesPlan(&plan, 0, .{
        .handle = handle,
        .element_type = .u32,
        .dims = &dims,
        .byte_size = 16,
    }));
    try std.testing.expect(!backendOutputMatchesPlan(&plan, 0, .{
        .handle = handle,
        .element_type = .f32,
        .dims = &.{4},
        .byte_size = 16,
    }));
    try std.testing.expect(!backendOutputMatchesPlan(&plan, 0, .{
        .handle = handle,
        .element_type = .f32,
        .dims = &dims,
        .byte_size = 12,
    }));
    try std.testing.expect(!backendOutputMatchesPlan(&plan, 1, .{
        .handle = handle,
        .element_type = .f32,
        .dims = &dims,
        .byte_size = 16,
    }));
}

test "executable graph materializes per-device scheduled nodes" {
    const client = try initMlxMetalClientForTest();
    defer client.deinit();
    const allocator = std.testing.allocator;
    const assignment = [_]i32{0};
    const dims = [_]i64{4};

    const values = try allocator.alloc(core.Value, 3);
    errdefer allocator.free(values);
    for (values, 0..) |*value, i| {
        value.* = .{
            .id = .{ .index = @intCast(i) },
            .role = if (i < 2) .parameter else .instruction_result,
            .descriptor = .{
                .element_type = .u8,
                .dims = try allocator.dupe(i64, &dims),
                .device_id = 0,
                .memory_id = 0,
                .shard_index = 0,
            },
        };
    }

    var parameter_shardings = try allocator.alloc(core.ShardingPlan, 2);
    parameter_shardings[0] = try testShardingPlan(allocator, &assignment);
    parameter_shardings[1] = try testShardingPlan(allocator, &assignment);
    var output_shardings = try allocator.alloc(core.ShardingPlan, 1);
    output_shardings[0] = try testShardingPlan(allocator, &assignment);

    var plan = core.ExecutablePlan{
        .allocator = allocator,
        .options = .{
            .num_replicas = 1,
            .num_partitions = 1,
            .device_assignment = try allocator.dupe(i32, &assignment),
        },
        .values = values,
        .parameter_shardings = parameter_shardings,
        .output_shardings = output_shardings,
        .output_ids = try allocator.dupe(core.ValueId, &.{.{ .index = 2 }}),
        .instructions = try allocator.dupe(core.PlanInstruction, &.{.{
            .kind = .add,
            .inputs = try allocator.dupe(core.ValueId, &.{ .{ .index = 0 }, .{ .index = 1 } }),
            .outputs = try allocator.dupe(core.ValueId, &.{.{ .index = 2 }}),
        }}),
    };
    defer plan.deinit();

    var graph = try ExecutableGraph.init(allocator, client, &plan);
    defer graph.deinit();

    try std.testing.expectEqualSlices(i32, &.{0}, graph.device_ids);
    try std.testing.expectEqual(@as(usize, 1), graph.nodes.len);
    try std.testing.expectEqual(GraphNodeKind.compute, graph.nodes[0].kind);
    try std.testing.expectEqual(@as(usize, 0), graph.nodes[0].device_index);
    try std.testing.expectEqual(@as(i32, 0), graph.nodes[0].device_id);
    try std.testing.expectEqual(true, graph.lowering.backend_executable_ready);
    try std.testing.expectEqual(@as(usize, 1), graph.lowering.lowered_instruction_count);
    try std.testing.expectEqual(GraphNodeKind.constant, graphNodeKind(.constant));
    try std.testing.expectEqual(GraphNodeKind.custom_call, graphNodeKind(.custom_call));
    try std.testing.expectEqual(GraphNodeKind.structural, graphNodeKind(.optimization_barrier));
    try std.testing.expectEqual(GraphNodeKind.control_flow, graphNodeKind(.while_));
}

test "executable graph device-only lowering rejects unsupported backend executable" {
    const client = try initMlxMetalClientForTest();
    defer client.deinit();
    const allocator = std.testing.allocator;

    var plan = core.ExecutablePlan{
        .allocator = allocator,
        .options = .{
            .num_replicas = 1,
            .num_partitions = 1,
            .device_assignment = try allocator.dupe(i32, &.{0}),
        },
        .values = &.{},
        .parameter_shardings = try allocator.alloc(core.ShardingPlan, 0),
        .output_shardings = try allocator.alloc(core.ShardingPlan, 0),
        .output_ids = try allocator.alloc(core.ValueId, 0),
        .instructions = try allocator.dupe(core.PlanInstruction, &.{
            .{ .kind = .custom_call },
        }),
    };
    defer plan.deinit();

    try std.testing.expectError(
        error.UnsupportedRuntimeFeature,
        ExecutableGraph.initWithOptions(allocator, client, &plan, .{}),
    );
}

test "executable graph executes through runtime buffers" {
    const client = try initMlxMetalClientForTest();
    defer client.deinit();
    const allocator = std.testing.allocator;
    const assignment = [_]i32{0};
    const dims = [_]i64{4};

    const values = try allocator.alloc(core.Value, 3);
    errdefer allocator.free(values);
    for (values, 0..) |*value, i| {
        value.* = .{
            .id = .{ .index = @intCast(i) },
            .role = if (i < 2) .parameter else .instruction_result,
            .descriptor = .{
                .element_type = .u8,
                .dims = try allocator.dupe(i64, &dims),
                .device_id = 0,
                .memory_id = 0,
                .shard_index = 0,
            },
        };
    }

    var parameter_shardings = try allocator.alloc(core.ShardingPlan, 2);
    parameter_shardings[0] = try testShardingPlan(allocator, &assignment);
    parameter_shardings[1] = try testShardingPlan(allocator, &assignment);
    var output_shardings = try allocator.alloc(core.ShardingPlan, 1);
    output_shardings[0] = try testShardingPlan(allocator, &assignment);

    var plan = core.ExecutablePlan{
        .allocator = allocator,
        .options = .{
            .num_replicas = 1,
            .num_partitions = 1,
            .device_assignment = try allocator.dupe(i32, &assignment),
        },
        .values = values,
        .parameter_shardings = parameter_shardings,
        .output_shardings = output_shardings,
        .output_ids = try allocator.dupe(core.ValueId, &.{.{ .index = 2 }}),
        .instructions = try allocator.dupe(core.PlanInstruction, &.{.{
            .kind = .add,
            .inputs = try allocator.dupe(core.ValueId, &.{ .{ .index = 0 }, .{ .index = 1 } }),
            .outputs = try allocator.dupe(core.ValueId, &.{.{ .index = 2 }}),
        }}),
    };
    defer plan.deinit();

    var graph = try ExecutableGraph.init(allocator, client, &plan);
    defer graph.deinit();
    try std.testing.expect(graph.backend_executable != null);
    try std.testing.expectEqual(true, graph.lowering.backend_executable_ready);
    try std.testing.expectEqual(@as(usize, 1), graph.lowering.lowered_instruction_count);

    const lhs_data = [_]u8{ 1, 2, 3, 4 };
    const rhs_data = [_]u8{ 10, 20, 30, 40 };
    const lhs = try Buffer.initHostCopyForBackend(allocator, client.backend, .u8, &dims, &client.devices[0], &client.memories[0], 0, &lhs_data);
    defer lhs.deinit();
    const rhs = try Buffer.initHostCopyForBackend(allocator, client.backend, .u8, &dims, &client.devices[0], &client.memories[0], 0, &rhs_data);
    defer rhs.deinit();

    client.memories[0].stats.capacity_bytes = client.memories[0].stats.bytes_in_use + 4;
    const h2d_before_execute = client.memories[0].stats.host_to_device_bytes;
    const d2h_before_execute = client.memories[0].stats.device_to_host_bytes;
    const result = try graph.executeDevice(allocator, client, &plan, 0, &.{ lhs, rhs });
    const outputs = result.outputs;
    defer allocator.free(outputs);
    defer {
        for (outputs) |output| output.deinit();
    }

    try std.testing.expectEqual(@as(usize, 1), outputs.len);
    try std.testing.expectEqual(@as(usize, 0), outputs[0].shard_index);
    try std.testing.expect(outputs[0].backend_buffer != null);
    try std.testing.expectEqual(@as(usize, 0), outputs[0].bytes.len);
    try std.testing.expectEqual(@as(u64, 4), graph.last_execute_cache_trim.requested_bytes);
    try std.testing.expectEqual(@as(u64, 0), graph.last_execute_cache_trim.freed_bytes);
    try std.testing.expect(!graph.last_execute_cache_trim.still_over_capacity);
    try std.testing.expectEqual(backend_api.ExecutionCompletionKind.completed, graph.last_backend_completion.kind);
    try std.testing.expect(result.completion_event.isReady());
    try std.testing.expectEqual(h2d_before_execute, client.memories[0].stats.host_to_device_bytes);
    try std.testing.expectEqual(d2h_before_execute, client.memories[0].stats.device_to_host_bytes);
    try expectBufferBytes(outputs[0], &.{ 11, 22, 33, 44 });
    try std.testing.expectEqual(d2h_before_execute + outputs[0].byte_size, client.memories[0].stats.device_to_host_bytes);

    const wrong_shard = try Buffer.initHostCopyForBackend(allocator, client.backend, .u8, &dims, &client.devices[0], &client.memories[0], 7, &lhs_data);
    defer wrong_shard.deinit();
    try std.testing.expectError(error.InvalidArgument, graph.executeDevice(allocator, client, &plan, 0, &.{ wrong_shard, rhs }));

    lhs.ready_event = Event.pending();
    try std.testing.expectError(error.BufferNotReady, graph.executeDevice(allocator, client, &plan, 0, &.{ lhs, rhs }));
    lhs.ready_event.setReady();
    rhs.ready_event = Event.failed("argument upload failed");
    try std.testing.expectError(error.BufferReadinessFailed, graph.executeDevice(allocator, client, &plan, 0, &.{ lhs, rhs }));
    rhs.ready_event = Event.ready();

    lhs.device_id = 1234;
    try std.testing.expectError(error.InvalidArgument, graph.executeDevice(allocator, client, &plan, 0, &.{ lhs, rhs }));
}

test "executable graph transfers donated parameter alias outputs without copying" {
    const client = try initMlxMetalClientForTest();
    defer client.deinit();
    const allocator = std.testing.allocator;
    const assignment = [_]i32{0};
    const dims = [_]i64{4};

    const values = try allocator.alloc(core.Value, 1);
    errdefer allocator.free(values);
    values[0] = .{
        .id = .{ .index = 0 },
        .role = .parameter,
        .descriptor = .{
            .element_type = .u8,
            .dims = try allocator.dupe(i64, &dims),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };

    var parameter_shardings = try allocator.alloc(core.ShardingPlan, 1);
    parameter_shardings[0] = try testShardingPlan(allocator, &assignment);
    var output_shardings = try allocator.alloc(core.ShardingPlan, 1);
    output_shardings[0] = try testShardingPlan(allocator, &assignment);

    var plan = core.ExecutablePlan{
        .allocator = allocator,
        .options = .{
            .num_replicas = 1,
            .num_partitions = 1,
            .device_assignment = try allocator.dupe(i32, &assignment),
        },
        .values = values,
        .parameter_shardings = parameter_shardings,
        .output_shardings = output_shardings,
        .output_ids = try allocator.dupe(core.ValueId, &.{.{ .index = 0 }}),
        .donated_parameter_indices = try allocator.dupe(u32, &.{0}),
        .instructions = try allocator.alloc(core.PlanInstruction, 0),
    };
    defer plan.deinit();

    var graph = try ExecutableGraph.init(allocator, client, &plan);
    defer graph.deinit();

    const data = [_]u8{ 9, 8, 7, 6 };
    const input = try Buffer.initHostCopyForBackend(allocator, client.backend, .u8, &dims, &client.devices[0], &client.memories[0], 0, &data);
    defer input.deinit();
    const before_execute_bytes = client.memories[0].stats.bytes_in_use;
    const before_execute_handle = input.backend_buffer;

    const result = try graph.executeDevice(allocator, client, &plan, 0, &.{input});
    defer allocator.free(result.outputs);
    defer {
        for (result.outputs) |output| output.deinit();
    }

    try std.testing.expectEqual(@as(usize, 1), result.outputs.len);
    try std.testing.expectEqual(before_execute_handle, result.outputs[0].backend_buffer);
    try std.testing.expectEqual(@as(?backend_api.BufferHandle, null), input.backend_buffer);
    input.markDonated();
    try std.testing.expectEqual(BufferState.donated, input.state);
    try std.testing.expect(result.outputs[0].hasBackendStorage());
    try expectBufferBytes(result.outputs[0], &data);
    try std.testing.expectEqual(before_execute_bytes, client.memories[0].stats.bytes_in_use);

    const stats = graph.backendExecutableStats().?;
    try std.testing.expectEqual(@as(usize, 1), stats.donation_alias_output_count);
    try std.testing.expectEqual(@as(usize, data.len), stats.donation_alias_output_bytes);
}

test "executable graph reuses cached backend executable handles" {
    const client = try initMlxMetalClientForTest();
    defer client.deinit();
    const allocator = std.testing.allocator;
    const assignment = [_]i32{0};
    const dims = [_]i64{4};

    const values = try allocator.alloc(core.Value, 3);
    errdefer allocator.free(values);
    for (values, 0..) |*value, i| {
        value.* = .{
            .id = .{ .index = @intCast(i) },
            .role = if (i < 2) .parameter else .instruction_result,
            .descriptor = .{
                .element_type = .u8,
                .dims = try allocator.dupe(i64, &dims),
                .device_id = 0,
                .memory_id = 0,
                .shard_index = 0,
            },
        };
    }

    var parameter_shardings = try allocator.alloc(core.ShardingPlan, 2);
    parameter_shardings[0] = try testShardingPlan(allocator, &assignment);
    parameter_shardings[1] = try testShardingPlan(allocator, &assignment);
    var output_shardings = try allocator.alloc(core.ShardingPlan, 1);
    output_shardings[0] = try testShardingPlan(allocator, &assignment);

    var plan = core.ExecutablePlan{
        .allocator = allocator,
        .options = .{
            .num_replicas = 1,
            .num_partitions = 1,
            .device_assignment = try allocator.dupe(i32, &assignment),
        },
        .values = values,
        .parameter_shardings = parameter_shardings,
        .output_shardings = output_shardings,
        .output_ids = try allocator.dupe(core.ValueId, &.{.{ .index = 2 }}),
        .instructions = try allocator.dupe(core.PlanInstruction, &.{.{
            .kind = .add,
            .inputs = try allocator.dupe(core.ValueId, &.{ .{ .index = 0 }, .{ .index = 1 } }),
            .outputs = try allocator.dupe(core.ValueId, &.{.{ .index = 2 }}),
        }}),
    };
    defer plan.deinit();

    const fingerprint = "cached-add";
    try std.testing.expect(!try client.recordExecutableCompile(fingerprint));
    var first = try ExecutableGraph.initWithOptions(allocator, client, &plan, .{
        .cache_fingerprint = fingerprint,
    });
    try std.testing.expect(first.backend_executable != null);
    try std.testing.expect(!first.lowering.backend_executable_cache_reused);

    const entry = client.executable_cache.get(fingerprint) orelse return error.TestUnexpectedResult;
    try std.testing.expect(entry.backend_executable != null);
    try std.testing.expectEqual(@as(usize, 1), entry.ref_count);

    try std.testing.expect(try client.recordExecutableCompile(fingerprint));
    var second = try ExecutableGraph.initWithOptions(allocator, client, &plan, .{
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
    const client = try initMlxMetalClientForTest();
    defer client.deinit();
    const allocator = std.testing.allocator;
    const assignment = [_]i32{0};
    const dims = [_]i64{4};
    const literal = [_]u8{ 1, 2, 3, 4 };

    const values = try allocator.alloc(core.Value, 1);
    errdefer allocator.free(values);
    values[0] = .{
        .id = .{ .index = 0 },
        .role = .instruction_result,
        .descriptor = .{
            .element_type = .u8,
            .dims = try allocator.dupe(i64, &dims),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };

    var output_shardings = try allocator.alloc(core.ShardingPlan, 1);
    output_shardings[0] = try testShardingPlan(allocator, &assignment);

    var plan = core.ExecutablePlan{
        .allocator = allocator,
        .options = .{
            .num_replicas = 1,
            .num_partitions = 1,
            .device_assignment = try allocator.dupe(i32, &assignment),
        },
        .values = values,
        .parameter_shardings = try allocator.alloc(core.ShardingPlan, 0),
        .output_shardings = output_shardings,
        .output_ids = try allocator.dupe(core.ValueId, &.{.{ .index = 0 }}),
        .instructions = try allocator.dupe(core.PlanInstruction, &.{.{
            .kind = .constant,
            .outputs = try allocator.dupe(core.ValueId, &.{.{ .index = 0 }}),
            .literal = try allocator.dupe(u8, &literal),
        }}),
    };
    defer plan.deinit();

    client.setExecutableCacheMaxResidentBytes(0);

    const fingerprint = "evict-idle-constant";
    try std.testing.expect(!try client.recordExecutableCompile(fingerprint));
    var first = try ExecutableGraph.initWithOptions(allocator, client, &plan, .{
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
    var second = try ExecutableGraph.initWithOptions(allocator, client, &plan, .{
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
    const client = try initMlxMetalClientForTest();
    defer client.deinit();
    const allocator = std.testing.allocator;
    const assignment = [_]i32{0};

    var small_plan = try constantU8ExecutablePlanForTest(allocator, &assignment, &.{ 1, 2, 3, 4 });
    defer small_plan.deinit();
    var large_plan = try constantU8ExecutablePlanForTest(allocator, &assignment, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer large_plan.deinit();

    try std.testing.expect(!try client.recordExecutableCompile("small-constant"));
    var small_graph = try ExecutableGraph.initWithOptions(allocator, client, &small_plan, .{
        .cache_fingerprint = "small-constant",
    });
    const small_entry = client.executable_cache.get("small-constant") orelse return error.TestUnexpectedResult;
    const small_resident_bytes = small_entry.resident_bytes;
    try std.testing.expect(small_resident_bytes >= 4);
    small_graph.deinit();
    try std.testing.expectEqual(@as(usize, 0), small_entry.ref_count);
    try std.testing.expect(small_entry.backend_executable != null);

    try std.testing.expect(!try client.recordExecutableCompile("large-constant"));
    var large_graph = try ExecutableGraph.initWithOptions(allocator, client, &large_plan, .{
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
    const client = try initMlxMetalClientForTest();
    defer client.deinit();
    const allocator = std.testing.allocator;
    const assignment = [_]i32{0};

    var small_plan = try constantU8ExecutablePlanForTest(allocator, &assignment, &.{ 1, 2, 3, 4 });
    defer small_plan.deinit();
    var large_plan = try constantU8ExecutablePlanForTest(allocator, &assignment, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer large_plan.deinit();

    try std.testing.expect(!try client.recordExecutableCompile("pressure-small-constant"));
    var small_graph = try ExecutableGraph.initWithOptions(allocator, client, &small_plan, .{
        .cache_fingerprint = "pressure-small-constant",
    });
    const small_entry = client.executable_cache.get("pressure-small-constant") orelse return error.TestUnexpectedResult;
    const small_resident_bytes = small_entry.resident_bytes;
    small_graph.deinit();

    try std.testing.expect(!try client.recordExecutableCompile("pressure-large-constant"));
    var large_graph = try ExecutableGraph.initWithOptions(allocator, client, &large_plan, .{
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
    const client = try initMlxMetalClientForTest();
    defer client.deinit();
    const allocator = std.testing.allocator;
    const assignment = [_]i32{0};

    var idle_plan = try constantU8ExecutablePlanForTest(allocator, &assignment, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer idle_plan.deinit();
    var new_plan = try constantU8ExecutablePlanForTest(allocator, &assignment, &.{ 9, 10, 11, 12 });
    defer new_plan.deinit();

    try std.testing.expect(!try client.recordExecutableCompile("compile-pressure-idle-constant"));
    var idle_graph = try ExecutableGraph.initWithOptions(allocator, client, &idle_plan, .{
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
    var new_graph = try ExecutableGraph.initWithOptions(allocator, client, &new_plan, .{
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

test "executable graph evicts idle executable cache before output allocation" {
    const client = try initMlxMetalClientForTest();
    defer client.deinit();
    const allocator = std.testing.allocator;
    const assignment = [_]i32{0};
    const dims = [_]i64{4};

    var idle_plan = try constantU8ExecutablePlanForTest(allocator, &assignment, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer idle_plan.deinit();

    try std.testing.expect(!try client.recordExecutableCompile("execute-pressure-idle-constant"));
    var idle_graph = try ExecutableGraph.initWithOptions(allocator, client, &idle_plan, .{
        .cache_fingerprint = "execute-pressure-idle-constant",
    });
    const idle_entry = client.executable_cache.get("execute-pressure-idle-constant") orelse return error.TestUnexpectedResult;
    const idle_resident_bytes = idle_entry.resident_bytes;
    try std.testing.expect(idle_resident_bytes >= 8);
    idle_graph.deinit();
    try std.testing.expectEqual(@as(usize, 0), idle_entry.ref_count);
    try std.testing.expect(idle_entry.backend_executable != null);

    var add_plan = try addU8ExecutablePlanForTest(allocator, &assignment, &dims);
    defer add_plan.deinit();
    var add_graph = try ExecutableGraph.init(allocator, client, &add_plan);
    defer add_graph.deinit();

    const lhs_data = [_]u8{ 1, 2, 3, 4 };
    const rhs_data = [_]u8{ 10, 20, 30, 40 };
    const lhs = try Buffer.initHostCopyForBackend(allocator, client.backend, .u8, &dims, &client.devices[0], &client.memories[0], 0, &lhs_data);
    defer lhs.deinit();
    const rhs = try Buffer.initHostCopyForBackend(allocator, client.backend, .u8, &dims, &client.devices[0], &client.memories[0], 0, &rhs_data);
    defer rhs.deinit();

    const output_bytes = try planOutputBytes(&add_plan);
    client.memories[0].stats.capacity_bytes = client.memories[0].stats.bytes_in_use + output_bytes;

    const result = try add_graph.executeDevice(allocator, client, &add_plan, 0, &.{ lhs, rhs });
    const outputs = result.outputs;
    defer allocator.free(outputs);
    defer {
        for (outputs) |output| output.deinit();
    }

    try std.testing.expectEqual(@as(usize, 1), outputs.len);
    try std.testing.expect(result.completion_event.isReady());
    try expectBufferBytes(outputs[0], &.{ 11, 22, 33, 44 });
    try std.testing.expectEqual(@as(u64, @intCast(output_bytes)), add_graph.last_execute_cache_trim.requested_bytes);
    try std.testing.expectEqual(@as(u64, 0), add_graph.last_execute_cache_trim.target_resident_bytes);
    try std.testing.expectEqual(idle_resident_bytes, add_graph.last_execute_cache_trim.freed_bytes);
    try std.testing.expectEqual(@as(u64, 1), add_graph.last_execute_cache_trim.evicted_entries);
    try std.testing.expectEqual(@as(u64, 0), add_graph.last_execute_cache_trim.remaining_resident_bytes);
    try std.testing.expect(!add_graph.last_execute_cache_trim.still_over_capacity);
    try std.testing.expect(idle_entry.backend_executable == null);
    try std.testing.expectEqual(@as(u64, 1), client.executable_cache.stats.pressure_trim_requests);
    try std.testing.expectEqual(idle_resident_bytes, client.executable_cache.stats.pressure_trimmed_bytes);
    try std.testing.expectEqual(@as(u64, 0), client.executable_cache.stats.pressure_trim_failures);
}

test "executable cache preserves more expensive equal-size resident executable" {
    const client = try initMlxMetalClientForTest();
    defer client.deinit();
    const allocator = std.testing.allocator;
    const assignment = [_]i32{0};

    var cheap_plan = try constantU8ExecutablePlanForTest(allocator, &assignment, &.{ 1, 2, 3, 4 });
    defer cheap_plan.deinit();
    var expensive_plan = try constantU8ExecutablePlanForTest(allocator, &assignment, &.{ 5, 6, 7, 8 });
    defer expensive_plan.deinit();

    try std.testing.expect(!try client.recordExecutableCompile("cheap-constant"));
    var cheap_graph = try ExecutableGraph.initWithOptions(allocator, client, &cheap_plan, .{
        .cache_fingerprint = "cheap-constant",
    });
    const cheap_entry = client.executable_cache.get("cheap-constant") orelse return error.TestUnexpectedResult;
    const resident_bytes = cheap_entry.resident_bytes;
    cheap_graph.deinit();
    cheap_entry.compile_latency_us = 10;

    try std.testing.expect(!try client.recordExecutableCompile("expensive-constant"));
    var expensive_graph = try ExecutableGraph.initWithOptions(allocator, client, &expensive_plan, .{
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

test "buffer keeps shard/device/memory ownership metadata" {
    const client = try initMlxMetalClientForTest();
    defer client.deinit();

    const dims = [_]i64{ 2, 2 };
    const data = [_]u8{ 1, 2, 3, 4 };
    const buffer = try initHostCopyForTest(.u8, &dims, &client.devices[0], &client.memories[0], 0, &data);
    defer buffer.deinit();

    try std.testing.expectEqual(@as(i32, 0), buffer.device_id);
    try std.testing.expectEqual(@as(i32, 0), buffer.memory_id);
    try std.testing.expectEqual(@as(i32, 0), buffer.device.id);
    try std.testing.expectEqual(@as(i32, 0), buffer.memory.id);
    try std.testing.expectEqual(@as(usize, 0), buffer.shard_index);
    try expectBufferBytes(buffer, &data);

    const rhs_data = [_]u8{ 10, 20, 30, 40 };
    const rhs = try initHostCopyForTest(.u8, &dims, &client.devices[0], &client.memories[0], 0, &rhs_data);
    defer rhs.deinit();
    const sum = try Buffer.initU8Add(std.testing.allocator, buffer, rhs, 0);
    defer sum.deinit();
    try expectBufferBytes(sum, &.{ 11, 22, 33, 44 });

    const difference = try Buffer.initU8Binary(std.testing.allocator, .subtract, rhs, buffer, 0);
    defer difference.deinit();
    try expectBufferBytes(difference, &.{ 9, 18, 27, 36 });

    const product = try Buffer.initU8Binary(std.testing.allocator, .multiply, buffer, rhs, 0);
    defer product.deinit();
    try expectBufferBytes(product, &.{ 10, 40, 90, 160 });

    const quotient = try Buffer.initU8Binary(std.testing.allocator, .divide, rhs, buffer, 0);
    defer quotient.deinit();
    try expectBufferBytes(quotient, &.{ 10, 10, 10, 10 });

    const negated = try Buffer.initU8Unary(std.testing.allocator, .negate, buffer, 0);
    defer negated.deinit();
    try expectBufferBytes(negated, &.{ 255, 254, 253, 252 });
}

test "buffer constructors reject memory not addressable by device" {
    const client = try initMlxMetalClientForTest();
    defer client.deinit();

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
    try std.testing.expect(!invalid_memory.isAddressableBy(&client.devices[0]));
    try std.testing.expectError(
        error.InvalidArgument,
        Buffer.initHostCopyForBackend(std.testing.allocator, client.backend, .u8, &dims, &client.devices[0], &invalid_memory, 0, &data),
    );
    try std.testing.expectError(
        error.InvalidArgument,
        Buffer.initPartitionId(std.testing.allocator, client.backend, .u32, &.{}, &client.devices[0], &invalid_memory, 0, 0),
    );

    if (try client.backend.bufferFromHost(client.devices[0].local_hardware_id, .u8, &dims, &data)) |backend_buffer| {
        try std.testing.expectError(
            error.InvalidArgument,
            Buffer.initBackendHandle(std.testing.allocator, client.backend, .u8, &dims, &client.devices[0], &invalid_memory, 0, data.len, backend_buffer),
        );
        client.backend.destroyBuffer(backend_buffer);
    }
}

test "buffer lifecycle rejects deleted and donated buffers" {
    const client = try initMlxMetalClientForTest();
    defer client.deinit();

    const dims = [_]i64{4};
    const data = [_]u8{ 1, 2, 3, 4 };
    const before_deleted = client.memories[0].stats.bytes_in_use;
    const deleted = try initHostCopyForTest(.u8, &dims, &client.devices[0], &client.memories[0], 0, &data);
    defer deleted.deinit();
    try std.testing.expectEqual(before_deleted + data.len, client.memories[0].stats.bytes_in_use);
    try std.testing.expect(deleted.hasBackendStorage());
    deleted.markDeleted();
    try std.testing.expectEqual(BufferState.deleted, deleted.state);
    try std.testing.expect(deleted.deleted);
    try std.testing.expect(!deleted.hasBackendStorage());
    try std.testing.expectEqual(before_deleted, client.memories[0].stats.bytes_in_use);
    try std.testing.expectError(error.BufferDeleted, deleted.ensureUsable());

    const before_donated = client.memories[0].stats.bytes_in_use;
    const donated = try initHostCopyForTest(.u8, &dims, &client.devices[0], &client.memories[0], 0, &data);
    defer donated.deinit();
    try std.testing.expectEqual(before_donated + data.len, client.memories[0].stats.bytes_in_use);
    try std.testing.expect(donated.hasBackendStorage());
    donated.markDonated();
    try std.testing.expectEqual(BufferState.donated, donated.state);
    try std.testing.expect(donated.deleted);
    try std.testing.expect(!donated.hasBackendStorage());
    try std.testing.expectEqual(before_donated, client.memories[0].stats.bytes_in_use);
    try std.testing.expectError(error.BufferDonated, donated.ensureUsable());
}

test "buffer copies respect readiness events" {
    const client = try initMlxMetalClientForTest();
    defer client.deinit();

    const dims = [_]i64{4};
    const data = [_]u8{ 1, 2, 3, 4 };
    const buffer = try initHostCopyForTest(.u8, &dims, &client.devices[0], &client.memories[0], 0, &data);
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

const EventCallbackTestState = struct {
    count: usize = 0,
    ready_count: usize = 0,
    failed_count: usize = 0,
    saw_expected_message: bool = false,
};

fn recordEventCallback(message: ?[]const u8, user_arg: ?*anyopaque) void {
    const state: *EventCallbackTestState = @ptrCast(@alignCast(user_arg.?));
    state.count += 1;
    if (message) |msg| {
        state.failed_count += 1;
        if (std.mem.eql(u8, msg, "boom") or std.mem.eql(u8, msg, "event destroyed before completion")) {
            state.saw_expected_message = true;
        }
    } else {
        state.ready_count += 1;
    }
}

test "event callbacks run on ready and failed transitions" {
    var ready_event = Event.pending();
    var ready_state = EventCallbackTestState{};
    try ready_event.onReady(recordEventCallback, &ready_state);
    try std.testing.expect(!ready_event.isReady());
    ready_event.setReady();
    try std.testing.expect(ready_event.isReady());
    try ready_event.awaitReady();
    try std.testing.expectEqual(@as(usize, 1), ready_state.count);
    try std.testing.expectEqual(@as(usize, 1), ready_state.ready_count);

    var failed_event = Event.pending();
    var failed_state = EventCallbackTestState{};
    try failed_event.onReady(recordEventCallback, &failed_state);
    failed_event.setFailed("boom");
    try std.testing.expect(failed_event.isReady());
    try std.testing.expectError(error.EventFailed, failed_event.awaitReady());
    try std.testing.expectEqual(@as(usize, 1), failed_state.count);
    try std.testing.expectEqual(@as(usize, 1), failed_state.failed_count);
    try std.testing.expect(failed_state.saw_expected_message);
}

test "event dependency chains readiness and failures" {
    var source_ready = Event.pending();
    var dependent_ready = Event.pending();
    try source_ready.chainTo(&dependent_ready);
    try std.testing.expect(!dependent_ready.isReady());
    source_ready.setReady();
    try std.testing.expect(dependent_ready.isReady());
    try dependent_ready.awaitReady();

    var source_failed = Event.pending();
    var dependent_failed = Event.pending();
    try source_failed.chainTo(&dependent_failed);
    source_failed.setFailed("boom");
    try std.testing.expect(dependent_failed.isReady());
    try std.testing.expectError(error.EventFailed, dependent_failed.awaitReady());
    try std.testing.expectEqualStrings("boom", dependent_failed.message);
}

test "backend pending completion without event handle fails closed" {
    const event = runtimeEventFromBackendCompletion(mlxMetalBackendForTest(), .{ .kind = .pending });
    try std.testing.expect(event.isReady());
    try std.testing.expectError(error.EventFailed, event.awaitReady());
}

test "event callbacks run immediately for completed events and reject overflow" {
    var ready_event = Event.ready();
    var ready_state = EventCallbackTestState{};
    try ready_event.onReady(recordEventCallback, &ready_state);
    try std.testing.expectEqual(@as(usize, 1), ready_state.count);
    try std.testing.expectEqual(@as(usize, 1), ready_state.ready_count);

    var failed_event = Event.failed("boom");
    var failed_state = EventCallbackTestState{};
    try failed_event.onReady(recordEventCallback, &failed_state);
    try std.testing.expectEqual(@as(usize, 1), failed_state.count);
    try std.testing.expectEqual(@as(usize, 1), failed_state.failed_count);
    try std.testing.expect(failed_state.saw_expected_message);

    var pending_event = Event.pending();
    var pending_state = EventCallbackTestState{};
    for (0..MAX_EVENT_CALLBACKS) |_| {
        try pending_event.onReady(recordEventCallback, &pending_state);
    }
    try std.testing.expectError(error.TooManyEventCallbacks, pending_event.onReady(recordEventCallback, &pending_state));
    pending_event.deinit();
    try std.testing.expectEqual(@as(usize, MAX_EVENT_CALLBACKS), pending_state.count);
    try std.testing.expectEqual(@as(usize, MAX_EVENT_CALLBACKS), pending_state.failed_count);
    try std.testing.expect(pending_state.saw_expected_message);
}

test "client records executable cache hits and memory byte accounting" {
    const client = try initMlxMetalClientForTest();
    defer client.deinit();

    try std.testing.expect(!try client.recordExecutableCompile("same-program"));
    try std.testing.expect(try client.recordExecutableCompile("same-program"));
    try std.testing.expectEqual(@as(u64, 1), client.executable_cache.stats.misses);
    try std.testing.expectEqual(@as(u64, 1), client.executable_cache.stats.hits);

    const dims = [_]i64{4};
    const data = [_]u8{ 1, 2, 3, 4 };
    const before = client.memories[0].stats.bytes_in_use;
    {
        const buffer = try initHostCopyForTest(.u8, &dims, &client.devices[0], &client.memories[0], 0, &data);
        defer buffer.deinit();
        try std.testing.expectEqual(before + data.len, client.memories[0].stats.bytes_in_use);
        try std.testing.expect(client.memories[0].stats.peak_bytes_in_use >= client.memories[0].stats.bytes_in_use);
        try std.testing.expect(client.memories[0].stats.host_to_device_bytes >= data.len);
    }
    try std.testing.expectEqual(before, client.memories[0].stats.bytes_in_use);
}

test "buffer elementwise arithmetic supports f32 execution" {
    const client = try initMlxMetalClientForTest();
    defer client.deinit();

    const dims = [_]i64{2};
    const lhs_values = [_]f32{ 1.5, -2.0 };
    const rhs_values = [_]f32{ 2.25, 4.0 };
    const lhs = try initHostCopyForTest(.f32, &dims, &client.devices[0], &client.memories[0], 0, std.mem.asBytes(&lhs_values));
    defer lhs.deinit();
    const rhs = try initHostCopyForTest(.f32, &dims, &client.devices[0], &client.memories[0], 0, std.mem.asBytes(&rhs_values));
    defer rhs.deinit();

    const sum = try Buffer.initElementwiseBinary(std.testing.allocator, .add, lhs, rhs, 0);
    defer sum.deinit();
    try expectBufferF32(sum, &.{ 3.75, 2.0 });

    const quotient = try Buffer.initElementwiseBinary(std.testing.allocator, .divide, rhs, lhs, 0);
    defer quotient.deinit();
    try expectBufferF32(quotient, &.{ 1.5, -2.0 });

    const negated = try Buffer.initElementwiseUnary(std.testing.allocator, .negate, lhs, 0);
    defer negated.deinit();
    try expectBufferF32(negated, &.{ -1.5, 2.0 });
}

test "buffer convert uses backend astype for resident buffers" {
    const client = try initMlxMetalClientForTest();
    defer client.deinit();

    const dims = [_]i64{2};
    const input = [_]u8{ 1, 255 };
    const source = try initHostCopyForTest(.u8, &dims, &client.devices[0], &client.memories[0], 0, &input);
    defer source.deinit();

    const converted = try Buffer.initConvert(std.testing.allocator, source, .f32, &dims, 0);
    defer converted.deinit();
    try std.testing.expect(converted.backend_buffer != null);
    try std.testing.expectEqual(@as(usize, 0), converted.bytes.len);
    try expectBufferF32(converted, &.{ 1.0, 255.0 });
}

test "buffer iota uses resident mlx backend path" {
    const client = try initMlxMetalClientForTest();
    defer client.deinit();

    const dims = [_]i64{ 2, 3 };
    const buffer = try Buffer.initIota(
        std.testing.allocator,
        client.backend,
        .f32,
        &dims,
        &client.devices[0],
        &client.memories[0],
        1,
        0,
    );
    defer buffer.deinit();
    try std.testing.expect(buffer.backend_buffer != null);
    try std.testing.expectEqual(@as(usize, 0), buffer.bytes.len);
    try expectBufferF32(buffer, &.{ 0.0, 1.0, 2.0, 0.0, 1.0, 2.0 });
}

test "buffer movement ops use resident mlx backend paths" {
    const client = try initMlxMetalClientForTest();
    defer client.deinit();

    const dims = [_]i64{ 3, 4 };
    const values = [_]f32{
        1.0, 2.0,  3.0,  4.0,
        5.0, 6.0,  7.0,  8.0,
        9.0, 10.0, 11.0, 12.0,
    };
    const source = try initHostCopyForTest(.f32, &dims, &client.devices[0], &client.memories[0], 0, std.mem.asBytes(&values));
    defer source.deinit();

    const start0: i32 = 1;
    const start1: i32 = 1;
    const start0_buffer = try initHostCopyForTest(.s32, &.{}, &client.devices[0], &client.memories[0], 0, std.mem.asBytes(&start0));
    defer start0_buffer.deinit();
    const start1_buffer = try initHostCopyForTest(.s32, &.{}, &client.devices[0], &client.memories[0], 0, std.mem.asBytes(&start1));
    defer start1_buffer.deinit();

    const slice_dims = [_]i64{ 2, 2 };
    const sliced = try Buffer.initDynamicSlice(std.testing.allocator, source, &.{ start0_buffer, start1_buffer }, &slice_dims, &slice_dims, 0);
    defer sliced.deinit();
    try std.testing.expect(sliced.backend_buffer != null);
    try std.testing.expectEqual(@as(usize, 0), sliced.bytes.len);
    try expectBufferF32(sliced, &.{ 6.0, 7.0, 10.0, 11.0 });

    const update_values = [_]f32{ 100.0, 101.0, 102.0, 103.0 };
    const update = try initHostCopyForTest(.f32, &slice_dims, &client.devices[0], &client.memories[0], 0, std.mem.asBytes(&update_values));
    defer update.deinit();
    const updated = try Buffer.initDynamicUpdateSlice(std.testing.allocator, source, update, &.{ start0_buffer, start1_buffer }, &dims, 0);
    defer updated.deinit();
    try std.testing.expect(updated.backend_buffer != null);
    try std.testing.expectEqual(@as(usize, 0), updated.bytes.len);
    try expectBufferF32(updated, &.{
        1.0, 2.0,   3.0,   4.0,
        5.0, 100.0, 101.0, 8.0,
        9.0, 102.0, 103.0, 12.0,
    });

    const pad_input_dims = [_]i64{2};
    const pad_output_dims = [_]i64{5};
    const pad_input_values = [_]f32{ 2.0, 3.0 };
    const pad_value: f32 = 0.0;
    const pad_input = try initHostCopyForTest(.f32, &pad_input_dims, &client.devices[0], &client.memories[0], 0, std.mem.asBytes(&pad_input_values));
    defer pad_input.deinit();
    const pad_value_buffer = try initHostCopyForTest(.f32, &.{}, &client.devices[0], &client.memories[0], 0, std.mem.asBytes(&pad_value));
    defer pad_value_buffer.deinit();
    const padded = try Buffer.initPad(std.testing.allocator, pad_input, pad_value_buffer, &.{1}, &.{2}, &.{0}, &pad_output_dims, 0);
    defer padded.deinit();
    try std.testing.expect(padded.backend_buffer != null);
    try std.testing.expectEqual(@as(usize, 0), padded.bytes.len);
    try expectBufferF32(padded, &.{ 0.0, 2.0, 3.0, 0.0, 0.0 });

    const reverse_dims = [_]i64{ 2, 3 };
    const reverse_values = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const reverse_source = try initHostCopyForTest(.f32, &reverse_dims, &client.devices[0], &client.memories[0], 0, std.mem.asBytes(&reverse_values));
    defer reverse_source.deinit();
    const reversed = try Buffer.initReverse(std.testing.allocator, reverse_source, &.{1}, &reverse_dims, 0);
    defer reversed.deinit();
    try std.testing.expect(reversed.backend_buffer != null);
    try std.testing.expectEqual(@as(usize, 0), reversed.bytes.len);
    try expectBufferF32(reversed, &.{ 3.0, 2.0, 1.0, 6.0, 5.0, 4.0 });

    const gather_operand_dims = [_]i64{ 3, 2 };
    const gather_output_dims = [_]i64{ 2, 2 };
    const gather_operand_values = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const gather_indices_values = [_]i32{ 2, 0 };
    const gather_operand = try initHostCopyForTest(.f32, &gather_operand_dims, &client.devices[0], &client.memories[0], 0, std.mem.asBytes(&gather_operand_values));
    defer gather_operand.deinit();
    const gather_indices = try initHostCopyForTest(.s32, &.{2}, &client.devices[0], &client.memories[0], 0, std.mem.asBytes(&gather_indices_values));
    defer gather_indices.deinit();
    const gathered = try Buffer.initGather(
        std.testing.allocator,
        gather_operand,
        gather_indices,
        &.{1},
        &.{0},
        &.{},
        &.{},
        &.{0},
        1,
        &.{ 1, 2 },
        &gather_output_dims,
        0,
    );
    defer gathered.deinit();
    try std.testing.expect(gathered.backend_buffer != null);
    try std.testing.expectEqual(@as(usize, 0), gathered.bytes.len);
    try expectBufferF32(gathered, &.{ 5.0, 6.0, 1.0, 2.0 });

    const sort_dims = [_]i64{ 2, 3 };
    const sort_values = [_]f32{ 3.0, 1.0, 2.0, 6.0, 4.0, 5.0 };
    const sort_source = try initHostCopyForTest(.f32, &sort_dims, &client.devices[0], &client.memories[0], 0, std.mem.asBytes(&sort_values));
    defer sort_source.deinit();
    const sorted = try Buffer.initSort(std.testing.allocator, sort_source, 1, &sort_dims, .lt, 0);
    defer sorted.deinit();
    try std.testing.expect(sorted.backend_buffer != null);
    try std.testing.expectEqual(@as(usize, 0), sorted.bytes.len);
    try expectBufferF32(sorted, &.{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 });

    const reverse_sorted = try Buffer.initSort(std.testing.allocator, sort_source, 1, &sort_dims, .gt, 0);
    defer reverse_sorted.deinit();
    try std.testing.expect(reverse_sorted.backend_buffer != null);
    try std.testing.expectEqual(@as(usize, 0), reverse_sorted.bytes.len);
    try expectBufferF32(reverse_sorted, &.{ 3.0, 2.0, 1.0, 6.0, 5.0, 4.0 });
}

test "buffer elementwise unary math supports f32 execution" {
    const client = try initMlxMetalClientForTest();
    defer client.deinit();

    const dims = [_]i64{2};
    const values = [_]f32{ 1.0, 4.0 };
    const input = try initHostCopyForTest(.f32, &dims, &client.devices[0], &client.memories[0], 0, std.mem.asBytes(&values));
    defer input.deinit();

    const exp = try Buffer.initElementwiseUnary(std.testing.allocator, .exp, input, 0);
    defer exp.deinit();
    try expectBufferF32(exp, &.{ std.math.exp(@as(f32, 1.0)), std.math.exp(@as(f32, 4.0)) });

    const tanh = try Buffer.initElementwiseUnary(std.testing.allocator, .tanh, input, 0);
    defer tanh.deinit();
    try expectBufferF32(tanh, &.{ std.math.tanh(@as(f32, 1.0)), std.math.tanh(@as(f32, 4.0)) });

    const sqrt = try Buffer.initElementwiseUnary(std.testing.allocator, .sqrt, input, 0);
    defer sqrt.deinit();
    try expectBufferF32(sqrt, &.{ 1.0, 2.0 });

    const rsqrt = try Buffer.initElementwiseUnary(std.testing.allocator, .rsqrt, input, 0);
    defer rsqrt.deinit();
    try expectBufferF32(rsqrt, &.{ 1.0, 0.5 });
}

test "buffer reshape preserves typed bytes and updates dimensions" {
    const client = try initMlxMetalClientForTest();
    defer client.deinit();

    const dims = [_]i64{ 2, 2 };
    const values = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const input = try initHostCopyForTest(.f32, &dims, &client.devices[0], &client.memories[0], 0, std.mem.asBytes(&values));
    defer input.deinit();

    const reshaped = try Buffer.initReshape(std.testing.allocator, input, &.{4}, 0);
    defer reshaped.deinit();

    try std.testing.expectEqualSlices(i64, &.{4}, reshaped.dims);
    try expectBufferF32(reshaped, &values);
}

test "buffer transpose permutes dense host bytes and updates dimensions" {
    const client = try initMlxMetalClientForTest();
    defer client.deinit();

    const dims = [_]i64{ 2, 3 };
    const values = [_]u8{ 1, 2, 3, 4, 5, 6 };
    const input = try initHostCopyForTest(.u8, &dims, &client.devices[0], &client.memories[0], 0, &values);
    defer input.deinit();

    const transposed = try Buffer.initTranspose(std.testing.allocator, input, &.{ 1, 0 }, &.{ 3, 2 }, 0);
    defer transposed.deinit();

    try std.testing.expectEqualSlices(i64, &.{ 3, 2 }, transposed.dims);
    try expectBufferBytes(transposed, &.{ 1, 4, 2, 5, 3, 6 });
}

test "buffer broadcast_in_dim expands dense host bytes and updates dimensions" {
    const client = try initMlxMetalClientForTest();
    defer client.deinit();

    const dims = [_]i64{3};
    const values = [_]u8{ 7, 8, 9 };
    const input = try initHostCopyForTest(.u8, &dims, &client.devices[0], &client.memories[0], 0, &values);
    defer input.deinit();

    const broadcasted = try Buffer.initBroadcastInDim(std.testing.allocator, input, &.{1}, &.{ 2, 3 }, 0);
    defer broadcasted.deinit();

    try std.testing.expectEqualSlices(i64, &.{ 2, 3 }, broadcasted.dims);
    try expectBufferBytes(broadcasted, &.{ 7, 8, 9, 7, 8, 9 });
}

test "buffer slice copies strided dense host bytes and updates dimensions" {
    const client = try initMlxMetalClientForTest();
    defer client.deinit();

    const dims = [_]i64{ 3, 4 };
    const values = [_]u8{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const input = try initHostCopyForTest(.u8, &dims, &client.devices[0], &client.memories[0], 0, &values);
    defer input.deinit();

    const sliced = try Buffer.initSlice(
        std.testing.allocator,
        input,
        &.{ 1, 0 },
        &.{ 3, 4 },
        &.{ 1, 2 },
        &.{ 2, 2 },
        0,
    );
    defer sliced.deinit();

    try std.testing.expectEqualSlices(i64, &.{ 2, 2 }, sliced.dims);
    try expectBufferBytes(sliced, &.{ 5, 7, 9, 11 });
}

test "buffer concatenate joins dense host bytes along an axis" {
    const client = try initMlxMetalClientForTest();
    defer client.deinit();

    const lhs_dims = [_]i64{ 2, 2 };
    const rhs_dims = [_]i64{ 2, 3 };
    const lhs_values = [_]u8{ 1, 2, 3, 4 };
    const rhs_values = [_]u8{ 5, 6, 7, 8, 9, 10 };
    const lhs = try initHostCopyForTest(.u8, &lhs_dims, &client.devices[0], &client.memories[0], 0, &lhs_values);
    defer lhs.deinit();
    const rhs = try initHostCopyForTest(.u8, &rhs_dims, &client.devices[0], &client.memories[0], 0, &rhs_values);
    defer rhs.deinit();

    const concatenated = try Buffer.initConcatenate(std.testing.allocator, lhs, rhs, 1, &.{ 2, 5 }, 0);
    defer concatenated.deinit();

    try std.testing.expectEqualSlices(i64, &.{ 2, 5 }, concatenated.dims);
    try expectBufferBytes(concatenated, &.{ 1, 2, 5, 6, 7, 3, 4, 8, 9, 10 });
}

test "buffer partition_id materializes scalar device partition" {
    const client = try initMlxMetalClientForTest();
    defer client.deinit();

    const partition = try Buffer.initPartitionId(std.testing.allocator, mlxMetalBackendForTest(), .u32, &.{}, &client.devices[0], &client.memories[0], 0, 0);
    defer partition.deinit();

    try std.testing.expectEqual(@as(i32, 0), partition.device_id);
    var bytes: [4]u8 = undefined;
    try partition.copyToHost(&bytes);
    try std.testing.expectEqual(@as(u32, 0), readU32LE(&bytes, 0));
}

test "buffer cholesky lowers through backend-native MLX buffers" {
    const client = try initMlxMetalClientForTest();
    defer client.deinit();

    const dims = [_]i64{ 2, 2 };
    const values = [_]f32{ 4.0, 2.0, 2.0, 3.0 };
    const input = try initHostCopyForTest(.f32, &dims, &client.devices[0], &client.memories[0], 0, std.mem.asBytes(&values));
    defer input.deinit();

    const factor = try Buffer.initCholesky(std.testing.allocator, input, true, &dims, 0);
    defer factor.deinit();

    try expectBufferF32(factor, &.{ 2.0, 0.0, 1.0, 1.4142135 });
}

test "buffer rng requires backend-native lowering on MLX buffers" {
    const client = try initMlxMetalClientForTest();
    defer client.deinit();

    const scalar = [_]i64{};
    const min_value = [_]f32{0.0};
    const max_value = [_]f32{1.0};
    const min = try initHostCopyForTest(.f32, &scalar, &client.devices[0], &client.memories[0], 0, std.mem.asBytes(&min_value));
    defer min.deinit();
    const max = try initHostCopyForTest(.f32, &scalar, &client.devices[0], &client.memories[0], 0, std.mem.asBytes(&max_value));
    defer max.deinit();

    try std.testing.expectError(error.UnsupportedRuntimeFeature, Buffer.initRngUniform(std.testing.allocator, min, max, .f32, &.{4}, 0));

    const state_words = [_]u32{ 1, 2 };
    const state = try initHostCopyForTest(.u32, &.{2}, &client.devices[0], &client.memories[0], 0, std.mem.asBytes(&state_words));
    defer state.deinit();
    try std.testing.expectError(error.UnsupportedRuntimeFeature, Buffer.initRngBits(std.testing.allocator, state, .u32, &.{4}, 0));
    try std.testing.expectError(error.UnsupportedRuntimeFeature, Buffer.initRngStateUpdate(std.testing.allocator, state, 0));
}
