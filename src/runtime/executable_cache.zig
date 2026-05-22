const std = @import("std");
const mlx_metal = @import("backend_selection.zig");
const ir = @import("src/compiler/ir");

const device_memory = @import("device_memory.zig");

const Memory = device_memory.Memory;

/// Aggregates compile-cache reuse, residency, and memory-pressure counters.
pub const Stats = struct {
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

const Entry = struct {
    fingerprint: []u8,
    backend_executable: ?mlx_metal.ExecutableHandle = null,
    resident_bytes: u64 = 0,
    compile_latency_us: u64 = 0,
    compile_latency_samples: u64 = 0,
    ref_count: usize = 0,
    last_use: u64 = 0,
};

/// Opaque retain token for a resident executable-cache entry.
pub const Lease = struct {
    token: *anyopaque,

    fn fromEntry(cache_entry: *Entry) Lease {
        return .{ .token = cache_entry };
    }

    fn entry(self: Lease) *Entry {
        return @ptrCast(@alignCast(self.token));
    }
};

/// Returns a retained cache entry and the concrete backend handle to execute.
pub const Retained = struct {
    lease: Lease,
    handle: mlx_metal.ExecutableHandle,
    reused: bool,
    compile_trim: Trim = .{},
};

/// Read-only entry state exposed to tests and diagnostics.
pub const EntrySnapshot = struct {
    resident: bool,
    resident_bytes: u64,
    reference_count: usize,
    compile_latency_us: u64,
};

/// Describes cache eviction performed for a memory-pressure request.
pub const Trim = struct {
    requested_bytes: u64 = 0,
    target_resident_bytes: u64 = 0,
    freed_bytes: u64 = 0,
    evicted_entries: u64 = 0,
    remaining_resident_bytes: u64 = 0,
    still_over_capacity: bool = false,
};

/// Owns backend executable residency for repeated PJRT executions.
pub const Cache = struct {
    allocator: std.mem.Allocator,
    entries: std.StringHashMapUnmanaged(*Entry) = .empty,
    stats: Stats = .{},
    max_resident_bytes: u64 = std.math.maxInt(u64),
    next_use: u64 = 0,

    /// Creates an empty executable cache using the runtime allocator.
    pub fn init(allocator: std.mem.Allocator) Cache {
        return .{ .allocator = allocator };
    }

    /// Releases every cached backend executable and owned fingerprint.
    pub const deinit = CacheStorage.deinit;

    /// Records a frontend compile request and returns true when it is a hit.
    pub const recordCompile = CacheStorage.recordCompile;

    /// Updates compile-latency metadata used to choose eviction victims.
    pub fn recordCompileLatency(self: *Cache, fingerprint: []const u8, latency_us: u64) void {
        const entry = self.get(fingerprint) orelse return;
        entry.compile_latency_us = latency_us;
        entry.compile_latency_samples += 1;
        self.stats.compile_latency_samples += 1;
        self.stats.compile_latency_us_total +|= latency_us;
        self.stats.compile_latency_us_peak = @max(self.stats.compile_latency_us_peak, latency_us);
    }

    /// Returns a copy of cache counters for memory accounting and diagnostics.
    pub fn statsSnapshot(self: *const Cache) Stats { return self.stats; }

    /// Returns currently resident backend executable bytes.
    pub fn residentBytes(self: *const Cache) u64 { return self.stats.resident_bytes; }

    /// Records pressure-trim accounting after an owner-triggered trim.
    pub fn recordPressureTrim(self: *Cache, trim: Trim) void {
        self.stats.pressure_trim_requests += 1;
        self.stats.pressure_trimmed_bytes +|= trim.freed_bytes;
        if (trim.still_over_capacity) self.stats.pressure_trim_failures += 1;
    }

    /// Changes the residency budget and evicts idle entries if needed.
    pub const setMaxResidentBytes = CacheEviction.setMaxResidentBytes;

    /// Evicts idle entries until residency is at or below the requested budget.
    pub const trimIdleToResidentBytes = CacheEviction.trimIdleToResidentBytes;

    /// Retains an already resident executable for the given fingerprint.
    pub fn retainResidentByFingerprint(self: *Cache, fingerprint: []const u8) ?Retained {
        const entry = self.entries.get(fingerprint) orelse return null;
        return self.retainResidentEntry(entry, true, .{});
    }

    /// Retains an already resident entry during a compile race recheck.
    pub fn retainResidentEntry(self: *Cache, entry: *Entry, reused: bool, compile_trim: Trim) ?Retained {
        const handle = entry.backend_executable orelse return null;
        CacheEntryRefs.retain(self, entry);
        return .{
            .lease = Lease.fromEntry(entry),
            .handle = handle,
            .reused = reused,
            .compile_trim = compile_trim,
        };
    }

    /// Marks a freshly compiled backend executable as resident for reuse.
    pub const acquireResident = CacheStorage.acquireResident;

    /// Increments an entry reference before runtime execution.
    pub const retain = CacheEntryRefs.retain;

    /// Releases an entry reference and applies residency limits.
    pub const release = CacheEntryRefs.release;

    /// Finds an entry by fingerprint without changing ownership.
    pub fn get(self: *Cache, fingerprint: []const u8) ?*Entry { return self.entries.get(fingerprint); }

    /// Finds a non-retained entry for the client compile path.
    pub fn entryForCompile(self: *Cache, fingerprint: []const u8) ?*Entry { return self.entries.get(fingerprint); }

    /// Returns read-only state for a cache entry.
    pub fn entrySnapshot(self: *const Cache, fingerprint: []const u8) ?EntrySnapshot {
        const entry = self.entries.get(fingerprint) orelse return null;
        return .{
            .resident = entry.backend_executable != null,
            .resident_bytes = entry.resident_bytes,
            .reference_count = entry.ref_count,
            .compile_latency_us = entry.compile_latency_us,
        };
    }

    /// Overrides compile-latency metadata for deterministic cache-policy tests.
    pub fn setCompileLatencyForTest(self: *Cache, fingerprint: []const u8, latency_us: u64) void {
        const entry = self.entries.get(fingerprint) orelse return;
        entry.compile_latency_us = latency_us;
    }

    fn touch(self: *Cache, entry: *Entry) void { self.next_use +|= 1; entry.last_use = self.next_use; }
};

const CacheStorage = struct {
    fn deinit(cache: *Cache, backend: mlx_metal.Backend) void {
        var it = cache.entries.valueIterator();
        while (it.next()) |entry_ptr| {
            const entry = entry_ptr.*;
            if (entry.backend_executable) |handle| backend.destroyExecutable(handle);
            cache.allocator.free(entry.fingerprint);
            cache.allocator.destroy(entry);
        }
        cache.entries.deinit(cache.allocator);
        cache.* = undefined;
    }

    fn recordCompile(cache: *Cache, fingerprint: []const u8) !bool {
        if (cache.entries.contains(fingerprint)) {
            cache.stats.hits += 1;
            return true;
        }
        const entry = try cache.allocator.create(Entry);
        errdefer cache.allocator.destroy(entry);
        entry.* = .{ .fingerprint = try cache.allocator.dupe(u8, fingerprint) };
        errdefer cache.allocator.free(entry.fingerprint);
        try cache.entries.put(cache.allocator, entry.fingerprint, entry);
        cache.stats.misses += 1;
        return false;
    }

    fn acquireResident(cache: *Cache, backend: mlx_metal.Backend, entry: *Entry, handle: mlx_metal.ExecutableHandle, resident_bytes: u64, compile_trim: Trim) Retained {
        entry.backend_executable = handle;
        entry.resident_bytes = resident_bytes;
        entry.ref_count = 1;
        cache.touch(entry);
        cache.stats.resident_entries += 1;
        cache.stats.resident_bytes +|= resident_bytes;
        cache.stats.peak_resident_bytes = @max(cache.stats.peak_resident_bytes, cache.stats.resident_bytes);
        cache.stats.largest_resident_bytes = @max(cache.stats.largest_resident_bytes, resident_bytes);
        CacheEviction.evictIdleUntilUnderLimit(cache, backend);
        return .{ .lease = Lease.fromEntry(entry), .handle = handle, .reused = false, .compile_trim = compile_trim };
    }
};

const CacheEntryRefs = struct {
    fn retain(cache: *Cache, entry: *Entry) void {
        entry.ref_count += 1;
        cache.touch(entry);
    }

    fn release(cache: *Cache, backend: mlx_metal.Backend, lease: Lease) void {
        const entry = lease.entry();
        if (entry.ref_count != 0) entry.ref_count -= 1;
        cache.touch(entry);
        CacheEviction.evictIdleUntilUnderLimit(cache, backend);
    }
};

const CacheEviction = struct {
    fn setMaxResidentBytes(cache: *Cache, backend: mlx_metal.Backend, max_resident_bytes: u64) void {
        cache.max_resident_bytes = max_resident_bytes;
        evictIdleUntilUnderLimit(cache, backend);
    }

    fn trimIdleToResidentBytes(cache: *Cache, backend: mlx_metal.Backend, target_resident_bytes: u64) Trim {
        const before_bytes = cache.stats.resident_bytes;
        const before_evictions = cache.stats.evictions;
        while (cache.stats.resident_bytes > target_resident_bytes) {
            const victim = pressureEvictionVictim(cache) orelse break;
            evictIdleEntry(cache, backend, victim);
        }
        return .{
            .target_resident_bytes = target_resident_bytes,
            .freed_bytes = before_bytes - cache.stats.resident_bytes,
            .evicted_entries = cache.stats.evictions - before_evictions,
            .remaining_resident_bytes = cache.stats.resident_bytes,
            .still_over_capacity = cache.stats.resident_bytes > target_resident_bytes,
        };
    }

    fn evictIdleUntilUnderLimit(self: *Cache, backend: mlx_metal.Backend) void {
        while (self.stats.resident_bytes > self.max_resident_bytes) {
            const victim = pressureEvictionVictim(self) orelse return;
            evictIdleEntry(self, backend, victim);
        }
    }

    fn pressureEvictionVictim(self: *Cache) ?*Entry {
        var victim: ?*Entry = null;
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

    fn evictIdleEntry(self: *Cache, backend: mlx_metal.Backend, entry: *Entry) void {
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

/// Owns executable-cache synchronization and PJRT-visible memory accounting.
pub const Residency = struct {
    cache: Cache,
    mutex: std.Io.Mutex = std.Io.Mutex.init,

    /// Creates an empty synchronized executable-residency cache.
    pub fn init(allocator: std.mem.Allocator) Residency {
        return .{ .cache = Cache.init(allocator) };
    }

    /// Releases all resident backend executables and cache entries.
    pub fn deinit(self: *Residency, backend: mlx_metal.Backend) void {
        self.cache.deinit(backend);
        self.* = undefined;
    }

    /// Records a compile request and mirrors cache counters into memory stats.
    pub fn recordCompile(self: *Residency, io: std.Io, memories: []Memory, fingerprint: []const u8) !bool {
        self.lock(io);
        defer self.unlock(io);
        const hit = try self.cache.recordCompile(fingerprint);
        self.syncMemoryStats(memories);
        return hit;
    }

    /// Returns executable-cache counters without exposing cache entries.
    pub fn statsSnapshot(self: *const Residency) Stats {
        return self.cache.statsSnapshot();
    }

    /// Returns a read-only cache entry snapshot for focused tests.
    pub fn entrySnapshot(self: *const Residency, fingerprint: []const u8) ?EntrySnapshot {
        return self.cache.entrySnapshot(fingerprint);
    }

    /// Overrides compile-latency metadata for deterministic cache-policy tests.
    pub fn setCompileLatencyForTest(self: *Residency, fingerprint: []const u8, latency_us: u64) void {
        self.cache.setCompileLatencyForTest(fingerprint, latency_us);
    }

    /// Sets the resident executable-cache budget and evicts idle entries as needed.
    pub fn setMaxResidentBytes(self: *Residency, io: std.Io, backend: mlx_metal.Backend, memories: []Memory, max_resident_bytes: u64) void {
        self.lock(io);
        defer self.unlock(io);
        self.cache.setMaxResidentBytes(backend, max_resident_bytes);
        self.syncMemoryStats(memories);
    }

    /// Trims idle resident executables to make room for a device allocation.
    pub fn trimForAllocation(self: *Residency, io: std.Io, backend: mlx_metal.Backend, memories: []Memory, memory: *Memory, allocation_bytes: usize) Trim {
        return ResidencyPressure.trimForAllocation(self, io, backend, memories, memory, allocation_bytes);
    }

    /// Acquires a retained resident backend executable, compiling one if no resident entry exists.
    pub fn acquireBackendExecutable(
        self: *Residency,
        io: std.Io,
        backend: mlx_metal.Backend,
        allocator: std.mem.Allocator,
        memories: []Memory,
        residency_memory: ?*Memory,
        fingerprint: []const u8,
        plan: *const ir.ExecutablePlan,
        device_local_hardware_ids: []const i32,
    ) (std.mem.Allocator.Error || mlx_metal.Error || error{UnsupportedRuntimeFeature})!?Retained {
        return ResidencyAcquire.acquireBackendExecutable(self, io, backend, allocator, memories, residency_memory, fingerprint, plan, device_local_hardware_ids);
    }

    /// Releases a retained executable-cache entry after residency teardown.
    pub fn release(self: *Residency, io: std.Io, backend: mlx_metal.Backend, memories: []Memory, lease: Lease) void {
        self.lock(io);
        defer self.unlock(io);
        self.cache.release(backend, lease);
        self.syncMemoryStats(memories);
    }

    fn lock(self: *Residency, io: std.Io) void {
        self.mutex.lockUncancelable(io);
    }

    fn unlock(self: *Residency, io: std.Io) void {
        self.mutex.unlock(io);
    }

    fn syncMemoryStats(self: *Residency, memories: []Memory) void {
        const stats = self.cache.statsSnapshot();
        for (memories) |*memory| {
            memory.stats.executable_cache_hits = stats.hits;
            memory.stats.executable_cache_misses = stats.misses;
            memory.stats.executable_cache_evictions = stats.evictions;
            memory.stats.executable_cache_resident_entries = stats.resident_entries;
            memory.stats.executable_cache_resident_bytes = stats.resident_bytes;
            memory.stats.executable_cache_peak_resident_bytes = stats.peak_resident_bytes;
            memory.stats.executable_cache_largest_resident_bytes = stats.largest_resident_bytes;
            memory.stats.executable_cache_pressure_trims = stats.pressure_trim_requests;
            memory.stats.executable_cache_pressure_trimmed_bytes = stats.pressure_trimmed_bytes;
            memory.stats.executable_cache_pressure_trim_failures = stats.pressure_trim_failures;
        }
    }
};

const ResidencyPressure = struct {
    fn trimForAllocation(self: *Residency, io: std.Io, backend: mlx_metal.Backend, memories: []Memory, memory: *Memory, allocation_bytes: usize) Trim {
        self.lock(io);
        defer self.unlock(io);
        if (memory.stats.capacity_bytes == 0) return .{};
        const allocation: u64 = @intCast(allocation_bytes);
        const bytes_without_cache = memory.stats.bytes_in_use +| allocation;
        const resident_bytes = self.cache.residentBytes();
        if (bytes_without_cache +| resident_bytes <= memory.stats.capacity_bytes) return .{
            .requested_bytes = allocation,
            .target_resident_bytes = resident_bytes,
            .remaining_resident_bytes = resident_bytes,
        };
        const target_resident_bytes = if (bytes_without_cache >= memory.stats.capacity_bytes) 0 else memory.stats.capacity_bytes - bytes_without_cache;
        var trim = self.cache.trimIdleToResidentBytes(backend, target_resident_bytes);
        trim.requested_bytes = allocation;
        trim.still_over_capacity = bytes_without_cache +| trim.remaining_resident_bytes > memory.stats.capacity_bytes;
        self.cache.recordPressureTrim(trim);
        self.syncMemoryStats(memories);
        return trim;
    }
};

const ResidencyAcquire = struct {
    const CompileEntry = union(enum) { missing, entry: *Entry, retained: Retained };

    fn acquireBackendExecutable(
        self: *Residency,
        io: std.Io,
        backend: mlx_metal.Backend,
        allocator: std.mem.Allocator,
        memories: []Memory,
        residency_memory: ?*Memory,
        fingerprint: []const u8,
        plan: *const ir.ExecutablePlan,
        device_local_hardware_ids: []const i32,
    ) (std.mem.Allocator.Error || mlx_metal.Error || error{UnsupportedRuntimeFeature})!?Retained {
        const entry = switch (entryForCompile(self, io, memories, fingerprint)) {
            .missing => return null,
            .retained => |retained| return retained,
            .entry => |entry| entry,
        };
        if (retainRaceWinner(self, io, memories, entry)) |retained| return retained;

        const compile_start = nowTimestamp(io);
        const handle = backend.compileExecutable(allocator, plan, device_local_hardware_ids) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => null,
        };
        const compile_latency_us = elapsedMicrosSince(io, compile_start);
        const owned_handle = handle orelse return null;
        const stats = backend.executableStats(owned_handle);
        const resident_constant_bytes: usize = @intCast(stats.resident_constant_bytes);
        const compile_trim = if (resident_constant_bytes != 0 and residency_memory != null)
            self.trimForAllocation(io, backend, memories, residency_memory.?, resident_constant_bytes)
        else
            Trim{};

        self.lock(io);
        defer self.unlock(io);
        if (self.cache.retainResidentEntry(entry, true, compile_trim)) |retained| {
            backend.destroyExecutable(owned_handle);
            self.syncMemoryStats(memories);
            return retained;
        }
        const retained = self.cache.acquireResident(backend, entry, owned_handle, resident_constant_bytes, compile_trim);
        self.cache.recordCompileLatency(fingerprint, compile_latency_us);
        self.syncMemoryStats(memories);
        return retained;
    }

    fn entryForCompile(self: *Residency, io: std.Io, memories: []Memory, fingerprint: []const u8) CompileEntry {
        self.lock(io);
        defer self.unlock(io);
        const entry = self.cache.entryForCompile(fingerprint) orelse return .missing;
        if (self.cache.retainResidentEntry(entry, true, .{})) |retained| {
            self.syncMemoryStats(memories);
            return .{ .retained = retained };
        }
        return .{ .entry = entry };
    }

    fn retainRaceWinner(self: *Residency, io: std.Io, memories: []Memory, entry: *Entry) ?Retained {
        self.lock(io);
        defer self.unlock(io);
        if (self.cache.retainResidentEntry(entry, true, .{})) |retained| {
            self.syncMemoryStats(memories);
            return retained;
        }
        return null;
    }
};

fn nowTimestamp(io: std.Io) std.Io.Timestamp { return std.Io.Timestamp.now(io, .awake); }

fn elapsedMicrosSince(io: std.Io, start: std.Io.Timestamp) u64 { return @intCast(@max(start.durationTo(nowTimestamp(io)).toMicroseconds(), 0)); }
