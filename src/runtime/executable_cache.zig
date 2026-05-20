const std = @import("std");
const mlx_metal = @import("src/backend/mlx_metal");

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

/// Owns one executable fingerprint and, when resident, its backend program.
pub const Entry = struct {
    fingerprint: []u8,
    backend_executable: ?mlx_metal.ExecutableHandle = null,
    resident_bytes: u64 = 0,
    compile_latency_us: u64 = 0,
    compile_latency_samples: u64 = 0,
    ref_count: usize = 0,
    last_use: u64 = 0,
};

/// Returns a retained cache entry and the concrete backend handle to execute.
pub const Retained = struct {
    entry: *Entry,
    handle: mlx_metal.ExecutableHandle,
    reused: bool,
    compile_trim: Trim = .{},
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
    pub fn deinit(self: *Cache, backend: mlx_metal.Backend) void {
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

    /// Records a frontend compile request and returns true when it is a hit.
    pub fn recordCompile(self: *Cache, fingerprint: []const u8) !bool {
        if (self.entries.contains(fingerprint)) {
            self.stats.hits += 1;
            return true;
        }
        const entry = try self.allocator.create(Entry);
        errdefer self.allocator.destroy(entry);
        entry.* = .{
            .fingerprint = try self.allocator.dupe(u8, fingerprint),
        };
        errdefer self.allocator.free(entry.fingerprint);
        try self.entries.put(self.allocator, entry.fingerprint, entry);
        self.stats.misses += 1;
        return false;
    }

    /// Updates compile-latency metadata used to choose eviction victims.
    pub fn recordCompileLatency(self: *Cache, fingerprint: []const u8, latency_us: u64) void {
        const entry = self.get(fingerprint) orelse return;
        entry.compile_latency_us = latency_us;
        entry.compile_latency_samples += 1;
        self.stats.compile_latency_samples += 1;
        self.stats.compile_latency_us_total +|= latency_us;
        self.stats.compile_latency_us_peak = @max(self.stats.compile_latency_us_peak, latency_us);
    }

    /// Changes the residency budget and evicts idle entries if needed.
    pub fn setMaxResidentBytes(self: *Cache, backend: mlx_metal.Backend, max_resident_bytes: u64) void {
        self.max_resident_bytes = max_resident_bytes;
        self.evictIdleUntilUnderLimit(backend);
    }

    /// Evicts idle entries until residency is at or below the requested budget.
    pub fn trimIdleToResidentBytes(self: *Cache, backend: mlx_metal.Backend, target_resident_bytes: u64) Trim {
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

    /// Marks a freshly compiled backend executable as resident for reuse.
    pub fn acquireResident(
        self: *Cache,
        backend: mlx_metal.Backend,
        entry: *Entry,
        handle: mlx_metal.ExecutableHandle,
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

    /// Increments an entry reference before runtime execution.
    pub fn retain(self: *Cache, entry: *Entry) void {
        entry.ref_count += 1;
        self.touch(entry);
    }

    /// Releases an entry reference and applies residency limits.
    pub fn release(self: *Cache, backend: mlx_metal.Backend, entry: *Entry) void {
        if (entry.ref_count != 0) entry.ref_count -= 1;
        self.touch(entry);
        self.evictIdleUntilUnderLimit(backend);
    }

    /// Finds an entry by fingerprint without changing ownership.
    pub fn get(self: *Cache, fingerprint: []const u8) ?*Entry {
        return self.entries.get(fingerprint);
    }

    fn touch(self: *Cache, entry: *Entry) void {
        self.next_use +|= 1;
        entry.last_use = self.next_use;
    }

    fn evictIdleUntilUnderLimit(self: *Cache, backend: mlx_metal.Backend) void {
        while (self.stats.resident_bytes > self.max_resident_bytes) {
            const victim = self.pressureEvictionVictim() orelse return;
            self.evictIdleEntry(backend, victim);
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
