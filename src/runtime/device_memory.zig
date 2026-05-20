const std = @import("std");
const ir = @import("src/compiler/ir");

/// Maximum device slots represented in compiler plans and runtime topology.
pub const MAX_DEVICES = ir.MAX_DEVICES;

/// Compiler-owned memory-kind vocabulary used by runtime placement.
pub const MemoryKind = ir.MemoryKind;

/// Describes one addressable Metal/MLX device exposed through PJRT.
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

/// Describes one PJRT memory space and the devices that can address it.
pub const Memory = struct {
    id: i32,
    kind: MemoryKind,
    debug_string: []const u8,
    addressable_device_ids: []const i32,
    addressable_devices: []*Device,
    stats: MemoryStats = .{},

    /// Returns true when this memory can be used by the given runtime device.
    pub fn isAddressableBy(self: Memory, device: *const Device) bool {
        for (self.addressable_device_ids) |device_id| {
            if (device_id == device.id) return true;
        }
        return false;
    }
};

/// Owns the replica/partition assignment visible to compilation and execute.
pub const Topology = struct {
    device_assignment: []const i32,
    num_replicas: i32,
    num_partitions: i32,

    /// Returns the number of logical device slots in the topology.
    pub fn numDevices(self: Topology) usize {
        return @intCast(self.num_replicas * self.num_partitions);
    }
};

/// Tracks device-memory residency, transfers, and executable-cache pressure.
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

    /// Accounts one newly-live allocation in this memory space.
    pub fn retain(self: *MemoryStats, bytes: usize) void {
        const amount: u64 = @intCast(bytes);
        self.bytes_in_use +|= amount;
        self.peak_bytes_in_use = @max(self.peak_bytes_in_use, self.bytes_in_use);
        self.live_allocs += 1;
        self.total_allocs += 1;
        self.largest_alloc_size = @max(self.largest_alloc_size, amount);
    }

    /// Accounts one allocation becoming no longer live.
    pub fn release(self: *MemoryStats, bytes: usize) void {
        const amount: u64 = @intCast(bytes);
        self.bytes_in_use = if (amount > self.bytes_in_use) 0 else self.bytes_in_use - amount;
        if (self.live_allocs != 0) self.live_allocs -= 1;
    }

    /// Returns buffer bytes plus resident executable-cache bytes.
    pub fn totalBytesInUse(self: MemoryStats) u64 {
        return self.bytes_in_use +| self.executable_cache_resident_bytes;
    }

    /// Returns the peak of buffer and executable-cache memory pressure.
    pub fn peakTotalBytesInUse(self: MemoryStats) u64 {
        return @max(self.totalBytesInUse(), self.peak_bytes_in_use +| self.executable_cache_peak_resident_bytes);
    }
};
