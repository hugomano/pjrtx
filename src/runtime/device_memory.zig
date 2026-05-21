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

/// Owns runtime device, memory, handle, and topology arrays derived from backend descriptors.
pub const DeviceMemoryTopology = struct {
    devices: []Device,
    memories: []Memory,
    device_handles: []*Device,
    memory_handles: []*Memory,
    topology: Topology,

    /// Builds runtime topology arrays from backend-provided device descriptors.
    pub fn initFromDescriptors(allocator: std.mem.Allocator, descriptors: []const ir.DeviceDescriptor) !DeviceMemoryTopology {
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

        return .{
            .devices = devices,
            .memories = memories,
            .device_handles = device_handles,
            .memory_handles = memory_handles,
            .topology = .{
                .device_assignment = assignment,
                .num_replicas = 1,
                .num_partitions = @intCast(descriptors.len),
            },
        };
    }

    /// Finds a device by stable PJRT device id.
    pub fn lookupDevice(self: *const DeviceMemoryTopology, id: i32) ?*const Device {
        for (self.devices) |*device| {
            if (device.id == id) return device;
        }
        return null;
    }

    /// Finds an addressable device by backend-local hardware id.
    pub fn lookupAddressableDeviceByLocalHardwareId(self: *const DeviceMemoryTopology, local_hardware_id: i32) ?*const Device {
        for (self.devices) |*device| {
            if (device.addressable and device.local_hardware_id == local_hardware_id) return device;
        }
        return null;
    }

    /// Finds a runtime memory by stable PJRT memory id.
    pub fn lookupMemory(self: *const DeviceMemoryTopology, id: i32) ?*const Memory {
        for (self.memories) |*memory| {
            if (memory.id == id) return memory;
        }
        return null;
    }

    /// Returns the number of addressable devices in this topology.
    pub fn deviceCount(self: *const DeviceMemoryTopology) usize {
        return self.devices.len;
    }

    /// Returns runtime devices for read-only topology and compiler planning.
    pub fn deviceSlice(self: *const DeviceMemoryTopology) []const Device {
        return self.devices;
    }

    /// Returns runtime memories for executable residency and cache accounting.
    pub fn memorySlice(self: *DeviceMemoryTopology) []Memory {
        return self.memories;
    }

    /// Returns runtime device handles in topology order for PJRT adapter lists.
    pub fn deviceHandleSlice(self: *const DeviceMemoryTopology) []const *Device {
        return self.device_handles;
    }

    /// Returns runtime memory handles in topology order for PJRT adapter lists.
    pub fn memoryHandleSlice(self: *const DeviceMemoryTopology) []const *Memory {
        return self.memory_handles;
    }

    /// Returns the default addressable device for placement defaults.
    pub fn defaultDevice(self: *DeviceMemoryTopology) *Device {
        return &self.devices[0];
    }

    /// Returns the default addressable memory for placement defaults.
    pub fn defaultMemory(self: *DeviceMemoryTopology) *Memory {
        return self.defaultDevice().default_memory;
    }

    /// Returns a device's logical index in this runtime topology.
    pub fn deviceIndex(self: *const DeviceMemoryTopology, device: *const Device) ?usize {
        for (self.devices, 0..) |*candidate, i| {
            if (candidate == device or candidate.id == device.id) return i;
        }
        return null;
    }

    /// Returns addressable device handles for a loaded executable.
    pub fn addressableDeviceHandlesForCount(self: *const DeviceMemoryTopology, count: usize) []const *Device {
        return self.device_handles[0..@min(count, self.device_handles.len)];
    }

    /// Finds the memory that should account executable residency for selected backend devices.
    pub fn executableResidencyMemory(self: *DeviceMemoryTopology, device_local_hardware_ids: []const i32) ?*Memory {
        if (device_local_hardware_ids.len != 0) {
            const first_local_hardware_id = device_local_hardware_ids[0];
            for (self.devices) |*device| {
                if (device.local_hardware_id == first_local_hardware_id) return device.default_memory;
            }
        }
        if (self.memories.len == 0) return null;
        return &self.memories[0];
    }

    /// Releases all topology arrays and per-device/per-memory owned strings.
    pub fn deinit(self: DeviceMemoryTopology, allocator: std.mem.Allocator) void {
        for (self.devices) |device| {
            allocator.free(device.addressable_memories);
            allocator.free(device.name);
            allocator.free(device.debug_string);
        }
        for (self.memories) |memory| {
            allocator.free(memory.addressable_devices);
            allocator.free(memory.addressable_device_ids);
            allocator.free(memory.debug_string);
        }
        allocator.free(self.topology.device_assignment);
        allocator.free(self.memory_handles);
        allocator.free(self.device_handles);
        allocator.free(self.memories);
        allocator.free(self.devices);
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
