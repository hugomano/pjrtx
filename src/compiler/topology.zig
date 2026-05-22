const std = @import("std");

/// Maximum number of devices the compiler IR topology vocabulary can describe.
pub const MAX_DEVICES = 64;

/// Memory class used by compiler-owned placement and topology descriptions.
pub const MemoryKind = enum {
    device,
    host_pinned,
    host_unpinned,
    accelerator_local,
    interconnect_visible,
};

/// Device facts discovered before compilation and consumed by placement.
pub const DeviceDescriptor = struct {
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
};

/// Memory facts discovered before compilation and consumed by placement.
pub const MemoryDescriptor = struct {
    id: i32,
    kind: MemoryKind,
    debug_string: []const u8,
};

/// Concrete device-memory placement for a compiler value.
pub const Placement = struct {
    device_id: i32,
    memory_id: i32,
    memory_kind: MemoryKind,
};

/// Replica and partition device assignment for a compiled program.
pub const Topology = struct {
    device_assignment: []const i32,
    num_replicas: i32,
    num_partitions: i32,

    /// Returns the total logical device slots required by replicas and partitions.
    pub fn numDevices(self: Topology) usize {
        return @intCast(self.num_replicas * self.num_partitions);
    }
};

/// User and runtime compile options normalized for compiler planning.
pub const CompileOptions = struct {
    num_replicas: i32 = 1,
    num_partitions: i32 = 1,
    use_shardy_partitioner: bool = true,
    device_assignment: []const i32 = &.{},

    /// Returns the total logical device slots requested by replicas and partitions.
    pub fn numDevices(self: CompileOptions) usize {
        return @intCast(self.num_replicas * self.num_partitions);
    }
};

/// Sharding strategy class attached to parameters and outputs.
pub const ShardingKind = enum {
    replicated,
    partitioned,
    manual,
};

/// Owned sharding metadata decoded from program attributes.
pub const ShardingMetadata = struct {
    kind: ShardingKind,
    mesh_name: []const u8,

    /// Releases metadata storage owned by this sharding record.
    pub fn deinit(self: ShardingMetadata, allocator: std.mem.Allocator) void {
        allocator.free(self.mesh_name);
    }
};

/// Executable-plan sharding assignment for one parameter or output.
pub const ShardingPlan = struct {
    kind: ShardingKind,
    mesh_name: []const u8,
    device_assignment: []const i32,
};
