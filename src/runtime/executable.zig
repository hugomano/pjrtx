const std = @import("std");
const backend_api = @import("src/backend/mlx_metal");
const ir = @import("src/compiler/ir");

const device_memory = @import("device_memory.zig");
const executable_cache = @import("executable_cache.zig");

const Device = device_memory.Device;
const Memory = device_memory.Memory;
const CachedBackendExecutable = executable_cache.Retained;
const ExecutableCacheEntry = executable_cache.Entry;
const ExecutableCacheTrim = executable_cache.Trim;

/// Aggregates donated-output aliasing performed during graph execution.
pub const DonationAliasStats = struct {
    output_count: usize = 0,
    output_bytes: usize = 0,
};

/// Owns the compiler plan and resident runtime graph returned by compilation.
pub const CompiledExecutable = struct {
    plan: *ir.ExecutablePlan,
    graph: ExecutableGraph,
    optimized_program: []u8,
    fingerprint: []u8,
    cache_hit: bool,
    backend_stats: backend_api.ExecutableStats,

    /// Releases graph residency, optimized program text, fingerprint, and plan.
    pub fn deinit(self: *CompiledExecutable, allocator: std.mem.Allocator) void {
        self.graph.deinit();
        allocator.free(self.fingerprint);
        allocator.free(self.optimized_program);
        self.plan.deinit();
        allocator.destroy(self.plan);
        self.* = undefined;
    }
};

/// Error set for acquiring a cached or freshly compiled backend executable.
pub const CacheAcquireError = std.mem.Allocator.Error || backend_api.Error || error{UnsupportedRuntimeFeature};

/// Callback used by executable graphs to acquire a resident backend executable.
pub const AcquireCachedExecutableFn = *const fn (*anyopaque, std.mem.Allocator, []const u8, *const ir.ExecutablePlan, []const i32) CacheAcquireError!?CachedBackendExecutable;

/// Callback used by executable graphs to release a retained cache entry.
pub const ReleaseCachedExecutableFn = *const fn (*anyopaque, *ExecutableCacheEntry) void;

/// Callback used by execution to trim resident executable cache for output allocation.
pub const TrimExecutableCacheFn = *const fn (*anyopaque, *Memory, usize) ExecutableCacheTrim;

/// Narrow runtime surface needed by executable lowering and execution.
pub const Context = struct {
    backend: backend_api.Backend,
    devices: []Device,
    user_context: ?*anyopaque = null,
    acquire_cached_executable: ?AcquireCachedExecutableFn = null,
    release_cached_executable: ?ReleaseCachedExecutableFn = null,
    trim_executable_cache: ?TrimExecutableCacheFn = null,

    /// Finds a device by stable PJRT id within the context topology.
    pub fn lookupDevice(self: Context, id: i32) ?*const Device {
        for (self.devices) |*device| {
            if (device.id == id) return device;
        }
        return null;
    }

    fn acquireCachedExecutable(self: Context, allocator: std.mem.Allocator, fingerprint: []const u8, plan: *const ir.ExecutablePlan, device_local_hardware_ids: []const i32) CacheAcquireError!?CachedBackendExecutable {
        const callback = self.acquire_cached_executable orelse return null;
        const user_context = self.user_context orelse return null;
        return callback(user_context, allocator, fingerprint, plan, device_local_hardware_ids);
    }

    fn releaseCachedExecutable(self: Context, entry: *ExecutableCacheEntry) void {
        if (self.release_cached_executable) |callback| {
            if (self.user_context) |user_context| callback(user_context, entry);
        } else if (entry.ref_count != 0) {
            entry.ref_count -= 1;
        }
    }

    /// Requests executable-cache pressure relief before execution allocates outputs.
    pub fn trimExecutableCacheForAllocation(self: Context, memory: *Memory, allocation_bytes: usize) ExecutableCacheTrim {
        const callback = self.trim_executable_cache orelse return .{ .requested_bytes = @intCast(allocation_bytes) };
        const user_context = self.user_context orelse return .{ .requested_bytes = @intCast(allocation_bytes) };
        return callback(user_context, memory, allocation_bytes);
    }
};

/// Classifies runtime graph nodes without exposing compiler instruction names.
pub const GraphNodeKind = enum {
    constant,
    parameter,
    compute,
    collective,
    custom_call,
    control_flow,
    structural,
};

/// Describes one scheduled plan instruction on one assigned runtime device.
pub const GraphNode = struct {
    instruction_index: usize,
    device_index: usize,
    device_id: i32,
    kind: GraphNodeKind,
};

/// Carries optional lowering diagnostics and cache identity into graph creation.
pub const LoweringOptions = struct {
    diagnostic_writer: ?*std.Io.Writer = null,
    cache_fingerprint: ?[]const u8 = null,
};

/// Records the runtime lowering outcome needed by execution and diagnostics.
pub const LoweringPipeline = struct {
    backend_executable_ready: bool,
    backend_executable_cache_reused: bool = false,
    lowered_instruction_count: usize,
};

/// Owns the resident backend executable and per-device runtime graph metadata.
pub const ExecutableGraph = struct {
    allocator: std.mem.Allocator,
    backend: backend_api.Backend,
    device_ids: []i32,
    device_local_hardware_ids: []i32,
    nodes: []GraphNode,
    backend_executable: ?backend_api.ExecutableHandle = null,
    backend_executable_cache_entry: ?*ExecutableCacheEntry = null,
    context: Context,
    lowering: LoweringPipeline,
    last_compile_cache_trim: ExecutableCacheTrim = .{},
    last_execute_cache_trim: ExecutableCacheTrim = .{},
    last_backend_completion: backend_api.ExecutionCompletion = .{},
    donation_alias_stats: DonationAliasStats = .{},

    /// Builds an executable graph for the provided runtime context and plan.
    pub fn init(allocator: std.mem.Allocator, context: Context, plan: *const ir.ExecutablePlan) !ExecutableGraph {
        return initWithOptions(allocator, context, plan, .{});
    }

    /// Builds an executable graph and optionally acquires a cached backend executable.
    pub fn initWithOptions(allocator: std.mem.Allocator, context: Context, plan: *const ir.ExecutablePlan, options: LoweringOptions) !ExecutableGraph {
        const device_count = plan.options.numDevices();
        if (device_count == 0 or device_count > context.devices.len) return error.InvalidGraph;

        const device_ids = try allocator.alloc(i32, device_count);
        errdefer allocator.free(device_ids);
        const device_local_hardware_ids = try allocator.alloc(i32, device_count);
        errdefer allocator.free(device_local_hardware_ids);
        for (device_ids, 0..) |*device_id, i| {
            device_id.* = if (plan.options.device_assignment.len != 0)
                plan.options.device_assignment[i]
            else
                context.devices[i].id;
            const device = context.lookupDevice(device_id.*) orelse return error.InvalidGraph;
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
        var backend_executable_cache_reused = false;
        var last_compile_cache_trim = ExecutableCacheTrim{};
        const backend_executable = if (options.cache_fingerprint) |fingerprint| blk: {
            const cached = try context.acquireCachedExecutable(allocator, fingerprint, plan, device_local_hardware_ids);
            if (cached) |entry| {
                backend_executable_cache_entry = entry.entry;
                backend_executable_cache_reused = entry.reused;
                last_compile_cache_trim = entry.compile_trim;
                break :blk entry.handle;
            }
            break :blk null;
        } else context.backend.compileExecutable(allocator, plan, device_local_hardware_ids) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => null,
        };
        errdefer if (backend_executable) |handle| {
            if (backend_executable_cache_entry) |entry| context.releaseCachedExecutable(entry) else context.backend.destroyExecutable(handle);
        };
        if (backend_executable == null) {
            if (options.diagnostic_writer) |writer| {
                context.backend.writeExecutableLoweringDiagnostic(plan, device_local_hardware_ids, writer) catch {};
            }
            return error.UnsupportedRuntimeFeature;
        }

        return .{
            .allocator = allocator,
            .backend = context.backend,
            .device_ids = device_ids,
            .device_local_hardware_ids = device_local_hardware_ids,
            .nodes = nodes,
            .backend_executable = backend_executable,
            .backend_executable_cache_entry = backend_executable_cache_entry,
            .context = context,
            .lowering = .{
                .backend_executable_ready = true,
                .backend_executable_cache_reused = backend_executable_cache_reused,
                .lowered_instruction_count = plan.instructions.len,
            },
            .last_compile_cache_trim = last_compile_cache_trim,
        };
    }

    /// Releases backend residency, cache references, and graph-owned storage.
    pub fn deinit(self: *ExecutableGraph) void {
        if (self.backend_executable) |handle| {
            if (self.backend_executable_cache_entry) |entry| {
                self.context.releaseCachedExecutable(entry);
            } else {
                self.backend.destroyExecutable(handle);
            }
        }
        self.allocator.free(self.device_local_hardware_ids);
        self.allocator.free(self.nodes);
        self.allocator.free(self.device_ids);
        self.* = undefined;
    }

    /// Returns backend residency statistics with runtime donation aliases included.
    pub fn backendExecutableStats(self: *const ExecutableGraph) ?backend_api.ExecutableStats {
        const handle = self.backend_executable orelse return null;
        var stats = self.backend.executableStats(handle);
        stats.donation_alias_output_count += self.donation_alias_stats.output_count;
        stats.donation_alias_output_bytes += self.donation_alias_stats.output_bytes;
        return stats;
    }
};

fn graphNodeKind(kind: ir.PlanInstructionKind) GraphNodeKind {
    return switch (kind) {
        .constant => .constant,
        .custom_call => .custom_call,
        .while_ => .control_flow,
        .tuple, .get_tuple_element, .optimization_barrier => .structural,
        else => .compute,
    };
}

const ExecutableTestContext = struct {
    allocator: std.mem.Allocator,
    backend: backend_api.Backend,
    descriptors: []ir.DeviceDescriptor,
    devices: []Device,
    memories: []Memory,
    addressable_device_ids: []i32,
    addressable_devices: []*Device,
    addressable_memories: []*Memory,

    fn init(allocator: std.mem.Allocator) !ExecutableTestContext {
        const backend = backend_api.create();
        const descriptors = try backend.enumerateDevices(allocator, 1);
        errdefer backend.releaseDeviceDescriptors(allocator, descriptors);
        if (descriptors.len == 0) return error.InvalidDeviceCount;

        const devices = try allocator.alloc(Device, 1);
        errdefer allocator.free(devices);

        const memories = try allocator.alloc(Memory, 1);
        errdefer allocator.free(memories);

        const addressable_device_ids = try allocator.alloc(i32, 1);
        errdefer allocator.free(addressable_device_ids);

        const addressable_devices = try allocator.alloc(*Device, 1);
        errdefer allocator.free(addressable_devices);

        const addressable_memories = try allocator.alloc(*Memory, 1);
        errdefer allocator.free(addressable_memories);

        const descriptor = descriptors[0];
        addressable_device_ids[0] = descriptor.id;
        memories[0] = .{
            .id = descriptor.default_memory_id,
            .kind = .device,
            .debug_string = "device",
            .addressable_device_ids = addressable_device_ids,
            .addressable_devices = addressable_devices,
            .stats = .{ .capacity_bytes = descriptor.memory_bytes },
        };
        devices[0] = .{
            .id = descriptor.id,
            .local_hardware_id = descriptor.local_hardware_id,
            .registry_id = descriptor.registry_id,
            .process_index = descriptor.process_index,
            .addressable = descriptor.addressable,
            .name = descriptor.name,
            .debug_string = descriptor.debug_string,
            .memory_bytes = descriptor.memory_bytes,
            .has_unified_memory = descriptor.has_unified_memory,
            .default_memory_id = descriptor.default_memory_id,
            .default_memory = &memories[0],
            .addressable_memories = addressable_memories,
        };
        addressable_devices[0] = &devices[0];
        addressable_memories[0] = &memories[0];

        return .{
            .allocator = allocator,
            .backend = backend,
            .descriptors = descriptors,
            .devices = devices,
            .memories = memories,
            .addressable_device_ids = addressable_device_ids,
            .addressable_devices = addressable_devices,
            .addressable_memories = addressable_memories,
        };
    }

    fn deinit(self: *ExecutableTestContext) void {
        self.allocator.free(self.addressable_memories);
        self.allocator.free(self.addressable_devices);
        self.allocator.free(self.addressable_device_ids);
        self.allocator.free(self.memories);
        self.allocator.free(self.devices);
        self.backend.releaseDeviceDescriptors(self.allocator, self.descriptors);
        self.* = undefined;
    }

    fn executableContext(self: *ExecutableTestContext) Context {
        return .{
            .backend = self.backend,
            .devices = self.devices,
        };
    }
};

fn testShardingPlan(allocator: std.mem.Allocator, assignment: []const i32) !ir.ShardingPlan {
    return .{
        .kind = .replicated,
        .mesh_name = try allocator.dupe(u8, ""),
        .device_assignment = try allocator.dupe(i32, assignment),
    };
}

test "executable graph materializes per-device scheduled nodes" {
    var ctx = try ExecutableTestContext.init(std.testing.allocator);
    defer ctx.deinit();
    const allocator = std.testing.allocator;
    const assignment = [_]i32{0};
    const dims = [_]i64{4};

    const values = try allocator.alloc(ir.Value, 3);
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

    var parameter_shardings = try allocator.alloc(ir.ShardingPlan, 2);
    parameter_shardings[0] = try testShardingPlan(allocator, &assignment);
    parameter_shardings[1] = try testShardingPlan(allocator, &assignment);
    var output_shardings = try allocator.alloc(ir.ShardingPlan, 1);
    output_shardings[0] = try testShardingPlan(allocator, &assignment);

    var plan = ir.ExecutablePlan{
        .allocator = allocator,
        .options = .{
            .num_replicas = 1,
            .num_partitions = 1,
            .device_assignment = try allocator.dupe(i32, &assignment),
        },
        .values = values,
        .parameter_shardings = parameter_shardings,
        .output_shardings = output_shardings,
        .output_ids = try allocator.dupe(ir.ValueId, &.{.{ .index = 2 }}),
        .instructions = try allocator.dupe(ir.PlanInstruction, &.{.{
            .kind = .add,
            .inputs = try allocator.dupe(ir.ValueId, &.{ .{ .index = 0 }, .{ .index = 1 } }),
            .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 2 }}),
        }}),
    };
    defer plan.deinit();

    var graph = try ExecutableGraph.init(allocator, ctx.executableContext(), &plan);
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
    var ctx = try ExecutableTestContext.init(std.testing.allocator);
    defer ctx.deinit();
    const allocator = std.testing.allocator;

    var plan = ir.ExecutablePlan{
        .allocator = allocator,
        .options = .{
            .num_replicas = 1,
            .num_partitions = 1,
            .device_assignment = try allocator.dupe(i32, &.{0}),
        },
        .values = &.{},
        .parameter_shardings = try allocator.alloc(ir.ShardingPlan, 0),
        .output_shardings = try allocator.alloc(ir.ShardingPlan, 0),
        .output_ids = try allocator.alloc(ir.ValueId, 0),
        .instructions = try allocator.dupe(ir.PlanInstruction, &.{
            .{ .kind = .custom_call },
        }),
    };
    defer plan.deinit();

    try std.testing.expectError(
        error.UnsupportedRuntimeFeature,
        ExecutableGraph.initWithOptions(allocator, ctx.executableContext(), &plan, .{}),
    );
}
