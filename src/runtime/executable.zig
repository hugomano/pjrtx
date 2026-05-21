const std = @import("std");
const backend_api = @import("src/backend/mlx_metal");
const ir = @import("src/compiler/ir");

const device_memory = @import("device_memory.zig");
const executable_cache = @import("executable_cache.zig");
const executable_residency = @import("executable_residency.zig");
const executable_schedule = @import("executable_schedule.zig");
const execution_context = @import("execution_context.zig");

const DeviceMemoryTopology = device_memory.DeviceMemoryTopology;
const ExecutableCacheTrim = executable_cache.Trim;
const ExecutableResidency = executable_residency.Residency;
const ExecutableSchedule = executable_schedule.Schedule;
const ScheduleNode = executable_schedule.Node;
const ScheduleNodeKind = executable_schedule.NodeKind;

/// Error set for acquiring a cached or freshly compiled backend executable.
pub const CacheAcquireError = execution_context.CacheAcquireError;

/// Carries optional backend diagnostics and cache identity into residency creation.
pub const BackendCompileOptions = executable_residency.CompileOptions;

/// Narrow runtime surface needed by executable residency and execution.
pub const Context = execution_context.Context;

/// Describes a donated parameter that may back an output buffer.
pub const OutputAlias = struct {
    parameter_index: usize,
    kind: ir.OutputAliasKind,
};

/// Owns the compiler plan and resident backend executable returned by compilation.
pub const CompiledExecutable = struct {
    plan: *ir.ExecutablePlan,
    schedule: ExecutableSchedule,
    residency: ExecutableResidency,
    optimized_program: []u8,
    fingerprint: []u8,
    cache_hit: bool,
    backend_stats: backend_api.ExecutableStats,
    resident_released: bool = false,

    /// Builds a compiled executable from owned compile artifacts and acquires backend residency.
    pub fn initResident(
        allocator: std.mem.Allocator,
        context: Context,
        plan: *ir.ExecutablePlan,
        optimized_program: []u8,
        fingerprint: []u8,
        cache_hit: bool,
        diagnostics: *std.Io.Writer,
    ) error{ UnsupportedRuntimeFeature, OutOfMemory, Internal }!CompiledExecutable {
        var resident = ResidentExecutable.init(allocator, context, plan, .{
            .diagnostic_writer = diagnostics,
            .cache_fingerprint = fingerprint,
        }) catch |err| switch (err) {
            error.UnsupportedRuntimeFeature => return error.UnsupportedRuntimeFeature,
            error.OutOfMemory => return error.OutOfMemory,
            else => return error.Internal,
        };
        errdefer resident.deinit();

        return .{
            .plan = plan,
            .schedule = resident.schedule,
            .residency = resident.residency,
            .optimized_program = optimized_program,
            .fingerprint = fingerprint,
            .cache_hit = cache_hit,
            .backend_stats = resident.residency.backendExecutableStats() orelse .{},
        };
    }

    /// Releases backend residency, optimized program text, fingerprint, and plan.
    pub fn deinit(self: *CompiledExecutable, allocator: std.mem.Allocator) void {
        self.releaseResidentStorage();
        allocator.free(self.fingerprint);
        allocator.free(self.optimized_program);
        self.plan.deinit();
        allocator.destroy(self.plan);
        self.* = undefined;
    }

    /// Releases backend residency while keeping executable metadata queryable.
    pub fn releaseResidentStorage(self: *CompiledExecutable) void {
        if (self.resident_released) return;
        self.residency.deinit();
        self.schedule.deinit();
        self.resident_released = true;
    }

    /// Returns the number of PJRT parameters expected on each execution device.
    pub fn parameterCount(self: *const CompiledExecutable) usize {
        return self.plan.parameter_shardings.len;
    }

    /// Returns the number of PJRT outputs produced on each execution device.
    pub fn outputCount(self: *const CompiledExecutable) usize {
        return self.plan.output_ids.len;
    }

    /// Returns the total bytes required for all outputs on one execution device.
    pub fn outputByteSize(self: *const CompiledExecutable) !usize {
        var total: usize = 0;
        for (self.plan.output_ids) |value_id| {
            if (value_id.index >= self.plan.values.len) return error.InvalidGraph;
            const descriptor = self.plan.values[value_id.index].descriptor;
            total = try std.math.add(usize, total, ir.denseByteSize(descriptor.element_type, descriptor.dims));
        }
        return total;
    }

    /// Returns the number of per-device execution slots embedded in this executable.
    pub fn deviceCount(self: *const CompiledExecutable) usize {
        return self.schedule.deviceCount();
    }

    /// Returns whether backend residency is available for execution.
    pub fn hasResidentBackendExecutable(self: *const CompiledExecutable) bool {
        return self.residency.hasBackendExecutable();
    }

    /// Returns the number of compiler instructions represented by the resident backend program.
    pub fn residentProgramInstructionCount(self: *const CompiledExecutable) usize {
        return self.residency.programInstructionCount();
    }

    /// Returns whether backend executable acquisition reused a resident cache entry.
    pub fn backendExecutableCacheReused(self: *const CompiledExecutable) bool {
        return self.residency.cacheReused();
    }

    /// Returns the stable PJRT device id assigned to one executable device slot.
    pub fn deviceIdAt(self: *const CompiledExecutable, index: usize) ?i32 {
        return self.schedule.deviceIdAt(index);
    }

    /// Returns the number of logical devices selected by compile options.
    pub fn executableDeviceCount(self: *const CompiledExecutable) usize {
        return self.plan.options.numDevices();
    }

    /// Returns the number of replicas requested by compile options.
    pub fn numReplicas(self: *const CompiledExecutable) i32 {
        return self.plan.options.num_replicas;
    }

    /// Returns the number of partitions requested by compile options.
    pub fn numPartitions(self: *const CompiledExecutable) i32 {
        return self.plan.options.num_partitions;
    }

    /// Returns the stable executable fingerprint string.
    pub fn fingerprintText(self: *const CompiledExecutable) []const u8 {
        return self.fingerprint;
    }

    /// Returns the optimized program text retained for PJRT metadata queries.
    pub fn optimizedProgramText(self: *const CompiledExecutable) []const u8 {
        return self.optimized_program;
    }

    /// Returns whether executing may consume ownership of a parameter buffer.
    pub fn donatesParameter(self: *const CompiledExecutable, parameter_index: usize) bool {
        for (self.plan.donated_parameter_indices) |candidate| {
            if (candidate == parameter_index) return true;
        }
        return false;
    }

    /// Returns the donated parameter that may alias a given output, if any.
    pub fn donatedParameterAliasForOutput(self: *const CompiledExecutable, output_index: usize) ?OutputAlias {
        for (self.plan.output_aliases) |alias| {
            if (alias.output_index == output_index and self.donatesParameter(alias.parameter_index)) {
                return .{ .parameter_index = alias.parameter_index, .kind = alias.kind };
            }
        }
        if (output_index >= self.plan.output_ids.len) return null;
        const output_id = self.plan.output_ids[output_index];
        var parameter_index: usize = 0;
        for (self.plan.values) |value| {
            if (value.role != .parameter) continue;
            if (value.id.index == output_id.index) {
                return if (self.donatesParameter(parameter_index)) .{ .parameter_index = parameter_index, .kind = .identity } else null;
            }
            parameter_index += 1;
        }
        return null;
    }

    /// Returns true when a backend output matches the compiler-owned output descriptor.
    pub fn backendOutputMatches(self: *const CompiledExecutable, output_index: usize, output: backend_api.ExecutableOutput) bool {
        if (output_index >= self.plan.output_ids.len) return false;
        const value_id = self.plan.output_ids[output_index];
        if (value_id.index >= self.plan.values.len) return false;
        const descriptor = self.plan.values[value_id.index].descriptor;
        if (output.element_type != descriptor.element_type) return false;
        if (!std.mem.eql(i64, output.dims, descriptor.dims)) return false;
        if (output.byte_size != ir.denseByteSize(descriptor.element_type, descriptor.dims)) return false;
        return true;
    }

    /// Returns backend executable statistics with runtime donation aliases included.
    pub fn backendExecutableStats(self: *const CompiledExecutable) ?backend_api.ExecutableStats {
        return self.residency.backendExecutableStats();
    }

    /// Returns the compile-time executable-cache trimming outcome.
    pub fn compileCacheTrim(self: *const CompiledExecutable) ExecutableCacheTrim {
        return self.residency.compileCacheTrim();
    }

    /// Returns the most recent execute-time executable-cache trimming outcome.
    pub fn executeCacheTrim(self: *const CompiledExecutable) ExecutableCacheTrim {
        return self.residency.executeCacheTrim();
    }

    /// Returns the most recent backend completion token observed by execution.
    pub fn backendCompletion(self: *const CompiledExecutable) backend_api.ExecutionCompletion {
        return self.residency.backendCompletion();
    }

    /// Returns the resident backend handle used by the execution dispatcher.
    pub fn backendExecutableForDispatch(self: *const CompiledExecutable) ?backend_api.ExecutableHandle {
        return self.residency.backendExecutableForDispatch();
    }

    /// Records cache trimming done immediately before output allocation.
    pub fn recordExecuteCacheTrim(self: *CompiledExecutable, trim: ExecutableCacheTrim) void {
        self.residency.recordExecuteCacheTrim(trim);
    }

    /// Records the latest backend completion token observed by execution.
    pub fn recordBackendCompletion(self: *CompiledExecutable, completion: backend_api.ExecutionCompletion) void {
        self.residency.recordBackendCompletion(completion);
    }

    /// Adds donation alias counters to this executable's residency statistics.
    pub fn recordDonationAlias(self: *CompiledExecutable, bytes: usize) void {
        self.residency.recordDonationAlias(bytes);
    }

    /// Removes donation alias counters when execution setup fails.
    pub fn rollbackDonationAlias(self: *CompiledExecutable, output_count: usize, output_bytes: usize) void {
        self.residency.rollbackDonationAlias(output_count, output_bytes);
    }

    /// Narrow test access for executable invariants that are not runtime API surface.
    pub const Testing = struct {
        /// Overrides donated parameters for focused runtime/PJRT ABI tests.
        pub fn setDonatedParameters(executable: *CompiledExecutable, donated_parameter_indices: []u32) void {
            executable.plan.donated_parameter_indices = donated_parameter_indices;
        }

        /// Builds a compiled executable around a caller-owned plan for focused runtime tests.
        pub fn initBorrowedResident(allocator: std.mem.Allocator, context: Context, plan: *ir.ExecutablePlan, options: BackendCompileOptions) !CompiledExecutable {
            var resident = try ResidentExecutable.init(allocator, context, plan, options);
            errdefer resident.deinit();
            return .{
                .plan = plan,
                .schedule = resident.schedule,
                .residency = resident.residency,
                .optimized_program = &.{},
                .fingerprint = options.cache_fingerprint orelse &.{},
                .cache_hit = false,
                .backend_stats = resident.residency.backendExecutableStats() orelse .{},
            };
        }

        /// Releases residency for a compiled executable that borrowed its plan and metadata.
        pub fn deinitBorrowedResident(executable: *CompiledExecutable) void {
            executable.releaseResidentStorage();
            executable.* = undefined;
        }

        /// Returns the number of scheduled instruction/device nodes.
        pub fn scheduledNodeCount(executable: *const CompiledExecutable) usize {
            return executable.schedule.nodeCount();
        }

        /// Returns one scheduled instruction/device node.
        pub fn scheduledNodeAt(executable: *const CompiledExecutable, index: usize) ?ScheduleNode {
            return executable.schedule.nodeAt(index);
        }
    };
};

const ResidentExecutable = struct {
    schedule: ExecutableSchedule,
    residency: ExecutableResidency,

    fn init(allocator: std.mem.Allocator, context: Context, plan: *const ir.ExecutablePlan, options: BackendCompileOptions) !ResidentExecutable {
        var schedule = try ExecutableSchedule.init(allocator, context, plan);
        errdefer schedule.deinit();

        var residency = try ExecutableResidency.init(context, plan, &schedule, options);
        errdefer residency.deinit();

        return .{
            .schedule = schedule,
            .residency = residency,
        };
    }

    fn deinit(self: *ResidentExecutable) void {
        self.residency.deinit();
        self.schedule.deinit();
        self.* = undefined;
    }
};

const ExecutableTestContext = struct {
    allocator: std.mem.Allocator,
    backend: backend_api.Backend,
    descriptors: []ir.DeviceDescriptor,
    topology: DeviceMemoryTopology,

    fn init(allocator: std.mem.Allocator) !ExecutableTestContext {
        const backend = backend_api.create();
        const descriptors = try backend.enumerateDevices(allocator, 1);
        errdefer backend.releaseDeviceDescriptors(allocator, descriptors);
        if (descriptors.len == 0) return error.InvalidDeviceCount;

        const topology = try DeviceMemoryTopology.initFromDescriptors(allocator, descriptors);
        errdefer topology.deinit(allocator);

        return .{
            .allocator = allocator,
            .backend = backend,
            .descriptors = descriptors,
            .topology = topology,
        };
    }

    fn deinit(self: *ExecutableTestContext) void {
        self.topology.deinit(self.allocator);
        self.backend.releaseDeviceDescriptors(self.allocator, self.descriptors);
        self.* = undefined;
    }

    fn executableContext(self: *ExecutableTestContext) Context {
        return .{
            .backend = self.backend,
            .devices = self.topology.deviceSlice(),
        };
    }
};

const ExecutableTestSupport = struct {
    fn shardingPlan(allocator: std.mem.Allocator, assignment: []const i32) !ir.ShardingPlan {
        return .{
            .kind = .replicated,
            .mesh_name = try allocator.dupe(u8, ""),
            .device_assignment = try allocator.dupe(i32, assignment),
        };
    }
};

test "compiled executable materializes per-device scheduled nodes" {
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
    parameter_shardings[0] = try ExecutableTestSupport.shardingPlan(allocator, &assignment);
    parameter_shardings[1] = try ExecutableTestSupport.shardingPlan(allocator, &assignment);
    var output_shardings = try allocator.alloc(ir.ShardingPlan, 1);
    output_shardings[0] = try ExecutableTestSupport.shardingPlan(allocator, &assignment);

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

    var executable = try CompiledExecutable.Testing.initBorrowedResident(allocator, ctx.executableContext(), &plan, .{});
    defer CompiledExecutable.Testing.deinitBorrowedResident(&executable);

    try std.testing.expectEqual(@as(i32, 0), executable.deviceIdAt(0).?);
    try std.testing.expectEqual(@as(usize, 1), CompiledExecutable.Testing.scheduledNodeCount(&executable));
    const node = CompiledExecutable.Testing.scheduledNodeAt(&executable, 0).?;
    try std.testing.expectEqual(ScheduleNodeKind.compute, node.kind);
    try std.testing.expectEqual(@as(usize, 0), node.device_index);
    try std.testing.expectEqual(@as(i32, 0), node.device_id);
    try std.testing.expect(executable.hasResidentBackendExecutable());
    try std.testing.expectEqual(@as(usize, 1), executable.residentProgramInstructionCount());
}

test "compiled executable rejects unsupported backend executable acquisition" {
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
        CompiledExecutable.Testing.initBorrowedResident(allocator, ctx.executableContext(), &plan, .{}),
    );
}
