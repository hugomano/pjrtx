const std = @import("std");
const backend_api = @import("backend_selection.zig");
const ir = @import("src/compiler/ir");

const executable_cache = @import("executable_cache.zig");
const executable_residency = @import("executable_residency.zig");
const executable_schedule = @import("executable_schedule.zig");
const execution_context = @import("execution_context.zig");

const ExecutableCacheTrim = executable_cache.Trim;
const ExecutableResidency = executable_residency.Residency;
const ExecutableSchedule = executable_schedule.Schedule;
const ScheduleNode = executable_schedule.Node;

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
    pub const initResident = CompiledExecutableLifecycle.initResident;

    /// Releases backend residency, optimized program text, fingerprint, and plan.
    pub const deinit = CompiledExecutableLifecycle.deinit;

    /// Releases backend residency while keeping executable metadata queryable.
    pub const releaseResidentStorage = CompiledExecutableLifecycle.releaseResidentStorage;

    /// Returns the number of PJRT parameters expected on each execution device.
    pub const parameterCount = CompiledExecutableMetadata.parameterCount;

    /// Returns the number of PJRT outputs produced on each execution device.
    pub const outputCount = CompiledExecutableMetadata.outputCount;

    /// Returns the total bytes required for all outputs on one execution device.
    pub const outputByteSize = CompiledExecutableMetadata.outputByteSize;

    /// Returns the number of per-device execution slots embedded in this executable.
    pub const deviceCount = CompiledExecutableMetadata.deviceCount;

    /// Returns whether backend residency is available for execution.
    pub const hasResidentBackendExecutable = CompiledExecutableMetadata.hasResidentBackendExecutable;

    /// Returns the number of compiler instructions represented by the resident backend program.
    pub const residentProgramInstructionCount = CompiledExecutableMetadata.residentProgramInstructionCount;

    /// Returns whether backend executable acquisition reused a resident cache entry.
    pub const backendExecutableCacheReused = CompiledExecutableMetadata.backendExecutableCacheReused;

    /// Returns the stable PJRT device id assigned to one executable device slot.
    pub const deviceIdAt = CompiledExecutableMetadata.deviceIdAt;

    /// Returns the number of logical devices selected by compile options.
    pub const executableDeviceCount = CompiledExecutableMetadata.executableDeviceCount;

    /// Returns the number of replicas requested by compile options.
    pub const numReplicas = CompiledExecutableMetadata.numReplicas;

    /// Returns the number of partitions requested by compile options.
    pub const numPartitions = CompiledExecutableMetadata.numPartitions;

    /// Returns the stable executable fingerprint string.
    pub const fingerprintText = CompiledExecutableMetadata.fingerprintText;

    /// Returns the optimized program text retained for PJRT metadata queries.
    pub const optimizedProgramText = CompiledExecutableMetadata.optimizedProgramText;

    /// Returns whether executing may consume ownership of a parameter buffer.
    pub const donatesParameter = CompiledExecutableDonation.donatesParameter;

    /// Returns the donated parameter that may alias a given output, if any.
    pub const donatedParameterAliasForOutput = CompiledExecutableDonation.aliasForOutput;

    /// Returns true when a backend output matches the compiler-owned output descriptor.
    pub const backendOutputMatches = CompiledExecutableMetadata.backendOutputMatches;

    /// Returns backend executable statistics with runtime donation aliases included.
    pub const backendExecutableStats = CompiledExecutableResidencyView.backendExecutableStats;

    /// Returns the compile-time executable-cache trimming outcome.
    pub const compileCacheTrim = CompiledExecutableResidencyView.compileCacheTrim;

    /// Returns the most recent execute-time executable-cache trimming outcome.
    pub const executeCacheTrim = CompiledExecutableResidencyView.executeCacheTrim;

    /// Returns the most recent backend completion token observed by execution.
    pub const backendCompletion = CompiledExecutableResidencyView.backendCompletion;

    /// Returns the resident backend handle used by the execution dispatcher.
    pub const backendExecutableForDispatch = CompiledExecutableResidencyView.backendExecutableForDispatch;

    /// Records cache trimming done immediately before output allocation.
    pub const recordExecuteCacheTrim = CompiledExecutableResidencyView.recordExecuteCacheTrim;

    /// Records the latest backend completion token observed by execution.
    pub const recordBackendCompletion = CompiledExecutableResidencyView.recordBackendCompletion;

    /// Adds donation alias counters to this executable's residency statistics.
    pub const recordDonationAlias = CompiledExecutableResidencyView.recordDonationAlias;

    /// Removes donation alias counters when execution setup fails.
    pub const rollbackDonationAlias = CompiledExecutableResidencyView.rollbackDonationAlias;

    /// Narrow test access for executable invariants that are not runtime API surface.
    pub const Testing = CompiledExecutableTesting;
};

const CompiledExecutableLifecycle = struct {
    fn initResident(
        allocator: std.mem.Allocator,
        context: Context,
        plan: *ir.ExecutablePlan,
        optimized_program: []u8,
        fingerprint: []u8,
        cache_hit: bool,
        diagnostics: *std.Io.Writer,
    ) error{ UnsupportedRuntimeFeature, OutOfMemory, Internal }!CompiledExecutable {
        var resident = ResidentExecutable.init(allocator, context, plan, .{ .diagnostic_writer = diagnostics, .cache_fingerprint = fingerprint }) catch |err| switch (err) {
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

    fn deinit(executable: *CompiledExecutable, allocator: std.mem.Allocator) void {
        executable.releaseResidentStorage();
        allocator.free(executable.fingerprint);
        allocator.free(executable.optimized_program);
        executable.plan.deinit();
        allocator.destroy(executable.plan);
        executable.* = undefined;
    }

    fn releaseResidentStorage(executable: *CompiledExecutable) void {
        if (executable.resident_released) return;
        executable.residency.deinit();
        executable.schedule.deinit();
        executable.resident_released = true;
    }
};

const CompiledExecutableMetadata = struct {
    fn parameterCount(executable: *const CompiledExecutable) usize {
        return executable.plan.parameter_shardings.len;
    }

    fn outputCount(executable: *const CompiledExecutable) usize {
        return executable.plan.output_ids.len;
    }

    fn deviceCount(executable: *const CompiledExecutable) usize {
        return executable.schedule.deviceCount();
    }

    fn hasResidentBackendExecutable(executable: *const CompiledExecutable) bool {
        return executable.residency.hasBackendExecutable();
    }

    fn residentProgramInstructionCount(executable: *const CompiledExecutable) usize {
        return executable.residency.programInstructionCount();
    }

    fn backendExecutableCacheReused(executable: *const CompiledExecutable) bool {
        return executable.residency.cacheReused();
    }

    fn deviceIdAt(executable: *const CompiledExecutable, index: usize) ?i32 {
        return executable.schedule.deviceIdAt(index);
    }

    fn executableDeviceCount(executable: *const CompiledExecutable) usize {
        return executable.plan.options.numDevices();
    }

    fn numReplicas(executable: *const CompiledExecutable) i32 {
        return executable.plan.options.num_replicas;
    }

    fn numPartitions(executable: *const CompiledExecutable) i32 {
        return executable.plan.options.num_partitions;
    }

    fn fingerprintText(executable: *const CompiledExecutable) []const u8 {
        return executable.fingerprint;
    }

    fn optimizedProgramText(executable: *const CompiledExecutable) []const u8 {
        return executable.optimized_program;
    }

    fn outputByteSize(executable: *const CompiledExecutable) !usize {
        var total: usize = 0;
        for (executable.plan.output_ids) |value_id| {
            if (value_id.index >= executable.plan.values.len) return error.InvalidGraph;
            const descriptor = executable.plan.values[value_id.index].descriptor;
            total = try std.math.add(usize, total, ir.denseByteSize(descriptor.element_type, descriptor.dims));
        }
        return total;
    }

    fn backendOutputMatches(executable: *const CompiledExecutable, output_index: usize, output: backend_api.ExecutableOutput) bool {
        if (output_index >= executable.plan.output_ids.len) return false;
        const value_id = executable.plan.output_ids[output_index];
        if (value_id.index >= executable.plan.values.len) return false;
        const descriptor = executable.plan.values[value_id.index].descriptor;
        return output.element_type == descriptor.element_type and std.mem.eql(i64, output.dims, descriptor.dims) and output.byte_size == ir.denseByteSize(descriptor.element_type, descriptor.dims);
    }

};

const CompiledExecutableResidencyView = struct {
    fn backendExecutableStats(executable: *const CompiledExecutable) ?backend_api.ExecutableStats {
        return executable.residency.backendExecutableStats();
    }

    fn compileCacheTrim(executable: *const CompiledExecutable) ExecutableCacheTrim {
        return executable.residency.compileCacheTrim();
    }

    fn executeCacheTrim(executable: *const CompiledExecutable) ExecutableCacheTrim {
        return executable.residency.executeCacheTrim();
    }

    fn backendCompletion(executable: *const CompiledExecutable) backend_api.ExecutionCompletion {
        return executable.residency.backendCompletion();
    }

    fn backendExecutableForDispatch(executable: *const CompiledExecutable) ?backend_api.ExecutableHandle {
        return executable.residency.backendExecutableForDispatch();
    }

    fn recordExecuteCacheTrim(executable: *CompiledExecutable, trim: ExecutableCacheTrim) void {
        executable.residency.recordExecuteCacheTrim(trim);
    }

    fn recordBackendCompletion(executable: *CompiledExecutable, completion: backend_api.ExecutionCompletion) void {
        executable.residency.recordBackendCompletion(completion);
    }

    fn recordDonationAlias(executable: *CompiledExecutable, bytes: usize) void {
        executable.residency.recordDonationAlias(bytes);
    }

    fn rollbackDonationAlias(executable: *CompiledExecutable, output_count: usize, output_bytes: usize) void {
        executable.residency.rollbackDonationAlias(output_count, output_bytes);
    }
};

const CompiledExecutableDonation = struct {
    fn donatesParameter(executable: *const CompiledExecutable, parameter_index: usize) bool {
        for (executable.plan.donated_parameter_indices) |candidate| if (candidate == parameter_index) return true;
        return false;
    }

    fn aliasForOutput(executable: *const CompiledExecutable, output_index: usize) ?OutputAlias {
        for (executable.plan.output_aliases) |alias| {
            if (alias.output_index == output_index and donatesParameter(executable, alias.parameter_index)) {
                return .{ .parameter_index = alias.parameter_index, .kind = alias.kind };
            }
        }
        if (output_index >= executable.plan.output_ids.len) return null;
        const output_id = executable.plan.output_ids[output_index];
        var parameter_index: usize = 0;
        for (executable.plan.values) |value| {
            if (value.role != .parameter) continue;
            if (value.id.index == output_id.index) {
                return if (donatesParameter(executable, parameter_index)) .{ .parameter_index = parameter_index, .kind = .identity } else null;
            }
            parameter_index += 1;
        }
        return null;
    }
};

const CompiledExecutableTesting = struct {
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
            .fingerprint = @constCast(options.cache_fingerprint orelse &.{}),
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
