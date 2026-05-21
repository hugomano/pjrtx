const std = @import("std");
const backend_api = @import("src/backend/mlx_metal");
const ir = @import("src/compiler/ir");

const executable_cache = @import("executable_cache.zig");
const executable_schedule = @import("executable_schedule.zig");
const execution_context = @import("execution_context.zig");

const Context = execution_context.Context;
const ExecutableCacheLease = executable_cache.Lease;
const ExecutableCacheTrim = executable_cache.Trim;
const Schedule = executable_schedule.Schedule;

const DonationAliasStats = struct {
    output_count: usize = 0,
    output_bytes: usize = 0,
};

const BackendState = struct {
    resident: bool,
    cache_reused: bool = false,
    program_instruction_count: usize,
};

/// Carries optional backend diagnostics and cache identity into residency creation.
pub const CompileOptions = struct {
    diagnostic_writer: ?*std.Io.Writer = null,
    cache_fingerprint: ?[]const u8 = null,
};

/// Owns backend executable residency, cache lease, completion, and residency stats.
pub const Residency = struct {
    backend: backend_api.Backend,
    context: Context,
    backend_executable: ?backend_api.ExecutableHandle = null,
    backend_executable_cache_lease: ?ExecutableCacheLease = null,
    backend_state: BackendState,
    last_compile_cache_trim: ExecutableCacheTrim = .{},
    last_execute_cache_trim: ExecutableCacheTrim = .{},
    last_backend_completion: backend_api.ExecutionCompletion = .{},
    donation_alias_stats: DonationAliasStats = .{},

    /// Acquires backend executable residency for a scheduled executable plan.
    pub fn init(context: Context, plan: *const ir.ExecutablePlan, schedule: *const Schedule, options: CompileOptions) !Residency {
        var backend_executable_cache_lease: ?ExecutableCacheLease = null;
        var cache_reused = false;
        var last_compile_cache_trim = ExecutableCacheTrim{};
        const backend_executable = if (options.cache_fingerprint) |fingerprint| blk: {
            const cached = try context.acquireCachedExecutable(schedule.allocator, fingerprint, plan, schedule.device_local_hardware_ids);
            if (cached) |entry| {
                backend_executable_cache_lease = entry.lease;
                cache_reused = entry.reused;
                last_compile_cache_trim = entry.compile_trim;
                break :blk entry.handle;
            }
            break :blk null;
        } else context.compileBackendExecutable(schedule.allocator, plan, schedule.device_local_hardware_ids) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => null,
        };
        errdefer if (backend_executable) |handle| {
            if (backend_executable_cache_lease) |lease| context.releaseCachedExecutable(lease) else context.destroyBackendExecutable(handle);
        };
        if (backend_executable == null) {
            if (options.diagnostic_writer) |writer| {
                context.writeBackendLoweringDiagnostic(plan, schedule.device_local_hardware_ids, writer);
            }
            return error.UnsupportedRuntimeFeature;
        }

        return .{
            .backend = context.backendForExecutableResidency(),
            .context = context,
            .backend_executable = backend_executable,
            .backend_executable_cache_lease = backend_executable_cache_lease,
            .backend_state = .{
                .resident = true,
                .cache_reused = cache_reused,
                .program_instruction_count = plan.instructions.len,
            },
            .last_compile_cache_trim = last_compile_cache_trim,
        };
    }

    /// Releases backend executable residency or the cache lease retaining it.
    pub fn deinit(self: *Residency) void {
        if (self.backend_executable) |handle| {
            if (self.backend_executable_cache_lease) |lease| {
                self.context.releaseCachedExecutable(lease);
            } else {
                self.backend.destroyExecutable(handle);
            }
        }
        self.* = undefined;
    }

    /// Returns whether backend executable residency is currently available.
    pub fn hasBackendExecutable(self: *const Residency) bool {
        return self.backend_executable != null;
    }

    /// Returns whether backend executable acquisition reused a resident cache entry.
    pub fn cacheReused(self: *const Residency) bool {
        return self.backend_state.cache_reused;
    }

    /// Returns the number of compiler instructions represented by the backend program.
    pub fn programInstructionCount(self: *const Residency) usize {
        return self.backend_state.program_instruction_count;
    }

    /// Returns the compile-time executable-cache trimming outcome.
    pub fn compileCacheTrim(self: *const Residency) ExecutableCacheTrim {
        return self.last_compile_cache_trim;
    }

    /// Returns the execute-time executable-cache trimming outcome.
    pub fn executeCacheTrim(self: *const Residency) ExecutableCacheTrim {
        return self.last_execute_cache_trim;
    }

    /// Returns the most recent backend completion token observed by execution.
    pub fn backendCompletion(self: *const Residency) backend_api.ExecutionCompletion {
        return self.last_backend_completion;
    }

    /// Returns backend residency statistics with runtime donation aliases included.
    pub fn backendExecutableStats(self: *const Residency) ?backend_api.ExecutableStats {
        const handle = self.backend_executable orelse return null;
        var stats = self.backend.executableStats(handle);
        stats.donation_alias_output_count += self.donation_alias_stats.output_count;
        stats.donation_alias_output_bytes += self.donation_alias_stats.output_bytes;
        return stats;
    }

    /// Returns the backend handle used by the execution dispatcher.
    pub fn backendExecutableForDispatch(self: *const Residency) ?backend_api.ExecutableHandle {
        return self.backend_executable;
    }

    /// Records cache trimming done immediately before output allocation.
    pub fn recordExecuteCacheTrim(self: *Residency, trim: ExecutableCacheTrim) void {
        self.last_execute_cache_trim = trim;
    }

    /// Records the latest backend completion token.
    pub fn recordBackendCompletion(self: *Residency, completion: backend_api.ExecutionCompletion) void {
        self.last_backend_completion = completion;
    }

    /// Adds donation alias counters to residency statistics.
    pub fn recordDonationAlias(self: *Residency, bytes: usize) void {
        self.donation_alias_stats.output_count += 1;
        self.donation_alias_stats.output_bytes += bytes;
    }

    /// Removes donation alias counters when execution setup fails.
    pub fn rollbackDonationAlias(self: *Residency, output_count: usize, output_bytes: usize) void {
        self.donation_alias_stats.output_count -|= output_count;
        self.donation_alias_stats.output_bytes -|= output_bytes;
    }
};
