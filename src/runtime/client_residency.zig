const std = @import("std");
const backend_api = @import("src/backend/mlx_metal");
const ir = @import("src/compiler/ir");

const client_mod = @import("client.zig");
const executable_cache = @import("executable_cache.zig");
const executable_mod = @import("executable.zig");

const CachedBackendExecutable = executable_cache.Retained;
const Client = client_mod.Client;
const CompiledExecutable = executable_mod.CompiledExecutable;
const ExecutableCacheLease = executable_cache.Lease;
const Memory = @import("device_memory.zig").Memory;

/// Records a compile request and returns whether the fingerprint was seen before.
pub fn recordCompile(client: *Client, fingerprint: []const u8) !bool {
    return client.executable_residency.recordCompile(client.io, client.device_memory.memorySlice(), fingerprint);
}

/// Returns executable-cache counters without exposing cache entries.
pub fn statsSnapshot(client: *const Client) executable_cache.Stats {
    return client.executable_residency.statsSnapshot();
}

/// Returns a snapshot of one executable-cache entry without exposing mutable cache storage.
pub fn entrySnapshot(client: *const Client, fingerprint: []const u8) ?executable_cache.EntrySnapshot {
    return client.executable_residency.entrySnapshot(fingerprint);
}

/// Overrides compile latency for one cache entry in focused cache-policy tests.
pub fn setCompileLatencyForTest(client: *Client, fingerprint: []const u8, latency_us: u64) void {
    client.executable_residency.setCompileLatencyForTest(fingerprint, latency_us);
}

/// Sets the resident executable-cache budget and evicts idle entries as needed.
pub fn setMaxResidentBytes(client: *Client, max_resident_bytes: u64) void {
    client.executable_residency.setMaxResidentBytes(client.io, client.backend, client.device_memory.memorySlice(), max_resident_bytes);
}

/// Trims idle resident executables to make room for a device allocation.
pub fn trimForAllocation(client: *Client, memory: *Memory, allocation_bytes: usize) executable_cache.Trim {
    return client.executable_residency.trimForAllocation(client.io, client.backend, client.device_memory.memorySlice(), memory, allocation_bytes);
}

/// Acquires a retained resident backend executable for a compiled executable.
pub fn acquireCachedBackendExecutable(
    client: *Client,
    allocator: std.mem.Allocator,
    fingerprint: []const u8,
    plan: *const ir.ExecutablePlan,
    device_local_hardware_ids: []const i32,
) !?CachedBackendExecutable {
    return client.executable_residency.acquireBackendExecutable(
        client.io,
        client.backend,
        allocator,
        client.device_memory.memorySlice(),
        client.device_memory.executableResidencyMemory(device_local_hardware_ids),
        fingerprint,
        plan,
        device_local_hardware_ids,
    );
}

/// Releases a retained executable-cache entry after residency teardown.
pub fn releaseCachedBackendExecutable(client: *Client, lease: ExecutableCacheLease) void {
    client.executable_residency.release(client.io, client.backend, client.device_memory.memorySlice(), lease);
}

const ClientResidencyTestSupport = struct {
    fn shardingPlan(allocator: std.mem.Allocator, assignment: []const i32) !ir.ShardingPlan {
        return .{
            .kind = .replicated,
            .mesh_name = try allocator.dupe(u8, ""),
            .device_assignment = try allocator.dupe(i32, assignment),
        };
    }

    fn constantU8ExecutablePlan(allocator: std.mem.Allocator, assignment: []const i32, literal: []const u8) !ir.ExecutablePlan {
        var values = try allocator.alloc(ir.Value, 1);
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

        var output_shardings = try allocator.alloc(ir.ShardingPlan, 1);
        errdefer allocator.free(output_shardings);
        output_shardings[0] = try shardingPlan(allocator, assignment);
        errdefer {
            allocator.free(output_shardings[0].mesh_name);
            allocator.free(output_shardings[0].device_assignment);
        }

        const output_ids = try allocator.dupe(ir.ValueId, &.{.{ .index = 0 }});
        errdefer allocator.free(output_ids);

        const literal_copy = try allocator.dupe(u8, literal);
        errdefer allocator.free(literal_copy);

        const instruction_outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 0 }});
        errdefer allocator.free(instruction_outputs);

        const instructions = try allocator.dupe(ir.PlanInstruction, &.{.{
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
            .parameter_shardings = try allocator.alloc(ir.ShardingPlan, 0),
            .output_shardings = output_shardings,
            .output_ids = output_ids,
            .instructions = instructions,
        };
    }

    fn addU8ExecutablePlan(allocator: std.mem.Allocator, assignment: []const i32, dims: []const i64) !ir.ExecutablePlan {
        const values = try allocator.alloc(ir.Value, 3);
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

        var parameter_shardings = try allocator.alloc(ir.ShardingPlan, 2);
        errdefer allocator.free(parameter_shardings);
        parameter_shardings[0] = try shardingPlan(allocator, assignment);
        errdefer {
            allocator.free(parameter_shardings[0].mesh_name);
            allocator.free(parameter_shardings[0].device_assignment);
        }
        parameter_shardings[1] = try shardingPlan(allocator, assignment);
        errdefer {
            allocator.free(parameter_shardings[1].mesh_name);
            allocator.free(parameter_shardings[1].device_assignment);
        }

        var output_shardings = try allocator.alloc(ir.ShardingPlan, 1);
        errdefer allocator.free(output_shardings);
        output_shardings[0] = try shardingPlan(allocator, assignment);
        errdefer {
            allocator.free(output_shardings[0].mesh_name);
            allocator.free(output_shardings[0].device_assignment);
        }

        const output_ids = try allocator.dupe(ir.ValueId, &.{.{ .index = 2 }});
        errdefer allocator.free(output_ids);

        const inputs = try allocator.dupe(ir.ValueId, &.{ .{ .index = 0 }, .{ .index = 1 } });
        errdefer allocator.free(inputs);

        const outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 2 }});
        errdefer allocator.free(outputs);

        const instructions = try allocator.dupe(ir.PlanInstruction, &.{.{
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

    fn cacheEntrySnapshot(client: *const Client, fingerprint: []const u8) !executable_cache.EntrySnapshot {
        return entrySnapshot(client, fingerprint) orelse error.TestUnexpectedResult;
    }
};

test "client executable cache reuses backend executable handles" {
    const client = try Client.init(std.testing.allocator, backend_api.create(), 1);
    defer client.deinit();
    const allocator = std.testing.allocator;
    const assignment = [_]i32{0};
    const dims = [_]i64{4};

    var plan = try ClientResidencyTestSupport.addU8ExecutablePlan(allocator, &assignment, &dims);
    defer plan.deinit();

    const fingerprint = "cached-add";
    try std.testing.expect(!try recordCompile(client, fingerprint));
    var first = try CompiledExecutable.Testing.initBorrowedResident(allocator, client.executableContext(), &plan, .{
        .cache_fingerprint = fingerprint,
    });
    try std.testing.expect(first.hasResidentBackendExecutable());
    try std.testing.expect(!first.backendExecutableCacheReused());

    var entry = try ClientResidencyTestSupport.cacheEntrySnapshot(client, fingerprint);
    try std.testing.expect(entry.resident);
    try std.testing.expectEqual(@as(usize, 1), entry.reference_count);

    try std.testing.expect(try recordCompile(client, fingerprint));
    var second = try CompiledExecutable.Testing.initBorrowedResident(allocator, client.executableContext(), &plan, .{
        .cache_fingerprint = fingerprint,
    });
    try std.testing.expect(second.hasResidentBackendExecutable());
    try std.testing.expect(second.backendExecutableCacheReused());
    try std.testing.expectEqual(first.backendExecutableForDispatch().?, second.backendExecutableForDispatch().?);
    try std.testing.expectEqual(@as(u64, 1), client.executableCacheStats().misses);
    try std.testing.expectEqual(@as(u64, 1), client.executableCacheStats().hits);
    entry = try ClientResidencyTestSupport.cacheEntrySnapshot(client, fingerprint);
    try std.testing.expectEqual(@as(usize, 2), entry.reference_count);

    CompiledExecutable.Testing.deinitBorrowedResident(&second);
    entry = try ClientResidencyTestSupport.cacheEntrySnapshot(client, fingerprint);
    try std.testing.expectEqual(@as(usize, 1), entry.reference_count);
    CompiledExecutable.Testing.deinitBorrowedResident(&first);
    entry = try ClientResidencyTestSupport.cacheEntrySnapshot(client, fingerprint);
    try std.testing.expectEqual(@as(usize, 0), entry.reference_count);
    try std.testing.expect(entry.resident);
}

test "executable cache evicts idle resident backend executables under byte limit" {
    const client = try Client.init(std.testing.allocator, backend_api.create(), 1);
    defer client.deinit();
    const allocator = std.testing.allocator;
    const assignment = [_]i32{0};
    const literal = [_]u8{ 1, 2, 3, 4 };

    var plan = try ClientResidencyTestSupport.constantU8ExecutablePlan(allocator, &assignment, &literal);
    defer plan.deinit();

    client.setExecutableCacheMaxResidentBytes(0);

    const fingerprint = "evict-idle-constant";
    try std.testing.expect(!try recordCompile(client, fingerprint));
    var first = try CompiledExecutable.Testing.initBorrowedResident(allocator, client.executableContext(), &plan, .{
        .cache_fingerprint = fingerprint,
    });
    var entry = try ClientResidencyTestSupport.cacheEntrySnapshot(client, fingerprint);
    try std.testing.expect(first.hasResidentBackendExecutable());
    try std.testing.expect(entry.resident);
    try std.testing.expect(entry.resident_bytes >= literal.len);
    try std.testing.expectEqual(@as(u64, 1), client.executableCacheStats().resident_entries);
    try std.testing.expect(client.executableCacheStats().resident_bytes >= literal.len);
    try std.testing.expect(client.defaultMemory().stats.totalBytesInUse() >= literal.len);
    try std.testing.expect(client.defaultMemory().stats.peakTotalBytesInUse() >= client.defaultMemory().stats.totalBytesInUse());
    try std.testing.expectEqual(@as(usize, 1), entry.reference_count);

    CompiledExecutable.Testing.deinitBorrowedResident(&first);
    entry = try ClientResidencyTestSupport.cacheEntrySnapshot(client, fingerprint);
    try std.testing.expectEqual(@as(usize, 0), entry.reference_count);
    try std.testing.expect(!entry.resident);
    try std.testing.expectEqual(@as(u64, 0), client.executableCacheStats().resident_entries);
    try std.testing.expectEqual(@as(u64, 0), client.executableCacheStats().resident_bytes);
    try std.testing.expectEqual(@as(u64, 0), client.defaultMemory().stats.totalBytesInUse());
    try std.testing.expectEqual(@as(u64, 1), client.executableCacheStats().evictions);
    try std.testing.expectEqual(@as(u64, 1), client.defaultMemory().stats.executable_cache_evictions);

    try std.testing.expect(try recordCompile(client, fingerprint));
    var second = try CompiledExecutable.Testing.initBorrowedResident(allocator, client.executableContext(), &plan, .{
        .cache_fingerprint = fingerprint,
    });
    defer CompiledExecutable.Testing.deinitBorrowedResident(&second);
    try std.testing.expect(second.hasResidentBackendExecutable());
    try std.testing.expect(!second.backendExecutableCacheReused());
    entry = try ClientResidencyTestSupport.cacheEntrySnapshot(client, fingerprint);
    try std.testing.expectEqual(@as(usize, 1), entry.reference_count);
    try std.testing.expect(entry.resident);
    try std.testing.expectEqual(@as(u64, 1), client.executableCacheStats().hits);
}

test "executable cache evicts largest idle resident executable before older smaller entries" {
    const client = try Client.init(std.testing.allocator, backend_api.create(), 1);
    defer client.deinit();
    const allocator = std.testing.allocator;
    const assignment = [_]i32{0};

    var small_plan = try ClientResidencyTestSupport.constantU8ExecutablePlan(allocator, &assignment, &.{ 1, 2, 3, 4 });
    defer small_plan.deinit();
    var large_plan = try ClientResidencyTestSupport.constantU8ExecutablePlan(allocator, &assignment, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer large_plan.deinit();

    try std.testing.expect(!try recordCompile(client, "small-constant"));
    var small_executable = try CompiledExecutable.Testing.initBorrowedResident(allocator, client.executableContext(), &small_plan, .{
        .cache_fingerprint = "small-constant",
    });
    var small_entry = try ClientResidencyTestSupport.cacheEntrySnapshot(client, "small-constant");
    const small_resident_bytes = small_entry.resident_bytes;
    try std.testing.expect(small_resident_bytes >= 4);
    CompiledExecutable.Testing.deinitBorrowedResident(&small_executable);
    small_entry = try ClientResidencyTestSupport.cacheEntrySnapshot(client, "small-constant");
    try std.testing.expectEqual(@as(usize, 0), small_entry.reference_count);
    try std.testing.expect(small_entry.resident);

    try std.testing.expect(!try recordCompile(client, "large-constant"));
    var large_executable = try CompiledExecutable.Testing.initBorrowedResident(allocator, client.executableContext(), &large_plan, .{
        .cache_fingerprint = "large-constant",
    });
    var large_entry = try ClientResidencyTestSupport.cacheEntrySnapshot(client, "large-constant");
    const large_resident_bytes = large_entry.resident_bytes;
    try std.testing.expect(large_resident_bytes > small_resident_bytes);
    CompiledExecutable.Testing.deinitBorrowedResident(&large_executable);
    large_entry = try ClientResidencyTestSupport.cacheEntrySnapshot(client, "large-constant");
    try std.testing.expectEqual(@as(usize, 0), large_entry.reference_count);
    try std.testing.expect(large_entry.resident);

    client.setExecutableCacheMaxResidentBytes(small_resident_bytes);
    small_entry = try ClientResidencyTestSupport.cacheEntrySnapshot(client, "small-constant");
    large_entry = try ClientResidencyTestSupport.cacheEntrySnapshot(client, "large-constant");
    try std.testing.expect(small_entry.resident);
    try std.testing.expect(!large_entry.resident);
    try std.testing.expectEqual(@as(u64, 1), client.executableCacheStats().resident_entries);
    try std.testing.expectEqual(small_resident_bytes, client.executableCacheStats().resident_bytes);
    try std.testing.expectEqual(@as(u64, 1), client.executableCacheStats().evictions);
    try std.testing.expectEqual(large_resident_bytes, client.executableCacheStats().evicted_resident_bytes);
}

test "client trims idle executable cache for tracked memory pressure" {
    const client = try Client.init(std.testing.allocator, backend_api.create(), 1);
    defer client.deinit();
    const allocator = std.testing.allocator;
    const assignment = [_]i32{0};

    var small_plan = try ClientResidencyTestSupport.constantU8ExecutablePlan(allocator, &assignment, &.{ 1, 2, 3, 4 });
    defer small_plan.deinit();
    var large_plan = try ClientResidencyTestSupport.constantU8ExecutablePlan(allocator, &assignment, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer large_plan.deinit();

    try std.testing.expect(!try recordCompile(client, "pressure-small-constant"));
    var small_executable = try CompiledExecutable.Testing.initBorrowedResident(allocator, client.executableContext(), &small_plan, .{
        .cache_fingerprint = "pressure-small-constant",
    });
    var small_entry = try ClientResidencyTestSupport.cacheEntrySnapshot(client, "pressure-small-constant");
    const small_resident_bytes = small_entry.resident_bytes;
    CompiledExecutable.Testing.deinitBorrowedResident(&small_executable);

    try std.testing.expect(!try recordCompile(client, "pressure-large-constant"));
    var large_executable = try CompiledExecutable.Testing.initBorrowedResident(allocator, client.executableContext(), &large_plan, .{
        .cache_fingerprint = "pressure-large-constant",
    });
    var large_entry = try ClientResidencyTestSupport.cacheEntrySnapshot(client, "pressure-large-constant");
    const large_resident_bytes = large_entry.resident_bytes;
    try std.testing.expect(large_resident_bytes > small_resident_bytes);
    CompiledExecutable.Testing.deinitBorrowedResident(&large_executable);

    client.defaultMemory().stats.capacity_bytes = small_resident_bytes;
    const trim = client.trimExecutableCacheForAllocation(client.defaultMemory(), 0);
    try std.testing.expectEqual(@as(u64, 0), trim.requested_bytes);
    try std.testing.expectEqual(small_resident_bytes, trim.target_resident_bytes);
    try std.testing.expectEqual(large_resident_bytes, trim.freed_bytes);
    try std.testing.expectEqual(@as(u64, 1), trim.evicted_entries);
    try std.testing.expect(!trim.still_over_capacity);
    small_entry = try ClientResidencyTestSupport.cacheEntrySnapshot(client, "pressure-small-constant");
    large_entry = try ClientResidencyTestSupport.cacheEntrySnapshot(client, "pressure-large-constant");
    try std.testing.expect(small_entry.resident);
    try std.testing.expect(!large_entry.resident);
    try std.testing.expectEqual(small_resident_bytes, client.executableCacheStats().resident_bytes);
    try std.testing.expectEqual(@as(u64, 1), client.executableCacheStats().pressure_trim_requests);
    try std.testing.expectEqual(large_resident_bytes, client.executableCacheStats().pressure_trimmed_bytes);
    try std.testing.expectEqual(@as(u64, 0), client.executableCacheStats().pressure_trim_failures);
    try std.testing.expectEqual(@as(u64, 1), client.defaultMemory().stats.executable_cache_pressure_trims);
    try std.testing.expectEqual(large_resident_bytes, client.defaultMemory().stats.executable_cache_pressure_trimmed_bytes);
    try std.testing.expectEqual(@as(u64, 0), client.defaultMemory().stats.executable_cache_pressure_trim_failures);
}

test "compiling resident executable trims idle cache under memory pressure" {
    const client = try Client.init(std.testing.allocator, backend_api.create(), 1);
    defer client.deinit();
    const allocator = std.testing.allocator;
    const assignment = [_]i32{0};

    var idle_plan = try ClientResidencyTestSupport.constantU8ExecutablePlan(allocator, &assignment, &.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer idle_plan.deinit();
    var new_plan = try ClientResidencyTestSupport.constantU8ExecutablePlan(allocator, &assignment, &.{ 9, 10, 11, 12 });
    defer new_plan.deinit();

    try std.testing.expect(!try recordCompile(client, "compile-pressure-idle-constant"));
    var idle_executable = try CompiledExecutable.Testing.initBorrowedResident(allocator, client.executableContext(), &idle_plan, .{
        .cache_fingerprint = "compile-pressure-idle-constant",
    });
    var idle_entry = try ClientResidencyTestSupport.cacheEntrySnapshot(client, "compile-pressure-idle-constant");
    const idle_resident_bytes = idle_entry.resident_bytes;
    try std.testing.expect(idle_resident_bytes >= 8);
    CompiledExecutable.Testing.deinitBorrowedResident(&idle_executable);
    idle_entry = try ClientResidencyTestSupport.cacheEntrySnapshot(client, "compile-pressure-idle-constant");
    try std.testing.expectEqual(@as(usize, 0), idle_entry.reference_count);
    try std.testing.expect(idle_entry.resident);

    client.defaultMemory().stats.capacity_bytes = idle_resident_bytes;

    try std.testing.expect(!try recordCompile(client, "compile-pressure-new-constant"));
    var new_executable = try CompiledExecutable.Testing.initBorrowedResident(allocator, client.executableContext(), &new_plan, .{
        .cache_fingerprint = "compile-pressure-new-constant",
    });
    defer CompiledExecutable.Testing.deinitBorrowedResident(&new_executable);
    const new_entry = try ClientResidencyTestSupport.cacheEntrySnapshot(client, "compile-pressure-new-constant");
    try std.testing.expect(new_executable.hasResidentBackendExecutable());
    try std.testing.expect(new_entry.resident);
    try std.testing.expect(new_entry.resident_bytes >= 4);
    try std.testing.expect(new_entry.resident_bytes <= idle_resident_bytes);
    idle_entry = try ClientResidencyTestSupport.cacheEntrySnapshot(client, "compile-pressure-idle-constant");
    try std.testing.expect(!idle_entry.resident);
    try std.testing.expectEqual(new_entry.resident_bytes, new_executable.compileCacheTrim().requested_bytes);
    try std.testing.expectEqual(idle_resident_bytes - new_entry.resident_bytes, new_executable.compileCacheTrim().target_resident_bytes);
    try std.testing.expectEqual(idle_resident_bytes, new_executable.compileCacheTrim().freed_bytes);
    try std.testing.expectEqual(@as(u64, 1), new_executable.compileCacheTrim().evicted_entries);
    try std.testing.expectEqual(@as(u64, 0), new_executable.compileCacheTrim().remaining_resident_bytes);
    try std.testing.expect(!new_executable.compileCacheTrim().still_over_capacity);
    try std.testing.expectEqual(@as(u64, 1), client.executableCacheStats().pressure_trim_requests);
    try std.testing.expectEqual(idle_resident_bytes, client.executableCacheStats().pressure_trimmed_bytes);
    try std.testing.expectEqual(@as(u64, 0), client.executableCacheStats().pressure_trim_failures);
    try std.testing.expectEqual(new_entry.resident_bytes, client.executableCacheStats().resident_bytes);
    try std.testing.expectEqual(@as(u64, 1), client.defaultMemory().stats.executable_cache_pressure_trims);
    try std.testing.expectEqual(idle_resident_bytes, client.defaultMemory().stats.executable_cache_pressure_trimmed_bytes);
}

test "executable cache preserves more expensive equal-size resident executable" {
    const client = try Client.init(std.testing.allocator, backend_api.create(), 1);
    defer client.deinit();
    const allocator = std.testing.allocator;
    const assignment = [_]i32{0};

    var cheap_plan = try ClientResidencyTestSupport.constantU8ExecutablePlan(allocator, &assignment, &.{ 1, 2, 3, 4 });
    defer cheap_plan.deinit();
    var expensive_plan = try ClientResidencyTestSupport.constantU8ExecutablePlan(allocator, &assignment, &.{ 5, 6, 7, 8 });
    defer expensive_plan.deinit();

    try std.testing.expect(!try recordCompile(client, "cheap-constant"));
    var cheap_executable = try CompiledExecutable.Testing.initBorrowedResident(allocator, client.executableContext(), &cheap_plan, .{
        .cache_fingerprint = "cheap-constant",
    });
    var cheap_entry = try ClientResidencyTestSupport.cacheEntrySnapshot(client, "cheap-constant");
    const resident_bytes = cheap_entry.resident_bytes;
    CompiledExecutable.Testing.deinitBorrowedResident(&cheap_executable);
    Client.Testing.setExecutableCompileLatency(client, "cheap-constant", 10);

    try std.testing.expect(!try recordCompile(client, "expensive-constant"));
    var expensive_executable = try CompiledExecutable.Testing.initBorrowedResident(allocator, client.executableContext(), &expensive_plan, .{
        .cache_fingerprint = "expensive-constant",
    });
    var expensive_entry = try ClientResidencyTestSupport.cacheEntrySnapshot(client, "expensive-constant");
    try std.testing.expectEqual(resident_bytes, expensive_entry.resident_bytes);
    CompiledExecutable.Testing.deinitBorrowedResident(&expensive_executable);
    Client.Testing.setExecutableCompileLatency(client, "expensive-constant", 1000);

    try std.testing.expectEqual(@as(u64, 2), client.executableCacheStats().compile_latency_samples);
    try std.testing.expect(client.executableCacheStats().compile_latency_us_total > 0);
    try std.testing.expect(client.executableCacheStats().compile_latency_us_peak > 0);

    client.setExecutableCacheMaxResidentBytes(resident_bytes);
    cheap_entry = try ClientResidencyTestSupport.cacheEntrySnapshot(client, "cheap-constant");
    expensive_entry = try ClientResidencyTestSupport.cacheEntrySnapshot(client, "expensive-constant");
    try std.testing.expect(!cheap_entry.resident);
    try std.testing.expect(expensive_entry.resident);
    try std.testing.expectEqual(@as(u64, 1), client.executableCacheStats().resident_entries);
    try std.testing.expectEqual(resident_bytes, client.executableCacheStats().resident_bytes);
    try std.testing.expectEqual(@as(u64, 1), client.executableCacheStats().evictions);
    try std.testing.expectEqual(resident_bytes, client.executableCacheStats().evicted_resident_bytes);
}
