const std = @import("std");
const ir = @import("src/compiler/ir");

const client_residency = @import("client_residency.zig");
const executable_cache = @import("executable_cache.zig");
const executable_mod = @import("executable.zig");

const CachedBackendExecutable = executable_cache.Retained;
const ClientResidencyContext = client_residency.Context;
const ExecutableCacheLease = executable_cache.Lease;
const ExecutableCacheTrim = executable_cache.Trim;
const ExecutableContext = executable_mod.Context;
const Memory = @import("device_memory.zig").Memory;

fn residencyFromExecutableContext(user_context: *anyopaque) *ClientResidencyContext {
    return @ptrCast(@alignCast(user_context));
}

fn acquireCachedBackendExecutableForContext(
    user_context: *anyopaque,
    allocator: std.mem.Allocator,
    fingerprint: []const u8,
    plan: *const ir.ExecutablePlan,
    device_local_hardware_ids: []const i32,
) executable_mod.CacheAcquireError!?CachedBackendExecutable {
    return client_residency.acquireCachedBackendExecutable(
        residencyFromExecutableContext(user_context),
        allocator,
        fingerprint,
        plan,
        device_local_hardware_ids,
    );
}

fn releaseCachedBackendExecutableForContext(user_context: *anyopaque, lease: ExecutableCacheLease) void {
    client_residency.releaseCachedBackendExecutable(residencyFromExecutableContext(user_context), lease);
}

fn trimExecutableCacheForContext(user_context: *anyopaque, memory: *Memory, allocation_bytes: usize) ExecutableCacheTrim {
    return client_residency.trimForAllocation(residencyFromExecutableContext(user_context), memory, allocation_bytes);
}

/// Builds the narrow executable context consumed by compiled executable residency and dispatch.
pub fn fromClient(client: anytype) ExecutableContext {
    return .{
        .backend = client.backend,
        .devices = client.device_memory.deviceSlice(),
        .user_context = &client.executable_residency_context,
        .acquire_cached_executable = acquireCachedBackendExecutableForContext,
        .release_cached_executable = releaseCachedBackendExecutableForContext,
        .trim_executable_cache = trimExecutableCacheForContext,
    };
}
