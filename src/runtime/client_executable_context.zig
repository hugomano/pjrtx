const std = @import("std");
const ir = @import("src/compiler/ir");

const client_mod = @import("client.zig");
const client_residency = @import("client_residency.zig");
const executable_cache = @import("executable_cache.zig");
const executable_mod = @import("executable.zig");

const CachedBackendExecutable = executable_cache.Retained;
const Client = client_mod.Client;
const ExecutableCacheLease = executable_cache.Lease;
const ExecutableCacheTrim = executable_cache.Trim;
const ExecutableContext = executable_mod.Context;
const Memory = @import("device_memory.zig").Memory;

fn clientFromExecutableContext(user_context: *anyopaque) *Client {
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
        clientFromExecutableContext(user_context),
        allocator,
        fingerprint,
        plan,
        device_local_hardware_ids,
    );
}

fn releaseCachedBackendExecutableForContext(user_context: *anyopaque, lease: ExecutableCacheLease) void {
    client_residency.releaseCachedBackendExecutable(clientFromExecutableContext(user_context), lease);
}

fn trimExecutableCacheForContext(user_context: *anyopaque, memory: *Memory, allocation_bytes: usize) ExecutableCacheTrim {
    return client_residency.trimForAllocation(clientFromExecutableContext(user_context), memory, allocation_bytes);
}

/// Builds the narrow executable context consumed by compiled executable residency and dispatch.
pub fn fromClient(client: *Client) ExecutableContext {
    return .{
        .backend = client.backend,
        .devices = client.device_memory.deviceSlice(),
        .user_context = client,
        .acquire_cached_executable = acquireCachedBackendExecutableForContext,
        .release_cached_executable = releaseCachedBackendExecutableForContext,
        .trim_executable_cache = trimExecutableCacheForContext,
    };
}
