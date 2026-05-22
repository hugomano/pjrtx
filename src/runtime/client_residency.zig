const std = @import("std");
const backend_api = @import("backend_selection.zig");
const ir = @import("src/compiler/ir");

const device_memory = @import("device_memory.zig");
const executable_cache = @import("executable_cache.zig");
const executable_mod = @import("executable.zig");

const CachedBackendExecutable = executable_cache.Retained;
const DeviceMemoryTopology = device_memory.DeviceMemoryTopology;
const CompiledExecutable = executable_mod.CompiledExecutable;
const ExecutableCacheLease = executable_cache.Lease;
const ExecutableResidencyCache = executable_cache.Residency;
const Memory = device_memory.Memory;

/// Narrow client-owned residency context used by cache policy and callbacks.
pub const Context = struct {
    backend: *backend_api.Backend,
    device_memory: *DeviceMemoryTopology,
    residency: *ExecutableResidencyCache,
    io: *std.Io,

    /// Binds cache policy to the owning client fields without exposing the client root.
    pub fn init(backend: *backend_api.Backend, topology: *DeviceMemoryTopology, residency: *ExecutableResidencyCache, io: *std.Io) Context {
        return .{
            .backend = backend,
            .device_memory = topology,
            .residency = residency,
            .io = io,
        };
    }
};

/// Records a compile request and returns whether the fingerprint was seen before.
pub fn recordCompile(owner: anytype, fingerprint: []const u8) !bool {
    const context = contextFrom(owner);
    return context.residency.recordCompile(context.io.*, context.device_memory.memorySlice(), fingerprint);
}

/// Returns executable-cache counters without exposing cache entries.
pub fn statsSnapshot(owner: anytype) executable_cache.Stats {
    const context = contextFrom(owner);
    return context.residency.statsSnapshot();
}

/// Returns a snapshot of one executable-cache entry without exposing mutable cache storage.
pub fn entrySnapshot(owner: anytype, fingerprint: []const u8) ?executable_cache.EntrySnapshot {
    const context = contextFrom(owner);
    return context.residency.entrySnapshot(fingerprint);
}

/// Overrides compile latency for one cache entry in focused cache-policy tests.
pub fn setCompileLatencyForTest(owner: anytype, fingerprint: []const u8, latency_us: u64) void {
    const context = contextFrom(owner);
    context.residency.setCompileLatencyForTest(fingerprint, latency_us);
}

/// Sets the resident executable-cache budget and evicts idle entries as needed.
pub fn setMaxResidentBytes(owner: anytype, max_resident_bytes: u64) void {
    const context = contextFrom(owner);
    context.residency.setMaxResidentBytes(context.io.*, context.backend.*, context.device_memory.memorySlice(), max_resident_bytes);
}

/// Trims idle resident executables to make room for a device allocation.
pub fn trimForAllocation(owner: anytype, memory: *Memory, allocation_bytes: usize) executable_cache.Trim {
    const context = contextFrom(owner);
    return context.residency.trimForAllocation(context.io.*, context.backend.*, context.device_memory.memorySlice(), memory, allocation_bytes);
}

/// Acquires a retained resident backend executable for a compiled executable.
pub fn acquireCachedBackendExecutable(
    owner: anytype,
    allocator: std.mem.Allocator,
    fingerprint: []const u8,
    plan: *const ir.ExecutablePlan,
    device_local_hardware_ids: []const i32,
) !?CachedBackendExecutable {
    const context = contextFrom(owner);
    return context.residency.acquireBackendExecutable(
        context.io.*,
        context.backend.*,
        allocator,
        context.device_memory.memorySlice(),
        context.device_memory.executableResidencyMemory(device_local_hardware_ids),
        fingerprint,
        plan,
        device_local_hardware_ids,
    );
}

/// Releases a retained executable-cache entry after residency teardown.
pub fn releaseCachedBackendExecutable(owner: anytype, lease: ExecutableCacheLease) void {
    const context = contextFrom(owner);
    context.residency.release(context.io.*, context.backend.*, context.device_memory.memorySlice(), lease);
}

fn contextFrom(owner: anytype) *Context {
    const Owner = @TypeOf(owner);
    return switch (Owner) {
        *Context => owner,
        *const Context => @constCast(owner),
        else => &owner.executable_residency_context,
    };
}
