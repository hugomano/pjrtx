const std = @import("std");
const backend_api = @import("src/backend/mlx_metal");
const ir = @import("src/compiler/ir");

const device_memory = @import("device_memory.zig");
const executable_cache = @import("executable_cache.zig");

const CachedBackendExecutable = executable_cache.Retained;
const ExecutableCacheLease = executable_cache.Lease;
const ExecutableCacheTrim = executable_cache.Trim;
const Device = device_memory.Device;
const Memory = device_memory.Memory;

/// Error set for acquiring a cached or freshly compiled backend executable.
pub const CacheAcquireError = std.mem.Allocator.Error || backend_api.Error || error{UnsupportedRuntimeFeature};

/// Callback used by executable graphs to acquire a resident backend executable.
pub const AcquireCachedExecutableFn = *const fn (*anyopaque, std.mem.Allocator, []const u8, *const ir.ExecutablePlan, []const i32) CacheAcquireError!?CachedBackendExecutable;

/// Callback used by executable graphs to release a retained cache entry.
pub const ReleaseCachedExecutableFn = *const fn (*anyopaque, ExecutableCacheLease) void;

/// Callback used by execution to trim resident executable cache for output allocation.
pub const TrimExecutableCacheFn = *const fn (*anyopaque, *Memory, usize) ExecutableCacheTrim;

/// Narrow runtime surface needed by executable residency and execution.
pub const Context = struct {
    backend: backend_api.Backend,
    devices: []const Device,
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

    /// Returns the number of runtime devices available to this executable context.
    pub fn deviceCount(self: Context) usize {
        return self.devices.len;
    }

    /// Returns a runtime device by executable device index.
    pub fn deviceAt(self: Context, index: usize) ?*const Device {
        if (index >= self.devices.len) return null;
        return &self.devices[index];
    }

    /// Returns the default device id for an executable device index.
    pub fn defaultDeviceIdAt(self: Context, index: usize) ?i32 {
        const device = self.deviceAt(index) orelse return null;
        return device.id;
    }

    /// Compiles an executable plan for the selected backend devices.
    pub fn compileBackendExecutable(
        self: Context,
        allocator: std.mem.Allocator,
        plan: *const ir.ExecutablePlan,
        device_local_hardware_ids: []const i32,
    ) backend_api.Error!?backend_api.ExecutableHandle {
        return self.backend.compileExecutable(allocator, plan, device_local_hardware_ids);
    }

    /// Destroys a backend executable that is not retained by the executable cache.
    pub fn destroyBackendExecutable(self: Context, executable: backend_api.ExecutableHandle) void {
        self.backend.destroyExecutable(executable);
    }

    /// Writes backend lowering diagnostics for a plan that could not be resident.
    pub fn writeBackendLoweringDiagnostic(
        self: Context,
        plan: *const ir.ExecutablePlan,
        device_local_hardware_ids: []const i32,
        writer: *std.Io.Writer,
    ) void {
        self.backend.writeExecutableLoweringDiagnostic(plan, device_local_hardware_ids, writer) catch {};
    }

    /// Executes a resident backend executable for one device through this context.
    pub fn executeBackendExecutable(
        self: Context,
        allocator: std.mem.Allocator,
        executable: backend_api.ExecutableHandle,
        device_index: usize,
        argument_handles: []const backend_api.BufferHandle,
    ) backend_api.Error!?backend_api.ExecutionResult {
        return self.backend.executeExecutable(allocator, executable, device_index, argument_handles);
    }

    /// Destroys a backend buffer produced during execution setup.
    pub fn destroyBackendBuffer(self: Context, buffer: backend_api.BufferHandle) void {
        self.backend.destroyBuffer(buffer);
    }

    /// Returns the concrete backend needed to attach owned storage to runtime buffers.
    pub fn bufferStorageBackend(self: Context) backend_api.Backend {
        return self.backend;
    }

    /// Converts a backend execution completion into a runtime-observable status.
    pub fn executionEventStatus(self: Context, event: backend_api.ExecutionEventHandle) backend_api.Error!backend_api.ExecutionEventStatus {
        return self.backend.executionEventStatus(event);
    }

    /// Releases a backend execution event after status observation.
    pub fn destroyExecutionEvent(self: Context, event: backend_api.ExecutionEventHandle) void {
        self.backend.destroyExecutionEvent(event);
    }

    /// Returns the backend object retained by executable residency.
    pub fn backendForExecutableResidency(self: Context) backend_api.Backend {
        return self.backend;
    }

    /// Acquires a cached backend executable through the owning runtime client.
    pub fn acquireCachedExecutable(self: Context, allocator: std.mem.Allocator, fingerprint: []const u8, plan: *const ir.ExecutablePlan, device_local_hardware_ids: []const i32) CacheAcquireError!?CachedBackendExecutable {
        const callback = self.acquire_cached_executable orelse return null;
        _ = self.release_cached_executable orelse return null;
        const user_context = self.user_context orelse return null;
        return callback(user_context, allocator, fingerprint, plan, device_local_hardware_ids);
    }

    /// Releases a retained cache lease through the owning runtime client.
    pub fn releaseCachedExecutable(self: Context, lease: ExecutableCacheLease) void {
        if (self.release_cached_executable) |callback| {
            if (self.user_context) |user_context| callback(user_context, lease);
        }
    }

    /// Requests executable-cache pressure relief before execution allocates outputs.
    pub fn trimExecutableCacheForAllocation(self: Context, memory: *Memory, allocation_bytes: usize) ExecutableCacheTrim {
        const callback = self.trim_executable_cache orelse return .{ .requested_bytes = @intCast(allocation_bytes) };
        const user_context = self.user_context orelse return .{ .requested_bytes = @intCast(allocation_bytes) };
        return callback(user_context, memory, allocation_bytes);
    }
};
