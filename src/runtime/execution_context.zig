const std = @import("std");
const backend_api = @import("backend_selection.zig");
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
    pub const lookupDevice = ContextTopology.lookupDevice;

    /// Returns the number of runtime devices available to this executable context.
    pub fn deviceCount(self: Context) usize {
        return self.devices.len;
    }

    /// Returns a runtime device by executable device index.
    pub const deviceAt = ContextTopology.deviceAt;

    /// Returns the default device id for an executable device index.
    pub const defaultDeviceIdAt = ContextTopology.defaultDeviceIdAt;

    /// Compiles an executable plan for the selected backend devices.
    pub const compileBackendExecutable = ContextBackend.compileExecutable;

    /// Destroys a backend executable that is not retained by the executable cache.
    pub const destroyBackendExecutable = ContextBackend.destroyExecutable;

    /// Writes backend lowering diagnostics for a plan that could not be resident.
    pub const writeBackendLoweringDiagnostic = ContextBackend.writeLoweringDiagnostic;

    /// Executes a resident backend executable for one device through this context.
    pub const executeBackendExecutable = ContextBackend.execute;

    /// Destroys a backend buffer produced during execution setup.
    pub const destroyBackendBuffer = ContextBackend.destroyBuffer;

    /// Returns the concrete backend needed to attach owned storage to runtime buffers.
    pub const bufferStorageBackend = ContextBackend.backend;

    /// Converts a backend execution completion into a runtime-observable status.
    pub const executionEventStatus = ContextBackend.eventStatus;

    /// Releases a backend execution event after status observation.
    pub const destroyExecutionEvent = ContextBackend.destroyEvent;

    /// Returns the backend object retained by executable residency.
    pub const backendForExecutableResidency = ContextBackend.backend;

    /// Acquires a cached backend executable through the owning runtime client.
    pub const acquireCachedExecutable = ContextCache.acquire;

    /// Releases a retained cache lease through the owning runtime client.
    pub const releaseCachedExecutable = ContextCache.release;

    /// Requests executable-cache pressure relief before execution allocates outputs.
    pub const trimExecutableCacheForAllocation = ContextCache.trimForAllocation;
};

const ContextTopology = struct {
    fn lookupDevice(context: Context, id: i32) ?*const Device {
        for (context.devices) |*device| if (device.id == id) return device;
        return null;
    }

    fn deviceAt(context: Context, index: usize) ?*const Device {
        return if (index >= context.devices.len) null else &context.devices[index];
    }

    fn defaultDeviceIdAt(context: Context, index: usize) ?i32 {
        const device = deviceAt(context, index) orelse return null;
        return device.id;
    }
};

const ContextBackend = struct {
    fn backend(context: Context) backend_api.Backend {
        return context.backend;
    }

    fn compileExecutable(context: Context, allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, device_local_hardware_ids: []const i32) backend_api.Error!?backend_api.ExecutableHandle {
        return context.backend.compileExecutable(allocator, plan, device_local_hardware_ids);
    }

    fn destroyExecutable(context: Context, executable: backend_api.ExecutableHandle) void {
        context.backend.destroyExecutable(executable);
    }

    fn writeLoweringDiagnostic(context: Context, plan: *const ir.ExecutablePlan, device_local_hardware_ids: []const i32, writer: *std.Io.Writer) void {
        context.backend.writeExecutableLoweringDiagnostic(plan, device_local_hardware_ids, writer) catch {};
    }

    fn execute(context: Context, allocator: std.mem.Allocator, executable: backend_api.ExecutableHandle, device_index: usize, argument_handles: []const backend_api.BufferHandle) backend_api.Error!?backend_api.ExecutionResult {
        return context.backend.executeExecutable(allocator, executable, device_index, argument_handles);
    }

    fn destroyBuffer(context: Context, buffer: backend_api.BufferHandle) void {
        context.backend.destroyBuffer(buffer);
    }

    fn eventStatus(context: Context, event: backend_api.ExecutionEventHandle) backend_api.Error!backend_api.ExecutionEventStatus {
        return context.backend.executionEventStatus(event);
    }

    fn destroyEvent(context: Context, event: backend_api.ExecutionEventHandle) void {
        context.backend.destroyExecutionEvent(event);
    }
};

const ContextCache = struct {
    fn acquire(context: Context, allocator: std.mem.Allocator, fingerprint: []const u8, plan: *const ir.ExecutablePlan, device_local_hardware_ids: []const i32) CacheAcquireError!?CachedBackendExecutable {
        const callback = context.acquire_cached_executable orelse return null;
        _ = context.release_cached_executable orelse return null;
        const user_context = context.user_context orelse return null;
        return callback(user_context, allocator, fingerprint, plan, device_local_hardware_ids);
    }

    fn release(context: Context, lease: ExecutableCacheLease) void {
        if (context.release_cached_executable) |callback| {
            if (context.user_context) |user_context| callback(user_context, lease);
        }
    }

    fn trimForAllocation(context: Context, memory: *Memory, allocation_bytes: usize) ExecutableCacheTrim {
        const callback = context.trim_executable_cache orelse return .{ .requested_bytes = @intCast(allocation_bytes) };
        const user_context = context.user_context orelse return .{ .requested_bytes = @intCast(allocation_bytes) };
        return callback(user_context, memory, allocation_bytes);
    }
};
