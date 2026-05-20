const std = @import("std");

const c = @import("c");

const async_h2d_mod = @import("async_h2d.zig");
const buffer_mod = @import("buffer.zig");
const client_mod = @import("client.zig");
const custom_call = @import("custom_call.zig");
const device_memory_mod = @import("device_memory.zig");
const errors_mod = @import("errors.zig");
const events_mod = @import("events.zig");
const executable_mod = @import("executable.zig");
const execute_mod = @import("execute.zig");
const plugin = @import("plugin.zig");
const topology_mod = @import("topology.zig");
const trace_mod = @import("trace.zig");

var api_storage: c.PJRT_Api = undefined;
var api_state: enum { cold, ready } = .cold;
var api_mutex: std.Io.Mutex = .init;

fn installOneScope(api: *c.PJRT_Api, comptime prefix: []const u8, comptime Api: type) void {
    inline for (@typeInfo(Api).@"struct".decls) |decl| {
        const name = prefix ++ decl.name;
        @field(api.*, name) = trace_mod.Api.Callback(name, @field(Api, decl.name)).call;
    }
}

fn installScope(api: *c.PJRT_Api, comptime prefix: []const u8, comptime scopes: anytype) void {
    switch (@typeInfo(@TypeOf(scopes))) {
        .type => installOneScope(api, prefix, scopes),
        .@"struct" => |info| {
            if (!info.is_tuple) @compileError("PJRT API scope must be a type or tuple of types");
            inline for (scopes) |Scope| installOneScope(api, prefix, Scope);
        },
        else => @compileError("PJRT API scope must be a type or tuple of types"),
    }
}

fn installApi(api: *c.PJRT_Api) void {
    installScope(api, "PJRT_Error_", errors_mod.Error.Api);
    installScope(api, "PJRT_Plugin_", plugin.Api);
    installScope(api, "PJRT_Event_", events_mod.Event.Api);
    installScope(api, "PJRT_Client_", client_mod.Client.Api);
    installScope(api, "PJRT_AsyncHostToDeviceTransferManager_", async_h2d_mod.TransferManager.Api);
    installScope(api, "PJRT_TopologyDescription_", topology_mod.TopologyDescription.Api);
    installScope(api, "PJRT_DeviceDescription_", device_memory_mod.DeviceDescription.Api);
    installScope(api, "PJRT_Device_", device_memory_mod.Device.Api);
    installScope(api, "PJRT_Memory_", device_memory_mod.Memory.Api);
    installScope(api, "PJRT_Executable_", executable_mod.ExecutableMetadata.Api);
    installScope(api, "PJRT_LoadedExecutable_", .{
        executable_mod.LoadedExecutable.Api,
        execute_mod.Execution.Api,
    });
    installScope(api, "PJRT_Buffer_", buffer_mod.Buffer.Api);
}

fn initApi() void {
    plugin.initialize();
    api_mutex.lockUncancelable(plugin.io());
    defer api_mutex.unlock(plugin.io());

    if (api_state == .ready) return;
    api_storage = std.mem.zeroes(c.PJRT_Api);
    api_storage.struct_size = c.PJRT_Api_STRUCT_SIZE;
    api_storage.extension_start = custom_call.extensionBase();
    api_storage.pjrt_api_version = .{
        .struct_size = c.PJRT_Api_Version_STRUCT_SIZE,
        .extension_start = null,
        .major_version = c.PJRT_API_MAJOR,
        .minor_version = c.PJRT_API_MINOR,
    };

    installApi(&api_storage);

    api_state = .ready;
}

/// Owns the process-lifetime PJRT API table exposed by `GetPjrtApi`.
pub const Table = struct {
    /// Initializes the PJRT API table once and returns the borrowed table.
    pub fn get() *const c.PJRT_Api {
        initApi();
        return &api_storage;
    }
};
