const std = @import("std");

const c = @import("c");
const abi = @import("pjrt_abi.zig");
const custom_call = @import("custom_call.zig");

comptime {
    _ = custom_call;
}

/// Owns process-lifetime metadata advertised through PJRT.
pub const Metadata = struct {
    /// PJRT plugin name reported through attributes.
    pub const plugin_name = "PjRTx";

    /// StableHLO version currently advertised by the plugin.
    pub const stablehlo_current_version = [_]i64{ 1, 16, 0 };

    /// Minimum StableHLO version accepted by the plugin.
    pub const stablehlo_minimum_version = [_]i64{ 1, 0, 0 };
};

/// Owns platform identity exposed to PJRT clients.
pub const Platform = struct {
    /// PJRT platform name used by JAX/ZML discovery.
    pub const name = "pjrtx";

    /// Human-readable PJRT platform version.
    pub const version = "PjRTx Metal/MLX";

    /// Device kind reported through PJRT device descriptions.
    pub const device_kind = "Metal";
};

/// Owns PJRT option names accepted by this plugin.
pub const Options = struct {
    /// PJRT client create option used to select the PjRTx backend.
    pub const backend = "pjrtx_backend";
};

/// Owns PJRT memory-kind names reported by the Metal/MLX plugin.
pub const MemoryKinds = struct {
    /// Default PJRT memory kind for device buffers.
    pub const device = "device";
};

const Context = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
};

const InitPhase = enum { cold, ready };

var context_state: InitPhase = .cold;
var context_storage: Context = undefined;
var context_mutex: std.Io.Mutex = .init;
var attrs_state: InitPhase = .cold;
var attrs: [5]abi.NamedValue = undefined;
var attrs_mutex: std.Io.Mutex = .init;

/// Initializes process-lifetime plugin state before any PJRT object is handed out.
pub fn initialize() void {
    const io_handle = std.Io.Threaded.global_single_threaded.io();
    context_mutex.lockUncancelable(io_handle);
    defer context_mutex.unlock(io_handle);

    if (context_state == .ready) return;
    context_storage = .{
        .allocator = std.heap.c_allocator,
        .io = io_handle,
    };
    context_state = .ready;
}

fn context() *const Context {
    initialize();
    return &context_storage;
}

/// Returns the allocator used for PJRT-owned objects that cross the C ABI.
pub fn allocator() std.mem.Allocator {
    return context().allocator;
}

/// Returns the IO handle used for plugin tracing and timestamps.
pub fn io() std.Io {
    return context().io;
}

/// Owns the process-lifetime PJRT plugin attributes returned to callers.
pub const Attributes = struct {
    /// Returns plugin attributes borrowed for the process lifetime.
    pub fn borrow() []abi.NamedValue {
        initialize();
        attrs_mutex.lockUncancelable(io());
        defer attrs_mutex.unlock(io());

        if (attrs_state == .ready) return attrs[0..];
        attrs = .{
            abi.NamedValue.string("plugin_name", Metadata.plugin_name),
            abi.NamedValue.string("xla_version", "local"),
            abi.NamedValue.int64List("stablehlo_current_version", Metadata.stablehlo_current_version.len, &Metadata.stablehlo_current_version),
            abi.NamedValue.int64List("stablehlo_minimum_version", Metadata.stablehlo_minimum_version.len, &Metadata.stablehlo_minimum_version),
            abi.NamedValue.string("pjrtx_default_backend", "metal_mlx"),
        };

        attrs_state = .ready;
        return attrs[0..];
    }
};

const PluginOp = enum { initialize, attributes };

fn PluginCallback(comptime Args: type, comptime op: PluginOp) type {
    return struct {
        fn call(raw: [*c]Args) callconv(.c) ?*c.PJRT_Error {
            switch (op) {
                .initialize => initialize(),
                .attributes => {
                    const values = Attributes.borrow();
                    raw[0].attributes = abi.NamedValue.borrowedMany(values);
                    raw[0].num_attributes = values.len;
                },
            }
            return null;
        }
    };
}

/// PJRT plugin metadata callbacks installed into the global API table.
pub const Api = struct {
    pub const Initialize = PluginCallback(c.PJRT_Plugin_Initialize_Args, .initialize).call;
    pub const Attributes = PluginCallback(c.PJRT_Plugin_Attributes_Args, .attributes).call;
};

/// Exports the PJRT C API table consumed by JAX, ZML, and other PJRT clients.
pub export fn GetPjrtApi() *const c.PJRT_Api {
    initialize();
    return @import("api.zig").Table.get();
}

test {
    @import("std").testing.refAllDecls(@This());
}
