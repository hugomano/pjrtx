const c = @import("c");

const custom_call = @import("custom_call.zig");
const plugin_process = @import("plugin_process.zig");

comptime {
    _ = custom_call;
}

/// Exports the PJRT C API table consumed by JAX, ZML, and other PJRT clients.
pub export fn GetPjrtApi() *const c.PJRT_Api {
    plugin_process.initialize();
    return @import("api.zig").Table.get();
}

test {
    @import("std").testing.refAllDecls(@This());
}
