const c = @import("c");

const api = @import("api.zig");
const custom_call = @import("custom_call.zig");

comptime {
    _ = custom_call;
}

pub export fn GetPjrtApi() *const c.PJRT_Api {
    return api.get();
}

test {
    @import("std").testing.refAllDecls(@This());
}
