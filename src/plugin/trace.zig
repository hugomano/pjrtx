const std = @import("std");
const c = @import("c");
const state = @import("state.zig");

pub const Timestamp = std.Io.Timestamp;
pub const io = state.io;

pub fn traceEnabled() bool {
    if (envFlag("PJRTX_PROFILE")) return true;
    return envFlag("PJRTX_TRACE");
}
pub fn envFlag(comptime name: [:0]const u8) bool {
    const value = std.c.getenv(name) orelse return false;
    const text = std.mem.span(value);
    return text.len != 0 and !std.mem.eql(u8, text, "0") and !std.ascii.eqlIgnoreCase(text, "false");
}
pub fn executableCacheMaxBytesFromEnv() ?u64 {
    const value = std.c.getenv("PJRTX_EXECUTABLE_CACHE_MAX_BYTES") orelse return null;
    const text = std.mem.span(value);
    if (text.len == 0) return null;
    return std.fmt.parseUnsigned(u64, text, 10) catch null;
}
pub fn now() Timestamp {
    return Timestamp.now(io, .awake);
}
pub fn elapsedSinceUs(start: Timestamp) i64 {
    return start.durationTo(now()).toMicroseconds();
}
pub fn tracePjrtApiCall(
    comptime name: []const u8,
    start: Timestamp,
    args_ptr: usize,
    args_struct_size: usize,
    return_ptr: usize,
    error_code: c.PJRT_Error_Code,
) void {
    if (!traceEnabled()) return;
    std.debug.print(
        "pjrtx_trace event=pjrt_api_call name={s} args=0x{x} args_struct_size={d} returns=0x{x} error_code={d} failed={d} elapsed_us={d}\n",
        .{
            name,
            args_ptr,
            args_struct_size,
            return_ptr,
            error_code,
            @intFromBool(return_ptr != 0),
            elapsedSinceUs(start),
        },
    );
}
