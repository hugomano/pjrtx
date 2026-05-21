const std = @import("std");

/// Returns whether backend profile events should be collected.
pub fn enabled() bool {
    return envFlag("PJRTX_PROFILE");
}

/// Returns whether per-schedule-item profile events should be emitted.
pub fn verbose() bool {
    const text = envText("PJRTX_PROFILE") orelse return false;
    return std.ascii.eqlIgnoreCase(text, "verbose") or std.ascii.eqlIgnoreCase(text, "2");
}

/// Returns whether backend trace diagnostics should be emitted.
pub fn traceEnabled() bool {
    return envText("PJRTX_TRACE") != null;
}

/// Returns whether MLX compiled-program creation is enabled for executables.
pub fn programCompileEnabled() bool {
    const text = envText("PJRTX_MLX_PROGRAM_COMPILE") orelse return true;
    return text.len == 0 or (!std.mem.eql(u8, text, "0") and !std.ascii.eqlIgnoreCase(text, "false"));
}

fn envFlag(comptime name: [:0]const u8) bool {
    const text = envText(name) orelse return false;
    return text.len != 0 and !std.mem.eql(u8, text, "0") and !std.ascii.eqlIgnoreCase(text, "false");
}

fn envText(comptime name: [:0]const u8) ?[]const u8 {
    const value = std.c.getenv(name) orelse return null;
    return std.mem.span(value);
}
