const std = @import("std");

const io = std.Io.Threaded.global_single_threaded.io();

/// Returns the backend IO handle used for timestamps and non-cancelable locks.
pub fn backendIo() std.Io {
    return io;
}

/// Starts a monotonic timer when profiling is enabled.
/// A zero return value is a disabled timer and always has zero elapsed time.
pub fn start(is_enabled: bool) std.Io.Timestamp {
    return if (is_enabled) nowTimestamp() else .zero;
}

/// Returns the elapsed microseconds since `start`.
pub fn elapsedUs(start_timestamp: std.Io.Timestamp) u64 {
    if (start_timestamp.nanoseconds == 0) return 0;
    return @intCast(@max(start_timestamp.durationTo(nowTimestamp()).toMicroseconds(), 0));
}

fn nowTimestamp() std.Io.Timestamp {
    return std.Io.Timestamp.now(io, .awake);
}
