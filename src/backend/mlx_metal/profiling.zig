const std = @import("std");

const clock = @import("profiling_clock.zig");
const env = @import("profiling_env.zig");
const stats = @import("profiling_stats.zig");
const writer = @import("profiling_writer.zig");

/// Aggregates one backend executable invocation profile.
/// Times are recorded in microseconds and saturating-add into executable stats.
pub const Execute = stats.Execute;

/// Schedule item categories used by MLX backend profile aggregation.
pub const ScheduleKind = stats.ScheduleKind;

/// Summary values required to render executable-level profile events.
/// The struct avoids coupling this module to the concrete executable owner.
pub const ExecutableSnapshot = writer.ExecutableSnapshot;

/// Summary values required to render one schedule item profile event.
/// Callers populate optional labels from the backend program graph.
pub const ScheduleItemSnapshot = writer.ScheduleItemSnapshot;

/// Summary values required to render a backend schedule failure trace.
pub const ScheduleFailureSnapshot = writer.ScheduleFailureSnapshot;

/// Summary values required to render a materialization failure trace.
pub const MaterializationFailureSnapshot = writer.MaterializationFailureSnapshot;

/// Summary values required to render executable-level Metal graph compile coverage.
pub const MetalGraphCompileSnapshot = writer.MetalGraphCompileSnapshot;

/// Summary values required to render executable-level Metal graph execution coverage.
pub const MetalGraphExecuteSnapshot = writer.MetalGraphExecuteSnapshot;

/// Returns the backend IO handle used for timestamps and non-cancelable locks.
pub fn backendIo() std.Io {
    return clock.backendIo();
}

/// Returns whether backend profile events should be collected.
pub fn enabled() bool {
    return env.enabled();
}

/// Returns whether per-schedule-item profile events should be emitted.
pub fn verbose() bool {
    return env.verbose();
}

/// Returns whether backend trace diagnostics should be emitted.
pub fn traceEnabled() bool {
    return env.traceEnabled();
}

/// Returns whether MLX compiled-program creation is enabled for executables.
pub fn programCompileEnabled() bool {
    return env.programCompileEnabled();
}

/// Returns whether direct backend-local metal-cpp kernels should be tried before MLX operations.
pub fn metalCppExecuteEnabled() bool {
    return env.metalCppExecuteEnabled();
}

/// Returns whether env-gated Metal-cpp fusion group execution should be attempted.
pub fn metalCppFusionRunnerEnabled() bool {
    return env.metalCppFusionRunnerEnabled();
}

/// Returns whether env-gated executable-level Metal-cpp graph execution should be attempted.
pub fn metalCppExecutableRunnerEnabled() bool {
    return env.metalCppExecutableRunnerEnabled();
}

/// Returns the optional minimum group size for executable-level generated Metal fusion.
pub fn metalCppExecutableFusionMinNodes() ?usize {
    return env.metalCppExecutableFusionMinNodes();
}

/// Returns whether conservative executable-level generated Metal fusion should be tried.
pub fn metalCppExecutableConservativeFusionEnabled() bool {
    return env.metalCppExecutableConservativeFusionEnabled();
}

/// Returns the byte threshold for scalar/control-like parameters kept dynamic during initial argument capture.
pub fn argumentCaptureSmallControlBytes(default_value: usize) usize {
    return env.argumentCaptureSmallControlBytes(default_value);
}

/// Returns the minimum stable input count required before building an argument-captured program.
pub fn argumentCaptureMinStableInputs(default_value: usize) usize {
    return env.argumentCaptureMinStableInputs(default_value);
}

/// Returns the optional stable input cap used to disable high-pressure argument-captured programs.
pub fn argumentCaptureMaxStableInputs() ?usize {
    return env.argumentCaptureMaxStableInputs();
}

/// Returns the directory where experimental Metal source artifacts should be written.
pub fn metalCppMslDir() ?[]const u8 {
    return env.metalCppMslDir();
}

/// Starts a monotonic timer when profiling is enabled.
/// A zero return value is a disabled timer and always has zero elapsed time.
pub fn start(is_enabled: bool) std.Io.Timestamp {
    return clock.start(is_enabled);
}

/// Returns the elapsed microseconds since `start`.
pub fn elapsedUs(start_timestamp: std.Io.Timestamp) u64 {
    return clock.elapsedUs(start_timestamp);
}

/// Accumulates one execute profile into an executable stats struct.
/// The pointed-to value must expose the backend executable stats fields.
pub fn recordExecute(stats_target: anytype, profile: Execute) void {
    stats.recordExecute(stats_target, profile);
}

/// Emits the executable-level backend profile event.
pub fn writeExecute(snapshot: ExecutableSnapshot, device_index: usize, argument_count: usize, output_count: usize, profile: Execute) void {
    writer.writeExecute(snapshot, device_index, argument_count, output_count, profile);
}

/// Emits one schedule item profile event.
pub fn writeScheduleItem(snapshot: ScheduleItemSnapshot, elapsed_us: u64) void {
    writer.writeScheduleItem(snapshot, elapsed_us);
}

/// Emits a backend schedule failure trace event.
pub fn writeScheduleFailure(snapshot: ScheduleFailureSnapshot) void {
    writer.writeScheduleFailure(snapshot);
}

/// Emits a materialization failure trace event.
pub fn writeMaterializationFailure(snapshot: MaterializationFailureSnapshot) void {
    writer.writeMaterializationFailure(snapshot);
}

/// Emits executable-level Metal graph compile coverage.
pub fn writeMetalGraphCompile(snapshot: MetalGraphCompileSnapshot) void {
    writer.writeMetalGraphCompile(snapshot);
}

/// Emits executable-level Metal graph execution coverage.
pub fn writeMetalGraphExecute(snapshot: MetalGraphExecuteSnapshot) void {
    writer.writeMetalGraphExecute(snapshot);
}
