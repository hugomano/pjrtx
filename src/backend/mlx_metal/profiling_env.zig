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

/// Returns whether direct backend-local metal-cpp kernels should be tried before MLX operations.
pub fn metalCppExecuteEnabled() bool {
    if (envFlag("PJRTX_METALCPP_EXECUTE")) return true;
    const selected = envText("PJRTX_RUNTIME_BACKEND") orelse envText("PJRTX_BACKEND") orelse return false;
    return std.mem.eql(u8, selected, "metalcpp") or std.mem.eql(u8, selected, "metal_cpp");
}

/// Returns whether env-gated Metal-cpp fusion group execution should be attempted.
pub fn metalCppFusionRunnerEnabled() bool {
    return envFlag("PJRTX_METALCPP_FUSION_RUNNER");
}

/// Returns whether env-gated executable-level Metal-cpp graph execution should be attempted.
pub fn metalCppExecutableRunnerEnabled() bool {
    return envFlag("PJRTX_METALCPP_EXECUTABLE_RUNNER");
}

/// Returns the optional minimum group size for executable-level generated Metal fusion.
pub fn metalCppExecutableFusionMinNodes() ?usize {
    return envUsize("PJRTX_METALCPP_EXECUTABLE_FUSION_MIN_NODES");
}

/// Returns whether conservative executable-level generated Metal fusion should be tried.
pub fn metalCppExecutableConservativeFusionEnabled() bool {
    return envFlag("PJRTX_METALCPP_EXECUTABLE_CONSERVATIVE_FUSION");
}

/// Returns the byte threshold for scalar/control-like parameters kept dynamic during initial argument capture.
pub fn argumentCaptureSmallControlBytes(default_value: usize) usize {
    return envUsize("PJRTX_MLX_ARGUMENT_CAPTURE_SMALL_CONTROL_BYTES") orelse default_value;
}

/// Returns the minimum stable input count required before building an argument-captured program.
pub fn argumentCaptureMinStableInputs(default_value: usize) usize {
    return envUsize("PJRTX_MLX_ARGUMENT_CAPTURE_MIN_STABLE_INPUTS") orelse default_value;
}

/// Returns the optional stable input cap used to disable high-pressure argument-captured programs.
pub fn argumentCaptureMaxStableInputs() ?usize {
    return envUsize("PJRTX_MLX_ARGUMENT_CAPTURE_MAX_STABLE_INPUTS");
}

/// Returns the directory where experimental Metal source artifacts should be written.
pub fn metalCppMslDir() ?[]const u8 {
    const text = envText("PJRTX_METALCPP_MSL_DIR") orelse return null;
    if (text.len == 0) return null;
    return text;
}

fn envFlag(comptime name: [:0]const u8) bool {
    const text = envText(name) orelse return false;
    return text.len != 0 and !std.mem.eql(u8, text, "0") and !std.ascii.eqlIgnoreCase(text, "false");
}

fn envUsize(comptime name: [:0]const u8) ?usize {
    const text = envText(name) orelse return null;
    if (text.len == 0) return null;
    return std.fmt.parseUnsigned(usize, text, 10) catch null;
}

fn envText(comptime name: [:0]const u8) ?[]const u8 {
    const value = std.c.getenv(name) orelse return null;
    return std.mem.span(value);
}
