const std = @import("std");
const compiler = @import("src/compiler");
const ir = @import("src/compiler/ir");

const compile_options = @import("compile_options.zig");
const device_memory = @import("device_memory.zig");

const Device = device_memory.Device;

/// Program bytes and compile options accepted by the runtime compile boundary.
pub const Program = struct {
    format: []const u8 = "",
    code: []const u8 = &.{},
    compile_options: []const u8 = &.{},
};

/// Errors surfaced by runtime compilation before PJRT ABI translation.
pub const Error = error{
    InvalidOptions,
    OptionsRequireMoreDevices,
    UnknownDevice,
    UnsupportedProgram,
    InvalidProgram,
    InvalidExecutablePlan,
    UnsupportedRuntimeFeature,
    OutOfMemory,
    Internal,
};

/// Owns parsed compile options and any allocation performed while parsing them.
pub const ParsedOptions = struct {
    options: ir.CompileOptions,
    owns_device_assignment: bool = false,

    /// Releases compile-option storage owned by the parser.
    pub fn deinit(self: *ParsedOptions, allocator: std.mem.Allocator) void {
        if (self.owns_device_assignment) allocator.free(self.options.device_assignment);
        self.* = undefined;
    }
};

/// Parses and validates compile options against the client topology.
pub fn parseOptions(
    allocator: std.mem.Allocator,
    devices: []const Device,
    text: []const u8,
    diagnostics: *std.Io.Writer,
) Error!ParsedOptions {
    var parsed: ParsedOptions = .{
        .options = .{ .num_partitions = @intCast(devices.len) },
    };
    if (text.len != 0) {
        if (compile_options.isPjrtxText(text)) {
            var options_reader: std.Io.Reader = .fixed(text);
            parsed.options = compiler.parseTextCompileOptionsFromReader(allocator, &options_reader) catch {
                writeDiagnostic(diagnostics, "invalid PjRTx text compile options", .{});
                return error.InvalidOptions;
            };
        } else {
            parsed.options = compile_options.parseXlaProto(allocator, text) catch {
                writeDiagnostic(diagnostics, "invalid XLA CompileOptionsProto", .{});
                return error.InvalidOptions;
            };
        }
        parsed.owns_device_assignment = true;
    }

    if (parsed.options.numDevices() > devices.len) {
        writeDiagnostic(diagnostics, "compile options require more devices than the client exposes", .{});
        return error.OptionsRequireMoreDevices;
    }
    for (parsed.options.device_assignment) |device_id| {
        if (lookupDevice(devices, device_id) == null) {
            writeDiagnostic(diagnostics, "compile options reference an unknown device id", .{});
            return error.UnknownDevice;
        }
    }
    return parsed;
}

/// Imports and analyzes program bytes using the compiler-owned frontend.
pub fn analyzeProgram(
    allocator: std.mem.Allocator,
    program: Program,
    diagnostics: *std.Io.Writer,
) Error!?compiler.ModuleAnalysis {
    if (program.code.len == 0) return null;
    var module_reader: std.Io.Reader = .fixed(program.code);
    return compiler.analyzeProgramFromReader(allocator, program.format, &module_reader, diagnostics) catch |err| switch (err) {
        error.UnsupportedOp, error.UnsupportedSharding, error.UnsupportedProgramEncoding, error.GspmdNotEnabled, error.InvalidManualComputation => error.UnsupportedProgram,
        error.UnsupportedProgramFormat, error.InvalidStablehloModule => error.InvalidProgram,
        error.OutOfMemory => error.OutOfMemory,
        else => error.Internal,
    };
}

/// Builds the compiler-owned executable plan consumed by runtime residency.
pub fn makeExecutablePlan(
    allocator: std.mem.Allocator,
    options: ir.CompileOptions,
    analysis: ?compiler.ModuleAnalysis,
) Error!ir.ExecutablePlan {
    return if (analysis) |owned_analysis|
        compiler.makeExecutablePlan(allocator, options, owned_analysis) catch |err| switch (err) {
            error.OutOfMemory => error.OutOfMemory,
        }
    else
        compiler.makeReplicatedPlan(allocator, options, 1, 1) catch |err| switch (err) {
            error.OutOfMemory => error.OutOfMemory,
        };
}

/// Verifies that the executable plan satisfies runtime-facing invariants.
pub fn verifyPlan(
    allocator: std.mem.Allocator,
    plan: ir.ExecutablePlan,
    diagnostics: *std.Io.Writer,
) Error!void {
    compiler.verifyExecutablePlan(allocator, plan, diagnostics) catch |err| switch (err) {
        error.InvalidExecutablePlan => return error.InvalidExecutablePlan,
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.Internal,
    };
}

/// Retains the optimized program text used by PJRT executable metadata.
pub fn retainOptimizedProgram(allocator: std.mem.Allocator, analysis: ?compiler.ModuleAnalysis) Error![]u8 {
    return if (analysis) |owned_analysis|
        allocator.dupe(u8, owned_analysis.source) catch return error.OutOfMemory
    else
        allocator.dupe(u8, "module {}\n") catch return error.OutOfMemory;
}

fn lookupDevice(devices: []const Device, id: i32) ?*const Device {
    for (devices) |*device| {
        if (device.id == id) return device;
    }
    return null;
}

fn writeDiagnostic(writer: *std.Io.Writer, comptime fmt: []const u8, args: anytype) void {
    writer.print(fmt, args) catch {};
}
