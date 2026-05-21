const std = @import("std");

const ir = @import("src/compiler/ir");
const argument_capture_mod = @import("executable_argument_capture.zig");
const compiled_program_mod = @import("executable_compiled_program.zig");
const constants_mod = @import("executable_constants.zig");
const lowering_mod = @import("lowering.zig");
const mlx_call = @import("mlx_call.zig");
const program_mod = @import("program.zig");
const program_build_mod = @import("program_build.zig");
const stats_mod = @import("executable_stats.zig");

/// Compiles an executable plan into a resident MLX/Metal executable.
pub fn run(
    comptime Executable: type,
    comptime CompiledProgramContext: type,
    allocator: std.mem.Allocator,
    plan: *const ir.ExecutablePlan,
    device_local_hardware_ids: []const i32,
    comptime build_callback: mlx_call.ProgramBuildCallback,
) program_mod.Error!?*anyopaque {
    if (lowering_mod.executableIssue(plan, device_local_hardware_ids)) |_| return null;

    const executable = try allocator.create(Executable);
    errdefer allocator.destroy(executable);
    const ids = try allocator.dupe(i32, device_local_hardware_ids);
    errdefer allocator.free(ids);
    const constant_handles = try constants_mod.allocInstructionSlots(allocator, plan.instructions.len, device_local_hardware_ids.len);
    errdefer allocator.free(constant_handles);
    errdefer constants_mod.destroy(constant_handles);
    var program = buildProgram(allocator, plan, null) catch |err| switch (err) {
        error.WriteFailed => unreachable,
        error.InvalidDeviceCount => return error.InvalidDeviceCount,
        error.InvalidProgram => return error.InvalidProgram,
        error.UnsupportedElementType => return error.UnsupportedElementType,
        error.ShapeMismatch => return error.ShapeMismatch,
        error.BufferAllocationFailed => return error.BufferAllocationFailed,
        error.CommandSubmissionFailed => return error.CommandSubmissionFailed,
        error.BufferCopyFailed => return error.BufferCopyFailed,
        error.OutOfMemory => return error.OutOfMemory,
        error.InvalidCustomCall => return error.InvalidCustomCall,
    };
    errdefer program.deinit();
    const while_constant_handles = try constants_mod.allocWhileSlots(allocator, program.control_flows.len, device_local_hardware_ids.len);
    errdefer allocator.free(while_constant_handles);
    errdefer constants_mod.destroy(while_constant_handles);
    const compiled_program_contexts = try allocator.alloc(CompiledProgramContext, device_local_hardware_ids.len);
    errdefer allocator.free(compiled_program_contexts);
    const compiled_program_handles = try compiled_program_mod.allocHandles(allocator, device_local_hardware_ids.len);
    errdefer allocator.free(compiled_program_handles);
    errdefer compiled_program_mod.destroyHandles(compiled_program_handles);
    const argument_capture_states = try argument_capture_mod.allocStates(allocator, device_local_hardware_ids.len);
    errdefer allocator.free(argument_capture_states);
    errdefer argument_capture_mod.destroyStates(allocator, argument_capture_states);
    const liveness_stats = try program.livenessStats();
    const constant_stats = try constants_mod.load(
        plan,
        program,
        device_local_hardware_ids,
        constant_handles,
        while_constant_handles,
    );

    executable.* = .{
        .allocator = allocator,
        .plan = plan,
        .device_local_hardware_ids = ids,
        .constant_handles = constant_handles,
        .while_constant_handles = while_constant_handles,
        .compiled_program_contexts = compiled_program_contexts,
        .compiled_program_handles = compiled_program_handles,
        .argument_capture_states = argument_capture_states,
        .program = program,
        .stats = stats_mod.init(program, liveness_stats, constant_stats.count, constant_stats.bytes, device_local_hardware_ids.len),
    };
    compiled_program_mod.initContexts(executable);
    try compiled_program_mod.compileIfSupported(executable, plan, build_callback);
    return @ptrCast(executable);
}

fn buildProgram(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, diagnostic_writer: ?*std.Io.Writer) !program_mod.Program {
    return program_build_mod.build(allocator, plan, diagnostic_writer);
}
