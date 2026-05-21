const ir = @import("src/compiler/ir");
const mlx_call = @import("mlx_call.zig");
const profiling_mod = @import("profiling.zig");

/// Opaque MLX compiled-program handle retained by executable residency.
pub const ProgramHandle = mlx_call.ProgramHandle;

/// Builds the device-local callback context type used by MLX compiled-program builders.
pub fn Context(comptime Executable: type) type {
    return struct {
        executable: *Executable,
        device_index: usize,
    };
}

/// Allocates empty MLX compiled-program handle slots.
pub fn allocHandles(allocator: anytype, device_count: usize) ![]?ProgramHandle {
    const handles = try allocator.alloc(?ProgramHandle, device_count);
    @memset(handles, null);
    return handles;
}

/// Initializes callback contexts after the executable object has been assigned.
pub fn initContexts(executable: anytype) void {
    for (executable.compiled_program_contexts, 0..) |*context, device_index| {
        context.* = .{
            .executable = executable,
            .device_index = device_index,
        };
    }
}

/// Creates per-device MLX compiled-program handles when the plan supports that path.
pub fn compileIfSupported(executable: anytype, plan: *const ir.ExecutablePlan, comptime build_callback: mlx_call.ProgramBuildCallback) !void {
    if (!profiling_mod.programCompileEnabled() or !planSupportsCompiledProgram(plan)) return;
    for (executable.compiled_program_handles, 0..) |*handle_slot, device_index| {
        handle_slot.* = mlx_call.programCreate(
            &executable.compiled_program_contexts[device_index],
            plan.parameter_shardings.len,
            plan.output_ids.len,
            build_callback,
        ) orelse return error.CommandSubmissionFailed;
    }
}

/// Releases MLX compiled-program handles.
pub fn destroyHandles(handles: []?ProgramHandle) void {
    for (handles) |maybe_handle| {
        if (maybe_handle) |handle| mlx_call.programDestroy(handle);
    }
}

fn planSupportsCompiledProgram(plan: *const ir.ExecutablePlan) bool {
    for (plan.instructions) |instruction| {
        switch (instruction.kind) {
            // The RNG path is device-native, but it uses a custom Metal kernel under MLX
            // and is not currently valid inside an mlx::compile graph builder.
            .rng_bit_generator => return false,
            else => {},
        }
    }
    return true;
}
