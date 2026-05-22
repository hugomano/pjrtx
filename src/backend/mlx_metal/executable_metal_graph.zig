const std = @import("std");

const ir = @import("src/compiler/ir");
const metalcpp_call = @import("metalcpp_call.zig");
const compile = @import("executable_metal_graph_compile.zig");
const program_mod = @import("program.zig");
const types = @import("execution_types.zig");

/// Resident executable-level generated Metal graph handle.
pub const Handle = metalcpp_call.ExecutableProgramHandle;

/// Allocates one executable-level Metal graph slot per addressable device.
pub fn allocHandles(allocator: std.mem.Allocator, device_count: usize) ![]?Handle {
    return compile.allocHandles(allocator, device_count);
}

/// Compiles a full executable-level generated Metal graph when the plan is eligible.
pub fn compileIfSupported(
    allocator: std.mem.Allocator,
    plan: *const ir.ExecutablePlan,
    program: *const program_mod.Program,
    device_local_hardware_ids: []const i32,
    handles: []?Handle,
) program_mod.Error!void {
    return compile.compileIfSupported(allocator, plan, program, device_local_hardware_ids, handles);
}

/// Releases executable-level generated Metal graph handles.
pub fn destroy(handles: []?Handle) void {
    compile.destroy(handles);
}

/// Returns one resident executable-level Metal graph handle for a device.
pub fn get(handles: []?Handle, device_index: usize) ?Handle {
    return compile.get(handles, device_index);
}

/// Executes an executable-level generated Metal graph over live arguments and constants.
pub fn execute(
    allocator: std.mem.Allocator,
    executable: anytype,
    device_index: usize,
    handle: Handle,
    arguments: []const types.BufferHandle,
) types.Error![]types.ExecutableOutput {
    return compile.execute(allocator, executable, device_index, handle, arguments);
}

test {
    _ = @import("executable_metal_graph_lowering_tests.zig");
    _ = @import("metal_graph_dot_tests.zig");
}
