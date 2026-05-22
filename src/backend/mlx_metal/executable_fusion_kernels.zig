const std = @import("std");

const ir = @import("src/compiler/ir");
const metalcpp_call = @import("metalcpp_call.zig");
const metalcpp_fusion_runner = @import("metalcpp_fusion_runner.zig");
const profiling = @import("profiling.zig");
const program_mod = @import("program.zig");

/// Resident generated Metal-cpp fusion kernel handle.
pub const KernelHandle = metalcpp_call.FusionKernelHandle;

/// Allocates device-by-fusion-group resident kernel slots.
pub fn allocSlots(allocator: std.mem.Allocator, device_count: usize, group_count: usize) ![]?KernelHandle {
    const handles = try allocator.alloc(?KernelHandle, device_count * group_count);
    @memset(handles, null);
    return handles;
}

/// Compiles supported generated fusion kernels into executable residency.
pub fn compileIfSupported(
    allocator: std.mem.Allocator,
    plan: *const ir.ExecutablePlan,
    program: *const program_mod.Program,
    device_local_hardware_ids: []const i32,
    handles: []?KernelHandle,
) program_mod.Error!void {
    if (!profiling.metalCppFusionRunnerEnabled()) return;
    if (program.fusion_groups.len == 0) return;
    if (handles.len != device_local_hardware_ids.len * program.fusion_groups.len) return error.CommandSubmissionFailed;

    for (program.fusion_groups) |group| {
        var request = (try metalcpp_fusion_runner.initKernelRequest(allocator, plan, program, group)) orelse continue;
        defer request.deinit(allocator);
        for (device_local_hardware_ids, 0..) |local_hardware_id, device_index| {
            const kernel = (try metalcpp_fusion_runner.createKernel(
                allocator,
                request,
                plan,
                group.id,
                local_hardware_id,
            )) orelse continue;
            errdefer metalcpp_call.fusionKernelDestroy(kernel);
            handles[slot(program.fusion_groups.len, device_index, group.id)] = kernel;
        }
    }
}

/// Returns the resident kernel for a device/group pair when one was compiled.
pub fn get(handles: []?KernelHandle, group_count: usize, device_index: usize, group_id: usize) ?KernelHandle {
    const index = slot(group_count, device_index, group_id);
    if (index >= handles.len) return null;
    return handles[index];
}

/// Releases all resident generated Metal-cpp fusion kernels.
pub fn destroy(handles: []?KernelHandle) void {
    for (handles) |maybe_handle| {
        if (maybe_handle) |handle| metalcpp_call.fusionKernelDestroy(handle);
    }
}

fn slot(group_count: usize, device_index: usize, group_id: usize) usize {
    return device_index * group_count + group_id;
}

test "fusion kernel slots are grouped by device then group" {
    try std.testing.expectEqual(@as(usize, 0), slot(3, 0, 0));
    try std.testing.expectEqual(@as(usize, 2), slot(3, 0, 2));
    try std.testing.expectEqual(@as(usize, 3), slot(3, 1, 0));
    try std.testing.expectEqual(@as(usize, 5), slot(3, 1, 2));
}
