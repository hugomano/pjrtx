const std = @import("std");

const ir = @import("src/compiler/ir");
const buffer_mod = @import("buffer.zig");
const lowering_mod = @import("lowering.zig");
const program_mod = @import("program.zig");

/// Opaque MLX/Metal buffer handle retained by executable residency.
pub const BufferHandle = *anyopaque;

/// Resident constant count and byte totals loaded during executable compile.
pub const LoadStats = struct {
    count: usize = 0,
    bytes: usize = 0,
};

/// Allocates empty resident instruction-constant slots.
pub fn allocInstructionSlots(allocator: std.mem.Allocator, instruction_count: usize, device_count: usize) ![]?BufferHandle {
    const handles = try allocator.alloc(?BufferHandle, instruction_count * device_count);
    @memset(handles, null);
    return handles;
}

/// Allocates empty resident while-pattern constant slots.
pub fn allocWhileSlots(allocator: std.mem.Allocator, control_flow_count: usize, device_count: usize) ![]?BufferHandle {
    const handles = try allocator.alloc(?BufferHandle, control_flow_count * device_count * 2);
    @memset(handles, null);
    return handles;
}

/// Loads all executable resident constants into device buffers.
pub fn load(
    plan: *const ir.ExecutablePlan,
    program: program_mod.Program,
    device_local_hardware_ids: []const i32,
    constant_handles: []?BufferHandle,
    while_constant_handles: []?BufferHandle,
) program_mod.Error!LoadStats {
    var stats: LoadStats = .{};
    try loadInstructionConstants(plan, device_local_hardware_ids, constant_handles, &stats);
    try loadWhilePatternConstants(program, device_local_hardware_ids, while_constant_handles, &stats);
    return stats;
}

/// Releases resident constant buffers.
pub fn destroy(handles: []?BufferHandle) void {
    for (handles) |maybe_handle| {
        if (maybe_handle) |handle| buffer_mod.Opaque.destroy(handle);
    }
}

/// Returns the resident constant slot for a plan instruction on one device.
pub fn constantIndex(instruction_count: usize, device_index: usize, instruction_index: usize) usize {
    return device_index * instruction_count + instruction_index;
}

/// Returns the resident while-pattern constant slot for one control-flow node.
pub fn whileConstantIndex(control_flow_count: usize, device_index: usize, control_flow_index: usize, constant_index: usize) usize {
    return ((device_index * control_flow_count) + control_flow_index) * 2 + constant_index;
}

fn loadInstructionConstants(
    plan: *const ir.ExecutablePlan,
    device_local_hardware_ids: []const i32,
    constant_handles: []?BufferHandle,
    stats: *LoadStats,
) program_mod.Error!void {
    for (device_local_hardware_ids, 0..) |local_hardware_id, device_index| {
        for (plan.instructions, 0..) |instruction, instruction_index| {
            if (instruction.kind != .constant) continue;
            const output_id = instruction.outputs[0];
            const descriptor = plan.values[output_id.index].descriptor;
            const literal = instruction.literal.?;
            constant_handles[constantIndex(plan.instructions.len, device_index, instruction_index)] = try materializeConstant(
                local_hardware_id,
                descriptor.element_type,
                descriptor.dims,
                literal,
            );
            stats.count += 1;
            stats.bytes += literal.len;
        }
    }
}

fn loadWhilePatternConstants(
    program: program_mod.Program,
    device_local_hardware_ids: []const i32,
    while_constant_handles: []?BufferHandle,
    stats: *LoadStats,
) program_mod.Error!void {
    for (program.control_flows, 0..) |control_flow, control_flow_index| {
        if (control_flow.condition_subprogram >= program.subprograms.len or control_flow.body_subprogram >= program.subprograms.len) return error.InvalidProgram;
        const pattern = lowering_mod.matchWhileF32LtAddPattern(program.subprograms[control_flow.condition_subprogram], program.subprograms[control_flow.body_subprogram]) orelse continue;
        for (device_local_hardware_ids, 0..) |local_hardware_id, device_index| {
            if (pattern.limit.role == .constant) {
                const limit_literal = pattern.limit.literal orelse return error.InvalidProgram;
                while_constant_handles[whileConstantIndex(program.control_flows.len, device_index, control_flow_index, 0)] = try materializeConstant(
                    local_hardware_id,
                    pattern.limit.descriptor.element_type,
                    pattern.limit.descriptor.dims,
                    limit_literal,
                );
                stats.count += 1;
                stats.bytes += limit_literal.len;
            }
            if (pattern.step.value.role == .constant) {
                const step_literal = pattern.step.value.literal orelse return error.InvalidProgram;
                while_constant_handles[whileConstantIndex(program.control_flows.len, device_index, control_flow_index, 1)] = try materializeConstant(
                    local_hardware_id,
                    pattern.step.value.descriptor.element_type,
                    pattern.step.value.descriptor.dims,
                    step_literal,
                );
                stats.count += 1;
                stats.bytes += step_literal.len;
            }
        }
    }
}

fn materializeConstant(local_hardware_id: i32, element_type: ir.BufferType, dims: []const i64, literal: []const u8) program_mod.Error!BufferHandle {
    const buffer = (try buffer_mod.Buffer.fromHost(local_hardware_id, element_type, dims, literal)) orelse return error.CommandSubmissionFailed;
    return buffer.toHandle();
}

test "resident constant indices are device-major" {
    try std.testing.expectEqual(@as(usize, 7), constantIndex(4, 1, 3));
    try std.testing.expectEqual(@as(usize, 10), whileConstantIndex(3, 1, 2, 0));
    try std.testing.expectEqual(@as(usize, 11), whileConstantIndex(3, 1, 2, 1));
}
