const std = @import("std");

const ir = @import("src/compiler/ir");
const program_mod = @import("program.zig");

/// Computes value ids releasable after scheduled executable Metal graph work.
pub fn releaseValuesAfterNode(allocator: std.mem.Allocator, program: *const program_mod.Program, node_index: usize) program_mod.Error![]const u64 {
    if (node_index >= program.nodes.len) return error.CommandSubmissionFailed;
    const node = program.nodes[node_index];
    var marks = try allocator.alloc(bool, program.values.len);
    defer allocator.free(marks);
    @memset(marks, false);
    for (node.inputs) |input_id| {
        if (input_id.index >= program.values.len) return error.CommandSubmissionFailed;
        const value = program.values[input_id.index];
        if (!releasableValue(program, value)) continue;
        if (value.last_use_node != @as(?usize, node_index)) continue;
        marks[input_id.index] = true;
    }
    return markedValueIndices(allocator, marks);
}

/// Computes value ids releasable after scheduled executable Metal graph work.
pub fn releaseValuesAfterNodeRange(allocator: std.mem.Allocator, program: *const program_mod.Program, node_indices: []const usize) program_mod.Error![]const u64 {
    var marks = try allocator.alloc(bool, program.values.len);
    defer allocator.free(marks);
    @memset(marks, false);
    var last_node: usize = 0;
    for (node_indices) |node_index| {
        if (node_index >= program.nodes.len) return error.CommandSubmissionFailed;
        last_node = @max(last_node, node_index);
    }
    for (node_indices) |node_index| {
        const node = program.nodes[node_index];
        for (node.inputs) |input_id| {
            if (input_id.index >= program.values.len) return error.CommandSubmissionFailed;
            const value = program.values[input_id.index];
            if (!releasableValue(program, value)) continue;
            const last_use = value.last_use_node orelse continue;
            if (last_use > last_node) continue;
            marks[input_id.index] = true;
        }
    }
    return markedValueIndices(allocator, marks);
}

/// Computes value ids releasable after scheduled executable Metal graph work.
pub fn releaseValuesAfterInstructionRange(
    allocator: std.mem.Allocator,
    plan: *const ir.ExecutablePlan,
    program: *const program_mod.Program,
    node_indices: []const usize,
) program_mod.Error![]const u64 {
    var marks = try allocator.alloc(bool, program.values.len);
    defer allocator.free(marks);
    @memset(marks, false);
    var last_node: usize = 0;
    for (node_indices) |node_index| {
        if (node_index >= program.nodes.len) return error.CommandSubmissionFailed;
        last_node = @max(last_node, node_index);
    }
    for (node_indices) |node_index| {
        const node = program.nodes[node_index];
        if (node.instruction_index >= plan.instructions.len) return error.CommandSubmissionFailed;
        const instruction = plan.instructions[node.instruction_index];
        for (instruction.inputs) |input_id| {
            if (input_id.index >= program.values.len) return error.CommandSubmissionFailed;
            const value = program.values[input_id.index];
            if (!releasableValue(program, value)) continue;
            const last_use = value.last_use_node orelse continue;
            if (last_use > last_node) continue;
            marks[input_id.index] = true;
        }
    }
    return markedValueIndices(allocator, marks);
}

/// Computes value ids releasable after scheduled executable Metal graph work.
pub fn releaseValuesAfterFusionGroup(allocator: std.mem.Allocator, program: *const program_mod.Program, group: program_mod.FusionGroup) program_mod.Error![]const u64 {
    var marks = try allocator.alloc(bool, program.values.len);
    defer allocator.free(marks);
    @memset(marks, false);
    for (group.node_indices) |node_index| {
        if (node_index >= program.nodes.len) return error.CommandSubmissionFailed;
        const node = program.nodes[node_index];
        for (node.inputs) |input_id| {
            if (input_id.index >= program.values.len) return error.CommandSubmissionFailed;
            const value = program.values[input_id.index];
            if (!releasableValue(program, value)) continue;
            const last_use = value.last_use_node orelse continue;
            if (last_use > group.last_node) continue;
            marks[input_id.index] = true;
        }
    }
    return markedValueIndices(allocator, marks);
}

fn releasableValue(program: *const program_mod.Program, value: program_mod.Value) bool {
    if (value.is_output or value.materialization_boundary != null) return false;
    const producer_node = value.producer_node orelse return false;
    if (producer_node >= program.nodes.len) return false;
    return program.nodes[producer_node].kind != .constant;
}

fn markedValueIndices(allocator: std.mem.Allocator, marks: []const bool) ![]const u64 {
    var count: usize = 0;
    for (marks) |mark| {
        if (mark) count += 1;
    }
    const indices = try allocator.alloc(u64, count);
    var out_index: usize = 0;
    for (marks, 0..) |mark, value_index| {
        if (!mark) continue;
        indices[out_index] = value_index;
        out_index += 1;
    }
    return indices;
}
