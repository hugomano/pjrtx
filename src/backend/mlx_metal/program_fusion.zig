const std = @import("std");
const ir = @import("src/compiler/ir");

const program_mod = @import("program.zig");

/// Builds owned fusion-group metadata from provisional contiguous fusible ranges.
pub fn build(
    allocator: std.mem.Allocator,
    groups: []const program_mod.FusionGroup,
    nodes: []const program_mod.Node,
    values: []const program_mod.Value,
    edges: []const program_mod.Edge,
) ![]program_mod.FusionGroup {
    const fusion_groups = try allocator.alloc(program_mod.FusionGroup, groups.len);
    errdefer allocator.free(fusion_groups);
    @memcpy(fusion_groups, groups);
    var initialized_groups: usize = 0;
    errdefer deinit(allocator, fusion_groups[0..initialized_groups]);

    const mark_count = groups.len * values.len;
    const input_marks = try allocator.alloc(bool, mark_count);
    defer allocator.free(input_marks);
    @memset(input_marks, false);
    const output_marks = try allocator.alloc(bool, mark_count);
    defer allocator.free(output_marks);
    @memset(output_marks, false);

    for (nodes) |node| {
        const group_id = node.fusion_group orelse continue;
        for (node.inputs) |input_id| {
            if (input_id.index >= values.len) continue;
            if (values[input_id.index].producer_node == null) {
                input_marks[groupMarkIndex(values.len, group_id, input_id.index)] = true;
            }
        }
        for (node.outputs) |output_id| {
            if (output_id.index >= values.len) continue;
            if (values[output_id.index].materialization_boundary != null) {
                output_marks[groupMarkIndex(values.len, group_id, output_id.index)] = true;
            }
        }
    }

    for (edges) |edge| {
        const to_group = nodes[edge.to_node].fusion_group;
        const from_group = nodes[edge.from_node].fusion_group;
        if (to_group) |group_id| {
            if (from_group != group_id) {
                input_marks[groupMarkIndex(values.len, group_id, edge.value_id.index)] = true;
            }
        }
        if (from_group) |group_id| {
            if (to_group != group_id) {
                output_marks[groupMarkIndex(values.len, group_id, edge.value_id.index)] = true;
            }
        }
    }

    for (fusion_groups) |*group| {
        group.node_indices = try fusionGroupNodeIndices(allocator, group.*);
        errdefer allocator.free(group.node_indices);
        group.input_values = try markedValueIds(allocator, values, input_marks[group.id * values.len ..][0..values.len]);
        errdefer allocator.free(group.input_values);
        group.output_values = try markedValueIds(allocator, values, output_marks[group.id * values.len ..][0..values.len]);
        initialized_groups += 1;
    }

    return fusion_groups;
}

/// Releases owned fusion-group metadata built by this module.
pub fn deinit(allocator: std.mem.Allocator, groups: []program_mod.FusionGroup) void {
    for (groups) |group| {
        group.deinit(allocator);
    }
    allocator.free(groups);
}

fn fusionGroupNodeIndices(allocator: std.mem.Allocator, group: program_mod.FusionGroup) ![]const usize {
    if (group.node_count == 0) return allocator.alloc(usize, 0);
    if (group.last_node < group.first_node) return error.CommandSubmissionFailed;
    if (group.last_node - group.first_node + 1 != group.node_count) return error.CommandSubmissionFailed;

    const node_indices = try allocator.alloc(usize, group.node_count);
    var index: usize = 0;
    var node_index = group.first_node;
    while (index < node_indices.len) : ({
        index += 1;
        node_index += 1;
    }) {
        node_indices[index] = node_index;
    }
    return node_indices;
}

fn groupMarkIndex(value_count: usize, group_id: usize, value_index: usize) usize {
    return group_id * value_count + value_index;
}

fn markedValueIds(allocator: std.mem.Allocator, values: []const program_mod.Value, marks: []const bool) ![]const ir.ValueId {
    var count: usize = 0;
    for (marks) |mark| {
        if (mark) count += 1;
    }
    const ids = try allocator.alloc(ir.ValueId, count);
    var index: usize = 0;
    for (values, marks) |value, mark| {
        if (!mark) continue;
        ids[index] = value.value_id;
        index += 1;
    }
    return ids;
}
