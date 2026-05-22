const std = @import("std");
const ir = @import("src/compiler/ir");

const program_mod = @import("program.zig");
const program_schedule_build = @import("program_schedule_build.zig");

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

test "mlx metal backend program fuses nonmaterializing tuple structural nodes with elementwise chain" {
    const allocator = std.testing.allocator;

    const values = [_]program_mod.Value{
        parameterValue(0),
        parameterValue(1),
        producedValue(2, 0),
        producedValue(3, 1),
        producedValue(4, 2),
        materializedOutputValue(5, 3),
    };
    const nodes = [_]program_mod.Node{
        fusionNode(0, .elementwise, &.{ id(0), id(1) }, &.{id(2)}),
        fusionNode(1, .structural, &.{id(2)}, &.{id(3)}),
        fusionNode(2, .structural, &.{ id(3), id(2) }, &.{id(4)}),
        fusionNode(3, .elementwise, &.{ id(4), id(1) }, &.{id(5)}),
    };
    const edges = [_]program_mod.Edge{
        valueEdge(2, 0, 1),
        valueEdge(3, 1, 2),
        valueEdge(2, 0, 2),
        valueEdge(4, 2, 3),
    };
    const provisional_groups = [_]program_mod.FusionGroup{provisionalGroup(0, 3, 4)};

    const fusion_groups = try build(allocator, &provisional_groups, &nodes, &values, &edges);
    defer deinit(allocator, fusion_groups);

    const boundaries = [_]program_mod.MaterializationBoundary{.{ .value_id = id(5), .reason = .pjrt_output }};
    const schedule = try program_schedule_build.build(allocator, &nodes, fusion_groups, &boundaries);
    defer allocator.free(schedule);

    try std.testing.expectEqual(@as(usize, 1), fusion_groups.len);
    try std.testing.expectEqual(@as(usize, 2), schedule.len);
    try std.testing.expectEqual(program_mod.ScheduleKind.fusion_group, schedule[0].kind);
    try std.testing.expectEqual(@as(usize, 4), schedule[0].count);
    try std.testing.expectEqual(program_mod.ScheduleKind.materialization_boundary, schedule[1].kind);

    const group = fusion_groups[0];
    try std.testing.expectEqual(@as(usize, 0), group.first_node);
    try std.testing.expectEqual(@as(usize, 3), group.last_node);
    try std.testing.expectEqual(@as(usize, 4), group.node_count);
    try std.testing.expectEqual(@as(usize, 4), group.node_indices.len);
    try std.testing.expectEqual(program_mod.NodeKind.elementwise, nodes[0].kind);
    try std.testing.expectEqual(program_mod.NodeKind.structural, nodes[1].kind);
    try std.testing.expectEqual(program_mod.NodeKind.structural, nodes[2].kind);
    try std.testing.expectEqual(program_mod.NodeKind.elementwise, nodes[3].kind);
    for (nodes) |node| {
        try std.testing.expectEqual(@as(?usize, 0), node.fusion_group);
    }

    try std.testing.expectEqual(@as(usize, 2), group.input_values.len);
    try std.testing.expectEqual(@as(usize, 0), group.input_values[0].index);
    try std.testing.expectEqual(@as(usize, 1), group.input_values[1].index);
    try std.testing.expectEqual(@as(usize, 1), group.output_values.len);
    try std.testing.expectEqual(@as(usize, 5), group.output_values[0].index);
}

test "mlx metal backend program fuses dtype and precision passthroughs with elementwise chain" {
    const allocator = std.testing.allocator;

    const values = [_]program_mod.Value{
        parameterValue(0),
        producedValue(1, 0),
        producedValue(2, 1),
        parameterValue(3),
        producedValue(4, 2),
        producedValue(5, 3),
        materializedOutputValue(6, 4),
    };
    const nodes = [_]program_mod.Node{
        fusionNode(0, .materialize, &.{id(0)}, &.{id(1)}),
        fusionNode(1, .materialize, &.{id(1)}, &.{id(2)}),
        fusionNode(2, .elementwise, &.{ id(2), id(3) }, &.{id(4)}),
        fusionNode(3, .materialize, &.{id(4)}, &.{id(5)}),
        fusionNode(4, .elementwise, &.{ id(5), id(3) }, &.{id(6)}),
    };
    const edges = [_]program_mod.Edge{
        valueEdge(1, 0, 1),
        valueEdge(2, 1, 2),
        valueEdge(4, 2, 3),
        valueEdge(5, 3, 4),
    };
    const provisional_groups = [_]program_mod.FusionGroup{provisionalGroup(0, 4, 5)};

    const fusion_groups = try build(allocator, &provisional_groups, &nodes, &values, &edges);
    defer deinit(allocator, fusion_groups);

    const boundaries = [_]program_mod.MaterializationBoundary{.{ .value_id = id(6), .reason = .pjrt_output }};
    const schedule = try program_schedule_build.build(allocator, &nodes, fusion_groups, &boundaries);
    defer allocator.free(schedule);

    try std.testing.expectEqual(@as(usize, 1), fusion_groups.len);
    try std.testing.expectEqual(@as(usize, 2), schedule.len);
    try std.testing.expectEqual(program_mod.ScheduleKind.fusion_group, schedule[0].kind);
    try std.testing.expectEqual(@as(usize, 5), schedule[0].count);
    try std.testing.expectEqual(program_mod.ScheduleKind.materialization_boundary, schedule[1].kind);

    const group = fusion_groups[0];
    try std.testing.expectEqual(@as(usize, 0), group.first_node);
    try std.testing.expectEqual(@as(usize, 4), group.last_node);
    try std.testing.expectEqual(@as(usize, 5), group.node_count);
    for (nodes) |node| {
        try std.testing.expectEqual(@as(?usize, 0), node.fusion_group);
    }

    try std.testing.expectEqual(@as(usize, 2), group.input_values.len);
    try std.testing.expectEqual(@as(usize, 0), group.input_values[0].index);
    try std.testing.expectEqual(@as(usize, 3), group.input_values[1].index);
    try std.testing.expectEqual(@as(usize, 1), group.output_values.len);
    try std.testing.expectEqual(@as(usize, 6), group.output_values[0].index);
}

fn valueEdge(value_index: u32, from_node: usize, to_node: usize) program_mod.Edge {
    return .{ .value_id = id(value_index), .from_node = from_node, .to_node = to_node };
}

fn fusionNode(
    instruction_index: usize,
    kind: program_mod.NodeKind,
    inputs: []const ir.ValueId,
    outputs: []const ir.ValueId,
) program_mod.Node {
    return .{
        .instruction_index = instruction_index,
        .kind = kind,
        .inputs = inputs,
        .outputs = outputs,
        .fusion_group = 0,
    };
}

fn id(index: u32) ir.ValueId {
    return .{ .index = index };
}

fn materializedOutputValue(index: u32, producer_node: usize) program_mod.Value {
    var value = producedValue(index, producer_node);
    value.is_output = true;
    value.materialization_boundary = 0;
    return value;
}

fn parameterValue(index: u32) program_mod.Value {
    return .{
        .value_id = id(index),
        .producer_node = null,
    };
}

fn producedValue(index: u32, producer_node: usize) program_mod.Value {
    return .{
        .value_id = id(index),
        .producer_node = producer_node,
    };
}

fn provisionalGroup(first_node: usize, last_node: usize, node_count: usize) program_mod.FusionGroup {
    return .{
        .id = 0,
        .kind = .view_elementwise,
        .first_node = first_node,
        .last_node = last_node,
        .node_count = node_count,
    };
}
