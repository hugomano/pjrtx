const std = @import("std");

const program_mod = @import("program.zig");

/// Builds the owned execution schedule from graph nodes, fusion groups, and materialization outputs.
pub fn build(
    allocator: std.mem.Allocator,
    nodes: []const program_mod.Node,
    fusion_groups: []const program_mod.FusionGroup,
    materialization_boundaries: []const program_mod.MaterializationBoundary,
) ![]program_mod.ScheduleItem {
    const max_schedule_len = nodes.len + if (materialization_boundaries.len == 0) @as(usize, 0) else 1;
    const max_schedule = try allocator.alloc(program_mod.ScheduleItem, max_schedule_len);
    defer allocator.free(max_schedule);

    var schedule_index: usize = 0;
    var node_index: usize = 0;
    while (node_index < nodes.len) {
        const node = nodes[node_index];
        if (node.fusion_group) |group_id| {
            const group = fusion_groups[group_id];
            if (group.first_node == node_index) {
                max_schedule[schedule_index] = .{
                    .kind = .fusion_group,
                    .index = group_id,
                    .count = group.node_indices.len,
                };
                schedule_index += 1;
                node_index = group.last_node + 1;
                continue;
            }
        }
        max_schedule[schedule_index] = .{
            .kind = .node,
            .index = node_index,
        };
        schedule_index += 1;
        node_index += 1;
    }

    if (materialization_boundaries.len != 0) {
        max_schedule[schedule_index] = .{
            .kind = .materialization_boundary,
            .index = 0,
            .count = materialization_boundaries.len,
        };
        schedule_index += 1;
    }

    return allocator.dupe(program_mod.ScheduleItem, max_schedule[0..schedule_index]);
}
