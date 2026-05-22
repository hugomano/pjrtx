const std = @import("std");

const diagnostic = @import("program_validation_diagnostic.zig");

const Error = diagnostic.Error;

/// Verifies schedule coverage and producer-before-consumer ordering.
pub fn validate(program: anytype, writer: ?*std.Io.Writer) (Error || std.Io.Writer.Error)!void {
    const scheduled_nodes = try program.allocator.alloc(bool, program.nodes.len);
    defer program.allocator.free(scheduled_nodes);
    @memset(scheduled_nodes, false);
    const schedule_ranks = try program.allocator.alloc(usize, program.nodes.len);
    defer program.allocator.free(schedule_ranks);
    @memset(schedule_ranks, std.math.maxInt(usize));
    const scheduled_boundaries = try program.allocator.alloc(bool, program.materialization_boundaries.len);
    defer program.allocator.free(scheduled_boundaries);
    @memset(scheduled_boundaries, false);
    const boundary_schedule_ranks = try program.allocator.alloc(usize, program.materialization_boundaries.len);
    defer program.allocator.free(boundary_schedule_ranks);
    @memset(boundary_schedule_ranks, std.math.maxInt(usize));
    var next_schedule_rank: usize = 0;

    for (program.schedule, 0..) |item, schedule_index| {
        if (item.count == 0) {
            return diagnostic.invalidProgram(writer, "schedule item {} has count=0", .{schedule_index});
        }
        switch (item.kind) {
            .node => {
                if (item.index >= program.nodes.len or item.count != 1) {
                    return diagnostic.invalidProgram(writer, "schedule item {} invalid node index={} count={} nodes.len={}", .{ schedule_index, item.index, item.count, program.nodes.len });
                }
                if (program.nodes[item.index].fusion_group != null) {
                    return diagnostic.invalidProgram(writer, "schedule item {} directly schedules fused node {}", .{ schedule_index, item.index });
                }
                if (scheduled_nodes[item.index]) {
                    return diagnostic.invalidProgram(writer, "schedule item {} schedules node {} more than once", .{ schedule_index, item.index });
                }
                scheduled_nodes[item.index] = true;
                schedule_ranks[item.index] = next_schedule_rank;
                next_schedule_rank += 1;
            },
            .fusion_group => {
                if (item.index >= program.fusion_groups.len) {
                    return diagnostic.invalidProgram(writer, "schedule item {} fusion_group={} is outside fusion_groups.len={}", .{ schedule_index, item.index, program.fusion_groups.len });
                }
                if (item.count != program.fusion_groups[item.index].node_indices.len) {
                    return diagnostic.invalidProgram(writer, "schedule item {} fusion_group={} count={} does not match node_indices.len={}", .{ schedule_index, item.index, item.count, program.fusion_groups[item.index].node_indices.len });
                }
                for (program.fusion_groups[item.index].node_indices) |node_index| {
                    if (scheduled_nodes[node_index]) {
                        return diagnostic.invalidProgram(writer, "schedule item {} schedules fusion group {} node {} more than once", .{ schedule_index, item.index, node_index });
                    }
                    scheduled_nodes[node_index] = true;
                    schedule_ranks[node_index] = next_schedule_rank;
                    next_schedule_rank += 1;
                }
            },
            .materialization_boundary => {
                if (item.index > program.materialization_boundaries.len) {
                    return diagnostic.invalidProgram(writer, "schedule item {} materialization index={} is outside materialization_boundaries.len={}", .{ schedule_index, item.index, program.materialization_boundaries.len });
                }
                if (item.count > program.materialization_boundaries.len - item.index) {
                    return diagnostic.invalidProgram(writer, "schedule item {} materialization range index={} count={} exceeds materialization_boundaries.len={}", .{ schedule_index, item.index, item.count, program.materialization_boundaries.len });
                }
                for (item.index..item.index + item.count) |boundary_index| {
                    if (scheduled_boundaries[boundary_index]) {
                        return diagnostic.invalidProgram(writer, "schedule item {} schedules materialization boundary {} more than once", .{ schedule_index, boundary_index });
                    }
                    scheduled_boundaries[boundary_index] = true;
                    boundary_schedule_ranks[boundary_index] = next_schedule_rank;
                    next_schedule_rank += 1;
                }
            },
        }
    }

    for (scheduled_nodes, 0..) |scheduled, node_index| {
        if (!scheduled) {
            return diagnostic.invalidProgram(writer, "node {} is not covered by schedule", .{node_index});
        }
    }

    for (scheduled_boundaries, 0..) |scheduled, boundary_index| {
        if (!scheduled) {
            return diagnostic.invalidProgram(writer, "materialization boundary {} is not covered by schedule", .{boundary_index});
        }
        const value_id = program.materialization_boundaries[boundary_index].value_id;
        if (program.values[value_id.index].producer_node) |producer_node| {
            if (schedule_ranks[producer_node] >= boundary_schedule_ranks[boundary_index]) {
                return diagnostic.invalidProgram(writer, "materialization boundary {} value_id={} is scheduled before producer node {}", .{ boundary_index, value_id.index, producer_node });
            }
        }
    }

    for (program.edges, 0..) |edge, edge_index| {
        if (schedule_ranks[edge.from_node] >= schedule_ranks[edge.to_node]) {
            return diagnostic.invalidProgram(writer, "edge {} value_id={} violates schedule order: producer node {} rank {} must run before consumer node {} rank {}", .{
                edge_index,
                edge.value_id.index,
                edge.from_node,
                schedule_ranks[edge.from_node],
                edge.to_node,
                schedule_ranks[edge.to_node],
            });
        }
    }
}
