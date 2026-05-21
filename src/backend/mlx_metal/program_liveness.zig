const std = @import("std");

const program_mod = @import("program.zig");

/// Computes backend-program release and peak-memory observations from schedule metadata.
pub fn compute(program: *const program_mod.Program) program_mod.Error!program_mod.LivenessStats {
    const live_values = program.allocator.alloc(bool, program.values.len) catch return error.OutOfMemory;
    defer program.allocator.free(live_values);
    @memset(live_values, false);

    var live_count: usize = 0;
    var live_bytes: usize = 0;
    for (program.values, 0..) |value, value_index| {
        if (value.producer_node != null) continue;
        if (value.last_use_node == null and !value.is_output) continue;
        live_values[value_index] = true;
        live_count += 1;
        live_bytes += value.byte_size;
    }

    var result = program_mod.LivenessStats{
        .peak_live_value_count = live_count,
        .peak_live_bytes = live_bytes,
    };
    for (program.schedule) |item| {
        switch (item.kind) {
            .node => {
                if (item.index >= program.nodes.len) return error.InvalidProgram;
                const node = program.nodes[item.index];
                try markNodeOutputsLive(program, node, live_values, &live_count, &live_bytes, &result);
                try releaseDeadNodeInputs(program, node, item.index, live_values, &live_count, &live_bytes, &result);
            },
            .fusion_group => {
                if (item.index >= program.fusion_groups.len) return error.InvalidProgram;
                const group = program.fusion_groups[item.index];
                if (item.count != group.node_indices.len) return error.InvalidProgram;
                for (group.node_indices) |node_index| {
                    if (node_index >= program.nodes.len) return error.InvalidProgram;
                    try markNodeOutputsLive(program, program.nodes[node_index], live_values, &live_count, &live_bytes, &result);
                }
                for (group.node_indices) |node_index| {
                    if (node_index >= program.nodes.len) return error.InvalidProgram;
                    try releaseDeadFusionNodeInputs(program, program.nodes[node_index], group.last_node, live_values, &live_count, &live_bytes, &result);
                }
            },
            .materialization_boundary => {},
        }
    }

    return result;
}

fn markNodeOutputsLive(
    program: *const program_mod.Program,
    node: program_mod.Node,
    live_values: []bool,
    live_count: *usize,
    live_bytes: *usize,
    stats: *program_mod.LivenessStats,
) program_mod.Error!void {
    for (node.outputs) |output| {
        if (output.index >= live_values.len or output.index >= program.values.len) return error.InvalidProgram;
        if (live_values[output.index]) continue;
        live_values[output.index] = true;
        live_count.* += 1;
        live_bytes.* += program.values[output.index].byte_size;
        stats.peak_live_value_count = @max(stats.peak_live_value_count, live_count.*);
        stats.peak_live_bytes = @max(stats.peak_live_bytes, live_bytes.*);
    }
}

fn releaseDeadNodeInputs(
    program: *const program_mod.Program,
    node: program_mod.Node,
    node_index: usize,
    live_values: []bool,
    live_count: *usize,
    live_bytes: *usize,
    stats: *program_mod.LivenessStats,
) program_mod.Error!void {
    for (node.inputs) |input| {
        if (input.index >= program.values.len or input.index >= live_values.len) return error.InvalidProgram;
        const value = program.values[input.index];
        if (value.last_use_node != @as(?usize, node_index)) continue;
        try releasePlannedValue(program, input.index, live_values, live_count, live_bytes, stats);
    }
}

fn releaseDeadFusionNodeInputs(
    program: *const program_mod.Program,
    node: program_mod.Node,
    group_last_node: usize,
    live_values: []bool,
    live_count: *usize,
    live_bytes: *usize,
    stats: *program_mod.LivenessStats,
) program_mod.Error!void {
    for (node.inputs) |input| {
        if (input.index >= program.values.len or input.index >= live_values.len) return error.InvalidProgram;
        const value = program.values[input.index];
        const last_use = value.last_use_node orelse continue;
        if (last_use > group_last_node) continue;
        try releasePlannedValue(program, input.index, live_values, live_count, live_bytes, stats);
    }
}

fn releasePlannedValue(
    program: *const program_mod.Program,
    value_index: usize,
    live_values: []bool,
    live_count: *usize,
    live_bytes: *usize,
    stats: *program_mod.LivenessStats,
) program_mod.Error!void {
    const value = program.values[value_index];
    if (value.is_output) return;
    const producer_node = value.producer_node orelse return;
    if (producer_node >= program.nodes.len) return error.InvalidProgram;
    if (program.nodes[producer_node].kind == .constant) return;
    if (!live_values[value_index]) return;
    live_values[value_index] = false;
    live_count.* -= 1;
    live_bytes.* -= value.byte_size;
    stats.planned_release_count += 1;
    stats.planned_release_bytes += value.byte_size;
}
