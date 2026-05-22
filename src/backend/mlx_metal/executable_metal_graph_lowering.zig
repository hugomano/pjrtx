const std = @import("std");

const ir = @import("src/compiler/ir");
const profiling = @import("profiling.zig");
const fusion_lowering = @import("executable_metal_graph_lowering_fusion.zig");
const release_lowering = @import("executable_metal_graph_lowering_release.zig");
const program_mod = @import("program.zig");

/// Origin category for one lowered executable Metal graph step.
pub const StepOrigin = enum {
    schedule_node,
    fusion_group,
    fusion_fallback_node,
};

/// One schedule-level decision made before executable Metal graph codegen.
pub const StepPlan = union(enum) {
    instruction: Instruction,
    fusion_group: FusionGroup,
    fusion_segment: FusionSegment,
    dot_group: DotGroup,

    /// References one compiler instruction that should lower to one graph step.
    pub const Instruction = struct {
        index: usize,
        origin: StepOrigin,
        release_values: []const u64 = &.{},
    };

    /// References one backend fusion group that should lower to one graph step.
    pub const FusionGroup = struct {
        index: usize,
        release_values: []const u64 = &.{},
    };

    /// References a contiguous subrange of one backend fusion group.
    pub const FusionSegment = struct {
        group_index: usize,
        node_indices: []const usize,
        input_values: []const ir.ValueId,
        output_values: []const ir.ValueId,
        release_values: []const u64 = &.{},
    };

    /// References consecutive dot-general nodes that should lower to one multi-output graph step.
    pub const DotGroup = struct {
        node_indices: []const usize,
        release_values: []const u64 = &.{},
    };
};

/// Counters describing executable Metal graph lowering coverage.
pub const Metrics = struct {
    schedule_item_count: usize = 0,
    schedule_node_count: usize = 0,
    schedule_fusion_group_count: usize = 0,
    schedule_materialization_boundary_count: usize = 0,
    input_value_count: usize = 0,
    constant_input_count: usize = 0,
    step_count: usize = 0,
    node_step_count: usize = 0,
    fusion_step_count: usize = 0,
    alias_step_count: usize = 0,
    fallback_group_count: usize = 0,
    fallback_node_step_count: usize = 0,
    planned_structural_step_count: usize = 0,
    planned_view_step_count: usize = 0,
    planned_map_step_count: usize = 0,
    planned_reduction_step_count: usize = 0,
    planned_matmul_step_count: usize = 0,
    planned_control_flow_step_count: usize = 0,
    planned_materialize_step_count: usize = 0,
    planned_library_call_step_count: usize = 0,
    planned_other_step_count: usize = 0,
    fallback_structural_step_count: usize = 0,
    fallback_view_step_count: usize = 0,
    fallback_map_step_count: usize = 0,
    fallback_reduction_step_count: usize = 0,
    fallback_matmul_step_count: usize = 0,
    fallback_control_flow_step_count: usize = 0,
    fallback_materialize_step_count: usize = 0,
    fallback_library_call_step_count: usize = 0,
    fallback_other_step_count: usize = 0,
    release_step_count: usize = 0,
    release_value_count: usize = 0,
    dot_step_count: usize = 0,
    tiled_dot_step_count: usize = 0,
    dot_group_step_count: usize = 0,
    dot_group_node_count: usize = 0,
    reduce_step_count: usize = 0,
    tiled_reduce_step_count: usize = 0,
    reduce_max_index_step_count: usize = 0,
    tiled_reduce_max_index_step_count: usize = 0,

    fn init(program: *const program_mod.Program) Metrics {
        var metrics: Metrics = .{
            .schedule_item_count = program.schedule.len,
            .schedule_fusion_group_count = program.fusion_groups.len,
            .schedule_materialization_boundary_count = program.materialization_boundaries.len,
        };
        for (program.schedule) |item| {
            switch (item.kind) {
                .node => metrics.schedule_node_count += 1,
                .fusion_group => {},
                .materialization_boundary => {},
            }
        }
        return metrics;
    }

    /// Records one concrete generated step after codegen has chosen alias/kernel form.
    pub fn recordStep(self: *Metrics, origin: StepOrigin, is_alias: bool) void {
        self.step_count += 1;
        if (is_alias) self.alias_step_count += 1;
        switch (origin) {
            .schedule_node => self.node_step_count += 1,
            .fusion_group => self.fusion_step_count += 1,
            .fusion_fallback_node => self.fallback_node_step_count += 1,
        }
    }

    fn recordReleases(self: *Metrics, release_values: []const u64) void {
        if (release_values.len != 0) self.release_step_count += 1;
        self.release_value_count += release_values.len;
    }

    fn recordPlannedInstructionStep(self: *Metrics, node_kind: program_mod.NodeKind, origin: StepOrigin) void {
        switch (node_kind) {
            .constant => return,
            .parameter => self.planned_other_step_count += 1,
            .structural => self.planned_structural_step_count += 1, .view => self.planned_view_step_count += 1,
            .elementwise => self.planned_map_step_count += 1, .reduction => self.planned_reduction_step_count += 1,
            .matmul => self.planned_matmul_step_count += 1, .control_flow => self.planned_control_flow_step_count += 1,
            .library_call => self.planned_library_call_step_count += 1, .materialize => self.planned_materialize_step_count += 1,
        }
        if (origin != .fusion_fallback_node) return;
        switch (node_kind) {
            .constant => unreachable,
            .parameter => self.fallback_other_step_count += 1,
            .structural => self.fallback_structural_step_count += 1, .view => self.fallback_view_step_count += 1,
            .elementwise => self.fallback_map_step_count += 1, .reduction => self.fallback_reduction_step_count += 1,
            .matmul => self.fallback_matmul_step_count += 1, .control_flow => self.fallback_control_flow_step_count += 1,
            .library_call => self.fallback_library_call_step_count += 1, .materialize => self.fallback_materialize_step_count += 1,
        }
    }
};

/// Owns the lowered step plan used by executable Metal graph codegen.
pub const Plan = struct {
    steps: []StepPlan,
    metrics: Metrics,

    /// Releases step-plan storage owned by this lowering result.
    pub fn deinit(self: *Plan, allocator: std.mem.Allocator) void {
        for (self.steps) |step| {
            switch (step) {
                .instruction => |instruction| allocator.free(instruction.release_values),
                .fusion_group => |group| allocator.free(group.release_values),
                .fusion_segment => |segment| {
                    allocator.free(segment.node_indices);
                    allocator.free(segment.input_values);
                    allocator.free(segment.output_values);
                    allocator.free(segment.release_values);
                },
                .dot_group => |group| {
                    allocator.free(group.node_indices);
                    allocator.free(group.release_values);
                },
            }
        }
        allocator.free(self.steps);
        self.* = undefined;
    }
};

/// Lowers a backend program schedule into executable Metal graph codegen steps.
pub fn run(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, program: *const program_mod.Program) program_mod.Error!Plan {
    var steps: std.ArrayList(StepPlan) = .empty;
    errdefer steps.deinit(allocator);
    var metrics = Metrics.init(program);

    var schedule_index: usize = 0;
    while (schedule_index < program.schedule.len) {
        const item = program.schedule[schedule_index];
        switch (item.kind) {
            .node => {
                if (try dotGroupAt(allocator, program, schedule_index)) |group| {
                    errdefer {
                        allocator.free(group.node_indices);
                        allocator.free(group.release_values);
                    }
                    metrics.recordReleases(group.release_values);
                    metrics.planned_matmul_step_count += 1;
                    metrics.dot_group_step_count += 1;
                    metrics.dot_group_node_count += group.node_indices.len;
                    try steps.append(allocator, .{ .dot_group = group });
                    schedule_index += group.node_indices.len;
                    continue;
                }
                if (item.index >= program.nodes.len) return error.CommandSubmissionFailed;
                const node = program.nodes[item.index];
                const release_values = try release_lowering.releaseValuesAfterNode(allocator, program, item.index);
                errdefer allocator.free(release_values);
                metrics.recordReleases(release_values);
                metrics.recordPlannedInstructionStep(node.kind, .schedule_node);
                try steps.append(allocator, .{ .instruction = .{
                    .index = node.instruction_index,
                    .origin = .schedule_node,
                    .release_values = release_values,
                } });
            },
            .fusion_group => {
                if (item.index >= program.fusion_groups.len) return error.CommandSubmissionFailed;
                const group = program.fusion_groups[item.index];
                if (item.count != group.node_indices.len) return error.CommandSubmissionFailed;
                if (fusion_lowering.groupEligible(plan, program, group)) {
                    const release_values = try release_lowering.releaseValuesAfterFusionGroup(allocator, program, group);
                    errdefer allocator.free(release_values);
                    metrics.recordReleases(release_values);
                    try steps.append(allocator, .{ .fusion_group = .{
                        .index = item.index,
                        .release_values = release_values,
                    } });
                } else {
                    metrics.fallback_group_count += 1;
                    try appendSplitFusionGroup(allocator, plan, program, item.index, group, &steps, &metrics);
                }
            },
            .materialization_boundary => {},
        }
        schedule_index += 1;
    }

    return .{
        .steps = try steps.toOwnedSlice(allocator),
        .metrics = metrics,
    };
}
fn appendSplitFusionGroup(
    allocator: std.mem.Allocator,
    plan: *const ir.ExecutablePlan,
    program: *const program_mod.Program,
    group_index: usize,
    group: program_mod.FusionGroup,
    steps: *std.ArrayList(StepPlan),
    metrics: *Metrics,
) program_mod.Error!void {
    var start: usize = 0;
    while (start < group.node_indices.len) {
        if (try fusionSegmentAt(allocator, plan, program, group_index, group, start)) |segment| {
            errdefer {
                allocator.free(segment.node_indices);
                allocator.free(segment.input_values);
                allocator.free(segment.output_values);
                allocator.free(segment.release_values);
            }
            metrics.recordReleases(segment.release_values);
            try steps.append(allocator, .{ .fusion_segment = segment });
            start += segment.node_indices.len;
            continue;
        }

        const node_index = group.node_indices[start];
        if (node_index >= program.nodes.len) return error.CommandSubmissionFailed;
        const node = program.nodes[node_index];
        if (node.fusion_group != group_index) return error.CommandSubmissionFailed;
        const release_values = try release_lowering.releaseValuesAfterNode(allocator, program, node_index);
        errdefer allocator.free(release_values);
        metrics.recordReleases(release_values);
        metrics.recordPlannedInstructionStep(node.kind, .fusion_fallback_node);
        try steps.append(allocator, .{ .instruction = .{
            .index = node.instruction_index,
            .origin = .fusion_fallback_node,
            .release_values = release_values,
        } });
        start += 1;
    }
}

fn fusionSegmentAt(
    allocator: std.mem.Allocator,
    plan: *const ir.ExecutablePlan,
    program: *const program_mod.Program,
    group_index: usize,
    group: program_mod.FusionGroup,
    start: usize,
) program_mod.Error!?StepPlan.FusionSegment {
    const min_nodes = profiling.metalCppExecutableFusionMinNodes() orelse
        if (profiling.metalCppExecutableConservativeFusionEnabled()) fusion_lowering.ConservativeFusionPass.min_node_count else return null;
    if (start + min_nodes > group.node_indices.len) return null;

    var end = group.node_indices.len;
    while (end >= start + min_nodes) : (end -= 1) {
        const node_indices = try allocator.dupe(usize, group.node_indices[start..end]);
        var group_built = false;
        errdefer if (!group_built) allocator.free(node_indices);
        const segment_group = try fusionSegmentGroup(allocator, plan, program, group_index, group, node_indices);
        group_built = true;
        errdefer segment_group.deinit(allocator);
        if (!fusion_lowering.groupEligible(plan, program, segment_group)) {
            segment_group.deinit(allocator);
            continue;
        }
        const release_values = try release_lowering.releaseValuesAfterInstructionRange(allocator, plan, program, node_indices);
        errdefer allocator.free(release_values);
        return .{
            .group_index = group_index,
            .node_indices = node_indices,
            .input_values = segment_group.input_values,
            .output_values = segment_group.output_values,
            .release_values = release_values,
        };
    }
    return null;
}

fn fusionSegmentGroup(
    allocator: std.mem.Allocator,
    plan: *const ir.ExecutablePlan,
    program: *const program_mod.Program,
    group_index: usize,
    group: program_mod.FusionGroup,
    node_indices: []const usize,
) program_mod.Error!program_mod.FusionGroup {
    var input_values: std.ArrayList(ir.ValueId) = .empty;
    errdefer input_values.deinit(allocator);
    var output_values: std.ArrayList(ir.ValueId) = .empty;
    errdefer output_values.deinit(allocator);

    for (node_indices) |node_index| {
        if (node_index >= program.nodes.len) return error.CommandSubmissionFailed;
        const node = program.nodes[node_index];
        if (node.fusion_group != group_index) return error.CommandSubmissionFailed;
        if (node.instruction_index >= plan.instructions.len) return error.CommandSubmissionFailed;
        const instruction = plan.instructions[node.instruction_index];
        for (instruction.inputs) |value_id| {
            if (!valueProducedInNodes(program, value_id, node_indices)) {
                try appendUniqueValue(allocator, &input_values, value_id);
            }
        }
        for (instruction.outputs) |value_id| {
            if (valueEscapesNodes(program, group, value_id, node_indices)) {
                try appendUniqueValue(allocator, &output_values, value_id);
            }
        }
    }

    return .{
        .id = group.id,
        .kind = group.kind,
        .first_node = node_indices[0],
        .last_node = node_indices[node_indices.len - 1],
        .node_count = node_indices.len,
        .node_indices = node_indices,
        .input_values = try input_values.toOwnedSlice(allocator),
        .output_values = try output_values.toOwnedSlice(allocator),
    };
}

fn valueProducedInNodes(program: *const program_mod.Program, value_id: ir.ValueId, node_indices: []const usize) bool {
    if (value_id.index >= program.values.len) return false;
    const producer_index = program.values[value_id.index].producer_node orelse return false;
    return containsNode(node_indices, producer_index);
}

fn valueEscapesNodes(program: *const program_mod.Program, group: program_mod.FusionGroup, value_id: ir.ValueId, node_indices: []const usize) bool {
    if (containsValue(group.output_values, value_id)) return true;
    if (value_id.index >= program.values.len) return true;
    const value = program.values[value_id.index];
    if (value.is_output or value.materialization_boundary != null) return true;
    const last_use = value.last_use_node orelse return false;
    return !containsNode(node_indices, last_use);
}

fn appendUniqueValue(allocator: std.mem.Allocator, values: *std.ArrayList(ir.ValueId), value_id: ir.ValueId) !void {
    if (containsValue(values.items, value_id)) return;
    try values.append(allocator, value_id);
}

fn containsValue(values: []const ir.ValueId, value_id: ir.ValueId) bool {
    for (values) |candidate| {
        if (candidate.index == value_id.index) return true;
    }
    return false;
}

fn containsNode(node_indices: []const usize, node_index: usize) bool {
    for (node_indices) |candidate| {
        if (candidate == node_index) return true;
    }
    return false;
}

fn dotGroupAt(allocator: std.mem.Allocator, program: *const program_mod.Program, schedule_index: usize) program_mod.Error!?StepPlan.DotGroup {
    if (schedule_index >= program.schedule.len) return null;
    const first_item = program.schedule[schedule_index];
    if (first_item.kind != .node or first_item.index >= program.nodes.len) return null;
    const first_node = program.nodes[first_item.index];
    if (!dotGroupNode(first_node)) return null;
    const lhs = first_node.inputs[0];

    var count: usize = 1;
    while (schedule_index + count < program.schedule.len) : (count += 1) {
        const item = program.schedule[schedule_index + count];
        if (item.kind != .node or item.index >= program.nodes.len) break;
        const node = program.nodes[item.index];
        if (!dotGroupNode(node) or node.inputs[0].index != lhs.index) break;
    }
    if (count < 2) return null;

    const node_indices = try allocator.alloc(usize, count);
    errdefer allocator.free(node_indices);
    for (node_indices, 0..) |*node_index, offset| {
        node_index.* = program.schedule[schedule_index + offset].index;
    }
    const release_values = try release_lowering.releaseValuesAfterNodeRange(allocator, program, node_indices);
    errdefer allocator.free(release_values);
    return .{
        .node_indices = node_indices,
        .release_values = release_values,
    };
}

fn dotGroupNode(node: program_mod.Node) bool {
    return node.kind == .matmul and
        node.inputs.len == 2 and
        node.outputs.len == 1 and
        node.subprograms.len == 0 and
        node.control_flow == null and
        node.fusion_group == null;
}
