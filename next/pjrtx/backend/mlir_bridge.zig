const std = @import("std");
const backend = @import("pjrtx/backend");
const compiler_facts = @import("pjrtx/compiler/facts");
const mlir_state = @import("pjrtx/compiler/mlir_state");
const core = @import("pjrtx/core");

pub fn commitExecutablePlan(
    allocator: std.mem.Allocator,
    session: *mlir_state.MlirSession,
    plan: backend.BackendExecutablePlan,
    diagnostics: *std.Io.Writer,
) !void {
    const calls = try allocator.alloc(mlir_state.BackendExecutableCallFact, plan.calls.len);
    defer allocator.free(calls);

    for (plan.calls, 0..) |call, index| {
        calls[index] = .{
            .index = call.index,
            .command_id = plan.command_id,
            .backend_kind = plan.backend_kind,
            .graph_instruction_id = call.graph_instruction_id,
            .graph_instruction_ids = call.graph_instruction_ids,
            .feature = @tagName(call.feature),
            .backend_operation = call.backend_operation,
            .input_value_ids = call.input_value_ids,
            .output_value_ids = call.output_value_ids,
            .expected_unit_id = call.expected_unit_id,
        };
    }

    const fact: mlir_state.BackendExecutablePlanFact = .{
        .backend_kind = plan.backend_kind,
        .command_id = plan.command_id,
        .backend_operation = plan.backend_operation,
        .calls = calls,
    };
    try mlir_state.commitBackendExecutablePlan(session, fact, diagnostics);
}

pub fn commitKernelGraph(
    allocator: std.mem.Allocator,
    session: *mlir_state.MlirSession,
    graph: backend.BackendKernelGraphPlan,
    diagnostics: *std.Io.Writer,
) !void {
    const nodes = try allocator.alloc(mlir_state.BackendKernelGraphNodeFact, graph.nodes.len);
    defer allocator.free(nodes);
    const edges = try allocator.alloc(mlir_state.BackendKernelGraphEdgeFact, graph.edges.len);
    defer allocator.free(edges);

    for (graph.nodes, 0..) |node, index| {
        nodes[index] = .{
            .index = node.id.index,
            .call_index = node.call_index,
            .graph_instruction_id = node.graph_instruction_id,
            .graph_instruction_ids = node.graph_instruction_ids,
            .feature = @tagName(node.feature),
            .backend_operation = node.backend_operation,
            .input_value_ids = node.input_value_ids,
            .output_value_ids = node.output_value_ids,
            .output_type = .{
                .element_type = node.output_type.element_type,
                .dims = node.output_type.dims,
                .layout = node.output_type.layout,
            },
            .attributes = kernelAttributesText(node.attributes),
        };
    }

    for (graph.edges, 0..) |edge, index| {
        edges[index] = .{
            .value_id = edge.value_id,
            .src_node_index = edge.src_node_id.index,
            .dst_node_index = edge.dst_node_id.index,
        };
    }

    const fact: mlir_state.BackendKernelGraphFact = .{
        .backend_kind = graph.backend_kind,
        .command_id = graph.command_id,
        .nodes = nodes,
        .edges = edges,
    };
    try mlir_state.commitBackendKernelGraph(session, fact, diagnostics);
}

pub fn commitProfileJoins(
    allocator: std.mem.Allocator,
    session: *mlir_state.MlirSession,
    executable: backend.BackendExecutablePlan,
    report: core.TraceReport,
    diagnostics: *std.Io.Writer,
) !void {
    var joins: std.ArrayList(mlir_state.BackendProfileJoinFact) = .empty;
    defer joins.deinit(allocator);

    for (executable.calls) |call| {
        const event = profileEventForCall(report.profile_events, executable.command_id, call.graph_instruction_ids) orelse continue;
        const index = std.math.cast(u32, joins.items.len) orelse return error.OutOfMemory;
        try joins.append(allocator, .{
            .index = index,
            .call_index = call.index,
            .command_id = executable.command_id,
            .graph_instruction_ids = call.graph_instruction_ids,
            .profile_event_id = event.id,
        });
    }

    const fact: mlir_state.BackendProfileJoinPlanFact = .{ .joins = joins.items };
    try mlir_state.commitBackendProfileJoins(session, fact, diagnostics);
}

fn kernelAttributesText(attributes: backend.BackendKernelAttributes) []const u8 {
    return switch (attributes) {
        .rank2_dot_general => "rank2_dot_general",
        .broadcast_in_dim => "broadcast_in_dim",
        .add => "add",
        .tanh => "tanh",
        .elementwise_fusion => "elementwise_fusion",
    };
}

fn profileEventForCall(
    profile_events: []const core.ProfileEvent,
    command_id: core.ScheduleCommandId,
    instruction_ids: []const compiler_facts.GraphInstructionId,
) ?core.ProfileEvent {
    for (profile_events) |event| {
        if (event.command_id == null or !event.command_id.?.eql(command_id)) continue;
        if (graphInstructionIdsEqual(event.graph_instruction_ids, instruction_ids)) return event;
    }
    return null;
}

fn graphInstructionIdsEqual(lhs: []const compiler_facts.GraphInstructionId, rhs: []const compiler_facts.GraphInstructionId) bool {
    if (lhs.len != rhs.len) return false;
    for (lhs, rhs) |left, right| {
        if (!left.eql(right)) return false;
    }
    return true;
}
