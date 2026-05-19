const std = @import("std");
const compiler_facts = @import("pjrtx/compiler/facts");
const core = @import("pjrtx/core");
const mlir_state = @import("pjrtx/compiler/mlir_state");
const runtime = @import("pjrtx/runtime");

pub fn commitAllocationPlan(
    allocator: std.mem.Allocator,
    session: *mlir_state.MlirSession,
    plan: runtime.AllocationPlan,
    diagnostics: *std.Io.Writer,
) !void {
    const allocations = try allocator.alloc(mlir_state.RuntimeAllocationFact, plan.allocations.len);
    defer allocator.free(allocations);
    const uses = try allocator.alloc(mlir_state.RuntimeBufferUseFact, plan.command_buffer_uses.len);
    defer allocator.free(uses);

    for (plan.allocations, 0..) |allocation, index| {
        const lifetime = lifetimeByBufferId(plan.lifetimes, allocation.id);
        allocations[index] = .{
            .index = allocation.id.index,
            .value_id = allocation.value_id,
            .placement = @tagName(allocation.placement),
            .memory_space_id = allocation.memory_space_id,
            .size_bytes = allocation.size_bytes,
            .first_command_id = lifetime.first_command_id,
            .last_command_id = lifetime.last_command_id,
        };
    }

    for (plan.command_buffer_uses, 0..) |use, index| {
        uses[index] = .{
            .command_id = use.command_id,
            .buffer_index = use.buffer_id.index,
            .access = @tagName(use.access),
        };
    }

    const fact: mlir_state.RuntimeAllocationPlanFact = .{
        .allocations = allocations,
        .command_buffer_uses = uses,
        .peak_device_bytes = plan.peak_device_bytes,
    };
    try mlir_state.commitRuntimeAllocationPlan(session, fact, diagnostics);
}

pub fn commitStreamPlan(
    allocator: std.mem.Allocator,
    session: *mlir_state.MlirSession,
    plan: runtime.StreamPlan,
    diagnostics: *std.Io.Writer,
) !void {
    const steps = try allocator.alloc(mlir_state.RuntimeStreamStepFact, plan.steps.len);
    defer allocator.free(steps);

    for (plan.steps, 0..) |step, index| {
        const wait_event_ids = try allocator.alloc(u32, step.wait_event_ids.len);
        errdefer allocator.free(wait_event_ids);
        for (step.wait_event_ids, 0..) |event_id, wait_index| {
            wait_event_ids[wait_index] = event_id.index;
        }
        steps[index] = .{
            .command_id = step.command_id,
            .stream = step.stream,
            .wait_event_ids = wait_event_ids,
            .start_event_id = step.start_event_id.index,
            .done_event_id = step.done_event_id.index,
        };
    }
    defer {
        for (steps) |step| allocator.free(step.wait_event_ids);
    }

    const fact: mlir_state.RuntimeStreamPlanFact = .{ .steps = steps };
    try mlir_state.commitRuntimeStreamPlan(session, fact, diagnostics);
}

pub fn commitProfile(
    allocator: std.mem.Allocator,
    session: *mlir_state.MlirSession,
    report: runtime.ProfiledTraceReport,
    diagnostics: *std.Io.Writer,
) !void {
    const events = try allocator.alloc(mlir_state.RuntimeProfileEventFact, report.report.profile_events.len);
    defer allocator.free(events);

    for (report.report.profile_events, 0..) |event, index| {
        events[index] = .{
            .index = event.id.index,
            .command_id = event.command_id,
            .graph_instruction_ids = event.graph_instruction_ids,
            .kind = @tagName(event.kind),
            .start_ns = event.start_ns,
            .duration_ns = event.duration_ns,
            .bytes = event.bytes,
            .logical_ops = event.logical_ops,
            .status = @tagName(event.status),
            .forced_synchronization = event.forced_synchronization,
        };
    }

    const fact: mlir_state.RuntimeProfileFact = .{ .events = events };
    try mlir_state.commitRuntimeProfile(session, fact, diagnostics);
}

pub fn commitProfileJoins(
    allocator: std.mem.Allocator,
    session: *mlir_state.MlirSession,
    report: runtime.ProfiledTraceReport,
    diagnostics: *std.Io.Writer,
) !void {
    var joins: std.ArrayList(mlir_state.RuntimeProfileJoinFact) = .empty;
    defer joins.deinit(allocator);

    var owned_event_id_slices: std.ArrayList([]core.ProfileEventId) = .empty;
    defer {
        for (owned_event_id_slices.items) |ids| allocator.free(ids);
        owned_event_id_slices.deinit(allocator);
    }

    for (report.report.schedule_commands) |command| {
        const event = profileEventForCommand(report.report.profile_events, command.id) orelse continue;
        const event_ids = try appendOwnedSingleEventId(allocator, &owned_event_id_slices, event.id);
        try appendProfileJoin(&joins, allocator, .{
            .subject_kind = "schedule_command",
            .subject_id = command.id.index,
            .command_id = command.id,
            .graph_instruction_ids = event.graph_instruction_ids,
            .profile_event_ids = event_ids,
        });
    }

    for (report.report.schedule_commands) |command| {
        if (command.kind != .backend_execute) continue;
        for (command.lowering_record_ids) |lowering_id| {
            const lowering = loweringRecord(report.report, lowering_id);
            const event = profileEventForLowering(report.report.profile_events, command.id, lowering.graph_instruction_ids) orelse continue;
            const event_ids = try appendOwnedSingleEventId(allocator, &owned_event_id_slices, event.id);
            try appendProfileJoin(&joins, allocator, .{
                .subject_kind = "lowering_record",
                .subject_id = lowering.id.index,
                .command_id = command.id,
                .graph_instruction_ids = lowering.graph_instruction_ids,
                .profile_event_ids = event_ids,
            });
        }
    }

    for (report.report.explain_records) |explain| {
        if (explain.profile_event_ids.len == 0) continue;
        const event = profileEventById(report.report.profile_events, explain.profile_event_ids[0]);
        try appendProfileJoin(&joins, allocator, .{
            .subject_kind = explainSubjectKind(explain.subject),
            .subject_id = explainSubjectId(explain.subject),
            .command_id = if (event) |profile_event| profile_event.command_id else null,
            .graph_instruction_ids = if (event) |profile_event| profile_event.graph_instruction_ids else &.{},
            .profile_event_ids = explain.profile_event_ids,
        });
    }

    const fact: mlir_state.RuntimeProfileJoinPlanFact = .{ .joins = joins.items };
    try mlir_state.commitRuntimeProfileJoins(session, fact, diagnostics);
}

fn lifetimeByBufferId(lifetimes: []const runtime.BufferLifetime, buffer_id: runtime.RuntimeBufferId) runtime.BufferLifetime {
    for (lifetimes) |lifetime| {
        if (lifetime.buffer_id.eql(buffer_id)) return lifetime;
    }
    unreachable;
}

const ProfileJoinInput = struct {
    subject_kind: []const u8,
    subject_id: u32,
    command_id: ?core.ScheduleCommandId,
    graph_instruction_ids: []const compiler_facts.GraphInstructionId,
    profile_event_ids: []const core.ProfileEventId,
};

fn appendProfileJoin(
    joins: *std.ArrayList(mlir_state.RuntimeProfileJoinFact),
    allocator: std.mem.Allocator,
    input: ProfileJoinInput,
) !void {
    const index = std.math.cast(u32, joins.items.len) orelse return error.OutOfMemory;
    try joins.append(allocator, .{
        .index = index,
        .subject_kind = input.subject_kind,
        .subject_id = input.subject_id,
        .command_id = input.command_id,
        .graph_instruction_ids = input.graph_instruction_ids,
        .profile_event_ids = input.profile_event_ids,
    });
}

fn appendOwnedSingleEventId(
    allocator: std.mem.Allocator,
    owned_event_id_slices: *std.ArrayList([]core.ProfileEventId),
    id: core.ProfileEventId,
) ![]core.ProfileEventId {
    const ids = try allocator.alloc(core.ProfileEventId, 1);
    errdefer allocator.free(ids);
    ids[0] = id;
    try owned_event_id_slices.append(allocator, ids);
    return ids;
}

fn profileEventForCommand(profile_events: []const core.ProfileEvent, command_id: core.ScheduleCommandId) ?core.ProfileEvent {
    for (profile_events) |event| {
        if (event.command_id) |event_command_id| {
            if (event_command_id.eql(command_id)) return event;
        }
    }
    return null;
}

fn profileEventForLowering(
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

fn profileEventById(profile_events: []const core.ProfileEvent, id: core.ProfileEventId) ?core.ProfileEvent {
    for (profile_events) |event| {
        if (event.id.eql(id)) return event;
    }
    return null;
}

fn loweringRecord(report: core.TraceReport, id: compiler_facts.LoweringRecordId) compiler_facts.LoweringRecord {
    const index: usize = std.math.cast(usize, id.index) orelse unreachable;
    return report.lowering_records[index];
}

fn graphInstructionIdsEqual(lhs: []const compiler_facts.GraphInstructionId, rhs: []const compiler_facts.GraphInstructionId) bool {
    if (lhs.len != rhs.len) return false;
    for (lhs, rhs) |left, right| {
        if (!left.eql(right)) return false;
    }
    return true;
}

fn explainSubjectKind(subject: core.ExplainSubject) []const u8 {
    return switch (subject) {
        .graph_instruction => "graph_instruction",
        .lowering_record => "lowering_record",
        .schedule_command => "schedule_command",
        .backend_binding => "backend_binding",
    };
}

fn explainSubjectId(subject: core.ExplainSubject) u32 {
    return switch (subject) {
        .graph_instruction => |id| id.index,
        .lowering_record => |id| id.index,
        .schedule_command => |id| id.index,
        .backend_binding => |id| id.index,
    };
}
