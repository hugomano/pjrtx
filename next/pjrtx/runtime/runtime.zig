const std = @import("std");
const compiler_facts = @import("pjrtx/compiler/facts");
const core = @import("pjrtx/core");
const target_pkg = @import("pjrtx/target");

pub const RuntimeCommandKind = core.CommandKind;

pub const RuntimeError = error{
    InvalidSchedule,
};

pub const RuntimeBufferId = struct {
    index: u32,

    pub fn eql(self: RuntimeBufferId, other: RuntimeBufferId) bool {
        return self.index == other.index;
    }
};

pub const RuntimeEventId = struct {
    index: u32,

    pub fn eql(self: RuntimeEventId, other: RuntimeEventId) bool {
        return self.index == other.index;
    }
};

pub const BufferPlacement = enum {
    host,
    device,
};

pub const BufferAccess = enum {
    read,
    write,
};

pub const BufferAllocation = struct {
    id: RuntimeBufferId,
    value_id: compiler_facts.GraphValueId,
    placement: BufferPlacement,
    memory_space_id: u32,
    size_bytes: u128,
};

pub const CommandBufferUse = struct {
    command_id: core.ScheduleCommandId,
    buffer_id: RuntimeBufferId,
    access: BufferAccess,
};

pub const BufferLifetime = struct {
    buffer_id: RuntimeBufferId,
    first_command_id: core.ScheduleCommandId,
    last_command_id: core.ScheduleCommandId,
};

pub const AllocationPlan = struct {
    allocator: std.mem.Allocator,
    allocations: []BufferAllocation,
    command_buffer_uses: []CommandBufferUse,
    lifetimes: []BufferLifetime,
    peak_device_bytes: u128,

    pub fn deinit(self: *AllocationPlan) void {
        self.allocator.free(self.lifetimes);
        self.allocator.free(self.command_buffer_uses);
        self.allocator.free(self.allocations);
        self.* = undefined;
    }
};

pub const StreamStep = struct {
    command_id: core.ScheduleCommandId,
    stream: core.StreamId,
    wait_event_ids: []const RuntimeEventId,
    start_event_id: RuntimeEventId,
    done_event_id: RuntimeEventId,
};

pub const StreamPlan = struct {
    allocator: std.mem.Allocator,
    steps: []StreamStep,

    pub fn deinit(self: *StreamPlan) void {
        for (self.steps) |step| {
            self.allocator.free(step.wait_event_ids);
        }
        self.allocator.free(self.steps);
        self.* = undefined;
    }
};

pub const ProfiledTraceReport = struct {
    allocator: std.mem.Allocator,
    report: core.TraceReport,

    pub fn deinit(self: *ProfiledTraceReport) void {
        for (self.report.profile_events) |event| {
            self.allocator.free(event.graph_instruction_ids);
        }
        self.allocator.free(self.report.profile_events);
        for (self.report.explain_records) |explain| {
            self.allocator.free(explain.profile_event_ids);
        }
        self.allocator.free(self.report.explain_records);
        self.* = undefined;
    }
};

pub fn writeCommandKind(writer: *std.Io.Writer, kind: RuntimeCommandKind) std.Io.Writer.Error!void {
    try writer.writeAll(@tagName(kind));
}

/// Allocation planning is a runtime-owned contract between the verified
/// schedule and future device allocators. V0 allocates values exposed at
/// schedule and lowering-region boundaries, while fused-internal graph values
/// remain provenance facts rather than standalone device buffers.
pub fn planAllocations(
    allocator: std.mem.Allocator,
    report: core.TraceReport,
    diagnostics: *std.Io.Writer,
) !AllocationPlan {
    core.validateTraceReport(report, diagnostics) catch {
        try diagnostics.writeAll("pass=runtime-allocation feature=trace-report reason=trace report failed validation\n");
        return RuntimeError.InvalidSchedule;
    };
    try verifyRuntimeBindings(report, diagnostics);
    try verifyRuntimeDependencies(report, diagnostics);

    const target = report.target orelse {
        try diagnostics.writeAll("pass=runtime-allocation feature=target reason=allocation planning requires a target description\n");
        return RuntimeError.InvalidSchedule;
    };
    const host_memory_space = try selectMemorySpace(target, .host, diagnostics);
    const device_memory_space = try selectMemorySpace(target, .device, diagnostics);

    var builder: AllocationPlanBuilder = .{
        .allocator = allocator,
        .report = report,
        .host_memory_space = host_memory_space,
        .device_memory_space = device_memory_space,
        .diagnostics = diagnostics,
    };
    errdefer builder.deinitPartial();

    for (report.schedule_commands) |command| {
        switch (command.kind) {
            .host_to_device => {
                for (command.inputs) |value_id| _ = try builder.ensureAllocation(value_id, .host);
                for (command.outputs) |value_id| _ = try builder.ensureAllocation(value_id, .device);
            },
            .device_to_host => {
                for (command.inputs) |value_id| _ = try builder.ensureAllocation(value_id, .device);
                for (command.outputs) |value_id| _ = try builder.ensureAllocation(value_id, .host);
            },
            .backend_execute, .event_record, .event_wait => {},
        }
    }

    try builder.addCommandUses();
    var plan: AllocationPlan = try builder.finish();
    errdefer plan.deinit();
    try verifyAllocationPlanFitsTarget(plan, target, diagnostics);
    try verifyTransferCommandsHaveEdges(report, plan, target, diagnostics);
    return plan;
}

/// Stream planning turns schedule dependencies into explicit event waits. V0
/// still uses the compiler-provided streams, but it refuses future dependencies
/// so accidental synchronization cannot hide an invalid command graph.
pub fn planStreams(
    allocator: std.mem.Allocator,
    report: core.TraceReport,
    diagnostics: *std.Io.Writer,
) !StreamPlan {
    core.validateTraceReport(report, diagnostics) catch {
        try diagnostics.writeAll("pass=runtime-streams feature=trace-report reason=trace report failed validation\n");
        return RuntimeError.InvalidSchedule;
    };
    try verifyRuntimeBindings(report, diagnostics);
    try verifyRuntimeDependencies(report, diagnostics);

    var steps: std.ArrayList(StreamStep) = .empty;
    errdefer {
        for (steps.items) |step| allocator.free(step.wait_event_ids);
        steps.deinit(allocator);
    }

    for (report.schedule_commands, 0..) |command, index| {
        try verifyCommandDependenciesAreEarlier(command, diagnostics);
        const wait_event_ids = try allocator.alloc(RuntimeEventId, command.dependencies.len);
        var wait_event_ids_owned = true;
        errdefer if (wait_event_ids_owned) allocator.free(wait_event_ids);
        for (command.dependencies, 0..) |dependency, dependency_index| {
            wait_event_ids[dependency_index] = doneEventForCommand(dependency.command_id);
        }
        const command_index: u32 = std.math.cast(u32, index) orelse unreachable;
        try steps.append(allocator, .{
            .command_id = command.id,
            .stream = command.stream,
            .wait_event_ids = wait_event_ids,
            .start_event_id = startEventForCommand(command.id),
            .done_event_id = doneEventForCommand(command.id),
        });
        wait_event_ids_owned = false;
        if (command.id.index != command_index) {
            try diagnostics.print("pass=runtime-streams feature=command reason=command ID order mismatch expected={d} actual={d}\n", .{ command_index, command.id.index });
            return RuntimeError.InvalidSchedule;
        }
    }
    return .{ .allocator = allocator, .steps = try steps.toOwnedSlice(allocator) };
}

/// This runtime path is intentionally synthetic: it proves that a verified
/// schedule can produce command-linked profile records without introducing a
/// reference fallback or pretending unsupported work executed.
pub fn executeSyntheticProfile(
    allocator: std.mem.Allocator,
    report: core.TraceReport,
    diagnostics: *std.Io.Writer,
) !ProfiledTraceReport {
    core.validateTraceReport(report, diagnostics) catch {
        try diagnostics.writeAll("pass=runtime-validate feature=trace-report reason=trace report failed validation\n");
        return RuntimeError.InvalidSchedule;
    };

    try verifyRuntimeBindings(report, diagnostics);
    try verifyRuntimeDependencies(report, diagnostics);

    var events: std.ArrayList(core.ProfileEvent) = .empty;
    errdefer {
        for (events.items) |event| allocator.free(event.graph_instruction_ids);
        events.deinit(allocator);
    }

    for (report.schedule_commands) |command| {
        try appendSyntheticCommandProfileEvent(allocator, report, command, &events);
        if (command.kind == .backend_execute) {
            for (command.lowering_record_ids) |lowering_id| {
                try appendSyntheticLoweringProfileEvent(allocator, report, command, loweringRecord(report, lowering_id), &events);
            }
        }
    }

    const profile_events = try events.toOwnedSlice(allocator);
    errdefer {
        for (profile_events) |event| allocator.free(event.graph_instruction_ids);
        allocator.free(profile_events);
    }
    const explain_records = try profiledExplainRecords(allocator, report, profile_events);
    errdefer freeProfiledExplainRecords(allocator, explain_records);
    return .{
        .allocator = allocator,
        .report = .{
            .sources = report.sources,
            .target = report.target,
            .graph_values = report.graph_values,
            .graph_instructions = report.graph_instructions,
            .mlir_pass_records = report.mlir_pass_records,
            .fusion_groups = report.fusion_groups,
            .placement_records = report.placement_records,
            .collective_plan_records = report.collective_plan_records,
            .cost_ledger = report.cost_ledger,
            .lowering_records = report.lowering_records,
            .memory_traffic_records = report.memory_traffic_records,
            .schedule_overlap_records = report.schedule_overlap_records,
            .schedule_commands = report.schedule_commands,
            .kernel_codegen_records = report.kernel_codegen_records,
            .backend_bindings = report.backend_bindings,
            .profile_events = profile_events,
            .explain_records = explain_records,
        },
    };
}

fn verifyRuntimeBindings(report: core.TraceReport, diagnostics: *std.Io.Writer) !void {
    for (report.schedule_commands) |command| {
        if (command.kind != .backend_execute) continue;
        var count: u32 = 0;
        for (report.backend_bindings) |binding| {
            if (binding.command_id.eql(command.id)) count += 1;
        }
        if (count != 1) {
            try diagnostics.print("pass=runtime-validate feature=backend-binding reason=backend command must have exactly one binding command={d} bindings={d}\n", .{ command.id.index, count });
            return RuntimeError.InvalidSchedule;
        }
    }
}

fn verifyRuntimeDependencies(report: core.TraceReport, diagnostics: *std.Io.Writer) !void {
    for (report.schedule_commands) |command| {
        try verifyCommandDependenciesAreEarlier(command, diagnostics);
    }
}

fn verifyCommandDependenciesAreEarlier(command: core.ScheduleCommand, diagnostics: *std.Io.Writer) !void {
    for (command.dependencies) |dependency| {
        if (dependency.command_id.index >= command.id.index) {
            try diagnostics.print(
                "pass=runtime-streams feature=dependency reason=command dependency must be earlier command={d} dependency={d}\n",
                .{ command.id.index, dependency.command_id.index },
            );
            return RuntimeError.InvalidSchedule;
        }
    }
}

fn verifyAllocationPlanFitsTarget(
    plan: AllocationPlan,
    target: target_pkg.TargetDescription,
    diagnostics: *std.Io.Writer,
) !void {
    for (target.memory_spaces) |memory_space| {
        const capacity = memory_space.capacity_bytes orelse continue;
        const peak_live_bytes = peakLiveBytesForMemory(plan, memory_space.id);
        if (peak_live_bytes > capacity) {
            try diagnostics.print(
                "pass=runtime-allocation feature=memory-capacity reason=peak live bytes exceed memory capacity memory={d} peak_live_bytes={d} capacity_bytes={d}\n",
                .{ memory_space.id, peak_live_bytes, capacity },
            );
            return RuntimeError.InvalidSchedule;
        }
    }
}

fn verifyTransferCommandsHaveEdges(
    report: core.TraceReport,
    plan: AllocationPlan,
    target: target_pkg.TargetDescription,
    diagnostics: *std.Io.Writer,
) !void {
    for (report.schedule_commands) |command| {
        switch (command.kind) {
            .host_to_device, .device_to_host => {
                if (transferMemoryPairForCommand(plan, command)) |memory_pair| {
                    if (transferEdgeForMemoryPair(target, memory_pair) != null) continue;
                    try diagnostics.print(
                        "pass=runtime-allocation feature=transfer-edge reason=transfer command has no target edge command={d} src_memory={d} dst_memory={d}\n",
                        .{ command.id.index, memory_pair.src, memory_pair.dst },
                    );
                    return RuntimeError.InvalidSchedule;
                }
                try diagnostics.print(
                    "pass=runtime-allocation feature=transfer-edge reason=transfer command has no allocated endpoint command={d}\n",
                    .{command.id.index},
                );
                return RuntimeError.InvalidSchedule;
            },
            .backend_execute, .event_record, .event_wait => {},
        }
    }
}

fn profileEventForCommand(profile_events: []const core.ProfileEvent, command_id: core.ScheduleCommandId) ?core.ProfileEvent {
    for (profile_events) |event| {
        if (event.command_id) |event_command_id| {
            if (event_command_id.eql(command_id)) return event;
        }
    }
    return null;
}

fn profiledExplainRecords(
    allocator: std.mem.Allocator,
    report: core.TraceReport,
    profile_events: []const core.ProfileEvent,
) ![]const core.ExplainRecord {
    const records = try allocator.alloc(core.ExplainRecord, report.explain_records.len);
    errdefer allocator.free(records);
    for (report.explain_records, 0..) |explain, index| {
        const profile_event_ids = try profileEventIdsForExplain(allocator, report, profile_events, explain);
        errdefer allocator.free(profile_event_ids);
        records[index] = .{
            .id = explain.id,
            .pass_name = explain.pass_name,
            .subject = explain.subject,
            .decision = explain.decision,
            .reason = explain.reason,
            .source_refs = explain.source_refs,
            .cost_ledger_ids = explain.cost_ledger_ids,
            .profile_event_ids = profile_event_ids,
        };
    }
    return records;
}

fn freeProfiledExplainRecords(allocator: std.mem.Allocator, records: []const core.ExplainRecord) void {
    for (records) |record| {
        allocator.free(record.profile_event_ids);
    }
    allocator.free(records);
}

fn profileEventIdsForExplain(
    allocator: std.mem.Allocator,
    report: core.TraceReport,
    profile_events: []const core.ProfileEvent,
    explain: core.ExplainRecord,
) ![]const core.ProfileEventId {
    var ids: std.ArrayList(core.ProfileEventId) = .empty;
    errdefer ids.deinit(allocator);

    switch (explain.subject) {
        .lowering_record => |lowering_id| {
            const lowering = loweringRecord(report, lowering_id);
            for (profile_events) |event| {
                if (graphInstructionIdsEqual(event.graph_instruction_ids, lowering.graph_instruction_ids)) {
                    try ids.append(allocator, event.id);
                }
            }
        },
        .backend_binding => |binding_id| {
            const binding_index: usize = std.math.cast(usize, binding_id.index) orelse unreachable;
            const binding = report.backend_bindings[binding_index];
            for (profile_events) |event| {
                if (event.command_id == null or !event.command_id.?.eql(binding.command_id)) continue;
                if (graphInstructionIdsEqual(event.graph_instruction_ids, binding.graph_instruction_ids)) {
                    try ids.append(allocator, event.id);
                }
            }
        },
        .schedule_command => |command_id| {
            for (profile_events) |event| {
                if (event.command_id) |event_command_id| {
                    if (event_command_id.eql(command_id)) try ids.append(allocator, event.id);
                }
            }
        },
        .graph_instruction => |instruction_id| {
            for (profile_events) |event| {
                if (instructionIdIn(instruction_id, event.graph_instruction_ids)) try ids.append(allocator, event.id);
            }
        },
    }

    return ids.toOwnedSlice(allocator);
}

fn startEventForCommand(command_id: core.ScheduleCommandId) RuntimeEventId {
    return .{ .index = command_id.index * 2 };
}

fn doneEventForCommand(command_id: core.ScheduleCommandId) RuntimeEventId {
    return .{ .index = command_id.index * 2 + 1 };
}

const AllocationPlanBuilder = struct {
    allocator: std.mem.Allocator,
    report: core.TraceReport,
    host_memory_space: u32,
    device_memory_space: u32,
    diagnostics: *std.Io.Writer,
    allocations: std.ArrayList(BufferAllocation) = .empty,
    command_buffer_uses: std.ArrayList(CommandBufferUse) = .empty,

    fn deinitPartial(self: *AllocationPlanBuilder) void {
        self.command_buffer_uses.deinit(self.allocator);
        self.allocations.deinit(self.allocator);
    }

    fn ensureAllocation(self: *AllocationPlanBuilder, value_id: compiler_facts.GraphValueId, placement: BufferPlacement) !RuntimeBufferId {
        if (self.findAllocation(value_id, placement)) |existing| return existing;

        const value = try graphValue(self.report, value_id, self.diagnostics);
        const id: RuntimeBufferId = .{ .index = std.math.cast(u32, self.allocations.items.len) orelse unreachable };
        try self.allocations.append(self.allocator, .{
            .id = id,
            .value_id = value_id,
            .placement = placement,
            .memory_space_id = switch (placement) {
                .host => self.host_memory_space,
                .device => self.device_memory_space,
            },
            .size_bytes = tensorBytes(value.ty),
        });
        return id;
    }

    fn addCommandUses(self: *AllocationPlanBuilder) !void {
        for (self.report.schedule_commands) |command| {
            switch (command.kind) {
                .host_to_device => {
                    for (command.inputs) |value_id| try self.addUse(command.id, try self.ensureAllocation(value_id, .host), .read);
                    for (command.outputs) |value_id| try self.addUse(command.id, try self.ensureAllocation(value_id, .device), .write);
                },
                .backend_execute => {
                    try self.addBackendCommandUses(command);
                },
                .device_to_host => {
                    for (command.inputs) |value_id| try self.addUse(command.id, try self.ensureAllocation(value_id, .device), .read);
                    for (command.outputs) |value_id| try self.addUse(command.id, try self.ensureAllocation(value_id, .host), .write);
                },
                .event_record, .event_wait => {},
            }
        }
    }

    fn addUse(self: *AllocationPlanBuilder, command_id: core.ScheduleCommandId, buffer_id: RuntimeBufferId, access: BufferAccess) !void {
        try self.command_buffer_uses.append(self.allocator, .{
            .command_id = command_id,
            .buffer_id = buffer_id,
            .access = access,
        });
    }

    fn findAllocation(self: *AllocationPlanBuilder, value_id: compiler_facts.GraphValueId, placement: BufferPlacement) ?RuntimeBufferId {
        for (self.allocations.items) |allocation| {
            if (allocation.value_id.eql(value_id) and allocation.placement == placement) return allocation.id;
        }
        return null;
    }

    fn finish(self: *AllocationPlanBuilder) !AllocationPlan {
        const allocations = try self.allocations.toOwnedSlice(self.allocator);
        errdefer self.allocator.free(allocations);
        const command_buffer_uses = try self.command_buffer_uses.toOwnedSlice(self.allocator);
        errdefer self.allocator.free(command_buffer_uses);
        const lifetimes = try bufferLifetimes(self.allocator, allocations, command_buffer_uses, self.diagnostics);
        errdefer self.allocator.free(lifetimes);
        return .{
            .allocator = self.allocator,
            .allocations = allocations,
            .command_buffer_uses = command_buffer_uses,
            .lifetimes = lifetimes,
            .peak_device_bytes = peakLiveDeviceBytes(allocations, lifetimes),
        };
    }

    fn addBackendCommandUses(self: *AllocationPlanBuilder, command: core.ScheduleCommand) !void {
        for (command.lowering_record_ids) |lowering_id| {
            const lowering = try self.loweringRecord(lowering_id);
            switch (lowering.decision) {
                .backend_kernel_graph => try self.addSingleInstructionLoweringUses(command.id, lowering),
                .elementwise_fusion => try self.addFusedLoweringUses(command.id, lowering),
                .transfer, .unsupported => {
                    try self.diagnostics.print(
                        "pass=runtime-allocation feature=lowering reason=backend command references non-executable lowering command={d} lowering={d} decision={s}\n",
                        .{ command.id.index, lowering.id.index, @tagName(lowering.decision) },
                    );
                    return RuntimeError.InvalidSchedule;
                },
            }
        }
    }

    fn addSingleInstructionLoweringUses(self: *AllocationPlanBuilder, command_id: core.ScheduleCommandId, lowering: compiler_facts.LoweringRecord) !void {
        if (lowering.graph_instruction_ids.len != 1) {
            try self.diagnostics.print(
                "pass=runtime-allocation feature=lowering reason=non-fused lowering must contain one instruction lowering={d} instructions={d}\n",
                .{ lowering.id.index, lowering.graph_instruction_ids.len },
            );
            return RuntimeError.InvalidSchedule;
        }
        const instruction = try self.graphInstruction(lowering.graph_instruction_ids[0]);
        for (instruction.inputs) |value_id| try self.addUse(command_id, try self.ensureAllocation(value_id, .device), .read);
        for (instruction.outputs) |value_id| try self.addUse(command_id, try self.ensureAllocation(value_id, .device), .write);
    }

    fn addFusedLoweringUses(self: *AllocationPlanBuilder, command_id: core.ScheduleCommandId, lowering: compiler_facts.LoweringRecord) !void {
        if (lowering.graph_instruction_ids.len == 0) {
            try self.diagnostics.print("pass=runtime-allocation feature=lowering reason=fused lowering has no instructions lowering={d}\n", .{lowering.id.index});
            return RuntimeError.InvalidSchedule;
        }

        var read_values: std.ArrayList(compiler_facts.GraphValueId) = .empty;
        defer read_values.deinit(self.allocator);
        var write_values: std.ArrayList(compiler_facts.GraphValueId) = .empty;
        defer write_values.deinit(self.allocator);

        for (lowering.graph_instruction_ids) |instruction_id| {
            const instruction = try self.graphInstruction(instruction_id);
            for (instruction.inputs) |value_id| {
                if (self.regionProducesValue(lowering.graph_instruction_ids, value_id)) continue;
                try appendUniqueValueId(self.allocator, &read_values, value_id);
            }
            for (instruction.outputs) |value_id| {
                if (self.regionConsumesValue(lowering.graph_instruction_ids, value_id)) continue;
                try appendUniqueValueId(self.allocator, &write_values, value_id);
            }
        }

        if (read_values.items.len == 0 or write_values.items.len == 0) {
            try self.diagnostics.print(
                "pass=runtime-allocation feature=fusion reason=fused lowering must expose external reads and writes lowering={d} reads={d} writes={d}\n",
                .{ lowering.id.index, read_values.items.len, write_values.items.len },
            );
            return RuntimeError.InvalidSchedule;
        }

        for (read_values.items) |value_id| try self.addUse(command_id, try self.ensureAllocation(value_id, .device), .read);
        for (write_values.items) |value_id| try self.addUse(command_id, try self.ensureAllocation(value_id, .device), .write);
    }

    fn loweringRecord(self: *AllocationPlanBuilder, id: compiler_facts.LoweringRecordId) !compiler_facts.LoweringRecord {
        const index: usize = std.math.cast(usize, id.index) orelse {
            try self.diagnostics.print("pass=runtime-allocation feature=lowering reason=lowering id does not fit host index lowering={d}\n", .{id.index});
            return RuntimeError.InvalidSchedule;
        };
        if (index >= self.report.lowering_records.len) {
            try self.diagnostics.print("pass=runtime-allocation feature=lowering reason=lowering id is out of bounds lowering={d}\n", .{id.index});
            return RuntimeError.InvalidSchedule;
        }
        return self.report.lowering_records[index];
    }

    fn graphInstruction(self: *AllocationPlanBuilder, id: compiler_facts.GraphInstructionId) !compiler_facts.GraphInstruction {
        const index: usize = std.math.cast(usize, id.index) orelse {
            try self.diagnostics.print("pass=runtime-allocation feature=instruction reason=instruction id does not fit host index instruction={d}\n", .{id.index});
            return RuntimeError.InvalidSchedule;
        };
        if (index >= self.report.graph_instructions.len) {
            try self.diagnostics.print("pass=runtime-allocation feature=instruction reason=instruction id is out of bounds instruction={d}\n", .{id.index});
            return RuntimeError.InvalidSchedule;
        }
        return self.report.graph_instructions[index];
    }

    fn regionProducesValue(self: *AllocationPlanBuilder, instruction_ids: []const compiler_facts.GraphInstructionId, value_id: compiler_facts.GraphValueId) bool {
        for (instruction_ids) |instruction_id| {
            const index: usize = std.math.cast(usize, instruction_id.index) orelse unreachable;
            const instruction = self.report.graph_instructions[index];
            for (instruction.outputs) |output_id| {
                if (output_id.eql(value_id)) return true;
            }
        }
        return false;
    }

    fn regionConsumesValue(self: *AllocationPlanBuilder, instruction_ids: []const compiler_facts.GraphInstructionId, value_id: compiler_facts.GraphValueId) bool {
        for (instruction_ids) |instruction_id| {
            const index: usize = std.math.cast(usize, instruction_id.index) orelse unreachable;
            const instruction = self.report.graph_instructions[index];
            for (instruction.inputs) |input_id| {
                if (input_id.eql(value_id)) return true;
            }
        }
        return false;
    }
};

fn appendUniqueValueId(allocator: std.mem.Allocator, values: *std.ArrayList(compiler_facts.GraphValueId), value_id: compiler_facts.GraphValueId) !void {
    for (values.items) |existing| {
        if (existing.eql(value_id)) return;
    }
    try values.append(allocator, value_id);
}

fn selectMemorySpace(target: target_pkg.TargetDescription, placement: BufferPlacement, diagnostics: *std.Io.Writer) !u32 {
    for (target.memory_spaces) |memory_space| {
        switch (placement) {
            .host => switch (memory_space.kind) {
                .host_pinned, .host_unpinned => return memory_space.id,
                else => {},
            },
            .device => switch (memory_space.kind) {
                .device_hbm, .device_unified => return memory_space.id,
                else => {},
            },
        }
    }
    try diagnostics.print("pass=runtime-allocation feature=memory-space reason=target has no {s} memory space\n", .{@tagName(placement)});
    return RuntimeError.InvalidSchedule;
}

fn graphValue(report: core.TraceReport, value_id: compiler_facts.GraphValueId, diagnostics: *std.Io.Writer) !compiler_facts.GraphValue {
    const index: usize = std.math.cast(usize, value_id.index) orelse {
        try diagnostics.writeAll("pass=runtime-allocation feature=value reason=value id does not fit host index\n");
        return RuntimeError.InvalidSchedule;
    };
    if (index >= report.graph_values.len) {
        try diagnostics.print("pass=runtime-allocation feature=value reason=value id is out of bounds value={d}\n", .{value_id.index});
        return RuntimeError.InvalidSchedule;
    }
    return report.graph_values[index];
}

fn bufferLifetimes(
    allocator: std.mem.Allocator,
    allocations: []const BufferAllocation,
    command_buffer_uses: []const CommandBufferUse,
    diagnostics: *std.Io.Writer,
) ![]BufferLifetime {
    const lifetimes = try allocator.alloc(BufferLifetime, allocations.len);
    errdefer allocator.free(lifetimes);

    for (allocations, 0..) |allocation, allocation_index| {
        var first: ?core.ScheduleCommandId = null;
        var last: ?core.ScheduleCommandId = null;
        for (command_buffer_uses) |use| {
            if (!use.buffer_id.eql(allocation.id)) continue;
            if (first == null or use.command_id.index < first.?.index) first = use.command_id;
            if (last == null or use.command_id.index > last.?.index) last = use.command_id;
        }
        if (first == null or last == null) {
            try diagnostics.print("pass=runtime-allocation feature=lifetime reason=allocation has no command use buffer={d}\n", .{allocation.id.index});
            return RuntimeError.InvalidSchedule;
        }
        lifetimes[allocation_index] = .{
            .buffer_id = allocation.id,
            .first_command_id = first.?,
            .last_command_id = last.?,
        };
    }
    return lifetimes;
}

fn peakLiveDeviceBytes(allocations: []const BufferAllocation, lifetimes: []const BufferLifetime) u128 {
    var peak: u128 = 0;
    var command_index: u32 = 0;
    while (command_index <= maxLifetimeCommand(lifetimes)) : (command_index += 1) {
        var live: u128 = 0;
        for (lifetimes) |lifetime| {
            if (command_index < lifetime.first_command_id.index or command_index > lifetime.last_command_id.index) continue;
            const allocation = allocationById(allocations, lifetime.buffer_id);
            if (allocation.placement == .device) live += allocation.size_bytes;
        }
        peak = @max(peak, live);
    }
    return peak;
}

fn peakLiveBytesForMemory(plan: AllocationPlan, memory_space_id: u32) u128 {
    var peak: u128 = 0;
    var command_index: u32 = 0;
    while (command_index <= maxLifetimeCommand(plan.lifetimes)) : (command_index += 1) {
        var live: u128 = 0;
        for (plan.lifetimes) |lifetime| {
            if (command_index < lifetime.first_command_id.index or command_index > lifetime.last_command_id.index) continue;
            const allocation = allocationById(plan.allocations, lifetime.buffer_id);
            if (allocation.memory_space_id == memory_space_id) live += allocation.size_bytes;
        }
        peak = @max(peak, live);
    }
    return peak;
}

fn memorySpaceById(memory_spaces: []const target_pkg.TargetMemorySpace, id: u32) ?target_pkg.TargetMemorySpace {
    for (memory_spaces) |memory_space| {
        if (memory_space.id == id) return memory_space;
    }
    return null;
}

fn transferEdgeForMemoryPair(
    target: target_pkg.TargetDescription,
    memory_pair: TransferMemoryPair,
) ?target_pkg.TargetTransferEdge {
    for (target.transfer_edges) |edge| {
        if (edge.src_memory_space == memory_pair.src and edge.dst_memory_space == memory_pair.dst) return edge;
    }
    return null;
}

const TransferMemoryPair = struct {
    src: u32,
    dst: u32,
};

fn transferMemoryPairForCommand(allocation_plan: AllocationPlan, command: core.ScheduleCommand) ?TransferMemoryPair {
    return switch (command.kind) {
        .host_to_device => transferMemoryPairFromValues(allocation_plan, command.inputs, .host, command.outputs, .device),
        .device_to_host => transferMemoryPairFromValues(allocation_plan, command.inputs, .device, command.outputs, .host),
        .backend_execute, .event_record, .event_wait => null,
    };
}

fn transferMemoryPairFromValues(
    allocation_plan: AllocationPlan,
    src_values: []const compiler_facts.GraphValueId,
    src_placement: BufferPlacement,
    dst_values: []const compiler_facts.GraphValueId,
    dst_placement: BufferPlacement,
) ?TransferMemoryPair {
    if (src_values.len == 0 or dst_values.len == 0) return null;
    const src = memorySpaceForValuePlacement(allocation_plan.allocations, src_values[0], src_placement) orelse return null;
    const dst = memorySpaceForValuePlacement(allocation_plan.allocations, dst_values[0], dst_placement) orelse return null;
    return .{ .src = src, .dst = dst };
}

fn memorySpaceForValuePlacement(
    allocations: []const BufferAllocation,
    value_id: compiler_facts.GraphValueId,
    placement: BufferPlacement,
) ?u32 {
    for (allocations) |allocation| {
        if (allocation.value_id.eql(value_id) and allocation.placement == placement) return allocation.memory_space_id;
    }
    return null;
}

fn maxLifetimeCommand(lifetimes: []const BufferLifetime) u32 {
    var max_index: u32 = 0;
    for (lifetimes) |lifetime| {
        max_index = @max(max_index, lifetime.last_command_id.index);
    }
    return max_index;
}

fn allocationById(allocations: []const BufferAllocation, buffer_id: RuntimeBufferId) BufferAllocation {
    for (allocations) |allocation| {
        if (allocation.id.eql(buffer_id)) return allocation;
    }
    unreachable;
}

fn lifetimeByBufferId(lifetimes: []const BufferLifetime, buffer_id: RuntimeBufferId) BufferLifetime {
    for (lifetimes) |lifetime| {
        if (lifetime.buffer_id.eql(buffer_id)) return lifetime;
    }
    unreachable;
}

fn profileInstructionIdsForCommand(
    allocator: std.mem.Allocator,
    report: core.TraceReport,
    command: core.ScheduleCommand,
) ![]const compiler_facts.GraphInstructionId {
    if (command.kind == .backend_execute) {
        for (report.backend_bindings) |binding| {
            if (binding.command_id.eql(command.id)) {
                const ids = try allocator.alloc(compiler_facts.GraphInstructionId, binding.graph_instruction_ids.len);
                errdefer allocator.free(ids);
                @memcpy(ids, binding.graph_instruction_ids);
                return ids;
            }
        }
    }
    return allocator.alloc(compiler_facts.GraphInstructionId, 0);
}

fn appendSyntheticCommandProfileEvent(
    allocator: std.mem.Allocator,
    report: core.TraceReport,
    command: core.ScheduleCommand,
    events: *std.ArrayList(core.ProfileEvent),
) !void {
    const graph_instruction_ids = try profileInstructionIdsForCommand(allocator, report, command);
    errdefer allocator.free(graph_instruction_ids);
    const metrics = profileMetricsForCommand(report, command);
    try appendSyntheticProfileEvent(
        allocator,
        events,
        command.id,
        graph_instruction_ids,
        profileKindForCommand(command.kind),
        syntheticStartNs(command.id.index),
        metrics,
    );
}

fn appendSyntheticLoweringProfileEvent(
    allocator: std.mem.Allocator,
    report: core.TraceReport,
    command: core.ScheduleCommand,
    lowering: compiler_facts.LoweringRecord,
    events: *std.ArrayList(core.ProfileEvent),
) !void {
    const graph_instruction_ids = try copyGraphInstructionIds(allocator, lowering.graph_instruction_ids);
    errdefer allocator.free(graph_instruction_ids);
    try appendSyntheticProfileEvent(
        allocator,
        events,
        command.id,
        graph_instruction_ids,
        .backend_execute,
        syntheticStartNs(command.id.index) + lowering.id.index + 1,
        costMetrics(report, lowering.cost_ledger_ids),
    );
}

fn appendSyntheticProfileEvent(
    allocator: std.mem.Allocator,
    events: *std.ArrayList(core.ProfileEvent),
    command_id: core.ScheduleCommandId,
    graph_instruction_ids: []const compiler_facts.GraphInstructionId,
    kind: core.ProfileEventKind,
    start_ns: u64,
    metrics: ProfileMetrics,
) !void {
    try events.append(allocator, .{
        .id = .{ .index = std.math.cast(u32, events.items.len) orelse unreachable },
        .command_id = command_id,
        .graph_instruction_ids = graph_instruction_ids,
        .kind = kind,
        .start_ns = start_ns,
        .duration_ns = syntheticDurationNsForMetrics(kind, metrics),
        .bytes = metrics.bytes,
        .logical_ops = metrics.logical_ops,
        .status = .ok,
        .forced_synchronization = false,
    });
}

fn copyGraphInstructionIds(allocator: std.mem.Allocator, ids: []const compiler_facts.GraphInstructionId) ![]const compiler_facts.GraphInstructionId {
    const copy = try allocator.alloc(compiler_facts.GraphInstructionId, ids.len);
    errdefer allocator.free(copy);
    @memcpy(copy, ids);
    return copy;
}

fn loweringRecord(report: core.TraceReport, id: compiler_facts.LoweringRecordId) compiler_facts.LoweringRecord {
    const index: usize = std.math.cast(usize, id.index) orelse unreachable;
    return report.lowering_records[index];
}

fn graphInstructionIdsEqual(left: []const compiler_facts.GraphInstructionId, right: []const compiler_facts.GraphInstructionId) bool {
    if (left.len != right.len) return false;
    for (left, right) |left_id, right_id| {
        if (!left_id.eql(right_id)) return false;
    }
    return true;
}

fn instructionIdIn(needle: compiler_facts.GraphInstructionId, haystack: []const compiler_facts.GraphInstructionId) bool {
    for (haystack) |id| {
        if (id.eql(needle)) return true;
    }
    return false;
}

const ProfileMetrics = struct {
    bytes: u128,
    logical_ops: u128,
};

fn profileMetricsForCommand(report: core.TraceReport, command: core.ScheduleCommand) ProfileMetrics {
    return switch (command.kind) {
        .host_to_device, .device_to_host => .{
            .bytes = valueBytes(report, if (command.kind == .host_to_device) command.outputs else command.inputs),
            .logical_ops = 0,
        },
        .backend_execute => costMetrics(report, command.cost_ledger_ids),
        .event_record, .event_wait => .{ .bytes = 0, .logical_ops = 0 },
    };
}

fn valueBytes(report: core.TraceReport, ids: []const compiler_facts.GraphValueId) u128 {
    var total: u128 = 0;
    for (ids) |id| {
        const index: usize = std.math.cast(usize, id.index) orelse unreachable;
        total += tensorBytes(report.graph_values[index].ty);
    }
    return total;
}

fn costMetrics(report: core.TraceReport, ids: []const compiler_facts.CostLedgerId) ProfileMetrics {
    var metrics: ProfileMetrics = .{ .bytes = 0, .logical_ops = 0 };
    for (ids) |id| {
        const index: usize = std.math.cast(usize, id.index) orelse unreachable;
        const entry = report.cost_ledger[index];
        metrics.bytes += entry.bytes_read + entry.bytes_written;
        metrics.logical_ops += entry.logical_ops;
    }
    return metrics;
}

fn writeInstructionIdList(writer: *std.Io.Writer, ids: []const compiler_facts.GraphInstructionId) std.Io.Writer.Error!void {
    for (ids, 0..) |id, index| {
        if (index != 0) try writer.writeAll(",");
        try writer.print("{d}", .{id.index});
    }
}

fn tensorBytes(ty: compiler_facts.TensorType) u128 {
    const element_size = ty.element_type.byteSize() orelse 0;
    var elements: u128 = 1;
    for (ty.dims) |dim| {
        elements *= std.math.cast(u128, dim) orelse unreachable;
    }
    return elements * element_size;
}

fn profileKindForCommand(kind: core.CommandKind) core.ProfileEventKind {
    return switch (kind) {
        .host_to_device => .h2d,
        .backend_execute => .backend_execute,
        .device_to_host => .d2h,
        .event_record, .event_wait => .compile_pass,
    };
}

fn syntheticStartNs(command_index: u32) u64 {
    return std.math.cast(u64, command_index) orelse unreachable;
}

fn syntheticDurationNsForMetrics(kind: core.ProfileEventKind, metrics: ProfileMetrics) u64 {
    return switch (kind) {
        .h2d, .d2h => std.math.cast(u64, metrics.bytes + 1) orelse std.math.maxInt(u64),
        .backend_execute => std.math.cast(u64, metrics.logical_ops + 1) orelse std.math.maxInt(u64),
        .compile_pass => 1,
    };
}

test "runtime command kind writes stable schedule names" {
    var output = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer output.deinit();

    try writeCommandKind(&output.writer, .backend_execute);
    try std.testing.expectEqualStrings("backend_execute", output.writer.buffered());
}

test "runtime package imports only new core namespace for now" {
    try std.testing.expect(core.idInBounds(0, 1));
}

test "runtime rejects backend execute without exactly one binding" {
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    try std.testing.expectError(
        RuntimeError.InvalidSchedule,
        executeSyntheticProfile(std.testing.allocator, missing_binding_report, &diagnostics.writer),
    );
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "backend command must have exactly one binding") != null);
}

test "stream planning rejects dependencies that do not point earlier" {
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    try std.testing.expectError(
        RuntimeError.InvalidSchedule,
        planStreams(std.testing.allocator, self_dependency_report, &diagnostics.writer),
    );
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "dependency must be earlier") != null);
}

test "allocation planning rejects transfer command without target edge" {
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    try std.testing.expectError(
        RuntimeError.InvalidSchedule,
        planAllocations(std.testing.allocator, missing_transfer_edge_report, &diagnostics.writer),
    );
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "no target edge") != null);
}

test "allocation planning rejects peak live bytes above memory capacity" {
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    try std.testing.expectError(
        RuntimeError.InvalidSchedule,
        planAllocations(std.testing.allocator, tiny_capacity_report, &diagnostics.writer),
    );
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "peak live bytes exceed memory capacity") != null);
}

const test_source: compiler_facts.SourceRef = .{ .id = .{ .index = 0 }, .frontend = .stablehlo, .op_name = "stablehlo.tanh", .source_index = 0, .location = "" };
const test_sources = [_]compiler_facts.SourceRef{test_source};
const test_dims = [_]i64{4};
const test_values = [_]compiler_facts.GraphValue{
    .{ .id = .{ .index = 0 }, .ty = .{ .element_type = .f32, .dims = &test_dims, .layout = .dense_row_major }, .role = .parameter, .source = null },
    .{ .id = .{ .index = 1 }, .ty = .{ .element_type = .f32, .dims = &test_dims, .layout = .dense_row_major }, .role = .instruction_result, .source = test_source },
};
const test_inputs = [_]compiler_facts.GraphValueId{.{ .index = 0 }};
const test_outputs = [_]compiler_facts.GraphValueId{.{ .index = 1 }};
const test_instruction_ids = [_]compiler_facts.GraphInstructionId{.{ .index = 0 }};
const test_cost_ids = [_]compiler_facts.CostLedgerId{.{ .index = 0 }};
const test_lowering_ids = [_]compiler_facts.LoweringRecordId{.{ .index = 0 }};
const test_instructions = [_]compiler_facts.GraphInstruction{
    .{ .id = .{ .index = 0 }, .kind = .elementwise_unary, .inputs = &test_inputs, .outputs = &test_outputs, .payload = .{ .elementwise_unary = .{ .op = .tanh } }, .source = test_source },
};
const test_costs = [_]compiler_facts.CostLedgerEntry{
    .{ .id = .{ .index = 0 }, .source = test_source, .graph_instruction_ids = &test_instruction_ids, .op_class = .transcendental, .dtype = .f32, .accumulation_dtype = null, .logical_ops = 4, .bytes_read = 16, .bytes_written = 16, .expected_unit_id = null, .formula = "numel(output)", .approximation = "" },
};
const test_lowerings = [_]compiler_facts.LoweringRecord{
    .{ .id = .{ .index = 0 }, .graph_instruction_ids = &test_instruction_ids, .decision = .elementwise_fusion, .reason = "test", .rejected_alternatives = &.{}, .cost_ledger_ids = &test_cost_ids },
};
const test_commands = [_]core.ScheduleCommand{
    .{ .id = .{ .index = 0 }, .kind = .backend_execute, .stream = .{ .index = 0 }, .inputs = &test_inputs, .outputs = &test_outputs, .dependencies = &.{}, .lowering_record_ids = &test_lowering_ids, .cost_ledger_ids = &test_cost_ids },
};
const self_dependencies = [_]core.CommandDependency{
    .{ .command_id = .{ .index = 0 }, .kind = .data },
};
const self_dependency_commands = [_]core.ScheduleCommand{
    .{ .id = .{ .index = 0 }, .kind = .backend_execute, .stream = .{ .index = 0 }, .inputs = &test_inputs, .outputs = &test_outputs, .dependencies = &self_dependencies, .lowering_record_ids = &test_lowering_ids, .cost_ledger_ids = &test_cost_ids },
};
const test_bindings = [_]core.BackendBinding{
    .{ .id = .{ .index = 0 }, .command_id = .{ .index = 0 }, .backend_kind = .npu_v0, .backend_operation = "npu_execute", .graph_instruction_ids = &test_instruction_ids, .expected_unit_id = null, .cost_ledger_ids = &test_cost_ids },
};
const missing_binding_report: core.TraceReport = .{
    .sources = &test_sources,
    .graph_values = &test_values,
    .graph_instructions = &test_instructions,
    .cost_ledger = &test_costs,
    .lowering_records = &test_lowerings,
    .schedule_commands = &test_commands,
    .backend_bindings = &.{},
    .profile_events = &.{},
    .explain_records = &.{},
};
const self_dependency_report: core.TraceReport = .{
    .sources = &test_sources,
    .graph_values = &test_values,
    .graph_instructions = &test_instructions,
    .cost_ledger = &test_costs,
    .lowering_records = &test_lowerings,
    .schedule_commands = &self_dependency_commands,
    .backend_bindings = &test_bindings,
    .profile_events = &.{},
    .explain_records = &.{},
};

const transfer_values = [_]compiler_facts.GraphValue{
    .{ .id = .{ .index = 0 }, .ty = .{ .element_type = .f32, .dims = &test_dims, .layout = .dense_row_major }, .role = .parameter, .source = null },
};
const transfer_value_ids = [_]compiler_facts.GraphValueId{.{ .index = 0 }};
const transfer_commands = [_]core.ScheduleCommand{
    .{ .id = .{ .index = 0 }, .kind = .host_to_device, .stream = .{ .index = 0 }, .inputs = &transfer_value_ids, .outputs = &transfer_value_ids, .dependencies = &.{}, .lowering_record_ids = &.{}, .cost_ledger_ids = &.{} },
};
const transfer_memory_spaces = [_]target_pkg.TargetMemorySpace{
    .{ .id = 0, .name = "host_pinned", .kind = .host_pinned, .capacity_bytes = null, .bandwidth_bytes_per_second = null, .note = "" },
    .{ .id = 1, .name = "device_hbm", .kind = .device_hbm, .capacity_bytes = 64, .bandwidth_bytes_per_second = null, .note = "" },
};
const tiny_transfer_memory_spaces = [_]target_pkg.TargetMemorySpace{
    .{ .id = 0, .name = "host_pinned", .kind = .host_pinned, .capacity_bytes = null, .bandwidth_bytes_per_second = null, .note = "" },
    .{ .id = 1, .name = "device_hbm", .kind = .device_hbm, .capacity_bytes = 8, .bandwidth_bytes_per_second = null, .note = "" },
};
const transfer_edges = [_]target_pkg.TargetTransferEdge{
    .{ .id = 0, .src_memory_space = 0, .dst_memory_space = 1, .bandwidth_bytes_per_second = 64000000000, .latency_ns = null, .supports_async = true, .engine_unit_id = 0, .note = "" },
};
const transfer_execution_units = [_]target_pkg.ExecutionUnit{
    .{ .id = 0, .name = "dma", .kind = .dma, .dtype_rates = &.{} },
};
const transfer_device_memory_spaces = [_]u32{ 0, 1 };
const transfer_device_execution_units = [_]u32{0};
const transfer_devices = [_]target_pkg.TargetDevice{
    .{ .id = 0, .local_hardware_id = 0, .name = "device", .memory_space_ids = &transfer_device_memory_spaces, .execution_unit_ids = &transfer_device_execution_units },
};
const missing_transfer_edge_target: target_pkg.TargetDescription = .{
    .name = "npu_v0",
    .kind = .npu_v0,
    .devices = &transfer_devices,
    .memory_spaces = &transfer_memory_spaces,
    .transfer_edges = &.{},
    .execution_units = &transfer_execution_units,
};
const tiny_capacity_target: target_pkg.TargetDescription = .{
    .name = "npu_v0",
    .kind = .npu_v0,
    .devices = &transfer_devices,
    .memory_spaces = &tiny_transfer_memory_spaces,
    .transfer_edges = &transfer_edges,
    .execution_units = &transfer_execution_units,
};
const missing_transfer_edge_report: core.TraceReport = .{
    .target = missing_transfer_edge_target,
    .sources = &.{},
    .graph_values = &transfer_values,
    .graph_instructions = &.{},
    .cost_ledger = &.{},
    .lowering_records = &.{},
    .schedule_commands = &transfer_commands,
    .backend_bindings = &.{},
    .profile_events = &.{},
    .explain_records = &.{},
};
const tiny_capacity_report: core.TraceReport = .{
    .target = tiny_capacity_target,
    .sources = &.{},
    .graph_values = &transfer_values,
    .graph_instructions = &.{},
    .cost_ledger = &.{},
    .lowering_records = &.{},
    .schedule_commands = &transfer_commands,
    .backend_bindings = &.{},
    .profile_events = &.{},
    .explain_records = &.{},
};
