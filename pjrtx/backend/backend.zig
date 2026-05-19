const std = @import("std");
const compiler_facts = @import("pjrtx/compiler/facts");
const core = @import("pjrtx/core");
const target_pkg = @import("pjrtx/target");

pub const BackendKind = core.BackendKind;

pub const BackendError = error{
    InvalidExecutablePlan,
};

pub const BackendFeature = enum {
    rank2_dot_general,
    broadcast_in_dim,
    reshape,
    transpose,
    add,
    tanh,
    elementwise_fusion,
};

pub const BackendCapability = struct {
    feature: BackendFeature,
    dtypes: []const core.BufferType,
    backend_operation: []const u8,
    expected_unit_id: ?u32,
};

pub const BackendCapabilitySet = struct {
    kind: BackendKind,
    capabilities: []const BackendCapability,
};

pub const BackendExecutableCall = struct {
    index: u32,
    graph_instruction_id: compiler_facts.GraphInstructionId,
    graph_instruction_ids: []const compiler_facts.GraphInstructionId,
    feature: BackendFeature,
    backend_operation: []const u8,
    input_value_ids: []const compiler_facts.GraphValueId,
    output_value_ids: []const compiler_facts.GraphValueId,
    expected_unit_id: ?u32,
};

pub const BackendExecutablePlan = struct {
    allocator: std.mem.Allocator,
    backend_kind: BackendKind,
    command_id: core.ScheduleCommandId,
    backend_operation: []const u8,
    calls: []const BackendExecutableCall,

    pub fn deinit(self: *BackendExecutablePlan) void {
        for (self.calls) |call| {
            self.allocator.free(call.graph_instruction_ids);
            self.allocator.free(call.output_value_ids);
            self.allocator.free(call.input_value_ids);
        }
        self.allocator.free(self.calls);
        self.* = undefined;
    }
};

pub const BackendKernelGraphNodeId = struct {
    index: u32,
};

pub const BackendKernelGraphNode = struct {
    id: BackendKernelGraphNodeId,
    call_index: u32,
    graph_instruction_id: compiler_facts.GraphInstructionId,
    graph_instruction_ids: []const compiler_facts.GraphInstructionId,
    feature: BackendFeature,
    backend_operation: []const u8,
    input_value_ids: []const compiler_facts.GraphValueId,
    output_value_ids: []const compiler_facts.GraphValueId,
    output_type: BackendTensorDescriptor,
    attributes: BackendKernelAttributes,
};

pub const BackendKernelGraphEdge = struct {
    value_id: compiler_facts.GraphValueId,
    src_node_id: BackendKernelGraphNodeId,
    dst_node_id: BackendKernelGraphNodeId,
};

pub const BackendTensorDescriptor = struct {
    element_type: core.BufferType,
    dims: []const i64,
    layout: compiler_facts.LayoutKind,
};

pub const BackendKernelAttributes = union(enum) {
    rank2_dot_general: compiler_facts.DotGeneralSpec,
    broadcast_in_dim: []const u32,
    add: void,
    tanh: void,
    elementwise_fusion: []const compiler_facts.GraphInstructionId,
};

pub const BackendKernelGraphPlan = struct {
    allocator: std.mem.Allocator,
    backend_kind: BackendKind,
    command_id: core.ScheduleCommandId,
    nodes: []const BackendKernelGraphNode,
    edges: []const BackendKernelGraphEdge,

    pub fn deinit(self: *BackendKernelGraphPlan) void {
        for (self.nodes) |node| {
            self.allocator.free(node.output_type.dims);
            deinitKernelAttributes(self.allocator, node.attributes);
            self.allocator.free(node.graph_instruction_ids);
            self.allocator.free(node.output_value_ids);
            self.allocator.free(node.input_value_ids);
        }
        self.allocator.free(self.edges);
        self.allocator.free(self.nodes);
        self.* = undefined;
    }
};

pub fn writeBackendKind(writer: *std.Io.Writer, kind: BackendKind) std.Io.Writer.Error!void {
    try writer.writeAll(@tagName(kind));
}

pub fn capabilitySet(kind: BackendKind) BackendCapabilitySet {
    return switch (kind) {
        .metal_v0 => .{ .kind = kind, .capabilities = &metal_capabilities },
        .npu_v0 => .{ .kind = kind, .capabilities = &npu_capabilities },
    };
}

pub fn featureForInstruction(instruction: compiler_facts.GraphInstruction) ?BackendFeature {
    return switch (instruction.payload) {
        .dot_general => .rank2_dot_general,
        .broadcast => .broadcast_in_dim,
        .reshape => .reshape,
        .transpose => .transpose,
        .elementwise_binary => |payload| switch (payload.op) {
            .add => .add,
        },
        .elementwise_unary => |payload| switch (payload.op) {
            .tanh => .tanh,
        },
        .collective => null,
        .return_ => null,
    };
}

pub fn supportsInstruction(capabilities: BackendCapabilitySet, instruction: compiler_facts.GraphInstruction, values: []const compiler_facts.GraphValue) bool {
    if (featureForInstruction(instruction) == null) return instruction.kind == .return_;
    return capabilityForInstruction(capabilities, instruction, values) != null;
}

pub fn capabilityForInstruction(capabilities: BackendCapabilitySet, instruction: compiler_facts.GraphInstruction, values: []const compiler_facts.GraphValue) ?BackendCapability {
    const feature = featureForInstruction(instruction) orelse return null;
    const dtype = instructionDtype(instruction, values) orelse return null;
    for (capabilities.capabilities) |capability| {
        if (capability.feature != feature) continue;
        if (supportsDtype(capability.dtypes, dtype)) return capability;
    }
    return null;
}

/// Backend executable planning is the boundary before real Metal or NPU
/// submission. It expands a verified backend binding into concrete backend
/// calls, so runtime can execute commands without interpreting graph ops.
pub fn planExecutable(
    allocator: std.mem.Allocator,
    report: core.TraceReport,
    binding: core.BackendBinding,
    diagnostics: *std.Io.Writer,
) !BackendExecutablePlan {
    core.validateTraceReport(report, diagnostics) catch {
        try diagnostics.writeAll("pass=backend-executable feature=trace-report reason=trace report failed validation\n");
        return BackendError.InvalidExecutablePlan;
    };

    try verifyBindingBelongsToReport(report, binding, diagnostics);
    try verifyBindingMatchesTarget(report, binding, diagnostics);
    const command = try commandForBinding(report, binding, diagnostics);
    const expected_backend_operation = executableOperationForBackend(binding.backend_kind);
    if (!std.mem.eql(u8, binding.backend_operation, expected_backend_operation)) {
        try diagnostics.print(
            "pass=backend-executable feature=backend-operation reason=binding operation does not match backend executable backend={s} expected={s} actual={s}\n",
            .{ @tagName(binding.backend_kind), expected_backend_operation, binding.backend_operation },
        );
        return BackendError.InvalidExecutablePlan;
    }
    if (command.kind != .backend_execute) {
        try diagnostics.print(
            "pass=backend-executable feature=command reason=binding command is not backend_execute command={d}\n",
            .{command.id.index},
        );
        return BackendError.InvalidExecutablePlan;
    }

    var calls: std.ArrayList(BackendExecutableCall) = .empty;
    errdefer {
        for (calls.items) |call| {
            allocator.free(call.graph_instruction_ids);
            allocator.free(call.output_value_ids);
            allocator.free(call.input_value_ids);
        }
        calls.deinit(allocator);
    }

    for (command.lowering_record_ids) |lowering_record_id| {
        const lowering = loweringRecord(report, lowering_record_id, diagnostics) catch return BackendError.InvalidExecutablePlan;
        try verifyLoweringBelongsToBinding(lowering, binding, diagnostics);
        const call = try callForLowering(allocator, report, binding.backend_kind, lowering, calls.items.len, diagnostics);
        errdefer {
            allocator.free(call.graph_instruction_ids);
            allocator.free(call.output_value_ids);
            allocator.free(call.input_value_ids);
        }
        try calls.append(allocator, call);
    }

    return .{
        .allocator = allocator,
        .backend_kind = binding.backend_kind,
        .command_id = binding.command_id,
        .backend_operation = binding.backend_operation,
        .calls = try calls.toOwnedSlice(allocator),
    };
}

fn callForLowering(
    allocator: std.mem.Allocator,
    report: core.TraceReport,
    backend_kind: core.BackendKind,
    lowering: compiler_facts.LoweringRecord,
    call_index: usize,
    diagnostics: *std.Io.Writer,
) !BackendExecutableCall {
    if (lowering.graph_instruction_ids.len == 0) {
        try diagnostics.print(
            "pass=backend-executable feature=lowering reason=lowering record has no instructions lowering={d}\n",
            .{lowering.id.index},
        );
        return BackendError.InvalidExecutablePlan;
    }

    const capabilities: BackendCapabilitySet = capabilitySet(backend_kind);
    if (lowering.decision == .elementwise_fusion) {
        try verifyElementwiseFusionLowering(report, capabilities, lowering, diagnostics);
        const graph_instruction_ids = try copyGraphInstructionIds(allocator, lowering.graph_instruction_ids);
        errdefer allocator.free(graph_instruction_ids);
        const input_value_ids = try regionInputValueIds(allocator, report, lowering.graph_instruction_ids, diagnostics);
        errdefer allocator.free(input_value_ids);
        const output_value_ids = try regionOutputValueIds(allocator, report, lowering.graph_instruction_ids, diagnostics);
        errdefer allocator.free(output_value_ids);
        return .{
            .index = std.math.cast(u32, call_index) orelse unreachable,
            .graph_instruction_id = lowering.graph_instruction_ids[0],
            .graph_instruction_ids = graph_instruction_ids,
            .feature = .elementwise_fusion,
            .backend_operation = fusedElementwiseOperationForBackend(backend_kind),
            .input_value_ids = input_value_ids,
            .output_value_ids = output_value_ids,
            .expected_unit_id = expectedUnitForLowering(report, capabilities, lowering),
        };
    }

    if (lowering.graph_instruction_ids.len != 1) {
        try diagnostics.print(
            "pass=backend-executable feature=lowering reason=non-fused lowering must contain exactly one instruction lowering={d} instructions={d}\n",
            .{ lowering.id.index, lowering.graph_instruction_ids.len },
        );
        return BackendError.InvalidExecutablePlan;
    }

    const instruction_id = lowering.graph_instruction_ids[0];
    const instruction = graphInstruction(report, instruction_id);
    const feature = featureForInstruction(instruction) orelse {
        try diagnostics.print(
            "pass=backend-executable feature=graph-instruction reason=backend binding includes non-executable instruction instruction={d}\n",
            .{instruction_id.index},
        );
        return BackendError.InvalidExecutablePlan;
    };
    const capability = capabilityForInstruction(capabilities, instruction, report.graph_values) orelse {
        try diagnostics.print(
            "pass=backend-executable feature=capability reason=backend cannot execute instruction backend={s} instruction={d}\n",
            .{ @tagName(backend_kind), instruction_id.index },
        );
        return BackendError.InvalidExecutablePlan;
    };

    const input_value_ids = try copyGraphValueIds(allocator, instruction.inputs);
    errdefer allocator.free(input_value_ids);
    const output_value_ids = try copyGraphValueIds(allocator, instruction.outputs);
    errdefer allocator.free(output_value_ids);
    const graph_instruction_ids = try copyGraphInstructionIds(allocator, lowering.graph_instruction_ids);
    errdefer allocator.free(graph_instruction_ids);

    return .{
        .index = std.math.cast(u32, call_index) orelse unreachable,
        .graph_instruction_id = instruction_id,
        .graph_instruction_ids = graph_instruction_ids,
        .feature = feature,
        .backend_operation = capability.backend_operation,
        .input_value_ids = input_value_ids,
        .output_value_ids = output_value_ids,
        .expected_unit_id = capability.expected_unit_id,
    };
}

fn verifyElementwiseFusionLowering(
    report: core.TraceReport,
    capabilities: BackendCapabilitySet,
    lowering: compiler_facts.LoweringRecord,
    diagnostics: *std.Io.Writer,
) !void {
    for (lowering.graph_instruction_ids) |instruction_id| {
        const instruction = graphInstruction(report, instruction_id);
        switch (instruction.kind) {
            .broadcast, .elementwise_binary, .elementwise_unary => {},
            else => {
                try diagnostics.print(
                    "pass=backend-executable feature=fusion reason=fused lowering contains non-elementwise instruction lowering={d} instruction={d}\n",
                    .{ lowering.id.index, instruction_id.index },
                );
                return BackendError.InvalidExecutablePlan;
            },
        }
        if (capabilityForInstruction(capabilities, instruction, report.graph_values) == null) {
            try diagnostics.print(
                "pass=backend-executable feature=capability reason=backend cannot execute fused instruction backend={s} instruction={d}\n",
                .{ @tagName(capabilities.kind), instruction_id.index },
            );
            return BackendError.InvalidExecutablePlan;
        }
    }
}

/// The Metal/MLS graph plan makes value flow explicit before any command buffer
/// submission exists. V0 is intentionally one node per verified backend call.
pub fn planKernelGraph(
    allocator: std.mem.Allocator,
    report: core.TraceReport,
    executable: BackendExecutablePlan,
    diagnostics: *std.Io.Writer,
) !BackendKernelGraphPlan {
    core.validateTraceReport(report, diagnostics) catch {
        try diagnostics.writeAll("pass=backend-kernel-graph feature=trace-report reason=trace report failed validation\n");
        return BackendError.InvalidExecutablePlan;
    };
    if (executable.backend_kind != .metal_v0) {
        try diagnostics.print(
            "pass=backend-kernel-graph feature=backend reason=kernel graph planning is Metal-only in V0 backend={s}\n",
            .{@tagName(executable.backend_kind)},
        );
        return BackendError.InvalidExecutablePlan;
    }

    var nodes: std.ArrayList(BackendKernelGraphNode) = .empty;
    var edges: std.ArrayList(BackendKernelGraphEdge) = .empty;
    errdefer {
        for (nodes.items) |node| {
            allocator.free(node.output_type.dims);
            deinitKernelAttributes(allocator, node.attributes);
            allocator.free(node.graph_instruction_ids);
            allocator.free(node.output_value_ids);
            allocator.free(node.input_value_ids);
        }
        nodes.deinit(allocator);
        edges.deinit(allocator);
    }

    for (executable.calls) |call| {
        const input_value_ids = try copyGraphValueIds(allocator, call.input_value_ids);
        errdefer allocator.free(input_value_ids);
        const output_value_ids = try copyGraphValueIds(allocator, call.output_value_ids);
        errdefer allocator.free(output_value_ids);
        const instruction = graphInstruction(report, call.graph_instruction_id);
        const output_type = try kernelOutputType(allocator, report, call, diagnostics);
        errdefer allocator.free(output_type.dims);
        const attributes = try kernelAttributes(allocator, report, call, instruction, diagnostics);
        errdefer deinitKernelAttributes(allocator, attributes);
        const graph_instruction_ids = try copyGraphInstructionIds(allocator, call.graph_instruction_ids);
        errdefer allocator.free(graph_instruction_ids);
        try nodes.append(allocator, .{
            .id = .{ .index = std.math.cast(u32, nodes.items.len) orelse unreachable },
            .call_index = call.index,
            .graph_instruction_id = call.graph_instruction_id,
            .graph_instruction_ids = graph_instruction_ids,
            .feature = call.feature,
            .backend_operation = call.backend_operation,
            .input_value_ids = input_value_ids,
            .output_value_ids = output_value_ids,
            .output_type = output_type,
            .attributes = attributes,
        });
    }

    for (nodes.items) |dst_node| {
        for (dst_node.input_value_ids) |input_value_id| {
            if (try producerNodeForValue(nodes.items, input_value_id, diagnostics)) |src_node_id| {
                try edges.append(allocator, .{
                    .value_id = input_value_id,
                    .src_node_id = src_node_id,
                    .dst_node_id = dst_node.id,
                });
            }
        }
    }

    return .{
        .allocator = allocator,
        .backend_kind = executable.backend_kind,
        .command_id = executable.command_id,
        .nodes = try nodes.toOwnedSlice(allocator),
        .edges = try edges.toOwnedSlice(allocator),
    };
}

pub fn writeExecutablePlanSummary(plan: BackendExecutablePlan, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    try writer.print(
        "backend executable\n  backend={s} command={d} operation={s} calls={d}\n",
        .{ @tagName(plan.backend_kind), plan.command_id.index, plan.backend_operation, plan.calls.len },
    );
    for (plan.calls) |call| {
        try writer.print(
            "  call.{d} instruction={d} instructions=",
            .{ call.index, call.graph_instruction_id.index },
        );
        try writeInstructionIdList(writer, call.graph_instruction_ids);
        try writer.print(" feature={s} operation={s} unit=", .{ @tagName(call.feature), call.backend_operation });
        if (call.expected_unit_id) |unit_id| {
            try writer.print("{d}", .{unit_id});
        } else {
            try writer.writeAll("unknown");
        }
        try writer.print(" inputs={d} outputs={d}\n", .{ call.input_value_ids.len, call.output_value_ids.len });
    }
}

pub fn writeBackendCallProfileSummary(
    report: core.TraceReport,
    executable: BackendExecutablePlan,
    writer: *std.Io.Writer,
) std.Io.Writer.Error!void {
    try writer.writeAll("backend call profiles\n");
    for (executable.calls) |call| {
        const predicted = callProfileMetrics(report, call.graph_instruction_ids);
        const observed = profileEventForCall(report.profile_events, executable.command_id, call.graph_instruction_ids);
        const ideal_compute_ps = idealComputePsForCall(report, call.graph_instruction_ids);
        const ideal_memory_ps = idealMemoryPsForCall(report, call);
        const limit = limitingResource(ideal_compute_ps, ideal_memory_ps);
        try writer.print(
            "  call.{d} command={d} operation={s} instructions=",
            .{ call.index, executable.command_id.index, call.backend_operation },
        );
        try writeInstructionIdList(writer, call.graph_instruction_ids);
        try writer.writeAll(" unit=");
        if (call.expected_unit_id) |unit_id| {
            try writer.print("{d}", .{unit_id});
        } else {
            try writer.writeAll("unknown");
        }
        try writer.print(" predicted_bytes={d} observed_bytes=", .{predicted.bytes});
        if (observed) |event| {
            try writer.print(
                "{d} predicted_ops={d} observed_ops={d} ideal_compute_ps={d} ideal_memory_ps={d} limiting={s} memory=",
                .{ event.bytes, predicted.logical_ops, event.logical_ops, ideal_compute_ps, ideal_memory_ps, limit },
            );
            try writeMemoryTrafficList(writer, report, call);
            try writer.print(" event=profile.{d}\n", .{event.id.index});
        } else {
            try writer.print(
                "missing predicted_ops={d} observed_ops=missing ideal_compute_ps={d} ideal_memory_ps={d} limiting={s} memory=",
                .{ predicted.logical_ops, ideal_compute_ps, ideal_memory_ps, limit },
            );
            try writeMemoryTrafficList(writer, report, call);
            try writer.writeAll(" event=missing\n");
        }
    }
}

fn executableOperationForBackend(kind: BackendKind) []const u8 {
    return switch (kind) {
        .metal_v0 => "metal_mls_graph_execute",
        .npu_v0 => "npu_execute",
    };
}

const ProfileMetrics = struct {
    bytes: u128,
    logical_ops: u128,
};

fn callProfileMetrics(report: core.TraceReport, instruction_ids: []const compiler_facts.GraphInstructionId) ProfileMetrics {
    var metrics: ProfileMetrics = .{ .bytes = 0, .logical_ops = 0 };
    for (report.cost_ledger) |entry| {
        if (!allInstructionIdsIn(entry.graph_instruction_ids, instruction_ids)) continue;
        metrics.bytes += entry.bytes_read + entry.bytes_written;
        metrics.logical_ops += entry.logical_ops;
    }
    return metrics;
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

fn allInstructionIdsIn(needles: []const compiler_facts.GraphInstructionId, haystack: []const compiler_facts.GraphInstructionId) bool {
    for (needles) |needle| {
        if (!instructionIdIn(needle, haystack)) return false;
    }
    return true;
}

fn instructionIdIn(needle: compiler_facts.GraphInstructionId, haystack: []const compiler_facts.GraphInstructionId) bool {
    for (haystack) |id| {
        if (id.eql(needle)) return true;
    }
    return false;
}

fn graphInstructionIdsEqual(left: []const compiler_facts.GraphInstructionId, right: []const compiler_facts.GraphInstructionId) bool {
    if (left.len != right.len) return false;
    for (left, right) |left_id, right_id| {
        if (!left_id.eql(right_id)) return false;
    }
    return true;
}

fn idealComputePsForCall(report: core.TraceReport, instruction_ids: []const compiler_facts.GraphInstructionId) u128 {
    const target = report.target orelse return 0;
    var total: u128 = 0;
    for (report.cost_ledger) |entry| {
        if (!allInstructionIdsIn(entry.graph_instruction_ids, instruction_ids)) continue;
        total += idealComputePsForCostEntry(target, entry);
    }
    return total;
}

fn idealComputePsForCostEntry(target: target_pkg.TargetDescription, entry: compiler_facts.CostLedgerEntry) u128 {
    const unit_id = entry.expected_unit_id orelse return 0;
    const unit = executionUnitById(target.execution_units, unit_id) orelse return 0;
    for (unit.dtype_rates) |rate| {
        if (entry.dtype == rate.dtype and costOpClassMatchesRate(entry.op_class, rate.op_class)) {
            return idealComputePs(entry.logical_ops, rate.ops_per_second);
        }
    }
    return 0;
}

fn executionUnitById(units: []const target_pkg.ExecutionUnit, unit_id: u32) ?target_pkg.ExecutionUnit {
    for (units) |unit| {
        if (unit.id == unit_id) return unit;
    }
    return null;
}

fn costOpClassMatchesRate(cost_class: compiler_facts.CostOpClass, rate_class: core.OpClass) bool {
    return switch (cost_class) {
        .matmul => rate_class == .matmul,
        .elementwise => rate_class == .elementwise,
        .transcendental => rate_class == .transcendental,
        .transfer => rate_class == .memory,
        .backend_kernel => false,
    };
}

fn idealComputePs(logical_ops: u128, ops_per_second: ?f64) u128 {
    if (logical_ops == 0) return 0;
    const peak = ops_per_second orelse return 0;
    if (peak <= 0) return 0;
    const ops: f64 = std.math.lossyCast(f64, logical_ops);
    const ideal = @ceil((ops * 1_000_000_000_000.0) / peak);
    return std.math.lossyCast(u128, ideal);
}

fn idealMemoryPsForCall(report: core.TraceReport, call: BackendExecutableCall) u128 {
    const target = report.target orelse return 0;
    var max_ps: u128 = 0;
    for (target.memory_spaces) |memory_space| {
        const bytes = memoryTrafficBytesForCall(report, call, memory_space.id);
        const ps = idealMemoryPs(bytes, memory_space.bandwidth_bytes_per_second);
        if (ps > max_ps) max_ps = ps;
    }
    return max_ps;
}

fn writeMemoryTrafficList(writer: *std.Io.Writer, report: core.TraceReport, call: BackendExecutableCall) std.Io.Writer.Error!void {
    const target = report.target orelse {
        try writer.writeAll("unknown");
        return;
    };
    var wrote = false;
    for (target.memory_spaces) |memory_space| {
        const bytes = memoryTrafficBytesForCall(report, call, memory_space.id);
        if (bytes == 0) continue;
        if (wrote) try writer.writeAll(",");
        try writer.print(
            "memory.{d}:{d}B/{d}ps",
            .{ memory_space.id, bytes, idealMemoryPs(bytes, memory_space.bandwidth_bytes_per_second) },
        );
        wrote = true;
    }
    if (!wrote) try writer.writeAll("none");
}

fn memoryTrafficBytesForCall(report: core.TraceReport, call: BackendExecutableCall, memory_space_id: u32) u128 {
    var total: u128 = 0;
    for (report.memory_traffic_records) |record| {
        if (record.memory_space_id != memory_space_id) continue;
        if (!graphInstructionIdsEqual(record.graph_instruction_ids, call.graph_instruction_ids)) continue;
        total += record.bytes_read + record.bytes_written;
    }
    return total;
}

fn idealMemoryPs(bytes: u128, bytes_per_second: ?f64) u128 {
    if (bytes == 0) return 0;
    const bandwidth = bytes_per_second orelse return 0;
    if (bandwidth <= 0) return 0;
    const byte_count: f64 = std.math.lossyCast(f64, bytes);
    const ideal = @ceil((byte_count * 1_000_000_000_000.0) / bandwidth);
    return std.math.lossyCast(u128, ideal);
}

fn limitingResource(ideal_compute_ps: u128, ideal_memory_ps: u128) []const u8 {
    if (ideal_compute_ps == 0 and ideal_memory_ps == 0) return "unknown";
    if (ideal_compute_ps == ideal_memory_ps) return "balanced";
    if (ideal_compute_ps > ideal_memory_ps) return "compute";
    return "memory";
}

fn verifyBindingBelongsToReport(report: core.TraceReport, binding: core.BackendBinding, diagnostics: *std.Io.Writer) !void {
    const index: usize = std.math.cast(usize, binding.id.index) orelse {
        try diagnostics.print("pass=backend-executable feature=backend-binding reason=binding index does not fit host index binding={d}\n", .{binding.id.index});
        return BackendError.InvalidExecutablePlan;
    };
    if (index >= report.backend_bindings.len) {
        try diagnostics.print("pass=backend-executable feature=backend-binding reason=binding is not present in report binding={d}\n", .{binding.id.index});
        return BackendError.InvalidExecutablePlan;
    }
    if (!backendBindingMatches(report.backend_bindings[index], binding)) {
        try diagnostics.print("pass=backend-executable feature=backend-binding reason=binding does not match report binding={d}\n", .{binding.id.index});
        return BackendError.InvalidExecutablePlan;
    }
}

fn verifyBindingMatchesTarget(report: core.TraceReport, binding: core.BackendBinding, diagnostics: *std.Io.Writer) !void {
    const target = report.target orelse return;
    const expected = backendKindForTarget(target.kind);
    if (binding.backend_kind != expected) {
        try diagnostics.print(
            "pass=backend-executable feature=target reason=binding backend does not match selected target target={s} backend={s}\n",
            .{ @tagName(target.kind), @tagName(binding.backend_kind) },
        );
        return BackendError.InvalidExecutablePlan;
    }
}

fn backendKindForTarget(kind: target_pkg.TargetKind) BackendKind {
    return switch (kind) {
        .metal_v0 => .metal_v0,
        .npu_v0 => .npu_v0,
    };
}

fn backendBindingMatches(left: core.BackendBinding, right: core.BackendBinding) bool {
    return left.id.index == right.id.index and
        left.command_id.index == right.command_id.index and
        left.backend_kind == right.backend_kind and
        std.mem.eql(u8, left.backend_operation, right.backend_operation) and
        optionalUnitIdMatches(left.expected_unit_id, right.expected_unit_id) and
        graphInstructionIdsMatch(left.graph_instruction_ids, right.graph_instruction_ids) and
        costLedgerIdsMatch(left.cost_ledger_ids, right.cost_ledger_ids);
}

fn optionalUnitIdMatches(left: ?u32, right: ?u32) bool {
    if (left == null and right == null) return true;
    if (left == null or right == null) return false;
    return left.? == right.?;
}

fn graphInstructionIdsMatch(left: []const compiler_facts.GraphInstructionId, right: []const compiler_facts.GraphInstructionId) bool {
    if (left.len != right.len) return false;
    for (left, right) |left_id, right_id| {
        if (left_id.index != right_id.index) return false;
    }
    return true;
}

fn costLedgerIdsMatch(left: []const compiler_facts.CostLedgerId, right: []const compiler_facts.CostLedgerId) bool {
    if (left.len != right.len) return false;
    for (left, right) |left_id, right_id| {
        if (left_id.index != right_id.index) return false;
    }
    return true;
}

fn commandForBinding(report: core.TraceReport, binding: core.BackendBinding, diagnostics: *std.Io.Writer) !core.ScheduleCommand {
    const index: usize = std.math.cast(usize, binding.command_id.index) orelse {
        try diagnostics.print("pass=backend-executable feature=command reason=command index does not fit host index command={d}\n", .{binding.command_id.index});
        return BackendError.InvalidExecutablePlan;
    };
    if (index >= report.schedule_commands.len) {
        try diagnostics.print("pass=backend-executable feature=command reason=binding references unknown command command={d}\n", .{binding.command_id.index});
        return BackendError.InvalidExecutablePlan;
    }
    return report.schedule_commands[index];
}

fn loweringRecord(report: core.TraceReport, id: compiler_facts.LoweringRecordId, diagnostics: *std.Io.Writer) !compiler_facts.LoweringRecord {
    const index: usize = std.math.cast(usize, id.index) orelse {
        try diagnostics.print("pass=backend-executable feature=lowering reason=lowering index does not fit host index lowering={d}\n", .{id.index});
        return BackendError.InvalidExecutablePlan;
    };
    if (index >= report.lowering_records.len) {
        try diagnostics.print("pass=backend-executable feature=lowering reason=command references unknown lowering lowering={d}\n", .{id.index});
        return BackendError.InvalidExecutablePlan;
    }
    return report.lowering_records[index];
}

fn verifyLoweringBelongsToBinding(lowering: compiler_facts.LoweringRecord, binding: core.BackendBinding, diagnostics: *std.Io.Writer) !void {
    for (lowering.graph_instruction_ids) |instruction_id| {
        if (!instructionInBinding(binding, instruction_id)) {
            try diagnostics.print(
                "pass=backend-executable feature=lowering reason=lowering instruction is outside backend binding lowering={d} instruction={d} binding={d}\n",
                .{ lowering.id.index, instruction_id.index, binding.id.index },
            );
            return BackendError.InvalidExecutablePlan;
        }
    }
}

fn instructionInBinding(binding: core.BackendBinding, instruction_id: compiler_facts.GraphInstructionId) bool {
    for (binding.graph_instruction_ids) |binding_instruction_id| {
        if (binding_instruction_id.eql(instruction_id)) return true;
    }
    return false;
}

fn graphInstruction(report: core.TraceReport, id: compiler_facts.GraphInstructionId) compiler_facts.GraphInstruction {
    const index: usize = std.math.cast(usize, id.index) orelse unreachable;
    return report.graph_instructions[index];
}

fn fusedElementwiseOperationForBackend(kind: core.BackendKind) []const u8 {
    return switch (kind) {
        .metal_v0 => "metal_mls_elementwise_fusion_kernel",
        .npu_v0 => "npu_elementwise_fusion",
    };
}

fn expectedUnitForLowering(report: core.TraceReport, capabilities: BackendCapabilitySet, lowering: compiler_facts.LoweringRecord) ?u32 {
    var selected_unit: ?u32 = null;
    var saw_unit = false;
    for (lowering.graph_instruction_ids) |instruction_id| {
        const capability = capabilityForInstruction(capabilities, graphInstruction(report, instruction_id), report.graph_values) orelse return null;
        const expected_unit_id = capability.expected_unit_id orelse return null;
        if (!saw_unit) {
            selected_unit = expected_unit_id;
            saw_unit = true;
        } else if (selected_unit != expected_unit_id) {
            return null;
        }
    }
    return selected_unit;
}

fn copyGraphValueIds(allocator: std.mem.Allocator, ids: []const compiler_facts.GraphValueId) ![]const compiler_facts.GraphValueId {
    const copy = try allocator.alloc(compiler_facts.GraphValueId, ids.len);
    @memcpy(copy, ids);
    return copy;
}

fn copyGraphInstructionIds(allocator: std.mem.Allocator, ids: []const compiler_facts.GraphInstructionId) ![]const compiler_facts.GraphInstructionId {
    const copy = try allocator.alloc(compiler_facts.GraphInstructionId, ids.len);
    @memcpy(copy, ids);
    return copy;
}

fn regionInputValueIds(
    allocator: std.mem.Allocator,
    report: core.TraceReport,
    instruction_ids: []const compiler_facts.GraphInstructionId,
    diagnostics: *std.Io.Writer,
) ![]const compiler_facts.GraphValueId {
    var values: std.ArrayList(compiler_facts.GraphValueId) = .empty;
    errdefer values.deinit(allocator);
    for (instruction_ids) |instruction_id| {
        const instruction = graphInstruction(report, instruction_id);
        for (instruction.inputs) |input_id| {
            if (regionProducesValue(report, instruction_ids, input_id)) continue;
            try appendUniqueValueId(allocator, &values, input_id);
        }
    }
    if (values.items.len == 0) {
        try diagnostics.print("pass=backend-executable feature=fusion reason=fused region has no external inputs instruction={d}\n", .{instruction_ids[0].index});
        return BackendError.InvalidExecutablePlan;
    }
    return values.toOwnedSlice(allocator);
}

fn regionOutputValueIds(
    allocator: std.mem.Allocator,
    report: core.TraceReport,
    instruction_ids: []const compiler_facts.GraphInstructionId,
    diagnostics: *std.Io.Writer,
) ![]const compiler_facts.GraphValueId {
    var values: std.ArrayList(compiler_facts.GraphValueId) = .empty;
    errdefer values.deinit(allocator);
    for (instruction_ids) |instruction_id| {
        const instruction = graphInstruction(report, instruction_id);
        for (instruction.outputs) |output_id| {
            if (regionConsumesValue(report, instruction_ids, output_id)) continue;
            try appendUniqueValueId(allocator, &values, output_id);
        }
    }
    if (values.items.len != 1) {
        try diagnostics.print("pass=backend-executable feature=fusion reason=fused region must have exactly one external output outputs={d}\n", .{values.items.len});
        return BackendError.InvalidExecutablePlan;
    }
    return values.toOwnedSlice(allocator);
}

fn appendUniqueValueId(allocator: std.mem.Allocator, values: *std.ArrayList(compiler_facts.GraphValueId), value_id: compiler_facts.GraphValueId) !void {
    for (values.items) |existing| {
        if (existing.eql(value_id)) return;
    }
    try values.append(allocator, value_id);
}

fn regionProducesValue(report: core.TraceReport, instruction_ids: []const compiler_facts.GraphInstructionId, value_id: compiler_facts.GraphValueId) bool {
    for (instruction_ids) |instruction_id| {
        const instruction = graphInstruction(report, instruction_id);
        for (instruction.outputs) |output_id| {
            if (output_id.eql(value_id)) return true;
        }
    }
    return false;
}

fn regionConsumesValue(report: core.TraceReport, instruction_ids: []const compiler_facts.GraphInstructionId, value_id: compiler_facts.GraphValueId) bool {
    for (instruction_ids) |instruction_id| {
        const instruction = graphInstruction(report, instruction_id);
        for (instruction.inputs) |input_id| {
            if (input_id.eql(value_id)) return true;
        }
    }
    return false;
}

fn kernelOutputType(
    allocator: std.mem.Allocator,
    report: core.TraceReport,
    call: BackendExecutableCall,
    diagnostics: *std.Io.Writer,
) !BackendTensorDescriptor {
    if (call.output_value_ids.len != 1) {
        try diagnostics.print(
            "pass=backend-kernel-graph feature=output-type reason=kernel graph node must have exactly one output call={d} outputs={d}\n",
            .{ call.index, call.output_value_ids.len },
        );
        return BackendError.InvalidExecutablePlan;
    }
    const value = graphValue(report, call.output_value_ids[0], diagnostics) catch return BackendError.InvalidExecutablePlan;
    const dims = try allocator.alloc(i64, value.ty.dims.len);
    @memcpy(dims, value.ty.dims);
    return .{
        .element_type = value.ty.element_type,
        .dims = dims,
        .layout = value.ty.layout,
    };
}

fn kernelAttributes(
    allocator: std.mem.Allocator,
    report: core.TraceReport,
    call: BackendExecutableCall,
    instruction: compiler_facts.GraphInstruction,
    diagnostics: *std.Io.Writer,
) !BackendKernelAttributes {
    _ = report;
    if (call.feature == .elementwise_fusion) {
        return .{ .elementwise_fusion = try copyGraphInstructionIds(allocator, call.graph_instruction_ids) };
    }
    return switch (instruction.payload) {
        .dot_general => |spec| .{ .rank2_dot_general = spec },
        .broadcast => |spec| .{ .broadcast_in_dim = try copyBroadcastDimensions(allocator, spec.dimensions) },
        .elementwise_binary => |spec| switch (spec.op) {
            .add => .{ .add = {} },
        },
        .elementwise_unary => |spec| switch (spec.op) {
            .tanh => .{ .tanh = {} },
        },
        .reshape, .transpose => {
            try diagnostics.print(
                "pass=backend-kernel-graph feature=attributes reason=non-identity shape/layout ops are unsupported backend nodes in V0 instruction={d}\n",
                .{instruction.id.index},
            );
            return BackendError.InvalidExecutablePlan;
        },
        .collective => {
            try diagnostics.print(
                "pass=backend-kernel-graph feature=attributes reason=collective ops must be lowered by collective_algorithm_select before backend kernel graph creation instruction={d}\n",
                .{instruction.id.index},
            );
            return BackendError.InvalidExecutablePlan;
        },
        .return_ => {
            try diagnostics.print(
                "pass=backend-kernel-graph feature=attributes reason=return is not a kernel graph node instruction={d}\n",
                .{instruction.id.index},
            );
            return BackendError.InvalidExecutablePlan;
        },
    };
}

fn copyBroadcastDimensions(allocator: std.mem.Allocator, dimensions: []const u32) ![]const u32 {
    const copy = try allocator.alloc(u32, dimensions.len);
    @memcpy(copy, dimensions);
    return copy;
}

fn deinitKernelAttributes(allocator: std.mem.Allocator, attributes: BackendKernelAttributes) void {
    switch (attributes) {
        .broadcast_in_dim => |dimensions| allocator.free(dimensions),
        .elementwise_fusion => |instruction_ids| allocator.free(instruction_ids),
        .rank2_dot_general, .add, .tanh => {},
    }
}

fn writeInstructionIdList(writer: *std.Io.Writer, ids: []const compiler_facts.GraphInstructionId) std.Io.Writer.Error!void {
    for (ids, 0..) |id, index| {
        if (index != 0) try writer.writeAll(",");
        try writer.print("{d}", .{id.index});
    }
}

fn graphValue(report: core.TraceReport, id: compiler_facts.GraphValueId, diagnostics: *std.Io.Writer) !compiler_facts.GraphValue {
    const index: usize = std.math.cast(usize, id.index) orelse {
        try diagnostics.print("pass=backend-kernel-graph feature=value reason=value index does not fit host index value={d}\n", .{id.index});
        return BackendError.InvalidExecutablePlan;
    };
    if (index >= report.graph_values.len) {
        try diagnostics.print("pass=backend-kernel-graph feature=value reason=value is out of bounds value={d}\n", .{id.index});
        return BackendError.InvalidExecutablePlan;
    }
    return report.graph_values[index];
}

fn producerNodeForValue(
    nodes: []const BackendKernelGraphNode,
    value_id: compiler_facts.GraphValueId,
    diagnostics: *std.Io.Writer,
) !?BackendKernelGraphNodeId {
    var producer: ?BackendKernelGraphNodeId = null;
    for (nodes) |node| {
        for (node.output_value_ids) |output_value_id| {
            if (!output_value_id.eql(value_id)) continue;
            if (producer != null) {
                try diagnostics.print(
                    "pass=backend-kernel-graph feature=value-flow reason=value has multiple producer nodes value={d} first_node={d} second_node={d}\n",
                    .{ value_id.index, producer.?.index, node.id.index },
                );
                return BackendError.InvalidExecutablePlan;
            }
            producer = node.id;
        }
    }
    return producer;
}

fn instructionDtype(instruction: compiler_facts.GraphInstruction, values: []const compiler_facts.GraphValue) ?core.BufferType {
    const value_id = if (instruction.outputs.len > 0) instruction.outputs[0] else if (instruction.inputs.len > 0) instruction.inputs[0] else return null;
    const index: usize = std.math.cast(usize, value_id.index) orelse return null;
    if (index >= values.len) return null;
    return values[index].ty.element_type;
}

fn supportsDtype(dtypes: []const core.BufferType, dtype: core.BufferType) bool {
    for (dtypes) |supported| {
        if (supported == dtype) return true;
    }
    return false;
}

const npu_matrix_dtypes = [_]core.BufferType{ .bf16, .f32 };
const npu_vector_dtypes = [_]core.BufferType{.f32};
const npu_capabilities = [_]BackendCapability{
    .{ .feature = .rank2_dot_general, .dtypes = &npu_matrix_dtypes, .backend_operation = "npu_matmul", .expected_unit_id = 0 },
    .{ .feature = .broadcast_in_dim, .dtypes = &npu_vector_dtypes, .backend_operation = "npu_elementwise_fusion", .expected_unit_id = 1 },
    .{ .feature = .add, .dtypes = &npu_vector_dtypes, .backend_operation = "npu_elementwise_fusion", .expected_unit_id = 1 },
    .{ .feature = .tanh, .dtypes = &npu_vector_dtypes, .backend_operation = "npu_elementwise_fusion", .expected_unit_id = 1 },
};

const metal_dtypes = [_]core.BufferType{ .f16, .f32 };
const metal_capabilities = [_]BackendCapability{
    .{ .feature = .rank2_dot_general, .dtypes = &metal_dtypes, .backend_operation = "metal_mls_matmul_kernel", .expected_unit_id = 0 },
    .{ .feature = .broadcast_in_dim, .dtypes = &metal_dtypes, .backend_operation = "metal_mls_broadcast_kernel", .expected_unit_id = 0 },
    .{ .feature = .add, .dtypes = &metal_dtypes, .backend_operation = "metal_mls_add_kernel", .expected_unit_id = 0 },
    .{ .feature = .tanh, .dtypes = &metal_dtypes, .backend_operation = "metal_mls_tanh_kernel", .expected_unit_id = 0 },
};

test "backend kind writes stable target names" {
    var output = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer output.deinit();

    try writeBackendKind(&output.writer, .npu_v0);
    try std.testing.expectEqualStrings("npu_v0", output.writer.buffered());
}

test "backend package imports only new core namespace" {
    try std.testing.expect(core.idInBounds(0, 1));
}

test "NPU capability supports V0 f32 tanh but rejects f64" {
    const capabilities: BackendCapabilitySet = capabilitySet(.npu_v0);

    try std.testing.expect(supportsInstruction(capabilities, f32_tanh_instruction, &f32_tanh_values));
    try std.testing.expect(!supportsInstruction(capabilities, f64_tanh_instruction, &f64_tanh_values));
}

test "backend executable plan expands Metal binding to concrete calls" {
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    var plan: BackendExecutablePlan = try planExecutable(std.testing.allocator, executable_report, executable_binding, &diagnostics.writer);
    defer plan.deinit();

    try std.testing.expectEqual(core.BackendKind.metal_v0, plan.backend_kind);
    try std.testing.expectEqualStrings("metal_mls_graph_execute", plan.backend_operation);
    try std.testing.expectEqual(1, plan.calls.len);
    try std.testing.expectEqual(BackendFeature.tanh, plan.calls[0].feature);
    try std.testing.expectEqualStrings("metal_mls_tanh_kernel", plan.calls[0].backend_operation);

    var output: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();
    try writeExecutablePlanSummary(plan, &output.writer);
    try std.testing.expect(std.mem.indexOf(u8, output.writer.buffered(), "backend=metal_v0 command=0 operation=metal_mls_graph_execute calls=1") != null);
    try std.testing.expect(std.mem.indexOf(u8, output.writer.buffered(), "call.0 instruction=0 instructions=0 feature=tanh operation=metal_mls_tanh_kernel unit=0 inputs=1 outputs=1") != null);
}

test "backend executable plan rejects operation mismatch" {
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    try std.testing.expectError(
        BackendError.InvalidExecutablePlan,
        planExecutable(std.testing.allocator, mismatched_operation_report, mismatched_operation_binding, &diagnostics.writer),
    );
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "binding operation does not match backend executable") != null);
}

test "backend executable plan rejects binding outside verified report" {
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    try std.testing.expectError(
        BackendError.InvalidExecutablePlan,
        planExecutable(std.testing.allocator, executable_report, mismatched_operation_binding, &diagnostics.writer),
    );
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "binding does not match report") != null);
}

test "kernel graph planning is Metal-only in V0" {
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    const calls = [_]BackendExecutableCall{};
    const plan: BackendExecutablePlan = .{
        .allocator = std.testing.allocator,
        .backend_kind = .npu_v0,
        .command_id = .{ .index = 0 },
        .backend_operation = "npu_execute",
        .calls = &calls,
    };
    try std.testing.expectError(
        BackendError.InvalidExecutablePlan,
        planKernelGraph(std.testing.allocator, executable_report, plan, &diagnostics.writer),
    );
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "Metal-only") != null);
}

const tanh_source: compiler_facts.SourceRef = .{ .id = .{ .index = 0 }, .frontend = .stablehlo, .op_name = "stablehlo.tanh", .source_index = 0, .location = "" };
const tanh_inputs = [_]compiler_facts.GraphValueId{.{ .index = 0 }};
const tanh_outputs = [_]compiler_facts.GraphValueId{.{ .index = 1 }};
const tanh_dims = [_]i64{4};
const f32_tanh_values = [_]compiler_facts.GraphValue{
    .{ .id = .{ .index = 0 }, .ty = .{ .element_type = .f32, .dims = &tanh_dims, .layout = .dense_row_major }, .role = .parameter, .source = null },
    .{ .id = .{ .index = 1 }, .ty = .{ .element_type = .f32, .dims = &tanh_dims, .layout = .dense_row_major }, .role = .instruction_result, .source = tanh_source },
};
const f64_tanh_values = [_]compiler_facts.GraphValue{
    .{ .id = .{ .index = 0 }, .ty = .{ .element_type = .f64, .dims = &tanh_dims, .layout = .dense_row_major }, .role = .parameter, .source = null },
    .{ .id = .{ .index = 1 }, .ty = .{ .element_type = .f64, .dims = &tanh_dims, .layout = .dense_row_major }, .role = .instruction_result, .source = tanh_source },
};
const f32_tanh_instruction: compiler_facts.GraphInstruction = .{
    .id = .{ .index = 0 },
    .kind = .elementwise_unary,
    .inputs = &tanh_inputs,
    .outputs = &tanh_outputs,
    .payload = .{ .elementwise_unary = .{ .op = .tanh } },
    .source = tanh_source,
};
const f64_tanh_instruction = f32_tanh_instruction;
const executable_instruction_ids = [_]compiler_facts.GraphInstructionId{.{ .index = 0 }};
const executable_cost_ids = [_]compiler_facts.CostLedgerId{.{ .index = 0 }};
const executable_lowering_ids = [_]compiler_facts.LoweringRecordId{.{ .index = 0 }};
const executable_costs = [_]compiler_facts.CostLedgerEntry{
    .{ .id = .{ .index = 0 }, .source = tanh_source, .graph_instruction_ids = &executable_instruction_ids, .op_class = .transcendental, .dtype = .f32, .accumulation_dtype = null, .logical_ops = 4, .bytes_read = 16, .bytes_written = 16, .expected_unit_id = 0, .formula = "numel(output)", .approximation = "" },
};
const executable_lowerings = [_]compiler_facts.LoweringRecord{
    .{ .id = .{ .index = 0 }, .graph_instruction_ids = &executable_instruction_ids, .decision = .backend_kernel_graph, .reason = "test", .rejected_alternatives = &.{}, .cost_ledger_ids = &executable_cost_ids },
};
const executable_commands = [_]core.ScheduleCommand{
    .{ .id = .{ .index = 0 }, .kind = .backend_execute, .stream = .{ .index = 0 }, .inputs = &tanh_inputs, .outputs = &tanh_outputs, .dependencies = &.{}, .lowering_record_ids = &executable_lowering_ids, .cost_ledger_ids = &executable_cost_ids },
};
const executable_binding: core.BackendBinding = .{
    .id = .{ .index = 0 },
    .command_id = .{ .index = 0 },
    .backend_kind = .metal_v0,
    .backend_operation = "metal_mls_graph_execute",
    .graph_instruction_ids = &executable_instruction_ids,
    .expected_unit_id = 0,
    .cost_ledger_ids = &executable_cost_ids,
};
const executable_bindings = [_]core.BackendBinding{executable_binding};
const mismatched_operation_binding: core.BackendBinding = .{
    .id = .{ .index = 0 },
    .command_id = .{ .index = 0 },
    .backend_kind = .metal_v0,
    .backend_operation = "reference_execute",
    .graph_instruction_ids = &executable_instruction_ids,
    .expected_unit_id = 0,
    .cost_ledger_ids = &executable_cost_ids,
};
const mismatched_operation_bindings = [_]core.BackendBinding{mismatched_operation_binding};
const executable_instructions = [_]compiler_facts.GraphInstruction{f32_tanh_instruction};
const executable_report: core.TraceReport = .{
    .sources = &.{tanh_source},
    .graph_values = &f32_tanh_values,
    .graph_instructions = &executable_instructions,
    .cost_ledger = &executable_costs,
    .lowering_records = &executable_lowerings,
    .schedule_commands = &executable_commands,
    .backend_bindings = &executable_bindings,
    .profile_events = &.{},
    .explain_records = &.{},
};
const mismatched_operation_report: core.TraceReport = .{
    .sources = &.{tanh_source},
    .graph_values = &f32_tanh_values,
    .graph_instructions = &executable_instructions,
    .cost_ledger = &executable_costs,
    .lowering_records = &executable_lowerings,
    .schedule_commands = &executable_commands,
    .backend_bindings = &mismatched_operation_bindings,
    .profile_events = &.{},
    .explain_records = &.{},
};
