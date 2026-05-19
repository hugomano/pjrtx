const std = @import("std");
const compiler_facts = @import("pjrtx/compiler/facts");
const target_pkg = @import("pjrtx/target");

/// Deprecated bootstrap vocabulary for the first vertical slice.
/// New schema belongs in the package that owns its invariants; this file should
/// shrink as facts move to target/compiler/backend/runtime/report packages.
pub const SourceId = compiler_facts.SourceId;

pub const GraphValueId = compiler_facts.GraphValueId;

pub const GraphInstructionId = compiler_facts.GraphInstructionId;

pub const CostLedgerId = compiler_facts.CostLedgerId;

pub const MemoryTrafficId = compiler_facts.MemoryTrafficId;

pub const LoweringRecordId = compiler_facts.LoweringRecordId;

pub const ScheduleOverlapId = struct {
    index: u32,

    pub fn eql(self: ScheduleOverlapId, other: ScheduleOverlapId) bool {
        return self.index == other.index;
    }
};

pub const ScheduleCommandId = struct {
    index: u32,

    pub fn eql(self: ScheduleCommandId, other: ScheduleCommandId) bool {
        return self.index == other.index;
    }
};

pub const KernelCodegenId = struct {
    index: u32,

    pub fn eql(self: KernelCodegenId, other: KernelCodegenId) bool {
        return self.index == other.index;
    }
};

pub const BackendBindingId = struct {
    index: u32,

    pub fn eql(self: BackendBindingId, other: BackendBindingId) bool {
        return self.index == other.index;
    }
};

pub const ProfileEventId = struct {
    index: u32,

    pub fn eql(self: ProfileEventId, other: ProfileEventId) bool {
        return self.index == other.index;
    }
};

pub const ExplainRecordId = struct {
    index: u32,

    pub fn eql(self: ExplainRecordId, other: ExplainRecordId) bool {
        return self.index == other.index;
    }
};

pub fn writeId(writer: *std.Io.Writer, prefix: []const u8, index: u32) std.Io.Writer.Error!void {
    try writer.print("{s}.{d}", .{ prefix, index });
}

pub fn idInBounds(index: u32, len: usize) bool {
    const position: usize = std.math.cast(usize, index) orelse return false;
    return position < len;
}

pub const ValidationError = error{
    InvalidTensorType,
    InvalidTargetDescription,
    InvalidTraceReport,
};

pub const BufferType = target_pkg.BufferType;

pub const LayoutKind = compiler_facts.LayoutKind;

pub const SourceFrontend = compiler_facts.SourceFrontend;

pub const SourceRef = compiler_facts.SourceRef;

pub const TensorType = compiler_facts.TensorType;

pub const GraphValueRole = compiler_facts.GraphValueRole;

pub const GraphValue = compiler_facts.GraphValue;

pub const GraphInstructionKind = compiler_facts.GraphInstructionKind;

pub const ElementwiseUnaryOp = compiler_facts.ElementwiseUnaryOp;

pub const ElementwiseBinaryOp = compiler_facts.ElementwiseBinaryOp;

pub const DotGeneralSpec = compiler_facts.DotGeneralSpec;

pub const ElementwiseUnarySpec = compiler_facts.ElementwiseUnarySpec;

pub const ElementwiseBinarySpec = compiler_facts.ElementwiseBinarySpec;

pub const BroadcastSpec = compiler_facts.BroadcastSpec;

pub const ReshapeSpec = compiler_facts.ReshapeSpec;

pub const TransposeSpec = compiler_facts.TransposeSpec;

pub const CollectiveOp = compiler_facts.CollectiveOp;

pub const CollectiveReduction = compiler_facts.CollectiveReduction;

pub const CollectiveSpec = compiler_facts.CollectiveSpec;

pub const ReturnSpec = compiler_facts.ReturnSpec;

pub const GraphPayload = compiler_facts.GraphPayload;

pub const GraphInstruction = compiler_facts.GraphInstruction;

pub const MlirPassStatus = compiler_facts.MlirPassStatus;

pub const MlirPassRecord = compiler_facts.MlirPassRecord;

pub const GraphRewriteDecision = compiler_facts.GraphRewriteDecision;

pub const GraphRewriteRecord = compiler_facts.GraphRewriteRecord;

pub const FusionDecision = compiler_facts.FusionDecision;

pub const FusionPressureDelta = compiler_facts.FusionPressureDelta;

pub const FusionGroup = compiler_facts.FusionGroup;

pub const PlacementRecord = compiler_facts.PlacementRecord;

pub const CollectivePlanDecision = compiler_facts.CollectivePlanDecision;

pub const CollectiveAlgorithm = compiler_facts.CollectiveAlgorithm;

pub const CollectivePlanRecord = compiler_facts.CollectivePlanRecord;

pub const CostOpClass = compiler_facts.CostOpClass;

pub const CostLedgerEntry = compiler_facts.CostLedgerEntry;

pub const MemoryTrafficKind = compiler_facts.MemoryTrafficKind;

pub const MemoryTrafficRecord = compiler_facts.MemoryTrafficRecord;

pub const LoweringDecision = compiler_facts.LoweringDecision;

pub const LoweringRecord = compiler_facts.LoweringRecord;

pub const StreamId = struct {
    index: u32,
};

pub const CommandKind = enum {
    host_to_device,
    backend_execute,
    device_to_host,
    event_record,
    event_wait,
};

pub const DependencyKind = enum {
    data,
    stream_order,
    memory_availability,
};

pub const CommandDependency = struct {
    command_id: ScheduleCommandId,
    kind: DependencyKind,
};

pub const ScheduleCommand = struct {
    id: ScheduleCommandId,
    kind: CommandKind,
    stream: StreamId,
    inputs: []const GraphValueId,
    outputs: []const GraphValueId,
    dependencies: []const CommandDependency,
    lowering_record_ids: []const LoweringRecordId,
    cost_ledger_ids: []const CostLedgerId,
};

pub const ScheduleOverlapDecision = enum {
    serialized,
    candidate,
    selected,
    rejected,
};

pub const ScheduleOverlapKind = enum {
    transfer_compute,
    compute_transfer,
    transfer_transfer,
    collective_compute,
};

pub const ScheduleOverlapRecord = struct {
    id: ScheduleOverlapId,
    decision: ScheduleOverlapDecision,
    kind: ScheduleOverlapKind,
    first_command_id: ScheduleCommandId,
    second_command_id: ScheduleCommandId,
    dependency_kind: DependencyKind,
    first_stream: StreamId,
    second_stream: StreamId,
    reason: []const u8,
};

pub const BackendKind = enum {
    metal_v0,
    npu_v0,
};

pub const OpClass = target_pkg.OpClass;

pub const RateSource = target_pkg.RateSource;

pub const KernelCodegenKind = enum {
    backend_kernel_graph,
    elementwise_fusion_kernel,
    library_call,
    collective_engine,
};

pub const KernelCodegenShape = struct {
    operation_count: u32,
    external_input_count: u32,
    external_output_count: u32,
    intermediate_value_count: u32,
};

pub const KernelMemoryPressure = struct {
    global_bytes_read: u128,
    global_bytes_written: u128,
    local_bytes_read: u128,
    local_bytes_written: u128,
};

pub const KernelCodegenRecord = struct {
    id: KernelCodegenId,
    lowering_record_id: LoweringRecordId,
    command_id: ScheduleCommandId,
    backend_kind: BackendKind,
    kind: KernelCodegenKind,
    operation: []const u8,
    shape: KernelCodegenShape,
    logical_tile_shape: []const i64,
    result_memory_space_id: u32,
    tile_memory_space_id: ?u32,
    memory_pressure: KernelMemoryPressure,
    external_input_ids: []const GraphValueId,
    external_output_ids: []const GraphValueId,
    intermediate_value_ids: []const GraphValueId,
    graph_instruction_ids: []const GraphInstructionId,
    cost_ledger_ids: []const CostLedgerId,
    memory_traffic_ids: []const MemoryTrafficId,
    expected_unit_id: ?u32,
    reason: []const u8,
};

pub const BackendBinding = struct {
    id: BackendBindingId,
    command_id: ScheduleCommandId,
    backend_kind: BackendKind,
    backend_operation: []const u8,
    graph_instruction_ids: []const GraphInstructionId,
    expected_unit_id: ?u32,
    cost_ledger_ids: []const CostLedgerId,
};

pub const ProfileEventKind = enum {
    compile_pass,
    h2d,
    backend_execute,
    d2h,
};

pub const ProfileStatus = enum {
    ok,
    failed,
};

pub const ProfileEvent = struct {
    id: ProfileEventId,
    command_id: ?ScheduleCommandId,
    graph_instruction_ids: []const GraphInstructionId,
    kind: ProfileEventKind,
    start_ns: u64,
    duration_ns: u64,
    bytes: u128,
    logical_ops: u128,
    status: ProfileStatus,
    forced_synchronization: bool,
};

pub const ExplainSubject = union(enum) {
    graph_instruction: GraphInstructionId,
    lowering_record: LoweringRecordId,
    schedule_command: ScheduleCommandId,
    backend_binding: BackendBindingId,
};

pub const ExplainRecord = struct {
    id: ExplainRecordId,
    pass_name: []const u8,
    subject: ExplainSubject,
    decision: []const u8,
    reason: []const u8,
    source_refs: []const SourceRef,
    cost_ledger_ids: []const CostLedgerId,
    profile_event_ids: []const ProfileEventId,
};

pub const TraceReport = struct {
    sources: []const SourceRef,
    target: ?target_pkg.TargetDescription = null,
    graph_values: []const GraphValue,
    graph_instructions: []const GraphInstruction,
    mlir_pass_records: []const MlirPassRecord = &.{},
    graph_rewrite_records: []const GraphRewriteRecord = &.{},
    fusion_groups: []const FusionGroup = &.{},
    placement_records: []const PlacementRecord = &.{},
    collective_plan_records: []const CollectivePlanRecord = &.{},
    cost_ledger: []const CostLedgerEntry,
    lowering_records: []const LoweringRecord,
    memory_traffic_records: []const MemoryTrafficRecord = &.{},
    schedule_overlap_records: []const ScheduleOverlapRecord = &.{},
    schedule_commands: []const ScheduleCommand,
    kernel_codegen_records: []const KernelCodegenRecord = &.{},
    backend_bindings: []const BackendBinding,
    profile_events: []const ProfileEvent,
    explain_records: []const ExplainRecord,
};

pub fn validateTargetDescription(description: target_pkg.TargetDescription, diagnostics: *std.Io.Writer) !void {
    target_pkg.validateTargetDescription(description, diagnostics) catch |err| switch (err) {
        error.InvalidTargetDescription => return ValidationError.InvalidTargetDescription,
        else => return err,
    };
}

/// The target summary is a stable human-readable schema, not debug prose. Tests
/// depend on unknown fields being printed explicitly so missing hardware facts
/// cannot disappear from reports.
pub fn writeTargetSummary(description: target_pkg.TargetDescription, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    try target_pkg.writeTargetSummary(description, writer);
}

/// Report validation protects vertical traceability. Every runtime-observable
/// command must join back to lowering/cost/source records before an executable
/// can be trusted by later compiler or PJRT layers.
pub fn validateTraceReport(report: TraceReport, diagnostics: *std.Io.Writer) !void {
    for (report.sources, 0..) |source, index| {
        const expected_index: u32 = std.math.cast(u32, index) orelse unreachable;
        if (source.id.index != expected_index) {
            try diagnostics.print("pass=pjrtx-report-validate feature=source reason=source ID order mismatch expected={d} actual={d}\n", .{ expected_index, source.id.index });
            return ValidationError.InvalidTraceReport;
        }
    }

    for (report.graph_values, 0..) |value, index| {
        const expected_index: u32 = std.math.cast(u32, index) orelse unreachable;
        if (value.id.index != expected_index) {
            try diagnostics.print("pass=pjrtx-report-validate feature=graph-value reason=value ID order mismatch expected={d} actual={d}\n", .{ expected_index, value.id.index });
            return ValidationError.InvalidTraceReport;
        }
        try TraceGraphFacts.validateTensorType(value.ty, diagnostics);
    }

    for (report.graph_instructions, 0..) |instruction, index| {
        const expected_index: u32 = std.math.cast(u32, index) orelse unreachable;
        if (instruction.id.index != expected_index) {
            try diagnostics.print("pass=pjrtx-report-validate feature=graph-instruction reason=instruction ID order mismatch expected={d} actual={d}\n", .{ expected_index, instruction.id.index });
            return ValidationError.InvalidTraceReport;
        }
        if (!TraceGraphFacts.payloadMatchesKind(instruction.kind, instruction.payload)) {
            try diagnostics.print("pass=pjrtx-report-validate feature=graph-instruction reason=payload kind mismatch instruction={d}\n", .{instruction.id.index});
            return ValidationError.InvalidTraceReport;
        }
        try validateGraphValueRefs("instruction input", instruction.inputs, report.graph_values.len, diagnostics);
        try validateGraphValueRefs("instruction output", instruction.outputs, report.graph_values.len, diagnostics);
        switch (instruction.payload) {
            .collective => |payload| try validateCollectivePayload(instruction.id, payload, diagnostics),
            else => {},
        }
    }

    for (report.mlir_pass_records, 0..) |record, index| {
        const expected_index: u32 = std.math.cast(u32, index) orelse unreachable;
        if (record.index != expected_index) {
            try diagnostics.print("pass=pjrtx-report-validate feature=mlir-pass reason=pass ID order mismatch expected={d} actual={d}\n", .{ expected_index, record.index });
            return ValidationError.InvalidTraceReport;
        }
        if (record.pass_name.len == 0 or record.reason.len == 0) {
            try diagnostics.print("pass=pjrtx-report-validate feature=mlir-pass reason=pass record missing name or reason pass={d}\n", .{record.index});
            return ValidationError.InvalidTraceReport;
        }
    }

    for (report.graph_rewrite_records, 0..) |record, index| {
        const expected_index: u32 = std.math.cast(u32, index) orelse unreachable;
        if (record.index != expected_index) {
            try diagnostics.print("pass=pjrtx-report-validate feature=graph-rewrite reason=rewrite ID order mismatch expected={d} actual={d}\n", .{ expected_index, record.index });
            return ValidationError.InvalidTraceReport;
        }
        if (record.pass_name.len == 0 or record.reason.len == 0) {
            try diagnostics.print("pass=pjrtx-report-validate feature=graph-rewrite reason=rewrite record missing name or reason rewrite={d}\n", .{record.index});
            return ValidationError.InvalidTraceReport;
        }
        if (!idInBounds(record.input_instruction_id.index, report.graph_instructions.len)) {
            try diagnostics.print("pass=pjrtx-report-validate feature=graph-rewrite reason=rewrite references unknown input instruction rewrite={d} instruction={d}\n", .{ record.index, record.input_instruction_id.index });
            return ValidationError.InvalidTraceReport;
        }
        if (record.output_instruction_id) |output_instruction_id| {
            if (!idInBounds(output_instruction_id.index, report.graph_instructions.len)) {
                try diagnostics.print("pass=pjrtx-report-validate feature=graph-rewrite reason=rewrite references unknown output instruction rewrite={d} instruction={d}\n", .{ record.index, output_instruction_id.index });
                return ValidationError.InvalidTraceReport;
            }
        }
        if (record.replaced_value_id) |value_id| {
            if (!idInBounds(value_id.index, report.graph_values.len)) {
                try diagnostics.print("pass=pjrtx-report-validate feature=graph-rewrite reason=rewrite references unknown replaced value rewrite={d} value={d}\n", .{ record.index, value_id.index });
                return ValidationError.InvalidTraceReport;
            }
        }
        if (record.replacement_value_id) |value_id| {
            if (!idInBounds(value_id.index, report.graph_values.len)) {
                try diagnostics.print("pass=pjrtx-report-validate feature=graph-rewrite reason=rewrite references unknown replacement value rewrite={d} value={d}\n", .{ record.index, value_id.index });
                return ValidationError.InvalidTraceReport;
            }
        }
    }

    for (report.fusion_groups, 0..) |group, index| {
        const expected_index: u32 = std.math.cast(u32, index) orelse unreachable;
        if (group.index != expected_index) {
            try diagnostics.print("pass=pjrtx-report-validate feature=fusion reason=fusion ID order mismatch expected={d} actual={d}\n", .{ expected_index, group.index });
            return ValidationError.InvalidTraceReport;
        }
        if (group.kind.len == 0 or group.reason.len == 0) {
            try diagnostics.print("pass=pjrtx-report-validate feature=fusion reason=fusion record missing kind or reason fusion={d}\n", .{group.index});
            return ValidationError.InvalidTraceReport;
        }
        try validateGraphInstructionRefs("fusion graph instruction", group.graph_instruction_ids, report.graph_instructions.len, diagnostics);
    }

    for (report.placement_records, 0..) |record, index| {
        const expected_index: u32 = std.math.cast(u32, index) orelse unreachable;
        if (record.index != expected_index) {
            try diagnostics.print("pass=pjrtx-report-validate feature=placement reason=placement ID order mismatch expected={d} actual={d}\n", .{ expected_index, record.index });
            return ValidationError.InvalidTraceReport;
        }
        if (!idInBounds(record.graph_instruction_id.index, report.graph_instructions.len)) {
            try diagnostics.print("pass=pjrtx-report-validate feature=placement reason=placement references unknown instruction placement={d} instruction={d}\n", .{ record.index, record.graph_instruction_id.index });
            return ValidationError.InvalidTraceReport;
        }
        try validateGraphValueRefs("placement output", record.output_value_ids, report.graph_values.len, diagnostics);
        for (record.logical_tile_shape) |dim| {
            if (dim < 0) {
                try diagnostics.print("pass=pjrtx-report-validate feature=placement reason=placement has dynamic tile dimension placement={d}\n", .{record.index});
                return ValidationError.InvalidTraceReport;
            }
        }
        if (report.target) |target| {
            if (!hasMemorySpace(target.memory_spaces, record.result_memory_space_id)) {
                try diagnostics.print("pass=pjrtx-report-validate feature=placement reason=unknown result memory space placement={d} memory={d}\n", .{ record.index, record.result_memory_space_id });
                return ValidationError.InvalidTraceReport;
            }
            if (record.tile_memory_space_id) |memory_space_id| {
                if (!hasMemorySpace(target.memory_spaces, memory_space_id)) {
                    try diagnostics.print("pass=pjrtx-report-validate feature=placement reason=unknown tile memory space placement={d} memory={d}\n", .{ record.index, memory_space_id });
                    return ValidationError.InvalidTraceReport;
                }
            }
        }
    }

    for (report.collective_plan_records, 0..) |record, index| {
        const expected_index: u32 = std.math.cast(u32, index) orelse unreachable;
        if (record.index != expected_index) {
            try diagnostics.print("pass=pjrtx-report-validate feature=collective reason=collective record ID order mismatch expected={d} actual={d}\n", .{ expected_index, record.index });
            return ValidationError.InvalidTraceReport;
        }
        if (record.reason.len == 0) {
            try diagnostics.print("pass=pjrtx-report-validate feature=collective reason=collective record missing reason collective={d}\n", .{record.index});
            return ValidationError.InvalidTraceReport;
        }
        if (record.decision == .no_collectives and
            (record.algorithm != .none or record.lowered_collective_count != 0 or record.unsupported_collective_count != 0 or record.estimated_bytes != 0 or record.estimated_latency_ns != null))
        {
            try diagnostics.print("pass=pjrtx-report-validate feature=collective reason=no-collectives record carries collective work collective={d}\n", .{record.index});
            return ValidationError.InvalidTraceReport;
        }
        if (record.decision == .selected and record.algorithm == .none) {
            try diagnostics.print("pass=pjrtx-report-validate feature=collective reason=selected collective has no algorithm collective={d}\n", .{record.index});
            return ValidationError.InvalidTraceReport;
        }
        if (record.decision == .rejected or record.decision == .unsupported) {
            try diagnostics.print("pass=pjrtx-report-validate feature=collective reason=non-executable collective plan cannot validate collective={d} decision={s}\n", .{ record.index, @tagName(record.decision) });
            return ValidationError.InvalidTraceReport;
        }
        const graph_instruction_count: u32 = std.math.cast(u32, report.graph_instructions.len) orelse unreachable;
        if (record.checked_graph_instruction_count > graph_instruction_count) {
            try diagnostics.print("pass=pjrtx-report-validate feature=collective reason=collective record checked too many instructions collective={d}\n", .{record.index});
            return ValidationError.InvalidTraceReport;
        }
    }

    for (report.cost_ledger, 0..) |entry, index| {
        const expected_index: u32 = std.math.cast(u32, index) orelse unreachable;
        if (entry.id.index != expected_index) {
            try diagnostics.print("pass=pjrtx-report-validate feature=cost-ledger reason=cost ID order mismatch expected={d} actual={d}\n", .{ expected_index, entry.id.index });
            return ValidationError.InvalidTraceReport;
        }
        if (entry.dtype.byteSize() == null) {
            try diagnostics.print("pass=pjrtx-report-validate feature=cost-ledger reason=invalid dtype cost={d}\n", .{entry.id.index});
            return ValidationError.InvalidTraceReport;
        }
        if (entry.formula.len == 0) {
            try diagnostics.print("pass=pjrtx-report-validate feature=cost-ledger reason=missing formula cost={d}\n", .{entry.id.index});
            return ValidationError.InvalidTraceReport;
        }
        try validateGraphInstructionRefs("cost graph instruction", entry.graph_instruction_ids, report.graph_instructions.len, diagnostics);
    }

    for (report.lowering_records, 0..) |record, index| {
        const expected_index: u32 = std.math.cast(u32, index) orelse unreachable;
        if (record.id.index != expected_index) {
            try diagnostics.print("pass=pjrtx-report-validate feature=lowering reason=lowering ID order mismatch expected={d} actual={d}\n", .{ expected_index, record.id.index });
            return ValidationError.InvalidTraceReport;
        }
        try validateGraphInstructionRefs("lowering graph instruction", record.graph_instruction_ids, report.graph_instructions.len, diagnostics);
        try validateCostRefs("lowering cost", record.cost_ledger_ids, report.cost_ledger.len, diagnostics);
    }

    for (report.memory_traffic_records, 0..) |record, index| {
        const expected_index: u32 = std.math.cast(u32, index) orelse unreachable;
        if (record.id.index != expected_index) {
            try diagnostics.print("pass=pjrtx-report-validate feature=memory-traffic reason=traffic ID order mismatch expected={d} actual={d}\n", .{ expected_index, record.id.index });
            return ValidationError.InvalidTraceReport;
        }
        if (!idInBounds(record.lowering_record_id.index, report.lowering_records.len)) {
            try diagnostics.print("pass=pjrtx-report-validate feature=memory-traffic reason=traffic references unknown lowering traffic={d} lowering={d}\n", .{ record.id.index, record.lowering_record_id.index });
            return ValidationError.InvalidTraceReport;
        }
        if (record.reason.len == 0) {
            try diagnostics.print("pass=pjrtx-report-validate feature=memory-traffic reason=traffic record missing reason traffic={d}\n", .{record.id.index});
            return ValidationError.InvalidTraceReport;
        }
        try validateGraphInstructionRefs("memory traffic graph instruction", record.graph_instruction_ids, report.graph_instructions.len, diagnostics);
        try validateCostRefs("memory traffic cost", record.cost_ledger_ids, report.cost_ledger.len, diagnostics);
        if (report.target) |target| {
            if (!hasMemorySpace(target.memory_spaces, record.memory_space_id)) {
                try diagnostics.print("pass=pjrtx-report-validate feature=memory-traffic reason=unknown memory space traffic={d} memory={d}\n", .{ record.id.index, record.memory_space_id });
                return ValidationError.InvalidTraceReport;
            }
        }
    }

    for (report.schedule_overlap_records, 0..) |record, index| {
        const expected_index: u32 = std.math.cast(u32, index) orelse unreachable;
        if (record.id.index != expected_index) {
            try diagnostics.print("pass=pjrtx-report-validate feature=schedule-overlap reason=overlap ID order mismatch expected={d} actual={d}\n", .{ expected_index, record.id.index });
            return ValidationError.InvalidTraceReport;
        }
        if (record.reason.len == 0) {
            try diagnostics.print("pass=pjrtx-report-validate feature=schedule-overlap reason=overlap record missing reason overlap={d}\n", .{record.id.index});
            return ValidationError.InvalidTraceReport;
        }
        if (!idInBounds(record.first_command_id.index, report.schedule_commands.len) or !idInBounds(record.second_command_id.index, report.schedule_commands.len)) {
            try diagnostics.print("pass=pjrtx-report-validate feature=schedule-overlap reason=overlap references unknown command overlap={d}\n", .{record.id.index});
            return ValidationError.InvalidTraceReport;
        }
        if (record.first_command_id.index >= record.second_command_id.index) {
            try diagnostics.print("pass=pjrtx-report-validate feature=schedule-overlap reason=overlap commands must be ordered overlap={d}\n", .{record.id.index});
            return ValidationError.InvalidTraceReport;
        }
    }

    for (report.schedule_commands, 0..) |command, index| {
        const expected_index: u32 = std.math.cast(u32, index) orelse unreachable;
        if (command.id.index != expected_index) {
            try diagnostics.print("pass=pjrtx-report-validate feature=schedule reason=command ID order mismatch expected={d} actual={d}\n", .{ expected_index, command.id.index });
            return ValidationError.InvalidTraceReport;
        }
        try validateGraphValueRefs("schedule input", command.inputs, report.graph_values.len, diagnostics);
        try validateGraphValueRefs("schedule output", command.outputs, report.graph_values.len, diagnostics);
        for (command.dependencies) |dependency| {
            if (!idInBounds(dependency.command_id.index, report.schedule_commands.len)) {
                try diagnostics.print("pass=pjrtx-report-validate feature=schedule reason=command references unknown dependency command={d} dependency={d}\n", .{ command.id.index, dependency.command_id.index });
                return ValidationError.InvalidTraceReport;
            }
        }
        try validateLoweringRefs("schedule lowering", command.lowering_record_ids, report.lowering_records.len, diagnostics);
        try validateCostRefs("schedule cost", command.cost_ledger_ids, report.cost_ledger.len, diagnostics);
        if (command.kind == .backend_execute and command.lowering_record_ids.len == 0) {
            try diagnostics.print("pass=pjrtx-report-validate feature=schedule reason=backend command missing lowering provenance command={d}\n", .{command.id.index});
            return ValidationError.InvalidTraceReport;
        }
    }

    for (report.kernel_codegen_records, 0..) |record, index| {
        const expected_index: u32 = std.math.cast(u32, index) orelse unreachable;
        if (record.id.index != expected_index) {
            try diagnostics.print("pass=pjrtx-report-validate feature=kernel-codegen reason=codegen ID order mismatch expected={d} actual={d}\n", .{ expected_index, record.id.index });
            return ValidationError.InvalidTraceReport;
        }
        if (!idInBounds(record.lowering_record_id.index, report.lowering_records.len)) {
            try diagnostics.print("pass=pjrtx-report-validate feature=kernel-codegen reason=codegen references unknown lowering codegen={d} lowering={d}\n", .{ record.id.index, record.lowering_record_id.index });
            return ValidationError.InvalidTraceReport;
        }
        if (!idInBounds(record.command_id.index, report.schedule_commands.len)) {
            try diagnostics.print("pass=pjrtx-report-validate feature=kernel-codegen reason=codegen references unknown command codegen={d} command={d}\n", .{ record.id.index, record.command_id.index });
            return ValidationError.InvalidTraceReport;
        }
        const command_index: usize = std.math.cast(usize, record.command_id.index) orelse unreachable;
        if (report.schedule_commands[command_index].kind != .backend_execute) {
            try diagnostics.print("pass=pjrtx-report-validate feature=kernel-codegen reason=codegen command is not backend execute codegen={d} command={d}\n", .{ record.id.index, record.command_id.index });
            return ValidationError.InvalidTraceReport;
        }
        if (record.operation.len == 0 or record.reason.len == 0) {
            try diagnostics.print("pass=pjrtx-report-validate feature=kernel-codegen reason=codegen record missing operation or reason codegen={d}\n", .{record.id.index});
            return ValidationError.InvalidTraceReport;
        }
        if (record.shape.operation_count == 0 or record.shape.external_output_count == 0) {
            try diagnostics.print("pass=pjrtx-report-validate feature=kernel-codegen reason=codegen record has invalid kernel shape codegen={d}\n", .{record.id.index});
            return ValidationError.InvalidTraceReport;
        }
        if (record.logical_tile_shape.len == 0) {
            try diagnostics.print("pass=pjrtx-report-validate feature=kernel-codegen reason=codegen record missing tile shape codegen={d}\n", .{record.id.index});
            return ValidationError.InvalidTraceReport;
        }
        for (record.logical_tile_shape) |dim| {
            if (dim <= 0) {
                try diagnostics.print("pass=pjrtx-report-validate feature=kernel-codegen reason=codegen has invalid tile dimension codegen={d}\n", .{record.id.index});
                return ValidationError.InvalidTraceReport;
            }
        }
        if (report.target) |target| {
            if (!hasMemorySpace(target.memory_spaces, record.result_memory_space_id)) {
                try diagnostics.print("pass=pjrtx-report-validate feature=kernel-codegen reason=unknown result memory space codegen={d} memory={d}\n", .{ record.id.index, record.result_memory_space_id });
                return ValidationError.InvalidTraceReport;
            }
            if (record.tile_memory_space_id) |memory_space_id| {
                if (!hasMemorySpace(target.memory_spaces, memory_space_id)) {
                    try diagnostics.print("pass=pjrtx-report-validate feature=kernel-codegen reason=unknown tile memory space codegen={d} memory={d}\n", .{ record.id.index, memory_space_id });
                    return ValidationError.InvalidTraceReport;
                }
            }
        }
        if (record.shape.external_input_count != record.external_input_ids.len or
            record.shape.external_output_count != record.external_output_ids.len or
            record.shape.intermediate_value_count != record.intermediate_value_ids.len)
        {
            try diagnostics.print("pass=pjrtx-report-validate feature=kernel-codegen reason=codegen shape counts do not match value-flow lists codegen={d}\n", .{record.id.index});
            return ValidationError.InvalidTraceReport;
        }
        try validateGraphValueRefs("kernel codegen external input", record.external_input_ids, report.graph_values.len, diagnostics);
        try validateGraphValueRefs("kernel codegen external output", record.external_output_ids, report.graph_values.len, diagnostics);
        try validateGraphValueRefs("kernel codegen intermediate value", record.intermediate_value_ids, report.graph_values.len, diagnostics);
        try validateGraphInstructionRefs("kernel codegen graph instruction", record.graph_instruction_ids, report.graph_instructions.len, diagnostics);
        try validateCostRefs("kernel codegen cost", record.cost_ledger_ids, report.cost_ledger.len, diagnostics);
        for (record.memory_traffic_ids) |traffic_id| {
            if (!idInBounds(traffic_id.index, report.memory_traffic_records.len)) {
                try diagnostics.print("pass=pjrtx-report-validate feature=kernel-codegen reason=codegen references unknown memory traffic codegen={d} traffic={d}\n", .{ record.id.index, traffic_id.index });
                return ValidationError.InvalidTraceReport;
            }
        }
    }

    for (report.backend_bindings, 0..) |binding, index| {
        const expected_index: u32 = std.math.cast(u32, index) orelse unreachable;
        if (binding.id.index != expected_index) {
            try diagnostics.print("pass=pjrtx-report-validate feature=backend-binding reason=binding ID order mismatch expected={d} actual={d}\n", .{ expected_index, binding.id.index });
            return ValidationError.InvalidTraceReport;
        }
        if (!idInBounds(binding.command_id.index, report.schedule_commands.len)) {
            try diagnostics.print("pass=pjrtx-report-validate feature=backend-binding reason=binding references unknown command binding={d} command={d}\n", .{ binding.id.index, binding.command_id.index });
            return ValidationError.InvalidTraceReport;
        }
        try validateGraphInstructionRefs("backend binding graph instruction", binding.graph_instruction_ids, report.graph_instructions.len, diagnostics);
        try validateCostRefs("backend binding cost", binding.cost_ledger_ids, report.cost_ledger.len, diagnostics);
    }

    for (report.profile_events, 0..) |event, index| {
        const expected_index: u32 = std.math.cast(u32, index) orelse unreachable;
        if (event.id.index != expected_index) {
            try diagnostics.print("pass=pjrtx-report-validate feature=profile reason=event ID order mismatch expected={d} actual={d}\n", .{ expected_index, event.id.index });
            return ValidationError.InvalidTraceReport;
        }
        if (event.command_id) |command_id| {
            if (!idInBounds(command_id.index, report.schedule_commands.len)) {
                try diagnostics.print("pass=pjrtx-report-validate feature=profile reason=event references unknown command event={d} command={d}\n", .{ event.id.index, command_id.index });
                return ValidationError.InvalidTraceReport;
            }
        }
        try validateGraphInstructionRefs("profile graph instruction", event.graph_instruction_ids, report.graph_instructions.len, diagnostics);
    }

    for (report.explain_records, 0..) |explain, index| {
        const expected_index: u32 = std.math.cast(u32, index) orelse unreachable;
        if (explain.id.index != expected_index) {
            try diagnostics.print("pass=pjrtx-report-validate feature=explain reason=explain ID order mismatch expected={d} actual={d}\n", .{ expected_index, explain.id.index });
            return ValidationError.InvalidTraceReport;
        }
        try validateExplainSubject(explain.subject, report, diagnostics);
        try validateCostRefs("explain cost", explain.cost_ledger_ids, report.cost_ledger.len, diagnostics);
        try validateProfileRefs("explain profile", explain.profile_event_ids, report.profile_events.len, diagnostics);
    }
}

/// This writer gives tests and humans one deterministic view of the trace while
/// the richer JSON/binary schema is still intentionally deferred.
pub fn writeTraceReportSummary(report: TraceReport, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    try writer.writeAll("sources\n");
    for (report.sources) |source| {
        try writer.print("  source.{d} {s} {s}\n", .{ source.id.index, @tagName(source.frontend), source.op_name });
    }

    if (report.target) |target| {
        try writeTargetSummary(target, writer);
    } else {
        try writer.writeAll("target: unknown\n");
    }

    try writer.writeAll("graph values\n");
    for (report.graph_values) |value| {
        try writer.print("  graph.value.{d} role={s} dtype={s}\n", .{ value.id.index, @tagName(value.role), @tagName(value.ty.element_type) });
    }

    try writer.writeAll("graph instructions\n");
    for (report.graph_instructions) |instruction| {
        try writer.print("  graph.instruction.{d} kind={s}\n", .{ instruction.id.index, @tagName(instruction.kind) });
    }

    try writer.writeAll("mlir pass records\n");
    for (report.mlir_pass_records) |record| {
        try writer.print("  pass.{d} name={s} status={s} shardy_preserved={}\n", .{ record.index, record.pass_name, @tagName(record.status), record.preserves_shardy_metadata });
    }

    try writer.writeAll("graph rewrite records\n");
    for (report.graph_rewrite_records) |record| {
        try writer.print(
            "  rewrite.{d} pass={s} decision={s} input_instruction={d} output_instruction=",
            .{ record.index, record.pass_name, @tagName(record.decision), record.input_instruction_id.index },
        );
        if (record.output_instruction_id) |instruction_id| {
            try writer.print("{d}", .{instruction_id.index});
        } else {
            try writer.writeAll("none");
        }
        try writer.writeAll(" replaced_value=");
        if (record.replaced_value_id) |value_id| {
            try writer.print("{d}", .{value_id.index});
        } else {
            try writer.writeAll("none");
        }
        try writer.writeAll(" replacement_value=");
        if (record.replacement_value_id) |value_id| {
            try writer.print("{d}", .{value_id.index});
        } else {
            try writer.writeAll("none");
        }
        try writer.print(" reason={s}\n", .{record.reason});
    }

    try writer.writeAll("fusion groups\n");
    for (report.fusion_groups) |group| {
        try writer.print("  fusion.{d} decision={s} kind={s} instructions=", .{ group.index, @tagName(group.decision), group.kind });
        try writeGraphInstructionIdList(writer, group.graph_instruction_ids);
        try writer.print(
            " bytes_saved={d} pressure=split_kernels:{d},fused_kernels:{d},split_peak:{d},fused_live:{d},additional_live:{d},global_saved:{d}\n",
            .{
                group.bytes_saved,
                group.pressure_delta.split_kernel_count,
                group.pressure_delta.fused_kernel_count,
                group.pressure_delta.split_peak_live_bytes,
                group.pressure_delta.fused_live_bytes,
                group.pressure_delta.additional_live_bytes,
                group.pressure_delta.global_bytes_saved,
            },
        );
    }

    try writer.writeAll("placement records\n");
    for (report.placement_records) |record| {
        try writer.print("  placement.{d} instruction={d} outputs=", .{ record.index, record.graph_instruction_id.index });
        try writeGraphValueIdList(writer, record.output_value_ids);
        try writer.print(" layout={s} tile=", .{@tagName(record.layout)});
        try writeI64Shape(writer, record.logical_tile_shape);
        try writer.print(" result_memory={d} tile_memory=", .{record.result_memory_space_id});
        if (record.tile_memory_space_id) |memory_space_id| {
            try writer.print("{d}", .{memory_space_id});
        } else {
            try writer.writeAll("none");
        }
        try writer.writeAll("\n");
    }

    try writer.writeAll("collective records\n");
    for (report.collective_plan_records) |record| {
        try writer.print(
            "  collective.{d} decision={s} algorithm={s} checked={d} lowered={d} unsupported={d} estimated_bytes={d} estimated_latency_ns=",
            .{
                record.index,
                @tagName(record.decision),
                @tagName(record.algorithm),
                record.checked_graph_instruction_count,
                record.lowered_collective_count,
                record.unsupported_collective_count,
                record.estimated_bytes,
            },
        );
        if (record.estimated_latency_ns) |latency_ns| {
            try writer.print("{d}", .{latency_ns});
        } else {
            try writer.writeAll("unknown");
        }
        try writer.writeAll("\n");
    }

    try writer.writeAll("cost ledger\n");
    for (report.cost_ledger) |entry| {
        try writer.print("  cost.{d} class={s} formula={s}\n", .{ entry.id.index, @tagName(entry.op_class), entry.formula });
    }

    try writer.writeAll("lowering records\n");
    for (report.lowering_records) |record| {
        try writer.print("  lowering.{d} decision={s}\n", .{ record.id.index, @tagName(record.decision) });
    }

    try writer.writeAll("memory traffic records\n");
    for (report.memory_traffic_records) |record| {
        try writer.print(
            "  traffic.{d} lowering={d} memory={d} kind={s} instructions=",
            .{ record.id.index, record.lowering_record_id.index, record.memory_space_id, @tagName(record.kind) },
        );
        try writeGraphInstructionIdList(writer, record.graph_instruction_ids);
        try writer.writeAll(" costs=");
        try writeCostLedgerIdList(writer, record.cost_ledger_ids);
        try writer.print(" bytes_read={d} bytes_written={d}\n", .{ record.bytes_read, record.bytes_written });
    }

    try writer.writeAll("schedule overlap records\n");
    for (report.schedule_overlap_records) |record| {
        try writer.print(
            "  overlap.{d} decision={s} kind={s} first_command={d} second_command={d} dependency={s} first_stream={d} second_stream={d} reason={s}\n",
            .{
                record.id.index,
                @tagName(record.decision),
                @tagName(record.kind),
                record.first_command_id.index,
                record.second_command_id.index,
                @tagName(record.dependency_kind),
                record.first_stream.index,
                record.second_stream.index,
                record.reason,
            },
        );
    }

    try writer.writeAll("schedule commands\n");
    for (report.schedule_commands) |command| {
        try writer.print("  command.{d} kind={s}\n", .{ command.id.index, @tagName(command.kind) });
    }

    try writer.writeAll("kernel codegen records\n");
    for (report.kernel_codegen_records) |record| {
        try writer.print(
            "  codegen.{d} lowering={d} command={d} backend={s} kind={s} op={s} shape=ops:{d},inputs:{d},outputs:{d},intermediates:{d} tile=",
            .{
                record.id.index,
                record.lowering_record_id.index,
                record.command_id.index,
                @tagName(record.backend_kind),
                @tagName(record.kind),
                record.operation,
                record.shape.operation_count,
                record.shape.external_input_count,
                record.shape.external_output_count,
                record.shape.intermediate_value_count,
            },
        );
        try writeI64Shape(writer, record.logical_tile_shape);
        try writer.print(
            " result_memory={d} tile_memory=",
            .{record.result_memory_space_id},
        );
        if (record.tile_memory_space_id) |memory_space_id| {
            try writer.print("{d}", .{memory_space_id});
        } else {
            try writer.writeAll("none");
        }
        try writer.print(
            " pressure=global:{d}/{d},local:{d}/{d} external_inputs=",
            .{
                record.memory_pressure.global_bytes_read,
                record.memory_pressure.global_bytes_written,
                record.memory_pressure.local_bytes_read,
                record.memory_pressure.local_bytes_written,
            },
        );
        try writeGraphValueIdList(writer, record.external_input_ids);
        try writer.writeAll(" external_outputs=");
        try writeGraphValueIdList(writer, record.external_output_ids);
        try writer.writeAll(" intermediates=");
        try writeGraphValueIdList(writer, record.intermediate_value_ids);
        try writer.writeAll(" instructions=");
        try writeGraphInstructionIdList(writer, record.graph_instruction_ids);
        try writer.writeAll(" costs=");
        try writeCostLedgerIdList(writer, record.cost_ledger_ids);
        try writer.writeAll(" traffic=");
        try writeMemoryTrafficIdList(writer, record.memory_traffic_ids);
        try writer.writeAll(" unit=");
        if (record.expected_unit_id) |unit_id| {
            try writer.print("{d}", .{unit_id});
        } else {
            try writer.writeAll("mixed");
        }
        try writer.print(" reason={s}\n", .{record.reason});
    }

    try writer.writeAll("backend bindings\n");
    for (report.backend_bindings) |binding| {
        try writer.print("  binding.{d} backend={s} op={s}\n", .{ binding.id.index, @tagName(binding.backend_kind), binding.backend_operation });
    }

    try writer.writeAll("profile events\n");
    for (report.profile_events) |event| {
        try writer.print("  profile.{d} kind={s} command=", .{ event.id.index, @tagName(event.kind) });
        if (event.command_id) |command_id| {
            try writer.print("{d}", .{command_id.index});
        } else {
            try writer.writeAll("none");
        }
        try writer.writeAll(" instructions=");
        try writeGraphInstructionIdList(writer, event.graph_instruction_ids);
        try writer.print(
            " bytes={d} logical_ops={d} status={s} forced_sync={} start_ns=<redacted> duration_ns=<redacted>\n",
            .{ event.bytes, event.logical_ops, @tagName(event.status), event.forced_synchronization },
        );
    }

    try writer.writeAll("explain records\n");
    for (report.explain_records) |explain| {
        try writer.print("  explain.{d} pass={s} subject=", .{ explain.id.index, explain.pass_name });
        try writeExplainSubject(writer, explain.subject);
        try writer.print(" decision={s} profiles=", .{explain.decision});
        try writeProfileEventIdList(writer, explain.profile_event_ids);
        try writer.writeAll("\n");
    }
}

fn hasMemorySpace(memory_spaces: []const target_pkg.TargetMemorySpace, id: u32) bool {
    return target_pkg.hasMemorySpace(memory_spaces, id);
}

fn hasExecutionUnit(execution_units: []const target_pkg.ExecutionUnit, id: u32) bool {
    return target_pkg.hasExecutionUnit(execution_units, id);
}

fn writeOptionalU64(writer: *std.Io.Writer, value: ?u64) std.Io.Writer.Error!void {
    if (value) |known| {
        try writer.print("{d}", .{known});
    } else {
        try writer.writeAll("unknown");
    }
}

fn writeOptionalF64(writer: *std.Io.Writer, value: ?f64) std.Io.Writer.Error!void {
    if (value) |known| {
        try writer.print("{d}", .{known});
    } else {
        try writer.writeAll("unknown");
    }
}

const TraceGraphFacts = struct {
    fn validateTensorType(tensor: TensorType, diagnostics: *std.Io.Writer) !void {
        compiler_facts.TensorFacts.validate(tensor, diagnostics) catch |err| switch (err) {
            error.InvalidTensorType => return ValidationError.InvalidTensorType,
            else => return err,
        };
    }

    fn payloadMatchesKind(kind: GraphInstructionKind, payload: GraphPayload) bool {
        return compiler_facts.GraphPayloadFacts.matchesKind(kind, payload);
    }
};

fn validateCollectivePayload(id: GraphInstructionId, payload: CollectiveSpec, diagnostics: *std.Io.Writer) !void {
    const expected_participants_u32 = std.math.mul(u32, payload.replica_group_count, payload.replica_group_size) catch {
        try diagnostics.print("pass=pjrtx-report-validate feature=collective reason=replica group shape overflows instruction={d}\n", .{id.index});
        return ValidationError.InvalidTraceReport;
    };
    const expected_participants: usize = std.math.cast(usize, expected_participants_u32) orelse {
        try diagnostics.print("pass=pjrtx-report-validate feature=collective reason=replica group shape does not fit host instruction={d}\n", .{id.index});
        return ValidationError.InvalidTraceReport;
    };
    if (payload.replica_group_count == 0 or payload.replica_group_size == 0 or payload.replica_groups.len != expected_participants) {
        try diagnostics.print("pass=pjrtx-report-validate feature=collective reason=invalid replica group shape instruction={d}\n", .{id.index});
        return ValidationError.InvalidTraceReport;
    }
    if ((payload.channel_id == null) != (payload.channel_type == null)) {
        try diagnostics.print("pass=pjrtx-report-validate feature=collective reason=channel handle is incomplete instruction={d}\n", .{id.index});
        return ValidationError.InvalidTraceReport;
    }
    if (payload.uses_token) {
        try diagnostics.print("pass=pjrtx-report-validate feature=collective reason=tokenized collective cannot validate in V0 instruction={d}\n", .{id.index});
        return ValidationError.InvalidTraceReport;
    }
}

fn validateGraphValueRefs(label: []const u8, ids: []const GraphValueId, len: usize, diagnostics: *std.Io.Writer) !void {
    for (ids) |id| {
        if (!idInBounds(id.index, len)) {
            try diagnostics.print("pass=pjrtx-report-validate feature=graph-value-ref reason={s} references unknown value value={d}\n", .{ label, id.index });
            return ValidationError.InvalidTraceReport;
        }
    }
}

fn validateGraphInstructionRefs(label: []const u8, ids: []const GraphInstructionId, len: usize, diagnostics: *std.Io.Writer) !void {
    for (ids) |id| {
        if (!idInBounds(id.index, len)) {
            try diagnostics.print("pass=pjrtx-report-validate feature=graph-instruction-ref reason={s} references unknown instruction instruction={d}\n", .{ label, id.index });
            return ValidationError.InvalidTraceReport;
        }
    }
}

fn validateCostRefs(label: []const u8, ids: []const CostLedgerId, len: usize, diagnostics: *std.Io.Writer) !void {
    for (ids) |id| {
        if (!idInBounds(id.index, len)) {
            try diagnostics.print("pass=pjrtx-report-validate feature=cost-ref reason={s} references unknown cost cost={d}\n", .{ label, id.index });
            return ValidationError.InvalidTraceReport;
        }
    }
}

fn validateLoweringRefs(label: []const u8, ids: []const LoweringRecordId, len: usize, diagnostics: *std.Io.Writer) !void {
    for (ids) |id| {
        if (!idInBounds(id.index, len)) {
            try diagnostics.print("pass=pjrtx-report-validate feature=lowering-ref reason={s} references unknown lowering lowering={d}\n", .{ label, id.index });
            return ValidationError.InvalidTraceReport;
        }
    }
}

fn validateProfileRefs(label: []const u8, ids: []const ProfileEventId, len: usize, diagnostics: *std.Io.Writer) !void {
    for (ids) |id| {
        if (!idInBounds(id.index, len)) {
            try diagnostics.print("pass=pjrtx-report-validate feature=profile-ref reason={s} references unknown profile event={d}\n", .{ label, id.index });
            return ValidationError.InvalidTraceReport;
        }
    }
}

fn validateExplainSubject(subject: ExplainSubject, report: TraceReport, diagnostics: *std.Io.Writer) !void {
    switch (subject) {
        .graph_instruction => |id| if (!idInBounds(id.index, report.graph_instructions.len)) {
            try diagnostics.print("pass=pjrtx-report-validate feature=explain reason=unknown graph instruction subject={d}\n", .{id.index});
            return ValidationError.InvalidTraceReport;
        },
        .lowering_record => |id| if (!idInBounds(id.index, report.lowering_records.len)) {
            try diagnostics.print("pass=pjrtx-report-validate feature=explain reason=unknown lowering subject={d}\n", .{id.index});
            return ValidationError.InvalidTraceReport;
        },
        .schedule_command => |id| if (!idInBounds(id.index, report.schedule_commands.len)) {
            try diagnostics.print("pass=pjrtx-report-validate feature=explain reason=unknown command subject={d}\n", .{id.index});
            return ValidationError.InvalidTraceReport;
        },
        .backend_binding => |id| if (!idInBounds(id.index, report.backend_bindings.len)) {
            try diagnostics.print("pass=pjrtx-report-validate feature=explain reason=unknown backend binding subject={d}\n", .{id.index});
            return ValidationError.InvalidTraceReport;
        },
    }
}

fn writeGraphInstructionIdList(writer: *std.Io.Writer, ids: []const GraphInstructionId) std.Io.Writer.Error!void {
    for (ids, 0..) |id, index| {
        if (index != 0) try writer.writeAll(",");
        try writer.print("{d}", .{id.index});
    }
}

fn writeGraphValueIdList(writer: *std.Io.Writer, ids: []const GraphValueId) std.Io.Writer.Error!void {
    for (ids, 0..) |id, index| {
        if (index != 0) try writer.writeAll(",");
        try writer.print("{d}", .{id.index});
    }
}

fn writeProfileEventIdList(writer: *std.Io.Writer, ids: []const ProfileEventId) std.Io.Writer.Error!void {
    for (ids, 0..) |id, index| {
        if (index != 0) try writer.writeAll(",");
        try writer.print("{d}", .{id.index});
    }
}

fn writeCostLedgerIdList(writer: *std.Io.Writer, ids: []const CostLedgerId) std.Io.Writer.Error!void {
    for (ids, 0..) |id, index| {
        if (index != 0) try writer.writeAll(",");
        try writer.print("{d}", .{id.index});
    }
}

fn writeMemoryTrafficIdList(writer: *std.Io.Writer, ids: []const MemoryTrafficId) std.Io.Writer.Error!void {
    for (ids, 0..) |id, index| {
        if (index != 0) try writer.writeAll(",");
        try writer.print("{d}", .{id.index});
    }
}

fn writeExplainSubject(writer: *std.Io.Writer, subject: ExplainSubject) std.Io.Writer.Error!void {
    switch (subject) {
        .graph_instruction => |id| try writer.print("graph_instruction.{d}", .{id.index}),
        .lowering_record => |id| try writer.print("lowering.{d}", .{id.index}),
        .schedule_command => |id| try writer.print("command.{d}", .{id.index}),
        .backend_binding => |id| try writer.print("binding.{d}", .{id.index}),
    }
}

fn writeI64Shape(writer: *std.Io.Writer, dims: []const i64) std.Io.Writer.Error!void {
    for (dims, 0..) |dim, index| {
        if (index != 0) try writer.writeAll("x");
        try writer.print("{d}", .{dim});
    }
}

test "ID bounds checks use explicit report index" {
    try std.testing.expect(idInBounds(0, 1));
    try std.testing.expect(!idInBounds(1, 1));
}

test "ID writer uses stable report format" {
    var output = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer output.deinit();

    try writeId(&output.writer, "graph.value", 7);
    try std.testing.expectEqualStrings("graph.value.7", output.writer.buffered());
}

test "trace report validates joined graph lowering schedule backend profile and explain records" {
    const source: SourceRef = .{ .id = .{ .index = 0 }, .frontend = .stablehlo, .op_name = "stablehlo.tanh", .source_index = 0, .location = "" };
    const sources = [_]SourceRef{source};
    const dims = [_]i64{4};
    const values = [_]GraphValue{
        .{ .id = .{ .index = 0 }, .ty = .{ .element_type = .f32, .dims = &dims, .layout = .dense_row_major }, .role = .parameter, .source = source },
        .{ .id = .{ .index = 1 }, .ty = .{ .element_type = .f32, .dims = &dims, .layout = .dense_row_major }, .role = .instruction_result, .source = source },
    };
    const instruction_inputs = [_]GraphValueId{.{ .index = 0 }};
    const instruction_outputs = [_]GraphValueId{.{ .index = 1 }};
    const instructions = [_]GraphInstruction{
        .{
            .id = .{ .index = 0 },
            .kind = .elementwise_unary,
            .inputs = &instruction_inputs,
            .outputs = &instruction_outputs,
            .payload = .{ .elementwise_unary = .{ .op = .tanh } },
            .source = source,
        },
    };
    const cost_instructions = [_]GraphInstructionId{.{ .index = 0 }};
    const costs = [_]CostLedgerEntry{
        .{
            .id = .{ .index = 0 },
            .source = source,
            .graph_instruction_ids = &cost_instructions,
            .op_class = .transcendental,
            .dtype = .f32,
            .accumulation_dtype = null,
            .logical_ops = 4,
            .bytes_read = 16,
            .bytes_written = 16,
            .expected_unit_id = null,
            .formula = "numel",
            .approximation = "",
        },
    };
    const lowering_costs = [_]CostLedgerId{.{ .index = 0 }};
    const lowering_instructions = [_]GraphInstructionId{.{ .index = 0 }};
    const lowerings = [_]LoweringRecord{
        .{
            .id = .{ .index = 0 },
            .graph_instruction_ids = &lowering_instructions,
            .decision = .backend_kernel_graph,
            .reason = "metal_v0",
            .rejected_alternatives = &.{},
            .cost_ledger_ids = &lowering_costs,
        },
    };
    const command_inputs = [_]GraphValueId{.{ .index = 0 }};
    const command_outputs = [_]GraphValueId{.{ .index = 1 }};
    const command_lowerings = [_]LoweringRecordId{.{ .index = 0 }};
    const command_costs = [_]CostLedgerId{.{ .index = 0 }};
    const commands = [_]ScheduleCommand{
        .{
            .id = .{ .index = 0 },
            .kind = .backend_execute,
            .stream = .{ .index = 0 },
            .inputs = &command_inputs,
            .outputs = &command_outputs,
            .dependencies = &.{},
            .lowering_record_ids = &command_lowerings,
            .cost_ledger_ids = &command_costs,
        },
    };
    const binding_instructions = [_]GraphInstructionId{.{ .index = 0 }};
    const binding_costs = [_]CostLedgerId{.{ .index = 0 }};
    const bindings = [_]BackendBinding{
        .{
            .id = .{ .index = 0 },
            .command_id = .{ .index = 0 },
            .backend_kind = .metal_v0,
            .backend_operation = "metal_mls_graph_execute",
            .graph_instruction_ids = &binding_instructions,
            .expected_unit_id = null,
            .cost_ledger_ids = &binding_costs,
        },
    };
    const profile_instructions = [_]GraphInstructionId{.{ .index = 0 }};
    const profile_events = [_]ProfileEvent{
        .{
            .id = .{ .index = 0 },
            .command_id = .{ .index = 0 },
            .graph_instruction_ids = &profile_instructions,
            .kind = .backend_execute,
            .start_ns = 10,
            .duration_ns = 20,
            .bytes = 32,
            .logical_ops = 4,
            .status = .ok,
            .forced_synchronization = false,
        },
    };
    const explain_sources = [_]SourceRef{source};
    const explain_costs = [_]CostLedgerId{.{ .index = 0 }};
    const explain_profiles = [_]ProfileEventId{.{ .index = 0 }};
    const explains = [_]ExplainRecord{
        .{
            .id = .{ .index = 0 },
            .pass_name = "lowering",
            .subject = .{ .backend_binding = .{ .index = 0 } },
            .decision = "kernel_graph",
            .reason = "bootstrap backend",
            .source_refs = &explain_sources,
            .cost_ledger_ids = &explain_costs,
            .profile_event_ids = &explain_profiles,
        },
    };
    const report: TraceReport = .{
        .sources = &sources,
        .graph_values = &values,
        .graph_instructions = &instructions,
        .cost_ledger = &costs,
        .lowering_records = &lowerings,
        .schedule_commands = &commands,
        .backend_bindings = &bindings,
        .profile_events = &profile_events,
        .explain_records = &explains,
    };
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    try validateTraceReport(report, &diagnostics.writer);
    try std.testing.expectEqualStrings("", diagnostics.writer.buffered());
}

test "trace report rejects no-collectives records that carry an algorithm" {
    const collectives = [_]CollectivePlanRecord{
        .{
            .index = 0,
            .decision = .no_collectives,
            .algorithm = .ring,
            .checked_graph_instruction_count = 0,
            .lowered_collective_count = 0,
            .unsupported_collective_count = 0,
            .estimated_bytes = 0,
            .estimated_latency_ns = null,
            .reason = "invalid test record",
        },
    };
    const report: TraceReport = .{
        .sources = &.{},
        .graph_values = &.{},
        .graph_instructions = &.{},
        .collective_plan_records = &collectives,
        .cost_ledger = &.{},
        .lowering_records = &.{},
        .schedule_commands = &.{},
        .backend_bindings = &.{},
        .profile_events = &.{},
        .explain_records = &.{},
    };
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    try std.testing.expectError(ValidationError.InvalidTraceReport, validateTraceReport(report, &diagnostics.writer));
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "no-collectives record carries collective work") != null);
}

test "trace report rejects non-executable collective plans" {
    const collectives = [_]CollectivePlanRecord{
        .{
            .index = 0,
            .decision = .unsupported,
            .algorithm = .none,
            .checked_graph_instruction_count = 0,
            .lowered_collective_count = 0,
            .unsupported_collective_count = 1,
            .estimated_bytes = 16,
            .estimated_latency_ns = null,
            .reason = "invalid executable report",
        },
    };
    const report: TraceReport = .{
        .sources = &.{},
        .graph_values = &.{},
        .graph_instructions = &.{},
        .collective_plan_records = &collectives,
        .cost_ledger = &.{},
        .lowering_records = &.{},
        .schedule_commands = &.{},
        .backend_bindings = &.{},
        .profile_events = &.{},
        .explain_records = &.{},
    };
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    try std.testing.expectError(ValidationError.InvalidTraceReport, validateTraceReport(report, &diagnostics.writer));
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "non-executable collective plan") != null);
}

test "trace report rejects backend commands without lowering provenance" {
    const dims = [_]i64{4};
    const values = [_]GraphValue{
        .{ .id = .{ .index = 0 }, .ty = .{ .element_type = .f32, .dims = &dims, .layout = .dense_row_major }, .role = .parameter, .source = null },
    };
    const commands = [_]ScheduleCommand{
        .{
            .id = .{ .index = 0 },
            .kind = .backend_execute,
            .stream = .{ .index = 0 },
            .inputs = &.{.{ .index = 0 }},
            .outputs = &.{},
            .dependencies = &.{},
            .lowering_record_ids = &.{},
            .cost_ledger_ids = &.{},
        },
    };
    const report: TraceReport = .{
        .sources = &.{},
        .graph_values = &values,
        .graph_instructions = &.{},
        .cost_ledger = &.{},
        .lowering_records = &.{},
        .schedule_commands = &commands,
        .backend_bindings = &.{},
        .profile_events = &.{},
        .explain_records = &.{},
    };
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    try std.testing.expectError(ValidationError.InvalidTraceReport, validateTraceReport(report, &diagnostics.writer));
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "backend command missing lowering provenance") != null);
}

test "trace report summary keeps report order and redacts profile duration" {
    const profile_events = [_]ProfileEvent{
        .{
            .id = .{ .index = 0 },
            .command_id = null,
            .graph_instruction_ids = &.{},
            .kind = .compile_pass,
            .start_ns = 10,
            .duration_ns = 999,
            .bytes = 0,
            .logical_ops = 0,
            .status = .ok,
            .forced_synchronization = false,
        },
    };
    const report: TraceReport = .{
        .sources = &.{},
        .graph_values = &.{},
        .graph_instructions = &.{},
        .cost_ledger = &.{},
        .lowering_records = &.{},
        .schedule_commands = &.{},
        .backend_bindings = &.{},
        .profile_events = &profile_events,
        .explain_records = &.{},
    };
    var output: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();

    try writeTraceReportSummary(report, &output.writer);
    try std.testing.expect(std.mem.indexOf(u8, output.writer.buffered(), "sources\ntarget: unknown\ngraph values") != null);
    try std.testing.expect(std.mem.indexOf(u8, output.writer.buffered(), "profile.0 kind=compile_pass command=none instructions= bytes=0 logical_ops=0 status=ok forced_sync=false start_ns=<redacted> duration_ns=<redacted>") != null);
    try std.testing.expect(std.mem.indexOf(u8, output.writer.buffered(), "explain records") != null);
    try std.testing.expect(std.mem.indexOf(u8, output.writer.buffered(), "duration_ns=<redacted>") != null);
    try std.testing.expect(std.mem.indexOf(u8, output.writer.buffered(), "999") == null);
    try std.testing.expect(std.mem.indexOf(u8, output.writer.buffered(), "10") == null);
}
