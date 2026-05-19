const std = @import("std");
const mlir = @import("c");
const core = @import("pjrtx/core");
const compiler_facts = @import("pjrtx/compiler/facts");
const target_pkg = @import("pjrtx/target");
const fusion_passes = @import("fusion_passes.zig");
const limits = @import("limits.zig");
const lowering_policy = @import("lowering_policy.zig");
const passes = @import("passes.zig");
const placement_collective_passes = @import("placement_collective_passes.zig");
const state_target_passes = @import("state_target_passes.zig");
const types = @import("types.zig");

pub const MlirStateError = types.MlirStateError;
pub const ModuleState = types.ModuleState;
pub const MlirSessionOptions = types.MlirSessionOptions;
pub const LoweringRegionFact = types.LoweringRegionFact;
pub const CostCapabilityFact = types.CostCapabilityFact;
pub const KernelCodegenCapabilityFact = types.KernelCodegenCapabilityFact;
pub const ExecutableContract = types.ExecutableContract;
pub const BackendExecutableCallFact = types.BackendExecutableCallFact;
pub const BackendExecutablePlanFact = types.BackendExecutablePlanFact;
pub const BackendProfileJoinFact = types.BackendProfileJoinFact;
pub const BackendProfileJoinPlanFact = types.BackendProfileJoinPlanFact;
pub const SchedulePlanFact = types.SchedulePlanFact;
pub const BackendTensorDescriptorFact = types.BackendTensorDescriptorFact;
pub const BackendKernelGraphNodeFact = types.BackendKernelGraphNodeFact;
pub const BackendKernelGraphEdgeFact = types.BackendKernelGraphEdgeFact;
pub const BackendKernelGraphFact = types.BackendKernelGraphFact;
pub const RuntimeAllocationFact = types.RuntimeAllocationFact;
pub const RuntimeBufferUseFact = types.RuntimeBufferUseFact;
pub const RuntimeAllocationPlanFact = types.RuntimeAllocationPlanFact;
pub const RuntimeStreamStepFact = types.RuntimeStreamStepFact;
pub const RuntimeStreamPlanFact = types.RuntimeStreamPlanFact;
pub const RuntimeProfileEventFact = types.RuntimeProfileEventFact;
pub const RuntimeProfileFact = types.RuntimeProfileFact;
pub const RuntimeProfileJoinFact = types.RuntimeProfileJoinFact;
pub const RuntimeProfileJoinPlanFact = types.RuntimeProfileJoinPlanFact;
pub const ExternalPassProbeRecord = types.ExternalPassProbeRecord;

pub const MlirSession = struct {
    allocator: std.mem.Allocator,
    registry: mlir.MlirDialectRegistry,
    context: mlir.MlirContext,
    module: mlir.MlirModule,
    pass_manager: mlir.MlirPassManager,

    pub fn initFromStablehloText(
        allocator: std.mem.Allocator,
        program_text: []const u8,
        options: MlirSessionOptions,
        diagnostics: *std.Io.Writer,
    ) !MlirSession {
        var session: MlirSession = .{
            .allocator = allocator,
            .registry = mlir.MlirDialectRegistry{ .ptr = null },
            .context = mlir.MlirContext{ .ptr = null },
            .module = mlir.MlirModule{ .ptr = null },
            .pass_manager = mlir.MlirPassManager{ .ptr = null },
        };
        errdefer session.deinit();

        session.registry = mlir.mlirDialectRegistryCreate();
        if (mlir.mlirDialectRegistryIsNull(session.registry)) {
            try diagnostics.writeAll("pass=mlir_state_init feature=mlir-registry reason=failed to create MLIR dialect registry\n");
            return MlirStateError.InvalidStablehlo;
        }

        insertDialect(session.registry, mlir.mlirGetDialectHandle__func__());
        insertDialect(session.registry, mlir.mlirGetDialectHandle__shape__());
        insertDialect(session.registry, mlir.mlirGetDialectHandle__chlo__());
        insertDialect(session.registry, mlir.mlirGetDialectHandle__sdy__());
        insertDialect(session.registry, mlir.mlirGetDialectHandle__stablehlo__());
        mlir.pjrtxMlirRegisterFuncExtensions(session.registry);

        session.context = mlir.mlirContextCreateWithRegistry(session.registry, false);
        if (mlir.mlirContextIsNull(session.context)) {
            try diagnostics.writeAll("pass=mlir_state_init feature=mlir-context reason=failed to create MLIR context\n");
            return MlirStateError.InvalidStablehlo;
        }

        // PjRTx attributes are intentionally unregistered until the real
        // dialect lands. They are still verified by this module before use.
        mlir.mlirContextSetAllowUnregisteredDialects(session.context, true);
        mlir.mlirContextLoadAllAvailableDialects(session.context);
        try loadDialect(session.context, mlir.mlirGetDialectHandle__func__(), diagnostics);
        try loadDialect(session.context, mlir.mlirGetDialectHandle__shape__(), diagnostics);
        try loadDialect(session.context, mlir.mlirGetDialectHandle__chlo__(), diagnostics);
        try loadDialect(session.context, mlir.mlirGetDialectHandle__sdy__(), diagnostics);
        try loadDialect(session.context, mlir.mlirGetDialectHandle__stablehlo__(), diagnostics);

        session.module = mlir.mlirModuleCreateParse(session.context, mlirStringRef(program_text));
        if (mlir.mlirModuleIsNull(session.module)) {
            try diagnostics.writeAll("pass=mlir_state_init feature=stablehlo reason=MLIR parser rejected module\n");
            return MlirStateError.InvalidStablehlo;
        }

        const module_op = mlir.mlirModuleGetOperation(session.module);
        if (!mlir.mlirOperationVerify(module_op)) {
            try diagnostics.writeAll("pass=mlir_state_init feature=stablehlo reason=MLIR verifier rejected module\n");
            return MlirStateError.InvalidStablehlo;
        }

        setStringAttr(session.context, module_op, "pjrtx.program_name", options.program_name);
        setModuleStateUnchecked(&session, .imported);
        return session;
    }

    pub fn deinit(self: *MlirSession) void {
        if (!mlir.mlirPassManagerIsNull(self.pass_manager)) {
            mlir.mlirPassManagerDestroy(self.pass_manager);
        }
        if (!mlir.mlirModuleIsNull(self.module)) {
            mlir.mlirModuleDestroy(self.module);
        }
        if (!mlir.mlirContextIsNull(self.context)) {
            mlir.mlirContextDestroy(self.context);
        }
        if (!mlir.mlirDialectRegistryIsNull(self.registry)) {
            mlir.mlirDialectRegistryDestroy(self.registry);
        }
        self.* = undefined;
    }

    pub fn moduleOperation(self: *const MlirSession) mlir.MlirOperation {
        return mlir.mlirModuleGetOperation(self.module);
    }
};

pub fn moduleState(session: *const MlirSession) ?ModuleState {
    const attr = getAttr(session.moduleOperation(), "pjrtx.state");
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAString(attr)) return null;
    return parseModuleState(mlirStringSlice(mlir.mlirStringAttrGetValue(attr)));
}

pub fn requireModuleState(
    session: *const MlirSession,
    expected: ModuleState,
    diagnostics: *std.Io.Writer,
) !void {
    const current = moduleState(session) orelse {
        try diagnostics.print(
            "pass=mlir_state_verify feature=state reason=missing module state expected={s}\n",
            .{expected.text()},
        );
        return MlirStateError.InvalidStateTransition;
    };
    if (current != expected) {
        try diagnostics.print(
            "pass=mlir_state_verify feature=state reason=unexpected module state expected={s} actual={s}\n",
            .{ expected.text(), current.text() },
        );
        return MlirStateError.InvalidStateTransition;
    }
}

fn requireRuntimeAllocationInputState(session: *const MlirSession, diagnostics: *std.Io.Writer) !void {
    const state = moduleState(session) orelse {
        try diagnostics.writeAll("pass=runtime_allocation_state feature=state reason=missing module state\n");
        return MlirStateError.InvalidStateTransition;
    };
    if (state != .backend_executable_planned and state != .backend_kernel_graph_planned) {
        try diagnostics.print(
            "pass=runtime_allocation_state feature=state reason=expected backend executable or kernel graph state actual={s}\n",
            .{state.text()},
        );
        return MlirStateError.InvalidStateTransition;
    }
}

fn requireScheduleExtractState(session: *const MlirSession, diagnostics: *std.Io.Writer) !void {
    const state = moduleState(session) orelse {
        try diagnostics.writeAll("pass=schedule_extract feature=state reason=missing module state\n");
        return MlirStateError.InvalidStateTransition;
    };
    if (state != .scheduled and state != .backend_bound and state != .executable_ready and state != .backend_executable_planned and state != .backend_kernel_graph_planned and state != .runtime_allocation_planned and state != .runtime_stream_planned and state != .runtime_profiled and state != .runtime_profile_joined and state != .backend_profile_joined) {
        try diagnostics.print(
            "pass=schedule_extract feature=state reason=expected scheduled or later state actual={s}\n",
            .{state.text()},
        );
        return MlirStateError.InvalidStateTransition;
    }
}

fn requireTargetExtractState(session: *const MlirSession, diagnostics: *std.Io.Writer) !void {
    const state = moduleState(session) orelse {
        try diagnostics.writeAll("pass=target_extract feature=state reason=missing module state\n");
        return MlirStateError.InvalidStateTransition;
    };
    if (state != .target_legal and state != .fusion_planned and state != .placement_planned and state != .collectives_planned and state != .lowering_planned and state != .performance_modeled and state != .codegen_planned and state != .scheduled and state != .backend_bound and state != .executable_ready and state != .backend_executable_planned and state != .backend_kernel_graph_planned and state != .runtime_allocation_planned and state != .runtime_stream_planned and state != .runtime_profiled and state != .runtime_profile_joined and state != .backend_profile_joined) {
        try diagnostics.print(
            "pass=target_extract feature=state reason=expected target legal or later state actual={s}\n",
            .{state.text()},
        );
        return MlirStateError.InvalidStateTransition;
    }
}

fn requireLoweringExtractState(session: *const MlirSession, diagnostics: *std.Io.Writer) !void {
    const state = moduleState(session) orelse {
        try diagnostics.writeAll("pass=lowering_extract feature=state reason=missing module state\n");
        return MlirStateError.InvalidStateTransition;
    };
    if (state != .lowering_planned and state != .performance_modeled and state != .codegen_planned and state != .scheduled and state != .backend_bound and state != .executable_ready and state != .backend_executable_planned and state != .backend_kernel_graph_planned and state != .runtime_allocation_planned and state != .runtime_stream_planned and state != .runtime_profiled and state != .runtime_profile_joined and state != .backend_profile_joined) {
        try diagnostics.print(
            "pass=lowering_extract feature=state reason=expected lowering planned or later state actual={s}\n",
            .{state.text()},
        );
        return MlirStateError.InvalidStateTransition;
    }
}

fn requirePerformanceExtractState(session: *const MlirSession, diagnostics: *std.Io.Writer) !void {
    const state = moduleState(session) orelse {
        try diagnostics.writeAll("pass=performance_extract feature=state reason=missing module state\n");
        return MlirStateError.InvalidStateTransition;
    };
    if (state != .performance_modeled and state != .codegen_planned and state != .scheduled and state != .backend_bound and state != .executable_ready and state != .backend_executable_planned and state != .backend_kernel_graph_planned and state != .runtime_allocation_planned and state != .runtime_stream_planned and state != .runtime_profiled and state != .runtime_profile_joined and state != .backend_profile_joined) {
        try diagnostics.print(
            "pass=performance_extract feature=state reason=expected performance modeled or later state actual={s}\n",
            .{state.text()},
        );
        return MlirStateError.InvalidStateTransition;
    }
}

fn requireCostLedgerExtractState(session: *const MlirSession, diagnostics: *std.Io.Writer) !void {
    const state = moduleState(session) orelse {
        try diagnostics.writeAll("pass=cost_ledger_extract feature=state reason=missing module state\n");
        return MlirStateError.InvalidStateTransition;
    };
    if (state != .lowering_planned and state != .performance_modeled and state != .codegen_planned and state != .scheduled and state != .backend_bound and state != .executable_ready and state != .backend_executable_planned and state != .backend_kernel_graph_planned and state != .runtime_allocation_planned and state != .runtime_stream_planned and state != .runtime_profiled and state != .runtime_profile_joined and state != .backend_profile_joined) {
        try diagnostics.print(
            "pass=cost_ledger_extract feature=state reason=expected lowering planned or later state actual={s}\n",
            .{state.text()},
        );
        return MlirStateError.InvalidStateTransition;
    }
}

fn requireBackendExecutableExtractState(session: *const MlirSession, diagnostics: *std.Io.Writer) !void {
    const state = moduleState(session) orelse {
        try diagnostics.writeAll("pass=backend_executable_extract feature=state reason=missing module state\n");
        return MlirStateError.InvalidStateTransition;
    };
    if (state != .backend_executable_planned and state != .backend_kernel_graph_planned and state != .runtime_allocation_planned and state != .runtime_stream_planned and state != .runtime_profiled and state != .runtime_profile_joined and state != .backend_profile_joined) {
        try diagnostics.print(
            "pass=backend_executable_extract feature=state reason=expected backend executable or later state actual={s}\n",
            .{state.text()},
        );
        return MlirStateError.InvalidStateTransition;
    }
}

fn requireBackendKernelGraphExtractState(session: *const MlirSession, diagnostics: *std.Io.Writer) !void {
    const state = moduleState(session) orelse {
        try diagnostics.writeAll("pass=backend_kernel_graph_extract feature=state reason=missing module state\n");
        return MlirStateError.InvalidStateTransition;
    };
    if (state != .backend_kernel_graph_planned and state != .runtime_allocation_planned and state != .runtime_stream_planned and state != .runtime_profiled and state != .runtime_profile_joined and state != .backend_profile_joined) {
        try diagnostics.print(
            "pass=backend_kernel_graph_extract feature=state reason=expected backend kernel graph or later state actual={s}\n",
            .{state.text()},
        );
        return MlirStateError.InvalidStateTransition;
    }
}

fn requireRuntimeAllocationExtractState(session: *const MlirSession, diagnostics: *std.Io.Writer) !void {
    const state = moduleState(session) orelse {
        try diagnostics.writeAll("pass=runtime_allocation_extract feature=state reason=missing module state\n");
        return MlirStateError.InvalidStateTransition;
    };
    if (state != .runtime_allocation_planned and state != .runtime_stream_planned and state != .runtime_profiled and state != .runtime_profile_joined and state != .backend_profile_joined) {
        try diagnostics.print(
            "pass=runtime_allocation_extract feature=state reason=expected runtime allocation or later runtime state actual={s}\n",
            .{state.text()},
        );
        return MlirStateError.InvalidStateTransition;
    }
}

fn requireRuntimeStreamExtractState(session: *const MlirSession, diagnostics: *std.Io.Writer) !void {
    const state = moduleState(session) orelse {
        try diagnostics.writeAll("pass=runtime_stream_extract feature=state reason=missing module state\n");
        return MlirStateError.InvalidStateTransition;
    };
    if (state != .runtime_stream_planned and state != .runtime_profiled and state != .runtime_profile_joined and state != .backend_profile_joined) {
        try diagnostics.print(
            "pass=runtime_stream_extract feature=state reason=expected runtime stream or later profile state actual={s}\n",
            .{state.text()},
        );
        return MlirStateError.InvalidStateTransition;
    }
}

fn requireRuntimeProfileExtractState(session: *const MlirSession, diagnostics: *std.Io.Writer) !void {
    const state = moduleState(session) orelse {
        try diagnostics.writeAll("pass=runtime_profile_extract feature=state reason=missing module state\n");
        return MlirStateError.InvalidStateTransition;
    };
    if (state != .runtime_profiled and state != .runtime_profile_joined and state != .backend_profile_joined) {
        try diagnostics.print(
            "pass=runtime_profile_extract feature=state reason=expected runtime profiled or later profile state actual={s}\n",
            .{state.text()},
        );
        return MlirStateError.InvalidStateTransition;
    }
}

fn requireRuntimeProfileJoinExtractState(session: *const MlirSession, diagnostics: *std.Io.Writer) !void {
    const state = moduleState(session) orelse {
        try diagnostics.writeAll("pass=runtime_profile_join_extract feature=state reason=missing module state\n");
        return MlirStateError.InvalidStateTransition;
    };
    if (state != .runtime_profile_joined and state != .backend_profile_joined) {
        try diagnostics.print(
            "pass=runtime_profile_join_extract feature=state reason=expected runtime or backend profile joined state actual={s}\n",
            .{state.text()},
        );
        return MlirStateError.InvalidStateTransition;
    }
}

pub fn attachTarget(
    session: *MlirSession,
    target: target_pkg.TargetDescription,
    replicas: u32,
    partitions: u32,
    diagnostics: *std.Io.Writer,
) !void {
    try requireModuleState(session, .imported, diagnostics);
    target_pkg.validateTargetDescription(target, diagnostics) catch {
        try diagnostics.writeAll("pass=target_attach feature=target reason=target description failed validation\n");
        return MlirStateError.InvalidTargetAttachment;
    };

    const module_op = session.moduleOperation();
    setStringAttr(session.context, module_op, "pjrtx.target.name", target.name);
    setStringAttr(session.context, module_op, "pjrtx.target.kind", @tagName(target.kind));

    const fingerprint = try targetFingerprintText(session.allocator, target);
    defer session.allocator.free(fingerprint);
    setStringAttr(session.context, module_op, "pjrtx.target.fingerprint", fingerprint);
    setIntegerAttr(session.context, module_op, "pjrtx.target.replicas", replicas);
    setIntegerAttr(session.context, module_op, "pjrtx.target.partitions", partitions);
    setTargetSpecAttr(session.context, module_op, target);
    transitionModuleState(session, .target_attached, diagnostics) catch return MlirStateError.InvalidTargetAttachment;
}

pub fn markTargetLegal(session: *MlirSession, diagnostics: *std.Io.Writer) !void {
    try runTargetLegalExternalPass(session, diagnostics);
}

pub fn planFusionFromCandidates(
    session: *MlirSession,
    diagnostics: *std.Io.Writer,
) !void {
    try requireModuleState(session, .target_legal, diagnostics);
    try runFusionPlanExternalPass(session, diagnostics);
}

pub fn extractFusionGroups(
    allocator: std.mem.Allocator,
    session: *const MlirSession,
    diagnostics: *std.Io.Writer,
) ![]compiler_facts.FusionGroup {
    try requireModuleState(session, .fusion_planned, diagnostics);

    const candidates_attr = getAttr(session.moduleOperation(), "pjrtx.fusion.candidates");
    if (mlir.mlirAttributeIsNull(candidates_attr) or !mlir.mlirAttributeIsAArray(candidates_attr)) {
        try diagnostics.writeAll("pass=fusion_extract feature=fusion-candidates reason=missing enriched MLIR fusion candidate attribute\n");
        return MlirStateError.InvalidExtractionState;
    }

    var groups: std.ArrayList(compiler_facts.FusionGroup) = .empty;
    errdefer {
        for (groups.items) |group| {
            allocator.free(group.kind);
            allocator.free(group.graph_instruction_ids);
            allocator.free(group.reason);
        }
        groups.deinit(allocator);
    }

    const count = mlir.mlirArrayAttrGetNumElements(candidates_attr);
    var index: isize = 0;
    while (index < count) : (index += 1) {
        const candidate_attr = mlir.mlirArrayAttrGetElement(candidates_attr, index);
        try groups.append(allocator, try parseFusionCandidateDecisionAttr(allocator, candidate_attr, diagnostics));
    }

    return groups.toOwnedSlice(allocator);
}

pub fn commitPlacementPlan(
    session: *MlirSession,
    records: []const compiler_facts.PlacementRecord,
    diagnostics: *std.Io.Writer,
) !void {
    try requireModuleState(session, .fusion_planned, diagnostics);
    try verifyPlacementRecords(records, diagnostics);
    setPlacementPlanAttr(session.context, session.moduleOperation(), records);
    try runPlacementPlanExternalPass(session, diagnostics);
}

pub fn extractPlacementRecords(
    allocator: std.mem.Allocator,
    session: *const MlirSession,
    diagnostics: *std.Io.Writer,
) ![]compiler_facts.PlacementRecord {
    try requireModuleState(session, .placement_planned, diagnostics);

    const placements_attr = getAttr(session.moduleOperation(), "pjrtx.placement.records");
    if (mlir.mlirAttributeIsNull(placements_attr) or !mlir.mlirAttributeIsAArray(placements_attr)) {
        try diagnostics.writeAll("pass=placement_extract feature=placement-records reason=missing MLIR placement records attribute\n");
        return MlirStateError.InvalidExtractionState;
    }

    var records: std.ArrayList(compiler_facts.PlacementRecord) = .empty;
    errdefer {
        for (records.items) |record| {
            allocator.free(record.output_value_ids);
            allocator.free(record.logical_tile_shape);
            allocator.free(record.reason);
        }
        records.deinit(allocator);
    }

    const count = mlir.mlirArrayAttrGetNumElements(placements_attr);
    var index: isize = 0;
    while (index < count) : (index += 1) {
        const record_attr = mlir.mlirArrayAttrGetElement(placements_attr, index);
        try records.append(allocator, try parsePlacementRecordAttr(allocator, record_attr, diagnostics));
    }

    const owned_records = try records.toOwnedSlice(allocator);
    errdefer deinitExtractedPlacementRecords(allocator, owned_records);
    try verifyPlacementRecords(owned_records, diagnostics);
    return owned_records;
}

pub fn commitCollectivePlan(
    session: *MlirSession,
    records: []const compiler_facts.CollectivePlanRecord,
    diagnostics: *std.Io.Writer,
) !void {
    try requireModuleState(session, .placement_planned, diagnostics);
    try verifyCollectivePlanRecords(records, diagnostics);
    setCollectivePlanAttr(session.context, session.moduleOperation(), records);
    try runCollectivePlanExternalPass(session, diagnostics);
}

pub fn extractCollectivePlanRecords(
    allocator: std.mem.Allocator,
    session: *const MlirSession,
    diagnostics: *std.Io.Writer,
) ![]compiler_facts.CollectivePlanRecord {
    try requireModuleState(session, .collectives_planned, diagnostics);

    const collectives_attr = getAttr(session.moduleOperation(), "pjrtx.collective.records");
    if (mlir.mlirAttributeIsNull(collectives_attr) or !mlir.mlirAttributeIsAArray(collectives_attr)) {
        try diagnostics.writeAll("pass=collective_extract feature=collective-records reason=missing MLIR collective records attribute\n");
        return MlirStateError.InvalidExtractionState;
    }

    var records: std.ArrayList(compiler_facts.CollectivePlanRecord) = .empty;
    errdefer {
        for (records.items) |record| allocator.free(record.reason);
        records.deinit(allocator);
    }

    const count = mlir.mlirArrayAttrGetNumElements(collectives_attr);
    var index: isize = 0;
    while (index < count) : (index += 1) {
        const record_attr = mlir.mlirArrayAttrGetElement(collectives_attr, index);
        try records.append(allocator, try parseCollectivePlanRecordAttr(allocator, record_attr, diagnostics));
    }

    const owned_records = try records.toOwnedSlice(allocator);
    errdefer deinitExtractedCollectivePlanRecords(allocator, owned_records);
    try verifyCollectivePlanRecords(owned_records, diagnostics);
    return owned_records;
}

pub fn commitLoweringPlan(
    session: *MlirSession,
    cost_ledger: []const compiler_facts.CostLedgerEntry,
    diagnostics: *std.Io.Writer,
) !void {
    try requireModuleState(session, .collectives_planned, diagnostics);
    try verifyCostLedgerEntries(cost_ledger, diagnostics);
    setCostLedgerAttr(session.context, session.moduleOperation(), cost_ledger);
    const records = try deriveLoweringRecordsFromMlir(session.allocator, session, diagnostics);
    defer deinitExtractedLoweringRecords(session.allocator, records);
    try verifyLoweringRecords(records, cost_ledger.len, diagnostics);
    const region_facts = try deriveLoweringRegionFactsFromMlir(session.allocator, session, records, diagnostics);
    defer deinitExtractedLoweringRegionFacts(session.allocator, region_facts);
    try verifyLoweringRegionFacts(region_facts, records, diagnostics);
    setLoweringPlanAttr(session.context, session.moduleOperation(), records);
    setLoweringRegionFactsAttr(session.context, session.moduleOperation(), region_facts);
    try runLoweringPlanExternalPass(session, diagnostics);
}

pub fn deriveCostLedgerEntries(
    allocator: std.mem.Allocator,
    values: []const compiler_facts.GraphValue,
    instructions: []const compiler_facts.GraphInstruction,
    capabilities: []const CostCapabilityFact,
    diagnostics: *std.Io.Writer,
) ![]compiler_facts.CostLedgerEntry {
    var entries: std.ArrayList(compiler_facts.CostLedgerEntry) = .empty;
    errdefer {
        for (entries.items) |entry| deinitCostLedgerEntryFields(allocator, entry);
        entries.deinit(allocator);
    }

    for (instructions) |instruction| {
        if (instruction.kind == .return_) continue;
        const capability = costCapabilityForInstruction(capabilities, instruction.id) orelse {
            try diagnostics.print(
                "pass=performance_cost_model feature=backend-capability reason=missing capability fact instruction={d}\n",
                .{instruction.id.index},
            );
            return MlirStateError.InvalidLoweringPlan;
        };

        const cost_id: compiler_facts.CostLedgerId = .{ .index = std.math.cast(u32, entries.items.len) orelse unreachable };
        const instruction_ids = try allocator.alloc(compiler_facts.GraphInstructionId, 1);
        var instruction_ids_owned = true;
        errdefer if (instruction_ids_owned) allocator.free(instruction_ids);
        instruction_ids[0] = instruction.id;

        const output = try costOutputValue(values, instruction, diagnostics);
        const formula = try allocator.dupe(u8, costFormulaForInstruction(instruction));
        var formula_owned = true;
        errdefer if (formula_owned) allocator.free(formula);
        const approximation = try allocator.dupe(u8, costApproximationForInstruction(instruction));
        var approximation_owned = true;
        errdefer if (approximation_owned) allocator.free(approximation);
        const source = try duplicateCostSourceRef(allocator, instruction.source);
        var source_owned = true;
        errdefer if (source_owned) {
            allocator.free(source.op_name);
            allocator.free(source.location);
        };

        try entries.append(allocator, .{
            .id = cost_id,
            .source = source,
            .graph_instruction_ids = instruction_ids,
            .op_class = costClassForInstruction(instruction),
            .dtype = output.ty.element_type,
            .accumulation_dtype = costAccumulationDtypeForInstruction(instruction, output.ty.element_type),
            .logical_ops = try costLogicalOpsForInstruction(values, instruction, diagnostics),
            .bytes_read = try costBytesReadForInstruction(values, instruction, diagnostics),
            .bytes_written = try costBytesWrittenForInstruction(values, instruction, diagnostics),
            .expected_unit_id = capability.expected_unit_id,
            .formula = formula,
            .approximation = approximation,
        });
        instruction_ids_owned = false;
        formula_owned = false;
        approximation_owned = false;
        source_owned = false;
    }

    if (entries.items.len == 0) {
        try diagnostics.writeAll("pass=performance_cost_model feature=graph reason=no executable graph instructions\n");
        return MlirStateError.InvalidLoweringPlan;
    }

    const owned_entries = try entries.toOwnedSlice(allocator);
    errdefer deinitExtractedCostLedgerEntries(allocator, owned_entries);
    try verifyCostLedgerEntries(owned_entries, diagnostics);
    return owned_entries;
}

pub fn extractLoweringRecords(
    allocator: std.mem.Allocator,
    session: *const MlirSession,
    cost_count: usize,
    diagnostics: *std.Io.Writer,
) ![]compiler_facts.LoweringRecord {
    try requireLoweringExtractState(session, diagnostics);

    const lowerings_attr = getAttr(session.moduleOperation(), "pjrtx.lowering.records");
    if (mlir.mlirAttributeIsNull(lowerings_attr) or !mlir.mlirAttributeIsAArray(lowerings_attr)) {
        try diagnostics.writeAll("pass=lowering_extract feature=lowering-records reason=missing MLIR lowering records attribute\n");
        return MlirStateError.InvalidExtractionState;
    }

    var records: std.ArrayList(compiler_facts.LoweringRecord) = .empty;
    errdefer {
        for (records.items) |record| deinitLoweringRecordFields(allocator, record);
        records.deinit(allocator);
    }

    const count = mlir.mlirArrayAttrGetNumElements(lowerings_attr);
    var index: isize = 0;
    while (index < count) : (index += 1) {
        const record_attr = mlir.mlirArrayAttrGetElement(lowerings_attr, index);
        try records.append(allocator, try parseLoweringRecordAttr(allocator, record_attr, diagnostics));
    }

    const owned_records = try records.toOwnedSlice(allocator);
    errdefer deinitExtractedLoweringRecords(allocator, owned_records);
    try verifyLoweringRecords(owned_records, cost_count, diagnostics);
    return owned_records;
}

pub fn extractLoweringRegionFacts(
    allocator: std.mem.Allocator,
    session: *const MlirSession,
    lowering_count: usize,
    diagnostics: *std.Io.Writer,
) ![]LoweringRegionFact {
    try requireLoweringExtractState(session, diagnostics);

    const facts_attr = getAttr(session.moduleOperation(), "pjrtx.lowering.region_facts");
    if (mlir.mlirAttributeIsNull(facts_attr) or !mlir.mlirAttributeIsAArray(facts_attr)) {
        try diagnostics.writeAll("pass=lowering_region_extract feature=region-facts reason=missing MLIR lowering region facts attribute\n");
        return MlirStateError.InvalidExtractionState;
    }

    var facts: std.ArrayList(LoweringRegionFact) = .empty;
    errdefer {
        for (facts.items) |fact| deinitLoweringRegionFactFields(allocator, fact);
        facts.deinit(allocator);
    }

    const count = mlir.mlirArrayAttrGetNumElements(facts_attr);
    if (count != std.math.cast(isize, lowering_count) orelse unreachable) {
        try diagnostics.writeAll("pass=lowering_region_extract feature=region-facts reason=region fact count does not match lowering count\n");
        return MlirStateError.InvalidLoweringPlan;
    }

    var index: isize = 0;
    while (index < count) : (index += 1) {
        const fact_attr = mlir.mlirArrayAttrGetElement(facts_attr, index);
        try facts.append(allocator, try parseLoweringRegionFactAttr(allocator, fact_attr, diagnostics));
    }

    const owned_facts = try facts.toOwnedSlice(allocator);
    errdefer deinitExtractedLoweringRegionFacts(allocator, owned_facts);
    try verifyExtractedLoweringRegionFacts(owned_facts, diagnostics);
    return owned_facts;
}

fn deriveLoweringRecordsFromMlir(
    allocator: std.mem.Allocator,
    session: *const MlirSession,
    diagnostics: *std.Io.Writer,
) ![]compiler_facts.LoweringRecord {
    const module_op = session.moduleOperation();
    const costs_attr = getAttr(module_op, "pjrtx.performance.cost_ledger");
    if (mlir.mlirAttributeIsNull(costs_attr) or !mlir.mlirAttributeIsAArray(costs_attr)) {
        try diagnostics.writeAll("pass=lowering_region_form feature=cost reason=missing MLIR cost ledger attribute\n");
        return MlirStateError.InvalidLoweringPlan;
    }

    var cost_ledger: std.ArrayList(compiler_facts.CostLedgerEntry) = .empty;
    defer {
        for (cost_ledger.items) |entry| deinitCostLedgerEntryFields(allocator, entry);
        cost_ledger.deinit(allocator);
    }
    const cost_count = mlir.mlirArrayAttrGetNumElements(costs_attr);
    var cost_index: isize = 0;
    while (cost_index < cost_count) : (cost_index += 1) {
        const cost_attr = mlir.mlirArrayAttrGetElement(costs_attr, cost_index);
        try cost_ledger.append(allocator, try parseCostLedgerEntryAttr(allocator, cost_attr, diagnostics));
    }
    try verifyCostLedgerEntries(cost_ledger.items, diagnostics);

    const placements_attr = getAttr(module_op, "pjrtx.placement.records");
    if (mlir.mlirAttributeIsNull(placements_attr) or !mlir.mlirAttributeIsAArray(placements_attr)) {
        try diagnostics.writeAll("pass=lowering_region_form feature=placement reason=missing MLIR placement records attribute\n");
        return MlirStateError.InvalidLoweringPlan;
    }

    const fusion_candidates_attr = getAttr(module_op, "pjrtx.fusion.candidates");
    if (mlir.mlirAttributeIsNull(fusion_candidates_attr) or !mlir.mlirAttributeIsAArray(fusion_candidates_attr)) {
        try diagnostics.writeAll("pass=lowering_region_form feature=fusion reason=missing MLIR fusion candidate attribute\n");
        return MlirStateError.InvalidLoweringPlan;
    }

    var placements: std.ArrayList(compiler_facts.PlacementRecord) = .empty;
    defer {
        for (placements.items) |placement| {
            allocator.free(placement.output_value_ids);
            allocator.free(placement.logical_tile_shape);
            allocator.free(placement.reason);
        }
        placements.deinit(allocator);
    }
    const placement_count = mlir.mlirArrayAttrGetNumElements(placements_attr);
    var placement_index: isize = 0;
    while (placement_index < placement_count) : (placement_index += 1) {
        const placement_attr = mlir.mlirArrayAttrGetElement(placements_attr, placement_index);
        try placements.append(allocator, try parsePlacementRecordAttr(allocator, placement_attr, diagnostics));
    }

    var fusion_groups: std.ArrayList(compiler_facts.FusionGroup) = .empty;
    defer {
        for (fusion_groups.items) |group| {
            allocator.free(group.kind);
            allocator.free(group.graph_instruction_ids);
            allocator.free(group.reason);
        }
        fusion_groups.deinit(allocator);
    }
    const fusion_count = mlir.mlirArrayAttrGetNumElements(fusion_candidates_attr);
    var fusion_index: isize = 0;
    while (fusion_index < fusion_count) : (fusion_index += 1) {
        const candidate_attr = mlir.mlirArrayAttrGetElement(fusion_candidates_attr, fusion_index);
        try fusion_groups.append(allocator, try parseFusionCandidateDecisionAttr(allocator, candidate_attr, diagnostics));
    }

    var records: std.ArrayList(compiler_facts.LoweringRecord) = .empty;
    var lowered_instructions: std.ArrayList(compiler_facts.GraphInstructionId) = .empty;
    defer lowered_instructions.deinit(allocator);
    errdefer {
        for (records.items) |record| deinitLoweringRecordFields(allocator, record);
        records.deinit(allocator);
    }

    for (placements.items) |placement| {
        if (graphInstructionIdInSlice(lowered_instructions.items, placement.graph_instruction_id)) continue;

        if (acceptedFusionGroupStartingAt(fusion_groups.items, placement.graph_instruction_id)) |group| {
            const instruction_ids = try copyGraphInstructionIdsForLowering(allocator, group.graph_instruction_ids);
            var instruction_ids_owned = true;
            errdefer if (instruction_ids_owned) allocator.free(instruction_ids);
            const cost_ids = try loweringCostIdsForInstructions(allocator, cost_ledger.items, instruction_ids, diagnostics);
            var cost_ids_owned = true;
            errdefer if (cost_ids_owned) allocator.free(cost_ids);
            const rejected = try duplicateLoweringAlternatives(allocator, lowering_policy.LoweringPolicy.rejectedAlternativesForDecision(.elementwise_fusion));
            var rejected_owned = true;
            errdefer if (rejected_owned) {
                for (rejected) |alternative| allocator.free(alternative);
                allocator.free(rejected);
            };
            const reason = try allocator.dupe(u8, "MLIR lowering_region_form selected this elementwise fusion region");
            var reason_owned = true;
            errdefer if (reason_owned) allocator.free(reason);

            try records.append(allocator, .{
                .id = .{ .index = std.math.cast(u32, records.items.len) orelse unreachable },
                .graph_instruction_ids = instruction_ids,
                .decision = .elementwise_fusion,
                .reason = reason,
                .rejected_alternatives = rejected,
                .cost_ledger_ids = cost_ids,
            });
            instruction_ids_owned = false;
            cost_ids_owned = false;
            rejected_owned = false;
            reason_owned = false;
            for (group.graph_instruction_ids) |instruction_id| try appendUniqueGraphInstructionId(allocator, &lowered_instructions, instruction_id);
            continue;
        }

        const instruction_ids = try allocator.alloc(compiler_facts.GraphInstructionId, 1);
        var instruction_ids_owned = true;
        errdefer if (instruction_ids_owned) allocator.free(instruction_ids);
        instruction_ids[0] = placement.graph_instruction_id;

        const cost_ids = try loweringCostIdsForInstructions(allocator, cost_ledger.items, instruction_ids, diagnostics);
        var cost_ids_owned = true;
        errdefer if (cost_ids_owned) allocator.free(cost_ids);

        const decision = loweringDecisionForCostLedger(cost_ledger.items, cost_ids[0]);
        const rejected = try duplicateLoweringAlternatives(allocator, lowering_policy.LoweringPolicy.rejectedAlternativesForDecision(decision));
        var rejected_owned = true;
        errdefer if (rejected_owned) {
            for (rejected) |alternative| allocator.free(alternative);
            allocator.free(rejected);
        };
        const reason = try allocator.dupe(u8, lowering_policy.LoweringPolicy.reasonForDecision(decision));
        var reason_owned = true;
        errdefer if (reason_owned) allocator.free(reason);

        try records.append(allocator, .{
            .id = .{ .index = std.math.cast(u32, records.items.len) orelse unreachable },
            .graph_instruction_ids = instruction_ids,
            .decision = decision,
            .reason = reason,
            .rejected_alternatives = rejected,
            .cost_ledger_ids = cost_ids,
        });
        instruction_ids_owned = false;
        cost_ids_owned = false;
        rejected_owned = false;
        reason_owned = false;
        try appendUniqueGraphInstructionId(allocator, &lowered_instructions, placement.graph_instruction_id);
    }

    const owned_records = try records.toOwnedSlice(allocator);
    errdefer deinitExtractedLoweringRecords(allocator, owned_records);
    try verifyLoweringRecords(owned_records, cost_ledger.items.len, diagnostics);
    return owned_records;
}

fn deriveLoweringRegionFactsFromMlir(
    allocator: std.mem.Allocator,
    session: *const MlirSession,
    lowerings: []const compiler_facts.LoweringRecord,
    diagnostics: *std.Io.Writer,
) ![]LoweringRegionFact {
    const module_op = session.moduleOperation();
    const placements_attr = getAttr(module_op, "pjrtx.placement.records");
    if (mlir.mlirAttributeIsNull(placements_attr) or !mlir.mlirAttributeIsAArray(placements_attr)) {
        try diagnostics.writeAll("pass=lowering_region_form feature=placement reason=missing MLIR placement records attribute\n");
        return MlirStateError.InvalidLoweringPlan;
    }

    const fusion_candidates_attr = getAttr(module_op, "pjrtx.fusion.candidates");
    if (mlir.mlirAttributeIsNull(fusion_candidates_attr) or !mlir.mlirAttributeIsAArray(fusion_candidates_attr)) {
        try diagnostics.writeAll("pass=lowering_region_form feature=fusion reason=missing MLIR fusion candidate attribute\n");
        return MlirStateError.InvalidLoweringPlan;
    }

    var placements: std.ArrayList(compiler_facts.PlacementRecord) = .empty;
    defer {
        for (placements.items) |placement| {
            allocator.free(placement.output_value_ids);
            allocator.free(placement.logical_tile_shape);
            allocator.free(placement.reason);
        }
        placements.deinit(allocator);
    }
    const placement_count = mlir.mlirArrayAttrGetNumElements(placements_attr);
    var placement_index: isize = 0;
    while (placement_index < placement_count) : (placement_index += 1) {
        const placement_attr = mlir.mlirArrayAttrGetElement(placements_attr, placement_index);
        try placements.append(allocator, try parsePlacementRecordAttr(allocator, placement_attr, diagnostics));
    }

    var fusion_groups: std.ArrayList(compiler_facts.FusionGroup) = .empty;
    defer {
        for (fusion_groups.items) |group| {
            allocator.free(group.kind);
            allocator.free(group.graph_instruction_ids);
            allocator.free(group.reason);
        }
        fusion_groups.deinit(allocator);
    }
    const fusion_count = mlir.mlirArrayAttrGetNumElements(fusion_candidates_attr);
    var fusion_index: isize = 0;
    while (fusion_index < fusion_count) : (fusion_index += 1) {
        const candidate_attr = mlir.mlirArrayAttrGetElement(fusion_candidates_attr, fusion_index);
        try fusion_groups.append(allocator, try parseFusionCandidateDecisionAttr(allocator, candidate_attr, diagnostics));
    }

    const facts = try allocator.alloc(LoweringRegionFact, lowerings.len);
    var initialized: usize = 0;
    errdefer {
        for (facts[0..initialized]) |fact| deinitLoweringRegionFactFields(allocator, fact);
        allocator.free(facts);
    }

    for (lowerings, 0..) |lowering, fact_index| {
        const first_placement = placementRecordForInstruction(placements.items, lowering.graph_instruction_ids[0]) orelse {
            try diagnostics.print("pass=lowering_region_form feature=placement reason=lowering has no first placement lowering={d}\n", .{lowering.id.index});
            return MlirStateError.InvalidLoweringPlan;
        };

        const placement_indices = try allocator.alloc(u32, lowering.graph_instruction_ids.len);
        var placement_indices_owned = true;
        errdefer if (placement_indices_owned) allocator.free(placement_indices);

        for (lowering.graph_instruction_ids, 0..) |instruction_id, index| {
            const placement = placementRecordForInstruction(placements.items, instruction_id) orelse {
                try diagnostics.print("pass=lowering_region_form feature=placement reason=lowering instruction has no placement lowering={d} instruction={d}\n", .{ lowering.id.index, instruction_id.index });
                return MlirStateError.InvalidLoweringPlan;
            };
            placement_indices[index] = placement.index;
        }

        const tile = try allocator.dupe(i64, first_placement.logical_tile_shape);
        var tile_owned = true;
        errdefer if (tile_owned) allocator.free(tile);

        const reason = try allocator.dupe(u8, "MLIR lowering_region_form joined fusion, placement, tile, memory, and codegen intent");
        var reason_owned = true;
        errdefer if (reason_owned) allocator.free(reason);

        facts[fact_index] = .{
            .lowering_record_id = lowering.id,
            .fusion_group_index = fusionGroupIndexForLowering(fusion_groups.items, lowering),
            .placement_record_indices = placement_indices,
            .logical_tile_shape = tile,
            .result_memory_space_id = first_placement.result_memory_space_id,
            .tile_memory_space_id = first_placement.tile_memory_space_id,
            .codegen_region = lowering.decision,
            .reason = reason,
        };
        placement_indices_owned = false;
        tile_owned = false;
        reason_owned = false;
        initialized += 1;
    }

    return facts;
}

pub fn commitKernelCodegenPlan(
    session: *MlirSession,
    records: []const core.KernelCodegenRecord,
    diagnostics: *std.Io.Writer,
) !void {
    try requireModuleState(session, .performance_modeled, diagnostics);
    try verifyKernelCodegenRecords(records, diagnostics);
    setKernelCodegenPlanAttr(session.context, session.moduleOperation(), records);
    try runKernelCodegenPlanExternalPass(session, diagnostics);
}

pub fn deriveKernelCodegenRecords(
    allocator: std.mem.Allocator,
    backend_kind: core.BackendKind,
    command_id: core.ScheduleCommandId,
    graph_execute_operation: []const u8,
    fused_elementwise_operation: []const u8,
    instructions: []const compiler_facts.GraphInstruction,
    lowerings: []const compiler_facts.LoweringRecord,
    placements: []const compiler_facts.PlacementRecord,
    memory_traffic: []const compiler_facts.MemoryTrafficRecord,
    capabilities: []const KernelCodegenCapabilityFact,
    diagnostics: *std.Io.Writer,
) ![]core.KernelCodegenRecord {
    var records: std.ArrayList(core.KernelCodegenRecord) = .empty;
    errdefer {
        for (records.items) |record| deinitKernelCodegenRecordFields(allocator, record, true);
        records.deinit(allocator);
    }

    for (lowerings) |lowering| {
        const graph_instruction_ids = try copyGraphInstructionIdsForLowering(allocator, lowering.graph_instruction_ids);
        var graph_instruction_ids_owned = true;
        errdefer if (graph_instruction_ids_owned) allocator.free(graph_instruction_ids);
        const cost_ledger_ids = try copyCostLedgerIds(allocator, lowering.cost_ledger_ids);
        var cost_ledger_ids_owned = true;
        errdefer if (cost_ledger_ids_owned) allocator.free(cost_ledger_ids);
        const memory_traffic_ids = try codegenMemoryTrafficIdsForLowering(allocator, memory_traffic, lowering.id);
        var memory_traffic_ids_owned = true;
        errdefer if (memory_traffic_ids_owned) allocator.free(memory_traffic_ids);
        const value_flow = try codegenValueFlow(allocator, instructions, lowering.graph_instruction_ids, diagnostics);
        var value_flow_owned = true;
        errdefer if (value_flow_owned) deinitCodegenValueFlow(allocator, value_flow);
        const placement = placementRecordForInstruction(placements, lowering.graph_instruction_ids[0]) orelse {
            try diagnostics.print("pass=kernel_codegen feature=placement reason=codegen lowering has no placement lowering={d}\n", .{lowering.id.index});
            return MlirStateError.InvalidKernelCodegenPlan;
        };
        const logical_tile_shape = try copyI64sForCodegen(allocator, placement.logical_tile_shape);
        var logical_tile_shape_owned = true;
        errdefer if (logical_tile_shape_owned) allocator.free(logical_tile_shape);
        const operation = try allocator.dupe(u8, codegenOperationForLowering(
            lowering,
            capabilities,
            graph_execute_operation,
            fused_elementwise_operation,
        ));
        var operation_owned = true;
        errdefer if (operation_owned) allocator.free(operation);
        const reason = try allocator.dupe(u8, codegenReasonForLowering(lowering.decision));
        var reason_owned = true;
        errdefer if (reason_owned) allocator.free(reason);

        try records.append(allocator, .{
            .id = .{ .index = std.math.cast(u32, records.items.len) orelse unreachable },
            .lowering_record_id = lowering.id,
            .command_id = command_id,
            .backend_kind = backend_kind,
            .kind = codegenKindForLowering(lowering.decision),
            .operation = operation,
            .shape = value_flow.shape,
            .logical_tile_shape = logical_tile_shape,
            .result_memory_space_id = placement.result_memory_space_id,
            .tile_memory_space_id = placement.tile_memory_space_id,
            .memory_pressure = codegenMemoryPressure(memory_traffic, memory_traffic_ids),
            .external_input_ids = value_flow.external_input_ids,
            .external_output_ids = value_flow.external_output_ids,
            .intermediate_value_ids = value_flow.intermediate_value_ids,
            .graph_instruction_ids = graph_instruction_ids,
            .cost_ledger_ids = cost_ledger_ids,
            .memory_traffic_ids = memory_traffic_ids,
            .expected_unit_id = codegenExpectedUnitForLowering(lowering, capabilities),
            .reason = reason,
        });
        graph_instruction_ids_owned = false;
        cost_ledger_ids_owned = false;
        memory_traffic_ids_owned = false;
        value_flow_owned = false;
        logical_tile_shape_owned = false;
        operation_owned = false;
        reason_owned = false;
    }

    const owned_records = try records.toOwnedSlice(allocator);
    errdefer deinitExtractedKernelCodegenRecords(allocator, owned_records);
    try verifyKernelCodegenRecords(owned_records, diagnostics);
    return owned_records;
}

pub fn commitPerformanceFacts(
    session: *MlirSession,
    cost_ledger: []const compiler_facts.CostLedgerEntry,
    memory_traffic: []const compiler_facts.MemoryTrafficRecord,
    lowering_count: usize,
    diagnostics: *std.Io.Writer,
) !void {
    try requireModuleState(session, .lowering_planned, diagnostics);
    try verifyCostLedgerEntries(cost_ledger, diagnostics);
    try verifyMemoryTrafficRecords(memory_traffic, cost_ledger.len, lowering_count, diagnostics);
    setPerformanceFactsAttrs(session.context, session.moduleOperation(), cost_ledger, memory_traffic);
    try runPerformanceFactsExternalPass(session, diagnostics);
}

pub fn deriveMemoryTrafficRecords(
    allocator: std.mem.Allocator,
    target: target_pkg.TargetDescription,
    values: []const compiler_facts.GraphValue,
    instructions: []const compiler_facts.GraphInstruction,
    cost_ledger: []const compiler_facts.CostLedgerEntry,
    lowerings: []const compiler_facts.LoweringRecord,
    placements: []const compiler_facts.PlacementRecord,
    diagnostics: *std.Io.Writer,
) ![]compiler_facts.MemoryTrafficRecord {
    var records: std.ArrayList(compiler_facts.MemoryTrafficRecord) = .empty;
    errdefer {
        for (records.items) |record| deinitMemoryTrafficRecordFields(allocator, record);
        records.deinit(allocator);
    }

    for (lowerings) |lowering| {
        try appendDeviceBoundaryTrafficRecord(
            allocator,
            &records,
            values,
            instructions,
            placements,
            lowering,
            diagnostics,
        );
        try appendTileMemoryTrafficRecords(
            allocator,
            &records,
            target,
            cost_ledger,
            lowering,
            placements,
            diagnostics,
        );
    }

    const owned_records = try records.toOwnedSlice(allocator);
    errdefer deinitExtractedMemoryTrafficRecords(allocator, owned_records);
    try verifyMemoryTrafficRecords(owned_records, cost_ledger.len, lowerings.len, diagnostics);
    return owned_records;
}

pub fn extractCostLedgerEntries(
    allocator: std.mem.Allocator,
    session: *const MlirSession,
    diagnostics: *std.Io.Writer,
) ![]compiler_facts.CostLedgerEntry {
    try requireCostLedgerExtractState(session, diagnostics);

    const costs_attr = getAttr(session.moduleOperation(), "pjrtx.performance.cost_ledger");
    if (mlir.mlirAttributeIsNull(costs_attr) or !mlir.mlirAttributeIsAArray(costs_attr)) {
        try diagnostics.writeAll("pass=performance_extract feature=cost-ledger reason=missing MLIR cost ledger attribute\n");
        return MlirStateError.InvalidExtractionState;
    }

    var costs: std.ArrayList(compiler_facts.CostLedgerEntry) = .empty;
    errdefer {
        for (costs.items) |entry| deinitCostLedgerEntryFields(allocator, entry);
        costs.deinit(allocator);
    }

    const count = mlir.mlirArrayAttrGetNumElements(costs_attr);
    var index: isize = 0;
    while (index < count) : (index += 1) {
        try costs.append(allocator, try parseCostLedgerEntryAttr(allocator, mlir.mlirArrayAttrGetElement(costs_attr, index), diagnostics));
    }

    const owned_costs = try costs.toOwnedSlice(allocator);
    errdefer deinitExtractedCostLedgerEntries(allocator, owned_costs);
    try verifyCostLedgerEntries(owned_costs, diagnostics);
    return owned_costs;
}

pub fn extractMemoryTrafficRecords(
    allocator: std.mem.Allocator,
    session: *const MlirSession,
    cost_count: usize,
    lowering_count: usize,
    diagnostics: *std.Io.Writer,
) ![]compiler_facts.MemoryTrafficRecord {
    try requirePerformanceExtractState(session, diagnostics);

    const traffic_attr = getAttr(session.moduleOperation(), "pjrtx.performance.memory_traffic");
    if (mlir.mlirAttributeIsNull(traffic_attr) or !mlir.mlirAttributeIsAArray(traffic_attr)) {
        try diagnostics.writeAll("pass=performance_extract feature=memory-traffic reason=missing MLIR memory traffic attribute\n");
        return MlirStateError.InvalidExtractionState;
    }

    var traffic: std.ArrayList(compiler_facts.MemoryTrafficRecord) = .empty;
    errdefer {
        for (traffic.items) |record| deinitMemoryTrafficRecordFields(allocator, record);
        traffic.deinit(allocator);
    }

    const count = mlir.mlirArrayAttrGetNumElements(traffic_attr);
    var index: isize = 0;
    while (index < count) : (index += 1) {
        try traffic.append(allocator, try parseMemoryTrafficRecordAttr(allocator, mlir.mlirArrayAttrGetElement(traffic_attr, index), diagnostics));
    }

    const owned_traffic = try traffic.toOwnedSlice(allocator);
    errdefer deinitExtractedMemoryTrafficRecords(allocator, owned_traffic);
    try verifyMemoryTrafficRecords(owned_traffic, cost_count, lowering_count, diagnostics);
    return owned_traffic;
}

pub fn extractKernelCodegenRecords(
    allocator: std.mem.Allocator,
    session: *const MlirSession,
    diagnostics: *std.Io.Writer,
) ![]core.KernelCodegenRecord {
    try requireModuleState(session, .codegen_planned, diagnostics);

    const codegen_attr = getAttr(session.moduleOperation(), "pjrtx.codegen.records");
    if (mlir.mlirAttributeIsNull(codegen_attr) or !mlir.mlirAttributeIsAArray(codegen_attr)) {
        try diagnostics.writeAll("pass=codegen_extract feature=codegen-records reason=missing MLIR codegen records attribute\n");
        return MlirStateError.InvalidExtractionState;
    }

    var records: std.ArrayList(core.KernelCodegenRecord) = .empty;
    errdefer {
        for (records.items) |record| deinitKernelCodegenRecordFields(allocator, record, true);
        records.deinit(allocator);
    }

    const count = mlir.mlirArrayAttrGetNumElements(codegen_attr);
    var index: isize = 0;
    while (index < count) : (index += 1) {
        const record_attr = mlir.mlirArrayAttrGetElement(codegen_attr, index);
        try records.append(allocator, try parseKernelCodegenRecordAttr(allocator, record_attr, diagnostics));
    }

    const owned_records = try records.toOwnedSlice(allocator);
    errdefer deinitExtractedKernelCodegenRecords(allocator, owned_records);
    try verifyKernelCodegenRecords(owned_records, diagnostics);
    return owned_records;
}

pub fn commitSchedulePlan(
    session: *MlirSession,
    commands: []const core.ScheduleCommand,
    overlaps: []const core.ScheduleOverlapRecord,
    diagnostics: *std.Io.Writer,
) !void {
    try requireModuleState(session, .codegen_planned, diagnostics);
    try verifyScheduleCommands(commands, diagnostics);
    try verifyScheduleOverlaps(overlaps, commands.len, diagnostics);
    setSchedulePlanAttrs(session.context, session.moduleOperation(), commands, overlaps);
    try runSchedulePlanExternalPass(session, diagnostics);
}

pub fn deriveSchedulePlan(
    allocator: std.mem.Allocator,
    parameter_ids: []const compiler_facts.GraphValueId,
    output_ids: []const compiler_facts.GraphValueId,
    lowerings: []const compiler_facts.LoweringRecord,
    cost_ledger: []const compiler_facts.CostLedgerEntry,
    diagnostics: *std.Io.Writer,
) !SchedulePlanFact {
    var commands: std.ArrayList(core.ScheduleCommand) = .empty;
    errdefer {
        for (commands.items) |command| deinitScheduleCommandFields(allocator, command);
        commands.deinit(allocator);
    }

    try appendScheduleCommand(
        allocator,
        &commands,
        .host_to_device,
        .{ .index = 0 },
        parameter_ids,
        parameter_ids,
        &.{},
        &.{},
        &.{},
    );

    const lowering_ids = try scheduleAllLoweringIds(allocator, lowerings);
    defer allocator.free(lowering_ids);
    const cost_ids = try scheduleAllCostIds(allocator, cost_ledger);
    defer allocator.free(cost_ids);
    const backend_dependencies = [_]core.CommandDependency{.{ .command_id = .{ .index = 0 }, .kind = .data }};
    try appendScheduleCommand(
        allocator,
        &commands,
        .backend_execute,
        .{ .index = 0 },
        parameter_ids,
        output_ids,
        &backend_dependencies,
        lowering_ids,
        cost_ids,
    );

    const d2h_dependencies = [_]core.CommandDependency{.{ .command_id = .{ .index = 1 }, .kind = .data }};
    try appendScheduleCommand(
        allocator,
        &commands,
        .device_to_host,
        .{ .index = 0 },
        output_ids,
        output_ids,
        &d2h_dependencies,
        &.{},
        &.{},
    );

    const owned_commands = try commands.toOwnedSlice(allocator);
    var commands_owned = true;
    errdefer if (commands_owned) deinitExtractedScheduleCommands(allocator, owned_commands);
    try verifyScheduleCommands(owned_commands, diagnostics);

    const overlaps = try deriveScheduleOverlapRecords(allocator, owned_commands, diagnostics);
    errdefer deinitExtractedScheduleOverlapRecords(allocator, overlaps);

    commands_owned = false;
    return .{ .commands = owned_commands, .overlaps = overlaps };
}

pub fn extractTargetDescription(
    allocator: std.mem.Allocator,
    session: *const MlirSession,
    diagnostics: *std.Io.Writer,
) !target_pkg.TargetDescription {
    try requireTargetExtractState(session, diagnostics);

    const target_attr = getAttr(session.moduleOperation(), "pjrtx.target.spec");
    if (mlir.mlirAttributeIsNull(target_attr) or !mlir.mlirAttributeIsADictionary(target_attr)) {
        try diagnostics.writeAll("pass=target_extract feature=target reason=missing MLIR target spec attribute\n");
        return MlirStateError.InvalidExtractionState;
    }

    const target = try parseTargetDescriptionAttr(allocator, target_attr, diagnostics);
    errdefer deinitExtractedTargetDescription(allocator, target);
    target_pkg.validateTargetDescription(target, diagnostics) catch {
        try diagnostics.writeAll("pass=target_extract feature=target reason=target description failed validation\n");
        return MlirStateError.InvalidTargetAttachment;
    };
    return target;
}

pub fn extractScheduleCommands(
    allocator: std.mem.Allocator,
    session: *const MlirSession,
    diagnostics: *std.Io.Writer,
) ![]core.ScheduleCommand {
    try requireScheduleExtractState(session, diagnostics);

    const commands_attr = getAttr(session.moduleOperation(), "pjrtx.schedule.commands");
    if (mlir.mlirAttributeIsNull(commands_attr) or !mlir.mlirAttributeIsAArray(commands_attr)) {
        try diagnostics.writeAll("pass=schedule_extract feature=commands reason=missing MLIR schedule commands attribute\n");
        return MlirStateError.InvalidExtractionState;
    }

    var commands: std.ArrayList(core.ScheduleCommand) = .empty;
    errdefer {
        for (commands.items) |command| deinitScheduleCommandFields(allocator, command);
        commands.deinit(allocator);
    }

    const count = mlir.mlirArrayAttrGetNumElements(commands_attr);
    var index: isize = 0;
    while (index < count) : (index += 1) {
        try commands.append(allocator, try parseScheduleCommandAttr(allocator, mlir.mlirArrayAttrGetElement(commands_attr, index), diagnostics));
    }

    const owned_commands = try commands.toOwnedSlice(allocator);
    errdefer deinitExtractedScheduleCommands(allocator, owned_commands);
    try verifyScheduleCommands(owned_commands, diagnostics);
    return owned_commands;
}

pub fn extractScheduleOverlapRecords(
    allocator: std.mem.Allocator,
    session: *const MlirSession,
    command_count: usize,
    diagnostics: *std.Io.Writer,
) ![]core.ScheduleOverlapRecord {
    try requireScheduleExtractState(session, diagnostics);

    const overlaps_attr = getAttr(session.moduleOperation(), "pjrtx.schedule.overlaps");
    if (mlir.mlirAttributeIsNull(overlaps_attr) or !mlir.mlirAttributeIsAArray(overlaps_attr)) {
        try diagnostics.writeAll("pass=schedule_extract feature=overlaps reason=missing MLIR schedule overlaps attribute\n");
        return MlirStateError.InvalidExtractionState;
    }

    var overlaps: std.ArrayList(core.ScheduleOverlapRecord) = .empty;
    errdefer {
        for (overlaps.items) |overlap| allocator.free(overlap.reason);
        overlaps.deinit(allocator);
    }

    const count = mlir.mlirArrayAttrGetNumElements(overlaps_attr);
    var index: isize = 0;
    while (index < count) : (index += 1) {
        try overlaps.append(allocator, try parseScheduleOverlapAttr(allocator, mlir.mlirArrayAttrGetElement(overlaps_attr, index), diagnostics));
    }

    const owned_overlaps = try overlaps.toOwnedSlice(allocator);
    errdefer deinitExtractedScheduleOverlapRecords(allocator, owned_overlaps);
    try verifyScheduleOverlaps(owned_overlaps, command_count, diagnostics);
    return owned_overlaps;
}

pub fn commitBackendBindings(
    session: *MlirSession,
    bindings: []const core.BackendBinding,
    diagnostics: *std.Io.Writer,
) !void {
    try requireModuleState(session, .scheduled, diagnostics);
    try verifyBackendBindings(bindings, diagnostics);
    setBackendBindingPlanAttr(session.context, session.moduleOperation(), bindings);
    try runBackendBindingExternalPass(session, diagnostics);
}

pub fn extractBackendBindings(
    allocator: std.mem.Allocator,
    session: *const MlirSession,
    diagnostics: *std.Io.Writer,
) ![]core.BackendBinding {
    try requireModuleState(session, .backend_bound, diagnostics);

    const bindings_attr = getAttr(session.moduleOperation(), "pjrtx.backend.bindings");
    if (mlir.mlirAttributeIsNull(bindings_attr) or !mlir.mlirAttributeIsAArray(bindings_attr)) {
        try diagnostics.writeAll("pass=backend_binding_extract feature=bindings reason=missing MLIR backend binding attribute\n");
        return MlirStateError.InvalidExtractionState;
    }

    var bindings: std.ArrayList(core.BackendBinding) = .empty;
    errdefer {
        for (bindings.items) |binding| deinitBackendBindingFields(allocator, binding, true);
        bindings.deinit(allocator);
    }

    const count = mlir.mlirArrayAttrGetNumElements(bindings_attr);
    var index: isize = 0;
    while (index < count) : (index += 1) {
        try bindings.append(allocator, try parseBackendBindingAttr(allocator, mlir.mlirArrayAttrGetElement(bindings_attr, index), diagnostics));
    }

    const owned_bindings = try bindings.toOwnedSlice(allocator);
    errdefer deinitExtractedBackendBindings(allocator, owned_bindings);
    try verifyBackendBindings(owned_bindings, diagnostics);
    return owned_bindings;
}

pub fn commitExecutableReadiness(
    session: *MlirSession,
    contract: ExecutableContract,
    diagnostics: *std.Io.Writer,
) !void {
    try requireModuleState(session, .backend_bound, diagnostics);
    try verifyExecutableContract(contract, diagnostics);
    setExecutableContractAttr(session.context, session.moduleOperation(), contract);
    try runExecutableReadyExternalPass(session, diagnostics);
}

pub fn extractExecutableContract(
    session: *const MlirSession,
    diagnostics: *std.Io.Writer,
) !ExecutableContract {
    try requireModuleState(session, .executable_ready, diagnostics);
    const contract_attr = getAttr(session.moduleOperation(), "pjrtx.executable.contract");
    return try parseExecutableContractAttr(contract_attr, diagnostics);
}

pub fn extractBackendExecutablePlan(
    allocator: std.mem.Allocator,
    session: *const MlirSession,
    diagnostics: *std.Io.Writer,
) !BackendExecutablePlanFact {
    try requireBackendExecutableExtractState(session, diagnostics);

    const plan_attr = getAttr(session.moduleOperation(), "pjrtx.backend.executable");
    if (mlir.mlirAttributeIsNull(plan_attr) or !mlir.mlirAttributeIsADictionary(plan_attr)) {
        try diagnostics.writeAll("pass=backend_executable_extract feature=plan reason=missing MLIR backend executable attribute\n");
        return MlirStateError.InvalidExtractionState;
    }

    const plan = try parseBackendExecutablePlanAttr(allocator, plan_attr, diagnostics);
    errdefer deinitExtractedBackendExecutablePlan(allocator, plan);
    try verifyBackendExecutablePlan(plan, diagnostics);
    return plan;
}

pub fn extractBackendKernelGraph(
    allocator: std.mem.Allocator,
    session: *const MlirSession,
    diagnostics: *std.Io.Writer,
) !BackendKernelGraphFact {
    try requireBackendKernelGraphExtractState(session, diagnostics);

    const graph_attr = getAttr(session.moduleOperation(), "pjrtx.backend.kernel_graph");
    if (mlir.mlirAttributeIsNull(graph_attr) or !mlir.mlirAttributeIsADictionary(graph_attr)) {
        try diagnostics.writeAll("pass=backend_kernel_graph_extract feature=graph reason=missing MLIR backend kernel graph attribute\n");
        return MlirStateError.InvalidExtractionState;
    }

    const graph = try parseBackendKernelGraphAttr(allocator, graph_attr, diagnostics);
    errdefer deinitExtractedBackendKernelGraph(allocator, graph);
    try verifyBackendKernelGraph(graph, diagnostics);
    return graph;
}

pub fn extractRuntimeAllocationPlan(
    allocator: std.mem.Allocator,
    session: *const MlirSession,
    diagnostics: *std.Io.Writer,
) !RuntimeAllocationPlanFact {
    try requireRuntimeAllocationExtractState(session, diagnostics);

    const plan_attr = getAttr(session.moduleOperation(), "pjrtx.runtime.allocation");
    if (mlir.mlirAttributeIsNull(plan_attr) or !mlir.mlirAttributeIsADictionary(plan_attr)) {
        try diagnostics.writeAll("pass=runtime_allocation_extract feature=plan reason=missing MLIR runtime allocation attribute\n");
        return MlirStateError.InvalidExtractionState;
    }

    const plan = try parseRuntimeAllocationPlanAttr(allocator, plan_attr, diagnostics);
    errdefer deinitExtractedRuntimeAllocationPlan(allocator, plan);
    try verifyRuntimeAllocationPlan(plan, diagnostics);
    return plan;
}

pub fn extractRuntimeStreamPlan(
    allocator: std.mem.Allocator,
    session: *const MlirSession,
    diagnostics: *std.Io.Writer,
) !RuntimeStreamPlanFact {
    try requireRuntimeStreamExtractState(session, diagnostics);

    const steps_attr = getAttr(session.moduleOperation(), "pjrtx.runtime.streams");
    if (mlir.mlirAttributeIsNull(steps_attr) or !mlir.mlirAttributeIsAArray(steps_attr)) {
        try diagnostics.writeAll("pass=runtime_stream_extract feature=steps reason=missing MLIR runtime stream steps attribute\n");
        return MlirStateError.InvalidExtractionState;
    }

    var steps: std.ArrayList(RuntimeStreamStepFact) = .empty;
    errdefer {
        for (steps.items) |step| deinitRuntimeStreamStepFields(allocator, step);
        steps.deinit(allocator);
    }

    const count = mlir.mlirArrayAttrGetNumElements(steps_attr);
    var index: isize = 0;
    while (index < count) : (index += 1) {
        try steps.append(allocator, try parseRuntimeStreamStepAttr(allocator, mlir.mlirArrayAttrGetElement(steps_attr, index), diagnostics));
    }

    const owned_steps = try steps.toOwnedSlice(allocator);
    errdefer allocator.free(owned_steps);
    const plan: RuntimeStreamPlanFact = .{ .steps = owned_steps };
    errdefer deinitExtractedRuntimeStreamPlan(allocator, plan);
    try verifyRuntimeStreamPlan(plan, diagnostics);
    return plan;
}

pub fn extractRuntimeProfileEvents(
    allocator: std.mem.Allocator,
    session: *const MlirSession,
    diagnostics: *std.Io.Writer,
) ![]RuntimeProfileEventFact {
    try requireRuntimeProfileExtractState(session, diagnostics);

    const events_attr = getAttr(session.moduleOperation(), "pjrtx.runtime.profile_events");
    if (mlir.mlirAttributeIsNull(events_attr) or !mlir.mlirAttributeIsAArray(events_attr)) {
        try diagnostics.writeAll("pass=runtime_profile_extract feature=events reason=missing MLIR runtime profile events attribute\n");
        return MlirStateError.InvalidExtractionState;
    }

    var events: std.ArrayList(RuntimeProfileEventFact) = .empty;
    errdefer {
        for (events.items) |event| deinitRuntimeProfileEventFields(allocator, event);
        events.deinit(allocator);
    }

    const count = mlir.mlirArrayAttrGetNumElements(events_attr);
    var index: isize = 0;
    while (index < count) : (index += 1) {
        try events.append(allocator, try parseRuntimeProfileEventAttr(allocator, mlir.mlirArrayAttrGetElement(events_attr, index), diagnostics));
    }

    const owned_events = try events.toOwnedSlice(allocator);
    errdefer deinitExtractedRuntimeProfileEvents(allocator, owned_events);
    try verifyRuntimeProfile(.{ .events = owned_events }, diagnostics);
    return owned_events;
}

pub fn extractRuntimeProfileJoins(
    allocator: std.mem.Allocator,
    session: *const MlirSession,
    diagnostics: *std.Io.Writer,
) ![]RuntimeProfileJoinFact {
    try requireRuntimeProfileJoinExtractState(session, diagnostics);

    const joins_attr = getAttr(session.moduleOperation(), "pjrtx.runtime.profile_joins");
    if (mlir.mlirAttributeIsNull(joins_attr) or !mlir.mlirAttributeIsAArray(joins_attr)) {
        try diagnostics.writeAll("pass=runtime_profile_join_extract feature=joins reason=missing MLIR runtime profile joins attribute\n");
        return MlirStateError.InvalidExtractionState;
    }

    var joins: std.ArrayList(RuntimeProfileJoinFact) = .empty;
    errdefer {
        for (joins.items) |join| deinitRuntimeProfileJoinFields(allocator, join);
        joins.deinit(allocator);
    }

    const count = mlir.mlirArrayAttrGetNumElements(joins_attr);
    var index: isize = 0;
    while (index < count) : (index += 1) {
        try joins.append(allocator, try parseRuntimeProfileJoinAttr(allocator, mlir.mlirArrayAttrGetElement(joins_attr, index), diagnostics));
    }

    const owned_joins = try joins.toOwnedSlice(allocator);
    errdefer deinitExtractedRuntimeProfileJoins(allocator, owned_joins);
    try verifyRuntimeProfileJoins(.{ .joins = owned_joins }, diagnostics);
    return owned_joins;
}

pub fn extractBackendProfileJoins(
    allocator: std.mem.Allocator,
    session: *const MlirSession,
    diagnostics: *std.Io.Writer,
) ![]BackendProfileJoinFact {
    try requireModuleState(session, .backend_profile_joined, diagnostics);

    const joins_attr = getAttr(session.moduleOperation(), "pjrtx.backend.profile_joins");
    if (mlir.mlirAttributeIsNull(joins_attr) or !mlir.mlirAttributeIsAArray(joins_attr)) {
        try diagnostics.writeAll("pass=backend_profile_join_extract feature=joins reason=missing MLIR backend profile joins attribute\n");
        return MlirStateError.InvalidExtractionState;
    }

    var joins: std.ArrayList(BackendProfileJoinFact) = .empty;
    errdefer {
        for (joins.items) |join| deinitBackendProfileJoinFields(allocator, join);
        joins.deinit(allocator);
    }

    const count = mlir.mlirArrayAttrGetNumElements(joins_attr);
    var index: isize = 0;
    while (index < count) : (index += 1) {
        try joins.append(allocator, try parseBackendProfileJoinAttr(allocator, mlir.mlirArrayAttrGetElement(joins_attr, index), diagnostics));
    }

    const owned_joins = try joins.toOwnedSlice(allocator);
    errdefer deinitExtractedBackendProfileJoins(allocator, owned_joins);
    try verifyBackendProfileJoins(.{ .joins = owned_joins }, diagnostics);
    return owned_joins;
}

pub fn writeBackendExecutablePlanFactSummary(plan: BackendExecutablePlanFact, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    try writer.print(
        "backend executable\n  backend={s} command={d} operation={s} calls={d}\n",
        .{ @tagName(plan.backend_kind), plan.command_id.index, plan.backend_operation, plan.calls.len },
    );
    for (plan.calls) |call| {
        try writer.print(
            "  call.{d} instruction={d} instructions=",
            .{ call.index, call.graph_instruction_id.index },
        );
        try writeFactInstructionIdList(writer, call.graph_instruction_ids);
        try writer.print(" feature={s} operation={s} unit=", .{ call.feature, call.backend_operation });
        if (call.expected_unit_id) |unit_id| {
            try writer.print("{d}", .{unit_id});
        } else {
            try writer.writeAll("unknown");
        }
        try writer.print(" inputs={d} outputs={d}\n", .{ call.input_value_ids.len, call.output_value_ids.len });
    }
}

pub fn writeBackendKernelGraphFactSummary(graph: BackendKernelGraphFact, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    try writer.print(
        "backend kernel graph\n  backend={s} command={d} nodes={d} edges={d}\n",
        .{ @tagName(graph.backend_kind), graph.command_id.index, graph.nodes.len, graph.edges.len },
    );
    for (graph.nodes) |node| {
        try writer.print(
            "  node.{d} call={d} instruction={d} instructions=",
            .{ node.index, node.call_index, node.graph_instruction_id.index },
        );
        try writeFactInstructionIdList(writer, node.graph_instruction_ids);
        try writer.print(
            " feature={s} operation={s} dtype={s} rank={d} inputs={d} outputs={d} attrs={s}\n",
            .{
                node.feature,
                node.backend_operation,
                @tagName(node.output_type.element_type),
                node.output_type.dims.len,
                node.input_value_ids.len,
                node.output_value_ids.len,
                node.attributes,
            },
        );
    }
    for (graph.edges, 0..) |edge, index| {
        try writer.print(
            "  edge.{d} value={d} node.{d}->node.{d}\n",
            .{ index, edge.value_id.index, edge.src_node_index, edge.dst_node_index },
        );
    }
}

pub fn writeRuntimeAllocationPlanFactSummary(plan: RuntimeAllocationPlanFact, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    try writer.writeAll("allocation plan\n");
    for (plan.allocations) |allocation| {
        try writer.print(
            "  buffer.{d} value={d} placement={s} memory={d} bytes={d} lifetime=command.{d}..command.{d}\n",
            .{
                allocation.index,
                allocation.value_id.index,
                allocation.placement,
                allocation.memory_space_id,
                allocation.size_bytes,
                allocation.first_command_id.index,
                allocation.last_command_id.index,
            },
        );
    }
    try writer.writeAll("command buffer uses\n");
    for (plan.command_buffer_uses) |use| {
        try writer.print("  command.{d} buffer.{d} access={s}\n", .{ use.command_id.index, use.buffer_index, use.access });
    }
    try writer.print("peak_device_bytes={d}\n", .{plan.peak_device_bytes});
}

pub fn writeRuntimeStreamPlanFactSummary(plan: RuntimeStreamPlanFact, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    try writer.writeAll("stream plan\n");
    for (plan.steps) |step| {
        try writer.print(
            "  command.{d} stream.{d} start=event.{d} done=event.{d} waits=",
            .{ step.command_id.index, step.stream.index, step.start_event_id, step.done_event_id },
        );
        if (step.wait_event_ids.len == 0) {
            try writer.writeAll("none");
        } else {
            for (step.wait_event_ids, 0..) |event_id, index| {
                if (index != 0) try writer.writeAll(",");
                try writer.print("event.{d}", .{event_id});
            }
        }
        try writer.writeByte('\n');
    }
}

pub fn writeRuntimeExecutionFactSummary(
    report: core.TraceReport,
    target: target_pkg.TargetDescription,
    cost_ledger: []const compiler_facts.CostLedgerEntry,
    schedule_commands: []const core.ScheduleCommand,
    allocation_plan: RuntimeAllocationPlanFact,
    stream_plan: RuntimeStreamPlanFact,
    profile_events: []const RuntimeProfileEventFact,
    writer: *std.Io.Writer,
) std.Io.Writer.Error!void {
    try writer.writeAll("runtime execution\n");
    try writer.print("peak_device_bytes={d}\n", .{allocation_plan.peak_device_bytes});
    try writer.writeAll("commands\n");
    for (schedule_commands) |command| {
        const predicted = profileMetricsForCommandFact(report, cost_ledger, command);
        const observed = runtimeProfileEventForCommand(profile_events, command.id);
        const stream_step = streamStepForCommandFact(stream_plan.steps, command.id);
        try writer.print(
            "  command.{d} kind={s} stream={d} predicted_bytes={d} observed_bytes=",
            .{ command.id.index, @tagName(command.kind), stream_step.stream.index, predicted.bytes },
        );
        if (observed) |event| {
            try writer.print(
                "{d} predicted_ops={d} observed_ops={d} event=profile.{d} ideal_transfer_ps={d}\n",
                .{
                    event.bytes,
                    predicted.logical_ops,
                    event.logical_ops,
                    event.index,
                    idealTransferPsForCommandFact(target, allocation_plan, command, predicted.bytes),
                },
            );
        } else {
            try writer.print(
                "missing predicted_ops={d} observed_ops=missing event=missing ideal_transfer_ps={d}\n",
                .{ predicted.logical_ops, idealTransferPsForCommandFact(target, allocation_plan, command, predicted.bytes) },
            );
        }
    }
    try writer.writeAll("streams\n");
    for (stream_plan.steps) |step| {
        try writer.print("  command.{d} start=event.{d} done=event.{d} waits={d}\n", .{ step.command_id.index, step.start_event_id, step.done_event_id, step.wait_event_ids.len });
    }
}

pub fn writeLoweringProfileFactSummary(
    report: core.TraceReport,
    cost_ledger: []const compiler_facts.CostLedgerEntry,
    schedule_commands: []const core.ScheduleCommand,
    profile_events: []const RuntimeProfileEventFact,
    writer: *std.Io.Writer,
) std.Io.Writer.Error!void {
    try writer.writeAll("lowering profiles\n");
    for (schedule_commands) |command| {
        if (command.kind != .backend_execute) continue;
        for (command.lowering_record_ids) |lowering_id| {
            const lowering = loweringRecordFact(report, lowering_id);
            const predicted = costMetricsFact(cost_ledger, lowering.cost_ledger_ids);
            const observed = runtimeProfileEventForLowering(profile_events, command.id, lowering.graph_instruction_ids);
            try writer.print(
                "  lowering.{d} command={d} decision={s} instructions=",
                .{ lowering.id.index, command.id.index, @tagName(lowering.decision) },
            );
            try writeFactInstructionIdList(writer, lowering.graph_instruction_ids);
            try writer.print(" predicted_bytes={d} observed_bytes=", .{predicted.bytes});
            if (observed) |event| {
                try writer.print(
                    "{d} predicted_ops={d} observed_ops={d} event=profile.{d}\n",
                    .{ event.bytes, predicted.logical_ops, event.logical_ops, event.index },
                );
            } else {
                try writer.print(
                    "missing predicted_ops={d} observed_ops=missing event=missing\n",
                    .{predicted.logical_ops},
                );
            }
        }
    }
}

pub fn writeRuntimeHardwareUtilizationFactSummary(
    report: core.TraceReport,
    target: target_pkg.TargetDescription,
    cost_ledger: []const compiler_facts.CostLedgerEntry,
    memory_traffic_records: []const compiler_facts.MemoryTrafficRecord,
    schedule_commands: []const core.ScheduleCommand,
    allocation_plan: RuntimeAllocationPlanFact,
    profile_events: []const RuntimeProfileEventFact,
    writer: *std.Io.Writer,
) std.Io.Writer.Error!void {
    try writer.writeAll("hardware utilization\n");

    try writer.writeAll("memory spaces\n");
    for (target.memory_spaces) |memory_space| {
        try writer.print(
            "  memory.{d} {s} peak_live_bytes={d} allocated_bytes={d}\n",
            .{
                memory_space.id,
                memory_space.name,
                peakLiveBytesForMemoryFact(allocation_plan, memory_space.id),
                allocatedBytesForMemoryFact(allocation_plan.allocations, memory_space.id),
            },
        );
    }

    try writer.writeAll("memory traffic\n");
    for (memory_traffic_records) |record| {
        const memory_space = memorySpaceByIdFact(target.memory_spaces, record.memory_space_id) orelse continue;
        const total_bytes = record.bytes_read + record.bytes_written;
        try writer.print(
            "  traffic.{d} lowering={d} memory={d} kind={s} bytes_read={d} bytes_written={d} total_bytes={d} bandwidth_bytes_per_second=",
            .{
                record.id.index,
                record.lowering_record_id.index,
                record.memory_space_id,
                @tagName(record.kind),
                record.bytes_read,
                record.bytes_written,
                total_bytes,
            },
        );
        try writeOptionalF64Fact(writer, memory_space.bandwidth_bytes_per_second);
        try writer.print(" ideal_memory_ps={d}\n", .{idealTransferPsFact(total_bytes, memory_space.bandwidth_bytes_per_second)});
    }

    try writer.writeAll("lowering roofline\n");
    for (report.lowering_records) |lowering| {
        const metrics = costMetricsFact(cost_ledger, lowering.cost_ledger_ids);
        const ideal_compute_ps = idealComputePsForLoweringFact(target, cost_ledger, lowering);
        const ideal_memory_ps = idealMemoryPsForLoweringFact(target, memory_traffic_records, lowering.id);
        const observed = runtimeProfileEventForLoweringRecord(report, profile_events, lowering);
        try writer.print(
            "  lowering.{d} decision={s} predicted_ops={d} predicted_bytes={d} observed_ops=",
            .{
                lowering.id.index,
                @tagName(lowering.decision),
                metrics.logical_ops,
                metrics.bytes,
            },
        );
        if (observed) |event| {
            try writer.print("{d} observed_bytes={d}", .{ event.logical_ops, event.bytes });
        } else {
            try writer.writeAll("missing observed_bytes=missing");
        }
        try writer.print(
            " ideal_compute_ps={d} ideal_memory_ps={d} limiting={s} event=",
            .{
                ideal_compute_ps,
                ideal_memory_ps,
                limitingResourceFact(ideal_compute_ps, ideal_memory_ps),
            },
        );
        if (observed) |event| {
            try writer.print("profile.{d}\n", .{event.index});
        } else {
            try writer.writeAll("missing\n");
        }
    }

    try writer.writeAll("transfer edges\n");
    for (target.transfer_edges) |edge| {
        const predicted_bytes = transferBytesForEdgeFact(report, target, cost_ledger, schedule_commands, allocation_plan, edge);
        try writer.print(
            "  edge.{d} memory.{d}->memory.{d} predicted_bytes={d} bandwidth_bytes_per_second=",
            .{ edge.id, edge.src_memory_space, edge.dst_memory_space, predicted_bytes },
        );
        try writeOptionalF64Fact(writer, edge.bandwidth_bytes_per_second);
        try writer.print(" ideal_transfer_ps={d}\n", .{idealTransferPsFact(predicted_bytes, edge.bandwidth_bytes_per_second)});
    }

    try writer.writeAll("execution units\n");
    for (target.execution_units) |unit| {
        const metrics = costMetricsForUnitFact(cost_ledger, unit.id);
        try writer.print(
            "  unit.{d} {s} kind={s} predicted_ops={d} predicted_bytes={d}\n",
            .{ unit.id, unit.name, @tagName(unit.kind), metrics.logical_ops, metrics.bytes },
        );
        for (unit.dtype_rates) |rate| {
            const rate_metrics = costMetricsForUnitRateFact(cost_ledger, unit.id, rate);
            try writer.print(
                "    rate dtype={s} class={s} peak_ops_per_second=",
                .{ @tagName(rate.dtype), @tagName(rate.op_class) },
            );
            try writeOptionalF64Fact(writer, rate.ops_per_second);
            try writer.print(
                " source={s} predicted_ops={d} predicted_bytes={d} ideal_compute_ps={d}\n",
                .{ @tagName(rate.source), rate_metrics.logical_ops, rate_metrics.bytes, idealComputePsFact(rate_metrics.logical_ops, rate.ops_per_second) },
            );
        }
    }
    const unknown = costMetricsForUnknownUnitFact(cost_ledger);
    if (unknown.logical_ops != 0 or unknown.bytes != 0) {
        try writer.print("  unit.unknown predicted_ops={d} predicted_bytes={d}\n", .{ unknown.logical_ops, unknown.bytes });
    }
}

pub fn writeBackendCallProfileFactSummary(
    target: target_pkg.TargetDescription,
    cost_ledger: []const compiler_facts.CostLedgerEntry,
    memory_traffic_records: []const compiler_facts.MemoryTrafficRecord,
    executable: BackendExecutablePlanFact,
    profile_events: []const RuntimeProfileEventFact,
    profile_joins: []const BackendProfileJoinFact,
    writer: *std.Io.Writer,
) std.Io.Writer.Error!void {
    try writer.writeAll("backend call profiles\n");
    for (executable.calls) |call| {
        const predicted = callProfileMetricsFact(cost_ledger, call.graph_instruction_ids);
        const observed = runtimeProfileEventForBackendCallJoin(profile_events, profile_joins, call.index);
        const ideal_compute_ps = idealComputePsForCallFact(target, cost_ledger, call.graph_instruction_ids);
        const ideal_memory_ps = idealMemoryPsForCallFact(target, memory_traffic_records, call);
        const limit = limitingResourceFact(ideal_compute_ps, ideal_memory_ps);
        try writer.print(
            "  call.{d} command={d} operation={s} instructions=",
            .{ call.index, executable.command_id.index, call.backend_operation },
        );
        try writeFactInstructionIdList(writer, call.graph_instruction_ids);
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
            try writeMemoryTrafficListFact(writer, target, memory_traffic_records, call);
            try writer.print(" event=profile.{d}\n", .{event.index});
        } else {
            try writer.print(
                "missing predicted_ops={d} observed_ops=missing ideal_compute_ps={d} ideal_memory_ps={d} limiting={s} memory=",
                .{ predicted.logical_ops, ideal_compute_ps, ideal_memory_ps, limit },
            );
            try writeMemoryTrafficListFact(writer, target, memory_traffic_records, call);
            try writer.writeAll(" event=missing\n");
        }
    }
}

pub fn commitBackendExecutablePlan(
    session: *MlirSession,
    plan: BackendExecutablePlanFact,
    diagnostics: *std.Io.Writer,
) !void {
    try requireModuleState(session, .executable_ready, diagnostics);
    try verifyBackendExecutablePlan(plan, diagnostics);
    setBackendExecutablePlanAttr(session.context, session.moduleOperation(), plan);
    try runBackendExecutablePlanExternalPass(session, diagnostics);
}

pub fn commitBackendKernelGraph(
    session: *MlirSession,
    graph: BackendKernelGraphFact,
    diagnostics: *std.Io.Writer,
) !void {
    try requireModuleState(session, .backend_executable_planned, diagnostics);
    try verifyBackendKernelGraph(graph, diagnostics);
    setBackendKernelGraphAttr(session.context, session.moduleOperation(), graph);
    try runBackendKernelGraphExternalPass(session, diagnostics);
}

pub fn commitRuntimeAllocationPlan(
    session: *MlirSession,
    plan: RuntimeAllocationPlanFact,
    diagnostics: *std.Io.Writer,
) !void {
    try requireRuntimeAllocationInputState(session, diagnostics);
    try verifyRuntimeAllocationPlan(plan, diagnostics);
    setRuntimeAllocationPlanAttr(session.context, session.moduleOperation(), plan);
    try runRuntimeAllocationExternalPass(session, diagnostics);
}

pub fn commitRuntimeStreamPlan(
    session: *MlirSession,
    plan: RuntimeStreamPlanFact,
    diagnostics: *std.Io.Writer,
) !void {
    try requireModuleState(session, .runtime_allocation_planned, diagnostics);
    try verifyRuntimeStreamPlan(plan, diagnostics);
    setRuntimeStreamPlanAttr(session.context, session.moduleOperation(), plan);
    try runRuntimeStreamExternalPass(session, diagnostics);
}

pub fn commitRuntimeProfile(
    session: *MlirSession,
    profile: RuntimeProfileFact,
    diagnostics: *std.Io.Writer,
) !void {
    try requireModuleState(session, .runtime_stream_planned, diagnostics);
    try verifyRuntimeProfile(profile, diagnostics);
    setRuntimeProfileAttr(session.context, session.moduleOperation(), profile);
    try runRuntimeProfileExternalPass(session, diagnostics);
}

pub fn commitRuntimeProfileJoins(
    session: *MlirSession,
    plan: RuntimeProfileJoinPlanFact,
    diagnostics: *std.Io.Writer,
) !void {
    try requireModuleState(session, .runtime_profiled, diagnostics);
    try verifyRuntimeProfileJoins(plan, diagnostics);
    setRuntimeProfileJoinAttr(session.context, session.moduleOperation(), plan);
    try runRuntimeProfileJoinExternalPass(session, diagnostics);
}

pub fn commitBackendProfileJoins(
    session: *MlirSession,
    plan: BackendProfileJoinPlanFact,
    diagnostics: *std.Io.Writer,
) !void {
    try requireModuleState(session, .runtime_profile_joined, diagnostics);
    try verifyBackendProfileJoins(plan, diagnostics);
    setBackendProfileJoinAttr(session.context, session.moduleOperation(), plan);
    try runBackendProfileJoinExternalPass(session, diagnostics);
}

pub fn deinitExtractedFusionGroups(allocator: std.mem.Allocator, groups: []compiler_facts.FusionGroup) void {
    for (groups) |group| {
        allocator.free(group.kind);
        allocator.free(group.graph_instruction_ids);
        allocator.free(group.reason);
    }
    allocator.free(groups);
}

pub fn deinitExtractedPlacementRecords(allocator: std.mem.Allocator, records: []compiler_facts.PlacementRecord) void {
    for (records) |record| {
        allocator.free(record.output_value_ids);
        allocator.free(record.logical_tile_shape);
        allocator.free(record.reason);
    }
    allocator.free(records);
}

pub fn deinitExtractedCollectivePlanRecords(allocator: std.mem.Allocator, records: []compiler_facts.CollectivePlanRecord) void {
    for (records) |record| allocator.free(record.reason);
    allocator.free(records);
}

pub fn deinitExtractedLoweringRecords(allocator: std.mem.Allocator, records: []compiler_facts.LoweringRecord) void {
    for (records) |record| deinitLoweringRecordFields(allocator, record);
    allocator.free(records);
}

pub fn deinitExtractedLoweringRegionFacts(allocator: std.mem.Allocator, facts: []LoweringRegionFact) void {
    for (facts) |fact| deinitLoweringRegionFactFields(allocator, fact);
    allocator.free(facts);
}

pub fn deinitExtractedKernelCodegenRecords(allocator: std.mem.Allocator, records: []core.KernelCodegenRecord) void {
    for (records) |record| deinitKernelCodegenRecordFields(allocator, record, true);
    allocator.free(records);
}

pub fn deinitExtractedScheduleCommands(allocator: std.mem.Allocator, commands: []core.ScheduleCommand) void {
    for (commands) |command| deinitScheduleCommandFields(allocator, command);
    allocator.free(commands);
}

pub fn deinitExtractedTargetDescription(allocator: std.mem.Allocator, target: target_pkg.TargetDescription) void {
    allocator.free(target.name);
    for (target.devices) |device| {
        allocator.free(device.name);
        allocator.free(device.memory_space_ids);
        allocator.free(device.execution_unit_ids);
    }
    allocator.free(target.devices);
    for (target.memory_spaces) |memory_space| {
        allocator.free(memory_space.name);
        allocator.free(memory_space.note);
    }
    allocator.free(target.memory_spaces);
    for (target.transfer_edges) |edge| allocator.free(edge.note);
    allocator.free(target.transfer_edges);
    for (target.execution_units) |unit| {
        allocator.free(unit.name);
        for (unit.dtype_rates) |rate| allocator.free(rate.note);
        allocator.free(unit.dtype_rates);
    }
    allocator.free(target.execution_units);
}

pub fn deinitExtractedCostLedgerEntries(allocator: std.mem.Allocator, entries: []compiler_facts.CostLedgerEntry) void {
    for (entries) |entry| deinitCostLedgerEntryFields(allocator, entry);
    allocator.free(entries);
}

pub fn deinitExtractedMemoryTrafficRecords(allocator: std.mem.Allocator, records: []compiler_facts.MemoryTrafficRecord) void {
    for (records) |record| deinitMemoryTrafficRecordFields(allocator, record);
    allocator.free(records);
}

pub fn deinitExtractedScheduleOverlapRecords(allocator: std.mem.Allocator, overlaps: []core.ScheduleOverlapRecord) void {
    for (overlaps) |overlap| allocator.free(overlap.reason);
    allocator.free(overlaps);
}

pub fn deinitDerivedSchedulePlan(allocator: std.mem.Allocator, plan: SchedulePlanFact) void {
    deinitExtractedScheduleCommands(allocator, plan.commands);
    deinitExtractedScheduleOverlapRecords(allocator, plan.overlaps);
}

pub fn deinitExtractedBackendBindings(allocator: std.mem.Allocator, bindings: []core.BackendBinding) void {
    for (bindings) |binding| deinitBackendBindingFields(allocator, binding, true);
    allocator.free(bindings);
}

pub fn deinitExtractedBackendExecutablePlan(allocator: std.mem.Allocator, plan: BackendExecutablePlanFact) void {
    allocator.free(plan.backend_operation);
    for (plan.calls) |call| deinitBackendExecutableCallFields(allocator, call);
    allocator.free(plan.calls);
}

pub fn deinitExtractedBackendKernelGraph(allocator: std.mem.Allocator, graph: BackendKernelGraphFact) void {
    for (graph.nodes) |node| deinitBackendKernelGraphNodeFields(allocator, node);
    allocator.free(graph.nodes);
    allocator.free(graph.edges);
}

pub fn deinitExtractedRuntimeAllocationPlan(allocator: std.mem.Allocator, plan: RuntimeAllocationPlanFact) void {
    for (plan.allocations) |allocation| deinitRuntimeAllocationFields(allocator, allocation);
    allocator.free(plan.allocations);
    for (plan.command_buffer_uses) |use| deinitRuntimeBufferUseFields(allocator, use);
    allocator.free(plan.command_buffer_uses);
}

pub fn deinitExtractedRuntimeStreamPlan(allocator: std.mem.Allocator, plan: RuntimeStreamPlanFact) void {
    for (plan.steps) |step| deinitRuntimeStreamStepFields(allocator, step);
    allocator.free(plan.steps);
}

pub fn deinitExtractedRuntimeProfileEvents(allocator: std.mem.Allocator, events: []RuntimeProfileEventFact) void {
    for (events) |event| deinitRuntimeProfileEventFields(allocator, event);
    allocator.free(events);
}

pub fn deinitExtractedRuntimeProfileJoins(allocator: std.mem.Allocator, joins: []RuntimeProfileJoinFact) void {
    for (joins) |join| deinitRuntimeProfileJoinFields(allocator, join);
    allocator.free(joins);
}

pub fn deinitExtractedBackendProfileJoins(allocator: std.mem.Allocator, joins: []BackendProfileJoinFact) void {
    for (joins) |join| deinitBackendProfileJoinFields(allocator, join);
    allocator.free(joins);
}

fn deinitKernelCodegenRecordFields(allocator: std.mem.Allocator, record: core.KernelCodegenRecord, strings_owned: bool) void {
    if (strings_owned) {
        allocator.free(record.operation);
        allocator.free(record.reason);
    }
    allocator.free(record.logical_tile_shape);
    allocator.free(record.external_input_ids);
    allocator.free(record.external_output_ids);
    allocator.free(record.intermediate_value_ids);
    allocator.free(record.graph_instruction_ids);
    allocator.free(record.cost_ledger_ids);
    allocator.free(record.memory_traffic_ids);
}

fn deinitLoweringRecordFields(allocator: std.mem.Allocator, record: compiler_facts.LoweringRecord) void {
    allocator.free(record.graph_instruction_ids);
    allocator.free(record.reason);
    for (record.rejected_alternatives) |alternative| allocator.free(alternative);
    allocator.free(record.rejected_alternatives);
    allocator.free(record.cost_ledger_ids);
}

fn deinitLoweringRegionFactFields(allocator: std.mem.Allocator, fact: LoweringRegionFact) void {
    allocator.free(fact.placement_record_indices);
    allocator.free(fact.logical_tile_shape);
    allocator.free(fact.reason);
}

fn placementRecordForInstruction(records: []const compiler_facts.PlacementRecord, instruction_id: compiler_facts.GraphInstructionId) ?compiler_facts.PlacementRecord {
    for (records) |record| {
        if (record.graph_instruction_id.eql(instruction_id)) return record;
    }
    return null;
}

fn acceptedFusionGroupStartingAt(groups: []const compiler_facts.FusionGroup, instruction_id: compiler_facts.GraphInstructionId) ?compiler_facts.FusionGroup {
    for (groups) |group| {
        if (group.decision != .accepted or group.graph_instruction_ids.len == 0) continue;
        if (group.graph_instruction_ids[0].eql(instruction_id)) return group;
    }
    return null;
}

fn fusionGroupIndexForLowering(groups: []const compiler_facts.FusionGroup, lowering: compiler_facts.LoweringRecord) ?u32 {
    for (groups) |group| {
        if (group.decision != .accepted) continue;
        if (graphInstructionIdSlicesEqual(group.graph_instruction_ids, lowering.graph_instruction_ids)) return group.index;
    }
    return null;
}

fn graphInstructionIdSlicesEqual(lhs: []const compiler_facts.GraphInstructionId, rhs: []const compiler_facts.GraphInstructionId) bool {
    if (lhs.len != rhs.len) return false;
    for (lhs, rhs) |lhs_id, rhs_id| {
        if (!lhs_id.eql(rhs_id)) return false;
    }
    return true;
}

fn graphInstructionIdInSlice(instructions: []const compiler_facts.GraphInstructionId, instruction_id: compiler_facts.GraphInstructionId) bool {
    for (instructions) |candidate| {
        if (candidate.eql(instruction_id)) return true;
    }
    return false;
}

fn appendUniqueGraphInstructionId(
    allocator: std.mem.Allocator,
    instructions: *std.ArrayList(compiler_facts.GraphInstructionId),
    instruction_id: compiler_facts.GraphInstructionId,
) !void {
    if (!graphInstructionIdInSlice(instructions.items, instruction_id)) try instructions.append(allocator, instruction_id);
}

fn copyGraphInstructionIdsForLowering(
    allocator: std.mem.Allocator,
    instructions: []const compiler_facts.GraphInstructionId,
) ![]const compiler_facts.GraphInstructionId {
    const copy = try allocator.alloc(compiler_facts.GraphInstructionId, instructions.len);
    @memcpy(copy, instructions);
    return copy;
}

fn loweringCostIdsForInstructions(
    allocator: std.mem.Allocator,
    cost_ledger: []const compiler_facts.CostLedgerEntry,
    instruction_ids: []const compiler_facts.GraphInstructionId,
    diagnostics: *std.Io.Writer,
) ![]const compiler_facts.CostLedgerId {
    const cost_ids = try allocator.alloc(compiler_facts.CostLedgerId, instruction_ids.len);
    errdefer allocator.free(cost_ids);
    for (instruction_ids, 0..) |instruction_id, index| {
        cost_ids[index] = costIdForInstruction(cost_ledger, instruction_id) orelse {
            try diagnostics.print("pass=lowering_region_form feature=cost reason=missing cost ledger entry instruction={d}\n", .{instruction_id.index});
            return MlirStateError.InvalidLoweringPlan;
        };
    }
    return cost_ids;
}

fn costIdForInstruction(cost_ledger: []const compiler_facts.CostLedgerEntry, instruction_id: compiler_facts.GraphInstructionId) ?compiler_facts.CostLedgerId {
    for (cost_ledger) |entry| {
        for (entry.graph_instruction_ids) |entry_instruction_id| {
            if (entry_instruction_id.eql(instruction_id)) return entry.id;
        }
    }
    return null;
}

fn costCapabilityForInstruction(capabilities: []const CostCapabilityFact, instruction_id: compiler_facts.GraphInstructionId) ?CostCapabilityFact {
    for (capabilities) |capability| {
        if (capability.graph_instruction_id.eql(instruction_id)) return capability;
    }
    return null;
}

fn duplicateCostSourceRef(allocator: std.mem.Allocator, source: compiler_facts.SourceRef) !compiler_facts.SourceRef {
    const op_name = try allocator.dupe(u8, source.op_name);
    errdefer allocator.free(op_name);
    const location = try allocator.dupe(u8, source.location);
    errdefer allocator.free(location);
    return .{
        .id = source.id,
        .frontend = source.frontend,
        .op_name = op_name,
        .source_index = source.source_index,
        .location = location,
    };
}

fn costOutputValue(
    values: []const compiler_facts.GraphValue,
    instruction: compiler_facts.GraphInstruction,
    diagnostics: *std.Io.Writer,
) !compiler_facts.GraphValue {
    if (instruction.outputs.len > 0) return costGraphValue(values, instruction.outputs[0], diagnostics);
    if (instruction.inputs.len > 0) return costGraphValue(values, instruction.inputs[0], diagnostics);
    try diagnostics.print(
        "pass=performance_cost_model feature=graph-value reason=instruction has no input or output instruction={d}\n",
        .{instruction.id.index},
    );
    return MlirStateError.InvalidLoweringPlan;
}

fn costGraphValue(
    values: []const compiler_facts.GraphValue,
    id: compiler_facts.GraphValueId,
    diagnostics: *std.Io.Writer,
) !compiler_facts.GraphValue {
    const index: usize = std.math.cast(usize, id.index) orelse {
        try diagnostics.print("pass=performance_cost_model feature=graph-value reason=invalid value id value={d}\n", .{id.index});
        return MlirStateError.InvalidLoweringPlan;
    };
    if (index >= values.len or !values[index].id.eql(id)) {
        try diagnostics.print("pass=performance_cost_model feature=graph-value reason=missing graph value value={d}\n", .{id.index});
        return MlirStateError.InvalidLoweringPlan;
    }
    return values[index];
}

fn costLogicalOpsForInstruction(
    values: []const compiler_facts.GraphValue,
    instruction: compiler_facts.GraphInstruction,
    diagnostics: *std.Io.Writer,
) !u128 {
    switch (instruction.payload) {
        .dot_general => |payload| {
            const lhs = try costGraphValue(values, instruction.inputs[0], diagnostics);
            const rhs = try costGraphValue(values, instruction.inputs[1], diagnostics);
            const lhs_contracting: usize = std.math.cast(usize, payload.lhs_contracting_dimension) orelse unreachable;
            const rhs_contracting: usize = std.math.cast(usize, payload.rhs_contracting_dimension) orelse unreachable;
            const lhs_non_contracting: usize = if (lhs_contracting == 0) 1 else 0;
            const rhs_non_contracting: usize = if (rhs_contracting == 0) 1 else 0;
            const m = costPositiveDim(lhs.ty.dims[lhs_non_contracting]);
            const n = costPositiveDim(rhs.ty.dims[rhs_non_contracting]);
            const k = costPositiveDim(lhs.ty.dims[lhs_contracting]);
            return 2 * m * n * k;
        },
        .elementwise_unary, .elementwise_binary, .broadcast, .reshape, .transpose, .collective => {
            const output = try costGraphValue(values, instruction.outputs[0], diagnostics);
            return costTensorElements(output.ty);
        },
        .return_ => return 0,
    }
}

fn costBytesReadForInstruction(
    values: []const compiler_facts.GraphValue,
    instruction: compiler_facts.GraphInstruction,
    diagnostics: *std.Io.Writer,
) !u128 {
    var total: u128 = 0;
    for (instruction.inputs) |id| {
        const value = try costGraphValue(values, id, diagnostics);
        total += costTensorBytes(value.ty);
    }
    return total;
}

fn costBytesWrittenForInstruction(
    values: []const compiler_facts.GraphValue,
    instruction: compiler_facts.GraphInstruction,
    diagnostics: *std.Io.Writer,
) !u128 {
    var total: u128 = 0;
    for (instruction.outputs) |id| {
        const value = try costGraphValue(values, id, diagnostics);
        total += costTensorBytes(value.ty);
    }
    return total;
}

fn costTensorBytes(ty: compiler_facts.TensorType) u128 {
    const element_size = ty.element_type.byteSize() orelse 0;
    return costTensorElements(ty) * element_size;
}

fn costTensorElements(ty: compiler_facts.TensorType) u128 {
    var total: u128 = 1;
    for (ty.dims) |dim| total *= costPositiveDim(dim);
    return total;
}

fn costPositiveDim(dim: i64) u128 {
    return std.math.cast(u128, dim) orelse unreachable;
}

fn costClassForInstruction(instruction: compiler_facts.GraphInstruction) compiler_facts.CostOpClass {
    return switch (instruction.payload) {
        .dot_general => .matmul,
        .elementwise_unary => |payload| switch (payload.op) {
            .tanh => .transcendental,
        },
        .elementwise_binary, .broadcast => .elementwise,
        .reshape, .transpose => .transfer,
        .collective => .transfer,
        .return_ => .backend_kernel,
    };
}

fn costAccumulationDtypeForInstruction(instruction: compiler_facts.GraphInstruction, dtype: core.BufferType) ?core.BufferType {
    return switch (instruction.payload) {
        .dot_general => dtype,
        else => null,
    };
}

fn costFormulaForInstruction(instruction: compiler_facts.GraphInstruction) []const u8 {
    return switch (instruction.payload) {
        .dot_general => "2*M*N*K",
        .elementwise_unary => "numel(output)",
        .elementwise_binary => "numel(output)",
        .broadcast => "numel(output)",
        .reshape => "numel(output)",
        .transpose => "numel(output)",
        .collective => "collective_bytes(input,output)",
        .return_ => "0",
    };
}

fn costApproximationForInstruction(instruction: compiler_facts.GraphInstruction) []const u8 {
    return switch (instruction.payload) {
        .dot_general => "rank-2 V0 matmul logical multiply-adds",
        .elementwise_unary => "one logical transcendental op per output element",
        .elementwise_binary => "one logical binary op per output element",
        .broadcast => "logical materialized output bytes; backend may fuse",
        .reshape => "metadata-only when folded; otherwise unsupported before backend lowering",
        .transpose => "layout permutation when folded; otherwise unsupported before backend lowering",
        .collective => "collective payload is imported for algorithm selection, not executable in V0",
        .return_ => "",
    };
}

fn appendDeviceBoundaryTrafficRecord(
    allocator: std.mem.Allocator,
    records: *std.ArrayList(compiler_facts.MemoryTrafficRecord),
    values: []const compiler_facts.GraphValue,
    instructions: []const compiler_facts.GraphInstruction,
    placements: []const compiler_facts.PlacementRecord,
    lowering: compiler_facts.LoweringRecord,
    diagnostics: *std.Io.Writer,
) !void {
    const placement = placementRecordForInstruction(placements, lowering.graph_instruction_ids[0]) orelse {
        try diagnostics.print("pass=memory_traffic_refine feature=placement reason=lowering has no placement lowering={d}\n", .{lowering.id.index});
        return MlirStateError.InvalidKernelCodegenPlan;
    };
    const input_value_ids = try trafficRegionInputValueIds(allocator, instructions, lowering.graph_instruction_ids, diagnostics);
    defer allocator.free(input_value_ids);
    const output_value_ids = try trafficRegionOutputValueIds(allocator, instructions, lowering.graph_instruction_ids, diagnostics);
    defer allocator.free(output_value_ids);
    try appendMemoryTrafficRecord(
        allocator,
        records,
        lowering,
        placement.result_memory_space_id,
        .global_memory,
        try trafficValuesBytes(values, input_value_ids, diagnostics),
        try trafficValuesBytes(values, output_value_ids, diagnostics),
        lowering.cost_ledger_ids,
        "external lowering inputs and outputs in global device memory",
    );
}

fn appendTileMemoryTrafficRecords(
    allocator: std.mem.Allocator,
    records: *std.ArrayList(compiler_facts.MemoryTrafficRecord),
    target: target_pkg.TargetDescription,
    cost_ledger: []const compiler_facts.CostLedgerEntry,
    lowering: compiler_facts.LoweringRecord,
    placements: []const compiler_facts.PlacementRecord,
    diagnostics: *std.Io.Writer,
) !void {
    for (target.memory_spaces) |memory_space| {
        switch (memory_space.kind) {
            .local_sram, .scratchpad => {},
            else => continue,
        }
        var bytes_read: u128 = 0;
        var bytes_written: u128 = 0;
        var cost_ids: std.ArrayList(compiler_facts.CostLedgerId) = .empty;
        defer cost_ids.deinit(allocator);
        for (lowering.cost_ledger_ids) |cost_id| {
            const entry = trafficCostEntry(cost_ledger, cost_id, diagnostics) catch |err| return err;
            if (!trafficCostEntryUsesTileMemory(entry, placements, memory_space.id)) continue;
            bytes_read += entry.bytes_read;
            bytes_written += entry.bytes_written;
            try cost_ids.append(allocator, cost_id);
        }
        if (bytes_read == 0 and bytes_written == 0) continue;
        try appendMemoryTrafficRecord(
            allocator,
            records,
            lowering,
            memory_space.id,
            trafficKindForMemorySpace(memory_space.kind),
            bytes_read,
            bytes_written,
            cost_ids.items,
            "lowering cost entries use local tile memory",
        );
    }
}

fn appendMemoryTrafficRecord(
    allocator: std.mem.Allocator,
    records: *std.ArrayList(compiler_facts.MemoryTrafficRecord),
    lowering: compiler_facts.LoweringRecord,
    memory_space_id: u32,
    kind: compiler_facts.MemoryTrafficKind,
    bytes_read: u128,
    bytes_written: u128,
    cost_ids: []const compiler_facts.CostLedgerId,
    reason_text: []const u8,
) !void {
    const instruction_ids = try copyGraphInstructionIdsForLowering(allocator, lowering.graph_instruction_ids);
    var instruction_ids_owned = true;
    errdefer if (instruction_ids_owned) allocator.free(instruction_ids);
    const owned_cost_ids = try copyCostLedgerIds(allocator, cost_ids);
    var cost_ids_owned = true;
    errdefer if (cost_ids_owned) allocator.free(owned_cost_ids);
    const reason = try allocator.dupe(u8, reason_text);
    var reason_owned = true;
    errdefer if (reason_owned) allocator.free(reason);

    try records.append(allocator, .{
        .id = .{ .index = std.math.cast(u32, records.items.len) orelse unreachable },
        .lowering_record_id = lowering.id,
        .memory_space_id = memory_space_id,
        .kind = kind,
        .graph_instruction_ids = instruction_ids,
        .cost_ledger_ids = owned_cost_ids,
        .bytes_read = bytes_read,
        .bytes_written = bytes_written,
        .reason = reason,
    });
    instruction_ids_owned = false;
    cost_ids_owned = false;
    reason_owned = false;
}

fn trafficRegionInputValueIds(
    allocator: std.mem.Allocator,
    instructions: []const compiler_facts.GraphInstruction,
    instruction_ids: []const compiler_facts.GraphInstructionId,
    diagnostics: *std.Io.Writer,
) ![]const compiler_facts.GraphValueId {
    var values: std.ArrayList(compiler_facts.GraphValueId) = .empty;
    errdefer values.deinit(allocator);
    for (instruction_ids) |instruction_id| {
        const instruction = try trafficGraphInstruction(instructions, instruction_id, diagnostics);
        for (instruction.inputs) |input_id| {
            if (try trafficRegionProducesValue(instructions, instruction_ids, input_id, diagnostics)) continue;
            try appendUniqueGraphValueId(allocator, &values, input_id);
        }
    }
    if (values.items.len == 0) {
        try diagnostics.print("pass=memory_traffic_refine feature=graph reason=lowering has no external inputs instruction={d}\n", .{instruction_ids[0].index});
        return MlirStateError.InvalidKernelCodegenPlan;
    }
    return values.toOwnedSlice(allocator);
}

fn trafficRegionOutputValueIds(
    allocator: std.mem.Allocator,
    instructions: []const compiler_facts.GraphInstruction,
    instruction_ids: []const compiler_facts.GraphInstructionId,
    diagnostics: *std.Io.Writer,
) ![]const compiler_facts.GraphValueId {
    var values: std.ArrayList(compiler_facts.GraphValueId) = .empty;
    errdefer values.deinit(allocator);
    for (instruction_ids) |instruction_id| {
        const instruction = try trafficGraphInstruction(instructions, instruction_id, diagnostics);
        for (instruction.outputs) |output_id| {
            if (try trafficRegionConsumesValue(instructions, instruction_ids, output_id, diagnostics)) continue;
            try appendUniqueGraphValueId(allocator, &values, output_id);
        }
    }
    if (values.items.len == 0) {
        try diagnostics.print("pass=memory_traffic_refine feature=graph reason=lowering has no external outputs instruction={d}\n", .{instruction_ids[0].index});
        return MlirStateError.InvalidKernelCodegenPlan;
    }
    return values.toOwnedSlice(allocator);
}

fn trafficRegionProducesValue(
    instructions: []const compiler_facts.GraphInstruction,
    instruction_ids: []const compiler_facts.GraphInstructionId,
    value_id: compiler_facts.GraphValueId,
    diagnostics: *std.Io.Writer,
) !bool {
    for (instruction_ids) |instruction_id| {
        const instruction = try trafficGraphInstruction(instructions, instruction_id, diagnostics);
        for (instruction.outputs) |output_id| {
            if (output_id.eql(value_id)) return true;
        }
    }
    return false;
}

fn trafficRegionConsumesValue(
    instructions: []const compiler_facts.GraphInstruction,
    instruction_ids: []const compiler_facts.GraphInstructionId,
    value_id: compiler_facts.GraphValueId,
    diagnostics: *std.Io.Writer,
) !bool {
    for (instruction_ids) |instruction_id| {
        const instruction = try trafficGraphInstruction(instructions, instruction_id, diagnostics);
        for (instruction.inputs) |input_id| {
            if (input_id.eql(value_id)) return true;
        }
    }
    return false;
}

fn trafficGraphInstruction(
    instructions: []const compiler_facts.GraphInstruction,
    instruction_id: compiler_facts.GraphInstructionId,
    diagnostics: *std.Io.Writer,
) !compiler_facts.GraphInstruction {
    const index: usize = std.math.cast(usize, instruction_id.index) orelse {
        try diagnostics.print("pass=memory_traffic_refine feature=graph reason=invalid instruction id instruction={d}\n", .{instruction_id.index});
        return MlirStateError.InvalidKernelCodegenPlan;
    };
    if (index >= instructions.len or !instructions[index].id.eql(instruction_id)) {
        try diagnostics.print("pass=memory_traffic_refine feature=graph reason=missing instruction instruction={d}\n", .{instruction_id.index});
        return MlirStateError.InvalidKernelCodegenPlan;
    }
    return instructions[index];
}

fn trafficValuesBytes(
    values: []const compiler_facts.GraphValue,
    value_ids: []const compiler_facts.GraphValueId,
    diagnostics: *std.Io.Writer,
) !u128 {
    var total: u128 = 0;
    for (value_ids) |id| {
        const value = try costGraphValue(values, id, diagnostics);
        total += costTensorBytes(value.ty);
    }
    return total;
}

fn trafficCostEntry(
    cost_ledger: []const compiler_facts.CostLedgerEntry,
    cost_id: compiler_facts.CostLedgerId,
    diagnostics: *std.Io.Writer,
) !compiler_facts.CostLedgerEntry {
    const index: usize = std.math.cast(usize, cost_id.index) orelse {
        try diagnostics.print("pass=memory_traffic_refine feature=cost reason=invalid cost id cost={d}\n", .{cost_id.index});
        return MlirStateError.InvalidKernelCodegenPlan;
    };
    if (index >= cost_ledger.len or cost_ledger[index].id.index != cost_id.index) {
        try diagnostics.print("pass=memory_traffic_refine feature=cost reason=missing cost entry cost={d}\n", .{cost_id.index});
        return MlirStateError.InvalidKernelCodegenPlan;
    }
    return cost_ledger[index];
}

fn trafficCostEntryUsesTileMemory(
    entry: compiler_facts.CostLedgerEntry,
    placements: []const compiler_facts.PlacementRecord,
    memory_space_id: u32,
) bool {
    for (entry.graph_instruction_ids) |instruction_id| {
        const placement = placementRecordForInstruction(placements, instruction_id) orelse continue;
        if (placement.tile_memory_space_id == null) continue;
        if (placement.tile_memory_space_id.? == memory_space_id) return true;
    }
    return false;
}

fn copyCostLedgerIds(
    allocator: std.mem.Allocator,
    ids: []const compiler_facts.CostLedgerId,
) ![]const compiler_facts.CostLedgerId {
    const copy = try allocator.alloc(compiler_facts.CostLedgerId, ids.len);
    @memcpy(copy, ids);
    return copy;
}

fn appendUniqueGraphValueId(
    allocator: std.mem.Allocator,
    values: *std.ArrayList(compiler_facts.GraphValueId),
    value_id: compiler_facts.GraphValueId,
) !void {
    for (values.items) |existing| {
        if (existing.eql(value_id)) return;
    }
    try values.append(allocator, value_id);
}

fn trafficKindForMemorySpace(kind: target_pkg.MemorySpaceKind) compiler_facts.MemoryTrafficKind {
    return switch (kind) {
        .host_unpinned, .host_pinned => .host_device_dma,
        .device_unified, .device_hbm, .remote_device => .global_memory,
        .local_sram, .scratchpad => .local_memory,
        .unknown => .global_memory,
    };
}

const CodegenValueFlow = struct {
    shape: core.KernelCodegenShape,
    external_input_ids: []const compiler_facts.GraphValueId,
    external_output_ids: []const compiler_facts.GraphValueId,
    intermediate_value_ids: []const compiler_facts.GraphValueId,
};

fn codegenValueFlow(
    allocator: std.mem.Allocator,
    instructions: []const compiler_facts.GraphInstruction,
    instruction_ids: []const compiler_facts.GraphInstructionId,
    diagnostics: *std.Io.Writer,
) !CodegenValueFlow {
    var external_inputs: std.ArrayList(compiler_facts.GraphValueId) = .empty;
    errdefer external_inputs.deinit(allocator);
    var external_outputs: std.ArrayList(compiler_facts.GraphValueId) = .empty;
    errdefer external_outputs.deinit(allocator);
    var intermediates: std.ArrayList(compiler_facts.GraphValueId) = .empty;
    errdefer intermediates.deinit(allocator);

    for (instruction_ids) |instruction_id| {
        const instruction = try trafficGraphInstruction(instructions, instruction_id, diagnostics);
        for (instruction.inputs) |input_id| {
            if (!try trafficRegionProducesValue(instructions, instruction_ids, input_id, diagnostics)) {
                try appendUniqueGraphValueId(allocator, &external_inputs, input_id);
            }
        }
        for (instruction.outputs) |output_id| {
            if (codegenValueEscapesRegion(instructions, instruction_ids, output_id)) {
                try appendUniqueGraphValueId(allocator, &external_outputs, output_id);
            } else {
                try appendUniqueGraphValueId(allocator, &intermediates, output_id);
            }
        }
    }

    const external_input_ids = try external_inputs.toOwnedSlice(allocator);
    errdefer allocator.free(external_input_ids);
    const external_output_ids = try external_outputs.toOwnedSlice(allocator);
    errdefer allocator.free(external_output_ids);
    const intermediate_value_ids = try intermediates.toOwnedSlice(allocator);
    errdefer allocator.free(intermediate_value_ids);

    return .{
        .shape = .{
            .operation_count = std.math.cast(u32, instruction_ids.len) orelse unreachable,
            .external_input_count = std.math.cast(u32, external_input_ids.len) orelse unreachable,
            .external_output_count = std.math.cast(u32, external_output_ids.len) orelse unreachable,
            .intermediate_value_count = std.math.cast(u32, intermediate_value_ids.len) orelse unreachable,
        },
        .external_input_ids = external_input_ids,
        .external_output_ids = external_output_ids,
        .intermediate_value_ids = intermediate_value_ids,
    };
}

fn deinitCodegenValueFlow(allocator: std.mem.Allocator, value_flow: CodegenValueFlow) void {
    allocator.free(value_flow.external_input_ids);
    allocator.free(value_flow.external_output_ids);
    allocator.free(value_flow.intermediate_value_ids);
}

fn codegenValueEscapesRegion(
    instructions: []const compiler_facts.GraphInstruction,
    instruction_ids: []const compiler_facts.GraphInstructionId,
    value_id: compiler_facts.GraphValueId,
) bool {
    for (instructions) |instruction| {
        if (graphInstructionIdInSlice(instruction_ids, instruction.id)) continue;
        for (instruction.inputs) |input_id| {
            if (input_id.eql(value_id)) return true;
        }
    }
    return false;
}

fn copyI64sForCodegen(allocator: std.mem.Allocator, values: []const i64) ![]const i64 {
    const copy = try allocator.alloc(i64, values.len);
    @memcpy(copy, values);
    return copy;
}

fn codegenMemoryTrafficIdsForLowering(
    allocator: std.mem.Allocator,
    memory_traffic: []const compiler_facts.MemoryTrafficRecord,
    lowering_id: compiler_facts.LoweringRecordId,
) ![]const compiler_facts.MemoryTrafficId {
    var ids: std.ArrayList(compiler_facts.MemoryTrafficId) = .empty;
    errdefer ids.deinit(allocator);
    for (memory_traffic) |record| {
        if (record.lowering_record_id.eql(lowering_id)) try ids.append(allocator, record.id);
    }
    return ids.toOwnedSlice(allocator);
}

fn codegenMemoryPressure(
    memory_traffic: []const compiler_facts.MemoryTrafficRecord,
    memory_traffic_ids: []const compiler_facts.MemoryTrafficId,
) core.KernelMemoryPressure {
    var pressure: core.KernelMemoryPressure = .{
        .global_bytes_read = 0,
        .global_bytes_written = 0,
        .local_bytes_read = 0,
        .local_bytes_written = 0,
    };
    for (memory_traffic_ids) |traffic_id| {
        const record_index: usize = std.math.cast(usize, traffic_id.index) orelse unreachable;
        const record = memory_traffic[record_index];
        switch (record.kind) {
            .global_memory, .host_device_dma, .interconnect => {
                pressure.global_bytes_read += record.bytes_read;
                pressure.global_bytes_written += record.bytes_written;
            },
            .local_memory => {
                pressure.local_bytes_read += record.bytes_read;
                pressure.local_bytes_written += record.bytes_written;
            },
        }
    }
    return pressure;
}

fn codegenKindForLowering(decision: compiler_facts.LoweringDecision) core.KernelCodegenKind {
    return switch (decision) {
        .backend_kernel_graph => .backend_kernel_graph,
        .elementwise_fusion => .elementwise_fusion_kernel,
        .transfer => .library_call,
        .unsupported => .library_call,
    };
}

fn codegenOperationForLowering(
    lowering: compiler_facts.LoweringRecord,
    capabilities: []const KernelCodegenCapabilityFact,
    graph_execute_operation: []const u8,
    fused_elementwise_operation: []const u8,
) []const u8 {
    if (lowering.decision == .elementwise_fusion) return fused_elementwise_operation;
    if (lowering.graph_instruction_ids.len == 1) {
        if (codegenCapabilityForInstruction(capabilities, lowering.graph_instruction_ids[0])) |capability| {
            return capability.backend_operation;
        }
    }
    return graph_execute_operation;
}

fn codegenExpectedUnitForLowering(
    lowering: compiler_facts.LoweringRecord,
    capabilities: []const KernelCodegenCapabilityFact,
) ?u32 {
    var selected_unit: ?u32 = null;
    var saw_unit = false;
    for (lowering.graph_instruction_ids) |instruction_id| {
        const capability = codegenCapabilityForInstruction(capabilities, instruction_id) orelse return null;
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

fn codegenCapabilityForInstruction(
    capabilities: []const KernelCodegenCapabilityFact,
    instruction_id: compiler_facts.GraphInstructionId,
) ?KernelCodegenCapabilityFact {
    for (capabilities) |capability| {
        if (capability.graph_instruction_id.eql(instruction_id)) return capability;
    }
    return null;
}

fn codegenReasonForLowering(decision: compiler_facts.LoweringDecision) []const u8 {
    return switch (decision) {
        .backend_kernel_graph => "kernel codegen plan keeps the backend kernel graph boundary explicit before backend binding",
        .elementwise_fusion => "kernel codegen plan preserves the fused elementwise lowering region as one generated-kernel candidate",
        .transfer => "transfer lowering is represented as a backend library or DMA call candidate",
        .unsupported => "unsupported lowering cannot create an executable kernel, but remains printable for failed compile reports",
    };
}

fn appendScheduleCommand(
    allocator: std.mem.Allocator,
    commands: *std.ArrayList(core.ScheduleCommand),
    kind: core.CommandKind,
    stream: core.StreamId,
    inputs: []const compiler_facts.GraphValueId,
    outputs: []const compiler_facts.GraphValueId,
    dependencies: []const core.CommandDependency,
    lowerings: []const compiler_facts.LoweringRecordId,
    costs: []const compiler_facts.CostLedgerId,
) !void {
    const owned_inputs = try copyGraphValueIds(allocator, inputs);
    var inputs_owned = true;
    errdefer if (inputs_owned) allocator.free(owned_inputs);
    const owned_outputs = try copyGraphValueIds(allocator, outputs);
    var outputs_owned = true;
    errdefer if (outputs_owned) allocator.free(owned_outputs);
    const owned_dependencies = try copyCommandDependencies(allocator, dependencies);
    var dependencies_owned = true;
    errdefer if (dependencies_owned) allocator.free(owned_dependencies);
    const owned_lowerings = try copyLoweringRecordIds(allocator, lowerings);
    var lowerings_owned = true;
    errdefer if (lowerings_owned) allocator.free(owned_lowerings);
    const owned_costs = try copyCostLedgerIds(allocator, costs);
    var costs_owned = true;
    errdefer if (costs_owned) allocator.free(owned_costs);

    try commands.append(allocator, .{
        .id = .{ .index = std.math.cast(u32, commands.items.len) orelse unreachable },
        .kind = kind,
        .stream = stream,
        .inputs = owned_inputs,
        .outputs = owned_outputs,
        .dependencies = owned_dependencies,
        .lowering_record_ids = owned_lowerings,
        .cost_ledger_ids = owned_costs,
    });
    inputs_owned = false;
    outputs_owned = false;
    dependencies_owned = false;
    lowerings_owned = false;
    costs_owned = false;
}

fn deriveScheduleOverlapRecords(
    allocator: std.mem.Allocator,
    commands: []const core.ScheduleCommand,
    diagnostics: *std.Io.Writer,
) ![]core.ScheduleOverlapRecord {
    var overlaps: std.ArrayList(core.ScheduleOverlapRecord) = .empty;
    errdefer {
        for (overlaps.items) |overlap| allocator.free(overlap.reason);
        overlaps.deinit(allocator);
    }

    for (commands) |command| {
        for (command.dependencies) |dependency| {
            const first = scheduleCommandById(commands, dependency.command_id, diagnostics) catch |err| return err;
            const kind = scheduleOverlapKindForCommands(first.kind, command.kind) orelse continue;
            const reason = try allocator.dupe(u8, scheduleOverlapReason(first.kind, command.kind, dependency.kind));
            var reason_owned = true;
            errdefer if (reason_owned) allocator.free(reason);
            try overlaps.append(allocator, .{
                .id = .{ .index = std.math.cast(u32, overlaps.items.len) orelse unreachable },
                .decision = .serialized,
                .kind = kind,
                .first_command_id = first.id,
                .second_command_id = command.id,
                .dependency_kind = dependency.kind,
                .first_stream = first.stream,
                .second_stream = command.stream,
                .reason = reason,
            });
            reason_owned = false;
        }
    }

    const owned_overlaps = try overlaps.toOwnedSlice(allocator);
    errdefer deinitExtractedScheduleOverlapRecords(allocator, owned_overlaps);
    try verifyScheduleOverlaps(owned_overlaps, commands.len, diagnostics);
    return owned_overlaps;
}

fn scheduleCommandById(
    commands: []const core.ScheduleCommand,
    command_id: core.ScheduleCommandId,
    diagnostics: *std.Io.Writer,
) !core.ScheduleCommand {
    const index: usize = std.math.cast(usize, command_id.index) orelse {
        try diagnostics.print("pass=schedule_plan feature=command reason=invalid command id command={d}\n", .{command_id.index});
        return MlirStateError.InvalidSchedulePlan;
    };
    if (index >= commands.len or commands[index].id.index != command_id.index) {
        try diagnostics.print("pass=schedule_plan feature=command reason=missing command command={d}\n", .{command_id.index});
        return MlirStateError.InvalidSchedulePlan;
    }
    return commands[index];
}

fn scheduleAllLoweringIds(
    allocator: std.mem.Allocator,
    lowerings: []const compiler_facts.LoweringRecord,
) ![]const compiler_facts.LoweringRecordId {
    const ids = try allocator.alloc(compiler_facts.LoweringRecordId, lowerings.len);
    for (ids, 0..) |*id, index| id.* = .{ .index = std.math.cast(u32, index) orelse unreachable };
    return ids;
}

fn scheduleAllCostIds(
    allocator: std.mem.Allocator,
    cost_ledger: []const compiler_facts.CostLedgerEntry,
) ![]const compiler_facts.CostLedgerId {
    const ids = try allocator.alloc(compiler_facts.CostLedgerId, cost_ledger.len);
    for (ids, 0..) |*id, index| id.* = .{ .index = std.math.cast(u32, index) orelse unreachable };
    return ids;
}

fn copyGraphValueIds(
    allocator: std.mem.Allocator,
    ids: []const compiler_facts.GraphValueId,
) ![]const compiler_facts.GraphValueId {
    const copy = try allocator.alloc(compiler_facts.GraphValueId, ids.len);
    @memcpy(copy, ids);
    return copy;
}

fn copyLoweringRecordIds(
    allocator: std.mem.Allocator,
    ids: []const compiler_facts.LoweringRecordId,
) ![]const compiler_facts.LoweringRecordId {
    const copy = try allocator.alloc(compiler_facts.LoweringRecordId, ids.len);
    @memcpy(copy, ids);
    return copy;
}

fn copyCommandDependencies(
    allocator: std.mem.Allocator,
    dependencies: []const core.CommandDependency,
) ![]const core.CommandDependency {
    const copy = try allocator.alloc(core.CommandDependency, dependencies.len);
    @memcpy(copy, dependencies);
    return copy;
}

fn scheduleOverlapKindForCommands(first: core.CommandKind, second: core.CommandKind) ?core.ScheduleOverlapKind {
    return switch (first) {
        .host_to_device => switch (second) {
            .backend_execute => .transfer_compute,
            .device_to_host => .transfer_transfer,
            else => null,
        },
        .backend_execute => switch (second) {
            .device_to_host => .compute_transfer,
            else => null,
        },
        .device_to_host => switch (second) {
            .host_to_device => .transfer_transfer,
            else => null,
        },
        .event_record, .event_wait => null,
    };
}

fn scheduleOverlapReason(first: core.CommandKind, second: core.CommandKind, dependency: core.DependencyKind) []const u8 {
    if (dependency == .data) {
        return switch (first) {
            .host_to_device => switch (second) {
                .backend_execute => "backend execution consumes host-to-device inputs, so V0 serializes this transfer/compute edge",
                .device_to_host => "device-to-host transfer depends on data made available by the prior transfer edge",
                else => "data dependency prevents overlap in V0",
            },
            .backend_execute => switch (second) {
                .device_to_host => "device-to-host transfer consumes backend outputs, so V0 serializes this compute/transfer edge",
                else => "data dependency prevents overlap in V0",
            },
            else => "data dependency prevents overlap in V0",
        };
    }
    return "V0 records the overlap edge but keeps command order conservative";
}

fn duplicateLoweringAlternatives(
    allocator: std.mem.Allocator,
    alternatives: []const []const u8,
) ![]const []const u8 {
    const copy = try allocator.alloc([]const u8, alternatives.len);
    var initialized: usize = 0;
    errdefer {
        for (copy[0..initialized]) |alternative| allocator.free(alternative);
        allocator.free(copy);
    }
    for (alternatives, 0..) |alternative, index| {
        copy[index] = try allocator.dupe(u8, alternative);
        initialized += 1;
    }
    return copy;
}

fn loweringDecisionForCostLedger(cost_ledger: []const compiler_facts.CostLedgerEntry, cost_id: compiler_facts.CostLedgerId) compiler_facts.LoweringDecision {
    const index: usize = std.math.cast(usize, cost_id.index) orelse return .unsupported;
    if (index >= cost_ledger.len) return .unsupported;
    return switch (cost_ledger[index].op_class) {
        .matmul, .backend_kernel => .backend_kernel_graph,
        .elementwise, .transcendental => .elementwise_fusion,
        .transfer => .transfer,
    };
}

fn deinitCostLedgerEntryFields(allocator: std.mem.Allocator, entry: compiler_facts.CostLedgerEntry) void {
    if (entry.source) |source| {
        allocator.free(source.op_name);
        allocator.free(source.location);
    }
    allocator.free(entry.graph_instruction_ids);
    allocator.free(entry.formula);
    allocator.free(entry.approximation);
}

fn deinitMemoryTrafficRecordFields(allocator: std.mem.Allocator, record: compiler_facts.MemoryTrafficRecord) void {
    allocator.free(record.graph_instruction_ids);
    allocator.free(record.cost_ledger_ids);
    allocator.free(record.reason);
}

fn deinitScheduleCommandFields(allocator: std.mem.Allocator, command: core.ScheduleCommand) void {
    allocator.free(command.inputs);
    allocator.free(command.outputs);
    allocator.free(command.dependencies);
    allocator.free(command.lowering_record_ids);
    allocator.free(command.cost_ledger_ids);
}

fn deinitBackendBindingFields(allocator: std.mem.Allocator, binding: core.BackendBinding, strings_owned: bool) void {
    if (strings_owned) allocator.free(binding.backend_operation);
    allocator.free(binding.graph_instruction_ids);
    allocator.free(binding.cost_ledger_ids);
}

fn deinitBackendExecutableCallFields(allocator: std.mem.Allocator, call: BackendExecutableCallFact) void {
    allocator.free(call.graph_instruction_ids);
    allocator.free(call.feature);
    allocator.free(call.backend_operation);
    allocator.free(call.input_value_ids);
    allocator.free(call.output_value_ids);
}

fn deinitBackendKernelGraphNodeFields(allocator: std.mem.Allocator, node: BackendKernelGraphNodeFact) void {
    allocator.free(node.graph_instruction_ids);
    allocator.free(node.feature);
    allocator.free(node.backend_operation);
    allocator.free(node.input_value_ids);
    allocator.free(node.output_value_ids);
    allocator.free(node.output_type.dims);
    allocator.free(node.attributes);
}

fn deinitRuntimeAllocationFields(allocator: std.mem.Allocator, allocation: RuntimeAllocationFact) void {
    allocator.free(allocation.placement);
}

fn deinitRuntimeBufferUseFields(allocator: std.mem.Allocator, use: RuntimeBufferUseFact) void {
    allocator.free(use.access);
}

fn deinitRuntimeStreamStepFields(allocator: std.mem.Allocator, step: RuntimeStreamStepFact) void {
    allocator.free(step.wait_event_ids);
}

fn deinitRuntimeProfileEventFields(allocator: std.mem.Allocator, event: RuntimeProfileEventFact) void {
    allocator.free(event.graph_instruction_ids);
    allocator.free(event.kind);
    allocator.free(event.status);
}

fn deinitRuntimeProfileJoinFields(allocator: std.mem.Allocator, join: RuntimeProfileJoinFact) void {
    allocator.free(join.subject_kind);
    allocator.free(join.graph_instruction_ids);
    allocator.free(join.profile_event_ids);
}

fn deinitBackendProfileJoinFields(allocator: std.mem.Allocator, join: BackendProfileJoinFact) void {
    allocator.free(join.graph_instruction_ids);
}

const FactProfileMetrics = struct {
    bytes: u128,
    logical_ops: u128,
};

fn writeFactInstructionIdList(writer: *std.Io.Writer, ids: []const compiler_facts.GraphInstructionId) std.Io.Writer.Error!void {
    for (ids, 0..) |id, index| {
        if (index != 0) try writer.writeAll(",");
        try writer.print("{d}", .{id.index});
    }
}

fn runtimeProfileEventForCommand(events: []const RuntimeProfileEventFact, command_id: core.ScheduleCommandId) ?RuntimeProfileEventFact {
    for (events) |event| {
        if (event.command_id) |event_command_id| {
            if (event_command_id.eql(command_id)) return event;
        }
    }
    return null;
}

fn runtimeProfileEventForLowering(
    events: []const RuntimeProfileEventFact,
    command_id: core.ScheduleCommandId,
    instruction_ids: []const compiler_facts.GraphInstructionId,
) ?RuntimeProfileEventFact {
    for (events) |event| {
        if (event.command_id == null or !event.command_id.?.eql(command_id)) continue;
        if (graphInstructionIdsEqualFact(event.graph_instruction_ids, instruction_ids)) return event;
    }
    return null;
}

fn runtimeProfileEventForLoweringRecord(
    report: core.TraceReport,
    events: []const RuntimeProfileEventFact,
    lowering: compiler_facts.LoweringRecord,
) ?RuntimeProfileEventFact {
    for (report.schedule_commands) |command| {
        if (command.kind != .backend_execute) continue;
        if (!loweringIdInFact(lowering.id, command.lowering_record_ids)) continue;
        return runtimeProfileEventForLowering(events, command.id, lowering.graph_instruction_ids);
    }
    return null;
}

fn loweringIdInFact(needle: compiler_facts.LoweringRecordId, haystack: []const compiler_facts.LoweringRecordId) bool {
    for (haystack) |id| {
        if (id.eql(needle)) return true;
    }
    return false;
}

fn streamStepForCommandFact(steps: []const RuntimeStreamStepFact, command_id: core.ScheduleCommandId) RuntimeStreamStepFact {
    for (steps) |step| {
        if (step.command_id.eql(command_id)) return step;
    }
    unreachable;
}

fn profileMetricsForCommandFact(report: core.TraceReport, cost_ledger: []const compiler_facts.CostLedgerEntry, command: core.ScheduleCommand) FactProfileMetrics {
    return switch (command.kind) {
        .host_to_device, .device_to_host => .{
            .bytes = valueBytesFact(report, if (command.kind == .host_to_device) command.outputs else command.inputs),
            .logical_ops = 0,
        },
        .backend_execute => costMetricsFact(cost_ledger, command.cost_ledger_ids),
        .event_record, .event_wait => .{ .bytes = 0, .logical_ops = 0 },
    };
}

fn callProfileMetricsFact(cost_ledger: []const compiler_facts.CostLedgerEntry, instruction_ids: []const compiler_facts.GraphInstructionId) FactProfileMetrics {
    var metrics: FactProfileMetrics = .{ .bytes = 0, .logical_ops = 0 };
    for (cost_ledger) |entry| {
        if (!allInstructionIdsInFact(entry.graph_instruction_ids, instruction_ids)) continue;
        metrics.bytes += entry.bytes_read + entry.bytes_written;
        metrics.logical_ops += entry.logical_ops;
    }
    return metrics;
}

fn costMetricsFact(cost_ledger: []const compiler_facts.CostLedgerEntry, ids: []const compiler_facts.CostLedgerId) FactProfileMetrics {
    var metrics: FactProfileMetrics = .{ .bytes = 0, .logical_ops = 0 };
    for (ids) |id| {
        const index: usize = std.math.cast(usize, id.index) orelse unreachable;
        const entry = cost_ledger[index];
        metrics.bytes += entry.bytes_read + entry.bytes_written;
        metrics.logical_ops += entry.logical_ops;
    }
    return metrics;
}

fn valueBytesFact(report: core.TraceReport, ids: []const compiler_facts.GraphValueId) u128 {
    var total: u128 = 0;
    for (ids) |id| {
        const index: usize = std.math.cast(usize, id.index) orelse unreachable;
        total += tensorBytesFact(report.graph_values[index].ty);
    }
    return total;
}

fn tensorBytesFact(ty: compiler_facts.TensorType) u128 {
    const element_size = ty.element_type.byteSize() orelse 0;
    var elements: u128 = 1;
    for (ty.dims) |dim| {
        elements *= std.math.cast(u128, dim) orelse unreachable;
    }
    return elements * element_size;
}

fn idealTransferPsForCommandFact(
    target: target_pkg.TargetDescription,
    allocation_plan: RuntimeAllocationPlanFact,
    command: core.ScheduleCommand,
    bytes: u128,
) u128 {
    const edge = transferEdgeForCommandFact(target, allocation_plan, command) orelse return 0;
    return idealTransferPsFact(bytes, edge.bandwidth_bytes_per_second);
}

fn transferBytesForEdgeFact(
    report: core.TraceReport,
    target: target_pkg.TargetDescription,
    cost_ledger: []const compiler_facts.CostLedgerEntry,
    schedule_commands: []const core.ScheduleCommand,
    allocation_plan: RuntimeAllocationPlanFact,
    edge: target_pkg.TargetTransferEdge,
) u128 {
    var total: u128 = 0;
    for (schedule_commands) |command| {
        const command_edge = transferEdgeForCommandFact(target, allocation_plan, command) orelse continue;
        if (command_edge.id == edge.id) total += profileMetricsForCommandFact(report, cost_ledger, command).bytes;
    }
    return total;
}

fn transferEdgeForCommandFact(
    target: target_pkg.TargetDescription,
    allocation_plan: RuntimeAllocationPlanFact,
    command: core.ScheduleCommand,
) ?target_pkg.TargetTransferEdge {
    const memory_pair = transferMemoryPairForCommandFact(allocation_plan, command) orelse return null;
    return transferEdgeForMemoryPairFact(target, memory_pair);
}

fn transferEdgeForMemoryPairFact(
    target: target_pkg.TargetDescription,
    memory_pair: TransferMemoryPairFact,
) ?target_pkg.TargetTransferEdge {
    for (target.transfer_edges) |edge| {
        if (edge.src_memory_space == memory_pair.src and edge.dst_memory_space == memory_pair.dst) return edge;
    }
    return null;
}

const TransferMemoryPairFact = struct {
    src: u32,
    dst: u32,
};

fn transferMemoryPairForCommandFact(allocation_plan: RuntimeAllocationPlanFact, command: core.ScheduleCommand) ?TransferMemoryPairFact {
    return switch (command.kind) {
        .host_to_device => transferMemoryPairFromValuesFact(allocation_plan, command.inputs, "host", command.outputs, "device"),
        .device_to_host => transferMemoryPairFromValuesFact(allocation_plan, command.inputs, "device", command.outputs, "host"),
        .backend_execute, .event_record, .event_wait => null,
    };
}

fn transferMemoryPairFromValuesFact(
    allocation_plan: RuntimeAllocationPlanFact,
    src_values: []const compiler_facts.GraphValueId,
    src_placement: []const u8,
    dst_values: []const compiler_facts.GraphValueId,
    dst_placement: []const u8,
) ?TransferMemoryPairFact {
    if (src_values.len == 0 or dst_values.len == 0) return null;
    const src = memorySpaceForValuePlacementFact(allocation_plan.allocations, src_values[0], src_placement) orelse return null;
    const dst = memorySpaceForValuePlacementFact(allocation_plan.allocations, dst_values[0], dst_placement) orelse return null;
    return .{ .src = src, .dst = dst };
}

fn memorySpaceForValuePlacementFact(
    allocations: []const RuntimeAllocationFact,
    value_id: compiler_facts.GraphValueId,
    placement: []const u8,
) ?u32 {
    for (allocations) |allocation| {
        if (allocation.value_id.eql(value_id) and std.mem.eql(u8, allocation.placement, placement)) return allocation.memory_space_id;
    }
    return null;
}

fn peakLiveBytesForMemoryFact(plan: RuntimeAllocationPlanFact, memory_space_id: u32) u128 {
    var peak: u128 = 0;
    var command_index: u32 = 0;
    while (command_index <= maxLifetimeCommandFact(plan.allocations)) : (command_index += 1) {
        var live: u128 = 0;
        for (plan.allocations) |allocation| {
            if (command_index < allocation.first_command_id.index or command_index > allocation.last_command_id.index) continue;
            if (allocation.memory_space_id == memory_space_id) live += allocation.size_bytes;
        }
        peak = @max(peak, live);
    }
    return peak;
}

fn maxLifetimeCommandFact(allocations: []const RuntimeAllocationFact) u32 {
    var max_index: u32 = 0;
    for (allocations) |allocation| {
        max_index = @max(max_index, allocation.last_command_id.index);
    }
    return max_index;
}

fn allocatedBytesForMemoryFact(allocations: []const RuntimeAllocationFact, memory_space_id: u32) u128 {
    var total: u128 = 0;
    for (allocations) |allocation| {
        if (allocation.memory_space_id == memory_space_id) total += allocation.size_bytes;
    }
    return total;
}

fn memorySpaceByIdFact(memory_spaces: []const target_pkg.TargetMemorySpace, id: u32) ?target_pkg.TargetMemorySpace {
    for (memory_spaces) |memory_space| {
        if (memory_space.id == id) return memory_space;
    }
    return null;
}

fn costMetricsForUnitFact(cost_ledger: []const compiler_facts.CostLedgerEntry, unit_id: u32) FactProfileMetrics {
    var metrics: FactProfileMetrics = .{ .bytes = 0, .logical_ops = 0 };
    for (cost_ledger) |entry| {
        if (entry.expected_unit_id) |expected_unit_id| {
            if (expected_unit_id == unit_id) {
                metrics.bytes += entry.bytes_read + entry.bytes_written;
                metrics.logical_ops += entry.logical_ops;
            }
        }
    }
    return metrics;
}

fn costMetricsForUnitRateFact(cost_ledger: []const compiler_facts.CostLedgerEntry, unit_id: u32, rate: target_pkg.DTypeRate) FactProfileMetrics {
    var metrics: FactProfileMetrics = .{ .bytes = 0, .logical_ops = 0 };
    for (cost_ledger) |entry| {
        if (entry.expected_unit_id) |expected_unit_id| {
            if (expected_unit_id == unit_id and entry.dtype == rate.dtype and costOpClassMatchesRateFact(entry.op_class, rate.op_class)) {
                metrics.bytes += entry.bytes_read + entry.bytes_written;
                metrics.logical_ops += entry.logical_ops;
            }
        }
    }
    return metrics;
}

fn costMetricsForUnknownUnitFact(cost_ledger: []const compiler_facts.CostLedgerEntry) FactProfileMetrics {
    var metrics: FactProfileMetrics = .{ .bytes = 0, .logical_ops = 0 };
    for (cost_ledger) |entry| {
        if (entry.expected_unit_id == null) {
            metrics.bytes += entry.bytes_read + entry.bytes_written;
            metrics.logical_ops += entry.logical_ops;
        }
    }
    return metrics;
}

fn idealComputePsForLoweringFact(target: target_pkg.TargetDescription, cost_ledger: []const compiler_facts.CostLedgerEntry, lowering: compiler_facts.LoweringRecord) u128 {
    var total: u128 = 0;
    for (lowering.cost_ledger_ids) |cost_id| {
        const index: usize = std.math.cast(usize, cost_id.index) orelse unreachable;
        total += idealComputePsForCostEntryFact(target, cost_ledger[index]);
    }
    return total;
}

fn idealComputePsForCallFact(target: target_pkg.TargetDescription, cost_ledger: []const compiler_facts.CostLedgerEntry, instruction_ids: []const compiler_facts.GraphInstructionId) u128 {
    var total: u128 = 0;
    for (cost_ledger) |entry| {
        if (!allInstructionIdsInFact(entry.graph_instruction_ids, instruction_ids)) continue;
        total += idealComputePsForCostEntryFact(target, entry);
    }
    return total;
}

fn idealComputePsForCostEntryFact(target: target_pkg.TargetDescription, entry: compiler_facts.CostLedgerEntry) u128 {
    const unit_id = entry.expected_unit_id orelse return 0;
    const unit = executionUnitByIdFact(target.execution_units, unit_id) orelse return 0;
    for (unit.dtype_rates) |rate| {
        if (entry.dtype == rate.dtype and costOpClassMatchesRateFact(entry.op_class, rate.op_class)) {
            return idealComputePsFact(entry.logical_ops, rate.ops_per_second);
        }
    }
    return 0;
}

fn idealMemoryPsForLoweringFact(
    target: target_pkg.TargetDescription,
    traffic_records: []const compiler_facts.MemoryTrafficRecord,
    lowering_id: compiler_facts.LoweringRecordId,
) u128 {
    var max_ps: u128 = 0;
    for (traffic_records) |record| {
        if (!record.lowering_record_id.eql(lowering_id)) continue;
        const memory_space = memorySpaceByIdFact(target.memory_spaces, record.memory_space_id) orelse continue;
        const total_bytes = record.bytes_read + record.bytes_written;
        const ps = idealTransferPsFact(total_bytes, memory_space.bandwidth_bytes_per_second);
        if (ps > max_ps) max_ps = ps;
    }
    return max_ps;
}

fn idealMemoryPsForCallFact(target: target_pkg.TargetDescription, memory_traffic_records: []const compiler_facts.MemoryTrafficRecord, call: BackendExecutableCallFact) u128 {
    var max_ps: u128 = 0;
    for (target.memory_spaces) |memory_space| {
        const bytes = memoryTrafficBytesForCallFact(memory_traffic_records, call, memory_space.id);
        const ps = idealTransferPsFact(bytes, memory_space.bandwidth_bytes_per_second);
        if (ps > max_ps) max_ps = ps;
    }
    return max_ps;
}

fn writeMemoryTrafficListFact(
    writer: *std.Io.Writer,
    target: target_pkg.TargetDescription,
    memory_traffic_records: []const compiler_facts.MemoryTrafficRecord,
    call: BackendExecutableCallFact,
) std.Io.Writer.Error!void {
    var wrote = false;
    for (target.memory_spaces) |memory_space| {
        const bytes = memoryTrafficBytesForCallFact(memory_traffic_records, call, memory_space.id);
        if (bytes == 0) continue;
        if (wrote) try writer.writeAll(",");
        try writer.print(
            "memory.{d}:{d}B/{d}ps",
            .{ memory_space.id, bytes, idealTransferPsFact(bytes, memory_space.bandwidth_bytes_per_second) },
        );
        wrote = true;
    }
    if (!wrote) try writer.writeAll("none");
}

fn memoryTrafficBytesForCallFact(memory_traffic_records: []const compiler_facts.MemoryTrafficRecord, call: BackendExecutableCallFact, memory_space_id: u32) u128 {
    var total: u128 = 0;
    for (memory_traffic_records) |record| {
        if (record.memory_space_id != memory_space_id) continue;
        if (!graphInstructionIdsEqualFact(record.graph_instruction_ids, call.graph_instruction_ids)) continue;
        total += record.bytes_read + record.bytes_written;
    }
    return total;
}

fn runtimeProfileEventForBackendCallJoin(
    profile_events: []const RuntimeProfileEventFact,
    joins: []const BackendProfileJoinFact,
    call_index: u32,
) ?RuntimeProfileEventFact {
    for (joins) |join| {
        if (join.call_index != call_index) continue;
        return runtimeProfileEventByIdFact(profile_events, join.profile_event_id);
    }
    return null;
}

fn runtimeProfileEventByIdFact(profile_events: []const RuntimeProfileEventFact, id: core.ProfileEventId) ?RuntimeProfileEventFact {
    for (profile_events) |event| {
        if (event.index == id.index) return event;
    }
    return null;
}

fn executionUnitByIdFact(units: []const target_pkg.ExecutionUnit, unit_id: u32) ?target_pkg.ExecutionUnit {
    for (units) |unit| {
        if (unit.id == unit_id) return unit;
    }
    return null;
}

fn costOpClassMatchesRateFact(cost_class: compiler_facts.CostOpClass, rate_class: core.OpClass) bool {
    return switch (cost_class) {
        .matmul => rate_class == .matmul,
        .elementwise => rate_class == .elementwise,
        .transcendental => rate_class == .transcendental,
        .transfer => rate_class == .memory,
        .backend_kernel => false,
    };
}

fn limitingResourceFact(ideal_compute_ps: u128, ideal_memory_ps: u128) []const u8 {
    if (ideal_compute_ps == 0 and ideal_memory_ps == 0) return "unknown";
    if (ideal_compute_ps == ideal_memory_ps) return "balanced";
    if (ideal_compute_ps > ideal_memory_ps) return "compute";
    return "memory";
}

fn writeOptionalF64Fact(writer: *std.Io.Writer, value: ?f64) std.Io.Writer.Error!void {
    if (value) |known| {
        try writer.print("{d}", .{known});
    } else {
        try writer.writeAll("unknown");
    }
}

fn idealComputePsFact(logical_ops: u128, ops_per_second: ?f64) u128 {
    if (logical_ops == 0) return 0;
    const peak = ops_per_second orelse return 0;
    if (peak <= 0) return 0;
    const ops: f64 = std.math.lossyCast(f64, logical_ops);
    const ideal = @ceil((ops * 1_000_000_000_000.0) / peak);
    return std.math.lossyCast(u128, ideal);
}

fn idealTransferPsFact(bytes: u128, bytes_per_second: ?f64) u128 {
    if (bytes == 0) return 0;
    const bandwidth = bytes_per_second orelse return 0;
    if (bandwidth <= 0) return 0;
    const byte_count: f64 = std.math.lossyCast(f64, bytes);
    const ideal = @ceil((byte_count * 1_000_000_000_000.0) / bandwidth);
    return std.math.lossyCast(u128, ideal);
}

fn loweringRecordFact(report: core.TraceReport, id: compiler_facts.LoweringRecordId) compiler_facts.LoweringRecord {
    const index: usize = std.math.cast(usize, id.index) orelse unreachable;
    return report.lowering_records[index];
}

fn allInstructionIdsInFact(needles: []const compiler_facts.GraphInstructionId, haystack: []const compiler_facts.GraphInstructionId) bool {
    for (needles) |needle| {
        if (!instructionIdInFact(needle, haystack)) return false;
    }
    return true;
}

fn instructionIdInFact(needle: compiler_facts.GraphInstructionId, haystack: []const compiler_facts.GraphInstructionId) bool {
    for (haystack) |id| {
        if (id.eql(needle)) return true;
    }
    return false;
}

fn graphInstructionIdsEqualFact(left: []const compiler_facts.GraphInstructionId, right: []const compiler_facts.GraphInstructionId) bool {
    if (left.len != right.len) return false;
    for (left, right) |left_id, right_id| {
        if (!left_id.eql(right_id)) return false;
    }
    return true;
}

pub fn writeModuleSnapshot(session: *const MlirSession, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    const SnapshotContext = struct {
        writer: *std.Io.Writer,
        failed: bool = false,

        fn callback(text: mlir.MlirStringRef, user_data: ?*anyopaque) callconv(.c) void {
            const self: *@This() = @ptrCast(@alignCast(user_data.?));
            self.writer.writeAll(mlirStringSlice(text)) catch {
                self.failed = true;
            };
        }
    };

    var context: SnapshotContext = .{ .writer = writer };
    mlir.mlirOperationPrint(session.moduleOperation(), SnapshotContext.callback, &context);
    if (context.failed) return std.Io.Writer.Error.WriteFailed;
}

pub fn writeStateSummary(session: *const MlirSession, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    try writer.writeAll("mlir_state_summary\n");
    const state_text = if (moduleState(session)) |state| state.text() else "unknown";
    try writer.print("  state={s}\n", .{state_text});

    const module_op = session.moduleOperation();
    try writer.print(
        "  target name={s} kind={s} replicas=",
        .{
            stringAttrValue(getAttr(module_op, "pjrtx.target.name")) orelse "unknown",
            stringAttrValue(getAttr(module_op, "pjrtx.target.kind")) orelse "unknown",
        },
    );
    try writeIntegerAttrText(getAttr(module_op, "pjrtx.target.replicas"), writer);
    try writer.writeAll(" partitions=");
    try writeIntegerAttrText(getAttr(module_op, "pjrtx.target.partitions"), writer);
    try writer.print(" fingerprint={s}\n", .{stringAttrValue(getAttr(module_op, "pjrtx.target.fingerprint")) orelse "unknown"});
    try writeFusionCandidateSummary(module_op, writer);
    try writeFusionPlanSummary(module_op, writer);
    try writePlacementPlanSummary(module_op, writer);
    try writeCollectivePlanSummary(module_op, writer);
    try writeLoweringPlanSummary(module_op, writer);
    try writePerformancePlanSummary(module_op, writer);
    try writeKernelCodegenPlanSummary(module_op, writer);
    try writeSchedulePlanSummary(module_op, writer);
    try writeBackendBindingSummary(module_op, writer);
    try writeExecutableContractSummary(module_op, writer);
    try writeBackendExecutablePlanSummary(module_op, writer);
    try writeBackendKernelGraphSummary(module_op, writer);
    try writeRuntimeAllocationSummary(module_op, writer);
    try writeRuntimeStreamSummary(module_op, writer);
    try writeRuntimeProfileSummary(module_op, writer);
    try writeRuntimeProfileJoinSummary(module_op, writer);
    try writeBackendProfileJoinSummary(module_op, writer);
}

fn runSingleExternalPass(
    session: *MlirSession,
    diagnostics: *std.Io.Writer,
    pass_name: []const u8,
    pass: mlir.MlirPass,
    manager_error: anyerror,
) !bool {
    const result = passes.PassRunner.runModulePass(
        &session.pass_manager,
        session.context,
        session.moduleOperation(),
        pass_name,
        pass,
        diagnostics,
    ) catch |err| switch (err) {
        error.PassManagerCreateFailed => return manager_error,
        else => return err,
    };
    return result.pass_manager_failed;
}

pub fn runExternalStateProbePass(
    session: *MlirSession,
    diagnostics: *std.Io.Writer,
) !ExternalPassProbeRecord {
    var pass_data: state_target_passes.ExternalStateProbePass.Data = .{};
    const pass = state_target_passes.ExternalStateProbePass.create(&pass_data);
    const pass_manager_failed = try runSingleExternalPass(session, diagnostics, "pjrtx_external_state_probe", pass, MlirStateError.ExternalPassFailed);

    if (pass_data.missing_state) {
        try diagnostics.writeAll("pass=pjrtx_external_state_probe feature=state reason=missing module state\n");
    }
    if (pass_data.run_count == 0) {
        try diagnostics.writeAll("pass=pjrtx_external_state_probe feature=run reason=external pass did not run\n");
        return MlirStateError.ExternalPassFailed;
    }
    if (pass_manager_failed or pass_data.missing_state) {
        return MlirStateError.ExternalPassFailed;
    }

    const proof_attr = getAttr(session.moduleOperation(), "pjrtx.external_pass.proof");
    if (std.mem.eql(u8, stringAttrValue(proof_attr) orelse "", "ran")) {
        return pass_data.record();
    }

    try diagnostics.writeAll("pass=pjrtx_external_state_probe feature=proof-attr reason=external pass did not stamp module\n");
    return MlirStateError.ExternalPassFailed;
}

pub fn runTargetLegalExternalPass(
    session: *MlirSession,
    diagnostics: *std.Io.Writer,
) !void {
    var pass_data: state_target_passes.TargetLegalExternalPass.Data = .{};
    const pass = state_target_passes.TargetLegalExternalPass.create(&pass_data);
    const pass_manager_failed = try runSingleExternalPass(session, diagnostics, "target_legal_external", pass, MlirStateError.InvalidStateTransition);

    if (pass_data.run_count == 0) {
        try diagnostics.writeAll("pass=target_legal_external feature=run reason=external pass did not run\n");
        return MlirStateError.InvalidStateTransition;
    }
    if (pass_data.invalid_state) {
        try diagnostics.writeAll("pass=target_legal_external feature=state reason=expected target_attached module state\n");
        return MlirStateError.InvalidStateTransition;
    }
    if (pass_data.missing_target_attr) {
        try diagnostics.writeAll("pass=target_legal_external feature=target reason=missing required target attachment attribute\n");
        return MlirStateError.InvalidTargetAttachment;
    }
    if (pass_manager_failed) {
        try diagnostics.writeAll("pass=target_legal_external feature=pass-manager reason=MLIR pass manager reported failure\n");
        return MlirStateError.InvalidStateTransition;
    }
    try requireModuleState(session, .target_legal, diagnostics);
}

pub fn runFusionPlanExternalPass(
    session: *MlirSession,
    diagnostics: *std.Io.Writer,
) !void {
    var pass_data: fusion_passes.FusionPlanExternalPass.Data = .{};
    const pass = fusion_passes.FusionPlanExternalPass.create(&pass_data);
    const pass_manager_failed = try runSingleExternalPass(session, diagnostics, "fusion_plan_external", pass, MlirStateError.InvalidFusionPlan);

    if (pass_data.run_count == 0) {
        try diagnostics.writeAll("pass=fusion_plan_external feature=run reason=external pass did not run\n");
        return MlirStateError.InvalidFusionPlan;
    }
    if (pass_data.invalid_state) {
        try diagnostics.writeAll("pass=fusion_plan_external feature=state reason=expected target_legal module state\n");
        return MlirStateError.InvalidStateTransition;
    }
    if (pass_data.missing_candidates) {
        try diagnostics.writeAll("pass=fusion_plan_external feature=fusion-candidates reason=missing MLIR fusion candidate attribute\n");
        return MlirStateError.InvalidFusionPlan;
    }
    if (pass_data.invalid_entry) {
        try diagnostics.writeAll("pass=fusion_plan_external feature=fusion-plan reason=invalid fusion plan entry\n");
        return MlirStateError.InvalidFusionPlan;
    }
    if (pass_manager_failed) {
        try diagnostics.writeAll("pass=fusion_plan_external feature=pass-manager reason=MLIR pass manager reported failure\n");
        return MlirStateError.InvalidFusionPlan;
    }
    try requireModuleState(session, .fusion_planned, diagnostics);
}

pub fn runFusionCandidateDiscoveryExternalPass(
    session: *MlirSession,
    diagnostics: *std.Io.Writer,
) !void {
    var pass_data: fusion_passes.FusionCandidateDiscoveryExternalPass.Data = .{};
    const pass = fusion_passes.FusionCandidateDiscoveryExternalPass.create(&pass_data);
    const pass_manager_failed = try runSingleExternalPass(session, diagnostics, "fusion_candidate_external", pass, MlirStateError.InvalidFusionPlan);

    if (pass_data.run_count == 0) {
        try diagnostics.writeAll("pass=fusion_candidate_external feature=run reason=external pass did not run\n");
        return MlirStateError.InvalidFusionPlan;
    }
    if (pass_data.invalid_state) {
        try diagnostics.writeAll("pass=fusion_candidate_external feature=state reason=expected target_legal module state\n");
        return MlirStateError.InvalidStateTransition;
    }
    if (pass_manager_failed) {
        try diagnostics.writeAll("pass=fusion_candidate_external feature=pass-manager reason=MLIR pass manager reported failure\n");
        return MlirStateError.InvalidFusionPlan;
    }
    try requireModuleState(session, .target_legal, diagnostics);
}

pub fn runPlacementPlanExternalPass(
    session: *MlirSession,
    diagnostics: *std.Io.Writer,
) !void {
    var pass_data: placement_collective_passes.PlacementPlanExternalPass.Data = .{};
    const pass = placement_collective_passes.PlacementPlanExternalPass.create(&pass_data);
    const pass_manager_failed = try runSingleExternalPass(session, diagnostics, "placement_plan_external", pass, MlirStateError.InvalidPlacementPlan);

    if (pass_data.run_count == 0) {
        try diagnostics.writeAll("pass=placement_plan_external feature=run reason=external pass did not run\n");
        return MlirStateError.InvalidPlacementPlan;
    }
    if (pass_data.invalid_state) {
        try diagnostics.writeAll("pass=placement_plan_external feature=state reason=expected fusion_planned module state\n");
        return MlirStateError.InvalidStateTransition;
    }
    if (pass_data.missing_placement_records) {
        try diagnostics.writeAll("pass=placement_plan_external feature=placement-records reason=missing MLIR placement records attribute\n");
        return MlirStateError.InvalidPlacementPlan;
    }
    if (pass_data.invalid_record) {
        try diagnostics.writeAll("pass=placement_plan_external feature=placement-record reason=invalid placement record entry\n");
        return MlirStateError.InvalidPlacementPlan;
    }
    if (pass_manager_failed) {
        try diagnostics.writeAll("pass=placement_plan_external feature=pass-manager reason=MLIR pass manager reported failure\n");
        return MlirStateError.InvalidPlacementPlan;
    }
    try requireModuleState(session, .placement_planned, diagnostics);
}

pub fn runCollectivePlanExternalPass(
    session: *MlirSession,
    diagnostics: *std.Io.Writer,
) !void {
    var pass_data: placement_collective_passes.CollectivePlanExternalPass.Data = .{};
    const pass = placement_collective_passes.CollectivePlanExternalPass.create(&pass_data);
    const pass_manager_failed = try runSingleExternalPass(session, diagnostics, "collective_plan_external", pass, MlirStateError.InvalidCollectivePlan);

    if (pass_data.run_count == 0) {
        try diagnostics.writeAll("pass=collective_plan_external feature=run reason=external pass did not run\n");
        return MlirStateError.InvalidCollectivePlan;
    }
    if (pass_data.invalid_state) {
        try diagnostics.writeAll("pass=collective_plan_external feature=state reason=expected placement_planned module state\n");
        return MlirStateError.InvalidStateTransition;
    }
    if (pass_data.missing_collective_records) {
        try diagnostics.writeAll("pass=collective_plan_external feature=collective-records reason=missing MLIR collective records attribute\n");
        return MlirStateError.InvalidCollectivePlan;
    }
    if (pass_data.invalid_record) {
        try diagnostics.writeAll("pass=collective_plan_external feature=collective-record reason=invalid collective plan record entry\n");
        return MlirStateError.InvalidCollectivePlan;
    }
    if (pass_manager_failed) {
        try diagnostics.writeAll("pass=collective_plan_external feature=pass-manager reason=MLIR pass manager reported failure\n");
        return MlirStateError.InvalidCollectivePlan;
    }
    try requireModuleState(session, .collectives_planned, diagnostics);
}

pub fn runLoweringPlanExternalPass(
    session: *MlirSession,
    diagnostics: *std.Io.Writer,
) !void {
    var pass_data: LoweringPlanExternalPass.Data = .{};
    const pass = LoweringPlanExternalPass.create(&pass_data);
    const pass_manager_failed = try runSingleExternalPass(session, diagnostics, "lowering_plan_external", pass, MlirStateError.InvalidLoweringPlan);

    if (pass_data.run_count == 0) {
        try diagnostics.writeAll("pass=lowering_plan_external feature=run reason=external pass did not run\n");
        return MlirStateError.InvalidLoweringPlan;
    }
    if (pass_data.invalid_state) {
        try diagnostics.writeAll("pass=lowering_plan_external feature=state reason=expected collectives_planned module state\n");
        return MlirStateError.InvalidStateTransition;
    }
    if (pass_data.missing_lowering_records) {
        try diagnostics.writeAll("pass=lowering_plan_external feature=lowering-records reason=missing MLIR lowering records attribute\n");
        return MlirStateError.InvalidLoweringPlan;
    }
    if (pass_data.missing_region_facts) {
        try diagnostics.writeAll("pass=lowering_plan_external feature=region-facts reason=missing MLIR lowering region facts attribute\n");
        return MlirStateError.InvalidLoweringPlan;
    }
    if (pass_data.invalid_record) {
        try diagnostics.writeAll("pass=lowering_plan_external feature=lowering-record reason=invalid lowering record entry\n");
        return MlirStateError.InvalidLoweringPlan;
    }
    if (pass_manager_failed) {
        try diagnostics.writeAll("pass=lowering_plan_external feature=pass-manager reason=MLIR pass manager reported failure\n");
        return MlirStateError.InvalidLoweringPlan;
    }
    try requireModuleState(session, .lowering_planned, diagnostics);
}

pub fn runKernelCodegenPlanExternalPass(
    session: *MlirSession,
    diagnostics: *std.Io.Writer,
) !void {
    var pass_data: KernelCodegenPlanExternalPass.Data = .{};
    const pass = KernelCodegenPlanExternalPass.create(&pass_data);
    const pass_manager_failed = try runSingleExternalPass(session, diagnostics, "codegen_plan_external", pass, MlirStateError.InvalidKernelCodegenPlan);

    if (pass_data.run_count == 0) {
        try diagnostics.writeAll("pass=codegen_plan_external feature=run reason=external pass did not run\n");
        return MlirStateError.InvalidKernelCodegenPlan;
    }
    if (pass_data.invalid_state) {
        try diagnostics.writeAll("pass=codegen_plan_external feature=state reason=expected performance_modeled module state\n");
        return MlirStateError.InvalidStateTransition;
    }
    if (pass_data.missing_codegen_records) {
        try diagnostics.writeAll("pass=codegen_plan_external feature=codegen-records reason=missing MLIR codegen records attribute\n");
        return MlirStateError.InvalidKernelCodegenPlan;
    }
    if (pass_data.invalid_record) {
        try diagnostics.writeAll("pass=codegen_plan_external feature=codegen-record reason=invalid codegen record entry\n");
        return MlirStateError.InvalidKernelCodegenPlan;
    }
    if (pass_manager_failed) {
        try diagnostics.writeAll("pass=codegen_plan_external feature=pass-manager reason=MLIR pass manager reported failure\n");
        return MlirStateError.InvalidKernelCodegenPlan;
    }
    try requireModuleState(session, .codegen_planned, diagnostics);
}

pub fn runPerformanceFactsExternalPass(
    session: *MlirSession,
    diagnostics: *std.Io.Writer,
) !void {
    var pass_data: PerformanceFactsExternalPass.Data = .{};
    const pass = PerformanceFactsExternalPass.create(&pass_data);
    const pass_manager_failed = try runSingleExternalPass(session, diagnostics, "performance_external", pass, MlirStateError.InvalidKernelCodegenPlan);

    if (pass_data.run_count == 0) {
        try diagnostics.writeAll("pass=performance_external feature=run reason=external pass did not run\n");
        return MlirStateError.InvalidKernelCodegenPlan;
    }
    if (pass_data.invalid_state) {
        try diagnostics.writeAll("pass=performance_external feature=state reason=expected lowering_planned module state\n");
        return MlirStateError.InvalidStateTransition;
    }
    if (pass_data.missing_performance_attrs) {
        try diagnostics.writeAll("pass=performance_external feature=attrs reason=missing MLIR performance attributes\n");
        return MlirStateError.InvalidKernelCodegenPlan;
    }
    if (pass_data.invalid_record) {
        try diagnostics.writeAll("pass=performance_external feature=record reason=invalid performance record entry\n");
        return MlirStateError.InvalidKernelCodegenPlan;
    }
    if (pass_manager_failed) {
        try diagnostics.writeAll("pass=performance_external feature=pass-manager reason=MLIR pass manager reported failure\n");
        return MlirStateError.InvalidKernelCodegenPlan;
    }
    try requireModuleState(session, .performance_modeled, diagnostics);
}

pub fn runSchedulePlanExternalPass(
    session: *MlirSession,
    diagnostics: *std.Io.Writer,
) !void {
    var pass_data: SchedulePlanExternalPass.Data = .{};
    const pass = SchedulePlanExternalPass.create(&pass_data);
    const pass_manager_failed = try runSingleExternalPass(session, diagnostics, "schedule_plan_external", pass, MlirStateError.InvalidSchedulePlan);

    if (pass_data.run_count == 0) {
        try diagnostics.writeAll("pass=schedule_plan_external feature=run reason=external pass did not run\n");
        return MlirStateError.InvalidSchedulePlan;
    }
    if (pass_data.invalid_state) {
        try diagnostics.writeAll("pass=schedule_plan_external feature=state reason=expected codegen_planned module state\n");
        return MlirStateError.InvalidStateTransition;
    }
    if (pass_data.missing_schedule_records) {
        try diagnostics.writeAll("pass=schedule_plan_external feature=schedule-records reason=missing MLIR schedule command or overlap attribute\n");
        return MlirStateError.InvalidSchedulePlan;
    }
    if (pass_data.invalid_record) {
        try diagnostics.writeAll("pass=schedule_plan_external feature=schedule-record reason=invalid schedule record entry\n");
        return MlirStateError.InvalidSchedulePlan;
    }
    if (pass_manager_failed) {
        try diagnostics.writeAll("pass=schedule_plan_external feature=pass-manager reason=MLIR pass manager reported failure\n");
        return MlirStateError.InvalidSchedulePlan;
    }
    try requireModuleState(session, .scheduled, diagnostics);
}

pub fn runBackendBindingExternalPass(
    session: *MlirSession,
    diagnostics: *std.Io.Writer,
) !void {
    var pass_data: BackendBindingExternalPass.Data = .{};
    const pass = BackendBindingExternalPass.create(&pass_data);
    const pass_manager_failed = try runSingleExternalPass(session, diagnostics, "backend_binding_external", pass, MlirStateError.InvalidBackendBindingPlan);

    if (pass_data.run_count == 0) {
        try diagnostics.writeAll("pass=backend_binding_external feature=run reason=external pass did not run\n");
        return MlirStateError.InvalidBackendBindingPlan;
    }
    if (pass_data.invalid_state) {
        try diagnostics.writeAll("pass=backend_binding_external feature=state reason=expected scheduled module state\n");
        return MlirStateError.InvalidStateTransition;
    }
    if (pass_data.missing_backend_bindings) {
        try diagnostics.writeAll("pass=backend_binding_external feature=bindings reason=missing MLIR backend binding attribute\n");
        return MlirStateError.InvalidBackendBindingPlan;
    }
    if (pass_data.invalid_record) {
        try diagnostics.writeAll("pass=backend_binding_external feature=binding reason=invalid backend binding entry\n");
        return MlirStateError.InvalidBackendBindingPlan;
    }
    if (pass_manager_failed) {
        try diagnostics.writeAll("pass=backend_binding_external feature=pass-manager reason=MLIR pass manager reported failure\n");
        return MlirStateError.InvalidBackendBindingPlan;
    }
    try requireModuleState(session, .backend_bound, diagnostics);
}

pub fn runExecutableReadyExternalPass(
    session: *MlirSession,
    diagnostics: *std.Io.Writer,
) !void {
    var pass_data: ExecutableReadyExternalPass.Data = .{};
    const pass = ExecutableReadyExternalPass.create(&pass_data);
    const pass_manager_failed = try runSingleExternalPass(session, diagnostics, "executable_ready_external", pass, MlirStateError.InvalidExecutableContract);

    if (pass_data.run_count == 0) {
        try diagnostics.writeAll("pass=executable_ready_external feature=run reason=external pass did not run\n");
        return MlirStateError.InvalidExecutableContract;
    }
    if (pass_data.invalid_state) {
        try diagnostics.writeAll("pass=executable_ready_external feature=state reason=expected backend_bound module state\n");
        return MlirStateError.InvalidStateTransition;
    }
    if (pass_data.missing_contract) {
        try diagnostics.writeAll("pass=executable_ready_external feature=contract reason=missing MLIR executable contract attribute\n");
        return MlirStateError.InvalidExecutableContract;
    }
    if (pass_data.invalid_contract) {
        try diagnostics.writeAll("pass=executable_ready_external feature=contract reason=invalid executable contract entry\n");
        return MlirStateError.InvalidExecutableContract;
    }
    if (pass_data.contract_mismatch) {
        try diagnostics.writeAll("pass=executable_ready_external feature=contract reason=executable contract does not match verified MLIR facts\n");
        return MlirStateError.InvalidExecutableContract;
    }
    if (pass_manager_failed) {
        try diagnostics.writeAll("pass=executable_ready_external feature=pass-manager reason=MLIR pass manager reported failure\n");
        return MlirStateError.InvalidExecutableContract;
    }
    try requireModuleState(session, .executable_ready, diagnostics);
}

pub fn runBackendExecutablePlanExternalPass(
    session: *MlirSession,
    diagnostics: *std.Io.Writer,
) !void {
    var pass_data: BackendExecutablePlanExternalPass.Data = .{};
    const pass = BackendExecutablePlanExternalPass.create(&pass_data);
    const pass_manager_failed = try runSingleExternalPass(session, diagnostics, "backend_executable_external", pass, MlirStateError.InvalidBackendExecutablePlan);

    if (pass_data.run_count == 0) {
        try diagnostics.writeAll("pass=backend_executable_external feature=run reason=external pass did not run\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    if (pass_data.invalid_state) {
        try diagnostics.writeAll("pass=backend_executable_external feature=state reason=expected executable_ready module state\n");
        return MlirStateError.InvalidStateTransition;
    }
    if (pass_data.missing_plan) {
        try diagnostics.writeAll("pass=backend_executable_external feature=plan reason=missing MLIR backend executable plan attribute\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    if (pass_data.invalid_plan) {
        try diagnostics.writeAll("pass=backend_executable_external feature=plan reason=invalid backend executable plan entry\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    if (pass_data.plan_mismatch) {
        try diagnostics.writeAll("pass=backend_executable_external feature=plan reason=backend executable plan does not match verified MLIR facts\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    if (pass_manager_failed) {
        try diagnostics.writeAll("pass=backend_executable_external feature=pass-manager reason=MLIR pass manager reported failure\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    try requireModuleState(session, .backend_executable_planned, diagnostics);
}

pub fn runBackendKernelGraphExternalPass(
    session: *MlirSession,
    diagnostics: *std.Io.Writer,
) !void {
    var pass_data: BackendKernelGraphExternalPass.Data = .{};
    const pass = BackendKernelGraphExternalPass.create(&pass_data);
    const pass_manager_failed = try runSingleExternalPass(session, diagnostics, "backend_kernel_graph_external", pass, MlirStateError.InvalidBackendExecutablePlan);

    if (pass_data.run_count == 0) {
        try diagnostics.writeAll("pass=backend_kernel_graph_external feature=run reason=external pass did not run\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    if (pass_data.invalid_state) {
        try diagnostics.writeAll("pass=backend_kernel_graph_external feature=state reason=expected backend_executable_planned module state\n");
        return MlirStateError.InvalidStateTransition;
    }
    if (pass_data.missing_graph) {
        try diagnostics.writeAll("pass=backend_kernel_graph_external feature=graph reason=missing MLIR backend kernel graph attribute\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    if (pass_data.invalid_graph) {
        try diagnostics.writeAll("pass=backend_kernel_graph_external feature=graph reason=invalid backend kernel graph entry\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    if (pass_data.graph_mismatch) {
        try diagnostics.writeAll("pass=backend_kernel_graph_external feature=graph reason=backend kernel graph does not match verified executable facts\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    if (pass_manager_failed) {
        try diagnostics.writeAll("pass=backend_kernel_graph_external feature=pass-manager reason=MLIR pass manager reported failure\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    try requireModuleState(session, .backend_kernel_graph_planned, diagnostics);
}

pub fn runRuntimeAllocationExternalPass(
    session: *MlirSession,
    diagnostics: *std.Io.Writer,
) !void {
    var pass_data: RuntimeAllocationExternalPass.Data = .{};
    const pass = RuntimeAllocationExternalPass.create(&pass_data);
    const pass_manager_failed = try runSingleExternalPass(session, diagnostics, "runtime_allocation_external", pass, MlirStateError.InvalidBackendExecutablePlan);

    if (pass_data.run_count == 0) {
        try diagnostics.writeAll("pass=runtime_allocation_external feature=run reason=external pass did not run\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    if (pass_data.invalid_state) {
        try diagnostics.writeAll("pass=runtime_allocation_external feature=state reason=expected backend executable or kernel graph module state\n");
        return MlirStateError.InvalidStateTransition;
    }
    if (pass_data.missing_plan) {
        try diagnostics.writeAll("pass=runtime_allocation_external feature=plan reason=missing MLIR runtime allocation attribute\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    if (pass_data.invalid_plan) {
        try diagnostics.writeAll("pass=runtime_allocation_external feature=plan reason=invalid runtime allocation entry\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    if (pass_manager_failed) {
        try diagnostics.writeAll("pass=runtime_allocation_external feature=pass-manager reason=MLIR pass manager reported failure\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    try requireModuleState(session, .runtime_allocation_planned, diagnostics);
}

pub fn runRuntimeStreamExternalPass(
    session: *MlirSession,
    diagnostics: *std.Io.Writer,
) !void {
    var pass_data: RuntimeStreamExternalPass.Data = .{};
    const pass = RuntimeStreamExternalPass.create(&pass_data);
    const pass_manager_failed = try runSingleExternalPass(session, diagnostics, "runtime_stream_external", pass, MlirStateError.InvalidBackendExecutablePlan);

    if (pass_data.run_count == 0) {
        try diagnostics.writeAll("pass=runtime_stream_external feature=run reason=external pass did not run\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    if (pass_data.invalid_state) {
        try diagnostics.writeAll("pass=runtime_stream_external feature=state reason=expected runtime_allocation_planned module state\n");
        return MlirStateError.InvalidStateTransition;
    }
    if (pass_data.missing_plan) {
        try diagnostics.writeAll("pass=runtime_stream_external feature=plan reason=missing MLIR runtime stream attribute\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    if (pass_data.invalid_plan) {
        try diagnostics.writeAll("pass=runtime_stream_external feature=plan reason=invalid runtime stream entry\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    if (pass_manager_failed) {
        try diagnostics.writeAll("pass=runtime_stream_external feature=pass-manager reason=MLIR pass manager reported failure\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    try requireModuleState(session, .runtime_stream_planned, diagnostics);
}

pub fn runRuntimeProfileExternalPass(
    session: *MlirSession,
    diagnostics: *std.Io.Writer,
) !void {
    var pass_data: RuntimeProfileExternalPass.Data = .{};
    const pass = RuntimeProfileExternalPass.create(&pass_data);
    const pass_manager_failed = try runSingleExternalPass(session, diagnostics, "runtime_profile_external", pass, MlirStateError.InvalidBackendExecutablePlan);

    if (pass_data.run_count == 0) {
        try diagnostics.writeAll("pass=runtime_profile_external feature=run reason=external pass did not run\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    if (pass_data.invalid_state) {
        try diagnostics.writeAll("pass=runtime_profile_external feature=state reason=expected runtime_stream_planned module state\n");
        return MlirStateError.InvalidStateTransition;
    }
    if (pass_data.missing_profile) {
        try diagnostics.writeAll("pass=runtime_profile_external feature=profile reason=missing MLIR runtime profile events attribute\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    if (pass_data.invalid_profile) {
        try diagnostics.writeAll("pass=runtime_profile_external feature=profile reason=invalid runtime profile event entry\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    if (pass_manager_failed) {
        try diagnostics.writeAll("pass=runtime_profile_external feature=pass-manager reason=MLIR pass manager reported failure\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    try requireModuleState(session, .runtime_profiled, diagnostics);
}

pub fn runRuntimeProfileJoinExternalPass(
    session: *MlirSession,
    diagnostics: *std.Io.Writer,
) !void {
    var pass_data: RuntimeProfileJoinExternalPass.Data = .{};
    const pass = RuntimeProfileJoinExternalPass.create(&pass_data);
    const pass_manager_failed = try runSingleExternalPass(session, diagnostics, "runtime_profile_join_external", pass, MlirStateError.InvalidBackendExecutablePlan);

    if (pass_data.run_count == 0) {
        try diagnostics.writeAll("pass=runtime_profile_join_external feature=run reason=external pass did not run\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    if (pass_data.invalid_state) {
        try diagnostics.writeAll("pass=runtime_profile_join_external feature=state reason=expected runtime_profiled module state\n");
        return MlirStateError.InvalidStateTransition;
    }
    if (pass_data.missing_profile) {
        try diagnostics.writeAll("pass=runtime_profile_join_external feature=profile reason=missing MLIR runtime profile events attribute\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    if (pass_data.missing_joins) {
        try diagnostics.writeAll("pass=runtime_profile_join_external feature=joins reason=missing MLIR runtime profile joins attribute\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    if (pass_data.invalid_joins) {
        try diagnostics.writeAll("pass=runtime_profile_join_external feature=joins reason=invalid runtime profile join entry\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    if (pass_manager_failed) {
        try diagnostics.writeAll("pass=runtime_profile_join_external feature=pass-manager reason=MLIR pass manager reported failure\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    try requireModuleState(session, .runtime_profile_joined, diagnostics);
}

pub fn runBackendProfileJoinExternalPass(
    session: *MlirSession,
    diagnostics: *std.Io.Writer,
) !void {
    var pass_data: BackendProfileJoinExternalPass.Data = .{};
    const pass = BackendProfileJoinExternalPass.create(&pass_data);
    const pass_manager_failed = try runSingleExternalPass(session, diagnostics, "backend_profile_join_external", pass, MlirStateError.InvalidBackendExecutablePlan);

    if (pass_data.run_count == 0) {
        try diagnostics.writeAll("pass=backend_profile_join_external feature=run reason=external pass did not run\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    if (pass_data.invalid_state) {
        try diagnostics.writeAll("pass=backend_profile_join_external feature=state reason=expected runtime_profile_joined module state\n");
        return MlirStateError.InvalidStateTransition;
    }
    if (pass_data.missing_executable) {
        try diagnostics.writeAll("pass=backend_profile_join_external feature=executable reason=missing MLIR backend executable attribute\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    if (pass_data.missing_profile) {
        try diagnostics.writeAll("pass=backend_profile_join_external feature=profile reason=missing MLIR runtime profile events attribute\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    if (pass_data.missing_joins) {
        try diagnostics.writeAll("pass=backend_profile_join_external feature=joins reason=missing MLIR backend profile joins attribute\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    if (pass_data.invalid_joins) {
        try diagnostics.writeAll("pass=backend_profile_join_external feature=joins reason=invalid backend profile join entry\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    if (pass_manager_failed) {
        try diagnostics.writeAll("pass=backend_profile_join_external feature=pass-manager reason=MLIR pass manager reported failure\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    try requireModuleState(session, .backend_profile_joined, diagnostics);
}

const LoweringPlanExternalPass = struct {
    var type_id_anchor: u64 = 0;

    const Data = struct {
        construct_count: u32 = 0,
        initialize_count: u32 = 0,
        run_count: u32 = 0,
        clone_count: u32 = 0,
        destruct_count: u32 = 0,
        invalid_state: bool = false,
        missing_lowering_records: bool = false,
        missing_region_facts: bool = false,
        invalid_record: bool = false,
    };

    fn create(data: *Data) mlir.MlirPass {
        const callbacks: mlir.MlirExternalPassCallbacks = .{
            .construct = construct,
            .destruct = destruct,
            .initialize = initialize,
            .clone = clone,
            .run = run,
        };
        return mlir.mlirCreateExternalPass(
            mlir.mlirTypeIDCreate(&type_id_anchor),
            mlirStringRef("pjrtx-lowering-plan-external"),
            mlirStringRef("pjrtx-lowering-plan-external"),
            mlirStringRef("Verifies PjRTx lowering region records and marks lowering planned"),
            mlirStringRef("builtin.module"),
            0,
            null,
            callbacks,
            data,
        );
    }

    fn construct(user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.construct_count += 1;
    }

    fn destruct(user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.destruct_count += 1;
    }

    fn initialize(_: mlir.MlirContext, user_data: ?*anyopaque) callconv(.c) mlir.MlirLogicalResult {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.initialize_count += 1;
        return mlir.MlirLogicalResult{ .value = 1 };
    }

    fn clone(user_data: ?*anyopaque) callconv(.c) ?*anyopaque {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.clone_count += 1;
        return user_data;
    }

    fn run(op: mlir.MlirOperation, pass: mlir.MlirExternalPass, user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.run_count += 1;

        const state_attr = getAttr(op, "pjrtx.state");
        if (!stringAttrEquals(state_attr, "collectives_planned")) {
            data.invalid_state = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        const records_attr = getAttr(op, "pjrtx.lowering.records");
        if (mlir.mlirAttributeIsNull(records_attr) or !mlir.mlirAttributeIsAArray(records_attr)) {
            data.missing_lowering_records = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        const region_facts_attr = getAttr(op, "pjrtx.lowering.region_facts");
        if (mlir.mlirAttributeIsNull(region_facts_attr) or !mlir.mlirAttributeIsAArray(region_facts_attr)) {
            data.missing_region_facts = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        if (!verifyLoweringRecordsAttr(records_attr)) {
            data.invalid_record = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }
        if (!verifyLoweringRegionFactsAttr(region_facts_attr, records_attr)) {
            data.invalid_record = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        const context = mlir.mlirOperationGetContext(op);
        setStringAttr(context, op, "pjrtx.state", ModuleState.lowering_planned.text());
        setStringAttr(context, op, "pjrtx.lowering_plan.pass", "pjrtx-lowering-plan-external");
    }

    fn verifyLoweringRecordsAttr(records_attr: mlir.MlirAttribute) bool {
        const count = mlir.mlirArrayAttrGetNumElements(records_attr);
        if (count <= 0 or count > limits.max_lowering_records) return false;
        var index: isize = 0;
        while (index < count) : (index += 1) {
            if (!verifyLoweringRecordAttr(mlir.mlirArrayAttrGetElement(records_attr, index))) return false;
        }
        return true;
    }

    fn verifyLoweringRecordAttr(record: mlir.MlirAttribute) bool {
        if (mlir.mlirAttributeIsNull(record) or !mlir.mlirAttributeIsADictionary(record)) return false;
        if (!hasIntegerDictAttr(record, "id")) return false;
        if (!hasKnownLoweringDecision(dictAttr(record, "decision"))) return false;
        if (!hasNonEmptyStringDictAttr(record, "reason")) return false;

        const instructions = dictAttr(record, "instructions");
        if (mlir.mlirAttributeIsNull(instructions) or !mlir.mlirAttributeIsAArray(instructions)) return false;
        const instruction_count = mlir.mlirArrayAttrGetNumElements(instructions);
        if (instruction_count <= 0 or instruction_count > limits.max_kernel_ref_ids) return false;

        const costs = dictAttr(record, "costs");
        if (mlir.mlirAttributeIsNull(costs) or !mlir.mlirAttributeIsAArray(costs)) return false;
        const cost_count = mlir.mlirArrayAttrGetNumElements(costs);
        if (cost_count <= 0 or cost_count > limits.max_kernel_ref_ids) return false;

        const alternatives = dictAttr(record, "rejected_alternatives");
        if (mlir.mlirAttributeIsNull(alternatives) or !mlir.mlirAttributeIsAArray(alternatives)) return false;
        const alternative_count = mlir.mlirArrayAttrGetNumElements(alternatives);
        if (alternative_count > limits.max_kernel_ref_ids) return false;
        var alternative_index: isize = 0;
        while (alternative_index < alternative_count) : (alternative_index += 1) {
            const alternative = mlir.mlirArrayAttrGetElement(alternatives, alternative_index);
            if ((stringAttrValue(alternative) orelse return false).len == 0) return false;
        }
        return true;
    }

    fn verifyLoweringRegionFactsAttr(region_facts_attr: mlir.MlirAttribute, records_attr: mlir.MlirAttribute) bool {
        const count = mlir.mlirArrayAttrGetNumElements(region_facts_attr);
        if (count <= 0 or count > limits.max_lowering_records) return false;
        if (count != mlir.mlirArrayAttrGetNumElements(records_attr)) return false;
        var index: isize = 0;
        while (index < count) : (index += 1) {
            const fact = mlir.mlirArrayAttrGetElement(region_facts_attr, index);
            const record = mlir.mlirArrayAttrGetElement(records_attr, index);
            if (!verifyLoweringRegionFactAttr(fact, record, index)) return false;
        }
        return true;
    }

    fn verifyLoweringRegionFactAttr(fact: mlir.MlirAttribute, record: mlir.MlirAttribute, expected_index: isize) bool {
        if (mlir.mlirAttributeIsNull(fact) or !mlir.mlirAttributeIsADictionary(fact)) return false;
        if (!hasIntegerDictAttr(fact, "lowering")) return false;
        if (mlir.mlirIntegerAttrGetValueInt(dictAttr(fact, "lowering")) != expected_index) return false;
        if (!validOptionalIntegerDictAttr(fact, "fusion_group")) return false;
        if (!verifyIntegerArrayAttr(dictAttr(fact, "placements"), limits.max_kernel_ref_ids, true)) return false;
        if (!verifyPositiveIntegerArrayAttr(dictAttr(fact, "tile"), limits.max_tile_rank)) return false;
        if (!hasIntegerDictAttr(fact, "result_memory")) return false;
        if (!validOptionalIntegerDictAttr(fact, "tile_memory")) return false;
        if (!hasKnownLoweringDecision(dictAttr(fact, "codegen_region"))) return false;
        if (!hasNonEmptyStringDictAttr(fact, "reason")) return false;

        const record_decision = dictAttr(record, "decision");
        if (!stringAttrEquals(dictAttr(fact, "codegen_region"), stringAttrValue(record_decision) orelse return false)) return false;
        const instructions = dictAttr(record, "instructions");
        if (!mlir.mlirAttributeIsAArray(instructions)) return false;
        if (mlir.mlirArrayAttrGetNumElements(dictAttr(fact, "placements")) != mlir.mlirArrayAttrGetNumElements(instructions)) return false;
        return true;
    }
};

const PerformanceFactsExternalPass = struct {
    var type_id_anchor: u64 = 0;

    const Data = struct {
        construct_count: u32 = 0,
        initialize_count: u32 = 0,
        run_count: u32 = 0,
        clone_count: u32 = 0,
        destruct_count: u32 = 0,
        invalid_state: bool = false,
        missing_performance_attrs: bool = false,
        invalid_record: bool = false,
    };

    fn create(data: *Data) mlir.MlirPass {
        const callbacks: mlir.MlirExternalPassCallbacks = .{
            .construct = construct,
            .destruct = destruct,
            .initialize = initialize,
            .clone = clone,
            .run = run,
        };
        return mlir.mlirCreateExternalPass(
            mlir.mlirTypeIDCreate(&type_id_anchor),
            mlirStringRef("pjrtx-performance-external"),
            mlirStringRef("pjrtx-performance-external"),
            mlirStringRef("Verifies PjRTx performance facts and marks performance modeled"),
            mlirStringRef("builtin.module"),
            0,
            null,
            callbacks,
            data,
        );
    }

    fn construct(user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.construct_count += 1;
    }

    fn destruct(user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.destruct_count += 1;
    }

    fn initialize(_: mlir.MlirContext, user_data: ?*anyopaque) callconv(.c) mlir.MlirLogicalResult {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.initialize_count += 1;
        return mlir.MlirLogicalResult{ .value = 1 };
    }

    fn clone(user_data: ?*anyopaque) callconv(.c) ?*anyopaque {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.clone_count += 1;
        return user_data;
    }

    fn run(op: mlir.MlirOperation, pass: mlir.MlirExternalPass, user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.run_count += 1;

        const state_attr = getAttr(op, "pjrtx.state");
        if (!stringAttrEquals(state_attr, "lowering_planned")) {
            data.invalid_state = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        const costs_attr = getAttr(op, "pjrtx.performance.cost_ledger");
        const traffic_attr = getAttr(op, "pjrtx.performance.memory_traffic");
        if (mlir.mlirAttributeIsNull(costs_attr) or !mlir.mlirAttributeIsAArray(costs_attr) or mlir.mlirAttributeIsNull(traffic_attr) or !mlir.mlirAttributeIsAArray(traffic_attr)) {
            data.missing_performance_attrs = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        if (!verifyCostLedgerAttr(costs_attr) or !verifyMemoryTrafficAttr(traffic_attr, mlir.mlirArrayAttrGetNumElements(costs_attr))) {
            data.invalid_record = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        const context = mlir.mlirOperationGetContext(op);
        setStringAttr(context, op, "pjrtx.state", ModuleState.performance_modeled.text());
        setStringAttr(context, op, "pjrtx.performance.pass", "pjrtx-performance-external");
    }

    fn verifyCostLedgerAttr(costs_attr: mlir.MlirAttribute) bool {
        const count = mlir.mlirArrayAttrGetNumElements(costs_attr);
        if (count <= 0 or count > limits.max_cost_ledger_entries) return false;
        var index: isize = 0;
        while (index < count) : (index += 1) {
            const entry = mlir.mlirArrayAttrGetElement(costs_attr, index);
            if (!verifyCostLedgerEntryAttr(entry)) return false;
        }
        return true;
    }

    fn verifyCostLedgerEntryAttr(entry: mlir.MlirAttribute) bool {
        if (mlir.mlirAttributeIsNull(entry) or !mlir.mlirAttributeIsADictionary(entry)) return false;
        if (!hasIntegerDictAttr(entry, "id")) return false;
        const instructions = dictAttr(entry, "instructions");
        if (mlir.mlirAttributeIsNull(instructions) or !mlir.mlirAttributeIsAArray(instructions)) return false;
        const instruction_count = mlir.mlirArrayAttrGetNumElements(instructions);
        if (instruction_count <= 0 or instruction_count > limits.max_kernel_ref_ids) return false;
        if (!hasKnownCostOpClass(dictAttr(entry, "op_class"))) return false;
        if (!hasKnownBufferType(dictAttr(entry, "dtype"))) return false;
        if (!optionalKnownBufferType(dictAttr(entry, "accumulation_dtype"))) return false;
        if (u128FromStringAttrNoDiag(dictAttr(entry, "logical_ops")) == null) return false;
        if (u128FromStringAttrNoDiag(dictAttr(entry, "bytes_read")) == null) return false;
        if (u128FromStringAttrNoDiag(dictAttr(entry, "bytes_written")) == null) return false;
        const expected_unit = dictAttr(entry, "expected_unit");
        if (!mlir.mlirAttributeIsNull(expected_unit) and !mlir.mlirAttributeIsAInteger(expected_unit) and !stringAttrEquals(expected_unit, "none")) return false;
        if (!hasNonEmptyStringDictAttr(entry, "formula")) return false;
        if (!hasNonEmptyStringDictAttr(entry, "approximation")) return false;
        return true;
    }

    fn verifyMemoryTrafficAttr(traffic_attr: mlir.MlirAttribute, cost_count: isize) bool {
        const count = mlir.mlirArrayAttrGetNumElements(traffic_attr);
        if (count > limits.max_memory_traffic_records) return false;
        var index: isize = 0;
        while (index < count) : (index += 1) {
            const record = mlir.mlirArrayAttrGetElement(traffic_attr, index);
            if (!verifyMemoryTrafficRecordAttr(record, cost_count)) return false;
        }
        return true;
    }

    fn verifyMemoryTrafficRecordAttr(record: mlir.MlirAttribute, cost_count: isize) bool {
        if (mlir.mlirAttributeIsNull(record) or !mlir.mlirAttributeIsADictionary(record)) return false;
        if (!hasIntegerDictAttr(record, "id")) return false;
        if (!hasIntegerDictAttr(record, "lowering")) return false;
        if (!hasIntegerDictAttr(record, "memory")) return false;
        if (!hasKnownMemoryTrafficKind(dictAttr(record, "kind"))) return false;
        const instructions = dictAttr(record, "instructions");
        if (mlir.mlirAttributeIsNull(instructions) or !mlir.mlirAttributeIsAArray(instructions)) return false;
        const instruction_count = mlir.mlirArrayAttrGetNumElements(instructions);
        if (instruction_count <= 0 or instruction_count > limits.max_kernel_ref_ids) return false;
        const costs = dictAttr(record, "costs");
        if (mlir.mlirAttributeIsNull(costs) or !mlir.mlirAttributeIsAArray(costs)) return false;
        const record_cost_count = mlir.mlirArrayAttrGetNumElements(costs);
        if (record_cost_count <= 0 or record_cost_count > limits.max_kernel_ref_ids) return false;
        var cost_index: isize = 0;
        while (cost_index < record_cost_count) : (cost_index += 1) {
            const cost_attr = mlir.mlirArrayAttrGetElement(costs, cost_index);
            if (!mlir.mlirAttributeIsAInteger(cost_attr)) return false;
            const cost_id = mlir.mlirIntegerAttrGetValueInt(cost_attr);
            if (cost_id < 0 or cost_id >= cost_count) return false;
        }
        if (u128FromStringAttrNoDiag(dictAttr(record, "bytes_read")) == null) return false;
        if (u128FromStringAttrNoDiag(dictAttr(record, "bytes_written")) == null) return false;
        if (!hasNonEmptyStringDictAttr(record, "reason")) return false;
        return true;
    }
};

const KernelCodegenPlanExternalPass = struct {
    var type_id_anchor: u64 = 0;

    const Data = struct {
        construct_count: u32 = 0,
        initialize_count: u32 = 0,
        run_count: u32 = 0,
        clone_count: u32 = 0,
        destruct_count: u32 = 0,
        invalid_state: bool = false,
        missing_codegen_records: bool = false,
        invalid_record: bool = false,
    };

    fn create(data: *Data) mlir.MlirPass {
        const callbacks: mlir.MlirExternalPassCallbacks = .{
            .construct = construct,
            .destruct = destruct,
            .initialize = initialize,
            .clone = clone,
            .run = run,
        };
        return mlir.mlirCreateExternalPass(
            mlir.mlirTypeIDCreate(&type_id_anchor),
            mlirStringRef("pjrtx-codegen-plan-external"),
            mlirStringRef("pjrtx-codegen-plan-external"),
            mlirStringRef("Verifies PjRTx kernel codegen records and marks codegen planned"),
            mlirStringRef("builtin.module"),
            0,
            null,
            callbacks,
            data,
        );
    }

    fn construct(user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.construct_count += 1;
    }

    fn destruct(user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.destruct_count += 1;
    }

    fn initialize(_: mlir.MlirContext, user_data: ?*anyopaque) callconv(.c) mlir.MlirLogicalResult {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.initialize_count += 1;
        return mlir.MlirLogicalResult{ .value = 1 };
    }

    fn clone(user_data: ?*anyopaque) callconv(.c) ?*anyopaque {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.clone_count += 1;
        return user_data;
    }

    fn run(op: mlir.MlirOperation, pass: mlir.MlirExternalPass, user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.run_count += 1;

        const state_attr = getAttr(op, "pjrtx.state");
        if (!stringAttrEquals(state_attr, "performance_modeled")) {
            data.invalid_state = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        const records_attr = getAttr(op, "pjrtx.codegen.records");
        if (mlir.mlirAttributeIsNull(records_attr) or !mlir.mlirAttributeIsAArray(records_attr)) {
            data.missing_codegen_records = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        if (!verifyKernelCodegenRecordsAttr(records_attr)) {
            data.invalid_record = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        const context = mlir.mlirOperationGetContext(op);
        setStringAttr(context, op, "pjrtx.state", ModuleState.codegen_planned.text());
        setStringAttr(context, op, "pjrtx.codegen_plan.pass", "pjrtx-codegen-plan-external");
    }

    fn verifyKernelCodegenRecordsAttr(records_attr: mlir.MlirAttribute) bool {
        const count = mlir.mlirArrayAttrGetNumElements(records_attr);
        if (count <= 0 or count > limits.max_kernel_codegen_records) return false;
        var index: isize = 0;
        while (index < count) : (index += 1) {
            if (!verifyKernelCodegenRecordAttr(mlir.mlirArrayAttrGetElement(records_attr, index))) return false;
        }
        return true;
    }

    fn verifyKernelCodegenRecordAttr(record: mlir.MlirAttribute) bool {
        if (mlir.mlirAttributeIsNull(record) or !mlir.mlirAttributeIsADictionary(record)) return false;
        if (!hasIntegerDictAttr(record, "id")) return false;
        if (!hasIntegerDictAttr(record, "lowering")) return false;
        if (!hasIntegerDictAttr(record, "command")) return false;
        if (!hasKnownBackendKind(dictAttr(record, "backend_kind"))) return false;
        if (!hasKnownKernelCodegenKind(dictAttr(record, "kind"))) return false;
        if (!hasNonEmptyStringDictAttr(record, "operation")) return false;
        if (!verifyKernelShapeAttr(dictAttr(record, "shape"))) return false;
        if (!verifyPositiveIntegerArrayAttr(dictAttr(record, "tile"), limits.max_tile_rank)) return false;
        if (!hasIntegerDictAttr(record, "result_memory")) return false;
        const tile_memory = dictAttr(record, "tile_memory");
        if (!mlir.mlirAttributeIsNull(tile_memory) and !mlir.mlirAttributeIsAInteger(tile_memory) and !stringAttrEquals(tile_memory, "none")) return false;
        if (!verifyKernelPressureAttr(dictAttr(record, "pressure"))) return false;
        if (!verifyIntegerArrayAttr(dictAttr(record, "external_inputs"), limits.max_kernel_ref_ids, true)) return false;
        if (!verifyIntegerArrayAttr(dictAttr(record, "external_outputs"), limits.max_kernel_ref_ids, true)) return false;
        if (!verifyIntegerArrayAttr(dictAttr(record, "intermediates"), limits.max_kernel_ref_ids, false)) return false;
        if (!verifyIntegerArrayAttr(dictAttr(record, "instructions"), limits.max_kernel_ref_ids, true)) return false;
        if (!verifyIntegerArrayAttr(dictAttr(record, "costs"), limits.max_kernel_ref_ids, true)) return false;
        if (!verifyIntegerArrayAttr(dictAttr(record, "traffic"), limits.max_kernel_ref_ids, false)) return false;
        const expected_unit = dictAttr(record, "expected_unit");
        if (!mlir.mlirAttributeIsNull(expected_unit) and !mlir.mlirAttributeIsAInteger(expected_unit) and !stringAttrEquals(expected_unit, "none")) return false;
        if (!hasNonEmptyStringDictAttr(record, "reason")) return false;
        return true;
    }
};

const SchedulePlanExternalPass = struct {
    var type_id_anchor: u64 = 0;

    const Data = struct {
        construct_count: u32 = 0,
        initialize_count: u32 = 0,
        run_count: u32 = 0,
        clone_count: u32 = 0,
        destruct_count: u32 = 0,
        invalid_state: bool = false,
        missing_schedule_records: bool = false,
        invalid_record: bool = false,
    };

    fn create(data: *Data) mlir.MlirPass {
        const callbacks: mlir.MlirExternalPassCallbacks = .{
            .construct = construct,
            .destruct = destruct,
            .initialize = initialize,
            .clone = clone,
            .run = run,
        };
        return mlir.mlirCreateExternalPass(
            mlir.mlirTypeIDCreate(&type_id_anchor),
            mlirStringRef("pjrtx-schedule-plan-external"),
            mlirStringRef("pjrtx-schedule-plan-external"),
            mlirStringRef("Verifies PjRTx schedule records and marks scheduled"),
            mlirStringRef("builtin.module"),
            0,
            null,
            callbacks,
            data,
        );
    }

    fn construct(user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.construct_count += 1;
    }

    fn destruct(user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.destruct_count += 1;
    }

    fn initialize(_: mlir.MlirContext, user_data: ?*anyopaque) callconv(.c) mlir.MlirLogicalResult {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.initialize_count += 1;
        return mlir.MlirLogicalResult{ .value = 1 };
    }

    fn clone(user_data: ?*anyopaque) callconv(.c) ?*anyopaque {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.clone_count += 1;
        return user_data;
    }

    fn run(op: mlir.MlirOperation, pass: mlir.MlirExternalPass, user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.run_count += 1;

        const state_attr = getAttr(op, "pjrtx.state");
        if (!stringAttrEquals(state_attr, "codegen_planned")) {
            data.invalid_state = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        const commands_attr = getAttr(op, "pjrtx.schedule.commands");
        const overlaps_attr = getAttr(op, "pjrtx.schedule.overlaps");
        if (mlir.mlirAttributeIsNull(commands_attr) or !mlir.mlirAttributeIsAArray(commands_attr) or
            mlir.mlirAttributeIsNull(overlaps_attr) or !mlir.mlirAttributeIsAArray(overlaps_attr))
        {
            data.missing_schedule_records = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        if (!verifyScheduleCommandsAttr(commands_attr) or !verifyScheduleOverlapsAttr(overlaps_attr, mlir.mlirArrayAttrGetNumElements(commands_attr))) {
            data.invalid_record = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        const context = mlir.mlirOperationGetContext(op);
        setStringAttr(context, op, "pjrtx.state", ModuleState.scheduled.text());
        setStringAttr(context, op, "pjrtx.schedule_plan.pass", "pjrtx-schedule-plan-external");
    }

    fn verifyScheduleCommandsAttr(commands_attr: mlir.MlirAttribute) bool {
        const count = mlir.mlirArrayAttrGetNumElements(commands_attr);
        if (count <= 0 or count > limits.max_schedule_commands) return false;
        var index: isize = 0;
        while (index < count) : (index += 1) {
            const command = mlir.mlirArrayAttrGetElement(commands_attr, index);
            if (!verifyScheduleCommandAttr(command)) return false;
        }
        return true;
    }

    fn verifyScheduleCommandAttr(command: mlir.MlirAttribute) bool {
        if (mlir.mlirAttributeIsNull(command) or !mlir.mlirAttributeIsADictionary(command)) return false;
        if (!hasIntegerDictAttr(command, "id")) return false;
        if (!hasKnownCommandKind(dictAttr(command, "kind"))) return false;
        if (!hasIntegerDictAttr(command, "stream")) return false;
        if (!verifyIntegerArrayAttr(dictAttr(command, "inputs"), limits.max_schedule_refs, false)) return false;
        if (!verifyIntegerArrayAttr(dictAttr(command, "outputs"), limits.max_schedule_refs, false)) return false;
        if (!verifyScheduleDependenciesAttr(dictAttr(command, "dependencies"))) return false;
        if (!verifyIntegerArrayAttr(dictAttr(command, "lowerings"), limits.max_schedule_refs, false)) return false;
        if (!verifyIntegerArrayAttr(dictAttr(command, "costs"), limits.max_schedule_refs, false)) return false;
        return true;
    }

    fn verifyScheduleDependenciesAttr(dependencies_attr: mlir.MlirAttribute) bool {
        if (mlir.mlirAttributeIsNull(dependencies_attr) or !mlir.mlirAttributeIsAArray(dependencies_attr)) return false;
        const count = mlir.mlirArrayAttrGetNumElements(dependencies_attr);
        if (count > limits.max_schedule_dependencies) return false;
        var index: isize = 0;
        while (index < count) : (index += 1) {
            const dependency = mlir.mlirArrayAttrGetElement(dependencies_attr, index);
            if (mlir.mlirAttributeIsNull(dependency) or !mlir.mlirAttributeIsADictionary(dependency)) return false;
            if (!hasIntegerDictAttr(dependency, "command")) return false;
            if (!hasKnownDependencyKind(dictAttr(dependency, "kind"))) return false;
        }
        return true;
    }

    fn verifyScheduleOverlapsAttr(overlaps_attr: mlir.MlirAttribute, command_count: isize) bool {
        const count = mlir.mlirArrayAttrGetNumElements(overlaps_attr);
        if (count > limits.max_schedule_overlaps) return false;
        var index: isize = 0;
        while (index < count) : (index += 1) {
            const overlap = mlir.mlirArrayAttrGetElement(overlaps_attr, index);
            if (!verifyScheduleOverlapAttr(overlap, command_count)) return false;
        }
        return true;
    }

    fn verifyScheduleOverlapAttr(overlap: mlir.MlirAttribute, command_count: isize) bool {
        if (mlir.mlirAttributeIsNull(overlap) or !mlir.mlirAttributeIsADictionary(overlap)) return false;
        if (!hasIntegerDictAttr(overlap, "id")) return false;
        if (!hasKnownScheduleOverlapDecision(dictAttr(overlap, "decision"))) return false;
        if (!hasKnownScheduleOverlapKind(dictAttr(overlap, "kind"))) return false;
        const first = integerAttrValue(dictAttr(overlap, "first_command")) orelse return false;
        const second = integerAttrValue(dictAttr(overlap, "second_command")) orelse return false;
        const command_count_u32 = std.math.cast(u32, command_count) orelse return false;
        if (first >= command_count_u32 or second >= command_count_u32) return false;
        if (!hasKnownDependencyKind(dictAttr(overlap, "dependency_kind"))) return false;
        if (!hasIntegerDictAttr(overlap, "first_stream")) return false;
        if (!hasIntegerDictAttr(overlap, "second_stream")) return false;
        if (!hasNonEmptyStringDictAttr(overlap, "reason")) return false;
        return true;
    }
};

const BackendBindingExternalPass = struct {
    var type_id_anchor: u64 = 0;

    const Data = struct {
        construct_count: u32 = 0,
        initialize_count: u32 = 0,
        run_count: u32 = 0,
        clone_count: u32 = 0,
        destruct_count: u32 = 0,
        invalid_state: bool = false,
        missing_backend_bindings: bool = false,
        invalid_record: bool = false,
    };

    fn create(data: *Data) mlir.MlirPass {
        const callbacks: mlir.MlirExternalPassCallbacks = .{
            .construct = construct,
            .destruct = destruct,
            .initialize = initialize,
            .clone = clone,
            .run = run,
        };
        return mlir.mlirCreateExternalPass(
            mlir.mlirTypeIDCreate(&type_id_anchor),
            mlirStringRef("pjrtx-backend-binding-external"),
            mlirStringRef("pjrtx-backend-binding-external"),
            mlirStringRef("Verifies PjRTx backend bindings and marks backend bound"),
            mlirStringRef("builtin.module"),
            0,
            null,
            callbacks,
            data,
        );
    }

    fn construct(user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.construct_count += 1;
    }

    fn destruct(user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.destruct_count += 1;
    }

    fn initialize(_: mlir.MlirContext, user_data: ?*anyopaque) callconv(.c) mlir.MlirLogicalResult {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.initialize_count += 1;
        return mlir.MlirLogicalResult{ .value = 1 };
    }

    fn clone(user_data: ?*anyopaque) callconv(.c) ?*anyopaque {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.clone_count += 1;
        return user_data;
    }

    fn run(op: mlir.MlirOperation, pass: mlir.MlirExternalPass, user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.run_count += 1;

        const state_attr = getAttr(op, "pjrtx.state");
        if (!stringAttrEquals(state_attr, "scheduled")) {
            data.invalid_state = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        const bindings_attr = getAttr(op, "pjrtx.backend.bindings");
        if (mlir.mlirAttributeIsNull(bindings_attr) or !mlir.mlirAttributeIsAArray(bindings_attr)) {
            data.missing_backend_bindings = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        if (!verifyBackendBindingsAttr(bindings_attr)) {
            data.invalid_record = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        const context = mlir.mlirOperationGetContext(op);
        setStringAttr(context, op, "pjrtx.state", ModuleState.backend_bound.text());
        setStringAttr(context, op, "pjrtx.backend_binding.pass", "pjrtx-backend-binding-external");
    }

    fn verifyBackendBindingsAttr(bindings_attr: mlir.MlirAttribute) bool {
        const count = mlir.mlirArrayAttrGetNumElements(bindings_attr);
        if (count <= 0 or count > limits.max_backend_bindings) return false;
        var index: isize = 0;
        while (index < count) : (index += 1) {
            const binding = mlir.mlirArrayAttrGetElement(bindings_attr, index);
            if (!verifyBackendBindingAttr(binding)) return false;
        }
        return true;
    }

    fn verifyBackendBindingAttr(binding: mlir.MlirAttribute) bool {
        if (mlir.mlirAttributeIsNull(binding) or !mlir.mlirAttributeIsADictionary(binding)) return false;
        if (!hasIntegerDictAttr(binding, "id")) return false;
        if (!hasIntegerDictAttr(binding, "command")) return false;
        if (!hasKnownBackendKind(dictAttr(binding, "backend_kind"))) return false;
        if (!hasNonEmptyStringDictAttr(binding, "operation")) return false;
        if (!verifyIntegerArrayAttr(dictAttr(binding, "instructions"), limits.max_kernel_ref_ids, true)) return false;
        const expected_unit = dictAttr(binding, "expected_unit");
        if (!mlir.mlirAttributeIsNull(expected_unit) and !mlir.mlirAttributeIsAInteger(expected_unit) and !stringAttrEquals(expected_unit, "none")) return false;
        if (!verifyIntegerArrayAttr(dictAttr(binding, "costs"), limits.max_kernel_ref_ids, true)) return false;
        return true;
    }
};

const ExecutableReadyExternalPass = struct {
    var type_id_anchor: u64 = 0;

    const Data = struct {
        construct_count: u32 = 0,
        initialize_count: u32 = 0,
        run_count: u32 = 0,
        clone_count: u32 = 0,
        destruct_count: u32 = 0,
        invalid_state: bool = false,
        missing_contract: bool = false,
        invalid_contract: bool = false,
        contract_mismatch: bool = false,
    };

    fn create(data: *Data) mlir.MlirPass {
        const callbacks: mlir.MlirExternalPassCallbacks = .{
            .construct = construct,
            .destruct = destruct,
            .initialize = initialize,
            .clone = clone,
            .run = run,
        };
        return mlir.mlirCreateExternalPass(
            mlir.mlirTypeIDCreate(&type_id_anchor),
            mlirStringRef("pjrtx-executable-ready-external"),
            mlirStringRef("pjrtx-executable-ready-external"),
            mlirStringRef("Verifies PjRTx executable contract and marks executable ready"),
            mlirStringRef("builtin.module"),
            0,
            null,
            callbacks,
            data,
        );
    }

    fn construct(user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.construct_count += 1;
    }

    fn destruct(user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.destruct_count += 1;
    }

    fn initialize(_: mlir.MlirContext, user_data: ?*anyopaque) callconv(.c) mlir.MlirLogicalResult {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.initialize_count += 1;
        return mlir.MlirLogicalResult{ .value = 1 };
    }

    fn clone(user_data: ?*anyopaque) callconv(.c) ?*anyopaque {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.clone_count += 1;
        return user_data;
    }

    fn run(op: mlir.MlirOperation, pass: mlir.MlirExternalPass, user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.run_count += 1;

        const state_attr = getAttr(op, "pjrtx.state");
        if (!stringAttrEquals(state_attr, "backend_bound")) {
            data.invalid_state = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        const contract_attr = getAttr(op, "pjrtx.executable.contract");
        if (mlir.mlirAttributeIsNull(contract_attr) or !mlir.mlirAttributeIsADictionary(contract_attr)) {
            data.missing_contract = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        const contract = executableContractFromAttrNoDiag(contract_attr) orelse {
            data.invalid_contract = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        };

        if (!contractMatchesVerifiedFacts(op, contract)) {
            data.contract_mismatch = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        const context = mlir.mlirOperationGetContext(op);
        setStringAttr(context, op, "pjrtx.state", ModuleState.executable_ready.text());
        setStringAttr(context, op, "pjrtx.executable_ready.pass", "pjrtx-executable-ready-external");
    }

    fn contractMatchesVerifiedFacts(op: mlir.MlirOperation, contract: ExecutableContract) bool {
        const commands = getAttr(op, "pjrtx.schedule.commands");
        const bindings = getAttr(op, "pjrtx.backend.bindings");
        const codegen = getAttr(op, "pjrtx.codegen.records");
        if (!arrayAttrCountEquals(commands, contract.schedule_command_count)) return false;
        if (!arrayAttrCountEquals(bindings, contract.backend_binding_count)) return false;
        if (!arrayAttrCountEquals(codegen, contract.kernel_codegen_count)) return false;
        return true;
    }
};

const BackendExecutablePlanExternalPass = struct {
    var type_id_anchor: u64 = 0;

    const Data = struct {
        construct_count: u32 = 0,
        initialize_count: u32 = 0,
        run_count: u32 = 0,
        clone_count: u32 = 0,
        destruct_count: u32 = 0,
        invalid_state: bool = false,
        missing_plan: bool = false,
        invalid_plan: bool = false,
        plan_mismatch: bool = false,
    };

    fn create(data: *Data) mlir.MlirPass {
        const callbacks: mlir.MlirExternalPassCallbacks = .{
            .construct = construct,
            .destruct = destruct,
            .initialize = initialize,
            .clone = clone,
            .run = run,
        };
        return mlir.mlirCreateExternalPass(
            mlir.mlirTypeIDCreate(&type_id_anchor),
            mlirStringRef("pjrtx-backend-executable-external"),
            mlirStringRef("pjrtx-backend-executable-external"),
            mlirStringRef("Verifies PjRTx backend executable calls and marks backend executable planned"),
            mlirStringRef("builtin.module"),
            0,
            null,
            callbacks,
            data,
        );
    }

    fn construct(user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.construct_count += 1;
    }

    fn destruct(user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.destruct_count += 1;
    }

    fn initialize(_: mlir.MlirContext, user_data: ?*anyopaque) callconv(.c) mlir.MlirLogicalResult {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.initialize_count += 1;
        return mlir.MlirLogicalResult{ .value = 1 };
    }

    fn clone(user_data: ?*anyopaque) callconv(.c) ?*anyopaque {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.clone_count += 1;
        return user_data;
    }

    fn run(op: mlir.MlirOperation, pass: mlir.MlirExternalPass, user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.run_count += 1;

        const state_attr = getAttr(op, "pjrtx.state");
        if (!stringAttrEquals(state_attr, "executable_ready")) {
            data.invalid_state = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        const plan_attr = getAttr(op, "pjrtx.backend.executable");
        if (mlir.mlirAttributeIsNull(plan_attr) or !mlir.mlirAttributeIsADictionary(plan_attr)) {
            data.missing_plan = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        if (!verifyBackendExecutablePlanAttr(plan_attr)) {
            data.invalid_plan = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        if (!planMatchesVerifiedFacts(op, plan_attr)) {
            data.plan_mismatch = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        const context = mlir.mlirOperationGetContext(op);
        setStringAttr(context, op, "pjrtx.state", ModuleState.backend_executable_planned.text());
        setStringAttr(context, op, "pjrtx.backend_executable.pass", "pjrtx-backend-executable-external");
    }

    fn verifyBackendExecutablePlanAttr(plan_attr: mlir.MlirAttribute) bool {
        const backend_kind = stringAttrValue(dictAttr(plan_attr, "backend_kind")) orelse return false;
        if (!std.mem.eql(u8, backend_kind, "metal_v0") and !std.mem.eql(u8, backend_kind, "npu_v0")) return false;
        if (integerAttrValue(dictAttr(plan_attr, "command")) == null) return false;
        const operation = stringAttrValue(dictAttr(plan_attr, "operation")) orelse return false;
        if (operation.len == 0) return false;
        const calls = dictAttr(plan_attr, "calls");
        if (mlir.mlirAttributeIsNull(calls) or !mlir.mlirAttributeIsAArray(calls)) return false;
        const count = mlir.mlirArrayAttrGetNumElements(calls);
        if (count <= 0 or count > limits.max_backend_executable_calls) return false;
        var index: isize = 0;
        while (index < count) : (index += 1) {
            const call = mlir.mlirArrayAttrGetElement(calls, index);
            if (!verifyBackendExecutableCallAttr(call)) return false;
        }
        return true;
    }

    fn verifyBackendExecutableCallAttr(call: mlir.MlirAttribute) bool {
        if (mlir.mlirAttributeIsNull(call) or !mlir.mlirAttributeIsADictionary(call)) return false;
        if (integerAttrValue(dictAttr(call, "index")) == null) return false;
        if (integerAttrValue(dictAttr(call, "command")) == null) return false;
        if (integerAttrValue(dictAttr(call, "instruction")) == null) return false;
        const backend_kind = stringAttrValue(dictAttr(call, "backend_kind")) orelse return false;
        if (!std.mem.eql(u8, backend_kind, "metal_v0") and !std.mem.eql(u8, backend_kind, "npu_v0")) return false;
        const feature = stringAttrValue(dictAttr(call, "feature")) orelse return false;
        if (feature.len == 0) return false;
        const operation = stringAttrValue(dictAttr(call, "operation")) orelse return false;
        if (operation.len == 0) return false;
        if (!nonEmptyArrayAttr(dictAttr(call, "instructions"))) return false;
        if (!nonEmptyArrayAttr(dictAttr(call, "inputs"))) return false;
        if (!nonEmptyArrayAttr(dictAttr(call, "outputs"))) return false;
        if (mlir.mlirAttributeIsNull(dictAttr(call, "expected_unit"))) return false;
        return true;
    }

    fn planMatchesVerifiedFacts(op: mlir.MlirOperation, plan_attr: mlir.MlirAttribute) bool {
        const contract = executableContractFromAttrNoDiag(getAttr(op, "pjrtx.executable.contract")) orelse return false;
        const calls = dictAttr(plan_attr, "calls");
        if (!arrayAttrCountEquals(calls, contract.kernel_codegen_count)) return false;
        const plan_backend = stringAttrValue(dictAttr(plan_attr, "backend_kind")) orelse return false;
        const target_backend = stringAttrValue(getAttr(op, "pjrtx.target.kind")) orelse return false;
        if (!std.mem.eql(u8, plan_backend, target_backend)) return false;
        return true;
    }
};

const BackendKernelGraphExternalPass = struct {
    var type_id_anchor: u64 = 0;

    const Data = struct {
        construct_count: u32 = 0,
        initialize_count: u32 = 0,
        run_count: u32 = 0,
        clone_count: u32 = 0,
        destruct_count: u32 = 0,
        invalid_state: bool = false,
        missing_graph: bool = false,
        invalid_graph: bool = false,
        graph_mismatch: bool = false,
    };

    fn create(data: *Data) mlir.MlirPass {
        const callbacks: mlir.MlirExternalPassCallbacks = .{
            .construct = construct,
            .destruct = destruct,
            .initialize = initialize,
            .clone = clone,
            .run = run,
        };
        return mlir.mlirCreateExternalPass(
            mlir.mlirTypeIDCreate(&type_id_anchor),
            mlirStringRef("pjrtx-backend-kernel-graph-external"),
            mlirStringRef("pjrtx-backend-kernel-graph-external"),
            mlirStringRef("Verifies PjRTx backend kernel graph nodes and marks kernel graph planned"),
            mlirStringRef("builtin.module"),
            0,
            null,
            callbacks,
            data,
        );
    }

    fn construct(user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.construct_count += 1;
    }

    fn destruct(user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.destruct_count += 1;
    }

    fn initialize(_: mlir.MlirContext, user_data: ?*anyopaque) callconv(.c) mlir.MlirLogicalResult {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.initialize_count += 1;
        return mlir.MlirLogicalResult{ .value = 1 };
    }

    fn clone(user_data: ?*anyopaque) callconv(.c) ?*anyopaque {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.clone_count += 1;
        return user_data;
    }

    fn run(op: mlir.MlirOperation, pass: mlir.MlirExternalPass, user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.run_count += 1;

        const state_attr = getAttr(op, "pjrtx.state");
        if (!stringAttrEquals(state_attr, "backend_executable_planned")) {
            data.invalid_state = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        const graph_attr = getAttr(op, "pjrtx.backend.kernel_graph");
        if (mlir.mlirAttributeIsNull(graph_attr) or !mlir.mlirAttributeIsADictionary(graph_attr)) {
            data.missing_graph = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        if (!verifyBackendKernelGraphAttr(graph_attr)) {
            data.invalid_graph = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        if (!graphMatchesExecutableFacts(op, graph_attr)) {
            data.graph_mismatch = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        const context = mlir.mlirOperationGetContext(op);
        setStringAttr(context, op, "pjrtx.state", ModuleState.backend_kernel_graph_planned.text());
        setStringAttr(context, op, "pjrtx.backend_kernel_graph.pass", "pjrtx-backend-kernel-graph-external");
    }

    fn verifyBackendKernelGraphAttr(graph_attr: mlir.MlirAttribute) bool {
        const backend_kind = stringAttrValue(dictAttr(graph_attr, "backend_kind")) orelse return false;
        if (!std.mem.eql(u8, backend_kind, "metal_v0")) return false;
        if (integerAttrValue(dictAttr(graph_attr, "command")) == null) return false;
        const nodes = dictAttr(graph_attr, "nodes");
        if (mlir.mlirAttributeIsNull(nodes) or !mlir.mlirAttributeIsAArray(nodes)) return false;
        const node_count = mlir.mlirArrayAttrGetNumElements(nodes);
        if (node_count <= 0 or node_count > limits.max_backend_executable_calls) return false;
        var node_index: isize = 0;
        while (node_index < node_count) : (node_index += 1) {
            if (!verifyBackendKernelGraphNodeAttr(mlir.mlirArrayAttrGetElement(nodes, node_index))) return false;
        }
        const edges = dictAttr(graph_attr, "edges");
        if (mlir.mlirAttributeIsNull(edges) or !mlir.mlirAttributeIsAArray(edges)) return false;
        const edge_count = mlir.mlirArrayAttrGetNumElements(edges);
        if (edge_count < 0 or edge_count > limits.max_backend_kernel_graph_edges) return false;
        var edge_index: isize = 0;
        while (edge_index < edge_count) : (edge_index += 1) {
            if (!verifyBackendKernelGraphEdgeAttr(mlir.mlirArrayAttrGetElement(edges, edge_index), node_count)) return false;
        }
        return true;
    }

    fn verifyBackendKernelGraphNodeAttr(node: mlir.MlirAttribute) bool {
        if (mlir.mlirAttributeIsNull(node) or !mlir.mlirAttributeIsADictionary(node)) return false;
        if (integerAttrValue(dictAttr(node, "index")) == null) return false;
        if (integerAttrValue(dictAttr(node, "call")) == null) return false;
        if (integerAttrValue(dictAttr(node, "instruction")) == null) return false;
        const feature = stringAttrValue(dictAttr(node, "feature")) orelse return false;
        if (feature.len == 0) return false;
        const operation = stringAttrValue(dictAttr(node, "operation")) orelse return false;
        if (operation.len == 0) return false;
        if (!nonEmptyArrayAttr(dictAttr(node, "instructions"))) return false;
        if (!nonEmptyArrayAttr(dictAttr(node, "inputs"))) return false;
        if (!nonEmptyArrayAttr(dictAttr(node, "outputs"))) return false;
        const output_type = dictAttr(node, "output_type");
        if (mlir.mlirAttributeIsNull(output_type) or !mlir.mlirAttributeIsADictionary(output_type)) return false;
        const dtype = stringAttrValue(dictAttr(output_type, "dtype")) orelse return false;
        if (dtype.len == 0) return false;
        const layout = stringAttrValue(dictAttr(output_type, "layout")) orelse return false;
        if (layout.len == 0) return false;
        if (!nonEmptyArrayAttr(dictAttr(output_type, "dims"))) return false;
        const attributes = stringAttrValue(dictAttr(node, "attributes")) orelse return false;
        return attributes.len > 0;
    }

    fn verifyBackendKernelGraphEdgeAttr(edge: mlir.MlirAttribute, node_count: isize) bool {
        if (mlir.mlirAttributeIsNull(edge) or !mlir.mlirAttributeIsADictionary(edge)) return false;
        _ = integerAttrValue(dictAttr(edge, "value")) orelse return false;
        const src = integerAttrValue(dictAttr(edge, "src")) orelse return false;
        const dst = integerAttrValue(dictAttr(edge, "dst")) orelse return false;
        const src_isize = std.math.cast(isize, src) orelse return false;
        const dst_isize = std.math.cast(isize, dst) orelse return false;
        return src_isize < node_count and dst_isize < node_count;
    }

    fn graphMatchesExecutableFacts(op: mlir.MlirOperation, graph_attr: mlir.MlirAttribute) bool {
        const executable = getAttr(op, "pjrtx.backend.executable");
        if (mlir.mlirAttributeIsNull(executable) or !mlir.mlirAttributeIsADictionary(executable)) return false;
        const executable_calls = dictAttr(executable, "calls");
        const graph_nodes = dictAttr(graph_attr, "nodes");
        if (!mlir.mlirAttributeIsAArray(executable_calls) or !mlir.mlirAttributeIsAArray(graph_nodes)) return false;
        if (mlir.mlirArrayAttrGetNumElements(executable_calls) != mlir.mlirArrayAttrGetNumElements(graph_nodes)) return false;
        const graph_backend = stringAttrValue(dictAttr(graph_attr, "backend_kind")) orelse return false;
        const executable_backend = stringAttrValue(dictAttr(executable, "backend_kind")) orelse return false;
        if (!std.mem.eql(u8, graph_backend, executable_backend)) return false;
        return integerAttrValue(dictAttr(graph_attr, "command")) == integerAttrValue(dictAttr(executable, "command"));
    }
};

const RuntimeAllocationExternalPass = struct {
    var type_id_anchor: u64 = 0;

    const Data = struct {
        construct_count: u32 = 0,
        initialize_count: u32 = 0,
        run_count: u32 = 0,
        clone_count: u32 = 0,
        destruct_count: u32 = 0,
        invalid_state: bool = false,
        missing_plan: bool = false,
        invalid_plan: bool = false,
    };

    fn create(data: *Data) mlir.MlirPass {
        const callbacks: mlir.MlirExternalPassCallbacks = .{
            .construct = construct,
            .destruct = destruct,
            .initialize = initialize,
            .clone = clone,
            .run = run,
        };
        return mlir.mlirCreateExternalPass(
            mlir.mlirTypeIDCreate(&type_id_anchor),
            mlirStringRef("pjrtx-runtime-allocation-external"),
            mlirStringRef("pjrtx-runtime-allocation-external"),
            mlirStringRef("Verifies PjRTx runtime allocation reservations and marks allocation planned"),
            mlirStringRef("builtin.module"),
            0,
            null,
            callbacks,
            data,
        );
    }

    fn construct(user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.construct_count += 1;
    }

    fn destruct(user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.destruct_count += 1;
    }

    fn initialize(_: mlir.MlirContext, user_data: ?*anyopaque) callconv(.c) mlir.MlirLogicalResult {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.initialize_count += 1;
        return mlir.MlirLogicalResult{ .value = 1 };
    }

    fn clone(user_data: ?*anyopaque) callconv(.c) ?*anyopaque {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.clone_count += 1;
        return user_data;
    }

    fn run(op: mlir.MlirOperation, pass: mlir.MlirExternalPass, user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.run_count += 1;

        const state_attr = getAttr(op, "pjrtx.state");
        if (!stringAttrEquals(state_attr, "backend_executable_planned") and !stringAttrEquals(state_attr, "backend_kernel_graph_planned")) {
            data.invalid_state = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        const plan_attr = getAttr(op, "pjrtx.runtime.allocation");
        if (mlir.mlirAttributeIsNull(plan_attr) or !mlir.mlirAttributeIsADictionary(plan_attr)) {
            data.missing_plan = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        if (!verifyRuntimeAllocationAttr(plan_attr)) {
            data.invalid_plan = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        const context = mlir.mlirOperationGetContext(op);
        setStringAttr(context, op, "pjrtx.state", ModuleState.runtime_allocation_planned.text());
        setStringAttr(context, op, "pjrtx.runtime_allocation.pass", "pjrtx-runtime-allocation-external");
    }

    fn verifyRuntimeAllocationAttr(plan_attr: mlir.MlirAttribute) bool {
        const allocations = dictAttr(plan_attr, "allocations");
        if (mlir.mlirAttributeIsNull(allocations) or !mlir.mlirAttributeIsAArray(allocations)) return false;
        const allocation_count = mlir.mlirArrayAttrGetNumElements(allocations);
        if (allocation_count <= 0 or allocation_count > limits.max_runtime_allocations) return false;

        const uses = dictAttr(plan_attr, "command_buffer_uses");
        if (mlir.mlirAttributeIsNull(uses) or !mlir.mlirAttributeIsAArray(uses)) return false;
        const use_count = mlir.mlirArrayAttrGetNumElements(uses);
        if (use_count <= 0 or use_count > limits.max_runtime_buffer_uses) return false;

        if (u128FromStringAttrNoDiag(dictAttr(plan_attr, "peak_device_bytes")) == null) return false;

        var allocation_index: isize = 0;
        while (allocation_index < allocation_count) : (allocation_index += 1) {
            if (!verifyRuntimeAllocationEntryAttr(mlir.mlirArrayAttrGetElement(allocations, allocation_index))) return false;
        }
        var use_index: isize = 0;
        while (use_index < use_count) : (use_index += 1) {
            if (!verifyRuntimeBufferUseAttr(mlir.mlirArrayAttrGetElement(uses, use_index), allocation_count)) return false;
        }
        return true;
    }

    fn verifyRuntimeAllocationEntryAttr(allocation: mlir.MlirAttribute) bool {
        if (mlir.mlirAttributeIsNull(allocation) or !mlir.mlirAttributeIsADictionary(allocation)) return false;
        _ = integerAttrValue(dictAttr(allocation, "index")) orelse return false;
        _ = integerAttrValue(dictAttr(allocation, "value")) orelse return false;
        const placement = stringAttrValue(dictAttr(allocation, "placement")) orelse return false;
        if (!std.mem.eql(u8, placement, "host") and !std.mem.eql(u8, placement, "device")) return false;
        _ = integerAttrValue(dictAttr(allocation, "memory")) orelse return false;
        if (u128FromStringAttrNoDiag(dictAttr(allocation, "bytes")) == null) return false;
        _ = integerAttrValue(dictAttr(allocation, "first_command")) orelse return false;
        _ = integerAttrValue(dictAttr(allocation, "last_command")) orelse return false;
        return true;
    }

    fn verifyRuntimeBufferUseAttr(use: mlir.MlirAttribute, allocation_count: isize) bool {
        if (mlir.mlirAttributeIsNull(use) or !mlir.mlirAttributeIsADictionary(use)) return false;
        _ = integerAttrValue(dictAttr(use, "command")) orelse return false;
        const buffer = integerAttrValue(dictAttr(use, "buffer")) orelse return false;
        const buffer_index = std.math.cast(isize, buffer) orelse return false;
        if (buffer_index >= allocation_count) return false;
        const access = stringAttrValue(dictAttr(use, "access")) orelse return false;
        return std.mem.eql(u8, access, "read") or std.mem.eql(u8, access, "write");
    }
};

const RuntimeStreamExternalPass = struct {
    var type_id_anchor: u64 = 0;

    const Data = struct {
        construct_count: u32 = 0,
        initialize_count: u32 = 0,
        run_count: u32 = 0,
        clone_count: u32 = 0,
        destruct_count: u32 = 0,
        invalid_state: bool = false,
        missing_plan: bool = false,
        invalid_plan: bool = false,
    };

    fn create(data: *Data) mlir.MlirPass {
        const callbacks: mlir.MlirExternalPassCallbacks = .{
            .construct = construct,
            .destruct = destruct,
            .initialize = initialize,
            .clone = clone,
            .run = run,
        };
        return mlir.mlirCreateExternalPass(
            mlir.mlirTypeIDCreate(&type_id_anchor),
            mlirStringRef("pjrtx-runtime-stream-external"),
            mlirStringRef("pjrtx-runtime-stream-external"),
            mlirStringRef("Verifies PjRTx runtime stream events and marks streams planned"),
            mlirStringRef("builtin.module"),
            0,
            null,
            callbacks,
            data,
        );
    }

    fn construct(user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.construct_count += 1;
    }

    fn destruct(user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.destruct_count += 1;
    }

    fn initialize(_: mlir.MlirContext, user_data: ?*anyopaque) callconv(.c) mlir.MlirLogicalResult {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.initialize_count += 1;
        return mlir.MlirLogicalResult{ .value = 1 };
    }

    fn clone(user_data: ?*anyopaque) callconv(.c) ?*anyopaque {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.clone_count += 1;
        return user_data;
    }

    fn run(op: mlir.MlirOperation, pass: mlir.MlirExternalPass, user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.run_count += 1;

        const state_attr = getAttr(op, "pjrtx.state");
        if (!stringAttrEquals(state_attr, "runtime_allocation_planned")) {
            data.invalid_state = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        const plan_attr = getAttr(op, "pjrtx.runtime.streams");
        if (mlir.mlirAttributeIsNull(plan_attr) or !mlir.mlirAttributeIsAArray(plan_attr)) {
            data.missing_plan = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        if (!verifyRuntimeStreamAttr(plan_attr)) {
            data.invalid_plan = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        const context = mlir.mlirOperationGetContext(op);
        setStringAttr(context, op, "pjrtx.state", ModuleState.runtime_stream_planned.text());
        setStringAttr(context, op, "pjrtx.runtime_stream.pass", "pjrtx-runtime-stream-external");
    }

    fn verifyRuntimeStreamAttr(plan_attr: mlir.MlirAttribute) bool {
        const step_count = mlir.mlirArrayAttrGetNumElements(plan_attr);
        if (step_count <= 0 or step_count > limits.max_runtime_stream_steps) return false;
        var index: isize = 0;
        while (index < step_count) : (index += 1) {
            if (!verifyRuntimeStreamStepAttr(mlir.mlirArrayAttrGetElement(plan_attr, index))) return false;
        }
        return true;
    }

    fn verifyRuntimeStreamStepAttr(step: mlir.MlirAttribute) bool {
        if (mlir.mlirAttributeIsNull(step) or !mlir.mlirAttributeIsADictionary(step)) return false;
        _ = integerAttrValue(dictAttr(step, "command")) orelse return false;
        _ = integerAttrValue(dictAttr(step, "stream")) orelse return false;
        _ = integerAttrValue(dictAttr(step, "start_event")) orelse return false;
        _ = integerAttrValue(dictAttr(step, "done_event")) orelse return false;
        const waits = dictAttr(step, "wait_events");
        if (mlir.mlirAttributeIsNull(waits) or !mlir.mlirAttributeIsAArray(waits)) return false;
        return mlir.mlirArrayAttrGetNumElements(waits) <= limits.max_runtime_wait_events;
    }
};

const RuntimeProfileExternalPass = struct {
    var type_id_anchor: u64 = 0;

    const Data = struct {
        construct_count: u32 = 0,
        initialize_count: u32 = 0,
        run_count: u32 = 0,
        clone_count: u32 = 0,
        destruct_count: u32 = 0,
        invalid_state: bool = false,
        missing_profile: bool = false,
        invalid_profile: bool = false,
    };

    fn create(data: *Data) mlir.MlirPass {
        const callbacks: mlir.MlirExternalPassCallbacks = .{
            .construct = construct,
            .destruct = destruct,
            .initialize = initialize,
            .clone = clone,
            .run = run,
        };
        return mlir.mlirCreateExternalPass(
            mlir.mlirTypeIDCreate(&type_id_anchor),
            mlirStringRef("pjrtx-runtime-profile-external"),
            mlirStringRef("pjrtx-runtime-profile-external"),
            mlirStringRef("Verifies PjRTx runtime profile events and marks runtime profiled"),
            mlirStringRef("builtin.module"),
            0,
            null,
            callbacks,
            data,
        );
    }

    fn construct(user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.construct_count += 1;
    }

    fn destruct(user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.destruct_count += 1;
    }

    fn initialize(_: mlir.MlirContext, user_data: ?*anyopaque) callconv(.c) mlir.MlirLogicalResult {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.initialize_count += 1;
        return mlir.MlirLogicalResult{ .value = 1 };
    }

    fn clone(user_data: ?*anyopaque) callconv(.c) ?*anyopaque {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.clone_count += 1;
        return user_data;
    }

    fn run(op: mlir.MlirOperation, pass: mlir.MlirExternalPass, user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.run_count += 1;

        const state_attr = getAttr(op, "pjrtx.state");
        if (!stringAttrEquals(state_attr, "runtime_stream_planned")) {
            data.invalid_state = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        const profile_attr = getAttr(op, "pjrtx.runtime.profile_events");
        if (mlir.mlirAttributeIsNull(profile_attr) or !mlir.mlirAttributeIsAArray(profile_attr)) {
            data.missing_profile = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        if (!verifyRuntimeProfileAttr(profile_attr)) {
            data.invalid_profile = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        const context = mlir.mlirOperationGetContext(op);
        setStringAttr(context, op, "pjrtx.state", ModuleState.runtime_profiled.text());
        setStringAttr(context, op, "pjrtx.runtime_profile.pass", "pjrtx-runtime-profile-external");
    }

    fn verifyRuntimeProfileAttr(profile_attr: mlir.MlirAttribute) bool {
        const event_count = mlir.mlirArrayAttrGetNumElements(profile_attr);
        if (event_count <= 0 or event_count > limits.max_runtime_profile_events) return false;
        var index: isize = 0;
        while (index < event_count) : (index += 1) {
            if (!verifyRuntimeProfileEventAttr(mlir.mlirArrayAttrGetElement(profile_attr, index))) return false;
        }
        return true;
    }

    fn verifyRuntimeProfileEventAttr(event: mlir.MlirAttribute) bool {
        if (mlir.mlirAttributeIsNull(event) or !mlir.mlirAttributeIsADictionary(event)) return false;
        _ = integerAttrValue(dictAttr(event, "index")) orelse return false;
        const command = dictAttr(event, "command");
        if (mlir.mlirAttributeIsNull(command)) return false;
        const instructions = dictAttr(event, "instructions");
        if (mlir.mlirAttributeIsNull(instructions) or !mlir.mlirAttributeIsAArray(instructions)) return false;
        const kind = stringAttrValue(dictAttr(event, "kind")) orelse return false;
        if (kind.len == 0) return false;
        if (!validU64StringAttr(dictAttr(event, "start_ns"))) return false;
        if (!validU64StringAttr(dictAttr(event, "duration_ns"))) return false;
        if (u128FromStringAttrNoDiag(dictAttr(event, "bytes")) == null) return false;
        if (u128FromStringAttrNoDiag(dictAttr(event, "logical_ops")) == null) return false;
        const status = stringAttrValue(dictAttr(event, "status")) orelse return false;
        if (!std.mem.eql(u8, status, "ok") and !std.mem.eql(u8, status, "failed")) return false;
        _ = boolAttrValue(dictAttr(event, "forced_sync")) orelse return false;
        return true;
    }
};

const RuntimeProfileJoinExternalPass = struct {
    var type_id_anchor: u64 = 0;

    const Data = struct {
        construct_count: u32 = 0,
        initialize_count: u32 = 0,
        run_count: u32 = 0,
        clone_count: u32 = 0,
        destruct_count: u32 = 0,
        invalid_state: bool = false,
        missing_profile: bool = false,
        missing_joins: bool = false,
        invalid_joins: bool = false,
    };

    fn create(data: *Data) mlir.MlirPass {
        const callbacks: mlir.MlirExternalPassCallbacks = .{
            .construct = construct,
            .destruct = destruct,
            .initialize = initialize,
            .clone = clone,
            .run = run,
        };
        return mlir.mlirCreateExternalPass(
            mlir.mlirTypeIDCreate(&type_id_anchor),
            mlirStringRef("pjrtx-runtime-profile-join-external"),
            mlirStringRef("pjrtx-runtime-profile-join-external"),
            mlirStringRef("Verifies PjRTx runtime profile joins and marks runtime profile joined"),
            mlirStringRef("builtin.module"),
            0,
            null,
            callbacks,
            data,
        );
    }

    fn construct(user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.construct_count += 1;
    }

    fn destruct(user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.destruct_count += 1;
    }

    fn initialize(_: mlir.MlirContext, user_data: ?*anyopaque) callconv(.c) mlir.MlirLogicalResult {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.initialize_count += 1;
        return mlir.MlirLogicalResult{ .value = 1 };
    }

    fn clone(user_data: ?*anyopaque) callconv(.c) ?*anyopaque {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.clone_count += 1;
        return user_data;
    }

    fn run(op: mlir.MlirOperation, pass: mlir.MlirExternalPass, user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.run_count += 1;

        const state_attr = getAttr(op, "pjrtx.state");
        if (!stringAttrEquals(state_attr, "runtime_profiled")) {
            data.invalid_state = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        const profile_attr = getAttr(op, "pjrtx.runtime.profile_events");
        if (mlir.mlirAttributeIsNull(profile_attr) or !mlir.mlirAttributeIsAArray(profile_attr)) {
            data.missing_profile = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        const joins_attr = getAttr(op, "pjrtx.runtime.profile_joins");
        if (mlir.mlirAttributeIsNull(joins_attr) or !mlir.mlirAttributeIsAArray(joins_attr)) {
            data.missing_joins = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        if (!verifyRuntimeProfileJoinAttr(joins_attr)) {
            data.invalid_joins = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        const context = mlir.mlirOperationGetContext(op);
        setStringAttr(context, op, "pjrtx.state", ModuleState.runtime_profile_joined.text());
        setStringAttr(context, op, "pjrtx.runtime_profile_join.pass", "pjrtx-runtime-profile-join-external");
    }

    fn verifyRuntimeProfileJoinAttr(joins_attr: mlir.MlirAttribute) bool {
        const join_count = mlir.mlirArrayAttrGetNumElements(joins_attr);
        if (join_count <= 0 or join_count > limits.max_runtime_profile_joins) return false;
        var index: isize = 0;
        while (index < join_count) : (index += 1) {
            if (!verifyRuntimeProfileJoinEntryAttr(mlir.mlirArrayAttrGetElement(joins_attr, index))) return false;
        }
        return true;
    }

    fn verifyRuntimeProfileJoinEntryAttr(join: mlir.MlirAttribute) bool {
        if (mlir.mlirAttributeIsNull(join) or !mlir.mlirAttributeIsADictionary(join)) return false;
        _ = integerAttrValue(dictAttr(join, "index")) orelse return false;
        const subject_kind = stringAttrValue(dictAttr(join, "subject_kind")) orelse return false;
        if (subject_kind.len == 0) return false;
        _ = integerAttrValue(dictAttr(join, "subject_id")) orelse return false;
        const command = dictAttr(join, "command");
        if (mlir.mlirAttributeIsNull(command)) return false;
        const instructions = dictAttr(join, "instructions");
        if (mlir.mlirAttributeIsNull(instructions) or !mlir.mlirAttributeIsAArray(instructions)) return false;
        const events = dictAttr(join, "events");
        if (mlir.mlirAttributeIsNull(events) or !mlir.mlirAttributeIsAArray(events)) return false;
        const event_count = mlir.mlirArrayAttrGetNumElements(events);
        return event_count > 0 and event_count <= limits.max_runtime_profile_join_events;
    }
};

const BackendProfileJoinExternalPass = struct {
    var type_id_anchor: u64 = 0;

    const Data = struct {
        construct_count: u32 = 0,
        initialize_count: u32 = 0,
        run_count: u32 = 0,
        clone_count: u32 = 0,
        destruct_count: u32 = 0,
        invalid_state: bool = false,
        missing_executable: bool = false,
        missing_profile: bool = false,
        missing_joins: bool = false,
        invalid_joins: bool = false,
    };

    fn create(data: *Data) mlir.MlirPass {
        const callbacks: mlir.MlirExternalPassCallbacks = .{
            .construct = construct,
            .destruct = destruct,
            .initialize = initialize,
            .clone = clone,
            .run = run,
        };
        return mlir.mlirCreateExternalPass(
            mlir.mlirTypeIDCreate(&type_id_anchor),
            mlirStringRef("pjrtx-backend-profile-join-external"),
            mlirStringRef("pjrtx-backend-profile-join-external"),
            mlirStringRef("Verifies PjRTx backend call profile joins and marks backend profile joined"),
            mlirStringRef("builtin.module"),
            0,
            null,
            callbacks,
            data,
        );
    }

    fn construct(user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.construct_count += 1;
    }

    fn destruct(user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.destruct_count += 1;
    }

    fn initialize(_: mlir.MlirContext, user_data: ?*anyopaque) callconv(.c) mlir.MlirLogicalResult {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.initialize_count += 1;
        return mlir.MlirLogicalResult{ .value = 1 };
    }

    fn clone(user_data: ?*anyopaque) callconv(.c) ?*anyopaque {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.clone_count += 1;
        return user_data;
    }

    fn run(op: mlir.MlirOperation, pass: mlir.MlirExternalPass, user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.run_count += 1;

        const state_attr = getAttr(op, "pjrtx.state");
        if (!stringAttrEquals(state_attr, "runtime_profile_joined")) {
            data.invalid_state = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        const executable_attr = getAttr(op, "pjrtx.backend.executable");
        if (mlir.mlirAttributeIsNull(executable_attr) or !mlir.mlirAttributeIsADictionary(executable_attr)) {
            data.missing_executable = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        const profile_attr = getAttr(op, "pjrtx.runtime.profile_events");
        if (mlir.mlirAttributeIsNull(profile_attr) or !mlir.mlirAttributeIsAArray(profile_attr)) {
            data.missing_profile = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        const joins_attr = getAttr(op, "pjrtx.backend.profile_joins");
        if (mlir.mlirAttributeIsNull(joins_attr) or !mlir.mlirAttributeIsAArray(joins_attr)) {
            data.missing_joins = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        if (!verifyBackendProfileJoinAttr(joins_attr)) {
            data.invalid_joins = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        const context = mlir.mlirOperationGetContext(op);
        setStringAttr(context, op, "pjrtx.state", ModuleState.backend_profile_joined.text());
        setStringAttr(context, op, "pjrtx.backend_profile_join.pass", "pjrtx-backend-profile-join-external");
    }

    fn verifyBackendProfileJoinAttr(joins_attr: mlir.MlirAttribute) bool {
        const join_count = mlir.mlirArrayAttrGetNumElements(joins_attr);
        if (join_count <= 0 or join_count > limits.max_backend_profile_joins) return false;
        var index: isize = 0;
        while (index < join_count) : (index += 1) {
            if (!verifyBackendProfileJoinEntryAttr(mlir.mlirArrayAttrGetElement(joins_attr, index))) return false;
        }
        return true;
    }

    fn verifyBackendProfileJoinEntryAttr(join: mlir.MlirAttribute) bool {
        if (mlir.mlirAttributeIsNull(join) or !mlir.mlirAttributeIsADictionary(join)) return false;
        _ = integerAttrValue(dictAttr(join, "index")) orelse return false;
        _ = integerAttrValue(dictAttr(join, "call")) orelse return false;
        _ = integerAttrValue(dictAttr(join, "command")) orelse return false;
        _ = integerAttrValue(dictAttr(join, "event")) orelse return false;
        const instructions = dictAttr(join, "instructions");
        if (mlir.mlirAttributeIsNull(instructions) or !mlir.mlirAttributeIsAArray(instructions)) return false;
        const instruction_count = mlir.mlirArrayAttrGetNumElements(instructions);
        return instruction_count > 0 and instruction_count <= limits.max_kernel_ref_ids;
    }
};

fn transitionModuleState(session: *MlirSession, next: ModuleState, diagnostics: *std.Io.Writer) !void {
    const current = moduleState(session) orelse {
        try diagnostics.print(
            "pass=mlir_state_transition feature=state reason=missing module state next={s}\n",
            .{next.text()},
        );
        return MlirStateError.InvalidStateTransition;
    };
    if (!isValidTransition(current, next)) {
        try diagnostics.print(
            "pass=mlir_state_transition feature=state reason=invalid transition from={s} to={s}\n",
            .{ current.text(), next.text() },
        );
        return MlirStateError.InvalidStateTransition;
    }
    setModuleStateUnchecked(session, next);
}

fn isValidTransition(current: ModuleState, next: ModuleState) bool {
    return switch (current) {
        .imported => next == .target_attached,
        .target_attached => next == .target_legal,
        .target_legal => next == .fusion_planned,
        .fusion_planned => next == .placement_planned,
        .placement_planned => next == .collectives_planned,
        .collectives_planned => next == .lowering_planned,
        .lowering_planned => next == .performance_modeled,
        .performance_modeled => next == .codegen_planned,
        .codegen_planned => next == .scheduled,
        .scheduled => next == .backend_bound,
        .backend_bound => next == .executable_ready,
        .executable_ready => next == .backend_executable_planned,
        .backend_executable_planned => next == .backend_kernel_graph_planned,
        .backend_kernel_graph_planned => next == .runtime_allocation_planned,
        .runtime_allocation_planned => next == .runtime_stream_planned,
        .runtime_stream_planned => next == .runtime_profiled,
        .runtime_profiled => next == .runtime_profile_joined,
        .runtime_profile_joined => next == .backend_profile_joined,
        .backend_profile_joined => false,
    };
}

fn setModuleStateUnchecked(session: *MlirSession, state: ModuleState) void {
    setStringAttr(session.context, session.moduleOperation(), "pjrtx.state", state.text());
}

fn verifyFusionGroups(groups: []const compiler_facts.FusionGroup, diagnostics: *std.Io.Writer) !void {
    for (groups) |group| {
        if (group.kind.len == 0) {
            try diagnostics.print("pass=fusion_plan_verify feature=fusion reason=missing kind fusion={d}\n", .{group.index});
            return MlirStateError.InvalidFusionPlan;
        }
        if (group.graph_instruction_ids.len == 0) {
            try diagnostics.print("pass=fusion_plan_verify feature=fusion reason=missing instruction IDs fusion={d}\n", .{group.index});
            return MlirStateError.InvalidFusionPlan;
        }
        if (group.decision == .rejected and group.reason.len == 0) {
            try diagnostics.print("pass=fusion_plan_verify feature=fusion reason=rejected fusion requires reason fusion={d}\n", .{group.index});
            return MlirStateError.InvalidFusionPlan;
        }
        if (group.pressure_delta.fused_live_bytes >= group.pressure_delta.split_peak_live_bytes) {
            const expected = group.pressure_delta.fused_live_bytes - group.pressure_delta.split_peak_live_bytes;
            if (group.pressure_delta.additional_live_bytes != expected) {
                try diagnostics.print("pass=fusion_plan_verify feature=pressure reason=additional live bytes mismatch fusion={d}\n", .{group.index});
                return MlirStateError.InvalidFusionPlan;
            }
        }
    }
}

fn verifyPlacementRecords(records: []const compiler_facts.PlacementRecord, diagnostics: *std.Io.Writer) !void {
    if (records.len > limits.max_placement_records) {
        try diagnostics.writeAll("pass=placement_plan_verify feature=placement reason=too many placement records\n");
        return MlirStateError.InvalidPlacementPlan;
    }
    for (records, 0..) |record, expected_index| {
        const expected_record_index: u32 = std.math.cast(u32, expected_index) orelse {
            try diagnostics.writeAll("pass=placement_plan_verify feature=placement reason=placement index exceeds u32\n");
            return MlirStateError.InvalidPlacementPlan;
        };
        if (record.index != expected_record_index) {
            try diagnostics.print("pass=placement_plan_verify feature=placement reason=placement index mismatch expected={d} actual={d}\n", .{ expected_record_index, record.index });
            return MlirStateError.InvalidPlacementPlan;
        }
        if (record.output_value_ids.len == 0 or record.output_value_ids.len > limits.max_placement_outputs) {
            try diagnostics.print("pass=placement_plan_verify feature=outputs reason=invalid output count placement={d}\n", .{record.index});
            return MlirStateError.InvalidPlacementPlan;
        }
        if (record.logical_tile_shape.len == 0 or record.logical_tile_shape.len > limits.max_tile_rank) {
            try diagnostics.print("pass=placement_plan_verify feature=tile reason=invalid tile rank placement={d}\n", .{record.index});
            return MlirStateError.InvalidPlacementPlan;
        }
        for (record.logical_tile_shape) |dim| {
            if (dim <= 0) {
                try diagnostics.print("pass=placement_plan_verify feature=tile reason=non-positive tile dimension placement={d}\n", .{record.index});
                return MlirStateError.InvalidPlacementPlan;
            }
        }
        if (record.reason.len == 0) {
            try diagnostics.print("pass=placement_plan_verify feature=reason reason=placement reason is empty placement={d}\n", .{record.index});
            return MlirStateError.InvalidPlacementPlan;
        }
    }
}

fn verifyCollectivePlanRecords(records: []const compiler_facts.CollectivePlanRecord, diagnostics: *std.Io.Writer) !void {
    if (records.len == 0 or records.len > limits.max_collective_records) {
        try diagnostics.writeAll("pass=collective_plan_verify feature=collective reason=invalid collective record count\n");
        return MlirStateError.InvalidCollectivePlan;
    }
    for (records, 0..) |record, expected_index| {
        const expected_record_index: u32 = std.math.cast(u32, expected_index) orelse {
            try diagnostics.writeAll("pass=collective_plan_verify feature=collective reason=collective index exceeds u32\n");
            return MlirStateError.InvalidCollectivePlan;
        };
        if (record.index != expected_record_index) {
            try diagnostics.print("pass=collective_plan_verify feature=collective reason=collective index mismatch expected={d} actual={d}\n", .{ expected_record_index, record.index });
            return MlirStateError.InvalidCollectivePlan;
        }
        if (record.reason.len == 0) {
            try diagnostics.print("pass=collective_plan_verify feature=reason reason=collective reason is empty collective={d}\n", .{record.index});
            return MlirStateError.InvalidCollectivePlan;
        }
        if (record.decision == .no_collectives and (record.lowered_collective_count != 0 or record.unsupported_collective_count != 0 or record.estimated_bytes != 0)) {
            try diagnostics.print("pass=collective_plan_verify feature=decision reason=no_collectives record carries work collective={d}\n", .{record.index});
            return MlirStateError.InvalidCollectivePlan;
        }
    }
}

fn verifyKernelCodegenRecords(records: []const core.KernelCodegenRecord, diagnostics: *std.Io.Writer) !void {
    if (records.len == 0 or records.len > limits.max_kernel_codegen_records) {
        try diagnostics.writeAll("pass=codegen_plan_verify feature=codegen reason=invalid codegen record count\n");
        return MlirStateError.InvalidKernelCodegenPlan;
    }
    for (records, 0..) |record, expected_index| {
        const expected_record_index: u32 = std.math.cast(u32, expected_index) orelse {
            try diagnostics.writeAll("pass=codegen_plan_verify feature=codegen reason=codegen index exceeds u32\n");
            return MlirStateError.InvalidKernelCodegenPlan;
        };
        if (record.id.index != expected_record_index) {
            try diagnostics.print("pass=codegen_plan_verify feature=codegen reason=codegen index mismatch expected={d} actual={d}\n", .{ expected_record_index, record.id.index });
            return MlirStateError.InvalidKernelCodegenPlan;
        }
        if (record.operation.len == 0 or record.reason.len == 0) {
            try diagnostics.print("pass=codegen_plan_verify feature=codegen reason=missing operation or reason codegen={d}\n", .{record.id.index});
            return MlirStateError.InvalidKernelCodegenPlan;
        }
        if (record.logical_tile_shape.len == 0 or record.logical_tile_shape.len > limits.max_tile_rank) {
            try diagnostics.print("pass=codegen_plan_verify feature=tile reason=invalid tile rank codegen={d}\n", .{record.id.index});
            return MlirStateError.InvalidKernelCodegenPlan;
        }
        for (record.logical_tile_shape) |dim| {
            if (dim <= 0) {
                try diagnostics.print("pass=codegen_plan_verify feature=tile reason=non-positive tile dimension codegen={d}\n", .{record.id.index});
                return MlirStateError.InvalidKernelCodegenPlan;
            }
        }
        if (record.external_input_ids.len > limits.max_kernel_ref_ids or
            record.external_output_ids.len > limits.max_kernel_ref_ids or
            record.intermediate_value_ids.len > limits.max_kernel_ref_ids or
            record.graph_instruction_ids.len > limits.max_kernel_ref_ids or
            record.cost_ledger_ids.len > limits.max_kernel_ref_ids or
            record.memory_traffic_ids.len > limits.max_kernel_ref_ids)
        {
            try diagnostics.print("pass=codegen_plan_verify feature=refs reason=too many codegen references codegen={d}\n", .{record.id.index});
            return MlirStateError.InvalidKernelCodegenPlan;
        }
    }
}

fn verifyLoweringRecords(records: []const compiler_facts.LoweringRecord, cost_count: usize, diagnostics: *std.Io.Writer) !void {
    if (records.len == 0 or records.len > limits.max_lowering_records) {
        try diagnostics.writeAll("pass=lowering_plan_verify feature=lowering reason=invalid lowering record count\n");
        return MlirStateError.InvalidLoweringPlan;
    }
    for (records, 0..) |record, expected_index| {
        const expected_record_index: u32 = std.math.cast(u32, expected_index) orelse {
            try diagnostics.writeAll("pass=lowering_plan_verify feature=lowering reason=lowering index exceeds u32\n");
            return MlirStateError.InvalidLoweringPlan;
        };
        if (record.id.index != expected_record_index) {
            try diagnostics.print("pass=lowering_plan_verify feature=lowering reason=lowering index mismatch expected={d} actual={d}\n", .{ expected_record_index, record.id.index });
            return MlirStateError.InvalidLoweringPlan;
        }
        if (record.graph_instruction_ids.len == 0 or record.graph_instruction_ids.len > limits.max_kernel_ref_ids or record.cost_ledger_ids.len == 0 or record.cost_ledger_ids.len > limits.max_kernel_ref_ids or record.rejected_alternatives.len > limits.max_kernel_ref_ids or record.reason.len == 0) {
            try diagnostics.print("pass=lowering_plan_verify feature=lowering reason=invalid lowering metadata lowering={d}\n", .{record.id.index});
            return MlirStateError.InvalidLoweringPlan;
        }
        for (record.cost_ledger_ids) |cost_id| {
            if (cost_id.index >= cost_count) {
                try diagnostics.print("pass=lowering_plan_verify feature=cost reason=unknown cost reference lowering={d} cost={d}\n", .{ record.id.index, cost_id.index });
                return MlirStateError.InvalidLoweringPlan;
            }
        }
        if (record.decision == .unsupported and record.rejected_alternatives.len == 0) {
            try diagnostics.print("pass=lowering_plan_verify feature=decision reason=unsupported lowering lacks rejected alternatives lowering={d}\n", .{record.id.index});
            return MlirStateError.InvalidLoweringPlan;
        }
    }
}

fn verifyLoweringRegionFacts(
    facts: []const LoweringRegionFact,
    records: []const compiler_facts.LoweringRecord,
    diagnostics: *std.Io.Writer,
) !void {
    if (facts.len != records.len) {
        try diagnostics.writeAll("pass=lowering_region_verify feature=region reason=region fact count must match lowering count\n");
        return MlirStateError.InvalidLoweringPlan;
    }
    try verifyExtractedLoweringRegionFacts(facts, diagnostics);
    for (facts, records) |fact, record| {
        if (!fact.lowering_record_id.eql(record.id)) {
            try diagnostics.print("pass=lowering_region_verify feature=lowering reason=region fact references wrong lowering expected={d} actual={d}\n", .{ record.id.index, fact.lowering_record_id.index });
            return MlirStateError.InvalidLoweringPlan;
        }
        if (fact.codegen_region != record.decision) {
            try diagnostics.print("pass=lowering_region_verify feature=codegen reason=region codegen intent diverges from lowering decision lowering={d}\n", .{record.id.index});
            return MlirStateError.InvalidLoweringPlan;
        }
        if (fact.placement_record_indices.len != record.graph_instruction_ids.len) {
            try diagnostics.print("pass=lowering_region_verify feature=placement reason=placement coverage does not match lowering region lowering={d}\n", .{record.id.index});
            return MlirStateError.InvalidLoweringPlan;
        }
    }
}

fn verifyExtractedLoweringRegionFacts(facts: []const LoweringRegionFact, diagnostics: *std.Io.Writer) !void {
    if (facts.len == 0 or facts.len > limits.max_lowering_records) {
        try diagnostics.writeAll("pass=lowering_region_verify feature=region reason=invalid region fact count\n");
        return MlirStateError.InvalidLoweringPlan;
    }
    for (facts, 0..) |fact, expected_index| {
        const expected_lowering_index: u32 = std.math.cast(u32, expected_index) orelse {
            try diagnostics.writeAll("pass=lowering_region_verify feature=region reason=lowering index exceeds u32\n");
            return MlirStateError.InvalidLoweringPlan;
        };
        if (fact.lowering_record_id.index != expected_lowering_index) {
            try diagnostics.print("pass=lowering_region_verify feature=region reason=lowering index mismatch expected={d} actual={d}\n", .{ expected_lowering_index, fact.lowering_record_id.index });
            return MlirStateError.InvalidLoweringPlan;
        }
        if (fact.placement_record_indices.len == 0 or fact.placement_record_indices.len > limits.max_kernel_ref_ids or fact.logical_tile_shape.len == 0 or fact.logical_tile_shape.len > limits.max_tile_rank or fact.reason.len == 0) {
            try diagnostics.print("pass=lowering_region_verify feature=region reason=invalid region metadata lowering={d}\n", .{fact.lowering_record_id.index});
            return MlirStateError.InvalidLoweringPlan;
        }
        for (fact.logical_tile_shape) |dim| {
            if (dim <= 0) {
                try diagnostics.print("pass=lowering_region_verify feature=tile reason=non-positive tile dimension lowering={d}\n", .{fact.lowering_record_id.index});
                return MlirStateError.InvalidLoweringPlan;
            }
        }
    }
}

fn verifyScheduleCommands(commands: []const core.ScheduleCommand, diagnostics: *std.Io.Writer) !void {
    if (commands.len == 0 or commands.len > limits.max_schedule_commands) {
        try diagnostics.writeAll("pass=schedule_plan_verify feature=commands reason=invalid command count\n");
        return MlirStateError.InvalidSchedulePlan;
    }
    for (commands, 0..) |command, expected_index| {
        const expected_command_index: u32 = std.math.cast(u32, expected_index) orelse {
            try diagnostics.writeAll("pass=schedule_plan_verify feature=commands reason=command index exceeds u32\n");
            return MlirStateError.InvalidSchedulePlan;
        };
        if (command.id.index != expected_command_index) {
            try diagnostics.print("pass=schedule_plan_verify feature=commands reason=command index mismatch expected={d} actual={d}\n", .{ expected_command_index, command.id.index });
            return MlirStateError.InvalidSchedulePlan;
        }
        if (command.inputs.len > limits.max_schedule_refs or command.outputs.len > limits.max_schedule_refs or command.dependencies.len > limits.max_schedule_dependencies or command.lowering_record_ids.len > limits.max_schedule_refs or command.cost_ledger_ids.len > limits.max_schedule_refs) {
            try diagnostics.print("pass=schedule_plan_verify feature=refs reason=too many command references command={d}\n", .{command.id.index});
            return MlirStateError.InvalidSchedulePlan;
        }
        for (command.dependencies) |dependency| {
            if (dependency.command_id.index >= commands.len) {
                try diagnostics.print("pass=schedule_plan_verify feature=dependency reason=dependency references unknown command command={d} dependency={d}\n", .{ command.id.index, dependency.command_id.index });
                return MlirStateError.InvalidSchedulePlan;
            }
        }
    }
}

fn verifyCostLedgerEntries(entries: []const compiler_facts.CostLedgerEntry, diagnostics: *std.Io.Writer) !void {
    if (entries.len == 0 or entries.len > limits.max_cost_ledger_entries) {
        try diagnostics.writeAll("pass=performance_verify feature=cost-ledger reason=invalid cost ledger count\n");
        return MlirStateError.InvalidKernelCodegenPlan;
    }
    for (entries, 0..) |entry, expected_index| {
        const expected_cost_index: u32 = std.math.cast(u32, expected_index) orelse {
            try diagnostics.writeAll("pass=performance_verify feature=cost-ledger reason=cost index exceeds u32\n");
            return MlirStateError.InvalidKernelCodegenPlan;
        };
        if (entry.id.index != expected_cost_index) {
            try diagnostics.print("pass=performance_verify feature=cost-ledger reason=cost index mismatch expected={d} actual={d}\n", .{ expected_cost_index, entry.id.index });
            return MlirStateError.InvalidKernelCodegenPlan;
        }
        if (entry.graph_instruction_ids.len == 0 or entry.graph_instruction_ids.len > limits.max_kernel_ref_ids) {
            try diagnostics.print("pass=performance_verify feature=cost-ledger reason=invalid instruction count cost={d}\n", .{entry.id.index});
            return MlirStateError.InvalidKernelCodegenPlan;
        }
        if (entry.dtype.byteSize() == null or entry.formula.len == 0 or entry.approximation.len == 0) {
            try diagnostics.print("pass=performance_verify feature=cost-ledger reason=missing cost metadata cost={d}\n", .{entry.id.index});
            return MlirStateError.InvalidKernelCodegenPlan;
        }
    }
}

fn verifyMemoryTrafficRecords(records: []const compiler_facts.MemoryTrafficRecord, cost_count: usize, lowering_count: usize, diagnostics: *std.Io.Writer) !void {
    if (records.len > limits.max_memory_traffic_records) {
        try diagnostics.writeAll("pass=performance_verify feature=memory-traffic reason=too many memory traffic records\n");
        return MlirStateError.InvalidKernelCodegenPlan;
    }
    for (records, 0..) |record, expected_index| {
        const expected_traffic_index: u32 = std.math.cast(u32, expected_index) orelse {
            try diagnostics.writeAll("pass=performance_verify feature=memory-traffic reason=traffic index exceeds u32\n");
            return MlirStateError.InvalidKernelCodegenPlan;
        };
        if (record.id.index != expected_traffic_index) {
            try diagnostics.print("pass=performance_verify feature=memory-traffic reason=traffic index mismatch expected={d} actual={d}\n", .{ expected_traffic_index, record.id.index });
            return MlirStateError.InvalidKernelCodegenPlan;
        }
        if (record.graph_instruction_ids.len == 0 or record.graph_instruction_ids.len > limits.max_kernel_ref_ids or record.cost_ledger_ids.len == 0 or record.cost_ledger_ids.len > limits.max_kernel_ref_ids or record.reason.len == 0) {
            try diagnostics.print("pass=performance_verify feature=memory-traffic reason=invalid traffic metadata traffic={d}\n", .{record.id.index});
            return MlirStateError.InvalidKernelCodegenPlan;
        }
        if (record.lowering_record_id.index >= lowering_count) {
            try diagnostics.print("pass=performance_verify feature=memory-traffic reason=unknown lowering reference traffic={d} lowering={d}\n", .{ record.id.index, record.lowering_record_id.index });
            return MlirStateError.InvalidKernelCodegenPlan;
        }
        for (record.cost_ledger_ids) |cost_id| {
            if (cost_id.index >= cost_count) {
                try diagnostics.print("pass=performance_verify feature=memory-traffic reason=unknown cost reference traffic={d} cost={d}\n", .{ record.id.index, cost_id.index });
                return MlirStateError.InvalidKernelCodegenPlan;
            }
        }
    }
}

fn verifyScheduleOverlaps(overlaps: []const core.ScheduleOverlapRecord, command_count: usize, diagnostics: *std.Io.Writer) !void {
    if (overlaps.len > limits.max_schedule_overlaps) {
        try diagnostics.writeAll("pass=schedule_plan_verify feature=overlaps reason=too many overlap records\n");
        return MlirStateError.InvalidSchedulePlan;
    }
    for (overlaps, 0..) |overlap, expected_index| {
        const expected_overlap_index: u32 = std.math.cast(u32, expected_index) orelse {
            try diagnostics.writeAll("pass=schedule_plan_verify feature=overlaps reason=overlap index exceeds u32\n");
            return MlirStateError.InvalidSchedulePlan;
        };
        if (overlap.id.index != expected_overlap_index) {
            try diagnostics.print("pass=schedule_plan_verify feature=overlaps reason=overlap index mismatch expected={d} actual={d}\n", .{ expected_overlap_index, overlap.id.index });
            return MlirStateError.InvalidSchedulePlan;
        }
        if (overlap.first_command_id.index >= command_count or overlap.second_command_id.index >= command_count) {
            try diagnostics.print("pass=schedule_plan_verify feature=overlaps reason=overlap references unknown command overlap={d}\n", .{overlap.id.index});
            return MlirStateError.InvalidSchedulePlan;
        }
        if (overlap.reason.len == 0) {
            try diagnostics.print("pass=schedule_plan_verify feature=overlaps reason=missing overlap reason overlap={d}\n", .{overlap.id.index});
            return MlirStateError.InvalidSchedulePlan;
        }
    }
}

fn verifyBackendBindings(bindings: []const core.BackendBinding, diagnostics: *std.Io.Writer) !void {
    if (bindings.len == 0 or bindings.len > limits.max_backend_bindings) {
        try diagnostics.writeAll("pass=backend_binding_verify feature=bindings reason=invalid binding count\n");
        return MlirStateError.InvalidBackendBindingPlan;
    }
    for (bindings, 0..) |binding, expected_index| {
        const expected_binding_index: u32 = std.math.cast(u32, expected_index) orelse {
            try diagnostics.writeAll("pass=backend_binding_verify feature=bindings reason=binding index exceeds u32\n");
            return MlirStateError.InvalidBackendBindingPlan;
        };
        if (binding.id.index != expected_binding_index) {
            try diagnostics.print("pass=backend_binding_verify feature=bindings reason=binding index mismatch expected={d} actual={d}\n", .{ expected_binding_index, binding.id.index });
            return MlirStateError.InvalidBackendBindingPlan;
        }
        if (binding.backend_operation.len == 0 or binding.graph_instruction_ids.len == 0 or binding.cost_ledger_ids.len == 0) {
            try diagnostics.print("pass=backend_binding_verify feature=bindings reason=binding missing operation or provenance binding={d}\n", .{binding.id.index});
            return MlirStateError.InvalidBackendBindingPlan;
        }
        if (binding.graph_instruction_ids.len > limits.max_kernel_ref_ids or binding.cost_ledger_ids.len > limits.max_kernel_ref_ids) {
            try diagnostics.print("pass=backend_binding_verify feature=bindings reason=too many binding references binding={d}\n", .{binding.id.index});
            return MlirStateError.InvalidBackendBindingPlan;
        }
    }
}

fn verifyExecutableContract(contract: ExecutableContract, diagnostics: *std.Io.Writer) !void {
    if (contract.schedule_command_count == 0) {
        try diagnostics.writeAll("pass=executable_contract_verify feature=schedule reason=empty executable command set\n");
        return MlirStateError.InvalidExecutableContract;
    }
    if (contract.backend_binding_count == 0) {
        try diagnostics.writeAll("pass=executable_contract_verify feature=backend-binding reason=empty backend binding set\n");
        return MlirStateError.InvalidExecutableContract;
    }
    if (contract.kernel_codegen_count == 0) {
        try diagnostics.writeAll("pass=executable_contract_verify feature=codegen reason=empty kernel codegen set\n");
        return MlirStateError.InvalidExecutableContract;
    }
    if (contract.schedule_command_count > limits.max_schedule_commands or contract.backend_binding_count > limits.max_backend_bindings or contract.kernel_codegen_count > limits.max_kernel_codegen_records) {
        try diagnostics.writeAll("pass=executable_contract_verify feature=contract reason=contract count exceeds V0 bounds\n");
        return MlirStateError.InvalidExecutableContract;
    }
}

fn verifyBackendExecutablePlan(plan: BackendExecutablePlanFact, diagnostics: *std.Io.Writer) !void {
    if (plan.backend_operation.len == 0) {
        try diagnostics.writeAll("pass=backend_executable_verify feature=operation reason=missing backend executable operation\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    if (plan.calls.len == 0 or plan.calls.len > limits.max_backend_executable_calls) {
        try diagnostics.writeAll("pass=backend_executable_verify feature=calls reason=invalid backend executable call count\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    for (plan.calls, 0..) |call, expected_index| {
        const expected_call_index: u32 = std.math.cast(u32, expected_index) orelse {
            try diagnostics.writeAll("pass=backend_executable_verify feature=calls reason=call index exceeds u32\n");
            return MlirStateError.InvalidBackendExecutablePlan;
        };
        if (call.index != expected_call_index) {
            try diagnostics.print("pass=backend_executable_verify feature=calls reason=call index mismatch expected={d} actual={d}\n", .{ expected_call_index, call.index });
            return MlirStateError.InvalidBackendExecutablePlan;
        }
        if (!call.command_id.eql(plan.command_id) or call.backend_kind != plan.backend_kind) {
            try diagnostics.print("pass=backend_executable_verify feature=calls reason=call does not match executable plan call={d}\n", .{call.index});
            return MlirStateError.InvalidBackendExecutablePlan;
        }
        if (call.feature.len == 0 or call.backend_operation.len == 0 or call.graph_instruction_ids.len == 0 or call.input_value_ids.len == 0 or call.output_value_ids.len == 0) {
            try diagnostics.print("pass=backend_executable_verify feature=calls reason=call missing operation or provenance call={d}\n", .{call.index});
            return MlirStateError.InvalidBackendExecutablePlan;
        }
        if (call.graph_instruction_ids.len > limits.max_kernel_ref_ids or call.input_value_ids.len > limits.max_kernel_ref_ids or call.output_value_ids.len > limits.max_kernel_ref_ids) {
            try diagnostics.print("pass=backend_executable_verify feature=calls reason=too many call references call={d}\n", .{call.index});
            return MlirStateError.InvalidBackendExecutablePlan;
        }
    }
}

fn verifyBackendKernelGraph(graph: BackendKernelGraphFact, diagnostics: *std.Io.Writer) !void {
    if (graph.backend_kind != .metal_v0) {
        try diagnostics.writeAll("pass=backend_kernel_graph_verify feature=backend reason=kernel graph facts are Metal-only in V0\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    if (graph.nodes.len == 0 or graph.nodes.len > limits.max_backend_executable_calls or graph.edges.len > limits.max_backend_kernel_graph_edges) {
        try diagnostics.writeAll("pass=backend_kernel_graph_verify feature=graph reason=invalid kernel graph size\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    for (graph.nodes, 0..) |node, expected_index| {
        const expected_node_index: u32 = std.math.cast(u32, expected_index) orelse {
            try diagnostics.writeAll("pass=backend_kernel_graph_verify feature=node reason=node index exceeds u32\n");
            return MlirStateError.InvalidBackendExecutablePlan;
        };
        if (node.index != expected_node_index) {
            try diagnostics.print("pass=backend_kernel_graph_verify feature=node reason=node index mismatch expected={d} actual={d}\n", .{ expected_node_index, node.index });
            return MlirStateError.InvalidBackendExecutablePlan;
        }
        if (node.feature.len == 0 or node.backend_operation.len == 0 or node.attributes.len == 0) {
            try diagnostics.print("pass=backend_kernel_graph_verify feature=node reason=node missing operation metadata node={d}\n", .{node.index});
            return MlirStateError.InvalidBackendExecutablePlan;
        }
        if (node.graph_instruction_ids.len == 0 or node.input_value_ids.len == 0 or node.output_value_ids.len == 0 or node.output_type.dims.len == 0) {
            try diagnostics.print("pass=backend_kernel_graph_verify feature=node reason=node missing provenance or tensor metadata node={d}\n", .{node.index});
            return MlirStateError.InvalidBackendExecutablePlan;
        }
        if (node.graph_instruction_ids.len > limits.max_kernel_ref_ids or node.input_value_ids.len > limits.max_kernel_ref_ids or node.output_value_ids.len > limits.max_kernel_ref_ids) {
            try diagnostics.print("pass=backend_kernel_graph_verify feature=node reason=too many node references node={d}\n", .{node.index});
            return MlirStateError.InvalidBackendExecutablePlan;
        }
    }
    for (graph.edges) |edge| {
        if (edge.src_node_index >= graph.nodes.len or edge.dst_node_index >= graph.nodes.len) {
            try diagnostics.writeAll("pass=backend_kernel_graph_verify feature=edge reason=edge references unknown node\n");
            return MlirStateError.InvalidBackendExecutablePlan;
        }
    }
}

fn verifyRuntimeAllocationPlan(plan: RuntimeAllocationPlanFact, diagnostics: *std.Io.Writer) !void {
    if (plan.allocations.len == 0 or plan.allocations.len > limits.max_runtime_allocations or plan.command_buffer_uses.len == 0 or plan.command_buffer_uses.len > limits.max_runtime_buffer_uses) {
        try diagnostics.writeAll("pass=runtime_allocation_verify feature=plan reason=invalid runtime allocation plan size\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    for (plan.allocations, 0..) |allocation, expected_index| {
        const expected_allocation_index: u32 = std.math.cast(u32, expected_index) orelse {
            try diagnostics.writeAll("pass=runtime_allocation_verify feature=allocation reason=allocation index exceeds u32\n");
            return MlirStateError.InvalidBackendExecutablePlan;
        };
        if (allocation.index != expected_allocation_index) {
            try diagnostics.print("pass=runtime_allocation_verify feature=allocation reason=allocation index mismatch expected={d} actual={d}\n", .{ expected_allocation_index, allocation.index });
            return MlirStateError.InvalidBackendExecutablePlan;
        }
        if (!std.mem.eql(u8, allocation.placement, "host") and !std.mem.eql(u8, allocation.placement, "device")) {
            try diagnostics.print("pass=runtime_allocation_verify feature=placement reason=invalid placement allocation={d}\n", .{allocation.index});
            return MlirStateError.InvalidBackendExecutablePlan;
        }
        if (allocation.size_bytes == 0 or allocation.first_command_id.index > allocation.last_command_id.index) {
            try diagnostics.print("pass=runtime_allocation_verify feature=lifetime reason=invalid size or lifetime allocation={d}\n", .{allocation.index});
            return MlirStateError.InvalidBackendExecutablePlan;
        }
    }
    for (plan.command_buffer_uses) |use| {
        if (use.buffer_index >= plan.allocations.len or (!std.mem.eql(u8, use.access, "read") and !std.mem.eql(u8, use.access, "write"))) {
            try diagnostics.writeAll("pass=runtime_allocation_verify feature=buffer-use reason=invalid runtime buffer use\n");
            return MlirStateError.InvalidBackendExecutablePlan;
        }
    }
}

fn verifyRuntimeStreamPlan(plan: RuntimeStreamPlanFact, diagnostics: *std.Io.Writer) !void {
    if (plan.steps.len == 0 or plan.steps.len > limits.max_runtime_stream_steps) {
        try diagnostics.writeAll("pass=runtime_stream_verify feature=plan reason=invalid runtime stream step count\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    for (plan.steps) |step| {
        if (step.wait_event_ids.len > limits.max_runtime_wait_events) {
            try diagnostics.print("pass=runtime_stream_verify feature=wait-events reason=too many wait events command={d}\n", .{step.command_id.index});
            return MlirStateError.InvalidBackendExecutablePlan;
        }
    }
}

fn verifyRuntimeProfile(profile: RuntimeProfileFact, diagnostics: *std.Io.Writer) !void {
    if (profile.events.len == 0 or profile.events.len > limits.max_runtime_profile_events) {
        try diagnostics.writeAll("pass=runtime_profile_verify feature=profile reason=invalid runtime profile event count\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    for (profile.events, 0..) |event, expected_index| {
        const expected_event_index: u32 = std.math.cast(u32, expected_index) orelse {
            try diagnostics.writeAll("pass=runtime_profile_verify feature=profile reason=profile event index exceeds u32\n");
            return MlirStateError.InvalidBackendExecutablePlan;
        };
        if (event.index != expected_event_index) {
            try diagnostics.print("pass=runtime_profile_verify feature=profile reason=profile event index mismatch expected={d} actual={d}\n", .{ expected_event_index, event.index });
            return MlirStateError.InvalidBackendExecutablePlan;
        }
        if (event.kind.len == 0 or event.status.len == 0) {
            try diagnostics.print("pass=runtime_profile_verify feature=profile reason=profile event missing kind or status event={d}\n", .{event.index});
            return MlirStateError.InvalidBackendExecutablePlan;
        }
    }
}

fn verifyRuntimeProfileJoins(plan: RuntimeProfileJoinPlanFact, diagnostics: *std.Io.Writer) !void {
    if (plan.joins.len == 0 or plan.joins.len > limits.max_runtime_profile_joins) {
        try diagnostics.writeAll("pass=runtime_profile_join_verify feature=joins reason=invalid runtime profile join count\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    for (plan.joins, 0..) |join, expected_index| {
        const expected_join_index: u32 = std.math.cast(u32, expected_index) orelse {
            try diagnostics.writeAll("pass=runtime_profile_join_verify feature=joins reason=profile join index exceeds u32\n");
            return MlirStateError.InvalidBackendExecutablePlan;
        };
        if (join.index != expected_join_index) {
            try diagnostics.print("pass=runtime_profile_join_verify feature=joins reason=profile join index mismatch expected={d} actual={d}\n", .{ expected_join_index, join.index });
            return MlirStateError.InvalidBackendExecutablePlan;
        }
        if (join.subject_kind.len == 0 or join.profile_event_ids.len == 0 or join.profile_event_ids.len > limits.max_runtime_profile_join_events) {
            try diagnostics.print("pass=runtime_profile_join_verify feature=joins reason=invalid subject or event count join={d}\n", .{join.index});
            return MlirStateError.InvalidBackendExecutablePlan;
        }
        if (join.graph_instruction_ids.len > limits.max_kernel_ref_ids) {
            try diagnostics.print("pass=runtime_profile_join_verify feature=joins reason=too many instructions join={d}\n", .{join.index});
            return MlirStateError.InvalidBackendExecutablePlan;
        }
    }
}

fn verifyBackendProfileJoins(plan: BackendProfileJoinPlanFact, diagnostics: *std.Io.Writer) !void {
    if (plan.joins.len == 0 or plan.joins.len > limits.max_backend_profile_joins) {
        try diagnostics.writeAll("pass=backend_profile_join_verify feature=joins reason=invalid backend profile join count\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    for (plan.joins, 0..) |join, expected_index| {
        const expected_join_index: u32 = std.math.cast(u32, expected_index) orelse {
            try diagnostics.writeAll("pass=backend_profile_join_verify feature=joins reason=backend profile join index exceeds u32\n");
            return MlirStateError.InvalidBackendExecutablePlan;
        };
        if (join.index != expected_join_index) {
            try diagnostics.print("pass=backend_profile_join_verify feature=joins reason=backend profile join index mismatch expected={d} actual={d}\n", .{ expected_join_index, join.index });
            return MlirStateError.InvalidBackendExecutablePlan;
        }
        if (join.graph_instruction_ids.len == 0 or join.graph_instruction_ids.len > limits.max_kernel_ref_ids) {
            try diagnostics.print("pass=backend_profile_join_verify feature=joins reason=invalid instruction count join={d}\n", .{join.index});
            return MlirStateError.InvalidBackendExecutablePlan;
        }
    }
}

fn setPlacementPlanAttr(context: mlir.MlirContext, module_op: mlir.MlirOperation, records: []const compiler_facts.PlacementRecord) void {
    var attrs: [limits.max_placement_records]mlir.MlirAttribute = undefined;
    for (records, 0..) |record, index| {
        attrs[index] = placementRecordAttr(context, record);
    }
    mlir.mlirOperationSetAttributeByName(
        module_op,
        mlirStringRef("pjrtx.placement.records"),
        mlir.mlirArrayAttrGet(context, std.math.cast(isize, records.len) orelse unreachable, &attrs),
    );
}

fn setTargetSpecAttr(context: mlir.MlirContext, module_op: mlir.MlirOperation, target: target_pkg.TargetDescription) void {
    mlir.mlirOperationSetAttributeByName(
        module_op,
        mlirStringRef("pjrtx.target.spec"),
        targetDescriptionAttr(context, target),
    );
}

fn setCollectivePlanAttr(context: mlir.MlirContext, module_op: mlir.MlirOperation, records: []const compiler_facts.CollectivePlanRecord) void {
    var attrs: [limits.max_collective_records]mlir.MlirAttribute = undefined;
    for (records, 0..) |record, index| {
        attrs[index] = collectivePlanRecordAttr(context, record);
    }
    mlir.mlirOperationSetAttributeByName(
        module_op,
        mlirStringRef("pjrtx.collective.records"),
        mlir.mlirArrayAttrGet(context, std.math.cast(isize, records.len) orelse unreachable, &attrs),
    );
}

fn setLoweringPlanAttr(context: mlir.MlirContext, module_op: mlir.MlirOperation, records: []const compiler_facts.LoweringRecord) void {
    var attrs: [limits.max_lowering_records]mlir.MlirAttribute = undefined;
    for (records, 0..) |record, index| {
        attrs[index] = loweringRecordAttr(context, record);
    }
    mlir.mlirOperationSetAttributeByName(
        module_op,
        mlirStringRef("pjrtx.lowering.records"),
        mlir.mlirArrayAttrGet(context, std.math.cast(isize, records.len) orelse unreachable, &attrs),
    );
}

fn setLoweringRegionFactsAttr(context: mlir.MlirContext, module_op: mlir.MlirOperation, facts: []const LoweringRegionFact) void {
    var attrs: [limits.max_lowering_records]mlir.MlirAttribute = undefined;
    for (facts, 0..) |fact, index| {
        attrs[index] = loweringRegionFactAttr(context, fact);
    }
    mlir.mlirOperationSetAttributeByName(
        module_op,
        mlirStringRef("pjrtx.lowering.region_facts"),
        mlir.mlirArrayAttrGet(context, std.math.cast(isize, facts.len) orelse unreachable, &attrs),
    );
}

fn setCostLedgerAttr(
    context: mlir.MlirContext,
    module_op: mlir.MlirOperation,
    cost_ledger: []const compiler_facts.CostLedgerEntry,
) void {
    var cost_attrs: [limits.max_cost_ledger_entries]mlir.MlirAttribute = undefined;
    for (cost_ledger, 0..) |entry, index| cost_attrs[index] = costLedgerEntryAttr(context, entry);
    mlir.mlirOperationSetAttributeByName(
        module_op,
        mlirStringRef("pjrtx.performance.cost_ledger"),
        mlir.mlirArrayAttrGet(context, std.math.cast(isize, cost_ledger.len) orelse unreachable, &cost_attrs),
    );
}

fn setKernelCodegenPlanAttr(context: mlir.MlirContext, module_op: mlir.MlirOperation, records: []const core.KernelCodegenRecord) void {
    var attrs: [limits.max_kernel_codegen_records]mlir.MlirAttribute = undefined;
    for (records, 0..) |record, index| {
        attrs[index] = kernelCodegenRecordAttr(context, record);
    }
    mlir.mlirOperationSetAttributeByName(
        module_op,
        mlirStringRef("pjrtx.codegen.records"),
        mlir.mlirArrayAttrGet(context, std.math.cast(isize, records.len) orelse unreachable, &attrs),
    );
}

fn setPerformanceFactsAttrs(
    context: mlir.MlirContext,
    module_op: mlir.MlirOperation,
    cost_ledger: []const compiler_facts.CostLedgerEntry,
    memory_traffic: []const compiler_facts.MemoryTrafficRecord,
) void {
    setCostLedgerAttr(context, module_op, cost_ledger);

    var traffic_attrs: [limits.max_memory_traffic_records]mlir.MlirAttribute = undefined;
    for (memory_traffic, 0..) |record, index| traffic_attrs[index] = memoryTrafficRecordAttr(context, record);
    mlir.mlirOperationSetAttributeByName(
        module_op,
        mlirStringRef("pjrtx.performance.memory_traffic"),
        mlir.mlirArrayAttrGet(context, std.math.cast(isize, memory_traffic.len) orelse unreachable, &traffic_attrs),
    );
}

fn setSchedulePlanAttrs(
    context: mlir.MlirContext,
    module_op: mlir.MlirOperation,
    commands: []const core.ScheduleCommand,
    overlaps: []const core.ScheduleOverlapRecord,
) void {
    var command_attrs: [limits.max_schedule_commands]mlir.MlirAttribute = undefined;
    for (commands, 0..) |command, index| {
        command_attrs[index] = scheduleCommandAttr(context, command);
    }
    mlir.mlirOperationSetAttributeByName(
        module_op,
        mlirStringRef("pjrtx.schedule.commands"),
        mlir.mlirArrayAttrGet(context, std.math.cast(isize, commands.len) orelse unreachable, &command_attrs),
    );

    var overlap_attrs: [limits.max_schedule_overlaps]mlir.MlirAttribute = undefined;
    for (overlaps, 0..) |overlap, index| {
        overlap_attrs[index] = scheduleOverlapAttr(context, overlap);
    }
    mlir.mlirOperationSetAttributeByName(
        module_op,
        mlirStringRef("pjrtx.schedule.overlaps"),
        mlir.mlirArrayAttrGet(context, std.math.cast(isize, overlaps.len) orelse unreachable, &overlap_attrs),
    );
}

fn setBackendBindingPlanAttr(context: mlir.MlirContext, module_op: mlir.MlirOperation, bindings: []const core.BackendBinding) void {
    var attrs: [limits.max_backend_bindings]mlir.MlirAttribute = undefined;
    for (bindings, 0..) |binding, index| {
        attrs[index] = backendBindingAttr(context, binding);
    }
    mlir.mlirOperationSetAttributeByName(
        module_op,
        mlirStringRef("pjrtx.backend.bindings"),
        mlir.mlirArrayAttrGet(context, std.math.cast(isize, bindings.len) orelse unreachable, &attrs),
    );
}

fn setExecutableContractAttr(context: mlir.MlirContext, module_op: mlir.MlirOperation, contract: ExecutableContract) void {
    mlir.mlirOperationSetAttributeByName(
        module_op,
        mlirStringRef("pjrtx.executable.contract"),
        executableContractAttr(context, contract),
    );
}

fn executableContractAttr(context: mlir.MlirContext, contract: ExecutableContract) mlir.MlirAttribute {
    const attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(context, "target_kind", stringAttr(context, @tagName(contract.target_kind))),
        namedAttr(context, "schedule_commands", integerAttr(context, contract.schedule_command_count)),
        namedAttr(context, "backend_bindings", integerAttr(context, contract.backend_binding_count)),
        namedAttr(context, "kernel_codegen_records", integerAttr(context, contract.kernel_codegen_count)),
    };
    return mlir.mlirDictionaryAttrGet(context, attrs.len, &attrs);
}

fn targetDescriptionAttr(context: mlir.MlirContext, target: target_pkg.TargetDescription) mlir.MlirAttribute {
    const attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(context, "name", stringAttr(context, target.name)),
        namedAttr(context, "kind", stringAttr(context, @tagName(target.kind))),
        namedAttr(context, "devices", targetDeviceArrayAttr(context, target.devices)),
        namedAttr(context, "memory_spaces", targetMemorySpaceArrayAttr(context, target.memory_spaces)),
        namedAttr(context, "transfer_edges", targetTransferEdgeArrayAttr(context, target.transfer_edges)),
        namedAttr(context, "execution_units", targetExecutionUnitArrayAttr(context, target.execution_units)),
    };
    return mlir.mlirDictionaryAttrGet(context, attrs.len, &attrs);
}

fn targetDeviceArrayAttr(context: mlir.MlirContext, devices: []const target_pkg.TargetDevice) mlir.MlirAttribute {
    var attrs: [limits.max_target_devices]mlir.MlirAttribute = undefined;
    for (devices, 0..) |device, index| attrs[index] = targetDeviceAttr(context, device);
    return mlir.mlirArrayAttrGet(context, std.math.cast(isize, devices.len) orelse unreachable, &attrs);
}

fn targetDeviceAttr(context: mlir.MlirContext, device: target_pkg.TargetDevice) mlir.MlirAttribute {
    const attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(context, "id", integerAttr(context, device.id)),
        namedAttr(context, "local_hardware_id", signedIntegerAttr(context, device.local_hardware_id)),
        namedAttr(context, "name", stringAttr(context, device.name)),
        namedAttr(context, "memory_spaces", targetU32ArrayAttr(context, device.memory_space_ids)),
        namedAttr(context, "execution_units", targetU32ArrayAttr(context, device.execution_unit_ids)),
    };
    return mlir.mlirDictionaryAttrGet(context, attrs.len, &attrs);
}

fn targetMemorySpaceArrayAttr(context: mlir.MlirContext, memory_spaces: []const target_pkg.TargetMemorySpace) mlir.MlirAttribute {
    var attrs: [limits.max_target_memory_spaces]mlir.MlirAttribute = undefined;
    for (memory_spaces, 0..) |memory_space, index| attrs[index] = targetMemorySpaceAttr(context, memory_space);
    return mlir.mlirArrayAttrGet(context, std.math.cast(isize, memory_spaces.len) orelse unreachable, &attrs);
}

fn targetMemorySpaceAttr(context: mlir.MlirContext, memory_space: target_pkg.TargetMemorySpace) mlir.MlirAttribute {
    const attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(context, "id", integerAttr(context, memory_space.id)),
        namedAttr(context, "name", stringAttr(context, memory_space.name)),
        namedAttr(context, "kind", stringAttr(context, @tagName(memory_space.kind))),
        namedAttr(context, "capacity_bytes", optionalU64StringAttr(context, memory_space.capacity_bytes)),
        namedAttr(context, "bandwidth_bytes_per_second", optionalF64StringAttr(context, memory_space.bandwidth_bytes_per_second)),
        namedAttr(context, "note", stringAttr(context, memory_space.note)),
    };
    return mlir.mlirDictionaryAttrGet(context, attrs.len, &attrs);
}

fn targetTransferEdgeArrayAttr(context: mlir.MlirContext, edges: []const target_pkg.TargetTransferEdge) mlir.MlirAttribute {
    var attrs: [limits.max_target_transfer_edges]mlir.MlirAttribute = undefined;
    for (edges, 0..) |edge, index| attrs[index] = targetTransferEdgeAttr(context, edge);
    return mlir.mlirArrayAttrGet(context, std.math.cast(isize, edges.len) orelse unreachable, &attrs);
}

fn targetTransferEdgeAttr(context: mlir.MlirContext, edge: target_pkg.TargetTransferEdge) mlir.MlirAttribute {
    const attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(context, "id", integerAttr(context, edge.id)),
        namedAttr(context, "src_memory", integerAttr(context, edge.src_memory_space)),
        namedAttr(context, "dst_memory", integerAttr(context, edge.dst_memory_space)),
        namedAttr(context, "bandwidth_bytes_per_second", optionalF64StringAttr(context, edge.bandwidth_bytes_per_second)),
        namedAttr(context, "latency_ns", optionalU64StringAttr(context, edge.latency_ns)),
        namedAttr(context, "supports_async", boolAttr(context, edge.supports_async)),
        namedAttr(context, "engine_unit", optionalIntegerAttr(context, edge.engine_unit_id)),
        namedAttr(context, "note", stringAttr(context, edge.note)),
    };
    return mlir.mlirDictionaryAttrGet(context, attrs.len, &attrs);
}

fn targetExecutionUnitArrayAttr(context: mlir.MlirContext, units: []const target_pkg.ExecutionUnit) mlir.MlirAttribute {
    var attrs: [limits.max_target_execution_units]mlir.MlirAttribute = undefined;
    for (units, 0..) |unit, index| attrs[index] = targetExecutionUnitAttr(context, unit);
    return mlir.mlirArrayAttrGet(context, std.math.cast(isize, units.len) orelse unreachable, &attrs);
}

fn targetExecutionUnitAttr(context: mlir.MlirContext, unit: target_pkg.ExecutionUnit) mlir.MlirAttribute {
    const attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(context, "id", integerAttr(context, unit.id)),
        namedAttr(context, "name", stringAttr(context, unit.name)),
        namedAttr(context, "kind", stringAttr(context, @tagName(unit.kind))),
        namedAttr(context, "dtype_rates", targetDTypeRateArrayAttr(context, unit.dtype_rates)),
    };
    return mlir.mlirDictionaryAttrGet(context, attrs.len, &attrs);
}

fn targetDTypeRateArrayAttr(context: mlir.MlirContext, rates: []const target_pkg.DTypeRate) mlir.MlirAttribute {
    var attrs: [limits.max_target_dtype_rates]mlir.MlirAttribute = undefined;
    for (rates, 0..) |rate, index| attrs[index] = targetDTypeRateAttr(context, rate);
    return mlir.mlirArrayAttrGet(context, std.math.cast(isize, rates.len) orelse unreachable, &attrs);
}

fn targetDTypeRateAttr(context: mlir.MlirContext, rate: target_pkg.DTypeRate) mlir.MlirAttribute {
    const attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(context, "dtype", stringAttr(context, @tagName(rate.dtype))),
        namedAttr(context, "op_class", stringAttr(context, @tagName(rate.op_class))),
        namedAttr(context, "ops_per_second", optionalF64StringAttr(context, rate.ops_per_second)),
        namedAttr(context, "source", stringAttr(context, @tagName(rate.source))),
        namedAttr(context, "note", stringAttr(context, rate.note)),
    };
    return mlir.mlirDictionaryAttrGet(context, attrs.len, &attrs);
}

fn costLedgerEntryAttr(context: mlir.MlirContext, entry: compiler_facts.CostLedgerEntry) mlir.MlirAttribute {
    const attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(context, "id", integerAttr(context, entry.id.index)),
        namedAttr(context, "source", optionalSourceRefAttr(context, entry.source)),
        namedAttr(context, "instructions", graphInstructionIdArrayAttr(context, entry.graph_instruction_ids)),
        namedAttr(context, "op_class", stringAttr(context, @tagName(entry.op_class))),
        namedAttr(context, "dtype", stringAttr(context, @tagName(entry.dtype))),
        namedAttr(context, "accumulation_dtype", optionalBufferTypeAttr(context, entry.accumulation_dtype)),
        namedAttr(context, "logical_ops", u128StringAttr(context, entry.logical_ops)),
        namedAttr(context, "bytes_read", u128StringAttr(context, entry.bytes_read)),
        namedAttr(context, "bytes_written", u128StringAttr(context, entry.bytes_written)),
        namedAttr(context, "expected_unit", optionalIntegerAttr(context, entry.expected_unit_id)),
        namedAttr(context, "formula", nonEmptyStringAttr(context, entry.formula, "unknown")),
        namedAttr(context, "approximation", nonEmptyStringAttr(context, entry.approximation, "unknown")),
    };
    return mlir.mlirDictionaryAttrGet(context, attrs.len, &attrs);
}

fn memoryTrafficRecordAttr(context: mlir.MlirContext, record: compiler_facts.MemoryTrafficRecord) mlir.MlirAttribute {
    const attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(context, "id", integerAttr(context, record.id.index)),
        namedAttr(context, "lowering", integerAttr(context, record.lowering_record_id.index)),
        namedAttr(context, "memory", integerAttr(context, record.memory_space_id)),
        namedAttr(context, "kind", stringAttr(context, @tagName(record.kind))),
        namedAttr(context, "instructions", graphInstructionIdArrayAttr(context, record.graph_instruction_ids)),
        namedAttr(context, "costs", costLedgerIdArrayAttr(context, record.cost_ledger_ids)),
        namedAttr(context, "bytes_read", u128StringAttr(context, record.bytes_read)),
        namedAttr(context, "bytes_written", u128StringAttr(context, record.bytes_written)),
        namedAttr(context, "reason", nonEmptyStringAttr(context, record.reason, "unknown")),
    };
    return mlir.mlirDictionaryAttrGet(context, attrs.len, &attrs);
}

fn optionalSourceRefAttr(context: mlir.MlirContext, source: ?compiler_facts.SourceRef) mlir.MlirAttribute {
    const source_ref = source orelse return stringAttr(context, "none");
    const attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(context, "id", integerAttr(context, source_ref.id.index)),
        namedAttr(context, "frontend", stringAttr(context, @tagName(source_ref.frontend))),
        namedAttr(context, "op_name", nonEmptyStringAttr(context, source_ref.op_name, "unknown")),
        namedAttr(context, "source_index", integerAttr(context, source_ref.source_index)),
        namedAttr(context, "location", nonEmptyStringAttr(context, source_ref.location, "unknown")),
    };
    return mlir.mlirDictionaryAttrGet(context, attrs.len, &attrs);
}

fn nonEmptyStringAttr(context: mlir.MlirContext, value: []const u8, default_value: []const u8) mlir.MlirAttribute {
    return stringAttr(context, if (value.len == 0) default_value else value);
}

fn optionalBufferTypeAttr(context: mlir.MlirContext, dtype: ?core.BufferType) mlir.MlirAttribute {
    const value = dtype orelse return stringAttr(context, "none");
    return stringAttr(context, @tagName(value));
}

fn setBackendExecutablePlanAttr(context: mlir.MlirContext, module_op: mlir.MlirOperation, plan: BackendExecutablePlanFact) void {
    mlir.mlirOperationSetAttributeByName(
        module_op,
        mlirStringRef("pjrtx.backend.executable"),
        backendExecutablePlanAttr(context, plan),
    );
}

fn setBackendKernelGraphAttr(context: mlir.MlirContext, module_op: mlir.MlirOperation, graph: BackendKernelGraphFact) void {
    mlir.mlirOperationSetAttributeByName(
        module_op,
        mlirStringRef("pjrtx.backend.kernel_graph"),
        backendKernelGraphAttr(context, graph),
    );
}

fn setRuntimeAllocationPlanAttr(context: mlir.MlirContext, module_op: mlir.MlirOperation, plan: RuntimeAllocationPlanFact) void {
    mlir.mlirOperationSetAttributeByName(
        module_op,
        mlirStringRef("pjrtx.runtime.allocation"),
        runtimeAllocationPlanAttr(context, plan),
    );
}

fn setRuntimeStreamPlanAttr(context: mlir.MlirContext, module_op: mlir.MlirOperation, plan: RuntimeStreamPlanFact) void {
    var step_attrs: [limits.max_runtime_stream_steps]mlir.MlirAttribute = undefined;
    for (plan.steps, 0..) |step, index| {
        step_attrs[index] = runtimeStreamStepAttr(context, step);
    }
    mlir.mlirOperationSetAttributeByName(
        module_op,
        mlirStringRef("pjrtx.runtime.streams"),
        mlir.mlirArrayAttrGet(context, std.math.cast(isize, plan.steps.len) orelse unreachable, &step_attrs),
    );
}

fn setRuntimeProfileAttr(context: mlir.MlirContext, module_op: mlir.MlirOperation, profile: RuntimeProfileFact) void {
    var event_attrs: [limits.max_runtime_profile_events]mlir.MlirAttribute = undefined;
    for (profile.events, 0..) |event, index| {
        event_attrs[index] = runtimeProfileEventAttr(context, event);
    }
    mlir.mlirOperationSetAttributeByName(
        module_op,
        mlirStringRef("pjrtx.runtime.profile_events"),
        mlir.mlirArrayAttrGet(context, std.math.cast(isize, profile.events.len) orelse unreachable, &event_attrs),
    );
}

fn setRuntimeProfileJoinAttr(context: mlir.MlirContext, module_op: mlir.MlirOperation, plan: RuntimeProfileJoinPlanFact) void {
    var join_attrs: [limits.max_runtime_profile_joins]mlir.MlirAttribute = undefined;
    for (plan.joins, 0..) |join, index| {
        join_attrs[index] = runtimeProfileJoinAttr(context, join);
    }
    mlir.mlirOperationSetAttributeByName(
        module_op,
        mlirStringRef("pjrtx.runtime.profile_joins"),
        mlir.mlirArrayAttrGet(context, std.math.cast(isize, plan.joins.len) orelse unreachable, &join_attrs),
    );
}

fn setBackendProfileJoinAttr(context: mlir.MlirContext, module_op: mlir.MlirOperation, plan: BackendProfileJoinPlanFact) void {
    var join_attrs: [limits.max_backend_profile_joins]mlir.MlirAttribute = undefined;
    for (plan.joins, 0..) |join, index| {
        join_attrs[index] = backendProfileJoinAttr(context, join);
    }
    mlir.mlirOperationSetAttributeByName(
        module_op,
        mlirStringRef("pjrtx.backend.profile_joins"),
        mlir.mlirArrayAttrGet(context, std.math.cast(isize, plan.joins.len) orelse unreachable, &join_attrs),
    );
}

fn runtimeProfileEventAttr(context: mlir.MlirContext, event: RuntimeProfileEventFact) mlir.MlirAttribute {
    const attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(context, "index", integerAttr(context, event.index)),
        namedAttr(context, "command", optionalIntegerAttr(context, if (event.command_id) |id| id.index else null)),
        namedAttr(context, "instructions", graphInstructionIdArrayAttr(context, event.graph_instruction_ids)),
        namedAttr(context, "kind", stringAttr(context, event.kind)),
        namedAttr(context, "start_ns", u64StackStringAttr(context, event.start_ns)),
        namedAttr(context, "duration_ns", u64StackStringAttr(context, event.duration_ns)),
        namedAttr(context, "bytes", u128StackStringAttr(context, event.bytes)),
        namedAttr(context, "logical_ops", u128StackStringAttr(context, event.logical_ops)),
        namedAttr(context, "status", stringAttr(context, event.status)),
        namedAttr(context, "forced_sync", boolAttr(context, event.forced_synchronization)),
    };
    return mlir.mlirDictionaryAttrGet(context, attrs.len, &attrs);
}

fn runtimeProfileJoinAttr(context: mlir.MlirContext, join: RuntimeProfileJoinFact) mlir.MlirAttribute {
    const attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(context, "index", integerAttr(context, join.index)),
        namedAttr(context, "subject_kind", stringAttr(context, join.subject_kind)),
        namedAttr(context, "subject_id", integerAttr(context, join.subject_id)),
        namedAttr(context, "command", optionalIntegerAttr(context, if (join.command_id) |id| id.index else null)),
        namedAttr(context, "instructions", graphInstructionIdArrayAttr(context, join.graph_instruction_ids)),
        namedAttr(context, "events", profileEventIdArrayAttr(context, join.profile_event_ids)),
    };
    return mlir.mlirDictionaryAttrGet(context, attrs.len, &attrs);
}

fn backendProfileJoinAttr(context: mlir.MlirContext, join: BackendProfileJoinFact) mlir.MlirAttribute {
    const attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(context, "index", integerAttr(context, join.index)),
        namedAttr(context, "call", integerAttr(context, join.call_index)),
        namedAttr(context, "command", integerAttr(context, join.command_id.index)),
        namedAttr(context, "event", integerAttr(context, join.profile_event_id.index)),
        namedAttr(context, "instructions", graphInstructionIdArrayAttr(context, join.graph_instruction_ids)),
    };
    return mlir.mlirDictionaryAttrGet(context, attrs.len, &attrs);
}

fn runtimeStreamStepAttr(context: mlir.MlirContext, step: RuntimeStreamStepFact) mlir.MlirAttribute {
    var wait_attrs: [limits.max_runtime_wait_events]mlir.MlirAttribute = undefined;
    for (step.wait_event_ids, 0..) |event_id, index| {
        wait_attrs[index] = integerAttr(context, event_id);
    }
    const attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(context, "command", integerAttr(context, step.command_id.index)),
        namedAttr(context, "stream", integerAttr(context, step.stream.index)),
        namedAttr(context, "wait_events", mlir.mlirArrayAttrGet(context, std.math.cast(isize, step.wait_event_ids.len) orelse unreachable, &wait_attrs)),
        namedAttr(context, "start_event", integerAttr(context, step.start_event_id)),
        namedAttr(context, "done_event", integerAttr(context, step.done_event_id)),
    };
    return mlir.mlirDictionaryAttrGet(context, attrs.len, &attrs);
}

fn runtimeAllocationPlanAttr(context: mlir.MlirContext, plan: RuntimeAllocationPlanFact) mlir.MlirAttribute {
    var allocation_attrs: [limits.max_runtime_allocations]mlir.MlirAttribute = undefined;
    for (plan.allocations, 0..) |allocation, index| {
        allocation_attrs[index] = runtimeAllocationAttr(context, allocation);
    }

    var use_attrs: [limits.max_runtime_buffer_uses]mlir.MlirAttribute = undefined;
    for (plan.command_buffer_uses, 0..) |use, index| {
        use_attrs[index] = runtimeBufferUseAttr(context, use);
    }

    const attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(context, "allocations", mlir.mlirArrayAttrGet(context, std.math.cast(isize, plan.allocations.len) orelse unreachable, &allocation_attrs)),
        namedAttr(context, "command_buffer_uses", mlir.mlirArrayAttrGet(context, std.math.cast(isize, plan.command_buffer_uses.len) orelse unreachable, &use_attrs)),
        namedAttr(context, "peak_device_bytes", u128StackStringAttr(context, plan.peak_device_bytes)),
    };
    return mlir.mlirDictionaryAttrGet(context, attrs.len, &attrs);
}

fn runtimeAllocationAttr(context: mlir.MlirContext, allocation: RuntimeAllocationFact) mlir.MlirAttribute {
    const attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(context, "index", integerAttr(context, allocation.index)),
        namedAttr(context, "value", integerAttr(context, allocation.value_id.index)),
        namedAttr(context, "placement", stringAttr(context, allocation.placement)),
        namedAttr(context, "memory", integerAttr(context, allocation.memory_space_id)),
        namedAttr(context, "bytes", u128StackStringAttr(context, allocation.size_bytes)),
        namedAttr(context, "first_command", integerAttr(context, allocation.first_command_id.index)),
        namedAttr(context, "last_command", integerAttr(context, allocation.last_command_id.index)),
    };
    return mlir.mlirDictionaryAttrGet(context, attrs.len, &attrs);
}

fn runtimeBufferUseAttr(context: mlir.MlirContext, use: RuntimeBufferUseFact) mlir.MlirAttribute {
    const attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(context, "command", integerAttr(context, use.command_id.index)),
        namedAttr(context, "buffer", integerAttr(context, use.buffer_index)),
        namedAttr(context, "access", stringAttr(context, use.access)),
    };
    return mlir.mlirDictionaryAttrGet(context, attrs.len, &attrs);
}

fn backendKernelGraphAttr(context: mlir.MlirContext, graph: BackendKernelGraphFact) mlir.MlirAttribute {
    var node_attrs: [limits.max_backend_executable_calls]mlir.MlirAttribute = undefined;
    for (graph.nodes, 0..) |node, index| {
        node_attrs[index] = backendKernelGraphNodeAttr(context, node);
    }

    var edge_attrs: [limits.max_backend_kernel_graph_edges]mlir.MlirAttribute = undefined;
    for (graph.edges, 0..) |edge, index| {
        edge_attrs[index] = backendKernelGraphEdgeAttr(context, edge);
    }

    const attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(context, "backend_kind", stringAttr(context, @tagName(graph.backend_kind))),
        namedAttr(context, "command", integerAttr(context, graph.command_id.index)),
        namedAttr(context, "nodes", mlir.mlirArrayAttrGet(context, std.math.cast(isize, graph.nodes.len) orelse unreachable, &node_attrs)),
        namedAttr(context, "edges", mlir.mlirArrayAttrGet(context, std.math.cast(isize, graph.edges.len) orelse unreachable, &edge_attrs)),
    };
    return mlir.mlirDictionaryAttrGet(context, attrs.len, &attrs);
}

fn backendKernelGraphNodeAttr(context: mlir.MlirContext, node: BackendKernelGraphNodeFact) mlir.MlirAttribute {
    const attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(context, "index", integerAttr(context, node.index)),
        namedAttr(context, "call", integerAttr(context, node.call_index)),
        namedAttr(context, "instruction", integerAttr(context, node.graph_instruction_id.index)),
        namedAttr(context, "instructions", graphInstructionIdArrayAttr(context, node.graph_instruction_ids)),
        namedAttr(context, "feature", stringAttr(context, node.feature)),
        namedAttr(context, "operation", stringAttr(context, node.backend_operation)),
        namedAttr(context, "inputs", graphValueIdArrayAttrBounded(context, node.input_value_ids)),
        namedAttr(context, "outputs", graphValueIdArrayAttrBounded(context, node.output_value_ids)),
        namedAttr(context, "output_type", backendTensorDescriptorAttr(context, node.output_type)),
        namedAttr(context, "attributes", stringAttr(context, node.attributes)),
    };
    return mlir.mlirDictionaryAttrGet(context, attrs.len, &attrs);
}

fn backendTensorDescriptorAttr(context: mlir.MlirContext, descriptor: BackendTensorDescriptorFact) mlir.MlirAttribute {
    const attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(context, "dtype", stringAttr(context, @tagName(descriptor.element_type))),
        namedAttr(context, "dims", i64ArrayAttr(context, descriptor.dims)),
        namedAttr(context, "layout", stringAttr(context, @tagName(descriptor.layout))),
    };
    return mlir.mlirDictionaryAttrGet(context, attrs.len, &attrs);
}

fn backendKernelGraphEdgeAttr(context: mlir.MlirContext, edge: BackendKernelGraphEdgeFact) mlir.MlirAttribute {
    const attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(context, "value", integerAttr(context, edge.value_id.index)),
        namedAttr(context, "src", integerAttr(context, edge.src_node_index)),
        namedAttr(context, "dst", integerAttr(context, edge.dst_node_index)),
    };
    return mlir.mlirDictionaryAttrGet(context, attrs.len, &attrs);
}

fn backendExecutablePlanAttr(context: mlir.MlirContext, plan: BackendExecutablePlanFact) mlir.MlirAttribute {
    var call_attrs: [limits.max_backend_executable_calls]mlir.MlirAttribute = undefined;
    for (plan.calls, 0..) |call, index| {
        call_attrs[index] = backendExecutableCallAttr(context, call);
    }
    const attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(context, "backend_kind", stringAttr(context, @tagName(plan.backend_kind))),
        namedAttr(context, "command", integerAttr(context, plan.command_id.index)),
        namedAttr(context, "operation", stringAttr(context, plan.backend_operation)),
        namedAttr(context, "calls", mlir.mlirArrayAttrGet(context, std.math.cast(isize, plan.calls.len) orelse unreachable, &call_attrs)),
    };
    return mlir.mlirDictionaryAttrGet(context, attrs.len, &attrs);
}

fn backendExecutableCallAttr(context: mlir.MlirContext, call: BackendExecutableCallFact) mlir.MlirAttribute {
    const attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(context, "index", integerAttr(context, call.index)),
        namedAttr(context, "command", integerAttr(context, call.command_id.index)),
        namedAttr(context, "backend_kind", stringAttr(context, @tagName(call.backend_kind))),
        namedAttr(context, "instruction", integerAttr(context, call.graph_instruction_id.index)),
        namedAttr(context, "instructions", graphInstructionIdArrayAttr(context, call.graph_instruction_ids)),
        namedAttr(context, "feature", stringAttr(context, call.feature)),
        namedAttr(context, "operation", stringAttr(context, call.backend_operation)),
        namedAttr(context, "inputs", graphValueIdArrayAttrBounded(context, call.input_value_ids)),
        namedAttr(context, "outputs", graphValueIdArrayAttrBounded(context, call.output_value_ids)),
        namedAttr(context, "expected_unit", optionalIntegerAttr(context, call.expected_unit_id)),
    };
    return mlir.mlirDictionaryAttrGet(context, attrs.len, &attrs);
}

fn backendBindingAttr(context: mlir.MlirContext, binding: core.BackendBinding) mlir.MlirAttribute {
    const attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(context, "id", integerAttr(context, binding.id.index)),
        namedAttr(context, "command", integerAttr(context, binding.command_id.index)),
        namedAttr(context, "backend_kind", stringAttr(context, @tagName(binding.backend_kind))),
        namedAttr(context, "operation", stringAttr(context, binding.backend_operation)),
        namedAttr(context, "instructions", graphInstructionIdArrayAttr(context, binding.graph_instruction_ids)),
        namedAttr(context, "expected_unit", optionalIntegerAttr(context, binding.expected_unit_id)),
        namedAttr(context, "costs", costLedgerIdArrayAttr(context, binding.cost_ledger_ids)),
    };
    return mlir.mlirDictionaryAttrGet(context, attrs.len, &attrs);
}

fn scheduleCommandAttr(context: mlir.MlirContext, command: core.ScheduleCommand) mlir.MlirAttribute {
    const attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(context, "id", integerAttr(context, command.id.index)),
        namedAttr(context, "kind", stringAttr(context, @tagName(command.kind))),
        namedAttr(context, "stream", integerAttr(context, command.stream.index)),
        namedAttr(context, "inputs", graphValueIdArrayAttrSchedule(context, command.inputs)),
        namedAttr(context, "outputs", graphValueIdArrayAttrSchedule(context, command.outputs)),
        namedAttr(context, "dependencies", commandDependencyArrayAttr(context, command.dependencies)),
        namedAttr(context, "lowerings", loweringRecordIdArrayAttrSchedule(context, command.lowering_record_ids)),
        namedAttr(context, "costs", costLedgerIdArrayAttrSchedule(context, command.cost_ledger_ids)),
    };
    return mlir.mlirDictionaryAttrGet(context, attrs.len, &attrs);
}

fn scheduleOverlapAttr(context: mlir.MlirContext, overlap: core.ScheduleOverlapRecord) mlir.MlirAttribute {
    const attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(context, "id", integerAttr(context, overlap.id.index)),
        namedAttr(context, "decision", stringAttr(context, @tagName(overlap.decision))),
        namedAttr(context, "kind", stringAttr(context, @tagName(overlap.kind))),
        namedAttr(context, "first_command", integerAttr(context, overlap.first_command_id.index)),
        namedAttr(context, "second_command", integerAttr(context, overlap.second_command_id.index)),
        namedAttr(context, "dependency_kind", stringAttr(context, @tagName(overlap.dependency_kind))),
        namedAttr(context, "first_stream", integerAttr(context, overlap.first_stream.index)),
        namedAttr(context, "second_stream", integerAttr(context, overlap.second_stream.index)),
        namedAttr(context, "reason", stringAttr(context, overlap.reason)),
    };
    return mlir.mlirDictionaryAttrGet(context, attrs.len, &attrs);
}

fn loweringRecordAttr(context: mlir.MlirContext, record: compiler_facts.LoweringRecord) mlir.MlirAttribute {
    const attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(context, "id", integerAttr(context, record.id.index)),
        namedAttr(context, "instructions", graphInstructionIdArrayAttr(context, record.graph_instruction_ids)),
        namedAttr(context, "decision", stringAttr(context, @tagName(record.decision))),
        namedAttr(context, "reason", nonEmptyStringAttr(context, record.reason, "unknown")),
        namedAttr(context, "rejected_alternatives", stringArrayAttr(context, record.rejected_alternatives)),
        namedAttr(context, "costs", costLedgerIdArrayAttr(context, record.cost_ledger_ids)),
    };
    return mlir.mlirDictionaryAttrGet(context, attrs.len, &attrs);
}

fn loweringRegionFactAttr(context: mlir.MlirContext, fact: LoweringRegionFact) mlir.MlirAttribute {
    const attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(context, "lowering", integerAttr(context, fact.lowering_record_id.index)),
        namedAttr(context, "fusion_group", optionalIntegerAttr(context, fact.fusion_group_index)),
        namedAttr(context, "placements", u32ArrayAttr(context, fact.placement_record_indices)),
        namedAttr(context, "tile", i64ArrayAttr(context, fact.logical_tile_shape)),
        namedAttr(context, "result_memory", integerAttr(context, fact.result_memory_space_id)),
        namedAttr(context, "tile_memory", optionalIntegerAttr(context, fact.tile_memory_space_id)),
        namedAttr(context, "codegen_region", stringAttr(context, @tagName(fact.codegen_region))),
        namedAttr(context, "reason", nonEmptyStringAttr(context, fact.reason, "unknown")),
    };
    return mlir.mlirDictionaryAttrGet(context, attrs.len, &attrs);
}

fn stringArrayAttr(context: mlir.MlirContext, values: []const []const u8) mlir.MlirAttribute {
    var attrs: [limits.max_kernel_ref_ids]mlir.MlirAttribute = undefined;
    for (values, 0..) |value, index| attrs[index] = nonEmptyStringAttr(context, value, "unknown");
    return mlir.mlirArrayAttrGet(context, std.math.cast(isize, values.len) orelse unreachable, &attrs);
}

fn kernelCodegenRecordAttr(context: mlir.MlirContext, record: core.KernelCodegenRecord) mlir.MlirAttribute {
    const attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(context, "id", integerAttr(context, record.id.index)),
        namedAttr(context, "lowering", integerAttr(context, record.lowering_record_id.index)),
        namedAttr(context, "command", integerAttr(context, record.command_id.index)),
        namedAttr(context, "backend_kind", stringAttr(context, @tagName(record.backend_kind))),
        namedAttr(context, "kind", stringAttr(context, @tagName(record.kind))),
        namedAttr(context, "operation", stringAttr(context, record.operation)),
        namedAttr(context, "shape", kernelShapeAttr(context, record.shape)),
        namedAttr(context, "tile", i64ArrayAttr(context, record.logical_tile_shape)),
        namedAttr(context, "result_memory", integerAttr(context, record.result_memory_space_id)),
        namedAttr(context, "tile_memory", optionalIntegerAttr(context, record.tile_memory_space_id)),
        namedAttr(context, "pressure", kernelPressureAttr(context, record.memory_pressure)),
        namedAttr(context, "external_inputs", graphValueIdArrayAttrBounded(context, record.external_input_ids)),
        namedAttr(context, "external_outputs", graphValueIdArrayAttrBounded(context, record.external_output_ids)),
        namedAttr(context, "intermediates", graphValueIdArrayAttrBounded(context, record.intermediate_value_ids)),
        namedAttr(context, "instructions", graphInstructionIdArrayAttr(context, record.graph_instruction_ids)),
        namedAttr(context, "costs", costLedgerIdArrayAttr(context, record.cost_ledger_ids)),
        namedAttr(context, "traffic", memoryTrafficIdArrayAttr(context, record.memory_traffic_ids)),
        namedAttr(context, "expected_unit", optionalIntegerAttr(context, record.expected_unit_id)),
        namedAttr(context, "reason", stringAttr(context, record.reason)),
    };
    return mlir.mlirDictionaryAttrGet(context, attrs.len, &attrs);
}

fn collectivePlanRecordAttr(context: mlir.MlirContext, record: compiler_facts.CollectivePlanRecord) mlir.MlirAttribute {
    const attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(context, "index", integerAttr(context, record.index)),
        namedAttr(context, "decision", stringAttr(context, @tagName(record.decision))),
        namedAttr(context, "algorithm", stringAttr(context, @tagName(record.algorithm))),
        namedAttr(context, "checked", integerAttr(context, record.checked_graph_instruction_count)),
        namedAttr(context, "lowered", integerAttr(context, record.lowered_collective_count)),
        namedAttr(context, "unsupported", integerAttr(context, record.unsupported_collective_count)),
        namedAttr(context, "estimated_bytes", u128StackStringAttr(context, record.estimated_bytes)),
        namedAttr(context, "estimated_latency_ns", optionalU64StringAttr(context, record.estimated_latency_ns)),
        namedAttr(context, "reason", stringAttr(context, record.reason)),
    };
    return mlir.mlirDictionaryAttrGet(context, attrs.len, &attrs);
}

fn placementRecordAttr(context: mlir.MlirContext, record: compiler_facts.PlacementRecord) mlir.MlirAttribute {
    const attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(context, "index", integerAttr(context, record.index)),
        namedAttr(context, "instruction", integerAttr(context, record.graph_instruction_id.index)),
        namedAttr(context, "outputs", graphValueIdArrayAttr(context, record.output_value_ids)),
        namedAttr(context, "layout", stringAttr(context, @tagName(record.layout))),
        namedAttr(context, "tile", i64ArrayAttr(context, record.logical_tile_shape)),
        namedAttr(context, "result_memory", integerAttr(context, record.result_memory_space_id)),
        namedAttr(context, "tile_memory", optionalIntegerAttr(context, record.tile_memory_space_id)),
        namedAttr(context, "reason", stringAttr(context, record.reason)),
    };
    return mlir.mlirDictionaryAttrGet(context, attrs.len, &attrs);
}

fn parseCollectivePlanRecordAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    diagnostics: *std.Io.Writer,
) !compiler_facts.CollectivePlanRecord {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsADictionary(attr)) {
        try diagnostics.writeAll("pass=collective_extract feature=collective-record reason=record entry is not a dictionary attribute\n");
        return MlirStateError.InvalidCollectivePlan;
    }

    var result: compiler_facts.CollectivePlanRecord = .{
        .index = 0,
        .decision = .unsupported,
        .algorithm = .none,
        .checked_graph_instruction_count = 0,
        .lowered_collective_count = 0,
        .unsupported_collective_count = 0,
        .estimated_bytes = 0,
        .estimated_latency_ns = null,
        .reason = &.{},
    };
    errdefer if (result.reason.len > 0) allocator.free(result.reason);

    result.index = try collectiveU32FromIntegerAttr(dictAttr(attr, "index"), "index", diagnostics);
    result.decision = try collectiveDecisionFromStringAttr(dictAttr(attr, "decision"), diagnostics);
    result.algorithm = try collectiveAlgorithmFromStringAttr(dictAttr(attr, "algorithm"), diagnostics);
    result.checked_graph_instruction_count = try collectiveU32FromIntegerAttr(dictAttr(attr, "checked"), "checked", diagnostics);
    result.lowered_collective_count = try collectiveU32FromIntegerAttr(dictAttr(attr, "lowered"), "lowered", diagnostics);
    result.unsupported_collective_count = try collectiveU32FromIntegerAttr(dictAttr(attr, "unsupported"), "unsupported", diagnostics);
    result.estimated_bytes = try collectiveU128FromStringAttr(dictAttr(attr, "estimated_bytes"), "estimated_bytes", diagnostics);
    result.estimated_latency_ns = try collectiveOptionalU64FromStringAttr(dictAttr(attr, "estimated_latency_ns"), "estimated_latency_ns", diagnostics);
    result.reason = try collectiveStringFromAttr(allocator, dictAttr(attr, "reason"), "reason", diagnostics);
    return result;
}

fn parseLoweringRecordAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    diagnostics: *std.Io.Writer,
) !compiler_facts.LoweringRecord {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsADictionary(attr)) {
        try diagnostics.writeAll("pass=lowering_extract feature=lowering-record reason=record entry is not a dictionary attribute\n");
        return MlirStateError.InvalidLoweringPlan;
    }

    var result: compiler_facts.LoweringRecord = .{
        .id = .{ .index = 0 },
        .graph_instruction_ids = &.{},
        .decision = .unsupported,
        .reason = &.{},
        .rejected_alternatives = &.{},
        .cost_ledger_ids = &.{},
    };
    errdefer {
        if (result.graph_instruction_ids.len > 0) allocator.free(result.graph_instruction_ids);
        if (result.reason.len > 0) allocator.free(result.reason);
        for (result.rejected_alternatives) |alternative| allocator.free(alternative);
        if (result.rejected_alternatives.len > 0) allocator.free(result.rejected_alternatives);
        if (result.cost_ledger_ids.len > 0) allocator.free(result.cost_ledger_ids);
    }

    result.id = .{ .index = try loweringU32FromIntegerAttr(dictAttr(attr, "id"), "id", diagnostics) };
    result.graph_instruction_ids = try graphInstructionIdsFromArrayAttr(allocator, dictAttr(attr, "instructions"), "instructions", true, diagnostics);
    result.decision = try loweringDecisionFromStringAttr(dictAttr(attr, "decision"), diagnostics);
    result.reason = try loweringStringFromAttr(allocator, dictAttr(attr, "reason"), "reason", diagnostics);
    result.rejected_alternatives = try loweringStringsFromArrayAttr(allocator, dictAttr(attr, "rejected_alternatives"), "rejected_alternatives", false, diagnostics);
    result.cost_ledger_ids = try loweringCostIdsFromArrayAttr(allocator, dictAttr(attr, "costs"), "costs", true, diagnostics);
    return result;
}

fn parseLoweringRegionFactAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    diagnostics: *std.Io.Writer,
) !LoweringRegionFact {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsADictionary(attr)) {
        try diagnostics.writeAll("pass=lowering_region_extract feature=region-fact reason=fact entry is not a dictionary attribute\n");
        return MlirStateError.InvalidLoweringPlan;
    }

    var result: LoweringRegionFact = .{
        .lowering_record_id = .{ .index = 0 },
        .fusion_group_index = null,
        .placement_record_indices = &.{},
        .logical_tile_shape = &.{},
        .result_memory_space_id = 0,
        .tile_memory_space_id = null,
        .codegen_region = .unsupported,
        .reason = &.{},
    };
    errdefer deinitLoweringRegionFactFields(allocator, result);

    result.lowering_record_id = .{ .index = try loweringU32FromIntegerAttr(dictAttr(attr, "lowering"), "lowering", diagnostics) };
    result.fusion_group_index = try loweringOptionalU32FromIntegerAttr(dictAttr(attr, "fusion_group"), "fusion_group", diagnostics);
    result.placement_record_indices = try loweringU32sFromArrayAttr(allocator, dictAttr(attr, "placements"), "placements", true, diagnostics);
    result.logical_tile_shape = try loweringI64sFromArrayAttr(allocator, dictAttr(attr, "tile"), "tile", diagnostics);
    result.result_memory_space_id = try loweringU32FromIntegerAttr(dictAttr(attr, "result_memory"), "result_memory", diagnostics);
    result.tile_memory_space_id = try loweringOptionalU32FromIntegerAttr(dictAttr(attr, "tile_memory"), "tile_memory", diagnostics);
    result.codegen_region = try loweringDecisionFromStringAttr(dictAttr(attr, "codegen_region"), diagnostics);
    result.reason = try loweringStringFromAttr(allocator, dictAttr(attr, "reason"), "reason", diagnostics);
    return result;
}

fn parseKernelCodegenRecordAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    diagnostics: *std.Io.Writer,
) !core.KernelCodegenRecord {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsADictionary(attr)) {
        try diagnostics.writeAll("pass=codegen_extract feature=codegen-record reason=record entry is not a dictionary attribute\n");
        return MlirStateError.InvalidKernelCodegenPlan;
    }

    var result: core.KernelCodegenRecord = .{
        .id = .{ .index = 0 },
        .lowering_record_id = .{ .index = 0 },
        .command_id = .{ .index = 0 },
        .backend_kind = .npu_v0,
        .kind = .backend_kernel_graph,
        .operation = &.{},
        .shape = .{
            .operation_count = 0,
            .external_input_count = 0,
            .external_output_count = 0,
            .intermediate_value_count = 0,
        },
        .logical_tile_shape = &.{},
        .result_memory_space_id = 0,
        .tile_memory_space_id = null,
        .memory_pressure = .{
            .global_bytes_read = 0,
            .global_bytes_written = 0,
            .local_bytes_read = 0,
            .local_bytes_written = 0,
        },
        .external_input_ids = &.{},
        .external_output_ids = &.{},
        .intermediate_value_ids = &.{},
        .graph_instruction_ids = &.{},
        .cost_ledger_ids = &.{},
        .memory_traffic_ids = &.{},
        .expected_unit_id = null,
        .reason = &.{},
    };
    errdefer deinitKernelCodegenRecordFields(allocator, result, true);

    result.id = .{ .index = try codegenU32FromIntegerAttr(dictAttr(attr, "id"), "id", diagnostics) };
    result.lowering_record_id = .{ .index = try codegenU32FromIntegerAttr(dictAttr(attr, "lowering"), "lowering", diagnostics) };
    result.command_id = .{ .index = try codegenU32FromIntegerAttr(dictAttr(attr, "command"), "command", diagnostics) };
    result.backend_kind = try backendKindFromStringAttr(dictAttr(attr, "backend_kind"), diagnostics);
    result.kind = try kernelCodegenKindFromStringAttr(dictAttr(attr, "kind"), diagnostics);
    result.operation = try codegenStringFromAttr(allocator, dictAttr(attr, "operation"), "operation", diagnostics);
    result.shape = try kernelShapeFromAttr(dictAttr(attr, "shape"), diagnostics);
    result.logical_tile_shape = try i64sFromArrayAttr(allocator, dictAttr(attr, "tile"), diagnostics);
    result.result_memory_space_id = try codegenU32FromIntegerAttr(dictAttr(attr, "result_memory"), "result_memory", diagnostics);
    result.tile_memory_space_id = try codegenOptionalU32FromIntegerAttr(dictAttr(attr, "tile_memory"), "tile_memory", diagnostics);
    result.memory_pressure = try kernelPressureFromAttr(dictAttr(attr, "pressure"), diagnostics);
    result.external_input_ids = try graphValueIdsFromArrayAttrBounded(allocator, dictAttr(attr, "external_inputs"), "external_inputs", false, diagnostics);
    result.external_output_ids = try graphValueIdsFromArrayAttrBounded(allocator, dictAttr(attr, "external_outputs"), "external_outputs", false, diagnostics);
    result.intermediate_value_ids = try graphValueIdsFromArrayAttrBounded(allocator, dictAttr(attr, "intermediates"), "intermediates", false, diagnostics);
    result.graph_instruction_ids = try graphInstructionIdsFromArrayAttr(allocator, dictAttr(attr, "instructions"), "instructions", true, diagnostics);
    result.cost_ledger_ids = try costLedgerIdsFromArrayAttr(allocator, dictAttr(attr, "costs"), "costs", true, diagnostics);
    result.memory_traffic_ids = try memoryTrafficIdsFromArrayAttr(allocator, dictAttr(attr, "traffic"), "traffic", false, diagnostics);
    result.expected_unit_id = try codegenOptionalU32FromIntegerAttr(dictAttr(attr, "expected_unit"), "expected_unit", diagnostics);
    result.reason = try codegenStringFromAttr(allocator, dictAttr(attr, "reason"), "reason", diagnostics);
    return result;
}

fn parseScheduleCommandAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    diagnostics: *std.Io.Writer,
) !core.ScheduleCommand {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsADictionary(attr)) {
        try diagnostics.writeAll("pass=schedule_extract feature=command reason=command entry is not a dictionary attribute\n");
        return MlirStateError.InvalidSchedulePlan;
    }

    var result: core.ScheduleCommand = .{
        .id = .{ .index = 0 },
        .kind = .host_to_device,
        .stream = .{ .index = 0 },
        .inputs = &.{},
        .outputs = &.{},
        .dependencies = &.{},
        .lowering_record_ids = &.{},
        .cost_ledger_ids = &.{},
    };
    errdefer deinitScheduleCommandFields(allocator, result);

    result.id = .{ .index = try scheduleU32FromIntegerAttr(dictAttr(attr, "id"), "id", diagnostics) };
    result.kind = try commandKindFromStringAttr(dictAttr(attr, "kind"), diagnostics);
    result.stream = .{ .index = try scheduleU32FromIntegerAttr(dictAttr(attr, "stream"), "stream", diagnostics) };
    result.inputs = try scheduleGraphValueIdsFromArrayAttr(allocator, dictAttr(attr, "inputs"), "inputs", false, diagnostics);
    result.outputs = try scheduleGraphValueIdsFromArrayAttr(allocator, dictAttr(attr, "outputs"), "outputs", false, diagnostics);
    result.dependencies = try commandDependenciesFromArrayAttr(allocator, dictAttr(attr, "dependencies"), diagnostics);
    result.lowering_record_ids = try scheduleLoweringIdsFromArrayAttr(allocator, dictAttr(attr, "lowerings"), "lowerings", false, diagnostics);
    result.cost_ledger_ids = try scheduleCostIdsFromArrayAttr(allocator, dictAttr(attr, "costs"), "costs", false, diagnostics);
    return result;
}

fn parseScheduleOverlapAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    diagnostics: *std.Io.Writer,
) !core.ScheduleOverlapRecord {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsADictionary(attr)) {
        try diagnostics.writeAll("pass=schedule_extract feature=overlap reason=overlap entry is not a dictionary attribute\n");
        return MlirStateError.InvalidSchedulePlan;
    }

    var result: core.ScheduleOverlapRecord = .{
        .id = .{ .index = 0 },
        .decision = .serialized,
        .kind = .transfer_compute,
        .first_command_id = .{ .index = 0 },
        .second_command_id = .{ .index = 0 },
        .dependency_kind = .data,
        .first_stream = .{ .index = 0 },
        .second_stream = .{ .index = 0 },
        .reason = &.{},
    };
    errdefer if (result.reason.len > 0) allocator.free(result.reason);

    result.id = .{ .index = try scheduleU32FromIntegerAttr(dictAttr(attr, "id"), "id", diagnostics) };
    result.decision = try scheduleOverlapDecisionFromStringAttr(dictAttr(attr, "decision"), diagnostics);
    result.kind = try scheduleOverlapKindFromStringAttr(dictAttr(attr, "kind"), diagnostics);
    result.first_command_id = .{ .index = try scheduleU32FromIntegerAttr(dictAttr(attr, "first_command"), "first_command", diagnostics) };
    result.second_command_id = .{ .index = try scheduleU32FromIntegerAttr(dictAttr(attr, "second_command"), "second_command", diagnostics) };
    result.dependency_kind = try dependencyKindFromStringAttr(dictAttr(attr, "dependency_kind"), diagnostics);
    result.first_stream = .{ .index = try scheduleU32FromIntegerAttr(dictAttr(attr, "first_stream"), "first_stream", diagnostics) };
    result.second_stream = .{ .index = try scheduleU32FromIntegerAttr(dictAttr(attr, "second_stream"), "second_stream", diagnostics) };
    result.reason = try scheduleStringFromAttr(allocator, dictAttr(attr, "reason"), "reason", diagnostics);
    return result;
}

fn parseBackendBindingAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    diagnostics: *std.Io.Writer,
) !core.BackendBinding {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsADictionary(attr)) {
        try diagnostics.writeAll("pass=backend_binding_extract feature=binding reason=binding entry is not a dictionary attribute\n");
        return MlirStateError.InvalidBackendBindingPlan;
    }

    var result: core.BackendBinding = .{
        .id = .{ .index = 0 },
        .command_id = .{ .index = 0 },
        .backend_kind = .npu_v0,
        .backend_operation = &.{},
        .graph_instruction_ids = &.{},
        .expected_unit_id = null,
        .cost_ledger_ids = &.{},
    };
    errdefer deinitBackendBindingFields(allocator, result, true);

    result.id = .{ .index = try backendBindingU32FromIntegerAttr(dictAttr(attr, "id"), "id", diagnostics) };
    result.command_id = .{ .index = try backendBindingU32FromIntegerAttr(dictAttr(attr, "command"), "command", diagnostics) };
    result.backend_kind = try backendKindFromStringAttrForBinding(dictAttr(attr, "backend_kind"), diagnostics);
    result.backend_operation = try backendBindingStringFromAttr(allocator, dictAttr(attr, "operation"), "operation", diagnostics);
    result.graph_instruction_ids = try backendBindingGraphInstructionIdsFromArrayAttr(allocator, dictAttr(attr, "instructions"), "instructions", true, diagnostics);
    result.expected_unit_id = try backendBindingOptionalU32FromIntegerAttr(dictAttr(attr, "expected_unit"), "expected_unit", diagnostics);
    result.cost_ledger_ids = try backendBindingCostIdsFromArrayAttr(allocator, dictAttr(attr, "costs"), "costs", true, diagnostics);
    return result;
}

fn parseExecutableContractAttr(attr: mlir.MlirAttribute, diagnostics: *std.Io.Writer) !ExecutableContract {
    const contract = executableContractFromAttrNoDiag(attr) orelse {
        try diagnostics.writeAll("pass=executable_contract_extract feature=contract reason=contract entry is not a valid dictionary attribute\n");
        return MlirStateError.InvalidExecutableContract;
    };
    try verifyExecutableContract(contract, diagnostics);
    return contract;
}

fn parseTargetDescriptionAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    diagnostics: *std.Io.Writer,
) !target_pkg.TargetDescription {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsADictionary(attr)) {
        try diagnostics.writeAll("pass=target_extract feature=target reason=target is not a dictionary attribute\n");
        return MlirStateError.InvalidTargetAttachment;
    }
    var target: target_pkg.TargetDescription = .{
        .name = &.{},
        .kind = .npu_v0,
        .devices = &.{},
        .memory_spaces = &.{},
        .transfer_edges = &.{},
        .execution_units = &.{},
    };
    errdefer deinitExtractedTargetDescription(allocator, target);

    target.name = try targetStringFromAttr(allocator, dictAttr(attr, "name"), "name", diagnostics);
    target.kind = try targetKindFromStringAttr(dictAttr(attr, "kind"), diagnostics);
    target.devices = try targetDevicesFromArrayAttr(allocator, dictAttr(attr, "devices"), diagnostics);
    target.memory_spaces = try targetMemorySpacesFromArrayAttr(allocator, dictAttr(attr, "memory_spaces"), diagnostics);
    target.transfer_edges = try targetTransferEdgesFromArrayAttr(allocator, dictAttr(attr, "transfer_edges"), diagnostics);
    target.execution_units = try targetExecutionUnitsFromArrayAttr(allocator, dictAttr(attr, "execution_units"), diagnostics);
    return target;
}

fn parseTargetDeviceAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    diagnostics: *std.Io.Writer,
) !target_pkg.TargetDevice {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsADictionary(attr)) {
        try diagnostics.writeAll("pass=target_extract feature=device reason=device is not a dictionary attribute\n");
        return MlirStateError.InvalidTargetAttachment;
    }
    var device: target_pkg.TargetDevice = .{
        .id = 0,
        .local_hardware_id = 0,
        .name = &.{},
        .memory_space_ids = &.{},
        .execution_unit_ids = &.{},
    };
    errdefer {
        allocator.free(device.name);
        allocator.free(device.memory_space_ids);
        allocator.free(device.execution_unit_ids);
    }

    device.id = try targetU32FromIntegerAttr(dictAttr(attr, "id"), "id", diagnostics);
    device.local_hardware_id = try targetI32FromIntegerAttr(dictAttr(attr, "local_hardware_id"), "local_hardware_id", diagnostics);
    device.name = try targetStringFromAttr(allocator, dictAttr(attr, "name"), "name", diagnostics);
    device.memory_space_ids = try targetU32sFromArrayAttr(allocator, dictAttr(attr, "memory_spaces"), "memory_spaces", true, diagnostics);
    device.execution_unit_ids = try targetU32sFromArrayAttr(allocator, dictAttr(attr, "execution_units"), "execution_units", true, diagnostics);
    return device;
}

fn parseTargetMemorySpaceAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    diagnostics: *std.Io.Writer,
) !target_pkg.TargetMemorySpace {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsADictionary(attr)) {
        try diagnostics.writeAll("pass=target_extract feature=memory-space reason=memory space is not a dictionary attribute\n");
        return MlirStateError.InvalidTargetAttachment;
    }
    var memory_space: target_pkg.TargetMemorySpace = .{
        .id = 0,
        .name = &.{},
        .kind = .unknown,
        .capacity_bytes = null,
        .bandwidth_bytes_per_second = null,
        .note = &.{},
    };
    errdefer {
        allocator.free(memory_space.name);
        allocator.free(memory_space.note);
    }

    memory_space.id = try targetU32FromIntegerAttr(dictAttr(attr, "id"), "id", diagnostics);
    memory_space.name = try targetStringFromAttr(allocator, dictAttr(attr, "name"), "name", diagnostics);
    memory_space.kind = try memorySpaceKindFromStringAttr(dictAttr(attr, "kind"), diagnostics);
    memory_space.capacity_bytes = try targetOptionalU64FromStringAttr(dictAttr(attr, "capacity_bytes"), "capacity_bytes", diagnostics);
    memory_space.bandwidth_bytes_per_second = try targetOptionalF64FromStringAttr(dictAttr(attr, "bandwidth_bytes_per_second"), "bandwidth_bytes_per_second", diagnostics);
    memory_space.note = try targetStringFromAttr(allocator, dictAttr(attr, "note"), "note", diagnostics);
    return memory_space;
}

fn parseTargetTransferEdgeAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    diagnostics: *std.Io.Writer,
) !target_pkg.TargetTransferEdge {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsADictionary(attr)) {
        try diagnostics.writeAll("pass=target_extract feature=transfer-edge reason=transfer edge is not a dictionary attribute\n");
        return MlirStateError.InvalidTargetAttachment;
    }
    var edge: target_pkg.TargetTransferEdge = .{
        .id = 0,
        .src_memory_space = 0,
        .dst_memory_space = 0,
        .bandwidth_bytes_per_second = null,
        .latency_ns = null,
        .supports_async = false,
        .engine_unit_id = null,
        .note = &.{},
    };
    errdefer allocator.free(edge.note);

    edge.id = try targetU32FromIntegerAttr(dictAttr(attr, "id"), "id", diagnostics);
    edge.src_memory_space = try targetU32FromIntegerAttr(dictAttr(attr, "src_memory"), "src_memory", diagnostics);
    edge.dst_memory_space = try targetU32FromIntegerAttr(dictAttr(attr, "dst_memory"), "dst_memory", diagnostics);
    edge.bandwidth_bytes_per_second = try targetOptionalF64FromStringAttr(dictAttr(attr, "bandwidth_bytes_per_second"), "bandwidth_bytes_per_second", diagnostics);
    edge.latency_ns = try targetOptionalU64FromStringAttr(dictAttr(attr, "latency_ns"), "latency_ns", diagnostics);
    edge.supports_async = boolAttrValue(dictAttr(attr, "supports_async")) orelse {
        try diagnostics.writeAll("pass=target_extract feature=transfer-edge reason=expected supports_async bool attribute\n");
        return MlirStateError.InvalidTargetAttachment;
    };
    edge.engine_unit_id = try targetOptionalU32FromIntegerAttr(dictAttr(attr, "engine_unit"), "engine_unit", diagnostics);
    edge.note = try targetStringFromAttr(allocator, dictAttr(attr, "note"), "note", diagnostics);
    return edge;
}

fn parseTargetExecutionUnitAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    diagnostics: *std.Io.Writer,
) !target_pkg.ExecutionUnit {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsADictionary(attr)) {
        try diagnostics.writeAll("pass=target_extract feature=execution-unit reason=execution unit is not a dictionary attribute\n");
        return MlirStateError.InvalidTargetAttachment;
    }
    var unit: target_pkg.ExecutionUnit = .{
        .id = 0,
        .name = &.{},
        .kind = .unknown,
        .dtype_rates = &.{},
    };
    errdefer {
        allocator.free(unit.name);
        for (unit.dtype_rates) |rate| allocator.free(rate.note);
        allocator.free(unit.dtype_rates);
    }

    unit.id = try targetU32FromIntegerAttr(dictAttr(attr, "id"), "id", diagnostics);
    unit.name = try targetStringFromAttr(allocator, dictAttr(attr, "name"), "name", diagnostics);
    unit.kind = try executionUnitKindFromStringAttr(dictAttr(attr, "kind"), diagnostics);
    unit.dtype_rates = try targetDTypeRatesFromArrayAttr(allocator, dictAttr(attr, "dtype_rates"), diagnostics);
    return unit;
}

fn parseTargetDTypeRateAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    diagnostics: *std.Io.Writer,
) !target_pkg.DTypeRate {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsADictionary(attr)) {
        try diagnostics.writeAll("pass=target_extract feature=dtype-rate reason=dtype rate is not a dictionary attribute\n");
        return MlirStateError.InvalidTargetAttachment;
    }
    var rate: target_pkg.DTypeRate = .{
        .dtype = .invalid,
        .op_class = .elementwise,
        .ops_per_second = null,
        .source = .unknown,
        .note = &.{},
    };
    errdefer allocator.free(rate.note);

    rate.dtype = try bufferTypeFromStringAttrForTarget(dictAttr(attr, "dtype"), diagnostics);
    rate.op_class = try opClassFromStringAttr(dictAttr(attr, "op_class"), diagnostics);
    rate.ops_per_second = try targetOptionalF64FromStringAttr(dictAttr(attr, "ops_per_second"), "ops_per_second", diagnostics);
    rate.source = try rateSourceFromStringAttr(dictAttr(attr, "source"), diagnostics);
    rate.note = try targetStringFromAttr(allocator, dictAttr(attr, "note"), "note", diagnostics);
    return rate;
}

fn parseCostLedgerEntryAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    diagnostics: *std.Io.Writer,
) !compiler_facts.CostLedgerEntry {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsADictionary(attr)) {
        try diagnostics.writeAll("pass=performance_extract feature=cost-ledger reason=cost entry is not a dictionary attribute\n");
        return MlirStateError.InvalidKernelCodegenPlan;
    }
    var entry: compiler_facts.CostLedgerEntry = .{
        .id = .{ .index = 0 },
        .source = null,
        .graph_instruction_ids = &.{},
        .op_class = .backend_kernel,
        .dtype = .invalid,
        .accumulation_dtype = null,
        .logical_ops = 0,
        .bytes_read = 0,
        .bytes_written = 0,
        .expected_unit_id = null,
        .formula = &.{},
        .approximation = &.{},
    };
    errdefer deinitCostLedgerEntryFields(allocator, entry);

    entry.id = .{ .index = try performanceU32FromIntegerAttr(dictAttr(attr, "id"), "id", diagnostics) };
    entry.source = try optionalSourceRefFromAttr(allocator, dictAttr(attr, "source"), diagnostics);
    entry.graph_instruction_ids = try performanceGraphInstructionIdsFromArrayAttr(allocator, dictAttr(attr, "instructions"), "instructions", true, diagnostics);
    entry.op_class = try costOpClassFromStringAttr(dictAttr(attr, "op_class"), diagnostics);
    entry.dtype = try bufferTypeFromStringAttrForPerformance(dictAttr(attr, "dtype"), diagnostics);
    entry.accumulation_dtype = try optionalBufferTypeFromStringAttrForPerformance(dictAttr(attr, "accumulation_dtype"), diagnostics);
    entry.logical_ops = try performanceU128FromStringAttr(dictAttr(attr, "logical_ops"), "logical_ops", diagnostics);
    entry.bytes_read = try performanceU128FromStringAttr(dictAttr(attr, "bytes_read"), "bytes_read", diagnostics);
    entry.bytes_written = try performanceU128FromStringAttr(dictAttr(attr, "bytes_written"), "bytes_written", diagnostics);
    entry.expected_unit_id = try performanceOptionalU32FromIntegerAttr(dictAttr(attr, "expected_unit"), "expected_unit", diagnostics);
    entry.formula = try performanceStringFromAttr(allocator, dictAttr(attr, "formula"), "formula", diagnostics);
    entry.approximation = try performanceStringFromAttr(allocator, dictAttr(attr, "approximation"), "approximation", diagnostics);
    return entry;
}

fn parseMemoryTrafficRecordAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    diagnostics: *std.Io.Writer,
) !compiler_facts.MemoryTrafficRecord {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsADictionary(attr)) {
        try diagnostics.writeAll("pass=performance_extract feature=memory-traffic reason=traffic record is not a dictionary attribute\n");
        return MlirStateError.InvalidKernelCodegenPlan;
    }
    var record: compiler_facts.MemoryTrafficRecord = .{
        .id = .{ .index = 0 },
        .lowering_record_id = .{ .index = 0 },
        .memory_space_id = 0,
        .kind = .global_memory,
        .graph_instruction_ids = &.{},
        .cost_ledger_ids = &.{},
        .bytes_read = 0,
        .bytes_written = 0,
        .reason = &.{},
    };
    errdefer deinitMemoryTrafficRecordFields(allocator, record);

    record.id = .{ .index = try performanceU32FromIntegerAttr(dictAttr(attr, "id"), "id", diagnostics) };
    record.lowering_record_id = .{ .index = try performanceU32FromIntegerAttr(dictAttr(attr, "lowering"), "lowering", diagnostics) };
    record.memory_space_id = try performanceU32FromIntegerAttr(dictAttr(attr, "memory"), "memory", diagnostics);
    record.kind = try memoryTrafficKindFromStringAttr(dictAttr(attr, "kind"), diagnostics);
    record.graph_instruction_ids = try performanceGraphInstructionIdsFromArrayAttr(allocator, dictAttr(attr, "instructions"), "instructions", true, diagnostics);
    record.cost_ledger_ids = try performanceCostIdsFromArrayAttr(allocator, dictAttr(attr, "costs"), "costs", true, diagnostics);
    record.bytes_read = try performanceU128FromStringAttr(dictAttr(attr, "bytes_read"), "bytes_read", diagnostics);
    record.bytes_written = try performanceU128FromStringAttr(dictAttr(attr, "bytes_written"), "bytes_written", diagnostics);
    record.reason = try performanceStringFromAttr(allocator, dictAttr(attr, "reason"), "reason", diagnostics);
    return record;
}

fn optionalSourceRefFromAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    diagnostics: *std.Io.Writer,
) !?compiler_facts.SourceRef {
    if (mlir.mlirAttributeIsNull(attr) or stringAttrEquals(attr, "none")) return null;
    if (!mlir.mlirAttributeIsADictionary(attr)) {
        try diagnostics.writeAll("pass=performance_extract feature=source reason=source is not a dictionary attribute\n");
        return MlirStateError.InvalidKernelCodegenPlan;
    }
    var source: compiler_facts.SourceRef = .{
        .id = .{ .index = 0 },
        .frontend = .internal,
        .op_name = &.{},
        .source_index = 0,
        .location = &.{},
    };
    errdefer {
        allocator.free(source.op_name);
        allocator.free(source.location);
    }
    source.id = .{ .index = try performanceU32FromIntegerAttr(dictAttr(attr, "id"), "source.id", diagnostics) };
    source.frontend = try sourceFrontendFromStringAttr(dictAttr(attr, "frontend"), diagnostics);
    source.op_name = try performanceStringFromAttr(allocator, dictAttr(attr, "op_name"), "source.op_name", diagnostics);
    source.source_index = try performanceU32FromIntegerAttr(dictAttr(attr, "source_index"), "source.source_index", diagnostics);
    source.location = try performanceOptionalStringFromAttr(allocator, dictAttr(attr, "location"), "source.location", diagnostics);
    return source;
}

fn parseBackendExecutablePlanAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    diagnostics: *std.Io.Writer,
) !BackendExecutablePlanFact {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsADictionary(attr)) {
        try diagnostics.writeAll("pass=backend_executable_extract feature=plan reason=plan is not a dictionary attribute\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    var plan: BackendExecutablePlanFact = .{
        .backend_kind = .npu_v0,
        .command_id = .{ .index = 0 },
        .backend_operation = &.{},
        .calls = &.{},
    };
    errdefer deinitExtractedBackendExecutablePlan(allocator, plan);

    plan.backend_kind = try backendKindFromStringAttrForExecutable(dictAttr(attr, "backend_kind"), diagnostics);
    plan.command_id = .{ .index = try backendPlanU32FromIntegerAttr(dictAttr(attr, "command"), "command", diagnostics) };
    plan.backend_operation = try backendPlanStringFromAttr(allocator, dictAttr(attr, "operation"), "operation", diagnostics);
    plan.calls = try backendExecutableCallsFromArrayAttr(allocator, dictAttr(attr, "calls"), diagnostics);
    return plan;
}

fn parseBackendExecutableCallAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    diagnostics: *std.Io.Writer,
) !BackendExecutableCallFact {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsADictionary(attr)) {
        try diagnostics.writeAll("pass=backend_executable_extract feature=call reason=call is not a dictionary attribute\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    var call: BackendExecutableCallFact = .{
        .index = 0,
        .command_id = .{ .index = 0 },
        .backend_kind = .npu_v0,
        .graph_instruction_id = .{ .index = 0 },
        .graph_instruction_ids = &.{},
        .feature = &.{},
        .backend_operation = &.{},
        .input_value_ids = &.{},
        .output_value_ids = &.{},
        .expected_unit_id = null,
    };
    errdefer deinitBackendExecutableCallFields(allocator, call);

    call.index = try backendPlanU32FromIntegerAttr(dictAttr(attr, "index"), "index", diagnostics);
    call.command_id = .{ .index = try backendPlanU32FromIntegerAttr(dictAttr(attr, "command"), "command", diagnostics) };
    call.backend_kind = try backendKindFromStringAttrForExecutable(dictAttr(attr, "backend_kind"), diagnostics);
    call.graph_instruction_id = .{ .index = try backendPlanU32FromIntegerAttr(dictAttr(attr, "instruction"), "instruction", diagnostics) };
    call.graph_instruction_ids = try backendPlanGraphInstructionIdsFromArrayAttr(allocator, dictAttr(attr, "instructions"), "instructions", true, diagnostics);
    call.feature = try backendPlanStringFromAttr(allocator, dictAttr(attr, "feature"), "feature", diagnostics);
    call.backend_operation = try backendPlanStringFromAttr(allocator, dictAttr(attr, "operation"), "operation", diagnostics);
    call.input_value_ids = try backendPlanGraphValueIdsFromArrayAttr(allocator, dictAttr(attr, "inputs"), "inputs", true, diagnostics);
    call.output_value_ids = try backendPlanGraphValueIdsFromArrayAttr(allocator, dictAttr(attr, "outputs"), "outputs", true, diagnostics);
    call.expected_unit_id = try backendPlanOptionalU32FromIntegerAttr(dictAttr(attr, "expected_unit"), "expected_unit", diagnostics);
    return call;
}

fn parseBackendKernelGraphAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    diagnostics: *std.Io.Writer,
) !BackendKernelGraphFact {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsADictionary(attr)) {
        try diagnostics.writeAll("pass=backend_kernel_graph_extract feature=graph reason=graph is not a dictionary attribute\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    var graph: BackendKernelGraphFact = .{
        .backend_kind = .metal_v0,
        .command_id = .{ .index = 0 },
        .nodes = &.{},
        .edges = &.{},
    };
    errdefer deinitExtractedBackendKernelGraph(allocator, graph);

    graph.backend_kind = try backendKindFromStringAttrForExecutable(dictAttr(attr, "backend_kind"), diagnostics);
    graph.command_id = .{ .index = try backendPlanU32FromIntegerAttr(dictAttr(attr, "command"), "command", diagnostics) };
    graph.nodes = try backendKernelGraphNodesFromArrayAttr(allocator, dictAttr(attr, "nodes"), diagnostics);
    graph.edges = try backendKernelGraphEdgesFromArrayAttr(allocator, dictAttr(attr, "edges"), diagnostics);
    return graph;
}

fn parseBackendKernelGraphNodeAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    diagnostics: *std.Io.Writer,
) !BackendKernelGraphNodeFact {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsADictionary(attr)) {
        try diagnostics.writeAll("pass=backend_kernel_graph_extract feature=node reason=node is not a dictionary attribute\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    var node: BackendKernelGraphNodeFact = .{
        .index = 0,
        .call_index = 0,
        .graph_instruction_id = .{ .index = 0 },
        .graph_instruction_ids = &.{},
        .feature = &.{},
        .backend_operation = &.{},
        .input_value_ids = &.{},
        .output_value_ids = &.{},
        .output_type = .{ .element_type = .invalid, .dims = &.{}, .layout = .dense_row_major },
        .attributes = &.{},
    };
    errdefer deinitBackendKernelGraphNodeFields(allocator, node);

    node.index = try backendPlanU32FromIntegerAttr(dictAttr(attr, "index"), "index", diagnostics);
    node.call_index = try backendPlanU32FromIntegerAttr(dictAttr(attr, "call"), "call", diagnostics);
    node.graph_instruction_id = .{ .index = try backendPlanU32FromIntegerAttr(dictAttr(attr, "instruction"), "instruction", diagnostics) };
    node.graph_instruction_ids = try backendPlanGraphInstructionIdsFromArrayAttr(allocator, dictAttr(attr, "instructions"), "instructions", true, diagnostics);
    node.feature = try backendPlanStringFromAttr(allocator, dictAttr(attr, "feature"), "feature", diagnostics);
    node.backend_operation = try backendPlanStringFromAttr(allocator, dictAttr(attr, "operation"), "operation", diagnostics);
    node.input_value_ids = try backendPlanGraphValueIdsFromArrayAttr(allocator, dictAttr(attr, "inputs"), "inputs", true, diagnostics);
    node.output_value_ids = try backendPlanGraphValueIdsFromArrayAttr(allocator, dictAttr(attr, "outputs"), "outputs", true, diagnostics);
    node.output_type = try backendTensorDescriptorFromAttr(allocator, dictAttr(attr, "output_type"), diagnostics);
    node.attributes = try backendPlanStringFromAttr(allocator, dictAttr(attr, "attributes"), "attributes", diagnostics);
    return node;
}

fn parseBackendKernelGraphEdgeAttr(attr: mlir.MlirAttribute, diagnostics: *std.Io.Writer) !BackendKernelGraphEdgeFact {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsADictionary(attr)) {
        try diagnostics.writeAll("pass=backend_kernel_graph_extract feature=edge reason=edge is not a dictionary attribute\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    return .{
        .value_id = .{ .index = try backendPlanU32FromIntegerAttr(dictAttr(attr, "value"), "value", diagnostics) },
        .src_node_index = try backendPlanU32FromIntegerAttr(dictAttr(attr, "src"), "src", diagnostics),
        .dst_node_index = try backendPlanU32FromIntegerAttr(dictAttr(attr, "dst"), "dst", diagnostics),
    };
}

fn backendTensorDescriptorFromAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    diagnostics: *std.Io.Writer,
) !BackendTensorDescriptorFact {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsADictionary(attr)) {
        try diagnostics.writeAll("pass=backend_kernel_graph_extract feature=output-type reason=output type is not a dictionary attribute\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    return .{
        .element_type = try bufferTypeFromStringAttrForBackendPlan(dictAttr(attr, "dtype"), diagnostics),
        .dims = try backendPlanI64sFromArrayAttr(allocator, dictAttr(attr, "dims"), "dims", diagnostics),
        .layout = try layoutFromStringAttrForBackendPlan(dictAttr(attr, "layout"), diagnostics),
    };
}

fn parseRuntimeAllocationPlanAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    diagnostics: *std.Io.Writer,
) !RuntimeAllocationPlanFact {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsADictionary(attr)) {
        try diagnostics.writeAll("pass=runtime_allocation_extract feature=plan reason=plan is not a dictionary attribute\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }

    var plan: RuntimeAllocationPlanFact = .{
        .allocations = &.{},
        .command_buffer_uses = &.{},
        .peak_device_bytes = 0,
    };
    errdefer deinitExtractedRuntimeAllocationPlan(allocator, plan);

    plan.allocations = try runtimeAllocationsFromArrayAttr(allocator, dictAttr(attr, "allocations"), diagnostics);
    plan.command_buffer_uses = try runtimeBufferUsesFromArrayAttr(allocator, dictAttr(attr, "command_buffer_uses"), diagnostics);
    plan.peak_device_bytes = try runtimeAllocationU128FromStringAttr(dictAttr(attr, "peak_device_bytes"), "peak_device_bytes", diagnostics);
    return plan;
}

fn parseRuntimeAllocationAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    diagnostics: *std.Io.Writer,
) !RuntimeAllocationFact {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsADictionary(attr)) {
        try diagnostics.writeAll("pass=runtime_allocation_extract feature=allocation reason=allocation is not a dictionary attribute\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }

    var allocation: RuntimeAllocationFact = .{
        .index = 0,
        .value_id = .{ .index = 0 },
        .placement = &.{},
        .memory_space_id = 0,
        .size_bytes = 0,
        .first_command_id = .{ .index = 0 },
        .last_command_id = .{ .index = 0 },
    };
    errdefer if (allocation.placement.len > 0) allocator.free(allocation.placement);

    allocation.index = try runtimeAllocationU32FromIntegerAttr(dictAttr(attr, "index"), "index", diagnostics);
    allocation.value_id = .{ .index = try runtimeAllocationU32FromIntegerAttr(dictAttr(attr, "value"), "value", diagnostics) };
    allocation.placement = try runtimeAllocationStringFromAttr(allocator, dictAttr(attr, "placement"), "placement", diagnostics);
    allocation.memory_space_id = try runtimeAllocationU32FromIntegerAttr(dictAttr(attr, "memory"), "memory", diagnostics);
    allocation.size_bytes = try runtimeAllocationU128FromStringAttr(dictAttr(attr, "bytes"), "bytes", diagnostics);
    allocation.first_command_id = .{ .index = try runtimeAllocationU32FromIntegerAttr(dictAttr(attr, "first_command"), "first_command", diagnostics) };
    allocation.last_command_id = .{ .index = try runtimeAllocationU32FromIntegerAttr(dictAttr(attr, "last_command"), "last_command", diagnostics) };
    return allocation;
}

fn parseRuntimeBufferUseAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    diagnostics: *std.Io.Writer,
) !RuntimeBufferUseFact {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsADictionary(attr)) {
        try diagnostics.writeAll("pass=runtime_allocation_extract feature=buffer-use reason=buffer use is not a dictionary attribute\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }

    var use: RuntimeBufferUseFact = .{
        .command_id = .{ .index = 0 },
        .buffer_index = 0,
        .access = &.{},
    };
    errdefer if (use.access.len > 0) allocator.free(use.access);

    use.command_id = .{ .index = try runtimeAllocationU32FromIntegerAttr(dictAttr(attr, "command"), "command", diagnostics) };
    use.buffer_index = try runtimeAllocationU32FromIntegerAttr(dictAttr(attr, "buffer"), "buffer", diagnostics);
    use.access = try runtimeAllocationStringFromAttr(allocator, dictAttr(attr, "access"), "access", diagnostics);
    return use;
}

fn parseRuntimeStreamStepAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    diagnostics: *std.Io.Writer,
) !RuntimeStreamStepFact {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsADictionary(attr)) {
        try diagnostics.writeAll("pass=runtime_stream_extract feature=step reason=step is not a dictionary attribute\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }

    var step: RuntimeStreamStepFact = .{
        .command_id = .{ .index = 0 },
        .stream = .{ .index = 0 },
        .wait_event_ids = &.{},
        .start_event_id = 0,
        .done_event_id = 0,
    };
    errdefer if (step.wait_event_ids.len > 0) allocator.free(step.wait_event_ids);

    step.command_id = .{ .index = try runtimeStreamU32FromIntegerAttr(dictAttr(attr, "command"), "command", diagnostics) };
    step.stream = .{ .index = try runtimeStreamU32FromIntegerAttr(dictAttr(attr, "stream"), "stream", diagnostics) };
    step.wait_event_ids = try u32sFromArrayAttr(allocator, dictAttr(attr, "wait_events"), "wait_events", limits.max_runtime_wait_events, diagnostics);
    step.start_event_id = try runtimeStreamU32FromIntegerAttr(dictAttr(attr, "start_event"), "start_event", diagnostics);
    step.done_event_id = try runtimeStreamU32FromIntegerAttr(dictAttr(attr, "done_event"), "done_event", diagnostics);
    return step;
}

fn parseRuntimeProfileEventAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    diagnostics: *std.Io.Writer,
) !RuntimeProfileEventFact {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsADictionary(attr)) {
        try diagnostics.writeAll("pass=runtime_profile_extract feature=event reason=event entry is not a dictionary attribute\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }

    var result: RuntimeProfileEventFact = .{
        .index = 0,
        .command_id = null,
        .graph_instruction_ids = &.{},
        .kind = &.{},
        .start_ns = 0,
        .duration_ns = 0,
        .bytes = 0,
        .logical_ops = 0,
        .status = &.{},
        .forced_synchronization = false,
    };
    errdefer deinitRuntimeProfileEventFields(allocator, result);

    result.index = try runtimeProfileU32FromIntegerAttr(dictAttr(attr, "index"), "index", diagnostics);
    result.command_id = if (try runtimeProfileOptionalU32FromIntegerAttr(dictAttr(attr, "command"), "command", diagnostics)) |index|
        .{ .index = index }
    else
        null;
    result.graph_instruction_ids = try profileJoinGraphInstructionIdsFromArrayAttr(allocator, dictAttr(attr, "instructions"), "instructions", false, diagnostics);
    result.kind = try runtimeProfileStringFromAttr(allocator, dictAttr(attr, "kind"), "kind", diagnostics);
    result.start_ns = try runtimeProfileU64FromStringAttr(dictAttr(attr, "start_ns"), "start_ns", diagnostics);
    result.duration_ns = try runtimeProfileU64FromStringAttr(dictAttr(attr, "duration_ns"), "duration_ns", diagnostics);
    result.bytes = try runtimeProfileU128FromStringAttr(dictAttr(attr, "bytes"), "bytes", diagnostics);
    result.logical_ops = try runtimeProfileU128FromStringAttr(dictAttr(attr, "logical_ops"), "logical_ops", diagnostics);
    result.status = try runtimeProfileStringFromAttr(allocator, dictAttr(attr, "status"), "status", diagnostics);
    result.forced_synchronization = boolAttrValue(dictAttr(attr, "forced_sync")) orelse {
        try diagnostics.writeAll("pass=runtime_profile_extract feature=forced_sync reason=expected bool attribute\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    };
    return result;
}

fn parseRuntimeProfileJoinAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    diagnostics: *std.Io.Writer,
) !RuntimeProfileJoinFact {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsADictionary(attr)) {
        try diagnostics.writeAll("pass=runtime_profile_join_extract feature=join reason=join entry is not a dictionary attribute\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }

    var result: RuntimeProfileJoinFact = .{
        .index = 0,
        .subject_kind = &.{},
        .subject_id = 0,
        .command_id = null,
        .graph_instruction_ids = &.{},
        .profile_event_ids = &.{},
    };
    errdefer {
        if (result.subject_kind.len > 0) allocator.free(result.subject_kind);
        if (result.graph_instruction_ids.len > 0) allocator.free(result.graph_instruction_ids);
        if (result.profile_event_ids.len > 0) allocator.free(result.profile_event_ids);
    }

    result.index = try profileJoinU32FromIntegerAttr(dictAttr(attr, "index"), "index", diagnostics);
    result.subject_kind = try profileJoinStringFromAttr(allocator, dictAttr(attr, "subject_kind"), "subject_kind", diagnostics);
    result.subject_id = try profileJoinU32FromIntegerAttr(dictAttr(attr, "subject_id"), "subject_id", diagnostics);
    result.command_id = if (try profileJoinOptionalU32FromIntegerAttr(dictAttr(attr, "command"), "command", diagnostics)) |index|
        .{ .index = index }
    else
        null;
    result.graph_instruction_ids = try profileJoinGraphInstructionIdsFromArrayAttr(allocator, dictAttr(attr, "instructions"), "instructions", false, diagnostics);
    result.profile_event_ids = try profileEventIdsFromArrayAttr(allocator, dictAttr(attr, "events"), "events", diagnostics);
    return result;
}

fn parseBackendProfileJoinAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    diagnostics: *std.Io.Writer,
) !BackendProfileJoinFact {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsADictionary(attr)) {
        try diagnostics.writeAll("pass=backend_profile_join_extract feature=join reason=join entry is not a dictionary attribute\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }

    var result: BackendProfileJoinFact = .{
        .index = 0,
        .call_index = 0,
        .command_id = .{ .index = 0 },
        .graph_instruction_ids = &.{},
        .profile_event_id = .{ .index = 0 },
    };
    errdefer if (result.graph_instruction_ids.len > 0) allocator.free(result.graph_instruction_ids);

    result.index = try profileJoinU32FromIntegerAttr(dictAttr(attr, "index"), "index", diagnostics);
    result.call_index = try profileJoinU32FromIntegerAttr(dictAttr(attr, "call"), "call", diagnostics);
    result.command_id = .{ .index = try profileJoinU32FromIntegerAttr(dictAttr(attr, "command"), "command", diagnostics) };
    result.profile_event_id = .{ .index = try profileJoinU32FromIntegerAttr(dictAttr(attr, "event"), "event", diagnostics) };
    result.graph_instruction_ids = try profileJoinGraphInstructionIdsFromArrayAttr(allocator, dictAttr(attr, "instructions"), "instructions", true, diagnostics);
    return result;
}

fn parsePlacementRecordAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    diagnostics: *std.Io.Writer,
) !compiler_facts.PlacementRecord {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsADictionary(attr)) {
        try diagnostics.writeAll("pass=placement_extract feature=placement-record reason=record entry is not a dictionary attribute\n");
        return MlirStateError.InvalidPlacementPlan;
    }

    var result: compiler_facts.PlacementRecord = .{
        .index = 0,
        .graph_instruction_id = .{ .index = 0 },
        .output_value_ids = &.{},
        .layout = .dense_row_major,
        .logical_tile_shape = &.{},
        .result_memory_space_id = 0,
        .tile_memory_space_id = null,
        .reason = &.{},
    };
    errdefer {
        if (result.output_value_ids.len > 0) allocator.free(result.output_value_ids);
        if (result.logical_tile_shape.len > 0) allocator.free(result.logical_tile_shape);
        if (result.reason.len > 0) allocator.free(result.reason);
    }

    result.index = try placementU32FromIntegerAttr(dictAttr(attr, "index"), "index", diagnostics);
    result.graph_instruction_id = .{ .index = try placementU32FromIntegerAttr(dictAttr(attr, "instruction"), "instruction", diagnostics) };
    result.output_value_ids = try graphValueIdsFromArrayAttr(allocator, dictAttr(attr, "outputs"), diagnostics);
    result.layout = try layoutFromStringAttr(dictAttr(attr, "layout"), diagnostics);
    result.logical_tile_shape = try i64sFromArrayAttr(allocator, dictAttr(attr, "tile"), diagnostics);
    result.result_memory_space_id = try placementU32FromIntegerAttr(dictAttr(attr, "result_memory"), "result_memory", diagnostics);
    result.tile_memory_space_id = try optionalU32FromIntegerAttr(dictAttr(attr, "tile_memory"), "tile_memory", diagnostics);
    result.reason = try placementStringFromAttr(allocator, dictAttr(attr, "reason"), "reason", diagnostics);
    return result;
}

fn parseFusionCandidateDecisionAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    diagnostics: *std.Io.Writer,
) !compiler_facts.FusionGroup {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsADictionary(attr)) {
        try diagnostics.writeAll("pass=fusion_extract feature=fusion-candidate reason=candidate entry is not a dictionary attribute\n");
        return MlirStateError.InvalidFusionPlan;
    }

    var result: compiler_facts.FusionGroup = .{
        .index = 0,
        .decision = .rejected,
        .kind = &.{},
        .graph_instruction_ids = &.{},
        .bytes_saved = 0,
        .launch_count_reduction = 0,
        .pressure_delta = .{},
        .reason = &.{},
    };
    errdefer {
        if (result.kind.len > 0) allocator.free(result.kind);
        if (result.graph_instruction_ids.len > 0) allocator.free(result.graph_instruction_ids);
        if (result.reason.len > 0) allocator.free(result.reason);
    }

    result.index = try u32FromIntegerAttr(dictAttr(attr, "plan_index"), "plan_index", diagnostics);
    const decision = try stringFromAttr(allocator, dictAttr(attr, "decision"), "decision", diagnostics);
    defer allocator.free(decision);
    result.decision = if (std.mem.eql(u8, decision, "accepted"))
        .accepted
    else if (std.mem.eql(u8, decision, "rejected"))
        .rejected
    else {
        try diagnostics.writeAll("pass=fusion_extract feature=fusion-candidate reason=unknown decision\n");
        return MlirStateError.InvalidFusionPlan;
    };
    result.kind = try stringFromAttr(allocator, dictAttr(attr, "kind"), "kind", diagnostics);
    result.graph_instruction_ids = try instructionIdsFromArrayAttr(allocator, dictAttr(attr, "instructions"), diagnostics);
    result.bytes_saved = try u128FromStringAttr(dictAttr(attr, "bytes_saved"), "bytes_saved", diagnostics);
    result.launch_count_reduction = try u32FromIntegerAttr(dictAttr(attr, "launch_count_reduction"), "launch_count_reduction", diagnostics);
    result.pressure_delta = try pressureFromAttr(dictAttr(attr, "pressure_delta"), diagnostics);
    result.reason = try stringFromAttr(allocator, dictAttr(attr, "decision_reason"), "decision_reason", diagnostics);

    try verifyFusionGroups(&.{result}, diagnostics);
    return result;
}

fn pressureFromAttr(attr: mlir.MlirAttribute, diagnostics: *std.Io.Writer) !compiler_facts.FusionPressureDelta {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsADictionary(attr)) {
        try diagnostics.writeAll("pass=fusion_extract feature=pressure reason=pressure delta is not a dictionary attribute\n");
        return MlirStateError.InvalidFusionPlan;
    }
    return .{
        .split_kernel_count = try u32FromIntegerAttr(dictAttr(attr, "split_kernel_count"), "split_kernel_count", diagnostics),
        .fused_kernel_count = try u32FromIntegerAttr(dictAttr(attr, "fused_kernel_count"), "fused_kernel_count", diagnostics),
        .split_peak_live_bytes = try u128FromStringAttr(dictAttr(attr, "split_peak_live_bytes"), "split_peak_live_bytes", diagnostics),
        .fused_live_bytes = try u128FromStringAttr(dictAttr(attr, "fused_live_bytes"), "fused_live_bytes", diagnostics),
        .additional_live_bytes = try u128FromStringAttr(dictAttr(attr, "additional_live_bytes"), "additional_live_bytes", diagnostics),
        .global_bytes_saved = try u128FromStringAttr(dictAttr(attr, "global_bytes_saved"), "global_bytes_saved", diagnostics),
    };
}

fn instructionIdsFromArrayAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    diagnostics: *std.Io.Writer,
) ![]compiler_facts.GraphInstructionId {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAArray(attr)) {
        try diagnostics.writeAll("pass=fusion_extract feature=instructions reason=instructions field is not an array attribute\n");
        return MlirStateError.InvalidFusionPlan;
    }
    const count = mlir.mlirArrayAttrGetNumElements(attr);
    const ids = try allocator.alloc(compiler_facts.GraphInstructionId, std.math.cast(usize, count) orelse unreachable);
    errdefer allocator.free(ids);
    var index: isize = 0;
    while (index < count) : (index += 1) {
        const id_index = std.math.cast(usize, index) orelse unreachable;
        ids[id_index] = .{
            .index = try u32FromIntegerAttr(mlir.mlirArrayAttrGetElement(attr, index), "instructions", diagnostics),
        };
    }
    return ids;
}

fn graphValueIdsFromArrayAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    diagnostics: *std.Io.Writer,
) ![]compiler_facts.GraphValueId {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAArray(attr)) {
        try diagnostics.writeAll("pass=placement_extract feature=outputs reason=outputs field is not an array attribute\n");
        return MlirStateError.InvalidPlacementPlan;
    }
    const count = mlir.mlirArrayAttrGetNumElements(attr);
    if (count <= 0 or count > limits.max_placement_outputs) {
        try diagnostics.writeAll("pass=placement_extract feature=outputs reason=invalid output count\n");
        return MlirStateError.InvalidPlacementPlan;
    }
    const ids = try allocator.alloc(compiler_facts.GraphValueId, std.math.cast(usize, count) orelse unreachable);
    errdefer allocator.free(ids);
    var index: isize = 0;
    while (index < count) : (index += 1) {
        const id_index = std.math.cast(usize, index) orelse unreachable;
        ids[id_index] = .{
            .index = try placementU32FromIntegerAttr(mlir.mlirArrayAttrGetElement(attr, index), "outputs", diagnostics),
        };
    }
    return ids;
}

fn graphValueIdsFromArrayAttrBounded(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    field: []const u8,
    require_non_empty: bool,
    diagnostics: *std.Io.Writer,
) ![]compiler_facts.GraphValueId {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAArray(attr)) {
        try diagnostics.print("pass=codegen_extract feature={s} reason=field is not an array attribute\n", .{field});
        return MlirStateError.InvalidKernelCodegenPlan;
    }
    const count = mlir.mlirArrayAttrGetNumElements(attr);
    if ((require_non_empty and count <= 0) or count > limits.max_kernel_ref_ids) {
        try diagnostics.print("pass=codegen_extract feature={s} reason=invalid reference count\n", .{field});
        return MlirStateError.InvalidKernelCodegenPlan;
    }
    const ids = try allocator.alloc(compiler_facts.GraphValueId, std.math.cast(usize, count) orelse unreachable);
    errdefer allocator.free(ids);
    var index: isize = 0;
    while (index < count) : (index += 1) {
        ids[std.math.cast(usize, index) orelse unreachable] = .{
            .index = try codegenU32FromIntegerAttr(mlir.mlirArrayAttrGetElement(attr, index), field, diagnostics),
        };
    }
    return ids;
}

fn graphInstructionIdsFromArrayAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    field: []const u8,
    require_non_empty: bool,
    diagnostics: *std.Io.Writer,
) ![]compiler_facts.GraphInstructionId {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAArray(attr)) {
        try diagnostics.print("pass=codegen_extract feature={s} reason=field is not an array attribute\n", .{field});
        return MlirStateError.InvalidKernelCodegenPlan;
    }
    const count = mlir.mlirArrayAttrGetNumElements(attr);
    if ((require_non_empty and count <= 0) or count > limits.max_kernel_ref_ids) {
        try diagnostics.print("pass=codegen_extract feature={s} reason=invalid reference count\n", .{field});
        return MlirStateError.InvalidKernelCodegenPlan;
    }
    const ids = try allocator.alloc(compiler_facts.GraphInstructionId, std.math.cast(usize, count) orelse unreachable);
    errdefer allocator.free(ids);
    var index: isize = 0;
    while (index < count) : (index += 1) {
        ids[std.math.cast(usize, index) orelse unreachable] = .{
            .index = try codegenU32FromIntegerAttr(mlir.mlirArrayAttrGetElement(attr, index), field, diagnostics),
        };
    }
    return ids;
}

fn costLedgerIdsFromArrayAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    field: []const u8,
    require_non_empty: bool,
    diagnostics: *std.Io.Writer,
) ![]compiler_facts.CostLedgerId {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAArray(attr)) {
        try diagnostics.print("pass=codegen_extract feature={s} reason=field is not an array attribute\n", .{field});
        return MlirStateError.InvalidKernelCodegenPlan;
    }
    const count = mlir.mlirArrayAttrGetNumElements(attr);
    if ((require_non_empty and count <= 0) or count > limits.max_kernel_ref_ids) {
        try diagnostics.print("pass=codegen_extract feature={s} reason=invalid reference count\n", .{field});
        return MlirStateError.InvalidKernelCodegenPlan;
    }
    const ids = try allocator.alloc(compiler_facts.CostLedgerId, std.math.cast(usize, count) orelse unreachable);
    errdefer allocator.free(ids);
    var index: isize = 0;
    while (index < count) : (index += 1) {
        ids[std.math.cast(usize, index) orelse unreachable] = .{
            .index = try codegenU32FromIntegerAttr(mlir.mlirArrayAttrGetElement(attr, index), field, diagnostics),
        };
    }
    return ids;
}

fn loweringCostIdsFromArrayAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    field: []const u8,
    require_non_empty: bool,
    diagnostics: *std.Io.Writer,
) ![]compiler_facts.CostLedgerId {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAArray(attr)) {
        try diagnostics.print("pass=lowering_extract feature={s} reason=field is not an array attribute\n", .{field});
        return MlirStateError.InvalidLoweringPlan;
    }
    const count = mlir.mlirArrayAttrGetNumElements(attr);
    if ((require_non_empty and count <= 0) or count > limits.max_kernel_ref_ids) {
        try diagnostics.print("pass=lowering_extract feature={s} reason=invalid reference count\n", .{field});
        return MlirStateError.InvalidLoweringPlan;
    }
    const ids = try allocator.alloc(compiler_facts.CostLedgerId, std.math.cast(usize, count) orelse unreachable);
    errdefer allocator.free(ids);
    var index: isize = 0;
    while (index < count) : (index += 1) {
        ids[std.math.cast(usize, index) orelse unreachable] = .{
            .index = try loweringU32FromIntegerAttr(mlir.mlirArrayAttrGetElement(attr, index), field, diagnostics),
        };
    }
    return ids;
}

fn loweringU32sFromArrayAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    field: []const u8,
    require_non_empty: bool,
    diagnostics: *std.Io.Writer,
) ![]u32 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAArray(attr)) {
        try diagnostics.print("pass=lowering_region_extract feature={s} reason=field is not an array attribute\n", .{field});
        return MlirStateError.InvalidLoweringPlan;
    }
    const count = mlir.mlirArrayAttrGetNumElements(attr);
    if ((require_non_empty and count <= 0) or count > limits.max_kernel_ref_ids) {
        try diagnostics.print("pass=lowering_region_extract feature={s} reason=invalid reference count\n", .{field});
        return MlirStateError.InvalidLoweringPlan;
    }
    const values = try allocator.alloc(u32, std.math.cast(usize, count) orelse unreachable);
    errdefer allocator.free(values);
    var index: isize = 0;
    while (index < count) : (index += 1) {
        values[std.math.cast(usize, index) orelse unreachable] = try loweringU32FromIntegerAttr(mlir.mlirArrayAttrGetElement(attr, index), field, diagnostics);
    }
    return values;
}

fn loweringI64sFromArrayAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    field: []const u8,
    diagnostics: *std.Io.Writer,
) ![]i64 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAArray(attr)) {
        try diagnostics.print("pass=lowering_region_extract feature={s} reason=field is not an array attribute\n", .{field});
        return MlirStateError.InvalidLoweringPlan;
    }
    const count = mlir.mlirArrayAttrGetNumElements(attr);
    if (count <= 0 or count > limits.max_tile_rank) {
        try diagnostics.print("pass=lowering_region_extract feature={s} reason=invalid tile rank\n", .{field});
        return MlirStateError.InvalidLoweringPlan;
    }
    const values = try allocator.alloc(i64, std.math.cast(usize, count) orelse unreachable);
    errdefer allocator.free(values);
    var index: isize = 0;
    while (index < count) : (index += 1) {
        const value_attr = mlir.mlirArrayAttrGetElement(attr, index);
        if (mlir.mlirAttributeIsNull(value_attr) or !mlir.mlirAttributeIsAInteger(value_attr)) {
            try diagnostics.print("pass=lowering_region_extract feature={s} reason=expected integer tile dimension\n", .{field});
            return MlirStateError.InvalidLoweringPlan;
        }
        values[std.math.cast(usize, index) orelse unreachable] = mlir.mlirIntegerAttrGetValueInt(value_attr);
    }
    return values;
}

fn memoryTrafficIdsFromArrayAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    field: []const u8,
    require_non_empty: bool,
    diagnostics: *std.Io.Writer,
) ![]compiler_facts.MemoryTrafficId {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAArray(attr)) {
        try diagnostics.print("pass=codegen_extract feature={s} reason=field is not an array attribute\n", .{field});
        return MlirStateError.InvalidKernelCodegenPlan;
    }
    const count = mlir.mlirArrayAttrGetNumElements(attr);
    if ((require_non_empty and count <= 0) or count > limits.max_kernel_ref_ids) {
        try diagnostics.print("pass=codegen_extract feature={s} reason=invalid reference count\n", .{field});
        return MlirStateError.InvalidKernelCodegenPlan;
    }
    const ids = try allocator.alloc(compiler_facts.MemoryTrafficId, std.math.cast(usize, count) orelse unreachable);
    errdefer allocator.free(ids);
    var index: isize = 0;
    while (index < count) : (index += 1) {
        ids[std.math.cast(usize, index) orelse unreachable] = .{
            .index = try codegenU32FromIntegerAttr(mlir.mlirArrayAttrGetElement(attr, index), field, diagnostics),
        };
    }
    return ids;
}

fn scheduleGraphValueIdsFromArrayAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    field: []const u8,
    require_non_empty: bool,
    diagnostics: *std.Io.Writer,
) ![]compiler_facts.GraphValueId {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAArray(attr)) {
        try diagnostics.print("pass=schedule_extract feature={s} reason=field is not an array attribute\n", .{field});
        return MlirStateError.InvalidSchedulePlan;
    }
    const count = mlir.mlirArrayAttrGetNumElements(attr);
    if ((require_non_empty and count <= 0) or count > limits.max_schedule_refs) {
        try diagnostics.print("pass=schedule_extract feature={s} reason=invalid reference count\n", .{field});
        return MlirStateError.InvalidSchedulePlan;
    }
    const ids = try allocator.alloc(compiler_facts.GraphValueId, std.math.cast(usize, count) orelse unreachable);
    errdefer allocator.free(ids);
    var index: isize = 0;
    while (index < count) : (index += 1) {
        ids[std.math.cast(usize, index) orelse unreachable] = .{
            .index = try scheduleU32FromIntegerAttr(mlir.mlirArrayAttrGetElement(attr, index), field, diagnostics),
        };
    }
    return ids;
}

fn scheduleLoweringIdsFromArrayAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    field: []const u8,
    require_non_empty: bool,
    diagnostics: *std.Io.Writer,
) ![]compiler_facts.LoweringRecordId {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAArray(attr)) {
        try diagnostics.print("pass=schedule_extract feature={s} reason=field is not an array attribute\n", .{field});
        return MlirStateError.InvalidSchedulePlan;
    }
    const count = mlir.mlirArrayAttrGetNumElements(attr);
    if ((require_non_empty and count <= 0) or count > limits.max_schedule_refs) {
        try diagnostics.print("pass=schedule_extract feature={s} reason=invalid reference count\n", .{field});
        return MlirStateError.InvalidSchedulePlan;
    }
    const ids = try allocator.alloc(compiler_facts.LoweringRecordId, std.math.cast(usize, count) orelse unreachable);
    errdefer allocator.free(ids);
    var index: isize = 0;
    while (index < count) : (index += 1) {
        ids[std.math.cast(usize, index) orelse unreachable] = .{
            .index = try scheduleU32FromIntegerAttr(mlir.mlirArrayAttrGetElement(attr, index), field, diagnostics),
        };
    }
    return ids;
}

fn scheduleCostIdsFromArrayAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    field: []const u8,
    require_non_empty: bool,
    diagnostics: *std.Io.Writer,
) ![]compiler_facts.CostLedgerId {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAArray(attr)) {
        try diagnostics.print("pass=schedule_extract feature={s} reason=field is not an array attribute\n", .{field});
        return MlirStateError.InvalidSchedulePlan;
    }
    const count = mlir.mlirArrayAttrGetNumElements(attr);
    if ((require_non_empty and count <= 0) or count > limits.max_schedule_refs) {
        try diagnostics.print("pass=schedule_extract feature={s} reason=invalid reference count\n", .{field});
        return MlirStateError.InvalidSchedulePlan;
    }
    const ids = try allocator.alloc(compiler_facts.CostLedgerId, std.math.cast(usize, count) orelse unreachable);
    errdefer allocator.free(ids);
    var index: isize = 0;
    while (index < count) : (index += 1) {
        ids[std.math.cast(usize, index) orelse unreachable] = .{
            .index = try scheduleU32FromIntegerAttr(mlir.mlirArrayAttrGetElement(attr, index), field, diagnostics),
        };
    }
    return ids;
}

fn backendBindingGraphInstructionIdsFromArrayAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    field: []const u8,
    require_non_empty: bool,
    diagnostics: *std.Io.Writer,
) ![]compiler_facts.GraphInstructionId {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAArray(attr)) {
        try diagnostics.print("pass=backend_binding_extract feature={s} reason=field is not an array attribute\n", .{field});
        return MlirStateError.InvalidBackendBindingPlan;
    }
    const count = mlir.mlirArrayAttrGetNumElements(attr);
    if ((require_non_empty and count <= 0) or count > limits.max_kernel_ref_ids) {
        try diagnostics.print("pass=backend_binding_extract feature={s} reason=invalid reference count\n", .{field});
        return MlirStateError.InvalidBackendBindingPlan;
    }
    const ids = try allocator.alloc(compiler_facts.GraphInstructionId, std.math.cast(usize, count) orelse unreachable);
    errdefer allocator.free(ids);
    var index: isize = 0;
    while (index < count) : (index += 1) {
        ids[std.math.cast(usize, index) orelse unreachable] = .{
            .index = try backendBindingU32FromIntegerAttr(mlir.mlirArrayAttrGetElement(attr, index), field, diagnostics),
        };
    }
    return ids;
}

fn profileJoinGraphInstructionIdsFromArrayAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    field: []const u8,
    require_non_empty: bool,
    diagnostics: *std.Io.Writer,
) ![]compiler_facts.GraphInstructionId {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAArray(attr)) {
        try diagnostics.print("pass=profile_join_extract feature={s} reason=field is not an array attribute\n", .{field});
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    const count = mlir.mlirArrayAttrGetNumElements(attr);
    if ((require_non_empty and count <= 0) or count > limits.max_kernel_ref_ids) {
        try diagnostics.print("pass=profile_join_extract feature={s} reason=invalid reference count\n", .{field});
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    const ids = try allocator.alloc(compiler_facts.GraphInstructionId, std.math.cast(usize, count) orelse unreachable);
    errdefer allocator.free(ids);
    var index: isize = 0;
    while (index < count) : (index += 1) {
        ids[std.math.cast(usize, index) orelse unreachable] = .{
            .index = try profileJoinU32FromIntegerAttr(mlir.mlirArrayAttrGetElement(attr, index), field, diagnostics),
        };
    }
    return ids;
}

fn backendExecutableCallsFromArrayAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    diagnostics: *std.Io.Writer,
) ![]BackendExecutableCallFact {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAArray(attr)) {
        try diagnostics.writeAll("pass=backend_executable_extract feature=calls reason=field is not an array attribute\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    const count = mlir.mlirArrayAttrGetNumElements(attr);
    if (count <= 0 or count > limits.max_backend_executable_calls) {
        try diagnostics.writeAll("pass=backend_executable_extract feature=calls reason=invalid call count\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    var calls: std.ArrayList(BackendExecutableCallFact) = .empty;
    errdefer {
        for (calls.items) |call| deinitBackendExecutableCallFields(allocator, call);
        calls.deinit(allocator);
    }
    var index: isize = 0;
    while (index < count) : (index += 1) {
        try calls.append(allocator, try parseBackendExecutableCallAttr(allocator, mlir.mlirArrayAttrGetElement(attr, index), diagnostics));
    }
    return calls.toOwnedSlice(allocator);
}

fn backendKernelGraphNodesFromArrayAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    diagnostics: *std.Io.Writer,
) ![]BackendKernelGraphNodeFact {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAArray(attr)) {
        try diagnostics.writeAll("pass=backend_kernel_graph_extract feature=nodes reason=field is not an array attribute\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    const count = mlir.mlirArrayAttrGetNumElements(attr);
    if (count <= 0 or count > limits.max_backend_executable_calls) {
        try diagnostics.writeAll("pass=backend_kernel_graph_extract feature=nodes reason=invalid node count\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    var nodes: std.ArrayList(BackendKernelGraphNodeFact) = .empty;
    errdefer {
        for (nodes.items) |node| deinitBackendKernelGraphNodeFields(allocator, node);
        nodes.deinit(allocator);
    }
    var index: isize = 0;
    while (index < count) : (index += 1) {
        try nodes.append(allocator, try parseBackendKernelGraphNodeAttr(allocator, mlir.mlirArrayAttrGetElement(attr, index), diagnostics));
    }
    return nodes.toOwnedSlice(allocator);
}

fn backendKernelGraphEdgesFromArrayAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    diagnostics: *std.Io.Writer,
) ![]BackendKernelGraphEdgeFact {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAArray(attr)) {
        try diagnostics.writeAll("pass=backend_kernel_graph_extract feature=edges reason=field is not an array attribute\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    const count = mlir.mlirArrayAttrGetNumElements(attr);
    if (count < 0 or count > limits.max_backend_kernel_graph_edges) {
        try diagnostics.writeAll("pass=backend_kernel_graph_extract feature=edges reason=invalid edge count\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    const edges = try allocator.alloc(BackendKernelGraphEdgeFact, std.math.cast(usize, count) orelse unreachable);
    errdefer allocator.free(edges);
    var index: isize = 0;
    while (index < count) : (index += 1) {
        edges[std.math.cast(usize, index) orelse unreachable] = try parseBackendKernelGraphEdgeAttr(mlir.mlirArrayAttrGetElement(attr, index), diagnostics);
    }
    return edges;
}

fn backendPlanGraphInstructionIdsFromArrayAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    field: []const u8,
    require_non_empty: bool,
    diagnostics: *std.Io.Writer,
) ![]compiler_facts.GraphInstructionId {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAArray(attr)) {
        try diagnostics.print("pass=backend_plan_extract feature={s} reason=field is not an array attribute\n", .{field});
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    const count = mlir.mlirArrayAttrGetNumElements(attr);
    if ((require_non_empty and count <= 0) or count > limits.max_kernel_ref_ids) {
        try diagnostics.print("pass=backend_plan_extract feature={s} reason=invalid reference count\n", .{field});
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    const ids = try allocator.alloc(compiler_facts.GraphInstructionId, std.math.cast(usize, count) orelse unreachable);
    errdefer allocator.free(ids);
    var index: isize = 0;
    while (index < count) : (index += 1) {
        ids[std.math.cast(usize, index) orelse unreachable] = .{
            .index = try backendPlanU32FromIntegerAttr(mlir.mlirArrayAttrGetElement(attr, index), field, diagnostics),
        };
    }
    return ids;
}

fn backendPlanGraphValueIdsFromArrayAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    field: []const u8,
    require_non_empty: bool,
    diagnostics: *std.Io.Writer,
) ![]compiler_facts.GraphValueId {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAArray(attr)) {
        try diagnostics.print("pass=backend_plan_extract feature={s} reason=field is not an array attribute\n", .{field});
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    const count = mlir.mlirArrayAttrGetNumElements(attr);
    if ((require_non_empty and count <= 0) or count > limits.max_kernel_ref_ids) {
        try diagnostics.print("pass=backend_plan_extract feature={s} reason=invalid reference count\n", .{field});
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    const ids = try allocator.alloc(compiler_facts.GraphValueId, std.math.cast(usize, count) orelse unreachable);
    errdefer allocator.free(ids);
    var index: isize = 0;
    while (index < count) : (index += 1) {
        ids[std.math.cast(usize, index) orelse unreachable] = .{
            .index = try backendPlanU32FromIntegerAttr(mlir.mlirArrayAttrGetElement(attr, index), field, diagnostics),
        };
    }
    return ids;
}

fn backendPlanI64sFromArrayAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    field: []const u8,
    diagnostics: *std.Io.Writer,
) ![]i64 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAArray(attr)) {
        try diagnostics.print("pass=backend_kernel_graph_extract feature={s} reason=field is not an array attribute\n", .{field});
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    const count = mlir.mlirArrayAttrGetNumElements(attr);
    if (count <= 0 or count > limits.max_tile_rank) {
        try diagnostics.print("pass=backend_kernel_graph_extract feature={s} reason=invalid rank\n", .{field});
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    const values = try allocator.alloc(i64, std.math.cast(usize, count) orelse unreachable);
    errdefer allocator.free(values);
    var index: isize = 0;
    while (index < count) : (index += 1) {
        values[std.math.cast(usize, index) orelse unreachable] = mlir.mlirIntegerAttrGetValueInt(mlir.mlirArrayAttrGetElement(attr, index));
    }
    return values;
}

fn runtimeAllocationsFromArrayAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    diagnostics: *std.Io.Writer,
) ![]RuntimeAllocationFact {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAArray(attr)) {
        try diagnostics.writeAll("pass=runtime_allocation_extract feature=allocations reason=field is not an array attribute\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    const count = mlir.mlirArrayAttrGetNumElements(attr);
    if (count <= 0 or count > limits.max_runtime_allocations) {
        try diagnostics.writeAll("pass=runtime_allocation_extract feature=allocations reason=invalid allocation count\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    var allocations: std.ArrayList(RuntimeAllocationFact) = .empty;
    errdefer {
        for (allocations.items) |allocation| deinitRuntimeAllocationFields(allocator, allocation);
        allocations.deinit(allocator);
    }
    var index: isize = 0;
    while (index < count) : (index += 1) {
        try allocations.append(allocator, try parseRuntimeAllocationAttr(allocator, mlir.mlirArrayAttrGetElement(attr, index), diagnostics));
    }
    return allocations.toOwnedSlice(allocator);
}

fn runtimeBufferUsesFromArrayAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    diagnostics: *std.Io.Writer,
) ![]RuntimeBufferUseFact {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAArray(attr)) {
        try diagnostics.writeAll("pass=runtime_allocation_extract feature=buffer-uses reason=field is not an array attribute\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    const count = mlir.mlirArrayAttrGetNumElements(attr);
    if (count <= 0 or count > limits.max_runtime_buffer_uses) {
        try diagnostics.writeAll("pass=runtime_allocation_extract feature=buffer-uses reason=invalid buffer use count\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    var uses: std.ArrayList(RuntimeBufferUseFact) = .empty;
    errdefer {
        for (uses.items) |use| deinitRuntimeBufferUseFields(allocator, use);
        uses.deinit(allocator);
    }
    var index: isize = 0;
    while (index < count) : (index += 1) {
        try uses.append(allocator, try parseRuntimeBufferUseAttr(allocator, mlir.mlirArrayAttrGetElement(attr, index), diagnostics));
    }
    return uses.toOwnedSlice(allocator);
}

fn targetDevicesFromArrayAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    diagnostics: *std.Io.Writer,
) ![]target_pkg.TargetDevice {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAArray(attr)) {
        try diagnostics.writeAll("pass=target_extract feature=devices reason=devices field is not an array attribute\n");
        return MlirStateError.InvalidTargetAttachment;
    }
    const count = mlir.mlirArrayAttrGetNumElements(attr);
    if (count <= 0 or count > limits.max_target_devices) {
        try diagnostics.writeAll("pass=target_extract feature=devices reason=invalid device count\n");
        return MlirStateError.InvalidTargetAttachment;
    }
    var devices: std.ArrayList(target_pkg.TargetDevice) = .empty;
    errdefer {
        for (devices.items) |device| {
            allocator.free(device.name);
            allocator.free(device.memory_space_ids);
            allocator.free(device.execution_unit_ids);
        }
        devices.deinit(allocator);
    }
    var index: isize = 0;
    while (index < count) : (index += 1) {
        try devices.append(allocator, try parseTargetDeviceAttr(allocator, mlir.mlirArrayAttrGetElement(attr, index), diagnostics));
    }
    return devices.toOwnedSlice(allocator);
}

fn targetMemorySpacesFromArrayAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    diagnostics: *std.Io.Writer,
) ![]target_pkg.TargetMemorySpace {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAArray(attr)) {
        try diagnostics.writeAll("pass=target_extract feature=memory-spaces reason=memory spaces field is not an array attribute\n");
        return MlirStateError.InvalidTargetAttachment;
    }
    const count = mlir.mlirArrayAttrGetNumElements(attr);
    if (count <= 0 or count > limits.max_target_memory_spaces) {
        try diagnostics.writeAll("pass=target_extract feature=memory-spaces reason=invalid memory space count\n");
        return MlirStateError.InvalidTargetAttachment;
    }
    var memory_spaces: std.ArrayList(target_pkg.TargetMemorySpace) = .empty;
    errdefer {
        for (memory_spaces.items) |memory_space| {
            allocator.free(memory_space.name);
            allocator.free(memory_space.note);
        }
        memory_spaces.deinit(allocator);
    }
    var index: isize = 0;
    while (index < count) : (index += 1) {
        try memory_spaces.append(allocator, try parseTargetMemorySpaceAttr(allocator, mlir.mlirArrayAttrGetElement(attr, index), diagnostics));
    }
    return memory_spaces.toOwnedSlice(allocator);
}

fn targetTransferEdgesFromArrayAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    diagnostics: *std.Io.Writer,
) ![]target_pkg.TargetTransferEdge {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAArray(attr)) {
        try diagnostics.writeAll("pass=target_extract feature=transfer-edges reason=transfer edges field is not an array attribute\n");
        return MlirStateError.InvalidTargetAttachment;
    }
    const count = mlir.mlirArrayAttrGetNumElements(attr);
    if (count > limits.max_target_transfer_edges) {
        try diagnostics.writeAll("pass=target_extract feature=transfer-edges reason=invalid transfer edge count\n");
        return MlirStateError.InvalidTargetAttachment;
    }
    var edges: std.ArrayList(target_pkg.TargetTransferEdge) = .empty;
    errdefer {
        for (edges.items) |edge| allocator.free(edge.note);
        edges.deinit(allocator);
    }
    var index: isize = 0;
    while (index < count) : (index += 1) {
        try edges.append(allocator, try parseTargetTransferEdgeAttr(allocator, mlir.mlirArrayAttrGetElement(attr, index), diagnostics));
    }
    return edges.toOwnedSlice(allocator);
}

fn targetExecutionUnitsFromArrayAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    diagnostics: *std.Io.Writer,
) ![]target_pkg.ExecutionUnit {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAArray(attr)) {
        try diagnostics.writeAll("pass=target_extract feature=execution-units reason=execution units field is not an array attribute\n");
        return MlirStateError.InvalidTargetAttachment;
    }
    const count = mlir.mlirArrayAttrGetNumElements(attr);
    if (count <= 0 or count > limits.max_target_execution_units) {
        try diagnostics.writeAll("pass=target_extract feature=execution-units reason=invalid execution unit count\n");
        return MlirStateError.InvalidTargetAttachment;
    }
    var units: std.ArrayList(target_pkg.ExecutionUnit) = .empty;
    errdefer {
        for (units.items) |unit| {
            allocator.free(unit.name);
            for (unit.dtype_rates) |rate| allocator.free(rate.note);
            allocator.free(unit.dtype_rates);
        }
        units.deinit(allocator);
    }
    var index: isize = 0;
    while (index < count) : (index += 1) {
        try units.append(allocator, try parseTargetExecutionUnitAttr(allocator, mlir.mlirArrayAttrGetElement(attr, index), diagnostics));
    }
    return units.toOwnedSlice(allocator);
}

fn targetDTypeRatesFromArrayAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    diagnostics: *std.Io.Writer,
) ![]target_pkg.DTypeRate {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAArray(attr)) {
        try diagnostics.writeAll("pass=target_extract feature=dtype-rates reason=dtype rates field is not an array attribute\n");
        return MlirStateError.InvalidTargetAttachment;
    }
    const count = mlir.mlirArrayAttrGetNumElements(attr);
    if (count > limits.max_target_dtype_rates) {
        try diagnostics.writeAll("pass=target_extract feature=dtype-rates reason=invalid dtype rate count\n");
        return MlirStateError.InvalidTargetAttachment;
    }
    var rates: std.ArrayList(target_pkg.DTypeRate) = .empty;
    errdefer {
        for (rates.items) |rate| allocator.free(rate.note);
        rates.deinit(allocator);
    }
    var index: isize = 0;
    while (index < count) : (index += 1) {
        try rates.append(allocator, try parseTargetDTypeRateAttr(allocator, mlir.mlirArrayAttrGetElement(attr, index), diagnostics));
    }
    return rates.toOwnedSlice(allocator);
}

fn targetU32sFromArrayAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    field: []const u8,
    require_non_empty: bool,
    diagnostics: *std.Io.Writer,
) ![]u32 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAArray(attr)) {
        try diagnostics.print("pass=target_extract feature={s} reason=field is not an array attribute\n", .{field});
        return MlirStateError.InvalidTargetAttachment;
    }
    const count = mlir.mlirArrayAttrGetNumElements(attr);
    if ((require_non_empty and count <= 0) or count > limits.max_target_refs) {
        try diagnostics.print("pass=target_extract feature={s} reason=invalid reference count\n", .{field});
        return MlirStateError.InvalidTargetAttachment;
    }
    const values = try allocator.alloc(u32, std.math.cast(usize, count) orelse unreachable);
    errdefer allocator.free(values);
    var index: isize = 0;
    while (index < count) : (index += 1) {
        values[std.math.cast(usize, index) orelse unreachable] = try targetU32FromIntegerAttr(mlir.mlirArrayAttrGetElement(attr, index), field, diagnostics);
    }
    return values;
}

fn performanceGraphInstructionIdsFromArrayAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    field: []const u8,
    require_non_empty: bool,
    diagnostics: *std.Io.Writer,
) ![]compiler_facts.GraphInstructionId {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAArray(attr)) {
        try diagnostics.print("pass=performance_extract feature={s} reason=field is not an array attribute\n", .{field});
        return MlirStateError.InvalidKernelCodegenPlan;
    }
    const count = mlir.mlirArrayAttrGetNumElements(attr);
    if ((require_non_empty and count <= 0) or count > limits.max_kernel_ref_ids) {
        try diagnostics.print("pass=performance_extract feature={s} reason=invalid reference count\n", .{field});
        return MlirStateError.InvalidKernelCodegenPlan;
    }
    const ids = try allocator.alloc(compiler_facts.GraphInstructionId, std.math.cast(usize, count) orelse unreachable);
    errdefer allocator.free(ids);
    var index: isize = 0;
    while (index < count) : (index += 1) {
        ids[std.math.cast(usize, index) orelse unreachable] = .{
            .index = try performanceU32FromIntegerAttr(mlir.mlirArrayAttrGetElement(attr, index), field, diagnostics),
        };
    }
    return ids;
}

fn performanceCostIdsFromArrayAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    field: []const u8,
    require_non_empty: bool,
    diagnostics: *std.Io.Writer,
) ![]compiler_facts.CostLedgerId {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAArray(attr)) {
        try diagnostics.print("pass=performance_extract feature={s} reason=field is not an array attribute\n", .{field});
        return MlirStateError.InvalidKernelCodegenPlan;
    }
    const count = mlir.mlirArrayAttrGetNumElements(attr);
    if ((require_non_empty and count <= 0) or count > limits.max_kernel_ref_ids) {
        try diagnostics.print("pass=performance_extract feature={s} reason=invalid reference count\n", .{field});
        return MlirStateError.InvalidKernelCodegenPlan;
    }
    const ids = try allocator.alloc(compiler_facts.CostLedgerId, std.math.cast(usize, count) orelse unreachable);
    errdefer allocator.free(ids);
    var index: isize = 0;
    while (index < count) : (index += 1) {
        ids[std.math.cast(usize, index) orelse unreachable] = .{
            .index = try performanceU32FromIntegerAttr(mlir.mlirArrayAttrGetElement(attr, index), field, diagnostics),
        };
    }
    return ids;
}

fn u32sFromArrayAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    field: []const u8,
    max_count: usize,
    diagnostics: *std.Io.Writer,
) ![]u32 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAArray(attr)) {
        try diagnostics.print("pass=runtime_stream_extract feature={s} reason=field is not an array attribute\n", .{field});
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    const count = mlir.mlirArrayAttrGetNumElements(attr);
    if (count > max_count) {
        try diagnostics.print("pass=runtime_stream_extract feature={s} reason=invalid item count\n", .{field});
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    const ids = try allocator.alloc(u32, std.math.cast(usize, count) orelse unreachable);
    errdefer allocator.free(ids);
    var index: isize = 0;
    while (index < count) : (index += 1) {
        ids[std.math.cast(usize, index) orelse unreachable] = try runtimeStreamU32FromIntegerAttr(mlir.mlirArrayAttrGetElement(attr, index), field, diagnostics);
    }
    return ids;
}

fn profileEventIdsFromArrayAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    field: []const u8,
    diagnostics: *std.Io.Writer,
) ![]core.ProfileEventId {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAArray(attr)) {
        try diagnostics.print("pass=profile_join_extract feature={s} reason=field is not an array attribute\n", .{field});
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    const count = mlir.mlirArrayAttrGetNumElements(attr);
    if (count <= 0 or count > limits.max_runtime_profile_join_events) {
        try diagnostics.print("pass=profile_join_extract feature={s} reason=invalid profile event count\n", .{field});
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    const ids = try allocator.alloc(core.ProfileEventId, std.math.cast(usize, count) orelse unreachable);
    errdefer allocator.free(ids);
    var index: isize = 0;
    while (index < count) : (index += 1) {
        ids[std.math.cast(usize, index) orelse unreachable] = .{
            .index = try profileJoinU32FromIntegerAttr(mlir.mlirArrayAttrGetElement(attr, index), field, diagnostics),
        };
    }
    return ids;
}

fn backendBindingCostIdsFromArrayAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    field: []const u8,
    require_non_empty: bool,
    diagnostics: *std.Io.Writer,
) ![]compiler_facts.CostLedgerId {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAArray(attr)) {
        try diagnostics.print("pass=backend_binding_extract feature={s} reason=field is not an array attribute\n", .{field});
        return MlirStateError.InvalidBackendBindingPlan;
    }
    const count = mlir.mlirArrayAttrGetNumElements(attr);
    if ((require_non_empty and count <= 0) or count > limits.max_kernel_ref_ids) {
        try diagnostics.print("pass=backend_binding_extract feature={s} reason=invalid reference count\n", .{field});
        return MlirStateError.InvalidBackendBindingPlan;
    }
    const ids = try allocator.alloc(compiler_facts.CostLedgerId, std.math.cast(usize, count) orelse unreachable);
    errdefer allocator.free(ids);
    var index: isize = 0;
    while (index < count) : (index += 1) {
        ids[std.math.cast(usize, index) orelse unreachable] = .{
            .index = try backendBindingU32FromIntegerAttr(mlir.mlirArrayAttrGetElement(attr, index), field, diagnostics),
        };
    }
    return ids;
}

fn commandDependenciesFromArrayAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    diagnostics: *std.Io.Writer,
) ![]core.CommandDependency {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAArray(attr)) {
        try diagnostics.writeAll("pass=schedule_extract feature=dependencies reason=dependencies field is not an array attribute\n");
        return MlirStateError.InvalidSchedulePlan;
    }
    const count = mlir.mlirArrayAttrGetNumElements(attr);
    if (count > limits.max_schedule_dependencies) {
        try diagnostics.writeAll("pass=schedule_extract feature=dependencies reason=too many dependencies\n");
        return MlirStateError.InvalidSchedulePlan;
    }
    const dependencies = try allocator.alloc(core.CommandDependency, std.math.cast(usize, count) orelse unreachable);
    errdefer allocator.free(dependencies);
    var index: isize = 0;
    while (index < count) : (index += 1) {
        const dependency = mlir.mlirArrayAttrGetElement(attr, index);
        if (mlir.mlirAttributeIsNull(dependency) or !mlir.mlirAttributeIsADictionary(dependency)) {
            try diagnostics.writeAll("pass=schedule_extract feature=dependencies reason=dependency is not a dictionary attribute\n");
            return MlirStateError.InvalidSchedulePlan;
        }
        dependencies[std.math.cast(usize, index) orelse unreachable] = .{
            .command_id = .{ .index = try scheduleU32FromIntegerAttr(dictAttr(dependency, "command"), "dependency.command", diagnostics) },
            .kind = try dependencyKindFromStringAttr(dictAttr(dependency, "kind"), diagnostics),
        };
    }
    return dependencies;
}

fn i64sFromArrayAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    diagnostics: *std.Io.Writer,
) ![]i64 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAArray(attr)) {
        try diagnostics.writeAll("pass=placement_extract feature=tile reason=tile field is not an array attribute\n");
        return MlirStateError.InvalidPlacementPlan;
    }
    const count = mlir.mlirArrayAttrGetNumElements(attr);
    if (count <= 0 or count > limits.max_tile_rank) {
        try diagnostics.writeAll("pass=placement_extract feature=tile reason=invalid tile rank\n");
        return MlirStateError.InvalidPlacementPlan;
    }
    const values = try allocator.alloc(i64, std.math.cast(usize, count) orelse unreachable);
    errdefer allocator.free(values);
    var index: isize = 0;
    while (index < count) : (index += 1) {
        const value_index = std.math.cast(usize, index) orelse unreachable;
        values[value_index] = try placementI64FromIntegerAttr(mlir.mlirArrayAttrGetElement(attr, index), "tile", diagnostics);
    }
    return values;
}

fn u32FromIntegerAttr(attr: mlir.MlirAttribute, field: []const u8, diagnostics: *std.Io.Writer) !u32 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAInteger(attr)) {
        try diagnostics.print("pass=fusion_extract feature={s} reason=expected integer attribute\n", .{field});
        return MlirStateError.InvalidFusionPlan;
    }
    return std.math.cast(u32, mlir.mlirIntegerAttrGetValueInt(attr)) orelse {
        try diagnostics.print("pass=fusion_extract feature={s} reason=integer exceeds u32\n", .{field});
        return MlirStateError.InvalidFusionPlan;
    };
}

fn placementU32FromIntegerAttr(attr: mlir.MlirAttribute, field: []const u8, diagnostics: *std.Io.Writer) !u32 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAInteger(attr)) {
        try diagnostics.print("pass=placement_extract feature={s} reason=expected integer attribute\n", .{field});
        return MlirStateError.InvalidPlacementPlan;
    }
    return std.math.cast(u32, mlir.mlirIntegerAttrGetValueInt(attr)) orelse {
        try diagnostics.print("pass=placement_extract feature={s} reason=integer exceeds u32\n", .{field});
        return MlirStateError.InvalidPlacementPlan;
    };
}

fn placementI64FromIntegerAttr(attr: mlir.MlirAttribute, field: []const u8, diagnostics: *std.Io.Writer) !i64 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAInteger(attr)) {
        try diagnostics.print("pass=placement_extract feature={s} reason=expected integer attribute\n", .{field});
        return MlirStateError.InvalidPlacementPlan;
    }
    const value = mlir.mlirIntegerAttrGetValueInt(attr);
    if (value <= 0) {
        try diagnostics.print("pass=placement_extract feature={s} reason=expected positive tile dimension\n", .{field});
        return MlirStateError.InvalidPlacementPlan;
    }
    return value;
}

fn optionalU32FromIntegerAttr(attr: mlir.MlirAttribute, field: []const u8, diagnostics: *std.Io.Writer) !?u32 {
    if (mlir.mlirAttributeIsNull(attr) or stringAttrEquals(attr, "none")) return null;
    return try placementU32FromIntegerAttr(attr, field, diagnostics);
}

fn collectiveU32FromIntegerAttr(attr: mlir.MlirAttribute, field: []const u8, diagnostics: *std.Io.Writer) !u32 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAInteger(attr)) {
        try diagnostics.print("pass=collective_extract feature={s} reason=expected integer attribute\n", .{field});
        return MlirStateError.InvalidCollectivePlan;
    }
    return std.math.cast(u32, mlir.mlirIntegerAttrGetValueInt(attr)) orelse {
        try diagnostics.print("pass=collective_extract feature={s} reason=integer exceeds u32\n", .{field});
        return MlirStateError.InvalidCollectivePlan;
    };
}

fn loweringU32FromIntegerAttr(attr: mlir.MlirAttribute, field: []const u8, diagnostics: *std.Io.Writer) !u32 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAInteger(attr)) {
        try diagnostics.print("pass=lowering_extract feature={s} reason=expected integer attribute\n", .{field});
        return MlirStateError.InvalidLoweringPlan;
    }
    return std.math.cast(u32, mlir.mlirIntegerAttrGetValueInt(attr)) orelse {
        try diagnostics.print("pass=lowering_extract feature={s} reason=integer exceeds u32\n", .{field});
        return MlirStateError.InvalidLoweringPlan;
    };
}

fn loweringOptionalU32FromIntegerAttr(attr: mlir.MlirAttribute, field: []const u8, diagnostics: *std.Io.Writer) !?u32 {
    if (mlir.mlirAttributeIsNull(attr) or stringAttrEquals(attr, "none")) return null;
    return try loweringU32FromIntegerAttr(attr, field, diagnostics);
}

fn codegenU32FromIntegerAttr(attr: mlir.MlirAttribute, field: []const u8, diagnostics: *std.Io.Writer) !u32 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAInteger(attr)) {
        try diagnostics.print("pass=codegen_extract feature={s} reason=expected integer attribute\n", .{field});
        return MlirStateError.InvalidKernelCodegenPlan;
    }
    return std.math.cast(u32, mlir.mlirIntegerAttrGetValueInt(attr)) orelse {
        try diagnostics.print("pass=codegen_extract feature={s} reason=integer exceeds u32\n", .{field});
        return MlirStateError.InvalidKernelCodegenPlan;
    };
}

fn scheduleU32FromIntegerAttr(attr: mlir.MlirAttribute, field: []const u8, diagnostics: *std.Io.Writer) !u32 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAInteger(attr)) {
        try diagnostics.print("pass=schedule_extract feature={s} reason=expected integer attribute\n", .{field});
        return MlirStateError.InvalidSchedulePlan;
    }
    return std.math.cast(u32, mlir.mlirIntegerAttrGetValueInt(attr)) orelse {
        try diagnostics.print("pass=schedule_extract feature={s} reason=integer exceeds u32\n", .{field});
        return MlirStateError.InvalidSchedulePlan;
    };
}

fn targetU32FromIntegerAttr(attr: mlir.MlirAttribute, field: []const u8, diagnostics: *std.Io.Writer) !u32 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAInteger(attr)) {
        try diagnostics.print("pass=target_extract feature={s} reason=expected integer attribute\n", .{field});
        return MlirStateError.InvalidTargetAttachment;
    }
    return std.math.cast(u32, mlir.mlirIntegerAttrGetValueInt(attr)) orelse {
        try diagnostics.print("pass=target_extract feature={s} reason=integer exceeds u32\n", .{field});
        return MlirStateError.InvalidTargetAttachment;
    };
}

fn targetI32FromIntegerAttr(attr: mlir.MlirAttribute, field: []const u8, diagnostics: *std.Io.Writer) !i32 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAInteger(attr)) {
        try diagnostics.print("pass=target_extract feature={s} reason=expected integer attribute\n", .{field});
        return MlirStateError.InvalidTargetAttachment;
    }
    return std.math.cast(i32, mlir.mlirIntegerAttrGetValueInt(attr)) orelse {
        try diagnostics.print("pass=target_extract feature={s} reason=integer exceeds i32\n", .{field});
        return MlirStateError.InvalidTargetAttachment;
    };
}

fn performanceU32FromIntegerAttr(attr: mlir.MlirAttribute, field: []const u8, diagnostics: *std.Io.Writer) !u32 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAInteger(attr)) {
        try diagnostics.print("pass=performance_extract feature={s} reason=expected integer attribute\n", .{field});
        return MlirStateError.InvalidKernelCodegenPlan;
    }
    return std.math.cast(u32, mlir.mlirIntegerAttrGetValueInt(attr)) orelse {
        try diagnostics.print("pass=performance_extract feature={s} reason=integer exceeds u32\n", .{field});
        return MlirStateError.InvalidKernelCodegenPlan;
    };
}

fn backendBindingU32FromIntegerAttr(attr: mlir.MlirAttribute, field: []const u8, diagnostics: *std.Io.Writer) !u32 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAInteger(attr)) {
        try diagnostics.print("pass=backend_binding_extract feature={s} reason=expected integer attribute\n", .{field});
        return MlirStateError.InvalidBackendBindingPlan;
    }
    return std.math.cast(u32, mlir.mlirIntegerAttrGetValueInt(attr)) orelse {
        try diagnostics.print("pass=backend_binding_extract feature={s} reason=integer exceeds u32\n", .{field});
        return MlirStateError.InvalidBackendBindingPlan;
    };
}

fn backendPlanU32FromIntegerAttr(attr: mlir.MlirAttribute, field: []const u8, diagnostics: *std.Io.Writer) !u32 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAInteger(attr)) {
        try diagnostics.print("pass=backend_plan_extract feature={s} reason=expected integer attribute\n", .{field});
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    return std.math.cast(u32, mlir.mlirIntegerAttrGetValueInt(attr)) orelse {
        try diagnostics.print("pass=backend_plan_extract feature={s} reason=integer exceeds u32\n", .{field});
        return MlirStateError.InvalidBackendExecutablePlan;
    };
}

fn runtimeAllocationU32FromIntegerAttr(attr: mlir.MlirAttribute, field: []const u8, diagnostics: *std.Io.Writer) !u32 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAInteger(attr)) {
        try diagnostics.print("pass=runtime_allocation_extract feature={s} reason=expected integer attribute\n", .{field});
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    return std.math.cast(u32, mlir.mlirIntegerAttrGetValueInt(attr)) orelse {
        try diagnostics.print("pass=runtime_allocation_extract feature={s} reason=integer exceeds u32\n", .{field});
        return MlirStateError.InvalidBackendExecutablePlan;
    };
}

fn runtimeStreamU32FromIntegerAttr(attr: mlir.MlirAttribute, field: []const u8, diagnostics: *std.Io.Writer) !u32 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAInteger(attr)) {
        try diagnostics.print("pass=runtime_stream_extract feature={s} reason=expected integer attribute\n", .{field});
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    return std.math.cast(u32, mlir.mlirIntegerAttrGetValueInt(attr)) orelse {
        try diagnostics.print("pass=runtime_stream_extract feature={s} reason=integer exceeds u32\n", .{field});
        return MlirStateError.InvalidBackendExecutablePlan;
    };
}

fn runtimeProfileU32FromIntegerAttr(attr: mlir.MlirAttribute, field: []const u8, diagnostics: *std.Io.Writer) !u32 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAInteger(attr)) {
        try diagnostics.print("pass=runtime_profile_extract feature={s} reason=expected integer attribute\n", .{field});
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    return std.math.cast(u32, mlir.mlirIntegerAttrGetValueInt(attr)) orelse {
        try diagnostics.print("pass=runtime_profile_extract feature={s} reason=integer exceeds u32\n", .{field});
        return MlirStateError.InvalidBackendExecutablePlan;
    };
}

fn profileJoinU32FromIntegerAttr(attr: mlir.MlirAttribute, field: []const u8, diagnostics: *std.Io.Writer) !u32 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAInteger(attr)) {
        try diagnostics.print("pass=profile_join_extract feature={s} reason=expected integer attribute\n", .{field});
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    return std.math.cast(u32, mlir.mlirIntegerAttrGetValueInt(attr)) orelse {
        try diagnostics.print("pass=profile_join_extract feature={s} reason=integer exceeds u32\n", .{field});
        return MlirStateError.InvalidBackendExecutablePlan;
    };
}

fn backendBindingOptionalU32FromIntegerAttr(attr: mlir.MlirAttribute, field: []const u8, diagnostics: *std.Io.Writer) !?u32 {
    if (mlir.mlirAttributeIsNull(attr) or stringAttrEquals(attr, "none")) return null;
    return try backendBindingU32FromIntegerAttr(attr, field, diagnostics);
}

fn backendPlanOptionalU32FromIntegerAttr(attr: mlir.MlirAttribute, field: []const u8, diagnostics: *std.Io.Writer) !?u32 {
    if (mlir.mlirAttributeIsNull(attr) or stringAttrEquals(attr, "none")) return null;
    return try backendPlanU32FromIntegerAttr(attr, field, diagnostics);
}

fn targetOptionalU32FromIntegerAttr(attr: mlir.MlirAttribute, field: []const u8, diagnostics: *std.Io.Writer) !?u32 {
    if (mlir.mlirAttributeIsNull(attr) or stringAttrEquals(attr, "none")) return null;
    return try targetU32FromIntegerAttr(attr, field, diagnostics);
}

fn performanceOptionalU32FromIntegerAttr(attr: mlir.MlirAttribute, field: []const u8, diagnostics: *std.Io.Writer) !?u32 {
    if (mlir.mlirAttributeIsNull(attr) or stringAttrEquals(attr, "none")) return null;
    return try performanceU32FromIntegerAttr(attr, field, diagnostics);
}

fn runtimeProfileOptionalU32FromIntegerAttr(attr: mlir.MlirAttribute, field: []const u8, diagnostics: *std.Io.Writer) !?u32 {
    if (mlir.mlirAttributeIsNull(attr) or stringAttrEquals(attr, "none")) return null;
    return try runtimeProfileU32FromIntegerAttr(attr, field, diagnostics);
}

fn profileJoinOptionalU32FromIntegerAttr(attr: mlir.MlirAttribute, field: []const u8, diagnostics: *std.Io.Writer) !?u32 {
    if (mlir.mlirAttributeIsNull(attr) or stringAttrEquals(attr, "none")) return null;
    return try profileJoinU32FromIntegerAttr(attr, field, diagnostics);
}

fn codegenOptionalU32FromIntegerAttr(attr: mlir.MlirAttribute, field: []const u8, diagnostics: *std.Io.Writer) !?u32 {
    if (mlir.mlirAttributeIsNull(attr) or stringAttrEquals(attr, "none")) return null;
    return try codegenU32FromIntegerAttr(attr, field, diagnostics);
}

fn u128FromStringAttr(attr: mlir.MlirAttribute, field: []const u8, diagnostics: *std.Io.Writer) !u128 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAString(attr)) {
        try diagnostics.print("pass=fusion_extract feature={s} reason=expected string-encoded integer attribute\n", .{field});
        return MlirStateError.InvalidFusionPlan;
    }
    return parseU128(mlirStringSlice(mlir.mlirStringAttrGetValue(attr)), diagnostics);
}

fn collectiveU128FromStringAttr(attr: mlir.MlirAttribute, field: []const u8, diagnostics: *std.Io.Writer) !u128 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAString(attr)) {
        try diagnostics.print("pass=collective_extract feature={s} reason=expected string-encoded integer attribute\n", .{field});
        return MlirStateError.InvalidCollectivePlan;
    }
    return std.fmt.parseUnsigned(u128, mlirStringSlice(mlir.mlirStringAttrGetValue(attr)), 10) catch {
        try diagnostics.print("pass=collective_extract feature={s} reason=invalid unsigned integer\n", .{field});
        return MlirStateError.InvalidCollectivePlan;
    };
}

fn codegenU128FromStringAttr(attr: mlir.MlirAttribute, field: []const u8, diagnostics: *std.Io.Writer) !u128 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAString(attr)) {
        try diagnostics.print("pass=codegen_extract feature={s} reason=expected string-encoded integer attribute\n", .{field});
        return MlirStateError.InvalidKernelCodegenPlan;
    }
    return std.fmt.parseUnsigned(u128, mlirStringSlice(mlir.mlirStringAttrGetValue(attr)), 10) catch {
        try diagnostics.print("pass=codegen_extract feature={s} reason=invalid unsigned integer\n", .{field});
        return MlirStateError.InvalidKernelCodegenPlan;
    };
}

fn performanceU128FromStringAttr(attr: mlir.MlirAttribute, field: []const u8, diagnostics: *std.Io.Writer) !u128 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAString(attr)) {
        try diagnostics.print("pass=performance_extract feature={s} reason=expected string-encoded integer attribute\n", .{field});
        return MlirStateError.InvalidKernelCodegenPlan;
    }
    return std.fmt.parseUnsigned(u128, mlirStringSlice(mlir.mlirStringAttrGetValue(attr)), 10) catch {
        try diagnostics.print("pass=performance_extract feature={s} reason=invalid unsigned integer\n", .{field});
        return MlirStateError.InvalidKernelCodegenPlan;
    };
}

fn runtimeAllocationU128FromStringAttr(attr: mlir.MlirAttribute, field: []const u8, diagnostics: *std.Io.Writer) !u128 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAString(attr)) {
        try diagnostics.print("pass=runtime_allocation_extract feature={s} reason=expected string-encoded integer attribute\n", .{field});
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    return std.fmt.parseUnsigned(u128, mlirStringSlice(mlir.mlirStringAttrGetValue(attr)), 10) catch {
        try diagnostics.print("pass=runtime_allocation_extract feature={s} reason=invalid unsigned integer\n", .{field});
        return MlirStateError.InvalidBackendExecutablePlan;
    };
}

fn runtimeProfileU64FromStringAttr(attr: mlir.MlirAttribute, field: []const u8, diagnostics: *std.Io.Writer) !u64 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAString(attr)) {
        try diagnostics.print("pass=runtime_profile_extract feature={s} reason=expected string-encoded integer attribute\n", .{field});
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    return std.fmt.parseUnsigned(u64, mlirStringSlice(mlir.mlirStringAttrGetValue(attr)), 10) catch {
        try diagnostics.print("pass=runtime_profile_extract feature={s} reason=invalid unsigned integer\n", .{field});
        return MlirStateError.InvalidBackendExecutablePlan;
    };
}

fn runtimeProfileU128FromStringAttr(attr: mlir.MlirAttribute, field: []const u8, diagnostics: *std.Io.Writer) !u128 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAString(attr)) {
        try diagnostics.print("pass=runtime_profile_extract feature={s} reason=expected string-encoded integer attribute\n", .{field});
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    return std.fmt.parseUnsigned(u128, mlirStringSlice(mlir.mlirStringAttrGetValue(attr)), 10) catch {
        try diagnostics.print("pass=runtime_profile_extract feature={s} reason=invalid unsigned integer\n", .{field});
        return MlirStateError.InvalidBackendExecutablePlan;
    };
}

fn collectiveOptionalU64FromStringAttr(attr: mlir.MlirAttribute, field: []const u8, diagnostics: *std.Io.Writer) !?u64 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAString(attr)) {
        try diagnostics.print("pass=collective_extract feature={s} reason=expected string-encoded optional integer attribute\n", .{field});
        return MlirStateError.InvalidCollectivePlan;
    }
    const text = mlirStringSlice(mlir.mlirStringAttrGetValue(attr));
    if (std.mem.eql(u8, text, "none")) return null;
    return std.fmt.parseUnsigned(u64, text, 10) catch {
        try diagnostics.print("pass=collective_extract feature={s} reason=invalid optional unsigned integer\n", .{field});
        return MlirStateError.InvalidCollectivePlan;
    };
}

fn targetOptionalU64FromStringAttr(attr: mlir.MlirAttribute, field: []const u8, diagnostics: *std.Io.Writer) !?u64 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAString(attr)) {
        try diagnostics.print("pass=target_extract feature={s} reason=expected string-encoded optional integer attribute\n", .{field});
        return MlirStateError.InvalidTargetAttachment;
    }
    const text = mlirStringSlice(mlir.mlirStringAttrGetValue(attr));
    if (std.mem.eql(u8, text, "none")) return null;
    return std.fmt.parseUnsigned(u64, text, 10) catch {
        try diagnostics.print("pass=target_extract feature={s} reason=invalid optional unsigned integer\n", .{field});
        return MlirStateError.InvalidTargetAttachment;
    };
}

fn targetOptionalF64FromStringAttr(attr: mlir.MlirAttribute, field: []const u8, diagnostics: *std.Io.Writer) !?f64 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAString(attr)) {
        try diagnostics.print("pass=target_extract feature={s} reason=expected string-encoded optional float attribute\n", .{field});
        return MlirStateError.InvalidTargetAttachment;
    }
    const text = mlirStringSlice(mlir.mlirStringAttrGetValue(attr));
    if (std.mem.eql(u8, text, "none")) return null;
    return std.fmt.parseFloat(f64, text) catch {
        try diagnostics.print("pass=target_extract feature={s} reason=invalid optional float\n", .{field});
        return MlirStateError.InvalidTargetAttachment;
    };
}

fn targetStringFromAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    field: []const u8,
    diagnostics: *std.Io.Writer,
) ![]const u8 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAString(attr)) {
        try diagnostics.print("pass=target_extract feature={s} reason=expected string attribute\n", .{field});
        return MlirStateError.InvalidTargetAttachment;
    }
    const value = mlirStringSlice(mlir.mlirStringAttrGetValue(attr));
    if (value.len == 0) {
        try diagnostics.print("pass=target_extract feature={s} reason=empty string attribute\n", .{field});
        return MlirStateError.InvalidTargetAttachment;
    }
    return allocator.dupe(u8, value);
}

fn performanceStringFromAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    field: []const u8,
    diagnostics: *std.Io.Writer,
) ![]const u8 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAString(attr)) {
        try diagnostics.print("pass=performance_extract feature={s} reason=expected string attribute\n", .{field});
        return MlirStateError.InvalidKernelCodegenPlan;
    }
    const value = mlirStringSlice(mlir.mlirStringAttrGetValue(attr));
    if (value.len == 0) {
        try diagnostics.print("pass=performance_extract feature={s} reason=empty string attribute\n", .{field});
        return MlirStateError.InvalidKernelCodegenPlan;
    }
    return allocator.dupe(u8, value);
}

fn performanceOptionalStringFromAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    field: []const u8,
    diagnostics: *std.Io.Writer,
) ![]const u8 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAString(attr)) {
        try diagnostics.print("pass=performance_extract feature={s} reason=expected string attribute\n", .{field});
        return MlirStateError.InvalidKernelCodegenPlan;
    }
    return allocator.dupe(u8, mlirStringSlice(mlir.mlirStringAttrGetValue(attr)));
}

fn loweringStringFromAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    field: []const u8,
    diagnostics: *std.Io.Writer,
) ![]const u8 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAString(attr)) {
        try diagnostics.print("pass=lowering_extract feature={s} reason=expected string attribute\n", .{field});
        return MlirStateError.InvalidLoweringPlan;
    }
    const value = mlirStringSlice(mlir.mlirStringAttrGetValue(attr));
    if (value.len == 0) {
        try diagnostics.print("pass=lowering_extract feature={s} reason=empty string attribute\n", .{field});
        return MlirStateError.InvalidLoweringPlan;
    }
    return allocator.dupe(u8, value);
}

fn loweringStringsFromArrayAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    field: []const u8,
    require_non_empty: bool,
    diagnostics: *std.Io.Writer,
) ![]const []const u8 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAArray(attr)) {
        try diagnostics.print("pass=lowering_extract feature={s} reason=expected string array attribute\n", .{field});
        return MlirStateError.InvalidLoweringPlan;
    }
    const count = mlir.mlirArrayAttrGetNumElements(attr);
    if ((require_non_empty and count == 0) or count > limits.max_kernel_ref_ids) {
        try diagnostics.print("pass=lowering_extract feature={s} reason=invalid string array length\n", .{field});
        return MlirStateError.InvalidLoweringPlan;
    }
    const values = try allocator.alloc([]const u8, std.math.cast(usize, count) orelse unreachable);
    errdefer allocator.free(values);
    var index: isize = 0;
    errdefer {
        const initialized: usize = std.math.cast(usize, index) orelse 0;
        for (values[0..initialized]) |value| allocator.free(value);
    }
    while (index < count) : (index += 1) {
        values[std.math.cast(usize, index) orelse unreachable] = try loweringStringFromAttr(allocator, mlir.mlirArrayAttrGetElement(attr, index), field, diagnostics);
    }
    return values;
}

fn placementStringFromAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    field: []const u8,
    diagnostics: *std.Io.Writer,
) ![]const u8 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAString(attr)) {
        try diagnostics.print("pass=placement_extract feature={s} reason=expected string attribute\n", .{field});
        return MlirStateError.InvalidPlacementPlan;
    }
    const value = mlirStringSlice(mlir.mlirStringAttrGetValue(attr));
    if (value.len == 0) {
        try diagnostics.print("pass=placement_extract feature={s} reason=empty string attribute\n", .{field});
        return MlirStateError.InvalidPlacementPlan;
    }
    return allocator.dupe(u8, value);
}

fn collectiveStringFromAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    field: []const u8,
    diagnostics: *std.Io.Writer,
) ![]const u8 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAString(attr)) {
        try diagnostics.print("pass=collective_extract feature={s} reason=expected string attribute\n", .{field});
        return MlirStateError.InvalidCollectivePlan;
    }
    const value = mlirStringSlice(mlir.mlirStringAttrGetValue(attr));
    if (value.len == 0) {
        try diagnostics.print("pass=collective_extract feature={s} reason=empty string attribute\n", .{field});
        return MlirStateError.InvalidCollectivePlan;
    }
    return allocator.dupe(u8, value);
}

fn codegenStringFromAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    field: []const u8,
    diagnostics: *std.Io.Writer,
) ![]const u8 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAString(attr)) {
        try diagnostics.print("pass=codegen_extract feature={s} reason=expected string attribute\n", .{field});
        return MlirStateError.InvalidKernelCodegenPlan;
    }
    const value = mlirStringSlice(mlir.mlirStringAttrGetValue(attr));
    if (value.len == 0) {
        try diagnostics.print("pass=codegen_extract feature={s} reason=empty string attribute\n", .{field});
        return MlirStateError.InvalidKernelCodegenPlan;
    }
    return allocator.dupe(u8, value);
}

fn scheduleStringFromAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    field: []const u8,
    diagnostics: *std.Io.Writer,
) ![]const u8 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAString(attr)) {
        try diagnostics.print("pass=schedule_extract feature={s} reason=expected string attribute\n", .{field});
        return MlirStateError.InvalidSchedulePlan;
    }
    const value = mlirStringSlice(mlir.mlirStringAttrGetValue(attr));
    if (value.len == 0) {
        try diagnostics.print("pass=schedule_extract feature={s} reason=empty string attribute\n", .{field});
        return MlirStateError.InvalidSchedulePlan;
    }
    return allocator.dupe(u8, value);
}

fn backendBindingStringFromAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    field: []const u8,
    diagnostics: *std.Io.Writer,
) ![]const u8 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAString(attr)) {
        try diagnostics.print("pass=backend_binding_extract feature={s} reason=expected string attribute\n", .{field});
        return MlirStateError.InvalidBackendBindingPlan;
    }
    const value = mlirStringSlice(mlir.mlirStringAttrGetValue(attr));
    if (value.len == 0) {
        try diagnostics.print("pass=backend_binding_extract feature={s} reason=empty string attribute\n", .{field});
        return MlirStateError.InvalidBackendBindingPlan;
    }
    return allocator.dupe(u8, value);
}

fn backendPlanStringFromAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    field: []const u8,
    diagnostics: *std.Io.Writer,
) ![]const u8 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAString(attr)) {
        try diagnostics.print("pass=backend_plan_extract feature={s} reason=expected string attribute\n", .{field});
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    const value = mlirStringSlice(mlir.mlirStringAttrGetValue(attr));
    if (value.len == 0) {
        try diagnostics.print("pass=backend_plan_extract feature={s} reason=empty string attribute\n", .{field});
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    return allocator.dupe(u8, value);
}

fn runtimeAllocationStringFromAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    field: []const u8,
    diagnostics: *std.Io.Writer,
) ![]const u8 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAString(attr)) {
        try diagnostics.print("pass=runtime_allocation_extract feature={s} reason=expected string attribute\n", .{field});
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    const value = mlirStringSlice(mlir.mlirStringAttrGetValue(attr));
    if (value.len == 0) {
        try diagnostics.print("pass=runtime_allocation_extract feature={s} reason=empty string attribute\n", .{field});
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    return allocator.dupe(u8, value);
}

fn runtimeProfileStringFromAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    field: []const u8,
    diagnostics: *std.Io.Writer,
) ![]const u8 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAString(attr)) {
        try diagnostics.print("pass=runtime_profile_extract feature={s} reason=expected string attribute\n", .{field});
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    const value = mlirStringSlice(mlir.mlirStringAttrGetValue(attr));
    if (value.len == 0) {
        try diagnostics.print("pass=runtime_profile_extract feature={s} reason=empty string attribute\n", .{field});
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    return allocator.dupe(u8, value);
}

fn profileJoinStringFromAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    field: []const u8,
    diagnostics: *std.Io.Writer,
) ![]const u8 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAString(attr)) {
        try diagnostics.print("pass=profile_join_extract feature={s} reason=expected string attribute\n", .{field});
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    const value = mlirStringSlice(mlir.mlirStringAttrGetValue(attr));
    if (value.len == 0) {
        try diagnostics.print("pass=profile_join_extract feature={s} reason=empty string attribute\n", .{field});
        return MlirStateError.InvalidBackendExecutablePlan;
    }
    return allocator.dupe(u8, value);
}

fn collectiveDecisionFromStringAttr(attr: mlir.MlirAttribute, diagnostics: *std.Io.Writer) !compiler_facts.CollectivePlanDecision {
    const value = stringAttrValue(attr) orelse {
        try diagnostics.writeAll("pass=collective_extract feature=decision reason=expected string attribute\n");
        return MlirStateError.InvalidCollectivePlan;
    };
    inline for (std.meta.fields(compiler_facts.CollectivePlanDecision)) |field| {
        if (std.mem.eql(u8, value, field.name)) return @enumFromInt(field.value);
    }
    try diagnostics.writeAll("pass=collective_extract feature=decision reason=unknown collective decision\n");
    return MlirStateError.InvalidCollectivePlan;
}

fn collectiveAlgorithmFromStringAttr(attr: mlir.MlirAttribute, diagnostics: *std.Io.Writer) !compiler_facts.CollectiveAlgorithm {
    const value = stringAttrValue(attr) orelse {
        try diagnostics.writeAll("pass=collective_extract feature=algorithm reason=expected string attribute\n");
        return MlirStateError.InvalidCollectivePlan;
    };
    inline for (std.meta.fields(compiler_facts.CollectiveAlgorithm)) |field| {
        if (std.mem.eql(u8, value, field.name)) return @enumFromInt(field.value);
    }
    try diagnostics.writeAll("pass=collective_extract feature=algorithm reason=unknown collective algorithm\n");
    return MlirStateError.InvalidCollectivePlan;
}

fn loweringDecisionFromStringAttr(attr: mlir.MlirAttribute, diagnostics: *std.Io.Writer) !compiler_facts.LoweringDecision {
    const value = stringAttrValue(attr) orelse {
        try diagnostics.writeAll("pass=lowering_extract feature=decision reason=expected string attribute\n");
        return MlirStateError.InvalidLoweringPlan;
    };
    inline for (std.meta.fields(compiler_facts.LoweringDecision)) |field| {
        if (std.mem.eql(u8, value, field.name)) return @enumFromInt(field.value);
    }
    try diagnostics.writeAll("pass=lowering_extract feature=decision reason=unknown lowering decision\n");
    return MlirStateError.InvalidLoweringPlan;
}

fn backendKindFromStringAttr(attr: mlir.MlirAttribute, diagnostics: *std.Io.Writer) !core.BackendKind {
    const value = stringAttrValue(attr) orelse {
        try diagnostics.writeAll("pass=codegen_extract feature=backend-kind reason=expected string attribute\n");
        return MlirStateError.InvalidKernelCodegenPlan;
    };
    inline for (std.meta.fields(core.BackendKind)) |field| {
        if (std.mem.eql(u8, value, field.name)) return @enumFromInt(field.value);
    }
    try diagnostics.writeAll("pass=codegen_extract feature=backend-kind reason=unknown backend kind\n");
    return MlirStateError.InvalidKernelCodegenPlan;
}

fn targetKindFromStringAttr(attr: mlir.MlirAttribute, diagnostics: *std.Io.Writer) !target_pkg.TargetKind {
    const value = stringAttrValue(attr) orelse {
        try diagnostics.writeAll("pass=target_extract feature=kind reason=expected string attribute\n");
        return MlirStateError.InvalidTargetAttachment;
    };
    inline for (std.meta.fields(target_pkg.TargetKind)) |field| {
        if (std.mem.eql(u8, value, field.name)) return @enumFromInt(field.value);
    }
    try diagnostics.writeAll("pass=target_extract feature=kind reason=unknown target kind\n");
    return MlirStateError.InvalidTargetAttachment;
}

fn memorySpaceKindFromStringAttr(attr: mlir.MlirAttribute, diagnostics: *std.Io.Writer) !target_pkg.MemorySpaceKind {
    const value = stringAttrValue(attr) orelse {
        try diagnostics.writeAll("pass=target_extract feature=memory-space-kind reason=expected string attribute\n");
        return MlirStateError.InvalidTargetAttachment;
    };
    inline for (std.meta.fields(target_pkg.MemorySpaceKind)) |field| {
        if (std.mem.eql(u8, value, field.name)) return @enumFromInt(field.value);
    }
    try diagnostics.writeAll("pass=target_extract feature=memory-space-kind reason=unknown memory space kind\n");
    return MlirStateError.InvalidTargetAttachment;
}

fn executionUnitKindFromStringAttr(attr: mlir.MlirAttribute, diagnostics: *std.Io.Writer) !target_pkg.ExecutionUnitKind {
    const value = stringAttrValue(attr) orelse {
        try diagnostics.writeAll("pass=target_extract feature=execution-unit-kind reason=expected string attribute\n");
        return MlirStateError.InvalidTargetAttachment;
    };
    inline for (std.meta.fields(target_pkg.ExecutionUnitKind)) |field| {
        if (std.mem.eql(u8, value, field.name)) return @enumFromInt(field.value);
    }
    try diagnostics.writeAll("pass=target_extract feature=execution-unit-kind reason=unknown execution unit kind\n");
    return MlirStateError.InvalidTargetAttachment;
}

fn bufferTypeFromStringAttrForTarget(attr: mlir.MlirAttribute, diagnostics: *std.Io.Writer) !core.BufferType {
    const value = stringAttrValue(attr) orelse {
        try diagnostics.writeAll("pass=target_extract feature=dtype reason=expected string attribute\n");
        return MlirStateError.InvalidTargetAttachment;
    };
    inline for (std.meta.fields(core.BufferType)) |field| {
        if (std.mem.eql(u8, value, field.name)) return @enumFromInt(field.value);
    }
    try diagnostics.writeAll("pass=target_extract feature=dtype reason=unknown buffer type\n");
    return MlirStateError.InvalidTargetAttachment;
}

fn opClassFromStringAttr(attr: mlir.MlirAttribute, diagnostics: *std.Io.Writer) !core.OpClass {
    const value = stringAttrValue(attr) orelse {
        try diagnostics.writeAll("pass=target_extract feature=op-class reason=expected string attribute\n");
        return MlirStateError.InvalidTargetAttachment;
    };
    inline for (std.meta.fields(core.OpClass)) |field| {
        if (std.mem.eql(u8, value, field.name)) return @enumFromInt(field.value);
    }
    try diagnostics.writeAll("pass=target_extract feature=op-class reason=unknown op class\n");
    return MlirStateError.InvalidTargetAttachment;
}

fn rateSourceFromStringAttr(attr: mlir.MlirAttribute, diagnostics: *std.Io.Writer) !core.RateSource {
    const value = stringAttrValue(attr) orelse {
        try diagnostics.writeAll("pass=target_extract feature=rate-source reason=expected string attribute\n");
        return MlirStateError.InvalidTargetAttachment;
    };
    inline for (std.meta.fields(core.RateSource)) |field| {
        if (std.mem.eql(u8, value, field.name)) return @enumFromInt(field.value);
    }
    try diagnostics.writeAll("pass=target_extract feature=rate-source reason=unknown rate source\n");
    return MlirStateError.InvalidTargetAttachment;
}

fn sourceFrontendFromStringAttr(attr: mlir.MlirAttribute, diagnostics: *std.Io.Writer) !compiler_facts.SourceFrontend {
    const value = stringAttrValue(attr) orelse {
        try diagnostics.writeAll("pass=performance_extract feature=source-frontend reason=expected string attribute\n");
        return MlirStateError.InvalidKernelCodegenPlan;
    };
    inline for (std.meta.fields(compiler_facts.SourceFrontend)) |field| {
        if (std.mem.eql(u8, value, field.name)) return @enumFromInt(field.value);
    }
    try diagnostics.writeAll("pass=performance_extract feature=source-frontend reason=unknown frontend\n");
    return MlirStateError.InvalidKernelCodegenPlan;
}

fn costOpClassFromStringAttr(attr: mlir.MlirAttribute, diagnostics: *std.Io.Writer) !compiler_facts.CostOpClass {
    const value = stringAttrValue(attr) orelse {
        try diagnostics.writeAll("pass=performance_extract feature=op-class reason=expected string attribute\n");
        return MlirStateError.InvalidKernelCodegenPlan;
    };
    inline for (std.meta.fields(compiler_facts.CostOpClass)) |field| {
        if (std.mem.eql(u8, value, field.name)) return @enumFromInt(field.value);
    }
    try diagnostics.writeAll("pass=performance_extract feature=op-class reason=unknown cost op class\n");
    return MlirStateError.InvalidKernelCodegenPlan;
}

fn memoryTrafficKindFromStringAttr(attr: mlir.MlirAttribute, diagnostics: *std.Io.Writer) !compiler_facts.MemoryTrafficKind {
    const value = stringAttrValue(attr) orelse {
        try diagnostics.writeAll("pass=performance_extract feature=memory-traffic-kind reason=expected string attribute\n");
        return MlirStateError.InvalidKernelCodegenPlan;
    };
    inline for (std.meta.fields(compiler_facts.MemoryTrafficKind)) |field| {
        if (std.mem.eql(u8, value, field.name)) return @enumFromInt(field.value);
    }
    try diagnostics.writeAll("pass=performance_extract feature=memory-traffic-kind reason=unknown memory traffic kind\n");
    return MlirStateError.InvalidKernelCodegenPlan;
}

fn bufferTypeFromStringAttrForPerformance(attr: mlir.MlirAttribute, diagnostics: *std.Io.Writer) !core.BufferType {
    const value = stringAttrValue(attr) orelse {
        try diagnostics.writeAll("pass=performance_extract feature=dtype reason=expected string attribute\n");
        return MlirStateError.InvalidKernelCodegenPlan;
    };
    inline for (std.meta.fields(core.BufferType)) |field| {
        if (std.mem.eql(u8, value, field.name)) return @enumFromInt(field.value);
    }
    try diagnostics.writeAll("pass=performance_extract feature=dtype reason=unknown buffer type\n");
    return MlirStateError.InvalidKernelCodegenPlan;
}

fn optionalBufferTypeFromStringAttrForPerformance(attr: mlir.MlirAttribute, diagnostics: *std.Io.Writer) !?core.BufferType {
    if (mlir.mlirAttributeIsNull(attr) or stringAttrEquals(attr, "none")) return null;
    return try bufferTypeFromStringAttrForPerformance(attr, diagnostics);
}

fn kernelCodegenKindFromStringAttr(attr: mlir.MlirAttribute, diagnostics: *std.Io.Writer) !core.KernelCodegenKind {
    const value = stringAttrValue(attr) orelse {
        try diagnostics.writeAll("pass=codegen_extract feature=kind reason=expected string attribute\n");
        return MlirStateError.InvalidKernelCodegenPlan;
    };
    inline for (std.meta.fields(core.KernelCodegenKind)) |field| {
        if (std.mem.eql(u8, value, field.name)) return @enumFromInt(field.value);
    }
    try diagnostics.writeAll("pass=codegen_extract feature=kind reason=unknown kernel codegen kind\n");
    return MlirStateError.InvalidKernelCodegenPlan;
}

fn commandKindFromStringAttr(attr: mlir.MlirAttribute, diagnostics: *std.Io.Writer) !core.CommandKind {
    const value = stringAttrValue(attr) orelse {
        try diagnostics.writeAll("pass=schedule_extract feature=kind reason=expected string attribute\n");
        return MlirStateError.InvalidSchedulePlan;
    };
    inline for (std.meta.fields(core.CommandKind)) |field| {
        if (std.mem.eql(u8, value, field.name)) return @enumFromInt(field.value);
    }
    try diagnostics.writeAll("pass=schedule_extract feature=kind reason=unknown command kind\n");
    return MlirStateError.InvalidSchedulePlan;
}

fn backendKindFromStringAttrForBinding(attr: mlir.MlirAttribute, diagnostics: *std.Io.Writer) !core.BackendKind {
    const value = stringAttrValue(attr) orelse {
        try diagnostics.writeAll("pass=backend_binding_extract feature=backend-kind reason=expected string attribute\n");
        return MlirStateError.InvalidBackendBindingPlan;
    };
    inline for (std.meta.fields(core.BackendKind)) |field| {
        if (std.mem.eql(u8, value, field.name)) return @enumFromInt(field.value);
    }
    try diagnostics.writeAll("pass=backend_binding_extract feature=backend-kind reason=unknown backend kind\n");
    return MlirStateError.InvalidBackendBindingPlan;
}

fn backendKindFromStringAttrForExecutable(attr: mlir.MlirAttribute, diagnostics: *std.Io.Writer) !core.BackendKind {
    const value = stringAttrValue(attr) orelse {
        try diagnostics.writeAll("pass=backend_plan_extract feature=backend-kind reason=expected string attribute\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    };
    inline for (std.meta.fields(core.BackendKind)) |field| {
        if (std.mem.eql(u8, value, field.name)) return @enumFromInt(field.value);
    }
    try diagnostics.writeAll("pass=backend_plan_extract feature=backend-kind reason=unknown backend kind\n");
    return MlirStateError.InvalidBackendExecutablePlan;
}

fn bufferTypeFromStringAttrForBackendPlan(attr: mlir.MlirAttribute, diagnostics: *std.Io.Writer) !core.BufferType {
    const value = stringAttrValue(attr) orelse {
        try diagnostics.writeAll("pass=backend_kernel_graph_extract feature=dtype reason=expected string attribute\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    };
    inline for (std.meta.fields(core.BufferType)) |field| {
        if (std.mem.eql(u8, value, field.name)) return @enumFromInt(field.value);
    }
    try diagnostics.writeAll("pass=backend_kernel_graph_extract feature=dtype reason=unknown buffer type\n");
    return MlirStateError.InvalidBackendExecutablePlan;
}

fn layoutFromStringAttrForBackendPlan(attr: mlir.MlirAttribute, diagnostics: *std.Io.Writer) !compiler_facts.LayoutKind {
    const value = stringAttrValue(attr) orelse {
        try diagnostics.writeAll("pass=backend_kernel_graph_extract feature=layout reason=expected string attribute\n");
        return MlirStateError.InvalidBackendExecutablePlan;
    };
    inline for (std.meta.fields(compiler_facts.LayoutKind)) |field| {
        if (std.mem.eql(u8, value, field.name)) return @enumFromInt(field.value);
    }
    try diagnostics.writeAll("pass=backend_kernel_graph_extract feature=layout reason=unknown layout\n");
    return MlirStateError.InvalidBackendExecutablePlan;
}

fn dependencyKindFromStringAttr(attr: mlir.MlirAttribute, diagnostics: *std.Io.Writer) !core.DependencyKind {
    const value = stringAttrValue(attr) orelse {
        try diagnostics.writeAll("pass=schedule_extract feature=dependency-kind reason=expected string attribute\n");
        return MlirStateError.InvalidSchedulePlan;
    };
    inline for (std.meta.fields(core.DependencyKind)) |field| {
        if (std.mem.eql(u8, value, field.name)) return @enumFromInt(field.value);
    }
    try diagnostics.writeAll("pass=schedule_extract feature=dependency-kind reason=unknown dependency kind\n");
    return MlirStateError.InvalidSchedulePlan;
}

fn scheduleOverlapDecisionFromStringAttr(attr: mlir.MlirAttribute, diagnostics: *std.Io.Writer) !core.ScheduleOverlapDecision {
    const value = stringAttrValue(attr) orelse {
        try diagnostics.writeAll("pass=schedule_extract feature=overlap-decision reason=expected string attribute\n");
        return MlirStateError.InvalidSchedulePlan;
    };
    inline for (std.meta.fields(core.ScheduleOverlapDecision)) |field| {
        if (std.mem.eql(u8, value, field.name)) return @enumFromInt(field.value);
    }
    try diagnostics.writeAll("pass=schedule_extract feature=overlap-decision reason=unknown overlap decision\n");
    return MlirStateError.InvalidSchedulePlan;
}

fn scheduleOverlapKindFromStringAttr(attr: mlir.MlirAttribute, diagnostics: *std.Io.Writer) !core.ScheduleOverlapKind {
    const value = stringAttrValue(attr) orelse {
        try diagnostics.writeAll("pass=schedule_extract feature=overlap-kind reason=expected string attribute\n");
        return MlirStateError.InvalidSchedulePlan;
    };
    inline for (std.meta.fields(core.ScheduleOverlapKind)) |field| {
        if (std.mem.eql(u8, value, field.name)) return @enumFromInt(field.value);
    }
    try diagnostics.writeAll("pass=schedule_extract feature=overlap-kind reason=unknown overlap kind\n");
    return MlirStateError.InvalidSchedulePlan;
}

fn kernelShapeFromAttr(attr: mlir.MlirAttribute, diagnostics: *std.Io.Writer) !core.KernelCodegenShape {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsADictionary(attr)) {
        try diagnostics.writeAll("pass=codegen_extract feature=shape reason=shape is not a dictionary attribute\n");
        return MlirStateError.InvalidKernelCodegenPlan;
    }
    return .{
        .operation_count = try codegenU32FromIntegerAttr(dictAttr(attr, "operations"), "shape.operations", diagnostics),
        .external_input_count = try codegenU32FromIntegerAttr(dictAttr(attr, "external_inputs"), "shape.external_inputs", diagnostics),
        .external_output_count = try codegenU32FromIntegerAttr(dictAttr(attr, "external_outputs"), "shape.external_outputs", diagnostics),
        .intermediate_value_count = try codegenU32FromIntegerAttr(dictAttr(attr, "intermediates"), "shape.intermediates", diagnostics),
    };
}

fn kernelPressureFromAttr(attr: mlir.MlirAttribute, diagnostics: *std.Io.Writer) !core.KernelMemoryPressure {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsADictionary(attr)) {
        try diagnostics.writeAll("pass=codegen_extract feature=pressure reason=pressure is not a dictionary attribute\n");
        return MlirStateError.InvalidKernelCodegenPlan;
    }
    return .{
        .global_bytes_read = try codegenU128FromStringAttr(dictAttr(attr, "global_read"), "pressure.global_read", diagnostics),
        .global_bytes_written = try codegenU128FromStringAttr(dictAttr(attr, "global_write"), "pressure.global_write", diagnostics),
        .local_bytes_read = try codegenU128FromStringAttr(dictAttr(attr, "local_read"), "pressure.local_read", diagnostics),
        .local_bytes_written = try codegenU128FromStringAttr(dictAttr(attr, "local_write"), "pressure.local_write", diagnostics),
    };
}

fn layoutFromStringAttr(attr: mlir.MlirAttribute, diagnostics: *std.Io.Writer) !compiler_facts.LayoutKind {
    const value = stringAttrValue(attr) orelse {
        try diagnostics.writeAll("pass=placement_extract feature=layout reason=expected string attribute\n");
        return MlirStateError.InvalidPlacementPlan;
    };
    inline for (std.meta.fields(compiler_facts.LayoutKind)) |field| {
        if (std.mem.eql(u8, value, field.name)) {
            return @enumFromInt(field.value);
        }
    }
    try diagnostics.writeAll("pass=placement_extract feature=layout reason=unknown layout\n");
    return MlirStateError.InvalidPlacementPlan;
}

fn stringFromAttr(
    allocator: std.mem.Allocator,
    attr: mlir.MlirAttribute,
    field: []const u8,
    diagnostics: *std.Io.Writer,
) ![]const u8 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAString(attr)) {
        try diagnostics.print("pass=fusion_extract feature={s} reason=expected string attribute\n", .{field});
        return MlirStateError.InvalidFusionPlan;
    }
    return allocator.dupe(u8, mlirStringSlice(mlir.mlirStringAttrGetValue(attr)));
}

fn stringAttrValue(attr: mlir.MlirAttribute) ?[]const u8 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAString(attr)) return null;
    return mlirStringSlice(mlir.mlirStringAttrGetValue(attr));
}

fn integerAttrValue(attr: mlir.MlirAttribute) ?u32 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAInteger(attr)) return null;
    return std.math.cast(u32, mlir.mlirIntegerAttrGetValueInt(attr));
}

fn executableContractFromAttrNoDiag(attr: mlir.MlirAttribute) ?ExecutableContract {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsADictionary(attr)) return null;
    const target_kind = targetKindFromStringAttrNoDiag(dictAttr(attr, "target_kind")) orelse return null;
    const schedule_command_count = integerAttrValue(dictAttr(attr, "schedule_commands")) orelse return null;
    const backend_binding_count = integerAttrValue(dictAttr(attr, "backend_bindings")) orelse return null;
    const kernel_codegen_count = integerAttrValue(dictAttr(attr, "kernel_codegen_records")) orelse return null;
    return .{
        .target_kind = target_kind,
        .schedule_command_count = schedule_command_count,
        .backend_binding_count = backend_binding_count,
        .kernel_codegen_count = kernel_codegen_count,
    };
}

fn targetKindFromStringAttrNoDiag(attr: mlir.MlirAttribute) ?target_pkg.TargetKind {
    const text = stringAttrValue(attr) orelse return null;
    inline for (std.meta.fields(target_pkg.TargetKind)) |field| {
        if (std.mem.eql(u8, text, field.name)) return @enumFromInt(field.value);
    }
    return null;
}

fn arrayAttrCountEquals(attr: mlir.MlirAttribute, expected_count: u32) bool {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAArray(attr)) return false;
    const count = mlir.mlirArrayAttrGetNumElements(attr);
    const expected_isize = std.math.cast(isize, expected_count) orelse return false;
    return count == expected_isize;
}

fn nonEmptyArrayAttr(attr: mlir.MlirAttribute) bool {
    return !mlir.mlirAttributeIsNull(attr) and mlir.mlirAttributeIsAArray(attr) and mlir.mlirArrayAttrGetNumElements(attr) > 0;
}

fn validU64StringAttr(attr: mlir.MlirAttribute) bool {
    const text = stringAttrValue(attr) orelse return false;
    _ = std.fmt.parseUnsigned(u64, text, 10) catch return false;
    return true;
}

fn boolAttrValue(attr: mlir.MlirAttribute) ?bool {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsABool(attr)) return null;
    return mlir.mlirBoolAttrGetValue(attr);
}

fn u128FromStringAttrNoDiag(attr: mlir.MlirAttribute) ?u128 {
    const value = stringAttrValue(attr) orelse return null;
    return std.fmt.parseUnsigned(u128, value, 10) catch null;
}

fn codegenU128FromStringAttrNoDiag(attr: mlir.MlirAttribute) ?u128 {
    const value = stringAttrValue(attr) orelse return null;
    return std.fmt.parseUnsigned(u128, value, 10) catch null;
}

fn stringAttrEquals(attr: mlir.MlirAttribute, expected: []const u8) bool {
    return std.mem.eql(u8, stringAttrValue(attr) orelse return false, expected);
}

fn hasNonEmptyStringDictAttr(attr: mlir.MlirAttribute, name: []const u8) bool {
    return (stringAttrValue(dictAttr(attr, name)) orelse return false).len != 0;
}

fn hasStringAttr(op: mlir.MlirOperation, name: []const u8) bool {
    const attr = getAttr(op, name);
    return !mlir.mlirAttributeIsNull(attr) and mlir.mlirAttributeIsAString(attr);
}

fn hasIntegerAttr(op: mlir.MlirOperation, name: []const u8) bool {
    const attr = getAttr(op, name);
    return !mlir.mlirAttributeIsNull(attr) and mlir.mlirAttributeIsAInteger(attr);
}

fn hasDictionaryAttr(op: mlir.MlirOperation, name: []const u8) bool {
    const attr = getAttr(op, name);
    return !mlir.mlirAttributeIsNull(attr) and mlir.mlirAttributeIsADictionary(attr);
}

fn hasIntegerDictAttr(attr: mlir.MlirAttribute, name: []const u8) bool {
    const field = dictAttr(attr, name);
    return !mlir.mlirAttributeIsNull(field) and mlir.mlirAttributeIsAInteger(field);
}

fn validOptionalIntegerDictAttr(attr: mlir.MlirAttribute, name: []const u8) bool {
    const field = dictAttr(attr, name);
    return stringAttrEquals(field, "none") or (!mlir.mlirAttributeIsNull(field) and mlir.mlirAttributeIsAInteger(field));
}

fn hasKnownBackendKind(attr: mlir.MlirAttribute) bool {
    const value = stringAttrValue(attr) orelse return false;
    inline for (std.meta.fields(core.BackendKind)) |field| {
        if (std.mem.eql(u8, value, field.name)) return true;
    }
    return false;
}

fn hasKnownCostOpClass(attr: mlir.MlirAttribute) bool {
    const value = stringAttrValue(attr) orelse return false;
    inline for (std.meta.fields(compiler_facts.CostOpClass)) |field| {
        if (std.mem.eql(u8, value, field.name)) return true;
    }
    return false;
}

fn hasKnownMemoryTrafficKind(attr: mlir.MlirAttribute) bool {
    const value = stringAttrValue(attr) orelse return false;
    inline for (std.meta.fields(compiler_facts.MemoryTrafficKind)) |field| {
        if (std.mem.eql(u8, value, field.name)) return true;
    }
    return false;
}

fn hasKnownLoweringDecision(attr: mlir.MlirAttribute) bool {
    const value = stringAttrValue(attr) orelse return false;
    inline for (std.meta.fields(compiler_facts.LoweringDecision)) |field| {
        if (std.mem.eql(u8, value, field.name)) return true;
    }
    return false;
}

fn hasKnownBufferType(attr: mlir.MlirAttribute) bool {
    const value = stringAttrValue(attr) orelse return false;
    inline for (std.meta.fields(core.BufferType)) |field| {
        if (std.mem.eql(u8, value, field.name)) return true;
    }
    return false;
}

fn optionalKnownBufferType(attr: mlir.MlirAttribute) bool {
    if (stringAttrEquals(attr, "none")) return true;
    return hasKnownBufferType(attr);
}

fn hasKnownKernelCodegenKind(attr: mlir.MlirAttribute) bool {
    const value = stringAttrValue(attr) orelse return false;
    inline for (std.meta.fields(core.KernelCodegenKind)) |field| {
        if (std.mem.eql(u8, value, field.name)) return true;
    }
    return false;
}

fn hasKnownCommandKind(attr: mlir.MlirAttribute) bool {
    const value = stringAttrValue(attr) orelse return false;
    inline for (std.meta.fields(core.CommandKind)) |field| {
        if (std.mem.eql(u8, value, field.name)) return true;
    }
    return false;
}

fn hasKnownDependencyKind(attr: mlir.MlirAttribute) bool {
    const value = stringAttrValue(attr) orelse return false;
    inline for (std.meta.fields(core.DependencyKind)) |field| {
        if (std.mem.eql(u8, value, field.name)) return true;
    }
    return false;
}

fn hasKnownScheduleOverlapDecision(attr: mlir.MlirAttribute) bool {
    const value = stringAttrValue(attr) orelse return false;
    inline for (std.meta.fields(core.ScheduleOverlapDecision)) |field| {
        if (std.mem.eql(u8, value, field.name)) return true;
    }
    return false;
}

fn hasKnownScheduleOverlapKind(attr: mlir.MlirAttribute) bool {
    const value = stringAttrValue(attr) orelse return false;
    inline for (std.meta.fields(core.ScheduleOverlapKind)) |field| {
        if (std.mem.eql(u8, value, field.name)) return true;
    }
    return false;
}

fn verifyIntegerArrayAttr(attr: mlir.MlirAttribute, max_count: isize, require_non_empty: bool) bool {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAArray(attr)) return false;
    const count = mlir.mlirArrayAttrGetNumElements(attr);
    if ((require_non_empty and count <= 0) or count > max_count) return false;
    var index: isize = 0;
    while (index < count) : (index += 1) {
        if (!mlir.mlirAttributeIsAInteger(mlir.mlirArrayAttrGetElement(attr, index))) return false;
    }
    return true;
}

fn verifyPositiveIntegerArrayAttr(attr: mlir.MlirAttribute, max_count: isize) bool {
    if (!verifyIntegerArrayAttr(attr, max_count, true)) return false;
    const count = mlir.mlirArrayAttrGetNumElements(attr);
    var index: isize = 0;
    while (index < count) : (index += 1) {
        if (mlir.mlirIntegerAttrGetValueInt(mlir.mlirArrayAttrGetElement(attr, index)) <= 0) return false;
    }
    return true;
}

fn verifyKernelShapeAttr(attr: mlir.MlirAttribute) bool {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsADictionary(attr)) return false;
    return hasIntegerDictAttr(attr, "operations") and
        hasIntegerDictAttr(attr, "external_inputs") and
        hasIntegerDictAttr(attr, "external_outputs") and
        hasIntegerDictAttr(attr, "intermediates");
}

fn verifyKernelPressureAttr(attr: mlir.MlirAttribute) bool {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsADictionary(attr)) return false;
    return codegenU128FromStringAttrNoDiag(dictAttr(attr, "global_read")) != null and
        codegenU128FromStringAttrNoDiag(dictAttr(attr, "global_write")) != null and
        codegenU128FromStringAttrNoDiag(dictAttr(attr, "local_read")) != null and
        codegenU128FromStringAttrNoDiag(dictAttr(attr, "local_write")) != null;
}

fn writeIntegerAttrText(attr: mlir.MlirAttribute, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAInteger(attr)) {
        try writer.writeAll("unknown");
        return;
    }
    try writer.print("{d}", .{mlir.mlirIntegerAttrGetValueInt(attr)});
}

fn writeInstructionArraySummary(attr: mlir.MlirAttribute, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAArray(attr)) {
        try writer.writeAll("unknown");
        return;
    }
    const count = mlir.mlirArrayAttrGetNumElements(attr);
    var index: isize = 0;
    while (index < count) : (index += 1) {
        if (index != 0) try writer.writeByte(',');
        try writeIntegerAttrText(mlir.mlirArrayAttrGetElement(attr, index), writer);
    }
}

fn writeIntegerArraySummary(attr: mlir.MlirAttribute, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAArray(attr)) {
        try writer.writeAll("unknown");
        return;
    }
    const count = mlir.mlirArrayAttrGetNumElements(attr);
    var index: isize = 0;
    while (index < count) : (index += 1) {
        if (index != 0) try writer.writeByte('x');
        try writeIntegerAttrText(mlir.mlirArrayAttrGetElement(attr, index), writer);
    }
}

fn writeFusionCandidateSummary(module_op: mlir.MlirOperation, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    const candidates = getAttr(module_op, "pjrtx.fusion.candidates");
    if (mlir.mlirAttributeIsNull(candidates) or !mlir.mlirAttributeIsAArray(candidates)) {
        try writer.writeAll("  fusion_candidates count=0\n");
        return;
    }

    const count = mlir.mlirArrayAttrGetNumElements(candidates);
    try writer.print("  fusion_candidates count={d}\n", .{count});
    var index: isize = 0;
    while (index < count) : (index += 1) {
        const candidate = mlir.mlirArrayAttrGetElement(candidates, index);
        try writer.writeAll("  candidate.");
        try writeIntegerAttrText(dictAttr(candidate, "index"), writer);
        try writer.print(
            " kind={s} root={s} operations=",
            .{
                stringAttrValue(dictAttr(candidate, "kind")) orelse "unknown",
                stringAttrValue(dictAttr(candidate, "root")) orelse "unknown",
            },
        );
        try writeIntegerAttrText(dictAttr(candidate, "operation_count"), writer);
        try writer.print(" reason={s}", .{stringAttrValue(dictAttr(candidate, "reason")) orelse "unknown"});
        const decision = stringAttrValue(dictAttr(candidate, "decision"));
        if (decision) |decision_text| {
            try writer.print(" decision={s} plan=", .{decision_text});
            try writeIntegerAttrText(dictAttr(candidate, "plan_index"), writer);
            try writer.print(" bytes_saved={s} launch_reduction=", .{stringAttrValue(dictAttr(candidate, "bytes_saved")) orelse "unknown"});
            try writeIntegerAttrText(dictAttr(candidate, "launch_count_reduction"), writer);
            try writer.print(" decision_reason={s}", .{stringAttrValue(dictAttr(candidate, "decision_reason")) orelse "unknown"});
        }
        try writer.writeByte('\n');
    }
}

fn writeFusionPlanSummary(module_op: mlir.MlirOperation, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    const candidates = getAttr(module_op, "pjrtx.fusion.candidates");
    if (mlir.mlirAttributeIsNull(candidates) or !mlir.mlirAttributeIsAArray(candidates)) {
        try writer.writeAll("  fusion_plan count=0\n");
        return;
    }

    const candidate_count = mlir.mlirArrayAttrGetNumElements(candidates);
    var decided_count: isize = 0;
    var count_index: isize = 0;
    while (count_index < candidate_count) : (count_index += 1) {
        const candidate = mlir.mlirArrayAttrGetElement(candidates, count_index);
        if (stringAttrValue(dictAttr(candidate, "decision")) != null) decided_count += 1;
    }

    try writer.print("  fusion_plan count={d}\n", .{decided_count});
    var index: isize = 0;
    while (index < candidate_count) : (index += 1) {
        const candidate = mlir.mlirArrayAttrGetElement(candidates, index);
        if (stringAttrValue(dictAttr(candidate, "decision")) == null) continue;
        try writeFusionPlanCandidateLine(candidate, writer);
    }
}

fn writeFusionPlanCandidateLine(candidate: mlir.MlirAttribute, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    try writer.writeAll("  fusion.");
    try writeIntegerAttrText(dictAttr(candidate, "plan_index"), writer);
    try writer.print(
        " decision={s} kind={s} instructions=",
        .{
            stringAttrValue(dictAttr(candidate, "decision")) orelse "unknown",
            stringAttrValue(dictAttr(candidate, "kind")) orelse "unknown",
        },
    );
    try writeInstructionArraySummary(dictAttr(candidate, "instructions"), writer);
    const pressure_attr = dictAttr(candidate, "pressure_delta");
    try writer.print(
        " bytes_saved={s} launch_reduction=",
        .{
            stringAttrValue(dictAttr(candidate, "bytes_saved")) orelse "unknown",
        },
    );
    try writeIntegerAttrText(dictAttr(candidate, "launch_count_reduction"), writer);
    try writer.writeAll(" pressure=split_kernels:");
    try writeIntegerAttrText(dictAttr(pressure_attr, "split_kernel_count"), writer);
    try writer.writeAll(",fused_kernels:");
    try writeIntegerAttrText(dictAttr(pressure_attr, "fused_kernel_count"), writer);
    try writer.print(
        ",split_peak:{s},fused_live:{s},additional_live:{s},global_saved:{s} reason={s}\n",
        .{
            stringAttrValue(dictAttr(pressure_attr, "split_peak_live_bytes")) orelse "unknown",
            stringAttrValue(dictAttr(pressure_attr, "fused_live_bytes")) orelse "unknown",
            stringAttrValue(dictAttr(pressure_attr, "additional_live_bytes")) orelse "unknown",
            stringAttrValue(dictAttr(pressure_attr, "global_bytes_saved")) orelse "unknown",
            stringAttrValue(dictAttr(candidate, "decision_reason")) orelse "unknown",
        },
    );
}

fn writePlacementPlanSummary(module_op: mlir.MlirOperation, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    const placements = getAttr(module_op, "pjrtx.placement.records");
    if (mlir.mlirAttributeIsNull(placements) or !mlir.mlirAttributeIsAArray(placements)) return;

    const count = mlir.mlirArrayAttrGetNumElements(placements);
    try writer.print("  placement_plan count={d}\n", .{count});
    var index: isize = 0;
    while (index < count) : (index += 1) {
        const placement = mlir.mlirArrayAttrGetElement(placements, index);
        try writer.writeAll("  placement.");
        try writeIntegerAttrText(dictAttr(placement, "index"), writer);
        try writer.writeAll(" instruction=");
        try writeIntegerAttrText(dictAttr(placement, "instruction"), writer);
        try writer.writeAll(" outputs=");
        try writeInstructionArraySummary(dictAttr(placement, "outputs"), writer);
        try writer.print(" layout={s} tile=", .{stringAttrValue(dictAttr(placement, "layout")) orelse "unknown"});
        try writeIntegerArraySummary(dictAttr(placement, "tile"), writer);
        try writer.writeAll(" result_memory=");
        try writeIntegerAttrText(dictAttr(placement, "result_memory"), writer);
        try writer.writeAll(" tile_memory=");
        const tile_memory = dictAttr(placement, "tile_memory");
        if (stringAttrEquals(tile_memory, "none")) {
            try writer.writeAll("none");
        } else {
            try writeIntegerAttrText(tile_memory, writer);
        }
        try writer.print(" reason={s}\n", .{stringAttrValue(dictAttr(placement, "reason")) orelse "unknown"});
    }
}

fn writeCollectivePlanSummary(module_op: mlir.MlirOperation, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    const collectives = getAttr(module_op, "pjrtx.collective.records");
    if (mlir.mlirAttributeIsNull(collectives) or !mlir.mlirAttributeIsAArray(collectives)) return;

    const count = mlir.mlirArrayAttrGetNumElements(collectives);
    try writer.print("  collective_plan count={d}\n", .{count});
    var index: isize = 0;
    while (index < count) : (index += 1) {
        const collective = mlir.mlirArrayAttrGetElement(collectives, index);
        try writer.writeAll("  collective.");
        try writeIntegerAttrText(dictAttr(collective, "index"), writer);
        try writer.print(
            " decision={s} algorithm={s} checked=",
            .{
                stringAttrValue(dictAttr(collective, "decision")) orelse "unknown",
                stringAttrValue(dictAttr(collective, "algorithm")) orelse "unknown",
            },
        );
        try writeIntegerAttrText(dictAttr(collective, "checked"), writer);
        try writer.writeAll(" lowered=");
        try writeIntegerAttrText(dictAttr(collective, "lowered"), writer);
        try writer.writeAll(" unsupported=");
        try writeIntegerAttrText(dictAttr(collective, "unsupported"), writer);
        try writer.print(
            " estimated_bytes={s} estimated_latency_ns={s} reason={s}\n",
            .{
                stringAttrValue(dictAttr(collective, "estimated_bytes")) orelse "unknown",
                stringAttrValue(dictAttr(collective, "estimated_latency_ns")) orelse "unknown",
                stringAttrValue(dictAttr(collective, "reason")) orelse "unknown",
            },
        );
    }
}

fn writeLoweringPlanSummary(module_op: mlir.MlirOperation, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    const lowerings = getAttr(module_op, "pjrtx.lowering.records");
    if (mlir.mlirAttributeIsNull(lowerings) or !mlir.mlirAttributeIsAArray(lowerings)) return;

    const count = mlir.mlirArrayAttrGetNumElements(lowerings);
    try writer.print("  lowering_plan count={d}\n", .{count});
    var index: isize = 0;
    while (index < count) : (index += 1) {
        const lowering = mlir.mlirArrayAttrGetElement(lowerings, index);
        try writer.writeAll("  lowering.");
        try writeIntegerAttrText(dictAttr(lowering, "id"), writer);
        try writer.print(
            " decision={s} instructions=",
            .{stringAttrValue(dictAttr(lowering, "decision")) orelse "unknown"},
        );
        try writeInstructionArraySummary(dictAttr(lowering, "instructions"), writer);
        try writer.writeAll(" costs=");
        try writeInstructionArraySummary(dictAttr(lowering, "costs"), writer);
        try writer.print(" reason={s}\n", .{stringAttrValue(dictAttr(lowering, "reason")) orelse "unknown"});
    }

    const region_facts = getAttr(module_op, "pjrtx.lowering.region_facts");
    if (mlir.mlirAttributeIsNull(region_facts) or !mlir.mlirAttributeIsAArray(region_facts)) return;
    const fact_count = mlir.mlirArrayAttrGetNumElements(region_facts);
    try writer.print("  lowering_region_facts count={d}\n", .{fact_count});
    index = 0;
    while (index < fact_count) : (index += 1) {
        const fact = mlir.mlirArrayAttrGetElement(region_facts, index);
        try writer.writeAll("  lowering_region.");
        try writeIntegerAttrText(dictAttr(fact, "lowering"), writer);
        try writer.writeAll(" fusion=");
        const fusion_group = dictAttr(fact, "fusion_group");
        if (stringAttrEquals(fusion_group, "none")) {
            try writer.writeAll("none");
        } else {
            try writeIntegerAttrText(fusion_group, writer);
        }
        try writer.writeAll(" placements=");
        try writeInstructionArraySummary(dictAttr(fact, "placements"), writer);
        try writer.writeAll(" tile=");
        try writeIntegerArraySummary(dictAttr(fact, "tile"), writer);
        try writer.writeAll(" result_memory=");
        try writeIntegerAttrText(dictAttr(fact, "result_memory"), writer);
        try writer.writeAll(" tile_memory=");
        const tile_memory = dictAttr(fact, "tile_memory");
        if (stringAttrEquals(tile_memory, "none")) {
            try writer.writeAll("none");
        } else {
            try writeIntegerAttrText(tile_memory, writer);
        }
        try writer.print(
            " codegen_region={s} reason={s}\n",
            .{
                stringAttrValue(dictAttr(fact, "codegen_region")) orelse "unknown",
                stringAttrValue(dictAttr(fact, "reason")) orelse "unknown",
            },
        );
    }
}

fn writePerformancePlanSummary(module_op: mlir.MlirOperation, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    const costs = getAttr(module_op, "pjrtx.performance.cost_ledger");
    if (!mlir.mlirAttributeIsNull(costs) and mlir.mlirAttributeIsAArray(costs)) {
        const cost_count = mlir.mlirArrayAttrGetNumElements(costs);
        try writer.print("  performance_cost_ledger count={d}\n", .{cost_count});
        var cost_index: isize = 0;
        while (cost_index < cost_count) : (cost_index += 1) {
            const cost = mlir.mlirArrayAttrGetElement(costs, cost_index);
            try writer.writeAll("  cost.");
            try writeIntegerAttrText(dictAttr(cost, "id"), writer);
            try writer.print(
                " class={s} dtype={s} instructions=",
                .{
                    stringAttrValue(dictAttr(cost, "op_class")) orelse "unknown",
                    stringAttrValue(dictAttr(cost, "dtype")) orelse "unknown",
                },
            );
            try writeInstructionArraySummary(dictAttr(cost, "instructions"), writer);
            try writer.print(
                " logical_ops={s} bytes_read={s} bytes_written={s} unit=",
                .{
                    stringAttrValue(dictAttr(cost, "logical_ops")) orelse "unknown",
                    stringAttrValue(dictAttr(cost, "bytes_read")) orelse "unknown",
                    stringAttrValue(dictAttr(cost, "bytes_written")) orelse "unknown",
                },
            );
            const expected_unit = dictAttr(cost, "expected_unit");
            if (stringAttrEquals(expected_unit, "none")) {
                try writer.writeAll("none");
            } else {
                try writeIntegerAttrText(expected_unit, writer);
            }
            try writer.print(" approximation={s}\n", .{stringAttrValue(dictAttr(cost, "approximation")) orelse "unknown"});
        }
    }

    const traffic = getAttr(module_op, "pjrtx.performance.memory_traffic");
    if (mlir.mlirAttributeIsNull(traffic) or !mlir.mlirAttributeIsAArray(traffic)) return;

    const traffic_count = mlir.mlirArrayAttrGetNumElements(traffic);
    try writer.print("  performance_memory_traffic count={d}\n", .{traffic_count});
    var traffic_index: isize = 0;
    while (traffic_index < traffic_count) : (traffic_index += 1) {
        const record = mlir.mlirArrayAttrGetElement(traffic, traffic_index);
        try writer.writeAll("  traffic.");
        try writeIntegerAttrText(dictAttr(record, "id"), writer);
        try writer.writeAll(" lowering=");
        try writeIntegerAttrText(dictAttr(record, "lowering"), writer);
        try writer.writeAll(" memory=");
        try writeIntegerAttrText(dictAttr(record, "memory"), writer);
        try writer.print(" kind={s} instructions=", .{stringAttrValue(dictAttr(record, "kind")) orelse "unknown"});
        try writeInstructionArraySummary(dictAttr(record, "instructions"), writer);
        try writer.print(
            " bytes_read={s} bytes_written={s} reason={s}\n",
            .{
                stringAttrValue(dictAttr(record, "bytes_read")) orelse "unknown",
                stringAttrValue(dictAttr(record, "bytes_written")) orelse "unknown",
                stringAttrValue(dictAttr(record, "reason")) orelse "unknown",
            },
        );
    }
}

fn writeKernelCodegenPlanSummary(module_op: mlir.MlirOperation, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    const codegen_records = getAttr(module_op, "pjrtx.codegen.records");
    if (mlir.mlirAttributeIsNull(codegen_records) or !mlir.mlirAttributeIsAArray(codegen_records)) return;

    const count = mlir.mlirArrayAttrGetNumElements(codegen_records);
    try writer.print("  codegen_plan count={d}\n", .{count});
    var index: isize = 0;
    while (index < count) : (index += 1) {
        const record = mlir.mlirArrayAttrGetElement(codegen_records, index);
        try writer.writeAll("  codegen.");
        try writeIntegerAttrText(dictAttr(record, "id"), writer);
        try writer.print(
            " kind={s} operation={s} lowering=",
            .{
                stringAttrValue(dictAttr(record, "kind")) orelse "unknown",
                stringAttrValue(dictAttr(record, "operation")) orelse "unknown",
            },
        );
        try writeIntegerAttrText(dictAttr(record, "lowering"), writer);
        try writer.writeAll(" command=");
        try writeIntegerAttrText(dictAttr(record, "command"), writer);
        try writer.print(" backend={s} tile=", .{stringAttrValue(dictAttr(record, "backend_kind")) orelse "unknown"});
        try writeIntegerArraySummary(dictAttr(record, "tile"), writer);
        try writer.writeAll(" result_memory=");
        try writeIntegerAttrText(dictAttr(record, "result_memory"), writer);
        try writer.writeAll(" tile_memory=");
        const tile_memory = dictAttr(record, "tile_memory");
        if (stringAttrEquals(tile_memory, "none")) {
            try writer.writeAll("none");
        } else {
            try writeIntegerAttrText(tile_memory, writer);
        }
        try writer.print(" reason={s}\n", .{stringAttrValue(dictAttr(record, "reason")) orelse "unknown"});
    }
}

fn writeSchedulePlanSummary(module_op: mlir.MlirOperation, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    const commands = getAttr(module_op, "pjrtx.schedule.commands");
    if (mlir.mlirAttributeIsNull(commands) or !mlir.mlirAttributeIsAArray(commands)) return;

    const command_count = mlir.mlirArrayAttrGetNumElements(commands);
    const overlaps = getAttr(module_op, "pjrtx.schedule.overlaps");
    const overlap_count = if (!mlir.mlirAttributeIsNull(overlaps) and mlir.mlirAttributeIsAArray(overlaps))
        mlir.mlirArrayAttrGetNumElements(overlaps)
    else
        0;
    try writer.print("  schedule_plan commands={d} overlaps={d}\n", .{ command_count, overlap_count });
    var index: isize = 0;
    while (index < command_count) : (index += 1) {
        const command = mlir.mlirArrayAttrGetElement(commands, index);
        try writer.writeAll("  command.");
        try writeIntegerAttrText(dictAttr(command, "id"), writer);
        try writer.print(
            " kind={s} stream=",
            .{stringAttrValue(dictAttr(command, "kind")) orelse "unknown"},
        );
        try writeIntegerAttrText(dictAttr(command, "stream"), writer);
        try writer.writeAll(" inputs=");
        try writeInstructionArraySummary(dictAttr(command, "inputs"), writer);
        try writer.writeAll(" outputs=");
        try writeInstructionArraySummary(dictAttr(command, "outputs"), writer);
        try writer.writeByte('\n');
    }
}

fn writeBackendBindingSummary(module_op: mlir.MlirOperation, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    const bindings = getAttr(module_op, "pjrtx.backend.bindings");
    if (mlir.mlirAttributeIsNull(bindings) or !mlir.mlirAttributeIsAArray(bindings)) return;

    const count = mlir.mlirArrayAttrGetNumElements(bindings);
    try writer.print("  backend_binding count={d}\n", .{count});
    var index: isize = 0;
    while (index < count) : (index += 1) {
        const binding = mlir.mlirArrayAttrGetElement(bindings, index);
        try writer.writeAll("  binding.");
        try writeIntegerAttrText(dictAttr(binding, "id"), writer);
        try writer.writeAll(" command=");
        try writeIntegerAttrText(dictAttr(binding, "command"), writer);
        try writer.print(
            " backend={s} operation={s} instructions=",
            .{
                stringAttrValue(dictAttr(binding, "backend_kind")) orelse "unknown",
                stringAttrValue(dictAttr(binding, "operation")) orelse "unknown",
            },
        );
        try writeInstructionArraySummary(dictAttr(binding, "instructions"), writer);
        try writer.writeByte('\n');
    }
}

fn writeExecutableContractSummary(module_op: mlir.MlirOperation, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    const contract = getAttr(module_op, "pjrtx.executable.contract");
    if (mlir.mlirAttributeIsNull(contract) or !mlir.mlirAttributeIsADictionary(contract)) return;

    try writer.print(
        "  executable_contract target={s} schedule_commands=",
        .{stringAttrValue(dictAttr(contract, "target_kind")) orelse "unknown"},
    );
    try writeIntegerAttrText(dictAttr(contract, "schedule_commands"), writer);
    try writer.writeAll(" backend_bindings=");
    try writeIntegerAttrText(dictAttr(contract, "backend_bindings"), writer);
    try writer.writeAll(" kernel_codegen_records=");
    try writeIntegerAttrText(dictAttr(contract, "kernel_codegen_records"), writer);
    try writer.writeByte('\n');
}

fn writeBackendExecutablePlanSummary(module_op: mlir.MlirOperation, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    const plan = getAttr(module_op, "pjrtx.backend.executable");
    if (mlir.mlirAttributeIsNull(plan) or !mlir.mlirAttributeIsADictionary(plan)) return;

    const calls = dictAttr(plan, "calls");
    const call_count = if (!mlir.mlirAttributeIsNull(calls) and mlir.mlirAttributeIsAArray(calls))
        mlir.mlirArrayAttrGetNumElements(calls)
    else
        0;
    try writer.print(
        "  backend_executable backend={s} command=",
        .{stringAttrValue(dictAttr(plan, "backend_kind")) orelse "unknown"},
    );
    try writeIntegerAttrText(dictAttr(plan, "command"), writer);
    try writer.print(" operation={s} calls={d}\n", .{
        stringAttrValue(dictAttr(plan, "operation")) orelse "unknown",
        call_count,
    });
    var index: isize = 0;
    while (index < call_count) : (index += 1) {
        const call = mlir.mlirArrayAttrGetElement(calls, index);
        try writer.writeAll("  executable_call.");
        try writeIntegerAttrText(dictAttr(call, "index"), writer);
        try writer.writeAll(" instruction=");
        try writeIntegerAttrText(dictAttr(call, "instruction"), writer);
        try writer.print(" feature={s} operation={s} instructions=", .{
            stringAttrValue(dictAttr(call, "feature")) orelse "unknown",
            stringAttrValue(dictAttr(call, "operation")) orelse "unknown",
        });
        try writeInstructionArraySummary(dictAttr(call, "instructions"), writer);
        try writer.writeByte('\n');
    }
}

fn writeBackendKernelGraphSummary(module_op: mlir.MlirOperation, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    const graph = getAttr(module_op, "pjrtx.backend.kernel_graph");
    if (mlir.mlirAttributeIsNull(graph) or !mlir.mlirAttributeIsADictionary(graph)) return;

    const nodes = dictAttr(graph, "nodes");
    const edges = dictAttr(graph, "edges");
    const node_count = if (!mlir.mlirAttributeIsNull(nodes) and mlir.mlirAttributeIsAArray(nodes))
        mlir.mlirArrayAttrGetNumElements(nodes)
    else
        0;
    const edge_count = if (!mlir.mlirAttributeIsNull(edges) and mlir.mlirAttributeIsAArray(edges))
        mlir.mlirArrayAttrGetNumElements(edges)
    else
        0;
    try writer.print(
        "  backend_kernel_graph backend={s} command=",
        .{stringAttrValue(dictAttr(graph, "backend_kind")) orelse "unknown"},
    );
    try writeIntegerAttrText(dictAttr(graph, "command"), writer);
    try writer.print(" nodes={d} edges={d}\n", .{ node_count, edge_count });

    var node_index: isize = 0;
    while (node_index < node_count) : (node_index += 1) {
        const node = mlir.mlirArrayAttrGetElement(nodes, node_index);
        try writer.writeAll("  kernel_node.");
        try writeIntegerAttrText(dictAttr(node, "index"), writer);
        try writer.writeAll(" call=");
        try writeIntegerAttrText(dictAttr(node, "call"), writer);
        try writer.writeAll(" instruction=");
        try writeIntegerAttrText(dictAttr(node, "instruction"), writer);
        const output_type = dictAttr(node, "output_type");
        try writer.print(" feature={s} operation={s} dtype={s} dims=", .{
            stringAttrValue(dictAttr(node, "feature")) orelse "unknown",
            stringAttrValue(dictAttr(node, "operation")) orelse "unknown",
            stringAttrValue(dictAttr(output_type, "dtype")) orelse "unknown",
        });
        try writeInstructionArraySummary(dictAttr(output_type, "dims"), writer);
        try writer.print(" attrs={s}\n", .{stringAttrValue(dictAttr(node, "attributes")) orelse "unknown"});
    }

    var edge_index: isize = 0;
    while (edge_index < edge_count) : (edge_index += 1) {
        const edge = mlir.mlirArrayAttrGetElement(edges, edge_index);
        try writer.writeAll("  kernel_edge.");
        try writer.print("{d} value=", .{edge_index});
        try writeIntegerAttrText(dictAttr(edge, "value"), writer);
        try writer.writeAll(" src=");
        try writeIntegerAttrText(dictAttr(edge, "src"), writer);
        try writer.writeAll(" dst=");
        try writeIntegerAttrText(dictAttr(edge, "dst"), writer);
        try writer.writeByte('\n');
    }
}

fn writeRuntimeAllocationSummary(module_op: mlir.MlirOperation, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    const plan = getAttr(module_op, "pjrtx.runtime.allocation");
    if (mlir.mlirAttributeIsNull(plan) or !mlir.mlirAttributeIsADictionary(plan)) return;

    const allocations = dictAttr(plan, "allocations");
    const uses = dictAttr(plan, "command_buffer_uses");
    const allocation_count = if (!mlir.mlirAttributeIsNull(allocations) and mlir.mlirAttributeIsAArray(allocations))
        mlir.mlirArrayAttrGetNumElements(allocations)
    else
        0;
    const use_count = if (!mlir.mlirAttributeIsNull(uses) and mlir.mlirAttributeIsAArray(uses))
        mlir.mlirArrayAttrGetNumElements(uses)
    else
        0;

    try writer.print("  runtime_allocation allocations={d} uses={d} peak_device_bytes={s}\n", .{
        allocation_count,
        use_count,
        stringAttrValue(dictAttr(plan, "peak_device_bytes")) orelse "unknown",
    });

    var allocation_index: isize = 0;
    while (allocation_index < allocation_count) : (allocation_index += 1) {
        const allocation = mlir.mlirArrayAttrGetElement(allocations, allocation_index);
        try writer.writeAll("  allocation.");
        try writeIntegerAttrText(dictAttr(allocation, "index"), writer);
        try writer.writeAll(" value=");
        try writeIntegerAttrText(dictAttr(allocation, "value"), writer);
        try writer.print(" placement={s} memory=", .{stringAttrValue(dictAttr(allocation, "placement")) orelse "unknown"});
        try writeIntegerAttrText(dictAttr(allocation, "memory"), writer);
        try writer.print(" bytes={s} lifetime=command.", .{stringAttrValue(dictAttr(allocation, "bytes")) orelse "unknown"});
        try writeIntegerAttrText(dictAttr(allocation, "first_command"), writer);
        try writer.writeAll("..command.");
        try writeIntegerAttrText(dictAttr(allocation, "last_command"), writer);
        try writer.writeByte('\n');
    }
}

fn writeRuntimeStreamSummary(module_op: mlir.MlirOperation, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    const steps = getAttr(module_op, "pjrtx.runtime.streams");
    if (mlir.mlirAttributeIsNull(steps) or !mlir.mlirAttributeIsAArray(steps)) return;

    const step_count = mlir.mlirArrayAttrGetNumElements(steps);
    try writer.print("  runtime_stream steps={d}\n", .{step_count});
    var index: isize = 0;
    while (index < step_count) : (index += 1) {
        const step = mlir.mlirArrayAttrGetElement(steps, index);
        try writer.writeAll("  stream_step command=");
        try writeIntegerAttrText(dictAttr(step, "command"), writer);
        try writer.writeAll(" stream=");
        try writeIntegerAttrText(dictAttr(step, "stream"), writer);
        try writer.writeAll(" start=event.");
        try writeIntegerAttrText(dictAttr(step, "start_event"), writer);
        try writer.writeAll(" done=event.");
        try writeIntegerAttrText(dictAttr(step, "done_event"), writer);
        try writer.writeAll(" waits=");
        try writeInstructionArraySummary(dictAttr(step, "wait_events"), writer);
        try writer.writeByte('\n');
    }
}

fn writeRuntimeProfileSummary(module_op: mlir.MlirOperation, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    const events = getAttr(module_op, "pjrtx.runtime.profile_events");
    if (mlir.mlirAttributeIsNull(events) or !mlir.mlirAttributeIsAArray(events)) return;

    const event_count = mlir.mlirArrayAttrGetNumElements(events);
    try writer.print("  runtime_profile events={d}\n", .{event_count});
    var index: isize = 0;
    while (index < event_count) : (index += 1) {
        const event = mlir.mlirArrayAttrGetElement(events, index);
        try writer.writeAll("  profile.");
        try writeIntegerAttrText(dictAttr(event, "index"), writer);
        try writer.print(" kind={s} command=", .{stringAttrValue(dictAttr(event, "kind")) orelse "unknown"});
        try writeIntegerAttrText(dictAttr(event, "command"), writer);
        try writer.writeAll(" instructions=");
        try writeInstructionArraySummary(dictAttr(event, "instructions"), writer);
        try writer.print(" bytes={s} logical_ops={s} status={s} forced_sync=", .{
            stringAttrValue(dictAttr(event, "bytes")) orelse "unknown",
            stringAttrValue(dictAttr(event, "logical_ops")) orelse "unknown",
            stringAttrValue(dictAttr(event, "status")) orelse "unknown",
        });
        try writer.print("{any}\n", .{boolAttrValue(dictAttr(event, "forced_sync")) orelse false});
    }
}

fn writeRuntimeProfileJoinSummary(module_op: mlir.MlirOperation, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    const joins = getAttr(module_op, "pjrtx.runtime.profile_joins");
    if (mlir.mlirAttributeIsNull(joins) or !mlir.mlirAttributeIsAArray(joins)) return;

    const join_count = mlir.mlirArrayAttrGetNumElements(joins);
    try writer.print("  runtime_profile_join joins={d}\n", .{join_count});
    var index: isize = 0;
    while (index < join_count) : (index += 1) {
        const join = mlir.mlirArrayAttrGetElement(joins, index);
        try writer.writeAll("  profile_join.");
        try writeIntegerAttrText(dictAttr(join, "index"), writer);
        try writer.print(" subject={s}.", .{stringAttrValue(dictAttr(join, "subject_kind")) orelse "unknown"});
        try writeIntegerAttrText(dictAttr(join, "subject_id"), writer);
        try writer.writeAll(" command=");
        try writeIntegerAttrText(dictAttr(join, "command"), writer);
        try writer.writeAll(" events=");
        try writeInstructionArraySummary(dictAttr(join, "events"), writer);
        try writer.writeAll(" instructions=");
        try writeInstructionArraySummary(dictAttr(join, "instructions"), writer);
        try writer.writeByte('\n');
    }
}

fn writeBackendProfileJoinSummary(module_op: mlir.MlirOperation, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    const joins = getAttr(module_op, "pjrtx.backend.profile_joins");
    if (mlir.mlirAttributeIsNull(joins) or !mlir.mlirAttributeIsAArray(joins)) return;

    const join_count = mlir.mlirArrayAttrGetNumElements(joins);
    try writer.print("  backend_profile_join joins={d}\n", .{join_count});
    var index: isize = 0;
    while (index < join_count) : (index += 1) {
        const join = mlir.mlirArrayAttrGetElement(joins, index);
        try writer.writeAll("  backend_profile_join.");
        try writeIntegerAttrText(dictAttr(join, "index"), writer);
        try writer.writeAll(" call=");
        try writeIntegerAttrText(dictAttr(join, "call"), writer);
        try writer.writeAll(" command=");
        try writeIntegerAttrText(dictAttr(join, "command"), writer);
        try writer.writeAll(" event=");
        try writeIntegerAttrText(dictAttr(join, "event"), writer);
        try writer.writeAll(" instructions=");
        try writeInstructionArraySummary(dictAttr(join, "instructions"), writer);
        try writer.writeByte('\n');
    }
}

fn targetU32ArrayAttr(context: mlir.MlirContext, values: []const u32) mlir.MlirAttribute {
    var attrs: [limits.max_target_refs]mlir.MlirAttribute = undefined;
    for (values, 0..) |value, index| attrs[index] = integerAttr(context, value);
    return mlir.mlirArrayAttrGet(context, std.math.cast(isize, values.len) orelse unreachable, &attrs);
}

fn graphValueIdArrayAttr(context: mlir.MlirContext, ids: []const compiler_facts.GraphValueId) mlir.MlirAttribute {
    var attrs: [limits.max_placement_outputs]mlir.MlirAttribute = undefined;
    for (ids, 0..) |id, index| {
        attrs[index] = integerAttr(context, id.index);
    }
    return mlir.mlirArrayAttrGet(context, std.math.cast(isize, ids.len) orelse unreachable, &attrs);
}

fn graphValueIdArrayAttrBounded(context: mlir.MlirContext, ids: []const compiler_facts.GraphValueId) mlir.MlirAttribute {
    var attrs: [limits.max_kernel_ref_ids]mlir.MlirAttribute = undefined;
    for (ids, 0..) |id, index| attrs[index] = integerAttr(context, id.index);
    return mlir.mlirArrayAttrGet(context, std.math.cast(isize, ids.len) orelse unreachable, &attrs);
}

fn graphInstructionIdArrayAttr(context: mlir.MlirContext, ids: []const compiler_facts.GraphInstructionId) mlir.MlirAttribute {
    var attrs: [limits.max_kernel_ref_ids]mlir.MlirAttribute = undefined;
    for (ids, 0..) |id, index| attrs[index] = integerAttr(context, id.index);
    return mlir.mlirArrayAttrGet(context, std.math.cast(isize, ids.len) orelse unreachable, &attrs);
}

fn profileEventIdArrayAttr(context: mlir.MlirContext, ids: []const core.ProfileEventId) mlir.MlirAttribute {
    var attrs: [limits.max_runtime_profile_join_events]mlir.MlirAttribute = undefined;
    for (ids, 0..) |id, index| attrs[index] = integerAttr(context, id.index);
    return mlir.mlirArrayAttrGet(context, std.math.cast(isize, ids.len) orelse unreachable, &attrs);
}

fn costLedgerIdArrayAttr(context: mlir.MlirContext, ids: []const compiler_facts.CostLedgerId) mlir.MlirAttribute {
    var attrs: [limits.max_kernel_ref_ids]mlir.MlirAttribute = undefined;
    for (ids, 0..) |id, index| attrs[index] = integerAttr(context, id.index);
    return mlir.mlirArrayAttrGet(context, std.math.cast(isize, ids.len) orelse unreachable, &attrs);
}

fn memoryTrafficIdArrayAttr(context: mlir.MlirContext, ids: []const compiler_facts.MemoryTrafficId) mlir.MlirAttribute {
    var attrs: [limits.max_kernel_ref_ids]mlir.MlirAttribute = undefined;
    for (ids, 0..) |id, index| attrs[index] = integerAttr(context, id.index);
    return mlir.mlirArrayAttrGet(context, std.math.cast(isize, ids.len) orelse unreachable, &attrs);
}

fn graphValueIdArrayAttrSchedule(context: mlir.MlirContext, ids: []const compiler_facts.GraphValueId) mlir.MlirAttribute {
    var attrs: [limits.max_schedule_refs]mlir.MlirAttribute = undefined;
    for (ids, 0..) |id, index| attrs[index] = integerAttr(context, id.index);
    return mlir.mlirArrayAttrGet(context, std.math.cast(isize, ids.len) orelse unreachable, &attrs);
}

fn loweringRecordIdArrayAttrSchedule(context: mlir.MlirContext, ids: []const compiler_facts.LoweringRecordId) mlir.MlirAttribute {
    var attrs: [limits.max_schedule_refs]mlir.MlirAttribute = undefined;
    for (ids, 0..) |id, index| attrs[index] = integerAttr(context, id.index);
    return mlir.mlirArrayAttrGet(context, std.math.cast(isize, ids.len) orelse unreachable, &attrs);
}

fn costLedgerIdArrayAttrSchedule(context: mlir.MlirContext, ids: []const compiler_facts.CostLedgerId) mlir.MlirAttribute {
    var attrs: [limits.max_schedule_refs]mlir.MlirAttribute = undefined;
    for (ids, 0..) |id, index| attrs[index] = integerAttr(context, id.index);
    return mlir.mlirArrayAttrGet(context, std.math.cast(isize, ids.len) orelse unreachable, &attrs);
}

fn commandDependencyArrayAttr(context: mlir.MlirContext, dependencies: []const core.CommandDependency) mlir.MlirAttribute {
    var attrs: [limits.max_schedule_dependencies]mlir.MlirAttribute = undefined;
    for (dependencies, 0..) |dependency, index| {
        const fields = [_]mlir.MlirNamedAttribute{
            namedAttr(context, "command", integerAttr(context, dependency.command_id.index)),
            namedAttr(context, "kind", stringAttr(context, @tagName(dependency.kind))),
        };
        attrs[index] = mlir.mlirDictionaryAttrGet(context, fields.len, &fields);
    }
    return mlir.mlirArrayAttrGet(context, std.math.cast(isize, dependencies.len) orelse unreachable, &attrs);
}

fn i64ArrayAttr(context: mlir.MlirContext, values: []const i64) mlir.MlirAttribute {
    var attrs: [limits.max_tile_rank]mlir.MlirAttribute = undefined;
    for (values, 0..) |value, index| {
        attrs[index] = i64Attr(context, value);
    }
    return mlir.mlirArrayAttrGet(context, std.math.cast(isize, values.len) orelse unreachable, &attrs);
}

fn u32ArrayAttr(context: mlir.MlirContext, values: []const u32) mlir.MlirAttribute {
    var attrs: [limits.max_kernel_ref_ids]mlir.MlirAttribute = undefined;
    for (values, 0..) |value, index| {
        attrs[index] = integerAttr(context, value);
    }
    return mlir.mlirArrayAttrGet(context, std.math.cast(isize, values.len) orelse unreachable, &attrs);
}

fn kernelShapeAttr(context: mlir.MlirContext, shape: core.KernelCodegenShape) mlir.MlirAttribute {
    const attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(context, "operations", integerAttr(context, shape.operation_count)),
        namedAttr(context, "external_inputs", integerAttr(context, shape.external_input_count)),
        namedAttr(context, "external_outputs", integerAttr(context, shape.external_output_count)),
        namedAttr(context, "intermediates", integerAttr(context, shape.intermediate_value_count)),
    };
    return mlir.mlirDictionaryAttrGet(context, attrs.len, &attrs);
}

fn kernelPressureAttr(context: mlir.MlirContext, pressure: core.KernelMemoryPressure) mlir.MlirAttribute {
    const attrs = [_]mlir.MlirNamedAttribute{
        namedAttr(context, "global_read", u128StackStringAttr(context, pressure.global_bytes_read)),
        namedAttr(context, "global_write", u128StackStringAttr(context, pressure.global_bytes_written)),
        namedAttr(context, "local_read", u128StackStringAttr(context, pressure.local_bytes_read)),
        namedAttr(context, "local_write", u128StackStringAttr(context, pressure.local_bytes_written)),
    };
    return mlir.mlirDictionaryAttrGet(context, attrs.len, &attrs);
}

fn u128StackStringAttr(context: mlir.MlirContext, value: u128) mlir.MlirAttribute {
    var buffer: [39]u8 = undefined;
    const text = std.fmt.bufPrint(&buffer, "{d}", .{value}) catch unreachable;
    return stringAttr(context, text);
}

fn u64StackStringAttr(context: mlir.MlirContext, value: u64) mlir.MlirAttribute {
    var buffer: [20]u8 = undefined;
    const text = std.fmt.bufPrint(&buffer, "{d}", .{value}) catch unreachable;
    return stringAttr(context, text);
}

fn dictAttr(attr: mlir.MlirAttribute, name: []const u8) mlir.MlirAttribute {
    return mlir.mlirDictionaryAttrGetElementByName(attr, mlirStringRef(name));
}

fn parseU128(text: []const u8, diagnostics: *std.Io.Writer) !u128 {
    return std.fmt.parseUnsigned(u128, text, 10) catch {
        try diagnostics.writeAll("pass=fusion_extract feature=fusion reason=invalid unsigned integer\n");
        return MlirStateError.InvalidFusionPlan;
    };
}

fn targetFingerprintText(allocator: std.mem.Allocator, target: target_pkg.TargetDescription) ![]u8 {
    var hasher = std.hash.Wyhash.init(0);
    hasher.update(target.name);
    hasher.update(@tagName(target.kind));
    for (target.memory_spaces) |memory_space| {
        hasher.update(memory_space.name);
        hasher.update(@tagName(memory_space.kind));
    }
    for (target.execution_units) |unit| {
        hasher.update(unit.name);
        hasher.update(@tagName(unit.kind));
    }
    return std.fmt.allocPrint(allocator, "{x}", .{hasher.final()});
}

fn parseModuleState(text: []const u8) ?ModuleState {
    inline for (std.meta.fields(ModuleState)) |field| {
        if (std.mem.eql(u8, text, field.name)) {
            return @enumFromInt(field.value);
        }
    }
    return null;
}

fn setStringAttr(context: mlir.MlirContext, op: mlir.MlirOperation, name: []const u8, value: []const u8) void {
    mlir.mlirOperationSetAttributeByName(
        op,
        mlirStringRef(name),
        stringAttr(context, value),
    );
}

fn setIntegerAttr(context: mlir.MlirContext, op: mlir.MlirOperation, name: []const u8, value: u32) void {
    mlir.mlirOperationSetAttributeByName(
        op,
        mlirStringRef(name),
        integerAttr(context, value),
    );
}

fn namedAttr(context: mlir.MlirContext, name: []const u8, attr: mlir.MlirAttribute) mlir.MlirNamedAttribute {
    return mlir.mlirNamedAttributeGet(mlir.mlirIdentifierGet(context, mlirStringRef(name)), attr);
}

fn stringAttr(context: mlir.MlirContext, value: []const u8) mlir.MlirAttribute {
    return mlir.mlirStringAttrGet(context, mlirStringRef(value));
}

fn u128StringAttr(context: mlir.MlirContext, value: u128) mlir.MlirAttribute {
    var buffer: [40]u8 = undefined;
    const text = std.fmt.bufPrint(&buffer, "{d}", .{value}) catch unreachable;
    return stringAttr(context, text);
}

fn boolAttr(context: mlir.MlirContext, value: bool) mlir.MlirAttribute {
    const int_value: c_int = if (value) 1 else 0;
    return mlir.mlirBoolAttrGet(context, int_value);
}

fn integerAttr(context: mlir.MlirContext, value: u32) mlir.MlirAttribute {
    return mlir.mlirIntegerAttrGet(mlir.mlirIntegerTypeGet(context, 32), value);
}

fn signedIntegerAttr(context: mlir.MlirContext, value: i32) mlir.MlirAttribute {
    return mlir.mlirIntegerAttrGet(mlir.mlirIntegerTypeGet(context, 32), value);
}

fn i64Attr(context: mlir.MlirContext, value: i64) mlir.MlirAttribute {
    return mlir.mlirIntegerAttrGet(mlir.mlirIntegerTypeGet(context, 64), value);
}

fn optionalIntegerAttr(context: mlir.MlirContext, value: ?u32) mlir.MlirAttribute {
    if (value) |integer| return integerAttr(context, integer);
    return stringAttr(context, "none");
}

fn optionalU64StringAttr(context: mlir.MlirContext, value: ?u64) mlir.MlirAttribute {
    const integer = value orelse return stringAttr(context, "none");
    var buffer: [20]u8 = undefined;
    const text = std.fmt.bufPrint(&buffer, "{d}", .{integer}) catch unreachable;
    return stringAttr(context, text);
}

fn optionalF64StringAttr(context: mlir.MlirContext, value: ?f64) mlir.MlirAttribute {
    const float = value orelse return stringAttr(context, "none");
    var buffer: [32]u8 = undefined;
    const text = std.fmt.bufPrint(&buffer, "{d}", .{float}) catch unreachable;
    return stringAttr(context, text);
}

fn getAttr(op: mlir.MlirOperation, name: []const u8) mlir.MlirAttribute {
    return mlir.mlirOperationGetAttributeByName(op, mlirStringRef(name));
}

fn insertDialect(registry: mlir.MlirDialectRegistry, handle: mlir.MlirDialectHandle) void {
    mlir.mlirDialectHandleInsertDialect(handle, registry);
}

fn loadDialect(context: mlir.MlirContext, handle: mlir.MlirDialectHandle, diagnostics: *std.Io.Writer) !void {
    const dialect = mlir.mlirDialectHandleLoadDialect(handle, context);
    if (mlir.mlirDialectIsNull(dialect)) {
        try diagnostics.writeAll("pass=mlir_state_init feature=mlir-dialect reason=failed to load MLIR dialect\n");
        return MlirStateError.InvalidStablehlo;
    }
}

fn mlirStringRef(text: []const u8) mlir.MlirStringRef {
    return mlir.mlirStringRefCreate(text.ptr, text.len);
}

fn mlirStringSlice(text: mlir.MlirStringRef) []const u8 {
    if (text.length == 0) return &.{};
    return text.data[0..text.length];
}

test "MLIR session attaches target, fusion facts, and extracts FusionGroup" {
    const program =
        \\module {
        \\  func.func @main(%arg0: tensor<2x4xf32>, %arg1: tensor<4x3xf32>, %arg2: tensor<3xf32>) -> tensor<2x3xf32> {
        \\    %0 = stablehlo.dot_general %arg0, %arg1,
        \\      contracting_dims = [1] x [0],
        \\      precision = [DEFAULT, DEFAULT] : (tensor<2x4xf32>, tensor<4x3xf32>) -> tensor<2x3xf32>
        \\    %1 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<3xf32>) -> tensor<2x3xf32>
        \\    %2 = stablehlo.add %0, %1 : tensor<2x3xf32>
        \\    %3 = stablehlo.tanh %2 : tensor<2x3xf32>
        \\    return %3 : tensor<2x3xf32>
        \\  }
        \\}
    ;
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    var session: MlirSession = try .initFromStablehloText(std.testing.allocator, program, .{ .program_name = "mlir_state_test" }, &diagnostics.writer);
    defer session.deinit();

    try std.testing.expectEqual(ModuleState.imported, moduleState(&session).?);
    try attachTarget(&session, testTarget(), 1, 1, &diagnostics.writer);
    try std.testing.expectEqual(ModuleState.target_attached, moduleState(&session).?);
    try markTargetLegal(&session, &diagnostics.writer);
    const target_legal_pass = getAttr(session.moduleOperation(), "pjrtx.target_legal.pass");
    try std.testing.expectEqualStrings("pjrtx-target-legal-external", stringAttrValue(target_legal_pass).?);
    try runFusionCandidateDiscoveryExternalPass(&session, &diagnostics.writer);
    const expected_candidate_count: u32 = 1;
    try std.testing.expectEqual(expected_candidate_count, integerAttrValue(getAttr(session.moduleOperation(), "pjrtx.fusion.candidates.matmul_epilogue")).?);
    try std.testing.expectEqual(expected_candidate_count, integerAttrValue(getAttr(session.moduleOperation(), "pjrtx.fusion.candidates.elementwise_chain")).?);
    try std.testing.expectEqualStrings("pjrtx-fusion-candidate-external", stringAttrValue(getAttr(session.moduleOperation(), "pjrtx.fusion_candidate.pass")).?);
    const candidate_attr = getAttr(session.moduleOperation(), "pjrtx.fusion.candidates");
    try std.testing.expect(mlir.mlirAttributeIsAArray(candidate_attr));
    const expected_structured_candidate_count: isize = 2;
    try std.testing.expectEqual(expected_structured_candidate_count, mlir.mlirArrayAttrGetNumElements(candidate_attr));

    try planFusionFromCandidates(&session, &diagnostics.writer);
    const fusion_plan_pass = getAttr(session.moduleOperation(), "pjrtx.fusion_plan.pass");
    try std.testing.expectEqualStrings("pjrtx-fusion-plan-external", stringAttrValue(fusion_plan_pass).?);
    const decided_candidates = getAttr(session.moduleOperation(), "pjrtx.fusion.candidates");
    const decided_candidate_0 = mlir.mlirArrayAttrGetElement(decided_candidates, 0);
    const decided_candidate_1 = mlir.mlirArrayAttrGetElement(decided_candidates, 1);
    try std.testing.expectEqualStrings("rejected", stringAttrValue(dictAttr(decided_candidate_0, "decision")).?);
    try std.testing.expectEqualStrings("accepted", stringAttrValue(dictAttr(decided_candidate_1, "decision")).?);
    const fusion_attr = getAttr(session.moduleOperation(), "pjrtx.fusion.plan");
    try std.testing.expect(mlir.mlirAttributeIsNull(fusion_attr));

    const extracted = try extractFusionGroups(std.testing.allocator, &session, &diagnostics.writer);
    defer deinitExtractedFusionGroups(std.testing.allocator, extracted);

    const expected_group_count: usize = 2;
    try std.testing.expectEqual(expected_group_count, extracted.len);
    try std.testing.expectEqual(compiler_facts.FusionDecision.rejected, extracted[0].decision);
    try std.testing.expectEqualStrings("matmul_epilogue", extracted[0].kind);
    try std.testing.expectEqual(compiler_facts.FusionDecision.accepted, extracted[1].decision);
    try std.testing.expectEqualStrings("elementwise_chain", extracted[1].kind);
    const expected_additional_live_bytes: u128 = 80;
    try std.testing.expectEqual(expected_additional_live_bytes, extracted[0].pressure_delta.additional_live_bytes);

    const placement_outputs = [_]compiler_facts.GraphValueId{.{ .index = 3 }};
    const placement_tile = [_]i64{ 2, 3 };
    const placements = [_]compiler_facts.PlacementRecord{.{
        .index = 0,
        .graph_instruction_id = .{ .index = 0 },
        .output_value_ids = &placement_outputs,
        .layout = .dense_row_major,
        .logical_tile_shape = &placement_tile,
        .result_memory_space_id = 0,
        .tile_memory_space_id = null,
        .reason = "unit test placement fact is committed to MLIR before extraction",
    }};
    try commitPlacementPlan(&session, &placements, &diagnostics.writer);
    try std.testing.expectEqual(ModuleState.placement_planned, moduleState(&session).?);
    const placement_plan_pass = getAttr(session.moduleOperation(), "pjrtx.placement_plan.pass");
    try std.testing.expectEqualStrings("pjrtx-placement-plan-external", stringAttrValue(placement_plan_pass).?);

    const extracted_placements = try extractPlacementRecords(std.testing.allocator, &session, &diagnostics.writer);
    defer deinitExtractedPlacementRecords(std.testing.allocator, extracted_placements);
    const expected_placement_count: usize = 1;
    try std.testing.expectEqual(expected_placement_count, extracted_placements.len);
    try std.testing.expectEqual(compiler_facts.GraphInstructionId{ .index = 0 }, extracted_placements[0].graph_instruction_id);
    try std.testing.expectEqualSlices(i64, &placement_tile, extracted_placements[0].logical_tile_shape);
    const expected_tile_memory: ?u32 = null;
    try std.testing.expectEqual(expected_tile_memory, extracted_placements[0].tile_memory_space_id);
    try std.testing.expectEqualStrings("unit test placement fact is committed to MLIR before extraction", extracted_placements[0].reason);

    const collective_records = [_]compiler_facts.CollectivePlanRecord{.{
        .index = 0,
        .decision = .no_collectives,
        .algorithm = .none,
        .checked_graph_instruction_count = 5,
        .lowered_collective_count = 0,
        .unsupported_collective_count = 0,
        .estimated_bytes = 0,
        .estimated_latency_ns = null,
        .reason = "unit test collective fact is committed to MLIR before extraction",
    }};
    try commitCollectivePlan(&session, &collective_records, &diagnostics.writer);
    try std.testing.expectEqual(ModuleState.collectives_planned, moduleState(&session).?);
    const collective_plan_pass = getAttr(session.moduleOperation(), "pjrtx.collective_plan.pass");
    try std.testing.expectEqualStrings("pjrtx-collective-plan-external", stringAttrValue(collective_plan_pass).?);

    const extracted_collectives = try extractCollectivePlanRecords(std.testing.allocator, &session, &diagnostics.writer);
    defer deinitExtractedCollectivePlanRecords(std.testing.allocator, extracted_collectives);
    const expected_collective_count: usize = 1;
    try std.testing.expectEqual(expected_collective_count, extracted_collectives.len);
    try std.testing.expectEqual(compiler_facts.CollectivePlanDecision.no_collectives, extracted_collectives[0].decision);
    try std.testing.expectEqual(compiler_facts.CollectiveAlgorithm.none, extracted_collectives[0].algorithm);
    try std.testing.expectEqualStrings("unit test collective fact is committed to MLIR before extraction", extracted_collectives[0].reason);

    const codegen_tile = [_]i64{ 2, 3 };
    const codegen_inputs = [_]compiler_facts.GraphValueId{ .{ .index = 0 }, .{ .index = 1 } };
    const codegen_outputs = [_]compiler_facts.GraphValueId{.{ .index = 3 }};
    const codegen_intermediates = [_]compiler_facts.GraphValueId{};
    const codegen_instructions = [_]compiler_facts.GraphInstructionId{.{ .index = 0 }};
    const codegen_costs = [_]compiler_facts.CostLedgerId{.{ .index = 0 }};
    const codegen_traffic = [_]compiler_facts.MemoryTrafficId{ .{ .index = 0 }, .{ .index = 1 } };
    const performance_costs = [_]compiler_facts.CostLedgerEntry{.{
        .id = .{ .index = 0 },
        .source = null,
        .graph_instruction_ids = &codegen_instructions,
        .op_class = .matmul,
        .dtype = .f32,
        .accumulation_dtype = null,
        .logical_ops = 16,
        .bytes_read = 8,
        .bytes_written = 8,
        .expected_unit_id = null,
        .formula = "unit test cost formula",
        .approximation = "exact",
    }};
    const lowering_region_placements = [_]u32{0};
    try commitLoweringPlan(&session, &performance_costs, &diagnostics.writer);
    try std.testing.expectEqual(ModuleState.lowering_planned, moduleState(&session).?);
    const lowering_plan_pass = getAttr(session.moduleOperation(), "pjrtx.lowering_plan.pass");
    try std.testing.expectEqualStrings("pjrtx-lowering-plan-external", stringAttrValue(lowering_plan_pass).?);

    const extracted_lowerings = try extractLoweringRecords(std.testing.allocator, &session, 1, &diagnostics.writer);
    defer deinitExtractedLoweringRecords(std.testing.allocator, extracted_lowerings);
    const expected_lowering_count: usize = 1;
    try std.testing.expectEqual(expected_lowering_count, extracted_lowerings.len);
    try std.testing.expectEqual(compiler_facts.LoweringDecision.backend_kernel_graph, extracted_lowerings[0].decision);
    try std.testing.expectEqualStrings("MLIR lowering_region_form keeps this operation as a backend kernel graph boundary", extracted_lowerings[0].reason);
    const extracted_region_facts = try extractLoweringRegionFacts(std.testing.allocator, &session, extracted_lowerings.len, &diagnostics.writer);
    defer deinitExtractedLoweringRegionFacts(std.testing.allocator, extracted_region_facts);
    try std.testing.expectEqual(expected_lowering_count, extracted_region_facts.len);
    try std.testing.expectEqual(compiler_facts.LoweringDecision.backend_kernel_graph, extracted_region_facts[0].codegen_region);
    try std.testing.expectEqualSlices(u32, &lowering_region_placements, extracted_region_facts[0].placement_record_indices);
    try std.testing.expectEqualSlices(i64, &codegen_tile, extracted_region_facts[0].logical_tile_shape);

    const performance_traffic = [_]compiler_facts.MemoryTrafficRecord{
        .{
            .id = .{ .index = 0 },
            .lowering_record_id = .{ .index = 0 },
            .memory_space_id = 0,
            .kind = .global_memory,
            .graph_instruction_ids = &codegen_instructions,
            .cost_ledger_ids = &codegen_costs,
            .bytes_read = 8,
            .bytes_written = 8,
            .reason = "unit test global traffic",
        },
        .{
            .id = .{ .index = 1 },
            .lowering_record_id = .{ .index = 0 },
            .memory_space_id = 0,
            .kind = .local_memory,
            .graph_instruction_ids = &codegen_instructions,
            .cost_ledger_ids = &codegen_costs,
            .bytes_read = 4,
            .bytes_written = 4,
            .reason = "unit test local traffic",
        },
    };
    try commitPerformanceFacts(&session, &performance_costs, &performance_traffic, extracted_lowerings.len, &diagnostics.writer);
    try std.testing.expectEqual(ModuleState.performance_modeled, moduleState(&session).?);
    const performance_pass = getAttr(session.moduleOperation(), "pjrtx.performance.pass");
    try std.testing.expectEqualStrings("pjrtx-performance-external", stringAttrValue(performance_pass).?);

    const extracted_costs = try extractCostLedgerEntries(std.testing.allocator, &session, &diagnostics.writer);
    defer deinitExtractedCostLedgerEntries(std.testing.allocator, extracted_costs);
    const extracted_traffic = try extractMemoryTrafficRecords(std.testing.allocator, &session, extracted_costs.len, extracted_lowerings.len, &diagnostics.writer);
    defer deinitExtractedMemoryTrafficRecords(std.testing.allocator, extracted_traffic);
    try std.testing.expectEqual(performance_costs.len, extracted_costs.len);
    try std.testing.expectEqual(compiler_facts.CostOpClass.matmul, extracted_costs[0].op_class);
    const expected_performance_ops: u128 = 16;
    try std.testing.expectEqual(expected_performance_ops, extracted_costs[0].logical_ops);
    try std.testing.expectEqual(performance_traffic.len, extracted_traffic.len);
    try std.testing.expectEqual(compiler_facts.MemoryTrafficKind.local_memory, extracted_traffic[1].kind);
    try std.testing.expectEqualStrings("unit test local traffic", extracted_traffic[1].reason);

    const codegen_records = [_]core.KernelCodegenRecord{.{
        .id = .{ .index = 0 },
        .lowering_record_id = .{ .index = 0 },
        .command_id = .{ .index = 1 },
        .backend_kind = .npu_v0,
        .kind = .backend_kernel_graph,
        .operation = "unit_test_kernel",
        .shape = .{
            .operation_count = 1,
            .external_input_count = 2,
            .external_output_count = 1,
            .intermediate_value_count = 0,
        },
        .logical_tile_shape = &codegen_tile,
        .result_memory_space_id = 0,
        .tile_memory_space_id = null,
        .memory_pressure = .{
            .global_bytes_read = 80,
            .global_bytes_written = 24,
            .local_bytes_read = 0,
            .local_bytes_written = 0,
        },
        .external_input_ids = &codegen_inputs,
        .external_output_ids = &codegen_outputs,
        .intermediate_value_ids = &codegen_intermediates,
        .graph_instruction_ids = &codegen_instructions,
        .cost_ledger_ids = &codegen_costs,
        .memory_traffic_ids = &codegen_traffic,
        .expected_unit_id = null,
        .reason = "unit test codegen fact is committed to MLIR before extraction",
    }};
    try commitKernelCodegenPlan(&session, &codegen_records, &diagnostics.writer);
    try std.testing.expectEqual(ModuleState.codegen_planned, moduleState(&session).?);
    const codegen_plan_pass = getAttr(session.moduleOperation(), "pjrtx.codegen_plan.pass");
    try std.testing.expectEqualStrings("pjrtx-codegen-plan-external", stringAttrValue(codegen_plan_pass).?);

    const extracted_codegen = try extractKernelCodegenRecords(std.testing.allocator, &session, &diagnostics.writer);
    defer deinitExtractedKernelCodegenRecords(std.testing.allocator, extracted_codegen);
    const expected_codegen_count: usize = 1;
    try std.testing.expectEqual(expected_codegen_count, extracted_codegen.len);
    try std.testing.expectEqual(core.KernelCodegenKind.backend_kernel_graph, extracted_codegen[0].kind);
    try std.testing.expectEqualStrings("unit_test_kernel", extracted_codegen[0].operation);
    try std.testing.expectEqualSlices(i64, &codegen_tile, extracted_codegen[0].logical_tile_shape);
    try std.testing.expectEqualStrings("unit test codegen fact is committed to MLIR before extraction", extracted_codegen[0].reason);

    const command_inputs = [_]compiler_facts.GraphValueId{ .{ .index = 0 }, .{ .index = 1 } };
    const command_outputs = [_]compiler_facts.GraphValueId{.{ .index = 3 }};
    const command_dependencies = [_]core.CommandDependency{.{
        .command_id = .{ .index = 0 },
        .kind = .data,
    }};
    const command_lowerings = [_]compiler_facts.LoweringRecordId{.{ .index = 0 }};
    const command_costs = [_]compiler_facts.CostLedgerId{.{ .index = 0 }};
    const empty_dependencies = [_]core.CommandDependency{};
    const empty_lowerings = [_]compiler_facts.LoweringRecordId{};
    const empty_costs = [_]compiler_facts.CostLedgerId{};
    const schedule_commands = [_]core.ScheduleCommand{
        .{
            .id = .{ .index = 0 },
            .kind = .host_to_device,
            .stream = .{ .index = 0 },
            .inputs = &command_inputs,
            .outputs = &command_inputs,
            .dependencies = &empty_dependencies,
            .lowering_record_ids = &empty_lowerings,
            .cost_ledger_ids = &empty_costs,
        },
        .{
            .id = .{ .index = 1 },
            .kind = .backend_execute,
            .stream = .{ .index = 0 },
            .inputs = &command_inputs,
            .outputs = &command_outputs,
            .dependencies = &command_dependencies,
            .lowering_record_ids = &command_lowerings,
            .cost_ledger_ids = &command_costs,
        },
    };
    const schedule_overlaps = [_]core.ScheduleOverlapRecord{.{
        .id = .{ .index = 0 },
        .decision = .serialized,
        .kind = .transfer_compute,
        .first_command_id = .{ .index = 0 },
        .second_command_id = .{ .index = 1 },
        .dependency_kind = .data,
        .first_stream = .{ .index = 0 },
        .second_stream = .{ .index = 0 },
        .reason = "unit test schedule fact is committed to MLIR before extraction",
    }};
    try commitSchedulePlan(&session, &schedule_commands, &schedule_overlaps, &diagnostics.writer);
    try std.testing.expectEqual(ModuleState.scheduled, moduleState(&session).?);
    const schedule_plan_pass = getAttr(session.moduleOperation(), "pjrtx.schedule_plan.pass");
    try std.testing.expectEqualStrings("pjrtx-schedule-plan-external", stringAttrValue(schedule_plan_pass).?);

    const extracted_commands = try extractScheduleCommands(std.testing.allocator, &session, &diagnostics.writer);
    defer deinitExtractedScheduleCommands(std.testing.allocator, extracted_commands);
    const extracted_overlaps = try extractScheduleOverlapRecords(std.testing.allocator, &session, extracted_commands.len, &diagnostics.writer);
    defer deinitExtractedScheduleOverlapRecords(std.testing.allocator, extracted_overlaps);
    const expected_command_count: usize = 2;
    const expected_overlap_count: usize = 1;
    try std.testing.expectEqual(expected_command_count, extracted_commands.len);
    try std.testing.expectEqual(expected_overlap_count, extracted_overlaps.len);
    try std.testing.expectEqual(core.CommandKind.backend_execute, extracted_commands[1].kind);
    try std.testing.expectEqual(core.ScheduleOverlapKind.transfer_compute, extracted_overlaps[0].kind);
    try std.testing.expectEqualStrings("unit test schedule fact is committed to MLIR before extraction", extracted_overlaps[0].reason);

    const binding_instructions = [_]compiler_facts.GraphInstructionId{.{ .index = 0 }};
    const binding_costs = [_]compiler_facts.CostLedgerId{.{ .index = 0 }};
    const backend_bindings = [_]core.BackendBinding{.{
        .id = .{ .index = 0 },
        .command_id = .{ .index = 1 },
        .backend_kind = .npu_v0,
        .backend_operation = "unit_test_backend",
        .graph_instruction_ids = &binding_instructions,
        .expected_unit_id = null,
        .cost_ledger_ids = &binding_costs,
    }};
    try commitBackendBindings(&session, &backend_bindings, &diagnostics.writer);
    try std.testing.expectEqual(ModuleState.backend_bound, moduleState(&session).?);
    const backend_binding_pass = getAttr(session.moduleOperation(), "pjrtx.backend_binding.pass");
    try std.testing.expectEqualStrings("pjrtx-backend-binding-external", stringAttrValue(backend_binding_pass).?);

    const extracted_bindings = try extractBackendBindings(std.testing.allocator, &session, &diagnostics.writer);
    defer deinitExtractedBackendBindings(std.testing.allocator, extracted_bindings);
    const expected_binding_count: usize = 1;
    try std.testing.expectEqual(expected_binding_count, extracted_bindings.len);
    try std.testing.expectEqual(core.BackendKind.npu_v0, extracted_bindings[0].backend_kind);
    try std.testing.expectEqualStrings("unit_test_backend", extracted_bindings[0].backend_operation);

    const executable_contract: ExecutableContract = .{
        .target_kind = .npu_v0,
        .schedule_command_count = 2,
        .backend_binding_count = 1,
        .kernel_codegen_count = 1,
    };
    try commitExecutableReadiness(&session, executable_contract, &diagnostics.writer);
    try std.testing.expectEqual(ModuleState.executable_ready, moduleState(&session).?);
    const executable_ready_pass = getAttr(session.moduleOperation(), "pjrtx.executable_ready.pass");
    try std.testing.expectEqualStrings("pjrtx-executable-ready-external", stringAttrValue(executable_ready_pass).?);

    const extracted_contract = try extractExecutableContract(&session, &diagnostics.writer);
    try std.testing.expectEqual(target_pkg.TargetKind.npu_v0, extracted_contract.target_kind);
    try std.testing.expectEqual(executable_contract.schedule_command_count, extracted_contract.schedule_command_count);
    try std.testing.expectEqual(executable_contract.backend_binding_count, extracted_contract.backend_binding_count);
    try std.testing.expectEqual(executable_contract.kernel_codegen_count, extracted_contract.kernel_codegen_count);

    var snapshot: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer snapshot.deinit();
    try writeModuleSnapshot(&session, &snapshot.writer);
    try std.testing.expect(std.mem.indexOf(u8, snapshot.writer.buffered(), "pjrtx.state") != null);
    try std.testing.expect(std.mem.indexOf(u8, snapshot.writer.buffered(), "pjrtx.fusion.candidates") != null);
    try std.testing.expect(std.mem.indexOf(u8, snapshot.writer.buffered(), "pjrtx.fusion.plan") == null);
    try std.testing.expect(std.mem.indexOf(u8, snapshot.writer.buffered(), "pjrtx.placement.records") != null);
    try std.testing.expect(std.mem.indexOf(u8, snapshot.writer.buffered(), "pjrtx.collective.records") != null);
    try std.testing.expect(std.mem.indexOf(u8, snapshot.writer.buffered(), "pjrtx.codegen.records") != null);
    try std.testing.expect(std.mem.indexOf(u8, snapshot.writer.buffered(), "pjrtx.schedule.commands") != null);
    try std.testing.expect(std.mem.indexOf(u8, snapshot.writer.buffered(), "pjrtx.backend.bindings") != null);
    try std.testing.expect(std.mem.indexOf(u8, snapshot.writer.buffered(), "pjrtx.executable.contract") != null);
}

test "fusion facts require target legal state and valid pressure arithmetic" {
    const program = "module { func.func @main() { return } }";
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    var session: MlirSession = try .initFromStablehloText(std.testing.allocator, program, .{}, &diagnostics.writer);
    defer session.deinit();

    try std.testing.expectError(MlirStateError.InvalidStateTransition, planFusionFromCandidates(&session, &diagnostics.writer));
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "unexpected module state") != null);
}

test "fusion plan external pass rejects missing candidate facts" {
    const program = "module { func.func @main() { return } }";
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    var session: MlirSession = try .initFromStablehloText(std.testing.allocator, program, .{}, &diagnostics.writer);
    defer session.deinit();

    try attachTarget(&session, testTarget(), 1, 1, &diagnostics.writer);
    try markTargetLegal(&session, &diagnostics.writer);
    try std.testing.expectError(MlirStateError.InvalidFusionPlan, runFusionPlanExternalPass(&session, &diagnostics.writer));
    try std.testing.expectEqual(ModuleState.target_legal, moduleState(&session).?);
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "pass=fusion_plan_external") != null);
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "missing MLIR fusion candidate attribute") != null);
}

test "fusion candidate external pass requires target legal state" {
    const program = "module { func.func @main() { return } }";
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    var session: MlirSession = try .initFromStablehloText(std.testing.allocator, program, .{}, &diagnostics.writer);
    defer session.deinit();

    try std.testing.expectError(MlirStateError.InvalidStateTransition, runFusionCandidateDiscoveryExternalPass(&session, &diagnostics.writer));
    try std.testing.expectEqual(ModuleState.imported, moduleState(&session).?);
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "pass=fusion_candidate_external") != null);
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "expected target_legal") != null);
}

test "Zig external MLIR pass runs through pass manager and stamps module state" {
    const program = "module { func.func @main() { return } }";
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    var session: MlirSession = try .initFromStablehloText(std.testing.allocator, program, .{}, &diagnostics.writer);
    defer session.deinit();

    const record: ExternalPassProbeRecord = try runExternalStateProbePass(&session, &diagnostics.writer);
    try std.testing.expectEqual(ModuleState.imported, moduleState(&session).?);
    try std.testing.expect(record.construct_count >= 1);
    try std.testing.expect(record.initialize_count >= 1);
    const expected_run_count: u32 = 1;
    try std.testing.expectEqual(expected_run_count, record.run_count);
    try std.testing.expect(record.destruct_count >= 1);

    const proof_attr = getAttr(session.moduleOperation(), "pjrtx.external_pass.proof");
    try std.testing.expectEqualStrings("ran", stringAttrValue(proof_attr).?);
    try std.testing.expectEqualStrings("", diagnostics.writer.buffered());
}

test "target legal external pass rejects missing target attachment" {
    const program = "module { func.func @main() { return } }";
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    var session: MlirSession = try .initFromStablehloText(std.testing.allocator, program, .{}, &diagnostics.writer);
    defer session.deinit();

    try std.testing.expectError(MlirStateError.InvalidStateTransition, markTargetLegal(&session, &diagnostics.writer));
    try std.testing.expectEqual(ModuleState.imported, moduleState(&session).?);
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "pass=target_legal_external") != null);
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "expected target_attached") != null);
}

fn testTarget() target_pkg.TargetDescription {
    const memory_spaces = struct {
        const values = [_]target_pkg.TargetMemorySpace{.{
            .id = 0,
            .name = "hbm",
            .kind = .device_hbm,
            .capacity_bytes = 1024,
            .bandwidth_bytes_per_second = 1000,
            .note = "test",
        }};
    }.values;
    const units = struct {
        const values = [_]target_pkg.ExecutionUnit{.{
            .id = 0,
            .name = "matrix",
            .kind = .matrix,
            .dtype_rates = &.{},
        }};
    }.values;
    const devices = struct {
        const memory_ids = [_]u32{0};
        const unit_ids = [_]u32{0};
        const values = [_]target_pkg.TargetDevice{.{
            .id = 0,
            .local_hardware_id = 0,
            .name = "device0",
            .memory_space_ids = &memory_ids,
            .execution_unit_ids = &unit_ids,
        }};
    }.values;
    return .{
        .name = "npu_test",
        .kind = .npu_v0,
        .devices = &devices,
        .memory_spaces = &memory_spaces,
        .transfer_edges = &.{},
        .execution_units = &units,
    };
}
