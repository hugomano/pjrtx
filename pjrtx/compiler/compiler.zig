const std = @import("std");
const mlir = @import("c");
const core = @import("pjrtx/core");
const compiler_facts = @import("pjrtx/compiler/facts");
const mlir_state = @import("pjrtx/compiler/mlir_state");
const target_pkg = @import("pjrtx/target");

pub const CompilePipelineStage = enum {
    input_setup,
    stablehlo_ingest,
    mlir_verify,
    mlir_lowering_pass_pipeline,
    graph_import,
    graph_verify,
    target_selection,
    target_legality,
    algebraic_normalization,
    fusion_candidate_discovery,
    fusion_decision_plan,
    placement_planning,
    collective_lowering,
    cost_ledger,
    lowering,
    schedule_build,
    schedule_verify,
    kernel_codegen,
    backend_binding,
    backend_binding_verify,
    executable_creation,
    report_emission,
};

pub const CompilePipelineError = error{
    InvalidInput,
    InvalidStablehlo,
    InvalidGraph,
    InvalidSchedule,
    UnsupportedTargetFeature,
    InvalidLowering,
    InvalidBackendBinding,
};

pub const ProgramFormat = enum {
    stablehlo_text,
    stablehlo_bytecode,
};

pub const CompileOptions = struct {
    num_replicas: u32 = 1,
    num_partitions: u32 = 1,
    use_shardy_partitioner: bool = true,
};

pub const CompileInput = struct {
    allocator: std.mem.Allocator,
    program_format: ProgramFormat,
    target_kind: target_pkg.TargetKind,
    compile_options: CompileOptions,
    program_bytes: []u8,

    pub fn deinit(self: *CompileInput) void {
        self.allocator.free(self.program_bytes);
        self.* = undefined;
    }
};

pub const SelectedTarget = struct {
    description: target_pkg.TargetDescription,
};

pub const CompiledExecutableView = struct {
    report: core.TraceReport,
};

pub const CompiledV0 = struct {
    graph: GraphModule,
    planned: PlannedTraceReport,

    pub fn deinit(self: *CompiledV0) void {
        self.planned.deinit();
        self.graph.deinit();
        self.* = undefined;
    }

    pub fn report(self: *const CompiledV0) core.TraceReport {
        return self.planned.report;
    }
};

pub const PlannedTraceReport = struct {
    allocator: std.mem.Allocator,
    report: core.TraceReport,
    compile_options: CompileOptions,
    fusion_group_strings_owned: bool = false,
    placement_reason_strings_owned: bool = false,
    collective_reason_strings_owned: bool = false,
    lowering_strings_owned: bool = false,
    performance_strings_owned: bool = false,
    kernel_codegen_strings_owned: bool = false,
    schedule_overlap_reason_strings_owned: bool = false,
    backend_binding_strings_owned: bool = false,

    pub fn deinit(self: *PlannedTraceReport) void {
        if (self.report.mlir_pass_records.len > 0) self.allocator.free(self.report.mlir_pass_records);
        if (self.report.graph_rewrite_records.len > 0) self.allocator.free(self.report.graph_rewrite_records);
        freeOwnedFusionGroups(self.allocator, self.report.fusion_groups, self.fusion_group_strings_owned);
        for (self.report.placement_records) |record| {
            self.allocator.free(record.output_value_ids);
            self.allocator.free(record.logical_tile_shape);
            if (self.placement_reason_strings_owned) self.allocator.free(record.reason);
        }
        if (self.report.placement_records.len > 0) self.allocator.free(self.report.placement_records);
        if (self.collective_reason_strings_owned) {
            for (self.report.collective_plan_records) |record| self.allocator.free(record.reason);
        }
        if (self.report.collective_plan_records.len > 0) self.allocator.free(self.report.collective_plan_records);
        for (self.report.cost_ledger) |entry| {
            if (self.performance_strings_owned) {
                if (entry.source) |source| {
                    self.allocator.free(source.op_name);
                    self.allocator.free(source.location);
                }
                self.allocator.free(entry.formula);
                self.allocator.free(entry.approximation);
            }
            self.allocator.free(entry.graph_instruction_ids);
        }
        for (self.report.lowering_records) |record| {
            if (self.lowering_strings_owned) {
                self.allocator.free(record.reason);
                for (record.rejected_alternatives) |alternative| self.allocator.free(alternative);
                self.allocator.free(record.rejected_alternatives);
            }
            self.allocator.free(record.graph_instruction_ids);
            self.allocator.free(record.cost_ledger_ids);
        }
        for (self.report.memory_traffic_records) |record| {
            if (self.performance_strings_owned) self.allocator.free(record.reason);
            self.allocator.free(record.graph_instruction_ids);
            self.allocator.free(record.cost_ledger_ids);
        }
        if (self.schedule_overlap_reason_strings_owned) {
            for (self.report.schedule_overlap_records) |record| self.allocator.free(record.reason);
        }
        if (self.report.schedule_overlap_records.len > 0) self.allocator.free(self.report.schedule_overlap_records);
        for (self.report.schedule_commands) |command| {
            self.allocator.free(command.inputs);
            self.allocator.free(command.outputs);
            self.allocator.free(command.dependencies);
            self.allocator.free(command.lowering_record_ids);
            self.allocator.free(command.cost_ledger_ids);
        }
        for (self.report.kernel_codegen_records) |record| {
            if (self.kernel_codegen_strings_owned) {
                self.allocator.free(record.operation);
                self.allocator.free(record.reason);
            }
            self.allocator.free(record.logical_tile_shape);
            self.allocator.free(record.external_input_ids);
            self.allocator.free(record.external_output_ids);
            self.allocator.free(record.intermediate_value_ids);
            self.allocator.free(record.graph_instruction_ids);
            self.allocator.free(record.cost_ledger_ids);
            self.allocator.free(record.memory_traffic_ids);
        }
        for (self.report.backend_bindings) |binding| {
            if (self.backend_binding_strings_owned) self.allocator.free(binding.backend_operation);
            self.allocator.free(binding.graph_instruction_ids);
            self.allocator.free(binding.cost_ledger_ids);
        }
        for (self.report.explain_records) |explain| {
            self.allocator.free(explain.source_refs);
            self.allocator.free(explain.cost_ledger_ids);
            self.allocator.free(explain.profile_event_ids);
        }
        self.allocator.free(self.report.cost_ledger);
        self.allocator.free(self.report.lowering_records);
        self.allocator.free(self.report.memory_traffic_records);
        self.allocator.free(self.report.schedule_commands);
        self.allocator.free(self.report.kernel_codegen_records);
        self.allocator.free(self.report.backend_bindings);
        self.allocator.free(self.report.profile_events);
        self.allocator.free(self.report.explain_records);
        self.* = undefined;
    }
};

fn freeOwnedFusionGroups(allocator: std.mem.Allocator, groups: []const compiler_facts.FusionGroup, owns_group_strings: bool) void {
    for (groups) |group| {
        allocator.free(group.graph_instruction_ids);
        if (owns_group_strings) {
            allocator.free(group.kind);
            allocator.free(group.reason);
        }
    }
    if (groups.len > 0) allocator.free(groups);
}

pub const MlirModuleArtifact = struct {
    registry: mlir.MlirDialectRegistry,
    context: mlir.MlirContext,
    module: mlir.MlirModule,

    pub fn deinit(self: *MlirModuleArtifact) void {
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
};

pub const GraphModule = struct {
    allocator: std.mem.Allocator,
    sources: []compiler_facts.SourceRef,
    values: []compiler_facts.GraphValue,
    instructions: []compiler_facts.GraphInstruction,

    pub fn deinit(self: *GraphModule) void {
        for (self.instructions) |instruction| {
            self.allocator.free(instruction.inputs);
            self.allocator.free(instruction.outputs);
            switch (instruction.payload) {
                .broadcast => |payload| self.allocator.free(payload.dimensions),
                .transpose => |payload| self.allocator.free(payload.permutation),
                .collective => |payload| self.allocator.free(payload.replica_groups),
                else => {},
            }
        }
        for (self.values) |value| {
            self.allocator.free(value.ty.dims);
        }
        for (self.sources) |source| {
            self.allocator.free(source.op_name);
            self.allocator.free(source.location);
        }
        self.allocator.free(self.instructions);
        self.allocator.free(self.values);
        self.allocator.free(self.sources);
        self.* = undefined;
    }
};

pub const GraphNormalizationResult = struct {
    allocator: std.mem.Allocator,
    graph: GraphModule,
    records: []const compiler_facts.GraphRewriteRecord,

    pub fn deinit(self: *GraphNormalizationResult) void {
        self.graph.deinit();
        if (self.records.len > 0) self.allocator.free(self.records);
        self.* = undefined;
    }
};

pub const MlirPassStatus = compiler_facts.MlirPassStatus;
pub const MlirPassRecord = compiler_facts.MlirPassRecord;

pub const MlirPassPipelineReport = struct {
    allocator: std.mem.Allocator,
    has_shardy_metadata: bool,
    shardy_requested: bool,
    records: []const compiler_facts.MlirPassRecord,

    pub fn deinit(self: *MlirPassPipelineReport) void {
        self.allocator.free(self.records);
        self.* = undefined;
    }
};

pub const FusionDecision = compiler_facts.FusionDecision;
pub const FusionGroup = compiler_facts.FusionGroup;

pub const PlacementRecord = compiler_facts.PlacementRecord;

pub const PlacementPlan = struct {
    allocator: std.mem.Allocator,
    records: []const compiler_facts.PlacementRecord,
    reason_strings_owned: bool = false,

    pub fn deinit(self: *PlacementPlan) void {
        for (self.records) |record| {
            self.allocator.free(record.output_value_ids);
            self.allocator.free(record.logical_tile_shape);
            if (self.reason_strings_owned) self.allocator.free(record.reason);
        }
        self.allocator.free(self.records);
        self.* = undefined;
    }
};

pub const CollectivePlan = compiler_facts.CollectivePlanRecord;

pub const CompilerPassArtifactKind = enum {
    input_bytes,
    mlir_module,
    graph_module,
    target_description,
    fusion_candidates,
    fusion_decisions,
    placement_plan,
    collective_plan,
    cost_ledger,
    lowering_plan,
    schedule_plan,
    kernel_codegen_plan,
    backend_binding,
    executable_view,
    trace_report,
};

pub const CompilerPassInvariant = enum {
    source_provenance,
    shardy_metadata,
    stablehlo_semantics,
    shape_dtype_layout,
    target_memory_spaces,
    collective_semantics,
    lowering_cost_links,
    schedule_dependency_order,
    backend_target_match,
};

pub const CompilerPassEffect = enum {
    verifies_only,
    records_boundary,
    may_change_ir,
    plans_performance,
    lowers_collectives,
    binds_backend,
};

pub const CompilerPassContract = struct {
    stable_name: []const u8,
    stage: CompilePipelineStage,
    input_artifact: CompilerPassArtifactKind,
    output_artifact: CompilerPassArtifactKind,
    required_target_facts: []const u8,
    preserved_invariants: []const CompilerPassInvariant,
    invalidated_analyses: []const u8,
    emitted_records: []const u8,
    effect: CompilerPassEffect,
    failure_feature: []const u8,
};

const parse_invariants = [_]CompilerPassInvariant{
    .source_provenance,
    .stablehlo_semantics,
};
const mlir_verify_invariants = [_]CompilerPassInvariant{
    .source_provenance,
    .shardy_metadata,
    .stablehlo_semantics,
};
const graph_invariants = [_]CompilerPassInvariant{
    .source_provenance,
    .shardy_metadata,
    .stablehlo_semantics,
    .shape_dtype_layout,
};
const target_invariants = [_]CompilerPassInvariant{
    .source_provenance,
    .shape_dtype_layout,
    .target_memory_spaces,
};
const fusion_invariants = [_]CompilerPassInvariant{
    .source_provenance,
    .shape_dtype_layout,
    .target_memory_spaces,
};
const collective_invariants = [_]CompilerPassInvariant{
    .source_provenance,
    .shardy_metadata,
    .collective_semantics,
    .target_memory_spaces,
};
const lowering_invariants = [_]CompilerPassInvariant{
    .source_provenance,
    .shape_dtype_layout,
    .target_memory_spaces,
    .lowering_cost_links,
};
const schedule_invariants = [_]CompilerPassInvariant{
    .source_provenance,
    .target_memory_spaces,
    .lowering_cost_links,
    .schedule_dependency_order,
};
const backend_invariants = [_]CompilerPassInvariant{
    .source_provenance,
    .target_memory_spaces,
    .lowering_cost_links,
    .schedule_dependency_order,
    .backend_target_match,
};

const v0_compiler_pass_contracts = [_]CompilerPassContract{
    .{
        .stable_name = "stablehlo_parse",
        .stage = .stablehlo_ingest,
        .input_artifact = .input_bytes,
        .output_artifact = .mlir_module,
        .required_target_facts = "none",
        .preserved_invariants = &parse_invariants,
        .invalidated_analyses = "all",
        .emitted_records = "MlirPassRecord",
        .effect = .records_boundary,
        .failure_feature = "stablehlo",
    },
    .{
        .stable_name = "mlir_verify",
        .stage = .mlir_verify,
        .input_artifact = .mlir_module,
        .output_artifact = .mlir_module,
        .required_target_facts = "none",
        .preserved_invariants = &mlir_verify_invariants,
        .invalidated_analyses = "none",
        .emitted_records = "MlirPassRecord",
        .effect = .verifies_only,
        .failure_feature = "stablehlo",
    },
    .{
        .stable_name = "collective_graph_payload_import",
        .stage = .collective_lowering,
        .input_artifact = .mlir_module,
        .output_artifact = .graph_module,
        .required_target_facts = "collective engine and topology availability",
        .preserved_invariants = &collective_invariants,
        .invalidated_analyses = "collective algorithm choices",
        .emitted_records = "GraphInstruction collective payload",
        .effect = .lowers_collectives,
        .failure_feature = "stablehlo.collective",
    },
    .{
        .stable_name = "mlir_canonicalize_cse",
        .stage = .mlir_lowering_pass_pipeline,
        .input_artifact = .mlir_module,
        .output_artifact = .mlir_module,
        .required_target_facts = "none",
        .preserved_invariants = &mlir_verify_invariants,
        .invalidated_analyses = "operation fingerprints",
        .emitted_records = "MlirPassRecord",
        .effect = .may_change_ir,
        .failure_feature = "canonicalization",
    },
    .{
        .stable_name = "shardy_metadata_propagation_report",
        .stage = .mlir_lowering_pass_pipeline,
        .input_artifact = .mlir_module,
        .output_artifact = .mlir_module,
        .required_target_facts = "mesh and partition count options",
        .preserved_invariants = &mlir_verify_invariants,
        .invalidated_analyses = "sharding summaries",
        .emitted_records = "MlirPassRecord",
        .effect = .records_boundary,
        .failure_feature = "shardy",
    },
    .{
        .stable_name = "pjrtx_graph_import",
        .stage = .graph_import,
        .input_artifact = .mlir_module,
        .output_artifact = .graph_module,
        .required_target_facts = "none",
        .preserved_invariants = &graph_invariants,
        .invalidated_analyses = "mlir operation handles",
        .emitted_records = "GraphValue,GraphInstruction,SourceRef",
        .effect = .records_boundary,
        .failure_feature = "program",
    },
    .{
        .stable_name = "pjrtx_graph_verify",
        .stage = .graph_verify,
        .input_artifact = .graph_module,
        .output_artifact = .graph_module,
        .required_target_facts = "none",
        .preserved_invariants = &graph_invariants,
        .invalidated_analyses = "none",
        .emitted_records = "diagnostics",
        .effect = .verifies_only,
        .failure_feature = "graph",
    },
    .{
        .stable_name = "target_feature_legality",
        .stage = .target_legality,
        .input_artifact = .graph_module,
        .output_artifact = .graph_module,
        .required_target_facts = "backend capabilities, dtype rates, memory spaces",
        .preserved_invariants = &target_invariants,
        .invalidated_analyses = "none",
        .emitted_records = "diagnostics",
        .effect = .verifies_only,
        .failure_feature = "backend-capability",
    },
    .{
        .stable_name = "broadcast_simplify",
        .stage = .algebraic_normalization,
        .input_artifact = .graph_module,
        .output_artifact = .graph_module,
        .required_target_facts = "none",
        .preserved_invariants = &graph_invariants,
        .invalidated_analyses = "instruction ids, fusion candidates, lowering regions",
        .emitted_records = "GraphRewriteRecord",
        .effect = .may_change_ir,
        .failure_feature = "broadcast",
    },
    .{
        .stable_name = "reshape_transpose_fold",
        .stage = .algebraic_normalization,
        .input_artifact = .graph_module,
        .output_artifact = .graph_module,
        .required_target_facts = "none",
        .preserved_invariants = &graph_invariants,
        .invalidated_analyses = "instruction ids, fusion candidates, lowering regions",
        .emitted_records = "GraphRewriteRecord",
        .effect = .may_change_ir,
        .failure_feature = "reshape-transpose",
    },
    .{
        .stable_name = "fusion_candidate_discovery",
        .stage = .fusion_candidate_discovery,
        .input_artifact = .graph_module,
        .output_artifact = .fusion_candidates,
        .required_target_facts = "launch overhead, memory bandwidth, supported fused ops",
        .preserved_invariants = &fusion_invariants,
        .invalidated_analyses = "lowering regions",
        .emitted_records = "FusionGroup",
        .effect = .plans_performance,
        .failure_feature = "fusion",
    },
    .{
        .stable_name = "matmul_epilogue_fusion_select",
        .stage = .fusion_decision_plan,
        .input_artifact = .fusion_candidates,
        .output_artifact = .fusion_decisions,
        .required_target_facts = "matmul epilogue support, math policy, dtype rates, local memory pressure",
        .preserved_invariants = &fusion_invariants,
        .invalidated_analyses = "lowering regions",
        .emitted_records = "FusionGroup",
        .effect = .plans_performance,
        .failure_feature = "fusion.matmul-epilogue",
    },
    .{
        .stable_name = "tile_shape_select",
        .stage = .placement_planning,
        .input_artifact = .fusion_decisions,
        .output_artifact = .placement_plan,
        .required_target_facts = "memory spaces, capacities, transfer bandwidths, target tile constraints",
        .preserved_invariants = &target_invariants,
        .invalidated_analyses = "memory traffic estimates",
        .emitted_records = "PlacementRecord",
        .effect = .plans_performance,
        .failure_feature = "placement",
    },
    .{
        .stable_name = "collective_group_channel_verify",
        .stage = .collective_lowering,
        .input_artifact = .graph_module,
        .output_artifact = .graph_module,
        .required_target_facts = "replica count, partition count, collective engine, channel handle type",
        .preserved_invariants = &collective_invariants,
        .invalidated_analyses = "collective algorithm choices",
        .emitted_records = "diagnostics",
        .effect = .verifies_only,
        .failure_feature = "collective.group-channel",
    },
    .{
        .stable_name = "collective_plan_v0",
        .stage = .collective_lowering,
        .input_artifact = .graph_module,
        .output_artifact = .collective_plan,
        .required_target_facts = "collective unit, topology, replica and partition counts",
        .preserved_invariants = &collective_invariants,
        .invalidated_analyses = "schedule overlap candidates",
        .emitted_records = "CollectivePlanRecord",
        .effect = .lowers_collectives,
        .failure_feature = "collective",
    },
    .{
        .stable_name = "collective_algorithm_select",
        .stage = .collective_lowering,
        .input_artifact = .collective_plan,
        .output_artifact = .collective_plan,
        .required_target_facts = "collective engine, topology, dtype, layout, bandwidth, latency",
        .preserved_invariants = &collective_invariants,
        .invalidated_analyses = "schedule overlap candidates",
        .emitted_records = "CollectivePlanRecord",
        .effect = .lowers_collectives,
        .failure_feature = "collective.algorithm",
    },
    .{
        .stable_name = "cost_ledger_build",
        .stage = .cost_ledger,
        .input_artifact = .graph_module,
        .output_artifact = .cost_ledger,
        .required_target_facts = "dtype rates and execution units",
        .preserved_invariants = &target_invariants,
        .invalidated_analyses = "roofline estimates",
        .emitted_records = "CostLedgerEntry",
        .effect = .plans_performance,
        .failure_feature = "cost-ledger",
    },
    .{
        .stable_name = "lowering_region_form",
        .stage = .lowering,
        .input_artifact = .cost_ledger,
        .output_artifact = .lowering_plan,
        .required_target_facts = "backend operation support and memory spaces",
        .preserved_invariants = &lowering_invariants,
        .invalidated_analyses = "schedule candidates",
        .emitted_records = "LoweringRecord,LoweringRegionFact,ExplainRecord",
        .effect = .plans_performance,
        .failure_feature = "lowering",
    },
    .{
        .stable_name = "memory_traffic_refine",
        .stage = .lowering,
        .input_artifact = .lowering_plan,
        .output_artifact = .lowering_plan,
        .required_target_facts = "placement records, memory-space kind, bandwidth, transfer edges",
        .preserved_invariants = &lowering_invariants,
        .invalidated_analyses = "roofline estimates",
        .emitted_records = "MemoryTrafficRecord",
        .effect = .plans_performance,
        .failure_feature = "memory-traffic",
    },
    .{
        .stable_name = "schedule_build",
        .stage = .schedule_build,
        .input_artifact = .lowering_plan,
        .output_artifact = .schedule_plan,
        .required_target_facts = "streams, DMA engines, transfer edges",
        .preserved_invariants = &schedule_invariants,
        .invalidated_analyses = "allocation lifetimes",
        .emitted_records = "ScheduleCommand",
        .effect = .plans_performance,
        .failure_feature = "schedule",
    },
    .{
        .stable_name = "schedule_overlap_plan",
        .stage = .schedule_build,
        .input_artifact = .schedule_plan,
        .output_artifact = .schedule_plan,
        .required_target_facts = "streams, DMA engines, collective engines, command dependencies, transfer async support",
        .preserved_invariants = &schedule_invariants,
        .invalidated_analyses = "allocation lifetimes",
        .emitted_records = "ScheduleOverlapRecord",
        .effect = .plans_performance,
        .failure_feature = "schedule-overlap",
    },
    .{
        .stable_name = "kernel_codegen_plan",
        .stage = .kernel_codegen,
        .input_artifact = .schedule_plan,
        .output_artifact = .kernel_codegen_plan,
        .required_target_facts = "backend operation names, execution units, lowering regions, memory traffic",
        .preserved_invariants = &backend_invariants,
        .invalidated_analyses = "backend executable cache",
        .emitted_records = "KernelCodegenRecord",
        .effect = .plans_performance,
        .failure_feature = "kernel-codegen",
    },
    .{
        .stable_name = "tile_legality_verify",
        .stage = .kernel_codegen,
        .input_artifact = .kernel_codegen_plan,
        .output_artifact = .kernel_codegen_plan,
        .required_target_facts = "target memory capacities, kernel tile shape, codegen value flow, memory pressure",
        .preserved_invariants = &backend_invariants,
        .invalidated_analyses = "none",
        .emitted_records = "diagnostics",
        .effect = .verifies_only,
        .failure_feature = "tile-legality",
    },
    .{
        .stable_name = "backend_binding_select",
        .stage = .backend_binding,
        .input_artifact = .kernel_codegen_plan,
        .output_artifact = .backend_binding,
        .required_target_facts = "kernel codegen records, backend operation names and execution units",
        .preserved_invariants = &backend_invariants,
        .invalidated_analyses = "none",
        .emitted_records = "BackendBinding",
        .effect = .binds_backend,
        .failure_feature = "backend-binding",
    },
    .{
        .stable_name = "executable_view_create",
        .stage = .executable_creation,
        .input_artifact = .trace_report,
        .output_artifact = .executable_view,
        .required_target_facts = "validated target and backend bindings",
        .preserved_invariants = &backend_invariants,
        .invalidated_analyses = "none",
        .emitted_records = "CompiledExecutableView",
        .effect = .verifies_only,
        .failure_feature = "trace-report",
    },
};

pub fn v0CompilerPassContracts() []const CompilerPassContract {
    return &v0_compiler_pass_contracts;
}

pub fn writeCompilerPassCatalog(writer: *std.Io.Writer) std.Io.Writer.Error!void {
    try writer.writeAll("compiler_pass_catalog\n");
    for (v0_compiler_pass_contracts, 0..) |contract, index| {
        try writer.print(
            "  pass.{d} name={s} stage={s} input={s} output={s} effect={s} invariants=",
            .{
                index,
                contract.stable_name,
                @tagName(contract.stage),
                @tagName(contract.input_artifact),
                @tagName(contract.output_artifact),
                @tagName(contract.effect),
            },
        );
        try writeInvariantList(writer, contract.preserved_invariants);
        try writer.print(
            " records={s} target_facts={s} invalidates={s}\n",
            .{ contract.emitted_records, contract.required_target_facts, contract.invalidated_analyses },
        );
    }
}

const CompilerBackendFeature = enum {
    rank2_dot_general,
    broadcast_in_dim,
    reshape,
    transpose,
    add,
    tanh,
};

const CompilerBackendCapability = struct {
    feature: CompilerBackendFeature,
    dtypes: []const core.BufferType,
    backend_operation: []const u8,
    expected_unit_id: ?u32,
};

const CompilerBackendCapabilitySet = struct {
    kind: core.BackendKind,
    capabilities: []const CompilerBackendCapability,
};

pub fn writeStageName(writer: *std.Io.Writer, stage: CompilePipelineStage) std.Io.Writer.Error!void {
    try writer.writeAll(@tagName(stage));
}

/// The V0 orchestrator is the first end-to-end compile path that keeps MLIR
/// pass provenance attached to the final trace. It still returns an owned
/// compile artifact instead of a real PJRT executable, but every stage from
/// input through executable-view validation has run.
pub fn compileV0FromReader(
    allocator: std.mem.Allocator,
    format_text: []const u8,
    target_text: []const u8,
    options_reader: *std.Io.Reader,
    program_reader: *std.Io.Reader,
    diagnostics: *std.Io.Writer,
) !CompiledV0 {
    var input: CompileInput = try setupCompileInputFromReader(
        allocator,
        format_text,
        target_text,
        options_reader,
        program_reader,
        diagnostics,
    );
    defer input.deinit();

    var module: MlirModuleArtifact = try ingestStablehloText(&input, diagnostics);
    defer module.deinit();

    var pass_report: MlirPassPipelineReport = try buildMlirPassPipelineReport(allocator, &input, module, diagnostics);
    defer pass_report.deinit();

    var imported_graph: GraphModule = try importGraphFromMlir(allocator, module, diagnostics);
    defer imported_graph.deinit();
    try verifyGraphModule(imported_graph, diagnostics);

    var normalized: GraphNormalizationResult = try normalizeGraphV0(allocator, imported_graph, diagnostics);
    var normalized_owned = true;
    errdefer if (normalized_owned) normalized.deinit();

    const target: SelectedTarget = try selectTarget(input.target_kind, diagnostics);
    var state_session: mlir_state.MlirSession = try .initFromStablehloText(
        allocator,
        input.program_bytes,
        .{ .program_name = "compile_v0" },
        diagnostics,
    );
    defer state_session.deinit();

    var planned: PlannedTraceReport = try planV0TraceReportWithMlirState(
        allocator,
        normalized.graph,
        target,
        input.compile_options,
        &state_session,
        diagnostics,
    );
    errdefer planned.deinit();

    planned.report.mlir_pass_records = try copyMlirPassRecords(allocator, pass_report.records);
    planned.report.graph_rewrite_records = normalized.records;
    normalized.records = &.{};
    normalized_owned = false;
    return .{
        .graph = normalized.graph,
        .planned = planned,
    };
}

/// Input setup is the only V0 stage that turns caller-owned readers and strings
/// into an owned compile artifact. Later stages can fail freely without losing
/// the original program bytes needed for diagnostics and reports.
pub fn setupCompileInputFromReader(
    allocator: std.mem.Allocator,
    format_text: []const u8,
    target_text: []const u8,
    options_reader: *std.Io.Reader,
    program_reader: *std.Io.Reader,
    diagnostics: *std.Io.Writer,
) !CompileInput {
    const program_format = try parseProgramFormat(format_text, diagnostics);
    const target_kind = try parseTargetKind(target_text, diagnostics);
    const compile_options = try parseCompileOptionsFromReader(allocator, options_reader, diagnostics);

    const program_bytes = try program_reader.allocRemaining(allocator, .limited(64 * 1024 * 1024));
    errdefer allocator.free(program_bytes);
    if (program_bytes.len == 0) {
        try writeStageFailure(diagnostics, .input_setup, "program", "program is empty");
        return CompilePipelineError.InvalidInput;
    }

    return .{
        .allocator = allocator,
        .program_format = program_format,
        .target_kind = target_kind,
        .compile_options = compile_options,
        .program_bytes = program_bytes,
    };
}

pub fn parseProgramFormat(format_text: []const u8, diagnostics: *std.Io.Writer) !ProgramFormat {
    const normalized = std.mem.trim(u8, format_text, " \t\r\n");
    if (std.mem.eql(u8, normalized, "stablehlo")) return .stablehlo_text;
    if (std.mem.eql(u8, normalized, "stablehlo_text")) return .stablehlo_text;
    if (std.mem.eql(u8, normalized, "mlir")) return .stablehlo_text;
    if (std.mem.eql(u8, normalized, "stablehlo_bytecode")) return .stablehlo_bytecode;
    if (std.mem.eql(u8, normalized, "vhlo")) return .stablehlo_bytecode;

    try writeStageFailure(diagnostics, .input_setup, "program-format", "unsupported program format");
    return CompilePipelineError.InvalidInput;
}

pub fn parseTargetKind(target_text: []const u8, diagnostics: *std.Io.Writer) !target_pkg.TargetKind {
    const normalized = std.mem.trim(u8, target_text, " \t\r\n");
    if (std.mem.eql(u8, normalized, "metal_v0")) return .metal_v0;
    if (std.mem.eql(u8, normalized, "npu_v0")) return .npu_v0;

    try writeStageFailure(diagnostics, .input_setup, "target", "unknown target");
    return CompilePipelineError.InvalidInput;
}

pub fn parseCompileOptionsFromReader(
    allocator: std.mem.Allocator,
    reader: *std.Io.Reader,
    diagnostics: *std.Io.Writer,
) !CompileOptions {
    const text = try reader.allocRemaining(allocator, .limited(64 * 1024));
    defer allocator.free(text);

    var options: CompileOptions = .{};
    var entries = std.mem.splitScalar(u8, text, ';');
    while (entries.next()) |entry_text| {
        const entry = std.mem.trim(u8, entry_text, " \t\r\n");
        if (entry.len == 0) continue;

        const separator = std.mem.indexOfScalar(u8, entry, '=') orelse {
            try writeStageFailure(diagnostics, .input_setup, "compile-options", "compile option is missing '='");
            return CompilePipelineError.InvalidInput;
        };
        const key = std.mem.trim(u8, entry[0..separator], " \t\r\n");
        const value = std.mem.trim(u8, entry[separator + 1 ..], " \t\r\n");

        if (std.mem.eql(u8, key, "replicas")) {
            options.num_replicas = parsePositiveU32(value, diagnostics) catch return CompilePipelineError.InvalidInput;
        } else if (std.mem.eql(u8, key, "partitions")) {
            options.num_partitions = parsePositiveU32(value, diagnostics) catch return CompilePipelineError.InvalidInput;
        } else if (std.mem.eql(u8, key, "use_shardy")) {
            options.use_shardy_partitioner = parseBool(value, diagnostics) catch return CompilePipelineError.InvalidInput;
        } else {
            try writeStageFailure(diagnostics, .input_setup, "compile-options", "unknown compile option");
            return CompilePipelineError.InvalidInput;
        }
    }
    return options;
}

/// Stage 2 is intentionally a real MLIR/StableHLO C API boundary from the
/// beginning. This keeps PjRTx from growing a shadow parser that would disagree
/// with the dialect verifier.
pub fn ingestStablehloText(input: *const CompileInput, diagnostics: *std.Io.Writer) !MlirModuleArtifact {
    if (input.program_format != .stablehlo_text) {
        try writeStageFailure(diagnostics, .stablehlo_ingest, "program-format", "only StableHLO text ingest is implemented in this slice");
        return CompilePipelineError.InvalidStablehlo;
    }

    var artifact: MlirModuleArtifact = try createMlirModuleArtifact(diagnostics);
    errdefer artifact.deinit();

    artifact.module = mlir.mlirModuleCreateParse(artifact.context, mlirStringRef(input.program_bytes));
    if (mlir.mlirModuleIsNull(artifact.module)) {
        try writeStageFailure(diagnostics, .stablehlo_ingest, "stablehlo", "MLIR parser rejected module");
        return CompilePipelineError.InvalidStablehlo;
    }

    const module_op = mlir.mlirModuleGetOperation(artifact.module);
    if (!mlir.mlirOperationVerify(module_op)) {
        try writeStageFailure(diagnostics, .mlir_verify, "stablehlo", "MLIR verifier rejected module");
        return CompilePipelineError.InvalidStablehlo;
    }

    return artifact;
}

/// The first compiler-middle artifact is deliberately report-first. Even when
/// V0 pass execution is conservative, the compiler records the pass boundary,
/// source-provenance contract, and Shardy preservation contract instead of
/// hiding those facts behind later backend planning.
pub fn buildMlirPassPipelineReport(
    allocator: std.mem.Allocator,
    input: *const CompileInput,
    artifact: MlirModuleArtifact,
    diagnostics: *std.Io.Writer,
) !MlirPassPipelineReport {
    if (mlir.mlirModuleIsNull(artifact.module)) {
        try writeStageFailure(diagnostics, .mlir_lowering_pass_pipeline, "mlir-module", "cannot build pass report for null module");
        return CompilePipelineError.InvalidStablehlo;
    }

    const has_shardy_metadata = textMentionsShardy(input.program_bytes);
    var records = try allocator.alloc(MlirPassRecord, 5);
    errdefer allocator.free(records);
    const input_fingerprint = stableFingerprint(input.program_bytes);

    records[0] = .{
        .index = 0,
        .pass_name = v0_compiler_pass_contracts[0].stable_name,
        .status = .ok,
        .input_fingerprint = input_fingerprint,
        .output_fingerprint = input_fingerprint,
        .preserves_source_provenance = true,
        .preserves_shardy_metadata = true,
        .reason = "StableHLO text parsed through the MLIR C API before PjRTx owns a typed graph",
    };
    records[1] = .{
        .index = 1,
        .pass_name = v0_compiler_pass_contracts[1].stable_name,
        .status = .ok,
        .input_fingerprint = input_fingerprint,
        .output_fingerprint = input_fingerprint,
        .preserves_source_provenance = true,
        .preserves_shardy_metadata = true,
        .reason = "MLIR verifier accepted the module before PjRTx graph import",
    };
    records[2] = .{
        .index = 2,
        .pass_name = v0_compiler_pass_contracts[2].stable_name,
        .status = .ok,
        .input_fingerprint = input_fingerprint,
        .output_fingerprint = input_fingerprint,
        .preserves_source_provenance = true,
        .preserves_shardy_metadata = true,
        .reason = "StableHLO collectives are preserved for typed graph payload import; unsupported algorithms fail later at collective_algorithm_select",
    };
    records[3] = .{
        .index = 3,
        .pass_name = v0_compiler_pass_contracts[3].stable_name,
        .status = .ok,
        .input_fingerprint = input_fingerprint,
        .output_fingerprint = input_fingerprint,
        .preserves_source_provenance = true,
        .preserves_shardy_metadata = true,
        .reason = "V0 records the canonicalization/CSE boundary before wiring transform passes",
    };
    records[4] = .{
        .index = 4,
        .pass_name = v0_compiler_pass_contracts[4].stable_name,
        .status = if (has_shardy_metadata and input.compile_options.use_shardy_partitioner) .ok else .skipped,
        .input_fingerprint = input_fingerprint,
        .output_fingerprint = input_fingerprint,
        .preserves_source_provenance = true,
        .preserves_shardy_metadata = has_shardy_metadata,
        .reason = if (has_shardy_metadata)
            "Shardy metadata is present and must remain visible to later sharding and collective stages"
        else
            "No Shardy metadata found in this V0 fixture",
    };

    return .{
        .allocator = allocator,
        .has_shardy_metadata = has_shardy_metadata,
        .shardy_requested = input.compile_options.use_shardy_partitioner,
        .records = records,
    };
}

pub fn writeMlirPassPipelineReport(report: MlirPassPipelineReport, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    try writer.print("mlir_pass_pipeline shardy_present={} shardy_requested={}\n", .{ report.has_shardy_metadata, report.shardy_requested });
    for (report.records) |record| {
        try writer.print(
            "  pass.{d} name={s} status={s} input={d} output={d} source_provenance={} shardy_preserved={} reason={s}\n",
            .{
                record.index,
                record.pass_name,
                @tagName(record.status),
                record.input_fingerprint,
                record.output_fingerprint,
                record.preserves_source_provenance,
                record.preserves_shardy_metadata,
                record.reason,
            },
        );
    }
}

pub fn planPlacement(
    allocator: std.mem.Allocator,
    graph: GraphModule,
    target: SelectedTarget,
    diagnostics: *std.Io.Writer,
) !PlacementPlan {
    try verifyGraphModule(graph, diagnostics);

    const result_memory = requiredMemorySpace(target.description, switch (target.description.kind) {
        .metal_v0 => .device_unified,
        .npu_v0 => .device_hbm,
    }, diagnostics) catch return CompilePipelineError.InvalidLowering;
    const tile_memory = optionalMemorySpace(target.description, .local_sram);

    var records: std.ArrayList(PlacementRecord) = .empty;
    errdefer {
        for (records.items) |record| {
            allocator.free(record.output_value_ids);
            allocator.free(record.logical_tile_shape);
        }
        records.deinit(allocator);
    }

    for (graph.instructions) |instruction| {
        if (instruction.kind == .return_) continue;
        if (instruction.outputs.len == 0) {
            try writeStageFailure(diagnostics, .placement_planning, "instruction", "executable instruction has no output for placement");
            return CompilePipelineError.InvalidLowering;
        }
        const output = try requireGraphValue(graph, instruction.outputs[0], diagnostics);
        const output_ids = try copyValueIds(allocator, instruction.outputs);
        errdefer allocator.free(output_ids);
        const tile_shape = try tileShapeForInstruction(allocator, output.ty, instruction, target.description);
        errdefer allocator.free(tile_shape);

        try records.append(allocator, .{
            .index = std.math.cast(u32, records.items.len) orelse unreachable,
            .graph_instruction_id = instruction.id,
            .output_value_ids = output_ids,
            .layout = output.ty.layout,
            .logical_tile_shape = tile_shape,
            .result_memory_space_id = result_memory,
            .tile_memory_space_id = tileMemoryForInstruction(instruction, target.description.kind, tile_memory),
            .reason = placementReasonForInstruction(instruction, target.description.kind),
        });
    }

    return .{ .allocator = allocator, .records = try records.toOwnedSlice(allocator) };
}

pub fn writePlacementPlan(plan: PlacementPlan, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    try writer.writeAll("placement_plan\n");
    for (plan.records) |record| {
        try writer.print(
            "  placement.{d} instruction={d} outputs=",
            .{ record.index, record.graph_instruction_id.index },
        );
        try writeValueIdList(writer, record.output_value_ids);
        try writer.print(" layout={s} tile=", .{@tagName(record.layout)});
        try writeI64List(writer, record.logical_tile_shape);
        try writer.print(" result_memory={d} tile_memory=", .{record.result_memory_space_id});
        if (record.tile_memory_space_id) |memory_space_id| {
            try writer.print("{d}", .{memory_space_id});
        } else {
            try writer.writeAll("none");
        }
        try writer.print(" reason={s}\n", .{record.reason});
    }
}

pub fn planCollectives(graph: GraphModule, target: SelectedTarget, diagnostics: *std.Io.Writer) !CollectivePlan {
    try verifyGraphModule(graph, diagnostics);

    const lowered_count: u32 = 0;
    var unsupported_count: u32 = 0;
    var estimated_bytes: u128 = 0;
    var first_collective: ?compiler_facts.GraphInstruction = null;
    for (graph.instructions) |instruction| {
        if (instruction.kind != .collective) continue;
        if (first_collective == null) first_collective = instruction;
        unsupported_count += 1;
        estimated_bytes += try bytesReadForInstruction(graph, instruction, diagnostics);
        estimated_bytes += try bytesWrittenForInstruction(graph, instruction, diagnostics);
    }

    if (first_collective) |instruction| {
        const spec = switch (instruction.payload) {
            .collective => |payload| payload,
            else => unreachable,
        };
        try diagnostics.print(
            "pass=collective_algorithm_select feature={s} decision=unsupported algorithm=none rejected_algorithms=direct,ring,tree,split backend={s} instruction={d} replica_groups={d} group_size={d} channel_id=",
            .{
                collectiveStablehloName(spec.op),
                @tagName(target.description.kind),
                instruction.id.index,
                spec.replica_group_count,
                spec.replica_group_size,
            },
        );
        if (spec.channel_id) |channel_id| {
            try diagnostics.print("{d}", .{channel_id});
        } else {
            try diagnostics.writeAll("none");
        }
        try diagnostics.writeAll(" channel_type=");
        if (spec.channel_type) |channel_type| {
            try diagnostics.print("{d}", .{channel_type});
        } else {
            try diagnostics.writeAll("none");
        }
        try diagnostics.writeAll(" reason=V0 imports collective graph payloads but has no executable collective lowering path\n");
        return .{
            .index = 0,
            .decision = .unsupported,
            .algorithm = .none,
            .checked_graph_instruction_count = std.math.cast(u32, graph.instructions.len) orelse unreachable,
            .lowered_collective_count = lowered_count,
            .unsupported_collective_count = unsupported_count,
            .estimated_bytes = estimated_bytes,
            .estimated_latency_ns = null,
            .reason = "Collective graph payloads were imported, but V0 has no executable collective algorithm for this target yet",
        };
    }

    return .{
        .index = 0,
        .decision = .no_collectives,
        .algorithm = .none,
        .checked_graph_instruction_count = std.math.cast(u32, graph.instructions.len) orelse unreachable,
        .lowered_collective_count = lowered_count,
        .unsupported_collective_count = 0,
        .estimated_bytes = 0,
        .estimated_latency_ns = null,
        .reason = "No collective graph instructions are present; collective_algorithm_select records algorithm=none",
    };
}

pub fn verifyCollectiveGroupChannel(
    graph: GraphModule,
    target: SelectedTarget,
    compile_options: CompileOptions,
    diagnostics: *std.Io.Writer,
) !void {
    try verifyGraphModule(graph, diagnostics);
    const total_participants = totalCollectiveParticipants(compile_options);
    for (graph.instructions) |instruction| {
        if (instruction.kind != .collective) continue;
        if (!targetHasCollectiveUnit(target.description)) {
            try diagnostics.print(
                "pass=collective_group_channel_verify feature=collective-engine reason=target has no collective execution unit backend={s} instruction={d}\n",
                .{ @tagName(target.description.kind), instruction.id.index },
            );
            return CompilePipelineError.UnsupportedTargetFeature;
        }
        const payload = switch (instruction.payload) {
            .collective => |collective| collective,
            else => unreachable,
        };
        const expected_participants_u32 = std.math.mul(u32, payload.replica_group_count, payload.replica_group_size) catch {
            try diagnostics.print(
                "pass=collective_group_channel_verify feature=replica-groups reason=replica group payload shape overflows instruction={d}\n",
                .{instruction.id.index},
            );
            return CompilePipelineError.InvalidLowering;
        };
        const expected_participants: usize = std.math.cast(usize, expected_participants_u32) orelse {
            try diagnostics.print(
                "pass=collective_group_channel_verify feature=replica-groups reason=replica group payload shape does not fit host instruction={d}\n",
                .{instruction.id.index},
            );
            return CompilePipelineError.InvalidLowering;
        };
        if (payload.replica_groups.len != expected_participants) {
            try diagnostics.print(
                "pass=collective_group_channel_verify feature=replica-groups reason=replica group payload shape mismatch instruction={d}\n",
                .{instruction.id.index},
            );
            return CompilePipelineError.InvalidLowering;
        }
        for (payload.replica_groups) |participant| {
            if (participant >= total_participants) {
                try diagnostics.print(
                    "pass=collective_group_channel_verify feature=replica-groups reason=participant outside compile topology instruction={d} participant={d} participants={d}\n",
                    .{ instruction.id.index, participant, total_participants },
                );
                return CompilePipelineError.InvalidLowering;
            }
        }
        if (hasDuplicateParticipant(payload.replica_groups)) {
            try diagnostics.print(
                "pass=collective_group_channel_verify feature=replica-groups reason=duplicate participant in replica groups instruction={d}\n",
                .{instruction.id.index},
            );
            return CompilePipelineError.InvalidLowering;
        }
        if ((payload.channel_id == null) != (payload.channel_type == null)) {
            try diagnostics.print(
                "pass=collective_group_channel_verify feature=channel reason=channel handle is incomplete instruction={d}\n",
                .{instruction.id.index},
            );
            return CompilePipelineError.InvalidLowering;
        }
        if (payload.uses_token) {
            try diagnostics.print(
                "pass=collective_group_channel_verify feature=token reason=tokenized collectives are not schedulable in V0 instruction={d}\n",
                .{instruction.id.index},
            );
            return CompilePipelineError.InvalidLowering;
        }
    }
}

/// Graph import is intentionally typed and narrow for V0. Unsupported StableHLO
/// operations fail here, before target legality or backend binding can pretend
/// the program is executable.
pub fn importGraphFromMlir(allocator: std.mem.Allocator, artifact: MlirModuleArtifact, diagnostics: *std.Io.Writer) !GraphModule {
    var builder: GraphImportBuilder = .{
        .allocator = allocator,
        .diagnostics = diagnostics,
    };
    errdefer builder.deinitPartial();

    const module_op = mlir.mlirModuleGetOperation(artifact.module);
    const imported_any = try builder.importModule(module_op);
    if (!imported_any) {
        try writeStageFailure(diagnostics, .graph_import, "program", "module has no importable function body");
        return CompilePipelineError.InvalidStablehlo;
    }

    return try builder.finish();
}

/// Graph verification is where imported StableHLO stops being merely parseable
/// and becomes compiler-owned data with explicit provenance and shape contracts.
pub fn verifyGraphModule(graph: GraphModule, diagnostics: *std.Io.Writer) !void {
    for (graph.sources, 0..) |source, index| {
        const expected_index: u32 = std.math.cast(u32, index) orelse unreachable;
        if (source.id.index != expected_index or source.op_name.len == 0) {
            try writeStageFailure(diagnostics, .graph_verify, "source", "source provenance is malformed");
            return CompilePipelineError.InvalidGraph;
        }
    }

    for (graph.values, 0..) |value, index| {
        const expected_index: u32 = std.math.cast(u32, index) orelse unreachable;
        if (value.id.index != expected_index) {
            try writeStageFailure(diagnostics, .graph_verify, "value", "graph value id is not canonical");
            return CompilePipelineError.InvalidGraph;
        }
        compiler_facts.TensorFacts.validate(value.ty, diagnostics) catch {
            try writeStageFailure(diagnostics, .graph_verify, "tensor-type", "graph value has invalid tensor type");
            return CompilePipelineError.InvalidGraph;
        };
        if (value.source) |source| try verifySourceInGraph(graph, source, diagnostics);
    }

    for (graph.instructions, 0..) |instruction, index| {
        const expected_index: u32 = std.math.cast(u32, index) orelse unreachable;
        if (instruction.id.index != expected_index) {
            try writeStageFailure(diagnostics, .graph_verify, "instruction", "graph instruction id is not canonical");
            return CompilePipelineError.InvalidGraph;
        }
        try verifySourceInGraph(graph, instruction.source, diagnostics);
        for (instruction.inputs) |input| _ = try requireGraphValue(graph, input, diagnostics);
        for (instruction.outputs) |output| _ = try requireGraphValue(graph, output, diagnostics);

        switch (instruction.payload) {
            .dot_general => |payload| try verifyDotGeneral(graph, instruction, payload, diagnostics),
            .elementwise_unary => try verifyElementwiseUnary(graph, instruction, diagnostics),
            .elementwise_binary => try verifyElementwiseBinary(graph, instruction, diagnostics),
            .broadcast => |payload| try verifyBroadcast(graph, instruction, payload, diagnostics),
            .reshape => try verifyReshape(graph, instruction, diagnostics),
            .transpose => |payload| try verifyTranspose(graph, instruction, payload, diagnostics),
            .collective => |payload| try verifyCollective(graph, instruction, payload, diagnostics),
            .return_ => try verifyReturn(instruction, diagnostics),
        }
    }
}

/// Algebraic normalization is deliberately conservative in V0. It only removes
/// identity value rewrites whose input and output tensor types are already
/// identical and whose dimension maps are exactly `0..rank`.
pub fn normalizeGraphV0(
    allocator: std.mem.Allocator,
    graph: GraphModule,
    diagnostics: *std.Io.Writer,
) !GraphNormalizationResult {
    try verifyGraphModule(graph, diagnostics);

    const sources = try copySources(allocator, graph.sources);
    errdefer freeSources(allocator, sources);
    const values = try copyValues(allocator, graph.values, sources);
    errdefer freeValues(allocator, values);

    const value_replacements = try allocator.alloc(compiler_facts.GraphValueId, graph.values.len);
    defer allocator.free(value_replacements);
    for (value_replacements, 0..) |*replacement, index| {
        replacement.* = .{ .index = std.math.cast(u32, index) orelse unreachable };
    }

    var instructions: std.ArrayList(compiler_facts.GraphInstruction) = .empty;
    errdefer freeInstructionList(allocator, &instructions);
    var records: std.ArrayList(compiler_facts.GraphRewriteRecord) = .empty;
    errdefer records.deinit(allocator);

    for (graph.instructions) |instruction| {
        if (try identityValueRewriteApplies(graph, instruction, diagnostics)) {
            const input_id = remappedValueId(value_replacements, instruction.inputs[0]);
            const output_id = instruction.outputs[0];
            const output_index: usize = std.math.cast(usize, output_id.index) orelse unreachable;
            value_replacements[output_index] = input_id;
            const pass_name = identityValueRewritePassName(instruction);
            try records.append(allocator, .{
                .index = std.math.cast(u32, records.items.len) orelse unreachable,
                .pass_name = pass_name,
                .decision = .applied,
                .input_instruction_id = instruction.id,
                .output_instruction_id = null,
                .replaced_value_id = output_id,
                .replacement_value_id = input_id,
                .reason = identityValueRewriteReason(instruction),
            });
            continue;
        }

        const copied_instruction = try copyInstructionWithValueRewrites(
            allocator,
            instruction,
            .{ .index = std.math.cast(u32, instructions.items.len) orelse unreachable },
            value_replacements,
            sources,
        );
        errdefer freeInstruction(allocator, copied_instruction);
        const copied_id = copied_instruction.id;
        try instructions.append(allocator, copied_instruction);

        if (instruction.kind == .broadcast) {
            try records.append(allocator, .{
                .index = std.math.cast(u32, records.items.len) orelse unreachable,
                .pass_name = "broadcast_simplify",
                .decision = .rejected,
                .input_instruction_id = instruction.id,
                .output_instruction_id = copied_id,
                .replaced_value_id = if (instruction.outputs.len > 0) instruction.outputs[0] else null,
                .replacement_value_id = null,
                .reason = "broadcast changes rank, shape, layout, dtype, or dimension mapping",
            });
        }
        if (instruction.kind == .reshape or instruction.kind == .transpose) {
            try records.append(allocator, .{
                .index = std.math.cast(u32, records.items.len) orelse unreachable,
                .pass_name = "reshape_transpose_fold",
                .decision = .rejected,
                .input_instruction_id = instruction.id,
                .output_instruction_id = copied_id,
                .replaced_value_id = if (instruction.outputs.len > 0) instruction.outputs[0] else null,
                .replacement_value_id = null,
                .reason = "reshape or transpose is not an identity mapping",
            });
        }
    }

    const instruction_slice = try instructions.toOwnedSlice(allocator);
    errdefer freeInstructions(allocator, instruction_slice);
    const record_slice = try records.toOwnedSlice(allocator);
    errdefer allocator.free(record_slice);

    var normalized: GraphModule = .{
        .allocator = allocator,
        .sources = sources,
        .values = values,
        .instructions = instruction_slice,
    };
    errdefer normalized.deinit();
    try verifyGraphModule(normalized, diagnostics);

    return .{
        .allocator = allocator,
        .graph = normalized,
        .records = record_slice,
    };
}

/// Target selection is the first place where abstract program facts meet a
/// concrete hardware model. Unknown performance numbers are allowed, but a
/// malformed target description is not allowed to reach legality checks.
pub fn selectTarget(kind: target_pkg.TargetKind, diagnostics: *std.Io.Writer) !SelectedTarget {
    const target: target_pkg.TargetDescription = switch (kind) {
        .metal_v0 => metalTarget(),
        .npu_v0 => npuTarget(),
    };
    target_pkg.validateTargetDescription(target, diagnostics) catch |err| switch (err) {
        target_pkg.ValidationError.InvalidTargetDescription => {
            try writeStageFailure(diagnostics, .target_selection, "target", "target description failed validation");
            return CompilePipelineError.UnsupportedTargetFeature;
        },
        else => return err,
    };
    return .{ .description = target };
}

/// Backend binding verification is deliberately separate from core report
/// validation. Core checks that links exist; the compiler checks executable
/// policy, including exactly one binding for each backend command.
pub fn verifyBackendBindings(report: core.TraceReport, target: SelectedTarget, diagnostics: *std.Io.Writer) !void {
    for (report.schedule_commands) |command| {
        if (command.kind != .backend_execute) continue;

        var binding_count: u32 = 0;
        for (report.backend_bindings) |binding| {
            if (!binding.command_id.eql(command.id)) continue;
            binding_count += 1;
            if (!backendKindMatchesTarget(binding.backend_kind, target.description.kind)) {
                try diagnostics.print("pass=backend_binding_verify feature=backend-kind reason=binding backend does not match selected target command={d} binding={d}\n", .{ command.id.index, binding.id.index });
                return CompilePipelineError.InvalidBackendBinding;
            }
        }

        if (binding_count == 0) {
            try diagnostics.print("pass=backend_binding_verify feature=backend-binding reason=backend command has no binding command={d}\n", .{command.id.index});
            return CompilePipelineError.InvalidBackendBinding;
        }
        if (binding_count > 1) {
            try diagnostics.print("pass=backend_binding_verify feature=backend-binding reason=backend command has multiple bindings command={d}\n", .{command.id.index});
            return CompilePipelineError.InvalidBackendBinding;
        }
    }
}

/// Executable creation remains a view until the runtime owns real command
/// buffers. The important contract already holds: only a report that validates
/// and passes backend binding verification can become executable-shaped.
pub fn createExecutableView(report: core.TraceReport, target: SelectedTarget, diagnostics: *std.Io.Writer) !CompiledExecutableView {
    core.validateTraceReport(report, diagnostics) catch |err| switch (err) {
        core.ValidationError.InvalidTraceReport, core.ValidationError.InvalidTensorType => {
            try writeStageFailure(diagnostics, .schedule_verify, "trace-report", "trace report failed validation");
            return CompilePipelineError.InvalidInput;
        },
        else => return err,
    };
    try verifySchedule(report, diagnostics);
    try verifyBackendBindings(report, target, diagnostics);
    return .{ .report = report };
}

/// Schedule verification enforces compiler-owned execution invariants that are
/// stricter than structural report validation. Runtime checks the same shape
/// again, but invalid command order should not become executable-shaped.
pub fn verifySchedule(report: core.TraceReport, diagnostics: *std.Io.Writer) !void {
    for (report.schedule_commands, 0..) |command, index| {
        const expected_index: u32 = std.math.cast(u32, index) orelse unreachable;
        if (command.id.index != expected_index) {
            try diagnostics.print("pass=schedule_verify feature=command reason=command ID order mismatch expected={d} actual={d}\n", .{ expected_index, command.id.index });
            return CompilePipelineError.InvalidSchedule;
        }
        for (command.dependencies) |dependency| {
            if (dependency.command_id.index >= command.id.index) {
                try diagnostics.print(
                    "pass=schedule_verify feature=dependency reason=command dependency must be earlier command={d} dependency={d}\n",
                    .{ command.id.index, dependency.command_id.index },
                );
                return CompilePipelineError.InvalidSchedule;
            }
        }
        if (command.kind == .backend_execute and command.dependencies.len == 0) {
            try diagnostics.print("pass=schedule_verify feature=dependency reason=backend command must depend on earlier transfer or wait command={d}\n", .{command.id.index});
            return CompilePipelineError.InvalidSchedule;
        }
    }
}

/// The V0 planner intentionally creates one explicit transfer/execute/transfer
/// schedule. It is not clever yet; it is a traceable baseline that proves every
/// supported graph instruction has cost, lowering, schedule, and backend edges.
/// Fusion planning is MLIR-owned; callers must provide the MLIR session that
/// produced the graph.
pub fn planV0TraceReportFromMlirSession(
    allocator: std.mem.Allocator,
    graph: GraphModule,
    target: SelectedTarget,
    compile_options: CompileOptions,
    state_session: *mlir_state.MlirSession,
    diagnostics: *std.Io.Writer,
) !PlannedTraceReport {
    return planV0TraceReportWithMlirState(allocator, graph, target, compile_options, state_session, diagnostics);
}

fn planV0TraceReportWithMlirState(
    allocator: std.mem.Allocator,
    graph: GraphModule,
    target: SelectedTarget,
    compile_options: CompileOptions,
    state_session: *mlir_state.MlirSession,
    diagnostics: *std.Io.Writer,
) !PlannedTraceReport {
    try verifyGraphModule(graph, diagnostics);
    try mlir_state.attachTarget(
        state_session,
        target.description,
        compile_options.num_replicas,
        compile_options.num_partitions,
        diagnostics,
    );
    try verifyTargetLegality(graph, target, diagnostics);
    try mlir_state.markTargetLegal(state_session, diagnostics);
    try mlir_state.runFusionCandidateDiscoveryExternalPass(state_session, diagnostics);

    var builder: TracePlanBuilder = .{
        .allocator = allocator,
        .graph = graph,
        .target = target,
        .compile_options = compile_options,
        .mlir_session = state_session,
        .diagnostics = diagnostics,
    };
    errdefer builder.deinitPartial();

    try builder.addCompilerMiddlePlans();
    try builder.addInstructionCosts();
    try builder.commitLoweringRecordsToMlir();
    try builder.addMemoryTrafficRecords();
    try builder.commitPerformanceFactsToMlir();
    const parameter_ids = try builder.parameterValueIds();
    defer allocator.free(parameter_ids);
    const output_ids = try builder.returnValueIds();
    defer allocator.free(output_ids);
    try builder.addSchedule(parameter_ids, output_ids);
    try builder.addKernelCodegenRecords();
    try builder.commitKernelCodegenRecordsToMlir();
    try builder.commitScheduleRecordsToMlir();
    try builder.verifyTileLegalityRecords();
    try builder.addBackendBinding();
    try builder.commitBackendBindingsToMlir();
    try builder.commitExecutableReadinessToMlir();
    try builder.addExplainRecords();

    var planned: PlannedTraceReport = try builder.finish();
    errdefer planned.deinit();
    _ = try createExecutableView(planned.report, target, diagnostics);
    return planned;
}

/// Target legality joins graph facts with backend capability declarations.
/// Failing here is the no-fallback contract: unsupported work never reaches
/// lowering, scheduling, allocation, or runtime profiling.
pub fn verifyTargetLegality(graph: GraphModule, target: SelectedTarget, diagnostics: *std.Io.Writer) !void {
    const capabilities: CompilerBackendCapabilitySet = compilerCapabilitySet(backendKindForTarget(target.description.kind));
    for (graph.instructions) |instruction| {
        const feature = compilerFeatureForInstruction(instruction) orelse continue;
        const dtype = instructionDtype(graph, instruction) orelse {
            try diagnostics.print("pass=target_legality feature=backend-capability reason=instruction has no dtype instruction={d}\n", .{instruction.id.index});
            return CompilePipelineError.UnsupportedTargetFeature;
        };
        if (!compilerSupportsInstruction(capabilities, instruction, graph.values)) {
            try diagnostics.print(
                "pass=target_legality feature=backend-capability reason=backend lacks support backend={s} op={s} dtype={s} instruction={d}\n",
                .{ @tagName(capabilities.kind), @tagName(feature), @tagName(dtype), instruction.id.index },
            );
            return CompilePipelineError.UnsupportedTargetFeature;
        }
    }
}

pub fn writeStageFailure(
    diagnostics: *std.Io.Writer,
    stage: CompilePipelineStage,
    feature: []const u8,
    reason: []const u8,
) std.Io.Writer.Error!void {
    try diagnostics.print("pass={s} feature={s} reason={s}\n", .{ @tagName(stage), feature, reason });
}

fn backendKindMatchesTarget(backend_kind: core.BackendKind, target_kind: target_pkg.TargetKind) bool {
    return switch (backend_kind) {
        .metal_v0 => target_kind == .metal_v0,
        .npu_v0 => target_kind == .npu_v0,
    };
}

fn instructionDtype(graph: GraphModule, instruction: compiler_facts.GraphInstruction) ?core.BufferType {
    const value_id = if (instruction.outputs.len > 0) instruction.outputs[0] else if (instruction.inputs.len > 0) instruction.inputs[0] else return null;
    const index: usize = std.math.cast(usize, value_id.index) orelse return null;
    if (index >= graph.values.len) return null;
    return graph.values[index].ty.element_type;
}

fn parsePositiveU32(value: []const u8, diagnostics: *std.Io.Writer) !u32 {
    const parsed = std.fmt.parseInt(u32, value, 10) catch {
        try writeStageFailure(diagnostics, .input_setup, "compile-options", "expected positive integer");
        return CompilePipelineError.InvalidInput;
    };
    if (parsed == 0) {
        try writeStageFailure(diagnostics, .input_setup, "compile-options", "expected positive integer");
        return CompilePipelineError.InvalidInput;
    }
    return parsed;
}

fn parseBool(value: []const u8, diagnostics: *std.Io.Writer) !bool {
    if (std.mem.eql(u8, value, "true")) return true;
    if (std.mem.eql(u8, value, "false")) return false;
    try writeStageFailure(diagnostics, .input_setup, "compile-options", "expected boolean");
    return CompilePipelineError.InvalidInput;
}

fn createMlirModuleArtifact(diagnostics: *std.Io.Writer) !MlirModuleArtifact {
    const registry = mlir.mlirDialectRegistryCreate();
    if (mlir.mlirDialectRegistryIsNull(registry)) {
        try writeStageFailure(diagnostics, .stablehlo_ingest, "mlir-registry", "failed to create MLIR dialect registry");
        return CompilePipelineError.InvalidStablehlo;
    }

    insertDialect(registry, mlir.mlirGetDialectHandle__func__());
    insertDialect(registry, mlir.mlirGetDialectHandle__shape__());
    insertDialect(registry, mlir.mlirGetDialectHandle__chlo__());
    insertDialect(registry, mlir.mlirGetDialectHandle__sdy__());
    insertDialect(registry, mlir.mlirGetDialectHandle__stablehlo__());
    mlir.pjrtxMlirRegisterFuncExtensions(registry);

    const context = mlir.mlirContextCreateWithRegistry(registry, false);
    if (mlir.mlirContextIsNull(context)) {
        mlir.mlirDialectRegistryDestroy(registry);
        try writeStageFailure(diagnostics, .stablehlo_ingest, "mlir-context", "failed to create MLIR context");
        return CompilePipelineError.InvalidStablehlo;
    }

    var artifact: MlirModuleArtifact = .{
        .registry = registry,
        .context = context,
        .module = mlir.mlirModuleCreateEmpty(mlir.mlirLocationUnknownGet(context)),
    };
    mlir.mlirModuleDestroy(artifact.module);
    artifact.module = mlir.MlirModule{ .ptr = null };

    mlir.mlirContextSetAllowUnregisteredDialects(context, false);
    mlir.mlirContextLoadAllAvailableDialects(context);
    try loadDialect(context, mlir.mlirGetDialectHandle__func__(), diagnostics);
    try loadDialect(context, mlir.mlirGetDialectHandle__shape__(), diagnostics);
    try loadDialect(context, mlir.mlirGetDialectHandle__chlo__(), diagnostics);
    try loadDialect(context, mlir.mlirGetDialectHandle__sdy__(), diagnostics);
    try loadDialect(context, mlir.mlirGetDialectHandle__stablehlo__(), diagnostics);
    return artifact;
}

fn mlirStringRef(text: []const u8) mlir.MlirStringRef {
    return mlir.mlirStringRefCreate(text.ptr, text.len);
}

fn insertDialect(registry: mlir.MlirDialectRegistry, handle: mlir.MlirDialectHandle) void {
    mlir.mlirDialectHandleInsertDialect(handle, registry);
}

fn loadDialect(context: mlir.MlirContext, handle: mlir.MlirDialectHandle, diagnostics: *std.Io.Writer) !void {
    const dialect = mlir.mlirDialectHandleLoadDialect(handle, context);
    if (mlir.mlirDialectIsNull(dialect)) {
        try writeStageFailure(diagnostics, .stablehlo_ingest, "mlir-dialect", "failed to load MLIR dialect");
        return CompilePipelineError.InvalidStablehlo;
    }
}

fn graphVerifyFailure(diagnostics: *std.Io.Writer, feature: []const u8, reason: []const u8) !void {
    try writeStageFailure(diagnostics, .graph_verify, feature, reason);
    return CompilePipelineError.InvalidGraph;
}

fn verifySourceInGraph(graph: GraphModule, source: compiler_facts.SourceRef, diagnostics: *std.Io.Writer) !void {
    const index: usize = std.math.cast(usize, source.id.index) orelse {
        try graphVerifyFailure(diagnostics, "source", "source id does not fit host index");
        unreachable;
    };
    if (index >= graph.sources.len) {
        try graphVerifyFailure(diagnostics, "source", "source id is out of bounds");
        unreachable;
    }
    const canonical = graph.sources[index];
    if (canonical.frontend != source.frontend or
        canonical.source_index != source.source_index or
        !std.mem.eql(u8, canonical.op_name, source.op_name))
    {
        try graphVerifyFailure(diagnostics, "source", "source reference does not match canonical source");
        unreachable;
    }
}

fn requireGraphValue(graph: GraphModule, id: compiler_facts.GraphValueId, diagnostics: *std.Io.Writer) !compiler_facts.GraphValue {
    const index: usize = std.math.cast(usize, id.index) orelse {
        try graphVerifyFailure(diagnostics, "value-ref", "graph value id does not fit host index");
        unreachable;
    };
    if (index >= graph.values.len) {
        try graphVerifyFailure(diagnostics, "value-ref", "graph value id is out of bounds");
        unreachable;
    }
    return graph.values[index];
}

fn verifyDotGeneral(
    graph: GraphModule,
    instruction: compiler_facts.GraphInstruction,
    payload: compiler_facts.DotGeneralSpec,
    diagnostics: *std.Io.Writer,
) !void {
    if (instruction.inputs.len != 2 or instruction.outputs.len != 1) {
        try graphVerifyFailure(diagnostics, "dot-general", "dot_general must have two inputs and one output");
        return;
    }

    const lhs = try requireGraphValue(graph, instruction.inputs[0], diagnostics);
    const rhs = try requireGraphValue(graph, instruction.inputs[1], diagnostics);
    const output = try requireGraphValue(graph, instruction.outputs[0], diagnostics);
    if (lhs.ty.dims.len != 2 or rhs.ty.dims.len != 2 or output.ty.dims.len != 2) {
        try graphVerifyFailure(diagnostics, "dot-general", "V0 dot_general requires rank-2 tensors");
        return;
    }

    const lhs_contracting: usize = std.math.cast(usize, payload.lhs_contracting_dimension) orelse {
        try graphVerifyFailure(diagnostics, "dot-general", "lhs contracting dimension does not fit host index");
        unreachable;
    };
    const rhs_contracting: usize = std.math.cast(usize, payload.rhs_contracting_dimension) orelse {
        try graphVerifyFailure(diagnostics, "dot-general", "rhs contracting dimension does not fit host index");
        unreachable;
    };
    if (lhs_contracting >= lhs.ty.dims.len or rhs_contracting >= rhs.ty.dims.len) {
        try graphVerifyFailure(diagnostics, "dot-general", "contracting dimension is out of bounds");
        return;
    }
    if (lhs.ty.element_type != rhs.ty.element_type or lhs.ty.element_type != output.ty.element_type) {
        try graphVerifyFailure(diagnostics, "dot-general", "dot_general operand dtypes must match");
        return;
    }
    if (lhs.ty.dims[lhs_contracting] != rhs.ty.dims[rhs_contracting]) {
        try graphVerifyFailure(diagnostics, "dot-general", "contracting dimensions have different sizes");
        return;
    }

    const lhs_batch: usize = if (lhs_contracting == 0) 1 else 0;
    const rhs_batch: usize = if (rhs_contracting == 0) 1 else 0;
    if (output.ty.dims[0] != lhs.ty.dims[lhs_batch] or output.ty.dims[1] != rhs.ty.dims[rhs_batch]) {
        try graphVerifyFailure(diagnostics, "dot-general", "output shape does not match V0 matrix product");
        return;
    }
}

fn verifyElementwiseUnary(graph: GraphModule, instruction: compiler_facts.GraphInstruction, diagnostics: *std.Io.Writer) !void {
    if (instruction.inputs.len != 1 or instruction.outputs.len != 1) {
        try graphVerifyFailure(diagnostics, "elementwise-unary", "unary op must have one input and one output");
        return;
    }
    const input = try requireGraphValue(graph, instruction.inputs[0], diagnostics);
    const output = try requireGraphValue(graph, instruction.outputs[0], diagnostics);
    if (!tensorTypesEqual(input.ty, output.ty)) {
        try graphVerifyFailure(diagnostics, "elementwise-unary", "unary op input and output types must match");
        return;
    }
}

fn verifyElementwiseBinary(graph: GraphModule, instruction: compiler_facts.GraphInstruction, diagnostics: *std.Io.Writer) !void {
    if (instruction.inputs.len != 2 or instruction.outputs.len != 1) {
        try graphVerifyFailure(diagnostics, "elementwise-binary", "binary op must have two inputs and one output");
        return;
    }
    const lhs = try requireGraphValue(graph, instruction.inputs[0], diagnostics);
    const rhs = try requireGraphValue(graph, instruction.inputs[1], diagnostics);
    const output = try requireGraphValue(graph, instruction.outputs[0], diagnostics);
    if (!tensorTypesEqual(lhs.ty, rhs.ty) or !tensorTypesEqual(lhs.ty, output.ty)) {
        try graphVerifyFailure(diagnostics, "elementwise-binary", "binary op input and output types must match");
        return;
    }
}

fn verifyBroadcast(
    graph: GraphModule,
    instruction: compiler_facts.GraphInstruction,
    payload: compiler_facts.BroadcastSpec,
    diagnostics: *std.Io.Writer,
) !void {
    if (instruction.inputs.len != 1 or instruction.outputs.len != 1) {
        try graphVerifyFailure(diagnostics, "broadcast", "broadcast must have one input and one output");
        return;
    }
    const input = try requireGraphValue(graph, instruction.inputs[0], diagnostics);
    const output = try requireGraphValue(graph, instruction.outputs[0], diagnostics);
    if (input.ty.element_type != output.ty.element_type or input.ty.layout != output.ty.layout) {
        try graphVerifyFailure(diagnostics, "broadcast", "broadcast input and output element layout must match");
        return;
    }
    if (payload.dimensions.len != input.ty.dims.len) {
        try graphVerifyFailure(diagnostics, "broadcast", "broadcast dimensions must map every input dimension");
        return;
    }
    for (payload.dimensions, 0..) |output_dimension_u32, input_dimension| {
        const output_dimension: usize = std.math.cast(usize, output_dimension_u32) orelse {
            try graphVerifyFailure(diagnostics, "broadcast", "broadcast dimension does not fit host index");
            unreachable;
        };
        if (output_dimension >= output.ty.dims.len) {
            try graphVerifyFailure(diagnostics, "broadcast", "broadcast dimension is out of bounds");
            return;
        }
        if (input.ty.dims[input_dimension] != output.ty.dims[output_dimension]) {
            try graphVerifyFailure(diagnostics, "broadcast", "broadcast mapped dimension size does not match");
            return;
        }
    }
}

fn verifyReshape(graph: GraphModule, instruction: compiler_facts.GraphInstruction, diagnostics: *std.Io.Writer) !void {
    if (instruction.inputs.len != 1 or instruction.outputs.len != 1) {
        try graphVerifyFailure(diagnostics, "reshape", "reshape must have one input and one output");
        return;
    }
    const input = try requireGraphValue(graph, instruction.inputs[0], diagnostics);
    const output = try requireGraphValue(graph, instruction.outputs[0], diagnostics);
    if (input.ty.element_type != output.ty.element_type or input.ty.layout != output.ty.layout) {
        try graphVerifyFailure(diagnostics, "reshape", "reshape input and output element layout must match");
        return;
    }
    if (tensorElements(input.ty) != tensorElements(output.ty)) {
        try graphVerifyFailure(diagnostics, "reshape", "reshape must preserve element count");
        return;
    }
}

fn verifyTranspose(
    graph: GraphModule,
    instruction: compiler_facts.GraphInstruction,
    payload: compiler_facts.TransposeSpec,
    diagnostics: *std.Io.Writer,
) !void {
    if (instruction.inputs.len != 1 or instruction.outputs.len != 1) {
        try graphVerifyFailure(diagnostics, "transpose", "transpose must have one input and one output");
        return;
    }
    const input = try requireGraphValue(graph, instruction.inputs[0], diagnostics);
    const output = try requireGraphValue(graph, instruction.outputs[0], diagnostics);
    if (input.ty.element_type != output.ty.element_type or input.ty.layout != output.ty.layout) {
        try graphVerifyFailure(diagnostics, "transpose", "transpose input and output element layout must match");
        return;
    }
    if (payload.permutation.len != input.ty.dims.len or output.ty.dims.len != input.ty.dims.len) {
        try graphVerifyFailure(diagnostics, "transpose", "transpose permutation must match rank");
        return;
    }
    var seen = [_]bool{false} ** 16;
    if (payload.permutation.len > seen.len) {
        try graphVerifyFailure(diagnostics, "transpose", "V0 transpose rank limit exceeded");
        return;
    }
    for (payload.permutation, 0..) |input_dimension_u32, output_dimension| {
        const input_dimension: usize = std.math.cast(usize, input_dimension_u32) orelse {
            try graphVerifyFailure(diagnostics, "transpose", "transpose dimension does not fit host index");
            unreachable;
        };
        if (input_dimension >= input.ty.dims.len or seen[input_dimension]) {
            try graphVerifyFailure(diagnostics, "transpose", "transpose permutation is invalid");
            return;
        }
        seen[input_dimension] = true;
        if (output.ty.dims[output_dimension] != input.ty.dims[input_dimension]) {
            try graphVerifyFailure(diagnostics, "transpose", "transpose output shape does not match permutation");
            return;
        }
    }
}

fn verifyCollective(
    graph: GraphModule,
    instruction: compiler_facts.GraphInstruction,
    payload: compiler_facts.CollectiveSpec,
    diagnostics: *std.Io.Writer,
) !void {
    if (instruction.inputs.len != 1 or instruction.outputs.len != 1) {
        try graphVerifyFailure(diagnostics, "collective", "V0 collective payloads require one tensor input and one tensor output");
        return;
    }
    const input = try requireGraphValue(graph, instruction.inputs[0], diagnostics);
    const output = try requireGraphValue(graph, instruction.outputs[0], diagnostics);
    if (!tensorTypesEqual(input.ty, output.ty)) {
        try graphVerifyFailure(diagnostics, "collective", "collective input and output tensor types must match");
        return;
    }
    if (payload.op != .all_reduce or payload.reduction != .add) {
        try graphVerifyFailure(diagnostics, "collective", "V0 only imports add all_reduce collective payloads");
        return;
    }
    if (payload.replica_group_count == 0 or payload.replica_group_size == 0) {
        try graphVerifyFailure(diagnostics, "collective", "collective replica groups must be explicit");
        return;
    }
    if (payload.uses_token) {
        try graphVerifyFailure(diagnostics, "collective", "tokenized collectives are not represented in V0 graph payloads");
        return;
    }
}

fn verifyReturn(instruction: compiler_facts.GraphInstruction, diagnostics: *std.Io.Writer) !void {
    if (instruction.outputs.len != 0) {
        try graphVerifyFailure(diagnostics, "return", "return must not define graph values");
        return;
    }
}

fn tensorTypesEqual(lhs: compiler_facts.TensorType, rhs: compiler_facts.TensorType) bool {
    return lhs.element_type == rhs.element_type and
        lhs.layout == rhs.layout and
        std.mem.eql(i64, lhs.dims, rhs.dims);
}

const TracePlanBuilder = struct {
    allocator: std.mem.Allocator,
    graph: GraphModule,
    target: SelectedTarget,
    compile_options: CompileOptions,
    mlir_session: *mlir_state.MlirSession,
    diagnostics: *std.Io.Writer,
    fusion_group_strings_owned: bool = false,
    cost_ledger: std.ArrayList(compiler_facts.CostLedgerEntry) = .empty,
    lowering_records: std.ArrayList(compiler_facts.LoweringRecord) = .empty,
    memory_traffic_records: std.ArrayList(compiler_facts.MemoryTrafficRecord) = .empty,
    performance_strings_owned: bool = false,
    schedule_overlap_records: std.ArrayList(core.ScheduleOverlapRecord) = .empty,
    lowering_strings_owned: bool = false,
    schedule_overlap_reason_strings_owned: bool = false,
    schedule_commands: std.ArrayList(core.ScheduleCommand) = .empty,
    kernel_codegen_records: std.ArrayList(core.KernelCodegenRecord) = .empty,
    kernel_codegen_strings_owned: bool = false,
    backend_bindings: std.ArrayList(core.BackendBinding) = .empty,
    backend_binding_strings_owned: bool = false,
    profile_events: std.ArrayList(core.ProfileEvent) = .empty,
    explain_records: std.ArrayList(core.ExplainRecord) = .empty,
    fusion_groups: []const compiler_facts.FusionGroup = &.{},
    placement_plan: ?PlacementPlan = null,
    placement_reason_strings_owned: bool = false,
    collective_plan_records: []const compiler_facts.CollectivePlanRecord = &.{},
    collective_reason_strings_owned: bool = false,

    fn deinitPartial(self: *TracePlanBuilder) void {
        freeOwnedFusionGroups(self.allocator, self.fusion_groups, self.fusion_group_strings_owned);
        if (self.placement_plan) |*plan| plan.deinit();
        if (self.collective_reason_strings_owned) {
            for (self.collective_plan_records) |record| self.allocator.free(record.reason);
        }
        if (self.collective_plan_records.len > 0) self.allocator.free(self.collective_plan_records);
        for (self.cost_ledger.items) |entry| {
            if (self.performance_strings_owned) {
                if (entry.source) |source| {
                    self.allocator.free(source.op_name);
                    self.allocator.free(source.location);
                }
                self.allocator.free(entry.formula);
                self.allocator.free(entry.approximation);
            }
            self.allocator.free(entry.graph_instruction_ids);
        }
        for (self.lowering_records.items) |record| {
            if (self.lowering_strings_owned) {
                self.allocator.free(record.reason);
                for (record.rejected_alternatives) |alternative| self.allocator.free(alternative);
                self.allocator.free(record.rejected_alternatives);
            }
            self.allocator.free(record.graph_instruction_ids);
            self.allocator.free(record.cost_ledger_ids);
        }
        for (self.memory_traffic_records.items) |record| {
            if (self.performance_strings_owned) self.allocator.free(record.reason);
            self.allocator.free(record.graph_instruction_ids);
            self.allocator.free(record.cost_ledger_ids);
        }
        if (self.schedule_overlap_reason_strings_owned) {
            for (self.schedule_overlap_records.items) |record| self.allocator.free(record.reason);
        }
        self.schedule_overlap_records.deinit(self.allocator);
        for (self.schedule_commands.items) |command| {
            self.allocator.free(command.inputs);
            self.allocator.free(command.outputs);
            self.allocator.free(command.dependencies);
            self.allocator.free(command.lowering_record_ids);
            self.allocator.free(command.cost_ledger_ids);
        }
        for (self.kernel_codegen_records.items) |record| {
            if (self.kernel_codegen_strings_owned) {
                self.allocator.free(record.operation);
                self.allocator.free(record.reason);
            }
            self.allocator.free(record.logical_tile_shape);
            self.allocator.free(record.external_input_ids);
            self.allocator.free(record.external_output_ids);
            self.allocator.free(record.intermediate_value_ids);
            self.allocator.free(record.graph_instruction_ids);
            self.allocator.free(record.cost_ledger_ids);
            self.allocator.free(record.memory_traffic_ids);
        }
        for (self.backend_bindings.items) |binding| {
            if (self.backend_binding_strings_owned) self.allocator.free(binding.backend_operation);
            self.allocator.free(binding.graph_instruction_ids);
            self.allocator.free(binding.cost_ledger_ids);
        }
        for (self.explain_records.items) |explain| {
            self.allocator.free(explain.source_refs);
            self.allocator.free(explain.cost_ledger_ids);
            self.allocator.free(explain.profile_event_ids);
        }
        self.cost_ledger.deinit(self.allocator);
        self.lowering_records.deinit(self.allocator);
        self.memory_traffic_records.deinit(self.allocator);
        self.schedule_commands.deinit(self.allocator);
        self.kernel_codegen_records.deinit(self.allocator);
        self.backend_bindings.deinit(self.allocator);
        self.profile_events.deinit(self.allocator);
        self.explain_records.deinit(self.allocator);
    }

    fn addCompilerMiddlePlans(self: *TracePlanBuilder) !void {
        try mlir_state.planFusionFromCandidates(self.mlir_session, self.diagnostics);
        self.fusion_groups = try mlir_state.extractFusionGroups(self.allocator, self.mlir_session, self.diagnostics);
        self.fusion_group_strings_owned = true;
        var placement_plan: PlacementPlan = try planPlacement(self.allocator, self.graph, self.target, self.diagnostics);
        defer placement_plan.deinit();
        try mlir_state.commitPlacementPlan(self.mlir_session, placement_plan.records, self.diagnostics);
        self.placement_plan = .{
            .allocator = self.allocator,
            .records = try mlir_state.extractPlacementRecords(self.allocator, self.mlir_session, self.diagnostics),
            .reason_strings_owned = true,
        };
        self.placement_reason_strings_owned = true;
        try verifyCollectiveGroupChannel(self.graph, self.target, self.compile_options, self.diagnostics);
        const collective_plan: CollectivePlan = try planCollectives(self.graph, self.target, self.diagnostics);
        try mlir_state.commitCollectivePlan(self.mlir_session, &.{collective_plan}, self.diagnostics);
        self.collective_plan_records = try mlir_state.extractCollectivePlanRecords(self.allocator, self.mlir_session, self.diagnostics);
        self.collective_reason_strings_owned = true;
        for (self.collective_plan_records) |record| try self.verifyCollectivePlanExecutable(record);
    }

    fn verifyCollectivePlanExecutable(self: *TracePlanBuilder, plan: CollectivePlan) !void {
        if (plan.decision == .no_collectives or plan.decision == .selected) return;
        try self.diagnostics.print(
            "pass=collective_algorithm_select feature=collective reason=unsupported collective plan cannot reach schedule decision={s} unsupported={d}\n",
            .{ @tagName(plan.decision), plan.unsupported_collective_count },
        );
        return CompilePipelineError.InvalidLowering;
    }

    fn addInstructionCosts(self: *TracePlanBuilder) !void {
        const capabilities: CompilerBackendCapabilitySet = compilerCapabilitySet(backendKindForTarget(self.target.description.kind));
        var capability_facts: std.ArrayList(mlir_state.CostCapabilityFact) = .empty;
        defer capability_facts.deinit(self.allocator);

        for (self.graph.instructions) |instruction| {
            if (instruction.kind == .return_) continue;
            const capability = compilerCapabilityForInstruction(capabilities, instruction, self.graph.values) orelse {
                try writeStageFailure(self.diagnostics, .target_legality, "backend-capability", "instruction has no backend capability");
                return CompilePipelineError.UnsupportedTargetFeature;
            };
            try capability_facts.append(self.allocator, .{
                .graph_instruction_id = instruction.id,
                .expected_unit_id = capability.expected_unit_id,
            });
        }

        const costs = mlir_state.deriveCostLedgerEntries(
            self.allocator,
            self.graph.values,
            self.graph.instructions,
            capability_facts.items,
            self.diagnostics,
        ) catch |err| switch (err) {
            mlir_state.MlirStateError.InvalidLoweringPlan => return CompilePipelineError.InvalidGraph,
            else => return err,
        };
        var costs_owned = true;
        errdefer if (costs_owned) mlir_state.deinitExtractedCostLedgerEntries(self.allocator, costs);
        try self.cost_ledger.appendSlice(self.allocator, costs);
        self.allocator.free(costs);
        costs_owned = false;
        self.performance_strings_owned = true;
    }

    fn addMemoryTrafficRecords(self: *TracePlanBuilder) !void {
        const placement_plan = self.placement_plan orelse {
            try writeStageFailure(self.diagnostics, .placement_planning, "placement", "memory traffic refinement requires placement plan");
            return CompilePipelineError.InvalidLowering;
        };
        const traffic = mlir_state.deriveMemoryTrafficRecords(
            self.allocator,
            self.target.description,
            self.graph.values,
            self.graph.instructions,
            self.cost_ledger.items,
            self.lowering_records.items,
            placement_plan.records,
            self.diagnostics,
        ) catch |err| switch (err) {
            mlir_state.MlirStateError.InvalidKernelCodegenPlan => return CompilePipelineError.InvalidLowering,
            mlir_state.MlirStateError.InvalidLoweringPlan => return CompilePipelineError.InvalidLowering,
            else => return err,
        };
        var traffic_owned = true;
        errdefer if (traffic_owned) mlir_state.deinitExtractedMemoryTrafficRecords(self.allocator, traffic);
        try self.memory_traffic_records.appendSlice(self.allocator, traffic);
        self.allocator.free(traffic);
        traffic_owned = false;
    }

    fn graphValue(self: *TracePlanBuilder, value_id: compiler_facts.GraphValueId) compiler_facts.GraphValue {
        const index: usize = std.math.cast(usize, value_id.index) orelse unreachable;
        return self.graph.values[index];
    }

    fn parameterValueIds(self: *TracePlanBuilder) ![]const compiler_facts.GraphValueId {
        var ids: std.ArrayList(compiler_facts.GraphValueId) = .empty;
        errdefer ids.deinit(self.allocator);
        for (self.graph.values) |value| {
            if (value.role == .parameter) try ids.append(self.allocator, value.id);
        }
        return ids.toOwnedSlice(self.allocator);
    }

    fn returnValueIds(self: *TracePlanBuilder) ![]const compiler_facts.GraphValueId {
        for (self.graph.instructions) |instruction| {
            if (instruction.kind == .return_) return copyValueIds(self.allocator, instruction.inputs);
        }
        try writeStageFailure(self.diagnostics, .schedule_build, "return", "graph has no return instruction");
        return CompilePipelineError.InvalidGraph;
    }

    fn addSchedule(self: *TracePlanBuilder, parameter_ids: []const compiler_facts.GraphValueId, output_ids: []const compiler_facts.GraphValueId) !void {
        var plan = mlir_state.deriveSchedulePlan(
            self.allocator,
            parameter_ids,
            output_ids,
            self.lowering_records.items,
            self.cost_ledger.items,
            self.diagnostics,
        ) catch |err| switch (err) {
            mlir_state.MlirStateError.InvalidSchedulePlan => return CompilePipelineError.InvalidLowering,
            else => return err,
        };
        var plan_owned = true;
        errdefer if (plan_owned) mlir_state.deinitDerivedSchedulePlan(self.allocator, plan);
        const command_slice = plan.commands;
        try self.schedule_commands.appendSlice(self.allocator, command_slice);
        self.allocator.free(command_slice);
        plan.commands = &.{};
        const overlap_slice = plan.overlaps;
        try self.schedule_overlap_records.appendSlice(self.allocator, overlap_slice);
        self.allocator.free(overlap_slice);
        plan.overlaps = &.{};
        plan_owned = false;
        self.schedule_overlap_reason_strings_owned = true;
    }

    fn addKernelCodegenRecords(self: *TracePlanBuilder) !void {
        const backend_kind = backendKindForTarget(self.target.description.kind);
        const capabilities: CompilerBackendCapabilitySet = compilerCapabilitySet(backend_kind);
        const command_id = try self.backendExecuteCommandId();
        var capability_facts: std.ArrayList(mlir_state.KernelCodegenCapabilityFact) = .empty;
        defer capability_facts.deinit(self.allocator);
        for (self.graph.instructions) |instruction| {
            if (instruction.kind == .return_) continue;
            const capability = compilerCapabilityForInstruction(capabilities, instruction, self.graph.values) orelse continue;
            try capability_facts.append(self.allocator, .{
                .graph_instruction_id = instruction.id,
                .backend_operation = capability.backend_operation,
                .expected_unit_id = capability.expected_unit_id,
            });
        }

        const placement_plan = self.placement_plan orelse {
            try writeStageFailure(self.diagnostics, .kernel_codegen, "placement", "kernel codegen requires placement plan");
            return CompilePipelineError.InvalidLowering;
        };
        const records = mlir_state.deriveKernelCodegenRecords(
            self.allocator,
            backend_kind,
            command_id,
            backendOperationForKind(backend_kind),
            fusedElementwiseOperationForCompilerBackend(backend_kind),
            self.graph.instructions,
            self.lowering_records.items,
            placement_plan.records,
            self.memory_traffic_records.items,
            capability_facts.items,
            self.diagnostics,
        ) catch |err| switch (err) {
            mlir_state.MlirStateError.InvalidKernelCodegenPlan => return CompilePipelineError.InvalidLowering,
            else => return err,
        };
        var records_owned = true;
        errdefer if (records_owned) mlir_state.deinitExtractedKernelCodegenRecords(self.allocator, records);
        try self.kernel_codegen_records.appendSlice(self.allocator, records);
        self.allocator.free(records);
        records_owned = false;
        self.kernel_codegen_strings_owned = true;
    }

    fn commitPerformanceFactsToMlir(self: *TracePlanBuilder) !void {
        try mlir_state.commitPerformanceFacts(self.mlir_session, self.cost_ledger.items, self.memory_traffic_records.items, self.lowering_records.items.len, self.diagnostics);
        const extracted_costs = try mlir_state.extractCostLedgerEntries(self.allocator, self.mlir_session, self.diagnostics);
        var costs_owned = true;
        errdefer if (costs_owned) mlir_state.deinitExtractedCostLedgerEntries(self.allocator, extracted_costs);
        const extracted_traffic = try mlir_state.extractMemoryTrafficRecords(self.allocator, self.mlir_session, extracted_costs.len, self.lowering_records.items.len, self.diagnostics);
        var traffic_owned = true;
        errdefer if (traffic_owned) mlir_state.deinitExtractedMemoryTrafficRecords(self.allocator, extracted_traffic);

        for (self.cost_ledger.items) |entry| {
            if (self.performance_strings_owned) {
                if (entry.source) |source| {
                    self.allocator.free(source.op_name);
                    self.allocator.free(source.location);
                }
                self.allocator.free(entry.formula);
                self.allocator.free(entry.approximation);
            }
            self.allocator.free(entry.graph_instruction_ids);
        }
        self.cost_ledger.deinit(self.allocator);
        self.cost_ledger = .empty;
        try self.cost_ledger.appendSlice(self.allocator, extracted_costs);
        self.allocator.free(extracted_costs);
        costs_owned = false;

        for (self.memory_traffic_records.items) |record| {
            self.allocator.free(record.graph_instruction_ids);
            self.allocator.free(record.cost_ledger_ids);
            if (self.performance_strings_owned) self.allocator.free(record.reason);
        }
        self.memory_traffic_records.deinit(self.allocator);
        self.memory_traffic_records = .empty;
        try self.memory_traffic_records.appendSlice(self.allocator, extracted_traffic);
        self.allocator.free(extracted_traffic);
        traffic_owned = false;
        self.performance_strings_owned = true;
    }

    fn commitLoweringRecordsToMlir(self: *TracePlanBuilder) !void {
        try mlir_state.commitLoweringPlan(self.mlir_session, self.cost_ledger.items, self.diagnostics);
        const extracted_records = try mlir_state.extractLoweringRecords(self.allocator, self.mlir_session, self.cost_ledger.items.len, self.diagnostics);
        var extracted_owned = true;
        errdefer if (extracted_owned) mlir_state.deinitExtractedLoweringRecords(self.allocator, extracted_records);
        const extracted_region_facts = try mlir_state.extractLoweringRegionFacts(self.allocator, self.mlir_session, extracted_records.len, self.diagnostics);
        defer mlir_state.deinitExtractedLoweringRegionFacts(self.allocator, extracted_region_facts);
        const extracted_costs = try mlir_state.extractCostLedgerEntries(self.allocator, self.mlir_session, self.diagnostics);
        var costs_owned = true;
        errdefer if (costs_owned) mlir_state.deinitExtractedCostLedgerEntries(self.allocator, extracted_costs);

        for (self.lowering_records.items) |record| {
            self.allocator.free(record.graph_instruction_ids);
            self.allocator.free(record.cost_ledger_ids);
        }
        self.lowering_records.deinit(self.allocator);
        self.lowering_records = .empty;
        try self.lowering_records.appendSlice(self.allocator, extracted_records);
        self.allocator.free(extracted_records);
        extracted_owned = false;
        self.lowering_strings_owned = true;

        for (self.cost_ledger.items) |entry| {
            if (self.performance_strings_owned) {
                if (entry.source) |source| {
                    self.allocator.free(source.op_name);
                    self.allocator.free(source.location);
                }
                self.allocator.free(entry.formula);
                self.allocator.free(entry.approximation);
            }
            self.allocator.free(entry.graph_instruction_ids);
        }
        self.cost_ledger.deinit(self.allocator);
        self.cost_ledger = .empty;
        try self.cost_ledger.appendSlice(self.allocator, extracted_costs);
        self.allocator.free(extracted_costs);
        costs_owned = false;
        self.performance_strings_owned = true;
    }

    fn commitKernelCodegenRecordsToMlir(self: *TracePlanBuilder) !void {
        try mlir_state.commitKernelCodegenPlan(self.mlir_session, self.kernel_codegen_records.items, self.diagnostics);
        const extracted_records = try mlir_state.extractKernelCodegenRecords(self.allocator, self.mlir_session, self.diagnostics);
        var extracted_owned = true;
        errdefer if (extracted_owned) mlir_state.deinitExtractedKernelCodegenRecords(self.allocator, extracted_records);

        for (self.kernel_codegen_records.items) |record| {
            if (self.kernel_codegen_strings_owned) {
                self.allocator.free(record.operation);
                self.allocator.free(record.reason);
            }
            self.allocator.free(record.logical_tile_shape);
            self.allocator.free(record.external_input_ids);
            self.allocator.free(record.external_output_ids);
            self.allocator.free(record.intermediate_value_ids);
            self.allocator.free(record.graph_instruction_ids);
            self.allocator.free(record.cost_ledger_ids);
            self.allocator.free(record.memory_traffic_ids);
        }
        self.kernel_codegen_records.deinit(self.allocator);
        self.kernel_codegen_records = .empty;
        try self.kernel_codegen_records.appendSlice(self.allocator, extracted_records);
        self.allocator.free(extracted_records);
        extracted_owned = false;
        self.kernel_codegen_strings_owned = true;
    }

    fn commitScheduleRecordsToMlir(self: *TracePlanBuilder) !void {
        try mlir_state.commitSchedulePlan(self.mlir_session, self.schedule_commands.items, self.schedule_overlap_records.items, self.diagnostics);
        const extracted_commands = try mlir_state.extractScheduleCommands(self.allocator, self.mlir_session, self.diagnostics);
        var commands_owned = true;
        errdefer if (commands_owned) mlir_state.deinitExtractedScheduleCommands(self.allocator, extracted_commands);
        const extracted_overlaps = try mlir_state.extractScheduleOverlapRecords(self.allocator, self.mlir_session, extracted_commands.len, self.diagnostics);
        var overlaps_owned = true;
        errdefer if (overlaps_owned) mlir_state.deinitExtractedScheduleOverlapRecords(self.allocator, extracted_overlaps);

        for (self.schedule_commands.items) |command| {
            self.allocator.free(command.inputs);
            self.allocator.free(command.outputs);
            self.allocator.free(command.dependencies);
            self.allocator.free(command.lowering_record_ids);
            self.allocator.free(command.cost_ledger_ids);
        }
        self.schedule_commands.deinit(self.allocator);
        self.schedule_commands = .empty;
        try self.schedule_commands.appendSlice(self.allocator, extracted_commands);
        self.allocator.free(extracted_commands);
        commands_owned = false;

        if (self.schedule_overlap_reason_strings_owned) {
            for (self.schedule_overlap_records.items) |overlap| self.allocator.free(overlap.reason);
        }
        self.schedule_overlap_records.deinit(self.allocator);
        self.schedule_overlap_records = .empty;
        try self.schedule_overlap_records.appendSlice(self.allocator, extracted_overlaps);
        self.allocator.free(extracted_overlaps);
        overlaps_owned = false;
        self.schedule_overlap_reason_strings_owned = true;
    }

    fn verifyTileLegalityRecords(self: *TracePlanBuilder) !void {
        for (self.kernel_codegen_records.items) |record| {
            try self.verifyResultMemoryCapacity(record);
            try self.verifyTileMemoryCapacity(record);
        }
    }

    fn verifyResultMemoryCapacity(self: *TracePlanBuilder, record: core.KernelCodegenRecord) !void {
        const memory_space = memorySpaceById(self.target.description, record.result_memory_space_id) orelse {
            try self.diagnostics.print("pass=tile_legality_verify feature=result-memory reason=unknown result memory codegen={d} memory={d}\n", .{ record.id.index, record.result_memory_space_id });
            return CompilePipelineError.InvalidLowering;
        };
        const capacity = memory_space.capacity_bytes orelse return;
        const required = self.valueBytes(record.external_output_ids);
        if (required > capacity) {
            try self.diagnostics.print("pass=tile_legality_verify feature=result-memory reason=result bytes exceed memory capacity codegen={d} memory={d} required={d} capacity={d}\n", .{ record.id.index, record.result_memory_space_id, required, capacity });
            return CompilePipelineError.InvalidLowering;
        }
    }

    fn verifyTileMemoryCapacity(self: *TracePlanBuilder, record: core.KernelCodegenRecord) !void {
        const memory_space_id = record.tile_memory_space_id orelse return;
        const memory_space = memorySpaceById(self.target.description, memory_space_id) orelse {
            try self.diagnostics.print("pass=tile_legality_verify feature=tile-memory reason=unknown tile memory codegen={d} memory={d}\n", .{ record.id.index, memory_space_id });
            return CompilePipelineError.InvalidLowering;
        };
        const capacity = memory_space.capacity_bytes orelse return;
        const required = self.codegenLiveBytes(record);
        if (required > capacity) {
            try self.diagnostics.print("pass=tile_legality_verify feature=tile-memory reason=tile live bytes exceed memory capacity codegen={d} memory={d} required={d} capacity={d}\n", .{ record.id.index, memory_space_id, required, capacity });
            return CompilePipelineError.InvalidLowering;
        }
    }

    fn addBackendBinding(self: *TracePlanBuilder) !void {
        const capabilities: CompilerBackendCapabilitySet = compilerCapabilitySet(backendKindForTarget(self.target.description.kind));
        const graph_instruction_ids = try self.allExecutableInstructionIds();
        errdefer self.allocator.free(graph_instruction_ids);
        const cost_ledger_ids = try self.allCostIds();
        errdefer self.allocator.free(cost_ledger_ids);
        try self.backend_bindings.append(self.allocator, .{
            .id = .{ .index = 0 },
            .command_id = .{ .index = 1 },
            .backend_kind = backendKindForTarget(self.target.description.kind),
            .backend_operation = backendOperationForGraph(capabilities, self.graph),
            .graph_instruction_ids = graph_instruction_ids,
            .expected_unit_id = expectedBackendUnitForGraph(capabilities, self.graph),
            .cost_ledger_ids = cost_ledger_ids,
        });
    }

    fn commitBackendBindingsToMlir(self: *TracePlanBuilder) !void {
        try mlir_state.commitBackendBindings(self.mlir_session, self.backend_bindings.items, self.diagnostics);
        const extracted_bindings = try mlir_state.extractBackendBindings(self.allocator, self.mlir_session, self.diagnostics);
        var extracted_owned = true;
        errdefer if (extracted_owned) mlir_state.deinitExtractedBackendBindings(self.allocator, extracted_bindings);

        for (self.backend_bindings.items) |binding| {
            self.allocator.free(binding.graph_instruction_ids);
            self.allocator.free(binding.cost_ledger_ids);
        }
        self.backend_bindings.deinit(self.allocator);
        self.backend_bindings = .empty;
        try self.backend_bindings.appendSlice(self.allocator, extracted_bindings);
        self.allocator.free(extracted_bindings);
        extracted_owned = false;
        self.backend_binding_strings_owned = true;
    }

    fn commitExecutableReadinessToMlir(self: *TracePlanBuilder) !void {
        const contract: mlir_state.ExecutableContract = .{
            .target_kind = self.target.description.kind,
            .schedule_command_count = std.math.cast(u32, self.schedule_commands.items.len) orelse {
                try writeStageFailure(self.diagnostics, .executable_creation, "schedule", "schedule command count exceeds V0 executable contract bounds");
                return CompilePipelineError.InvalidSchedule;
            },
            .backend_binding_count = std.math.cast(u32, self.backend_bindings.items.len) orelse {
                try writeStageFailure(self.diagnostics, .executable_creation, "backend-binding", "backend binding count exceeds V0 executable contract bounds");
                return CompilePipelineError.InvalidBackendBinding;
            },
            .kernel_codegen_count = std.math.cast(u32, self.kernel_codegen_records.items.len) orelse {
                try writeStageFailure(self.diagnostics, .executable_creation, "codegen", "kernel codegen count exceeds V0 executable contract bounds");
                return CompilePipelineError.InvalidLowering;
            },
        };
        try mlir_state.commitExecutableReadiness(self.mlir_session, contract, self.diagnostics);
    }

    fn addExplainRecords(self: *TracePlanBuilder) !void {
        for (self.lowering_records.items) |lowering| {
            const source_refs = try self.sourceRefsForInstructions(lowering.graph_instruction_ids);
            errdefer self.allocator.free(source_refs);
            const profile_refs = try self.allocator.alloc(core.ProfileEventId, 0);
            errdefer self.allocator.free(profile_refs);
            const cost_ledger_ids = try copyCostIds(self.allocator, lowering.cost_ledger_ids);
            errdefer self.allocator.free(cost_ledger_ids);
            try self.explain_records.append(self.allocator, .{
                .id = .{ .index = std.math.cast(u32, self.explain_records.items.len) orelse unreachable },
                .pass_name = "lowering",
                .subject = .{ .lowering_record = lowering.id },
                .decision = @tagName(lowering.decision),
                .reason = lowering.reason,
                .source_refs = source_refs,
                .cost_ledger_ids = cost_ledger_ids,
                .profile_event_ids = profile_refs,
            });
        }

        const source_refs = try self.allocator.alloc(compiler_facts.SourceRef, 0);
        errdefer self.allocator.free(source_refs);
        const profile_refs = try self.allocator.alloc(core.ProfileEventId, 0);
        errdefer self.allocator.free(profile_refs);
        const cost_ledger_ids = try self.allCostIds();
        errdefer self.allocator.free(cost_ledger_ids);
        try self.explain_records.append(self.allocator, .{
            .id = .{ .index = std.math.cast(u32, self.explain_records.items.len) orelse unreachable },
            .pass_name = "schedule_build",
            .subject = .{ .backend_binding = .{ .index = 0 } },
            .decision = "single_v0_backend_execute",
            .reason = "V0 binds all supported compute lowerings to one backend execute command with explicit H2D and D2H commands",
            .source_refs = source_refs,
            .cost_ledger_ids = cost_ledger_ids,
            .profile_event_ids = profile_refs,
        });
    }

    fn sourceRefsForInstructions(self: *TracePlanBuilder, instruction_ids: []const compiler_facts.GraphInstructionId) ![]const compiler_facts.SourceRef {
        const source_refs = try self.allocator.alloc(compiler_facts.SourceRef, instruction_ids.len);
        errdefer self.allocator.free(source_refs);
        for (instruction_ids, 0..) |instruction_id, index| {
            const instruction_index: usize = std.math.cast(usize, instruction_id.index) orelse unreachable;
            source_refs[index] = self.graph.instructions[instruction_index].source;
        }
        return source_refs;
    }

    fn allExecutableInstructionIds(self: *TracePlanBuilder) ![]const compiler_facts.GraphInstructionId {
        var ids: std.ArrayList(compiler_facts.GraphInstructionId) = .empty;
        errdefer ids.deinit(self.allocator);
        for (self.graph.instructions) |instruction| {
            if (instruction.kind != .return_) try ids.append(self.allocator, instruction.id);
        }
        return ids.toOwnedSlice(self.allocator);
    }

    fn allCostIds(self: *TracePlanBuilder) ![]const compiler_facts.CostLedgerId {
        const ids = try self.allocator.alloc(compiler_facts.CostLedgerId, self.cost_ledger.items.len);
        errdefer self.allocator.free(ids);
        for (ids, 0..) |*id, index| {
            id.* = .{ .index = std.math.cast(u32, index) orelse unreachable };
        }
        return ids;
    }

    fn finish(self: *TracePlanBuilder) !PlannedTraceReport {
        const fusion_groups: []const compiler_facts.FusionGroup = self.fusion_groups;
        const placement_records: []const compiler_facts.PlacementRecord = if (self.placement_plan) |plan| plan.records else &.{};
        const cost_ledger = try self.cost_ledger.toOwnedSlice(self.allocator);
        errdefer freeCostLedger(self.allocator, cost_ledger, self.performance_strings_owned);
        const lowering_records = try self.lowering_records.toOwnedSlice(self.allocator);
        errdefer freeLoweringRecords(self.allocator, lowering_records, self.lowering_strings_owned);
        const memory_traffic_records = try self.memory_traffic_records.toOwnedSlice(self.allocator);
        errdefer freeMemoryTrafficRecords(self.allocator, memory_traffic_records, self.performance_strings_owned);
        const schedule_overlap_records = try self.schedule_overlap_records.toOwnedSlice(self.allocator);
        errdefer self.allocator.free(schedule_overlap_records);
        const schedule_commands = try self.schedule_commands.toOwnedSlice(self.allocator);
        errdefer freeScheduleCommands(self.allocator, schedule_commands);
        const kernel_codegen_records = try self.kernel_codegen_records.toOwnedSlice(self.allocator);
        errdefer freeKernelCodegenRecords(self.allocator, kernel_codegen_records, self.kernel_codegen_strings_owned);
        const backend_bindings = try self.backend_bindings.toOwnedSlice(self.allocator);
        errdefer freeBackendBindings(self.allocator, backend_bindings, self.backend_binding_strings_owned);
        const profile_events = try self.profile_events.toOwnedSlice(self.allocator);
        errdefer self.allocator.free(profile_events);
        const explain_records = try self.explain_records.toOwnedSlice(self.allocator);
        errdefer freeExplainRecords(self.allocator, explain_records);

        return .{
            .allocator = self.allocator,
            .compile_options = self.compile_options,
            .fusion_group_strings_owned = self.fusion_group_strings_owned,
            .placement_reason_strings_owned = self.placement_reason_strings_owned,
            .collective_reason_strings_owned = self.collective_reason_strings_owned,
            .lowering_strings_owned = self.lowering_strings_owned,
            .performance_strings_owned = self.performance_strings_owned,
            .kernel_codegen_strings_owned = self.kernel_codegen_strings_owned,
            .schedule_overlap_reason_strings_owned = self.schedule_overlap_reason_strings_owned,
            .backend_binding_strings_owned = self.backend_binding_strings_owned,
            .report = .{
                .sources = self.graph.sources,
                .target = self.target.description,
                .graph_values = self.graph.values,
                .graph_instructions = self.graph.instructions,
                .mlir_pass_records = &.{},
                .graph_rewrite_records = &.{},
                .fusion_groups = fusion_groups,
                .placement_records = placement_records,
                .collective_plan_records = self.collective_plan_records,
                .cost_ledger = cost_ledger,
                .lowering_records = lowering_records,
                .memory_traffic_records = memory_traffic_records,
                .schedule_overlap_records = schedule_overlap_records,
                .schedule_commands = schedule_commands,
                .kernel_codegen_records = kernel_codegen_records,
                .backend_bindings = backend_bindings,
                .profile_events = profile_events,
                .explain_records = explain_records,
            },
        };
    }

    fn codegenLiveBytes(self: *TracePlanBuilder, record: core.KernelCodegenRecord) u128 {
        var total: u128 = 0;
        total += self.valueBytes(record.external_input_ids);
        total += self.valueBytesNotAlreadyCounted(record.external_output_ids, record.external_input_ids, &.{});
        total += self.valueBytesNotAlreadyCounted(record.intermediate_value_ids, record.external_input_ids, record.external_output_ids);
        return total;
    }

    fn valueBytes(self: *TracePlanBuilder, value_ids: []const compiler_facts.GraphValueId) u128 {
        var total: u128 = 0;
        for (value_ids) |value_id| {
            total += tensorBytes(self.graphValue(value_id).ty);
        }
        return total;
    }

    fn valueBytesNotAlreadyCounted(
        self: *TracePlanBuilder,
        value_ids: []const compiler_facts.GraphValueId,
        counted_a: []const compiler_facts.GraphValueId,
        counted_b: []const compiler_facts.GraphValueId,
    ) u128 {
        var total: u128 = 0;
        for (value_ids) |value_id| {
            if (valueIdInSlice(counted_a, value_id) or valueIdInSlice(counted_b, value_id)) continue;
            total += tensorBytes(self.graphValue(value_id).ty);
        }
        return total;
    }

    fn backendExecuteCommandId(self: *TracePlanBuilder) !core.ScheduleCommandId {
        for (self.schedule_commands.items) |command| {
            if (command.kind == .backend_execute) return command.id;
        }
        try writeStageFailure(self.diagnostics, .kernel_codegen, "schedule", "kernel codegen requires a backend execute command");
        return CompilePipelineError.InvalidLowering;
    }
};

fn copyValueIds(allocator: std.mem.Allocator, values: []const compiler_facts.GraphValueId) ![]const compiler_facts.GraphValueId {
    const copy = try allocator.alloc(compiler_facts.GraphValueId, values.len);
    errdefer allocator.free(copy);
    @memcpy(copy, values);
    return copy;
}

fn copyCostIds(allocator: std.mem.Allocator, values: []const compiler_facts.CostLedgerId) ![]const compiler_facts.CostLedgerId {
    const copy = try allocator.alloc(compiler_facts.CostLedgerId, values.len);
    errdefer allocator.free(copy);
    @memcpy(copy, values);
    return copy;
}

fn copySources(allocator: std.mem.Allocator, sources: []const compiler_facts.SourceRef) ![]compiler_facts.SourceRef {
    const copy = try allocator.alloc(compiler_facts.SourceRef, sources.len);
    errdefer allocator.free(copy);
    var copied_count: usize = 0;
    errdefer {
        for (copy[0..copied_count]) |source| freeSource(allocator, source);
    }
    for (sources, 0..) |source, index| {
        copy[index] = try copySource(allocator, source);
        copied_count += 1;
    }
    return copy;
}

fn copySource(allocator: std.mem.Allocator, source: compiler_facts.SourceRef) !compiler_facts.SourceRef {
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

fn freeSources(allocator: std.mem.Allocator, sources: []compiler_facts.SourceRef) void {
    for (sources) |source| freeSource(allocator, source);
    allocator.free(sources);
}

fn freeSource(allocator: std.mem.Allocator, source: compiler_facts.SourceRef) void {
    allocator.free(source.op_name);
    allocator.free(source.location);
}

fn copyValues(allocator: std.mem.Allocator, values: []const compiler_facts.GraphValue, sources: []const compiler_facts.SourceRef) ![]compiler_facts.GraphValue {
    const copy = try allocator.alloc(compiler_facts.GraphValue, values.len);
    errdefer allocator.free(copy);
    var copied_count: usize = 0;
    errdefer {
        for (copy[0..copied_count]) |value| allocator.free(value.ty.dims);
    }
    for (values, 0..) |value, index| {
        const dims = try allocator.dupe(i64, value.ty.dims);
        errdefer allocator.free(dims);
        copy[index] = .{
            .id = value.id,
            .ty = .{
                .element_type = value.ty.element_type,
                .dims = dims,
                .layout = value.ty.layout,
            },
            .role = value.role,
            .source = if (value.source) |source| copiedSourceRef(sources, source) else null,
        };
        copied_count += 1;
    }
    return copy;
}

fn freeValues(allocator: std.mem.Allocator, values: []compiler_facts.GraphValue) void {
    for (values) |value| allocator.free(value.ty.dims);
    allocator.free(values);
}

fn copyInstructionWithValueRewrites(
    allocator: std.mem.Allocator,
    instruction: compiler_facts.GraphInstruction,
    new_id: compiler_facts.GraphInstructionId,
    value_replacements: []const compiler_facts.GraphValueId,
    sources: []const compiler_facts.SourceRef,
) !compiler_facts.GraphInstruction {
    const inputs = try allocator.alloc(compiler_facts.GraphValueId, instruction.inputs.len);
    errdefer allocator.free(inputs);
    for (instruction.inputs, 0..) |input, index| inputs[index] = remappedValueId(value_replacements, input);

    const outputs = try copyValueIds(allocator, instruction.outputs);
    errdefer allocator.free(outputs);
    const payload = try copyGraphPayload(allocator, instruction.payload);
    errdefer freeGraphPayload(allocator, payload);

    return .{
        .id = new_id,
        .kind = instruction.kind,
        .inputs = inputs,
        .outputs = outputs,
        .payload = payload,
        .source = copiedSourceRef(sources, instruction.source),
    };
}

fn copyGraphPayload(allocator: std.mem.Allocator, payload: compiler_facts.GraphPayload) !compiler_facts.GraphPayload {
    return switch (payload) {
        .dot_general => |dot| .{ .dot_general = dot },
        .elementwise_unary => |unary| .{ .elementwise_unary = unary },
        .elementwise_binary => |binary| .{ .elementwise_binary = binary },
        .broadcast => |broadcast| .{ .broadcast = .{ .dimensions = try allocator.dupe(u32, broadcast.dimensions) } },
        .reshape => |reshape| .{ .reshape = reshape },
        .transpose => |transpose| .{ .transpose = .{ .permutation = try allocator.dupe(u32, transpose.permutation) } },
        .collective => |collective| .{ .collective = .{
            .op = collective.op,
            .reduction = collective.reduction,
            .replica_group_count = collective.replica_group_count,
            .replica_group_size = collective.replica_group_size,
            .replica_groups = try allocator.dupe(u32, collective.replica_groups),
            .channel_id = collective.channel_id,
            .channel_type = collective.channel_type,
            .uses_token = collective.uses_token,
        } },
        .return_ => |return_payload| .{ .return_ = return_payload },
    };
}

fn freeInstructionList(allocator: std.mem.Allocator, instructions: *std.ArrayList(compiler_facts.GraphInstruction)) void {
    for (instructions.items) |instruction| freeInstruction(allocator, instruction);
    instructions.deinit(allocator);
}

fn freeInstructions(allocator: std.mem.Allocator, instructions: []compiler_facts.GraphInstruction) void {
    for (instructions) |instruction| freeInstruction(allocator, instruction);
    allocator.free(instructions);
}

fn freeInstruction(allocator: std.mem.Allocator, instruction: compiler_facts.GraphInstruction) void {
    allocator.free(instruction.inputs);
    allocator.free(instruction.outputs);
    freeGraphPayload(allocator, instruction.payload);
}

fn freeGraphPayload(allocator: std.mem.Allocator, payload: compiler_facts.GraphPayload) void {
    switch (payload) {
        .broadcast => |broadcast| allocator.free(broadcast.dimensions),
        .transpose => |transpose| allocator.free(transpose.permutation),
        .collective => |collective| allocator.free(collective.replica_groups),
        else => {},
    }
}

fn copiedSourceRef(sources: []const compiler_facts.SourceRef, source: compiler_facts.SourceRef) compiler_facts.SourceRef {
    const index: usize = std.math.cast(usize, source.id.index) orelse unreachable;
    return sources[index];
}

fn remappedValueId(value_replacements: []const compiler_facts.GraphValueId, value_id: compiler_facts.GraphValueId) compiler_facts.GraphValueId {
    const index: usize = std.math.cast(usize, value_id.index) orelse unreachable;
    return value_replacements[index];
}

fn identityValueRewriteApplies(graph: GraphModule, instruction: compiler_facts.GraphInstruction, diagnostics: *std.Io.Writer) !bool {
    if (instruction.inputs.len != 1 or instruction.outputs.len != 1) return false;
    const input = try requireGraphValue(graph, instruction.inputs[0], diagnostics);
    const output = try requireGraphValue(graph, instruction.outputs[0], diagnostics);
    if (!tensorTypesEqual(input.ty, output.ty)) return false;
    return switch (instruction.payload) {
        .broadcast => |payload| isIdentityDimensionMap(payload.dimensions),
        .reshape => true,
        .transpose => |payload| isIdentityDimensionMap(payload.permutation),
        else => false,
    };
}

fn identityValueRewritePassName(instruction: compiler_facts.GraphInstruction) []const u8 {
    return switch (instruction.payload) {
        .broadcast => "broadcast_simplify",
        .reshape, .transpose => "reshape_transpose_fold",
        else => unreachable,
    };
}

fn identityValueRewriteReason(instruction: compiler_facts.GraphInstruction) []const u8 {
    return switch (instruction.payload) {
        .broadcast => "identity broadcast has identical input/output tensor type and dimension map",
        .reshape => "identity reshape has identical input/output tensor type",
        .transpose => "identity transpose has identical input/output tensor type and permutation",
        else => unreachable,
    };
}

fn isIdentityDimensionMap(dimensions: []const u32) bool {
    for (dimensions, 0..) |dimension, index| {
        const expected: u32 = std.math.cast(u32, index) orelse return false;
        if (dimension != expected) return false;
    }
    return true;
}

fn memorySpaceById(target: target_pkg.TargetDescription, memory_space_id: u32) ?target_pkg.TargetMemorySpace {
    for (target.memory_spaces) |memory_space| {
        if (memory_space.id == memory_space_id) return memory_space;
    }
    return null;
}

fn valueIdInSlice(values: []const compiler_facts.GraphValueId, value_id: compiler_facts.GraphValueId) bool {
    for (values) |candidate| {
        if (candidate.eql(value_id)) return true;
    }
    return false;
}

fn bytesReadForInstruction(graph: GraphModule, instruction: compiler_facts.GraphInstruction, diagnostics: *std.Io.Writer) !u128 {
    var total: u128 = 0;
    for (instruction.inputs) |id| {
        const value = try requireGraphValue(graph, id, diagnostics);
        total += tensorBytes(value.ty);
    }
    return total;
}

fn bytesWrittenForInstruction(graph: GraphModule, instruction: compiler_facts.GraphInstruction, diagnostics: *std.Io.Writer) !u128 {
    var total: u128 = 0;
    for (instruction.outputs) |id| {
        const value = try requireGraphValue(graph, id, diagnostics);
        total += tensorBytes(value.ty);
    }
    return total;
}

fn tensorBytes(ty: compiler_facts.TensorType) u128 {
    const element_size = ty.element_type.byteSize() orelse 0;
    return tensorElements(ty) * element_size;
}

fn tensorElements(ty: compiler_facts.TensorType) u128 {
    var total: u128 = 1;
    for (ty.dims) |dim| total *= positiveDim(dim);
    return total;
}

fn positiveDim(dim: i64) u128 {
    return std.math.cast(u128, dim) orelse unreachable;
}

fn backendKindForTarget(target_kind: target_pkg.TargetKind) core.BackendKind {
    return switch (target_kind) {
        .metal_v0 => .metal_v0,
        .npu_v0 => .npu_v0,
    };
}

fn totalCollectiveParticipants(options: CompileOptions) u32 {
    return std.math.mul(u32, options.num_replicas, options.num_partitions) catch std.math.maxInt(u32);
}

fn targetHasCollectiveUnit(target: target_pkg.TargetDescription) bool {
    for (target.execution_units) |unit| {
        if (unit.kind == .collective) return true;
    }
    return false;
}

fn hasDuplicateParticipant(participants: []const u32) bool {
    for (participants, 0..) |participant, index| {
        for (participants[index + 1 ..]) |other| {
            if (participant == other) return true;
        }
    }
    return false;
}

fn backendOperationForGraph(capabilities: CompilerBackendCapabilitySet, graph: GraphModule) []const u8 {
    var operation: ?[]const u8 = null;
    for (graph.instructions) |instruction| {
        if (instruction.kind == .return_) continue;
        const capability = compilerCapabilityForInstruction(capabilities, instruction, graph.values) orelse continue;
        if (operation) |existing| {
            if (!std.mem.eql(u8, existing, capability.backend_operation)) return backendOperationForKind(capabilities.kind);
        } else {
            operation = capability.backend_operation;
        }
    }
    return operation orelse backendOperationForKind(capabilities.kind);
}

fn expectedBackendUnitForGraph(capabilities: CompilerBackendCapabilitySet, graph: GraphModule) ?u32 {
    var selected_unit: ?u32 = null;
    var saw_unit = false;
    for (graph.instructions) |instruction| {
        if (instruction.kind == .return_) continue;
        const capability = compilerCapabilityForInstruction(capabilities, instruction, graph.values) orelse return null;
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

fn fusedElementwiseOperationForCompilerBackend(kind: core.BackendKind) []const u8 {
    return switch (kind) {
        .metal_v0 => "metal_mls_elementwise_fusion_kernel",
        .npu_v0 => "npu_elementwise_fusion",
    };
}

fn backendOperationForKind(kind: core.BackendKind) []const u8 {
    return switch (kind) {
        .metal_v0 => "metal_mls_graph_execute",
        .npu_v0 => "npu_execute",
    };
}

fn compilerCapabilitySet(kind: core.BackendKind) CompilerBackendCapabilitySet {
    return switch (kind) {
        .metal_v0 => .{ .kind = kind, .capabilities = &metal_compiler_capabilities },
        .npu_v0 => .{ .kind = kind, .capabilities = &npu_compiler_capabilities },
    };
}

fn compilerFeatureForInstruction(instruction: compiler_facts.GraphInstruction) ?CompilerBackendFeature {
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

fn compilerSupportsInstruction(capabilities: CompilerBackendCapabilitySet, instruction: compiler_facts.GraphInstruction, values: []const compiler_facts.GraphValue) bool {
    if (compilerFeatureForInstruction(instruction) == null) return true;
    return compilerCapabilityForInstruction(capabilities, instruction, values) != null;
}

fn compilerCapabilityForInstruction(
    capabilities: CompilerBackendCapabilitySet,
    instruction: compiler_facts.GraphInstruction,
    values: []const compiler_facts.GraphValue,
) ?CompilerBackendCapability {
    const feature = compilerFeatureForInstruction(instruction) orelse return null;
    const dtype = instructionDtypeFromValues(instruction, values) orelse return null;
    for (capabilities.capabilities) |capability| {
        if (capability.feature != feature) continue;
        if (compilerCapabilitySupportsDtype(capability.dtypes, dtype)) return capability;
    }
    return null;
}

fn instructionDtypeFromValues(instruction: compiler_facts.GraphInstruction, values: []const compiler_facts.GraphValue) ?core.BufferType {
    const value_id = if (instruction.outputs.len > 0) instruction.outputs[0] else if (instruction.inputs.len > 0) instruction.inputs[0] else return null;
    const index: usize = std.math.cast(usize, value_id.index) orelse return null;
    if (index >= values.len) return null;
    return values[index].ty.element_type;
}

fn compilerCapabilitySupportsDtype(dtypes: []const core.BufferType, dtype: core.BufferType) bool {
    for (dtypes) |supported| {
        if (supported == dtype) return true;
    }
    return false;
}

fn stableFingerprint(bytes: []const u8) u64 {
    return std.hash.Wyhash.hash(0, bytes);
}

fn textMentionsShardy(bytes: []const u8) bool {
    return std.mem.indexOf(u8, bytes, "sdy.") != null or
        std.mem.indexOf(u8, bytes, "shardy") != null or
        std.mem.indexOf(u8, bytes, "mhlo.sharding") != null;
}

fn collectiveStablehloName(op: compiler_facts.CollectiveOp) []const u8 {
    return switch (op) {
        .all_reduce => "stablehlo.all_reduce",
    };
}

fn requiredMemorySpace(target: target_pkg.TargetDescription, kind: target_pkg.MemorySpaceKind, diagnostics: *std.Io.Writer) !u32 {
    if (optionalMemorySpace(target, kind)) |id| return id;
    try diagnostics.print("pass=placement_planning feature=memory-space reason=target is missing required memory space kind={s}\n", .{@tagName(kind)});
    return CompilePipelineError.InvalidLowering;
}

fn optionalMemorySpace(target: target_pkg.TargetDescription, kind: target_pkg.MemorySpaceKind) ?u32 {
    for (target.memory_spaces) |memory_space| {
        if (memory_space.kind == kind) return memory_space.id;
    }
    return null;
}

fn tileMemoryForInstruction(instruction: compiler_facts.GraphInstruction, target_kind: target_pkg.TargetKind, tile_memory: ?u32) ?u32 {
    if (target_kind != .npu_v0) return null;
    return switch (instruction.payload) {
        .dot_general, .broadcast, .elementwise_binary, .elementwise_unary => tile_memory,
        .reshape, .transpose, .collective => null,
        .return_ => null,
    };
}

fn tileShapeForInstruction(
    allocator: std.mem.Allocator,
    output_type: compiler_facts.TensorType,
    instruction: compiler_facts.GraphInstruction,
    target: target_pkg.TargetDescription,
) ![]const i64 {
    return switch (target.kind) {
        .metal_v0 => allocator.dupe(i64, output_type.dims),
        .npu_v0 => switch (instruction.payload) {
            .dot_general, .broadcast, .elementwise_binary, .elementwise_unary => boundedNpuTileShape(allocator, output_type.dims),
            .reshape, .transpose, .collective, .return_ => allocator.dupe(i64, output_type.dims),
        },
    };
}

fn boundedNpuTileShape(allocator: std.mem.Allocator, dims: []const i64) ![]const i64 {
    const tile_shape = try allocator.alloc(i64, dims.len);
    errdefer allocator.free(tile_shape);
    for (dims, 0..) |dim, index| {
        tile_shape[index] = @min(dim, npu_v0_tile_dim_limit);
    }
    return tile_shape;
}

fn placementReasonForInstruction(instruction: compiler_facts.GraphInstruction, target_kind: target_pkg.TargetKind) []const u8 {
    return switch (target_kind) {
        .metal_v0 => "Metal V0 uses unified device memory and records whole-tensor logical tiles before MLS codegen",
        .npu_v0 => switch (instruction.payload) {
            .dot_general => "TRN2-like NPU places persistent results in HBM and records bounded local SRAM tile staging for matrix work",
            .broadcast, .elementwise_binary, .elementwise_unary => "TRN2-like NPU keeps persistent results in HBM while recording bounded local SRAM tiles for fusible vector work",
            .reshape, .transpose => "non-identity reshape/transpose is rejected before NPU placement in V0",
            .collective => "collective payload is preserved for algorithm selection and rejected before executable scheduling in V0",
            .return_ => "return has no placement",
        },
    };
}

fn writeValueIdList(writer: *std.Io.Writer, ids: []const compiler_facts.GraphValueId) std.Io.Writer.Error!void {
    for (ids, 0..) |id, index| {
        if (index != 0) try writer.writeAll(",");
        try writer.print("{d}", .{id.index});
    }
}

fn writeInvariantList(writer: *std.Io.Writer, invariants: []const CompilerPassInvariant) std.Io.Writer.Error!void {
    if (invariants.len == 0) {
        try writer.writeAll("none");
        return;
    }
    for (invariants, 0..) |invariant, index| {
        if (index != 0) try writer.writeAll(",");
        try writer.writeAll(@tagName(invariant));
    }
}

fn writeI64List(writer: *std.Io.Writer, values: []const i64) std.Io.Writer.Error!void {
    for (values, 0..) |value, index| {
        if (index != 0) try writer.writeAll("x");
        try writer.print("{d}", .{value});
    }
}

fn copyMlirPassRecords(allocator: std.mem.Allocator, records: []const compiler_facts.MlirPassRecord) ![]const compiler_facts.MlirPassRecord {
    const copy = try allocator.alloc(compiler_facts.MlirPassRecord, records.len);
    errdefer allocator.free(copy);
    @memcpy(copy, records);
    return copy;
}

fn freeCostLedger(allocator: std.mem.Allocator, entries: []compiler_facts.CostLedgerEntry, strings_owned: bool) void {
    for (entries) |entry| {
        if (strings_owned) {
            if (entry.source) |source| {
                allocator.free(source.op_name);
                allocator.free(source.location);
            }
            allocator.free(entry.formula);
            allocator.free(entry.approximation);
        }
        allocator.free(entry.graph_instruction_ids);
    }
    allocator.free(entries);
}

fn freeLoweringRecords(allocator: std.mem.Allocator, records: []compiler_facts.LoweringRecord, strings_owned: bool) void {
    for (records) |record| {
        if (strings_owned) {
            allocator.free(record.reason);
            for (record.rejected_alternatives) |alternative| allocator.free(alternative);
            allocator.free(record.rejected_alternatives);
        }
        allocator.free(record.graph_instruction_ids);
        allocator.free(record.cost_ledger_ids);
    }
    allocator.free(records);
}

fn freeMemoryTrafficRecords(allocator: std.mem.Allocator, records: []compiler_facts.MemoryTrafficRecord, strings_owned: bool) void {
    for (records) |record| {
        if (strings_owned) allocator.free(record.reason);
        allocator.free(record.graph_instruction_ids);
        allocator.free(record.cost_ledger_ids);
    }
    allocator.free(records);
}

fn freeScheduleCommands(allocator: std.mem.Allocator, commands: []core.ScheduleCommand) void {
    for (commands) |command| {
        allocator.free(command.inputs);
        allocator.free(command.outputs);
        allocator.free(command.dependencies);
        allocator.free(command.lowering_record_ids);
        allocator.free(command.cost_ledger_ids);
    }
    allocator.free(commands);
}

fn freeKernelCodegenRecords(allocator: std.mem.Allocator, records: []core.KernelCodegenRecord, strings_owned: bool) void {
    for (records) |record| {
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
    allocator.free(records);
}

fn freeBackendBindings(allocator: std.mem.Allocator, bindings: []core.BackendBinding, strings_owned: bool) void {
    for (bindings) |binding| {
        if (strings_owned) allocator.free(binding.backend_operation);
        allocator.free(binding.graph_instruction_ids);
        allocator.free(binding.cost_ledger_ids);
    }
    allocator.free(bindings);
}

fn freeExplainRecords(allocator: std.mem.Allocator, records: []core.ExplainRecord) void {
    for (records) |record| {
        allocator.free(record.source_refs);
        allocator.free(record.cost_ledger_ids);
        allocator.free(record.profile_event_ids);
    }
    allocator.free(records);
}

const npu_v0_tile_dim_limit: i64 = 128;

const metal_compiler_dtypes = [_]core.BufferType{ .f16, .f32 };
const metal_compiler_capabilities = [_]CompilerBackendCapability{
    .{ .feature = .rank2_dot_general, .dtypes = &metal_compiler_dtypes, .backend_operation = "metal_mls_matmul_kernel", .expected_unit_id = 0 },
    .{ .feature = .broadcast_in_dim, .dtypes = &metal_compiler_dtypes, .backend_operation = "metal_mls_broadcast_kernel", .expected_unit_id = 0 },
    .{ .feature = .add, .dtypes = &metal_compiler_dtypes, .backend_operation = "metal_mls_add_kernel", .expected_unit_id = 0 },
    .{ .feature = .tanh, .dtypes = &metal_compiler_dtypes, .backend_operation = "metal_mls_tanh_kernel", .expected_unit_id = 0 },
};

const npu_compiler_matrix_dtypes = [_]core.BufferType{ .bf16, .f32 };
const npu_compiler_vector_dtypes = [_]core.BufferType{.f32};
const npu_compiler_capabilities = [_]CompilerBackendCapability{
    .{ .feature = .rank2_dot_general, .dtypes = &npu_compiler_matrix_dtypes, .backend_operation = "npu_matmul", .expected_unit_id = 0 },
    .{ .feature = .broadcast_in_dim, .dtypes = &npu_compiler_vector_dtypes, .backend_operation = "npu_elementwise_fusion", .expected_unit_id = 1 },
    .{ .feature = .add, .dtypes = &npu_compiler_vector_dtypes, .backend_operation = "npu_elementwise_fusion", .expected_unit_id = 1 },
    .{ .feature = .tanh, .dtypes = &npu_compiler_vector_dtypes, .backend_operation = "npu_elementwise_fusion", .expected_unit_id = 1 },
};

const GraphImportBuilder = struct {
    allocator: std.mem.Allocator,
    diagnostics: *std.Io.Writer,
    sources: std.ArrayList(compiler_facts.SourceRef) = .empty,
    values: std.ArrayList(compiler_facts.GraphValue) = .empty,
    instructions: std.ArrayList(compiler_facts.GraphInstruction) = .empty,
    value_map: std.ArrayList(ValueMapEntry) = .empty,

    fn deinitPartial(self: *GraphImportBuilder) void {
        for (self.instructions.items) |instruction| {
            self.allocator.free(instruction.inputs);
            self.allocator.free(instruction.outputs);
            switch (instruction.payload) {
                .broadcast => |payload| self.allocator.free(payload.dimensions),
                .transpose => |payload| self.allocator.free(payload.permutation),
                .collective => |payload| self.allocator.free(payload.replica_groups),
                else => {},
            }
        }
        for (self.values.items) |value| {
            self.allocator.free(value.ty.dims);
        }
        for (self.sources.items) |source| {
            self.allocator.free(source.op_name);
            self.allocator.free(source.location);
        }
        self.instructions.deinit(self.allocator);
        self.values.deinit(self.allocator);
        self.sources.deinit(self.allocator);
        self.value_map.deinit(self.allocator);
    }

    fn finish(self: *GraphImportBuilder) !GraphModule {
        const sources = try self.sources.toOwnedSlice(self.allocator);
        errdefer {
            for (sources) |source| self.freeSource(source);
            self.allocator.free(sources);
        }

        const values = try self.values.toOwnedSlice(self.allocator);
        errdefer {
            for (values) |value| self.allocator.free(value.ty.dims);
            self.allocator.free(values);
        }

        const instructions = try self.instructions.toOwnedSlice(self.allocator);
        errdefer {
            for (instructions) |instruction| {
                self.allocator.free(instruction.inputs);
                self.allocator.free(instruction.outputs);
                switch (instruction.payload) {
                    .broadcast => |payload| self.allocator.free(payload.dimensions),
                    .transpose => |payload| self.allocator.free(payload.permutation),
                    .collective => |payload| self.allocator.free(payload.replica_groups),
                    else => {},
                }
            }
            self.allocator.free(instructions);
        }

        self.value_map.deinit(self.allocator);
        return .{
            .allocator = self.allocator,
            .sources = sources,
            .values = values,
            .instructions = instructions,
        };
    }

    fn importModule(self: *GraphImportBuilder, module_op: mlir.MlirOperation) !bool {
        var imported_any = false;
        const region_count = mlir.mlirOperationGetNumRegions(module_op);
        var region_index: isize = 0;
        while (region_index < region_count) : (region_index += 1) {
            var block = mlir.mlirRegionGetFirstBlock(mlir.mlirOperationGetRegion(module_op, region_index));
            while (!mlir.mlirBlockIsNull(block)) : (block = mlir.mlirBlockGetNextInRegion(block)) {
                var op = mlir.mlirBlockGetFirstOperation(block);
                while (!mlir.mlirOperationIsNull(op)) : (op = mlir.mlirOperationGetNextInBlock(op)) {
                    if (std.mem.eql(u8, operationName(op), "func.func")) {
                        try self.importFunction(op);
                        imported_any = true;
                    }
                }
            }
        }
        return imported_any;
    }

    fn importFunction(self: *GraphImportBuilder, function_op: mlir.MlirOperation) !void {
        if (mlir.mlirOperationGetNumRegions(function_op) == 0) return;
        const body = mlir.mlirRegionGetFirstBlock(mlir.mlirOperationGetRegion(function_op, 0));
        if (mlir.mlirBlockIsNull(body)) return;

        const arg_count = mlir.mlirBlockGetNumArguments(body);
        var arg_index: isize = 0;
        while (arg_index < arg_count) : (arg_index += 1) {
            const argument = mlir.mlirBlockGetArgument(body, arg_index);
            _ = try self.registerValue(argument, .parameter, try tensorTypeFromMlir(self.allocator, mlir.mlirValueGetType(argument), self.diagnostics));
        }

        var op = mlir.mlirBlockGetFirstOperation(body);
        while (!mlir.mlirOperationIsNull(op)) : (op = mlir.mlirOperationGetNextInBlock(op)) {
            try self.importOperation(op);
        }
    }

    fn importOperation(self: *GraphImportBuilder, op: mlir.MlirOperation) !void {
        const name = operationName(op);
        if (std.mem.eql(u8, name, "stablehlo.dot_general")) {
            const dims = try dotGeneralSpec(op, self.diagnostics);
            try self.appendInstruction(op, .dot_general, .{ .dot_general = dims });
        } else if (std.mem.eql(u8, name, "stablehlo.broadcast_in_dim")) {
            const dimensions = try broadcastDimensions(self.allocator, op, self.diagnostics);
            errdefer self.allocator.free(dimensions);
            try self.appendInstruction(op, .broadcast, .{ .broadcast = .{ .dimensions = dimensions } });
        } else if (std.mem.eql(u8, name, "stablehlo.reshape")) {
            try self.appendInstruction(op, .reshape, .{ .reshape = .{} });
        } else if (std.mem.eql(u8, name, "stablehlo.transpose")) {
            const permutation = try transposePermutation(self.allocator, op, self.diagnostics);
            errdefer self.allocator.free(permutation);
            try self.appendInstruction(op, .transpose, .{ .transpose = .{ .permutation = permutation } });
        } else if (std.mem.eql(u8, name, "stablehlo.add")) {
            try self.appendInstruction(op, .elementwise_binary, .{ .elementwise_binary = .{ .op = .add } });
        } else if (std.mem.eql(u8, name, "stablehlo.tanh")) {
            try self.appendInstruction(op, .elementwise_unary, .{ .elementwise_unary = .{ .op = .tanh } });
        } else if (std.mem.eql(u8, name, "stablehlo.all_reduce")) {
            const spec = try allReduceSpec(self.allocator, op, self.diagnostics);
            errdefer self.allocator.free(spec.replica_groups);
            try self.appendInstruction(op, .collective, .{ .collective = spec });
        } else if (std.mem.eql(u8, name, "func.return")) {
            try self.appendInstruction(op, .return_, .{ .return_ = .{} });
        } else {
            try self.diagnostics.print("pass=graph_import feature=unsupported-op reason=unsupported operation op={s}\n", .{name});
            return CompilePipelineError.InvalidStablehlo;
        }
    }

    fn appendInstruction(
        self: *GraphImportBuilder,
        op: mlir.MlirOperation,
        kind: compiler_facts.GraphInstructionKind,
        payload: compiler_facts.GraphPayload,
    ) !void {
        const source = try self.addSource(op);

        const inputs = try self.valueIdsForOperands(op);
        errdefer self.allocator.free(inputs);
        const outputs = try self.registerResults(op);
        errdefer self.allocator.free(outputs);

        const instruction_id: compiler_facts.GraphInstructionId = .{ .index = std.math.cast(u32, self.instructions.items.len) orelse unreachable };
        try self.instructions.append(self.allocator, .{
            .id = instruction_id,
            .kind = kind,
            .inputs = inputs,
            .outputs = outputs,
            .payload = payload,
            .source = source,
        });
    }

    fn addSource(self: *GraphImportBuilder, op: mlir.MlirOperation) !compiler_facts.SourceRef {
        const source: compiler_facts.SourceRef = .{
            .id = .{ .index = std.math.cast(u32, self.sources.items.len) orelse unreachable },
            .frontend = if (std.mem.startsWith(u8, operationName(op), "stablehlo.")) .stablehlo else .internal,
            .op_name = try self.allocator.dupe(u8, operationName(op)),
            .source_index = std.math.cast(u32, self.sources.items.len) orelse unreachable,
            .location = try self.allocator.dupe(u8, ""),
        };
        errdefer self.freeSource(source);
        try self.sources.append(self.allocator, source);
        return source;
    }

    fn freeSource(self: *GraphImportBuilder, source: compiler_facts.SourceRef) void {
        self.allocator.free(source.op_name);
        self.allocator.free(source.location);
    }

    fn valueIdsForOperands(self: *GraphImportBuilder, op: mlir.MlirOperation) ![]const compiler_facts.GraphValueId {
        const operand_count: usize = std.math.cast(usize, mlir.mlirOperationGetNumOperands(op)) orelse unreachable;
        const ids = try self.allocator.alloc(compiler_facts.GraphValueId, operand_count);
        errdefer self.allocator.free(ids);

        var operand_index: isize = 0;
        while (operand_index < mlir.mlirOperationGetNumOperands(op)) : (operand_index += 1) {
            const index: usize = std.math.cast(usize, operand_index) orelse unreachable;
            const operand = mlir.mlirOperationGetOperand(op, operand_index);
            ids[index] = self.lookupValue(operand) orelse {
                try self.diagnostics.print("pass=graph_import feature=value-ref reason=operand references unknown value operand={d}\n", .{operand_index});
                return CompilePipelineError.InvalidStablehlo;
            };
        }
        return ids;
    }

    fn registerResults(self: *GraphImportBuilder, op: mlir.MlirOperation) ![]const compiler_facts.GraphValueId {
        const result_count: usize = std.math.cast(usize, mlir.mlirOperationGetNumResults(op)) orelse unreachable;
        const ids = try self.allocator.alloc(compiler_facts.GraphValueId, result_count);
        errdefer self.allocator.free(ids);

        var result_index: isize = 0;
        while (result_index < mlir.mlirOperationGetNumResults(op)) : (result_index += 1) {
            const index: usize = std.math.cast(usize, result_index) orelse unreachable;
            const result = mlir.mlirOperationGetResult(op, result_index);
            ids[index] = try self.registerValue(result, .instruction_result, try tensorTypeFromMlir(self.allocator, mlir.mlirValueGetType(result), self.diagnostics));
        }
        return ids;
    }

    fn registerValue(self: *GraphImportBuilder, value: mlir.MlirValue, role: compiler_facts.GraphValueRole, tensor_type: compiler_facts.TensorType) !compiler_facts.GraphValueId {
        if (self.lookupValue(value)) |existing| {
            self.allocator.free(tensor_type.dims);
            return existing;
        }

        const id: compiler_facts.GraphValueId = .{ .index = std.math.cast(u32, self.values.items.len) orelse unreachable };
        try self.values.append(self.allocator, .{ .id = id, .ty = tensor_type, .role = role, .source = null });
        errdefer _ = self.values.pop();
        try self.value_map.append(self.allocator, .{ .mlir_value = value, .id = id });
        return id;
    }

    fn lookupValue(self: *GraphImportBuilder, value: mlir.MlirValue) ?compiler_facts.GraphValueId {
        for (self.value_map.items) |entry| {
            if (mlir.mlirValueEqual(entry.mlir_value, value)) return entry.id;
        }
        return null;
    }
};

const ValueMapEntry = struct {
    mlir_value: mlir.MlirValue,
    id: compiler_facts.GraphValueId,
};

fn operationName(op: mlir.MlirOperation) []const u8 {
    return mlirStringSlice(mlir.mlirIdentifierStr(mlir.mlirOperationGetName(op)));
}

fn mlirStringSlice(text: mlir.MlirStringRef) []const u8 {
    return text.data[0..text.length];
}

fn tensorTypeFromMlir(allocator: std.mem.Allocator, ty: mlir.MlirType, diagnostics: *std.Io.Writer) !compiler_facts.TensorType {
    if (mlir.mlirTypeIsNull(ty) or !mlir.mlirTypeIsAShaped(ty) or !mlir.mlirShapedTypeHasRank(ty)) {
        try writeStageFailure(diagnostics, .graph_import, "tensor-type", "V0 requires ranked tensor types");
        return CompilePipelineError.InvalidStablehlo;
    }

    const rank = mlir.mlirShapedTypeGetRank(ty);
    const dims = try allocator.alloc(i64, std.math.cast(usize, rank) orelse unreachable);
    errdefer allocator.free(dims);

    var dim_index: isize = 0;
    while (dim_index < rank) : (dim_index += 1) {
        const index: usize = std.math.cast(usize, dim_index) orelse unreachable;
        dims[index] = mlir.mlirShapedTypeGetDimSize(ty, dim_index);
        if (dims[index] < 0) {
            try writeStageFailure(diagnostics, .graph_import, "tensor-type", "dynamic dimensions are unsupported in V0");
            return CompilePipelineError.InvalidStablehlo;
        }
    }

    const element = mlir.mlirShapedTypeGetElementType(ty);
    const element_type = bufferTypeFromMlir(element) orelse {
        allocator.free(dims);
        try writeStageFailure(diagnostics, .graph_import, "tensor-type", "unsupported element type");
        return CompilePipelineError.InvalidStablehlo;
    };
    return .{ .element_type = element_type, .dims = dims, .layout = .dense_row_major };
}

fn bufferTypeFromMlir(element: mlir.MlirType) ?core.BufferType {
    if (mlir.mlirTypeIsAF16(element)) return .f16;
    if (mlir.mlirTypeIsAF32(element)) return .f32;
    if (mlir.mlirTypeIsAF64(element)) return .f64;
    if (mlir.mlirTypeIsABF16(element)) return .bf16;
    if (mlir.mlirTypeIsAInteger(element)) {
        const width = mlir.mlirIntegerTypeGetWidth(element);
        if (width == 1) return .pred;
        if (width == 8) return if (mlir.mlirIntegerTypeIsUnsigned(element)) .u8 else .s8;
        if (width == 16) return if (mlir.mlirIntegerTypeIsUnsigned(element)) .u16 else .s16;
        if (width == 32) return if (mlir.mlirIntegerTypeIsUnsigned(element)) .u32 else .s32;
        if (width == 64) return if (mlir.mlirIntegerTypeIsUnsigned(element)) .u64 else .s64;
    }
    return null;
}

fn dotGeneralSpec(op: mlir.MlirOperation, diagnostics: *std.Io.Writer) !compiler_facts.DotGeneralSpec {
    const attr = mlir.mlirOperationGetAttributeByName(op, mlirStringRef("dot_dimension_numbers"));
    if (mlir.mlirAttributeIsNull(attr) or !mlir.stablehloAttributeIsADotDimensionNumbers(attr)) {
        try writeStageFailure(diagnostics, .graph_import, "dot-general", "missing dot dimension numbers");
        return CompilePipelineError.InvalidStablehlo;
    }
    if (mlir.stablehloDotDimensionNumbersGetLhsContractingDimensionsSize(attr) != 1 or
        mlir.stablehloDotDimensionNumbersGetRhsContractingDimensionsSize(attr) != 1)
    {
        try writeStageFailure(diagnostics, .graph_import, "dot-general", "V0 supports exactly one contracting dimension per operand");
        return CompilePipelineError.InvalidStablehlo;
    }
    return .{
        .lhs_contracting_dimension = std.math.cast(u32, mlir.stablehloDotDimensionNumbersGetLhsContractingDimensionsElem(attr, 0)) orelse unreachable,
        .rhs_contracting_dimension = std.math.cast(u32, mlir.stablehloDotDimensionNumbersGetRhsContractingDimensionsElem(attr, 0)) orelse unreachable,
    };
}

fn broadcastDimensions(allocator: std.mem.Allocator, op: mlir.MlirOperation, diagnostics: *std.Io.Writer) ![]const u32 {
    var attr = mlir.mlirOperationGetAttributeByName(op, mlirStringRef("broadcast_dimensions"));
    if (mlir.mlirAttributeIsNull(attr)) {
        attr = mlir.mlirOperationGetAttributeByName(op, mlirStringRef("dims"));
    }
    if (mlir.mlirAttributeIsNull(attr)) {
        try writeStageFailure(diagnostics, .graph_import, "broadcast", "missing broadcast dimensions");
        return CompilePipelineError.InvalidStablehlo;
    }
    return u32ListAttribute(allocator, attr, diagnostics);
}

fn transposePermutation(allocator: std.mem.Allocator, op: mlir.MlirOperation, diagnostics: *std.Io.Writer) ![]const u32 {
    var attr = mlir.mlirOperationGetAttributeByName(op, mlirStringRef("permutation"));
    if (mlir.mlirAttributeIsNull(attr)) {
        attr = mlir.mlirOperationGetAttributeByName(op, mlirStringRef("dims"));
    }
    if (mlir.mlirAttributeIsNull(attr)) {
        try writeStageFailure(diagnostics, .graph_import, "transpose", "missing transpose permutation");
        return CompilePipelineError.InvalidStablehlo;
    }
    return u32ListAttribute(allocator, attr, diagnostics);
}

fn allReduceSpec(allocator: std.mem.Allocator, op: mlir.MlirOperation, diagnostics: *std.Io.Writer) !compiler_facts.CollectiveSpec {
    if (!allReduceRegionIsAdd(op)) {
        try writeStageFailure(diagnostics, .graph_import, "collective", "V0 only imports all_reduce regions that reduce with stablehlo.add");
        return CompilePipelineError.InvalidStablehlo;
    }
    const groups = try replicaGroups(allocator, op, diagnostics);
    errdefer allocator.free(groups.ids);
    const channel = try channelHandle(op, diagnostics);
    return .{
        .op = .all_reduce,
        .reduction = .add,
        .replica_group_count = groups.count,
        .replica_group_size = groups.size,
        .replica_groups = groups.ids,
        .channel_id = channel.id,
        .channel_type = channel.kind,
        .uses_token = false,
    };
}

const ReplicaGroups = struct {
    count: u32,
    size: u32,
    ids: []const u32,
};

const ChannelHandle = struct {
    id: ?u64,
    kind: ?u32,
};

fn replicaGroups(allocator: std.mem.Allocator, op: mlir.MlirOperation, diagnostics: *std.Io.Writer) !ReplicaGroups {
    const attr = mlir.mlirOperationGetAttributeByName(op, mlirStringRef("replica_groups"));
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsADenseIntElements(attr)) {
        try writeStageFailure(diagnostics, .graph_import, "collective", "missing dense replica_groups attribute");
        return CompilePipelineError.InvalidStablehlo;
    }

    const attr_type = mlir.mlirAttributeGetType(attr);
    if (mlir.mlirTypeIsNull(attr_type) or !mlir.mlirTypeIsAShaped(attr_type) or !mlir.mlirShapedTypeHasRank(attr_type)) {
        try writeStageFailure(diagnostics, .graph_import, "collective", "replica_groups must have ranked tensor type");
        return CompilePipelineError.InvalidStablehlo;
    }

    const rank = mlir.mlirShapedTypeGetRank(attr_type);
    if (rank != 2) {
        try writeStageFailure(diagnostics, .graph_import, "collective", "V0 requires replica_groups to be rank-2");
        return CompilePipelineError.InvalidStablehlo;
    }
    const group_count = mlir.mlirShapedTypeGetDimSize(attr_type, 0);
    const group_size = mlir.mlirShapedTypeGetDimSize(attr_type, 1);
    if (group_count <= 0 or group_size <= 0) {
        try writeStageFailure(diagnostics, .graph_import, "collective", "replica_groups must be static and non-empty");
        return CompilePipelineError.InvalidStablehlo;
    }
    const element_count = mlir.mlirElementsAttrGetNumElements(attr);
    if (element_count != group_count * group_size) {
        try writeStageFailure(diagnostics, .graph_import, "collective", "replica_groups element count does not match shape");
        return CompilePipelineError.InvalidStablehlo;
    }
    const ids = try allocator.alloc(u32, std.math.cast(usize, element_count) orelse unreachable);
    errdefer allocator.free(ids);
    var index: isize = 0;
    while (index < element_count) : (index += 1) {
        ids[std.math.cast(usize, index) orelse unreachable] = std.math.cast(u32, mlir.mlirDenseElementsAttrGetInt64Value(attr, index)) orelse {
            try writeStageFailure(diagnostics, .graph_import, "collective", "replica group participant does not fit u32");
            return CompilePipelineError.InvalidStablehlo;
        };
    }
    return .{
        .count = std.math.cast(u32, group_count) orelse {
            try writeStageFailure(diagnostics, .graph_import, "collective", "replica group count does not fit u32");
            return CompilePipelineError.InvalidStablehlo;
        },
        .size = std.math.cast(u32, group_size) orelse {
            try writeStageFailure(diagnostics, .graph_import, "collective", "replica group size does not fit u32");
            return CompilePipelineError.InvalidStablehlo;
        },
        .ids = ids,
    };
}

fn channelHandle(op: mlir.MlirOperation, diagnostics: *std.Io.Writer) !ChannelHandle {
    const attr = mlir.mlirOperationGetAttributeByName(op, mlirStringRef("channel_handle"));
    if (mlir.mlirAttributeIsNull(attr)) return .{ .id = null, .kind = null };
    if (!mlir.stablehloAttributeIsChannelHandle(attr)) {
        try writeStageFailure(diagnostics, .graph_import, "collective", "channel_handle is not a StableHLO channel handle");
        return CompilePipelineError.InvalidStablehlo;
    }
    const handle = mlir.stablehloChannelHandleGetHandle(attr);
    const channel_type = mlir.stablehloChannelHandleGetType(attr);
    if (handle < 0 or channel_type < 0) {
        try writeStageFailure(diagnostics, .graph_import, "collective", "channel_handle must be non-negative");
        return CompilePipelineError.InvalidStablehlo;
    }
    return .{
        .id = std.math.cast(u64, handle) orelse {
            try writeStageFailure(diagnostics, .graph_import, "collective", "channel handle does not fit u64");
            return CompilePipelineError.InvalidStablehlo;
        },
        .kind = std.math.cast(u32, channel_type) orelse {
            try writeStageFailure(diagnostics, .graph_import, "collective", "channel handle type does not fit u32");
            return CompilePipelineError.InvalidStablehlo;
        },
    };
}

fn allReduceRegionIsAdd(op: mlir.MlirOperation) bool {
    if (mlir.mlirOperationGetNumRegions(op) != 1) return false;
    const block = mlir.mlirRegionGetFirstBlock(mlir.mlirOperationGetRegion(op, 0));
    if (mlir.mlirBlockIsNull(block)) return false;

    var saw_add = false;
    var saw_return = false;
    var child = mlir.mlirBlockGetFirstOperation(block);
    while (!mlir.mlirOperationIsNull(child)) : (child = mlir.mlirOperationGetNextInBlock(child)) {
        const name = operationName(child);
        if (std.mem.eql(u8, name, "stablehlo.add")) {
            saw_add = true;
        } else if (std.mem.eql(u8, name, "stablehlo.return")) {
            saw_return = true;
        } else {
            return false;
        }
    }
    return saw_add and saw_return;
}

fn u32ListAttribute(allocator: std.mem.Allocator, attr: mlir.MlirAttribute, diagnostics: *std.Io.Writer) ![]const u32 {
    if (mlir.mlirAttributeIsADenseI64Array(attr)) {
        const count = mlir.mlirDenseArrayGetNumElements(attr);
        const values = try allocator.alloc(u32, std.math.cast(usize, count) orelse unreachable);
        errdefer allocator.free(values);
        var index: isize = 0;
        while (index < count) : (index += 1) {
            values[std.math.cast(usize, index) orelse unreachable] = std.math.cast(u32, mlir.mlirDenseI64ArrayGetElement(attr, index)) orelse {
                try writeStageFailure(diagnostics, .graph_import, "attribute", "dimension attribute value does not fit u32");
                return CompilePipelineError.InvalidStablehlo;
            };
        }
        return values;
    }

    if (mlir.mlirAttributeIsADenseIntElements(attr)) {
        const count = mlir.mlirElementsAttrGetNumElements(attr);
        const values = try allocator.alloc(u32, std.math.cast(usize, count) orelse unreachable);
        errdefer allocator.free(values);
        var index: isize = 0;
        while (index < count) : (index += 1) {
            values[std.math.cast(usize, index) orelse unreachable] = std.math.cast(u32, mlir.mlirDenseElementsAttrGetInt64Value(attr, index)) orelse {
                try writeStageFailure(diagnostics, .graph_import, "attribute", "dimension attribute value does not fit u32");
                return CompilePipelineError.InvalidStablehlo;
            };
        }
        return values;
    }

    try writeStageFailure(diagnostics, .graph_import, "attribute", "expected dense integer dimension list");
    return CompilePipelineError.InvalidStablehlo;
}

fn metalTarget() target_pkg.TargetDescription {
    return .{
        .name = "metal_v0",
        .kind = .metal_v0,
        .devices = &metal_devices,
        .memory_spaces = &metal_memory_spaces,
        .transfer_edges = &metal_transfer_edges,
        .execution_units = &metal_execution_units,
    };
}

fn npuTarget() target_pkg.TargetDescription {
    return .{
        .name = "npu_v0",
        .kind = .npu_v0,
        .devices = &trn2_like_devices,
        .memory_spaces = &npu_memory_spaces,
        .transfer_edges = &npu_transfer_edges,
        .execution_units = &npu_execution_units,
    };
}

const metal_memory_spaces = [_]target_pkg.TargetMemorySpace{
    .{ .id = 0, .name = "host_unpinned", .kind = .host_unpinned, .capacity_bytes = null, .bandwidth_bytes_per_second = null, .note = "bootstrap host memory" },
    .{ .id = 1, .name = "device_unified", .kind = .device_unified, .capacity_bytes = null, .bandwidth_bytes_per_second = null, .note = "Metal-visible unified device memory" },
};
const metal_dtype_rates = [_]target_pkg.DTypeRate{
    .{ .dtype = .f32, .op_class = .matmul, .ops_per_second = null, .source = .unknown, .note = "Metal V0 does not expose a stable V0 rate" },
    .{ .dtype = .f32, .op_class = .transcendental, .ops_per_second = null, .source = .unknown, .note = "Metal V0 does not expose a stable V0 rate" },
};
const metal_execution_units = [_]target_pkg.ExecutionUnit{
    .{ .id = 0, .name = "metal_shader_core", .kind = .unknown, .dtype_rates = &metal_dtype_rates },
    .{ .id = 1, .name = "unknown_gpu", .kind = .unknown, .dtype_rates = &.{} },
};
const metal_transfer_edges = [_]target_pkg.TargetTransferEdge{
    .{ .id = 0, .src_memory_space = 0, .dst_memory_space = 1, .bandwidth_bytes_per_second = null, .latency_ns = null, .supports_async = false, .engine_unit_id = null, .note = "logical H2D accounting even on unified memory" },
    .{ .id = 1, .src_memory_space = 1, .dst_memory_space = 0, .bandwidth_bytes_per_second = null, .latency_ns = null, .supports_async = false, .engine_unit_id = null, .note = "logical D2H accounting even on unified memory" },
};
const metal_device_memory_spaces = [_]u32{ 0, 1 };
const metal_device_execution_units = [_]u32{ 0, 1 };
const metal_devices = [_]target_pkg.TargetDevice{
    .{ .id = 0, .local_hardware_id = 0, .name = "metal_device", .memory_space_ids = &metal_device_memory_spaces, .execution_unit_ids = &metal_device_execution_units },
};

const npu_memory_spaces = [_]target_pkg.TargetMemorySpace{
    .{ .id = 0, .name = "host_pinned", .kind = .host_pinned, .capacity_bytes = null, .bandwidth_bytes_per_second = null, .note = "synthetic host staging memory" },
    .{ .id = 1, .name = "device_hbm", .kind = .device_hbm, .capacity_bytes = 34359738368, .bandwidth_bytes_per_second = 1000000000000, .note = "TRN2-like synthetic HBM" },
    .{ .id = 2, .name = "local_sram", .kind = .local_sram, .capacity_bytes = 67108864, .bandwidth_bytes_per_second = 20000000000000, .note = "TRN2-like synthetic local SRAM" },
};
const npu_matrix_rates = [_]target_pkg.DTypeRate{
    .{ .dtype = .bf16, .op_class = .matmul, .ops_per_second = 100000000000000, .source = .synthetic, .note = "TRN2-like synthetic roofline" },
    .{ .dtype = .f32, .op_class = .matmul, .ops_per_second = 25000000000000, .source = .synthetic, .note = "TRN2-like synthetic roofline" },
};
const npu_vector_rates = [_]target_pkg.DTypeRate{
    .{ .dtype = .f32, .op_class = .elementwise, .ops_per_second = 5000000000000, .source = .synthetic, .note = "TRN2-like synthetic roofline" },
    .{ .dtype = .f32, .op_class = .transcendental, .ops_per_second = 500000000000, .source = .synthetic, .note = "TRN2-like synthetic roofline" },
};
const npu_execution_units = [_]target_pkg.ExecutionUnit{
    .{ .id = 0, .name = "trn2_tensor_engine", .kind = .matrix, .dtype_rates = &npu_matrix_rates },
    .{ .id = 1, .name = "trn2_vector_engine", .kind = .vector, .dtype_rates = &npu_vector_rates },
    .{ .id = 2, .name = "trn2_dma_engine", .kind = .dma, .dtype_rates = &.{} },
    .{ .id = 3, .name = "trn2_collective_engine", .kind = .collective, .dtype_rates = &.{} },
};
const npu_transfer_edges = [_]target_pkg.TargetTransferEdge{
    .{ .id = 0, .src_memory_space = 0, .dst_memory_space = 1, .bandwidth_bytes_per_second = 64000000000, .latency_ns = null, .supports_async = true, .engine_unit_id = 2, .note = "synthetic H2D" },
    .{ .id = 1, .src_memory_space = 1, .dst_memory_space = 0, .bandwidth_bytes_per_second = 64000000000, .latency_ns = null, .supports_async = true, .engine_unit_id = 2, .note = "synthetic D2H" },
    .{ .id = 2, .src_memory_space = 1, .dst_memory_space = 2, .bandwidth_bytes_per_second = 20000000000000, .latency_ns = null, .supports_async = true, .engine_unit_id = 2, .note = "TRN2-like synthetic HBM to SRAM" },
    .{ .id = 3, .src_memory_space = 2, .dst_memory_space = 1, .bandwidth_bytes_per_second = 20000000000000, .latency_ns = null, .supports_async = true, .engine_unit_id = 2, .note = "synthetic SRAM to HBM" },
};
const trn2_like_device_memory_spaces = [_]u32{ 0, 1, 2 };
const trn2_like_device_execution_units = [_]u32{ 0, 1, 2, 3 };
const trn2_like_devices = [_]target_pkg.TargetDevice{
    .{ .id = 0, .local_hardware_id = 0, .name = "trn2_like_device", .memory_space_ids = &trn2_like_device_memory_spaces, .execution_unit_ids = &trn2_like_device_execution_units },
};

test "pipeline stage names are stable report strings" {
    var output = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer output.deinit();

    try writeStageName(&output.writer, .target_legality);
    try std.testing.expectEqualStrings("target_legality", output.writer.buffered());
}

test "compiler imports core without depending on legacy src namespace" {
    try std.testing.expect(core.idInBounds(0, 1));
}

test "input setup owns program bytes and parses options" {
    var options_reader: std.Io.Reader = .fixed("replicas=2; partitions=3; use_shardy=false");
    var program_reader: std.Io.Reader = .fixed("module {}");
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    var input: CompileInput = try setupCompileInputFromReader(
        std.testing.allocator,
        "stablehlo_text",
        "npu_v0",
        &options_reader,
        &program_reader,
        &diagnostics.writer,
    );
    defer input.deinit();

    const expected_replicas: u32 = 2;
    const expected_partitions: u32 = 3;
    try std.testing.expectEqual(ProgramFormat.stablehlo_text, input.program_format);
    try std.testing.expectEqual(target_pkg.TargetKind.npu_v0, input.target_kind);
    try std.testing.expectEqual(expected_replicas, input.compile_options.num_replicas);
    try std.testing.expectEqual(expected_partitions, input.compile_options.num_partitions);
    try std.testing.expect(!input.compile_options.use_shardy_partitioner);
    try std.testing.expectEqualStrings("module {}", input.program_bytes);
    try std.testing.expectEqualStrings("", diagnostics.writer.buffered());
}

test "input setup rejects unsupported format before reading target legality" {
    var options_reader: std.Io.Reader = .fixed("");
    var program_reader: std.Io.Reader = .fixed("module {}");
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    try std.testing.expectError(
        CompilePipelineError.InvalidInput,
        setupCompileInputFromReader(std.testing.allocator, "hlo_proto", "npu_v0", &options_reader, &program_reader, &diagnostics.writer),
    );
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "feature=program-format") != null);
}

test "input setup rejects empty program" {
    var options_reader: std.Io.Reader = .fixed("");
    var program_reader: std.Io.Reader = .fixed("");
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    try std.testing.expectError(
        CompilePipelineError.InvalidInput,
        setupCompileInputFromReader(std.testing.allocator, "stablehlo", "metal_v0", &options_reader, &program_reader, &diagnostics.writer),
    );
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "program is empty") != null);
}

test "compile options reject unknown keys with diagnostics" {
    var options_reader: std.Io.Reader = .fixed("surprise=1");
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    try std.testing.expectError(
        CompilePipelineError.InvalidInput,
        parseCompileOptionsFromReader(std.testing.allocator, &options_reader, &diagnostics.writer),
    );
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "unknown compile option") != null);
}

test "stablehlo text ingest parses and verifies module through MLIR C API" {
    var options_reader: std.Io.Reader = .fixed("");
    var program_reader: std.Io.Reader = .fixed(tanh_dot_bias_module);
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    var input: CompileInput = try setupCompileInputFromReader(
        std.testing.allocator,
        "stablehlo_text",
        "metal_v0",
        &options_reader,
        &program_reader,
        &diagnostics.writer,
    );
    defer input.deinit();

    var artifact: MlirModuleArtifact = try ingestStablehloText(&input, &diagnostics.writer);
    defer artifact.deinit();
    try std.testing.expect(!mlir.mlirModuleIsNull(artifact.module));
}

test "graph import creates typed V0 graph from verified StableHLO" {
    var options_reader: std.Io.Reader = .fixed("");
    var program_reader: std.Io.Reader = .fixed(tanh_dot_bias_module);
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    var input: CompileInput = try setupCompileInputFromReader(
        std.testing.allocator,
        "stablehlo_text",
        "npu_v0",
        &options_reader,
        &program_reader,
        &diagnostics.writer,
    );
    defer input.deinit();

    var artifact: MlirModuleArtifact = try ingestStablehloText(&input, &diagnostics.writer);
    defer artifact.deinit();

    var graph: GraphModule = try importGraphFromMlir(std.testing.allocator, artifact, &diagnostics.writer);
    defer graph.deinit();
    try verifyGraphModule(graph, &diagnostics.writer);

    const target: SelectedTarget = try selectTarget(.npu_v0, &diagnostics.writer);
    var session: mlir_state.MlirSession = try .initFromStablehloText(
        std.testing.allocator,
        tanh_dot_bias_module,
        .{ .program_name = "compiler_unit_test" },
        &diagnostics.writer,
    );
    defer session.deinit();
    var planned: PlannedTraceReport = try planV0TraceReportFromMlirSession(std.testing.allocator, graph, target, input.compile_options, &session, &diagnostics.writer);
    defer planned.deinit();

    const expected_sources: usize = 5;
    const expected_values: usize = 7;
    const expected_instructions: usize = 5;
    const expected_costs: usize = 4;
    const expected_lowerings: usize = 2;
    const expected_commands: usize = 3;
    const expected_schedule_overlap_records: usize = 2;
    const expected_kernel_codegen_records: usize = 2;
    const expected_fusion_groups: usize = 2;
    const expected_placements: usize = 4;
    const expected_collective_records: usize = 1;
    const expected_memory_traffic_records: usize = 4;
    const expected_epilogue_split_peak_live: u128 = 108;
    const expected_epilogue_fused_live: u128 = 188;
    const expected_epilogue_additional_live: u128 = 80;
    const expected_epilogue_global_saved: u128 = 72;
    const expected_matmul_device_read: u128 = 80;
    const expected_matmul_device_write: u128 = 24;
    const expected_matmul_codegen_inputs: u32 = 2;
    const expected_elementwise_codegen_ops: u32 = 3;
    const expected_elementwise_codegen_inputs: u32 = 2;
    const expected_elementwise_codegen_outputs: u32 = 1;
    const expected_elementwise_codegen_intermediates: u32 = 2;
    const expected_codegen_tile = [_]i64{ 2, 3 };
    const expected_codegen_result_memory: u32 = 1;
    const expected_codegen_tile_memory: ?u32 = 2;
    const expected_elementwise_global_read: u128 = 36;
    const expected_elementwise_global_write: u128 = 24;
    const expected_elementwise_local_read: u128 = 84;
    const expected_elementwise_local_write: u128 = 72;
    try std.testing.expectEqual(expected_sources, graph.sources.len);
    try std.testing.expectEqual(expected_values, graph.values.len);
    try std.testing.expectEqual(expected_instructions, graph.instructions.len);
    try std.testing.expectEqual(expected_fusion_groups, planned.report.fusion_groups.len);
    try std.testing.expectEqualStrings("matmul_epilogue", planned.report.fusion_groups[0].kind);
    try std.testing.expectEqual(FusionDecision.rejected, planned.report.fusion_groups[0].decision);
    try std.testing.expectEqual(4, planned.report.fusion_groups[0].graph_instruction_ids.len);
    try std.testing.expectEqual(expected_epilogue_split_peak_live, planned.report.fusion_groups[0].pressure_delta.split_peak_live_bytes);
    try std.testing.expectEqual(expected_epilogue_fused_live, planned.report.fusion_groups[0].pressure_delta.fused_live_bytes);
    try std.testing.expectEqual(expected_epilogue_additional_live, planned.report.fusion_groups[0].pressure_delta.additional_live_bytes);
    try std.testing.expectEqual(expected_epilogue_global_saved, planned.report.fusion_groups[0].pressure_delta.global_bytes_saved);
    try std.testing.expectEqual(expected_placements, planned.report.placement_records.len);
    try std.testing.expectEqual(expected_collective_records, planned.report.collective_plan_records.len);
    try std.testing.expectEqual(expected_costs, planned.report.cost_ledger.len);
    try std.testing.expectEqual(expected_lowerings, planned.report.lowering_records.len);
    try std.testing.expectEqual(expected_memory_traffic_records, planned.report.memory_traffic_records.len);
    try std.testing.expectEqual(expected_commands, planned.report.schedule_commands.len);
    try std.testing.expectEqual(expected_schedule_overlap_records, planned.report.schedule_overlap_records.len);
    try std.testing.expectEqual(expected_kernel_codegen_records, planned.report.kernel_codegen_records.len);
    try std.testing.expectEqual(compiler_facts.GraphInstructionKind.dot_general, graph.instructions[0].kind);
    try std.testing.expectEqual(compiler_facts.GraphInstructionKind.broadcast, graph.instructions[1].kind);
    try std.testing.expectEqual(compiler_facts.GraphInstructionKind.elementwise_binary, graph.instructions[2].kind);
    try std.testing.expectEqual(compiler_facts.GraphInstructionKind.elementwise_unary, graph.instructions[3].kind);
    try std.testing.expectEqual(compiler_facts.GraphInstructionKind.return_, graph.instructions[4].kind);
    try std.testing.expectEqual(compiler_facts.CostOpClass.matmul, planned.report.cost_ledger[0].op_class);
    try std.testing.expectEqual(compiler_facts.CostOpClass.transcendental, planned.report.cost_ledger[3].op_class);
    try std.testing.expectEqual(compiler_facts.LoweringDecision.backend_kernel_graph, planned.report.lowering_records[0].decision);
    try std.testing.expectEqual(compiler_facts.LoweringDecision.elementwise_fusion, planned.report.lowering_records[1].decision);
    try std.testing.expectEqual(compiler_facts.MemoryTrafficKind.global_memory, planned.report.memory_traffic_records[0].kind);
    try std.testing.expectEqual(compiler_facts.MemoryTrafficKind.local_memory, planned.report.memory_traffic_records[1].kind);
    try std.testing.expectEqual(expected_matmul_device_read, planned.report.memory_traffic_records[0].bytes_read);
    try std.testing.expectEqual(expected_matmul_device_write, planned.report.memory_traffic_records[0].bytes_written);
    try std.testing.expectEqual(core.ScheduleOverlapDecision.serialized, planned.report.schedule_overlap_records[0].decision);
    try std.testing.expectEqual(core.ScheduleOverlapKind.transfer_compute, planned.report.schedule_overlap_records[0].kind);
    try std.testing.expectEqual(core.ScheduleOverlapDecision.serialized, planned.report.schedule_overlap_records[1].decision);
    try std.testing.expectEqual(core.ScheduleOverlapKind.compute_transfer, planned.report.schedule_overlap_records[1].kind);
    try std.testing.expectEqual(core.KernelCodegenKind.backend_kernel_graph, planned.report.kernel_codegen_records[0].kind);
    try std.testing.expectEqualStrings("npu_matmul", planned.report.kernel_codegen_records[0].operation);
    try std.testing.expectEqual(expected_matmul_codegen_inputs, planned.report.kernel_codegen_records[0].shape.external_input_count);
    try std.testing.expectEqualSlices(i64, &expected_codegen_tile, planned.report.kernel_codegen_records[0].logical_tile_shape);
    try std.testing.expectEqual(expected_codegen_result_memory, planned.report.kernel_codegen_records[0].result_memory_space_id);
    try std.testing.expectEqual(expected_codegen_tile_memory, planned.report.kernel_codegen_records[0].tile_memory_space_id);
    try std.testing.expectEqual(expected_matmul_device_read, planned.report.kernel_codegen_records[0].memory_pressure.global_bytes_read);
    try std.testing.expectEqual(expected_matmul_device_write, planned.report.kernel_codegen_records[0].memory_pressure.global_bytes_written);
    try std.testing.expectEqual(expected_matmul_device_read, planned.report.kernel_codegen_records[0].memory_pressure.local_bytes_read);
    try std.testing.expectEqual(expected_matmul_device_write, planned.report.kernel_codegen_records[0].memory_pressure.local_bytes_written);
    try std.testing.expectEqual(core.KernelCodegenKind.elementwise_fusion_kernel, planned.report.kernel_codegen_records[1].kind);
    try std.testing.expectEqualStrings("npu_elementwise_fusion", planned.report.kernel_codegen_records[1].operation);
    try std.testing.expectEqual(expected_elementwise_codegen_ops, planned.report.kernel_codegen_records[1].shape.operation_count);
    try std.testing.expectEqual(expected_elementwise_codegen_inputs, planned.report.kernel_codegen_records[1].shape.external_input_count);
    try std.testing.expectEqual(expected_elementwise_codegen_outputs, planned.report.kernel_codegen_records[1].shape.external_output_count);
    try std.testing.expectEqual(expected_elementwise_codegen_intermediates, planned.report.kernel_codegen_records[1].shape.intermediate_value_count);
    try std.testing.expectEqualSlices(i64, &expected_codegen_tile, planned.report.kernel_codegen_records[1].logical_tile_shape);
    try std.testing.expectEqual(expected_codegen_result_memory, planned.report.kernel_codegen_records[1].result_memory_space_id);
    try std.testing.expectEqual(expected_codegen_tile_memory, planned.report.kernel_codegen_records[1].tile_memory_space_id);
    try std.testing.expectEqual(expected_elementwise_global_read, planned.report.kernel_codegen_records[1].memory_pressure.global_bytes_read);
    try std.testing.expectEqual(expected_elementwise_global_write, planned.report.kernel_codegen_records[1].memory_pressure.global_bytes_written);
    try std.testing.expectEqual(expected_elementwise_local_read, planned.report.kernel_codegen_records[1].memory_pressure.local_bytes_read);
    try std.testing.expectEqual(expected_elementwise_local_write, planned.report.kernel_codegen_records[1].memory_pressure.local_bytes_written);
    const expected_fused_instructions: usize = 3;
    try std.testing.expectEqual(expected_fused_instructions, planned.report.lowering_records[1].graph_instruction_ids.len);
    try std.testing.expectEqual(core.CommandKind.backend_execute, planned.report.schedule_commands[1].kind);
    try std.testing.expectEqual(core.BackendKind.npu_v0, planned.report.backend_bindings[0].backend_kind);
    const expected_matrix_unit: ?u32 = 0;
    const expected_vector_unit: ?u32 = 1;
    const expected_mixed_binding_unit: ?u32 = null;
    try std.testing.expectEqual(expected_matrix_unit, planned.report.cost_ledger[0].expected_unit_id);
    try std.testing.expectEqual(expected_vector_unit, planned.report.cost_ledger[1].expected_unit_id);
    try std.testing.expectEqual(expected_mixed_binding_unit, planned.report.backend_bindings[0].expected_unit_id);

    switch (graph.instructions[0].payload) {
        .dot_general => |payload| {
            const lhs_contracting: u32 = 1;
            const rhs_contracting: u32 = 0;
            try std.testing.expectEqual(lhs_contracting, payload.lhs_contracting_dimension);
            try std.testing.expectEqual(rhs_contracting, payload.rhs_contracting_dimension);
        },
        else => return error.TestExpectedEqual,
    }
    switch (graph.instructions[1].payload) {
        .broadcast => |payload| try std.testing.expectEqualSlices(u32, &[_]u32{1}, payload.dimensions),
        else => return error.TestExpectedEqual,
    }
    try std.testing.expectEqualStrings("", diagnostics.writer.buffered());
}

test "tile_legality_verify rejects codegen records exceeding local SRAM capacity" {
    var options_reader: std.Io.Reader = .fixed("");
    var program_reader: std.Io.Reader = .fixed(tanh_dot_bias_module);
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    var input: CompileInput = try setupCompileInputFromReader(
        std.testing.allocator,
        "stablehlo_text",
        "npu_v0",
        &options_reader,
        &program_reader,
        &diagnostics.writer,
    );
    defer input.deinit();

    var artifact: MlirModuleArtifact = try ingestStablehloText(&input, &diagnostics.writer);
    defer artifact.deinit();

    var graph: GraphModule = try importGraphFromMlir(std.testing.allocator, artifact, &diagnostics.writer);
    defer graph.deinit();
    try verifyGraphModule(graph, &diagnostics.writer);

    var target: SelectedTarget = try selectTarget(.npu_v0, &diagnostics.writer);
    const tiny_sram_capacity: u64 = 64;
    const low_capacity_memory_spaces = [_]target_pkg.TargetMemorySpace{
        .{ .id = 0, .name = "host_pinned", .kind = .host_pinned, .capacity_bytes = null, .bandwidth_bytes_per_second = null, .note = "synthetic host staging memory" },
        .{ .id = 1, .name = "device_hbm", .kind = .device_hbm, .capacity_bytes = 34359738368, .bandwidth_bytes_per_second = 1000000000000, .note = "TRN2-like synthetic HBM" },
        .{ .id = 2, .name = "local_sram", .kind = .local_sram, .capacity_bytes = tiny_sram_capacity, .bandwidth_bytes_per_second = 20000000000000, .note = "intentionally tiny local SRAM for tile legality test" },
    };
    target.description.memory_spaces = &low_capacity_memory_spaces;

    var session: mlir_state.MlirSession = try .initFromStablehloText(
        std.testing.allocator,
        large_tiling_module,
        .{ .program_name = "compiler_tile_legality_test" },
        &diagnostics.writer,
    );
    defer session.deinit();
    try std.testing.expectError(
        CompilePipelineError.InvalidLowering,
        planV0TraceReportFromMlirSession(std.testing.allocator, graph, target, input.compile_options, &session, &diagnostics.writer),
    );
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "pass=tile_legality_verify") != null);
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "tile live bytes exceed memory capacity") != null);
}

test "graph import rejects dynamic shapes in V0" {
    var options_reader: std.Io.Reader = .fixed("");
    var program_reader: std.Io.Reader = .fixed(dynamic_shape_module);
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    var input: CompileInput = try setupCompileInputFromReader(
        std.testing.allocator,
        "stablehlo_text",
        "npu_v0",
        &options_reader,
        &program_reader,
        &diagnostics.writer,
    );
    defer input.deinit();

    var artifact: MlirModuleArtifact = try ingestStablehloText(&input, &diagnostics.writer);
    defer artifact.deinit();

    try std.testing.expectError(CompilePipelineError.InvalidStablehlo, importGraphFromMlir(std.testing.allocator, artifact, &diagnostics.writer));
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "dynamic dimensions are unsupported") != null);
}

test "graph import rejects unsupported StableHLO before target legality" {
    var options_reader: std.Io.Reader = .fixed("");
    var program_reader: std.Io.Reader = .fixed(unsupported_multiply_module);
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    var input: CompileInput = try setupCompileInputFromReader(
        std.testing.allocator,
        "stablehlo_text",
        "npu_v0",
        &options_reader,
        &program_reader,
        &diagnostics.writer,
    );
    defer input.deinit();

    var artifact: MlirModuleArtifact = try ingestStablehloText(&input, &diagnostics.writer);
    defer artifact.deinit();

    try std.testing.expectError(CompilePipelineError.InvalidStablehlo, importGraphFromMlir(std.testing.allocator, artifact, &diagnostics.writer));
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "feature=unsupported-op") != null);
}

test "graph verification rejects invalid dot shape before backend binding" {
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    const source: compiler_facts.SourceRef = .{ .id = .{ .index = 0 }, .frontend = .stablehlo, .op_name = "stablehlo.dot_general", .source_index = 0, .location = "" };
    const lhs_dims = [_]i64{ 2, 4 };
    const rhs_dims = [_]i64{ 5, 3 };
    const output_dims = [_]i64{ 2, 3 };
    var values = [_]compiler_facts.GraphValue{
        .{ .id = .{ .index = 0 }, .ty = .{ .element_type = .f32, .dims = &lhs_dims, .layout = .dense_row_major }, .role = .parameter, .source = null },
        .{ .id = .{ .index = 1 }, .ty = .{ .element_type = .f32, .dims = &rhs_dims, .layout = .dense_row_major }, .role = .parameter, .source = null },
        .{ .id = .{ .index = 2 }, .ty = .{ .element_type = .f32, .dims = &output_dims, .layout = .dense_row_major }, .role = .instruction_result, .source = source },
    };
    var inputs = [_]compiler_facts.GraphValueId{ .{ .index = 0 }, .{ .index = 1 } };
    var outputs = [_]compiler_facts.GraphValueId{.{ .index = 2 }};
    var instructions = [_]compiler_facts.GraphInstruction{
        .{
            .id = .{ .index = 0 },
            .kind = .dot_general,
            .inputs = &inputs,
            .outputs = &outputs,
            .payload = .{ .dot_general = .{ .lhs_contracting_dimension = 1, .rhs_contracting_dimension = 0 } },
            .source = source,
        },
    };
    var sources = [_]compiler_facts.SourceRef{source};
    const graph: GraphModule = .{
        .allocator = std.testing.allocator,
        .sources = &sources,
        .values = &values,
        .instructions = &instructions,
    };

    try std.testing.expectError(CompilePipelineError.InvalidGraph, verifyGraphModule(graph, &diagnostics.writer));
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "contracting dimensions have different sizes") != null);
}

test "stablehlo text ingest rejects malformed module before graph import" {
    var options_reader: std.Io.Reader = .fixed("");
    var program_reader: std.Io.Reader = .fixed("module { stablehlo.nope }");
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    var input: CompileInput = try setupCompileInputFromReader(
        std.testing.allocator,
        "stablehlo_text",
        "metal_v0",
        &options_reader,
        &program_reader,
        &diagnostics.writer,
    );
    defer input.deinit();

    try std.testing.expectError(CompilePipelineError.InvalidStablehlo, ingestStablehloText(&input, &diagnostics.writer));
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "MLIR parser rejected module") != null);
}

test "stablehlo bytecode ingest fails explicitly until portable artifact support lands" {
    var options_reader: std.Io.Reader = .fixed("");
    var program_reader: std.Io.Reader = .fixed("not bytecode");
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    var input: CompileInput = try setupCompileInputFromReader(
        std.testing.allocator,
        "stablehlo_bytecode",
        "metal_v0",
        &options_reader,
        &program_reader,
        &diagnostics.writer,
    );
    defer input.deinit();

    try std.testing.expectError(CompilePipelineError.InvalidStablehlo, ingestStablehloText(&input, &diagnostics.writer));
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "only StableHLO text ingest is implemented") != null);
}

test "target selection returns validated NPU model" {
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    const selected: SelectedTarget = try selectTarget(.npu_v0, &diagnostics.writer);
    try std.testing.expectEqual(target_pkg.TargetKind.npu_v0, selected.description.kind);
    try std.testing.expectEqualStrings("", diagnostics.writer.buffered());
}

test "target legality rejects missing backend dtype capability before lowering" {
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    const source: compiler_facts.SourceRef = .{ .id = .{ .index = 0 }, .frontend = .stablehlo, .op_name = "stablehlo.tanh", .source_index = 0, .location = "" };
    const dims = [_]i64{4};
    var values = [_]compiler_facts.GraphValue{
        .{ .id = .{ .index = 0 }, .ty = .{ .element_type = .f64, .dims = &dims, .layout = .dense_row_major }, .role = .parameter, .source = null },
        .{ .id = .{ .index = 1 }, .ty = .{ .element_type = .f64, .dims = &dims, .layout = .dense_row_major }, .role = .instruction_result, .source = source },
    };
    var inputs = [_]compiler_facts.GraphValueId{.{ .index = 0 }};
    var outputs = [_]compiler_facts.GraphValueId{.{ .index = 1 }};
    var instructions = [_]compiler_facts.GraphInstruction{
        .{ .id = .{ .index = 0 }, .kind = .elementwise_unary, .inputs = &inputs, .outputs = &outputs, .payload = .{ .elementwise_unary = .{ .op = .tanh } }, .source = source },
    };
    var sources = [_]compiler_facts.SourceRef{source};
    const graph: GraphModule = .{
        .allocator = std.testing.allocator,
        .sources = &sources,
        .values = &values,
        .instructions = &instructions,
    };

    const target: SelectedTarget = try selectTarget(.npu_v0, &diagnostics.writer);
    try std.testing.expectError(CompilePipelineError.UnsupportedTargetFeature, verifyTargetLegality(graph, target, &diagnostics.writer));
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "backend lacks support") != null);
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "dtype=f64") != null);
}

test "backend binding verification rejects missing binding before executable creation" {
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    const target: SelectedTarget = try selectTarget(.metal_v0, &diagnostics.writer);
    const report: core.TraceReport = reportWithBackendCommandWithoutBinding();
    try std.testing.expectError(CompilePipelineError.InvalidBackendBinding, createExecutableView(report, target, &diagnostics.writer));
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "backend command has no binding") != null);
}

test "backend binding verification rejects backend target mismatch" {
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    const target: SelectedTarget = try selectTarget(.npu_v0, &diagnostics.writer);
    const report: core.TraceReport = executableReport(.metal_v0);
    try std.testing.expectError(CompilePipelineError.InvalidBackendBinding, createExecutableView(report, target, &diagnostics.writer));
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "binding backend does not match selected target") != null);
}

test "schedule verification rejects backend command without dependency" {
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    try std.testing.expectError(CompilePipelineError.InvalidSchedule, verifySchedule(reportWithBackendCommandWithoutDependency(), &diagnostics.writer));
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "backend command must depend on earlier transfer") != null);
}

test "schedule verification rejects future dependency" {
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    try std.testing.expectError(CompilePipelineError.InvalidSchedule, verifySchedule(futureDependencyReport(), &diagnostics.writer));
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "dependency must be earlier") != null);
}

test "executable view requires validated report and backend binding" {
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    const target: SelectedTarget = try selectTarget(.metal_v0, &diagnostics.writer);
    const report: core.TraceReport = executableReport(.metal_v0);
    const executable: CompiledExecutableView = try createExecutableView(report, target, &diagnostics.writer);
    try std.testing.expectEqual(report.schedule_commands.len, executable.report.schedule_commands.len);
}

fn futureDependencyReport() core.TraceReport {
    return .{
        .sources = &executable_sources,
        .graph_values = &executable_values,
        .graph_instructions = &executable_instructions,
        .cost_ledger = &executable_costs,
        .lowering_records = &executable_lowerings,
        .schedule_commands = &future_dependency_commands,
        .backend_bindings = &metal_executable_bindings,
        .profile_events = &.{},
        .explain_records = &.{},
    };
}

fn reportWithBackendCommandWithoutDependency() core.TraceReport {
    return .{
        .sources = &executable_sources,
        .graph_values = &executable_values,
        .graph_instructions = &executable_instructions,
        .cost_ledger = &executable_costs,
        .lowering_records = &executable_lowerings,
        .schedule_commands = &backend_without_dependency_commands,
        .backend_bindings = &backend_without_dependency_bindings,
        .profile_events = &.{},
        .explain_records = &.{},
    };
}

fn reportWithBackendCommandWithoutBinding() core.TraceReport {
    return .{
        .sources = &executable_sources,
        .graph_values = &executable_values,
        .graph_instructions = &executable_instructions,
        .cost_ledger = &executable_costs,
        .lowering_records = &executable_lowerings,
        .schedule_commands = &executable_commands,
        .backend_bindings = &.{},
        .profile_events = &.{},
        .explain_records = &.{},
    };
}

fn executableReport(backend_kind: core.BackendKind) core.TraceReport {
    const bindings = switch (backend_kind) {
        .metal_v0 => &metal_executable_bindings,
        .npu_v0 => &npu_executable_bindings,
    };
    return .{
        .sources = &executable_sources,
        .graph_values = &executable_values,
        .graph_instructions = &executable_instructions,
        .cost_ledger = &executable_costs,
        .lowering_records = &executable_lowerings,
        .schedule_commands = &executable_commands,
        .backend_bindings = bindings,
        .profile_events = &.{},
        .explain_records = &.{},
    };
}

const executable_source: compiler_facts.SourceRef = .{ .id = .{ .index = 0 }, .frontend = .stablehlo, .op_name = "stablehlo.tanh", .source_index = 0, .location = "" };
const executable_sources = [_]compiler_facts.SourceRef{executable_source};
const executable_dims = [_]i64{4};
const executable_values = [_]compiler_facts.GraphValue{
    .{ .id = .{ .index = 0 }, .ty = .{ .element_type = .f32, .dims = &executable_dims, .layout = .dense_row_major }, .role = .parameter, .source = executable_source },
    .{ .id = .{ .index = 1 }, .ty = .{ .element_type = .f32, .dims = &executable_dims, .layout = .dense_row_major }, .role = .instruction_result, .source = executable_source },
};
const executable_inputs = [_]compiler_facts.GraphValueId{.{ .index = 0 }};
const executable_outputs = [_]compiler_facts.GraphValueId{.{ .index = 1 }};
const executable_instruction_refs = [_]compiler_facts.GraphInstructionId{.{ .index = 0 }};
const executable_cost_refs = [_]compiler_facts.CostLedgerId{.{ .index = 0 }};
const executable_lowering_refs = [_]compiler_facts.LoweringRecordId{.{ .index = 0 }};
const executable_instructions = [_]compiler_facts.GraphInstruction{
    .{ .id = .{ .index = 0 }, .kind = .elementwise_unary, .inputs = &executable_inputs, .outputs = &executable_outputs, .payload = .{ .elementwise_unary = .{ .op = .tanh } }, .source = executable_source },
};
const executable_costs = [_]compiler_facts.CostLedgerEntry{
    .{ .id = .{ .index = 0 }, .source = executable_source, .graph_instruction_ids = &executable_instruction_refs, .op_class = .transcendental, .dtype = .f32, .accumulation_dtype = null, .logical_ops = 4, .bytes_read = 16, .bytes_written = 16, .expected_unit_id = null, .formula = "numel", .approximation = "exact" },
};
const executable_lowerings = [_]compiler_facts.LoweringRecord{
    .{ .id = .{ .index = 0 }, .graph_instruction_ids = &executable_instruction_refs, .decision = .backend_kernel_graph, .reason = "v0", .rejected_alternatives = &.{}, .cost_ledger_ids = &executable_cost_refs },
};
const executable_backend_dependencies = [_]core.CommandDependency{.{ .command_id = .{ .index = 0 }, .kind = .data }};
const executable_commands = [_]core.ScheduleCommand{
    .{ .id = .{ .index = 0 }, .kind = .host_to_device, .stream = .{ .index = 0 }, .inputs = &executable_inputs, .outputs = &executable_inputs, .dependencies = &.{}, .lowering_record_ids = &.{}, .cost_ledger_ids = &.{} },
    .{ .id = .{ .index = 1 }, .kind = .backend_execute, .stream = .{ .index = 0 }, .inputs = &executable_inputs, .outputs = &executable_outputs, .dependencies = &executable_backend_dependencies, .lowering_record_ids = &executable_lowering_refs, .cost_ledger_ids = &executable_cost_refs },
};
const backend_without_dependency_commands = [_]core.ScheduleCommand{
    .{ .id = .{ .index = 0 }, .kind = .backend_execute, .stream = .{ .index = 0 }, .inputs = &executable_inputs, .outputs = &executable_outputs, .dependencies = &.{}, .lowering_record_ids = &executable_lowering_refs, .cost_ledger_ids = &executable_cost_refs },
};
const backend_without_dependency_bindings = [_]core.BackendBinding{
    .{ .id = .{ .index = 0 }, .command_id = .{ .index = 0 }, .backend_kind = .metal_v0, .backend_operation = "metal_mls_graph_execute", .graph_instruction_ids = &executable_instruction_refs, .expected_unit_id = null, .cost_ledger_ids = &executable_cost_refs },
};
const future_dependency_refs = [_]core.CommandDependency{.{ .command_id = .{ .index = 1 }, .kind = .data }};
const future_dependency_commands = [_]core.ScheduleCommand{
    .{ .id = .{ .index = 0 }, .kind = .backend_execute, .stream = .{ .index = 0 }, .inputs = &executable_inputs, .outputs = &executable_outputs, .dependencies = &future_dependency_refs, .lowering_record_ids = &executable_lowering_refs, .cost_ledger_ids = &executable_cost_refs },
    .{ .id = .{ .index = 1 }, .kind = .event_record, .stream = .{ .index = 0 }, .inputs = &.{}, .outputs = &.{}, .dependencies = &.{}, .lowering_record_ids = &.{}, .cost_ledger_ids = &.{} },
};
const metal_executable_bindings = [_]core.BackendBinding{
    .{ .id = .{ .index = 0 }, .command_id = .{ .index = 1 }, .backend_kind = .metal_v0, .backend_operation = "metal_mls_graph_execute", .graph_instruction_ids = &executable_instruction_refs, .expected_unit_id = null, .cost_ledger_ids = &executable_cost_refs },
};
const npu_executable_bindings = [_]core.BackendBinding{
    .{ .id = .{ .index = 0 }, .command_id = .{ .index = 1 }, .backend_kind = .npu_v0, .backend_operation = "npu_execute", .graph_instruction_ids = &executable_instruction_refs, .expected_unit_id = 1, .cost_ledger_ids = &executable_cost_refs },
};

const tanh_dot_bias_module =
    \\module {
    \\  func.func @main(%arg0: tensor<2x4xf32>, %arg1: tensor<4x3xf32>, %arg2: tensor<3xf32>) -> tensor<2x3xf32> {
    \\    %0 = stablehlo.dot_general %arg0, %arg1,
    \\      contracting_dims = [1] x [0],
    \\      precision = [DEFAULT, DEFAULT]
    \\      : (tensor<2x4xf32>, tensor<4x3xf32>) -> tensor<2x3xf32>
    \\    %1 = stablehlo.broadcast_in_dim %arg2, dims = [1] : (tensor<3xf32>) -> tensor<2x3xf32>
    \\    %2 = stablehlo.add %0, %1 : tensor<2x3xf32>
    \\    %3 = stablehlo.tanh %2 : tensor<2x3xf32>
    \\    return %3 : tensor<2x3xf32>
    \\  }
    \\}
;

const large_tiling_module =
    \\module {
    \\  func.func @main(%arg0: tensor<4096x4096xf32>, %arg1: tensor<4096x4096xf32>) -> tensor<4096x4096xf32> {
    \\    %0 = stablehlo.dot_general %arg0, %arg1,
    \\      contracting_dims = [1] x [0],
    \\      precision = [DEFAULT, DEFAULT] : (tensor<4096x4096xf32>, tensor<4096x4096xf32>) -> tensor<4096x4096xf32>
    \\    return %0 : tensor<4096x4096xf32>
    \\  }
    \\}
;

const unsupported_multiply_module =
    \\module {
    \\  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
    \\    %0 = stablehlo.multiply %arg0, %arg1 : tensor<2xf32>
    \\    return %0 : tensor<2xf32>
    \\  }
    \\}
;

const dynamic_shape_module =
    \\module {
    \\  func.func @main(%arg0: tensor<?xf32>) -> tensor<?xf32> {
    \\    return %arg0 : tensor<?xf32>
    \\  }
    \\}
;
