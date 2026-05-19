const std = @import("std");
const compiler = @import("pjrtx/compiler");
const compiler_facts = @import("pjrtx/compiler/facts");
const mlir_state = @import("pjrtx/compiler/mlir_state");
const core = @import("pjrtx/core");
const runfiles = @import("runfiles.zig");

test "compiler middle records Shardy-aware passes fusion placement and collectives" {
    const fixture: []u8 = try runfiles.readRunfile(std.testing.allocator, "pjrtx/tests/fixtures/shardy_tiny_mesh.mlir");
    defer std.testing.allocator.free(fixture);

    var options_reader: std.Io.Reader = .fixed("use_shardy=true");
    var program_reader: std.Io.Reader = .fixed(fixture);
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    var input: compiler.CompileInput = try compiler.setupCompileInputFromReader(
        std.testing.allocator,
        "stablehlo_text",
        "npu_v0",
        &options_reader,
        &program_reader,
        &diagnostics.writer,
    );
    defer input.deinit();

    var module: compiler.MlirModuleArtifact = try compiler.ingestStablehloText(&input, &diagnostics.writer);
    defer module.deinit();

    var pass_report: compiler.MlirPassPipelineReport = try compiler.buildMlirPassPipelineReport(std.testing.allocator, &input, module, &diagnostics.writer);
    defer pass_report.deinit();

    var graph: compiler.GraphModule = try compiler.importGraphFromMlir(std.testing.allocator, module, &diagnostics.writer);
    defer graph.deinit();
    try compiler.verifyGraphModule(graph, &diagnostics.writer);

    const target: compiler.SelectedTarget = try compiler.selectTarget(.npu_v0, &diagnostics.writer);
    var session: mlir_state.MlirSession = try .initFromStablehloText(
        std.testing.allocator,
        fixture,
        .{ .program_name = "lowering_test" },
        &diagnostics.writer,
    );
    defer session.deinit();

    var planned: compiler.PlannedTraceReport = try compiler.planV0TraceReportFromMlirSession(
        std.testing.allocator,
        graph,
        target,
        input.compile_options,
        &session,
        &diagnostics.writer,
    );
    defer planned.deinit();

    const expected_passes: usize = 5;
    const expected_fusion_groups: usize = 2;
    const expected_placements: usize = 4;
    const expected_graph_instructions: u32 = 5;
    const expected_tile_memory: ?u32 = 2;
    const expected_split_kernels: u32 = 2;
    const expected_fused_kernels: u32 = 1;
    const expected_split_peak_live_bytes: u128 = 108;
    const expected_fused_live_bytes: u128 = 188;
    const expected_additional_live_bytes: u128 = 80;
    const expected_global_bytes_saved: u128 = 72;
    try std.testing.expect(pass_report.has_shardy_metadata);
    try std.testing.expect(pass_report.shardy_requested);
    try std.testing.expectEqual(expected_passes, pass_report.records.len);
    try std.testing.expectEqualStrings("collective_graph_payload_import", pass_report.records[2].pass_name);
    try std.testing.expectEqual(compiler.MlirPassStatus.ok, pass_report.records[2].status);
    try std.testing.expectEqualStrings("shardy_metadata_propagation_report", pass_report.records[4].pass_name);
    try std.testing.expectEqual(compiler.MlirPassStatus.ok, pass_report.records[4].status);
    try std.testing.expect(pass_report.records[4].preserves_shardy_metadata);
    try std.testing.expectEqual(expected_fusion_groups, planned.report.fusion_groups.len);
    try std.testing.expectEqual(compiler.FusionDecision.rejected, planned.report.fusion_groups[0].decision);
    try std.testing.expectEqualStrings("matmul_epilogue", planned.report.fusion_groups[0].kind);
    try std.testing.expectEqual(expected_split_kernels, planned.report.fusion_groups[0].pressure_delta.split_kernel_count);
    try std.testing.expectEqual(expected_fused_kernels, planned.report.fusion_groups[0].pressure_delta.fused_kernel_count);
    try std.testing.expectEqual(expected_split_peak_live_bytes, planned.report.fusion_groups[0].pressure_delta.split_peak_live_bytes);
    try std.testing.expectEqual(expected_fused_live_bytes, planned.report.fusion_groups[0].pressure_delta.fused_live_bytes);
    try std.testing.expectEqual(expected_additional_live_bytes, planned.report.fusion_groups[0].pressure_delta.additional_live_bytes);
    try std.testing.expectEqual(expected_global_bytes_saved, planned.report.fusion_groups[0].pressure_delta.global_bytes_saved);
    try std.testing.expectEqual(compiler.FusionDecision.accepted, planned.report.fusion_groups[1].decision);
    try std.testing.expectEqual(expected_placements, planned.report.placement_records.len);
    try std.testing.expectEqual(expected_tile_memory, planned.report.placement_records[0].tile_memory_space_id);
    try std.testing.expectEqual(compiler_facts.CollectivePlanDecision.no_collectives, planned.report.collective_plan_records[0].decision);
    try std.testing.expectEqual(compiler_facts.CollectiveAlgorithm.none, planned.report.collective_plan_records[0].algorithm);
    try std.testing.expectEqual(expected_graph_instructions, planned.report.collective_plan_records[0].checked_graph_instruction_count);
    try std.testing.expectEqualStrings("", diagnostics.writer.buffered());

    var summary: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer summary.deinit();
    try compiler.writeMlirPassPipelineReport(pass_report, &summary.writer);
    try core.writeTraceReportSummary(planned.report, &summary.writer);

    const text = summary.writer.buffered();
    try std.testing.expect(std.mem.indexOf(u8, text, "shardy_present=true shardy_requested=true") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "pass.2 name=collective_graph_payload_import status=ok") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "pass.4 name=shardy_metadata_propagation_report status=ok") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "fusion.0 decision=rejected kind=matmul_epilogue instructions=0,1,2,3") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "pressure=split_kernels:2,fused_kernels:1,split_peak:108,fused_live:188,additional_live:80,global_saved:72") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "fusion.1 decision=accepted kind=elementwise_chain instructions=1,2,3") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "placement.0 instruction=0 outputs=3 layout=dense_row_major tile=2x3 result_memory=1 tile_memory=2") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "collective.0 decision=no_collectives algorithm=none checked=5 lowered=0 unsupported=0 estimated_bytes=0 estimated_latency_ns=unknown") != null);
}

test "compile orchestration joins Shardy pass records into final trace" {
    const fixture: []u8 = try runfiles.readRunfile(std.testing.allocator, "pjrtx/tests/fixtures/shardy_tiny_mesh.mlir");
    defer std.testing.allocator.free(fixture);

    var options_reader: std.Io.Reader = .fixed("use_shardy=true");
    var program_reader: std.Io.Reader = .fixed(fixture);
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    var compiled: compiler.CompiledV0 = try compiler.compileV0FromReader(
        std.testing.allocator,
        "stablehlo_text",
        "npu_v0",
        &options_reader,
        &program_reader,
        &diagnostics.writer,
    );
    defer compiled.deinit();

    const report = compiled.report();
    const expected_passes: usize = 5;
    const expected_fusion_groups: usize = 2;
    const expected_placements: usize = 4;
    const expected_collective_records: usize = 1;
    const expected_lowerings: usize = 2;
    const expected_memory_traffic_records: usize = 4;
    const expected_explains: usize = 3;
    const expected_fused_instructions: usize = 3;
    try std.testing.expectEqual(expected_passes, report.mlir_pass_records.len);
    try std.testing.expectEqual(1, report.graph_rewrite_records.len);
    try std.testing.expectEqual(compiler_facts.GraphRewriteDecision.rejected, report.graph_rewrite_records[0].decision);
    try std.testing.expectEqualStrings("broadcast_simplify", report.graph_rewrite_records[0].pass_name);
    try std.testing.expectEqualStrings("collective_graph_payload_import", report.mlir_pass_records[2].pass_name);
    try std.testing.expectEqual(compiler.MlirPassStatus.ok, report.mlir_pass_records[2].status);
    try std.testing.expectEqualStrings("shardy_metadata_propagation_report", report.mlir_pass_records[4].pass_name);
    try std.testing.expect(report.mlir_pass_records[4].preserves_shardy_metadata);
    try std.testing.expectEqual(expected_fusion_groups, report.fusion_groups.len);
    try std.testing.expectEqual(expected_placements, report.placement_records.len);
    try std.testing.expectEqual(expected_collective_records, report.collective_plan_records.len);
    try std.testing.expectEqual(expected_lowerings, report.lowering_records.len);
    try std.testing.expectEqual(expected_memory_traffic_records, report.memory_traffic_records.len);
    try std.testing.expectEqual(expected_explains, report.explain_records.len);
    try std.testing.expectEqual(expected_fused_instructions, report.lowering_records[1].graph_instruction_ids.len);
    try std.testing.expectEqualStrings("matmul_epilogue", report.fusion_groups[0].kind);
    try std.testing.expectEqual(compiler_facts.MemoryTrafficKind.global_memory, report.memory_traffic_records[2].kind);
    try std.testing.expectEqual(compiler_facts.MemoryTrafficKind.local_memory, report.memory_traffic_records[3].kind);
    try std.testing.expectEqual(core.ExplainSubject{ .lowering_record = .{ .index = 0 } }, report.explain_records[0].subject);
    try std.testing.expectEqual(core.ExplainSubject{ .lowering_record = .{ .index = 1 } }, report.explain_records[1].subject);
    try std.testing.expectEqualStrings("elementwise_fusion", report.explain_records[1].decision);
    try std.testing.expectEqual(compiler.FusionDecision.accepted, report.fusion_groups[1].decision);
    try std.testing.expectEqualStrings("", diagnostics.writer.buffered());

    var summary: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer summary.deinit();
    try compiler.writeMlirPassPipelineReport(.{
        .allocator = std.testing.allocator,
        .has_shardy_metadata = true,
        .shardy_requested = true,
        .records = report.mlir_pass_records,
    }, &summary.writer);
    try std.testing.expect(std.mem.indexOf(u8, summary.writer.buffered(), "pass.4 name=shardy_metadata_propagation_report status=ok") != null);
}

test "unsupported collective fails during compiler-middle lowering gate" {
    const fixture: []u8 = try runfiles.readRunfile(std.testing.allocator, "pjrtx/tests/fixtures/unsupported_all_reduce.mlir");
    defer std.testing.allocator.free(fixture);

    var options_reader: std.Io.Reader = .fixed("replicas=2; use_shardy=true");
    var program_reader: std.Io.Reader = .fixed(fixture);
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    var input: compiler.CompileInput = try compiler.setupCompileInputFromReader(
        std.testing.allocator,
        "stablehlo_text",
        "npu_v0",
        &options_reader,
        &program_reader,
        &diagnostics.writer,
    );
    defer input.deinit();

    var module: compiler.MlirModuleArtifact = try compiler.ingestStablehloText(&input, &diagnostics.writer);
    defer module.deinit();

    var pass_report: compiler.MlirPassPipelineReport = try compiler.buildMlirPassPipelineReport(std.testing.allocator, &input, module, &diagnostics.writer);
    defer pass_report.deinit();

    var graph: compiler.GraphModule = try compiler.importGraphFromMlir(std.testing.allocator, module, &diagnostics.writer);
    defer graph.deinit();
    try compiler.verifyGraphModule(graph, &diagnostics.writer);
    try std.testing.expectEqual(compiler_facts.GraphInstructionKind.collective, graph.instructions[0].kind);
    switch (graph.instructions[0].payload) {
        .collective => |payload| {
            try std.testing.expectEqual(compiler_facts.CollectiveOp.all_reduce, payload.op);
            try std.testing.expectEqual(compiler_facts.CollectiveReduction.add, payload.reduction);
            try std.testing.expectEqual(1, payload.replica_group_count);
            try std.testing.expectEqual(2, payload.replica_group_size);
            try std.testing.expectEqualSlices(u32, &[_]u32{ 0, 1 }, payload.replica_groups);
            try std.testing.expectEqual(null, payload.channel_id);
            try std.testing.expectEqual(null, payload.channel_type);
            try std.testing.expect(!payload.uses_token);
        },
        else => return error.TestExpectedEqual,
    }

    const target: compiler.SelectedTarget = try compiler.selectTarget(.npu_v0, &diagnostics.writer);
    try compiler.verifyCollectiveGroupChannel(graph, target, input.compile_options, &diagnostics.writer);
    const collective_plan: compiler.CollectivePlan = try compiler.planCollectives(graph, target, &diagnostics.writer);
    try std.testing.expectEqual(compiler_facts.CollectivePlanDecision.unsupported, collective_plan.decision);
    try std.testing.expectEqual(compiler_facts.CollectiveAlgorithm.none, collective_plan.algorithm);
    try std.testing.expectEqual(1, collective_plan.unsupported_collective_count);
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "pass=collective_algorithm_select") != null);
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "stablehlo.all_reduce") != null);
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "rejected_algorithms=direct,ring,tree,split") != null);

    var compile_options_reader: std.Io.Reader = .fixed("replicas=2; use_shardy=true");
    var compile_program_reader: std.Io.Reader = .fixed(fixture);
    try std.testing.expectError(
        compiler.CompilePipelineError.InvalidLowering,
        compiler.compileV0FromReader(
            std.testing.allocator,
            "stablehlo_text",
            "npu_v0",
            &compile_options_reader,
            &compile_program_reader,
            &diagnostics.writer,
        ),
    );
}

test "collective_group_channel_verify parses channels and rejects topology drift" {
    const channel_fixture: []u8 = try runfiles.readRunfile(std.testing.allocator, "pjrtx/tests/fixtures/all_reduce_channel.mlir");
    defer std.testing.allocator.free(channel_fixture);

    var options_reader: std.Io.Reader = .fixed("replicas=2");
    var program_reader: std.Io.Reader = .fixed(channel_fixture);
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    var input: compiler.CompileInput = try compiler.setupCompileInputFromReader(
        std.testing.allocator,
        "stablehlo_text",
        "npu_v0",
        &options_reader,
        &program_reader,
        &diagnostics.writer,
    );
    defer input.deinit();

    var module: compiler.MlirModuleArtifact = try compiler.ingestStablehloText(&input, &diagnostics.writer);
    defer module.deinit();

    var graph: compiler.GraphModule = try compiler.importGraphFromMlir(std.testing.allocator, module, &diagnostics.writer);
    defer graph.deinit();
    const target: compiler.SelectedTarget = try compiler.selectTarget(.npu_v0, &diagnostics.writer);
    try compiler.verifyCollectiveGroupChannel(graph, target, input.compile_options, &diagnostics.writer);

    switch (graph.instructions[0].payload) {
        .collective => |payload| {
            const expected_channel_id: ?u64 = 7;
            const expected_channel_type: ?u32 = 0;
            try std.testing.expectEqual(expected_channel_id, payload.channel_id);
            try std.testing.expectEqual(expected_channel_type, payload.channel_type);
        },
        else => return error.TestExpectedEqual,
    }

    var bad_options: compiler.CompileOptions = input.compile_options;
    bad_options.num_replicas = 1;
    try std.testing.expectError(
        compiler.CompilePipelineError.InvalidLowering,
        compiler.verifyCollectiveGroupChannel(graph, target, bad_options, &diagnostics.writer),
    );
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "pass=collective_group_channel_verify") != null);
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "participant outside compile topology") != null);
}

test "V0 compiler pass catalog exposes explicit XLA-like pass contracts" {
    const contracts = compiler.v0CompilerPassContracts();
    try std.testing.expect(contracts.len >= 24);
    try std.testing.expectEqualStrings("stablehlo_parse", contracts[0].stable_name);
    try std.testing.expectEqual(.stablehlo_ingest, contracts[0].stage);
    try std.testing.expectEqual(.input_bytes, contracts[0].input_artifact);
    try std.testing.expectEqual(.mlir_module, contracts[0].output_artifact);
    try std.testing.expectEqualStrings("broadcast_simplify", contracts[8].stable_name);
    try std.testing.expectEqual(.algebraic_normalization, contracts[8].stage);
    try std.testing.expectEqual(.may_change_ir, contracts[8].effect);
    try std.testing.expectEqualStrings("reshape_transpose_fold", contracts[9].stable_name);
    try std.testing.expectEqual(.algebraic_normalization, contracts[9].stage);
    try std.testing.expectEqual(.may_change_ir, contracts[9].effect);
    try std.testing.expectEqualStrings("fusion_candidate_discovery", contracts[10].stable_name);
    try std.testing.expectEqual(.plans_performance, contracts[10].effect);
    try std.testing.expectEqualStrings("matmul_epilogue_fusion_select", contracts[11].stable_name);
    try std.testing.expectEqual(.fusion_decision_plan, contracts[11].stage);
    try std.testing.expectEqual(.plans_performance, contracts[11].effect);
    try std.testing.expectEqualStrings("tile_shape_select", contracts[12].stable_name);
    try std.testing.expectEqual(.placement_planning, contracts[12].stage);
    try std.testing.expectEqual(.plans_performance, contracts[12].effect);
    try std.testing.expectEqualStrings("collective_group_channel_verify", contracts[13].stable_name);
    try std.testing.expectEqual(.verifies_only, contracts[13].effect);
    try std.testing.expectEqualStrings("collective_plan_v0", contracts[14].stable_name);
    try std.testing.expectEqual(.lowers_collectives, contracts[14].effect);
    try std.testing.expectEqualStrings("collective_algorithm_select", contracts[15].stable_name);
    try std.testing.expectEqual(.collective_lowering, contracts[15].stage);
    try std.testing.expectEqual(.lowers_collectives, contracts[15].effect);
    try std.testing.expectEqualStrings("memory_traffic_refine", contracts[18].stable_name);
    try std.testing.expectEqual(.lowering, contracts[18].stage);
    try std.testing.expectEqual(.plans_performance, contracts[18].effect);
    try std.testing.expectEqualStrings("schedule_overlap_plan", contracts[20].stable_name);
    try std.testing.expectEqual(.schedule_build, contracts[20].stage);
    try std.testing.expectEqual(.plans_performance, contracts[20].effect);
    try std.testing.expectEqualStrings("kernel_codegen_plan", contracts[21].stable_name);
    try std.testing.expectEqual(.kernel_codegen, contracts[21].stage);
    try std.testing.expectEqual(.plans_performance, contracts[21].effect);
    try std.testing.expectEqualStrings("tile_legality_verify", contracts[22].stable_name);
    try std.testing.expectEqual(.kernel_codegen, contracts[22].stage);
    try std.testing.expectEqual(.verifies_only, contracts[22].effect);
    try std.testing.expectEqualStrings("backend_binding_select", contracts[23].stable_name);
    try std.testing.expectEqual(.binds_backend, contracts[23].effect);

    var catalog: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer catalog.deinit();
    try compiler.writeCompilerPassCatalog(&catalog.writer);
    const text = catalog.writer.buffered();
    try std.testing.expect(std.mem.indexOf(u8, text, "pass.8 name=broadcast_simplify stage=algebraic_normalization") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "pass.9 name=reshape_transpose_fold stage=algebraic_normalization") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "pass.10 name=fusion_candidate_discovery stage=fusion_candidate_discovery") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "pass.11 name=matmul_epilogue_fusion_select stage=fusion_decision_plan") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "pass.12 name=tile_shape_select stage=placement_planning") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "pass.2 name=collective_graph_payload_import stage=collective_lowering") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "pass.13 name=collective_group_channel_verify stage=collective_lowering") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "pass.14 name=collective_plan_v0 stage=collective_lowering") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "pass.15 name=collective_algorithm_select stage=collective_lowering") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "pass.18 name=memory_traffic_refine stage=lowering") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "pass.20 name=schedule_overlap_plan stage=schedule_build") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "pass.21 name=kernel_codegen_plan stage=kernel_codegen") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "pass.22 name=tile_legality_verify stage=kernel_codegen") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "records=LoweringRecord,LoweringRegionFact,ExplainRecord") != null);
}

test "tile_shape_select bounds large NPU matrix tiles before backend binding" {
    const fixture: []u8 = try runfiles.readRunfile(std.testing.allocator, "pjrtx/tests/fixtures/large_tiling.mlir");
    defer std.testing.allocator.free(fixture);

    var options_reader: std.Io.Reader = .fixed("");
    var program_reader: std.Io.Reader = .fixed(fixture);
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    var input: compiler.CompileInput = try compiler.setupCompileInputFromReader(
        std.testing.allocator,
        "stablehlo_text",
        "npu_v0",
        &options_reader,
        &program_reader,
        &diagnostics.writer,
    );
    defer input.deinit();

    var module: compiler.MlirModuleArtifact = try compiler.ingestStablehloText(&input, &diagnostics.writer);
    defer module.deinit();

    var graph: compiler.GraphModule = try compiler.importGraphFromMlir(std.testing.allocator, module, &diagnostics.writer);
    defer graph.deinit();
    try compiler.verifyGraphModule(graph, &diagnostics.writer);

    const target: compiler.SelectedTarget = try compiler.selectTarget(.npu_v0, &diagnostics.writer);
    var placement_plan: compiler.PlacementPlan = try compiler.planPlacement(std.testing.allocator, graph, target, &diagnostics.writer);
    defer placement_plan.deinit();

    const expected_placements: usize = 1;
    const expected_tile = [_]i64{ 128, 128 };
    const expected_result_memory: u32 = 1;
    const expected_tile_memory: ?u32 = 2;
    try std.testing.expectEqual(expected_placements, placement_plan.records.len);
    try std.testing.expectEqualSlices(i64, &expected_tile, placement_plan.records[0].logical_tile_shape);
    try std.testing.expectEqual(expected_result_memory, placement_plan.records[0].result_memory_space_id);
    try std.testing.expectEqual(expected_tile_memory, placement_plan.records[0].tile_memory_space_id);
    try std.testing.expect(std.mem.indexOf(u8, placement_plan.records[0].reason, "bounded local SRAM tile") != null);
    try std.testing.expectEqualStrings("", diagnostics.writer.buffered());

    var summary: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer summary.deinit();
    try compiler.writePlacementPlan(placement_plan, &summary.writer);
    try std.testing.expect(std.mem.indexOf(u8, summary.writer.buffered(), "placement.0 instruction=0 outputs=2 layout=dense_row_major tile=128x128 result_memory=1 tile_memory=2") != null);
}

test "broadcast_simplify removes identity broadcast before fusion and lowering" {
    const fixture: []u8 = try runfiles.readRunfile(std.testing.allocator, "pjrtx/tests/fixtures/identity_broadcast.mlir");
    defer std.testing.allocator.free(fixture);

    var options_reader: std.Io.Reader = .fixed("");
    var program_reader: std.Io.Reader = .fixed(fixture);
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    var input: compiler.CompileInput = try compiler.setupCompileInputFromReader(
        std.testing.allocator,
        "stablehlo_text",
        "npu_v0",
        &options_reader,
        &program_reader,
        &diagnostics.writer,
    );
    defer input.deinit();

    var module: compiler.MlirModuleArtifact = try compiler.ingestStablehloText(&input, &diagnostics.writer);
    defer module.deinit();

    var graph: compiler.GraphModule = try compiler.importGraphFromMlir(std.testing.allocator, module, &diagnostics.writer);
    defer graph.deinit();
    try compiler.verifyGraphModule(graph, &diagnostics.writer);

    var normalized: compiler.GraphNormalizationResult = try compiler.normalizeGraphV0(std.testing.allocator, graph, &diagnostics.writer);
    defer normalized.deinit();

    try std.testing.expectEqual(2, graph.instructions.len);
    try std.testing.expectEqual(1, normalized.graph.instructions.len);
    try std.testing.expectEqual(.return_, normalized.graph.instructions[0].kind);
    try std.testing.expectEqual(0, normalized.graph.instructions[0].inputs[0].index);
    try std.testing.expectEqual(1, normalized.records.len);
    try std.testing.expectEqual(compiler_facts.GraphRewriteDecision.applied, normalized.records[0].decision);
    try std.testing.expectEqualStrings("broadcast_simplify", normalized.records[0].pass_name);
    try std.testing.expectEqual(1, normalized.records[0].replaced_value_id.?.index);
    try std.testing.expectEqual(0, normalized.records[0].replacement_value_id.?.index);
    try std.testing.expectEqualStrings("", diagnostics.writer.buffered());
}

test "reshape_transpose_fold removes identity reshape and transpose before target legality" {
    const fixture: []u8 = try runfiles.readRunfile(std.testing.allocator, "pjrtx/tests/fixtures/identity_reshape_transpose.mlir");
    defer std.testing.allocator.free(fixture);

    var options_reader: std.Io.Reader = .fixed("");
    var program_reader: std.Io.Reader = .fixed(fixture);
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    var input: compiler.CompileInput = try compiler.setupCompileInputFromReader(
        std.testing.allocator,
        "stablehlo_text",
        "npu_v0",
        &options_reader,
        &program_reader,
        &diagnostics.writer,
    );
    defer input.deinit();

    var module: compiler.MlirModuleArtifact = try compiler.ingestStablehloText(&input, &diagnostics.writer);
    defer module.deinit();

    var graph: compiler.GraphModule = try compiler.importGraphFromMlir(std.testing.allocator, module, &diagnostics.writer);
    defer graph.deinit();

    var normalized: compiler.GraphNormalizationResult = try compiler.normalizeGraphV0(std.testing.allocator, graph, &diagnostics.writer);
    defer normalized.deinit();

    try std.testing.expectEqual(3, graph.instructions.len);
    try std.testing.expectEqual(1, normalized.graph.instructions.len);
    try std.testing.expectEqual(.return_, normalized.graph.instructions[0].kind);
    try std.testing.expectEqual(0, normalized.graph.instructions[0].inputs[0].index);
    try std.testing.expectEqual(2, normalized.records.len);
    try std.testing.expectEqual(compiler_facts.GraphRewriteDecision.applied, normalized.records[0].decision);
    try std.testing.expectEqualStrings("reshape_transpose_fold", normalized.records[0].pass_name);
    try std.testing.expectEqual(compiler_facts.GraphRewriteDecision.applied, normalized.records[1].decision);
    try std.testing.expectEqualStrings("reshape_transpose_fold", normalized.records[1].pass_name);
    try std.testing.expectEqualStrings("", diagnostics.writer.buffered());
}

test "non-identity transpose is rejected by no-fallback target legality" {
    const fixture: []u8 = try runfiles.readRunfile(std.testing.allocator, "pjrtx/tests/fixtures/non_identity_transpose.mlir");
    defer std.testing.allocator.free(fixture);

    var options_reader: std.Io.Reader = .fixed("");
    var program_reader: std.Io.Reader = .fixed(fixture);
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    try std.testing.expectError(
        compiler.CompilePipelineError.UnsupportedTargetFeature,
        compiler.compileV0FromReader(
            std.testing.allocator,
            "stablehlo_text",
            "npu_v0",
            &options_reader,
            &program_reader,
            &diagnostics.writer,
        ),
    );
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "pass=target_legality") != null);
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "op=transpose") != null);
}
