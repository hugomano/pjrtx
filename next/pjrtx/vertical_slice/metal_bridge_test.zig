const std = @import("std");
const backend = @import("pjrtx/backend");
const backend_mlir = @import("pjrtx/backend/mlir_bridge");
const compiler = @import("pjrtx/compiler");
const mlir_state = @import("pjrtx/compiler/mlir_state");
const runtime = @import("pjrtx/runtime");
const runfiles = @import("runfiles.zig");

test "vertical slice expands Metal backend binding into executable calls" {
    const fixture: []u8 = try runfiles.readRunfile(std.testing.allocator, "next/pjrtx/fixtures/tanh_dot_bias.mlir");
    defer std.testing.allocator.free(fixture);

    var options_reader: std.Io.Reader = .fixed("");
    var program_reader: std.Io.Reader = .fixed(fixture);
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    var input: compiler.CompileInput = try compiler.setupCompileInputFromReader(
        std.testing.allocator,
        "stablehlo_text",
        "metal_v0",
        &options_reader,
        &program_reader,
        &diagnostics.writer,
    );
    defer input.deinit();

    var module: compiler.MlirModuleArtifact = try compiler.ingestStablehloText(&input, &diagnostics.writer);
    defer module.deinit();

    var graph: compiler.GraphModule = try compiler.importGraphFromMlir(std.testing.allocator, module, &diagnostics.writer);
    defer graph.deinit();

    const target: compiler.SelectedTarget = try compiler.selectTarget(.metal_v0, &diagnostics.writer);
    var session: mlir_state.MlirSession = try .initFromStablehloText(
        std.testing.allocator,
        fixture,
        .{ .program_name = "metal_bridge_test" },
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

    const binding = planned.report.backend_bindings[0];
    var executable: backend.BackendExecutablePlan = try backend.planExecutable(std.testing.allocator, planned.report, binding, &diagnostics.writer);
    defer executable.deinit();
    try backend_mlir.commitExecutablePlan(std.testing.allocator, &session, executable, &diagnostics.writer);
    try std.testing.expectEqual(mlir_state.ModuleState.backend_executable_planned, mlir_state.moduleState(&session).?);
    const extracted_executable: mlir_state.BackendExecutablePlanFact = try mlir_state.extractBackendExecutablePlan(std.testing.allocator, &session, &diagnostics.writer);
    defer mlir_state.deinitExtractedBackendExecutablePlan(std.testing.allocator, extracted_executable);

    var kernel_graph: backend.BackendKernelGraphPlan = try backend.planKernelGraph(std.testing.allocator, planned.report, executable, &diagnostics.writer);
    defer kernel_graph.deinit();
    try backend_mlir.commitKernelGraph(std.testing.allocator, &session, kernel_graph, &diagnostics.writer);
    try std.testing.expectEqual(mlir_state.ModuleState.backend_kernel_graph_planned, mlir_state.moduleState(&session).?);
    const extracted_kernel_graph: mlir_state.BackendKernelGraphFact = try mlir_state.extractBackendKernelGraph(std.testing.allocator, &session, &diagnostics.writer);
    defer mlir_state.deinitExtractedBackendKernelGraph(std.testing.allocator, extracted_kernel_graph);

    var profiled: runtime.ProfiledTraceReport = try runtime.executeSyntheticProfile(std.testing.allocator, planned.report, &diagnostics.writer);
    defer profiled.deinit();

    var summary: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer summary.deinit();
    try mlir_state.writeBackendExecutablePlanFactSummary(extracted_executable, &summary.writer);
    try mlir_state.writeBackendKernelGraphFactSummary(extracted_kernel_graph, &summary.writer);
    try backend.writeBackendCallProfileSummary(profiled.report, executable, &summary.writer);

    const expected_calls: usize = 2;
    const expected_edges: usize = 1;
    try std.testing.expectEqual(expected_calls, executable.calls.len);
    try std.testing.expectEqual(expected_calls, extracted_executable.calls.len);
    try std.testing.expectEqual(.metal_v0, extracted_executable.backend_kind);
    try std.testing.expectEqual(1, extracted_executable.command_id.index);
    try std.testing.expectEqualStrings("metal_mls_graph_execute", extracted_executable.backend_operation);
    try std.testing.expectEqualStrings("metal_mls_elementwise_fusion_kernel", extracted_executable.calls[1].backend_operation);
    try std.testing.expectEqualStrings("elementwise_fusion", extracted_executable.calls[1].feature);
    try std.testing.expectEqual(3, extracted_executable.calls[1].graph_instruction_ids.len);
    try std.testing.expectEqual(expected_calls, kernel_graph.nodes.len);
    try std.testing.expectEqual(expected_calls, extracted_kernel_graph.nodes.len);
    try std.testing.expectEqual(expected_edges, kernel_graph.edges.len);
    try std.testing.expectEqual(expected_edges, extracted_kernel_graph.edges.len);
    try std.testing.expectEqual(.metal_v0, extracted_kernel_graph.backend_kind);
    try std.testing.expectEqualStrings("metal_mls_elementwise_fusion_kernel", extracted_kernel_graph.nodes[1].backend_operation);
    try std.testing.expectEqual(.f32, extracted_kernel_graph.nodes[1].output_type.element_type);
    try std.testing.expectEqual(2, extracted_kernel_graph.nodes[1].output_type.dims[0]);
    try std.testing.expectEqual(3, extracted_kernel_graph.nodes[1].output_type.dims[1]);
    try std.testing.expectEqual(3, extracted_kernel_graph.edges[0].value_id.index);
    try std.testing.expectEqual(.metal_v0, executable.backend_kind);
    try std.testing.expectEqualStrings("metal_mls_graph_execute", executable.backend_operation);
    try std.testing.expectEqualStrings("metal_mls_matmul_kernel", executable.calls[0].backend_operation);
    try std.testing.expectEqualStrings("metal_mls_elementwise_fusion_kernel", executable.calls[1].backend_operation);
    try std.testing.expectEqual(.elementwise_fusion, executable.calls[1].feature);
    try std.testing.expectEqual(3, executable.calls[1].graph_instruction_ids.len);
    try std.testing.expectEqual(2, executable.calls[1].input_value_ids.len);
    try std.testing.expectEqual(1, executable.calls[1].output_value_ids.len);
    try std.testing.expectEqual(2, executable.calls[1].input_value_ids[0].index);
    try std.testing.expectEqual(3, executable.calls[1].input_value_ids[1].index);
    try std.testing.expectEqual(6, executable.calls[1].output_value_ids[0].index);
    try std.testing.expect(std.mem.indexOf(u8, summary.writer.buffered(), "backend=metal_v0 command=1 operation=metal_mls_graph_execute calls=2") != null);
    try std.testing.expect(std.mem.indexOf(u8, summary.writer.buffered(), "call.1 instruction=1 instructions=1,2,3 feature=elementwise_fusion operation=metal_mls_elementwise_fusion_kernel unit=0 inputs=2 outputs=1") != null);
    try std.testing.expect(std.mem.indexOf(u8, summary.writer.buffered(), "backend=metal_v0 command=1 nodes=2 edges=1") != null);
    try std.testing.expect(std.mem.indexOf(u8, summary.writer.buffered(), "node.0 call=0 instruction=0 instructions=0 feature=rank2_dot_general operation=metal_mls_matmul_kernel dtype=f32 rank=2 inputs=2 outputs=1 attrs=rank2_dot_general") != null);
    try std.testing.expect(std.mem.indexOf(u8, summary.writer.buffered(), "node.1 call=1 instruction=1 instructions=1,2,3 feature=elementwise_fusion operation=metal_mls_elementwise_fusion_kernel dtype=f32 rank=2 inputs=2 outputs=1 attrs=elementwise_fusion") != null);
    try std.testing.expect(std.mem.indexOf(u8, summary.writer.buffered(), "edge.0 value=3 node.0->node.1") != null);
    try std.testing.expect(std.mem.indexOf(u8, summary.writer.buffered(), "backend call profiles") != null);
    try std.testing.expect(std.mem.indexOf(u8, summary.writer.buffered(), "call.0 command=1 operation=metal_mls_matmul_kernel instructions=0 unit=0 predicted_bytes=104 observed_bytes=104 predicted_ops=48 observed_ops=48 ideal_compute_ps=0 ideal_memory_ps=0 limiting=unknown memory=memory.1:104B/0ps event=profile.2") != null);

    var mlir_summary: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer mlir_summary.deinit();
    try mlir_state.writeStateSummary(&session, &mlir_summary.writer);
    try std.testing.expect(std.mem.indexOf(u8, mlir_summary.writer.buffered(), "state=backend_kernel_graph_planned") != null);
    try std.testing.expect(std.mem.indexOf(u8, mlir_summary.writer.buffered(), "backend_executable backend=metal_v0 command=1 operation=metal_mls_graph_execute calls=2") != null);
    try std.testing.expect(std.mem.indexOf(u8, mlir_summary.writer.buffered(), "executable_call.1 instruction=1 feature=elementwise_fusion operation=metal_mls_elementwise_fusion_kernel instructions=1,2,3") != null);
    try std.testing.expect(std.mem.indexOf(u8, mlir_summary.writer.buffered(), "backend_kernel_graph backend=metal_v0 command=1 nodes=2 edges=1") != null);
    try std.testing.expect(std.mem.indexOf(u8, mlir_summary.writer.buffered(), "kernel_node.1 call=1 instruction=1 feature=elementwise_fusion operation=metal_mls_elementwise_fusion_kernel dtype=f32 dims=2,3 attrs=elementwise_fusion") != null);
    try std.testing.expect(std.mem.indexOf(u8, mlir_summary.writer.buffered(), "kernel_edge.0 value=3 src=0 dst=1") != null);
    try std.testing.expect(std.mem.indexOf(u8, summary.writer.buffered(), "call.1 command=1 operation=metal_mls_elementwise_fusion_kernel instructions=1,2,3 unit=0 predicted_bytes=156 observed_bytes=156 predicted_ops=18 observed_ops=18 ideal_compute_ps=0 ideal_memory_ps=0 limiting=unknown memory=memory.1:60B/0ps event=profile.3") != null);
    try std.testing.expectEqualStrings("", diagnostics.writer.buffered());
}
