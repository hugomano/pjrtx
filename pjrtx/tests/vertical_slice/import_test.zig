const std = @import("std");
const compiler = @import("pjrtx/compiler");
const compiler_facts = @import("pjrtx/compiler/facts");
const mlir_state = @import("pjrtx/compiler/mlir_state");
const core = @import("pjrtx/core");
const runfiles = @import("runfiles.zig");

test "vertical slice StableHLO fixture ingests through MLIR C API" {
    const fixture: []u8 = try runfiles.readRunfile(std.testing.allocator, "pjrtx/tests/fixtures/tanh_dot_bias.mlir");
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
    var session: mlir_state.MlirSession = try .initFromStablehloText(
        std.testing.allocator,
        fixture,
        .{ .program_name = "import_test" },
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

    const expected_values: usize = 7;
    const expected_instructions: usize = 5;
    const expected_commands: usize = 3;
    const expected_overlap_records: usize = 2;
    const expected_codegen_records: usize = 2;
    const h2d_command_id: u32 = 0;
    const backend_command_id: u32 = 1;
    const d2h_command_id: u32 = 2;
    const matmul_external_inputs: u32 = 2;
    const fused_elementwise_ops: u32 = 3;
    const fused_elementwise_external_inputs: u32 = 2;
    const fused_elementwise_external_outputs: u32 = 1;
    const fused_elementwise_intermediates: u32 = 2;
    const matmul_input_ids = [_]compiler_facts.GraphValueId{ .{ .index = 0 }, .{ .index = 1 } };
    const matmul_output_ids = [_]compiler_facts.GraphValueId{.{ .index = 3 }};
    const fused_input_ids = [_]compiler_facts.GraphValueId{ .{ .index = 2 }, .{ .index = 3 } };
    const fused_output_ids = [_]compiler_facts.GraphValueId{.{ .index = 6 }};
    const fused_intermediate_ids = [_]compiler_facts.GraphValueId{ .{ .index = 4 }, .{ .index = 5 } };
    const expected_tile = [_]i64{ 2, 3 };
    const expected_result_memory: u32 = 1;
    const expected_tile_memory: ?u32 = 2;
    const expected_matmul_global_read: u128 = 80;
    const expected_matmul_global_write: u128 = 24;
    const expected_matmul_local_read: u128 = 80;
    const expected_matmul_local_write: u128 = 24;
    const expected_elementwise_global_read: u128 = 36;
    const expected_elementwise_global_write: u128 = 24;
    const expected_elementwise_local_read: u128 = 84;
    const expected_elementwise_local_write: u128 = 72;
    try std.testing.expectEqual(expected_values, graph.values.len);
    try std.testing.expectEqual(expected_instructions, graph.instructions.len);
    try std.testing.expectEqual(expected_commands, planned.report.schedule_commands.len);
    try std.testing.expectEqual(expected_overlap_records, planned.report.schedule_overlap_records.len);
    try std.testing.expectEqual(expected_codegen_records, planned.report.kernel_codegen_records.len);
    try std.testing.expectEqual(core.ScheduleOverlapDecision.serialized, planned.report.schedule_overlap_records[0].decision);
    try std.testing.expectEqual(core.ScheduleOverlapKind.transfer_compute, planned.report.schedule_overlap_records[0].kind);
    try std.testing.expectEqual(h2d_command_id, planned.report.schedule_overlap_records[0].first_command_id.index);
    try std.testing.expectEqual(backend_command_id, planned.report.schedule_overlap_records[0].second_command_id.index);
    try std.testing.expectEqual(core.DependencyKind.data, planned.report.schedule_overlap_records[0].dependency_kind);
    try std.testing.expectEqual(core.ScheduleOverlapDecision.serialized, planned.report.schedule_overlap_records[1].decision);
    try std.testing.expectEqual(core.ScheduleOverlapKind.compute_transfer, planned.report.schedule_overlap_records[1].kind);
    try std.testing.expectEqual(backend_command_id, planned.report.schedule_overlap_records[1].first_command_id.index);
    try std.testing.expectEqual(d2h_command_id, planned.report.schedule_overlap_records[1].second_command_id.index);
    try std.testing.expectEqual(core.DependencyKind.data, planned.report.schedule_overlap_records[1].dependency_kind);
    try std.testing.expectEqual(core.KernelCodegenKind.backend_kernel_graph, planned.report.kernel_codegen_records[0].kind);
    try std.testing.expectEqualStrings("npu_matmul", planned.report.kernel_codegen_records[0].operation);
    try std.testing.expectEqual(matmul_external_inputs, planned.report.kernel_codegen_records[0].shape.external_input_count);
    try std.testing.expectEqualSlices(i64, &expected_tile, planned.report.kernel_codegen_records[0].logical_tile_shape);
    try std.testing.expectEqual(expected_result_memory, planned.report.kernel_codegen_records[0].result_memory_space_id);
    try std.testing.expectEqual(expected_tile_memory, planned.report.kernel_codegen_records[0].tile_memory_space_id);
    try std.testing.expectEqual(expected_matmul_global_read, planned.report.kernel_codegen_records[0].memory_pressure.global_bytes_read);
    try std.testing.expectEqual(expected_matmul_global_write, planned.report.kernel_codegen_records[0].memory_pressure.global_bytes_written);
    try std.testing.expectEqual(expected_matmul_local_read, planned.report.kernel_codegen_records[0].memory_pressure.local_bytes_read);
    try std.testing.expectEqual(expected_matmul_local_write, planned.report.kernel_codegen_records[0].memory_pressure.local_bytes_written);
    try std.testing.expectEqualSlices(compiler_facts.GraphValueId, &matmul_input_ids, planned.report.kernel_codegen_records[0].external_input_ids);
    try std.testing.expectEqualSlices(compiler_facts.GraphValueId, &matmul_output_ids, planned.report.kernel_codegen_records[0].external_output_ids);
    try std.testing.expectEqual(core.KernelCodegenKind.elementwise_fusion_kernel, planned.report.kernel_codegen_records[1].kind);
    try std.testing.expectEqualStrings("npu_elementwise_fusion", planned.report.kernel_codegen_records[1].operation);
    try std.testing.expectEqual(fused_elementwise_ops, planned.report.kernel_codegen_records[1].shape.operation_count);
    try std.testing.expectEqual(fused_elementwise_external_inputs, planned.report.kernel_codegen_records[1].shape.external_input_count);
    try std.testing.expectEqual(fused_elementwise_external_outputs, planned.report.kernel_codegen_records[1].shape.external_output_count);
    try std.testing.expectEqual(fused_elementwise_intermediates, planned.report.kernel_codegen_records[1].shape.intermediate_value_count);
    try std.testing.expectEqualSlices(i64, &expected_tile, planned.report.kernel_codegen_records[1].logical_tile_shape);
    try std.testing.expectEqual(expected_result_memory, planned.report.kernel_codegen_records[1].result_memory_space_id);
    try std.testing.expectEqual(expected_tile_memory, planned.report.kernel_codegen_records[1].tile_memory_space_id);
    try std.testing.expectEqual(expected_elementwise_global_read, planned.report.kernel_codegen_records[1].memory_pressure.global_bytes_read);
    try std.testing.expectEqual(expected_elementwise_global_write, planned.report.kernel_codegen_records[1].memory_pressure.global_bytes_written);
    try std.testing.expectEqual(expected_elementwise_local_read, planned.report.kernel_codegen_records[1].memory_pressure.local_bytes_read);
    try std.testing.expectEqual(expected_elementwise_local_write, planned.report.kernel_codegen_records[1].memory_pressure.local_bytes_written);
    try std.testing.expectEqualSlices(compiler_facts.GraphValueId, &fused_input_ids, planned.report.kernel_codegen_records[1].external_input_ids);
    try std.testing.expectEqualSlices(compiler_facts.GraphValueId, &fused_output_ids, planned.report.kernel_codegen_records[1].external_output_ids);
    try std.testing.expectEqualSlices(compiler_facts.GraphValueId, &fused_intermediate_ids, planned.report.kernel_codegen_records[1].intermediate_value_ids);
    try std.testing.expectEqualStrings("", diagnostics.writer.buffered());
}
