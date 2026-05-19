const std = @import("std");
const compiler = @import("pjrtx/compiler");
const mlir_state = @import("pjrtx/compiler/mlir_state");
const runfiles = @import("runfiles.zig");

test "MLIR state boundary snapshot records target and fusion facts" {
    const fixture: []u8 = try runfiles.readRunfile(std.testing.allocator, "next/pjrtx/fixtures/tanh_dot_bias.mlir");
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
        .{ .program_name = "mlir_state_boundary_test" },
        &diagnostics.writer,
    );
    defer session.deinit();

    try mlir_state.attachTarget(&session, target.description, input.compile_options.num_replicas, input.compile_options.num_partitions, &diagnostics.writer);
    try mlir_state.markTargetLegal(&session, &diagnostics.writer);
    try mlir_state.runFusionCandidateDiscoveryExternalPass(&session, &diagnostics.writer);
    try mlir_state.planFusionFromCandidates(&session, &diagnostics.writer);

    var summary: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer summary.deinit();
    try mlir_state.writeStateSummary(&session, &summary.writer);

    const expected: []u8 = try runfiles.readRunfile(std.testing.allocator, "next/pjrtx/vertical_slice/testdata/mlir_state_after_fusion.txt");
    defer std.testing.allocator.free(expected);
    try std.testing.expectEqualStrings(expected, summary.writer.buffered());
    try std.testing.expectEqualStrings("", diagnostics.writer.buffered());
}
