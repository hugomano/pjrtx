const std = @import("std");
const compiler_facts = @import("pjrtx/compiler/facts");
const core = @import("pjrtx/core");
const runfiles = @import("runfiles.zig");

test "vertical slice report can write stable IDs" {
    var output = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer output.deinit();

    try core.writeId(&output.writer, "source", 0);
    try std.testing.expectEqualStrings("source.0", output.writer.buffered());
}

test "vertical slice report matches golden summary" {
    const report: core.TraceReport = tinyTraceReport();
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    try core.validateTraceReport(report, &diagnostics.writer);
    try std.testing.expectEqualStrings("", diagnostics.writer.buffered());

    var output: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();

    try core.writeTraceReportSummary(report, &output.writer);
    const expected: []const u8 = try runfiles.readRunfile(std.testing.allocator, "next/pjrtx/vertical_slice/testdata/tiny_trace_report.txt");
    defer std.testing.allocator.free(expected);
    try std.testing.expectEqualStrings(expected, output.writer.buffered());
}

/// The golden report fixture proves the report writer's public contract without
/// depending on compiler import yet. Compiler stages can later replace this
/// hand-built trace while preserving the same observable report shape.
fn tinyTraceReport() core.TraceReport {
    return .{
        .sources = &tiny_sources,
        .graph_values = &tiny_values,
        .graph_instructions = &tiny_instructions,
        .cost_ledger = &tiny_costs,
        .lowering_records = &tiny_lowerings,
        .memory_traffic_records = &tiny_memory_traffic,
        .schedule_commands = &tiny_commands,
        .backend_bindings = &tiny_bindings,
        .profile_events = &tiny_profiles,
        .explain_records = &tiny_explains,
    };
}

const tiny_source: compiler_facts.SourceRef = .{
    .id = .{ .index = 0 },
    .frontend = .stablehlo,
    .op_name = "stablehlo.tanh",
    .source_index = 0,
    .location = "",
};
const tiny_sources = [_]compiler_facts.SourceRef{tiny_source};
const tiny_dims = [_]i64{4};
const tiny_values = [_]compiler_facts.GraphValue{
    .{ .id = .{ .index = 0 }, .ty = .{ .element_type = .f32, .dims = &tiny_dims, .layout = .dense_row_major }, .role = .parameter, .source = tiny_source },
    .{ .id = .{ .index = 1 }, .ty = .{ .element_type = .f32, .dims = &tiny_dims, .layout = .dense_row_major }, .role = .instruction_result, .source = tiny_source },
};
const tiny_value_input_refs = [_]compiler_facts.GraphValueId{.{ .index = 0 }};
const tiny_value_output_refs = [_]compiler_facts.GraphValueId{.{ .index = 1 }};
const tiny_instruction_refs = [_]compiler_facts.GraphInstructionId{.{ .index = 0 }};
const tiny_cost_refs = [_]compiler_facts.CostLedgerId{.{ .index = 0 }};
const tiny_lowering_refs = [_]compiler_facts.LoweringRecordId{.{ .index = 0 }};
const tiny_profile_refs = [_]core.ProfileEventId{.{ .index = 0 }};
const tiny_instructions = [_]compiler_facts.GraphInstruction{
    .{
        .id = .{ .index = 0 },
        .kind = .elementwise_unary,
        .inputs = &tiny_value_input_refs,
        .outputs = &tiny_value_output_refs,
        .payload = .{ .elementwise_unary = .{ .op = .tanh } },
        .source = tiny_source,
    },
};
const tiny_costs = [_]compiler_facts.CostLedgerEntry{
    .{
        .id = .{ .index = 0 },
        .source = tiny_source,
        .graph_instruction_ids = &tiny_instruction_refs,
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
const tiny_lowerings = [_]compiler_facts.LoweringRecord{
    .{
        .id = .{ .index = 0 },
        .graph_instruction_ids = &tiny_instruction_refs,
        .decision = .backend_kernel_graph,
        .reason = "metal_v0",
        .rejected_alternatives = &.{},
        .cost_ledger_ids = &tiny_cost_refs,
    },
};
const tiny_memory_traffic = [_]compiler_facts.MemoryTrafficRecord{
    .{
        .id = .{ .index = 0 },
        .lowering_record_id = .{ .index = 0 },
        .memory_space_id = 1,
        .kind = .global_memory,
        .graph_instruction_ids = &tiny_instruction_refs,
        .cost_ledger_ids = &tiny_cost_refs,
        .bytes_read = 16,
        .bytes_written = 16,
        .reason = "tiny stable report fixture",
    },
};
const tiny_commands = [_]core.ScheduleCommand{
    .{
        .id = .{ .index = 0 },
        .kind = .backend_execute,
        .stream = .{ .index = 0 },
        .inputs = &tiny_value_input_refs,
        .outputs = &tiny_value_output_refs,
        .dependencies = &.{},
        .lowering_record_ids = &tiny_lowering_refs,
        .cost_ledger_ids = &tiny_cost_refs,
    },
};
const tiny_bindings = [_]core.BackendBinding{
    .{
        .id = .{ .index = 0 },
        .command_id = .{ .index = 0 },
        .backend_kind = .metal_v0,
        .backend_operation = "metal_mls_graph_execute",
        .graph_instruction_ids = &tiny_instruction_refs,
        .expected_unit_id = null,
        .cost_ledger_ids = &tiny_cost_refs,
    },
};
const tiny_profiles = [_]core.ProfileEvent{
    .{
        .id = .{ .index = 0 },
        .command_id = .{ .index = 0 },
        .graph_instruction_ids = &tiny_instruction_refs,
        .kind = .backend_execute,
        .start_ns = 10,
        .duration_ns = 20,
        .bytes = 32,
        .logical_ops = 4,
        .status = .ok,
        .forced_synchronization = false,
    },
};
const tiny_explain_sources = [_]compiler_facts.SourceRef{tiny_source};
const tiny_explains = [_]core.ExplainRecord{
    .{
        .id = .{ .index = 0 },
        .pass_name = "lowering",
        .subject = .{ .backend_binding = .{ .index = 0 } },
        .decision = "kernel_graph",
        .reason = "bootstrap backend",
        .source_refs = &tiny_explain_sources,
        .cost_ledger_ids = &tiny_cost_refs,
        .profile_event_ids = &tiny_profile_refs,
    },
};
