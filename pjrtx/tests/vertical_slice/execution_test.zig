const std = @import("std");
const backend = @import("pjrtx/backend");
const backend_mlir = @import("pjrtx/backend/mlir_bridge");
const compiler = @import("pjrtx/compiler");
const compiler_facts = @import("pjrtx/compiler/facts");
const core = @import("pjrtx/core");
const mlir_state = @import("pjrtx/compiler/mlir_state");
const runtime = @import("pjrtx/runtime");
const runtime_mlir = @import("pjrtx/runtime/mlir_bridge");
const runfiles = @import("runfiles.zig");
const target_pkg = @import("pjrtx/target");

test "execution smoke starts from schedule command names" {
    var output = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer output.deinit();

    try runtime.writeCommandKind(&output.writer, .host_to_device);
    try std.testing.expectEqualStrings("host_to_device", output.writer.buffered());
}

test "vertical slice planned schedule produces synthetic profile events" {
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

    const target: compiler.SelectedTarget = try compiler.selectTarget(.npu_v0, &diagnostics.writer);
    var session: mlir_state.MlirSession = try .initFromStablehloText(
        std.testing.allocator,
        fixture,
        .{ .program_name = "execution_test" },
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

    var allocation_plan: runtime.AllocationPlan = try runtime.planAllocations(std.testing.allocator, planned.report, &diagnostics.writer);
    defer allocation_plan.deinit();
    try runtime_mlir.commitAllocationPlan(std.testing.allocator, &session, allocation_plan, &diagnostics.writer);
    try std.testing.expectEqual(mlir_state.ModuleState.runtime_allocation_planned, mlir_state.moduleState(&session).?);

    var stream_plan: runtime.StreamPlan = try runtime.planStreams(std.testing.allocator, planned.report, &diagnostics.writer);
    defer stream_plan.deinit();
    try runtime_mlir.commitStreamPlan(std.testing.allocator, &session, stream_plan, &diagnostics.writer);
    try std.testing.expectEqual(mlir_state.ModuleState.runtime_stream_planned, mlir_state.moduleState(&session).?);

    var profiled: runtime.ProfiledTraceReport = try runtime.executeSyntheticProfile(std.testing.allocator, planned.report, &diagnostics.writer);
    defer profiled.deinit();
    try runtime_mlir.commitProfile(std.testing.allocator, &session, profiled, &diagnostics.writer);
    try std.testing.expectEqual(mlir_state.ModuleState.runtime_profiled, mlir_state.moduleState(&session).?);
    try runtime_mlir.commitProfileJoins(std.testing.allocator, &session, profiled, &diagnostics.writer);
    try std.testing.expectEqual(mlir_state.ModuleState.runtime_profile_joined, mlir_state.moduleState(&session).?);
    try backend_mlir.commitProfileJoins(std.testing.allocator, &session, executable, profiled.report, &diagnostics.writer);
    try std.testing.expectEqual(mlir_state.ModuleState.backend_profile_joined, mlir_state.moduleState(&session).?);
    const extracted_target: target_pkg.TargetDescription = try mlir_state.extractTargetDescription(std.testing.allocator, &session, &diagnostics.writer);
    defer mlir_state.deinitExtractedTargetDescription(std.testing.allocator, extracted_target);
    const extracted_cost_ledger: []compiler_facts.CostLedgerEntry = try mlir_state.extractCostLedgerEntries(std.testing.allocator, &session, &diagnostics.writer);
    defer mlir_state.deinitExtractedCostLedgerEntries(std.testing.allocator, extracted_cost_ledger);
    const extracted_lowerings: []compiler_facts.LoweringRecord = try mlir_state.extractLoweringRecords(std.testing.allocator, &session, extracted_cost_ledger.len, &diagnostics.writer);
    defer mlir_state.deinitExtractedLoweringRecords(std.testing.allocator, extracted_lowerings);
    const extracted_lowering_regions: []mlir_state.LoweringRegionFact = try mlir_state.extractLoweringRegionFacts(std.testing.allocator, &session, extracted_lowerings.len, &diagnostics.writer);
    defer mlir_state.deinitExtractedLoweringRegionFacts(std.testing.allocator, extracted_lowering_regions);
    const extracted_memory_traffic: []compiler_facts.MemoryTrafficRecord = try mlir_state.extractMemoryTrafficRecords(std.testing.allocator, &session, extracted_cost_ledger.len, extracted_lowerings.len, &diagnostics.writer);
    defer mlir_state.deinitExtractedMemoryTrafficRecords(std.testing.allocator, extracted_memory_traffic);
    const extracted_schedule_commands: []core.ScheduleCommand = try mlir_state.extractScheduleCommands(std.testing.allocator, &session, &diagnostics.writer);
    defer mlir_state.deinitExtractedScheduleCommands(std.testing.allocator, extracted_schedule_commands);
    const extracted_backend_executable: mlir_state.BackendExecutablePlanFact = try mlir_state.extractBackendExecutablePlan(std.testing.allocator, &session, &diagnostics.writer);
    defer mlir_state.deinitExtractedBackendExecutablePlan(std.testing.allocator, extracted_backend_executable);
    const extracted_runtime_allocation: mlir_state.RuntimeAllocationPlanFact = try mlir_state.extractRuntimeAllocationPlan(std.testing.allocator, &session, &diagnostics.writer);
    defer mlir_state.deinitExtractedRuntimeAllocationPlan(std.testing.allocator, extracted_runtime_allocation);
    const extracted_runtime_streams: mlir_state.RuntimeStreamPlanFact = try mlir_state.extractRuntimeStreamPlan(std.testing.allocator, &session, &diagnostics.writer);
    defer mlir_state.deinitExtractedRuntimeStreamPlan(std.testing.allocator, extracted_runtime_streams);
    const extracted_runtime_profile_events: []mlir_state.RuntimeProfileEventFact = try mlir_state.extractRuntimeProfileEvents(std.testing.allocator, &session, &diagnostics.writer);
    defer mlir_state.deinitExtractedRuntimeProfileEvents(std.testing.allocator, extracted_runtime_profile_events);
    const extracted_runtime_profile_joins: []mlir_state.RuntimeProfileJoinFact = try mlir_state.extractRuntimeProfileJoins(std.testing.allocator, &session, &diagnostics.writer);
    defer mlir_state.deinitExtractedRuntimeProfileJoins(std.testing.allocator, extracted_runtime_profile_joins);
    const extracted_backend_profile_joins: []mlir_state.BackendProfileJoinFact = try mlir_state.extractBackendProfileJoins(std.testing.allocator, &session, &diagnostics.writer);
    defer mlir_state.deinitExtractedBackendProfileJoins(std.testing.allocator, extracted_backend_profile_joins);

    var allocation_summary: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer allocation_summary.deinit();
    try mlir_state.writeRuntimeAllocationPlanFactSummary(extracted_runtime_allocation, &allocation_summary.writer);

    var stream_summary: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer stream_summary.deinit();
    try mlir_state.writeRuntimeStreamPlanFactSummary(extracted_runtime_streams, &stream_summary.writer);

    var runtime_summary: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer runtime_summary.deinit();
    try mlir_state.writeRuntimeExecutionFactSummary(profiled.report, extracted_target, extracted_cost_ledger, extracted_schedule_commands, extracted_runtime_allocation, extracted_runtime_streams, extracted_runtime_profile_events, &runtime_summary.writer);
    try mlir_state.writeLoweringProfileFactSummary(profiled.report, extracted_cost_ledger, extracted_schedule_commands, extracted_runtime_profile_events, &runtime_summary.writer);

    const expected_report: []u8 = try runfiles.readRunfile(std.testing.allocator, "pjrtx/tests/vertical_slice/testdata/v0_execution_report.txt");
    defer std.testing.allocator.free(expected_report);
    var execution_report: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer execution_report.deinit();
    try mlir_state.writeRuntimeAllocationPlanFactSummary(extracted_runtime_allocation, &execution_report.writer);
    try mlir_state.writeRuntimeStreamPlanFactSummary(extracted_runtime_streams, &execution_report.writer);
    try mlir_state.writeRuntimeExecutionFactSummary(profiled.report, extracted_target, extracted_cost_ledger, extracted_schedule_commands, extracted_runtime_allocation, extracted_runtime_streams, extracted_runtime_profile_events, &execution_report.writer);
    try mlir_state.writeLoweringProfileFactSummary(profiled.report, extracted_cost_ledger, extracted_schedule_commands, extracted_runtime_profile_events, &execution_report.writer);
    try mlir_state.writeRuntimeHardwareUtilizationFactSummary(profiled.report, extracted_target, extracted_cost_ledger, extracted_memory_traffic, extracted_schedule_commands, extracted_runtime_allocation, extracted_runtime_profile_events, &execution_report.writer);

    const expected_allocations: usize = 9;
    const expected_buffer_uses: usize = 14;
    const expected_lifetimes: usize = 9;
    const expected_stream_steps: usize = 3;
    const expected_events: usize = 5;
    const expected_profile_joins: usize = 8;
    const expected_backend_profile_joins: usize = 2;
    const expected_peak_device_bytes: u128 = 140;
    try std.testing.expectEqual(expected_allocations, allocation_plan.allocations.len);
    try std.testing.expectEqual(planned.report.schedule_commands.len, extracted_schedule_commands.len);
    try std.testing.expectEqual(target_pkg.TargetKind.npu_v0, extracted_target.kind);
    try std.testing.expectEqual(planned.report.target.?.memory_spaces.len, extracted_target.memory_spaces.len);
    try std.testing.expectEqual(planned.report.target.?.execution_units.len, extracted_target.execution_units.len);
    try std.testing.expectEqualStrings("device_hbm", extracted_target.memory_spaces[1].name);
    try std.testing.expectEqual(1000000000000, extracted_target.memory_spaces[1].bandwidth_bytes_per_second.?);
    try std.testing.expectEqual(planned.report.cost_ledger.len, extracted_cost_ledger.len);
    try std.testing.expectEqual(compiler_facts.CostOpClass.matmul, extracted_cost_ledger[0].op_class);
    try std.testing.expectEqual(48, extracted_cost_ledger[0].logical_ops);
    try std.testing.expectEqual(planned.report.lowering_records.len, extracted_lowerings.len);
    try std.testing.expectEqual(compiler_facts.LoweringDecision.backend_kernel_graph, extracted_lowerings[0].decision);
    try std.testing.expectEqual(compiler_facts.LoweringDecision.elementwise_fusion, extracted_lowerings[1].decision);
    try std.testing.expectEqual(extracted_lowerings.len, extracted_lowering_regions.len);
    try std.testing.expectEqual(null, extracted_lowering_regions[0].fusion_group_index);
    try std.testing.expectEqual(compiler_facts.LoweringDecision.backend_kernel_graph, extracted_lowering_regions[0].codegen_region);
    try std.testing.expectEqual(1, extracted_lowering_regions[0].placement_record_indices.len);
    try std.testing.expectEqual(0, extracted_lowering_regions[0].placement_record_indices[0]);
    try std.testing.expectEqual(1, extracted_lowering_regions[1].fusion_group_index.?);
    try std.testing.expectEqual(compiler_facts.LoweringDecision.elementwise_fusion, extracted_lowering_regions[1].codegen_region);
    try std.testing.expectEqual(3, extracted_lowering_regions[1].placement_record_indices.len);
    try std.testing.expectEqual(planned.report.memory_traffic_records.len, extracted_memory_traffic.len);
    try std.testing.expectEqual(compiler_facts.MemoryTrafficKind.global_memory, extracted_memory_traffic[0].kind);
    try std.testing.expectEqual(104, extracted_memory_traffic[0].bytes_read + extracted_memory_traffic[0].bytes_written);
    try std.testing.expectEqual(core.CommandKind.backend_execute, extracted_schedule_commands[1].kind);
    try std.testing.expectEqual(1, extracted_schedule_commands[1].id.index);
    try std.testing.expectEqual(2, extracted_schedule_commands[1].lowering_record_ids.len);
    try std.testing.expectEqual(expected_buffer_uses, allocation_plan.command_buffer_uses.len);
    try std.testing.expectEqual(expected_lifetimes, allocation_plan.lifetimes.len);
    try std.testing.expectEqual(expected_peak_device_bytes, allocation_plan.peak_device_bytes);
    try std.testing.expectEqual(expected_allocations, extracted_runtime_allocation.allocations.len);
    try std.testing.expectEqual(expected_buffer_uses, extracted_runtime_allocation.command_buffer_uses.len);
    try std.testing.expectEqual(expected_peak_device_bytes, extracted_runtime_allocation.peak_device_bytes);
    try std.testing.expectEqualStrings("host", extracted_runtime_allocation.allocations[0].placement);
    try std.testing.expectEqual(0, extracted_runtime_allocation.allocations[0].value_id.index);
    try std.testing.expectEqual(32, extracted_runtime_allocation.allocations[0].size_bytes);
    try std.testing.expectEqual(0, extracted_runtime_allocation.allocations[0].first_command_id.index);
    try std.testing.expectEqual(0, extracted_runtime_allocation.allocations[0].last_command_id.index);
    try std.testing.expectEqualStrings("read", extracted_runtime_allocation.command_buffer_uses[0].access);
    try std.testing.expectEqual(.host, allocation_plan.allocations[0].placement);
    try std.testing.expect(allocation_plan.lifetimes[0].first_command_id.index <= allocation_plan.lifetimes[0].last_command_id.index);
    try std.testing.expect(std.mem.indexOf(u8, allocation_summary.writer.buffered(), "peak_device_bytes=140") != null);
    try std.testing.expectEqual(expected_stream_steps, stream_plan.steps.len);
    try std.testing.expectEqual(expected_stream_steps, extracted_runtime_streams.steps.len);
    try std.testing.expectEqual(1, extracted_runtime_streams.steps[1].command_id.index);
    try std.testing.expectEqual(0, extracted_runtime_streams.steps[1].stream.index);
    try std.testing.expectEqual(2, extracted_runtime_streams.steps[1].start_event_id);
    try std.testing.expectEqual(3, extracted_runtime_streams.steps[1].done_event_id);
    try std.testing.expectEqual(1, extracted_runtime_streams.steps[1].wait_event_ids[0]);
    try std.testing.expectEqual(runtime.RuntimeEventId{ .index = 1 }, stream_plan.steps[1].wait_event_ids[0]);
    try std.testing.expect(std.mem.indexOf(u8, stream_summary.writer.buffered(), "command.1 stream.0 start=event.2 done=event.3 waits=event.1") != null);
    try std.testing.expectEqual(.npu_v0, extracted_backend_executable.backend_kind);
    try std.testing.expectEqual(1, extracted_backend_executable.command_id.index);
    try std.testing.expectEqualStrings("npu_execute", extracted_backend_executable.backend_operation);
    try std.testing.expectEqual(2, extracted_backend_executable.calls.len);
    try std.testing.expectEqualStrings("npu_elementwise_fusion", extracted_backend_executable.calls[1].backend_operation);
    try std.testing.expectEqualStrings("elementwise_fusion", extracted_backend_executable.calls[1].feature);
    try std.testing.expectEqual(3, extracted_backend_executable.calls[1].graph_instruction_ids.len);
    try std.testing.expectEqual(1, extracted_backend_executable.calls[1].graph_instruction_ids[0].index);
    try std.testing.expectEqual(1, extracted_backend_executable.calls[1].expected_unit_id.?);
    try std.testing.expectEqual(expected_events, profiled.report.profile_events.len);
    try std.testing.expectEqual(.h2d, profiled.report.profile_events[0].kind);
    try std.testing.expectEqual(.backend_execute, profiled.report.profile_events[1].kind);
    try std.testing.expectEqual(.backend_execute, profiled.report.profile_events[2].kind);
    try std.testing.expectEqual(.backend_execute, profiled.report.profile_events[3].kind);
    try std.testing.expectEqual(.d2h, profiled.report.profile_events[4].kind);
    try std.testing.expectEqual(expected_events, extracted_runtime_profile_events.len);
    try std.testing.expectEqualStrings("backend_execute", extracted_runtime_profile_events[3].kind);
    try std.testing.expectEqual(1, extracted_runtime_profile_events[3].command_id.?.index);
    try std.testing.expectEqual(156, extracted_runtime_profile_events[3].bytes);
    try std.testing.expectEqual(18, extracted_runtime_profile_events[3].logical_ops);
    try std.testing.expectEqualStrings("ok", extracted_runtime_profile_events[3].status);
    try std.testing.expectEqual(false, extracted_runtime_profile_events[3].forced_synchronization);
    try std.testing.expectEqual(3, extracted_runtime_profile_events[3].graph_instruction_ids.len);
    try std.testing.expectEqual(1, extracted_runtime_profile_events[3].graph_instruction_ids[0].index);
    try std.testing.expectEqual(expected_profile_joins, extracted_runtime_profile_joins.len);
    try std.testing.expectEqualStrings("lowering_record", extracted_runtime_profile_joins[4].subject_kind);
    try std.testing.expectEqual(1, extracted_runtime_profile_joins[4].subject_id);
    try std.testing.expectEqual(1, extracted_runtime_profile_joins[4].command_id.?.index);
    try std.testing.expectEqual(3, extracted_runtime_profile_joins[4].profile_event_ids[0].index);
    try std.testing.expectEqual(3, extracted_runtime_profile_joins[4].graph_instruction_ids.len);
    try std.testing.expectEqual(1, extracted_runtime_profile_joins[4].graph_instruction_ids[0].index);
    try std.testing.expectEqual(expected_backend_profile_joins, extracted_backend_profile_joins.len);
    try std.testing.expectEqual(0, extracted_backend_profile_joins[0].call_index);
    try std.testing.expectEqual(2, extracted_backend_profile_joins[0].profile_event_id.index);
    try std.testing.expectEqual(1, extracted_backend_profile_joins[1].call_index);
    try std.testing.expectEqual(3, extracted_backend_profile_joins[1].profile_event_id.index);
    try std.testing.expectEqual(3, extracted_backend_profile_joins[1].graph_instruction_ids.len);
    try std.testing.expectEqual(planned.report.placement_records.len, profiled.report.placement_records.len);
    try std.testing.expectEqual(planned.report.memory_traffic_records.len, profiled.report.memory_traffic_records.len);
    const expected_explains: usize = 3;
    try std.testing.expectEqual(expected_explains, profiled.report.explain_records.len);
    try std.testing.expectEqual(1, profiled.report.explain_records[0].profile_event_ids.len);
    try std.testing.expectEqual(2, profiled.report.explain_records[0].profile_event_ids[0].index);
    try std.testing.expectEqual(1, profiled.report.explain_records[1].profile_event_ids.len);
    try std.testing.expectEqual(3, profiled.report.explain_records[1].profile_event_ids[0].index);
    try std.testing.expectEqual(1, profiled.report.explain_records[2].profile_event_ids.len);
    try std.testing.expectEqual(1, profiled.report.explain_records[2].profile_event_ids[0].index);
    try std.testing.expect(profiled.report.profile_events[1].logical_ops > 0);
    try std.testing.expect(profiled.report.profile_events[1].graph_instruction_ids.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, runtime_summary.writer.buffered(), "command.1 kind=backend_execute stream=0 predicted_bytes=260 observed_bytes=260 predicted_ops=66 observed_ops=66 event=profile.1") != null);
    try std.testing.expect(std.mem.indexOf(u8, runtime_summary.writer.buffered(), "lowering.0 command=1 decision=backend_kernel_graph instructions=0 predicted_bytes=104 observed_bytes=104 predicted_ops=48 observed_ops=48 event=profile.2") != null);
    try std.testing.expect(std.mem.indexOf(u8, runtime_summary.writer.buffered(), "lowering.1 command=1 decision=elementwise_fusion instructions=1,2,3 predicted_bytes=156 observed_bytes=156 predicted_ops=18 observed_ops=18 event=profile.3") != null);
    try std.testing.expect(std.mem.indexOf(u8, execution_report.writer.buffered(), "traffic.2 lowering=1 memory=1 kind=global_memory bytes_read=36 bytes_written=24 total_bytes=60 bandwidth_bytes_per_second=1000000000000 ideal_memory_ps=60") != null);
    try std.testing.expect(std.mem.indexOf(u8, execution_report.writer.buffered(), "traffic.3 lowering=1 memory=2 kind=local_memory bytes_read=84 bytes_written=72 total_bytes=156 bandwidth_bytes_per_second=20000000000000 ideal_memory_ps=8") != null);
    try std.testing.expect(std.mem.indexOf(u8, execution_report.writer.buffered(), "lowering.0 decision=backend_kernel_graph predicted_ops=48 predicted_bytes=104 observed_ops=48 observed_bytes=104 ideal_compute_ps=2 ideal_memory_ps=104 limiting=memory event=profile.2") != null);
    try std.testing.expect(std.mem.indexOf(u8, execution_report.writer.buffered(), "lowering.1 decision=elementwise_fusion predicted_ops=18 predicted_bytes=156 observed_ops=18 observed_bytes=156 ideal_compute_ps=16 ideal_memory_ps=60 limiting=memory event=profile.3") != null);

    var backend_summary: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer backend_summary.deinit();
    try mlir_state.writeBackendCallProfileFactSummary(extracted_target, extracted_cost_ledger, extracted_memory_traffic, extracted_backend_executable, extracted_runtime_profile_events, extracted_backend_profile_joins, &backend_summary.writer);
    try std.testing.expect(std.mem.indexOf(u8, backend_summary.writer.buffered(), "call.0 command=1 operation=npu_matmul instructions=0 unit=0 predicted_bytes=104 observed_bytes=104 predicted_ops=48 observed_ops=48 ideal_compute_ps=2 ideal_memory_ps=104 limiting=memory memory=memory.1:104B/104ps,memory.2:104B/6ps event=profile.2") != null);
    try std.testing.expect(std.mem.indexOf(u8, backend_summary.writer.buffered(), "call.1 command=1 operation=npu_elementwise_fusion instructions=1,2,3 unit=1 predicted_bytes=156 observed_bytes=156 predicted_ops=18 observed_ops=18 ideal_compute_ps=16 ideal_memory_ps=60 limiting=memory memory=memory.1:60B/60ps,memory.2:156B/8ps event=profile.3") != null);
    var mlir_summary: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer mlir_summary.deinit();
    try mlir_state.writeStateSummary(&session, &mlir_summary.writer);
    try std.testing.expect(std.mem.indexOf(u8, mlir_summary.writer.buffered(), "state=backend_profile_joined") != null);
    try std.testing.expect(std.mem.indexOf(u8, mlir_summary.writer.buffered(), "runtime_allocation allocations=9 uses=14 peak_device_bytes=140") != null);
    try std.testing.expect(std.mem.indexOf(u8, mlir_summary.writer.buffered(), "allocation.0 value=0 placement=host memory=0 bytes=32 lifetime=command.0..command.0") != null);
    try std.testing.expect(std.mem.indexOf(u8, mlir_summary.writer.buffered(), "runtime_stream steps=3") != null);
    try std.testing.expect(std.mem.indexOf(u8, mlir_summary.writer.buffered(), "stream_step command=1 stream=0 start=event.2 done=event.3 waits=1") != null);
    try std.testing.expect(std.mem.indexOf(u8, mlir_summary.writer.buffered(), "lowering_plan count=2") != null);
    try std.testing.expect(std.mem.indexOf(u8, mlir_summary.writer.buffered(), "lowering.1 decision=elementwise_fusion instructions=1,2,3 costs=1,2,3") != null);
    try std.testing.expect(std.mem.indexOf(u8, mlir_summary.writer.buffered(), "lowering_region_facts count=2") != null);
    try std.testing.expect(std.mem.indexOf(u8, mlir_summary.writer.buffered(), "lowering_region.1 fusion=1 placements=1,2,3") != null);
    try std.testing.expect(std.mem.indexOf(u8, mlir_summary.writer.buffered(), "codegen_region=elementwise_fusion") != null);
    try std.testing.expect(std.mem.indexOf(u8, mlir_summary.writer.buffered(), "performance_cost_ledger count=4") != null);
    try std.testing.expect(std.mem.indexOf(u8, mlir_summary.writer.buffered(), "cost.0 class=matmul dtype=f32 instructions=0 logical_ops=48 bytes_read=80 bytes_written=24 unit=0") != null);
    try std.testing.expect(std.mem.indexOf(u8, mlir_summary.writer.buffered(), "performance_memory_traffic count=4") != null);
    try std.testing.expect(std.mem.indexOf(u8, mlir_summary.writer.buffered(), "traffic.3 lowering=1 memory=2 kind=local_memory instructions=1,2,3 bytes_read=84 bytes_written=72") != null);
    try std.testing.expect(std.mem.indexOf(u8, mlir_summary.writer.buffered(), "runtime_profile events=5") != null);
    try std.testing.expect(std.mem.indexOf(u8, mlir_summary.writer.buffered(), "profile.3 kind=backend_execute command=1 instructions=1,2,3 bytes=156 logical_ops=18 status=ok forced_sync=false") != null);
    try std.testing.expect(std.mem.indexOf(u8, mlir_summary.writer.buffered(), "runtime_profile_join joins=8") != null);
    try std.testing.expect(std.mem.indexOf(u8, mlir_summary.writer.buffered(), "profile_join.4 subject=lowering_record.1 command=1 events=3 instructions=1,2,3") != null);
    try std.testing.expect(std.mem.indexOf(u8, mlir_summary.writer.buffered(), "profile_join.7 subject=backend_binding.0 command=1 events=1 instructions=0,1,2,3") != null);
    try std.testing.expectEqual(expected_profile_joins, countProfileJoinSummaryLines(mlir_summary.writer.buffered()));
    try std.testing.expect(std.mem.indexOf(u8, mlir_summary.writer.buffered(), "backend_profile_join joins=2") != null);
    try std.testing.expect(std.mem.indexOf(u8, mlir_summary.writer.buffered(), "backend_profile_join.0 call=0 command=1 event=2 instructions=0") != null);
    try std.testing.expect(std.mem.indexOf(u8, mlir_summary.writer.buffered(), "backend_profile_join.1 call=1 command=1 event=3 instructions=1,2,3") != null);
    try std.testing.expectEqual(expected_backend_profile_joins, countBackendProfileJoinSummaryLines(mlir_summary.writer.buffered()));
    try std.testing.expectEqualStrings(expected_report, execution_report.writer.buffered());
    try std.testing.expectEqualStrings("", diagnostics.writer.buffered());
}

fn countProfileJoinSummaryLines(text: []const u8) usize {
    var count: usize = 0;
    var rest = text;
    while (std.mem.indexOf(u8, rest, "  profile_join.")) |index| {
        count += 1;
        rest = rest[index + "  profile_join.".len ..];
    }
    return count;
}

fn countBackendProfileJoinSummaryLines(text: []const u8) usize {
    var count: usize = 0;
    var rest = text;
    while (std.mem.indexOf(u8, rest, "  backend_profile_join.")) |index| {
        count += 1;
        rest = rest[index + "  backend_profile_join.".len ..];
    }
    return count;
}
