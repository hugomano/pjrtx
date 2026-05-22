const std = @import("std");

const ir = @import("src/compiler/ir");
const mlx_metal = @import("src/backend/mlx_metal");
const dense = @import("msl_dense_elementwise.zig");
const refs = @import("msl_artifact_refs.zig");

/// Writes the manifest that maps compiler/backend IR to generated Metal artifacts.
pub fn writeManifest(writer: *std.Io.Writer, stem: []const u8, plan: *const ir.ExecutablePlan, program: *const mlx_metal.Program) !void {
    try writeHeader(writer, stem, plan, program);
    try writeValues(writer, plan, program);
    try writeInstructions(writer, stem, plan);
    try writeMetalCppKernels(writer, stem, plan);
    try writeNodes(writer, plan, program);
    try writeSchedule(writer, plan, program);
    try writeMaterializationBoundaries(writer, plan, program);
    try writeSubprograms(writer, program);
}

fn writeHeader(writer: *std.Io.Writer, stem: []const u8, plan: *const ir.ExecutablePlan, program: *const mlx_metal.Program) !void {
    try writer.print(
        \\backend_program_manifest version=1 artifact={s}
        \\counts values={d} instructions={d} schedule_items={d} nodes={d} edges={d} fusion_groups={d} materialization_boundaries={d} subprograms={d} control_flows={d}
        \\topology replicas={d} partitions={d} assigned_devices={d}
        \\
        \\values:
        \\
    , .{
        stem,
        plan.values.len,
        plan.instructions.len,
        program.schedule.len,
        program.nodes.len,
        program.edges.len,
        program.fusion_groups.len,
        program.materialization_boundaries.len,
        program.subprograms.len,
        program.control_flows.len,
        plan.options.num_replicas,
        plan.options.num_partitions,
        plan.options.device_assignment.len,
    });
}

fn writeValues(writer: *std.Io.Writer, plan: *const ir.ExecutablePlan, program: *const mlx_metal.Program) !void {
    for (plan.values, 0..) |value, value_index| {
        try writer.print("  v{d} role={s} storage={s} ", .{ value_index, @tagName(value.role), @tagName(value.storage) });
        try refs.writeDescriptor(writer, value.descriptor);
        if (value.elements.len != 0) {
            try refs.writeValueRefList(writer, plan, " elements", value.elements);
        }
        if (value_index < program.values.len) {
            const program_value = program.values[value_index];
            try writer.print(" bytes={d}", .{program_value.byte_size});
            try refs.writeOptionalUsize(writer, " producer_node", program_value.producer_node);
            try refs.writeOptionalUsize(writer, " last_use_node", program_value.last_use_node);
            try writer.print(" output={d}", .{@intFromBool(program_value.is_output)});
            try refs.writeOptionalUsize(writer, " materialization_boundary", program_value.materialization_boundary);
        }
        try writer.writeByte('\n');
    }
}

fn writeInstructions(writer: *std.Io.Writer, stem: []const u8, plan: *const ir.ExecutablePlan) !void {
    try writer.writeAll("\ninstructions:\n");
    for (plan.instructions, 0..) |instruction, instruction_index| {
        try writer.print("  i{d} op={s}", .{ instruction_index, @tagName(instruction.kind) });
        try refs.writeValueRefList(writer, plan, " inputs", instruction.inputs);
        try refs.writeValueRefList(writer, plan, " outputs", instruction.outputs);
        try refs.writeInstructionAttributes(writer, instruction);
        if (dense.Kernel.init(plan, instruction, instruction_index)) |kernel| {
            try writer.writeAll(" manifest_label=dense_elementwise runner=metalcpp_dense_elementwise_f32 kernel=");
            try kernel.writeSymbol(writer);
            try writer.print(" element_count={d}", .{kernel.element_count});
        }
        _ = stem;
        try writer.writeByte('\n');
    }
}

fn writeMetalCppKernels(writer: *std.Io.Writer, stem: []const u8, plan: *const ir.ExecutablePlan) !void {
    try writer.writeAll("\nmetalcpp_kernels:\n");
    for (plan.instructions, 0..) |instruction, instruction_index| {
        const kernel = dense.Kernel.init(plan, instruction, instruction_index) orelse continue;
        try kernel.writeManifest(writer, stem, plan);
    }
}

fn writeNodes(writer: *std.Io.Writer, plan: *const ir.ExecutablePlan, program: *const mlx_metal.Program) !void {
    try writer.writeAll("\nnodes:\n");
    for (program.nodes, 0..) |node, node_index| {
        try writer.print("  n{d} instruction={d}", .{ node_index, node.instruction_index });
        if (node.instruction_index < plan.instructions.len) {
            try writer.print(" op={s}", .{@tagName(plan.instructions[node.instruction_index].kind)});
        }
        try writer.print(" kind={s} materializes={d}", .{ @tagName(node.kind), @intFromBool(node.materializes) });
        try refs.writeOptionalUsize(writer, " fusion_group", node.fusion_group);
        try refs.writeValueRefList(writer, plan, " inputs", node.inputs);
        try refs.writeValueRefList(writer, plan, " outputs", node.outputs);
        try refs.writeUsizeList(writer, " subprograms", node.subprograms);
        try refs.writeOptionalUsize(writer, " control_flow", node.control_flow);
        try writer.writeByte('\n');
    }
}

fn writeSchedule(writer: *std.Io.Writer, plan: *const ir.ExecutablePlan, program: *const mlx_metal.Program) !void {
    try writer.writeAll("\nschedule:\n");
    for (program.schedule, 0..) |item, schedule_index| {
        try writer.print("  s{d} kind={s} index={d} count={d}", .{ schedule_index, @tagName(item.kind), item.index, item.count });
        switch (item.kind) {
            .node => try writeNodeScheduleItem(writer, plan, program, item.index),
            .fusion_group => try writeFusionGroupScheduleItem(writer, plan, program, item.index),
            .materialization_boundary => try writeBoundaryScheduleItem(writer, plan, program, item.index),
        }
        try writer.writeByte('\n');
    }
}

fn writeNodeScheduleItem(writer: *std.Io.Writer, plan: *const ir.ExecutablePlan, program: *const mlx_metal.Program, index: usize) !void {
    if (index >= program.nodes.len) return;
    const node = program.nodes[index];
    try writer.print(" node_kind={s} instruction={d}", .{ @tagName(node.kind), node.instruction_index });
    if (node.instruction_index < plan.instructions.len) {
        try writer.print(" op={s}", .{@tagName(plan.instructions[node.instruction_index].kind)});
    }
}

fn writeFusionGroupScheduleItem(writer: *std.Io.Writer, plan: *const ir.ExecutablePlan, program: *const mlx_metal.Program, index: usize) !void {
    if (index >= program.fusion_groups.len) return;
    const group = program.fusion_groups[index];
    try writer.print(" group_kind={s} first_node={d} last_node={d} node_count={d}", .{ @tagName(group.kind), group.first_node, group.last_node, group.node_count });
    try refs.writeUsizeList(writer, " nodes", group.node_indices);
    try refs.writeValueRefList(writer, plan, " inputs", group.input_values);
    try refs.writeValueRefList(writer, plan, " outputs", group.output_values);
    try refs.writeFusionGroupOps(writer, plan, program, group);
}

fn writeBoundaryScheduleItem(writer: *std.Io.Writer, plan: *const ir.ExecutablePlan, program: *const mlx_metal.Program, index: usize) !void {
    if (index >= program.materialization_boundaries.len) return;
    const boundary = program.materialization_boundaries[index];
    try writer.print(" reason={s} value=", .{@tagName(boundary.reason)});
    try refs.writeValueRef(writer, plan, boundary.value_id);
}

fn writeMaterializationBoundaries(writer: *std.Io.Writer, plan: *const ir.ExecutablePlan, program: *const mlx_metal.Program) !void {
    try writer.writeAll("\nmaterialization_boundaries:\n");
    for (program.materialization_boundaries, 0..) |boundary, boundary_index| {
        try writer.print("  m{d} reason={s} value=", .{ boundary_index, @tagName(boundary.reason) });
        try refs.writeValueRef(writer, plan, boundary.value_id);
        try writer.writeByte('\n');
    }
}

fn writeSubprograms(writer: *std.Io.Writer, program: *const mlx_metal.Program) !void {
    try writer.writeAll("\nsubprograms:\n");
    for (program.subprograms, 0..) |subprogram, subprogram_index| {
        try writer.print("  sub{d} kind={s} parent_node={d} region={d} values={d} instructions={d} returns={d}", .{
            subprogram_index,
            @tagName(subprogram.kind),
            subprogram.parent_node,
            subprogram.region_id.index,
            subprogram.values.len,
            subprogram.instructions.len,
            subprogram.return_descriptors.len,
        });
        try writer.writeByte('\n');
    }
}

