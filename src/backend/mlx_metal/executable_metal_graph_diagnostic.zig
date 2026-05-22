const std = @import("std");

const ir = @import("src/compiler/ir");
const metal_graph_lowering = @import("executable_metal_graph_lowering.zig");
const profiling = @import("profiling.zig");
const step_mod = @import("executable_metal_graph_step.zig");
const step_storage = @import("executable_metal_graph_step_storage.zig");
const tensor = @import("executable_metal_graph_tensor.zig");

/// Reports executable Metal graph compile coverage and first unsupported detail.
pub const CompileCoverage = struct {
    pub fn write(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, maybe_metrics: ?metal_graph_lowering.Metrics, device_count: usize, compiled_count: usize, reason: []const u8, compile_error: ?[]const u8) void {
        writeCompileCoverage(allocator, plan, maybe_metrics, device_count, compiled_count, reason, compile_error);
    }
};

fn writeCompileCoverage(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, maybe_metrics: ?metal_graph_lowering.Metrics, device_count: usize, compiled_count: usize, reason: []const u8, compile_error: ?[]const u8) void {
    if (!profiling.enabled() and !profiling.traceEnabled()) return;
    const unsupported = if (compiled_count == 0) firstUnsupportedDetail(allocator, plan) catch null else null;
    defer if (unsupported) |detail| allocator.free(detail);
    const metrics = maybe_metrics orelse metal_graph_lowering.Metrics{};
    profiling.writeMetalGraphCompile(.{
        .plan_address = @intFromPtr(plan),
        .value_count = plan.values.len,
        .instruction_count = plan.instructions.len,
        .output_count = plan.output_ids.len,
        .device_count = device_count,
        .compiled_device_count = compiled_count,
        .schedule_item_count = metrics.schedule_item_count,
        .schedule_node_count = metrics.schedule_node_count,
        .schedule_fusion_group_count = metrics.schedule_fusion_group_count,
        .schedule_materialization_boundary_count = metrics.schedule_materialization_boundary_count,
        .input_value_count = metrics.input_value_count,
        .constant_input_count = metrics.constant_input_count,
        .step_count = metrics.step_count,
        .node_step_count = metrics.node_step_count,
        .fusion_step_count = metrics.fusion_step_count,
        .alias_step_count = metrics.alias_step_count,
        .fallback_group_count = metrics.fallback_group_count,
        .fallback_node_step_count = metrics.fallback_node_step_count,
        .planned_structural_step_count = metrics.planned_structural_step_count,
        .planned_view_step_count = metrics.planned_view_step_count,
        .planned_map_step_count = metrics.planned_map_step_count,
        .planned_reduction_step_count = metrics.planned_reduction_step_count,
        .planned_matmul_step_count = metrics.planned_matmul_step_count,
        .planned_control_flow_step_count = metrics.planned_control_flow_step_count,
        .planned_materialize_step_count = metrics.planned_materialize_step_count,
        .planned_library_call_step_count = metrics.planned_library_call_step_count,
        .planned_other_step_count = metrics.planned_other_step_count,
        .fallback_structural_step_count = metrics.fallback_structural_step_count,
        .fallback_view_step_count = metrics.fallback_view_step_count,
        .fallback_map_step_count = metrics.fallback_map_step_count,
        .fallback_reduction_step_count = metrics.fallback_reduction_step_count,
        .fallback_matmul_step_count = metrics.fallback_matmul_step_count,
        .fallback_control_flow_step_count = metrics.fallback_control_flow_step_count,
        .fallback_materialize_step_count = metrics.fallback_materialize_step_count,
        .fallback_library_call_step_count = metrics.fallback_library_call_step_count,
        .fallback_other_step_count = metrics.fallback_other_step_count,
        .release_step_count = metrics.release_step_count,
        .release_value_count = metrics.release_value_count,
        .dot_step_count = metrics.dot_step_count,
        .tiled_dot_step_count = metrics.tiled_dot_step_count,
        .dot_group_step_count = metrics.dot_group_step_count,
        .dot_group_node_count = metrics.dot_group_node_count,
        .reduce_step_count = metrics.reduce_step_count,
        .tiled_reduce_step_count = metrics.tiled_reduce_step_count,
        .reduce_max_index_step_count = metrics.reduce_max_index_step_count,
        .tiled_reduce_max_index_step_count = metrics.tiled_reduce_max_index_step_count,
        .reason = reason,
        .first_unsupported_op = unsupported,
        .compile_error = compile_error,
    });
}

fn firstUnsupportedDetail(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan) ![]const u8 {
    for (plan.values, 0..) |value, index| {
        if (value.id.index != index) {
            return std.fmt.allocPrint(allocator, "value={d} id_mismatch actual={d}", .{ index, value.id.index });
        }
        if (value.storage != .tensor) {
            return std.fmt.allocPrint(allocator, "value={d} storage={s}", .{ index, @tagName(value.storage) });
        }
        if (value.descriptor.layout != .dense_row_major) {
            return std.fmt.allocPrint(allocator, "value={d} layout={s}", .{ index, @tagName(value.descriptor.layout) });
        }
        if (!tensor.supportedProgramElementType(value.descriptor.element_type)) {
            return valueUnsupportedDetail(allocator, index, value, "unsupported_dtype");
        }
        if (tensor.tensorSpec(value.descriptor) == null) {
            return valueUnsupportedDetail(allocator, index, value, "unsupported_tensor_spec");
        }
    }

    for (plan.output_ids, 0..) |output_id, output_index| {
        if (output_id.index >= plan.values.len) {
            return std.fmt.allocPrint(allocator, "output={d} value={d} out_of_range", .{ output_index, output_id.index });
        }
    }

    for (plan.instructions, 0..) |instruction, instruction_index| {
        switch (instruction.kind) {
            .constant,
            .add,
            .subtract,
            .multiply,
            .divide,
            .maximum,
            .minimum,
            .power,
            .atan2,
            .remainder,
            .negate,
            .exp,
            .expm1,
            .tanh,
            .sqrt,
            .rsqrt,
            .abs,
            .ceil,
            .floor,
            .log,
            .log1p,
            .logistic,
            .sine,
            .cosine,
            .sign,
            .compare,
            .select,
            .clamp,
            .reshape,
            .slice,
            .copy_arg0,
            .reduce_precision,
            .convert,
            .bitcast_convert,
            .broadcast_in_dim,
            .dynamic_slice,
            .dynamic_update_slice,
            .transpose,
            .gather,
            .concatenate,
            .iota,
            .custom_call,
            .reduce_sum,
            .reduce_max,
            .reduce_min,
            .dot_general,
            => {},
            else => return instructionUnsupportedDetail(allocator, plan, instruction, instruction_index, "unsupported_op"),
        }
        if (instruction.kind == .constant) continue;
        if (try step_mod.makeStep(allocator, plan, instruction, instruction_index)) |step| {
            step_storage.StepStorage.deinitStep(allocator, step);
        } else {
            return instructionUnsupportedDetail(allocator, plan, instruction, instruction_index, "unsupported_form");
        }
    }
    return try allocator.dupe(u8, "unknown");
}

fn valueUnsupportedDetail(allocator: std.mem.Allocator, value_index: usize, value: ir.Value, reason: []const u8) ![]const u8 {
    var text = std.Io.Writer.Allocating.init(allocator);
    errdefer text.deinit();
    try text.writer.print("value={d} reason={s} role={s} dtype={s} rank={d} shape=", .{
        value_index,
        reason,
        @tagName(value.role),
        @tagName(value.descriptor.element_type),
        value.descriptor.dims.len,
    });
    try writeDims(&text.writer, value.descriptor.dims);
    return try text.toOwnedSlice();
}

fn instructionUnsupportedDetail(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, reason: []const u8) ![]const u8 {
    var text = std.Io.Writer.Allocating.init(allocator);
    errdefer text.deinit();
    try text.writer.print("instruction={d} op={s} reason={s} inputs=", .{ instruction_index, @tagName(instruction.kind), reason });
    try writeValueList(&text.writer, plan, instruction.inputs);
    try text.writer.writeAll(" outputs=");
    try writeValueList(&text.writer, plan, instruction.outputs);
    if (instruction.custom_call_target) |target| {
        try text.writer.print(" custom_call=\"{s}\"", .{target});
    }
    return try text.toOwnedSlice();
}

fn writeValueList(writer: *std.Io.Writer, plan: *const ir.ExecutablePlan, value_ids: []const ir.ValueId) !void {
    try writer.writeAll("[");
    for (value_ids, 0..) |value_id, index| {
        if (index != 0) try writer.writeAll(",");
        if (value_id.index >= plan.values.len) {
            try writer.print("v{d}:out_of_range", .{value_id.index});
            continue;
        }
        const value = plan.values[value_id.index];
        try writer.print("v{d}:{s}", .{ value_id.index, @tagName(value.descriptor.element_type) });
        try writeDims(writer, value.descriptor.dims);
    }
    try writer.writeAll("]");
}

fn writeDims(writer: *std.Io.Writer, dims: []const i64) !void {
    try writer.writeAll("[");
    for (dims, 0..) |dim, index| {
        if (index != 0) try writer.writeAll(",");
        try writer.print("{d}", .{dim});
    }
    try writer.writeAll("]");
}
