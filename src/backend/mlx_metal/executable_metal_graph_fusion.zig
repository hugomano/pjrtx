const std = @import("std");

const ir = @import("src/compiler/ir");
const metalcpp_call = @import("metalcpp_call.zig");
const metal_graph_lowering = @import("executable_metal_graph_lowering.zig");
const program_mod = @import("program.zig");
const request_mod = @import("executable_metal_graph_fusion_request.zig");
const tensor = @import("executable_metal_graph_tensor.zig");

/// Builds one generated Metal step for a backend fusion group.
pub fn makeFusionGroupStep(
    allocator: std.mem.Allocator,
    plan: *const ir.ExecutablePlan,
    program: *const program_mod.Program,
    group: program_mod.FusionGroup,
) program_mod.Error!?metalcpp_call.ExecutableStepSpec {
    var request = (try request_mod.Request.init(allocator, plan, program, group)) orelse return null;
    defer request.deinit(allocator);
    const kernel_name = try std.fmt.allocPrint(allocator, "pjrtx_metalcpp_program_fusion_{x}_{d}", .{ @intFromPtr(plan), group.id });
    errdefer allocator.free(kernel_name);
    var source = std.Io.Writer.Allocating.init(allocator);
    errdefer source.deinit();
    request.writeMsl(allocator, &source.writer, kernel_name) catch return error.OutOfMemory;
    const inputs = try tensor.valueIndices(allocator, group.input_values);
    errdefer allocator.free(inputs);
    const outputs = try tensor.valueIndices(allocator, group.output_values);
    errdefer allocator.free(outputs);
    return .{
        .kernel_name = kernel_name,
        .source = try source.toOwnedSlice(),
        .inputs = inputs,
        .outputs = outputs,
        .release_values = &.{},
        .element_count = request.elementCount(),
    };
}

/// Builds one generated Metal step for a split fusion segment.
pub fn makeFusionSegmentStep(
    allocator: std.mem.Allocator,
    plan: *const ir.ExecutablePlan,
    program: *const program_mod.Program,
    segment: metal_graph_lowering.StepPlan.FusionSegment,
) program_mod.Error!?metalcpp_call.ExecutableStepSpec {
    if (segment.group_index >= program.fusion_groups.len or segment.node_indices.len == 0) return null;
    const owner = program.fusion_groups[segment.group_index];
    const group = program_mod.FusionGroup{
        .id = owner.id,
        .kind = owner.kind,
        .first_node = segment.node_indices[0],
        .last_node = segment.node_indices[segment.node_indices.len - 1],
        .node_count = segment.node_indices.len,
        .node_indices = segment.node_indices,
        .input_values = segment.input_values,
        .output_values = segment.output_values,
    };
    return makeFusionGroupStep(allocator, plan, program, group);
}
