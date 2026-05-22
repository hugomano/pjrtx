const std = @import("std");

const ir = @import("src/compiler/ir");
const custom = @import("executable_metal_graph_custom_call.zig");
const dot = @import("metal_graph_dot.zig");
const fusion = @import("executable_metal_graph_fusion.zig");
const generation = @import("executable_metal_graph_generation.zig");
const gather = @import("executable_metal_graph_gather.zig");
const map_mod = @import("executable_metal_graph_map.zig");
const map_rules = @import("executable_metal_graph_map_rules.zig");
const metalcpp_call = @import("metalcpp_call.zig");
const metal_graph_lowering = @import("executable_metal_graph_lowering.zig");
const program_mod = @import("program.zig");
const reduction = @import("executable_metal_graph_reduction.zig");
const slice_mod = @import("executable_metal_graph_slice.zig");
const step_storage = @import("executable_metal_graph_step_storage.zig");
const structural = @import("executable_metal_graph_structural.zig");
const tensor = @import("executable_metal_graph_tensor.zig");

/// Builds one executable Metal graph step for a supported instruction.
pub fn makeStep(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize) program_mod.Error!?metalcpp_call.ExecutableStepSpec {
    return makeStepWithReleasedInputs(allocator, plan, instruction, instruction_index, &.{});
}

/// Builds one executable Metal graph step with planned value releases attached later by the caller.
pub fn makeStepWithReleasedInputs(
    allocator: std.mem.Allocator,
    plan: *const ir.ExecutablePlan,
    instruction: ir.PlanInstruction,
    instruction_index: usize,
    release_values: []const u64,
) program_mod.Error!?metalcpp_call.ExecutableStepSpec {
    if (try structural.makeAliasStep(allocator, plan, instruction)) |step| return step;
    if (map_rules.kindFor(instruction.kind)) |map_kind| {
        return map_mod.makeMapStep(allocator, plan, instruction, instruction_index, map_kind);
    }
    return switch (instruction.kind) {
        .reshape, .copy_arg0, .reduce_precision, .bitcast_convert => structural.makeCopyStep(allocator, plan, instruction, instruction_index),
        .convert => structural.makeConvertStep(allocator, plan, instruction, instruction_index),
        .slice => slice_mod.makeSliceStep(allocator, plan, instruction, instruction_index),
        .broadcast_in_dim => structural.makeBroadcastStep(allocator, plan, instruction, instruction_index),
        .dynamic_slice => slice_mod.makeDynamicSliceStep(allocator, plan, instruction, instruction_index),
        .dynamic_update_slice => slice_mod.makeDynamicUpdateSliceStep(allocator, plan, instruction, instruction_index, release_values),
        .transpose => structural.makeTransposeStep(allocator, plan, instruction, instruction_index),
        .gather => gather.makeGatherStep(allocator, plan, instruction, instruction_index),
        .concatenate => generation.makeConcatenateStep(allocator, plan, instruction, instruction_index),
        .iota => generation.makeIotaStep(allocator, plan, instruction, instruction_index),
        .custom_call => custom.makeCustomCallStep(allocator, plan, instruction, instruction_index),
        .reduce_sum, .reduce_max, .reduce_min => reduction.makeReduceStep(allocator, plan, instruction, instruction_index),
        .dot_general => dot.makeStep(allocator, plan, instruction, instruction_index),
        else => null,
    };
}

/// Appends one instruction step and updates lowering coverage metrics.
pub fn appendInstructionStep(
    allocator: std.mem.Allocator,
    plan: *const ir.ExecutablePlan,
    instruction: ir.PlanInstruction,
    instruction_index: usize,
    origin: metal_graph_lowering.StepOrigin,
    release_values: []const u64,
    steps: *std.ArrayList(metalcpp_call.ExecutableStepSpec),
    metrics: *metal_graph_lowering.Metrics,
) program_mod.Error!void {
    if (instruction.kind == .constant) return;
    const maybe_step = try makeStepWithReleasedInputs(allocator, plan, instruction, instruction_index, release_values);
    var step = maybe_step orelse return error.UnsupportedElementType;
    try step_storage.StepStorage.attachReleaseValues(allocator, &step, release_values);
    try step_storage.StepStorage.append(allocator, steps, step);
    metrics.recordStep(origin, step.source.len == 0);
    recordOperationCoverage(metrics, instruction.kind, step.kernel_name);
}

fn recordOperationCoverage(metrics: *metal_graph_lowering.Metrics, kind: ir.PlanInstructionKind, kernel_name: []const u8) void {
    switch (kind) {
        .dot_general => {
            metrics.dot_step_count += 1;
            if (std.mem.indexOf(u8, kernel_name, "tiled_dot") != null) metrics.tiled_dot_step_count += 1;
        },
        .reduce_sum, .reduce_max, .reduce_min => {
            if (std.mem.indexOf(u8, kernel_name, "reduce_max_index") != null) {
                metrics.reduce_max_index_step_count += 1;
                if (std.mem.indexOf(u8, kernel_name, "tiled_reduce_max_index") != null) metrics.tiled_reduce_max_index_step_count += 1;
            } else {
                metrics.reduce_step_count += 1;
                if (std.mem.indexOf(u8, kernel_name, "tiled_reduce") != null) metrics.tiled_reduce_step_count += 1;
            }
        },
        else => {},
    }
}

/// Records coverage for a generated multi-dot step.
pub fn recordDotGroupCoverage(metrics: *metal_graph_lowering.Metrics, node_count: usize, kernel_name: []const u8) void {
    metrics.dot_step_count += node_count;
    if (std.mem.indexOf(u8, kernel_name, "tiled_dot_group") != null) {
        metrics.tiled_dot_step_count += node_count;
    }
}
