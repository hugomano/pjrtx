const std = @import("std");

const ir = @import("src/compiler/ir");
const dot = @import("metal_graph_dot.zig");
const metalcpp_call = @import("metalcpp_call.zig");
const metal_graph_lowering = @import("executable_metal_graph_lowering.zig");
const program_mod = @import("program.zig");
const resident_inputs = @import("executable_metal_graph_resident_inputs.zig");
const step_mod = @import("executable_metal_graph_step.zig");
const step_storage = @import("executable_metal_graph_step_storage.zig");
const tensor = @import("executable_metal_graph_tensor.zig");
const types = @import("execution_types.zig");

/// Owns value specs, inputs, outputs, and steps for a full executable Metal graph program.
pub const Request = struct {
    value_specs: []metalcpp_call.ExecutableValueSpec,
    input_values: []u64,
    output_values: []u64,
    constant_instruction_indices: []usize,
    steps: []metalcpp_call.ExecutableStepSpec,
    metrics: metal_graph_lowering.Metrics,

    pub fn init(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, program: *const program_mod.Program) program_mod.Error!?Request {
        return initRequest(allocator, plan, program);
    }

    pub fn deinit(self: *Request, allocator: std.mem.Allocator) void {
        deinitRequest(self, allocator);
    }

    /// Collects runtime arguments plus resident constants for this request.
    pub fn inputHandles(self: Request, allocator: std.mem.Allocator, source: resident_inputs.Source, arguments: []const types.BufferHandle) types.Error![]types.BufferHandle {
        return ProgramRequestInputs.collect(allocator, self, source, arguments);
    }
};

fn initRequest(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, program: *const program_mod.Program) program_mod.Error!?Request {
    if (plan.output_ids.len == 0 or plan.values.len == 0) return null;
    const value_specs = try allocator.alloc(metalcpp_call.ExecutableValueSpec, plan.values.len);
    errdefer allocator.free(value_specs);
    for (plan.values, 0..) |value, index| {
        if (value.storage != .tensor) return null;
        if (value.id.index != index or value.descriptor.layout != .dense_row_major or !tensor.supportedProgramElementType(value.descriptor.element_type)) return null;
        value_specs[index] = tensor.tensorSpec(value.descriptor) orelse return null;
    }

    const parameter_count = plan.parameter_shardings.len;
    const constant_count = tensor.countConstants(plan);
    const input_values = try allocator.alloc(u64, parameter_count + constant_count);
    errdefer allocator.free(input_values);
    const constant_instruction_indices = try allocator.alloc(usize, constant_count);
    errdefer allocator.free(constant_instruction_indices);
    var parameter_index: usize = 0;
    for (plan.values) |value| {
        if (value.role != .parameter) continue;
        if (parameter_index >= parameter_count) return error.CommandSubmissionFailed;
        input_values[parameter_index] = value.id.index;
        parameter_index += 1;
    }
    if (parameter_index != parameter_count) return error.CommandSubmissionFailed;
    var constant_index: usize = 0;
    for (plan.instructions, 0..) |instruction, instruction_index| {
        if (instruction.kind != .constant) continue;
        if (instruction.outputs.len != 1) return null;
        const output_id = instruction.outputs[0];
        if (output_id.index >= plan.values.len) return null;
        input_values[parameter_count + constant_index] = output_id.index;
        constant_instruction_indices[constant_index] = instruction_index;
        constant_index += 1;
    }
    const output_values = try allocator.alloc(u64, plan.output_ids.len);
    errdefer allocator.free(output_values);
    for (plan.output_ids, 0..) |output_id, output_index| {
        if (output_id.index >= plan.values.len) return null;
        output_values[output_index] = output_id.index;
    }

    var lowered = try metal_graph_lowering.run(allocator, plan, program);
    defer lowered.deinit(allocator);
    var steps: std.ArrayList(metalcpp_call.ExecutableStepSpec) = .empty;
    var metrics = lowered.metrics;
    metrics.input_value_count = input_values.len;
    metrics.constant_input_count = constant_instruction_indices.len;
    errdefer step_storage.StepStorage.deinitSteps(allocator, steps.items);
    errdefer steps.deinit(allocator);
    for (lowered.steps) |step_plan| {
        switch (step_plan) {
            .instruction => |instruction_plan| {
                if (instruction_plan.index >= plan.instructions.len) return error.CommandSubmissionFailed;
                const instruction = plan.instructions[instruction_plan.index];
                try step_mod.appendInstructionStep(allocator, plan, instruction, instruction_plan.index, instruction_plan.origin, instruction_plan.release_values, &steps, &metrics);
            },
            .fusion_group => |group_plan| {
                const group_index = group_plan.index;
                if (group_index >= program.fusion_groups.len) return error.CommandSubmissionFailed;
                const group = program.fusion_groups[group_index];
                if (try @import("executable_metal_graph_fusion.zig").makeFusionGroupStep(allocator, plan, program, group)) |step| {
                    var owned_step = step;
                    try step_storage.StepStorage.attachReleaseValues(allocator, &owned_step, group_plan.release_values);
                    try step_storage.StepStorage.append(allocator, &steps, owned_step);
                    metrics.recordStep(.fusion_group, step.source.len == 0);
                } else {
                    metrics.fallback_group_count += 1;
                    for (group.node_indices) |node_index| {
                        if (node_index >= program.nodes.len) return error.CommandSubmissionFailed;
                        const node = program.nodes[node_index];
                        if (node.fusion_group != group_index) return error.CommandSubmissionFailed;
                        const instruction = plan.instructions[node.instruction_index];
                        const release_values = if (node_index == group.last_node) group_plan.release_values else &.{};
                        try step_mod.appendInstructionStep(allocator, plan, instruction, node.instruction_index, .fusion_fallback_node, release_values, &steps, &metrics);
                    }
                }
            },
            .fusion_segment => |segment_plan| {
                if (try @import("executable_metal_graph_fusion.zig").makeFusionSegmentStep(allocator, plan, program, segment_plan)) |segment_step| {
                    var step = segment_step;
                    try step_storage.StepStorage.attachReleaseValues(allocator, &step, segment_plan.release_values);
                    try step_storage.StepStorage.append(allocator, &steps, step);
                    metrics.recordStep(.fusion_group, step.source.len == 0);
                } else {
                    metrics.fallback_group_count += 1;
                    for (segment_plan.node_indices) |node_index| {
                        if (node_index >= program.nodes.len) return error.CommandSubmissionFailed;
                        const node = program.nodes[node_index];
                        const instruction = plan.instructions[node.instruction_index];
                        const release_values = if (node_index == segment_plan.node_indices[segment_plan.node_indices.len - 1]) segment_plan.release_values else &.{};
                        try step_mod.appendInstructionStep(allocator, plan, instruction, node.instruction_index, .fusion_fallback_node, release_values, &steps, &metrics);
                    }
                }
            },
            .dot_group => |group_plan| {
                var step = (try dot.makeGroupStep(allocator, plan, program, group_plan.node_indices)) orelse return error.UnsupportedElementType;
                try step_storage.StepStorage.attachReleaseValues(allocator, &step, group_plan.release_values);
                try step_storage.StepStorage.append(allocator, &steps, step);
                metrics.recordStep(.schedule_node, false);
                step_mod.recordDotGroupCoverage(&metrics, group_plan.node_indices.len, step.kernel_name);
            },
        }
    }
    if (steps.items.len == 0) return null;

    return .{
        .value_specs = value_specs,
        .input_values = input_values,
        .output_values = output_values,
        .constant_instruction_indices = constant_instruction_indices,
        .steps = try steps.toOwnedSlice(allocator),
        .metrics = metrics,
    };
}

fn deinitRequest(self: *Request, allocator: std.mem.Allocator) void {
    step_storage.StepStorage.deinitSteps(allocator, self.steps);
    allocator.free(self.steps);
    allocator.free(self.constant_instruction_indices);
    allocator.free(self.output_values);
    allocator.free(self.input_values);
    allocator.free(self.value_specs);
    self.* = undefined;
}

const ProgramRequestInputs = struct {
    fn collect(allocator: std.mem.Allocator, request: Request, source: resident_inputs.Source, arguments: []const types.BufferHandle) types.Error![]types.BufferHandle {
        if (arguments.len + request.constant_instruction_indices.len != request.input_values.len) return error.CommandSubmissionFailed;
        const inputs = try allocator.alloc(types.BufferHandle, request.input_values.len);
        errdefer allocator.free(inputs);
        @memcpy(inputs[0..arguments.len], arguments);
        for (request.constant_instruction_indices, 0..) |instruction_index, constant_index| {
            inputs[arguments.len + constant_index] = try source.instructionConstant(instruction_index);
        }
        return inputs;
    }
};
