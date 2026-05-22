const std = @import("std");

const ir = @import("src/compiler/ir");
const map_rules = @import("executable_metal_graph_map_rules.zig");
const profiling = @import("profiling.zig");
const program_mod = @import("program.zig");

/// Reports whether a fusion group may lower as one executable Metal graph step.
pub fn groupEligible(plan: *const ir.ExecutablePlan, program: *const program_mod.Program, group: program_mod.FusionGroup) bool {
    if (profiling.metalCppExecutableFusionMinNodes()) |min_nodes| {
        return group.kind == .view_elementwise and group.node_indices.len >= min_nodes and
            FusionBoundaryPass.init(plan, program, group).allows();
    }
    if (!profiling.metalCppExecutableConservativeFusionEnabled()) return false;
    return ConservativeFusionPass.init(plan, program, group).allows();
}

/// Rejects fusion groups that would recompute expanded generated expressions.
pub const FusionBoundaryPass = struct {
    plan: *const ir.ExecutablePlan,
    program: *const program_mod.Program,
    group: program_mod.FusionGroup,

    pub fn init(plan: *const ir.ExecutablePlan, program: *const program_mod.Program, group: program_mod.FusionGroup) FusionBoundaryPass {
        return .{ .plan = plan, .program = program, .group = group };
    }

    pub fn allows(self: FusionBoundaryPass) bool {
        for (self.group.node_indices) |node_index| {
            if (node_index >= self.program.nodes.len) return false;
            const node = self.program.nodes[node_index];
            if (node.fusion_group != self.group.id or node.instruction_index >= self.plan.instructions.len) return false;
            if (self.expandsComputedExpression(self.plan.instructions[node.instruction_index])) return false;
        }
        return true;
    }

    fn expandsComputedExpression(self: FusionBoundaryPass, instruction: ir.PlanInstruction) bool {
        if (instruction.kind != .broadcast_in_dim or instruction.inputs.len != 1 or instruction.outputs.len != 1) return false;
        const input = self.valueDescriptor(instruction.inputs[0]) orelse return true;
        const output = self.valueDescriptor(instruction.outputs[0]) orelse return true;
        const input_count = denseElementCount(input);
        const output_count = denseElementCount(output);
        if (input_count == 0 or output_count == 0 or output_count <= input_count) return false;
        const producer_index = self.groupProducer(instruction.inputs[0]) orelse return false;
        const producer = self.program.nodes[producer_index];
        return switch (producer.kind) {
            .view, .structural => false,
            else => true,
        };
    }

    fn groupProducer(self: FusionBoundaryPass, value_id: ir.ValueId) ?usize {
        if (value_id.index >= self.program.values.len) return null;
        const producer_index = self.program.values[value_id.index].producer_node orelse return null;
        if (producer_index >= self.program.nodes.len) return null;
        const producer = self.program.nodes[producer_index];
        if (producer.fusion_group != self.group.id) return null;
        if (!containsNode(self.group.node_indices, producer_index)) return null;
        return producer_index;
    }

    fn valueDescriptor(self: FusionBoundaryPass, value_id: ir.ValueId) ?ir.BufferDescriptor {
        if (value_id.index >= self.plan.values.len) return null;
        return self.plan.values[value_id.index].descriptor;
    }
};

/// Accepts only conservative small same-shape fusion groups for executable Metal graph lowering.
pub const ConservativeFusionPass = struct {
    pub const min_node_count = 2;
    const max_node_count = 4;

    plan: *const ir.ExecutablePlan,
    program: *const program_mod.Program,
    group: program_mod.FusionGroup,

    pub fn init(plan: *const ir.ExecutablePlan, program: *const program_mod.Program, group: program_mod.FusionGroup) ConservativeFusionPass {
        return .{ .plan = plan, .program = program, .group = group };
    }

    pub fn allows(self: ConservativeFusionPass) bool {
        if (self.group.kind != .view_elementwise) return false;
        if (self.group.node_indices.len < min_node_count or self.group.node_indices.len > max_node_count) return false;
        if (self.group.output_values.len != 1) return false;
        if (!FusionBoundaryPass.init(self.plan, self.program, self.group).allows()) return false;

        const output = self.valueDescriptor(self.group.output_values[0]) orelse return false;
        if (!self.supportedDescriptor(output)) return false;
        const output_count = denseElementCount(output);
        if (output_count == 0) return false;

        for (self.group.node_indices) |node_index| {
            if (node_index >= self.program.nodes.len) return false;
            const node = self.program.nodes[node_index];
            if (node.fusion_group != self.group.id or node.subprograms.len != 0 or node.control_flow != null) return false;
            if (node.instruction_index >= self.plan.instructions.len) return false;
            const instruction = self.plan.instructions[node.instruction_index];
            if (!self.supportedInstruction(node, instruction, output_count)) return false;
        }
        return true;
    }

    fn supportedInstruction(self: ConservativeFusionPass, node: program_mod.Node, instruction: ir.PlanInstruction, output_count: usize) bool {
        if (instruction.outputs.len != 1) return false;
        const descriptor = self.valueDescriptor(instruction.outputs[0]) orelse return false;
        if (!self.supportedDescriptor(descriptor) or denseElementCount(descriptor) != output_count) return false;

        if (map_rules.kindFor(instruction.kind) != null) {
            return node.kind == .elementwise and self.sameShapeInputs(instruction, output_count);
        }
        return switch (instruction.kind) {
            .convert, .bitcast_convert, .copy_arg0, .reduce_precision => (node.kind == .view or node.kind == .elementwise or node.kind == .structural) and self.sameShapeInputs(instruction, output_count),
            .reshape => (node.kind == .view or node.kind == .structural) and self.sameShapeInputs(instruction, output_count),
            else => false,
        };
    }

    fn sameShapeInputs(self: ConservativeFusionPass, instruction: ir.PlanInstruction, output_count: usize) bool {
        for (instruction.inputs) |input_id| {
            const input = self.valueDescriptor(input_id) orelse return false;
            if (!self.supportedDescriptor(input)) return false;
            const input_count = denseElementCount(input);
            if (input_count != output_count and input_count != 1) return false;
        }
        return true;
    }

    fn supportedDescriptor(_: ConservativeFusionPass, descriptor: ir.BufferDescriptor) bool {
        if (descriptor.layout != .dense_row_major) return false;
        return switch (descriptor.element_type) {
            .pred, .bf16, .f16, .f32, .s32, .u32, .u64 => true,
            else => false,
        };
    }

    fn valueDescriptor(self: ConservativeFusionPass, value_id: ir.ValueId) ?ir.BufferDescriptor {
        if (value_id.index >= self.plan.values.len) return null;
        return self.plan.values[value_id.index].descriptor;
    }
};
fn denseElementCount(descriptor: ir.BufferDescriptor) usize {
    const byte_size = ir.denseByteSize(descriptor.element_type, descriptor.dims);
    const element_size = descriptor.element_type.byteSize();
    if (byte_size == 0 or element_size == 0) return 0;
    return byte_size / element_size;
}

fn containsNode(node_indices: []const usize, node_index: usize) bool {
    for (node_indices) |candidate| {
        if (candidate == node_index) return true;
    }
    return false;
}
