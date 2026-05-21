const std = @import("std");

const ir = @import("src/compiler/ir");
const program_mod = @import("program.zig");
const diagnostic = @import("lowering_diagnostic.zig");

const Issue = diagnostic.Issue;

const DefaultWhileMaxIterations: u64 = 1_000_000;

pub const WhilePatternOperand = struct {
    value: ir.RegionValue,
    producer_instruction_index: ?usize = null,
};

/// Recognized device-side while pattern used by MLX lowering and execution.
pub const WhileF32LtAddPattern = struct {
    limit: ir.RegionValue,
    step: WhilePatternOperand,
    state_index: usize = 0,
    compare_direction: ir.CompareOp = .lt,
    update_op: ir.ElementwiseBinaryOp = .add,
    state_count: usize = 1,
    max_iterations: u64 = DefaultWhileMaxIterations,
};

pub fn descriptorsEqual(a: ir.BufferDescriptor, b: ir.BufferDescriptor) bool {
    return a.element_type == b.element_type and std.mem.eql(i64, a.dims, b.dims);
}

/// Looks up a region value by id inside a cloned backend subprogram.
pub fn regionValueById(subprogram: program_mod.Subprogram, id: ir.RegionValueId) ?ir.RegionValue {
    if (id.index >= subprogram.values.len) return null;
    return subprogram.values[id.index];
}

fn regionValueIsArgumentIndex(subprogram: program_mod.Subprogram, id: ir.RegionValueId, index: usize) bool {
    const value = regionValueById(subprogram, id) orelse return false;
    return value.role == .argument and id.index == index;
}

fn constantCompatibleWithState(value: ir.RegionValue, state: ir.BufferDescriptor) bool {
    if (value.role != .constant or value.literal == null) return false;
    if (value.descriptor.element_type != state.element_type) return false;
    if (value.descriptor.element_type != .f32 and value.descriptor.element_type != .bf16) return false;
    if (value.descriptor.dims.len == 0) return true;
    return std.mem.eql(i64, value.descriptor.dims, state.dims);
}

fn whileOperandCompatibleWithState(value: ir.RegionValue, state: ir.BufferDescriptor) bool {
    if (value.role != .constant and value.role != .argument and value.role != .instruction_result) return false;
    if (value.role == .constant and value.literal == null) return false;
    if (value.descriptor.element_type != state.element_type) return false;
    if (value.descriptor.element_type != .f32 and value.descriptor.element_type != .bf16) return false;
    if (value.descriptor.dims.len == 0) return true;
    return std.mem.eql(i64, value.descriptor.dims, state.dims);
}

fn addInstructionStepOperand(subprogram: program_mod.Subprogram, instruction: ir.RegionInstruction, state: ir.BufferDescriptor, state_index: usize, update_instruction_index: usize) ?WhilePatternOperand {
    if (instruction.inputs.len != 2) return null;
    if (regionValueIsArgumentIndex(subprogram, instruction.inputs[0], state_index)) {
        const rhs = regionValueById(subprogram, instruction.inputs[1]) orelse return null;
        return whileStepOperandFromRegionValue(subprogram, rhs, state, state_index, update_instruction_index);
    }
    if (instruction.kind == .add and regionValueIsArgumentIndex(subprogram, instruction.inputs[1], state_index)) {
        const lhs = regionValueById(subprogram, instruction.inputs[0]) orelse return null;
        return whileStepOperandFromRegionValue(subprogram, lhs, state, state_index, update_instruction_index);
    }
    return null;
}

fn whileStepOperandFromRegionValue(subprogram: program_mod.Subprogram, value: ir.RegionValue, state: ir.BufferDescriptor, state_index: usize, update_instruction_index: usize) ?WhilePatternOperand {
    if (!whileOperandCompatibleWithState(value, state)) return null;
    switch (value.role) {
        .constant, .argument => return .{ .value = value },
        .instruction_result => {
            const producer_index = loopInvariantProducerInstructionIndex(subprogram, value.id, state, state_index, update_instruction_index) orelse return null;
            return .{ .value = value, .producer_instruction_index = producer_index };
        },
    }
}

fn loopInvariantProducerInstructionIndex(subprogram: program_mod.Subprogram, output_id: ir.RegionValueId, state: ir.BufferDescriptor, state_index: usize, update_instruction_index: usize) ?usize {
    var instruction_index: usize = 0;
    while (instruction_index < update_instruction_index) : (instruction_index += 1) {
        const instruction = subprogram.instructions[instruction_index];
        if (instruction.outputs.len != 1 or instruction.outputs[0].index != output_id.index) continue;
        if (instruction.result_descriptors.len != 1 or !descriptorsEqual(instruction.result_descriptors[0], state)) return null;
        if (!regionBinaryOpSupported(instruction.kind)) return null;
        if (instruction.inputs.len != 2) return null;
        for (instruction.inputs) |input_id| {
            const input = regionValueById(subprogram, input_id) orelse return null;
            if (!whileOperandCompatibleWithState(input, state)) return null;
            if (input.role != .argument) return null;
            if (input.id.index == state_index) return null;
        }
        return instruction_index;
    }
    return null;
}

fn regionBinaryOpSupported(kind: ir.PlanInstructionKind) bool {
    return switch (kind) {
        .add,
        .subtract,
        .multiply,
        .divide,
        .maximum,
        .minimum,
        .power,
        .atan2,
        .remainder,
        .and_,
        .or_,
        .xor,
        .shift_left,
        .shift_right_arithmetic,
        .shift_right_logical,
        => true,
        else => false,
    };
}

fn compareDirectionSupportedInWhile(direction: ir.CompareOp) bool {
    return switch (direction) {
        .lt, .le, .gt, .ge => true,
        .eq, .ne => false,
    };
}

/// Validates while region contracts and the supported device-side loop pattern.
pub fn validateWhile(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
    if (instruction.inputs.len == 0 or instruction.outputs.len != instruction.inputs.len or instruction.region_ids.len != 2) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "while lowering requires state inputs, matching state outputs, and cond/body regions",
        .feature = "mlx-while-region-contract",
    };
    const cond_id = instruction.region_ids[0];
    const body_id = instruction.region_ids[1];
    if (cond_id.index >= plan.regions.len or body_id.index >= plan.regions.len) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "while region ids must reference captured PjRTx region summaries",
        .feature = "mlx-while-region-contract",
    };
    const cond = plan.regions[cond_id.index];
    const body = plan.regions[body_id.index];
    if (cond.kind != .while_cond or body.kind != .while_body) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "while lowering requires cond region followed by body region",
        .feature = "mlx-while-region-contract",
    };
    if (cond.argument_descriptors.len != instruction.inputs.len or body.argument_descriptors.len != instruction.inputs.len) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "while cond/body region arguments must match the loop state arity",
        .feature = "mlx-while-region-contract",
    };
    if (cond.return_descriptors.len != 1 or cond.return_descriptors[0].element_type != .pred or cond.return_descriptors[0].dims.len != 0) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "while cond region must return a scalar predicate",
        .feature = "mlx-while-region-contract",
    };
    if (body.return_descriptors.len != instruction.outputs.len) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "while body region return arity must match the loop state arity",
        .feature = "mlx-while-region-contract",
    };
    for (instruction.inputs, instruction.outputs, 0..) |input_id, state_output_id, state_index| {
        if (input_id.index >= plan.values.len or state_output_id.index >= plan.values.len) return .{
            .instruction_index = instruction_index,
            .value_id = state_output_id,
            .op = instruction.kind,
            .detail = "while state values must reference executable value descriptors",
            .feature = "mlx-executable-values",
        };
        const input = plan.values[input_id.index].descriptor;
        const output = plan.values[state_output_id.index].descriptor;
        if (!descriptorsEqual(input, output) or
            !descriptorsEqual(input, cond.argument_descriptors[state_index]) or
            !descriptorsEqual(input, body.argument_descriptors[state_index]) or
            !descriptorsEqual(input, body.return_descriptors[state_index]))
        {
            return .{
                .instruction_index = instruction_index,
                .value_id = state_output_id,
                .op = instruction.kind,
                .detail = "while loop state descriptors must be invariant across inputs, outputs, body args, and body returns",
                .feature = "mlx-while-region-contract",
            };
        }
    }
    const matched = matchWhileF32LtAddPattern(.{
        .id = 0,
        .parent_node = instruction_index,
        .region_id = cond.id,
        .kind = cond.kind,
        .values = cond.values,
        .argument_descriptors = cond.argument_descriptors,
        .instructions = cond.instructions,
        .return_descriptors = cond.return_descriptors,
        .terminator_operands = cond.terminator_operands,
        .terminator_operand_descriptors = cond.terminator_operand_descriptors,
    }, .{
        .id = 1,
        .parent_node = instruction_index,
        .region_id = body.id,
        .kind = body.kind,
        .values = body.values,
        .argument_descriptors = body.argument_descriptors,
        .instructions = body.instructions,
        .return_descriptors = body.return_descriptors,
        .terminator_operands = body.terminator_operands,
        .terminator_operand_descriptors = body.terminator_operand_descriptors,
    }) != null;
    if (matched) return null;
    return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "while region subprogram lowering requires a supported device-side loop pattern; host-loop execution is disabled",
        .feature = "mlx-while-region-pattern",
    };
}

/// Recognizes the currently supported device-side while compare/add pattern.
pub fn matchWhileF32LtAddPattern(cond: program_mod.Subprogram, body: program_mod.Subprogram) ?WhileF32LtAddPattern {
    if (cond.kind != .while_cond or body.kind != .while_body) return null;
    if (cond.argument_descriptors.len == 0 or body.argument_descriptors.len != cond.argument_descriptors.len) return null;
    if (cond.instructions.len != 1 or body.instructions.len == 0 or body.instructions.len > 3) return null;
    if (cond.terminator_operands.len != 1 or body.terminator_operands.len != body.argument_descriptors.len) return null;
    for (cond.argument_descriptors, 0..) |descriptor, argument_index| {
        if (!descriptorsEqual(descriptor, body.argument_descriptors[argument_index])) return null;
    }

    const compare_instruction = cond.instructions[0];
    if (compare_instruction.kind != .compare) return null;
    const compare_direction = compare_instruction.compare_direction orelse return null;
    if (!compareDirectionSupportedInWhile(compare_direction)) return null;
    if (compare_instruction.inputs.len != 2 or compare_instruction.outputs.len != 1) return null;
    if (compare_instruction.outputs[0].index != cond.terminator_operands[0].index) return null;

    var loop_state_index: ?usize = null;
    var argument_index: usize = 0;
    while (argument_index < cond.argument_descriptors.len) : (argument_index += 1) {
        if (regionValueIsArgumentIndex(cond, compare_instruction.inputs[0], argument_index)) {
            loop_state_index = argument_index;
            break;
        }
    }
    const state_index = loop_state_index orelse return null;
    const state = cond.argument_descriptors[state_index];
    if (state.element_type != .f32 and state.element_type != .bf16) return null;
    const limit = regionValueById(cond, compare_instruction.inputs[1]) orelse return null;
    if (!whileOperandCompatibleWithState(limit, state)) return null;

    var update_instruction_index: ?usize = null;
    var body_instruction_index: usize = 0;
    while (body_instruction_index < body.instructions.len) : (body_instruction_index += 1) {
        const candidate = body.instructions[body_instruction_index];
        if (candidate.kind != .add and candidate.kind != .subtract) continue;
        if (candidate.outputs.len != 1) continue;
        if (addInstructionStepOperand(body, candidate, state, state_index, body_instruction_index) == null) continue;
        update_instruction_index = body_instruction_index;
        break;
    }
    const update_index = update_instruction_index orelse return null;
    const update_instruction = body.instructions[update_index];
    if ((update_instruction.kind != .add and update_instruction.kind != .subtract) or update_instruction.outputs.len != 1) return null;
    const step = addInstructionStepOperand(body, update_instruction, state, state_index, update_index) orelse return null;
    if (update_index + 1 == body.instructions.len) {
        if (update_instruction.outputs[0].index != body.terminator_operands[state_index].index) return null;
    } else {
        if (update_index + 2 != body.instructions.len) return null;
        const cast_instruction = body.instructions[update_index + 1];
        if (cast_instruction.kind != .convert or cast_instruction.inputs.len != 1 or cast_instruction.outputs.len != 1) return null;
        if (cast_instruction.inputs[0].index != update_instruction.outputs[0].index) return null;
        if (cast_instruction.outputs[0].index != body.terminator_operands[state_index].index) return null;
        if (cast_instruction.result_descriptors.len != 1 or !descriptorsEqual(cast_instruction.result_descriptors[0], state)) return null;
    }
    var invariant_index: usize = 0;
    while (invariant_index < cond.argument_descriptors.len) : (invariant_index += 1) {
        if (invariant_index == state_index) continue;
        if (!regionValueIsArgumentIndex(body, body.terminator_operands[invariant_index], invariant_index)) return null;
    }
    const update_op: ir.ElementwiseBinaryOp = if (update_instruction.kind == .subtract) .subtract else .add;
    return .{ .limit = limit, .step = step, .state_index = state_index, .compare_direction = compare_direction, .update_op = update_op, .state_count = cond.argument_descriptors.len };
}
