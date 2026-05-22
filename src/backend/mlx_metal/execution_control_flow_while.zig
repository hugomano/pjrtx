const ir = @import("src/compiler/ir");

const buffer_mod = @import("buffer.zig");
const executable_mod = @import("executable.zig");
const liveness_mod = @import("execution_liveness.zig");
const lowering_mod = @import("lowering.zig");
const program_mod = @import("program.zig");
const types = @import("execution_types.zig");
const values_mod = @import("execution_values.zig");

const BufferHandle = types.BufferHandle;
const Error = types.Error;
const CompiledExecutable = executable_mod.Executable;
const ValueBindings = values_mod.ValueBindings;
const WhilePatternOperand = lowering_mod.WhilePatternOperand;

const WhileOperandHandle = struct {
    handle: BufferHandle,
    owned: bool = false,
};

/// Executes recognized device-side while patterns without host predicate reads.
pub const WhileLoopDispatch = struct {
    executable: *CompiledExecutable,
    device_index: usize,
    values: *ValueBindings,
    release_inputs: bool,

    /// Runs one while control-flow node and stores its state outputs.
    pub fn run(self: WhileLoopDispatch, node: program_mod.Node, instruction: ir.PlanInstruction, control_flow_index: usize, control_flow: program_mod.ControlFlow) Error!?void {
        if (instruction.kind != .while_) return error.CommandSubmissionFailed;
        const executable = self.executable;
        if (control_flow.condition_subprogram >= executable.program.subprograms.len or control_flow.body_subprogram >= executable.program.subprograms.len) return error.CommandSubmissionFailed;
        const body = executable.program.subprograms[control_flow.body_subprogram];
        const pattern = lowering_mod.matchWhileF32LtAddPattern(executable.program.subprograms[control_flow.condition_subprogram], body) orelse return null;
        if (instruction.inputs.len != pattern.state_count or instruction.outputs.len != pattern.state_count or pattern.state_index >= instruction.inputs.len) return null;

        const state = try self.valueHandle(instruction.inputs[pattern.state_index]);
        const output_id = instruction.outputs[pattern.state_index];
        if (output_id.index >= executable.plan.values.len) return error.CommandSubmissionFailed;
        const limit = try self.operandHandle(instruction, control_flow_index, pattern.limit, 0);
        const step = try self.stepHandle(instruction, control_flow_index, body, pattern.step);
        defer if (step.owned) buffer_mod.Opaque.destroy(step.handle);
        const next = (try buffer_mod.Opaque.whileF32CompareAdd(
            state,
            limit,
            step.handle,
            pattern.compare_direction,
            pattern.update_op,
            executable.plan.values[output_id.index].descriptor.dims,
            pattern.max_iterations,
        )) orelse return null;
        try values_mod.storeOwnedValueHandle(self.values.handles, self.values.owned, output_id, next);
        try self.cloneInvariantOutputs(instruction, pattern.state_count, pattern.state_index);
        self.releaseInputsIfNeeded(node);
        return {};
    }

    fn cloneInvariantOutputs(self: WhileLoopDispatch, instruction: ir.PlanInstruction, state_count: usize, state_index: usize) Error!void {
        for (0..state_count) |index| {
            if (index == state_index) continue;
            const cloned = (try buffer_mod.Opaque.clone(try self.valueHandle(instruction.inputs[index]))) orelse return error.CommandSubmissionFailed;
            try values_mod.storeOwnedValueHandle(self.values.handles, self.values.owned, instruction.outputs[index], cloned);
        }
    }

    fn releaseInputsIfNeeded(self: WhileLoopDispatch, node: program_mod.Node) void {
        if (!self.release_inputs) return;
        const released = (liveness_mod.LivenessRelease{ .program = &self.executable.program, .values = self.values }).afterNode(node.inputs, node.instruction_index);
        if (released != 0) self.executable.recordReleasedIntermediateValues(released);
    }

    fn operandHandle(self: WhileLoopDispatch, instruction: ir.PlanInstruction, control_flow_index: usize, value: ir.RegionValue, constant_slot: usize) Error!BufferHandle {
        return switch (value.role) {
            .constant => self.executable.while_constant_handles[executable_mod.whileConstantIndex(self.executable.program.control_flows.len, self.device_index, control_flow_index, constant_slot)] orelse error.CommandSubmissionFailed,
            .argument => try self.valueHandle(instruction.inputs[value.id.index]),
            else => error.CommandSubmissionFailed,
        };
    }

    fn stepHandle(self: WhileLoopDispatch, instruction: ir.PlanInstruction, control_flow_index: usize, body: program_mod.Subprogram, operand: WhilePatternOperand) Error!WhileOperandHandle {
        if (operand.producer_instruction_index == null) return .{ .handle = try self.operandHandle(instruction, control_flow_index, operand.value, 1) };
        const producer_index = operand.producer_instruction_index.?;
        if (producer_index >= body.instructions.len) return error.CommandSubmissionFailed;
        const handle = (try self.executeLoopInvariantInstruction(instruction, body, body.instructions[producer_index])) orelse return error.CommandSubmissionFailed;
        return .{ .handle = handle, .owned = true };
    }

    fn executeLoopInvariantInstruction(self: WhileLoopDispatch, parent_instruction: ir.PlanInstruction, subprogram: program_mod.Subprogram, instruction: ir.RegionInstruction) Error!?BufferHandle {
        if (instruction.outputs.len != 1 or instruction.result_descriptors.len != 1) return null;
        return switch (instruction.kind) {
            .add, .subtract, .multiply, .divide, .maximum, .minimum => blk: {
                if (instruction.inputs.len != 2) return null;
                const op = lowering_mod.executableBinaryOp(instruction.kind) orelse return null;
                const lhs = try self.loopInvariantOperandHandle(parent_instruction, subprogram, instruction.inputs[0]);
                const rhs = try self.loopInvariantOperandHandle(parent_instruction, subprogram, instruction.inputs[1]);
                break :blk (try buffer_mod.Opaque.binaryWithOutputDims(lhs, rhs, op, instruction.result_descriptors[0].dims)) orelse return null;
            },
            else => null,
        };
    }

    fn loopInvariantOperandHandle(self: WhileLoopDispatch, parent_instruction: ir.PlanInstruction, subprogram: program_mod.Subprogram, value_id: ir.RegionValueId) Error!BufferHandle {
        const value = lowering_mod.regionValueById(subprogram, value_id) orelse return error.CommandSubmissionFailed;
        if (value.role != .argument or value.id.index >= parent_instruction.inputs.len) return error.CommandSubmissionFailed;
        return self.valueHandle(parent_instruction.inputs[value.id.index]);
    }

    fn valueHandle(self: WhileLoopDispatch, value_id: ir.ValueId) Error!BufferHandle {
        if (value_id.index >= self.values.handles.len) return error.CommandSubmissionFailed;
        return self.values.handles[value_id.index] orelse error.CommandSubmissionFailed;
    }
};
