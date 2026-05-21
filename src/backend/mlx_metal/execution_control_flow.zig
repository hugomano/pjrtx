const std = @import("std");
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

/// Executes backend control-flow nodes using device-side MLX primitives only.
pub const ControlFlowDispatch = struct {
    executable: *CompiledExecutable,
    device_index: usize,
    values: *ValueBindings,
    release_inputs: bool,

    /// Executes one scheduled control-flow node and stores its output handles.
    pub fn run(self: ControlFlowDispatch, node: program_mod.Node, instruction: ir.PlanInstruction) Error!?void {
        const executable = self.executable;
        const value_handles = self.values.handles;
        const control_flow_index = node.control_flow orelse return error.CommandSubmissionFailed;
        if (control_flow_index >= executable.program.control_flows.len) return error.CommandSubmissionFailed;
        const control_flow = executable.program.control_flows[control_flow_index];
        switch (control_flow.kind) {
            .while_loop => {
                if (instruction.kind != .while_) return error.CommandSubmissionFailed;
                if (control_flow.condition_subprogram >= executable.program.subprograms.len or
                    control_flow.body_subprogram >= executable.program.subprograms.len)
                {
                    return error.CommandSubmissionFailed;
                }
                const pattern = lowering_mod.matchWhileF32LtAddPattern(
                    executable.program.subprograms[control_flow.condition_subprogram],
                    executable.program.subprograms[control_flow.body_subprogram],
                ) orelse return null;
                if (instruction.inputs.len != pattern.state_count or instruction.outputs.len != pattern.state_count) return null;
                if (pattern.state_index >= instruction.inputs.len) return error.CommandSubmissionFailed;
                const state_id = instruction.inputs[pattern.state_index];
                if (state_id.index >= value_handles.len) return error.CommandSubmissionFailed;
                const state = value_handles[state_id.index] orelse return error.CommandSubmissionFailed;
                const output_id = instruction.outputs[pattern.state_index];
                if (output_id.index >= executable.plan.values.len) return error.CommandSubmissionFailed;
                const output_descriptor = executable.plan.values[output_id.index].descriptor;
                const limit = try self.whilePatternOperandHandle(instruction, control_flow_index, pattern.limit, 0);
                const step = try self.whileStepOperandHandle(
                    instruction,
                    control_flow_index,
                    executable.program.subprograms[control_flow.body_subprogram],
                    pattern.step,
                );
                defer if (step.owned) buffer_mod.Opaque.destroy(step.handle);
                const next = (try buffer_mod.Opaque.whileF32CompareAdd(
                    state,
                    limit,
                    step.handle,
                    pattern.compare_direction,
                    pattern.update_op,
                    output_descriptor.dims,
                    pattern.max_iterations,
                )) orelse return null;
                try values_mod.storeOwnedValueHandle(value_handles, self.values.owned, output_id, next);
                var invariant_index: usize = 1;
                invariant_index = 0;
                while (invariant_index < pattern.state_count) : (invariant_index += 1) {
                    if (invariant_index == pattern.state_index) continue;
                    const input_id = instruction.inputs[invariant_index];
                    const invariant_output_id = instruction.outputs[invariant_index];
                    if (input_id.index >= value_handles.len) return error.CommandSubmissionFailed;
                    const input = value_handles[input_id.index] orelse return error.CommandSubmissionFailed;
                    const cloned = (try buffer_mod.Opaque.clone(input)) orelse return null;
                    try values_mod.storeOwnedValueHandle(value_handles, self.values.owned, invariant_output_id, cloned);
                }
                if (self.release_inputs) {
                    const released = (liveness_mod.LivenessRelease{ .program = &executable.program, .values = self.values }).afterNode(node.inputs, node.instruction_index);
                    if (released != 0) executable.recordReleasedIntermediateValues(released);
                }
                return {};
            },
        }
    }

    fn whilePatternOperandHandle(
        self: ControlFlowDispatch,
        instruction: ir.PlanInstruction,
        control_flow_index: usize,
        value: ir.RegionValue,
        constant_slot: usize,
    ) Error!BufferHandle {
        const executable = self.executable;
        const value_handles = self.values.handles;
        return switch (value.role) {
            .constant => executable.while_constant_handles[executable_mod.whileConstantIndex(executable.program.control_flows.len, self.device_index, control_flow_index, constant_slot)] orelse error.CommandSubmissionFailed,
            .argument => blk: {
                const argument_index: usize = value.id.index;
                if (argument_index >= instruction.inputs.len) return error.CommandSubmissionFailed;
                const input_id = instruction.inputs[argument_index];
                if (input_id.index >= value_handles.len) return error.CommandSubmissionFailed;
                break :blk value_handles[input_id.index] orelse error.CommandSubmissionFailed;
            },
            else => error.CommandSubmissionFailed,
        };
    }

    fn whileStepOperandHandle(
        self: ControlFlowDispatch,
        instruction: ir.PlanInstruction,
        control_flow_index: usize,
        body: program_mod.Subprogram,
        operand: WhilePatternOperand,
    ) Error!WhileOperandHandle {
        if (operand.producer_instruction_index == null) {
            return .{
                .handle = try self.whilePatternOperandHandle(instruction, control_flow_index, operand.value, 1),
                .owned = false,
            };
        }
        const producer_index = operand.producer_instruction_index.?;
        if (producer_index >= body.instructions.len) return error.CommandSubmissionFailed;
        const producer = body.instructions[producer_index];
        const handle = (try self.executeLoopInvariantRegionInstruction(
            instruction,
            body,
            producer,
        )) orelse return error.CommandSubmissionFailed;
        return .{ .handle = handle, .owned = true };
    }

    fn executeLoopInvariantRegionInstruction(
        self: ControlFlowDispatch,
        parent_instruction: ir.PlanInstruction,
        subprogram: program_mod.Subprogram,
        instruction: ir.RegionInstruction,
    ) Error!?BufferHandle {
        if (instruction.outputs.len != 1 or instruction.result_descriptors.len != 1) return null;
        const output_dims = instruction.result_descriptors[0].dims;
        switch (instruction.kind) {
            .add, .subtract, .multiply, .divide, .maximum, .minimum => {
                if (instruction.inputs.len != 2) return null;
                const op = lowering_mod.executableBinaryOp(instruction.kind) orelse return null;
                const lhs = try self.loopInvariantRegionOperandHandle(parent_instruction, subprogram, instruction.inputs[0]);
                const rhs = try self.loopInvariantRegionOperandHandle(parent_instruction, subprogram, instruction.inputs[1]);
                return (try buffer_mod.Opaque.binaryWithOutputDims(lhs, rhs, op, output_dims)) orelse return null;
            },
            else => return null,
        }
    }

    fn loopInvariantRegionOperandHandle(
        self: ControlFlowDispatch,
        parent_instruction: ir.PlanInstruction,
        subprogram: program_mod.Subprogram,
        value_id: ir.RegionValueId,
    ) Error!BufferHandle {
        const value_handles = self.values.handles;
        const value = lowering_mod.regionValueById(subprogram, value_id) orelse return error.CommandSubmissionFailed;
        if (value.role != .argument) return error.CommandSubmissionFailed;
        const argument_index: usize = value.id.index;
        if (argument_index >= parent_instruction.inputs.len) return error.CommandSubmissionFailed;
        const input_id = parent_instruction.inputs[argument_index];
        if (input_id.index >= value_handles.len) return error.CommandSubmissionFailed;
        return value_handles[input_id.index] orelse error.CommandSubmissionFailed;
    }
};

test "mlx metal backend executes f32 lt/add while loop on device" {
    const execution_mod = @import("execution.zig");
    const allocator = std.testing.allocator;
    const local_hardware_id: i32 = 0;
    const assignment = [_]i32{0};

    const scalar_f32 = ir.BufferDescriptor{
        .element_type = .f32,
        .dims = &.{},
        .device_id = 0,
        .memory_id = 0,
        .shard_index = 0,
    };
    const scalar_pred = ir.BufferDescriptor{
        .element_type = .pred,
        .dims = &.{},
        .device_id = 0,
        .memory_id = 0,
        .shard_index = 0,
    };

    const values = try allocator.alloc(ir.Value, 2);
    errdefer allocator.free(values);
    values[0] = .{ .id = .{ .index = 0 }, .role = .parameter, .descriptor = scalar_f32 };
    values[1] = .{ .id = .{ .index = 1 }, .role = .instruction_result, .descriptor = scalar_f32 };

    var parameter_shardings = try allocator.alloc(ir.ShardingPlan, 1);
    parameter_shardings[0] = .{
        .kind = .replicated,
        .mesh_name = try allocator.dupe(u8, ""),
        .device_assignment = try allocator.dupe(i32, &assignment),
    };
    var output_shardings = try allocator.alloc(ir.ShardingPlan, 1);
    output_shardings[0] = .{
        .kind = .replicated,
        .mesh_name = try allocator.dupe(u8, ""),
        .device_assignment = try allocator.dupe(i32, &assignment),
    };

    var limit_value: f32 = 4.0;
    var step_value: f32 = 1.0;
    const cond_values = try allocator.dupe(ir.RegionValue, &.{
        .{ .id = .{ .index = 0 }, .role = .argument, .descriptor = scalar_f32 },
        .{ .id = .{ .index = 1 }, .role = .constant, .descriptor = scalar_f32, .literal = try allocator.dupe(u8, std.mem.asBytes(&limit_value)) },
        .{ .id = .{ .index = 2 }, .role = .instruction_result, .descriptor = scalar_pred },
    });
    const body_values = try allocator.dupe(ir.RegionValue, &.{
        .{ .id = .{ .index = 0 }, .role = .argument, .descriptor = scalar_f32 },
        .{ .id = .{ .index = 1 }, .role = .constant, .descriptor = scalar_f32, .literal = try allocator.dupe(u8, std.mem.asBytes(&step_value)) },
        .{ .id = .{ .index = 2 }, .role = .instruction_result, .descriptor = scalar_f32 },
    });

    var regions = try allocator.alloc(ir.PlanRegion, 2);
    regions[0] = .{
        .id = .{ .index = 0 },
        .parent_instruction_index = 0,
        .kind = .while_cond,
        .values = cond_values,
        .argument_descriptors = try allocator.dupe(ir.BufferDescriptor, &.{scalar_f32}),
        .instructions = try allocator.dupe(ir.RegionInstruction, &.{.{
            .kind = .compare,
            .inputs = try allocator.dupe(ir.RegionValueId, &.{ .{ .index = 0 }, .{ .index = 1 } }),
            .outputs = try allocator.dupe(ir.RegionValueId, &.{.{ .index = 2 }}),
            .operand_descriptors = try allocator.dupe(ir.BufferDescriptor, &.{ scalar_f32, scalar_f32 }),
            .result_descriptors = try allocator.dupe(ir.BufferDescriptor, &.{scalar_pred}),
            .compare_direction = .lt,
        }}),
        .return_descriptors = try allocator.dupe(ir.BufferDescriptor, &.{scalar_pred}),
        .terminator_operands = try allocator.dupe(ir.RegionValueId, &.{.{ .index = 2 }}),
        .terminator_operand_descriptors = try allocator.dupe(ir.BufferDescriptor, &.{scalar_pred}),
    };
    regions[1] = .{
        .id = .{ .index = 1 },
        .parent_instruction_index = 0,
        .kind = .while_body,
        .values = body_values,
        .argument_descriptors = try allocator.dupe(ir.BufferDescriptor, &.{scalar_f32}),
        .instructions = try allocator.dupe(ir.RegionInstruction, &.{.{
            .kind = .add,
            .inputs = try allocator.dupe(ir.RegionValueId, &.{ .{ .index = 0 }, .{ .index = 1 } }),
            .outputs = try allocator.dupe(ir.RegionValueId, &.{.{ .index = 2 }}),
            .operand_descriptors = try allocator.dupe(ir.BufferDescriptor, &.{ scalar_f32, scalar_f32 }),
            .result_descriptors = try allocator.dupe(ir.BufferDescriptor, &.{scalar_f32}),
        }}),
        .return_descriptors = try allocator.dupe(ir.BufferDescriptor, &.{scalar_f32}),
        .terminator_operands = try allocator.dupe(ir.RegionValueId, &.{.{ .index = 2 }}),
        .terminator_operand_descriptors = try allocator.dupe(ir.BufferDescriptor, &.{scalar_f32}),
    };

    var plan = ir.ExecutablePlan{
        .allocator = allocator,
        .options = .{
            .num_replicas = 1,
            .num_partitions = 1,
            .device_assignment = try allocator.dupe(i32, &assignment),
        },
        .values = values,
        .regions = regions,
        .parameter_shardings = parameter_shardings,
        .output_shardings = output_shardings,
        .output_ids = try allocator.dupe(ir.ValueId, &.{.{ .index = 1 }}),
        .instructions = try allocator.dupe(ir.PlanInstruction, &.{.{
            .kind = .while_,
            .inputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 0 }}),
            .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 1 }}),
            .region_ids = try allocator.dupe(ir.RegionId, &.{ .{ .index = 0 }, .{ .index = 1 } }),
        }}),
    };
    defer plan.deinit();

    const executable = (try executable_mod.compile(allocator, &plan, &.{local_hardware_id}, execution_mod.compiledProgramBuildCallback)) orelse return error.TestUnexpectedResult;
    defer execution_mod.destroy(executable);
    var state: f32 = 0.0;
    const state_buffer = (try buffer_mod.Buffer.fromHost(local_hardware_id, .f32, &.{}, std.mem.asBytes(&state))) orelse return error.TestUnexpectedResult;
    defer state_buffer.destroy();
    const result = (try execution_mod.execute(allocator, executable, 0, &.{state_buffer.toHandle()})) orelse return error.TestUnexpectedResult;
    defer {
        for (result.outputs) |output| buffer_mod.Opaque.destroy(output.handle);
        allocator.free(result.outputs);
    }
    var actual: f32 = 0.0;
    try buffer_mod.Opaque.copyToHost(result.outputs[0].handle, std.mem.asBytes(&actual));
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), actual, 0.0001);
}
