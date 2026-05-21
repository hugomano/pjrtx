const std = @import("std");

const ir = @import("src/compiler/ir");

const buffer_mod = @import("buffer.zig");
const diagnostic = @import("lowering_diagnostic.zig");
const shapes = @import("lowering_shapes.zig");

const Issue = diagnostic.Issue;
const dimsEqual = shapes.dimsEqual;
const inputDescriptor = shapes.inputDescriptor;
const isSupportedFloat = shapes.isSupportedFloat;

/// Validates bitcast_convert descriptors while preserving dense byte size.
pub fn validateBitcastConvert(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
    if (instruction.inputs.len != 1 or instruction.outputs.len != 1) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "bitcast_convert lowering requires exactly one input and one output",
        .feature = "mlx-bitcast",
    };
    const input = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[0],
        .op = instruction.kind,
        .detail = "bitcast_convert input is outside the executable value table",
        .feature = "mlx-bitcast",
    };
    const output = plan.values[output_id.index].descriptor;
    if (!buffer_mod.supportsElementType(input.element_type) or !buffer_mod.supportsElementType(output.element_type)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "bitcast_convert lowering requires MLX-supported input and output dtypes",
        .feature = "mlx-bitcast-dtype",
    };
    if (ir.denseByteSize(input.element_type, input.dims) != ir.denseByteSize(output.element_type, output.dims)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "bitcast_convert must preserve dense byte size",
        .feature = "mlx-bitcast-shape",
    };
    return null;
}

/// Validates partition_id output shape and dtype supported by MLX lowering.
pub fn validatePartitionId(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
    const output = plan.values[output_id.index].descriptor;
    if (instruction.inputs.len != 0 or instruction.outputs.len != 1) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "partition_id lowering requires no inputs and exactly one output",
        .feature = "mlx-partition-id-arity",
    };
    if (output.dims.len != 0) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "partition_id lowering requires a scalar output",
        .feature = "mlx-partition-id-shape",
    };
    if (output.element_type != .u32 and output.element_type != .s32) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "partition_id lowering supports u32 or s32 scalar outputs",
        .feature = "mlx-partition-id-dtype",
    };
    return null;
}

/// Validates StableHLO rng metadata and descriptors supported by MLX lowering.
pub fn validateRng(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
    if (instruction.inputs.len != 2 or instruction.outputs.len != 1) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "rng lowering requires low/scale inputs and one output",
        .feature = "mlx-rng",
    };
    if (instruction.rng_distribution == null) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "rng lowering requires rng_distribution metadata",
        .feature = "mlx-rng-distribution",
    };
    const a = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[0],
        .op = instruction.kind,
        .detail = "rng first input is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const b = inputDescriptor(plan, instruction, 1) orelse return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[1],
        .op = instruction.kind,
        .detail = "rng second input is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const output = plan.values[output_id.index].descriptor;
    if (a.dims.len != 0 or b.dims.len != 0) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "rng lowering currently requires scalar distribution parameters",
        .feature = "mlx-rng-params",
    };
    if (a.element_type != b.element_type or a.element_type != output.element_type) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "rng distribution parameters and output must use the same dtype",
        .feature = "mlx-rng-dtype",
    };
    if (!isSupportedFloat(output.element_type)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "rng lowering supports MLX floating output dtypes only",
        .feature = "mlx-rng-dtype",
    };
    return null;
}

/// Validates optimization_barrier tuple-like passthrough descriptors.
pub fn validateOptimizationBarrier(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize) ?Issue {
    if (instruction.inputs.len == 0 or instruction.inputs.len != instruction.outputs.len) return .{
        .instruction_index = instruction_index,
        .op = instruction.kind,
        .detail = "optimization_barrier lowering requires one output for each input",
        .feature = "mlx-optimization-barrier",
    };
    for (instruction.inputs, instruction.outputs) |input_id, output_id| {
        if (input_id.index >= plan.values.len) return .{
            .instruction_index = instruction_index,
            .value_id = input_id,
            .op = instruction.kind,
            .detail = "optimization_barrier input is outside the executable value table",
            .feature = "mlx-optimization-barrier",
        };
        if (output_id.index >= plan.values.len) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "optimization_barrier output is outside the executable value table",
            .feature = "mlx-optimization-barrier",
        };
        const input = plan.values[input_id.index].descriptor;
        const output = plan.values[output_id.index].descriptor;
        if (input.element_type != output.element_type or !dimsEqual(input.dims, output.dims)) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "optimization_barrier output must match corresponding input dtype and shape",
            .feature = "mlx-optimization-barrier",
        };
    }
    return null;
}

/// Validates tuple structural values captured in the executable plan.
pub fn validateTuple(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
    const output = plan.values[output_id.index];
    if (output.storage != .tuple) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "tuple lowering requires a tuple storage output value",
        .feature = "mlx-tuple-structural",
    };
    if (output.elements.len != instruction.inputs.len) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "tuple output element list must match tuple operands",
        .feature = "mlx-tuple-structural",
    };
    for (output.elements, instruction.inputs) |element_id, input_id| {
        if (element_id.index >= plan.values.len or input_id.index >= plan.values.len or element_id.index != input_id.index) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "tuple output element list must reference tuple operands in order",
            .feature = "mlx-tuple-structural",
        };
    }
    return null;
}

/// Validates get_tuple_element structural extraction from plan tuples.
pub fn validateGetTupleElement(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
    if (instruction.inputs.len != 1 or instruction.outputs.len != 1) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "get_tuple_element lowering requires exactly one tuple input and one output",
        .feature = "mlx-get-tuple-element",
    };
    const tuple_index = instruction.tuple_index orelse return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "get_tuple_element lowering requires a tuple index",
        .feature = "mlx-get-tuple-element",
    };
    if (tuple_index < 0) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "get_tuple_element tuple index must be non-negative",
        .feature = "mlx-get-tuple-element",
    };
    const tuple_id = instruction.inputs[0];
    if (tuple_id.index >= plan.values.len) return .{
        .instruction_index = instruction_index,
        .value_id = tuple_id,
        .op = instruction.kind,
        .detail = "get_tuple_element input is outside the executable value table",
        .feature = "mlx-get-tuple-element",
    };
    const tuple_value = plan.values[tuple_id.index];
    if (tuple_value.storage != .tuple or tuple_index >= @as(i64, @intCast(tuple_value.elements.len))) return .{
        .instruction_index = instruction_index,
        .value_id = tuple_id,
        .op = instruction.kind,
        .detail = "get_tuple_element input must be a tuple with the requested element",
        .feature = "mlx-get-tuple-element",
    };
    const element_id = tuple_value.elements[@intCast(tuple_index)];
    if (element_id.index >= plan.values.len) return .{
        .instruction_index = instruction_index,
        .value_id = element_id,
        .op = instruction.kind,
        .detail = "get_tuple_element selected element is outside the executable value table",
        .feature = "mlx-get-tuple-element",
    };
    if (plan.values[output_id.index].storage != .tensor or plan.values[element_id.index].storage != .tensor) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "get_tuple_element currently lowers tensor tuple elements only",
        .feature = "mlx-get-tuple-element",
    };
    const element = plan.values[element_id.index].descriptor;
    const output = plan.values[output_id.index].descriptor;
    if (element.element_type != output.element_type or !dimsEqual(element.dims, output.dims)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "get_tuple_element output descriptor must match the selected tuple element",
        .feature = "mlx-get-tuple-element",
    };
    return null;
}

/// Validates rng_bit_generator state and output descriptors.
pub fn validateRngBitGenerator(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
    if (instruction.inputs.len != 1 or instruction.outputs.len != 2) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "rng_bit_generator lowering requires one state input and state/bits outputs",
        .feature = "mlx-rng-arity",
    };
    const state = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[0],
        .op = instruction.kind,
        .detail = "rng_bit_generator state input is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const state_output = plan.values[instruction.outputs[0].index].descriptor;
    const bits_output = plan.values[instruction.outputs[1].index].descriptor;
    if (!std.mem.eql(i64, state.dims, &.{2}) or (state.element_type != .u32 and state.element_type != .u64)) return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[0],
        .op = instruction.kind,
        .detail = "rng_bit_generator state must be u32[2] or u64[2]",
        .feature = "mlx-rng-state",
    };
    if (state.element_type != state_output.element_type or !std.mem.eql(i64, state.dims, state_output.dims)) return .{
        .instruction_index = instruction_index,
        .value_id = instruction.outputs[0],
        .op = instruction.kind,
        .detail = "rng_bit_generator state output must preserve state dtype and shape",
        .feature = "mlx-rng-state",
    };
    if (bits_output.element_type != .u8 and bits_output.element_type != .u16 and bits_output.element_type != .u32 and bits_output.element_type != .u64) return .{
        .instruction_index = instruction_index,
        .value_id = instruction.outputs[1],
        .op = instruction.kind,
        .detail = "rng_bit_generator bits output supports u8/u16/u32/u64 only",
        .feature = "mlx-rng-bits",
    };
    if (bits_output.element_type == .u64 and state.element_type != .u64) return .{
        .instruction_index = instruction_index,
        .value_id = instruction.outputs[1],
        .op = instruction.kind,
        .detail = "rng_bit_generator u64 bits require the StableHLO u64 state path",
        .feature = "mlx-rng-u64-state",
    };
    return null;
}
