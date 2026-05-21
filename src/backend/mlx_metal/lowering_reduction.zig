const std = @import("std");

const ir = @import("src/compiler/ir");

const diagnostic = @import("lowering_diagnostic.zig");
const shapes = @import("lowering_shapes.zig");

const Issue = diagnostic.Issue;
const inputDescriptor = shapes.inputDescriptor;
const isSupportedFloat = shapes.isSupportedFloat;
const isSupportedInteger = shapes.isSupportedInteger;
const validReduceShape = shapes.validReduceShape;
const validReduceWindowShape = shapes.validReduceWindowShape;

/// Validates reduce and argmax-like reduce forms supported by MLX lowering.
pub fn validateReduce(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
    if (instruction.inputs.len == 2 and instruction.outputs.len == 2 and instruction.kind == .reduce_max) {
        const values = inputDescriptor(plan, instruction, 0) orelse return .{
            .instruction_index = instruction_index,
            .value_id = instruction.inputs[0],
            .op = instruction.kind,
            .detail = "reduce argmax values input is outside the executable value table",
            .feature = "mlx-executable-values",
        };
        const indices = inputDescriptor(plan, instruction, 1) orelse return .{
            .instruction_index = instruction_index,
            .value_id = instruction.inputs[1],
            .op = instruction.kind,
            .detail = "reduce argmax indices input is outside the executable value table",
            .feature = "mlx-executable-values",
        };
        const values_output = plan.values[instruction.outputs[0].index].descriptor;
        const indices_output = plan.values[instruction.outputs[1].index].descriptor;
        if (!isSupportedFloat(values.element_type) or values.element_type != values_output.element_type) return .{
            .instruction_index = instruction_index,
            .value_id = instruction.outputs[0],
            .op = instruction.kind,
            .detail = "reduce argmax values must preserve f16/bf16/f32 dtype",
            .feature = "mlx-reduce-types",
        };
        if ((indices.element_type != .s32 and indices.element_type != .u32) or indices.element_type != indices_output.element_type) return .{
            .instruction_index = instruction_index,
            .value_id = instruction.outputs[1],
            .op = instruction.kind,
            .detail = "reduce argmax indices must preserve s32/u32 dtype",
            .feature = "mlx-reduce-types",
        };
        if (!std.mem.eql(i64, values.dims, indices.dims) or !std.mem.eql(i64, values_output.dims, indices_output.dims)) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "reduce argmax values and indices must have matching input and output shapes",
            .feature = "mlx-reduce-shape",
        };
        if (!validReduceShape(values.dims, instruction.reduce_dimensions orelse &.{}, values_output.dims)) return .{
            .instruction_index = instruction_index,
            .value_id = instruction.outputs[0],
            .op = instruction.kind,
            .detail = "reduce argmax output shape must equal input shape with reduced axes removed",
            .feature = "mlx-reduce-shape",
        };
        return null;
    }
    const input = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 0) instruction.inputs[0] else output_id,
        .op = instruction.kind,
        .detail = "reduce operand is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const output = plan.values[output_id.index].descriptor;
    const supported_reduce_type = switch (instruction.kind) {
        .reduce_sum, .reduce_max, .reduce_min => input.element_type == output.element_type and (isSupportedFloat(input.element_type) or isSupportedInteger(input.element_type)),
        .reduce_and, .reduce_or => input.element_type == .pred and output.element_type == .pred,
        else => false,
    };
    if (!supported_reduce_type) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "reduce lowering supports integer and f16/bf16/f32 sum/max/min plus pred and/or only",
        .feature = "mlx-reduce-types",
    };
    if (!validReduceShape(input.dims, instruction.reduce_dimensions orelse &.{}, output.dims)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "reduce output shape must equal input shape with reduced axes removed",
        .feature = "mlx-reduce-shape",
    };
    return null;
}

/// Validates reduce_window and max-pool-like forms supported by MLX lowering.
pub fn validateReduceWindow(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
    if (instruction.inputs.len == 2 and instruction.outputs.len == 2 and instruction.kind == .reduce_window_max) {
        const values = inputDescriptor(plan, instruction, 0) orelse return .{
            .instruction_index = instruction_index,
            .value_id = instruction.inputs[0],
            .op = instruction.kind,
            .detail = "reduce_window max-pool values input is outside the executable value table",
            .feature = "mlx-executable-values",
        };
        const indices = inputDescriptor(plan, instruction, 1) orelse return .{
            .instruction_index = instruction_index,
            .value_id = instruction.inputs[1],
            .op = instruction.kind,
            .detail = "reduce_window max-pool indices input is outside the executable value table",
            .feature = "mlx-executable-values",
        };
        const values_output = plan.values[instruction.outputs[0].index].descriptor;
        const indices_output = plan.values[instruction.outputs[1].index].descriptor;
        if (!isSupportedFloat(values.element_type) or values.element_type != values_output.element_type) return .{
            .instruction_index = instruction_index,
            .value_id = instruction.outputs[0],
            .op = instruction.kind,
            .detail = "reduce_window max-pool values must preserve f16/bf16/f32 dtype",
            .feature = "mlx-reduce-window-types",
        };
        if ((indices.element_type != .s32 and indices.element_type != .u32) or indices.element_type != indices_output.element_type) return .{
            .instruction_index = instruction_index,
            .value_id = instruction.outputs[1],
            .op = instruction.kind,
            .detail = "reduce_window max-pool indices must preserve s32/u32 dtype",
            .feature = "mlx-reduce-window-types",
        };
        if (!std.mem.eql(i64, values.dims, indices.dims) or !std.mem.eql(i64, values_output.dims, indices_output.dims)) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "reduce_window max-pool values and indices must have matching input and output shapes",
            .feature = "mlx-reduce-window-shape",
        };
        if (!validReduceWindowShape(
            values.dims,
            instruction.window_dimensions orelse &.{},
            instruction.window_strides orelse &.{},
            instruction.base_dilations orelse &.{},
            instruction.window_dilations orelse &.{},
            instruction.edge_padding_low orelse &.{},
            instruction.edge_padding_high orelse &.{},
            values_output.dims,
        )) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "reduce_window max-pool shape metadata must match StableHLO static window shape formula with unit base dilation",
            .feature = "mlx-reduce-window-shape",
        };
        return null;
    }
    if (instruction.inputs.len != 1 or instruction.outputs.len != 1) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "reduce_window lowering currently supports exactly one operand/result",
        .feature = "mlx-reduce-window-arity",
    };
    const input = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[0],
        .op = instruction.kind,
        .detail = "reduce_window operand is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const output = plan.values[output_id.index].descriptor;
    if (input.element_type != output.element_type or !isSupportedFloat(input.element_type)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "reduce_window lowering currently supports f16/bf16/f32 sum/max only",
        .feature = "mlx-reduce-window-types",
    };
    if (!validReduceWindowShape(
        input.dims,
        instruction.window_dimensions orelse &.{},
        instruction.window_strides orelse &.{},
        instruction.base_dilations orelse &.{},
        instruction.window_dilations orelse &.{},
        instruction.edge_padding_low orelse &.{},
        instruction.edge_padding_high orelse &.{},
        output.dims,
    )) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "reduce_window shape metadata must match StableHLO static window shape formula with unit base dilation",
        .feature = "mlx-reduce-window-shape",
    };
    return null;
}
