const ir = @import("src/compiler/ir");

const diagnostic = @import("lowering_diagnostic.zig");
const shapes = @import("lowering_shapes.zig");

const Issue = diagnostic.Issue;
const dimsEqual = shapes.dimsEqual;
const inputDescriptor = shapes.inputDescriptor;
const isSupportedComparable = shapes.isSupportedComparable;
const isSupportedFloat = shapes.isSupportedFloat;
const isSupportedInteger = shapes.isSupportedInteger;
const validElementwiseBroadcast = shapes.validElementwiseBroadcast;

/// Validates binary elementwise forms that MLX can lower without changing semantics.
pub fn validateBinary(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
    const lhs = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 0) instruction.inputs[0] else output_id,
        .op = instruction.kind,
        .detail = "binary operand lhs is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const rhs = inputDescriptor(plan, instruction, 1) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 1) instruction.inputs[1] else output_id,
        .op = instruction.kind,
        .detail = "binary operand rhs is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const output = plan.values[output_id.index].descriptor;
    if (lhs.element_type != rhs.element_type or lhs.element_type != output.element_type or !validElementwiseBroadcast(lhs.dims, rhs.dims, output.dims)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "binary lowering requires matching dtypes and broadcast-compatible input/output shapes",
        .feature = "mlx-elementwise-binary",
    };
    switch (instruction.kind) {
        .atan2 => if (!isSupportedFloat(lhs.element_type)) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "atan2 lowering requires an MLX-supported floating dtype",
            .feature = "mlx-atan2",
        },
        .and_, .or_, .xor => if (!isSupportedInteger(lhs.element_type) and lhs.element_type != .pred) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "logical/bitwise lowering requires pred or MLX-supported integer dtype",
            .feature = "mlx-bitwise",
        },
        .shift_left, .shift_right_arithmetic, .shift_right_logical => if (!isSupportedInteger(lhs.element_type)) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "shift lowering requires an MLX-supported integer dtype",
            .feature = "mlx-shift",
        },
        else => {},
    }
    return null;
}

/// Validates unary elementwise forms that MLX can lower without changing semantics.
pub fn validateUnary(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
    const input = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 0) instruction.inputs[0] else output_id,
        .op = instruction.kind,
        .detail = "unary operand is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const output = plan.values[output_id.index].descriptor;
    if (!dimsEqual(input.dims, output.dims)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "unary lowering requires matching input/output shapes",
        .feature = "mlx-elementwise-unary",
    };
    switch (instruction.kind) {
        .abs => {
            const valid_abs = if (input.element_type == .c64)
                output.element_type == .f32
            else
                output.element_type == input.element_type and (isSupportedFloat(input.element_type) or isSupportedInteger(input.element_type));
            if (!valid_abs) return .{
                .instruction_index = instruction_index,
                .value_id = output_id,
                .op = instruction.kind,
                .detail = "abs lowering requires MLX-supported real/integer input with matching output or c64 input with f32 output",
                .feature = "mlx-abs",
            };
        },
        .expm1, .round_nearest_even => if (!isSupportedFloat(input.element_type) or output.element_type != input.element_type) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "floating unary lowering requires matching MLX-supported floating dtype",
            .feature = "mlx-unary-float",
        },
        .cbrt, .round_nearest_afz => if (input.element_type != .f32 or output.element_type != .f32) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "custom Metal unary lowering currently supports f32 tensors only",
            .feature = "mlx-metal-unary-f32",
        },
        .popcnt, .count_leading_zeros => if (!isSupportedInteger(input.element_type) or output.element_type != input.element_type) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "integer unary lowering requires matching MLX-supported integer dtype",
            .feature = "mlx-integer-unary",
        },
        .is_finite => if (!isSupportedFloat(input.element_type) or output.element_type != .pred) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "is_finite lowering requires floating input and pred output",
            .feature = "mlx-is-finite",
        },
        .not_ => if ((!isSupportedInteger(input.element_type) and input.element_type != .pred) or output.element_type != input.element_type) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "not lowering requires pred or MLX-supported integer dtype",
            .feature = "mlx-not",
        },
        else => {},
    }
    return null;
}

/// Validates StableHLO complex construction for the MLX complex representation.
pub fn validateComplex(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
    const real = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 0) instruction.inputs[0] else output_id,
        .op = instruction.kind,
        .detail = "complex real operand is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const imag = inputDescriptor(plan, instruction, 1) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 1) instruction.inputs[1] else output_id,
        .op = instruction.kind,
        .detail = "complex imaginary operand is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const output = plan.values[output_id.index].descriptor;
    if (real.element_type != .f32 or imag.element_type != .f32 or output.element_type != .c64) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "complex lowering currently supports f32 operands producing c64 tensors",
        .feature = "mlx-complex-dtype",
    };
    if (!dimsEqual(real.dims, imag.dims) or !dimsEqual(real.dims, output.dims)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "complex lowering requires matching real, imaginary, and output shapes",
        .feature = "mlx-complex-shape",
    };
    return null;
}

/// Validates real and imaginary extraction against MLX-supported tensor descriptors.
pub fn validateRealImag(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
    const input = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 0) instruction.inputs[0] else output_id,
        .op = instruction.kind,
        .detail = "real/imag operand is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const output = plan.values[output_id.index].descriptor;
    if (!dimsEqual(input.dims, output.dims)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "real/imag lowering requires matching input/output shapes",
        .feature = "mlx-real-imag-shape",
    };
    if (input.element_type == .c64 and output.element_type == .f32) return null;
    if (isSupportedFloat(input.element_type) and output.element_type == input.element_type) return null;
    return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "real/imag lowering supports real floating tensors or c64-to-f32 extraction",
        .feature = "mlx-real-imag-dtype",
    };
}

/// Validates compare operands and predicate output descriptors.
pub fn validateCompare(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
    const lhs = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 0) instruction.inputs[0] else output_id,
        .op = instruction.kind,
        .detail = "compare lhs is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const rhs = inputDescriptor(plan, instruction, 1) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 1) instruction.inputs[1] else output_id,
        .op = instruction.kind,
        .detail = "compare rhs is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const output = plan.values[output_id.index].descriptor;
    if (lhs.element_type != rhs.element_type or !isSupportedComparable(lhs.element_type) or output.element_type != .pred or !validElementwiseBroadcast(lhs.dims, rhs.dims, output.dims)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "compare lowering requires MLX-supported comparable inputs, pred output, and broadcast-compatible shapes",
        .feature = "mlx-compare",
    };
    return null;
}

/// Validates select predicate/value descriptors for MLX lowering.
pub fn validateSelect(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
    const pred = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 0) instruction.inputs[0] else output_id,
        .op = instruction.kind,
        .detail = "select predicate is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const on_true = inputDescriptor(plan, instruction, 1) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 1) instruction.inputs[1] else output_id,
        .op = instruction.kind,
        .detail = "select true operand is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const on_false = inputDescriptor(plan, instruction, 2) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 2) instruction.inputs[2] else output_id,
        .op = instruction.kind,
        .detail = "select false operand is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const output = plan.values[output_id.index].descriptor;
    if (pred.element_type != .pred or on_true.element_type != on_false.element_type or !dimsEqual(pred.dims, on_true.dims) or !dimsEqual(on_true.dims, on_false.dims) or !dimsEqual(on_true.dims, output.dims)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "select lowering requires same-shape pred/true/false tensors",
        .feature = "mlx-select",
    };
    return null;
}

/// Validates clamp bounds and result descriptors for MLX lowering.
pub fn validateClamp(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
    const min = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 0) instruction.inputs[0] else output_id,
        .op = instruction.kind,
        .detail = "clamp min operand is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const value = inputDescriptor(plan, instruction, 1) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 1) instruction.inputs[1] else output_id,
        .op = instruction.kind,
        .detail = "clamp value operand is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const max = inputDescriptor(plan, instruction, 2) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 2) instruction.inputs[2] else output_id,
        .op = instruction.kind,
        .detail = "clamp max operand is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const output = plan.values[output_id.index].descriptor;
    if (min.element_type != value.element_type or max.element_type != value.element_type or output.element_type != value.element_type or !dimsEqual(value.dims, output.dims)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "clamp lowering requires matching min/value/max/output dtypes and value/output shapes",
        .feature = "mlx-clamp",
    };
    if (min.dims.len != 0 and !dimsEqual(min.dims, value.dims)) return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[0],
        .op = instruction.kind,
        .detail = "clamp min must be scalar or match the value shape",
        .feature = "mlx-clamp-bounds",
    };
    if (max.dims.len != 0 and !dimsEqual(max.dims, value.dims)) return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[2],
        .op = instruction.kind,
        .detail = "clamp max must be scalar or match the value shape",
        .feature = "mlx-clamp-bounds",
    };
    return null;
}
