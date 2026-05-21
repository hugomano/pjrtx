const ir = @import("src/compiler/ir");

const buffer_mod = @import("buffer.zig");
const custom_call_mod = @import("custom_call.zig");
const diagnostic = @import("lowering_diagnostic.zig");
const shapes = @import("lowering_shapes.zig");

const Issue = diagnostic.Issue;
const dimsEqual = shapes.dimsEqual;
const inputDescriptor = shapes.inputDescriptor;
const isSupportedFloat = shapes.isSupportedFloat;
const isSupportedInteger = shapes.isSupportedInteger;

pub fn validate(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
    const target = instruction.custom_call_target orelse return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "custom_call lowering requires call_target_name",
        .feature = "mlx-custom-call",
    };
    const spec = custom_call_mod.lookup(target) orelse return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "custom_call target has no registered MLX backend implementation",
        .feature = target,
    };
    return switch (spec.kind) {
        .identity => validateIdentityCustomCallLowering(plan, instruction, instruction_index, output_id, target),
        .unary => validateUnaryCustomCallLowering(plan, instruction, instruction_index, output_id, target, spec.unary_op.?),
        .binary => validateBinaryCustomCallLowering(plan, instruction, instruction_index, output_id, target, spec.binary_op.?),
        .metal_kernel_binary_add_f32 => validateMetalKernelBinaryAddF32CustomCallLowering(plan, instruction, instruction_index, output_id, target),
        .scaled_dot_product_attention => validateScaledDotProductAttentionCustomCallLowering(plan, instruction, instruction_index, output_id, target),
    };
}

fn validateIdentityCustomCallLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId, target: []const u8) ?Issue {
    if (instruction.inputs.len != 1 or instruction.outputs.len != 1) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "identity custom_call lowering requires exactly one input and one output",
        .feature = target,
    };
    const input = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[0],
        .op = instruction.kind,
        .detail = "custom_call input is outside the executable value table",
        .feature = target,
    };
    const output = plan.values[output_id.index].descriptor;
    if (input.element_type != output.element_type or !dimsEqual(input.dims, output.dims)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "identity custom_call output must match input dtype and shape",
        .feature = target,
    };
    return null;
}

fn validateUnaryCustomCallLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId, target: []const u8, op: ir.ElementwiseUnaryOp) ?Issue {
    if (instruction.inputs.len != 1 or instruction.outputs.len != 1) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "unary custom_call lowering requires exactly one input and one output",
        .feature = target,
    };
    if (!buffer_mod.supportsUnaryOp(op)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "registered unary custom_call uses an MLX unsupported unary op",
        .feature = target,
    };
    const input = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[0],
        .op = instruction.kind,
        .detail = "custom_call input is outside the executable value table",
        .feature = target,
    };
    const output = plan.values[output_id.index].descriptor;
    if (!dimsEqual(input.dims, output.dims)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "unary custom_call lowering requires matching input/output shapes",
        .feature = target,
    };
    switch (op) {
        .is_finite => if (!isSupportedFloat(input.element_type) or output.element_type != .pred) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "is_finite custom_call lowering requires floating input and pred output",
            .feature = target,
        },
        .not_ => if ((!isSupportedInteger(input.element_type) and input.element_type != .pred) or output.element_type != input.element_type) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "not custom_call lowering requires pred or MLX-supported integer dtype",
            .feature = target,
        },
        .expm1, .round_nearest_even => if (!isSupportedFloat(input.element_type) or output.element_type != input.element_type) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "floating unary custom_call lowering requires matching MLX-supported floating dtype",
            .feature = target,
        },
        .cbrt, .round_nearest_afz => if (input.element_type != .f32 or output.element_type != .f32) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "custom Metal unary custom_call lowering currently supports f32 tensors only",
            .feature = target,
        },
        .popcnt, .count_leading_zeros => if (!isSupportedInteger(input.element_type) or output.element_type != input.element_type) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "integer unary custom_call lowering requires matching MLX-supported integer dtype",
            .feature = target,
        },
        else => if (output.element_type != input.element_type) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "unary custom_call output dtype must match input dtype",
            .feature = target,
        },
    }
    return null;
}

fn validateBinaryCustomCallLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId, target: []const u8, op: ir.ElementwiseBinaryOp) ?Issue {
    if (instruction.inputs.len != 2 or instruction.outputs.len != 1) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "binary custom_call lowering requires exactly two inputs and one output",
        .feature = target,
    };
    const lhs = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[0],
        .op = instruction.kind,
        .detail = "custom_call lhs input is outside the executable value table",
        .feature = target,
    };
    const rhs = inputDescriptor(plan, instruction, 1) orelse return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[1],
        .op = instruction.kind,
        .detail = "custom_call rhs input is outside the executable value table",
        .feature = target,
    };
    const output = plan.values[output_id.index].descriptor;
    if (lhs.element_type != rhs.element_type or lhs.element_type != output.element_type or !dimsEqual(lhs.dims, rhs.dims) or !dimsEqual(lhs.dims, output.dims)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "binary custom_call lowering requires matching input/output dtypes and shapes",
        .feature = target,
    };
    switch (op) {
        .atan2 => if (!isSupportedFloat(lhs.element_type)) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "atan2 custom_call lowering requires an MLX-supported floating dtype",
            .feature = target,
        },
        .and_, .or_, .xor => if (!isSupportedInteger(lhs.element_type) and lhs.element_type != .pred) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "logical/bitwise custom_call lowering requires pred or MLX-supported integer dtype",
            .feature = target,
        },
        .shift_left, .shift_right_arithmetic, .shift_right_logical => if (!isSupportedInteger(lhs.element_type)) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "shift custom_call lowering requires an MLX-supported integer dtype",
            .feature = target,
        },
        else => {},
    }
    return null;
}

fn validateMetalKernelBinaryAddF32CustomCallLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId, target: []const u8) ?Issue {
    const issue = validateBinaryCustomCallLowering(plan, instruction, instruction_index, output_id, target, .add);
    if (issue) |found| return found;
    const lhs = inputDescriptor(plan, instruction, 0).?;
    const output = plan.values[output_id.index].descriptor;
    if (lhs.element_type != .f32 or output.element_type != .f32) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "built-in Metal custom_call binary add currently requires f32 tensors",
        .feature = target,
    };
    return null;
}

fn validateScaledDotProductAttentionCustomCallLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId, target: []const u8) ?Issue {
    if (instruction.inputs.len != 4 or instruction.outputs.len != 1) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "scaled-dot-product-attention custom_call requires q, k, v, and token_index inputs with one output",
        .feature = target,
    };
    const q = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[0],
        .op = instruction.kind,
        .detail = "attention q input is outside the executable value table",
        .feature = target,
    };
    const k = inputDescriptor(plan, instruction, 1) orelse return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[1],
        .op = instruction.kind,
        .detail = "attention k input is outside the executable value table",
        .feature = target,
    };
    const v = inputDescriptor(plan, instruction, 2) orelse return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[2],
        .op = instruction.kind,
        .detail = "attention v input is outside the executable value table",
        .feature = target,
    };
    const token_index = inputDescriptor(plan, instruction, 3) orelse return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[3],
        .op = instruction.kind,
        .detail = "attention token_index input is outside the executable value table",
        .feature = target,
    };
    const output = plan.values[output_id.index].descriptor;
    if (!isSupportedFloat(q.element_type) or q.element_type != k.element_type or q.element_type != v.element_type or output.element_type != q.element_type) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "attention custom_call requires matching f16/bf16/f32 q/k/v/output dtypes",
        .feature = target,
    };
    if (q.dims.len != 3 and q.dims.len != 4) return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[0],
        .op = instruction.kind,
        .detail = "attention custom_call supports rank-3 [q,h,hd] or rank-4 [b,q,h,hd] q tensors",
        .feature = target,
    };
    if (k.dims.len != q.dims.len or v.dims.len != q.dims.len or !dimsEqual(k.dims, v.dims) or !dimsEqual(q.dims, output.dims)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "attention custom_call requires k/v matching shape and output shape matching q",
        .feature = target,
    };
    const query_axis: usize = if (q.dims.len == 4) 1 else 0;
    const head_axis: usize = if (q.dims.len == 4) 2 else 1;
    const dim_axis: usize = if (q.dims.len == 4) 3 else 2;
    if (q.dims[dim_axis] != k.dims[dim_axis] or q.dims[head_axis] <= 0 or k.dims[head_axis] <= 0 or @mod(q.dims[head_axis], k.dims[head_axis]) != 0 or q.dims[query_axis] <= 0 or k.dims[query_axis] <= 0) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "attention custom_call requires positive q/k lengths, equal head_dim, and q_heads divisible by kv_heads",
        .feature = target,
    };
    if (q.dims.len == 4 and q.dims[0] != k.dims[0]) return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[1],
        .op = instruction.kind,
        .detail = "attention custom_call requires q and k batch dimensions to match",
        .feature = target,
    };
    if (!isSupportedInteger(token_index.element_type) or token_index.dims.len > 1 or (token_index.dims.len == 1 and token_index.dims[0] != 1)) return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[3],
        .op = instruction.kind,
        .detail = "attention custom_call token_index must be an integer scalar or length-1 tensor",
        .feature = target,
    };
    return null;
}
