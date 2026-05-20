const std = @import("std");

const ir = @import("src/compiler/ir");
const program_mod = @import("program.zig");
const buffer_mod = @import("buffer.zig");
const custom_call_mod = @import("custom_call.zig");

/// Describes a rejected MLX backend lowering form with diagnostic context.
pub const Issue = struct {
    instruction_index: ?usize = null,
    value_id: ?ir.ValueId = null,
    op: ?ir.PlanInstructionKind = null,
    detail: []const u8,
    feature: []const u8 = "mlx-backend-executable",
};

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

fn executableSupportsInstruction(kind_: ir.PlanInstructionKind) bool {
    return switch (kind_) {
        .constant,
        .iota,
        .partition_id,
        .copy_arg0,
        .custom_call,
        .optimization_barrier,
        .reduce_precision,
        .convert,
        .bitcast_convert,
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
        .negate,
        .exp,
        .expm1,
        .tanh,
        .sqrt,
        .rsqrt,
        .abs,
        .cbrt,
        .ceil,
        .floor,
        .log,
        .log1p,
        .logistic,
        .sine,
        .cosine,
        .not_,
        .sign,
        .is_finite,
        .round_nearest_afz,
        .round_nearest_even,
        .popcnt,
        .count_leading_zeros,
        .complex,
        .real,
        .imag,
        .reshape,
        .transpose,
        .broadcast_in_dim,
        .slice,
        .dynamic_slice,
        .dynamic_update_slice,
        .pad,
        .reverse,
        .concatenate,
        .gather,
        .scatter,
        .tuple,
        .get_tuple_element,
        .sort,
        .top_k,
        .dot_general,
        .convolution,
        .cholesky,
        .triangular_solve,
        .fft,
        .rng,
        .rng_bit_generator,
        .while_,
        .reduce_sum,
        .reduce_max,
        .reduce_min,
        .reduce_and,
        .reduce_or,
        .reduce_window_sum,
        .reduce_window_max,
        .compare,
        .select,
        .clamp,
        => true,
        else => false,
    };
}

/// Returns the first MLX backend lowering issue for a compiler executable plan.
pub fn executableIssue(plan: *const ir.ExecutablePlan, device_local_hardware_ids: []const i32) ?Issue {
    if (device_local_hardware_ids.len == 0) return .{
        .detail = "backend executable requires at least one device",
        .feature = "mlx-device-assignment",
    };
    for (plan.output_ids) |output_id| {
        if (output_id.index >= plan.values.len) return .{
            .value_id = output_id,
            .detail = "plan output value is outside the executable value table",
            .feature = "mlx-executable-values",
        };
        if (plan.values[output_id.index].storage != .tensor and
            !(plan.values[output_id.index].storage == .complex_pair and plan.values[output_id.index].descriptor.element_type == .c64))
            return .{
                .value_id = output_id,
                .detail = "MLX executable PJRT outputs must be tensor values",
                .feature = "mlx-structured-output",
            };
    }
    for (plan.instructions, 0..) |instruction, instruction_index| {
        if (!executableSupportsInstruction(instruction.kind)) return .{
            .instruction_index = instruction_index,
            .op = instruction.kind,
            .detail = "operation is not supported by the MLX backend executable",
        };
        const valid_output_count = instruction.outputs.len == 1 or
            ((instruction.kind == .sort or instruction.kind == .top_k) and instruction.outputs.len == 2) or
            (instruction.kind == .reduce_max and instruction.inputs.len == 2 and instruction.outputs.len == 2) or
            (instruction.kind == .reduce_window_max and instruction.inputs.len == 2 and instruction.outputs.len == 2) or
            (instruction.kind == .rng_bit_generator and instruction.inputs.len == 1 and instruction.outputs.len == 2) or
            (instruction.kind == .optimization_barrier and instruction.outputs.len == instruction.inputs.len) or
            (instruction.kind == .while_ and instruction.outputs.len != 0 and instruction.outputs.len == instruction.inputs.len);
        if (!valid_output_count) return .{
            .instruction_index = instruction_index,
            .op = instruction.kind,
            .detail = "MLX executable lowering requires one output per instruction except two-output sort/top_k/reduce_max/reduce_window_max/rng_bit_generator",
            .feature = "mlx-executable-values",
        };
        for (instruction.outputs) |output_id| {
            if (output_id.index >= plan.values.len) return .{
                .instruction_index = instruction_index,
                .value_id = output_id,
                .op = instruction.kind,
                .detail = "instruction output value is outside the executable value table",
                .feature = "mlx-executable-values",
            };
        }
        if (instructionIssue(plan, instruction, instruction_index, instruction.outputs[0])) |issue| return issue;
    }
    return null;
}

fn instructionIssue(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
    const output_descriptor = plan.values[output_id.index].descriptor;
    return switch (instruction.kind) {
        .constant => if (instruction.literal == null) .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "constant lowering requires an embedded literal",
            .feature = "mlx-constant-literal",
        } else null,
        .custom_call => validateCustomCallLowering(plan, instruction, instruction_index, output_id),
        .fft => validateFftLowering(plan, instruction, instruction_index, output_id),
        .optimization_barrier => validateOptimizationBarrierLowering(plan, instruction, instruction_index),
        .iota => blk: {
            const dim = instruction.iota_dimension orelse break :blk .{
                .instruction_index = instruction_index,
                .value_id = output_id,
                .op = instruction.kind,
                .detail = "iota lowering requires an iota dimension",
                .feature = "mlx-iota",
            };
            if (dim < 0 or dim >= @as(i64, @intCast(output_descriptor.dims.len))) break :blk .{
                .instruction_index = instruction_index,
                .value_id = output_id,
                .op = instruction.kind,
                .detail = "iota dimension is outside the output rank",
                .feature = "mlx-iota",
            };
            break :blk null;
        },
        .partition_id => validatePartitionIdLowering(plan, instruction, instruction_index, output_id),
        .rng => validateRngLowering(plan, instruction, instruction_index, output_id),
        .bitcast_convert => validateBitcastConvertLowering(plan, instruction, instruction_index, output_id),
        .atan2, .and_, .or_, .xor, .shift_left, .shift_right_arithmetic, .shift_right_logical => validateBinaryElementwiseLowering(plan, instruction, instruction_index, output_id),
        .complex => validateComplexLowering(plan, instruction, instruction_index, output_id),
        .real, .imag => validateRealImagLowering(plan, instruction, instruction_index, output_id),
        .expm1, .cbrt, .not_, .is_finite, .round_nearest_afz, .round_nearest_even, .popcnt, .count_leading_zeros => validateUnaryElementwiseLowering(plan, instruction, instruction_index, output_id),
        .transpose => if (instruction.permutation == null) .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "transpose lowering requires a permutation",
            .feature = "mlx-layout",
        } else null,
        .broadcast_in_dim => if (instruction.broadcast_dimensions == null) .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "broadcast_in_dim lowering requires broadcast dimensions",
            .feature = "mlx-shape",
        } else null,
        .slice => if (instruction.start_indices == null or instruction.limit_indices == null or instruction.strides == null) .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "slice lowering requires static starts, limits, and strides",
            .feature = "mlx-slice",
        } else null,
        .dynamic_slice => if (instruction.slice_sizes == null) .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "dynamic_slice lowering requires static slice sizes",
            .feature = "mlx-dynamic-slice",
        } else null,
        .dynamic_update_slice => if (instruction.inputs.len < 2) .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "dynamic_update_slice lowering requires operand and update inputs",
            .feature = "mlx-dynamic-update-slice",
        } else null,
        .pad => validatePadLowering(plan, instruction, instruction_index, output_id),
        .concatenate => if (instruction.dimension == null) .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "concatenate lowering requires a dimension",
            .feature = "mlx-concatenate",
        } else null,
        .gather => validateGatherLowering(plan, instruction, instruction_index, output_id),
        .scatter => validateScatterLowering(plan, instruction, instruction_index, output_id),
        .tuple => validateTupleLowering(plan, instruction, instruction_index, output_id),
        .get_tuple_element => validateGetTupleElementLowering(plan, instruction, instruction_index, output_id),
        .sort => validateSortLowering(instruction, instruction_index, output_id),
        .top_k => validateTopKLowering(plan, instruction, instruction_index, output_id),
        .dot_general => validateDotGeneralLowering(plan, instruction, instruction_index, output_id),
        .convolution => validateConvolutionLowering(plan, instruction, instruction_index, output_id),
        .cholesky => validateCholeskyLowering(plan, instruction, instruction_index, output_id),
        .triangular_solve => validateTriangularSolveLowering(plan, instruction, instruction_index, output_id),
        .reduce_sum, .reduce_max, .reduce_min, .reduce_and, .reduce_or => validateReduceLowering(plan, instruction, instruction_index, output_id),
        .reduce_window_sum, .reduce_window_max => validateReduceWindowLowering(plan, instruction, instruction_index, output_id),
        .rng_bit_generator => validateRngBitGeneratorLowering(plan, instruction, instruction_index, output_id),
        .while_ => validateWhileLowering(plan, instruction, instruction_index, output_id),
        .compare => validateCompareLowering(plan, instruction, instruction_index, output_id),
        .select => validateSelectLowering(plan, instruction, instruction_index, output_id),
        .clamp => validateClampLowering(plan, instruction, instruction_index, output_id),
        else => null,
    };
}

fn validateBitcastConvertLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
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

fn validatePartitionIdLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
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

fn validateRngLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
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

fn validateCustomCallLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
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

fn validateOptimizationBarrierLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize) ?Issue {
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

fn validateBinaryElementwiseLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
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

fn validateUnaryElementwiseLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
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

fn validateComplexLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
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

fn validateRealImagLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
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

fn validatePadLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
    const low = instruction.edge_padding_low orelse return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "pad lowering requires low edge padding",
        .feature = "mlx-pad",
    };
    const high = instruction.edge_padding_high orelse return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "pad lowering requires high edge padding",
        .feature = "mlx-pad",
    };
    const interior = instruction.interior_padding orelse return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "pad lowering requires interior padding metadata",
        .feature = "mlx-pad",
    };
    if (instruction.inputs.len < 2) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "pad lowering requires operand and scalar padding value inputs",
        .feature = "mlx-pad",
    };
    const input_descriptor = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[0],
        .op = instruction.kind,
        .detail = "pad operand is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const padding_descriptor = inputDescriptor(plan, instruction, 1) orelse return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[1],
        .op = instruction.kind,
        .detail = "pad scalar value is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const output_descriptor = plan.values[output_id.index].descriptor;
    if (padding_descriptor.element_type != input_descriptor.element_type or padding_descriptor.dims.len != 0) return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[1],
        .op = instruction.kind,
        .detail = "pad lowering requires a scalar padding value with the operand dtype",
        .feature = "mlx-pad-scalar",
    };
    if (low.len != input_descriptor.dims.len or high.len != input_descriptor.dims.len or interior.len != input_descriptor.dims.len or output_descriptor.dims.len != input_descriptor.dims.len) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "pad metadata rank does not match operand/output rank",
        .feature = "mlx-pad",
    };
    for (input_descriptor.dims, 0..) |dim, axis| {
        if (low[axis] < 0 or high[axis] < 0 or interior[axis] < 0) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "pad lowering requires non-negative edge and interior padding",
            .feature = "mlx-pad",
        };
        const interior_slots = if (dim > 0) (dim - 1) * interior[axis] else 0;
        if (output_descriptor.dims[axis] != dim + low[axis] + high[axis] + interior_slots) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "pad output shape must equal StableHLO edge plus interior padding formula",
            .feature = "mlx-pad",
        };
    }
    return null;
}

fn validateGatherLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
    const operand = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 0) instruction.inputs[0] else output_id,
        .op = instruction.kind,
        .detail = "gather operand is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const indices = inputDescriptor(plan, instruction, 1) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 1) instruction.inputs[1] else output_id,
        .op = instruction.kind,
        .detail = "gather indices are outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const output = plan.values[output_id.index].descriptor;
    if (operand.element_type != output.element_type or !isSupportedInteger(indices.element_type)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "gather lowering requires matching operand/output dtypes and integer indices",
        .feature = "mlx-gather-types",
    };
    const slice_sizes = instruction.slice_sizes orelse return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "gather lowering requires static slice sizes",
        .feature = "mlx-gather-slice-sizes",
    };
    if (slice_sizes.len != operand.dims.len) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "gather slice size rank must match operand rank",
        .feature = "mlx-gather-slice-sizes",
    };
    if (!validGatherShape(
        operand.dims,
        indices.dims,
        instruction.start_index_map orelse &.{},
        instruction.collapsed_slice_dims orelse &.{},
        instruction.operand_batching_dims orelse &.{},
        instruction.start_indices_batching_dims orelse &.{},
        instruction.index_vector_dim orelse 0,
        slice_sizes,
        instruction.offset_dims orelse &.{},
        output.dims,
    )) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "gather metadata and output shape must match MLX general gather semantics",
        .feature = "mlx-gather-general-shape",
    };
    return null;
}

fn validateScatterLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
    const operand = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 0) instruction.inputs[0] else output_id,
        .op = instruction.kind,
        .detail = "scatter operand is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const indices = inputDescriptor(plan, instruction, 1) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 1) instruction.inputs[1] else output_id,
        .op = instruction.kind,
        .detail = "scatter indices are outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const updates = inputDescriptor(plan, instruction, 2) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 2) instruction.inputs[2] else output_id,
        .op = instruction.kind,
        .detail = "scatter updates are outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const output = plan.values[output_id.index].descriptor;
    if (operand.element_type != updates.element_type or operand.element_type != output.element_type or !dimsEqual(operand.dims, output.dims) or !isSupportedInteger(indices.element_type)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "scatter lowering requires matching operand/update/output dtypes and integer indices",
        .feature = "mlx-scatter-types",
    };
    if (instruction.scatter_update_kind == null) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "scatter combiner must be set or add",
        .feature = "mlx-scatter-combiner",
    };
    if (supportedScatterAxis(instruction)) |axis| {
        if (axis < 0 or axis >= @as(i64, @intCast(operand.dims.len))) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "scatter axis is outside the operand rank",
            .feature = "mlx-scatter-axis",
        };
        if (!scatterUpdateShapeMatchesAxis(operand.dims, indices.dims, updates.dims, instruction.index_vector_dim orelse 0, axis)) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "scatter update shape must match MLX scatter semantics for the scattered axis",
            .feature = "mlx-scatter-update-shape",
        };
        return null;
    }
    if (!validScatterShape(
        operand.dims,
        indices.dims,
        updates.dims,
        instruction.scatter_dims_to_operand_dims orelse &.{},
        instruction.inserted_window_dims orelse &.{},
        instruction.update_window_dims orelse &.{},
        instruction.input_batching_dims orelse &.{},
        instruction.scatter_indices_batching_dims orelse &.{},
        instruction.index_vector_dim orelse 0,
        output.dims,
    )) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "scatter metadata and update shape must match MLX scatter semantics",
        .feature = "mlx-scatter-shape",
    };
    return null;
}

fn validateTupleLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
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

fn validateGetTupleElementLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
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

fn validateSortLowering(instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
    if (instruction.dimension == null) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "sort lowering requires a sort dimension",
        .feature = "mlx-sort",
    };
    if (!(instruction.inputs.len == 1 and instruction.outputs.len == 1) and !(instruction.inputs.len == 2 and instruction.outputs.len == 2)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "sort lowering supports single-value sort or key/value sort with two operands and two outputs",
        .feature = "mlx-sort-arity",
    };
    switch (instruction.compare_direction orelse .lt) {
        .lt, .le, .gt, .ge => return null,
        else => return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "sort lowering supports lt/le/gt/ge comparator directions only",
            .feature = "mlx-sort-comparator",
        },
    }
}

fn validateTopKLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
    if (instruction.inputs.len != 1 or instruction.outputs.len != 2) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "top_k lowering requires one input and values/indices outputs",
        .feature = "mlx-top-k-arity",
    };
    const k = instruction.top_k_k orelse return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "top_k lowering requires static k",
        .feature = "mlx-top-k",
    };
    const input = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = instruction.inputs[0],
        .op = instruction.kind,
        .detail = "top_k input is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    if (input.dims.len == 0 or k < 0 or k > input.dims[input.dims.len - 1]) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "top_k supports static k along the last non-scalar dimension",
        .feature = "mlx-top-k-shape",
    };
    return null;
}

fn validateDotGeneralLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
    const lhs = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 0) instruction.inputs[0] else output_id,
        .op = instruction.kind,
        .detail = "dot_general lhs is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const rhs = inputDescriptor(plan, instruction, 1) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 1) instruction.inputs[1] else output_id,
        .op = instruction.kind,
        .detail = "dot_general rhs is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const output = plan.values[output_id.index].descriptor;
    if (lhs.element_type != rhs.element_type or lhs.element_type != output.element_type or !isSupportedFloat(lhs.element_type)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "dot_general lowering currently supports same-dtype MLX floating matmul-like tensors only",
        .feature = "mlx-dot-general-float",
    };
    if (!dotGeneralIsMatmulLike(
        lhs.dims,
        rhs.dims,
        instruction.lhs_batch_dimensions orelse &.{},
        instruction.rhs_batch_dimensions orelse &.{},
        instruction.lhs_contracting_dimensions orelse &.{},
        instruction.rhs_contracting_dimensions orelse &.{},
        output.dims,
    )) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "dot_general lowering currently supports matmul-like contracting dimensions only",
        .feature = "mlx-dot-general-matmul",
    };
    return null;
}

fn validateConvolutionLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
    const lhs = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 0) instruction.inputs[0] else output_id,
        .op = instruction.kind,
        .detail = "convolution lhs is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const rhs = inputDescriptor(plan, instruction, 1) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 1) instruction.inputs[1] else output_id,
        .op = instruction.kind,
        .detail = "convolution rhs is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const output = plan.values[output_id.index].descriptor;
    if (lhs.element_type != rhs.element_type or lhs.element_type != output.element_type or !isSupportedFloat(lhs.element_type)) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "convolution lowering requires matching MLX-supported floating dtypes",
        .feature = "mlx-convolution-dtype",
    };
    if (lhs.dims.len != rhs.dims.len or lhs.dims.len != output.dims.len or lhs.dims.len < 3 or lhs.dims.len > 5) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "convolution lowering supports rank 3, 4, or 5 tensors only",
        .feature = "mlx-convolution-rank",
    };
    const spatial_rank = lhs.dims.len - 2;
    if (!std.mem.eql(i64, instruction.input_spatial_dimensions orelse &.{}, defaultSpatialDims(spatial_rank)) or
        !std.mem.eql(i64, instruction.kernel_spatial_dimensions orelse &.{}, defaultSpatialDims(spatial_rank)) or
        !std.mem.eql(i64, instruction.output_spatial_dimensions orelse &.{}, defaultSpatialDims(spatial_rank)) or
        instruction.input_batch_dimension != 0 or instruction.input_feature_dimension != 1 or
        instruction.kernel_output_feature_dimension != 0 or instruction.kernel_input_feature_dimension != 1 or
        instruction.output_batch_dimension != 0 or instruction.output_feature_dimension != 1)
    {
        return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "convolution lowering currently supports ZML NCHW/OIHW-style dimension numbers only",
            .feature = "mlx-convolution-layout",
        };
    }
    if ((instruction.batch_group_count orelse 1) != 1) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "convolution lowering does not support batch_group_count yet",
        .feature = "mlx-convolution-batch-groups",
    };
    if (!convMetadataLen(instruction.window_strides, spatial_rank) or
        !convMetadataLen(instruction.edge_padding_low, spatial_rank) or
        !convMetadataLen(instruction.edge_padding_high, spatial_rank) or
        !convMetadataLen(instruction.base_dilations, spatial_rank) or
        !convMetadataLen(instruction.window_dilations, spatial_rank) or
        !convReversalLen(instruction.window_reversal, spatial_rank))
    {
        return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "convolution lowering requires static spatial window metadata",
            .feature = "mlx-convolution-window",
        };
    }
    for (instruction.window_reversal orelse &.{}) |reversed| {
        if (reversed) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "convolution lowering does not support window_reversal yet",
            .feature = "mlx-convolution-window-reversal",
        };
    }
    const groups = instruction.feature_group_count orelse 1;
    if (groups <= 0 or @rem(lhs.dims[1], groups) != 0 or rhs.dims[1] * groups != lhs.dims[1]) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "convolution feature groups must divide input channels and match kernel input channels",
        .feature = "mlx-convolution-feature-groups",
    };
    if (output.dims[0] != lhs.dims[0] or output.dims[1] != rhs.dims[0]) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "convolution output batch/features must match lhs batch and rhs output features",
        .feature = "mlx-convolution-shape",
    };
    return null;
}

fn convMetadataLen(maybe_values: ?[]const i64, expected: usize) bool {
    const values = maybe_values orelse return false;
    return values.len == expected;
}

fn convReversalLen(maybe_values: ?[]const bool, expected: usize) bool {
    const values = maybe_values orelse return false;
    return values.len == expected;
}

fn defaultSpatialDims(rank: usize) []const i64 {
    return switch (rank) {
        1 => &.{2},
        2 => &.{ 2, 3 },
        3 => &.{ 2, 3, 4 },
        else => &.{},
    };
}

fn validateCholeskyLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
    const input = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 0) instruction.inputs[0] else output_id,
        .op = instruction.kind,
        .detail = "cholesky input is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const output = plan.values[output_id.index].descriptor;
    if (input.element_type != .f32 or output.element_type != .f32) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "cholesky lowering currently supports f32 tensors only",
        .feature = "mlx-cholesky-dtype",
    };
    if (!std.mem.eql(i64, input.dims, output.dims) or output.dims.len < 2) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "cholesky output shape must match a rank >= 2 input",
        .feature = "mlx-cholesky-shape",
    };
    const n = output.dims[output.dims.len - 1];
    if (n <= 0 or output.dims[output.dims.len - 2] != n) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "cholesky lowering requires square minor dimensions",
        .feature = "mlx-cholesky-shape",
    };
    return null;
}

fn validateTriangularSolveLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
    const a = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 0) instruction.inputs[0] else output_id,
        .op = instruction.kind,
        .detail = "triangular_solve matrix input is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const b = inputDescriptor(plan, instruction, 1) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 1) instruction.inputs[1] else output_id,
        .op = instruction.kind,
        .detail = "triangular_solve rhs input is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const output = plan.values[output_id.index].descriptor;
    if (a.element_type != .f32 or b.element_type != .f32 or output.element_type != .f32) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "triangular_solve lowering currently supports f32 tensors only",
        .feature = "mlx-triangular-solve-dtype",
    };
    if (!std.mem.eql(i64, b.dims, output.dims) or a.dims.len != b.dims.len or b.dims.len < 2) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "triangular_solve output shape must match rhs and ranks must match",
        .feature = "mlx-triangular-solve-shape",
    };
    const n = a.dims[a.dims.len - 1];
    if (n <= 0 or a.dims[a.dims.len - 2] != n) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "triangular_solve matrix input must have square minor dimensions",
        .feature = "mlx-triangular-solve-shape",
    };
    for (a.dims[0 .. a.dims.len - 2], b.dims[0 .. b.dims.len - 2]) |a_dim, b_dim| {
        if (a_dim != b_dim) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "triangular_solve lowering currently requires identical batch dimensions",
            .feature = "mlx-triangular-solve-batch",
        };
    }
    if (instruction.triangular_left_side orelse true) {
        if (b.dims[b.dims.len - 2] != n) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "left-side triangular_solve requires rhs row dimension to match matrix size",
            .feature = "mlx-triangular-solve-shape",
        };
    } else if (b.dims[b.dims.len - 1] != n) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "right-side triangular_solve requires rhs column dimension to match matrix size",
        .feature = "mlx-triangular-solve-shape",
    };
    return null;
}

fn validateFftLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
    const input = inputDescriptor(plan, instruction, 0) orelse return .{
        .instruction_index = instruction_index,
        .value_id = if (instruction.inputs.len > 0) instruction.inputs[0] else output_id,
        .op = instruction.kind,
        .detail = "fft input is outside the executable value table",
        .feature = "mlx-executable-values",
    };
    const output = plan.values[output_id.index].descriptor;
    const lengths = instruction.dimensions orelse return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "fft lowering requires StableHLO fft_length metadata",
        .feature = "mlx-fft-metadata",
    };
    const fft_kind = instruction.fft_kind orelse return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "fft lowering requires StableHLO fft_type metadata",
        .feature = "mlx-fft-metadata",
    };
    if (lengths.len == 0 or lengths.len > 3 or lengths.len > input.dims.len) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "fft lowering supports one to three innermost FFT dimensions",
        .feature = "mlx-fft-rank",
    };
    if (input.dims.len != output.dims.len) return .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "fft lowering requires input and output tensors to have the same rank",
        .feature = "mlx-fft-shape",
    };
    for (lengths) |length| {
        if (length <= 0) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "fft_length values must be positive",
            .feature = "mlx-fft-shape",
        };
    }
    switch (fft_kind) {
        .fft, .ifft => if (input.element_type != .c64 or output.element_type != .c64) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "FFT/IFFT lowering currently supports c64 tensors only",
            .feature = "mlx-fft-dtype",
        },
        .rfft => if (input.element_type != .f32 or output.element_type != .c64) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "RFFT lowering currently supports f32 input and c64 output only",
            .feature = "mlx-fft-dtype",
        },
        .irfft => if (input.element_type != .c64 or output.element_type != .f32) return .{
            .instruction_index = instruction_index,
            .value_id = output_id,
            .op = instruction.kind,
            .detail = "IRFFT lowering currently supports c64 input and f32 output only",
            .feature = "mlx-fft-dtype",
        },
    }
    const first_fft_axis = input.dims.len - lengths.len;
    for (lengths, 0..) |length, index| {
        const axis = first_fft_axis + index;
        const input_dim = input.dims[axis];
        const output_dim = output.dims[axis];
        switch (fft_kind) {
            .fft, .ifft => if (input_dim != length or output_dim != length) return .{
                .instruction_index = instruction_index,
                .value_id = output_id,
                .op = instruction.kind,
                .detail = "FFT/IFFT lowering requires innermost input/output dimensions to match fft_length",
                .feature = "mlx-fft-shape",
            },
            .rfft => {
                const expected_output = if (index == lengths.len - 1) @divFloor(length, 2) + 1 else length;
                if (input_dim != length or output_dim != expected_output) return .{
                    .instruction_index = instruction_index,
                    .value_id = output_id,
                    .op = instruction.kind,
                    .detail = "RFFT lowering requires innermost input dimensions to match fft_length and final output dimension length/2+1",
                    .feature = "mlx-fft-shape",
                };
            },
            .irfft => {
                const expected_input = if (index == lengths.len - 1) @divFloor(length, 2) + 1 else length;
                if (input_dim != expected_input or output_dim != length) return .{
                    .instruction_index = instruction_index,
                    .value_id = output_id,
                    .op = instruction.kind,
                    .detail = "IRFFT lowering requires final input dimension fft_length/2+1 and output dimensions to match fft_length",
                    .feature = "mlx-fft-shape",
                };
            },
        }
    }
    return null;
}

fn validateRngBitGeneratorLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
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

fn descriptorsEqual(a: ir.BufferDescriptor, b: ir.BufferDescriptor) bool {
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
        if (executableBinaryOp(instruction.kind) == null) return null;
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

fn compareDirectionSupportedInWhile(direction: ir.CompareOp) bool {
    return switch (direction) {
        .lt, .le, .gt, .ge => true,
        .eq, .ne => false,
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

fn validateWhileLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
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

fn validateReduceLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
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

fn validateReduceWindowLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
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

fn validateCompareLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
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

fn validateSelectLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
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

fn validateClampLowering(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
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

fn inputDescriptor(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, input_index: usize) ?ir.BufferDescriptor {
    if (input_index >= instruction.inputs.len) return null;
    const id = instruction.inputs[input_index];
    if (id.index >= plan.values.len) return null;
    return plan.values[id.index].descriptor;
}

fn dimsEqual(lhs: []const i64, rhs: []const i64) bool {
    if (lhs.len != rhs.len) return false;
    for (lhs, rhs) |a, b| if (a != b) return false;
    return true;
}

fn validReduceShape(input_dims: []const i64, dimensions: []const i64, output_dims: []const i64) bool {
    var reduced = [_]bool{false} ** 64;
    if (input_dims.len > reduced.len) return false;
    for (dimensions) |dim_i64| {
        if (dim_i64 < 0 or dim_i64 >= @as(i64, @intCast(input_dims.len))) return false;
        const dim: usize = @intCast(dim_i64);
        if (reduced[dim]) return false;
        reduced[dim] = true;
    }
    var expected_rank: usize = 0;
    for (0..input_dims.len) |axis| {
        if (!reduced[axis]) expected_rank += 1;
    }
    if (output_dims.len != expected_rank) return false;
    var out_axis: usize = 0;
    for (input_dims, 0..) |dim, axis| {
        if (reduced[axis]) continue;
        if (output_dims[out_axis] != dim) return false;
        out_axis += 1;
    }
    return true;
}

fn validReduceWindowShape(input_dims: []const i64, window_dimensions: []const i64, window_strides: []const i64, base_dilations: []const i64, window_dilations: []const i64, padding_low: []const i64, padding_high: []const i64, output_dims: []const i64) bool {
    const rank = input_dims.len;
    if (rank == 0 or output_dims.len != rank or window_dimensions.len != rank or window_strides.len != rank or base_dilations.len != rank or window_dilations.len != rank or padding_low.len != rank or padding_high.len != rank) return false;
    for (0..rank) |axis| {
        if (input_dims[axis] < 0 or window_dimensions[axis] <= 0 or window_strides[axis] <= 0 or base_dilations[axis] != 1 or window_dilations[axis] <= 0 or padding_low[axis] < 0 or padding_high[axis] < 0) return false;
        const padded = padding_low[axis] + input_dims[axis] + padding_high[axis];
        const window = (window_dimensions[axis] - 1) * window_dilations[axis] + 1;
        const expected = if (padded < window) 0 else @divFloor(padded - window, window_strides[axis]) + 1;
        if (output_dims[axis] != expected) return false;
    }
    return true;
}

fn isSupportedFloat(element_type: ir.BufferType) bool {
    return switch (element_type) {
        .f16, .f32, .bf16 => true,
        else => false,
    };
}

fn isSupportedInteger(element_type: ir.BufferType) bool {
    return switch (element_type) {
        .s8, .s32, .u8, .u16, .u32, .u64 => true,
        else => false,
    };
}

fn isSupportedComparable(element_type: ir.BufferType) bool {
    return element_type == .pred or isSupportedFloat(element_type) or isSupportedInteger(element_type);
}

fn dimsBroadcastTo(input_dims: []const i64, output_dims: []const i64) bool {
    if (input_dims.len > output_dims.len) return false;
    const offset = output_dims.len - input_dims.len;
    for (input_dims, 0..) |dim, index| {
        const output_dim = output_dims[offset + index];
        if (dim != 1 and dim != output_dim) return false;
    }
    return true;
}

fn validElementwiseBroadcast(lhs_dims: []const i64, rhs_dims: []const i64, output_dims: []const i64) bool {
    return dimsBroadcastTo(lhs_dims, output_dims) and dimsBroadcastTo(rhs_dims, output_dims);
}

fn dotGeneralIsMatmulLike(lhs_dims: []const i64, rhs_dims: []const i64, lhs_batch: []const i64, rhs_batch: []const i64, lhs_contract: []const i64, rhs_contract: []const i64, output_dims: []const i64) bool {
    if (lhs_contract.len != 1 or rhs_contract.len != 1 or lhs_batch.len != rhs_batch.len or lhs_dims.len == 0 or rhs_dims.len < 2 or output_dims.len == 0) return false;
    const lhs_k = lhs_contract[0];
    const rhs_k = rhs_contract[0];
    if (lhs_k < 0 or rhs_k < 0) return false;
    if (@as(usize, @intCast(lhs_k)) >= lhs_dims.len or @as(usize, @intCast(rhs_k)) >= rhs_dims.len) return false;
    if (lhs_dims[@intCast(lhs_k)] != rhs_dims[@intCast(rhs_k)]) return false;
    var lhs_used_buf: [16]bool = [_]bool{false} ** 16;
    var rhs_used_buf: [16]bool = [_]bool{false} ** 16;
    if (lhs_dims.len > lhs_used_buf.len or rhs_dims.len > rhs_used_buf.len) return false;
    const lhs_used = lhs_used_buf[0..lhs_dims.len];
    const rhs_used = rhs_used_buf[0..rhs_dims.len];
    lhs_used[@intCast(lhs_k)] = true;
    rhs_used[@intCast(rhs_k)] = true;
    for (lhs_batch, rhs_batch) |lhs_axis, rhs_axis| {
        if (lhs_axis < 0 or rhs_axis < 0) return false;
        if (@as(usize, @intCast(lhs_axis)) >= lhs_dims.len or @as(usize, @intCast(rhs_axis)) >= rhs_dims.len) return false;
        if (lhs_used[@intCast(lhs_axis)] or rhs_used[@intCast(rhs_axis)]) return false;
        if (lhs_dims[@intCast(lhs_axis)] != rhs_dims[@intCast(rhs_axis)]) return false;
        lhs_used[@intCast(lhs_axis)] = true;
        rhs_used[@intCast(rhs_axis)] = true;
    }
    var expected_buf: [32]i64 = undefined;
    var expected: std.ArrayListUnmanaged(i64) = .initBuffer(&expected_buf);
    for (lhs_batch) |axis| expected.appendBounded(lhs_dims[@intCast(axis)]) catch return false;
    for (lhs_dims, 0..) |dim, axis| if (!lhs_used[axis]) expected.appendBounded(dim) catch return false;
    for (rhs_dims, 0..) |dim, axis| if (!rhs_used[axis]) expected.appendBounded(dim) catch return false;
    return std.mem.eql(i64, expected.items, output_dims);
}

/// Writes a stable human-readable lowering diagnostic for one issue.
pub fn writeIssue(plan: *const ir.ExecutablePlan, issue: Issue, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    try writer.writeAll("invalid executable plan: pass=mlx-backend-legalization");
    if (issue.instruction_index) |index| try writer.print(" instruction={d}", .{index});
    if (issue.op) |op| try writer.print(" op={s}", .{@tagName(op)});
    if (issue.value_id) |value_id| {
        try writer.print(" value={d}", .{value_id.index});
        if (value_id.index < plan.values.len) {
            const descriptor = plan.values[value_id.index].descriptor;
            try writer.print(" dtype={s} rank={d} shape=", .{ @tagName(descriptor.element_type), descriptor.dims.len });
            try writeDims(writer, descriptor.dims);
            try writer.print(" sharding={s}", .{shardingLabel(plan, value_id)});
        }
    }
    try writer.print(" detail=\"{s}\" feature={s}", .{ issue.detail, issue.feature });
}

fn writeDims(writer: *std.Io.Writer, dims: []const i64) std.Io.Writer.Error!void {
    try writer.writeAll("[");
    for (dims, 0..) |dim, index| {
        if (index != 0) try writer.writeAll(",");
        try writer.print("{d}", .{dim});
    }
    try writer.writeAll("]");
}

fn shardingLabel(plan: *const ir.ExecutablePlan, value_id: ir.ValueId) []const u8 {
    for (plan.output_ids, 0..) |output_id, index| {
        if (output_id.index == value_id.index and index < plan.output_shardings.len) return @tagName(plan.output_shardings[index].kind);
    }
    if (value_id.index < plan.values.len and plan.values[value_id.index].role == .parameter) {
        var parameter_index: usize = 0;
        for (plan.values) |value| {
            if (value.role != .parameter) continue;
            if (value.id.index == value_id.index and parameter_index < plan.parameter_shardings.len) return @tagName(plan.parameter_shardings[parameter_index].kind);
            parameter_index += 1;
        }
    }
    return "internal";
}

fn supportedGatherAxis(instruction: ir.PlanInstruction) ?i64 {
    const start_index_map = instruction.start_index_map orelse return null;
    const slice_sizes = instruction.slice_sizes orelse return null;
    const collapsed_slice_dims = instruction.collapsed_slice_dims orelse return null;
    if (start_index_map.len != 1 or collapsed_slice_dims.len != 1) return null;
    const axis = start_index_map[0];
    if (axis < 0 or collapsed_slice_dims[0] != axis) return null;
    if (slice_sizes.len == 0 or axis >= @as(i64, @intCast(slice_sizes.len)) or slice_sizes[@intCast(axis)] != 1) return null;
    return axis;
}

fn gatherHasExplicitIndexVector(indices_dims: []const i64, start_axis_count: usize, index_vector_dim: i64) bool {
    if (index_vector_dim < 0 or index_vector_dim >= @as(i64, @intCast(indices_dims.len))) return false;
    return indices_dims[@intCast(index_vector_dim)] == @as(i64, @intCast(start_axis_count));
}

fn markUniqueAxis(seen: []bool, axis: i64) bool {
    if (axis < 0 or axis >= @as(i64, @intCast(seen.len))) return false;
    const index: usize = @intCast(axis);
    if (seen[index]) return false;
    seen[index] = true;
    return true;
}

fn validGatherShape(operand_dims: []const i64, indices_dims: []const i64, start_index_map: []const i64, collapsed_slice_dims: []const i64, operand_batching_dims: []const i64, start_indices_batching_dims: []const i64, index_vector_dim: i64, slice_sizes: []const i64, offset_dims: []const i64, output_dims: []const i64) bool {
    if (operand_dims.len > 64 or output_dims.len > 64 or start_index_map.len == 0 or slice_sizes.len != operand_dims.len) return false;
    if (start_index_map.len > 1 and !gatherHasExplicitIndexVector(indices_dims, start_index_map.len, index_vector_dim)) return false;
    if (operand_batching_dims.len != start_indices_batching_dims.len) return false;

    var gathered = [_]bool{false} ** 64;
    for (start_index_map) |axis| {
        if (!markUniqueAxis(gathered[0..operand_dims.len], axis)) return false;
    }
    for (operand_batching_dims, start_indices_batching_dims) |operand_axis, indices_axis| {
        if (!markUniqueAxis(gathered[0..operand_dims.len], operand_axis)) return false;
        if (indices_axis < 0 or indices_axis >= @as(i64, @intCast(indices_dims.len)) or indices_axis == index_vector_dim) return false;
        if (operand_dims[@intCast(operand_axis)] != indices_dims[@intCast(indices_axis)]) return false;
        if (slice_sizes[@intCast(operand_axis)] != 1) return false;
    }

    var collapsed = [_]bool{false} ** 64;
    for (collapsed_slice_dims) |axis| {
        if (!markUniqueAxis(collapsed[0..operand_dims.len], axis)) return false;
        if (slice_sizes[@intCast(axis)] != 1) return false;
    }
    for (operand_batching_dims) |axis| {
        if (!markUniqueAxis(collapsed[0..operand_dims.len], axis)) return false;
    }

    var non_collapsed_slice_rank: usize = 0;
    for (operand_dims, 0..) |dim, axis| {
        if (dim < 0 or slice_sizes[axis] < 0 or slice_sizes[axis] > dim) return false;
        if (!collapsed[axis]) non_collapsed_slice_rank += 1;
    }
    if (offset_dims.len != non_collapsed_slice_rank) return false;

    const explicit_vector = gatherHasExplicitIndexVector(indices_dims, start_index_map.len, index_vector_dim);
    const index_prefix_rank = indices_dims.len - @as(usize, if (explicit_vector) 1 else 0);
    if (output_dims.len != index_prefix_rank + non_collapsed_slice_rank) return false;

    var output_is_offset = [_]bool{false} ** 64;
    for (offset_dims) |axis| {
        if (!markUniqueAxis(output_is_offset[0..output_dims.len], axis)) return false;
    }

    var index_axis: usize = 0;
    var slice_axis: usize = 0;
    for (output_dims, 0..) |output_dim, output_axis| {
        if (output_is_offset[output_axis]) {
            while (slice_axis < operand_dims.len and collapsed[slice_axis]) slice_axis += 1;
            if (slice_axis >= slice_sizes.len or output_dim != slice_sizes[slice_axis]) return false;
            slice_axis += 1;
        } else {
            while (explicit_vector and index_axis == @as(usize, @intCast(index_vector_dim))) index_axis += 1;
            if (index_axis >= indices_dims.len or output_dim != indices_dims[index_axis]) return false;
            index_axis += 1;
        }
    }
    return true;
}

/// Returns the axis for scatter forms executable by the MLX axis fast path.
pub fn supportedScatterAxis(instruction: ir.PlanInstruction) ?i64 {
    const scatter_dims_to_operand_dims = instruction.scatter_dims_to_operand_dims orelse return null;
    const inserted_window_dims = instruction.inserted_window_dims orelse return null;
    const input_batching_dims = instruction.input_batching_dims orelse &.{};
    const scatter_indices_batching_dims = instruction.scatter_indices_batching_dims orelse &.{};
    if (scatter_dims_to_operand_dims.len != 1 or inserted_window_dims.len != 1) return null;
    if (input_batching_dims.len != 0 or scatter_indices_batching_dims.len != 0) return null;
    const axis = scatter_dims_to_operand_dims[0];
    if (axis < 0 or inserted_window_dims[0] != axis) return null;
    return axis;
}

fn validScatterShape(operand_dims: []const i64, indices_dims: []const i64, update_dims: []const i64, scatter_dims_to_operand_dims: []const i64, inserted_window_dims: []const i64, update_window_dims: []const i64, input_batching_dims: []const i64, scatter_indices_batching_dims: []const i64, index_vector_dim: i64, output_dims: []const i64) bool {
    if (operand_dims.len > 64 or !dimsEqual(operand_dims, output_dims)) return false;
    if (scatter_dims_to_operand_dims.len == 0 or inserted_window_dims.len + update_window_dims.len + input_batching_dims.len != operand_dims.len or input_batching_dims.len != scatter_indices_batching_dims.len) return false;
    const explicit_vector = gatherHasExplicitIndexVector(indices_dims, scatter_dims_to_operand_dims.len, index_vector_dim);
    if (scatter_dims_to_operand_dims.len > 1 and !explicit_vector) return false;

    var scatter_axes = [_]bool{false} ** 64;
    for (scatter_dims_to_operand_dims) |axis| {
        if (!markUniqueAxis(scatter_axes[0..operand_dims.len], axis)) return false;
    }
    for (input_batching_dims, scatter_indices_batching_dims) |operand_axis, indices_axis| {
        if (!markUniqueAxis(scatter_axes[0..operand_dims.len], operand_axis)) return false;
        if (indices_axis < 0 or indices_axis >= @as(i64, @intCast(indices_dims.len)) or indices_axis == index_vector_dim) return false;
        if (operand_dims[@intCast(operand_axis)] != indices_dims[@intCast(indices_axis)]) return false;
    }

    var window_axes = [_]bool{false} ** 64;
    for (inserted_window_dims) |axis| {
        if (!markUniqueAxis(window_axes[0..operand_dims.len], axis)) return false;
    }
    for (update_window_dims) |axis| {
        if (!markUniqueAxis(window_axes[0..operand_dims.len], axis)) return false;
    }
    for (input_batching_dims) |axis| {
        if (axis < 0 or axis >= @as(i64, @intCast(operand_dims.len)) or window_axes[@intCast(axis)]) return false;
    }

    const index_prefix_rank = indices_dims.len - @as(usize, if (explicit_vector) 1 else 0);
    if (update_dims.len != index_prefix_rank + update_window_dims.len) return false;
    var update_axis: usize = 0;
    for (indices_dims, 0..) |dim, axis| {
        if (explicit_vector and axis == @as(usize, @intCast(index_vector_dim))) continue;
        if (update_axis >= update_dims.len or update_dims[update_axis] != dim) return false;
        update_axis += 1;
    }
    if (update_axis != index_prefix_rank) return false;
    for (update_window_dims, 0..) |operand_axis, window_axis| {
        if (operand_axis < 0 or operand_axis >= @as(i64, @intCast(operand_dims.len))) return false;
        const dim = update_dims[index_prefix_rank + window_axis];
        if (dim < 0 or dim > operand_dims[@intCast(operand_axis)]) return false;
    }
    return true;
}

fn scatterUpdateShapeMatchesAxis(operand_dims: []const i64, indices_dims: []const i64, update_dims: []const i64, index_vector_dim: i64, axis: i64) bool {
    if (axis < 0 or axis >= @as(i64, @intCast(operand_dims.len))) return false;
    var expected_rank: usize = operand_dims.len - 1;
    for (indices_dims, 0..) |dim, dim_index| {
        if (index_vector_dim >= 0 and dim_index == @as(usize, @intCast(index_vector_dim)) and dim == 1) continue;
        expected_rank += 1;
    }
    if (update_dims.len != expected_rank) return false;

    var update_index: usize = 0;
    for (operand_dims[0..@intCast(axis)]) |dim| {
        if (update_dims[update_index] != dim) return false;
        update_index += 1;
    }
    for (indices_dims, 0..) |dim, dim_index| {
        if (index_vector_dim >= 0 and dim_index == @as(usize, @intCast(index_vector_dim)) and dim == 1) continue;
        if (update_dims[update_index] != dim) return false;
        update_index += 1;
    }
    const axis_index: usize = @intCast(axis);
    for (operand_dims[axis_index + 1 ..]) |dim| {
        if (update_dims[update_index] != dim) return false;
        update_index += 1;
    }
    return update_index == update_dims.len;
}

fn gatherOutputShapeMatchesTake(operand_dims: []const i64, indices_dims: []const i64, index_vector_dim: i64, axis: i64, output_dims: []const i64) bool {
    if (axis < 0 or axis >= @as(i64, @intCast(operand_dims.len))) return false;
    var expected_rank: usize = operand_dims.len - 1;
    for (indices_dims, 0..) |dim, dim_index| {
        if (index_vector_dim >= 0 and dim_index == @as(usize, @intCast(index_vector_dim)) and dim == 1) continue;
        expected_rank += 1;
    }
    if (output_dims.len != expected_rank) return false;

    var out_index: usize = 0;
    for (operand_dims[0..@intCast(axis)]) |dim| {
        if (output_dims[out_index] != dim) return false;
        out_index += 1;
    }
    for (indices_dims, 0..) |dim, dim_index| {
        if (index_vector_dim >= 0 and dim_index == @as(usize, @intCast(index_vector_dim)) and dim == 1) continue;
        if (output_dims[out_index] != dim) return false;
        out_index += 1;
    }
    const axis_index: usize = @intCast(axis);
    for (operand_dims[axis_index + 1 ..]) |dim| {
        if (output_dims[out_index] != dim) return false;
        out_index += 1;
    }
    return out_index == output_dims.len;
}

/// Maps a compiler instruction kind to the MLX executable binary op payload.
pub fn executableBinaryOp(instruction_kind: ir.PlanInstructionKind) ?ir.ElementwiseBinaryOp {
    return switch (instruction_kind) {
        .add => .add,
        .subtract => .subtract,
        .multiply => .multiply,
        .divide => .divide,
        .maximum => .maximum,
        .minimum => .minimum,
        .power => .power,
        .atan2 => .atan2,
        .remainder => .remainder,
        .and_ => .and_,
        .or_ => .or_,
        .xor => .xor,
        .shift_left => .shift_left,
        .shift_right_arithmetic, .shift_right_logical => .shift_right_logical,
        else => null,
    };
}

/// Maps a compiler instruction kind to the MLX executable unary op payload.
pub fn executableUnaryOp(instruction_kind: ir.PlanInstructionKind) ?ir.ElementwiseUnaryOp {
    return switch (instruction_kind) {
        .negate => .negate,
        .exp => .exp,
        .expm1 => .expm1,
        .tanh => .tanh,
        .sqrt => .sqrt,
        .rsqrt => .rsqrt,
        .abs => .abs,
        .cbrt => .cbrt,
        .ceil => .ceil,
        .floor => .floor,
        .log => .log,
        .log1p => .log1p,
        .logistic => .logistic,
        .sine => .sine,
        .cosine => .cosine,
        .not_ => .not_,
        .sign => .sign,
        .is_finite => .is_finite,
        .round_nearest_afz => .round_nearest_afz,
        .round_nearest_even => .round_nearest_even,
        .popcnt => .popcnt,
        .count_leading_zeros => .count_leading_zeros,
        else => null,
    };
}

test "mlx metal backend rejects gspmd custom call targets precisely" {
    const allocator = std.testing.allocator;
    const assignment = [_]i32{0};
    const dims = [_]i64{4};

    const values = try allocator.alloc(ir.Value, 2);
    for (values, 0..) |*value, index| {
        value.* = .{
            .id = .{ .index = @intCast(index) },
            .role = if (index == 0) .parameter else .instruction_result,
            .descriptor = .{
                .element_type = .u8,
                .dims = try allocator.dupe(i64, &dims),
                .device_id = 0,
                .memory_id = 0,
                .shard_index = 0,
            },
        };
    }

    const parameter_shardings = try allocator.alloc(ir.ShardingPlan, 1);
    parameter_shardings[0] = .{
        .kind = .replicated,
        .mesh_name = try allocator.dupe(u8, "test"),
        .device_assignment = try allocator.dupe(i32, &assignment),
    };

    var plan = ir.ExecutablePlan{
        .allocator = allocator,
        .options = .{
            .num_replicas = 1,
            .num_partitions = 1,
            .device_assignment = try allocator.dupe(i32, &assignment),
        },
        .values = values,
        .parameter_shardings = parameter_shardings,
        .output_shardings = try allocator.alloc(ir.ShardingPlan, 0),
        .output_ids = try allocator.dupe(ir.ValueId, &.{.{ .index = 1 }}),
        .instructions = try allocator.dupe(ir.PlanInstruction, &.{.{
            .kind = .custom_call,
            .inputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 0 }}),
            .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 1 }}),
            .dims = try allocator.dupe(i64, &dims),
            .custom_call_target = try allocator.dupe(u8, "Sharding"),
        }}),
    };
    defer plan.deinit();

    const issue = executableIssue(&plan, &assignment) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("Sharding", issue.feature);

    var diagnostics = std.Io.Writer.Allocating.init(allocator);
    defer diagnostics.deinit();
    try writeIssue(&plan, issue, &diagnostics.writer);
    const message = diagnostics.writer.buffered();
    try std.testing.expect(std.mem.indexOf(u8, message, "op=custom_call") != null);
    try std.testing.expect(std.mem.indexOf(u8, message, "feature=Sharding") != null);
}

test "mlx metal backend executable rejects unsupported gather form during lowering" {
    const allocator = std.testing.allocator;
    const assignment = [_]i32{0};
    const operand_dims = [_]i64{ 4, 2 };
    const index_dims = [_]i64{2};
    const output_dims = [_]i64{ 2, 2 };

    const values = try allocator.alloc(ir.Value, 3);
    errdefer allocator.free(values);
    values[0] = .{
        .id = .{ .index = 0 },
        .role = .parameter,
        .descriptor = .{
            .element_type = .f32,
            .dims = try allocator.dupe(i64, &operand_dims),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };
    values[1] = .{
        .id = .{ .index = 1 },
        .role = .parameter,
        .descriptor = .{
            .element_type = .s32,
            .dims = try allocator.dupe(i64, &index_dims),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };
    values[2] = .{
        .id = .{ .index = 2 },
        .role = .instruction_result,
        .descriptor = .{
            .element_type = .f32,
            .dims = try allocator.dupe(i64, &output_dims),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };

    var parameter_shardings = try allocator.alloc(ir.ShardingPlan, 2);
    parameter_shardings[0] = .{
        .kind = .replicated,
        .mesh_name = try allocator.dupe(u8, ""),
        .device_assignment = try allocator.dupe(i32, &assignment),
    };
    parameter_shardings[1] = .{
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

    var plan = ir.ExecutablePlan{
        .allocator = allocator,
        .options = .{
            .num_replicas = 1,
            .num_partitions = 1,
            .device_assignment = try allocator.dupe(i32, &assignment),
        },
        .values = values,
        .parameter_shardings = parameter_shardings,
        .output_shardings = output_shardings,
        .output_ids = try allocator.dupe(ir.ValueId, &.{.{ .index = 2 }}),
        .instructions = try allocator.dupe(ir.PlanInstruction, &.{.{
            .kind = .gather,
            .inputs = try allocator.dupe(ir.ValueId, &.{ .{ .index = 0 }, .{ .index = 1 } }),
            .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 2 }}),
            .dims = try allocator.dupe(i64, &output_dims),
            .start_index_map = try allocator.dupe(i64, &.{ 0, 1 }),
            .collapsed_slice_dims = try allocator.dupe(i64, &.{1}),
            .slice_sizes = try allocator.dupe(i64, &.{ 4, 1 }),
            .index_vector_dim = 1,
        }}),
    };
    defer plan.deinit();

    const issue = executableIssue(&plan, &assignment) orelse return error.TestUnexpectedResult;
    try std.testing.expectEqualStrings("mlx-gather-general-shape", issue.feature);
    var diagnostics = std.Io.Writer.Allocating.init(allocator);
    defer diagnostics.deinit();
    try writeIssue(&plan, issue, &diagnostics.writer);
    const message = diagnostics.writer.buffered();
    try std.testing.expect(std.mem.indexOf(u8, message, "op=gather") != null);
    try std.testing.expect(std.mem.indexOf(u8, message, "feature=mlx-gather-general-shape") != null);
}
