const ir = @import("src/compiler/ir");

const diagnostic = @import("lowering_diagnostic.zig");
const shapes = @import("lowering_shapes.zig");

const Issue = diagnostic.Issue;
const dimsEqual = shapes.dimsEqual;
const inputDescriptor = shapes.inputDescriptor;
const isSupportedInteger = shapes.isSupportedInteger;
const scatterUpdateShapeMatchesAxis = shapes.scatterUpdateShapeMatchesAxis;
const supportedScatterAxis = shapes.supportedScatterAxis;
const validGatherShape = shapes.validGatherShape;
const validScatterShape = shapes.validScatterShape;

/// Validates transpose metadata required by MLX layout lowering.
pub fn validateTranspose(instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
    return if (instruction.permutation == null) .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "transpose lowering requires a permutation",
        .feature = "mlx-layout",
    } else null;
}

/// Validates broadcast_in_dim metadata required by MLX shape lowering.
pub fn validateBroadcastInDim(instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
    return if (instruction.broadcast_dimensions == null) .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "broadcast_in_dim lowering requires broadcast dimensions",
        .feature = "mlx-shape",
    } else null;
}

/// Validates static slice metadata required by MLX slice lowering.
pub fn validateSlice(instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
    return if (instruction.start_indices == null or instruction.limit_indices == null or instruction.strides == null) .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "slice lowering requires static starts, limits, and strides",
        .feature = "mlx-slice",
    } else null;
}

/// Validates dynamic_slice static metadata required by MLX lowering.
pub fn validateDynamicSlice(instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
    return if (instruction.slice_sizes == null) .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "dynamic_slice lowering requires static slice sizes",
        .feature = "mlx-dynamic-slice",
    } else null;
}

/// Validates dynamic_update_slice arity required by MLX lowering.
pub fn validateDynamicUpdateSlice(instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
    return if (instruction.inputs.len < 2) .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "dynamic_update_slice lowering requires operand and update inputs",
        .feature = "mlx-dynamic-update-slice",
    } else null;
}

/// Validates concatenate metadata required by MLX lowering.
pub fn validateConcatenate(instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
    return if (instruction.dimension == null) .{
        .instruction_index = instruction_index,
        .value_id = output_id,
        .op = instruction.kind,
        .detail = "concatenate lowering requires a dimension",
        .feature = "mlx-concatenate",
    } else null;
}

/// Validates static StableHLO pad metadata before MLX pad lowering.
pub fn validatePad(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
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

/// Validates gather metadata and descriptors before MLX gather lowering.
pub fn validateGather(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
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

/// Validates scatter metadata and descriptors before MLX scatter lowering.
pub fn validateScatter(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
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

/// Validates supported StableHLO sort forms for MLX lowering.
pub fn validateSort(instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
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

/// Validates supported top-k metadata for MLX lowering.
pub fn validateTopK(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
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
