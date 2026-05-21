const std = @import("std");

const ir = @import("src/compiler/ir");

const diagnostic = @import("lowering_diagnostic.zig");
const shapes = @import("lowering_shapes.zig");

const Issue = diagnostic.Issue;
const dotGeneralIsMatmulLike = shapes.dotGeneralIsMatmulLike;
const inputDescriptor = shapes.inputDescriptor;
const isSupportedFloat = shapes.isSupportedFloat;

/// Validates matmul-like dot_general forms supported by MLX lowering.
pub fn validateDotGeneral(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
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

/// Validates convolution dimension numbers and metadata supported by MLX lowering.
pub fn validateConvolution(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
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

/// Validates cholesky tensor descriptors supported by the MLX backend.
pub fn validateCholesky(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
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

/// Validates triangular_solve descriptors and options supported by the MLX backend.
pub fn validateTriangularSolve(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
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

/// Validates FFT metadata, shapes, and dtypes supported by MLX lowering.
pub fn validateFft(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, output_id: ir.ValueId) ?Issue {
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
