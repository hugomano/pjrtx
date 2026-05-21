const std = @import("std");

const ir = @import("src/compiler/ir");
const diagnostic = @import("lowering_diagnostic.zig");
const custom_call_lowering = @import("lowering_custom_call.zig");
const control_flow = @import("lowering_control_flow.zig");
const elementwise_lowering = @import("lowering_elementwise.zig");
const indexing_lowering = @import("lowering_indexing.zig");
const linalg_lowering = @import("lowering_linalg.zig");
const reduction_lowering = @import("lowering_reduction.zig");
const stateful_lowering = @import("lowering_stateful.zig");

const Issue = diagnostic.Issue;
const writeIssue = diagnostic.writeIssue;

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
        .custom_call => custom_call_lowering.validate(plan, instruction, instruction_index, output_id),
        .fft => linalg_lowering.validateFft(plan, instruction, instruction_index, output_id),
        .optimization_barrier => stateful_lowering.validateOptimizationBarrier(plan, instruction, instruction_index),
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
        .partition_id => stateful_lowering.validatePartitionId(plan, instruction, instruction_index, output_id),
        .rng => stateful_lowering.validateRng(plan, instruction, instruction_index, output_id),
        .bitcast_convert => stateful_lowering.validateBitcastConvert(plan, instruction, instruction_index, output_id),
        .atan2, .and_, .or_, .xor, .shift_left, .shift_right_arithmetic, .shift_right_logical => elementwise_lowering.validateBinary(plan, instruction, instruction_index, output_id),
        .complex => elementwise_lowering.validateComplex(plan, instruction, instruction_index, output_id),
        .real, .imag => elementwise_lowering.validateRealImag(plan, instruction, instruction_index, output_id),
        .expm1, .cbrt, .not_, .is_finite, .round_nearest_afz, .round_nearest_even, .popcnt, .count_leading_zeros => elementwise_lowering.validateUnary(plan, instruction, instruction_index, output_id),
        .transpose => indexing_lowering.validateTranspose(instruction, instruction_index, output_id),
        .broadcast_in_dim => indexing_lowering.validateBroadcastInDim(instruction, instruction_index, output_id),
        .slice => indexing_lowering.validateSlice(instruction, instruction_index, output_id),
        .dynamic_slice => indexing_lowering.validateDynamicSlice(instruction, instruction_index, output_id),
        .dynamic_update_slice => indexing_lowering.validateDynamicUpdateSlice(instruction, instruction_index, output_id),
        .pad => indexing_lowering.validatePad(plan, instruction, instruction_index, output_id),
        .concatenate => indexing_lowering.validateConcatenate(instruction, instruction_index, output_id),
        .gather => indexing_lowering.validateGather(plan, instruction, instruction_index, output_id),
        .scatter => indexing_lowering.validateScatter(plan, instruction, instruction_index, output_id),
        .tuple => stateful_lowering.validateTuple(plan, instruction, instruction_index, output_id),
        .get_tuple_element => stateful_lowering.validateGetTupleElement(plan, instruction, instruction_index, output_id),
        .sort => indexing_lowering.validateSort(instruction, instruction_index, output_id),
        .top_k => indexing_lowering.validateTopK(plan, instruction, instruction_index, output_id),
        .dot_general => linalg_lowering.validateDotGeneral(plan, instruction, instruction_index, output_id),
        .convolution => linalg_lowering.validateConvolution(plan, instruction, instruction_index, output_id),
        .cholesky => linalg_lowering.validateCholesky(plan, instruction, instruction_index, output_id),
        .triangular_solve => linalg_lowering.validateTriangularSolve(plan, instruction, instruction_index, output_id),
        .reduce_sum, .reduce_max, .reduce_min, .reduce_and, .reduce_or => reduction_lowering.validateReduce(plan, instruction, instruction_index, output_id),
        .reduce_window_sum, .reduce_window_max => reduction_lowering.validateReduceWindow(plan, instruction, instruction_index, output_id),
        .rng_bit_generator => stateful_lowering.validateRngBitGenerator(plan, instruction, instruction_index, output_id),
        .while_ => control_flow.validateWhile(plan, instruction, instruction_index, output_id),
        .compare => elementwise_lowering.validateCompare(plan, instruction, instruction_index, output_id),
        .select => elementwise_lowering.validateSelect(plan, instruction, instruction_index, output_id),
        .clamp => elementwise_lowering.validateClamp(plan, instruction, instruction_index, output_id),
        else => null,
    };
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
