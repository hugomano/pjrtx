const std = @import("std");
const ir = @import("src/compiler/ir");
const model = @import("compiler_model.zig");
const compile_options = @import("compile_options.zig");
const plan_build = @import("executable_plan_build.zig");
const plan_verify = @import("executable_plan_verify.zig");
const stablehlo_import = @import("stablehlo_import.zig");

const CompileOptions = compile_options.CompileOptions;
const parseTextCompileOptions = compile_options.parseTextCompileOptions;
const ModuleAnalysis = model.ModuleAnalysis;
const ShardingKind = model.ShardingKind;
const PlanInstructionKind = model.PlanInstructionKind;
const ValueId = model.ValueId;
const ValueRole = model.ValueRole;
const PlanInstruction = model.PlanInstruction;
const ExecutablePlan = model.ExecutablePlan;
const makeReplicatedPlan = plan_build.makeReplicatedPlan;
const makeExecutablePlan = plan_build.makeExecutablePlan;
const verifyExecutablePlan = plan_verify.verifyExecutablePlan;
const analyzeProgramFromReader = stablehlo_import.analyzeProgramFromReader;

test "executable plan records region summaries for region-bodied ops" {
    const module_text =
        \\module {
        \\  func.func @main(%arg0: tensor<2x3xf32>) -> tensor<2xf32> {
        \\    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        \\    %0 = "stablehlo.reduce"(%arg0, %cst) ({
        \\    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
        \\      %sum = stablehlo.add %lhs, %rhs : tensor<f32>
        \\      stablehlo.return %sum : tensor<f32>
        \\    }) {dimensions = array<i64: 1>} : (tensor<2x3xf32>, tensor<f32>) -> tensor<2xf32>
        \\    return %0 : tensor<2xf32>
        \\  }
        \\}
    ;
    var reader: std.Io.Reader = .fixed(module_text);
    var diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer diagnostics.deinit();

    var analysis = try analyzeProgramFromReader(std.testing.allocator, "mlir", &reader, &diagnostics.writer);
    defer analysis.deinit();
    var plan = try makeExecutablePlan(std.testing.allocator, .{}, analysis);
    defer plan.deinit();

    try std.testing.expectEqual(@as(usize, 1), plan.instructions.len);
    try std.testing.expectEqual(PlanInstructionKind.reduce_sum, plan.instructions[0].kind);
    try std.testing.expectEqual(@as(usize, 1), plan.instructions[0].region_ids.len);
    const region = plan.regions[plan.instructions[0].region_ids[0].index];
    try std.testing.expectEqual(ir.RegionKind.reducer, region.kind);
    try std.testing.expectEqual(@as(usize, 0), region.parent_instruction_index);
    try std.testing.expectEqual(@as(usize, 2), region.argument_descriptors.len);
    try std.testing.expectEqual(@as(usize, 1), region.return_descriptors.len);
    try std.testing.expectEqual(ir.BufferType.f32, region.return_descriptors[0].element_type);
}

test "executable plan records while region subprogram summaries" {
    const module_text =
        \\module {
        \\  func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
        \\    %0 = stablehlo.while(%iterArg = %arg0) : tensor<f32>
        \\     cond {
        \\      %limit = stablehlo.constant dense<4.000000e+00> : tensor<f32>
        \\      %pred = stablehlo.compare LT, %iterArg, %limit, FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
        \\      stablehlo.return %pred : tensor<i1>
        \\    } do {
        \\      %one = stablehlo.constant dense<1.000000e+00> : tensor<f32>
        \\      %next_state = stablehlo.add %iterArg, %one : tensor<f32>
        \\      stablehlo.return %next_state : tensor<f32>
        \\    }
        \\    return %0 : tensor<f32>
        \\  }
        \\}
    ;
    var reader: std.Io.Reader = .fixed(module_text);
    var diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer diagnostics.deinit();

    var analysis = try analyzeProgramFromReader(std.testing.allocator, "mlir", &reader, &diagnostics.writer);
    defer analysis.deinit();
    var plan = try makeExecutablePlan(std.testing.allocator, .{}, analysis);
    defer plan.deinit();

    try std.testing.expectEqual(@as(usize, 1), plan.instructions.len);
    const while_instruction = plan.instructions[0];
    try std.testing.expectEqual(PlanInstructionKind.while_, while_instruction.kind);
    try std.testing.expectEqual(@as(usize, 1), while_instruction.inputs.len);
    try std.testing.expectEqual(@as(usize, 1), while_instruction.outputs.len);
    try std.testing.expectEqual(@as(usize, 2), while_instruction.region_ids.len);

    const cond = plan.regions[while_instruction.region_ids[0].index];
    try std.testing.expectEqual(ir.RegionKind.while_cond, cond.kind);
    try std.testing.expectEqual(@as(usize, 0), cond.parent_instruction_index);
    try std.testing.expectEqual(@as(usize, 3), cond.values.len);
    try std.testing.expectEqual(ir.RegionValueRole.argument, cond.values[0].role);
    try std.testing.expectEqual(ir.RegionValueRole.constant, cond.values[1].role);
    try std.testing.expect(cond.values[1].literal != null);
    try std.testing.expectEqual(ir.RegionValueRole.instruction_result, cond.values[2].role);
    try std.testing.expectEqual(@as(usize, 1), cond.argument_descriptors.len);
    try std.testing.expectEqual(@as(usize, 1), cond.instructions.len);
    try std.testing.expectEqual(PlanInstructionKind.compare, cond.instructions[0].kind);
    try std.testing.expectEqual(ir.CompareOp.lt, cond.instructions[0].compare_direction.?);
    try std.testing.expectEqual(@as(usize, 2), cond.instructions[0].inputs.len);
    try std.testing.expectEqual(@as(u32, 0), cond.instructions[0].inputs[0].index);
    try std.testing.expectEqual(@as(u32, 1), cond.instructions[0].inputs[1].index);
    try std.testing.expectEqual(@as(u32, 2), cond.instructions[0].outputs[0].index);
    try std.testing.expectEqual(@as(usize, 1), cond.return_descriptors.len);
    try std.testing.expectEqual(ir.BufferType.pred, cond.return_descriptors[0].element_type);
    try std.testing.expectEqual(@as(usize, 1), cond.terminator_operands.len);
    try std.testing.expectEqual(@as(u32, 2), cond.terminator_operands[0].index);
    try std.testing.expectEqual(@as(usize, 1), cond.terminator_operand_descriptors.len);

    const body = plan.regions[while_instruction.region_ids[1].index];
    try std.testing.expectEqual(ir.RegionKind.while_body, body.kind);
    try std.testing.expectEqual(@as(usize, 3), body.values.len);
    try std.testing.expectEqual(@as(usize, 1), body.argument_descriptors.len);
    try std.testing.expectEqual(ir.RegionValueRole.constant, body.values[1].role);
    try std.testing.expect(body.values[1].literal != null);
    try std.testing.expectEqual(@as(usize, 1), body.instructions.len);
    try std.testing.expectEqual(PlanInstructionKind.add, body.instructions[0].kind);
    try std.testing.expectEqual(@as(u32, 2), body.instructions[0].outputs[0].index);
    try std.testing.expectEqual(@as(usize, 1), body.return_descriptors.len);
    try std.testing.expectEqual(ir.BufferType.f32, body.return_descriptors[0].element_type);
    try std.testing.expectEqual(@as(usize, 1), body.terminator_operands.len);
    try std.testing.expectEqual(@as(u32, 2), body.terminator_operands[0].index);
    try std.testing.expectEqual(@as(usize, 1), body.terminator_operand_descriptors.len);
}

test "executable plan lowers convolution metadata" {
    const module_text =
        \\module {
        \\  func.func @main(%arg0: tensor<1x1x4xf32>, %arg1: tensor<1x1x2xf32>) -> tensor<1x1x3xf32> {
        \\    %0 = stablehlo.convolution(%arg0, %arg1)
        \\      dim_numbers = [b, f, 0]x[o, i, 0]->[b, f, 0],
        \\      window = {}
        \\      {batch_group_count = 1 : i64, feature_group_count = 1 : i64}
        \\      : (tensor<1x1x4xf32>, tensor<1x1x2xf32>) -> tensor<1x1x3xf32>
        \\    return %0 : tensor<1x1x3xf32>
        \\  }
        \\}
    ;
    var reader: std.Io.Reader = .fixed(module_text);
    var diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer diagnostics.deinit();

    var analysis = try analyzeProgramFromReader(std.testing.allocator, "mlir", &reader, &diagnostics.writer);
    defer analysis.deinit();
    var plan = try makeExecutablePlan(std.testing.allocator, .{}, analysis);
    defer plan.deinit();

    try std.testing.expectEqual(@as(usize, 1), plan.instructions.len);
    const instruction = plan.instructions[0];
    try std.testing.expectEqual(PlanInstructionKind.convolution, instruction.kind);
    try std.testing.expectEqualSlices(i64, &.{1}, instruction.window_strides.?);
    try std.testing.expectEqualSlices(i64, &.{0}, instruction.edge_padding_low.?);
    try std.testing.expectEqualSlices(i64, &.{0}, instruction.edge_padding_high.?);
    try std.testing.expectEqualSlices(i64, &.{1}, instruction.base_dilations.?);
    try std.testing.expectEqualSlices(i64, &.{1}, instruction.window_dilations.?);
    try std.testing.expectEqualSlices(bool, &.{false}, instruction.window_reversal.?);
    try std.testing.expectEqual(@as(?i64, 0), instruction.input_batch_dimension);
    try std.testing.expectEqual(@as(?i64, 1), instruction.input_feature_dimension);
    try std.testing.expectEqualSlices(i64, &.{2}, instruction.input_spatial_dimensions.?);
    try std.testing.expectEqual(@as(?i64, 1), instruction.kernel_input_feature_dimension);
    try std.testing.expectEqual(@as(?i64, 0), instruction.kernel_output_feature_dimension);
    try std.testing.expectEqualSlices(i64, &.{2}, instruction.kernel_spatial_dimensions.?);
    try std.testing.expectEqual(@as(?i64, 0), instruction.output_batch_dimension);
    try std.testing.expectEqual(@as(?i64, 1), instruction.output_feature_dimension);
    try std.testing.expectEqualSlices(i64, &.{2}, instruction.output_spatial_dimensions.?);
    try std.testing.expectEqual(@as(?i64, 1), instruction.feature_group_count);
    try std.testing.expectEqual(@as(?i64, 1), instruction.batch_group_count);
}

test "executable plan lowers reduce_window sum metadata" {
    const module_text =
        \\module {
        \\  func.func @main(%arg0: tensor<3xf32>) -> tensor<3xf32> {
        \\    %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        \\    %0 = "stablehlo.reduce_window"(%arg0, %cst) ({
        \\    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
        \\      %sum = stablehlo.add %lhs, %rhs : tensor<f32>
        \\      stablehlo.return %sum : tensor<f32>
        \\    }) {
        \\      base_dilations = array<i64: 1>,
        \\      padding = dense<[[1, 0]]> : tensor<1x2xi64>,
        \\      window_dimensions = array<i64: 2>,
        \\      window_dilations = array<i64: 1>,
        \\      window_strides = array<i64: 1>
        \\    } : (tensor<3xf32>, tensor<f32>) -> tensor<3xf32>
        \\    return %0 : tensor<3xf32>
        \\  }
        \\}
    ;
    var reader: std.Io.Reader = .fixed(module_text);
    var diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer diagnostics.deinit();

    var analysis = try analyzeProgramFromReader(std.testing.allocator, "mlir", &reader, &diagnostics.writer);
    defer analysis.deinit();

    var plan = try makeExecutablePlan(std.testing.allocator, .{}, analysis);
    defer plan.deinit();

    try std.testing.expectEqual(@as(usize, 1), plan.instructions.len);
    try std.testing.expectEqual(PlanInstructionKind.reduce_window_sum, plan.instructions[0].kind);
    try std.testing.expectEqualSlices(i64, &.{2}, plan.instructions[0].window_dimensions.?);
    try std.testing.expectEqualSlices(i64, &.{1}, plan.instructions[0].window_strides.?);
    try std.testing.expectEqualSlices(i64, &.{1}, plan.instructions[0].base_dilations.?);
    try std.testing.expectEqualSlices(i64, &.{1}, plan.instructions[0].window_dilations.?);
    try std.testing.expectEqualSlices(i64, &.{1}, plan.instructions[0].edge_padding_low.?);
    try std.testing.expectEqualSlices(i64, &.{0}, plan.instructions[0].edge_padding_high.?);
    try std.testing.expectEqual(@as(usize, 1), plan.instructions[0].region_ids.len);
}

test "executable plan records structured storage for tuple and complex values" {
    const module_text =
        \\module {
        \\  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> (tuple<tensor<2xf32>, tensor<2xf32>>, tensor<2xcomplex<f32>>) {
        \\    %0 = "stablehlo.tuple"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tuple<tensor<2xf32>, tensor<2xf32>>
        \\    %1 = stablehlo.complex %arg0, %arg1 : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xcomplex<f32>>
        \\    return %0, %1 : tuple<tensor<2xf32>, tensor<2xf32>>, tensor<2xcomplex<f32>>
        \\  }
        \\}
    ;
    var reader: std.Io.Reader = .fixed(module_text);
    var diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer diagnostics.deinit();

    var analysis = try analyzeProgramFromReader(std.testing.allocator, "mlir", &reader, &diagnostics.writer);
    defer analysis.deinit();
    var plan = try makeExecutablePlan(std.testing.allocator, .{}, analysis);
    defer plan.deinit();

    try std.testing.expectEqual(@as(usize, 2), plan.instructions.len);
    const tuple_value = plan.values[plan.instructions[0].outputs[0].index];
    try std.testing.expectEqual(ir.ValueStorageKind.tuple, tuple_value.storage);
    try std.testing.expectEqual(@as(usize, 2), tuple_value.elements.len);
    try std.testing.expectEqual(@as(u32, 0), tuple_value.elements[0].index);
    try std.testing.expectEqual(@as(u32, 1), tuple_value.elements[1].index);
    const complex_value = plan.values[plan.instructions[1].outputs[0].index];
    try std.testing.expectEqual(ir.ValueStorageKind.complex_pair, complex_value.storage);
    try std.testing.expectEqual(@as(usize, 2), complex_value.elements.len);
}

test "executable plan lowers tuple extraction as structured value use" {
    const module_text =
        \\module {
        \\  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
        \\    %0 = "stablehlo.tuple"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tuple<tensor<2xf32>, tensor<2xf32>>
        \\    %1 = "stablehlo.get_tuple_element"(%0) {index = 1 : i32} : (tuple<tensor<2xf32>, tensor<2xf32>>) -> tensor<2xf32>
        \\    return %1 : tensor<2xf32>
        \\  }
        \\}
    ;
    var reader: std.Io.Reader = .fixed(module_text);
    var diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer diagnostics.deinit();

    var analysis = try analyzeProgramFromReader(std.testing.allocator, "mlir", &reader, &diagnostics.writer);
    defer analysis.deinit();
    var plan = try makeExecutablePlan(std.testing.allocator, .{}, analysis);
    defer plan.deinit();

    try std.testing.expectEqual(@as(usize, 2), plan.instructions.len);
    try std.testing.expectEqual(PlanInstructionKind.tuple, plan.instructions[0].kind);
    try std.testing.expectEqual(PlanInstructionKind.get_tuple_element, plan.instructions[1].kind);
    try std.testing.expectEqual(@as(?i64, 1), plan.instructions[1].tuple_index);
    try std.testing.expectEqual(ir.ValueStorageKind.tuple, plan.values[plan.instructions[0].outputs[0].index].storage);

    var verify_diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer verify_diagnostics.deinit();
    try verifyExecutablePlan(std.testing.allocator, plan, &verify_diagnostics.writer);
}

test "executable plan lowers heavy random and structural StableHLO op shells" {
    const module_text =
        \\module {
        \\  func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
        \\    %0 = "stablehlo.reduce_precision"(%arg0) <{exponent_bits = 8 : i32, mantissa_bits = 23 : i32}> : (tensor<2x2xf32>) -> tensor<2x2xf32>
        \\    %1 = "stablehlo.cholesky"(%0) <{lower = true}> : (tensor<2x2xf32>) -> tensor<2x2xf32>
        \\    %2 = "stablehlo.triangular_solve"(%1, %arg1) <{left_side = true, lower = true, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = false}> : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
        \\    %3 = "stablehlo.custom_call"(%2) {call_target_name = "pjrtx.test"} : (tensor<2x2xf32>) -> tensor<2x2xf32>
        \\    return %3 : tensor<2x2xf32>
        \\  }
        \\}
    ;
    var reader: std.Io.Reader = .fixed(module_text);
    var diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer diagnostics.deinit();

    var analysis = try analyzeProgramFromReader(std.testing.allocator, "mlir", &reader, &diagnostics.writer);
    defer analysis.deinit();

    var plan = try makeExecutablePlan(std.testing.allocator, .{}, analysis);
    defer plan.deinit();

    try std.testing.expectEqual(@as(usize, 4), plan.instructions.len);
    try std.testing.expectEqual(PlanInstructionKind.reduce_precision, plan.instructions[0].kind);
    try std.testing.expectEqual(PlanInstructionKind.cholesky, plan.instructions[1].kind);
    try std.testing.expectEqual(@as(?bool, true), plan.instructions[1].lower);
    try std.testing.expectEqual(PlanInstructionKind.triangular_solve, plan.instructions[2].kind);
    try std.testing.expectEqual(@as(?bool, true), plan.instructions[2].triangular_left_side);
    try std.testing.expectEqual(@as(?bool, true), plan.instructions[2].triangular_lower);
    try std.testing.expectEqual(@as(?bool, false), plan.instructions[2].triangular_unit_diagonal);
    try std.testing.expectEqual(ir.TriangularSolveTranspose.no_transpose, plan.instructions[2].triangular_transpose.?);
    try std.testing.expectEqual(PlanInstructionKind.custom_call, plan.instructions[3].kind);
    try std.testing.expectEqualStrings("pjrtx.test", plan.instructions[3].custom_call_target.?);
}

test "executable plan lowers deprecated rng distribution metadata" {
    const module_text =
        \\module {
        \\  func.func @main() -> tensor<2x3xf32> {
        \\    %mean = stablehlo.constant dense<0.000000e+00> : tensor<f32>
        \\    %stddev = stablehlo.constant dense<1.000000e+00> : tensor<f32>
        \\    %shape = stablehlo.constant dense<[2, 3]> : tensor<2xi64>
        \\    %0 = "stablehlo.rng"(%mean, %stddev, %shape) <{rng_distribution = #stablehlo<rng_distribution NORMAL>}> : (tensor<f32>, tensor<f32>, tensor<2xi64>) -> tensor<2x3xf32>
        \\    return %0 : tensor<2x3xf32>
        \\  }
        \\}
    ;
    var reader: std.Io.Reader = .fixed(module_text);
    var diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer diagnostics.deinit();

    var analysis = try analyzeProgramFromReader(std.testing.allocator, "mlir", &reader, &diagnostics.writer);
    defer analysis.deinit();

    var plan = try makeExecutablePlan(std.testing.allocator, .{}, analysis);
    defer plan.deinit();

    try std.testing.expectEqual(@as(usize, 3), plan.instructions.len);
    const rng_instruction = plan.instructions[2];
    try std.testing.expectEqual(PlanInstructionKind.rng, rng_instruction.kind);
    try std.testing.expectEqual(ir.RngDistribution.normal, rng_instruction.rng_distribution.?);
    try std.testing.expectEqual(@as(usize, 2), rng_instruction.inputs.len);
    try std.testing.expectEqualSlices(i64, &.{ 2, 3 }, rng_instruction.dims.?);
}

test "analyze stablehlo text registers dialects and supported ops" {
    const module_text =
        \\module {
        \\  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
        \\    %0 = stablehlo.add %arg0, %arg1 : tensor<2xf32>
        \\    %1 = stablehlo.tanh %0 : tensor<2xf32>
        \\    return %1 : tensor<2xf32>
        \\  }
        \\}
    ;
    var reader: std.Io.Reader = .fixed(module_text);
    var diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer diagnostics.deinit();

    var analysis = try analyzeProgramFromReader(std.testing.allocator, "mlir", &reader, &diagnostics.writer);
    defer analysis.deinit();

    try std.testing.expectEqual(@as(usize, 2), analysis.num_parameters);
    try std.testing.expectEqual(@as(usize, 1), analysis.num_outputs);
    try std.testing.expectEqual(@as(usize, 2), analysis.ops.len);
    try std.testing.expectEqualStrings("add", analysis.ops[0].name);
    try std.testing.expectEqualStrings("f32", analysis.ops[0].dtype);
    try std.testing.expectEqual(@as(?usize, 1), analysis.ops[0].rank);
    try std.testing.expectEqualStrings("", diagnostics.writer.buffered());
}

test "analyze stablehlo text loads real shardy dialect" {
    const module_text =
        \\sdy.mesh @mesh = <["x"=2]>
        \\func.func @main(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> tensor<4xf32> {
        \\  %0 = stablehlo.add %arg0, %arg0 : tensor<4xf32>
        \\  return %0 : tensor<4xf32>
        \\}
    ;
    var reader: std.Io.Reader = .fixed(module_text);
    var diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer diagnostics.deinit();

    var analysis = try analyzeProgramFromReader(std.testing.allocator, "mlir", &reader, &diagnostics.writer);
    defer analysis.deinit();

    var saw_shardy = false;
    for (analysis.dialects) |dialect| {
        if (dialect == .sdy) saw_shardy = true;
    }
    try std.testing.expect(saw_shardy);
    try std.testing.expectEqual(@as(usize, 1), analysis.ops.len);
    try std.testing.expectEqualStrings("", diagnostics.writer.buffered());
}

test "analyze stablehlo text lowers optimization barrier with multiple results" {
    const module_text =
        \\sdy.mesh @mesh = <["x"=2]>
        \\func.func @main(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
        \\  %0, %1 = "stablehlo.optimization_barrier"(%arg0, %arg1) : (tensor<4xf32>, tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>)
        \\  return %0, %1 : tensor<4xf32>, tensor<4xf32>
        \\}
    ;
    var reader: std.Io.Reader = .fixed(module_text);
    var diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer diagnostics.deinit();

    var analysis = try analyzeProgramFromReader(std.testing.allocator, "mlir", &reader, &diagnostics.writer);
    defer analysis.deinit();
    var plan = try makeExecutablePlan(std.testing.allocator, .{}, analysis);
    defer plan.deinit();

    try std.testing.expectEqual(@as(usize, 1), plan.instructions.len);
    try std.testing.expectEqual(PlanInstructionKind.optimization_barrier, plan.instructions[0].kind);
    try std.testing.expectEqual(@as(usize, 2), plan.instructions[0].inputs.len);
    try std.testing.expectEqual(@as(usize, 2), plan.instructions[0].outputs.len);
    try std.testing.expectEqualStrings("", diagnostics.writer.buffered());
}

test "analyze stablehlo text reports unsupported op with location dtype rank and sharding" {
    const module_text =
        \\sdy.mesh @mesh = <["x"=2]>
        \\func.func @main(%arg0: tensor<f32>) -> tensor<4xf32> {
        \\  %0 = "stablehlo.broadcast"(%arg0) {
        \\    broadcast_sizes = array<i64: 4>,
        \\    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>
        \\  } : (tensor<f32>) -> tensor<4xf32>
        \\  return %0 : tensor<4xf32>
        \\}
    ;
    var reader: std.Io.Reader = .fixed(module_text);
    var diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer diagnostics.deinit();

    try std.testing.expectError(
        error.UnsupportedOp,
        analyzeProgramFromReader(std.testing.allocator, "mlir", &reader, &diagnostics.writer),
    );
    const text = diagnostics.writer.buffered();
    try std.testing.expect(std.mem.indexOf(u8, text, "op=broadcast") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "dtype=f32") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "rank=1") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "sharding=sdy.sharding_per_value") != null);
}

test "analyze stablehlo bytecode reports deserialization failures precisely" {
    var reader: std.Io.Reader = .fixed("\x00not-a-stablehlo-portable-artifact");
    var diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer diagnostics.deinit();

    try std.testing.expectError(
        error.InvalidStablehloModule,
        analyzeProgramFromReader(std.testing.allocator, "stablehlo_bytecode", &reader, &diagnostics.writer),
    );
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "portable artifact deserialization failed") != null);
}
