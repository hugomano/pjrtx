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

test "compile options preserve replicas partitions shardy and assignment" {
    const options = try parseTextCompileOptions(
        std.testing.allocator,
        "replicas=2; partitions=2; use_shardy=true; assignment=0,1,2,3",
    );
    defer std.testing.allocator.free(options.device_assignment);

    try std.testing.expectEqual(@as(i32, 2), options.num_replicas);
    try std.testing.expectEqual(@as(i32, 2), options.num_partitions);
    try std.testing.expect(options.use_shardy_partitioner);
    try std.testing.expectEqual(@as(usize, 4), options.numDevices());
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 2, 3 }, options.device_assignment);
}

test "replicated executable plan has per-value sharding metadata" {
    var plan = try makeReplicatedPlan(std.testing.allocator, .{ .num_partitions = 4 }, 2, 1);
    defer plan.deinit();

    try std.testing.expectEqual(@as(usize, 4), plan.options.numDevices());
    try std.testing.expectEqual(@as(usize, 2), plan.parameter_shardings.len);
    try std.testing.expectEqual(ShardingKind.replicated, plan.output_shardings[0].kind);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 2, 3 }, plan.output_shardings[0].device_assignment);
    try std.testing.expectEqual(@as(usize, 3), plan.values.len);
    try std.testing.expectEqual(ValueRole.parameter, plan.values[0].role);
    try std.testing.expectEqual(ValueRole.parameter, plan.values[1].role);
    try std.testing.expectEqual(ValueRole.instruction_result, plan.values[2].role);
    try std.testing.expectEqual(@as(usize, 1), plan.instructions.len);
    try std.testing.expectEqual(PlanInstructionKind.copy_arg0, plan.instructions[0].kind);
    try std.testing.expectEqual(@as(u32, 0), plan.instructions[0].inputs[0].index);
    try std.testing.expectEqual(@as(u32, 2), plan.instructions[0].outputs[0].index);

    var diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer diagnostics.deinit();
    try verifyExecutablePlan(std.testing.allocator, plan, &diagnostics.writer);
}

test "executable plan values carry parameter and result descriptors" {
    const module_text =
        \\module {
        \\  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
        \\    %0 = stablehlo.tanh %arg0 : tensor<4xf32>
        \\    return %0 : tensor<4xf32>
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

    try std.testing.expectEqual(@as(usize, 2), plan.values.len);
    try std.testing.expectEqual(ir.BufferType.f32, plan.values[0].descriptor.element_type);
    try std.testing.expectEqual(ir.BufferType.f32, plan.values[1].descriptor.element_type);
    try std.testing.expectEqualSlices(i64, &.{4}, plan.values[0].descriptor.dims);
    try std.testing.expectEqualSlices(i64, &.{4}, plan.values[1].descriptor.dims);
}

test "executable plan supports parameter alias returns without extra instructions" {
    const module_text =
        \\module {
        \\  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
        \\    return %arg0 : tensor<4xf32>
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

    try std.testing.expectEqual(@as(usize, 0), plan.instructions.len);
    try std.testing.expectEqual(@as(usize, 1), plan.output_ids.len);
    try std.testing.expectEqual(@as(u32, 0), plan.output_ids[0].index);

    var verify_diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer verify_diagnostics.deinit();
    try verifyExecutablePlan(std.testing.allocator, plan, &verify_diagnostics.writer);
}

test "executable plan imports function input output alias donation metadata" {
    const module_text =
        \\module {
        \\  func.func @main(%arg0: tensor<4xf32> {tf.aliasing_output = 0 : i32}) -> tensor<4xf32> {
        \\    %0 = stablehlo.tanh %arg0 : tensor<4xf32>
        \\    return %0 : tensor<4xf32>
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

    try std.testing.expectEqual(@as(usize, 1), plan.output_aliases.len);
    try std.testing.expectEqual(@as(u32, 0), plan.output_aliases[0].output_index);
    try std.testing.expectEqual(@as(u32, 0), plan.output_aliases[0].parameter_index);
    try std.testing.expectEqual(ir.OutputAliasKind.donation, plan.output_aliases[0].kind);
    try std.testing.expectEqualSlices(u32, &.{0}, plan.donated_parameter_indices);

    var verify_diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer verify_diagnostics.deinit();
    try verifyExecutablePlan(std.testing.allocator, plan, &verify_diagnostics.writer);
}

test "executable plan verifier rejects unknown value references with diagnostics" {
    var plan = try makeReplicatedPlan(std.testing.allocator, .{}, 1, 1);
    defer plan.deinit();
    std.testing.allocator.free(plan.instructions[0].inputs);
    plan.instructions[0].inputs = try std.testing.allocator.dupe(ValueId, &.{.{ .index = 99 }});

    var diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer diagnostics.deinit();
    try std.testing.expectError(error.InvalidExecutablePlan, verifyExecutablePlan(std.testing.allocator, plan, &diagnostics.writer));
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "pass=pjrtx-plan-verify") != null);
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "instruction input references an unknown value") != null);
}

test "executable plan ignores non-entry helper function parameters" {
    const module_text =
        \\module {
        \\  func.func @helper(%arg0: tensor<4xf32>) -> tensor<4xf32> {
        \\    %0 = stablehlo.negate %arg0 : tensor<4xf32>
        \\    return %0 : tensor<4xf32>
        \\  }
        \\  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
        \\    %0 = stablehlo.tanh %arg0 : tensor<4xf32>
        \\    return %0 : tensor<4xf32>
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

    try std.testing.expectEqual(@as(usize, 1), plan.parameter_shardings.len);
    var verify_diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer verify_diagnostics.deinit();
    try verifyExecutablePlan(std.testing.allocator, plan, &verify_diagnostics.writer);
}

test "executable plan ignores GSPMD mhlo sharding metadata" {
    const module_text =
        \\module {
        \\  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
        \\    %0 = stablehlo.add %arg0, %arg0 {mhlo.sharding = "{replicated}"} : tensor<4xf32>
        \\    return %0 : tensor<4xf32>
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

    try std.testing.expectEqualStrings("unspecified", analysis.ops[0].sharding);
    try std.testing.expectEqual(ShardingKind.replicated, plan.parameter_shardings[0].kind);
    try std.testing.expectEqual(ShardingKind.replicated, plan.output_shardings[0].kind);
}

test "executable plan verifier rejects invalid donation aliases" {
    var plan = try makeReplicatedPlan(std.testing.allocator, .{}, 1, 1);
    defer plan.deinit();
    plan.donated_parameter_indices = try std.testing.allocator.dupe(u32, &.{1});

    var diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer diagnostics.deinit();
    try std.testing.expectError(error.InvalidExecutablePlan, verifyExecutablePlan(std.testing.allocator, plan, &diagnostics.writer));
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "feature=donation-alias") != null);
}

test "executable plan verifier rejects shape type mismatch with diagnostics" {
    var plan = try makeReplicatedPlan(std.testing.allocator, .{}, 1, 1);
    defer plan.deinit();
    plan.values[0].descriptor.element_type = .f32;
    plan.values[0].descriptor.dims = try std.testing.allocator.dupe(i64, &.{4});
    plan.values[1].descriptor.element_type = .f32;
    plan.values[1].descriptor.dims = try std.testing.allocator.dupe(i64, &.{2});

    var diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer diagnostics.deinit();
    try std.testing.expectError(error.InvalidExecutablePlan, verifyExecutablePlan(std.testing.allocator, plan, &diagnostics.writer));
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "feature=shape-type") != null);
}

test "executable plan preserves verified shardy parameter and output metadata" {
    const module_text =
        \\sdy.mesh @mesh = <["x"=2]>
        \\func.func @main(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> tensor<4xf32> {
        \\  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<4xf32>
        \\  return %0 : tensor<4xf32>
        \\}
    ;
    var reader: std.Io.Reader = .fixed(module_text);
    var diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer diagnostics.deinit();

    var analysis = try analyzeProgramFromReader(std.testing.allocator, "mlir", &reader, &diagnostics.writer);
    defer analysis.deinit();

    var plan = try makeExecutablePlan(std.testing.allocator, .{ .num_partitions = 2 }, analysis);
    defer plan.deinit();

    try std.testing.expectEqual(@as(usize, 1), plan.parameter_shardings.len);
    try std.testing.expectEqual(@as(usize, 1), plan.output_shardings.len);
    try std.testing.expectEqual(ShardingKind.partitioned, plan.parameter_shardings[0].kind);
    try std.testing.expectEqual(ShardingKind.partitioned, plan.output_shardings[0].kind);
    try std.testing.expectEqualStrings("mesh", plan.parameter_shardings[0].mesh_name);
    try std.testing.expectEqualStrings("mesh", plan.output_shardings[0].mesh_name);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1 }, plan.output_shardings[0].device_assignment);
    try std.testing.expectEqual(@as(usize, 1), plan.instructions.len);
    try std.testing.expectEqual(PlanInstructionKind.add, plan.instructions[0].kind);
}

test "executable plan lowers initial arithmetic StableHLO ops" {
    const module_text =
        \\module {
        \\  func.func @main(%arg0: tensor<4xi8>, %arg1: tensor<4xi8>) -> tensor<4xi8> {
        \\    %0 = stablehlo.subtract %arg0, %arg1 : tensor<4xi8>
        \\    %1 = stablehlo.negate %0 : tensor<4xi8>
        \\    return %1 : tensor<4xi8>
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
    try std.testing.expectEqual(PlanInstructionKind.subtract, plan.instructions[0].kind);
    try std.testing.expectEqual(PlanInstructionKind.negate, plan.instructions[1].kind);
    try std.testing.expectEqual(@as(usize, 4), plan.values.len);
    try std.testing.expectEqual(ValueRole.parameter, plan.values[0].role);
    try std.testing.expectEqual(ValueRole.parameter, plan.values[1].role);
    try std.testing.expectEqual(ValueRole.instruction_result, plan.values[2].role);
    try std.testing.expectEqual(ValueRole.instruction_result, plan.values[3].role);
    try std.testing.expectEqual(@as(u32, 0), plan.instructions[0].inputs[0].index);
    try std.testing.expectEqual(@as(u32, 1), plan.instructions[0].inputs[1].index);
    try std.testing.expectEqual(@as(u32, 2), plan.instructions[0].outputs[0].index);
    try std.testing.expectEqual(@as(u32, 2), plan.instructions[1].inputs[0].index);
    try std.testing.expectEqual(@as(u32, 3), plan.instructions[1].outputs[0].index);

    var verify_diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer verify_diagnostics.deinit();
    try verifyExecutablePlan(std.testing.allocator, plan, &verify_diagnostics.writer);
}

test "executable plan lowers f32 unary math StableHLO ops" {
    const module_text =
        \\module {
        \\  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
        \\    %0 = stablehlo.exponential %arg0 : tensor<4xf32>
        \\    %1 = stablehlo.tanh %0 : tensor<4xf32>
        \\    %2 = stablehlo.sqrt %1 : tensor<4xf32>
        \\    %3 = stablehlo.rsqrt %2 : tensor<4xf32>
        \\    return %3 : tensor<4xf32>
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
    try std.testing.expectEqual(PlanInstructionKind.exp, plan.instructions[0].kind);
    try std.testing.expectEqual(PlanInstructionKind.tanh, plan.instructions[1].kind);
    try std.testing.expectEqual(PlanInstructionKind.sqrt, plan.instructions[2].kind);
    try std.testing.expectEqual(PlanInstructionKind.rsqrt, plan.instructions[3].kind);
}

test "executable plan lowers reshape with result dimensions" {
    const module_text =
        \\module {
        \\  func.func @main(%arg0: tensor<2x2xf32>) -> tensor<4xf32> {
        \\    %0 = stablehlo.reshape %arg0 : (tensor<2x2xf32>) -> tensor<4xf32>
        \\    return %0 : tensor<4xf32>
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
    try std.testing.expectEqual(PlanInstructionKind.reshape, plan.instructions[0].kind);
    try std.testing.expectEqualSlices(i64, &.{4}, plan.instructions[0].dims.?);
}

test "executable plan lowers transpose with permutation and result dimensions" {
    const module_text =
        \\module {
        \\  func.func @main(%arg0: tensor<2x3xf32>) -> tensor<3x2xf32> {
        \\    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
        \\    return %0 : tensor<3x2xf32>
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
    try std.testing.expectEqual(PlanInstructionKind.transpose, plan.instructions[0].kind);
    try std.testing.expectEqualSlices(i64, &.{ 3, 2 }, plan.instructions[0].dims.?);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0 }, plan.instructions[0].permutation.?);
}

test "executable plan lowers broadcast_in_dim with dimensions and result shape" {
    const module_text =
        \\module {
        \\  func.func @main(%arg0: tensor<3xf32>) -> tensor<2x3xf32> {
        \\    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<3xf32>) -> tensor<2x3xf32>
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

    try std.testing.expectEqual(@as(usize, 1), plan.instructions.len);
    try std.testing.expectEqual(PlanInstructionKind.broadcast_in_dim, plan.instructions[0].kind);
    try std.testing.expectEqualSlices(i64, &.{ 2, 3 }, plan.instructions[0].dims.?);
    try std.testing.expectEqualSlices(i64, &.{1}, plan.instructions[0].broadcast_dimensions.?);
}

test "executable plan lowers slice with bounds strides and result shape" {
    const module_text =
        \\module {
        \\  func.func @main(%arg0: tensor<3x4xf32>) -> tensor<2x2xf32> {
        \\    %0 = stablehlo.slice %arg0 [1:3, 0:4:2] : (tensor<3x4xf32>) -> tensor<2x2xf32>
        \\    return %0 : tensor<2x2xf32>
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
    try std.testing.expectEqual(PlanInstructionKind.slice, plan.instructions[0].kind);
    try std.testing.expectEqualSlices(i64, &.{ 2, 2 }, plan.instructions[0].dims.?);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0 }, plan.instructions[0].start_indices.?);
    try std.testing.expectEqualSlices(i64, &.{ 3, 4 }, plan.instructions[0].limit_indices.?);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2 }, plan.instructions[0].strides.?);
}

test "executable plan lowers concatenate with dimension and result shape" {
    const module_text =
        \\module {
        \\  func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x5xf32> {
        \\    %0 = stablehlo.concatenate %arg0, %arg1, dim = 1 : (tensor<2x2xf32>, tensor<2x3xf32>) -> tensor<2x5xf32>
        \\    return %0 : tensor<2x5xf32>
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
    try std.testing.expectEqual(PlanInstructionKind.concatenate, plan.instructions[0].kind);
    try std.testing.expectEqualSlices(i64, &.{ 2, 5 }, plan.instructions[0].dims.?);
    try std.testing.expectEqual(@as(?i64, 1), plan.instructions[0].dimension);
}

test "executable plan lowers sort with dimension and comparator direction" {
    const module_text =
        \\module {
        \\  func.func @main(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
        \\    %0 = "stablehlo.sort"(%arg0) ({
        \\    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
        \\      %pred = stablehlo.compare  LT, %lhs, %rhs,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
        \\      stablehlo.return %pred : tensor<i1>
        \\    }) {dimension = 1 : i64, is_stable = true} : (tensor<2x3xf32>) -> tensor<2x3xf32>
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

    try std.testing.expectEqual(@as(usize, 1), plan.instructions.len);
    try std.testing.expectEqual(PlanInstructionKind.sort, plan.instructions[0].kind);
    try std.testing.expectEqualSlices(i64, &.{ 2, 3 }, plan.instructions[0].dims.?);
    try std.testing.expectEqual(@as(?i64, 1), plan.instructions[0].dimension);
    try std.testing.expectEqual(ir.CompareOp.lt, plan.instructions[0].compare_direction.?);
}

