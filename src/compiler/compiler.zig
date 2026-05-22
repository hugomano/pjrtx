const ir = @import("src/compiler/ir");
const model = @import("compiler_model.zig");
const compile_options = @import("compile_options.zig");
const plan_build = @import("executable_plan_build.zig");
const plan_verify = @import("executable_plan_verify.zig");
const stablehlo_import = @import("stablehlo_import.zig");

/// Partitioner preference parsed from compile options and PJRT metadata.
pub const Partitioner = model.Partitioner;
/// Stable compile options consumed by plan construction.
pub const CompileOptions = compile_options.CompileOptions;
/// Program encoding accepted by the StableHLO frontend.
pub const ProgramFormat = model.ProgramFormat;
/// MLIR dialects observed while importing a program.
pub const Dialect = model.Dialect;
/// Decoded StableHLO operation used before executable-plan lowering.
pub const Operation = model.Operation;
/// Sharding kind attached to parameters and outputs.
pub const ShardingKind = model.ShardingKind;
/// Sharding metadata decoded from Shardy attributes.
pub const ShardingMetadata = model.ShardingMetadata;
/// Owns imported module data and releases all decoded compiler allocations.
pub const ModuleAnalysis = model.ModuleAnalysis;
/// Errors emitted by StableHLO import and analysis.
pub const AnalyzeError = model.AnalyzeError;
/// Sharding plan metadata consumed by runtime and backend executable plans.
pub const ShardingPlan = ir.ShardingPlan;
/// Tensor or structured value descriptor in a PjRTx executable plan.
pub const Value = ir.Value;
/// Stable numeric identifier for a plan value.
pub const ValueId = ir.ValueId;
/// Role assigned to a value in an executable plan.
pub const ValueRole = ir.ValueRole;
/// Operation kind used by backend-neutral executable-plan instructions.
pub const PlanInstructionKind = ir.PlanInstructionKind;
/// Backend-neutral operation plus metadata emitted by the compiler.
pub const PlanInstruction = ir.PlanInstruction;
/// Compiler-owned executable plan passed to runtime/backend layers.
pub const ExecutablePlan = ir.ExecutablePlan;
/// Errors emitted while checking executable-plan invariants before runtime use.
pub const VerifyError = plan_verify.VerifyError;

/// Parses textual PJRT compile options from a streaming reader.
pub const parseTextCompileOptionsFromReader = compile_options.parseTextCompileOptionsFromReader;
/// Parses textual PJRT compile options from an in-memory string.
pub const parseTextCompileOptions = compile_options.parseTextCompileOptions;
/// Creates a bootstrap executable plan with replicated sharding.
pub const makeReplicatedPlan = plan_build.makeReplicatedPlan;
/// Lowers imported module analysis into a backend-neutral executable plan.
pub const makeExecutablePlan = plan_build.makeExecutablePlan;
/// Verifies executable-plan graph, topology, shape, donation, and sharding invariants.
pub const verifyExecutablePlan = plan_verify.verifyExecutablePlan;
/// Imports StableHLO/VHLO program bytes and records compiler analysis.
pub const analyzeProgramFromReader = stablehlo_import.analyzeProgramFromReader;

comptime {
    _ = @import("compiler_plan_tests.zig");
    _ = @import("stablehlo_import_tests.zig");
}
