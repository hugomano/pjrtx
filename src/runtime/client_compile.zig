const std = @import("std");

const client_residency = @import("client_residency.zig");
const compile_pipeline = @import("compile_pipeline.zig");
const executable_fingerprint = @import("executable_fingerprint.zig");
const executable_mod = @import("executable.zig");

const CompiledExecutable = executable_mod.CompiledExecutable;
const ExecutablePlan = @import("src/compiler/ir").ExecutablePlan;

/// Program bytes and compile options accepted by the runtime compile boundary.
pub const Program = compile_pipeline.Program;

/// Errors surfaced by runtime compilation before PJRT ABI translation.
pub const Error = compile_pipeline.Error;

/// Compiles program bytes into a resident executable plan and backend residency.
pub fn compile(
    client: anytype,
    allocator: std.mem.Allocator,
    program: Program,
    diagnostics: *std.Io.Writer,
) Error!CompiledExecutable {
    var parsed_options = try compile_pipeline.parseOptions(allocator, client.device_memory.deviceSlice(), program.compile_options, diagnostics);
    defer parsed_options.deinit(allocator);

    var analysis = try compile_pipeline.analyzeProgram(allocator, program, diagnostics);
    defer if (analysis) |*owned_analysis| owned_analysis.deinit();

    var plan = try compile_pipeline.makeExecutablePlan(allocator, parsed_options.options, analysis);
    var plan_moved = false;
    errdefer if (!plan_moved) plan.deinit();

    try compile_pipeline.verifyPlan(allocator, plan, diagnostics);

    const optimized_program = try compile_pipeline.retainOptimizedProgram(allocator, analysis);
    errdefer allocator.free(optimized_program);

    const fingerprint = executable_fingerprint.alloc(allocator, client.backend, &client.device_memory, optimized_program, &plan) catch return error.OutOfMemory;
    errdefer allocator.free(fingerprint);

    const cache_hit = client_residency.recordCompile(&client.executable_residency_context, fingerprint) catch return error.Internal;

    const plan_ptr = allocator.create(ExecutablePlan) catch return error.OutOfMemory;
    plan_ptr.* = plan;
    plan_moved = true;
    errdefer {
        plan_ptr.deinit();
        allocator.destroy(plan_ptr);
    }

    return CompiledExecutable.initResident(allocator, client.executableContext(), plan_ptr, optimized_program, fingerprint, cache_hit, diagnostics) catch |err| switch (err) {
        error.UnsupportedRuntimeFeature => return error.UnsupportedRuntimeFeature,
        error.OutOfMemory => return error.OutOfMemory,
        error.Internal => return error.Internal,
    };
}
