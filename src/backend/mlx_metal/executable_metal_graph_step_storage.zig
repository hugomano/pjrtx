const std = @import("std");

const metalcpp_call = @import("metalcpp_call.zig");
const program_mod = @import("program.zig");

/// Owns executable Metal graph step storage cleanup and append semantics.
pub const StepStorage = struct {
    pub fn attachReleaseValues(allocator: std.mem.Allocator, step: *metalcpp_call.ExecutableStepSpec, release_values: []const u64) !void {
        step.release_values = try allocator.dupe(u64, release_values);
    }

    pub fn append(allocator: std.mem.Allocator, steps: *std.ArrayList(metalcpp_call.ExecutableStepSpec), step: metalcpp_call.ExecutableStepSpec) program_mod.Error!void {
        steps.append(allocator, step) catch |err| {
            deinitStep(allocator, step);
            return err;
        };
    }

    pub fn deinitSteps(allocator: std.mem.Allocator, steps: []metalcpp_call.ExecutableStepSpec) void {
        for (steps) |step| deinitStep(allocator, step);
    }

    pub fn deinitStep(allocator: std.mem.Allocator, step: metalcpp_call.ExecutableStepSpec) void {
        allocator.free(step.kernel_name);
        allocator.free(step.source);
        allocator.free(step.inputs);
        allocator.free(step.outputs);
        allocator.free(step.release_values);
    }
};
