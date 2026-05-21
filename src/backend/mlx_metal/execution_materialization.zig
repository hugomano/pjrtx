const std = @import("std");

const buffer_mod = @import("buffer.zig");
const profiling_mod = @import("profiling.zig");
const program_mod = @import("program.zig");
const types = @import("execution_types.zig");
const values_mod = @import("execution_values.zig");

const BufferHandle = types.BufferHandle;
const Error = types.Error;
const ValueBindings = values_mod.ValueBindings;

/// Evaluates scheduled MLX materialization boundaries without reading values to host.
pub const MaterializationBoundaryEval = struct {
    program: *const program_mod.Program,
    values: *ValueBindings,

    /// Evaluates a contiguous materialization-boundary range from the backend program schedule.
    pub fn run(self: MaterializationBoundaryEval, first_boundary: usize, boundary_count: usize) Error!void {
        const program = self.program;
        const value_handles = self.values.handles;
        if (first_boundary > program.materialization_boundaries.len) return error.CommandSubmissionFailed;
        if (boundary_count > program.materialization_boundaries.len - first_boundary) return error.CommandSubmissionFailed;

        var stack_handles: [8]BufferHandle = undefined;
        if (boundary_count <= stack_handles.len) {
            var count: usize = 0;
            for (program.materialization_boundaries[first_boundary..][0..boundary_count]) |boundary| {
                const value_index = boundary.value_id.index;
                if (value_index >= value_handles.len) {
                    self.traceFailure("out_of_range", value_index, boundary.reason);
                    return error.CommandSubmissionFailed;
                }
                stack_handles[count] = value_handles[value_index] orelse {
                    self.traceFailure("missing_handle", value_index, boundary.reason);
                    return error.CommandSubmissionFailed;
                };
                count += 1;
            }
            buffer_mod.Opaque.evalMany(stack_handles[0..count]) catch {
                for (stack_handles[0..count]) |handle| {
                    buffer_mod.Opaque.eval(handle) catch |err| {
                        self.traceFailure(@errorName(err), std.math.maxInt(usize), .backend_requirement);
                        return err;
                    };
                }
            };
            return;
        }

        for (program.materialization_boundaries[first_boundary..][0..boundary_count]) |boundary| {
            const value_index = boundary.value_id.index;
            if (value_index >= value_handles.len) {
                self.traceFailure("out_of_range", value_index, boundary.reason);
                return error.CommandSubmissionFailed;
            }
            const handle = value_handles[value_index] orelse {
                self.traceFailure("missing_handle", value_index, boundary.reason);
                return error.CommandSubmissionFailed;
            };
            buffer_mod.Opaque.eval(handle) catch |err| {
                self.traceFailure(@errorName(err), value_index, boundary.reason);
                return err;
            };
        }
    }

    fn traceFailure(_: MaterializationBoundaryEval, detail: []const u8, value_index: usize, reason: program_mod.MaterializationReason) void {
        profiling_mod.writeMaterializationFailure(.{
            .detail = detail,
            .value_index = value_index,
            .reason = @tagName(reason),
        });
    }
};
