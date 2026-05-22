const ir = @import("src/compiler/ir");

const buffer_handle = @import("buffer_handle.zig");
const encoding = @import("buffer_encoding.zig");
const mlx_call = @import("mlx_call.zig");

pub const Error = buffer_handle.Error;

/// Runs the while-compare-add backend fast path.
pub fn whileF32CompareAdd(state: anytype, limit: @TypeOf(state), step: @TypeOf(state), compare_direction: ir.CompareOp, update_op: ir.ElementwiseBinaryOp, output_dims: []const i64, max_iterations: u64) Error!?@TypeOf(state) {
    const update_code = encoding.binaryOp(update_op) orelse return error.CommandSubmissionFailed;
    return buffer_handle.wrap(@TypeOf(state), mlx_call.bufferWhileF32CompareAdd(state.handle, limit.handle, step.handle, encoding.compareOp(compare_direction), update_code, output_dims, max_iterations), error.CommandSubmissionFailed);
}
