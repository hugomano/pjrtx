const ir = @import("src/compiler/ir");

const buffer_mod = @import("buffer.zig");
const encoding = @import("buffer_encoding.zig");
const mlx_call = @import("mlx_call.zig");

const Buffer = buffer_mod.Buffer;
const Error = buffer_mod.Error;

/// Runs the while-compare-add backend fast path.
pub fn whileF32CompareAdd(state: Buffer, limit: Buffer, step: Buffer, compare_direction: ir.CompareOp, update_op: ir.ElementwiseBinaryOp, output_dims: []const i64, max_iterations: u64) Error!?Buffer {
    const update_code = encoding.binaryOp(update_op) orelse return error.CommandSubmissionFailed;
    return wrap(mlx_call.bufferWhileF32CompareAdd(state.handle, limit.handle, step.handle, encoding.compareOp(compare_direction), update_code, output_dims, max_iterations), error.CommandSubmissionFailed);
}

fn wrap(handle: ?mlx_call.BufferHandle, err: Error) Error!?Buffer {
    return if (handle) |ptr| Buffer{ .handle = ptr } else err;
}
