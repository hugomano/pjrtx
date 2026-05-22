const ir = @import("src/compiler/ir");

const opaque_ref = @import("buffer_opaque_ref.zig");

const Error = opaque_ref.Error;
const maybeHandle = opaque_ref.maybeHandle;
const ref = opaque_ref.ref;
const refs = opaque_ref.refs;

/// Runs the while-compare-add backend fast path over opaque handles.
pub fn whileF32CompareAdd(state: *anyopaque, limit: *anyopaque, step: *anyopaque, compare_direction: ir.CompareOp, update_op: ir.ElementwiseBinaryOp, output_dims: []const i64, max_iterations: u64) Error!?*anyopaque { return maybeHandle(try opaque_ref.Ref.whileF32CompareAdd(ref(state), ref(limit), ref(step), compare_direction, update_op, output_dims, max_iterations)); }
