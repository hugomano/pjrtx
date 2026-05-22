const ir = @import("src/compiler/ir");

const control_flow = @import("buffer_control_flow.zig");

/// Builds control-flow fast-path methods for the typed backend buffer owner.
pub fn Typed(comptime Buffer: type) type {
    return struct {
        /// Runs the while-compare-add backend fast path.
        pub fn whileF32CompareAdd(state: Buffer, limit: Buffer, step: Buffer, compare_direction: ir.CompareOp, update_op: ir.ElementwiseBinaryOp, output_dims: []const i64, max_iterations: u64) control_flow.Error!?Buffer { return control_flow.whileF32CompareAdd(state, limit, step, compare_direction, update_op, output_dims, max_iterations); }
    };
}
