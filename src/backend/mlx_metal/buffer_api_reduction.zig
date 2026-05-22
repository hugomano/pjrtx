const ir = @import("src/compiler/ir");

const pair_mod = @import("buffer_pair.zig");
const reduction = @import("buffer_reduction.zig");

/// Builds reduction methods for the typed backend buffer owner.
pub fn Typed(comptime Buffer: type) type {
    const Pair = pair_mod.Pair(Buffer);
    return struct {
        /// Runs a reduction operation on this MLX/Metal buffer.
        pub fn reduce(self: Buffer, op: ir.PlanInstructionKind, dimensions: []const i64, output_dims: []const i64) reduction.Error!?Buffer { return reduction.reduce(self, op, dimensions, output_dims); }
        /// Runs max-reduction and returns values plus indices.
        pub fn reduceMaxWithIndices(values: Buffer, indices: Buffer, dimensions: []const i64, output_dims: []const i64) reduction.Error!?Pair { const pair = (try reduction.reduceMaxWithIndices(values, indices, dimensions, output_dims)) orelse return null; return .{ .first = pair.first, .second = pair.second }; }
        /// Runs a windowed reduction operation on this MLX/Metal buffer.
        pub fn reduceWindow(self: Buffer, op: ir.PlanInstructionKind, window_dimensions: []const i64, window_strides: []const i64, base_dilations: []const i64, window_dilations: []const i64, padding_low: []const i64, padding_high: []const i64, output_dims: []const i64) reduction.Error!?Buffer { return reduction.reduceWindow(self, op, window_dimensions, window_strides, base_dilations, window_dilations, padding_low, padding_high, output_dims); }
        /// Runs windowed max-reduction and returns values plus indices.
        pub fn reduceWindowMaxWithIndices(values: Buffer, indices: Buffer, window_dimensions: []const i64, window_strides: []const i64, base_dilations: []const i64, window_dilations: []const i64, padding_low: []const i64, padding_high: []const i64, output_dims: []const i64) reduction.Error!?Pair { const pair = (try reduction.reduceWindowMaxWithIndices(values, indices, window_dimensions, window_strides, base_dilations, window_dilations, padding_low, padding_high, output_dims)) orelse return null; return .{ .first = pair.first, .second = pair.second }; }
    };
}
