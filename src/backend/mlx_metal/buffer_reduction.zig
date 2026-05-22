const ir = @import("src/compiler/ir");

const buffer_handle = @import("buffer_handle.zig");
const encoding = @import("buffer_encoding.zig");
const mlx_call = @import("mlx_call.zig");

pub const Error = buffer_handle.Error;

/// Runs a reduction operation on this MLX/Metal buffer.
pub fn reduce(src: anytype, op: ir.PlanInstructionKind, dimensions: []const i64, output_dims: []const i64) Error!?@TypeOf(src) {
    const code = encoding.reduceOp(op) orelse return error.CommandSubmissionFailed;
    return buffer_handle.wrap(@TypeOf(src), mlx_call.bufferReduce(src.handle, code, dimensions, output_dims), error.CommandSubmissionFailed);
}

/// Runs max-reduction and returns values plus indices.
pub fn reduceMaxWithIndices(values: anytype, indices: @TypeOf(values), dimensions: []const i64, output_dims: []const i64) Error!?buffer_handle.Pair(@TypeOf(values)) {
    const pair = mlx_call.bufferReduceMaxWithIndices(values.handle, indices.handle, dimensions, output_dims) orelse return error.CommandSubmissionFailed;
    return .{
        .first = .{ .handle = pair.first },
        .second = .{ .handle = pair.second },
    };
}

/// Runs a windowed reduction operation on this MLX/Metal buffer.
pub fn reduceWindow(src: anytype, op: ir.PlanInstructionKind, window_dimensions: []const i64, window_strides: []const i64, base_dilations: []const i64, window_dilations: []const i64, padding_low: []const i64, padding_high: []const i64, output_dims: []const i64) Error!?@TypeOf(src) {
    const code = encoding.reduceWindowOp(op) orelse return error.CommandSubmissionFailed;
    return buffer_handle.wrap(@TypeOf(src), mlx_call.bufferReduceWindow(src.handle, code, window_dimensions, window_strides, base_dilations, window_dilations, padding_low, padding_high, output_dims), error.CommandSubmissionFailed);
}

/// Runs windowed max-reduction and returns values plus indices.
pub fn reduceWindowMaxWithIndices(values: anytype, indices: @TypeOf(values), window_dimensions: []const i64, window_strides: []const i64, base_dilations: []const i64, window_dilations: []const i64, padding_low: []const i64, padding_high: []const i64, output_dims: []const i64) Error!?buffer_handle.Pair(@TypeOf(values)) {
    const pair = mlx_call.bufferReduceWindowMaxWithIndices(values.handle, indices.handle, window_dimensions, window_strides, base_dilations, window_dilations, padding_low, padding_high, output_dims) orelse return error.CommandSubmissionFailed;
    return .{
        .first = .{ .handle = pair.first },
        .second = .{ .handle = pair.second },
    };
}
