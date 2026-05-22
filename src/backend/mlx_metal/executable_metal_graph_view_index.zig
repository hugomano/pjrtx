const std = @import("std");

const ir = @import("src/compiler/ir");
const tensor = @import("executable_metal_graph_tensor.zig");

/// Builds an input-linear-index expression for broadcasted fusion views.
pub fn broadcastIndexExpression(
    allocator: std.mem.Allocator,
    input: ir.BufferDescriptor,
    output: ir.BufferDescriptor,
    broadcast_dimensions: []const i64,
    output_index_expr: []const u8,
) !?[]const u8 {
    if (broadcast_dimensions.len != input.dims.len) return null;
    if (input.dims.len > 64 or output.dims.len > 64) return null;
    var mapped_output_axes = [_]bool{false} ** 64;
    for (input.dims, broadcast_dimensions) |input_dim, output_axis_i64| {
        if (input_dim < 0 or output_axis_i64 < 0 or output_axis_i64 >= @as(i64, @intCast(output.dims.len))) return null;
        const output_axis: usize = @intCast(output_axis_i64);
        if (mapped_output_axes[output_axis]) return null;
        mapped_output_axes[output_axis] = true;
        const output_dim = output.dims[output_axis];
        if (output_dim < 0 or (input_dim != 1 and input_dim != output_dim)) return null;
    }

    var text = std.Io.Writer.Allocating.init(allocator);
    errdefer text.deinit();
    var wrote_term = false;
    for (input.dims, broadcast_dimensions, 0..) |input_dim, output_axis_i64, input_axis| {
        if (input_dim == 1) continue;
        const output_axis: usize = @intCast(output_axis_i64);
        if (wrote_term) try text.writer.writeAll(" + ");
        const output_stride = tensor.denseStride(output.dims, output_axis + 1);
        const input_stride = tensor.denseStride(input.dims, input_axis + 1);
        try text.writer.print("((({s} / {d}u) % {d}u) * {d}u)", .{
            output_index_expr,
            output_stride,
            @as(u64, @intCast(output.dims[output_axis])),
            input_stride,
        });
        wrote_term = true;
    }
    if (!wrote_term) try text.writer.writeAll("0");
    return try text.toOwnedSlice();
}

/// Builds an input-linear-index expression for sliced fusion views.
pub fn sliceIndexExpression(
    allocator: std.mem.Allocator,
    input: ir.BufferDescriptor,
    output: ir.BufferDescriptor,
    starts: []const i64,
    limits: []const i64,
    strides: []const i64,
    output_index_expr: []const u8,
) !?[]const u8 {
    if (input.dims.len != output.dims.len) return null;
    if (starts.len != input.dims.len or limits.len != input.dims.len or strides.len != input.dims.len) return null;
    if (input.dims.len > 64) return null;
    for (input.dims, output.dims, starts, limits, strides) |input_dim, output_dim, start, limit, stride| {
        if (input_dim < 0 or output_dim < 0 or start < 0 or limit < start or limit > input_dim or stride <= 0) return null;
        if (@divTrunc(limit - start + stride - 1, stride) != output_dim) return null;
    }

    var text = std.Io.Writer.Allocating.init(allocator);
    errdefer text.deinit();
    for (output.dims, starts, strides, 0..) |output_dim, start, stride, axis| {
        if (axis != 0) try text.writer.writeAll(" + ");
        const output_stride = tensor.denseStride(output.dims, axis + 1);
        const input_stride = tensor.denseStride(input.dims, axis + 1);
        try text.writer.print("(({d}u + (({s} / {d}u) % {d}u) * {d}u) * {d}u)", .{
            @as(u64, @intCast(start)),
            output_index_expr,
            output_stride,
            @as(u64, @intCast(output_dim)),
            @as(u64, @intCast(stride)),
            input_stride,
        });
    }
    return try text.toOwnedSlice();
}
