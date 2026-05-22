const std = @import("std");

const storage = @import("executable_metal_graph_fusion_request_storage.zig");
const tensor = @import("executable_metal_graph_tensor.zig");

/// Writes the Metal source for a generated fusion request.
pub const FusionRequestWriter = struct {
    /// Emits one fusion kernel using the request's stored expressions.
    pub fn write(request: storage.Request, allocator: std.mem.Allocator, writer: *std.Io.Writer, kernel_name: []const u8) !void {
        try writeHeader(request, writer, kernel_name);
        for (request.local_statements.items) |statement| {
            try writer.writeAll(statement);
        }
        try writeOutputs(request, allocator, writer);
        try writer.writeAll("}\n");
    }

    fn writeHeader(request: storage.Request, writer: *std.Io.Writer, kernel_name: []const u8) !void {
        try writer.print(
            \\#include <metal_stdlib>
            \\using namespace metal;
            \\kernel void {s}(
            \\
        , .{kernel_name});
        for (request.input_specs, 0..) |input_spec, input_index| {
            const scalar_type = tensor.metalSpecScalarType(input_spec.dtype) orelse unreachable;
            try writer.print("    device const {s}* in{d} [[buffer({d})]],\n", .{ scalar_type, input_index, input_index });
        }
        for (request.output_specs, 0..) |output_spec, output_index| {
            const scalar_type = tensor.metalSpecScalarType(output_spec.dtype) orelse unreachable;
            try writer.print("    device {s}* out{d} [[buffer({d})]],\n", .{ scalar_type, output_index, request.input_values.len + output_index });
        }
        try writer.print(
            \\    constant uint& count [[buffer({d})]],
            \\    uint elem [[thread_position_in_grid]]) {{
            \\    if (elem >= count) return;
            \\
        , .{request.input_values.len + request.output_values.len});
    }

    fn writeOutputs(request: storage.Request, allocator: std.mem.Allocator, writer: *std.Io.Writer) !void {
        for (request.output_values, 0..) |output_id, output_index| {
            const output_type = tensor.metalSpecScalarType(request.output_specs[output_index].dtype) orelse unreachable;
            const output_count = request.output_counts[output_index];
            const output_expr = storage.ExpressionStore.get(request, output_id).?;
            const rendered = try output_expr.at(allocator, "elem");
            defer allocator.free(rendered);
            if (output_count == @as(usize, @intCast(request.element_count))) {
                try writer.print("    out{d}[elem] = {s}(({s}));\n", .{ output_index, output_type, rendered });
            } else {
                try writer.print("    if (elem < {d}u) out{d}[elem] = {s}(({s}));\n", .{ @as(u64, @intCast(output_count)), output_index, output_type, rendered });
            }
        }
    }
};
