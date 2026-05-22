const std = @import("std");

const storage = @import("executable_metal_graph_graph_request_storage.zig");
const tensor = @import("executable_metal_graph_tensor.zig");

/// Writes the Metal source for a single-kernel graph request.
pub const GraphRequestWriter = struct {
    /// Emits one kernel using the request's stored expressions.
    pub fn write(request: storage.Request, writer: *std.Io.Writer, kernel_name: []const u8) !void {
        const scalar_type = tensor.metalScalarType(request.output_type) orelse unreachable;
        try writer.print(
            \\#include <metal_stdlib>
            \\using namespace metal;
            \\kernel void {s}(
            \\
        , .{kernel_name});
        for (request.input_specs, 0..) |_, index| {
            try writer.print("    device const {s}* in{d} [[buffer({d})]],\n", .{ scalar_type, index, index });
        }
        for (request.output_specs, 0..) |_, index| {
            try writer.print("    device {s}* out{d} [[buffer({d})]],\n", .{ scalar_type, index, request.input_specs.len + index });
        }
        try writer.print(
            \\    constant uint& count [[buffer({d})]],
            \\    uint elem [[thread_position_in_grid]]) {{
            \\    if (elem >= count) return;
            \\
        , .{request.input_specs.len + request.output_specs.len});
        for (request.output_specs, 0..) |_, index| {
            const output_id = request.output_ids[index];
            const expression = storage.ExpressionStore.get(request, output_id).?;
            try writer.print("    out{d}[elem] = {s};\n", .{ index, expression });
        }
        try writer.writeAll("}\n");
    }
};
