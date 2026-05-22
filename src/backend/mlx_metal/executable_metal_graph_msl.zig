const std = @import("std");

const tensor = @import("executable_metal_graph_tensor.zig");

/// Writes common MSL kernel prologues and coordinate helpers for executable graph kernels.
pub const KernelWriter = struct {
    pub fn writeSimpleKernelPrefix(writer: *std.Io.Writer, kernel_name: []const u8, scalar_type: []const u8, input_count: usize, output_count: usize) !void {
        try writer.print(
            \\#include <metal_stdlib>
            \\using namespace metal;
            \\kernel void {s}(
            \\
        , .{kernel_name});
        for (0..input_count) |index| {
            try writer.print("    device const {s}* in{d} [[buffer({d})]],\n", .{ scalar_type, index, index });
        }
        for (0..output_count) |index| {
            try writer.print("    device {s}* out{d} [[buffer({d})]],\n", .{ scalar_type, index, input_count + index });
        }
        try writer.print(
            \\    constant uint& count [[buffer({d})]],
            \\    uint elem [[thread_position_in_grid]]) {{
            \\    if (elem >= count) return;
            \\
        , .{input_count + output_count});
    }

    pub fn writeOutputCoordinates(writer: *std.Io.Writer, output_dims: []const i64) !void {
        if (output_dims.len == 0) return;
        try writer.writeAll("    uint remaining = elem;\n");
        for (output_dims, 0..) |_, axis| {
            const stride = tensor.denseStride(output_dims, axis + 1);
            if (stride == 1) {
                try writer.print("    const uint coord{d} = remaining;\n", .{axis});
            } else {
                try writer.print("    const uint coord{d} = remaining / {d}u;\n", .{ axis, stride });
                try writer.print("    remaining -= coord{d} * {d}u;\n", .{ axis, stride });
            }
        }
    }
};
