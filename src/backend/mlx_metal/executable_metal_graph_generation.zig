const std = @import("std");

const ir = @import("src/compiler/ir");
const metalcpp_call = @import("metalcpp_call.zig");
const program_mod = @import("program.zig");
const tensor = @import("executable_metal_graph_tensor.zig");

pub fn makeConcatenateStep(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize) program_mod.Error!?metalcpp_call.ExecutableStepSpec {
    if (instruction.inputs.len == 0 or instruction.outputs.len != 1) return null;
    const output = tensor.valueDescriptor(plan, instruction.outputs[0]) orelse return null;
    const concat_axis_i64 = instruction.dimension orelse return null;
    if (concat_axis_i64 < 0 or concat_axis_i64 >= @as(i64, @intCast(output.dims.len))) return null;
    const concat_axis: usize = @intCast(concat_axis_i64);
    const output_count = tensor.denseElementCount(output);
    if (output_count == 0 or output_count > std.math.maxInt(u32)) return null;
    var concat_dim: i64 = 0;
    for (instruction.inputs) |input_id| {
        const input = tensor.valueDescriptor(plan, input_id) orelse return null;
        if (input.element_type != output.element_type or input.dims.len != output.dims.len) return null;
        for (input.dims, output.dims, 0..) |input_dim, output_dim, axis| {
            if (input_dim < 0 or output_dim < 0) return null;
            if (axis == concat_axis) {
                concat_dim += input_dim;
            } else if (input_dim != output_dim) return null;
        }
    }
    if (concat_dim != output.dims[concat_axis]) return null;
    const scalar_type = tensor.metalScalarType(output.element_type) orelse return null;
    const inputs = try tensor.valueIndices(allocator, instruction.inputs);
    errdefer allocator.free(inputs);
    const outputs = try tensor.valueIndices(allocator, instruction.outputs);
    errdefer allocator.free(outputs);
    const kernel_name = try tensor.kernelName(allocator, "concatenate", instruction_index);
    errdefer allocator.free(kernel_name);
    var source = std.Io.Writer.Allocating.init(allocator);
    errdefer source.deinit();
    writeConcatenateKernel(&source.writer, kernel_name, scalar_type, plan, instruction, output, concat_axis) catch return error.OutOfMemory;
    return .{
        .kernel_name = kernel_name,
        .source = try source.toOwnedSlice(),
        .inputs = inputs,
        .outputs = outputs,
        .element_count = @intCast(output_count),
    };
}

pub fn makeIotaStep(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize) program_mod.Error!?metalcpp_call.ExecutableStepSpec {
    if (instruction.inputs.len != 0 or instruction.outputs.len != 1) return null;
    const output = tensor.valueDescriptor(plan, instruction.outputs[0]) orelse return null;
    const iota_dimension_i64 = instruction.iota_dimension orelse return null;
    if (iota_dimension_i64 < 0 or iota_dimension_i64 >= @as(i64, @intCast(output.dims.len))) return null;
    const iota_dimension: usize = @intCast(iota_dimension_i64);
    const output_count = tensor.denseElementCount(output);
    if (output_count == 0 or output_count > std.math.maxInt(u32)) return null;
    const scalar_type = tensor.metalProgramScalarType(output.element_type) orelse return null;
    const inputs = try allocator.alloc(u64, 0);
    errdefer allocator.free(inputs);
    const outputs = try tensor.valueIndices(allocator, instruction.outputs);
    errdefer allocator.free(outputs);
    const kernel_name = try tensor.kernelName(allocator, "iota", instruction_index);
    errdefer allocator.free(kernel_name);
    var source = std.Io.Writer.Allocating.init(allocator);
    errdefer source.deinit();
    writeIotaKernel(&source.writer, kernel_name, scalar_type, output, iota_dimension) catch return error.OutOfMemory;
    return .{
        .kernel_name = kernel_name,
        .source = try source.toOwnedSlice(),
        .inputs = inputs,
        .outputs = outputs,
        .element_count = @intCast(output_count),
    };
}
fn writeIotaKernel(writer: *std.Io.Writer, kernel_name: []const u8, scalar_type: []const u8, output: ir.BufferDescriptor, iota_dimension: usize) !void {
    try writer.print(
        \\#include <metal_stdlib>
        \\using namespace metal;
        \\kernel void {s}(
        \\    device {s}* out0 [[buffer(0)]],
        \\    constant uint& count [[buffer(1)]],
        \\    uint elem [[thread_position_in_grid]]) {{
        \\    if (elem >= count) return;
        \\
    , .{ kernel_name, scalar_type });
    const stride = tensor.denseStride(output.dims, iota_dimension + 1);
    const dim: u64 = @intCast(output.dims[iota_dimension]);
    try writer.print("    out0[elem] = {s}((elem / {d}u) % {d}u);\n}}\n", .{ scalar_type, stride, dim });
}

fn writeConcatenateKernel(writer: *std.Io.Writer, kernel_name: []const u8, scalar_type: []const u8, plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, output: ir.BufferDescriptor, concat_axis: usize) !void {
    try writer.print(
        \\#include <metal_stdlib>
        \\using namespace metal;
        \\kernel void {s}(
        \\
    , .{kernel_name});
    for (instruction.inputs, 0..) |_, index| {
        try writer.print("    device const {s}* in{d} [[buffer({d})]],\n", .{ scalar_type, index, index });
    }
    try writer.print(
        \\    device {s}* out0 [[buffer({d})]],
        \\    constant uint& count [[buffer({d})]],
        \\    uint elem [[thread_position_in_grid]]) {{
        \\    if (elem >= count) return;
        \\    uint remaining = elem;
        \\
    , .{ scalar_type, instruction.inputs.len, instruction.inputs.len + 1 });
    for (output.dims, 0..) |_, axis| {
        const stride = tensor.denseStride(output.dims, axis + 1);
        if (stride == 1) {
            try writer.print("    const uint coord{d} = remaining;\n", .{axis});
        } else {
            try writer.print("    const uint coord{d} = remaining / {d}u;\n", .{ axis, stride });
            try writer.print("    remaining -= coord{d} * {d}u;\n", .{ axis, stride });
        }
    }
    var offset: u64 = 0;
    for (instruction.inputs, 0..) |input_id, input_index| {
        const input = tensor.valueDescriptor(plan, input_id) orelse unreachable;
        const input_dim: u64 = @intCast(input.dims[concat_axis]);
        const branch = if (input_index == 0) "if" else "else if";
        try writer.print("    {s} (coord{d} < {d}u) {{\n", .{ branch, concat_axis, offset + input_dim });
        try writer.writeAll("        uint input_index = 0;\n");
        for (input.dims, 0..) |_, axis| {
            const stride = tensor.denseStride(input.dims, axis + 1);
            if (axis == concat_axis) {
                try writer.print("        input_index += (coord{d} - {d}u) * {d}u;\n", .{ axis, offset, stride });
            } else {
                try writer.print("        input_index += coord{d} * {d}u;\n", .{ axis, stride });
            }
        }
        try writer.print("        out0[elem] = in{d}[input_index];\n        return;\n    }}\n", .{input_index});
        offset += input_dim;
    }
    try writer.writeAll("}\n");
}
