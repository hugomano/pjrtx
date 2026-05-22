const std = @import("std");

const ir = @import("src/compiler/ir");
const gather_axis = @import("executable_metal_graph_gather_axis.zig");
const metalcpp_call = @import("metalcpp_call.zig");
const msl = @import("executable_metal_graph_msl.zig");
const program_mod = @import("program.zig");
const tensor = @import("executable_metal_graph_tensor.zig");

const GatherAxis = gather_axis.Axis;

/// Builds a generated Metal gather step for single-axis gather forms.
pub fn makeGatherStep(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize) program_mod.Error!?metalcpp_call.ExecutableStepSpec {
    if (instruction.inputs.len != 2 or instruction.outputs.len != 1) return null;
    const operand = tensor.valueDescriptor(plan, instruction.inputs[0]) orelse return null;
    const indices = tensor.valueDescriptor(plan, instruction.inputs[1]) orelse return null;
    const output = tensor.valueDescriptor(plan, instruction.outputs[0]) orelse return null;
    if (operand.element_type != output.element_type) return null;
    const scalar_type = tensor.metalScalarType(output.element_type) orelse return null;
    const index_type = tensor.metalIndexScalarType(indices.element_type) orelse return null;
    const output_count = tensor.denseElementCount(output);
    const operand_count = tensor.denseElementCount(operand);
    const indices_count = tensor.denseElementCount(indices);
    if (output_count == 0 or operand_count == 0 or indices_count == 0) return null;
    if (output_count > std.math.maxInt(u32) or operand_count > std.math.maxInt(u32) or indices_count > std.math.maxInt(u32)) return null;
    const gather = gather_axis.analyze(instruction, operand, indices, output) orelse return null;

    const inputs = try tensor.valueIndices(allocator, instruction.inputs);
    errdefer allocator.free(inputs);
    const outputs = try tensor.valueIndices(allocator, instruction.outputs);
    errdefer allocator.free(outputs);
    const kernel_name = try tensor.kernelName(allocator, "gather", instruction_index);
    errdefer allocator.free(kernel_name);
    var source = std.Io.Writer.Allocating.init(allocator);
    errdefer source.deinit();
    writeGatherKernel(&source.writer, kernel_name, scalar_type, index_type, operand, indices, output, gather) catch return error.OutOfMemory;
    return .{
        .kernel_name = kernel_name,
        .source = try source.toOwnedSlice(),
        .inputs = inputs,
        .outputs = outputs,
        .element_count = @intCast(output_count),
    };
}
fn writeGatherKernel(writer: *std.Io.Writer, kernel_name: []const u8, scalar_type: []const u8, index_type: []const u8, operand: ir.BufferDescriptor, indices: ir.BufferDescriptor, output: ir.BufferDescriptor, gather: GatherAxis) !void {
    try writer.print(
        \\#include <metal_stdlib>
        \\using namespace metal;
        \\kernel void {s}(
        \\    device const {s}* in0 [[buffer(0)]],
        \\    device const {s}* in1 [[buffer(1)]],
        \\    device {s}* out0 [[buffer(2)]],
        \\    constant uint& count [[buffer(3)]],
        \\    uint elem [[thread_position_in_grid]]) {{
        \\    if (elem >= count) return;
        \\
    , .{ kernel_name, scalar_type, index_type, scalar_type });
    try msl.KernelWriter.writeOutputCoordinates(writer, output.dims);
    try writeGatherIndexLinear(writer, indices.dims, output.dims.len, gather);
    const max_axis_index: u64 = @intCast(operand.dims[gather.axis] - 1);
    if (std.mem.eql(u8, index_type, "uint")) {
        try writer.print("    const uint gather_index = min(uint(in1[index_linear]), {d}u);\n", .{max_axis_index});
    } else {
        try writer.print("    const uint gather_index = uint(clamp(int(in1[index_linear]), 0, {d}));\n", .{max_axis_index});
    }
    try writeGatherOperandLinear(writer, operand.dims, output.dims.len, gather);
    try writer.writeAll("    out0[elem] = in0[operand_linear];\n}\n");
}
fn writeGatherIndexLinear(writer: *std.Io.Writer, indices_dims: []const i64, output_rank: usize, gather: GatherAxis) !void {
    try writer.writeAll("    uint index_linear = 0;\n");
    for (indices_dims, 0..) |_, indices_axis| {
        const stride = tensor.denseStride(indices_dims, indices_axis + 1);
        if (gather.explicit_index_vector and indices_axis == @as(usize, @intCast(gather.index_vector_dim))) continue;
        const output_axis = gather.outputAxisForIndexAxis(output_rank, indices_axis) orelse unreachable;
        try writer.print("    index_linear += coord{d} * {d}u;\n", .{ output_axis, stride });
    }
}

fn writeGatherOperandLinear(writer: *std.Io.Writer, operand_dims: []const i64, output_rank: usize, gather: GatherAxis) !void {
    try writer.writeAll("    uint operand_linear = 0;\n");
    for (operand_dims, 0..) |_, operand_axis| {
        const stride = tensor.denseStride(operand_dims, operand_axis + 1);
        if (operand_axis == gather.axis) {
            try writer.print("    operand_linear += gather_index * {d}u;\n", .{stride});
        } else if (gather.batchingIndexAxis(operand_axis)) |indices_axis| {
            const output_axis = gather.outputAxisForIndexAxis(output_rank, indices_axis) orelse unreachable;
            try writer.print("    operand_linear += coord{d} * {d}u;\n", .{ output_axis, stride });
        } else if (gather.outputAxisForOperandAxis(operand_axis)) |output_axis| {
            try writer.print("    operand_linear += coord{d} * {d}u;\n", .{ output_axis, stride });
        }
    }
}
