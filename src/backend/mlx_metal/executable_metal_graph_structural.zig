const std = @import("std");

const ir = @import("src/compiler/ir");
const metalcpp_call = @import("metalcpp_call.zig");
const msl = @import("executable_metal_graph_msl.zig");
const program_mod = @import("program.zig");
const tensor = @import("executable_metal_graph_tensor.zig");

pub fn makeAliasStep(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction) program_mod.Error!?metalcpp_call.ExecutableStepSpec {
    if (instruction.inputs.len != 1 or instruction.outputs.len != 1) return null;
    const input = tensor.valueDescriptor(plan, instruction.inputs[0]) orelse return null;
    const output = tensor.valueDescriptor(plan, instruction.outputs[0]) orelse return null;
    switch (instruction.kind) {
        .reshape, .copy_arg0, .reduce_precision, .bitcast_convert => {
            if (input.element_type != output.element_type or tensor.denseElementCount(input) != tensor.denseElementCount(output)) return null;
        },
        .broadcast_in_dim => {
            if (input.element_type != output.element_type or !tensor.sameDims(input.dims, output.dims)) return null;
        },
        else => return null,
    }
    const inputs = try tensor.valueIndices(allocator, instruction.inputs);
    errdefer allocator.free(inputs);
    const outputs = try tensor.valueIndices(allocator, instruction.outputs);
    errdefer allocator.free(outputs);
    return .{
        .kernel_name = &.{},
        .source = &.{},
        .inputs = inputs,
        .outputs = outputs,
        .release_values = &.{},
        .element_count = 1,
    };
}

pub fn makeCopyStep(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize) program_mod.Error!?metalcpp_call.ExecutableStepSpec {
    if (instruction.inputs.len != 1 or instruction.outputs.len != 1) return null;
    const input = tensor.valueDescriptor(plan, instruction.inputs[0]) orelse return null;
    const output = tensor.valueDescriptor(plan, instruction.outputs[0]) orelse return null;
    if (input.element_type != output.element_type or tensor.denseElementCount(input) != tensor.denseElementCount(output)) return null;
    return makeCopyLikeStep(allocator, plan, instruction.inputs, instruction.outputs, instruction_index, "copy");
}

pub fn makeBroadcastStep(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize) program_mod.Error!?metalcpp_call.ExecutableStepSpec {
    if (instruction.inputs.len != 1 or instruction.outputs.len != 1) return null;
    const input = tensor.valueDescriptor(plan, instruction.inputs[0]) orelse return null;
    const output = tensor.valueDescriptor(plan, instruction.outputs[0]) orelse return null;
    if (input.element_type != output.element_type) return null;
    const broadcast_dimensions = instruction.broadcast_dimensions orelse return null;
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
    const input_count = tensor.denseElementCount(input);
    const output_count = tensor.denseElementCount(output);
    if (input_count == 0 or output_count == 0 or output_count > std.math.maxInt(u32)) return null;
    const scalar_type = tensor.metalProgramScalarType(output.element_type) orelse return null;
    const inputs = try tensor.valueIndices(allocator, instruction.inputs);
    errdefer allocator.free(inputs);
    const outputs = try tensor.valueIndices(allocator, instruction.outputs);
    errdefer allocator.free(outputs);
    const kernel_name = try tensor.kernelName(allocator, "broadcast", instruction_index);
    errdefer allocator.free(kernel_name);
    var source = std.Io.Writer.Allocating.init(allocator);
    errdefer source.deinit();
    writeBroadcastKernel(&source.writer, kernel_name, scalar_type, input, output, broadcast_dimensions) catch return error.OutOfMemory;
    return .{
        .kernel_name = kernel_name,
        .source = try source.toOwnedSlice(),
        .inputs = inputs,
        .outputs = outputs,
        .element_count = @intCast(output_count),
    };
}

pub fn makeConvertStep(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize) program_mod.Error!?metalcpp_call.ExecutableStepSpec {
    if (instruction.inputs.len != 1 or instruction.outputs.len != 1) return null;
    const input = tensor.valueDescriptor(plan, instruction.inputs[0]) orelse return null;
    const output = tensor.valueDescriptor(plan, instruction.outputs[0]) orelse return null;
    if (tensor.denseElementCount(input) != tensor.denseElementCount(output)) return null;
    const output_count = tensor.denseElementCount(output);
    if (output_count == 0 or output_count > std.math.maxInt(u32)) return null;
    const input_type = tensor.metalProgramScalarType(input.element_type) orelse return null;
    const output_type = tensor.metalProgramScalarType(output.element_type) orelse return null;
    const inputs = try tensor.valueIndices(allocator, instruction.inputs);
    errdefer allocator.free(inputs);
    const outputs = try tensor.valueIndices(allocator, instruction.outputs);
    errdefer allocator.free(outputs);
    const kernel_name = try tensor.kernelName(allocator, "convert", instruction_index);
    errdefer allocator.free(kernel_name);
    var source = std.Io.Writer.Allocating.init(allocator);
    errdefer source.deinit();
    writeConvertKernel(&source.writer, kernel_name, input_type, output_type) catch return error.OutOfMemory;
    return .{
        .kernel_name = kernel_name,
        .source = try source.toOwnedSlice(),
        .inputs = inputs,
        .outputs = outputs,
        .element_count = @intCast(output_count),
    };
}

pub fn makeCopyLikeStep(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, inputs_ids: []const ir.ValueId, output_ids: []const ir.ValueId, instruction_index: usize, label: []const u8) program_mod.Error!?metalcpp_call.ExecutableStepSpec {
    const output = tensor.valueDescriptor(plan, output_ids[0]) orelse return null;
    const input = tensor.valueDescriptor(plan, inputs_ids[0]) orelse return null;
    const output_count = tensor.denseElementCount(output);
    const scalar_type = tensor.metalProgramScalarType(output.element_type) orelse return null;
    const inputs = try tensor.valueIndices(allocator, inputs_ids);
    errdefer allocator.free(inputs);
    const outputs = try tensor.valueIndices(allocator, output_ids);
    errdefer allocator.free(outputs);
    const kernel_name = try tensor.kernelName(allocator, label, instruction_index);
    errdefer allocator.free(kernel_name);
    var source = std.Io.Writer.Allocating.init(allocator);
    errdefer source.deinit();
    const input_expr = if (tensor.denseElementCount(input) == 1) "in0[0]" else "in0[elem]";
    msl.KernelWriter.writeSimpleKernelPrefix(&source.writer, kernel_name, scalar_type, 1, 1) catch return error.OutOfMemory;
    source.writer.print("    out0[elem] = {s};\n}}\n", .{input_expr}) catch return error.OutOfMemory;
    return .{ .kernel_name = kernel_name, .source = try source.toOwnedSlice(), .inputs = inputs, .outputs = outputs, .element_count = @intCast(output_count) };
}

pub fn makeTransposeStep(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize) program_mod.Error!?metalcpp_call.ExecutableStepSpec {
    if (instruction.inputs.len != 1 or instruction.outputs.len != 1) return null;
    const permutation = instruction.permutation orelse return null;
    const input = tensor.valueDescriptor(plan, instruction.inputs[0]) orelse return null;
    const output = tensor.valueDescriptor(plan, instruction.outputs[0]) orelse return null;
    if (input.element_type != output.element_type or input.dims.len != 2 or output.dims.len != 2 or permutation.len != 2 or permutation[0] != 1 or permutation[1] != 0) return null;
    const output_count = tensor.denseElementCount(output);
    const scalar_type = tensor.metalScalarType(output.element_type) orelse return null;
    const inputs = try tensor.valueIndices(allocator, instruction.inputs);
    errdefer allocator.free(inputs);
    const outputs = try tensor.valueIndices(allocator, instruction.outputs);
    errdefer allocator.free(outputs);
    const kernel_name = try tensor.kernelName(allocator, "transpose", instruction_index);
    errdefer allocator.free(kernel_name);
    var source = std.Io.Writer.Allocating.init(allocator);
    errdefer source.deinit();
    msl.KernelWriter.writeSimpleKernelPrefix(&source.writer, kernel_name, scalar_type, 1, 1) catch return error.OutOfMemory;
    source.writer.print(
        \\    const uint rows = {d};
        \\    const uint cols = {d};
        \\    const uint row = elem / rows;
        \\    const uint col = elem - row * rows;
        \\    out0[elem] = in0[col * cols + row];
        \\}}
        \\
    , .{ input.dims[0], input.dims[1] }) catch return error.OutOfMemory;
    return .{ .kernel_name = kernel_name, .source = try source.toOwnedSlice(), .inputs = inputs, .outputs = outputs, .element_count = @intCast(output_count) };
}

fn writeConvertKernel(writer: *std.Io.Writer, kernel_name: []const u8, input_type: []const u8, output_type: []const u8) !void {
    try writer.print(
        \\#include <metal_stdlib>
        \\using namespace metal;
        \\kernel void {s}(
        \\    device const {s}* in0 [[buffer(0)]],
        \\    device {s}* out0 [[buffer(1)]],
        \\    constant uint& count [[buffer(2)]],
        \\    uint elem [[thread_position_in_grid]]) {{
        \\    if (elem >= count) return;
        \\    out0[elem] = {s}(in0[elem]);
        \\}}
        \\
    , .{ kernel_name, input_type, output_type, output_type });
}

fn writeBroadcastKernel(writer: *std.Io.Writer, kernel_name: []const u8, scalar_type: []const u8, input: ir.BufferDescriptor, output: ir.BufferDescriptor, broadcast_dimensions: []const i64) !void {
    try writer.print(
        \\#include <metal_stdlib>
        \\using namespace metal;
        \\kernel void {s}(
        \\    device const {s}* in0 [[buffer(0)]],
        \\    device {s}* out0 [[buffer(1)]],
        \\    constant uint& count [[buffer(2)]],
        \\    uint elem [[thread_position_in_grid]]) {{
        \\    if (elem >= count) return;
        \\
    , .{ kernel_name, scalar_type, scalar_type });
    if (input.dims.len == 0 or tensor.denseElementCount(input) == 1) {
        try writer.writeAll("    out0[elem] = in0[0];\n}\n");
        return;
    }
    try msl.KernelWriter.writeOutputCoordinates(writer, output.dims);
    try writer.writeAll("    uint input_index = 0;\n");
    for (input.dims, broadcast_dimensions, 0..) |input_dim, output_axis_i64, input_axis| {
        const input_stride = tensor.denseStride(input.dims, input_axis + 1);
        const output_axis: usize = @intCast(output_axis_i64);
        if (input_dim == 1) {
            continue;
        }
        try writer.print("    input_index += coord{d} * {d}u;\n", .{ output_axis, input_stride });
    }
    try writer.writeAll("    out0[elem] = in0[input_index];\n}\n");
}
