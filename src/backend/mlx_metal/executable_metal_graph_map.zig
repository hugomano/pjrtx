const std = @import("std");

const ir = @import("src/compiler/ir");
const map_rules = @import("executable_metal_graph_map_rules.zig");
const metalcpp_call = @import("metalcpp_call.zig");
const msl = @import("executable_metal_graph_msl.zig");
const program_mod = @import("program.zig");
const tensor = @import("executable_metal_graph_tensor.zig");

/// Describes generated elementwise map kernel arity.
pub const Kind = map_rules.Kind;

pub fn makeMapStep(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, map_kind: Kind) program_mod.Error!?metalcpp_call.ExecutableStepSpec {
    if (instruction.outputs.len != 1) return null;
    if (instruction.inputs.len != map_rules.inputCount(map_kind)) return null;
    const output = tensor.valueDescriptor(plan, instruction.outputs[0]) orelse return null;
    const output_count = tensor.denseElementCount(output);
    if (output_count == 0 or output_count > std.math.maxInt(u32)) return null;
    const scalar_type = map_rules.outputScalarType(instruction.kind, output.element_type) orelse return null;
    const inputs = try tensor.valueIndices(allocator, instruction.inputs);
    errdefer allocator.free(inputs);
    const outputs = try tensor.valueIndices(allocator, instruction.outputs);
    errdefer allocator.free(outputs);
    for (instruction.inputs) |input_id| {
        const input = tensor.valueDescriptor(plan, input_id) orelse return null;
        const input_count = tensor.denseElementCount(input);
        if (tensor.metalProgramScalarType(input.element_type) == null) return null;
        if (input.layout != .dense_row_major or (input_count != output_count and input_count != 1)) return null;
    }

    const kernel_name = try tensor.kernelName(allocator, "map", instruction_index);
    errdefer allocator.free(kernel_name);
    var source = std.Io.Writer.Allocating.init(allocator);
    errdefer source.deinit();
    writeMapKernel(allocator, &source.writer, kernel_name, scalar_type, plan, instruction, output_count, map_kind) catch return error.OutOfMemory;
    return .{
        .kernel_name = kernel_name,
        .source = try source.toOwnedSlice(),
        .inputs = inputs,
        .outputs = outputs,
        .element_count = @intCast(output_count),
    };
}

fn writeMapKernel(allocator: std.mem.Allocator, writer: *std.Io.Writer, kernel_name: []const u8, scalar_type: []const u8, plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, output_count: usize, map_kind: Kind) !void {
    _ = output_count;
    _ = map_kind;
    try writer.print(
        \\#include <metal_stdlib>
        \\using namespace metal;
        \\kernel void {s}(
        \\
    , .{kernel_name});
    for (instruction.inputs, 0..) |input_id, index| {
        const input_type = tensor.metalProgramScalarType((tensor.valueDescriptor(plan, input_id) orelse unreachable).element_type) orelse unreachable;
        try writer.print("    device const {s}* in{d} [[buffer({d})]],\n", .{ input_type, index, index });
    }
    try writer.print("    device {s}* out0 [[buffer({d})]],\n", .{ scalar_type, instruction.inputs.len });
    try writer.print(
        \\    constant uint& count [[buffer({d})]],
        \\    uint elem [[thread_position_in_grid]]) {{
        \\    if (elem >= count) return;
        \\
    , .{instruction.inputs.len + 1});
    const expression = try mapExpression(allocator, plan, instruction);
    defer allocator.free(expression);
    try writer.print("    out0[elem] = {s}({s});\n}}\n", .{ scalar_type, expression });
}

pub fn writeUnaryExpression(writer: *std.Io.Writer, scalar_type: []const u8, input: []const u8, op: ir.ElementwiseUnaryOp) !void {
    const expression = (try map_rules.unaryExpression(std.heap.page_allocator, map_rules.unaryInstructionKind(op), input, scalar_type, .plain)) orelse return error.UnsupportedOperation;
    defer std.heap.page_allocator.free(expression);
    try writer.print("    out0[elem] = {s};\n}}\n", .{expression});
}

pub fn writeBinaryExpression(writer: *std.Io.Writer, scalar_type: []const u8, lhs: []const u8, rhs: []const u8, op: ir.ElementwiseBinaryOp) !void {
    _ = scalar_type;
    const expression = (try map_rules.binaryExpression(std.heap.page_allocator, map_rules.binaryInstructionKind(op), lhs, rhs)) orelse return error.UnsupportedOperation;
    defer std.heap.page_allocator.free(expression);
    try writer.print("    out0[elem] = {s};\n}}\n", .{expression});
}

pub fn graphUnarySupported(op: ir.ElementwiseUnaryOp) bool {
    return map_rules.unarySupported(map_rules.unaryInstructionKind(op));
}

pub fn graphBinarySupported(op: ir.ElementwiseBinaryOp) bool {
    return map_rules.binarySupported(map_rules.binaryInstructionKind(op));
}

fn mapExpression(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction) ![]const u8 {
    const input0 = inputRef(plan, instruction.inputs[0], 0);
    if (map_rules.binarySupported(instruction.kind)) {
        return (try map_rules.binaryExpression(allocator, instruction.kind, input0, inputRef(plan, instruction.inputs[1], 1))).?;
    }
    if (map_rules.unarySupported(instruction.kind)) {
        return (try map_rules.unaryExpression(allocator, instruction.kind, input0, tensor.metalScalarType((tensor.valueDescriptor(plan, instruction.outputs[0]) orelse unreachable).element_type).?, .plain)).?;
    }
    return switch (instruction.kind) {
        .compare => try map_rules.compareExpression(allocator, input0, inputRef(plan, instruction.inputs[1], 1), instruction.compare_direction orelse .eq),
        .select => try map_rules.selectExpression(allocator, input0, inputRef(plan, instruction.inputs[1], 1), inputRef(plan, instruction.inputs[2], 2)),
        .clamp => try map_rules.clampExpression(allocator, input0, inputRef(plan, instruction.inputs[1], 1), inputRef(plan, instruction.inputs[2], 2)),
        else => unreachable,
    };
}

fn inputRef(plan: *const ir.ExecutablePlan, value_id: ir.ValueId, index: usize) []const u8 {
    const input = tensor.valueDescriptor(plan, value_id) orelse unreachable;
    return if (tensor.denseElementCount(input) == 1)
        switch (index) {
            0 => "in0[0]",
            1 => "in1[0]",
            2 => "in2[0]",
            else => unreachable,
        }
    else switch (index) {
        0 => "in0[elem]",
        1 => "in1[elem]",
        2 => "in2[elem]",
        else => unreachable,
    };
}
