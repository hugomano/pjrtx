const std = @import("std");

const ir = @import("src/compiler/ir");
const metalcpp_call = @import("metalcpp_call.zig");
const msl = @import("executable_metal_graph_msl.zig");
const program_mod = @import("program.zig");
const tensor = @import("executable_metal_graph_tensor.zig");

const tiled_threadgroup_width: u64 = 256;

/// Builds generated Metal reduction steps, including tiled suffix reductions.
pub fn makeReduceStep(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize) program_mod.Error!?metalcpp_call.ExecutableStepSpec {
    if (instruction.kind == .reduce_max and instruction.inputs.len == 2 and instruction.outputs.len == 2) {
        return makeReduceMaxWithIndexStep(allocator, plan, instruction, instruction_index);
    }
    if (instruction.inputs.len < 1 or instruction.outputs.len != 1) return null;
    const reduce_dimensions = instruction.reduce_dimensions orelse return null;
    const input = tensor.valueDescriptor(plan, instruction.inputs[0]) orelse return null;
    const output = tensor.valueDescriptor(plan, instruction.outputs[0]) orelse return null;
    if (input.element_type != output.element_type or input.dims.len < reduce_dimensions.len) return null;
    const output_count = tensor.denseElementCount(output);
    const input_count = tensor.denseElementCount(input);
    if (output_count == 0 or input_count == 0 or input_count % output_count != 0) return null;
    for (reduce_dimensions, 0..) |dimension, i| {
        if (dimension != @as(i64, @intCast(input.dims.len - reduce_dimensions.len + i))) return null;
    }
    const reduce_count = input_count / output_count;
    const scalar_type = tensor.metalScalarType(output.element_type) orelse return null;
    if (try makeTiledReduceStep(allocator, instruction, instruction_index, output, output_count, reduce_count, scalar_type)) |step| return step;
    const inputs = try tensor.valueIndices(allocator, instruction.inputs[0..1]);
    errdefer allocator.free(inputs);
    const outputs = try tensor.valueIndices(allocator, instruction.outputs);
    errdefer allocator.free(outputs);
    const kernel_name = try tensor.kernelName(allocator, "reduce", instruction_index);
    errdefer allocator.free(kernel_name);
    var source = std.Io.Writer.Allocating.init(allocator);
    errdefer source.deinit();
    msl.KernelWriter.writeSimpleKernelPrefix(&source.writer, kernel_name, scalar_type, 1, 1) catch return error.OutOfMemory;
    const init_expr = switch (instruction.kind) {
        .reduce_sum => try std.fmt.allocPrint(allocator, "{s}(0)", .{scalar_type}),
        .reduce_max, .reduce_min => try std.fmt.allocPrint(allocator, "in0[elem * {d}]", .{reduce_count}),
        else => unreachable,
    };
    defer allocator.free(init_expr);
    source.writer.print("    {s} acc = {s};\n", .{ scalar_type, init_expr }) catch return error.OutOfMemory;
    source.writer.print("    for (uint i = {d}; i < {d}; ++i) {{\n", .{ if (instruction.kind == .reduce_sum) @as(usize, 0) else 1, reduce_count }) catch return error.OutOfMemory;
    switch (instruction.kind) {
        .reduce_sum => source.writer.print("        acc += in0[elem * {d} + i];\n", .{reduce_count}) catch return error.OutOfMemory,
        .reduce_max => source.writer.print("        acc = max(acc, in0[elem * {d} + i]);\n", .{reduce_count}) catch return error.OutOfMemory,
        .reduce_min => source.writer.print("        acc = min(acc, in0[elem * {d} + i]);\n", .{reduce_count}) catch return error.OutOfMemory,
        else => unreachable,
    }
    source.writer.writeAll("    }\n    out0[elem] = acc;\n}\n") catch return error.OutOfMemory;
    return .{ .kernel_name = kernel_name, .source = try source.toOwnedSlice(), .inputs = inputs, .outputs = outputs, .element_count = @intCast(output_count) };
}

fn makeTiledReduceStep(
    allocator: std.mem.Allocator,
    instruction: ir.PlanInstruction,
    instruction_index: usize,
    output: ir.BufferDescriptor,
    output_count: usize,
    reduce_count: usize,
    scalar_type: []const u8,
) program_mod.Error!?metalcpp_call.ExecutableStepSpec {
    if (reduce_count < tiled_threadgroup_width) return null;
    if (!tensor.tiledFloatAccumulatorType(output.element_type)) return null;
    const total_threads = @as(u64, @intCast(output_count)) * tiled_threadgroup_width;
    if (total_threads > std.math.maxInt(u32)) return null;
    const inputs = try tensor.valueIndices(allocator, instruction.inputs[0..1]);
    errdefer allocator.free(inputs);
    const outputs = try tensor.valueIndices(allocator, instruction.outputs);
    errdefer allocator.free(outputs);
    const kernel_name = try tensor.kernelName(allocator, "tiled_reduce", instruction_index);
    errdefer allocator.free(kernel_name);
    var source = std.Io.Writer.Allocating.init(allocator);
    errdefer source.deinit();
    writeTiledReduceKernel(&source.writer, kernel_name, scalar_type, instruction.kind, reduce_count, output_count) catch return error.OutOfMemory;
    return .{
        .kernel_name = kernel_name,
        .source = try source.toOwnedSlice(),
        .inputs = inputs,
        .outputs = outputs,
        .element_count = total_threads,
        .threads_per_threadgroup = @intCast(tiled_threadgroup_width),
    };
}

fn writeTiledReduceKernel(
    writer: *std.Io.Writer,
    kernel_name: []const u8,
    scalar_type: []const u8,
    kind: ir.PlanInstructionKind,
    reduce_count: usize,
    output_count: usize,
) !void {
    const init_expr = switch (kind) {
        .reduce_sum => "0.0f",
        .reduce_max => "-3.402823466e+38f",
        .reduce_min => "3.402823466e+38f",
        else => unreachable,
    };
    const update_expr = switch (kind) {
        .reduce_sum => "acc += value;",
        .reduce_max => "acc = max(acc, value);",
        .reduce_min => "acc = min(acc, value);",
        else => unreachable,
    };
    const merge_expr = switch (kind) {
        .reduce_sum => "partials[tid] += partials[tid + stride];",
        .reduce_max => "partials[tid] = max(partials[tid], partials[tid + stride]);",
        .reduce_min => "partials[tid] = min(partials[tid], partials[tid + stride]);",
        else => unreachable,
    };
    try writer.print(
        \\#include <metal_stdlib>
        \\using namespace metal;
        \\kernel void {s}(
        \\    device const {s}* in0 [[buffer(0)]],
        \\    device {s}* out0 [[buffer(1)]],
        \\    constant uint& count [[buffer(2)]],
        \\    uint3 elem_pos [[thread_position_in_grid]],
        \\    uint3 tid_pos [[thread_position_in_threadgroup]],
        \\    uint3 group_pos [[threadgroup_position_in_grid]]) {{
        \\    const uint elem = elem_pos.x;
        \\    const uint tid = tid_pos.x;
        \\    if (elem >= count) return;
        \\    const uint out_elem = group_pos.x;
        \\    if (out_elem >= {d}u) return;
        \\    threadgroup float partials[256];
        \\    float acc = {s};
        \\    for (uint i = tid; i < {d}u; i += 256u) {{
        \\        const float value = float(in0[out_elem * {d}u + i]);
        \\        {s}
        \\    }}
        \\    partials[tid] = acc;
        \\    threadgroup_barrier(mem_flags::mem_threadgroup);
        \\    for (uint stride = 128u; stride > 0u; stride >>= 1u) {{
        \\        if (tid < stride) {{
        \\            {s}
        \\        }}
        \\        threadgroup_barrier(mem_flags::mem_threadgroup);
        \\    }}
        \\    if (tid == 0u) out0[out_elem] = {s}(partials[0]);
        \\}}
        \\
    , .{
        kernel_name,
        scalar_type,
        scalar_type,
        output_count,
        init_expr,
        reduce_count,
        reduce_count,
        update_expr,
        merge_expr,
        scalar_type,
    });
}

fn makeReduceMaxWithIndexStep(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize) program_mod.Error!?metalcpp_call.ExecutableStepSpec {
    const reduce_dimensions = instruction.reduce_dimensions orelse return null;
    const input_value = tensor.valueDescriptor(plan, instruction.inputs[0]) orelse return null;
    const input_index = tensor.valueDescriptor(plan, instruction.inputs[1]) orelse return null;
    const output_value = tensor.valueDescriptor(plan, instruction.outputs[0]) orelse return null;
    const output_index = tensor.valueDescriptor(plan, instruction.outputs[1]) orelse return null;
    if (input_value.element_type != output_value.element_type or input_index.element_type != output_index.element_type) return null;
    if (!tensor.sameDims(input_value.dims, input_index.dims) or !tensor.sameDims(output_value.dims, output_index.dims)) return null;
    if (input_value.dims.len < reduce_dimensions.len) return null;
    const output_count = tensor.denseElementCount(output_value);
    const input_count = tensor.denseElementCount(input_value);
    if (output_count == 0 or input_count == 0 or input_count % output_count != 0) return null;
    for (reduce_dimensions, 0..) |dimension, i| {
        if (dimension != @as(i64, @intCast(input_value.dims.len - reduce_dimensions.len + i))) return null;
    }
    const reduce_count = input_count / output_count;
    const value_type = tensor.metalScalarType(output_value.element_type) orelse return null;
    const index_type = tensor.metalProgramScalarType(output_index.element_type) orelse return null;
    if (try makeTiledReduceMaxWithIndexStep(allocator, instruction, instruction_index, output_value, output_count, reduce_count, value_type, index_type)) |step| return step;
    const inputs = try tensor.valueIndices(allocator, instruction.inputs);
    errdefer allocator.free(inputs);
    const outputs = try tensor.valueIndices(allocator, instruction.outputs);
    errdefer allocator.free(outputs);
    const kernel_name = try tensor.kernelName(allocator, "reduce_max_index", instruction_index);
    errdefer allocator.free(kernel_name);
    var source = std.Io.Writer.Allocating.init(allocator);
    errdefer source.deinit();
    writeReduceMaxWithIndexKernel(&source.writer, kernel_name, value_type, index_type, reduce_count) catch return error.OutOfMemory;
    return .{
        .kernel_name = kernel_name,
        .source = try source.toOwnedSlice(),
        .inputs = inputs,
        .outputs = outputs,
        .element_count = @intCast(output_count),
    };
}

fn makeTiledReduceMaxWithIndexStep(
    allocator: std.mem.Allocator,
    instruction: ir.PlanInstruction,
    instruction_index: usize,
    output_value: ir.BufferDescriptor,
    output_count: usize,
    reduce_count: usize,
    value_type: []const u8,
    index_type: []const u8,
) program_mod.Error!?metalcpp_call.ExecutableStepSpec {
    if (reduce_count < tiled_threadgroup_width) return null;
    if (!tensor.tiledFloatAccumulatorType(output_value.element_type)) return null;
    const total_threads = @as(u64, @intCast(output_count)) * tiled_threadgroup_width;
    if (total_threads > std.math.maxInt(u32)) return null;
    const inputs = try tensor.valueIndices(allocator, instruction.inputs);
    errdefer allocator.free(inputs);
    const outputs = try tensor.valueIndices(allocator, instruction.outputs);
    errdefer allocator.free(outputs);
    const kernel_name = try tensor.kernelName(allocator, "tiled_reduce_max_index", instruction_index);
    errdefer allocator.free(kernel_name);
    var source = std.Io.Writer.Allocating.init(allocator);
    errdefer source.deinit();
    writeTiledReduceMaxWithIndexKernel(&source.writer, kernel_name, value_type, index_type, reduce_count, output_count) catch return error.OutOfMemory;
    return .{
        .kernel_name = kernel_name,
        .source = try source.toOwnedSlice(),
        .inputs = inputs,
        .outputs = outputs,
        .element_count = total_threads,
        .threads_per_threadgroup = @intCast(tiled_threadgroup_width),
    };
}
fn writeReduceMaxWithIndexKernel(writer: *std.Io.Writer, kernel_name: []const u8, value_type: []const u8, index_type: []const u8, reduce_count: usize) !void {
    try writer.print(
        \\#include <metal_stdlib>
        \\using namespace metal;
        \\kernel void {s}(
        \\    device const {s}* in0 [[buffer(0)]],
        \\    device const {s}* in1 [[buffer(1)]],
        \\    device {s}* out0 [[buffer(2)]],
        \\    device {s}* out1 [[buffer(3)]],
        \\    constant uint& count [[buffer(4)]],
        \\    uint elem [[thread_position_in_grid]]) {{
        \\    if (elem >= count) return;
        \\    const uint base = elem * {d}u;
        \\    {s} best_value = in0[base];
        \\    {s} best_index = in1[base];
        \\    for (uint i = 1; i < {d}u; ++i) {{
        \\        const {s} candidate_value = in0[base + i];
        \\        const {s} candidate_index = in1[base + i];
        \\        if (float(candidate_value) > float(best_value)) {{
        \\            best_value = candidate_value;
        \\            best_index = candidate_index;
        \\        }}
        \\    }}
        \\    out0[elem] = best_value;
        \\    out1[elem] = best_index;
        \\}}
        \\
    , .{ kernel_name, value_type, index_type, value_type, index_type, reduce_count, value_type, index_type, reduce_count, value_type, index_type });
}

fn writeTiledReduceMaxWithIndexKernel(writer: *std.Io.Writer, kernel_name: []const u8, value_type: []const u8, index_type: []const u8, reduce_count: usize, output_count: usize) !void {
    try writer.print(
        \\#include <metal_stdlib>
        \\using namespace metal;
        \\kernel void {s}(
        \\    device const {s}* in0 [[buffer(0)]],
        \\    device const {s}* in1 [[buffer(1)]],
        \\    device {s}* out0 [[buffer(2)]],
        \\    device {s}* out1 [[buffer(3)]],
        \\    constant uint& count [[buffer(4)]],
        \\    uint3 elem_pos [[thread_position_in_grid]],
        \\    uint3 tid_pos [[thread_position_in_threadgroup]],
        \\    uint3 group_pos [[threadgroup_position_in_grid]]) {{
        \\    const uint elem = elem_pos.x;
        \\    const uint tid = tid_pos.x;
        \\    if (elem >= count) return;
        \\    const uint out_elem = group_pos.x;
        \\    if (out_elem >= {d}u) return;
        \\    const uint base = out_elem * {d}u;
        \\    threadgroup float partial_values[256];
        \\    threadgroup {s} partial_indices[256];
        \\    float best_value = -3.402823466e+38f;
        \\    {s} best_index = {s}(0);
        \\    for (uint i = tid; i < {d}u; i += 256u) {{
        \\        const uint input_index = base + i;
        \\        const float candidate_value = float(in0[input_index]);
        \\        const {s} candidate_index = in1[input_index];
        \\        if (candidate_value > best_value) {{
        \\            best_value = candidate_value;
        \\            best_index = candidate_index;
        \\        }}
        \\    }}
        \\    partial_values[tid] = best_value;
        \\    partial_indices[tid] = best_index;
        \\    threadgroup_barrier(mem_flags::mem_threadgroup);
        \\    for (uint stride = 128u; stride > 0u; stride >>= 1u) {{
        \\        if (tid < stride && partial_values[tid + stride] > partial_values[tid]) {{
        \\            partial_values[tid] = partial_values[tid + stride];
        \\            partial_indices[tid] = partial_indices[tid + stride];
        \\        }}
        \\        threadgroup_barrier(mem_flags::mem_threadgroup);
        \\    }}
        \\    if (tid == 0u) {{
        \\        out0[out_elem] = {s}(partial_values[0]);
        \\        out1[out_elem] = partial_indices[0];
        \\    }}
        \\}}
        \\
    , .{
        kernel_name,
        value_type,
        index_type,
        value_type,
        index_type,
        output_count,
        reduce_count,
        index_type,
        index_type,
        index_type,
        reduce_count,
        index_type,
        value_type,
    });
}
