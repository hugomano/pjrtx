const std = @import("std");

const ir = @import("src/compiler/ir");
const metalcpp_call = @import("metalcpp_call.zig");
const program_mod = @import("program.zig");
const tensor = @import("executable_metal_graph_tensor.zig");

pub fn makeSliceStep(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize) program_mod.Error!?metalcpp_call.ExecutableStepSpec {
    if (instruction.inputs.len != 1 or instruction.outputs.len != 1) return null;
    const starts = instruction.start_indices orelse return null;
    const limits = instruction.limit_indices orelse return null;
    const strides = instruction.strides orelse return null;
    const input = tensor.valueDescriptor(plan, instruction.inputs[0]) orelse return null;
    const output = tensor.valueDescriptor(plan, instruction.outputs[0]) orelse return null;
    if (input.element_type != output.element_type or input.dims.len != output.dims.len) return null;
    if (starts.len != input.dims.len or limits.len != input.dims.len or strides.len != input.dims.len) return null;
    const output_count = tensor.denseElementCount(output);
    if (output_count == 0 or output_count > std.math.maxInt(u32)) return null;
    for (input.dims, output.dims, starts, limits, strides) |input_dim, output_dim, start, limit, stride| {
        if (input_dim < 0 or output_dim < 0 or start < 0 or limit < start or limit > input_dim or stride <= 0) return null;
        if (@divTrunc(limit - start + stride - 1, stride) != output_dim) return null;
    }
    const scalar_type = tensor.metalProgramScalarType(output.element_type) orelse return null;
    const inputs = try tensor.valueIndices(allocator, instruction.inputs);
    errdefer allocator.free(inputs);
    const outputs = try tensor.valueIndices(allocator, instruction.outputs);
    errdefer allocator.free(outputs);
    const kernel_name = try tensor.kernelName(allocator, "slice", instruction_index);
    errdefer allocator.free(kernel_name);
    var source = std.Io.Writer.Allocating.init(allocator);
    errdefer source.deinit();
    writeSliceKernel(&source.writer, kernel_name, scalar_type, input, output, starts, strides) catch return error.OutOfMemory;
    return .{
        .kernel_name = kernel_name,
        .source = try source.toOwnedSlice(),
        .inputs = inputs,
        .outputs = outputs,
        .element_count = @intCast(output_count),
    };
}

pub fn makeDynamicSliceStep(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize) program_mod.Error!?metalcpp_call.ExecutableStepSpec {
    if (instruction.outputs.len != 1 or instruction.inputs.len == 0) return null;
    const slice_sizes = instruction.slice_sizes orelse return null;
    const operand = tensor.valueDescriptor(plan, instruction.inputs[0]) orelse return null;
    const output = tensor.valueDescriptor(plan, instruction.outputs[0]) orelse return null;
    if (operand.element_type != output.element_type or operand.dims.len != output.dims.len or slice_sizes.len != operand.dims.len) return null;
    if (instruction.inputs.len != operand.dims.len + 1) return null;
    const output_count = tensor.denseElementCount(output);
    const operand_count = tensor.denseElementCount(operand);
    if (output_count == 0 or output_count > std.math.maxInt(u32) or operand_count == 0 or operand_count > std.math.maxInt(u32)) return null;
    for (operand.dims, output.dims, slice_sizes) |operand_dim, output_dim, slice_size| {
        if (operand_dim < 0 or output_dim < 0 or slice_size < 0) return null;
        if (slice_size > operand_dim or output_dim != slice_size) return null;
    }
    for (instruction.inputs[1..]) |start_id| {
        const start = tensor.valueDescriptor(plan, start_id) orelse return null;
        if (!tensor.dynamicSliceStartCompatible(start)) return null;
    }

    const scalar_type = tensor.metalScalarType(output.element_type) orelse return null;
    const inputs = try tensor.valueIndices(allocator, instruction.inputs);
    errdefer allocator.free(inputs);
    const outputs = try tensor.valueIndices(allocator, instruction.outputs);
    errdefer allocator.free(outputs);
    const kernel_name = try tensor.kernelName(allocator, "dynamic_slice", instruction_index);
    errdefer allocator.free(kernel_name);
    var source = std.Io.Writer.Allocating.init(allocator);
    errdefer source.deinit();
    writeDynamicSliceKernel(&source.writer, kernel_name, scalar_type, plan, instruction, operand, output, slice_sizes) catch return error.OutOfMemory;
    return .{
        .kernel_name = kernel_name,
        .source = try source.toOwnedSlice(),
        .inputs = inputs,
        .outputs = outputs,
        .element_count = @intCast(output_count),
    };
}

pub fn makeDynamicUpdateSliceStep(
    allocator: std.mem.Allocator,
    plan: *const ir.ExecutablePlan,
    instruction: ir.PlanInstruction,
    instruction_index: usize,
    release_values: []const u64,
) program_mod.Error!?metalcpp_call.ExecutableStepSpec {
    if (instruction.outputs.len != 1 or instruction.inputs.len < 2) return null;
    const input = tensor.valueDescriptor(plan, instruction.inputs[0]) orelse return null;
    const update = tensor.valueDescriptor(plan, instruction.inputs[1]) orelse return null;
    const output = tensor.valueDescriptor(plan, instruction.outputs[0]) orelse return null;
    if (input.element_type != update.element_type or input.element_type != output.element_type) return null;
    if (!tensor.sameDims(input.dims, output.dims) or input.dims.len != update.dims.len) return null;
    if (instruction.inputs.len != input.dims.len + 2) return null;
    const output_count = tensor.denseElementCount(output);
    if (output_count == 0 or output_count > std.math.maxInt(u32)) return null;
    for (input.dims, update.dims) |input_dim, update_dim| {
        if (input_dim < 0 or update_dim < 0 or update_dim > input_dim) return null;
    }
    for (instruction.inputs[2..]) |start_id| {
        const start = tensor.valueDescriptor(plan, start_id) orelse return null;
        if (!tensor.dynamicSliceStartCompatible(start)) return null;
    }
    const scalar_type = tensor.metalScalarType(output.element_type) orelse return null;
    const inputs = try tensor.valueIndices(allocator, instruction.inputs);
    errdefer allocator.free(inputs);
    const outputs = try tensor.valueIndices(allocator, instruction.outputs);
    errdefer allocator.free(outputs);
    const in_place = dynamicUpdateSliceInPlaceInput(plan, instruction, release_values) != null;
    const kernel_name = try tensor.kernelName(allocator, if (in_place) "dynamic_update_slice_in_place" else "dynamic_update_slice", instruction_index);
    errdefer allocator.free(kernel_name);
    var source = std.Io.Writer.Allocating.init(allocator);
    errdefer source.deinit();
    const element_count = if (in_place) tensor.denseElementCount(update) else output_count;
    if (in_place) {
        writeDynamicUpdateSliceInPlaceKernel(&source.writer, kernel_name, scalar_type, plan, instruction, input, update) catch return error.OutOfMemory;
    } else {
        writeDynamicUpdateSliceKernel(&source.writer, kernel_name, scalar_type, plan, instruction, input, update, output) catch return error.OutOfMemory;
    }
    return .{
        .kernel_name = kernel_name,
        .source = try source.toOwnedSlice(),
        .inputs = inputs,
        .outputs = outputs,
        .element_count = @intCast(element_count),
        .in_place_input = if (in_place) 0 else null,
    };
}

fn dynamicUpdateSliceInPlaceInput(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, release_values: []const u64) ?ir.ValueId {
    if (instruction.inputs.len < 2 or instruction.outputs.len != 1) return null;
    const operand_id = instruction.inputs[0];
    const output_id = instruction.outputs[0];
    if (operand_id.index >= plan.values.len or output_id.index >= plan.values.len) return null;
    if (containsIndex(release_values, operand_id.index)) return operand_id;
    if (plan.values[operand_id.index].role == .parameter) {
        const parameter_index = parameterIndexForValue(plan, operand_id) orelse return null;
        for (plan.output_aliases) |alias| {
            if (alias.kind != .donation or alias.parameter_index != parameter_index) continue;
            if (alias.output_index >= plan.output_ids.len) continue;
            if (plan.output_ids[alias.output_index].index == output_id.index) return operand_id;
        }
    }
    return null;
}

fn containsIndex(indices: []const u64, index: usize) bool {
    for (indices) |candidate| {
        if (candidate == index) return true;
    }
    return false;
}

fn parameterIndexForValue(plan: *const ir.ExecutablePlan, value_id: ir.ValueId) ?u32 {
    var parameter_index: u32 = 0;
    for (plan.values) |value| {
        if (value.role != .parameter) continue;
        if (value.id.index == value_id.index) return parameter_index;
        parameter_index += 1;
    }
    return null;
}
fn writeSliceKernel(writer: *std.Io.Writer, kernel_name: []const u8, scalar_type: []const u8, input: ir.BufferDescriptor, output: ir.BufferDescriptor, starts: []const i64, strides: []const i64) !void {
    try writer.print(
        \\#include <metal_stdlib>
        \\using namespace metal;
        \\kernel void {s}(
        \\    device const {s}* in0 [[buffer(0)]],
        \\    device {s}* out0 [[buffer(1)]],
        \\    constant uint& count [[buffer(2)]],
        \\    uint elem [[thread_position_in_grid]]) {{
        \\    if (elem >= count) return;
        \\    uint input_index = 0;
        \\
    , .{ kernel_name, scalar_type, scalar_type });
    for (output.dims, starts, strides, 0..) |output_dim, start, stride, axis| {
        const output_stride = tensor.denseStride(output.dims, axis + 1);
        const input_stride = tensor.denseStride(input.dims, axis + 1);
        try writer.print(
            \\    const uint coord{d} = (elem / {d}u) % {d}u;
            \\    input_index += ({d}u + coord{d} * {d}u) * {d}u;
            \\
        , .{
            axis,
            output_stride,
            @as(u64, @intCast(output_dim)),
            @as(u64, @intCast(start)),
            axis,
            @as(u64, @intCast(stride)),
            input_stride,
        });
    }
    try writer.writeAll("    out0[elem] = in0[input_index];\n}\n");
}

fn writeDynamicSliceKernel(
    writer: *std.Io.Writer,
    kernel_name: []const u8,
    scalar_type: []const u8,
    plan: *const ir.ExecutablePlan,
    instruction: ir.PlanInstruction,
    operand: ir.BufferDescriptor,
    output: ir.BufferDescriptor,
    slice_sizes: []const i64,
) !void {
    try writer.print(
        \\#include <metal_stdlib>
        \\using namespace metal;
        \\kernel void {s}(
        \\    device const {s}* in0 [[buffer(0)]],
        \\
    , .{ kernel_name, scalar_type });
    for (instruction.inputs[1..], 0..) |start_id, start_index| {
        const start = tensor.valueDescriptor(plan, start_id) orelse unreachable;
        const index_type = tensor.metalIndexScalarType(start.element_type) orelse unreachable;
        try writer.print("    device const {s}* in{d} [[buffer({d})]],\n", .{ index_type, start_index + 1, start_index + 1 });
    }
    try writer.print(
        \\    device {s}* out0 [[buffer({d})]],
        \\    constant uint& count [[buffer({d})]],
        \\    uint elem [[thread_position_in_grid]]) {{
        \\    if (elem >= count) return;
        \\    uint operand_index = 0;
        \\
    , .{ scalar_type, instruction.inputs.len, instruction.inputs.len + 1 });

    for (operand.dims, output.dims, slice_sizes, 0..) |operand_dim, output_dim, slice_size, axis| {
        const output_stride = tensor.denseStride(output.dims, axis + 1);
        const operand_stride = tensor.denseStride(operand.dims, axis + 1);
        const max_start = operand_dim - slice_size;
        try writer.print(
            \\    const uint coord{d} = (elem / {d}) % {d};
            \\    const int raw_start{d} = int(in{d}[0]);
            \\    const uint start{d} = uint(clamp(raw_start{d}, 0, {d}));
            \\    operand_index += (start{d} + coord{d}) * {d};
            \\
        , .{
            axis,
            output_stride,
            output_dim,
            axis,
            axis + 1,
            axis,
            axis,
            max_start,
            axis,
            axis,
            operand_stride,
        });
    }
    try writer.writeAll(
        \\    out0[elem] = in0[operand_index];
        \\}
        \\
    );
}

fn writeDynamicUpdateSliceKernel(
    writer: *std.Io.Writer,
    kernel_name: []const u8,
    scalar_type: []const u8,
    plan: *const ir.ExecutablePlan,
    instruction: ir.PlanInstruction,
    input: ir.BufferDescriptor,
    update: ir.BufferDescriptor,
    output: ir.BufferDescriptor,
) !void {
    try writer.print(
        \\#include <metal_stdlib>
        \\using namespace metal;
        \\kernel void {s}(
        \\    device const {s}* in0 [[buffer(0)]],
        \\    device const {s}* in1 [[buffer(1)]],
        \\
    , .{ kernel_name, scalar_type, scalar_type });
    for (instruction.inputs[2..], 0..) |start_id, start_index| {
        const start = tensor.valueDescriptor(plan, start_id) orelse unreachable;
        const index_type = tensor.metalIndexScalarType(start.element_type) orelse unreachable;
        try writer.print("    device const {s}* in{d} [[buffer({d})]],\n", .{ index_type, start_index + 2, start_index + 2 });
    }
    try writer.print(
        \\    device {s}* out0 [[buffer({d})]],
        \\    constant uint& count [[buffer({d})]],
        \\    uint elem [[thread_position_in_grid]]) {{
        \\    if (elem >= count) return;
        \\    uint update_index = 0;
        \\    bool inside_update = true;
        \\
    , .{ scalar_type, instruction.inputs.len, instruction.inputs.len + 1 });
    for (output.dims, update.dims, 0..) |output_dim, update_dim, axis| {
        const output_stride = tensor.denseStride(output.dims, axis + 1);
        const update_stride = tensor.denseStride(update.dims, axis + 1);
        const max_start = input.dims[axis] - update_dim;
        try writer.print(
            \\    const uint coord{d} = (elem / {d}u) % {d}u;
            \\    const int raw_start{d} = int(in{d}[0]);
            \\    const uint start{d} = uint(clamp(raw_start{d}, 0, {d}));
            \\    const bool in_axis{d} = coord{d} >= start{d} && coord{d} < start{d} + {d}u;
            \\    inside_update = inside_update && in_axis{d};
            \\    update_index += (coord{d} - start{d}) * {d}u;
            \\
        , .{
            axis,
            output_stride,
            @as(u64, @intCast(output_dim)),
            axis,
            axis + 2,
            axis,
            axis,
            @as(u64, @intCast(max_start)),
            axis,
            axis,
            axis,
            axis,
            axis,
            @as(u64, @intCast(update_dim)),
            axis,
            axis,
            axis,
            update_stride,
        });
    }
    try writer.writeAll("    out0[elem] = inside_update ? in1[update_index] : in0[elem];\n}\n");
}

fn writeDynamicUpdateSliceInPlaceKernel(
    writer: *std.Io.Writer,
    kernel_name: []const u8,
    scalar_type: []const u8,
    plan: *const ir.ExecutablePlan,
    instruction: ir.PlanInstruction,
    input: ir.BufferDescriptor,
    update: ir.BufferDescriptor,
) !void {
    try writer.print(
        \\#include <metal_stdlib>
        \\using namespace metal;
        \\kernel void {s}(
        \\    device const {s}* in0 [[buffer(0)]],
        \\    device const {s}* in1 [[buffer(1)]],
        \\
    , .{ kernel_name, scalar_type, scalar_type });
    for (instruction.inputs[2..], 0..) |start_id, start_index| {
        const start = tensor.valueDescriptor(plan, start_id) orelse unreachable;
        const index_type = tensor.metalIndexScalarType(start.element_type) orelse unreachable;
        try writer.print("    device const {s}* in{d} [[buffer({d})]],\n", .{ index_type, start_index + 2, start_index + 2 });
    }
    try writer.print(
        \\    device {s}* out0 [[buffer({d})]],
        \\    constant uint& count [[buffer({d})]],
        \\    uint elem [[thread_position_in_grid]]) {{
        \\    if (elem >= count) return;
        \\    uint input_index = 0;
        \\
    , .{ scalar_type, instruction.inputs.len, instruction.inputs.len + 1 });
    for (input.dims, update.dims, 0..) |input_dim, update_dim, axis| {
        const update_stride = tensor.denseStride(update.dims, axis + 1);
        const input_stride = tensor.denseStride(input.dims, axis + 1);
        const max_start = input_dim - update_dim;
        try writer.print(
            \\    const uint coord{d} = (elem / {d}u) % {d}u;
            \\    const int raw_start{d} = int(in{d}[0]);
            \\    const uint start{d} = uint(clamp(raw_start{d}, 0, {d}));
            \\    input_index += (start{d} + coord{d}) * {d}u;
            \\
        , .{
            axis,
            update_stride,
            @as(u64, @intCast(update_dim)),
            axis,
            axis + 2,
            axis,
            axis,
            @as(u64, @intCast(max_start)),
            axis,
            axis,
            input_stride,
        });
    }
    try writer.writeAll("    out0[input_index] = in1[elem];\n}\n");
}
