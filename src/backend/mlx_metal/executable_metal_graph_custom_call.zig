const std = @import("std");

const ir = @import("src/compiler/ir");
const custom_call_mod = @import("custom_call.zig");
const map_mod = @import("executable_metal_graph_map.zig");
const metalcpp_call = @import("metalcpp_call.zig");
const msl = @import("executable_metal_graph_msl.zig");
const program_mod = @import("program.zig");
const structural = @import("executable_metal_graph_structural.zig");
const tensor = @import("executable_metal_graph_tensor.zig");

/// Builds generated Metal steps for backend-registered custom calls.
pub fn makeCustomCallStep(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize) program_mod.Error!?metalcpp_call.ExecutableStepSpec {
    const target = instruction.custom_call_target orelse return null;
    const spec = custom_call_mod.lookup(target) orelse return null;
    return switch (spec.kind) {
        .identity => structural.makeCopyLikeStep(allocator, plan, instruction.inputs, instruction.outputs, instruction_index, "custom_identity"),
        .unary => makeCustomUnaryStep(allocator, plan, instruction, instruction_index, spec.unary_op.?),
        .binary => makeCustomBinaryStep(allocator, plan, instruction, instruction_index, spec.binary_op.?),
        .metal_kernel_binary_add_f32 => makeCustomBinaryStep(allocator, plan, instruction, instruction_index, .add),
        .scaled_dot_product_attention => makeScaledDotProductAttentionStep(allocator, plan, instruction, instruction_index),
    };
}

fn makeCustomUnaryStep(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, op: ir.ElementwiseUnaryOp) program_mod.Error!?metalcpp_call.ExecutableStepSpec {
    if (!map_mod.graphUnarySupported(op)) return null;
    if (instruction.inputs.len != 1 or instruction.outputs.len != 1) return null;
    const input = tensor.valueDescriptor(plan, instruction.inputs[0]) orelse return null;
    const output = tensor.valueDescriptor(plan, instruction.outputs[0]) orelse return null;
    if (!tensor.sameDims(input.dims, output.dims)) return null;
    const output_count = tensor.denseElementCount(output);
    if (output_count == 0 or output_count > std.math.maxInt(u32)) return null;
    const scalar_type = tensor.metalScalarType(output.element_type) orelse return null;
    const inputs = try tensor.valueIndices(allocator, instruction.inputs);
    errdefer allocator.free(inputs);
    const outputs = try tensor.valueIndices(allocator, instruction.outputs);
    errdefer allocator.free(outputs);
    const kernel_name = try tensor.kernelName(allocator, "custom_unary", instruction_index);
    errdefer allocator.free(kernel_name);
    var source = std.Io.Writer.Allocating.init(allocator);
    errdefer source.deinit();
    msl.KernelWriter.writeSimpleKernelPrefix(&source.writer, kernel_name, scalar_type, 1, 1) catch return error.OutOfMemory;
    const input_ref = if (tensor.denseElementCount(input) == 1) "in0[0]" else "in0[elem]";
    map_mod.writeUnaryExpression(&source.writer, scalar_type, input_ref, op) catch return error.OutOfMemory;
    return .{
        .kernel_name = kernel_name,
        .source = try source.toOwnedSlice(),
        .inputs = inputs,
        .outputs = outputs,
        .element_count = @intCast(output_count),
    };
}

fn makeCustomBinaryStep(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize, op: ir.ElementwiseBinaryOp) program_mod.Error!?metalcpp_call.ExecutableStepSpec {
    if (!map_mod.graphBinarySupported(op)) return null;
    if (instruction.inputs.len != 2 or instruction.outputs.len != 1) return null;
    const lhs = tensor.valueDescriptor(plan, instruction.inputs[0]) orelse return null;
    const rhs = tensor.valueDescriptor(plan, instruction.inputs[1]) orelse return null;
    const output = tensor.valueDescriptor(plan, instruction.outputs[0]) orelse return null;
    if (lhs.element_type != rhs.element_type or lhs.element_type != output.element_type) return null;
    if (!tensor.sameDims(lhs.dims, rhs.dims) or !tensor.sameDims(lhs.dims, output.dims)) return null;
    const output_count = tensor.denseElementCount(output);
    if (output_count == 0 or output_count > std.math.maxInt(u32)) return null;
    const scalar_type = tensor.metalScalarType(output.element_type) orelse return null;
    const inputs = try tensor.valueIndices(allocator, instruction.inputs);
    errdefer allocator.free(inputs);
    const outputs = try tensor.valueIndices(allocator, instruction.outputs);
    errdefer allocator.free(outputs);
    const kernel_name = try tensor.kernelName(allocator, "custom_binary", instruction_index);
    errdefer allocator.free(kernel_name);
    var source = std.Io.Writer.Allocating.init(allocator);
    errdefer source.deinit();
    msl.KernelWriter.writeSimpleKernelPrefix(&source.writer, kernel_name, scalar_type, 2, 1) catch return error.OutOfMemory;
    map_mod.writeBinaryExpression(&source.writer, scalar_type, "in0[elem]", "in1[elem]", op) catch return error.OutOfMemory;
    return .{
        .kernel_name = kernel_name,
        .source = try source.toOwnedSlice(),
        .inputs = inputs,
        .outputs = outputs,
        .element_count = @intCast(output_count),
    };
}

fn makeScaledDotProductAttentionStep(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize) program_mod.Error!?metalcpp_call.ExecutableStepSpec {
    if (instruction.inputs.len != custom_call_mod.ScaledDotProductAttention.InputCount or instruction.outputs.len != custom_call_mod.ScaledDotProductAttention.OutputCount) return null;
    const q = tensor.valueDescriptor(plan, instruction.inputs[custom_call_mod.ScaledDotProductAttention.Input.q.index()]) orelse return null;
    const k = tensor.valueDescriptor(plan, instruction.inputs[custom_call_mod.ScaledDotProductAttention.Input.k.index()]) orelse return null;
    const v = tensor.valueDescriptor(plan, instruction.inputs[custom_call_mod.ScaledDotProductAttention.Input.v.index()]) orelse return null;
    const token_index = tensor.valueDescriptor(plan, instruction.inputs[custom_call_mod.ScaledDotProductAttention.Input.token_index.index()]) orelse return null;
    const output = tensor.valueDescriptor(plan, instruction.outputs[0]) orelse return null;
    if (q.element_type != k.element_type or q.element_type != v.element_type or q.element_type != output.element_type) return null;
    if (!tensor.sameDims(k.dims, v.dims) or !tensor.sameDims(q.dims, output.dims)) return null;
    if (q.dims.len != 3 and q.dims.len != 4) return null;
    if (k.dims.len != q.dims.len or v.dims.len != q.dims.len) return null;
    if (token_index.dims.len > 1 or (token_index.dims.len == 1 and token_index.dims[0] != 1)) return null;
    if (tensor.metalIndexScalarType(token_index.element_type) == null) return null;
    const query_axis: usize = if (q.dims.len == 4) 1 else 0;
    const head_axis: usize = if (q.dims.len == 4) 2 else 1;
    const dim_axis: usize = if (q.dims.len == 4) 3 else 2;
    if (q.dims[query_axis] <= 0 or q.dims[head_axis] <= 0 or q.dims[dim_axis] <= 0) return null;
    if (k.dims[query_axis] <= 0 or k.dims[head_axis] <= 0 or k.dims[dim_axis] != q.dims[dim_axis]) return null;
    if (@mod(q.dims[head_axis], k.dims[head_axis]) != 0) return null;
    if (q.dims.len == 4 and q.dims[0] != k.dims[0]) return null;
    const output_count = tensor.denseElementCount(output);
    if (output_count == 0 or output_count > std.math.maxInt(u32)) return null;
    const scalar_type = tensor.metalScalarType(output.element_type) orelse return null;
    const token_type = tensor.metalIndexScalarType(token_index.element_type) orelse return null;
    const inputs = try tensor.valueIndices(allocator, instruction.inputs);
    errdefer allocator.free(inputs);
    const outputs = try tensor.valueIndices(allocator, instruction.outputs);
    errdefer allocator.free(outputs);
    const kernel_name = try tensor.kernelName(allocator, "custom_attention", instruction_index);
    errdefer allocator.free(kernel_name);
    var source = std.Io.Writer.Allocating.init(allocator);
    errdefer source.deinit();
    writeScaledDotProductAttentionKernel(&source.writer, kernel_name, scalar_type, token_type, q, k, output) catch return error.OutOfMemory;
    return .{
        .kernel_name = kernel_name,
        .source = try source.toOwnedSlice(),
        .inputs = inputs,
        .outputs = outputs,
        .element_count = @intCast(output_count),
    };
}

fn writeScaledDotProductAttentionKernel(writer: *std.Io.Writer, kernel_name: []const u8, scalar_type: []const u8, token_type: []const u8, q: ir.BufferDescriptor, k: ir.BufferDescriptor, output: ir.BufferDescriptor) !void {
    const has_batch = q.dims.len == 4;
    const query_axis: usize = if (has_batch) 1 else 0;
    const head_axis: usize = if (has_batch) 2 else 1;
    const dim_axis: usize = if (has_batch) 3 else 2;
    const batch: i64 = if (has_batch) q.dims[0] else 1;
    const queries = q.dims[query_axis];
    const q_heads = q.dims[head_axis];
    const kv_len = k.dims[query_axis];
    const kv_heads = k.dims[head_axis];
    const head_dim = q.dims[dim_axis];
    const head_group = @divExact(q_heads, kv_heads);
    try writer.print(
        \\#include <metal_stdlib>
        \\using namespace metal;
        \\kernel void {s}(
        \\    device const {s}* q [[buffer(0)]],
        \\    device const {s}* k [[buffer(1)]],
        \\    device const {s}* v [[buffer(2)]],
        \\    device const {s}* token_index [[buffer(3)]],
        \\    device {s}* out [[buffer(4)]],
        \\    constant uint& count [[buffer(5)]],
        \\    uint elem [[thread_position_in_grid]]) {{
        \\    if (elem >= count) return;
        \\    const uint head_dim = {d}u;
        \\    const uint q_heads = {d}u;
        \\    const uint kv_heads = {d}u;
        \\    const uint kv_len = {d}u;
        \\    const uint queries = {d}u;
        \\    const uint q_head = (elem / head_dim) % q_heads;
        \\    const uint query = (elem / (head_dim * q_heads)) % queries;
        \\    const uint batch = {s};
        \\    const uint dim = elem % head_dim;
        \\    const uint kv_head = q_head / {d}u;
        \\    const uint allowed_last = min(kv_len - 1u, uint(token_index[0]) + query);
        \\    float max_score = -3.402823466e+38f;
        \\
    , .{
        kernel_name,
        scalar_type,
        scalar_type,
        scalar_type,
        token_type,
        scalar_type,
        @as(u64, @intCast(head_dim)),
        @as(u64, @intCast(q_heads)),
        @as(u64, @intCast(kv_heads)),
        @as(u64, @intCast(kv_len)),
        @as(u64, @intCast(queries)),
        if (has_batch) "(elem / (head_dim * q_heads * queries))" else "0u",
        @as(u64, @intCast(head_group)),
    });
    try writer.print(
        \\    for (uint key = 0; key < kv_len; ++key) {{
        \\        if (key > allowed_last) continue;
        \\        float score = 0.0f;
        \\        for (uint inner = 0; inner < head_dim; ++inner) {{
        \\            const uint q_index = {s};
        \\            const uint k_index = {s};
        \\            score += float(q[q_index]) * float(k[k_index]);
        \\        }}
        \\        score *= rsqrt(float(head_dim));
        \\        max_score = max(max_score, score);
        \\    }}
        \\    float denom = 0.0f;
        \\    float acc = 0.0f;
        \\    for (uint key = 0; key < kv_len; ++key) {{
        \\        if (key > allowed_last) continue;
        \\        float score = 0.0f;
        \\        for (uint inner = 0; inner < head_dim; ++inner) {{
        \\            const uint q_index = {s};
        \\            const uint k_index = {s};
        \\            score += float(q[q_index]) * float(k[k_index]);
        \\        }}
        \\        score *= rsqrt(float(head_dim));
        \\        const float weight = exp(score - max_score);
        \\        const uint v_index = {s};
        \\        denom += weight;
        \\        acc += weight * float(v[v_index]);
        \\    }}
        \\    out[elem] = {s}(acc / denom);
        \\}}
        \\
    , .{
        if (has_batch) "(((batch * queries + query) * q_heads + q_head) * head_dim + inner)" else "((query * q_heads + q_head) * head_dim + inner)",
        if (has_batch) "(((batch * kv_len + key) * kv_heads + kv_head) * head_dim + inner)" else "((key * kv_heads + kv_head) * head_dim + inner)",
        if (has_batch) "(((batch * queries + query) * q_heads + q_head) * head_dim + inner)" else "((query * q_heads + q_head) * head_dim + inner)",
        if (has_batch) "(((batch * kv_len + key) * kv_heads + kv_head) * head_dim + inner)" else "((key * kv_heads + kv_head) * head_dim + inner)",
        if (has_batch) "(((batch * kv_len + key) * kv_heads + kv_head) * head_dim + dim)" else "((key * kv_heads + kv_head) * head_dim + dim)",
        scalar_type,
    });
    _ = output;
    _ = batch;
}
