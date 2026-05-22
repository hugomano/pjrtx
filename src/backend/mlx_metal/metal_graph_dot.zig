const std = @import("std");

const ir = @import("src/compiler/ir");
const metalcpp_call = @import("metalcpp_call.zig");
const program_mod = @import("program.zig");

const tiled_threadgroup_width: u64 = 256;
const dot_column_tile: u64 = 4;

/// Builds a Metal graph step for one supported `dot_general` instruction.
/// The returned step owns its allocated kernel name, source, inputs, and
/// outputs; callers must release those fields with the executable graph step
/// cleanup path after handing the step to the Metal graph program.
pub fn makeStep(
    allocator: std.mem.Allocator,
    plan: *const ir.ExecutablePlan,
    instruction: ir.PlanInstruction,
    instruction_index: usize,
) program_mod.Error!?metalcpp_call.ExecutableStepSpec {
    if (instruction.inputs.len != 2 or instruction.outputs.len != 1) return null;
    const lhs = valueDescriptor(plan, instruction.inputs[0]) orelse return null;
    const rhs = valueDescriptor(plan, instruction.inputs[1]) orelse return null;
    const output = valueDescriptor(plan, instruction.outputs[0]) orelse return null;
    if (lhs.element_type != rhs.element_type or lhs.element_type != output.element_type or lhs.dims.len != 2 or rhs.dims.len != 2 or output.dims.len != 2) return null;
    const lhs_contract = instruction.lhs_contracting_dimensions orelse return null;
    const rhs_contract = instruction.rhs_contracting_dimensions orelse return null;
    if (lhs_contract.len != 1 or rhs_contract.len != 1 or lhs_contract[0] != 1) return null;
    const rhs_contract_axis_i64 = rhs_contract[0];
    if (rhs_contract_axis_i64 != 0 and rhs_contract_axis_i64 != 1) return null;
    const rhs_contract_axis: usize = @intCast(rhs_contract_axis_i64);
    if (instruction.lhs_batch_dimensions) |dims| if (dims.len != 0) return null;
    if (instruction.rhs_batch_dimensions) |dims| if (dims.len != 0) return null;
    const m = lhs.dims[0];
    const k = lhs.dims[1];
    const n = rhs.dims[1 - rhs_contract_axis];
    if (rhs.dims[rhs_contract_axis] != k or output.dims[0] != m or output.dims[1] != n) return null;
    const output_count = denseElementCount(output);
    const scalar_type = metalScalarType(output.element_type) orelse return null;
    if (try makeTiledStep(allocator, instruction, instruction_index, output_count, rhs_contract_axis, @intCast(m), @intCast(n), @intCast(k), scalar_type, output.element_type)) |step| return step;
    const inputs = try valueIndices(allocator, instruction.inputs);
    errdefer allocator.free(inputs);
    const outputs = try valueIndices(allocator, instruction.outputs);
    errdefer allocator.free(outputs);
    const kernel_name = try kernelName(allocator, "dot", instruction_index);
    errdefer allocator.free(kernel_name);
    var source = std.Io.Writer.Allocating.init(allocator);
    errdefer source.deinit();
    writeSimpleKernelPrefix(&source.writer, kernel_name, scalar_type, 2, 1) catch return error.OutOfMemory;
    const rhs_expr = if (rhs_contract_axis == 0) "in1[i * n + col]" else "in1[col * k + i]";
    source.writer.print(
        \\    const uint m = {d};
        \\    const uint n = {d};
        \\    const uint k = {d};
        \\    const uint row = elem / n;
        \\    const uint col = elem - row * n;
        \\    float acc = 0.0f;
        \\    for (uint i = 0; i < k; ++i) {{
        \\        acc += float(in0[row * k + i]) * float({s});
        \\    }}
        \\    out0[elem] = {s}(acc);
        \\}}
        \\
    , .{ m, n, k, rhs_expr, scalar_type }) catch return error.OutOfMemory;
    return .{ .kernel_name = kernel_name, .source = try source.toOwnedSlice(), .inputs = inputs, .outputs = outputs, .element_count = @intCast(output_count) };
}

/// Builds one tiled Metal graph step for adjacent dot nodes that share an LHS.
/// The returned step owns its generated source and value lists; callers append
/// release metadata and deinitialize through the executable graph step owner.
pub fn makeGroupStep(
    allocator: std.mem.Allocator,
    plan: *const ir.ExecutablePlan,
    program: *const program_mod.Program,
    node_indices: []const usize,
) program_mod.Error!?metalcpp_call.ExecutableStepSpec {
    if (node_indices.len < 2) return null;
    const first_node_index = node_indices[0];
    if (first_node_index >= program.nodes.len) return error.CommandSubmissionFailed;
    const first_instruction_index = program.nodes[first_node_index].instruction_index;
    if (first_instruction_index >= plan.instructions.len) return error.CommandSubmissionFailed;
    const first_instruction = plan.instructions[first_instruction_index];
    if (first_instruction.inputs.len != 2) return null;
    const lhs = valueDescriptor(plan, first_instruction.inputs[0]) orelse return null;
    if (!tiledFloatAccumulatorType(lhs.element_type) or lhs.dims.len != 2) return null;
    const k: u64 = @intCast(lhs.dims[1]);
    if (k < tiled_threadgroup_width) return null;

    const inputs = try allocator.alloc(u64, node_indices.len + 1);
    var inputs_owned = true;
    defer if (inputs_owned) allocator.free(inputs);
    inputs[0] = first_instruction.inputs[0].index;
    const outputs = try allocator.alloc(u64, node_indices.len);
    var outputs_owned = true;
    defer if (outputs_owned) allocator.free(outputs);
    const members = try allocator.alloc(DotGroupMember, node_indices.len);
    defer allocator.free(members);

    var total_tile_count: u64 = 0;
    for (node_indices, 0..) |node_index, member_index| {
        if (node_index >= program.nodes.len) return error.CommandSubmissionFailed;
        const node = program.nodes[node_index];
        if (node.kind != .matmul or node.instruction_index >= plan.instructions.len) return null;
        const instruction = plan.instructions[node.instruction_index];
        if (instruction.kind != .dot_general or instruction.inputs.len != 2 or instruction.outputs.len != 1) return null;
        if (instruction.inputs[0].index != inputs[0]) return null;
        const rhs = valueDescriptor(plan, instruction.inputs[1]) orelse return null;
        const output = valueDescriptor(plan, instruction.outputs[0]) orelse return null;
        if (rhs.element_type != lhs.element_type or output.element_type != lhs.element_type or rhs.dims.len != 2 or output.dims.len != 2) return null;
        const lhs_contract = instruction.lhs_contracting_dimensions orelse return null;
        const rhs_contract = instruction.rhs_contracting_dimensions orelse return null;
        if (lhs_contract.len != 1 or rhs_contract.len != 1 or lhs_contract[0] != 1) return null;
        const rhs_contract_axis_i64 = rhs_contract[0];
        if (rhs_contract_axis_i64 != 0 and rhs_contract_axis_i64 != 1) return null;
        const rhs_contract_axis: usize = @intCast(rhs_contract_axis_i64);
        if (instruction.lhs_batch_dimensions) |dims| if (dims.len != 0) return null;
        if (instruction.rhs_batch_dimensions) |dims| if (dims.len != 0) return null;
        const n: u64 = @intCast(rhs.dims[1 - rhs_contract_axis]);
        if (rhs.dims[rhs_contract_axis] != lhs.dims[1] or output.dims[0] != lhs.dims[0] or output.dims[1] != @as(i64, @intCast(n))) return null;
        const output_count = @as(u64, @intCast(denseElementCount(output)));
        if (output_count == 0) return null;
        inputs[member_index + 1] = instruction.inputs[1].index;
        outputs[member_index] = instruction.outputs[0].index;
        members[member_index] = .{
            .rhs_contract_axis = rhs_contract_axis,
            .m = @intCast(lhs.dims[0]),
            .n = n,
            .k = k,
            .output_count = output_count,
            .tile_count = dotTileCount(@intCast(lhs.dims[0]), n),
        };
        total_tile_count += members[member_index].tile_count;
    }
    const total_threads = total_tile_count * tiled_threadgroup_width;
    if (total_threads == 0 or total_threads > std.math.maxInt(u32)) return null;

    const scalar_type = metalScalarType(lhs.element_type) orelse return null;
    const kernel_name = try kernelName(allocator, "tiled_dot_group", first_instruction_index);
    errdefer allocator.free(kernel_name);
    var source = std.Io.Writer.Allocating.init(allocator);
    errdefer source.deinit();
    writeTiledDotGroupKernel(&source.writer, kernel_name, scalar_type, members, total_tile_count) catch return error.OutOfMemory;
    inputs_owned = false;
    outputs_owned = false;
    return .{
        .kernel_name = kernel_name,
        .source = try source.toOwnedSlice(),
        .inputs = inputs,
        .outputs = outputs,
        .element_count = total_threads,
        .threads_per_threadgroup = @intCast(tiled_threadgroup_width),
    };
}

const DotGroupMember = struct {
    rhs_contract_axis: usize,
    m: u64,
    n: u64,
    k: u64,
    output_count: u64,
    tile_count: u64,
};

fn makeTiledStep(
    allocator: std.mem.Allocator,
    instruction: ir.PlanInstruction,
    instruction_index: usize,
    output_count: usize,
    rhs_contract_axis: usize,
    m: u64,
    n: u64,
    k: u64,
    scalar_type: []const u8,
    element_type: ir.BufferType,
) program_mod.Error!?metalcpp_call.ExecutableStepSpec {
    _ = output_count;
    if (k < tiled_threadgroup_width) return null;
    if (!tiledFloatAccumulatorType(element_type)) return null;
    const total_threads = dotTileCount(m, n) * tiled_threadgroup_width;
    if (total_threads > std.math.maxInt(u32)) return null;
    const inputs = try valueIndices(allocator, instruction.inputs);
    errdefer allocator.free(inputs);
    const outputs = try valueIndices(allocator, instruction.outputs);
    errdefer allocator.free(outputs);
    const kernel_name = try kernelName(allocator, "tiled_dot", instruction_index);
    errdefer allocator.free(kernel_name);
    var source = std.Io.Writer.Allocating.init(allocator);
    errdefer source.deinit();
    writeTiledDotKernel(&source.writer, kernel_name, scalar_type, rhs_contract_axis, m, n, k) catch return error.OutOfMemory;
    return .{
        .kernel_name = kernel_name,
        .source = try source.toOwnedSlice(),
        .inputs = inputs,
        .outputs = outputs,
        .element_count = total_threads,
        .threads_per_threadgroup = @intCast(tiled_threadgroup_width),
    };
}

fn writeTiledDotKernel(
    writer: *std.Io.Writer,
    kernel_name: []const u8,
    scalar_type: []const u8,
    rhs_contract_axis: usize,
    m: u64,
    n: u64,
    k: u64,
) !void {
    const rhs_expr = if (rhs_contract_axis == 0) "in1[i * n + col]" else "in1[col * k + i]";
    try writer.print(
        \\#include <metal_stdlib>
        \\using namespace metal;
        \\kernel void {s}(
        \\    device const {s}* in0 [[buffer(0)]],
        \\    device const {s}* in1 [[buffer(1)]],
        \\    device {s}* out0 [[buffer(2)]],
        \\    constant uint& count [[buffer(3)]],
        \\    uint3 elem_pos [[thread_position_in_grid]],
        \\    uint3 tid_pos [[thread_position_in_threadgroup]],
        \\    uint3 group_pos [[threadgroup_position_in_grid]]) {{
        \\    const uint elem = elem_pos.x;
        \\    const uint tid = tid_pos.x;
        \\    if (elem >= count) return;
        \\    const uint tile = group_pos.x;
        \\    if (tile >= {d}u) return;
        \\    const uint n = {d}u;
        \\    const uint k = {d}u;
        \\    const uint col_tiles = (n + {d}u) / {d}u;
        \\    const uint row = tile / col_tiles;
        \\    const uint col_base = (tile - row * col_tiles) * {d}u;
        \\    threadgroup float partials[{d}];
        \\
    , .{
        kernel_name,
        scalar_type,
        scalar_type,
        scalar_type,
        dotTileCount(m, n),
        n,
        k,
        dot_column_tile - 1,
        dot_column_tile,
        dot_column_tile,
        dot_column_tile * tiled_threadgroup_width,
    });
    try writeTiledDotAccumulators(writer, "    ");
    try writer.print(
        \\    for (uint i = tid; i < k; i += {d}u) {{
        \\        const float lhs = float(in0[row * k + i]);
        \\
    , .{tiled_threadgroup_width});
    try writeTiledDotAccumulation(writer, "        ", rhs_expr);
    try writer.writeAll("    }\n");
    try writeTiledDotReduction(writer, "    ");
    try writeTiledDotStores(writer, "    ", "out0", scalar_type);
    try writer.writeAll("}\n");
}

fn writeTiledDotGroupKernel(writer: *std.Io.Writer, kernel_name: []const u8, scalar_type: []const u8, members: []const DotGroupMember, total_tile_count: u64) !void {
    try writer.print(
        \\#include <metal_stdlib>
        \\using namespace metal;
        \\kernel void {s}(
        \\    device const {s}* in0 [[buffer(0)]],
        \\
    , .{ kernel_name, scalar_type });
    for (members, 0..) |_, member_index| {
        try writer.print("    device const {s}* in{d} [[buffer({d})]],\n", .{ scalar_type, member_index + 1, member_index + 1 });
    }
    for (members, 0..) |_, member_index| {
        try writer.print("    device {s}* out{d} [[buffer({d})]],\n", .{ scalar_type, member_index, members.len + 1 + member_index });
    }
    try writer.print(
        \\    constant uint& count [[buffer({d})]],
        \\    uint3 elem_pos [[thread_position_in_grid]],
        \\    uint3 tid_pos [[thread_position_in_threadgroup]],
        \\    uint3 group_pos [[threadgroup_position_in_grid]]) {{
        \\    const uint elem = elem_pos.x;
        \\    const uint tid = tid_pos.x;
        \\    if (elem >= count) return;
        \\    const uint global_tile = group_pos.x;
        \\    if (global_tile >= {d}u) return;
        \\    threadgroup float partials[{d}];
        \\    uint local = global_tile;
        \\
    , .{ members.len * 2 + 1, total_tile_count, dot_column_tile * tiled_threadgroup_width });
    for (members, 0..) |member, member_index| {
        try writer.print(
            \\    if (local < {d}u) {{
            \\        const uint n = {d}u;
            \\        const uint k = {d}u;
            \\        const uint col_tiles = (n + {d}u) / {d}u;
            \\        const uint row = local / col_tiles;
            \\        const uint col_base = (local - row * col_tiles) * {d}u;
            \\
        , .{
            member.tile_count,
            member.n,
            member.k,
            dot_column_tile - 1,
            dot_column_tile,
            dot_column_tile,
        });
        try writeTiledDotAccumulators(writer, "        ");
        try writer.print(
            \\        for (uint i = tid; i < k; i += {d}u) {{
            \\            const float lhs = float(in0[row * k + i]);
            \\
        , .{tiled_threadgroup_width});
        try writeTiledDotGroupAccumulation(writer, "            ", member_index + 1, member.rhs_contract_axis);
        try writer.writeAll("        }\n");
        try writeTiledDotReduction(writer, "        ");
        try writeTiledDotGroupStores(writer, "        ", member_index, scalar_type);
        try writer.print(
            \\        return;
            \\    }}
            \\    local -= {d}u;
            \\
        , .{member.tile_count});
    }
    try writer.writeAll(
        \\}
        \\
    );
}

fn writeTiledDotAccumulators(writer: *std.Io.Writer, indent: []const u8) !void {
    inline for (0..dot_column_tile) |column| {
        try writer.print("{s}float acc{d} = 0.0f;\n", .{ indent, column });
    }
}

fn writeTiledDotAccumulation(writer: *std.Io.Writer, indent: []const u8, rhs_expr: []const u8) !void {
    inline for (0..dot_column_tile) |column| {
        if (column == 0) {
            try writer.print("{s}uint col = col_base;\n", .{indent});
        } else {
            try writer.print("{s}col = col_base + {d}u;\n", .{ indent, column });
        }
        try writer.print("{s}if (col < n) acc{d} += lhs * float({s});\n", .{ indent, column, rhs_expr });
    }
}

fn writeTiledDotGroupAccumulation(writer: *std.Io.Writer, indent: []const u8, input_index: usize, rhs_contract_axis: usize) !void {
    inline for (0..dot_column_tile) |column| {
        if (column == 0) {
            try writer.print("{s}uint col = col_base;\n", .{indent});
        } else {
            try writer.print("{s}col = col_base + {d}u;\n", .{ indent, column });
        }
        try writer.print("{s}if (col < n) acc{d} += lhs * float(", .{ indent, column });
        if (rhs_contract_axis == 0) {
            try writer.print("in{d}[i * n + col]", .{input_index});
        } else {
            try writer.print("in{d}[col * k + i]", .{input_index});
        }
        try writer.writeAll(");\n");
    }
}

fn writeTiledDotReduction(writer: *std.Io.Writer, indent: []const u8) !void {
    inline for (0..dot_column_tile) |column| {
        try writer.print("{s}partials[tid + {d}u] = acc{d};\n", .{ indent, column * tiled_threadgroup_width, column });
    }
    try writer.print(
        \\{s}threadgroup_barrier(mem_flags::mem_threadgroup);
        \\{s}for (uint stride = {d}u; stride > 0u; stride >>= 1u) {{
        \\{s}    if (tid < stride) {{
        \\
    , .{ indent, indent, tiled_threadgroup_width / 2, indent });
    inline for (0..dot_column_tile) |column| {
        const offset = column * tiled_threadgroup_width;
        try writer.print("{s}        partials[tid + {d}u] += partials[tid + {d}u + stride];\n", .{ indent, offset, offset });
    }
    try writer.print(
        \\{s}    }}
        \\{s}    threadgroup_barrier(mem_flags::mem_threadgroup);
        \\{s}}}
        \\
    , .{ indent, indent, indent });
}

fn writeTiledDotStores(writer: *std.Io.Writer, indent: []const u8, output_name: []const u8, scalar_type: []const u8) !void {
    try writer.print(
        \\{s}if (tid == 0u) {{
        \\
    , .{indent});
    inline for (0..dot_column_tile) |column| {
        if (column == 0) {
            try writer.print("{s}    uint col = col_base;\n", .{indent});
        } else {
            try writer.print("{s}    col = col_base + {d}u;\n", .{ indent, column });
        }
        try writer.print("{s}    if (col < n) {s}[row * n + col] = {s}(partials[{d}]);\n", .{ indent, output_name, scalar_type, column * tiled_threadgroup_width });
    }
    try writer.print(
        \\{s}}}
        \\
    , .{indent});
}

fn writeTiledDotGroupStores(writer: *std.Io.Writer, indent: []const u8, output_index: usize, scalar_type: []const u8) !void {
    try writer.print(
        \\{s}if (tid == 0u) {{
        \\
    , .{indent});
    inline for (0..dot_column_tile) |column| {
        if (column == 0) {
            try writer.print("{s}    uint col = col_base;\n", .{indent});
        } else {
            try writer.print("{s}    col = col_base + {d}u;\n", .{ indent, column });
        }
        try writer.print("{s}    if (col < n) out{d}[row * n + col] = {s}(partials[{d}]);\n", .{ indent, output_index, scalar_type, column * tiled_threadgroup_width });
    }
    try writer.print(
        \\{s}}}
        \\
    , .{indent});
}

fn writeSimpleKernelPrefix(writer: *std.Io.Writer, kernel_name: []const u8, scalar_type: []const u8, input_count: usize, output_count: usize) !void {
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

fn valueDescriptor(plan: *const ir.ExecutablePlan, value_id: ir.ValueId) ?ir.BufferDescriptor {
    if (value_id.index >= plan.values.len) return null;
    const value = plan.values[value_id.index];
    if (value.storage != .tensor or value.descriptor.layout != .dense_row_major) return null;
    return value.descriptor;
}

fn valueIndices(allocator: std.mem.Allocator, value_ids: []const ir.ValueId) ![]u64 {
    const indices = try allocator.alloc(u64, value_ids.len);
    for (indices, value_ids) |*index, value_id| index.* = value_id.index;
    return indices;
}

fn kernelName(allocator: std.mem.Allocator, label: []const u8, instruction_index: usize) ![]const u8 {
    return std.fmt.allocPrint(allocator, "pjrtx_metalcpp_{s}_{d}", .{ label, instruction_index });
}

fn dotTileCount(m: u64, n: u64) u64 {
    return m * @divTrunc(n + dot_column_tile - 1, dot_column_tile);
}

fn metalScalarType(element_type: ir.BufferType) ?[]const u8 {
    return switch (element_type) {
        .pred => "bool",
        .bf16 => "bfloat",
        .f16 => "half",
        .f32 => "float",
        else => null,
    };
}

fn tiledFloatAccumulatorType(element_type: ir.BufferType) bool {
    return switch (element_type) {
        .bf16, .f16, .f32 => true,
        else => false,
    };
}

fn denseElementCount(descriptor: ir.BufferDescriptor) usize {
    const byte_size = ir.denseByteSize(descriptor.element_type, descriptor.dims);
    const element_size = descriptor.element_type.byteSize();
    if (byte_size == 0 or element_size == 0) return 0;
    return byte_size / element_size;
}

fn deinitStep(allocator: std.mem.Allocator, step: metalcpp_call.ExecutableStepSpec) void {
    allocator.free(step.kernel_name);
    allocator.free(step.source);
    allocator.free(step.inputs);
    allocator.free(step.outputs);
    allocator.free(step.release_values);
}
