const std = @import("std");

const ir = @import("src/compiler/ir");
const mlx_metal = @import("src/backend/mlx_metal");

/// Writes a list of executable-plan values using manifest/debug artifact syntax.
pub fn writeValueRefList(writer: *std.Io.Writer, plan: *const ir.ExecutablePlan, label: []const u8, values: []const ir.ValueId) !void {
    try writer.print("{s}=[", .{label});
    for (values, 0..) |value, index| {
        if (index != 0) try writer.writeAll(",");
        try writeValueRef(writer, plan, value);
    }
    try writer.writeByte(']');
}

/// Writes one executable-plan value reference with dtype and shape context.
pub fn writeValueRef(writer: *std.Io.Writer, plan: *const ir.ExecutablePlan, value: ir.ValueId) !void {
    try writer.print("v{d}", .{value.index});
    if (value.index >= plan.values.len) {
        try writer.writeAll(":unknown");
        return;
    }
    const plan_value = plan.values[value.index];
    try writer.print(":{s}", .{@tagName(plan_value.descriptor.element_type)});
    try writeShape(writer, plan_value.descriptor.dims);
}

/// Writes a tensor descriptor for artifact manifests.
pub fn writeDescriptor(writer: *std.Io.Writer, descriptor: ir.BufferDescriptor) !void {
    try writer.print("dtype={s} shape=", .{@tagName(descriptor.element_type)});
    try writeShape(writer, descriptor.dims);
    try writer.print(" layout={s} device={d} memory={d} shard={d}", .{
        @tagName(descriptor.layout),
        descriptor.device_id,
        descriptor.memory_id,
        descriptor.shard_index,
    });
}

/// Writes dense tensor dimensions as `[d0,d1,...]`.
pub fn writeShape(writer: *std.Io.Writer, dims: []const i64) !void {
    try writer.writeByte('[');
    for (dims, 0..) |dim, index| {
        if (index != 0) try writer.writeAll(",");
        try writer.print("{d}", .{dim});
    }
    try writer.writeByte(']');
}

/// Writes a list of backend-program indices for artifact manifests.
pub fn writeUsizeList(writer: *std.Io.Writer, label: []const u8, values: []const usize) !void {
    try writer.print("{s}=[", .{label});
    for (values, 0..) |value, index| {
        if (index != 0) try writer.writeAll(",");
        try writer.print("{d}", .{value});
    }
    try writer.writeByte(']');
}

/// Writes an optional backend-program index using manifest syntax.
pub fn writeOptionalUsize(writer: *std.Io.Writer, label: []const u8, value: ?usize) !void {
    try writer.print("{s}=", .{label});
    if (value) |some| {
        try writer.print("{d}", .{some});
    } else {
        try writer.writeAll("none");
    }
}

/// Writes StableHLO-derived instruction attributes into source comments/manifests.
pub fn writeInstructionAttributes(writer: *std.Io.Writer, instruction: ir.PlanInstruction) !void {
    try writeOptionalDimsField(writer, " dims", instruction.dims);
    try writeOptionalDimsField(writer, " permutation", instruction.permutation);
    try writeOptionalDimsField(writer, " broadcast_dims", instruction.broadcast_dimensions);
    try writeOptionalDimsField(writer, " start", instruction.start_indices);
    try writeOptionalDimsField(writer, " limit", instruction.limit_indices);
    try writeOptionalDimsField(writer, " strides", instruction.strides);
    try writeOptionalDimsField(writer, " slice_sizes", instruction.slice_sizes);
    try writeOptionalDimsField(writer, " window", instruction.window_dimensions);
    try writeOptionalDimsField(writer, " reduce_dims", instruction.reduce_dimensions);
    try writeOptionalDimsField(writer, " lhs_batch", instruction.lhs_batch_dimensions);
    try writeOptionalDimsField(writer, " rhs_batch", instruction.rhs_batch_dimensions);
    try writeOptionalDimsField(writer, " lhs_contract", instruction.lhs_contracting_dimensions);
    try writeOptionalDimsField(writer, " rhs_contract", instruction.rhs_contracting_dimensions);
    try writeOptionalDimsField(writer, " dimensions", instruction.dimensions);
    try writeOptionalI64Field(writer, " dimension", instruction.dimension);
    try writeOptionalI64Field(writer, " tuple_index", instruction.tuple_index);
    try writeOptionalI64Field(writer, " iota_dimension", instruction.iota_dimension);
    try writeOptionalI64Field(writer, " top_k_k", instruction.top_k_k);
    if (instruction.compare_direction) |direction| {
        try writer.print(" compare={s}", .{@tagName(direction)});
    }
    if (instruction.fft_kind) |kind| {
        try writer.print(" fft={s}", .{@tagName(kind)});
    }
    if (instruction.rng_distribution) |distribution| {
        try writer.print(" rng={s}", .{@tagName(distribution)});
    }
    if (instruction.custom_call_target) |target| {
        try writer.print(" custom_call=\"{s}\"", .{target});
    }
    if (instruction.literal) |literal| {
        try writer.print(" literal_bytes={d}", .{literal.len});
    }
    if (instruction.region_ids.len != 0) {
        try writer.writeAll(" regions=[");
        for (instruction.region_ids, 0..) |region_id, index| {
            if (index != 0) try writer.writeAll(",");
            try writer.print("r{d}", .{region_id.index});
        }
        try writer.writeByte(']');
    }
}

/// Writes operation names represented by one fusion group.
pub fn writeFusionGroupOps(writer: *std.Io.Writer, plan: *const ir.ExecutablePlan, program: *const mlx_metal.Program, group: mlx_metal.FusionGroup) !void {
    try writer.writeAll(" ops=[");
    var wrote_any = false;
    for (group.node_indices) |node_index| {
        if (node_index >= program.nodes.len) continue;
        const instruction_index = program.nodes[node_index].instruction_index;
        if (instruction_index >= plan.instructions.len) continue;
        if (wrote_any) try writer.writeByte(',');
        try writer.print("{s}", .{@tagName(plan.instructions[instruction_index].kind)});
        wrote_any = true;
    }
    try writer.writeByte(']');
}

fn writeOptionalDimsField(writer: *std.Io.Writer, label: []const u8, dims: ?[]const i64) !void {
    const values = dims orelse return;
    try writer.print("{s}=", .{label});
    try writeShape(writer, values);
}

fn writeOptionalI64Field(writer: *std.Io.Writer, label: []const u8, value: ?i64) !void {
    const some = value orelse return;
    try writer.print("{s}={d}", .{ label, some });
}

