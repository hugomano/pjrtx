const std = @import("std");

const ir = @import("src/compiler/ir");
const program_build_mod = @import("program_build.zig");

/// Describes a rejected MLX backend lowering form with diagnostic context.
pub const Issue = struct {
    instruction_index: ?usize = null,
    value_id: ?ir.ValueId = null,
    op: ?ir.PlanInstructionKind = null,
    detail: []const u8,
    feature: []const u8 = "mlx-backend-executable",
};

/// Writes a backend-program diagnostic when validation found no precise issue.
pub fn writeRejectedWithoutSpecificIssue(plan: *const ir.ExecutablePlan, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    var program = program_build_mod.build(std.heap.page_allocator, plan, writer) catch |err| {
        if (err == error.InvalidProgram) return;
        if (err == error.OutOfMemory) {
            try writer.writeAll("invalid executable plan: pass=mlx-backend-legalization detail=\"failed to allocate backend program diagnostic\" feature=mlx-backend-executable");
            return;
        }
        try writer.writeAll("invalid executable plan: pass=mlx-backend-legalization detail=\"failed to build backend program diagnostic\" feature=mlx-backend-executable");
        return;
    };
    defer program.deinit();
    try writer.writeAll("invalid executable plan: pass=mlx-backend-legalization detail=\"MLX backend rejected executable plan without a specific issue\" feature=mlx-backend-executable");
}

/// Writes a stable human-readable lowering diagnostic for one issue.
pub fn writeIssue(plan: *const ir.ExecutablePlan, issue: Issue, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    try writer.writeAll("invalid executable plan: pass=mlx-backend-legalization");
    if (issue.instruction_index) |index| try writer.print(" instruction={d}", .{index});
    if (issue.op) |op| try writer.print(" op={s}", .{@tagName(op)});
    if (issue.value_id) |value_id| {
        try writer.print(" value={d}", .{value_id.index});
        if (value_id.index < plan.values.len) {
            const descriptor = plan.values[value_id.index].descriptor;
            try writer.print(" dtype={s} rank={d} shape=", .{ @tagName(descriptor.element_type), descriptor.dims.len });
            try writeDims(writer, descriptor.dims);
            try writer.print(" sharding={s}", .{shardingLabel(plan, value_id)});
        }
    }
    try writer.print(" detail=\"{s}\" feature={s}", .{ issue.detail, issue.feature });
}

fn writeDims(writer: *std.Io.Writer, dims: []const i64) std.Io.Writer.Error!void {
    try writer.writeAll("[");
    for (dims, 0..) |dim, index| {
        if (index != 0) try writer.writeAll(",");
        try writer.print("{d}", .{dim});
    }
    try writer.writeAll("]");
}

fn shardingLabel(plan: *const ir.ExecutablePlan, value_id: ir.ValueId) []const u8 {
    for (plan.output_ids, 0..) |output_id, index| {
        if (output_id.index == value_id.index and index < plan.output_shardings.len) return @tagName(plan.output_shardings[index].kind);
    }
    if (value_id.index < plan.values.len and plan.values[value_id.index].role == .parameter) {
        var parameter_index: usize = 0;
        for (plan.values) |value| {
            if (value.role != .parameter) continue;
            if (value.id.index == value_id.index and parameter_index < plan.parameter_shardings.len) return @tagName(plan.parameter_shardings[parameter_index].kind);
            parameter_index += 1;
        }
    }
    return "internal";
}
