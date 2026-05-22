const std = @import("std");

const ir = @import("src/compiler/ir");
const map_rules = @import("executable_metal_graph_map_rules.zig");
const program_mod = @import("program.zig");
const storage = @import("executable_metal_graph_graph_request_storage.zig");
const tensor = @import("executable_metal_graph_tensor.zig");

/// Routes one plan instruction into the single-kernel expression store.
pub const GraphExpressionRecorder = struct {
    /// Records one supported instruction or returns null when the graph path cannot lower it.
    pub fn run(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, request: *storage.Request, instruction: ir.PlanInstruction) program_mod.Error!?void {
        if (map_rules.kindFor(instruction.kind)) |map_kind| {
            return MapExpression.record(allocator, plan, request, instruction, map_kind);
        }
        return switch (instruction.kind) {
            .constant, .tuple => {},
            .reshape, .broadcast_in_dim, .copy_arg0, .reduce_precision, .convert, .bitcast_convert => AliasExpression.record(allocator, plan, request, instruction),
            .get_tuple_element => TupleElementExpression.record(allocator, plan, request, instruction),
            else => null,
        };
    }
};

const MapExpression = struct {
    fn record(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, request: *storage.Request, instruction: ir.PlanInstruction, map_kind: map_rules.Kind) program_mod.Error!?void {
        if (instruction.inputs.len != map_rules.inputCount(map_kind) or instruction.outputs.len != 1) return null;
        const output_id = instruction.outputs[0];
        if (map_kind == .compare) {
            if (!storage.ExpressionStore.predicateCompatible(request.*, plan, output_id)) return null;
        } else if (!storage.ExpressionStore.outputCompatible(request.*, plan, output_id)) return null;
        for (instruction.inputs, 0..) |input_id, input_index| {
            const compatible = if (map_kind == .select and input_index == 0)
                storage.ExpressionStore.predicateCompatible(request.*, plan, input_id)
            else
                storage.ExpressionStore.inputCompatible(request.*, plan, input_id);
            if (!compatible) return null;
        }
        const rendered = (try render(allocator, request.*, instruction, map_kind)) orelse return null;
        try storage.ExpressionStore.put(request, allocator, output_id, rendered);
        return {};
    }

    fn render(allocator: std.mem.Allocator, request: storage.Request, instruction: ir.PlanInstruction, map_kind: map_rules.Kind) !?[]const u8 {
        const input0 = storage.ExpressionStore.get(request, instruction.inputs[0]) orelse return null;
        return switch (map_kind) {
            .binary => map_rules.binaryExpression(allocator, instruction.kind, input0, storage.ExpressionStore.get(request, instruction.inputs[1]) orelse return null),
            .unary => map_rules.unaryExpression(allocator, instruction.kind, input0, tensor.metalScalarType(request.output_type) orelse return null, .typed),
            .compare => try map_rules.compareExpression(allocator, input0, storage.ExpressionStore.get(request, instruction.inputs[1]) orelse return null, instruction.compare_direction orelse .eq),
            .select => try map_rules.selectExpression(allocator, input0, storage.ExpressionStore.get(request, instruction.inputs[1]) orelse return null, storage.ExpressionStore.get(request, instruction.inputs[2]) orelse return null),
            .clamp => try map_rules.clampExpression(allocator, input0, storage.ExpressionStore.get(request, instruction.inputs[1]) orelse return null, storage.ExpressionStore.get(request, instruction.inputs[2]) orelse return null),
        };
    }
};

const AliasExpression = struct {
    fn record(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, request: *storage.Request, instruction: ir.PlanInstruction) program_mod.Error!?void {
        if (instruction.inputs.len != 1 or instruction.outputs.len != 1) return null;
        const input_id = instruction.inputs[0];
        const output_id = instruction.outputs[0];
        if (input_id.index >= plan.values.len or output_id.index >= plan.values.len) return null;
        if (!storage.ExpressionStore.outputCompatible(request.*, plan, output_id)) return null;
        const input_descriptor = plan.values[input_id.index].descriptor;
        const output_descriptor = plan.values[output_id.index].descriptor;
        switch (instruction.kind) {
            .reshape => {
                if (input_descriptor.layout != .dense_row_major or input_descriptor.element_type != output_descriptor.element_type or tensor.denseElementCount(input_descriptor) != tensor.denseElementCount(output_descriptor)) return null;
            },
            .broadcast_in_dim => {
                if (input_descriptor.layout != .dense_row_major or input_descriptor.element_type != output_descriptor.element_type) return null;
                if (tensor.denseElementCount(input_descriptor) != 1 and !tensor.sameDims(input_descriptor.dims, output_descriptor.dims)) return null;
            },
            .copy_arg0, .reduce_precision, .convert, .bitcast_convert => {
                if (input_descriptor.layout != .dense_row_major or input_descriptor.element_type != output_descriptor.element_type or !tensor.sameDims(input_descriptor.dims, output_descriptor.dims)) return null;
            },
            else => unreachable,
        }
        const input = storage.ExpressionStore.get(request.*, input_id) orelse return null;
        try storage.ExpressionStore.put(request, allocator, output_id, try allocator.dupe(u8, input));
        return {};
    }
};

const TupleElementExpression = struct {
    fn record(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, request: *storage.Request, instruction: ir.PlanInstruction) program_mod.Error!?void {
        if (instruction.inputs.len != 1 or instruction.outputs.len != 1) return null;
        const tuple_id = instruction.inputs[0];
        const output_id = instruction.outputs[0];
        if (tuple_id.index >= plan.values.len) return null;
        if (!storage.ExpressionStore.outputCompatible(request.*, plan, output_id)) return null;
        const tuple_value = plan.values[tuple_id.index];
        if (tuple_value.storage != .tuple) return null;
        const tuple_index = instruction.tuple_index orelse return null;
        if (tuple_index < 0 or tuple_index >= @as(i64, @intCast(tuple_value.elements.len))) return null;
        const element = storage.ExpressionStore.get(request.*, tuple_value.elements[@intCast(tuple_index)]) orelse return null;
        try storage.ExpressionStore.put(request, allocator, output_id, try allocator.dupe(u8, element));
        return {};
    }
};
