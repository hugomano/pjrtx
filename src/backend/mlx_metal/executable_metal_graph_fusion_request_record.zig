const std = @import("std");

const ir = @import("src/compiler/ir");
const fusion_expr = @import("executable_metal_graph_fusion_expression.zig");
const map_rules = @import("executable_metal_graph_map_rules.zig");
const program_mod = @import("program.zig");
const storage = @import("executable_metal_graph_fusion_request_storage.zig");
const tensor = @import("executable_metal_graph_tensor.zig");

/// Routes one fusion instruction into the fusion expression store.
pub const FusionExpressionRecorder = struct {
    /// Records one supported instruction or returns null when this fusion path cannot lower it.
    pub fn run(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, request: *storage.Request, instruction: ir.PlanInstruction) program_mod.Error!?void {
        if (map_rules.kindFor(instruction.kind)) |map_kind| {
            return switch (map_kind) {
                .binary => BinaryExpression.record(allocator, plan, request, instruction),
                .unary => UnaryExpression.record(allocator, plan, request, instruction),
                .compare => PredicateExpression.recordCompare(allocator, plan, request, instruction),
                .select => PredicateExpression.recordSelect(allocator, plan, request, instruction),
                .clamp => PredicateExpression.recordClamp(allocator, plan, request, instruction),
            };
        }
        return switch (instruction.kind) {
            .reshape, .slice, .broadcast_in_dim, .copy_arg0, .reduce_precision, .convert, .bitcast_convert => ViewExpression.record(allocator, plan, request, instruction),
            else => null,
        };
    }
};

const InputExpression = struct {
    fn matching(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, request: *const storage.Request, value_id: ir.ValueId, output: ir.BufferDescriptor) program_mod.Error!?[]const u8 {
        const descriptor = tensor.valueDescriptor(plan, value_id) orelse return null;
        if (descriptor.element_type != output.element_type) return null;
        const element_count = tensor.denseElementCount(descriptor);
        const output_count = tensor.denseElementCount(output);
        if (element_count == output_count) return indexed(allocator, request, value_id, fusion_expr.Expression.IndexToken);
        if (element_count == 1) return indexed(allocator, request, value_id, "0");
        return null;
    }

    fn indexed(allocator: std.mem.Allocator, request: *const storage.Request, value_id: ir.ValueId, index_expr: []const u8) program_mod.Error!?[]const u8 {
        const expression = storage.ExpressionStore.get(request.*, value_id) orelse return null;
        return expression.at(allocator, index_expr) catch return error.OutOfMemory;
    }
};

const BinaryExpression = struct {
    fn record(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, request: *storage.Request, instruction: ir.PlanInstruction) program_mod.Error!?void {
        if (instruction.inputs.len != 2 or instruction.outputs.len != 1) return null;
        const output_id = instruction.outputs[0];
        const output = tensor.valueDescriptor(plan, output_id) orelse return null;
        if (!storage.ExpressionStore.outputCompatible(request.*, output)) return null;
        const lhs = (try InputExpression.matching(allocator, plan, request, instruction.inputs[0], output)) orelse return null;
        defer allocator.free(lhs);
        const rhs = (try InputExpression.matching(allocator, plan, request, instruction.inputs[1], output)) orelse return null;
        defer allocator.free(rhs);
        const expression = (try map_rules.binaryExpression(allocator, instruction.kind, lhs, rhs)) orelse return null;
        try storage.ExpressionStore.putComputed(request, allocator, output_id, output, expression);
        return {};
    }
};

const UnaryExpression = struct {
    fn record(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, request: *storage.Request, instruction: ir.PlanInstruction) program_mod.Error!?void {
        if (instruction.inputs.len != 1 or instruction.outputs.len != 1) return null;
        const output_id = instruction.outputs[0];
        const output = tensor.valueDescriptor(plan, output_id) orelse return null;
        if (!storage.ExpressionStore.outputCompatible(request.*, output)) return null;
        const input = (try InputExpression.matching(allocator, plan, request, instruction.inputs[0], output)) orelse return null;
        defer allocator.free(input);
        const scalar_type = tensor.metalProgramScalarType(output.element_type) orelse return null;
        const expression = (try map_rules.unaryExpression(allocator, instruction.kind, input, scalar_type, .typed)) orelse return null;
        try storage.ExpressionStore.putComputed(request, allocator, output_id, output, expression);
        return {};
    }
};

const PredicateExpression = struct {
    fn recordCompare(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, request: *storage.Request, instruction: ir.PlanInstruction) program_mod.Error!?void {
        if (instruction.inputs.len != 2 or instruction.outputs.len != 1) return null;
        const output_id = instruction.outputs[0];
        const output = tensor.valueDescriptor(plan, output_id) orelse return null;
        if (output.element_type != .pred or !storage.ExpressionStore.outputCompatible(request.*, output)) return null;
        const lhs_descriptor = tensor.valueDescriptor(plan, instruction.inputs[0]) orelse return null;
        const rhs_descriptor = tensor.valueDescriptor(plan, instruction.inputs[1]) orelse return null;
        if (lhs_descriptor.element_type != rhs_descriptor.element_type) return null;
        const output_count = tensor.denseElementCount(output);
        const lhs_count = tensor.denseElementCount(lhs_descriptor);
        const rhs_count = tensor.denseElementCount(rhs_descriptor);
        if ((lhs_count != output_count and lhs_count != 1) or (rhs_count != output_count and rhs_count != 1)) return null;
        const lhs = (try InputExpression.indexed(allocator, request, instruction.inputs[0], if (lhs_count == 1) "0" else fusion_expr.Expression.IndexToken)) orelse return null;
        defer allocator.free(lhs);
        const rhs = (try InputExpression.indexed(allocator, request, instruction.inputs[1], if (rhs_count == 1) "0" else fusion_expr.Expression.IndexToken)) orelse return null;
        defer allocator.free(rhs);
        const expression = try map_rules.compareExpression(allocator, lhs, rhs, instruction.compare_direction orelse .eq);
        try storage.ExpressionStore.putComputed(request, allocator, output_id, output, expression);
        return {};
    }

    fn recordSelect(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, request: *storage.Request, instruction: ir.PlanInstruction) program_mod.Error!?void {
        if (instruction.inputs.len != 3 or instruction.outputs.len != 1) return null;
        const output_id = instruction.outputs[0];
        const output = tensor.valueDescriptor(plan, output_id) orelse return null;
        if (!storage.ExpressionStore.outputCompatible(request.*, output)) return null;
        const pred = try predicateInput(allocator, plan, request, instruction.inputs[0], output);
        defer allocator.free(pred);
        const on_true = (try InputExpression.matching(allocator, plan, request, instruction.inputs[1], output)) orelse return null;
        defer allocator.free(on_true);
        const on_false = (try InputExpression.matching(allocator, plan, request, instruction.inputs[2], output)) orelse return null;
        defer allocator.free(on_false);
        try storage.ExpressionStore.putComputed(request, allocator, output_id, output, try map_rules.selectExpression(allocator, pred, on_true, on_false));
        return {};
    }

    fn recordClamp(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, request: *storage.Request, instruction: ir.PlanInstruction) program_mod.Error!?void {
        if (instruction.inputs.len != 3 or instruction.outputs.len != 1) return null;
        const output_id = instruction.outputs[0];
        const output = tensor.valueDescriptor(plan, output_id) orelse return null;
        if (!storage.ExpressionStore.outputCompatible(request.*, output)) return null;
        const min_value = (try InputExpression.matching(allocator, plan, request, instruction.inputs[0], output)) orelse return null;
        defer allocator.free(min_value);
        const value = (try InputExpression.matching(allocator, plan, request, instruction.inputs[1], output)) orelse return null;
        defer allocator.free(value);
        const max_value = (try InputExpression.matching(allocator, plan, request, instruction.inputs[2], output)) orelse return null;
        defer allocator.free(max_value);
        try storage.ExpressionStore.putComputed(request, allocator, output_id, output, try map_rules.clampExpression(allocator, min_value, value, max_value));
        return {};
    }

    fn predicateInput(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, request: *const storage.Request, value_id: ir.ValueId, output: ir.BufferDescriptor) program_mod.Error![]const u8 {
        const descriptor = tensor.valueDescriptor(plan, value_id) orelse return error.CommandSubmissionFailed;
        const output_count = tensor.denseElementCount(output);
        const pred_count = tensor.denseElementCount(descriptor);
        if (descriptor.element_type != .pred or (pred_count != output_count and pred_count != 1)) return error.CommandSubmissionFailed;
        return (try InputExpression.indexed(allocator, request, value_id, if (pred_count == 1) "0" else fusion_expr.Expression.IndexToken)) orelse error.CommandSubmissionFailed;
    }
};

const ViewExpression = struct {
    fn record(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, request: *storage.Request, instruction: ir.PlanInstruction) program_mod.Error!?void {
        if (instruction.inputs.len != 1 or instruction.outputs.len != 1) return null;
        const input = tensor.valueDescriptor(plan, instruction.inputs[0]) orelse return null;
        const output_id = instruction.outputs[0];
        const output = tensor.valueDescriptor(plan, output_id) orelse return null;
        if (!storage.ExpressionStore.outputCompatible(request.*, output)) return null;
        const input_expr = storage.ExpressionStore.get(request.*, instruction.inputs[0]) orelse return null;
        return switch (instruction.kind) {
            .reshape => reshape(allocator, request, output_id, input, output, input_expr),
            .slice => slice(allocator, request, output_id, input, output, input_expr, instruction),
            .broadcast_in_dim => broadcast(allocator, request, output_id, input, output, input_expr, instruction),
            .copy_arg0, .reduce_precision, .bitcast_convert => sameShape(allocator, request, output_id, input, output, input_expr),
            .convert => convert(allocator, request, output_id, input, output, input_expr),
            else => unreachable,
        };
    }

    fn reshape(allocator: std.mem.Allocator, request: *storage.Request, output_id: ir.ValueId, input: ir.BufferDescriptor, output: ir.BufferDescriptor, input_expr: fusion_expr.Expression) program_mod.Error!?void {
        if (input.element_type != output.element_type or tensor.denseElementCount(input) != tensor.denseElementCount(output)) return null;
        try storage.ExpressionStore.putSource(request, allocator, output_id, output, input_expr.at(allocator, fusion_expr.Expression.IndexToken) catch return error.OutOfMemory);
        return {};
    }

    fn slice(allocator: std.mem.Allocator, request: *storage.Request, output_id: ir.ValueId, input: ir.BufferDescriptor, output: ir.BufferDescriptor, input_expr: fusion_expr.Expression, instruction: ir.PlanInstruction) program_mod.Error!?void {
        const starts = instruction.start_indices orelse return null;
        const limits = instruction.limit_indices orelse return null;
        const strides = instruction.strides orelse return null;
        if (input.element_type != output.element_type) return null;
        const input_index = (fusion_expr.sliceIndexExpression(allocator, input, output, starts, limits, strides, fusion_expr.Expression.IndexToken) catch return error.OutOfMemory) orelse return null;
        defer allocator.free(input_index);
        if (!input_expr.canRenderAt(input_index)) return null;
        try storage.ExpressionStore.putSource(request, allocator, output_id, output, input_expr.at(allocator, input_index) catch return error.OutOfMemory);
        return {};
    }

    fn sameShape(allocator: std.mem.Allocator, request: *storage.Request, output_id: ir.ValueId, input: ir.BufferDescriptor, output: ir.BufferDescriptor, input_expr: fusion_expr.Expression) program_mod.Error!?void {
        if (input.element_type != output.element_type or !tensor.sameDims(input.dims, output.dims)) return null;
        try storage.ExpressionStore.putSource(request, allocator, output_id, output, input_expr.at(allocator, fusion_expr.Expression.IndexToken) catch return error.OutOfMemory);
        return {};
    }
};

const ViewBroadcastConvert = struct {
    fn broadcast(allocator: std.mem.Allocator, request: *storage.Request, output_id: ir.ValueId, input: ir.BufferDescriptor, output: ir.BufferDescriptor, input_expr: fusion_expr.Expression, instruction: ir.PlanInstruction) program_mod.Error!?void {
        const input_count = tensor.denseElementCount(input);
        const output_count = tensor.denseElementCount(output);
        if (input.element_type != output.element_type) return null;
        if (input_count == 1) return storeAt(allocator, request, output_id, output, input_expr, "0");
        if (input_count == output_count or tensor.sameDims(input.dims, output.dims)) {
            return storeAt(allocator, request, output_id, output, input_expr, fusion_expr.Expression.IndexToken);
        }
        const broadcast_dimensions = instruction.broadcast_dimensions orelse return null;
        const input_index = (fusion_expr.broadcastIndexExpression(allocator, input, output, broadcast_dimensions, fusion_expr.Expression.IndexToken) catch return error.OutOfMemory) orelse return null;
        defer allocator.free(input_index);
        return storeAt(allocator, request, output_id, output, input_expr, input_index);
    }

    fn convert(allocator: std.mem.Allocator, request: *storage.Request, output_id: ir.ValueId, input: ir.BufferDescriptor, output: ir.BufferDescriptor, input_expr: fusion_expr.Expression) program_mod.Error!?void {
        if (tensor.denseElementCount(input) != tensor.denseElementCount(output)) return null;
        const output_type = tensor.metalProgramScalarType(output.element_type) orelse return null;
        const input_source = input_expr.at(allocator, fusion_expr.Expression.IndexToken) catch return error.OutOfMemory;
        defer allocator.free(input_source);
        try storage.ExpressionStore.putComputed(request, allocator, output_id, output, try std.fmt.allocPrint(allocator, "{s}(({s}))", .{ output_type, input_source }));
        return {};
    }

    fn storeAt(allocator: std.mem.Allocator, request: *storage.Request, output_id: ir.ValueId, output: ir.BufferDescriptor, input_expr: fusion_expr.Expression, index_expr: []const u8) program_mod.Error!?void {
        if (!input_expr.canRenderAt(index_expr)) return null;
        try storage.ExpressionStore.putSource(request, allocator, output_id, output, input_expr.at(allocator, index_expr) catch return error.OutOfMemory);
        return {};
    }
};

const broadcast = ViewBroadcastConvert.broadcast;
const convert = ViewBroadcastConvert.convert;
