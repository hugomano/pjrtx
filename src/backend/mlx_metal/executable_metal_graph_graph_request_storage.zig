const std = @import("std");

const ir = @import("src/compiler/ir");
const metalcpp_call = @import("metalcpp_call.zig");
const tensor = @import("executable_metal_graph_tensor.zig");

/// Stores generated expressions and tensor specs for one single-kernel graph request.
pub const Request = struct {
    input_specs: []metalcpp_call.TensorSpec,
    output_specs: []metalcpp_call.TensorSpec,
    output_ids: []const ir.ValueId,
    constant_instruction_indices: []usize,
    output_type: ir.BufferType,
    element_count: u64,
    expressions: []?[]const u8,

    /// Releases all heap-owned request storage.
    pub fn deinit(self: *Request, allocator: std.mem.Allocator) void {
        ExpressionStore.freeAll(allocator, self.expressions);
        allocator.free(self.expressions);
        allocator.free(self.output_specs);
        allocator.free(self.constant_instruction_indices);
        allocator.free(self.input_specs);
        self.* = undefined;
    }
};

/// Owns expression lookup, compatibility checks, and replacement.
pub const ExpressionStore = struct {
    /// Releases each stored expression.
    pub fn freeAll(allocator: std.mem.Allocator, expressions: []?[]const u8) void {
        for (expressions) |maybe_expression| {
            if (maybe_expression) |expression| allocator.free(expression);
        }
    }

    /// Returns the expression for one value if it has already been recorded.
    pub fn get(request: Request, value_id: ir.ValueId) ?[]const u8 {
        if (value_id.index >= request.expressions.len) return null;
        return request.expressions[value_id.index];
    }

    /// Replaces the expression for one value, taking ownership of the new string.
    pub fn put(request: *Request, allocator: std.mem.Allocator, value_id: ir.ValueId, expression: []const u8) !void {
        if (value_id.index >= request.expressions.len) return error.CommandSubmissionFailed;
        if (request.expressions[value_id.index]) |old| allocator.free(old);
        request.expressions[value_id.index] = expression;
    }

    /// Returns true when a value can be stored as a normal kernel output.
    pub fn outputCompatible(request: Request, plan: *const ir.ExecutablePlan, output_id: ir.ValueId) bool {
        if (output_id.index >= plan.values.len) return false;
        const descriptor = plan.values[output_id.index].descriptor;
        return tensor.sameTensor(descriptor, request.output_type, @intCast(request.element_count));
    }

    /// Returns true when a value can be read by the kernel expression for this request.
    pub fn inputCompatible(request: Request, plan: *const ir.ExecutablePlan, input_id: ir.ValueId) bool {
        if (input_id.index >= plan.values.len) return false;
        return tensor.broadcastCompatible(plan.values[input_id.index].descriptor, request.output_type, @intCast(request.element_count));
    }

    /// Returns true when a value is a dense predicate matching the kernel element count.
    pub fn predicateCompatible(request: Request, plan: *const ir.ExecutablePlan, value_id: ir.ValueId) bool {
        if (value_id.index >= plan.values.len) return false;
        const descriptor = plan.values[value_id.index].descriptor;
        return descriptor.layout == .dense_row_major and
            descriptor.element_type == .pred and
            tensor.denseElementCount(descriptor) == @as(usize, @intCast(request.element_count));
    }
};
