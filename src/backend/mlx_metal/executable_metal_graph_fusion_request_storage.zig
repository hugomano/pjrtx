const std = @import("std");

const ir = @import("src/compiler/ir");
const fusion_expr = @import("executable_metal_graph_fusion_expression.zig");
const metalcpp_call = @import("metalcpp_call.zig");
const program_mod = @import("program.zig");
const tensor = @import("executable_metal_graph_tensor.zig");

/// Stores generated expressions, statements, and specs for one fusion kernel.
pub const Request = struct {
    input_specs: []metalcpp_call.TensorSpec,
    output_specs: []metalcpp_call.TensorSpec,
    output_counts: []const usize,
    input_values: []const ir.ValueId,
    output_values: []const ir.ValueId,
    element_count: u64,
    expressions: []?fusion_expr.Expression,
    local_statements: std.ArrayList([]const u8),

    /// Releases all heap-owned request storage.
    pub fn deinit(self: *Request, allocator: std.mem.Allocator) void {
        ExpressionStore.freeStatements(allocator, self.local_statements.items);
        self.local_statements.deinit(allocator);
        ExpressionStore.freeAll(allocator, self.expressions);
        allocator.free(self.expressions);
        allocator.free(self.output_counts);
        allocator.free(self.output_specs);
        allocator.free(self.input_specs);
        self.* = undefined;
    }
};

/// Owns fusion expression lookup, compatibility checks, and replacement.
pub const ExpressionStore = struct {
    /// Releases every generated local statement.
    pub fn freeStatements(allocator: std.mem.Allocator, statements: []const []const u8) void {
        for (statements) |statement| allocator.free(statement);
    }

    /// Releases every stored expression source.
    pub fn freeAll(allocator: std.mem.Allocator, expressions: []?fusion_expr.Expression) void {
        fusion_expr.freeExpressions(allocator, expressions);
    }

    /// Returns the expression for one value if it has already been recorded.
    pub fn get(request: Request, value_id: ir.ValueId) ?fusion_expr.Expression {
        if (value_id.index >= request.expressions.len) return null;
        return request.expressions[value_id.index];
    }

    /// Returns true when an output descriptor can be emitted by this fusion kernel.
    pub fn outputCompatible(request: Request, descriptor: ir.BufferDescriptor) bool {
        const element_count = tensor.denseElementCount(descriptor);
        return descriptor.layout == .dense_row_major and
            tensor.supportedProgramElementType(descriptor.element_type) and
            fusion_expr.elementCountCompatible(element_count, @intCast(request.element_count));
    }

    /// Stores a source expression, taking ownership of the source slice.
    pub fn putSource(request: *Request, allocator: std.mem.Allocator, value_id: ir.ValueId, descriptor: ir.BufferDescriptor, source: []const u8) !void {
        errdefer allocator.free(source);
        try put(request, allocator, value_id, .{ .descriptor = descriptor, .source = source });
    }

    /// Stores a computed source, hoisting full-sized expressions into local variables.
    pub fn putComputed(request: *Request, allocator: std.mem.Allocator, value_id: ir.ValueId, descriptor: ir.BufferDescriptor, source: []const u8) program_mod.Error!void {
        if (tensor.denseElementCount(descriptor) != @as(usize, @intCast(request.element_count))) {
            try putSource(request, allocator, value_id, descriptor, source);
            return;
        }
        try ComputedSourceStore.put(request, allocator, value_id, descriptor, source);
    }

    fn put(request: *Request, allocator: std.mem.Allocator, value_id: ir.ValueId, expression: fusion_expr.Expression) !void {
        if (value_id.index >= request.expressions.len) return error.CommandSubmissionFailed;
        if (request.expressions[value_id.index]) |old| allocator.free(old.source);
        request.expressions[value_id.index] = expression;
    }
};

const ComputedSourceStore = struct {
    fn put(request: *Request, allocator: std.mem.Allocator, value_id: ir.ValueId, descriptor: ir.BufferDescriptor, source: []const u8) program_mod.Error!void {
        const scalar_type = tensor.metalProgramScalarType(descriptor.element_type) orelse {
            allocator.free(source);
            return error.CommandSubmissionFailed;
        };
        const local_name = std.fmt.allocPrint(allocator, "v{d}", .{value_id.index}) catch {
            allocator.free(source);
            return error.OutOfMemory;
        };
        const rendered_source = (fusion_expr.Expression{ .descriptor = descriptor, .source = source }).at(allocator, "elem") catch {
            allocator.free(local_name);
            allocator.free(source);
            return error.OutOfMemory;
        };
        defer allocator.free(rendered_source);
        const statement = std.fmt.allocPrint(allocator, "    const {s} {s} = {s}(({s}));\n", .{ scalar_type, local_name, scalar_type, rendered_source }) catch {
            allocator.free(local_name);
            allocator.free(source);
            return error.OutOfMemory;
        };
        request.local_statements.append(allocator, statement) catch {
            allocator.free(statement);
            allocator.free(local_name);
            allocator.free(source);
            return error.OutOfMemory;
        };
        allocator.free(source);
        ExpressionStore.put(request, allocator, value_id, .{ .descriptor = descriptor, .source = local_name }) catch |err| {
            allocator.free(local_name);
            return err;
        };
    }
};
