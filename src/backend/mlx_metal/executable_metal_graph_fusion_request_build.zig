const std = @import("std");

const ir = @import("src/compiler/ir");
const fusion_expr = @import("executable_metal_graph_fusion_expression.zig");
const metalcpp_call = @import("metalcpp_call.zig");
const program_mod = @import("program.zig");
const recorder = @import("executable_metal_graph_fusion_request_record.zig");
const storage = @import("executable_metal_graph_fusion_request_storage.zig");
const tensor = @import("executable_metal_graph_tensor.zig");

/// Builds storage for one view/elementwise fusion request.
pub const FusionRequestBuilder = struct {
    /// Creates request storage when the fusion group can be represented as one map kernel.
    pub fn init(
        allocator: std.mem.Allocator,
        plan: *const ir.ExecutablePlan,
        program: *const program_mod.Program,
        group: program_mod.FusionGroup,
    ) program_mod.Error!?storage.Request {
        const shape = FusionShape.fromPlan(plan, group) orelse return null;
        var allocation = try RequestAllocation.init(allocator, plan, group);
        errdefer allocation.deinit(allocator);

        OutputSpecs.record(plan, group, shape, &allocation) orelse return null;
        (try InputSpecs.record(allocator, plan, group, &allocation)) orelse return null;
        var request = allocation.finish(group, shape);
        errdefer request.deinit(allocator);
        for (group.node_indices) |node_index| {
            if (node_index >= program.nodes.len) return error.CommandSubmissionFailed;
            const node = program.nodes[node_index];
            if (node.fusion_group != group.id or node.subprograms.len != 0 or node.control_flow != null) return error.CommandSubmissionFailed;
            const instruction = plan.instructions[node.instruction_index];
            (try recorder.FusionExpressionRecorder.run(allocator, plan, &request, instruction)) orelse return null;
        }
        for (group.output_values) |output_id| {
            if (storage.ExpressionStore.get(request, output_id) == null) return null;
        }
        return request;
    }
};

const FusionShape = struct {
    element_count: usize,

    fn fromPlan(plan: *const ir.ExecutablePlan, group: program_mod.FusionGroup) ?FusionShape {
        if (group.kind != .view_elementwise or group.output_values.len == 0 or group.node_indices.len == 0) return null;
        const first_output = tensor.valueDescriptor(plan, group.output_values[0]) orelse return null;
        if (!tensor.supportedProgramElementType(first_output.element_type)) return null;
        const element_count = fusion_expr.groupElementCount(plan, group) orelse return null;
        if (element_count == 0 or element_count > std.math.maxInt(u32)) return null;
        return .{ .element_count = element_count };
    }
};

const RequestAllocation = struct {
    input_specs: []metalcpp_call.TensorSpec,
    output_specs: []metalcpp_call.TensorSpec,
    output_counts: []usize,
    expressions: []?fusion_expr.Expression,

    fn init(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, group: program_mod.FusionGroup) !RequestAllocation {
        const input_specs = try allocator.alloc(metalcpp_call.TensorSpec, group.input_values.len);
        errdefer allocator.free(input_specs);
        const output_specs = try allocator.alloc(metalcpp_call.TensorSpec, group.output_values.len);
        errdefer allocator.free(output_specs);
        const output_counts = try allocator.alloc(usize, group.output_values.len);
        errdefer allocator.free(output_counts);
        const expressions = try allocator.alloc(?fusion_expr.Expression, plan.values.len);
        errdefer allocator.free(expressions);
        @memset(expressions, null);
        return .{ .input_specs = input_specs, .output_specs = output_specs, .output_counts = output_counts, .expressions = expressions };
    }

    fn deinit(self: *RequestAllocation, allocator: std.mem.Allocator) void {
        storage.ExpressionStore.freeAll(allocator, self.expressions);
        allocator.free(self.expressions);
        allocator.free(self.output_counts);
        allocator.free(self.output_specs);
        allocator.free(self.input_specs);
        self.* = undefined;
    }

    fn finish(self: *RequestAllocation, group: program_mod.FusionGroup, shape: FusionShape) storage.Request {
        const request = storage.Request{
            .input_specs = self.input_specs,
            .output_specs = self.output_specs,
            .output_counts = self.output_counts,
            .input_values = group.input_values,
            .output_values = group.output_values,
            .element_count = @intCast(shape.element_count),
            .expressions = self.expressions,
            .local_statements = .empty,
        };
        self.* = undefined;
        return request;
    }
};

const OutputSpecs = struct {
    fn record(plan: *const ir.ExecutablePlan, group: program_mod.FusionGroup, shape: FusionShape, allocation: *RequestAllocation) ?void {
        for (group.output_values, 0..) |output_id, output_index| {
            const output = tensor.valueDescriptor(plan, output_id) orelse return null;
            const output_count = tensor.denseElementCount(output);
            if (!tensor.supportedProgramElementType(output.element_type) or !fusion_expr.elementCountCompatible(output_count, shape.element_count)) return null;
            allocation.output_specs[output_index] = tensor.tensorSpec(output) orelse return null;
            allocation.output_counts[output_index] = output_count;
        }
    }
};

const InputSpecs = struct {
    fn record(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, group: program_mod.FusionGroup, allocation: *RequestAllocation) program_mod.Error!?void {
        for (group.input_values, 0..) |input_id, input_index| {
            const input = tensor.valueDescriptor(plan, input_id) orelse return null;
            if (!tensor.supportedProgramElementType(input.element_type)) return null;
            const input_count = tensor.denseElementCount(input);
            allocation.input_specs[input_index] = tensor.tensorSpec(input) orelse return null;
            allocation.expressions[input_id.index] = .{
                .descriptor = input,
                .source = if (input_count == 1)
                    try std.fmt.allocPrint(allocator, "in{d}[0]", .{input_index})
                else
                    try std.fmt.allocPrint(allocator, "in{d}[{s}]", .{ input_index, fusion_expr.Expression.IndexToken }),
            };
        }
        return {};
    }
};
