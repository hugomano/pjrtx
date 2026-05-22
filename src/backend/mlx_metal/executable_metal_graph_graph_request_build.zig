const std = @import("std");

const ir = @import("src/compiler/ir");
const metalcpp_call = @import("metalcpp_call.zig");
const program_mod = @import("program.zig");
const recorder = @import("executable_metal_graph_graph_request_record.zig");
const storage = @import("executable_metal_graph_graph_request_storage.zig");
const tensor = @import("executable_metal_graph_tensor.zig");

/// Builds storage for one legacy single-kernel graph request.
pub const GraphRequestBuilder = struct {
    /// Creates request storage when every instruction can be expressed in one map kernel.
    pub fn init(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan) program_mod.Error!?storage.Request {
        const output = OutputShape.fromPlan(plan) orelse return null;
        var allocation = try RequestAllocation.init(allocator, plan, output);
        errdefer allocation.deinit(allocator);

        (try ParameterInputs.record(allocator, plan, output, &allocation)) orelse return null;
        (try ConstantInputs.record(allocator, plan, output, &allocation)) orelse return null;
        OutputSpecs.record(plan, &allocation) orelse return null;

        var request = allocation.finish(plan, output);
        errdefer request.deinit(allocator);
        for (plan.instructions) |instruction| {
            (try recorder.GraphExpressionRecorder.run(allocator, plan, &request, instruction)) orelse return null;
        }
        for (plan.output_ids) |output_id| {
            if (storage.ExpressionStore.get(request, output_id) == null) return null;
        }
        return request;
    }
};

const OutputShape = struct {
    descriptor: ir.BufferDescriptor,
    element_count: usize,

    fn fromPlan(plan: *const ir.ExecutablePlan) ?OutputShape {
        if (plan.output_ids.len == 0) return null;
        const output_id = plan.output_ids[0];
        if (output_id.index >= plan.values.len) return null;
        const descriptor = plan.values[output_id.index].descriptor;
        if (!tensor.supportedElementType(descriptor.element_type) or descriptor.layout != .dense_row_major) return null;
        const element_count = tensor.denseElementCount(descriptor);
        if (element_count == 0 or element_count > std.math.maxInt(u32)) return null;
        for (plan.output_ids) |candidate_id| {
            if (candidate_id.index >= plan.values.len) return null;
            if (!tensor.sameTensor(plan.values[candidate_id.index].descriptor, descriptor.element_type, element_count)) return null;
        }
        return .{ .descriptor = descriptor, .element_count = element_count };
    }
};

const RequestAllocation = struct {
    input_specs: []metalcpp_call.TensorSpec,
    output_specs: []metalcpp_call.TensorSpec,
    constant_instruction_indices: []usize,
    expressions: []?[]const u8,

    fn init(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, output: OutputShape) !RequestAllocation {
        const parameter_count = plan.parameter_shardings.len;
        const constant_count = tensor.countConstants(plan);
        const input_specs = try allocator.alloc(metalcpp_call.TensorSpec, parameter_count + constant_count);
        errdefer allocator.free(input_specs);
        const constant_indices = try allocator.alloc(usize, constant_count);
        errdefer allocator.free(constant_indices);
        const output_specs = try allocator.alloc(metalcpp_call.TensorSpec, plan.output_ids.len);
        errdefer allocator.free(output_specs);
        const expressions = try allocator.alloc(?[]const u8, plan.values.len);
        errdefer allocator.free(expressions);
        _ = output;
        @memset(expressions, null);
        return .{ .input_specs = input_specs, .output_specs = output_specs, .constant_instruction_indices = constant_indices, .expressions = expressions };
    }

    fn deinit(self: *RequestAllocation, allocator: std.mem.Allocator) void {
        storage.ExpressionStore.freeAll(allocator, self.expressions);
        allocator.free(self.expressions);
        allocator.free(self.output_specs);
        allocator.free(self.constant_instruction_indices);
        allocator.free(self.input_specs);
        self.* = undefined;
    }

    fn finish(self: *RequestAllocation, plan: *const ir.ExecutablePlan, output: OutputShape) storage.Request {
        const request = storage.Request{
            .input_specs = self.input_specs,
            .output_specs = self.output_specs,
            .output_ids = plan.output_ids,
            .constant_instruction_indices = self.constant_instruction_indices,
            .output_type = output.descriptor.element_type,
            .element_count = @intCast(output.element_count),
            .expressions = self.expressions,
        };
        self.* = undefined;
        return request;
    }
};

const ParameterInputs = struct {
    fn record(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, output: OutputShape, allocation: *RequestAllocation) program_mod.Error!?void {
        var parameter_index: usize = 0;
        for (plan.values) |value| {
            if (value.role != .parameter) continue;
            if (parameter_index >= plan.parameter_shardings.len or value.id.index >= allocation.expressions.len) return error.CommandSubmissionFailed;
            if (!tensor.broadcastCompatible(value.descriptor, output.descriptor.element_type, output.element_count)) return null;
            allocation.input_specs[parameter_index] = tensor.tensorSpec(value.descriptor) orelse return null;
            allocation.expressions[value.id.index] = try tensor.inputExpression(allocator, parameter_index, value.descriptor);
            parameter_index += 1;
        }
        if (parameter_index != plan.parameter_shardings.len) return error.CommandSubmissionFailed;
        return {};
    }
};

const ConstantInputs = struct {
    fn record(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, output: OutputShape, allocation: *RequestAllocation) program_mod.Error!?void {
        var constant_index: usize = 0;
        const parameter_count = plan.parameter_shardings.len;
        for (plan.instructions, 0..) |instruction, instruction_index| {
            if (instruction.kind != .constant) continue;
            if (instruction.outputs.len != 1) return null;
            const output_id = instruction.outputs[0];
            if (output_id.index >= plan.values.len) return null;
            const descriptor = plan.values[output_id.index].descriptor;
            if (!tensor.broadcastCompatible(descriptor, output.descriptor.element_type, output.element_count)) return null;
            allocation.input_specs[parameter_count + constant_index] = tensor.tensorSpec(descriptor) orelse return null;
            allocation.constant_instruction_indices[constant_index] = instruction_index;
            allocation.expressions[output_id.index] = try tensor.inputExpression(allocator, parameter_count + constant_index, descriptor);
            constant_index += 1;
        }
        return {};
    }
};

const OutputSpecs = struct {
    fn record(plan: *const ir.ExecutablePlan, allocation: *RequestAllocation) ?void {
        for (plan.output_ids, 0..) |output_id, output_index| {
            allocation.output_specs[output_index] = tensor.tensorSpec(plan.values[output_id.index].descriptor) orelse return null;
        }
    }
};
