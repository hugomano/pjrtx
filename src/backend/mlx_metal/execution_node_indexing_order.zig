const std = @import("std");
const ir = @import("src/compiler/ir");

const buffer_mod = @import("buffer.zig");
const types = @import("execution_types.zig");
const values_mod = @import("execution_values.zig");

const Error = types.Error;
const ValueBindings = values_mod.ValueBindings;

/// Stores key/value sort and top-k multi-output indexing results.
pub fn sortKeyValue(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, values: *ValueBindings, instruction: ir.PlanInstruction) Error!?void {
    const dimension = instruction.dimension orelse return null;
    const direction = instruction.compare_direction orelse .lt;
    const keys = try handle(values, instruction.inputs[0]);
    const source_values = try handle(values, instruction.inputs[1]);
    const key_output_id = instruction.outputs[0];
    const value_output_id = instruction.outputs[1];
    if (key_output_id.index >= plan.values.len or value_output_id.index >= plan.values.len) return error.CommandSubmissionFailed;
    const key_dims = instruction.dims orelse plan.values[key_output_id.index].descriptor.dims;
    const value_descriptor = plan.values[value_output_id.index].descriptor;
    const sorted_keys = (try buffer_mod.Opaque.sort(keys, dimension, key_dims)) orelse return null;
    const directed_keys = (try values_mod.reverseIfDescending(sorted_keys, dimension, key_dims, direction)) orelse return null;
    errdefer buffer_mod.Opaque.destroy(directed_keys);
    const order = (try buffer_mod.Opaque.argsort(keys, dimension, value_descriptor.element_type, value_descriptor.dims)) orelse return null;
    const directed_order = (try values_mod.reverseIfDescending(order, dimension, value_descriptor.dims, direction)) orelse return null;
    errdefer buffer_mod.Opaque.destroy(directed_order);
    const sorted_values = (try buffer_mod.Opaque.takeAlongAxis(source_values, directed_order, dimension, value_descriptor.dims)) orelse return null;
    try values_mod.storeOwnedValueHandle(values.handles, values.owned, key_output_id, directed_keys);
    errdefer values.owned[key_output_id.index] = false;
    try values_mod.storeOwnedValueHandle(values.handles, values.owned, value_output_id, sorted_values);
    buffer_mod.Opaque.destroy(directed_order);
    _ = allocator;
    return {};
}

/// Stores top-k values and indices for the last-dimension top-k form.
pub fn topK(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, values: *ValueBindings, instruction: ir.PlanInstruction) Error!?void {
    const input_id = instruction.inputs[0];
    const input = try handle(values, input_id);
    const input_descriptor = plan.values[input_id.index].descriptor;
    if (input_descriptor.dims.len == 0) return null;
    const axis: i64 = @intCast(input_descriptor.dims.len - 1);
    const k = instruction.top_k_k orelse return null;
    const values_id = instruction.outputs[0];
    const indices_id = instruction.outputs[1];
    if (values_id.index >= plan.values.len or indices_id.index >= plan.values.len) return error.CommandSubmissionFailed;
    const slice = try TopKSlice.init(allocator, input_descriptor.dims, k);
    defer slice.deinit(allocator);
    const sorted_values = (try buffer_mod.Opaque.sort(input, axis, input_descriptor.dims)) orelse return null;
    const descending_values = (try values_mod.reverseIfDescending(sorted_values, axis, input_descriptor.dims, .gt)) orelse return null;
    errdefer buffer_mod.Opaque.destroy(descending_values);
    const top_values = (try buffer_mod.Opaque.slice(descending_values, slice.starts, slice.limits, slice.strides, plan.values[values_id.index].descriptor.dims)) orelse return null;
    buffer_mod.Opaque.destroy(descending_values);
    const sorted_indices = (try buffer_mod.Opaque.argsort(input, axis, plan.values[indices_id.index].descriptor.element_type, input_descriptor.dims)) orelse return null;
    const descending_indices = (try values_mod.reverseIfDescending(sorted_indices, axis, input_descriptor.dims, .gt)) orelse return null;
    errdefer buffer_mod.Opaque.destroy(descending_indices);
    const top_indices = (try buffer_mod.Opaque.slice(descending_indices, slice.starts, slice.limits, slice.strides, plan.values[indices_id.index].descriptor.dims)) orelse return null;
    buffer_mod.Opaque.destroy(descending_indices);
    try values_mod.storeOwnedValueHandle(values.handles, values.owned, values_id, top_values);
    errdefer values.owned[values_id.index] = false;
    try values_mod.storeOwnedValueHandle(values.handles, values.owned, indices_id, top_indices);
    return {};
}

const TopKSlice = struct {
    starts: []i64,
    limits: []i64,
    strides: []i64,

    fn init(allocator: std.mem.Allocator, dims: []const i64, k: i64) !TopKSlice {
        const starts = try allocator.alloc(i64, dims.len);
        errdefer allocator.free(starts);
        const limits = try allocator.dupe(i64, dims);
        errdefer allocator.free(limits);
        const strides = try allocator.alloc(i64, dims.len);
        @memset(starts, 0);
        @memset(strides, 1);
        limits[limits.len - 1] = k;
        return .{ .starts = starts, .limits = limits, .strides = strides };
    }

    fn deinit(self: TopKSlice, allocator: std.mem.Allocator) void {
        allocator.free(self.starts);
        allocator.free(self.limits);
        allocator.free(self.strides);
    }
};

fn handle(values: *ValueBindings, value_id: ir.ValueId) Error!types.BufferHandle {
    if (value_id.index >= values.handles.len) return error.CommandSubmissionFailed;
    return values.handles[value_id.index] orelse error.CommandSubmissionFailed;
}
