const std = @import("std");
const ir = @import("src/compiler/ir");

const buffer_mod = @import("buffer.zig");
const lowering_mod = @import("lowering.zig");
const types = @import("execution_types.zig");
const values_mod = @import("execution_values.zig");

const BufferHandle = types.BufferHandle;
const Error = types.Error;
const ValueBindings = values_mod.ValueBindings;

/// Executes shape, indexing, and ordering node forms.
pub const Context = struct {
    allocator: std.mem.Allocator,
    plan: *const ir.ExecutablePlan,
    values: *ValueBindings,

    /// Stores multi-output indexing results into the active value table.
    pub fn runStored(self: Context, instruction: ir.PlanInstruction) Error!?void {
        if (instruction.kind == .sort and instruction.inputs.len == 2 and instruction.outputs.len == 2) {
            return self.sortKeyValue(instruction);
        }
        if (instruction.kind == .top_k and instruction.inputs.len == 1 and instruction.outputs.len == 2) {
            return self.topK(instruction);
        }
        return null;
    }

    /// Returns the device buffer produced by one single-output indexing instruction.
    pub fn run(self: Context, instruction: ir.PlanInstruction, output_dims: []const i64) Error!?BufferHandle {
        return switch (instruction.kind) {
            .reshape => blk: {
                const input = try self.handle(instruction.inputs[0]);
                break :blk (try buffer_mod.Opaque.reshape(input, output_dims)) orelse return null;
            },
            .transpose => blk: {
                const input = try self.handle(instruction.inputs[0]);
                break :blk (try buffer_mod.Opaque.transpose(input, instruction.permutation orelse return null)) orelse return null;
            },
            .broadcast_in_dim => blk: {
                const input = try self.handle(instruction.inputs[0]);
                break :blk (try buffer_mod.Opaque.broadcastInDim(input, instruction.broadcast_dimensions orelse return null, output_dims)) orelse return null;
            },
            .slice => blk: {
                const input = try self.handle(instruction.inputs[0]);
                break :blk (try buffer_mod.Opaque.slice(
                    input,
                    instruction.start_indices orelse return null,
                    instruction.limit_indices orelse return null,
                    instruction.strides orelse return null,
                    output_dims,
                )) orelse return null;
            },
            .dynamic_slice => blk: {
                const input = try self.handle(instruction.inputs[0]);
                const starts = try values_mod.startHandles(self.allocator, self.values.handles, instruction.inputs[1..]);
                defer self.allocator.free(starts);
                break :blk (try buffer_mod.Opaque.dynamicSlice(input, starts, instruction.slice_sizes orelse return null, output_dims)) orelse return null;
            },
            .dynamic_update_slice => blk: {
                const input = try self.handle(instruction.inputs[0]);
                const update = try self.handle(instruction.inputs[1]);
                const starts = try values_mod.startHandles(self.allocator, self.values.handles, instruction.inputs[2..]);
                defer self.allocator.free(starts);
                break :blk (try buffer_mod.Opaque.dynamicUpdateSlice(input, update, starts, output_dims)) orelse return null;
            },
            .pad => blk: {
                const input = try self.handle(instruction.inputs[0]);
                const padding_value = try self.handle(instruction.inputs[1]);
                break :blk (try buffer_mod.Opaque.pad(
                    input,
                    padding_value,
                    instruction.edge_padding_low orelse return null,
                    instruction.edge_padding_high orelse return null,
                    instruction.interior_padding orelse return null,
                    output_dims,
                )) orelse return null;
            },
            .reverse => blk: {
                const input = try self.handle(instruction.inputs[0]);
                break :blk (try buffer_mod.Opaque.reverse(input, instruction.dimensions orelse &.{}, output_dims)) orelse return null;
            },
            .concatenate => blk: {
                const lhs = try self.handle(instruction.inputs[0]);
                const rhs = try self.handle(instruction.inputs[1]);
                break :blk (try buffer_mod.Opaque.concatenate(lhs, rhs, instruction.dimension orelse return null, output_dims)) orelse return null;
            },
            .gather => blk: {
                const operand = try self.handle(instruction.inputs[0]);
                const indices = try self.handle(instruction.inputs[1]);
                break :blk (try buffer_mod.Opaque.gather(
                    operand,
                    indices,
                    instruction.start_index_map orelse return null,
                    instruction.collapsed_slice_dims orelse &.{},
                    instruction.operand_batching_dims orelse &.{},
                    instruction.start_indices_batching_dims orelse &.{},
                    instruction.index_vector_dim orelse 0,
                    instruction.slice_sizes orelse return null,
                    instruction.offset_dims orelse &.{},
                    output_dims,
                )) orelse return null;
            },
            .scatter => self.scatter(instruction, output_dims),
            .sort => blk: {
                const input = try self.handle(instruction.inputs[0]);
                const dimension = instruction.dimension orelse return null;
                const sorted = (try buffer_mod.Opaque.sort(input, dimension, output_dims)) orelse return null;
                break :blk (try values_mod.reverseIfDescending(sorted, dimension, output_dims, instruction.compare_direction orelse .lt)) orelse return null;
            },
            else => null,
        };
    }

    fn sortKeyValue(self: Context, instruction: ir.PlanInstruction) Error!?void {
        const dimension = instruction.dimension orelse return null;
        const direction = instruction.compare_direction orelse .lt;
        const keys = try self.handle(instruction.inputs[0]);
        const values = try self.handle(instruction.inputs[1]);
        const key_output_id = instruction.outputs[0];
        const value_output_id = instruction.outputs[1];
        if (key_output_id.index >= self.plan.values.len or value_output_id.index >= self.plan.values.len) return error.CommandSubmissionFailed;
        const key_descriptor = self.plan.values[key_output_id.index].descriptor;
        const value_descriptor = self.plan.values[value_output_id.index].descriptor;
        const key_dims = instruction.dims orelse key_descriptor.dims;
        const value_dims = value_descriptor.dims;
        const sorted_keys = (try buffer_mod.Opaque.sort(keys, dimension, key_dims)) orelse return null;
        const directed_keys = (try values_mod.reverseIfDescending(sorted_keys, dimension, key_dims, direction)) orelse return null;
        errdefer buffer_mod.Opaque.destroy(directed_keys);
        const order = (try buffer_mod.Opaque.argsort(keys, dimension, value_descriptor.element_type, value_dims)) orelse return null;
        const directed_order = (try values_mod.reverseIfDescending(order, dimension, value_dims, direction)) orelse return null;
        errdefer buffer_mod.Opaque.destroy(directed_order);
        const sorted_values = (try buffer_mod.Opaque.takeAlongAxis(values, directed_order, dimension, value_dims)) orelse return null;

        try values_mod.storeOwnedValueHandle(self.values.handles, self.values.owned, key_output_id, directed_keys);
        errdefer self.values.owned[key_output_id.index] = false;
        try values_mod.storeOwnedValueHandle(self.values.handles, self.values.owned, value_output_id, sorted_values);
        buffer_mod.Opaque.destroy(directed_order);
        return {};
    }

    fn topK(self: Context, instruction: ir.PlanInstruction) Error!?void {
        const input_id = instruction.inputs[0];
        const input = try self.handle(input_id);
        const input_descriptor = self.plan.values[input_id.index].descriptor;
        if (input_descriptor.dims.len == 0) return null;
        const axis: i64 = @intCast(input_descriptor.dims.len - 1);
        const k = instruction.top_k_k orelse return null;
        const values_id = instruction.outputs[0];
        const indices_id = instruction.outputs[1];
        if (values_id.index >= self.plan.values.len or indices_id.index >= self.plan.values.len) return error.CommandSubmissionFailed;
        const values_descriptor = self.plan.values[values_id.index].descriptor;
        const indices_descriptor = self.plan.values[indices_id.index].descriptor;

        const starts = try self.allocator.alloc(i64, input_descriptor.dims.len);
        defer self.allocator.free(starts);
        const limits = try self.allocator.dupe(i64, input_descriptor.dims);
        defer self.allocator.free(limits);
        const strides = try self.allocator.alloc(i64, input_descriptor.dims.len);
        defer self.allocator.free(strides);
        @memset(starts, 0);
        @memset(strides, 1);
        limits[limits.len - 1] = k;

        const sorted_values = (try buffer_mod.Opaque.sort(input, axis, input_descriptor.dims)) orelse return null;
        const descending_values = (try values_mod.reverseIfDescending(sorted_values, axis, input_descriptor.dims, .gt)) orelse return null;
        errdefer buffer_mod.Opaque.destroy(descending_values);
        const top_values = (try buffer_mod.Opaque.slice(descending_values, starts, limits, strides, values_descriptor.dims)) orelse return null;
        buffer_mod.Opaque.destroy(descending_values);

        const sorted_indices = (try buffer_mod.Opaque.argsort(input, axis, indices_descriptor.element_type, input_descriptor.dims)) orelse return null;
        const descending_indices = (try values_mod.reverseIfDescending(sorted_indices, axis, input_descriptor.dims, .gt)) orelse return null;
        errdefer buffer_mod.Opaque.destroy(descending_indices);
        const top_indices = (try buffer_mod.Opaque.slice(descending_indices, starts, limits, strides, indices_descriptor.dims)) orelse return null;
        buffer_mod.Opaque.destroy(descending_indices);

        try values_mod.storeOwnedValueHandle(self.values.handles, self.values.owned, values_id, top_values);
        errdefer self.values.owned[values_id.index] = false;
        try values_mod.storeOwnedValueHandle(self.values.handles, self.values.owned, indices_id, top_indices);
        return {};
    }

    fn scatter(self: Context, instruction: ir.PlanInstruction, output_dims: []const i64) Error!?BufferHandle {
        const operand = try self.handle(instruction.inputs[0]);
        const indices = try self.handle(instruction.inputs[1]);
        const updates = try self.handle(instruction.inputs[2]);
        if (lowering_mod.supportedScatterAxis(instruction)) |scatter_axis| {
            return (try buffer_mod.Opaque.scatterAxis(
                operand,
                indices,
                updates,
                scatter_axis,
                instruction.index_vector_dim orelse 0,
                instruction.scatter_update_kind orelse .set,
                output_dims,
            )) orelse null;
        }
        return (try buffer_mod.Opaque.scatter(
            operand,
            indices,
            updates,
            instruction.scatter_dims_to_operand_dims orelse return null,
            instruction.inserted_window_dims orelse return null,
            instruction.update_window_dims orelse &.{},
            instruction.input_batching_dims orelse &.{},
            instruction.scatter_indices_batching_dims orelse &.{},
            instruction.index_vector_dim orelse 0,
            instruction.scatter_update_kind orelse .set,
            output_dims,
        )) orelse null;
    }

    fn handle(self: Context, value_id: ir.ValueId) Error!BufferHandle {
        if (value_id.index >= self.values.handles.len) return error.CommandSubmissionFailed;
        return self.values.handles[value_id.index] orelse error.CommandSubmissionFailed;
    }
};
