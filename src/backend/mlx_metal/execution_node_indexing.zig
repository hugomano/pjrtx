const std = @import("std");
const ir = @import("src/compiler/ir");

const buffer_mod = @import("buffer.zig");
const gather_mod = @import("execution_node_indexing_gather.zig");
const order_mod = @import("execution_node_indexing_order.zig");
const pad_mod = @import("execution_node_indexing_pad.zig");
const scatter_mod = @import("execution_node_indexing_scatter.zig");
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
            return order_mod.sortKeyValue(self.allocator, self.plan, self.values, instruction);
        }
        if (instruction.kind == .top_k and instruction.inputs.len == 1 and instruction.outputs.len == 2) {
            return order_mod.topK(self.allocator, self.plan, self.values, instruction);
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
            .pad => pad_mod.pad(self.values, instruction, output_dims),
            .reverse => blk: {
                const input = try self.handle(instruction.inputs[0]);
                break :blk (try buffer_mod.Opaque.reverse(input, instruction.dimensions orelse &.{}, output_dims)) orelse return null;
            },
            .concatenate => blk: {
                const lhs = try self.handle(instruction.inputs[0]);
                const rhs = try self.handle(instruction.inputs[1]);
                break :blk (try buffer_mod.Opaque.concatenate(lhs, rhs, instruction.dimension orelse return null, output_dims)) orelse return null;
            },
            .gather => gather_mod.gather(self.values, instruction, output_dims),
            .scatter => scatter_mod.run(self.values, instruction, output_dims),
            .sort => blk: {
                const input = try self.handle(instruction.inputs[0]);
                const dimension = instruction.dimension orelse return null;
                const sorted = (try buffer_mod.Opaque.sort(input, dimension, output_dims)) orelse return null;
                break :blk (try values_mod.reverseIfDescending(sorted, dimension, output_dims, instruction.compare_direction orelse .lt)) orelse return null;
            },
            else => null,
        };
    }

    fn handle(self: Context, value_id: ir.ValueId) Error!BufferHandle {
        if (value_id.index >= self.values.handles.len) return error.CommandSubmissionFailed;
        return self.values.handles[value_id.index] orelse error.CommandSubmissionFailed;
    }
};
