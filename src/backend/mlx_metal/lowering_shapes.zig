const std = @import("std");

const ir = @import("src/compiler/ir");

pub fn inputDescriptor(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, input_index: usize) ?ir.BufferDescriptor {
    if (input_index >= instruction.inputs.len) return null;
    const id = instruction.inputs[input_index];
    if (id.index >= plan.values.len) return null;
    return plan.values[id.index].descriptor;
}

pub fn dimsEqual(lhs: []const i64, rhs: []const i64) bool {
    if (lhs.len != rhs.len) return false;
    for (lhs, rhs) |a, b| if (a != b) return false;
    return true;
}

pub fn validReduceShape(input_dims: []const i64, dimensions: []const i64, output_dims: []const i64) bool {
    var reduced = [_]bool{false} ** 64;
    if (input_dims.len > reduced.len) return false;
    for (dimensions) |dim_i64| {
        if (dim_i64 < 0 or dim_i64 >= @as(i64, @intCast(input_dims.len))) return false;
        const dim: usize = @intCast(dim_i64);
        if (reduced[dim]) return false;
        reduced[dim] = true;
    }
    var expected_rank: usize = 0;
    for (0..input_dims.len) |axis| {
        if (!reduced[axis]) expected_rank += 1;
    }
    if (output_dims.len != expected_rank) return false;
    var out_axis: usize = 0;
    for (input_dims, 0..) |dim, axis| {
        if (reduced[axis]) continue;
        if (output_dims[out_axis] != dim) return false;
        out_axis += 1;
    }
    return true;
}

pub fn validReduceWindowShape(input_dims: []const i64, window_dimensions: []const i64, window_strides: []const i64, base_dilations: []const i64, window_dilations: []const i64, padding_low: []const i64, padding_high: []const i64, output_dims: []const i64) bool {
    const rank = input_dims.len;
    if (rank == 0 or output_dims.len != rank or window_dimensions.len != rank or window_strides.len != rank or base_dilations.len != rank or window_dilations.len != rank or padding_low.len != rank or padding_high.len != rank) return false;
    for (0..rank) |axis| {
        if (input_dims[axis] < 0 or window_dimensions[axis] <= 0 or window_strides[axis] <= 0 or base_dilations[axis] != 1 or window_dilations[axis] <= 0 or padding_low[axis] < 0 or padding_high[axis] < 0) return false;
        const padded = padding_low[axis] + input_dims[axis] + padding_high[axis];
        const window = (window_dimensions[axis] - 1) * window_dilations[axis] + 1;
        const expected = if (padded < window) 0 else @divFloor(padded - window, window_strides[axis]) + 1;
        if (output_dims[axis] != expected) return false;
    }
    return true;
}

pub fn isSupportedFloat(element_type: ir.BufferType) bool {
    return switch (element_type) {
        .f16, .f32, .bf16 => true,
        else => false,
    };
}

pub fn isSupportedInteger(element_type: ir.BufferType) bool {
    return switch (element_type) {
        .s8, .s32, .u8, .u16, .u32, .u64 => true,
        else => false,
    };
}

pub fn isSupportedComparable(element_type: ir.BufferType) bool {
    return element_type == .pred or isSupportedFloat(element_type) or isSupportedInteger(element_type);
}

fn dimsBroadcastTo(input_dims: []const i64, output_dims: []const i64) bool {
    if (input_dims.len > output_dims.len) return false;
    const offset = output_dims.len - input_dims.len;
    for (input_dims, 0..) |dim, index| {
        const output_dim = output_dims[offset + index];
        if (dim != 1 and dim != output_dim) return false;
    }
    return true;
}

pub fn validElementwiseBroadcast(lhs_dims: []const i64, rhs_dims: []const i64, output_dims: []const i64) bool {
    return dimsBroadcastTo(lhs_dims, output_dims) and dimsBroadcastTo(rhs_dims, output_dims);
}

pub fn dotGeneralIsMatmulLike(lhs_dims: []const i64, rhs_dims: []const i64, lhs_batch: []const i64, rhs_batch: []const i64, lhs_contract: []const i64, rhs_contract: []const i64, output_dims: []const i64) bool {
    if (lhs_contract.len != 1 or rhs_contract.len != 1 or lhs_batch.len != rhs_batch.len or lhs_dims.len == 0 or rhs_dims.len < 2 or output_dims.len == 0) return false;
    const lhs_k = lhs_contract[0];
    const rhs_k = rhs_contract[0];
    if (lhs_k < 0 or rhs_k < 0) return false;
    if (@as(usize, @intCast(lhs_k)) >= lhs_dims.len or @as(usize, @intCast(rhs_k)) >= rhs_dims.len) return false;
    if (lhs_dims[@intCast(lhs_k)] != rhs_dims[@intCast(rhs_k)]) return false;
    var lhs_used_buf: [16]bool = [_]bool{false} ** 16;
    var rhs_used_buf: [16]bool = [_]bool{false} ** 16;
    if (lhs_dims.len > lhs_used_buf.len or rhs_dims.len > rhs_used_buf.len) return false;
    const lhs_used = lhs_used_buf[0..lhs_dims.len];
    const rhs_used = rhs_used_buf[0..rhs_dims.len];
    lhs_used[@intCast(lhs_k)] = true;
    rhs_used[@intCast(rhs_k)] = true;
    for (lhs_batch, rhs_batch) |lhs_axis, rhs_axis| {
        if (lhs_axis < 0 or rhs_axis < 0) return false;
        if (@as(usize, @intCast(lhs_axis)) >= lhs_dims.len or @as(usize, @intCast(rhs_axis)) >= rhs_dims.len) return false;
        if (lhs_used[@intCast(lhs_axis)] or rhs_used[@intCast(rhs_axis)]) return false;
        if (lhs_dims[@intCast(lhs_axis)] != rhs_dims[@intCast(rhs_axis)]) return false;
        lhs_used[@intCast(lhs_axis)] = true;
        rhs_used[@intCast(rhs_axis)] = true;
    }
    var expected_buf: [32]i64 = undefined;
    var expected: std.ArrayListUnmanaged(i64) = .initBuffer(&expected_buf);
    for (lhs_batch) |axis| expected.appendBounded(lhs_dims[@intCast(axis)]) catch return false;
    for (lhs_dims, 0..) |dim, axis| if (!lhs_used[axis]) expected.appendBounded(dim) catch return false;
    for (rhs_dims, 0..) |dim, axis| if (!rhs_used[axis]) expected.appendBounded(dim) catch return false;
    return std.mem.eql(i64, expected.items, output_dims);
}

pub fn supportedGatherAxis(instruction: ir.PlanInstruction) ?i64 {
    const start_index_map = instruction.start_index_map orelse return null;
    const slice_sizes = instruction.slice_sizes orelse return null;
    const collapsed_slice_dims = instruction.collapsed_slice_dims orelse return null;
    if (start_index_map.len != 1 or collapsed_slice_dims.len != 1) return null;
    const axis = start_index_map[0];
    if (axis < 0 or collapsed_slice_dims[0] != axis) return null;
    if (slice_sizes.len == 0 or axis >= @as(i64, @intCast(slice_sizes.len)) or slice_sizes[@intCast(axis)] != 1) return null;
    return axis;
}

fn gatherHasExplicitIndexVector(indices_dims: []const i64, start_axis_count: usize, index_vector_dim: i64) bool {
    if (index_vector_dim < 0 or index_vector_dim >= @as(i64, @intCast(indices_dims.len))) return false;
    return indices_dims[@intCast(index_vector_dim)] == @as(i64, @intCast(start_axis_count));
}

fn markUniqueAxis(seen: []bool, axis: i64) bool {
    if (axis < 0 or axis >= @as(i64, @intCast(seen.len))) return false;
    const index: usize = @intCast(axis);
    if (seen[index]) return false;
    seen[index] = true;
    return true;
}

pub fn validGatherShape(operand_dims: []const i64, indices_dims: []const i64, start_index_map: []const i64, collapsed_slice_dims: []const i64, operand_batching_dims: []const i64, start_indices_batching_dims: []const i64, index_vector_dim: i64, slice_sizes: []const i64, offset_dims: []const i64, output_dims: []const i64) bool {
    if (operand_dims.len > 64 or output_dims.len > 64 or start_index_map.len == 0 or slice_sizes.len != operand_dims.len) return false;
    if (start_index_map.len > 1 and !gatherHasExplicitIndexVector(indices_dims, start_index_map.len, index_vector_dim)) return false;
    if (operand_batching_dims.len != start_indices_batching_dims.len) return false;

    var gathered = [_]bool{false} ** 64;
    for (start_index_map) |axis| {
        if (!markUniqueAxis(gathered[0..operand_dims.len], axis)) return false;
    }
    for (operand_batching_dims, start_indices_batching_dims) |operand_axis, indices_axis| {
        if (!markUniqueAxis(gathered[0..operand_dims.len], operand_axis)) return false;
        if (indices_axis < 0 or indices_axis >= @as(i64, @intCast(indices_dims.len)) or indices_axis == index_vector_dim) return false;
        if (operand_dims[@intCast(operand_axis)] != indices_dims[@intCast(indices_axis)]) return false;
        if (slice_sizes[@intCast(operand_axis)] != 1) return false;
    }

    var collapsed = [_]bool{false} ** 64;
    for (collapsed_slice_dims) |axis| {
        if (!markUniqueAxis(collapsed[0..operand_dims.len], axis)) return false;
        if (slice_sizes[@intCast(axis)] != 1) return false;
    }
    for (operand_batching_dims) |axis| {
        if (!markUniqueAxis(collapsed[0..operand_dims.len], axis)) return false;
    }

    var non_collapsed_slice_rank: usize = 0;
    for (operand_dims, 0..) |dim, axis| {
        if (dim < 0 or slice_sizes[axis] < 0 or slice_sizes[axis] > dim) return false;
        if (!collapsed[axis]) non_collapsed_slice_rank += 1;
    }
    if (offset_dims.len != non_collapsed_slice_rank) return false;

    const explicit_vector = gatherHasExplicitIndexVector(indices_dims, start_index_map.len, index_vector_dim);
    const index_prefix_rank = indices_dims.len - @as(usize, if (explicit_vector) 1 else 0);
    if (output_dims.len != index_prefix_rank + non_collapsed_slice_rank) return false;

    var output_is_offset = [_]bool{false} ** 64;
    for (offset_dims) |axis| {
        if (!markUniqueAxis(output_is_offset[0..output_dims.len], axis)) return false;
    }

    var index_axis: usize = 0;
    var slice_axis: usize = 0;
    for (output_dims, 0..) |output_dim, output_axis| {
        if (output_is_offset[output_axis]) {
            while (slice_axis < operand_dims.len and collapsed[slice_axis]) slice_axis += 1;
            if (slice_axis >= slice_sizes.len or output_dim != slice_sizes[slice_axis]) return false;
            slice_axis += 1;
        } else {
            while (explicit_vector and index_axis == @as(usize, @intCast(index_vector_dim))) index_axis += 1;
            if (index_axis >= indices_dims.len or output_dim != indices_dims[index_axis]) return false;
            index_axis += 1;
        }
    }
    return true;
}

/// Returns the axis for scatter forms executable by the MLX axis fast path.
pub fn supportedScatterAxis(instruction: ir.PlanInstruction) ?i64 {
    const scatter_dims_to_operand_dims = instruction.scatter_dims_to_operand_dims orelse return null;
    const inserted_window_dims = instruction.inserted_window_dims orelse return null;
    const input_batching_dims = instruction.input_batching_dims orelse &.{};
    const scatter_indices_batching_dims = instruction.scatter_indices_batching_dims orelse &.{};
    if (scatter_dims_to_operand_dims.len != 1 or inserted_window_dims.len != 1) return null;
    if (input_batching_dims.len != 0 or scatter_indices_batching_dims.len != 0) return null;
    const axis = scatter_dims_to_operand_dims[0];
    if (axis < 0 or inserted_window_dims[0] != axis) return null;
    return axis;
}

pub fn validScatterShape(operand_dims: []const i64, indices_dims: []const i64, update_dims: []const i64, scatter_dims_to_operand_dims: []const i64, inserted_window_dims: []const i64, update_window_dims: []const i64, input_batching_dims: []const i64, scatter_indices_batching_dims: []const i64, index_vector_dim: i64, output_dims: []const i64) bool {
    if (operand_dims.len > 64 or !dimsEqual(operand_dims, output_dims)) return false;
    if (scatter_dims_to_operand_dims.len == 0 or inserted_window_dims.len + update_window_dims.len + input_batching_dims.len != operand_dims.len or input_batching_dims.len != scatter_indices_batching_dims.len) return false;
    const explicit_vector = gatherHasExplicitIndexVector(indices_dims, scatter_dims_to_operand_dims.len, index_vector_dim);
    if (scatter_dims_to_operand_dims.len > 1 and !explicit_vector) return false;

    var scatter_axes = [_]bool{false} ** 64;
    for (scatter_dims_to_operand_dims) |axis| {
        if (!markUniqueAxis(scatter_axes[0..operand_dims.len], axis)) return false;
    }
    for (input_batching_dims, scatter_indices_batching_dims) |operand_axis, indices_axis| {
        if (!markUniqueAxis(scatter_axes[0..operand_dims.len], operand_axis)) return false;
        if (indices_axis < 0 or indices_axis >= @as(i64, @intCast(indices_dims.len)) or indices_axis == index_vector_dim) return false;
        if (operand_dims[@intCast(operand_axis)] != indices_dims[@intCast(indices_axis)]) return false;
    }

    var window_axes = [_]bool{false} ** 64;
    for (inserted_window_dims) |axis| {
        if (!markUniqueAxis(window_axes[0..operand_dims.len], axis)) return false;
    }
    for (update_window_dims) |axis| {
        if (!markUniqueAxis(window_axes[0..operand_dims.len], axis)) return false;
    }
    for (input_batching_dims) |axis| {
        if (axis < 0 or axis >= @as(i64, @intCast(operand_dims.len)) or window_axes[@intCast(axis)]) return false;
    }

    const index_prefix_rank = indices_dims.len - @as(usize, if (explicit_vector) 1 else 0);
    if (update_dims.len != index_prefix_rank + update_window_dims.len) return false;
    var update_axis: usize = 0;
    for (indices_dims, 0..) |dim, axis| {
        if (explicit_vector and axis == @as(usize, @intCast(index_vector_dim))) continue;
        if (update_axis >= update_dims.len or update_dims[update_axis] != dim) return false;
        update_axis += 1;
    }
    if (update_axis != index_prefix_rank) return false;
    for (update_window_dims, 0..) |operand_axis, window_axis| {
        if (operand_axis < 0 or operand_axis >= @as(i64, @intCast(operand_dims.len))) return false;
        const dim = update_dims[index_prefix_rank + window_axis];
        if (dim < 0 or dim > operand_dims[@intCast(operand_axis)]) return false;
    }
    return true;
}

pub fn scatterUpdateShapeMatchesAxis(operand_dims: []const i64, indices_dims: []const i64, update_dims: []const i64, index_vector_dim: i64, axis: i64) bool {
    if (axis < 0 or axis >= @as(i64, @intCast(operand_dims.len))) return false;
    var expected_rank: usize = operand_dims.len - 1;
    for (indices_dims, 0..) |dim, dim_index| {
        if (index_vector_dim >= 0 and dim_index == @as(usize, @intCast(index_vector_dim)) and dim == 1) continue;
        expected_rank += 1;
    }
    if (update_dims.len != expected_rank) return false;

    var update_index: usize = 0;
    for (operand_dims[0..@intCast(axis)]) |dim| {
        if (update_dims[update_index] != dim) return false;
        update_index += 1;
    }
    for (indices_dims, 0..) |dim, dim_index| {
        if (index_vector_dim >= 0 and dim_index == @as(usize, @intCast(index_vector_dim)) and dim == 1) continue;
        if (update_dims[update_index] != dim) return false;
        update_index += 1;
    }
    const axis_index: usize = @intCast(axis);
    for (operand_dims[axis_index + 1 ..]) |dim| {
        if (update_dims[update_index] != dim) return false;
        update_index += 1;
    }
    return update_index == update_dims.len;
}

pub fn gatherOutputShapeMatchesTake(operand_dims: []const i64, indices_dims: []const i64, index_vector_dim: i64, axis: i64, output_dims: []const i64) bool {
    if (axis < 0 or axis >= @as(i64, @intCast(operand_dims.len))) return false;
    var expected_rank: usize = operand_dims.len - 1;
    for (indices_dims, 0..) |dim, dim_index| {
        if (index_vector_dim >= 0 and dim_index == @as(usize, @intCast(index_vector_dim)) and dim == 1) continue;
        expected_rank += 1;
    }
    if (output_dims.len != expected_rank) return false;

    var out_index: usize = 0;
    for (operand_dims[0..@intCast(axis)]) |dim| {
        if (output_dims[out_index] != dim) return false;
        out_index += 1;
    }
    for (indices_dims, 0..) |dim, dim_index| {
        if (index_vector_dim >= 0 and dim_index == @as(usize, @intCast(index_vector_dim)) and dim == 1) continue;
        if (output_dims[out_index] != dim) return false;
        out_index += 1;
    }
    const axis_index: usize = @intCast(axis);
    for (operand_dims[axis_index + 1 ..]) |dim| {
        if (output_dims[out_index] != dim) return false;
        out_index += 1;
    }
    return out_index == output_dims.len;
}
