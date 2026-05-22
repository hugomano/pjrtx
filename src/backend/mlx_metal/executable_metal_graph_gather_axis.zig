const ir = @import("src/compiler/ir");
const tensor = @import("executable_metal_graph_tensor.zig");

/// Describes the single-axis gather form supported by executable Metal graph kernels.
pub const Axis = struct {
    axis: usize,
    explicit_index_vector: bool,
    collapsed_slice_dims: []const i64,
    operand_batching_dims: []const i64,
    start_indices_batching_dims: []const i64,
    offset_dims: []const i64,
    index_vector_dim: i64,

    pub fn outputAxisForIndexAxis(self: Axis, output_rank: usize, indices_axis: usize) ?usize {
        if (self.explicit_index_vector and indices_axis == @as(usize, @intCast(self.index_vector_dim))) return null;
        var prefix_index: usize = 0;
        for (0..indices_axis) |axis| {
            if (self.explicit_index_vector and axis == @as(usize, @intCast(self.index_vector_dim))) continue;
            prefix_index += 1;
        }
        var seen_prefix: usize = 0;
        for (0..output_rank) |output_axis| {
            if (containsAxis(self.offset_dims, output_axis)) continue;
            if (seen_prefix == prefix_index) return output_axis;
            seen_prefix += 1;
        }
        return null;
    }

    pub fn outputAxisForOperandAxis(self: Axis, operand_axis: usize) ?usize {
        if (containsAxis(self.collapsed_slice_dims, operand_axis) or containsAxis(self.operand_batching_dims, operand_axis)) return null;
        var offset_index: usize = 0;
        for (0..operand_axis) |axis| {
            if (containsAxis(self.collapsed_slice_dims, axis) or containsAxis(self.operand_batching_dims, axis)) continue;
            offset_index += 1;
        }
        if (offset_index >= self.offset_dims.len) return null;
        return @intCast(self.offset_dims[offset_index]);
    }

    pub fn batchingIndexAxis(self: Axis, operand_axis: usize) ?usize {
        for (self.operand_batching_dims, self.start_indices_batching_dims) |operand_axis_i64, indices_axis_i64| {
            if (operand_axis_i64 == @as(i64, @intCast(operand_axis))) return @intCast(indices_axis_i64);
        }
        return null;
    }
};

/// Recognizes and validates the supported single-axis gather shape contract.
pub fn analyze(instruction: ir.PlanInstruction, operand: ir.BufferDescriptor, indices: ir.BufferDescriptor, output: ir.BufferDescriptor) ?Axis {
    const start_index_map = instruction.start_index_map orelse return null;
    const slice_sizes = instruction.slice_sizes orelse return null;
    if (start_index_map.len != 1 or slice_sizes.len != operand.dims.len) return null;
    if (operand.dims.len > 64 or indices.dims.len > 64 or output.dims.len > 64) return null;
    const collapsed_slice_dims = instruction.collapsed_slice_dims orelse &.{};
    const operand_batching_dims = instruction.operand_batching_dims orelse &.{};
    const start_indices_batching_dims = instruction.start_indices_batching_dims orelse &.{};
    const offset_dims = instruction.offset_dims orelse &.{};
    const index_vector_dim = instruction.index_vector_dim orelse 0;
    if (operand_batching_dims.len != start_indices_batching_dims.len) return null;
    const axis_i64 = start_index_map[0];
    if (axis_i64 < 0 or axis_i64 >= @as(i64, @intCast(operand.dims.len))) return null;
    const axis: usize = @intCast(axis_i64);
    if (operand.dims[axis] <= 0 or slice_sizes[axis] != 1 or !containsAxis(collapsed_slice_dims, axis)) return null;
    if (!validateOperandAxes(operand, indices, slice_sizes, operand_batching_dims, start_indices_batching_dims, collapsed_slice_dims, index_vector_dim)) return null;
    const explicit_index_vector = tensor.hasExplicitIndexVector(indices.dims, index_vector_dim);
    if (!validateOutputShape(operand, indices, output, slice_sizes, collapsed_slice_dims, operand_batching_dims, offset_dims, explicit_index_vector, index_vector_dim)) return null;
    return .{
        .axis = axis,
        .explicit_index_vector = explicit_index_vector,
        .collapsed_slice_dims = collapsed_slice_dims,
        .operand_batching_dims = operand_batching_dims,
        .start_indices_batching_dims = start_indices_batching_dims,
        .offset_dims = offset_dims,
        .index_vector_dim = index_vector_dim,
    };
}

fn validateOperandAxes(operand: ir.BufferDescriptor, indices: ir.BufferDescriptor, slice_sizes: []const i64, operand_batching_dims: []const i64, start_indices_batching_dims: []const i64, collapsed_slice_dims: []const i64, index_vector_dim: i64) bool {
    for (operand_batching_dims, start_indices_batching_dims) |operand_axis_i64, indices_axis_i64| {
        if (indices_axis_i64 < 0 or indices_axis_i64 >= @as(i64, @intCast(indices.dims.len)) or indices_axis_i64 == index_vector_dim) return false;
        const operand_axis: usize = @intCast(operand_axis_i64);
        const indices_axis: usize = @intCast(indices_axis_i64);
        if (operand_axis >= operand.dims.len or operand.dims[operand_axis] != indices.dims[indices_axis] or slice_sizes[operand_axis] != 1) return false;
    }
    for (collapsed_slice_dims) |collapsed_i64| {
        if (collapsed_i64 < 0 or collapsed_i64 >= @as(i64, @intCast(operand.dims.len))) return false;
        if (slice_sizes[@intCast(collapsed_i64)] != 1) return false;
    }
    return true;
}

fn validateOutputShape(operand: ir.BufferDescriptor, indices: ir.BufferDescriptor, output: ir.BufferDescriptor, slice_sizes: []const i64, collapsed_slice_dims: []const i64, operand_batching_dims: []const i64, offset_dims: []const i64, explicit_index_vector: bool, index_vector_dim: i64) bool {
    const index_prefix_rank = indices.dims.len - @as(usize, if (explicit_index_vector) 1 else 0);
    var non_collapsed_rank: usize = 0;
    for (operand.dims, 0..) |dim, operand_axis| {
        if (dim <= 0 or slice_sizes[operand_axis] < 0 or slice_sizes[operand_axis] > dim) return false;
        if (!containsAxis(collapsed_slice_dims, operand_axis) and !containsAxis(operand_batching_dims, operand_axis)) non_collapsed_rank += 1;
    }
    if (offset_dims.len != non_collapsed_rank or output.dims.len != index_prefix_rank + non_collapsed_rank) return false;
    var output_is_offset = [_]bool{false} ** 64;
    for (offset_dims) |offset_i64| if (!markAxisI64(output_is_offset[0..output.dims.len], offset_i64)) return false;
    return outputAxesMatch(operand, indices, output, slice_sizes, collapsed_slice_dims, operand_batching_dims, output_is_offset[0..output.dims.len], explicit_index_vector, index_vector_dim);
}

fn outputAxesMatch(operand: ir.BufferDescriptor, indices: ir.BufferDescriptor, output: ir.BufferDescriptor, slice_sizes: []const i64, collapsed_slice_dims: []const i64, operand_batching_dims: []const i64, output_is_offset: []const bool, explicit_index_vector: bool, index_vector_dim: i64) bool {
    var index_axis: usize = 0;
    var slice_axis: usize = 0;
    for (output.dims, 0..) |output_dim, output_axis| {
        if (output_is_offset[output_axis]) {
            while (slice_axis < operand.dims.len and (containsAxis(collapsed_slice_dims, slice_axis) or containsAxis(operand_batching_dims, slice_axis))) slice_axis += 1;
            if (slice_axis >= slice_sizes.len or output_dim != slice_sizes[slice_axis]) return false;
            slice_axis += 1;
        } else {
            while (explicit_index_vector and index_axis == @as(usize, @intCast(index_vector_dim))) index_axis += 1;
            if (index_axis >= indices.dims.len or output_dim != indices.dims[index_axis]) return false;
            index_axis += 1;
        }
    }
    return true;
}

fn containsAxis(axes: []const i64, axis: usize) bool {
    for (axes) |candidate| if (candidate == @as(i64, @intCast(axis))) return true;
    return false;
}

fn markAxisI64(seen: []bool, axis: i64) bool {
    if (axis < 0 or axis >= @as(i64, @intCast(seen.len)) or seen[@intCast(axis)]) return false;
    seen[@intCast(axis)] = true;
    return true;
}
