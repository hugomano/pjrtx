const std = @import("std");

const operation = @import("operation.zig");
const region = @import("region.zig");
const tensor = @import("tensor.zig");
const topology = @import("topology.zig");

/// Stable identifier for a value in an executable plan.
pub const ValueId = struct {
    index: u32,

    /// Sentinel identifier used when no executable-plan value is available.
    pub const invalid: ValueId = .{ .index = std.math.maxInt(u32) };
};

/// Role a value plays in the top-level executable plan.
pub const ValueRole = enum {
    parameter,
    constant,
    instruction_result,
    output,
};

/// Storage shape of a top-level executable-plan value.
pub const ValueStorageKind = enum {
    tensor,
    tuple,
    token,
    complex_pair,
};

/// Top-level executable-plan value with tensor descriptor and ownership role.
pub const Value = struct {
    id: ValueId,
    role: ValueRole,
    descriptor: tensor.BufferDescriptor,
    storage: ValueStorageKind = .tensor,
    elements: []const ValueId = &.{},
};

/// Output aliasing relationship class for result and parameter storage.
pub const OutputAliasKind = enum(u8) {
    /// The output value is the same logical value as the parameter.
    identity,
    /// The output may reuse the parameter's storage if the backend materializes it in place.
    donation,
};

/// Output-to-parameter aliasing record for executable residency planning.
pub const OutputAlias = struct {
    output_index: u32,
    parameter_index: u32,
    kind: OutputAliasKind,
};

/// Top-level instruction record used by executable plans.
pub const PlanInstruction = struct {
    kind: operation.PlanInstructionKind,
    inputs: []const ValueId = &.{},
    outputs: []const ValueId = &.{},
    region_ids: []const region.RegionId = &.{},
    dims: ?[]const i64 = null,
    permutation: ?[]const i64 = null,
    broadcast_dimensions: ?[]const i64 = null,
    start_indices: ?[]const i64 = null,
    limit_indices: ?[]const i64 = null,
    strides: ?[]const i64 = null,
    slice_sizes: ?[]const i64 = null,
    edge_padding_low: ?[]const i64 = null,
    edge_padding_high: ?[]const i64 = null,
    interior_padding: ?[]const i64 = null,
    window_dimensions: ?[]const i64 = null,
    window_strides: ?[]const i64 = null,
    base_dilations: ?[]const i64 = null,
    window_dilations: ?[]const i64 = null,
    window_reversal: ?[]const bool = null,
    offset_dims: ?[]const i64 = null,
    collapsed_slice_dims: ?[]const i64 = null,
    operand_batching_dims: ?[]const i64 = null,
    start_indices_batching_dims: ?[]const i64 = null,
    start_index_map: ?[]const i64 = null,
    update_window_dims: ?[]const i64 = null,
    inserted_window_dims: ?[]const i64 = null,
    input_batching_dims: ?[]const i64 = null,
    scatter_indices_batching_dims: ?[]const i64 = null,
    scatter_dims_to_operand_dims: ?[]const i64 = null,
    index_vector_dim: ?i64 = null,
    scatter_update_kind: ?operation.ScatterUpdateKind = null,
    dimension: ?i64 = null,
    top_k_k: ?i64 = null,
    iota_dimension: ?i64 = null,
    fft_kind: ?operation.FftKind = null,
    dimensions: ?[]const i64 = null,
    tuple_index: ?i64 = null,
    lower: ?bool = null,
    triangular_left_side: ?bool = null,
    triangular_lower: ?bool = null,
    triangular_unit_diagonal: ?bool = null,
    triangular_transpose: ?operation.TriangularSolveTranspose = null,
    custom_call_target: ?[]const u8 = null,
    rng_distribution: ?operation.RngDistribution = null,
    reduce_dimensions: ?[]const i64 = null,
    lhs_batch_dimensions: ?[]const i64 = null,
    rhs_batch_dimensions: ?[]const i64 = null,
    lhs_contracting_dimensions: ?[]const i64 = null,
    rhs_contracting_dimensions: ?[]const i64 = null,
    input_batch_dimension: ?i64 = null,
    input_feature_dimension: ?i64 = null,
    input_spatial_dimensions: ?[]const i64 = null,
    kernel_input_feature_dimension: ?i64 = null,
    kernel_output_feature_dimension: ?i64 = null,
    kernel_spatial_dimensions: ?[]const i64 = null,
    output_batch_dimension: ?i64 = null,
    output_feature_dimension: ?i64 = null,
    output_spatial_dimensions: ?[]const i64 = null,
    feature_group_count: ?i64 = null,
    batch_group_count: ?i64 = null,
    compare_direction: ?operation.CompareOp = null,
    literal: ?[]const u8 = null,
};

/// Owned compiler executable plan handed to runtime and backend layers.
pub const ExecutablePlan = struct {
    allocator: std.mem.Allocator,
    options: topology.CompileOptions,
    values: []Value = &.{},
    regions: []region.PlanRegion = &.{},
    parameter_shardings: []topology.ShardingPlan,
    output_shardings: []topology.ShardingPlan,
    output_ids: []const ValueId = &.{},
    donated_parameter_indices: []const u32 = &.{},
    output_aliases: []const OutputAlias = &.{},
    instructions: []PlanInstruction,

    /// Releases all allocations owned by this executable plan.
    pub fn deinit(self: *ExecutablePlan) void {
        deinitExecutablePlan(self);
    }
};

fn deinitExecutablePlan(self: *ExecutablePlan) void {
self.allocator.free(self.options.device_assignment);
for (self.parameter_shardings) |plan| {
    self.allocator.free(plan.mesh_name);
    self.allocator.free(plan.device_assignment);
}
for (self.output_shardings) |plan| {
    self.allocator.free(plan.mesh_name);
    self.allocator.free(plan.device_assignment);
}
for (self.instructions) |instruction| {
    if (instruction.inputs.len != 0) self.allocator.free(instruction.inputs);
    if (instruction.outputs.len != 0) self.allocator.free(instruction.outputs);
    if (instruction.region_ids.len != 0) self.allocator.free(instruction.region_ids);
    if (instruction.dims) |dims| self.allocator.free(dims);
    if (instruction.permutation) |permutation| self.allocator.free(permutation);
    if (instruction.broadcast_dimensions) |broadcast_dimensions| self.allocator.free(broadcast_dimensions);
    if (instruction.start_indices) |start_indices| self.allocator.free(start_indices);
    if (instruction.limit_indices) |limit_indices| self.allocator.free(limit_indices);
    if (instruction.strides) |strides| self.allocator.free(strides);
    if (instruction.slice_sizes) |slice_sizes| self.allocator.free(slice_sizes);
    if (instruction.edge_padding_low) |padding| self.allocator.free(padding);
    if (instruction.edge_padding_high) |padding| self.allocator.free(padding);
    if (instruction.interior_padding) |padding| self.allocator.free(padding);
    if (instruction.window_dimensions) |dims| self.allocator.free(dims);
    if (instruction.window_strides) |dims| self.allocator.free(dims);
    if (instruction.base_dilations) |dims| self.allocator.free(dims);
    if (instruction.window_dilations) |dims| self.allocator.free(dims);
    if (instruction.window_reversal) |dims| self.allocator.free(dims);
    if (instruction.offset_dims) |dims| self.allocator.free(dims);
    if (instruction.collapsed_slice_dims) |dims| self.allocator.free(dims);
    if (instruction.operand_batching_dims) |dims| self.allocator.free(dims);
    if (instruction.start_indices_batching_dims) |dims| self.allocator.free(dims);
    if (instruction.start_index_map) |dims| self.allocator.free(dims);
    if (instruction.update_window_dims) |dims| self.allocator.free(dims);
    if (instruction.inserted_window_dims) |dims| self.allocator.free(dims);
    if (instruction.input_batching_dims) |dims| self.allocator.free(dims);
    if (instruction.scatter_indices_batching_dims) |dims| self.allocator.free(dims);
    if (instruction.scatter_dims_to_operand_dims) |dims| self.allocator.free(dims);
    if (instruction.dimensions) |dimensions| self.allocator.free(dimensions);
    if (instruction.custom_call_target) |target| self.allocator.free(target);
    if (instruction.reduce_dimensions) |reduce_dimensions| self.allocator.free(reduce_dimensions);
    if (instruction.lhs_batch_dimensions) |dims| self.allocator.free(dims);
    if (instruction.rhs_batch_dimensions) |dims| self.allocator.free(dims);
    if (instruction.lhs_contracting_dimensions) |dims| self.allocator.free(dims);
    if (instruction.rhs_contracting_dimensions) |dims| self.allocator.free(dims);
    if (instruction.input_spatial_dimensions) |dims| self.allocator.free(dims);
    if (instruction.kernel_spatial_dimensions) |dims| self.allocator.free(dims);
    if (instruction.output_spatial_dimensions) |dims| self.allocator.free(dims);
    if (instruction.literal) |literal| self.allocator.free(literal);
}
for (self.values) |value| {
    if (value.descriptor.dims.len != 0) self.allocator.free(value.descriptor.dims);
    if (value.elements.len != 0) self.allocator.free(value.elements);
}
for (self.regions) |plan_region| {
    for (plan_region.values) |value| {
        if (value.descriptor.dims.len != 0) self.allocator.free(value.descriptor.dims);
        if (value.literal) |literal| self.allocator.free(literal);
    }
    for (plan_region.argument_descriptors) |descriptor| {
        if (descriptor.dims.len != 0) self.allocator.free(descriptor.dims);
    }
    for (plan_region.instructions) |instruction| {
        if (instruction.inputs.len != 0) self.allocator.free(instruction.inputs);
        if (instruction.outputs.len != 0) self.allocator.free(instruction.outputs);
        for (instruction.operand_descriptors) |descriptor| {
            if (descriptor.dims.len != 0) self.allocator.free(descriptor.dims);
        }
        for (instruction.result_descriptors) |descriptor| {
            if (descriptor.dims.len != 0) self.allocator.free(descriptor.dims);
        }
        if (instruction.operand_descriptors.len != 0) self.allocator.free(instruction.operand_descriptors);
        if (instruction.result_descriptors.len != 0) self.allocator.free(instruction.result_descriptors);
    }
    for (plan_region.return_descriptors) |descriptor| {
        if (descriptor.dims.len != 0) self.allocator.free(descriptor.dims);
    }
    for (plan_region.terminator_operand_descriptors) |descriptor| {
        if (descriptor.dims.len != 0) self.allocator.free(descriptor.dims);
    }
    if (plan_region.values.len != 0) self.allocator.free(plan_region.values);
    if (plan_region.argument_descriptors.len != 0) self.allocator.free(plan_region.argument_descriptors);
    if (plan_region.instructions.len != 0) self.allocator.free(plan_region.instructions);
    if (plan_region.return_descriptors.len != 0) self.allocator.free(plan_region.return_descriptors);
    if (plan_region.terminator_operands.len != 0) self.allocator.free(plan_region.terminator_operands);
    if (plan_region.terminator_operand_descriptors.len != 0) self.allocator.free(plan_region.terminator_operand_descriptors);
}
if (self.values.len != 0) self.allocator.free(self.values);
if (self.regions.len != 0) self.allocator.free(self.regions);
self.allocator.free(self.parameter_shardings);
self.allocator.free(self.output_shardings);
self.allocator.free(self.output_ids);
if (self.donated_parameter_indices.len != 0) self.allocator.free(self.donated_parameter_indices);
if (self.output_aliases.len != 0) self.allocator.free(self.output_aliases);
self.allocator.free(self.instructions);
}
