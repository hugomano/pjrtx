const std = @import("std");
const ir = @import("src/compiler/ir");

pub const Partitioner = enum {
    shardy,
    gspmd,
};

pub const CompileOptions = ir.CompileOptions;

pub const ProgramFormat = enum {
    stablehlo_text,
    stablehlo_bytecode,
    unknown,

    pub fn parse(text: []const u8) ProgramFormat {
        if (text.len == 0) return .stablehlo_text;
        if (std.mem.eql(u8, text, "mlir") or
            std.mem.eql(u8, text, "mlir_text") or
            std.mem.eql(u8, text, "stablehlo") or
            std.mem.eql(u8, text, "stablehlo_text"))
        {
            return .stablehlo_text;
        }
        if (std.mem.eql(u8, text, "stablehlo_bytecode") or std.mem.eql(u8, text, "mlir_bytecode")) {
            return .stablehlo_bytecode;
        }
        return .unknown;
    }
};

pub const Dialect = enum {
    chlo,
    func,
    stablehlo,
    sdy,
};

pub const Operation = struct {
    name: []const u8,
    line: usize,
    column: usize,
    inputs: []const ir.ValueId = &.{},
    outputs: []const ir.ValueId = &.{},
    region_ids: []const ir.RegionId = &.{},
    dtype: []const u8 = "unknown",
    rank: ?usize = null,
    dims: []const i64 = &.{},
    permutation: []const i64 = &.{},
    broadcast_dimensions: []const i64 = &.{},
    start_indices: []const i64 = &.{},
    limit_indices: []const i64 = &.{},
    strides: []const i64 = &.{},
    slice_sizes: []const i64 = &.{},
    edge_padding_low: []const i64 = &.{},
    edge_padding_high: []const i64 = &.{},
    interior_padding: []const i64 = &.{},
    window_dimensions: []const i64 = &.{},
    window_strides: []const i64 = &.{},
    base_dilations: []const i64 = &.{},
    window_dilations: []const i64 = &.{},
    window_reversal: []const bool = &.{},
    offset_dims: []const i64 = &.{},
    collapsed_slice_dims: []const i64 = &.{},
    operand_batching_dims: []const i64 = &.{},
    start_indices_batching_dims: []const i64 = &.{},
    start_index_map: []const i64 = &.{},
    update_window_dims: []const i64 = &.{},
    inserted_window_dims: []const i64 = &.{},
    input_batching_dims: []const i64 = &.{},
    scatter_indices_batching_dims: []const i64 = &.{},
    scatter_dims_to_operand_dims: []const i64 = &.{},
    index_vector_dim: ?i64 = null,
    scatter_update_kind: ?ir.ScatterUpdateKind = null,
    dimension: ?i64 = null,
    top_k_k: ?i64 = null,
    iota_dimension: ?i64 = null,
    fft_kind: ?ir.FftKind = null,
    dimensions: []const i64 = &.{},
    tuple_index: ?i64 = null,
    lower: ?bool = null,
    triangular_left_side: ?bool = null,
    triangular_lower: ?bool = null,
    triangular_unit_diagonal: ?bool = null,
    triangular_transpose: ?ir.TriangularSolveTranspose = null,
    custom_call_target: []const u8 = &.{},
    rng_distribution: ?ir.RngDistribution = null,
    reduce_dimensions: []const i64 = &.{},
    lhs_batch_dimensions: []const i64 = &.{},
    rhs_batch_dimensions: []const i64 = &.{},
    lhs_contracting_dimensions: []const i64 = &.{},
    rhs_contracting_dimensions: []const i64 = &.{},
    input_batch_dimension: ?i64 = null,
    input_feature_dimension: ?i64 = null,
    input_spatial_dimensions: []const i64 = &.{},
    kernel_input_feature_dimension: ?i64 = null,
    kernel_output_feature_dimension: ?i64 = null,
    kernel_spatial_dimensions: []const i64 = &.{},
    output_batch_dimension: ?i64 = null,
    output_feature_dimension: ?i64 = null,
    output_spatial_dimensions: []const i64 = &.{},
    feature_group_count: ?i64 = null,
    batch_group_count: ?i64 = null,
    compare_direction: ?ir.CompareOp = null,
    literal: []const u8 = &.{},
    sharding: []const u8 = "unspecified",

    pub fn deinit(self: Operation, allocator: std.mem.Allocator) void {
        deinitOperation(allocator, self);
    }
};


fn deinitOperation(allocator: std.mem.Allocator, self: Operation) void {
allocator.free(self.name);
allocator.free(self.inputs);
allocator.free(self.outputs);
allocator.free(self.region_ids);
allocator.free(self.dtype);
allocator.free(self.dims);
allocator.free(self.permutation);
allocator.free(self.broadcast_dimensions);
allocator.free(self.start_indices);
allocator.free(self.limit_indices);
allocator.free(self.strides);
allocator.free(self.slice_sizes);
allocator.free(self.edge_padding_low);
allocator.free(self.edge_padding_high);
allocator.free(self.interior_padding);
allocator.free(self.window_dimensions);
allocator.free(self.window_strides);
allocator.free(self.base_dilations);
allocator.free(self.window_dilations);
allocator.free(self.window_reversal);
allocator.free(self.offset_dims);
allocator.free(self.collapsed_slice_dims);
allocator.free(self.operand_batching_dims);
allocator.free(self.start_indices_batching_dims);
allocator.free(self.start_index_map);
allocator.free(self.update_window_dims);
allocator.free(self.inserted_window_dims);
allocator.free(self.input_batching_dims);
allocator.free(self.scatter_indices_batching_dims);
allocator.free(self.scatter_dims_to_operand_dims);
allocator.free(self.dimensions);
allocator.free(self.custom_call_target);
allocator.free(self.reduce_dimensions);
allocator.free(self.lhs_batch_dimensions);
allocator.free(self.rhs_batch_dimensions);
allocator.free(self.lhs_contracting_dimensions);
allocator.free(self.rhs_contracting_dimensions);
allocator.free(self.input_spatial_dimensions);
allocator.free(self.kernel_spatial_dimensions);
allocator.free(self.output_spatial_dimensions);
allocator.free(self.literal);
allocator.free(self.sharding);
}
pub const ShardingKind = ir.ShardingKind;
pub const ShardingMetadata = ir.ShardingMetadata;
pub const ShardingPlan = ir.ShardingPlan;
pub const Value = ir.Value;
pub const ValueId = ir.ValueId;
pub const ValueRole = ir.ValueRole;
pub const PlanInstructionKind = ir.PlanInstructionKind;
pub const PlanInstruction = ir.PlanInstruction;
pub const ExecutablePlan = ir.ExecutablePlan;

pub const ModuleAnalysis = struct {
    allocator: std.mem.Allocator,
    source: []u8,
    dialects: []Dialect,
    ops: []Operation,
    values: []ir.Value,
    regions: []ir.PlanRegion,
    output_ids: []const ir.ValueId,
    output_aliases: []const ir.OutputAlias,
    parameter_descriptors: []ir.BufferDescriptor,
    num_parameters: usize,
    num_outputs: usize,
    parameter_shardings: []ShardingMetadata,
    output_shardings: []ShardingMetadata,

    pub fn deinit(self: *ModuleAnalysis) void {
        for (self.output_shardings) |sharding| sharding.deinit(self.allocator);
        for (self.parameter_shardings) |sharding| sharding.deinit(self.allocator);
        for (self.parameter_descriptors) |descriptor| self.allocator.free(descriptor.dims);
        for (self.values) |value| {
            self.allocator.free(value.descriptor.dims);
            self.allocator.free(value.elements);
        }
        freeRegions(self.allocator, self.regions);
        for (self.ops) |op| op.deinit(self.allocator);
        self.allocator.free(self.output_ids);
        self.allocator.free(self.output_aliases);
        self.allocator.free(self.values);
        self.allocator.free(self.output_shardings);
        self.allocator.free(self.parameter_shardings);
        self.allocator.free(self.parameter_descriptors);
        self.allocator.free(self.ops);
        self.allocator.free(self.dialects);
        self.allocator.free(self.source);
    }
};

/// Source line and column decoded from an MLIR location.
pub const SourceLoc = struct {
    line: usize,
    column: usize,
};

fn freeDescriptorList(allocator: std.mem.Allocator, descriptors: []const ir.BufferDescriptor) void {
    for (descriptors) |descriptor| allocator.free(descriptor.dims);
    allocator.free(descriptors);
}

fn freeRegionValueList(allocator: std.mem.Allocator, values: []const ir.RegionValue) void {
    for (values) |value| {
        allocator.free(value.descriptor.dims);
        if (value.literal) |literal| allocator.free(literal);
    }
    allocator.free(values);
}

fn freeRegionInstructionList(allocator: std.mem.Allocator, instructions: []const ir.RegionInstruction) void {
    for (instructions) |instruction| {
        allocator.free(instruction.inputs);
        allocator.free(instruction.outputs);
        freeDescriptorList(allocator, instruction.operand_descriptors);
        freeDescriptorList(allocator, instruction.result_descriptors);
    }
    allocator.free(instructions);
}

fn freeRegions(allocator: std.mem.Allocator, regions: []const ir.PlanRegion) void {
    for (regions) |region| {
        freeRegionValueList(allocator, region.values);
        freeDescriptorList(allocator, region.argument_descriptors);
        freeRegionInstructionList(allocator, region.instructions);
        freeDescriptorList(allocator, region.return_descriptors);
        allocator.free(region.terminator_operands);
        freeDescriptorList(allocator, region.terminator_operand_descriptors);
    }
    allocator.free(regions);
}

pub const AnalyzeError = error{
    UnsupportedProgramFormat,
    UnsupportedProgramEncoding,
    InvalidStablehloModule,
    InvalidManualComputation,
    GspmdNotEnabled,
    UnsupportedOp,
    UnsupportedElementType,
    UnsupportedSharding,
    OutOfMemory,
    ReadFailed,
    StreamTooLong,
    WriteFailed,
};

