const std = @import("std");

pub const MAX_DEVICES = 64;

pub const BackendKind = enum {
    metal_mlx,
};

pub const MemoryKind = enum {
    device,
    host_pinned,
    host_unpinned,
    accelerator_local,
    interconnect_visible,
};

pub const BufferType = enum {
    invalid,
    pred,
    s8,
    s16,
    s32,
    s64,
    u8,
    u16,
    u32,
    u64,
    f16,
    f32,
    f64,
    bf16,

    pub fn byteSize(self: BufferType) usize {
        return switch (self) {
            .invalid => 0,
            .pred, .s8, .u8 => 1,
            .s16, .u16, .f16, .bf16 => 2,
            .s32, .u32, .f32 => 4,
            .s64, .u64, .f64 => 8,
        };
    }
};

pub const LayoutKind = enum {
    dense_row_major,
    opaque_backend,
};

pub const ElementwiseBinaryOp = enum {
    add,
    subtract,
    multiply,
    divide,
    maximum,
    minimum,
    power,
    atan2,
    remainder,
    and_,
    or_,
    xor,
    shift_left,
    shift_right_arithmetic,
    shift_right_logical,
};

pub const ElementwiseUnaryOp = enum {
    negate,
    exp,
    expm1,
    tanh,
    sqrt,
    rsqrt,
    abs,
    cbrt,
    ceil,
    floor,
    log,
    log1p,
    logistic,
    sine,
    cosine,
    not_,
    sign,
    is_finite,
    round_nearest_afz,
    round_nearest_even,
    popcnt,
    count_leading_zeros,
};

pub const CompareOp = enum {
    eq,
    ne,
    ge,
    gt,
    le,
    lt,
};

pub const ScatterUpdateKind = enum {
    set,
    add,
};

pub const DeviceDescriptor = struct {
    id: i32,
    local_hardware_id: i32,
    registry_id: u64 = 0,
    process_index: i32 = 0,
    addressable: bool = true,
    name: []const u8,
    debug_string: []const u8,
    memory_bytes: u64 = 0,
    has_unified_memory: bool = false,
    default_memory_id: i32,
};

pub const MemoryDescriptor = struct {
    id: i32,
    kind: MemoryKind,
    debug_string: []const u8,
};

pub const BufferDescriptor = struct {
    element_type: BufferType,
    dims: []const i64,
    layout: LayoutKind = .dense_row_major,
    device_id: i32,
    memory_id: i32,
    shard_index: usize,
};

pub const Placement = struct {
    device_id: i32,
    memory_id: i32,
    memory_kind: MemoryKind,
};

pub const Topology = struct {
    device_assignment: []const i32,
    num_replicas: i32,
    num_partitions: i32,

    pub fn numDevices(self: Topology) usize {
        return @intCast(self.num_replicas * self.num_partitions);
    }
};

pub const CompileOptions = struct {
    num_replicas: i32 = 1,
    num_partitions: i32 = 1,
    use_shardy_partitioner: bool = true,
    device_assignment: []const i32 = &.{},

    pub fn numDevices(self: CompileOptions) usize {
        return @intCast(self.num_replicas * self.num_partitions);
    }
};

pub const ShardingKind = enum {
    replicated,
    partitioned,
    manual,
};

pub const ShardingMetadata = struct {
    kind: ShardingKind,
    mesh_name: []const u8,

    pub fn deinit(self: ShardingMetadata, allocator: std.mem.Allocator) void {
        allocator.free(self.mesh_name);
    }
};

pub const ShardingPlan = struct {
    kind: ShardingKind,
    mesh_name: []const u8,
    device_assignment: []const i32,
};

pub const ValueId = struct {
    index: u32,

    pub const invalid: ValueId = .{ .index = std.math.maxInt(u32) };
};

pub const ValueRole = enum {
    parameter,
    constant,
    instruction_result,
    output,
};

pub const Value = struct {
    id: ValueId,
    role: ValueRole,
    descriptor: BufferDescriptor,
};

pub const PlanInstructionKind = enum {
    constant,
    copy_arg0,
    add,
    subtract,
    multiply,
    divide,
    maximum,
    minimum,
    power,
    atan2,
    remainder,
    and_,
    or_,
    xor,
    shift_left,
    shift_right_arithmetic,
    shift_right_logical,
    negate,
    exp,
    expm1,
    tanh,
    sqrt,
    rsqrt,
    abs,
    cbrt,
    ceil,
    floor,
    log,
    log1p,
    logistic,
    sine,
    cosine,
    not_,
    sign,
    is_finite,
    round_nearest_afz,
    round_nearest_even,
    popcnt,
    count_leading_zeros,
    convert,
    bitcast_convert,
    reshape,
    transpose,
    broadcast_in_dim,
    slice,
    dynamic_slice,
    dynamic_update_slice,
    pad,
    reverse,
    concatenate,
    iota,
    gather,
    sort,
    top_k,
    dot_general,
    reduce_sum,
    reduce_max,
    reduce_and,
    reduce_or,
    compare,
    select,
    clamp,
    cholesky,
    complex,
    convolution,
    custom_call,
    fft,
    get_tuple_element,
    imag,
    partition_id,
    real,
    reduce_precision,
    rng,
    rng_bit_generator,
    scatter,
    triangular_solve,
    tuple,
    while_,
    unsupported,
};

pub const PlanInstruction = struct {
    kind: PlanInstructionKind,
    inputs: []const ValueId = &.{},
    outputs: []const ValueId = &.{},
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
    scatter_update_kind: ?ScatterUpdateKind = null,
    dimension: ?i64 = null,
    top_k_k: ?i64 = null,
    iota_dimension: ?i64 = null,
    dimensions: ?[]const i64 = null,
    tuple_index: ?i64 = null,
    lower: ?bool = null,
    custom_call_target: ?[]const u8 = null,
    reduce_dimensions: ?[]const i64 = null,
    lhs_batch_dimensions: ?[]const i64 = null,
    rhs_batch_dimensions: ?[]const i64 = null,
    lhs_contracting_dimensions: ?[]const i64 = null,
    rhs_contracting_dimensions: ?[]const i64 = null,
    compare_direction: ?CompareOp = null,
    literal: ?[]const u8 = null,
};

pub const ExecutablePlan = struct {
    allocator: std.mem.Allocator,
    options: CompileOptions,
    values: []Value = &.{},
    parameter_shardings: []ShardingPlan,
    output_shardings: []ShardingPlan,
    output_ids: []const ValueId = &.{},
    instructions: []PlanInstruction,

    pub fn deinit(self: *ExecutablePlan) void {
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
            if (instruction.literal) |literal| self.allocator.free(literal);
        }
        for (self.values) |value| {
            if (value.descriptor.dims.len != 0) self.allocator.free(value.descriptor.dims);
        }
        if (self.values.len != 0) self.allocator.free(self.values);
        self.allocator.free(self.parameter_shardings);
        self.allocator.free(self.output_shardings);
        self.allocator.free(self.output_ids);
        self.allocator.free(self.instructions);
    }
};

pub fn denseByteSize(element_type: BufferType, dims: []const i64) usize {
    const element_size = element_type.byteSize();
    if (element_size == 0) return 0;
    var elements: usize = 1;
    for (dims) |dim| {
        if (dim < 0) return 0;
        elements = std.math.mul(usize, elements, @intCast(dim)) catch return 0;
    }
    return std.math.mul(usize, elements, element_size) catch 0;
}

test "core dense byte size rejects invalid dimensions" {
    try std.testing.expectEqual(@as(usize, 16), denseByteSize(.f32, &.{ 2, 2 }));
    try std.testing.expectEqual(@as(usize, 0), denseByteSize(.f32, &.{ -1, 2 }));
}
