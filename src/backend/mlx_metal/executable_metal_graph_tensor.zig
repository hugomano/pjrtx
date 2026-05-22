const std = @import("std");

const ir = @import("src/compiler/ir");
const encoding = @import("buffer_encoding.zig");
const metalcpp_call = @import("metalcpp_call.zig");

/// Returns the dense tensor descriptor for a graph value when it can be addressed by generated Metal code.
pub fn valueDescriptor(plan: *const ir.ExecutablePlan, value_id: ir.ValueId) ?ir.BufferDescriptor {
    if (value_id.index >= plan.values.len) return null;
    const value = plan.values[value_id.index];
    if (value.storage != .tensor or value.descriptor.layout != .dense_row_major) return null;
    return value.descriptor;
}

pub fn valueIndices(allocator: std.mem.Allocator, value_ids: []const ir.ValueId) ![]u64 {
    const indices = try allocator.alloc(u64, value_ids.len);
    for (indices, value_ids) |*index, value_id| index.* = value_id.index;
    return indices;
}

pub fn kernelName(allocator: std.mem.Allocator, label: []const u8, instruction_index: usize) ![]const u8 {
    return std.fmt.allocPrint(allocator, "pjrtx_metalcpp_{s}_{d}", .{ label, instruction_index });
}

pub fn compareToken(direction: ir.CompareOp) []const u8 {
    return switch (direction) {
        .eq => "==",
        .ne => "!=",
        .ge => ">=",
        .gt => ">",
        .le => "<=",
        .lt => "<",
    };
}

pub fn countConstants(plan: *const ir.ExecutablePlan) usize {
    var count: usize = 0;
    for (plan.instructions) |instruction| {
        if (instruction.kind == .constant) count += 1;
    }
    return count;
}

pub fn inputExpression(allocator: std.mem.Allocator, input_index: usize, descriptor: ir.BufferDescriptor) ![]const u8 {
    return if (denseElementCount(descriptor) == 1)
        try std.fmt.allocPrint(allocator, "in{d}[0]", .{input_index})
    else
        try std.fmt.allocPrint(allocator, "in{d}[elem]", .{input_index});
}

pub fn tensorSpec(descriptor: ir.BufferDescriptor) ?metalcpp_call.TensorSpec {
    const dtype = encoding.dtype(descriptor.element_type) orelse return null;
    return .{ .dtype = @intFromEnum(dtype), .dims = descriptor.dims };
}

pub fn supportedProgramElementType(element_type: ir.BufferType) bool {
    return switch (element_type) {
        .s32, .u32, .u64 => true,
        else => supportedElementType(element_type),
    };
}

pub fn supportedElementType(element_type: ir.BufferType) bool {
    return switch (element_type) {
        .pred, .bf16, .f16, .f32 => true,
        else => false,
    };
}

pub fn tiledFloatAccumulatorType(element_type: ir.BufferType) bool {
    return switch (element_type) {
        .bf16, .f16, .f32 => true,
        else => false,
    };
}

pub fn metalIndexScalarType(element_type: ir.BufferType) ?[]const u8 {
    return switch (element_type) {
        .s32 => "int",
        .u32 => "uint",
        .u64 => "ulong",
        else => null,
    };
}

pub fn metalProgramScalarType(element_type: ir.BufferType) ?[]const u8 {
    return metalScalarType(element_type) orelse metalIndexScalarType(element_type);
}

pub fn metalSpecScalarType(dtype: c_int) ?[]const u8 {
    if (dtype == @intFromEnum((encoding.dtype(.pred) orelse unreachable))) return "bool";
    if (dtype == @intFromEnum((encoding.dtype(.bf16) orelse unreachable))) return "bfloat";
    if (dtype == @intFromEnum((encoding.dtype(.f16) orelse unreachable))) return "half";
    if (dtype == @intFromEnum((encoding.dtype(.f32) orelse unreachable))) return "float";
    if (dtype == @intFromEnum((encoding.dtype(.s32) orelse unreachable))) return "int";
    if (dtype == @intFromEnum((encoding.dtype(.u32) orelse unreachable))) return "uint";
    if (dtype == @intFromEnum((encoding.dtype(.u64) orelse unreachable))) return "ulong";
    return null;
}

pub fn metalScalarType(element_type: ir.BufferType) ?[]const u8 {
    return switch (element_type) {
        .pred => "bool",
        .bf16 => "bfloat",
        .f16 => "half",
        .f32 => "float",
        else => null,
    };
}

pub fn dynamicSliceStartCompatible(descriptor: ir.BufferDescriptor) bool {
    return descriptor.layout == .dense_row_major and denseElementCount(descriptor) == 1 and metalIndexScalarType(descriptor.element_type) != null;
}

pub fn hasExplicitIndexVector(indices_dims: []const i64, index_vector_dim: i64) bool {
    if (index_vector_dim < 0 or index_vector_dim >= @as(i64, @intCast(indices_dims.len))) return false;
    return indices_dims[@intCast(index_vector_dim)] == 1;
}

pub fn denseStride(dims: []const i64, first_axis: usize) u64 {
    var stride: u64 = 1;
    for (dims[first_axis..]) |dim| {
        stride *= @intCast(dim);
    }
    return stride;
}

pub fn sameTensor(descriptor: ir.BufferDescriptor, element_type: ir.BufferType, element_count: usize) bool {
    return descriptor.layout == .dense_row_major and descriptor.element_type == element_type and denseElementCount(descriptor) == element_count;
}

pub fn broadcastCompatible(descriptor: ir.BufferDescriptor, element_type: ir.BufferType, element_count: usize) bool {
    const input_count = denseElementCount(descriptor);
    return descriptor.layout == .dense_row_major and descriptor.element_type == element_type and (input_count == element_count or input_count == 1);
}

pub fn denseElementCount(descriptor: ir.BufferDescriptor) usize {
    const byte_size = ir.denseByteSize(descriptor.element_type, descriptor.dims);
    const element_size = descriptor.element_type.byteSize();
    if (byte_size == 0 or element_size == 0) return 0;
    return byte_size / element_size;
}

pub fn sameDims(lhs: []const i64, rhs: []const i64) bool {
    if (lhs.len != rhs.len) return false;
    for (lhs, rhs) |lhs_dim, rhs_dim| {
        if (lhs_dim != rhs_dim) return false;
    }
    return true;
}

pub fn freeExpressions(allocator: std.mem.Allocator, expressions: []?[]const u8) void {
    for (expressions) |maybe_expr| {
        if (maybe_expr) |expr| allocator.free(expr);
    }
}
