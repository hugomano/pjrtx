const std = @import("std");

const operation = @import("operation.zig");
const tensor = @import("tensor.zig");

/// Stable identifier for a nested executable-plan region.
pub const RegionId = struct {
    index: u32,

    /// Sentinel identifier used when no executable-plan region is available.
    pub const invalid: RegionId = .{ .index = std.math.maxInt(u32) };
};

/// Stable identifier for a value scoped to a nested executable-plan region.
pub const RegionValueId = struct {
    index: u32,

    /// Sentinel identifier used when no region-scoped value is available.
    pub const invalid: RegionValueId = .{ .index = std.math.maxInt(u32) };
};

/// Semantic role of a nested executable-plan region.
pub const RegionKind = enum {
    generic,
    reducer,
    comparator,
    scatter_update,
    while_cond,
    while_body,
    composite,
};

/// Role a value plays inside a nested executable-plan region.
pub const RegionValueRole = enum {
    argument,
    constant,
    instruction_result,
};

/// Region-scoped value with tensor descriptor and optional literal payload.
pub const RegionValue = struct {
    id: RegionValueId,
    role: RegionValueRole,
    descriptor: tensor.BufferDescriptor,
    literal: ?[]const u8 = null,
};

/// Nested executable-plan region used by control-flow and reducer operations.
pub const PlanRegion = struct {
    id: RegionId,
    parent_instruction_index: usize,
    kind: RegionKind,
    values: []const RegionValue = &.{},
    argument_descriptors: []const tensor.BufferDescriptor = &.{},
    instructions: []const RegionInstruction = &.{},
    return_descriptors: []const tensor.BufferDescriptor = &.{},
    terminator_operands: []const RegionValueId = &.{},
    terminator_operand_descriptors: []const tensor.BufferDescriptor = &.{},
};

/// Region-scoped instruction record used by nested computation bodies.
pub const RegionInstruction = struct {
    kind: operation.PlanInstructionKind,
    line: usize = 0,
    column: usize = 0,
    inputs: []const RegionValueId = &.{},
    outputs: []const RegionValueId = &.{},
    operand_descriptors: []const tensor.BufferDescriptor = &.{},
    result_descriptors: []const tensor.BufferDescriptor = &.{},
    compare_direction: ?operation.CompareOp = null,
};
