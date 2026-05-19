const std = @import("std");
const ids = @import("ids.zig");
const target_pkg = @import("pjrtx/target");

/// Graph facts are extracted compiler views of MLIR state. They stay free of
/// backend/runtime policy so graph correctness is checked before executable
/// planning starts.
pub const ValidationError = error{
    InvalidTensorType,
};

pub const LayoutKind = enum {
    dense_row_major,
    opaque_backend,
};

pub const SourceFrontend = enum {
    pjrt,
    stablehlo,
    shardy,
    internal,
};

pub const SourceRef = struct {
    id: ids.SourceId,
    frontend: SourceFrontend,
    op_name: []const u8,
    source_index: u32,
    location: []const u8,
};

pub const TensorType = struct {
    element_type: target_pkg.BufferType,
    dims: []const i64,
    layout: LayoutKind,
};

pub const TensorFacts = struct {
    pub fn validate(tensor: TensorType, diagnostics: *std.Io.Writer) !void {
        if (tensor.element_type.byteSize() == null) {
            try diagnostics.writeAll("pass=pjrtx-compiler-facts-validate feature=tensor-type reason=invalid element type\n");
            return ValidationError.InvalidTensorType;
        }

        for (tensor.dims) |dim| {
            if (dim < 0) {
                try diagnostics.writeAll("pass=pjrtx-compiler-facts-validate feature=tensor-type reason=dynamic dimensions are unsupported in V0\n");
                return ValidationError.InvalidTensorType;
            }
        }
    }
};

pub const GraphValueRole = enum {
    parameter,
    constant,
    instruction_result,
    output,
};

pub const GraphValue = struct {
    id: ids.GraphValueId,
    ty: TensorType,
    role: GraphValueRole,
    source: ?SourceRef,
};

pub const GraphInstructionKind = enum {
    dot_general,
    elementwise_unary,
    elementwise_binary,
    broadcast,
    reshape,
    transpose,
    collective,
    return_,
};

pub const ElementwiseUnaryOp = enum {
    tanh,
};

pub const ElementwiseBinaryOp = enum {
    add,
};

pub const DotGeneralSpec = struct {
    lhs_contracting_dimension: u32,
    rhs_contracting_dimension: u32,
};

pub const ElementwiseUnarySpec = struct {
    op: ElementwiseUnaryOp,
};

pub const ElementwiseBinarySpec = struct {
    op: ElementwiseBinaryOp,
};

pub const BroadcastSpec = struct {
    dimensions: []const u32,
};

pub const ReshapeSpec = struct {};

pub const TransposeSpec = struct {
    permutation: []const u32,
};

pub const CollectiveOp = enum {
    all_reduce,
};

pub const CollectiveReduction = enum {
    add,
};

pub const CollectiveSpec = struct {
    op: CollectiveOp,
    reduction: CollectiveReduction,
    replica_group_count: u32,
    replica_group_size: u32,
    replica_groups: []const u32,
    channel_id: ?u64,
    channel_type: ?u32,
    uses_token: bool,
};

pub const ReturnSpec = struct {};

pub const GraphPayload = union(enum) {
    dot_general: DotGeneralSpec,
    elementwise_unary: ElementwiseUnarySpec,
    elementwise_binary: ElementwiseBinarySpec,
    broadcast: BroadcastSpec,
    reshape: ReshapeSpec,
    transpose: TransposeSpec,
    collective: CollectiveSpec,
    return_: ReturnSpec,
};

pub const GraphPayloadFacts = struct {
    pub fn matchesKind(kind: GraphInstructionKind, payload: GraphPayload) bool {
        return switch (kind) {
            .dot_general => payload == .dot_general,
            .elementwise_unary => payload == .elementwise_unary,
            .elementwise_binary => payload == .elementwise_binary,
            .broadcast => payload == .broadcast,
            .reshape => payload == .reshape,
            .transpose => payload == .transpose,
            .collective => payload == .collective,
            .return_ => payload == .return_,
        };
    }
};

pub const GraphInstruction = struct {
    id: ids.GraphInstructionId,
    kind: GraphInstructionKind,
    inputs: []const ids.GraphValueId,
    outputs: []const ids.GraphValueId,
    payload: GraphPayload,
    source: SourceRef,
};

test "tensor facts reject V0 dynamic dimensions with diagnostics" {
    const tensor: TensorType = .{
        .element_type = .f32,
        .dims = &.{ 4, -1 },
        .layout = .dense_row_major,
    };
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    try std.testing.expectError(ValidationError.InvalidTensorType, TensorFacts.validate(tensor, &diagnostics.writer));
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "dynamic dimensions are unsupported in V0") != null);
}

test "graph payload facts match graph instruction kind" {
    const payload: GraphPayload = .{ .elementwise_unary = .{ .op = .tanh } };

    try std.testing.expect(GraphPayloadFacts.matchesKind(.elementwise_unary, payload));
    try std.testing.expect(!GraphPayloadFacts.matchesKind(.elementwise_binary, payload));
}
