const std = @import("std");

const ir = @import("src/compiler/ir");
const tensor = @import("executable_metal_graph_tensor.zig");

/// Describes generated elementwise map kernel arity.
pub const Kind = enum { binary, unary, compare, select, clamp };

/// Controls whether generated scalar literals carry explicit Metal casts.
pub const LiteralStyle = enum { plain, typed };

const BinaryRule = struct {
    kind: ir.PlanInstructionKind,
    token: []const u8,
    call: bool = false,
};

const UnaryRule = struct {
    kind: ir.PlanInstructionKind,
    op: UnaryOp,
};

const UnaryOp = enum {
    negate,
    exp,
    expm1,
    tanh,
    sqrt,
    rsqrt,
    abs,
    ceil,
    floor,
    log,
    log1p,
    logistic,
    sine,
    cosine,
    sign,
};

const binary_rules = [_]BinaryRule{
    .{ .kind = .add, .token = "+" },
    .{ .kind = .subtract, .token = "-" },
    .{ .kind = .multiply, .token = "*" },
    .{ .kind = .divide, .token = "/" },
    .{ .kind = .maximum, .token = "max", .call = true },
    .{ .kind = .minimum, .token = "min", .call = true },
    .{ .kind = .power, .token = "pow", .call = true },
    .{ .kind = .atan2, .token = "atan2", .call = true },
    .{ .kind = .remainder, .token = "fmod", .call = true },
};

const unary_rules = [_]UnaryRule{
    .{ .kind = .negate, .op = .negate },
    .{ .kind = .exp, .op = .exp },
    .{ .kind = .expm1, .op = .expm1 },
    .{ .kind = .tanh, .op = .tanh },
    .{ .kind = .sqrt, .op = .sqrt },
    .{ .kind = .rsqrt, .op = .rsqrt },
    .{ .kind = .abs, .op = .abs },
    .{ .kind = .ceil, .op = .ceil },
    .{ .kind = .floor, .op = .floor },
    .{ .kind = .log, .op = .log },
    .{ .kind = .log1p, .op = .log1p },
    .{ .kind = .logistic, .op = .logistic },
    .{ .kind = .sine, .op = .sine },
    .{ .kind = .cosine, .op = .cosine },
    .{ .kind = .sign, .op = .sign },
};

/// Returns the map family for an instruction kind, when the graph map path supports it.
pub fn kindFor(kind: ir.PlanInstructionKind) ?Kind {
    if (binary(kind) != null) return .binary;
    if (unary(kind) != null) return .unary;
    return switch (kind) {
        .compare => .compare,
        .select => .select,
        .clamp => .clamp,
        else => null,
    };
}

/// Returns the number of inputs required by one map family.
pub fn inputCount(kind: Kind) usize {
    return switch (kind) {
        .unary => 1,
        .binary, .compare => 2,
        .select, .clamp => 3,
    };
}

/// Returns the output scalar type used by the standalone map kernel path.
pub fn outputScalarType(kind: ir.PlanInstructionKind, element_type: ir.BufferType) ?[]const u8 {
    return switch (kind) {
        .add,
        .subtract,
        .multiply,
        .maximum,
        .minimum,
        .select,
        .clamp,
        => tensor.metalProgramScalarType(element_type),
        else => tensor.metalScalarType(element_type),
    };
}

/// Returns true when a unary operation is supported by generated graph maps.
pub fn unarySupported(kind: ir.PlanInstructionKind) bool {
    return unary(kind) != null;
}

/// Returns true when a binary operation is supported by generated graph maps.
pub fn binarySupported(kind: ir.PlanInstructionKind) bool {
    return binary(kind) != null;
}

/// Maps a compiler unary-op enum to the shared instruction kind table.
pub fn unaryInstructionKind(op: ir.ElementwiseUnaryOp) ir.PlanInstructionKind {
    return switch (op) {
        .negate => .negate,
        .exp => .exp,
        .expm1 => .expm1,
        .tanh => .tanh,
        .sqrt => .sqrt,
        .rsqrt => .rsqrt,
        .abs => .abs,
        .cbrt => .cbrt,
        .ceil => .ceil,
        .floor => .floor,
        .log => .log,
        .log1p => .log1p,
        .logistic => .logistic,
        .sine => .sine,
        .cosine => .cosine,
        .not_ => .not_,
        .sign => .sign,
        .is_finite => .is_finite,
        .round_nearest_afz => .round_nearest_afz,
        .round_nearest_even => .round_nearest_even,
        .popcnt => .popcnt,
        .count_leading_zeros => .count_leading_zeros,
    };
}

/// Maps a compiler binary-op enum to the shared instruction kind table.
pub fn binaryInstructionKind(op: ir.ElementwiseBinaryOp) ir.PlanInstructionKind {
    return switch (op) {
        .add => .add,
        .subtract => .subtract,
        .multiply => .multiply,
        .divide => .divide,
        .maximum => .maximum,
        .minimum => .minimum,
        .power => .power,
        .atan2 => .atan2,
        .remainder => .remainder,
        .and_ => .and_,
        .or_ => .or_,
        .xor => .xor,
        .shift_left => .shift_left,
        .shift_right_arithmetic => .shift_right_arithmetic,
        .shift_right_logical => .shift_right_logical,
    };
}

/// Allocates the Metal expression for one supported binary map operation.
pub fn binaryExpression(allocator: std.mem.Allocator, kind: ir.PlanInstructionKind, lhs: []const u8, rhs: []const u8) !?[]const u8 {
    const rule = binary(kind) orelse return null;
    if (rule.call) return try std.fmt.allocPrint(allocator, "{s}(({s}), ({s}))", .{ rule.token, lhs, rhs });
    return try std.fmt.allocPrint(allocator, "(({s}) {s} ({s}))", .{ lhs, rule.token, rhs });
}

/// Allocates the Metal expression for one supported unary map operation.
pub fn unaryExpression(allocator: std.mem.Allocator, kind: ir.PlanInstructionKind, input: []const u8, scalar_type: []const u8, literal_style: LiteralStyle) !?[]const u8 {
    const rule = unary(kind) orelse return null;
    return switch (rule.op) {
        .negate => try std.fmt.allocPrint(allocator, "(-({s}))", .{input}),
        .exp => try std.fmt.allocPrint(allocator, "exp(({s}))", .{input}),
        .expm1 => if (literal_style == .typed)
            try std.fmt.allocPrint(allocator, "(exp(({s})) - {s}(1))", .{ input, scalar_type })
        else
            try std.fmt.allocPrint(allocator, "(exp(({s})) - 1)", .{input}),
        .tanh => try std.fmt.allocPrint(allocator, "tanh(({s}))", .{input}),
        .sqrt => try std.fmt.allocPrint(allocator, "sqrt(({s}))", .{input}),
        .rsqrt => try std.fmt.allocPrint(allocator, "rsqrt(({s}))", .{input}),
        .abs => try std.fmt.allocPrint(allocator, "abs(({s}))", .{input}),
        .ceil => try std.fmt.allocPrint(allocator, "ceil(({s}))", .{input}),
        .floor => try std.fmt.allocPrint(allocator, "floor(({s}))", .{input}),
        .log => try std.fmt.allocPrint(allocator, "log(({s}))", .{input}),
        .log1p => if (literal_style == .typed)
            try std.fmt.allocPrint(allocator, "log(({s}(1) + ({s})))", .{ scalar_type, input })
        else
            try std.fmt.allocPrint(allocator, "log((1 + ({s})))", .{input}),
        .logistic => if (literal_style == .typed)
            try std.fmt.allocPrint(allocator, "({s}(1) / ({s}(1) + exp(-({s}))))", .{ scalar_type, scalar_type, input })
        else
            try std.fmt.allocPrint(allocator, "(1 / (1 + exp(-({s}))))", .{input}),
        .sine => try std.fmt.allocPrint(allocator, "sin(({s}))", .{input}),
        .cosine => try std.fmt.allocPrint(allocator, "cos(({s}))", .{input}),
        .sign => if (literal_style == .typed)
            try std.fmt.allocPrint(allocator, "((({s}) > {s}(0)) ? {s}(1) : ((({s}) < {s}(0)) ? {s}(-1) : {s}(0)))", .{ input, scalar_type, scalar_type, input, scalar_type, scalar_type, scalar_type })
        else
            try std.fmt.allocPrint(allocator, "((({s}) > 0) ? 1 : ((({s}) < 0) ? -1 : 0))", .{ input, input }),
    };
}

/// Allocates a Metal compare expression.
pub fn compareExpression(allocator: std.mem.Allocator, lhs: []const u8, rhs: []const u8, direction: ir.CompareOp) ![]const u8 {
    return try std.fmt.allocPrint(allocator, "(({s}) {s} ({s}))", .{ lhs, tensor.compareToken(direction), rhs });
}

/// Allocates a Metal select expression.
pub fn selectExpression(allocator: std.mem.Allocator, pred: []const u8, on_true: []const u8, on_false: []const u8) ![]const u8 {
    return try std.fmt.allocPrint(allocator, "(({s}) ? ({s}) : ({s}))", .{ pred, on_true, on_false });
}

/// Allocates a Metal clamp expression.
pub fn clampExpression(allocator: std.mem.Allocator, min_value: []const u8, value: []const u8, max_value: []const u8) ![]const u8 {
    return try std.fmt.allocPrint(allocator, "clamp(({s}), ({s}), ({s}))", .{ value, min_value, max_value });
}

fn binary(kind: ir.PlanInstructionKind) ?BinaryRule {
    inline for (binary_rules) |rule| {
        if (kind == rule.kind) return rule;
    }
    return null;
}

fn unary(kind: ir.PlanInstructionKind) ?UnaryRule {
    inline for (unary_rules) |rule| {
        if (kind == rule.kind) return rule;
    }
    return null;
}
