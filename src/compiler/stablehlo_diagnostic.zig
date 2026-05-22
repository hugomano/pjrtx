const std = @import("std");
const model = @import("compiler_model.zig");

const Operation = model.Operation;

pub fn stablehloOpSupported(name: []const u8) bool {
    const supported = [_][]const u8{
        "constant",
        "return",
        "add",
        "subtract",
        "multiply",
        "divide",
        "maximum",
        "minimum",
        "power",
        "atan2",
        "remainder",
        "and",
        "or",
        "xor",
        "shift_left",
        "shift_right_arithmetic",
        "shift_right_logical",
        "negate",
        "abs",
        "cbrt",
        "ceil",
        "exponential",
        "exponential_minus_one",
        "floor",
        "log",
        "log_plus_one",
        "logistic",
        "sine",
        "cosine",
        "not",
        "sign",
        "is_finite",
        "round_nearest_afz",
        "round_nearest_even",
        "popcnt",
        "count_leading_zeros",
        "tanh",
        "sqrt",
        "rsqrt",
        "convert",
        "bitcast_convert",
        "compare",
        "select",
        "clamp",
        "reshape",
        "broadcast_in_dim",
        "transpose",
        "slice",
        "dynamic_slice",
        "dynamic_update_slice",
        "pad",
        "reverse",
        "concatenate",
        "iota",
        "gather",
        "sort",
        "reduce",
        "reduce_window",
        "dot_general",
        "cholesky",
        "complex",
        "convolution",
        "custom_call",
        "fft",
        "get_tuple_element",
        "imag",
        "optimization_barrier",
        "partition_id",
        "real",
        "reduce_precision",
        "rng",
        "rng_bit_generator",
        "scatter",
        "triangular_solve",
        "tuple",
        "while",
    };
    for (supported) |candidate| {
        if (std.mem.eql(u8, name, candidate)) return true;
    }
    return false;
}

pub fn writeOpDiagnostic(
    writer: *std.Io.Writer,
    comptime label: []const u8,
    op: Operation,
    missing_feature: []const u8,
) std.Io.Writer.Error!void {
    try writer.print(
        "{s}: op={s} loc={d}:{d} dtype={s} rank=",
        .{
            label,
            op.name,
            op.line,
            op.column,
            op.dtype,
        },
    );
    if (op.rank) |rank| {
        try writer.print("{d}", .{rank});
    } else {
        try writer.writeAll("unknown");
    }
    try writer.print(" sharding={s} feature={s}", .{ op.sharding, missing_feature });
}

pub fn writeSimpleDiagnostic(
    writer: *std.Io.Writer,
    comptime label: []const u8,
    line: usize,
    column: usize,
    detail: []const u8,
    feature: []const u8,
) std.Io.Writer.Error!void {
    try writer.print("{s}: loc={d}:{d} detail=\"{s}\" feature={s}", .{ label, line, column, detail, feature });
}

