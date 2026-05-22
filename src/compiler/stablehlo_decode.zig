const std = @import("std");
const mlir = @import("c");
const ir = @import("src/compiler/ir");
const model = @import("compiler_model.zig");
const mlir_session = @import("mlir_session.zig");
const plan_instruction = @import("plan_instruction.zig");

const Dialect = model.Dialect;
const Operation = model.Operation;
const SourceLoc = model.SourceLoc;
const mlirStringRef = mlir_session.mlirStringRef;
const mlirStringSlice = mlir_session.mlirStringSlice;
const getOperationAttribute = mlir_session.getOperationAttribute;
const makeDescriptor = plan_instruction.makeDescriptor;
const bufferTypeFromDtype = plan_instruction.bufferTypeFromDtype;

pub fn descriptorFromType(allocator: std.mem.Allocator, ty: mlir.MlirType) !ir.BufferDescriptor {
    if (mlir.mlirTypeIsNull(ty)) return makeDescriptor(try allocator.dupe(i64, &.{}), .invalid);
    const dtype = try typeDtype(allocator, ty);
    defer allocator.free(dtype);
    return makeDescriptor(try typeDims(allocator, ty), bufferTypeFromDtype(dtype));
}


pub fn addDialect(list: *std.ArrayList(Dialect), allocator: std.mem.Allocator, dialect: Dialect) !void {
    for (list.items) |existing| {
        if (existing == dialect) return;
    }
    try list.append(allocator, dialect);
}

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

pub fn mlirLocationLineColumn(loc: mlir.MlirLocation) SourceLoc {
    if (mlir.mlirLocationIsAFileLineColRange(loc)) {
        return .{
            .line = @intCast(mlir.mlirLocationFileLineColRangeGetStartLine(loc)),
            .column = @intCast(mlir.mlirLocationFileLineColRangeGetStartColumn(loc)),
        };
    }
    if (mlir.mlirLocationIsAName(loc)) return mlirLocationLineColumn(mlir.mlirLocationNameGetChildLoc(loc));
    if (mlir.mlirLocationIsACallSite(loc)) return mlirLocationLineColumn(mlir.mlirLocationCallSiteGetCallee(loc));
    if (mlir.mlirLocationIsAFused(loc) and mlir.mlirLocationFusedGetNumLocations(loc) > 0) {
        var child: mlir.MlirLocation = undefined;
        mlir.mlirLocationFusedGetLocations(loc, &child);
        return mlirLocationLineColumn(child);
    }
    return .{ .line = 0, .column = 0 };
}

pub fn typeDtype(allocator: std.mem.Allocator, ty: mlir.MlirType) ![]u8 {
    const element = if (mlir.mlirTypeIsAShaped(ty)) mlir.mlirShapedTypeGetElementType(ty) else ty;
    if (mlir.mlirTypeIsAComplex(element)) {
        const complex_element = mlir.mlirComplexTypeGetElementType(element);
        if (mlir.mlirTypeIsAF32(complex_element)) return allocator.dupe(u8, "c64");
        if (mlir.mlirTypeIsAF64(complex_element)) return allocator.dupe(u8, "c128");
        return allocator.dupe(u8, "complex<unknown>");
    }
    if (mlir.mlirTypeIsAF16(element)) return allocator.dupe(u8, "f16");
    if (mlir.mlirTypeIsAF32(element)) return allocator.dupe(u8, "f32");
    if (mlir.mlirTypeIsAF64(element)) return allocator.dupe(u8, "f64");
    if (mlir.mlirTypeIsABF16(element)) return allocator.dupe(u8, "bf16");
    if (mlir.mlirTypeIsAInteger(element)) {
        const width = mlir.mlirIntegerTypeGetWidth(element);
        if (width == 1) return allocator.dupe(u8, "pred");
        const prefix: []const u8 = if (mlir.mlirIntegerTypeIsUnsigned(element)) "u" else "i";
        return std.fmt.allocPrint(allocator, "{s}{d}", .{ prefix, width });
    }
    return allocator.dupe(u8, "unknown");
}

pub fn typeRank(ty: mlir.MlirType) ?usize {
    if (!mlir.mlirTypeIsAShaped(ty) or !mlir.mlirShapedTypeHasRank(ty)) return null;
    return @intCast(mlir.mlirShapedTypeGetRank(ty));
}

pub fn typeDims(allocator: std.mem.Allocator, ty: mlir.MlirType) ![]const i64 {
    if (!mlir.mlirTypeIsAShaped(ty) or !mlir.mlirShapedTypeHasRank(ty)) return allocator.dupe(i64, &.{});
    const rank: usize = @intCast(mlir.mlirShapedTypeGetRank(ty));
    const dims = try allocator.alloc(i64, rank);
    var dim_index: isize = 0;
    while (dim_index < @as(isize, @intCast(rank))) : (dim_index += 1) {
        dims[@intCast(dim_index)] = mlir.mlirShapedTypeGetDimSize(ty, dim_index);
    }
    return dims;
}

pub fn intListAttribute(allocator: std.mem.Allocator, attr: mlir.MlirAttribute) ![]const i64 {
    if (mlir.mlirAttributeIsNull(attr)) return allocator.dupe(i64, &.{});
    if (mlir.mlirAttributeIsADenseI64Array(attr)) {
        const count: usize = @intCast(mlir.mlirDenseArrayGetNumElements(attr));
        const values = try allocator.alloc(i64, count);
        var index: isize = 0;
        while (index < @as(isize, @intCast(count))) : (index += 1) {
            values[@intCast(index)] = mlir.mlirDenseI64ArrayGetElement(attr, index);
        }
        return values;
    }
    if (mlir.mlirAttributeIsADenseIntElements(attr)) {
        const count: usize = @intCast(mlir.mlirElementsAttrGetNumElements(attr));
        const values = try allocator.alloc(i64, count);
        var index: isize = 0;
        while (index < @as(isize, @intCast(count))) : (index += 1) {
            values[@intCast(index)] = mlir.mlirDenseElementsAttrGetInt64Value(attr, index);
        }
        return values;
    }
    return allocator.dupe(i64, &.{});
}

pub fn paddingPairList(allocator: std.mem.Allocator, flat: []const i64, pair_index: usize) ![]const i64 {
    if (flat.len % 2 != 0 or pair_index > 1) return allocator.dupe(i64, &.{});
    const values = try allocator.alloc(i64, flat.len / 2);
    for (values, 0..) |*value, index| value.* = flat[index * 2 + pair_index];
    return values;
}

pub fn filledIntList(allocator: std.mem.Allocator, count: usize, value: i64) ![]const i64 {
    const values = try allocator.alloc(i64, count);
    @memset(values, value);
    return values;
}

pub fn intListAttributeOrFill(allocator: std.mem.Allocator, attr: mlir.MlirAttribute, count: usize, value: i64) ![]const i64 {
    const parsed = try intListAttribute(allocator, attr);
    if (parsed.len != 0) return parsed;
    allocator.free(parsed);
    return filledIntList(allocator, count, value);
}

pub fn boolListAttribute(allocator: std.mem.Allocator, attr: mlir.MlirAttribute) ![]const bool {
    if (mlir.mlirAttributeIsNull(attr)) return allocator.dupe(bool, &.{});
    if (mlir.mlirAttributeIsADenseBoolArray(attr)) {
        const count: usize = @intCast(mlir.mlirDenseArrayGetNumElements(attr));
        const values = try allocator.alloc(bool, count);
        var index: isize = 0;
        while (index < @as(isize, @intCast(count))) : (index += 1) {
            values[@intCast(index)] = mlir.mlirDenseBoolArrayGetElement(attr, index);
        }
        return values;
    }
    if (mlir.mlirAttributeIsADenseElements(attr)) {
        const count: usize = @intCast(mlir.mlirElementsAttrGetNumElements(attr));
        const values = try allocator.alloc(bool, count);
        var index: isize = 0;
        while (index < @as(isize, @intCast(count))) : (index += 1) {
            values[@intCast(index)] = mlir.mlirDenseElementsAttrGetBoolValue(attr, index);
        }
        return values;
    }
    return allocator.dupe(bool, &.{});
}

pub fn boolListAttributeOrFill(allocator: std.mem.Allocator, attr: mlir.MlirAttribute, count: usize, value: bool) ![]const bool {
    const parsed = try boolListAttribute(allocator, attr);
    if (parsed.len != 0) return parsed;
    allocator.free(parsed);
    const values = try allocator.alloc(bool, count);
    @memset(values, value);
    return values;
}

pub fn intAttribute(attr: mlir.MlirAttribute) ?i64 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAInteger(attr)) return null;
    return mlir.mlirIntegerAttrGetValueInt(attr);
}

pub fn boolAttribute(attr: mlir.MlirAttribute) ?bool {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsABool(attr)) return null;
    return mlir.mlirBoolAttrGetValue(attr);
}

pub fn stringAttribute(allocator: std.mem.Allocator, attr: mlir.MlirAttribute) ![]const u8 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAString(attr)) return allocator.dupe(u8, &.{});
    return allocator.dupe(u8, mlirStringSlice(mlir.mlirStringAttrGetValue(attr)));
}

pub fn fftKindFromAttr(attr: mlir.MlirAttribute) ?ir.FftKind {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.stablehloAttributeIsAFftTypeAttr(attr)) return null;
    const text = mlirStringSlice(mlir.stablehloFftTypeAttrGetValue(attr));
    if (std.mem.eql(u8, text, "FFT")) return .fft;
    if (std.mem.eql(u8, text, "IFFT")) return .ifft;
    if (std.mem.eql(u8, text, "RFFT")) return .rfft;
    if (std.mem.eql(u8, text, "IRFFT")) return .irfft;
    return null;
}

pub fn rngDistributionFromAttr(attr: mlir.MlirAttribute) ?ir.RngDistribution {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.stablehloAttributeIsARngDistributionAttr(attr)) return null;
    const text = mlirStringSlice(mlir.stablehloRngDistributionAttrGetValue(attr));
    if (std.mem.eql(u8, text, "UNIFORM")) return .uniform;
    if (std.mem.eql(u8, text, "NORMAL")) return .normal;
    return null;
}

pub fn triangularTransposeFromAttr(attr: mlir.MlirAttribute) ?ir.TriangularSolveTranspose {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.stablehloAttributeIsATransposeAttr(attr)) return null;
    const text = mlirStringSlice(mlir.stablehloTransposeAttrGetValue(attr));
    if (std.mem.eql(u8, text, "NO_TRANSPOSE")) return .no_transpose;
    if (std.mem.eql(u8, text, "TRANSPOSE")) return .transpose;
    if (std.mem.eql(u8, text, "ADJOINT")) return .adjoint;
    return null;
}

pub fn resultOrOperandType(op: mlir.MlirOperation) mlir.MlirType {
    if (mlir.mlirOperationGetNumResults(op) > 0) {
        return mlir.mlirValueGetType(mlir.mlirOperationGetResult(op, 0));
    }
    if (mlir.mlirOperationGetNumOperands(op) > 0) {
        return mlir.mlirValueGetType(mlir.mlirOperationGetOperand(op, 0));
    }
    return .{ .ptr = null };
}


pub fn functionSymbolName(op: mlir.MlirOperation) ?[]const u8 {
    const sym_name = getOperationAttribute(op, "sym_name");
    if (mlir.mlirAttributeIsNull(sym_name) or !mlir.mlirAttributeIsAString(sym_name)) return null;
    return mlirStringSlice(mlir.mlirStringAttrGetValue(sym_name));
}

pub fn isEntryFunction(op: mlir.MlirOperation) bool {
    const sym_name = functionSymbolName(op) orelse return false;
    return std.mem.eql(u8, sym_name, "main");
}
