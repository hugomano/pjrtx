const std = @import("std");
const mlir = @import("c");
const model = @import("compiler_model.zig");
const decode = @import("stablehlo_decode.zig");
const diagnostic = @import("stablehlo_diagnostic.zig");
const value_import = @import("stablehlo_value_import.zig");
const analysis = @import("stablehlo_analysis_builder.zig");
const sharding = @import("stablehlo_sharding.zig");
const mlir_session = @import("mlir_session.zig");

const AnalyzeError = model.AnalyzeError;
const Dialect = model.Dialect;
const CapiAnalysisBuilder = analysis.CapiAnalysisBuilder;
const getOperationAttribute = mlir_session.getOperationAttribute;
const mlirStringSlice = mlir_session.mlirStringSlice;

fn emptyI64(allocator: std.mem.Allocator) ![]const i64 { return allocator.dupe(i64, &.{}); }
fn emptyU8(allocator: std.mem.Allocator) ![]const u8 { return allocator.dupe(u8, &.{}); }

fn analyzeTopKOperationFromCapi(comptime analyzeConstant: anytype, builder: *CapiAnalysisBuilder, op: mlir.MlirOperation, dialect: Dialect, k: i64) AnalyzeError!void {
    builder.saw_program_body = true;
    try decode.addDialect(&builder.dialects, builder.allocator, dialect);
    const ty = decode.resultOrOperandType(op);
    const dtype = if (mlir.mlirTypeIsNull(ty)) try builder.allocator.dupe(u8, "unknown") else try decode.typeDtype(builder.allocator, ty);
    const dims = if (mlir.mlirTypeIsNull(ty)) try emptyI64(builder.allocator) else try decode.typeDims(builder.allocator, ty);
    const inputs = try value_import.valueIdsForOperands(analyzeConstant, builder, op);
    const outputs = try value_import.registerResultValues(builder, op, .instruction_result);
    const sharding_label = try sharding.operationShardingLabel(builder.allocator, op);
    const loc = decode.mlirLocationLineColumn(mlir.mlirOperationGetLocation(op));
    const name = try builder.allocator.dupe(u8, "top_k");

    try builder.ops.append(builder.allocator, .{
        .name = name,
        .line = loc.line,
        .column = loc.column,
        .inputs = inputs,
        .outputs = outputs,
        .dtype = dtype,
        .rank = if (mlir.mlirTypeIsNull(ty)) null else decode.typeRank(ty),
        .dims = dims,
        .permutation = try emptyI64(builder.allocator),
        .broadcast_dimensions = try emptyI64(builder.allocator),
        .start_indices = try emptyI64(builder.allocator),
        .limit_indices = try emptyI64(builder.allocator),
        .strides = try emptyI64(builder.allocator),
        .slice_sizes = try emptyI64(builder.allocator),
        .edge_padding_low = try emptyI64(builder.allocator),
        .edge_padding_high = try emptyI64(builder.allocator),
        .interior_padding = try emptyI64(builder.allocator),
        .offset_dims = try emptyI64(builder.allocator),
        .collapsed_slice_dims = try emptyI64(builder.allocator),
        .operand_batching_dims = try emptyI64(builder.allocator),
        .start_indices_batching_dims = try emptyI64(builder.allocator),
        .start_index_map = try emptyI64(builder.allocator),
        .update_window_dims = try emptyI64(builder.allocator),
        .inserted_window_dims = try emptyI64(builder.allocator),
        .input_batching_dims = try emptyI64(builder.allocator),
        .scatter_indices_batching_dims = try emptyI64(builder.allocator),
        .scatter_dims_to_operand_dims = try emptyI64(builder.allocator),
        .top_k_k = k,
        .dimensions = try emptyI64(builder.allocator),
        .custom_call_target = try emptyU8(builder.allocator),
        .reduce_dimensions = try emptyI64(builder.allocator),
        .lhs_batch_dimensions = try emptyI64(builder.allocator),
        .rhs_batch_dimensions = try emptyI64(builder.allocator),
        .lhs_contracting_dimensions = try emptyI64(builder.allocator),
        .rhs_contracting_dimensions = try emptyI64(builder.allocator),
        .literal = try emptyU8(builder.allocator),
        .sharding = sharding_label,
    });
}

pub fn analyzeChloTopKOperationFromCapi(comptime analyzeConstant: anytype, builder: *CapiAnalysisBuilder, op: mlir.MlirOperation) AnalyzeError!void {
    const k = decode.intAttribute(getOperationAttribute(op, "k")) orelse {
        const loc = decode.mlirLocationLineColumn(mlir.mlirOperationGetLocation(op));
        try diagnostic.writeSimpleDiagnostic(builder.diagnostic_writer, "unsupported op", loc.line, loc.column, "chlo.top_k requires static k", "chlo-top-k");
        return error.UnsupportedOp;
    };
    try analyzeTopKOperationFromCapi(analyzeConstant, builder, op, .chlo, k);
}

pub fn analyzeCompositeTopKOperationFromCapi(comptime analyzeConstant: anytype, builder: *CapiAnalysisBuilder, op: mlir.MlirOperation) AnalyzeError!bool {
    const name_attr = getOperationAttribute(op, "name");
    if (!mlir.mlirAttributeIsNull(name_attr) and mlir.mlirAttributeIsAString(name_attr)) {
        const composite_name = mlirStringSlice(mlir.mlirStringAttrGetValue(name_attr));
        if (std.mem.indexOf(u8, composite_name, "top_k") == null) return false;
    } else if (mlir.mlirOperationGetNumResults(op) != 2) {
        return false;
    }
    const ty = decode.resultOrOperandType(op);
    if (mlir.mlirTypeIsNull(ty)) return false;
    const dims = try decode.typeDims(builder.allocator, ty);
    defer builder.allocator.free(dims);
    if (dims.len == 0) return false;
    try analyzeTopKOperationFromCapi(analyzeConstant, builder, op, .stablehlo, dims[dims.len - 1]);
    return true;
}
