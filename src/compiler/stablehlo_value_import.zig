const std = @import("std");
const mlir = @import("c");
const ir = @import("src/compiler/ir");
const model = @import("compiler_model.zig");
const analysis = @import("stablehlo_analysis_builder.zig");
const decode = @import("stablehlo_decode.zig");
const diagnostic = @import("stablehlo_diagnostic.zig");
const mlir_session = @import("mlir_session.zig");

const AnalyzeError = model.AnalyzeError;
const ValueId = model.ValueId;
const ValueRole = model.ValueRole;
const CapiAnalysisBuilder = analysis.CapiAnalysisBuilder;
const operationName = mlir_session.operationName;
const getOperationAttribute = mlir_session.getOperationAttribute;
const mlirLocationLineColumn = decode.mlirLocationLineColumn;

pub fn valueIdsForOperands(comptime analyzeConstant: anytype, builder: *CapiAnalysisBuilder, op: mlir.MlirOperation) ![]const ValueId {
    return valueIdsForOperandsLimit(analyzeConstant, builder, op, @intCast(mlir.mlirOperationGetNumOperands(op)));
}

pub fn valueIdsForOperandsLimit(comptime analyzeConstant: anytype, builder: *CapiAnalysisBuilder, op: mlir.MlirOperation, count: usize) ![]const ValueId {
    const ids = try builder.allocator.alloc(ValueId, count);
    errdefer builder.allocator.free(ids);
    var index: isize = 0;
    while (index < @as(isize, @intCast(count))) : (index += 1) {
        const operand = mlir.mlirOperationGetOperand(op, index);
        ids[@intCast(index)] = builder.lookupValue(operand) orelse blk: {
            var owner_name: []const u8 = "<none>";
            if (mlir.mlirValueIsAOpResult(operand)) {
                const owner = mlir.mlirOpResultGetOwner(operand);
                owner_name = operationName(owner);
                if (std.mem.eql(u8, owner_name, "stablehlo.constant") or std.mem.eql(u8, owner_name, "sdy.constant")) {
                    try analyzeConstant(builder, owner, "stablehlo.constant");
                    if (builder.lookupValue(operand)) |id| break :blk id;
                }
            }
            const loc = mlirLocationLineColumn(mlir.mlirOperationGetLocation(op));
            try builder.diagnostic_writer.print(
                "invalid StableHLO module: loc={d}:{d} op={s} operand={d} owner={s} detail=\"operand does not reference a previously registered top-level value\" feature=value-graph",
                .{ loc.line, loc.column, operationName(op), index, owner_name },
            );
            return error.InvalidStablehloModule;
        };
    }
    return ids;
}

pub fn registerResultValues(builder: *CapiAnalysisBuilder, op: mlir.MlirOperation, role: ValueRole) ![]const ValueId {
    const count: usize = @intCast(mlir.mlirOperationGetNumResults(op));
    const ids = try builder.allocator.alloc(ValueId, count);
    errdefer builder.allocator.free(ids);
    var index: isize = 0;
    while (index < @as(isize, @intCast(count))) : (index += 1) {
        const result = mlir.mlirOperationGetResult(op, index);
        ids[@intCast(index)] = try builder.registerValue(result, role, try decode.descriptorFromType(builder.allocator, mlir.mlirValueGetType(result)));
    }
    return ids;
}

fn customCallAliasOutputIndex(attr: mlir.MlirAttribute, default_index: usize) ?usize {
    if (!mlir.stablehloAttributeIsAOutputOperandAlias(attr)) return null;
    const tuple_index_count = mlir.stablehloOutputOperandAliasGetOutputTupleIndicesSize(attr);
    if (tuple_index_count == 0) return default_index;
    if (tuple_index_count != 1) return null;
    const output_index = mlir.stablehloOutputOperandAliasGetOutputTupleIndicesElem(attr, 0);
    if (output_index < 0) return null;
    return @intCast(output_index);
}

pub fn registerCustomCallOutputOperandAliases(builder: *CapiAnalysisBuilder, op: mlir.MlirOperation, inputs: []const ValueId, outputs: []const ValueId) AnalyzeError!void {
    const aliases = getOperationAttribute(op, "output_operand_aliases");
    if (mlir.mlirAttributeIsNull(aliases) or !mlir.mlirAttributeIsAArray(aliases)) return;
    const alias_count = mlir.mlirArrayAttrGetNumElements(aliases);
    var alias_index: isize = 0;
    while (alias_index < alias_count) : (alias_index += 1) {
        const alias_attr = mlir.mlirArrayAttrGetElement(aliases, alias_index);
        const output_index = customCallAliasOutputIndex(alias_attr, @intCast(alias_index)) orelse continue;
        if (output_index >= outputs.len) continue;
        const operand_index = mlir.stablehloOutputOperandAliasGetOperandIndex(alias_attr);
        const operand_tuple_index_count = mlir.stablehloOutputOperandAliasGetOperandTupleIndicesSize(alias_attr);
        if (operand_index < 0 or operand_tuple_index_count != 0 or @as(usize, @intCast(operand_index)) >= inputs.len) continue;
        const parameter_index = analysis.parameterAliasForValue(builder, inputs[@intCast(operand_index)]) orelse continue;
        try analysis.appendValueParameterAlias(builder, outputs[output_index], parameter_index);
    }
}

pub fn aliasFirstRegionBlockArgumentsToOperands(comptime analyzeConstant: anytype, builder: *CapiAnalysisBuilder, op: mlir.MlirOperation) AnalyzeError!void {
    if (mlir.mlirOperationGetNumRegions(op) == 0) return;
    const block = mlir.mlirRegionGetFirstBlock(mlir.mlirOperationGetRegion(op, 0));
    if (mlir.mlirBlockIsNull(block)) return;
    const count = @min(mlir.mlirBlockGetNumArguments(block), mlir.mlirOperationGetNumOperands(op));
    const operand_ids = try valueIdsForOperandsLimit(analyzeConstant, builder, op, @intCast(count));
    defer builder.allocator.free(operand_ids);
    var index: isize = 0;
    while (index < count) : (index += 1) try analysis.appendValueAlias(builder, mlir.mlirBlockGetArgument(block, index), operand_ids[@intCast(index)]);
}

pub fn aliasOperationResultsToFirstRegionReturn(builder: *CapiAnalysisBuilder, op: mlir.MlirOperation) AnalyzeError!void {
    if (mlir.mlirOperationGetNumRegions(op) == 0) return;
    const block = mlir.mlirRegionGetFirstBlock(mlir.mlirOperationGetRegion(op, 0));
    if (mlir.mlirBlockIsNull(block)) return;
    const terminator = mlir.mlirBlockGetTerminator(block);
    if (mlir.mlirOperationIsNull(terminator) or !std.mem.eql(u8, operationName(terminator), "stablehlo.return")) return;
    const count = @min(mlir.mlirOperationGetNumResults(op), mlir.mlirOperationGetNumOperands(terminator));
    var index: isize = 0;
    while (index < count) : (index += 1) {
        const result_id = builder.lookupValue(mlir.mlirOperationGetOperand(terminator, index)) orelse {
            const loc = mlirLocationLineColumn(mlir.mlirOperationGetLocation(op));
            try builder.diagnostic_writer.print(
                "invalid StableHLO module: loc={d}:{d} op=stablehlo.composite result={d} detail=\"composite return operand does not reference a lowered value\" feature=stablehlo-composite",
                .{ loc.line, loc.column, index },
            );
            return error.InvalidStablehloModule;
        };
        try analysis.appendValueAlias(builder, mlir.mlirOperationGetResult(op, index), result_id);
    }
}

pub fn aliasOperationResultsToOperands(comptime analyzeConstant: anytype, builder: *CapiAnalysisBuilder, op: mlir.MlirOperation) AnalyzeError!void {
    const count = @min(mlir.mlirOperationGetNumResults(op), mlir.mlirOperationGetNumOperands(op));
    const operand_ids = try valueIdsForOperandsLimit(analyzeConstant, builder, op, @intCast(count));
    defer builder.allocator.free(operand_ids);
    var index: isize = 0;
    while (index < count) : (index += 1) try analysis.appendValueAlias(builder, mlir.mlirOperationGetResult(op, index), operand_ids[@intCast(index)]);
}
