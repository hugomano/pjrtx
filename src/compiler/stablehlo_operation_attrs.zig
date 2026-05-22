const std = @import("std");
const mlir = @import("c");
const ir = @import("src/compiler/ir");
const decode = @import("stablehlo_decode.zig");
const mlir_session = @import("mlir_session.zig");

const mlirStringSlice = mlir_session.mlirStringSlice;
const operationName = mlir_session.operationName;
const getOperationAttribute = mlir_session.getOperationAttribute;

pub fn denseLiteralBytes(allocator: std.mem.Allocator, attr: mlir.MlirAttribute, element_type: ir.BufferType, dims: []const i64) ![]const u8 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsADenseElements(attr)) return allocator.dupe(u8, &.{});
    const byte_size = ir.denseByteSize(element_type, dims);
    const bytes = try allocator.alloc(u8, byte_size);
    errdefer allocator.free(bytes);
    const element_count: usize = @intCast(mlir.mlirElementsAttrGetNumElements(attr));
    if (byte_size == 0 or element_count == 0) return bytes;
    switch (element_type) {
        .pred => {
            for (0..element_count) |i| bytes[i] = if (mlir.mlirDenseElementsAttrGetBoolValue(attr, @intCast(i))) 1 else 0;
        },
        .u8 => {
            for (0..element_count) |i| bytes[i] = mlir.mlirDenseElementsAttrGetUInt8Value(attr, @intCast(i));
        },
        .s8 => {
            for (0..element_count) |i| bytes[i] = @bitCast(mlir.mlirDenseElementsAttrGetInt8Value(attr, @intCast(i)));
        },
        .s16, .u16, .u32, .u64, .f16, .f64, .bf16, .c64, .c128 => {
            const raw = mlir.mlirDenseElementsAttrGetRawData(attr) orelse return error.UnsupportedElementType;
            const raw_bytes: [*]const u8 = @ptrCast(raw);
            const element_size = element_type.byteSize();
            if (mlir.mlirDenseElementsAttrIsSplat(attr)) {
                for (0..element_count) |i| {
                    @memcpy(bytes[i * element_size ..][0..element_size], raw_bytes[0..element_size]);
                }
            } else {
                @memcpy(bytes[0..byte_size], raw_bytes[0..byte_size]);
            }
        },
        .f32 => {
            for (0..element_count) |i| {
                const value = mlir.mlirDenseElementsAttrGetFloatValue(attr, @intCast(i));
                std.mem.writeInt(u32, bytes[i * 4 ..][0..4], @bitCast(value), .little);
            }
        },
        .s32 => {
            for (0..element_count) |i| {
                const value: i32 = mlir.mlirDenseElementsAttrGetInt32Value(attr, @intCast(i));
                std.mem.writeInt(u32, bytes[i * 4 ..][0..4], @bitCast(value), .little);
            }
        },
        .s64 => {
            for (0..element_count) |i| {
                const value: i64 = mlir.mlirDenseElementsAttrGetInt64Value(attr, @intCast(i));
                std.mem.writeInt(u64, bytes[i * 8 ..][0..8], @bitCast(value), .little);
            }
        },
        else => return error.UnsupportedElementType,
    }
    return bytes;
}

pub fn stablehloDotDims(allocator: std.mem.Allocator, attr: mlir.MlirAttribute, comptime which: []const u8) ![]const i64 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.stablehloAttributeIsADotDimensionNumbers(attr)) return allocator.dupe(i64, &.{});
    const count: isize = if (std.mem.eql(u8, which, "lhs_batch"))
        mlir.stablehloDotDimensionNumbersGetLhsBatchingDimensionsSize(attr)
    else if (std.mem.eql(u8, which, "rhs_batch"))
        mlir.stablehloDotDimensionNumbersGetRhsBatchingDimensionsSize(attr)
    else if (std.mem.eql(u8, which, "lhs_contract"))
        mlir.stablehloDotDimensionNumbersGetLhsContractingDimensionsSize(attr)
    else
        mlir.stablehloDotDimensionNumbersGetRhsContractingDimensionsSize(attr);
    const values = try allocator.alloc(i64, @intCast(count));
    var index: isize = 0;
    while (index < count) : (index += 1) {
        values[@intCast(index)] = if (std.mem.eql(u8, which, "lhs_batch"))
            mlir.stablehloDotDimensionNumbersGetLhsBatchingDimensionsElem(attr, index)
        else if (std.mem.eql(u8, which, "rhs_batch"))
            mlir.stablehloDotDimensionNumbersGetRhsBatchingDimensionsElem(attr, index)
        else if (std.mem.eql(u8, which, "lhs_contract"))
            mlir.stablehloDotDimensionNumbersGetLhsContractingDimensionsElem(attr, index)
        else
            mlir.stablehloDotDimensionNumbersGetRhsContractingDimensionsElem(attr, index);
    }
    return values;
}

pub fn stablehloConvDims(allocator: std.mem.Allocator, attr: mlir.MlirAttribute, comptime which: []const u8) ![]const i64 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.stablehloAttributeIsAConvDimensionNumbers(attr)) return allocator.dupe(i64, &.{});
    const count = if (std.mem.eql(u8, which, "input_spatial"))
        mlir.stablehloConvDimensionNumbersGetInputSpatialDimensionsSize(attr)
    else if (std.mem.eql(u8, which, "kernel_spatial"))
        mlir.stablehloConvDimensionNumbersGetKernelSpatialDimensionsSize(attr)
    else
        mlir.stablehloConvDimensionNumbersGetOutputSpatialDimensionsSize(attr);
    const values = try allocator.alloc(i64, @intCast(count));
    var index: isize = 0;
    while (index < count) : (index += 1) {
        values[@intCast(index)] = if (std.mem.eql(u8, which, "input_spatial"))
            mlir.stablehloConvDimensionNumbersGetInputSpatialDimensionsElem(attr, index)
        else if (std.mem.eql(u8, which, "kernel_spatial"))
            mlir.stablehloConvDimensionNumbersGetKernelSpatialDimensionsElem(attr, index)
        else
            mlir.stablehloConvDimensionNumbersGetOutputSpatialDimensionsElem(attr, index);
    }
    return values;
}

pub fn stablehloConvDim(attr: mlir.MlirAttribute, comptime which: []const u8) ?i64 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.stablehloAttributeIsAConvDimensionNumbers(attr)) return null;
    return if (std.mem.eql(u8, which, "input_batch"))
        mlir.stablehloConvDimensionNumbersGetInputBatchDimension(attr)
    else if (std.mem.eql(u8, which, "input_feature"))
        mlir.stablehloConvDimensionNumbersGetInputFeatureDimension(attr)
    else if (std.mem.eql(u8, which, "kernel_input_feature"))
        mlir.stablehloConvDimensionNumbersGetKernelInputFeatureDimension(attr)
    else if (std.mem.eql(u8, which, "kernel_output_feature"))
        mlir.stablehloConvDimensionNumbersGetKernelOutputFeatureDimension(attr)
    else if (std.mem.eql(u8, which, "output_batch"))
        mlir.stablehloConvDimensionNumbersGetOutputBatchDimension(attr)
    else
        mlir.stablehloConvDimensionNumbersGetOutputFeatureDimension(attr);
}

pub fn stablehloGatherDims(allocator: std.mem.Allocator, attr: mlir.MlirAttribute, comptime which: []const u8) ![]const i64 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.stablehloAttributeIsAGatherDimensionNumbers(attr)) return allocator.dupe(i64, &.{});
    const count = if (std.mem.eql(u8, which, "offset"))
        mlir.stablehloGatherDimensionNumbersGetOffsetDimsSize(attr)
    else if (std.mem.eql(u8, which, "collapsed"))
        mlir.stablehloGatherDimensionNumbersGetCollapsedSliceDimsSize(attr)
    else if (std.mem.eql(u8, which, "operand_batching"))
        mlir.stablehloGatherDimensionNumbersGetOperandBatchingDimsSize(attr)
    else if (std.mem.eql(u8, which, "start_indices_batching"))
        mlir.stablehloGatherDimensionNumbersGetStartIndicesBatchingDimsSize(attr)
    else
        mlir.stablehloGatherDimensionNumbersGetStartIndexMapSize(attr);
    const values = try allocator.alloc(i64, @intCast(count));
    var index: isize = 0;
    while (index < count) : (index += 1) {
        values[@intCast(index)] = if (std.mem.eql(u8, which, "offset"))
            mlir.stablehloGatherDimensionNumbersGetOffsetDimsElem(attr, index)
        else if (std.mem.eql(u8, which, "collapsed"))
            mlir.stablehloGatherDimensionNumbersGetCollapsedSliceDimsElem(attr, index)
        else if (std.mem.eql(u8, which, "operand_batching"))
            mlir.stablehloGatherDimensionNumbersGetOperandBatchingDimsElem(attr, index)
        else if (std.mem.eql(u8, which, "start_indices_batching"))
            mlir.stablehloGatherDimensionNumbersGetStartIndicesBatchingDimsElem(attr, index)
        else
            mlir.stablehloGatherDimensionNumbersGetStartIndexMapElem(attr, index);
    }
    return values;
}

pub fn stablehloGatherIndexVectorDim(attr: mlir.MlirAttribute) ?i64 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.stablehloAttributeIsAGatherDimensionNumbers(attr)) return null;
    return mlir.stablehloGatherDimensionNumbersGetIndexVectorDim(attr);
}

pub fn stablehloScatterDims(allocator: std.mem.Allocator, attr: mlir.MlirAttribute, comptime which: []const u8) ![]const i64 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.stablehloAttributeIsAScatterDimensionNumbers(attr)) return allocator.dupe(i64, &.{});
    const count = if (std.mem.eql(u8, which, "update_window"))
        mlir.stablehloScatterDimensionNumbersGetUpdateWindowDimsSize(attr)
    else if (std.mem.eql(u8, which, "inserted_window"))
        mlir.stablehloScatterDimensionNumbersGetInsertedWindowDimsSize(attr)
    else if (std.mem.eql(u8, which, "input_batching"))
        mlir.stablehloScatterDimensionNumbersGetInputBatchingDimsSize(attr)
    else if (std.mem.eql(u8, which, "scatter_indices_batching"))
        mlir.stablehloScatterDimensionNumbersGetScatterIndicesBatchingDimsSize(attr)
    else
        mlir.stablehloScatterDimensionNumbersGetScatteredDimsToOperandDimsSize(attr);
    const values = try allocator.alloc(i64, @intCast(count));
    var index: isize = 0;
    while (index < count) : (index += 1) {
        values[@intCast(index)] = if (std.mem.eql(u8, which, "update_window"))
            mlir.stablehloScatterDimensionNumbersGetUpdateWindowDimsElem(attr, index)
        else if (std.mem.eql(u8, which, "inserted_window"))
            mlir.stablehloScatterDimensionNumbersGetInsertedWindowDimsElem(attr, index)
        else if (std.mem.eql(u8, which, "input_batching"))
            mlir.stablehloScatterDimensionNumbersGetInputBatchingDimsElem(attr, index)
        else if (std.mem.eql(u8, which, "scatter_indices_batching"))
            mlir.stablehloScatterDimensionNumbersGetScatterIndicesBatchingDimsElem(attr, index)
        else
            mlir.stablehloScatterDimensionNumbersGetScatteredDimsToOperandDimsElem(attr, index);
    }
    return values;
}

pub fn stablehloScatterIndexVectorDim(attr: mlir.MlirAttribute) ?i64 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.stablehloAttributeIsAScatterDimensionNumbers(attr)) return null;
    return mlir.stablehloDimensionNumbersGetIndexVectorDim(attr);
}

pub fn scatterUpdateKindFromRegion(op: mlir.MlirOperation) ?ir.ScatterUpdateKind {
    const n_regions = mlir.mlirOperationGetNumRegions(op);
    var region_index: isize = 0;
    while (region_index < n_regions) : (region_index += 1) {
        var block = mlir.mlirRegionGetFirstBlock(mlir.mlirOperationGetRegion(op, region_index));
        while (!mlir.mlirBlockIsNull(block)) : (block = mlir.mlirBlockGetNextInRegion(block)) {
            var saw_add = false;
            var child = mlir.mlirBlockGetFirstOperation(block);
            while (!mlir.mlirOperationIsNull(child)) : (child = mlir.mlirOperationGetNextInBlock(child)) {
                if (std.mem.eql(u8, operationName(child), "stablehlo.add")) saw_add = true;
                if (std.mem.eql(u8, operationName(child), "stablehlo.return")) return if (saw_add) .add else .set;
            }
        }
    }
    return null;
}

pub fn compareDirectionFromAttr(attr: mlir.MlirAttribute) ?ir.CompareOp {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.stablehloAttributeIsAComparisonDirectionAttr(attr)) return null;
    const text = mlirStringSlice(mlir.stablehloComparisonDirectionAttrGetValue(attr));
    if (std.mem.eql(u8, text, "EQ")) return .eq;
    if (std.mem.eql(u8, text, "NE")) return .ne;
    if (std.mem.eql(u8, text, "GE")) return .ge;
    if (std.mem.eql(u8, text, "GT")) return .gt;
    if (std.mem.eql(u8, text, "LE")) return .le;
    if (std.mem.eql(u8, text, "LT")) return .lt;
    return null;
}

pub fn reduceKindFromRegion(op: mlir.MlirOperation) []const u8 {
    if (mlir.mlirOperationGetNumResults(op) == 2 and
        regionContainsOperation(op, "stablehlo.compare") and
        regionContainsOperation(op, "stablehlo.select"))
    {
        return "reduce_max";
    }
    const n_regions = mlir.mlirOperationGetNumRegions(op);
    var region_index: isize = 0;
    while (region_index < n_regions) : (region_index += 1) {
        var block = mlir.mlirRegionGetFirstBlock(mlir.mlirOperationGetRegion(op, region_index));
        while (!mlir.mlirBlockIsNull(block)) : (block = mlir.mlirBlockGetNextInRegion(block)) {
            var child = mlir.mlirBlockGetFirstOperation(block);
            while (!mlir.mlirOperationIsNull(child)) : (child = mlir.mlirOperationGetNextInBlock(child)) {
                const name = operationName(child);
                if (std.mem.eql(u8, name, "stablehlo.add")) return "reduce_sum";
                if (std.mem.eql(u8, name, "stablehlo.maximum")) return "reduce_max";
                if (std.mem.eql(u8, name, "stablehlo.minimum")) return "reduce_min";
                if (std.mem.eql(u8, name, "stablehlo.and")) return "reduce_and";
                if (std.mem.eql(u8, name, "stablehlo.or")) return "reduce_or";
            }
        }
    }
    return "reduce";
}

pub fn reduceWindowKindFromRegion(op: mlir.MlirOperation) []const u8 {
    const reduce_kind = reduceKindFromRegion(op);
    if (std.mem.eql(u8, reduce_kind, "reduce_sum")) return "reduce_window_sum";
    if (std.mem.eql(u8, reduce_kind, "reduce_max")) return "reduce_window_max";
    if (mlir.mlirOperationGetNumResults(op) == 2 and
        regionContainsOperation(op, "stablehlo.compare") and
        regionContainsOperation(op, "stablehlo.select"))
    {
        return "reduce_window_max";
    }
    return "reduce_window";
}

fn regionContainsOperation(op: mlir.MlirOperation, target_name: []const u8) bool {
    const n_regions = mlir.mlirOperationGetNumRegions(op);
    var region_index: isize = 0;
    while (region_index < n_regions) : (region_index += 1) {
        var block = mlir.mlirRegionGetFirstBlock(mlir.mlirOperationGetRegion(op, region_index));
        while (!mlir.mlirBlockIsNull(block)) : (block = mlir.mlirBlockGetNextInRegion(block)) {
            var child = mlir.mlirBlockGetFirstOperation(block);
            while (!mlir.mlirOperationIsNull(child)) : (child = mlir.mlirOperationGetNextInBlock(child)) {
                if (std.mem.eql(u8, operationName(child), target_name)) return true;
            }
        }
    }
    return false;
}

pub fn compareDirectionFromSortRegion(op: mlir.MlirOperation) ?ir.CompareOp {
    var last_compare: ?ir.CompareOp = null;
    const n_regions = mlir.mlirOperationGetNumRegions(op);
    var region_index: isize = 0;
    while (region_index < n_regions) : (region_index += 1) {
        var block = mlir.mlirRegionGetFirstBlock(mlir.mlirOperationGetRegion(op, region_index));
        while (!mlir.mlirBlockIsNull(block)) : (block = mlir.mlirBlockGetNextInRegion(block)) {
            var child = mlir.mlirBlockGetFirstOperation(block);
            while (!mlir.mlirOperationIsNull(child)) : (child = mlir.mlirOperationGetNextInBlock(child)) {
                if (std.mem.eql(u8, operationName(child), "stablehlo.compare")) {
                    last_compare = compareDirectionFromAttr(getOperationAttribute(child, "comparison_direction"));
                }
            }
        }
    }
    return last_compare;
}
