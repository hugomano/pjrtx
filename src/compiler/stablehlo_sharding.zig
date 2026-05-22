const std = @import("std");
const mlir = @import("c");
const model = @import("compiler_model.zig");
const decode = @import("stablehlo_decode.zig");
const mlir_session = @import("mlir_session.zig");

const ShardingKind = model.ShardingKind;
const ShardingMetadata = model.ShardingMetadata;
const getOperationAttribute = mlir_session.getOperationAttribute;
const mlirStringRef = mlir_session.mlirStringRef;
const mlirStringSlice = mlir_session.mlirStringSlice;
const intAttribute = decode.intAttribute;

pub fn operationShardingLabel(allocator: std.mem.Allocator, op: mlir.MlirOperation) ![]u8 {
    const sharding = getOperationAttribute(op, "sdy.sharding");
    if (!mlir.mlirAttributeIsNull(sharding)) {
        if (mlir.sdyAttributeIsATensorShardingPerValueAttr(sharding)) return allocator.dupe(u8, "sdy.sharding_per_value");
        return allocator.dupe(u8, "sdy.sharding");
    }
    return allocator.dupe(u8, "unspecified");
}

pub fn meshNameFromTensorSharding(allocator: std.mem.Allocator, attr: mlir.MlirAttribute) ![]u8 {
    const mesh = mlir.sdyTensorShardingAttrGetMeshOrRef(attr);
    if (!mlir.mlirAttributeIsNull(mesh)) {
        if (mlir.mlirAttributeIsAFlatSymbolRef(mesh)) {
            return allocator.dupe(u8, mlirStringSlice(mlir.mlirFlatSymbolRefAttrGetValue(mesh)));
        }
        if (mlir.mlirAttributeIsASymbolRef(mesh)) {
            return allocator.dupe(u8, mlirStringSlice(mlir.mlirSymbolRefAttrGetRootReference(mesh)));
        }
    }
    return allocator.dupe(u8, "pjrtx_mesh");
}

pub fn makeShardingMetadata(allocator: std.mem.Allocator, kind: ShardingKind, mesh_name: []const u8) !ShardingMetadata {
    return .{ .kind = kind, .mesh_name = try allocator.dupe(u8, mesh_name) };
}

pub fn metadataFromTensorSharding(allocator: std.mem.Allocator, attr: mlir.MlirAttribute) !ShardingMetadata {
    var kind: ShardingKind = .replicated;
    const dim_count = mlir.sdyTensorShardingAttrGetDimShardingsSize(attr);
    var dim_index: isize = 0;
    while (dim_index < dim_count) : (dim_index += 1) {
        const dim = mlir.sdyTensorShardingAttrGetDimShardingsElem(attr, dim_index);
        if (mlir.sdyAttributeIsADimensionShardingAttr(dim) and mlir.sdyDimensionShardingAttrGetAxesSize(dim) > 0) {
            kind = .partitioned;
            break;
        }
    }

    const mesh_name = try meshNameFromTensorSharding(allocator, attr);
    return .{ .kind = kind, .mesh_name = mesh_name };
}

pub fn metadataFromShardingAttribute(allocator: std.mem.Allocator, attr: mlir.MlirAttribute, value_index: usize) !?ShardingMetadata {
    if (mlir.mlirAttributeIsNull(attr)) return null;
    if (mlir.sdyAttributeIsATensorShardingAttr(attr)) return try metadataFromTensorSharding(allocator, attr);
    if (mlir.sdyAttributeIsATensorShardingPerValueAttr(attr)) {
        const count = mlir.sdyTensorShardingPerValueAttrGetShardingsSize(attr);
        if (value_index >= @as(usize, @intCast(count))) return null;
        return try metadataFromTensorSharding(allocator, mlir.sdyTensorShardingPerValueAttrGetShardingsElem(attr, @intCast(value_index)));
    }
    return null;
}

pub fn metadataFromDictionarySharding(allocator: std.mem.Allocator, dictionary: mlir.MlirAttribute) !?ShardingMetadata {
    if (mlir.mlirAttributeIsNull(dictionary) or !mlir.mlirAttributeIsADictionary(dictionary)) return null;
    return try metadataFromShardingAttribute(
        allocator,
        mlir.mlirDictionaryAttrGetElementByName(dictionary, mlirStringRef("sdy.sharding")),
        0,
    );
}

pub fn metadataFromArrayDictionarySharding(allocator: std.mem.Allocator, array: mlir.MlirAttribute, index: usize) !?ShardingMetadata {
    if (mlir.mlirAttributeIsNull(array) or !mlir.mlirAttributeIsAArray(array)) return null;
    const count = mlir.mlirArrayAttrGetNumElements(array);
    if (index >= @as(usize, @intCast(count))) return null;
    return try metadataFromDictionarySharding(allocator, mlir.mlirArrayAttrGetElement(array, @intCast(index)));
}

pub fn aliasingOutputFromArrayDictionary(array: mlir.MlirAttribute, index: usize) ?u32 {
    if (mlir.mlirAttributeIsNull(array) or !mlir.mlirAttributeIsAArray(array)) return null;
    const count = mlir.mlirArrayAttrGetNumElements(array);
    if (index >= @as(usize, @intCast(count))) return null;
    const dictionary = mlir.mlirArrayAttrGetElement(array, @intCast(index));
    if (mlir.mlirAttributeIsNull(dictionary) or !mlir.mlirAttributeIsADictionary(dictionary)) return null;
    const alias_attr = mlir.mlirDictionaryAttrGetElementByName(dictionary, mlirStringRef("tf.aliasing_output"));
    const alias_index = intAttribute(alias_attr) orelse return null;
    if (alias_index < 0 or alias_index > std.math.maxInt(u32)) return null;
    return @intCast(alias_index);
}


pub fn copyShardingMetadata(allocator: std.mem.Allocator, metadata: ShardingMetadata) !ShardingMetadata {
    return makeShardingMetadata(allocator, metadata.kind, metadata.mesh_name);
}

