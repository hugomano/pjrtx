const std = @import("std");
const mlir = @import("c");

pub fn mlirStringRef(text: []const u8) mlir.MlirStringRef {
    return mlir.mlirStringRefCreate(text.ptr, text.len);
}

pub fn mlirStringSlice(text: mlir.MlirStringRef) []const u8 {
    if (text.length == 0) return &.{};
    return text.data[0..text.length];
}

pub fn getAttr(op: mlir.MlirOperation, name: []const u8) mlir.MlirAttribute {
    return mlir.mlirOperationGetAttributeByName(op, mlirStringRef(name));
}

pub fn setStringAttr(context: mlir.MlirContext, op: mlir.MlirOperation, name: []const u8, value: []const u8) void {
    mlir.mlirOperationSetAttributeByName(
        op,
        mlirStringRef(name),
        stringAttr(context, value),
    );
}

pub fn setIntegerAttr(context: mlir.MlirContext, op: mlir.MlirOperation, name: []const u8, value: u32) void {
    mlir.mlirOperationSetAttributeByName(
        op,
        mlirStringRef(name),
        integerAttr(context, value),
    );
}

pub fn stringAttr(context: mlir.MlirContext, value: []const u8) mlir.MlirAttribute {
    return mlir.mlirStringAttrGet(context, mlirStringRef(value));
}

pub fn integerAttr(context: mlir.MlirContext, value: u32) mlir.MlirAttribute {
    return mlir.mlirIntegerAttrGet(mlir.mlirIntegerTypeGet(context, 32), value);
}

pub fn stringAttrValue(attr: mlir.MlirAttribute) ?[]const u8 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAString(attr)) return null;
    return mlirStringSlice(mlir.mlirStringAttrGetValue(attr));
}

pub fn stringAttrEquals(attr: mlir.MlirAttribute, expected: []const u8) bool {
    return std.mem.eql(u8, stringAttrValue(attr) orelse return false, expected);
}

pub fn hasStringAttr(op: mlir.MlirOperation, name: []const u8) bool {
    const attr = getAttr(op, name);
    return !mlir.mlirAttributeIsNull(attr) and mlir.mlirAttributeIsAString(attr);
}

pub fn hasIntegerAttr(op: mlir.MlirOperation, name: []const u8) bool {
    const attr = getAttr(op, name);
    return !mlir.mlirAttributeIsNull(attr) and mlir.mlirAttributeIsAInteger(attr);
}

pub fn hasDictionaryAttr(op: mlir.MlirOperation, name: []const u8) bool {
    const attr = getAttr(op, name);
    return !mlir.mlirAttributeIsNull(attr) and mlir.mlirAttributeIsADictionary(attr);
}
