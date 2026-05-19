const std = @import("std");
const mlir = @import("c");
const compiler_facts = @import("pjrtx/compiler/facts");
const mlir_attrs = @import("attrs.zig");
const limits = @import("limits.zig");
const types = @import("types.zig");

pub const PlacementPlanExternalPass = struct {
    var type_id_anchor: u64 = 0;

    pub const Data = struct {
        construct_count: u32 = 0,
        initialize_count: u32 = 0,
        run_count: u32 = 0,
        clone_count: u32 = 0,
        destruct_count: u32 = 0,
        invalid_state: bool = false,
        missing_placement_records: bool = false,
        invalid_record: bool = false,
    };

    pub fn create(data: *Data) mlir.MlirPass {
        const callbacks: mlir.MlirExternalPassCallbacks = .{
            .construct = construct,
            .destruct = destruct,
            .initialize = initialize,
            .clone = clone,
            .run = run,
        };
        return mlir.mlirCreateExternalPass(
            mlir.mlirTypeIDCreate(&type_id_anchor),
            mlirStringRef("pjrtx-placement-plan-external"),
            mlirStringRef("pjrtx-placement-plan-external"),
            mlirStringRef("Verifies PjRTx placement records and marks placement planned"),
            mlirStringRef("builtin.module"),
            0,
            null,
            callbacks,
            data,
        );
    }

    fn construct(user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.construct_count += 1;
    }

    fn destruct(user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.destruct_count += 1;
    }

    fn initialize(_: mlir.MlirContext, user_data: ?*anyopaque) callconv(.c) mlir.MlirLogicalResult {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.initialize_count += 1;
        return mlir.MlirLogicalResult{ .value = 1 };
    }

    fn clone(user_data: ?*anyopaque) callconv(.c) ?*anyopaque {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.clone_count += 1;
        return user_data;
    }

    fn run(op: mlir.MlirOperation, pass: mlir.MlirExternalPass, user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.run_count += 1;

        const state_attr = getAttr(op, "pjrtx.state");
        if (!stringAttrEquals(state_attr, "fusion_planned")) {
            data.invalid_state = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        const records_attr = getAttr(op, "pjrtx.placement.records");
        if (mlir.mlirAttributeIsNull(records_attr) or !mlir.mlirAttributeIsAArray(records_attr)) {
            data.missing_placement_records = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        if (!verifyPlacementRecordsAttr(records_attr)) {
            data.invalid_record = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        const context = mlir.mlirOperationGetContext(op);
        setStringAttr(context, op, "pjrtx.state", types.ModuleState.placement_planned.text());
        setStringAttr(context, op, "pjrtx.placement_plan.pass", "pjrtx-placement-plan-external");
    }

    fn verifyPlacementRecordsAttr(records_attr: mlir.MlirAttribute) bool {
        const count = mlir.mlirArrayAttrGetNumElements(records_attr);
        if (count > limits.max_placement_records) return false;
        var index: isize = 0;
        while (index < count) : (index += 1) {
            const record = mlir.mlirArrayAttrGetElement(records_attr, index);
            if (!verifyPlacementRecordAttr(record)) return false;
        }
        return true;
    }

    fn verifyPlacementRecordAttr(record: mlir.MlirAttribute) bool {
        if (mlir.mlirAttributeIsNull(record) or !mlir.mlirAttributeIsADictionary(record)) return false;
        if (mlir.mlirAttributeIsNull(dictAttr(record, "index")) or !mlir.mlirAttributeIsAInteger(dictAttr(record, "index"))) return false;
        if (mlir.mlirAttributeIsNull(dictAttr(record, "instruction")) or !mlir.mlirAttributeIsAInteger(dictAttr(record, "instruction"))) return false;
        if (!hasNonEmptyStringDictAttr(record, "layout")) return false;
        if (mlir.mlirAttributeIsNull(dictAttr(record, "result_memory")) or !mlir.mlirAttributeIsAInteger(dictAttr(record, "result_memory"))) return false;
        if (!hasNonEmptyStringDictAttr(record, "reason")) return false;

        const outputs = dictAttr(record, "outputs");
        if (mlir.mlirAttributeIsNull(outputs) or !mlir.mlirAttributeIsAArray(outputs)) return false;
        const output_count = mlir.mlirArrayAttrGetNumElements(outputs);
        if (output_count <= 0 or output_count > limits.max_placement_outputs) return false;
        var output_index: isize = 0;
        while (output_index < output_count) : (output_index += 1) {
            if (!mlir.mlirAttributeIsAInteger(mlir.mlirArrayAttrGetElement(outputs, output_index))) return false;
        }

        const tile = dictAttr(record, "tile");
        if (mlir.mlirAttributeIsNull(tile) or !mlir.mlirAttributeIsAArray(tile)) return false;
        const tile_rank = mlir.mlirArrayAttrGetNumElements(tile);
        if (tile_rank <= 0 or tile_rank > limits.max_tile_rank) return false;
        var tile_index: isize = 0;
        while (tile_index < tile_rank) : (tile_index += 1) {
            const dim_attr = mlir.mlirArrayAttrGetElement(tile, tile_index);
            if (!mlir.mlirAttributeIsAInteger(dim_attr) or mlir.mlirIntegerAttrGetValueInt(dim_attr) <= 0) return false;
        }

        const tile_memory = dictAttr(record, "tile_memory");
        if (!mlir.mlirAttributeIsNull(tile_memory) and
            !mlir.mlirAttributeIsAInteger(tile_memory) and
            !stringAttrEquals(tile_memory, "none")) return false;
        return true;
    }
};

pub const CollectivePlanExternalPass = struct {
    var type_id_anchor: u64 = 0;

    pub const Data = struct {
        construct_count: u32 = 0,
        initialize_count: u32 = 0,
        run_count: u32 = 0,
        clone_count: u32 = 0,
        destruct_count: u32 = 0,
        invalid_state: bool = false,
        missing_collective_records: bool = false,
        invalid_record: bool = false,
    };

    pub fn create(data: *Data) mlir.MlirPass {
        const callbacks: mlir.MlirExternalPassCallbacks = .{
            .construct = construct,
            .destruct = destruct,
            .initialize = initialize,
            .clone = clone,
            .run = run,
        };
        return mlir.mlirCreateExternalPass(
            mlir.mlirTypeIDCreate(&type_id_anchor),
            mlirStringRef("pjrtx-collective-plan-external"),
            mlirStringRef("pjrtx-collective-plan-external"),
            mlirStringRef("Verifies PjRTx collective plan records and marks collectives planned"),
            mlirStringRef("builtin.module"),
            0,
            null,
            callbacks,
            data,
        );
    }

    fn construct(user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.construct_count += 1;
    }

    fn destruct(user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.destruct_count += 1;
    }

    fn initialize(_: mlir.MlirContext, user_data: ?*anyopaque) callconv(.c) mlir.MlirLogicalResult {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.initialize_count += 1;
        return mlir.MlirLogicalResult{ .value = 1 };
    }

    fn clone(user_data: ?*anyopaque) callconv(.c) ?*anyopaque {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.clone_count += 1;
        return user_data;
    }

    fn run(op: mlir.MlirOperation, pass: mlir.MlirExternalPass, user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.run_count += 1;

        const state_attr = getAttr(op, "pjrtx.state");
        if (!stringAttrEquals(state_attr, "placement_planned")) {
            data.invalid_state = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        const records_attr = getAttr(op, "pjrtx.collective.records");
        if (mlir.mlirAttributeIsNull(records_attr) or !mlir.mlirAttributeIsAArray(records_attr)) {
            data.missing_collective_records = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        if (!verifyCollectiveRecordsAttr(records_attr)) {
            data.invalid_record = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        const context = mlir.mlirOperationGetContext(op);
        setStringAttr(context, op, "pjrtx.state", types.ModuleState.collectives_planned.text());
        setStringAttr(context, op, "pjrtx.collective_plan.pass", "pjrtx-collective-plan-external");
    }

    fn verifyCollectiveRecordsAttr(records_attr: mlir.MlirAttribute) bool {
        const count = mlir.mlirArrayAttrGetNumElements(records_attr);
        if (count <= 0 or count > limits.max_collective_records) return false;
        var index: isize = 0;
        while (index < count) : (index += 1) {
            if (!verifyCollectiveRecordAttr(mlir.mlirArrayAttrGetElement(records_attr, index))) return false;
        }
        return true;
    }

    fn verifyCollectiveRecordAttr(record: mlir.MlirAttribute) bool {
        if (mlir.mlirAttributeIsNull(record) or !mlir.mlirAttributeIsADictionary(record)) return false;
        if (mlir.mlirAttributeIsNull(dictAttr(record, "index")) or !mlir.mlirAttributeIsAInteger(dictAttr(record, "index"))) return false;
        if (!hasKnownCollectiveDecision(dictAttr(record, "decision"))) return false;
        if (!hasKnownCollectiveAlgorithm(dictAttr(record, "algorithm"))) return false;
        if (mlir.mlirAttributeIsNull(dictAttr(record, "checked")) or !mlir.mlirAttributeIsAInteger(dictAttr(record, "checked"))) return false;
        if (mlir.mlirAttributeIsNull(dictAttr(record, "lowered")) or !mlir.mlirAttributeIsAInteger(dictAttr(record, "lowered"))) return false;
        if (mlir.mlirAttributeIsNull(dictAttr(record, "unsupported")) or !mlir.mlirAttributeIsAInteger(dictAttr(record, "unsupported"))) return false;
        if (collectiveU128FromStringAttrNoDiag(dictAttr(record, "estimated_bytes")) == null) return false;
        if (!validCollectiveOptionalU64StringAttr(dictAttr(record, "estimated_latency_ns"))) return false;
        if (!hasNonEmptyStringDictAttr(record, "reason")) return false;
        return true;
    }
};

fn mlirStringRef(text: []const u8) mlir.MlirStringRef {
    return mlir_attrs.mlirStringRef(text);
}

fn getAttr(op: mlir.MlirOperation, name: []const u8) mlir.MlirAttribute {
    return mlir_attrs.getAttr(op, name);
}

fn setStringAttr(context: mlir.MlirContext, op: mlir.MlirOperation, name: []const u8, value: []const u8) void {
    mlir_attrs.setStringAttr(context, op, name, value);
}

fn stringAttrValue(attr: mlir.MlirAttribute) ?[]const u8 {
    return mlir_attrs.stringAttrValue(attr);
}

fn stringAttrEquals(attr: mlir.MlirAttribute, expected: []const u8) bool {
    return mlir_attrs.stringAttrEquals(attr, expected);
}

fn dictAttr(attr: mlir.MlirAttribute, name: []const u8) mlir.MlirAttribute {
    return mlir.mlirDictionaryAttrGetElementByName(attr, mlirStringRef(name));
}

fn hasNonEmptyStringDictAttr(attr: mlir.MlirAttribute, name: []const u8) bool {
    return (stringAttrValue(dictAttr(attr, name)) orelse return false).len != 0;
}

fn collectiveU128FromStringAttrNoDiag(attr: mlir.MlirAttribute) ?u128 {
    const value = stringAttrValue(attr) orelse return null;
    return std.fmt.parseUnsigned(u128, value, 10) catch null;
}

fn validCollectiveOptionalU64StringAttr(attr: mlir.MlirAttribute) bool {
    const value = stringAttrValue(attr) orelse return false;
    if (std.mem.eql(u8, value, "none")) return true;
    _ = std.fmt.parseUnsigned(u64, value, 10) catch return false;
    return true;
}

fn hasKnownCollectiveDecision(attr: mlir.MlirAttribute) bool {
    const value = stringAttrValue(attr) orelse return false;
    inline for (std.meta.fields(compiler_facts.CollectivePlanDecision)) |field| {
        if (std.mem.eql(u8, value, field.name)) return true;
    }
    return false;
}

fn hasKnownCollectiveAlgorithm(attr: mlir.MlirAttribute) bool {
    const value = stringAttrValue(attr) orelse return false;
    inline for (std.meta.fields(compiler_facts.CollectiveAlgorithm)) |field| {
        if (std.mem.eql(u8, value, field.name)) return true;
    }
    return false;
}
