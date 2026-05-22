const std = @import("std");
const mlir = @import("c");
const ir = @import("src/compiler/ir");
const model = @import("compiler_model.zig");
const analysis = @import("stablehlo_analysis_builder.zig");
const attrs = @import("stablehlo_operation_attrs.zig");
const decode = @import("stablehlo_decode.zig");
const memory = @import("plan_memory.zig");
const mlir_session = @import("mlir_session.zig");
const plan_instruction = @import("plan_instruction.zig");

const AnalyzeError = model.AnalyzeError;
const CapiAnalysisBuilder = analysis.CapiAnalysisBuilder;
const operationName = mlir_session.operationName;
const getOperationAttribute = mlir_session.getOperationAttribute;
const instructionKindFromStablehlo = plan_instruction.instructionKindFromStablehlo;
const freeDescriptorList = memory.freeDescriptorList;
const freeRegionContents = memory.freeRegionContents;
const freeRegionInstructionList = memory.freeRegionInstructionList;
const freeRegionValueList = memory.freeRegionValueList;
const cloneDescriptorList = memory.cloneDescriptorList;

pub fn regionKindForOperation(short_name: []const u8, region_index: usize) ir.RegionKind {
    if (std.mem.eql(u8, short_name, "reduce_sum") or
        std.mem.eql(u8, short_name, "reduce_max") or
        std.mem.eql(u8, short_name, "reduce_min") or
        std.mem.eql(u8, short_name, "reduce_and") or
        std.mem.eql(u8, short_name, "reduce_or") or
        std.mem.eql(u8, short_name, "reduce_window_sum") or
        std.mem.eql(u8, short_name, "reduce_window_max"))
    {
        return .reducer;
    }
    if (std.mem.eql(u8, short_name, "sort")) return .comparator;
    if (std.mem.eql(u8, short_name, "scatter")) return .scatter_update;
    if (std.mem.eql(u8, short_name, "while")) return if (region_index == 0) .while_cond else .while_body;
    if (std.mem.eql(u8, short_name, "composite")) return .composite;
    return .generic;
}

fn descriptorListFromBlockArguments(allocator: std.mem.Allocator, block: mlir.MlirBlock) ![]const ir.BufferDescriptor {
    if (mlir.mlirBlockIsNull(block)) return allocator.alloc(ir.BufferDescriptor, 0);
    const count: usize = @intCast(mlir.mlirBlockGetNumArguments(block));
    const descriptors = try allocator.alloc(ir.BufferDescriptor, count);
    var initialized: usize = 0;
    errdefer {
        for (descriptors[0..initialized]) |descriptor| allocator.free(descriptor.dims);
        allocator.free(descriptors);
    }
    var index: isize = 0;
    while (index < @as(isize, @intCast(count))) : (index += 1) {
        descriptors[@intCast(index)] = try decode.descriptorFromType(allocator, mlir.mlirValueGetType(mlir.mlirBlockGetArgument(block, index)));
        initialized += 1;
    }
    return descriptors;
}

fn descriptorListFromTerminatorOperands(allocator: std.mem.Allocator, block: mlir.MlirBlock) ![]const ir.BufferDescriptor {
    if (mlir.mlirBlockIsNull(block)) return allocator.alloc(ir.BufferDescriptor, 0);
    const terminator = mlir.mlirBlockGetTerminator(block);
    if (mlir.mlirOperationIsNull(terminator)) return allocator.alloc(ir.BufferDescriptor, 0);
    const name = operationName(terminator);
    if (!std.mem.eql(u8, name, "stablehlo.return") and !std.mem.eql(u8, name, "func.return") and !std.mem.eql(u8, name, "sdy.return")) {
        return allocator.alloc(ir.BufferDescriptor, 0);
    }
    const count: usize = @intCast(mlir.mlirOperationGetNumOperands(terminator));
    const descriptors = try allocator.alloc(ir.BufferDescriptor, count);
    var initialized: usize = 0;
    errdefer {
        for (descriptors[0..initialized]) |descriptor| allocator.free(descriptor.dims);
        allocator.free(descriptors);
    }
    var index: isize = 0;
    while (index < @as(isize, @intCast(count))) : (index += 1) {
        descriptors[@intCast(index)] = try decode.descriptorFromType(allocator, mlir.mlirValueGetType(mlir.mlirOperationGetOperand(terminator, index)));
        initialized += 1;
    }
    return descriptors;
}

fn descriptorListFromOperationOperands(allocator: std.mem.Allocator, op: mlir.MlirOperation) ![]const ir.BufferDescriptor {
    const count: usize = @intCast(mlir.mlirOperationGetNumOperands(op));
    const descriptors = try allocator.alloc(ir.BufferDescriptor, count);
    var initialized: usize = 0;
    errdefer {
        for (descriptors[0..initialized]) |descriptor| allocator.free(descriptor.dims);
        allocator.free(descriptors);
    }
    var index: isize = 0;
    while (index < @as(isize, @intCast(count))) : (index += 1) {
        descriptors[@intCast(index)] = try decode.descriptorFromType(allocator, mlir.mlirValueGetType(mlir.mlirOperationGetOperand(op, index)));
        initialized += 1;
    }
    return descriptors;
}

fn descriptorListFromOperationResults(allocator: std.mem.Allocator, op: mlir.MlirOperation) ![]const ir.BufferDescriptor {
    const count: usize = @intCast(mlir.mlirOperationGetNumResults(op));
    const descriptors = try allocator.alloc(ir.BufferDescriptor, count);
    var initialized: usize = 0;
    errdefer {
        for (descriptors[0..initialized]) |descriptor| allocator.free(descriptor.dims);
        allocator.free(descriptors);
    }
    var index: isize = 0;
    while (index < @as(isize, @intCast(count))) : (index += 1) {
        descriptors[@intCast(index)] = try decode.descriptorFromType(allocator, mlir.mlirValueGetType(mlir.mlirOperationGetResult(op, index)));
        initialized += 1;
    }
    return descriptors;
}

const RegionValueMapEntry = struct {
    mlir_value: mlir.MlirValue,
    id: ir.RegionValueId,
};

const CapturedRegionBlock = struct {
    values: []const ir.RegionValue,
    instructions: []const ir.RegionInstruction,
    terminator_operands: []const ir.RegionValueId,
};

fn freeCapturedRegionBlock(allocator: std.mem.Allocator, block: CapturedRegionBlock) void {
    freeRegionValueList(allocator, block.values);
    freeRegionInstructionList(allocator, block.instructions);
    allocator.free(block.terminator_operands);
}

fn lookupRegionValue(map: []const RegionValueMapEntry, value: mlir.MlirValue) ?ir.RegionValueId {
    for (map) |entry| {
        if (mlir.mlirValueEqual(entry.mlir_value, value)) return entry.id;
    }
    return null;
}

fn appendRegionValue(
    allocator: std.mem.Allocator,
    values: *std.ArrayList(ir.RegionValue),
    value_map: *std.ArrayList(RegionValueMapEntry),
    value: mlir.MlirValue,
    role: ir.RegionValueRole,
) !ir.RegionValueId {
    if (lookupRegionValue(value_map.items, value)) |existing| return existing;
    const id: ir.RegionValueId = .{ .index = @intCast(values.items.len) };
    const descriptor = try decode.descriptorFromType(allocator, mlir.mlirValueGetType(value));
    errdefer allocator.free(descriptor.dims);
    var actual_role = role;
    var literal: ?[]const u8 = null;
    if (mlir.mlirValueIsAOpResult(value)) {
        const owner = mlir.mlirOpResultGetOwner(value);
        const owner_name = operationName(owner);
        if (std.mem.eql(u8, owner_name, "stablehlo.constant") or std.mem.eql(u8, owner_name, "sdy.constant")) {
            actual_role = .constant;
            literal = try attrs.denseLiteralBytes(allocator, getOperationAttribute(owner, "value"), descriptor.element_type, descriptor.dims);
        }
    }
    errdefer if (literal) |bytes| allocator.free(bytes);
    try values.append(allocator, .{
        .id = id,
        .role = actual_role,
        .descriptor = descriptor,
        .literal = literal,
    });
    errdefer _ = values.pop();
    try value_map.append(allocator, .{ .mlir_value = value, .id = id });
    return id;
}

fn registerRegionBlockArguments(
    allocator: std.mem.Allocator,
    values: *std.ArrayList(ir.RegionValue),
    value_map: *std.ArrayList(RegionValueMapEntry),
    block: mlir.MlirBlock,
) !void {
    const count = mlir.mlirBlockGetNumArguments(block);
    var index: isize = 0;
    while (index < count) : (index += 1) {
        _ = try appendRegionValue(allocator, values, value_map, mlir.mlirBlockGetArgument(block, index), .argument);
    }
}

fn regionValueIdsForOperationOperands(
    allocator: std.mem.Allocator,
    values: *std.ArrayList(ir.RegionValue),
    value_map: *std.ArrayList(RegionValueMapEntry),
    op: mlir.MlirOperation,
) ![]const ir.RegionValueId {
    const count: usize = @intCast(mlir.mlirOperationGetNumOperands(op));
    const ids = try allocator.alloc(ir.RegionValueId, count);
    errdefer allocator.free(ids);
    var index: isize = 0;
    while (index < @as(isize, @intCast(count))) : (index += 1) {
        const operand = mlir.mlirOperationGetOperand(op, index);
        ids[@intCast(index)] = lookupRegionValue(value_map.items, operand) orelse try appendRegionValue(
            allocator,
            values,
            value_map,
            operand,
            .instruction_result,
        );
    }
    return ids;
}

fn regionValueIdsForOperationResults(
    allocator: std.mem.Allocator,
    values: *std.ArrayList(ir.RegionValue),
    value_map: *std.ArrayList(RegionValueMapEntry),
    op: mlir.MlirOperation,
) ![]const ir.RegionValueId {
    const count: usize = @intCast(mlir.mlirOperationGetNumResults(op));
    const ids = try allocator.alloc(ir.RegionValueId, count);
    errdefer allocator.free(ids);
    var index: isize = 0;
    while (index < @as(isize, @intCast(count))) : (index += 1) {
        ids[@intCast(index)] = try appendRegionValue(allocator, values, value_map, mlir.mlirOperationGetResult(op, index), .instruction_result);
    }
    return ids;
}

fn terminatorOperandIds(
    allocator: std.mem.Allocator,
    value_map: []const RegionValueMapEntry,
    block: mlir.MlirBlock,
) ![]const ir.RegionValueId {
    if (mlir.mlirBlockIsNull(block)) return allocator.alloc(ir.RegionValueId, 0);
    const terminator = mlir.mlirBlockGetTerminator(block);
    if (mlir.mlirOperationIsNull(terminator)) return allocator.alloc(ir.RegionValueId, 0);
    const name = operationName(terminator);
    if (!std.mem.eql(u8, name, "stablehlo.return") and !std.mem.eql(u8, name, "func.return") and !std.mem.eql(u8, name, "sdy.return")) {
        return allocator.alloc(ir.RegionValueId, 0);
    }
    const count: usize = @intCast(mlir.mlirOperationGetNumOperands(terminator));
    const ids = try allocator.alloc(ir.RegionValueId, count);
    errdefer allocator.free(ids);
    var index: isize = 0;
    while (index < @as(isize, @intCast(count))) : (index += 1) {
        const operand = mlir.mlirOperationGetOperand(terminator, index);
        ids[@intCast(index)] = lookupRegionValue(value_map, operand) orelse ir.RegionValueId.invalid;
    }
    return ids;
}

fn regionInstructionKindFromOperationName(name: []const u8) ir.PlanInstructionKind {
    if (std.mem.startsWith(u8, name, "stablehlo.")) {
        const raw_name = name["stablehlo.".len..];
        return instructionKindFromStablehlo(raw_name);
    }
    if (std.mem.startsWith(u8, name, "chlo.")) {
        const raw_name = name["chlo.".len..];
        if (std.mem.eql(u8, raw_name, "top_k")) return .top_k;
    }
    return .unsupported;
}

fn captureRegionBlock(allocator: std.mem.Allocator, block: mlir.MlirBlock) !CapturedRegionBlock {
    if (mlir.mlirBlockIsNull(block)) return .{
        .values = try allocator.alloc(ir.RegionValue, 0),
        .instructions = try allocator.alloc(ir.RegionInstruction, 0),
        .terminator_operands = try allocator.alloc(ir.RegionValueId, 0),
    };

    var values: std.ArrayList(ir.RegionValue) = .empty;
    errdefer {
        for (values.items) |value| allocator.free(value.descriptor.dims);
        values.deinit(allocator);
    }

    var value_map: std.ArrayList(RegionValueMapEntry) = .empty;
    defer value_map.deinit(allocator);

    var instructions: std.ArrayList(ir.RegionInstruction) = .empty;
    errdefer {
        for (instructions.items) |instruction| {
            allocator.free(instruction.inputs);
            allocator.free(instruction.outputs);
            freeDescriptorList(allocator, instruction.operand_descriptors);
            freeDescriptorList(allocator, instruction.result_descriptors);
        }
        instructions.deinit(allocator);
    }

    try registerRegionBlockArguments(allocator, &values, &value_map, block);

    var child = mlir.mlirBlockGetFirstOperation(block);
    while (!mlir.mlirOperationIsNull(child)) : (child = mlir.mlirOperationGetNextInBlock(child)) {
        const name = operationName(child);
        if (std.mem.eql(u8, name, "stablehlo.return") or
            std.mem.eql(u8, name, "func.return") or
            std.mem.eql(u8, name, "sdy.return"))
        {
            continue;
        }
        if (std.mem.eql(u8, name, "stablehlo.constant") or std.mem.eql(u8, name, "sdy.constant")) {
            allocator.free(try regionValueIdsForOperationResults(allocator, &values, &value_map, child));
            continue;
        }
        const inputs = try regionValueIdsForOperationOperands(allocator, &values, &value_map, child);
        errdefer allocator.free(inputs);
        const outputs = try regionValueIdsForOperationResults(allocator, &values, &value_map, child);
        errdefer allocator.free(outputs);
        const operands = try descriptorListFromOperationOperands(allocator, child);
        errdefer freeDescriptorList(allocator, operands);
        const results = try descriptorListFromOperationResults(allocator, child);
        errdefer freeDescriptorList(allocator, results);
        const loc = decode.mlirLocationLineColumn(mlir.mlirOperationGetLocation(child));
        try instructions.append(allocator, .{
            .kind = regionInstructionKindFromOperationName(name),
            .line = loc.line,
            .column = loc.column,
            .inputs = inputs,
            .outputs = outputs,
            .operand_descriptors = operands,
            .result_descriptors = results,
            .compare_direction = if (std.mem.eql(u8, name, "stablehlo.compare")) attrs.compareDirectionFromAttr(getOperationAttribute(child, "comparison_direction")) else null,
        });
    }

    const terminator_operands = try terminatorOperandIds(allocator, value_map.items, block);
    errdefer allocator.free(terminator_operands);
    return .{
        .values = try values.toOwnedSlice(allocator),
        .instructions = try instructions.toOwnedSlice(allocator),
        .terminator_operands = terminator_operands,
    };
}

pub fn captureOperationRegions(builder: *CapiAnalysisBuilder, op: mlir.MlirOperation, short_name: []const u8) AnalyzeError![]const ir.RegionId {
    const region_count: usize = @intCast(mlir.mlirOperationGetNumRegions(op));
    var ids: std.ArrayList(ir.RegionId) = .empty;
    errdefer ids.deinit(builder.allocator);
    const first_new_region = builder.regions.items.len;
    errdefer {
        freeRegionContents(builder.allocator, builder.regions.items[first_new_region..]);
        builder.regions.shrinkRetainingCapacity(first_new_region);
    }
    const split_region_blocks = std.mem.eql(u8, short_name, "while");
    var captured_region_index: usize = 0;

    var region_index: isize = 0;
    while (region_index < @as(isize, @intCast(region_count))) : (region_index += 1) {
        const region = mlir.mlirOperationGetRegion(op, region_index);
        var block = mlir.mlirRegionGetFirstBlock(region);
        while (!mlir.mlirBlockIsNull(block)) : (block = mlir.mlirBlockGetNextInRegion(block)) {
            const argument_descriptors = try descriptorListFromBlockArguments(builder.allocator, block);
            errdefer freeDescriptorList(builder.allocator, argument_descriptors);
            const captured_block = try captureRegionBlock(builder.allocator, block);
            errdefer freeCapturedRegionBlock(builder.allocator, captured_block);
            const return_descriptors = try descriptorListFromTerminatorOperands(builder.allocator, block);
            errdefer freeDescriptorList(builder.allocator, return_descriptors);
            const terminator_operand_descriptors = try cloneDescriptorList(builder.allocator, return_descriptors);
            errdefer freeDescriptorList(builder.allocator, terminator_operand_descriptors);
            const id: ir.RegionId = .{ .index = @intCast(builder.regions.items.len) };
            try ids.append(builder.allocator, id);
            try builder.regions.append(builder.allocator, .{
                .id = id,
                .parent_instruction_index = builder.ops.items.len,
                .kind = regionKindForOperation(short_name, captured_region_index),
                .values = captured_block.values,
                .argument_descriptors = argument_descriptors,
                .instructions = captured_block.instructions,
                .return_descriptors = return_descriptors,
                .terminator_operands = captured_block.terminator_operands,
                .terminator_operand_descriptors = terminator_operand_descriptors,
            });
            captured_region_index += 1;
            if (!split_region_blocks) break;
        }
    }
    return try ids.toOwnedSlice(builder.allocator);
}
