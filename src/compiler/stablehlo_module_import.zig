const std = @import("std");
const mlir = @import("c");
const ir = @import("src/compiler/ir");
const model = @import("compiler_model.zig");
const analysis = @import("stablehlo_analysis_builder.zig");
const decode = @import("stablehlo_decode.zig");
const diagnostic = @import("stablehlo_diagnostic.zig");
const op_import = @import("stablehlo_operation_import.zig");
const sharding = @import("stablehlo_sharding.zig");
const topk = @import("stablehlo_topk_import.zig");
const value_import = @import("stablehlo_value_import.zig");
const mlir_session = @import("mlir_session.zig");
const memory = @import("plan_memory.zig");
const plan_instruction = @import("plan_instruction.zig");

const AnalyzeError = model.AnalyzeError;
const Dialect = model.Dialect;
const ModuleAnalysis = model.ModuleAnalysis;
const ShardingMetadata = model.ShardingMetadata;
const SourceLoc = model.SourceLoc;
const ValueRole = model.ValueRole;
const CapiAnalysisBuilder = analysis.CapiAnalysisBuilder;
const attributeUsesShardy = mlir_session.attributeUsesShardy;
const addDialect = decode.addDialect;
const getOperationAttribute = mlir_session.getOperationAttribute;
const isEntryFunction = decode.isEntryFunction;
const mlirStringSlice = mlir_session.mlirStringSlice;
const namedAttributeName = mlir_session.namedAttributeName;
const operationName = mlir_session.operationName;
const freeRegions = memory.freeRegions;
const MlirSession = mlir_session.MlirSession;

fn valueShardingMetadata(builder: *CapiAnalysisBuilder, value: mlir.MlirValue) AnalyzeError!?ShardingMetadata {
    if (mlir.mlirValueIsNull(value)) return null;
    if (mlir.mlirValueIsABlockArgument(value)) {
        const arg_index = @as(usize, @intCast(mlir.mlirBlockArgumentGetArgNumber(value)));
        if (arg_index < builder.parameter_shardings.items.len) {
            return try sharding.copyShardingMetadata(builder.allocator, builder.parameter_shardings.items[arg_index]);
        }
        return null;
    }
    if (mlir.mlirValueIsAOpResult(value)) {
        const owner = mlir.mlirOpResultGetOwner(value);
        const result_index = @as(usize, @intCast(mlir.mlirOpResultGetResultNumber(value)));
        return try sharding.metadataFromShardingAttribute(builder.allocator, getOperationAttribute(owner, "sdy.sharding"), result_index);
    }
    return null;
}

fn analyzeFunctionFromCapi(builder: *CapiAnalysisBuilder, op: mlir.MlirOperation) AnalyzeError!void {
    builder.saw_program_body = true;
    try decode.addDialect(&builder.dialects, builder.allocator, .func);

    var num_inputs: usize = 0;
    var num_results: usize = 0;
    const function_type_attr = getOperationAttribute(op, "function_type");
    if (!mlir.mlirAttributeIsNull(function_type_attr) and mlir.mlirAttributeIsAType(function_type_attr)) {
        const function_type = mlir.mlirTypeAttrGetValue(function_type_attr);
        if (mlir.mlirTypeIsAFunction(function_type)) {
            num_inputs = @intCast(mlir.mlirFunctionTypeGetNumInputs(function_type));
            num_results = @intCast(mlir.mlirFunctionTypeGetNumResults(function_type));
        }
    }

    if (mlir.mlirOperationGetNumRegions(op) > 0) {
        const entry = mlir.mlirRegionGetFirstBlock(mlir.mlirOperationGetRegion(op, 0));
        if (!mlir.mlirBlockIsNull(entry)) {
            num_inputs = @max(num_inputs, @as(usize, @intCast(mlir.mlirBlockGetNumArguments(entry))));
            const terminator = mlir.mlirBlockGetTerminator(entry);
            if (!mlir.mlirOperationIsNull(terminator) and std.mem.eql(u8, operationName(terminator), "func.return")) {
                num_results = @max(num_results, @as(usize, @intCast(mlir.mlirOperationGetNumOperands(terminator))));
            }
        }
    }

    builder.num_parameters = @max(builder.num_parameters, num_inputs);
    builder.num_outputs = @max(builder.num_outputs, num_results);
    try builder.ensureParameterDescriptors(builder.num_parameters);
    try builder.ensureShardings(&builder.parameter_shardings, builder.num_parameters);
    try builder.ensureShardings(&builder.output_shardings, builder.num_outputs);

    const function_type_attr_for_params = getOperationAttribute(op, "function_type");
    if (!mlir.mlirAttributeIsNull(function_type_attr_for_params) and mlir.mlirAttributeIsAType(function_type_attr_for_params)) {
        const function_type = mlir.mlirTypeAttrGetValue(function_type_attr_for_params);
        if (mlir.mlirTypeIsAFunction(function_type)) {
            const typed_inputs = @min(num_inputs, @as(usize, @intCast(mlir.mlirFunctionTypeGetNumInputs(function_type))));
            var input_index: usize = 0;
            while (input_index < typed_inputs) : (input_index += 1) {
                builder.replaceParameterDescriptor(input_index, try decode.descriptorFromType(builder.allocator, mlir.mlirFunctionTypeGetInput(function_type, @intCast(input_index))));
            }
        }
    }
    if (mlir.mlirOperationGetNumRegions(op) > 0) {
        const entry = mlir.mlirRegionGetFirstBlock(mlir.mlirOperationGetRegion(op, 0));
        if (!mlir.mlirBlockIsNull(entry)) {
            const block_args = @min(num_inputs, @as(usize, @intCast(mlir.mlirBlockGetNumArguments(entry))));
            var arg_index_for_type: usize = 0;
            while (arg_index_for_type < block_args) : (arg_index_for_type += 1) {
                builder.replaceParameterDescriptor(arg_index_for_type, try decode.descriptorFromType(builder.allocator, mlir.mlirValueGetType(mlir.mlirBlockGetArgument(entry, @intCast(arg_index_for_type)))));
            }
            var arg_index_for_value: usize = 0;
            while (arg_index_for_value < block_args) : (arg_index_for_value += 1) {
                const block_arg = mlir.mlirBlockGetArgument(entry, @intCast(arg_index_for_value));
                const descriptor = builder.parameter_descriptors.items[arg_index_for_value];
                const parameter_id = try builder.registerValue(
                    block_arg,
                    .parameter,
                    plan_instruction.makeDescriptor(try builder.allocator.dupe(i64, descriptor.dims), descriptor.element_type),
                );
                try analysis.appendValueParameterAlias(builder, parameter_id, @intCast(arg_index_for_value));
            }
        }
    }

    const arg_attrs = getOperationAttribute(op, "arg_attrs");
    var arg_index: usize = 0;
    while (arg_index < num_inputs) : (arg_index += 1) {
        if (try sharding.metadataFromArrayDictionarySharding(builder.allocator, arg_attrs, arg_index)) |metadata| {
            builder.replaceMetadata(&builder.parameter_shardings, arg_index, metadata);
        }
        if (sharding.aliasingOutputFromArrayDictionary(arg_attrs, arg_index)) |output_index| {
            try analysis.appendOutputAlias(builder, output_index, @intCast(arg_index), .donation);
        }
    }

    const res_attrs = getOperationAttribute(op, "res_attrs");
    var result_index: usize = 0;
    while (result_index < num_results) : (result_index += 1) {
        if (try sharding.metadataFromArrayDictionarySharding(builder.allocator, res_attrs, result_index)) |metadata| {
            builder.replaceMetadata(&builder.output_shardings, result_index, metadata);
        }
    }

    if (mlir.mlirOperationGetNumRegions(op) > 0) {
        const entry = mlir.mlirRegionGetFirstBlock(mlir.mlirOperationGetRegion(op, 0));
        if (!mlir.mlirBlockIsNull(entry)) {
            const terminator = mlir.mlirBlockGetTerminator(entry);
            if (!mlir.mlirOperationIsNull(terminator) and std.mem.eql(u8, operationName(terminator), "func.return")) {
                const operand_count = mlir.mlirOperationGetNumOperands(terminator);
                var operand_index: isize = 0;
                while (operand_index < operand_count) : (operand_index += 1) {
                    const output_index = @as(usize, @intCast(operand_index));
                    if (output_index >= builder.output_shardings.items.len) break;
                    if (try valueShardingMetadata(builder, mlir.mlirOperationGetOperand(terminator, operand_index))) |metadata| {
                        builder.replaceMetadata(&builder.output_shardings, output_index, metadata);
                    }
                }
            }
        }
    }
}

fn inspectOperationAttributesFromCapi(builder: *CapiAnalysisBuilder, op: mlir.MlirOperation, loc: SourceLoc) AnalyzeError!void {
    _ = loc;
    const n_attrs = mlir.mlirOperationGetNumAttributes(op);
    var attr_index: isize = 0;
    while (attr_index < n_attrs) : (attr_index += 1) {
        const named = mlir.mlirOperationGetAttribute(op, attr_index);
        if (std.mem.eql(u8, namedAttributeName(named), "mhlo.sharding")) continue;
        if (std.mem.startsWith(u8, namedAttributeName(named), "sdy.") or attributeUsesShardy(named.attribute)) {
            builder.saw_shardy = true;
            try decode.addDialect(&builder.dialects, builder.allocator, .sdy);
        }
    }
}

fn appendFunctionReturnIds(builder: *CapiAnalysisBuilder, op: mlir.MlirOperation) AnalyzeError!void {
    if (mlir.mlirOperationGetNumRegions(op) == 0) return;
    const entry = mlir.mlirRegionGetFirstBlock(mlir.mlirOperationGetRegion(op, 0));
    if (mlir.mlirBlockIsNull(entry)) return;
    const terminator = mlir.mlirBlockGetTerminator(entry);
    if (mlir.mlirOperationIsNull(terminator) or !std.mem.eql(u8, operationName(terminator), "func.return")) return;

    const operand_count = mlir.mlirOperationGetNumOperands(terminator);
    builder.output_ids.clearRetainingCapacity();
    var operand_index: isize = 0;
    while (operand_index < operand_count) : (operand_index += 1) {
        const return_operand = mlir.mlirOperationGetOperand(terminator, operand_index);
        const id = builder.lookupValue(return_operand) orelse {
            const loc = decode.mlirLocationLineColumn(mlir.mlirOperationGetLocation(terminator));
            var owner_name: []const u8 = "<none>";
            if (mlir.mlirValueIsAOpResult(return_operand)) owner_name = operationName(mlir.mlirOpResultGetOwner(return_operand));
            try builder.diagnostic_writer.print(
                "invalid StableHLO module: loc={d}:{d} op=func.return operand={d} owner={s} detail=\"return operand does not reference a lowered value\" feature=value-graph",
                .{ loc.line, loc.column, operand_index, owner_name },
            );
            return error.InvalidStablehloModule;
        };
        try builder.output_ids.append(builder.allocator, id);
        if (analysis.parameterAliasForValue(builder, id)) |parameter_index| {
            const output_index: u32 = @intCast(operand_index);
            const role = if (id.index < builder.values.items.len) builder.values.items[id.index].role else ValueRole.output;
            if (role == .parameter) {
                analysis.promoteExistingOutputAlias(builder, output_index, parameter_index, .identity);
            } else {
                try analysis.appendOutputAlias(builder, output_index, parameter_index, .donation);
            }
        }
    }
}

fn visitOperationFromCapi(builder: *CapiAnalysisBuilder, op: mlir.MlirOperation) AnalyzeError!void {
    const name = operationName(op);
    const loc = decode.mlirLocationLineColumn(mlir.mlirOperationGetLocation(op));
    try inspectOperationAttributesFromCapi(builder, op, loc);

    var recurse_children = true;
    if (std.mem.eql(u8, name, "func.func")) {
        const visibility = getOperationAttribute(op, "sym_visibility");
        if (!isEntryFunction(op) or
            (!mlir.mlirAttributeIsNull(visibility) and mlir.mlirAttributeIsAString(visibility) and std.mem.eql(u8, mlirStringSlice(mlir.mlirStringAttrGetValue(visibility)), "private")))
        {
            recurse_children = false;
        } else {
            try analyzeFunctionFromCapi(builder, op);
        }
    } else if (std.mem.startsWith(u8, name, "stablehlo.")) {
        if (std.mem.eql(u8, name, "stablehlo.composite")) {
            if (try topk.analyzeCompositeTopKOperationFromCapi(op_import.analyzeStablehloOperationFromCapi, builder, op)) {
                recurse_children = false;
            } else {
                builder.saw_program_body = true;
                try addDialect(&builder.dialects, builder.allocator, .stablehlo);
                try value_import.aliasFirstRegionBlockArgumentsToOperands(op_import.analyzeStablehloOperationFromCapi, builder, op);
            }
        } else {
            try op_import.analyzeStablehloOperationFromCapi(builder, op, name);
            recurse_children = false;
        }
    } else if (std.mem.startsWith(u8, name, "chlo.")) {
        if (std.mem.eql(u8, name, "chlo.top_k")) {
            try topk.analyzeChloTopKOperationFromCapi(op_import.analyzeStablehloOperationFromCapi, builder, op);
            recurse_children = false;
        } else {
            try diagnostic.writeSimpleDiagnostic(builder.diagnostic_writer, "unsupported op", loc.line, loc.column, "CHLO legalization is not wired yet", "chlo-legalization");
            return error.UnsupportedOp;
        }
    } else if (std.mem.startsWith(u8, name, "shape.")) {
        try diagnostic.writeSimpleDiagnostic(builder.diagnostic_writer, "unsupported op", loc.line, loc.column, "shape dialect interop is not wired yet", "shape-legalization");
        return error.UnsupportedOp;
    } else if (std.mem.startsWith(u8, name, "sdy.")) {
        builder.saw_shardy = true;
        try decode.addDialect(&builder.dialects, builder.allocator, .sdy);
        const short_name = name["sdy.".len..];
        if (std.mem.eql(u8, short_name, "manual_computation")) {
            builder.saw_manual_computation = true;
            builder.manual_line = loc.line;
            builder.manual_column = loc.column;
        } else if (std.mem.eql(u8, short_name, "constant")) {
            try op_import.analyzeStablehloOperationFromCapi(builder, op, "stablehlo.constant");
            recurse_children = false;
        } else if (std.mem.eql(u8, short_name, "reshard")) {
            try value_import.aliasOperationResultsToOperands(op_import.analyzeStablehloOperationFromCapi, builder, op);
            recurse_children = false;
        } else if (std.mem.eql(u8, short_name, "return")) {
            builder.saw_sdy_return = true;
        } else if (!std.mem.eql(u8, short_name, "mesh") and
            !std.mem.eql(u8, short_name, "sharding") and
            !std.mem.eql(u8, short_name, "sharding_per_value"))
        {
            try diagnostic.writeSimpleDiagnostic(builder.diagnostic_writer, "unsupported sharding", loc.line, loc.column, short_name, "shardy-construct");
            return error.UnsupportedSharding;
        }
    }

    const n_regions = mlir.mlirOperationGetNumRegions(op);
    if (recurse_children) {
        var region_index: isize = 0;
        while (region_index < n_regions) : (region_index += 1) {
            var block = mlir.mlirRegionGetFirstBlock(mlir.mlirOperationGetRegion(op, region_index));
            while (!mlir.mlirBlockIsNull(block)) : (block = mlir.mlirBlockGetNextInRegion(block)) {
                var child = mlir.mlirBlockGetFirstOperation(block);
                while (!mlir.mlirOperationIsNull(child)) : (child = mlir.mlirOperationGetNextInBlock(child)) {
                    try visitOperationFromCapi(builder, child);
                }
            }
        }
    }
    if (std.mem.eql(u8, name, "stablehlo.composite")) try value_import.aliasOperationResultsToFirstRegionReturn(builder, op);
    if (std.mem.eql(u8, name, "func.func") and recurse_children) try appendFunctionReturnIds(builder, op);
}

pub fn analyzeMlirSessionWithCapi(
    allocator: std.mem.Allocator,
    source: []u8,
    session: MlirSession,
    diagnostic_writer: *std.Io.Writer,
) AnalyzeError!ModuleAnalysis {
    var builder: CapiAnalysisBuilder = .{
        .allocator = allocator,
        .diagnostic_writer = diagnostic_writer,
    };
    errdefer builder.deinitPartial();

    try visitOperationFromCapi(&builder, mlir.mlirModuleGetOperation(session.module));
    if (builder.saw_manual_computation and !builder.saw_sdy_return) {
        try diagnostic.writeSimpleDiagnostic(diagnostic_writer, "unsupported sharding", builder.manual_line, builder.manual_column, "sdy.manual_computation requires sdy.return in milestone 1", "shardy-manual-computation");
        return error.InvalidManualComputation;
    }
    if (!builder.saw_program_body and !builder.saw_shardy) {
        try diagnostic_writer.writeAll("invalid StableHLO module: expected func.func, stablehlo.*, or sdy.* operation");
        return error.InvalidStablehloModule;
    }
    if (builder.dialects.items.len == 0) try addDialect(&builder.dialects, allocator, .stablehlo);
    if (builder.num_outputs == 0) {
        builder.num_outputs = 1;
        try builder.ensureShardings(&builder.output_shardings, builder.num_outputs);
    }
    if (builder.output_ids.items.len == 0 and builder.values.items.len != 0) {
        try builder.output_ids.append(builder.allocator, builder.values.items[builder.values.items.len - 1].id);
    }

    const dialects = try builder.dialects.toOwnedSlice(allocator);
    errdefer allocator.free(dialects);
    const ops = try builder.ops.toOwnedSlice(allocator);
    errdefer {
        for (ops) |op| op.deinit(allocator);
        allocator.free(ops);
    }
    const values = try builder.values.toOwnedSlice(allocator);
    errdefer {
        for (values) |value| {
            allocator.free(value.descriptor.dims);
            allocator.free(value.elements);
        }
        allocator.free(values);
    }
    const regions = try builder.regions.toOwnedSlice(allocator);
    errdefer freeRegions(allocator, regions);
    const output_ids = try builder.output_ids.toOwnedSlice(allocator);
    errdefer allocator.free(output_ids);
    const output_aliases = try builder.output_aliases.toOwnedSlice(allocator);
    errdefer allocator.free(output_aliases);
    const parameter_descriptors = try builder.parameter_descriptors.toOwnedSlice(allocator);
    errdefer {
        for (parameter_descriptors) |descriptor| allocator.free(descriptor.dims);
        allocator.free(parameter_descriptors);
    }
    const parameter_shardings = try builder.parameter_shardings.toOwnedSlice(allocator);
    errdefer {
        for (parameter_shardings) |metadata| metadata.deinit(allocator);
        allocator.free(parameter_shardings);
    }
    const output_shardings = try builder.output_shardings.toOwnedSlice(allocator);
    errdefer {
        for (output_shardings) |metadata| metadata.deinit(allocator);
        allocator.free(output_shardings);
    }
    builder.value_parameter_aliases.deinit(allocator);
    builder.value_map.deinit(allocator);

    return .{
        .allocator = allocator,
        .source = source,
        .dialects = dialects,
        .ops = ops,
        .values = values,
        .regions = regions,
        .output_ids = output_ids,
        .output_aliases = output_aliases,
        .parameter_descriptors = parameter_descriptors,
        .num_parameters = builder.num_parameters,
        .num_outputs = builder.num_outputs,
        .parameter_shardings = parameter_shardings,
        .output_shardings = output_shardings,
    };
}
