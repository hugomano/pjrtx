const std = @import("std");
const mlir = @import("c");
const ir = @import("src/compiler/ir");
const model = @import("compiler_model.zig");
const analysis = @import("stablehlo_analysis_builder.zig");
const attrs = @import("stablehlo_operation_attrs.zig");
const decode = @import("stablehlo_decode.zig");
const diagnostic = @import("stablehlo_diagnostic.zig");
const region_import = @import("stablehlo_region_import.zig");
const sharding = @import("stablehlo_sharding.zig");
const value_import = @import("stablehlo_value_import.zig");
const mlir_session = @import("mlir_session.zig");
const plan_instruction = @import("plan_instruction.zig");

const AnalyzeError = model.AnalyzeError;
const Operation = model.Operation;
const ValueRole = model.ValueRole;
const CapiAnalysisBuilder = analysis.CapiAnalysisBuilder;
const bufferTypeFromDtype = plan_instruction.bufferTypeFromDtype;
const getOperationAttribute = mlir_session.getOperationAttribute;

pub fn analyzeStablehloOperationFromCapi(builder: *CapiAnalysisBuilder, op: mlir.MlirOperation, op_name: []const u8) AnalyzeError!void {
    builder.saw_program_body = true;
    try decode.addDialect(&builder.dialects, builder.allocator, .stablehlo);

    const raw_short_name = op_name["stablehlo.".len..];
    if (std.mem.eql(u8, raw_short_name, "return")) return;
    const short_name = if (std.mem.eql(u8, raw_short_name, "reduce"))
        attrs.reduceKindFromRegion(op)
    else if (std.mem.eql(u8, raw_short_name, "reduce_window"))
        attrs.reduceWindowKindFromRegion(op)
    else
        raw_short_name;
    const is_reduce_window = std.mem.startsWith(u8, short_name, "reduce_window_");
    const is_convolution = std.mem.eql(u8, short_name, "convolution");
    const ty = decode.resultOrOperandType(op);
    const dtype = if (mlir.mlirTypeIsNull(ty)) try builder.allocator.dupe(u8, "unknown") else try decode.typeDtype(builder.allocator, ty);
    var owns_dtype = true;
    defer if (owns_dtype) builder.allocator.free(dtype);
    const dims = if (mlir.mlirTypeIsNull(ty)) try builder.allocator.dupe(i64, &.{}) else try decode.typeDims(builder.allocator, ty);
    var owns_dims = true;
    defer if (owns_dims) builder.allocator.free(dims);
    const permutation = if (std.mem.eql(u8, short_name, "transpose"))
        try decode.intListAttribute(builder.allocator, getOperationAttribute(op, "permutation"))
    else
        try builder.allocator.dupe(i64, &.{});
    var owns_permutation = true;
    defer if (owns_permutation) builder.allocator.free(permutation);
    const broadcast_dimensions = if (std.mem.eql(u8, short_name, "broadcast_in_dim"))
        try decode.intListAttribute(builder.allocator, getOperationAttribute(op, "broadcast_dimensions"))
    else
        try builder.allocator.dupe(i64, &.{});
    var owns_broadcast_dimensions = true;
    defer if (owns_broadcast_dimensions) builder.allocator.free(broadcast_dimensions);
    const start_indices = if (std.mem.eql(u8, short_name, "slice"))
        try decode.intListAttribute(builder.allocator, getOperationAttribute(op, "start_indices"))
    else
        try builder.allocator.dupe(i64, &.{});
    var owns_start_indices = true;
    defer if (owns_start_indices) builder.allocator.free(start_indices);
    const limit_indices = if (std.mem.eql(u8, short_name, "slice"))
        try decode.intListAttribute(builder.allocator, getOperationAttribute(op, "limit_indices"))
    else
        try builder.allocator.dupe(i64, &.{});
    var owns_limit_indices = true;
    defer if (owns_limit_indices) builder.allocator.free(limit_indices);
    const strides = if (std.mem.eql(u8, short_name, "slice"))
        try decode.intListAttribute(builder.allocator, getOperationAttribute(op, "strides"))
    else
        try builder.allocator.dupe(i64, &.{});
    var owns_strides = true;
    defer if (owns_strides) builder.allocator.free(strides);
    const slice_sizes = if (std.mem.eql(u8, short_name, "dynamic_slice") or std.mem.eql(u8, short_name, "gather"))
        try decode.intListAttribute(builder.allocator, getOperationAttribute(op, "slice_sizes"))
    else
        try builder.allocator.dupe(i64, &.{});
    var owns_slice_sizes = true;
    defer if (owns_slice_sizes) builder.allocator.free(slice_sizes);
    const edge_padding_low = if (std.mem.eql(u8, short_name, "pad"))
        try decode.intListAttribute(builder.allocator, getOperationAttribute(op, "edge_padding_low"))
    else
        try builder.allocator.dupe(i64, &.{});
    var owns_edge_padding_low = true;
    defer if (owns_edge_padding_low) builder.allocator.free(edge_padding_low);
    const edge_padding_high = if (std.mem.eql(u8, short_name, "pad"))
        try decode.intListAttribute(builder.allocator, getOperationAttribute(op, "edge_padding_high"))
    else
        try builder.allocator.dupe(i64, &.{});
    var owns_edge_padding_high = true;
    defer if (owns_edge_padding_high) builder.allocator.free(edge_padding_high);
    const interior_padding = if (std.mem.eql(u8, short_name, "pad"))
        try decode.intListAttribute(builder.allocator, getOperationAttribute(op, "interior_padding"))
    else
        try builder.allocator.dupe(i64, &.{});
    var owns_interior_padding = true;
    defer if (owns_interior_padding) builder.allocator.free(interior_padding);
    const window_dimensions = if (std.mem.startsWith(u8, short_name, "reduce_window_"))
        try decode.intListAttribute(builder.allocator, getOperationAttribute(op, "window_dimensions"))
    else
        try builder.allocator.dupe(i64, &.{});
    var owns_window_dimensions = true;
    defer if (owns_window_dimensions) builder.allocator.free(window_dimensions);
    const convolution_spatial_count = if (is_convolution and dims.len >= 2) dims.len - 2 else 0;
    const window_strides = if (std.mem.startsWith(u8, short_name, "reduce_window_"))
        try decode.intListAttributeOrFill(builder.allocator, getOperationAttribute(op, "window_strides"), dims.len, 1)
    else if (is_convolution)
        try decode.intListAttributeOrFill(builder.allocator, getOperationAttribute(op, "window_strides"), convolution_spatial_count, 1)
    else
        try builder.allocator.dupe(i64, &.{});
    var owns_window_strides = true;
    defer if (owns_window_strides) builder.allocator.free(window_strides);
    const base_dilations = if (std.mem.startsWith(u8, short_name, "reduce_window_"))
        try decode.intListAttributeOrFill(builder.allocator, getOperationAttribute(op, "base_dilations"), dims.len, 1)
    else if (is_convolution)
        try decode.intListAttributeOrFill(builder.allocator, getOperationAttribute(op, "lhs_dilation"), convolution_spatial_count, 1)
    else
        try builder.allocator.dupe(i64, &.{});
    var owns_base_dilations = true;
    defer if (owns_base_dilations) builder.allocator.free(base_dilations);
    const window_dilations = if (std.mem.startsWith(u8, short_name, "reduce_window_"))
        try decode.intListAttributeOrFill(builder.allocator, getOperationAttribute(op, "window_dilations"), dims.len, 1)
    else if (is_convolution)
        try decode.intListAttributeOrFill(builder.allocator, getOperationAttribute(op, "rhs_dilation"), convolution_spatial_count, 1)
    else
        try builder.allocator.dupe(i64, &.{});
    var owns_window_dilations = true;
    defer if (owns_window_dilations) builder.allocator.free(window_dilations);
    const window_reversal = if (is_convolution)
        try decode.boolListAttributeOrFill(builder.allocator, getOperationAttribute(op, "window_reversal"), convolution_spatial_count, false)
    else
        try builder.allocator.dupe(bool, &.{});
    var owns_window_reversal = true;
    defer if (owns_window_reversal) builder.allocator.free(window_reversal);
    const reduce_window_padding = if (std.mem.startsWith(u8, short_name, "reduce_window_") or is_convolution)
        try decode.intListAttribute(builder.allocator, getOperationAttribute(op, "padding"))
    else
        try builder.allocator.dupe(i64, &.{});
    const owns_reduce_window_padding = true;
    defer if (owns_reduce_window_padding) builder.allocator.free(reduce_window_padding);
    const reduce_window_padding_count = if (is_convolution) convolution_spatial_count else dims.len;
    const reduce_window_padding_low = if (std.mem.startsWith(u8, short_name, "reduce_window_") or is_convolution)
        if (reduce_window_padding.len == 0) try decode.filledIntList(builder.allocator, reduce_window_padding_count, 0) else try decode.paddingPairList(builder.allocator, reduce_window_padding, 0)
    else
        try builder.allocator.dupe(i64, &.{});
    var owns_reduce_window_padding_low = true;
    defer if (owns_reduce_window_padding_low) builder.allocator.free(reduce_window_padding_low);
    const reduce_window_padding_high = if (std.mem.startsWith(u8, short_name, "reduce_window_") or is_convolution)
        if (reduce_window_padding.len == 0) try decode.filledIntList(builder.allocator, reduce_window_padding_count, 0) else try decode.paddingPairList(builder.allocator, reduce_window_padding, 1)
    else
        try builder.allocator.dupe(i64, &.{});
    var owns_reduce_window_padding_high = true;
    defer if (owns_reduce_window_padding_high) builder.allocator.free(reduce_window_padding_high);
    const gather_dimensions = getOperationAttribute(op, "dimension_numbers");
    const offset_dims = if (std.mem.eql(u8, short_name, "gather")) try attrs.stablehloGatherDims(builder.allocator, gather_dimensions, "offset") else try builder.allocator.dupe(i64, &.{});
    var owns_offset_dims = true;
    defer if (owns_offset_dims) builder.allocator.free(offset_dims);
    const collapsed_slice_dims = if (std.mem.eql(u8, short_name, "gather")) try attrs.stablehloGatherDims(builder.allocator, gather_dimensions, "collapsed") else try builder.allocator.dupe(i64, &.{});
    var owns_collapsed_slice_dims = true;
    defer if (owns_collapsed_slice_dims) builder.allocator.free(collapsed_slice_dims);
    const operand_batching_dims = if (std.mem.eql(u8, short_name, "gather")) try attrs.stablehloGatherDims(builder.allocator, gather_dimensions, "operand_batching") else try builder.allocator.dupe(i64, &.{});
    var owns_operand_batching_dims = true;
    defer if (owns_operand_batching_dims) builder.allocator.free(operand_batching_dims);
    const start_indices_batching_dims = if (std.mem.eql(u8, short_name, "gather")) try attrs.stablehloGatherDims(builder.allocator, gather_dimensions, "start_indices_batching") else try builder.allocator.dupe(i64, &.{});
    var owns_start_indices_batching_dims = true;
    defer if (owns_start_indices_batching_dims) builder.allocator.free(start_indices_batching_dims);
    const start_index_map = if (std.mem.eql(u8, short_name, "gather")) try attrs.stablehloGatherDims(builder.allocator, gather_dimensions, "start_index_map") else try builder.allocator.dupe(i64, &.{});
    var owns_start_index_map = true;
    defer if (owns_start_index_map) builder.allocator.free(start_index_map);
    const scatter_dimensions = getOperationAttribute(op, "scatter_dimension_numbers");
    const update_window_dims = if (std.mem.eql(u8, short_name, "scatter")) try attrs.stablehloScatterDims(builder.allocator, scatter_dimensions, "update_window") else try builder.allocator.dupe(i64, &.{});
    var owns_update_window_dims = true;
    defer if (owns_update_window_dims) builder.allocator.free(update_window_dims);
    const inserted_window_dims = if (std.mem.eql(u8, short_name, "scatter")) try attrs.stablehloScatterDims(builder.allocator, scatter_dimensions, "inserted_window") else try builder.allocator.dupe(i64, &.{});
    var owns_inserted_window_dims = true;
    defer if (owns_inserted_window_dims) builder.allocator.free(inserted_window_dims);
    const input_batching_dims = if (std.mem.eql(u8, short_name, "scatter")) try attrs.stablehloScatterDims(builder.allocator, scatter_dimensions, "input_batching") else try builder.allocator.dupe(i64, &.{});
    var owns_input_batching_dims = true;
    defer if (owns_input_batching_dims) builder.allocator.free(input_batching_dims);
    const scatter_indices_batching_dims = if (std.mem.eql(u8, short_name, "scatter")) try attrs.stablehloScatterDims(builder.allocator, scatter_dimensions, "scatter_indices_batching") else try builder.allocator.dupe(i64, &.{});
    var owns_scatter_indices_batching_dims = true;
    defer if (owns_scatter_indices_batching_dims) builder.allocator.free(scatter_indices_batching_dims);
    const scatter_dims_to_operand_dims = if (std.mem.eql(u8, short_name, "scatter")) try attrs.stablehloScatterDims(builder.allocator, scatter_dimensions, "scatter_dims_to_operand") else try builder.allocator.dupe(i64, &.{});
    var owns_scatter_dims_to_operand_dims = true;
    defer if (owns_scatter_dims_to_operand_dims) builder.allocator.free(scatter_dims_to_operand_dims);
    const index_vector_dim = if (std.mem.eql(u8, short_name, "gather"))
        attrs.stablehloGatherIndexVectorDim(gather_dimensions)
    else if (std.mem.eql(u8, short_name, "scatter"))
        attrs.stablehloScatterIndexVectorDim(scatter_dimensions)
    else
        null;
    const dimension = if (std.mem.eql(u8, short_name, "concatenate") or std.mem.eql(u8, short_name, "sort"))
        decode.intAttribute(getOperationAttribute(op, "dimension"))
    else
        null;
    const iota_dimension = if (std.mem.eql(u8, short_name, "iota"))
        decode.intAttribute(getOperationAttribute(op, "iota_dimension"))
    else
        null;
    const fft_kind = if (std.mem.eql(u8, short_name, "fft"))
        decode.fftKindFromAttr(getOperationAttribute(op, "fft_type"))
    else
        null;
    const dimensions = if (std.mem.eql(u8, short_name, "reverse"))
        try decode.intListAttribute(builder.allocator, getOperationAttribute(op, "dimensions"))
    else if (std.mem.eql(u8, short_name, "fft"))
        try decode.intListAttribute(builder.allocator, getOperationAttribute(op, "fft_length"))
    else
        try builder.allocator.dupe(i64, &.{});
    var owns_dimensions = true;
    defer if (owns_dimensions) builder.allocator.free(dimensions);
    const tuple_index = if (std.mem.eql(u8, short_name, "get_tuple_element"))
        decode.intAttribute(getOperationAttribute(op, "index"))
    else
        null;
    const lower = if (std.mem.eql(u8, short_name, "cholesky"))
        decode.boolAttribute(getOperationAttribute(op, "lower"))
    else
        null;
    const triangular_left_side = if (std.mem.eql(u8, short_name, "triangular_solve"))
        decode.boolAttribute(getOperationAttribute(op, "left_side"))
    else
        null;
    const triangular_lower = if (std.mem.eql(u8, short_name, "triangular_solve"))
        decode.boolAttribute(getOperationAttribute(op, "lower"))
    else
        null;
    const triangular_unit_diagonal = if (std.mem.eql(u8, short_name, "triangular_solve"))
        decode.boolAttribute(getOperationAttribute(op, "unit_diagonal"))
    else
        null;
    const triangular_transpose = if (std.mem.eql(u8, short_name, "triangular_solve"))
        decode.triangularTransposeFromAttr(getOperationAttribute(op, "transpose_a"))
    else
        null;
    const custom_call_target = if (std.mem.eql(u8, short_name, "custom_call"))
        try decode.stringAttribute(builder.allocator, getOperationAttribute(op, "call_target_name"))
    else
        try builder.allocator.dupe(u8, &.{});
    var owns_custom_call_target = true;
    defer if (owns_custom_call_target) builder.allocator.free(custom_call_target);
    const rng_distribution = if (std.mem.eql(u8, short_name, "rng"))
        decode.rngDistributionFromAttr(getOperationAttribute(op, "rng_distribution"))
    else
        null;
    const reduce_dimensions = if (std.mem.startsWith(u8, short_name, "reduce_"))
        try decode.intListAttribute(builder.allocator, getOperationAttribute(op, "dimensions"))
    else
        try builder.allocator.dupe(i64, &.{});
    var owns_reduce_dimensions = true;
    defer if (owns_reduce_dimensions) builder.allocator.free(reduce_dimensions);
    const dot_dimensions = getOperationAttribute(op, "dot_dimension_numbers");
    const lhs_batch_dimensions = if (std.mem.eql(u8, short_name, "dot_general")) try attrs.stablehloDotDims(builder.allocator, dot_dimensions, "lhs_batch") else try builder.allocator.dupe(i64, &.{});
    var owns_lhs_batch_dimensions = true;
    defer if (owns_lhs_batch_dimensions) builder.allocator.free(lhs_batch_dimensions);
    const rhs_batch_dimensions = if (std.mem.eql(u8, short_name, "dot_general")) try attrs.stablehloDotDims(builder.allocator, dot_dimensions, "rhs_batch") else try builder.allocator.dupe(i64, &.{});
    var owns_rhs_batch_dimensions = true;
    defer if (owns_rhs_batch_dimensions) builder.allocator.free(rhs_batch_dimensions);
    const lhs_contracting_dimensions = if (std.mem.eql(u8, short_name, "dot_general")) try attrs.stablehloDotDims(builder.allocator, dot_dimensions, "lhs_contract") else try builder.allocator.dupe(i64, &.{});
    var owns_lhs_contracting_dimensions = true;
    defer if (owns_lhs_contracting_dimensions) builder.allocator.free(lhs_contracting_dimensions);
    const rhs_contracting_dimensions = if (std.mem.eql(u8, short_name, "dot_general")) try attrs.stablehloDotDims(builder.allocator, dot_dimensions, "rhs_contract") else try builder.allocator.dupe(i64, &.{});
    var owns_rhs_contracting_dimensions = true;
    defer if (owns_rhs_contracting_dimensions) builder.allocator.free(rhs_contracting_dimensions);
    const conv_dimensions = getOperationAttribute(op, "dimension_numbers");
    const input_batch_dimension = if (is_convolution) attrs.stablehloConvDim(conv_dimensions, "input_batch") else null;
    const input_feature_dimension = if (is_convolution) attrs.stablehloConvDim(conv_dimensions, "input_feature") else null;
    const input_spatial_dimensions = if (is_convolution) try attrs.stablehloConvDims(builder.allocator, conv_dimensions, "input_spatial") else try builder.allocator.dupe(i64, &.{});
    var owns_input_spatial_dimensions = true;
    defer if (owns_input_spatial_dimensions) builder.allocator.free(input_spatial_dimensions);
    const kernel_input_feature_dimension = if (is_convolution) attrs.stablehloConvDim(conv_dimensions, "kernel_input_feature") else null;
    const kernel_output_feature_dimension = if (is_convolution) attrs.stablehloConvDim(conv_dimensions, "kernel_output_feature") else null;
    const kernel_spatial_dimensions = if (is_convolution) try attrs.stablehloConvDims(builder.allocator, conv_dimensions, "kernel_spatial") else try builder.allocator.dupe(i64, &.{});
    var owns_kernel_spatial_dimensions = true;
    defer if (owns_kernel_spatial_dimensions) builder.allocator.free(kernel_spatial_dimensions);
    const output_batch_dimension = if (is_convolution) attrs.stablehloConvDim(conv_dimensions, "output_batch") else null;
    const output_feature_dimension = if (is_convolution) attrs.stablehloConvDim(conv_dimensions, "output_feature") else null;
    const output_spatial_dimensions = if (is_convolution) try attrs.stablehloConvDims(builder.allocator, conv_dimensions, "output_spatial") else try builder.allocator.dupe(i64, &.{});
    var owns_output_spatial_dimensions = true;
    defer if (owns_output_spatial_dimensions) builder.allocator.free(output_spatial_dimensions);
    const feature_group_count = if (is_convolution) decode.intAttribute(getOperationAttribute(op, "feature_group_count")) orelse 1 else null;
    const batch_group_count = if (is_convolution) decode.intAttribute(getOperationAttribute(op, "batch_group_count")) orelse 1 else null;
    const compare_direction = if (std.mem.eql(u8, short_name, "compare"))
        attrs.compareDirectionFromAttr(getOperationAttribute(op, "comparison_direction"))
    else if (std.mem.eql(u8, short_name, "sort"))
        attrs.compareDirectionFromSortRegion(op)
    else
        null;
    const scatter_update_kind = if (std.mem.eql(u8, short_name, "scatter")) attrs.scatterUpdateKindFromRegion(op) else null;
    const operand_count: usize = @intCast(mlir.mlirOperationGetNumOperands(op));
    const input_count = if (is_reduce_window)
        operand_count / 2
    else if (std.mem.startsWith(u8, short_name, "reduce_"))
        operand_count / 2
    else
        operand_count;
    const inputs = try value_import.valueIdsForOperandsLimit(analyzeStablehloOperationFromCapi, builder, op, input_count);
    var owns_inputs = true;
    defer if (owns_inputs) builder.allocator.free(inputs);
    const value_role: ValueRole = if (std.mem.eql(u8, short_name, "constant")) .constant else .instruction_result;
    const outputs = try value_import.registerResultValues(builder, op, value_role);
    var owns_outputs = true;
    defer if (owns_outputs) builder.allocator.free(outputs);
    const element_type = bufferTypeFromDtype(dtype);
    const literal = if (std.mem.eql(u8, short_name, "constant"))
        try attrs.denseLiteralBytes(builder.allocator, getOperationAttribute(op, "value"), element_type, dims)
    else
        try builder.allocator.dupe(u8, &.{});
    var owns_literal = true;
    defer if (owns_literal) builder.allocator.free(literal);
    const sharding_label = try sharding.operationShardingLabel(builder.allocator, op);
    var owns_sharding = true;
    defer if (owns_sharding) builder.allocator.free(sharding_label);
    const loc = decode.mlirLocationLineColumn(mlir.mlirOperationGetLocation(op));
    const owned_name = try builder.allocator.dupe(u8, short_name);
    var owns_name = true;
    defer if (owns_name) builder.allocator.free(owned_name);
    const empty_region_ids = try builder.allocator.dupe(ir.RegionId, &.{});
    var owns_region_ids = true;
    defer if (owns_region_ids) builder.allocator.free(empty_region_ids);

    var analyzed: Operation = .{
        .name = owned_name,
        .line = loc.line,
        .column = loc.column,
        .inputs = inputs,
        .outputs = outputs,
        .region_ids = empty_region_ids,
        .dtype = dtype,
        .rank = if (mlir.mlirTypeIsNull(ty)) null else decode.typeRank(ty),
        .dims = dims,
        .permutation = permutation,
        .broadcast_dimensions = broadcast_dimensions,
        .start_indices = start_indices,
        .limit_indices = limit_indices,
        .strides = strides,
        .slice_sizes = slice_sizes,
        .edge_padding_low = if (is_reduce_window or is_convolution) reduce_window_padding_low else edge_padding_low,
        .edge_padding_high = if (is_reduce_window or is_convolution) reduce_window_padding_high else edge_padding_high,
        .interior_padding = interior_padding,
        .window_dimensions = window_dimensions,
        .window_strides = window_strides,
        .base_dilations = base_dilations,
        .window_dilations = window_dilations,
        .window_reversal = window_reversal,
        .offset_dims = offset_dims,
        .collapsed_slice_dims = collapsed_slice_dims,
        .operand_batching_dims = operand_batching_dims,
        .start_indices_batching_dims = start_indices_batching_dims,
        .start_index_map = start_index_map,
        .update_window_dims = update_window_dims,
        .inserted_window_dims = inserted_window_dims,
        .input_batching_dims = input_batching_dims,
        .scatter_indices_batching_dims = scatter_indices_batching_dims,
        .scatter_dims_to_operand_dims = scatter_dims_to_operand_dims,
        .index_vector_dim = index_vector_dim,
        .scatter_update_kind = scatter_update_kind,
        .dimension = dimension,
        .iota_dimension = iota_dimension,
        .fft_kind = fft_kind,
        .dimensions = dimensions,
        .tuple_index = tuple_index,
        .lower = lower,
        .triangular_left_side = triangular_left_side,
        .triangular_lower = triangular_lower,
        .triangular_unit_diagonal = triangular_unit_diagonal,
        .triangular_transpose = triangular_transpose,
        .custom_call_target = custom_call_target,
        .rng_distribution = rng_distribution,
        .reduce_dimensions = reduce_dimensions,
        .lhs_batch_dimensions = lhs_batch_dimensions,
        .rhs_batch_dimensions = rhs_batch_dimensions,
        .lhs_contracting_dimensions = lhs_contracting_dimensions,
        .rhs_contracting_dimensions = rhs_contracting_dimensions,
        .input_batch_dimension = input_batch_dimension,
        .input_feature_dimension = input_feature_dimension,
        .input_spatial_dimensions = input_spatial_dimensions,
        .kernel_input_feature_dimension = kernel_input_feature_dimension,
        .kernel_output_feature_dimension = kernel_output_feature_dimension,
        .kernel_spatial_dimensions = kernel_spatial_dimensions,
        .output_batch_dimension = output_batch_dimension,
        .output_feature_dimension = output_feature_dimension,
        .output_spatial_dimensions = output_spatial_dimensions,
        .feature_group_count = feature_group_count,
        .batch_group_count = batch_group_count,
        .compare_direction = compare_direction,
        .literal = literal,
        .sharding = sharding_label,
    };
    var analyzed_owned = true;
    errdefer if (analyzed_owned) analyzed.deinit(builder.allocator);
    owns_inputs = false;
    owns_outputs = false;
    owns_dtype = false;
    owns_dims = false;
    owns_permutation = false;
    owns_broadcast_dimensions = false;
    owns_start_indices = false;
    owns_limit_indices = false;
    owns_strides = false;
    owns_slice_sizes = false;
    if (is_reduce_window or is_convolution) {
        owns_reduce_window_padding_low = false;
        owns_reduce_window_padding_high = false;
    } else {
        owns_edge_padding_low = false;
        owns_edge_padding_high = false;
    }
    owns_interior_padding = false;
    owns_window_dimensions = false;
    owns_window_strides = false;
    owns_base_dilations = false;
    owns_window_dilations = false;
    owns_window_reversal = false;
    owns_offset_dims = false;
    owns_collapsed_slice_dims = false;
    owns_operand_batching_dims = false;
    owns_start_indices_batching_dims = false;
    owns_start_index_map = false;
    owns_update_window_dims = false;
    owns_inserted_window_dims = false;
    owns_input_batching_dims = false;
    owns_scatter_indices_batching_dims = false;
    owns_scatter_dims_to_operand_dims = false;
    owns_dimensions = false;
    owns_custom_call_target = false;
    owns_reduce_dimensions = false;
    owns_lhs_batch_dimensions = false;
    owns_rhs_batch_dimensions = false;
    owns_lhs_contracting_dimensions = false;
    owns_rhs_contracting_dimensions = false;
    owns_input_spatial_dimensions = false;
    owns_kernel_spatial_dimensions = false;
    owns_output_spatial_dimensions = false;
    owns_literal = false;
    owns_sharding = false;
    owns_name = false;
    owns_region_ids = false;
    if (!decode.stablehloOpSupported(raw_short_name) or std.mem.eql(u8, short_name, "reduce") or std.mem.eql(u8, short_name, "reduce_window")) {
        try diagnostic.writeOpDiagnostic(builder.diagnostic_writer, "unsupported op", analyzed, "stablehlo-op");
        return error.UnsupportedOp;
    }
    const region_ids = try region_import.captureOperationRegions(builder, op, short_name);
    builder.allocator.free(analyzed.region_ids);
    owns_region_ids = false;
    analyzed.region_ids = region_ids;
    if (std.mem.eql(u8, short_name, "tuple")) {
        for (outputs) |output| try builder.setValueStorage(output, .tuple, inputs);
    } else if (std.mem.eql(u8, short_name, "complex")) {
        for (outputs) |output| try builder.setValueStorage(output, .complex_pair, inputs);
    }
    try builder.ops.append(builder.allocator, analyzed);
    if (std.mem.eql(u8, short_name, "custom_call")) {
        try value_import.registerCustomCallOutputOperandAliases(builder, op, inputs, outputs);
    }
    analyzed_owned = false;
}
