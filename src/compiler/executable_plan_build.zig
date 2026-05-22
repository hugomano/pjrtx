const std = @import("std");
const ir = @import("src/compiler/ir");
const model = @import("compiler_model.zig");
const compile_options = @import("compile_options.zig");
const plan_instruction_mod = @import("plan_instruction.zig");
const memory = @import("plan_memory.zig");

const CompileOptions = compile_options.CompileOptions;
const ModuleAnalysis = model.ModuleAnalysis;
const Operation = model.Operation;
const ShardingKind = model.ShardingKind;
const ShardingPlan = model.ShardingPlan;
const Value = model.Value;
const ValueId = model.ValueId;
const ValueRole = model.ValueRole;
const PlanInstruction = model.PlanInstruction;
const PlanInstructionKind = model.PlanInstructionKind;
const ExecutablePlan = model.ExecutablePlan;
const makeDescriptor = plan_instruction_mod.makeDescriptor;
const makeBootstrapValues = plan_instruction_mod.makeBootstrapValues;
const instructionKindFromStablehlo = plan_instruction_mod.instructionKindFromStablehlo;
const isUnaryKind = plan_instruction_mod.isUnaryKind;
const isBinaryKind = plan_instruction_mod.isBinaryKind;
const freeInstructions = memory.freeInstructions;
const cloneValues = memory.cloneValues;
const cloneRegions = memory.cloneRegions;

pub fn makeReplicatedPlan(
    allocator: std.mem.Allocator,
    options: CompileOptions,
    num_parameters: usize,
    num_outputs: usize,
) !ExecutablePlan {
    const assignment = if (options.device_assignment.len == 0) blk: {
        const generated = try allocator.alloc(i32, options.numDevices());
        for (generated, 0..) |*id, i| id.* = @intCast(i);
        break :blk generated;
    } else try allocator.dupe(i32, options.device_assignment);
    errdefer allocator.free(assignment);

    var owned_options = options;
    owned_options.device_assignment = assignment;

    const parameter_shardings = try allocator.alloc(ShardingPlan, num_parameters);
    errdefer allocator.free(parameter_shardings);
    const output_shardings = try allocator.alloc(ShardingPlan, num_outputs);
    errdefer allocator.free(output_shardings);

    const parameter_descriptors = try allocator.alloc(ir.BufferDescriptor, num_parameters);
    defer allocator.free(parameter_descriptors);
    @memset(parameter_descriptors, makeDescriptor(&.{}, .invalid));
    const values = try makeBootstrapValues(allocator, parameter_descriptors, &.{}, 1);
    errdefer {
        for (values) |value| {
            allocator.free(value.descriptor.dims);
            allocator.free(value.elements);
        }
        allocator.free(values);
    }
    const instructions = try makeCopyArg0Instructions(allocator, num_parameters);
    errdefer freeInstructions(allocator, instructions);
    const output_ids = try allocator.alloc(ValueId, num_outputs);
    errdefer allocator.free(output_ids);
    for (output_ids, 0..) |*id, index| id.* = .{ .index = @intCast(num_parameters + @min(index, instructions.len - 1)) };

    var initialized_parameters: usize = 0;
    errdefer for (parameter_shardings[0..initialized_parameters]) |plan| {
        allocator.free(plan.mesh_name);
        allocator.free(plan.device_assignment);
    };
    for (parameter_shardings) |*plan| {
        plan.* = try makeShardingPlan(allocator, .replicated, "pjrtx_mesh", assignment);
        initialized_parameters += 1;
    }

    var initialized_outputs: usize = 0;
    errdefer for (output_shardings[0..initialized_outputs]) |plan| {
        allocator.free(plan.mesh_name);
        allocator.free(plan.device_assignment);
    };
    for (output_shardings) |*plan| {
        plan.* = try makeShardingPlan(allocator, .replicated, "pjrtx_mesh", assignment);
        initialized_outputs += 1;
    }

    return .{
        .allocator = allocator,
        .options = owned_options,
        .values = values,
        .parameter_shardings = parameter_shardings,
        .output_shardings = output_shardings,
        .output_ids = output_ids,
        .instructions = instructions,
    };
}

pub fn makeShardingPlan(
    allocator: std.mem.Allocator,
    kind: ShardingKind,
    mesh_name: []const u8,
    assignment: []const i32,
) !ShardingPlan {
    const owned_mesh_name = try allocator.dupe(u8, mesh_name);
    errdefer allocator.free(owned_mesh_name);
    return .{
        .kind = kind,
        .mesh_name = owned_mesh_name,
        .device_assignment = try allocator.dupe(i32, assignment),
    };
}

pub fn replaceShardingPlan(
    allocator: std.mem.Allocator,
    target: *ShardingPlan,
    kind: ShardingKind,
    mesh_name: []const u8,
    assignment: []const i32,
) !void {
    const replacement = try makeShardingPlan(allocator, kind, mesh_name, assignment);
    allocator.free(target.mesh_name);
    allocator.free(target.device_assignment);
    target.* = replacement;
}

pub fn applyAnalysisShardingMetadataToPlan(
    allocator: std.mem.Allocator,
    plan: *ExecutablePlan,
    analysis: ModuleAnalysis,
) !void {
    for (analysis.parameter_shardings, 0..) |metadata, index| {
        if (index >= plan.parameter_shardings.len) break;
        try replaceShardingPlan(allocator, &plan.parameter_shardings[index], metadata.kind, metadata.mesh_name, plan.options.device_assignment);
    }
    for (analysis.output_shardings, 0..) |metadata, index| {
        if (index >= plan.output_shardings.len) break;
        try replaceShardingPlan(allocator, &plan.output_shardings[index], metadata.kind, metadata.mesh_name, plan.options.device_assignment);
    }
}

pub fn donatedParametersFromOutputAliases(allocator: std.mem.Allocator, output_aliases: []const ir.OutputAlias) ![]const u32 {
    var donated: std.ArrayList(u32) = .empty;
    defer donated.deinit(allocator);
    for (output_aliases) |alias| {
        for (donated.items) |existing| {
            if (existing == alias.parameter_index) break;
        } else {
            try donated.append(allocator, alias.parameter_index);
        }
    }
    return try donated.toOwnedSlice(allocator);
}

pub fn makeExecutablePlan(
    allocator: std.mem.Allocator,
    options: CompileOptions,
    analysis: ModuleAnalysis,
) !ExecutablePlan {
    var plan = try makeReplicatedPlan(allocator, options, analysis.num_parameters, analysis.num_outputs);
    errdefer plan.deinit();
    if (options.use_shardy_partitioner) {
        try applyAnalysisShardingMetadataToPlan(allocator, &plan, analysis);
    }
    plan.output_aliases = try allocator.dupe(ir.OutputAlias, analysis.output_aliases);
    plan.donated_parameter_indices = try donatedParametersFromOutputAliases(allocator, plan.output_aliases);
    freeInstructions(allocator, plan.instructions);
    plan.instructions = &.{};
    plan.instructions = try lowerAnalysisOpsToPlan(allocator, analysis.ops, analysis.num_parameters);
    plan.regions = try cloneRegions(allocator, analysis.regions);
    allocator.free(plan.output_ids);
    plan.output_ids = try allocator.dupe(ValueId, analysis.output_ids);
    for (plan.values) |value| {
        allocator.free(value.descriptor.dims);
        allocator.free(value.elements);
    }
    allocator.free(plan.values);
    plan.values = &.{};
    plan.values = try cloneValues(allocator, analysis.values);
    plan.instructions = try pruneDeadPlanInstructions(allocator, plan.instructions, plan.output_ids, plan.values.len, plan.regions);
    return plan;
}


pub fn isReduceKind(kind: PlanInstructionKind) bool {
    return switch (kind) {
        .reduce_sum, .reduce_max, .reduce_min, .reduce_and, .reduce_or => true,
        else => false,
    };
}

pub fn isReduceWindowKind(kind: PlanInstructionKind) bool {
    return switch (kind) {
        .reduce_window_sum, .reduce_window_max => true,
        else => false,
    };
}

pub fn instructionInputs(allocator: std.mem.Allocator, kind: PlanInstructionKind, op: Operation) ![]const ValueId {
    if (isReduceKind(kind) and op.inputs.len >= 1) {
        const reduce_input_count = if (op.outputs.len != 0 and op.inputs.len >= op.outputs.len) op.outputs.len else 1;
        return allocator.dupe(ValueId, op.inputs[0..reduce_input_count]);
    }
    if (kind == .rng and op.inputs.len >= 2) return allocator.dupe(ValueId, op.inputs[0..2]);
    if (op.inputs.len != 0 or kind == .constant) return allocator.dupe(ValueId, op.inputs);
    return switch (kind) {
        .constant, .iota, .partition_id => &.{},
        .select, .clamp => allocator.dupe(ValueId, &.{ .{ .index = 0 }, .{ .index = 1 }, .{ .index = 2 } }),
        .rng => allocator.dupe(ValueId, &.{ .{ .index = 0 }, .{ .index = 1 } }),
        .custom_call, .rng_bit_generator, .scatter, .tuple, .while_ => allocator.dupe(ValueId, op.inputs),
        .optimization_barrier => allocator.dupe(ValueId, op.inputs),
        else => if (isUnaryKind(kind))
            allocator.dupe(ValueId, &.{.{ .index = 0 }})
        else if (isBinaryKind(kind))
            allocator.dupe(ValueId, &.{ .{ .index = 0 }, .{ .index = 1 } })
        else
            &.{},
    };
}

pub fn makeCopyArg0Instructions(allocator: std.mem.Allocator, num_parameters: usize) ![]PlanInstruction {
    const output: ValueId = .{ .index = @intCast(num_parameters) };
    const inputs = try allocator.dupe(ValueId, &.{.{ .index = 0 }});
    errdefer allocator.free(inputs);
    const outputs = try allocator.dupe(ValueId, &.{output});
    errdefer allocator.free(outputs);
    return allocator.dupe(PlanInstruction, &.{.{
        .kind = .copy_arg0,
        .inputs = inputs,
        .outputs = outputs,
    }});
}

pub fn makePlanInstruction(
    allocator: std.mem.Allocator,
    op: Operation,
    kind: PlanInstructionKind,
) !PlanInstruction {
    const inputs = try instructionInputs(allocator, kind, op);
    errdefer if (inputs.len != 0) allocator.free(inputs);
    const outputs = try allocator.dupe(ValueId, op.outputs);
    errdefer allocator.free(outputs);
    const region_ids = try allocator.dupe(ir.RegionId, op.region_ids);
    errdefer allocator.free(region_ids);
    const dims = if (kind != .constant and kind != .unsupported) try allocator.dupe(i64, op.dims) else null;
    errdefer if (dims) |owned| allocator.free(owned);
    const permutation = if (kind == .transpose) try allocator.dupe(i64, op.permutation) else null;
    errdefer if (permutation) |owned| allocator.free(owned);
    const broadcast_dimensions = if (kind == .broadcast_in_dim) try allocator.dupe(i64, op.broadcast_dimensions) else null;
    errdefer if (broadcast_dimensions) |owned| allocator.free(owned);
    const start_indices = if (kind == .slice) try allocator.dupe(i64, op.start_indices) else null;
    errdefer if (start_indices) |owned| allocator.free(owned);
    const limit_indices = if (kind == .slice) try allocator.dupe(i64, op.limit_indices) else null;
    errdefer if (limit_indices) |owned| allocator.free(owned);
    const strides = if (kind == .slice) try allocator.dupe(i64, op.strides) else null;
    errdefer if (strides) |owned| allocator.free(owned);
    const slice_sizes = if (kind == .dynamic_slice or kind == .gather) try allocator.dupe(i64, op.slice_sizes) else null;
    errdefer if (slice_sizes) |owned| allocator.free(owned);
    const edge_padding_low = if (kind == .pad or isReduceWindowKind(kind) or kind == .convolution) try allocator.dupe(i64, op.edge_padding_low) else null;
    errdefer if (edge_padding_low) |owned| allocator.free(owned);
    const edge_padding_high = if (kind == .pad or isReduceWindowKind(kind) or kind == .convolution) try allocator.dupe(i64, op.edge_padding_high) else null;
    errdefer if (edge_padding_high) |owned| allocator.free(owned);
    const interior_padding = if (kind == .pad) try allocator.dupe(i64, op.interior_padding) else null;
    errdefer if (interior_padding) |owned| allocator.free(owned);
    const window_dimensions = if (isReduceWindowKind(kind)) try allocator.dupe(i64, op.window_dimensions) else null;
    errdefer if (window_dimensions) |owned| allocator.free(owned);
    const window_strides = if (isReduceWindowKind(kind) or kind == .convolution) try allocator.dupe(i64, op.window_strides) else null;
    errdefer if (window_strides) |owned| allocator.free(owned);
    const base_dilations = if (isReduceWindowKind(kind) or kind == .convolution) try allocator.dupe(i64, op.base_dilations) else null;
    errdefer if (base_dilations) |owned| allocator.free(owned);
    const window_dilations = if (isReduceWindowKind(kind) or kind == .convolution) try allocator.dupe(i64, op.window_dilations) else null;
    errdefer if (window_dilations) |owned| allocator.free(owned);
    const window_reversal = if (kind == .convolution) try allocator.dupe(bool, op.window_reversal) else null;
    errdefer if (window_reversal) |owned| allocator.free(owned);
    const offset_dims = if (kind == .gather) try allocator.dupe(i64, op.offset_dims) else null;
    errdefer if (offset_dims) |owned| allocator.free(owned);
    const collapsed_slice_dims = if (kind == .gather) try allocator.dupe(i64, op.collapsed_slice_dims) else null;
    errdefer if (collapsed_slice_dims) |owned| allocator.free(owned);
    const operand_batching_dims = if (kind == .gather) try allocator.dupe(i64, op.operand_batching_dims) else null;
    errdefer if (operand_batching_dims) |owned| allocator.free(owned);
    const start_indices_batching_dims = if (kind == .gather) try allocator.dupe(i64, op.start_indices_batching_dims) else null;
    errdefer if (start_indices_batching_dims) |owned| allocator.free(owned);
    const start_index_map = if (kind == .gather) try allocator.dupe(i64, op.start_index_map) else null;
    errdefer if (start_index_map) |owned| allocator.free(owned);
    const update_window_dims = if (kind == .scatter) try allocator.dupe(i64, op.update_window_dims) else null;
    errdefer if (update_window_dims) |owned| allocator.free(owned);
    const inserted_window_dims = if (kind == .scatter) try allocator.dupe(i64, op.inserted_window_dims) else null;
    errdefer if (inserted_window_dims) |owned| allocator.free(owned);
    const input_batching_dims = if (kind == .scatter) try allocator.dupe(i64, op.input_batching_dims) else null;
    errdefer if (input_batching_dims) |owned| allocator.free(owned);
    const scatter_indices_batching_dims = if (kind == .scatter) try allocator.dupe(i64, op.scatter_indices_batching_dims) else null;
    errdefer if (scatter_indices_batching_dims) |owned| allocator.free(owned);
    const scatter_dims_to_operand_dims = if (kind == .scatter) try allocator.dupe(i64, op.scatter_dims_to_operand_dims) else null;
    errdefer if (scatter_dims_to_operand_dims) |owned| allocator.free(owned);
    const dimensions = if (kind == .reverse or kind == .fft) try allocator.dupe(i64, op.dimensions) else null;
    errdefer if (dimensions) |owned| allocator.free(owned);
    const custom_call_target = if (kind == .custom_call) try allocator.dupe(u8, op.custom_call_target) else null;
    errdefer if (custom_call_target) |owned| allocator.free(owned);
    const reduce_dimensions = if (isReduceKind(kind)) try allocator.dupe(i64, op.reduce_dimensions) else null;
    errdefer if (reduce_dimensions) |owned| allocator.free(owned);
    const lhs_batch_dimensions = if (kind == .dot_general) try allocator.dupe(i64, op.lhs_batch_dimensions) else null;
    errdefer if (lhs_batch_dimensions) |owned| allocator.free(owned);
    const rhs_batch_dimensions = if (kind == .dot_general) try allocator.dupe(i64, op.rhs_batch_dimensions) else null;
    errdefer if (rhs_batch_dimensions) |owned| allocator.free(owned);
    const lhs_contracting_dimensions = if (kind == .dot_general) try allocator.dupe(i64, op.lhs_contracting_dimensions) else null;
    errdefer if (lhs_contracting_dimensions) |owned| allocator.free(owned);
    const rhs_contracting_dimensions = if (kind == .dot_general) try allocator.dupe(i64, op.rhs_contracting_dimensions) else null;
    errdefer if (rhs_contracting_dimensions) |owned| allocator.free(owned);
    const input_spatial_dimensions = if (kind == .convolution) try allocator.dupe(i64, op.input_spatial_dimensions) else null;
    errdefer if (input_spatial_dimensions) |owned| allocator.free(owned);
    const kernel_spatial_dimensions = if (kind == .convolution) try allocator.dupe(i64, op.kernel_spatial_dimensions) else null;
    errdefer if (kernel_spatial_dimensions) |owned| allocator.free(owned);
    const output_spatial_dimensions = if (kind == .convolution) try allocator.dupe(i64, op.output_spatial_dimensions) else null;
    errdefer if (output_spatial_dimensions) |owned| allocator.free(owned);
    const literal = if (kind == .constant) try allocator.dupe(u8, op.literal) else null;
    errdefer if (literal) |owned| allocator.free(owned);

    return .{
        .kind = kind,
        .inputs = inputs,
        .outputs = outputs,
        .region_ids = region_ids,
        .dims = dims,
        .permutation = permutation,
        .broadcast_dimensions = broadcast_dimensions,
        .start_indices = start_indices,
        .limit_indices = limit_indices,
        .strides = strides,
        .slice_sizes = slice_sizes,
        .edge_padding_low = edge_padding_low,
        .edge_padding_high = edge_padding_high,
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
        .index_vector_dim = if (kind == .gather or kind == .scatter) op.index_vector_dim else null,
        .scatter_update_kind = if (kind == .scatter) op.scatter_update_kind else null,
        .dimension = if (kind == .concatenate or kind == .sort) op.dimension else null,
        .top_k_k = if (kind == .top_k) op.top_k_k else null,
        .iota_dimension = if (kind == .iota) op.iota_dimension else null,
        .fft_kind = if (kind == .fft) op.fft_kind else null,
        .dimensions = dimensions,
        .tuple_index = if (kind == .get_tuple_element) op.tuple_index else null,
        .lower = if (kind == .cholesky) op.lower else null,
        .triangular_left_side = if (kind == .triangular_solve) op.triangular_left_side else null,
        .triangular_lower = if (kind == .triangular_solve) op.triangular_lower else null,
        .triangular_unit_diagonal = if (kind == .triangular_solve) op.triangular_unit_diagonal else null,
        .triangular_transpose = if (kind == .triangular_solve) op.triangular_transpose else null,
        .custom_call_target = custom_call_target,
        .rng_distribution = if (kind == .rng) op.rng_distribution else null,
        .reduce_dimensions = reduce_dimensions,
        .lhs_batch_dimensions = lhs_batch_dimensions,
        .rhs_batch_dimensions = rhs_batch_dimensions,
        .lhs_contracting_dimensions = lhs_contracting_dimensions,
        .rhs_contracting_dimensions = rhs_contracting_dimensions,
        .input_batch_dimension = if (kind == .convolution) op.input_batch_dimension else null,
        .input_feature_dimension = if (kind == .convolution) op.input_feature_dimension else null,
        .input_spatial_dimensions = input_spatial_dimensions,
        .kernel_input_feature_dimension = if (kind == .convolution) op.kernel_input_feature_dimension else null,
        .kernel_output_feature_dimension = if (kind == .convolution) op.kernel_output_feature_dimension else null,
        .kernel_spatial_dimensions = kernel_spatial_dimensions,
        .output_batch_dimension = if (kind == .convolution) op.output_batch_dimension else null,
        .output_feature_dimension = if (kind == .convolution) op.output_feature_dimension else null,
        .output_spatial_dimensions = output_spatial_dimensions,
        .feature_group_count = if (kind == .convolution) op.feature_group_count else null,
        .batch_group_count = if (kind == .convolution) op.batch_group_count else null,
        .compare_direction = if (kind == .compare or kind == .sort) op.compare_direction else null,
        .literal = literal,
    };
}

pub fn lowerAnalysisOpsToPlan(allocator: std.mem.Allocator, ops: []const Operation, num_parameters: usize) ![]PlanInstruction {
    _ = num_parameters;
    if (ops.len == 0) return allocator.alloc(PlanInstruction, 0);
    const plan_instructions = try allocator.alloc(PlanInstruction, ops.len);
    errdefer {
        freeInstructions(allocator, plan_instructions);
    }
    @memset(plan_instructions, .{ .kind = .unsupported });
    for (ops, plan_instructions, 0..) |op, *plan_instruction, index| {
        _ = index;
        const kind = instructionKindFromStablehlo(op.name);
        plan_instruction.* = try makePlanInstruction(allocator, op, kind);
    }
    return plan_instructions;
}

pub fn pruneDeadPlanInstructions(allocator: std.mem.Allocator, instructions: []PlanInstruction, output_ids: []const ValueId, value_count: usize, regions: []ir.PlanRegion) ![]PlanInstruction {
    if (instructions.len == 0) return instructions;
    var live_values = try allocator.alloc(bool, value_count);
    defer allocator.free(live_values);
    @memset(live_values, false);
    for (output_ids) |id| {
        if (id.index < live_values.len) live_values[id.index] = true;
    }

    var live_instructions = try allocator.alloc(bool, instructions.len);
    defer allocator.free(live_instructions);
    @memset(live_instructions, false);

    var reverse_index = instructions.len;
    while (reverse_index > 0) {
        reverse_index -= 1;
        const instruction = instructions[reverse_index];
        var live = instruction.outputs.len == 0;
        for (instruction.outputs) |output| {
            if (output.index < live_values.len and live_values[output.index]) {
                live = true;
                break;
            }
        }
        if (!live) continue;
        live_instructions[reverse_index] = true;
        for (instruction.inputs) |input| {
            if (input.index < live_values.len) live_values[input.index] = true;
        }
    }

    var live_count: usize = 0;
    for (live_instructions) |live| {
        if (live) live_count += 1;
    }
    if (live_count == instructions.len) return instructions;

    var old_to_new = try allocator.alloc(usize, instructions.len);
    defer allocator.free(old_to_new);
    @memset(old_to_new, std.math.maxInt(usize));

    var next_index: usize = 0;
    for (live_instructions, 0..) |live, old_index| {
        if (live) {
            old_to_new[old_index] = next_index;
            next_index += 1;
        }
    }

    for (regions) |*region| {
        if (region.parent_instruction_index < old_to_new.len) {
            const new_parent = old_to_new[region.parent_instruction_index];
            if (new_parent != std.math.maxInt(usize)) {
                region.parent_instruction_index = new_parent;
            }
        }
    }

    const pruned = try allocator.alloc(PlanInstruction, live_count);
    var out_index: usize = 0;
    for (instructions, live_instructions) |*instruction, live| {
        if (live) {
            pruned[out_index] = instruction.*;
            instruction.* = .{ .kind = .unsupported };
            out_index += 1;
        }
    }
    freeInstructions(allocator, instructions);
    return pruned;
}

