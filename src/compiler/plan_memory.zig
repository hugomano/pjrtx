const std = @import("std");
const ir = @import("src/compiler/ir");
const model = @import("compiler_model.zig");
const plan_instruction_mod = @import("plan_instruction.zig");
const makeDescriptor = plan_instruction_mod.makeDescriptor;
const PlanInstruction = ir.PlanInstruction;
const Value = model.Value;
const ValueId = model.ValueId;
const makeStructuredValue = plan_instruction_mod.makeStructuredValue;

pub fn freeDescriptorList(allocator: std.mem.Allocator, descriptors: []const ir.BufferDescriptor) void {
    for (descriptors) |descriptor| allocator.free(descriptor.dims);
    allocator.free(descriptors);
}

pub fn freeRegionValueList(allocator: std.mem.Allocator, values: []const ir.RegionValue) void {
    for (values) |value| {
        allocator.free(value.descriptor.dims);
        if (value.literal) |literal| allocator.free(literal);
    }
    allocator.free(values);
}

pub fn freeRegionInstructionList(allocator: std.mem.Allocator, instructions: []const ir.RegionInstruction) void {
    for (instructions) |instruction| {
        allocator.free(instruction.inputs);
        allocator.free(instruction.outputs);
        freeDescriptorList(allocator, instruction.operand_descriptors);
        freeDescriptorList(allocator, instruction.result_descriptors);
    }
    allocator.free(instructions);
}

pub fn freeRegions(allocator: std.mem.Allocator, regions: []const ir.PlanRegion) void {
    for (regions) |region| {
        freeRegionValueList(allocator, region.values);
        freeDescriptorList(allocator, region.argument_descriptors);
        freeRegionInstructionList(allocator, region.instructions);
        freeDescriptorList(allocator, region.return_descriptors);
        allocator.free(region.terminator_operands);
        freeDescriptorList(allocator, region.terminator_operand_descriptors);
    }
    allocator.free(regions);
}

pub fn freeRegionContents(allocator: std.mem.Allocator, regions: []const ir.PlanRegion) void {
    for (regions) |region| {
        freeRegionValueList(allocator, region.values);
        freeDescriptorList(allocator, region.argument_descriptors);
        freeRegionInstructionList(allocator, region.instructions);
        freeDescriptorList(allocator, region.return_descriptors);
        allocator.free(region.terminator_operands);
        freeDescriptorList(allocator, region.terminator_operand_descriptors);
    }
}
pub fn cloneValues(allocator: std.mem.Allocator, source: []const Value) ![]Value {
    const values = try allocator.alloc(Value, source.len);
    var initialized: usize = 0;
    errdefer {
        for (values[0..initialized]) |value| {
            allocator.free(value.descriptor.dims);
            allocator.free(value.elements);
        }
        allocator.free(values);
    }
    for (source, values) |src, *dst| {
        const dims = try allocator.dupe(i64, src.descriptor.dims);
        errdefer allocator.free(dims);
        const elements = try allocator.dupe(ValueId, src.elements);
        errdefer allocator.free(elements);
        dst.* = makeStructuredValue(src.id.index, src.role, makeDescriptor(dims, src.descriptor.element_type), src.storage, elements);
        initialized += 1;
    }
    return values;
}

pub fn cloneDescriptorList(allocator: std.mem.Allocator, source: []const ir.BufferDescriptor) ![]const ir.BufferDescriptor {
    const descriptors = try allocator.alloc(ir.BufferDescriptor, source.len);
    var initialized: usize = 0;
    errdefer {
        for (descriptors[0..initialized]) |descriptor| allocator.free(descriptor.dims);
        allocator.free(descriptors);
    }
    for (source, descriptors) |src, *dst| {
        dst.* = makeDescriptor(try allocator.dupe(i64, src.dims), src.element_type);
        dst.layout = src.layout;
        dst.device_id = src.device_id;
        dst.memory_id = src.memory_id;
        dst.shard_index = src.shard_index;
        initialized += 1;
    }
    return descriptors;
}

pub fn cloneRegionValueList(allocator: std.mem.Allocator, source: []const ir.RegionValue) ![]const ir.RegionValue {
    const values = try allocator.alloc(ir.RegionValue, source.len);
    var initialized: usize = 0;
    errdefer {
        for (values[0..initialized]) |value| {
            allocator.free(value.descriptor.dims);
            if (value.literal) |literal| allocator.free(literal);
        }
        allocator.free(values);
    }
    for (source, values) |src, *dst| {
        const literal = if (src.literal) |bytes| try allocator.dupe(u8, bytes) else null;
        var literal_owned = true;
        errdefer if (literal_owned) if (literal) |bytes| allocator.free(bytes);
        const dims = try allocator.dupe(i64, src.descriptor.dims);
        var dims_owned = true;
        errdefer if (dims_owned) allocator.free(dims);
        dst.* = .{
            .id = src.id,
            .role = src.role,
            .descriptor = makeDescriptor(dims, src.descriptor.element_type),
            .literal = literal,
        };
        literal_owned = false;
        dims_owned = false;
        dst.descriptor.layout = src.descriptor.layout;
        dst.descriptor.device_id = src.descriptor.device_id;
        dst.descriptor.memory_id = src.descriptor.memory_id;
        dst.descriptor.shard_index = src.descriptor.shard_index;
        initialized += 1;
    }
    return values;
}

pub fn cloneRegionInstructionList(allocator: std.mem.Allocator, source: []const ir.RegionInstruction) ![]const ir.RegionInstruction {
    const instructions = try allocator.alloc(ir.RegionInstruction, source.len);
    var initialized: usize = 0;
    errdefer {
        for (instructions[0..initialized]) |instruction| {
            allocator.free(instruction.inputs);
            allocator.free(instruction.outputs);
            freeDescriptorList(allocator, instruction.operand_descriptors);
            freeDescriptorList(allocator, instruction.result_descriptors);
        }
        allocator.free(instructions);
    }
    for (source, instructions) |src, *dst| {
        const inputs = try allocator.dupe(ir.RegionValueId, src.inputs);
        errdefer allocator.free(inputs);
        const outputs = try allocator.dupe(ir.RegionValueId, src.outputs);
        errdefer allocator.free(outputs);
        const operands = try cloneDescriptorList(allocator, src.operand_descriptors);
        errdefer freeDescriptorList(allocator, operands);
        const results = try cloneDescriptorList(allocator, src.result_descriptors);
        errdefer freeDescriptorList(allocator, results);
        dst.* = .{
            .kind = src.kind,
            .line = src.line,
            .column = src.column,
            .inputs = inputs,
            .outputs = outputs,
            .operand_descriptors = operands,
            .result_descriptors = results,
            .compare_direction = src.compare_direction,
        };
        initialized += 1;
    }
    return instructions;
}

pub fn cloneRegions(allocator: std.mem.Allocator, source: []const ir.PlanRegion) ![]ir.PlanRegion {
    const regions = try allocator.alloc(ir.PlanRegion, source.len);
    var initialized: usize = 0;
    errdefer {
        for (regions[0..initialized]) |region| {
            freeRegionValueList(allocator, region.values);
            freeDescriptorList(allocator, region.argument_descriptors);
            freeRegionInstructionList(allocator, region.instructions);
            freeDescriptorList(allocator, region.return_descriptors);
            allocator.free(region.terminator_operands);
            freeDescriptorList(allocator, region.terminator_operand_descriptors);
        }
        allocator.free(regions);
    }
    for (source, regions) |src, *dst| {
        const values = try cloneRegionValueList(allocator, src.values);
        errdefer freeRegionValueList(allocator, values);
        const args = try cloneDescriptorList(allocator, src.argument_descriptors);
        errdefer freeDescriptorList(allocator, args);
        const instructions = try cloneRegionInstructionList(allocator, src.instructions);
        errdefer freeRegionInstructionList(allocator, instructions);
        const returns = try cloneDescriptorList(allocator, src.return_descriptors);
        errdefer freeDescriptorList(allocator, returns);
        const terminator_operand_ids = try allocator.dupe(ir.RegionValueId, src.terminator_operands);
        errdefer allocator.free(terminator_operand_ids);
        const terminator_operand_descriptors = try cloneDescriptorList(allocator, src.terminator_operand_descriptors);
        errdefer freeDescriptorList(allocator, terminator_operand_descriptors);
        dst.* = .{
            .id = src.id,
            .parent_instruction_index = src.parent_instruction_index,
            .kind = src.kind,
            .values = values,
            .argument_descriptors = args,
            .instructions = instructions,
            .return_descriptors = returns,
            .terminator_operands = terminator_operand_ids,
            .terminator_operand_descriptors = terminator_operand_descriptors,
        };
        initialized += 1;
    }
    return regions;
}

pub fn freeInstructions(allocator: std.mem.Allocator, instructions: []PlanInstruction) void {
    for (instructions) |instruction| {
        if (instruction.inputs.len != 0) allocator.free(instruction.inputs);
        if (instruction.outputs.len != 0) allocator.free(instruction.outputs);
        if (instruction.region_ids.len != 0) allocator.free(instruction.region_ids);
        if (instruction.dims) |dims| allocator.free(dims);
        if (instruction.permutation) |permutation| allocator.free(permutation);
        if (instruction.broadcast_dimensions) |broadcast_dimensions| allocator.free(broadcast_dimensions);
        if (instruction.start_indices) |start_indices| allocator.free(start_indices);
        if (instruction.limit_indices) |limit_indices| allocator.free(limit_indices);
        if (instruction.strides) |strides| allocator.free(strides);
        if (instruction.slice_sizes) |slice_sizes| allocator.free(slice_sizes);
        if (instruction.edge_padding_low) |padding| allocator.free(padding);
        if (instruction.edge_padding_high) |padding| allocator.free(padding);
        if (instruction.interior_padding) |padding| allocator.free(padding);
        if (instruction.window_dimensions) |dims| allocator.free(dims);
        if (instruction.window_strides) |dims| allocator.free(dims);
        if (instruction.base_dilations) |dims| allocator.free(dims);
        if (instruction.window_dilations) |dims| allocator.free(dims);
        if (instruction.window_reversal) |dims| allocator.free(dims);
        if (instruction.offset_dims) |dims| allocator.free(dims);
        if (instruction.collapsed_slice_dims) |dims| allocator.free(dims);
        if (instruction.operand_batching_dims) |dims| allocator.free(dims);
        if (instruction.start_indices_batching_dims) |dims| allocator.free(dims);
        if (instruction.start_index_map) |dims| allocator.free(dims);
        if (instruction.update_window_dims) |dims| allocator.free(dims);
        if (instruction.inserted_window_dims) |dims| allocator.free(dims);
        if (instruction.input_batching_dims) |dims| allocator.free(dims);
        if (instruction.scatter_indices_batching_dims) |dims| allocator.free(dims);
        if (instruction.scatter_dims_to_operand_dims) |dims| allocator.free(dims);
        if (instruction.dimensions) |dimensions| allocator.free(dimensions);
        if (instruction.custom_call_target) |target| allocator.free(target);
        if (instruction.reduce_dimensions) |reduce_dimensions| allocator.free(reduce_dimensions);
        if (instruction.lhs_batch_dimensions) |dims| allocator.free(dims);
        if (instruction.rhs_batch_dimensions) |dims| allocator.free(dims);
        if (instruction.lhs_contracting_dimensions) |dims| allocator.free(dims);
        if (instruction.rhs_contracting_dimensions) |dims| allocator.free(dims);
        if (instruction.input_spatial_dimensions) |dims| allocator.free(dims);
        if (instruction.kernel_spatial_dimensions) |dims| allocator.free(dims);
        if (instruction.output_spatial_dimensions) |dims| allocator.free(dims);
        if (instruction.literal) |literal| allocator.free(literal);
    }
    allocator.free(instructions);
}
