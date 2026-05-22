const std = @import("std");
const backend_api = @import("backend_selection.zig");
const ir = @import("src/compiler/ir");

const device_memory = @import("device_memory.zig");

const DeviceMemoryTopology = device_memory.DeviceMemoryTopology;

/// Allocates the stable executable-cache fingerprint for a compiler plan and target topology.
pub fn alloc(
    allocator: std.mem.Allocator,
    backend: backend_api.Backend,
    topology: *const DeviceMemoryTopology,
    optimized_program: []const u8,
    plan: *const ir.ExecutablePlan,
) ![]u8 {
    var hasher = std.hash.Wyhash.init(0);
    hasher.update("pjrtx-executable-cache-v4");
    hasher.update(optimized_program);
    updateCapabilities(&hasher, backend);
    updateTargetDevices(&hasher, topology, plan);
    updateCompileOptions(&hasher, plan);
    updatePlanBody(&hasher, plan);
    return std.fmt.allocPrint(allocator, "pjrtx-{x}", .{hasher.final()});
}

fn updateCapabilities(hasher: *std.hash.Wyhash, backend: backend_api.Backend) void {
    const caps = backend.capabilities();
    hasher.update(caps.name);
    hasher.update(std.mem.asBytes(&caps.supports_device_buffers));
    hasher.update(std.mem.asBytes(&caps.supports_unified_memory));
    hasher.update(std.mem.asBytes(&caps.supports_async_execution));
    const custom_call_registry_version = backend.customCallRegistryVersion();
    hasher.update(std.mem.asBytes(&custom_call_registry_version));
}

fn updateTargetDevices(hasher: *std.hash.Wyhash, topology: *const DeviceMemoryTopology, plan: *const ir.ExecutablePlan) void {
    const devices = topology.deviceSlice();
    const device_count = plan.options.numDevices();
    hasher.update(std.mem.asBytes(&device_count));
    for (0..device_count) |i| {
        const device_id = if (plan.options.device_assignment.len != 0) plan.options.device_assignment[i] else devices[i].id;
        hasher.update(std.mem.asBytes(&device_id));
        const device = topology.lookupDevice(device_id) orelse {
            const missing_device: i32 = -1;
            hasher.update(std.mem.asBytes(&missing_device));
            continue;
        };
        hasher.update(std.mem.asBytes(&device.local_hardware_id));
        hasher.update(std.mem.asBytes(&device.registry_id));
        hasher.update(std.mem.asBytes(&device.process_index));
        hasher.update(std.mem.asBytes(&device.addressable));
        hasher.update(std.mem.asBytes(&device.memory_bytes));
        hasher.update(std.mem.asBytes(&device.has_unified_memory));
        hasher.update(std.mem.asBytes(&device.default_memory_id));
        hasher.update(device.name);
        hasher.update(device.debug_string);
        const memory = topology.lookupMemory(device.default_memory_id) orelse continue;
        hasher.update(std.mem.asBytes(&memory.id));
        hasher.update(std.mem.asBytes(&memory.kind));
        hasher.update(std.mem.sliceAsBytes(memory.addressable_device_ids));
    }
}

fn updateCompileOptions(hasher: *std.hash.Wyhash, plan: *const ir.ExecutablePlan) void {
    hasher.update(std.mem.asBytes(&plan.options.num_replicas));
    hasher.update(std.mem.asBytes(&plan.options.num_partitions));
    hasher.update(std.mem.asBytes(&plan.options.use_shardy_partitioner));
    hasher.update(std.mem.sliceAsBytes(plan.options.device_assignment));
    updateSharding(hasher, plan.parameter_shardings);
    updateSharding(hasher, plan.output_shardings);
}

fn updatePlanBody(hasher: *std.hash.Wyhash, plan: *const ir.ExecutablePlan) void {
    hasher.update(std.mem.sliceAsBytes(plan.output_ids));
    hasher.update(std.mem.sliceAsBytes(plan.donated_parameter_indices));
    for (plan.output_aliases) |alias| {
        hasher.update(std.mem.asBytes(&alias.output_index));
        hasher.update(std.mem.asBytes(&alias.parameter_index));
        hasher.update(std.mem.asBytes(&alias.kind));
    }
    for (plan.values) |value| {
        hasher.update(std.mem.asBytes(&value.role));
        hasher.update(std.mem.asBytes(&value.descriptor.element_type));
        hasher.update(std.mem.sliceAsBytes(value.descriptor.dims));
        hasher.update(std.mem.asBytes(&value.descriptor.layout));
        hasher.update(std.mem.asBytes(&value.descriptor.device_id));
        hasher.update(std.mem.asBytes(&value.descriptor.memory_id));
        hasher.update(std.mem.asBytes(&value.descriptor.shard_index));
        hasher.update(std.mem.asBytes(&value.storage));
        hasher.update(std.mem.sliceAsBytes(value.elements));
    }
    for (plan.regions) |region| updateRegion(hasher, region);
    for (plan.instructions) |instruction| updateInstruction(hasher, instruction);
}

fn updateRegion(hasher: *std.hash.Wyhash, region: ir.PlanRegion) void {
    hasher.update(std.mem.asBytes(&region.id));
    hasher.update(std.mem.asBytes(&region.parent_instruction_index));
    hasher.update(std.mem.asBytes(&region.kind));
    for (region.argument_descriptors) |descriptor| updateDescriptor(hasher, descriptor);
    for (region.return_descriptors) |descriptor| updateDescriptor(hasher, descriptor);
}

fn updateDescriptor(hasher: *std.hash.Wyhash, descriptor: ir.BufferDescriptor) void {
    hasher.update(std.mem.asBytes(&descriptor.element_type));
    hasher.update(std.mem.sliceAsBytes(descriptor.dims));
    hasher.update(std.mem.asBytes(&descriptor.layout));
}

fn updateSharding(hasher: *std.hash.Wyhash, shardings: []const ir.ShardingPlan) void {
    hasher.update(std.mem.asBytes(&shardings.len));
    for (shardings) |sharding| {
        hasher.update(std.mem.asBytes(&sharding.kind));
        hasher.update(sharding.mesh_name);
        hasher.update(std.mem.sliceAsBytes(sharding.device_assignment));
    }
}

fn updateInstruction(hasher: *std.hash.Wyhash, instruction: ir.PlanInstruction) void {
    hasher.update(std.mem.asBytes(&instruction.kind));
    hasher.update(std.mem.sliceAsBytes(instruction.inputs));
    hasher.update(std.mem.sliceAsBytes(instruction.outputs));
    hasher.update(std.mem.sliceAsBytes(instruction.region_ids));
    updateOptionalI64Slice(hasher, instruction.dims);
    updateOptionalI64Slice(hasher, instruction.permutation);
    updateOptionalI64Slice(hasher, instruction.broadcast_dimensions);
    updateOptionalI64Slice(hasher, instruction.start_indices);
    updateOptionalI64Slice(hasher, instruction.limit_indices);
    updateOptionalI64Slice(hasher, instruction.strides);
    updateOptionalI64Slice(hasher, instruction.slice_sizes);
    updateOptionalI64Slice(hasher, instruction.edge_padding_low);
    updateOptionalI64Slice(hasher, instruction.edge_padding_high);
    updateOptionalI64Slice(hasher, instruction.interior_padding);
    updateOptionalI64Slice(hasher, instruction.offset_dims);
    updateOptionalI64Slice(hasher, instruction.collapsed_slice_dims);
    updateOptionalI64Slice(hasher, instruction.operand_batching_dims);
    updateOptionalI64Slice(hasher, instruction.start_indices_batching_dims);
    updateOptionalI64Slice(hasher, instruction.start_index_map);
    updateOptionalI64Slice(hasher, instruction.update_window_dims);
    updateOptionalI64Slice(hasher, instruction.inserted_window_dims);
    updateOptionalI64Slice(hasher, instruction.input_batching_dims);
    updateOptionalI64Slice(hasher, instruction.scatter_indices_batching_dims);
    updateOptionalI64Slice(hasher, instruction.scatter_dims_to_operand_dims);
    hasher.update(std.mem.asBytes(&instruction.index_vector_dim));
    hasher.update(std.mem.asBytes(&instruction.scatter_update_kind));
    hasher.update(std.mem.asBytes(&instruction.dimension));
    hasher.update(std.mem.asBytes(&instruction.top_k_k));
    hasher.update(std.mem.asBytes(&instruction.iota_dimension));
    hasher.update(std.mem.asBytes(&instruction.fft_kind));
    updateOptionalI64Slice(hasher, instruction.dimensions);
    hasher.update(std.mem.asBytes(&instruction.tuple_index));
    hasher.update(std.mem.asBytes(&instruction.lower));
    hasher.update(std.mem.asBytes(&instruction.triangular_left_side));
    hasher.update(std.mem.asBytes(&instruction.triangular_lower));
    hasher.update(std.mem.asBytes(&instruction.triangular_unit_diagonal));
    hasher.update(std.mem.asBytes(&instruction.triangular_transpose));
    updateOptionalBytes(hasher, instruction.custom_call_target);
    updateOptionalI64Slice(hasher, instruction.reduce_dimensions);
    updateOptionalI64Slice(hasher, instruction.lhs_batch_dimensions);
    updateOptionalI64Slice(hasher, instruction.rhs_batch_dimensions);
    updateOptionalI64Slice(hasher, instruction.lhs_contracting_dimensions);
    updateOptionalI64Slice(hasher, instruction.rhs_contracting_dimensions);
    hasher.update(std.mem.asBytes(&instruction.compare_direction));
    updateOptionalBytes(hasher, instruction.literal);
}

fn updateOptionalI64Slice(hasher: *std.hash.Wyhash, values: ?[]const i64) void {
    const present = values != null;
    hasher.update(std.mem.asBytes(&present));
    if (values) |slice| hasher.update(std.mem.sliceAsBytes(slice));
}

fn updateOptionalBytes(hasher: *std.hash.Wyhash, bytes: ?[]const u8) void {
    const present = bytes != null;
    hasher.update(std.mem.asBytes(&present));
    if (bytes) |slice| hasher.update(slice);
}
