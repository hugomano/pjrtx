const std = @import("std");
const mlir = @import("c");
const ir = @import("src/compiler/ir");
const model = @import("compiler_model.zig");
const memory = @import("plan_memory.zig");
const plan_instruction = @import("plan_instruction.zig");
const sharding = @import("stablehlo_sharding.zig");

const AnalyzeError = model.AnalyzeError;
const Dialect = model.Dialect;
const Operation = model.Operation;
const ShardingMetadata = model.ShardingMetadata;
const Value = model.Value;
const ValueId = model.ValueId;
const ValueRole = model.ValueRole;
const makeValue = plan_instruction.makeValue;
const makeUnknownDescriptor = plan_instruction.makeUnknownDescriptor;
const makeShardingMetadata = sharding.makeShardingMetadata;
const freeRegionContents = memory.freeRegionContents;

pub const CapiAnalysisBuilder = struct {
    allocator: std.mem.Allocator,
    diagnostic_writer: *std.Io.Writer,
    dialects: std.ArrayList(Dialect) = .empty,
    ops: std.ArrayList(Operation) = .empty,
    regions: std.ArrayList(ir.PlanRegion) = .empty,
    parameter_descriptors: std.ArrayList(ir.BufferDescriptor) = .empty,
    parameter_shardings: std.ArrayList(ShardingMetadata) = .empty,
    output_shardings: std.ArrayList(ShardingMetadata) = .empty,
    values: std.ArrayList(Value) = .empty,
    value_map: std.ArrayList(ValueMapEntry) = .empty,
    value_parameter_aliases: std.ArrayList(ValueParameterAlias) = .empty,
    output_ids: std.ArrayList(ValueId) = .empty,
    output_aliases: std.ArrayList(ir.OutputAlias) = .empty,
    num_parameters: usize = 0,
    num_outputs: usize = 0,
    saw_program_body: bool = false,
    saw_shardy: bool = false,
    saw_manual_computation: bool = false,
    saw_sdy_return: bool = false,
    manual_line: usize = 0,
    manual_column: usize = 0,

    pub fn deinitPartial(self: *CapiAnalysisBuilder) void {
        for (self.output_shardings.items) |metadata| metadata.deinit(self.allocator);
        for (self.parameter_shardings.items) |metadata| metadata.deinit(self.allocator);
        for (self.parameter_descriptors.items) |descriptor| self.allocator.free(descriptor.dims);
        for (self.values.items) |value| {
            self.allocator.free(value.descriptor.dims);
            self.allocator.free(value.elements);
        }
        freeRegionContents(self.allocator, self.regions.items);
        for (self.ops.items) |op| op.deinit(self.allocator);
        self.output_aliases.deinit(self.allocator);
        self.output_ids.deinit(self.allocator);
        self.value_parameter_aliases.deinit(self.allocator);
        self.value_map.deinit(self.allocator);
        self.values.deinit(self.allocator);
        self.output_shardings.deinit(self.allocator);
        self.parameter_shardings.deinit(self.allocator);
        self.parameter_descriptors.deinit(self.allocator);
        self.regions.deinit(self.allocator);
        self.ops.deinit(self.allocator);
        self.dialects.deinit(self.allocator);
    }

    pub fn ensureShardings(self: *CapiAnalysisBuilder, list: *std.ArrayList(ShardingMetadata), count: usize) !void {
        while (list.items.len < count) {
            try list.append(self.allocator, try makeShardingMetadata(self.allocator, .replicated, "pjrtx_mesh"));
        }
    }

    pub fn replaceMetadata(self: *CapiAnalysisBuilder, list: *std.ArrayList(ShardingMetadata), index: usize, metadata: ShardingMetadata) void {
        list.items[index].deinit(self.allocator);
        list.items[index] = metadata;
    }

    pub fn ensureParameterDescriptors(self: *CapiAnalysisBuilder, count: usize) !void {
        while (self.parameter_descriptors.items.len < count) {
            try self.parameter_descriptors.append(self.allocator, try makeUnknownDescriptor(self.allocator));
        }
    }

    pub fn replaceParameterDescriptor(self: *CapiAnalysisBuilder, index: usize, descriptor: ir.BufferDescriptor) void {
        self.allocator.free(self.parameter_descriptors.items[index].dims);
        self.parameter_descriptors.items[index] = descriptor;
    }

    pub fn registerValue(self: *CapiAnalysisBuilder, value: mlir.MlirValue, role: ValueRole, descriptor: ir.BufferDescriptor) !ValueId {
        if (self.lookupValue(value)) |existing| {
            self.allocator.free(descriptor.dims);
            return existing;
        }
        const id: ValueId = .{ .index = @intCast(self.values.items.len) };
        try self.values.append(self.allocator, makeValue(id.index, role, descriptor));
        errdefer _ = self.values.pop();
        try self.value_map.append(self.allocator, .{ .mlir_value = value, .id = id });
        return id;
    }

    pub fn setValueStorage(self: *CapiAnalysisBuilder, id: ValueId, storage: ir.ValueStorageKind, elements: []const ValueId) !void {
        if (id.index >= self.values.items.len) return error.InvalidStablehloModule;
        const owned_elements = try self.allocator.dupe(ValueId, elements);
        errdefer self.allocator.free(owned_elements);
        const value = &self.values.items[id.index];
        self.allocator.free(value.elements);
        value.storage = storage;
        value.elements = owned_elements;
    }

    pub fn lookupValue(self: *CapiAnalysisBuilder, value: mlir.MlirValue) ?ValueId {
        for (self.value_map.items) |entry| {
            if (mlir.mlirValueEqual(entry.mlir_value, value)) return entry.id;
        }
        return null;
    }
};



pub const ValueMapEntry = struct {
    mlir_value: mlir.MlirValue,
    id: ValueId,
};

pub const ValueParameterAlias = struct {
    value_id: ValueId,
    parameter_index: u32,
};

pub fn appendValueAlias(builder: *CapiAnalysisBuilder, value: mlir.MlirValue, id: ValueId) !void {
    if (builder.lookupValue(value) != null) return;
    try builder.value_map.append(builder.allocator, .{ .mlir_value = value, .id = id });
}

pub fn appendValueParameterAlias(builder: *CapiAnalysisBuilder, value_id: ValueId, parameter_index: u32) !void {
    for (builder.value_parameter_aliases.items) |alias| {
        if (alias.value_id.index == value_id.index and alias.parameter_index == parameter_index) return;
    }
    try builder.value_parameter_aliases.append(builder.allocator, .{ .value_id = value_id, .parameter_index = parameter_index });
}

pub fn parameterAliasForValue(builder: *const CapiAnalysisBuilder, value_id: ValueId) ?u32 {
    for (builder.value_parameter_aliases.items) |alias| if (alias.value_id.index == value_id.index) return alias.parameter_index;
    return null;
}

pub fn appendOutputAlias(builder: *CapiAnalysisBuilder, output_index: u32, parameter_index: u32, kind: ir.OutputAliasKind) !void {
    for (builder.output_aliases.items) |*alias| {
        if (alias.output_index == output_index and alias.parameter_index == parameter_index) {
            if (alias.kind == .donation and kind == .identity) alias.kind = .identity;
            return;
        }
    }
    try builder.output_aliases.append(builder.allocator, .{ .output_index = output_index, .parameter_index = parameter_index, .kind = kind });
}

pub fn promoteExistingOutputAlias(builder: *CapiAnalysisBuilder, output_index: u32, parameter_index: u32, kind: ir.OutputAliasKind) void {
    for (builder.output_aliases.items) |*alias| {
        if (alias.output_index == output_index and alias.parameter_index == parameter_index) {
            if (alias.kind == .donation and kind == .identity) alias.kind = .identity;
            return;
        }
    }
}
