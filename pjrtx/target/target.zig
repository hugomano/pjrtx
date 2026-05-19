const std = @import("std");

/// Target owns hardware facts used to explain performance ceilings and
/// legality before compiler, backend, or runtime policy can act on a program.
pub const ValidationError = error{
    InvalidTargetDescription,
};

pub const BufferType = enum {
    invalid,
    pred,
    s8,
    s16,
    s32,
    s64,
    u8,
    u16,
    u32,
    u64,
    f16,
    f32,
    f64,
    bf16,

    pub fn byteSize(self: BufferType) ?u16 {
        return switch (self) {
            .invalid => null,
            .pred, .s8, .u8 => 1,
            .s16, .u16, .f16, .bf16 => 2,
            .s32, .u32, .f32 => 4,
            .s64, .u64, .f64 => 8,
        };
    }
};

pub const TargetKind = enum {
    metal_v0,
    npu_v0,
};

pub const ExecutionUnitKind = enum {
    scalar,
    vector,
    matrix,
    library,
    dma,
    collective,
    unknown,
};

pub const OpClass = enum {
    elementwise,
    matmul,
    transcendental,
    memory,
    collective,
};

pub const RateSource = enum {
    measured,
    vendor_spec,
    heuristic,
    synthetic,
    unknown,
};

pub const DTypeRate = struct {
    dtype: BufferType,
    op_class: OpClass,
    ops_per_second: ?f64,
    source: RateSource,
    note: []const u8,
};

pub const ExecutionUnit = struct {
    id: u32,
    name: []const u8,
    kind: ExecutionUnitKind,
    dtype_rates: []const DTypeRate,
};

pub const MemorySpaceKind = enum {
    host_unpinned,
    host_pinned,
    device_unified,
    device_hbm,
    local_sram,
    scratchpad,
    remote_device,
    unknown,
};

pub const TargetMemorySpace = struct {
    id: u32,
    name: []const u8,
    kind: MemorySpaceKind,
    capacity_bytes: ?u64,
    bandwidth_bytes_per_second: ?f64,
    note: []const u8,
};

pub const TargetTransferEdge = struct {
    id: u32,
    src_memory_space: u32,
    dst_memory_space: u32,
    bandwidth_bytes_per_second: ?f64,
    latency_ns: ?u64,
    supports_async: bool,
    engine_unit_id: ?u32,
    note: []const u8,
};

pub const TargetDevice = struct {
    id: u32,
    local_hardware_id: i32,
    name: []const u8,
    memory_space_ids: []const u32,
    execution_unit_ids: []const u32,
};

pub const TargetDescription = struct {
    name: []const u8,
    kind: TargetKind,
    devices: []const TargetDevice,
    memory_spaces: []const TargetMemorySpace,
    transfer_edges: []const TargetTransferEdge,
    execution_units: []const ExecutionUnit,
};

/// Target validation is intentionally structural. A backend can still reject a
/// legal target later, but missing devices, memory spaces, transfer endpoints,
/// or execution units must be caught before compiler planning uses them.
pub fn validateTargetDescription(target: TargetDescription, diagnostics: *std.Io.Writer) !void {
    if (target.devices.len == 0) {
        try diagnostics.writeAll("pass=pjrtx-target-validate feature=target reason=target has no devices\n");
        return ValidationError.InvalidTargetDescription;
    }
    if (target.memory_spaces.len == 0) {
        try diagnostics.writeAll("pass=pjrtx-target-validate feature=target reason=target has no memory spaces\n");
        return ValidationError.InvalidTargetDescription;
    }
    if (target.execution_units.len == 0) {
        try diagnostics.writeAll("pass=pjrtx-target-validate feature=target reason=target has no execution units\n");
        return ValidationError.InvalidTargetDescription;
    }

    for (target.devices) |device| {
        for (device.memory_space_ids) |memory_space_id| {
            if (!hasMemorySpace(target.memory_spaces, memory_space_id)) {
                try diagnostics.print("pass=pjrtx-target-validate feature=memory-space reason=device references unknown memory space device={d} memory_space={d}\n", .{ device.id, memory_space_id });
                return ValidationError.InvalidTargetDescription;
            }
        }
        for (device.execution_unit_ids) |execution_unit_id| {
            if (!hasExecutionUnit(target.execution_units, execution_unit_id)) {
                try diagnostics.print("pass=pjrtx-target-validate feature=execution-unit reason=device references unknown execution unit device={d} unit={d}\n", .{ device.id, execution_unit_id });
                return ValidationError.InvalidTargetDescription;
            }
        }
    }

    for (target.transfer_edges) |edge| {
        if (!hasMemorySpace(target.memory_spaces, edge.src_memory_space)) {
            try diagnostics.print("pass=pjrtx-target-validate feature=transfer-edge reason=edge references unknown source memory edge={d} memory_space={d}\n", .{ edge.id, edge.src_memory_space });
            return ValidationError.InvalidTargetDescription;
        }
        if (!hasMemorySpace(target.memory_spaces, edge.dst_memory_space)) {
            try diagnostics.print("pass=pjrtx-target-validate feature=transfer-edge reason=edge references unknown destination memory edge={d} memory_space={d}\n", .{ edge.id, edge.dst_memory_space });
            return ValidationError.InvalidTargetDescription;
        }
        if (edge.engine_unit_id) |engine_unit_id| {
            if (!hasExecutionUnit(target.execution_units, engine_unit_id)) {
                try diagnostics.print("pass=pjrtx-target-validate feature=transfer-edge reason=edge references unknown engine unit edge={d} unit={d}\n", .{ edge.id, engine_unit_id });
                return ValidationError.InvalidTargetDescription;
            }
        }
    }
}

/// The target summary is a stable human-readable schema, not debug prose. Tests
/// depend on unknown fields being printed explicitly so missing hardware facts
/// cannot disappear from reports.
pub fn writeTargetSummary(target: TargetDescription, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    try writer.print("target:\n  name: {s}\n  kind: {s}\n", .{ target.name, @tagName(target.kind) });
    try writer.writeAll("memory_spaces:\n");
    for (target.memory_spaces) |memory_space| {
        try writer.print("  memory.{d} {s} capacity=", .{ memory_space.id, memory_space.name });
        try writeOptionalU64(writer, memory_space.capacity_bytes);
        try writer.writeAll(" bandwidth=");
        try writeOptionalF64(writer, memory_space.bandwidth_bytes_per_second);
        try writer.writeAll("\n");
    }
    try writer.writeAll("execution_units:\n");
    for (target.execution_units) |unit| {
        try writer.print("  unit.{d} {s} kind={s}\n", .{ unit.id, unit.name, @tagName(unit.kind) });
    }
    try writer.writeAll("transfer_edges:\n");
    for (target.transfer_edges) |edge| {
        try writer.print("  edge.{d} {d}->{d} async={} bandwidth=", .{ edge.id, edge.src_memory_space, edge.dst_memory_space, edge.supports_async });
        try writeOptionalF64(writer, edge.bandwidth_bytes_per_second);
        try writer.writeAll("\n");
    }
}

pub fn hasMemorySpace(memory_spaces: []const TargetMemorySpace, id: u32) bool {
    for (memory_spaces) |memory_space| {
        if (memory_space.id == id) return true;
    }
    return false;
}

pub fn hasExecutionUnit(execution_units: []const ExecutionUnit, id: u32) bool {
    for (execution_units) |execution_unit| {
        if (execution_unit.id == id) return true;
    }
    return false;
}

fn writeOptionalU64(writer: *std.Io.Writer, value: ?u64) std.Io.Writer.Error!void {
    if (value) |known| {
        try writer.print("{d}", .{known});
    } else {
        try writer.writeAll("unknown");
    }
}

fn writeOptionalF64(writer: *std.Io.Writer, value: ?f64) std.Io.Writer.Error!void {
    if (value) |known| {
        try writer.print("{d}", .{known});
    } else {
        try writer.writeAll("unknown");
    }
}

test "buffer types expose explicit byte sizes" {
    const f32_bytes: ?u16 = 4;
    const invalid_bytes: ?u16 = null;

    try std.testing.expectEqual(f32_bytes, BufferType.f32.byteSize());
    try std.testing.expectEqual(invalid_bytes, BufferType.invalid.byteSize());
}

test "target validation accepts a minimal NPU target" {
    const memory_spaces = [_]TargetMemorySpace{
        .{ .id = 0, .name = "host_pinned", .kind = .host_pinned, .capacity_bytes = null, .bandwidth_bytes_per_second = null, .note = "" },
        .{ .id = 1, .name = "device_hbm", .kind = .device_hbm, .capacity_bytes = 34359738368, .bandwidth_bytes_per_second = 1000000000000, .note = "" },
    };
    const units = [_]ExecutionUnit{
        .{ .id = 0, .name = "trn2_dma_engine", .kind = .dma, .dtype_rates = &.{} },
        .{ .id = 1, .name = "trn2_tensor_engine", .kind = .matrix, .dtype_rates = &.{} },
    };
    const edges = [_]TargetTransferEdge{
        .{ .id = 0, .src_memory_space = 0, .dst_memory_space = 1, .bandwidth_bytes_per_second = null, .latency_ns = null, .supports_async = true, .engine_unit_id = 0, .note = "" },
    };
    const device_memory_spaces = [_]u32{ 0, 1 };
    const device_execution_units = [_]u32{ 0, 1 };
    const devices = [_]TargetDevice{
        .{ .id = 0, .local_hardware_id = 0, .name = "npu", .memory_space_ids = &device_memory_spaces, .execution_unit_ids = &device_execution_units },
    };
    const description: TargetDescription = .{
        .name = "npu_v0",
        .kind = .npu_v0,
        .devices = &devices,
        .memory_spaces = &memory_spaces,
        .transfer_edges = &edges,
        .execution_units = &units,
    };
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    try validateTargetDescription(description, &diagnostics.writer);
    try std.testing.expectEqualStrings("", diagnostics.writer.buffered());
}

test "target validation rejects missing transfer memory spaces" {
    const memory_spaces = [_]TargetMemorySpace{
        .{ .id = 0, .name = "host_pinned", .kind = .host_pinned, .capacity_bytes = null, .bandwidth_bytes_per_second = null, .note = "" },
    };
    const units = [_]ExecutionUnit{
        .{ .id = 0, .name = "trn2_dma_engine", .kind = .dma, .dtype_rates = &.{} },
    };
    const edges = [_]TargetTransferEdge{
        .{ .id = 0, .src_memory_space = 0, .dst_memory_space = 99, .bandwidth_bytes_per_second = null, .latency_ns = null, .supports_async = true, .engine_unit_id = 0, .note = "" },
    };
    const device_memory_spaces = [_]u32{0};
    const device_execution_units = [_]u32{0};
    const devices = [_]TargetDevice{
        .{ .id = 0, .local_hardware_id = 0, .name = "npu", .memory_space_ids = &device_memory_spaces, .execution_unit_ids = &device_execution_units },
    };
    const description: TargetDescription = .{
        .name = "npu_v0",
        .kind = .npu_v0,
        .devices = &devices,
        .memory_spaces = &memory_spaces,
        .transfer_edges = &edges,
        .execution_units = &units,
    };
    var diagnostics: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer diagnostics.deinit();

    try std.testing.expectError(ValidationError.InvalidTargetDescription, validateTargetDescription(description, &diagnostics.writer));
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "unknown destination memory") != null);
}

test "target summary renders unknown fields explicitly" {
    const memory_spaces = [_]TargetMemorySpace{
        .{ .id = 0, .name = "device_unified", .kind = .device_unified, .capacity_bytes = null, .bandwidth_bytes_per_second = null, .note = "" },
    };
    const units = [_]ExecutionUnit{
        .{ .id = 0, .name = "metal_shader_core", .kind = .unknown, .dtype_rates = &.{} },
    };
    const devices = [_]TargetDevice{
        .{ .id = 0, .local_hardware_id = 0, .name = "metal", .memory_space_ids = &[_]u32{0}, .execution_unit_ids = &[_]u32{0} },
    };
    const description: TargetDescription = .{
        .name = "metal_v0",
        .kind = .metal_v0,
        .devices = &devices,
        .memory_spaces = &memory_spaces,
        .transfer_edges = &.{},
        .execution_units = &units,
    };
    var output: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer output.deinit();

    try writeTargetSummary(description, &output.writer);
    try std.testing.expect(std.mem.indexOf(u8, output.writer.buffered(), "capacity=unknown bandwidth=unknown") != null);
}
