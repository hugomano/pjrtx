const std = @import("std");
const mlx_metal = @import("c");

pub const MAX_DEVICES = 64;

pub const MemoryKind = enum {
    device,
    host_pinned,
    host_unpinned,
};

pub const BackendKind = enum {
    synthetic,
    metal_mlx,
};

pub const Device = struct {
    id: i32,
    local_hardware_id: i32,
    registry_id: u64 = 0,
    process_index: i32 = 0,
    addressable: bool = true,
    name: []const u8,
    debug_string: []const u8,
    memory_bytes: u64 = 0,
    has_unified_memory: bool = false,
    default_memory_id: i32,
    default_memory: *Memory,
    addressable_memories: []*Memory,
};

pub const Memory = struct {
    id: i32,
    kind: MemoryKind,
    debug_string: []const u8,
    addressable_device_ids: []const i32,
    addressable_devices: []*Device,
};

pub const Topology = struct {
    device_assignment: []const i32,
    num_replicas: i32,
    num_partitions: i32,

    pub fn numDevices(self: Topology) usize {
        return @intCast(self.num_replicas * self.num_partitions);
    }
};

pub const Client = struct {
    allocator: std.mem.Allocator,
    backend_kind: BackendKind,
    devices: []Device,
    memories: []Memory,
    device_handles: []*Device,
    memory_handles: []*Memory,
    topology: Topology,

    pub fn initForBackend(allocator: std.mem.Allocator, backend_kind: BackendKind, device_count: usize) !*Client {
        if (device_count == 0 or device_count > MAX_DEVICES) return error.InvalidDeviceCount;

        const client = try allocator.create(Client);
        errdefer allocator.destroy(client);

        const devices = try allocator.alloc(Device, device_count);
        errdefer allocator.free(devices);

        const memories = try allocator.alloc(Memory, device_count);
        errdefer allocator.free(memories);

        const device_handles = try allocator.alloc(*Device, device_count);
        errdefer allocator.free(device_handles);

        const memory_handles = try allocator.alloc(*Memory, device_count);
        errdefer allocator.free(memory_handles);

        const assignment = try allocator.alloc(i32, device_count);
        errdefer allocator.free(assignment);

        for (0..device_count) |i| {
            const id: i32 = @intCast(i);
            const default_name = switch (backend_kind) {
                .synthetic => "Synthetic Metal device",
                .metal_mlx => "Metal/MLX device",
            };
            const debug_string_text = switch (backend_kind) {
                .synthetic => "PjRTx synthetic Metal device",
                .metal_mlx => "PjRTx Metal/MLX device",
            };
            const name = try allocator.dupe(u8, default_name);
            errdefer allocator.free(name);

            const debug_string = try allocator.dupe(u8, debug_string_text);
            errdefer allocator.free(debug_string);

            const memory_debug_string = try allocator.dupe(u8, "device");
            errdefer allocator.free(memory_debug_string);

            const device_memories = try allocator.alloc(*Memory, 1);
            errdefer allocator.free(device_memories);

            const memory_devices = try allocator.alloc(*Device, 1);
            errdefer allocator.free(memory_devices);

            const ids = try allocator.alloc(i32, 1);
            errdefer allocator.free(ids);

            devices[i] = .{
                .id = id,
                .local_hardware_id = id,
                .name = name,
                .debug_string = debug_string,
                .default_memory_id = id,
                .default_memory = &memories[i],
                .addressable_memories = device_memories,
            };
            ids[0] = id;
            memories[i] = .{
                .id = id,
                .kind = .device,
                .debug_string = memory_debug_string,
                .addressable_device_ids = ids,
                .addressable_devices = memory_devices,
            };
            devices[i].addressable_memories[0] = &memories[i];
            memories[i].addressable_devices[0] = &devices[i];
            device_handles[i] = &devices[i];
            memory_handles[i] = &memories[i];
            assignment[i] = id;
        }

        client.* = .{
            .allocator = allocator,
            .backend_kind = backend_kind,
            .devices = devices,
            .memories = memories,
            .device_handles = device_handles,
            .memory_handles = memory_handles,
            .topology = .{
                .device_assignment = assignment,
                .num_replicas = 1,
                .num_partitions = @intCast(device_count),
            },
        };
        return client;
    }

    pub fn initSynthetic(allocator: std.mem.Allocator, device_count: usize) !*Client {
        return initForBackend(allocator, .synthetic, device_count);
    }

    pub fn initMetalMlxBootstrap(allocator: std.mem.Allocator) !*Client {
        var metal_devices: [MAX_DEVICES]mlx_metal.PjrtxMlxMetalDeviceInfo = undefined;
        const copied = mlx_metal.pjrtx_mlx_metal_copy_devices(&metal_devices, MAX_DEVICES);
        if (copied <= 0) return initForBackend(allocator, .metal_mlx, 1);
        return initMetalMlxFromDevices(allocator, metal_devices[0..@intCast(copied)]);
    }

    pub fn deinit(self: *Client) void {
        for (self.devices) |device| {
            self.allocator.free(device.addressable_memories);
            self.allocator.free(device.name);
            self.allocator.free(device.debug_string);
        }
        for (self.memories) |memory| {
            self.allocator.free(memory.addressable_devices);
            self.allocator.free(memory.addressable_device_ids);
            self.allocator.free(memory.debug_string);
        }
        self.allocator.free(self.topology.device_assignment);
        self.allocator.free(self.memory_handles);
        self.allocator.free(self.device_handles);
        self.allocator.free(self.memories);
        self.allocator.free(self.devices);
        self.allocator.destroy(self);
    }

    fn initMetalMlxFromDevices(
        allocator: std.mem.Allocator,
        metal_devices: []const mlx_metal.PjrtxMlxMetalDeviceInfo,
    ) !*Client {
        const client = try initForBackend(allocator, .metal_mlx, metal_devices.len);
        errdefer client.deinit();

        for (metal_devices, 0..) |metal_device, i| {
            const name_bytes = cNameBytes(&metal_device.name);
            const name = try allocator.dupe(u8, name_bytes);
            errdefer allocator.free(name);

            var debug_buffer: [256]u8 = undefined;
            var debug_writer = std.Io.Writer.fixed(&debug_buffer);
            try debug_writer.print("PjRTx Metal/MLX device {d}: {s}", .{ i, name });
            const debug_string = try allocator.dupe(u8, debug_writer.buffered());
            errdefer allocator.free(debug_string);

            allocator.free(client.devices[i].name);
            allocator.free(client.devices[i].debug_string);
            client.devices[i].name = name;
            client.devices[i].debug_string = debug_string;
            client.devices[i].local_hardware_id = metal_device.ordinal;
            client.devices[i].registry_id = metal_device.registry_id;
            client.devices[i].memory_bytes = metal_device.recommended_max_working_set_size;
            client.devices[i].has_unified_memory = metal_device.has_unified_memory != 0;
        }

        return client;
    }

    pub fn lookupDevice(self: *const Client, id: i32) ?*const Device {
        for (self.devices) |*device| {
            if (device.id == id) return device;
        }
        return null;
    }

    pub fn lookupMemory(self: *const Client, id: i32) ?*const Memory {
        for (self.memories) |*memory| {
            if (memory.id == id) return memory;
        }
        return null;
    }
};

fn cNameBytes(name: *const [128]u8) []const u8 {
    const end = std.mem.indexOfScalar(u8, name, 0) orelse name.len;
    return name[0..end];
}

fn fillCName(name: *[128]u8, text: []const u8) void {
    @memset(name, 0);
    const len = @min(name.len - 1, text.len);
    for (text[0..len], 0..) |byte, i| name[i] = byte;
}

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

    pub fn byteSize(self: BufferType) usize {
        return switch (self) {
            .invalid => 0,
            .pred, .s8, .u8 => 1,
            .s16, .u16, .f16, .bf16 => 2,
            .s32, .u32, .f32 => 4,
            .s64, .u64, .f64 => 8,
        };
    }
};

pub const ElementwiseBinaryOp = enum {
    add,
    subtract,
    multiply,
    divide,
};

pub const ElementwiseUnaryOp = enum {
    negate,
    exp,
    tanh,
    sqrt,
    rsqrt,
};

pub const Buffer = struct {
    allocator: std.mem.Allocator,
    backend_kind: BackendKind,
    element_type: BufferType,
    dims: []i64,
    device_id: i32,
    memory_id: i32,
    device: *Device,
    memory: *Memory,
    shard_index: usize,
    bytes: []u8,
    mlx_buffer: ?*mlx_metal.PjrtxMlxMetalBuffer = null,
    deleted: bool = false,

    pub fn initHostCopy(
        allocator: std.mem.Allocator,
        element_type: BufferType,
        dims_: []const i64,
        device: *Device,
        memory: *Memory,
        shard_index: usize,
        src: []const u8,
    ) !*Buffer {
        return initHostCopyForBackend(allocator, .synthetic, element_type, dims_, device, memory, shard_index, src);
    }

    pub fn initHostCopyForBackend(
        allocator: std.mem.Allocator,
        backend_kind: BackendKind,
        element_type: BufferType,
        dims_: []const i64,
        device: *Device,
        memory: *Memory,
        shard_index: usize,
        src: []const u8,
    ) !*Buffer {
        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, dims_);
        errdefer allocator.free(dims);

        const bytes = try allocator.dupe(u8, src);
        errdefer allocator.free(bytes);

        var mlx_buffer: ?*mlx_metal.PjrtxMlxMetalBuffer = null;
        if (backend_kind == .metal_mlx and src.len != 0) {
            const dtype = mlxDtype(element_type) orelse return error.UnsupportedElementType;
            mlx_buffer = mlx_metal.pjrtx_mlx_metal_buffer_from_host_typed(device.local_hardware_id, src.ptr, src.len, dtype, dims.ptr, dims.len);
            if (mlx_buffer == null) return error.MlxBufferAllocationFailed;
        }
        errdefer if (mlx_buffer) |owned| mlx_metal.pjrtx_mlx_metal_buffer_destroy(owned);

        buffer.* = .{
            .allocator = allocator,
            .backend_kind = backend_kind,
            .element_type = element_type,
            .dims = dims,
            .device_id = device.id,
            .memory_id = memory.id,
            .device = device,
            .memory = memory,
            .shard_index = shard_index,
            .bytes = bytes,
            .mlx_buffer = mlx_buffer,
        };
        return buffer;
    }

    pub fn initDeviceCopy(
        allocator: std.mem.Allocator,
        src: *Buffer,
        shard_index: usize,
    ) !*Buffer {
        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, src.dims);
        errdefer allocator.free(dims);

        const bytes = try allocator.dupe(u8, src.bytes);
        errdefer allocator.free(bytes);

        var mlx_buffer: ?*mlx_metal.PjrtxMlxMetalBuffer = null;
        if (src.mlx_buffer) |src_mlx| {
            mlx_buffer = mlx_metal.pjrtx_mlx_metal_buffer_clone(src_mlx);
            if (mlx_buffer == null) return error.MlxCommandSubmissionFailed;
        }
        errdefer if (mlx_buffer) |owned| mlx_metal.pjrtx_mlx_metal_buffer_destroy(owned);

        buffer.* = .{
            .allocator = allocator,
            .backend_kind = src.backend_kind,
            .element_type = src.element_type,
            .dims = dims,
            .device_id = src.device_id,
            .memory_id = src.memory_id,
            .device = src.device,
            .memory = src.memory,
            .shard_index = shard_index,
            .bytes = bytes,
            .mlx_buffer = mlx_buffer,
        };
        return buffer;
    }

    pub fn initElementwiseBinary(
        allocator: std.mem.Allocator,
        op: ElementwiseBinaryOp,
        lhs: *Buffer,
        rhs: *Buffer,
        shard_index: usize,
    ) !*Buffer {
        if (lhs.element_type != rhs.element_type) return error.UnsupportedElementType;
        if (lhs.element_type != .u8 and lhs.element_type != .f32) return error.UnsupportedElementType;
        if (!std.mem.eql(i64, lhs.dims, rhs.dims) or lhs.bytes.len != rhs.bytes.len) return error.ShapeMismatch;

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, lhs.dims);
        errdefer allocator.free(dims);

        const bytes = try allocator.alloc(u8, lhs.bytes.len);
        errdefer allocator.free(bytes);
        switch (lhs.element_type) {
            .u8 => {
                for (lhs.bytes, rhs.bytes, 0..) |a, b, i| {
                    bytes[i] = switch (op) {
                        .add => a +% b,
                        .subtract => a -% b,
                        .multiply => a *% b,
                        .divide => if (b == 0) 0 else a / b,
                    };
                }
            },
            .f32 => {
                var offset: usize = 0;
                while (offset < lhs.bytes.len) : (offset += 4) {
                    const a_bits = std.mem.readInt(u32, lhs.bytes[offset..][0..4], .little);
                    const b_bits = std.mem.readInt(u32, rhs.bytes[offset..][0..4], .little);
                    const a: f32 = @bitCast(a_bits);
                    const b: f32 = @bitCast(b_bits);
                    const value: f32 = switch (op) {
                        .add => a + b,
                        .subtract => a - b,
                        .multiply => a * b,
                        .divide => a / b,
                    };
                    std.mem.writeInt(u32, bytes[offset..][0..4], @bitCast(value), .little);
                }
            },
            else => unreachable,
        }

        var mlx_buffer: ?*mlx_metal.PjrtxMlxMetalBuffer = null;
        if (lhs.mlx_buffer != null and rhs.mlx_buffer != null) {
            mlx_buffer = mlx_metal.pjrtx_mlx_metal_buffer_binary(lhs.mlx_buffer.?, rhs.mlx_buffer.?, mlxBinaryOpCode(op));
            if (mlx_buffer == null) return error.MlxCommandSubmissionFailed;
        }
        errdefer if (mlx_buffer) |owned| mlx_metal.pjrtx_mlx_metal_buffer_destroy(owned);

        buffer.* = .{
            .allocator = allocator,
            .backend_kind = lhs.backend_kind,
            .element_type = lhs.element_type,
            .dims = dims,
            .device_id = lhs.device_id,
            .memory_id = lhs.memory_id,
            .device = lhs.device,
            .memory = lhs.memory,
            .shard_index = shard_index,
            .bytes = bytes,
            .mlx_buffer = mlx_buffer,
        };
        return buffer;
    }

    pub fn initU8Add(
        allocator: std.mem.Allocator,
        lhs: *Buffer,
        rhs: *Buffer,
        shard_index: usize,
    ) !*Buffer {
        return initElementwiseBinary(allocator, .add, lhs, rhs, shard_index);
    }

    pub fn initU8Binary(
        allocator: std.mem.Allocator,
        op: ElementwiseBinaryOp,
        lhs: *Buffer,
        rhs: *Buffer,
        shard_index: usize,
    ) !*Buffer {
        if (lhs.element_type != .u8 or rhs.element_type != .u8) return error.UnsupportedElementType;
        return initElementwiseBinary(allocator, op, lhs, rhs, shard_index);
    }

    pub fn initElementwiseUnary(
        allocator: std.mem.Allocator,
        op: ElementwiseUnaryOp,
        src: *Buffer,
        shard_index: usize,
    ) !*Buffer {
        if (src.element_type != .u8 and src.element_type != .f32) return error.UnsupportedElementType;
        if (src.element_type == .u8 and op != .negate) return error.UnsupportedElementType;

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, src.dims);
        errdefer allocator.free(dims);

        const bytes = try allocator.alloc(u8, src.bytes.len);
        errdefer allocator.free(bytes);
        switch (src.element_type) {
            .u8 => {
                for (src.bytes, 0..) |value, i| {
                    bytes[i] = switch (op) {
                        .negate => 0 -% value,
                        else => unreachable,
                    };
                }
            },
            .f32 => {
                var offset: usize = 0;
                while (offset < src.bytes.len) : (offset += 4) {
                    const bits = std.mem.readInt(u32, src.bytes[offset..][0..4], .little);
                    const value: f32 = @bitCast(bits);
                    const out: f32 = switch (op) {
                        .negate => -value,
                        .exp => std.math.exp(value),
                        .tanh => std.math.tanh(value),
                        .sqrt => std.math.sqrt(value),
                        .rsqrt => 1.0 / std.math.sqrt(value),
                    };
                    std.mem.writeInt(u32, bytes[offset..][0..4], @bitCast(out), .little);
                }
            },
            else => unreachable,
        }

        var mlx_buffer: ?*mlx_metal.PjrtxMlxMetalBuffer = null;
        if (src.mlx_buffer) |src_mlx| {
            mlx_buffer = mlx_metal.pjrtx_mlx_metal_buffer_unary(src_mlx, mlxUnaryOpCode(op));
            if (mlx_buffer == null) return error.MlxCommandSubmissionFailed;
        }
        errdefer if (mlx_buffer) |owned| mlx_metal.pjrtx_mlx_metal_buffer_destroy(owned);

        buffer.* = .{
            .allocator = allocator,
            .backend_kind = src.backend_kind,
            .element_type = src.element_type,
            .dims = dims,
            .device_id = src.device_id,
            .memory_id = src.memory_id,
            .device = src.device,
            .memory = src.memory,
            .shard_index = shard_index,
            .bytes = bytes,
            .mlx_buffer = mlx_buffer,
        };
        return buffer;
    }

    pub fn initReshape(
        allocator: std.mem.Allocator,
        src: *Buffer,
        new_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (denseByteSize(src.element_type, new_dims) != src.bytes.len) return error.ShapeMismatch;

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, new_dims);
        errdefer allocator.free(dims);

        const bytes = try allocator.dupe(u8, src.bytes);
        errdefer allocator.free(bytes);

        var mlx_buffer: ?*mlx_metal.PjrtxMlxMetalBuffer = null;
        if (src.backend_kind == .metal_mlx and src.bytes.len != 0) {
            const dtype = mlxDtype(src.element_type) orelse return error.UnsupportedElementType;
            mlx_buffer = mlx_metal.pjrtx_mlx_metal_buffer_from_host_typed(src.device.local_hardware_id, src.bytes.ptr, src.bytes.len, dtype, new_dims.ptr, new_dims.len);
            if (mlx_buffer == null) return error.MlxBufferAllocationFailed;
        }
        errdefer if (mlx_buffer) |owned| mlx_metal.pjrtx_mlx_metal_buffer_destroy(owned);

        buffer.* = .{
            .allocator = allocator,
            .backend_kind = src.backend_kind,
            .element_type = src.element_type,
            .dims = dims,
            .device_id = src.device_id,
            .memory_id = src.memory_id,
            .device = src.device,
            .memory = src.memory,
            .shard_index = shard_index,
            .bytes = bytes,
            .mlx_buffer = mlx_buffer,
        };
        return buffer;
    }

    pub fn initTranspose(
        allocator: std.mem.Allocator,
        src: *Buffer,
        permutation: []const i64,
        new_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (!validPermutation(permutation, src.dims.len)) return error.ShapeMismatch;
        if (new_dims.len != permutation.len) return error.ShapeMismatch;
        if (denseByteSize(src.element_type, new_dims) != src.bytes.len) return error.ShapeMismatch;
        const element_size = src.element_type.byteSize();
        if (element_size == 0) return error.UnsupportedElementType;

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, new_dims);
        errdefer allocator.free(dims);

        const bytes = try allocator.alloc(u8, src.bytes.len);
        errdefer allocator.free(bytes);

        const src_strides = try rowMajorStrides(allocator, src.dims);
        defer allocator.free(src_strides);
        const dst_strides = try rowMajorStrides(allocator, new_dims);
        defer allocator.free(dst_strides);
        const coords = try allocator.alloc(usize, new_dims.len);
        defer allocator.free(coords);

        const element_count = if (element_size == 0) 0 else src.bytes.len / element_size;
        for (0..element_count) |dst_index| {
            unravelIndex(dst_index, new_dims, dst_strides, coords);
            var src_index: usize = 0;
            for (permutation, 0..) |src_axis_i64, dst_axis| {
                src_index += coords[dst_axis] * src_strides[@intCast(src_axis_i64)];
            }
            @memcpy(
                bytes[dst_index * element_size ..][0..element_size],
                src.bytes[src_index * element_size ..][0..element_size],
            );
        }

        var mlx_buffer: ?*mlx_metal.PjrtxMlxMetalBuffer = null;
        if (src.mlx_buffer) |src_mlx| {
            mlx_buffer = mlx_metal.pjrtx_mlx_metal_buffer_transpose(src_mlx, permutation.ptr, permutation.len);
            if (mlx_buffer == null) return error.MlxCommandSubmissionFailed;
        }
        errdefer if (mlx_buffer) |owned| mlx_metal.pjrtx_mlx_metal_buffer_destroy(owned);

        buffer.* = .{
            .allocator = allocator,
            .backend_kind = src.backend_kind,
            .element_type = src.element_type,
            .dims = dims,
            .device_id = src.device_id,
            .memory_id = src.memory_id,
            .device = src.device,
            .memory = src.memory,
            .shard_index = shard_index,
            .bytes = bytes,
            .mlx_buffer = mlx_buffer,
        };
        return buffer;
    }

    pub fn initBroadcastInDim(
        allocator: std.mem.Allocator,
        src: *Buffer,
        broadcast_dimensions: []const i64,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (!validBroadcastDimensions(broadcast_dimensions, src.dims, output_dims)) return error.ShapeMismatch;
        const element_size = src.element_type.byteSize();
        if (element_size == 0) return error.UnsupportedElementType;
        const output_byte_size = denseByteSize(src.element_type, output_dims);

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, output_dims);
        errdefer allocator.free(dims);

        const bytes = try allocator.alloc(u8, output_byte_size);
        errdefer allocator.free(bytes);

        const src_strides = try rowMajorStrides(allocator, src.dims);
        defer allocator.free(src_strides);
        const dst_strides = try rowMajorStrides(allocator, output_dims);
        defer allocator.free(dst_strides);
        const coords = try allocator.alloc(usize, output_dims.len);
        defer allocator.free(coords);

        const element_count = if (element_size == 0) 0 else output_byte_size / element_size;
        for (0..element_count) |dst_index| {
            unravelIndex(dst_index, output_dims, dst_strides, coords);
            var src_index: usize = 0;
            for (broadcast_dimensions, 0..) |output_axis_i64, src_axis| {
                const output_axis: usize = @intCast(output_axis_i64);
                const src_coord = if (src.dims[src_axis] == 1) 0 else coords[output_axis];
                src_index += src_coord * src_strides[src_axis];
            }
            @memcpy(
                bytes[dst_index * element_size ..][0..element_size],
                src.bytes[src_index * element_size ..][0..element_size],
            );
        }

        var mlx_buffer: ?*mlx_metal.PjrtxMlxMetalBuffer = null;
        if (src.mlx_buffer) |src_mlx| {
            mlx_buffer = mlx_metal.pjrtx_mlx_metal_buffer_broadcast_in_dim(src_mlx, broadcast_dimensions.ptr, broadcast_dimensions.len, output_dims.ptr, output_dims.len);
            if (mlx_buffer == null) return error.MlxCommandSubmissionFailed;
        }
        errdefer if (mlx_buffer) |owned| mlx_metal.pjrtx_mlx_metal_buffer_destroy(owned);

        buffer.* = .{
            .allocator = allocator,
            .backend_kind = src.backend_kind,
            .element_type = src.element_type,
            .dims = dims,
            .device_id = src.device_id,
            .memory_id = src.memory_id,
            .device = src.device,
            .memory = src.memory,
            .shard_index = shard_index,
            .bytes = bytes,
            .mlx_buffer = mlx_buffer,
        };
        return buffer;
    }

    pub fn initSlice(
        allocator: std.mem.Allocator,
        src: *Buffer,
        start_indices: []const i64,
        limit_indices: []const i64,
        strides_: []const i64,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (!validSlice(start_indices, limit_indices, strides_, src.dims, output_dims)) return error.ShapeMismatch;
        const element_size = src.element_type.byteSize();
        if (element_size == 0) return error.UnsupportedElementType;
        const output_byte_size = denseByteSize(src.element_type, output_dims);

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, output_dims);
        errdefer allocator.free(dims);

        const bytes = try allocator.alloc(u8, output_byte_size);
        errdefer allocator.free(bytes);

        const src_strides = try rowMajorStrides(allocator, src.dims);
        defer allocator.free(src_strides);
        const dst_strides = try rowMajorStrides(allocator, output_dims);
        defer allocator.free(dst_strides);
        const coords = try allocator.alloc(usize, output_dims.len);
        defer allocator.free(coords);

        const element_count = if (element_size == 0) 0 else output_byte_size / element_size;
        for (0..element_count) |dst_index| {
            unravelIndex(dst_index, output_dims, dst_strides, coords);
            var src_index: usize = 0;
            for (coords, 0..) |coord, axis| {
                const src_coord = @as(usize, @intCast(start_indices[axis])) + coord * @as(usize, @intCast(strides_[axis]));
                src_index += src_coord * src_strides[axis];
            }
            @memcpy(
                bytes[dst_index * element_size ..][0..element_size],
                src.bytes[src_index * element_size ..][0..element_size],
            );
        }

        var mlx_buffer: ?*mlx_metal.PjrtxMlxMetalBuffer = null;
        if (src.mlx_buffer) |src_mlx| {
            mlx_buffer = mlx_metal.pjrtx_mlx_metal_buffer_slice(
                src_mlx,
                start_indices.ptr,
                limit_indices.ptr,
                strides_.ptr,
                start_indices.len,
                output_dims.ptr,
                output_dims.len,
            );
            if (mlx_buffer == null) return error.MlxCommandSubmissionFailed;
        }
        errdefer if (mlx_buffer) |owned| mlx_metal.pjrtx_mlx_metal_buffer_destroy(owned);

        buffer.* = .{
            .allocator = allocator,
            .backend_kind = src.backend_kind,
            .element_type = src.element_type,
            .dims = dims,
            .device_id = src.device_id,
            .memory_id = src.memory_id,
            .device = src.device,
            .memory = src.memory,
            .shard_index = shard_index,
            .bytes = bytes,
            .mlx_buffer = mlx_buffer,
        };
        return buffer;
    }

    pub fn initConcatenate(
        allocator: std.mem.Allocator,
        lhs: *Buffer,
        rhs: *Buffer,
        dimension: i64,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (lhs.element_type != rhs.element_type) return error.UnsupportedElementType;
        if (!validConcatenate(lhs.dims, rhs.dims, dimension, output_dims)) return error.ShapeMismatch;
        const element_size = lhs.element_type.byteSize();
        if (element_size == 0) return error.UnsupportedElementType;
        const output_byte_size = denseByteSize(lhs.element_type, output_dims);

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, output_dims);
        errdefer allocator.free(dims);

        const bytes = try allocator.alloc(u8, output_byte_size);
        errdefer allocator.free(bytes);

        const lhs_strides = try rowMajorStrides(allocator, lhs.dims);
        defer allocator.free(lhs_strides);
        const rhs_strides = try rowMajorStrides(allocator, rhs.dims);
        defer allocator.free(rhs_strides);
        const dst_strides = try rowMajorStrides(allocator, output_dims);
        defer allocator.free(dst_strides);
        const coords = try allocator.alloc(usize, output_dims.len);
        defer allocator.free(coords);

        const concat_axis: usize = @intCast(dimension);
        const lhs_axis_len: usize = @intCast(lhs.dims[concat_axis]);
        const element_count = if (element_size == 0) 0 else output_byte_size / element_size;
        for (0..element_count) |dst_index| {
            unravelIndex(dst_index, output_dims, dst_strides, coords);
            const use_lhs = coords[concat_axis] < lhs_axis_len;
            var src_index: usize = 0;
            if (use_lhs) {
                for (coords, 0..) |coord, axis| {
                    src_index += coord * lhs_strides[axis];
                }
                @memcpy(
                    bytes[dst_index * element_size ..][0..element_size],
                    lhs.bytes[src_index * element_size ..][0..element_size],
                );
            } else {
                for (coords, 0..) |coord, axis| {
                    const src_coord = if (axis == concat_axis) coord - lhs_axis_len else coord;
                    src_index += src_coord * rhs_strides[axis];
                }
                @memcpy(
                    bytes[dst_index * element_size ..][0..element_size],
                    rhs.bytes[src_index * element_size ..][0..element_size],
                );
            }
        }

        var mlx_buffer: ?*mlx_metal.PjrtxMlxMetalBuffer = null;
        if (lhs.mlx_buffer != null and rhs.mlx_buffer != null) {
            mlx_buffer = mlx_metal.pjrtx_mlx_metal_buffer_concatenate(lhs.mlx_buffer.?, rhs.mlx_buffer.?, dimension, output_dims.ptr, output_dims.len);
            if (mlx_buffer == null) return error.MlxCommandSubmissionFailed;
        }
        errdefer if (mlx_buffer) |owned| mlx_metal.pjrtx_mlx_metal_buffer_destroy(owned);

        buffer.* = .{
            .allocator = allocator,
            .backend_kind = lhs.backend_kind,
            .element_type = lhs.element_type,
            .dims = dims,
            .device_id = lhs.device_id,
            .memory_id = lhs.memory_id,
            .device = lhs.device,
            .memory = lhs.memory,
            .shard_index = shard_index,
            .bytes = bytes,
            .mlx_buffer = mlx_buffer,
        };
        return buffer;
    }

    pub fn initU8Unary(
        allocator: std.mem.Allocator,
        op: ElementwiseUnaryOp,
        src: *Buffer,
        shard_index: usize,
    ) !*Buffer {
        if (src.element_type != .u8) return error.UnsupportedElementType;
        return initElementwiseUnary(allocator, op, src, shard_index);
    }

    pub fn deinit(self: *Buffer) void {
        if (self.mlx_buffer) |mlx_buffer| mlx_metal.pjrtx_mlx_metal_buffer_destroy(mlx_buffer);
        self.allocator.free(self.bytes);
        self.allocator.free(self.dims);
        self.allocator.destroy(self);
    }

    pub fn copyToHost(self: *Buffer, dst: []u8) !void {
        if (dst.len < self.bytes.len) return error.DestinationTooSmall;
        if (self.mlx_buffer) |mlx_buffer| {
            const ok = mlx_metal.pjrtx_mlx_metal_buffer_copy_to_host(mlx_buffer, dst.ptr, dst.len);
            if (ok == 0) return error.MlxBufferCopyFailed;
            return;
        }
        @memcpy(dst[0..self.bytes.len], self.bytes);
    }

    pub fn hasMlxStorage(self: *const Buffer) bool {
        return self.mlx_buffer != null;
    }
};

fn mlxBinaryOpCode(op: ElementwiseBinaryOp) c_int {
    return switch (op) {
        .add => mlx_metal.PJRTX_MLX_METAL_U8_BINARY_ADD,
        .subtract => mlx_metal.PJRTX_MLX_METAL_U8_BINARY_SUBTRACT,
        .multiply => mlx_metal.PJRTX_MLX_METAL_U8_BINARY_MULTIPLY,
        .divide => mlx_metal.PJRTX_MLX_METAL_U8_BINARY_DIVIDE,
    };
}

fn mlxUnaryOpCode(op: ElementwiseUnaryOp) c_int {
    return switch (op) {
        .negate => mlx_metal.PJRTX_MLX_METAL_U8_UNARY_NEGATE,
        .exp => mlx_metal.PJRTX_MLX_METAL_UNARY_EXP,
        .tanh => mlx_metal.PJRTX_MLX_METAL_UNARY_TANH,
        .sqrt => mlx_metal.PJRTX_MLX_METAL_UNARY_SQRT,
        .rsqrt => mlx_metal.PJRTX_MLX_METAL_UNARY_RSQRT,
    };
}

fn mlxDtype(element_type: BufferType) ?c_int {
    return switch (element_type) {
        .u8 => mlx_metal.PJRTX_MLX_METAL_DTYPE_U8,
        .f32 => mlx_metal.PJRTX_MLX_METAL_DTYPE_F32,
        else => null,
    };
}

fn denseByteSize(element_type: BufferType, dims: []const i64) usize {
    const element_size = element_type.byteSize();
    if (element_size == 0) return 0;
    var elements: usize = 1;
    for (dims) |dim| {
        if (dim < 0) return 0;
        elements = std.math.mul(usize, elements, @intCast(dim)) catch return 0;
    }
    return std.math.mul(usize, elements, element_size) catch 0;
}

fn validPermutation(permutation: []const i64, rank: usize) bool {
    if (permutation.len != rank) return false;
    var seen_storage = [_]bool{false} ** 16;
    if (rank > seen_storage.len) return false;
    const seen = seen_storage[0..rank];
    for (permutation) |axis| {
        if (axis < 0 or axis >= @as(i64, @intCast(rank))) return false;
        const axis_index: usize = @intCast(axis);
        if (seen[axis_index]) return false;
        seen[axis_index] = true;
    }
    return true;
}

fn validBroadcastDimensions(broadcast_dimensions: []const i64, input_dims: []const i64, output_dims: []const i64) bool {
    if (broadcast_dimensions.len != input_dims.len) return false;
    var seen_storage = [_]bool{false} ** 64;
    if (output_dims.len > seen_storage.len) return false;
    const seen = seen_storage[0..output_dims.len];
    for (broadcast_dimensions, 0..) |axis, input_axis| {
        if (axis < 0 or axis >= @as(i64, @intCast(output_dims.len))) return false;
        const output_axis: usize = @intCast(axis);
        if (seen[output_axis]) return false;
        seen[output_axis] = true;
        if (input_dims[input_axis] != 1 and input_dims[input_axis] != output_dims[output_axis]) return false;
    }
    return true;
}

fn validSlice(start_indices: []const i64, limit_indices: []const i64, strides: []const i64, input_dims: []const i64, output_dims: []const i64) bool {
    const rank = input_dims.len;
    if (start_indices.len != rank or limit_indices.len != rank or strides.len != rank or output_dims.len != rank) return false;
    for (0..rank) |axis| {
        const input_dim = input_dims[axis];
        const start = start_indices[axis];
        const limit = limit_indices[axis];
        const stride = strides[axis];
        const output_dim = output_dims[axis];
        if (input_dim < 0 or output_dim < 0) return false;
        if (start < 0 or limit < start or limit > input_dim or stride <= 0) return false;
        const span = limit - start;
        const expected = if (span == 0) 0 else @divTrunc(span + stride - 1, stride);
        if (output_dim != expected) return false;
    }
    return true;
}

fn validConcatenate(lhs_dims: []const i64, rhs_dims: []const i64, dimension: i64, output_dims: []const i64) bool {
    const rank = lhs_dims.len;
    if (rank == 0 or rhs_dims.len != rank or output_dims.len != rank) return false;
    if (dimension < 0 or dimension >= @as(i64, @intCast(rank))) return false;
    const concat_axis: usize = @intCast(dimension);
    for (0..rank) |axis| {
        if (lhs_dims[axis] < 0 or rhs_dims[axis] < 0 or output_dims[axis] < 0) return false;
        const expected = if (axis == concat_axis) lhs_dims[axis] + rhs_dims[axis] else lhs_dims[axis];
        if (output_dims[axis] != expected) return false;
        if (axis != concat_axis and lhs_dims[axis] != rhs_dims[axis]) return false;
    }
    return true;
}

fn rowMajorStrides(allocator: std.mem.Allocator, dims: []const i64) ![]usize {
    const strides = try allocator.alloc(usize, dims.len);
    var stride: usize = 1;
    var reverse_index = dims.len;
    while (reverse_index > 0) {
        reverse_index -= 1;
        strides[reverse_index] = stride;
        if (dims[reverse_index] < 0) return error.ShapeMismatch;
        stride = std.math.mul(usize, stride, @intCast(dims[reverse_index])) catch return error.ShapeMismatch;
    }
    return strides;
}

fn unravelIndex(index: usize, dims: []const i64, strides: []const usize, out: []usize) void {
    for (0..dims.len) |axis| {
        _ = dims[axis];
        out[axis] = index / strides[axis] % @as(usize, @intCast(dims[axis]));
    }
}

fn readF32LE(bytes: []const u8, index: usize) f32 {
    return @bitCast(std.mem.readInt(u32, bytes[index * 4 ..][0..4], .little));
}

test "synthetic client models multiple devices and memories" {
    const client = try Client.initSynthetic(std.testing.allocator, 4);
    defer client.deinit();

    try std.testing.expectEqual(@as(usize, 4), client.devices.len);
    try std.testing.expectEqual(BackendKind.synthetic, client.backend_kind);
    try std.testing.expectEqual(@as(usize, 4), client.memories.len);
    try std.testing.expectEqual(@as(usize, 4), client.device_handles.len);
    try std.testing.expectEqual(@as(usize, 4), client.memory_handles.len);
    try std.testing.expectEqual(@as(i32, 4), client.topology.num_partitions);
    try std.testing.expectEqual(@as(usize, 4), client.topology.numDevices());
    try std.testing.expect(client.lookupDevice(2) != null);
    try std.testing.expect(client.lookupMemory(3) != null);
    try std.testing.expectEqual(@as(i32, 1), client.device_handles[1].id);
    try std.testing.expectEqual(@as(i32, 1), client.memory_handles[1].id);
    try std.testing.expectEqual(@as(i32, 2), client.devices[2].default_memory.id);
    try std.testing.expectEqual(@as(i32, 3), client.memories[3].addressable_devices[0].id);
}

test "metal mlx bootstrap client uses explicit backend kind" {
    const client = try Client.initMetalMlxBootstrap(std.testing.allocator);
    defer client.deinit();

    try std.testing.expectEqual(BackendKind.metal_mlx, client.backend_kind);
    try std.testing.expectEqual(@as(usize, 1), client.devices.len);
    try std.testing.expect(client.devices[0].name.len != 0);
    try std.testing.expect(std.mem.startsWith(u8, client.devices[0].debug_string, "PjRTx Metal/MLX device"));
}

test "metal mlx bootstrap accepts copied C device metadata" {
    var metal_devices = [_]mlx_metal.PjrtxMlxMetalDeviceInfo{
        .{
            .ordinal = 7,
            .registry_id = 42,
            .recommended_max_working_set_size = 4096,
            .has_unified_memory = 1,
            .name = undefined,
        },
        .{
            .ordinal = 8,
            .registry_id = 43,
            .recommended_max_working_set_size = 8192,
            .has_unified_memory = 0,
            .name = undefined,
        },
    };
    fillCName(&metal_devices[0].name, "Synthetic MTL A");
    fillCName(&metal_devices[1].name, "Synthetic MTL B");

    const client = try Client.initMetalMlxFromDevices(std.testing.allocator, &metal_devices);
    defer client.deinit();

    try std.testing.expectEqual(BackendKind.metal_mlx, client.backend_kind);
    try std.testing.expectEqual(@as(usize, 2), client.devices.len);
    try std.testing.expectEqualStrings("Synthetic MTL A", client.devices[0].name);
    try std.testing.expectEqual(@as(i32, 7), client.devices[0].local_hardware_id);
    try std.testing.expectEqual(@as(u64, 42), client.devices[0].registry_id);
    try std.testing.expectEqual(@as(u64, 4096), client.devices[0].memory_bytes);
    try std.testing.expect(client.devices[0].has_unified_memory);
    try std.testing.expectEqualStrings("Synthetic MTL B", client.devices[1].name);
    try std.testing.expectEqual(@as(i32, 8), client.devices[1].local_hardware_id);
}

test "buffer keeps shard/device/memory ownership metadata" {
    const client = try Client.initSynthetic(std.testing.allocator, 2);
    defer client.deinit();

    const dims = [_]i64{ 2, 2 };
    const data = [_]u8{ 1, 2, 3, 4 };
    const buffer = try Buffer.initHostCopy(std.testing.allocator, .u8, &dims, &client.devices[1], &client.memories[1], 1, &data);
    defer buffer.deinit();

    try std.testing.expectEqual(@as(i32, 1), buffer.device_id);
    try std.testing.expectEqual(@as(i32, 1), buffer.memory_id);
    try std.testing.expectEqual(@as(i32, 1), buffer.device.id);
    try std.testing.expectEqual(@as(i32, 1), buffer.memory.id);
    try std.testing.expectEqual(@as(usize, 1), buffer.shard_index);
    try std.testing.expectEqualSlices(u8, &data, buffer.bytes);

    const rhs_data = [_]u8{ 10, 20, 30, 40 };
    const rhs = try Buffer.initHostCopy(std.testing.allocator, .u8, &dims, &client.devices[1], &client.memories[1], 1, &rhs_data);
    defer rhs.deinit();
    const sum = try Buffer.initU8Add(std.testing.allocator, buffer, rhs, 1);
    defer sum.deinit();
    try std.testing.expectEqualSlices(u8, &.{ 11, 22, 33, 44 }, sum.bytes);

    const difference = try Buffer.initU8Binary(std.testing.allocator, .subtract, rhs, buffer, 1);
    defer difference.deinit();
    try std.testing.expectEqualSlices(u8, &.{ 9, 18, 27, 36 }, difference.bytes);

    const product = try Buffer.initU8Binary(std.testing.allocator, .multiply, buffer, rhs, 1);
    defer product.deinit();
    try std.testing.expectEqualSlices(u8, &.{ 10, 40, 90, 160 }, product.bytes);

    const quotient = try Buffer.initU8Binary(std.testing.allocator, .divide, rhs, buffer, 1);
    defer quotient.deinit();
    try std.testing.expectEqualSlices(u8, &.{ 10, 10, 10, 10 }, quotient.bytes);

    const negated = try Buffer.initU8Unary(std.testing.allocator, .negate, buffer, 1);
    defer negated.deinit();
    try std.testing.expectEqualSlices(u8, &.{ 255, 254, 253, 252 }, negated.bytes);
}

test "buffer elementwise arithmetic supports f32 host execution" {
    const client = try Client.initSynthetic(std.testing.allocator, 1);
    defer client.deinit();

    const dims = [_]i64{2};
    const lhs_values = [_]f32{ 1.5, -2.0 };
    const rhs_values = [_]f32{ 2.25, 4.0 };
    const lhs = try Buffer.initHostCopy(std.testing.allocator, .f32, &dims, &client.devices[0], &client.memories[0], 0, std.mem.asBytes(&lhs_values));
    defer lhs.deinit();
    const rhs = try Buffer.initHostCopy(std.testing.allocator, .f32, &dims, &client.devices[0], &client.memories[0], 0, std.mem.asBytes(&rhs_values));
    defer rhs.deinit();

    const sum = try Buffer.initElementwiseBinary(std.testing.allocator, .add, lhs, rhs, 0);
    defer sum.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 3.75), readF32LE(sum.bytes, 0), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), readF32LE(sum.bytes, 1), 0.0001);

    const quotient = try Buffer.initElementwiseBinary(std.testing.allocator, .divide, rhs, lhs, 0);
    defer quotient.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), readF32LE(quotient.bytes, 0), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, -2.0), readF32LE(quotient.bytes, 1), 0.0001);

    const negated = try Buffer.initElementwiseUnary(std.testing.allocator, .negate, lhs, 0);
    defer negated.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, -1.5), readF32LE(negated.bytes, 0), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), readF32LE(negated.bytes, 1), 0.0001);
}

test "buffer elementwise unary math supports f32 host execution" {
    const client = try Client.initSynthetic(std.testing.allocator, 1);
    defer client.deinit();

    const dims = [_]i64{2};
    const values = [_]f32{ 1.0, 4.0 };
    const input = try Buffer.initHostCopy(std.testing.allocator, .f32, &dims, &client.devices[0], &client.memories[0], 0, std.mem.asBytes(&values));
    defer input.deinit();

    const exp = try Buffer.initElementwiseUnary(std.testing.allocator, .exp, input, 0);
    defer exp.deinit();
    try std.testing.expectApproxEqAbs(std.math.exp(@as(f32, 1.0)), readF32LE(exp.bytes, 0), 0.0001);
    try std.testing.expectApproxEqAbs(std.math.exp(@as(f32, 4.0)), readF32LE(exp.bytes, 1), 0.0001);

    const tanh = try Buffer.initElementwiseUnary(std.testing.allocator, .tanh, input, 0);
    defer tanh.deinit();
    try std.testing.expectApproxEqAbs(std.math.tanh(@as(f32, 1.0)), readF32LE(tanh.bytes, 0), 0.0001);
    try std.testing.expectApproxEqAbs(std.math.tanh(@as(f32, 4.0)), readF32LE(tanh.bytes, 1), 0.0001);

    const sqrt = try Buffer.initElementwiseUnary(std.testing.allocator, .sqrt, input, 0);
    defer sqrt.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), readF32LE(sqrt.bytes, 0), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), readF32LE(sqrt.bytes, 1), 0.0001);

    const rsqrt = try Buffer.initElementwiseUnary(std.testing.allocator, .rsqrt, input, 0);
    defer rsqrt.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), readF32LE(rsqrt.bytes, 0), 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), readF32LE(rsqrt.bytes, 1), 0.0001);
}

test "buffer reshape preserves typed bytes and updates dimensions" {
    const client = try Client.initSynthetic(std.testing.allocator, 1);
    defer client.deinit();

    const dims = [_]i64{ 2, 2 };
    const values = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const input = try Buffer.initHostCopy(std.testing.allocator, .f32, &dims, &client.devices[0], &client.memories[0], 0, std.mem.asBytes(&values));
    defer input.deinit();

    const reshaped = try Buffer.initReshape(std.testing.allocator, input, &.{4}, 0);
    defer reshaped.deinit();

    try std.testing.expectEqualSlices(i64, &.{4}, reshaped.dims);
    try std.testing.expectEqualSlices(u8, input.bytes, reshaped.bytes);
}

test "buffer transpose permutes dense host bytes and updates dimensions" {
    const client = try Client.initSynthetic(std.testing.allocator, 1);
    defer client.deinit();

    const dims = [_]i64{ 2, 3 };
    const values = [_]u8{ 1, 2, 3, 4, 5, 6 };
    const input = try Buffer.initHostCopy(std.testing.allocator, .u8, &dims, &client.devices[0], &client.memories[0], 0, &values);
    defer input.deinit();

    const transposed = try Buffer.initTranspose(std.testing.allocator, input, &.{ 1, 0 }, &.{ 3, 2 }, 0);
    defer transposed.deinit();

    try std.testing.expectEqualSlices(i64, &.{ 3, 2 }, transposed.dims);
    try std.testing.expectEqualSlices(u8, &.{ 1, 4, 2, 5, 3, 6 }, transposed.bytes);
}

test "buffer broadcast_in_dim expands dense host bytes and updates dimensions" {
    const client = try Client.initSynthetic(std.testing.allocator, 1);
    defer client.deinit();

    const dims = [_]i64{3};
    const values = [_]u8{ 7, 8, 9 };
    const input = try Buffer.initHostCopy(std.testing.allocator, .u8, &dims, &client.devices[0], &client.memories[0], 0, &values);
    defer input.deinit();

    const broadcasted = try Buffer.initBroadcastInDim(std.testing.allocator, input, &.{1}, &.{ 2, 3 }, 0);
    defer broadcasted.deinit();

    try std.testing.expectEqualSlices(i64, &.{ 2, 3 }, broadcasted.dims);
    try std.testing.expectEqualSlices(u8, &.{ 7, 8, 9, 7, 8, 9 }, broadcasted.bytes);
}

test "buffer slice copies strided dense host bytes and updates dimensions" {
    const client = try Client.initSynthetic(std.testing.allocator, 1);
    defer client.deinit();

    const dims = [_]i64{ 3, 4 };
    const values = [_]u8{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const input = try Buffer.initHostCopy(std.testing.allocator, .u8, &dims, &client.devices[0], &client.memories[0], 0, &values);
    defer input.deinit();

    const sliced = try Buffer.initSlice(
        std.testing.allocator,
        input,
        &.{ 1, 0 },
        &.{ 3, 4 },
        &.{ 1, 2 },
        &.{ 2, 2 },
        0,
    );
    defer sliced.deinit();

    try std.testing.expectEqualSlices(i64, &.{ 2, 2 }, sliced.dims);
    try std.testing.expectEqualSlices(u8, &.{ 5, 7, 9, 11 }, sliced.bytes);
}

test "buffer concatenate joins dense host bytes along an axis" {
    const client = try Client.initSynthetic(std.testing.allocator, 1);
    defer client.deinit();

    const lhs_dims = [_]i64{ 2, 2 };
    const rhs_dims = [_]i64{ 2, 3 };
    const lhs_values = [_]u8{ 1, 2, 3, 4 };
    const rhs_values = [_]u8{ 5, 6, 7, 8, 9, 10 };
    const lhs = try Buffer.initHostCopy(std.testing.allocator, .u8, &lhs_dims, &client.devices[0], &client.memories[0], 0, &lhs_values);
    defer lhs.deinit();
    const rhs = try Buffer.initHostCopy(std.testing.allocator, .u8, &rhs_dims, &client.devices[0], &client.memories[0], 0, &rhs_values);
    defer rhs.deinit();

    const concatenated = try Buffer.initConcatenate(std.testing.allocator, lhs, rhs, 1, &.{ 2, 5 }, 0);
    defer concatenated.deinit();

    try std.testing.expectEqualSlices(i64, &.{ 2, 5 }, concatenated.dims);
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 5, 6, 7, 3, 4, 8, 9, 10 }, concatenated.bytes);
}

test "metal mlx buffer owns mlx handle when backend is available" {
    const client = try Client.initMetalMlxBootstrap(std.testing.allocator);
    defer client.deinit();

    const dims = [_]i64{4};
    const data = [_]u8{ 9, 8, 7, 6 };
    const buffer = Buffer.initHostCopyForBackend(std.testing.allocator, .metal_mlx, .u8, &dims, &client.devices[0], &client.memories[0], 0, &data) catch |err| switch (err) {
        error.MlxBufferAllocationFailed => return error.SkipZigTest,
        else => return err,
    };
    defer buffer.deinit();

    try std.testing.expect(buffer.hasMlxStorage());
    var output: [4]u8 = undefined;
    try buffer.copyToHost(&output);
    try std.testing.expectEqualSlices(u8, &data, &output);

    const cloned = try Buffer.initDeviceCopy(std.testing.allocator, buffer, 0);
    defer cloned.deinit();
    try std.testing.expect(cloned.hasMlxStorage());
    var cloned_output: [4]u8 = undefined;
    try cloned.copyToHost(&cloned_output);
    try std.testing.expectEqualSlices(u8, &data, &cloned_output);

    const rhs_data = [_]u8{ 1, 2, 3, 4 };
    const rhs = Buffer.initHostCopyForBackend(std.testing.allocator, .metal_mlx, .u8, &dims, &client.devices[0], &client.memories[0], 0, &rhs_data) catch |err| switch (err) {
        error.MlxBufferAllocationFailed => return error.SkipZigTest,
        else => return err,
    };
    defer rhs.deinit();

    const sum = Buffer.initU8Add(std.testing.allocator, buffer, rhs, 0) catch |err| switch (err) {
        error.MlxCommandSubmissionFailed => return error.SkipZigTest,
        else => return err,
    };
    defer sum.deinit();
    try std.testing.expect(sum.hasMlxStorage());
    var sum_output: [4]u8 = undefined;
    try sum.copyToHost(&sum_output);
    try std.testing.expectEqualSlices(u8, &.{ 10, 10, 10, 10 }, &sum_output);
}
