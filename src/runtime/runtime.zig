const std = @import("std");
const backend_api = @import("src/backend");
const core = @import("src/core");

pub const MAX_DEVICES = core.MAX_DEVICES;
pub const MemoryKind = core.MemoryKind;
pub const BackendKind = core.BackendKind;
pub const BufferType = core.BufferType;
pub const ElementwiseBinaryOp = core.ElementwiseBinaryOp;
pub const ElementwiseUnaryOp = core.ElementwiseUnaryOp;
pub const CompareOp = core.CompareOp;

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
    backend: backend_api.Backend,
    backend_kind: BackendKind,
    devices: []Device,
    memories: []Memory,
    device_handles: []*Device,
    memory_handles: []*Memory,
    topology: Topology,

    pub fn init(allocator: std.mem.Allocator, backend_impl: backend_api.Backend, device_count: usize) !*Client {
        if (device_count == 0 or device_count > MAX_DEVICES) return error.InvalidDeviceCount;
        const backend_kind = backend_impl.kind();

        const client = try allocator.create(Client);
        errdefer allocator.destroy(client);

        const descriptors = try backend_impl.enumerateDevices(allocator, device_count);
        defer backend_impl.releaseDeviceDescriptors(allocator, descriptors);
        if (descriptors.len == 0 or descriptors.len > MAX_DEVICES) return error.InvalidDeviceCount;

        const devices = try allocator.alloc(Device, descriptors.len);
        errdefer allocator.free(devices);

        const memories = try allocator.alloc(Memory, descriptors.len);
        errdefer allocator.free(memories);

        const device_handles = try allocator.alloc(*Device, descriptors.len);
        errdefer allocator.free(device_handles);

        const memory_handles = try allocator.alloc(*Memory, descriptors.len);
        errdefer allocator.free(memory_handles);

        const assignment = try allocator.alloc(i32, descriptors.len);
        errdefer allocator.free(assignment);

        for (descriptors, 0..) |descriptor, i| {
            const name = try allocator.dupe(u8, descriptor.name);
            errdefer allocator.free(name);

            const debug_string = try allocator.dupe(u8, descriptor.debug_string);
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
                .id = descriptor.id,
                .local_hardware_id = descriptor.local_hardware_id,
                .registry_id = descriptor.registry_id,
                .process_index = descriptor.process_index,
                .addressable = descriptor.addressable,
                .name = name,
                .debug_string = debug_string,
                .memory_bytes = descriptor.memory_bytes,
                .has_unified_memory = descriptor.has_unified_memory,
                .default_memory_id = descriptor.default_memory_id,
                .default_memory = &memories[i],
                .addressable_memories = device_memories,
            };
            ids[0] = descriptor.id;
            memories[i] = .{
                .id = descriptor.default_memory_id,
                .kind = .device,
                .debug_string = memory_debug_string,
                .addressable_device_ids = ids,
                .addressable_devices = memory_devices,
            };
            devices[i].addressable_memories[0] = &memories[i];
            memories[i].addressable_devices[0] = &devices[i];
            device_handles[i] = &devices[i];
            memory_handles[i] = &memories[i];
            assignment[i] = descriptor.id;
        }

        client.* = .{
            .allocator = allocator,
            .backend = backend_impl,
            .backend_kind = backend_kind,
            .devices = devices,
            .memories = memories,
            .device_handles = device_handles,
            .memory_handles = memory_handles,
            .topology = .{
                .device_assignment = assignment,
                .num_replicas = 1,
                .num_partitions = @intCast(descriptors.len),
            },
        };
        return client;
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

pub const Buffer = struct {
    allocator: std.mem.Allocator,
    backend: backend_api.Backend,
    backend_kind: BackendKind,
    element_type: BufferType,
    dims: []i64,
    device_id: i32,
    memory_id: i32,
    device: *Device,
    memory: *Memory,
    shard_index: usize,
    bytes: []u8,
    backend_buffer: ?backend_api.BufferHandle = null,
    deleted: bool = false,

    pub fn initHostCopyForBackend(
        allocator: std.mem.Allocator,
        backend_impl: backend_api.Backend,
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

        const backend_buffer = try backend_impl.bufferFromHost(device.local_hardware_id, element_type, dims, src);
        errdefer if (backend_buffer) |owned| backend_impl.destroyBuffer(owned);

        buffer.* = .{
            .allocator = allocator,
            .backend = backend_impl,
            .backend_kind = backend_impl.kind(),
            .element_type = element_type,
            .dims = dims,
            .device_id = device.id,
            .memory_id = memory.id,
            .device = device,
            .memory = memory,
            .shard_index = shard_index,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
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

        var backend_buffer: ?backend_api.BufferHandle = null;
        if (src.backend_buffer) |src_backend| {
            backend_buffer = try src.backend.cloneBuffer(src_backend);
        }
        errdefer if (backend_buffer) |owned| src.backend.destroyBuffer(owned);

        buffer.* = .{
            .allocator = allocator,
            .backend = src.backend,
            .backend_kind = src.backend_kind,
            .element_type = src.element_type,
            .dims = dims,
            .device_id = src.device_id,
            .memory_id = src.memory_id,
            .device = src.device,
            .memory = src.memory,
            .shard_index = shard_index,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
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
        if (lhs.element_type.byteSize() == 0) return error.UnsupportedElementType;
        if (!std.mem.eql(i64, lhs.dims, rhs.dims) or lhs.bytes.len != rhs.bytes.len) return error.ShapeMismatch;

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, lhs.dims);
        errdefer allocator.free(dims);

        const bytes = try allocator.alloc(u8, lhs.bytes.len);
        errdefer allocator.free(bytes);
        switch (lhs.element_type) {
            .pred => {
                for (lhs.bytes, rhs.bytes, 0..) |a, b, i| {
                    bytes[i] = switch (op) {
                        .and_ => if (a != 0 and b != 0) 1 else 0,
                        .or_ => if (a != 0 or b != 0) 1 else 0,
                        .xor => if ((a != 0) != (b != 0)) 1 else 0,
                        else => return error.UnsupportedElementType,
                    };
                }
            },
            .u8 => {
                for (lhs.bytes, rhs.bytes, 0..) |a, b, i| {
                    bytes[i] = switch (op) {
                        .add => a +% b,
                        .subtract => a -% b,
                        .multiply => a *% b,
                        .divide => if (b == 0) 0 else a / b,
                        .maximum => @max(a, b),
                        .minimum => @min(a, b),
                        .remainder => if (b == 0) 0 else a % b,
                        .and_ => a & b,
                        .or_ => a | b,
                        .xor => a ^ b,
                        .shift_left => a << @intCast(@min(b, 7)),
                        .shift_right_logical, .shift_right_arithmetic => a >> @intCast(@min(b, 7)),
                        else => return error.UnsupportedElementType,
                    };
                }
            },
            .s32 => {
                const element_count = lhs.bytes.len / 4;
                for (0..element_count) |i| {
                    const a = readI32LE(lhs.bytes, i);
                    const b = readI32LE(rhs.bytes, i);
                    const value: i32 = switch (op) {
                        .add => a +% b,
                        .subtract => a -% b,
                        .multiply => a *% b,
                        .divide => if (b == 0) 0 else @divTrunc(a, b),
                        .maximum => @max(a, b),
                        .minimum => @min(a, b),
                        .remainder => if (b == 0) 0 else @rem(a, b),
                        .and_ => a & b,
                        .or_ => a | b,
                        .xor => a ^ b,
                        .shift_left => a << @intCast(@min(@as(u5, @intCast(@max(b, 0))), 31)),
                        .shift_right_arithmetic => a >> @intCast(@min(@as(u5, @intCast(@max(b, 0))), 31)),
                        .shift_right_logical => @bitCast(@as(u32, @bitCast(a)) >> @intCast(@min(@as(u5, @intCast(@max(b, 0))), 31))),
                        else => return error.UnsupportedElementType,
                    };
                    writeI32LE(bytes, i, value);
                }
            },
            .u32 => {
                const element_count = lhs.bytes.len / 4;
                for (0..element_count) |i| {
                    const a = readU32LE(lhs.bytes, i);
                    const b = readU32LE(rhs.bytes, i);
                    const shift: u5 = @intCast(@min(b, 31));
                    const value: u32 = switch (op) {
                        .add => a +% b,
                        .subtract => a -% b,
                        .multiply => a *% b,
                        .divide => if (b == 0) 0 else a / b,
                        .maximum => @max(a, b),
                        .minimum => @min(a, b),
                        .remainder => if (b == 0) 0 else a % b,
                        .and_ => a & b,
                        .or_ => a | b,
                        .xor => a ^ b,
                        .shift_left => a << shift,
                        .shift_right_logical, .shift_right_arithmetic => a >> shift,
                        else => return error.UnsupportedElementType,
                    };
                    writeU32LE(bytes, i, value);
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
                        .maximum => @max(a, b),
                        .minimum => @min(a, b),
                        .power => std.math.pow(f32, a, b),
                        .atan2 => std.math.atan2(a, b),
                        .remainder => @mod(a, b),
                        else => return error.UnsupportedElementType,
                    };
                    std.mem.writeInt(u32, bytes[offset..][0..4], @bitCast(value), .little);
                }
            },
            else => return error.UnsupportedElementType,
        }

        var backend_buffer: ?backend_api.BufferHandle = null;
        if (lhs.backend_buffer != null and rhs.backend_buffer != null) {
            backend_buffer = lhs.backend.binary(lhs.backend_buffer.?, rhs.backend_buffer.?, op) catch null;
        }
        errdefer if (backend_buffer) |owned| lhs.backend.destroyBuffer(owned);

        buffer.* = .{
            .allocator = allocator,
            .backend = lhs.backend,
            .backend_kind = lhs.backend_kind,
            .element_type = lhs.element_type,
            .dims = dims,
            .device_id = lhs.device_id,
            .memory_id = lhs.memory_id,
            .device = lhs.device,
            .memory = lhs.memory,
            .shard_index = shard_index,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
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
        return initElementwiseUnaryTyped(allocator, op, src, src.element_type, src.dims, shard_index);
    }

    pub fn initElementwiseUnaryTyped(
        allocator: std.mem.Allocator,
        op: ElementwiseUnaryOp,
        src: *Buffer,
        output_type: BufferType,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (!std.mem.eql(i64, src.dims, output_dims)) return error.ShapeMismatch;
        if (src.element_type.byteSize() == 0 or output_type.byteSize() == 0) return error.UnsupportedElementType;

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, output_dims);
        errdefer allocator.free(dims);

        const bytes = try allocator.alloc(u8, denseByteSize(output_type, output_dims));
        errdefer allocator.free(bytes);
        switch (src.element_type) {
            .pred => {
                if (output_type != .pred or op != .not_) return error.UnsupportedElementType;
                for (src.bytes, 0..) |value, i| bytes[i] = if (value == 0) 1 else 0;
            },
            .u8 => {
                if (output_type != .u8 and output_type != .s8 and output_type != .pred) return error.UnsupportedElementType;
                for (src.bytes, 0..) |value, i| {
                    _ = switch (op) {
                        .negate => bytes[i] = 0 -% value,
                        .not_ => bytes[i] = ~value,
                        .popcnt => bytes[i] = @popCount(value),
                        .count_leading_zeros => bytes[i] = @clz(value),
                        .sign => bytes[i] = if (value == 0) 0 else 1,
                        .is_finite => bytes[i] = 1,
                        else => return error.UnsupportedElementType,
                    };
                }
            },
            .s32 => {
                if (output_type != .s32 and output_type != .pred) return error.UnsupportedElementType;
                const element_count = src.bytes.len / 4;
                for (0..element_count) |i| {
                    const value = readI32LE(src.bytes, i);
                    _ = switch (op) {
                        .negate => writeI32LE(bytes, i, -%value),
                        .not_ => writeI32LE(bytes, i, ~value),
                        .popcnt => writeI32LE(bytes, i, @intCast(@popCount(value))),
                        .count_leading_zeros => writeI32LE(bytes, i, @intCast(@clz(value))),
                        .sign => writeI32LE(bytes, i, if (value < 0) -1 else if (value > 0) 1 else 0),
                        .is_finite => bytes[i] = 1,
                        else => return error.UnsupportedElementType,
                    };
                }
            },
            .u32 => {
                if (output_type != .u32 and output_type != .pred) return error.UnsupportedElementType;
                const element_count = src.bytes.len / 4;
                for (0..element_count) |i| {
                    const value = readU32LE(src.bytes, i);
                    _ = switch (op) {
                        .not_ => writeU32LE(bytes, i, ~value),
                        .popcnt => writeU32LE(bytes, i, @intCast(@popCount(value))),
                        .count_leading_zeros => writeU32LE(bytes, i, @intCast(@clz(value))),
                        .sign => writeU32LE(bytes, i, if (value == 0) 0 else 1),
                        .is_finite => bytes[i] = 1,
                        else => return error.UnsupportedElementType,
                    };
                }
            },
            .f32 => {
                if (output_type != .f32 and output_type != .pred) return error.UnsupportedElementType;
                var offset: usize = 0;
                var index: usize = 0;
                while (offset < src.bytes.len) : (offset += 4) {
                    const bits = std.mem.readInt(u32, src.bytes[offset..][0..4], .little);
                    const value: f32 = @bitCast(bits);
                    if (op == .is_finite) {
                        bytes[index] = if (std.math.isFinite(value)) 1 else 0;
                        index += 1;
                    } else {
                        const out: f32 = switch (op) {
                            .negate => -value,
                            .exp => std.math.exp(value),
                            .expm1 => std.math.exp(value) - 1.0,
                            .tanh => std.math.tanh(value),
                            .sqrt => std.math.sqrt(value),
                            .rsqrt => 1.0 / std.math.sqrt(value),
                            .abs => @abs(value),
                            .cbrt => std.math.cbrt(value),
                            .ceil => @ceil(value),
                            .floor => @floor(value),
                            .log => @log(value),
                            .log1p => @log(value + 1.0),
                            .logistic => 1.0 / (1.0 + std.math.exp(-value)),
                            .sine => @sin(value),
                            .cosine => @cos(value),
                            .sign => if (value < 0.0) -1.0 else if (value > 0.0) 1.0 else 0.0,
                            .round_nearest_afz => @round(value),
                            .round_nearest_even => @round(value),
                            else => return error.UnsupportedElementType,
                        };
                        std.mem.writeInt(u32, bytes[offset..][0..4], @bitCast(out), .little);
                    }
                }
            },
            else => return error.UnsupportedElementType,
        }

        var backend_buffer: ?backend_api.BufferHandle = null;
        if (src.backend_buffer) |src_backend| {
            backend_buffer = src.backend.unary(src_backend, op) catch null;
        }
        errdefer if (backend_buffer) |owned| src.backend.destroyBuffer(owned);

        buffer.* = .{
            .allocator = allocator,
            .backend = src.backend,
            .backend_kind = src.backend_kind,
            .element_type = output_type,
            .dims = dims,
            .device_id = src.device_id,
            .memory_id = src.memory_id,
            .device = src.device,
            .memory = src.memory,
            .shard_index = shard_index,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
        };
        return buffer;
    }

    pub fn initReshape(
        allocator: std.mem.Allocator,
        src: *Buffer,
        new_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (src.element_type.byteSize() == 0) return error.UnsupportedElementType;
        if (denseByteSize(src.element_type, new_dims) != src.bytes.len) return error.ShapeMismatch;

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, new_dims);
        errdefer allocator.free(dims);

        const bytes = try allocator.dupe(u8, src.bytes);
        errdefer allocator.free(bytes);

        const backend_buffer = src.backend.reshape(src.device.local_hardware_id, src.element_type, src.bytes, new_dims) catch null;
        errdefer if (backend_buffer) |owned| src.backend.destroyBuffer(owned);

        buffer.* = .{
            .allocator = allocator,
            .backend = src.backend,
            .backend_kind = src.backend_kind,
            .element_type = src.element_type,
            .dims = dims,
            .device_id = src.device_id,
            .memory_id = src.memory_id,
            .device = src.device,
            .memory = src.memory,
            .shard_index = shard_index,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
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

        var backend_buffer: ?backend_api.BufferHandle = null;
        if (src.backend_buffer) |src_backend| {
            backend_buffer = src.backend.transpose(src_backend, permutation) catch null;
        }
        errdefer if (backend_buffer) |owned| src.backend.destroyBuffer(owned);

        buffer.* = .{
            .allocator = allocator,
            .backend = src.backend,
            .backend_kind = src.backend_kind,
            .element_type = src.element_type,
            .dims = dims,
            .device_id = src.device_id,
            .memory_id = src.memory_id,
            .device = src.device,
            .memory = src.memory,
            .shard_index = shard_index,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
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

        var backend_buffer: ?backend_api.BufferHandle = null;
        if (src.backend_buffer) |src_backend| {
            backend_buffer = src.backend.broadcastInDim(src_backend, broadcast_dimensions, output_dims) catch null;
        }
        errdefer if (backend_buffer) |owned| src.backend.destroyBuffer(owned);

        buffer.* = .{
            .allocator = allocator,
            .backend = src.backend,
            .backend_kind = src.backend_kind,
            .element_type = src.element_type,
            .dims = dims,
            .device_id = src.device_id,
            .memory_id = src.memory_id,
            .device = src.device,
            .memory = src.memory,
            .shard_index = shard_index,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
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

        var backend_buffer: ?backend_api.BufferHandle = null;
        if (src.backend_buffer) |src_backend| {
            backend_buffer = src.backend.slice(src_backend, start_indices, limit_indices, strides_, output_dims) catch null;
        }
        errdefer if (backend_buffer) |owned| src.backend.destroyBuffer(owned);

        buffer.* = .{
            .allocator = allocator,
            .backend = src.backend,
            .backend_kind = src.backend_kind,
            .element_type = src.element_type,
            .dims = dims,
            .device_id = src.device_id,
            .memory_id = src.memory_id,
            .device = src.device,
            .memory = src.memory,
            .shard_index = shard_index,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
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

        var backend_buffer: ?backend_api.BufferHandle = null;
        if (lhs.backend_buffer != null and rhs.backend_buffer != null) {
            backend_buffer = lhs.backend.concatenate(lhs.backend_buffer.?, rhs.backend_buffer.?, dimension, output_dims) catch null;
        }
        errdefer if (backend_buffer) |owned| lhs.backend.destroyBuffer(owned);

        buffer.* = .{
            .allocator = allocator,
            .backend = lhs.backend,
            .backend_kind = lhs.backend_kind,
            .element_type = lhs.element_type,
            .dims = dims,
            .device_id = lhs.device_id,
            .memory_id = lhs.memory_id,
            .device = lhs.device,
            .memory = lhs.memory,
            .shard_index = shard_index,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
        };
        return buffer;
    }

    pub fn initDotGeneral(
        allocator: std.mem.Allocator,
        lhs: *Buffer,
        rhs: *Buffer,
        lhs_batch_dimensions: []const i64,
        rhs_batch_dimensions: []const i64,
        lhs_contracting_dimensions: []const i64,
        rhs_contracting_dimensions: []const i64,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (lhs.element_type != .f32 or rhs.element_type != .f32) return error.UnsupportedElementType;
        if (!validDotGeneral(lhs.dims, rhs.dims, lhs_batch_dimensions, rhs_batch_dimensions, lhs_contracting_dimensions, rhs_contracting_dimensions, output_dims)) return error.ShapeMismatch;

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, output_dims);
        errdefer allocator.free(dims);

        const output_byte_size = denseByteSize(.f32, output_dims);
        const bytes = try allocator.alloc(u8, output_byte_size);
        errdefer allocator.free(bytes);

        try evalDotGeneralF32(allocator, lhs.bytes, lhs.dims, rhs.bytes, rhs.dims, lhs_batch_dimensions, rhs_batch_dimensions, lhs_contracting_dimensions, rhs_contracting_dimensions, output_dims, bytes);

        var backend_buffer: ?backend_api.BufferHandle = null;
        if (lhs.backend_buffer != null and rhs.backend_buffer != null) {
            backend_buffer = lhs.backend.dotGeneral(lhs.backend_buffer.?, rhs.backend_buffer.?, lhs_batch_dimensions, rhs_batch_dimensions, lhs_contracting_dimensions, rhs_contracting_dimensions, output_dims) catch null;
        }
        errdefer if (backend_buffer) |owned| lhs.backend.destroyBuffer(owned);

        buffer.* = .{
            .allocator = allocator,
            .backend = lhs.backend,
            .backend_kind = lhs.backend_kind,
            .element_type = .f32,
            .dims = dims,
            .device_id = lhs.device_id,
            .memory_id = lhs.memory_id,
            .device = lhs.device,
            .memory = lhs.memory,
            .shard_index = shard_index,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
        };
        return buffer;
    }

    pub fn initReduce(
        allocator: std.mem.Allocator,
        kind: core.PlanInstructionKind,
        src: *Buffer,
        dimensions: []const i64,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (src.element_type != .f32) return error.UnsupportedElementType;
        if (kind != .reduce_sum and kind != .reduce_max) return error.UnsupportedElementType;
        if (!validReduce(src.dims, dimensions, output_dims)) return error.ShapeMismatch;

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, output_dims);
        errdefer allocator.free(dims);

        const output_byte_size = denseByteSize(.f32, output_dims);
        const bytes = try allocator.alloc(u8, output_byte_size);
        errdefer allocator.free(bytes);

        try evalReduceF32(allocator, kind, src.bytes, src.dims, dimensions, output_dims, bytes);

        var backend_buffer: ?backend_api.BufferHandle = null;
        if (src.backend_buffer) |src_backend| {
            backend_buffer = src.backend.reduce(src_backend, kind, dimensions, output_dims) catch null;
        }
        errdefer if (backend_buffer) |owned| src.backend.destroyBuffer(owned);

        buffer.* = .{
            .allocator = allocator,
            .backend = src.backend,
            .backend_kind = src.backend_kind,
            .element_type = .f32,
            .dims = dims,
            .device_id = src.device_id,
            .memory_id = src.memory_id,
            .device = src.device,
            .memory = src.memory,
            .shard_index = shard_index,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
        };
        return buffer;
    }

    pub fn initCompare(
        allocator: std.mem.Allocator,
        lhs: *Buffer,
        rhs: *Buffer,
        direction: CompareOp,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (lhs.element_type != rhs.element_type) return error.UnsupportedElementType;
        if (!std.mem.eql(i64, lhs.dims, rhs.dims) or !std.mem.eql(i64, lhs.dims, output_dims)) return error.ShapeMismatch;

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, output_dims);
        errdefer allocator.free(dims);

        const element_size = lhs.element_type.byteSize();
        if (element_size == 0) return error.UnsupportedElementType;
        const element_count = lhs.bytes.len / element_size;
        const bytes = try allocator.alloc(u8, element_count);
        errdefer allocator.free(bytes);
        for (0..element_count) |i| {
            bytes[i] = if (compareElement(lhs.element_type, lhs.bytes, rhs.bytes, i, direction)) 1 else 0;
        }

        var backend_buffer: ?backend_api.BufferHandle = null;
        if (lhs.backend_buffer != null and rhs.backend_buffer != null) {
            backend_buffer = lhs.backend.compare(lhs.backend_buffer.?, rhs.backend_buffer.?, direction, output_dims) catch null;
        }
        errdefer if (backend_buffer) |owned| lhs.backend.destroyBuffer(owned);

        buffer.* = .{
            .allocator = allocator,
            .backend = lhs.backend,
            .backend_kind = lhs.backend_kind,
            .element_type = .pred,
            .dims = dims,
            .device_id = lhs.device_id,
            .memory_id = lhs.memory_id,
            .device = lhs.device,
            .memory = lhs.memory,
            .shard_index = shard_index,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
        };
        return buffer;
    }

    pub fn initSelect(
        allocator: std.mem.Allocator,
        pred: *Buffer,
        on_true: *Buffer,
        on_false: *Buffer,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (pred.element_type != .pred) return error.UnsupportedElementType;
        if (on_true.element_type != on_false.element_type) return error.UnsupportedElementType;
        if (!std.mem.eql(i64, on_true.dims, on_false.dims) or !std.mem.eql(i64, on_true.dims, output_dims) or !std.mem.eql(i64, pred.dims, output_dims)) return error.ShapeMismatch;
        const element_size = on_true.element_type.byteSize();
        if (element_size == 0) return error.UnsupportedElementType;

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, output_dims);
        errdefer allocator.free(dims);

        const bytes = try allocator.alloc(u8, on_true.bytes.len);
        errdefer allocator.free(bytes);
        const element_count = on_true.bytes.len / element_size;
        for (0..element_count) |i| {
            const src = if (pred.bytes[i] != 0) on_true.bytes else on_false.bytes;
            @memcpy(bytes[i * element_size ..][0..element_size], src[i * element_size ..][0..element_size]);
        }

        var backend_buffer: ?backend_api.BufferHandle = null;
        if (pred.backend_buffer != null and on_true.backend_buffer != null and on_false.backend_buffer != null) {
            backend_buffer = on_true.backend.select(pred.backend_buffer.?, on_true.backend_buffer.?, on_false.backend_buffer.?, output_dims) catch null;
        }
        errdefer if (backend_buffer) |owned| on_true.backend.destroyBuffer(owned);

        buffer.* = .{
            .allocator = allocator,
            .backend = on_true.backend,
            .backend_kind = on_true.backend_kind,
            .element_type = on_true.element_type,
            .dims = dims,
            .device_id = on_true.device_id,
            .memory_id = on_true.memory_id,
            .device = on_true.device,
            .memory = on_true.memory,
            .shard_index = shard_index,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
        };
        return buffer;
    }

    pub fn initConvert(
        allocator: std.mem.Allocator,
        src: *Buffer,
        output_type: BufferType,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (!std.mem.eql(i64, src.dims, output_dims)) return error.ShapeMismatch;
        if (src.element_type.byteSize() == 0 or output_type.byteSize() == 0) return error.UnsupportedElementType;

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, output_dims);
        errdefer allocator.free(dims);

        const output_size = denseByteSize(output_type, output_dims);
        const bytes = try allocator.alloc(u8, output_size);
        errdefer allocator.free(bytes);

        const element_count = if (output_type.byteSize() == 0) 0 else output_size / output_type.byteSize();
        for (0..element_count) |i| {
            const value = readScalarAsF64(src.element_type, src.bytes, i);
            writeScalarFromF64(output_type, bytes, i, value);
        }

        const backend_buffer = src.backend.bufferFromHost(src.device.local_hardware_id, output_type, output_dims, bytes) catch null;
        errdefer if (backend_buffer) |owned| src.backend.destroyBuffer(owned);

        buffer.* = .{
            .allocator = allocator,
            .backend = src.backend,
            .backend_kind = src.backend_kind,
            .element_type = output_type,
            .dims = dims,
            .device_id = src.device_id,
            .memory_id = src.memory_id,
            .device = src.device,
            .memory = src.memory,
            .shard_index = shard_index,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
        };
        return buffer;
    }

    pub fn initIota(
        allocator: std.mem.Allocator,
        backend_impl: backend_api.Backend,
        element_type: BufferType,
        output_dims: []const i64,
        device: *Device,
        memory: *Memory,
        iota_dimension: i64,
        shard_index: usize,
    ) !*Buffer {
        if (iota_dimension < 0 or iota_dimension >= @as(i64, @intCast(output_dims.len))) return error.ShapeMismatch;
        if (element_type.byteSize() == 0) return error.UnsupportedElementType;

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);

        const dims = try allocator.dupe(i64, output_dims);
        errdefer allocator.free(dims);

        const bytes = try allocator.alloc(u8, denseByteSize(element_type, output_dims));
        errdefer allocator.free(bytes);
        const strides = try rowMajorStrides(allocator, output_dims);
        defer allocator.free(strides);
        const coords = try allocator.alloc(usize, output_dims.len);
        defer allocator.free(coords);
        const element_count = if (element_type.byteSize() == 0) 0 else bytes.len / element_type.byteSize();
        const axis: usize = @intCast(iota_dimension);
        for (0..element_count) |i| {
            unravelIndex(i, output_dims, strides, coords);
            writeScalarFromF64(element_type, bytes, i, @floatFromInt(coords[axis]));
        }

        const backend_buffer = backend_impl.bufferFromHost(device.local_hardware_id, element_type, output_dims, bytes) catch null;
        errdefer if (backend_buffer) |owned| backend_impl.destroyBuffer(owned);

        buffer.* = .{
            .allocator = allocator,
            .backend = backend_impl,
            .backend_kind = backend_impl.kind(),
            .element_type = element_type,
            .dims = dims,
            .device_id = device.id,
            .memory_id = memory.id,
            .device = device,
            .memory = memory,
            .shard_index = shard_index,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
        };
        return buffer;
    }

    pub fn initReverse(
        allocator: std.mem.Allocator,
        src: *Buffer,
        dimensions: []const i64,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (!std.mem.eql(i64, src.dims, output_dims)) return error.ShapeMismatch;
        const element_size = src.element_type.byteSize();
        if (element_size == 0) return error.UnsupportedElementType;

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);
        const dims = try allocator.dupe(i64, output_dims);
        errdefer allocator.free(dims);
        const bytes = try allocator.alloc(u8, src.bytes.len);
        errdefer allocator.free(bytes);
        const strides = try rowMajorStrides(allocator, src.dims);
        defer allocator.free(strides);
        const coords = try allocator.alloc(usize, src.dims.len);
        defer allocator.free(coords);
        const element_count = src.bytes.len / element_size;
        for (0..element_count) |dst_index| {
            unravelIndex(dst_index, output_dims, strides, coords);
            var src_index: usize = 0;
            for (coords, 0..) |coord, axis| {
                const reverse_axis = axisInList(axis, dimensions);
                const src_coord = if (reverse_axis) @as(usize, @intCast(src.dims[axis])) - 1 - coord else coord;
                src_index += src_coord * strides[axis];
            }
            @memcpy(bytes[dst_index * element_size ..][0..element_size], src.bytes[src_index * element_size ..][0..element_size]);
        }
        const backend_buffer = src.backend.bufferFromHost(src.device.local_hardware_id, src.element_type, output_dims, bytes) catch null;
        errdefer if (backend_buffer) |owned| src.backend.destroyBuffer(owned);
        buffer.* = .{
            .allocator = allocator,
            .backend = src.backend,
            .backend_kind = src.backend_kind,
            .element_type = src.element_type,
            .dims = dims,
            .device_id = src.device_id,
            .memory_id = src.memory_id,
            .device = src.device,
            .memory = src.memory,
            .shard_index = shard_index,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
        };
        return buffer;
    }

    pub fn initClamp(
        allocator: std.mem.Allocator,
        min_buffer: *Buffer,
        value_buffer: *Buffer,
        max_buffer: *Buffer,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (value_buffer.element_type != min_buffer.element_type or value_buffer.element_type != max_buffer.element_type) return error.UnsupportedElementType;
        if (!std.mem.eql(i64, value_buffer.dims, output_dims)) return error.ShapeMismatch;
        const element_size = value_buffer.element_type.byteSize();
        if (element_size == 0) return error.UnsupportedElementType;
        const min_scalar = min_buffer.dims.len == 0;
        const max_scalar = max_buffer.dims.len == 0;
        if (!min_scalar and !std.mem.eql(i64, min_buffer.dims, output_dims)) return error.ShapeMismatch;
        if (!max_scalar and !std.mem.eql(i64, max_buffer.dims, output_dims)) return error.ShapeMismatch;

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);
        const dims = try allocator.dupe(i64, output_dims);
        errdefer allocator.free(dims);
        const bytes = try allocator.alloc(u8, value_buffer.bytes.len);
        errdefer allocator.free(bytes);
        const element_count = value_buffer.bytes.len / element_size;
        for (0..element_count) |i| {
            const min_value = readScalarAsF64(value_buffer.element_type, min_buffer.bytes, if (min_scalar) 0 else i);
            const value = readScalarAsF64(value_buffer.element_type, value_buffer.bytes, i);
            const max_value = readScalarAsF64(value_buffer.element_type, max_buffer.bytes, if (max_scalar) 0 else i);
            writeScalarFromF64(value_buffer.element_type, bytes, i, @min(@max(value, min_value), max_value));
        }
        const backend_buffer = value_buffer.backend.bufferFromHost(value_buffer.device.local_hardware_id, value_buffer.element_type, output_dims, bytes) catch null;
        errdefer if (backend_buffer) |owned| value_buffer.backend.destroyBuffer(owned);
        buffer.* = .{
            .allocator = allocator,
            .backend = value_buffer.backend,
            .backend_kind = value_buffer.backend_kind,
            .element_type = value_buffer.element_type,
            .dims = dims,
            .device_id = value_buffer.device_id,
            .memory_id = value_buffer.memory_id,
            .device = value_buffer.device,
            .memory = value_buffer.memory,
            .shard_index = shard_index,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
        };
        return buffer;
    }

    pub fn initDynamicSlice(
        allocator: std.mem.Allocator,
        src: *Buffer,
        start_buffers: []const *Buffer,
        slice_sizes: []const i64,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (start_buffers.len != src.dims.len or slice_sizes.len != src.dims.len) return error.ShapeMismatch;
        var starts = try allocator.alloc(i64, start_buffers.len);
        defer allocator.free(starts);
        var limits = try allocator.alloc(i64, start_buffers.len);
        defer allocator.free(limits);
        var strides = try allocator.alloc(i64, start_buffers.len);
        defer allocator.free(strides);
        for (start_buffers, 0..) |start_buffer, i| {
            if (start_buffer.dims.len != 0) return error.ShapeMismatch;
            starts[i] = scalarIndex(start_buffer);
            if (starts[i] < 0) starts[i] = 0;
            if (starts[i] + slice_sizes[i] > src.dims[i]) starts[i] = src.dims[i] - slice_sizes[i];
            limits[i] = starts[i] + slice_sizes[i];
            strides[i] = 1;
        }
        return initSlice(allocator, src, starts, limits, strides, output_dims, shard_index);
    }

    pub fn initDynamicUpdateSlice(
        allocator: std.mem.Allocator,
        src: *Buffer,
        update: *Buffer,
        start_buffers: []const *Buffer,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (src.element_type != update.element_type) return error.UnsupportedElementType;
        if (!std.mem.eql(i64, src.dims, output_dims) or src.dims.len != update.dims.len or start_buffers.len != src.dims.len) return error.ShapeMismatch;
        const element_size = src.element_type.byteSize();
        if (element_size == 0) return error.UnsupportedElementType;

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);
        const dims = try allocator.dupe(i64, output_dims);
        errdefer allocator.free(dims);
        const bytes = try allocator.dupe(u8, src.bytes);
        errdefer allocator.free(bytes);

        var starts = try allocator.alloc(i64, start_buffers.len);
        defer allocator.free(starts);
        for (start_buffers, 0..) |start_buffer, i| {
            if (start_buffer.dims.len != 0) return error.ShapeMismatch;
            starts[i] = scalarIndex(start_buffer);
            if (starts[i] < 0) starts[i] = 0;
            if (starts[i] + update.dims[i] > src.dims[i]) starts[i] = src.dims[i] - update.dims[i];
        }

        const src_strides = try rowMajorStrides(allocator, src.dims);
        defer allocator.free(src_strides);
        const update_strides = try rowMajorStrides(allocator, update.dims);
        defer allocator.free(update_strides);
        const coords = try allocator.alloc(usize, update.dims.len);
        defer allocator.free(coords);
        const update_count = update.bytes.len / element_size;
        for (0..update_count) |update_index| {
            unravelIndex(update_index, update.dims, update_strides, coords);
            var dst_index: usize = 0;
            for (coords, 0..) |coord, axis| {
                dst_index += (@as(usize, @intCast(starts[axis])) + coord) * src_strides[axis];
            }
            @memcpy(bytes[dst_index * element_size ..][0..element_size], update.bytes[update_index * element_size ..][0..element_size]);
        }
        const backend_buffer = src.backend.bufferFromHost(src.device.local_hardware_id, src.element_type, output_dims, bytes) catch null;
        errdefer if (backend_buffer) |owned| src.backend.destroyBuffer(owned);
        buffer.* = .{
            .allocator = allocator,
            .backend = src.backend,
            .backend_kind = src.backend_kind,
            .element_type = src.element_type,
            .dims = dims,
            .device_id = src.device_id,
            .memory_id = src.memory_id,
            .device = src.device,
            .memory = src.memory,
            .shard_index = shard_index,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
        };
        return buffer;
    }

    pub fn initPad(
        allocator: std.mem.Allocator,
        src: *Buffer,
        padding_value: *Buffer,
        edge_padding_low: []const i64,
        edge_padding_high: []const i64,
        interior_padding: []const i64,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        if (src.element_type != padding_value.element_type or padding_value.dims.len != 0) return error.UnsupportedElementType;
        const rank = src.dims.len;
        if (edge_padding_low.len != rank or edge_padding_high.len != rank or interior_padding.len != rank or output_dims.len != rank) return error.ShapeMismatch;
        const element_size = src.element_type.byteSize();
        if (element_size == 0) return error.UnsupportedElementType;

        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);
        const dims = try allocator.dupe(i64, output_dims);
        errdefer allocator.free(dims);
        const bytes = try allocator.alloc(u8, denseByteSize(src.element_type, output_dims));
        errdefer allocator.free(bytes);
        const out_count = bytes.len / element_size;
        for (0..out_count) |i| @memcpy(bytes[i * element_size ..][0..element_size], padding_value.bytes[0..element_size]);

        const src_strides = try rowMajorStrides(allocator, src.dims);
        defer allocator.free(src_strides);
        const dst_strides = try rowMajorStrides(allocator, output_dims);
        defer allocator.free(dst_strides);
        const coords = try allocator.alloc(usize, src.dims.len);
        defer allocator.free(coords);
        const src_count = src.bytes.len / element_size;
        for (0..src_count) |src_index| {
            unravelIndex(src_index, src.dims, src_strides, coords);
            var dst_index: usize = 0;
            for (coords, 0..) |coord, axis| {
                const interior = @as(usize, @intCast(interior_padding[axis]));
                const low = @as(usize, @intCast(edge_padding_low[axis]));
                dst_index += (low + coord * (interior + 1)) * dst_strides[axis];
            }
            @memcpy(bytes[dst_index * element_size ..][0..element_size], src.bytes[src_index * element_size ..][0..element_size]);
        }
        const backend_buffer = src.backend.bufferFromHost(src.device.local_hardware_id, src.element_type, output_dims, bytes) catch null;
        errdefer if (backend_buffer) |owned| src.backend.destroyBuffer(owned);
        buffer.* = .{
            .allocator = allocator,
            .backend = src.backend,
            .backend_kind = src.backend_kind,
            .element_type = src.element_type,
            .dims = dims,
            .device_id = src.device_id,
            .memory_id = src.memory_id,
            .device = src.device,
            .memory = src.memory,
            .shard_index = shard_index,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
        };
        return buffer;
    }

    pub fn initGather(
        allocator: std.mem.Allocator,
        operand: *Buffer,
        indices: *Buffer,
        offset_dims: []const i64,
        collapsed_slice_dims: []const i64,
        operand_batching_dims: []const i64,
        start_indices_batching_dims: []const i64,
        start_index_map: []const i64,
        index_vector_dim: i64,
        slice_sizes: []const i64,
        output_dims: []const i64,
        shard_index: usize,
    ) !*Buffer {
        _ = offset_dims;
        _ = operand_batching_dims;
        _ = start_indices_batching_dims;
        if (operand.element_type.byteSize() == 0) return error.UnsupportedElementType;
        if (start_index_map.len == 0 or slice_sizes.len != operand.dims.len) return error.ShapeMismatch;
        if (index_vector_dim < 0) return error.ShapeMismatch;

        const element_size = operand.element_type.byteSize();
        const buffer = try allocator.create(Buffer);
        errdefer allocator.destroy(buffer);
        const dims = try allocator.dupe(i64, output_dims);
        errdefer allocator.free(dims);
        const bytes = try allocator.alloc(u8, denseByteSize(operand.element_type, output_dims));
        errdefer allocator.free(bytes);

        const operand_strides = try rowMajorStrides(allocator, operand.dims);
        defer allocator.free(operand_strides);
        const output_strides = try rowMajorStrides(allocator, output_dims);
        defer allocator.free(output_strides);
        const out_coords = try allocator.alloc(usize, output_dims.len);
        defer allocator.free(out_coords);
        const out_count = bytes.len / element_size;

        if (collapsed_slice_dims.len == 1 and start_index_map.len == 1 and slice_sizes[@intCast(start_index_map[0])] == 1) {
            const gather_axis: usize = @intCast(start_index_map[0]);
            const index_rank = indices.dims.len;
            const index_vector_axis: usize = @intCast(index_vector_dim);
            const index_prefix_rank = if (index_rank > 0 and index_vector_axis < index_rank and indices.dims[index_vector_axis] == @as(i64, @intCast(start_index_map.len))) index_rank - 1 else index_rank;
            if (output_dims.len < index_prefix_rank + operand.dims.len - 1) return error.ShapeMismatch;
            const index_strides = try rowMajorStrides(allocator, indices.dims);
            defer allocator.free(index_strides);
            for (0..out_count) |out_index| {
                unravelIndex(out_index, output_dims, output_strides, out_coords);
                var index_flat: usize = 0;
                var prefix_axis: usize = 0;
                for (0..index_rank) |axis| {
                    const coord = if (axis == index_vector_axis and index_rank != index_prefix_rank) 0 else blk: {
                        const c = out_coords[prefix_axis];
                        prefix_axis += 1;
                        break :blk c;
                    };
                    index_flat += coord * index_strides[axis];
                }
                const gathered = scalarIndexAt(indices, index_flat);
                var operand_index: usize = 0;
                var offset_axis: usize = 0;
                for (0..operand.dims.len) |axis| {
                    const coord = if (axis == gather_axis) @as(usize, @intCast(@max(@as(i64, 0), @min(gathered, operand.dims[axis] - 1)))) else blk: {
                        const c = out_coords[index_prefix_rank + offset_axis];
                        offset_axis += 1;
                        break :blk c;
                    };
                    operand_index += coord * operand_strides[axis];
                }
                @memcpy(bytes[out_index * element_size ..][0..element_size], operand.bytes[operand_index * element_size ..][0..element_size]);
            }
        } else {
            return error.UnsupportedElementType;
        }

        const backend_buffer = operand.backend.bufferFromHost(operand.device.local_hardware_id, operand.element_type, output_dims, bytes) catch null;
        errdefer if (backend_buffer) |owned| operand.backend.destroyBuffer(owned);
        buffer.* = .{
            .allocator = allocator,
            .backend = operand.backend,
            .backend_kind = operand.backend_kind,
            .element_type = operand.element_type,
            .dims = dims,
            .device_id = operand.device_id,
            .memory_id = operand.memory_id,
            .device = operand.device,
            .memory = operand.memory,
            .shard_index = shard_index,
            .bytes = bytes,
            .backend_buffer = backend_buffer,
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
        if (self.backend_buffer) |backend_buffer| self.backend.destroyBuffer(backend_buffer);
        self.allocator.free(self.bytes);
        self.allocator.free(self.dims);
        self.allocator.destroy(self);
    }

    pub fn copyToHost(self: *Buffer, dst: []u8) !void {
        if (dst.len < self.bytes.len) return error.DestinationTooSmall;
        if (self.backend_buffer) |backend_buffer| {
            self.backend.copyToHost(backend_buffer, dst) catch return error.BackendBufferCopyFailed;
            return;
        }
        @memcpy(dst[0..self.bytes.len], self.bytes);
    }

    pub fn hasBackendStorage(self: *const Buffer) bool {
        return self.backend_buffer != null;
    }
};

fn denseByteSize(element_type: BufferType, dims: []const i64) usize {
    return core.denseByteSize(element_type, dims);
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

fn validDotGeneral(
    lhs_dims: []const i64,
    rhs_dims: []const i64,
    lhs_batch_dimensions: []const i64,
    rhs_batch_dimensions: []const i64,
    lhs_contracting_dimensions: []const i64,
    rhs_contracting_dimensions: []const i64,
    output_dims: []const i64,
) bool {
    if (lhs_contracting_dimensions.len != 1 or rhs_contracting_dimensions.len != 1) return false;
    if (lhs_batch_dimensions.len != rhs_batch_dimensions.len) return false;
    const lhs_contract: usize = if (lhs_contracting_dimensions[0] < 0) return false else @intCast(lhs_contracting_dimensions[0]);
    const rhs_contract: usize = if (rhs_contracting_dimensions[0] < 0) return false else @intCast(rhs_contracting_dimensions[0]);
    if (lhs_contract >= lhs_dims.len or rhs_contract >= rhs_dims.len) return false;
    if (lhs_dims[lhs_contract] != rhs_dims[rhs_contract]) return false;
    for (lhs_batch_dimensions, rhs_batch_dimensions) |lhs_axis_i64, rhs_axis_i64| {
        if (lhs_axis_i64 < 0 or rhs_axis_i64 < 0) return false;
        const lhs_axis: usize = @intCast(lhs_axis_i64);
        const rhs_axis: usize = @intCast(rhs_axis_i64);
        if (lhs_axis >= lhs_dims.len or rhs_axis >= rhs_dims.len or lhs_dims[lhs_axis] != rhs_dims[rhs_axis]) return false;
    }
    return denseByteSize(.f32, output_dims) != 0;
}

fn validReduce(input_dims: []const i64, dimensions: []const i64, output_dims: []const i64) bool {
    var reduced = [_]bool{false} ** 64;
    if (input_dims.len > reduced.len) return false;
    for (dimensions) |dim_i64| {
        if (dim_i64 < 0 or dim_i64 >= @as(i64, @intCast(input_dims.len))) return false;
        const dim: usize = @intCast(dim_i64);
        if (reduced[dim]) return false;
        reduced[dim] = true;
    }
    var expected_rank: usize = 0;
    for (0..input_dims.len) |axis| {
        if (!reduced[axis]) expected_rank += 1;
    }
    if (output_dims.len != expected_rank) return false;
    var out_axis: usize = 0;
    for (input_dims, 0..) |dim, axis| {
        if (!reduced[axis]) {
            if (output_dims[out_axis] != dim) return false;
            out_axis += 1;
        }
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

fn writeF32LE(bytes: []u8, index: usize, value: f32) void {
    std.mem.writeInt(u32, bytes[index * 4 ..][0..4], @bitCast(value), .little);
}

fn readI32LE(bytes: []const u8, index: usize) i32 {
    return @bitCast(std.mem.readInt(u32, bytes[index * 4 ..][0..4], .little));
}

fn writeI32LE(bytes: []u8, index: usize, value: i32) void {
    std.mem.writeInt(u32, bytes[index * 4 ..][0..4], @bitCast(value), .little);
}

fn readU32LE(bytes: []const u8, index: usize) u32 {
    return std.mem.readInt(u32, bytes[index * 4 ..][0..4], .little);
}

fn writeU32LE(bytes: []u8, index: usize, value: u32) void {
    std.mem.writeInt(u32, bytes[index * 4 ..][0..4], value, .little);
}

fn compareF32(lhs: f32, rhs: f32, direction: CompareOp) bool {
    return switch (direction) {
        .eq => lhs == rhs,
        .ne => lhs != rhs,
        .ge => lhs >= rhs,
        .gt => lhs > rhs,
        .le => lhs <= rhs,
        .lt => lhs < rhs,
    };
}

fn compareScalar(lhs: f64, rhs: f64, direction: CompareOp) bool {
    return switch (direction) {
        .eq => lhs == rhs,
        .ne => lhs != rhs,
        .ge => lhs >= rhs,
        .gt => lhs > rhs,
        .le => lhs <= rhs,
        .lt => lhs < rhs,
    };
}

fn compareElement(element_type: BufferType, lhs: []const u8, rhs: []const u8, index: usize, direction: CompareOp) bool {
    return compareScalar(readScalarAsF64(element_type, lhs, index), readScalarAsF64(element_type, rhs, index), direction);
}

fn readScalarAsF64(element_type: BufferType, bytes: []const u8, index: usize) f64 {
    return switch (element_type) {
        .pred, .u8 => @floatFromInt(bytes[index]),
        .s8 => @floatFromInt(@as(i8, @bitCast(bytes[index]))),
        .s32 => @floatFromInt(readI32LE(bytes, index)),
        .u32 => @floatFromInt(readU32LE(bytes, index)),
        .f32 => readF32LE(bytes, index),
        else => 0.0,
    };
}

fn writeScalarFromF64(element_type: BufferType, bytes: []u8, index: usize, value: f64) void {
    switch (element_type) {
        .pred => bytes[index] = if (value != 0.0) 1 else 0,
        .u8 => bytes[index] = @intFromFloat(@min(@max(value, 0.0), 255.0)),
        .s8 => bytes[index] = @bitCast(@as(i8, @intFromFloat(@min(@max(value, -128.0), 127.0)))),
        .s32 => writeI32LE(bytes, index, @intFromFloat(@min(@max(value, -2147483648.0), 2147483647.0))),
        .u32 => writeU32LE(bytes, index, @intFromFloat(@min(@max(value, 0.0), 4294967295.0))),
        .f32 => writeF32LE(bytes, index, @floatCast(value)),
        else => {},
    }
}

fn scalarIndex(buffer: *const Buffer) i64 {
    return scalarIndexAt(buffer, 0);
}

fn scalarIndexAt(buffer: *const Buffer, index: usize) i64 {
    return switch (buffer.element_type) {
        .pred, .u8 => buffer.bytes[index],
        .s8 => @as(i8, @bitCast(buffer.bytes[index])),
        .s32 => readI32LE(buffer.bytes, index),
        .u32 => @intCast(readU32LE(buffer.bytes, index)),
        .f32 => @intFromFloat(readF32LE(buffer.bytes, index)),
        else => 0,
    };
}

fn axisInList(axis: usize, axes: []const i64) bool {
    for (axes) |candidate| {
        if (candidate >= 0 and @as(usize, @intCast(candidate)) == axis) return true;
    }
    return false;
}

fn outputAxesWithout(input_rank: usize, removed_axes: []const i64, allocator: std.mem.Allocator) ![]usize {
    var axes = try allocator.alloc(usize, input_rank - removed_axes.len);
    var out: usize = 0;
    for (0..input_rank) |axis| {
        if (!axisInList(axis, removed_axes)) {
            axes[out] = axis;
            out += 1;
        }
    }
    return axes;
}

fn evalReduceF32(
    allocator: std.mem.Allocator,
    kind: core.PlanInstructionKind,
    src_bytes: []const u8,
    input_dims: []const i64,
    dimensions: []const i64,
    output_dims: []const i64,
    out_bytes: []u8,
) !void {
    const input_strides = try rowMajorStrides(allocator, input_dims);
    defer allocator.free(input_strides);
    const output_strides = try rowMajorStrides(allocator, output_dims);
    defer allocator.free(output_strides);
    const coords = try allocator.alloc(usize, input_dims.len);
    defer allocator.free(coords);
    const output_axes = try outputAxesWithout(input_dims.len, dimensions, allocator);
    defer allocator.free(output_axes);

    const output_count = if (output_dims.len == 0) 1 else denseByteSize(.f32, output_dims) / 4;
    for (0..output_count) |i| {
        writeF32LE(out_bytes, i, if (kind == .reduce_sum) 0.0 else -std.math.inf(f32));
    }
    const input_count = src_bytes.len / 4;
    for (0..input_count) |input_index| {
        unravelIndex(input_index, input_dims, input_strides, coords);
        var output_index: usize = 0;
        for (output_axes, 0..) |input_axis, output_axis| {
            output_index += coords[input_axis] * output_strides[output_axis];
        }
        const current = readF32LE(out_bytes, output_index);
        const value = readF32LE(src_bytes, input_index);
        writeF32LE(out_bytes, output_index, if (kind == .reduce_sum) current + value else @max(current, value));
    }
}

fn evalDotGeneralF32(
    allocator: std.mem.Allocator,
    lhs_bytes: []const u8,
    lhs_dims: []const i64,
    rhs_bytes: []const u8,
    rhs_dims: []const i64,
    lhs_batch_dimensions: []const i64,
    rhs_batch_dimensions: []const i64,
    lhs_contracting_dimensions: []const i64,
    rhs_contracting_dimensions: []const i64,
    output_dims: []const i64,
    out_bytes: []u8,
) !void {
    const lhs_contract: usize = @intCast(lhs_contracting_dimensions[0]);
    const rhs_contract: usize = @intCast(rhs_contracting_dimensions[0]);
    const lhs_strides = try rowMajorStrides(allocator, lhs_dims);
    defer allocator.free(lhs_strides);
    const rhs_strides = try rowMajorStrides(allocator, rhs_dims);
    defer allocator.free(rhs_strides);
    const out_strides = try rowMajorStrides(allocator, output_dims);
    defer allocator.free(out_strides);
    const out_coords = try allocator.alloc(usize, output_dims.len);
    defer allocator.free(out_coords);

    const lhs_free_axes = try outputAxesWithout(lhs_dims.len, lhs_contracting_dimensions, allocator);
    defer allocator.free(lhs_free_axes);
    const rhs_free_axes = try outputAxesWithout(rhs_dims.len, rhs_contracting_dimensions, allocator);
    defer allocator.free(rhs_free_axes);
    const batch_count = lhs_batch_dimensions.len;
    const lhs_non_batch_count = lhs_free_axes.len - batch_count;
    const rhs_non_batch_count = rhs_free_axes.len - batch_count;
    const contract_size: usize = @intCast(lhs_dims[lhs_contract]);
    const output_count = out_bytes.len / 4;

    for (0..output_count) |out_index| {
        unravelIndex(out_index, output_dims, out_strides, out_coords);
        var lhs_base: usize = 0;
        var rhs_base: usize = 0;
        var out_axis: usize = 0;
        for (0..batch_count) |i| {
            const coord = out_coords[out_axis];
            lhs_base += coord * lhs_strides[@intCast(lhs_batch_dimensions[i])];
            rhs_base += coord * rhs_strides[@intCast(rhs_batch_dimensions[i])];
            out_axis += 1;
        }
        for (lhs_free_axes[batch_count..], 0..) |lhs_axis, i| {
            lhs_base += out_coords[out_axis + i] * lhs_strides[lhs_axis];
        }
        out_axis += lhs_non_batch_count;
        for (rhs_free_axes[batch_count..], 0..) |rhs_axis, i| {
            rhs_base += out_coords[out_axis + i] * rhs_strides[rhs_axis];
        }
        _ = rhs_non_batch_count;
        var acc: f32 = 0.0;
        for (0..contract_size) |k| {
            acc += readF32LE(lhs_bytes, lhs_base + k * lhs_strides[lhs_contract]) * readF32LE(rhs_bytes, rhs_base + k * rhs_strides[rhs_contract]);
        }
        writeF32LE(out_bytes, out_index, acc);
    }
}

fn syntheticBackendForTest() backend_api.Backend {
    return @import("src/backend/synthetic").create();
}

fn initSyntheticClientForTest(device_count: usize) !*Client {
    return Client.init(std.testing.allocator, syntheticBackendForTest(), device_count);
}

fn initHostCopyForTest(
    element_type: BufferType,
    dims: []const i64,
    device: *Device,
    memory: *Memory,
    shard_index: usize,
    src: []const u8,
) !*Buffer {
    return Buffer.initHostCopyForBackend(std.testing.allocator, syntheticBackendForTest(), element_type, dims, device, memory, shard_index, src);
}

test "synthetic client models multiple devices and memories" {
    const client = try initSyntheticClientForTest(4);
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

test "buffer keeps shard/device/memory ownership metadata" {
    const client = try initSyntheticClientForTest(2);
    defer client.deinit();

    const dims = [_]i64{ 2, 2 };
    const data = [_]u8{ 1, 2, 3, 4 };
    const buffer = try initHostCopyForTest(.u8, &dims, &client.devices[1], &client.memories[1], 1, &data);
    defer buffer.deinit();

    try std.testing.expectEqual(@as(i32, 1), buffer.device_id);
    try std.testing.expectEqual(@as(i32, 1), buffer.memory_id);
    try std.testing.expectEqual(@as(i32, 1), buffer.device.id);
    try std.testing.expectEqual(@as(i32, 1), buffer.memory.id);
    try std.testing.expectEqual(@as(usize, 1), buffer.shard_index);
    try std.testing.expectEqualSlices(u8, &data, buffer.bytes);

    const rhs_data = [_]u8{ 10, 20, 30, 40 };
    const rhs = try initHostCopyForTest(.u8, &dims, &client.devices[1], &client.memories[1], 1, &rhs_data);
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
    const client = try initSyntheticClientForTest(1);
    defer client.deinit();

    const dims = [_]i64{2};
    const lhs_values = [_]f32{ 1.5, -2.0 };
    const rhs_values = [_]f32{ 2.25, 4.0 };
    const lhs = try initHostCopyForTest(.f32, &dims, &client.devices[0], &client.memories[0], 0, std.mem.asBytes(&lhs_values));
    defer lhs.deinit();
    const rhs = try initHostCopyForTest(.f32, &dims, &client.devices[0], &client.memories[0], 0, std.mem.asBytes(&rhs_values));
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
    const client = try initSyntheticClientForTest(1);
    defer client.deinit();

    const dims = [_]i64{2};
    const values = [_]f32{ 1.0, 4.0 };
    const input = try initHostCopyForTest(.f32, &dims, &client.devices[0], &client.memories[0], 0, std.mem.asBytes(&values));
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
    const client = try initSyntheticClientForTest(1);
    defer client.deinit();

    const dims = [_]i64{ 2, 2 };
    const values = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const input = try initHostCopyForTest(.f32, &dims, &client.devices[0], &client.memories[0], 0, std.mem.asBytes(&values));
    defer input.deinit();

    const reshaped = try Buffer.initReshape(std.testing.allocator, input, &.{4}, 0);
    defer reshaped.deinit();

    try std.testing.expectEqualSlices(i64, &.{4}, reshaped.dims);
    try std.testing.expectEqualSlices(u8, input.bytes, reshaped.bytes);
}

test "buffer transpose permutes dense host bytes and updates dimensions" {
    const client = try initSyntheticClientForTest(1);
    defer client.deinit();

    const dims = [_]i64{ 2, 3 };
    const values = [_]u8{ 1, 2, 3, 4, 5, 6 };
    const input = try initHostCopyForTest(.u8, &dims, &client.devices[0], &client.memories[0], 0, &values);
    defer input.deinit();

    const transposed = try Buffer.initTranspose(std.testing.allocator, input, &.{ 1, 0 }, &.{ 3, 2 }, 0);
    defer transposed.deinit();

    try std.testing.expectEqualSlices(i64, &.{ 3, 2 }, transposed.dims);
    try std.testing.expectEqualSlices(u8, &.{ 1, 4, 2, 5, 3, 6 }, transposed.bytes);
}

test "buffer broadcast_in_dim expands dense host bytes and updates dimensions" {
    const client = try initSyntheticClientForTest(1);
    defer client.deinit();

    const dims = [_]i64{3};
    const values = [_]u8{ 7, 8, 9 };
    const input = try initHostCopyForTest(.u8, &dims, &client.devices[0], &client.memories[0], 0, &values);
    defer input.deinit();

    const broadcasted = try Buffer.initBroadcastInDim(std.testing.allocator, input, &.{1}, &.{ 2, 3 }, 0);
    defer broadcasted.deinit();

    try std.testing.expectEqualSlices(i64, &.{ 2, 3 }, broadcasted.dims);
    try std.testing.expectEqualSlices(u8, &.{ 7, 8, 9, 7, 8, 9 }, broadcasted.bytes);
}

test "buffer slice copies strided dense host bytes and updates dimensions" {
    const client = try initSyntheticClientForTest(1);
    defer client.deinit();

    const dims = [_]i64{ 3, 4 };
    const values = [_]u8{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
    const input = try initHostCopyForTest(.u8, &dims, &client.devices[0], &client.memories[0], 0, &values);
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
    const client = try initSyntheticClientForTest(1);
    defer client.deinit();

    const lhs_dims = [_]i64{ 2, 2 };
    const rhs_dims = [_]i64{ 2, 3 };
    const lhs_values = [_]u8{ 1, 2, 3, 4 };
    const rhs_values = [_]u8{ 5, 6, 7, 8, 9, 10 };
    const lhs = try initHostCopyForTest(.u8, &lhs_dims, &client.devices[0], &client.memories[0], 0, &lhs_values);
    defer lhs.deinit();
    const rhs = try initHostCopyForTest(.u8, &rhs_dims, &client.devices[0], &client.memories[0], 0, &rhs_values);
    defer rhs.deinit();

    const concatenated = try Buffer.initConcatenate(std.testing.allocator, lhs, rhs, 1, &.{ 2, 5 }, 0);
    defer concatenated.deinit();

    try std.testing.expectEqualSlices(i64, &.{ 2, 5 }, concatenated.dims);
    try std.testing.expectEqualSlices(u8, &.{ 1, 2, 5, 6, 7, 3, 4, 8, 9, 10 }, concatenated.bytes);
}
