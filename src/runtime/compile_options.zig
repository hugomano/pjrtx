const std = @import("std");
const ir = @import("src/compiler/ir");

const CompileOptions = ir.CompileOptions;

const ProtoWire = enum(u3) {
    varint = 0,
    fixed64 = 1,
    length_delimited = 2,
    fixed32 = 5,
};

const ProtoField = struct {
    number: u32,
    wire: ProtoWire,
};

const ProtoReader = struct {
    bytes: []const u8,
    index: usize = 0,

    fn eof(self: ProtoReader) bool {
        return self.index >= self.bytes.len;
    }

    fn readByte(self: *ProtoReader) !u8 {
        if (self.index >= self.bytes.len) return error.UnexpectedEnd;
        defer self.index += 1;
        return self.bytes[self.index];
    }

    fn readVarint(self: *ProtoReader) !u64 {
        var result: u64 = 0;
        var shift: u6 = 0;
        while (true) {
            const byte = try self.readByte();
            result |= @as(u64, byte & 0x7f) << shift;
            if ((byte & 0x80) == 0) return result;
            if (shift >= 63) return error.InvalidVarint;
            shift += 7;
        }
    }

    fn readField(self: *ProtoReader) !ProtoField {
        const tag = try self.readVarint();
        if (tag == 0) return error.InvalidTag;
        const wire_value: u3 = @intCast(tag & 0x7);
        const wire: ProtoWire = switch (wire_value) {
            0 => .varint,
            1 => .fixed64,
            2 => .length_delimited,
            5 => .fixed32,
            else => return error.UnsupportedWireType,
        };
        return .{
            .number = @intCast(tag >> 3),
            .wire = wire,
        };
    }

    fn readLengthDelimited(self: *ProtoReader) ![]const u8 {
        const len: usize = @intCast(try self.readVarint());
        if (len > self.bytes.len - self.index) return error.UnexpectedEnd;
        defer self.index += len;
        return self.bytes[self.index .. self.index + len];
    }

    fn skip(self: *ProtoReader, wire: ProtoWire) !void {
        switch (wire) {
            .varint => _ = try self.readVarint(),
            .fixed64 => {
                if (8 > self.bytes.len - self.index) return error.UnexpectedEnd;
                self.index += 8;
            },
            .length_delimited => _ = try self.readLengthDelimited(),
            .fixed32 => {
                if (4 > self.bytes.len - self.index) return error.UnexpectedEnd;
                self.index += 4;
            },
        }
    }
};

const DeviceAssignmentProto = struct {
    replica_count: i32 = 0,
    computation_count: i32 = 0,
    computation_devices: std.ArrayList([]i32) = .empty,

    fn deinit(self: *DeviceAssignmentProto, allocator: std.mem.Allocator) void {
        for (self.computation_devices.items) |devices| allocator.free(devices);
        self.computation_devices.deinit(allocator);
    }

    fn parse(allocator: std.mem.Allocator, bytes: []const u8) !DeviceAssignmentProto {
        var assignment: DeviceAssignmentProto = .{};
        errdefer assignment.deinit(allocator);

        var reader: ProtoReader = .{ .bytes = bytes };
        while (!reader.eof()) {
            const field = try reader.readField();
            switch (field.number) {
                1 => {
                    if (field.wire != .varint) return error.InvalidCompileOptionsProto;
                    assignment.replica_count = @intCast(try reader.readVarint());
                },
                2 => {
                    if (field.wire != .varint) return error.InvalidCompileOptionsProto;
                    assignment.computation_count = @intCast(try reader.readVarint());
                },
                3 => {
                    if (field.wire != .length_delimited) return error.InvalidCompileOptionsProto;
                    const devices = try parseComputationDevice(allocator, try reader.readLengthDelimited());
                    errdefer allocator.free(devices);
                    try assignment.computation_devices.append(allocator, devices);
                },
                else => try reader.skip(field.wire),
            }
        }
        return assignment;
    }

    fn parseComputationDevice(allocator: std.mem.Allocator, bytes: []const u8) ![]i32 {
        var ids = std.ArrayList(i32).empty;
        errdefer ids.deinit(allocator);

        var reader: ProtoReader = .{ .bytes = bytes };
        while (!reader.eof()) {
            const field = try reader.readField();
            switch (field.number) {
                1 => switch (field.wire) {
                    .varint => try ids.append(allocator, @intCast(try reader.readVarint())),
                    .length_delimited => {
                        var packed_ids: ProtoReader = .{ .bytes = try reader.readLengthDelimited() };
                        while (!packed_ids.eof()) try ids.append(allocator, @intCast(try packed_ids.readVarint()));
                    },
                    else => return error.InvalidCompileOptionsProto,
                },
                else => try reader.skip(field.wire),
            }
        }
        return ids.toOwnedSlice(allocator);
    }

    fn appendFlattened(self: DeviceAssignmentProto, allocator: std.mem.Allocator, out: *std.ArrayList(i32)) !void {
        const partitions = if (self.computation_count > 0) @as(usize, @intCast(self.computation_count)) else self.computation_devices.items.len;
        const replicas = if (self.replica_count > 0) @as(usize, @intCast(self.replica_count)) else blk: {
            var max_replicas: usize = 0;
            for (self.computation_devices.items) |devices| max_replicas = @max(max_replicas, devices.len);
            break :blk max_replicas;
        };
        if (partitions == 0 or replicas == 0) return;
        if (self.computation_devices.items.len < partitions) return error.InvalidCompileOptionsProto;
        for (0..replicas) |replica| {
            for (0..partitions) |partition| {
                const devices = self.computation_devices.items[partition];
                if (replica >= devices.len) return error.InvalidCompileOptionsProto;
                try out.append(allocator, devices[replica]);
            }
        }
    }
};

/// Identifies the small textual compile-options format used by PjRTx tests.
pub fn isPjrtxText(text: []const u8) bool {
    if (text.len == 0) return false;
    for (text) |byte| {
        if (!(std.ascii.isPrint(byte) or std.ascii.isWhitespace(byte))) return false;
    }
    return std.mem.indexOf(u8, text, "replicas=") != null or
        std.mem.indexOf(u8, text, "partitions=") != null or
        std.mem.indexOf(u8, text, "assignment=") != null or
        std.mem.indexOf(u8, text, "use_shardy=") != null;
}

/// Decodes XLA `CompileOptionsProto` into compiler-owned PjRTx options.
pub fn parseXlaProto(allocator: std.mem.Allocator, bytes: []const u8) !CompileOptions {
    var options: CompileOptions = .{};
    var assignment = std.ArrayList(i32).empty;
    errdefer assignment.deinit(allocator);

    var reader: ProtoReader = .{ .bytes = bytes };
    while (!reader.eof()) {
        const field = try reader.readField();
        switch (field.number) {
            3 => {
                if (field.wire != .length_delimited) return error.InvalidCompileOptionsProto;
                try parseExecutableBuildOptionsProto(allocator, try reader.readLengthDelimited(), &options, &assignment);
            },
            else => try reader.skip(field.wire),
        }
    }

    if (options.num_replicas < 1 or options.num_partitions < 1) return error.InvalidCompileOptionsProto;
    if (assignment.items.len != 0 and assignment.items.len < options.numDevices()) return error.InvalidCompileOptionsProto;
    options.device_assignment = try assignment.toOwnedSlice(allocator);
    return options;
}

fn parseExecutableBuildOptionsProto(
    allocator: std.mem.Allocator,
    bytes: []const u8,
    options: *CompileOptions,
    assignment: *std.ArrayList(i32),
) !void {
    var reader: ProtoReader = .{ .bytes = bytes };
    while (!reader.eof()) {
        const field = try reader.readField();
        switch (field.number) {
            4 => {
                if (field.wire != .varint) return error.InvalidCompileOptionsProto;
                options.num_replicas = @intCast(try reader.readVarint());
            },
            5 => {
                if (field.wire != .varint) return error.InvalidCompileOptionsProto;
                options.num_partitions = @intCast(try reader.readVarint());
            },
            9 => {
                if (field.wire != .length_delimited) return error.InvalidCompileOptionsProto;
                var device_assignment = try DeviceAssignmentProto.parse(allocator, try reader.readLengthDelimited());
                defer device_assignment.deinit(allocator);
                try device_assignment.appendFlattened(allocator, assignment);
            },
            19 => {
                if (field.wire != .varint) return error.InvalidCompileOptionsProto;
                options.use_shardy_partitioner = (try reader.readVarint()) != 0;
            },
            else => try reader.skip(field.wire),
        }
    }
}

test "parse XLA CompileOptionsProto emitted by ZML" {
    const compile_options_proto = &[_]u8{
        0x1a, 0x14, // CompileOptionsProto.executable_build_options
        0x20, 0x01, // ExecutableBuildOptionsProto.num_replicas = 1
        0x28, 0x01, // ExecutableBuildOptionsProto.num_partitions = 1
        0x30, 0x01, // ExecutableBuildOptionsProto.use_spmd_partitioning = true
        0x4a, 0x09, // ExecutableBuildOptionsProto.device_assignment
        0x08, 0x01, // DeviceAssignmentProto.replica_count = 1
        0x10, 0x01, // DeviceAssignmentProto.computation_count = 1
        0x1a, 0x03, // DeviceAssignmentProto.computation_devices[0]
        0x0a, 0x01, 0x00, // ComputationDevice.replica_device_ids = [0]
        0x98, 0x01, 0x01, // ExecutableBuildOptionsProto.use_shardy_partitioner = true
    };

    const allocator = std.testing.allocator;
    const options = try parseXlaProto(allocator, compile_options_proto);
    defer allocator.free(options.device_assignment);

    try std.testing.expectEqual(@as(i32, 1), options.num_replicas);
    try std.testing.expectEqual(@as(i32, 1), options.num_partitions);
    try std.testing.expect(options.use_shardy_partitioner);
    try std.testing.expectEqualSlices(i32, &.{0}, options.device_assignment);
}

test "parse XLA CompileOptionsProto flattens device assignment in PJRT order" {
    const compile_options_proto = &[_]u8{
        0x1a, 0x19, // CompileOptionsProto.executable_build_options
        0x20, 0x02, // num_replicas = 2
        0x28, 0x02, // num_partitions = 2
        0x4a, 0x10, // device_assignment
        0x08, 0x02, // replica_count = 2
        0x10, 0x02, // computation_count = 2
        0x1a, 0x04, // computation_devices[0]
        0x0a, 0x02, 0x00, 0x02, // replica_device_ids = [0, 2]
        0x1a, 0x04, // computation_devices[1]
        0x0a, 0x02, 0x01, 0x03, // replica_device_ids = [1, 3]
        0x98, 0x01, 0x00, // use_shardy_partitioner = false
    };

    const allocator = std.testing.allocator;
    const options = try parseXlaProto(allocator, compile_options_proto);
    defer allocator.free(options.device_assignment);

    try std.testing.expectEqual(@as(i32, 2), options.num_replicas);
    try std.testing.expectEqual(@as(i32, 2), options.num_partitions);
    try std.testing.expect(!options.use_shardy_partitioner);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 2, 3 }, options.device_assignment);
}
