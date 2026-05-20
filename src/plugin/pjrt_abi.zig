const std = @import("std");

const runtime = @import("src/runtime");
const c = @import("c");

pub fn structSize(comptime T: type) usize {
    const typedef_name = comptime blk: {
        const needle = ".struct_";
        const idx = std.mem.indexOf(u8, @typeName(T), needle).?;
        break :blk @typeName(T)[idx + needle.len ..];
    };
    return @field(c, typedef_name ++ "_STRUCT_SIZE");
}

pub fn Struct(comptime T: type) type {
    const fields = std.meta.fields(T);
    var names: [fields.len][]const u8 = undefined;
    var types: [fields.len]type = undefined;
    var attributes: [fields.len]std.builtin.Type.StructField.Attributes = undefined;
    for (fields, &names, &types, &attributes) |field, *name, *type_, *attr| {
        name.* = field.name;
        type_.* = field.type;
        attr.* = .{
            .default_value_ptr = @ptrCast(if (std.mem.eql(u8, field.name, "struct_size"))
                &structSize(T)
            else
                &std.mem.zeroes(field.type)),
        };
    }
    return @Struct(.@"extern", null, &names, &types, &attributes);
}

fn view(comptime T: type, ptr: anytype) *T {
    return switch (@typeInfo(@TypeOf(ptr))) {
        .optional => @ptrCast(@alignCast(ptr.?)),
        .pointer => @ptrCast(@alignCast(ptr)),
        else => @compileError("PJRT ABI view expects an opaque pointer"),
    };
}

fn viewConst(comptime T: type, ptr: anytype) *const T {
    return switch (@typeInfo(@TypeOf(ptr))) {
        .optional => @ptrCast(@alignCast(ptr.?)),
        .pointer => @ptrCast(@alignCast(ptr)),
        else => @compileError("PJRT ABI const view expects an opaque pointer"),
    };
}

pub fn Opaque(comptime State: type, comptime CHandle: type) type {
    return struct {
        pub fn view(raw: anytype) *State {
            return pjrt_abi.view(State, raw);
        }

        pub fn viewConst(raw: anytype) *const State {
            return pjrt_abi.viewConst(State, raw);
        }

        pub fn handle(state: anytype) *CHandle {
            return @ptrCast(state);
        }

        pub fn handleSlice(states: []const *State) [*c]*CHandle {
            return @ptrCast(@constCast(states.ptr));
        }

        pub fn optionalHandle(state: anytype) ?*CHandle {
            return if (state) |ptr| handle(ptr) else null;
        }
    };
}

const pjrt_abi = @This();

pub const Client = Opaque(runtime.Client, c.PJRT_Client);
pub const TopologyDescription = Opaque(runtime.Client, c.PJRT_TopologyDescription);
pub const Device = Opaque(runtime.Device, c.PJRT_Device);
pub const DeviceDescription = Opaque(runtime.Device, c.PJRT_DeviceDescription);
pub const Memory = Opaque(runtime.Memory, c.PJRT_Memory);
pub const Buffer = Opaque(runtime.Buffer, c.PJRT_Buffer);
pub const Event = Opaque(runtime.Event, c.PJRT_Event);

pub fn Executable(comptime State: type) type {
    return Opaque(State, c.PJRT_Executable);
}

pub fn LoadedExecutable(comptime State: type) type {
    return Opaque(State, c.PJRT_LoadedExecutable);
}

pub fn AsyncHostToDeviceTransferManager(comptime State: type) type {
    return Opaque(State, c.PJRT_AsyncHostToDeviceTransferManager);
}

pub fn SerializedTopology(comptime State: type) type {
    return Opaque(State, c.PJRT_SerializedTopology);
}

pub fn DeviceAttributes(comptime State: type) type {
    return Opaque(State, c.PJRT_Device_Attributes);
}

pub fn UserData(comptime State: type) type {
    return Opaque(State, anyopaque);
}

pub const NamedValue = extern struct {
    comptime {
        std.debug.assert(@sizeOf(NamedValue) == @sizeOf(c.PJRT_NamedValue));
    }

    inner: c.PJRT_NamedValue,

    pub const Kind = enum(c.PJRT_NamedValue_Type) {
        string = c.PJRT_NamedValue_kString,
        int64 = c.PJRT_NamedValue_kInt64,
        int64list = c.PJRT_NamedValue_kInt64List,
        float = c.PJRT_NamedValue_kFloat,
        bool = c.PJRT_NamedValue_kBool,
    };

    pub const Value = union(Kind) {
        string: []const u8,
        int64: i64,
        int64list: []const i64,
        float: f32,
        bool: bool,
    };

    pub fn init(comptime kind_: Kind, name_: []const u8, value_: std.meta.fieldInfo(Value, kind_).type) NamedValue {
        return .{
            .inner = .{
                .struct_size = c.PJRT_NamedValue_STRUCT_SIZE,
                .extension_start = null,
                .name = name_.ptr,
                .name_size = name_.len,
                .type = @intFromEnum(kind_),
                .unnamed_0 = switch (kind_) {
                    .string => .{ .string_value = value_.ptr },
                    .int64 => .{ .int64_value = value_ },
                    .int64list => .{ .int64_array_value = value_.ptr },
                    .float => .{ .float_value = value_ },
                    .bool => .{ .bool_value = value_ },
                },
                .value_size = switch (kind_) {
                    .string, .int64list => value_.len,
                    inline else => 1,
                },
            },
        };
    }

    pub fn string(key: []const u8, value_: []const u8) NamedValue {
        return init(.string, key, value_);
    }

    pub fn int64(key: []const u8, value_: i64) NamedValue {
        return init(.int64, key, value_);
    }

    pub fn boolean(key: []const u8, value_: bool) NamedValue {
        return init(.bool, key, value_);
    }

    pub fn int64List(key: []const u8, comptime len: usize, values: *const [len]i64) NamedValue {
        return init(.int64list, key, values[0..]);
    }

    pub fn ptr(values: []NamedValue) [*c]c.PJRT_NamedValue {
        return @ptrCast(values.ptr);
    }

    pub fn view(inner: c.PJRT_NamedValue) NamedValue {
        return .{ .inner = inner };
    }

    pub fn kind(self: NamedValue) Kind {
        return @enumFromInt(self.inner.type);
    }

    pub fn name(self: NamedValue) []const u8 {
        return Slice.bytes(self.inner.name, self.inner.name_size) orelse &.{};
    }

    pub fn value(self: NamedValue) Value {
        return switch (self.kind()) {
            .string => .{ .string = Slice.bytes(self.inner.unnamed_0.string_value, self.inner.value_size) orelse &.{} },
            .int64 => .{ .int64 = self.inner.unnamed_0.int64_value },
            .int64list => .{ .int64list = self.inner.unnamed_0.int64_array_value[0..self.inner.value_size] },
            .float => .{ .float = self.inner.unnamed_0.float_value },
            .bool => .{ .bool = self.inner.unnamed_0.bool_value },
        };
    }

    pub fn stringValue(self: NamedValue) ?[]const u8 {
        return switch (self.value()) {
            .string => |text| text,
            else => null,
        };
    }
};

pub const Slice = struct {
    pub fn ptrList(values: [][*c]const u8) [*c][*c]const u8 {
        return @ptrCast(values.ptr);
    }

    pub fn bytes(ptr: [*c]const u8, len: usize) ?[]const u8 {
        if (ptr == null and len != 0) return null;
        if (len == 0) return &.{};
        return ptr[0..len];
    }

    pub fn mutBytes(ptr: ?*anyopaque, len: usize) ?[]u8 {
        if (ptr == null and len != 0) return null;
        if (len == 0) return &.{};
        return @as([*]u8, @ptrCast(ptr.?))[0..len];
    }

    pub fn constBytes(ptr: ?*const anyopaque, len: usize) ?[]const u8 {
        if (ptr == null and len != 0) return null;
        if (len == 0) return &.{};
        return @as([*]const u8, @ptrCast(ptr.?))[0..len];
    }
};

pub const Args = struct {
    pub fn writeBytes(comptime ptr_field: []const u8, comptime len_field: []const u8, args: anytype, value: []const u8) void {
        @field(args, ptr_field) = value.ptr;
        @field(args, len_field) = value.len;
    }
};

pub const Placement = struct {
    pub fn deviceIndex(client: *const runtime.Client, device: *const runtime.Device) ?usize {
        for (client.devices, 0..) |*candidate, i| {
            if (candidate == device or candidate.id == device.id) return i;
        }
        return null;
    }
};

test "named value helpers build PJRT strings and lists" {
    const text = NamedValue.string("name", "value");
    try std.testing.expectEqualStrings("name", text.name());
    try std.testing.expectEqualStrings("value", text.stringValue().?);

    const values = [_]i64{ 1, 16, 0 };
    const list = NamedValue.int64List("version", values.len, &values);
    try std.testing.expectEqual(@as(usize, 3), list.inner.value_size);
    try std.testing.expectEqual(NamedValue.Kind.int64list, list.kind());
    try std.testing.expectEqualSlices(i64, &values, list.value().int64list);
}
