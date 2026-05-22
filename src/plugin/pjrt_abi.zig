const std = @import("std");

const c = @import("c");

fn ref(comptime T: type, ptr: anytype) *T {
    return switch (@typeInfo(@TypeOf(ptr))) {
        .optional => @ptrCast(@alignCast(ptr.?)),
        .pointer => @ptrCast(@alignCast(ptr)),
        else => @compileError("PJRT ABI ref expects an opaque pointer"),
    };
}

fn refConst(comptime T: type, ptr: anytype) *const T {
    return switch (@typeInfo(@TypeOf(ptr))) {
        .optional => @ptrCast(@alignCast(ptr.?)),
        .pointer => @ptrCast(@alignCast(ptr)),
        else => @compileError("PJRT ABI const ref expects an opaque pointer"),
    };
}

/// Defines typed conversions between a PjRTx owner and an opaque PJRT handle.
pub fn Opaque(comptime Owner: type, comptime CHandle: type) type {
    return struct {
        /// Borrows a mutable PjRTx owner from an opaque PJRT handle.
        pub fn ref(raw: anytype) *Owner {
            return pjrt_abi.ref(Owner, raw);
        }

        /// Borrows an immutable PjRTx owner from an opaque PJRT handle.
        pub fn refConst(raw: anytype) *const Owner {
            return pjrt_abi.refConst(Owner, raw);
        }

        /// Reinterprets a PjRTx owner as the opaque PJRT handle exported to callers.
        pub fn handle(owner: anytype) *CHandle {
            return @ptrCast(owner);
        }

        /// Reinterprets a list of PjRTx owners as a PJRT handle array.
        pub fn handleSlice(owners: []const *Owner) [*c]*CHandle {
            return @ptrCast(@constCast(owners.ptr));
        }
    };
}

const pjrt_abi = @This();

/// PJRT named-value discriminant used by compile options and plugin attributes.
pub const NamedValueKind = enum(c.PJRT_NamedValue_Type) {
    string = c.PJRT_NamedValue_kString,
    int64 = c.PJRT_NamedValue_kInt64,
    int64list = c.PJRT_NamedValue_kInt64List,
    float = c.PJRT_NamedValue_kFloat,
    bool = c.PJRT_NamedValue_kBool,
};

/// Zig view of the tagged value carried by a PJRT named value.
pub const NamedValuePayload = union(NamedValueKind) {
    string: []const u8,
    int64: i64,
    int64list: []const i64,
    float: f32,
    bool: bool,
};

/// Typed wrapper over PJRT named values used for options and plugin attributes.
pub const NamedValue = extern struct {
    comptime {
        std.debug.assert(@sizeOf(NamedValue) == @sizeOf(c.PJRT_NamedValue));
    }

    raw: c.PJRT_NamedValue,

    /// PJRT named-value discriminant used by this wrapper.
    pub const Kind = NamedValueKind;
    /// Tagged payload exposed through the wrapper.
    pub const Value = NamedValuePayload;

    /// Builds a PJRT named value that borrows its name and payload.
    pub fn init(comptime kind_: Kind, name_: []const u8, value_: std.meta.fieldInfo(Value, kind_).type) NamedValue {
        return .{
            .raw = .{
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

    /// Builds a borrowed string named value.
    pub fn string(key: []const u8, value_: []const u8) NamedValue {
        return init(.string, key, value_);
    }

    /// Builds an integer named value.
    pub fn int64(key: []const u8, value_: i64) NamedValue {
        return init(.int64, key, value_);
    }

    /// Builds a boolean named value.
    pub fn boolean(key: []const u8, value_: bool) NamedValue {
        return init(.bool, key, value_);
    }

    /// Builds a borrowed integer-list named value.
    pub fn int64List(key: []const u8, comptime len: usize, values: *const [len]i64) NamedValue {
        return init(.int64list, key, values[0..]);
    }

    /// Exposes a named-value array to PJRT without copying it.
    pub fn borrowedMany(values: []NamedValue) [*c]c.PJRT_NamedValue {
        return @ptrCast(values.ptr);
    }

    /// Wraps a PJRT named value supplied by the caller.
    pub fn borrow(raw: c.PJRT_NamedValue) NamedValue {
        return .{ .raw = raw };
    }

    /// Returns the tagged kind of a PJRT named value.
    pub fn kind(self: NamedValue) Kind {
        return @enumFromInt(self.raw.type);
    }

    /// Returns the borrowed PJRT named-value key.
    pub fn name(self: NamedValue) []const u8 {
        return Slice.bytes(self.raw.name, self.raw.name_size) orelse &.{};
    }

    /// Returns the borrowed or scalar payload of a PJRT named value.
    pub fn value(self: NamedValue) Value {
        return switch (self.kind()) {
            .string => .{ .string = Slice.bytes(self.raw.unnamed_0.string_value, self.raw.value_size) orelse &.{} },
            .int64 => .{ .int64 = self.raw.unnamed_0.int64_value },
            .int64list => .{ .int64list = self.raw.unnamed_0.int64_array_value[0..self.raw.value_size] },
            .float => .{ .float = self.raw.unnamed_0.float_value },
            .bool => .{ .bool = self.raw.unnamed_0.bool_value },
        };
    }

    /// Returns the PJRT payload element count for tests and ABI validation.
    pub fn valueSize(self: NamedValue) usize {
        return self.raw.value_size;
    }

    /// Returns the string payload when this named value is a string.
    pub fn stringValue(self: NamedValue) ?[]const u8 {
        return switch (self.value()) {
            .string => |text| text,
            else => null,
        };
    }
};

/// Converts C pointer/length pairs into Zig slices at the PJRT ABI boundary.
pub const Slice = struct {
    /// Exposes an array of byte pointers to PJRT without copying it.
    pub fn ptrList(values: [][*c]const u8) [*c][*c]const u8 {
        return @ptrCast(values.ptr);
    }

    /// Borrows a typed caller-owned pointer/count list.
    pub fn constList(comptime T: type, ptr: [*c]const T, len: usize) ?[]const T {
        if (ptr == null and len != 0) return null;
        if (len == 0) return &.{};
        return ptr[0..len];
    }

    /// Borrows a caller-owned byte pointer/count pair.
    pub fn bytes(ptr: [*c]const u8, len: usize) ?[]const u8 {
        if (ptr == null and len != 0) return null;
        if (len == 0) return &.{};
        return ptr[0..len];
    }

    /// Borrows a mutable caller-owned byte buffer.
    pub fn mutBytes(ptr: ?*anyopaque, len: usize) ?[]u8 {
        if (ptr == null and len != 0) return null;
        if (len == 0) return &.{};
        return @as([*]u8, @ptrCast(ptr.?))[0..len];
    }

    /// Borrows an immutable caller-owned byte buffer.
    pub fn constBytes(ptr: ?*const anyopaque, len: usize) ?[]const u8 {
        if (ptr == null and len != 0) return null;
        if (len == 0) return &.{};
        return @as([*]const u8, @ptrCast(ptr.?))[0..len];
    }
};

/// Writes borrowed values into PJRT callback output fields.
pub const Out = struct {
    /// Writes a byte slice into paired pointer and length fields.
    pub fn writeBytes(comptime ptr_field: []const u8, comptime len_field: []const u8, args: anytype, value: []const u8) void {
        @field(args, ptr_field) = value.ptr;
        @field(args, len_field) = value.len;
    }
};

/// Numeric conversions needed when writing PJRT ABI scalar fields.
pub const Scalar = struct {
    /// Clamps an unsigned byte/count value into a signed PJRT `int64_t` field.
    pub fn clampI64(value: u64) i64 {
        return @intCast(@min(value, @as(u64, @intCast(std.math.maxInt(i64)))));
    }
};

test "named value helpers build PJRT strings and lists" {
    const text = NamedValue.string("name", "value");
    try std.testing.expectEqualStrings("name", text.name());
    try std.testing.expectEqualStrings("value", text.stringValue().?);

    const values = [_]i64{ 1, 16, 0 };
    const list = NamedValue.int64List("version", values.len, &values);
    try std.testing.expectEqual(@as(usize, 3), list.valueSize());
    try std.testing.expectEqual(NamedValue.Kind.int64list, list.kind());
    try std.testing.expectEqualSlices(i64, &values, list.value().int64list);
}
