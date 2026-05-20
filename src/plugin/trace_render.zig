const std = @import("std");

fn token(writer: *std.Io.Writer, text: []const u8) !void {
    for (text) |byte| switch (byte) {
        'a'...'z', 'A'...'Z', '0'...'9', '_', '-', '.', '/', ':', '+', '=' => try writer.writeByte(byte),
        ' ', '\t', '\n', '\r' => try writer.writeByte('_'),
        else => try writer.print("%{x:0>2}", .{byte}),
    };
}

fn typeName(comptime T: type) []const u8 {
    const full = @typeName(T);
    const dot = std.mem.lastIndexOfScalar(u8, full, '.') orelse return full;
    const short = full[dot + 1 ..];
    return if (std.mem.startsWith(u8, short, "struct_")) short["struct_".len..] else short;
}

fn ignored(comptime name: []const u8) bool {
    return std.mem.eql(u8, name, "struct_size") or std.mem.eql(u8, name, "extension_start");
}

fn pointer(comptime T: type, ptr_value: T, writer: *std.Io.Writer) !void {
    const pointer_meta = @typeInfo(T).pointer;
    switch (pointer_meta.size) {
        .c => if (ptr_value == null) try writer.writeAll("null") else try writer.print("{s}(non_null)", .{typeName(pointer_meta.child)}),
        .slice => try writer.print("{s}[{d}]", .{ typeName(pointer_meta.child), ptr_value.len }),
        else => try writer.print("{s}(non_null)", .{typeName(pointer_meta.child)}),
    }
}

fn value(comptime T: type, item: T, writer: *std.Io.Writer, comptime depth: u8) !void {
    switch (@typeInfo(T)) {
        .void, .null => try writer.writeAll("null"),
        .bool => try writer.print("{}", .{item}),
        .int, .comptime_int => try writer.print("{d}", .{item}),
        .float, .comptime_float => try writer.print("{d}", .{item}),
        .@"enum" => try writer.writeAll(@tagName(item)),
        .pointer => try pointer(T, item, writer),
        .optional => if (item) |some| try value(@TypeOf(some), some, writer, depth) else try writer.writeAll("null"),
        .array => |array| {
            if (array.child == u8) return token(writer, &item);
            try writer.writeByte('[');
            for (item, 0..) |child, index| {
                if (index != 0) try writer.writeByte(',');
                try value(@TypeOf(child), child, writer, depth + 1);
            }
            try writer.writeByte(']');
        },
        .@"struct" => |struct_meta| {
            if (depth > 1) return writer.print("{s}{{...}}", .{typeName(T)});
            try writer.print("{s}{{", .{typeName(T)});
            var first = true;
            inline for (struct_meta.fields) |field| if (!ignored(field.name)) {
                if (!first) try writer.writeByte(',');
                first = false;
                try writer.print("{s}=", .{field.name});
                try value(field.type, @field(item, field.name), writer, depth + 1);
            };
            try writer.writeByte('}');
        },
        else => try writer.print("{s}", .{typeName(T)}),
    }
}

/// Renders PJRT callback arguments and values for structured trace lines.
pub const Render = struct {
    /// Writes text as a single trace-safe token.
    pub fn tokenText(writer: *std.Io.Writer, text: []const u8) !void {
        try token(writer, text);
    }

    /// Writes an arbitrary value using the generic PJRT trace renderer.
    pub fn valueOf(comptime T: type, item: T, writer: *std.Io.Writer, comptime depth: u8) !void {
        try value(T, item, writer, depth);
    }

    /// Renders a PJRT callback argument pointer into the supplied scratch buffer.
    pub fn args(comptime Args: type, raw: Args, out: *std.Io.Writer.Allocating) []const u8 {
        if (raw == null) return "null";
        value(@typeInfo(Args).pointer.child, raw[0], &out.writer, 0) catch {
            out.clearRetainingCapacity();
            out.writer.writeAll("<trace_args_error>") catch {};
        };
        return out.writer.buffered();
    }
};
