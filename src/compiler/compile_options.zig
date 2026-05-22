const std = @import("std");
const ir = @import("src/compiler/ir");

/// Stable compile options accepted by PjRTx textual PJRT options.
pub const CompileOptions = ir.CompileOptions;

pub fn parseTextCompileOptionsFromReader(allocator: std.mem.Allocator, reader: *std.Io.Reader) !CompileOptions {
    const text = try reader.allocRemaining(allocator, .limited(64 * 1024));
    defer allocator.free(text);

    var options: CompileOptions = .{};
    var assignment = std.ArrayList(i32).empty;
    errdefer assignment.deinit(allocator);

    var it = std.mem.splitScalar(u8, text, ';');
    while (it.next()) |part| {
        const trimmed = std.mem.trim(u8, part, " \n\t");
        if (trimmed.len == 0) continue;
        const eq = std.mem.indexOfScalar(u8, trimmed, '=') orelse return error.InvalidCompileOption;
        const key = std.mem.trim(u8, trimmed[0..eq], " \n\t");
        const value = std.mem.trim(u8, trimmed[eq + 1 ..], " \n\t");

        if (std.mem.eql(u8, key, "replicas")) {
            options.num_replicas = try std.fmt.parseInt(i32, value, 10);
        } else if (std.mem.eql(u8, key, "partitions")) {
            options.num_partitions = try std.fmt.parseInt(i32, value, 10);
        } else if (std.mem.eql(u8, key, "use_shardy")) {
            options.use_shardy_partitioner = std.mem.eql(u8, value, "true") or std.mem.eql(u8, value, "1");
        } else if (std.mem.eql(u8, key, "assignment")) {
            var ids = std.mem.splitScalar(u8, value, ',');
            while (ids.next()) |id_text| {
                const id_trimmed = std.mem.trim(u8, id_text, " \n\t");
                if (id_trimmed.len == 0) continue;
                try assignment.append(allocator, try std.fmt.parseInt(i32, id_trimmed, 10));
            }
        } else {
            return error.UnknownCompileOption;
        }
    }

    if (options.num_replicas < 1 or options.num_partitions < 1) return error.InvalidDeviceTopology;
    if (assignment.items.len != 0 and assignment.items.len < options.numDevices()) return error.InvalidDeviceAssignment;
    options.device_assignment = try assignment.toOwnedSlice(allocator);
    return options;
}

pub fn parseTextCompileOptions(allocator: std.mem.Allocator, text: []const u8) !CompileOptions {
    var reader: std.Io.Reader = .fixed(text);
    return parseTextCompileOptionsFromReader(allocator, &reader);
}
