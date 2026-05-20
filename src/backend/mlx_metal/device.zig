const std = @import("std");

const ir = @import("src/compiler/ir");
const mlx_call = @import("mlx_call.zig");

/// Enumerates MLX/Metal devices and converts runtime metadata into compiler IR descriptors.
pub const DeviceList = struct {
    /// Copies device descriptors for the backend boundary; caller owns the returned slice.
    pub fn enumerate(allocator: std.mem.Allocator) Error![]ir.DeviceDescriptor {
        var metal_devices: [ir.MAX_DEVICES]mlx_call.DeviceInfo = undefined;
        const copied = mlx_call.copyDevices(&metal_devices);
        const count: usize = if (copied <= 0) 1 else @intCast(copied);
        const devices = try allocator.alloc(ir.DeviceDescriptor, count);
        errdefer allocator.free(devices);

        for (devices, 0..) |*out_device, index| {
            const has_metal_device = copied > 0;
            const metal_device = if (has_metal_device) metal_devices[index] else fallbackDevice(@intCast(index));
            const name = try allocator.dupe(u8, nameBytes(metal_device.name[0..]));
            errdefer allocator.free(name);
            const debug_string = try debugString(allocator, index, name);
            errdefer allocator.free(debug_string);

            const id: i32 = @intCast(index);
            out_device.* = .{
                .id = id,
                .local_hardware_id = metal_device.ordinal,
                .registry_id = metal_device.registry_id,
                .name = name,
                .debug_string = debug_string,
                .memory_bytes = metal_device.recommended_max_working_set_size,
                .has_unified_memory = !has_metal_device or metal_device.has_unified_memory,
                .default_memory_id = id,
            };
        }
        return devices;
    }

    /// Releases descriptor storage allocated by `enumerate`.
    pub fn release(allocator: std.mem.Allocator, descriptors: []ir.DeviceDescriptor) void {
        for (descriptors) |descriptor| {
            allocator.free(descriptor.name);
            allocator.free(descriptor.debug_string);
        }
        allocator.free(descriptors);
    }
};

/// Error set returned while converting MLX/Metal device metadata into descriptors.
pub const Error = error{OutOfMemory};

fn fallbackDevice(index: i32) mlx_call.DeviceInfo {
    var info = std.mem.zeroes(mlx_call.DeviceInfo);
    info.ordinal = index;
    info.has_unified_memory = true;
    @memcpy(info.name[0.."Metal/MLX device".len], "Metal/MLX device");
    return info;
}

fn nameBytes(name: []const u8) []const u8 {
    const end = std.mem.indexOfScalar(u8, name, 0) orelse name.len;
    return name[0..end];
}

fn debugString(allocator: std.mem.Allocator, index: usize, name: []const u8) Error![]const u8 {
    var buffer: [256]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buffer);
    writer.print("PjRTx Metal/MLX device {d}: {s}", .{ index, name }) catch return error.OutOfMemory;
    return allocator.dupe(u8, writer.buffered());
}

test "fallback device preserves existing descriptor defaults" {
    const device = fallbackDevice(0);
    try std.testing.expectEqual(@as(i32, 0), device.ordinal);
    try std.testing.expect(device.has_unified_memory);
    try std.testing.expectEqualStrings("Metal/MLX device", nameBytes(&device.name));
}
