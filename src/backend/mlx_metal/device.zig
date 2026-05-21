const std = @import("std");

const ir = @import("src/compiler/ir");
const mlx_call = @import("mlx_call.zig");

/// Static MLX/Metal backend capabilities reported to the runtime.
pub const Capabilities = struct {
    /// Stable backend name used in diagnostics and compile-cache keys.
    name: []const u8,
    /// Whether runtime buffers are backed by device-resident MLX/Metal storage.
    supports_device_buffers: bool,
    /// Whether the backend device memory model is unified with host-visible memory.
    supports_unified_memory: bool,
    /// Whether execution may complete asynchronously through backend events.
    supports_async_execution: bool = false,
};

/// Returns the capabilities of the concrete MLX/Metal backend.
pub fn capabilities() Capabilities {
    return .{
        .name = "metal_mlx",
        .supports_device_buffers = true,
        .supports_unified_memory = true,
    };
}

/// Enumerates MLX/Metal devices and converts runtime metadata into compiler IR descriptors.
pub const DeviceList = struct {
    /// Copies device descriptors for the backend boundary; caller owns the returned slice.
    pub fn enumerate(allocator: std.mem.Allocator) Error![]ir.DeviceDescriptor {
        var metal_devices: [ir.MAX_DEVICES]mlx_call.DeviceInfo = undefined;
        const copied = mlx_call.copyDevices(&metal_devices);
        if (copied <= 0) return error.InvalidDeviceCount;
        const count: usize = @intCast(copied);
        const devices = try allocator.alloc(ir.DeviceDescriptor, count);
        errdefer allocator.free(devices);

        for (devices, 0..) |*out_device, index| {
            const metal_device = metal_devices[index];
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
                .has_unified_memory = metal_device.has_unified_memory,
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
pub const Error = error{ InvalidDeviceCount, OutOfMemory };

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

test "device names stop at first nul byte" {
    var name = std.mem.zeroes([mlx_call.DeviceNameBytes]u8);
    @memcpy(name[0.."Metal".len], "Metal");
    try std.testing.expectEqualStrings("Metal", nameBytes(&name));
}
