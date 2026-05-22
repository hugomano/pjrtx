const c = @import("c");

const raw = @import("mlx_call_raw.zig");

const DeviceInfo = raw.DeviceInfo;
const DeviceNameBytes = raw.DeviceNameBytes;

/// Copies MLX/Metal device metadata into Zig-owned boundary structs.
pub fn copyDevices(out_devices: []DeviceInfo) i32 {
    var raw_devices: [256]c.PjrtxMlxMetalDeviceInfo = undefined;
    const max_devices = @min(out_devices.len, raw_devices.len);
    const copied = c.pjrtx_mlx_metal_copy_devices(&raw_devices, @intCast(max_devices));
    if (copied <= 0) return copied;
    const count: usize = @min(@as(usize, @intCast(copied)), max_devices);
    for (out_devices[0..count], raw_devices[0..count]) |*out_device, raw_device| {
        out_device.* = .{
            .ordinal = raw_device.ordinal,
            .registry_id = raw_device.registry_id,
            .recommended_max_working_set_size = raw_device.recommended_max_working_set_size,
            .has_unified_memory = raw_device.has_unified_memory != 0,
            .name = raw_device.name,
        };
    }
    return copied;
}

