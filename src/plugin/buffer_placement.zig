const c = @import("c");
const runtime = @import("src/runtime");

const handles = @import("pjrt_handles.zig");

/// Maps a runtime device pointer back to its index in a PJRT client topology.
fn deviceIndex(client: *const runtime.Client, device: *const runtime.Device) ?usize {
    return client.deviceIndex(device);
}

/// Describes where a PJRT buffer is placed in a client topology.
pub const Placement = struct {
    client: *runtime.Client,
    device: *runtime.Device,
    memory: *runtime.Memory,
    shard_index: usize,

    /// Placement resolution failures that map to PJRT invalid-argument errors.
    pub const Error = error{ MemoryHasNoDevices, DeviceNotInClient, MemoryNotAddressable };

    /// Resolves optional PJRT device/memory handles for host-imported buffers.
    pub fn forHostBuffer(client: *runtime.Client, device_arg: ?*c.PJRT_Device, memory_arg: ?*c.PJRT_Memory) Error!Placement {
        const device = if (device_arg) |dev| handles.Device.ref(dev) else client.defaultDevice();
        const memory = if (memory_arg) |mem| handles.Memory.ref(mem) else device.default_memory;
        return forResolved(client, device, memory);
    }

    /// Resolves optional PJRT device/memory handles for device-allocated buffers.
    pub fn forDeviceBuffer(client: *runtime.Client, device_arg: ?*c.PJRT_Device, memory_arg: ?*c.PJRT_Memory) Error!Placement {
        const memory = if (memory_arg) |mem| handles.Memory.ref(mem) else blk: {
            const device = if (device_arg) |dev| handles.Device.ref(dev) else client.defaultDevice();
            break :blk device.default_memory;
        };
        const device = if (device_arg) |dev| handles.Device.ref(dev) else blk: {
            if (memory.addressable_devices.len == 0) return error.MemoryHasNoDevices;
            break :blk memory.addressable_devices[0];
        };
        return forResolved(client, device, memory);
    }

    /// Returns the logical shard index for a resolved client device.
    pub fn index(client: *const runtime.Client, device: *const runtime.Device) ?usize {
        return deviceIndex(client, device);
    }

    fn forResolved(client: *runtime.Client, device: *runtime.Device, memory: *runtime.Memory) Error!Placement {
        const shard_index = deviceIndex(client, device) orelse return error.DeviceNotInClient;
        if (!memory.isAddressableBy(device)) return error.MemoryNotAddressable;
        return .{
            .client = client,
            .device = device,
            .memory = memory,
            .shard_index = shard_index,
        };
    }
};
