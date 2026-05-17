const std = @import("std");
const backend = @import("src/backend");
const core = @import("src/core");

pub fn create() backend.Backend {
    return .{ .vtable = &vtable };
}

fn kind(_: backend.Backend) core.BackendKind {
    return .synthetic;
}

fn capabilities(_: backend.Backend) backend.Capabilities {
    return .{
        .kind = .synthetic,
        .name = "synthetic",
        .supports_device_buffers = false,
        .supports_unified_memory = true,
    };
}

fn enumerateDevices(_: backend.Backend, allocator: std.mem.Allocator, device_count_hint: usize) backend.Error![]core.DeviceDescriptor {
    if (device_count_hint == 0 or device_count_hint > core.MAX_DEVICES) return error.InvalidDeviceCount;
    const devices = try allocator.alloc(core.DeviceDescriptor, device_count_hint);
    errdefer allocator.free(devices);
    for (devices, 0..) |*device, i| {
        const id: i32 = @intCast(i);
        device.* = .{
            .id = id,
            .local_hardware_id = id,
            .name = try allocator.dupe(u8, "Synthetic device"),
            .debug_string = try allocator.dupe(u8, "PjRTx synthetic backend device"),
            .has_unified_memory = true,
            .default_memory_id = id,
        };
    }
    return devices;
}

fn releaseDeviceDescriptors(_: backend.Backend, allocator: std.mem.Allocator, descriptors: []core.DeviceDescriptor) void {
    for (descriptors) |descriptor| {
        allocator.free(descriptor.name);
        allocator.free(descriptor.debug_string);
    }
    allocator.free(descriptors);
}

fn noBufferFromHost(_: backend.Backend, _: i32, _: core.BufferType, _: []const i64, _: []const u8) backend.Error!?backend.BufferHandle {
    return null;
}

fn noClone(_: backend.Backend, _: backend.BufferHandle) backend.Error!?backend.BufferHandle {
    return null;
}

fn noBinary(_: backend.Backend, _: backend.BufferHandle, _: backend.BufferHandle, _: core.ElementwiseBinaryOp) backend.Error!?backend.BufferHandle {
    return null;
}

fn noUnary(_: backend.Backend, _: backend.BufferHandle, _: core.ElementwiseUnaryOp) backend.Error!?backend.BufferHandle {
    return null;
}

fn noReshape(_: backend.Backend, _: i32, _: core.BufferType, _: []const u8, _: []const i64) backend.Error!?backend.BufferHandle {
    return null;
}

fn noTranspose(_: backend.Backend, _: backend.BufferHandle, _: []const i64) backend.Error!?backend.BufferHandle {
    return null;
}

fn noBroadcast(_: backend.Backend, _: backend.BufferHandle, _: []const i64, _: []const i64) backend.Error!?backend.BufferHandle {
    return null;
}

fn noSlice(_: backend.Backend, _: backend.BufferHandle, _: []const i64, _: []const i64, _: []const i64, _: []const i64) backend.Error!?backend.BufferHandle {
    return null;
}

fn noConcatenate(_: backend.Backend, _: backend.BufferHandle, _: backend.BufferHandle, _: i64, _: []const i64) backend.Error!?backend.BufferHandle {
    return null;
}

fn noCopy(_: backend.Backend, _: backend.BufferHandle, _: []u8) backend.Error!void {
    return error.BufferCopyFailed;
}

fn noDestroy(_: backend.Backend, _: backend.BufferHandle) void {}

const vtable: backend.Backend.VTable = .{
    .kind = kind,
    .capabilities = capabilities,
    .enumerateDevices = enumerateDevices,
    .releaseDeviceDescriptors = releaseDeviceDescriptors,
    .bufferFromHost = noBufferFromHost,
    .cloneBuffer = noClone,
    .binary = noBinary,
    .unary = noUnary,
    .reshape = noReshape,
    .transpose = noTranspose,
    .broadcastInDim = noBroadcast,
    .slice = noSlice,
    .concatenate = noConcatenate,
    .copyToHost = noCopy,
    .destroyBuffer = noDestroy,
};

test "synthetic backend enumerates requested devices" {
    const b = create();
    const devices = try b.enumerateDevices(std.testing.allocator, 2);
    defer b.releaseDeviceDescriptors(std.testing.allocator, devices);
    try std.testing.expectEqual(@as(usize, 2), devices.len);
    try std.testing.expectEqual(core.BackendKind.synthetic, b.kind());
}
