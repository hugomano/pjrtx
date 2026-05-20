const c = @import("c");
const runtime = @import("src/runtime");

const abi = @import("pjrt_abi.zig");

/// Opaque PJRT handle conversions for runtime clients.
pub const Client = abi.Opaque(runtime.Client, c.PJRT_Client);

/// Opaque PJRT handle conversions for topology descriptions.
pub const TopologyDescription = abi.Opaque(runtime.Client, c.PJRT_TopologyDescription);

/// Opaque PJRT handle conversions for runtime devices.
pub const Device = abi.Opaque(runtime.Device, c.PJRT_Device);

/// Opaque PJRT handle conversions for runtime device descriptions.
pub const DeviceDescription = abi.Opaque(runtime.Device, c.PJRT_DeviceDescription);

/// Opaque PJRT handle conversions for runtime memories.
pub const Memory = abi.Opaque(runtime.Memory, c.PJRT_Memory);

/// Opaque PJRT handle conversions for runtime buffers.
pub const Buffer = abi.Opaque(runtime.Buffer, c.PJRT_Buffer);

/// Opaque PJRT handle conversions for runtime events.
pub const Event = abi.Opaque(runtime.Event, c.PJRT_Event);

/// Creates opaque PJRT executable handle conversions for an executable owner.
pub fn Executable(comptime Owner: type) type {
    return abi.Opaque(Owner, c.PJRT_Executable);
}

/// Creates opaque PJRT loaded-executable handle conversions for an owner.
pub fn LoadedExecutable(comptime Owner: type) type {
    return abi.Opaque(Owner, c.PJRT_LoadedExecutable);
}

/// Creates opaque PJRT async-transfer-manager handle conversions for an owner.
pub fn AsyncHostToDeviceTransferManager(comptime Owner: type) type {
    return abi.Opaque(Owner, c.PJRT_AsyncHostToDeviceTransferManager);
}

/// Creates opaque PJRT serialized-topology handle conversions for an owner.
pub fn SerializedTopology(comptime Owner: type) type {
    return abi.Opaque(Owner, c.PJRT_SerializedTopology);
}

/// Creates opaque PJRT device-attributes handle conversions for an owner.
pub fn DeviceAttributes(comptime Owner: type) type {
    return abi.Opaque(Owner, c.PJRT_Device_Attributes);
}

/// Creates opaque user-argument handle conversions for plugin callback payloads.
pub fn UserArg(comptime Owner: type) type {
    return abi.Opaque(Owner, anyopaque);
}
