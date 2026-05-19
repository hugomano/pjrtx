const std = @import("std");

const runtime = @import("src/runtime");
const c = @import("c");
const abi = @import("pjrt_abi.zig");
const errors = @import("errors.zig");
const state = @import("state.zig");

const allocator = state.allocator;
const platform_name = state.platform_name;
const platform_version = state.platform_version;
const SerializedTopology = state.SerializedTopology;
const SerializedTopologyHandle = abi.SerializedTopology(SerializedTopology);
const makeError = errors.makeError;
const unimplemented = errors.unimplemented;

const TopologyDescription = struct {
    client: *const runtime.Client,

    fn at(raw: anytype) TopologyDescription {
        return .{ .client = abi.TopologyDescription.viewConst(raw) };
    }

    fn platformName(self: TopologyDescription) []const u8 {
        _ = self;
        return platform_name;
    }

    fn platformVersion(self: TopologyDescription) []const u8 {
        _ = self;
        return platform_version;
    }

    fn devices(self: TopologyDescription) []const *runtime.Device {
        return self.client.device_handles;
    }

    fn fingerprint(self: TopologyDescription) u64 {
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(platform_name);
        hasher.update(platform_version);
        for (self.client.devices) |device| {
            hasher.update(std.mem.asBytes(&device.id));
            hasher.update(std.mem.asBytes(&device.local_hardware_id));
            hasher.update(std.mem.asBytes(&device.process_index));
            hasher.update(device.name);
        }
        return hasher.final();
    }

    fn serialize(self: TopologyDescription) ?*SerializedTopology {
        var writer = std.Io.Writer.Allocating.init(allocator);
        defer writer.deinit();
        writer.writer.print("platform={s};version={s};devices={d}", .{ platform_name, platform_version, self.client.devices.len }) catch return null;
        for (self.client.devices) |device| {
            writer.writer.print(";device={d}:{d}:{d}:{s}", .{ device.id, device.local_hardware_id, device.process_index, device.name }) catch return null;
        }
        const topology = allocator.create(SerializedTopology) catch return null;
        topology.* = .{ .bytes = writer.toOwnedSlice() catch {
            allocator.destroy(topology);
            return null;
        } };
        return topology;
    }
};

const TopologyOp = enum {
    platform_name,
    platform_version,
    device_descriptions,
    attributes,
    fingerprint,
};

fn TopologyCallback(comptime Args: type, comptime op: TopologyOp) type {
    return struct {
        fn call(raw: [*c]Args) callconv(.c) ?*c.PJRT_Error {
            const args = &raw[0];
            const topology = TopologyDescription.at(args.topology);
            switch (op) {
                .platform_name => abi.writeBytes("platform_name", "platform_name_size", args, topology.platformName()),
                .platform_version => abi.writeBytes("platform_version", "platform_version_size", args, topology.platformVersion()),
                .device_descriptions => {
                    const devices = topology.devices();
                    args.descriptions = abi.DeviceDescription.handleSlice(devices);
                    args.num_descriptions = devices.len;
                },
                .attributes => {
                    args.attributes = null;
                    args.num_attributes = 0;
                },
                .fingerprint => args.fingerprint = topology.fingerprint(),
            }
            return null;
        }
    };
}

pub fn topologyDescriptionCreate(_: [*c]c.PJRT_TopologyDescription_Create_Args) callconv(.c) ?*c.PJRT_Error {
    return unimplemented("standalone topology creation is not implemented yet");
}
pub fn topologyDescriptionDestroy(_: [*c]c.PJRT_TopologyDescription_Destroy_Args) callconv(.c) ?*c.PJRT_Error {
    return null;
}
pub fn topologySerializedDelete(serialized_topology: ?*c.PJRT_SerializedTopology) callconv(.c) void {
    if (serialized_topology) |opaque_topology| {
        const topology = SerializedTopologyHandle.view(opaque_topology);
        allocator.free(topology.bytes);
        allocator.destroy(topology);
    }
}
pub fn topologyDescriptionSerialize(args: [*c]c.PJRT_TopologyDescription_Serialize_Args) callconv(.c) ?*c.PJRT_Error {
    const topology = TopologyDescription.at(args[0].topology).serialize() orelse return makeError(c.PJRT_Error_Code_INTERNAL, "failed to serialize topology");

    args[0].serialized_bytes = topology.bytes.ptr;
    args[0].serialized_bytes_size = topology.bytes.len;
    args[0].serialized_topology = SerializedTopologyHandle.handle(topology);
    args[0].serialized_topology_deleter = topologySerializedDelete;
    return null;
}
pub fn topologyDescriptionDeserialize(_: [*c]c.PJRT_TopologyDescription_Deserialize_Args) callconv(.c) ?*c.PJRT_Error {
    return unimplemented("topology deserialization is not implemented yet");
}
pub const Api = struct {
    pub const Create = topologyDescriptionCreate;
    pub const Destroy = topologyDescriptionDestroy;
    pub const PlatformName = TopologyCallback(c.PJRT_TopologyDescription_PlatformName_Args, .platform_name).call;
    pub const PlatformVersion = TopologyCallback(c.PJRT_TopologyDescription_PlatformVersion_Args, .platform_version).call;
    pub const GetDeviceDescriptions = TopologyCallback(c.PJRT_TopologyDescription_GetDeviceDescriptions_Args, .device_descriptions).call;
    pub const Serialize = topologyDescriptionSerialize;
    pub const Deserialize = topologyDescriptionDeserialize;
    pub const Attributes = TopologyCallback(c.PJRT_TopologyDescription_Attributes_Args, .attributes).call;
    pub const Fingerprint = TopologyCallback(c.PJRT_TopologyDescription_Fingerprint_Args, .fingerprint).call;
};
