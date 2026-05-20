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
const PjrtError = errors.Error;

const SerializedTopologyView = struct {
    ptr: *SerializedTopology,

    fn at(raw: *c.PJRT_SerializedTopology) SerializedTopologyView {
        return .{ .ptr = SerializedTopologyHandle.view(raw) };
    }

    fn handle(self: SerializedTopologyView) *c.PJRT_SerializedTopology {
        return SerializedTopologyHandle.handle(self.ptr);
    }

    fn delete(raw: ?*c.PJRT_SerializedTopology) callconv(.c) void {
        const opaque_topology = raw orelse return;
        const topology = at(opaque_topology).ptr;
        allocator.free(topology.bytes);
        allocator.destroy(topology);
    }
};

const TopologyDescription = struct {
    client: *const runtime.Client,

    const Api = struct {
        pub const Create = TopologyLifecycle.create;
        pub const Destroy = TopologyLifecycle.destroy;
        pub const PlatformName = TopologyCallback(c.PJRT_TopologyDescription_PlatformName_Args, .platform_name).call;
        pub const PlatformVersion = TopologyCallback(c.PJRT_TopologyDescription_PlatformVersion_Args, .platform_version).call;
        pub const GetDeviceDescriptions = TopologyCallback(c.PJRT_TopologyDescription_GetDeviceDescriptions_Args, .device_descriptions).call;
        pub const Serialize = TopologyLifecycle.serialize;
        pub const Deserialize = TopologyLifecycle.deserialize;
        pub const Attributes = TopologyCallback(c.PJRT_TopologyDescription_Attributes_Args, .attributes).call;
        pub const Fingerprint = TopologyCallback(c.PJRT_TopologyDescription_Fingerprint_Args, .fingerprint).call;
    };

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

    fn writeSerialized(self: TopologyDescription, args: anytype) ?*c.PJRT_Error {
        const topology = self.serialize() orelse return PjrtError.make(c.PJRT_Error_Code_INTERNAL, "failed to serialize topology");
        const view: SerializedTopologyView = .{ .ptr = topology };
        args.serialized_bytes = topology.bytes.ptr;
        args.serialized_bytes_size = topology.bytes.len;
        args.serialized_topology = view.handle();
        args.serialized_topology_deleter = SerializedTopologyView.delete;
        return null;
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
                .platform_name => abi.Args.writeBytes("platform_name", "platform_name_size", args, topology.platformName()),
                .platform_version => abi.Args.writeBytes("platform_version", "platform_version_size", args, topology.platformVersion()),
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

const TopologyLifecycle = struct {
    fn create(_: [*c]c.PJRT_TopologyDescription_Create_Args) callconv(.c) ?*c.PJRT_Error {
        return PjrtError.unimplemented("standalone topology creation is not implemented yet");
    }

    fn destroy(_: [*c]c.PJRT_TopologyDescription_Destroy_Args) callconv(.c) ?*c.PJRT_Error {
        return null;
    }

    fn serialize(raw: [*c]c.PJRT_TopologyDescription_Serialize_Args) callconv(.c) ?*c.PJRT_Error {
        return TopologyDescription.at(raw[0].topology).writeSerialized(&raw[0]);
    }

    fn deserialize(_: [*c]c.PJRT_TopologyDescription_Deserialize_Args) callconv(.c) ?*c.PJRT_Error {
        return PjrtError.unimplemented("topology deserialization is not implemented yet");
    }
};

pub const Api = TopologyDescription.Api;
