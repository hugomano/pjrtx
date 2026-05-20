const std = @import("std");

const c = @import("c");
const runtime = @import("src/runtime");

const abi = @import("pjrt_abi.zig");
const errors = @import("errors.zig");
const handles = @import("pjrt_handles.zig");
const plugin = @import("plugin.zig");

const PjrtError = errors.Error;

const SerializedTopology = struct {
    bytes: []u8,
};

const SerializedTopologyHandle = handles.SerializedTopology(SerializedTopology);

const SerializedTopologyRef = struct {
    ptr: *SerializedTopology,

    fn at(raw: *c.PJRT_SerializedTopology) SerializedTopologyRef {
        return .{ .ptr = SerializedTopologyHandle.ref(raw) };
    }

    fn handle(self: SerializedTopologyRef) *c.PJRT_SerializedTopology {
        return SerializedTopologyHandle.handle(self.ptr);
    }

    fn delete(raw: ?*c.PJRT_SerializedTopology) callconv(.c) void {
        const opaque_topology = raw orelse return;
        const topology = at(opaque_topology).ptr;
        plugin.allocator().free(topology.bytes);
        plugin.allocator().destroy(topology);
    }
};

/// Borrowed PJRT topology description backed by a runtime client.
pub const TopologyDescription = struct {
    client: *const runtime.Client,

    pub const Api = struct {
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
        return .{ .client = handles.TopologyDescription.refConst(raw) };
    }

    fn platformName(self: TopologyDescription) []const u8 {
        _ = self;
        return plugin.Platform.name;
    }

    fn platformVersion(self: TopologyDescription) []const u8 {
        _ = self;
        return plugin.Platform.version;
    }

    fn devices(self: TopologyDescription) []const *runtime.Device {
        return self.client.device_handles;
    }

    fn fingerprint(self: TopologyDescription) u64 {
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(plugin.Platform.name);
        hasher.update(plugin.Platform.version);
        for (self.client.devices) |device| {
            hasher.update(std.mem.asBytes(&device.id));
            hasher.update(std.mem.asBytes(&device.local_hardware_id));
            hasher.update(std.mem.asBytes(&device.process_index));
            hasher.update(device.name);
        }
        return hasher.final();
    }

    fn serialize(self: TopologyDescription) ?*SerializedTopology {
        var writer = std.Io.Writer.Allocating.init(plugin.allocator());
        defer writer.deinit();
        writer.writer.print("platform={s};version={s};devices={d}", .{ plugin.Platform.name, plugin.Platform.version, self.client.devices.len }) catch return null;
        for (self.client.devices) |device| {
            writer.writer.print(";device={d}:{d}:{d}:{s}", .{ device.id, device.local_hardware_id, device.process_index, device.name }) catch return null;
        }
        const topology = plugin.allocator().create(SerializedTopology) catch return null;
        topology.* = .{ .bytes = writer.toOwnedSlice() catch {
            plugin.allocator().destroy(topology);
            return null;
        } };
        return topology;
    }

    fn writeSerialized(self: TopologyDescription, args: anytype) ?*c.PJRT_Error {
        const topology = self.serialize() orelse return PjrtError.internal("failed to serialize topology");
        const serialized: SerializedTopologyRef = .{ .ptr = topology };
        args.serialized_bytes = topology.bytes.ptr;
        args.serialized_bytes_size = topology.bytes.len;
        args.serialized_topology = serialized.handle();
        args.serialized_topology_deleter = SerializedTopologyRef.delete;
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
                .platform_name => abi.Out.writeBytes("platform_name", "platform_name_size", args, topology.platformName()),
                .platform_version => abi.Out.writeBytes("platform_version", "platform_version_size", args, topology.platformVersion()),
                .device_descriptions => {
                    const devices = topology.devices();
                    args.descriptions = handles.DeviceDescription.handleSlice(devices);
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
