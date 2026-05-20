const runtime = @import("src/runtime");
const c = @import("c");
const abi = @import("pjrt_abi.zig");
const errors = @import("errors.zig");
const PjrtError = errors.Error;
const state = @import("state.zig");

const allocator = state.allocator;

const DeviceDescription = struct {
    device: *runtime.Device,

    const Api = struct {
        pub const Id = DeviceDescriptionScalar(c.PJRT_DeviceDescription_Id_Args, "id", .id).call;
        pub const ProcessIndex = DeviceDescriptionScalar(c.PJRT_DeviceDescription_ProcessIndex_Args, "process_index", .process_index).call;
        pub const Attributes = DeviceDescriptionCallback(c.PJRT_DeviceDescription_Attributes_Args, .attributes).call;
        pub const Kind = DeviceDescriptionText(c.PJRT_DeviceDescription_Kind_Args, "device_kind", "device_kind_size", .kind).call;
        pub const DebugString = DeviceDescriptionText(c.PJRT_DeviceDescription_DebugString_Args, "debug_string", "debug_string_size", .debug_string).call;
        pub const ToString = DeviceDescriptionText(c.PJRT_DeviceDescription_ToString_Args, "to_string", "to_string_size", .debug_string).call;
    };

    fn at(raw: anytype) DeviceDescription {
        return .{ .device = abi.DeviceDescription.view(raw) };
    }

    fn id(self: DeviceDescription) i32 {
        return self.device.id;
    }

    fn processIndex(self: DeviceDescription) i32 {
        return self.device.process_index;
    }

    fn kind(_: DeviceDescription) []const u8 {
        return state.device_kind;
    }

    fn debugString(self: DeviceDescription) []const u8 {
        return self.device.debug_string;
    }
};

const Device = struct {
    ptr: *runtime.Device,

    const Api = struct {
        pub const GetDescription = DeviceCallback(c.PJRT_Device_GetDescription_Args, .description).call;
        pub const IsAddressable = DeviceCallback(c.PJRT_Device_IsAddressable_Args, .is_addressable).call;
        pub const LocalHardwareId = DeviceCallback(c.PJRT_Device_LocalHardwareId_Args, .local_hardware_id).call;
        pub const AddressableMemories = DeviceCallback(c.PJRT_Device_AddressableMemories_Args, .addressable_memories).call;
        pub const DefaultMemory = DeviceCallback(c.PJRT_Device_DefaultMemory_Args, .default_memory).call;
        pub const MemoryStats = DeviceCallback(c.PJRT_Device_MemoryStats_Args, .memory_stats).call;
        pub const GetAttributes = DeviceCallback(c.PJRT_Device_GetAttributes_Args, .attributes).call;
    };

    fn at(raw: anytype) Device {
        return .{ .ptr = abi.Device.view(raw) };
    }

    fn description(self: Device) *c.PJRT_DeviceDescription {
        return abi.DeviceDescription.handle(self.ptr);
    }

    fn addressableMemories(self: Device) []const *runtime.Memory {
        return self.ptr.addressable_memories;
    }

    fn defaultMemory(self: Device) *runtime.Memory {
        return self.ptr.default_memory;
    }

    fn isAddressable(self: Device) bool {
        return self.ptr.addressable;
    }

    fn localHardwareId(self: Device) i32 {
        return self.ptr.local_hardware_id;
    }

    fn writeAddressableMemories(self: Device, args: anytype) void {
        const memories = self.addressableMemories();
        args.memories = abi.Memory.handleSlice(memories);
        args.num_memories = memories.len;
    }

    fn writeAttributes(self: Device, args: anytype) ?*c.PJRT_Error {
        const owned = allocator.create(DeviceAttributes) catch {
            return PjrtError.internal("failed to allocate device attributes");
        };
        owned.* = DeviceAttributes.init(self.ptr);
        args.attributes = abi.NamedValue.ptr(owned.attrs[0..]);
        args.num_attributes = owned.attrs.len;
        args.device_attributes = DeviceAttributesHandle.handle(owned);
        args.attributes_deleter = DeviceAttributes.delete;
        return null;
    }
};

const Memory = struct {
    ptr: *runtime.Memory,

    const Api = struct {
        pub const Id = MemoryCallback(c.PJRT_Memory_Id_Args, .id).call;
        pub const Kind = MemoryText(c.PJRT_Memory_Kind_Args, "kind", "kind_size", .kind).call;
        pub const DebugString = MemoryText(c.PJRT_Memory_DebugString_Args, "debug_string", "debug_string_size", .debug_string).call;
        pub const ToString = MemoryText(c.PJRT_Memory_ToString_Args, "to_string", "to_string_size", .debug_string).call;
        pub const AddressableByDevices = MemoryCallback(c.PJRT_Memory_AddressableByDevices_Args, .addressable_devices).call;
    };

    fn at(raw: anytype) Memory {
        return .{ .ptr = abi.Memory.view(raw) };
    }

    fn kind(self: Memory) []const u8 {
        return @tagName(self.ptr.kind);
    }

    fn debugString(self: Memory) []const u8 {
        return self.ptr.debug_string;
    }

    fn id(self: Memory) i32 {
        return self.ptr.id;
    }

    fn writeAddressableDevices(self: Memory, args: anytype) void {
        args.devices = abi.Device.handleSlice(self.ptr.addressable_devices);
        args.num_devices = self.ptr.addressable_devices.len;
    }
};

const DeviceAttribute = enum {
    name,
    registry_id,
    recommended_working_set_size,
    has_unified_memory,
    default_memory_id,

    const fields = @typeInfo(DeviceAttribute).@"enum".fields;

    fn namedValue(comptime attr: DeviceAttribute, device: *const runtime.Device) abi.NamedValue {
        return switch (attr) {
            .name => abi.NamedValue.string("device_name", device.name),
            .registry_id => abi.NamedValue.int64("pjrtx_registry_id", state.Scalar.clampI64(device.registry_id)),
            .recommended_working_set_size => abi.NamedValue.int64("pjrtx_recommended_working_set_size", state.Scalar.clampI64(device.memory_bytes)),
            .has_unified_memory => abi.NamedValue.int64("pjrtx_has_unified_memory", if (device.has_unified_memory) 1 else 0),
            .default_memory_id => abi.NamedValue.int64("pjrtx_default_memory_id", device.default_memory_id),
        };
    }
};

const DeviceAttributes = struct {
    attrs: [DeviceAttribute.fields.len]abi.NamedValue,

    fn init(device: *const runtime.Device) DeviceAttributes {
        var attrs: [DeviceAttribute.fields.len]abi.NamedValue = undefined;
        inline for (DeviceAttribute.fields, 0..) |field, i| {
            attrs[i] = @as(DeviceAttribute, @enumFromInt(field.value)).namedValue(device);
        }
        return .{ .attrs = attrs };
    }

    fn delete(raw: ?*c.PJRT_Device_Attributes) callconv(.c) void {
        if (raw) |opaque_attrs| allocator.destroy(DeviceAttributesHandle.view(opaque_attrs));
    }
};

const DeviceAttributesHandle = abi.DeviceAttributes(DeviceAttributes);

fn set(comptime field: []const u8, raw: anytype, value: anytype) void {
    @field(raw[0], field) = value;
}

fn setText(comptime ptr_field: []const u8, comptime len_field: []const u8, raw: anytype, value: []const u8) void {
    abi.Args.writeBytes(ptr_field, len_field, &raw[0], value);
}

fn DeviceDescriptionScalar(
    comptime Args: type,
    comptime field: []const u8,
    comptime property: enum { id, process_index },
) type {
    return struct {
        fn call(raw: [*c]Args) callconv(.c) ?*c.PJRT_Error {
            const desc = DeviceDescription.at(raw[0].device_description);
            set(field, raw, switch (property) {
                .id => desc.id(),
                .process_index => desc.processIndex(),
            });
            return null;
        }
    };
}

fn DeviceDescriptionText(
    comptime Args: type,
    comptime ptr_field: []const u8,
    comptime len_field: []const u8,
    comptime property: enum { kind, debug_string },
) type {
    return struct {
        fn call(raw: [*c]Args) callconv(.c) ?*c.PJRT_Error {
            const desc = DeviceDescription.at(raw[0].device_description);
            setText(ptr_field, len_field, raw, switch (property) {
                .kind => desc.kind(),
                .debug_string => desc.debugString(),
            });
            return null;
        }
    };
}

fn MemoryText(
    comptime Args: type,
    comptime ptr_field: []const u8,
    comptime len_field: []const u8,
    comptime property: enum { kind, debug_string },
) type {
    return struct {
        fn call(raw: [*c]Args) callconv(.c) ?*c.PJRT_Error {
            const memory = Memory.at(raw[0].memory);
            setText(ptr_field, len_field, raw, switch (property) {
                .kind => memory.kind(),
                .debug_string => memory.debugString(),
            });
            return null;
        }
    };
}

const MemoryStats = struct {
    args: *allowzero c.PJRT_Device_MemoryStats_Args,
    stats: runtime.MemoryStats,

    fn write(raw: [*c]c.PJRT_Device_MemoryStats_Args, stats: runtime.MemoryStats) void {
        (MemoryStats{ .args = &raw[0], .stats = stats }).writeAll();
    }

    fn writeAll(self: MemoryStats) void {
        self.writeCore();
        if (self.stats.capacity_bytes != 0) self.writeCapacity();
    }

    fn writeCore(self: MemoryStats) void {
        self.args.bytes_in_use = state.Scalar.clampI64(self.stats.totalBytesInUse());
        self.args.peak_bytes_in_use = state.Scalar.clampI64(self.stats.peakTotalBytesInUse());
        self.args.peak_bytes_in_use_is_set = true;
        self.args.num_allocs = state.Scalar.clampI64(self.stats.live_allocs +| self.stats.executable_cache_resident_entries);
        self.args.num_allocs_is_set = true;
        self.args.largest_alloc_size = state.Scalar.clampI64(@max(self.stats.largest_alloc_size, self.stats.executable_cache_largest_resident_bytes));
        self.args.largest_alloc_size_is_set = true;
    }

    fn writeCapacity(self: MemoryStats) void {
        self.args.bytes_limit = state.Scalar.clampI64(self.stats.capacity_bytes);
        self.args.bytes_limit_is_set = true;
        self.args.largest_free_block_bytes = state.Scalar.clampI64(self.freeBytes());
        self.args.largest_free_block_bytes_is_set = true;
        self.args.bytes_reservable_limit = self.args.largest_free_block_bytes;
        self.args.bytes_reservable_limit_is_set = true;
    }

    fn freeBytes(self: MemoryStats) u64 {
        const live_bytes = self.stats.totalBytesInUse();
        return if (live_bytes >= self.stats.capacity_bytes) 0 else self.stats.capacity_bytes - live_bytes;
    }
};

const DeviceDescriptionOp = enum { attributes };

fn DeviceDescriptionCallback(comptime Args: type, comptime op: DeviceDescriptionOp) type {
    return struct {
        fn call(raw: [*c]Args) callconv(.c) ?*c.PJRT_Error {
            const args = &raw[0];
            switch (op) {
                .attributes => {
                    _ = DeviceDescription.at(args.device_description);
                    args.attributes = null;
                    args.num_attributes = 0;
                },
            }
            return null;
        }
    };
}

const DeviceOp = enum {
    description,
    is_addressable,
    local_hardware_id,
    addressable_memories,
    default_memory,
    memory_stats,
    attributes,
};

fn DeviceCallback(comptime Args: type, comptime op: DeviceOp) type {
    return struct {
        fn call(raw: [*c]Args) callconv(.c) ?*c.PJRT_Error {
            const args = &raw[0];
            const device = Device.at(args.device);
            switch (op) {
                .description => args.device_description = device.description(),
                .is_addressable => args.is_addressable = device.isAddressable(),
                .local_hardware_id => args.local_hardware_id = device.localHardwareId(),
                .addressable_memories => device.writeAddressableMemories(args),
                .default_memory => args.memory = abi.Memory.handle(device.defaultMemory()),
                .memory_stats => MemoryStats.write(args, device.defaultMemory().stats),
                .attributes => return device.writeAttributes(args),
            }
            return null;
        }
    };
}

const MemoryOp = enum { id, addressable_devices };

fn MemoryCallback(comptime Args: type, comptime op: MemoryOp) type {
    return struct {
        fn call(raw: [*c]Args) callconv(.c) ?*c.PJRT_Error {
            const args = &raw[0];
            const memory = Memory.at(args.memory);
            switch (op) {
                .id => args.id = memory.id(),
                .addressable_devices => memory.writeAddressableDevices(args),
            }
            return null;
        }
    };
}

pub const DeviceDescriptionApi = DeviceDescription.Api;
pub const DeviceApi = Device.Api;
pub const MemoryApi = Memory.Api;
