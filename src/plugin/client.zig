const std = @import("std");

const c = @import("c");
const runtime_mod = @import("src/runtime");

const abi = @import("pjrt_abi.zig");
const async_h2d = @import("async_h2d.zig");
const buffer_element = @import("buffer_element.zig");
const buffer_placement = @import("buffer_placement.zig");
const device_memory = @import("device_memory.zig");
const errors = @import("errors.zig");
const events = @import("events.zig");
const executable_mod = @import("executable.zig");
const handles = @import("pjrt_handles.zig");
const plugin = @import("plugin.zig");
const trace_mod = @import("trace.zig");

const Executable = executable_mod.Executable;
const LoadedExecutableHandle = handles.LoadedExecutable(Executable);
const BufferPlacement = buffer_placement.Placement;
const PjrtError = errors.Error;
const PjrtEvent = events.Event;

/// Borrowed PJRT client reference backed by a runtime client.
pub const Client = struct {
    ptr: *runtime_mod.Client,

    pub const Api = struct {
        pub const Create = ClientCreate.call;
        pub const Destroy = ClientDestroy.call;
        pub const PlatformName = ClientTextCallback(c.PJRT_Client_PlatformName_Args, "platform_name", "platform_name_size", .platform_name).call;
        pub const ProcessIndex = ClientProcessIndex.call;
        pub const PlatformVersion = ClientTextCallback(c.PJRT_Client_PlatformVersion_Args, "platform_version", "platform_version_size", .platform_version).call;
        pub const TopologyDescription = ClientTopologyDescription.call;
        pub const Devices = ClientDeviceListCallback(c.PJRT_Client_Devices_Args, "devices", "num_devices").call;
        pub const AddressableDevices = ClientDeviceListCallback(c.PJRT_Client_AddressableDevices_Args, "addressable_devices", "num_addressable_devices").call;
        pub const LookupDevice = ClientLookupDevice.call;
        pub const LookupAddressableDevice = ClientLookupAddressableDevice.call;
        pub const AddressableMemories = ClientAddressableMemories.call;
        pub const Compile = ClientCompile.call;
        pub const DefaultDeviceAssignment = ClientDefaultDeviceAssignment.call;
        pub const BufferFromHostBuffer = ClientBufferFromHostBuffer.call;
        pub const CreateUninitializedBuffer = ClientCreateUninitializedBuffer.call;
        pub const CreateBuffersForAsyncHostToDevice = ClientCreateBuffersForAsyncHostToDevice.call;
        pub const DmaMap = ClientDmaMap.call;
        pub const DmaUnmap = ClientDmaUnmap.call;
    };

    fn at(raw: anytype) Client {
        return .{ .ptr = handles.Client.ref(raw) };
    }

    fn platformName(self: Client) []const u8 {
        _ = self;
        return plugin.Platform.name;
    }

    fn platformVersion(self: Client) []const u8 {
        _ = self;
        return plugin.Platform.version;
    }

    fn processIndex(self: Client) c_int {
        _ = self;
        return 0;
    }

    fn topology(self: Client) *c.PJRT_TopologyDescription {
        return handles.TopologyDescription.handle(self.ptr);
    }

    fn devices(self: Client) []const *runtime_mod.Device {
        return self.ptr.device_handles;
    }

    fn memories(self: Client) []const *runtime_mod.Memory {
        return self.ptr.memory_handles;
    }

    fn lookupDevice(self: Client, id: i32) ?*const runtime_mod.Device {
        return self.ptr.lookupDevice(id);
    }

    fn lookupAddressableDevice(self: Client, local_hardware_id: i32) ?*const runtime_mod.Device {
        return self.ptr.lookupAddressableDeviceByLocalHardwareId(local_hardware_id);
    }
};

const ClientCreateRequest = struct {
    fn decode(args: c.PJRT_Client_Create_Args) !ClientCreateRequest {
        if (args.create_options != null) {
            for (0..args.num_options) |i| {
                const option = abi.NamedValue.borrow(args.create_options[i]);
                if (std.mem.eql(u8, option.name(), plugin.Options.backend)) {
                    const value = option.stringValue() orelse return error.InvalidBackend;
                    if (!std.mem.eql(u8, value, "metal_mlx")) return error.InvalidBackend;
                } else {
                    return error.InvalidBackend;
                }
            }
        }
        return .{};
    }
};

const HostBufferRequest = struct {
    placement: BufferPlacement,
    element_type: runtime_mod.BufferType,
    dims: []const i64,
    byte_size: usize,
    data: []const u8,

    fn decode(raw: *allowzero c.PJRT_Client_BufferFromHostBuffer_Args) DecodeError!HostBufferRequest {
        const client = handles.Client.ref(raw.client);
        const placement = BufferPlacement.forHostBuffer(client, raw.device, raw.memory) catch |err| return err;
        const dims = abi.Slice.constList(i64, raw.dims, raw.num_dims) orelse return error.NullDims;
        const byte_size = buffer_element.ElementType.denseByteSize(raw.type, dims);
        return .{
            .placement = placement,
            .element_type = buffer_element.ElementType.fromPjrt(raw.type),
            .dims = dims,
            .byte_size = byte_size,
            .data = abi.Slice.constBytes(raw.data, byte_size) orelse return error.NullBytes,
        };
    }

    fn fail(err: DecodeError) ?*c.PJRT_Error {
        return switch (err) {
            error.MemoryHasNoDevices => PjrtError.invalidArgument("memory is not addressable by any device"),
            error.DeviceNotInClient => PjrtError.invalidArgument("buffer device does not belong to client"),
            error.MemoryNotAddressable => PjrtError.invalidArgument("memory is not addressable by requested device"),
            error.NullDims => PjrtError.invalidArgument("buffer dimensions are null"),
            error.NullBytes => PjrtError.invalidArgument("source buffer data is null"),
        };
    }

    const DecodeError = BufferPlacement.Error || error{ NullDims, NullBytes };
};

const DeviceBufferRequest = struct {
    placement: BufferPlacement,
    element_type: runtime_mod.BufferType,
    dims: []const i64,
    byte_size: usize,

    fn decode(raw: *allowzero c.PJRT_Client_CreateUninitializedBuffer_Args) DecodeError!DeviceBufferRequest {
        const client = handles.Client.ref(raw.client);
        const placement = BufferPlacement.forDeviceBuffer(client, raw.device, raw.memory) catch |err| return err;
        const dims = abi.Slice.constList(i64, raw.shape_dims, raw.shape_num_dims) orelse return error.NullDims;
        const byte_size = buffer_element.ElementType.denseByteSize(raw.shape_element_type, dims);
        return .{
            .placement = placement,
            .element_type = buffer_element.ElementType.fromPjrt(raw.shape_element_type),
            .dims = dims,
            .byte_size = byte_size,
        };
    }

    fn fail(err: DecodeError) ?*c.PJRT_Error {
        return switch (err) {
            error.MemoryHasNoDevices => PjrtError.invalidArgument("memory is not addressable by any device"),
            error.DeviceNotInClient => PjrtError.invalidArgument("buffer device does not belong to client"),
            error.MemoryNotAddressable => PjrtError.invalidArgument("memory is not addressable by requested device"),
            error.NullDims => PjrtError.invalidArgument("buffer shape dimensions are null"),
        };
    }

    const DecodeError = BufferPlacement.Error || error{NullDims};
};

const BufferCreateFailure = struct {
    fn host(err: runtime_mod.BufferCreateError) ?*c.PJRT_Error {
        return switch (err) {
            error.InvalidArgument => PjrtError.invalidArgument("memory is not addressable by requested device"),
            error.UnsupportedElementType, error.UnsupportedRuntimeFeature => PjrtError.unimplemented("backend cannot import requested buffer type"),
            error.BufferAllocationFailed, error.OutOfMemory => PjrtError.resourceExhausted("backend failed to allocate buffer storage"),
            error.InvalidDeviceCount,
            error.InvalidProgram,
            error.ShapeMismatch,
            error.CommandSubmissionFailed,
            error.BufferCopyFailed,
            error.InvalidCustomCall,
            => PjrtError.internal("failed to create host buffer copy"),
        };
    }

    fn device(err: runtime_mod.BufferCreateError) ?*c.PJRT_Error {
        return switch (err) {
            error.InvalidArgument => PjrtError.invalidArgument("memory is not addressable by requested device"),
            error.UnsupportedElementType, error.UnsupportedRuntimeFeature => PjrtError.unimplemented("backend cannot allocate requested buffer type"),
            error.BufferAllocationFailed, error.OutOfMemory => PjrtError.resourceExhausted("backend failed to allocate buffer storage"),
            error.InvalidDeviceCount,
            error.InvalidProgram,
            error.ShapeMismatch,
            error.CommandSubmissionFailed,
            error.BufferCopyFailed,
            error.InvalidCustomCall,
            => PjrtError.internal("failed to create device buffer"),
        };
    }
};

const CompileCall = struct {
    raw: *allowzero c.PJRT_Client_Compile_Args,

    fn init(raw: *allowzero c.PJRT_Client_Compile_Args) CompileCall {
        return .{ .raw = raw };
    }

    fn client(self: CompileCall) *runtime_mod.Client {
        return handles.Client.ref(self.raw.client);
    }

    fn program(self: CompileCall) !runtime_mod.CompileProgram {
        const raw_program = self.raw.program orelse return error.InvalidProgram;
        const pjrt_program = raw_program[0];
        const format = abi.Slice.bytes(pjrt_program.format, pjrt_program.format_size) orelse return error.InvalidProgram;
        const code = abi.Slice.bytes(pjrt_program.code, pjrt_program.code_size) orelse return error.InvalidProgram;
        const options = abi.Slice.bytes(self.raw.compile_options, self.raw.compile_options_size) orelse return error.InvalidOptions;
        return .{
            .format = format,
            .code = code,
            .compile_options = options,
        };
    }

    fn fail(err: runtime_mod.CompileProgramError, diagnostics: []const u8) ?*c.PJRT_Error {
        const text = if (diagnostics.len != 0) diagnostics else defaultErrorMessage(err);
        return switch (err) {
            error.InvalidOptions,
            error.OptionsRequireMoreDevices,
            error.UnknownDevice,
            error.InvalidProgram,
            error.InvalidExecutablePlan,
            => PjrtError.invalidArgument(text),
            error.UnsupportedProgram,
            error.UnsupportedRuntimeFeature,
            => PjrtError.unimplemented(text),
            error.OutOfMemory => PjrtError.resourceExhausted(text),
            error.Internal => PjrtError.internal(text),
        };
    }

    fn defaultErrorMessage(err: runtime_mod.CompileProgramError) []const u8 {
        return switch (err) {
            error.InvalidOptions => "invalid PjRTx text compile options",
            error.OptionsRequireMoreDevices => "compile options require more devices than the client exposes",
            error.UnknownDevice => "compile options reference an unknown device id",
            error.UnsupportedProgram => "failed to ingest StableHLO/MLIR program",
            error.InvalidProgram => "failed to ingest StableHLO/MLIR program",
            error.InvalidExecutablePlan => "invalid executable plan",
            error.UnsupportedRuntimeFeature => "program is not fully lowered to the MLX backend executable; runtime execution is device-only",
            error.OutOfMemory => "compile exhausted memory",
            error.Internal => "failed to compile executable",
        };
    }
};

fn ClientTextCallback(
    comptime Args: type,
    comptime ptr_field: []const u8,
    comptime size_field: []const u8,
    comptime text: enum { platform_name, platform_version },
) type {
    return struct {
        fn call(raw: [*c]Args) callconv(.c) ?*c.PJRT_Error {
            const client = Client.at(raw[0].client);
            abi.Out.writeBytes(ptr_field, size_field, &raw[0], switch (text) {
                .platform_name => client.platformName(),
                .platform_version => client.platformVersion(),
            });
            return null;
        }
    };
}

fn ClientDeviceListCallback(comptime Args: type, comptime devices_field: []const u8, comptime count_field: []const u8) type {
    return struct {
        fn call(raw: [*c]Args) callconv(.c) ?*c.PJRT_Error {
            const devices = Client.at(raw[0].client).devices();
            @field(raw[0], devices_field) = handles.Device.handleSlice(devices);
            @field(raw[0], count_field) = devices.len;
            return null;
        }
    };
}

const ClientCreate = struct {
    fn call(args: [*c]c.PJRT_Client_Create_Args) callconv(.c) ?*c.PJRT_Error {
        _ = ClientCreateRequest.decode(args[0]) catch {
            return PjrtError.invalidArgument("invalid PjRTx client create option");
        };
        const client = runtime_mod.createClient(plugin.allocator(), .{
            .executable_cache_max_resident_bytes = trace_mod.Env.executableCacheMaxBytes(),
        }) catch {
            return PjrtError.internal("failed to create PjRTx client");
        };
        args[0].client = handles.Client.handle(client);
        return null;
    }
};

const ClientDestroy = struct {
    fn call(args: [*c]c.PJRT_Client_Destroy_Args) callconv(.c) ?*c.PJRT_Error {
        if (args[0].client) |client| {
            const runtime_client = handles.Client.ref(client);
            device_memory.Lifetime.releaseClientCaches(runtime_client);
            runtime_client.deinit();
        }
        return null;
    }
};

const ClientProcessIndex = struct {
    fn call(args: [*c]c.PJRT_Client_ProcessIndex_Args) callconv(.c) ?*c.PJRT_Error {
        args[0].process_index = Client.at(args[0].client).processIndex();
        return null;
    }
};

const ClientTopologyDescription = struct {
    fn call(args: [*c]c.PJRT_Client_TopologyDescription_Args) callconv(.c) ?*c.PJRT_Error {
        args[0].topology = Client.at(args[0].client).topology();
        return null;
    }
};

const ClientLookupDevice = struct {
    fn call(args: [*c]c.PJRT_Client_LookupDevice_Args) callconv(.c) ?*c.PJRT_Error {
        const device = Client.at(args[0].client).lookupDevice(args[0].id) orelse return PjrtError.notFound("device id not found");
        args[0].device = handles.Device.handle(@constCast(device));
        return null;
    }
};

const ClientLookupAddressableDevice = struct {
    fn call(args: [*c]c.PJRT_Client_LookupAddressableDevice_Args) callconv(.c) ?*c.PJRT_Error {
        const device = Client.at(args[0].client).lookupAddressableDevice(args[0].local_hardware_id) orelse return PjrtError.notFound("local hardware id not found");
        args[0].addressable_device = handles.Device.handle(@constCast(device));
        return null;
    }
};

const ClientAddressableMemories = struct {
    fn call(args: [*c]c.PJRT_Client_AddressableMemories_Args) callconv(.c) ?*c.PJRT_Error {
        const memories = Client.at(args[0].client).memories();
        args[0].addressable_memories = handles.Memory.handleSlice(memories);
        args[0].num_addressable_memories = memories.len;
        return null;
    }
};

const ClientCompile = struct {
    fn call(args: [*c]c.PJRT_Client_Compile_Args) callconv(.c) ?*c.PJRT_Error {
        const call_info = CompileCall.init(&args[0]);
        const client = call_info.client();

        var diagnostics = std.Io.Writer.Allocating.init(plugin.allocator());
        defer diagnostics.deinit();
        const program = call_info.program() catch |err| return switch (err) {
            error.InvalidOptions => PjrtError.invalidArgument("compile options pointer is null"),
            error.InvalidProgram => PjrtError.invalidArgument("compile program is null or malformed"),
        };
        var compiled = client.compileProgram(plugin.allocator(), program, &diagnostics.writer) catch |err| {
            const message = diagnostics.writer.buffered();
            return CompileCall.fail(err, message);
        };
        var compiled_owned = true;
        defer if (compiled_owned) compiled.deinit(plugin.allocator());

        const executable = Executable.create(client, &compiled) catch |err| {
            return switch (err) {
                error.OutOfMemory => PjrtError.resourceExhausted("failed to allocate executable metadata"),
            };
        };
        compiled_owned = false;
        args[0].executable = LoadedExecutableHandle.handle(executable);
        return null;
    }
};
const ClientDefaultDeviceAssignment = struct {
    fn call(args: [*c]c.PJRT_Client_DefaultDeviceAssignment_Args) callconv(.c) ?*c.PJRT_Error {
        const client = handles.Client.ref(args[0].client);
        const needed: usize = @intCast(args[0].num_replicas * args[0].num_partitions);
        if (needed > client.devices.len or needed > args[0].default_assignment_size) {
            return PjrtError.invalidArgument("invalid default device assignment request");
        }
        for (0..needed) |i| args[0].default_assignment[i] = @intCast(i);
        return null;
    }
};

const ClientBufferFromHostBuffer = struct {
    fn call(args: [*c]c.PJRT_Client_BufferFromHostBuffer_Args) callconv(.c) ?*c.PJRT_Error {
        const request = HostBufferRequest.decode(&args[0]) catch |err| return HostBufferRequest.fail(err);
        const placement = request.placement;
        _ = placement.client.trimExecutableCacheForAllocation(placement.memory, request.byte_size);
        const buffer = placement.client.createHostBufferFromBytes(plugin.allocator(), request.element_type, request.dims, placement.device, placement.memory, placement.shard_index, request.data) catch |err| {
            return BufferCreateFailure.host(err);
        };
        args[0].buffer = handles.Buffer.handle(buffer);
        args[0].done_with_host_buffer = PjrtEvent.ready();
        return null;
    }
};

const ClientCreateUninitializedBuffer = struct {
    fn call(args: [*c]c.PJRT_Client_CreateUninitializedBuffer_Args) callconv(.c) ?*c.PJRT_Error {
        const request = DeviceBufferRequest.decode(&args[0]) catch |err| return DeviceBufferRequest.fail(err);
        const placement = request.placement;
        _ = placement.client.trimExecutableCacheForAllocation(placement.memory, request.byte_size);
        const buffer = placement.client.createDeviceBuffer(plugin.allocator(), request.element_type, request.dims, placement.device, placement.memory, placement.shard_index) catch |err| {
            return BufferCreateFailure.device(err);
        };
        args[0].buffer = handles.Buffer.handle(buffer);
        return null;
    }
};

const ClientCreateBuffersForAsyncHostToDevice = struct {
    fn call(args: [*c]c.PJRT_Client_CreateBuffersForAsyncHostToDevice_Args) callconv(.c) ?*c.PJRT_Error {
        const client = handles.Client.ref(args[0].client);
        const memory = if (args[0].memory) |mem| handles.Memory.ref(mem) else client.devices[0].default_memory;
        if (args[0].shape_specs == null and args[0].num_shape_specs != 0) {
            return PjrtError.invalidArgument("shape specs are null");
        }
        const raw_shape_specs = abi.Slice.constList(c.PJRT_ShapeSpec, args[0].shape_specs, args[0].num_shape_specs) orelse return PjrtError.invalidArgument("shape specs are null");
        const manager = async_h2d.create(client, memory, raw_shape_specs) catch |err| {
            return switch (err) {
                error.InvalidArgument => PjrtError.invalidArgument("memory is not addressable by any device"),
                error.InvalidShapeSpec => PjrtError.invalidArgument("shape spec dimensions are null"),
                error.OutOfMemory => PjrtError.resourceExhausted("failed to allocate async shape specs"),
                error.UnsupportedRuntimeFeature, error.UnsupportedElementType => PjrtError.unimplemented("backend cannot allocate async transfer buffer"),
                error.InvalidDeviceCount,
                error.InvalidProgram,
                error.ShapeMismatch,
                error.BufferAllocationFailed,
                error.CommandSubmissionFailed,
                error.BufferCopyFailed,
                error.InvalidCustomCall,
                => PjrtError.internal("failed to create async host-to-device transfer manager"),
            };
        };
        args[0].transfer_manager = manager;
        return null;
    }
};

const ClientDmaMap = struct {
    fn call(_: [*c]c.PJRT_Client_DmaMap_Args) callconv(.c) ?*c.PJRT_Error {
        return PjrtError.unimplemented("PJRT DMA map is not implemented for the MLX async transfer path");
    }
};

const ClientDmaUnmap = struct {
    fn call(_: [*c]c.PJRT_Client_DmaUnmap_Args) callconv(.c) ?*c.PJRT_Error {
        return PjrtError.unimplemented("PJRT DMA unmap is not implemented for the MLX async transfer path");
    }
};
