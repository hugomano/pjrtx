const std = @import("std");

const runtime = @import("src/runtime");
const c = @import("c");
const abi = @import("pjrt_abi.zig");
const async_h2d = @import("async_h2d.zig");
const errors = @import("errors.zig");
const events = @import("events.zig");
const state = @import("state.zig");
const trace_mod = @import("trace.zig");
const types = @import("types.zig");

const allocator = state.allocator;
const backend_option = state.backend_option;
const platform_name = state.platform_name;
const platform_version = state.platform_version;
const Executable = state.Executable;
const LoadedExecutableHandle = abi.LoadedExecutable(Executable);
const AsyncHostToDeviceTransferManager = async_h2d.AsyncHostToDeviceTransferManager;
const ShapeSpec = async_h2d.ShapeSpec;
const PjrtError = errors.Error;
const PjrtEvent = events.Event;
const executableCacheMaxBytesFromEnv = trace_mod.Env.executableCacheMaxBytes;

const ClientCreateConfig = struct {
    backend_kind: runtime.BackendKind = .metal_mlx,
};

fn clientCreateConfigFromArgs(args: c.PJRT_Client_Create_Args) !ClientCreateConfig {
    var config: ClientCreateConfig = .{};
    if (args.create_options != null) {
        for (0..args.num_options) |i| {
            const option = abi.NamedValue.view(args.create_options[i]);
            if (std.mem.eql(u8, option.name(), backend_option)) {
                const value = option.stringValue() orelse return error.InvalidBackend;
                if (std.mem.eql(u8, value, "metal_mlx")) {
                    config.backend_kind = .metal_mlx;
                } else {
                    return error.InvalidBackend;
                }
            } else {
                return error.InvalidBackend;
            }
        }
    }
    return config;
}
const ClientView = struct {
    ptr: *runtime.Client,

    const Api = struct {
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

    fn at(raw: anytype) ClientView {
        return .{ .ptr = abi.Client.view(raw) };
    }

    fn platformName(self: ClientView) []const u8 {
        _ = self;
        return platform_name;
    }

    fn platformVersion(self: ClientView) []const u8 {
        _ = self;
        return platform_version;
    }

    fn processIndex(self: ClientView) c_int {
        _ = self;
        return 0;
    }

    fn topology(self: ClientView) *c.PJRT_TopologyDescription {
        return abi.TopologyDescription.handle(self.ptr);
    }

    fn devices(self: ClientView) []const *runtime.Device {
        return self.ptr.device_handles;
    }

    fn memories(self: ClientView) []const *runtime.Memory {
        return self.ptr.memory_handles;
    }

    fn lookupDevice(self: ClientView, id: i32) ?*const runtime.Device {
        return self.ptr.lookupDevice(id);
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
            const client = ClientView.at(raw[0].client);
            abi.Args.writeBytes(ptr_field, size_field, &raw[0], switch (text) {
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
            const devices = ClientView.at(raw[0].client).devices();
            @field(raw[0], devices_field) = abi.Device.handleSlice(devices);
            @field(raw[0], count_field) = devices.len;
            return null;
        }
    };
}

const BufferPlacement = struct {
    client: *runtime.Client,
    device: *runtime.Device,
    memory: *runtime.Memory,
    shard_index: usize,

    fn forHostBuffer(client: *runtime.Client, device_arg: ?*c.PJRT_Device, memory_arg: ?*c.PJRT_Memory) ?BufferPlacement {
        const device = if (device_arg) |dev| abi.Device.view(dev) else &client.devices[0];
        const memory = if (memory_arg) |mem| abi.Memory.view(mem) else device.default_memory;
        const shard_index = abi.Placement.deviceIndex(client, device) orelse return null;
        return .{
            .client = client,
            .device = device,
            .memory = memory,
            .shard_index = shard_index,
        };
    }

    fn forDeviceBuffer(client: *runtime.Client, device_arg: ?*c.PJRT_Device, memory_arg: ?*c.PJRT_Memory) PlacementError!BufferPlacement {
        const memory = if (memory_arg) |mem| abi.Memory.view(mem) else blk: {
            const device = if (device_arg) |dev| abi.Device.view(dev) else &client.devices[0];
            break :blk device.default_memory;
        };
        const device = if (device_arg) |dev| abi.Device.view(dev) else blk: {
            if (memory.addressable_devices.len == 0) return error.MemoryHasNoDevices;
            break :blk memory.addressable_devices[0];
        };
        const shard_index = abi.Placement.deviceIndex(client, device) orelse return error.DeviceNotInClient;
        if (!memory.isAddressableBy(device)) return error.MemoryNotAddressable;
        return .{
            .client = client,
            .device = device,
            .memory = memory,
            .shard_index = shard_index,
        };
    }

    const PlacementError = error{ MemoryHasNoDevices, DeviceNotInClient, MemoryNotAddressable };
};

const HostBufferRequest = struct {
    placement: BufferPlacement,
    element_type: runtime.BufferType,
    dims: []const i64,
    byte_size: usize,
    data: []const u8,

    fn decode(raw: *allowzero c.PJRT_Client_BufferFromHostBuffer_Args, out: *HostBufferRequest) ?*c.PJRT_Error {
        const client = abi.Client.view(raw.client);
        const placement = BufferPlacement.forHostBuffer(client, raw.device, raw.memory) orelse {
            return PjrtError.invalidArgument("buffer device does not belong to client");
        };
        const dims = raw.dims[0..raw.num_dims];
        const byte_size = types.BufferType.denseByteSize(raw.type, dims);
        out.* = .{
            .placement = placement,
            .element_type = types.BufferType.fromPjrt(raw.type),
            .dims = dims,
            .byte_size = byte_size,
            .data = abi.Slice.constBytes(raw.data, byte_size) orelse return PjrtError.invalidArgument("source buffer data is null"),
        };
        return null;
    }
};

const DeviceBufferRequest = struct {
    placement: BufferPlacement,
    element_type: runtime.BufferType,
    dims: []const i64,
    byte_size: usize,

    fn decode(raw: *allowzero c.PJRT_Client_CreateUninitializedBuffer_Args, out: *DeviceBufferRequest) ?*c.PJRT_Error {
        const client = abi.Client.view(raw.client);
        const placement = BufferPlacement.forDeviceBuffer(client, raw.device, raw.memory) catch |err| return switch (err) {
            error.MemoryHasNoDevices => PjrtError.invalidArgument("memory is not addressable by any device"),
            error.DeviceNotInClient => PjrtError.invalidArgument("buffer device does not belong to client"),
            error.MemoryNotAddressable => PjrtError.invalidArgument("memory is not addressable by requested device"),
        };
        const dims = raw.shape_dims[0..raw.shape_num_dims];
        const byte_size = types.BufferType.denseByteSize(raw.shape_element_type, dims);
        out.* = .{
            .placement = placement,
            .element_type = types.BufferType.fromPjrt(raw.shape_element_type),
            .dims = dims,
            .byte_size = byte_size,
        };
        return null;
    }
};

const CompileCall = struct {
    raw: *allowzero c.PJRT_Client_Compile_Args,

    fn init(raw: *allowzero c.PJRT_Client_Compile_Args) CompileCall {
        return .{ .raw = raw };
    }

    fn client(self: CompileCall) *runtime.Client {
        return abi.Client.view(self.raw.client);
    }

    fn program(self: CompileCall) runtime.CompileProgram {
        const pjrt_program = if (self.raw.program) |raw_program| raw_program[0] else std.mem.zeroes(c.PJRT_Program);
        return .{
            .format = if (pjrt_program.format != null) pjrt_program.format[0..pjrt_program.format_size] else "",
            .code = if (pjrt_program.code != null and pjrt_program.code_size != 0) pjrt_program.code[0..pjrt_program.code_size] else &.{},
            .compile_options = if (self.raw.compile_options != null and self.raw.compile_options_size != 0) self.raw.compile_options[0..self.raw.compile_options_size] else &.{},
        };
    }

    fn fail(err: runtime.CompileProgramError, diagnostics: []const u8) ?*c.PJRT_Error {
        const text = if (diagnostics.len != 0) diagnostics else defaultErrorMessage(err);
        return PjrtError.make(errorCode(err), text);
    }

    fn errorCode(err: runtime.CompileProgramError) c.PJRT_Error_Code {
        return @intCast(switch (err) {
            error.InvalidOptions, error.OptionsRequireMoreDevices, error.UnknownDevice, error.InvalidProgram, error.InvalidExecutablePlan => c.PJRT_Error_Code_INVALID_ARGUMENT,
            error.UnsupportedProgram, error.UnsupportedRuntimeFeature => c.PJRT_Error_Code_UNIMPLEMENTED,
            error.OutOfMemory => c.PJRT_Error_Code_RESOURCE_EXHAUSTED,
            error.Internal => c.PJRT_Error_Code_INTERNAL,
        });
    }

    fn defaultErrorMessage(err: runtime.CompileProgramError) []const u8 {
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

const ClientCreate = struct {
    fn call(args: [*c]c.PJRT_Client_Create_Args) callconv(.c) ?*c.PJRT_Error {
        const config = clientCreateConfigFromArgs(args[0]) catch {
            return PjrtError.invalidArgument("invalid PjRTx client create option");
        };
        const client = runtime.createClient(allocator, .{
            .backend_kind = config.backend_kind,
            .device_count = 1,
            .executable_cache_max_resident_bytes = executableCacheMaxBytesFromEnv(),
        }) catch {
            return PjrtError.internal("failed to create PjRTx client");
        };
        args[0].client = abi.Client.handle(client);
        return null;
    }
};

const ClientDestroy = struct {
    fn call(args: [*c]c.PJRT_Client_Destroy_Args) callconv(.c) ?*c.PJRT_Error {
        if (args[0].client) |client| abi.Client.view(client).deinit();
        return null;
    }
};

const ClientProcessIndex = struct {
    fn call(args: [*c]c.PJRT_Client_ProcessIndex_Args) callconv(.c) ?*c.PJRT_Error {
        args[0].process_index = ClientView.at(args[0].client).processIndex();
        return null;
    }
};

const ClientTopologyDescription = struct {
    fn call(args: [*c]c.PJRT_Client_TopologyDescription_Args) callconv(.c) ?*c.PJRT_Error {
        args[0].topology = ClientView.at(args[0].client).topology();
        return null;
    }
};

const ClientLookupDevice = struct {
    fn call(args: [*c]c.PJRT_Client_LookupDevice_Args) callconv(.c) ?*c.PJRT_Error {
        const device = ClientView.at(args[0].client).lookupDevice(args[0].id) orelse return PjrtError.notFound("device id not found");
        args[0].device = abi.Device.handle(@constCast(device));
        return null;
    }
};

const ClientLookupAddressableDevice = struct {
    fn call(args: [*c]c.PJRT_Client_LookupAddressableDevice_Args) callconv(.c) ?*c.PJRT_Error {
        const device = ClientView.at(args[0].client).lookupDevice(args[0].local_hardware_id) orelse return PjrtError.notFound("local hardware id not found");
        args[0].addressable_device = abi.Device.handle(@constCast(device));
        return null;
    }
};

const ClientAddressableMemories = struct {
    fn call(args: [*c]c.PJRT_Client_AddressableMemories_Args) callconv(.c) ?*c.PJRT_Error {
        const memories = ClientView.at(args[0].client).memories();
        args[0].addressable_memories = abi.Memory.handleSlice(memories);
        args[0].num_addressable_memories = memories.len;
        return null;
    }
};

const ClientCompile = struct {
    fn call(args: [*c]c.PJRT_Client_Compile_Args) callconv(.c) ?*c.PJRT_Error {
        const call_info = CompileCall.init(&args[0]);
        const client = call_info.client();

        var diagnostics = std.Io.Writer.Allocating.init(allocator);
        defer diagnostics.deinit();
        var compiled = client.compileProgram(allocator, call_info.program(), &diagnostics.writer) catch |err| {
            const message = diagnostics.writer.buffered();
            return CompileCall.fail(err, message);
        };
        errdefer compiled.deinit(allocator);

        const plan = compiled.plan;

        const logical_ids = allocator.alloc(c.PJRT_LogicalDeviceIds, plan.options.numDevices()) catch return PjrtError.make(c.PJRT_Error_Code_INTERNAL, "failed to allocate executable logical device ids");
        errdefer allocator.free(logical_ids);
        for (logical_ids, 0..) |*id, index| {
            id.* = .{
                .replica = @intCast(index / @as(usize, @intCast(plan.options.num_partitions))),
                .partition = @intCast(index % @as(usize, @intCast(plan.options.num_partitions))),
            };
        }

        const parameter_memory_kinds = allocator.alloc([*c]const u8, plan.parameter_shardings.len) catch return PjrtError.make(c.PJRT_Error_Code_INTERNAL, "failed to allocate parameter memory kind table");
        errdefer allocator.free(parameter_memory_kinds);
        const parameter_memory_kind_sizes = allocator.alloc(usize, plan.parameter_shardings.len) catch return PjrtError.make(c.PJRT_Error_Code_INTERNAL, "failed to allocate parameter memory kind sizes");
        errdefer allocator.free(parameter_memory_kind_sizes);
        types.MemoryKind.fillDefault(parameter_memory_kinds, parameter_memory_kind_sizes);

        const output_memory_kinds = allocator.alloc([*c]const u8, plan.output_shardings.len) catch return PjrtError.make(c.PJRT_Error_Code_INTERNAL, "failed to allocate output memory kind table");
        errdefer allocator.free(output_memory_kinds);
        const output_memory_kind_sizes = allocator.alloc(usize, plan.output_shardings.len) catch return PjrtError.make(c.PJRT_Error_Code_INTERNAL, "failed to allocate output memory kind sizes");
        errdefer allocator.free(output_memory_kind_sizes);
        types.MemoryKind.fillDefault(output_memory_kinds, output_memory_kind_sizes);

        const executable = allocator.create(Executable) catch return PjrtError.make(c.PJRT_Error_Code_INTERNAL, "failed to allocate executable");
        executable.* = .{
            .client = client,
            .plan = compiled.plan,
            .graph = compiled.graph,
            .logical_ids = logical_ids,
            .optimized_program = compiled.optimized_program,
            .parameter_memory_kinds = parameter_memory_kinds,
            .parameter_memory_kind_sizes = parameter_memory_kind_sizes,
            .output_memory_kinds = output_memory_kinds,
            .output_memory_kind_sizes = output_memory_kind_sizes,
            .fingerprint = compiled.fingerprint,
        };
        args[0].executable = LoadedExecutableHandle.handle(executable);
        return null;
    }
};
const ClientDefaultDeviceAssignment = struct {
    fn call(args: [*c]c.PJRT_Client_DefaultDeviceAssignment_Args) callconv(.c) ?*c.PJRT_Error {
        const client = abi.Client.view(args[0].client);
        const needed: usize = @intCast(args[0].num_replicas * args[0].num_partitions);
        if (needed > client.devices.len or needed > args[0].default_assignment_size) {
            return PjrtError.make(c.PJRT_Error_Code_INVALID_ARGUMENT, "invalid default device assignment request");
        }
        for (0..needed) |i| args[0].default_assignment[i] = @intCast(i);
        return null;
    }
};

const ClientBufferFromHostBuffer = struct {
    fn call(args: [*c]c.PJRT_Client_BufferFromHostBuffer_Args) callconv(.c) ?*c.PJRT_Error {
        var request: HostBufferRequest = undefined;
        if (HostBufferRequest.decode(&args[0], &request)) |err| return err;
        const placement = request.placement;
        _ = placement.client.trimExecutableCacheForAllocation(placement.memory, request.byte_size);
        const buffer = placement.client.createHostBufferFromBytes(allocator, request.element_type, request.dims, placement.device, placement.memory, placement.shard_index, request.data) catch |err| {
            return switch (err) {
                error.InvalidArgument => PjrtError.make(c.PJRT_Error_Code_INVALID_ARGUMENT, "memory is not addressable by requested device"),
                else => PjrtError.make(c.PJRT_Error_Code_INTERNAL, "failed to create host buffer copy"),
            };
        };
        args[0].buffer = abi.Buffer.handle(buffer);
        args[0].done_with_host_buffer = PjrtEvent.pending();
        PjrtEvent.setReady(args[0].done_with_host_buffer);
        return null;
    }
};

const ClientCreateUninitializedBuffer = struct {
    fn call(args: [*c]c.PJRT_Client_CreateUninitializedBuffer_Args) callconv(.c) ?*c.PJRT_Error {
        var request: DeviceBufferRequest = undefined;
        if (DeviceBufferRequest.decode(&args[0], &request)) |err| return err;
        const placement = request.placement;
        _ = placement.client.trimExecutableCacheForAllocation(placement.memory, request.byte_size);
        const buffer = placement.client.createDeviceBuffer(allocator, request.element_type, request.dims, placement.device, placement.memory, placement.shard_index) catch |err| {
            return switch (err) {
                error.InvalidArgument => PjrtError.make(c.PJRT_Error_Code_INVALID_ARGUMENT, "memory is not addressable by requested device"),
                error.UnsupportedElementType, error.UnsupportedRuntimeFeature => PjrtError.make(c.PJRT_Error_Code_UNIMPLEMENTED, "backend cannot allocate requested buffer type"),
                else => PjrtError.make(c.PJRT_Error_Code_INTERNAL, "failed to create device buffer"),
            };
        };
        args[0].buffer = abi.Buffer.handle(buffer);
        return null;
    }
};

const ClientCreateBuffersForAsyncHostToDevice = struct {
    fn call(args: [*c]c.PJRT_Client_CreateBuffersForAsyncHostToDevice_Args) callconv(.c) ?*c.PJRT_Error {
        const client = abi.Client.view(args[0].client);
        const memory = if (args[0].memory) |mem| abi.Memory.view(mem) else client.devices[0].default_memory;
        if (args[0].shape_specs == null and args[0].num_shape_specs != 0) {
            return PjrtError.invalidArgument("shape specs are null");
        }
        const raw_shape_specs = args[0].shape_specs[0..args[0].num_shape_specs];
        const shape_specs = allocator.alloc(ShapeSpec, raw_shape_specs.len) catch {
            return PjrtError.internal("failed to allocate async shape specs");
        };
        defer allocator.free(shape_specs);
        for (raw_shape_specs, shape_specs) |raw, *shape_spec| shape_spec.* = ShapeSpec.fromPjrt(raw);
        const manager = AsyncHostToDeviceTransferManager.create(client, memory, shape_specs) catch |err| {
            return switch (err) {
                error.InvalidArgument => PjrtError.invalidArgument("memory is not addressable by any device"),
                error.UnsupportedRuntimeFeature, error.UnsupportedElementType => PjrtError.unimplemented("backend cannot allocate async transfer buffer"),
                else => PjrtError.internal("failed to create async host-to-device transfer manager"),
            };
        };
        args[0].transfer_manager = AsyncHostToDeviceTransferManager.handle(manager);
        return null;
    }
};

const ClientDmaMap = struct {
    fn call(_: [*c]c.PJRT_Client_DmaMap_Args) callconv(.c) ?*c.PJRT_Error {
        return null;
    }
};

const ClientDmaUnmap = struct {
    fn call(_: [*c]c.PJRT_Client_DmaUnmap_Args) callconv(.c) ?*c.PJRT_Error {
        return null;
    }
};

pub const Api = ClientView.Api;
