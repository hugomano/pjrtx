const std = @import("std");

const backend_api = @import("src/backend");
const backend_registry = @import("src/backend/registry");
const compiler = @import("src/compiler");
const runtime = @import("src/runtime");
const c = @import("c");

const allocator = std.heap.c_allocator;

const platform_name = "pjrtx_metal";
const platform_version = "PjRTx bootstrap Metal/MLX skeleton";
const device_kind = "Metal";
const plugin_name = "PjRTx";
const stablehlo_version = [_]i64{ 1, 0, 0 };
const backend_option = "pjrtx_backend";
const default_memory_kind = "device";

const MachTimebaseInfo = extern struct {
    numer: u32,
    denom: u32,
};

extern "c" fn mach_absolute_time() u64;
extern "c" fn mach_timebase_info(info: *MachTimebaseInfo) c_int;

const PjrtxError = struct {
    base: c.PJRT_Error,
    code: c.PJRT_Error_Code,
    message: []u8,
};

const Event = struct {
    ready: bool = true,
    err: ?*PjrtxError = null,
};

const DeviceAttributes = struct {
    attrs: [5]c.PJRT_NamedValue,
};

const SerializedTopology = struct {
    bytes: []u8,
};

const Executable = struct {
    client: *runtime.Client,
    plan: *compiler.ExecutablePlan,
    graph: runtime.ExecutableGraph,
    logical_ids: []c.PJRT_LogicalDeviceIds,
    optimized_program: []u8,
    parameter_memory_kinds: [][*c]const u8,
    parameter_memory_kind_sizes: []usize,
    output_memory_kinds: [][*c]const u8,
    output_memory_kind_sizes: []usize,
    fingerprint: []u8,
    name: []const u8 = "pjrtx_executable",
    deleted: bool = false,

    fn deinit(self: *Executable) void {
        self.graph.deinit();
        allocator.free(self.fingerprint);
        allocator.free(self.output_memory_kind_sizes);
        allocator.free(self.output_memory_kinds);
        allocator.free(self.parameter_memory_kind_sizes);
        allocator.free(self.parameter_memory_kinds);
        allocator.free(self.optimized_program);
        allocator.free(self.logical_ids);
        self.plan.deinit();
        allocator.destroy(self.plan);
        allocator.destroy(self);
    }
};

fn fillMemoryKindArrays(kinds: [][*c]const u8, sizes: []usize) void {
    for (kinds, sizes) |*kind, *size| {
        kind.* = default_memory_kind.ptr;
        size.* = default_memory_kind.len;
    }
}

var api_storage: c.PJRT_Api = undefined;
var api_ready = false;
var attrs_ready = false;
var attrs: [5]c.PJRT_NamedValue = undefined;

fn cstrLen(ptr: [*c]const u8) usize {
    return std.mem.len(ptr);
}

fn makeError(code: c.PJRT_Error_Code, message: []const u8) ?*c.PJRT_Error {
    const err = allocator.create(PjrtxError) catch return null;
    err.* = .{
        .base = .{ .vtable = null },
        .code = code,
        .message = allocator.dupe(u8, message) catch {
            allocator.destroy(err);
            return null;
        },
    };
    return @ptrCast(err);
}

fn unimplemented(message: []const u8) ?*c.PJRT_Error {
    return makeError(c.PJRT_Error_Code_UNIMPLEMENTED, message);
}

fn elementSize(t: c.PJRT_Buffer_Type) usize {
    return switch (t) {
        c.PJRT_Buffer_Type_PRED, c.PJRT_Buffer_Type_S8, c.PJRT_Buffer_Type_U8 => 1,
        c.PJRT_Buffer_Type_S16, c.PJRT_Buffer_Type_U16, c.PJRT_Buffer_Type_F16, c.PJRT_Buffer_Type_BF16 => 2,
        c.PJRT_Buffer_Type_S32, c.PJRT_Buffer_Type_U32, c.PJRT_Buffer_Type_F32 => 4,
        c.PJRT_Buffer_Type_S64, c.PJRT_Buffer_Type_U64, c.PJRT_Buffer_Type_F64 => 8,
        else => 0,
    };
}

fn runtimeTypeFromPjrt(t: c.PJRT_Buffer_Type) runtime.BufferType {
    return switch (t) {
        c.PJRT_Buffer_Type_PRED => .pred,
        c.PJRT_Buffer_Type_S8 => .s8,
        c.PJRT_Buffer_Type_S16 => .s16,
        c.PJRT_Buffer_Type_S32 => .s32,
        c.PJRT_Buffer_Type_S64 => .s64,
        c.PJRT_Buffer_Type_U8 => .u8,
        c.PJRT_Buffer_Type_U16 => .u16,
        c.PJRT_Buffer_Type_U32 => .u32,
        c.PJRT_Buffer_Type_U64 => .u64,
        c.PJRT_Buffer_Type_F16 => .f16,
        c.PJRT_Buffer_Type_F32 => .f32,
        c.PJRT_Buffer_Type_F64 => .f64,
        c.PJRT_Buffer_Type_BF16 => .bf16,
        else => .invalid,
    };
}

fn pjrtTypeFromRuntime(t: runtime.BufferType) c.PJRT_Buffer_Type {
    return switch (t) {
        .invalid => c.PJRT_Buffer_Type_INVALID,
        .pred => c.PJRT_Buffer_Type_PRED,
        .s8 => c.PJRT_Buffer_Type_S8,
        .s16 => c.PJRT_Buffer_Type_S16,
        .s32 => c.PJRT_Buffer_Type_S32,
        .s64 => c.PJRT_Buffer_Type_S64,
        .u8 => c.PJRT_Buffer_Type_U8,
        .u16 => c.PJRT_Buffer_Type_U16,
        .u32 => c.PJRT_Buffer_Type_U32,
        .u64 => c.PJRT_Buffer_Type_U64,
        .f16 => c.PJRT_Buffer_Type_F16,
        .f32 => c.PJRT_Buffer_Type_F32,
        .f64 => c.PJRT_Buffer_Type_F64,
        .bf16 => c.PJRT_Buffer_Type_BF16,
    };
}

fn denseByteSize(t: c.PJRT_Buffer_Type, dims: []const i64) usize {
    var elems: usize = 1;
    for (dims) |dim| elems *= @intCast(dim);
    return elems * elementSize(t);
}

fn clientFromC(client: ?*c.PJRT_Client) *runtime.Client {
    return @ptrCast(@alignCast(client.?));
}

fn deviceFromC(device: ?*c.PJRT_Device) *runtime.Device {
    return @ptrCast(@alignCast(device.?));
}

fn memoryFromC(memory: ?*c.PJRT_Memory) *runtime.Memory {
    return @ptrCast(@alignCast(memory.?));
}

fn bufferFromC(buffer: ?*c.PJRT_Buffer) *runtime.Buffer {
    return @ptrCast(@alignCast(buffer.?));
}

fn executableFromC(executable: ?*c.PJRT_LoadedExecutable) *Executable {
    return @ptrCast(@alignCast(executable.?));
}

fn topologyFromC(topology: ?*const c.PJRT_TopologyDescription) *runtime.Client {
    return @ptrCast(@alignCast(@constCast(topology.?)));
}

fn initAttrs() void {
    if (attrs_ready) return;
    attrs = std.mem.zeroes(@TypeOf(attrs));

    attrs[0].struct_size = c.PJRT_NamedValue_STRUCT_SIZE;
    attrs[0].name = "plugin_name";
    attrs[0].name_size = "plugin_name".len;
    attrs[0].type = c.PJRT_NamedValue_kString;
    attrs[0].unnamed_0.string_value = plugin_name;
    attrs[0].value_size = plugin_name.len;

    attrs[1].struct_size = c.PJRT_NamedValue_STRUCT_SIZE;
    attrs[1].name = "xla_version";
    attrs[1].name_size = "xla_version".len;
    attrs[1].type = c.PJRT_NamedValue_kString;
    attrs[1].unnamed_0.string_value = "local";
    attrs[1].value_size = "local".len;

    attrs[2].struct_size = c.PJRT_NamedValue_STRUCT_SIZE;
    attrs[2].name = "stablehlo_current_version";
    attrs[2].name_size = "stablehlo_current_version".len;
    attrs[2].type = c.PJRT_NamedValue_kInt64List;
    attrs[2].unnamed_0.int64_array_value = &stablehlo_version;
    attrs[2].value_size = stablehlo_version.len;

    attrs[3].struct_size = c.PJRT_NamedValue_STRUCT_SIZE;
    attrs[3].name = "stablehlo_minimum_version";
    attrs[3].name_size = "stablehlo_minimum_version".len;
    attrs[3].type = c.PJRT_NamedValue_kInt64List;
    attrs[3].unnamed_0.int64_array_value = &stablehlo_version;
    attrs[3].value_size = stablehlo_version.len;

    attrs[4].struct_size = c.PJRT_NamedValue_STRUCT_SIZE;
    attrs[4].name = "pjrtx_default_backend";
    attrs[4].name_size = "pjrtx_default_backend".len;
    attrs[4].type = c.PJRT_NamedValue_kString;
    attrs[4].unnamed_0.string_value = "metal_mlx";
    attrs[4].value_size = "metal_mlx".len;

    attrs_ready = true;
}

const ClientCreateConfig = struct {
    backend_kind: runtime.BackendKind = .metal_mlx,
};

fn traceEnabled() bool {
    const value = std.c.getenv("PJRTX_TRACE") orelse return false;
    const text = std.mem.span(value);
    return text.len != 0 and !std.mem.eql(u8, text, "0") and !std.ascii.eqlIgnoreCase(text, "false");
}

fn nowNs() u64 {
    var info: MachTimebaseInfo = undefined;
    if (mach_timebase_info(&info) != 0 or info.denom == 0) return mach_absolute_time();
    const ticks: u128 = mach_absolute_time();
    return @intCast((ticks * info.numer) / info.denom);
}

fn elapsedUs(start_ns: u64) u64 {
    return @intCast((nowNs() -| start_ns) / 1000);
}

fn trace(comptime fmt: []const u8, args: anytype) void {
    if (!traceEnabled()) return;
    std.debug.print("pjrtx_trace " ++ fmt ++ "\n", args);
}

fn clientCreateConfigFromArgs(args: c.PJRT_Client_Create_Args) !ClientCreateConfig {
    var config: ClientCreateConfig = .{};
    if (args.create_options != null) {
        for (0..args.num_options) |i| {
            const option = args.create_options[i];
            const name = option.name[0..option.name_size];
            if (std.mem.eql(u8, name, backend_option)) {
                if (option.type != c.PJRT_NamedValue_kString) return error.InvalidBackend;
                const value = option.unnamed_0.string_value[0..option.value_size];
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

fn isPjrtxTextCompileOptions(text: []const u8) bool {
    if (text.len == 0) return false;
    for (text) |byte| {
        if (!(std.ascii.isPrint(byte) or std.ascii.isWhitespace(byte))) return false;
    }
    return std.mem.indexOf(u8, text, "replicas=") != null or
        std.mem.indexOf(u8, text, "partitions=") != null or
        std.mem.indexOf(u8, text, "assignment=") != null or
        std.mem.indexOf(u8, text, "use_shardy=") != null;
}

fn pjrtErrorDestroy(args: [*c]c.PJRT_Error_Destroy_Args) callconv(.c) void {
    if (args[0].@"error") |base| {
        const err: *PjrtxError = @ptrCast(@alignCast(base));
        allocator.free(err.message);
        allocator.destroy(err);
    }
}

fn pjrtErrorMessage(args: [*c]c.PJRT_Error_Message_Args) callconv(.c) void {
    const err: *const PjrtxError = @ptrCast(@alignCast(args[0].@"error".?));
    args[0].message = err.message.ptr;
    args[0].message_size = err.message.len;
}

fn pjrtErrorGetCode(args: [*c]c.PJRT_Error_GetCode_Args) callconv(.c) ?*c.PJRT_Error {
    const err: *const PjrtxError = @ptrCast(@alignCast(args[0].@"error".?));
    args[0].code = err.code;
    return null;
}

fn pjrtErrorForEachPayload(_: [*c]c.PJRT_Error_ForEachPayload_Args) callconv(.c) ?*c.PJRT_Error {
    return null;
}

fn pjrtPluginInitialize(_: [*c]c.PJRT_Plugin_Initialize_Args) callconv(.c) ?*c.PJRT_Error {
    return null;
}

fn pjrtPluginAttributes(args: [*c]c.PJRT_Plugin_Attributes_Args) callconv(.c) ?*c.PJRT_Error {
    initAttrs();
    args[0].attributes = &attrs;
    args[0].num_attributes = attrs.len;
    return null;
}

fn eventCreateReady() ?*c.PJRT_Event {
    const event = allocator.create(Event) catch return null;
    event.* = .{};
    return @ptrCast(event);
}

fn pjrtEventDestroy(args: [*c]c.PJRT_Event_Destroy_Args) callconv(.c) ?*c.PJRT_Error {
    if (args[0].event) |event| allocator.destroy(@as(*Event, @ptrCast(@alignCast(event))));
    return null;
}

fn pjrtEventIsReady(args: [*c]c.PJRT_Event_IsReady_Args) callconv(.c) ?*c.PJRT_Error {
    const event: *Event = @ptrCast(@alignCast(args[0].event.?));
    args[0].is_ready = event.ready;
    return null;
}

fn pjrtEventError(args: [*c]c.PJRT_Event_Error_Args) callconv(.c) ?*c.PJRT_Error {
    const event: *Event = @ptrCast(@alignCast(args[0].event.?));
    if (event.err) |err| return @ptrCast(err);
    return null;
}

fn pjrtEventAwait(args: [*c]c.PJRT_Event_Await_Args) callconv(.c) ?*c.PJRT_Error {
    const event: *Event = @ptrCast(@alignCast(args[0].event.?));
    event.ready = true;
    if (event.err) |err| return @ptrCast(err);
    return null;
}

fn pjrtEventOnReady(args: [*c]c.PJRT_Event_OnReady_Args) callconv(.c) ?*c.PJRT_Error {
    if (args[0].callback) |callback| callback(null, args[0].user_arg);
    return null;
}

fn pjrtClientCreate(args: [*c]c.PJRT_Client_Create_Args) callconv(.c) ?*c.PJRT_Error {
    const config = clientCreateConfigFromArgs(args[0]) catch {
        return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "invalid PjRTx client create option");
    };
    const client = switch (config.backend_kind) {
        .metal_mlx => runtime.Client.init(allocator, backend_registry.create(.metal_mlx), 1),
    } catch {
        return makeError(c.PJRT_Error_Code_INTERNAL, "failed to create PjRTx client");
    };
    args[0].client = @ptrCast(client);
    return null;
}

fn pjrtClientDestroy(args: [*c]c.PJRT_Client_Destroy_Args) callconv(.c) ?*c.PJRT_Error {
    if (args[0].client) |client| clientFromC(client).deinit();
    return null;
}

fn pjrtClientPlatformName(args: [*c]c.PJRT_Client_PlatformName_Args) callconv(.c) ?*c.PJRT_Error {
    _ = clientFromC(args[0].client);
    args[0].platform_name = platform_name;
    args[0].platform_name_size = platform_name.len;
    return null;
}

fn pjrtClientProcessIndex(args: [*c]c.PJRT_Client_ProcessIndex_Args) callconv(.c) ?*c.PJRT_Error {
    _ = clientFromC(args[0].client);
    args[0].process_index = 0;
    return null;
}

fn pjrtClientPlatformVersion(args: [*c]c.PJRT_Client_PlatformVersion_Args) callconv(.c) ?*c.PJRT_Error {
    _ = clientFromC(args[0].client);
    args[0].platform_version = platform_version;
    args[0].platform_version_size = platform_version.len;
    return null;
}

fn pjrtClientTopologyDescription(args: [*c]c.PJRT_Client_TopologyDescription_Args) callconv(.c) ?*c.PJRT_Error {
    args[0].topology = @ptrCast(clientFromC(args[0].client));
    return null;
}

fn pjrtClientDevices(args: [*c]c.PJRT_Client_Devices_Args) callconv(.c) ?*c.PJRT_Error {
    const client = clientFromC(args[0].client);
    args[0].devices = @ptrCast(client.device_handles.ptr);
    args[0].num_devices = client.device_handles.len;
    return null;
}

fn pjrtClientAddressableDevices(args: [*c]c.PJRT_Client_AddressableDevices_Args) callconv(.c) ?*c.PJRT_Error {
    const client = clientFromC(args[0].client);
    args[0].addressable_devices = @ptrCast(client.device_handles.ptr);
    args[0].num_addressable_devices = client.device_handles.len;
    return null;
}

fn pjrtClientLookupDevice(args: [*c]c.PJRT_Client_LookupDevice_Args) callconv(.c) ?*c.PJRT_Error {
    const client = clientFromC(args[0].client);
    const device = client.lookupDevice(args[0].id) orelse return makeError(c.PJRT_Error_Code_NOT_FOUND, "device id not found");
    args[0].device = @ptrCast(@constCast(device));
    return null;
}

fn pjrtClientLookupAddressableDevice(args: [*c]c.PJRT_Client_LookupAddressableDevice_Args) callconv(.c) ?*c.PJRT_Error {
    const client = clientFromC(args[0].client);
    const device = client.lookupDevice(args[0].local_hardware_id) orelse return makeError(c.PJRT_Error_Code_NOT_FOUND, "local hardware id not found");
    args[0].addressable_device = @ptrCast(@constCast(device));
    return null;
}

fn pjrtClientAddressableMemories(args: [*c]c.PJRT_Client_AddressableMemories_Args) callconv(.c) ?*c.PJRT_Error {
    const client = clientFromC(args[0].client);
    args[0].addressable_memories = @ptrCast(client.memory_handles.ptr);
    args[0].num_addressable_memories = client.memory_handles.len;
    return null;
}

fn pjrtClientCompile(args: [*c]c.PJRT_Client_Compile_Args) callconv(.c) ?*c.PJRT_Error {
    const trace_start_ns = nowNs();
    const client = clientFromC(args[0].client);
    var options: compiler.CompileOptions = .{
        .num_partitions = @intCast(client.devices.len),
    };
    var parsed_options = false;
    if (args[0].compile_options != null and args[0].compile_options_size != 0) {
        const text = args[0].compile_options[0..args[0].compile_options_size];
        if (isPjrtxTextCompileOptions(text)) {
            var options_reader: std.Io.Reader = .fixed(text);
            options = compiler.parseTextCompileOptionsFromReader(allocator, &options_reader) catch {
                return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "invalid PjRTx text compile options");
            };
            parsed_options = true;
        }
    }
    defer if (parsed_options) allocator.free(options.device_assignment);

    if (options.numDevices() > client.devices.len) {
        return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "compile options require more devices than the client exposes");
    }
    for (options.device_assignment) |device_id| {
        if (client.lookupDevice(device_id) == null) {
            return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "compile options reference an unknown device id");
        }
    }

    var analysis: ?compiler.ModuleAnalysis = null;
    defer if (analysis) |*owned_analysis| owned_analysis.deinit();
    if (args[0].program != null and args[0].program[0].code != null and args[0].program[0].code_size != 0) {
        const program = args[0].program[0];
        const module_bytes = program.code[0..program.code_size];
        const format_text = if (program.format != null) program.format[0..program.format_size] else "";
        var module_reader: std.Io.Reader = .fixed(module_bytes);
        var diagnostics = std.Io.Writer.Allocating.init(allocator);
        defer diagnostics.deinit();
        analysis = compiler.analyzeProgramFromReader(allocator, format_text, &module_reader, &diagnostics.writer) catch |err| {
            const message = diagnostics.writer.buffered();
            const fallback = "failed to ingest StableHLO/MLIR program";
            const text = if (message.len == 0) fallback else message;
            const code: c.PJRT_Error_Code = @intCast(switch (err) {
                error.UnsupportedOp, error.UnsupportedSharding, error.UnsupportedProgramEncoding, error.GspmdNotEnabled, error.InvalidManualComputation => c.PJRT_Error_Code_UNIMPLEMENTED,
                error.UnsupportedProgramFormat, error.InvalidStablehloModule => c.PJRT_Error_Code_INVALID_ARGUMENT,
                else => c.PJRT_Error_Code_INTERNAL,
            });
            return makeError(code, text);
        };
    }

    var plan = if (analysis) |owned_analysis|
        compiler.makeExecutablePlan(allocator, options, owned_analysis) catch return makeError(c.PJRT_Error_Code_INTERNAL, "failed to allocate executable plan")
    else
        compiler.makeReplicatedPlan(allocator, options, 1, 1) catch return makeError(c.PJRT_Error_Code_INTERNAL, "failed to allocate executable plan");
    var plan_moved = false;
    errdefer if (!plan_moved) plan.deinit();

    var plan_diagnostics = std.Io.Writer.Allocating.init(allocator);
    defer plan_diagnostics.deinit();
    compiler.verifyExecutablePlan(allocator, plan, &plan_diagnostics.writer) catch |err| {
        const message = plan_diagnostics.writer.buffered();
        const fallback = "invalid executable plan";
        const text = if (message.len == 0) fallback else message;
        const code: c.PJRT_Error_Code = @intCast(switch (err) {
            error.InvalidExecutablePlan => c.PJRT_Error_Code_INVALID_ARGUMENT,
            else => c.PJRT_Error_Code_INTERNAL,
        });
        return makeError(code, text);
    };

    const logical_ids = allocator.alloc(c.PJRT_LogicalDeviceIds, plan.options.numDevices()) catch return makeError(c.PJRT_Error_Code_INTERNAL, "failed to allocate executable logical device ids");
    errdefer allocator.free(logical_ids);
    for (logical_ids, 0..) |*id, index| {
        id.* = .{
            .replica = @intCast(index / @as(usize, @intCast(plan.options.num_partitions))),
            .partition = @intCast(index % @as(usize, @intCast(plan.options.num_partitions))),
        };
    }

    const optimized_program = if (analysis) |owned_analysis|
        allocator.dupe(u8, owned_analysis.source) catch return makeError(c.PJRT_Error_Code_INTERNAL, "failed to allocate optimized program")
    else
        allocator.dupe(u8, "module {}\n") catch return makeError(c.PJRT_Error_Code_INTERNAL, "failed to allocate optimized program");
    errdefer allocator.free(optimized_program);

    const parameter_memory_kinds = allocator.alloc([*c]const u8, plan.parameter_shardings.len) catch return makeError(c.PJRT_Error_Code_INTERNAL, "failed to allocate parameter memory kind table");
    errdefer allocator.free(parameter_memory_kinds);
    const parameter_memory_kind_sizes = allocator.alloc(usize, plan.parameter_shardings.len) catch return makeError(c.PJRT_Error_Code_INTERNAL, "failed to allocate parameter memory kind sizes");
    errdefer allocator.free(parameter_memory_kind_sizes);
    fillMemoryKindArrays(parameter_memory_kinds, parameter_memory_kind_sizes);

    const output_memory_kinds = allocator.alloc([*c]const u8, plan.output_shardings.len) catch return makeError(c.PJRT_Error_Code_INTERNAL, "failed to allocate output memory kind table");
    errdefer allocator.free(output_memory_kinds);
    const output_memory_kind_sizes = allocator.alloc(usize, plan.output_shardings.len) catch return makeError(c.PJRT_Error_Code_INTERNAL, "failed to allocate output memory kind sizes");
    errdefer allocator.free(output_memory_kind_sizes);
    fillMemoryKindArrays(output_memory_kinds, output_memory_kind_sizes);

    const fingerprint_value = std.hash.Wyhash.hash(0, optimized_program);
    const fingerprint = std.fmt.allocPrint(allocator, "pjrtx-{x}", .{fingerprint_value}) catch return makeError(c.PJRT_Error_Code_INTERNAL, "failed to allocate executable fingerprint");
    errdefer allocator.free(fingerprint);

    const plan_ptr = allocator.create(compiler.ExecutablePlan) catch return makeError(c.PJRT_Error_Code_INTERNAL, "failed to allocate executable plan storage");
    plan_ptr.* = plan;
    plan_moved = true;
    errdefer {
        plan_ptr.deinit();
        allocator.destroy(plan_ptr);
    }

    var graph = runtime.ExecutableGraph.init(allocator, client, plan_ptr) catch {
        return makeError(c.PJRT_Error_Code_INTERNAL, "failed to build executable graph");
    };
    errdefer graph.deinit();

    const executable = allocator.create(Executable) catch return makeError(c.PJRT_Error_Code_INTERNAL, "failed to allocate executable");
    executable.* = .{
        .client = client,
        .plan = plan_ptr,
        .graph = graph,
        .logical_ids = logical_ids,
        .optimized_program = optimized_program,
        .parameter_memory_kinds = parameter_memory_kinds,
        .parameter_memory_kind_sizes = parameter_memory_kind_sizes,
        .output_memory_kinds = output_memory_kinds,
        .output_memory_kind_sizes = output_memory_kind_sizes,
        .fingerprint = fingerprint,
    };
    args[0].executable = @ptrCast(executable);
    trace(
        "event=compile backend={s} devices={d} values={d} instructions={d} outputs={d} backend_executable={d} elapsed_us={d}",
        .{
            @tagName(client.backend_kind),
            plan_ptr.options.numDevices(),
            plan_ptr.values.len,
            plan_ptr.instructions.len,
            plan_ptr.output_ids.len,
            @intFromBool(graph.backend_executable != null),
            elapsedUs(trace_start_ns),
        },
    );
    return null;
}

fn pjrtClientDefaultDeviceAssignment(args: [*c]c.PJRT_Client_DefaultDeviceAssignment_Args) callconv(.c) ?*c.PJRT_Error {
    const client = clientFromC(args[0].client);
    const needed: usize = @intCast(args[0].num_replicas * args[0].num_partitions);
    if (needed > client.devices.len or needed > args[0].default_assignment_size) {
        return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "invalid default device assignment request");
    }
    for (0..needed) |i| args[0].default_assignment[i] = @intCast(i);
    return null;
}

fn pjrtClientBufferFromHostBuffer(args: [*c]c.PJRT_Client_BufferFromHostBuffer_Args) callconv(.c) ?*c.PJRT_Error {
    const trace_start_ns = nowNs();
    const client = clientFromC(args[0].client);
    const device = if (args[0].device) |dev| deviceFromC(dev) else &client.devices[0];
    const memory = if (args[0].memory) |mem| memoryFromC(mem) else &client.memories[@intCast(device.default_memory_id)];
    const dims = args[0].dims[0..args[0].num_dims];
    const byte_size = denseByteSize(args[0].type, dims);
    const data = @as([*]const u8, @ptrCast(args[0].data))[0..byte_size];
    const buffer = runtime.Buffer.initHostCopyForBackend(allocator, client.backend, runtimeTypeFromPjrt(args[0].type), dims, device, memory, @intCast(device.id), data) catch {
        return makeError(c.PJRT_Error_Code_INTERNAL, "failed to create host buffer copy");
    };
    args[0].buffer = @ptrCast(buffer);
    args[0].done_with_host_buffer = eventCreateReady();
    trace(
        "event=h2d bytes={d} dtype={s} rank={d} device={d} backend_storage={d} elapsed_us={d}",
        .{
            byte_size,
            @tagName(buffer.element_type),
            dims.len,
            device.id,
            @intFromBool(buffer.hasBackendStorage()),
            elapsedUs(trace_start_ns),
        },
    );
    return null;
}

fn topologyDescriptionCreate(_: [*c]c.PJRT_TopologyDescription_Create_Args) callconv(.c) ?*c.PJRT_Error {
    return unimplemented("standalone topology creation is not implemented yet");
}

fn topologyDescriptionDestroy(_: [*c]c.PJRT_TopologyDescription_Destroy_Args) callconv(.c) ?*c.PJRT_Error {
    return null;
}

fn topologyDescriptionPlatformName(args: [*c]c.PJRT_TopologyDescription_PlatformName_Args) callconv(.c) ?*c.PJRT_Error {
    _ = topologyFromC(args[0].topology);
    args[0].platform_name = platform_name;
    args[0].platform_name_size = platform_name.len;
    return null;
}

fn topologyDescriptionPlatformVersion(args: [*c]c.PJRT_TopologyDescription_PlatformVersion_Args) callconv(.c) ?*c.PJRT_Error {
    _ = topologyFromC(args[0].topology);
    args[0].platform_version = platform_version;
    args[0].platform_version_size = platform_version.len;
    return null;
}

fn topologyDescriptionGetDeviceDescriptions(args: [*c]c.PJRT_TopologyDescription_GetDeviceDescriptions_Args) callconv(.c) ?*c.PJRT_Error {
    const client = topologyFromC(args[0].topology);
    args[0].descriptions = @ptrCast(client.device_handles.ptr);
    args[0].num_descriptions = client.device_handles.len;
    return null;
}

fn topologySerializedDelete(serialized_topology: ?*c.PJRT_SerializedTopology) callconv(.c) void {
    if (serialized_topology) |opaque_topology| {
        const topology: *SerializedTopology = @ptrCast(@alignCast(opaque_topology));
        allocator.free(topology.bytes);
        allocator.destroy(topology);
    }
}

fn topologyDescriptionSerialize(args: [*c]c.PJRT_TopologyDescription_Serialize_Args) callconv(.c) ?*c.PJRT_Error {
    const client = topologyFromC(args[0].topology);
    var writer = std.Io.Writer.Allocating.init(allocator);
    defer writer.deinit();
    writer.writer.print("platform={s};version={s};devices={d}", .{ platform_name, platform_version, client.devices.len }) catch {
        return makeError(c.PJRT_Error_Code_INTERNAL, "failed to serialize topology");
    };
    for (client.devices) |device| {
        writer.writer.print(";device={d}:{d}:{d}:{s}", .{ device.id, device.local_hardware_id, device.process_index, device.name }) catch {
            return makeError(c.PJRT_Error_Code_INTERNAL, "failed to serialize topology");
        };
    }

    const topology = allocator.create(SerializedTopology) catch {
        return makeError(c.PJRT_Error_Code_INTERNAL, "failed to allocate serialized topology");
    };
    topology.* = .{ .bytes = writer.toOwnedSlice() catch {
        allocator.destroy(topology);
        return makeError(c.PJRT_Error_Code_INTERNAL, "failed to allocate serialized topology bytes");
    } };

    args[0].serialized_bytes = topology.bytes.ptr;
    args[0].serialized_bytes_size = topology.bytes.len;
    args[0].serialized_topology = @ptrCast(topology);
    args[0].serialized_topology_deleter = topologySerializedDelete;
    return null;
}

fn topologyDescriptionDeserialize(_: [*c]c.PJRT_TopologyDescription_Deserialize_Args) callconv(.c) ?*c.PJRT_Error {
    return unimplemented("topology deserialization is not implemented yet");
}

fn topologyDescriptionAttributes(args: [*c]c.PJRT_TopologyDescription_Attributes_Args) callconv(.c) ?*c.PJRT_Error {
    _ = topologyFromC(args[0].topology);
    args[0].attributes = null;
    args[0].num_attributes = 0;
    return null;
}

fn topologyDescriptionFingerprint(args: [*c]c.PJRT_TopologyDescription_Fingerprint_Args) callconv(.c) ?*c.PJRT_Error {
    const client = topologyFromC(args[0].topology);
    var hasher = std.hash.Wyhash.init(0);
    hasher.update(platform_name);
    hasher.update(platform_version);
    for (client.devices) |device| {
        hasher.update(std.mem.asBytes(&device.id));
        hasher.update(std.mem.asBytes(&device.local_hardware_id));
        hasher.update(std.mem.asBytes(&device.process_index));
        hasher.update(device.name);
    }
    args[0].fingerprint = hasher.final();
    return null;
}

fn deviceDescriptionId(args: [*c]c.PJRT_DeviceDescription_Id_Args) callconv(.c) ?*c.PJRT_Error {
    const device: *runtime.Device = @ptrCast(@alignCast(args[0].device_description.?));
    args[0].id = device.id;
    return null;
}

fn deviceDescriptionProcessIndex(args: [*c]c.PJRT_DeviceDescription_ProcessIndex_Args) callconv(.c) ?*c.PJRT_Error {
    const device: *runtime.Device = @ptrCast(@alignCast(args[0].device_description.?));
    args[0].process_index = device.process_index;
    return null;
}

fn deviceDescriptionAttributes(args: [*c]c.PJRT_DeviceDescription_Attributes_Args) callconv(.c) ?*c.PJRT_Error {
    args[0].attributes = null;
    args[0].num_attributes = 0;
    return null;
}

fn deviceDescriptionKind(args: [*c]c.PJRT_DeviceDescription_Kind_Args) callconv(.c) ?*c.PJRT_Error {
    args[0].device_kind = device_kind;
    args[0].device_kind_size = device_kind.len;
    return null;
}

fn deviceDescriptionDebugString(args: [*c]c.PJRT_DeviceDescription_DebugString_Args) callconv(.c) ?*c.PJRT_Error {
    const device: *runtime.Device = @ptrCast(@alignCast(args[0].device_description.?));
    args[0].debug_string = device.debug_string.ptr;
    args[0].debug_string_size = device.debug_string.len;
    return null;
}

fn deviceDescriptionToString(args: [*c]c.PJRT_DeviceDescription_ToString_Args) callconv(.c) ?*c.PJRT_Error {
    return deviceDescriptionDebugString(@ptrCast(args));
}

fn deviceGetDescription(args: [*c]c.PJRT_Device_GetDescription_Args) callconv(.c) ?*c.PJRT_Error {
    args[0].device_description = @ptrCast(args[0].device);
    return null;
}

fn deviceIsAddressable(args: [*c]c.PJRT_Device_IsAddressable_Args) callconv(.c) ?*c.PJRT_Error {
    args[0].is_addressable = deviceFromC(args[0].device).addressable;
    return null;
}

fn deviceLocalHardwareId(args: [*c]c.PJRT_Device_LocalHardwareId_Args) callconv(.c) ?*c.PJRT_Error {
    args[0].local_hardware_id = deviceFromC(args[0].device).local_hardware_id;
    return null;
}

fn deviceAddressableMemories(args: [*c]c.PJRT_Device_AddressableMemories_Args) callconv(.c) ?*c.PJRT_Error {
    const device = deviceFromC(args[0].device);
    args[0].memories = @ptrCast(device.addressable_memories.ptr);
    args[0].num_memories = device.addressable_memories.len;
    return null;
}

fn deviceDefaultMemory(args: [*c]c.PJRT_Device_DefaultMemory_Args) callconv(.c) ?*c.PJRT_Error {
    const device = deviceFromC(args[0].device);
    args[0].memory = @ptrCast(device.default_memory);
    return null;
}

fn deviceMemoryStats(args: [*c]c.PJRT_Device_MemoryStats_Args) callconv(.c) ?*c.PJRT_Error {
    args[0].bytes_in_use = 0;
    args[0].peak_bytes_in_use = 0;
    args[0].peak_bytes_in_use_is_set = true;
    return null;
}

fn clampI64(value: u64) i64 {
    return @intCast(@min(value, @as(u64, @intCast(std.math.maxInt(i64)))));
}

fn initDeviceAttrString(attr: *c.PJRT_NamedValue, name: []const u8, value: []const u8) void {
    attr.* = std.mem.zeroes(c.PJRT_NamedValue);
    attr.struct_size = c.PJRT_NamedValue_STRUCT_SIZE;
    attr.name = name.ptr;
    attr.name_size = name.len;
    attr.type = c.PJRT_NamedValue_kString;
    attr.unnamed_0.string_value = value.ptr;
    attr.value_size = value.len;
}

fn initDeviceAttrInt64(attr: *c.PJRT_NamedValue, name: []const u8, value: i64) void {
    attr.* = std.mem.zeroes(c.PJRT_NamedValue);
    attr.struct_size = c.PJRT_NamedValue_STRUCT_SIZE;
    attr.name = name.ptr;
    attr.name_size = name.len;
    attr.type = c.PJRT_NamedValue_kInt64;
    attr.unnamed_0.int64_value = value;
    attr.value_size = 1;
}

fn initDeviceAttrBool(attr: *c.PJRT_NamedValue, name: []const u8, value: bool) void {
    attr.* = std.mem.zeroes(c.PJRT_NamedValue);
    attr.struct_size = c.PJRT_NamedValue_STRUCT_SIZE;
    attr.name = name.ptr;
    attr.name_size = name.len;
    attr.type = c.PJRT_NamedValue_kBool;
    attr.unnamed_0.bool_value = value;
    attr.value_size = 1;
}

fn deviceAttributesDelete(device_attributes: ?*c.PJRT_Device_Attributes) callconv(.c) void {
    if (device_attributes) |opaque_attrs| {
        const owned: *DeviceAttributes = @ptrCast(@alignCast(opaque_attrs));
        allocator.destroy(owned);
    }
}

fn deviceGetAttributes(args: [*c]c.PJRT_Device_GetAttributes_Args) callconv(.c) ?*c.PJRT_Error {
    const device = deviceFromC(args[0].device);
    const owned = allocator.create(DeviceAttributes) catch {
        return makeError(c.PJRT_Error_Code_INTERNAL, "failed to allocate device attributes");
    };
    owned.* = .{ .attrs = std.mem.zeroes([5]c.PJRT_NamedValue) };

    initDeviceAttrString(&owned.attrs[0], "device_name", device.name);
    initDeviceAttrInt64(&owned.attrs[1], "pjrtx_registry_id", clampI64(device.registry_id));
    initDeviceAttrInt64(&owned.attrs[2], "pjrtx_recommended_working_set_size", clampI64(device.memory_bytes));
    initDeviceAttrInt64(&owned.attrs[3], "pjrtx_has_unified_memory", if (device.has_unified_memory) 1 else 0);
    initDeviceAttrInt64(&owned.attrs[4], "pjrtx_default_memory_id", device.default_memory_id);

    args[0].attributes = &owned.attrs;
    args[0].num_attributes = owned.attrs.len;
    args[0].device_attributes = @ptrCast(owned);
    args[0].attributes_deleter = deviceAttributesDelete;
    return null;
}

fn memoryId(args: [*c]c.PJRT_Memory_Id_Args) callconv(.c) ?*c.PJRT_Error {
    args[0].id = memoryFromC(args[0].memory).id;
    return null;
}

fn memoryKind(args: [*c]c.PJRT_Memory_Kind_Args) callconv(.c) ?*c.PJRT_Error {
    const memory = memoryFromC(args[0].memory);
    const text = @tagName(memory.kind);
    args[0].kind = text.ptr;
    args[0].kind_size = text.len;
    return null;
}

fn memoryDebugString(args: [*c]c.PJRT_Memory_DebugString_Args) callconv(.c) ?*c.PJRT_Error {
    const memory = memoryFromC(args[0].memory);
    args[0].debug_string = memory.debug_string.ptr;
    args[0].debug_string_size = memory.debug_string.len;
    return null;
}

fn memoryToString(args: [*c]c.PJRT_Memory_ToString_Args) callconv(.c) ?*c.PJRT_Error {
    const memory = memoryFromC(args[0].memory);
    args[0].to_string = memory.debug_string.ptr;
    args[0].to_string_size = memory.debug_string.len;
    return null;
}

fn memoryAddressableByDevices(args: [*c]c.PJRT_Memory_AddressableByDevices_Args) callconv(.c) ?*c.PJRT_Error {
    const memory = memoryFromC(args[0].memory);
    args[0].devices = @ptrCast(memory.addressable_devices.ptr);
    args[0].num_devices = memory.addressable_devices.len;
    return null;
}

fn loadedExecutableDestroy(args: [*c]c.PJRT_LoadedExecutable_Destroy_Args) callconv(.c) ?*c.PJRT_Error {
    executableFromC(args[0].executable).deinit();
    return null;
}

fn loadedExecutableGetExecutable(args: [*c]c.PJRT_LoadedExecutable_GetExecutable_Args) callconv(.c) ?*c.PJRT_Error {
    args[0].executable = @ptrCast(args[0].loaded_executable);
    return null;
}

fn loadedExecutableAddressableDevices(args: [*c]c.PJRT_LoadedExecutable_AddressableDevices_Args) callconv(.c) ?*c.PJRT_Error {
    const executable = executableFromC(args[0].executable);
    const num_devices = @min(executable.plan.options.numDevices(), executable.client.device_handles.len);
    args[0].addressable_devices = @ptrCast(executable.client.device_handles[0..num_devices].ptr);
    args[0].num_addressable_devices = num_devices;
    return null;
}

fn loadedExecutableAddressableDeviceLogicalIds(args: [*c]c.PJRT_LoadedExecutable_AddressableDeviceLogicalIds_Args) callconv(.c) ?*c.PJRT_Error {
    const executable = executableFromC(args[0].executable);
    args[0].addressable_device_logical_ids = executable.logical_ids.ptr;
    args[0].num_addressable_device_logical_ids = executable.logical_ids.len;
    return null;
}

fn deviceAssignmentSerializedDeleter(_: ?*c.PJRT_DeviceAssignmentSerialized) callconv(.c) void {}

fn loadedExecutableGetDeviceAssignment(args: [*c]c.PJRT_LoadedExecutable_GetDeviceAssignment_Args) callconv(.c) ?*c.PJRT_Error {
    args[0].serialized_bytes = null;
    args[0].serialized_bytes_size = 0;
    args[0].serialized_device_assignment = null;
    args[0].serialized_device_assignment_deleter = deviceAssignmentSerializedDeleter;
    return null;
}

fn loadedExecutableDelete(args: [*c]c.PJRT_LoadedExecutable_Delete_Args) callconv(.c) ?*c.PJRT_Error {
    executableFromC(args[0].executable).deleted = true;
    return null;
}

fn loadedExecutableIsDeleted(args: [*c]c.PJRT_LoadedExecutable_IsDeleted_Args) callconv(.c) ?*c.PJRT_Error {
    args[0].is_deleted = executableFromC(args[0].executable).deleted;
    return null;
}

fn runtimeBinaryOp(kind: compiler.PlanInstructionKind) ?runtime.ElementwiseBinaryOp {
    return switch (kind) {
        .add => .add,
        .subtract => .subtract,
        .multiply => .multiply,
        .divide => .divide,
        .maximum => .maximum,
        .minimum => .minimum,
        .power => .power,
        .atan2 => .atan2,
        .remainder => .remainder,
        .and_ => .and_,
        .or_ => .or_,
        .xor => .xor,
        .shift_left => .shift_left,
        .shift_right_arithmetic => .shift_right_arithmetic,
        .shift_right_logical => .shift_right_logical,
        else => null,
    };
}

fn runtimeUnaryOp(kind: compiler.PlanInstructionKind) ?runtime.ElementwiseUnaryOp {
    return switch (kind) {
        .negate => .negate,
        .exp => .exp,
        .expm1 => .expm1,
        .tanh => .tanh,
        .sqrt => .sqrt,
        .rsqrt => .rsqrt,
        .abs => .abs,
        .cbrt => .cbrt,
        .ceil => .ceil,
        .floor => .floor,
        .log => .log,
        .log1p => .log1p,
        .logistic => .logistic,
        .sine => .sine,
        .cosine => .cosine,
        .not_ => .not_,
        .sign => .sign,
        .is_finite => .is_finite,
        .round_nearest_afz => .round_nearest_afz,
        .round_nearest_even => .round_nearest_even,
        .popcnt => .popcnt,
        .count_leading_zeros => .count_leading_zeros,
        else => null,
    };
}

fn destroyOwnedBuffer(buffer: *runtime.Buffer, owned: bool) void {
    if (owned) buffer.deinit();
}

fn destroyExecuteValues(value_buffers: []?*runtime.Buffer, value_owned: []const bool) void {
    for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
}

fn unsupportedRuntimeFeature(kind: compiler.PlanInstructionKind) ?*c.PJRT_Error {
    const message = switch (kind) {
        .complex => "StableHLO complex execution is staged for complex dtype support; feature=heavy-control-random-structural",
        .real => "StableHLO real execution is staged for complex dtype support; feature=heavy-control-random-structural",
        .imag => "StableHLO imag execution is staged for complex dtype support; feature=heavy-control-random-structural",
        .fft => "StableHLO fft execution is staged for complex dtype and backend FFT lowering; feature=heavy-control-random-structural",
        .convolution => "StableHLO convolution execution is staged for dimension-number lowering and backend legalization; feature=heavy-control-random-structural",
        .scatter => "StableHLO scatter execution is staged for region-aware update lowering; feature=heavy-control-random-structural",
        .custom_call => "StableHLO custom_call execution requires a registered PjRTx custom target; feature=heavy-control-random-structural",
        .get_tuple_element, .tuple => "StableHLO tuple execution is staged for structural value lowering; feature=heavy-control-random-structural",
        .while_ => "StableHLO while execution is staged for region/control-flow lowering; feature=heavy-control-random-structural",
        .triangular_solve => "StableHLO triangular_solve execution is staged for option-aware linear algebra lowering; feature=heavy-control-random-structural",
        else => "StableHLO execution is staged for this operation; feature=heavy-control-random-structural",
    };
    return makeError(c.PJRT_Error_Code_UNIMPLEMENTED, message);
}

fn graphExecuteError(err: runtime.GraphExecuteError) ?*c.PJRT_Error {
    return switch (err) {
        error.OutOfMemory => makeError(c.PJRT_Error_Code_INTERNAL, "failed to allocate executable graph execution state"),
        error.InvalidArgument => makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "invalid executable graph arguments or device assignment"),
        error.UnsupportedElementType => makeError(c.PJRT_Error_Code_UNIMPLEMENTED, "executable graph contains an operation unsupported for this element type"),
        error.ShapeMismatch => makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "executable graph shape validation failed during execution"),
        error.UnsupportedRuntimeFeature => makeError(c.PJRT_Error_Code_UNIMPLEMENTED, "executable graph contains an operation that is not lowered to the runtime yet"),
        error.Internal => makeError(c.PJRT_Error_Code_INTERNAL, "failed to execute executable graph"),
    };
}

fn loadedExecutableExecute(args: [*c]c.PJRT_LoadedExecutable_Execute_Args) callconv(.c) ?*c.PJRT_Error {
    const trace_start_ns = nowNs();
    const executable = executableFromC(args[0].executable);
    if (args[0].num_args == 0 and executable.plan.parameter_shardings.len != 0) return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "PjRTx execute expects arguments for executable parameters");
    if (args[0].num_devices > executable.graph.device_ids.len) return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "PjRTx execute requested more devices than the executable graph contains");
    var total_outputs: usize = 0;
    var backend_candidate = executable.graph.backend_executable != null;
    for (0..args[0].num_devices) |device_index| {
        const arguments = allocator.alloc(*runtime.Buffer, args[0].num_args) catch return makeError(c.PJRT_Error_Code_INTERNAL, "failed to allocate executable graph argument list");
        defer allocator.free(arguments);
        for (arguments, 0..) |*argument, argument_index| {
            argument.* = bufferFromC(args[0].argument_lists[device_index][argument_index]);
            backend_candidate = backend_candidate and argument.*.hasBackendStorage();
        }

        const outputs = executable.graph.executeDevice(allocator, executable.client, executable.plan, device_index, arguments) catch |err| {
            return graphExecuteError(err);
        };
        defer allocator.free(outputs);
        total_outputs += outputs.len;
        for (outputs, 0..) |output, output_index| {
            args[0].output_lists[device_index][output_index] = @ptrCast(output);
        }
        if (args[0].device_complete_events) |events| events[device_index] = eventCreateReady();
    }
    trace(
        "event=execute devices={d} args={d} outputs={d} backend_candidate={d} elapsed_us={d}",
        .{
            args[0].num_devices,
            args[0].num_args,
            total_outputs,
            @intFromBool(backend_candidate),
            elapsedUs(trace_start_ns),
        },
    );
    return null;
}

fn loadedExecutableExecuteLegacy(args: [*c]c.PJRT_LoadedExecutable_Execute_Args) callconv(.c) ?*c.PJRT_Error {
    const executable = executableFromC(args[0].executable);
    if (args[0].num_args == 0 and executable.plan.parameter_shardings.len != 0) return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "PjRTx bootstrap execute expects arguments for executable parameters");
    if (executable.plan.instructions.len == 0) return makeError(c.PJRT_Error_Code_UNIMPLEMENTED, "executable plan has no runtime operations");
    for (0..args[0].num_devices) |device_index| {
        const value_buffers = allocator.alloc(?*runtime.Buffer, executable.plan.values.len) catch return makeError(c.PJRT_Error_Code_INTERNAL, "failed to allocate execute value table");
        defer allocator.free(value_buffers);
        @memset(value_buffers, null);
        const value_owned = allocator.alloc(bool, executable.plan.values.len) catch {
            allocator.free(value_buffers);
            return makeError(c.PJRT_Error_Code_INTERNAL, "failed to allocate execute ownership table");
        };
        defer allocator.free(value_owned);
        @memset(value_owned, false);
        var output_value_ids = allocator.alloc(bool, executable.plan.values.len) catch {
            allocator.free(value_buffers);
            allocator.free(value_owned);
            return makeError(c.PJRT_Error_Code_INTERNAL, "failed to allocate execute output id table");
        };
        defer allocator.free(output_value_ids);
        @memset(output_value_ids, false);
        for (executable.plan.output_ids) |id| {
            if (id.index < output_value_ids.len) output_value_ids[id.index] = true;
        }
        var parameter_index: usize = 0;
        for (executable.plan.values) |value| {
            if (value.role != .parameter) continue;
            if (parameter_index >= args[0].num_args) {
                for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "PjRTx bootstrap execute received fewer arguments than executable parameters");
            }
            value_buffers[value.id.index] = bufferFromC(args[0].argument_lists[device_index][parameter_index]);
            parameter_index += 1;
        }
        for (executable.plan.instructions) |plan_instruction| {
            if (plan_instruction.kind == .rng_bit_generator) {
                if (plan_instruction.inputs.len < 1 or plan_instruction.outputs.len != 2) {
                    destroyExecuteValues(value_buffers, value_owned);
                    return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "rng_bit_generator StableHLO plan must have one state input and two outputs");
                }
                const state_input = value_buffers[plan_instruction.inputs[0].index].?;
                const state_output = runtime.Buffer.initRngStateUpdate(allocator, state_input, device_index) catch {
                    destroyExecuteValues(value_buffers, value_owned);
                    return makeError(c.PJRT_Error_Code_INTERNAL, "failed to execute rng_bit_generator state update");
                };
                const bits_descriptor = executable.plan.values[plan_instruction.outputs[1].index].descriptor;
                const bits_output = runtime.Buffer.initRngBits(
                    allocator,
                    state_input,
                    bits_descriptor.element_type,
                    plan_instruction.dims orelse bits_descriptor.dims,
                    device_index,
                ) catch |err| switch (err) {
                    error.UnsupportedElementType => {
                        state_output.deinit();
                        destroyExecuteValues(value_buffers, value_owned);
                        return makeError(c.PJRT_Error_Code_UNIMPLEMENTED, "rng_bit_generator StableHLO execution currently supports u32/s32 random bits");
                    },
                    else => {
                        state_output.deinit();
                        destroyExecuteValues(value_buffers, value_owned);
                        return makeError(c.PJRT_Error_Code_INTERNAL, "failed to execute rng_bit_generator StableHLO op");
                    },
                };
                value_buffers[plan_instruction.outputs[0].index] = state_output;
                value_owned[plan_instruction.outputs[0].index] = true;
                value_buffers[plan_instruction.outputs[1].index] = bits_output;
                value_owned[plan_instruction.outputs[1].index] = true;
                continue;
            }
            if (plan_instruction.outputs.len != 1) {
                destroyExecuteValues(value_buffers, value_owned);
                return makeError(c.PJRT_Error_Code_INTERNAL, "executable plan instruction arity is invalid");
            }
            const output_id = plan_instruction.outputs[0];
            const next = switch (plan_instruction.kind) {
                .constant => blk: {
                    const descriptor = executable.plan.values[output_id.index].descriptor;
                    const device = &executable.client.devices[device_index];
                    const memory = device.default_memory;
                    break :blk runtime.Buffer.initHostCopyForBackend(
                        allocator,
                        executable.client.backend,
                        descriptor.element_type,
                        descriptor.dims,
                        device,
                        memory,
                        device_index,
                        plan_instruction.literal orelse &.{},
                    ) catch {
                        for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                        return makeError(c.PJRT_Error_Code_INTERNAL, "failed to allocate StableHLO constant");
                    };
                },
                .copy_arg0 => runtime.Buffer.initDeviceCopy(allocator, value_buffers[plan_instruction.inputs[0].index].?, device_index) catch {
                    for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                    return makeError(c.PJRT_Error_Code_INTERNAL, "failed to allocate execute output");
                },
                .reduce_precision => runtime.Buffer.initDeviceCopy(allocator, value_buffers[plan_instruction.inputs[0].index].?, device_index) catch {
                    destroyExecuteValues(value_buffers, value_owned);
                    return makeError(c.PJRT_Error_Code_INTERNAL, "failed to execute reduce_precision StableHLO op");
                },
                .partition_id => blk: {
                    const descriptor = executable.plan.values[output_id.index].descriptor;
                    const device = &executable.client.devices[device_index];
                    break :blk runtime.Buffer.initPartitionId(
                        allocator,
                        executable.client.backend,
                        descriptor.element_type,
                        descriptor.dims,
                        device,
                        device.default_memory,
                        @intCast(device_index),
                        device_index,
                    ) catch |err| switch (err) {
                        error.UnsupportedElementType => {
                            destroyExecuteValues(value_buffers, value_owned);
                            return makeError(c.PJRT_Error_Code_UNIMPLEMENTED, "partition_id StableHLO execution currently supports u32/s32 scalar outputs");
                        },
                        error.ShapeMismatch => {
                            destroyExecuteValues(value_buffers, value_owned);
                            return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "partition_id StableHLO output must be scalar");
                        },
                        else => {
                            destroyExecuteValues(value_buffers, value_owned);
                            return makeError(c.PJRT_Error_Code_INTERNAL, "failed to execute partition_id StableHLO op");
                        },
                    };
                },
                .cholesky => runtime.Buffer.initCholesky(
                    allocator,
                    value_buffers[plan_instruction.inputs[0].index].?,
                    plan_instruction.lower orelse true,
                    plan_instruction.dims orelse executable.plan.values[output_id.index].descriptor.dims,
                    device_index,
                ) catch |err| switch (err) {
                    error.UnsupportedElementType => {
                        destroyExecuteValues(value_buffers, value_owned);
                        return makeError(c.PJRT_Error_Code_UNIMPLEMENTED, "cholesky StableHLO execution currently supports dense f32 tensors only");
                    },
                    error.ShapeMismatch => {
                        destroyExecuteValues(value_buffers, value_owned);
                        return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "cholesky StableHLO input must be positive-definite square matrices");
                    },
                    else => {
                        destroyExecuteValues(value_buffers, value_owned);
                        return makeError(c.PJRT_Error_Code_INTERNAL, "failed to execute cholesky StableHLO op");
                    },
                },
                .rng => runtime.Buffer.initRngUniform(
                    allocator,
                    value_buffers[plan_instruction.inputs[0].index].?,
                    value_buffers[plan_instruction.inputs[1].index].?,
                    executable.plan.values[output_id.index].descriptor.element_type,
                    plan_instruction.dims orelse executable.plan.values[output_id.index].descriptor.dims,
                    device_index,
                ) catch |err| switch (err) {
                    error.UnsupportedElementType => {
                        destroyExecuteValues(value_buffers, value_owned);
                        return makeError(c.PJRT_Error_Code_UNIMPLEMENTED, "rng StableHLO execution currently supports deterministic uniform f32/u32/s32 generation");
                    },
                    error.ShapeMismatch => {
                        destroyExecuteValues(value_buffers, value_owned);
                        return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "rng StableHLO min/max inputs must be scalar");
                    },
                    else => {
                        destroyExecuteValues(value_buffers, value_owned);
                        return makeError(c.PJRT_Error_Code_INTERNAL, "failed to execute rng StableHLO op");
                    },
                },
                .add, .subtract, .multiply, .divide, .maximum, .minimum, .power, .atan2, .remainder, .and_, .or_, .xor, .shift_left, .shift_right_arithmetic, .shift_right_logical => blk: {
                    const lhs = value_buffers[plan_instruction.inputs[0].index].?;
                    const rhs = value_buffers[plan_instruction.inputs[1].index].?;
                    break :blk runtime.Buffer.initElementwiseBinary(allocator, runtimeBinaryOp(plan_instruction.kind).?, lhs, rhs, device_index) catch |err| switch (err) {
                        error.UnsupportedElementType => {
                            for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                            return makeError(c.PJRT_Error_Code_UNIMPLEMENTED, "binary StableHLO execution currently supports u8 and f32 buffers only");
                        },
                        error.ShapeMismatch => {
                            for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                            return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "binary StableHLO arguments must have the same shape");
                        },
                        else => {
                            for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                            return makeError(c.PJRT_Error_Code_INTERNAL, "failed to execute binary StableHLO op");
                        },
                    };
                },
                .negate, .exp, .expm1, .tanh, .sqrt, .rsqrt, .abs, .cbrt, .ceil, .floor, .log, .log1p, .logistic, .sine, .cosine, .not_, .sign, .is_finite, .round_nearest_afz, .round_nearest_even, .popcnt, .count_leading_zeros => runtime.Buffer.initElementwiseUnaryTyped(
                    allocator,
                    runtimeUnaryOp(plan_instruction.kind).?,
                    value_buffers[plan_instruction.inputs[0].index].?,
                    executable.plan.values[output_id.index].descriptor.element_type,
                    plan_instruction.dims orelse executable.plan.values[output_id.index].descriptor.dims,
                    device_index,
                ) catch |err| switch (err) {
                    error.UnsupportedElementType => {
                        for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                        return makeError(c.PJRT_Error_Code_UNIMPLEMENTED, "unary StableHLO execution currently supports u8 negate and f32 math buffers only");
                    },
                    else => {
                        for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                        return makeError(c.PJRT_Error_Code_INTERNAL, "failed to execute unary StableHLO op");
                    },
                },
                .convert => runtime.Buffer.initConvert(
                    allocator,
                    value_buffers[plan_instruction.inputs[0].index].?,
                    executable.plan.values[output_id.index].descriptor.element_type,
                    plan_instruction.dims orelse executable.plan.values[output_id.index].descriptor.dims,
                    device_index,
                ) catch |err| switch (err) {
                    error.UnsupportedElementType => {
                        for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                        return makeError(c.PJRT_Error_Code_UNIMPLEMENTED, "convert StableHLO execution currently supports pred/u8/s8/s32/u32/f32 scalar conversions");
                    },
                    error.ShapeMismatch => {
                        for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                        return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "convert StableHLO output shape must match input shape");
                    },
                    else => {
                        for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                        return makeError(c.PJRT_Error_Code_INTERNAL, "failed to execute convert StableHLO op");
                    },
                },
                .bitcast_convert => {
                    for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                    return makeError(c.PJRT_Error_Code_UNIMPLEMENTED, "bitcast_convert StableHLO execution needs backend dtype reinterpretation support");
                },
                .iota => blk: {
                    const descriptor = executable.plan.values[output_id.index].descriptor;
                    const device = &executable.client.devices[device_index];
                    break :blk runtime.Buffer.initIota(
                        allocator,
                        executable.client.backend,
                        descriptor.element_type,
                        plan_instruction.dims orelse descriptor.dims,
                        device,
                        device.default_memory,
                        plan_instruction.iota_dimension orelse 0,
                        device_index,
                    ) catch |err| switch (err) {
                        error.UnsupportedElementType => {
                            for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                            return makeError(c.PJRT_Error_Code_UNIMPLEMENTED, "iota StableHLO execution currently supports scalar numeric element types");
                        },
                        error.ShapeMismatch => {
                            for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                            return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "iota StableHLO dimension is invalid");
                        },
                        else => {
                            for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                            return makeError(c.PJRT_Error_Code_INTERNAL, "failed to execute iota StableHLO op");
                        },
                    };
                },
                .reshape => runtime.Buffer.initReshape(allocator, value_buffers[plan_instruction.inputs[0].index].?, plan_instruction.dims orelse &.{}, device_index) catch |err| switch (err) {
                    error.UnsupportedElementType => {
                        for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                        return makeError(c.PJRT_Error_Code_UNIMPLEMENTED, "reshape StableHLO execution currently supports u8 and f32 MLX buffers only");
                    },
                    error.ShapeMismatch => {
                        for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                        return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "reshape StableHLO output shape must preserve byte size");
                    },
                    else => {
                        for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                        return makeError(c.PJRT_Error_Code_INTERNAL, "failed to execute reshape StableHLO op");
                    },
                },
                .transpose => runtime.Buffer.initTranspose(allocator, value_buffers[plan_instruction.inputs[0].index].?, plan_instruction.permutation orelse &.{}, plan_instruction.dims orelse &.{}, device_index) catch |err| switch (err) {
                    error.UnsupportedElementType => {
                        for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                        return makeError(c.PJRT_Error_Code_UNIMPLEMENTED, "transpose StableHLO execution currently supports dense u8 and f32 buffers only");
                    },
                    error.ShapeMismatch => {
                        for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                        return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "transpose StableHLO permutation or output shape is invalid");
                    },
                    else => {
                        for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                        return makeError(c.PJRT_Error_Code_INTERNAL, "failed to execute transpose StableHLO op");
                    },
                },
                .broadcast_in_dim => runtime.Buffer.initBroadcastInDim(allocator, value_buffers[plan_instruction.inputs[0].index].?, plan_instruction.broadcast_dimensions orelse &.{}, plan_instruction.dims orelse &.{}, device_index) catch |err| switch (err) {
                    error.UnsupportedElementType => {
                        for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                        return makeError(c.PJRT_Error_Code_UNIMPLEMENTED, "broadcast_in_dim StableHLO execution currently supports dense u8 and f32 buffers only");
                    },
                    error.ShapeMismatch => {
                        for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                        return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "broadcast_in_dim StableHLO dimensions or output shape are invalid");
                    },
                    else => {
                        for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                        return makeError(c.PJRT_Error_Code_INTERNAL, "failed to execute broadcast_in_dim StableHLO op");
                    },
                },
                .slice => runtime.Buffer.initSlice(
                    allocator,
                    value_buffers[plan_instruction.inputs[0].index].?,
                    plan_instruction.start_indices orelse &.{},
                    plan_instruction.limit_indices orelse &.{},
                    plan_instruction.strides orelse &.{},
                    plan_instruction.dims orelse &.{},
                    device_index,
                ) catch |err| switch (err) {
                    error.UnsupportedElementType => {
                        for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                        return makeError(c.PJRT_Error_Code_UNIMPLEMENTED, "slice StableHLO execution currently supports dense u8 and f32 buffers only");
                    },
                    error.ShapeMismatch => {
                        for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                        return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "slice StableHLO bounds, strides, or output shape are invalid");
                    },
                    else => {
                        for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                        return makeError(c.PJRT_Error_Code_INTERNAL, "failed to execute slice StableHLO op");
                    },
                },
                .dynamic_slice => blk: {
                    const src = value_buffers[plan_instruction.inputs[0].index].?;
                    const starts = allocator.alloc(*runtime.Buffer, plan_instruction.inputs.len - 1) catch {
                        for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                        return makeError(c.PJRT_Error_Code_INTERNAL, "failed to allocate dynamic_slice start table");
                    };
                    defer allocator.free(starts);
                    for (plan_instruction.inputs[1..], 0..) |input_id, i| starts[i] = value_buffers[input_id.index].?;
                    break :blk runtime.Buffer.initDynamicSlice(allocator, src, starts, plan_instruction.slice_sizes orelse &.{}, plan_instruction.dims orelse &.{}, device_index) catch |err| switch (err) {
                        error.UnsupportedElementType => {
                            for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                            return makeError(c.PJRT_Error_Code_UNIMPLEMENTED, "dynamic_slice StableHLO execution currently supports dense host fallback buffers only");
                        },
                        error.ShapeMismatch => {
                            for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                            return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "dynamic_slice StableHLO starts or slice sizes are invalid");
                        },
                        else => {
                            for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                            return makeError(c.PJRT_Error_Code_INTERNAL, "failed to execute dynamic_slice StableHLO op");
                        },
                    };
                },
                .dynamic_update_slice => blk: {
                    const starts = allocator.alloc(*runtime.Buffer, plan_instruction.inputs.len - 2) catch {
                        for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                        return makeError(c.PJRT_Error_Code_INTERNAL, "failed to allocate dynamic_update_slice start table");
                    };
                    defer allocator.free(starts);
                    for (plan_instruction.inputs[2..], 0..) |input_id, i| starts[i] = value_buffers[input_id.index].?;
                    break :blk runtime.Buffer.initDynamicUpdateSlice(allocator, value_buffers[plan_instruction.inputs[0].index].?, value_buffers[plan_instruction.inputs[1].index].?, starts, plan_instruction.dims orelse &.{}, device_index) catch |err| switch (err) {
                        error.UnsupportedElementType => {
                            for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                            return makeError(c.PJRT_Error_Code_UNIMPLEMENTED, "dynamic_update_slice StableHLO execution currently supports matching dense element types");
                        },
                        error.ShapeMismatch => {
                            for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                            return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "dynamic_update_slice StableHLO update or starts are invalid");
                        },
                        else => {
                            for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                            return makeError(c.PJRT_Error_Code_INTERNAL, "failed to execute dynamic_update_slice StableHLO op");
                        },
                    };
                },
                .pad => runtime.Buffer.initPad(allocator, value_buffers[plan_instruction.inputs[0].index].?, value_buffers[plan_instruction.inputs[1].index].?, plan_instruction.edge_padding_low orelse &.{}, plan_instruction.edge_padding_high orelse &.{}, plan_instruction.interior_padding orelse &.{}, plan_instruction.dims orelse &.{}, device_index) catch |err| switch (err) {
                    error.UnsupportedElementType => {
                        for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                        return makeError(c.PJRT_Error_Code_UNIMPLEMENTED, "pad StableHLO execution currently supports scalar padding values with dense buffers");
                    },
                    error.ShapeMismatch => {
                        for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                        return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "pad StableHLO padding attributes are invalid");
                    },
                    else => {
                        for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                        return makeError(c.PJRT_Error_Code_INTERNAL, "failed to execute pad StableHLO op");
                    },
                },
                .reverse => runtime.Buffer.initReverse(allocator, value_buffers[plan_instruction.inputs[0].index].?, plan_instruction.dimensions orelse &.{}, plan_instruction.dims orelse &.{}, device_index) catch |err| switch (err) {
                    error.UnsupportedElementType => {
                        for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                        return makeError(c.PJRT_Error_Code_UNIMPLEMENTED, "reverse StableHLO execution currently supports dense buffers only");
                    },
                    error.ShapeMismatch => {
                        for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                        return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "reverse StableHLO dimensions are invalid");
                    },
                    else => {
                        for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                        return makeError(c.PJRT_Error_Code_INTERNAL, "failed to execute reverse StableHLO op");
                    },
                },
                .concatenate => blk: {
                    const lhs = value_buffers[plan_instruction.inputs[0].index].?;
                    const rhs = value_buffers[plan_instruction.inputs[1].index].?;
                    break :blk runtime.Buffer.initConcatenate(
                        allocator,
                        lhs,
                        rhs,
                        plan_instruction.dimension orelse 0,
                        plan_instruction.dims orelse &.{},
                        device_index,
                    ) catch |err| switch (err) {
                        error.UnsupportedElementType => {
                            for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                            return makeError(c.PJRT_Error_Code_UNIMPLEMENTED, "concatenate StableHLO execution currently supports dense u8 and f32 buffers only");
                        },
                        error.ShapeMismatch => {
                            for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                            return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "concatenate StableHLO input shapes, dimension, or output shape are invalid");
                        },
                        else => {
                            for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                            return makeError(c.PJRT_Error_Code_INTERNAL, "failed to execute concatenate StableHLO op");
                        },
                    };
                },
                .dot_general => runtime.Buffer.initDotGeneral(
                    allocator,
                    value_buffers[plan_instruction.inputs[0].index].?,
                    value_buffers[plan_instruction.inputs[1].index].?,
                    plan_instruction.lhs_batch_dimensions orelse &.{},
                    plan_instruction.rhs_batch_dimensions orelse &.{},
                    plan_instruction.lhs_contracting_dimensions orelse &.{},
                    plan_instruction.rhs_contracting_dimensions orelse &.{},
                    plan_instruction.dims orelse &.{},
                    device_index,
                ) catch |err| switch (err) {
                    error.UnsupportedElementType => {
                        for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                        return makeError(c.PJRT_Error_Code_UNIMPLEMENTED, "dot_general StableHLO execution currently supports f32 matmul-like tensors only");
                    },
                    error.ShapeMismatch => {
                        for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                        return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "dot_general StableHLO dimension numbers or output shape are invalid");
                    },
                    else => {
                        for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                        return makeError(c.PJRT_Error_Code_INTERNAL, "failed to execute dot_general StableHLO op");
                    },
                },
                .reduce_sum, .reduce_max => runtime.Buffer.initReduce(allocator, plan_instruction.kind, value_buffers[plan_instruction.inputs[0].index].?, plan_instruction.reduce_dimensions orelse &.{}, plan_instruction.dims orelse &.{}, device_index) catch |err| switch (err) {
                    error.UnsupportedElementType => {
                        for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                        return makeError(c.PJRT_Error_Code_UNIMPLEMENTED, "reduce StableHLO execution currently supports f32 sum/max only");
                    },
                    error.ShapeMismatch => {
                        for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                        return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "reduce StableHLO dimensions or output shape are invalid");
                    },
                    else => {
                        for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                        return makeError(c.PJRT_Error_Code_INTERNAL, "failed to execute reduce StableHLO op");
                    },
                },
                .gather => runtime.Buffer.initGather(allocator, value_buffers[plan_instruction.inputs[0].index].?, value_buffers[plan_instruction.inputs[1].index].?, plan_instruction.offset_dims orelse &.{}, plan_instruction.collapsed_slice_dims orelse &.{}, plan_instruction.operand_batching_dims orelse &.{}, plan_instruction.start_indices_batching_dims orelse &.{}, plan_instruction.start_index_map orelse &.{}, plan_instruction.index_vector_dim orelse 0, plan_instruction.slice_sizes orelse &.{}, plan_instruction.dims orelse &.{}, device_index) catch |err| switch (err) {
                    error.UnsupportedElementType => {
                        for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                        return makeError(c.PJRT_Error_Code_UNIMPLEMENTED, "gather StableHLO execution currently supports embedding-style collapsed single-axis gathers");
                    },
                    error.ShapeMismatch => {
                        for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                        return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "gather StableHLO dimension numbers or slice sizes are invalid");
                    },
                    else => {
                        for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                        return makeError(c.PJRT_Error_Code_INTERNAL, "failed to execute gather StableHLO op");
                    },
                },
                .sort => runtime.Buffer.initSort(allocator, value_buffers[plan_instruction.inputs[0].index].?, plan_instruction.dimension orelse 0, plan_instruction.dims orelse &.{}, plan_instruction.compare_direction orelse .lt, device_index) catch |err| switch (err) {
                    error.UnsupportedElementType => {
                        for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                        return makeError(c.PJRT_Error_Code_UNIMPLEMENTED, "sort StableHLO execution currently supports numeric dense buffers with lt/le/gt/ge comparator direction");
                    },
                    error.ShapeMismatch => {
                        for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                        return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "sort StableHLO dimension, comparator, or output shape is invalid");
                    },
                    error.UnsupportedRuntimeFeature => {
                        for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                        return makeError(c.PJRT_Error_Code_UNIMPLEMENTED, "sort StableHLO execution requires a supported MLX device path");
                    },
                    else => {
                        for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                        return makeError(c.PJRT_Error_Code_INTERNAL, "failed to execute sort StableHLO op");
                    },
                },
                .compare => runtime.Buffer.initCompare(allocator, value_buffers[plan_instruction.inputs[0].index].?, value_buffers[plan_instruction.inputs[1].index].?, plan_instruction.compare_direction orelse .eq, plan_instruction.dims orelse &.{}, device_index) catch |err| switch (err) {
                    error.UnsupportedElementType => {
                        for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                        return makeError(c.PJRT_Error_Code_UNIMPLEMENTED, "compare StableHLO execution currently supports f32 predicates only");
                    },
                    error.ShapeMismatch => {
                        for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                        return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "compare StableHLO input or output shape is invalid");
                    },
                    else => {
                        for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                        return makeError(c.PJRT_Error_Code_INTERNAL, "failed to execute compare StableHLO op");
                    },
                },
                .select => runtime.Buffer.initSelect(allocator, value_buffers[plan_instruction.inputs[0].index].?, value_buffers[plan_instruction.inputs[1].index].?, value_buffers[plan_instruction.inputs[2].index].?, plan_instruction.dims orelse &.{}, device_index) catch |err| switch (err) {
                    error.UnsupportedElementType => {
                        for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                        return makeError(c.PJRT_Error_Code_UNIMPLEMENTED, "select StableHLO execution currently supports pred with matching dense data buffers");
                    },
                    error.ShapeMismatch => {
                        for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                        return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "select StableHLO input or output shape is invalid");
                    },
                    else => {
                        for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                        return makeError(c.PJRT_Error_Code_INTERNAL, "failed to execute select StableHLO op");
                    },
                },
                .clamp => runtime.Buffer.initClamp(allocator, value_buffers[plan_instruction.inputs[0].index].?, value_buffers[plan_instruction.inputs[1].index].?, value_buffers[plan_instruction.inputs[2].index].?, plan_instruction.dims orelse &.{}, device_index) catch |err| switch (err) {
                    error.UnsupportedElementType => {
                        for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                        return makeError(c.PJRT_Error_Code_UNIMPLEMENTED, "clamp StableHLO execution currently supports matching dense numeric buffers");
                    },
                    error.ShapeMismatch => {
                        for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                        return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "clamp StableHLO input or output shape is invalid");
                    },
                    else => {
                        for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                        return makeError(c.PJRT_Error_Code_INTERNAL, "failed to execute clamp StableHLO op");
                    },
                },
                .rng_bit_generator => unreachable,
                .complex, .real, .imag, .fft, .convolution, .custom_call, .get_tuple_element, .scatter, .triangular_solve, .tuple, .while_ => {
                    destroyExecuteValues(value_buffers, value_owned);
                    return unsupportedRuntimeFeature(plan_instruction.kind);
                },
                .unsupported => {
                    for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                    return makeError(c.PJRT_Error_Code_UNIMPLEMENTED, "executable plan contains an unsupported runtime operation");
                },
            };
            value_buffers[output_id.index] = next;
            value_owned[output_id.index] = true;
        }
        for (executable.plan.output_ids, 0..) |output_id, output_index| {
            const output_buffer = value_buffers[output_id.index] orelse {
                for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                return makeError(c.PJRT_Error_Code_INTERNAL, "executable output value was not produced");
            };
            if (value_owned[output_id.index]) {
                args[0].output_lists[device_index][output_index] = @ptrCast(output_buffer);
                value_owned[output_id.index] = false;
            } else {
                const copied = runtime.Buffer.initDeviceCopy(allocator, output_buffer, device_index) catch {
                    for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
                    return makeError(c.PJRT_Error_Code_INTERNAL, "failed to copy argument output");
                };
                args[0].output_lists[device_index][output_index] = @ptrCast(copied);
            }
        }
        for (value_buffers, value_owned) |maybe_buffer, owned| if (maybe_buffer) |buffer| destroyOwnedBuffer(buffer, owned);
        if (args[0].device_complete_events) |events| events[device_index] = eventCreateReady();
    }
    return null;
}

fn executableName(args: [*c]c.PJRT_Executable_Name_Args) callconv(.c) ?*c.PJRT_Error {
    const executable: *Executable = @ptrCast(@alignCast(args[0].executable.?));
    args[0].executable_name = executable.name.ptr;
    args[0].executable_name_size = executable.name.len;
    return null;
}

fn executableDestroy(args: [*c]c.PJRT_Executable_Destroy_Args) callconv(.c) ?*c.PJRT_Error {
    _ = args;
    return null;
}

fn executableNumReplicas(args: [*c]c.PJRT_Executable_NumReplicas_Args) callconv(.c) ?*c.PJRT_Error {
    const executable: *Executable = @ptrCast(@alignCast(args[0].executable.?));
    args[0].num_replicas = @intCast(executable.plan.options.num_replicas);
    return null;
}

fn executableNumPartitions(args: [*c]c.PJRT_Executable_NumPartitions_Args) callconv(.c) ?*c.PJRT_Error {
    const executable: *Executable = @ptrCast(@alignCast(args[0].executable.?));
    args[0].num_partitions = @intCast(executable.plan.options.num_partitions);
    return null;
}

fn executableNumOutputs(args: [*c]c.PJRT_Executable_NumOutputs_Args) callconv(.c) ?*c.PJRT_Error {
    const executable: *Executable = @ptrCast(@alignCast(args[0].executable.?));
    args[0].num_outputs = executable.plan.output_shardings.len;
    return null;
}

fn executableOptimizedProgram(args: [*c]c.PJRT_Executable_OptimizedProgram_Args) callconv(.c) ?*c.PJRT_Error {
    const executable: *Executable = @ptrCast(@alignCast(args[0].executable.?));
    const program = args[0].program;
    program[0].format = "mlir";
    program[0].format_size = "mlir".len;
    program[0].code_size = executable.optimized_program.len;
    if (program[0].code) |code| {
        @memcpy(code[0..executable.optimized_program.len], executable.optimized_program);
    }
    return null;
}

fn executableFingerprint(args: [*c]c.PJRT_Executable_Fingerprint_Args) callconv(.c) ?*c.PJRT_Error {
    const executable: *Executable = @ptrCast(@alignCast(args[0].executable.?));
    args[0].executable_fingerprint = executable.fingerprint.ptr;
    args[0].executable_fingerprint_size = executable.fingerprint.len;
    return null;
}

fn executableParameterMemoryKinds(args: [*c]c.PJRT_Executable_ParameterMemoryKinds_Args) callconv(.c) ?*c.PJRT_Error {
    const executable: *Executable = @ptrCast(@alignCast(args[0].executable.?));
    args[0].num_parameters = executable.parameter_memory_kinds.len;
    args[0].memory_kinds = @ptrCast(executable.parameter_memory_kinds.ptr);
    args[0].memory_kind_sizes = executable.parameter_memory_kind_sizes.ptr;
    return null;
}

fn executableOutputMemoryKinds(args: [*c]c.PJRT_Executable_OutputMemoryKinds_Args) callconv(.c) ?*c.PJRT_Error {
    const executable: *Executable = @ptrCast(@alignCast(args[0].executable.?));
    args[0].num_outputs = executable.output_memory_kinds.len;
    args[0].memory_kinds = @ptrCast(executable.output_memory_kinds.ptr);
    args[0].memory_kind_sizes = executable.output_memory_kind_sizes.ptr;
    return null;
}

fn loadedExecutableFingerprint(args: [*c]c.PJRT_LoadedExecutable_Fingerprint_Args) callconv(.c) ?*c.PJRT_Error {
    const executable = executableFromC(args[0].executable);
    args[0].executable_fingerprint = executable.fingerprint.ptr;
    args[0].executable_fingerprint_size = executable.fingerprint.len;
    return null;
}

fn bufferDestroy(args: [*c]c.PJRT_Buffer_Destroy_Args) callconv(.c) ?*c.PJRT_Error {
    if (args[0].buffer) |buffer| bufferFromC(buffer).deinit();
    return null;
}

fn bufferElementType(args: [*c]c.PJRT_Buffer_ElementType_Args) callconv(.c) ?*c.PJRT_Error {
    args[0].type = pjrtTypeFromRuntime(bufferFromC(args[0].buffer).element_type);
    return null;
}

fn bufferDimensions(args: [*c]c.PJRT_Buffer_Dimensions_Args) callconv(.c) ?*c.PJRT_Error {
    const buffer = bufferFromC(args[0].buffer);
    args[0].dims = buffer.dims.ptr;
    args[0].num_dims = buffer.dims.len;
    return null;
}

fn bufferOnDeviceSize(args: [*c]c.PJRT_Buffer_OnDeviceSizeInBytes_Args) callconv(.c) ?*c.PJRT_Error {
    args[0].on_device_size_in_bytes = bufferFromC(args[0].buffer).byte_size;
    return null;
}

fn bufferDelete(args: [*c]c.PJRT_Buffer_Delete_Args) callconv(.c) ?*c.PJRT_Error {
    bufferFromC(args[0].buffer).deleted = true;
    return null;
}

fn bufferIsDeleted(args: [*c]c.PJRT_Buffer_IsDeleted_Args) callconv(.c) ?*c.PJRT_Error {
    args[0].is_deleted = bufferFromC(args[0].buffer).deleted;
    return null;
}

fn bufferIsOnCpu(args: [*c]c.PJRT_Buffer_IsOnCpu_Args) callconv(.c) ?*c.PJRT_Error {
    _ = bufferFromC(args[0].buffer);
    args[0].is_on_cpu = false;
    return null;
}

fn bufferDynamicDimensionIndices(args: [*c]c.PJRT_Buffer_DynamicDimensionIndices_Args) callconv(.c) ?*c.PJRT_Error {
    _ = bufferFromC(args[0].buffer);
    args[0].dynamic_dim_indices = null;
    args[0].num_dynamic_dims = 0;
    return null;
}

fn bufferToHost(args: [*c]c.PJRT_Buffer_ToHostBuffer_Args) callconv(.c) ?*c.PJRT_Error {
    const trace_start_ns = nowNs();
    const buffer = bufferFromC(args[0].src);
    if (args[0].dst == null) {
        args[0].dst_size = buffer.byte_size;
        return null;
    }
    if (args[0].dst_size < buffer.byte_size) return makeError(c.PJRT_Error_Code_INVALID_ARGUMENT, "destination buffer is too small");
    buffer.copyToHost(@as([*]u8, @ptrCast(args[0].dst))[0..buffer.byte_size]) catch {
        return makeError(c.PJRT_Error_Code_INTERNAL, "failed to copy buffer to host");
    };
    args[0].event = eventCreateReady();
    trace(
        "event=d2h bytes={d} dtype={s} rank={d} device={d} backend_storage={d} elapsed_us={d}",
        .{
            buffer.byte_size,
            @tagName(buffer.element_type),
            buffer.dims.len,
            buffer.device_id,
            @intFromBool(buffer.hasBackendStorage()),
            elapsedUs(trace_start_ns),
        },
    );
    return null;
}

fn bufferReadyEvent(args: [*c]c.PJRT_Buffer_ReadyEvent_Args) callconv(.c) ?*c.PJRT_Error {
    args[0].event = eventCreateReady();
    return null;
}

fn bufferDevice(args: [*c]c.PJRT_Buffer_Device_Args) callconv(.c) ?*c.PJRT_Error {
    const buffer = bufferFromC(args[0].buffer);
    args[0].device = @ptrCast(buffer.device);
    return null;
}

fn bufferMemory(args: [*c]c.PJRT_Buffer_Memory_Args) callconv(.c) ?*c.PJRT_Error {
    const buffer = bufferFromC(args[0].buffer);
    args[0].memory = @ptrCast(buffer.memory);
    return null;
}

fn initApi() void {
    if (api_ready) return;
    api_storage = std.mem.zeroes(c.PJRT_Api);
    api_storage.struct_size = c.PJRT_Api_STRUCT_SIZE;
    api_storage.extension_start = null;
    api_storage.pjrt_api_version = .{
        .struct_size = c.PJRT_Api_Version_STRUCT_SIZE,
        .extension_start = null,
        .major_version = c.PJRT_API_MAJOR,
        .minor_version = c.PJRT_API_MINOR,
    };

    api_storage.PJRT_Error_Destroy = pjrtErrorDestroy;
    api_storage.PJRT_Error_Message = pjrtErrorMessage;
    api_storage.PJRT_Error_GetCode = pjrtErrorGetCode;
    api_storage.PJRT_Error_ForEachPayload = pjrtErrorForEachPayload;
    api_storage.PJRT_Plugin_Initialize = pjrtPluginInitialize;
    api_storage.PJRT_Plugin_Attributes = pjrtPluginAttributes;
    api_storage.PJRT_Event_Destroy = pjrtEventDestroy;
    api_storage.PJRT_Event_IsReady = pjrtEventIsReady;
    api_storage.PJRT_Event_Error = pjrtEventError;
    api_storage.PJRT_Event_Await = pjrtEventAwait;
    api_storage.PJRT_Event_OnReady = pjrtEventOnReady;
    api_storage.PJRT_Client_Create = pjrtClientCreate;
    api_storage.PJRT_Client_Destroy = pjrtClientDestroy;
    api_storage.PJRT_Client_PlatformName = pjrtClientPlatformName;
    api_storage.PJRT_Client_ProcessIndex = pjrtClientProcessIndex;
    api_storage.PJRT_Client_PlatformVersion = pjrtClientPlatformVersion;
    api_storage.PJRT_Client_TopologyDescription = pjrtClientTopologyDescription;
    api_storage.PJRT_Client_Devices = pjrtClientDevices;
    api_storage.PJRT_Client_AddressableDevices = pjrtClientAddressableDevices;
    api_storage.PJRT_Client_LookupDevice = pjrtClientLookupDevice;
    api_storage.PJRT_Client_LookupAddressableDevice = pjrtClientLookupAddressableDevice;
    api_storage.PJRT_Client_AddressableMemories = pjrtClientAddressableMemories;
    api_storage.PJRT_Client_Compile = pjrtClientCompile;
    api_storage.PJRT_Client_DefaultDeviceAssignment = pjrtClientDefaultDeviceAssignment;
    api_storage.PJRT_Client_BufferFromHostBuffer = pjrtClientBufferFromHostBuffer;
    api_storage.PJRT_TopologyDescription_Create = topologyDescriptionCreate;
    api_storage.PJRT_TopologyDescription_Destroy = topologyDescriptionDestroy;
    api_storage.PJRT_TopologyDescription_PlatformName = topologyDescriptionPlatformName;
    api_storage.PJRT_TopologyDescription_PlatformVersion = topologyDescriptionPlatformVersion;
    api_storage.PJRT_TopologyDescription_GetDeviceDescriptions = topologyDescriptionGetDeviceDescriptions;
    api_storage.PJRT_TopologyDescription_Serialize = topologyDescriptionSerialize;
    api_storage.PJRT_TopologyDescription_Deserialize = topologyDescriptionDeserialize;
    api_storage.PJRT_TopologyDescription_Attributes = topologyDescriptionAttributes;
    api_storage.PJRT_TopologyDescription_Fingerprint = topologyDescriptionFingerprint;
    api_storage.PJRT_DeviceDescription_Id = deviceDescriptionId;
    api_storage.PJRT_DeviceDescription_ProcessIndex = deviceDescriptionProcessIndex;
    api_storage.PJRT_DeviceDescription_Attributes = deviceDescriptionAttributes;
    api_storage.PJRT_DeviceDescription_Kind = deviceDescriptionKind;
    api_storage.PJRT_DeviceDescription_DebugString = deviceDescriptionDebugString;
    api_storage.PJRT_DeviceDescription_ToString = deviceDescriptionToString;
    api_storage.PJRT_Device_GetDescription = deviceGetDescription;
    api_storage.PJRT_Device_IsAddressable = deviceIsAddressable;
    api_storage.PJRT_Device_LocalHardwareId = deviceLocalHardwareId;
    api_storage.PJRT_Device_AddressableMemories = deviceAddressableMemories;
    api_storage.PJRT_Device_DefaultMemory = deviceDefaultMemory;
    api_storage.PJRT_Device_MemoryStats = deviceMemoryStats;
    api_storage.PJRT_Device_GetAttributes = deviceGetAttributes;
    api_storage.PJRT_Memory_Id = memoryId;
    api_storage.PJRT_Memory_Kind = memoryKind;
    api_storage.PJRT_Memory_DebugString = memoryDebugString;
    api_storage.PJRT_Memory_ToString = memoryToString;
    api_storage.PJRT_Memory_AddressableByDevices = memoryAddressableByDevices;
    api_storage.PJRT_Executable_Destroy = executableDestroy;
    api_storage.PJRT_Executable_Name = executableName;
    api_storage.PJRT_Executable_NumReplicas = executableNumReplicas;
    api_storage.PJRT_Executable_NumPartitions = executableNumPartitions;
    api_storage.PJRT_Executable_NumOutputs = executableNumOutputs;
    api_storage.PJRT_Executable_OptimizedProgram = executableOptimizedProgram;
    api_storage.PJRT_Executable_Fingerprint = executableFingerprint;
    api_storage.PJRT_Executable_ParameterMemoryKinds = executableParameterMemoryKinds;
    api_storage.PJRT_Executable_OutputMemoryKinds = executableOutputMemoryKinds;
    api_storage.PJRT_LoadedExecutable_Destroy = loadedExecutableDestroy;
    api_storage.PJRT_LoadedExecutable_GetExecutable = loadedExecutableGetExecutable;
    api_storage.PJRT_LoadedExecutable_AddressableDevices = loadedExecutableAddressableDevices;
    api_storage.PJRT_LoadedExecutable_AddressableDeviceLogicalIds = loadedExecutableAddressableDeviceLogicalIds;
    api_storage.PJRT_LoadedExecutable_GetDeviceAssignment = loadedExecutableGetDeviceAssignment;
    api_storage.PJRT_LoadedExecutable_Fingerprint = loadedExecutableFingerprint;
    api_storage.PJRT_LoadedExecutable_Delete = loadedExecutableDelete;
    api_storage.PJRT_LoadedExecutable_IsDeleted = loadedExecutableIsDeleted;
    api_storage.PJRT_LoadedExecutable_Execute = loadedExecutableExecute;
    api_storage.PJRT_Buffer_Destroy = bufferDestroy;
    api_storage.PJRT_Buffer_ElementType = bufferElementType;
    api_storage.PJRT_Buffer_Dimensions = bufferDimensions;
    api_storage.PJRT_Buffer_OnDeviceSizeInBytes = bufferOnDeviceSize;
    api_storage.PJRT_Buffer_Device = bufferDevice;
    api_storage.PJRT_Buffer_Memory = bufferMemory;
    api_storage.PJRT_Buffer_DynamicDimensionIndices = bufferDynamicDimensionIndices;
    api_storage.PJRT_Buffer_Delete = bufferDelete;
    api_storage.PJRT_Buffer_IsDeleted = bufferIsDeleted;
    api_storage.PJRT_Buffer_IsOnCpu = bufferIsOnCpu;
    api_storage.PJRT_Buffer_ToHostBuffer = bufferToHost;
    api_storage.PJRT_Buffer_ReadyEvent = bufferReadyEvent;

    api_ready = true;
}

pub export fn GetPjrtApi() *const c.PJRT_Api {
    initApi();
    return &api_storage;
}

test "api table exposes bootstrap PJRT surface" {
    const api = GetPjrtApi();
    try std.testing.expectEqual(@as(usize, c.PJRT_Api_STRUCT_SIZE), api.struct_size);
    try std.testing.expect(api.PJRT_Plugin_Initialize != null);
    try std.testing.expect(api.PJRT_Client_Create != null);
    try std.testing.expect(api.PJRT_LoadedExecutable_Execute != null);
    try std.testing.expect(api.PJRT_Buffer_ToHostBuffer != null);
}

test "plugin attributes and client create expose backend selection" {
    const api = GetPjrtApi();

    var attrs_args = std.mem.zeroes(c.PJRT_Plugin_Attributes_Args);
    attrs_args.struct_size = c.PJRT_Plugin_Attributes_Args_STRUCT_SIZE;
    try expectOk(api.PJRT_Plugin_Attributes.?(&attrs_args));
    try std.testing.expectEqual(@as(usize, 5), attrs_args.num_attributes);
    try std.testing.expectEqualStrings("pjrtx_default_backend", attrs_args.attributes[4].name[0..attrs_args.attributes[4].name_size]);
    try std.testing.expectEqualStrings("metal_mlx", attrs_args.attributes[4].unnamed_0.string_value[0..attrs_args.attributes[4].value_size]);

    var option = std.mem.zeroes(c.PJRT_NamedValue);
    option.struct_size = c.PJRT_NamedValue_STRUCT_SIZE;
    option.name = backend_option;
    option.name_size = backend_option.len;
    option.type = c.PJRT_NamedValue_kString;
    option.unnamed_0.string_value = "metal_mlx";
    option.value_size = "metal_mlx".len;

    var create_args = std.mem.zeroes(c.PJRT_Client_Create_Args);
    create_args.struct_size = c.PJRT_Client_Create_Args_STRUCT_SIZE;
    create_args.create_options = &option;
    create_args.num_options = 1;
    try expectOk(api.PJRT_Client_Create.?(&create_args));
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_Client_Destroy_Args);
        destroy_args.struct_size = c.PJRT_Client_Destroy_Args_STRUCT_SIZE;
        destroy_args.client = create_args.client;
        _ = api.PJRT_Client_Destroy.?(&destroy_args);
    }

    try std.testing.expectEqual(runtime.BackendKind.metal_mlx, clientFromC(create_args.client).backend_kind);
}

fn expectOk(err: [*c]c.PJRT_Error) !void {
    if (err) |actual| {
        const api = GetPjrtApi();
        std.debug.print("unexpected PJRT error: {s}\n", .{errorMessage(api, actual)});
        destroyError(api, actual);
        return error.TestUnexpectedResult;
    }
}

fn destroyError(api: *const c.PJRT_Api, err: [*c]c.PJRT_Error) void {
    var destroy_args = std.mem.zeroes(c.PJRT_Error_Destroy_Args);
    destroy_args.struct_size = c.PJRT_Error_Destroy_Args_STRUCT_SIZE;
    destroy_args.@"error" = err;
    api.PJRT_Error_Destroy.?(&destroy_args);
}

fn errorMessage(api: *const c.PJRT_Api, err: [*c]c.PJRT_Error) []const u8 {
    var message_args = std.mem.zeroes(c.PJRT_Error_Message_Args);
    message_args.struct_size = c.PJRT_Error_Message_Args_STRUCT_SIZE;
    message_args.@"error" = err;
    api.PJRT_Error_Message.?(&message_args);
    return message_args.message[0..message_args.message_size];
}

test "client device memory and buffer ownership callbacks return stable handles" {
    const api = GetPjrtApi();

    var option = std.mem.zeroes(c.PJRT_NamedValue);
    option.struct_size = c.PJRT_NamedValue_STRUCT_SIZE;
    option.name = backend_option;
    option.name_size = backend_option.len;
    option.type = c.PJRT_NamedValue_kString;
    option.unnamed_0.string_value = "metal_mlx";
    option.value_size = "metal_mlx".len;

    var create_args = std.mem.zeroes(c.PJRT_Client_Create_Args);
    create_args.struct_size = c.PJRT_Client_Create_Args_STRUCT_SIZE;
    create_args.create_options = &option;
    create_args.num_options = 1;
    try expectOk(api.PJRT_Client_Create.?(&create_args));
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_Client_Destroy_Args);
        destroy_args.struct_size = c.PJRT_Client_Destroy_Args_STRUCT_SIZE;
        destroy_args.client = create_args.client;
        _ = api.PJRT_Client_Destroy.?(&destroy_args);
    }

    var devices_args = std.mem.zeroes(c.PJRT_Client_Devices_Args);
    devices_args.struct_size = c.PJRT_Client_Devices_Args_STRUCT_SIZE;
    devices_args.client = create_args.client;
    try expectOk(api.PJRT_Client_Devices.?(&devices_args));
    try std.testing.expectEqual(@as(usize, 1), devices_args.num_devices);
    try std.testing.expect(devices_args.devices[0] != null);
    try std.testing.expect(api.PJRT_Device_GetAttributes != null);

    var device_attrs_args = std.mem.zeroes(c.PJRT_Device_GetAttributes_Args);
    device_attrs_args.struct_size = c.PJRT_Device_GetAttributes_Args_STRUCT_SIZE;
    device_attrs_args.device = devices_args.devices[0];
    try expectOk(api.PJRT_Device_GetAttributes.?(&device_attrs_args));
    defer if (device_attrs_args.attributes_deleter) |deleter| deleter(device_attrs_args.device_attributes);
    try std.testing.expectEqual(@as(usize, 5), device_attrs_args.num_attributes);
    try std.testing.expectEqualStrings("device_name", device_attrs_args.attributes[0].name[0..device_attrs_args.attributes[0].name_size]);
    try std.testing.expectEqual(@as(@TypeOf(device_attrs_args.attributes[0].type), c.PJRT_NamedValue_kString), device_attrs_args.attributes[0].type);
    try std.testing.expect(device_attrs_args.attributes[0].value_size != 0);
    try std.testing.expectEqualStrings("pjrtx_default_memory_id", device_attrs_args.attributes[4].name[0..device_attrs_args.attributes[4].name_size]);
    try std.testing.expectEqual(@as(i64, 0), device_attrs_args.attributes[4].unnamed_0.int64_value);

    var memories_args = std.mem.zeroes(c.PJRT_Client_AddressableMemories_Args);
    memories_args.struct_size = c.PJRT_Client_AddressableMemories_Args_STRUCT_SIZE;
    memories_args.client = create_args.client;
    try expectOk(api.PJRT_Client_AddressableMemories.?(&memories_args));
    try std.testing.expectEqual(@as(usize, 1), memories_args.num_addressable_memories);
    try std.testing.expect(memories_args.addressable_memories[0] != null);

    var default_memory_args = std.mem.zeroes(c.PJRT_Device_DefaultMemory_Args);
    default_memory_args.struct_size = c.PJRT_Device_DefaultMemory_Args_STRUCT_SIZE;
    default_memory_args.device = devices_args.devices[0];
    try expectOk(api.PJRT_Device_DefaultMemory.?(&default_memory_args));
    try std.testing.expectEqual(memories_args.addressable_memories[0], default_memory_args.memory);

    var device_memories_args = std.mem.zeroes(c.PJRT_Device_AddressableMemories_Args);
    device_memories_args.struct_size = c.PJRT_Device_AddressableMemories_Args_STRUCT_SIZE;
    device_memories_args.device = devices_args.devices[0];
    try expectOk(api.PJRT_Device_AddressableMemories.?(&device_memories_args));
    try std.testing.expectEqual(@as(usize, 1), device_memories_args.num_memories);
    try std.testing.expectEqual(default_memory_args.memory, device_memories_args.memories[0]);

    var memory_devices_args = std.mem.zeroes(c.PJRT_Memory_AddressableByDevices_Args);
    memory_devices_args.struct_size = c.PJRT_Memory_AddressableByDevices_Args_STRUCT_SIZE;
    memory_devices_args.memory = default_memory_args.memory;
    try expectOk(api.PJRT_Memory_AddressableByDevices.?(&memory_devices_args));
    try std.testing.expectEqual(@as(usize, 1), memory_devices_args.num_devices);
    try std.testing.expectEqual(devices_args.devices[0], memory_devices_args.devices[0]);

    const dims = [_]i64{4};
    const input = [_]u8{ 1, 2, 3, 4 };
    var from_host_args = std.mem.zeroes(c.PJRT_Client_BufferFromHostBuffer_Args);
    from_host_args.struct_size = c.PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    from_host_args.client = create_args.client;
    from_host_args.data = &input;
    from_host_args.type = c.PJRT_Buffer_Type_U8;
    from_host_args.dims = &dims;
    from_host_args.num_dims = dims.len;
    from_host_args.device = devices_args.devices[0];
    from_host_args.memory = default_memory_args.memory;
    try expectOk(api.PJRT_Client_BufferFromHostBuffer.?(&from_host_args));
    defer {
        if (from_host_args.done_with_host_buffer) |event| {
            var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
            event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
            event_destroy_args.event = event;
            _ = api.PJRT_Event_Destroy.?(&event_destroy_args);
        }
        var buffer_destroy_args = std.mem.zeroes(c.PJRT_Buffer_Destroy_Args);
        buffer_destroy_args.struct_size = c.PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        buffer_destroy_args.buffer = from_host_args.buffer;
        _ = api.PJRT_Buffer_Destroy.?(&buffer_destroy_args);
    }

    var buffer_device_args = std.mem.zeroes(c.PJRT_Buffer_Device_Args);
    buffer_device_args.struct_size = c.PJRT_Buffer_Device_Args_STRUCT_SIZE;
    buffer_device_args.buffer = from_host_args.buffer;
    try expectOk(api.PJRT_Buffer_Device.?(&buffer_device_args));
    try std.testing.expectEqual(devices_args.devices[0], buffer_device_args.device);

    var buffer_memory_args = std.mem.zeroes(c.PJRT_Buffer_Memory_Args);
    buffer_memory_args.struct_size = c.PJRT_Buffer_Memory_Args_STRUCT_SIZE;
    buffer_memory_args.buffer = from_host_args.buffer;
    try expectOk(api.PJRT_Buffer_Memory.?(&buffer_memory_args));
    try std.testing.expectEqual(default_memory_args.memory, buffer_memory_args.memory);

    var output: [4]u8 = undefined;
    var to_host_args = std.mem.zeroes(c.PJRT_Buffer_ToHostBuffer_Args);
    to_host_args.struct_size = c.PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    to_host_args.src = from_host_args.buffer;
    to_host_args.dst = &output;
    to_host_args.dst_size = output.len;
    try expectOk(api.PJRT_Buffer_ToHostBuffer.?(&to_host_args));
    defer if (to_host_args.event) |event| {
        var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
        event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
        event_destroy_args.event = event;
        _ = api.PJRT_Event_Destroy.?(&event_destroy_args);
    };
    try std.testing.expectEqualSlices(u8, &input, &output);
}

test "loaded executable execute chains u8 StableHLO ops through PJRT argument lists" {
    const api = GetPjrtApi();

    var option = std.mem.zeroes(c.PJRT_NamedValue);
    option.struct_size = c.PJRT_NamedValue_STRUCT_SIZE;
    option.name = backend_option;
    option.name_size = backend_option.len;
    option.type = c.PJRT_NamedValue_kString;
    option.unnamed_0.string_value = "metal_mlx";
    option.value_size = "metal_mlx".len;

    var create_args = std.mem.zeroes(c.PJRT_Client_Create_Args);
    create_args.struct_size = c.PJRT_Client_Create_Args_STRUCT_SIZE;
    create_args.create_options = &option;
    create_args.num_options = 1;
    try expectOk(api.PJRT_Client_Create.?(&create_args));
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_Client_Destroy_Args);
        destroy_args.struct_size = c.PJRT_Client_Destroy_Args_STRUCT_SIZE;
        destroy_args.client = create_args.client;
        _ = api.PJRT_Client_Destroy.?(&destroy_args);
    }

    var devices_args = std.mem.zeroes(c.PJRT_Client_Devices_Args);
    devices_args.struct_size = c.PJRT_Client_Devices_Args_STRUCT_SIZE;
    devices_args.client = create_args.client;
    try expectOk(api.PJRT_Client_Devices.?(&devices_args));
    try std.testing.expectEqual(@as(usize, 1), devices_args.num_devices);

    var default_memory_args = std.mem.zeroes(c.PJRT_Device_DefaultMemory_Args);
    default_memory_args.struct_size = c.PJRT_Device_DefaultMemory_Args_STRUCT_SIZE;
    default_memory_args.device = devices_args.devices[0];
    try expectOk(api.PJRT_Device_DefaultMemory.?(&default_memory_args));

    const dims = [_]i64{4};
    const lhs = [_]u8{ 1, 2, 3, 4 };
    const rhs = [_]u8{ 10, 20, 30, 40 };

    var lhs_from_host_args = std.mem.zeroes(c.PJRT_Client_BufferFromHostBuffer_Args);
    lhs_from_host_args.struct_size = c.PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    lhs_from_host_args.client = create_args.client;
    lhs_from_host_args.data = &lhs;
    lhs_from_host_args.type = c.PJRT_Buffer_Type_S8;
    lhs_from_host_args.dims = &dims;
    lhs_from_host_args.num_dims = dims.len;
    lhs_from_host_args.device = devices_args.devices[0];
    lhs_from_host_args.memory = default_memory_args.memory;
    try expectOk(api.PJRT_Client_BufferFromHostBuffer.?(&lhs_from_host_args));
    defer {
        if (lhs_from_host_args.done_with_host_buffer) |event| {
            var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
            event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
            event_destroy_args.event = event;
            _ = api.PJRT_Event_Destroy.?(&event_destroy_args);
        }
        var buffer_destroy_args = std.mem.zeroes(c.PJRT_Buffer_Destroy_Args);
        buffer_destroy_args.struct_size = c.PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        buffer_destroy_args.buffer = lhs_from_host_args.buffer;
        _ = api.PJRT_Buffer_Destroy.?(&buffer_destroy_args);
    }

    var rhs_from_host_args = std.mem.zeroes(c.PJRT_Client_BufferFromHostBuffer_Args);
    rhs_from_host_args.struct_size = c.PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    rhs_from_host_args.client = create_args.client;
    rhs_from_host_args.data = &rhs;
    rhs_from_host_args.type = c.PJRT_Buffer_Type_S8;
    rhs_from_host_args.dims = &dims;
    rhs_from_host_args.num_dims = dims.len;
    rhs_from_host_args.device = devices_args.devices[0];
    rhs_from_host_args.memory = default_memory_args.memory;
    try expectOk(api.PJRT_Client_BufferFromHostBuffer.?(&rhs_from_host_args));
    defer {
        if (rhs_from_host_args.done_with_host_buffer) |event| {
            var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
            event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
            event_destroy_args.event = event;
            _ = api.PJRT_Event_Destroy.?(&event_destroy_args);
        }
        var buffer_destroy_args = std.mem.zeroes(c.PJRT_Buffer_Destroy_Args);
        buffer_destroy_args.struct_size = c.PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        buffer_destroy_args.buffer = rhs_from_host_args.buffer;
        _ = api.PJRT_Buffer_Destroy.?(&buffer_destroy_args);
    }

    const module_text =
        \\module {
        \\  func.func @main(%arg0: tensor<4xi8>, %arg1: tensor<4xi8>) -> tensor<4xi8> {
        \\    %0 = stablehlo.subtract %arg0, %arg1 : tensor<4xi8>
        \\    %1 = stablehlo.negate %0 : tensor<4xi8>
        \\    return %1 : tensor<4xi8>
        \\  }
        \\}
    ;
    var program = std.mem.zeroes(c.PJRT_Program);
    program.struct_size = c.PJRT_Program_STRUCT_SIZE;
    program.code = @constCast(module_text.ptr);
    program.code_size = module_text.len;
    program.format = "mlir";
    program.format_size = "mlir".len;

    var compile_args = std.mem.zeroes(c.PJRT_Client_Compile_Args);
    compile_args.struct_size = c.PJRT_Client_Compile_Args_STRUCT_SIZE;
    compile_args.client = create_args.client;
    compile_args.program = &program;
    try expectOk(api.PJRT_Client_Compile.?(&compile_args));
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Destroy_Args);
        destroy_args.struct_size = c.PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
        destroy_args.executable = compile_args.executable;
        _ = api.PJRT_LoadedExecutable_Destroy.?(&destroy_args);
    }

    var device_args = [_]?*c.PJRT_Buffer{ lhs_from_host_args.buffer, rhs_from_host_args.buffer };
    var argument_lists = [_][*c]const ?*c.PJRT_Buffer{&device_args};
    var device_outputs = [_]?*c.PJRT_Buffer{null};
    var output_lists = [_][*c]?*c.PJRT_Buffer{&device_outputs};
    var device_events = [_]?*c.PJRT_Event{null};

    var execute_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Execute_Args);
    execute_args.struct_size = c.PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    execute_args.executable = compile_args.executable;
    execute_args.argument_lists = &argument_lists;
    execute_args.num_devices = 1;
    execute_args.num_args = device_args.len;
    execute_args.output_lists = &output_lists;
    execute_args.device_complete_events = &device_events;
    try expectOk(api.PJRT_LoadedExecutable_Execute.?(&execute_args));
    defer if (device_events[0]) |event| {
        var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
        event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
        event_destroy_args.event = event;
        _ = api.PJRT_Event_Destroy.?(&event_destroy_args);
    };
    defer if (device_outputs[0]) |buffer| {
        var buffer_destroy_args = std.mem.zeroes(c.PJRT_Buffer_Destroy_Args);
        buffer_destroy_args.struct_size = c.PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        buffer_destroy_args.buffer = buffer;
        _ = api.PJRT_Buffer_Destroy.?(&buffer_destroy_args);
    };

    try std.testing.expect(device_outputs[0] != null);
    var output: [4]u8 = undefined;
    var to_host_args = std.mem.zeroes(c.PJRT_Buffer_ToHostBuffer_Args);
    to_host_args.struct_size = c.PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    to_host_args.src = device_outputs[0];
    to_host_args.dst = &output;
    to_host_args.dst_size = output.len;
    try expectOk(api.PJRT_Buffer_ToHostBuffer.?(&to_host_args));
    defer if (to_host_args.event) |event| {
        var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
        event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
        event_destroy_args.event = event;
        _ = api.PJRT_Event_Destroy.?(&event_destroy_args);
    };
    try std.testing.expectEqualSlices(u8, &.{ 9, 18, 27, 36 }, &output);
}

test "loaded executable execute chains f32 binary and unary buffers through PJRT argument lists" {
    const api = GetPjrtApi();

    var option = std.mem.zeroes(c.PJRT_NamedValue);
    option.struct_size = c.PJRT_NamedValue_STRUCT_SIZE;
    option.name = backend_option;
    option.name_size = backend_option.len;
    option.type = c.PJRT_NamedValue_kString;
    option.unnamed_0.string_value = "metal_mlx";
    option.value_size = "metal_mlx".len;

    var create_args = std.mem.zeroes(c.PJRT_Client_Create_Args);
    create_args.struct_size = c.PJRT_Client_Create_Args_STRUCT_SIZE;
    create_args.create_options = &option;
    create_args.num_options = 1;
    try expectOk(api.PJRT_Client_Create.?(&create_args));
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_Client_Destroy_Args);
        destroy_args.struct_size = c.PJRT_Client_Destroy_Args_STRUCT_SIZE;
        destroy_args.client = create_args.client;
        _ = api.PJRT_Client_Destroy.?(&destroy_args);
    }

    var devices_args = std.mem.zeroes(c.PJRT_Client_Devices_Args);
    devices_args.struct_size = c.PJRT_Client_Devices_Args_STRUCT_SIZE;
    devices_args.client = create_args.client;
    try expectOk(api.PJRT_Client_Devices.?(&devices_args));

    var default_memory_args = std.mem.zeroes(c.PJRT_Device_DefaultMemory_Args);
    default_memory_args.struct_size = c.PJRT_Device_DefaultMemory_Args_STRUCT_SIZE;
    default_memory_args.device = devices_args.devices[0];
    try expectOk(api.PJRT_Device_DefaultMemory.?(&default_memory_args));

    const dims = [_]i64{2};
    const lhs = [_]f32{ 1.5, -2.0 };
    const rhs = [_]f32{ 2.25, 4.0 };

    var lhs_from_host_args = std.mem.zeroes(c.PJRT_Client_BufferFromHostBuffer_Args);
    lhs_from_host_args.struct_size = c.PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    lhs_from_host_args.client = create_args.client;
    lhs_from_host_args.data = &lhs;
    lhs_from_host_args.type = c.PJRT_Buffer_Type_F32;
    lhs_from_host_args.dims = &dims;
    lhs_from_host_args.num_dims = dims.len;
    lhs_from_host_args.device = devices_args.devices[0];
    lhs_from_host_args.memory = default_memory_args.memory;
    try expectOk(api.PJRT_Client_BufferFromHostBuffer.?(&lhs_from_host_args));
    defer {
        if (lhs_from_host_args.done_with_host_buffer) |event| {
            var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
            event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
            event_destroy_args.event = event;
            _ = api.PJRT_Event_Destroy.?(&event_destroy_args);
        }
        var buffer_destroy_args = std.mem.zeroes(c.PJRT_Buffer_Destroy_Args);
        buffer_destroy_args.struct_size = c.PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        buffer_destroy_args.buffer = lhs_from_host_args.buffer;
        _ = api.PJRT_Buffer_Destroy.?(&buffer_destroy_args);
    }

    var rhs_from_host_args = std.mem.zeroes(c.PJRT_Client_BufferFromHostBuffer_Args);
    rhs_from_host_args.struct_size = c.PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    rhs_from_host_args.client = create_args.client;
    rhs_from_host_args.data = &rhs;
    rhs_from_host_args.type = c.PJRT_Buffer_Type_F32;
    rhs_from_host_args.dims = &dims;
    rhs_from_host_args.num_dims = dims.len;
    rhs_from_host_args.device = devices_args.devices[0];
    rhs_from_host_args.memory = default_memory_args.memory;
    try expectOk(api.PJRT_Client_BufferFromHostBuffer.?(&rhs_from_host_args));
    defer {
        if (rhs_from_host_args.done_with_host_buffer) |event| {
            var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
            event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
            event_destroy_args.event = event;
            _ = api.PJRT_Event_Destroy.?(&event_destroy_args);
        }
        var buffer_destroy_args = std.mem.zeroes(c.PJRT_Buffer_Destroy_Args);
        buffer_destroy_args.struct_size = c.PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        buffer_destroy_args.buffer = rhs_from_host_args.buffer;
        _ = api.PJRT_Buffer_Destroy.?(&buffer_destroy_args);
    }

    const module_text =
        \\module {
        \\  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
        \\    %0 = stablehlo.add %arg0, %arg1 : tensor<2xf32>
        \\    %1 = stablehlo.sqrt %0 : tensor<2xf32>
        \\    return %1 : tensor<2xf32>
        \\  }
        \\}
    ;
    var program = std.mem.zeroes(c.PJRT_Program);
    program.struct_size = c.PJRT_Program_STRUCT_SIZE;
    program.code = @constCast(module_text.ptr);
    program.code_size = module_text.len;
    program.format = "mlir";
    program.format_size = "mlir".len;

    var compile_args = std.mem.zeroes(c.PJRT_Client_Compile_Args);
    compile_args.struct_size = c.PJRT_Client_Compile_Args_STRUCT_SIZE;
    compile_args.client = create_args.client;
    compile_args.program = &program;
    try expectOk(api.PJRT_Client_Compile.?(&compile_args));
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Destroy_Args);
        destroy_args.struct_size = c.PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
        destroy_args.executable = compile_args.executable;
        _ = api.PJRT_LoadedExecutable_Destroy.?(&destroy_args);
    }

    var device_args = [_]?*c.PJRT_Buffer{ lhs_from_host_args.buffer, rhs_from_host_args.buffer };
    var argument_lists = [_][*c]const ?*c.PJRT_Buffer{&device_args};
    var device_outputs = [_]?*c.PJRT_Buffer{null};
    var output_lists = [_][*c]?*c.PJRT_Buffer{&device_outputs};

    var execute_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Execute_Args);
    execute_args.struct_size = c.PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    execute_args.executable = compile_args.executable;
    execute_args.argument_lists = &argument_lists;
    execute_args.num_devices = 1;
    execute_args.num_args = device_args.len;
    execute_args.output_lists = &output_lists;
    try expectOk(api.PJRT_LoadedExecutable_Execute.?(&execute_args));
    defer if (device_outputs[0]) |buffer| {
        var buffer_destroy_args = std.mem.zeroes(c.PJRT_Buffer_Destroy_Args);
        buffer_destroy_args.struct_size = c.PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        buffer_destroy_args.buffer = buffer;
        _ = api.PJRT_Buffer_Destroy.?(&buffer_destroy_args);
    };

    var output: [2]f32 = undefined;
    var to_host_args = std.mem.zeroes(c.PJRT_Buffer_ToHostBuffer_Args);
    to_host_args.struct_size = c.PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    to_host_args.src = device_outputs[0];
    to_host_args.dst = &output;
    to_host_args.dst_size = std.mem.asBytes(&output).len;
    try expectOk(api.PJRT_Buffer_ToHostBuffer.?(&to_host_args));
    defer if (to_host_args.event) |event| {
        var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
        event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
        event_destroy_args.event = event;
        _ = api.PJRT_Event_Destroy.?(&event_destroy_args);
    };
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f32, 3.75)), output[0], 0.0001);
    try std.testing.expectApproxEqAbs(std.math.sqrt(@as(f32, 2.0)), output[1], 0.0001);
}

test "loaded executable execute reshapes f32 buffer metadata through PJRT" {
    const api = GetPjrtApi();

    var option = std.mem.zeroes(c.PJRT_NamedValue);
    option.struct_size = c.PJRT_NamedValue_STRUCT_SIZE;
    option.name = backend_option;
    option.name_size = backend_option.len;
    option.type = c.PJRT_NamedValue_kString;
    option.unnamed_0.string_value = "metal_mlx";
    option.value_size = "metal_mlx".len;

    var create_args = std.mem.zeroes(c.PJRT_Client_Create_Args);
    create_args.struct_size = c.PJRT_Client_Create_Args_STRUCT_SIZE;
    create_args.create_options = &option;
    create_args.num_options = 1;
    try expectOk(api.PJRT_Client_Create.?(&create_args));
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_Client_Destroy_Args);
        destroy_args.struct_size = c.PJRT_Client_Destroy_Args_STRUCT_SIZE;
        destroy_args.client = create_args.client;
        _ = api.PJRT_Client_Destroy.?(&destroy_args);
    }

    var devices_args = std.mem.zeroes(c.PJRT_Client_Devices_Args);
    devices_args.struct_size = c.PJRT_Client_Devices_Args_STRUCT_SIZE;
    devices_args.client = create_args.client;
    try expectOk(api.PJRT_Client_Devices.?(&devices_args));

    var default_memory_args = std.mem.zeroes(c.PJRT_Device_DefaultMemory_Args);
    default_memory_args.struct_size = c.PJRT_Device_DefaultMemory_Args_STRUCT_SIZE;
    default_memory_args.device = devices_args.devices[0];
    try expectOk(api.PJRT_Device_DefaultMemory.?(&default_memory_args));

    const dims = [_]i64{ 2, 2 };
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0 };

    var from_host_args = std.mem.zeroes(c.PJRT_Client_BufferFromHostBuffer_Args);
    from_host_args.struct_size = c.PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    from_host_args.client = create_args.client;
    from_host_args.data = &input;
    from_host_args.type = c.PJRT_Buffer_Type_F32;
    from_host_args.dims = &dims;
    from_host_args.num_dims = dims.len;
    from_host_args.device = devices_args.devices[0];
    from_host_args.memory = default_memory_args.memory;
    try expectOk(api.PJRT_Client_BufferFromHostBuffer.?(&from_host_args));
    defer {
        if (from_host_args.done_with_host_buffer) |event| {
            var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
            event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
            event_destroy_args.event = event;
            _ = api.PJRT_Event_Destroy.?(&event_destroy_args);
        }
        var buffer_destroy_args = std.mem.zeroes(c.PJRT_Buffer_Destroy_Args);
        buffer_destroy_args.struct_size = c.PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        buffer_destroy_args.buffer = from_host_args.buffer;
        _ = api.PJRT_Buffer_Destroy.?(&buffer_destroy_args);
    }

    const module_text =
        \\module {
        \\  func.func @main(%arg0: tensor<2x2xf32>) -> tensor<4xf32> {
        \\    %0 = stablehlo.reshape %arg0 : (tensor<2x2xf32>) -> tensor<4xf32>
        \\    return %0 : tensor<4xf32>
        \\  }
        \\}
    ;
    var program = std.mem.zeroes(c.PJRT_Program);
    program.struct_size = c.PJRT_Program_STRUCT_SIZE;
    program.code = @constCast(module_text.ptr);
    program.code_size = module_text.len;
    program.format = "mlir";
    program.format_size = "mlir".len;

    var compile_args = std.mem.zeroes(c.PJRT_Client_Compile_Args);
    compile_args.struct_size = c.PJRT_Client_Compile_Args_STRUCT_SIZE;
    compile_args.client = create_args.client;
    compile_args.program = &program;
    try expectOk(api.PJRT_Client_Compile.?(&compile_args));
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Destroy_Args);
        destroy_args.struct_size = c.PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
        destroy_args.executable = compile_args.executable;
        _ = api.PJRT_LoadedExecutable_Destroy.?(&destroy_args);
    }

    var device_args = [_]?*c.PJRT_Buffer{from_host_args.buffer};
    var argument_lists = [_][*c]const ?*c.PJRT_Buffer{&device_args};
    var device_outputs = [_]?*c.PJRT_Buffer{null};
    var output_lists = [_][*c]?*c.PJRT_Buffer{&device_outputs};

    var execute_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Execute_Args);
    execute_args.struct_size = c.PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    execute_args.executable = compile_args.executable;
    execute_args.argument_lists = &argument_lists;
    execute_args.num_devices = 1;
    execute_args.num_args = device_args.len;
    execute_args.output_lists = &output_lists;
    try expectOk(api.PJRT_LoadedExecutable_Execute.?(&execute_args));
    defer if (device_outputs[0]) |buffer| {
        var buffer_destroy_args = std.mem.zeroes(c.PJRT_Buffer_Destroy_Args);
        buffer_destroy_args.struct_size = c.PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        buffer_destroy_args.buffer = buffer;
        _ = api.PJRT_Buffer_Destroy.?(&buffer_destroy_args);
    };

    var dimensions_args = std.mem.zeroes(c.PJRT_Buffer_Dimensions_Args);
    dimensions_args.struct_size = c.PJRT_Buffer_Dimensions_Args_STRUCT_SIZE;
    dimensions_args.buffer = device_outputs[0];
    try expectOk(api.PJRT_Buffer_Dimensions.?(&dimensions_args));
    try std.testing.expectEqualSlices(i64, &.{4}, dimensions_args.dims[0..dimensions_args.num_dims]);

    var output: [4]f32 = undefined;
    var to_host_args = std.mem.zeroes(c.PJRT_Buffer_ToHostBuffer_Args);
    to_host_args.struct_size = c.PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    to_host_args.src = device_outputs[0];
    to_host_args.dst = &output;
    to_host_args.dst_size = std.mem.asBytes(&output).len;
    try expectOk(api.PJRT_Buffer_ToHostBuffer.?(&to_host_args));
    defer if (to_host_args.event) |event| {
        var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
        event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
        event_destroy_args.event = event;
        _ = api.PJRT_Event_Destroy.?(&event_destroy_args);
    };
    try std.testing.expectEqualSlices(u8, std.mem.asBytes(&input), std.mem.asBytes(&output));
}

test "loaded executable execute transposes f32 buffer data through PJRT" {
    const api = GetPjrtApi();

    var option = std.mem.zeroes(c.PJRT_NamedValue);
    option.struct_size = c.PJRT_NamedValue_STRUCT_SIZE;
    option.name = backend_option;
    option.name_size = backend_option.len;
    option.type = c.PJRT_NamedValue_kString;
    option.unnamed_0.string_value = "metal_mlx";
    option.value_size = "metal_mlx".len;

    var create_args = std.mem.zeroes(c.PJRT_Client_Create_Args);
    create_args.struct_size = c.PJRT_Client_Create_Args_STRUCT_SIZE;
    create_args.create_options = &option;
    create_args.num_options = 1;
    try expectOk(api.PJRT_Client_Create.?(&create_args));
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_Client_Destroy_Args);
        destroy_args.struct_size = c.PJRT_Client_Destroy_Args_STRUCT_SIZE;
        destroy_args.client = create_args.client;
        _ = api.PJRT_Client_Destroy.?(&destroy_args);
    }

    var devices_args = std.mem.zeroes(c.PJRT_Client_Devices_Args);
    devices_args.struct_size = c.PJRT_Client_Devices_Args_STRUCT_SIZE;
    devices_args.client = create_args.client;
    try expectOk(api.PJRT_Client_Devices.?(&devices_args));

    var default_memory_args = std.mem.zeroes(c.PJRT_Device_DefaultMemory_Args);
    default_memory_args.struct_size = c.PJRT_Device_DefaultMemory_Args_STRUCT_SIZE;
    default_memory_args.device = devices_args.devices[0];
    try expectOk(api.PJRT_Device_DefaultMemory.?(&default_memory_args));

    const dims = [_]i64{ 2, 3 };
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };

    var from_host_args = std.mem.zeroes(c.PJRT_Client_BufferFromHostBuffer_Args);
    from_host_args.struct_size = c.PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    from_host_args.client = create_args.client;
    from_host_args.data = &input;
    from_host_args.type = c.PJRT_Buffer_Type_F32;
    from_host_args.dims = &dims;
    from_host_args.num_dims = dims.len;
    from_host_args.device = devices_args.devices[0];
    from_host_args.memory = default_memory_args.memory;
    try expectOk(api.PJRT_Client_BufferFromHostBuffer.?(&from_host_args));
    defer {
        if (from_host_args.done_with_host_buffer) |event| {
            var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
            event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
            event_destroy_args.event = event;
            _ = api.PJRT_Event_Destroy.?(&event_destroy_args);
        }
        var buffer_destroy_args = std.mem.zeroes(c.PJRT_Buffer_Destroy_Args);
        buffer_destroy_args.struct_size = c.PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        buffer_destroy_args.buffer = from_host_args.buffer;
        _ = api.PJRT_Buffer_Destroy.?(&buffer_destroy_args);
    }

    const module_text =
        \\module {
        \\  func.func @main(%arg0: tensor<2x3xf32>) -> tensor<3x2xf32> {
        \\    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
        \\    return %0 : tensor<3x2xf32>
        \\  }
        \\}
    ;
    var program = std.mem.zeroes(c.PJRT_Program);
    program.struct_size = c.PJRT_Program_STRUCT_SIZE;
    program.code = @constCast(module_text.ptr);
    program.code_size = module_text.len;
    program.format = "mlir";
    program.format_size = "mlir".len;

    var compile_args = std.mem.zeroes(c.PJRT_Client_Compile_Args);
    compile_args.struct_size = c.PJRT_Client_Compile_Args_STRUCT_SIZE;
    compile_args.client = create_args.client;
    compile_args.program = &program;
    try expectOk(api.PJRT_Client_Compile.?(&compile_args));
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Destroy_Args);
        destroy_args.struct_size = c.PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
        destroy_args.executable = compile_args.executable;
        _ = api.PJRT_LoadedExecutable_Destroy.?(&destroy_args);
    }

    var device_args = [_]?*c.PJRT_Buffer{from_host_args.buffer};
    var argument_lists = [_][*c]const ?*c.PJRT_Buffer{&device_args};
    var device_outputs = [_]?*c.PJRT_Buffer{null};
    var output_lists = [_][*c]?*c.PJRT_Buffer{&device_outputs};

    var execute_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Execute_Args);
    execute_args.struct_size = c.PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    execute_args.executable = compile_args.executable;
    execute_args.argument_lists = &argument_lists;
    execute_args.num_devices = 1;
    execute_args.num_args = device_args.len;
    execute_args.output_lists = &output_lists;
    try expectOk(api.PJRT_LoadedExecutable_Execute.?(&execute_args));
    defer if (device_outputs[0]) |buffer| {
        var buffer_destroy_args = std.mem.zeroes(c.PJRT_Buffer_Destroy_Args);
        buffer_destroy_args.struct_size = c.PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        buffer_destroy_args.buffer = buffer;
        _ = api.PJRT_Buffer_Destroy.?(&buffer_destroy_args);
    };

    var dimensions_args = std.mem.zeroes(c.PJRT_Buffer_Dimensions_Args);
    dimensions_args.struct_size = c.PJRT_Buffer_Dimensions_Args_STRUCT_SIZE;
    dimensions_args.buffer = device_outputs[0];
    try expectOk(api.PJRT_Buffer_Dimensions.?(&dimensions_args));
    try std.testing.expectEqualSlices(i64, &.{ 3, 2 }, dimensions_args.dims[0..dimensions_args.num_dims]);

    var output: [6]f32 = undefined;
    var to_host_args = std.mem.zeroes(c.PJRT_Buffer_ToHostBuffer_Args);
    to_host_args.struct_size = c.PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    to_host_args.src = device_outputs[0];
    to_host_args.dst = &output;
    to_host_args.dst_size = std.mem.asBytes(&output).len;
    try expectOk(api.PJRT_Buffer_ToHostBuffer.?(&to_host_args));
    defer if (to_host_args.event) |event| {
        var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
        event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
        event_destroy_args.event = event;
        _ = api.PJRT_Event_Destroy.?(&event_destroy_args);
    };
    try std.testing.expectEqualSlices(f32, &.{ 1.0, 4.0, 2.0, 5.0, 3.0, 6.0 }, &output);
}

test "loaded executable execute broadcasts f32 buffer data through PJRT" {
    const api = GetPjrtApi();

    var option = std.mem.zeroes(c.PJRT_NamedValue);
    option.struct_size = c.PJRT_NamedValue_STRUCT_SIZE;
    option.name = backend_option;
    option.name_size = backend_option.len;
    option.type = c.PJRT_NamedValue_kString;
    option.unnamed_0.string_value = "metal_mlx";
    option.value_size = "metal_mlx".len;

    var create_args = std.mem.zeroes(c.PJRT_Client_Create_Args);
    create_args.struct_size = c.PJRT_Client_Create_Args_STRUCT_SIZE;
    create_args.create_options = &option;
    create_args.num_options = 1;
    try expectOk(api.PJRT_Client_Create.?(&create_args));
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_Client_Destroy_Args);
        destroy_args.struct_size = c.PJRT_Client_Destroy_Args_STRUCT_SIZE;
        destroy_args.client = create_args.client;
        _ = api.PJRT_Client_Destroy.?(&destroy_args);
    }

    var devices_args = std.mem.zeroes(c.PJRT_Client_Devices_Args);
    devices_args.struct_size = c.PJRT_Client_Devices_Args_STRUCT_SIZE;
    devices_args.client = create_args.client;
    try expectOk(api.PJRT_Client_Devices.?(&devices_args));

    var default_memory_args = std.mem.zeroes(c.PJRT_Device_DefaultMemory_Args);
    default_memory_args.struct_size = c.PJRT_Device_DefaultMemory_Args_STRUCT_SIZE;
    default_memory_args.device = devices_args.devices[0];
    try expectOk(api.PJRT_Device_DefaultMemory.?(&default_memory_args));

    const dims = [_]i64{3};
    const input = [_]f32{ 7.0, 8.0, 9.0 };

    var from_host_args = std.mem.zeroes(c.PJRT_Client_BufferFromHostBuffer_Args);
    from_host_args.struct_size = c.PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    from_host_args.client = create_args.client;
    from_host_args.data = &input;
    from_host_args.type = c.PJRT_Buffer_Type_F32;
    from_host_args.dims = &dims;
    from_host_args.num_dims = dims.len;
    from_host_args.device = devices_args.devices[0];
    from_host_args.memory = default_memory_args.memory;
    try expectOk(api.PJRT_Client_BufferFromHostBuffer.?(&from_host_args));
    defer {
        if (from_host_args.done_with_host_buffer) |event| {
            var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
            event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
            event_destroy_args.event = event;
            _ = api.PJRT_Event_Destroy.?(&event_destroy_args);
        }
        var buffer_destroy_args = std.mem.zeroes(c.PJRT_Buffer_Destroy_Args);
        buffer_destroy_args.struct_size = c.PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        buffer_destroy_args.buffer = from_host_args.buffer;
        _ = api.PJRT_Buffer_Destroy.?(&buffer_destroy_args);
    }

    const module_text =
        \\module {
        \\  func.func @main(%arg0: tensor<3xf32>) -> tensor<2x3xf32> {
        \\    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<3xf32>) -> tensor<2x3xf32>
        \\    return %0 : tensor<2x3xf32>
        \\  }
        \\}
    ;
    var program = std.mem.zeroes(c.PJRT_Program);
    program.struct_size = c.PJRT_Program_STRUCT_SIZE;
    program.code = @constCast(module_text.ptr);
    program.code_size = module_text.len;
    program.format = "mlir";
    program.format_size = "mlir".len;

    var compile_args = std.mem.zeroes(c.PJRT_Client_Compile_Args);
    compile_args.struct_size = c.PJRT_Client_Compile_Args_STRUCT_SIZE;
    compile_args.client = create_args.client;
    compile_args.program = &program;
    try expectOk(api.PJRT_Client_Compile.?(&compile_args));
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Destroy_Args);
        destroy_args.struct_size = c.PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
        destroy_args.executable = compile_args.executable;
        _ = api.PJRT_LoadedExecutable_Destroy.?(&destroy_args);
    }

    var device_args = [_]?*c.PJRT_Buffer{from_host_args.buffer};
    var argument_lists = [_][*c]const ?*c.PJRT_Buffer{&device_args};
    var device_outputs = [_]?*c.PJRT_Buffer{null};
    var output_lists = [_][*c]?*c.PJRT_Buffer{&device_outputs};

    var execute_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Execute_Args);
    execute_args.struct_size = c.PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    execute_args.executable = compile_args.executable;
    execute_args.argument_lists = &argument_lists;
    execute_args.num_devices = 1;
    execute_args.num_args = device_args.len;
    execute_args.output_lists = &output_lists;
    try expectOk(api.PJRT_LoadedExecutable_Execute.?(&execute_args));
    defer if (device_outputs[0]) |buffer| {
        var buffer_destroy_args = std.mem.zeroes(c.PJRT_Buffer_Destroy_Args);
        buffer_destroy_args.struct_size = c.PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        buffer_destroy_args.buffer = buffer;
        _ = api.PJRT_Buffer_Destroy.?(&buffer_destroy_args);
    };

    var dimensions_args = std.mem.zeroes(c.PJRT_Buffer_Dimensions_Args);
    dimensions_args.struct_size = c.PJRT_Buffer_Dimensions_Args_STRUCT_SIZE;
    dimensions_args.buffer = device_outputs[0];
    try expectOk(api.PJRT_Buffer_Dimensions.?(&dimensions_args));
    try std.testing.expectEqualSlices(i64, &.{ 2, 3 }, dimensions_args.dims[0..dimensions_args.num_dims]);

    var output: [6]f32 = undefined;
    var to_host_args = std.mem.zeroes(c.PJRT_Buffer_ToHostBuffer_Args);
    to_host_args.struct_size = c.PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    to_host_args.src = device_outputs[0];
    to_host_args.dst = &output;
    to_host_args.dst_size = std.mem.asBytes(&output).len;
    try expectOk(api.PJRT_Buffer_ToHostBuffer.?(&to_host_args));
    defer if (to_host_args.event) |event| {
        var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
        event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
        event_destroy_args.event = event;
        _ = api.PJRT_Event_Destroy.?(&event_destroy_args);
    };
    try std.testing.expectEqualSlices(f32, &.{ 7.0, 8.0, 9.0, 7.0, 8.0, 9.0 }, &output);
}

test "loaded executable execute slices f32 buffer data through PJRT" {
    const api = GetPjrtApi();

    var option = std.mem.zeroes(c.PJRT_NamedValue);
    option.struct_size = c.PJRT_NamedValue_STRUCT_SIZE;
    option.name = backend_option;
    option.name_size = backend_option.len;
    option.type = c.PJRT_NamedValue_kString;
    option.unnamed_0.string_value = "metal_mlx";
    option.value_size = "metal_mlx".len;

    var create_args = std.mem.zeroes(c.PJRT_Client_Create_Args);
    create_args.struct_size = c.PJRT_Client_Create_Args_STRUCT_SIZE;
    create_args.create_options = &option;
    create_args.num_options = 1;
    try expectOk(api.PJRT_Client_Create.?(&create_args));
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_Client_Destroy_Args);
        destroy_args.struct_size = c.PJRT_Client_Destroy_Args_STRUCT_SIZE;
        destroy_args.client = create_args.client;
        _ = api.PJRT_Client_Destroy.?(&destroy_args);
    }

    var devices_args = std.mem.zeroes(c.PJRT_Client_Devices_Args);
    devices_args.struct_size = c.PJRT_Client_Devices_Args_STRUCT_SIZE;
    devices_args.client = create_args.client;
    try expectOk(api.PJRT_Client_Devices.?(&devices_args));

    var default_memory_args = std.mem.zeroes(c.PJRT_Device_DefaultMemory_Args);
    default_memory_args.struct_size = c.PJRT_Device_DefaultMemory_Args_STRUCT_SIZE;
    default_memory_args.device = devices_args.devices[0];
    try expectOk(api.PJRT_Device_DefaultMemory.?(&default_memory_args));

    const dims = [_]i64{ 3, 4 };
    const input = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0 };

    var from_host_args = std.mem.zeroes(c.PJRT_Client_BufferFromHostBuffer_Args);
    from_host_args.struct_size = c.PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    from_host_args.client = create_args.client;
    from_host_args.data = &input;
    from_host_args.type = c.PJRT_Buffer_Type_F32;
    from_host_args.dims = &dims;
    from_host_args.num_dims = dims.len;
    from_host_args.device = devices_args.devices[0];
    from_host_args.memory = default_memory_args.memory;
    try expectOk(api.PJRT_Client_BufferFromHostBuffer.?(&from_host_args));
    defer {
        if (from_host_args.done_with_host_buffer) |event| {
            var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
            event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
            event_destroy_args.event = event;
            _ = api.PJRT_Event_Destroy.?(&event_destroy_args);
        }
        var buffer_destroy_args = std.mem.zeroes(c.PJRT_Buffer_Destroy_Args);
        buffer_destroy_args.struct_size = c.PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        buffer_destroy_args.buffer = from_host_args.buffer;
        _ = api.PJRT_Buffer_Destroy.?(&buffer_destroy_args);
    }

    const module_text =
        \\module {
        \\  func.func @main(%arg0: tensor<3x4xf32>) -> tensor<2x2xf32> {
        \\    %0 = stablehlo.slice %arg0 [1:3, 0:4:2] : (tensor<3x4xf32>) -> tensor<2x2xf32>
        \\    return %0 : tensor<2x2xf32>
        \\  }
        \\}
    ;
    var program = std.mem.zeroes(c.PJRT_Program);
    program.struct_size = c.PJRT_Program_STRUCT_SIZE;
    program.code = @constCast(module_text.ptr);
    program.code_size = module_text.len;
    program.format = "mlir";
    program.format_size = "mlir".len;

    var compile_args = std.mem.zeroes(c.PJRT_Client_Compile_Args);
    compile_args.struct_size = c.PJRT_Client_Compile_Args_STRUCT_SIZE;
    compile_args.client = create_args.client;
    compile_args.program = &program;
    try expectOk(api.PJRT_Client_Compile.?(&compile_args));
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Destroy_Args);
        destroy_args.struct_size = c.PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
        destroy_args.executable = compile_args.executable;
        _ = api.PJRT_LoadedExecutable_Destroy.?(&destroy_args);
    }

    var device_args = [_]?*c.PJRT_Buffer{from_host_args.buffer};
    var argument_lists = [_][*c]const ?*c.PJRT_Buffer{&device_args};
    var device_outputs = [_]?*c.PJRT_Buffer{null};
    var output_lists = [_][*c]?*c.PJRT_Buffer{&device_outputs};

    var execute_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Execute_Args);
    execute_args.struct_size = c.PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    execute_args.executable = compile_args.executable;
    execute_args.argument_lists = &argument_lists;
    execute_args.num_devices = 1;
    execute_args.num_args = device_args.len;
    execute_args.output_lists = &output_lists;
    try expectOk(api.PJRT_LoadedExecutable_Execute.?(&execute_args));
    defer if (device_outputs[0]) |buffer| {
        var buffer_destroy_args = std.mem.zeroes(c.PJRT_Buffer_Destroy_Args);
        buffer_destroy_args.struct_size = c.PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        buffer_destroy_args.buffer = buffer;
        _ = api.PJRT_Buffer_Destroy.?(&buffer_destroy_args);
    };

    var dimensions_args = std.mem.zeroes(c.PJRT_Buffer_Dimensions_Args);
    dimensions_args.struct_size = c.PJRT_Buffer_Dimensions_Args_STRUCT_SIZE;
    dimensions_args.buffer = device_outputs[0];
    try expectOk(api.PJRT_Buffer_Dimensions.?(&dimensions_args));
    try std.testing.expectEqualSlices(i64, &.{ 2, 2 }, dimensions_args.dims[0..dimensions_args.num_dims]);

    var output: [4]f32 = undefined;
    var to_host_args = std.mem.zeroes(c.PJRT_Buffer_ToHostBuffer_Args);
    to_host_args.struct_size = c.PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    to_host_args.src = device_outputs[0];
    to_host_args.dst = &output;
    to_host_args.dst_size = std.mem.asBytes(&output).len;
    try expectOk(api.PJRT_Buffer_ToHostBuffer.?(&to_host_args));
    defer if (to_host_args.event) |event| {
        var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
        event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
        event_destroy_args.event = event;
        _ = api.PJRT_Event_Destroy.?(&event_destroy_args);
    };
    try std.testing.expectEqualSlices(f32, &.{ 5.0, 7.0, 9.0, 11.0 }, &output);
}

test "loaded executable execute concatenates f32 buffers through PJRT" {
    const api = GetPjrtApi();

    var option = std.mem.zeroes(c.PJRT_NamedValue);
    option.struct_size = c.PJRT_NamedValue_STRUCT_SIZE;
    option.name = backend_option;
    option.name_size = backend_option.len;
    option.type = c.PJRT_NamedValue_kString;
    option.unnamed_0.string_value = "metal_mlx";
    option.value_size = "metal_mlx".len;

    var create_args = std.mem.zeroes(c.PJRT_Client_Create_Args);
    create_args.struct_size = c.PJRT_Client_Create_Args_STRUCT_SIZE;
    create_args.create_options = &option;
    create_args.num_options = 1;
    try expectOk(api.PJRT_Client_Create.?(&create_args));
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_Client_Destroy_Args);
        destroy_args.struct_size = c.PJRT_Client_Destroy_Args_STRUCT_SIZE;
        destroy_args.client = create_args.client;
        _ = api.PJRT_Client_Destroy.?(&destroy_args);
    }

    var devices_args = std.mem.zeroes(c.PJRT_Client_Devices_Args);
    devices_args.struct_size = c.PJRT_Client_Devices_Args_STRUCT_SIZE;
    devices_args.client = create_args.client;
    try expectOk(api.PJRT_Client_Devices.?(&devices_args));

    var default_memory_args = std.mem.zeroes(c.PJRT_Device_DefaultMemory_Args);
    default_memory_args.struct_size = c.PJRT_Device_DefaultMemory_Args_STRUCT_SIZE;
    default_memory_args.device = devices_args.devices[0];
    try expectOk(api.PJRT_Device_DefaultMemory.?(&default_memory_args));

    const lhs_dims = [_]i64{ 2, 2 };
    const rhs_dims = [_]i64{ 2, 3 };
    const lhs = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const rhs = [_]f32{ 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };

    var lhs_from_host_args = std.mem.zeroes(c.PJRT_Client_BufferFromHostBuffer_Args);
    lhs_from_host_args.struct_size = c.PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    lhs_from_host_args.client = create_args.client;
    lhs_from_host_args.data = &lhs;
    lhs_from_host_args.type = c.PJRT_Buffer_Type_F32;
    lhs_from_host_args.dims = &lhs_dims;
    lhs_from_host_args.num_dims = lhs_dims.len;
    lhs_from_host_args.device = devices_args.devices[0];
    lhs_from_host_args.memory = default_memory_args.memory;
    try expectOk(api.PJRT_Client_BufferFromHostBuffer.?(&lhs_from_host_args));
    defer {
        if (lhs_from_host_args.done_with_host_buffer) |event| {
            var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
            event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
            event_destroy_args.event = event;
            _ = api.PJRT_Event_Destroy.?(&event_destroy_args);
        }
        var buffer_destroy_args = std.mem.zeroes(c.PJRT_Buffer_Destroy_Args);
        buffer_destroy_args.struct_size = c.PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        buffer_destroy_args.buffer = lhs_from_host_args.buffer;
        _ = api.PJRT_Buffer_Destroy.?(&buffer_destroy_args);
    }

    var rhs_from_host_args = std.mem.zeroes(c.PJRT_Client_BufferFromHostBuffer_Args);
    rhs_from_host_args.struct_size = c.PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    rhs_from_host_args.client = create_args.client;
    rhs_from_host_args.data = &rhs;
    rhs_from_host_args.type = c.PJRT_Buffer_Type_F32;
    rhs_from_host_args.dims = &rhs_dims;
    rhs_from_host_args.num_dims = rhs_dims.len;
    rhs_from_host_args.device = devices_args.devices[0];
    rhs_from_host_args.memory = default_memory_args.memory;
    try expectOk(api.PJRT_Client_BufferFromHostBuffer.?(&rhs_from_host_args));
    defer {
        if (rhs_from_host_args.done_with_host_buffer) |event| {
            var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
            event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
            event_destroy_args.event = event;
            _ = api.PJRT_Event_Destroy.?(&event_destroy_args);
        }
        var buffer_destroy_args = std.mem.zeroes(c.PJRT_Buffer_Destroy_Args);
        buffer_destroy_args.struct_size = c.PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        buffer_destroy_args.buffer = rhs_from_host_args.buffer;
        _ = api.PJRT_Buffer_Destroy.?(&buffer_destroy_args);
    }

    const module_text =
        \\module {
        \\  func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x5xf32> {
        \\    %0 = stablehlo.concatenate %arg0, %arg1, dim = 1 : (tensor<2x2xf32>, tensor<2x3xf32>) -> tensor<2x5xf32>
        \\    return %0 : tensor<2x5xf32>
        \\  }
        \\}
    ;
    var program = std.mem.zeroes(c.PJRT_Program);
    program.struct_size = c.PJRT_Program_STRUCT_SIZE;
    program.code = @constCast(module_text.ptr);
    program.code_size = module_text.len;
    program.format = "mlir";
    program.format_size = "mlir".len;

    var compile_args = std.mem.zeroes(c.PJRT_Client_Compile_Args);
    compile_args.struct_size = c.PJRT_Client_Compile_Args_STRUCT_SIZE;
    compile_args.client = create_args.client;
    compile_args.program = &program;
    try expectOk(api.PJRT_Client_Compile.?(&compile_args));
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Destroy_Args);
        destroy_args.struct_size = c.PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
        destroy_args.executable = compile_args.executable;
        _ = api.PJRT_LoadedExecutable_Destroy.?(&destroy_args);
    }

    var device_args = [_]?*c.PJRT_Buffer{ lhs_from_host_args.buffer, rhs_from_host_args.buffer };
    var argument_lists = [_][*c]const ?*c.PJRT_Buffer{&device_args};
    var device_outputs = [_]?*c.PJRT_Buffer{null};
    var output_lists = [_][*c]?*c.PJRT_Buffer{&device_outputs};

    var execute_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Execute_Args);
    execute_args.struct_size = c.PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    execute_args.executable = compile_args.executable;
    execute_args.argument_lists = &argument_lists;
    execute_args.num_devices = 1;
    execute_args.num_args = device_args.len;
    execute_args.output_lists = &output_lists;
    try expectOk(api.PJRT_LoadedExecutable_Execute.?(&execute_args));
    defer if (device_outputs[0]) |buffer| {
        var buffer_destroy_args = std.mem.zeroes(c.PJRT_Buffer_Destroy_Args);
        buffer_destroy_args.struct_size = c.PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        buffer_destroy_args.buffer = buffer;
        _ = api.PJRT_Buffer_Destroy.?(&buffer_destroy_args);
    };

    var dimensions_args = std.mem.zeroes(c.PJRT_Buffer_Dimensions_Args);
    dimensions_args.struct_size = c.PJRT_Buffer_Dimensions_Args_STRUCT_SIZE;
    dimensions_args.buffer = device_outputs[0];
    try expectOk(api.PJRT_Buffer_Dimensions.?(&dimensions_args));
    try std.testing.expectEqualSlices(i64, &.{ 2, 5 }, dimensions_args.dims[0..dimensions_args.num_dims]);

    var output: [10]f32 = undefined;
    var to_host_args = std.mem.zeroes(c.PJRT_Buffer_ToHostBuffer_Args);
    to_host_args.struct_size = c.PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    to_host_args.src = device_outputs[0];
    to_host_args.dst = &output;
    to_host_args.dst_size = std.mem.asBytes(&output).len;
    try expectOk(api.PJRT_Buffer_ToHostBuffer.?(&to_host_args));
    defer if (to_host_args.event) |event| {
        var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
        event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
        event_destroy_args.event = event;
        _ = api.PJRT_Event_Destroy.?(&event_destroy_args);
    };
    try std.testing.expectEqualSlices(f32, &.{ 1.0, 2.0, 5.0, 6.0, 7.0, 3.0, 4.0, 8.0, 9.0, 10.0 }, &output);
}

test "client default device assignment follows available Metal devices through PJRT" {
    const api = GetPjrtApi();

    var create_args = std.mem.zeroes(c.PJRT_Client_Create_Args);
    create_args.struct_size = c.PJRT_Client_Create_Args_STRUCT_SIZE;
    try expectOk(api.PJRT_Client_Create.?(&create_args));
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_Client_Destroy_Args);
        destroy_args.struct_size = c.PJRT_Client_Destroy_Args_STRUCT_SIZE;
        destroy_args.client = create_args.client;
        _ = api.PJRT_Client_Destroy.?(&destroy_args);
    }

    var devices_args = std.mem.zeroes(c.PJRT_Client_Devices_Args);
    devices_args.struct_size = c.PJRT_Client_Devices_Args_STRUCT_SIZE;
    devices_args.client = create_args.client;
    try expectOk(api.PJRT_Client_Devices.?(&devices_args));
    try std.testing.expect(devices_args.num_devices >= 1);

    var assignment = [_]c_int{-1};
    var assignment_args = std.mem.zeroes(c.PJRT_Client_DefaultDeviceAssignment_Args);
    assignment_args.struct_size = c.PJRT_Client_DefaultDeviceAssignment_Args_STRUCT_SIZE;
    assignment_args.client = create_args.client;
    assignment_args.num_replicas = 1;
    assignment_args.num_partitions = 1;
    assignment_args.default_assignment_size = assignment.len;
    assignment_args.default_assignment = &assignment;
    try expectOk(api.PJRT_Client_DefaultDeviceAssignment.?(&assignment_args));
    try std.testing.expectEqualSlices(c_int, &.{0}, &assignment);
}

test "compile preserves bootstrap replicas partitions and shardy validation" {
    const api = GetPjrtApi();

    var create_args = std.mem.zeroes(c.PJRT_Client_Create_Args);
    create_args.struct_size = c.PJRT_Client_Create_Args_STRUCT_SIZE;
    try expectOk(api.PJRT_Client_Create.?(&create_args));
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_Client_Destroy_Args);
        destroy_args.struct_size = c.PJRT_Client_Destroy_Args_STRUCT_SIZE;
        destroy_args.client = create_args.client;
        _ = api.PJRT_Client_Destroy.?(&destroy_args);
    }

    const module_text =
        \\module {
        \\  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
        \\    %0 = stablehlo.add %arg0, %arg1 : tensor<2xf32>
        \\    return %0 : tensor<2xf32>
        \\  }
        \\}
    ;
    var program = std.mem.zeroes(c.PJRT_Program);
    program.struct_size = c.PJRT_Program_STRUCT_SIZE;
    program.code = @constCast(module_text.ptr);
    program.code_size = module_text.len;
    program.format = "mlir";
    program.format_size = "mlir".len;

    const compile_options = "replicas=1; partitions=1; use_shardy=true; assignment=0";
    var compile_args = std.mem.zeroes(c.PJRT_Client_Compile_Args);
    compile_args.struct_size = c.PJRT_Client_Compile_Args_STRUCT_SIZE;
    compile_args.client = create_args.client;
    compile_args.program = &program;
    compile_args.compile_options = compile_options;
    compile_args.compile_options_size = compile_options.len;
    try expectOk(api.PJRT_Client_Compile.?(&compile_args));
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Destroy_Args);
        destroy_args.struct_size = c.PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
        destroy_args.executable = compile_args.executable;
        _ = api.PJRT_LoadedExecutable_Destroy.?(&destroy_args);
    }

    var get_executable_args = std.mem.zeroes(c.PJRT_LoadedExecutable_GetExecutable_Args);
    get_executable_args.struct_size = c.PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE;
    get_executable_args.loaded_executable = compile_args.executable;
    try expectOk(api.PJRT_LoadedExecutable_GetExecutable.?(&get_executable_args));

    var replicas_args = std.mem.zeroes(c.PJRT_Executable_NumReplicas_Args);
    replicas_args.struct_size = c.PJRT_Executable_NumReplicas_Args_STRUCT_SIZE;
    replicas_args.executable = get_executable_args.executable;
    try expectOk(api.PJRT_Executable_NumReplicas.?(&replicas_args));
    try std.testing.expectEqual(@as(usize, 1), replicas_args.num_replicas);

    var partitions_args = std.mem.zeroes(c.PJRT_Executable_NumPartitions_Args);
    partitions_args.struct_size = c.PJRT_Executable_NumPartitions_Args_STRUCT_SIZE;
    partitions_args.executable = get_executable_args.executable;
    try expectOk(api.PJRT_Executable_NumPartitions.?(&partitions_args));
    try std.testing.expectEqual(@as(usize, 1), partitions_args.num_partitions);
}

test "compile rejects unavailable multi-device shardy executable plan through PJRT" {
    const api = GetPjrtApi();

    var create_args = std.mem.zeroes(c.PJRT_Client_Create_Args);
    create_args.struct_size = c.PJRT_Client_Create_Args_STRUCT_SIZE;
    try expectOk(api.PJRT_Client_Create.?(&create_args));
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_Client_Destroy_Args);
        destroy_args.struct_size = c.PJRT_Client_Destroy_Args_STRUCT_SIZE;
        destroy_args.client = create_args.client;
        _ = api.PJRT_Client_Destroy.?(&destroy_args);
    }

    const module_text =
        \\sdy.mesh @mesh = <["x"=2]>
        \\func.func @main(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> tensor<4xf32> {
        \\  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<4xf32>
        \\  return %0 : tensor<4xf32>
        \\}
    ;
    var program = std.mem.zeroes(c.PJRT_Program);
    program.struct_size = c.PJRT_Program_STRUCT_SIZE;
    program.code = @constCast(module_text.ptr);
    program.code_size = module_text.len;
    program.format = "mlir";
    program.format_size = "mlir".len;

    const compile_options = "partitions=2; use_shardy=true; assignment=0,1";
    var compile_args = std.mem.zeroes(c.PJRT_Client_Compile_Args);
    compile_args.struct_size = c.PJRT_Client_Compile_Args_STRUCT_SIZE;
    compile_args.client = create_args.client;
    compile_args.program = &program;
    compile_args.compile_options = compile_options;
    compile_args.compile_options_size = compile_options.len;
    const err = api.PJRT_Client_Compile.?(&compile_args) orelse return error.TestUnexpectedResult;
    defer destroyError(api, err);
    try std.testing.expectEqualStrings("compile options require more devices than the client exposes", errorMessage(api, err));
}

test "compile unsupported op returns detailed PJRT diagnostic" {
    const api = GetPjrtApi();

    var create_args = std.mem.zeroes(c.PJRT_Client_Create_Args);
    create_args.struct_size = c.PJRT_Client_Create_Args_STRUCT_SIZE;
    try expectOk(api.PJRT_Client_Create.?(&create_args));
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_Client_Destroy_Args);
        destroy_args.struct_size = c.PJRT_Client_Destroy_Args_STRUCT_SIZE;
        destroy_args.client = create_args.client;
        _ = api.PJRT_Client_Destroy.?(&destroy_args);
    }

    const module_text =
        \\sdy.mesh @mesh = <["x"=2]>
        \\func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
        \\  %0 = "stablehlo.optimization_barrier"(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<4xf32>) -> tensor<4xf32>
        \\  return %0 : tensor<4xf32>
        \\}
    ;
    var program = std.mem.zeroes(c.PJRT_Program);
    program.struct_size = c.PJRT_Program_STRUCT_SIZE;
    program.code = @constCast(module_text.ptr);
    program.code_size = module_text.len;
    program.format = "mlir";
    program.format_size = "mlir".len;

    var compile_args = std.mem.zeroes(c.PJRT_Client_Compile_Args);
    compile_args.struct_size = c.PJRT_Client_Compile_Args_STRUCT_SIZE;
    compile_args.client = create_args.client;
    compile_args.program = &program;
    const err = api.PJRT_Client_Compile.?(&compile_args);
    try std.testing.expect(err != null);
    defer destroyError(api, err);

    var code_args = std.mem.zeroes(c.PJRT_Error_GetCode_Args);
    code_args.struct_size = c.PJRT_Error_GetCode_Args_STRUCT_SIZE;
    code_args.@"error" = err;
    try expectOk(api.PJRT_Error_GetCode.?(&code_args));
    try std.testing.expectEqual(@as(c.PJRT_Error_Code, @intCast(c.PJRT_Error_Code_UNIMPLEMENTED)), code_args.code);

    const message = errorMessage(api, err);
    try std.testing.expect(std.mem.indexOf(u8, message, "op=optimization_barrier") != null);
    try std.testing.expect(std.mem.indexOf(u8, message, "dtype=f32") != null);
    try std.testing.expect(std.mem.indexOf(u8, message, "rank=1") != null);
    try std.testing.expect(std.mem.indexOf(u8, message, "sharding=sdy.sharding_per_value") != null);
}
