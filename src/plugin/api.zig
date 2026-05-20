const std = @import("std");

const runtime = @import("src/runtime");
const c = @import("c");
const abi = @import("pjrt_abi.zig");
const async_h2d_mod = @import("async_h2d.zig");
const buffer_mod = @import("buffer.zig");
const client_mod = @import("client.zig");
const custom_call = @import("custom_call.zig");
const device_memory_mod = @import("device_memory.zig");
const errors_mod = @import("errors.zig");
const events_mod = @import("events.zig");
const executable_mod = @import("executable.zig");
const execute_mod = @import("execute.zig");
const state_mod = @import("state.zig");
const topology_mod = @import("topology.zig");
const trace_mod = @import("trace.zig");

const allocator = state_mod.allocator;
const backend_option = state_mod.backend_option;
const default_memory_kind = state_mod.default_memory_kind;
const Executable = state_mod.Executable;
const ExecutableHandle = abi.Executable(Executable);
const PjrtEventCallbackStateHandle = abi.UserData(PjrtEventCallbackState);
const PjrtEvent = events_mod.Event;
const PjRTx_RegisterCustomCallBinary = custom_call.PjRTx_RegisterCustomCallBinary;
const PjRTx_UnregisterCustomCall = custom_call.PjRTx_UnregisterCustomCall;

var api_storage: c.PJRT_Api = undefined;
var api_ready = false;

fn installScope(api: *c.PJRT_Api, comptime prefix: []const u8, comptime Api: type) void {
    inline for (@typeInfo(Api).@"struct".decls) |decl| {
        const name = prefix ++ decl.name;
        @field(api.*, name) = trace_mod.Api.Callback(name, @field(Api, decl.name)).call;
    }
}

fn installApi(api: *c.PJRT_Api) void {
    installScope(api, "PJRT_Error_", errors_mod.ErrorApi);
    installScope(api, "PJRT_Plugin_", errors_mod.PluginApi);
    installScope(api, "PJRT_Event_", events_mod.Api);
    installScope(api, "PJRT_Client_", client_mod.Api);
    installScope(api, "PJRT_AsyncHostToDeviceTransferManager_", async_h2d_mod.Api);
    installScope(api, "PJRT_TopologyDescription_", topology_mod.Api);
    installScope(api, "PJRT_DeviceDescription_", device_memory_mod.DeviceDescriptionApi);
    installScope(api, "PJRT_Device_", device_memory_mod.DeviceApi);
    installScope(api, "PJRT_Memory_", device_memory_mod.MemoryApi);
    installScope(api, "PJRT_Executable_", executable_mod.ExecutableApi);
    installScope(api, "PJRT_LoadedExecutable_", executable_mod.LoadedExecutableApi);
    installScope(api, "PJRT_LoadedExecutable_", execute_mod.LoadedExecutableApi);
    installScope(api, "PJRT_Buffer_", buffer_mod.Api);
}

fn initApi() void {
    if (api_ready) return;
    api_storage = std.mem.zeroes(c.PJRT_Api);
    api_storage.struct_size = c.PJRT_Api_STRUCT_SIZE;
    custom_call.gpu_custom_call_extension.base.next = null;
    api_storage.extension_start = @ptrCast(&custom_call.gpu_custom_call_extension.base);
    api_storage.pjrt_api_version = .{
        .struct_size = c.PJRT_Api_Version_STRUCT_SIZE,
        .extension_start = null,
        .major_version = c.PJRT_API_MAJOR,
        .minor_version = c.PJRT_API_MINOR,
    };

    installApi(&api_storage);

    api_ready = true;
}

pub const Table = struct {
    pub fn get() *const c.PJRT_Api {
        initApi();
        return &api_storage;
    }
};

fn GetPjrtApi() *const c.PJRT_Api {
    return Table.get();
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

    try std.testing.expectEqual(runtime.BackendKind.metal_mlx, abi.Client.view(create_args.client).backend_kind);
}
test "PJRT memory stats include resident executable cache bytes" {
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
    abi.Client.view(create_args.client).setExecutableCacheMaxResidentBytes(0);

    var devices_args = std.mem.zeroes(c.PJRT_Client_Devices_Args);
    devices_args.struct_size = c.PJRT_Client_Devices_Args_STRUCT_SIZE;
    devices_args.client = create_args.client;
    try expectOk(api.PJRT_Client_Devices.?(&devices_args));

    var before_stats = std.mem.zeroes(c.PJRT_Device_MemoryStats_Args);
    before_stats.struct_size = c.PJRT_Device_MemoryStats_Args_STRUCT_SIZE;
    before_stats.device = devices_args.devices[0];
    try expectOk(api.PJRT_Device_MemoryStats.?(&before_stats));
    try std.testing.expect(before_stats.peak_bytes_in_use_is_set);
    try std.testing.expect(before_stats.num_allocs_is_set);
    try std.testing.expect(before_stats.largest_alloc_size_is_set);
    if (before_stats.bytes_limit_is_set) {
        try std.testing.expect(before_stats.bytes_limit >= before_stats.bytes_in_use);
        try std.testing.expect(before_stats.largest_free_block_bytes_is_set);
        try std.testing.expect(before_stats.bytes_reservable_limit_is_set);
    }

    const module_text =
        \\module {
        \\  func.func @main() -> tensor<4xf32> {
        \\    %0 = stablehlo.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
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

    var after_compile_stats = std.mem.zeroes(c.PJRT_Device_MemoryStats_Args);
    after_compile_stats.struct_size = c.PJRT_Device_MemoryStats_Args_STRUCT_SIZE;
    after_compile_stats.device = devices_args.devices[0];
    try expectOk(api.PJRT_Device_MemoryStats.?(&after_compile_stats));
    try std.testing.expect(after_compile_stats.bytes_in_use >= before_stats.bytes_in_use + 16);
    try std.testing.expect(after_compile_stats.peak_bytes_in_use >= after_compile_stats.bytes_in_use);
    try std.testing.expect(after_compile_stats.num_allocs_is_set);
    try std.testing.expect(after_compile_stats.num_allocs >= before_stats.num_allocs + 1);
    try std.testing.expect(after_compile_stats.largest_alloc_size_is_set);
    try std.testing.expect(after_compile_stats.largest_alloc_size >= 16);
    if (after_compile_stats.bytes_limit_is_set) {
        try std.testing.expect(after_compile_stats.bytes_limit >= after_compile_stats.bytes_in_use);
        try std.testing.expect(after_compile_stats.largest_free_block_bytes_is_set);
        try std.testing.expect(after_compile_stats.bytes_reservable_limit_is_set);
    }

    var invalid_device_outputs = [_]?*c.PJRT_Buffer{null};
    var invalid_output_lists = [_][*c]?*c.PJRT_Buffer{&invalid_device_outputs};
    var wrong_arg_count_execute_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Execute_Args);
    wrong_arg_count_execute_args.struct_size = c.PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    wrong_arg_count_execute_args.executable = compile_args.executable;
    wrong_arg_count_execute_args.num_devices = 1;
    wrong_arg_count_execute_args.num_args = 1;
    wrong_arg_count_execute_args.output_lists = &invalid_output_lists;
    const wrong_arg_count_err = api.PJRT_LoadedExecutable_Execute.?(&wrong_arg_count_execute_args) orelse return error.TestUnexpectedResult;
    try std.testing.expect(std.mem.indexOf(u8, errorMessage(api, wrong_arg_count_err), "argument count") != null);
    destroyError(api, wrong_arg_count_err);

    var missing_outputs_execute_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Execute_Args);
    missing_outputs_execute_args.struct_size = c.PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    missing_outputs_execute_args.executable = compile_args.executable;
    missing_outputs_execute_args.num_devices = 1;
    missing_outputs_execute_args.num_args = 0;
    const missing_outputs_err = api.PJRT_LoadedExecutable_Execute.?(&missing_outputs_execute_args) orelse return error.TestUnexpectedResult;
    try std.testing.expect(std.mem.indexOf(u8, errorMessage(api, missing_outputs_err), "output_lists") != null);
    destroyError(api, missing_outputs_err);

    var executable_delete_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Delete_Args);
    executable_delete_args.struct_size = c.PJRT_LoadedExecutable_Delete_Args_STRUCT_SIZE;
    executable_delete_args.executable = compile_args.executable;
    try expectOk(api.PJRT_LoadedExecutable_Delete.?(&executable_delete_args));

    var executable_deleted_args = std.mem.zeroes(c.PJRT_LoadedExecutable_IsDeleted_Args);
    executable_deleted_args.struct_size = c.PJRT_LoadedExecutable_IsDeleted_Args_STRUCT_SIZE;
    executable_deleted_args.executable = compile_args.executable;
    try expectOk(api.PJRT_LoadedExecutable_IsDeleted.?(&executable_deleted_args));
    try std.testing.expect(executable_deleted_args.is_deleted);

    var device_outputs = [_]?*c.PJRT_Buffer{null};
    var output_lists = [_][*c]?*c.PJRT_Buffer{&device_outputs};
    var execute_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Execute_Args);
    execute_args.struct_size = c.PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    execute_args.executable = compile_args.executable;
    execute_args.num_devices = 1;
    execute_args.num_args = 0;
    execute_args.output_lists = &output_lists;
    const execute_err = api.PJRT_LoadedExecutable_Execute.?(&execute_args) orelse return error.TestUnexpectedResult;
    try std.testing.expect(std.mem.indexOf(u8, errorMessage(api, execute_err), "deleted") != null);
    destroyError(api, execute_err);

    var after_destroy_stats = std.mem.zeroes(c.PJRT_Device_MemoryStats_Args);
    after_destroy_stats.struct_size = c.PJRT_Device_MemoryStats_Args_STRUCT_SIZE;
    after_destroy_stats.device = devices_args.devices[0];
    try expectOk(api.PJRT_Device_MemoryStats.?(&after_destroy_stats));
    try std.testing.expectEqual(before_stats.bytes_in_use, after_destroy_stats.bytes_in_use);
    try std.testing.expectEqual(before_stats.num_allocs, after_destroy_stats.num_allocs);
    try std.testing.expect(after_destroy_stats.peak_bytes_in_use >= after_compile_stats.bytes_in_use);

    var executable_destroy_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Destroy_Args);
    executable_destroy_args.struct_size = c.PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
    executable_destroy_args.executable = compile_args.executable;
    try expectOk(api.PJRT_LoadedExecutable_Destroy.?(&executable_destroy_args));
}
test "PJRT compile trims idle resident executable cache under memory pressure" {
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

    const client = abi.Client.view(create_args.client);
    client.setExecutableCacheMaxResidentBytes(std.math.maxInt(u64));

    const large_module =
        \\module {
        \\  func.func @main() -> tensor<8xf32> {
        \\    %0 = stablehlo.constant dense<[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]> : tensor<8xf32>
        \\    return %0 : tensor<8xf32>
        \\  }
        \\}
    ;
    var large_program = std.mem.zeroes(c.PJRT_Program);
    large_program.struct_size = c.PJRT_Program_STRUCT_SIZE;
    large_program.code = @constCast(large_module.ptr);
    large_program.code_size = large_module.len;
    large_program.format = "mlir";
    large_program.format_size = "mlir".len;

    var large_compile_args = std.mem.zeroes(c.PJRT_Client_Compile_Args);
    large_compile_args.struct_size = c.PJRT_Client_Compile_Args_STRUCT_SIZE;
    large_compile_args.client = create_args.client;
    large_compile_args.program = &large_program;
    try expectOk(api.PJRT_Client_Compile.?(&large_compile_args));
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Destroy_Args);
        destroy_args.struct_size = c.PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
        destroy_args.executable = large_compile_args.executable;
        _ = api.PJRT_LoadedExecutable_Destroy.?(&destroy_args);
    }

    const large_resident_bytes = client.executable_cache.stats.resident_bytes;
    try std.testing.expect(large_resident_bytes >= 32);
    try std.testing.expectEqual(@as(u64, 1), client.executable_cache.stats.resident_entries);

    var large_delete_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Delete_Args);
    large_delete_args.struct_size = c.PJRT_LoadedExecutable_Delete_Args_STRUCT_SIZE;
    large_delete_args.executable = large_compile_args.executable;
    try expectOk(api.PJRT_LoadedExecutable_Delete.?(&large_delete_args));
    try std.testing.expectEqual(large_resident_bytes, client.executable_cache.stats.resident_bytes);

    client.memories[0].stats.capacity_bytes = 4;

    const small_module =
        \\module {
        \\  func.func @main() -> tensor<1xf32> {
        \\    %0 = stablehlo.constant dense<[9.0]> : tensor<1xf32>
        \\    return %0 : tensor<1xf32>
        \\  }
        \\}
    ;
    var small_program = std.mem.zeroes(c.PJRT_Program);
    small_program.struct_size = c.PJRT_Program_STRUCT_SIZE;
    small_program.code = @constCast(small_module.ptr);
    small_program.code_size = small_module.len;
    small_program.format = "mlir";
    small_program.format_size = "mlir".len;

    var small_compile_args = std.mem.zeroes(c.PJRT_Client_Compile_Args);
    small_compile_args.struct_size = c.PJRT_Client_Compile_Args_STRUCT_SIZE;
    small_compile_args.client = create_args.client;
    small_compile_args.program = &small_program;
    try expectOk(api.PJRT_Client_Compile.?(&small_compile_args));
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Destroy_Args);
        destroy_args.struct_size = c.PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
        destroy_args.executable = small_compile_args.executable;
        _ = api.PJRT_LoadedExecutable_Destroy.?(&destroy_args);
    }

    const small_executable = ExecutableHandle.view(small_compile_args.executable);
    const small_resident_bytes = small_executable.graph.backendExecutableStats().?.resident_constant_bytes;
    try std.testing.expectEqual(@as(u64, 4), small_resident_bytes);
    try std.testing.expectEqual(@as(u64, 4), small_executable.graph.last_compile_cache_trim.requested_bytes);
    try std.testing.expectEqual(@as(u64, 0), small_executable.graph.last_compile_cache_trim.target_resident_bytes);
    try std.testing.expectEqual(large_resident_bytes, small_executable.graph.last_compile_cache_trim.freed_bytes);
    try std.testing.expectEqual(@as(u64, 1), small_executable.graph.last_compile_cache_trim.evicted_entries);
    try std.testing.expectEqual(@as(u64, 0), small_executable.graph.last_compile_cache_trim.remaining_resident_bytes);
    try std.testing.expect(!small_executable.graph.last_compile_cache_trim.still_over_capacity);
    try std.testing.expectEqual(@as(u64, 1), client.executable_cache.stats.pressure_trim_requests);
    try std.testing.expectEqual(large_resident_bytes, client.executable_cache.stats.pressure_trimmed_bytes);
    try std.testing.expectEqual(@as(u64, 0), client.executable_cache.stats.pressure_trim_failures);
    try std.testing.expectEqual(@as(u64, 1), client.executable_cache.stats.resident_entries);
    try std.testing.expectEqual(@as(u64, 4), client.executable_cache.stats.resident_bytes);
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
const PjrtEventCallbackState = struct {
    count: usize = 0,
    ready_count: usize = 0,
    error_count: usize = 0,
    saw_expected_message: bool = false,
};
fn testPjrtEventCallback(err: [*c]c.PJRT_Error, user_arg: ?*anyopaque) callconv(.c) void {
    const state = PjrtEventCallbackStateHandle.view(user_arg);
    state.count += 1;
    if (err) |actual| {
        state.error_count += 1;
        const api = GetPjrtApi();
        const message = errorMessage(api, actual);
        if (std.mem.indexOf(u8, message, "buffer has been deleted") != null) {
            state.saw_expected_message = true;
        }
        destroyError(api, actual);
    } else {
        state.ready_count += 1;
    }
}
test "event callbacks bridge PJRT errors and pending runtime transitions" {
    const api = GetPjrtApi();

    const ready_event = PjrtEvent.ready().?;
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
        destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
        destroy_args.event = ready_event;
        _ = api.PJRT_Event_Destroy.?(&destroy_args);
    }
    var ready_state = PjrtEventCallbackState{};
    var ready_on_ready_args = std.mem.zeroes(c.PJRT_Event_OnReady_Args);
    ready_on_ready_args.struct_size = c.PJRT_Event_OnReady_Args_STRUCT_SIZE;
    ready_on_ready_args.event = ready_event;
    ready_on_ready_args.callback = testPjrtEventCallback;
    ready_on_ready_args.user_arg = &ready_state;
    try expectOk(api.PJRT_Event_OnReady.?(&ready_on_ready_args));
    try std.testing.expectEqual(@as(usize, 1), ready_state.count);
    try std.testing.expectEqual(@as(usize, 1), ready_state.ready_count);

    const failed_event = PjrtEvent.failed("buffer has been deleted").?;
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
        destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
        destroy_args.event = failed_event;
        _ = api.PJRT_Event_Destroy.?(&destroy_args);
    }
    var failed_state = PjrtEventCallbackState{};
    var failed_on_ready_args = std.mem.zeroes(c.PJRT_Event_OnReady_Args);
    failed_on_ready_args.struct_size = c.PJRT_Event_OnReady_Args_STRUCT_SIZE;
    failed_on_ready_args.event = failed_event;
    failed_on_ready_args.callback = testPjrtEventCallback;
    failed_on_ready_args.user_arg = &failed_state;
    try expectOk(api.PJRT_Event_OnReady.?(&failed_on_ready_args));
    try std.testing.expectEqual(@as(usize, 1), failed_state.count);
    try std.testing.expectEqual(@as(usize, 1), failed_state.error_count);
    try std.testing.expect(failed_state.saw_expected_message);

    const pending_runtime_event = allocator.create(runtime.Event) catch return error.OutOfMemory;
    pending_runtime_event.* = runtime.Event.pending();
    const pending_event = abi.Event.handle(pending_runtime_event);
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
        destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
        destroy_args.event = pending_event;
        _ = api.PJRT_Event_Destroy.?(&destroy_args);
    }
    var pending_state = PjrtEventCallbackState{};
    var pending_on_ready_args = std.mem.zeroes(c.PJRT_Event_OnReady_Args);
    pending_on_ready_args.struct_size = c.PJRT_Event_OnReady_Args_STRUCT_SIZE;
    pending_on_ready_args.event = pending_event;
    pending_on_ready_args.callback = testPjrtEventCallback;
    pending_on_ready_args.user_arg = &pending_state;
    try expectOk(api.PJRT_Event_OnReady.?(&pending_on_ready_args));
    try std.testing.expectEqual(@as(usize, 0), pending_state.count);
    pending_runtime_event.setReady();
    try std.testing.expectEqual(@as(usize, 1), pending_state.count);
    try std.testing.expectEqual(@as(usize, 1), pending_state.ready_count);
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

    const async_dims = [_]i64{4};
    var async_shape = std.mem.zeroes(c.PJRT_ShapeSpec);
    async_shape.struct_size = c.PJRT_ShapeSpec_STRUCT_SIZE;
    async_shape.dims = &async_dims;
    async_shape.num_dims = async_dims.len;
    async_shape.element_type = c.PJRT_Buffer_Type_U8;
    var async_create_args = std.mem.zeroes(c.PJRT_Client_CreateBuffersForAsyncHostToDevice_Args);
    async_create_args.struct_size = c.PJRT_Client_CreateBuffersForAsyncHostToDevice_Args_STRUCT_SIZE;
    async_create_args.client = create_args.client;
    async_create_args.shape_specs = &async_shape;
    async_create_args.num_shape_specs = 1;
    async_create_args.memory = default_memory_args.memory;
    try expectOk(api.PJRT_Client_CreateBuffersForAsyncHostToDevice.?(&async_create_args));
    defer {
        if (async_create_args.transfer_manager) |manager| {
            var destroy_transfer_args = std.mem.zeroes(c.PJRT_AsyncHostToDeviceTransferManager_Destroy_Args);
            destroy_transfer_args.struct_size = c.PJRT_AsyncHostToDeviceTransferManager_Destroy_Args_STRUCT_SIZE;
            destroy_transfer_args.transfer_manager = manager;
            _ = api.PJRT_AsyncHostToDeviceTransferManager_Destroy.?(&destroy_transfer_args);
        }
    }

    var async_size_args = std.mem.zeroes(c.PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args);
    async_size_args.struct_size = c.PJRT_AsyncHostToDeviceTransferManager_BufferSize_Args_STRUCT_SIZE;
    async_size_args.transfer_manager = async_create_args.transfer_manager;
    async_size_args.buffer_index = 0;
    try expectOk(api.PJRT_AsyncHostToDeviceTransferManager_BufferSize.?(&async_size_args));
    try std.testing.expectEqual(@as(usize, 4), async_size_args.buffer_size);

    var async_retrieve_args = std.mem.zeroes(c.PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args);
    async_retrieve_args.struct_size = c.PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer_Args_STRUCT_SIZE;
    async_retrieve_args.transfer_manager = async_create_args.transfer_manager;
    async_retrieve_args.buffer_index = 0;
    try expectOk(api.PJRT_AsyncHostToDeviceTransferManager_RetrieveBuffer.?(&async_retrieve_args));
    defer {
        var destroy_buffer_args = std.mem.zeroes(c.PJRT_Buffer_Destroy_Args);
        destroy_buffer_args.struct_size = c.PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        destroy_buffer_args.buffer = async_retrieve_args.buffer_out;
        _ = api.PJRT_Buffer_Destroy.?(&destroy_buffer_args);
    }

    const async_input = [_]u8{ 9, 8, 7, 6 };
    var async_transfer_args = std.mem.zeroes(c.PJRT_AsyncHostToDeviceTransferManager_TransferData_Args);
    async_transfer_args.struct_size = c.PJRT_AsyncHostToDeviceTransferManager_TransferData_Args_STRUCT_SIZE;
    async_transfer_args.transfer_manager = async_create_args.transfer_manager;
    async_transfer_args.buffer_index = 0;
    async_transfer_args.data = &async_input;
    async_transfer_args.offset = 0;
    async_transfer_args.transfer_size = 2;
    async_transfer_args.is_last_transfer = false;
    try expectOk(api.PJRT_AsyncHostToDeviceTransferManager_TransferData.?(&async_transfer_args));
    if (async_transfer_args.done_with_h2d_transfer) |event| {
        var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
        event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
        event_destroy_args.event = event;
        _ = api.PJRT_Event_Destroy.?(&event_destroy_args);
    }
    async_transfer_args.done_with_h2d_transfer = null;
    async_transfer_args.data = &async_input[2];
    async_transfer_args.offset = 2;
    async_transfer_args.transfer_size = 2;
    async_transfer_args.is_last_transfer = true;
    try expectOk(api.PJRT_AsyncHostToDeviceTransferManager_TransferData.?(&async_transfer_args));
    if (async_transfer_args.done_with_h2d_transfer) |event| {
        var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
        event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
        event_destroy_args.event = event;
        _ = api.PJRT_Event_Destroy.?(&event_destroy_args);
    }

    var async_output: [4]u8 = undefined;
    var async_to_host_args = std.mem.zeroes(c.PJRT_Buffer_ToHostBuffer_Args);
    async_to_host_args.struct_size = c.PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    async_to_host_args.src = async_retrieve_args.buffer_out;
    async_to_host_args.dst = &async_output;
    async_to_host_args.dst_size = async_output.len;
    try expectOk(api.PJRT_Buffer_ToHostBuffer.?(&async_to_host_args));
    try std.testing.expectEqualSlices(u8, &async_input, &async_output);

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

    var memory_stats_before_delete = std.mem.zeroes(c.PJRT_Device_MemoryStats_Args);
    memory_stats_before_delete.struct_size = c.PJRT_Device_MemoryStats_Args_STRUCT_SIZE;
    memory_stats_before_delete.device = devices_args.devices[0];
    try expectOk(api.PJRT_Device_MemoryStats.?(&memory_stats_before_delete));
    try std.testing.expect(memory_stats_before_delete.bytes_in_use >= input.len);

    var delete_args = std.mem.zeroes(c.PJRT_Buffer_Delete_Args);
    delete_args.struct_size = c.PJRT_Buffer_Delete_Args_STRUCT_SIZE;
    delete_args.buffer = from_host_args.buffer;
    try expectOk(api.PJRT_Buffer_Delete.?(&delete_args));

    var memory_stats_after_delete = std.mem.zeroes(c.PJRT_Device_MemoryStats_Args);
    memory_stats_after_delete.struct_size = c.PJRT_Device_MemoryStats_Args_STRUCT_SIZE;
    memory_stats_after_delete.device = devices_args.devices[0];
    try expectOk(api.PJRT_Device_MemoryStats.?(&memory_stats_after_delete));
    try std.testing.expectEqual(memory_stats_before_delete.bytes_in_use - @as(i64, @intCast(input.len)), memory_stats_after_delete.bytes_in_use);

    var ready_event_args = std.mem.zeroes(c.PJRT_Buffer_ReadyEvent_Args);
    ready_event_args.struct_size = c.PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE;
    ready_event_args.buffer = from_host_args.buffer;
    try expectOk(api.PJRT_Buffer_ReadyEvent.?(&ready_event_args));
    defer if (ready_event_args.event) |event| {
        var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
        event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
        event_destroy_args.event = event;
        _ = api.PJRT_Event_Destroy.?(&event_destroy_args);
    };
    var ready_event_error_args = std.mem.zeroes(c.PJRT_Event_Error_Args);
    ready_event_error_args.struct_size = c.PJRT_Event_Error_Args_STRUCT_SIZE;
    ready_event_error_args.event = ready_event_args.event;
    const ready_err = api.PJRT_Event_Error.?(&ready_event_error_args) orelse return error.TestUnexpectedResult;
    defer destroyError(api, ready_err);
    try std.testing.expect(std.mem.indexOf(u8, errorMessage(api, ready_err), "deleted") != null);

    var output: [4]u8 = undefined;
    var to_host_args = std.mem.zeroes(c.PJRT_Buffer_ToHostBuffer_Args);
    to_host_args.struct_size = c.PJRT_Buffer_ToHostBuffer_Args_STRUCT_SIZE;
    to_host_args.src = from_host_args.buffer;
    to_host_args.dst = &output;
    to_host_args.dst_size = output.len;
    const to_host_err = api.PJRT_Buffer_ToHostBuffer.?(&to_host_args) orelse return error.TestUnexpectedResult;
    defer destroyError(api, to_host_err);
    try std.testing.expect(std.mem.indexOf(u8, errorMessage(api, to_host_err), "deleted or donated") != null);
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
    ExecutableHandle.view(compile_args.executable).plan.donated_parameter_indices = try allocator.dupe(u32, &.{0});
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Destroy_Args);
        destroy_args.struct_size = c.PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE;
        destroy_args.executable = compile_args.executable;
        _ = api.PJRT_LoadedExecutable_Destroy.?(&destroy_args);
    }

    var deleted_from_host_args = std.mem.zeroes(c.PJRT_Client_BufferFromHostBuffer_Args);
    deleted_from_host_args.struct_size = c.PJRT_Client_BufferFromHostBuffer_Args_STRUCT_SIZE;
    deleted_from_host_args.client = create_args.client;
    deleted_from_host_args.data = &lhs;
    deleted_from_host_args.type = c.PJRT_Buffer_Type_S8;
    deleted_from_host_args.dims = &dims;
    deleted_from_host_args.num_dims = dims.len;
    deleted_from_host_args.device = devices_args.devices[0];
    deleted_from_host_args.memory = default_memory_args.memory;
    try expectOk(api.PJRT_Client_BufferFromHostBuffer.?(&deleted_from_host_args));
    defer {
        if (deleted_from_host_args.done_with_host_buffer) |event| {
            var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
            event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
            event_destroy_args.event = event;
            _ = api.PJRT_Event_Destroy.?(&event_destroy_args);
        }
        var buffer_destroy_args = std.mem.zeroes(c.PJRT_Buffer_Destroy_Args);
        buffer_destroy_args.struct_size = c.PJRT_Buffer_Destroy_Args_STRUCT_SIZE;
        buffer_destroy_args.buffer = deleted_from_host_args.buffer;
        _ = api.PJRT_Buffer_Destroy.?(&buffer_destroy_args);
    }
    var delete_buffer_args = std.mem.zeroes(c.PJRT_Buffer_Delete_Args);
    delete_buffer_args.struct_size = c.PJRT_Buffer_Delete_Args_STRUCT_SIZE;
    delete_buffer_args.buffer = deleted_from_host_args.buffer;
    try expectOk(api.PJRT_Buffer_Delete.?(&delete_buffer_args));

    var bad_device_args = [_]?*c.PJRT_Buffer{ deleted_from_host_args.buffer, rhs_from_host_args.buffer };
    var bad_argument_lists = [_][*c]const ?*c.PJRT_Buffer{&bad_device_args};
    var bad_device_outputs = [_]?*c.PJRT_Buffer{null};
    var bad_output_lists = [_][*c]?*c.PJRT_Buffer{&bad_device_outputs};
    var bad_device_events = [_]?*c.PJRT_Event{null};
    var bad_execute_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Execute_Args);
    bad_execute_args.struct_size = c.PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    bad_execute_args.executable = compile_args.executable;
    bad_execute_args.argument_lists = &bad_argument_lists;
    bad_execute_args.num_devices = 1;
    bad_execute_args.num_args = bad_device_args.len;
    bad_execute_args.output_lists = &bad_output_lists;
    bad_execute_args.device_complete_events = &bad_device_events;
    const bad_execute_err = api.PJRT_LoadedExecutable_Execute.?(&bad_execute_args) orelse return error.TestUnexpectedResult;
    try std.testing.expect(std.mem.indexOf(u8, errorMessage(api, bad_execute_err), "deleted buffer") != null);
    destroyError(api, bad_execute_err);
    try std.testing.expectEqual(@as(?*c.PJRT_Buffer, null), bad_device_outputs[0]);
    try std.testing.expectEqual(@as(?*c.PJRT_Event, null), bad_device_events[0]);

    var alias_device_args = [_]?*c.PJRT_Buffer{ lhs_from_host_args.buffer, lhs_from_host_args.buffer };
    var alias_argument_lists = [_][*c]const ?*c.PJRT_Buffer{&alias_device_args};
    var alias_device_outputs = [_]?*c.PJRT_Buffer{null};
    var alias_output_lists = [_][*c]?*c.PJRT_Buffer{&alias_device_outputs};
    var alias_device_events = [_]?*c.PJRT_Event{null};
    var alias_execute_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Execute_Args);
    alias_execute_args.struct_size = c.PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
    alias_execute_args.executable = compile_args.executable;
    alias_execute_args.argument_lists = &alias_argument_lists;
    alias_execute_args.num_devices = 1;
    alias_execute_args.num_args = alias_device_args.len;
    alias_execute_args.output_lists = &alias_output_lists;
    alias_execute_args.device_complete_events = &alias_device_events;
    const alias_execute_err = api.PJRT_LoadedExecutable_Execute.?(&alias_execute_args) orelse return error.TestUnexpectedResult;
    try std.testing.expect(std.mem.indexOf(u8, errorMessage(api, alias_execute_err), "aliases another argument") != null);
    destroyError(api, alias_execute_err);
    try std.testing.expectEqual(@as(?*c.PJRT_Buffer, null), alias_device_outputs[0]);
    try std.testing.expectEqual(@as(?*c.PJRT_Event, null), alias_device_events[0]);

    var lhs_after_alias_args = std.mem.zeroes(c.PJRT_Buffer_IsDeleted_Args);
    lhs_after_alias_args.struct_size = c.PJRT_Buffer_IsDeleted_Args_STRUCT_SIZE;
    lhs_after_alias_args.buffer = lhs_from_host_args.buffer;
    try expectOk(api.PJRT_Buffer_IsDeleted.?(&lhs_after_alias_args));
    try std.testing.expect(!lhs_after_alias_args.is_deleted);

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
    var execute_event_ready_args = std.mem.zeroes(c.PJRT_Event_IsReady_Args);
    execute_event_ready_args.struct_size = c.PJRT_Event_IsReady_Args_STRUCT_SIZE;
    execute_event_ready_args.event = device_events[0];
    try expectOk(api.PJRT_Event_IsReady.?(&execute_event_ready_args));
    try std.testing.expect(execute_event_ready_args.is_ready);
    var execute_event_await_args = std.mem.zeroes(c.PJRT_Event_Await_Args);
    execute_event_await_args.struct_size = c.PJRT_Event_Await_Args_STRUCT_SIZE;
    execute_event_await_args.event = device_events[0];
    try expectOk(api.PJRT_Event_Await.?(&execute_event_await_args));
    var execute_event_callback_state = PjrtEventCallbackState{};
    var execute_event_on_ready_args = std.mem.zeroes(c.PJRT_Event_OnReady_Args);
    execute_event_on_ready_args.struct_size = c.PJRT_Event_OnReady_Args_STRUCT_SIZE;
    execute_event_on_ready_args.event = device_events[0];
    execute_event_on_ready_args.callback = testPjrtEventCallback;
    execute_event_on_ready_args.user_arg = &execute_event_callback_state;
    try expectOk(api.PJRT_Event_OnReady.?(&execute_event_on_ready_args));
    try std.testing.expectEqual(@as(usize, 1), execute_event_callback_state.ready_count);
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

    var lhs_deleted_args = std.mem.zeroes(c.PJRT_Buffer_IsDeleted_Args);
    lhs_deleted_args.struct_size = c.PJRT_Buffer_IsDeleted_Args_STRUCT_SIZE;
    lhs_deleted_args.buffer = lhs_from_host_args.buffer;
    try expectOk(api.PJRT_Buffer_IsDeleted.?(&lhs_deleted_args));
    try std.testing.expect(lhs_deleted_args.is_deleted);

    var rhs_deleted_args = std.mem.zeroes(c.PJRT_Buffer_IsDeleted_Args);
    rhs_deleted_args.struct_size = c.PJRT_Buffer_IsDeleted_Args_STRUCT_SIZE;
    rhs_deleted_args.buffer = rhs_from_host_args.buffer;
    try expectOk(api.PJRT_Buffer_IsDeleted.?(&rhs_deleted_args));
    try std.testing.expect(!rhs_deleted_args.is_deleted);

    try std.testing.expect(device_outputs[0] != null);
    var output_ready_event_args = std.mem.zeroes(c.PJRT_Buffer_ReadyEvent_Args);
    output_ready_event_args.struct_size = c.PJRT_Buffer_ReadyEvent_Args_STRUCT_SIZE;
    output_ready_event_args.buffer = device_outputs[0];
    try expectOk(api.PJRT_Buffer_ReadyEvent.?(&output_ready_event_args));
    defer if (output_ready_event_args.event) |event| {
        var event_destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
        event_destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
        event_destroy_args.event = event;
        _ = api.PJRT_Event_Destroy.?(&event_destroy_args);
    };
    var output_ready_await_args = std.mem.zeroes(c.PJRT_Event_Await_Args);
    output_ready_await_args.struct_size = c.PJRT_Event_Await_Args_STRUCT_SIZE;
    output_ready_await_args.event = output_ready_event_args.event;
    try expectOk(api.PJRT_Event_Await.?(&output_ready_await_args));
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
test "loaded executable execute chains f32 unary and metadata custom_call through PJRT argument lists" {
    const api = GetPjrtApi();
    const registered_add_target = "pjrtx.test.plugin_binary_add";
    const add_op = "add";
    try expectOk(PjRTx_RegisterCustomCallBinary(registered_add_target.ptr, registered_add_target.len, add_op.ptr, add_op.len));
    defer PjRTx_UnregisterCustomCall(registered_add_target.ptr, registered_add_target.len);

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
        \\    %0 = "stablehlo.custom_call"(%arg0, %arg1) {call_target_name = "pjrtx.test.plugin_binary_add"} : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
        \\    %1 = stablehlo.sqrt %0 : tensor<2xf32>
        \\    %2 = "stablehlo.custom_call"(%1) {call_target_name = "annotate_device_placement"} : (tensor<2xf32>) -> tensor<2xf32>
        \\    return %2 : tensor<2xf32>
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
        \\func.func @main(%arg0: tensor<f32>) -> tensor<4xf32> {
        \\  %0 = "stablehlo.broadcast"(%arg0) {
        \\    broadcast_sizes = array<i64: 4>,
        \\    sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>
        \\  } : (tensor<f32>) -> tensor<4xf32>
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
    try std.testing.expect(std.mem.indexOf(u8, message, "op=broadcast") != null);
    try std.testing.expect(std.mem.indexOf(u8, message, "dtype=f32") != null);
    try std.testing.expect(std.mem.indexOf(u8, message, "rank=1") != null);
    try std.testing.expect(std.mem.indexOf(u8, message, "sharding=sdy.sharding_per_value") != null);
}
test "compile rejects frontend-supported op without MLX executable lowering" {
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
        \\func.func @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
        \\  %0 = "stablehlo.custom_call"(%arg0) {call_target_name = "pjrtx.test"} : (tensor<2x2xf32>) -> tensor<2x2xf32>
        \\  return %0 : tensor<2x2xf32>
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
    try std.testing.expect(std.mem.indexOf(u8, message, "pass=mlx-backend-legalization") != null);
    try std.testing.expect(std.mem.indexOf(u8, message, "op=custom_call") != null);
    try std.testing.expect(std.mem.indexOf(u8, message, "feature=pjrtx.test") != null);
}
