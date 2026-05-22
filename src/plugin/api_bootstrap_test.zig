const std = @import("std");

const h = @import("pjrt_api_test_harness.zig");
const c = h.c;
const runtime = h.runtime;
const handles = h.handles;
const plugin_process = h.plugin_process;
const backend_option = h.backend_option;
const default_memory_kind = h.default_memory_kind;
const Executable = h.Executable;
const LoadedExecutableHandle = h.LoadedExecutableHandle;
const PjrtEvent = h.PjrtEvent;
const PjrtError = h.PjrtError;
const PjRTx_RegisterCustomCallBinary = h.PjRTx_RegisterCustomCallBinary;
const PjRTx_UnregisterCustomCall = h.PjRTx_UnregisterCustomCall;
const GetPjrtApi = h.GetPjrtApi;
const expectOk = h.expectOk;
const destroyError = h.destroyError;
const errorMessage = h.errorMessage;
const EventCallbackRecord = h.EventCallbackRecord;
const EventCallbackRecordHandle = h.EventCallbackRecordHandle;
const testPjrtEventCallback = h.testPjrtEventCallback;

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

    try std.testing.expect(handles.Client.ref(create_args.client).deviceCount() != 0);
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
    handles.Client.ref(create_args.client).setExecutableCacheMaxResidentBytes(0);

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

    const client = handles.Client.ref(create_args.client);
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

    const large_resident_bytes = client.executableCacheStats().resident_bytes;
    try std.testing.expect(large_resident_bytes >= 32);
    try std.testing.expectEqual(@as(u64, 1), client.executableCacheStats().resident_entries);

    var large_delete_args = std.mem.zeroes(c.PJRT_LoadedExecutable_Delete_Args);
    large_delete_args.struct_size = c.PJRT_LoadedExecutable_Delete_Args_STRUCT_SIZE;
    large_delete_args.executable = large_compile_args.executable;
    try expectOk(api.PJRT_LoadedExecutable_Delete.?(&large_delete_args));
    try std.testing.expectEqual(large_resident_bytes, client.executableCacheStats().resident_bytes);

    client.defaultMemory().stats.capacity_bytes = 4;

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

    const small_executable = LoadedExecutableHandle.ref(small_compile_args.executable);
    const small_resident_bytes = Executable.Testing.backendExecutableStats(small_executable).?.resident_constant_bytes;
    const small_trim = Executable.Testing.lastCompileCacheTrim(small_executable);
    try std.testing.expectEqual(@as(u64, 4), small_resident_bytes);
    try std.testing.expectEqual(@as(u64, 4), small_trim.requested_bytes);
    try std.testing.expectEqual(@as(u64, 0), small_trim.target_resident_bytes);
    try std.testing.expectEqual(large_resident_bytes, small_trim.freed_bytes);
    try std.testing.expectEqual(@as(u64, 1), small_trim.evicted_entries);
    try std.testing.expectEqual(@as(u64, 0), small_trim.remaining_resident_bytes);
    try std.testing.expect(!small_trim.still_over_capacity);
    try std.testing.expectEqual(@as(u64, 1), client.executableCacheStats().pressure_trim_requests);
    try std.testing.expectEqual(large_resident_bytes, client.executableCacheStats().pressure_trimmed_bytes);
    try std.testing.expectEqual(@as(u64, 0), client.executableCacheStats().pressure_trim_failures);
    try std.testing.expectEqual(@as(u64, 1), client.executableCacheStats().resident_entries);
    try std.testing.expectEqual(@as(u64, 4), client.executableCacheStats().resident_bytes);
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
    var ready_state = EventCallbackRecord{};
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
    var failed_state = EventCallbackRecord{};
    var failed_on_ready_args = std.mem.zeroes(c.PJRT_Event_OnReady_Args);
    failed_on_ready_args.struct_size = c.PJRT_Event_OnReady_Args_STRUCT_SIZE;
    failed_on_ready_args.event = failed_event;
    failed_on_ready_args.callback = testPjrtEventCallback;
    failed_on_ready_args.user_arg = &failed_state;
    try expectOk(api.PJRT_Event_OnReady.?(&failed_on_ready_args));
    try std.testing.expectEqual(@as(usize, 1), failed_state.count);
    try std.testing.expectEqual(@as(usize, 1), failed_state.error_count);
    try std.testing.expect(failed_state.saw_expected_message);

    const pending_runtime_event = plugin_process.allocator().create(runtime.Event) catch return error.OutOfMemory;
    pending_runtime_event.* = runtime.Event.pending();
    const pending_event = handles.Event.handle(pending_runtime_event);
    defer {
        var destroy_args = std.mem.zeroes(c.PJRT_Event_Destroy_Args);
        destroy_args.struct_size = c.PJRT_Event_Destroy_Args_STRUCT_SIZE;
        destroy_args.event = pending_event;
        _ = api.PJRT_Event_Destroy.?(&destroy_args);
    }
    var pending_state = EventCallbackRecord{};
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
