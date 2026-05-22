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
    Executable.Testing.setDonatedParameters(LoadedExecutableHandle.ref(compile_args.executable), try plugin_process.allocator().dupe(u32, &.{0}));
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
    var execute_event_callback_state = EventCallbackRecord{};
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
