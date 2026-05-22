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
