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
