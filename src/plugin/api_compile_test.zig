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
