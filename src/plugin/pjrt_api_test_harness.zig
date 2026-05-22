const std = @import("std");

pub const c = @import("c");
pub const runtime = @import("src/runtime");

const custom_call = @import("custom_call.zig");
const errors_mod = @import("errors.zig");
const events_mod = @import("events.zig");
const executable_mod = @import("executable.zig");
pub const handles = @import("pjrt_handles.zig");
pub const plugin_process = @import("plugin_process.zig");
const api_mod = @import("api.zig");

pub const backend_option = plugin_process.Options.backend;
pub const default_memory_kind = plugin_process.MemoryKinds.device;
pub const Executable = executable_mod.Executable;
pub const LoadedExecutableHandle = handles.LoadedExecutable(Executable);
pub const PjrtEvent = events_mod.Event;
pub const PjrtError = errors_mod.Error;
pub const PjRTx_RegisterCustomCallBinary = custom_call.PjRTx_RegisterCustomCallBinary;
pub const PjRTx_UnregisterCustomCall = custom_call.PjRTx_UnregisterCustomCall;

/// Typed user-argument handle for event callback test state.
pub const EventCallbackRecordHandle = handles.UserArg(EventCallbackRecord);

/// Returns the process-wide PJRT API table used by plugin API tests.
pub fn GetPjrtApi() *const c.PJRT_Api {
    return api_mod.Table.get();
}

/// Fails the current test when a PJRT call returns an error.
pub fn expectOk(err: [*c]c.PJRT_Error) !void {
    if (err) |actual| {
        const api = GetPjrtApi();
        std.debug.print("unexpected PJRT error: {s}\n", .{errorMessage(api, actual)});
        destroyError(api, actual);
        return error.TestUnexpectedResult;
    }
}

/// Releases a PJRT error returned by the API under test.
pub fn destroyError(api: *const c.PJRT_Api, err: [*c]c.PJRT_Error) void {
    var destroy_args = std.mem.zeroes(c.PJRT_Error_Destroy_Args);
    destroy_args.struct_size = c.PJRT_Error_Destroy_Args_STRUCT_SIZE;
    destroy_args.@"error" = err;
    api.PJRT_Error_Destroy.?(&destroy_args);
}

/// Returns the message owned by a PJRT error object.
pub fn errorMessage(api: *const c.PJRT_Api, err: [*c]c.PJRT_Error) []const u8 {
    var message_args = std.mem.zeroes(c.PJRT_Error_Message_Args);
    message_args.struct_size = c.PJRT_Error_Message_Args_STRUCT_SIZE;
    message_args.@"error" = err;
    api.PJRT_Error_Message.?(&message_args);
    return message_args.message[0..message_args.message_size];
}

/// Records callback observations for PJRT event bridge tests.
pub const EventCallbackRecord = struct {
    count: usize = 0,
    ready_count: usize = 0,
    error_count: usize = 0,
    saw_expected_message: bool = false,
};

/// PJRT event callback used by event bridge tests.
pub fn testPjrtEventCallback(err: [*c]c.PJRT_Error, user_arg: ?*anyopaque) callconv(.c) void {
    const state = EventCallbackRecordHandle.ref(user_arg);
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
