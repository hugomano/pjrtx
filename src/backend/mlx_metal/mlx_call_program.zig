const std = @import("std");
const c = @import("c");

const raw = @import("mlx_call_raw.zig");

const AsyncTransferHandle = raw.AsyncTransferHandle;
const BufferHandle = raw.BufferHandle;
const Dtype = raw.Dtype;
const ProgramBuildAdapter = raw.ProgramBuildAdapter;
const ProgramBuildCallback = raw.ProgramBuildCallback;
const ProgramHandle = raw.ProgramHandle;
const ProgramOutputs = raw.ProgramOutputs;
const code = raw.code;
const flag = raw.flag;
const fromRawBuffer = raw.fromRawBuffer;
const fromRawProgram = raw.fromRawProgram;
const fromRawTransfer = raw.fromRawTransfer;
const rawBufferList = raw.rawBufferList;
const rawProgram = raw.rawProgram;
const rawTransfer = raw.rawTransfer;

/// Creates an MLX/Metal async host-to-device transfer.
pub fn asyncH2DCreate(device_ordinal: i32, dtype: Dtype, dims: []const i64, byte_size: usize) ?AsyncTransferHandle {
    return fromRawTransfer(c.pjrtx_mlx_metal_async_h2d_create(device_ordinal, code(dtype), dims.ptr, dims.len, byte_size));
}

/// Writes one byte range into an MLX/Metal async host-to-device transfer.
pub fn asyncH2DWrite(transfer: AsyncTransferHandle, offset: usize, src: []const u8) bool {
    return c.pjrtx_mlx_metal_async_h2d_write(rawTransfer(transfer), offset, src.ptr, src.len) != 0;
}

/// Finishes an MLX/Metal async host-to-device transfer and returns its buffer.
pub fn asyncH2DFinish(transfer: AsyncTransferHandle) ?BufferHandle {
    return fromRawBuffer(c.pjrtx_mlx_metal_async_h2d_finish(rawTransfer(transfer)));
}

/// Destroys an MLX/Metal async host-to-device transfer.
pub fn asyncH2DDestroy(transfer: AsyncTransferHandle) void {
    c.pjrtx_mlx_metal_async_h2d_destroy(rawTransfer(transfer));
}

/// Creates an MLX compiled program with every argument dynamic.
pub fn programCreate(
    user_data: ?*anyopaque,
    input_count: usize,
    output_count: usize,
    comptime callback: ProgramBuildCallback,
) ?ProgramHandle {
    return fromRawProgram(c.pjrtx_mlx_metal_program_create(
        user_data,
        input_count,
        output_count,
        ProgramBuildAdapter(callback).call,
    ));
}

/// Creates an MLX compiled program with stable captured inputs and dynamic indices.
pub fn programCreateWithCaptures(
    user_data: ?*anyopaque,
    input_count: usize,
    output_count: usize,
    comptime callback: ProgramBuildCallback,
    arguments: []const BufferHandle,
    dynamic_indices: []const u64,
) ?ProgramHandle {
    return fromRawProgram(c.pjrtx_mlx_metal_program_create_with_captures(
        user_data,
        input_count,
        output_count,
        ProgramBuildAdapter(callback).call,
        rawBufferList(arguments),
        dynamic_indices.ptr,
        dynamic_indices.len,
    ));
}

/// Executes an MLX compiled program with optional donated input indices.
pub fn programExecuteWithDonation(program: ProgramHandle, arguments: []const BufferHandle, donated_input_indices: []const u64) ?ProgramOutputs {
    var raw_outputs: [*c]?*c.PjrtxMlxMetalBuffer = null;
    var raw_output_count: u64 = 0;
    var raw_profile: c.PjrtxMlxMetalProgramExecuteProfile = std.mem.zeroes(c.PjrtxMlxMetalProgramExecuteProfile);
    const ok = c.pjrtx_mlx_metal_program_execute_with_donation(
        rawProgram(program),
        rawBufferList(arguments),
        arguments.len,
        if (donated_input_indices.len == 0) null else donated_input_indices.ptr,
        donated_input_indices.len,
        &raw_outputs,
        &raw_output_count,
        &raw_profile,
    );
    if (ok == 0 or raw_outputs == null) {
        if (raw_outputs != null) c.pjrtx_mlx_metal_program_output_array_destroy(raw_outputs);
        return null;
    }
    return .{
        .raw_outputs = raw_outputs,
        .count = @intCast(raw_output_count),
        .profile = .{
            .host_enqueue_us = raw_profile.host_enqueue_us,
            .device_sync_wait_us = raw_profile.device_sync_wait_us,
            .output_count = raw_profile.output_count,
            .donated_input_count = raw_profile.donated_input_count,
            .measured_device_sync = raw_profile.measured_device_sync != 0,
            .first_execute = raw_profile.first_execute != 0,
        },
    };
}

/// Destroys an MLX compiled program handle.
pub fn programDestroy(program: ProgramHandle) void {
    c.pjrtx_mlx_metal_program_destroy(rawProgram(program));
}

