const std = @import("std");

const ir = @import("src/compiler/ir");
const buffer_mod = @import("buffer.zig");
const artifacts = @import("executable_metal_graph_artifacts.zig");
const diagnostic = @import("executable_metal_graph_diagnostic.zig");
const graph_request_mod = @import("executable_metal_graph_graph_request.zig");
const metalcpp_call = @import("metalcpp_call.zig");
const metal_graph_lowering = @import("executable_metal_graph_lowering.zig");
const profiling = @import("profiling.zig");
const program_mod = @import("program.zig");
const program_request_mod = @import("executable_metal_graph_program_request.zig");
const resident_inputs = @import("executable_metal_graph_resident_inputs.zig");
const types = @import("execution_types.zig");

/// Resident executable-level generated Metal graph handle.
pub const Handle = metalcpp_call.ExecutableProgramHandle;

/// Allocates one executable-level Metal graph slot per addressable device.
pub fn allocHandles(allocator: std.mem.Allocator, device_count: usize) ![]?Handle {
    const handles = try allocator.alloc(?Handle, device_count);
    @memset(handles, null);
    return handles;
}

/// Compiles a full executable-level generated Metal graph when the plan is eligible.
pub fn compileIfSupported(
    allocator: std.mem.Allocator,
    plan: *const ir.ExecutablePlan,
    program: *const program_mod.Program,
    device_local_hardware_ids: []const i32,
    handles: []?Handle,
) program_mod.Error!void {
    if (!profiling.metalCppExecutableRunnerEnabled()) {
        diagnostic.CompileCoverage.write(allocator, plan, null, device_local_hardware_ids.len, 0, "runner_disabled", null);
        return;
    }
    if (handles.len != device_local_hardware_ids.len) {
        diagnostic.CompileCoverage.write(allocator, plan, null, device_local_hardware_ids.len, 0, "device_count_mismatch", null);
        return error.CommandSubmissionFailed;
    }
    var compiled_count: usize = 0;
    var metrics: ?metal_graph_lowering.Metrics = null;
    var compile_error: ?[]u8 = null;
    defer if (compile_error) |message| allocator.free(message);
    for (device_local_hardware_ids, 0..) |local_hardware_id, device_index| {
        const compiled = (try createKernel(allocator, plan, program, local_hardware_id)) orelse {
            if (compile_error == null) {
                compile_error = try allocator.dupe(u8, metalcpp_call.lastError() orelse "metal-cpp executable program create returned null");
            }
            continue;
        };
        errdefer metalcpp_call.executableProgramDestroy(compiled.handle);
        handles[device_index] = compiled.handle;
        if (metrics == null) metrics = compiled.metrics;
        compiled_count += 1;
    }
    diagnostic.CompileCoverage.write(allocator, plan, metrics, device_local_hardware_ids.len, compiled_count, if (compiled_count == 0) "unsupported" else "compiled", compile_error);
}

/// Releases executable-level generated Metal graph handles.
pub fn destroy(handles: []?Handle) void {
    for (handles) |maybe_handle| {
        if (maybe_handle) |handle| metalcpp_call.executableProgramDestroy(handle);
    }
}

/// Returns one resident executable-level Metal graph handle for a device.
pub fn get(handles: []?Handle, device_index: usize) ?Handle {
    if (device_index >= handles.len) return null;
    return handles[device_index];
}

/// Executes an executable-level generated Metal graph over live arguments and constants.
pub fn execute(
    allocator: std.mem.Allocator,
    executable: anytype,
    device_index: usize,
    handle: Handle,
    arguments: []const types.BufferHandle,
) types.Error![]types.ExecutableOutput {
    var maybe_program_request = try program_request_mod.Request.init(allocator, executable.plan, &executable.program);
    defer if (maybe_program_request) |*request| request.deinit(allocator);
    var maybe_graph_request: ?graph_request_mod.Request = null;
    defer if (maybe_graph_request) |*request| request.deinit(allocator);
    const constant_source = resident_inputs.Source{
        .instruction_count = executable.plan.instructions.len,
        .device_index = device_index,
        .constant_handles = executable.constant_handles,
    };
    const inputs = if (maybe_program_request) |request|
        try request.inputHandles(allocator, constant_source, arguments)
    else blk: {
        maybe_graph_request = (try graph_request_mod.Request.init(allocator, executable.plan)) orelse return error.CommandSubmissionFailed;
        break :blk try maybe_graph_request.?.inputHandles(allocator, constant_source, arguments);
    };
    defer allocator.free(inputs);
    const raw_outputs = (try metalcpp_call.executableProgramExecute(allocator, handle, inputs)) orelse return error.CommandSubmissionFailed;
    defer raw_outputs.deinit();
    if (raw_outputs.count != executable.plan.output_ids.len) return error.CommandSubmissionFailed;

    const outputs = try allocator.alloc(types.ExecutableOutput, executable.plan.output_ids.len);
    errdefer allocator.free(outputs);
    var initialized: usize = 0;
    errdefer {
        for (outputs[0..initialized]) |output| buffer_mod.Opaque.destroy(output.handle);
    }
    for (outputs, 0..) |*output, output_index| {
        const output_id = executable.plan.output_ids[output_index];
        if (output_id.index >= executable.plan.values.len) return error.CommandSubmissionFailed;
        const descriptor = executable.plan.values[output_id.index].descriptor;
        const output_handle = raw_outputs.take(output_index) orelse return error.CommandSubmissionFailed;
        output.* = .{
            .handle = output_handle,
            .element_type = descriptor.element_type,
            .dims = descriptor.dims,
            .byte_size = ir.denseByteSize(descriptor.element_type, descriptor.dims),
        };
        initialized += 1;
    }
    return outputs;
}

const CompiledProgramRequest = struct {
    handle: Handle,
    metrics: metal_graph_lowering.Metrics,
};

fn createKernel(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, program: *const program_mod.Program, device_local_hardware_id: i32) program_mod.Error!?CompiledProgramRequest {
    if (try program_request_mod.Request.init(allocator, plan, program)) |request| {
        var program_request = request;
        defer program_request.deinit(allocator);
        artifacts.ArtifactDump.dumpProgramRequestIfEnabled(allocator, plan, device_local_hardware_id, program_request) catch {};
        const handle = (try metalcpp_call.executableProgramCreate(allocator, .{
            .device_ordinal = device_local_hardware_id,
            .entry_kernel_name = program_request.steps[0].kernel_name,
            .source = program_request.steps[0].source,
            .inputs = &.{},
            .outputs = &.{},
            .element_count = program_request.steps[0].element_count,
            .values = program_request.value_specs,
            .input_values = program_request.input_values,
            .output_values = program_request.output_values,
            .steps = program_request.steps,
        })) orelse return null;
        return .{ .handle = handle, .metrics = program_request.metrics };
    }

    var request = (try graph_request_mod.Request.init(allocator, plan)) orelse return null;
    defer request.deinit(allocator);
    const kernel_name = try std.fmt.allocPrint(allocator, "pjrtx_metalcpp_executable_{x}", .{@intFromPtr(plan)});
    defer allocator.free(kernel_name);
    var source = std.Io.Writer.Allocating.init(allocator);
    defer source.deinit();
    request.writeMsl(&source.writer, kernel_name) catch return error.OutOfMemory;
    const handle = (try metalcpp_call.executableProgramCreate(allocator, .{
        .device_ordinal = device_local_hardware_id,
        .entry_kernel_name = kernel_name,
        .source = source.written(),
        .inputs = request.inputSpecs(),
        .outputs = request.outputSpecs(),
        .element_count = request.elementCount(),
    })) orelse return null;
    return .{ .handle = handle, .metrics = .{ .step_count = 1, .node_step_count = 1 } };
}
