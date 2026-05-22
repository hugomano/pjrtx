const std = @import("std");
const ir = @import("src/compiler/ir");

const buffer_mod = @import("buffer.zig");
const metalcpp_call = @import("metalcpp_call.zig");
const profiling = @import("profiling.zig");
const program_mod = @import("program.zig");
const request_mod = @import("executable_metal_graph_fusion_request.zig");
const types = @import("execution_types.zig");
const values_mod = @import("execution_values.zig");

const BufferHandle = types.BufferHandle;
const Error = types.Error;
const ValueBindings = values_mod.ValueBindings;

/// Executes the env-gated resident Metal-cpp fusion path for dense view/elementwise groups.
pub const FusionRunner = struct {
    allocator: std.mem.Allocator,
    plan: *const ir.ExecutablePlan,
    program: *const program_mod.Program,
    device_local_hardware_id: i32,
    resident_kernel: ?metalcpp_call.FusionKernelHandle = null,
    values: *ValueBindings,

    /// Runs a supported fusion group and stores its output, or returns null for MLX replay.
    pub fn run(self: FusionRunner, group: program_mod.FusionGroup) Error!?void {
        if (!enabled()) return null;
        var request = (try request_mod.Request.init(self.allocator, self.plan, self.program, group)) orelse return null;
        defer request.deinit(self.allocator);
        if (request.outputValues().len != 1) return null;
        const input_handles = try FusionInputs.collect(self.allocator, request, self.values);
        defer self.allocator.free(input_handles);

        const transient_kernel = if (self.resident_kernel == null)
            (try createKernel(self.allocator, request, self.plan, group.id, self.device_local_hardware_id)) orelse return null
        else
            null;
        defer if (transient_kernel) |kernel| metalcpp_call.fusionKernelDestroy(kernel);
        const kernel = self.resident_kernel orelse transient_kernel.?;

        const outputs = (try metalcpp_call.fusionKernelExecute(self.allocator, kernel, input_handles)) orelse return null;
        defer outputs.deinit();
        if (outputs.count != 1) return error.CommandSubmissionFailed;
        const output = outputs.take(0) orelse return error.CommandSubmissionFailed;
        errdefer buffer_mod.Opaque.destroy(output);
        try values_mod.storeOwnedValueHandle(self.values.handles, self.values.owned, request.outputValues()[0], output);
        return {};
    }
};

/// Compiles one generated fusion kernel for an executable-resident slot.
pub fn createKernel(
    allocator: std.mem.Allocator,
    request: request_mod.Request,
    plan: *const ir.ExecutablePlan,
    group_id: usize,
    device_local_hardware_id: i32,
) Error!?metalcpp_call.FusionKernelHandle {
    const kernel_name = try std.fmt.allocPrint(allocator, "pjrtx_metalcpp_fusion_{x}_{d}", .{ @intFromPtr(plan), group_id });
    defer allocator.free(kernel_name);

    var source = std.Io.Writer.Allocating.init(allocator);
    defer source.deinit();
    request.writeMsl(allocator, &source.writer, kernel_name) catch return error.OutOfMemory;

    return try metalcpp_call.fusionKernelCreate(allocator, .{
        .device_ordinal = device_local_hardware_id,
        .kernel_name = kernel_name,
        .source = source.written(),
        .inputs = request.inputSpecs(),
        .outputs = request.outputSpecs(),
        .element_count = request.elementCount(),
    });
}

/// Builds a generated-kernel request for resident fusion-kernel compilation.
pub fn initKernelRequest(
    allocator: std.mem.Allocator,
    plan: *const ir.ExecutablePlan,
    program: *const program_mod.Program,
    group: program_mod.FusionGroup,
) Error!?request_mod.Request {
    const request = (try request_mod.Request.init(allocator, plan, program, group)) orelse return null;
    if (request.outputValues().len != 1) {
        var mutable = request;
        mutable.deinit(allocator);
        return null;
    }
    return request;
}

const FusionInputs = struct {
    fn collect(allocator: std.mem.Allocator, request: request_mod.Request, values: *ValueBindings) Error![]BufferHandle {
        const handles = try allocator.alloc(BufferHandle, request.inputValues().len);
        errdefer allocator.free(handles);
        for (request.inputValues(), 0..) |input_id, input_index| {
            if (input_id.index >= values.handles.len) return error.CommandSubmissionFailed;
            handles[input_index] = values.handles[input_id.index] orelse return error.CommandSubmissionFailed;
        }
        return handles;
    }
};

fn enabled() bool {
    return profiling.metalCppFusionRunnerEnabled();
}
