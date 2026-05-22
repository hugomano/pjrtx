const std = @import("std");

const c = @import("c");

/// Opaque MLX-owned Metal buffer accepted by direct metal-cpp kernels.
pub const BufferHandle = *anyopaque;

/// Opaque compiled Metal-cpp fusion kernel owned by the backend-local C++ shim.
pub const FusionKernelHandle = *anyopaque;

/// Opaque executable-level Metal-cpp program owned by the backend-local C++ shim.
pub const ExecutableProgramHandle = *anyopaque;

/// Dense tensor descriptor accepted by backend-local generated Metal-cpp shims.
pub const TensorSpec = struct {
    dtype: c_int,
    dims: []const i64,
};

/// Compile-time payload for one generated Metal-cpp fusion kernel.
pub const FusionKernelSpec = struct {
    device_ordinal: i32,
    kernel_name: []const u8,
    source: []const u8,
    inputs: []const TensorSpec,
    outputs: []const TensorSpec,
    element_count: u64,
    threads_per_threadgroup: u32 = 256,
};

/// Compile-time payload for one generated executable program value.
pub const ExecutableValueSpec = TensorSpec;

/// Compile-time payload for one generated executable program step.
pub const ExecutableStepSpec = struct {
    kernel_name: []const u8,
    source: []const u8,
    inputs: []const u64,
    outputs: []const u64,
    release_values: []const u64 = &.{},
    element_count: u64,
    threads_per_threadgroup: u32 = 256,
    in_place_input: ?u64 = null,
};

/// Compile-time payload for one generated executable-level Metal-cpp program.
pub const ExecutableProgramSpec = struct {
    device_ordinal: i32,
    entry_kernel_name: []const u8,
    source: []const u8,
    inputs: []const TensorSpec,
    outputs: []const TensorSpec,
    element_count: u64,
    threads_per_threadgroup: u32 = 256,
    values: []const ExecutableValueSpec = &.{},
    input_values: []const u64 = &.{},
    output_values: []const u64 = &.{},
    steps: []const ExecutableStepSpec = &.{},
};

/// Output buffer array returned by a Metal-cpp fusion kernel execution.
pub const FusionOutputs = struct {
    raw_outputs: [*c]?*c.PjrtxMlxMetalBuffer,
    count: usize,

    /// Releases array storage that still belongs to the C++ shim.
    pub fn deinit(self: FusionOutputs) void {
        c.pjrtx_metalcpp_fusion_output_array_destroy(self.raw_outputs);
    }

    /// Transfers one output buffer handle to the caller.
    pub fn take(self: FusionOutputs, index: usize) ?BufferHandle {
        if (index >= self.count) return null;
        const raw = self.raw_outputs[index] orelse return null;
        self.raw_outputs[index] = null;
        return fromRawBuffer(raw);
    }
};

/// Output buffer array returned by a Metal-cpp executable-program execution.
pub const ExecutableProgramOutputs = struct {
    raw_outputs: [*c]?*c.PjrtxMlxMetalBuffer,
    count: usize,

    /// Releases array storage that still belongs to the C++ shim.
    pub fn deinit(self: ExecutableProgramOutputs) void {
        c.pjrtx_metalcpp_executable_program_output_array_destroy(self.raw_outputs);
    }

    /// Transfers one output buffer handle to the caller.
    pub fn take(self: ExecutableProgramOutputs, index: usize) ?BufferHandle {
        if (index >= self.count) return null;
        const raw = self.raw_outputs[index] orelse return null;
        self.raw_outputs[index] = null;
        return fromRawBuffer(raw);
    }
};

/// Runs a dense row-major binary operation through the backend-local metal-cpp path.
pub fn denseBinaryOut(
    lhs: BufferHandle,
    rhs: BufferHandle,
    op: c_int,
    output_dims: []const i64,
) ?BufferHandle {
    const raw = c.pjrtx_metalcpp_buffer_dense_binary_out(
        rawBuffer(lhs),
        rawBuffer(rhs),
        op,
        if (output_dims.len == 0) null else output_dims.ptr,
        output_dims.len,
    ) orelse return null;
    return fromRawBuffer(raw);
}

/// Compiles one generated Metal-cpp fusion kernel.
pub fn fusionKernelCreate(
    allocator: std.mem.Allocator,
    spec: FusionKernelSpec,
) !?FusionKernelHandle {
    const raw_inputs = try rawTensorSpecs(allocator, spec.inputs);
    defer allocator.free(raw_inputs);
    const raw_outputs = try rawTensorSpecs(allocator, spec.outputs);
    defer allocator.free(raw_outputs);
    var args = c.PjrtxMetalCppFusionKernelCreateArgs{
        .struct_size = @sizeOf(c.PjrtxMetalCppFusionKernelCreateArgs),
        .device_ordinal = spec.device_ordinal,
        .kernel_name = spec.kernel_name.ptr,
        .kernel_name_size = spec.kernel_name.len,
        .source = spec.source.ptr,
        .source_size = spec.source.len,
        .inputs = raw_inputs.ptr,
        .input_count = raw_inputs.len,
        .outputs = raw_outputs.ptr,
        .output_count = raw_outputs.len,
        .element_count = spec.element_count,
        .threads_per_threadgroup = spec.threads_per_threadgroup,
    };
    const raw = c.pjrtx_metalcpp_fusion_kernel_create(&args) orelse return null;
    return @ptrCast(raw);
}

/// Executes one generated Metal-cpp fusion kernel over MLX-owned buffers.
pub fn fusionKernelExecute(
    allocator: std.mem.Allocator,
    kernel: FusionKernelHandle,
    inputs: []const BufferHandle,
) !?FusionOutputs {
    const raw_inputs = try rawBufferArray(allocator, inputs);
    defer allocator.free(raw_inputs);
    var raw_outputs: [*c]?*c.PjrtxMlxMetalBuffer = undefined;
    var output_count: u64 = 0;
    if (c.pjrtx_metalcpp_fusion_kernel_execute(
        rawFusionKernel(kernel),
        raw_inputs.ptr,
        raw_inputs.len,
        &raw_outputs,
        &output_count,
    ) == 0) return null;
    return .{
        .raw_outputs = raw_outputs,
        .count = @intCast(output_count),
    };
}

/// Destroys one generated Metal-cpp fusion kernel.
pub fn fusionKernelDestroy(kernel: FusionKernelHandle) void {
    c.pjrtx_metalcpp_fusion_kernel_destroy(rawFusionKernel(kernel));
}

/// Compiles one generated executable-level Metal-cpp program.
pub fn executableProgramCreate(
    allocator: std.mem.Allocator,
    spec: ExecutableProgramSpec,
) !?ExecutableProgramHandle {
    const raw_inputs = try rawTensorSpecs(allocator, spec.inputs);
    defer allocator.free(raw_inputs);
    const raw_outputs = try rawTensorSpecs(allocator, spec.outputs);
    defer allocator.free(raw_outputs);
    const raw_values = try rawExecutableValues(allocator, spec.values);
    defer allocator.free(raw_values);
    const raw_steps = try rawExecutableSteps(allocator, spec.steps);
    defer raw_steps.deinit(allocator);
    var args = c.PjrtxMetalCppExecutableProgramCreateArgs{
        .struct_size = @sizeOf(c.PjrtxMetalCppExecutableProgramCreateArgs),
        .device_ordinal = spec.device_ordinal,
        .entry_kernel_name = spec.entry_kernel_name.ptr,
        .entry_kernel_name_size = spec.entry_kernel_name.len,
        .source = spec.source.ptr,
        .source_size = spec.source.len,
        .inputs = raw_inputs.ptr,
        .input_count = raw_inputs.len,
        .outputs = raw_outputs.ptr,
        .output_count = raw_outputs.len,
        .element_count = spec.element_count,
        .threads_per_threadgroup = spec.threads_per_threadgroup,
        .values = if (raw_values.len == 0) null else raw_values.ptr,
        .value_count = raw_values.len,
        .input_values = if (spec.input_values.len == 0) null else spec.input_values.ptr,
        .input_value_count = spec.input_values.len,
        .output_values = if (spec.output_values.len == 0) null else spec.output_values.ptr,
        .output_value_count = spec.output_values.len,
        .steps = if (raw_steps.steps.len == 0) null else raw_steps.steps.ptr,
        .step_count = raw_steps.steps.len,
    };
    const raw =
        c.pjrtx_metalcpp_executable_program_create(&args) orelse return null;
    return @ptrCast(raw);
}

/// Executes one generated executable-level Metal-cpp program over MLX-owned buffers.
pub fn executableProgramExecute(
    allocator: std.mem.Allocator,
    program: ExecutableProgramHandle,
    inputs: []const BufferHandle,
) !?ExecutableProgramOutputs {
    const raw_inputs = try rawBufferArray(allocator, inputs);
    defer allocator.free(raw_inputs);
    var raw_outputs: [*c]?*c.PjrtxMlxMetalBuffer = undefined;
    var output_count: u64 = 0;
    if (c.pjrtx_metalcpp_executable_program_execute(
        rawExecutableProgram(program),
        raw_inputs.ptr,
        raw_inputs.len,
        &raw_outputs,
        &output_count,
    ) == 0) return null;
    return .{
        .raw_outputs = raw_outputs,
        .count = @intCast(output_count),
    };
}

/// Destroys one generated executable-level Metal-cpp program.
pub fn executableProgramDestroy(program: ExecutableProgramHandle) void {
    c.pjrtx_metalcpp_executable_program_destroy(rawExecutableProgram(program));
}

/// Returns the last backend-local Metal-cpp create/execute failure for diagnostics.
pub fn lastError() ?[]const u8 {
    const raw = c.pjrtx_metalcpp_last_error() orelse return null;
    const message = std.mem.span(raw);
    return if (message.len == 0) null else message;
}

fn rawTensorSpecs(
    allocator: std.mem.Allocator,
    specs: []const TensorSpec,
) ![]c.PjrtxMetalCppTensorSpec {
    const raw = try allocator.alloc(c.PjrtxMetalCppTensorSpec, specs.len);
    for (raw, specs) |*out, spec| {
        out.* = .{
            .dtype = spec.dtype,
            .dims = if (spec.dims.len == 0) null else spec.dims.ptr,
            .rank = spec.dims.len,
        };
    }
    return raw;
}

fn rawExecutableValues(
    allocator: std.mem.Allocator,
    values: []const ExecutableValueSpec,
) ![]c.PjrtxMetalCppExecutableValue {
    const raw = try allocator.alloc(c.PjrtxMetalCppExecutableValue, values.len);
    for (raw, values) |*out, value| {
        out.* = .{
            .spec = .{
                .dtype = value.dtype,
                .dims = if (value.dims.len == 0) null else value.dims.ptr,
                .rank = value.dims.len,
            },
        };
    }
    return raw;
}

const RawExecutableSteps = struct {
    steps: []c.PjrtxMetalCppExecutableStep,
    input_indices: [][]u64,
    output_indices: [][]u64,
    release_indices: [][]u64,

    fn deinit(self: RawExecutableSteps, allocator: std.mem.Allocator) void {
        for (self.input_indices) |indices| allocator.free(indices);
        for (self.output_indices) |indices| allocator.free(indices);
        for (self.release_indices) |indices| allocator.free(indices);
        allocator.free(self.input_indices);
        allocator.free(self.output_indices);
        allocator.free(self.release_indices);
        allocator.free(self.steps);
    }
};

fn rawExecutableSteps(
    allocator: std.mem.Allocator,
    steps: []const ExecutableStepSpec,
) !RawExecutableSteps {
    const raw_steps = try allocator.alloc(c.PjrtxMetalCppExecutableStep, steps.len);
    errdefer allocator.free(raw_steps);
    const input_indices = try allocator.alloc([]u64, steps.len);
    errdefer allocator.free(input_indices);
    const output_indices = try allocator.alloc([]u64, steps.len);
    errdefer allocator.free(output_indices);
    const release_indices = try allocator.alloc([]u64, steps.len);
    errdefer allocator.free(release_indices);
    for (input_indices) |*indices| indices.* = @constCast(&[_]u64{});
    for (output_indices) |*indices| indices.* = @constCast(&[_]u64{});
    for (release_indices) |*indices| indices.* = @constCast(&[_]u64{});

    var initialized: usize = 0;
    errdefer {
        for (input_indices[0..initialized]) |indices| allocator.free(indices);
        for (output_indices[0..initialized]) |indices| allocator.free(indices);
        for (release_indices[0..initialized]) |indices| allocator.free(indices);
    }

    for (raw_steps, steps, 0..) |*out, step, i| {
        input_indices[i] = try allocator.dupe(u64, step.inputs);
        errdefer if (initialized == i) allocator.free(input_indices[i]);
        output_indices[i] = try allocator.dupe(u64, step.outputs);
        errdefer if (initialized == i) allocator.free(output_indices[i]);
        release_indices[i] = try allocator.dupe(u64, step.release_values);
        errdefer if (initialized == i) allocator.free(release_indices[i]);
        initialized += 1;
        out.* = .{
            .kernel_name = step.kernel_name.ptr,
            .kernel_name_size = step.kernel_name.len,
            .source = step.source.ptr,
            .source_size = step.source.len,
            .inputs = if (input_indices[i].len == 0) null else input_indices[i].ptr,
            .input_count = input_indices[i].len,
            .outputs = if (output_indices[i].len == 0) null else output_indices[i].ptr,
            .output_count = output_indices[i].len,
            .release_values = if (release_indices[i].len == 0) null else release_indices[i].ptr,
            .release_value_count = release_indices[i].len,
            .element_count = step.element_count,
            .threads_per_threadgroup = step.threads_per_threadgroup,
            .in_place_input_plus_one = if (step.in_place_input) |input_index| input_index + 1 else 0,
        };
    }

    return .{
        .steps = raw_steps,
        .input_indices = input_indices,
        .output_indices = output_indices,
        .release_indices = release_indices,
    };
}

fn rawBufferArray(
    allocator: std.mem.Allocator,
    handles: []const BufferHandle,
) ![]?*c.PjrtxMlxMetalBuffer {
    const raw = try allocator.alloc(?*c.PjrtxMlxMetalBuffer, handles.len);
    for (raw, handles) |*out, handle| {
        out.* = rawBuffer(handle);
    }
    return raw;
}

fn fromRawBuffer(raw: *c.PjrtxMlxMetalBuffer) BufferHandle {
    return @ptrCast(raw);
}

fn rawBuffer(handle: BufferHandle) *c.PjrtxMlxMetalBuffer {
    return @ptrCast(@alignCast(handle));
}

fn rawFusionKernel(handle: FusionKernelHandle) *c.PjrtxMetalCppFusionKernel {
    return @ptrCast(@alignCast(handle));
}

fn rawExecutableProgram(
    handle: ExecutableProgramHandle,
) *c.PjrtxMetalCppExecutableProgram {
    return @ptrCast(@alignCast(handle));
}
