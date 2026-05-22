const std = @import("std");
const c = @import("c");

/// Opaque MLX/Metal buffer owned by the foreign backend runtime.
pub const BufferHandle = *anyopaque;

/// Opaque MLX/Metal async host-to-device transfer owned by the foreign backend runtime.
pub const AsyncTransferHandle = *anyopaque;

/// Opaque compiled MLX/Metal program owned by the foreign backend runtime.
pub const ProgramHandle = *anyopaque;

/// Device-buffer callback frame used while MLX records a compiled program.
pub const ProgramBuildCall = struct {
    raw_inputs: [*c]const ?*c.PjrtxMlxMetalBuffer,
    input_count: usize,
    raw_outputs: [*c]?*c.PjrtxMlxMetalBuffer,
    output_count: usize,

    /// Returns the number of dynamic inputs presented by the compiled program.
    pub fn inputCount(self: ProgramBuildCall) usize {
        return self.input_count;
    }

    /// Returns the number of outputs expected from the compiled program builder.
    pub fn outputCount(self: ProgramBuildCall) usize {
        return self.output_count;
    }

    /// Borrows one input buffer from the compiled program builder.
    pub fn input(self: ProgramBuildCall, index: usize) ?BufferHandle {
        if (index >= self.input_count) return null;
        return fromRawBuffer(self.raw_inputs[index]);
    }

    /// Publishes one output buffer produced by the compiled program builder.
    pub fn setOutput(self: ProgramBuildCall, index: usize, handle: BufferHandle) bool {
        if (index >= self.output_count) return false;
        self.raw_outputs[index] = rawBuffer(handle);
        return true;
    }
};

/// Callback invoked while MLX records the body of a compiled program.
pub const ProgramBuildCallback = fn (?*anyopaque, ProgramBuildCall) bool;

/// Outputs returned from an executed compiled MLX/Metal program.
pub const ProgramOutputs = struct {
    raw_outputs: [*c]?*c.PjrtxMlxMetalBuffer,
    count: usize,
    profile: ProgramExecuteProfile,

    /// Releases any output array storage still owned by the foreign runtime.
    pub fn deinit(self: ProgramOutputs) void {
        c.pjrtx_mlx_metal_program_output_array_destroy(self.raw_outputs);
    }

    /// Returns the number of buffers produced by the compiled program.
    pub fn len(self: ProgramOutputs) usize {
        return self.count;
    }

    /// Transfers one output buffer handle to the caller.
    pub fn take(self: ProgramOutputs, index: usize) ?BufferHandle {
        if (index >= self.count) return null;
        const raw = self.raw_outputs[index] orelse return null;
        self.raw_outputs[index] = null;
        return fromRawBuffer(raw);
    }
};

/// Timing counters reported by an MLX compiled-program execution.
pub const ProgramExecuteProfile = struct {
    /// Time spent building/enqueuing the compiled MLX graph on the host.
    host_enqueue_us: u64,
    /// Optional synchronized device wait time when explicit device-sync profiling is enabled.
    device_sync_wait_us: u64,
    /// Number of device buffers returned by the compiled program.
    output_count: u64,
    /// Number of dynamic input buffers donated to this execution.
    donated_input_count: u64,
    /// Whether `device_sync_wait_us` was measured by synchronizing the Metal stream.
    measured_device_sync: bool,
    /// Whether the profile belongs to the program's first execution.
    first_execute: bool,
};

/// Device metadata copied out of the MLX/Metal runtime without exposing C layout.
pub const DeviceInfo = struct {
    /// MLX/Metal device ordinal used for subsequent backend calls.
    ordinal: i32,
    /// Metal registry identifier when the runtime can provide it.
    registry_id: u64,
    /// Runtime-provided memory working-set recommendation.
    recommended_max_working_set_size: u64,
    /// Whether Metal reports unified memory for this device.
    has_unified_memory: bool,
    /// NUL-padded device name bytes copied from the C boundary.
    name: [DeviceNameBytes]u8,
};

/// MLX/Metal dtype code accepted only by the raw C boundary calls.
pub const Dtype = enum(c_int) {
    pred = c.PJRTX_MLX_METAL_DTYPE_PRED,
    s8 = c.PJRTX_MLX_METAL_DTYPE_S8,
    s32 = c.PJRTX_MLX_METAL_DTYPE_S32,
    u8 = c.PJRTX_MLX_METAL_DTYPE_U8,
    u16 = c.PJRTX_MLX_METAL_DTYPE_U16,
    u32 = c.PJRTX_MLX_METAL_DTYPE_U32,
    u64 = c.PJRTX_MLX_METAL_DTYPE_U64,
    f16 = c.PJRTX_MLX_METAL_DTYPE_F16,
    f32 = c.PJRTX_MLX_METAL_DTYPE_F32,
    bf16 = c.PJRTX_MLX_METAL_DTYPE_BF16,
    c64 = c.PJRTX_MLX_METAL_DTYPE_C64,
};

/// MLX/Metal binary operation code accepted only by the raw C boundary calls.
pub const BinaryOp = enum(c_int) {
    add = c.PJRTX_MLX_METAL_U8_BINARY_ADD,
    subtract = c.PJRTX_MLX_METAL_U8_BINARY_SUBTRACT,
    multiply = c.PJRTX_MLX_METAL_U8_BINARY_MULTIPLY,
    divide = c.PJRTX_MLX_METAL_U8_BINARY_DIVIDE,
    maximum = c.PJRTX_MLX_METAL_BINARY_MAXIMUM,
    minimum = c.PJRTX_MLX_METAL_BINARY_MINIMUM,
    power = c.PJRTX_MLX_METAL_BINARY_POWER,
    remainder = c.PJRTX_MLX_METAL_BINARY_REMAINDER,
    atan2 = c.PJRTX_MLX_METAL_BINARY_ATAN2,
    and_ = c.PJRTX_MLX_METAL_BINARY_AND,
    or_ = c.PJRTX_MLX_METAL_BINARY_OR,
    xor = c.PJRTX_MLX_METAL_BINARY_XOR,
    shift_left = c.PJRTX_MLX_METAL_BINARY_SHIFT_LEFT,
    shift_right = c.PJRTX_MLX_METAL_BINARY_SHIFT_RIGHT,
};

/// MLX/Metal unary operation code accepted only by the raw C boundary calls.
pub const UnaryOp = enum(c_int) {
    negate = c.PJRTX_MLX_METAL_U8_UNARY_NEGATE,
    exp = c.PJRTX_MLX_METAL_UNARY_EXP,
    tanh = c.PJRTX_MLX_METAL_UNARY_TANH,
    sqrt = c.PJRTX_MLX_METAL_UNARY_SQRT,
    rsqrt = c.PJRTX_MLX_METAL_UNARY_RSQRT,
    abs = c.PJRTX_MLX_METAL_UNARY_ABS,
    ceil = c.PJRTX_MLX_METAL_UNARY_CEIL,
    floor = c.PJRTX_MLX_METAL_UNARY_FLOOR,
    log = c.PJRTX_MLX_METAL_UNARY_LOG,
    log1p = c.PJRTX_MLX_METAL_UNARY_LOG1P,
    logistic = c.PJRTX_MLX_METAL_UNARY_LOGISTIC,
    sine = c.PJRTX_MLX_METAL_UNARY_SIN,
    cosine = c.PJRTX_MLX_METAL_UNARY_COS,
    sign = c.PJRTX_MLX_METAL_UNARY_SIGN,
    expm1 = c.PJRTX_MLX_METAL_UNARY_EXPM1,
    not_ = c.PJRTX_MLX_METAL_UNARY_NOT,
    is_finite = c.PJRTX_MLX_METAL_UNARY_ISFINITE,
    round_nearest_even = c.PJRTX_MLX_METAL_UNARY_ROUND,
    cbrt = c.PJRTX_MLX_METAL_UNARY_CBRT,
    round_nearest_afz = c.PJRTX_MLX_METAL_UNARY_ROUND_AFZ,
    popcnt = c.PJRTX_MLX_METAL_UNARY_POPCNT,
    count_leading_zeros = c.PJRTX_MLX_METAL_UNARY_CLZ,
};

/// MLX/Metal reduction operation code accepted only by the raw C boundary calls.
pub const ReduceOp = enum(c_int) {
    sum = c.PJRTX_MLX_METAL_REDUCE_SUM,
    max = c.PJRTX_MLX_METAL_REDUCE_MAX,
    min = c.PJRTX_MLX_METAL_REDUCE_MIN,
    and_ = c.PJRTX_MLX_METAL_REDUCE_AND,
    or_ = c.PJRTX_MLX_METAL_REDUCE_OR,
};

/// MLX/Metal scatter update code accepted only by the raw C boundary calls.
pub const ScatterUpdate = enum(c_int) {
    set = c.PJRTX_MLX_METAL_SCATTER_SET,
    add = c.PJRTX_MLX_METAL_SCATTER_ADD,
};

/// MLX/Metal comparison code accepted only by the raw C boundary calls.
pub const CompareOp = enum(c_int) {
    eq = c.PJRTX_MLX_METAL_COMPARE_EQ,
    ne = c.PJRTX_MLX_METAL_COMPARE_NE,
    ge = c.PJRTX_MLX_METAL_COMPARE_GE,
    gt = c.PJRTX_MLX_METAL_COMPARE_GT,
    le = c.PJRTX_MLX_METAL_COMPARE_LE,
    lt = c.PJRTX_MLX_METAL_COMPARE_LT,
};

/// MLX/Metal FFT code accepted only by the raw C boundary calls.
pub const FftKind = enum(c_int) {
    fft = c.PJRTX_MLX_METAL_FFT,
    ifft = c.PJRTX_MLX_METAL_IFFT,
    rfft = c.PJRTX_MLX_METAL_RFFT,
    irfft = c.PJRTX_MLX_METAL_IRFFT,
};

/// MLX/Metal random distribution code accepted only by the raw C boundary calls.
pub const RngDistribution = enum(c_int) {
    uniform = c.PJRTX_MLX_METAL_RNG_UNIFORM,
    normal = c.PJRTX_MLX_METAL_RNG_NORMAL,
};

/// MLX/Metal triangular solve transpose mode accepted only by the raw C boundary calls.
pub const TriangularTranspose = enum(c_int) {
    none = 0,
    transpose = 1,
};

/// Pair of MLX/Metal buffers produced by an operation with two outputs.
pub const BufferPair = struct {
    /// First returned buffer handle.
    first: BufferHandle,
    /// Second returned buffer handle.
    second: BufferHandle,
};

/// Fixed byte capacity of a copied MLX/Metal device name.
pub const DeviceNameBytes = c.PJRTX_MLX_METAL_DEVICE_NAME_BYTES;

pub fn ProgramBuildAdapter(comptime callback: ProgramBuildCallback) type {
    return struct {
        pub fn call(
            user_data: ?*anyopaque,
            raw_inputs: [*c]const ?*c.PjrtxMlxMetalBuffer,
            input_count: u64,
            raw_outputs: [*c]?*c.PjrtxMlxMetalBuffer,
            output_count: u64,
        ) callconv(.c) c_int {
            const frame: ProgramBuildCall = .{
                .raw_inputs = raw_inputs,
                .input_count = @intCast(input_count),
                .raw_outputs = raw_outputs,
                .output_count = @intCast(output_count),
            };
            return flag(callback(user_data, frame));
        }
    };
}

pub fn code(value: anytype) c_int {
    return @intFromEnum(value);
}

pub fn flag(value: bool) c_int {
    return if (value) 1 else 0;
}

pub fn fromRawBuffer(handle: ?*c.PjrtxMlxMetalBuffer) ?BufferHandle {
    return if (handle) |ptr| @ptrCast(ptr) else null;
}

pub fn rawBuffer(handle: BufferHandle) *c.PjrtxMlxMetalBuffer {
    return @ptrCast(@alignCast(handle));
}

pub fn rawBufferList(handles: []const BufferHandle) [*c]?*c.PjrtxMlxMetalBuffer {
    return @ptrCast(@constCast(handles.ptr));
}

pub fn fromRawTransfer(handle: ?*c.PjrtxMlxMetalAsyncHostToDeviceTransfer) ?AsyncTransferHandle {
    return if (handle) |ptr| @ptrCast(ptr) else null;
}

pub fn rawTransfer(handle: AsyncTransferHandle) *c.PjrtxMlxMetalAsyncHostToDeviceTransfer {
    return @ptrCast(@alignCast(handle));
}

pub fn fromRawProgram(handle: ?*c.PjrtxMlxMetalProgram) ?ProgramHandle {
    return if (handle) |ptr| @ptrCast(ptr) else null;
}

pub fn rawProgram(handle: ProgramHandle) *c.PjrtxMlxMetalProgram {
    return @ptrCast(@alignCast(handle));
}
