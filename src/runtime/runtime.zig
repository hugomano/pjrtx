const backend_api = @import("backend_selection.zig");
const ir = @import("src/compiler/ir");

const buffer_mod = @import("buffer.zig");
const client_mod = @import("client.zig");
const custom_call = @import("custom_call.zig");
const device_memory = @import("device_memory.zig");
const event_mod = @import("event.zig");
const execution_mod = @import("execution.zig");
const executable_mod = @import("executable.zig");
const executable_cache = @import("executable_cache.zig");

comptime {
    if (@import("builtin").is_test) {
        _ = @import("buffer_tests.zig");
        _ = @import("client_residency_tests.zig");
        _ = @import("executable_tests.zig");
    }
}

/// Compiler-owned element type vocabulary exposed for PJRT buffer metadata.
pub const BufferType = ir.BufferType;

/// Backend residency counters surfaced through runtime executable metadata.
pub const ExecutableStats = backend_api.ExecutableStats;

/// Opaque backend transfer handle owned by async host-to-device copy calls.
pub const AsyncHostToDeviceTransferHandle = client_mod.AsyncHostToDeviceTransferHandle;

/// Error set for creating device-resident runtime buffers.
pub const BufferCreateError = buffer_mod.BufferCreateError;

/// Error set for reserving buffers completed by asynchronous backend transfers.
pub const PendingBackendTransferBufferError = buffer_mod.PendingBackendTransferBufferError;

/// Error set for starting asynchronous host-to-device transfers.
pub const AsyncTransferBeginError = client_mod.AsyncTransferBeginError;

/// Error set for writing asynchronous host-to-device transfer chunks.
pub const AsyncTransferWriteError = client_mod.AsyncTransferWriteError;

/// Error set for finishing asynchronous host-to-device transfers.
pub const AsyncTransferFinishError = client_mod.AsyncTransferFinishError;

/// Runtime-owned PJRT device record.
pub const Device = device_memory.Device;

/// Runtime-owned PJRT memory record and memory accounting.
pub const Memory = device_memory.Memory;

/// Runtime memory accounting counters.
pub const MemoryStats = device_memory.MemoryStats;

/// Runtime readiness event shared by buffers and execution results.
pub const Event = event_mod.Event;

/// Custom-call registration kind accepted by the Metal/MLX runtime.
pub const CustomCallKind = custom_call.CustomCallKind;

/// Custom-call registration payload forwarded to the backend registry.
pub const CustomCallRegistration = custom_call.CustomCallRegistration;

/// Error set for runtime custom-call registry mutation.
pub const CustomCallRegistrationError = custom_call.CustomCallRegistrationError;

/// Result of trimming idle resident backend executables.
pub const ExecutableCacheTrim = executable_cache.Trim;

/// Runtime-owned device buffer with placement, readiness, and backend storage.
pub const Buffer = buffer_mod.Buffer;

/// Runtime-owned compiled executable returned by client compilation.
pub const CompiledExecutable = executable_mod.CompiledExecutable;

/// Result of executing one device slice of a compiled executable.
pub const ExecutionResult = execution_mod.ExecutionResult;

/// Error set for per-device compiled executable execution.
pub const ExecutionError = execution_mod.ExecutionError;

/// Options for creating a Metal/MLX runtime client.
pub const ClientCreateOptions = client_mod.ClientCreateOptions;

/// Program bytes and compile options submitted to the runtime client.
pub const CompileProgram = client_mod.CompileProgram;

/// Error set for compiling a program through the runtime client.
pub const CompileProgramError = client_mod.CompileProgramError;

/// Runtime client owning topology, memory, executable cache, and backend access.
pub const Client = client_mod.Client;

/// Creates a runtime client backed by the process-wide Metal/MLX backend.
pub const createClient = client_mod.createClient;

/// Executes one device slice of a compiled executable through resident backend storage.
pub const executeCompiledExecutable = execution_mod.executeCompiledExecutable;

/// Registers a fully typed custom-call target.
pub const registerCustomCall = custom_call.registerCustomCall;

/// Registers a named binary custom-call target.
pub const registerBinaryCustomCall = custom_call.registerBinaryCustomCall;

/// Registers an identity custom-call target.
pub const registerIdentityCustomCall = custom_call.registerIdentityCustomCall;

/// Registers a named unary custom-call target.
pub const registerUnaryCustomCall = custom_call.registerUnaryCustomCall;

/// Registers the built-in square-root custom-call marker.
pub const registerUnarySqrtCustomCall = custom_call.registerUnarySqrtCustomCall;

/// Registers the built-in binary add custom-call marker.
pub const registerBinaryAddCustomCall = custom_call.registerBinaryAddCustomCall;

/// Removes a custom-call target from the backend registry.
pub const unregisterCustomCall = custom_call.unregisterCustomCall;
