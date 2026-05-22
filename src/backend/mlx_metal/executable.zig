const std = @import("std");

const ir = @import("src/compiler/ir");
const argument_capture_mod = @import("executable_argument_capture.zig");
const compiled_program_mod = @import("executable_compiled_program.zig");
const compile_mod = @import("executable_compile.zig");
const constants_mod = @import("executable_constants.zig");
const fusion_kernels_mod = @import("executable_fusion_kernels.zig");
const metal_graph_mod = @import("executable_metal_graph.zig");
const mlx_call = @import("mlx_call.zig");
const profiling_mod = @import("profiling.zig");
const program_mod = @import("program.zig");
const stats_mod = @import("executable_stats.zig");

/// Opaque MLX/Metal buffer handle owned by runtime buffer/executable storage.
pub const BufferHandle = *anyopaque;

/// Runtime-visible execution counters for one compiled MLX/Metal executable.
pub const Stats = stats_mod.Stats;

/// Device-local callback context passed to MLX compiled-program builders.
pub const CompiledProgramContext = compiled_program_mod.Context(Executable);

/// Tracks stable captured arguments and dynamic argument indices for MLX compile reuse.
pub const ArgumentCaptureState = argument_capture_mod.State;

/// Owns the compiled backend program, resident constants, and execution counters.
pub const Executable = struct {
    allocator: std.mem.Allocator,
    plan: *const ir.ExecutablePlan,
    device_local_hardware_ids: []i32,
    constant_handles: []?BufferHandle,
    while_constant_handles: []?BufferHandle,
    compiled_program_contexts: []CompiledProgramContext,
    compiled_program_handles: []?mlx_call.ProgramHandle,
    metalcpp_fusion_kernel_handles: []?fusion_kernels_mod.KernelHandle,
    metalcpp_graph_handles: []?metal_graph_mod.Handle,
    argument_capture_states: []ArgumentCaptureState,
    argument_capture_mutex: std.Io.Mutex = .init,
    program: program_mod.Program,
    stats_mutex: std.Io.Mutex = .init,
    stats: Stats = .{},

    /// Decodes an opaque backend executable handle into its owning executable.
    pub fn fromHandle(handle: *anyopaque) *Executable {
        return @ptrCast(@alignCast(handle));
    }

    /// Returns a synchronized copy of the runtime-visible executable counters.
    pub fn snapshotStats(self: *Executable) Stats { return stats_mod.snapshot(self); }

    /// Records a successful device execution and the concrete hardware target.
    pub fn recordExecute(self: *Executable, device_index: usize, local_hardware_id: i32) void { stats_mod.recordExecute(self, device_index, local_hardware_id); }

    /// Records one fused schedule group execution.
    pub fn recordFusionGroupExecute(self: *Executable) void { stats_mod.recordFusionGroupExecute(self); }

    /// Records one MLX compiled-program execution and its output count.
    pub fn recordCompiledProgramExecute(self: *Executable, output_count: usize) void { stats_mod.recordCompiledProgramExecute(self, output_count); }

    /// Records resident executable-level generated Metal graph compilation.
    pub fn recordMetalGraphCompile(self: *Executable, compiled_device_count: usize) void { stats_mod.recordMetalGraphCompile(self, compiled_device_count); }

    /// Records one executable-level generated Metal graph execution.
    pub fn recordMetalGraphExecute(self: *Executable, output_count: usize) void { stats_mod.recordMetalGraphExecute(self, output_count); }

    /// Records one argument-captured MLX program execution.
    pub fn recordCapturedProgramExecute(self: *Executable, dynamic_count: usize, captured_count: usize) void { stats_mod.recordCapturedProgramExecute(self, dynamic_count, captured_count); }

    /// Records an explicit materialization boundary evaluation.
    pub fn recordMaterializationEval(self: *Executable, buffer_count: usize) void { stats_mod.recordMaterializationEval(self, buffer_count); }

    /// Records planned intermediate value releases after their final use.
    pub fn recordReleasedIntermediateValues(self: *Executable, count: usize) void { stats_mod.recordReleasedIntermediateValues(self, count); }

    /// Records a schedule node that borrowed a resident constant buffer.
    pub fn recordBorrowedConstantNode(self: *Executable) void { stats_mod.recordBorrowedConstantNode(self); }

    /// Accumulates one backend execution profile into executable counters.
    pub fn recordExecuteProfile(self: *Executable, profile: profiling_mod.Execute) void { stats_mod.recordExecuteProfile(self, profile); }

    /// Returns a resident generated Metal-cpp fusion kernel for one device/group.
    pub fn metalCppFusionKernel(self: *const Executable, device_index: usize, group_id: usize) ?fusion_kernels_mod.KernelHandle {
        return fusion_kernels_mod.get(self.metalcpp_fusion_kernel_handles, self.program.fusion_groups.len, device_index, group_id);
    }

    /// Returns a resident executable-level generated Metal graph for one device.
    pub fn metalCppGraph(self: *const Executable, device_index: usize) ?metal_graph_mod.Handle {
        return metal_graph_mod.get(self.metalcpp_graph_handles, device_index);
    }

    /// Locks the executable argument-capture state for mutation by execution.
    pub fn lockArgumentCapture(self: *Executable) void { argument_capture_mod.lock(self); }

    /// Unlocks the executable argument-capture state after mutation.
    pub fn unlockArgumentCapture(self: *Executable) void { argument_capture_mod.unlock(self); }

    /// Releases all resident backend storage owned by this executable.
    pub fn deinit(self: *Executable) void {
        const allocator = self.allocator;
        metal_graph_mod.destroy(self.metalcpp_graph_handles);
        fusion_kernels_mod.destroy(self.metalcpp_fusion_kernel_handles);
        compiled_program_mod.destroyHandles(self.compiled_program_handles);
        argument_capture_mod.destroyStates(allocator, self.argument_capture_states);
        self.program.deinit();
        constants_mod.destroy(self.constant_handles);
        constants_mod.destroy(self.while_constant_handles);
        allocator.free(self.metalcpp_fusion_kernel_handles);
        allocator.free(self.metalcpp_graph_handles);
        allocator.free(self.compiled_program_handles);
        allocator.free(self.compiled_program_contexts);
        allocator.free(self.argument_capture_states);
        allocator.free(self.constant_handles);
        allocator.free(self.while_constant_handles);
        allocator.free(self.device_local_hardware_ids);
        allocator.destroy(self);
    }
};

/// Compiles an executable plan into a resident MLX/Metal executable.
pub fn compile(
    allocator: std.mem.Allocator,
    plan: *const ir.ExecutablePlan,
    device_local_hardware_ids: []const i32,
    comptime build_callback: mlx_call.ProgramBuildCallback,
) program_mod.Error!?BufferHandle {
    return compile_mod.run(Executable, CompiledProgramContext, allocator, plan, device_local_hardware_ids, build_callback);
}

/// Returns the resident constant slot for a plan instruction on one device.
pub fn constantIndex(instruction_count: usize, device_index: usize, instruction_index: usize) usize {
    return constants_mod.constantIndex(instruction_count, device_index, instruction_index);
}

/// Returns the resident while-pattern constant slot for one control-flow node.
pub fn whileConstantIndex(control_flow_count: usize, device_index: usize, control_flow_index: usize, constant_index: usize) usize {
    return constants_mod.whileConstantIndex(control_flow_count, device_index, control_flow_index, constant_index);
}
