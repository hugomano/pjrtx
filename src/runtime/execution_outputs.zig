const std = @import("std");
const backend_api = @import("src/backend/mlx_metal");
const ir = @import("src/compiler/ir");

const buffer_mod = @import("buffer.zig");
const executable_mod = @import("executable.zig");
const execution_donation = @import("execution_donation.zig");

const Buffer = buffer_mod.Buffer;
const CompiledExecutable = executable_mod.CompiledExecutable;
const ExecutionContext = executable_mod.Context;

/// Wraps backend-owned output handles into runtime buffers after plan validation.
pub fn wrapBackendOutputs(
    allocator: std.mem.Allocator,
    context: ExecutionContext,
    executable: *CompiledExecutable,
    device_index: usize,
    arguments: []const *Buffer,
    backend_outputs: []backend_api.ExecutableOutput,
) ![]*Buffer {
    defer allocator.free(backend_outputs);

    var wrapped_backend_outputs: usize = 0;
    errdefer {
        for (backend_outputs[wrapped_backend_outputs..]) |output| context.destroyBackendBuffer(output.handle);
    }

    if (backend_outputs.len != executable.outputCount()) return error.Internal;
    for (backend_outputs, 0..) |output, output_index| {
        if (!descriptorMatches(executable, output_index, output)) return error.Internal;
    }

    const device = context.deviceAt(device_index) orelse return error.InvalidArgument;
    const memory = device.default_memory;
    const outputs = try allocator.alloc(*Buffer, backend_outputs.len);
    errdefer allocator.free(outputs);

    var initialized: usize = 0;
    errdefer {
        for (outputs[0..initialized]) |buffer| buffer.deinit();
    }

    var donation_delta = execution_donation.Delta{};
    errdefer execution_donation.rollback(executable, donation_delta);

    for (backend_outputs, 0..) |output, i| {
        const adopted_handle = try execution_donation.takeOutputHandle(
            context,
            executable,
            i,
            output,
            arguments,
            &donation_delta,
        );
        outputs[i] = Buffer.initBackendHandle(
            allocator,
            context.bufferStorageBackend(),
            output.element_type,
            output.dims,
            device,
            memory,
            device_index,
            output.byte_size,
            adopted_handle,
        ) catch |err| {
            if (adopted_handle != output.handle) context.destroyBackendBuffer(adopted_handle);
            return err;
        };
        initialized += 1;
        wrapped_backend_outputs = initialized;
    }
    return outputs;
}

fn descriptorMatches(executable: *const CompiledExecutable, output_index: usize, output: backend_api.ExecutableOutput) bool {
    return executable.backendOutputMatches(output_index, output);
}

const OutputTestSupport = struct {
    fn shardingPlan(allocator: std.mem.Allocator, assignment: []const i32) !ir.ShardingPlan {
        return .{
            .kind = .replicated,
            .mesh_name = try allocator.dupe(u8, ""),
            .device_assignment = try allocator.dupe(i32, assignment),
        };
    }
};

test "backend output descriptor validation matches executable plan outputs" {
    const allocator = std.testing.allocator;
    const assignment = [_]i32{0};
    const dims = [_]i64{ 2, 2 };

    const values = try allocator.alloc(ir.Value, 1);
    errdefer allocator.free(values);
    values[0] = .{
        .id = .{ .index = 0 },
        .role = .instruction_result,
        .descriptor = .{
            .element_type = .f32,
            .dims = try allocator.dupe(i64, &dims),
            .device_id = 0,
            .memory_id = 0,
            .shard_index = 0,
        },
    };

    var output_shardings = try allocator.alloc(ir.ShardingPlan, 1);
    output_shardings[0] = try OutputTestSupport.shardingPlan(allocator, &assignment);

    var plan = ir.ExecutablePlan{
        .allocator = allocator,
        .options = .{
            .num_replicas = 1,
            .num_partitions = 1,
            .device_assignment = try allocator.dupe(i32, &assignment),
        },
        .values = values,
        .parameter_shardings = try allocator.alloc(ir.ShardingPlan, 0),
        .output_shardings = output_shardings,
        .output_ids = try allocator.dupe(ir.ValueId, &.{.{ .index = 0 }}),
        .instructions = try allocator.alloc(ir.PlanInstruction, 0),
    };
    defer plan.deinit();

    const compiled = CompiledExecutable{
        .plan = &plan,
        .schedule = undefined,
        .residency = undefined,
        .optimized_program = &.{},
        .fingerprint = &.{},
        .cache_hit = false,
        .backend_stats = .{},
        .resident_released = true,
    };
    const handle: backend_api.BufferHandle = @ptrFromInt(0x1000);
    try std.testing.expect(descriptorMatches(&compiled, 0, .{
        .handle = handle,
        .element_type = .f32,
        .dims = &dims,
        .byte_size = 16,
    }));
    try std.testing.expect(!descriptorMatches(&compiled, 0, .{
        .handle = handle,
        .element_type = .u32,
        .dims = &dims,
        .byte_size = 16,
    }));
    try std.testing.expect(!descriptorMatches(&compiled, 0, .{
        .handle = handle,
        .element_type = .f32,
        .dims = &.{4},
        .byte_size = 16,
    }));
    try std.testing.expect(!descriptorMatches(&compiled, 0, .{
        .handle = handle,
        .element_type = .f32,
        .dims = &dims,
        .byte_size = 12,
    }));
    try std.testing.expect(!descriptorMatches(&compiled, 1, .{
        .handle = handle,
        .element_type = .f32,
        .dims = &dims,
        .byte_size = 16,
    }));
}
