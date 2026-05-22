const std = @import("std");
const backend_api = @import("backend_selection.zig");
const ir = @import("src/compiler/ir");

const device_memory = @import("device_memory.zig");
const executable_mod = @import("executable.zig");
const executable_schedule = @import("executable_schedule.zig");

const CompiledExecutable = executable_mod.CompiledExecutable;
const Context = executable_mod.Context;
const DeviceMemoryTopology = device_memory.DeviceMemoryTopology;
const ScheduleNodeKind = executable_schedule.NodeKind;

const ExecutableTestContext = struct {
    allocator: std.mem.Allocator,
    backend: backend_api.Backend,
    descriptors: []ir.DeviceDescriptor,
    topology: DeviceMemoryTopology,

    fn init(allocator: std.mem.Allocator) !ExecutableTestContext {
        const backend = backend_api.create();
        const descriptors = try backend.enumerateDevices(allocator, 1);
        errdefer backend.releaseDeviceDescriptors(allocator, descriptors);
        if (descriptors.len == 0) return error.InvalidDeviceCount;
        const topology = try DeviceMemoryTopology.initFromDescriptors(allocator, descriptors);
        errdefer topology.deinit(allocator);
        return .{ .allocator = allocator, .backend = backend, .descriptors = descriptors, .topology = topology };
    }

    fn deinit(self: *ExecutableTestContext) void {
        self.topology.deinit(self.allocator);
        self.backend.releaseDeviceDescriptors(self.allocator, self.descriptors);
        self.* = undefined;
    }

    fn executableContext(self: *ExecutableTestContext) Context {
        return .{ .backend = self.backend, .devices = self.topology.deviceSlice() };
    }
};

const ExecutableTestSupport = struct {
    fn shardingPlan(allocator: std.mem.Allocator, assignment: []const i32) !ir.ShardingPlan {
        return .{ .kind = .replicated, .mesh_name = try allocator.dupe(u8, ""), .device_assignment = try allocator.dupe(i32, assignment) };
    }
};

test "compiled executable materializes per-device scheduled nodes" {
    var ctx = try ExecutableTestContext.init(std.testing.allocator);
    defer ctx.deinit();
    const allocator = std.testing.allocator;
    const assignment = [_]i32{0};
    const dims = [_]i64{4};

    const values = try allocator.alloc(ir.Value, 3);
    errdefer allocator.free(values);
    for (values, 0..) |*value, i| {
        value.* = .{
            .id = .{ .index = @intCast(i) },
            .role = if (i < 2) .parameter else .instruction_result,
            .descriptor = .{ .element_type = .u8, .dims = try allocator.dupe(i64, &dims), .device_id = 0, .memory_id = 0, .shard_index = 0 },
        };
    }

    var parameter_shardings = try allocator.alloc(ir.ShardingPlan, 2);
    parameter_shardings[0] = try ExecutableTestSupport.shardingPlan(allocator, &assignment);
    parameter_shardings[1] = try ExecutableTestSupport.shardingPlan(allocator, &assignment);
    var output_shardings = try allocator.alloc(ir.ShardingPlan, 1);
    output_shardings[0] = try ExecutableTestSupport.shardingPlan(allocator, &assignment);

    var plan = ir.ExecutablePlan{
        .allocator = allocator,
        .options = .{ .num_replicas = 1, .num_partitions = 1, .device_assignment = try allocator.dupe(i32, &assignment) },
        .values = values,
        .parameter_shardings = parameter_shardings,
        .output_shardings = output_shardings,
        .output_ids = try allocator.dupe(ir.ValueId, &.{.{ .index = 2 }}),
        .instructions = try allocator.dupe(ir.PlanInstruction, &.{.{ .kind = .add, .inputs = try allocator.dupe(ir.ValueId, &.{ .{ .index = 0 }, .{ .index = 1 } }), .outputs = try allocator.dupe(ir.ValueId, &.{.{ .index = 2 }}) }}),
    };
    defer plan.deinit();

    var executable = try CompiledExecutable.Testing.initBorrowedResident(allocator, ctx.executableContext(), &plan, .{});
    defer CompiledExecutable.Testing.deinitBorrowedResident(&executable);

    try std.testing.expectEqual(@as(i32, 0), executable.deviceIdAt(0).?);
    try std.testing.expectEqual(@as(usize, 1), CompiledExecutable.Testing.scheduledNodeCount(&executable));
    const node = CompiledExecutable.Testing.scheduledNodeAt(&executable, 0).?;
    try std.testing.expectEqual(ScheduleNodeKind.compute, node.kind);
    try std.testing.expectEqual(@as(usize, 0), node.device_index);
    try std.testing.expectEqual(@as(i32, 0), node.device_id);
    try std.testing.expect(executable.hasResidentBackendExecutable());
    try std.testing.expectEqual(@as(usize, 1), executable.residentProgramInstructionCount());
}

test "compiled executable rejects unsupported backend executable acquisition" {
    var ctx = try ExecutableTestContext.init(std.testing.allocator);
    defer ctx.deinit();
    const allocator = std.testing.allocator;

    var plan = ir.ExecutablePlan{
        .allocator = allocator,
        .options = .{ .num_replicas = 1, .num_partitions = 1, .device_assignment = try allocator.dupe(i32, &.{0}) },
        .values = &.{},
        .parameter_shardings = try allocator.alloc(ir.ShardingPlan, 0),
        .output_shardings = try allocator.alloc(ir.ShardingPlan, 0),
        .output_ids = try allocator.alloc(ir.ValueId, 0),
        .instructions = try allocator.dupe(ir.PlanInstruction, &.{.{ .kind = .custom_call }}),
    };
    defer plan.deinit();

    try std.testing.expectError(error.UnsupportedRuntimeFeature, CompiledExecutable.Testing.initBorrowedResident(allocator, ctx.executableContext(), &plan, .{}));
}
