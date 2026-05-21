const std = @import("std");
const ir = @import("src/compiler/ir");

const execution_context = @import("execution_context.zig");

const Context = execution_context.Context;

/// Classifies scheduled runtime executable nodes by execution behavior.
pub const NodeKind = enum {
    constant,
    parameter,
    compute,
    collective,
    custom_call,
    control_flow,
    structural,
};

/// One scheduled compiler instruction on one executable device slot.
pub const Node = struct {
    instruction_index: usize,
    device_index: usize,
    device_id: i32,
    kind: NodeKind,
};

/// Owns per-device executable assignment and scheduled executable nodes.
pub const Schedule = struct {
    allocator: std.mem.Allocator,
    device_ids: []i32,
    device_local_hardware_ids: []i32,
    nodes: []Node,

    /// Builds schedule metadata from compiler device assignment and plan instructions.
    pub fn init(allocator: std.mem.Allocator, context: Context, plan: *const ir.ExecutablePlan) !Schedule {
        const device_count = plan.options.numDevices();
        if (device_count == 0 or device_count > context.deviceCount()) return error.InvalidGraph;

        const device_ids = try allocator.alloc(i32, device_count);
        errdefer allocator.free(device_ids);

        const device_local_hardware_ids = try allocator.alloc(i32, device_count);
        errdefer allocator.free(device_local_hardware_ids);

        for (device_ids, 0..) |*device_id, i| {
            device_id.* = if (plan.options.device_assignment.len != 0)
                plan.options.device_assignment[i]
            else
                context.defaultDeviceIdAt(i) orelse return error.InvalidGraph;
            const device = context.lookupDevice(device_id.*) orelse return error.InvalidGraph;
            device_local_hardware_ids[i] = device.local_hardware_id;
        }

        const node_count = std.math.mul(usize, plan.instructions.len, device_count) catch return error.InvalidGraph;
        const nodes = try allocator.alloc(Node, node_count);
        errdefer allocator.free(nodes);

        var out: usize = 0;
        for (0..device_count) |device_index| {
            for (plan.instructions, 0..) |instruction, instruction_index| {
                nodes[out] = .{
                    .instruction_index = instruction_index,
                    .device_index = device_index,
                    .device_id = device_ids[device_index],
                    .kind = nodeKind(instruction.kind),
                };
                out += 1;
            }
        }

        return .{
            .allocator = allocator,
            .device_ids = device_ids,
            .device_local_hardware_ids = device_local_hardware_ids,
            .nodes = nodes,
        };
    }

    /// Releases schedule-owned arrays.
    pub fn deinit(self: *Schedule) void {
        self.allocator.free(self.device_local_hardware_ids);
        self.allocator.free(self.nodes);
        self.allocator.free(self.device_ids);
        self.* = undefined;
    }

    /// Returns the number of runtime device slots embedded in this schedule.
    pub fn deviceCount(self: *const Schedule) usize {
        return self.device_ids.len;
    }

    /// Returns the stable PJRT device id assigned to one schedule device slot.
    pub fn deviceIdAt(self: *const Schedule, index: usize) ?i32 {
        if (index >= self.device_ids.len) return null;
        return self.device_ids[index];
    }

    /// Returns the number of scheduled instruction/device nodes.
    pub fn nodeCount(self: *const Schedule) usize {
        return self.nodes.len;
    }

    /// Returns one scheduled instruction/device node.
    pub fn nodeAt(self: *const Schedule, index: usize) ?Node {
        if (index >= self.nodes.len) return null;
        return self.nodes[index];
    }
};

fn nodeKind(kind: ir.PlanInstructionKind) NodeKind {
    return switch (kind) {
        .constant => .constant,
        .custom_call => .custom_call,
        .while_ => .control_flow,
        .tuple, .get_tuple_element, .optimization_barrier => .structural,
        else => .compute,
    };
}
