const std = @import("std");

const runtime = @import("src/runtime");
const c = @import("c");
const abi = @import("pjrt_abi.zig");

pub const allocator = std.heap.c_allocator;

pub const io = std.Io.Threaded.global_single_threaded.io();

pub const platform_name = "pjrtx";

pub const platform_version = "PjRTx Metal/MLX";

pub const device_kind = "Metal";

pub const plugin_name = "PjRTx";

pub const stablehlo_current_version = [_]i64{ 1, 16, 0 };
pub const stablehlo_minimum_version = [_]i64{ 1, 0, 0 };

pub const backend_option = "pjrtx_backend";

pub const default_memory_kind = "device";

pub const SerializedTopology = struct {
    bytes: []u8,
};

pub const Executable = struct {
    client: *runtime.Client,
    plan: *runtime.ExecutablePlan,
    graph: runtime.ExecutableGraph,
    logical_ids: []c.PJRT_LogicalDeviceIds,
    optimized_program: []u8,
    parameter_memory_kinds: [][*c]const u8,
    parameter_memory_kind_sizes: []usize,
    output_memory_kinds: [][*c]const u8,
    output_memory_kind_sizes: []usize,
    fingerprint: []u8,
    name: []const u8 = "pjrtx_executable",
    deleted: bool = false,
    graph_released: bool = false,

    pub fn deinit(self: *Executable) void {
        self.releaseGraph();
        allocator.free(self.fingerprint);
        allocator.free(self.output_memory_kind_sizes);
        allocator.free(self.output_memory_kinds);
        allocator.free(self.parameter_memory_kind_sizes);
        allocator.free(self.parameter_memory_kinds);
        allocator.free(self.optimized_program);
        allocator.free(self.logical_ids);
        self.plan.deinit();
        allocator.destroy(self.plan);
        allocator.destroy(self);
    }

    pub fn releaseGraph(self: *Executable) void {
        if (self.graph_released) return;
        self.graph.deinit();
        self.graph_released = true;
    }
};

var attrs_ready = false;
var attrs: [5]abi.NamedValue = undefined;

pub fn initAttrs() []abi.NamedValue {
    if (attrs_ready) return attrs[0..];
    attrs = .{
        abi.NamedValue.string("plugin_name", plugin_name),
        abi.NamedValue.string("xla_version", "local"),
        abi.NamedValue.int64List("stablehlo_current_version", stablehlo_current_version.len, &stablehlo_current_version),
        abi.NamedValue.int64List("stablehlo_minimum_version", stablehlo_minimum_version.len, &stablehlo_minimum_version),
        abi.NamedValue.string("pjrtx_default_backend", "metal_mlx"),
    };

    attrs_ready = true;
    return attrs[0..];
}

pub fn clampI64(value: u64) i64 {
    return @intCast(@min(value, @as(u64, @intCast(std.math.maxInt(i64)))));
}
