const std = @import("std");

const ir = @import("src/compiler/ir");
const builder = @import("executable_metal_graph_graph_request_build.zig");
const inputs = @import("executable_metal_graph_graph_request_inputs.zig");
const resident_inputs = @import("executable_metal_graph_resident_inputs.zig");
const storage = @import("executable_metal_graph_graph_request_storage.zig");
const writer_mod = @import("executable_metal_graph_graph_request_writer.zig");
const metalcpp_call = @import("metalcpp_call.zig");
const program_mod = @import("program.zig");
const types = @import("execution_types.zig");

/// Owns one legacy single-kernel executable graph request.
pub const Request = struct {
    storage: storage.Request,

    /// Builds the request if the executable plan can be represented as one Metal kernel.
    pub fn init(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan) program_mod.Error!?Request {
        const built = (try builder.GraphRequestBuilder.init(allocator, plan)) orelse return null;
        return .{ .storage = built };
    }

    /// Releases generated expressions and Metal tensor specs owned by this request.
    pub fn deinit(self: *Request, allocator: std.mem.Allocator) void {
        self.storage.deinit(allocator);
        self.* = undefined;
    }

    /// Collects runtime arguments plus resident constants for this request.
    pub fn inputHandles(
        self: Request,
        allocator: std.mem.Allocator,
        source: resident_inputs.Source,
        arguments: []const types.BufferHandle,
    ) types.Error![]types.BufferHandle {
        return inputs.GraphRequestInputs.collect(allocator, self.storage, source, arguments);
    }

    /// Emits the Metal Shading Language for this single-kernel request.
    pub fn writeMsl(self: Request, writer: *std.Io.Writer, kernel_name: []const u8) !void {
        try writer_mod.GraphRequestWriter.write(self.storage, writer, kernel_name);
    }

    /// Returns the Metal-cpp input tensor specs for program creation.
    pub fn inputSpecs(self: Request) []const metalcpp_call.TensorSpec {
        return self.storage.input_specs;
    }

    /// Returns the Metal-cpp output tensor specs for program creation.
    pub fn outputSpecs(self: Request) []const metalcpp_call.TensorSpec {
        return self.storage.output_specs;
    }

    /// Returns the element count executed by the generated kernel.
    pub fn elementCount(self: Request) u64 {
        return self.storage.element_count;
    }
};
