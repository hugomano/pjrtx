const std = @import("std");

const ir = @import("src/compiler/ir");
const build = @import("executable_metal_graph_fusion_request_build.zig");
const metalcpp_call = @import("metalcpp_call.zig");
const program_mod = @import("program.zig");
const storage = @import("executable_metal_graph_fusion_request_storage.zig");
const writer_mod = @import("executable_metal_graph_fusion_request_writer.zig");

/// Owns generated MSL request state for one executable fusion group.
pub const Request = struct {
    storage: storage.Request,

    /// Builds a fusion request when the group is a view/elementwise Metal kernel.
    pub fn init(
        allocator: std.mem.Allocator,
        plan: *const ir.ExecutablePlan,
        program: *const program_mod.Program,
        group: program_mod.FusionGroup,
    ) program_mod.Error!?Request {
        const built = (try build.FusionRequestBuilder.init(allocator, plan, program, group)) orelse return null;
        return .{ .storage = built };
    }

    /// Releases generated expressions, statements, and tensor specs.
    pub fn deinit(self: *Request, allocator: std.mem.Allocator) void {
        self.storage.deinit(allocator);
        self.* = undefined;
    }

    /// Emits the Metal Shading Language for this fusion request.
    pub fn writeMsl(self: Request, allocator: std.mem.Allocator, writer: *std.Io.Writer, kernel_name: []const u8) !void {
        try writer_mod.FusionRequestWriter.write(self.storage, allocator, writer, kernel_name);
    }

    /// Returns the Metal-cpp input tensor specs for this fusion kernel.
    pub fn inputSpecs(self: Request) []const metalcpp_call.TensorSpec {
        return self.storage.input_specs;
    }

    /// Returns the Metal-cpp output tensor specs for this fusion kernel.
    pub fn outputSpecs(self: Request) []const metalcpp_call.TensorSpec {
        return self.storage.output_specs;
    }

    /// Returns the executable value ids consumed by this fusion kernel.
    pub fn inputValues(self: Request) []const ir.ValueId {
        return self.storage.input_values;
    }

    /// Returns the executable value ids produced by this fusion kernel.
    pub fn outputValues(self: Request) []const ir.ValueId {
        return self.storage.output_values;
    }

    /// Returns the element count executed by the generated kernel.
    pub fn elementCount(self: Request) u64 {
        return self.storage.element_count;
    }
};
