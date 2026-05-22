const std = @import("std");

const ir = @import("src/compiler/ir");
const profiling = @import("profiling.zig");
const program_request_mod = @import("executable_metal_graph_program_request.zig");

/// Writes optional executable Metal graph MSL and step manifests.
pub const ArtifactDump = struct {
    pub fn dumpProgramRequestIfEnabled(
        allocator: std.mem.Allocator,
        plan: *const ir.ExecutablePlan,
        device_local_hardware_id: i32,
        request: program_request_mod.Request,
    ) !void {
        const dir = profiling.metalCppMslDir() orelse return;
        const io = profiling.backendIo();
        try std.Io.Dir.cwd().createDirPath(io, dir);

        const stem = try std.fmt.allocPrint(allocator, "pjrtx_metalcpp_executable_{x}_device_{d}", .{ @intFromPtr(plan), device_local_hardware_id });
        defer allocator.free(stem);
        const manifest_path = try std.fmt.allocPrint(allocator, "{s}/{s}.steps.txt", .{ dir, stem });
        defer allocator.free(manifest_path);

        var manifest = std.Io.Writer.Allocating.init(allocator);
        defer manifest.deinit();
        try manifest.writer.print(
            \\executable_steps version=1 plan=0x{x} device={d} values={d} inputs={d} outputs={d} steps={d}
            \\
        , .{
            @intFromPtr(plan),
            device_local_hardware_id,
            request.value_specs.len,
            request.input_values.len,
            request.output_values.len,
            request.steps.len,
        });

        for (request.steps, 0..) |step, step_index| {
            const source_path = if (step.source.len == 0)
                null
            else
                try std.fmt.allocPrint(allocator, "{s}/{s}_step_{d}.metal", .{ dir, stem, step_index });
            defer if (source_path) |path| allocator.free(path);
            if (source_path) |path| try writeFile(io, path, step.source);

            try manifest.writer.print("step={d} alias={d} kernel=\"{s}\"", .{
                step_index,
                @intFromBool(step.source.len == 0),
                step.kernel_name,
            });
            if (source_path) |path| try manifest.writer.print(" source=\"{s}\"", .{path});
            try manifest.writer.print(" element_count={d} threads_per_threadgroup={d}", .{ step.element_count, step.threads_per_threadgroup });
            try writeU64List(&manifest.writer, " inputs", step.inputs);
            try writeU64List(&manifest.writer, " outputs", step.outputs);
            try writeU64List(&manifest.writer, " release_values", step.release_values);
            try manifest.writer.writeByte('\n');
        }

        try writeFile(io, manifest_path, manifest.written());
    }

    fn writeFile(io: std.Io, path: []const u8, bytes: []const u8) !void {
        if (std.Io.Dir.path.isAbsolute(path)) {
            var file = try std.Io.Dir.createFileAbsolute(io, path, .{ .truncate = true });
            defer file.close(io);
            try file.writeStreamingAll(io, bytes);
            return;
        }
        try std.Io.Dir.cwd().writeFile(io, .{
            .sub_path = path,
            .data = bytes,
            .flags = .{ .truncate = true },
        });
    }

    fn writeU64List(writer: *std.Io.Writer, label: []const u8, values: []const u64) !void {
        try writer.print("{s}=[", .{label});
        for (values, 0..) |value, index| {
            if (index != 0) try writer.writeByte(',');
            try writer.print("{d}", .{value});
        }
        try writer.writeByte(']');
    }
};
