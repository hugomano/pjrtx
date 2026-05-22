const std = @import("std");

const ir = @import("src/compiler/ir");
const mlx_metal = @import("src/backend/mlx_metal");
const manifest = @import("msl_manifest.zig");
const source = @import("msl_source.zig");

/// Writes experimental Metal source artifacts for one backend program when requested.
pub fn dumpIfEnabled(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, program: *const mlx_metal.Program) !void {
    const dir = artifactDir() orelse return;
    const io = std.Io.Threaded.global_single_threaded.io();
    try std.Io.Dir.cwd().createDirPath(io, dir);

    const stem = try std.fmt.allocPrint(allocator, "pjrtx_metalcpp_{x}", .{@intFromPtr(plan)});
    defer allocator.free(stem);
    const metal_path = try std.fmt.allocPrint(allocator, "{s}/{s}.metal", .{ dir, stem });
    defer allocator.free(metal_path);
    const metalcpp_path = try std.fmt.allocPrint(allocator, "{s}/{s}.metalcpp.cc", .{ dir, stem });
    defer allocator.free(metalcpp_path);
    const manifest_path = try std.fmt.allocPrint(allocator, "{s}/{s}.manifest.txt", .{ dir, stem });
    defer allocator.free(manifest_path);

    var metal_source = std.Io.Writer.Allocating.init(allocator);
    defer metal_source.deinit();
    try source.writeMsl(&metal_source.writer, plan, program);
    try writeFile(io, metal_path, metal_source.written());

    var metalcpp_source = std.Io.Writer.Allocating.init(allocator);
    defer metalcpp_source.deinit();
    try source.writeMetalCppHost(&metalcpp_source.writer, stem, plan, program);
    try writeFile(io, metalcpp_path, metalcpp_source.written());

    var manifest_text = std.Io.Writer.Allocating.init(allocator);
    defer manifest_text.deinit();
    try manifest.writeManifest(&manifest_text.writer, stem, plan, program);
    try writeFile(io, manifest_path, manifest_text.written());
}

fn artifactDir() ?[]const u8 {
    if (envText("PJRTX_METALCPP_ARTIFACT_DIR")) |path| return path;
    return envText("PJRTX_METALCPP_MSL_DIR");
}

fn envText(comptime name: [:0]const u8) ?[]const u8 {
    const value = std.c.getenv(name) orelse return null;
    const text = std.mem.span(value);
    if (text.len == 0) return null;
    return text;
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

test "metalcpp artifact writer emits executable shape" {
    var output = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer output.deinit();

    var plan = ir.ExecutablePlan{
        .allocator = std.testing.allocator,
        .options = .{
            .num_replicas = 1,
            .num_partitions = 1,
            .device_assignment = &.{},
        },
        .parameter_shardings = &.{},
        .output_shardings = &.{},
        .instructions = &.{},
    };
    const program = mlx_metal.Program{
        .allocator = std.testing.allocator,
        .values = &.{},
        .nodes = &.{},
        .edges = &.{},
        .schedule = &.{},
        .subprograms = &.{},
        .control_flows = &.{},
        .fusion_groups = &.{},
        .materialization_boundaries = &.{},
    };

    try source.writeMsl(&output.writer, &plan, &program);
    try std.testing.expect(std.mem.indexOf(u8, output.written(), "instructions=0") != null);
}

test "metalcpp artifact writer emits runnable dense elementwise source and manifest labels" {
    const dims = [_]i64{4};
    var values = [_]ir.Value{
        .{ .id = .{ .index = 0 }, .role = .parameter, .descriptor = .{ .element_type = .f32, .dims = dims[0..], .device_id = 0, .memory_id = 0, .shard_index = 0 } },
        .{ .id = .{ .index = 1 }, .role = .parameter, .descriptor = .{ .element_type = .f32, .dims = dims[0..], .device_id = 0, .memory_id = 0, .shard_index = 0 } },
        .{ .id = .{ .index = 2 }, .role = .instruction_result, .descriptor = .{ .element_type = .f32, .dims = dims[0..], .device_id = 0, .memory_id = 0, .shard_index = 0 } },
    };
    const inputs = [_]ir.ValueId{ .{ .index = 0 }, .{ .index = 1 } };
    const outputs = [_]ir.ValueId{.{ .index = 2 }};
    var instructions = [_]ir.PlanInstruction{.{
        .kind = .add,
        .inputs = inputs[0..],
        .outputs = outputs[0..],
    }};
    var plan = ir.ExecutablePlan{
        .allocator = std.testing.allocator,
        .options = .{ .num_replicas = 1, .num_partitions = 1, .device_assignment = &.{} },
        .values = values[0..],
        .parameter_shardings = &.{},
        .output_shardings = &.{},
        .instructions = instructions[0..],
    };
    const program = emptyProgram();

    var msl_output = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer msl_output.deinit();
    try source.writeMsl(&msl_output.writer, &plan, &program);
    const msl = msl_output.written();
    try std.testing.expect(std.mem.indexOf(u8, msl, "struct PjrtxDenseElementwiseShape") != null);
    try std.testing.expect(std.mem.indexOf(u8, msl, "kernel void pjrtx_i0_add_f32_dense(") != null);
    try std.testing.expect(std.mem.indexOf(u8, msl, "out[gid] = pjrtx_dense_elementwise_add(lhs[gid], rhs[gid]);") != null);

    var manifest_output = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer manifest_output.deinit();
    try manifest.writeManifest(&manifest_output.writer, "test_program", &plan, &program);
    const text = manifest_output.written();
    try std.testing.expect(std.mem.indexOf(u8, text, "manifest_label=dense_elementwise runner=metalcpp_dense_elementwise_f32 kernel=pjrtx_i0_add_f32_dense element_count=4") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "kernel instruction=0 label=dense_elementwise runner=metalcpp_dense_elementwise_f32 symbol=pjrtx_i0_add_f32_dense source=test_program.metal op=add dtype=f32 element_count=4 shape=[4] inputs=[v0:f32[4],v1:f32[4]] outputs=[v2:f32[4]] buffers=lhs:0,rhs:1,out:2,shape:3") != null);
}

test "metalcpp manifest includes value shapes and schedule summaries" {
    var output = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer output.deinit();

    const lhs_dims = [_]i64{ 2, 4 };
    const rhs_dims = [_]i64{ 4, 8 };
    const out_dims = [_]i64{ 2, 8 };
    var values = [_]ir.Value{
        .{ .id = .{ .index = 0 }, .role = .parameter, .descriptor = .{ .element_type = .f16, .dims = lhs_dims[0..], .device_id = 0, .memory_id = 0, .shard_index = 0 } },
        .{ .id = .{ .index = 1 }, .role = .parameter, .descriptor = .{ .element_type = .f16, .dims = rhs_dims[0..], .device_id = 0, .memory_id = 0, .shard_index = 0 } },
        .{ .id = .{ .index = 2 }, .role = .instruction_result, .descriptor = .{ .element_type = .f16, .dims = out_dims[0..], .device_id = 0, .memory_id = 0, .shard_index = 0 } },
    };
    const dot_inputs = [_]ir.ValueId{ .{ .index = 0 }, .{ .index = 1 } };
    const dot_outputs = [_]ir.ValueId{.{ .index = 2 }};
    const lhs_contract = [_]i64{1};
    const rhs_contract = [_]i64{0};
    var instructions = [_]ir.PlanInstruction{.{
        .kind = .dot_general,
        .inputs = dot_inputs[0..],
        .outputs = dot_outputs[0..],
        .lhs_contracting_dimensions = lhs_contract[0..],
        .rhs_contracting_dimensions = rhs_contract[0..],
    }};
    var plan = ir.ExecutablePlan{
        .allocator = std.testing.allocator,
        .options = .{ .num_replicas = 1, .num_partitions = 1, .device_assignment = &.{} },
        .values = values[0..],
        .parameter_shardings = &.{},
        .output_shardings = &.{},
        .instructions = instructions[0..],
    };

    var program_values = [_]mlx_metal.ProgramValue{
        .{ .value_id = .{ .index = 0 }, .byte_size = 16, .last_use_node = 0 },
        .{ .value_id = .{ .index = 1 }, .byte_size = 32, .last_use_node = 0 },
        .{ .value_id = .{ .index = 2 }, .byte_size = 32, .producer_node = 0, .is_output = true },
    };
    var nodes = [_]mlx_metal.ProgramNode{.{
        .instruction_index = 0,
        .kind = .matmul,
        .inputs = dot_inputs[0..],
        .outputs = dot_outputs[0..],
        .materializes = true,
    }};
    var schedule = [_]mlx_metal.ProgramScheduleItem{.{ .kind = .node, .index = 0 }};
    const program = mlx_metal.Program{
        .allocator = std.testing.allocator,
        .values = program_values[0..],
        .nodes = nodes[0..],
        .edges = &.{},
        .schedule = schedule[0..],
        .subprograms = &.{},
        .control_flows = &.{},
        .fusion_groups = &.{},
        .materialization_boundaries = &.{},
    };

    try manifest.writeManifest(&output.writer, "test_program", &plan, &program);
    const text = output.written();
    try std.testing.expect(std.mem.indexOf(u8, text, "backend_program_manifest version=1") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "v0 role=parameter storage=tensor dtype=f16 shape=[2,4]") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "i0 op=dot_general inputs=[v0:f16[2,4],v1:f16[4,8]] outputs=[v2:f16[2,8]] lhs_contract=[1] rhs_contract=[0]") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "n0 instruction=0 op=dot_general kind=matmul") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "s0 kind=node index=0 count=1 node_kind=matmul instruction=0 op=dot_general") != null);
}

fn emptyProgram() mlx_metal.Program {
    return .{
        .allocator = std.testing.allocator,
        .values = &.{},
        .nodes = &.{},
        .edges = &.{},
        .schedule = &.{},
        .subprograms = &.{},
        .control_flows = &.{},
        .fusion_groups = &.{},
        .materialization_boundaries = &.{},
    };
}

