const std = @import("std");

const ir = @import("src/compiler/ir");
const program_mod = @import("program.zig");
const profiling = @import("profiling.zig");
const text = @import("metalcpp_msl_text.zig");

const DenseElementwiseKernel = struct {
    instruction_index: usize,
    kind: ir.PlanInstructionKind,
    input_ids: []const ir.ValueId,
    output_id: ir.ValueId,
    descriptor: ir.BufferDescriptor,
    element_count: usize,

    fn init(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize) ?DenseElementwiseKernel {
        if (!isDenseElementwiseKind(instruction.kind)) return null;
        if (instruction.inputs.len != 2 or instruction.outputs.len != 1) return null;

        const output_id = instruction.outputs[0];
        if (output_id.index >= plan.values.len) return null;

        const descriptor = plan.values[output_id.index].descriptor;
        if (descriptor.element_type != .f32 or descriptor.layout != .dense_row_major) return null;
        for (instruction.inputs) |input_id| {
            if (input_id.index >= plan.values.len) return null;
            const input_descriptor = plan.values[input_id.index].descriptor;
            if (input_descriptor.element_type != .f32 or input_descriptor.layout != .dense_row_major) return null;
            if (!sameDims(input_descriptor.dims, descriptor.dims)) return null;
        }

        const element_count = denseElementCount(descriptor);
        if (element_count == 0) return null;

        return .{
            .instruction_index = instruction_index,
            .kind = instruction.kind,
            .input_ids = instruction.inputs,
            .output_id = output_id,
            .descriptor = descriptor,
            .element_count = element_count,
        };
    }

    fn writeSymbol(self: DenseElementwiseKernel, writer: *std.Io.Writer) !void {
        try writer.print("pjrtx_i{d}_{s}_f32_dense", .{ self.instruction_index, @tagName(self.kind) });
    }

    fn writeMsl(self: DenseElementwiseKernel, writer: *std.Io.Writer) !void {
        try writer.writeAll("kernel void ");
        try self.writeSymbol(writer);
        try writer.print(
            \\(
            \\    device const float* lhs [[buffer(0)]],
            \\    device const float* rhs [[buffer(1)]],
            \\    device float* out [[buffer(2)]],
            \\    constant PjrtxDenseElementwiseShape& shape [[buffer(3)]],
            \\    uint gid [[thread_position_in_grid]]) {{
            \\    if (gid >= shape.element_count) return;
            \\    out[gid] = pjrtx_dense_elementwise_{s}(lhs[gid], rhs[gid]);
            \\}}
            \\
            \\
        , .{@tagName(self.kind)});
    }

    fn writeManifest(self: DenseElementwiseKernel, writer: *std.Io.Writer, stem: []const u8, plan: *const ir.ExecutablePlan) !void {
        try writer.writeAll("  kernel instruction=");
        try writer.print("{d} label=dense_elementwise runner=metalcpp_dense_elementwise_f32 symbol=", .{self.instruction_index});
        try self.writeSymbol(writer);
        try writer.print(" source={s}.metal op={s} dtype=f32 element_count={d} shape=", .{ stem, @tagName(self.kind), self.element_count });
        try text.writeShape(writer, self.descriptor.dims);
        try text.writeValueRefList(writer, plan, " inputs", self.input_ids);
        try writer.writeAll(" outputs=[");
        try text.writeValueRef(writer, plan, self.output_id);
        try writer.writeAll("] buffers=lhs:0,rhs:1,out:2,shape:3\n");
    }
};

/// Writes experimental Metal source artifacts for one backend program when requested.
pub fn dumpIfEnabled(allocator: std.mem.Allocator, plan: *const ir.ExecutablePlan, program: *const program_mod.Program) !void {
    const dir = profiling.metalCppMslDir() orelse return;
    const io = profiling.backendIo();
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
    try writeMsl(&metal_source.writer, plan, program);
    try writeFile(io, metal_path, metal_source.written());

    var metalcpp_source = std.Io.Writer.Allocating.init(allocator);
    defer metalcpp_source.deinit();
    try writeMetalCppHost(&metalcpp_source.writer, stem, plan, program);
    try writeFile(io, metalcpp_path, metalcpp_source.written());

    var manifest = std.Io.Writer.Allocating.init(allocator);
    defer manifest.deinit();
    try writeManifest(&manifest.writer, stem, plan, program);
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

pub fn writeMsl(writer: *std.Io.Writer, plan: *const ir.ExecutablePlan, program: *const program_mod.Program) !void {
    try writer.writeAll(
        \\#include <metal_stdlib>
        \\using namespace metal;
        \\
        \\// Generated by PjRTx from compiler/runtime executable IR.
        \\// This file is an experimental metal-cpp backend artifact. It is not
        \\// used by the default MLX execution path unless a future env-selected
        \\// backend runner consumes it.
        \\
    );
    try writer.print("// values={d} instructions={d} schedule_items={d} nodes={d} fusion_groups={d}\n\n", .{
        plan.values.len,
        plan.instructions.len,
        program.schedule.len,
        program.nodes.len,
        program.fusion_groups.len,
    });

    if (hasDenseElementwiseKernels(plan)) {
        try writeDenseElementwiseSourceHelper(writer);
    }

    for (plan.instructions, 0..) |instruction, instruction_index| {
        try writer.print("// instruction {d}: {s}", .{ instruction_index, @tagName(instruction.kind) });
        try text.writeValueRefList(writer, plan, " inputs", instruction.inputs);
        try text.writeValueRefList(writer, plan, " outputs", instruction.outputs);
        try text.writeInstructionAttributes(writer, instruction);
        try writer.writeByte('\n');

        if (DenseElementwiseKernel.init(plan, instruction, instruction_index)) |kernel| {
            try kernel.writeMsl(writer);
        } else switch (instruction.kind) {
            .dot_general => try writeDotKernelSkeleton(writer, instruction_index, instruction),
            else => {},
        }
    }
}

pub fn writeManifest(writer: *std.Io.Writer, stem: []const u8, plan: *const ir.ExecutablePlan, program: *const program_mod.Program) !void {
    try writer.print(
        \\backend_program_manifest version=1 artifact={s}
        \\counts values={d} instructions={d} schedule_items={d} nodes={d} edges={d} fusion_groups={d} materialization_boundaries={d} subprograms={d} control_flows={d}
        \\topology replicas={d} partitions={d} assigned_devices={d}
        \\
        \\values:
        \\
    , .{
        stem,
        plan.values.len,
        plan.instructions.len,
        program.schedule.len,
        program.nodes.len,
        program.edges.len,
        program.fusion_groups.len,
        program.materialization_boundaries.len,
        program.subprograms.len,
        program.control_flows.len,
        plan.options.num_replicas,
        plan.options.num_partitions,
        plan.options.device_assignment.len,
    });

    for (plan.values, 0..) |value, value_index| {
        try writer.print("  v{d} role={s} storage={s} ", .{ value_index, @tagName(value.role), @tagName(value.storage) });
        try text.writeDescriptor(writer, value.descriptor);
        if (value.elements.len != 0) {
            try text.writeValueRefList(writer, plan, " elements", value.elements);
        }
        if (value_index < program.values.len) {
            const program_value = program.values[value_index];
            try writer.print(" bytes={d}", .{program_value.byte_size});
            try text.writeOptionalUsize(writer, " producer_node", program_value.producer_node);
            try text.writeOptionalUsize(writer, " last_use_node", program_value.last_use_node);
            try writer.print(" output={d}", .{@intFromBool(program_value.is_output)});
            try text.writeOptionalUsize(writer, " materialization_boundary", program_value.materialization_boundary);
        }
        try writer.writeByte('\n');
    }

    try writer.writeAll("\ninstructions:\n");
    for (plan.instructions, 0..) |instruction, instruction_index| {
        try writer.print("  i{d} op={s}", .{ instruction_index, @tagName(instruction.kind) });
        try text.writeValueRefList(writer, plan, " inputs", instruction.inputs);
        try text.writeValueRefList(writer, plan, " outputs", instruction.outputs);
        try text.writeInstructionAttributes(writer, instruction);
        if (DenseElementwiseKernel.init(plan, instruction, instruction_index)) |kernel| {
            try writer.writeAll(" manifest_label=dense_elementwise runner=metalcpp_dense_elementwise_f32 kernel=");
            try kernel.writeSymbol(writer);
            try writer.print(" element_count={d}", .{kernel.element_count});
        }
        try writer.writeByte('\n');
    }

    try writer.writeAll("\nmetalcpp_kernels:\n");
    for (plan.instructions, 0..) |instruction, instruction_index| {
        const kernel = DenseElementwiseKernel.init(plan, instruction, instruction_index) orelse continue;
        try kernel.writeManifest(writer, stem, plan);
    }

    try writer.writeAll("\nnodes:\n");
    for (program.nodes, 0..) |node, node_index| {
        try writer.print("  n{d} instruction={d}", .{ node_index, node.instruction_index });
        if (node.instruction_index < plan.instructions.len) {
            try writer.print(" op={s}", .{@tagName(plan.instructions[node.instruction_index].kind)});
        }
        try writer.print(" kind={s} materializes={d}", .{ @tagName(node.kind), @intFromBool(node.materializes) });
        try text.writeOptionalUsize(writer, " fusion_group", node.fusion_group);
        try text.writeValueRefList(writer, plan, " inputs", node.inputs);
        try text.writeValueRefList(writer, plan, " outputs", node.outputs);
        try text.writeUsizeList(writer, " subprograms", node.subprograms);
        try text.writeOptionalUsize(writer, " control_flow", node.control_flow);
        try writer.writeByte('\n');
    }

    try writer.writeAll("\nschedule:\n");
    for (program.schedule, 0..) |item, schedule_index| {
        try writer.print("  s{d} kind={s} index={d} count={d}", .{ schedule_index, @tagName(item.kind), item.index, item.count });
        switch (item.kind) {
            .node => if (item.index < program.nodes.len) {
                const node = program.nodes[item.index];
                try writer.print(" node_kind={s} instruction={d}", .{ @tagName(node.kind), node.instruction_index });
                if (node.instruction_index < plan.instructions.len) {
                    try writer.print(" op={s}", .{@tagName(plan.instructions[node.instruction_index].kind)});
                }
            },
            .fusion_group => if (item.index < program.fusion_groups.len) {
                const group = program.fusion_groups[item.index];
                try writer.print(" group_kind={s} first_node={d} last_node={d} node_count={d}", .{ @tagName(group.kind), group.first_node, group.last_node, group.node_count });
                try text.writeUsizeList(writer, " nodes", group.node_indices);
                try text.writeValueRefList(writer, plan, " inputs", group.input_values);
                try text.writeValueRefList(writer, plan, " outputs", group.output_values);
                try text.writeFusionGroupOps(writer, plan, program, group);
            },
            .materialization_boundary => if (item.index < program.materialization_boundaries.len) {
                const boundary = program.materialization_boundaries[item.index];
                try writer.print(" reason={s} value=", .{@tagName(boundary.reason)});
                try text.writeValueRef(writer, plan, boundary.value_id);
            },
        }
        try writer.writeByte('\n');
    }

    try writer.writeAll("\nmaterialization_boundaries:\n");
    for (program.materialization_boundaries, 0..) |boundary, boundary_index| {
        try writer.print("  m{d} reason={s} value=", .{ boundary_index, @tagName(boundary.reason) });
        try text.writeValueRef(writer, plan, boundary.value_id);
        try writer.writeByte('\n');
    }

    try writer.writeAll("\nsubprograms:\n");
    for (program.subprograms, 0..) |subprogram, subprogram_index| {
        try writer.print("  sub{d} kind={s} parent_node={d} region={d} values={d} instructions={d} returns={d}", .{
            subprogram_index,
            @tagName(subprogram.kind),
            subprogram.parent_node,
            subprogram.region_id.index,
            subprogram.values.len,
            subprogram.instructions.len,
            subprogram.return_descriptors.len,
        });
        try writer.writeByte('\n');
    }
}

fn hasDenseElementwiseKernels(plan: *const ir.ExecutablePlan) bool {
    for (plan.instructions, 0..) |instruction, instruction_index| {
        if (DenseElementwiseKernel.init(plan, instruction, instruction_index) != null) return true;
    }
    return false;
}

fn isDenseElementwiseKind(kind: ir.PlanInstructionKind) bool {
    return switch (kind) {
        .add, .subtract, .multiply, .divide, .maximum, .minimum => true,
        else => false,
    };
}

fn denseElementCount(descriptor: ir.BufferDescriptor) usize {
    const element_size = descriptor.element_type.byteSize();
    if (element_size == 0) return 0;
    return ir.denseByteSize(descriptor.element_type, descriptor.dims) / element_size;
}

fn sameDims(lhs: []const i64, rhs: []const i64) bool {
    if (lhs.len != rhs.len) return false;
    for (lhs, rhs) |lhs_dim, rhs_dim| {
        if (lhs_dim != rhs_dim) return false;
    }
    return true;
}

fn writeDenseElementwiseSourceHelper(writer: *std.Io.Writer) !void {
    try writer.writeAll(
        \\struct PjrtxDenseElementwiseShape {
        \\    uint element_count;
        \\};
        \\
        \\static inline float pjrtx_dense_elementwise_add(float lhs, float rhs) {
        \\    return lhs + rhs;
        \\}
        \\
        \\static inline float pjrtx_dense_elementwise_subtract(float lhs, float rhs) {
        \\    return lhs - rhs;
        \\}
        \\
        \\static inline float pjrtx_dense_elementwise_multiply(float lhs, float rhs) {
        \\    return lhs * rhs;
        \\}
        \\
        \\static inline float pjrtx_dense_elementwise_divide(float lhs, float rhs) {
        \\    return lhs / rhs;
        \\}
        \\
        \\static inline float pjrtx_dense_elementwise_maximum(float lhs, float rhs) {
        \\    return max(lhs, rhs);
        \\}
        \\
        \\static inline float pjrtx_dense_elementwise_minimum(float lhs, float rhs) {
        \\    return min(lhs, rhs);
        \\}
        \\
    );
}

fn writeDotKernelSkeleton(writer: *std.Io.Writer, instruction_index: usize, instruction: ir.PlanInstruction) !void {
    try writer.print(
        \\// dot_general lowering candidate {d}
        \\// lhs_contract=
    , .{instruction_index});
    try text.writeOptionalDims(writer, instruction.lhs_contracting_dimensions);
    try writer.writeAll(" rhs_contract=");
    try text.writeOptionalDims(writer, instruction.rhs_contracting_dimensions);
    try writer.writeByte('\n');
    try writer.print(
        \\kernel void pjrtx_i{d}_dot_general_placeholder(
        \\    device const float* lhs [[buffer(0)]],
        \\    device const float* rhs [[buffer(1)]],
        \\    device float* out [[buffer(2)]],
        \\    uint gid [[thread_position_in_grid]]) {{
        \\    // TODO: specialize tile sizes and memory layout from PjRTx IR.
        \\    out[gid] = 0.0f;
        \\}}
        \\
    , .{instruction_index});
}

fn writeMetalCppHost(writer: *std.Io.Writer, stem: []const u8, plan: *const ir.ExecutablePlan, program: *const program_mod.Program) !void {
    try writer.print(
        \\// Generated by PjRTx from compiler/runtime executable IR.
        \\// This is an experimental metal-cpp host scaffold for {s}.metal.
        \\#include <Metal/Metal.hpp>
        \\
        \\namespace pjrtx::metalcpp_generated {{
        \\
        \\struct ProgramShape {{
        \\  static constexpr unsigned values = {d};
        \\  static constexpr unsigned instructions = {d};
        \\  static constexpr unsigned schedule_items = {d};
        \\  static constexpr unsigned nodes = {d};
        \\  static constexpr unsigned fusion_groups = {d};
        \\}};
        \\
        \\}}  // namespace pjrtx::metalcpp_generated
        \\
    , .{
        stem,
        plan.values.len,
        plan.instructions.len,
        program.schedule.len,
        program.nodes.len,
        program.fusion_groups.len,
    });
}
