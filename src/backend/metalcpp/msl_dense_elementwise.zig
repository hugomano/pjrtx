const std = @import("std");

const ir = @import("src/compiler/ir");
const refs = @import("msl_artifact_refs.zig");

/// Describes one dense f32 elementwise kernel emitted into MetalCPP artifacts.
pub const Kernel = struct {
    instruction_index: usize,
    kind: ir.PlanInstructionKind,
    input_ids: []const ir.ValueId,
    output_id: ir.ValueId,
    descriptor: ir.BufferDescriptor,
    element_count: usize,

    /// Returns a kernel descriptor when the instruction has the supported dense form.
    pub fn init(plan: *const ir.ExecutablePlan, instruction: ir.PlanInstruction, instruction_index: usize) ?Kernel {
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

    /// Writes the stable symbol used by generated source and manifests.
    pub fn writeSymbol(self: Kernel, writer: *std.Io.Writer) !void {
        try writer.print("pjrtx_i{d}_{s}_f32_dense", .{ self.instruction_index, @tagName(self.kind) });
    }

    /// Writes one runnable dense elementwise Metal kernel.
    pub fn writeMsl(self: Kernel, writer: *std.Io.Writer) !void {
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

    /// Writes this kernel's manifest row.
    pub fn writeManifest(self: Kernel, writer: *std.Io.Writer, stem: []const u8, plan: *const ir.ExecutablePlan) !void {
        try writer.writeAll("  kernel instruction=");
        try writer.print("{d} label=dense_elementwise runner=metalcpp_dense_elementwise_f32 symbol=", .{self.instruction_index});
        try self.writeSymbol(writer);
        try writer.print(" source={s}.metal op={s} dtype=f32 element_count={d} shape=", .{ stem, @tagName(self.kind), self.element_count });
        try refs.writeShape(writer, self.descriptor.dims);
        try refs.writeValueRefList(writer, plan, " inputs", self.input_ids);
        try writer.writeAll(" outputs=[");
        try refs.writeValueRef(writer, plan, self.output_id);
        try writer.writeAll("] buffers=lhs:0,rhs:1,out:2,shape:3\n");
    }
};

/// Reports whether an executable plan has any dense f32 elementwise artifact kernels.
pub fn hasKernels(plan: *const ir.ExecutablePlan) bool {
    for (plan.instructions, 0..) |instruction, instruction_index| {
        if (Kernel.init(plan, instruction, instruction_index) != null) return true;
    }
    return false;
}

/// Writes shared helper declarations for dense f32 elementwise kernels.
pub fn writeSourceHelper(writer: *std.Io.Writer) !void {
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

