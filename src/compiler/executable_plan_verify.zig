const std = @import("std");
const ir = @import("src/compiler/ir");
const model = @import("compiler_model.zig");
const plan_instruction_mod = @import("plan_instruction.zig");
const plan_build = @import("executable_plan_build.zig");

const ValueId = model.ValueId;
const PlanInstruction = model.PlanInstruction;
const PlanInstructionKind = model.PlanInstructionKind;
const ExecutablePlan = model.ExecutablePlan;
const isUnaryKind = plan_instruction_mod.isUnaryKind;
const isBinaryKind = plan_instruction_mod.isBinaryKind;
const isReduceKind = plan_build.isReduceKind;
const isReduceWindowKind = plan_build.isReduceWindowKind;

/// Errors emitted while checking executable-plan invariants before runtime use.
pub const VerifyError = std.Io.Writer.Error || error{ InvalidExecutablePlan, OutOfMemory };

pub fn expectedInputCount(kind: PlanInstructionKind) ?usize {
    return switch (kind) {
        .constant, .iota, .partition_id => 0,
        .dynamic_slice, .dynamic_update_slice, .concatenate, .custom_call, .optimization_barrier, .rng_bit_generator, .scatter, .sort, .tuple, .while_ => null,
        .select, .clamp => 3,
        .reduce_sum, .reduce_max, .reduce_min, .reduce_and, .reduce_or => null,
        .rng => null,
        .unsupported => null,
        else => if (isUnaryKind(kind)) 1 else if (isBinaryKind(kind)) 2 else null,
    };
}

pub fn failPlanVerification(
    writer: *std.Io.Writer,
    pass_name: []const u8,
    instruction_index: ?usize,
    value_id: ?ValueId,
    detail: []const u8,
    feature: []const u8,
) VerifyError {
    try writer.print("invalid executable plan: pass={s}", .{pass_name});
    if (instruction_index) |index| try writer.print(" instruction={d}", .{index});
    if (value_id) |id| try writer.print(" value={d}", .{id.index});
    try writer.print(" detail=\"{s}\" feature={s}", .{ detail, feature });
    return error.InvalidExecutablePlan;
}

pub fn valueInPlan(plan: ExecutablePlan, id: ValueId) bool {
    const index: usize = id.index;
    return index < plan.values.len and plan.values[index].id.index == id.index;
}

pub fn descriptorKnown(descriptor: ir.BufferDescriptor) bool {
    return descriptor.element_type != .invalid;
}

pub fn sameShape(lhs: ir.BufferDescriptor, rhs: ir.BufferDescriptor) bool {
    return std.mem.eql(i64, lhs.dims, rhs.dims);
}

pub fn sameTypeAndShape(lhs: ir.BufferDescriptor, rhs: ir.BufferDescriptor) bool {
    if (!descriptorKnown(lhs) or !descriptorKnown(rhs)) return true;
    return lhs.element_type == rhs.element_type and sameShape(lhs, rhs);
}

pub fn valueIsTensor(plan: ExecutablePlan, id: ValueId) bool {
    if (!valueInPlan(plan, id)) return false;
    return plan.values[id.index].storage == .tensor;
}

pub fn instructionOutputDims(instruction: PlanInstruction) ?[]const i64 {
    return switch (instruction.kind) {
        .constant, .copy_arg0, .unsupported => null,
        else => instruction.dims,
    };
}

pub fn verifyInstructionDescriptors(
    plan: ExecutablePlan,
    instruction: PlanInstruction,
    instruction_index: usize,
    writer: *std.Io.Writer,
) VerifyError!void {
    const pass_name = "pjrtx-plan-verify";
    if (instruction.outputs.len == 0) return;
    if (!valueIsTensor(plan, instruction.outputs[0])) return;
    const output = plan.values[instruction.outputs[0].index].descriptor;
    for (instruction.inputs) |input_id| {
        if (!valueIsTensor(plan, input_id)) continue;
        const input = plan.values[input_id.index].descriptor;
        if (!descriptorKnown(input) or !descriptorKnown(output)) continue;
        if (input.layout != .dense_row_major or output.layout != .dense_row_major) {
            return failPlanVerification(writer, pass_name, instruction_index, instruction.outputs[0], "bootstrap plan supports dense row-major layouts only", "layout");
        }
        for (input.dims) |dim| {
            if (dim < 0) return failPlanVerification(writer, pass_name, instruction_index, input_id, "input shape contains dynamic dimensions", "shape");
        }
        for (output.dims) |dim| {
            if (dim < 0) return failPlanVerification(writer, pass_name, instruction_index, instruction.outputs[0], "output shape contains dynamic dimensions", "shape");
        }
    }

    switch (instruction.kind) {
        .constant => {
            if (descriptorKnown(output) and instruction.literal == null) {
                return failPlanVerification(writer, pass_name, instruction_index, instruction.outputs[0], "constant instruction must carry literal bytes", "constant");
            }
        },
        .copy_arg0, .negate, .exp, .expm1, .tanh, .sqrt, .rsqrt, .cbrt, .ceil, .floor, .log, .log1p, .logistic, .sine, .cosine, .not_, .sign, .round_nearest_afz, .round_nearest_even, .popcnt, .count_leading_zeros, .reduce_precision => {
            if (!sameTypeAndShape(plan.values[instruction.inputs[0].index].descriptor, output)) {
                return failPlanVerification(writer, pass_name, instruction_index, instruction.outputs[0], "unary instruction must preserve input dtype and shape", "shape-type");
            }
        },
        .abs => {
            const input = plan.values[instruction.inputs[0].index].descriptor;
            if (descriptorKnown(input) and descriptorKnown(output) and !sameShape(input, output)) {
                return failPlanVerification(writer, pass_name, instruction_index, instruction.outputs[0], "abs output must preserve input shape", "shape-type");
            }
            const valid_abs_dtype = if (input.element_type == .c64)
                output.element_type == .f32
            else
                input.element_type == output.element_type;
            if (descriptorKnown(input) and descriptorKnown(output) and !valid_abs_dtype) {
                return failPlanVerification(writer, pass_name, instruction_index, instruction.outputs[0], "abs output dtype must match input except c64 inputs produce f32", "shape-type");
            }
        },
        .is_finite => {
            const input = plan.values[instruction.inputs[0].index].descriptor;
            if (descriptorKnown(input) and descriptorKnown(output) and !sameShape(input, output)) {
                return failPlanVerification(writer, pass_name, instruction_index, instruction.outputs[0], "is_finite output must preserve input shape", "shape-type");
            }
            if (descriptorKnown(output) and output.element_type != .pred) {
                return failPlanVerification(writer, pass_name, instruction_index, instruction.outputs[0], "is_finite output must be pred", "shape-type");
            }
        },
        .convert => {
            const input = plan.values[instruction.inputs[0].index].descriptor;
            if (descriptorKnown(input) and descriptorKnown(output) and !sameShape(input, output)) {
                return failPlanVerification(writer, pass_name, instruction_index, instruction.outputs[0], "convert output must preserve input shape", "shape-type");
            }
        },
        .bitcast_convert => {
            const input = plan.values[instruction.inputs[0].index].descriptor;
            if (descriptorKnown(input) and descriptorKnown(output) and ir.denseByteSize(input.element_type, input.dims) != ir.denseByteSize(output.element_type, output.dims)) {
                return failPlanVerification(writer, pass_name, instruction_index, instruction.outputs[0], "bitcast_convert must preserve dense byte size", "shape-type");
            }
        },
        .add, .subtract, .multiply, .divide, .maximum, .minimum, .power, .atan2, .remainder, .and_, .or_, .xor, .shift_left, .shift_right_arithmetic, .shift_right_logical => {
            const lhs = plan.values[instruction.inputs[0].index].descriptor;
            const rhs = plan.values[instruction.inputs[1].index].descriptor;
            if (!sameTypeAndShape(lhs, rhs)) {
                return failPlanVerification(writer, pass_name, instruction_index, instruction.inputs[1], "binary inputs must have matching dtype and shape", "shape-type");
            }
            if (!sameTypeAndShape(lhs, output)) {
                return failPlanVerification(writer, pass_name, instruction_index, instruction.outputs[0], "binary output must match input dtype and shape", "shape-type");
            }
        },
        .reshape => {
            const input = plan.values[instruction.inputs[0].index].descriptor;
            if (descriptorKnown(input) and descriptorKnown(output) and input.element_type != output.element_type) {
                return failPlanVerification(writer, pass_name, instruction_index, instruction.outputs[0], "reshape must preserve dtype", "shape-type");
            }
            if (descriptorKnown(input) and descriptorKnown(output) and ir.denseByteSize(input.element_type, input.dims) != ir.denseByteSize(output.element_type, output.dims)) {
                return failPlanVerification(writer, pass_name, instruction_index, instruction.outputs[0], "reshape must preserve dense byte size", "shape-type");
            }
        },
        .transpose, .broadcast_in_dim, .slice, .dynamic_slice, .dynamic_update_slice, .pad, .reverse, .gather => {
            const input = plan.values[instruction.inputs[0].index].descriptor;
            if (descriptorKnown(input) and descriptorKnown(output) and input.element_type != output.element_type) {
                return failPlanVerification(writer, pass_name, instruction_index, instruction.outputs[0], "view instruction must preserve dtype", "shape-type");
            }
        },
        .sort => {
            if (!((instruction.inputs.len == 1 and instruction.outputs.len == 1) or (instruction.inputs.len == 2 and instruction.outputs.len == 2))) {
                return failPlanVerification(writer, pass_name, instruction_index, instruction.outputs[0], "sort must have either one input/output or key-value two input/output form", "instruction-arity");
            }
            for (instruction.inputs, instruction.outputs) |input_id, output_id| {
                const input = plan.values[input_id.index].descriptor;
                const sort_output = plan.values[output_id.index].descriptor;
                if (descriptorKnown(input) and descriptorKnown(sort_output) and !sameTypeAndShape(input, sort_output)) {
                    return failPlanVerification(writer, pass_name, instruction_index, output_id, "sort output must preserve each input dtype and shape", "shape-type");
                }
            }
        },
        .top_k => {
            if (instruction.inputs.len != 1 or instruction.outputs.len != 2) {
                return failPlanVerification(writer, pass_name, instruction_index, instruction.outputs[0], "top_k must have one input and two outputs", "instruction-arity");
            }
            const input = plan.values[instruction.inputs[0].index].descriptor;
            const values = plan.values[instruction.outputs[0].index].descriptor;
            const indices = plan.values[instruction.outputs[1].index].descriptor;
            if (descriptorKnown(input) and descriptorKnown(values) and input.element_type != values.element_type) {
                return failPlanVerification(writer, pass_name, instruction_index, instruction.outputs[0], "top_k values output must preserve input dtype", "shape-type");
            }
            if (descriptorKnown(indices) and indices.element_type != .s32 and indices.element_type != .u32) {
                return failPlanVerification(writer, pass_name, instruction_index, instruction.outputs[1], "top_k indices output must be int32-like", "shape-type");
            }
        },
        .concatenate => {
            const lhs = plan.values[instruction.inputs[0].index].descriptor;
            const rhs = plan.values[instruction.inputs[1].index].descriptor;
            if (descriptorKnown(lhs) and descriptorKnown(rhs) and lhs.element_type != rhs.element_type) {
                return failPlanVerification(writer, pass_name, instruction_index, instruction.inputs[1], "concatenate inputs must have matching dtype", "shape-type");
            }
            if (descriptorKnown(lhs) and descriptorKnown(output) and lhs.element_type != output.element_type) {
                return failPlanVerification(writer, pass_name, instruction_index, instruction.outputs[0], "concatenate output must match input dtype", "shape-type");
            }
        },
        .dot_general => {
            const lhs = plan.values[instruction.inputs[0].index].descriptor;
            const rhs = plan.values[instruction.inputs[1].index].descriptor;
            const valid_float_dot = lhs.element_type == rhs.element_type and
                lhs.element_type == output.element_type and
                (lhs.element_type == .f16 or lhs.element_type == .bf16 or lhs.element_type == .f32);
            if (descriptorKnown(lhs) and descriptorKnown(rhs) and descriptorKnown(output) and !valid_float_dot) {
                return failPlanVerification(writer, pass_name, instruction_index, instruction.outputs[0], "dot_general bootstrap lowering supports same-dtype f16, bf16, or f32 tensors only", "dot-general");
            }
        },
        .reduce_sum, .reduce_max, .reduce_min, .reduce_and, .reduce_or, .reduce_window_sum, .reduce_window_max => {
            if (instruction.kind == .reduce_max and instruction.inputs.len == 2 and instruction.outputs.len == 2) {
                const values = plan.values[instruction.inputs[0].index].descriptor;
                const indices = plan.values[instruction.inputs[1].index].descriptor;
                const values_output = plan.values[instruction.outputs[0].index].descriptor;
                const indices_output = plan.values[instruction.outputs[1].index].descriptor;
                if (descriptorKnown(values) and descriptorKnown(values_output) and values.element_type != values_output.element_type) {
                    return failPlanVerification(writer, pass_name, instruction_index, instruction.outputs[0], "reduce argmax values output must preserve dtype", "shape-type");
                }
                if (descriptorKnown(indices) and descriptorKnown(indices_output) and indices.element_type != indices_output.element_type) {
                    return failPlanVerification(writer, pass_name, instruction_index, instruction.outputs[1], "reduce argmax indices output must preserve dtype", "shape-type");
                }
                if (descriptorKnown(values) and descriptorKnown(indices) and !sameShape(values, indices)) {
                    return failPlanVerification(writer, pass_name, instruction_index, instruction.inputs[1], "reduce argmax value and index inputs must have matching shape", "shape-type");
                }
                if (descriptorKnown(values_output) and descriptorKnown(indices_output) and !sameShape(values_output, indices_output)) {
                    return failPlanVerification(writer, pass_name, instruction_index, instruction.outputs[1], "reduce argmax value and index outputs must have matching shape", "shape-type");
                }
                return;
            }
            const input = plan.values[instruction.inputs[0].index].descriptor;
            if (descriptorKnown(input) and descriptorKnown(output) and input.element_type != output.element_type) {
                return failPlanVerification(writer, pass_name, instruction_index, instruction.outputs[0], "reduce output must preserve dtype", "shape-type");
            }
        },
        .compare => {
            const lhs = plan.values[instruction.inputs[0].index].descriptor;
            const rhs = plan.values[instruction.inputs[1].index].descriptor;
            if (!sameTypeAndShape(lhs, rhs)) {
                return failPlanVerification(writer, pass_name, instruction_index, instruction.inputs[1], "compare inputs must have matching dtype and shape", "shape-type");
            }
            if (descriptorKnown(output) and output.element_type != .pred) {
                return failPlanVerification(writer, pass_name, instruction_index, instruction.outputs[0], "compare output must be pred", "shape-type");
            }
        },
        .select => {
            const pred = plan.values[instruction.inputs[0].index].descriptor;
            const on_true = plan.values[instruction.inputs[1].index].descriptor;
            const on_false = plan.values[instruction.inputs[2].index].descriptor;
            if (descriptorKnown(pred) and pred.element_type != .pred) {
                return failPlanVerification(writer, pass_name, instruction_index, instruction.inputs[0], "select predicate must be pred", "shape-type");
            }
            if (!sameTypeAndShape(on_true, on_false) or !sameTypeAndShape(on_true, output)) {
                return failPlanVerification(writer, pass_name, instruction_index, instruction.outputs[0], "select data inputs and output must match", "shape-type");
            }
        },
        .clamp => {
            const min = plan.values[instruction.inputs[0].index].descriptor;
            const value = plan.values[instruction.inputs[1].index].descriptor;
            const max = plan.values[instruction.inputs[2].index].descriptor;
            if (descriptorKnown(min) and descriptorKnown(value) and min.element_type != value.element_type) {
                return failPlanVerification(writer, pass_name, instruction_index, instruction.inputs[0], "clamp min and operand dtypes must match", "shape-type");
            }
            if (descriptorKnown(max) and descriptorKnown(value) and max.element_type != value.element_type) {
                return failPlanVerification(writer, pass_name, instruction_index, instruction.inputs[2], "clamp max and operand dtypes must match", "shape-type");
            }
            if (!sameTypeAndShape(value, output)) {
                return failPlanVerification(writer, pass_name, instruction_index, instruction.outputs[0], "clamp output must match operand", "shape-type");
            }
        },
        .iota => {},
        .partition_id => {
            if (descriptorKnown(output) and output.dims.len != 0) {
                return failPlanVerification(writer, pass_name, instruction_index, instruction.outputs[0], "partition_id output must be scalar", "shape-type");
            }
            if (descriptorKnown(output) and output.element_type != .u32 and output.element_type != .s32) {
                return failPlanVerification(writer, pass_name, instruction_index, instruction.outputs[0], "partition_id output must be u32 or s32", "shape-type");
            }
        },
        .cholesky => {
            const input = plan.values[instruction.inputs[0].index].descriptor;
            if (descriptorKnown(input) and descriptorKnown(output) and !sameTypeAndShape(input, output)) {
                return failPlanVerification(writer, pass_name, instruction_index, instruction.outputs[0], "cholesky output must match input dtype and shape", "shape-type");
            }
            if (descriptorKnown(output) and output.element_type != .f32) {
                return failPlanVerification(writer, pass_name, instruction_index, instruction.outputs[0], "cholesky bootstrap lowering supports f32 tensors only", "cholesky");
            }
            if (descriptorKnown(output) and output.dims.len < 2) {
                return failPlanVerification(writer, pass_name, instruction_index, instruction.outputs[0], "cholesky requires rank >= 2", "shape");
            }
        },
        .triangular_solve => {
            const rhs = plan.values[instruction.inputs[1].index].descriptor;
            if (descriptorKnown(rhs) and descriptorKnown(output) and !sameTypeAndShape(rhs, output)) {
                return failPlanVerification(writer, pass_name, instruction_index, instruction.outputs[0], "triangular_solve output must match rhs dtype and shape", "shape-type");
            }
        },
        .rng => {
            if (instruction.inputs.len != 2 or instruction.outputs.len != 1) {
                return failPlanVerification(writer, pass_name, instruction_index, null, "rng must consume low/scale operands and produce one tensor", "random");
            }
            if (instruction.rng_distribution == null) {
                return failPlanVerification(writer, pass_name, instruction_index, instruction.outputs[0], "rng lowering requires rng_distribution metadata", "random");
            }
        },
        .rng_bit_generator => {
            if (instruction.outputs.len != 2) {
                return failPlanVerification(writer, pass_name, instruction_index, null, "rng_bit_generator must produce state and random bits", "random");
            }
        },
        .complex, .real, .imag, .fft, .convolution, .scatter, .custom_call, .optimization_barrier, .get_tuple_element, .tuple, .while_ => {},
        .unsupported => {},
    }

    if (instructionOutputDims(instruction)) |dims| {
        if (descriptorKnown(output) and !std.mem.eql(i64, output.dims, dims)) {
            return failPlanVerification(writer, pass_name, instruction_index, instruction.outputs[0], "instruction shape metadata must match output value descriptor", "shape");
        }
    }
}

pub fn verifyExecutablePlan(
    allocator: std.mem.Allocator,
    plan: ExecutablePlan,
    writer: *std.Io.Writer,
) VerifyError!void {
    const pass_name = "pjrtx-plan-verify";
    if (plan.options.num_replicas < 1 or plan.options.num_partitions < 1) {
        return failPlanVerification(writer, pass_name, null, null, "replicas and partitions must be positive", "topology");
    }
    if (plan.options.device_assignment.len < plan.options.numDevices()) {
        return failPlanVerification(writer, pass_name, null, null, "device assignment is smaller than replicas * partitions", "topology");
    }
    if (plan.values.len == 0) {
        return failPlanVerification(writer, pass_name, null, null, "plan must define at least one value", "value-graph");
    }

    var defined = try allocator.alloc(bool, plan.values.len);
    defer allocator.free(defined);
    @memset(defined, false);

    var parameter_count: usize = 0;
    for (plan.values, 0..) |value, index| {
        if (value.id.index != index) {
            return failPlanVerification(writer, pass_name, null, value.id, "value id must match value table index", "value-graph");
        }
        switch (value.role) {
            .parameter => {
                parameter_count += 1;
                defined[index] = true;
            },
            .constant => {},
            .instruction_result, .output => {},
        }
        if (value.storage == .tensor and value.elements.len != 0) {
            return failPlanVerification(writer, pass_name, null, value.id, "tensor values must not carry element references", "value-storage");
        }
        const concrete_complex_pair = value.storage == .complex_pair and
            (value.descriptor.element_type == .c64 or value.descriptor.element_type == .c128);
        if (value.storage != .tensor and !concrete_complex_pair and descriptorKnown(value.descriptor)) {
            return failPlanVerification(writer, pass_name, null, value.id, "structured values must not claim a concrete tensor dtype", "value-storage");
        }
        for (value.elements) |element_id| {
            if (!valueInPlan(plan, element_id)) {
                return failPlanVerification(writer, pass_name, null, value.id, "structured value element references an unknown value", "value-storage");
            }
        }
    }

    if (plan.parameter_shardings.len != parameter_count) {
        try writer.print("invalid executable plan: pass={s} detail=\"parameter sharding count must match parameter value count: shardings={d} parameters={d}\" feature=sharding", .{ pass_name, plan.parameter_shardings.len, parameter_count });
        return error.InvalidExecutablePlan;
    }
    for (plan.donated_parameter_indices) |parameter_index| {
        if (parameter_index >= parameter_count) {
            return failPlanVerification(writer, pass_name, null, null, "donated parameter index is outside parameter count", "donation-alias");
        }
    }
    for (plan.output_aliases) |alias| {
        if (alias.output_index >= plan.output_ids.len) {
            return failPlanVerification(writer, pass_name, null, null, "output alias references an unknown output", "donation-alias");
        }
        if (alias.parameter_index >= parameter_count) {
            return failPlanVerification(writer, pass_name, null, null, "output alias references an unknown parameter", "donation-alias");
        }
        var donates = false;
        for (plan.donated_parameter_indices) |parameter_index| {
            if (parameter_index == alias.parameter_index) {
                donates = true;
                break;
            }
        }
        if (!donates) {
            return failPlanVerification(writer, pass_name, null, null, "output alias parameter must be marked donated", "donation-alias");
        }
    }
    for (plan.instructions, 0..) |instruction, instruction_index| {
        if (instruction.kind == .unsupported) {
            return failPlanVerification(writer, pass_name, instruction_index, null, "unsupported instruction cannot enter executable plan", "instruction-kind");
        }
        if (expectedInputCount(instruction.kind)) |expected_inputs| if (instruction.inputs.len != expected_inputs) {
            return failPlanVerification(writer, pass_name, instruction_index, null, "instruction input arity mismatch", "instruction-arity");
        };
        if (instruction.outputs.len == 0) {
            return failPlanVerification(writer, pass_name, instruction_index, null, "instructions must produce at least one value", "instruction-arity");
        }
        for (instruction.region_ids) |region_id| {
            if (region_id.index >= plan.regions.len or plan.regions[region_id.index].id.index != region_id.index) {
                return failPlanVerification(writer, pass_name, instruction_index, null, "instruction region id references an unknown region", "region-graph");
            }
        }
        for (instruction.inputs) |input| {
            if (!valueInPlan(plan, input)) {
                return failPlanVerification(writer, pass_name, instruction_index, input, "instruction input references an unknown value", "value-graph");
            }
            if (!defined[input.index]) {
                return failPlanVerification(writer, pass_name, instruction_index, input, "instruction input must be defined before use", "value-graph");
            }
        }
        for (instruction.outputs) |output| {
            if (!valueInPlan(plan, output)) {
                return failPlanVerification(writer, pass_name, instruction_index, output, "instruction output references an unknown value", "value-graph");
            }
            if (defined[output.index]) {
                return failPlanVerification(writer, pass_name, instruction_index, output, "instruction output value is already defined", "value-graph");
            }
            const role = plan.values[output.index].role;
            if (role != .instruction_result and role != .output and role != .constant) {
                return failPlanVerification(writer, pass_name, instruction_index, output, "instruction output must target an instruction-result or output value", "value-role");
            }
            defined[output.index] = true;
        }
        try verifyInstructionDescriptors(plan, instruction, instruction_index, writer);
    }
    if (plan.output_ids.len != plan.output_shardings.len) {
        return failPlanVerification(writer, pass_name, null, null, "output id count must match output sharding count", "value-graph");
    }
    for (plan.output_ids) |output_id| {
        if (!valueInPlan(plan, output_id) or !defined[output_id.index]) {
            return failPlanVerification(writer, pass_name, null, output_id, "plan output must reference a defined value", "value-graph");
        }
    }
}
