const std = @import("std");
const ir = @import("src/compiler/ir");
const model = @import("compiler_model.zig");
const Operation = model.Operation;
const Value = model.Value;
const ValueId = model.ValueId;
const ValueRole = model.ValueRole;
const PlanInstructionKind = model.PlanInstructionKind;

pub fn instructionKindFromStablehlo(name: []const u8) PlanInstructionKind {
    if (std.mem.eql(u8, name, "constant")) return .constant;
    if (std.mem.eql(u8, name, "add")) return .add;
    if (std.mem.eql(u8, name, "subtract")) return .subtract;
    if (std.mem.eql(u8, name, "multiply")) return .multiply;
    if (std.mem.eql(u8, name, "divide")) return .divide;
    if (std.mem.eql(u8, name, "maximum")) return .maximum;
    if (std.mem.eql(u8, name, "minimum")) return .minimum;
    if (std.mem.eql(u8, name, "power")) return .power;
    if (std.mem.eql(u8, name, "atan2")) return .atan2;
    if (std.mem.eql(u8, name, "remainder")) return .remainder;
    if (std.mem.eql(u8, name, "and")) return .and_;
    if (std.mem.eql(u8, name, "or")) return .or_;
    if (std.mem.eql(u8, name, "xor")) return .xor;
    if (std.mem.eql(u8, name, "shift_left")) return .shift_left;
    if (std.mem.eql(u8, name, "shift_right_arithmetic")) return .shift_right_arithmetic;
    if (std.mem.eql(u8, name, "shift_right_logical")) return .shift_right_logical;
    if (std.mem.eql(u8, name, "negate")) return .negate;
    if (std.mem.eql(u8, name, "exponential")) return .exp;
    if (std.mem.eql(u8, name, "exponential_minus_one")) return .expm1;
    if (std.mem.eql(u8, name, "tanh")) return .tanh;
    if (std.mem.eql(u8, name, "sqrt")) return .sqrt;
    if (std.mem.eql(u8, name, "rsqrt")) return .rsqrt;
    if (std.mem.eql(u8, name, "abs")) return .abs;
    if (std.mem.eql(u8, name, "cbrt")) return .cbrt;
    if (std.mem.eql(u8, name, "ceil")) return .ceil;
    if (std.mem.eql(u8, name, "floor")) return .floor;
    if (std.mem.eql(u8, name, "log")) return .log;
    if (std.mem.eql(u8, name, "log_plus_one")) return .log1p;
    if (std.mem.eql(u8, name, "logistic")) return .logistic;
    if (std.mem.eql(u8, name, "sine")) return .sine;
    if (std.mem.eql(u8, name, "cosine")) return .cosine;
    if (std.mem.eql(u8, name, "not")) return .not_;
    if (std.mem.eql(u8, name, "sign")) return .sign;
    if (std.mem.eql(u8, name, "is_finite")) return .is_finite;
    if (std.mem.eql(u8, name, "round_nearest_afz")) return .round_nearest_afz;
    if (std.mem.eql(u8, name, "round_nearest_even")) return .round_nearest_even;
    if (std.mem.eql(u8, name, "popcnt")) return .popcnt;
    if (std.mem.eql(u8, name, "count_leading_zeros")) return .count_leading_zeros;
    if (std.mem.eql(u8, name, "convert")) return .convert;
    if (std.mem.eql(u8, name, "bitcast_convert")) return .bitcast_convert;
    if (std.mem.eql(u8, name, "reshape")) return .reshape;
    if (std.mem.eql(u8, name, "transpose")) return .transpose;
    if (std.mem.eql(u8, name, "broadcast_in_dim")) return .broadcast_in_dim;
    if (std.mem.eql(u8, name, "slice")) return .slice;
    if (std.mem.eql(u8, name, "dynamic_slice")) return .dynamic_slice;
    if (std.mem.eql(u8, name, "dynamic_update_slice")) return .dynamic_update_slice;
    if (std.mem.eql(u8, name, "pad")) return .pad;
    if (std.mem.eql(u8, name, "reverse")) return .reverse;
    if (std.mem.eql(u8, name, "concatenate")) return .concatenate;
    if (std.mem.eql(u8, name, "iota")) return .iota;
    if (std.mem.eql(u8, name, "gather")) return .gather;
    if (std.mem.eql(u8, name, "sort")) return .sort;
    if (std.mem.eql(u8, name, "top_k")) return .top_k;
    if (std.mem.eql(u8, name, "dot_general")) return .dot_general;
    if (std.mem.eql(u8, name, "reduce_sum")) return .reduce_sum;
    if (std.mem.eql(u8, name, "reduce_max")) return .reduce_max;
    if (std.mem.eql(u8, name, "reduce_min")) return .reduce_min;
    if (std.mem.eql(u8, name, "reduce_and")) return .reduce_and;
    if (std.mem.eql(u8, name, "reduce_or")) return .reduce_or;
    if (std.mem.eql(u8, name, "reduce_window_sum")) return .reduce_window_sum;
    if (std.mem.eql(u8, name, "reduce_window_max")) return .reduce_window_max;
    if (std.mem.eql(u8, name, "compare")) return .compare;
    if (std.mem.eql(u8, name, "select")) return .select;
    if (std.mem.eql(u8, name, "clamp")) return .clamp;
    if (std.mem.eql(u8, name, "cholesky")) return .cholesky;
    if (std.mem.eql(u8, name, "complex")) return .complex;
    if (std.mem.eql(u8, name, "convolution")) return .convolution;
    if (std.mem.eql(u8, name, "custom_call")) return .custom_call;
    if (std.mem.eql(u8, name, "fft")) return .fft;
    if (std.mem.eql(u8, name, "get_tuple_element")) return .get_tuple_element;
    if (std.mem.eql(u8, name, "imag")) return .imag;
    if (std.mem.eql(u8, name, "optimization_barrier")) return .optimization_barrier;
    if (std.mem.eql(u8, name, "partition_id")) return .partition_id;
    if (std.mem.eql(u8, name, "real")) return .real;
    if (std.mem.eql(u8, name, "reduce_precision")) return .reduce_precision;
    if (std.mem.eql(u8, name, "rng")) return .rng;
    if (std.mem.eql(u8, name, "rng_bit_generator")) return .rng_bit_generator;
    if (std.mem.eql(u8, name, "scatter")) return .scatter;
    if (std.mem.eql(u8, name, "triangular_solve")) return .triangular_solve;
    if (std.mem.eql(u8, name, "tuple")) return .tuple;
    if (std.mem.eql(u8, name, "while")) return .while_;
    return .unsupported;
}

pub fn bufferTypeFromDtype(dtype: []const u8) ir.BufferType {
    if (std.mem.eql(u8, dtype, "pred")) return .pred;
    if (std.mem.eql(u8, dtype, "i8") or std.mem.eql(u8, dtype, "s8")) return .s8;
    if (std.mem.eql(u8, dtype, "i16") or std.mem.eql(u8, dtype, "s16")) return .s16;
    if (std.mem.eql(u8, dtype, "i32") or std.mem.eql(u8, dtype, "s32")) return .s32;
    if (std.mem.eql(u8, dtype, "i64") or std.mem.eql(u8, dtype, "s64")) return .s64;
    if (std.mem.eql(u8, dtype, "u8")) return .u8;
    if (std.mem.eql(u8, dtype, "u16")) return .u16;
    if (std.mem.eql(u8, dtype, "u32")) return .u32;
    if (std.mem.eql(u8, dtype, "u64")) return .u64;
    if (std.mem.eql(u8, dtype, "f16")) return .f16;
    if (std.mem.eql(u8, dtype, "f32")) return .f32;
    if (std.mem.eql(u8, dtype, "f64")) return .f64;
    if (std.mem.eql(u8, dtype, "bf16")) return .bf16;
    if (std.mem.eql(u8, dtype, "c64") or std.mem.eql(u8, dtype, "complex<f32>")) return .c64;
    if (std.mem.eql(u8, dtype, "c128") or std.mem.eql(u8, dtype, "complex<f64>")) return .c128;
    return .invalid;
}

pub fn makeDescriptor(dims: []const i64, element_type: ir.BufferType) ir.BufferDescriptor {
    return .{
        .element_type = element_type,
        .dims = dims,
        .device_id = -1,
        .memory_id = -1,
        .shard_index = 0,
    };
}

pub fn descriptorFromOperation(allocator: std.mem.Allocator, op: Operation) !ir.BufferDescriptor {
    return makeDescriptor(try allocator.dupe(i64, op.dims), bufferTypeFromDtype(op.dtype));
}


pub fn isUnaryKind(kind: PlanInstructionKind) bool {
    return switch (kind) {
        .copy_arg0,
        .negate,
        .exp,
        .expm1,
        .tanh,
        .sqrt,
        .rsqrt,
        .abs,
        .cbrt,
        .ceil,
        .floor,
        .log,
        .log1p,
        .logistic,
        .sine,
        .cosine,
        .not_,
        .sign,
        .is_finite,
        .round_nearest_afz,
        .round_nearest_even,
        .popcnt,
        .count_leading_zeros,
        .convert,
        .bitcast_convert,
        .reshape,
        .transpose,
        .broadcast_in_dim,
        .slice,
        .sort,
        .top_k,
        .reverse,
        .reduce_sum,
        .reduce_max,
        .reduce_min,
        .reduce_and,
        .reduce_or,
        .cholesky,
        .fft,
        .get_tuple_element,
        .imag,
        .real,
        .reduce_precision,
        => true,
        else => false,
    };
}

pub fn isBinaryKind(kind: PlanInstructionKind) bool {
    return switch (kind) {
        .add,
        .subtract,
        .multiply,
        .divide,
        .maximum,
        .minimum,
        .power,
        .atan2,
        .remainder,
        .and_,
        .or_,
        .xor,
        .shift_left,
        .shift_right_arithmetic,
        .shift_right_logical,
        .concatenate,
        .dot_general,
        .compare,
        .gather,
        .pad,
        .complex,
        .convolution,
        .triangular_solve,
        => true,
        else => false,
    };
}

pub fn makeValue(id: u32, role: ValueRole, descriptor: ir.BufferDescriptor) Value {
    return .{
        .id = .{ .index = id },
        .role = role,
        .descriptor = descriptor,
    };
}

pub fn makeStructuredValue(
    id: u32,
    role: ValueRole,
    descriptor: ir.BufferDescriptor,
    storage: ir.ValueStorageKind,
    elements: []const ValueId,
) Value {
    return .{
        .id = .{ .index = id },
        .role = role,
        .descriptor = descriptor,
        .storage = storage,
        .elements = elements,
    };
}

pub fn makeUnknownDescriptor(allocator: std.mem.Allocator) !ir.BufferDescriptor {
    return makeDescriptor(try allocator.dupe(i64, &.{}), .invalid);
}

pub fn makeBootstrapValues(
    allocator: std.mem.Allocator,
    parameter_descriptors: []const ir.BufferDescriptor,
    ops: []const Operation,
    num_instruction_results: usize,
) ![]Value {
    const num_parameters = parameter_descriptors.len;
    const value_count = num_parameters + num_instruction_results;
    const values = try allocator.alloc(Value, value_count);
    var initialized: usize = 0;
    errdefer {
        for (values[0..initialized]) |value| {
            allocator.free(value.descriptor.dims);
            allocator.free(value.elements);
        }
        allocator.free(values);
    }
    for (values[0..num_parameters], 0..) |*value, index| {
        const descriptor = parameter_descriptors[index];
        value.* = makeValue(@intCast(index), .parameter, makeDescriptor(try allocator.dupe(i64, descriptor.dims), descriptor.element_type));
        initialized += 1;
    }
    for (values[num_parameters..], 0..) |*value, index| {
        const descriptor = if (index < ops.len) try descriptorFromOperation(allocator, ops[index]) else try makeUnknownDescriptor(allocator);
        value.* = makeValue(@intCast(num_parameters + index), .instruction_result, descriptor);
        initialized += 1;
    }
    return values;
}

