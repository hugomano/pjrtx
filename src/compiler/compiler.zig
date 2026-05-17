const std = @import("std");
const mlir = @import("c");
const core = @import("src/core");

var shardy_pass_registration_mutex: std.atomic.Mutex = .unlocked;
var shardy_passes_registered = false;
var transform_pass_registration_mutex: std.atomic.Mutex = .unlocked;
var transform_passes_registered = false;

pub const Partitioner = enum {
    shardy,
    gspmd,
};

pub const CompileOptions = core.CompileOptions;

pub const ProgramFormat = enum {
    stablehlo_text,
    stablehlo_bytecode,
    unknown,

    pub fn parse(text: []const u8) ProgramFormat {
        if (text.len == 0) return .stablehlo_text;
        if (std.mem.eql(u8, text, "mlir") or
            std.mem.eql(u8, text, "mlir_text") or
            std.mem.eql(u8, text, "stablehlo") or
            std.mem.eql(u8, text, "stablehlo_text"))
        {
            return .stablehlo_text;
        }
        if (std.mem.eql(u8, text, "stablehlo_bytecode") or std.mem.eql(u8, text, "mlir_bytecode")) {
            return .stablehlo_bytecode;
        }
        return .unknown;
    }
};

pub const Dialect = enum {
    func,
    stablehlo,
    sdy,
};

pub const Operation = struct {
    name: []const u8,
    line: usize,
    column: usize,
    dtype: []const u8 = "unknown",
    rank: ?usize = null,
    dims: []const i64 = &.{},
    permutation: []const i64 = &.{},
    broadcast_dimensions: []const i64 = &.{},
    start_indices: []const i64 = &.{},
    limit_indices: []const i64 = &.{},
    strides: []const i64 = &.{},
    dimension: ?i64 = null,
    sharding: []const u8 = "unspecified",

    fn deinit(self: Operation, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.dtype);
        allocator.free(self.dims);
        allocator.free(self.permutation);
        allocator.free(self.broadcast_dimensions);
        allocator.free(self.start_indices);
        allocator.free(self.limit_indices);
        allocator.free(self.strides);
        allocator.free(self.sharding);
    }
};

pub const ShardingKind = core.ShardingKind;
pub const ShardingMetadata = core.ShardingMetadata;

pub const ModuleAnalysis = struct {
    allocator: std.mem.Allocator,
    source: []u8,
    dialects: []Dialect,
    ops: []Operation,
    num_parameters: usize,
    num_outputs: usize,
    parameter_shardings: []ShardingMetadata,
    output_shardings: []ShardingMetadata,

    pub fn deinit(self: *ModuleAnalysis) void {
        for (self.output_shardings) |sharding| sharding.deinit(self.allocator);
        for (self.parameter_shardings) |sharding| sharding.deinit(self.allocator);
        for (self.ops) |op| op.deinit(self.allocator);
        self.allocator.free(self.output_shardings);
        self.allocator.free(self.parameter_shardings);
        self.allocator.free(self.ops);
        self.allocator.free(self.dialects);
        self.allocator.free(self.source);
    }
};

pub const AnalyzeError = error{
    UnsupportedProgramFormat,
    UnsupportedProgramEncoding,
    InvalidStablehloModule,
    InvalidManualComputation,
    GspmdNotEnabled,
    UnsupportedOp,
    UnsupportedSharding,
    OutOfMemory,
    ReadFailed,
    StreamTooLong,
    WriteFailed,
};

const MlirSession = struct {
    registry: mlir.MlirDialectRegistry,
    context: mlir.MlirContext,
    module: mlir.MlirModule,
    pass_manager: mlir.MlirPassManager,

    fn deinit(self: *MlirSession) void {
        if (!mlir.mlirPassManagerIsNull(self.pass_manager)) {
            mlir.mlirPassManagerDestroy(self.pass_manager);
        }
        if (!mlir.mlirModuleIsNull(self.module)) {
            mlir.mlirModuleDestroy(self.module);
        }
        if (!mlir.mlirContextIsNull(self.context)) {
            mlir.mlirContextDestroy(self.context);
        }
        if (!mlir.mlirDialectRegistryIsNull(self.registry)) {
            mlir.mlirDialectRegistryDestroy(self.registry);
        }
    }
};

const SourceLoc = struct {
    line: usize,
    column: usize,
};

fn mlirStringRef(text: []const u8) mlir.MlirStringRef {
    return mlir.mlirStringRefCreate(text.ptr, @intCast(text.len));
}

fn mlirStringSlice(text: mlir.MlirStringRef) []const u8 {
    return text.data[0..@intCast(text.length)];
}

fn operationName(op: mlir.MlirOperation) []const u8 {
    return mlirStringSlice(mlir.mlirIdentifierStr(mlir.mlirOperationGetName(op)));
}

fn namedAttributeName(named: mlir.MlirNamedAttribute) []const u8 {
    return mlirStringSlice(mlir.mlirIdentifierStr(named.name));
}

fn getOperationAttribute(op: mlir.MlirOperation, name: []const u8) mlir.MlirAttribute {
    return mlir.mlirOperationGetAttributeByName(op, mlirStringRef(name));
}

const MlirStringCallbackCtx = struct {
    writer: *std.Io.Writer,
    err: ?std.Io.Writer.Error = null,
};

fn writeMlirCallback(text: mlir.MlirStringRef, user_data: ?*anyopaque) callconv(.c) void {
    const ctx: *MlirStringCallbackCtx = @ptrCast(@alignCast(user_data.?));
    ctx.writer.writeAll(text.data[0..@intCast(text.length)]) catch |err| {
        ctx.err = err;
    };
}

fn loadDialect(context: mlir.MlirContext, handle: mlir.MlirDialectHandle) AnalyzeError!void {
    const dialect = mlir.mlirDialectHandleLoadDialect(handle, context);
    if (mlir.mlirDialectIsNull(dialect)) return error.InvalidStablehloModule;
}

fn insertDialect(registry: mlir.MlirDialectRegistry, handle: mlir.MlirDialectHandle) void {
    mlir.mlirDialectHandleInsertDialect(handle, registry);
}

fn registerShardyPassesOnce() void {
    while (!shardy_pass_registration_mutex.tryLock()) std.atomic.spinLoopHint();
    defer shardy_pass_registration_mutex.unlock();

    if (shardy_passes_registered) return;
    mlir.mlirRegisterAllSdyPassesAndPipelines();
    shardy_passes_registered = true;
}

fn registerTransformPassesOnce() void {
    while (!transform_pass_registration_mutex.tryLock()) std.atomic.spinLoopHint();
    defer transform_pass_registration_mutex.unlock();

    if (transform_passes_registered) return;
    mlir.mlirRegisterTransformsPasses();
    transform_passes_registered = true;
}

fn addPipeline(
    pass_manager: mlir.MlirPassManager,
    pipeline: []const u8,
    writer: *std.Io.Writer,
) AnalyzeError!void {
    const op_pass_manager = mlir.mlirPassManagerGetAsOpPassManager(pass_manager);
    var buffer: [4096]u8 = undefined;
    var scratch = std.Io.Writer.fixed(&buffer);
    var ctx: MlirStringCallbackCtx = .{ .writer = &scratch };
    const result = mlir.mlirOpPassManagerAddPipeline(
        op_pass_manager,
        mlirStringRef(pipeline),
        writeMlirCallback,
        &ctx,
    );
    if (ctx.err != null or !mlir.mlirLogicalResultIsSuccess(result)) {
        try writer.print("invalid StableHLO module: failed to construct MLIR pass pipeline {s}", .{pipeline});
        const detail = scratch.buffered();
        if (detail.len != 0) try writer.print(": {s}", .{detail});
        return error.InvalidStablehloModule;
    }
}

fn attributeUsesShardy(attr: mlir.MlirAttribute) bool {
    if (mlir.mlirAttributeIsNull(attr)) return false;
    if (mlir.mlirAttributeIsAArray(attr)) {
        const len = mlir.mlirArrayAttrGetNumElements(attr);
        var i: isize = 0;
        while (i < len) : (i += 1) {
            if (attributeUsesShardy(mlir.mlirArrayAttrGetElement(attr, i))) return true;
        }
    }
    if (mlir.mlirAttributeIsADictionary(attr)) {
        const len = mlir.mlirDictionaryAttrGetNumElements(attr);
        var i: isize = 0;
        while (i < len) : (i += 1) {
            const named = mlir.mlirDictionaryAttrGetElement(attr, i);
            if (std.mem.startsWith(u8, namedAttributeName(named), "sdy.")) return true;
            if (attributeUsesShardy(named.attribute)) return true;
        }
    }
    return false;
}

fn operationUsesShardy(op: mlir.MlirOperation) bool {
    if (std.mem.startsWith(u8, operationName(op), "sdy.")) return true;

    const n_attrs = mlir.mlirOperationGetNumAttributes(op);
    var attr_index: isize = 0;
    while (attr_index < n_attrs) : (attr_index += 1) {
        const named = mlir.mlirOperationGetAttribute(op, attr_index);
        if (std.mem.startsWith(u8, namedAttributeName(named), "sdy.") or attributeUsesShardy(named.attribute)) return true;
    }

    const n_regions = mlir.mlirOperationGetNumRegions(op);
    var region_index: isize = 0;
    while (region_index < n_regions) : (region_index += 1) {
        var block = mlir.mlirRegionGetFirstBlock(mlir.mlirOperationGetRegion(op, region_index));
        while (!mlir.mlirBlockIsNull(block)) : (block = mlir.mlirBlockGetNextInRegion(block)) {
            var child = mlir.mlirBlockGetFirstOperation(block);
            while (!mlir.mlirOperationIsNull(child)) : (child = mlir.mlirOperationGetNextInBlock(child)) {
                if (operationUsesShardy(child)) return true;
            }
        }
    }
    return false;
}

fn parseAndRunMlirWithCapi(module_text: []const u8, writer: *std.Io.Writer) AnalyzeError!MlirSession {
    const registry = mlir.mlirDialectRegistryCreate();
    if (mlir.mlirDialectRegistryIsNull(registry)) {
        try writer.writeAll("invalid StableHLO module: failed to create MLIR dialect registry");
        return error.InvalidStablehloModule;
    }

    insertDialect(registry, mlir.mlirGetDialectHandle__func__());
    insertDialect(registry, mlir.mlirGetDialectHandle__shape__());
    insertDialect(registry, mlir.mlirGetDialectHandle__chlo__());
    insertDialect(registry, mlir.mlirGetDialectHandle__sdy__());
    insertDialect(registry, mlir.mlirGetDialectHandle__stablehlo__());
    mlir.mlirRegisterFuncExtensions(registry);

    const context = mlir.mlirContextCreateWithRegistry(registry, false);
    if (mlir.mlirContextIsNull(context)) {
        mlir.mlirDialectRegistryDestroy(registry);
        try writer.writeAll("invalid StableHLO module: failed to create MLIR context");
        return error.InvalidStablehloModule;
    }
    var session: MlirSession = .{
        .registry = registry,
        .context = context,
        .module = .{ .ptr = null },
        .pass_manager = .{ .ptr = null },
    };
    errdefer session.deinit();

    mlir.mlirContextSetAllowUnregisteredDialects(context, false);
    mlir.mlirContextLoadAllAvailableDialects(context);

    try loadDialect(context, mlir.mlirGetDialectHandle__func__());
    try loadDialect(context, mlir.mlirGetDialectHandle__shape__());
    try loadDialect(context, mlir.mlirGetDialectHandle__chlo__());
    try loadDialect(context, mlir.mlirGetDialectHandle__sdy__());
    try loadDialect(context, mlir.mlirGetDialectHandle__stablehlo__());

    session.module = mlir.mlirModuleCreateParse(context, mlirStringRef(module_text));
    if (mlir.mlirModuleIsNull(session.module)) {
        try writer.writeAll("invalid StableHLO module: MLIR parser rejected module");
        return error.InvalidStablehloModule;
    }

    const op = mlir.mlirModuleGetOperation(session.module);
    if (!mlir.mlirOperationVerify(op)) {
        try writer.writeAll("invalid StableHLO module: MLIR verifier rejected module");
        return error.InvalidStablehloModule;
    }

    const uses_shardy = operationUsesShardy(op);
    registerTransformPassesOnce();
    if (uses_shardy) registerShardyPassesOnce();

    session.pass_manager = mlir.mlirPassManagerCreate(context);
    if (mlir.mlirPassManagerIsNull(session.pass_manager)) {
        try writer.writeAll("invalid StableHLO module: failed to create MLIR pass manager");
        return error.InvalidStablehloModule;
    }
    mlir.mlirPassManagerEnableVerifier(session.pass_manager, true);

    if (uses_shardy) try addPipeline(session.pass_manager, "sdy-propagation-pipeline", writer);
    try addPipeline(session.pass_manager, "canonicalize", writer);
    try addPipeline(session.pass_manager, "cse", writer);
    try addPipeline(session.pass_manager, "canonicalize", writer);

    if (!mlir.pjrtxMlirPassManagerRunOnOpSucceeded(session.pass_manager, op)) {
        try writer.writeAll("invalid StableHLO module: MLIR pass pipeline failed");
        return error.InvalidStablehloModule;
    }

    return session;
}

pub const ShardingPlan = core.ShardingPlan;
pub const Value = core.Value;
pub const ValueId = core.ValueId;
pub const ValueRole = core.ValueRole;
pub const PlanInstructionKind = core.PlanInstructionKind;
pub const PlanInstruction = core.PlanInstruction;
pub const ExecutablePlan = core.ExecutablePlan;

pub fn parseTextCompileOptionsFromReader(allocator: std.mem.Allocator, reader: *std.Io.Reader) !CompileOptions {
    const text = try reader.allocRemaining(allocator, .limited(64 * 1024));
    defer allocator.free(text);

    var options: CompileOptions = .{};
    var assignment = std.ArrayList(i32).empty;
    errdefer assignment.deinit(allocator);

    var it = std.mem.splitScalar(u8, text, ';');
    while (it.next()) |part| {
        const trimmed = std.mem.trim(u8, part, " \n\t");
        if (trimmed.len == 0) continue;
        const eq = std.mem.indexOfScalar(u8, trimmed, '=') orelse return error.InvalidCompileOption;
        const key = std.mem.trim(u8, trimmed[0..eq], " \n\t");
        const value = std.mem.trim(u8, trimmed[eq + 1 ..], " \n\t");

        if (std.mem.eql(u8, key, "replicas")) {
            options.num_replicas = try std.fmt.parseInt(i32, value, 10);
        } else if (std.mem.eql(u8, key, "partitions")) {
            options.num_partitions = try std.fmt.parseInt(i32, value, 10);
        } else if (std.mem.eql(u8, key, "use_shardy")) {
            options.use_shardy_partitioner = std.mem.eql(u8, value, "true") or std.mem.eql(u8, value, "1");
        } else if (std.mem.eql(u8, key, "assignment")) {
            var ids = std.mem.splitScalar(u8, value, ',');
            while (ids.next()) |id_text| {
                const id_trimmed = std.mem.trim(u8, id_text, " \n\t");
                if (id_trimmed.len == 0) continue;
                try assignment.append(allocator, try std.fmt.parseInt(i32, id_trimmed, 10));
            }
        } else {
            return error.UnknownCompileOption;
        }
    }

    if (options.num_replicas < 1 or options.num_partitions < 1) return error.InvalidDeviceTopology;
    if (assignment.items.len != 0 and assignment.items.len < options.numDevices()) return error.InvalidDeviceAssignment;
    options.device_assignment = try assignment.toOwnedSlice(allocator);
    return options;
}

pub fn parseTextCompileOptions(allocator: std.mem.Allocator, text: []const u8) !CompileOptions {
    var reader: std.Io.Reader = .fixed(text);
    return parseTextCompileOptionsFromReader(allocator, &reader);
}

pub fn makeReplicatedPlan(
    allocator: std.mem.Allocator,
    options: CompileOptions,
    num_parameters: usize,
    num_outputs: usize,
) !ExecutablePlan {
    const assignment = if (options.device_assignment.len == 0) blk: {
        const generated = try allocator.alloc(i32, options.numDevices());
        for (generated, 0..) |*id, i| id.* = @intCast(i);
        break :blk generated;
    } else try allocator.dupe(i32, options.device_assignment);
    errdefer allocator.free(assignment);

    var owned_options = options;
    owned_options.device_assignment = assignment;

    const parameter_shardings = try allocator.alloc(ShardingPlan, num_parameters);
    errdefer allocator.free(parameter_shardings);
    const output_shardings = try allocator.alloc(ShardingPlan, num_outputs);
    errdefer allocator.free(output_shardings);
    const values = try makeBootstrapValues(allocator, num_parameters, 1);
    errdefer allocator.free(values);
    const instructions = try makeCopyArg0Instructions(allocator, num_parameters);
    errdefer freeInstructions(allocator, instructions);

    var initialized_parameters: usize = 0;
    errdefer for (parameter_shardings[0..initialized_parameters]) |plan| {
        allocator.free(plan.mesh_name);
        allocator.free(plan.device_assignment);
    };
    for (parameter_shardings) |*plan| {
        plan.* = try makeShardingPlan(allocator, .replicated, "pjrtx_mesh", assignment);
        initialized_parameters += 1;
    }

    var initialized_outputs: usize = 0;
    errdefer for (output_shardings[0..initialized_outputs]) |plan| {
        allocator.free(plan.mesh_name);
        allocator.free(plan.device_assignment);
    };
    for (output_shardings) |*plan| {
        plan.* = try makeShardingPlan(allocator, .replicated, "pjrtx_mesh", assignment);
        initialized_outputs += 1;
    }

    return .{
        .allocator = allocator,
        .options = owned_options,
        .values = values,
        .parameter_shardings = parameter_shardings,
        .output_shardings = output_shardings,
        .instructions = instructions,
    };
}

fn makeShardingPlan(
    allocator: std.mem.Allocator,
    kind: ShardingKind,
    mesh_name: []const u8,
    assignment: []const i32,
) !ShardingPlan {
    const owned_mesh_name = try allocator.dupe(u8, mesh_name);
    errdefer allocator.free(owned_mesh_name);
    return .{
        .kind = kind,
        .mesh_name = owned_mesh_name,
        .device_assignment = try allocator.dupe(i32, assignment),
    };
}

fn replaceShardingPlan(
    allocator: std.mem.Allocator,
    target: *ShardingPlan,
    kind: ShardingKind,
    mesh_name: []const u8,
    assignment: []const i32,
) !void {
    const replacement = try makeShardingPlan(allocator, kind, mesh_name, assignment);
    allocator.free(target.mesh_name);
    allocator.free(target.device_assignment);
    target.* = replacement;
}

fn applyAnalysisShardingMetadataToPlan(
    allocator: std.mem.Allocator,
    plan: *ExecutablePlan,
    analysis: ModuleAnalysis,
) !void {
    for (analysis.parameter_shardings, 0..) |metadata, index| {
        if (index >= plan.parameter_shardings.len) break;
        try replaceShardingPlan(allocator, &plan.parameter_shardings[index], metadata.kind, metadata.mesh_name, plan.options.device_assignment);
    }
    for (analysis.output_shardings, 0..) |metadata, index| {
        if (index >= plan.output_shardings.len) break;
        try replaceShardingPlan(allocator, &plan.output_shardings[index], metadata.kind, metadata.mesh_name, plan.options.device_assignment);
    }
}

pub fn makeExecutablePlan(
    allocator: std.mem.Allocator,
    options: CompileOptions,
    analysis: ModuleAnalysis,
) !ExecutablePlan {
    var plan = try makeReplicatedPlan(allocator, options, analysis.num_parameters, analysis.num_outputs);
    errdefer plan.deinit();
    if (options.use_shardy_partitioner) {
        try applyAnalysisShardingMetadataToPlan(allocator, &plan, analysis);
    }
    freeInstructions(allocator, plan.instructions);
    plan.instructions = &.{};
    plan.instructions = try lowerAnalysisOpsToPlan(allocator, analysis.ops, analysis.num_parameters);
    allocator.free(plan.values);
    plan.values = &.{};
    plan.values = try makeBootstrapValues(allocator, analysis.num_parameters, plan.instructions.len);
    return plan;
}

fn instructionKindFromStablehlo(name: []const u8) PlanInstructionKind {
    if (std.mem.eql(u8, name, "add")) return .add;
    if (std.mem.eql(u8, name, "subtract")) return .subtract;
    if (std.mem.eql(u8, name, "multiply")) return .multiply;
    if (std.mem.eql(u8, name, "divide")) return .divide;
    if (std.mem.eql(u8, name, "negate")) return .negate;
    if (std.mem.eql(u8, name, "exponential")) return .exp;
    if (std.mem.eql(u8, name, "tanh")) return .tanh;
    if (std.mem.eql(u8, name, "sqrt")) return .sqrt;
    if (std.mem.eql(u8, name, "rsqrt")) return .rsqrt;
    if (std.mem.eql(u8, name, "reshape")) return .reshape;
    if (std.mem.eql(u8, name, "transpose")) return .transpose;
    if (std.mem.eql(u8, name, "broadcast_in_dim")) return .broadcast_in_dim;
    if (std.mem.eql(u8, name, "slice")) return .slice;
    if (std.mem.eql(u8, name, "concatenate")) return .concatenate;
    return .unsupported;
}

fn makeValue(id: u32, role: ValueRole) Value {
    return .{
        .id = .{ .index = id },
        .role = role,
        .descriptor = .{
            .element_type = .invalid,
            .dims = &.{},
            .device_id = -1,
            .memory_id = -1,
            .shard_index = 0,
        },
    };
}

fn makeBootstrapValues(allocator: std.mem.Allocator, num_parameters: usize, num_instruction_results: usize) ![]Value {
    const value_count = num_parameters + num_instruction_results;
    const values = try allocator.alloc(Value, value_count);
    for (values[0..num_parameters], 0..) |*value, index| {
        value.* = makeValue(@intCast(index), .parameter);
    }
    for (values[num_parameters..], 0..) |*value, index| {
        value.* = makeValue(@intCast(num_parameters + index), .instruction_result);
    }
    return values;
}

fn instructionInputs(allocator: std.mem.Allocator, kind: PlanInstructionKind, previous_value: ValueId, second_parameter: ?ValueId) ![]const ValueId {
    return switch (kind) {
        .copy_arg0, .negate, .exp, .tanh, .sqrt, .rsqrt, .reshape, .transpose, .broadcast_in_dim, .slice => allocator.dupe(ValueId, &.{previous_value}),
        .add, .subtract, .multiply, .divide, .concatenate => blk: {
            const rhs = second_parameter orelse previous_value;
            break :blk allocator.dupe(ValueId, &.{ previous_value, rhs });
        },
        .unsupported => &.{},
    };
}

fn makeCopyArg0Instructions(allocator: std.mem.Allocator, num_parameters: usize) ![]PlanInstruction {
    const output: ValueId = .{ .index = @intCast(num_parameters) };
    const inputs = try allocator.dupe(ValueId, &.{.{ .index = 0 }});
    errdefer allocator.free(inputs);
    const outputs = try allocator.dupe(ValueId, &.{output});
    errdefer allocator.free(outputs);
    return allocator.dupe(PlanInstruction, &.{.{
        .kind = .copy_arg0,
        .inputs = inputs,
        .outputs = outputs,
    }});
}

fn freeInstructions(allocator: std.mem.Allocator, instructions: []PlanInstruction) void {
    for (instructions) |instruction| {
        if (instruction.inputs.len != 0) allocator.free(instruction.inputs);
        if (instruction.outputs.len != 0) allocator.free(instruction.outputs);
        if (instruction.dims) |dims| allocator.free(dims);
        if (instruction.permutation) |permutation| allocator.free(permutation);
        if (instruction.broadcast_dimensions) |broadcast_dimensions| allocator.free(broadcast_dimensions);
        if (instruction.start_indices) |start_indices| allocator.free(start_indices);
        if (instruction.limit_indices) |limit_indices| allocator.free(limit_indices);
        if (instruction.strides) |strides| allocator.free(strides);
    }
    allocator.free(instructions);
}

fn lowerAnalysisOpsToPlan(allocator: std.mem.Allocator, ops: []const Operation, num_parameters: usize) ![]PlanInstruction {
    if (ops.len == 0) return makeCopyArg0Instructions(allocator, num_parameters);
    const plan_instructions = try allocator.alloc(PlanInstruction, ops.len);
    errdefer {
        freeInstructions(allocator, plan_instructions);
    }
    @memset(plan_instructions, .{ .kind = .unsupported });
    var previous_value: ValueId = .{ .index = 0 };
    const second_parameter: ?ValueId = if (num_parameters > 1) .{ .index = 1 } else null;
    for (ops, plan_instructions, 0..) |op, *plan_instruction, index| {
        const kind = instructionKindFromStablehlo(op.name);
        const output: ValueId = .{ .index = @intCast(num_parameters + index) };
        plan_instruction.* = .{
            .kind = kind,
            .inputs = try instructionInputs(allocator, kind, previous_value, second_parameter),
            .outputs = try allocator.dupe(ValueId, &.{output}),
            .dims = if (kind == .reshape or kind == .transpose or kind == .broadcast_in_dim or kind == .slice or kind == .concatenate) try allocator.dupe(i64, op.dims) else null,
            .permutation = if (kind == .transpose) try allocator.dupe(i64, op.permutation) else null,
            .broadcast_dimensions = if (kind == .broadcast_in_dim) try allocator.dupe(i64, op.broadcast_dimensions) else null,
            .start_indices = if (kind == .slice) try allocator.dupe(i64, op.start_indices) else null,
            .limit_indices = if (kind == .slice) try allocator.dupe(i64, op.limit_indices) else null,
            .strides = if (kind == .slice) try allocator.dupe(i64, op.strides) else null,
            .dimension = if (kind == .concatenate) op.dimension else null,
        };
        previous_value = output;
    }
    return plan_instructions;
}

fn addDialect(list: *std.ArrayList(Dialect), allocator: std.mem.Allocator, dialect: Dialect) !void {
    for (list.items) |existing| {
        if (existing == dialect) return;
    }
    try list.append(allocator, dialect);
}

fn stablehloOpSupported(name: []const u8) bool {
    const supported = [_][]const u8{
        "constant",
        "return",
        "add",
        "subtract",
        "multiply",
        "divide",
        "negate",
        "exponential",
        "tanh",
        "sqrt",
        "rsqrt",
        "compare",
        "select",
        "reshape",
        "broadcast_in_dim",
        "transpose",
        "slice",
        "concatenate",
        "reduce",
        "dot_general",
    };
    for (supported) |candidate| {
        if (std.mem.eql(u8, name, candidate)) return true;
    }
    return false;
}

fn writeOpDiagnostic(
    writer: *std.Io.Writer,
    comptime label: []const u8,
    op: Operation,
    missing_feature: []const u8,
) std.Io.Writer.Error!void {
    try writer.print(
        "{s}: op={s} loc={d}:{d} dtype={s} rank=",
        .{
            label,
            op.name,
            op.line,
            op.column,
            op.dtype,
        },
    );
    if (op.rank) |rank| {
        try writer.print("{d}", .{rank});
    } else {
        try writer.writeAll("unknown");
    }
    try writer.print(" sharding={s} feature={s}", .{ op.sharding, missing_feature });
}

fn writeSimpleDiagnostic(
    writer: *std.Io.Writer,
    comptime label: []const u8,
    line: usize,
    column: usize,
    detail: []const u8,
    feature: []const u8,
) std.Io.Writer.Error!void {
    try writer.print("{s}: loc={d}:{d} detail=\"{s}\" feature={s}", .{ label, line, column, detail, feature });
}

fn mlirLocationLineColumn(loc: mlir.MlirLocation) SourceLoc {
    if (mlir.mlirLocationIsAFileLineColRange(loc)) {
        return .{
            .line = @intCast(mlir.mlirLocationFileLineColRangeGetStartLine(loc)),
            .column = @intCast(mlir.mlirLocationFileLineColRangeGetStartColumn(loc)),
        };
    }
    if (mlir.mlirLocationIsAName(loc)) return mlirLocationLineColumn(mlir.mlirLocationNameGetChildLoc(loc));
    if (mlir.mlirLocationIsACallSite(loc)) return mlirLocationLineColumn(mlir.mlirLocationCallSiteGetCallee(loc));
    if (mlir.mlirLocationIsAFused(loc) and mlir.mlirLocationFusedGetNumLocations(loc) > 0) {
        var child: mlir.MlirLocation = undefined;
        mlir.mlirLocationFusedGetLocations(loc, &child);
        return mlirLocationLineColumn(child);
    }
    return .{ .line = 0, .column = 0 };
}

fn typeDtype(allocator: std.mem.Allocator, ty: mlir.MlirType) ![]u8 {
    const element = if (mlir.mlirTypeIsAShaped(ty)) mlir.mlirShapedTypeGetElementType(ty) else ty;
    if (mlir.mlirTypeIsAF16(element)) return allocator.dupe(u8, "f16");
    if (mlir.mlirTypeIsAF32(element)) return allocator.dupe(u8, "f32");
    if (mlir.mlirTypeIsAF64(element)) return allocator.dupe(u8, "f64");
    if (mlir.mlirTypeIsABF16(element)) return allocator.dupe(u8, "bf16");
    if (mlir.mlirTypeIsAInteger(element)) {
        const width = mlir.mlirIntegerTypeGetWidth(element);
        if (width == 1) return allocator.dupe(u8, "pred");
        const prefix: []const u8 = if (mlir.mlirIntegerTypeIsUnsigned(element)) "u" else "i";
        return std.fmt.allocPrint(allocator, "{s}{d}", .{ prefix, width });
    }
    return allocator.dupe(u8, "unknown");
}

fn typeRank(ty: mlir.MlirType) ?usize {
    if (!mlir.mlirTypeIsAShaped(ty) or !mlir.mlirShapedTypeHasRank(ty)) return null;
    return @intCast(mlir.mlirShapedTypeGetRank(ty));
}

fn typeDims(allocator: std.mem.Allocator, ty: mlir.MlirType) ![]const i64 {
    if (!mlir.mlirTypeIsAShaped(ty) or !mlir.mlirShapedTypeHasRank(ty)) return allocator.dupe(i64, &.{});
    const rank: usize = @intCast(mlir.mlirShapedTypeGetRank(ty));
    const dims = try allocator.alloc(i64, rank);
    var dim_index: isize = 0;
    while (dim_index < @as(isize, @intCast(rank))) : (dim_index += 1) {
        dims[@intCast(dim_index)] = mlir.mlirShapedTypeGetDimSize(ty, dim_index);
    }
    return dims;
}

fn intListAttribute(allocator: std.mem.Allocator, attr: mlir.MlirAttribute) ![]const i64 {
    if (mlir.mlirAttributeIsNull(attr)) return allocator.dupe(i64, &.{});
    if (mlir.mlirAttributeIsADenseI64Array(attr)) {
        const count: usize = @intCast(mlir.mlirDenseArrayGetNumElements(attr));
        const values = try allocator.alloc(i64, count);
        var index: isize = 0;
        while (index < @as(isize, @intCast(count))) : (index += 1) {
            values[@intCast(index)] = mlir.mlirDenseI64ArrayGetElement(attr, index);
        }
        return values;
    }
    if (mlir.mlirAttributeIsADenseIntElements(attr)) {
        const count: usize = @intCast(mlir.mlirElementsAttrGetNumElements(attr));
        const values = try allocator.alloc(i64, count);
        var index: isize = 0;
        while (index < @as(isize, @intCast(count))) : (index += 1) {
            values[@intCast(index)] = mlir.mlirDenseElementsAttrGetInt64Value(attr, index);
        }
        return values;
    }
    return allocator.dupe(i64, &.{});
}

fn intAttribute(attr: mlir.MlirAttribute) ?i64 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAInteger(attr)) return null;
    return mlir.mlirIntegerAttrGetValueInt(attr);
}

fn resultOrOperandType(op: mlir.MlirOperation) mlir.MlirType {
    if (mlir.mlirOperationGetNumResults(op) > 0) {
        return mlir.mlirValueGetType(mlir.mlirOperationGetResult(op, 0));
    }
    if (mlir.mlirOperationGetNumOperands(op) > 0) {
        return mlir.mlirValueGetType(mlir.mlirOperationGetOperand(op, 0));
    }
    return .{ .ptr = null };
}

fn operationShardingLabel(allocator: std.mem.Allocator, op: mlir.MlirOperation) ![]u8 {
    const sharding = getOperationAttribute(op, "sdy.sharding");
    if (!mlir.mlirAttributeIsNull(sharding)) {
        if (mlir.sdyAttributeIsATensorShardingPerValueAttr(sharding)) return allocator.dupe(u8, "sdy.sharding_per_value");
        return allocator.dupe(u8, "sdy.sharding");
    }
    if (!mlir.mlirAttributeIsNull(getOperationAttribute(op, "mhlo.sharding"))) return allocator.dupe(u8, "mhlo.sharding");
    return allocator.dupe(u8, "unspecified");
}

fn meshNameFromTensorSharding(allocator: std.mem.Allocator, attr: mlir.MlirAttribute) ![]u8 {
    const mesh = mlir.sdyTensorShardingAttrGetMeshOrRef(attr);
    if (!mlir.mlirAttributeIsNull(mesh)) {
        if (mlir.mlirAttributeIsAFlatSymbolRef(mesh)) {
            return allocator.dupe(u8, mlirStringSlice(mlir.mlirFlatSymbolRefAttrGetValue(mesh)));
        }
        if (mlir.mlirAttributeIsASymbolRef(mesh)) {
            return allocator.dupe(u8, mlirStringSlice(mlir.mlirSymbolRefAttrGetRootReference(mesh)));
        }
    }
    return allocator.dupe(u8, "pjrtx_mesh");
}

fn makeShardingMetadata(allocator: std.mem.Allocator, kind: ShardingKind, mesh_name: []const u8) !ShardingMetadata {
    return .{ .kind = kind, .mesh_name = try allocator.dupe(u8, mesh_name) };
}

fn metadataFromTensorSharding(allocator: std.mem.Allocator, attr: mlir.MlirAttribute) !ShardingMetadata {
    var kind: ShardingKind = .replicated;
    const dim_count = mlir.sdyTensorShardingAttrGetDimShardingsSize(attr);
    var dim_index: isize = 0;
    while (dim_index < dim_count) : (dim_index += 1) {
        const dim = mlir.sdyTensorShardingAttrGetDimShardingsElem(attr, dim_index);
        if (mlir.sdyAttributeIsADimensionShardingAttr(dim) and mlir.sdyDimensionShardingAttrGetAxesSize(dim) > 0) {
            kind = .partitioned;
            break;
        }
    }

    const mesh_name = try meshNameFromTensorSharding(allocator, attr);
    return .{ .kind = kind, .mesh_name = mesh_name };
}

fn metadataFromShardingAttribute(allocator: std.mem.Allocator, attr: mlir.MlirAttribute, value_index: usize) !?ShardingMetadata {
    if (mlir.mlirAttributeIsNull(attr)) return null;
    if (mlir.sdyAttributeIsATensorShardingAttr(attr)) return try metadataFromTensorSharding(allocator, attr);
    if (mlir.sdyAttributeIsATensorShardingPerValueAttr(attr)) {
        const count = mlir.sdyTensorShardingPerValueAttrGetShardingsSize(attr);
        if (value_index >= @as(usize, @intCast(count))) return null;
        return try metadataFromTensorSharding(allocator, mlir.sdyTensorShardingPerValueAttrGetShardingsElem(attr, @intCast(value_index)));
    }
    return null;
}

fn metadataFromDictionarySharding(allocator: std.mem.Allocator, dictionary: mlir.MlirAttribute) !?ShardingMetadata {
    if (mlir.mlirAttributeIsNull(dictionary) or !mlir.mlirAttributeIsADictionary(dictionary)) return null;
    return try metadataFromShardingAttribute(
        allocator,
        mlir.mlirDictionaryAttrGetElementByName(dictionary, mlirStringRef("sdy.sharding")),
        0,
    );
}

fn metadataFromArrayDictionarySharding(allocator: std.mem.Allocator, array: mlir.MlirAttribute, index: usize) !?ShardingMetadata {
    if (mlir.mlirAttributeIsNull(array) or !mlir.mlirAttributeIsAArray(array)) return null;
    const count = mlir.mlirArrayAttrGetNumElements(array);
    if (index >= @as(usize, @intCast(count))) return null;
    return try metadataFromDictionarySharding(allocator, mlir.mlirArrayAttrGetElement(array, @intCast(index)));
}

fn copyShardingMetadata(allocator: std.mem.Allocator, metadata: ShardingMetadata) !ShardingMetadata {
    return makeShardingMetadata(allocator, metadata.kind, metadata.mesh_name);
}

const CapiAnalysisBuilder = struct {
    allocator: std.mem.Allocator,
    diagnostic_writer: *std.Io.Writer,
    dialects: std.ArrayList(Dialect) = .empty,
    ops: std.ArrayList(Operation) = .empty,
    parameter_shardings: std.ArrayList(ShardingMetadata) = .empty,
    output_shardings: std.ArrayList(ShardingMetadata) = .empty,
    num_parameters: usize = 0,
    num_outputs: usize = 0,
    saw_program_body: bool = false,
    saw_shardy: bool = false,
    saw_manual_computation: bool = false,
    saw_sdy_return: bool = false,
    manual_line: usize = 0,
    manual_column: usize = 0,

    fn deinitPartial(self: *CapiAnalysisBuilder) void {
        for (self.output_shardings.items) |sharding| sharding.deinit(self.allocator);
        for (self.parameter_shardings.items) |sharding| sharding.deinit(self.allocator);
        for (self.ops.items) |op| op.deinit(self.allocator);
        self.output_shardings.deinit(self.allocator);
        self.parameter_shardings.deinit(self.allocator);
        self.ops.deinit(self.allocator);
        self.dialects.deinit(self.allocator);
    }

    fn ensureShardings(self: *CapiAnalysisBuilder, list: *std.ArrayList(ShardingMetadata), count: usize) !void {
        while (list.items.len < count) {
            try list.append(self.allocator, try makeShardingMetadata(self.allocator, .replicated, "pjrtx_mesh"));
        }
    }

    fn replaceMetadata(self: *CapiAnalysisBuilder, list: *std.ArrayList(ShardingMetadata), index: usize, metadata: ShardingMetadata) void {
        list.items[index].deinit(self.allocator);
        list.items[index] = metadata;
    }
};

fn operationHasAttributeNamed(op: mlir.MlirOperation, name: []const u8) bool {
    return !mlir.mlirAttributeIsNull(getOperationAttribute(op, name));
}

fn analyzeStablehloOperationFromCapi(builder: *CapiAnalysisBuilder, op: mlir.MlirOperation, op_name: []const u8) AnalyzeError!void {
    builder.saw_program_body = true;
    try addDialect(&builder.dialects, builder.allocator, .stablehlo);

    const short_name = op_name["stablehlo.".len..];
    const ty = resultOrOperandType(op);
    const dtype = if (mlir.mlirTypeIsNull(ty)) try builder.allocator.dupe(u8, "unknown") else try typeDtype(builder.allocator, ty);
    var owns_dtype = true;
    errdefer if (owns_dtype) builder.allocator.free(dtype);
    const dims = if (mlir.mlirTypeIsNull(ty)) try builder.allocator.dupe(i64, &.{}) else try typeDims(builder.allocator, ty);
    var owns_dims = true;
    errdefer if (owns_dims) builder.allocator.free(dims);
    const permutation = if (std.mem.eql(u8, short_name, "transpose"))
        try intListAttribute(builder.allocator, getOperationAttribute(op, "permutation"))
    else
        try builder.allocator.dupe(i64, &.{});
    var owns_permutation = true;
    errdefer if (owns_permutation) builder.allocator.free(permutation);
    const broadcast_dimensions = if (std.mem.eql(u8, short_name, "broadcast_in_dim"))
        try intListAttribute(builder.allocator, getOperationAttribute(op, "broadcast_dimensions"))
    else
        try builder.allocator.dupe(i64, &.{});
    var owns_broadcast_dimensions = true;
    errdefer if (owns_broadcast_dimensions) builder.allocator.free(broadcast_dimensions);
    const start_indices = if (std.mem.eql(u8, short_name, "slice"))
        try intListAttribute(builder.allocator, getOperationAttribute(op, "start_indices"))
    else
        try builder.allocator.dupe(i64, &.{});
    var owns_start_indices = true;
    errdefer if (owns_start_indices) builder.allocator.free(start_indices);
    const limit_indices = if (std.mem.eql(u8, short_name, "slice"))
        try intListAttribute(builder.allocator, getOperationAttribute(op, "limit_indices"))
    else
        try builder.allocator.dupe(i64, &.{});
    var owns_limit_indices = true;
    errdefer if (owns_limit_indices) builder.allocator.free(limit_indices);
    const strides = if (std.mem.eql(u8, short_name, "slice"))
        try intListAttribute(builder.allocator, getOperationAttribute(op, "strides"))
    else
        try builder.allocator.dupe(i64, &.{});
    var owns_strides = true;
    errdefer if (owns_strides) builder.allocator.free(strides);
    const dimension = if (std.mem.eql(u8, short_name, "concatenate"))
        intAttribute(getOperationAttribute(op, "dimension"))
    else
        null;
    const sharding = try operationShardingLabel(builder.allocator, op);
    var owns_sharding = true;
    errdefer if (owns_sharding) builder.allocator.free(sharding);
    const loc = mlirLocationLineColumn(mlir.mlirOperationGetLocation(op));
    const owned_name = try builder.allocator.dupe(u8, short_name);
    var owns_name = true;
    errdefer if (owns_name) builder.allocator.free(owned_name);

    const analyzed: Operation = .{
        .name = owned_name,
        .line = loc.line,
        .column = loc.column,
        .dtype = dtype,
        .rank = if (mlir.mlirTypeIsNull(ty)) null else typeRank(ty),
        .dims = dims,
        .permutation = permutation,
        .broadcast_dimensions = broadcast_dimensions,
        .start_indices = start_indices,
        .limit_indices = limit_indices,
        .strides = strides,
        .dimension = dimension,
        .sharding = sharding,
    };
    if (!stablehloOpSupported(short_name)) {
        try writeOpDiagnostic(builder.diagnostic_writer, "unsupported op", analyzed, "stablehlo-op");
        analyzed.deinit(builder.allocator);
        owns_dtype = false;
        owns_dims = false;
        owns_permutation = false;
        owns_broadcast_dimensions = false;
        owns_start_indices = false;
        owns_limit_indices = false;
        owns_strides = false;
        owns_sharding = false;
        owns_name = false;
        return error.UnsupportedOp;
    }
    try builder.ops.append(builder.allocator, analyzed);
    owns_dtype = false;
    owns_dims = false;
    owns_permutation = false;
    owns_broadcast_dimensions = false;
    owns_start_indices = false;
    owns_limit_indices = false;
    owns_strides = false;
    owns_sharding = false;
    owns_name = false;
}

fn valueShardingMetadata(builder: *CapiAnalysisBuilder, value: mlir.MlirValue) AnalyzeError!?ShardingMetadata {
    if (mlir.mlirValueIsNull(value)) return null;
    if (mlir.mlirValueIsABlockArgument(value)) {
        const arg_index = @as(usize, @intCast(mlir.mlirBlockArgumentGetArgNumber(value)));
        if (arg_index < builder.parameter_shardings.items.len) {
            return try copyShardingMetadata(builder.allocator, builder.parameter_shardings.items[arg_index]);
        }
        return null;
    }
    if (mlir.mlirValueIsAOpResult(value)) {
        const owner = mlir.mlirOpResultGetOwner(value);
        const result_index = @as(usize, @intCast(mlir.mlirOpResultGetResultNumber(value)));
        return try metadataFromShardingAttribute(builder.allocator, getOperationAttribute(owner, "sdy.sharding"), result_index);
    }
    return null;
}

fn analyzeFunctionFromCapi(builder: *CapiAnalysisBuilder, op: mlir.MlirOperation) AnalyzeError!void {
    builder.saw_program_body = true;
    try addDialect(&builder.dialects, builder.allocator, .func);

    var num_inputs: usize = 0;
    var num_results: usize = 0;
    const function_type_attr = getOperationAttribute(op, "function_type");
    if (!mlir.mlirAttributeIsNull(function_type_attr) and mlir.mlirAttributeIsAType(function_type_attr)) {
        const function_type = mlir.mlirTypeAttrGetValue(function_type_attr);
        if (mlir.mlirTypeIsAFunction(function_type)) {
            num_inputs = @intCast(mlir.mlirFunctionTypeGetNumInputs(function_type));
            num_results = @intCast(mlir.mlirFunctionTypeGetNumResults(function_type));
        }
    }

    if (mlir.mlirOperationGetNumRegions(op) > 0) {
        const entry = mlir.mlirRegionGetFirstBlock(mlir.mlirOperationGetRegion(op, 0));
        if (!mlir.mlirBlockIsNull(entry)) {
            num_inputs = @max(num_inputs, @as(usize, @intCast(mlir.mlirBlockGetNumArguments(entry))));
            const terminator = mlir.mlirBlockGetTerminator(entry);
            if (!mlir.mlirOperationIsNull(terminator) and std.mem.eql(u8, operationName(terminator), "func.return")) {
                num_results = @max(num_results, @as(usize, @intCast(mlir.mlirOperationGetNumOperands(terminator))));
            }
        }
    }

    builder.num_parameters = @max(builder.num_parameters, num_inputs);
    builder.num_outputs = @max(builder.num_outputs, num_results);
    try builder.ensureShardings(&builder.parameter_shardings, builder.num_parameters);
    try builder.ensureShardings(&builder.output_shardings, builder.num_outputs);

    const arg_attrs = getOperationAttribute(op, "arg_attrs");
    var arg_index: usize = 0;
    while (arg_index < num_inputs) : (arg_index += 1) {
        if (try metadataFromArrayDictionarySharding(builder.allocator, arg_attrs, arg_index)) |metadata| {
            builder.replaceMetadata(&builder.parameter_shardings, arg_index, metadata);
        }
    }

    const res_attrs = getOperationAttribute(op, "res_attrs");
    var result_index: usize = 0;
    while (result_index < num_results) : (result_index += 1) {
        if (try metadataFromArrayDictionarySharding(builder.allocator, res_attrs, result_index)) |metadata| {
            builder.replaceMetadata(&builder.output_shardings, result_index, metadata);
        }
    }

    if (mlir.mlirOperationGetNumRegions(op) > 0) {
        const entry = mlir.mlirRegionGetFirstBlock(mlir.mlirOperationGetRegion(op, 0));
        if (!mlir.mlirBlockIsNull(entry)) {
            const terminator = mlir.mlirBlockGetTerminator(entry);
            if (!mlir.mlirOperationIsNull(terminator) and std.mem.eql(u8, operationName(terminator), "func.return")) {
                const operand_count = mlir.mlirOperationGetNumOperands(terminator);
                var operand_index: isize = 0;
                while (operand_index < operand_count) : (operand_index += 1) {
                    const output_index = @as(usize, @intCast(operand_index));
                    if (output_index >= builder.output_shardings.items.len) break;
                    if (try valueShardingMetadata(builder, mlir.mlirOperationGetOperand(terminator, operand_index))) |metadata| {
                        builder.replaceMetadata(&builder.output_shardings, output_index, metadata);
                    }
                }
            }
        }
    }
}

fn inspectOperationAttributesFromCapi(builder: *CapiAnalysisBuilder, op: mlir.MlirOperation, loc: SourceLoc) AnalyzeError!void {
    if (operationHasAttributeNamed(op, "mhlo.sharding")) {
        try writeSimpleDiagnostic(builder.diagnostic_writer, "unsupported sharding", loc.line, loc.column, "GSPMD mhlo.sharding is not enabled; use Shardy", "gspmd-compat");
        return error.GspmdNotEnabled;
    }
    const n_attrs = mlir.mlirOperationGetNumAttributes(op);
    var attr_index: isize = 0;
    while (attr_index < n_attrs) : (attr_index += 1) {
        const named = mlir.mlirOperationGetAttribute(op, attr_index);
        if (std.mem.startsWith(u8, namedAttributeName(named), "sdy.") or attributeUsesShardy(named.attribute)) {
            builder.saw_shardy = true;
            try addDialect(&builder.dialects, builder.allocator, .sdy);
        }
    }
}

fn visitOperationFromCapi(builder: *CapiAnalysisBuilder, op: mlir.MlirOperation) AnalyzeError!void {
    const name = operationName(op);
    const loc = mlirLocationLineColumn(mlir.mlirOperationGetLocation(op));
    try inspectOperationAttributesFromCapi(builder, op, loc);

    if (std.mem.eql(u8, name, "func.func")) {
        try analyzeFunctionFromCapi(builder, op);
    } else if (std.mem.startsWith(u8, name, "stablehlo.")) {
        try analyzeStablehloOperationFromCapi(builder, op, name);
    } else if (std.mem.startsWith(u8, name, "chlo.")) {
        try writeSimpleDiagnostic(builder.diagnostic_writer, "unsupported op", loc.line, loc.column, "CHLO legalization is not wired yet", "chlo-legalization");
        return error.UnsupportedOp;
    } else if (std.mem.startsWith(u8, name, "shape.")) {
        try writeSimpleDiagnostic(builder.diagnostic_writer, "unsupported op", loc.line, loc.column, "shape dialect interop is not wired yet", "shape-legalization");
        return error.UnsupportedOp;
    } else if (std.mem.startsWith(u8, name, "sdy.")) {
        builder.saw_shardy = true;
        try addDialect(&builder.dialects, builder.allocator, .sdy);
        const short_name = name["sdy.".len..];
        if (std.mem.eql(u8, short_name, "manual_computation")) {
            builder.saw_manual_computation = true;
            builder.manual_line = loc.line;
            builder.manual_column = loc.column;
        } else if (std.mem.eql(u8, short_name, "return")) {
            builder.saw_sdy_return = true;
        } else if (!std.mem.eql(u8, short_name, "mesh") and
            !std.mem.eql(u8, short_name, "sharding") and
            !std.mem.eql(u8, short_name, "sharding_per_value"))
        {
            try writeSimpleDiagnostic(builder.diagnostic_writer, "unsupported sharding", loc.line, loc.column, short_name, "shardy-construct");
            return error.UnsupportedSharding;
        }
    }

    const n_regions = mlir.mlirOperationGetNumRegions(op);
    var region_index: isize = 0;
    while (region_index < n_regions) : (region_index += 1) {
        var block = mlir.mlirRegionGetFirstBlock(mlir.mlirOperationGetRegion(op, region_index));
        while (!mlir.mlirBlockIsNull(block)) : (block = mlir.mlirBlockGetNextInRegion(block)) {
            var child = mlir.mlirBlockGetFirstOperation(block);
            while (!mlir.mlirOperationIsNull(child)) : (child = mlir.mlirOperationGetNextInBlock(child)) {
                try visitOperationFromCapi(builder, child);
            }
        }
    }
}

fn analyzeMlirSessionWithCapi(
    allocator: std.mem.Allocator,
    source: []u8,
    session: MlirSession,
    diagnostic_writer: *std.Io.Writer,
) AnalyzeError!ModuleAnalysis {
    var builder: CapiAnalysisBuilder = .{
        .allocator = allocator,
        .diagnostic_writer = diagnostic_writer,
    };
    errdefer builder.deinitPartial();

    try visitOperationFromCapi(&builder, mlir.mlirModuleGetOperation(session.module));
    if (builder.saw_manual_computation and !builder.saw_sdy_return) {
        try writeSimpleDiagnostic(diagnostic_writer, "unsupported sharding", builder.manual_line, builder.manual_column, "sdy.manual_computation requires sdy.return in milestone 1", "shardy-manual-computation");
        return error.InvalidManualComputation;
    }
    if (!builder.saw_program_body and !builder.saw_shardy) {
        try diagnostic_writer.writeAll("invalid StableHLO module: expected func.func, stablehlo.*, or sdy.* operation");
        return error.InvalidStablehloModule;
    }
    if (builder.dialects.items.len == 0) try addDialect(&builder.dialects, allocator, .stablehlo);
    if (builder.num_outputs == 0) {
        builder.num_outputs = 1;
        try builder.ensureShardings(&builder.output_shardings, builder.num_outputs);
    }

    return .{
        .allocator = allocator,
        .source = source,
        .dialects = try builder.dialects.toOwnedSlice(allocator),
        .ops = try builder.ops.toOwnedSlice(allocator),
        .num_parameters = builder.num_parameters,
        .num_outputs = builder.num_outputs,
        .parameter_shardings = try builder.parameter_shardings.toOwnedSlice(allocator),
        .output_shardings = try builder.output_shardings.toOwnedSlice(allocator),
    };
}

pub fn analyzeProgramFromReader(
    allocator: std.mem.Allocator,
    format_text: []const u8,
    program_reader: *std.Io.Reader,
    diagnostic_writer: *std.Io.Writer,
) AnalyzeError!ModuleAnalysis {
    const format = ProgramFormat.parse(format_text);
    if (format == .unknown) {
        try diagnostic_writer.print("unsupported program format: {s}", .{format_text});
        return error.UnsupportedProgramFormat;
    }
    if (format == .stablehlo_bytecode) {
        try diagnostic_writer.writeAll("unsupported program encoding: StableHLO bytecode deserialization is not wired in milestone 1");
        return error.UnsupportedProgramEncoding;
    }

    const source = try program_reader.allocRemaining(allocator, .limited(64 * 1024 * 1024));
    errdefer allocator.free(source);
    var session = try parseAndRunMlirWithCapi(source, diagnostic_writer);
    defer session.deinit();
    return analyzeMlirSessionWithCapi(allocator, source, session, diagnostic_writer);
}

test "compile options preserve replicas partitions shardy and assignment" {
    const options = try parseTextCompileOptions(
        std.testing.allocator,
        "replicas=2; partitions=2; use_shardy=true; assignment=0,1,2,3",
    );
    defer std.testing.allocator.free(options.device_assignment);

    try std.testing.expectEqual(@as(i32, 2), options.num_replicas);
    try std.testing.expectEqual(@as(i32, 2), options.num_partitions);
    try std.testing.expect(options.use_shardy_partitioner);
    try std.testing.expectEqual(@as(usize, 4), options.numDevices());
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 2, 3 }, options.device_assignment);
}

test "replicated executable plan has per-value sharding metadata" {
    var plan = try makeReplicatedPlan(std.testing.allocator, .{ .num_partitions = 4 }, 2, 1);
    defer plan.deinit();

    try std.testing.expectEqual(@as(usize, 4), plan.options.numDevices());
    try std.testing.expectEqual(@as(usize, 2), plan.parameter_shardings.len);
    try std.testing.expectEqual(ShardingKind.replicated, plan.output_shardings[0].kind);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 2, 3 }, plan.output_shardings[0].device_assignment);
    try std.testing.expectEqual(@as(usize, 3), plan.values.len);
    try std.testing.expectEqual(ValueRole.parameter, plan.values[0].role);
    try std.testing.expectEqual(ValueRole.parameter, plan.values[1].role);
    try std.testing.expectEqual(ValueRole.instruction_result, plan.values[2].role);
    try std.testing.expectEqual(@as(usize, 1), plan.instructions.len);
    try std.testing.expectEqual(PlanInstructionKind.copy_arg0, plan.instructions[0].kind);
    try std.testing.expectEqual(@as(u32, 0), plan.instructions[0].inputs[0].index);
    try std.testing.expectEqual(@as(u32, 2), plan.instructions[0].outputs[0].index);
}

test "executable plan preserves verified shardy parameter and output metadata" {
    const module_text =
        \\sdy.mesh @mesh = <["x"=2]>
        \\func.func @main(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> tensor<4xf32> {
        \\  %0 = stablehlo.add %arg0, %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<4xf32>
        \\  return %0 : tensor<4xf32>
        \\}
    ;
    var reader: std.Io.Reader = .fixed(module_text);
    var diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer diagnostics.deinit();

    var analysis = try analyzeProgramFromReader(std.testing.allocator, "mlir", &reader, &diagnostics.writer);
    defer analysis.deinit();

    var plan = try makeExecutablePlan(std.testing.allocator, .{ .num_partitions = 2 }, analysis);
    defer plan.deinit();

    try std.testing.expectEqual(@as(usize, 1), plan.parameter_shardings.len);
    try std.testing.expectEqual(@as(usize, 1), plan.output_shardings.len);
    try std.testing.expectEqual(ShardingKind.partitioned, plan.parameter_shardings[0].kind);
    try std.testing.expectEqual(ShardingKind.partitioned, plan.output_shardings[0].kind);
    try std.testing.expectEqualStrings("mesh", plan.parameter_shardings[0].mesh_name);
    try std.testing.expectEqualStrings("mesh", plan.output_shardings[0].mesh_name);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1 }, plan.output_shardings[0].device_assignment);
    try std.testing.expectEqual(@as(usize, 1), plan.instructions.len);
    try std.testing.expectEqual(PlanInstructionKind.add, plan.instructions[0].kind);
}

test "executable plan lowers initial arithmetic StableHLO ops" {
    const module_text =
        \\module {
        \\  func.func @main(%arg0: tensor<4xi8>, %arg1: tensor<4xi8>) -> tensor<4xi8> {
        \\    %0 = stablehlo.subtract %arg0, %arg1 : tensor<4xi8>
        \\    %1 = stablehlo.negate %0 : tensor<4xi8>
        \\    return %1 : tensor<4xi8>
        \\  }
        \\}
    ;
    var reader: std.Io.Reader = .fixed(module_text);
    var diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer diagnostics.deinit();

    var analysis = try analyzeProgramFromReader(std.testing.allocator, "mlir", &reader, &diagnostics.writer);
    defer analysis.deinit();

    var plan = try makeExecutablePlan(std.testing.allocator, .{}, analysis);
    defer plan.deinit();

    try std.testing.expectEqual(@as(usize, 2), plan.instructions.len);
    try std.testing.expectEqual(PlanInstructionKind.subtract, plan.instructions[0].kind);
    try std.testing.expectEqual(PlanInstructionKind.negate, plan.instructions[1].kind);
    try std.testing.expectEqual(@as(usize, 4), plan.values.len);
    try std.testing.expectEqual(ValueRole.parameter, plan.values[0].role);
    try std.testing.expectEqual(ValueRole.parameter, plan.values[1].role);
    try std.testing.expectEqual(ValueRole.instruction_result, plan.values[2].role);
    try std.testing.expectEqual(ValueRole.instruction_result, plan.values[3].role);
    try std.testing.expectEqual(@as(u32, 0), plan.instructions[0].inputs[0].index);
    try std.testing.expectEqual(@as(u32, 1), plan.instructions[0].inputs[1].index);
    try std.testing.expectEqual(@as(u32, 2), plan.instructions[0].outputs[0].index);
    try std.testing.expectEqual(@as(u32, 2), plan.instructions[1].inputs[0].index);
    try std.testing.expectEqual(@as(u32, 3), plan.instructions[1].outputs[0].index);
}

test "executable plan lowers f32 unary math StableHLO ops" {
    const module_text =
        \\module {
        \\  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
        \\    %0 = stablehlo.exponential %arg0 : tensor<4xf32>
        \\    %1 = stablehlo.tanh %0 : tensor<4xf32>
        \\    %2 = stablehlo.sqrt %1 : tensor<4xf32>
        \\    %3 = stablehlo.rsqrt %2 : tensor<4xf32>
        \\    return %3 : tensor<4xf32>
        \\  }
        \\}
    ;
    var reader: std.Io.Reader = .fixed(module_text);
    var diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer diagnostics.deinit();

    var analysis = try analyzeProgramFromReader(std.testing.allocator, "mlir", &reader, &diagnostics.writer);
    defer analysis.deinit();

    var plan = try makeExecutablePlan(std.testing.allocator, .{}, analysis);
    defer plan.deinit();

    try std.testing.expectEqual(@as(usize, 4), plan.instructions.len);
    try std.testing.expectEqual(PlanInstructionKind.exp, plan.instructions[0].kind);
    try std.testing.expectEqual(PlanInstructionKind.tanh, plan.instructions[1].kind);
    try std.testing.expectEqual(PlanInstructionKind.sqrt, plan.instructions[2].kind);
    try std.testing.expectEqual(PlanInstructionKind.rsqrt, plan.instructions[3].kind);
}

test "executable plan lowers reshape with result dimensions" {
    const module_text =
        \\module {
        \\  func.func @main(%arg0: tensor<2x2xf32>) -> tensor<4xf32> {
        \\    %0 = stablehlo.reshape %arg0 : (tensor<2x2xf32>) -> tensor<4xf32>
        \\    return %0 : tensor<4xf32>
        \\  }
        \\}
    ;
    var reader: std.Io.Reader = .fixed(module_text);
    var diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer diagnostics.deinit();

    var analysis = try analyzeProgramFromReader(std.testing.allocator, "mlir", &reader, &diagnostics.writer);
    defer analysis.deinit();

    var plan = try makeExecutablePlan(std.testing.allocator, .{}, analysis);
    defer plan.deinit();

    try std.testing.expectEqual(@as(usize, 1), plan.instructions.len);
    try std.testing.expectEqual(PlanInstructionKind.reshape, plan.instructions[0].kind);
    try std.testing.expectEqualSlices(i64, &.{4}, plan.instructions[0].dims.?);
}

test "executable plan lowers transpose with permutation and result dimensions" {
    const module_text =
        \\module {
        \\  func.func @main(%arg0: tensor<2x3xf32>) -> tensor<3x2xf32> {
        \\    %0 = stablehlo.transpose %arg0, dims = [1, 0] : (tensor<2x3xf32>) -> tensor<3x2xf32>
        \\    return %0 : tensor<3x2xf32>
        \\  }
        \\}
    ;
    var reader: std.Io.Reader = .fixed(module_text);
    var diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer diagnostics.deinit();

    var analysis = try analyzeProgramFromReader(std.testing.allocator, "mlir", &reader, &diagnostics.writer);
    defer analysis.deinit();

    var plan = try makeExecutablePlan(std.testing.allocator, .{}, analysis);
    defer plan.deinit();

    try std.testing.expectEqual(@as(usize, 1), plan.instructions.len);
    try std.testing.expectEqual(PlanInstructionKind.transpose, plan.instructions[0].kind);
    try std.testing.expectEqualSlices(i64, &.{ 3, 2 }, plan.instructions[0].dims.?);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0 }, plan.instructions[0].permutation.?);
}

test "executable plan lowers broadcast_in_dim with dimensions and result shape" {
    const module_text =
        \\module {
        \\  func.func @main(%arg0: tensor<3xf32>) -> tensor<2x3xf32> {
        \\    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<3xf32>) -> tensor<2x3xf32>
        \\    return %0 : tensor<2x3xf32>
        \\  }
        \\}
    ;
    var reader: std.Io.Reader = .fixed(module_text);
    var diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer diagnostics.deinit();

    var analysis = try analyzeProgramFromReader(std.testing.allocator, "mlir", &reader, &diagnostics.writer);
    defer analysis.deinit();

    var plan = try makeExecutablePlan(std.testing.allocator, .{}, analysis);
    defer plan.deinit();

    try std.testing.expectEqual(@as(usize, 1), plan.instructions.len);
    try std.testing.expectEqual(PlanInstructionKind.broadcast_in_dim, plan.instructions[0].kind);
    try std.testing.expectEqualSlices(i64, &.{ 2, 3 }, plan.instructions[0].dims.?);
    try std.testing.expectEqualSlices(i64, &.{1}, plan.instructions[0].broadcast_dimensions.?);
}

test "executable plan lowers slice with bounds strides and result shape" {
    const module_text =
        \\module {
        \\  func.func @main(%arg0: tensor<3x4xf32>) -> tensor<2x2xf32> {
        \\    %0 = stablehlo.slice %arg0 [1:3, 0:4:2] : (tensor<3x4xf32>) -> tensor<2x2xf32>
        \\    return %0 : tensor<2x2xf32>
        \\  }
        \\}
    ;
    var reader: std.Io.Reader = .fixed(module_text);
    var diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer diagnostics.deinit();

    var analysis = try analyzeProgramFromReader(std.testing.allocator, "mlir", &reader, &diagnostics.writer);
    defer analysis.deinit();

    var plan = try makeExecutablePlan(std.testing.allocator, .{}, analysis);
    defer plan.deinit();

    try std.testing.expectEqual(@as(usize, 1), plan.instructions.len);
    try std.testing.expectEqual(PlanInstructionKind.slice, plan.instructions[0].kind);
    try std.testing.expectEqualSlices(i64, &.{ 2, 2 }, plan.instructions[0].dims.?);
    try std.testing.expectEqualSlices(i64, &.{ 1, 0 }, plan.instructions[0].start_indices.?);
    try std.testing.expectEqualSlices(i64, &.{ 3, 4 }, plan.instructions[0].limit_indices.?);
    try std.testing.expectEqualSlices(i64, &.{ 1, 2 }, plan.instructions[0].strides.?);
}

test "executable plan lowers concatenate with dimension and result shape" {
    const module_text =
        \\module {
        \\  func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x3xf32>) -> tensor<2x5xf32> {
        \\    %0 = stablehlo.concatenate %arg0, %arg1, dim = 1 : (tensor<2x2xf32>, tensor<2x3xf32>) -> tensor<2x5xf32>
        \\    return %0 : tensor<2x5xf32>
        \\  }
        \\}
    ;
    var reader: std.Io.Reader = .fixed(module_text);
    var diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer diagnostics.deinit();

    var analysis = try analyzeProgramFromReader(std.testing.allocator, "mlir", &reader, &diagnostics.writer);
    defer analysis.deinit();

    var plan = try makeExecutablePlan(std.testing.allocator, .{}, analysis);
    defer plan.deinit();

    try std.testing.expectEqual(@as(usize, 1), plan.instructions.len);
    try std.testing.expectEqual(PlanInstructionKind.concatenate, plan.instructions[0].kind);
    try std.testing.expectEqualSlices(i64, &.{ 2, 5 }, plan.instructions[0].dims.?);
    try std.testing.expectEqual(@as(?i64, 1), plan.instructions[0].dimension);
}

test "analyze stablehlo text registers dialects and supported ops" {
    const module_text =
        \\module {
        \\  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
        \\    %0 = stablehlo.add %arg0, %arg1 : tensor<2xf32>
        \\    %1 = stablehlo.tanh %0 : tensor<2xf32>
        \\    return %1 : tensor<2xf32>
        \\  }
        \\}
    ;
    var reader: std.Io.Reader = .fixed(module_text);
    var diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer diagnostics.deinit();

    var analysis = try analyzeProgramFromReader(std.testing.allocator, "mlir", &reader, &diagnostics.writer);
    defer analysis.deinit();

    try std.testing.expectEqual(@as(usize, 2), analysis.num_parameters);
    try std.testing.expectEqual(@as(usize, 1), analysis.num_outputs);
    try std.testing.expectEqual(@as(usize, 2), analysis.ops.len);
    try std.testing.expectEqualStrings("add", analysis.ops[0].name);
    try std.testing.expectEqualStrings("f32", analysis.ops[0].dtype);
    try std.testing.expectEqual(@as(?usize, 1), analysis.ops[0].rank);
    try std.testing.expectEqualStrings("", diagnostics.writer.buffered());
}

test "analyze stablehlo text loads real shardy dialect" {
    const module_text =
        \\sdy.mesh @mesh = <["x"=2]>
        \\func.func @main(%arg0: tensor<4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> tensor<4xf32> {
        \\  %0 = stablehlo.add %arg0, %arg0 : tensor<4xf32>
        \\  return %0 : tensor<4xf32>
        \\}
    ;
    var reader: std.Io.Reader = .fixed(module_text);
    var diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer diagnostics.deinit();

    var analysis = try analyzeProgramFromReader(std.testing.allocator, "mlir", &reader, &diagnostics.writer);
    defer analysis.deinit();

    var saw_shardy = false;
    for (analysis.dialects) |dialect| {
        if (dialect == .sdy) saw_shardy = true;
    }
    try std.testing.expect(saw_shardy);
    try std.testing.expectEqual(@as(usize, 1), analysis.ops.len);
    try std.testing.expectEqualStrings("", diagnostics.writer.buffered());
}

test "analyze stablehlo text reports unsupported op with location dtype rank and sharding" {
    const module_text =
        \\sdy.mesh @mesh = <["x"=2]>
        \\func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
        \\  %0 = stablehlo.log %arg0 {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : tensor<4xf32>
        \\  return %0 : tensor<4xf32>
        \\}
    ;
    var reader: std.Io.Reader = .fixed(module_text);
    var diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer diagnostics.deinit();

    try std.testing.expectError(
        error.UnsupportedOp,
        analyzeProgramFromReader(std.testing.allocator, "mlir", &reader, &diagnostics.writer),
    );
    const text = diagnostics.writer.buffered();
    try std.testing.expect(std.mem.indexOf(u8, text, "op=log") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "dtype=f32") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "rank=1") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "sharding=sdy.sharding_per_value") != null);
}

test "analyze stablehlo bytecode is a precise unsupported encoding" {
    var reader: std.Io.Reader = .fixed("MLIR bytecode placeholder");
    var diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer diagnostics.deinit();

    try std.testing.expectError(
        error.UnsupportedProgramEncoding,
        analyzeProgramFromReader(std.testing.allocator, "stablehlo_bytecode", &reader, &diagnostics.writer),
    );
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "bytecode") != null);
}
