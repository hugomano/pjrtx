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
    inputs: []const core.ValueId = &.{},
    outputs: []const core.ValueId = &.{},
    dtype: []const u8 = "unknown",
    rank: ?usize = null,
    dims: []const i64 = &.{},
    permutation: []const i64 = &.{},
    broadcast_dimensions: []const i64 = &.{},
    start_indices: []const i64 = &.{},
    limit_indices: []const i64 = &.{},
    strides: []const i64 = &.{},
    slice_sizes: []const i64 = &.{},
    edge_padding_low: []const i64 = &.{},
    edge_padding_high: []const i64 = &.{},
    interior_padding: []const i64 = &.{},
    offset_dims: []const i64 = &.{},
    collapsed_slice_dims: []const i64 = &.{},
    operand_batching_dims: []const i64 = &.{},
    start_indices_batching_dims: []const i64 = &.{},
    start_index_map: []const i64 = &.{},
    index_vector_dim: ?i64 = null,
    dimension: ?i64 = null,
    iota_dimension: ?i64 = null,
    dimensions: []const i64 = &.{},
    tuple_index: ?i64 = null,
    lower: ?bool = null,
    custom_call_target: []const u8 = &.{},
    reduce_dimensions: []const i64 = &.{},
    lhs_batch_dimensions: []const i64 = &.{},
    rhs_batch_dimensions: []const i64 = &.{},
    lhs_contracting_dimensions: []const i64 = &.{},
    rhs_contracting_dimensions: []const i64 = &.{},
    compare_direction: ?core.CompareOp = null,
    literal: []const u8 = &.{},
    sharding: []const u8 = "unspecified",

    fn deinit(self: Operation, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.inputs);
        allocator.free(self.outputs);
        allocator.free(self.dtype);
        allocator.free(self.dims);
        allocator.free(self.permutation);
        allocator.free(self.broadcast_dimensions);
        allocator.free(self.start_indices);
        allocator.free(self.limit_indices);
        allocator.free(self.strides);
        allocator.free(self.slice_sizes);
        allocator.free(self.edge_padding_low);
        allocator.free(self.edge_padding_high);
        allocator.free(self.interior_padding);
        allocator.free(self.offset_dims);
        allocator.free(self.collapsed_slice_dims);
        allocator.free(self.operand_batching_dims);
        allocator.free(self.start_indices_batching_dims);
        allocator.free(self.start_index_map);
        allocator.free(self.dimensions);
        allocator.free(self.custom_call_target);
        allocator.free(self.reduce_dimensions);
        allocator.free(self.lhs_batch_dimensions);
        allocator.free(self.rhs_batch_dimensions);
        allocator.free(self.lhs_contracting_dimensions);
        allocator.free(self.rhs_contracting_dimensions);
        allocator.free(self.literal);
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
    values: []core.Value,
    output_ids: []const core.ValueId,
    parameter_descriptors: []core.BufferDescriptor,
    num_parameters: usize,
    num_outputs: usize,
    parameter_shardings: []ShardingMetadata,
    output_shardings: []ShardingMetadata,

    pub fn deinit(self: *ModuleAnalysis) void {
        for (self.output_shardings) |sharding| sharding.deinit(self.allocator);
        for (self.parameter_shardings) |sharding| sharding.deinit(self.allocator);
        for (self.parameter_descriptors) |descriptor| self.allocator.free(descriptor.dims);
        for (self.values) |value| self.allocator.free(value.descriptor.dims);
        for (self.ops) |op| op.deinit(self.allocator);
        self.allocator.free(self.output_ids);
        self.allocator.free(self.values);
        self.allocator.free(self.output_shardings);
        self.allocator.free(self.parameter_shardings);
        self.allocator.free(self.parameter_descriptors);
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
    UnsupportedElementType,
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
    const ok = mlir.pjrtxMlirOpPassManagerAddPipelineSucceeded(
        op_pass_manager,
        mlirStringRef(pipeline),
        writeMlirCallback,
        &ctx,
    );
    if (ctx.err != null or !ok) {
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

fn createMlirSession(writer: *std.Io.Writer) AnalyzeError!MlirSession {
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

    return session;
}

fn runMlirCanonicalization(session: *MlirSession, writer: *std.Io.Writer) AnalyzeError!void {
    const op = mlir.mlirModuleGetOperation(session.module);
    if (!mlir.mlirOperationVerify(op)) {
        try writer.writeAll("invalid StableHLO module: MLIR verifier rejected module");
        return error.InvalidStablehloModule;
    }

    const uses_shardy = operationUsesShardy(op);
    registerTransformPassesOnce();
    if (uses_shardy) registerShardyPassesOnce();

    session.pass_manager = mlir.mlirPassManagerCreate(session.context);
    if (mlir.mlirPassManagerIsNull(session.pass_manager)) {
        try writer.writeAll("invalid StableHLO module: failed to create MLIR pass manager");
        return error.InvalidStablehloModule;
    }
    mlir.mlirPassManagerEnableVerifier(session.pass_manager, true);

    if (uses_shardy) try addPipeline(session.pass_manager, "sdy-propagation-pipeline", writer);
    try addPipeline(session.pass_manager, "inline", writer);
    try addPipeline(session.pass_manager, "canonicalize", writer);
    try addPipeline(session.pass_manager, "cse", writer);
    try addPipeline(session.pass_manager, "canonicalize", writer);

    if (!mlir.pjrtxMlirPassManagerRunOnOpSucceeded(session.pass_manager, op)) {
        try writer.writeAll("invalid StableHLO module: MLIR pass pipeline failed");
        return error.InvalidStablehloModule;
    }
}

fn parseAndRunMlirWithCapi(module_text: []const u8, writer: *std.Io.Writer) AnalyzeError!MlirSession {
    var session = try createMlirSession(writer);
    errdefer session.deinit();

    session.module = mlir.mlirModuleCreateParse(session.context, mlirStringRef(module_text));
    if (mlir.mlirModuleIsNull(session.module)) {
        try writer.writeAll("invalid StableHLO module: MLIR parser rejected module");
        return error.InvalidStablehloModule;
    }

    try runMlirCanonicalization(&session, writer);
    return session;
}

fn deserializeStablehloPortableArtifactWithCapi(artifact: []const u8, writer: *std.Io.Writer) AnalyzeError!MlirSession {
    var session = try createMlirSession(writer);
    errdefer session.deinit();

    session.module = mlir.stablehloDeserializePortableArtifactNoError(mlirStringRef(artifact), session.context);
    if (mlir.mlirModuleIsNull(session.module)) {
        try writer.writeAll("invalid StableHLO module: StableHLO portable artifact deserialization failed");
        return error.InvalidStablehloModule;
    }

    try runMlirCanonicalization(&session, writer);
    return session;
}

fn writeModuleText(session: MlirSession, writer: *std.Io.Writer) AnalyzeError!void {
    var ctx: MlirStringCallbackCtx = .{ .writer = writer };
    mlir.mlirOperationPrint(mlir.mlirModuleGetOperation(session.module), writeMlirCallback, &ctx);
    if (ctx.err != null) return error.WriteFailed;
}

fn isLikelyTextMlir(source: []const u8) bool {
    for (source[0..@min(source.len, 256)]) |byte| {
        if (byte == 0) return false;
        if (byte < 0x09) return false;
        if (byte > 0x0d and byte < 0x20) return false;
    }
    return true;
}

pub const ShardingPlan = core.ShardingPlan;
pub const Value = core.Value;
pub const ValueId = core.ValueId;
pub const ValueRole = core.ValueRole;
pub const PlanInstructionKind = core.PlanInstructionKind;
pub const PlanInstruction = core.PlanInstruction;
pub const ExecutablePlan = core.ExecutablePlan;
pub const VerifyError = std.Io.Writer.Error || error{
    InvalidExecutablePlan,
    OutOfMemory,
};

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

    const parameter_descriptors = try allocator.alloc(core.BufferDescriptor, num_parameters);
    defer allocator.free(parameter_descriptors);
    @memset(parameter_descriptors, makeDescriptor(&.{}, .invalid));
    const values = try makeBootstrapValues(allocator, parameter_descriptors, &.{}, 1);
    errdefer {
        for (values) |value| allocator.free(value.descriptor.dims);
        allocator.free(values);
    }
    const instructions = try makeCopyArg0Instructions(allocator, num_parameters);
    errdefer freeInstructions(allocator, instructions);
    const output_ids = try allocator.alloc(ValueId, num_outputs);
    errdefer allocator.free(output_ids);
    for (output_ids, 0..) |*id, index| id.* = .{ .index = @intCast(num_parameters + @min(index, instructions.len - 1)) };

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
        .output_ids = output_ids,
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
    allocator.free(plan.output_ids);
    plan.output_ids = try allocator.dupe(ValueId, analysis.output_ids);
    for (plan.values) |value| allocator.free(value.descriptor.dims);
    allocator.free(plan.values);
    plan.values = &.{};
    plan.values = try cloneValues(allocator, analysis.values);
    return plan;
}

fn instructionKindFromStablehlo(name: []const u8) PlanInstructionKind {
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
    if (std.mem.eql(u8, name, "dot_general")) return .dot_general;
    if (std.mem.eql(u8, name, "reduce_sum")) return .reduce_sum;
    if (std.mem.eql(u8, name, "reduce_max")) return .reduce_max;
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

fn bufferTypeFromDtype(dtype: []const u8) core.BufferType {
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
    return .invalid;
}

fn makeDescriptor(dims: []const i64, element_type: core.BufferType) core.BufferDescriptor {
    return .{
        .element_type = element_type,
        .dims = dims,
        .device_id = -1,
        .memory_id = -1,
        .shard_index = 0,
    };
}

fn descriptorFromOperation(allocator: std.mem.Allocator, op: Operation) !core.BufferDescriptor {
    return makeDescriptor(try allocator.dupe(i64, op.dims), bufferTypeFromDtype(op.dtype));
}

fn descriptorFromType(allocator: std.mem.Allocator, ty: mlir.MlirType) !core.BufferDescriptor {
    if (mlir.mlirTypeIsNull(ty)) return makeDescriptor(try allocator.dupe(i64, &.{}), .invalid);
    const dtype = try typeDtype(allocator, ty);
    defer allocator.free(dtype);
    return makeDescriptor(try typeDims(allocator, ty), bufferTypeFromDtype(dtype));
}

fn isUnaryKind(kind: PlanInstructionKind) bool {
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
        .reverse,
        .reduce_sum,
        .reduce_max,
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

fn isBinaryKind(kind: PlanInstructionKind) bool {
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

fn makeValue(id: u32, role: ValueRole, descriptor: core.BufferDescriptor) Value {
    return .{
        .id = .{ .index = id },
        .role = role,
        .descriptor = descriptor,
    };
}

fn makeUnknownDescriptor(allocator: std.mem.Allocator) !core.BufferDescriptor {
    return makeDescriptor(try allocator.dupe(i64, &.{}), .invalid);
}

fn makeBootstrapValues(
    allocator: std.mem.Allocator,
    parameter_descriptors: []const core.BufferDescriptor,
    ops: []const Operation,
    num_instruction_results: usize,
) ![]Value {
    const num_parameters = parameter_descriptors.len;
    const value_count = num_parameters + num_instruction_results;
    const values = try allocator.alloc(Value, value_count);
    var initialized: usize = 0;
    errdefer {
        for (values[0..initialized]) |value| allocator.free(value.descriptor.dims);
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

fn cloneValues(allocator: std.mem.Allocator, source: []const Value) ![]Value {
    const values = try allocator.alloc(Value, source.len);
    var initialized: usize = 0;
    errdefer {
        for (values[0..initialized]) |value| allocator.free(value.descriptor.dims);
        allocator.free(values);
    }
    for (source, values) |src, *dst| {
        dst.* = makeValue(
            src.id.index,
            src.role,
            makeDescriptor(try allocator.dupe(i64, src.descriptor.dims), src.descriptor.element_type),
        );
        initialized += 1;
    }
    return values;
}

fn instructionInputs(allocator: std.mem.Allocator, kind: PlanInstructionKind, op: Operation) ![]const ValueId {
    if ((kind == .reduce_sum or kind == .reduce_max) and op.inputs.len >= 1) {
        return allocator.dupe(ValueId, op.inputs[0..1]);
    }
    if (op.inputs.len != 0 or kind == .constant) return allocator.dupe(ValueId, op.inputs);
    return switch (kind) {
        .constant, .iota, .partition_id => &.{},
        .select, .clamp => allocator.dupe(ValueId, &.{ .{ .index = 0 }, .{ .index = 1 }, .{ .index = 2 } }),
        .rng => allocator.dupe(ValueId, &.{ .{ .index = 0 }, .{ .index = 1 } }),
        .custom_call, .rng_bit_generator, .scatter, .tuple, .while_ => allocator.dupe(ValueId, op.inputs),
        else => if (isUnaryKind(kind))
            allocator.dupe(ValueId, &.{.{ .index = 0 }})
        else if (isBinaryKind(kind))
            allocator.dupe(ValueId, &.{ .{ .index = 0 }, .{ .index = 1 } })
        else
            &.{},
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
        if (instruction.slice_sizes) |slice_sizes| allocator.free(slice_sizes);
        if (instruction.edge_padding_low) |padding| allocator.free(padding);
        if (instruction.edge_padding_high) |padding| allocator.free(padding);
        if (instruction.interior_padding) |padding| allocator.free(padding);
        if (instruction.offset_dims) |dims| allocator.free(dims);
        if (instruction.collapsed_slice_dims) |dims| allocator.free(dims);
        if (instruction.operand_batching_dims) |dims| allocator.free(dims);
        if (instruction.start_indices_batching_dims) |dims| allocator.free(dims);
        if (instruction.start_index_map) |dims| allocator.free(dims);
        if (instruction.dimensions) |dimensions| allocator.free(dimensions);
        if (instruction.custom_call_target) |target| allocator.free(target);
        if (instruction.reduce_dimensions) |reduce_dimensions| allocator.free(reduce_dimensions);
        if (instruction.lhs_batch_dimensions) |dims| allocator.free(dims);
        if (instruction.rhs_batch_dimensions) |dims| allocator.free(dims);
        if (instruction.lhs_contracting_dimensions) |dims| allocator.free(dims);
        if (instruction.rhs_contracting_dimensions) |dims| allocator.free(dims);
        if (instruction.literal) |literal| allocator.free(literal);
    }
    allocator.free(instructions);
}

fn makePlanInstruction(
    allocator: std.mem.Allocator,
    op: Operation,
    kind: PlanInstructionKind,
) !PlanInstruction {
    const inputs = try instructionInputs(allocator, kind, op);
    errdefer if (inputs.len != 0) allocator.free(inputs);
    const outputs = try allocator.dupe(ValueId, op.outputs);
    errdefer allocator.free(outputs);
    const dims = if (kind != .constant and kind != .unsupported) try allocator.dupe(i64, op.dims) else null;
    errdefer if (dims) |owned| allocator.free(owned);
    const permutation = if (kind == .transpose) try allocator.dupe(i64, op.permutation) else null;
    errdefer if (permutation) |owned| allocator.free(owned);
    const broadcast_dimensions = if (kind == .broadcast_in_dim) try allocator.dupe(i64, op.broadcast_dimensions) else null;
    errdefer if (broadcast_dimensions) |owned| allocator.free(owned);
    const start_indices = if (kind == .slice) try allocator.dupe(i64, op.start_indices) else null;
    errdefer if (start_indices) |owned| allocator.free(owned);
    const limit_indices = if (kind == .slice) try allocator.dupe(i64, op.limit_indices) else null;
    errdefer if (limit_indices) |owned| allocator.free(owned);
    const strides = if (kind == .slice) try allocator.dupe(i64, op.strides) else null;
    errdefer if (strides) |owned| allocator.free(owned);
    const slice_sizes = if (kind == .dynamic_slice or kind == .gather) try allocator.dupe(i64, op.slice_sizes) else null;
    errdefer if (slice_sizes) |owned| allocator.free(owned);
    const edge_padding_low = if (kind == .pad) try allocator.dupe(i64, op.edge_padding_low) else null;
    errdefer if (edge_padding_low) |owned| allocator.free(owned);
    const edge_padding_high = if (kind == .pad) try allocator.dupe(i64, op.edge_padding_high) else null;
    errdefer if (edge_padding_high) |owned| allocator.free(owned);
    const interior_padding = if (kind == .pad) try allocator.dupe(i64, op.interior_padding) else null;
    errdefer if (interior_padding) |owned| allocator.free(owned);
    const offset_dims = if (kind == .gather) try allocator.dupe(i64, op.offset_dims) else null;
    errdefer if (offset_dims) |owned| allocator.free(owned);
    const collapsed_slice_dims = if (kind == .gather) try allocator.dupe(i64, op.collapsed_slice_dims) else null;
    errdefer if (collapsed_slice_dims) |owned| allocator.free(owned);
    const operand_batching_dims = if (kind == .gather) try allocator.dupe(i64, op.operand_batching_dims) else null;
    errdefer if (operand_batching_dims) |owned| allocator.free(owned);
    const start_indices_batching_dims = if (kind == .gather) try allocator.dupe(i64, op.start_indices_batching_dims) else null;
    errdefer if (start_indices_batching_dims) |owned| allocator.free(owned);
    const start_index_map = if (kind == .gather) try allocator.dupe(i64, op.start_index_map) else null;
    errdefer if (start_index_map) |owned| allocator.free(owned);
    const dimensions = if (kind == .reverse or kind == .fft) try allocator.dupe(i64, op.dimensions) else null;
    errdefer if (dimensions) |owned| allocator.free(owned);
    const custom_call_target = if (kind == .custom_call) try allocator.dupe(u8, op.custom_call_target) else null;
    errdefer if (custom_call_target) |owned| allocator.free(owned);
    const reduce_dimensions = if (kind == .reduce_sum or kind == .reduce_max) try allocator.dupe(i64, op.reduce_dimensions) else null;
    errdefer if (reduce_dimensions) |owned| allocator.free(owned);
    const lhs_batch_dimensions = if (kind == .dot_general) try allocator.dupe(i64, op.lhs_batch_dimensions) else null;
    errdefer if (lhs_batch_dimensions) |owned| allocator.free(owned);
    const rhs_batch_dimensions = if (kind == .dot_general) try allocator.dupe(i64, op.rhs_batch_dimensions) else null;
    errdefer if (rhs_batch_dimensions) |owned| allocator.free(owned);
    const lhs_contracting_dimensions = if (kind == .dot_general) try allocator.dupe(i64, op.lhs_contracting_dimensions) else null;
    errdefer if (lhs_contracting_dimensions) |owned| allocator.free(owned);
    const rhs_contracting_dimensions = if (kind == .dot_general) try allocator.dupe(i64, op.rhs_contracting_dimensions) else null;
    errdefer if (rhs_contracting_dimensions) |owned| allocator.free(owned);
    const literal = if (kind == .constant) try allocator.dupe(u8, op.literal) else null;
    errdefer if (literal) |owned| allocator.free(owned);

    return .{
        .kind = kind,
        .inputs = inputs,
        .outputs = outputs,
        .dims = dims,
        .permutation = permutation,
        .broadcast_dimensions = broadcast_dimensions,
        .start_indices = start_indices,
        .limit_indices = limit_indices,
        .strides = strides,
        .slice_sizes = slice_sizes,
        .edge_padding_low = edge_padding_low,
        .edge_padding_high = edge_padding_high,
        .interior_padding = interior_padding,
        .offset_dims = offset_dims,
        .collapsed_slice_dims = collapsed_slice_dims,
        .operand_batching_dims = operand_batching_dims,
        .start_indices_batching_dims = start_indices_batching_dims,
        .start_index_map = start_index_map,
        .index_vector_dim = if (kind == .gather) op.index_vector_dim else null,
        .dimension = if (kind == .concatenate or kind == .sort) op.dimension else null,
        .iota_dimension = if (kind == .iota) op.iota_dimension else null,
        .dimensions = dimensions,
        .tuple_index = if (kind == .get_tuple_element) op.tuple_index else null,
        .lower = if (kind == .cholesky) op.lower else null,
        .custom_call_target = custom_call_target,
        .reduce_dimensions = reduce_dimensions,
        .lhs_batch_dimensions = lhs_batch_dimensions,
        .rhs_batch_dimensions = rhs_batch_dimensions,
        .lhs_contracting_dimensions = lhs_contracting_dimensions,
        .rhs_contracting_dimensions = rhs_contracting_dimensions,
        .compare_direction = if (kind == .compare or kind == .sort) op.compare_direction else null,
        .literal = literal,
    };
}

fn lowerAnalysisOpsToPlan(allocator: std.mem.Allocator, ops: []const Operation, num_parameters: usize) ![]PlanInstruction {
    if (ops.len == 0) return makeCopyArg0Instructions(allocator, num_parameters);
    const plan_instructions = try allocator.alloc(PlanInstruction, ops.len);
    errdefer {
        freeInstructions(allocator, plan_instructions);
    }
    @memset(plan_instructions, .{ .kind = .unsupported });
    for (ops, plan_instructions, 0..) |op, *plan_instruction, index| {
        _ = index;
        const kind = instructionKindFromStablehlo(op.name);
        plan_instruction.* = try makePlanInstruction(allocator, op, kind);
    }
    return plan_instructions;
}

fn expectedInputCount(kind: PlanInstructionKind) ?usize {
    return switch (kind) {
        .constant, .iota, .partition_id => 0,
        .dynamic_slice, .dynamic_update_slice, .concatenate, .custom_call, .rng_bit_generator, .scatter, .tuple, .while_ => null,
        .select, .clamp => 3,
        .rng => null,
        .unsupported => null,
        else => if (isUnaryKind(kind)) 1 else if (isBinaryKind(kind)) 2 else null,
    };
}

fn failPlanVerification(
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

fn valueInPlan(plan: ExecutablePlan, id: ValueId) bool {
    const index: usize = id.index;
    return index < plan.values.len and plan.values[index].id.index == id.index;
}

fn descriptorKnown(descriptor: core.BufferDescriptor) bool {
    return descriptor.element_type != .invalid;
}

fn sameShape(lhs: core.BufferDescriptor, rhs: core.BufferDescriptor) bool {
    return std.mem.eql(i64, lhs.dims, rhs.dims);
}

fn sameTypeAndShape(lhs: core.BufferDescriptor, rhs: core.BufferDescriptor) bool {
    if (!descriptorKnown(lhs) or !descriptorKnown(rhs)) return true;
    return lhs.element_type == rhs.element_type and sameShape(lhs, rhs);
}

fn instructionOutputDims(instruction: PlanInstruction) ?[]const i64 {
    return switch (instruction.kind) {
        .constant, .copy_arg0, .unsupported => null,
        else => instruction.dims,
    };
}

fn verifyInstructionDescriptors(
    plan: ExecutablePlan,
    instruction: PlanInstruction,
    instruction_index: usize,
    writer: *std.Io.Writer,
) VerifyError!void {
    const pass_name = "pjrtx-plan-verify";
    const output = plan.values[instruction.outputs[0].index].descriptor;
    for (instruction.inputs) |input_id| {
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
        .copy_arg0, .negate, .exp, .expm1, .tanh, .sqrt, .rsqrt, .abs, .cbrt, .ceil, .floor, .log, .log1p, .logistic, .sine, .cosine, .not_, .sign, .round_nearest_afz, .round_nearest_even, .popcnt, .count_leading_zeros, .reduce_precision => {
            if (!sameTypeAndShape(plan.values[instruction.inputs[0].index].descriptor, output)) {
                return failPlanVerification(writer, pass_name, instruction_index, instruction.outputs[0], "unary instruction must preserve input dtype and shape", "shape-type");
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
            if (descriptorKnown(input) and descriptorKnown(output) and core.denseByteSize(input.element_type, input.dims) != core.denseByteSize(output.element_type, output.dims)) {
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
            if (descriptorKnown(input) and descriptorKnown(output) and core.denseByteSize(input.element_type, input.dims) != core.denseByteSize(output.element_type, output.dims)) {
                return failPlanVerification(writer, pass_name, instruction_index, instruction.outputs[0], "reshape must preserve dense byte size", "shape-type");
            }
        },
        .transpose, .broadcast_in_dim, .slice, .dynamic_slice, .dynamic_update_slice, .pad, .reverse, .gather, .sort => {
            const input = plan.values[instruction.inputs[0].index].descriptor;
            if (descriptorKnown(input) and descriptorKnown(output) and input.element_type != output.element_type) {
                return failPlanVerification(writer, pass_name, instruction_index, instruction.outputs[0], "view instruction must preserve dtype", "shape-type");
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
            if (descriptorKnown(lhs) and descriptorKnown(rhs) and (lhs.element_type != .f32 or rhs.element_type != .f32 or output.element_type != .f32)) {
                return failPlanVerification(writer, pass_name, instruction_index, instruction.outputs[0], "dot_general bootstrap lowering supports f32 tensors only", "dot-general");
            }
        },
        .reduce_sum, .reduce_max => {
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
        .rng => {},
        .rng_bit_generator => {
            if (instruction.outputs.len != 2) {
                return failPlanVerification(writer, pass_name, instruction_index, null, "rng_bit_generator must produce state and random bits", "random");
            }
        },
        .complex, .real, .imag, .fft, .convolution, .scatter, .custom_call, .get_tuple_element, .tuple, .while_ => {},
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
    }

    if (plan.parameter_shardings.len != parameter_count) {
        return failPlanVerification(writer, pass_name, null, null, "parameter sharding count must match parameter value count", "sharding");
    }
    if (plan.instructions.len == 0) {
        return failPlanVerification(writer, pass_name, null, null, "plan must contain at least one instruction", "instruction-graph");
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
        "maximum",
        "minimum",
        "power",
        "atan2",
        "remainder",
        "and",
        "or",
        "xor",
        "shift_left",
        "shift_right_arithmetic",
        "shift_right_logical",
        "negate",
        "abs",
        "cbrt",
        "ceil",
        "exponential",
        "exponential_minus_one",
        "floor",
        "log",
        "log_plus_one",
        "logistic",
        "sine",
        "cosine",
        "not",
        "sign",
        "is_finite",
        "round_nearest_afz",
        "round_nearest_even",
        "popcnt",
        "count_leading_zeros",
        "tanh",
        "sqrt",
        "rsqrt",
        "convert",
        "bitcast_convert",
        "compare",
        "select",
        "clamp",
        "reshape",
        "broadcast_in_dim",
        "transpose",
        "slice",
        "dynamic_slice",
        "dynamic_update_slice",
        "pad",
        "reverse",
        "concatenate",
        "iota",
        "gather",
        "sort",
        "reduce",
        "dot_general",
        "cholesky",
        "complex",
        "convolution",
        "custom_call",
        "fft",
        "get_tuple_element",
        "imag",
        "partition_id",
        "real",
        "reduce_precision",
        "rng",
        "rng_bit_generator",
        "scatter",
        "triangular_solve",
        "tuple",
        "while",
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

fn boolAttribute(attr: mlir.MlirAttribute) ?bool {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsABool(attr)) return null;
    return mlir.mlirBoolAttrGetValue(attr);
}

fn stringAttribute(allocator: std.mem.Allocator, attr: mlir.MlirAttribute) ![]const u8 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsAString(attr)) return allocator.dupe(u8, &.{});
    return allocator.dupe(u8, mlirStringSlice(mlir.mlirStringAttrGetValue(attr)));
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
    parameter_descriptors: std.ArrayList(core.BufferDescriptor) = .empty,
    parameter_shardings: std.ArrayList(ShardingMetadata) = .empty,
    output_shardings: std.ArrayList(ShardingMetadata) = .empty,
    values: std.ArrayList(Value) = .empty,
    value_map: std.ArrayList(ValueMapEntry) = .empty,
    output_ids: std.ArrayList(ValueId) = .empty,
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
        for (self.parameter_descriptors.items) |descriptor| self.allocator.free(descriptor.dims);
        for (self.values.items) |value| self.allocator.free(value.descriptor.dims);
        for (self.ops.items) |op| op.deinit(self.allocator);
        self.output_ids.deinit(self.allocator);
        self.value_map.deinit(self.allocator);
        self.values.deinit(self.allocator);
        self.output_shardings.deinit(self.allocator);
        self.parameter_shardings.deinit(self.allocator);
        self.parameter_descriptors.deinit(self.allocator);
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

    fn ensureParameterDescriptors(self: *CapiAnalysisBuilder, count: usize) !void {
        while (self.parameter_descriptors.items.len < count) {
            try self.parameter_descriptors.append(self.allocator, try makeUnknownDescriptor(self.allocator));
        }
    }

    fn replaceParameterDescriptor(self: *CapiAnalysisBuilder, index: usize, descriptor: core.BufferDescriptor) void {
        self.allocator.free(self.parameter_descriptors.items[index].dims);
        self.parameter_descriptors.items[index] = descriptor;
    }

    fn registerValue(self: *CapiAnalysisBuilder, value: mlir.MlirValue, role: ValueRole, descriptor: core.BufferDescriptor) !ValueId {
        if (self.lookupValue(value)) |existing| {
            self.allocator.free(descriptor.dims);
            return existing;
        }
        const id: ValueId = .{ .index = @intCast(self.values.items.len) };
        try self.values.append(self.allocator, makeValue(id.index, role, descriptor));
        errdefer _ = self.values.pop();
        try self.value_map.append(self.allocator, .{ .mlir_value = value, .id = id });
        return id;
    }

    fn lookupValue(self: *CapiAnalysisBuilder, value: mlir.MlirValue) ?ValueId {
        for (self.value_map.items) |entry| {
            if (mlir.mlirValueEqual(entry.mlir_value, value)) return entry.id;
        }
        return null;
    }
};

const ValueMapEntry = struct {
    mlir_value: mlir.MlirValue,
    id: ValueId,
};

fn operationHasAttributeNamed(op: mlir.MlirOperation, name: []const u8) bool {
    return !mlir.mlirAttributeIsNull(getOperationAttribute(op, name));
}

fn valueIdsForOperands(builder: *CapiAnalysisBuilder, op: mlir.MlirOperation) ![]const ValueId {
    return valueIdsForOperandsLimit(builder, op, @intCast(mlir.mlirOperationGetNumOperands(op)));
}

fn valueIdsForOperandsLimit(builder: *CapiAnalysisBuilder, op: mlir.MlirOperation, count: usize) ![]const ValueId {
    const ids = try builder.allocator.alloc(ValueId, count);
    errdefer builder.allocator.free(ids);
    var index: isize = 0;
    while (index < @as(isize, @intCast(count))) : (index += 1) {
        const operand = mlir.mlirOperationGetOperand(op, index);
        ids[@intCast(index)] = builder.lookupValue(operand) orelse blk: {
            var owner_name: []const u8 = "<none>";
            if (mlir.mlirValueIsAOpResult(operand)) {
                const owner = mlir.mlirOpResultGetOwner(operand);
                owner_name = operationName(owner);
                if (std.mem.eql(u8, owner_name, "stablehlo.constant") or std.mem.eql(u8, owner_name, "sdy.constant")) {
                    try analyzeStablehloOperationFromCapi(builder, owner, "stablehlo.constant");
                    if (builder.lookupValue(operand)) |id| break :blk id;
                }
            }
            const loc = mlirLocationLineColumn(mlir.mlirOperationGetLocation(op));
            try builder.diagnostic_writer.print(
                "invalid StableHLO module: loc={d}:{d} op={s} operand={d} owner={s} detail=\"operand does not reference a previously registered top-level value\" feature=value-graph",
                .{ loc.line, loc.column, operationName(op), index, owner_name },
            );
            return error.InvalidStablehloModule;
        };
    }
    return ids;
}

fn registerResultValues(builder: *CapiAnalysisBuilder, op: mlir.MlirOperation, role: ValueRole) ![]const ValueId {
    const count: usize = @intCast(mlir.mlirOperationGetNumResults(op));
    const ids = try builder.allocator.alloc(ValueId, count);
    errdefer builder.allocator.free(ids);
    var index: isize = 0;
    while (index < @as(isize, @intCast(count))) : (index += 1) {
        const result = mlir.mlirOperationGetResult(op, index);
        ids[@intCast(index)] = try builder.registerValue(result, role, try descriptorFromType(builder.allocator, mlir.mlirValueGetType(result)));
    }
    return ids;
}

fn denseLiteralBytes(allocator: std.mem.Allocator, attr: mlir.MlirAttribute, element_type: core.BufferType, dims: []const i64) ![]const u8 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsADenseElements(attr)) return allocator.dupe(u8, &.{});
    const byte_size = core.denseByteSize(element_type, dims);
    const bytes = try allocator.alloc(u8, byte_size);
    errdefer allocator.free(bytes);
    const element_count: usize = @intCast(mlir.mlirElementsAttrGetNumElements(attr));
    if (byte_size == 0 or element_count == 0) return bytes;
    switch (element_type) {
        .pred => {
            for (0..element_count) |i| bytes[i] = if (mlir.mlirDenseElementsAttrGetBoolValue(attr, @intCast(i))) 1 else 0;
        },
        .u8 => {
            for (0..element_count) |i| bytes[i] = mlir.mlirDenseElementsAttrGetUInt8Value(attr, @intCast(i));
        },
        .s8 => {
            for (0..element_count) |i| bytes[i] = @bitCast(mlir.mlirDenseElementsAttrGetInt8Value(attr, @intCast(i)));
        },
        .f32 => {
            for (0..element_count) |i| {
                const value = mlir.mlirDenseElementsAttrGetFloatValue(attr, @intCast(i));
                std.mem.writeInt(u32, bytes[i * 4 ..][0..4], @bitCast(value), .little);
            }
        },
        .s32 => {
            for (0..element_count) |i| {
                const value: i32 = mlir.mlirDenseElementsAttrGetInt32Value(attr, @intCast(i));
                std.mem.writeInt(u32, bytes[i * 4 ..][0..4], @bitCast(value), .little);
            }
        },
        else => return error.UnsupportedElementType,
    }
    return bytes;
}

fn stablehloDotDims(allocator: std.mem.Allocator, attr: mlir.MlirAttribute, comptime which: []const u8) ![]const i64 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.stablehloAttributeIsADotDimensionNumbers(attr)) return allocator.dupe(i64, &.{});
    const count: isize = if (std.mem.eql(u8, which, "lhs_batch"))
        mlir.stablehloDotDimensionNumbersGetLhsBatchingDimensionsSize(attr)
    else if (std.mem.eql(u8, which, "rhs_batch"))
        mlir.stablehloDotDimensionNumbersGetRhsBatchingDimensionsSize(attr)
    else if (std.mem.eql(u8, which, "lhs_contract"))
        mlir.stablehloDotDimensionNumbersGetLhsContractingDimensionsSize(attr)
    else
        mlir.stablehloDotDimensionNumbersGetRhsContractingDimensionsSize(attr);
    const values = try allocator.alloc(i64, @intCast(count));
    var index: isize = 0;
    while (index < count) : (index += 1) {
        values[@intCast(index)] = if (std.mem.eql(u8, which, "lhs_batch"))
            mlir.stablehloDotDimensionNumbersGetLhsBatchingDimensionsElem(attr, index)
        else if (std.mem.eql(u8, which, "rhs_batch"))
            mlir.stablehloDotDimensionNumbersGetRhsBatchingDimensionsElem(attr, index)
        else if (std.mem.eql(u8, which, "lhs_contract"))
            mlir.stablehloDotDimensionNumbersGetLhsContractingDimensionsElem(attr, index)
        else
            mlir.stablehloDotDimensionNumbersGetRhsContractingDimensionsElem(attr, index);
    }
    return values;
}

fn stablehloGatherDims(allocator: std.mem.Allocator, attr: mlir.MlirAttribute, comptime which: []const u8) ![]const i64 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.stablehloAttributeIsAGatherDimensionNumbers(attr)) return allocator.dupe(i64, &.{});
    const count = if (std.mem.eql(u8, which, "offset"))
        mlir.stablehloGatherDimensionNumbersGetOffsetDimsSize(attr)
    else if (std.mem.eql(u8, which, "collapsed"))
        mlir.stablehloGatherDimensionNumbersGetCollapsedSliceDimsSize(attr)
    else if (std.mem.eql(u8, which, "operand_batching"))
        mlir.stablehloGatherDimensionNumbersGetOperandBatchingDimsSize(attr)
    else if (std.mem.eql(u8, which, "start_indices_batching"))
        mlir.stablehloGatherDimensionNumbersGetStartIndicesBatchingDimsSize(attr)
    else
        mlir.stablehloGatherDimensionNumbersGetStartIndexMapSize(attr);
    const values = try allocator.alloc(i64, @intCast(count));
    var index: isize = 0;
    while (index < count) : (index += 1) {
        values[@intCast(index)] = if (std.mem.eql(u8, which, "offset"))
            mlir.stablehloGatherDimensionNumbersGetOffsetDimsElem(attr, index)
        else if (std.mem.eql(u8, which, "collapsed"))
            mlir.stablehloGatherDimensionNumbersGetCollapsedSliceDimsElem(attr, index)
        else if (std.mem.eql(u8, which, "operand_batching"))
            mlir.stablehloGatherDimensionNumbersGetOperandBatchingDimsElem(attr, index)
        else if (std.mem.eql(u8, which, "start_indices_batching"))
            mlir.stablehloGatherDimensionNumbersGetStartIndicesBatchingDimsElem(attr, index)
        else
            mlir.stablehloGatherDimensionNumbersGetStartIndexMapElem(attr, index);
    }
    return values;
}

fn stablehloGatherIndexVectorDim(attr: mlir.MlirAttribute) ?i64 {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.stablehloAttributeIsAGatherDimensionNumbers(attr)) return null;
    return mlir.stablehloGatherDimensionNumbersGetIndexVectorDim(attr);
}

fn compareDirectionFromAttr(attr: mlir.MlirAttribute) ?core.CompareOp {
    if (mlir.mlirAttributeIsNull(attr) or !mlir.stablehloAttributeIsAComparisonDirectionAttr(attr)) return null;
    const text = mlirStringSlice(mlir.stablehloComparisonDirectionAttrGetValue(attr));
    if (std.mem.eql(u8, text, "EQ")) return .eq;
    if (std.mem.eql(u8, text, "NE")) return .ne;
    if (std.mem.eql(u8, text, "GE")) return .ge;
    if (std.mem.eql(u8, text, "GT")) return .gt;
    if (std.mem.eql(u8, text, "LE")) return .le;
    if (std.mem.eql(u8, text, "LT")) return .lt;
    return null;
}

fn reduceKindFromRegion(op: mlir.MlirOperation) []const u8 {
    const n_regions = mlir.mlirOperationGetNumRegions(op);
    var region_index: isize = 0;
    while (region_index < n_regions) : (region_index += 1) {
        var block = mlir.mlirRegionGetFirstBlock(mlir.mlirOperationGetRegion(op, region_index));
        while (!mlir.mlirBlockIsNull(block)) : (block = mlir.mlirBlockGetNextInRegion(block)) {
            var child = mlir.mlirBlockGetFirstOperation(block);
            while (!mlir.mlirOperationIsNull(child)) : (child = mlir.mlirOperationGetNextInBlock(child)) {
                const name = operationName(child);
                if (std.mem.eql(u8, name, "stablehlo.add")) return "reduce_sum";
                if (std.mem.eql(u8, name, "stablehlo.maximum")) return "reduce_max";
            }
        }
    }
    return "reduce";
}

fn compareDirectionFromSortRegion(op: mlir.MlirOperation) ?core.CompareOp {
    var last_compare: ?core.CompareOp = null;
    const n_regions = mlir.mlirOperationGetNumRegions(op);
    var region_index: isize = 0;
    while (region_index < n_regions) : (region_index += 1) {
        var block = mlir.mlirRegionGetFirstBlock(mlir.mlirOperationGetRegion(op, region_index));
        while (!mlir.mlirBlockIsNull(block)) : (block = mlir.mlirBlockGetNextInRegion(block)) {
            var child = mlir.mlirBlockGetFirstOperation(block);
            while (!mlir.mlirOperationIsNull(child)) : (child = mlir.mlirOperationGetNextInBlock(child)) {
                if (std.mem.eql(u8, operationName(child), "stablehlo.compare")) {
                    last_compare = compareDirectionFromAttr(getOperationAttribute(child, "comparison_direction"));
                }
            }
        }
    }
    return last_compare;
}

fn analyzeStablehloOperationFromCapi(builder: *CapiAnalysisBuilder, op: mlir.MlirOperation, op_name: []const u8) AnalyzeError!void {
    builder.saw_program_body = true;
    try addDialect(&builder.dialects, builder.allocator, .stablehlo);

    const raw_short_name = op_name["stablehlo.".len..];
    if (std.mem.eql(u8, raw_short_name, "return")) return;
    const short_name = if (std.mem.eql(u8, raw_short_name, "reduce")) reduceKindFromRegion(op) else raw_short_name;
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
    const slice_sizes = if (std.mem.eql(u8, short_name, "dynamic_slice") or std.mem.eql(u8, short_name, "gather"))
        try intListAttribute(builder.allocator, getOperationAttribute(op, "slice_sizes"))
    else
        try builder.allocator.dupe(i64, &.{});
    var owns_slice_sizes = true;
    errdefer if (owns_slice_sizes) builder.allocator.free(slice_sizes);
    const edge_padding_low = if (std.mem.eql(u8, short_name, "pad"))
        try intListAttribute(builder.allocator, getOperationAttribute(op, "edge_padding_low"))
    else
        try builder.allocator.dupe(i64, &.{});
    var owns_edge_padding_low = true;
    errdefer if (owns_edge_padding_low) builder.allocator.free(edge_padding_low);
    const edge_padding_high = if (std.mem.eql(u8, short_name, "pad"))
        try intListAttribute(builder.allocator, getOperationAttribute(op, "edge_padding_high"))
    else
        try builder.allocator.dupe(i64, &.{});
    var owns_edge_padding_high = true;
    errdefer if (owns_edge_padding_high) builder.allocator.free(edge_padding_high);
    const interior_padding = if (std.mem.eql(u8, short_name, "pad"))
        try intListAttribute(builder.allocator, getOperationAttribute(op, "interior_padding"))
    else
        try builder.allocator.dupe(i64, &.{});
    var owns_interior_padding = true;
    errdefer if (owns_interior_padding) builder.allocator.free(interior_padding);
    const gather_dimensions = getOperationAttribute(op, "dimension_numbers");
    const offset_dims = if (std.mem.eql(u8, short_name, "gather")) try stablehloGatherDims(builder.allocator, gather_dimensions, "offset") else try builder.allocator.dupe(i64, &.{});
    var owns_offset_dims = true;
    errdefer if (owns_offset_dims) builder.allocator.free(offset_dims);
    const collapsed_slice_dims = if (std.mem.eql(u8, short_name, "gather")) try stablehloGatherDims(builder.allocator, gather_dimensions, "collapsed") else try builder.allocator.dupe(i64, &.{});
    var owns_collapsed_slice_dims = true;
    errdefer if (owns_collapsed_slice_dims) builder.allocator.free(collapsed_slice_dims);
    const operand_batching_dims = if (std.mem.eql(u8, short_name, "gather")) try stablehloGatherDims(builder.allocator, gather_dimensions, "operand_batching") else try builder.allocator.dupe(i64, &.{});
    var owns_operand_batching_dims = true;
    errdefer if (owns_operand_batching_dims) builder.allocator.free(operand_batching_dims);
    const start_indices_batching_dims = if (std.mem.eql(u8, short_name, "gather")) try stablehloGatherDims(builder.allocator, gather_dimensions, "start_indices_batching") else try builder.allocator.dupe(i64, &.{});
    var owns_start_indices_batching_dims = true;
    errdefer if (owns_start_indices_batching_dims) builder.allocator.free(start_indices_batching_dims);
    const start_index_map = if (std.mem.eql(u8, short_name, "gather")) try stablehloGatherDims(builder.allocator, gather_dimensions, "start_index_map") else try builder.allocator.dupe(i64, &.{});
    var owns_start_index_map = true;
    errdefer if (owns_start_index_map) builder.allocator.free(start_index_map);
    const index_vector_dim = if (std.mem.eql(u8, short_name, "gather")) stablehloGatherIndexVectorDim(gather_dimensions) else null;
    const dimension = if (std.mem.eql(u8, short_name, "concatenate") or std.mem.eql(u8, short_name, "sort"))
        intAttribute(getOperationAttribute(op, "dimension"))
    else
        null;
    const iota_dimension = if (std.mem.eql(u8, short_name, "iota"))
        intAttribute(getOperationAttribute(op, "iota_dimension"))
    else
        null;
    const dimensions = if (std.mem.eql(u8, short_name, "reverse"))
        try intListAttribute(builder.allocator, getOperationAttribute(op, "dimensions"))
    else if (std.mem.eql(u8, short_name, "fft"))
        try intListAttribute(builder.allocator, getOperationAttribute(op, "fft_length"))
    else
        try builder.allocator.dupe(i64, &.{});
    var owns_dimensions = true;
    errdefer if (owns_dimensions) builder.allocator.free(dimensions);
    const tuple_index = if (std.mem.eql(u8, short_name, "get_tuple_element"))
        intAttribute(getOperationAttribute(op, "index"))
    else
        null;
    const lower = if (std.mem.eql(u8, short_name, "cholesky"))
        boolAttribute(getOperationAttribute(op, "lower"))
    else
        null;
    const custom_call_target = if (std.mem.eql(u8, short_name, "custom_call"))
        try stringAttribute(builder.allocator, getOperationAttribute(op, "call_target_name"))
    else
        try builder.allocator.dupe(u8, &.{});
    var owns_custom_call_target = true;
    errdefer if (owns_custom_call_target) builder.allocator.free(custom_call_target);
    const reduce_dimensions = if (std.mem.startsWith(u8, short_name, "reduce_"))
        try intListAttribute(builder.allocator, getOperationAttribute(op, "dimensions"))
    else
        try builder.allocator.dupe(i64, &.{});
    var owns_reduce_dimensions = true;
    errdefer if (owns_reduce_dimensions) builder.allocator.free(reduce_dimensions);
    const dot_dimensions = getOperationAttribute(op, "dot_dimension_numbers");
    const lhs_batch_dimensions = if (std.mem.eql(u8, short_name, "dot_general")) try stablehloDotDims(builder.allocator, dot_dimensions, "lhs_batch") else try builder.allocator.dupe(i64, &.{});
    var owns_lhs_batch_dimensions = true;
    errdefer if (owns_lhs_batch_dimensions) builder.allocator.free(lhs_batch_dimensions);
    const rhs_batch_dimensions = if (std.mem.eql(u8, short_name, "dot_general")) try stablehloDotDims(builder.allocator, dot_dimensions, "rhs_batch") else try builder.allocator.dupe(i64, &.{});
    var owns_rhs_batch_dimensions = true;
    errdefer if (owns_rhs_batch_dimensions) builder.allocator.free(rhs_batch_dimensions);
    const lhs_contracting_dimensions = if (std.mem.eql(u8, short_name, "dot_general")) try stablehloDotDims(builder.allocator, dot_dimensions, "lhs_contract") else try builder.allocator.dupe(i64, &.{});
    var owns_lhs_contracting_dimensions = true;
    errdefer if (owns_lhs_contracting_dimensions) builder.allocator.free(lhs_contracting_dimensions);
    const rhs_contracting_dimensions = if (std.mem.eql(u8, short_name, "dot_general")) try stablehloDotDims(builder.allocator, dot_dimensions, "rhs_contract") else try builder.allocator.dupe(i64, &.{});
    var owns_rhs_contracting_dimensions = true;
    errdefer if (owns_rhs_contracting_dimensions) builder.allocator.free(rhs_contracting_dimensions);
    const compare_direction = if (std.mem.eql(u8, short_name, "compare"))
        compareDirectionFromAttr(getOperationAttribute(op, "comparison_direction"))
    else if (std.mem.eql(u8, short_name, "sort"))
        compareDirectionFromSortRegion(op)
    else
        null;
    const input_count = if (std.mem.startsWith(u8, short_name, "reduce_")) @as(usize, 1) else @as(usize, @intCast(mlir.mlirOperationGetNumOperands(op)));
    const inputs = try valueIdsForOperandsLimit(builder, op, input_count);
    var owns_inputs = true;
    errdefer if (owns_inputs) builder.allocator.free(inputs);
    const value_role: ValueRole = if (std.mem.eql(u8, short_name, "constant")) .constant else .instruction_result;
    const outputs = try registerResultValues(builder, op, value_role);
    var owns_outputs = true;
    errdefer if (owns_outputs) builder.allocator.free(outputs);
    const element_type = bufferTypeFromDtype(dtype);
    const literal = if (std.mem.eql(u8, short_name, "constant"))
        try denseLiteralBytes(builder.allocator, getOperationAttribute(op, "value"), element_type, dims)
    else
        try builder.allocator.dupe(u8, &.{});
    var owns_literal = true;
    errdefer if (owns_literal) builder.allocator.free(literal);
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
        .inputs = inputs,
        .outputs = outputs,
        .dtype = dtype,
        .rank = if (mlir.mlirTypeIsNull(ty)) null else typeRank(ty),
        .dims = dims,
        .permutation = permutation,
        .broadcast_dimensions = broadcast_dimensions,
        .start_indices = start_indices,
        .limit_indices = limit_indices,
        .strides = strides,
        .slice_sizes = slice_sizes,
        .edge_padding_low = edge_padding_low,
        .edge_padding_high = edge_padding_high,
        .interior_padding = interior_padding,
        .offset_dims = offset_dims,
        .collapsed_slice_dims = collapsed_slice_dims,
        .operand_batching_dims = operand_batching_dims,
        .start_indices_batching_dims = start_indices_batching_dims,
        .start_index_map = start_index_map,
        .index_vector_dim = index_vector_dim,
        .dimension = dimension,
        .iota_dimension = iota_dimension,
        .dimensions = dimensions,
        .tuple_index = tuple_index,
        .lower = lower,
        .custom_call_target = custom_call_target,
        .reduce_dimensions = reduce_dimensions,
        .lhs_batch_dimensions = lhs_batch_dimensions,
        .rhs_batch_dimensions = rhs_batch_dimensions,
        .lhs_contracting_dimensions = lhs_contracting_dimensions,
        .rhs_contracting_dimensions = rhs_contracting_dimensions,
        .compare_direction = compare_direction,
        .literal = literal,
        .sharding = sharding,
    };
    if (!stablehloOpSupported(raw_short_name) or std.mem.eql(u8, short_name, "reduce")) {
        try writeOpDiagnostic(builder.diagnostic_writer, "unsupported op", analyzed, "stablehlo-op");
        analyzed.deinit(builder.allocator);
        owns_inputs = false;
        owns_outputs = false;
        owns_dtype = false;
        owns_dims = false;
        owns_permutation = false;
        owns_broadcast_dimensions = false;
        owns_start_indices = false;
        owns_limit_indices = false;
        owns_strides = false;
        owns_slice_sizes = false;
        owns_edge_padding_low = false;
        owns_edge_padding_high = false;
        owns_interior_padding = false;
        owns_offset_dims = false;
        owns_collapsed_slice_dims = false;
        owns_operand_batching_dims = false;
        owns_start_indices_batching_dims = false;
        owns_start_index_map = false;
        owns_dimensions = false;
        owns_custom_call_target = false;
        owns_reduce_dimensions = false;
        owns_lhs_batch_dimensions = false;
        owns_rhs_batch_dimensions = false;
        owns_lhs_contracting_dimensions = false;
        owns_rhs_contracting_dimensions = false;
        owns_literal = false;
        owns_sharding = false;
        owns_name = false;
        return error.UnsupportedOp;
    }
    try builder.ops.append(builder.allocator, analyzed);
    owns_inputs = false;
    owns_outputs = false;
    owns_dtype = false;
    owns_dims = false;
    owns_permutation = false;
    owns_broadcast_dimensions = false;
    owns_start_indices = false;
    owns_limit_indices = false;
    owns_strides = false;
    owns_slice_sizes = false;
    owns_edge_padding_low = false;
    owns_edge_padding_high = false;
    owns_interior_padding = false;
    owns_offset_dims = false;
    owns_collapsed_slice_dims = false;
    owns_operand_batching_dims = false;
    owns_start_indices_batching_dims = false;
    owns_start_index_map = false;
    owns_dimensions = false;
    owns_custom_call_target = false;
    owns_reduce_dimensions = false;
    owns_lhs_batch_dimensions = false;
    owns_rhs_batch_dimensions = false;
    owns_lhs_contracting_dimensions = false;
    owns_rhs_contracting_dimensions = false;
    owns_literal = false;
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
    try builder.ensureParameterDescriptors(builder.num_parameters);
    try builder.ensureShardings(&builder.parameter_shardings, builder.num_parameters);
    try builder.ensureShardings(&builder.output_shardings, builder.num_outputs);

    const function_type_attr_for_params = getOperationAttribute(op, "function_type");
    if (!mlir.mlirAttributeIsNull(function_type_attr_for_params) and mlir.mlirAttributeIsAType(function_type_attr_for_params)) {
        const function_type = mlir.mlirTypeAttrGetValue(function_type_attr_for_params);
        if (mlir.mlirTypeIsAFunction(function_type)) {
            const typed_inputs = @min(num_inputs, @as(usize, @intCast(mlir.mlirFunctionTypeGetNumInputs(function_type))));
            var input_index: usize = 0;
            while (input_index < typed_inputs) : (input_index += 1) {
                builder.replaceParameterDescriptor(input_index, try descriptorFromType(builder.allocator, mlir.mlirFunctionTypeGetInput(function_type, @intCast(input_index))));
            }
        }
    }
    if (mlir.mlirOperationGetNumRegions(op) > 0) {
        const entry = mlir.mlirRegionGetFirstBlock(mlir.mlirOperationGetRegion(op, 0));
        if (!mlir.mlirBlockIsNull(entry)) {
            const block_args = @min(num_inputs, @as(usize, @intCast(mlir.mlirBlockGetNumArguments(entry))));
            var arg_index_for_type: usize = 0;
            while (arg_index_for_type < block_args) : (arg_index_for_type += 1) {
                builder.replaceParameterDescriptor(arg_index_for_type, try descriptorFromType(builder.allocator, mlir.mlirValueGetType(mlir.mlirBlockGetArgument(entry, @intCast(arg_index_for_type)))));
            }
            var arg_index_for_value: usize = 0;
            while (arg_index_for_value < block_args) : (arg_index_for_value += 1) {
                const block_arg = mlir.mlirBlockGetArgument(entry, @intCast(arg_index_for_value));
                const descriptor = builder.parameter_descriptors.items[arg_index_for_value];
                _ = try builder.registerValue(
                    block_arg,
                    .parameter,
                    makeDescriptor(try builder.allocator.dupe(i64, descriptor.dims), descriptor.element_type),
                );
            }
        }
    }

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

fn appendFunctionReturnIds(builder: *CapiAnalysisBuilder, op: mlir.MlirOperation) AnalyzeError!void {
    if (mlir.mlirOperationGetNumRegions(op) == 0) return;
    const entry = mlir.mlirRegionGetFirstBlock(mlir.mlirOperationGetRegion(op, 0));
    if (mlir.mlirBlockIsNull(entry)) return;
    const terminator = mlir.mlirBlockGetTerminator(entry);
    if (mlir.mlirOperationIsNull(terminator) or !std.mem.eql(u8, operationName(terminator), "func.return")) return;

    const operand_count = mlir.mlirOperationGetNumOperands(terminator);
    builder.output_ids.clearRetainingCapacity();
    var operand_index: isize = 0;
    while (operand_index < operand_count) : (operand_index += 1) {
        const id = builder.lookupValue(mlir.mlirOperationGetOperand(terminator, operand_index)) orelse return error.InvalidStablehloModule;
        try builder.output_ids.append(builder.allocator, id);
    }
}

fn visitOperationFromCapi(builder: *CapiAnalysisBuilder, op: mlir.MlirOperation) AnalyzeError!void {
    const name = operationName(op);
    const loc = mlirLocationLineColumn(mlir.mlirOperationGetLocation(op));
    try inspectOperationAttributesFromCapi(builder, op, loc);

    var recurse_children = true;
    if (std.mem.eql(u8, name, "func.func")) {
        try analyzeFunctionFromCapi(builder, op);
    } else if (std.mem.startsWith(u8, name, "stablehlo.")) {
        try analyzeStablehloOperationFromCapi(builder, op, name);
        recurse_children = false;
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
        } else if (std.mem.eql(u8, short_name, "constant")) {
            try analyzeStablehloOperationFromCapi(builder, op, "stablehlo.constant");
            recurse_children = false;
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
    if (recurse_children) {
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
    if (std.mem.eql(u8, name, "func.func")) try appendFunctionReturnIds(builder, op);
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
    if (builder.output_ids.items.len == 0 and builder.values.items.len != 0) {
        try builder.output_ids.append(builder.allocator, builder.values.items[builder.values.items.len - 1].id);
    }

    const dialects = try builder.dialects.toOwnedSlice(allocator);
    errdefer allocator.free(dialects);
    const ops = try builder.ops.toOwnedSlice(allocator);
    errdefer {
        for (ops) |op| op.deinit(allocator);
        allocator.free(ops);
    }
    const values = try builder.values.toOwnedSlice(allocator);
    errdefer {
        for (values) |value| allocator.free(value.descriptor.dims);
        allocator.free(values);
    }
    const output_ids = try builder.output_ids.toOwnedSlice(allocator);
    errdefer allocator.free(output_ids);
    const parameter_descriptors = try builder.parameter_descriptors.toOwnedSlice(allocator);
    errdefer {
        for (parameter_descriptors) |descriptor| allocator.free(descriptor.dims);
        allocator.free(parameter_descriptors);
    }
    const parameter_shardings = try builder.parameter_shardings.toOwnedSlice(allocator);
    errdefer {
        for (parameter_shardings) |sharding| sharding.deinit(allocator);
        allocator.free(parameter_shardings);
    }
    const output_shardings = try builder.output_shardings.toOwnedSlice(allocator);
    errdefer {
        for (output_shardings) |sharding| sharding.deinit(allocator);
        allocator.free(output_shardings);
    }
    builder.value_map.deinit(allocator);

    return .{
        .allocator = allocator,
        .source = source,
        .dialects = dialects,
        .ops = ops,
        .values = values,
        .output_ids = output_ids,
        .parameter_descriptors = parameter_descriptors,
        .num_parameters = builder.num_parameters,
        .num_outputs = builder.num_outputs,
        .parameter_shardings = parameter_shardings,
        .output_shardings = output_shardings,
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

    var source = try program_reader.allocRemaining(allocator, .limited(64 * 1024 * 1024));
    errdefer allocator.free(source);
    const use_portable_artifact = format == .stablehlo_bytecode or !isLikelyTextMlir(source);
    var session = if (use_portable_artifact) blk: {
        const artifact = source;
        var deserialized_text = std.Io.Writer.Allocating.init(allocator);
        errdefer deserialized_text.deinit();
        const deserialized = try deserializeStablehloPortableArtifactWithCapi(artifact, diagnostic_writer);
        try writeModuleText(deserialized, &deserialized_text.writer);
        source = try deserialized_text.toOwnedSlice();
        allocator.free(artifact);
        break :blk deserialized;
    } else try parseAndRunMlirWithCapi(source, diagnostic_writer);
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

    var diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer diagnostics.deinit();
    try verifyExecutablePlan(std.testing.allocator, plan, &diagnostics.writer);
}

test "executable plan values carry parameter and result descriptors" {
    const module_text =
        \\module {
        \\  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
        \\    %0 = stablehlo.tanh %arg0 : tensor<4xf32>
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

    try std.testing.expectEqual(@as(usize, 2), plan.values.len);
    try std.testing.expectEqual(core.BufferType.f32, plan.values[0].descriptor.element_type);
    try std.testing.expectEqual(core.BufferType.f32, plan.values[1].descriptor.element_type);
    try std.testing.expectEqualSlices(i64, &.{4}, plan.values[0].descriptor.dims);
    try std.testing.expectEqualSlices(i64, &.{4}, plan.values[1].descriptor.dims);
}

test "executable plan verifier rejects unknown value references with diagnostics" {
    var plan = try makeReplicatedPlan(std.testing.allocator, .{}, 1, 1);
    defer plan.deinit();
    std.testing.allocator.free(plan.instructions[0].inputs);
    plan.instructions[0].inputs = try std.testing.allocator.dupe(ValueId, &.{.{ .index = 99 }});

    var diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer diagnostics.deinit();
    try std.testing.expectError(error.InvalidExecutablePlan, verifyExecutablePlan(std.testing.allocator, plan, &diagnostics.writer));
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "pass=pjrtx-plan-verify") != null);
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "instruction input references an unknown value") != null);
}

test "executable plan verifier rejects shape type mismatch with diagnostics" {
    var plan = try makeReplicatedPlan(std.testing.allocator, .{}, 1, 1);
    defer plan.deinit();
    plan.values[0].descriptor.element_type = .f32;
    plan.values[0].descriptor.dims = try std.testing.allocator.dupe(i64, &.{4});
    plan.values[1].descriptor.element_type = .f32;
    plan.values[1].descriptor.dims = try std.testing.allocator.dupe(i64, &.{2});

    var diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer diagnostics.deinit();
    try std.testing.expectError(error.InvalidExecutablePlan, verifyExecutablePlan(std.testing.allocator, plan, &diagnostics.writer));
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "feature=shape-type") != null);
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

    var verify_diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer verify_diagnostics.deinit();
    try verifyExecutablePlan(std.testing.allocator, plan, &verify_diagnostics.writer);
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

test "executable plan lowers sort with dimension and comparator direction" {
    const module_text =
        \\module {
        \\  func.func @main(%arg0: tensor<2x3xf32>) -> tensor<2x3xf32> {
        \\    %0 = "stablehlo.sort"(%arg0) ({
        \\    ^bb0(%lhs: tensor<f32>, %rhs: tensor<f32>):
        \\      %pred = stablehlo.compare  LT, %lhs, %rhs,  FLOAT : (tensor<f32>, tensor<f32>) -> tensor<i1>
        \\      stablehlo.return %pred : tensor<i1>
        \\    }) {dimension = 1 : i64, is_stable = true} : (tensor<2x3xf32>) -> tensor<2x3xf32>
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
    try std.testing.expectEqual(PlanInstructionKind.sort, plan.instructions[0].kind);
    try std.testing.expectEqualSlices(i64, &.{ 2, 3 }, plan.instructions[0].dims.?);
    try std.testing.expectEqual(@as(?i64, 1), plan.instructions[0].dimension);
    try std.testing.expectEqual(core.CompareOp.lt, plan.instructions[0].compare_direction.?);
}

test "executable plan lowers heavy random and structural StableHLO op shells" {
    const module_text =
        \\module {
        \\  func.func @main(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
        \\    %0 = "stablehlo.reduce_precision"(%arg0) <{exponent_bits = 8 : i32, mantissa_bits = 23 : i32}> : (tensor<2x2xf32>) -> tensor<2x2xf32>
        \\    %1 = "stablehlo.cholesky"(%0) <{lower = true}> : (tensor<2x2xf32>) -> tensor<2x2xf32>
        \\    %2 = "stablehlo.custom_call"(%1) {call_target_name = "pjrtx.test"} : (tensor<2x2xf32>) -> tensor<2x2xf32>
        \\    return %2 : tensor<2x2xf32>
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

    try std.testing.expectEqual(@as(usize, 3), plan.instructions.len);
    try std.testing.expectEqual(PlanInstructionKind.reduce_precision, plan.instructions[0].kind);
    try std.testing.expectEqual(PlanInstructionKind.cholesky, plan.instructions[1].kind);
    try std.testing.expectEqual(@as(?bool, true), plan.instructions[1].lower);
    try std.testing.expectEqual(PlanInstructionKind.custom_call, plan.instructions[2].kind);
    try std.testing.expectEqualStrings("pjrtx.test", plan.instructions[2].custom_call_target.?);
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
        \\  %0 = "stablehlo.optimization_barrier"(%arg0) {sdy.sharding = #sdy.sharding_per_value<[<@mesh, [{"x"}]>]>} : (tensor<4xf32>) -> tensor<4xf32>
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
    try std.testing.expect(std.mem.indexOf(u8, text, "op=optimization_barrier") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "dtype=f32") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "rank=1") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "sharding=sdy.sharding_per_value") != null);
}

test "analyze stablehlo bytecode reports deserialization failures precisely" {
    var reader: std.Io.Reader = .fixed("\x00not-a-stablehlo-portable-artifact");
    var diagnostics = std.Io.Writer.Allocating.init(std.testing.allocator);
    defer diagnostics.deinit();

    try std.testing.expectError(
        error.InvalidStablehloModule,
        analyzeProgramFromReader(std.testing.allocator, "stablehlo_bytecode", &reader, &diagnostics.writer),
    );
    try std.testing.expect(std.mem.indexOf(u8, diagnostics.writer.buffered(), "portable artifact deserialization failed") != null);
}
