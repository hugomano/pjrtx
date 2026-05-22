const std = @import("std");
const mlir = @import("c");
const model = @import("compiler_model.zig");
const AnalyzeError = model.AnalyzeError;

var shardy_pass_registration_mutex: std.atomic.Mutex = .unlocked;
var shardy_passes_registered = false;
var transform_pass_registration_mutex: std.atomic.Mutex = .unlocked;
var transform_passes_registered = false;

/// Owns an MLIR registry/context/module/pass-manager bundle for one import.
pub const MlirSession = struct {
    registry: mlir.MlirDialectRegistry,
    context: mlir.MlirContext,
    module: mlir.MlirModule,
    pass_manager: mlir.MlirPassManager,

    pub fn deinit(self: *MlirSession) void {
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

pub fn mlirStringRef(text: []const u8) mlir.MlirStringRef {
    return mlir.mlirStringRefCreate(text.ptr, @intCast(text.len));
}

pub fn mlirStringSlice(text: mlir.MlirStringRef) []const u8 {
    return text.data[0..@intCast(text.length)];
}

pub fn operationName(op: mlir.MlirOperation) []const u8 {
    return mlirStringSlice(mlir.mlirIdentifierStr(mlir.mlirOperationGetName(op)));
}

pub fn namedAttributeName(named: mlir.MlirNamedAttribute) []const u8 {
    return mlirStringSlice(mlir.mlirIdentifierStr(named.name));
}

pub fn getOperationAttribute(op: mlir.MlirOperation, name: []const u8) mlir.MlirAttribute {
    return mlir.mlirOperationGetAttributeByName(op, mlirStringRef(name));
}

const MlirStringCallbackCtx = struct {
    writer: *std.Io.Writer,
    err: ?std.Io.Writer.Error = null,
};

pub fn writeMlirCallback(text: mlir.MlirStringRef, user_data: ?*anyopaque) callconv(.c) void {
    const ctx: *MlirStringCallbackCtx = @ptrCast(@alignCast(user_data.?));
    ctx.writer.writeAll(text.data[0..@intCast(text.length)]) catch |err| {
        ctx.err = err;
    };
}

pub fn loadDialect(context: mlir.MlirContext, handle: mlir.MlirDialectHandle) AnalyzeError!void {
    const dialect = mlir.mlirDialectHandleLoadDialect(handle, context);
    if (mlir.mlirDialectIsNull(dialect)) return error.InvalidStablehloModule;
}

pub fn insertDialect(registry: mlir.MlirDialectRegistry, handle: mlir.MlirDialectHandle) void {
    mlir.mlirDialectHandleInsertDialect(handle, registry);
}

pub fn registerShardyPassesOnce() void {
    while (!shardy_pass_registration_mutex.tryLock()) std.atomic.spinLoopHint();
    defer shardy_pass_registration_mutex.unlock();

    if (shardy_passes_registered) return;
    mlir.mlirRegisterAllSdyPassesAndPipelines();
    shardy_passes_registered = true;
}

pub fn registerTransformPassesOnce() void {
    while (!transform_pass_registration_mutex.tryLock()) std.atomic.spinLoopHint();
    defer transform_pass_registration_mutex.unlock();

    if (transform_passes_registered) return;
    mlir.mlirRegisterTransformsPasses();
    transform_passes_registered = true;
}

pub fn addPipeline(
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

pub fn attributeUsesShardy(attr: mlir.MlirAttribute) bool {
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

pub fn operationUsesShardy(op: mlir.MlirOperation) bool {
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

pub fn createMlirSession(writer: *std.Io.Writer) AnalyzeError!MlirSession {
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

pub fn runMlirCanonicalization(session: *MlirSession, writer: *std.Io.Writer) AnalyzeError!void {
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

pub fn parseAndRunMlirWithCapi(module_text: []const u8, writer: *std.Io.Writer) AnalyzeError!MlirSession {
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

pub fn deserializeStablehloPortableArtifactWithCapi(artifact: []const u8, writer: *std.Io.Writer) AnalyzeError!MlirSession {
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

pub fn writeModuleText(session: MlirSession, writer: *std.Io.Writer) AnalyzeError!void {
    var ctx: MlirStringCallbackCtx = .{ .writer = writer };
    mlir.mlirOperationPrint(mlir.mlirModuleGetOperation(session.module), writeMlirCallback, &ctx);
    if (ctx.err != null) return error.WriteFailed;
}

pub fn isLikelyTextMlir(source: []const u8) bool {
    for (source[0..@min(source.len, 256)]) |byte| {
        if (byte == 0) return false;
        if (byte < 0x09) return false;
        if (byte > 0x0d and byte < 0x20) return false;
    }
    return true;
}

