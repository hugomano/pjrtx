const mlir = @import("c");
const attrs = @import("attrs.zig");
const types = @import("types.zig");

pub const ExternalStateProbePass = struct {
    var type_id_anchor: u64 = 0;

    pub const Data = struct {
        construct_count: u32 = 0,
        initialize_count: u32 = 0,
        run_count: u32 = 0,
        clone_count: u32 = 0,
        destruct_count: u32 = 0,
        missing_state: bool = false,

        pub fn record(self: Data) types.ExternalPassProbeRecord {
            return .{
                .construct_count = self.construct_count,
                .initialize_count = self.initialize_count,
                .run_count = self.run_count,
                .clone_count = self.clone_count,
                .destruct_count = self.destruct_count,
            };
        }
    };

    pub fn create(data: *Data) mlir.MlirPass {
        const callbacks: mlir.MlirExternalPassCallbacks = .{
            .construct = construct,
            .destruct = destruct,
            .initialize = initialize,
            .clone = clone,
            .run = run,
        };
        return mlir.mlirCreateExternalPass(
            mlir.mlirTypeIDCreate(&type_id_anchor),
            attrs.mlirStringRef("pjrtx-external-state-probe"),
            attrs.mlirStringRef("pjrtx-external-state-probe"),
            attrs.mlirStringRef("Proves that PjRTx can run Zig-owned MLIR external passes"),
            attrs.mlirStringRef("builtin.module"),
            0,
            null,
            callbacks,
            data,
        );
    }

    fn construct(user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.construct_count += 1;
    }

    fn destruct(user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.destruct_count += 1;
    }

    fn initialize(_: mlir.MlirContext, user_data: ?*anyopaque) callconv(.c) mlir.MlirLogicalResult {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.initialize_count += 1;
        return mlir.MlirLogicalResult{ .value = 1 };
    }

    fn clone(user_data: ?*anyopaque) callconv(.c) ?*anyopaque {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.clone_count += 1;
        return user_data;
    }

    fn run(op: mlir.MlirOperation, pass: mlir.MlirExternalPass, user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.run_count += 1;

        const state_attr = attrs.getAttr(op, "pjrtx.state");
        if (mlir.mlirAttributeIsNull(state_attr) or !mlir.mlirAttributeIsAString(state_attr)) {
            data.missing_state = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        const context = mlir.mlirOperationGetContext(op);
        attrs.setStringAttr(context, op, "pjrtx.external_pass.proof", "ran");
        attrs.setIntegerAttr(context, op, "pjrtx.external_pass.run_count", data.run_count);
    }
};

pub const TargetLegalExternalPass = struct {
    var type_id_anchor: u64 = 0;

    pub const Data = struct {
        construct_count: u32 = 0,
        initialize_count: u32 = 0,
        run_count: u32 = 0,
        clone_count: u32 = 0,
        destruct_count: u32 = 0,
        invalid_state: bool = false,
        missing_target_attr: bool = false,
    };

    pub fn create(data: *Data) mlir.MlirPass {
        const callbacks: mlir.MlirExternalPassCallbacks = .{
            .construct = construct,
            .destruct = destruct,
            .initialize = initialize,
            .clone = clone,
            .run = run,
        };
        return mlir.mlirCreateExternalPass(
            mlir.mlirTypeIDCreate(&type_id_anchor),
            attrs.mlirStringRef("pjrtx-target-legal-external"),
            attrs.mlirStringRef("pjrtx-target-legal-external"),
            attrs.mlirStringRef("Verifies PjRTx target attachment and marks target legality"),
            attrs.mlirStringRef("builtin.module"),
            0,
            null,
            callbacks,
            data,
        );
    }

    fn construct(user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.construct_count += 1;
    }

    fn destruct(user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.destruct_count += 1;
    }

    fn initialize(_: mlir.MlirContext, user_data: ?*anyopaque) callconv(.c) mlir.MlirLogicalResult {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.initialize_count += 1;
        return mlir.MlirLogicalResult{ .value = 1 };
    }

    fn clone(user_data: ?*anyopaque) callconv(.c) ?*anyopaque {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.clone_count += 1;
        return user_data;
    }

    fn run(op: mlir.MlirOperation, pass: mlir.MlirExternalPass, user_data: ?*anyopaque) callconv(.c) void {
        const data: *Data = @ptrCast(@alignCast(user_data.?));
        data.run_count += 1;

        const state_attr = attrs.getAttr(op, "pjrtx.state");
        if (!attrs.stringAttrEquals(state_attr, "target_attached")) {
            data.invalid_state = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        if (!attrs.hasStringAttr(op, "pjrtx.target.name") or
            !attrs.hasStringAttr(op, "pjrtx.target.kind") or
            !attrs.hasStringAttr(op, "pjrtx.target.fingerprint") or
            !attrs.hasIntegerAttr(op, "pjrtx.target.replicas") or
            !attrs.hasIntegerAttr(op, "pjrtx.target.partitions") or
            !attrs.hasDictionaryAttr(op, "pjrtx.target.spec"))
        {
            data.missing_target_attr = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        const context = mlir.mlirOperationGetContext(op);
        attrs.setStringAttr(context, op, "pjrtx.state", types.ModuleState.target_legal.text());
        attrs.setStringAttr(context, op, "pjrtx.target_legal.pass", "pjrtx-target-legal-external");
    }
};
