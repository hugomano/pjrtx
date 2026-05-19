const std = @import("std");
const mlir = @import("c");
const compiler_facts = @import("pjrtx/compiler/facts");
const mlir_attrs = @import("attrs.zig");
const limits = @import("limits.zig");
const types = @import("types.zig");

pub const FusionPlanExternalPass = struct {
    var type_id_anchor: u64 = 0;

    pub const Data = struct {
        construct_count: u32 = 0,
        initialize_count: u32 = 0,
        run_count: u32 = 0,
        clone_count: u32 = 0,
        destruct_count: u32 = 0,
        invalid_state: bool = false,
        missing_candidates: bool = false,
        invalid_entry: bool = false,
    };

    const matmul_epilogue_rejection_reason =
        "dot+broadcast/add/tanh is a legal epilogue candidate, but V0 rejects matmul epilogue fusion until kernel IR, math policy, and backend epilogue support are explicit";
    const elementwise_chain_acceptance_reason =
        "compatible broadcast/add/tanh chain preserves strict elementwise semantics and removes intermediate materialization pressure";

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
            mlirStringRef("pjrtx-fusion-plan-external"),
            mlirStringRef("pjrtx-fusion-plan-external"),
            mlirStringRef("Verifies PjRTx fusion plan attributes and marks fusion planned"),
            mlirStringRef("builtin.module"),
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

        const state_attr = getAttr(op, "pjrtx.state");
        if (!stringAttrEquals(state_attr, "target_legal")) {
            data.invalid_state = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        const context = mlir.mlirOperationGetContext(op);
        const candidates_attr = getAttr(op, "pjrtx.fusion.candidates");
        if (mlir.mlirAttributeIsNull(candidates_attr) or !mlir.mlirAttributeIsAArray(candidates_attr)) {
            data.missing_candidates = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        if (!stampV0DecisionAttrsFromCandidates(context, op, candidates_attr, data)) {
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }
        setStringAttr(context, op, "pjrtx.state", types.ModuleState.fusion_planned.text());
        setStringAttr(context, op, "pjrtx.fusion_plan.pass", "pjrtx-fusion-plan-external");
    }

    fn verifyPressureAttr(attr: mlir.MlirAttribute) bool {
        if (mlir.mlirAttributeIsNull(attr) or !mlir.mlirAttributeIsADictionary(attr)) return false;
        if (mlir.mlirAttributeIsNull(dictAttr(attr, "split_kernel_count")) or !mlir.mlirAttributeIsAInteger(dictAttr(attr, "split_kernel_count"))) return false;
        if (mlir.mlirAttributeIsNull(dictAttr(attr, "fused_kernel_count")) or !mlir.mlirAttributeIsAInteger(dictAttr(attr, "fused_kernel_count"))) return false;

        const split_peak = u128FromStringAttrNoDiag(dictAttr(attr, "split_peak_live_bytes")) orelse return false;
        const fused_live = u128FromStringAttrNoDiag(dictAttr(attr, "fused_live_bytes")) orelse return false;
        const additional_live = u128FromStringAttrNoDiag(dictAttr(attr, "additional_live_bytes")) orelse return false;
        _ = u128FromStringAttrNoDiag(dictAttr(attr, "global_bytes_saved")) orelse return false;
        if (fused_live < split_peak) return true;
        return additional_live == fused_live - split_peak;
    }

    fn verifyCandidateBaseAttr(candidate: mlir.MlirAttribute) bool {
        if (mlir.mlirAttributeIsNull(candidate) or !mlir.mlirAttributeIsADictionary(candidate)) return false;
        return !mlir.mlirAttributeIsNull(dictAttr(candidate, "index")) and
            mlir.mlirAttributeIsAInteger(dictAttr(candidate, "index")) and
            hasNonEmptyStringDictAttr(candidate, "kind") and
            hasNonEmptyStringDictAttr(candidate, "root") and
            !mlir.mlirAttributeIsNull(dictAttr(candidate, "operation_count")) and
            mlir.mlirAttributeIsAInteger(dictAttr(candidate, "operation_count")) and
            hasNonEmptyStringDictAttr(candidate, "reason");
    }

    fn verifyCandidateDecisionInputAttr(candidate: mlir.MlirAttribute) bool {
        if (!verifyCandidateBaseAttr(candidate)) return false;
        if (!hasNonEmptyStringDictAttr(candidate, "bytes_saved")) return false;
        if (mlir.mlirAttributeIsNull(dictAttr(candidate, "launch_count_reduction")) or
            !mlir.mlirAttributeIsAInteger(dictAttr(candidate, "launch_count_reduction")))
        {
            return false;
        }
        const instructions = dictAttr(candidate, "instructions");
        if (mlir.mlirAttributeIsNull(instructions) or !mlir.mlirAttributeIsAArray(instructions) or mlir.mlirArrayAttrGetNumElements(instructions) == 0) return false;
        var instruction_index: isize = 0;
        while (instruction_index < mlir.mlirArrayAttrGetNumElements(instructions)) : (instruction_index += 1) {
            if (!mlir.mlirAttributeIsAInteger(mlir.mlirArrayAttrGetElement(instructions, instruction_index))) return false;
        }
        return verifyPressureAttr(dictAttr(candidate, "pressure_delta"));
    }

    fn stampV0DecisionAttrsFromCandidates(
        context: mlir.MlirContext,
        module_op: mlir.MlirOperation,
        candidates_attr: mlir.MlirAttribute,
        data: *Data,
    ) bool {
        const candidate_count = mlir.mlirArrayAttrGetNumElements(candidates_attr);
        if (candidate_count > limits.max_fusion_candidates) {
            data.invalid_entry = true;
            return false;
        }

        var updated_candidates: [limits.max_fusion_candidates]mlir.MlirAttribute = undefined;
        var candidate_index: isize = 0;
        while (candidate_index < candidate_count) : (candidate_index += 1) {
            const candidate = mlir.mlirArrayAttrGetElement(candidates_attr, candidate_index);
            if (!verifyCandidateDecisionInputAttr(candidate)) {
                data.invalid_entry = true;
                return false;
            }
            const update_index = std.math.cast(usize, candidate_index) orelse unreachable;
            updated_candidates[update_index] = candidateWithV0DecisionAttr(context, candidate);
        }

        mlir.mlirOperationSetAttributeByName(
            module_op,
            mlirStringRef("pjrtx.fusion.candidates"),
            mlir.mlirArrayAttrGet(context, candidate_count, &updated_candidates),
        );
        return true;
    }

    fn candidateWithV0DecisionAttr(context: mlir.MlirContext, candidate: mlir.MlirAttribute) mlir.MlirAttribute {
        const kind = stringAttrValue(dictAttr(candidate, "kind")) orelse "";
        const decision = if (std.mem.eql(u8, kind, "elementwise_chain")) "accepted" else "rejected";
        const reason = if (std.mem.eql(u8, kind, "elementwise_chain"))
            elementwise_chain_acceptance_reason
        else
            matmul_epilogue_rejection_reason;
        const attrs = [_]mlir.MlirNamedAttribute{
            namedAttr(context, "index", dictAttr(candidate, "index")),
            namedAttr(context, "kind", dictAttr(candidate, "kind")),
            namedAttr(context, "root", dictAttr(candidate, "root")),
            namedAttr(context, "operation_count", dictAttr(candidate, "operation_count")),
            namedAttr(context, "reason", dictAttr(candidate, "reason")),
            namedAttr(context, "plan_index", dictAttr(candidate, "index")),
            namedAttr(context, "decision", stringAttr(context, decision)),
            namedAttr(context, "instructions", dictAttr(candidate, "instructions")),
            namedAttr(context, "bytes_saved", dictAttr(candidate, "bytes_saved")),
            namedAttr(context, "launch_count_reduction", dictAttr(candidate, "launch_count_reduction")),
            namedAttr(context, "pressure_delta", dictAttr(candidate, "pressure_delta")),
            namedAttr(context, "decision_reason", stringAttr(context, reason)),
        };
        return mlir.mlirDictionaryAttrGet(context, attrs.len, &attrs);
    }
};

pub const FusionCandidateDiscoveryExternalPass = struct {
    var type_id_anchor: u64 = 0;

    const max_tracked_operations = 128;

    const CandidateKind = enum {
        matmul_epilogue,
        elementwise_chain,

        fn text(self: CandidateKind) []const u8 {
            return switch (self) {
                .matmul_epilogue => "matmul_epilogue",
                .elementwise_chain => "elementwise_chain",
            };
        }

        fn root(self: CandidateKind) []const u8 {
            _ = self;
            return "stablehlo.tanh";
        }

        fn operationCount(self: CandidateKind) u32 {
            return switch (self) {
                .matmul_epilogue => 4,
                .elementwise_chain => 3,
            };
        }

        fn reason(self: CandidateKind) []const u8 {
            return switch (self) {
                .matmul_epilogue => "dot+broadcast/add/tanh epilogue candidate",
                .elementwise_chain => "broadcast/add/tanh strict elementwise chain",
            };
        }
    };

    const OperationRecord = struct {
        op: mlir.MlirOperation,
        instruction_id: u32,
        result_bytes: u128,
    };

    const CandidateRecord = struct {
        kind: CandidateKind,
        instruction_ids: [4]u32,
        instruction_count: u32,
        bytes_saved: u128,
        launch_count_reduction: u32,
        pressure_delta: compiler_facts.FusionPressureDelta = .{},
    };

    pub const Data = struct {
        construct_count: u32 = 0,
        initialize_count: u32 = 0,
        run_count: u32 = 0,
        clone_count: u32 = 0,
        destruct_count: u32 = 0,
        invalid_state: bool = false,
        matmul_epilogue_count: u32 = 0,
        elementwise_chain_count: u32 = 0,
        operation_records: [max_tracked_operations]OperationRecord = undefined,
        operation_count: u32 = 0,
        candidate_records: [limits.max_fusion_candidates]CandidateRecord = undefined,
        candidate_count: u32 = 0,
        candidate_overflow: bool = false,
        invalid_candidate_metrics: bool = false,

        fn appendCandidate(self: *Data, candidate: CandidateRecord) void {
            if (self.candidate_count >= limits.max_fusion_candidates) {
                self.candidate_overflow = true;
                return;
            }
            self.candidate_records[self.candidate_count] = candidate;
            self.candidate_count += 1;
            switch (candidate.kind) {
                .matmul_epilogue => self.matmul_epilogue_count += 1,
                .elementwise_chain => self.elementwise_chain_count += 1,
            }
        }

        fn recordOperation(self: *Data, op: mlir.MlirOperation) void {
            if (self.operation_count >= max_tracked_operations) {
                self.candidate_overflow = true;
                return;
            }
            const record_index = self.operation_count;
            const index = std.math.cast(usize, record_index) orelse unreachable;
            self.operation_records[index] = .{
                .op = op,
                .instruction_id = record_index,
                .result_bytes = operationResultBytes(op) orelse 0,
            };
            self.operation_count += 1;
        }

        fn instructionId(self: *const Data, op: mlir.MlirOperation) ?u32 {
            var index: u32 = 0;
            while (index < self.operation_count) : (index += 1) {
                const record = self.operation_records[std.math.cast(usize, index) orelse unreachable];
                if (mlir.mlirOperationEqual(record.op, op)) return record.instruction_id;
            }
            return null;
        }

        fn resultBytes(self: *const Data, op: mlir.MlirOperation) ?u128 {
            var index: u32 = 0;
            while (index < self.operation_count) : (index += 1) {
                const record = self.operation_records[std.math.cast(usize, index) orelse unreachable];
                if (mlir.mlirOperationEqual(record.op, op)) return record.result_bytes;
            }
            return null;
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
            mlirStringRef("pjrtx-fusion-candidate-external"),
            mlirStringRef("pjrtx-fusion-candidate-external"),
            mlirStringRef("Discovers first PjRTx fusion candidates from StableHLO structure"),
            mlirStringRef("builtin.module"),
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

        const state_attr = getAttr(op, "pjrtx.state");
        if (!stringAttrEquals(state_attr, "target_legal")) {
            data.invalid_state = true;
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        walkOperations(op, data);
        if (data.candidate_overflow or data.invalid_candidate_metrics) {
            mlir.mlirExternalPassSignalFailure(pass);
            return;
        }

        const context = mlir.mlirOperationGetContext(op);
        setFusionCandidateArrayAttr(context, op, data);
        setIntegerAttr(context, op, "pjrtx.fusion.candidates.matmul_epilogue", data.matmul_epilogue_count);
        setIntegerAttr(context, op, "pjrtx.fusion.candidates.elementwise_chain", data.elementwise_chain_count);
        setStringAttr(context, op, "pjrtx.fusion_candidate.pass", "pjrtx-fusion-candidate-external");
    }

    fn walkOperations(op: mlir.MlirOperation, data: *Data) void {
        if (isImportedGraphOperation(op)) {
            data.recordOperation(op);
            if (operationNameEquals(op, "stablehlo.tanh")) {
                appendTanhFusionCandidates(op, data);
            }
        }

        const region_count = mlir.mlirOperationGetNumRegions(op);
        var region_index: isize = 0;
        while (region_index < region_count) : (region_index += 1) {
            var block = mlir.mlirRegionGetFirstBlock(mlir.mlirOperationGetRegion(op, region_index));
            while (!mlir.mlirBlockIsNull(block)) : (block = mlir.mlirBlockGetNextInRegion(block)) {
                var child = mlir.mlirBlockGetFirstOperation(block);
                while (!mlir.mlirOperationIsNull(child)) : (child = mlir.mlirOperationGetNextInBlock(child)) {
                    walkOperations(child, data);
                }
            }
        }
    }

    fn appendTanhFusionCandidates(tanh_op: mlir.MlirOperation, data: *Data) void {
        const add_op = operandOwnerNamed(tanh_op, 0, "stablehlo.add") orelse return;
        const broadcast_op = broadcastOperandOwner(add_op) orelse return;
        const elementwise_candidate = elementwiseCandidateRecord(data, broadcast_op, add_op, tanh_op) orelse {
            data.invalid_candidate_metrics = true;
            return;
        };
        if (dotOperandOwner(add_op)) |dot_op| {
            const matmul_candidate = matmulEpilogueCandidateRecord(data, dot_op, broadcast_op, add_op, tanh_op) orelse {
                data.invalid_candidate_metrics = true;
                return;
            };
            data.appendCandidate(matmul_candidate);
        }
        data.appendCandidate(elementwise_candidate);
    }

    fn isImportedGraphOperation(op: mlir.MlirOperation) bool {
        const parent = mlir.mlirOperationGetParentOperation(op);
        if (mlir.mlirOperationIsNull(parent) or !operationNameEquals(parent, "func.func")) return false;
        return std.mem.startsWith(u8, operationName(op), "stablehlo.") or operationNameEquals(op, "func.return");
    }

    fn broadcastOperandOwner(add_op: mlir.MlirOperation) ?mlir.MlirOperation {
        if (operandOwnerNamed(add_op, 0, "stablehlo.broadcast_in_dim")) |op| return op;
        return operandOwnerNamed(add_op, 1, "stablehlo.broadcast_in_dim");
    }

    fn dotOperandOwner(add_op: mlir.MlirOperation) ?mlir.MlirOperation {
        if (operandOwnerNamed(add_op, 0, "stablehlo.dot_general")) |op| return op;
        return operandOwnerNamed(add_op, 1, "stablehlo.dot_general");
    }

    fn elementwiseCandidateRecord(
        data: *const Data,
        broadcast_op: mlir.MlirOperation,
        add_op: mlir.MlirOperation,
        tanh_op: mlir.MlirOperation,
    ) ?CandidateRecord {
        const broadcast_id = data.instructionId(broadcast_op) orelse return null;
        const add_id = data.instructionId(add_op) orelse return null;
        const tanh_id = data.instructionId(tanh_op) orelse return null;
        const broadcast_bytes = data.resultBytes(broadcast_op) orelse return null;
        const add_bytes = data.resultBytes(add_op) orelse return null;
        return .{
            .kind = .elementwise_chain,
            .instruction_ids = .{ broadcast_id, add_id, tanh_id, 0 },
            .instruction_count = 3,
            .bytes_saved = broadcast_bytes + add_bytes,
            .launch_count_reduction = 2,
            .pressure_delta = .{},
        };
    }

    fn matmulEpilogueCandidateRecord(
        data: *const Data,
        dot_op: mlir.MlirOperation,
        broadcast_op: mlir.MlirOperation,
        add_op: mlir.MlirOperation,
        tanh_op: mlir.MlirOperation,
    ) ?CandidateRecord {
        const dot_id = data.instructionId(dot_op) orelse return null;
        const broadcast_id = data.instructionId(broadcast_op) orelse return null;
        const add_id = data.instructionId(add_op) orelse return null;
        const tanh_id = data.instructionId(tanh_op) orelse return null;
        const dot_output_bytes = data.resultBytes(dot_op) orelse return null;
        const broadcast_output_bytes = data.resultBytes(broadcast_op) orelse return null;
        const add_output_bytes = data.resultBytes(add_op) orelse return null;
        const tanh_output_bytes = data.resultBytes(tanh_op) orelse return null;
        const dot_input_bytes = valueBytes(mlir.mlirOperationGetOperand(dot_op, 0)) orelse return null;
        const rhs_input_bytes = valueBytes(mlir.mlirOperationGetOperand(dot_op, 1)) orelse return null;
        const broadcast_input_bytes = valueBytes(mlir.mlirOperationGetOperand(broadcast_op, 0)) orelse return null;
        const matmul_live_bytes = dot_input_bytes + rhs_input_bytes + dot_output_bytes;
        const epilogue_live_bytes = dot_output_bytes + broadcast_input_bytes + broadcast_output_bytes + add_output_bytes + tanh_output_bytes;
        const fused_live_bytes = dot_input_bytes + rhs_input_bytes + broadcast_input_bytes + dot_output_bytes + broadcast_output_bytes + add_output_bytes + tanh_output_bytes;
        const split_peak_live_bytes = @max(matmul_live_bytes, epilogue_live_bytes);
        const bytes_saved = dot_output_bytes + broadcast_output_bytes + add_output_bytes;
        return .{
            .kind = .matmul_epilogue,
            .instruction_ids = .{ dot_id, broadcast_id, add_id, tanh_id },
            .instruction_count = 4,
            .bytes_saved = bytes_saved,
            .launch_count_reduction = 0,
            .pressure_delta = .{
                .split_kernel_count = 2,
                .fused_kernel_count = 1,
                .split_peak_live_bytes = split_peak_live_bytes,
                .fused_live_bytes = fused_live_bytes,
                .additional_live_bytes = if (fused_live_bytes > split_peak_live_bytes) fused_live_bytes - split_peak_live_bytes else 0,
                .global_bytes_saved = bytes_saved,
            },
        };
    }

    fn operandOwnerNamed(op: mlir.MlirOperation, operand_index: isize, expected_name: []const u8) ?mlir.MlirOperation {
        if (operand_index >= mlir.mlirOperationGetNumOperands(op)) return null;
        const operand = mlir.mlirOperationGetOperand(op, operand_index);
        if (mlir.mlirValueIsNull(operand) or !mlir.mlirValueIsAOpResult(operand)) return null;
        const owner = mlir.mlirOpResultGetOwner(operand);
        if (mlir.mlirOperationIsNull(owner) or !operationNameEquals(owner, expected_name)) return null;
        return owner;
    }

    fn setFusionCandidateArrayAttr(context: mlir.MlirContext, module_op: mlir.MlirOperation, data: *const Data) void {
        var attrs: [limits.max_fusion_candidates]mlir.MlirAttribute = undefined;
        var index: u32 = 0;
        while (index < data.candidate_count) : (index += 1) {
            const candidate_index = std.math.cast(usize, index) orelse unreachable;
            attrs[candidate_index] = fusionCandidateAttr(context, index, data.candidate_records[candidate_index]);
        }
        mlir.mlirOperationSetAttributeByName(
            module_op,
            mlirStringRef("pjrtx.fusion.candidates"),
            mlir.mlirArrayAttrGet(context, std.math.cast(isize, data.candidate_count) orelse unreachable, &attrs),
        );
    }

    fn fusionCandidateAttr(context: mlir.MlirContext, index: u32, candidate: CandidateRecord) mlir.MlirAttribute {
        const kind = candidate.kind;
        const attrs = [_]mlir.MlirNamedAttribute{
            namedAttr(context, "index", integerAttr(context, index)),
            namedAttr(context, "kind", stringAttr(context, kind.text())),
            namedAttr(context, "root", stringAttr(context, kind.root())),
            namedAttr(context, "operation_count", integerAttr(context, kind.operationCount())),
            namedAttr(context, "reason", stringAttr(context, kind.reason())),
            namedAttr(
                context,
                "instructions",
                instructionArrayAttr(context, candidate.instruction_ids[0 .. std.math.cast(usize, candidate.instruction_count) orelse unreachable]),
            ),
            namedAttr(context, "bytes_saved", u128StackStringAttr(context, candidate.bytes_saved)),
            namedAttr(context, "launch_count_reduction", integerAttr(context, candidate.launch_count_reduction)),
            namedAttr(context, "pressure_delta", pressureAttr(context, candidate.pressure_delta)),
        };
        return mlir.mlirDictionaryAttrGet(context, attrs.len, &attrs);
    }
};

fn mlirStringRef(text: []const u8) mlir.MlirStringRef {
    return mlir_attrs.mlirStringRef(text);
}

fn mlirStringSlice(text: mlir.MlirStringRef) []const u8 {
    return mlir_attrs.mlirStringSlice(text);
}

fn getAttr(op: mlir.MlirOperation, name: []const u8) mlir.MlirAttribute {
    return mlir_attrs.getAttr(op, name);
}

fn setStringAttr(context: mlir.MlirContext, op: mlir.MlirOperation, name: []const u8, value: []const u8) void {
    mlir_attrs.setStringAttr(context, op, name, value);
}

fn setIntegerAttr(context: mlir.MlirContext, op: mlir.MlirOperation, name: []const u8, value: u32) void {
    mlir_attrs.setIntegerAttr(context, op, name, value);
}

fn namedAttr(context: mlir.MlirContext, name: []const u8, attr: mlir.MlirAttribute) mlir.MlirNamedAttribute {
    return mlir.mlirNamedAttributeGet(mlir.mlirIdentifierGet(context, mlirStringRef(name)), attr);
}

fn stringAttr(context: mlir.MlirContext, value: []const u8) mlir.MlirAttribute {
    return mlir_attrs.stringAttr(context, value);
}

fn integerAttr(context: mlir.MlirContext, value: u32) mlir.MlirAttribute {
    return mlir_attrs.integerAttr(context, value);
}

fn stringAttrValue(attr: mlir.MlirAttribute) ?[]const u8 {
    return mlir_attrs.stringAttrValue(attr);
}

fn stringAttrEquals(attr: mlir.MlirAttribute, expected: []const u8) bool {
    return mlir_attrs.stringAttrEquals(attr, expected);
}

fn dictAttr(attr: mlir.MlirAttribute, name: []const u8) mlir.MlirAttribute {
    return mlir.mlirDictionaryAttrGetElementByName(attr, mlirStringRef(name));
}

fn hasNonEmptyStringDictAttr(attr: mlir.MlirAttribute, name: []const u8) bool {
    return (stringAttrValue(dictAttr(attr, name)) orelse return false).len != 0;
}

fn u128FromStringAttrNoDiag(attr: mlir.MlirAttribute) ?u128 {
    const value = stringAttrValue(attr) orelse return null;
    return std.fmt.parseUnsigned(u128, value, 10) catch null;
}

fn instructionArrayAttr(context: mlir.MlirContext, ids: []const u32) mlir.MlirAttribute {
    var attr_values: [4]mlir.MlirAttribute = undefined;
    for (ids, 0..) |id, index| {
        attr_values[index] = integerAttr(context, id);
    }
    return mlir.mlirArrayAttrGet(context, std.math.cast(isize, ids.len) orelse unreachable, &attr_values);
}

fn pressureAttr(context: mlir.MlirContext, pressure: compiler_facts.FusionPressureDelta) mlir.MlirAttribute {
    const attr_values = [_]mlir.MlirNamedAttribute{
        namedAttr(context, "split_kernel_count", integerAttr(context, pressure.split_kernel_count)),
        namedAttr(context, "fused_kernel_count", integerAttr(context, pressure.fused_kernel_count)),
        namedAttr(context, "split_peak_live_bytes", u128StackStringAttr(context, pressure.split_peak_live_bytes)),
        namedAttr(context, "fused_live_bytes", u128StackStringAttr(context, pressure.fused_live_bytes)),
        namedAttr(context, "additional_live_bytes", u128StackStringAttr(context, pressure.additional_live_bytes)),
        namedAttr(context, "global_bytes_saved", u128StackStringAttr(context, pressure.global_bytes_saved)),
    };
    return mlir.mlirDictionaryAttrGet(context, attr_values.len, &attr_values);
}

fn u128StackStringAttr(context: mlir.MlirContext, value: u128) mlir.MlirAttribute {
    var buffer: [39]u8 = undefined;
    const text = std.fmt.bufPrint(&buffer, "{d}", .{value}) catch unreachable;
    return stringAttr(context, text);
}

fn operationName(op: mlir.MlirOperation) []const u8 {
    return mlirStringSlice(mlir.mlirIdentifierStr(mlir.mlirOperationGetName(op)));
}

fn operationNameEquals(op: mlir.MlirOperation, expected: []const u8) bool {
    return std.mem.eql(u8, operationName(op), expected);
}

fn operationResultBytes(op: mlir.MlirOperation) ?u128 {
    if (mlir.mlirOperationGetNumResults(op) == 0) return 0;
    return valueBytes(mlir.mlirOperationGetResult(op, 0));
}

fn valueBytes(value: mlir.MlirValue) ?u128 {
    if (mlir.mlirValueIsNull(value)) return null;
    const ty = mlir.mlirValueGetType(value);
    if (!mlir.mlirTypeIsARankedTensor(ty)) return null;
    const element = mlir.mlirShapedTypeGetElementType(ty);
    const element_bytes = elementByteSize(element) orelse return null;
    var elements: u128 = 1;
    const rank = mlir.mlirShapedTypeGetRank(ty);
    var dim_index: isize = 0;
    while (dim_index < rank) : (dim_index += 1) {
        const dim = mlir.mlirShapedTypeGetDimSize(ty, dim_index);
        if (dim < 0) return null;
        elements *= std.math.cast(u128, dim) orelse return null;
    }
    return elements * element_bytes;
}

fn elementByteSize(element: mlir.MlirType) ?u128 {
    if (mlir.mlirTypeIsAF16(element) or mlir.mlirTypeIsABF16(element)) return 2;
    if (mlir.mlirTypeIsAF32(element)) return 4;
    if (mlir.mlirTypeIsAF64(element)) return 8;
    if (mlir.mlirTypeIsAInteger(element)) {
        const width = mlir.mlirIntegerTypeGetWidth(element);
        if (width == 1) return 1;
        if (width > 0 and @mod(width, 8) == 0) return std.math.cast(u128, @divExact(width, 8)) orelse null;
    }
    return null;
}
