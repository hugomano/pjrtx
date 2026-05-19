const std = @import("std");
const mlir = @import("c");

pub const PassRunError = error{
    PassManagerCreateFailed,
};

pub const PassRunResult = struct {
    pass_manager_failed: bool,
};

/// Owns the common MLIR pass-manager lifecycle for a single module pass.
/// Individual pass structs still own their callbacks and semantic validation.
pub const PassRunner = struct {
    pub fn runModulePass(
        pass_manager_slot: *mlir.MlirPassManager,
        context: mlir.MlirContext,
        module_op: mlir.MlirOperation,
        pass_name: []const u8,
        pass: mlir.MlirPass,
        diagnostics: *std.Io.Writer,
    ) !PassRunResult {
        if (!mlir.mlirPassManagerIsNull(pass_manager_slot.*)) {
            mlir.mlirPassManagerDestroy(pass_manager_slot.*);
            pass_manager_slot.* = mlir.MlirPassManager{ .ptr = null };
        }

        pass_manager_slot.* = mlir.mlirPassManagerCreate(context);
        if (mlir.mlirPassManagerIsNull(pass_manager_slot.*)) {
            try diagnostics.print("pass={s} feature=pass-manager reason=failed to create MLIR pass manager\n", .{pass_name});
            return PassRunError.PassManagerCreateFailed;
        }
        mlir.mlirPassManagerEnableVerifier(pass_manager_slot.*, true);

        mlir.mlirPassManagerAddOwnedPass(pass_manager_slot.*, pass);

        const result = mlir.mlirPassManagerRunOnOp(pass_manager_slot.*, module_op);
        const pass_manager_failed = result.value == 0;

        mlir.mlirPassManagerDestroy(pass_manager_slot.*);
        pass_manager_slot.* = mlir.MlirPassManager{ .ptr = null };

        return .{ .pass_manager_failed = pass_manager_failed };
    }
};
