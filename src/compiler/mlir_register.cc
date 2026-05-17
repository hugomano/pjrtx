#include "src/compiler/mlir_c_api.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"

extern "C" void mlirRegisterFuncExtensions(MlirDialectRegistry registry) {
  mlir::func::registerAllExtensions(*unwrap(registry));
}

extern "C" bool pjrtxMlirOpPassManagerAddPipelineSucceeded(
    MlirOpPassManager op_pass_manager, MlirStringRef pipeline,
    MlirStringCallback callback, void *user_data) {
  return mlirLogicalResultIsSuccess(mlirOpPassManagerAddPipeline(
      op_pass_manager, pipeline, callback, user_data));
}

extern "C" bool pjrtxMlirPassManagerRunOnOpSucceeded(
    MlirPassManager pass_manager, MlirOperation op) {
  return mlirLogicalResultIsSuccess(mlirPassManagerRunOnOp(pass_manager, op));
}
