#ifndef PJRTX_COMPILER_MLIR_C_API_H_
#define PJRTX_COMPILER_MLIR_C_API_H_

#include "mlir-c/Dialect/Func.h"
#include "mlir-c/Dialect/Shape.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/IR.h"
#include "mlir-c/Pass.h"
#include "mlir-c/Support.h"
#include "mlir-c/Transforms.h"
#include "shardy/integrations/c/attributes.h"
#include "shardy/integrations/c/dialect.h"
#include "shardy/integrations/c/passes.h"
#include "stablehlo/integrations/c/ChloDialect.h"
#include "stablehlo/integrations/c/StablehloDialect.h"
#include "stablehlo/integrations/c/StablehloDialectApi.h"
#include "stablehlo/integrations/c/StablehloAttributes.h"

#ifdef __cplusplus
extern "C" void mlirRegisterFuncExtensions(MlirDialectRegistry registry);
extern "C" bool pjrtxMlirOpPassManagerAddPipelineSucceeded(
    MlirOpPassManager op_pass_manager, MlirStringRef pipeline,
    MlirStringCallback callback, void *user_data);
extern "C" bool pjrtxMlirPassManagerRunOnOpSucceeded(
    MlirPassManager pass_manager, MlirOperation op);
#else
void mlirRegisterFuncExtensions(MlirDialectRegistry registry);
bool pjrtxMlirOpPassManagerAddPipelineSucceeded(
    MlirOpPassManager op_pass_manager, MlirStringRef pipeline,
    MlirStringCallback callback, void *user_data);
bool pjrtxMlirPassManagerRunOnOpSucceeded(MlirPassManager pass_manager,
                                          MlirOperation op);
#endif

#endif  // PJRTX_COMPILER_MLIR_C_API_H_
