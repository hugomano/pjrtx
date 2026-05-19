#ifndef PJRTX_NEW_COMPILER_MLIR_C_API_H_
#define PJRTX_NEW_COMPILER_MLIR_C_API_H_

#include "mlir-c/Dialect/Func.h"
#include "mlir-c/Dialect/Shape.h"
#include "mlir-c/BuiltinAttributes.h"
#include "mlir-c/BuiltinTypes.h"
#include "mlir-c/IR.h"
#include "mlir-c/Pass.h"
#include "mlir-c/Support.h"
#include "stablehlo/integrations/c/ChloDialect.h"
#include "stablehlo/integrations/c/StablehloAttributes.h"
#include "stablehlo/integrations/c/StablehloDialect.h"
#include "shardy/integrations/c/dialect.h"

#ifdef __cplusplus
extern "C" void pjrtxMlirRegisterFuncExtensions(MlirDialectRegistry registry);
#else
void pjrtxMlirRegisterFuncExtensions(MlirDialectRegistry registry);
#endif

#endif  // PJRTX_NEW_COMPILER_MLIR_C_API_H_
