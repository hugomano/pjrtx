#include "pjrtx/compiler/mlir_c_api.h"

#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"

extern "C" void pjrtxMlirRegisterFuncExtensions(
    MlirDialectRegistry registry) {
  mlir::func::registerAllExtensions(*unwrap(registry));
}
