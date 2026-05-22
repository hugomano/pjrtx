#ifndef PJRTX_BACKEND_MLX_METAL_API_INTERNAL_H_
#define PJRTX_BACKEND_MLX_METAL_API_INTERNAL_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "mlx/allocator.h"
#include "mlx/array.h"
#include "src/backend/mlx_metal/api.h"

namespace MTL {
class ComputePipelineState;
}  // namespace MTL

struct PjrtxMlxMetalBuffer {
  std::unique_ptr<mlx::core::array> array;
  uint64_t byte_size;
  int dtype;
  std::vector<int64_t> dims;
  int device_ordinal;

  PjrtxMlxMetalBuffer(std::unique_ptr<mlx::core::array> array_,
                      uint64_t byte_size_, int dtype_,
                      std::vector<int64_t> dims_, int device_ordinal_)
      : array(std::move(array_)),
        byte_size(byte_size_),
        dtype(dtype_),
        dims(std::move(dims_)),
        device_ordinal(device_ordinal_) {}
};

struct PjrtxMlxMetalProgram {
  void* user_data;
  uint64_t input_count;
  uint64_t full_input_count;
  uint64_t output_count;
  PjrtxMlxMetalProgramBuildFn build_fn;
  std::vector<std::unique_ptr<PjrtxMlxMetalBuffer>> captured_inputs;
  std::vector<uint64_t> dynamic_indices;
  std::vector<int64_t> full_to_dynamic_input;
  std::function<std::vector<mlx::core::array>(
      const std::vector<mlx::core::array>&)>
      compiled;
  std::mutex mutex;
  int active_device_ordinal = 0;
  uint64_t execute_count = 0;
};

struct PjrtxMlxMetalAsyncHostToDeviceTransfer {
  mlx::core::allocator::Buffer staging;
  uint64_t byte_size;
  int dtype;
  std::vector<int64_t> dims;
  int device_ordinal;
  bool finished = false;

  PjrtxMlxMetalAsyncHostToDeviceTransfer(
      mlx::core::allocator::Buffer staging_, uint64_t byte_size_, int dtype_,
      std::vector<int64_t> dims_, int device_ordinal_)
      : staging(staging_),
        byte_size(byte_size_),
        dtype(dtype_),
        dims(std::move(dims_)),
        device_ordinal(device_ordinal_) {}
};

struct PjrtxMetalCppOwnedTensorSpec {
  int dtype;
  std::vector<int64_t> dims;
};

struct PjrtxMetalCppFusionKernel {
  int device_ordinal;
  std::string kernel_name;
  std::string source;
  MTL::ComputePipelineState* pipeline;
  std::vector<PjrtxMetalCppOwnedTensorSpec> inputs;
  std::vector<PjrtxMetalCppOwnedTensorSpec> outputs;
  uint64_t element_count;
  uint32_t threads_per_threadgroup;
  bool allow_mismatched_input_elements = false;
};

#endif  // PJRTX_BACKEND_MLX_METAL_API_INTERNAL_H_
