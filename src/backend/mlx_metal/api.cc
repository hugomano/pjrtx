#include "src/backend/mlx_metal/api.h"
#include "src/backend/mlx_metal/api_internal.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "mlx/array.h"
#include "mlx/allocator.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/compile.h"
#include "mlx/device.h"
#include "mlx/fast.h"
#include "mlx/fft.h"
#include "mlx/ops.h"
#include "mlx/random.h"
#include "mlx/stream.h"
#include "mlx/transforms.h"
#include "mlx/version.h"

namespace {

std::atomic<uint64_t> g_eval_capture_index{0};
std::atomic<uint64_t> g_program_capture_index{0};
std::atomic<uint64_t> g_dot_canonical_count{0};
std::atomic<uint64_t> g_dot_rhs_transposed_count{0};
std::atomic<uint64_t> g_dot_normalized_count{0};
std::atomic<uint64_t> g_dot_shape_sample_count{0};
std::atomic<uint64_t> g_attention_shape_sample_count{0};

bool env_flag_enabled(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return false;
  }
  return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0 &&
         std::strcmp(value, "FALSE") != 0;
}

bool env_flag_default_enabled(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return true;
  }
  return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0 &&
         std::strcmp(value, "FALSE") != 0;
}

bool profile_enabled() { return env_flag_enabled("PJRTX_PROFILE"); }

bool device_sync_profile_enabled() {
  return env_flag_enabled("PJRTX_PROFILE_DEVICE_SYNC");
}

bool dot_profile_enabled() { return env_flag_enabled("PJRTX_PROFILE_DOT"); }

bool attention_profile_enabled() {
  return env_flag_enabled("PJRTX_PROFILE_ATTENTION");
}

bool compile_eval_enabled() { return env_flag_enabled("PJRTX_MLX_COMPILE_EVAL"); }

bool compile_no_fuse_enabled() {
  return env_flag_enabled("PJRTX_MLX_COMPILE_NO_FUSE");
}

bool program_compile_no_fuse_enabled() {
  return env_flag_enabled("PJRTX_MLX_PROGRAM_COMPILE_NO_FUSE");
}

bool dot_tensordot_enabled() {
  return env_flag_default_enabled("PJRTX_MLX_DOT_TENSORDOT");
}

bool program_async_eval_enabled() {
  return env_flag_enabled("PJRTX_MLX_PROGRAM_ASYNC_EVAL");
}

std::optional<uint64_t> program_async_eval_max_output_bytes() {
  const char* value = std::getenv("PJRTX_MLX_PROGRAM_ASYNC_EVAL_MAX_OUTPUT_BYTES");
  if (value == nullptr || value[0] == '\0') {
    return std::nullopt;
  }
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(value, &end, 10);
  if (end == value || (end != nullptr && *end != '\0')) {
    return std::nullopt;
  }
  return static_cast<uint64_t>(parsed);
}

bool msl_execute_enabled() { return env_flag_enabled("PJRTX_MSL_EXECUTE"); }

std::optional<std::string> next_metal_capture_path(const char* prefix,
                                                   uint64_t index) {
  const char* dir = std::getenv("PJRTX_METAL_CAPTURE_DIR");
  if (dir == nullptr || dir[0] == '\0') {
    return std::nullopt;
  }
  if (dir[0] != '/') {
    if (profile_enabled()) {
      std::fprintf(stderr,
                   "pjrtx_profile event=metal_capture_reject reason=path_not_absolute dir=\"%s\"\n",
                   dir);
    }
    return std::nullopt;
  }
  std::string path(dir);
  if (!path.empty() && path.back() != '/') {
    path.push_back('/');
  }
  path += prefix;
  path += std::to_string(index);
  path += ".gputrace";
  return path;
}

std::optional<std::string> next_eval_capture_path() {
  const uint64_t index = g_eval_capture_index.fetch_add(1);
  return next_metal_capture_path("pjrtx_mlx_eval_", index);
}

std::optional<std::string> next_program_capture_path() {
  const uint64_t index = g_program_capture_index.fetch_add(1);
  return next_metal_capture_path("pjrtx_mlx_program_", index);
}

const char* metal_capture_hint(const char* error) {
  if (error != nullptr &&
      std::strstr(error, "Capture layer is not inserted") != nullptr) {
    return " set MTL_CAPTURE_ENABLED=1 when launching the process";
  }
  return "";
}

uint64_t now_micros() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

uint64_t elapsed_micros_since(uint64_t start_us) {
  const uint64_t end_us = now_micros();
  return end_us >= start_us ? end_us - start_us : 0;
}

std::vector<int64_t> dims_from_mlx_shape(const mlx::core::Shape& shape) {
  std::vector<int64_t> dims;
  dims.reserve(shape.size());
  for (auto dim : shape) {
    dims.push_back(static_cast<int64_t>(dim));
  }
  return dims;
}

int dtype_code_from_mlx_dtype(mlx::core::Dtype dtype) {
  if (dtype == mlx::core::bool_) {
    return PJRTX_MLX_METAL_DTYPE_PRED;
  }
  if (dtype == mlx::core::uint8) {
    return PJRTX_MLX_METAL_DTYPE_U8;
  }
  if (dtype == mlx::core::uint16) {
    return PJRTX_MLX_METAL_DTYPE_U16;
  }
  if (dtype == mlx::core::uint32) {
    return PJRTX_MLX_METAL_DTYPE_U32;
  }
  if (dtype == mlx::core::uint64) {
    return PJRTX_MLX_METAL_DTYPE_U64;
  }
  if (dtype == mlx::core::int8) {
    return PJRTX_MLX_METAL_DTYPE_S8;
  }
  if (dtype == mlx::core::int32) {
    return PJRTX_MLX_METAL_DTYPE_S32;
  }
  if (dtype == mlx::core::float16) {
    return PJRTX_MLX_METAL_DTYPE_F16;
  }
  if (dtype == mlx::core::bfloat16) {
    return PJRTX_MLX_METAL_DTYPE_BF16;
  }
  if (dtype == mlx::core::float32) {
    return PJRTX_MLX_METAL_DTYPE_F32;
  }
  if (dtype == mlx::core::complex64) {
    return PJRTX_MLX_METAL_DTYPE_C64;
  }
  return PJRTX_MLX_METAL_DTYPE_INVALID;
}

std::unique_ptr<PjrtxMlxMetalBuffer> buffer_from_array_copy(
    const mlx::core::array& array, int device_ordinal) {
  const int dtype = dtype_code_from_mlx_dtype(array.dtype());
  if (dtype == PJRTX_MLX_METAL_DTYPE_INVALID) {
    return nullptr;
  }
  auto dims = dims_from_mlx_shape(array.shape());
  const uint64_t byte_size = static_cast<uint64_t>(array.nbytes());
  return std::make_unique<PjrtxMlxMetalBuffer>(
      std::make_unique<mlx::core::array>(array), byte_size, dtype,
      std::move(dims), device_ordinal);
}

bool is_valid_binary_u8_op(int op) {
  switch (op) {
    case PJRTX_MLX_METAL_U8_BINARY_ADD:
    case PJRTX_MLX_METAL_U8_BINARY_SUBTRACT:
    case PJRTX_MLX_METAL_U8_BINARY_MULTIPLY:
    case PJRTX_MLX_METAL_U8_BINARY_DIVIDE:
    case PJRTX_MLX_METAL_BINARY_MAXIMUM:
    case PJRTX_MLX_METAL_BINARY_MINIMUM:
    case PJRTX_MLX_METAL_BINARY_POWER:
    case PJRTX_MLX_METAL_BINARY_REMAINDER:
    case PJRTX_MLX_METAL_BINARY_ATAN2:
    case PJRTX_MLX_METAL_BINARY_AND:
    case PJRTX_MLX_METAL_BINARY_OR:
    case PJRTX_MLX_METAL_BINARY_XOR:
    case PJRTX_MLX_METAL_BINARY_SHIFT_LEFT:
    case PJRTX_MLX_METAL_BINARY_SHIFT_RIGHT:
      return true;
    default:
      return false;
  }
}

bool is_valid_unary_op(int op) {
  switch (op) {
    case PJRTX_MLX_METAL_U8_UNARY_NEGATE:
    case PJRTX_MLX_METAL_UNARY_EXP:
    case PJRTX_MLX_METAL_UNARY_TANH:
    case PJRTX_MLX_METAL_UNARY_SQRT:
    case PJRTX_MLX_METAL_UNARY_RSQRT:
    case PJRTX_MLX_METAL_UNARY_ABS:
    case PJRTX_MLX_METAL_UNARY_CEIL:
    case PJRTX_MLX_METAL_UNARY_FLOOR:
    case PJRTX_MLX_METAL_UNARY_LOG:
    case PJRTX_MLX_METAL_UNARY_LOG1P:
    case PJRTX_MLX_METAL_UNARY_LOGISTIC:
    case PJRTX_MLX_METAL_UNARY_SIN:
    case PJRTX_MLX_METAL_UNARY_COS:
    case PJRTX_MLX_METAL_UNARY_SIGN:
    case PJRTX_MLX_METAL_UNARY_EXPM1:
    case PJRTX_MLX_METAL_UNARY_NOT:
    case PJRTX_MLX_METAL_UNARY_ISFINITE:
    case PJRTX_MLX_METAL_UNARY_ROUND:
    case PJRTX_MLX_METAL_UNARY_CBRT:
    case PJRTX_MLX_METAL_UNARY_ROUND_AFZ:
    case PJRTX_MLX_METAL_UNARY_POPCNT:
    case PJRTX_MLX_METAL_UNARY_CLZ:
      return true;
    default:
      return false;
  }
}

size_t dtype_size(int dtype) {
  switch (dtype) {
    case PJRTX_MLX_METAL_DTYPE_PRED:
      return sizeof(uint8_t);
    case PJRTX_MLX_METAL_DTYPE_U8:
      return sizeof(uint8_t);
    case PJRTX_MLX_METAL_DTYPE_U16:
      return sizeof(uint16_t);
    case PJRTX_MLX_METAL_DTYPE_S8:
      return sizeof(int8_t);
    case PJRTX_MLX_METAL_DTYPE_S32:
      return sizeof(int32_t);
    case PJRTX_MLX_METAL_DTYPE_U32:
      return sizeof(uint32_t);
    case PJRTX_MLX_METAL_DTYPE_U64:
      return sizeof(uint64_t);
    case PJRTX_MLX_METAL_DTYPE_F16:
      return sizeof(uint16_t);
    case PJRTX_MLX_METAL_DTYPE_BF16:
      return sizeof(uint16_t);
    case PJRTX_MLX_METAL_DTYPE_F32:
      return sizeof(float);
    case PJRTX_MLX_METAL_DTYPE_C64:
      return sizeof(mlx::core::complex64_t);
    default:
      return 0;
  }
}

const mlx::core::Dtype* mlx_dtype_from_code(int dtype) {
  switch (dtype) {
    case PJRTX_MLX_METAL_DTYPE_U8:
      return &mlx::core::uint8;
    case PJRTX_MLX_METAL_DTYPE_U16:
      return &mlx::core::uint16;
    case PJRTX_MLX_METAL_DTYPE_S8:
      return &mlx::core::int8;
    case PJRTX_MLX_METAL_DTYPE_S32:
      return &mlx::core::int32;
    case PJRTX_MLX_METAL_DTYPE_U32:
      return &mlx::core::uint32;
    case PJRTX_MLX_METAL_DTYPE_U64:
      return &mlx::core::uint64;
    case PJRTX_MLX_METAL_DTYPE_F16:
      return &mlx::core::float16;
    case PJRTX_MLX_METAL_DTYPE_BF16:
      return &mlx::core::bfloat16;
    case PJRTX_MLX_METAL_DTYPE_F32:
      return &mlx::core::float32;
    case PJRTX_MLX_METAL_DTYPE_C64:
      return &mlx::core::complex64;
    default:
      return nullptr;
  }
}

bool byte_size_matches_shape(uint64_t byte_size, int dtype,
                             const std::vector<int64_t>& dims) {
  const size_t element_size = dtype_size(dtype);
  if (element_size == 0) {
    return false;
  }
  uint64_t elements = 1;
  for (int64_t dim : dims) {
    if (dim < 0) {
      return false;
    }
    elements *= static_cast<uint64_t>(dim);
  }
  return elements * element_size == byte_size;
}

bool permutation_is_valid(const std::vector<int64_t>& permutation, size_t rank) {
  if (permutation.size() != rank) {
    return false;
  }
  std::vector<bool> seen(rank, false);
  for (int64_t dim : permutation) {
    if (dim < 0 || static_cast<size_t>(dim) >= rank) {
      return false;
    }
    if (seen[static_cast<size_t>(dim)]) {
      return false;
    }
    seen[static_cast<size_t>(dim)] = true;
  }
  return true;
}

bool broadcast_dimensions_are_valid(
    const std::vector<int64_t>& broadcast_dimensions,
    const std::vector<int64_t>& operand_dims,
    const std::vector<int64_t>& output_dims) {
  if (broadcast_dimensions.size() != operand_dims.size()) {
    return false;
  }
  std::vector<bool> seen(output_dims.size(), false);
  for (size_t i = 0; i < broadcast_dimensions.size(); ++i) {
    const int64_t output_axis = broadcast_dimensions[i];
    if (output_axis < 0 || static_cast<size_t>(output_axis) >= output_dims.size()) {
      return false;
    }
    const size_t axis = static_cast<size_t>(output_axis);
    if (seen[axis]) {
      return false;
    }
    seen[axis] = true;
    if (operand_dims[i] != 1 && operand_dims[i] != output_dims[axis]) {
      return false;
    }
  }
  return true;
}

uint64_t byte_size_for_shape(int dtype, const std::vector<int64_t>& dims) {
  const size_t element_size = dtype_size(dtype);
  if (element_size == 0) {
    return 0;
  }
  uint64_t byte_size = static_cast<uint64_t>(element_size);
  for (int64_t dim : dims) {
    if (dim < 0) {
      return 0;
    }
    byte_size *= static_cast<uint64_t>(dim);
  }
  return byte_size;
}

uint64_t element_count_for_shape(const std::vector<int64_t>& dims) {
  uint64_t elements = 1;
  for (int64_t dim : dims) {
    if (dim < 0) {
      return 0;
    }
    elements *= static_cast<uint64_t>(dim);
  }
  return elements;
}

bool slice_is_valid(const std::vector<int64_t>& start,
                    const std::vector<int64_t>& stop,
                    const std::vector<int64_t>& strides,
                    const std::vector<int64_t>& input_dims,
                    const std::vector<int64_t>& output_dims) {
  const size_t rank = input_dims.size();
  if (start.size() != rank || stop.size() != rank || strides.size() != rank ||
      output_dims.size() != rank) {
    return false;
  }
  for (size_t axis = 0; axis < rank; ++axis) {
    if (input_dims[axis] < 0 || output_dims[axis] < 0) {
      return false;
    }
    if (start[axis] < 0 || stop[axis] < start[axis] ||
        stop[axis] > input_dims[axis] || strides[axis] <= 0) {
      return false;
    }
    const int64_t span = stop[axis] - start[axis];
    const int64_t expected =
        span == 0 ? 0 : (span + strides[axis] - 1) / strides[axis];
    if (output_dims[axis] != expected) {
      return false;
    }
  }
  return true;
}

bool dynamic_slice_is_valid(const std::vector<int64_t>& slice_sizes,
                            const std::vector<int64_t>& input_dims,
                            const std::vector<int64_t>& output_dims) {
  if (slice_sizes.size() != input_dims.size() ||
      output_dims.size() != input_dims.size()) {
    return false;
  }
  for (size_t axis = 0; axis < input_dims.size(); ++axis) {
    if (input_dims[axis] < 0 || slice_sizes[axis] < 0 ||
        slice_sizes[axis] > input_dims[axis] ||
        output_dims[axis] != slice_sizes[axis]) {
      return false;
    }
  }
  return true;
}

bool concatenate_is_valid(const std::vector<int64_t>& lhs_dims,
                          const std::vector<int64_t>& rhs_dims,
                          int64_t dimension,
                          const std::vector<int64_t>& output_dims) {
  const size_t rank = lhs_dims.size();
  if (rank == 0 || rhs_dims.size() != rank || output_dims.size() != rank ||
      dimension < 0 || static_cast<size_t>(dimension) >= rank) {
    return false;
  }
  const size_t axis = static_cast<size_t>(dimension);
  for (size_t i = 0; i < rank; ++i) {
    if (lhs_dims[i] < 0 || rhs_dims[i] < 0 || output_dims[i] < 0) {
      return false;
    }
    const int64_t expected = i == axis ? lhs_dims[i] + rhs_dims[i] : lhs_dims[i];
    if (output_dims[i] != expected) {
      return false;
    }
    if (i != axis && lhs_dims[i] != rhs_dims[i]) {
      return false;
    }
  }
  return true;
}

std::vector<int64_t> permuted_dims(const std::vector<int64_t>& dims,
                                   const std::vector<int64_t>& permutation) {
  std::vector<int64_t> out;
  out.reserve(permutation.size());
  for (int64_t dim : permutation) {
    out.push_back(dims[static_cast<size_t>(dim)]);
  }
  return out;
}

std::vector<int64_t> expanded_broadcast_dims(
    const std::vector<int64_t>& operand_dims,
    const std::vector<int64_t>& broadcast_dimensions, size_t output_rank) {
  std::vector<int64_t> out(output_rank, 1);
  for (size_t i = 0; i < broadcast_dimensions.size(); ++i) {
    out[static_cast<size_t>(broadcast_dimensions[i])] = operand_dims[i];
  }
  return out;
}

mlx::core::Shape mlx_shape(const std::vector<int64_t>& dims) {
  mlx::core::Shape shape;
  shape.reserve(dims.size());
  for (int64_t dim : dims) {
    shape.push_back(static_cast<mlx::core::ShapeElem>(dim));
  }
  return shape;
}

std::vector<int> all_axes(size_t rank) {
  std::vector<int> axes;
  axes.reserve(rank);
  for (size_t axis = 0; axis < rank; ++axis) {
    axes.push_back(static_cast<int>(axis));
  }
  return axes;
}

bool axis_in_dimensions(int64_t axis, const std::vector<int64_t>& dimensions) {
  for (int64_t dim : dimensions) {
    if (dim == axis) {
      return true;
    }
  }
  return false;
}

bool gather_index_vector_is_explicit(const std::vector<int64_t>& indices_dims,
                                     size_t num_start_axes,
                                     int64_t index_vector_dim) {
  if (index_vector_dim < 0 ||
      static_cast<size_t>(index_vector_dim) >= indices_dims.size()) {
    return false;
  }
  return indices_dims[static_cast<size_t>(index_vector_dim)] ==
         static_cast<int64_t>(num_start_axes);
}

bool gather_metadata_is_valid(const std::vector<int64_t>& operand_dims,
                              const std::vector<int64_t>& indices_dims,
                              const std::vector<int64_t>& start_index_map,
                              const std::vector<int64_t>& collapsed_slice_dims,
                              const std::vector<int64_t>& operand_batching_dims,
                              const std::vector<int64_t>& start_indices_batching_dims,
                              int64_t index_vector_dim,
                              const std::vector<int64_t>& slice_sizes,
                              const std::vector<int64_t>& offset_dims,
                              const std::vector<int64_t>& output_dims) {
  if (start_index_map.empty() || slice_sizes.size() != operand_dims.size()) {
    return false;
  }
  if (start_index_map.size() > 1 &&
      !gather_index_vector_is_explicit(
          indices_dims, start_index_map.size(), index_vector_dim)) {
    return false;
  }
  if (operand_batching_dims.size() != start_indices_batching_dims.size()) {
    return false;
  }

  std::vector<bool> gathered_axes(operand_dims.size(), false);
  for (int64_t axis : start_index_map) {
    if (axis < 0 || static_cast<size_t>(axis) >= operand_dims.size() ||
        gathered_axes[static_cast<size_t>(axis)]) {
      return false;
    }
    gathered_axes[static_cast<size_t>(axis)] = true;
  }
  for (size_t i = 0; i < operand_batching_dims.size(); ++i) {
    const int64_t operand_axis = operand_batching_dims[i];
    const int64_t indices_axis = start_indices_batching_dims[i];
    if (operand_axis < 0 || static_cast<size_t>(operand_axis) >= operand_dims.size() ||
        indices_axis < 0 || static_cast<size_t>(indices_axis) >= indices_dims.size() ||
        indices_axis == index_vector_dim ||
        gathered_axes[static_cast<size_t>(operand_axis)] ||
        operand_dims[static_cast<size_t>(operand_axis)] !=
            indices_dims[static_cast<size_t>(indices_axis)] ||
        slice_sizes[static_cast<size_t>(operand_axis)] != 1) {
      return false;
    }
    gathered_axes[static_cast<size_t>(operand_axis)] = true;
  }

  std::vector<bool> collapsed_axes(operand_dims.size(), false);
  for (int64_t axis : collapsed_slice_dims) {
    if (axis < 0 || static_cast<size_t>(axis) >= operand_dims.size() ||
        collapsed_axes[static_cast<size_t>(axis)] ||
        slice_sizes[static_cast<size_t>(axis)] != 1) {
      return false;
    }
    collapsed_axes[static_cast<size_t>(axis)] = true;
  }
  for (int64_t axis : operand_batching_dims) {
    if (axis < 0 || static_cast<size_t>(axis) >= operand_dims.size() ||
        collapsed_axes[static_cast<size_t>(axis)]) {
      return false;
    }
    collapsed_axes[static_cast<size_t>(axis)] = true;
  }

  size_t non_collapsed_slice_rank = 0;
  for (size_t axis = 0; axis < operand_dims.size(); ++axis) {
    if (operand_dims[axis] < 0 || slice_sizes[axis] < 0 ||
        slice_sizes[axis] > operand_dims[axis]) {
      return false;
    }
    if (!collapsed_axes[axis]) {
      non_collapsed_slice_rank += 1;
    }
  }
  if (offset_dims.size() != non_collapsed_slice_rank) {
    return false;
  }

  const bool explicit_vector = gather_index_vector_is_explicit(
      indices_dims, start_index_map.size(), index_vector_dim);
  const size_t index_prefix_rank =
      indices_dims.size() - (explicit_vector ? 1 : 0);
  if (output_dims.size() != index_prefix_rank + non_collapsed_slice_rank) {
    return false;
  }

  std::vector<bool> output_is_offset(output_dims.size(), false);
  for (int64_t dim : offset_dims) {
    if (dim < 0 || static_cast<size_t>(dim) >= output_dims.size() ||
        output_is_offset[static_cast<size_t>(dim)]) {
      return false;
    }
    output_is_offset[static_cast<size_t>(dim)] = true;
  }

  size_t index_axis = 0;
  size_t slice_axis = 0;
  for (size_t output_axis = 0; output_axis < output_dims.size(); ++output_axis) {
    if (output_is_offset[output_axis]) {
      while (slice_axis < operand_dims.size() && collapsed_axes[slice_axis]) {
        slice_axis += 1;
      }
      if (slice_axis >= slice_sizes.size() ||
          output_dims[output_axis] != slice_sizes[slice_axis]) {
        return false;
      }
      slice_axis += 1;
    } else {
      while (explicit_vector && index_axis == static_cast<size_t>(index_vector_dim)) {
        index_axis += 1;
      }
      if (index_axis >= indices_dims.size() ||
          output_dims[output_axis] != indices_dims[index_axis]) {
        return false;
      }
      index_axis += 1;
    }
  }
  return true;
}

std::vector<int64_t> gather_index_prefix_dims(const std::vector<int64_t>& indices_dims,
                                              bool explicit_vector,
                                              int64_t index_vector_dim) {
  std::vector<int64_t> prefix;
  prefix.reserve(indices_dims.size());
  for (size_t axis = 0; axis < indices_dims.size(); ++axis) {
    if (explicit_vector && axis == static_cast<size_t>(index_vector_dim)) {
      continue;
    }
    prefix.push_back(indices_dims[axis]);
  }
  return prefix;
}

mlx::core::array gather_batch_index_array(
    const std::vector<int64_t>& indices_dims,
    const std::vector<int64_t>& index_prefix_dims, int64_t start_indices_axis,
    bool explicit_vector, int64_t index_vector_dim, int dtype,
    const mlx::core::Device& device) {
  const size_t prefix_axis = static_cast<size_t>(
      start_indices_axis - (explicit_vector && start_indices_axis > index_vector_dim ? 1 : 0));
  const mlx::core::Dtype* mlx_dtype = mlx_dtype_from_code(dtype);
  auto base = mlx::core::arange(
      0.0, static_cast<double>(indices_dims[static_cast<size_t>(start_indices_axis)]),
      1.0, *mlx_dtype, device);
  mlx::core::Shape reshape_dims(index_prefix_dims.size(), 1);
  reshape_dims[prefix_axis] =
      static_cast<mlx::core::ShapeElem>(index_prefix_dims[prefix_axis]);
  auto shaped = mlx::core::reshape(base, std::move(reshape_dims), device);
  return mlx::core::broadcast_to(shaped, mlx_shape(index_prefix_dims), device);
}

bool scatter_index_vector_is_explicit(const std::vector<int64_t>& indices_dims,
                                      size_t num_scatter_axes,
                                      int64_t index_vector_dim) {
  if (index_vector_dim < 0 ||
      static_cast<size_t>(index_vector_dim) >= indices_dims.size()) {
    return false;
  }
  return indices_dims[static_cast<size_t>(index_vector_dim)] ==
         static_cast<int64_t>(num_scatter_axes);
}

bool scatter_metadata_is_valid(
    const std::vector<int64_t>& operand_dims,
    const std::vector<int64_t>& indices_dims,
    const std::vector<int64_t>& update_dims,
    const std::vector<int64_t>& scatter_dims_to_operand_dims,
    const std::vector<int64_t>& inserted_window_dims,
    const std::vector<int64_t>& update_window_dims,
    const std::vector<int64_t>& input_batching_dims,
    const std::vector<int64_t>& scatter_indices_batching_dims,
    int64_t index_vector_dim,
    const std::vector<int64_t>& output_dims) {
  if (operand_dims.empty() || operand_dims != output_dims ||
      scatter_dims_to_operand_dims.empty() ||
      inserted_window_dims.size() + update_window_dims.size() +
              input_batching_dims.size() !=
          operand_dims.size() ||
      input_batching_dims.size() != scatter_indices_batching_dims.size()) {
    return false;
  }
  const bool explicit_vector = scatter_index_vector_is_explicit(
      indices_dims, scatter_dims_to_operand_dims.size(), index_vector_dim);
  if (scatter_dims_to_operand_dims.size() > 1 && !explicit_vector) {
    return false;
  }

  std::vector<bool> axes(operand_dims.size(), false);
  for (int64_t axis : scatter_dims_to_operand_dims) {
    if (axis < 0 || static_cast<size_t>(axis) >= operand_dims.size() ||
        axes[static_cast<size_t>(axis)]) {
      return false;
    }
    axes[static_cast<size_t>(axis)] = true;
  }
  for (size_t i = 0; i < input_batching_dims.size(); ++i) {
    const int64_t operand_axis = input_batching_dims[i];
    const int64_t indices_axis = scatter_indices_batching_dims[i];
    if (operand_axis < 0 || static_cast<size_t>(operand_axis) >= operand_dims.size() ||
        indices_axis < 0 || static_cast<size_t>(indices_axis) >= indices_dims.size() ||
        indices_axis == index_vector_dim ||
        axes[static_cast<size_t>(operand_axis)] ||
        operand_dims[static_cast<size_t>(operand_axis)] !=
            indices_dims[static_cast<size_t>(indices_axis)]) {
      return false;
    }
    axes[static_cast<size_t>(operand_axis)] = true;
  }
  std::vector<bool> inserted(operand_dims.size(), false);
  for (int64_t axis : inserted_window_dims) {
    if (axis < 0 || static_cast<size_t>(axis) >= operand_dims.size() ||
        inserted[static_cast<size_t>(axis)]) {
      return false;
    }
    inserted[static_cast<size_t>(axis)] = true;
  }
  std::vector<bool> update_window(operand_dims.size(), false);
  for (int64_t axis : update_window_dims) {
    if (axis < 0 || static_cast<size_t>(axis) >= operand_dims.size() ||
        inserted[static_cast<size_t>(axis)] ||
        update_window[static_cast<size_t>(axis)]) {
      return false;
    }
    update_window[static_cast<size_t>(axis)] = true;
  }
  for (int64_t axis : input_batching_dims) {
    if (axis < 0 || static_cast<size_t>(axis) >= operand_dims.size() ||
        inserted[static_cast<size_t>(axis)] ||
        update_window[static_cast<size_t>(axis)]) {
      return false;
    }
  }

  const size_t index_prefix_rank =
      indices_dims.size() - (explicit_vector ? 1 : 0);
  if (update_dims.size() != index_prefix_rank + update_window_dims.size()) {
    return false;
  }
  size_t index_axis = 0;
  for (size_t axis = 0; axis < indices_dims.size(); ++axis) {
    if (explicit_vector && axis == static_cast<size_t>(index_vector_dim)) {
      continue;
    }
    if (index_axis >= update_dims.size() || update_dims[index_axis] != indices_dims[axis]) {
      return false;
    }
    index_axis += 1;
  }
  if (index_axis != index_prefix_rank) {
    return false;
  }
  for (size_t window_axis = 0; window_axis < update_window_dims.size(); ++window_axis) {
    const int64_t operand_axis = update_window_dims[window_axis];
    const size_t update_axis = index_prefix_rank + window_axis;
    if (update_dims[update_axis] > operand_dims[static_cast<size_t>(operand_axis)]) {
      return false;
    }
  }
  return true;
}

std::unique_ptr<mlx::core::array> make_start_array(
    PjrtxMlxMetalBuffer* const* start_buffers, uint64_t rank,
    int device_ordinal, const mlx::core::Device& device) {
  if (start_buffers == nullptr || rank == 0) {
    return nullptr;
  }
  std::vector<mlx::core::array> parts;
  parts.reserve(rank);
  for (uint64_t i = 0; i < rank; ++i) {
    PjrtxMlxMetalBuffer* start = start_buffers[i];
    if (start == nullptr || start->array == nullptr || !start->dims.empty() ||
        start->device_ordinal != device_ordinal) {
      return nullptr;
    }
    auto casted = mlx::core::astype(*start->array, mlx::core::int32, device);
    parts.push_back(mlx::core::reshape(casted, mlx::core::Shape{1}, device));
  }
  auto out = parts.size() == 1
                 ? parts[0]
                 : mlx::core::concatenate(std::move(parts), 0, device);
  return std::make_unique<mlx::core::array>(std::move(out));
}

std::unique_ptr<mlx::core::array> make_mlx_array(const uint8_t* host,
                                                 uint64_t byte_size,
                                                 int dtype,
                                                 const std::vector<int64_t>& dims) {
  const auto shape = mlx_shape(dims);
  switch (dtype) {
    case PJRTX_MLX_METAL_DTYPE_PRED: {
      std::vector<uint8_t> values(byte_size);
      std::memcpy(values.data(), host, static_cast<size_t>(byte_size));
      return std::make_unique<mlx::core::array>(
          values.begin(), shape, mlx::core::bool_);
    }
    case PJRTX_MLX_METAL_DTYPE_U8: {
      std::vector<uint8_t> values(byte_size);
      std::memcpy(values.data(), host, static_cast<size_t>(byte_size));
      return std::make_unique<mlx::core::array>(
          values.begin(), shape, mlx::core::uint8);
    }
    case PJRTX_MLX_METAL_DTYPE_U16: {
      std::vector<uint16_t> values(byte_size / sizeof(uint16_t));
      std::memcpy(values.data(), host, static_cast<size_t>(byte_size));
      return std::make_unique<mlx::core::array>(
          values.begin(), shape, mlx::core::uint16);
    }
    case PJRTX_MLX_METAL_DTYPE_S8: {
      std::vector<int8_t> values(byte_size);
      std::memcpy(values.data(), host, static_cast<size_t>(byte_size));
      return std::make_unique<mlx::core::array>(
          values.begin(), shape, mlx::core::int8);
    }
    case PJRTX_MLX_METAL_DTYPE_S32: {
      std::vector<int32_t> values(byte_size / sizeof(int32_t));
      std::memcpy(values.data(), host, static_cast<size_t>(byte_size));
      return std::make_unique<mlx::core::array>(
          values.begin(), shape, mlx::core::int32);
    }
    case PJRTX_MLX_METAL_DTYPE_U32: {
      std::vector<uint32_t> values(byte_size / sizeof(uint32_t));
      std::memcpy(values.data(), host, static_cast<size_t>(byte_size));
      return std::make_unique<mlx::core::array>(
          values.begin(), shape, mlx::core::uint32);
    }
    case PJRTX_MLX_METAL_DTYPE_U64: {
      std::vector<uint64_t> values(byte_size / sizeof(uint64_t));
      std::memcpy(values.data(), host, static_cast<size_t>(byte_size));
      return std::make_unique<mlx::core::array>(
          values.begin(), shape, mlx::core::uint64);
    }
    case PJRTX_MLX_METAL_DTYPE_F16: {
      std::vector<mlx::core::float16_t> values(
          byte_size / sizeof(mlx::core::float16_t));
      std::memcpy(values.data(), host, static_cast<size_t>(byte_size));
      return std::make_unique<mlx::core::array>(
          values.begin(), shape, mlx::core::float16);
    }
    case PJRTX_MLX_METAL_DTYPE_BF16: {
      std::vector<mlx::core::bfloat16_t> values(
          byte_size / sizeof(mlx::core::bfloat16_t));
      std::memcpy(values.data(), host, static_cast<size_t>(byte_size));
      return std::make_unique<mlx::core::array>(
          values.begin(), shape, mlx::core::bfloat16);
    }
    case PJRTX_MLX_METAL_DTYPE_F32: {
      std::vector<float> values(byte_size / sizeof(float));
      std::memcpy(values.data(), host, static_cast<size_t>(byte_size));
      return std::make_unique<mlx::core::array>(
          values.begin(), shape, mlx::core::float32);
    }
    case PJRTX_MLX_METAL_DTYPE_C64: {
      std::vector<mlx::core::complex64_t> values(
          byte_size / sizeof(mlx::core::complex64_t));
      std::memcpy(values.data(), host, static_cast<size_t>(byte_size));
      return std::make_unique<mlx::core::array>(
          values.begin(), shape, mlx::core::complex64);
    }
    default:
      return nullptr;
  }
}

mlx::core::array mlx_astype_array(const mlx::core::array& src, int dtype,
                                  const mlx::core::Device& device) {
  switch (dtype) {
    case PJRTX_MLX_METAL_DTYPE_PRED:
      return mlx::core::astype(src, mlx::core::bool_, device);
    case PJRTX_MLX_METAL_DTYPE_U8:
      return mlx::core::astype(src, mlx::core::uint8, device);
    case PJRTX_MLX_METAL_DTYPE_U16:
      return mlx::core::astype(src, mlx::core::uint16, device);
    case PJRTX_MLX_METAL_DTYPE_S8:
      return mlx::core::astype(src, mlx::core::int8, device);
    case PJRTX_MLX_METAL_DTYPE_S32:
      return mlx::core::astype(src, mlx::core::int32, device);
    case PJRTX_MLX_METAL_DTYPE_U32:
      return mlx::core::astype(src, mlx::core::uint32, device);
    case PJRTX_MLX_METAL_DTYPE_U64:
      return mlx::core::astype(src, mlx::core::uint64, device);
    case PJRTX_MLX_METAL_DTYPE_F16:
      return mlx::core::astype(src, mlx::core::float16, device);
    case PJRTX_MLX_METAL_DTYPE_BF16:
      return mlx::core::astype(src, mlx::core::bfloat16, device);
    case PJRTX_MLX_METAL_DTYPE_F32:
      return mlx::core::astype(src, mlx::core::float32, device);
    case PJRTX_MLX_METAL_DTYPE_C64:
      return mlx::core::astype(src, mlx::core::complex64, device);
    default:
      return src;
  }
}

mlx::core::array attention_allowed_mask(PjrtxMlxMetalBuffer* token_index,
                                        int64_t queries, int64_t kv_len,
                                        const mlx::core::Device& device) {
  auto key_positions = mlx::core::reshape(
      mlx::core::arange(0, static_cast<int>(kv_len), device),
      mlx::core::Shape{1, 1, 1, static_cast<int>(kv_len)}, device);
  auto query_offsets = mlx::core::reshape(
      mlx::core::arange(0, static_cast<int>(queries), device),
      mlx::core::Shape{1, 1, static_cast<int>(queries), 1}, device);
  auto token = mlx_astype_array(*token_index->array, PJRTX_MLX_METAL_DTYPE_S32,
                                device);
  token = mlx::core::reshape(token, mlx::core::Shape{1, 1, 1, 1}, device);
  return mlx::core::less_equal(key_positions, token + query_offsets, device);
}

int launch_element_count(const mlx::core::array& src) {
  if (src.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::invalid_argument("custom Metal unary launch grid is too large");
  }
  return static_cast<int>(src.size());
}

mlx::core::array metal_kernel_input_array(const mlx::core::array& src,
                                          const mlx::core::Device& device) {
  if (src.ndim() != 0) {
    return src;
  }
  return mlx::core::reshape(src, mlx::core::Shape{1}, device);
}

bool msl_elementwise_dtype_supported(int dtype) {
  switch (dtype) {
    case PJRTX_MLX_METAL_DTYPE_F32:
    case PJRTX_MLX_METAL_DTYPE_F16:
    case PJRTX_MLX_METAL_DTYPE_BF16:
      return true;
    default:
      return false;
  }
}

const char* msl_binary_kernel_name(int op) {
  switch (op) {
    case PJRTX_MLX_METAL_U8_BINARY_ADD:
      return "pjrtx_msl_binary_add";
    case PJRTX_MLX_METAL_U8_BINARY_SUBTRACT:
      return "pjrtx_msl_binary_subtract";
    case PJRTX_MLX_METAL_U8_BINARY_MULTIPLY:
      return "pjrtx_msl_binary_multiply";
    case PJRTX_MLX_METAL_U8_BINARY_DIVIDE:
      return "pjrtx_msl_binary_divide";
    case PJRTX_MLX_METAL_BINARY_MAXIMUM:
      return "pjrtx_msl_binary_maximum";
    case PJRTX_MLX_METAL_BINARY_MINIMUM:
      return "pjrtx_msl_binary_minimum";
    default:
      return nullptr;
  }
}

const char* msl_binary_source(int op) {
  switch (op) {
    case PJRTX_MLX_METAL_U8_BINARY_ADD:
      return R"(
        uint elem = thread_position_in_grid.x;
        out[elem] = lhs[elem] + rhs[elem];
      )";
    case PJRTX_MLX_METAL_U8_BINARY_SUBTRACT:
      return R"(
        uint elem = thread_position_in_grid.x;
        out[elem] = lhs[elem] - rhs[elem];
      )";
    case PJRTX_MLX_METAL_U8_BINARY_MULTIPLY:
      return R"(
        uint elem = thread_position_in_grid.x;
        out[elem] = lhs[elem] * rhs[elem];
      )";
    case PJRTX_MLX_METAL_U8_BINARY_DIVIDE:
      return R"(
        uint elem = thread_position_in_grid.x;
        out[elem] = lhs[elem] / rhs[elem];
      )";
    case PJRTX_MLX_METAL_BINARY_MAXIMUM:
      return R"(
        uint elem = thread_position_in_grid.x;
        out[elem] = max(lhs[elem], rhs[elem]);
      )";
    case PJRTX_MLX_METAL_BINARY_MINIMUM:
      return R"(
        uint elem = thread_position_in_grid.x;
        out[elem] = min(lhs[elem], rhs[elem]);
      )";
    default:
      return nullptr;
  }
}

std::optional<mlx::core::array> try_msl_binary_array(
    const mlx::core::array& lhs, const mlx::core::array& rhs, int op,
    int dtype, const mlx::core::Shape& output_shape, uint64_t output_elements,
    const mlx::core::Device& device) {
  if (!msl_execute_enabled() || !msl_elementwise_dtype_supported(dtype)) {
    return std::nullopt;
  }
  const char* name = msl_binary_kernel_name(op);
  const char* source = msl_binary_source(op);
  const mlx::core::Dtype* mlx_dtype = mlx_dtype_from_code(dtype);
  if (name == nullptr || source == nullptr || mlx_dtype == nullptr) {
    return std::nullopt;
  }
  if (output_elements > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    return std::nullopt;
  }
  if (lhs.data_size() < output_elements || rhs.data_size() < output_elements) {
    return std::nullopt;
  }
  const auto lhs_input = metal_kernel_input_array(lhs, device);
  const auto rhs_input = metal_kernel_input_array(rhs, device);
  const auto kernel = mlx::core::fast::metal_kernel(
      name, {"lhs", "rhs"}, {"out"}, source);
  auto outputs = kernel(
      {lhs_input, rhs_input},
      {output_shape},
      {*mlx_dtype},
      {static_cast<int>(output_elements), 1, 1},
      {256, 1, 1},
      {},
      std::nullopt,
      false,
      device);
  if (outputs.size() != 1) {
    throw std::runtime_error("MSL binary custom kernel did not return one output");
  }
  return std::move(outputs[0]);
}

size_t xla_threefry_split_dim(const std::vector<int64_t>& dims) {
  if (dims.empty()) {
    return 0;
  }
  for (size_t i = 0; i < dims.size(); ++i) {
    if ((dims[i] % 2) == 0) {
      return i;
    }
  }
  return static_cast<size_t>(
      std::max_element(dims.begin(), dims.end()) - dims.begin());
}

uint64_t xla_threefry_half_count(const std::vector<int64_t>& dims,
                                 size_t split_dim) {
  if (dims.empty()) {
    return 1;
  }
  uint64_t count = 1;
  for (size_t i = 0; i < dims.size(); ++i) {
    const uint64_t dim = static_cast<uint64_t>(dims[i]);
    count *= i == split_dim ? (dim + 1) / 2 : dim;
  }
  return count;
}

std::string xla_threefry_kernel_source(const std::vector<int64_t>& dims,
                                       size_t split_dim,
                                       bool output_u64) {
  std::string source = R"(
    uint elem = thread_position_in_grid.x;
    uint lane = 0u;
    uint half_linear = 0u;
    uint half_stride = 1u;
  )";

  if (output_u64) {
    source += "half_linear = elem;\n";
  } else if (!dims.empty()) {
    source += "uint tmp = elem;\n";
    for (size_t reverse_index = dims.size(); reverse_index > 0; --reverse_index) {
      const size_t axis = reverse_index - 1;
      const uint64_t dim = static_cast<uint64_t>(dims[axis]);
      const uint64_t half_dim =
          axis == split_dim ? (dim + 1) / 2 : dim;
      source += "{ uint idx = tmp % " + std::to_string(dim) + "u; ";
      source += "tmp = tmp / " + std::to_string(dim) + "u; ";
      if (axis == split_dim) {
        source += "lane = idx & 1u; idx = idx >> 1u; ";
      }
      source += "half_linear += idx * half_stride; ";
      source += "half_stride *= " + std::to_string(half_dim) + "u; }\n";
    }
  }

  source += R"(
    ulong key64 = state[0];
    ulong counter64 = state[1] + ulong(half_linear);
    uint2 key = uint2(uint(key64), uint(key64 >> 32));
    uint2 input = uint2(uint(counter64), uint(counter64 >> 32));
    uint2 hash = pjrtx_threefry2x32(input, key);
  )";
  if (output_u64) {
    source += R"(
    bits[elem] = ulong(hash.x) | (ulong(hash.y) << 32);
    )";
  } else {
    source += R"(
    bits[elem] = lane == 0u ? hash.x : hash.y;
    )";
  }

  source += R"(
    if (elem == 0u) {
      ulong next_counter = state[1] + ulong(COUNTER_ADVANCE);
      state_out[0] = uint(key64);
      state_out[1] = uint(key64 >> 32);
      state_out[2] = uint(next_counter);
      state_out[3] = uint(next_counter >> 32);
    }
  )";
  return source;
}

const char* xla_threefry_kernel_header() {
  return R"(
    uint pjrtx_rotl32(uint v, uint distance) {
      return (v << distance) | (v >> (32u - distance));
    }

    uint2 pjrtx_threefry2x32(uint2 input, uint2 key) {
      uint3 ks = uint3(key.x, key.y, key.x ^ key.y ^ 0x1BD11BDAu);
      uint2 x = uint2(input.x + ks.x, input.y + ks.y);

      x.x += x.y; x.y = pjrtx_rotl32(x.y, 13u) ^ x.x;
      x.x += x.y; x.y = pjrtx_rotl32(x.y, 15u) ^ x.x;
      x.x += x.y; x.y = pjrtx_rotl32(x.y, 26u) ^ x.x;
      x.x += x.y; x.y = pjrtx_rotl32(x.y, 6u) ^ x.x;
      x.x += ks.y; x.y += ks.z + 1u;

      x.x += x.y; x.y = pjrtx_rotl32(x.y, 17u) ^ x.x;
      x.x += x.y; x.y = pjrtx_rotl32(x.y, 29u) ^ x.x;
      x.x += x.y; x.y = pjrtx_rotl32(x.y, 16u) ^ x.x;
      x.x += x.y; x.y = pjrtx_rotl32(x.y, 24u) ^ x.x;
      x.x += ks.z; x.y += ks.x + 2u;

      x.x += x.y; x.y = pjrtx_rotl32(x.y, 13u) ^ x.x;
      x.x += x.y; x.y = pjrtx_rotl32(x.y, 15u) ^ x.x;
      x.x += x.y; x.y = pjrtx_rotl32(x.y, 26u) ^ x.x;
      x.x += x.y; x.y = pjrtx_rotl32(x.y, 6u) ^ x.x;
      x.x += ks.x; x.y += ks.y + 3u;

      x.x += x.y; x.y = pjrtx_rotl32(x.y, 17u) ^ x.x;
      x.x += x.y; x.y = pjrtx_rotl32(x.y, 29u) ^ x.x;
      x.x += x.y; x.y = pjrtx_rotl32(x.y, 16u) ^ x.x;
      x.x += x.y; x.y = pjrtx_rotl32(x.y, 24u) ^ x.x;
      x.x += ks.y; x.y += ks.z + 4u;

      x.x += x.y; x.y = pjrtx_rotl32(x.y, 13u) ^ x.x;
      x.x += x.y; x.y = pjrtx_rotl32(x.y, 15u) ^ x.x;
      x.x += x.y; x.y = pjrtx_rotl32(x.y, 26u) ^ x.x;
      x.x += x.y; x.y = pjrtx_rotl32(x.y, 6u) ^ x.x;
      x.x += ks.z; x.y += ks.x + 5u;
      return x;
    }
  )";
}

bool xla_threefry_rng_bit_generator(PjrtxMlxMetalBuffer* state,
                                    int output_dtype,
                                    const std::vector<int64_t>& out_dims,
                                    const mlx::core::Device& device,
                                    PjrtxMlxMetalBuffer** out_state,
                                    PjrtxMlxMetalBuffer** out_bits) {
  if (state->dtype != PJRTX_MLX_METAL_DTYPE_U64 ||
      (output_dtype != PJRTX_MLX_METAL_DTYPE_U8 &&
       output_dtype != PJRTX_MLX_METAL_DTYPE_U16 &&
       output_dtype != PJRTX_MLX_METAL_DTYPE_U32 &&
       output_dtype != PJRTX_MLX_METAL_DTYPE_U64)) {
    return false;
  }
  const uint64_t elements = element_count_for_shape(out_dims);
  if (elements == 0 ||
      elements > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    return false;
  }
  const mlx::core::Dtype* bits_dtype = mlx_dtype_from_code(output_dtype);
  if (bits_dtype == nullptr) {
    return false;
  }

  auto state64 = state->array->dtype() == mlx::core::uint64
                     ? *state->array
                     : mlx::core::view(*state->array, mlx::core::uint64,
                                       device);
  if (state64.shape() != mlx::core::Shape{2}) {
    state64 = mlx::core::reshape(state64, {2}, device);
  }

  const bool output_u64 = output_dtype == PJRTX_MLX_METAL_DTYPE_U64;
  const size_t split_dim = output_u64 ? 0 : xla_threefry_split_dim(out_dims);
  const uint64_t counter_advance =
      output_u64 ? elements : xla_threefry_half_count(out_dims, split_dim);
  if (counter_advance > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    return false;
  }
  auto source = xla_threefry_kernel_source(out_dims, split_dim, output_u64);
  const auto kernel = mlx::core::fast::metal_kernel(
      "pjrtx_xla_threefry_rng_bit_generator",
      {"state"},
      {"bits", "state_out"},
      source,
      xla_threefry_kernel_header());
  auto outputs = kernel(
      {state64},
      {mlx_shape(out_dims), mlx::core::Shape{4}},
      {*bits_dtype, mlx::core::uint32},
      {static_cast<int>(elements), 1, 1},
      {256, 1, 1},
      {{"COUNTER_ADVANCE", static_cast<int>(counter_advance)}},
      std::nullopt,
      false,
      device);
  if (outputs.size() != 2) {
    return false;
  }

  const uint64_t state_byte_size = byte_size_for_shape(state->dtype, state->dims);
  const uint64_t bits_byte_size = byte_size_for_shape(output_dtype, out_dims);
  *out_bits = new PjrtxMlxMetalBuffer(
      std::make_unique<mlx::core::array>(std::move(outputs[0])),
      bits_byte_size, output_dtype, out_dims, state->device_ordinal);
  *out_state = new PjrtxMlxMetalBuffer(
      std::make_unique<mlx::core::array>(std::move(outputs[1])),
      state_byte_size, state->dtype, state->dims, state->device_ordinal);
  return true;
}

void copy_name(char* dst, const char* src) {
  if (dst == nullptr) {
    return;
  }
  if (src == nullptr) {
    src = "Metal device";
  }
  std::snprintf(dst, PJRTX_MLX_METAL_DEVICE_NAME_BYTES, "%s", src);
}

template <typename T>
T device_info_value(
    const std::unordered_map<std::string, std::variant<std::string, size_t>>&
        info,
    const char* key, T fallback) {
  const auto found = info.find(key);
  if (found == info.end()) {
    return fallback;
  }
  if (const auto* value = std::get_if<T>(&found->second)) {
    return *value;
  }
  return fallback;
}

void fill_device_info(PjrtxMlxMetalDeviceInfo* out, int ordinal) {
  const mlx::core::Device device(mlx::core::Device::gpu, ordinal);
  const auto& info = mlx::core::device_info(device);
  const auto name = device_info_value<std::string>(info, "device_name",
                                                   "MLX Metal device");
  const auto working_set = device_info_value<size_t>(
      info, "max_recommended_working_set_size", 0);

  out->ordinal = ordinal;
  out->registry_id = static_cast<uint64_t>(ordinal);
  out->recommended_max_working_set_size = static_cast<uint64_t>(working_set);
  out->has_unified_memory = 1;
  copy_name(out->name, name.c_str());
}

mlx::core::array mlx_binary_array(const mlx::core::array& lhs,
                                  const mlx::core::array& rhs, int op,
                                  int dtype,
                                  const mlx::core::Device& device) {
  switch (op) {
    case PJRTX_MLX_METAL_U8_BINARY_ADD:
      return mlx::core::add(lhs, rhs, device);
    case PJRTX_MLX_METAL_U8_BINARY_SUBTRACT:
      return mlx::core::subtract(lhs, rhs, device);
    case PJRTX_MLX_METAL_U8_BINARY_MULTIPLY:
      return mlx::core::multiply(lhs, rhs, device);
    case PJRTX_MLX_METAL_U8_BINARY_DIVIDE:
      return dtype == PJRTX_MLX_METAL_DTYPE_U8
                 ? mlx::core::floor_divide(lhs, rhs, device)
                 : mlx::core::divide(lhs, rhs, device);
    case PJRTX_MLX_METAL_BINARY_MAXIMUM:
      return mlx::core::maximum(lhs, rhs, device);
    case PJRTX_MLX_METAL_BINARY_MINIMUM:
      return mlx::core::minimum(lhs, rhs, device);
    case PJRTX_MLX_METAL_BINARY_POWER:
      return mlx::core::power(lhs, rhs, device);
    case PJRTX_MLX_METAL_BINARY_REMAINDER:
      return mlx::core::remainder(lhs, rhs, device);
    case PJRTX_MLX_METAL_BINARY_ATAN2:
      return mlx::core::arctan2(lhs, rhs, device);
    case PJRTX_MLX_METAL_BINARY_AND:
      return dtype == PJRTX_MLX_METAL_DTYPE_PRED
                 ? mlx::core::logical_and(lhs, rhs, device)
                 : mlx::core::bitwise_and(lhs, rhs, device);
    case PJRTX_MLX_METAL_BINARY_OR:
      return dtype == PJRTX_MLX_METAL_DTYPE_PRED
                 ? mlx::core::logical_or(lhs, rhs, device)
                 : mlx::core::bitwise_or(lhs, rhs, device);
    case PJRTX_MLX_METAL_BINARY_XOR:
      return mlx::core::bitwise_xor(lhs, rhs, device);
    case PJRTX_MLX_METAL_BINARY_SHIFT_LEFT:
      return mlx::core::left_shift(lhs, rhs, device);
    case PJRTX_MLX_METAL_BINARY_SHIFT_RIGHT:
      return mlx::core::right_shift(lhs, rhs, device);
    default:
      throw std::invalid_argument("unknown PjRTx MLX binary op");
  }
}

mlx::core::array mlx_unary_array(const mlx::core::array& src, int op,
                                 const mlx::core::Device& device) {
  switch (op) {
    case PJRTX_MLX_METAL_U8_UNARY_NEGATE:
      return mlx::core::negative(src, device);
    case PJRTX_MLX_METAL_UNARY_EXP:
      return mlx::core::exp(src, device);
    case PJRTX_MLX_METAL_UNARY_TANH:
      return mlx::core::tanh(src, device);
    case PJRTX_MLX_METAL_UNARY_SQRT:
      return mlx::core::sqrt(src, device);
    case PJRTX_MLX_METAL_UNARY_RSQRT:
      return mlx::core::rsqrt(src, device);
    case PJRTX_MLX_METAL_UNARY_ABS:
      return mlx::core::abs(src, device);
    case PJRTX_MLX_METAL_UNARY_CEIL:
      return mlx::core::ceil(src, device);
    case PJRTX_MLX_METAL_UNARY_FLOOR:
      return mlx::core::floor(src, device);
    case PJRTX_MLX_METAL_UNARY_LOG:
      return mlx::core::log(src, device);
    case PJRTX_MLX_METAL_UNARY_LOG1P:
      return mlx::core::log1p(src, device);
    case PJRTX_MLX_METAL_UNARY_LOGISTIC:
      return mlx::core::sigmoid(src, device);
    case PJRTX_MLX_METAL_UNARY_SIN:
      return mlx::core::sin(src, device);
    case PJRTX_MLX_METAL_UNARY_COS:
      return mlx::core::cos(src, device);
    case PJRTX_MLX_METAL_UNARY_SIGN:
      return mlx::core::sign(src, device);
    case PJRTX_MLX_METAL_UNARY_EXPM1:
      return mlx::core::expm1(src, device);
    case PJRTX_MLX_METAL_UNARY_NOT:
      return mlx::core::bitwise_invert(src, device);
    case PJRTX_MLX_METAL_UNARY_ISFINITE:
      return mlx::core::isfinite(src, device);
    case PJRTX_MLX_METAL_UNARY_ROUND:
      return mlx::core::round(src, device);
    case PJRTX_MLX_METAL_UNARY_CBRT: {
      const auto input = metal_kernel_input_array(src, device);
      const auto kernel = mlx::core::fast::metal_kernel(
          "pjrtx_unary_cbrt_f32",
          {"in"},
          {"out"},
          R"(
            uint elem = thread_position_in_grid.x;
            float x = static_cast<float>(in[elem]);
            float magnitude = pow(fabs(x), 0.3333333432674408f);
            out[elem] = x < 0.0f ? -magnitude : magnitude;
          )");
      auto outputs = kernel(
          {input},
          {src.shape()},
          {src.dtype()},
          {launch_element_count(src), 1, 1},
          {256, 1, 1},
          {},
          std::nullopt,
          false,
          device);
      if (outputs.size() != 1) {
        throw std::runtime_error("cbrt custom kernel did not return one output");
      }
      return std::move(outputs[0]);
    }
    case PJRTX_MLX_METAL_UNARY_ROUND_AFZ: {
      const auto input = metal_kernel_input_array(src, device);
      const auto kernel = mlx::core::fast::metal_kernel(
          "pjrtx_unary_round_afz_f32",
          {"in"},
          {"out"},
          R"(
            uint elem = thread_position_in_grid.x;
            float x = static_cast<float>(in[elem]);
            float magnitude = floor(fabs(x) + 0.5f);
            out[elem] = x < 0.0f ? -magnitude : magnitude;
          )");
      auto outputs = kernel(
          {input},
          {src.shape()},
          {src.dtype()},
          {launch_element_count(src), 1, 1},
          {256, 1, 1},
          {},
          std::nullopt,
          false,
          device);
      if (outputs.size() != 1) {
        throw std::runtime_error(
            "round_afz custom kernel did not return one output");
      }
      return std::move(outputs[0]);
    }
    case PJRTX_MLX_METAL_UNARY_POPCNT:
    case PJRTX_MLX_METAL_UNARY_CLZ: {
      const auto input = metal_kernel_input_array(src, device);
      const int bits = src.dtype() == mlx::core::uint8 ||
                               src.dtype() == mlx::core::int8
                           ? 8
                           : 32;
      const auto kernel = mlx::core::fast::metal_kernel(
          op == PJRTX_MLX_METAL_UNARY_POPCNT ? "pjrtx_unary_popcnt"
                                             : "pjrtx_unary_clz",
          {"in"},
          {"out"},
          op == PJRTX_MLX_METAL_UNARY_POPCNT
              ? R"(
                  uint elem = thread_position_in_grid.x;
                  uint raw = static_cast<uint>(in[elem]);
                  if (BITS == 8) {
                    raw = raw & 0xffu;
                  }
                  out[elem] = popcount(raw);
                )"
              : R"(
                  uint elem = thread_position_in_grid.x;
                  uint raw = static_cast<uint>(in[elem]);
                  if (BITS == 8) {
                    raw = raw & 0xffu;
                    out[elem] = raw == 0u ? 8u : clz(raw) - 24u;
                  } else {
                    out[elem] = clz(raw);
                  }
                )");
      auto outputs = kernel(
          {input},
          {src.shape()},
          {src.dtype()},
          {launch_element_count(src), 1, 1},
          {256, 1, 1},
          {{"BITS", bits}},
          std::nullopt,
          false,
          device);
      if (outputs.size() != 1) {
        throw std::runtime_error(
            "integer unary custom kernel did not return one output");
      }
      return std::move(outputs[0]);
    }
    default:
      throw std::invalid_argument("unknown PjRTx MLX unary op");
  }
}

int mlx_unary_output_dtype(int input_dtype, int op) {
  if (op == PJRTX_MLX_METAL_UNARY_ISFINITE) {
    return PJRTX_MLX_METAL_DTYPE_PRED;
  }
  if (op == PJRTX_MLX_METAL_UNARY_ABS &&
      input_dtype == PJRTX_MLX_METAL_DTYPE_C64) {
    return PJRTX_MLX_METAL_DTYPE_F32;
  }
  return input_dtype;
}

void eval_on_device(mlx::core::array& array, const mlx::core::Device& device) {
  if (!mlx::core::is_available(device)) {
    array.wait();
    return;
  }
  mlx::core::eval(array);
  mlx::core::synchronize(mlx::core::default_stream(device));
  array.wait();
}

bool dot_general_is_matmul_like(const std::vector<int64_t>& lhs_dims,
                                const std::vector<int64_t>& rhs_dims,
                                const std::vector<int64_t>& lhs_batch,
                                const std::vector<int64_t>& rhs_batch,
                                const std::vector<int64_t>& lhs_contract,
                                const std::vector<int64_t>& rhs_contract,
                                const std::vector<int64_t>& output_dims) {
  if (lhs_contract.size() != 1 || rhs_contract.size() != 1 ||
      lhs_batch.size() != rhs_batch.size() || lhs_dims.empty() ||
      rhs_dims.size() < 2 || output_dims.empty()) {
    return false;
  }
  const int64_t lhs_k = lhs_contract[0];
  const int64_t rhs_k = rhs_contract[0];
  if (lhs_k < 0 || rhs_k < 0 ||
      static_cast<size_t>(lhs_k) >= lhs_dims.size() ||
      static_cast<size_t>(rhs_k) >= rhs_dims.size()) {
    return false;
  }
  std::vector<bool> lhs_used(lhs_dims.size(), false);
  std::vector<bool> rhs_used(rhs_dims.size(), false);
  lhs_used[static_cast<size_t>(lhs_k)] = true;
  rhs_used[static_cast<size_t>(rhs_k)] = true;
  if (lhs_dims[static_cast<size_t>(lhs_k)] !=
      rhs_dims[static_cast<size_t>(rhs_k)]) {
    return false;
  }
  for (size_t i = 0; i < lhs_batch.size(); ++i) {
    if (lhs_batch[i] < 0 || rhs_batch[i] < 0 ||
        static_cast<size_t>(lhs_batch[i]) >= lhs_dims.size() ||
        static_cast<size_t>(rhs_batch[i]) >= rhs_dims.size() ||
        lhs_used[static_cast<size_t>(lhs_batch[i])] ||
        rhs_used[static_cast<size_t>(rhs_batch[i])] ||
        lhs_dims[static_cast<size_t>(lhs_batch[i])] !=
            rhs_dims[static_cast<size_t>(rhs_batch[i])]) {
      return false;
    }
    lhs_used[static_cast<size_t>(lhs_batch[i])] = true;
    rhs_used[static_cast<size_t>(rhs_batch[i])] = true;
  }
  std::vector<int64_t> expected_output;
  expected_output.reserve(output_dims.size());
  for (int64_t axis : lhs_batch) {
    expected_output.push_back(lhs_dims[static_cast<size_t>(axis)]);
  }
  for (size_t axis = 0; axis < lhs_dims.size(); ++axis) {
    if (!lhs_used[axis]) {
      expected_output.push_back(lhs_dims[axis]);
    }
  }
  for (size_t axis = 0; axis < rhs_dims.size(); ++axis) {
    if (!rhs_used[axis]) {
      expected_output.push_back(rhs_dims[axis]);
    }
  }
  if (expected_output != output_dims) {
    return false;
  }
  return true;
}

bool dot_general_is_canonical_matmul(const std::vector<int64_t>& lhs_dims,
                                     const std::vector<int64_t>& rhs_dims,
                                     const std::vector<int64_t>& lhs_batch,
                                     const std::vector<int64_t>& rhs_batch,
                                     const std::vector<int64_t>& lhs_contract,
                                     const std::vector<int64_t>& rhs_contract,
                                     const std::vector<int64_t>& output_dims) {
  if (lhs_contract.size() != 1 || rhs_contract.size() != 1 ||
      lhs_batch.size() != rhs_batch.size() || lhs_dims.size() < 2) {
    return false;
  }
  const size_t batch_rank = lhs_batch.size();
  if (lhs_dims.size() < batch_rank + 2 ||
      rhs_dims.size() != batch_rank + 2 ||
      output_dims.size() != lhs_dims.size()) {
    return false;
  }
  if (batch_rank != 0 && lhs_dims.size() != batch_rank + 2) {
    return false;
  }
  for (size_t axis = 0; axis < batch_rank; ++axis) {
    if (lhs_batch[axis] != static_cast<int64_t>(axis) ||
        rhs_batch[axis] != static_cast<int64_t>(axis) ||
        lhs_dims[axis] != rhs_dims[axis] ||
        output_dims[axis] != lhs_dims[axis]) {
      return false;
      }
  }
  if (lhs_contract[0] != static_cast<int64_t>(lhs_dims.size() - 1) ||
      rhs_contract[0] != static_cast<int64_t>(batch_rank) ||
      lhs_dims.back() != rhs_dims[batch_rank]) {
    return false;
  }
  for (size_t axis = batch_rank; axis + 1 < lhs_dims.size(); ++axis) {
    if (output_dims[axis] != lhs_dims[axis]) return false;
  }
  if (output_dims.back() != rhs_dims[batch_rank + 1]) return false;
  return true;
}

bool dot_general_is_rhs_transposed_matmul(
    const std::vector<int64_t>& lhs_dims, const std::vector<int64_t>& rhs_dims,
    const std::vector<int64_t>& lhs_batch,
    const std::vector<int64_t>& rhs_batch,
    const std::vector<int64_t>& lhs_contract,
    const std::vector<int64_t>& rhs_contract,
    const std::vector<int64_t>& output_dims) {
  if (!lhs_batch.empty() || !rhs_batch.empty() ||
      lhs_contract.size() != 1 || rhs_contract.size() != 1 ||
      lhs_dims.size() < 2 || rhs_dims.size() != 2 ||
      output_dims.size() != lhs_dims.size()) {
    return false;
  }
  if (lhs_contract[0] != static_cast<int64_t>(lhs_dims.size() - 1) ||
      rhs_contract[0] != 1 || lhs_dims.back() != rhs_dims[1]) {
    return false;
  }
  for (size_t axis = 0; axis + 1 < lhs_dims.size(); ++axis) {
    if (output_dims[axis] != lhs_dims[axis]) return false;
  }
  if (output_dims.back() != rhs_dims[0]) return false;
  return true;
}

void record_dot_general_path(const char* path) {
  if (!dot_profile_enabled()) return;
  uint64_t canonical = g_dot_canonical_count.load(std::memory_order_relaxed);
  uint64_t rhs_transposed =
      g_dot_rhs_transposed_count.load(std::memory_order_relaxed);
  uint64_t normalized = g_dot_normalized_count.load(std::memory_order_relaxed);
  if (std::strcmp(path, "canonical") == 0) {
    canonical = g_dot_canonical_count.fetch_add(1, std::memory_order_relaxed) + 1;
  } else if (std::strcmp(path, "rhs_transposed") == 0) {
    rhs_transposed =
        g_dot_rhs_transposed_count.fetch_add(1, std::memory_order_relaxed) + 1;
  } else {
    normalized = g_dot_normalized_count.fetch_add(1, std::memory_order_relaxed) + 1;
  }
  const uint64_t total = canonical + rhs_transposed + normalized;
  if ((total % 64) == 0) {
    std::fprintf(stderr,
                 "pjrtx_profile event=dot_general_paths total=%llu canonical=%llu rhs_transposed=%llu normalized=%llu\n",
                 static_cast<unsigned long long>(total),
                 static_cast<unsigned long long>(canonical),
                 static_cast<unsigned long long>(rhs_transposed),
                 static_cast<unsigned long long>(normalized));
  }
}

std::string i64_vector_string(const std::vector<int64_t>& values) {
  std::string out = "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i != 0) out += ",";
    out += std::to_string(values[i]);
  }
  out += "]";
  return out;
}

void record_dot_general_shape(const std::vector<int64_t>& lhs_dims,
                              const std::vector<int64_t>& rhs_dims,
                              const std::vector<int64_t>& lhs_batch,
                              const std::vector<int64_t>& rhs_batch,
                              const std::vector<int64_t>& lhs_contract,
                              const std::vector<int64_t>& rhs_contract,
                              const std::vector<int64_t>& output_dims) {
  if (!dot_profile_enabled()) return;
  const uint64_t sample =
      g_dot_shape_sample_count.fetch_add(1, std::memory_order_relaxed);
  if (sample >= 32) return;
  std::fprintf(
      stderr,
      "pjrtx_profile event=dot_general_shape sample=%llu lhs=%s rhs=%s lhs_batch=%s rhs_batch=%s lhs_contract=%s rhs_contract=%s output=%s\n",
      static_cast<unsigned long long>(sample),
      i64_vector_string(lhs_dims).c_str(), i64_vector_string(rhs_dims).c_str(),
      i64_vector_string(lhs_batch).c_str(), i64_vector_string(rhs_batch).c_str(),
      i64_vector_string(lhs_contract).c_str(),
      i64_vector_string(rhs_contract).c_str(),
      i64_vector_string(output_dims).c_str());
}

mlx::core::array mlx_compare_array(const mlx::core::array& lhs,
                                   const mlx::core::array& rhs, int direction,
                                   const mlx::core::Device& device) {
  switch (direction) {
    case PJRTX_MLX_METAL_COMPARE_EQ:
      return mlx::core::equal(lhs, rhs, device);
    case PJRTX_MLX_METAL_COMPARE_NE:
      return mlx::core::not_equal(lhs, rhs, device);
    case PJRTX_MLX_METAL_COMPARE_GE:
      return mlx::core::greater_equal(lhs, rhs, device);
    case PJRTX_MLX_METAL_COMPARE_GT:
      return mlx::core::greater(lhs, rhs, device);
    case PJRTX_MLX_METAL_COMPARE_LE:
      return mlx::core::less_equal(lhs, rhs, device);
    case PJRTX_MLX_METAL_COMPARE_LT:
      return mlx::core::less(lhs, rhs, device);
    default:
      throw std::invalid_argument("unknown PjRTx MLX compare op");
  }
}

}  // namespace

int pjrtx_mlx_metal_version_major(void) { return MLX_VERSION_MAJOR; }

int pjrtx_mlx_metal_version_minor(void) { return MLX_VERSION_MINOR; }

int pjrtx_mlx_metal_version_patch(void) { return MLX_VERSION_PATCH; }

int pjrtx_mlx_metal_has_upstream_mlx_metal_api(void) {
  return sizeof(&mlx::core::metal::is_available) > 0 ? 1 : 0;
}

int pjrtx_mlx_metal_device_count(void) {
  try {
    return mlx::core::device_count(mlx::core::Device::gpu);
  } catch (const std::exception& e) {
    if (std::getenv("PJRTX_TRACE") != nullptr) {
      std::fprintf(stderr, "pjrtx_trace event=mlx_exception op=eval_many what=\"%s\"\n", e.what());
    }
    return 0;
  } catch (...) {
    if (std::getenv("PJRTX_TRACE") != nullptr) {
      std::fprintf(stderr, "pjrtx_trace event=mlx_exception op=eval_many what=\"unknown\"\n");
    }
    return 0;
  }
}

int pjrtx_mlx_metal_copy_devices(PjrtxMlxMetalDeviceInfo* out_devices,
                                 int max_devices) {
  if (out_devices == nullptr || max_devices <= 0) {
    return 0;
  }

  try {
    const int available = mlx::core::device_count(mlx::core::Device::gpu);
    const int copied = available < max_devices ? available : max_devices;
    for (int i = 0; i < copied; ++i) {
      fill_device_info(&out_devices[i], i);
    }
    return copied;
  } catch (const std::exception&) {
    return 0;
  } catch (...) {
    return 0;
  }
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_from_host(int device_ordinal,
                                                      const void* data,
                                                      uint64_t byte_size) {
  int64_t dim = static_cast<int64_t>(byte_size);
  return pjrtx_mlx_metal_buffer_from_host_typed(
      device_ordinal, data, byte_size, PJRTX_MLX_METAL_DTYPE_U8, &dim, 1);
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_from_host_typed(
    int device_ordinal, const void* data, uint64_t byte_size, int dtype,
    const int64_t* dims, uint64_t rank) {
  if (data == nullptr || byte_size == 0) {
    return nullptr;
  }
  if (rank > 0 && dims == nullptr) {
    return nullptr;
  }

  const bool profile = profile_enabled();
  const uint64_t start_us = profile ? now_micros() : 0;
  try {
    std::vector<int64_t> shape;
    if (rank == 0) {
      shape = {};
    } else {
      shape.assign(dims, dims + rank);
    }
    if (!byte_size_matches_shape(byte_size, dtype, shape)) {
      return nullptr;
    }

    const auto* bytes = static_cast<const uint8_t*>(data);
    std::unique_ptr<mlx::core::array> array;
    try {
      array = make_mlx_array(bytes, byte_size, dtype, shape);
    } catch (const std::exception& e) {
      if (std::getenv("PJRTX_TRACE") != nullptr) {
        std::fprintf(stderr, "pjrtx_trace event=mlx_exception op=from_host_make_array dtype=%d rank=%llu bytes=%llu what=\"%s\"\n",
                     dtype, static_cast<unsigned long long>(rank),
                     static_cast<unsigned long long>(byte_size), e.what());
      }
      array.reset();
    } catch (...) {
      if (std::getenv("PJRTX_TRACE") != nullptr) {
        std::fprintf(stderr, "pjrtx_trace event=mlx_exception op=from_host_make_array dtype=%d rank=%llu bytes=%llu what=\"unknown\"\n",
                     dtype, static_cast<unsigned long long>(rank),
                     static_cast<unsigned long long>(byte_size));
      }
      array.reset();
    }
    if (array == nullptr) {
      return nullptr;
    }
    if (profile) {
      std::fprintf(stderr,
                   "pjrtx_profile event=mlx_from_host device=%d dtype=%d rank=%llu bytes=%llu elapsed_us=%llu\n",
                   device_ordinal, dtype,
                   static_cast<unsigned long long>(rank),
                   static_cast<unsigned long long>(byte_size),
                   static_cast<unsigned long long>(elapsed_micros_since(start_us)));
    }
    return new PjrtxMlxMetalBuffer(std::move(array), byte_size, dtype,
                                   std::move(shape), device_ordinal);
  } catch (const std::exception& e) {
    if (std::getenv("PJRTX_TRACE") != nullptr) {
      std::fprintf(stderr, "pjrtx_trace event=mlx_exception op=from_host dtype=%d rank=%llu bytes=%llu what=\"%s\"\n",
                   dtype, static_cast<unsigned long long>(rank),
                   static_cast<unsigned long long>(byte_size), e.what());
    }
    return nullptr;
  } catch (...) {
    if (std::getenv("PJRTX_TRACE") != nullptr) {
      std::fprintf(stderr, "pjrtx_trace event=mlx_exception op=from_host dtype=%d rank=%llu bytes=%llu what=\"unknown\"\n",
                   dtype, static_cast<unsigned long long>(rank),
                   static_cast<unsigned long long>(byte_size));
    }
    return nullptr;
  }
}

PjrtxMlxMetalAsyncHostToDeviceTransfer*
pjrtx_mlx_metal_async_h2d_create(int device_ordinal, int dtype,
                                 const int64_t* dims, uint64_t rank,
                                 uint64_t byte_size) {
  if (byte_size == 0 || (rank > 0 && dims == nullptr)) {
    return nullptr;
  }
  try {
    std::vector<int64_t> shape;
    if (rank != 0) {
      shape.assign(dims, dims + rank);
    }
    if (!byte_size_matches_shape(byte_size, dtype, shape)) {
      return nullptr;
    }
    const mlx::core::Device device(mlx::core::Device::gpu, device_ordinal);
    if (!mlx::core::is_available(device)) {
      return nullptr;
    }
    auto staging = mlx::core::allocator::malloc(static_cast<size_t>(byte_size));
    if (staging.ptr() == nullptr) {
      return nullptr;
    }
    return new PjrtxMlxMetalAsyncHostToDeviceTransfer(
        staging, byte_size, dtype, std::move(shape), device_ordinal);
  } catch (...) {
    return nullptr;
  }
}

int pjrtx_mlx_metal_async_h2d_write(
    PjrtxMlxMetalAsyncHostToDeviceTransfer* transfer, uint64_t offset,
    const void* data, uint64_t byte_size) {
  if (transfer == nullptr || transfer->finished ||
      transfer->staging.raw_ptr() == nullptr || (byte_size != 0 && data == nullptr) ||
      offset > transfer->byte_size || byte_size > transfer->byte_size - offset) {
    return 0;
  }
  if (byte_size != 0) {
    auto* dst = static_cast<uint8_t*>(transfer->staging.raw_ptr()) + offset;
    std::memcpy(dst, data, static_cast<size_t>(byte_size));
  }
  return 1;
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_async_h2d_finish(
    PjrtxMlxMetalAsyncHostToDeviceTransfer* transfer) {
  if (transfer == nullptr || transfer->finished ||
      transfer->staging.ptr() == nullptr) {
    return nullptr;
  }
  const bool profile = profile_enabled();
  const uint64_t start_us = profile ? now_micros() : 0;
  try {
    const mlx::core::Dtype* dtype = mlx_dtype_from_code(transfer->dtype);
    if (dtype == nullptr) {
      return nullptr;
    }
    auto array = std::make_unique<mlx::core::array>(
        transfer->staging, mlx_shape(transfer->dims), *dtype);
    transfer->finished = true;
    transfer->staging = mlx::core::allocator::Buffer{nullptr};
    if (profile) {
      std::fprintf(stderr,
                   "pjrtx_profile event=mlx_async_h2d_finish device=%d dtype=%d rank=%llu bytes=%llu elapsed_us=%llu\n",
                   transfer->device_ordinal, transfer->dtype,
                   static_cast<unsigned long long>(transfer->dims.size()),
                   static_cast<unsigned long long>(transfer->byte_size),
                   static_cast<unsigned long long>(elapsed_micros_since(start_us)));
    }
    return new PjrtxMlxMetalBuffer(std::move(array), transfer->byte_size,
                                   transfer->dtype, transfer->dims,
                                   transfer->device_ordinal);
  } catch (const std::exception& e) {
    if (std::getenv("PJRTX_TRACE") != nullptr) {
      std::fprintf(stderr,
                   "pjrtx_trace event=mlx_exception op=async_h2d_finish dtype=%d rank=%llu bytes=%llu what=\"%s\"\n",
                   transfer->dtype,
                   static_cast<unsigned long long>(transfer->dims.size()),
                   static_cast<unsigned long long>(transfer->byte_size),
                   e.what());
    }
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

void pjrtx_mlx_metal_async_h2d_destroy(
    PjrtxMlxMetalAsyncHostToDeviceTransfer* transfer) {
  if (transfer == nullptr) {
    return;
  }
  if (!transfer->finished && transfer->staging.ptr() != nullptr) {
    mlx::core::allocator::free(transfer->staging);
    transfer->staging = mlx::core::allocator::Buffer{nullptr};
  }
  delete transfer;
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_iota(
    int device_ordinal, int dtype, const int64_t* dims, uint64_t rank,
    int64_t iota_dimension) {
  if (dims == nullptr || iota_dimension < 0 ||
      static_cast<uint64_t>(iota_dimension) >= rank) {
    return nullptr;
  }
  std::vector<int64_t> out_dims;
  if (rank > 0) {
    out_dims.assign(dims, dims + rank);
  }
  const uint64_t byte_size = byte_size_for_shape(dtype, out_dims);
  const mlx::core::Dtype* mlx_dtype = mlx_dtype_from_code(dtype);
  if (byte_size == 0 || mlx_dtype == nullptr) {
    return nullptr;
  }
  for (int64_t dim : out_dims) {
    if (dim < 0) {
      return nullptr;
    }
  }

  const mlx::core::Device device(mlx::core::Device::gpu, device_ordinal);
  if (!mlx::core::is_available(device)) {
    return nullptr;
  }

  try {
    const size_t axis = static_cast<size_t>(iota_dimension);
    auto base = mlx::core::arange(0.0, static_cast<double>(out_dims[axis]),
                                  1.0, *mlx_dtype, device);
    mlx::core::Shape reshape_dims(rank, 1);
    reshape_dims[axis] = static_cast<mlx::core::ShapeElem>(out_dims[axis]);
    auto shaped = mlx::core::reshape(base, std::move(reshape_dims), device);
    auto out = mlx::core::broadcast_to(shaped, mlx_shape(out_dims), device);
    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(std::move(array), byte_size, dtype,
                                   std::move(out_dims), device_ordinal);
  } catch (const std::exception&) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_partition_id(
    int device_ordinal, int dtype, uint32_t partition_id) {
  if (dtype != PJRTX_MLX_METAL_DTYPE_U32 &&
      dtype != PJRTX_MLX_METAL_DTYPE_S32) {
    return nullptr;
  }

  const mlx::core::Device device(mlx::core::Device::gpu, device_ordinal);
  if (!mlx::core::is_available(device)) {
    return nullptr;
  }

  try {
    mlx::core::array scalar =
        dtype == PJRTX_MLX_METAL_DTYPE_U32
            ? mlx::core::array(partition_id, mlx::core::uint32)
            : mlx::core::array(static_cast<int32_t>(partition_id),
                               mlx::core::int32);
    auto out = mlx::core::full(mlx::core::Shape{}, scalar, device);
    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(std::move(array), dtype_size(dtype), dtype,
                                   std::vector<int64_t>{}, device_ordinal);
  } catch (const std::exception&) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_clone(PjrtxMlxMetalBuffer* src) {
  if (src == nullptr || src->byte_size == 0) {
    return nullptr;
  }

  try {
    if (src->array == nullptr) {
      return nullptr;
    }
    auto array = std::make_unique<mlx::core::array>(*src->array);
    return new PjrtxMlxMetalBuffer(std::move(array), src->byte_size,
                                   src->dtype, src->dims,
                                   src->device_ordinal);
  } catch (const std::exception&) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_zero_like(
    PjrtxMlxMetalBuffer* src) {
  if (src == nullptr || src->byte_size == 0 || src->array == nullptr) {
    return nullptr;
  }
  const mlx::core::Dtype* dtype = mlx_dtype_from_code(src->dtype);
  if (dtype == nullptr) {
    return nullptr;
  }

  const mlx::core::Device device(mlx::core::Device::gpu, src->device_ordinal);
  if (!mlx::core::is_available(device)) {
    return nullptr;
  }

  try {
    auto out = mlx::core::zeros(mlx_shape(src->dims), *dtype, device);
    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(std::move(array), src->byte_size,
                                   src->dtype, src->dims,
                                   src->device_ordinal);
  } catch (const std::exception&) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_zeros(
    int device_ordinal, int dtype, const int64_t* dims, uint64_t rank) {
  if (rank > 0 && dims == nullptr) {
    return nullptr;
  }
  const mlx::core::Dtype* mlx_dtype = mlx_dtype_from_code(dtype);
  if (mlx_dtype == nullptr) {
    return nullptr;
  }
  std::vector<int64_t> out_dims(dims, dims + rank);
  const uint64_t byte_size = byte_size_for_shape(dtype, out_dims);
  if (byte_size == 0) {
    return nullptr;
  }
  const mlx::core::Device device(mlx::core::Device::gpu, device_ordinal);
  if (!mlx::core::is_available(device)) {
    return nullptr;
  }
  try {
    auto out = mlx::core::zeros(mlx_shape(out_dims), *mlx_dtype, device);
    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(std::move(array), byte_size, dtype,
                                   std::move(out_dims), device_ordinal);
  } catch (const std::exception&) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_complex(
    PjrtxMlxMetalBuffer* real, PjrtxMlxMetalBuffer* imag,
    const int64_t* output_dims, uint64_t output_rank) {
  if (real == nullptr || imag == nullptr || real->array == nullptr ||
      imag->array == nullptr || real->byte_size == 0 || imag->byte_size == 0 ||
      output_dims == nullptr || real->dtype != PJRTX_MLX_METAL_DTYPE_F32 ||
      imag->dtype != PJRTX_MLX_METAL_DTYPE_F32 ||
      real->device_ordinal != imag->device_ordinal ||
      output_rank != real->dims.size() || output_rank != imag->dims.size()) {
    return nullptr;
  }

  std::vector<int64_t> out_dims;
  if (output_rank != 0) {
    if (output_dims == nullptr) {
      return nullptr;
    }
    out_dims.assign(output_dims, output_dims + output_rank);
  }
  if (out_dims != real->dims || out_dims != imag->dims ||
      real->byte_size != imag->byte_size) {
    return nullptr;
  }

  const uint64_t byte_size =
      byte_size_for_shape(PJRTX_MLX_METAL_DTYPE_C64, out_dims);
  if (byte_size == 0 || byte_size != real->byte_size * 2) {
    return nullptr;
  }

  const mlx::core::Device device(mlx::core::Device::gpu, real->device_ordinal);
  if (!mlx::core::is_available(device)) {
    return nullptr;
  }

  try {
    auto real_c = mlx::core::astype(*real->array, mlx::core::complex64, device);
    auto imag_c = mlx::core::astype(*imag->array, mlx::core::complex64, device);
    mlx::core::array imaginary_unit(
        mlx::core::complex64_t{0.0f, 1.0f}, mlx::core::complex64);
    auto out = mlx::core::add(
        real_c, mlx::core::multiply(imag_c, imaginary_unit, device), device);
    if (out.shape() != mlx_shape(out_dims)) {
      out = mlx::core::reshape(out, mlx_shape(out_dims), device);
    }
    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(std::move(array), byte_size,
                                   PJRTX_MLX_METAL_DTYPE_C64,
                                   std::move(out_dims), real->device_ordinal);
  } catch (const std::exception&) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_real(
    PjrtxMlxMetalBuffer* src, const int64_t* output_dims,
    uint64_t output_rank) {
  if (src == nullptr || src->array == nullptr || src->byte_size == 0 ||
      output_dims == nullptr || output_rank != src->dims.size()) {
    return nullptr;
  }

  std::vector<int64_t> out_dims(output_dims, output_dims + output_rank);
  if (out_dims != src->dims) {
    return nullptr;
  }

  const uint64_t byte_size =
      byte_size_for_shape(PJRTX_MLX_METAL_DTYPE_F32, out_dims);
  if (byte_size == 0) {
    return nullptr;
  }

  const mlx::core::Device device(mlx::core::Device::gpu, src->device_ordinal);
  if (!mlx::core::is_available(device)) {
    return nullptr;
  }

  try {
    auto out = mlx::core::real(*src->array, device);
    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(std::move(array), byte_size,
                                   PJRTX_MLX_METAL_DTYPE_F32, std::move(out_dims),
                                   src->device_ordinal);
  } catch (const std::exception&) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_imag(
    PjrtxMlxMetalBuffer* src, const int64_t* output_dims,
    uint64_t output_rank) {
  if (src == nullptr || src->array == nullptr || src->byte_size == 0 ||
      output_dims == nullptr || output_rank != src->dims.size()) {
    return nullptr;
  }

  std::vector<int64_t> out_dims(output_dims, output_dims + output_rank);
  if (out_dims != src->dims) {
    return nullptr;
  }

  const uint64_t byte_size =
      byte_size_for_shape(PJRTX_MLX_METAL_DTYPE_F32, out_dims);
  if (byte_size == 0) {
    return nullptr;
  }

  const mlx::core::Device device(mlx::core::Device::gpu, src->device_ordinal);
  if (!mlx::core::is_available(device)) {
    return nullptr;
  }

  try {
    auto out = mlx::core::imag(*src->array, device);
    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(std::move(array), byte_size,
                                   PJRTX_MLX_METAL_DTYPE_F32, std::move(out_dims),
                                   src->device_ordinal);
  } catch (const std::exception&) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_astype(
    PjrtxMlxMetalBuffer* src, int dtype) {
  if (src == nullptr || src->byte_size == 0 || src->array == nullptr ||
      dtype_size(dtype) == 0) {
    return nullptr;
  }

  const uint64_t byte_size = byte_size_for_shape(dtype, src->dims);
  if (byte_size == 0) {
    return nullptr;
  }

  const mlx::core::Device device(mlx::core::Device::gpu, src->device_ordinal);
  if (!mlx::core::is_available(device)) {
    return nullptr;
  }

  try {
    auto out = mlx_astype_array(*src->array, dtype, device);
    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(std::move(array), byte_size, dtype,
                                   src->dims, src->device_ordinal);
  } catch (const std::exception&) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_view_dtype(
    PjrtxMlxMetalBuffer* src, int dtype, const int64_t* dims, uint64_t rank) {
  if (src == nullptr || src->byte_size == 0 || src->array == nullptr ||
      dtype_size(dtype) == 0 || (rank > 0 && dims == nullptr)) {
    return nullptr;
  }

  std::vector<int64_t> out_dims;
  if (rank > 0) {
    out_dims.assign(dims, dims + rank);
  }
  const uint64_t byte_size = byte_size_for_shape(dtype, out_dims);
  if (byte_size == 0 || byte_size != src->byte_size) {
    return nullptr;
  }

  const mlx::core::Dtype* out_dtype = mlx_dtype_from_code(dtype);
  if (out_dtype == nullptr) {
    return nullptr;
  }

  const mlx::core::Device device(mlx::core::Device::gpu, src->device_ordinal);
  if (!mlx::core::is_available(device)) {
    return nullptr;
  }

  try {
    auto out = mlx::core::view(*src->array, *out_dtype, device);
    if (out.shape() != mlx_shape(out_dims)) {
      out = mlx::core::reshape(out, mlx_shape(out_dims), device);
    }
    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(std::move(array), byte_size, dtype,
                                   out_dims, src->device_ordinal);
  } catch (const std::exception&) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_add_u8(
    PjrtxMlxMetalBuffer* lhs, PjrtxMlxMetalBuffer* rhs) {
  return pjrtx_mlx_metal_buffer_binary_u8(
      lhs, rhs, PJRTX_MLX_METAL_U8_BINARY_ADD);
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_binary_u8(
    PjrtxMlxMetalBuffer* lhs, PjrtxMlxMetalBuffer* rhs, int op) {
  return pjrtx_mlx_metal_buffer_binary(lhs, rhs, op);
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_unary_u8(
    PjrtxMlxMetalBuffer* src, int op) {
  return pjrtx_mlx_metal_buffer_unary(src, op);
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_binary(
    PjrtxMlxMetalBuffer* lhs, PjrtxMlxMetalBuffer* rhs, int op) {
  if (lhs == nullptr) {
    return nullptr;
  }
  return pjrtx_mlx_metal_buffer_binary_out(lhs, rhs, op, lhs->dims.data(),
                                           lhs->dims.size());
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_binary_out(
    PjrtxMlxMetalBuffer* lhs, PjrtxMlxMetalBuffer* rhs, int op,
    const int64_t* output_dims, uint64_t output_rank) {
  if (lhs == nullptr || rhs == nullptr || lhs->byte_size == 0 ||
      rhs->byte_size == 0 || (output_rank > 0 && output_dims == nullptr) ||
      lhs->dtype != rhs->dtype ||
      lhs->device_ordinal != rhs->device_ordinal) {
    return nullptr;
  }
  std::vector<int64_t> out_dims;
  if (output_rank > 0) {
    out_dims.assign(output_dims, output_dims + output_rank);
  }
  const uint64_t byte_size = byte_size_for_shape(lhs->dtype, out_dims);
  if (byte_size == 0) {
    return nullptr;
  }

  if (!is_valid_binary_u8_op(op)) {
    return nullptr;
  }

  const mlx::core::Device device(mlx::core::Device::gpu, lhs->device_ordinal);
  if (!mlx::core::is_available(device) || lhs->array == nullptr ||
      rhs->array == nullptr) {
    return nullptr;
  }

  try {
    auto lhs_array = *lhs->array;
    auto rhs_array = *rhs->array;
    const auto output_shape = mlx_shape(out_dims);
    if (lhs_array.shape() != output_shape) {
      lhs_array = mlx::core::broadcast_to(lhs_array, output_shape, device);
    }
    if (rhs_array.shape() != output_shape) {
      rhs_array = mlx::core::broadcast_to(rhs_array, output_shape, device);
    }
    const uint64_t output_elements = element_count_for_shape(out_dims);
    auto msl_out = try_msl_binary_array(lhs_array, rhs_array, op, lhs->dtype,
                                        output_shape, output_elements, device);
    auto out = msl_out.has_value()
                   ? std::move(*msl_out)
                   : mlx_binary_array(lhs_array, rhs_array, op, lhs->dtype,
                                      device);
    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(std::move(array), byte_size,
                                   lhs->dtype, std::move(out_dims),
                                   lhs->device_ordinal);
  } catch (const std::exception&) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_unary(
    PjrtxMlxMetalBuffer* src, int op) {
  if (src == nullptr || src->byte_size == 0) {
    return nullptr;
  }

  if (!is_valid_unary_op(op)) {
    return nullptr;
  }

  const mlx::core::Device device(mlx::core::Device::gpu, src->device_ordinal);
  if (!mlx::core::is_available(device) || src->array == nullptr) {
    return nullptr;
  }

  try {
    auto out = mlx_unary_array(*src->array, op, device);
    const int output_dtype = mlx_unary_output_dtype(src->dtype, op);
    const uint64_t byte_size = byte_size_for_shape(output_dtype, src->dims);
    if (byte_size == 0) {
      return nullptr;
    }
    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(std::move(array), byte_size, output_dtype,
                                   src->dims, src->device_ordinal);
  } catch (const std::exception&) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_reshape(
    PjrtxMlxMetalBuffer* src, const int64_t* dims, uint64_t rank) {
  if (src == nullptr || (rank > 0 && dims == nullptr) || src->array == nullptr) {
    return nullptr;
  }
  std::vector<int64_t> out_dims(dims, dims + rank);
  const uint64_t byte_size = byte_size_for_shape(src->dtype, out_dims);
  if (byte_size == 0 || byte_size != src->byte_size) {
    return nullptr;
  }
  const mlx::core::Device device(mlx::core::Device::gpu, src->device_ordinal);
  if (!mlx::core::is_available(device)) {
    return nullptr;
  }
  try {
    auto out = mlx::core::reshape(*src->array, mlx_shape(out_dims), device);
    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(std::move(array), byte_size, src->dtype,
                                   std::move(out_dims), src->device_ordinal);
  } catch (const std::exception&) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_transpose(
    PjrtxMlxMetalBuffer* src, const int64_t* permutation, uint64_t rank) {
  if (src == nullptr || permutation == nullptr || src->byte_size == 0 ||
      rank != src->dims.size()) {
    return nullptr;
  }
  std::vector<int64_t> axes(permutation, permutation + rank);
  if (!permutation_is_valid(axes, src->dims.size())) {
    return nullptr;
  }
  const mlx::core::Device device(mlx::core::Device::gpu, src->device_ordinal);
  if (!mlx::core::is_available(device) || src->array == nullptr) {
    return nullptr;
  }

  try {
    std::vector<int> mlx_axes;
    mlx_axes.reserve(axes.size());
    for (int64_t axis : axes) {
      mlx_axes.push_back(static_cast<int>(axis));
    }
    auto out = mlx::core::transpose(*src->array, std::move(mlx_axes), device);
    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(std::move(array), src->byte_size,
                                   src->dtype, permuted_dims(src->dims, axes),
                                   src->device_ordinal);
  } catch (const std::exception&) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_broadcast_in_dim(
    PjrtxMlxMetalBuffer* src, const int64_t* broadcast_dimensions,
    uint64_t operand_rank, const int64_t* output_dims, uint64_t output_rank) {
  if (src == nullptr || broadcast_dimensions == nullptr || output_dims == nullptr ||
      src->byte_size == 0 || operand_rank != src->dims.size()) {
    return nullptr;
  }
  std::vector<int64_t> axes(broadcast_dimensions,
                            broadcast_dimensions + operand_rank);
  std::vector<int64_t> out_dims(output_dims, output_dims + output_rank);
  if (!broadcast_dimensions_are_valid(axes, src->dims, out_dims) ||
      byte_size_for_shape(src->dtype, out_dims) == 0) {
    return nullptr;
  }

  const mlx::core::Device device(mlx::core::Device::gpu, src->device_ordinal);
  if (!mlx::core::is_available(device) || src->array == nullptr) {
    return nullptr;
  }

  try {
    auto reshaped = mlx::core::reshape(
        *src->array, mlx_shape(expanded_broadcast_dims(src->dims, axes, out_dims.size())),
        device);
    auto out = mlx::core::broadcast_to(reshaped, mlx_shape(out_dims), device);
    uint64_t byte_size = byte_size_for_shape(src->dtype, out_dims);
    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(std::move(array), byte_size, src->dtype,
                                   std::move(out_dims),
                                   src->device_ordinal);
  } catch (const std::exception&) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_slice(
    PjrtxMlxMetalBuffer* src, const int64_t* start_indices,
    const int64_t* limit_indices, const int64_t* strides, uint64_t rank,
    const int64_t* output_dims, uint64_t output_rank) {
  if (src == nullptr || start_indices == nullptr || limit_indices == nullptr ||
      strides == nullptr || output_dims == nullptr || src->byte_size == 0 ||
      rank != src->dims.size() || output_rank != src->dims.size()) {
    return nullptr;
  }
  std::vector<int64_t> start(start_indices, start_indices + rank);
  std::vector<int64_t> stop(limit_indices, limit_indices + rank);
  std::vector<int64_t> step(strides, strides + rank);
  std::vector<int64_t> out_dims(output_dims, output_dims + output_rank);
  const uint64_t byte_size = byte_size_for_shape(src->dtype, out_dims);
  if (!slice_is_valid(start, stop, step, src->dims, out_dims) ||
      byte_size == 0) {
    return nullptr;
  }

  const mlx::core::Device device(mlx::core::Device::gpu, src->device_ordinal);
  if (!mlx::core::is_available(device) || src->array == nullptr) {
    return nullptr;
  }

  try {
    auto out = mlx::core::slice(*src->array, mlx_shape(start), mlx_shape(stop),
                                mlx_shape(step), device);
    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(std::move(array), byte_size, src->dtype,
                                   std::move(out_dims),
                                   src->device_ordinal);
  } catch (const std::exception&) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_dynamic_slice(
    PjrtxMlxMetalBuffer* src, PjrtxMlxMetalBuffer* const* start_buffers,
    uint64_t num_start_buffers, const int64_t* slice_sizes, uint64_t rank,
    const int64_t* output_dims, uint64_t output_rank) {
  if (src == nullptr || start_buffers == nullptr || slice_sizes == nullptr ||
      output_dims == nullptr || src->byte_size == 0 ||
      rank != src->dims.size() || output_rank != src->dims.size() ||
      num_start_buffers != rank) {
    return nullptr;
  }
  std::vector<int64_t> sizes(slice_sizes, slice_sizes + rank);
  std::vector<int64_t> out_dims(output_dims, output_dims + output_rank);
  const uint64_t byte_size = byte_size_for_shape(src->dtype, out_dims);
  if (!dynamic_slice_is_valid(sizes, src->dims, out_dims) || byte_size == 0) {
    return nullptr;
  }

  const mlx::core::Device device(mlx::core::Device::gpu, src->device_ordinal);
  if (!mlx::core::is_available(device) || src->array == nullptr) {
    return nullptr;
  }

  try {
    auto start = make_start_array(start_buffers, rank, src->device_ordinal, device);
    if (start == nullptr) {
      return nullptr;
    }
    auto out = mlx::core::slice(*src->array, *start, all_axes(rank),
                                mlx_shape(sizes), device);
    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(std::move(array), byte_size, src->dtype,
                                   std::move(out_dims), src->device_ordinal);
  } catch (const std::exception&) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_dynamic_update_slice(
    PjrtxMlxMetalBuffer* src, PjrtxMlxMetalBuffer* update,
    PjrtxMlxMetalBuffer* const* start_buffers, uint64_t num_start_buffers,
    const int64_t* output_dims, uint64_t output_rank) {
  if (src == nullptr || update == nullptr || start_buffers == nullptr ||
      output_dims == nullptr || src->byte_size == 0 || update->byte_size == 0 ||
      src->dtype != update->dtype || src->device_ordinal != update->device_ordinal ||
      output_rank != src->dims.size() || num_start_buffers != src->dims.size() ||
      update->dims.size() != src->dims.size()) {
    return nullptr;
  }
  std::vector<int64_t> out_dims(output_dims, output_dims + output_rank);
  const uint64_t byte_size = byte_size_for_shape(src->dtype, out_dims);
  if (byte_size == 0 || out_dims != src->dims) {
    return nullptr;
  }
  for (size_t axis = 0; axis < src->dims.size(); ++axis) {
    if (update->dims[axis] < 0 || update->dims[axis] > src->dims[axis]) {
      return nullptr;
    }
  }

  const mlx::core::Device device(mlx::core::Device::gpu, src->device_ordinal);
  if (!mlx::core::is_available(device) || src->array == nullptr ||
      update->array == nullptr) {
    return nullptr;
  }

  try {
    auto start =
        make_start_array(start_buffers, src->dims.size(), src->device_ordinal, device);
    if (start == nullptr) {
      return nullptr;
    }
    auto out = mlx::core::slice_update(*src->array, *update->array, *start,
                                       all_axes(src->dims.size()), device);
    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(std::move(array), byte_size, src->dtype,
                                   std::move(out_dims), src->device_ordinal);
  } catch (const std::exception&) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_pad(
    PjrtxMlxMetalBuffer* src, PjrtxMlxMetalBuffer* padding_value,
    const int64_t* edge_padding_low, const int64_t* edge_padding_high,
    const int64_t* interior_padding, uint64_t rank, const int64_t* output_dims,
    uint64_t output_rank) {
  if (src == nullptr || padding_value == nullptr || edge_padding_low == nullptr ||
      edge_padding_high == nullptr || interior_padding == nullptr ||
      output_dims == nullptr || src->byte_size == 0 ||
      src->dtype != padding_value->dtype || !padding_value->dims.empty() ||
      src->device_ordinal != padding_value->device_ordinal ||
      rank != src->dims.size() || output_rank != src->dims.size()) {
    return nullptr;
  }
  std::vector<int64_t> low(edge_padding_low, edge_padding_low + rank);
  std::vector<int64_t> high(edge_padding_high, edge_padding_high + rank);
  std::vector<int64_t> interior(interior_padding, interior_padding + rank);
  std::vector<int64_t> out_dims(output_dims, output_dims + output_rank);
  bool has_interior_padding = false;
  for (uint64_t axis = 0; axis < rank; ++axis) {
    if (low[axis] < 0 || high[axis] < 0 || interior[axis] < 0) {
      return nullptr;
    }
    if (interior[axis] != 0) {
      has_interior_padding = true;
    }
    const int64_t interior_slots =
        src->dims[axis] > 0 ? (src->dims[axis] - 1) * interior[axis] : 0;
    if (out_dims[axis] !=
        src->dims[axis] + low[axis] + high[axis] + interior_slots) {
      return nullptr;
    }
  }
  const uint64_t byte_size = byte_size_for_shape(src->dtype, out_dims);
  if (byte_size == 0) {
    return nullptr;
  }

  const mlx::core::Device device(mlx::core::Device::gpu, src->device_ordinal);
  if (!mlx::core::is_available(device) || src->array == nullptr ||
      padding_value->array == nullptr) {
    return nullptr;
  }

  try {
    mlx::core::array out =
        has_interior_padding
            ? mlx::core::full(mlx_shape(out_dims), *padding_value->array, device)
            : mlx::core::pad(*src->array, all_axes(rank), mlx_shape(low),
                             mlx_shape(high), *padding_value->array,
                             "constant", device);
    if (has_interior_padding) {
      mlx::core::Shape start;
      mlx::core::Shape stop;
      mlx::core::Shape strides;
      start.reserve(rank);
      stop.reserve(rank);
      strides.reserve(rank);
      for (uint64_t axis = 0; axis < rank; ++axis) {
        const int64_t stride = interior[axis] + 1;
        start.push_back(static_cast<mlx::core::ShapeElem>(low[axis]));
        stop.push_back(static_cast<mlx::core::ShapeElem>(
            low[axis] + src->dims[axis] * stride));
        strides.push_back(static_cast<mlx::core::ShapeElem>(stride));
      }
      out = mlx::core::slice_update(out, *src->array, std::move(start),
                                    std::move(stop), std::move(strides),
                                    device);
    }
    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(std::move(array), byte_size, src->dtype,
                                   std::move(out_dims), src->device_ordinal);
  } catch (const std::exception&) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_reverse(
    PjrtxMlxMetalBuffer* src, const int64_t* dimensions,
    uint64_t num_dimensions, const int64_t* output_dims, uint64_t output_rank) {
  if (src == nullptr || dimensions == nullptr || output_dims == nullptr ||
      src->byte_size == 0 || output_rank != src->dims.size() ||
      src->array == nullptr) {
    return nullptr;
  }
  std::vector<int64_t> dims(dimensions, dimensions + num_dimensions);
  std::vector<int64_t> out_dims(output_dims, output_dims + output_rank);
  if (out_dims != src->dims) {
    return nullptr;
  }
  for (int64_t dim : dims) {
    if (dim < 0 || static_cast<size_t>(dim) >= src->dims.size()) {
      return nullptr;
    }
  }

  const mlx::core::Device device(mlx::core::Device::gpu, src->device_ordinal);
  if (!mlx::core::is_available(device)) {
    return nullptr;
  }

  try {
    mlx::core::Shape start;
    mlx::core::Shape stop;
    mlx::core::Shape strides;
    start.reserve(src->dims.size());
    stop.reserve(src->dims.size());
    strides.reserve(src->dims.size());
    for (size_t axis = 0; axis < src->dims.size(); ++axis) {
      const int64_t dim = src->dims[axis];
      if (dim < 0) {
        return nullptr;
      }
      if (axis_in_dimensions(static_cast<int64_t>(axis), dims)) {
        start.push_back(static_cast<mlx::core::ShapeElem>(dim - 1));
        stop.push_back(static_cast<mlx::core::ShapeElem>(-dim - 1));
        strides.push_back(-1);
      } else {
        start.push_back(0);
        stop.push_back(static_cast<mlx::core::ShapeElem>(dim));
        strides.push_back(1);
      }
    }
    auto out = mlx::core::slice(*src->array, std::move(start), std::move(stop),
                                std::move(strides), device);
    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(std::move(array), src->byte_size,
                                   src->dtype, std::move(out_dims),
                                   src->device_ordinal);
  } catch (const std::exception&) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_concatenate(
    PjrtxMlxMetalBuffer* lhs, PjrtxMlxMetalBuffer* rhs, int64_t dimension,
    const int64_t* output_dims, uint64_t output_rank) {
  if (lhs == nullptr || rhs == nullptr || output_dims == nullptr ||
      lhs->byte_size == 0 || rhs->byte_size == 0 || lhs->dtype != rhs->dtype ||
      lhs->device_ordinal != rhs->device_ordinal) {
    return nullptr;
  }
  std::vector<int64_t> out_dims(output_dims, output_dims + output_rank);
  const uint64_t byte_size = byte_size_for_shape(lhs->dtype, out_dims);
  if (!concatenate_is_valid(lhs->dims, rhs->dims, dimension, out_dims) ||
      byte_size == 0) {
    return nullptr;
  }

  const mlx::core::Device device(mlx::core::Device::gpu, lhs->device_ordinal);
  if (!mlx::core::is_available(device) || lhs->array == nullptr ||
      rhs->array == nullptr) {
    return nullptr;
  }

  try {
    std::vector<mlx::core::array> arrays;
    arrays.reserve(2);
    arrays.push_back(*lhs->array);
    arrays.push_back(*rhs->array);
    auto out = mlx::core::concatenate(std::move(arrays),
                                      static_cast<int>(dimension), device);
    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(std::move(array), byte_size, lhs->dtype,
                                   std::move(out_dims),
                                   lhs->device_ordinal);
  } catch (const std::exception&) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_gather_axis(
    PjrtxMlxMetalBuffer* operand, PjrtxMlxMetalBuffer* indices, int64_t axis,
    int64_t index_vector_dim, const int64_t* output_dims,
    uint64_t output_rank) {
  if (operand == nullptr || indices == nullptr || output_dims == nullptr ||
      operand->byte_size == 0 || indices->byte_size == 0 ||
      operand->device_ordinal != indices->device_ordinal) {
    return nullptr;
  }
  std::vector<int64_t> out_dims(output_dims, output_dims + output_rank);
  const uint64_t byte_size = byte_size_for_shape(operand->dtype, out_dims);
  if (byte_size == 0 || axis < 0 ||
      static_cast<size_t>(axis) >= operand->dims.size()) {
    return nullptr;
  }

  const mlx::core::Device device(mlx::core::Device::gpu, operand->device_ordinal);
  if (!mlx::core::is_available(device) || operand->array == nullptr ||
      indices->array == nullptr) {
    return nullptr;
  }

  try {
    mlx::core::array index_array = *indices->array;
    if (index_vector_dim >= 0 &&
        static_cast<size_t>(index_vector_dim) < indices->dims.size() &&
        indices->dims[static_cast<size_t>(index_vector_dim)] == 1) {
      std::vector<int64_t> reshaped_dims = indices->dims;
      reshaped_dims.erase(reshaped_dims.begin() +
                          static_cast<size_t>(index_vector_dim));
      index_array = mlx::core::reshape(index_array, mlx_shape(reshaped_dims), device);
    }
    auto out = mlx::core::take(*operand->array, index_array, static_cast<int>(axis),
                               device);
    if (out.shape() != mlx_shape(out_dims)) {
      out = mlx::core::reshape(out, mlx_shape(out_dims), device);
    }
    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(std::move(array), byte_size, operand->dtype,
                                   std::move(out_dims),
                                   operand->device_ordinal);
  } catch (const std::exception&) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_gather(
    PjrtxMlxMetalBuffer* operand, PjrtxMlxMetalBuffer* indices,
    const int64_t* start_index_map, uint64_t num_start_axes,
    const int64_t* collapsed_slice_dims, uint64_t num_collapsed_slice_dims,
    const int64_t* operand_batching_dims, uint64_t num_operand_batching_dims,
    const int64_t* start_indices_batching_dims,
    uint64_t num_start_indices_batching_dims,
    int64_t index_vector_dim, const int64_t* slice_sizes, uint64_t slice_rank,
    const int64_t* offset_dims, uint64_t num_offset_dims,
    const int64_t* output_dims, uint64_t output_rank) {
  if (operand == nullptr || indices == nullptr || slice_sizes == nullptr ||
      output_dims == nullptr || operand->byte_size == 0 ||
      indices->byte_size == 0 ||
      operand->device_ordinal != indices->device_ordinal ||
      (num_start_axes > 0 && start_index_map == nullptr) ||
      (num_collapsed_slice_dims > 0 && collapsed_slice_dims == nullptr) ||
      (num_operand_batching_dims > 0 && operand_batching_dims == nullptr) ||
      (num_start_indices_batching_dims > 0 &&
       start_indices_batching_dims == nullptr) ||
      (num_offset_dims > 0 && offset_dims == nullptr)) {
    return nullptr;
  }

  std::vector<int64_t> axes_i64;
  axes_i64.assign(start_index_map, start_index_map + num_start_axes);
  std::vector<int64_t> collapsed;
  if (num_collapsed_slice_dims > 0) {
    collapsed.assign(collapsed_slice_dims,
                     collapsed_slice_dims + num_collapsed_slice_dims);
  }
  std::vector<int64_t> operand_batching;
  if (num_operand_batching_dims > 0) {
    operand_batching.assign(operand_batching_dims,
                            operand_batching_dims + num_operand_batching_dims);
  }
  std::vector<int64_t> start_indices_batching;
  if (num_start_indices_batching_dims > 0) {
    start_indices_batching.assign(
        start_indices_batching_dims,
        start_indices_batching_dims + num_start_indices_batching_dims);
  }
  std::vector<int64_t> slices(slice_sizes, slice_sizes + slice_rank);
  std::vector<int64_t> offsets;
  if (num_offset_dims > 0) {
    offsets.assign(offset_dims, offset_dims + num_offset_dims);
  }
  std::vector<int64_t> out_dims(output_dims, output_dims + output_rank);
  const uint64_t byte_size = byte_size_for_shape(operand->dtype, out_dims);
  if (byte_size == 0 ||
      !gather_metadata_is_valid(operand->dims, indices->dims, axes_i64,
                                collapsed, operand_batching,
                                start_indices_batching, index_vector_dim,
                                slices, offsets, out_dims)) {
    return nullptr;
  }

  const mlx::core::Device device(mlx::core::Device::gpu, operand->device_ordinal);
  if (!mlx::core::is_available(device) || operand->array == nullptr ||
      indices->array == nullptr) {
    return nullptr;
  }
  if (!operand_batching.empty() && mlx_dtype_from_code(indices->dtype) == nullptr) {
    return nullptr;
  }

  try {
    const bool explicit_vector = gather_index_vector_is_explicit(
        indices->dims, axes_i64.size(), index_vector_dim);
    const std::vector<int64_t> index_prefix_dims =
        gather_index_prefix_dims(indices->dims, explicit_vector, index_vector_dim);
    std::vector<mlx::core::array> index_arrays;
    index_arrays.reserve(axes_i64.size() + operand_batching.size());
    if (axes_i64.size() == 1) {
      mlx::core::array index_array = *indices->array;
      if (explicit_vector) {
        std::vector<int64_t> reshaped_dims = indices->dims;
        reshaped_dims.erase(reshaped_dims.begin() +
                            static_cast<size_t>(index_vector_dim));
        index_array = mlx::core::reshape(index_array, mlx_shape(reshaped_dims),
                                         device);
      }
      index_arrays.push_back(std::move(index_array));
    } else {
      for (size_t index = 0; index < axes_i64.size(); ++index) {
        std::vector<int64_t> start(indices->dims.size(), 0);
        std::vector<int64_t> stop = indices->dims;
        std::vector<int64_t> strides(indices->dims.size(), 1);
        start[static_cast<size_t>(index_vector_dim)] =
            static_cast<int64_t>(index);
        stop[static_cast<size_t>(index_vector_dim)] =
            static_cast<int64_t>(index + 1);
        auto part = mlx::core::slice(*indices->array, mlx_shape(start),
                                     mlx_shape(stop), mlx_shape(strides),
                                     device);
        part = mlx::core::squeeze(part, static_cast<int>(index_vector_dim),
                                  device);
        index_arrays.push_back(std::move(part));
      }
    }
    for (size_t i = 0; i < operand_batching.size(); ++i) {
      index_arrays.push_back(gather_batch_index_array(
          indices->dims, index_prefix_dims, start_indices_batching[i],
          explicit_vector, index_vector_dim, indices->dtype, device));
    }

    std::vector<int> axes;
    axes.reserve(axes_i64.size() + operand_batching.size());
    for (int64_t axis : axes_i64) {
      axes.push_back(static_cast<int>(axis));
    }
    for (int64_t axis : operand_batching) {
      axes.push_back(static_cast<int>(axis));
    }
    auto out = mlx::core::gather(*operand->array, std::move(index_arrays),
                                 std::move(axes), mlx_shape(slices), device);

    const size_t index_prefix_rank = index_prefix_dims.size();
    std::vector<int> squeeze_axes;
    squeeze_axes.reserve(collapsed.size() + operand_batching.size());
    for (int64_t dim : collapsed) {
      squeeze_axes.push_back(
          static_cast<int>(index_prefix_rank + static_cast<size_t>(dim)));
    }
    for (int64_t dim : operand_batching) {
      squeeze_axes.push_back(
          static_cast<int>(index_prefix_rank + static_cast<size_t>(dim)));
    }
    if (!squeeze_axes.empty()) {
      std::sort(squeeze_axes.begin(), squeeze_axes.end());
      out = mlx::core::squeeze(out, squeeze_axes, device);
    }

    std::vector<bool> output_is_offset(out_dims.size(), false);
    for (int64_t dim : offsets) {
      output_is_offset[static_cast<size_t>(dim)] = true;
    }
    std::vector<int> transpose_axes(out_dims.size(), 0);
    size_t next_index_axis = 0;
    size_t next_slice_axis = 0;
    for (size_t output_axis = 0; output_axis < out_dims.size(); ++output_axis) {
      if (output_is_offset[output_axis]) {
        transpose_axes[output_axis] =
            static_cast<int>(index_prefix_rank + next_slice_axis);
        next_slice_axis += 1;
      } else {
        transpose_axes[output_axis] = static_cast<int>(next_index_axis);
        next_index_axis += 1;
      }
    }
    bool identity = true;
    for (size_t axis = 0; axis < transpose_axes.size(); ++axis) {
      if (transpose_axes[axis] != static_cast<int>(axis)) {
        identity = false;
        break;
      }
    }
    if (!identity) {
      out = mlx::core::transpose(out, std::move(transpose_axes), device);
    }
    if (out.shape() != mlx_shape(out_dims)) {
      return nullptr;
    }

    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(std::move(array), byte_size, operand->dtype,
                                   std::move(out_dims),
                                   operand->device_ordinal);
  } catch (const std::exception&) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_scatter_axis(
    PjrtxMlxMetalBuffer* operand, PjrtxMlxMetalBuffer* indices,
    PjrtxMlxMetalBuffer* updates, int64_t axis, int64_t index_vector_dim,
    int update_kind, const int64_t* output_dims, uint64_t output_rank) {
  if (operand == nullptr || indices == nullptr || updates == nullptr ||
      output_dims == nullptr || operand->byte_size == 0 ||
      indices->byte_size == 0 || updates->byte_size == 0 ||
      operand->dtype != updates->dtype ||
      operand->device_ordinal != indices->device_ordinal ||
      operand->device_ordinal != updates->device_ordinal) {
    return nullptr;
  }
  std::vector<int64_t> out_dims(output_dims, output_dims + output_rank);
  const uint64_t byte_size = byte_size_for_shape(operand->dtype, out_dims);
  if (byte_size == 0 || out_dims != operand->dims || axis < 0 ||
      static_cast<size_t>(axis) >= operand->dims.size()) {
    return nullptr;
  }

  const mlx::core::Device device(mlx::core::Device::gpu, operand->device_ordinal);
  if (!mlx::core::is_available(device) || operand->array == nullptr ||
      indices->array == nullptr || updates->array == nullptr) {
    return nullptr;
  }

  try {
    mlx::core::array index_array = *indices->array;
    if (index_vector_dim >= 0 &&
        static_cast<size_t>(index_vector_dim) < indices->dims.size() &&
        indices->dims[static_cast<size_t>(index_vector_dim)] == 1) {
      std::vector<int64_t> reshaped_dims = indices->dims;
      reshaped_dims.erase(reshaped_dims.begin() +
                          static_cast<size_t>(index_vector_dim));
      index_array = mlx::core::reshape(index_array, mlx_shape(reshaped_dims), device);
    }
    mlx::core::array out =
        update_kind == PJRTX_MLX_METAL_SCATTER_ADD
            ? mlx::core::scatter_add_axis(*operand->array, index_array,
                                          *updates->array,
                                          static_cast<int>(axis), device)
            : mlx::core::put_along_axis(*operand->array, index_array,
                                        *updates->array,
                                        static_cast<int>(axis), device);
    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(std::move(array), byte_size, operand->dtype,
                                   std::move(out_dims),
                                   operand->device_ordinal);
  } catch (const std::exception&) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_scatter(
    PjrtxMlxMetalBuffer* operand, PjrtxMlxMetalBuffer* indices,
    PjrtxMlxMetalBuffer* updates, const int64_t* scatter_dims_to_operand_dims,
    uint64_t num_scatter_axes, const int64_t* inserted_window_dims,
    uint64_t num_inserted_window_dims, const int64_t* update_window_dims,
    uint64_t num_update_window_dims, const int64_t* input_batching_dims,
    uint64_t num_input_batching_dims,
    const int64_t* scatter_indices_batching_dims,
    uint64_t num_scatter_indices_batching_dims, int64_t index_vector_dim,
    int update_kind, const int64_t* output_dims, uint64_t output_rank) {
  if (operand == nullptr || indices == nullptr || updates == nullptr ||
      output_dims == nullptr || operand->byte_size == 0 ||
      indices->byte_size == 0 || updates->byte_size == 0 ||
      operand->dtype != updates->dtype ||
      operand->device_ordinal != indices->device_ordinal ||
      operand->device_ordinal != updates->device_ordinal ||
      (num_scatter_axes > 0 && scatter_dims_to_operand_dims == nullptr) ||
      (num_inserted_window_dims > 0 && inserted_window_dims == nullptr) ||
      (num_update_window_dims > 0 && update_window_dims == nullptr) ||
      (num_input_batching_dims > 0 && input_batching_dims == nullptr) ||
      (num_scatter_indices_batching_dims > 0 &&
       scatter_indices_batching_dims == nullptr)) {
    return nullptr;
  }
  std::vector<int64_t> axes_i64;
  if (num_scatter_axes > 0) {
    axes_i64.assign(scatter_dims_to_operand_dims,
                    scatter_dims_to_operand_dims + num_scatter_axes);
  }
  std::vector<int64_t> inserted;
  if (num_inserted_window_dims > 0) {
    inserted.assign(inserted_window_dims,
                    inserted_window_dims + num_inserted_window_dims);
  }
  std::vector<int64_t> update_window;
  if (num_update_window_dims > 0) {
    update_window.assign(update_window_dims,
                         update_window_dims + num_update_window_dims);
  }
  std::vector<int64_t> input_batching;
  if (num_input_batching_dims > 0) {
    input_batching.assign(input_batching_dims,
                          input_batching_dims + num_input_batching_dims);
  }
  std::vector<int64_t> scatter_indices_batching;
  if (num_scatter_indices_batching_dims > 0) {
    scatter_indices_batching.assign(
        scatter_indices_batching_dims,
        scatter_indices_batching_dims + num_scatter_indices_batching_dims);
  }
  std::vector<int64_t> out_dims(output_dims, output_dims + output_rank);
  const uint64_t byte_size = byte_size_for_shape(operand->dtype, out_dims);
  if (byte_size == 0 ||
      !scatter_metadata_is_valid(operand->dims, indices->dims, updates->dims,
                                 axes_i64, inserted, update_window,
                                 input_batching, scatter_indices_batching,
                                 index_vector_dim, out_dims)) {
    return nullptr;
  }

  const mlx::core::Device device(mlx::core::Device::gpu, operand->device_ordinal);
  if (!mlx::core::is_available(device) || operand->array == nullptr ||
      indices->array == nullptr || updates->array == nullptr) {
    return nullptr;
  }
  if (!input_batching.empty() && mlx_dtype_from_code(indices->dtype) == nullptr) {
    return nullptr;
  }

  try {
    const bool explicit_vector = scatter_index_vector_is_explicit(
        indices->dims, axes_i64.size(), index_vector_dim);
    const std::vector<int64_t> index_prefix_dims =
        gather_index_prefix_dims(indices->dims, explicit_vector, index_vector_dim);
    std::vector<mlx::core::array> index_arrays;
    index_arrays.reserve(axes_i64.size() + input_batching.size());
    if (axes_i64.size() == 1) {
      mlx::core::array index_array = *indices->array;
      if (explicit_vector) {
        std::vector<int64_t> reshaped_dims = indices->dims;
        reshaped_dims.erase(reshaped_dims.begin() +
                            static_cast<size_t>(index_vector_dim));
        index_array = mlx::core::reshape(index_array, mlx_shape(reshaped_dims),
                                         device);
      }
      index_arrays.push_back(std::move(index_array));
    } else {
      for (size_t index = 0; index < axes_i64.size(); ++index) {
        std::vector<int64_t> start(indices->dims.size(), 0);
        std::vector<int64_t> stop = indices->dims;
        std::vector<int64_t> strides(indices->dims.size(), 1);
        start[static_cast<size_t>(index_vector_dim)] =
            static_cast<int64_t>(index);
        stop[static_cast<size_t>(index_vector_dim)] =
            static_cast<int64_t>(index + 1);
        auto part = mlx::core::slice(*indices->array, mlx_shape(start),
                                     mlx_shape(stop), mlx_shape(strides),
                                     device);
        part = mlx::core::squeeze(part, static_cast<int>(index_vector_dim),
                                  device);
        index_arrays.push_back(std::move(part));
      }
    }
    for (size_t i = 0; i < input_batching.size(); ++i) {
      index_arrays.push_back(gather_batch_index_array(
          indices->dims, index_prefix_dims, scatter_indices_batching[i],
          explicit_vector, index_vector_dim, indices->dtype, device));
    }

    std::vector<int> axes;
    axes.reserve(axes_i64.size() + input_batching.size());
    for (int64_t axis : axes_i64) {
      axes.push_back(static_cast<int>(axis));
    }
    for (int64_t axis : input_batching) {
      axes.push_back(static_cast<int>(axis));
    }

    const size_t index_prefix_rank = index_prefix_dims.size();
    std::vector<int64_t> reshaped_update_dims(updates->dims.begin(),
                                              updates->dims.begin() +
                                                  static_cast<std::ptrdiff_t>(
                                                      index_prefix_rank));
    reshaped_update_dims.reserve(index_prefix_rank + operand->dims.size());
    for (size_t axis = 0; axis < operand->dims.size(); ++axis) {
      auto it = std::find(update_window.begin(), update_window.end(),
                          static_cast<int64_t>(axis));
      if (it == update_window.end()) {
        reshaped_update_dims.push_back(1);
      } else {
        const size_t window_axis =
            static_cast<size_t>(std::distance(update_window.begin(), it));
        reshaped_update_dims.push_back(updates->dims[index_prefix_rank + window_axis]);
      }
    }
    auto reshaped_updates =
        mlx::core::reshape(*updates->array, mlx_shape(reshaped_update_dims),
                           device);

    mlx::core::array out =
        update_kind == PJRTX_MLX_METAL_SCATTER_ADD
            ? mlx::core::scatter_add(*operand->array, std::move(index_arrays),
                                     reshaped_updates, std::move(axes), device)
            : mlx::core::scatter(*operand->array, std::move(index_arrays),
                                 reshaped_updates, std::move(axes), device);
    if (out.shape() != mlx_shape(out_dims)) {
      return nullptr;
    }
    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(std::move(array), byte_size, operand->dtype,
                                   std::move(out_dims),
                                   operand->device_ordinal);
  } catch (const std::exception&) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_sort(
    PjrtxMlxMetalBuffer* src, int64_t dimension, const int64_t* output_dims,
    uint64_t output_rank) {
  if (src == nullptr || output_dims == nullptr || src->byte_size == 0 ||
      output_rank != src->dims.size() || src->array == nullptr ||
      dimension < 0 || static_cast<size_t>(dimension) >= src->dims.size()) {
    return nullptr;
  }
  std::vector<int64_t> out_dims(output_dims, output_dims + output_rank);
  if (out_dims != src->dims) {
    return nullptr;
  }
  const mlx::core::Device device(mlx::core::Device::gpu, src->device_ordinal);
  if (!mlx::core::is_available(device)) {
    return nullptr;
  }
  try {
    auto out = mlx::core::sort(*src->array, static_cast<int>(dimension), device);
    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(std::move(array), src->byte_size,
                                   src->dtype, std::move(out_dims),
                                   src->device_ordinal);
  } catch (const std::exception&) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_argsort(
    PjrtxMlxMetalBuffer* src, int64_t dimension, int output_dtype,
    const int64_t* output_dims, uint64_t output_rank) {
  if (src == nullptr || output_dims == nullptr || src->byte_size == 0 ||
      output_rank != src->dims.size() || src->array == nullptr ||
      dimension < 0 || static_cast<size_t>(dimension) >= src->dims.size()) {
    return nullptr;
  }
  std::vector<int64_t> out_dims(output_dims, output_dims + output_rank);
  if (out_dims != src->dims) {
    return nullptr;
  }
  const mlx::core::Device device(mlx::core::Device::gpu, src->device_ordinal);
  if (!mlx::core::is_available(device)) {
    return nullptr;
  }
  try {
    auto order = mlx::core::argsort(*src->array, static_cast<int>(dimension),
                                    device);
    auto out = mlx_astype_array(order, output_dtype, device);
    uint64_t byte_size = byte_size_for_shape(output_dtype, out_dims);
    if (byte_size == 0) {
      return nullptr;
    }
    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(std::move(array), byte_size, output_dtype,
                                   std::move(out_dims), src->device_ordinal);
  } catch (const std::exception&) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_take_along_axis(
    PjrtxMlxMetalBuffer* src, PjrtxMlxMetalBuffer* indices, int64_t dimension,
    const int64_t* output_dims, uint64_t output_rank) {
  if (src == nullptr || indices == nullptr || output_dims == nullptr ||
      src->byte_size == 0 || indices->byte_size == 0 ||
      src->array == nullptr || indices->array == nullptr ||
      output_rank != src->dims.size() || output_rank != indices->dims.size() ||
      src->device_ordinal != indices->device_ordinal || dimension < 0 ||
      static_cast<size_t>(dimension) >= src->dims.size()) {
    return nullptr;
  }
  std::vector<int64_t> out_dims(output_dims, output_dims + output_rank);
  if (out_dims != src->dims || out_dims != indices->dims) {
    return nullptr;
  }
  const mlx::core::Device device(mlx::core::Device::gpu, src->device_ordinal);
  if (!mlx::core::is_available(device)) {
    return nullptr;
  }
  try {
    auto index_array = mlx_astype_array(*indices->array,
                                        PJRTX_MLX_METAL_DTYPE_S32, device);
    auto out = mlx::core::take_along_axis(
        *src->array, index_array, static_cast<int>(dimension), device);
    if (out.shape() != mlx_shape(out_dims)) {
      return nullptr;
    }
    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(std::move(array), src->byte_size,
                                   src->dtype, std::move(out_dims),
                                   src->device_ordinal);
  } catch (const std::exception&) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_dot_general(
    PjrtxMlxMetalBuffer* lhs, PjrtxMlxMetalBuffer* rhs,
    const int64_t* lhs_batch_dimensions, uint64_t lhs_batch_rank,
    const int64_t* rhs_batch_dimensions, uint64_t rhs_batch_rank,
    const int64_t* lhs_contracting_dimensions, uint64_t lhs_contracting_rank,
    const int64_t* rhs_contracting_dimensions, uint64_t rhs_contracting_rank,
    const int64_t* output_dims, uint64_t output_rank) {
  if (lhs == nullptr || rhs == nullptr || output_dims == nullptr ||
      lhs->dtype != rhs->dtype ||
      (lhs->dtype != PJRTX_MLX_METAL_DTYPE_F32 &&
       lhs->dtype != PJRTX_MLX_METAL_DTYPE_F16 &&
       lhs->dtype != PJRTX_MLX_METAL_DTYPE_BF16) ||
      lhs->device_ordinal != rhs->device_ordinal ||
      lhs_batch_rank != rhs_batch_rank ||
      (lhs_batch_rank > 0 && lhs_batch_dimensions == nullptr) ||
      (rhs_batch_rank > 0 && rhs_batch_dimensions == nullptr) ||
      lhs_contracting_dimensions == nullptr ||
      rhs_contracting_dimensions == nullptr) {
    return nullptr;
  }
  std::vector<int64_t> lhs_batch(lhs_batch_dimensions,
                                 lhs_batch_dimensions + lhs_batch_rank);
  std::vector<int64_t> rhs_batch(rhs_batch_dimensions,
                                 rhs_batch_dimensions + rhs_batch_rank);
  std::vector<int64_t> lhs_contract(lhs_contracting_dimensions,
                                    lhs_contracting_dimensions +
                                        lhs_contracting_rank);
  std::vector<int64_t> rhs_contract(rhs_contracting_dimensions,
                                    rhs_contracting_dimensions +
                                        rhs_contracting_rank);
  std::vector<int64_t> out_dims(output_dims, output_dims + output_rank);
  const uint64_t byte_size = byte_size_for_shape(lhs->dtype, out_dims);
  if (byte_size == 0 ||
      !dot_general_is_matmul_like(lhs->dims, rhs->dims, lhs_batch, rhs_batch,
                                  lhs_contract, rhs_contract, out_dims)) {
    return nullptr;
  }
  const mlx::core::Device device(mlx::core::Device::gpu, lhs->device_ordinal);
  if (!mlx::core::is_available(device) || lhs->array == nullptr ||
      rhs->array == nullptr) {
    return nullptr;
  }
  try {
    record_dot_general_shape(lhs->dims, rhs->dims, lhs_batch, rhs_batch,
                             lhs_contract, rhs_contract, out_dims);
    if (dot_general_is_canonical_matmul(lhs->dims, rhs->dims, lhs_batch,
                                        rhs_batch, lhs_contract, rhs_contract,
                                        out_dims)) {
      record_dot_general_path("canonical");
      auto out = mlx::core::matmul(*lhs->array, *rhs->array, device);
      if (out.shape() != mlx_shape(out_dims)) {
        return nullptr;
      }
      auto array = std::make_unique<mlx::core::array>(std::move(out));
      return new PjrtxMlxMetalBuffer(std::move(array), byte_size, lhs->dtype,
                                     std::move(out_dims),
                                     lhs->device_ordinal);
    }
    if (dot_general_is_rhs_transposed_matmul(lhs->dims, rhs->dims, lhs_batch,
                                             rhs_batch, lhs_contract,
                                             rhs_contract, out_dims)) {
      record_dot_general_path("rhs_transposed");
      auto out = dot_tensordot_enabled()
                     ? mlx::core::tensordot(*lhs->array, *rhs->array,
                                            std::vector<int>{1},
                                            std::vector<int>{1}, device)
                     : mlx::core::matmul(
                           *lhs->array,
                           mlx::core::transpose(*rhs->array,
                                                std::vector<int>{1, 0},
                                                device),
                           device);
      if (out.shape() != mlx_shape(out_dims)) {
        return nullptr;
      }
      auto array = std::make_unique<mlx::core::array>(std::move(out));
      return new PjrtxMlxMetalBuffer(std::move(array), byte_size, lhs->dtype,
                                     std::move(out_dims),
                                     lhs->device_ordinal);
    }
    record_dot_general_path("normalized");

    std::vector<bool> lhs_used(lhs->dims.size(), false);
    std::vector<bool> rhs_used(rhs->dims.size(), false);
    std::vector<int> lhs_order;
    std::vector<int> rhs_order;
    std::vector<int64_t> batch_dims;
    lhs_order.reserve(lhs->dims.size());
    rhs_order.reserve(rhs->dims.size());
    batch_dims.reserve(lhs_batch.size());
    for (size_t i = 0; i < lhs_batch.size(); ++i) {
      lhs_order.push_back(static_cast<int>(lhs_batch[i]));
      rhs_order.push_back(static_cast<int>(rhs_batch[i]));
      lhs_used[static_cast<size_t>(lhs_batch[i])] = true;
      rhs_used[static_cast<size_t>(rhs_batch[i])] = true;
      batch_dims.push_back(lhs->dims[static_cast<size_t>(lhs_batch[i])]);
    }
    lhs_used[static_cast<size_t>(lhs_contract[0])] = true;
    rhs_used[static_cast<size_t>(rhs_contract[0])] = true;
    std::vector<int64_t> lhs_free_dims;
    std::vector<int64_t> rhs_free_dims;
    int64_t lhs_free_elements = 1;
    int64_t rhs_free_elements = 1;
    for (size_t axis = 0; axis < lhs->dims.size(); ++axis) {
      if (!lhs_used[axis]) {
        lhs_order.push_back(static_cast<int>(axis));
        lhs_free_dims.push_back(lhs->dims[axis]);
        lhs_free_elements *= lhs->dims[axis];
      }
    }
    lhs_order.push_back(static_cast<int>(lhs_contract[0]));
    rhs_order.push_back(static_cast<int>(rhs_contract[0]));
    for (size_t axis = 0; axis < rhs->dims.size(); ++axis) {
      if (!rhs_used[axis]) {
        rhs_order.push_back(static_cast<int>(axis));
        rhs_free_dims.push_back(rhs->dims[axis]);
        rhs_free_elements *= rhs->dims[axis];
      }
    }

    auto lhs_array = mlx::core::transpose(*lhs->array, std::move(lhs_order), device);
    auto rhs_array = mlx::core::transpose(*rhs->array, std::move(rhs_order), device);
    mlx::core::Shape lhs_shape;
    mlx::core::Shape rhs_shape;
    lhs_shape.reserve(batch_dims.size() + 2);
    rhs_shape.reserve(batch_dims.size() + 2);
    for (int64_t dim : batch_dims) {
      lhs_shape.push_back(static_cast<int>(dim));
      rhs_shape.push_back(static_cast<int>(dim));
    }
    lhs_shape.push_back(static_cast<int>(lhs_free_elements));
    lhs_shape.push_back(static_cast<int>(lhs->dims[static_cast<size_t>(lhs_contract[0])]));
    rhs_shape.push_back(static_cast<int>(rhs->dims[static_cast<size_t>(rhs_contract[0])]));
    rhs_shape.push_back(static_cast<int>(rhs_free_elements));
    lhs_array = mlx::core::reshape(lhs_array, std::move(lhs_shape), device);
    rhs_array = mlx::core::reshape(rhs_array, std::move(rhs_shape), device);
    auto out = mlx::core::matmul(lhs_array, rhs_array, device);
    if (out.shape() != mlx_shape(out_dims)) {
      out = mlx::core::reshape(out, mlx_shape(out_dims), device);
    }
    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(std::move(array), byte_size,
                                   lhs->dtype, std::move(out_dims),
                                   lhs->device_ordinal);
  } catch (const std::exception&) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_convolution(
    PjrtxMlxMetalBuffer* lhs, PjrtxMlxMetalBuffer* rhs,
    const int64_t* window_strides, const int64_t* padding_low,
    const int64_t* padding_high, const int64_t* lhs_dilation,
    const int64_t* rhs_dilation, const uint8_t* window_reversal,
    uint64_t spatial_rank, int64_t feature_group_count,
    const int64_t* output_dims, uint64_t output_rank) {
  if (lhs == nullptr || rhs == nullptr || lhs->array == nullptr ||
      rhs->array == nullptr || lhs->byte_size == 0 || rhs->byte_size == 0 ||
      lhs->device_ordinal != rhs->device_ordinal || output_dims == nullptr ||
      spatial_rank == 0 || spatial_rank > 3 || output_rank != spatial_rank + 2 ||
      lhs->dims.size() != output_rank || rhs->dims.size() != output_rank ||
      window_strides == nullptr || padding_low == nullptr ||
      padding_high == nullptr || lhs_dilation == nullptr ||
      rhs_dilation == nullptr || window_reversal == nullptr ||
      feature_group_count <= 0) {
    return nullptr;
  }
  if (lhs->dtype != rhs->dtype ||
      (lhs->dtype != PJRTX_MLX_METAL_DTYPE_F32 &&
       lhs->dtype != PJRTX_MLX_METAL_DTYPE_F16 &&
       lhs->dtype != PJRTX_MLX_METAL_DTYPE_BF16)) {
    return nullptr;
  }

  std::vector<int64_t> out_dims(output_dims, output_dims + output_rank);
  const uint64_t byte_size = byte_size_for_shape(lhs->dtype, out_dims);
  if (byte_size == 0) {
    return nullptr;
  }

  std::vector<int> stride;
  std::vector<int> pad_lo;
  std::vector<int> pad_hi;
  std::vector<int> input_dilation;
  std::vector<int> kernel_dilation;
  stride.reserve(spatial_rank);
  pad_lo.reserve(spatial_rank);
  pad_hi.reserve(spatial_rank);
  input_dilation.reserve(spatial_rank);
  kernel_dilation.reserve(spatial_rank);
  bool flip = false;
  for (uint64_t i = 0; i < spatial_rank; ++i) {
    if (window_strides[i] <= 0 || lhs_dilation[i] <= 0 ||
        rhs_dilation[i] <= 0) {
      return nullptr;
    }
    if (window_reversal[i] != 0) {
      flip = true;
    }
    stride.push_back(static_cast<int>(window_strides[i]));
    pad_lo.push_back(static_cast<int>(padding_low[i]));
    pad_hi.push_back(static_cast<int>(padding_high[i]));
    input_dilation.push_back(static_cast<int>(lhs_dilation[i]));
    kernel_dilation.push_back(static_cast<int>(rhs_dilation[i]));
  }

  const mlx::core::Device device(mlx::core::Device::gpu, lhs->device_ordinal);
  if (!mlx::core::is_available(device)) {
    return nullptr;
  }

  try {
    std::vector<int> lhs_to_mlx;
    std::vector<int> rhs_to_mlx;
    std::vector<int> out_from_mlx;
    lhs_to_mlx.reserve(output_rank);
    rhs_to_mlx.reserve(output_rank);
    out_from_mlx.reserve(output_rank);
    lhs_to_mlx.push_back(0);
    rhs_to_mlx.push_back(0);
    for (uint64_t i = 0; i < spatial_rank; ++i) {
      lhs_to_mlx.push_back(static_cast<int>(i + 2));
      rhs_to_mlx.push_back(static_cast<int>(i + 2));
    }
    lhs_to_mlx.push_back(1);
    rhs_to_mlx.push_back(1);
    out_from_mlx.push_back(0);
    out_from_mlx.push_back(static_cast<int>(output_rank - 1));
    for (uint64_t i = 0; i < spatial_rank; ++i) {
      out_from_mlx.push_back(static_cast<int>(i + 1));
    }

    auto lhs_mlx = mlx::core::transpose(*lhs->array, lhs_to_mlx, device);
    auto rhs_mlx = mlx::core::transpose(*rhs->array, rhs_to_mlx, device);

    mlx::core::array conv({0}, mlx::core::float32);
    if (spatial_rank == 1) {
      auto lhs_shape = lhs_mlx.shape();
      auto rhs_shape = rhs_mlx.shape();
      lhs_shape.insert(lhs_shape.begin() + 2, 1);
      rhs_shape.insert(rhs_shape.begin() + 2, 1);
      lhs_mlx = mlx::core::reshape(lhs_mlx, std::move(lhs_shape), device);
      rhs_mlx = mlx::core::reshape(rhs_mlx, std::move(rhs_shape), device);
      stride.push_back(1);
      pad_lo.push_back(0);
      pad_hi.push_back(0);
      input_dilation.push_back(1);
      kernel_dilation.push_back(1);
    }

    conv = mlx::core::conv_general(
        std::move(lhs_mlx), std::move(rhs_mlx), std::move(stride),
        std::move(pad_lo), std::move(pad_hi), std::move(kernel_dilation),
        std::move(input_dilation), static_cast<int>(feature_group_count), flip,
        device);
    if (spatial_rank == 1) {
      conv = mlx::core::reshape(
          conv,
          {static_cast<int>(out_dims[0]), static_cast<int>(out_dims[2]),
           static_cast<int>(out_dims[1])},
          device);
    }
    auto out = mlx::core::transpose(conv, out_from_mlx, device);
    if (out.shape() != mlx_shape(out_dims)) {
      out = mlx::core::reshape(out, mlx_shape(out_dims), device);
    }
    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(std::move(array), byte_size, lhs->dtype,
                                   std::move(out_dims),
                                   lhs->device_ordinal);
  } catch (const std::exception&) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_cholesky(
    PjrtxMlxMetalBuffer* src, int lower, const int64_t* output_dims,
    uint64_t output_rank) {
  if (src == nullptr || src->array == nullptr || src->byte_size == 0 ||
      output_dims == nullptr || output_rank != src->dims.size() ||
      output_rank < 2 || src->dtype != PJRTX_MLX_METAL_DTYPE_F32) {
    return nullptr;
  }
  std::vector<int64_t> out_dims(output_dims, output_dims + output_rank);
  if (out_dims != src->dims) {
    return nullptr;
  }
  const int64_t n_i64 = out_dims[output_rank - 1];
  if (n_i64 <= 0 || out_dims[output_rank - 2] != n_i64 ||
      n_i64 > static_cast<int64_t>(std::numeric_limits<int>::max())) {
    return nullptr;
  }
  const int n = static_cast<int>(n_i64);
  const uint64_t matrix_elems = static_cast<uint64_t>(n) * static_cast<uint64_t>(n);
  const uint64_t elements = src->byte_size / sizeof(float);
  if (elements == 0 || elements % matrix_elems != 0 ||
      elements / matrix_elems > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    return nullptr;
  }
  const int batches = static_cast<int>(elements / matrix_elems);

  const mlx::core::Device device(mlx::core::Device::gpu, src->device_ordinal);
  if (!mlx::core::is_available(device)) {
    return nullptr;
  }

  try {
    const auto kernel = mlx::core::fast::metal_kernel(
        "pjrtx_cholesky_f32",
        {"a"},
        {"out"},
        R"(
          uint batch = thread_position_in_grid.x;
          uint base = batch * N * N;
          for (int r = 0; r < N; ++r) {
            for (int c = 0; c < N; ++c) {
              out[base + r * N + c] = 0.0f;
            }
          }
          for (int i = 0; i < N; ++i) {
            for (int j = 0; j <= i; ++j) {
              float sum = static_cast<float>(a[base + i * N + j]);
              for (int k = 0; k < j; ++k) {
                sum -= out[base + i * N + k] * out[base + j * N + k];
              }
              if (i == j) {
                out[base + i * N + j] = sqrt(sum);
              } else {
                out[base + i * N + j] = sum / out[base + j * N + j];
              }
            }
          }
          if (LOWER == 0) {
            for (int r = 0; r < N; ++r) {
              for (int c = r + 1; c < N; ++c) {
                out[base + r * N + c] = out[base + c * N + r];
              }
              for (int c = 0; c < r; ++c) {
                out[base + r * N + c] = 0.0f;
              }
            }
          }
        )");
    auto outputs = kernel(
        {*src->array},
        {mlx_shape(out_dims)},
        {mlx::core::float32},
        {batches, 1, 1},
        {1, 1, 1},
        {{"N", n}, {"LOWER", lower != 0 ? 1 : 0}},
        std::nullopt,
        false,
        device);
    if (outputs.size() != 1) {
      return nullptr;
    }
    auto array = std::make_unique<mlx::core::array>(std::move(outputs[0]));
    return new PjrtxMlxMetalBuffer(std::move(array), src->byte_size, src->dtype,
                                   std::move(out_dims), src->device_ordinal);
  } catch (const std::exception&) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_triangular_solve(
    PjrtxMlxMetalBuffer* a, PjrtxMlxMetalBuffer* b, int left_side, int lower,
    int unit_diagonal, int transpose_a, const int64_t* output_dims,
    uint64_t output_rank) {
  if (a == nullptr || b == nullptr || a->array == nullptr || b->array == nullptr ||
      a->byte_size == 0 || b->byte_size == 0 || output_dims == nullptr ||
      output_rank != b->dims.size() || a->dims.size() != b->dims.size() ||
      output_rank < 2 || a->dtype != PJRTX_MLX_METAL_DTYPE_F32 ||
      b->dtype != PJRTX_MLX_METAL_DTYPE_F32 ||
      a->device_ordinal != b->device_ordinal) {
    return nullptr;
  }
  std::vector<int64_t> out_dims(output_dims, output_dims + output_rank);
  if (out_dims != b->dims) {
    return nullptr;
  }
  const int64_t a_rows = a->dims[output_rank - 2];
  const int64_t a_cols = a->dims[output_rank - 1];
  if (a_rows <= 0 || a_rows != a_cols ||
      a_rows > static_cast<int64_t>(std::numeric_limits<int>::max())) {
    return nullptr;
  }
  const int64_t n_i64 = a_rows;
  const int64_t m_i64 = left_side != 0 ? b->dims[output_rank - 1]
                                       : b->dims[output_rank - 2];
  if (m_i64 <= 0 || m_i64 > static_cast<int64_t>(std::numeric_limits<int>::max())) {
    return nullptr;
  }
  if ((left_side != 0 && b->dims[output_rank - 2] != n_i64) ||
      (left_side == 0 && b->dims[output_rank - 1] != n_i64)) {
    return nullptr;
  }
  for (uint64_t i = 0; i + 2 < output_rank; ++i) {
    if (a->dims[i] != b->dims[i]) {
      return nullptr;
    }
  }
  uint64_t batches_u64 = 1;
  for (uint64_t i = 0; i + 2 < output_rank; ++i) {
    batches_u64 *= static_cast<uint64_t>(b->dims[i]);
  }
  if (batches_u64 > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    return nullptr;
  }
  const int batches = static_cast<int>(batches_u64);
  const int n = static_cast<int>(n_i64);
  const int m = static_cast<int>(m_i64);
  const uint64_t byte_size = byte_size_for_shape(PJRTX_MLX_METAL_DTYPE_F32, out_dims);
  if (byte_size == 0 || byte_size != b->byte_size) {
    return nullptr;
  }

  const mlx::core::Device device(mlx::core::Device::gpu, a->device_ordinal);
  if (!mlx::core::is_available(device)) {
    return nullptr;
  }

  try {
    const auto kernel = mlx::core::fast::metal_kernel(
        "pjrtx_triangular_solve_f32",
        {"a", "b"},
        {"out"},
        R"(
          uint batch = thread_position_in_grid.x;
          uint a_base = batch * N * N;
          uint b_base = batch * (LEFT_SIDE ? N * M : M * N);
          for (int idx = 0; idx < (LEFT_SIDE ? N * M : M * N); ++idx) {
            out[b_base + idx] = static_cast<float>(b[b_base + idx]);
          }
          bool trans = TRANSPOSE_A != 0;
          bool eff_lower = trans ? !static_cast<bool>(LOWER) : static_cast<bool>(LOWER);
          if (LEFT_SIDE) {
            if (eff_lower) {
              for (int i = 0; i < N; ++i) {
                for (int col = 0; col < M; ++col) {
                  float sum = out[b_base + i * M + col];
                  for (int k = 0; k < i; ++k) {
                    float coeff = trans ? static_cast<float>(a[a_base + k * N + i])
                                        : static_cast<float>(a[a_base + i * N + k]);
                    sum -= coeff * out[b_base + k * M + col];
                  }
                  if (!UNIT) {
                    float diag = static_cast<float>(a[a_base + i * N + i]);
                    sum /= diag;
                  }
                  out[b_base + i * M + col] = sum;
                }
              }
            } else {
              for (int i = N - 1; i >= 0; --i) {
                for (int col = 0; col < M; ++col) {
                  float sum = out[b_base + i * M + col];
                  for (int k = i + 1; k < N; ++k) {
                    float coeff = trans ? static_cast<float>(a[a_base + k * N + i])
                                        : static_cast<float>(a[a_base + i * N + k]);
                    sum -= coeff * out[b_base + k * M + col];
                  }
                  if (!UNIT) {
                    float diag = static_cast<float>(a[a_base + i * N + i]);
                    sum /= diag;
                  }
                  out[b_base + i * M + col] = sum;
                }
              }
            }
          } else {
            for (int row = 0; row < M; ++row) {
              if (eff_lower) {
                for (int j = N - 1; j >= 0; --j) {
                  float sum = out[b_base + row * N + j];
                  for (int k = j + 1; k < N; ++k) {
                    float coeff = trans ? static_cast<float>(a[a_base + j * N + k])
                                        : static_cast<float>(a[a_base + k * N + j]);
                    sum -= out[b_base + row * N + k] * coeff;
                  }
                  if (!UNIT) {
                    float diag = static_cast<float>(a[a_base + j * N + j]);
                    sum /= diag;
                  }
                  out[b_base + row * N + j] = sum;
                }
              } else {
                for (int j = 0; j < N; ++j) {
                  float sum = out[b_base + row * N + j];
                  for (int k = 0; k < j; ++k) {
                    float coeff = trans ? static_cast<float>(a[a_base + j * N + k])
                                        : static_cast<float>(a[a_base + k * N + j]);
                    sum -= out[b_base + row * N + k] * coeff;
                  }
                  if (!UNIT) {
                    float diag = static_cast<float>(a[a_base + j * N + j]);
                    sum /= diag;
                  }
                  out[b_base + row * N + j] = sum;
                }
              }
            }
          }
        )");
    auto outputs = kernel(
        {*a->array, *b->array},
        {mlx_shape(out_dims)},
        {mlx::core::float32},
        {batches, 1, 1},
        {1, 1, 1},
        {
            {"N", n},
            {"M", m},
            {"LEFT_SIDE", left_side != 0 ? 1 : 0},
            {"LOWER", lower != 0 ? 1 : 0},
            {"UNIT", unit_diagonal != 0 ? 1 : 0},
            {"TRANSPOSE_A", transpose_a},
        },
        std::nullopt,
        false,
        device);
    if (outputs.size() != 1) {
      return nullptr;
    }
    auto array = std::make_unique<mlx::core::array>(std::move(outputs[0]));
    return new PjrtxMlxMetalBuffer(std::move(array), byte_size, b->dtype,
                                   std::move(out_dims), b->device_ordinal);
  } catch (const std::exception&) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_fft(
    PjrtxMlxMetalBuffer* src, int fft_kind, const int64_t* fft_lengths,
    uint64_t fft_rank, const int64_t* output_dims, uint64_t output_rank) {
  if (src == nullptr || src->array == nullptr || src->byte_size == 0 ||
      output_dims == nullptr || fft_lengths == nullptr || fft_rank == 0 ||
      fft_rank > 3 || output_rank != src->dims.size() ||
      fft_rank > output_rank) {
    return nullptr;
  }

  int output_dtype = PJRTX_MLX_METAL_DTYPE_INVALID;
  switch (fft_kind) {
    case PJRTX_MLX_METAL_FFT:
    case PJRTX_MLX_METAL_IFFT:
      if (src->dtype != PJRTX_MLX_METAL_DTYPE_C64) {
        return nullptr;
      }
      output_dtype = PJRTX_MLX_METAL_DTYPE_C64;
      break;
    case PJRTX_MLX_METAL_RFFT:
      if (src->dtype != PJRTX_MLX_METAL_DTYPE_F32) {
        return nullptr;
      }
      output_dtype = PJRTX_MLX_METAL_DTYPE_C64;
      break;
    case PJRTX_MLX_METAL_IRFFT:
      if (src->dtype != PJRTX_MLX_METAL_DTYPE_C64) {
        return nullptr;
      }
      output_dtype = PJRTX_MLX_METAL_DTYPE_F32;
      break;
    default:
      return nullptr;
  }

  std::vector<int64_t> out_dims(output_dims, output_dims + output_rank);
  const uint64_t byte_size = byte_size_for_shape(output_dtype, out_dims);
  if (byte_size == 0) {
    return nullptr;
  }

  mlx::core::Shape lengths;
  std::vector<int> axes;
  lengths.reserve(fft_rank);
  axes.reserve(fft_rank);
  const size_t first_fft_axis = output_rank - fft_rank;
  for (uint64_t i = 0; i < fft_rank; ++i) {
    if (fft_lengths[i] <= 0) {
      return nullptr;
    }
    lengths.push_back(static_cast<mlx::core::ShapeElem>(fft_lengths[i]));
    axes.push_back(static_cast<int>(first_fft_axis + i));
  }

  const mlx::core::Device device(mlx::core::Device::gpu, src->device_ordinal);
  if (!mlx::core::is_available(device)) {
    return nullptr;
  }

  try {
    mlx::core::array out({0}, mlx::core::float32);
    switch (fft_kind) {
      case PJRTX_MLX_METAL_FFT:
        out = mlx::core::fft::fftn(*src->array, lengths, axes,
                                   mlx::core::fft::FFTNorm::Backward, device);
        break;
      case PJRTX_MLX_METAL_IFFT:
        out = mlx::core::fft::ifftn(*src->array, lengths, axes,
                                    mlx::core::fft::FFTNorm::Backward, device);
        break;
      case PJRTX_MLX_METAL_RFFT:
        out = mlx::core::fft::rfftn(*src->array, lengths, axes,
                                    mlx::core::fft::FFTNorm::Backward, device);
        break;
      case PJRTX_MLX_METAL_IRFFT:
        out = mlx::core::fft::irfftn(*src->array, lengths, axes,
                                     mlx::core::fft::FFTNorm::Backward, device);
        break;
      default:
        return nullptr;
    }
    if (out.shape() != mlx_shape(out_dims)) {
      out = mlx::core::reshape(out, mlx_shape(out_dims), device);
    }
    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(std::move(array), byte_size, output_dtype,
                                   std::move(out_dims), src->device_ordinal);
  } catch (const std::exception&) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_reduce(
    PjrtxMlxMetalBuffer* src, int op, const int64_t* dimensions,
    uint64_t num_dimensions, const int64_t* output_dims, uint64_t output_rank) {
  if (src == nullptr || output_dims == nullptr ||
      (num_dimensions > 0 && dimensions == nullptr)) {
    return nullptr;
  }
  if ((op == PJRTX_MLX_METAL_REDUCE_SUM ||
       op == PJRTX_MLX_METAL_REDUCE_MAX ||
       op == PJRTX_MLX_METAL_REDUCE_MIN) &&
      src->dtype != PJRTX_MLX_METAL_DTYPE_F16 &&
      src->dtype != PJRTX_MLX_METAL_DTYPE_F32 &&
      src->dtype != PJRTX_MLX_METAL_DTYPE_BF16 &&
      src->dtype != PJRTX_MLX_METAL_DTYPE_S8 &&
      src->dtype != PJRTX_MLX_METAL_DTYPE_S32 &&
      src->dtype != PJRTX_MLX_METAL_DTYPE_U8 &&
      src->dtype != PJRTX_MLX_METAL_DTYPE_U16 &&
      src->dtype != PJRTX_MLX_METAL_DTYPE_U32 &&
      src->dtype != PJRTX_MLX_METAL_DTYPE_U64) {
    return nullptr;
  }
  if ((op == PJRTX_MLX_METAL_REDUCE_AND ||
       op == PJRTX_MLX_METAL_REDUCE_OR) &&
      src->dtype != PJRTX_MLX_METAL_DTYPE_PRED) {
    return nullptr;
  }
  std::vector<int64_t> out_dims(output_dims, output_dims + output_rank);
  const uint64_t byte_size = byte_size_for_shape(src->dtype, out_dims);
  if (byte_size == 0) {
    return nullptr;
  }
  std::vector<int> axes;
  axes.reserve(num_dimensions);
  for (uint64_t i = 0; i < num_dimensions; ++i) {
    axes.push_back(static_cast<int>(dimensions[i]));
  }
  const mlx::core::Device device(mlx::core::Device::gpu, src->device_ordinal);
  if (!mlx::core::is_available(device) || src->array == nullptr) {
    return nullptr;
  }
  try {
    mlx::core::array out =
        op == PJRTX_MLX_METAL_REDUCE_SUM
            ? mlx::core::sum(*src->array, axes, false, device)
            : op == PJRTX_MLX_METAL_REDUCE_MAX
                  ? mlx::core::max(*src->array, axes, false, device)
                  : op == PJRTX_MLX_METAL_REDUCE_MIN
                        ? mlx::core::min(*src->array, axes, false, device)
                        : op == PJRTX_MLX_METAL_REDUCE_AND
                              ? mlx::core::all(*src->array, axes, false,
                                               device)
                              : mlx::core::any(*src->array, axes, false,
                                               device);
    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(std::move(array), byte_size, src->dtype,
                                   std::move(out_dims),
                                   src->device_ordinal);
  } catch (const std::exception&) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

int pjrtx_mlx_metal_buffer_reduce_max_with_indices(
    PjrtxMlxMetalBuffer* values, PjrtxMlxMetalBuffer* indices,
    const int64_t* dimensions, uint64_t num_dimensions,
    const int64_t* output_dims, uint64_t output_rank,
    PjrtxMlxMetalBuffer** out_values, PjrtxMlxMetalBuffer** out_indices) {
  if (out_values == nullptr || out_indices == nullptr) {
    return 0;
  }
  *out_values = nullptr;
  *out_indices = nullptr;
  if (values == nullptr || indices == nullptr ||
      (num_dimensions > 0 && dimensions == nullptr) ||
      (output_rank > 0 && output_dims == nullptr) ||
      values->array == nullptr || indices->array == nullptr ||
      values->dims != indices->dims ||
      values->device_ordinal != indices->device_ordinal ||
      (values->dtype != PJRTX_MLX_METAL_DTYPE_F16 &&
       values->dtype != PJRTX_MLX_METAL_DTYPE_F32 &&
       values->dtype != PJRTX_MLX_METAL_DTYPE_BF16) ||
      (indices->dtype != PJRTX_MLX_METAL_DTYPE_S32 &&
       indices->dtype != PJRTX_MLX_METAL_DTYPE_U32)) {
    return 0;
  }

  std::vector<int64_t> out_dims(output_dims, output_dims + output_rank);
  const uint64_t values_byte_size = byte_size_for_shape(values->dtype, out_dims);
  const uint64_t indices_byte_size =
      byte_size_for_shape(indices->dtype, out_dims);
  if (values_byte_size == 0 || indices_byte_size == 0) {
    return 0;
  }

  std::vector<int> axes;
  axes.reserve(num_dimensions);
  std::vector<int64_t> keep_dims = values->dims;
  for (uint64_t i = 0; i < num_dimensions; ++i) {
    if (dimensions[i] < 0 ||
        static_cast<size_t>(dimensions[i]) >= values->dims.size()) {
      return 0;
    }
    axes.push_back(static_cast<int>(dimensions[i]));
    keep_dims[static_cast<size_t>(dimensions[i])] = 1;
  }
  if (num_dimensions == 0) {
    if (out_dims != values->dims) {
      return 0;
    }
  }

  const mlx::core::Device device(mlx::core::Device::gpu,
                                 values->device_ordinal);
  if (!mlx::core::is_available(device)) {
    return 0;
  }

  try {
    auto reduced_values =
        num_dimensions == 0 ? *values->array
                            : mlx::core::max(*values->array, axes, false,
                                             device);
    auto reduced_indices = *indices->array;
    if (num_dimensions != 0) {
      auto max_keep =
          mlx::core::reshape(reduced_values, mlx_shape(keep_dims), device);
      auto max_broadcast =
          mlx::core::broadcast_to(max_keep, mlx_shape(values->dims), device);
      auto equal_max =
          mlx::core::equal(*values->array, max_broadcast, device);
      auto value_nan =
          mlx::core::not_equal(*values->array, *values->array, device);
      auto max_nan =
          mlx::core::not_equal(max_broadcast, max_broadcast, device);
      auto nan_match = mlx::core::logical_and(value_nan, max_nan, device);
      auto match = mlx::core::logical_or(equal_max, nan_match, device);
      const auto input_shape = mlx_shape(values->dims);
      auto sentinel_scalar =
          indices->dtype == PJRTX_MLX_METAL_DTYPE_U32
              ? mlx::core::array(std::numeric_limits<uint32_t>::max(),
                                 mlx::core::uint32)
              : mlx::core::array(std::numeric_limits<int32_t>::max(),
                                 mlx::core::int32);
      auto sentinel = mlx::core::full(input_shape, sentinel_scalar, device);
      auto candidate =
          mlx::core::where(match, *indices->array, sentinel, device);
      reduced_indices = mlx::core::min(candidate, axes, false, device);
    }
    const auto output_shape = mlx_shape(out_dims);
    if (reduced_values.shape() != output_shape) {
      reduced_values = mlx::core::reshape(reduced_values, output_shape, device);
    }
    if (reduced_indices.shape() != output_shape) {
      reduced_indices =
          mlx::core::reshape(reduced_indices, output_shape, device);
    }

    *out_values = new PjrtxMlxMetalBuffer(
        std::make_unique<mlx::core::array>(std::move(reduced_values)),
        values_byte_size, values->dtype, out_dims, values->device_ordinal);
    *out_indices = new PjrtxMlxMetalBuffer(
        std::make_unique<mlx::core::array>(std::move(reduced_indices)),
        indices_byte_size, indices->dtype, std::move(out_dims),
        indices->device_ordinal);
    return 1;
  } catch (const std::exception&) {
    if (*out_values != nullptr) {
      delete *out_values;
      *out_values = nullptr;
    }
    if (*out_indices != nullptr) {
      delete *out_indices;
      *out_indices = nullptr;
    }
    return 0;
  } catch (...) {
    if (*out_values != nullptr) {
      delete *out_values;
      *out_values = nullptr;
    }
    if (*out_indices != nullptr) {
      delete *out_indices;
      *out_indices = nullptr;
    }
    return 0;
  }
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_rng(
    PjrtxMlxMetalBuffer* a, PjrtxMlxMetalBuffer* b, int distribution,
    int output_dtype, const int64_t* output_dims, uint64_t output_rank) {
  if (a == nullptr || b == nullptr || a->array == nullptr ||
      b->array == nullptr || (output_rank > 0 && output_dims == nullptr) ||
      a->dtype != b->dtype || a->device_ordinal != b->device_ordinal ||
      output_dtype != a->dtype ||
      (output_dtype != PJRTX_MLX_METAL_DTYPE_F16 &&
       output_dtype != PJRTX_MLX_METAL_DTYPE_F32 &&
       output_dtype != PJRTX_MLX_METAL_DTYPE_BF16)) {
    return nullptr;
  }

  std::vector<int64_t> out_dims(output_dims, output_dims + output_rank);
  const uint64_t byte_size = byte_size_for_shape(output_dtype, out_dims);
  const mlx::core::Dtype* dtype = mlx_dtype_from_code(output_dtype);
  if (byte_size == 0 || dtype == nullptr) {
    return nullptr;
  }

  const mlx::core::Device device(mlx::core::Device::gpu, a->device_ordinal);
  if (!mlx::core::is_available(device)) {
    return nullptr;
  }

  try {
    std::unique_ptr<mlx::core::array> out;
    switch (distribution) {
      case PJRTX_MLX_METAL_RNG_UNIFORM:
        out = std::make_unique<mlx::core::array>(
            mlx::core::random::uniform(*a->array, *b->array,
                                       mlx_shape(out_dims), *dtype,
                                       std::nullopt, device));
        break;
      case PJRTX_MLX_METAL_RNG_NORMAL:
        out = std::make_unique<mlx::core::array>(
            mlx::core::random::normal(mlx_shape(out_dims), *dtype,
                                      std::make_optional(*a->array),
                                      std::make_optional(*b->array),
                                      std::nullopt, device));
        break;
      default:
        return nullptr;
    }
    return new PjrtxMlxMetalBuffer(std::move(out), byte_size, output_dtype,
                                   std::move(out_dims), a->device_ordinal);
  } catch (const std::exception&) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

int pjrtx_mlx_metal_buffer_rng_bit_generator(
    PjrtxMlxMetalBuffer* state, int output_dtype, const int64_t* output_dims,
    uint64_t output_rank, PjrtxMlxMetalBuffer** out_state,
    PjrtxMlxMetalBuffer** out_bits) {
  if (out_state == nullptr || out_bits == nullptr) {
    return 0;
  }
  *out_state = nullptr;
  *out_bits = nullptr;
  if (state == nullptr || state->array == nullptr ||
      (output_rank > 0 && output_dims == nullptr) ||
      state->dims != std::vector<int64_t>{2} ||
      (state->dtype != PJRTX_MLX_METAL_DTYPE_U32 &&
       state->dtype != PJRTX_MLX_METAL_DTYPE_U64)) {
    return 0;
  }

  const int width = output_dtype == PJRTX_MLX_METAL_DTYPE_U32
                        ? 4
                        : output_dtype == PJRTX_MLX_METAL_DTYPE_U16
                              ? 2
                              : output_dtype == PJRTX_MLX_METAL_DTYPE_U8
                                    ? 1
                                    : output_dtype == PJRTX_MLX_METAL_DTYPE_U64
                                          ? 8
                                          : 0;
  if (width == 0) {
    return 0;
  }

  std::vector<int64_t> out_dims(output_dims, output_dims + output_rank);
  const uint64_t state_byte_size = byte_size_for_shape(state->dtype, state->dims);
  const uint64_t bits_byte_size = byte_size_for_shape(output_dtype, out_dims);
  if (state_byte_size == 0 || bits_byte_size == 0) {
    return 0;
  }

  const mlx::core::Device device(mlx::core::Device::gpu, state->device_ordinal);
  if (!mlx::core::is_available(device)) {
    return 0;
  }

  try {
    if (state->dtype == PJRTX_MLX_METAL_DTYPE_U64 &&
        xla_threefry_rng_bit_generator(state, output_dtype, out_dims, device,
                                       out_state, out_bits)) {
      return 1;
    }

    auto key =
        state->dtype == PJRTX_MLX_METAL_DTYPE_U32
            ? *state->array
            : mlx::core::slice(
                  state->array->dtype() == mlx::core::uint32
                      ? *state->array
                      : mlx::core::view(*state->array, mlx::core::uint32,
                                        device),
                  {0}, {2}, {1}, device);
    if (key.shape() != mlx::core::Shape{2}) {
      key = mlx::core::reshape(key, {2}, device);
    }
    auto split = mlx::core::random::split(key, device);
    auto next_state =
        state->dtype == PJRTX_MLX_METAL_DTYPE_U32
            ? split.first
            : mlx::core::concatenate({split.first, split.second}, 0, device);
    auto bits = mlx::core::random::bits(mlx_shape(out_dims), width, key,
                                        device);

    *out_state = new PjrtxMlxMetalBuffer(
        std::make_unique<mlx::core::array>(std::move(next_state)),
        state_byte_size, state->dtype, state->dims, state->device_ordinal);
    *out_bits = new PjrtxMlxMetalBuffer(
        std::make_unique<mlx::core::array>(std::move(bits)), bits_byte_size,
        output_dtype, std::move(out_dims), state->device_ordinal);
    return 1;
  } catch (const std::exception&) {
    if (*out_state != nullptr) {
      delete *out_state;
      *out_state = nullptr;
    }
    if (*out_bits != nullptr) {
      delete *out_bits;
      *out_bits = nullptr;
    }
    return 0;
  } catch (...) {
    if (*out_state != nullptr) {
      delete *out_state;
      *out_state = nullptr;
    }
    if (*out_bits != nullptr) {
      delete *out_bits;
      *out_bits = nullptr;
    }
    return 0;
  }
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_reduce_window(
    PjrtxMlxMetalBuffer* src, int op, const int64_t* window_dimensions,
    const int64_t* window_strides, const int64_t* base_dilations,
    const int64_t* window_dilations, const int64_t* padding_low,
    const int64_t* padding_high, uint64_t rank, const int64_t* output_dims,
    uint64_t output_rank) {
  if (src == nullptr || src->array == nullptr || src->byte_size == 0 ||
      window_dimensions == nullptr || window_strides == nullptr ||
      base_dilations == nullptr || window_dilations == nullptr ||
      padding_low == nullptr || padding_high == nullptr ||
      output_dims == nullptr || rank == 0 || rank != src->dims.size() ||
      output_rank != rank ||
      (src->dtype != PJRTX_MLX_METAL_DTYPE_F16 &&
       src->dtype != PJRTX_MLX_METAL_DTYPE_F32 &&
       src->dtype != PJRTX_MLX_METAL_DTYPE_BF16) ||
      (op != PJRTX_MLX_METAL_REDUCE_SUM &&
       op != PJRTX_MLX_METAL_REDUCE_MAX)) {
    return nullptr;
  }

  std::vector<int64_t> window(window_dimensions, window_dimensions + rank);
  std::vector<int64_t> strides(window_strides, window_strides + rank);
  std::vector<int64_t> base(base_dilations, base_dilations + rank);
  std::vector<int64_t> dilation(window_dilations, window_dilations + rank);
  std::vector<int64_t> low(padding_low, padding_low + rank);
  std::vector<int64_t> high(padding_high, padding_high + rank);
  std::vector<int64_t> out_dims(output_dims, output_dims + output_rank);

  for (uint64_t axis = 0; axis < rank; ++axis) {
    if (window[axis] <= 0 || strides[axis] <= 0 || base[axis] != 1 ||
        dilation[axis] <= 0 || low[axis] < 0 || high[axis] < 0) {
      return nullptr;
    }
    const int64_t dilated_input = src->dims[axis];
    const int64_t padded_input = low[axis] + dilated_input + high[axis];
    const int64_t dilated_window = (window[axis] - 1) * dilation[axis] + 1;
    const int64_t expected =
        padded_input < dilated_window
            ? 0
            : ((padded_input - dilated_window) / strides[axis]) + 1;
    if (out_dims[axis] != expected) {
      return nullptr;
    }
  }

  const uint64_t byte_size = byte_size_for_shape(src->dtype, out_dims);
  const mlx::core::Dtype* dtype = mlx_dtype_from_code(src->dtype);
  if (byte_size == 0 || dtype == nullptr) {
    return nullptr;
  }

  const mlx::core::Device device(mlx::core::Device::gpu, src->device_ordinal);
  if (!mlx::core::is_available(device)) {
    return nullptr;
  }

  try {
    mlx::core::array working = *src->array;
    bool has_padding = false;
    for (uint64_t axis = 0; axis < rank; ++axis) {
      has_padding = has_padding || low[axis] != 0 || high[axis] != 0;
    }
    if (has_padding) {
      const float pad_scalar =
          op == PJRTX_MLX_METAL_REDUCE_SUM
              ? 0.0f
              : -std::numeric_limits<float>::infinity();
      mlx::core::array pad_value(pad_scalar, *dtype);
      working = mlx::core::pad(working, all_axes(rank), mlx_shape(low),
                               mlx_shape(high), pad_value, "constant",
                               device);
    }

    mlx::core::Shape window_shape;
    mlx::core::Strides window_view_strides;
    window_shape.reserve(rank * 2);
    window_view_strides.reserve(rank * 2);
    for (uint64_t axis = 0; axis < rank; ++axis) {
      window_shape.push_back(static_cast<mlx::core::ShapeElem>(out_dims[axis]));
      window_view_strides.push_back(working.strides(static_cast<int>(axis)) *
                                    strides[axis]);
    }
    for (uint64_t axis = 0; axis < rank; ++axis) {
      window_shape.push_back(static_cast<mlx::core::ShapeElem>(window[axis]));
      window_view_strides.push_back(working.strides(static_cast<int>(axis)) *
                                    dilation[axis]);
    }

    auto windows =
        mlx::core::as_strided(working, std::move(window_shape),
                              std::move(window_view_strides), 0, device);
    std::vector<int> axes;
    axes.reserve(rank);
    for (uint64_t axis = 0; axis < rank; ++axis) {
      axes.push_back(static_cast<int>(rank + axis));
    }
    mlx::core::array out =
        op == PJRTX_MLX_METAL_REDUCE_SUM
            ? mlx::core::sum(windows, axes, false, device)
            : mlx::core::max(windows, axes, false, device);
    if (out.shape() != mlx_shape(out_dims)) {
      return nullptr;
    }
    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(std::move(array), byte_size, src->dtype,
                                   std::move(out_dims), src->device_ordinal);
  } catch (const std::exception&) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

int pjrtx_mlx_metal_buffer_reduce_window_max_with_indices(
    PjrtxMlxMetalBuffer* values, PjrtxMlxMetalBuffer* indices,
    const int64_t* window_dimensions, const int64_t* window_strides,
    const int64_t* base_dilations, const int64_t* window_dilations,
    const int64_t* padding_low, const int64_t* padding_high, uint64_t rank,
    const int64_t* output_dims, uint64_t output_rank,
    PjrtxMlxMetalBuffer** out_values, PjrtxMlxMetalBuffer** out_indices) {
  if (out_values != nullptr) {
    *out_values = nullptr;
  }
  if (out_indices != nullptr) {
    *out_indices = nullptr;
  }
  if (values == nullptr || indices == nullptr || values->array == nullptr ||
      indices->array == nullptr || values->byte_size == 0 ||
      indices->byte_size == 0 || values->device_ordinal != indices->device_ordinal ||
      values->dims != indices->dims || window_dimensions == nullptr ||
      window_strides == nullptr || base_dilations == nullptr ||
      window_dilations == nullptr || padding_low == nullptr ||
      padding_high == nullptr || output_dims == nullptr || out_values == nullptr ||
      out_indices == nullptr || rank == 0 || rank != values->dims.size() ||
      output_rank != rank ||
      (values->dtype != PJRTX_MLX_METAL_DTYPE_F16 &&
       values->dtype != PJRTX_MLX_METAL_DTYPE_F32 &&
       values->dtype != PJRTX_MLX_METAL_DTYPE_BF16) ||
      (indices->dtype != PJRTX_MLX_METAL_DTYPE_S32 &&
       indices->dtype != PJRTX_MLX_METAL_DTYPE_U32)) {
    return 0;
  }

  std::vector<int64_t> window(window_dimensions, window_dimensions + rank);
  std::vector<int64_t> strides(window_strides, window_strides + rank);
  std::vector<int64_t> base(base_dilations, base_dilations + rank);
  std::vector<int64_t> dilation(window_dilations, window_dilations + rank);
  std::vector<int64_t> low(padding_low, padding_low + rank);
  std::vector<int64_t> high(padding_high, padding_high + rank);
  std::vector<int64_t> out_dims(output_dims, output_dims + output_rank);

  int64_t window_volume = 1;
  for (uint64_t axis = 0; axis < rank; ++axis) {
    if (window[axis] <= 0 || strides[axis] <= 0 || base[axis] != 1 ||
        dilation[axis] <= 0 || low[axis] < 0 || high[axis] < 0) {
      return 0;
    }
    const int64_t padded_input = low[axis] + values->dims[axis] + high[axis];
    const int64_t dilated_window = (window[axis] - 1) * dilation[axis] + 1;
    const int64_t expected =
        padded_input < dilated_window
            ? 0
            : ((padded_input - dilated_window) / strides[axis]) + 1;
    if (out_dims[axis] != expected) {
      return 0;
    }
    window_volume *= window[axis];
  }

  const uint64_t values_byte_size = byte_size_for_shape(values->dtype, out_dims);
  const uint64_t indices_byte_size =
      byte_size_for_shape(indices->dtype, out_dims);
  const mlx::core::Dtype* values_dtype = mlx_dtype_from_code(values->dtype);
  const mlx::core::Dtype* indices_dtype = mlx_dtype_from_code(indices->dtype);
  if (values_byte_size == 0 || indices_byte_size == 0 ||
      values_dtype == nullptr || indices_dtype == nullptr) {
    return 0;
  }

  const mlx::core::Device device(mlx::core::Device::gpu, values->device_ordinal);
  if (!mlx::core::is_available(device)) {
    return 0;
  }

  try {
    mlx::core::array value_working = *values->array;
    mlx::core::array index_working = *indices->array;
    bool has_padding = false;
    for (uint64_t axis = 0; axis < rank; ++axis) {
      has_padding = has_padding || low[axis] != 0 || high[axis] != 0;
    }
    if (has_padding) {
      mlx::core::array value_pad(-std::numeric_limits<float>::infinity(),
                                 *values_dtype);
      mlx::core::array index_pad(0, *indices_dtype);
      value_working = mlx::core::pad(value_working, all_axes(rank),
                                     mlx_shape(low), mlx_shape(high),
                                     value_pad, "constant", device);
      index_working = mlx::core::pad(index_working, all_axes(rank),
                                     mlx_shape(low), mlx_shape(high),
                                     index_pad, "constant", device);
    }

    mlx::core::Shape window_shape;
    mlx::core::Strides value_window_strides;
    mlx::core::Strides index_window_strides;
    window_shape.reserve(rank * 2);
    value_window_strides.reserve(rank * 2);
    index_window_strides.reserve(rank * 2);
    for (uint64_t axis = 0; axis < rank; ++axis) {
      window_shape.push_back(static_cast<mlx::core::ShapeElem>(out_dims[axis]));
      value_window_strides.push_back(
          value_working.strides(static_cast<int>(axis)) * strides[axis]);
      index_window_strides.push_back(
          index_working.strides(static_cast<int>(axis)) * strides[axis]);
    }
    for (uint64_t axis = 0; axis < rank; ++axis) {
      window_shape.push_back(static_cast<mlx::core::ShapeElem>(window[axis]));
      value_window_strides.push_back(
          value_working.strides(static_cast<int>(axis)) * dilation[axis]);
      index_window_strides.push_back(
          index_working.strides(static_cast<int>(axis)) * dilation[axis]);
    }

    auto value_windows = mlx::core::as_strided(
        value_working, window_shape, std::move(value_window_strides), 0,
        device);
    auto index_windows = mlx::core::as_strided(
        index_working, std::move(window_shape), std::move(index_window_strides),
        0, device);

    mlx::core::Shape flat_shape = mlx_shape(out_dims);
    flat_shape.push_back(static_cast<mlx::core::ShapeElem>(window_volume));
    auto value_flat = mlx::core::reshape(value_windows, flat_shape, device);
    auto index_flat = mlx::core::reshape(index_windows, flat_shape, device);
    const int flat_axis = static_cast<int>(flat_shape.size() - 1);
    auto order = mlx::core::argsort(value_flat, flat_axis, device);
    auto ordered_values =
        mlx::core::take_along_axis(value_flat, order, flat_axis, device);
    auto ordered_indices =
        mlx::core::take_along_axis(index_flat, order, flat_axis, device);

    mlx::core::Shape start(flat_shape.size(), 0);
    mlx::core::Shape stop = flat_shape;
    mlx::core::Shape step(flat_shape.size(), 1);
    start[flat_axis] = static_cast<mlx::core::ShapeElem>(window_volume - 1);
    auto max_values =
        mlx::core::slice(ordered_values, start, stop, step, device);
    auto max_indices =
        mlx::core::slice(ordered_indices, std::move(start), std::move(stop),
                         std::move(step), device);
    auto max_values_out = mlx::core::reshape(max_values, mlx_shape(out_dims),
                                             device);
    auto max_indices_out = mlx::core::reshape(max_indices, mlx_shape(out_dims),
                                              device);

    *out_values = new PjrtxMlxMetalBuffer(
        std::make_unique<mlx::core::array>(std::move(max_values_out)),
        values_byte_size, values->dtype, out_dims, values->device_ordinal);
    *out_indices = new PjrtxMlxMetalBuffer(
        std::make_unique<mlx::core::array>(std::move(max_indices_out)),
        indices_byte_size, indices->dtype, std::move(out_dims),
        values->device_ordinal);
    return 1;
  } catch (const std::exception&) {
    if (*out_values != nullptr) {
      delete *out_values;
      *out_values = nullptr;
    }
    if (*out_indices != nullptr) {
      delete *out_indices;
      *out_indices = nullptr;
    }
    return 0;
  } catch (...) {
    if (*out_values != nullptr) {
      delete *out_values;
      *out_values = nullptr;
    }
    if (*out_indices != nullptr) {
      delete *out_indices;
      *out_indices = nullptr;
    }
    return 0;
  }
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_compare(
    PjrtxMlxMetalBuffer* lhs, PjrtxMlxMetalBuffer* rhs, int direction,
    const int64_t* output_dims, uint64_t output_rank) {
  if (lhs == nullptr || rhs == nullptr ||
      (output_rank > 0 && output_dims == nullptr) ||
      lhs->byte_size == 0 || rhs->byte_size == 0 ||
      lhs->dtype != rhs->dtype ||
      lhs->device_ordinal != rhs->device_ordinal) {
    return nullptr;
  }
  std::vector<int64_t> out_dims;
  if (output_rank > 0) {
    out_dims.assign(output_dims, output_dims + output_rank);
  }
  const uint64_t byte_size = byte_size_for_shape(PJRTX_MLX_METAL_DTYPE_PRED,
                                                 out_dims);
  if (byte_size == 0) {
    return nullptr;
  }
  const mlx::core::Device device(mlx::core::Device::gpu, lhs->device_ordinal);
  if (!mlx::core::is_available(device) || lhs->array == nullptr ||
      rhs->array == nullptr) {
    return nullptr;
  }
  try {
    auto lhs_array = *lhs->array;
    auto rhs_array = *rhs->array;
    const auto output_shape = mlx_shape(out_dims);
    if (lhs_array.shape() != output_shape) {
      lhs_array = mlx::core::broadcast_to(lhs_array, output_shape, device);
    }
    if (rhs_array.shape() != output_shape) {
      rhs_array = mlx::core::broadcast_to(rhs_array, output_shape, device);
    }
    auto out = mlx_compare_array(lhs_array, rhs_array, direction, device);
    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(std::move(array), byte_size,
                                   PJRTX_MLX_METAL_DTYPE_PRED,
                                   std::move(out_dims), lhs->device_ordinal);
  } catch (const std::exception&) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_select(
    PjrtxMlxMetalBuffer* pred, PjrtxMlxMetalBuffer* on_true,
    PjrtxMlxMetalBuffer* on_false, const int64_t* output_dims,
    uint64_t output_rank) {
  if (pred == nullptr || on_true == nullptr || on_false == nullptr ||
      output_dims == nullptr || pred->dtype != PJRTX_MLX_METAL_DTYPE_PRED ||
      on_true->dtype != on_false->dtype || on_true->dims != on_false->dims ||
      pred->dims != on_true->dims ||
      pred->device_ordinal != on_true->device_ordinal ||
      on_true->device_ordinal != on_false->device_ordinal) {
    return nullptr;
  }
  std::vector<int64_t> out_dims(output_dims, output_dims + output_rank);
  const uint64_t byte_size = byte_size_for_shape(on_true->dtype, out_dims);
  if (byte_size == 0 || out_dims != on_true->dims) {
    return nullptr;
  }
  const mlx::core::Device device(mlx::core::Device::gpu, on_true->device_ordinal);
  if (!mlx::core::is_available(device) || pred->array == nullptr ||
      on_true->array == nullptr || on_false->array == nullptr) {
    return nullptr;
  }
  try {
    auto out = mlx::core::where(*pred->array, *on_true->array, *on_false->array,
                                device);
    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(std::move(array), byte_size,
                                   on_true->dtype, std::move(out_dims),
                                   on_true->device_ordinal);
  } catch (const std::exception&) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_clamp(
    PjrtxMlxMetalBuffer* min, PjrtxMlxMetalBuffer* value,
    PjrtxMlxMetalBuffer* max, const int64_t* output_dims,
    uint64_t output_rank) {
  if (min == nullptr || value == nullptr || max == nullptr ||
      output_dims == nullptr || min->dtype != value->dtype ||
      max->dtype != value->dtype ||
      min->device_ordinal != value->device_ordinal ||
      max->device_ordinal != value->device_ordinal) {
    return nullptr;
  }
  std::vector<int64_t> out_dims(output_dims, output_dims + output_rank);
  const bool min_scalar = min->dims.empty();
  const bool max_scalar = max->dims.empty();
  if (out_dims != value->dims || (!min_scalar && min->dims != value->dims) ||
      (!max_scalar && max->dims != value->dims)) {
    return nullptr;
  }
  const uint64_t byte_size = byte_size_for_shape(value->dtype, out_dims);
  if (byte_size == 0) {
    return nullptr;
  }
  const mlx::core::Device device(mlx::core::Device::gpu, value->device_ordinal);
  if (!mlx::core::is_available(device) || min->array == nullptr ||
      value->array == nullptr || max->array == nullptr) {
    return nullptr;
  }
  try {
    auto lower_bounded = mlx::core::maximum(*value->array, *min->array, device);
    auto out = mlx::core::minimum(lower_bounded, *max->array, device);
    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(std::move(array), byte_size, value->dtype,
                                   std::move(out_dims),
                                   value->device_ordinal);
  } catch (const std::exception&) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_while_f32_lt_add(
    PjrtxMlxMetalBuffer* state, PjrtxMlxMetalBuffer* limit,
    PjrtxMlxMetalBuffer* step, const int64_t* output_dims,
    uint64_t output_rank, uint64_t max_iterations) {
  return pjrtx_mlx_metal_buffer_while_f32_compare_add(
      state, limit, step, PJRTX_MLX_METAL_COMPARE_LT,
      PJRTX_MLX_METAL_U8_BINARY_ADD, output_dims, output_rank,
      max_iterations);
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_while_f32_compare_add(
    PjrtxMlxMetalBuffer* state, PjrtxMlxMetalBuffer* limit,
    PjrtxMlxMetalBuffer* step, int compare_direction, int update_op,
    const int64_t* output_dims, uint64_t output_rank,
    uint64_t max_iterations) {
  if (state == nullptr || limit == nullptr || step == nullptr ||
      state->array == nullptr || limit->array == nullptr ||
      step->array == nullptr || state->byte_size == 0 ||
      (state->dtype != PJRTX_MLX_METAL_DTYPE_F32 &&
       state->dtype != PJRTX_MLX_METAL_DTYPE_BF16) ||
      limit->dtype != state->dtype || step->dtype != state->dtype ||
      state->device_ordinal != limit->device_ordinal ||
      state->device_ordinal != step->device_ordinal ||
      max_iterations > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    return nullptr;
  }
  if (compare_direction != PJRTX_MLX_METAL_COMPARE_LT &&
      compare_direction != PJRTX_MLX_METAL_COMPARE_LE &&
      compare_direction != PJRTX_MLX_METAL_COMPARE_GT &&
      compare_direction != PJRTX_MLX_METAL_COMPARE_GE) {
    return nullptr;
  }
  if (update_op != PJRTX_MLX_METAL_U8_BINARY_ADD &&
      update_op != PJRTX_MLX_METAL_U8_BINARY_SUBTRACT) {
    return nullptr;
  }

  std::vector<int64_t> out_dims;
  if (output_rank != 0) {
    if (output_dims == nullptr) {
      return nullptr;
    }
    out_dims.assign(output_dims, output_dims + output_rank);
  }
  if (out_dims != state->dims) {
    return nullptr;
  }
  const bool limit_scalar = limit->dims.empty();
  const bool step_scalar = step->dims.empty();
  if ((!limit_scalar && limit->dims != out_dims) ||
      (!step_scalar && step->dims != out_dims)) {
    return nullptr;
  }
  const uint64_t elements = element_count_for_shape(out_dims);
  if (elements == 0 ||
      elements > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
    return nullptr;
  }
  const uint64_t byte_size =
      byte_size_for_shape(state->dtype, out_dims);
  if (byte_size == 0 || byte_size != state->byte_size) {
    return nullptr;
  }

  const mlx::core::Device device(mlx::core::Device::gpu, state->device_ordinal);
  if (!mlx::core::is_available(device)) {
    return nullptr;
  }

  try {
    auto state_input = metal_kernel_input_array(
        mlx_astype_array(*state->array, PJRTX_MLX_METAL_DTYPE_F32, device),
        device);
    auto limit_input = metal_kernel_input_array(
        mlx_astype_array(*limit->array, PJRTX_MLX_METAL_DTYPE_F32, device),
        device);
    auto step_input = metal_kernel_input_array(
        mlx_astype_array(*step->array, PJRTX_MLX_METAL_DTYPE_F32, device),
        device);
    const auto kernel = mlx::core::fast::metal_kernel(
        "pjrtx_while_f32_compare_add",
        {"state", "limit", "step"},
        {"out"},
        R"(
          uint elem = thread_position_in_grid.x;
          float value = state[elem];
          float limit_value = LIMIT_SCALAR ? limit[0] : limit[elem];
          float step_value = STEP_SCALAR ? step[0] : step[elem];
          for (uint iter = 0; iter < uint(MAX_ITERATIONS); ++iter) {
            bool keep_going = false;
            if (COMPARE_DIRECTION == 2) {
              keep_going = value >= limit_value;
            } else if (COMPARE_DIRECTION == 3) {
              keep_going = value > limit_value;
            } else if (COMPARE_DIRECTION == 4) {
              keep_going = value <= limit_value;
            } else {
              keep_going = value < limit_value;
            }
            if (!keep_going) {
              break;
            }
            if (UPDATE_OP == 1) {
              value -= step_value;
            } else {
              value += step_value;
            }
          }
          out[elem] = value;
        )");
    auto outputs = kernel(
        {state_input, limit_input, step_input},
        {mlx_shape(out_dims)},
        {mlx::core::float32},
        {static_cast<int>(elements), 1, 1},
        {256, 1, 1},
        {{"MAX_ITERATIONS", static_cast<int>(max_iterations)},
         {"COMPARE_DIRECTION", compare_direction},
         {"UPDATE_OP", update_op},
         {"LIMIT_SCALAR", limit_scalar ? 1 : 0},
         {"STEP_SCALAR", step_scalar ? 1 : 0}},
        std::nullopt,
        false,
        device);
    if (outputs.size() != 1) {
      return nullptr;
    }
    auto out = state->dtype == PJRTX_MLX_METAL_DTYPE_BF16
                   ? mlx::core::astype(outputs[0], mlx::core::bfloat16, device)
                   : std::move(outputs[0]);
    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(std::move(array), byte_size,
                                   state->dtype,
                                   std::move(out_dims),
                                   state->device_ordinal);
  } catch (const std::exception&) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_custom_call_binary_add_f32(
    PjrtxMlxMetalBuffer* lhs, PjrtxMlxMetalBuffer* rhs) {
  if (lhs == nullptr || rhs == nullptr || lhs->byte_size == 0 ||
      lhs->byte_size != rhs->byte_size ||
      lhs->dtype != PJRTX_MLX_METAL_DTYPE_F32 ||
      rhs->dtype != PJRTX_MLX_METAL_DTYPE_F32 || lhs->dims != rhs->dims ||
      lhs->device_ordinal != rhs->device_ordinal || lhs->array == nullptr ||
      rhs->array == nullptr) {
    return nullptr;
  }

  const mlx::core::Device device(mlx::core::Device::gpu, lhs->device_ordinal);
  if (!mlx::core::is_available(device)) {
    return nullptr;
  }

  try {
    const uint64_t elements = lhs->byte_size / sizeof(float);
    if (elements > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
      return nullptr;
    }
    const auto kernel = mlx::core::fast::metal_kernel(
        "pjrtx_custom_call_binary_add_f32",
        {"lhs", "rhs"},
        {"out"},
        R"(
          uint elem = thread_position_in_grid.x;
          out[elem] = lhs[elem] + rhs[elem];
        )");
    auto outputs = kernel(
        {*lhs->array, *rhs->array},
        {mlx_shape(lhs->dims)},
        {mlx::core::float32},
        {static_cast<int>(elements), 1, 1},
        {256, 1, 1},
        {},
        std::nullopt,
        false,
        device);
    if (outputs.size() != 1) {
      return nullptr;
    }
    auto array = std::make_unique<mlx::core::array>(std::move(outputs[0]));
    return new PjrtxMlxMetalBuffer(std::move(array), lhs->byte_size, lhs->dtype,
                                   lhs->dims, lhs->device_ordinal);
  } catch (const std::exception&) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

PjrtxMlxMetalBuffer*
pjrtx_mlx_metal_custom_call_scaled_dot_product_attention(
    PjrtxMlxMetalBuffer* q, PjrtxMlxMetalBuffer* k, PjrtxMlxMetalBuffer* v,
    PjrtxMlxMetalBuffer* token_index) {
  if (q == nullptr || k == nullptr || v == nullptr || token_index == nullptr ||
      q->array == nullptr || k->array == nullptr || v->array == nullptr ||
      token_index->array == nullptr || q->byte_size == 0 || k->byte_size == 0 ||
      v->byte_size == 0 || q->dtype != k->dtype || q->dtype != v->dtype ||
      (q->dtype != PJRTX_MLX_METAL_DTYPE_F32 &&
       q->dtype != PJRTX_MLX_METAL_DTYPE_F16 &&
       q->dtype != PJRTX_MLX_METAL_DTYPE_BF16) ||
      q->device_ordinal != k->device_ordinal ||
      q->device_ordinal != v->device_ordinal ||
      q->device_ordinal != token_index->device_ordinal ||
      (q->dims.size() != 3 && q->dims.size() != 4) ||
      k->dims.size() != q->dims.size() || v->dims.size() != q->dims.size() ||
      k->dims != v->dims) {
    return nullptr;
  }

  const bool has_batch = q->dims.size() == 4;
  const size_t query_axis = has_batch ? 1 : 0;
  const size_t head_axis = has_batch ? 2 : 1;
  const size_t dim_axis = has_batch ? 3 : 2;
  const int64_t batch = has_batch ? q->dims[0] : 1;
  const int64_t queries = q->dims[query_axis];
  const int64_t q_heads = q->dims[head_axis];
  const int64_t kv_len = k->dims[query_axis];
  const int64_t kv_heads = k->dims[head_axis];
  const int64_t head_dim = q->dims[dim_axis];
  if (batch <= 0 || queries <= 0 || q_heads <= 0 || kv_len <= 0 ||
      kv_heads <= 0 || head_dim <= 0 || k->dims[dim_axis] != head_dim ||
      q_heads % kv_heads != 0) {
    return nullptr;
  }
  if (has_batch && (k->dims[0] != batch || v->dims[0] != batch)) {
    return nullptr;
  }

  if (attention_profile_enabled()) {
    const uint64_t sample = g_attention_shape_sample_count.fetch_add(1);
    if (sample < 128) {
      std::fprintf(stderr,
                   "pjrtx_profile event=attention_shape sample=%llu rank=%llu batch=%lld queries=%lld q_heads=%lld kv_len=%lld kv_heads=%lld head_dim=%lld dtype=%d token_rank=%llu token_dtype=%d\n",
                   static_cast<unsigned long long>(sample),
                   static_cast<unsigned long long>(q->dims.size()),
                   static_cast<long long>(batch),
                   static_cast<long long>(queries),
                   static_cast<long long>(q_heads),
                   static_cast<long long>(kv_len),
                   static_cast<long long>(kv_heads),
                   static_cast<long long>(head_dim), q->dtype,
                   static_cast<unsigned long long>(token_index->dims.size()),
                   token_index->dtype);
    }
  }

  const mlx::core::Device device(mlx::core::Device::gpu, q->device_ordinal);
  if (!mlx::core::is_available(device)) {
    return nullptr;
  }

  try {
    auto q4 = has_batch
                  ? mlx::core::transpose(*q->array, std::vector<int>{0, 2, 1, 3},
                                         device)
                  : mlx::core::reshape(
                        mlx::core::transpose(*q->array, std::vector<int>{1, 0, 2},
                                             device),
                        mlx::core::Shape{1, static_cast<int>(q_heads),
                                         static_cast<int>(queries),
                                         static_cast<int>(head_dim)},
                        device);
    auto k4 = has_batch
                  ? mlx::core::transpose(*k->array, std::vector<int>{0, 2, 1, 3},
                                         device)
                  : mlx::core::reshape(
                        mlx::core::transpose(*k->array, std::vector<int>{1, 0, 2},
                                             device),
                        mlx::core::Shape{1, static_cast<int>(kv_heads),
                                         static_cast<int>(kv_len),
                                         static_cast<int>(head_dim)},
                        device);
    auto v4 = has_batch
                  ? mlx::core::transpose(*v->array, std::vector<int>{0, 2, 1, 3},
                                         device)
                  : mlx::core::reshape(
                        mlx::core::transpose(*v->array, std::vector<int>{1, 0, 2},
                                             device),
                        mlx::core::Shape{1, static_cast<int>(kv_heads),
                                         static_cast<int>(kv_len),
                                         static_cast<int>(head_dim)},
                        device);
    auto allowed = attention_allowed_mask(token_index, queries, kv_len, device);

    const float scale =
        1.0f / std::sqrt(static_cast<float>(head_dim));
    auto out4 = mlx::core::fast::scaled_dot_product_attention(
        q4, k4, v4, scale, "", allowed, std::nullopt, device);
    auto out = has_batch
                   ? mlx::core::transpose(out4, std::vector<int>{0, 2, 1, 3},
                                          device)
                   : mlx::core::transpose(
                         mlx::core::reshape(
                             out4,
                             mlx::core::Shape{static_cast<int>(q_heads),
                                              static_cast<int>(queries),
                                              static_cast<int>(head_dim)},
                             device),
                         std::vector<int>{1, 0, 2}, device);
    if (out.shape() != mlx_shape(q->dims)) {
      out = mlx::core::reshape(out, mlx_shape(q->dims), device);
    }
    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(std::move(array), q->byte_size, q->dtype,
                                   q->dims, q->device_ordinal);
  } catch (const std::exception& e) {
    if (env_flag_enabled("PJRTX_TRACE")) {
      std::fprintf(stderr,
                   "pjrtx_trace event=mlx_exception op=scaled_dot_product_attention q_rank=%llu k_rank=%llu dtype=%d what=\"%s\"\n",
                   static_cast<unsigned long long>(q->dims.size()),
                   static_cast<unsigned long long>(k->dims.size()), q->dtype,
                   e.what());
    }
    return nullptr;
  } catch (...) {
    if (env_flag_enabled("PJRTX_TRACE")) {
      std::fprintf(stderr,
                   "pjrtx_trace event=mlx_exception op=scaled_dot_product_attention q_rank=%llu k_rank=%llu dtype=%d what=\"unknown\"\n",
                   static_cast<unsigned long long>(q->dims.size()),
                   static_cast<unsigned long long>(k->dims.size()), q->dtype);
    }
    return nullptr;
  }
}

uint64_t pjrtx_mlx_metal_buffer_size(PjrtxMlxMetalBuffer* buffer) {
  if (buffer == nullptr) {
    return 0;
  }
  return buffer->byte_size;
}

int pjrtx_mlx_metal_buffer_has_host_shadow(PjrtxMlxMetalBuffer* buffer) {
  return buffer == nullptr ? 0 : 0;
}

int pjrtx_mlx_metal_buffer_eval(PjrtxMlxMetalBuffer* buffer) {
  return pjrtx_mlx_metal_buffer_eval_many(&buffer, 1);
}

int pjrtx_mlx_metal_buffer_eval_many(PjrtxMlxMetalBuffer* const* buffers,
                                     uint64_t count) {
  if (buffers == nullptr || count == 0) {
    return 0;
  }

  const bool profile = profile_enabled();
  const bool sync_profile = device_sync_profile_enabled();
  const std::optional<std::string> capture_path = next_eval_capture_path();
  bool capture_started = false;
  try {
    std::vector<mlx::core::array> arrays;
    arrays.reserve(static_cast<size_t>(count));
    int device_ordinal = -1;
    uint64_t byte_size = 0;
    for (uint64_t i = 0; i < count; ++i) {
      PjrtxMlxMetalBuffer* buffer = buffers[i];
      if (buffer == nullptr || buffer->array == nullptr) {
        if (std::getenv("PJRTX_TRACE") != nullptr) {
          std::fprintf(stderr, "pjrtx_trace event=mlx_eval_reject reason=null_buffer index=%llu\n",
                       static_cast<unsigned long long>(i));
        }
        return 0;
      }
      if (device_ordinal < 0) {
        device_ordinal = buffer->device_ordinal;
      } else if (device_ordinal != buffer->device_ordinal) {
        if (std::getenv("PJRTX_TRACE") != nullptr) {
          std::fprintf(stderr, "pjrtx_trace event=mlx_eval_reject reason=device_mismatch index=%llu expected=%d actual=%d\n",
                       static_cast<unsigned long long>(i), device_ordinal,
                       buffer->device_ordinal);
        }
        return 0;
      }
      byte_size += buffer->byte_size;
      arrays.push_back(*buffer->array);
    }
    const mlx::core::Device device(mlx::core::Device::gpu, device_ordinal);
    if (!mlx::core::is_available(device)) {
      if (std::getenv("PJRTX_TRACE") != nullptr) {
        std::fprintf(stderr, "pjrtx_trace event=mlx_eval_reject reason=device_unavailable device=%d\n",
                     device_ordinal);
      }
      return 0;
    }
    if (capture_path.has_value()) {
      try {
        mlx::core::metal::start_capture(*capture_path);
        capture_started = true;
        if (profile) {
          std::fprintf(stderr,
                       "pjrtx_profile event=metal_capture_start path=\"%s\" device=%d buffers=%llu bytes=%llu\n",
                       capture_path->c_str(),
                       device_ordinal,
                       static_cast<unsigned long long>(count),
                       static_cast<unsigned long long>(byte_size));
        }
      } catch (const std::exception& e) {
        if (profile) {
          std::fprintf(stderr,
                       "pjrtx_profile event=metal_capture_reject reason=start_failed path=\"%s\" error=\"%s\" hint=\"%s\"\n",
                       capture_path->c_str(), e.what(),
                       metal_capture_hint(e.what()));
        }
      } catch (...) {
        if (profile) {
          std::fprintf(stderr,
                       "pjrtx_profile event=metal_capture_reject reason=start_failed path=\"%s\" error=\"unknown\" hint=\"\"\n",
                       capture_path->c_str());
        }
      }
    }

    const bool compile_eval = compile_eval_enabled();
    const bool compile_no_fuse = compile_no_fuse_enabled();
    uint64_t compile_create_us = 0;
    uint64_t compile_call_us = 0;
    const uint64_t enqueue_start_us = profile ? now_micros() : 0;
    if (compile_eval) {
      const uint64_t compile_create_start_us = profile ? now_micros() : 0;
      if (compile_no_fuse) {
        mlx::core::set_compile_mode(mlx::core::CompileMode::no_fuse);
      } else {
        mlx::core::set_compile_mode(mlx::core::CompileMode::enabled);
      }
      auto compiled = mlx::core::compile(
          [outputs = std::move(arrays)](const std::vector<mlx::core::array>&) {
            return outputs;
          },
          false);
      compile_create_us =
          profile ? elapsed_micros_since(compile_create_start_us) : 0;
      const uint64_t compile_call_start_us = profile ? now_micros() : 0;
      auto compiled_outputs = compiled({});
      compile_call_us = profile ? elapsed_micros_since(compile_call_start_us) : 0;
      mlx::core::async_eval(std::move(compiled_outputs));
    } else {
      mlx::core::async_eval(std::move(arrays));
    }
    const uint64_t host_enqueue_us =
        profile ? elapsed_micros_since(enqueue_start_us) : 0;

    uint64_t device_sync_wait_us = 0;
    if (sync_profile || capture_started) {
      const uint64_t sync_start_us = now_micros();
      mlx::core::synchronize(mlx::core::default_stream(device));
      device_sync_wait_us = elapsed_micros_since(sync_start_us);
    }

    if (capture_started) {
      try {
        mlx::core::metal::stop_capture();
        if (profile) {
          std::fprintf(stderr,
                       "pjrtx_profile event=metal_capture_stop path=\"%s\" device=%d\n",
                       capture_path->c_str(), device_ordinal);
        }
      } catch (const std::exception& e) {
        if (profile) {
          std::fprintf(stderr,
                       "pjrtx_profile event=metal_capture_stop_failed path=\"%s\" error=\"%s\"\n",
                       capture_path->c_str(), e.what());
        }
      } catch (...) {
        if (profile) {
          std::fprintf(stderr,
                       "pjrtx_profile event=metal_capture_stop_failed path=\"%s\" error=\"unknown\"\n",
                       capture_path->c_str());
        }
      }
    }

    if (profile) {
      const char* compile_mode_label =
          compile_eval ? (compile_no_fuse ? "no_fuse" : "enabled") : "none";
      const int compile_shapeless = compile_eval ? 0 : -1;
      std::fprintf(stderr,
                   "pjrtx_profile event=mlx_eval_many device=%d buffers=%llu bytes=%llu host_enqueue_us=%llu compile_eval=%d compile_scope=per_eval_capturing_outputs compile_mode=%s compile_shapeless=%d compile_create_us=%llu compile_call_us=%llu device_sync_wait_us=%llu measurement=%s capture_path=\"%s\"\n",
                   device_ordinal,
                   static_cast<unsigned long long>(count),
                   static_cast<unsigned long long>(byte_size),
                   static_cast<unsigned long long>(host_enqueue_us),
                   compile_eval ? 1 : 0,
                   compile_mode_label,
                   compile_shapeless,
                   static_cast<unsigned long long>(compile_create_us),
                   static_cast<unsigned long long>(compile_call_us),
                   static_cast<unsigned long long>(device_sync_wait_us),
                   (sync_profile || capture_started) ? "device_sync_wall"
                                                     : "enqueue",
                   capture_started ? capture_path->c_str() : "");
    }
    return 1;
  } catch (const std::exception& e) {
    if (capture_started) {
      try {
        mlx::core::metal::stop_capture();
      } catch (...) {
      }
    }
    if (profile_enabled()) {
      std::fprintf(stderr,
                   "pjrtx_profile event=mlx_eval_many_failed reason=exception error=\"%s\" compile_eval=%d buffers=%llu\n",
                   e.what(), compile_eval_enabled() ? 1 : 0,
                   static_cast<unsigned long long>(count));
    }
    if (compile_eval_enabled()) {
      mlx::core::set_compile_mode(mlx::core::CompileMode::enabled);
    }
    return 0;
  } catch (...) {
    if (capture_started) {
      try {
        mlx::core::metal::stop_capture();
      } catch (...) {
      }
    }
    if (profile_enabled()) {
      std::fprintf(stderr,
                   "pjrtx_profile event=mlx_eval_many_failed reason=unknown_exception compile_eval=%d buffers=%llu\n",
                   compile_eval_enabled() ? 1 : 0,
                   static_cast<unsigned long long>(count));
    }
    if (compile_eval_enabled()) {
      mlx::core::set_compile_mode(mlx::core::CompileMode::enabled);
    }
    return 0;
  }
}

PjrtxMlxMetalProgram* pjrtx_mlx_metal_program_create_with_captures(
    void* user_data, uint64_t full_input_count, uint64_t output_count,
    PjrtxMlxMetalProgramBuildFn build_fn,
    PjrtxMlxMetalBuffer* const* captured_inputs,
    const uint64_t* dynamic_indices, uint64_t dynamic_count) {
  if (build_fn == nullptr || output_count == 0 ||
      (dynamic_count != 0 && dynamic_indices == nullptr)) {
    return nullptr;
  }
  try {
    auto program = std::make_unique<PjrtxMlxMetalProgram>();
    program->user_data = user_data;
    program->input_count = dynamic_count;
    program->full_input_count = full_input_count;
    program->output_count = output_count;
    program->build_fn = build_fn;
    program->captured_inputs.resize(static_cast<size_t>(full_input_count));
    if (dynamic_count != 0) {
      program->dynamic_indices.assign(dynamic_indices,
                                      dynamic_indices + dynamic_count);
    }
    program->full_to_dynamic_input.assign(static_cast<size_t>(full_input_count),
                                          -1);
    for (uint64_t dynamic_index = 0; dynamic_index < dynamic_count;
         ++dynamic_index) {
      const uint64_t full_index = dynamic_indices[dynamic_index];
      if (full_index >= full_input_count ||
          program->full_to_dynamic_input[static_cast<size_t>(full_index)] !=
              -1) {
        return nullptr;
      }
      program->full_to_dynamic_input[static_cast<size_t>(full_index)] =
          static_cast<int64_t>(dynamic_index);
    }

    for (uint64_t full_index = 0; full_index < full_input_count; ++full_index) {
      if (program->full_to_dynamic_input[static_cast<size_t>(full_index)] >=
          0) {
        continue;
      }
      if (captured_inputs == nullptr) {
        return nullptr;
      }
      PjrtxMlxMetalBuffer* captured = captured_inputs[full_index];
      if (captured == nullptr || captured->array == nullptr) {
        return nullptr;
      }
      auto copied =
          buffer_from_array_copy(*captured->array, captured->device_ordinal);
      if (!copied) {
        return nullptr;
      }
      program->captured_inputs[static_cast<size_t>(full_index)] =
          std::move(copied);
    }

    PjrtxMlxMetalProgram* raw = program.get();
    program->compiled = mlx::core::compile(
        [raw](const std::vector<mlx::core::array>& inputs) {
          if (inputs.size() != raw->input_count) {
            throw std::invalid_argument(
                "[pjrtx] compiled program input arity mismatch");
          }

          std::vector<std::unique_ptr<PjrtxMlxMetalBuffer>> input_storage;
          input_storage.reserve(inputs.size());
          std::vector<PjrtxMlxMetalBuffer*> input_ptrs(
              static_cast<size_t>(raw->full_input_count), nullptr);
          for (const auto& input : inputs) {
            auto wrapper =
                buffer_from_array_copy(input, raw->active_device_ordinal);
            if (!wrapper) {
              throw std::invalid_argument(
                  "[pjrtx] failed to wrap compiled program input");
            }
            input_storage.push_back(std::move(wrapper));
          }
          for (uint64_t full_index = 0; full_index < raw->full_input_count;
               ++full_index) {
            const int64_t dynamic_index =
                raw->full_to_dynamic_input[static_cast<size_t>(full_index)];
            if (dynamic_index >= 0) {
              input_ptrs[static_cast<size_t>(full_index)] =
                  input_storage[static_cast<size_t>(dynamic_index)].get();
            } else {
              input_ptrs[static_cast<size_t>(full_index)] =
                  raw->captured_inputs[static_cast<size_t>(full_index)].get();
            }
          }

          std::vector<PjrtxMlxMetalBuffer*> output_ptrs(
              static_cast<size_t>(raw->output_count), nullptr);
          const int ok = raw->build_fn(
              raw->user_data, input_ptrs.data(), raw->full_input_count,
              output_ptrs.data(), raw->output_count);
          if (ok == 0) {
            throw std::runtime_error(
                "[pjrtx] compiled program graph builder failed");
          }

          std::vector<mlx::core::array> outputs;
          outputs.reserve(output_ptrs.size());
          for (auto* output : output_ptrs) {
            if (output == nullptr || output->array == nullptr) {
              for (auto* maybe_output : output_ptrs) {
                pjrtx_mlx_metal_buffer_destroy(maybe_output);
              }
              throw std::runtime_error(
                  "[pjrtx] compiled program graph builder returned null output");
            }
            outputs.push_back(*output->array);
          }
          for (auto* output : output_ptrs) {
            pjrtx_mlx_metal_buffer_destroy(output);
          }
          return outputs;
        },
        false);
    return program.release();
  } catch (const std::exception& e) {
    if (profile_enabled()) {
      std::fprintf(stderr,
                   "pjrtx_profile event=mlx_program_create_failed error=\"%s\"\n",
                   e.what());
    }
    return nullptr;
  } catch (...) {
    if (profile_enabled()) {
      std::fprintf(stderr,
                   "pjrtx_profile event=mlx_program_create_failed error=\"unknown\"\n");
    }
    return nullptr;
  }
}

PjrtxMlxMetalProgram* pjrtx_mlx_metal_program_create(
    void* user_data, uint64_t input_count, uint64_t output_count,
    PjrtxMlxMetalProgramBuildFn build_fn) {
  std::vector<uint64_t> dynamic_indices(static_cast<size_t>(input_count));
  for (uint64_t index = 0; index < input_count; ++index) {
    dynamic_indices[static_cast<size_t>(index)] = index;
  }
  return pjrtx_mlx_metal_program_create_with_captures(
      user_data, input_count, output_count, build_fn,
      nullptr, dynamic_indices.data(), input_count);
}

int pjrtx_mlx_metal_program_execute(
    PjrtxMlxMetalProgram* program, PjrtxMlxMetalBuffer* const* inputs,
    uint64_t input_count, PjrtxMlxMetalBuffer*** out_outputs,
    uint64_t* out_output_count) {
  return pjrtx_mlx_metal_program_execute_with_donation(
      program, inputs, input_count, nullptr, 0, out_outputs, out_output_count,
      nullptr);
}

int pjrtx_mlx_metal_program_execute_with_donation(
    PjrtxMlxMetalProgram* program, PjrtxMlxMetalBuffer* const* inputs,
    uint64_t input_count, const uint64_t* donated_input_indices,
    uint64_t donated_input_count, PjrtxMlxMetalBuffer*** out_outputs,
    uint64_t* out_output_count,
    PjrtxMlxMetalProgramExecuteProfile* out_profile) {
  if (program == nullptr || out_outputs == nullptr ||
      out_output_count == nullptr || input_count != program->input_count ||
      (input_count != 0 && inputs == nullptr) ||
      (donated_input_count != 0 && donated_input_indices == nullptr)) {
    return 0;
  }
  *out_outputs = nullptr;
  *out_output_count = 0;
  if (out_profile != nullptr) {
    *out_profile = {};
  }
  for (uint64_t i = 0; i < donated_input_count; ++i) {
    const uint64_t input_index = donated_input_indices[i];
    if (input_index >= input_count) {
      return 0;
    }
  }

  const bool profile = profile_enabled();
  const bool sync_profile = device_sync_profile_enabled();
  const bool no_fuse = program_compile_no_fuse_enabled();
  const uint64_t start_us = profile ? now_micros() : 0;
  std::optional<std::string> capture_path;
  bool capture_started = false;
  std::lock_guard<std::mutex> lock(program->mutex);
  try {
    std::vector<mlx::core::array> arrays;
    arrays.reserve(static_cast<size_t>(input_count));
    std::vector<uint8_t> donated(static_cast<size_t>(input_count), 0);
    for (uint64_t i = 0; i < donated_input_count; ++i) {
      donated[static_cast<size_t>(donated_input_indices[i])] = 1;
    }
    int device_ordinal = -1;
    for (uint64_t i = 0; i < input_count; ++i) {
      PjrtxMlxMetalBuffer* input = inputs[i];
      if (input == nullptr || input->array == nullptr) {
        return 0;
      }
      if (device_ordinal < 0) {
        device_ordinal = input->device_ordinal;
      } else if (device_ordinal != input->device_ordinal) {
        return 0;
      }
      if (donated[static_cast<size_t>(i)] != 0) {
        arrays.push_back(std::move(*input->array));
      } else {
        arrays.push_back(*input->array);
      }
    }
    if (device_ordinal < 0) {
      for (const auto& captured : program->captured_inputs) {
        if (captured != nullptr) {
          device_ordinal = captured->device_ordinal;
          break;
        }
      }
    }
    if (device_ordinal < 0) {
      device_ordinal = 0;
    }
    const mlx::core::Device device(mlx::core::Device::gpu, device_ordinal);
    if (!mlx::core::is_available(device)) {
      return 0;
    }
    capture_path = next_program_capture_path();
    if (capture_path.has_value()) {
      try {
        mlx::core::metal::start_capture(*capture_path);
        capture_started = true;
        if (profile) {
          std::fprintf(
              stderr,
              "pjrtx_profile event=metal_capture_start path=\"%s\" device=%d program=0x%llx inputs=%llu full_inputs=%llu captured_inputs=%llu\n",
              capture_path->c_str(), device_ordinal,
              static_cast<unsigned long long>(
                  reinterpret_cast<std::uintptr_t>(program)),
              static_cast<unsigned long long>(input_count),
              static_cast<unsigned long long>(program->full_input_count),
              static_cast<unsigned long long>(
                  program->full_input_count - program->input_count));
        }
      } catch (const std::exception& e) {
        if (profile) {
          std::fprintf(stderr,
                       "pjrtx_profile event=metal_capture_reject reason=start_failed path=\"%s\" error=\"%s\" hint=\"%s\"\n",
                       capture_path->c_str(), e.what(),
                       metal_capture_hint(e.what()));
        }
      } catch (...) {
        if (profile) {
          std::fprintf(stderr,
                       "pjrtx_profile event=metal_capture_reject reason=start_failed path=\"%s\" error=\"unknown\" hint=\"\"\n",
                       capture_path->c_str());
        }
      }
    }

    program->active_device_ordinal = device_ordinal;
    mlx::core::set_compile_mode(no_fuse ? mlx::core::CompileMode::no_fuse
                                        : mlx::core::CompileMode::enabled);
    const bool first_execute = program->execute_count == 0;
    auto compiled_outputs = program->compiled(arrays);
    for (uint64_t i = 0; i < input_count; ++i) {
      if (donated[static_cast<size_t>(i)] != 0) {
        inputs[i]->array.reset();
      }
    }
    mlx::core::set_compile_mode(mlx::core::CompileMode::enabled);
    if (program_async_eval_enabled()) {
      if (auto max_output_bytes = program_async_eval_max_output_bytes()) {
        std::vector<mlx::core::array> eval_outputs;
        eval_outputs.reserve(compiled_outputs.size());
        for (const auto& output : compiled_outputs) {
          if (static_cast<uint64_t>(output.nbytes()) <= *max_output_bytes) {
            eval_outputs.push_back(output);
          }
        }
        if (!eval_outputs.empty()) {
          mlx::core::async_eval(eval_outputs);
        }
      } else {
        mlx::core::async_eval(compiled_outputs);
      }
    }
    const uint64_t host_enqueue_us =
        profile ? elapsed_micros_since(start_us) : 0;

    uint64_t device_sync_wait_us = 0;
    if (sync_profile || capture_started) {
      const uint64_t sync_start_us = now_micros();
      mlx::core::synchronize(mlx::core::default_stream(device));
      device_sync_wait_us = elapsed_micros_since(sync_start_us);
    }

    if (capture_started) {
      try {
        mlx::core::metal::stop_capture();
        if (profile) {
          std::fprintf(
              stderr,
              "pjrtx_profile event=metal_capture_stop path=\"%s\" device=%d program=0x%llx\n",
              capture_path->c_str(), device_ordinal,
              static_cast<unsigned long long>(
                  reinterpret_cast<std::uintptr_t>(program)));
        }
      } catch (const std::exception& e) {
        if (profile) {
          std::fprintf(stderr,
                       "pjrtx_profile event=metal_capture_stop_failed path=\"%s\" error=\"%s\"\n",
                       capture_path->c_str(), e.what());
        }
      } catch (...) {
        if (profile) {
          std::fprintf(stderr,
                       "pjrtx_profile event=metal_capture_stop_failed path=\"%s\" error=\"unknown\"\n",
                       capture_path->c_str());
        }
      }
    }

    auto** result = new PjrtxMlxMetalBuffer*[compiled_outputs.size()];
    uint64_t initialized = 0;
    try {
      for (const auto& output : compiled_outputs) {
        auto wrapper = buffer_from_array_copy(output, device_ordinal);
        if (!wrapper) {
          throw std::runtime_error(
              "[pjrtx] failed to wrap compiled program output");
        }
        result[initialized] = wrapper.release();
        initialized += 1;
      }
    } catch (...) {
      for (uint64_t i = 0; i < initialized; ++i) {
        pjrtx_mlx_metal_buffer_destroy(result[i]);
      }
      delete[] result;
      throw;
    }

    program->execute_count += 1;
    *out_outputs = result;
    *out_output_count = static_cast<uint64_t>(compiled_outputs.size());
    if (out_profile != nullptr) {
      out_profile->host_enqueue_us = host_enqueue_us;
      out_profile->device_sync_wait_us = device_sync_wait_us;
      out_profile->output_count = *out_output_count;
      out_profile->donated_input_count = donated_input_count;
      out_profile->measured_device_sync =
          (sync_profile || capture_started) ? 1 : 0;
      out_profile->first_execute = first_execute ? 1 : 0;
    }
    if (profile) {
      std::fprintf(stderr,
                   "pjrtx_profile event=mlx_program_execute program=0x%llx inputs=%llu full_inputs=%llu captured_inputs=%llu donated_inputs=%llu outputs=%llu first_execute=%d compile_mode=%s host_enqueue_us=%llu device_sync_wait_us=%llu measurement=%s capture_path=\"%s\"\n",
                   static_cast<unsigned long long>(
                       reinterpret_cast<std::uintptr_t>(program)),
                   static_cast<unsigned long long>(input_count),
                   static_cast<unsigned long long>(program->full_input_count),
                   static_cast<unsigned long long>(
                       program->full_input_count - program->input_count),
                   static_cast<unsigned long long>(donated_input_count),
                   static_cast<unsigned long long>(*out_output_count),
                   first_execute ? 1 : 0, no_fuse ? "no_fuse" : "enabled",
                   static_cast<unsigned long long>(host_enqueue_us),
                   static_cast<unsigned long long>(device_sync_wait_us),
                   (sync_profile || capture_started) ? "device_sync_wall"
                                                     : "enqueue",
                   capture_started ? capture_path->c_str() : "");
    }
    return 1;
  } catch (const std::exception& e) {
    if (capture_started) {
      try {
        mlx::core::metal::stop_capture();
      } catch (...) {
      }
    }
    mlx::core::set_compile_mode(mlx::core::CompileMode::enabled);
    if (profile_enabled()) {
      std::fprintf(stderr,
                   "pjrtx_profile event=mlx_program_execute_failed program=0x%llx error=\"%s\"\n",
                   static_cast<unsigned long long>(
                       reinterpret_cast<std::uintptr_t>(program)),
                   e.what());
    }
    return 0;
  } catch (...) {
    if (capture_started) {
      try {
        mlx::core::metal::stop_capture();
      } catch (...) {
      }
    }
    mlx::core::set_compile_mode(mlx::core::CompileMode::enabled);
    if (profile_enabled()) {
      std::fprintf(stderr,
                   "pjrtx_profile event=mlx_program_execute_failed program=0x%llx error=\"unknown\"\n",
                   static_cast<unsigned long long>(
                       reinterpret_cast<std::uintptr_t>(program)));
    }
    return 0;
  }
}

void pjrtx_mlx_metal_program_output_array_destroy(
    PjrtxMlxMetalBuffer** outputs) {
  delete[] outputs;
}

void pjrtx_mlx_metal_program_destroy(PjrtxMlxMetalProgram* program) {
  delete program;
}

int pjrtx_mlx_metal_buffer_copy_to_host(PjrtxMlxMetalBuffer* buffer, void* dst,
                                        uint64_t dst_size) {
  if (buffer == nullptr || dst == nullptr || dst_size < buffer->byte_size) {
    if (std::getenv("PJRTX_TRACE") != nullptr) {
      std::fprintf(stderr, "pjrtx_trace event=mlx_copy_reject reason=invalid_args dst_size=%llu byte_size=%llu\n",
                   static_cast<unsigned long long>(dst_size),
                   static_cast<unsigned long long>(buffer == nullptr ? 0 : buffer->byte_size));
    }
    return 0;
  }

  const bool profile = profile_enabled();
  const uint64_t start_us = profile ? now_micros() : 0;
  try {
    if (buffer->array == nullptr) {
      if (std::getenv("PJRTX_TRACE") != nullptr) {
        std::fprintf(stderr, "pjrtx_trace event=mlx_copy_reject reason=null_array\n");
      }
      return 0;
    }
    const mlx::core::Device device(mlx::core::Device::gpu,
                                   buffer->device_ordinal);
    auto contiguous = mlx::core::contiguous(*buffer->array, false, device);
    eval_on_device(contiguous, device);
    std::memcpy(dst, contiguous.data<uint8_t>(),
                static_cast<size_t>(buffer->byte_size));
    if (profile) {
      std::fprintf(stderr,
                   "pjrtx_profile event=mlx_copy_to_host device=%d dtype=%d rank=%llu bytes=%llu elapsed_us=%llu\n",
                   buffer->device_ordinal, buffer->dtype,
                   static_cast<unsigned long long>(buffer->dims.size()),
                   static_cast<unsigned long long>(buffer->byte_size),
                   static_cast<unsigned long long>(elapsed_micros_since(start_us)));
    }
    return 1;
  } catch (const std::exception& e) {
    if (std::getenv("PJRTX_TRACE") != nullptr) {
      std::fprintf(stderr, "pjrtx_trace event=mlx_exception op=copy_to_host what=\"%s\"\n", e.what());
    }
    return 0;
  } catch (...) {
    if (std::getenv("PJRTX_TRACE") != nullptr) {
      std::fprintf(stderr, "pjrtx_trace event=mlx_exception op=copy_to_host what=\"unknown\"\n");
    }
    return 0;
  }
}

void pjrtx_mlx_metal_buffer_destroy(PjrtxMlxMetalBuffer* buffer) {
  if (buffer == nullptr) {
    return;
  }
  delete buffer;
}
