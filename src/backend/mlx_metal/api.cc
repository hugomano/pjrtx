#include "src/backend/mlx_metal/api.h"

#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "mlx/array.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/device.h"
#include "mlx/ops.h"
#include "mlx/stream.h"
#include "mlx/transforms.h"
#include "mlx/version.h"

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

namespace {

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
    case PJRTX_MLX_METAL_DTYPE_S8:
      return sizeof(int8_t);
    case PJRTX_MLX_METAL_DTYPE_S32:
      return sizeof(int32_t);
    case PJRTX_MLX_METAL_DTYPE_U32:
      return sizeof(uint32_t);
    case PJRTX_MLX_METAL_DTYPE_F16:
      return sizeof(uint16_t);
    case PJRTX_MLX_METAL_DTYPE_BF16:
      return sizeof(uint16_t);
    case PJRTX_MLX_METAL_DTYPE_F32:
      return sizeof(float);
    default:
      return 0;
  }
}

const mlx::core::Dtype* mlx_dtype_from_code(int dtype) {
  switch (dtype) {
    case PJRTX_MLX_METAL_DTYPE_U8:
      return &mlx::core::uint8;
    case PJRTX_MLX_METAL_DTYPE_S8:
      return &mlx::core::int8;
    case PJRTX_MLX_METAL_DTYPE_S32:
      return &mlx::core::int32;
    case PJRTX_MLX_METAL_DTYPE_U32:
      return &mlx::core::uint32;
    case PJRTX_MLX_METAL_DTYPE_F16:
      return &mlx::core::float16;
    case PJRTX_MLX_METAL_DTYPE_BF16:
      return &mlx::core::bfloat16;
    case PJRTX_MLX_METAL_DTYPE_F32:
      return &mlx::core::float32;
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
    case PJRTX_MLX_METAL_DTYPE_S8:
      return mlx::core::astype(src, mlx::core::int8, device);
    case PJRTX_MLX_METAL_DTYPE_S32:
      return mlx::core::astype(src, mlx::core::int32, device);
    case PJRTX_MLX_METAL_DTYPE_U32:
      return mlx::core::astype(src, mlx::core::uint32, device);
    case PJRTX_MLX_METAL_DTYPE_F16:
      return mlx::core::astype(src, mlx::core::float16, device);
    case PJRTX_MLX_METAL_DTYPE_BF16:
      return mlx::core::astype(src, mlx::core::bfloat16, device);
    case PJRTX_MLX_METAL_DTYPE_F32:
      return mlx::core::astype(src, mlx::core::float32, device);
    default:
      return src;
  }
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
    default:
      throw std::invalid_argument("unknown PjRTx MLX unary op");
  }
}

int mlx_unary_output_dtype(int input_dtype, int op) {
  return op == PJRTX_MLX_METAL_UNARY_ISFINITE ? PJRTX_MLX_METAL_DTYPE_PRED
                                              : input_dtype;
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
  if (lhs_k != static_cast<int64_t>(lhs_dims.size() - 1) ||
      rhs_k != static_cast<int64_t>(rhs_dims.size() - 2)) {
    return false;
  }
  if (lhs_dims[static_cast<size_t>(lhs_k)] !=
      rhs_dims[static_cast<size_t>(rhs_k)]) {
    return false;
  }
  for (size_t i = 0; i < lhs_batch.size(); ++i) {
    if (lhs_batch[i] < 0 || rhs_batch[i] < 0 ||
        static_cast<size_t>(lhs_batch[i]) >= lhs_dims.size() ||
        static_cast<size_t>(rhs_batch[i]) >= rhs_dims.size() ||
        lhs_dims[static_cast<size_t>(lhs_batch[i])] !=
            rhs_dims[static_cast<size_t>(rhs_batch[i])]) {
      return false;
    }
  }
  return true;
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
  } catch (const std::exception&) {
    return 0;
  } catch (...) {
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
    } catch (const std::exception&) {
      array.reset();
    } catch (...) {
      array.reset();
    }
    if (array == nullptr) {
      return nullptr;
    }
    return new PjrtxMlxMetalBuffer(std::move(array), byte_size, dtype,
                                   std::move(shape), device_ordinal);
  } catch (const std::exception&) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_iota(
    int device_ordinal, int dtype, const int64_t* dims, uint64_t rank,
    int64_t iota_dimension) {
  if (dims == nullptr || iota_dimension < 0 ||
      static_cast<uint64_t>(iota_dimension) >= rank) {
    return nullptr;
  }
  std::vector<int64_t> out_dims(dims, dims + rank);
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
  if (lhs == nullptr || rhs == nullptr || lhs->byte_size == 0 ||
      lhs->byte_size != rhs->byte_size || lhs->dtype != rhs->dtype ||
      lhs->dims != rhs->dims || lhs->device_ordinal != rhs->device_ordinal) {
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
    auto out = mlx_binary_array(*lhs->array, *rhs->array, op, lhs->dtype, device);
    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(std::move(array), lhs->byte_size,
                                   lhs->dtype, lhs->dims, lhs->device_ordinal);
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
  std::vector<int64_t> out_dims(output_dims, output_dims + output_rank);
  for (uint64_t axis = 0; axis < rank; ++axis) {
    if (low[axis] < 0 || high[axis] < 0 || interior_padding[axis] != 0 ||
        out_dims[axis] != src->dims[axis] + low[axis] + high[axis]) {
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
    auto out = mlx::core::pad(*src->array, all_axes(rank), mlx_shape(low),
                              mlx_shape(high), *padding_value->array,
                              "constant", device);
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

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_dot_general(
    PjrtxMlxMetalBuffer* lhs, PjrtxMlxMetalBuffer* rhs,
    const int64_t* lhs_batch_dimensions, uint64_t lhs_batch_rank,
    const int64_t* rhs_batch_dimensions, uint64_t rhs_batch_rank,
    const int64_t* lhs_contracting_dimensions, uint64_t lhs_contracting_rank,
    const int64_t* rhs_contracting_dimensions, uint64_t rhs_contracting_rank,
    const int64_t* output_dims, uint64_t output_rank) {
  if (lhs == nullptr || rhs == nullptr || output_dims == nullptr ||
      lhs->dtype != PJRTX_MLX_METAL_DTYPE_F32 ||
      rhs->dtype != PJRTX_MLX_METAL_DTYPE_F32 ||
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
  const uint64_t byte_size = byte_size_for_shape(PJRTX_MLX_METAL_DTYPE_F32,
                                                 out_dims);
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
    auto out = mlx::core::matmul(*lhs->array, *rhs->array, device);
    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(std::move(array), byte_size,
                                   PJRTX_MLX_METAL_DTYPE_F32,
                                   std::move(out_dims), lhs->device_ordinal);
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
      (num_dimensions > 0 && dimensions == nullptr) ||
      src->dtype != PJRTX_MLX_METAL_DTYPE_F32) {
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
            : mlx::core::max(*src->array, axes, false, device);
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

PjrtxMlxMetalBuffer* pjrtx_mlx_metal_buffer_compare(
    PjrtxMlxMetalBuffer* lhs, PjrtxMlxMetalBuffer* rhs, int direction,
    const int64_t* output_dims, uint64_t output_rank) {
  if (lhs == nullptr || rhs == nullptr || output_dims == nullptr ||
      lhs->byte_size != rhs->byte_size || lhs->dtype != rhs->dtype ||
      lhs->dims != rhs->dims || lhs->device_ordinal != rhs->device_ordinal) {
    return nullptr;
  }
  std::vector<int64_t> out_dims(output_dims, output_dims + output_rank);
  const uint64_t byte_size = byte_size_for_shape(PJRTX_MLX_METAL_DTYPE_PRED,
                                                 out_dims);
  if (byte_size == 0 || out_dims != lhs->dims) {
    return nullptr;
  }
  const mlx::core::Device device(mlx::core::Device::gpu, lhs->device_ordinal);
  if (!mlx::core::is_available(device) || lhs->array == nullptr ||
      rhs->array == nullptr) {
    return nullptr;
  }
  try {
    auto out = mlx_compare_array(*lhs->array, *rhs->array, direction, device);
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

uint64_t pjrtx_mlx_metal_buffer_size(PjrtxMlxMetalBuffer* buffer) {
  if (buffer == nullptr) {
    return 0;
  }
  return buffer->byte_size;
}

int pjrtx_mlx_metal_buffer_has_host_shadow(PjrtxMlxMetalBuffer* buffer) {
  return buffer == nullptr ? 0 : 0;
}

int pjrtx_mlx_metal_buffer_copy_to_host(PjrtxMlxMetalBuffer* buffer, void* dst,
                                        uint64_t dst_size) {
  if (buffer == nullptr || dst == nullptr || dst_size < buffer->byte_size) {
    return 0;
  }

  try {
    if (buffer->array == nullptr) {
      return 0;
    }
    const mlx::core::Device device(mlx::core::Device::gpu,
                                   buffer->device_ordinal);
    auto contiguous = mlx::core::contiguous(*buffer->array, false, device);
    eval_on_device(contiguous, device);
    std::memcpy(dst, contiguous.data<uint8_t>(),
                static_cast<size_t>(buffer->byte_size));
    return 1;
  } catch (const std::exception&) {
    return 0;
  } catch (...) {
    return 0;
  }
}

void pjrtx_mlx_metal_buffer_destroy(PjrtxMlxMetalBuffer* buffer) {
  if (buffer == nullptr) {
    return;
  }
  delete buffer;
}
