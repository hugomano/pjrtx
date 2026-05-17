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
  std::vector<uint8_t> host;
  std::unique_ptr<mlx::core::array> array;
  uint64_t byte_size;
  int dtype;
  std::vector<int64_t> dims;
  int device_ordinal;

  PjrtxMlxMetalBuffer(std::vector<uint8_t> host_,
                      std::unique_ptr<mlx::core::array> array_,
                      uint64_t byte_size_, int dtype_,
                      std::vector<int64_t> dims_, int device_ordinal_)
      : host(std::move(host_)),
        array(std::move(array_)),
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
    case PJRTX_MLX_METAL_DTYPE_F32:
      return sizeof(float);
    default:
      return 0;
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

std::unique_ptr<mlx::core::array> make_mlx_array(const std::vector<uint8_t>& host,
                                                 int dtype,
                                                 const std::vector<int64_t>& dims) {
  const auto shape = mlx_shape(dims);
  switch (dtype) {
    case PJRTX_MLX_METAL_DTYPE_PRED:
      return std::make_unique<mlx::core::array>(
          host.begin(), shape, mlx::core::bool_);
    case PJRTX_MLX_METAL_DTYPE_U8:
      return std::make_unique<mlx::core::array>(
          host.begin(), shape, mlx::core::uint8);
    case PJRTX_MLX_METAL_DTYPE_F32: {
      std::vector<float> values(host.size() / sizeof(float));
      std::memcpy(values.data(), host.data(), host.size());
      return std::make_unique<mlx::core::array>(
          values.begin(), shape, mlx::core::float32);
    }
    default:
      return nullptr;
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
    default:
      throw std::invalid_argument("unknown PjRTx MLX unary op");
  }
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

std::vector<uint8_t> copy_array_bytes(mlx::core::array& array,
                                      uint64_t byte_size,
                                      const mlx::core::Device& device) {
  eval_on_device(array, device);
  std::vector<uint8_t> host(byte_size);
  std::memcpy(host.data(), array.data<uint8_t>(), host.size());
  return host;
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
    std::vector<uint8_t> host(bytes, bytes + byte_size);
    std::unique_ptr<mlx::core::array> array;
    try {
      array = make_mlx_array(host, dtype, shape);
    } catch (const std::exception&) {
      array.reset();
    } catch (...) {
      array.reset();
    }
    return new PjrtxMlxMetalBuffer(std::move(host), std::move(array),
                                   byte_size, dtype, std::move(shape),
                                   device_ordinal);
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
    return pjrtx_mlx_metal_buffer_from_host_typed(
        src->device_ordinal, src->host.data(), src->host.size(), src->dtype,
        src->dims.data(), src->dims.size());
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
    eval_on_device(out, device);
    std::vector<uint8_t> host(lhs->byte_size);
    std::memcpy(host.data(), out.data<uint8_t>(), host.size());
    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(std::move(host), std::move(array),
                                   lhs->byte_size, lhs->dtype, lhs->dims,
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
    eval_on_device(out, device);
    std::vector<uint8_t> host(src->byte_size);
    std::memcpy(host.data(), out.data<uint8_t>(), host.size());
    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(std::move(host), std::move(array),
                                   src->byte_size, src->dtype, src->dims,
                                   src->device_ordinal);
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
    eval_on_device(out, device);
    std::vector<uint8_t> host(src->byte_size);
    std::memcpy(host.data(), out.data<uint8_t>(), host.size());
    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(
        std::move(host), std::move(array), src->byte_size, src->dtype,
        permuted_dims(src->dims, axes), src->device_ordinal);
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
    eval_on_device(out, device);
    uint64_t byte_size = byte_size_for_shape(src->dtype, out_dims);
    std::vector<uint8_t> host(byte_size);
    std::memcpy(host.data(), out.data<uint8_t>(), host.size());
    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(std::move(host), std::move(array), byte_size,
                                   src->dtype, std::move(out_dims),
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
    eval_on_device(out, device);
    std::vector<uint8_t> host(byte_size);
    std::memcpy(host.data(), out.data<uint8_t>(), host.size());
    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(std::move(host), std::move(array), byte_size,
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
    eval_on_device(out, device);
    std::vector<uint8_t> host(byte_size);
    std::memcpy(host.data(), out.data<uint8_t>(), host.size());
    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(std::move(host), std::move(array), byte_size,
                                   lhs->dtype, std::move(out_dims),
                                   lhs->device_ordinal);
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
    auto host = copy_array_bytes(out, byte_size, device);
    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(std::move(host), std::move(array),
                                   byte_size, PJRTX_MLX_METAL_DTYPE_F32,
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
    auto host = copy_array_bytes(out, byte_size, device);
    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(std::move(host), std::move(array),
                                   byte_size, src->dtype, std::move(out_dims),
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
      lhs->dims != rhs->dims || lhs->device_ordinal != rhs->device_ordinal ||
      lhs->dtype != PJRTX_MLX_METAL_DTYPE_F32) {
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
    auto host = copy_array_bytes(out, byte_size, device);
    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(std::move(host), std::move(array),
                                   byte_size, PJRTX_MLX_METAL_DTYPE_PRED,
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
    auto host = copy_array_bytes(out, byte_size, device);
    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(std::move(host), std::move(array),
                                   byte_size, on_true->dtype,
                                   std::move(out_dims),
                                   on_true->device_ordinal);
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

int pjrtx_mlx_metal_buffer_copy_to_host(PjrtxMlxMetalBuffer* buffer, void* dst,
                                        uint64_t dst_size) {
  if (buffer == nullptr || dst == nullptr || dst_size < buffer->byte_size) {
    return 0;
  }

  try {
    if (buffer->array != nullptr) {
      const mlx::core::Device device(mlx::core::Device::gpu,
                                     buffer->device_ordinal);
      eval_on_device(*buffer->array, device);
      std::memcpy(dst, buffer->array->data<uint8_t>(),
                  static_cast<size_t>(buffer->byte_size));
      return 1;
    }
    std::memcpy(dst, buffer->host.data(),
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
