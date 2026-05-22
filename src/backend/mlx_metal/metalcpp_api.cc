#include "src/backend/mlx_metal/api.h"

#include <Metal/Metal.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "mlx/allocator.h"
#include "mlx/array.h"
#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"
#include "mlx/device.h"
#include "mlx/dtype.h"
#include "mlx/stream.h"
#include "src/backend/mlx_metal/api_internal.h"

struct PjrtxMetalCppExecutableOwnedStep {
  PjrtxMetalCppFusionKernel kernel;
  std::vector<uint64_t> inputs;
  std::vector<uint64_t> outputs;
  std::vector<uint64_t> release_values;
  int64_t in_place_input = -1;
  bool alias = false;
};

struct PjrtxMetalCppExecutableProgram {
  int device_ordinal;
  std::vector<PjrtxMetalCppOwnedTensorSpec> values;
  std::vector<uint64_t> input_values;
  std::vector<uint64_t> output_values;
  std::vector<PjrtxMetalCppExecutableOwnedStep> steps;
};

namespace {

thread_local std::string g_last_error;
std::mutex g_kernel_compile_mutex;

void clear_last_error() { g_last_error.clear(); }

bool fail_with(const std::string &message) {
  g_last_error = message;
  return false;
}

bool fail_with(const char *message) {
  g_last_error = message == nullptr ? "unknown" : message;
  return false;
}

void print_last_error_if_tracing(const char *event) {
  if (g_last_error.empty()) {
    return;
  }
  if (std::getenv("PJRTX_TRACE") == nullptr &&
      std::getenv("PJRTX_PROFILE") == nullptr) {
    return;
  }
  std::fprintf(stderr, "pjrtx_trace event=%s error=\"%s\"\n", event,
               g_last_error.c_str());
}

std::optional<mlx::core::Dtype> mlx_dtype_from_code(int dtype) {
  switch (dtype) {
  case PJRTX_MLX_METAL_DTYPE_PRED:
    return mlx::core::bool_;
  case PJRTX_MLX_METAL_DTYPE_U8:
    return mlx::core::uint8;
  case PJRTX_MLX_METAL_DTYPE_S32:
    return mlx::core::int32;
  case PJRTX_MLX_METAL_DTYPE_U32:
    return mlx::core::uint32;
  case PJRTX_MLX_METAL_DTYPE_U64:
    return mlx::core::uint64;
  case PJRTX_MLX_METAL_DTYPE_F16:
    return mlx::core::float16;
  case PJRTX_MLX_METAL_DTYPE_F32:
    return mlx::core::float32;
  case PJRTX_MLX_METAL_DTYPE_BF16:
    return mlx::core::bfloat16;
  default:
    return std::nullopt;
  }
}

size_t dtype_size(int dtype) {
  switch (dtype) {
  case PJRTX_MLX_METAL_DTYPE_PRED:
  case PJRTX_MLX_METAL_DTYPE_U8:
    return 1;
  case PJRTX_MLX_METAL_DTYPE_F16:
  case PJRTX_MLX_METAL_DTYPE_BF16:
    return 2;
  case PJRTX_MLX_METAL_DTYPE_S32:
  case PJRTX_MLX_METAL_DTYPE_U32:
  case PJRTX_MLX_METAL_DTYPE_F32:
    return 4;
  case PJRTX_MLX_METAL_DTYPE_U64:
    return 8;
  default:
    return 0;
  }
}

mlx::core::Shape mlx_shape(const std::vector<int64_t> &dims) {
  mlx::core::Shape shape;
  shape.reserve(dims.size());
  for (int64_t dim : dims) {
    shape.push_back(static_cast<mlx::core::ShapeElem>(dim));
  }
  return shape;
}

uint64_t element_count_for_shape(const std::vector<int64_t> &dims) {
  if (dims.empty()) {
    return 1;
  }
  uint64_t count = 1;
  for (int64_t dim : dims) {
    if (dim < 0) {
      return 0;
    }
    const uint64_t u_dim = static_cast<uint64_t>(dim);
    if (u_dim != 0 && count > std::numeric_limits<uint64_t>::max() / u_dim) {
      return 0;
    }
    count *= u_dim;
  }
  return count;
}

uint64_t byte_size_for_shape(int dtype, const std::vector<int64_t> &dims) {
  const size_t elem_size = dtype_size(dtype);
  const uint64_t elems = element_count_for_shape(dims);
  if (elem_size == 0 || elems == 0 ||
      elems > std::numeric_limits<uint64_t>::max() / elem_size) {
    return 0;
  }
  return elems * elem_size;
}

bool supported_dense_binary_dtype(int dtype) {
  switch (dtype) {
  case PJRTX_MLX_METAL_DTYPE_F32:
  case PJRTX_MLX_METAL_DTYPE_F16:
    return true;
  default:
    return false;
  }
}

bool supported_tensor_spec_dtype(int dtype) {
  switch (dtype) {
  case PJRTX_MLX_METAL_DTYPE_PRED:
  case PJRTX_MLX_METAL_DTYPE_BF16:
  case PJRTX_MLX_METAL_DTYPE_F32:
  case PJRTX_MLX_METAL_DTYPE_F16:
  case PJRTX_MLX_METAL_DTYPE_S32:
  case PJRTX_MLX_METAL_DTYPE_U32:
  case PJRTX_MLX_METAL_DTYPE_U64:
    return true;
  default:
    return false;
  }
}

const char *dense_binary_type_name(int dtype) {
  switch (dtype) {
  case PJRTX_MLX_METAL_DTYPE_F32:
    return "float";
  case PJRTX_MLX_METAL_DTYPE_F16:
    return "half";
  default:
    return nullptr;
  }
}

const char *dense_binary_kernel_name(int op, int dtype) {
  switch (op) {
  case PJRTX_MLX_METAL_U8_BINARY_ADD:
    return dtype == PJRTX_MLX_METAL_DTYPE_BF16
               ? "pjrtx_metalcpp_dense_binary_bfloat16_add"
           : dtype == PJRTX_MLX_METAL_DTYPE_F16
               ? "pjrtx_metalcpp_dense_binary_half_add"
               : "pjrtx_metalcpp_dense_binary_float_add";
  case PJRTX_MLX_METAL_U8_BINARY_SUBTRACT:
    return dtype == PJRTX_MLX_METAL_DTYPE_BF16
               ? "pjrtx_metalcpp_dense_binary_bfloat16_sub"
           : dtype == PJRTX_MLX_METAL_DTYPE_F16
               ? "pjrtx_metalcpp_dense_binary_half_sub"
               : "pjrtx_metalcpp_dense_binary_float_sub";
  case PJRTX_MLX_METAL_U8_BINARY_MULTIPLY:
    return dtype == PJRTX_MLX_METAL_DTYPE_BF16
               ? "pjrtx_metalcpp_dense_binary_bfloat16_mul"
           : dtype == PJRTX_MLX_METAL_DTYPE_F16
               ? "pjrtx_metalcpp_dense_binary_half_mul"
               : "pjrtx_metalcpp_dense_binary_float_mul";
  case PJRTX_MLX_METAL_U8_BINARY_DIVIDE:
    return dtype == PJRTX_MLX_METAL_DTYPE_BF16
               ? "pjrtx_metalcpp_dense_binary_bfloat16_div"
           : dtype == PJRTX_MLX_METAL_DTYPE_F16
               ? "pjrtx_metalcpp_dense_binary_half_div"
               : "pjrtx_metalcpp_dense_binary_float_div";
  case PJRTX_MLX_METAL_BINARY_MAXIMUM:
    return dtype == PJRTX_MLX_METAL_DTYPE_BF16
               ? "pjrtx_metalcpp_dense_binary_bfloat16_max"
           : dtype == PJRTX_MLX_METAL_DTYPE_F16
               ? "pjrtx_metalcpp_dense_binary_half_max"
               : "pjrtx_metalcpp_dense_binary_float_max";
  case PJRTX_MLX_METAL_BINARY_MINIMUM:
    return dtype == PJRTX_MLX_METAL_DTYPE_BF16
               ? "pjrtx_metalcpp_dense_binary_bfloat16_min"
           : dtype == PJRTX_MLX_METAL_DTYPE_F16
               ? "pjrtx_metalcpp_dense_binary_half_min"
               : "pjrtx_metalcpp_dense_binary_float_min";
  default:
    return nullptr;
  }
}

const char *dense_binary_expression(int op) {
  switch (op) {
  case PJRTX_MLX_METAL_U8_BINARY_ADD:
    return "lhs[elem] + rhs[elem]";
  case PJRTX_MLX_METAL_U8_BINARY_SUBTRACT:
    return "lhs[elem] - rhs[elem]";
  case PJRTX_MLX_METAL_U8_BINARY_MULTIPLY:
    return "lhs[elem] * rhs[elem]";
  case PJRTX_MLX_METAL_U8_BINARY_DIVIDE:
    return "lhs[elem] / rhs[elem]";
  case PJRTX_MLX_METAL_BINARY_MAXIMUM:
    return "max(lhs[elem], rhs[elem])";
  case PJRTX_MLX_METAL_BINARY_MINIMUM:
    return "min(lhs[elem], rhs[elem])";
  default:
    return nullptr;
  }
}

std::optional<std::string> dense_binary_source(int op, int dtype) {
  const char *type_name = dense_binary_type_name(dtype);
  const char *kernel_name = dense_binary_kernel_name(op, dtype);
  const char *expression = dense_binary_expression(op);
  if (type_name == nullptr || kernel_name == nullptr || expression == nullptr) {
    return std::nullopt;
  }
  std::string source = "#include <metal_stdlib>\nusing namespace metal;\n";
  source += "kernel void ";
  source += kernel_name;
  source += "(const device ";
  source += type_name;
  source += "* lhs [[buffer(0)]], const device ";
  source += type_name;
  source += "* rhs [[buffer(1)]], device ";
  source += type_name;
  source += "* out [[buffer(2)]], constant uint& count [[buffer(3)]], uint "
            "elem [[thread_position_in_grid]]) {\n";
  source += "  if (elem >= count) return;\n";
  source += "  out[elem] = ";
  source += expression;
  source += ";\n}\n";
  return source;
}

bool dense_array_ready(const mlx::core::array &array, uint64_t elems) {
  return array.offset() == 0 && array.flags().row_contiguous &&
         array.data_size() >= elems;
}

struct DenseInputRequest {
  PjrtxMlxMetalBuffer *buffer;
  uint64_t elems;
};

bool ensure_dense_inputs_ready(const std::vector<DenseInputRequest> &inputs,
                               int device_ordinal) {
  if (inputs.empty()) {
    return true;
  }
  try {
    const mlx::core::Device device(mlx::core::Device::gpu, device_ordinal);
    if (!mlx::core::is_available(device)) {
      return false;
    }
    bool needs_sync = false;
    for (const auto &input : inputs) {
      if (input.buffer == nullptr || input.buffer->array == nullptr) {
        return false;
      }
      if (dense_array_ready(*input.buffer->array, input.elems)) {
        continue;
      }
      input.buffer->array->eval();
      needs_sync = true;
    }
    if (needs_sync) {
      mlx::core::synchronize(mlx::core::default_stream(device));
    }
    for (const auto &input : inputs) {
      input.buffer->array->wait();
      if (!dense_array_ready(*input.buffer->array, input.elems)) {
        return false;
      }
    }
  } catch (...) {
    return false;
  }
  return true;
}

std::optional<PjrtxMetalCppOwnedTensorSpec>
copy_tensor_spec(const PjrtxMetalCppTensorSpec &spec) {
  if (!supported_tensor_spec_dtype(spec.dtype) ||
      (spec.rank > 0 && spec.dims == nullptr)) {
    return std::nullopt;
  }
  PjrtxMetalCppOwnedTensorSpec owned;
  owned.dtype = spec.dtype;
  if (spec.rank > 0) {
    owned.dims.assign(spec.dims, spec.dims + spec.rank);
  }
  if (byte_size_for_shape(owned.dtype, owned.dims) == 0) {
    return std::nullopt;
  }
  return owned;
}

bool tensor_spec_matches_buffer(const PjrtxMetalCppOwnedTensorSpec &spec,
                                PjrtxMlxMetalBuffer *buffer,
                                uint64_t element_count, int device_ordinal) {
  if (buffer == nullptr || buffer->array == nullptr ||
      buffer->dtype != spec.dtype || buffer->dims != spec.dims ||
      buffer->device_ordinal != device_ordinal) {
    return false;
  }
  const uint64_t expected_elements = element_count_for_shape(spec.dims);
  if (expected_elements != element_count && expected_elements != 1) {
    return false;
  }
  return ensure_dense_inputs_ready({DenseInputRequest{buffer, expected_elements}},
                                   device_ordinal);
}

std::string dims_string(const std::vector<int64_t> &dims) {
  std::ostringstream out;
  out << "[";
  for (size_t i = 0; i < dims.size(); ++i) {
    if (i != 0) {
      out << ",";
    }
    out << dims[i];
  }
  out << "]";
  return out.str();
}

std::string tensor_mismatch_detail(const PjrtxMetalCppOwnedTensorSpec &spec,
                                   PjrtxMlxMetalBuffer *buffer,
                                   uint64_t element_count,
                                   int device_ordinal) {
  std::ostringstream out;
  out << " expected_dtype=" << spec.dtype
      << " expected_dims=" << dims_string(spec.dims)
      << " expected_elements=" << element_count
      << " expected_device=" << device_ordinal;
  if (buffer == nullptr) {
    out << " actual=null";
    return out.str();
  }
  out << " actual_dtype=" << buffer->dtype
      << " actual_dims=" << dims_string(buffer->dims)
      << " actual_device=" << buffer->device_ordinal;
  if (buffer->array == nullptr) {
    out << " actual_array=null";
    return out.str();
  }
  out << " actual_elements=" << element_count_for_shape(buffer->dims)
      << " dense_ready="
      << (dense_array_ready(*buffer->array, element_count_for_shape(spec.dims))
              ? "true"
              : "false");
  return out.str();
}


bool metalcpp_fusion_sync_enabled() {
  const char *value = std::getenv("PJRTX_METALCPP_FUSION_SYNC");
  if (value == nullptr || value[0] == '\0') {
    return false;
  }
  return std::strcmp(value, "0") != 0 && std::strcmp(value, "false") != 0 &&
         std::strcmp(value, "FALSE") != 0;
}

bool executable_step_profile_enabled() {
  const char *value = std::getenv("PJRTX_METALCPP_EXECUTABLE_STEP_PROFILE");
  return value != nullptr && value[0] != '\0';
}

bool executable_step_sync_enabled() {
  const char *value = std::getenv("PJRTX_METALCPP_EXECUTABLE_STEP_SYNC");
  return value != nullptr && value[0] != '\0';
}

using ProfileClock = std::chrono::steady_clock;

uint64_t elapsed_us(ProfileClock::time_point start) {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
          ProfileClock::now() - start)
          .count());
}

struct GeneratedKernelCreateRequest {
  int device_ordinal;
  const char *kernel_name;
  uint64_t kernel_name_size;
  const char *source;
  uint64_t source_size;
  const PjrtxMetalCppTensorSpec *inputs;
  uint64_t input_count;
  const PjrtxMetalCppTensorSpec *outputs;
  uint64_t output_count;
  uint64_t element_count;
  uint32_t threads_per_threadgroup;
  bool allow_mismatched_input_elements = false;
  bool allow_mismatched_output_elements = false;
};

struct GeneratedKernelDispatchRequest {
  PjrtxMetalCppFusionKernel *kernel;
  std::vector<std::optional<mlx::core::array>> *values;
  const std::vector<uint64_t> *inputs;
  const std::vector<uint64_t> *outputs;
};

struct ExecutableStepCreateRequest {
  int device_ordinal;
  const PjrtxMetalCppExecutableStep *step;
  const std::vector<PjrtxMetalCppOwnedTensorSpec> *values;
};

constexpr uint64_t kLegacyExecutableProgramCreateArgsSize =
    offsetof(PjrtxMetalCppExecutableProgramCreateArgs, values);

bool initialize_generated_kernel(PjrtxMetalCppFusionKernel *kernel,
                                 const GeneratedKernelCreateRequest &request) {
  if (kernel == nullptr || request.kernel_name == nullptr ||
      request.kernel_name_size == 0 || request.source == nullptr ||
      request.source_size == 0 ||
      (request.input_count != 0 && request.inputs == nullptr) ||
      request.outputs == nullptr ||
      request.output_count == 0 || request.element_count == 0 ||
      request.element_count >
          static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    return fail_with("invalid generated kernel create request");
  }

  const mlx::core::Device device(mlx::core::Device::gpu,
                                 request.device_ordinal);
  if (!mlx::core::is_available(device)) {
    return fail_with("metal device unavailable");
  }

  kernel->device_ordinal = request.device_ordinal;
  kernel->kernel_name.assign(request.kernel_name,
                             request.kernel_name + request.kernel_name_size);
  kernel->source.assign(request.source, request.source + request.source_size);
  kernel->pipeline = nullptr;
  kernel->inputs.clear();
  kernel->outputs.clear();
  kernel->element_count = request.element_count;
  kernel->threads_per_threadgroup = request.threads_per_threadgroup == 0
                                        ? 256
                                        : request.threads_per_threadgroup;
  kernel->allow_mismatched_input_elements =
      request.allow_mismatched_input_elements;

  kernel->inputs.reserve(static_cast<size_t>(request.input_count));
  for (uint64_t i = 0; i < request.input_count; ++i) {
    auto spec = copy_tensor_spec(request.inputs[i]);
    if (!spec.has_value()) {
      return fail_with("invalid generated kernel input tensor spec");
    }
    const uint64_t input_elements = element_count_for_shape(spec->dims);
    if (input_elements == 0 ||
        (!request.allow_mismatched_input_elements &&
         input_elements != request.element_count && input_elements != 1)) {
      return fail_with("generated kernel input element count mismatch");
    }
    kernel->inputs.push_back(std::move(*spec));
  }

  kernel->outputs.reserve(static_cast<size_t>(request.output_count));
  for (uint64_t i = 0; i < request.output_count; ++i) {
    auto spec = copy_tensor_spec(request.outputs[i]);
    if (!spec.has_value()) {
      return fail_with("invalid generated kernel output tensor spec");
    }
    const uint64_t output_elements = element_count_for_shape(spec->dims);
    if (output_elements == 0 ||
        (!request.allow_mismatched_output_elements &&
         output_elements != request.element_count)) {
      return fail_with("generated kernel output element count mismatch");
    }
    kernel->outputs.push_back(std::move(*spec));
  }

  try {
    auto stream = mlx::core::default_stream(device);
    auto &metal_device = mlx::core::metal::device(stream.device);
    std::lock_guard<std::mutex> lock(g_kernel_compile_mutex);
    auto lib = metal_device.get_library(kernel->kernel_name,
                                        [&]() { return kernel->source; });
    kernel->pipeline = metal_device.get_kernel(kernel->kernel_name, lib);
    if (kernel->pipeline == nullptr) {
      return fail_with("metal pipeline creation returned null");
    }
  } catch (const std::exception &err) {
    return fail_with(std::string("metal pipeline creation failed: ") +
                     err.what());
  } catch (...) {
    return fail_with("metal pipeline creation failed: unknown exception");
  }

  return true;
}

bool dispatch_generated_kernel(const GeneratedKernelDispatchRequest &request) {
  if (request.kernel == nullptr || request.values == nullptr ||
      request.inputs == nullptr || request.outputs == nullptr ||
      request.inputs->size() != request.kernel->inputs.size() ||
      request.outputs->size() != request.kernel->outputs.size() ||
      request.kernel->pipeline == nullptr) {
    return fail_with("invalid generated kernel dispatch request");
  }

  auto stream = mlx::core::default_stream(mlx::core::Device(
      mlx::core::Device::gpu, request.kernel->device_ordinal));
  auto &encoder = mlx::core::metal::get_command_encoder(stream);
  encoder.set_compute_pipeline_state(request.kernel->pipeline);
  for (size_t i = 0; i < request.inputs->size(); ++i) {
    const uint64_t value_index = (*request.inputs)[i];
    if (value_index >= request.values->size() ||
        !(*request.values)[value_index].has_value()) {
      return fail_with("generated kernel input value index out of range");
    }
    encoder.set_input_array((*request.values)[value_index].value(),
                            static_cast<int>(i));
  }
  for (size_t i = 0; i < request.outputs->size(); ++i) {
    const uint64_t value_index = (*request.outputs)[i];
    if (value_index >= request.values->size() ||
        !(*request.values)[value_index].has_value()) {
      return fail_with("generated kernel output value index out of range");
    }
    encoder.set_output_array((*request.values)[value_index].value(),
                             static_cast<int>(request.inputs->size() + i));
  }
  const uint32_t count = static_cast<uint32_t>(request.kernel->element_count);
  encoder.set_bytes(count, static_cast<int>(request.inputs->size() +
                                            request.outputs->size()));
  const uint64_t threads =
      std::min<uint64_t>(request.kernel->threads_per_threadgroup == 0
                             ? 256
                             : request.kernel->threads_per_threadgroup,
                         request.kernel->element_count);
  encoder.dispatch_threads(MTL::Size(request.kernel->element_count, 1, 1),
                           MTL::Size(threads, 1, 1));
  return true;
}

bool copy_value_indices(const uint64_t *raw_indices, uint64_t count,
                        uint64_t value_count,
                        std::vector<uint64_t> *out_indices) {
  if (out_indices == nullptr || (count > 0 && raw_indices == nullptr)) {
    return false;
  }
  out_indices->clear();
  out_indices->reserve(static_cast<size_t>(count));
  for (uint64_t i = 0; i < count; ++i) {
    if (raw_indices[i] >= value_count) {
      return false;
    }
    out_indices->push_back(raw_indices[i]);
  }
  return true;
}

bool initialize_executable_step(PjrtxMetalCppExecutableOwnedStep *owned_step,
                                const ExecutableStepCreateRequest &request) {
  if (owned_step == nullptr || request.step == nullptr ||
      request.values == nullptr ||
      (request.step->input_count > 0 && request.step->inputs == nullptr) ||
      (request.step->output_count > 0 && request.step->outputs == nullptr)) {
    return false;
  }
  if (request.step->source_size == 0) {
    if (request.step->input_count != 1 || request.step->output_count != 1) {
      return fail_with("invalid executable alias step");
    }
    owned_step->alias = true;
    return true;
  }
  auto *kernel = &owned_step->kernel;

  std::vector<PjrtxMetalCppTensorSpec> inputs;
  inputs.reserve(static_cast<size_t>(request.step->input_count));
  for (uint64_t i = 0; i < request.step->input_count; ++i) {
    const uint64_t value_index = request.step->inputs[i];
    if (value_index >= request.values->size()) {
      return false;
    }
    const auto &value = (*request.values)[value_index];
    inputs.push_back({value.dtype,
                      value.dims.empty() ? nullptr : value.dims.data(),
                      value.dims.size()});
  }

  std::vector<PjrtxMetalCppTensorSpec> outputs;
  outputs.reserve(static_cast<size_t>(request.step->output_count));
  for (uint64_t i = 0; i < request.step->output_count; ++i) {
    const uint64_t value_index = request.step->outputs[i];
    if (value_index >= request.values->size()) {
      return false;
    }
    const auto &value = (*request.values)[value_index];
    outputs.push_back({value.dtype,
                       value.dims.empty() ? nullptr : value.dims.data(),
                       value.dims.size()});
  }

  return initialize_generated_kernel(kernel,
                                     {
                                         request.device_ordinal,
                                         request.step->kernel_name,
                                         request.step->kernel_name_size,
                                         request.step->source,
                                         request.step->source_size,
                                         inputs.data(),
                                         inputs.size(),
                                         outputs.data(),
                                         outputs.size(),
                                         request.step->element_count,
                                         request.step->threads_per_threadgroup,
                                         true,
                                         true,
                                     });
}

bool executable_value_is_output(const PjrtxMetalCppExecutableProgram *program,
                                uint64_t value_index) {
  for (const auto output_value : program->output_values) {
    if (output_value == value_index) {
      return true;
    }
  }
  return false;
}

bool release_executable_values(
    const PjrtxMetalCppExecutableProgram *program,
    const PjrtxMetalCppExecutableOwnedStep &step,
    std::vector<std::optional<mlx::core::array>> *values) {
  if (program == nullptr || values == nullptr) {
    return fail_with("invalid executable program release request");
  }
  for (const auto value_index : step.release_values) {
    if (value_index >= values->size()) {
      return fail_with("executable program release value index out of range");
    }
    if (executable_value_is_output(program, value_index)) {
      return fail_with("executable program attempted to release output value");
    }
    (*values)[value_index].reset();
  }
  return true;
}

std::optional<mlx::core::array>
allocate_executable_value(const PjrtxMetalCppOwnedTensorSpec &spec) {
  const auto dtype = mlx_dtype_from_code(spec.dtype);
  if (!dtype.has_value()) {
    fail_with("executable program value dtype unsupported");
    return std::nullopt;
  }
  mlx::core::array value(mlx_shape(spec.dims), *dtype, nullptr, {});
  value.set_data(mlx::core::allocator::malloc(value.nbytes()));
  return value;
}

int execute_generated_kernel(PjrtxMetalCppFusionKernel *kernel,
                             PjrtxMlxMetalBuffer *const *inputs,
                             uint64_t input_count,
                             PjrtxMlxMetalBuffer ***out_outputs,
                             uint64_t *out_output_count) {
  if (kernel == nullptr || inputs == nullptr || out_outputs == nullptr ||
      out_output_count == nullptr || input_count != kernel->inputs.size()) {
    fail_with("invalid fusion kernel execute request");
    return 0;
  }
  *out_outputs = nullptr;
  *out_output_count = 0;

  try {
    std::vector<mlx::core::array> input_arrays;
    input_arrays.reserve(kernel->inputs.size());
    for (size_t i = 0; i < kernel->inputs.size(); ++i) {
      const uint64_t expected_elements =
          kernel->allow_mismatched_input_elements
              ? element_count_for_shape(kernel->inputs[i].dims)
              : kernel->element_count;
      if (!tensor_spec_matches_buffer(kernel->inputs[i], inputs[i],
                                      expected_elements,
                                      kernel->device_ordinal)) {
        fail_with(std::string("fusion kernel input tensor mismatch index=") +
                  std::to_string(i) +
                  tensor_mismatch_detail(kernel->inputs[i], inputs[i],
                                         expected_elements,
                                         kernel->device_ordinal));
        return 0;
      }
      input_arrays.push_back(*inputs[i]->array);
    }

    const mlx::core::Device device(mlx::core::Device::gpu,
                                   kernel->device_ordinal);
    if (!mlx::core::is_available(device)) {
      fail_with("fusion kernel metal device unavailable");
      return 0;
    }

    std::vector<mlx::core::array> output_arrays;
    output_arrays.reserve(kernel->outputs.size());
    for (const auto &spec : kernel->outputs) {
      const auto dtype = mlx_dtype_from_code(spec.dtype);
      if (!dtype.has_value()) {
        fail_with("fusion kernel output dtype unsupported");
        return 0;
      }
      const auto shape = mlx_shape(spec.dims);
      mlx::core::array out(shape, *dtype, nullptr, {});
      out.set_data(mlx::core::allocator::malloc(out.nbytes()));
      output_arrays.push_back(std::move(out));
    }

    auto stream = mlx::core::default_stream(device);
    if (kernel->pipeline == nullptr) {
      fail_with("fusion kernel pipeline missing");
      return 0;
    }
    auto &encoder = mlx::core::metal::get_command_encoder(stream);
    encoder.set_compute_pipeline_state(kernel->pipeline);
    for (size_t i = 0; i < input_arrays.size(); ++i) {
      encoder.set_input_array(input_arrays[i], static_cast<int>(i));
    }
    for (size_t i = 0; i < output_arrays.size(); ++i) {
      encoder.set_output_array(output_arrays[i],
                               static_cast<int>(input_arrays.size() + i));
    }
    const uint32_t count = static_cast<uint32_t>(kernel->element_count);
    encoder.set_bytes(
        count, static_cast<int>(input_arrays.size() + output_arrays.size()));
    const uint64_t threads = std::min<uint64_t>(
        kernel->threads_per_threadgroup == 0 ? 256
                                             : kernel->threads_per_threadgroup,
        kernel->element_count);
    encoder.dispatch_threads(MTL::Size(kernel->element_count, 1, 1),
                             MTL::Size(threads, 1, 1));
    const bool sync_after_dispatch = metalcpp_fusion_sync_enabled();
    if (sync_after_dispatch) {
      encoder.synchronize();
    } else {
      encoder.end_encoding();
      encoder.commit();
    }

    auto **raw_outputs = new PjrtxMlxMetalBuffer *[output_arrays.size()]();
    uint64_t initialized = 0;
    try {
      for (size_t i = 0; i < output_arrays.size(); ++i) {
        if (sync_after_dispatch) {
          output_arrays[i].set_status(mlx::core::array::Status::available);
        } else {
          output_arrays[i].set_status(mlx::core::array::Status::evaluated);
        }
        const auto &spec = kernel->outputs[i];
        const uint64_t byte_size = byte_size_for_shape(spec.dtype, spec.dims);
        auto array =
            std::make_unique<mlx::core::array>(std::move(output_arrays[i]));
        raw_outputs[i] =
            new PjrtxMlxMetalBuffer(std::move(array), byte_size, spec.dtype,
                                    spec.dims, kernel->device_ordinal);
        ++initialized;
      }
    } catch (...) {
      for (uint64_t i = 0; i < initialized; ++i) {
        delete raw_outputs[i];
      }
      delete[] raw_outputs;
      throw;
    }

    *out_outputs = raw_outputs;
    *out_output_count = output_arrays.size();
    return 1;
  } catch (const std::exception &err) {
    fail_with(std::string("fusion kernel execute failed: ") + err.what());
    return 0;
  } catch (...) {
    fail_with("fusion kernel execute failed: unknown exception");
    return 0;
  }
}

bool initialize_legacy_executable_program(
    PjrtxMetalCppExecutableProgram *program,
    const PjrtxMetalCppExecutableProgramCreateArgs *args) {
  if (program == nullptr || args == nullptr) {
    return fail_with("invalid legacy executable program create request");
  }
  program->device_ordinal = args->device_ordinal;

  auto kernel = std::make_unique<PjrtxMetalCppFusionKernel>();
  if (!initialize_generated_kernel(kernel.get(),
                                   {
                                       args->device_ordinal,
                                       args->entry_kernel_name,
                                       args->entry_kernel_name_size,
                                       args->source,
                                       args->source_size,
                                       args->inputs,
                                       args->input_count,
                                       args->outputs,
                                       args->output_count,
                                       args->element_count,
                                       args->threads_per_threadgroup,
                                   })) {
    return false;
  }

  program->values.clear();
  program->input_values.clear();
  program->output_values.clear();
  program->steps.clear();
  program->values.reserve(
      static_cast<size_t>(args->input_count + args->output_count));
  for (const auto &spec : kernel->inputs) {
    program->values.push_back(spec);
  }
  for (const auto &spec : kernel->outputs) {
    program->values.push_back(spec);
  }
  for (uint64_t i = 0; i < args->input_count; ++i) {
    program->input_values.push_back(i);
  }
  for (uint64_t i = 0; i < args->output_count; ++i) {
    program->output_values.push_back(args->input_count + i);
  }
    PjrtxMetalCppExecutableOwnedStep step;
    step.kernel = std::move(*kernel);
    step.inputs = program->input_values;
    step.outputs = program->output_values;
    step.alias = false;
    program->steps.push_back(std::move(step));
  return true;
}

bool initialize_executable_program(
    PjrtxMetalCppExecutableProgram *program,
    const PjrtxMetalCppExecutableProgramCreateArgs *args) {
  if (program == nullptr || args == nullptr ||
      args->struct_size < kLegacyExecutableProgramCreateArgsSize) {
    return fail_with("invalid executable program create request");
  }
  if (args->struct_size < sizeof(PjrtxMetalCppExecutableProgramCreateArgs) ||
      args->values == nullptr || args->value_count == 0 ||
      args->steps == nullptr || args->step_count == 0) {
    return initialize_legacy_executable_program(program, args);
  }

  const mlx::core::Device device(mlx::core::Device::gpu, args->device_ordinal);
  if (!mlx::core::is_available(device) ||
      (args->input_value_count > 0 && args->input_values == nullptr) ||
      (args->output_value_count > 0 && args->output_values == nullptr)) {
    return fail_with("invalid executable program topology or value indices");
  }

  program->device_ordinal = args->device_ordinal;
  program->values.clear();
  program->input_values.clear();
  program->output_values.clear();
  program->steps.clear();

  program->values.reserve(static_cast<size_t>(args->value_count));
  for (uint64_t i = 0; i < args->value_count; ++i) {
    auto spec = copy_tensor_spec(args->values[i].spec);
    if (!spec.has_value()) {
      return fail_with("invalid executable program value tensor spec");
    }
    program->values.push_back(std::move(*spec));
  }

  if (!copy_value_indices(args->input_values, args->input_value_count,
                          args->value_count, &program->input_values) ||
      !copy_value_indices(args->output_values, args->output_value_count,
                          args->value_count, &program->output_values)) {
    return fail_with("invalid executable program input/output value index");
  }

  program->steps.reserve(static_cast<size_t>(args->step_count));
  for (uint64_t i = 0; i < args->step_count; ++i) {
    PjrtxMetalCppExecutableOwnedStep step;
    if (!initialize_executable_step(
            &step,
            {args->device_ordinal, &args->steps[i], &program->values})) {
      if (g_last_error.empty()) {
        return fail_with("executable program step initialization failed");
      }
      return false;
    }
    if (!copy_value_indices(args->steps[i].inputs, args->steps[i].input_count,
                            args->value_count, &step.inputs) ||
        !copy_value_indices(args->steps[i].outputs, args->steps[i].output_count,
                            args->value_count, &step.outputs) ||
        !copy_value_indices(args->steps[i].release_values,
                            args->steps[i].release_value_count,
                            args->value_count, &step.release_values)) {
      return fail_with("invalid executable program step value index");
    }
    if (step.alias && (step.inputs.size() != 1 || step.outputs.size() != 1)) {
      return fail_with("invalid executable program alias value index");
    }
    step.in_place_input = args->steps[i].in_place_input_plus_one == 0
                              ? -1
                              : static_cast<int64_t>(
                                    args->steps[i].in_place_input_plus_one - 1);
    if (step.in_place_input >= 0 &&
        (step.alias || step.outputs.size() != 1 ||
         static_cast<uint64_t>(step.in_place_input) >= step.inputs.size())) {
      return fail_with("invalid executable program in-place input index");
    }
    program->steps.push_back(std::move(step));
  }
  return true;
}

int execute_executable_program(PjrtxMetalCppExecutableProgram *program,
                               PjrtxMlxMetalBuffer *const *inputs,
                               uint64_t input_count,
                               PjrtxMlxMetalBuffer ***out_outputs,
                               uint64_t *out_output_count) {
  if (program == nullptr || inputs == nullptr || out_outputs == nullptr ||
      out_output_count == nullptr ||
      input_count != program->input_values.size()) {
    fail_with("invalid executable program execute request");
    return 0;
  }
  *out_outputs = nullptr;
  *out_output_count = 0;

  try {
    const mlx::core::Device device(mlx::core::Device::gpu,
                                   program->device_ordinal);
    if (!mlx::core::is_available(device)) {
      fail_with("executable program metal device unavailable");
      return 0;
    }

    std::vector<std::optional<mlx::core::array>> values(program->values.size());

    for (size_t i = 0; i < program->input_values.size(); ++i) {
      const uint64_t value_index = program->input_values[i];
      const auto &spec = program->values[value_index];
      const uint64_t expected_elements = element_count_for_shape(spec.dims);
      if (inputs[i] == nullptr || inputs[i]->array == nullptr ||
          inputs[i]->dtype != spec.dtype || inputs[i]->dims != spec.dims ||
          inputs[i]->device_ordinal != program->device_ordinal) {
        fail_with(std::string("executable program input tensor mismatch input=") +
                  std::to_string(i) + " value=" + std::to_string(value_index) +
                  tensor_mismatch_detail(spec, inputs[i], expected_elements,
                                         program->device_ordinal));
        return 0;
      }
    }

    std::vector<DenseInputRequest> dense_inputs;
    dense_inputs.reserve(program->input_values.size());
    for (size_t i = 0; i < program->input_values.size(); ++i) {
      const auto &spec = program->values[program->input_values[i]];
      const uint64_t expected_elements = element_count_for_shape(spec.dims);
      if (!dense_array_ready(*inputs[i]->array, expected_elements)) {
        dense_inputs.push_back({inputs[i], expected_elements});
      }
    }
    if (!ensure_dense_inputs_ready(dense_inputs, program->device_ordinal)) {
      fail_with("executable program input materialization failed");
      return 0;
    }

    for (size_t i = 0; i < program->input_values.size(); ++i) {
      const uint64_t value_index = program->input_values[i];
      values[value_index].emplace(*inputs[i]->array);
    }

    const bool profile_steps = executable_step_profile_enabled();
    const bool sync_steps = executable_step_sync_enabled();
    for (size_t step_index = 0; step_index < program->steps.size();
         ++step_index) {
      auto &step = program->steps[step_index];
      if (step.alias) {
        const auto step_start = ProfileClock::now();
        if (step.inputs[0] >= values.size() || step.outputs[0] >= values.size() ||
            !values[step.inputs[0]].has_value()) {
          fail_with("executable program alias input missing");
          return 0;
        }
        values[step.outputs[0]] = values[step.inputs[0]].value();
        if (!release_executable_values(program, step, &values)) {
          return 0;
        }
        if (profile_steps) {
          std::fprintf(stderr,
                       "pjrtx_profile event=metal_graph_step_execute "
                       "step=%zu alias=1 kernel=\"\" inputs=%zu outputs=%zu "
                       "release_values=%zu element_count=0 "
                       "threads_per_threadgroup=0 sync=0 elapsed_us=%llu\n",
                       step_index, step.inputs.size(), step.outputs.size(),
                       step.release_values.size(),
                       static_cast<unsigned long long>(elapsed_us(step_start)));
        }
        continue;
      }
      if (step.in_place_input >= 0) {
        const uint64_t input_position = static_cast<uint64_t>(step.in_place_input);
        const uint64_t input_index = step.inputs[input_position];
        const uint64_t output_index = step.outputs[0];
        if (input_index >= values.size() || output_index >= values.size() ||
            !values[input_index].has_value()) {
          fail_with("executable program in-place input missing");
          return 0;
        }
        values[output_index] = values[input_index].value();
      } else {
        for (const auto output_index : step.outputs) {
          if (output_index >= values.size() ||
              output_index >= program->values.size()) {
            fail_with("executable program step output value index out of range");
            return 0;
          }
          auto allocated =
              allocate_executable_value(program->values[output_index]);
          if (!allocated.has_value()) {
            return 0;
          }
          values[output_index] = std::move(allocated);
        }
      }
      const auto step_start = ProfileClock::now();
      if (!dispatch_generated_kernel({&step.kernel, &values, &step.inputs,
                                      &step.outputs})) {
        if (g_last_error.empty()) {
          fail_with("executable program step dispatch failed");
        }
        return 0;
      }
      if (sync_steps) {
        auto stream = mlx::core::default_stream(device);
        auto &encoder = mlx::core::metal::get_command_encoder(stream);
        encoder.synchronize();
      }
      if (!release_executable_values(program, step, &values)) {
        return 0;
      }
      if (profile_steps) {
        std::fprintf(stderr,
                     "pjrtx_profile event=metal_graph_step_execute "
                     "step=%zu alias=0 kernel=\"%s\" inputs=%zu outputs=%zu "
                     "release_values=%zu element_count=%llu "
                     "threads_per_threadgroup=%u in_place_input=%lld sync=%d "
                     "elapsed_us=%llu\n",
                     step_index, step.kernel.kernel_name.c_str(),
                     step.inputs.size(), step.outputs.size(),
                     step.release_values.size(),
                     static_cast<unsigned long long>(step.kernel.element_count),
                     step.kernel.threads_per_threadgroup,
                     static_cast<long long>(step.in_place_input),
                     sync_steps ? 1 : 0,
                     static_cast<unsigned long long>(elapsed_us(step_start)));
      }
    }

    auto stream = mlx::core::default_stream(device);
    const bool sync_after_dispatch = metalcpp_fusion_sync_enabled();
    auto &encoder = mlx::core::metal::get_command_encoder(stream);
    if (sync_after_dispatch) {
      encoder.synchronize();
    } else {
      encoder.end_encoding();
      encoder.commit();
    }

    auto **raw_outputs =
        new PjrtxMlxMetalBuffer *[program->output_values.size()]();
    uint64_t initialized = 0;
    try {
      for (size_t i = 0; i < program->output_values.size(); ++i) {
        const uint64_t value_index = program->output_values[i];
        if (value_index >= values.size() || !values[value_index].has_value()) {
          fail_with("executable program output value index out of range");
          return 0;
        }
        if (sync_after_dispatch) {
          values[value_index].value().set_status(
              mlx::core::array::Status::available);
        } else {
          values[value_index].value().set_status(
              mlx::core::array::Status::evaluated);
        }
        const auto &spec = program->values[value_index];
        const uint64_t byte_size = byte_size_for_shape(spec.dtype, spec.dims);
        auto array =
            std::make_unique<mlx::core::array>(
                std::move(values[value_index].value()));
        raw_outputs[i] =
            new PjrtxMlxMetalBuffer(std::move(array), byte_size, spec.dtype,
                                    spec.dims, program->device_ordinal);
        ++initialized;
      }
    } catch (...) {
      for (uint64_t i = 0; i < initialized; ++i) {
        delete raw_outputs[i];
      }
      delete[] raw_outputs;
      throw;
    }

    *out_outputs = raw_outputs;
    *out_output_count = program->output_values.size();
    return 1;
  } catch (const std::exception &err) {
    fail_with(std::string("executable program execute failed: ") + err.what());
    return 0;
  } catch (...) {
    fail_with("executable program execute failed: unknown exception");
    return 0;
  }
}

} // namespace

extern "C" {

PjrtxMlxMetalBuffer *pjrtx_metalcpp_buffer_dense_binary_out(
    PjrtxMlxMetalBuffer *lhs, PjrtxMlxMetalBuffer *rhs, int op,
    const int64_t *output_dims, uint64_t output_rank) {
  if (lhs == nullptr || rhs == nullptr || lhs->array == nullptr ||
      rhs->array == nullptr || lhs->dtype != rhs->dtype ||
      lhs->device_ordinal != rhs->device_ordinal ||
      !supported_dense_binary_dtype(lhs->dtype) ||
      (output_rank > 0 && output_dims == nullptr)) {
    return nullptr;
  }

  std::vector<int64_t> out_dims;
  if (output_rank > 0) {
    out_dims.assign(output_dims, output_dims + output_rank);
  }
  const uint64_t output_elements = element_count_for_shape(out_dims);
  const uint64_t byte_size = byte_size_for_shape(lhs->dtype, out_dims);
  if (output_elements == 0 || byte_size == 0 ||
      output_elements >
          static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) {
    return nullptr;
  }

  const auto dtype = mlx_dtype_from_code(lhs->dtype);
  const char *kernel_name = dense_binary_kernel_name(op, lhs->dtype);
  auto source = dense_binary_source(op, lhs->dtype);
  if (!dtype.has_value() || kernel_name == nullptr || !source.has_value()) {
    return nullptr;
  }

  const mlx::core::Device device(mlx::core::Device::gpu, lhs->device_ordinal);
  if (!mlx::core::is_available(device)) {
    return nullptr;
  }

  try {
    mlx::core::array lhs_array = *lhs->array;
    mlx::core::array rhs_array = *rhs->array;
    const auto output_shape = mlx_shape(out_dims);
    if (lhs_array.shape() != output_shape ||
        rhs_array.shape() != output_shape ||
        !dense_array_ready(lhs_array, output_elements) ||
        !dense_array_ready(rhs_array, output_elements)) {
      return nullptr;
    }

    mlx::core::array out(output_shape, *dtype, nullptr, {});
    out.set_data(mlx::core::allocator::malloc(out.nbytes()));

    auto stream = mlx::core::default_stream(device);
    auto &metal_device = mlx::core::metal::device(stream.device);
    auto lib = metal_device.get_library(kernel_name, [&]() { return *source; });
    auto kernel = metal_device.get_kernel(kernel_name, lib);
    auto &encoder = mlx::core::metal::get_command_encoder(stream);
    encoder.set_compute_pipeline_state(kernel);
    encoder.set_input_array(lhs_array, 0);
    encoder.set_input_array(rhs_array, 1);
    encoder.set_output_array(out, 2);
    const uint32_t count = static_cast<uint32_t>(output_elements);
    encoder.set_bytes(count, 3);
    encoder.dispatch_threads(
        MTL::Size(output_elements, 1, 1),
        MTL::Size(std::min<uint64_t>(256, output_elements), 1, 1));
    encoder.synchronize();
    out.set_status(mlx::core::array::Status::available);

    auto array = std::make_unique<mlx::core::array>(std::move(out));
    return new PjrtxMlxMetalBuffer(std::move(array), byte_size, lhs->dtype,
                                   std::move(out_dims), lhs->device_ordinal);
  } catch (const std::exception &) {
    return nullptr;
  } catch (...) {
    return nullptr;
  }
}

PjrtxMetalCppFusionKernel *pjrtx_metalcpp_fusion_kernel_create(
    const PjrtxMetalCppFusionKernelCreateArgs *args) {
  clear_last_error();
  auto kernel = std::make_unique<PjrtxMetalCppFusionKernel>();
  if (args == nullptr ||
      args->struct_size < sizeof(PjrtxMetalCppFusionKernelCreateArgs)) {
    fail_with("invalid fusion kernel create args");
    print_last_error_if_tracing("metalcpp_fusion_kernel_create_failed");
    return nullptr;
  }
  if (!initialize_generated_kernel(kernel.get(),
                                   {
                                       args->device_ordinal,
                                       args->kernel_name,
                                       args->kernel_name_size,
                                       args->source,
                                       args->source_size,
                                       args->inputs,
                                       args->input_count,
                                       args->outputs,
                                       args->output_count,
                                       args->element_count,
                                       args->threads_per_threadgroup,
                                   })) {
    print_last_error_if_tracing("metalcpp_fusion_kernel_create_failed");
    return nullptr;
  }

  return kernel.release();
}

int pjrtx_metalcpp_fusion_kernel_execute(PjrtxMetalCppFusionKernel *kernel,
                                         PjrtxMlxMetalBuffer *const *inputs,
                                         uint64_t input_count,
                                         PjrtxMlxMetalBuffer ***out_outputs,
                                         uint64_t *out_output_count) {
  clear_last_error();
  const int ok = execute_generated_kernel(kernel, inputs, input_count, out_outputs,
                                          out_output_count);
  if (ok == 0) {
    print_last_error_if_tracing("metalcpp_fusion_kernel_execute_failed");
  }
  return ok;
}

void pjrtx_metalcpp_fusion_output_array_destroy(PjrtxMlxMetalBuffer **outputs) {
  delete[] outputs;
}

void pjrtx_metalcpp_fusion_kernel_destroy(PjrtxMetalCppFusionKernel *kernel) {
  delete kernel;
}

PjrtxMetalCppExecutableProgram *pjrtx_metalcpp_executable_program_create(
    const PjrtxMetalCppExecutableProgramCreateArgs *args) {
  clear_last_error();
  auto program = std::make_unique<PjrtxMetalCppExecutableProgram>();
  if (!initialize_executable_program(program.get(), args)) {
    print_last_error_if_tracing("metalcpp_executable_program_create_failed");
    return nullptr;
  }

  return program.release();
}

int pjrtx_metalcpp_executable_program_execute(
    PjrtxMetalCppExecutableProgram *program, PjrtxMlxMetalBuffer *const *inputs,
    uint64_t input_count, PjrtxMlxMetalBuffer ***out_outputs,
    uint64_t *out_output_count) {
  clear_last_error();
  const int ok = execute_executable_program(program, inputs, input_count,
                                            out_outputs, out_output_count);
  if (ok == 0) {
    print_last_error_if_tracing("metalcpp_executable_program_execute_failed");
  }
  return ok;
}

void pjrtx_metalcpp_executable_program_output_array_destroy(
    PjrtxMlxMetalBuffer **outputs) {
  delete[] outputs;
}

void pjrtx_metalcpp_executable_program_destroy(
    PjrtxMetalCppExecutableProgram *program) {
  delete program;
}

const char *pjrtx_metalcpp_last_error(void) {
  return g_last_error.empty() ? nullptr : g_last_error.c_str();
}

} // extern "C"
