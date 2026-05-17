#include "src/backend/mlx_metal_api.h"

#include <cassert>
#include <cmath>

bool near(float lhs, float rhs) { return std::fabs(lhs - rhs) < 0.0001f; }

int main() {
  assert(pjrtx_mlx_metal_version_major() == 0);
  assert(pjrtx_mlx_metal_version_minor() == 32);
  assert(pjrtx_mlx_metal_version_patch() == 0);
  assert(pjrtx_mlx_metal_has_upstream_mlx_metal_api() == 1);

  const int count = pjrtx_mlx_metal_device_count();
  assert(count >= 0);

  PjrtxMlxMetalDeviceInfo devices[8] = {};
  const int copied = pjrtx_mlx_metal_copy_devices(devices, 8);
  assert(copied >= 0);
  assert(copied <= 8);
  if (copied > 0) {
    assert(devices[0].ordinal == 0);
    assert(devices[0].name[0] != '\0');

    const unsigned char input[] = {1, 2, 3, 4, 5, 6};
    PjrtxMlxMetalBuffer* buffer =
        pjrtx_mlx_metal_buffer_from_host(devices[0].ordinal, input,
                                         sizeof(input));
    assert(buffer != nullptr);
    assert(pjrtx_mlx_metal_buffer_size(buffer) == sizeof(input));

    unsigned char output[sizeof(input)] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(buffer, output,
                                               sizeof(output)) == 1);
    for (int i = 0; i < static_cast<int>(sizeof(input)); ++i) {
      assert(output[i] == input[i]);
    }
    assert(pjrtx_mlx_metal_buffer_copy_to_host(buffer, output,
                                               sizeof(output) - 1) == 0);

    PjrtxMlxMetalBuffer* cloned = pjrtx_mlx_metal_buffer_clone(buffer);
    assert(cloned != nullptr);
    assert(pjrtx_mlx_metal_buffer_size(cloned) == sizeof(input));
    unsigned char cloned_output[sizeof(input)] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(cloned, cloned_output,
                                               sizeof(cloned_output)) == 1);
    for (int i = 0; i < static_cast<int>(sizeof(input)); ++i) {
      assert(cloned_output[i] == input[i]);
    }
    pjrtx_mlx_metal_buffer_destroy(cloned);

    const unsigned char rhs[] = {10, 20, 30, 40, 50, 60};
    PjrtxMlxMetalBuffer* rhs_buffer =
        pjrtx_mlx_metal_buffer_from_host(devices[0].ordinal, rhs, sizeof(rhs));
    assert(rhs_buffer != nullptr);
    PjrtxMlxMetalBuffer* sum =
        pjrtx_mlx_metal_buffer_add_u8(buffer, rhs_buffer);
    assert(sum != nullptr);
    unsigned char sum_output[sizeof(input)] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(sum, sum_output,
                                               sizeof(sum_output)) == 1);
    for (int i = 0; i < static_cast<int>(sizeof(input)); ++i) {
      assert(sum_output[i] == static_cast<unsigned char>(input[i] + rhs[i]));
    }
    pjrtx_mlx_metal_buffer_destroy(sum);

    PjrtxMlxMetalBuffer* difference = pjrtx_mlx_metal_buffer_binary_u8(
        rhs_buffer, buffer, PJRTX_MLX_METAL_U8_BINARY_SUBTRACT);
    assert(difference != nullptr);
    unsigned char difference_output[sizeof(input)] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               difference, difference_output, sizeof(difference_output)) == 1);
    for (int i = 0; i < static_cast<int>(sizeof(input)); ++i) {
      assert(difference_output[i] ==
             static_cast<unsigned char>(rhs[i] - input[i]));
    }
    pjrtx_mlx_metal_buffer_destroy(difference);

    PjrtxMlxMetalBuffer* product = pjrtx_mlx_metal_buffer_binary_u8(
        buffer, rhs_buffer, PJRTX_MLX_METAL_U8_BINARY_MULTIPLY);
    assert(product != nullptr);
    unsigned char product_output[sizeof(input)] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(product, product_output,
                                               sizeof(product_output)) == 1);
    for (int i = 0; i < static_cast<int>(sizeof(input)); ++i) {
      assert(product_output[i] ==
             static_cast<unsigned char>(input[i] * rhs[i]));
    }
    pjrtx_mlx_metal_buffer_destroy(product);

    PjrtxMlxMetalBuffer* quotient = pjrtx_mlx_metal_buffer_binary_u8(
        rhs_buffer, buffer, PJRTX_MLX_METAL_U8_BINARY_DIVIDE);
    assert(quotient != nullptr);
    unsigned char quotient_output[sizeof(input)] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               quotient, quotient_output, sizeof(quotient_output)) == 1);
    for (int i = 0; i < static_cast<int>(sizeof(input)); ++i) {
      assert(quotient_output[i] ==
             static_cast<unsigned char>(rhs[i] / input[i]));
    }
    pjrtx_mlx_metal_buffer_destroy(quotient);

    PjrtxMlxMetalBuffer* negated = pjrtx_mlx_metal_buffer_unary_u8(
        buffer, PJRTX_MLX_METAL_U8_UNARY_NEGATE);
    assert(negated != nullptr);
    unsigned char negated_output[sizeof(input)] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               negated, negated_output, sizeof(negated_output)) == 1);
    for (int i = 0; i < static_cast<int>(sizeof(input)); ++i) {
      assert(negated_output[i] == static_cast<unsigned char>(0 - input[i]));
    }
    pjrtx_mlx_metal_buffer_destroy(negated);

    pjrtx_mlx_metal_buffer_destroy(rhs_buffer);
    pjrtx_mlx_metal_buffer_destroy(buffer);

    const int64_t f32_dims[] = {3};
    const float f32_input[] = {1.5f, -2.0f, 4.0f};
    const float f32_rhs[] = {2.0f, 3.0f, -0.5f};
    PjrtxMlxMetalBuffer* f32_buffer =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, f32_input, sizeof(f32_input),
            PJRTX_MLX_METAL_DTYPE_F32, f32_dims, 1);
    assert(f32_buffer != nullptr);
    float f32_output[3] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(f32_buffer, f32_output,
                                               sizeof(f32_output)) == 1);
    for (int i = 0; i < 3; ++i) {
      assert(near(f32_output[i], f32_input[i]));
    }

    PjrtxMlxMetalBuffer* f32_rhs_buffer =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, f32_rhs, sizeof(f32_rhs),
            PJRTX_MLX_METAL_DTYPE_F32, f32_dims, 1);
    assert(f32_rhs_buffer != nullptr);
    PjrtxMlxMetalBuffer* f32_sum = pjrtx_mlx_metal_buffer_binary(
        f32_buffer, f32_rhs_buffer, PJRTX_MLX_METAL_U8_BINARY_ADD);
    if (f32_sum != nullptr) {
      float f32_sum_output[3] = {};
      assert(pjrtx_mlx_metal_buffer_copy_to_host(
                 f32_sum, f32_sum_output, sizeof(f32_sum_output)) == 1);
      for (int i = 0; i < 3; ++i) {
        assert(near(f32_sum_output[i], f32_input[i] + f32_rhs[i]));
      }
      pjrtx_mlx_metal_buffer_destroy(f32_sum);
    }
    PjrtxMlxMetalBuffer* f32_sqrt = pjrtx_mlx_metal_buffer_unary(
        f32_buffer, PJRTX_MLX_METAL_UNARY_SQRT);
    if (f32_sqrt != nullptr) {
      float f32_sqrt_output[3] = {};
      assert(pjrtx_mlx_metal_buffer_copy_to_host(
                 f32_sqrt, f32_sqrt_output, sizeof(f32_sqrt_output)) == 1);
      assert(near(f32_sqrt_output[0], std::sqrt(f32_input[0])));
      assert(std::isnan(f32_sqrt_output[1]));
      assert(near(f32_sqrt_output[2], std::sqrt(f32_input[2])));
      pjrtx_mlx_metal_buffer_destroy(f32_sqrt);
    }

    const int64_t transpose_dims[] = {2, 3};
    const int64_t permutation[] = {1, 0};
    const float transpose_input[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    PjrtxMlxMetalBuffer* transpose_buffer =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, transpose_input, sizeof(transpose_input),
            PJRTX_MLX_METAL_DTYPE_F32, transpose_dims, 2);
    assert(transpose_buffer != nullptr);
    PjrtxMlxMetalBuffer* transposed = pjrtx_mlx_metal_buffer_transpose(
        transpose_buffer, permutation, 2);
    if (transposed != nullptr) {
      float transposed_output[6] = {};
      assert(pjrtx_mlx_metal_buffer_copy_to_host(
                 transposed, transposed_output, sizeof(transposed_output)) == 1);
      const float expected[] = {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};
      for (int i = 0; i < 6; ++i) {
        assert(near(transposed_output[i], expected[i]));
      }
      pjrtx_mlx_metal_buffer_destroy(transposed);
    }
    pjrtx_mlx_metal_buffer_destroy(transpose_buffer);

    const int64_t broadcast_dims[] = {3};
    const int64_t broadcast_dimensions[] = {1};
    const int64_t broadcast_output_dims[] = {2, 3};
    const float broadcast_input[] = {7.0f, 8.0f, 9.0f};
    PjrtxMlxMetalBuffer* broadcast_buffer =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, broadcast_input, sizeof(broadcast_input),
            PJRTX_MLX_METAL_DTYPE_F32, broadcast_dims, 1);
    assert(broadcast_buffer != nullptr);
    PjrtxMlxMetalBuffer* broadcasted =
        pjrtx_mlx_metal_buffer_broadcast_in_dim(
            broadcast_buffer, broadcast_dimensions, 1, broadcast_output_dims, 2);
    if (broadcasted != nullptr) {
      float broadcast_output[6] = {};
      assert(pjrtx_mlx_metal_buffer_copy_to_host(
                 broadcasted, broadcast_output, sizeof(broadcast_output)) == 1);
      const float expected[] = {7.0f, 8.0f, 9.0f, 7.0f, 8.0f, 9.0f};
      for (int i = 0; i < 6; ++i) {
        assert(near(broadcast_output[i], expected[i]));
      }
      pjrtx_mlx_metal_buffer_destroy(broadcasted);
    }
    pjrtx_mlx_metal_buffer_destroy(broadcast_buffer);

    const int64_t slice_dims[] = {3, 4};
    const int64_t slice_start[] = {1, 0};
    const int64_t slice_limit[] = {3, 4};
    const int64_t slice_strides[] = {1, 2};
    const int64_t slice_output_dims[] = {2, 2};
    const float slice_input[] = {1.0f, 2.0f, 3.0f, 4.0f,
                                 5.0f, 6.0f, 7.0f, 8.0f,
                                 9.0f, 10.0f, 11.0f, 12.0f};
    PjrtxMlxMetalBuffer* slice_buffer =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, slice_input, sizeof(slice_input),
            PJRTX_MLX_METAL_DTYPE_F32, slice_dims, 2);
    assert(slice_buffer != nullptr);
    PjrtxMlxMetalBuffer* sliced = pjrtx_mlx_metal_buffer_slice(
        slice_buffer, slice_start, slice_limit, slice_strides, 2,
        slice_output_dims, 2);
    if (sliced != nullptr) {
      float slice_output[4] = {};
      assert(pjrtx_mlx_metal_buffer_copy_to_host(
                 sliced, slice_output, sizeof(slice_output)) == 1);
      const float expected[] = {5.0f, 7.0f, 9.0f, 11.0f};
      for (int i = 0; i < 4; ++i) {
        assert(near(slice_output[i], expected[i]));
      }
      pjrtx_mlx_metal_buffer_destroy(sliced);
    }
    pjrtx_mlx_metal_buffer_destroy(slice_buffer);

    const int64_t concat_lhs_dims[] = {2, 2};
    const int64_t concat_rhs_dims[] = {2, 3};
    const int64_t concat_output_dims[] = {2, 5};
    const float concat_lhs_input[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float concat_rhs_input[] = {5.0f, 6.0f, 7.0f,
                                      8.0f, 9.0f, 10.0f};
    PjrtxMlxMetalBuffer* concat_lhs =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, concat_lhs_input, sizeof(concat_lhs_input),
            PJRTX_MLX_METAL_DTYPE_F32, concat_lhs_dims, 2);
    assert(concat_lhs != nullptr);
    PjrtxMlxMetalBuffer* concat_rhs =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, concat_rhs_input, sizeof(concat_rhs_input),
            PJRTX_MLX_METAL_DTYPE_F32, concat_rhs_dims, 2);
    assert(concat_rhs != nullptr);
    PjrtxMlxMetalBuffer* concatenated = pjrtx_mlx_metal_buffer_concatenate(
        concat_lhs, concat_rhs, 1, concat_output_dims, 2);
    if (concatenated != nullptr) {
      float concat_output[10] = {};
      assert(pjrtx_mlx_metal_buffer_copy_to_host(
                 concatenated, concat_output, sizeof(concat_output)) == 1);
      const float expected[] = {1.0f, 2.0f, 5.0f, 6.0f, 7.0f,
                                3.0f, 4.0f, 8.0f, 9.0f, 10.0f};
      for (int i = 0; i < 10; ++i) {
        assert(near(concat_output[i], expected[i]));
      }
      pjrtx_mlx_metal_buffer_destroy(concatenated);
    }
    pjrtx_mlx_metal_buffer_destroy(concat_rhs);
    pjrtx_mlx_metal_buffer_destroy(concat_lhs);

    pjrtx_mlx_metal_buffer_destroy(f32_rhs_buffer);
    pjrtx_mlx_metal_buffer_destroy(f32_buffer);
  }

  assert(pjrtx_mlx_metal_copy_devices(nullptr, 8) == 0);
  assert(pjrtx_mlx_metal_copy_devices(devices, 0) == 0);
  assert(pjrtx_mlx_metal_buffer_from_host(0, nullptr, 1) == nullptr);
  assert(pjrtx_mlx_metal_buffer_from_host(0, devices, 0) == nullptr);
  assert(pjrtx_mlx_metal_buffer_clone(nullptr) == nullptr);
  assert(pjrtx_mlx_metal_buffer_add_u8(nullptr, nullptr) == nullptr);
  assert(pjrtx_mlx_metal_buffer_binary_u8(nullptr, nullptr,
                                          PJRTX_MLX_METAL_U8_BINARY_ADD) ==
         nullptr);
  assert(pjrtx_mlx_metal_buffer_unary_u8(nullptr,
                                         PJRTX_MLX_METAL_U8_UNARY_NEGATE) ==
         nullptr);
  return 0;
}
