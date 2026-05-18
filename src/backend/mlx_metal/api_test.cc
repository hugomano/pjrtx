#include "src/backend/mlx_metal/api.h"

#include <cassert>
#include <cmath>
#include <cstdint>

bool near(float lhs, float rhs) { return std::fabs(lhs - rhs) < 0.0001f; }
bool near_relaxed(float lhs, float rhs) { return std::fabs(lhs - rhs) < 0.01f; }

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
    assert(pjrtx_mlx_metal_buffer_has_host_shadow(buffer) == 0);

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
    assert(pjrtx_mlx_metal_buffer_has_host_shadow(cloned) == 0);
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
    assert(pjrtx_mlx_metal_buffer_has_host_shadow(sum) == 0);
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

    PjrtxMlxMetalBuffer* bitwise_and = pjrtx_mlx_metal_buffer_binary_u8(
        buffer, rhs_buffer, PJRTX_MLX_METAL_BINARY_AND);
    assert(bitwise_and != nullptr);
    unsigned char bitwise_and_output[sizeof(input)] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               bitwise_and, bitwise_and_output, sizeof(bitwise_and_output)) ==
           1);
    for (int i = 0; i < static_cast<int>(sizeof(input)); ++i) {
      assert(bitwise_and_output[i] == static_cast<unsigned char>(input[i] & rhs[i]));
    }
    pjrtx_mlx_metal_buffer_destroy(bitwise_and);

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

    const int64_t iota_dims[] = {2, 3};
    PjrtxMlxMetalBuffer* iota = pjrtx_mlx_metal_buffer_iota(
        devices[0].ordinal, PJRTX_MLX_METAL_DTYPE_F32, iota_dims, 2, 1);
    assert(iota != nullptr);
    float iota_output[6] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               iota, iota_output, sizeof(iota_output)) == 1);
    {
      const float expected[] = {0.0f, 1.0f, 2.0f, 0.0f, 1.0f, 2.0f};
      for (int i = 0; i < 6; ++i) {
        assert(near(iota_output[i], expected[i]));
      }
    }
    pjrtx_mlx_metal_buffer_destroy(iota);

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
    const float clamp_min_scalar = -1.0f;
    const float clamp_max_scalar = 2.0f;
    PjrtxMlxMetalBuffer* clamp_min =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, &clamp_min_scalar, sizeof(clamp_min_scalar),
            PJRTX_MLX_METAL_DTYPE_F32, nullptr, 0);
    PjrtxMlxMetalBuffer* clamp_max =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, &clamp_max_scalar, sizeof(clamp_max_scalar),
            PJRTX_MLX_METAL_DTYPE_F32, nullptr, 0);
    assert(clamp_min != nullptr);
    assert(clamp_max != nullptr);
    PjrtxMlxMetalBuffer* clamped = pjrtx_mlx_metal_buffer_clamp(
        clamp_min, f32_buffer, clamp_max, f32_dims, 1);
    assert(clamped != nullptr);
    float clamp_output[3] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               clamped, clamp_output, sizeof(clamp_output)) == 1);
    {
      const float expected[] = {1.5f, -1.0f, 2.0f};
      for (int i = 0; i < 3; ++i) {
        assert(near(clamp_output[i], expected[i]));
      }
    }
    pjrtx_mlx_metal_buffer_destroy(clamped);
    pjrtx_mlx_metal_buffer_destroy(clamp_max);
    pjrtx_mlx_metal_buffer_destroy(clamp_min);

    const int64_t half_dims[] = {2};
    const uint16_t f16_input[] = {0x3c00, 0xc000};
    PjrtxMlxMetalBuffer* f16_buffer =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, f16_input, sizeof(f16_input),
            PJRTX_MLX_METAL_DTYPE_F16, half_dims, 1);
    assert(f16_buffer != nullptr);
    uint16_t f16_output[2] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(f16_buffer, f16_output,
                                               sizeof(f16_output)) == 1);
    assert(f16_output[0] == f16_input[0]);
    assert(f16_output[1] == f16_input[1]);
    PjrtxMlxMetalBuffer* f16_as_f32 = pjrtx_mlx_metal_buffer_astype(
        f16_buffer, PJRTX_MLX_METAL_DTYPE_F32);
    assert(f16_as_f32 != nullptr);
    assert(pjrtx_mlx_metal_buffer_size(f16_as_f32) == sizeof(float) * 2);
    assert(pjrtx_mlx_metal_buffer_has_host_shadow(f16_as_f32) == 0);
    float f16_as_f32_output[2] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               f16_as_f32, f16_as_f32_output, sizeof(f16_as_f32_output)) == 1);
    assert(near(f16_as_f32_output[0], 1.0f));
    assert(near(f16_as_f32_output[1], -2.0f));
    pjrtx_mlx_metal_buffer_destroy(f16_as_f32);
    pjrtx_mlx_metal_buffer_destroy(f16_buffer);

    const uint16_t bf16_input[] = {0x3f80, 0xc000};
    PjrtxMlxMetalBuffer* bf16_buffer =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, bf16_input, sizeof(bf16_input),
            PJRTX_MLX_METAL_DTYPE_BF16, half_dims, 1);
    assert(bf16_buffer != nullptr);
    uint16_t bf16_output[2] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(bf16_buffer, bf16_output,
                                               sizeof(bf16_output)) == 1);
    assert(bf16_output[0] == bf16_input[0]);
    assert(bf16_output[1] == bf16_input[1]);
    pjrtx_mlx_metal_buffer_destroy(bf16_buffer);

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
    PjrtxMlxMetalBuffer* f32_atan2 = pjrtx_mlx_metal_buffer_binary(
        f32_buffer, f32_rhs_buffer, PJRTX_MLX_METAL_BINARY_ATAN2);
    assert(f32_atan2 != nullptr);
    float f32_atan2_output[3] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               f32_atan2, f32_atan2_output, sizeof(f32_atan2_output)) == 1);
    for (int i = 0; i < 3; ++i) {
      assert(near(f32_atan2_output[i], std::atan2(f32_input[i], f32_rhs[i])));
    }
    pjrtx_mlx_metal_buffer_destroy(f32_atan2);
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
    PjrtxMlxMetalBuffer* f32_expm1 = pjrtx_mlx_metal_buffer_unary(
        f32_buffer, PJRTX_MLX_METAL_UNARY_EXPM1);
    assert(f32_expm1 != nullptr);
    float f32_expm1_output[3] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               f32_expm1, f32_expm1_output, sizeof(f32_expm1_output)) == 1);
    for (int i = 0; i < 3; ++i) {
      assert(near_relaxed(f32_expm1_output[i], std::expm1(f32_input[i])));
    }
    pjrtx_mlx_metal_buffer_destroy(f32_expm1);
    PjrtxMlxMetalBuffer* f32_isfinite = pjrtx_mlx_metal_buffer_unary(
        f32_buffer, PJRTX_MLX_METAL_UNARY_ISFINITE);
    assert(f32_isfinite != nullptr);
    unsigned char f32_isfinite_output[3] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               f32_isfinite, f32_isfinite_output,
               sizeof(f32_isfinite_output)) == 1);
    for (int i = 0; i < 3; ++i) {
      assert(f32_isfinite_output[i] == 1);
    }
    pjrtx_mlx_metal_buffer_destroy(f32_isfinite);

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

    const int32_t dynamic_start0 = 1;
    const int32_t dynamic_start1 = 1;
    PjrtxMlxMetalBuffer* dynamic_start_buffers[] = {
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, &dynamic_start0, sizeof(dynamic_start0),
            PJRTX_MLX_METAL_DTYPE_S32, nullptr, 0),
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, &dynamic_start1, sizeof(dynamic_start1),
            PJRTX_MLX_METAL_DTYPE_S32, nullptr, 0),
    };
    assert(dynamic_start_buffers[0] != nullptr);
    assert(dynamic_start_buffers[1] != nullptr);

    PjrtxMlxMetalBuffer* dynamic_source =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, slice_input, sizeof(slice_input),
            PJRTX_MLX_METAL_DTYPE_F32, slice_dims, 2);
    assert(dynamic_source != nullptr);
    const int64_t dynamic_slice_sizes[] = {2, 2};
    PjrtxMlxMetalBuffer* dynamic_sliced =
        pjrtx_mlx_metal_buffer_dynamic_slice(
            dynamic_source, dynamic_start_buffers, 2, dynamic_slice_sizes, 2,
            slice_output_dims, 2);
    assert(dynamic_sliced != nullptr);
    float dynamic_slice_output[4] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               dynamic_sliced, dynamic_slice_output,
               sizeof(dynamic_slice_output)) == 1);
    {
      const float expected[] = {6.0f, 7.0f, 10.0f, 11.0f};
      for (int i = 0; i < 4; ++i) {
        assert(near(dynamic_slice_output[i], expected[i]));
      }
    }
    pjrtx_mlx_metal_buffer_destroy(dynamic_sliced);

    const float dynamic_update_input[] = {100.0f, 101.0f, 102.0f, 103.0f};
    PjrtxMlxMetalBuffer* dynamic_update =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, dynamic_update_input,
            sizeof(dynamic_update_input), PJRTX_MLX_METAL_DTYPE_F32,
            slice_output_dims, 2);
    assert(dynamic_update != nullptr);
    PjrtxMlxMetalBuffer* dynamic_updated =
        pjrtx_mlx_metal_buffer_dynamic_update_slice(
            dynamic_source, dynamic_update, dynamic_start_buffers, 2,
            slice_dims, 2);
    assert(dynamic_updated != nullptr);
    float dynamic_update_output[12] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               dynamic_updated, dynamic_update_output,
               sizeof(dynamic_update_output)) == 1);
    {
      const float expected[] = {1.0f, 2.0f, 3.0f, 4.0f,
                                5.0f, 100.0f, 101.0f, 8.0f,
                                9.0f, 102.0f, 103.0f, 12.0f};
      for (int i = 0; i < 12; ++i) {
        assert(near(dynamic_update_output[i], expected[i]));
      }
    }
    pjrtx_mlx_metal_buffer_destroy(dynamic_updated);
    pjrtx_mlx_metal_buffer_destroy(dynamic_update);

    const int64_t pad_dims[] = {2};
    const int64_t pad_low[] = {1};
    const int64_t pad_high[] = {2};
    const int64_t pad_interior[] = {0};
    const int64_t pad_output_dims[] = {5};
    const float pad_input[] = {2.0f, 3.0f};
    const float pad_value = 0.0f;
    PjrtxMlxMetalBuffer* pad_buffer =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, pad_input, sizeof(pad_input),
            PJRTX_MLX_METAL_DTYPE_F32, pad_dims, 1);
    PjrtxMlxMetalBuffer* pad_value_buffer =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, &pad_value, sizeof(pad_value),
            PJRTX_MLX_METAL_DTYPE_F32, nullptr, 0);
    assert(pad_buffer != nullptr);
    assert(pad_value_buffer != nullptr);
    PjrtxMlxMetalBuffer* padded = pjrtx_mlx_metal_buffer_pad(
        pad_buffer, pad_value_buffer, pad_low, pad_high, pad_interior, 1,
        pad_output_dims, 1);
    assert(padded != nullptr);
    float pad_output[5] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(padded, pad_output,
                                               sizeof(pad_output)) == 1);
    {
      const float expected[] = {0.0f, 2.0f, 3.0f, 0.0f, 0.0f};
      for (int i = 0; i < 5; ++i) {
        assert(near(pad_output[i], expected[i]));
      }
    }

    const int64_t pad_interior_strided[] = {1};
    const int64_t pad_interior_output_dims[] = {6};
    PjrtxMlxMetalBuffer* interior_padded = pjrtx_mlx_metal_buffer_pad(
        pad_buffer, pad_value_buffer, pad_low, pad_high, pad_interior_strided,
        1, pad_interior_output_dims, 1);
    assert(interior_padded != nullptr);
    float interior_pad_output[6] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               interior_padded, interior_pad_output,
               sizeof(interior_pad_output)) == 1);
    {
      const float expected[] = {0.0f, 2.0f, 0.0f, 3.0f, 0.0f, 0.0f};
      for (int i = 0; i < 6; ++i) {
        assert(near(interior_pad_output[i], expected[i]));
      }
    }
    pjrtx_mlx_metal_buffer_destroy(interior_padded);
    pjrtx_mlx_metal_buffer_destroy(padded);
    pjrtx_mlx_metal_buffer_destroy(pad_value_buffer);
    pjrtx_mlx_metal_buffer_destroy(pad_buffer);

    const int64_t reverse_dims[] = {2, 3};
    const int64_t reverse_axes[] = {1};
    const float reverse_input[] = {1.0f, 2.0f, 3.0f,
                                   4.0f, 5.0f, 6.0f};
    PjrtxMlxMetalBuffer* reverse_buffer =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, reverse_input, sizeof(reverse_input),
            PJRTX_MLX_METAL_DTYPE_F32, reverse_dims, 2);
    assert(reverse_buffer != nullptr);
    PjrtxMlxMetalBuffer* reversed = pjrtx_mlx_metal_buffer_reverse(
        reverse_buffer, reverse_axes, 1, reverse_dims, 2);
    assert(reversed != nullptr);
    float reverse_output[6] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               reversed, reverse_output, sizeof(reverse_output)) == 1);
    {
      const float expected[] = {3.0f, 2.0f, 1.0f, 6.0f, 5.0f, 4.0f};
      for (int i = 0; i < 6; ++i) {
        assert(near(reverse_output[i], expected[i]));
      }
    }
    pjrtx_mlx_metal_buffer_destroy(reversed);
    pjrtx_mlx_metal_buffer_destroy(reverse_buffer);

    const int64_t gather_operand_dims[] = {3, 2};
    const int64_t gather_indices_dims[] = {2};
    const int64_t gather_output_dims[] = {2, 2};
    const float gather_operand_input[] = {1.0f, 2.0f, 3.0f,
                                          4.0f, 5.0f, 6.0f};
    const int32_t gather_indices_input[] = {2, 0};
    PjrtxMlxMetalBuffer* gather_operand =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, gather_operand_input,
            sizeof(gather_operand_input), PJRTX_MLX_METAL_DTYPE_F32,
            gather_operand_dims, 2);
    PjrtxMlxMetalBuffer* gather_indices =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, gather_indices_input,
            sizeof(gather_indices_input), PJRTX_MLX_METAL_DTYPE_S32,
            gather_indices_dims, 1);
    assert(gather_operand != nullptr);
    assert(gather_indices != nullptr);
    const int32_t gather_zero_input[] = {0, 0};
    const int32_t gather_bound_input[] = {3, 3};
    PjrtxMlxMetalBuffer* gather_zero =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, gather_zero_input, sizeof(gather_zero_input),
            PJRTX_MLX_METAL_DTYPE_S32, gather_indices_dims, 1);
    PjrtxMlxMetalBuffer* gather_bound =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, gather_bound_input, sizeof(gather_bound_input),
            PJRTX_MLX_METAL_DTYPE_S32, gather_indices_dims, 1);
    assert(gather_zero != nullptr);
    assert(gather_bound != nullptr);
    PjrtxMlxMetalBuffer* gather_negative_mask = pjrtx_mlx_metal_buffer_compare(
        gather_indices, gather_zero, PJRTX_MLX_METAL_COMPARE_LT,
        gather_indices_dims, 1);
    assert(gather_negative_mask != nullptr);
    PjrtxMlxMetalBuffer* gather_normalized_indices = pjrtx_mlx_metal_buffer_select(
        gather_negative_mask, gather_bound, gather_indices, gather_indices_dims,
        1);
    assert(gather_normalized_indices != nullptr);
    PjrtxMlxMetalBuffer* gathered = pjrtx_mlx_metal_buffer_gather_axis(
        gather_operand, gather_normalized_indices, 0, 1, gather_output_dims, 2);
    assert(gathered != nullptr);
    float gather_output[4] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               gathered, gather_output, sizeof(gather_output)) == 1);
    {
      const float expected[] = {5.0f, 6.0f, 1.0f, 2.0f};
      for (int i = 0; i < 4; ++i) {
        assert(near(gather_output[i], expected[i]));
      }
    }
    pjrtx_mlx_metal_buffer_destroy(gathered);
    pjrtx_mlx_metal_buffer_destroy(gather_normalized_indices);
    pjrtx_mlx_metal_buffer_destroy(gather_negative_mask);
    pjrtx_mlx_metal_buffer_destroy(gather_bound);
    pjrtx_mlx_metal_buffer_destroy(gather_zero);
    pjrtx_mlx_metal_buffer_destroy(gather_indices);
    pjrtx_mlx_metal_buffer_destroy(gather_operand);

    const int32_t gather_axis1_indices_input[] = {1, 0};
    const int64_t gather_axis1_output_dims[] = {3, 2};
    PjrtxMlxMetalBuffer* gather_axis1_operand =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, gather_operand_input,
            sizeof(gather_operand_input), PJRTX_MLX_METAL_DTYPE_F32,
            gather_operand_dims, 2);
    PjrtxMlxMetalBuffer* gather_axis1_indices =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, gather_axis1_indices_input,
            sizeof(gather_axis1_indices_input), PJRTX_MLX_METAL_DTYPE_S32,
            gather_indices_dims, 1);
    assert(gather_axis1_operand != nullptr);
    assert(gather_axis1_indices != nullptr);
    PjrtxMlxMetalBuffer* gathered_axis1 = pjrtx_mlx_metal_buffer_gather_axis(
        gather_axis1_operand, gather_axis1_indices, 1, 1,
        gather_axis1_output_dims, 2);
    assert(gathered_axis1 != nullptr);
    float gather_axis1_output[6] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               gathered_axis1, gather_axis1_output,
               sizeof(gather_axis1_output)) == 1);
    {
      const float expected[] = {2.0f, 1.0f, 4.0f, 3.0f, 6.0f, 5.0f};
      for (int i = 0; i < 6; ++i) {
        assert(near(gather_axis1_output[i], expected[i]));
      }
    }
    pjrtx_mlx_metal_buffer_destroy(gathered_axis1);
    pjrtx_mlx_metal_buffer_destroy(gather_axis1_indices);
    pjrtx_mlx_metal_buffer_destroy(gather_axis1_operand);

    const int32_t gather_nd_indices_input[] = {2, 1, 0, 0};
    const int64_t gather_nd_indices_dims[] = {2, 2};
    const int64_t gather_nd_start_map[] = {0, 1};
    const int64_t gather_nd_collapsed[] = {0, 1};
    const int64_t gather_nd_slice_sizes[] = {1, 1};
    const int64_t gather_nd_output_dims[] = {2};
    PjrtxMlxMetalBuffer* gather_nd_operand =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, gather_operand_input,
            sizeof(gather_operand_input), PJRTX_MLX_METAL_DTYPE_F32,
            gather_operand_dims, 2);
    PjrtxMlxMetalBuffer* gather_nd_indices =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, gather_nd_indices_input,
            sizeof(gather_nd_indices_input), PJRTX_MLX_METAL_DTYPE_S32,
            gather_nd_indices_dims, 2);
    assert(gather_nd_operand != nullptr);
    assert(gather_nd_indices != nullptr);
    PjrtxMlxMetalBuffer* gathered_nd = pjrtx_mlx_metal_buffer_gather(
        gather_nd_operand, gather_nd_indices, gather_nd_start_map, 2,
        gather_nd_collapsed, 2, nullptr, 0, nullptr, 0, 1,
        gather_nd_slice_sizes, 2, nullptr, 0, gather_nd_output_dims, 1);
    assert(gathered_nd != nullptr);
    float gather_nd_output[2] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               gathered_nd, gather_nd_output, sizeof(gather_nd_output)) == 1);
    {
      const float expected[] = {6.0f, 1.0f};
      for (int i = 0; i < 2; ++i) {
        assert(near(gather_nd_output[i], expected[i]));
      }
    }
    pjrtx_mlx_metal_buffer_destroy(gathered_nd);
    pjrtx_mlx_metal_buffer_destroy(gather_nd_indices);
    pjrtx_mlx_metal_buffer_destroy(gather_nd_operand);

    const int64_t batched_gather_operand_dims[] = {2, 3, 4};
    const int64_t batched_gather_indices_dims[] = {2, 2, 1};
    const int64_t batched_gather_start_map[] = {1};
    const int64_t batched_gather_collapsed[] = {1};
    const int64_t batched_gather_operand_batching[] = {0};
    const int64_t batched_gather_start_batching[] = {0};
    const int64_t batched_gather_slice_sizes[] = {1, 1, 4};
    const int64_t batched_gather_offset_dims[] = {2};
    const int64_t batched_gather_output_dims[] = {2, 2, 4};
    const float batched_gather_operand_input[] = {
        0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,
        6.0f,  7.0f,  8.0f,  9.0f,  10.0f, 11.0f,
        12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f,
        18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f,
    };
    const int32_t batched_gather_indices_input[] = {2, 0, 1, 2};
    PjrtxMlxMetalBuffer* batched_gather_operand =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, batched_gather_operand_input,
            sizeof(batched_gather_operand_input), PJRTX_MLX_METAL_DTYPE_F32,
            batched_gather_operand_dims, 3);
    PjrtxMlxMetalBuffer* batched_gather_indices =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, batched_gather_indices_input,
            sizeof(batched_gather_indices_input), PJRTX_MLX_METAL_DTYPE_S32,
            batched_gather_indices_dims, 3);
    assert(batched_gather_operand != nullptr);
    assert(batched_gather_indices != nullptr);
    PjrtxMlxMetalBuffer* batched_gathered = pjrtx_mlx_metal_buffer_gather(
        batched_gather_operand, batched_gather_indices,
        batched_gather_start_map, 1, batched_gather_collapsed, 1,
        batched_gather_operand_batching, 1, batched_gather_start_batching, 1,
        2, batched_gather_slice_sizes, 3, batched_gather_offset_dims, 1,
        batched_gather_output_dims, 3);
    assert(batched_gathered != nullptr);
    float batched_gather_output[16] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               batched_gathered, batched_gather_output,
               sizeof(batched_gather_output)) == 1);
    {
      const float expected[] = {
          8.0f,  9.0f,  10.0f, 11.0f, 0.0f,  1.0f,  2.0f,  3.0f,
          16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f,
      };
      for (std::size_t i = 0; i < 16; ++i) {
        assert(near(batched_gather_output[i], expected[i]));
      }
    }
    pjrtx_mlx_metal_buffer_destroy(batched_gathered);
    pjrtx_mlx_metal_buffer_destroy(batched_gather_indices);
    pjrtx_mlx_metal_buffer_destroy(batched_gather_operand);

    const int64_t scatter_dims[] = {4};
    const int64_t scatter_indices_dims[] = {2};
    const float scatter_operand_input[] = {0.0f, 10.0f, 20.0f, 30.0f};
    const int32_t scatter_indices_input[] = {1, 3};
    const float scatter_updates_input[] = {5.0f, 7.0f};
    PjrtxMlxMetalBuffer* scatter_operand =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, scatter_operand_input,
            sizeof(scatter_operand_input), PJRTX_MLX_METAL_DTYPE_F32,
            scatter_dims, 1);
    PjrtxMlxMetalBuffer* scatter_indices =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, scatter_indices_input,
            sizeof(scatter_indices_input), PJRTX_MLX_METAL_DTYPE_S32,
            scatter_indices_dims, 1);
    PjrtxMlxMetalBuffer* scatter_updates =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, scatter_updates_input,
            sizeof(scatter_updates_input), PJRTX_MLX_METAL_DTYPE_F32,
            scatter_indices_dims, 1);
    assert(scatter_operand != nullptr);
    assert(scatter_indices != nullptr);
    assert(scatter_updates != nullptr);
    PjrtxMlxMetalBuffer* scattered = pjrtx_mlx_metal_buffer_scatter_axis(
        scatter_operand, scatter_indices, scatter_updates, 0, 1,
        PJRTX_MLX_METAL_SCATTER_SET, scatter_dims, 1);
    assert(scattered != nullptr);
    float scatter_output[4] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               scattered, scatter_output, sizeof(scatter_output)) == 1);
    {
      const float expected[] = {0.0f, 5.0f, 20.0f, 7.0f};
      for (int i = 0; i < 4; ++i) {
        assert(near(scatter_output[i], expected[i]));
      }
    }
    PjrtxMlxMetalBuffer* scatter_added = pjrtx_mlx_metal_buffer_scatter_axis(
        scatter_operand, scatter_indices, scatter_updates, 0, 1,
        PJRTX_MLX_METAL_SCATTER_ADD, scatter_dims, 1);
    assert(scatter_added != nullptr);
    float scatter_add_output[4] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               scatter_added, scatter_add_output,
               sizeof(scatter_add_output)) == 1);
    {
      const float expected[] = {0.0f, 15.0f, 20.0f, 37.0f};
      for (int i = 0; i < 4; ++i) {
        assert(near(scatter_add_output[i], expected[i]));
      }
    }
    pjrtx_mlx_metal_buffer_destroy(scatter_added);
    pjrtx_mlx_metal_buffer_destroy(scattered);
    pjrtx_mlx_metal_buffer_destroy(scatter_updates);
    pjrtx_mlx_metal_buffer_destroy(scatter_indices);
    pjrtx_mlx_metal_buffer_destroy(scatter_operand);

    const int64_t point_scatter_dims[] = {3, 2};
    const int64_t point_scatter_indices_dims[] = {2, 2};
    const int64_t point_scatter_axes[] = {0, 1};
    const int64_t point_scatter_updates_dims[] = {2};
    const float point_scatter_operand_input[] = {1.0f, 2.0f, 3.0f,
                                                 4.0f, 5.0f, 6.0f};
    const int32_t point_scatter_indices_input[] = {2, 1, 0, 0};
    const float point_scatter_updates_input[] = {50.0f, 60.0f};
    PjrtxMlxMetalBuffer* point_scatter_operand =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, point_scatter_operand_input,
            sizeof(point_scatter_operand_input), PJRTX_MLX_METAL_DTYPE_F32,
            point_scatter_dims, 2);
    PjrtxMlxMetalBuffer* point_scatter_indices =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, point_scatter_indices_input,
            sizeof(point_scatter_indices_input), PJRTX_MLX_METAL_DTYPE_S32,
            point_scatter_indices_dims, 2);
    PjrtxMlxMetalBuffer* point_scatter_updates =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, point_scatter_updates_input,
            sizeof(point_scatter_updates_input), PJRTX_MLX_METAL_DTYPE_F32,
            point_scatter_updates_dims, 1);
    assert(point_scatter_operand != nullptr);
    assert(point_scatter_indices != nullptr);
    assert(point_scatter_updates != nullptr);
    PjrtxMlxMetalBuffer* point_scattered = pjrtx_mlx_metal_buffer_scatter(
        point_scatter_operand, point_scatter_indices, point_scatter_updates,
        point_scatter_axes, 2, point_scatter_axes, 2, nullptr, 0, nullptr, 0,
        nullptr, 0, 1,
        PJRTX_MLX_METAL_SCATTER_SET, point_scatter_dims, 2);
    assert(point_scattered != nullptr);
    float point_scatter_output[6] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               point_scattered, point_scatter_output,
               sizeof(point_scatter_output)) == 1);
    {
      const float expected[] = {60.0f, 2.0f, 3.0f, 4.0f, 5.0f, 50.0f};
      for (int i = 0; i < 6; ++i) {
        assert(near(point_scatter_output[i], expected[i]));
      }
    }
    PjrtxMlxMetalBuffer* point_scatter_added = pjrtx_mlx_metal_buffer_scatter(
        point_scatter_operand, point_scatter_indices, point_scatter_updates,
        point_scatter_axes, 2, point_scatter_axes, 2, nullptr, 0, nullptr, 0,
        nullptr, 0, 1,
        PJRTX_MLX_METAL_SCATTER_ADD, point_scatter_dims, 2);
    assert(point_scatter_added != nullptr);
    float point_scatter_add_output[6] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               point_scatter_added, point_scatter_add_output,
               sizeof(point_scatter_add_output)) == 1);
    {
      const float expected[] = {61.0f, 2.0f, 3.0f, 4.0f, 5.0f, 56.0f};
      for (int i = 0; i < 6; ++i) {
        assert(near(point_scatter_add_output[i], expected[i]));
      }
    }
    pjrtx_mlx_metal_buffer_destroy(point_scatter_added);
    pjrtx_mlx_metal_buffer_destroy(point_scattered);
    pjrtx_mlx_metal_buffer_destroy(point_scatter_updates);
    pjrtx_mlx_metal_buffer_destroy(point_scatter_indices);
    pjrtx_mlx_metal_buffer_destroy(point_scatter_operand);

    const int64_t window_scatter_dims[] = {3, 4};
    const int64_t window_scatter_indices_dims[] = {2};
    const int64_t window_scatter_axis[] = {0};
    const int64_t window_scatter_update_window[] = {1};
    const int64_t window_scatter_updates_dims[] = {2, 2};
    const float window_scatter_operand_input[] = {
        1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,
        7.0f,  8.0f,  9.0f,  10.0f, 11.0f, 12.0f,
    };
    const int32_t window_scatter_indices_input[] = {2, 0};
    const float window_scatter_updates_input[] = {50.0f, 60.0f, 70.0f, 80.0f};
    PjrtxMlxMetalBuffer* window_scatter_operand =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, window_scatter_operand_input,
            sizeof(window_scatter_operand_input), PJRTX_MLX_METAL_DTYPE_F32,
            window_scatter_dims, 2);
    PjrtxMlxMetalBuffer* window_scatter_indices =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, window_scatter_indices_input,
            sizeof(window_scatter_indices_input), PJRTX_MLX_METAL_DTYPE_S32,
            window_scatter_indices_dims, 1);
    PjrtxMlxMetalBuffer* window_scatter_updates =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, window_scatter_updates_input,
            sizeof(window_scatter_updates_input), PJRTX_MLX_METAL_DTYPE_F32,
            window_scatter_updates_dims, 2);
    assert(window_scatter_operand != nullptr);
    assert(window_scatter_indices != nullptr);
    assert(window_scatter_updates != nullptr);
    PjrtxMlxMetalBuffer* window_scattered = pjrtx_mlx_metal_buffer_scatter(
        window_scatter_operand, window_scatter_indices, window_scatter_updates,
        window_scatter_axis, 1, window_scatter_axis, 1,
        window_scatter_update_window, 1, nullptr, 0, nullptr, 0, 1,
        PJRTX_MLX_METAL_SCATTER_SET,
        window_scatter_dims, 2);
    assert(window_scattered != nullptr);
    float window_scatter_output[12] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               window_scattered, window_scatter_output,
               sizeof(window_scatter_output)) == 1);
    {
      const float expected[] = {
          70.0f, 80.0f, 3.0f,  4.0f,  5.0f,  6.0f,
          7.0f,  8.0f,  50.0f, 60.0f, 11.0f, 12.0f,
      };
      for (int i = 0; i < 12; ++i) {
        assert(near(window_scatter_output[i], expected[i]));
      }
    }
    PjrtxMlxMetalBuffer* window_scatter_added = pjrtx_mlx_metal_buffer_scatter(
        window_scatter_operand, window_scatter_indices, window_scatter_updates,
        window_scatter_axis, 1, window_scatter_axis, 1,
        window_scatter_update_window, 1, nullptr, 0, nullptr, 0, 1,
        PJRTX_MLX_METAL_SCATTER_ADD,
        window_scatter_dims, 2);
    assert(window_scatter_added != nullptr);
    float window_scatter_add_output[12] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               window_scatter_added, window_scatter_add_output,
               sizeof(window_scatter_add_output)) == 1);
    {
      const float expected[] = {
          71.0f, 82.0f, 3.0f,  4.0f,  5.0f,  6.0f,
          7.0f,  8.0f,  59.0f, 70.0f, 11.0f, 12.0f,
      };
      for (int i = 0; i < 12; ++i) {
        assert(near(window_scatter_add_output[i], expected[i]));
      }
    }
    pjrtx_mlx_metal_buffer_destroy(window_scatter_added);
    pjrtx_mlx_metal_buffer_destroy(window_scattered);
    pjrtx_mlx_metal_buffer_destroy(window_scatter_updates);
    pjrtx_mlx_metal_buffer_destroy(window_scatter_indices);
    pjrtx_mlx_metal_buffer_destroy(window_scatter_operand);

    const int64_t batched_scatter_dims[] = {2, 3, 4};
    const int64_t batched_scatter_indices_dims[] = {2, 2, 1};
    const int64_t batched_scatter_axis[] = {1};
    const int64_t batched_scatter_input_batching[] = {0};
    const int64_t batched_scatter_indices_batching[] = {0};
    const int64_t batched_scatter_update_window[] = {2};
    const int64_t batched_scatter_updates_dims[] = {2, 2, 4};
    const float batched_scatter_operand_input[] = {
        0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,
        6.0f,  7.0f,  8.0f,  9.0f,  10.0f, 11.0f,
        12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f,
        18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f,
    };
    const int32_t batched_scatter_indices_input[] = {2, 0, 1, 2};
    const float batched_scatter_updates_input[] = {
        100.0f, 101.0f, 102.0f, 103.0f,
        104.0f, 105.0f, 106.0f, 107.0f,
        108.0f, 109.0f, 110.0f, 111.0f,
        112.0f, 113.0f, 114.0f, 115.0f,
    };
    PjrtxMlxMetalBuffer* batched_scatter_operand =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, batched_scatter_operand_input,
            sizeof(batched_scatter_operand_input), PJRTX_MLX_METAL_DTYPE_F32,
            batched_scatter_dims, 3);
    PjrtxMlxMetalBuffer* batched_scatter_indices =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, batched_scatter_indices_input,
            sizeof(batched_scatter_indices_input), PJRTX_MLX_METAL_DTYPE_S32,
            batched_scatter_indices_dims, 3);
    PjrtxMlxMetalBuffer* batched_scatter_updates =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, batched_scatter_updates_input,
            sizeof(batched_scatter_updates_input), PJRTX_MLX_METAL_DTYPE_F32,
            batched_scatter_updates_dims, 3);
    assert(batched_scatter_operand != nullptr);
    assert(batched_scatter_indices != nullptr);
    assert(batched_scatter_updates != nullptr);
    PjrtxMlxMetalBuffer* batched_scattered = pjrtx_mlx_metal_buffer_scatter(
        batched_scatter_operand, batched_scatter_indices,
        batched_scatter_updates, batched_scatter_axis, 1,
        batched_scatter_axis, 1, batched_scatter_update_window, 1,
        batched_scatter_input_batching, 1,
        batched_scatter_indices_batching, 1, 2,
        PJRTX_MLX_METAL_SCATTER_SET, batched_scatter_dims, 3);
    assert(batched_scattered != nullptr);
    float batched_scatter_output[24] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               batched_scattered, batched_scatter_output,
               sizeof(batched_scatter_output)) == 1);
    {
      const float expected[] = {
          104.0f, 105.0f, 106.0f, 107.0f,
          4.0f,   5.0f,   6.0f,   7.0f,
          100.0f, 101.0f, 102.0f, 103.0f,
          12.0f,  13.0f,  14.0f,  15.0f,
          108.0f, 109.0f, 110.0f, 111.0f,
          112.0f, 113.0f, 114.0f, 115.0f,
      };
      for (int i = 0; i < 24; ++i) {
        assert(near(batched_scatter_output[i], expected[i]));
      }
    }
    PjrtxMlxMetalBuffer* batched_scatter_added = pjrtx_mlx_metal_buffer_scatter(
        batched_scatter_operand, batched_scatter_indices,
        batched_scatter_updates, batched_scatter_axis, 1,
        batched_scatter_axis, 1, batched_scatter_update_window, 1,
        batched_scatter_input_batching, 1,
        batched_scatter_indices_batching, 1, 2,
        PJRTX_MLX_METAL_SCATTER_ADD, batched_scatter_dims, 3);
    assert(batched_scatter_added != nullptr);
    float batched_scatter_add_output[24] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               batched_scatter_added, batched_scatter_add_output,
               sizeof(batched_scatter_add_output)) == 1);
    {
      const float expected[] = {
          104.0f, 106.0f, 108.0f, 110.0f,
          4.0f,   5.0f,   6.0f,   7.0f,
          108.0f, 110.0f, 112.0f, 114.0f,
          12.0f,  13.0f,  14.0f,  15.0f,
          124.0f, 126.0f, 128.0f, 130.0f,
          132.0f, 134.0f, 136.0f, 138.0f,
      };
      for (int i = 0; i < 24; ++i) {
        assert(near(batched_scatter_add_output[i], expected[i]));
      }
    }
    pjrtx_mlx_metal_buffer_destroy(batched_scatter_added);
    pjrtx_mlx_metal_buffer_destroy(batched_scattered);
    pjrtx_mlx_metal_buffer_destroy(batched_scatter_updates);
    pjrtx_mlx_metal_buffer_destroy(batched_scatter_indices);
    pjrtx_mlx_metal_buffer_destroy(batched_scatter_operand);

    const int64_t sort_dims[] = {2, 3};
    const float sort_input[] = {3.0f, 1.0f, 2.0f, 6.0f, 4.0f, 5.0f};
    PjrtxMlxMetalBuffer* sort_buffer = pjrtx_mlx_metal_buffer_from_host_typed(
        devices[0].ordinal, sort_input, sizeof(sort_input),
        PJRTX_MLX_METAL_DTYPE_F32, sort_dims, 2);
    assert(sort_buffer != nullptr);
    PjrtxMlxMetalBuffer* sorted =
        pjrtx_mlx_metal_buffer_sort(sort_buffer, 1, sort_dims, 2);
    assert(sorted != nullptr);
    float sort_output[6] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               sorted, sort_output, sizeof(sort_output)) == 1);
    {
      const float expected[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
      for (int i = 0; i < 6; ++i) {
        assert(near(sort_output[i], expected[i]));
      }
    }
    pjrtx_mlx_metal_buffer_destroy(sorted);

    PjrtxMlxMetalBuffer* order = pjrtx_mlx_metal_buffer_argsort(
        sort_buffer, 1, PJRTX_MLX_METAL_DTYPE_S32, sort_dims, 2);
    assert(order != nullptr);
    int32_t order_output[6] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               order, order_output, sizeof(order_output)) == 1);
    {
      const int32_t expected[] = {1, 2, 0, 1, 2, 0};
      for (int i = 0; i < 6; ++i) {
        assert(order_output[i] == expected[i]);
      }
    }
    const int32_t value_input[] = {10, 11, 12, 20, 21, 22};
    PjrtxMlxMetalBuffer* value_buffer = pjrtx_mlx_metal_buffer_from_host_typed(
        devices[0].ordinal, value_input, sizeof(value_input),
        PJRTX_MLX_METAL_DTYPE_S32, sort_dims, 2);
    assert(value_buffer != nullptr);
    PjrtxMlxMetalBuffer* sorted_values =
        pjrtx_mlx_metal_buffer_take_along_axis(value_buffer, order, 1,
                                               sort_dims, 2);
    assert(sorted_values != nullptr);
    int32_t value_output[6] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               sorted_values, value_output, sizeof(value_output)) == 1);
    {
      const int32_t expected[] = {11, 12, 10, 21, 22, 20};
      for (int i = 0; i < 6; ++i) {
        assert(value_output[i] == expected[i]);
      }
    }
    pjrtx_mlx_metal_buffer_destroy(sorted_values);
    pjrtx_mlx_metal_buffer_destroy(value_buffer);
    pjrtx_mlx_metal_buffer_destroy(order);
    pjrtx_mlx_metal_buffer_destroy(sort_buffer);

    const int64_t pred_reduce_dims[] = {2, 3};
    const int64_t pred_reduce_axes[] = {1};
    const int64_t pred_reduce_output_dims[] = {2};
    const uint8_t pred_reduce_input[] = {1, 0, 1, 1, 1, 1};
    PjrtxMlxMetalBuffer* pred_reduce_buffer =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, pred_reduce_input, sizeof(pred_reduce_input),
            PJRTX_MLX_METAL_DTYPE_PRED, pred_reduce_dims, 2);
    assert(pred_reduce_buffer != nullptr);
    PjrtxMlxMetalBuffer* pred_reduce_all = pjrtx_mlx_metal_buffer_reduce(
        pred_reduce_buffer, PJRTX_MLX_METAL_REDUCE_AND, pred_reduce_axes, 1,
        pred_reduce_output_dims, 1);
    PjrtxMlxMetalBuffer* pred_reduce_any = pjrtx_mlx_metal_buffer_reduce(
        pred_reduce_buffer, PJRTX_MLX_METAL_REDUCE_OR, pred_reduce_axes, 1,
        pred_reduce_output_dims, 1);
    assert(pred_reduce_all != nullptr);
    assert(pred_reduce_any != nullptr);
    uint8_t pred_all_output[2] = {};
    uint8_t pred_any_output[2] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               pred_reduce_all, pred_all_output, sizeof(pred_all_output)) == 1);
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               pred_reduce_any, pred_any_output, sizeof(pred_any_output)) == 1);
    assert(pred_all_output[0] == 0);
    assert(pred_all_output[1] == 1);
    assert(pred_any_output[0] == 1);
    assert(pred_any_output[1] == 1);
    pjrtx_mlx_metal_buffer_destroy(pred_reduce_any);
    pjrtx_mlx_metal_buffer_destroy(pred_reduce_all);
    pjrtx_mlx_metal_buffer_destroy(pred_reduce_buffer);

    pjrtx_mlx_metal_buffer_destroy(dynamic_source);
    pjrtx_mlx_metal_buffer_destroy(dynamic_start_buffers[1]);
    pjrtx_mlx_metal_buffer_destroy(dynamic_start_buffers[0]);

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
  assert(pjrtx_mlx_metal_buffer_astype(nullptr,
                                       PJRTX_MLX_METAL_DTYPE_F32) == nullptr);
  assert(pjrtx_mlx_metal_buffer_add_u8(nullptr, nullptr) == nullptr);
  assert(pjrtx_mlx_metal_buffer_binary_u8(nullptr, nullptr,
                                          PJRTX_MLX_METAL_U8_BINARY_ADD) ==
         nullptr);
  assert(pjrtx_mlx_metal_buffer_unary_u8(nullptr,
                                         PJRTX_MLX_METAL_U8_UNARY_NEGATE) ==
         nullptr);
  return 0;
}
