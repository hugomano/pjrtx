#include "src/backend/mlx_metal/api.h"

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
    PjrtxMlxMetalBuffer* gathered = pjrtx_mlx_metal_buffer_gather_axis(
        gather_operand, gather_indices, 0, 1, gather_output_dims, 2);
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
    pjrtx_mlx_metal_buffer_destroy(gather_indices);
    pjrtx_mlx_metal_buffer_destroy(gather_operand);

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
    pjrtx_mlx_metal_buffer_destroy(sort_buffer);

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
