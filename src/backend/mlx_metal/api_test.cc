#include "src/backend/mlx_metal/api.h"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstdint>

bool near(float lhs, float rhs) { return std::fabs(lhs - rhs) < 0.0001f; }
bool near_relaxed(float lhs, float rhs) { return std::fabs(lhs - rhs) < 0.01f; }

int main() {
  assert(pjrtx_mlx_metal_version_major() == 0);
  assert(pjrtx_mlx_metal_version_minor() >= 31);
  assert(pjrtx_mlx_metal_version_patch() >= 0);
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

    PjrtxMlxMetalBuffer* popcnt = pjrtx_mlx_metal_buffer_unary_u8(
        buffer, PJRTX_MLX_METAL_UNARY_POPCNT);
    assert(popcnt != nullptr);
    unsigned char popcnt_output[sizeof(input)] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               popcnt, popcnt_output, sizeof(popcnt_output)) == 1);
    const unsigned char expected_popcnt[] = {1, 1, 2, 1, 2, 2};
    for (int i = 0; i < static_cast<int>(sizeof(input)); ++i) {
      assert(popcnt_output[i] == expected_popcnt[i]);
    }
    pjrtx_mlx_metal_buffer_destroy(popcnt);

    PjrtxMlxMetalBuffer* clz = pjrtx_mlx_metal_buffer_unary_u8(
        buffer, PJRTX_MLX_METAL_UNARY_CLZ);
    assert(clz != nullptr);
    unsigned char clz_output[sizeof(input)] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(clz, clz_output,
                                               sizeof(clz_output)) == 1);
    const unsigned char expected_clz[] = {7, 6, 6, 5, 5, 5};
    for (int i = 0; i < static_cast<int>(sizeof(input)); ++i) {
      assert(clz_output[i] == expected_clz[i]);
    }
    pjrtx_mlx_metal_buffer_destroy(clz);

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

    {
      const int64_t broadcast_dims[] = {2, 3};
      const int64_t* scalar_dims = nullptr;
      const float matrix_input[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
      const float scalar_input[] = {2.0f};
      PjrtxMlxMetalBuffer* matrix =
          pjrtx_mlx_metal_buffer_from_host_typed(
              devices[0].ordinal, matrix_input, sizeof(matrix_input),
              PJRTX_MLX_METAL_DTYPE_F32, broadcast_dims, 2);
      PjrtxMlxMetalBuffer* scalar =
          pjrtx_mlx_metal_buffer_from_host_typed(
              devices[0].ordinal, scalar_input, sizeof(scalar_input),
              PJRTX_MLX_METAL_DTYPE_F32, scalar_dims, 0);
      assert(matrix != nullptr);
      assert(scalar != nullptr);
      PjrtxMlxMetalBuffer* broadcast_sum =
          pjrtx_mlx_metal_buffer_binary_out(
              matrix, scalar, PJRTX_MLX_METAL_U8_BINARY_ADD,
              broadcast_dims, 2);
      assert(broadcast_sum != nullptr);
      float broadcast_sum_output[6] = {};
      assert(pjrtx_mlx_metal_buffer_copy_to_host(
                 broadcast_sum, broadcast_sum_output,
                 sizeof(broadcast_sum_output)) == 1);
      for (int i = 0; i < 6; ++i) {
        assert(near(broadcast_sum_output[i],
                    matrix_input[i] + scalar_input[0]));
      }
      PjrtxMlxMetalBuffer* broadcast_compare =
          pjrtx_mlx_metal_buffer_compare(
              matrix, scalar, PJRTX_MLX_METAL_COMPARE_GT, broadcast_dims, 2);
      assert(broadcast_compare != nullptr);
      unsigned char broadcast_compare_output[6] = {};
      assert(pjrtx_mlx_metal_buffer_copy_to_host(
                 broadcast_compare, broadcast_compare_output,
                 sizeof(broadcast_compare_output)) == 1);
      const unsigned char expected_compare[] = {0, 0, 1, 1, 1, 1};
      for (int i = 0; i < 6; ++i) {
        assert(broadcast_compare_output[i] == expected_compare[i]);
      }
      pjrtx_mlx_metal_buffer_destroy(broadcast_compare);
      pjrtx_mlx_metal_buffer_destroy(broadcast_sum);
      pjrtx_mlx_metal_buffer_destroy(scalar);
      pjrtx_mlx_metal_buffer_destroy(matrix);
    }

    {
      const int64_t complex_dims[] = {2};
      const float complex_input[] = {3.0f, 4.0f, 5.0f, 12.0f};
      PjrtxMlxMetalBuffer* complex_buffer =
          pjrtx_mlx_metal_buffer_from_host_typed(
              devices[0].ordinal, complex_input, sizeof(complex_input),
              PJRTX_MLX_METAL_DTYPE_C64, complex_dims, 1);
      assert(complex_buffer != nullptr);
      PjrtxMlxMetalBuffer* abs_complex =
          pjrtx_mlx_metal_buffer_unary(complex_buffer,
                                       PJRTX_MLX_METAL_UNARY_ABS);
      assert(abs_complex != nullptr);
      assert(pjrtx_mlx_metal_buffer_size(abs_complex) ==
             2 * sizeof(float));
      float abs_complex_output[2] = {};
      assert(pjrtx_mlx_metal_buffer_copy_to_host(
                 abs_complex, abs_complex_output,
                 sizeof(abs_complex_output)) == 1);
      assert(near(abs_complex_output[0], 5.0f));
      assert(near(abs_complex_output[1], 13.0f));
      pjrtx_mlx_metal_buffer_destroy(abs_complex);
      pjrtx_mlx_metal_buffer_destroy(complex_buffer);
    }

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

    PjrtxMlxMetalBuffer* partition_id =
        pjrtx_mlx_metal_buffer_partition_id(
            devices[0].ordinal, PJRTX_MLX_METAL_DTYPE_U32, 7);
    assert(partition_id != nullptr);
    uint32_t partition_output = 0;
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               partition_id, &partition_output, sizeof(partition_output)) == 1);
    assert(partition_output == 7);
    pjrtx_mlx_metal_buffer_destroy(partition_id);

    {
      const int64_t dims[] = {4};
      const float input[] = {1.25f, -2.5f, 3.75f, 8.0f};
      auto* transfer = pjrtx_mlx_metal_async_h2d_create(
          devices[0].ordinal, PJRTX_MLX_METAL_DTYPE_F32, dims, 1,
          sizeof(input));
      assert(transfer != nullptr);
      assert(pjrtx_mlx_metal_async_h2d_write(
                 transfer, 0, input, 2 * sizeof(float)) == 1);
      assert(pjrtx_mlx_metal_async_h2d_write(
                 transfer, 2 * sizeof(float), input + 2,
                 2 * sizeof(float)) == 1);
      PjrtxMlxMetalBuffer* async_buffer =
          pjrtx_mlx_metal_async_h2d_finish(transfer);
      assert(async_buffer != nullptr);
      float output[4] = {};
      assert(pjrtx_mlx_metal_buffer_copy_to_host(
                 async_buffer, output, sizeof(output)) == 1);
      for (int i = 0; i < 4; ++i) {
        assert(near(output[i], input[i]));
      }
      pjrtx_mlx_metal_buffer_destroy(async_buffer);
      pjrtx_mlx_metal_async_h2d_destroy(transfer);
    }

    {
      const int64_t q_dims[] = {1, 1, 2};
      const int64_t kv_dims[] = {2, 1, 2};
      const float q_input[] = {1.0f, 0.0f};
      const float k_input[] = {1.0f, 0.0f, 0.0f, 1.0f};
      const float v_input[] = {10.0f, 0.0f, 0.0f, 20.0f};
      const uint32_t token_index = 0;
      PjrtxMlxMetalBuffer* q_buffer =
          pjrtx_mlx_metal_buffer_from_host_typed(
              devices[0].ordinal, q_input, sizeof(q_input),
              PJRTX_MLX_METAL_DTYPE_F32, q_dims, 3);
      PjrtxMlxMetalBuffer* k_buffer =
          pjrtx_mlx_metal_buffer_from_host_typed(
              devices[0].ordinal, k_input, sizeof(k_input),
              PJRTX_MLX_METAL_DTYPE_F32, kv_dims, 3);
      PjrtxMlxMetalBuffer* v_buffer =
          pjrtx_mlx_metal_buffer_from_host_typed(
              devices[0].ordinal, v_input, sizeof(v_input),
              PJRTX_MLX_METAL_DTYPE_F32, kv_dims, 3);
      PjrtxMlxMetalBuffer* token_buffer =
          pjrtx_mlx_metal_buffer_from_host_typed(
              devices[0].ordinal, &token_index, sizeof(token_index),
              PJRTX_MLX_METAL_DTYPE_U32, nullptr, 0);
      assert(q_buffer != nullptr);
      assert(k_buffer != nullptr);
      assert(v_buffer != nullptr);
      assert(token_buffer != nullptr);
      PjrtxMlxMetalBuffer* attention =
          pjrtx_mlx_metal_custom_call_scaled_dot_product_attention(
              q_buffer, k_buffer, v_buffer, token_buffer);
      assert(attention != nullptr);
      float attention_output[2] = {};
      assert(pjrtx_mlx_metal_buffer_copy_to_host(
                 attention, attention_output, sizeof(attention_output)) == 1);
      assert(near_relaxed(attention_output[0], 10.0f));
      assert(near_relaxed(attention_output[1], 0.0f));
      pjrtx_mlx_metal_buffer_destroy(attention);
      pjrtx_mlx_metal_buffer_destroy(token_buffer);
      pjrtx_mlx_metal_buffer_destroy(v_buffer);
      pjrtx_mlx_metal_buffer_destroy(k_buffer);
      pjrtx_mlx_metal_buffer_destroy(q_buffer);
    }

    {
      const int64_t q_dims[] = {1, 4, 2};
      const int64_t kv_dims[] = {2, 2, 2};
      const float q_input[] = {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
      const float k_input[] = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 1.0f};
      const float v_input[] = {10.0f, 0.0f, 20.0f, 0.0f, 0.0f, 30.0f, 0.0f, 40.0f};
      const uint32_t token_index = 0;
      PjrtxMlxMetalBuffer* q_buffer =
          pjrtx_mlx_metal_buffer_from_host_typed(
              devices[0].ordinal, q_input, sizeof(q_input),
              PJRTX_MLX_METAL_DTYPE_F32, q_dims, 3);
      PjrtxMlxMetalBuffer* k_buffer =
          pjrtx_mlx_metal_buffer_from_host_typed(
              devices[0].ordinal, k_input, sizeof(k_input),
              PJRTX_MLX_METAL_DTYPE_F32, kv_dims, 3);
      PjrtxMlxMetalBuffer* v_buffer =
          pjrtx_mlx_metal_buffer_from_host_typed(
              devices[0].ordinal, v_input, sizeof(v_input),
              PJRTX_MLX_METAL_DTYPE_F32, kv_dims, 3);
      PjrtxMlxMetalBuffer* token_buffer =
          pjrtx_mlx_metal_buffer_from_host_typed(
              devices[0].ordinal, &token_index, sizeof(token_index),
              PJRTX_MLX_METAL_DTYPE_U32, nullptr, 0);
      assert(q_buffer != nullptr);
      assert(k_buffer != nullptr);
      assert(v_buffer != nullptr);
      assert(token_buffer != nullptr);
      PjrtxMlxMetalBuffer* attention =
          pjrtx_mlx_metal_custom_call_scaled_dot_product_attention(
              q_buffer, k_buffer, v_buffer, token_buffer);
      assert(attention != nullptr);
      float attention_output[8] = {};
      assert(pjrtx_mlx_metal_buffer_copy_to_host(
                 attention, attention_output, sizeof(attention_output)) == 1);
      const float expected[] = {10.0f, 0.0f, 10.0f, 0.0f, 20.0f, 0.0f, 20.0f, 0.0f};
      for (int i = 0; i < 8; ++i) {
        assert(near_relaxed(attention_output[i], expected[i]));
      }
      pjrtx_mlx_metal_buffer_destroy(attention);
      pjrtx_mlx_metal_buffer_destroy(token_buffer);
      pjrtx_mlx_metal_buffer_destroy(v_buffer);
      pjrtx_mlx_metal_buffer_destroy(k_buffer);
      pjrtx_mlx_metal_buffer_destroy(q_buffer);
    }

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
    PjrtxMlxMetalBuffer* complex_rhs_buffer =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, f32_rhs, sizeof(f32_rhs),
            PJRTX_MLX_METAL_DTYPE_F32, f32_dims, 1);
    assert(complex_rhs_buffer != nullptr);
    PjrtxMlxMetalBuffer* complex_buffer = pjrtx_mlx_metal_buffer_complex(
        f32_buffer, complex_rhs_buffer, f32_dims, 1);
    assert(complex_buffer != nullptr);
    float complex_output[6] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               complex_buffer, complex_output, sizeof(complex_output)) == 1);
    for (int i = 0; i < 3; ++i) {
      assert(near(complex_output[2 * i], f32_input[i]));
      assert(near(complex_output[2 * i + 1], f32_rhs[i]));
    }
    PjrtxMlxMetalBuffer* real_part =
        pjrtx_mlx_metal_buffer_real(complex_buffer, f32_dims, 1);
    PjrtxMlxMetalBuffer* imag_part =
        pjrtx_mlx_metal_buffer_imag(complex_buffer, f32_dims, 1);
    assert(real_part != nullptr);
    assert(imag_part != nullptr);
    float real_output[3] = {};
    float imag_output[3] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               real_part, real_output, sizeof(real_output)) == 1);
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               imag_part, imag_output, sizeof(imag_output)) == 1);
    for (int i = 0; i < 3; ++i) {
      assert(near(real_output[i], f32_input[i]));
      assert(near(imag_output[i], f32_rhs[i]));
    }
    pjrtx_mlx_metal_buffer_destroy(imag_part);
    pjrtx_mlx_metal_buffer_destroy(real_part);
    pjrtx_mlx_metal_buffer_destroy(complex_buffer);
    pjrtx_mlx_metal_buffer_destroy(complex_rhs_buffer);
    PjrtxMlxMetalBuffer* f32_zero_like =
        pjrtx_mlx_metal_buffer_zero_like(f32_buffer);
    assert(f32_zero_like != nullptr);
    float f32_zero_like_output[3] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               f32_zero_like, f32_zero_like_output,
               sizeof(f32_zero_like_output)) == 1);
    for (int i = 0; i < 3; ++i) {
      assert(near(f32_zero_like_output[i], 0.0f));
    }
    pjrtx_mlx_metal_buffer_destroy(f32_zero_like);
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

    const int64_t reduce_window_rank1[] = {2};
    const int64_t reduce_window_stride1[] = {1};
    const int64_t reduce_window_base1[] = {1};
    const int64_t reduce_window_dilation1[] = {1};
    const int64_t reduce_window_low[] = {1};
    const int64_t reduce_window_high[] = {0};
    const int64_t reduce_window_sum_dims[] = {3};
    PjrtxMlxMetalBuffer* window_sum = pjrtx_mlx_metal_buffer_reduce_window(
        f32_buffer, PJRTX_MLX_METAL_REDUCE_SUM, reduce_window_rank1,
        reduce_window_stride1, reduce_window_base1, reduce_window_dilation1,
        reduce_window_low, reduce_window_high, 1, reduce_window_sum_dims, 1);
    assert(window_sum != nullptr);
    float window_sum_output[3] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               window_sum, window_sum_output, sizeof(window_sum_output)) == 1);
    {
      const float expected[] = {1.5f, -0.5f, 2.0f};
      for (int i = 0; i < 3; ++i) {
        assert(near(window_sum_output[i], expected[i]));
      }
    }
    pjrtx_mlx_metal_buffer_destroy(window_sum);

    const int64_t reduce_window_no_pad[] = {0};
    const int64_t reduce_window_max_dims[] = {2};
    PjrtxMlxMetalBuffer* window_max = pjrtx_mlx_metal_buffer_reduce_window(
        f32_buffer, PJRTX_MLX_METAL_REDUCE_MAX, reduce_window_rank1,
        reduce_window_stride1, reduce_window_base1, reduce_window_dilation1,
        reduce_window_no_pad, reduce_window_no_pad, 1, reduce_window_max_dims,
        1);
    assert(window_max != nullptr);
    float window_max_output[2] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               window_max, window_max_output, sizeof(window_max_output)) == 1);
    assert(near(window_max_output[0], 1.5f));
    assert(near(window_max_output[1], 4.0f));
    pjrtx_mlx_metal_buffer_destroy(window_max);

    const uint16_t f16_window_input[] = {0x3c00, 0x4000, 0x4200, 0x4400};
    const int64_t f16_window_dims[] = {4};
    PjrtxMlxMetalBuffer* f16_window_buffer =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, f16_window_input, sizeof(f16_window_input),
            PJRTX_MLX_METAL_DTYPE_F16, f16_window_dims, 1);
    assert(f16_window_buffer != nullptr);
    const int64_t f16_window_output_dims[] = {3};
    PjrtxMlxMetalBuffer* f16_window_sum =
        pjrtx_mlx_metal_buffer_reduce_window(
            f16_window_buffer, PJRTX_MLX_METAL_REDUCE_SUM,
            reduce_window_rank1, reduce_window_stride1,
            reduce_window_base1, reduce_window_dilation1,
            reduce_window_no_pad, reduce_window_no_pad, 1,
            f16_window_output_dims, 1);
    PjrtxMlxMetalBuffer* f16_window_max =
        pjrtx_mlx_metal_buffer_reduce_window(
            f16_window_buffer, PJRTX_MLX_METAL_REDUCE_MAX,
            reduce_window_rank1, reduce_window_stride1,
            reduce_window_base1, reduce_window_dilation1,
            reduce_window_no_pad, reduce_window_no_pad, 1,
            f16_window_output_dims, 1);
    assert(f16_window_sum != nullptr);
    assert(f16_window_max != nullptr);
    PjrtxMlxMetalBuffer* f16_window_sum_f32 =
        pjrtx_mlx_metal_buffer_astype(f16_window_sum,
                                      PJRTX_MLX_METAL_DTYPE_F32);
    PjrtxMlxMetalBuffer* f16_window_max_f32 =
        pjrtx_mlx_metal_buffer_astype(f16_window_max,
                                      PJRTX_MLX_METAL_DTYPE_F32);
    assert(f16_window_sum_f32 != nullptr);
    assert(f16_window_max_f32 != nullptr);
    float f16_window_sum_output[3] = {};
    float f16_window_max_output[3] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               f16_window_sum_f32, f16_window_sum_output,
               sizeof(f16_window_sum_output)) == 1);
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               f16_window_max_f32, f16_window_max_output,
               sizeof(f16_window_max_output)) == 1);
    const float f16_window_sum_expected[] = {3.0f, 5.0f, 7.0f};
    const float f16_window_max_expected[] = {2.0f, 3.0f, 4.0f};
    for (int i = 0; i < 3; ++i) {
      assert(near(f16_window_sum_output[i], f16_window_sum_expected[i]));
      assert(near(f16_window_max_output[i], f16_window_max_expected[i]));
    }
    pjrtx_mlx_metal_buffer_destroy(f16_window_max_f32);
    pjrtx_mlx_metal_buffer_destroy(f16_window_sum_f32);
    pjrtx_mlx_metal_buffer_destroy(f16_window_max);
    pjrtx_mlx_metal_buffer_destroy(f16_window_sum);
    pjrtx_mlx_metal_buffer_destroy(f16_window_buffer);

    const float pool_values_input[] = {1.0f, 3.0f, 2.0f, 4.0f, 6.0f, 5.0f};
    const int32_t pool_indices_input[] = {0, 1, 2, 0, 1, 2};
    const int64_t pool_dims[] = {2, 3};
    const int64_t pool_window[] = {1, 2};
    const int64_t pool_strides[] = {1, 1};
    const int64_t pool_dilation[] = {1, 1};
    const int64_t pool_padding[] = {0, 0};
    const int64_t pool_output_dims[] = {2, 2};
    PjrtxMlxMetalBuffer* pool_values =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, pool_values_input, sizeof(pool_values_input),
            PJRTX_MLX_METAL_DTYPE_F32, pool_dims, 2);
    PjrtxMlxMetalBuffer* pool_indices =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, pool_indices_input, sizeof(pool_indices_input),
            PJRTX_MLX_METAL_DTYPE_S32, pool_dims, 2);
    assert(pool_values != nullptr);
    assert(pool_indices != nullptr);
    PjrtxMlxMetalBuffer* pooled_values = nullptr;
    PjrtxMlxMetalBuffer* pooled_indices = nullptr;
    assert(pjrtx_mlx_metal_buffer_reduce_window_max_with_indices(
               pool_values, pool_indices, pool_window, pool_strides,
               pool_dilation, pool_dilation, pool_padding, pool_padding, 2,
               pool_output_dims, 2, &pooled_values, &pooled_indices) == 1);
    assert(pooled_values != nullptr);
    assert(pooled_indices != nullptr);
    float pooled_values_output[4] = {};
    int32_t pooled_indices_output[4] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               pooled_values, pooled_values_output,
               sizeof(pooled_values_output)) == 1);
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               pooled_indices, pooled_indices_output,
               sizeof(pooled_indices_output)) == 1);
    const float expected_pool_values[] = {3.0f, 3.0f, 6.0f, 6.0f};
    const int32_t expected_pool_indices[] = {1, 1, 1, 1};
    for (int i = 0; i < 4; ++i) {
      assert(near(pooled_values_output[i], expected_pool_values[i]));
      assert(pooled_indices_output[i] == expected_pool_indices[i]);
    }
    pjrtx_mlx_metal_buffer_destroy(pooled_indices);
    pjrtx_mlx_metal_buffer_destroy(pooled_values);
    pjrtx_mlx_metal_buffer_destroy(pool_indices);
    pjrtx_mlx_metal_buffer_destroy(pool_values);

    const uint32_t u32_bits_input[] = {0x3f800000u, 0xc0000000u};
    const int64_t u32_bits_dims[] = {2};
    PjrtxMlxMetalBuffer* u32_bits =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, u32_bits_input, sizeof(u32_bits_input),
            PJRTX_MLX_METAL_DTYPE_U32, u32_bits_dims, 1);
    assert(u32_bits != nullptr);
    PjrtxMlxMetalBuffer* u32_bits_as_f32 =
        pjrtx_mlx_metal_buffer_view_dtype(
            u32_bits, PJRTX_MLX_METAL_DTYPE_F32, u32_bits_dims, 1);
    assert(u32_bits_as_f32 != nullptr);
    assert(pjrtx_mlx_metal_buffer_size(u32_bits_as_f32) ==
           sizeof(u32_bits_input));
    assert(pjrtx_mlx_metal_buffer_has_host_shadow(u32_bits_as_f32) == 0);
    float u32_bits_as_f32_output[2] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               u32_bits_as_f32, u32_bits_as_f32_output,
               sizeof(u32_bits_as_f32_output)) == 1);
    assert(near(u32_bits_as_f32_output[0], 1.0f));
    assert(near(u32_bits_as_f32_output[1], -2.0f));
    pjrtx_mlx_metal_buffer_destroy(u32_bits_as_f32);
    pjrtx_mlx_metal_buffer_destroy(u32_bits);

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

    {
      const int64_t mat_dims[] = {2, 2};
      const int64_t lhs_contract[] = {1};
      const int64_t rhs_contract[] = {0};
      const uint16_t lhs_f16[] = {0x3c00, 0x4000, 0x4200, 0x4400};
      const uint16_t rhs_f16[] = {0x4500, 0x4600, 0x4700, 0x4800};
      PjrtxMlxMetalBuffer* lhs =
          pjrtx_mlx_metal_buffer_from_host_typed(
              devices[0].ordinal, lhs_f16, sizeof(lhs_f16),
              PJRTX_MLX_METAL_DTYPE_F16, mat_dims, 2);
      PjrtxMlxMetalBuffer* rhs =
          pjrtx_mlx_metal_buffer_from_host_typed(
              devices[0].ordinal, rhs_f16, sizeof(rhs_f16),
              PJRTX_MLX_METAL_DTYPE_F16, mat_dims, 2);
      assert(lhs != nullptr);
      assert(rhs != nullptr);
      PjrtxMlxMetalBuffer* product =
          pjrtx_mlx_metal_buffer_dot_general(
              lhs, rhs, nullptr, 0, nullptr, 0, lhs_contract, 1,
              rhs_contract, 1, mat_dims, 2);
      assert(product != nullptr);
      assert(pjrtx_mlx_metal_buffer_size(product) == sizeof(lhs_f16));
      PjrtxMlxMetalBuffer* product_f32 =
          pjrtx_mlx_metal_buffer_astype(product, PJRTX_MLX_METAL_DTYPE_F32);
      assert(product_f32 != nullptr);
      float product_output[4] = {};
      assert(pjrtx_mlx_metal_buffer_copy_to_host(
                 product_f32, product_output, sizeof(product_output)) == 1);
      const float expected[] = {19.0f, 22.0f, 43.0f, 50.0f};
      for (int i = 0; i < 4; ++i) {
        assert(near(product_output[i], expected[i]));
      }
      pjrtx_mlx_metal_buffer_destroy(product_f32);
      pjrtx_mlx_metal_buffer_destroy(product);
      pjrtx_mlx_metal_buffer_destroy(rhs);
      pjrtx_mlx_metal_buffer_destroy(lhs);

      const uint16_t lhs_bf16[] = {0x3f80, 0x4000, 0x4040, 0x4080};
      const uint16_t rhs_bf16[] = {0x40a0, 0x40c0, 0x40e0, 0x4100};
      PjrtxMlxMetalBuffer* lhs_b =
          pjrtx_mlx_metal_buffer_from_host_typed(
              devices[0].ordinal, lhs_bf16, sizeof(lhs_bf16),
              PJRTX_MLX_METAL_DTYPE_BF16, mat_dims, 2);
      PjrtxMlxMetalBuffer* rhs_b =
          pjrtx_mlx_metal_buffer_from_host_typed(
              devices[0].ordinal, rhs_bf16, sizeof(rhs_bf16),
              PJRTX_MLX_METAL_DTYPE_BF16, mat_dims, 2);
      assert(lhs_b != nullptr);
      assert(rhs_b != nullptr);
      PjrtxMlxMetalBuffer* product_b =
          pjrtx_mlx_metal_buffer_dot_general(
              lhs_b, rhs_b, nullptr, 0, nullptr, 0, lhs_contract, 1,
              rhs_contract, 1, mat_dims, 2);
      assert(product_b != nullptr);
      assert(pjrtx_mlx_metal_buffer_size(product_b) == sizeof(lhs_bf16));
      PjrtxMlxMetalBuffer* product_b_f32 =
          pjrtx_mlx_metal_buffer_astype(product_b,
                                        PJRTX_MLX_METAL_DTYPE_F32);
      assert(product_b_f32 != nullptr);
      float product_b_output[4] = {};
      assert(pjrtx_mlx_metal_buffer_copy_to_host(
                 product_b_f32, product_b_output,
                 sizeof(product_b_output)) == 1);
      for (int i = 0; i < 4; ++i) {
        assert(near(product_b_output[i], expected[i]));
      }
      pjrtx_mlx_metal_buffer_destroy(product_b_f32);
      pjrtx_mlx_metal_buffer_destroy(product_b);
      pjrtx_mlx_metal_buffer_destroy(rhs_b);
      pjrtx_mlx_metal_buffer_destroy(lhs_b);
    }

    {
      const int64_t q_dims[] = {2, 3, 4, 5};
      const int64_t k_dims[] = {2, 6, 5};
      const int64_t lhs_batch[] = {0};
      const int64_t rhs_batch[] = {0};
      const int64_t lhs_contract[] = {3};
      const int64_t rhs_contract[] = {2};
      const int64_t attn_dims[] = {2, 3, 4, 6};
      float q_values[2 * 3 * 4 * 5] = {};
      float k_values[2 * 6 * 5] = {};
      for (int i = 0; i < 2 * 3 * 4 * 5; ++i) {
        q_values[i] = static_cast<float>((i % 11) - 5) / 7.0f;
      }
      for (int i = 0; i < 2 * 6 * 5; ++i) {
        k_values[i] = static_cast<float>((i % 13) - 6) / 9.0f;
      }
      PjrtxMlxMetalBuffer* q_buffer =
          pjrtx_mlx_metal_buffer_from_host_typed(
              devices[0].ordinal, q_values, sizeof(q_values),
              PJRTX_MLX_METAL_DTYPE_F32, q_dims, 4);
      PjrtxMlxMetalBuffer* k_buffer =
          pjrtx_mlx_metal_buffer_from_host_typed(
              devices[0].ordinal, k_values, sizeof(k_values),
              PJRTX_MLX_METAL_DTYPE_F32, k_dims, 3);
      assert(q_buffer != nullptr);
      assert(k_buffer != nullptr);
      PjrtxMlxMetalBuffer* attn_scores =
          pjrtx_mlx_metal_buffer_dot_general(
              q_buffer, k_buffer, lhs_batch, 1, rhs_batch, 1, lhs_contract, 1,
              rhs_contract, 1, attn_dims, 4);
      assert(attn_scores != nullptr);
      float attn_output[2 * 3 * 4 * 6] = {};
      assert(pjrtx_mlx_metal_buffer_copy_to_host(
                 attn_scores, attn_output, sizeof(attn_output)) == 1);
      for (int b = 0; b < 2; ++b) {
        for (int hq = 0; hq < 3; ++hq) {
          for (int q = 0; q < 4; ++q) {
            for (int s = 0; s < 6; ++s) {
              float expected = 0.0f;
              for (int d = 0; d < 5; ++d) {
                const int q_index = (((b * 3 + hq) * 4 + q) * 5 + d);
                const int k_index = ((b * 6 + s) * 5 + d);
                expected += q_values[q_index] * k_values[k_index];
              }
              const int out_index = (((b * 3 + hq) * 4 + q) * 6 + s);
              assert(near_relaxed(attn_output[out_index], expected));
            }
          }
        }
      }
      pjrtx_mlx_metal_buffer_destroy(attn_scores);
      pjrtx_mlx_metal_buffer_destroy(k_buffer);
      pjrtx_mlx_metal_buffer_destroy(q_buffer);
    }

    {
      const int64_t prob_dims[] = {2, 3, 4, 6};
      const int64_t v_dims[] = {2, 6, 5};
      const int64_t lhs_batch[] = {0};
      const int64_t rhs_batch[] = {0};
      const int64_t lhs_contract[] = {3};
      const int64_t rhs_contract[] = {1};
      const int64_t out_dims[] = {2, 3, 4, 5};
      float prob_values[2 * 3 * 4 * 6] = {};
      float v_values[2 * 6 * 5] = {};
      for (int i = 0; i < 2 * 3 * 4 * 6; ++i) {
        prob_values[i] = static_cast<float>((i % 17) + 1) / 17.0f;
      }
      for (int i = 0; i < 2 * 6 * 5; ++i) {
        v_values[i] = static_cast<float>((i % 19) - 4) / 13.0f;
      }
      PjrtxMlxMetalBuffer* prob_buffer =
          pjrtx_mlx_metal_buffer_from_host_typed(
              devices[0].ordinal, prob_values, sizeof(prob_values),
              PJRTX_MLX_METAL_DTYPE_F32, prob_dims, 4);
      PjrtxMlxMetalBuffer* v_buffer =
          pjrtx_mlx_metal_buffer_from_host_typed(
              devices[0].ordinal, v_values, sizeof(v_values),
              PJRTX_MLX_METAL_DTYPE_F32, v_dims, 3);
      assert(prob_buffer != nullptr);
      assert(v_buffer != nullptr);
      PjrtxMlxMetalBuffer* context =
          pjrtx_mlx_metal_buffer_dot_general(
              prob_buffer, v_buffer, lhs_batch, 1, rhs_batch, 1, lhs_contract,
              1, rhs_contract, 1, out_dims, 4);
      assert(context != nullptr);
      float context_output[2 * 3 * 4 * 5] = {};
      assert(pjrtx_mlx_metal_buffer_copy_to_host(
                 context, context_output, sizeof(context_output)) == 1);
      for (int b = 0; b < 2; ++b) {
        for (int hq = 0; hq < 3; ++hq) {
          for (int q = 0; q < 4; ++q) {
            for (int d = 0; d < 5; ++d) {
              float expected = 0.0f;
              for (int s = 0; s < 6; ++s) {
                const int prob_index = (((b * 3 + hq) * 4 + q) * 6 + s);
                const int v_index = ((b * 6 + s) * 5 + d);
                expected += prob_values[prob_index] * v_values[v_index];
              }
              const int out_index = (((b * 3 + hq) * 4 + q) * 5 + d);
              assert(near_relaxed(context_output[out_index], expected));
            }
          }
        }
      }
      pjrtx_mlx_metal_buffer_destroy(context);
      pjrtx_mlx_metal_buffer_destroy(v_buffer);
      pjrtx_mlx_metal_buffer_destroy(prob_buffer);
    }

    {
      const int64_t reduce_dims[] = {2, 2};
      const int64_t reduce_axes[] = {1};
      const int64_t reduce_output_dims[] = {2};
      const uint16_t f16_values[] = {0x3c00, 0x4000, 0x4200, 0x4400};
      const uint16_t bf16_values[] = {0x3f80, 0x4000, 0x4040, 0x4080};

      PjrtxMlxMetalBuffer* f16_reduce_buffer =
          pjrtx_mlx_metal_buffer_from_host_typed(
              devices[0].ordinal, f16_values, sizeof(f16_values),
              PJRTX_MLX_METAL_DTYPE_F16, reduce_dims, 2);
      assert(f16_reduce_buffer != nullptr);
      PjrtxMlxMetalBuffer* f16_reduce_sum = pjrtx_mlx_metal_buffer_reduce(
          f16_reduce_buffer, PJRTX_MLX_METAL_REDUCE_SUM, reduce_axes, 1,
          reduce_output_dims, 1);
      PjrtxMlxMetalBuffer* f16_reduce_max = pjrtx_mlx_metal_buffer_reduce(
          f16_reduce_buffer, PJRTX_MLX_METAL_REDUCE_MAX, reduce_axes, 1,
          reduce_output_dims, 1);
      PjrtxMlxMetalBuffer* f16_reduce_min = pjrtx_mlx_metal_buffer_reduce(
          f16_reduce_buffer, PJRTX_MLX_METAL_REDUCE_MIN, reduce_axes, 1,
          reduce_output_dims, 1);
      assert(f16_reduce_sum != nullptr);
      assert(f16_reduce_max != nullptr);
      assert(f16_reduce_min != nullptr);
      assert(pjrtx_mlx_metal_buffer_size(f16_reduce_sum) ==
             sizeof(uint16_t) * 2);
      PjrtxMlxMetalBuffer* f16_reduce_sum_f32 =
          pjrtx_mlx_metal_buffer_astype(f16_reduce_sum,
                                        PJRTX_MLX_METAL_DTYPE_F32);
      PjrtxMlxMetalBuffer* f16_reduce_max_f32 =
          pjrtx_mlx_metal_buffer_astype(f16_reduce_max,
                                        PJRTX_MLX_METAL_DTYPE_F32);
      PjrtxMlxMetalBuffer* f16_reduce_min_f32 =
          pjrtx_mlx_metal_buffer_astype(f16_reduce_min,
                                        PJRTX_MLX_METAL_DTYPE_F32);
      assert(f16_reduce_sum_f32 != nullptr);
      assert(f16_reduce_max_f32 != nullptr);
      assert(f16_reduce_min_f32 != nullptr);
      float f16_reduce_sum_output[2] = {};
      float f16_reduce_max_output[2] = {};
      float f16_reduce_min_output[2] = {};
      assert(pjrtx_mlx_metal_buffer_copy_to_host(
                 f16_reduce_sum_f32, f16_reduce_sum_output,
                 sizeof(f16_reduce_sum_output)) == 1);
      assert(pjrtx_mlx_metal_buffer_copy_to_host(
                 f16_reduce_max_f32, f16_reduce_max_output,
                 sizeof(f16_reduce_max_output)) == 1);
      assert(pjrtx_mlx_metal_buffer_copy_to_host(
                 f16_reduce_min_f32, f16_reduce_min_output,
                 sizeof(f16_reduce_min_output)) == 1);
      assert(near(f16_reduce_sum_output[0], 3.0f));
      assert(near(f16_reduce_sum_output[1], 7.0f));
      assert(near(f16_reduce_max_output[0], 2.0f));
      assert(near(f16_reduce_max_output[1], 4.0f));
      assert(near(f16_reduce_min_output[0], 1.0f));
      assert(near(f16_reduce_min_output[1], 3.0f));
      pjrtx_mlx_metal_buffer_destroy(f16_reduce_min_f32);
      pjrtx_mlx_metal_buffer_destroy(f16_reduce_max_f32);
      pjrtx_mlx_metal_buffer_destroy(f16_reduce_sum_f32);
      pjrtx_mlx_metal_buffer_destroy(f16_reduce_min);
      pjrtx_mlx_metal_buffer_destroy(f16_reduce_max);
      pjrtx_mlx_metal_buffer_destroy(f16_reduce_sum);
      pjrtx_mlx_metal_buffer_destroy(f16_reduce_buffer);

      PjrtxMlxMetalBuffer* bf16_reduce_buffer =
          pjrtx_mlx_metal_buffer_from_host_typed(
              devices[0].ordinal, bf16_values, sizeof(bf16_values),
              PJRTX_MLX_METAL_DTYPE_BF16, reduce_dims, 2);
      assert(bf16_reduce_buffer != nullptr);
      PjrtxMlxMetalBuffer* bf16_reduce_sum = pjrtx_mlx_metal_buffer_reduce(
          bf16_reduce_buffer, PJRTX_MLX_METAL_REDUCE_SUM, reduce_axes, 1,
          reduce_output_dims, 1);
      assert(bf16_reduce_sum != nullptr);
      assert(pjrtx_mlx_metal_buffer_size(bf16_reduce_sum) ==
             sizeof(uint16_t) * 2);
      PjrtxMlxMetalBuffer* bf16_reduce_sum_f32 =
          pjrtx_mlx_metal_buffer_astype(bf16_reduce_sum,
                                        PJRTX_MLX_METAL_DTYPE_F32);
      assert(bf16_reduce_sum_f32 != nullptr);
      float bf16_reduce_sum_output[2] = {};
      assert(pjrtx_mlx_metal_buffer_copy_to_host(
                 bf16_reduce_sum_f32, bf16_reduce_sum_output,
                 sizeof(bf16_reduce_sum_output)) == 1);
      assert(near(bf16_reduce_sum_output[0], 3.0f));
      assert(near(bf16_reduce_sum_output[1], 7.0f));
      pjrtx_mlx_metal_buffer_destroy(bf16_reduce_sum_f32);
      pjrtx_mlx_metal_buffer_destroy(bf16_reduce_sum);
      pjrtx_mlx_metal_buffer_destroy(bf16_reduce_buffer);
    }

    {
      const int64_t argmax_dims[] = {2, 4};
      const int64_t argmax_axes[] = {1};
      const int64_t argmax_output_dims[] = {2};
      const float argmax_values[] = {1.0f, 7.0f, 7.0f, 3.0f,
                                     -2.0f, 5.0f, 4.0f, 5.0f};
      const int32_t argmax_indices[] = {0, 1, 2, 3, 0, 1, 2, 3};
      PjrtxMlxMetalBuffer* values_buffer =
          pjrtx_mlx_metal_buffer_from_host_typed(
              devices[0].ordinal, argmax_values, sizeof(argmax_values),
              PJRTX_MLX_METAL_DTYPE_F32, argmax_dims, 2);
      PjrtxMlxMetalBuffer* indices_buffer =
          pjrtx_mlx_metal_buffer_from_host_typed(
              devices[0].ordinal, argmax_indices, sizeof(argmax_indices),
              PJRTX_MLX_METAL_DTYPE_S32, argmax_dims, 2);
      assert(values_buffer != nullptr);
      assert(indices_buffer != nullptr);
      PjrtxMlxMetalBuffer* argmax_out_values = nullptr;
      PjrtxMlxMetalBuffer* argmax_out_indices = nullptr;
      assert(pjrtx_mlx_metal_buffer_reduce_max_with_indices(
                 values_buffer, indices_buffer, argmax_axes, 1,
                 argmax_output_dims, 1, &argmax_out_values,
                 &argmax_out_indices) == 1);
      assert(argmax_out_values != nullptr);
      assert(argmax_out_indices != nullptr);
      float out_values[2] = {};
      int32_t out_indices[2] = {};
      assert(pjrtx_mlx_metal_buffer_copy_to_host(
                 argmax_out_values, out_values, sizeof(out_values)) == 1);
      assert(pjrtx_mlx_metal_buffer_copy_to_host(
                 argmax_out_indices, out_indices, sizeof(out_indices)) == 1);
      assert(near(out_values[0], 7.0f));
      assert(near(out_values[1], 5.0f));
      assert(out_indices[0] == 1);
      assert(out_indices[1] == 1);
      pjrtx_mlx_metal_buffer_destroy(argmax_out_indices);
      pjrtx_mlx_metal_buffer_destroy(argmax_out_values);
      pjrtx_mlx_metal_buffer_destroy(indices_buffer);
      pjrtx_mlx_metal_buffer_destroy(values_buffer);
    }

    {
      const int64_t state_dims[] = {2};
      const int64_t random_dims[] = {2, 4};
      const uint64_t seed_state[] = {0x123456789abcdef0ULL,
                                     0x0fedcba987654321ULL};
      PjrtxMlxMetalBuffer* state_a =
          pjrtx_mlx_metal_buffer_from_host_typed(
              devices[0].ordinal, seed_state, sizeof(seed_state),
              PJRTX_MLX_METAL_DTYPE_U64, state_dims, 1);
      PjrtxMlxMetalBuffer* state_b =
          pjrtx_mlx_metal_buffer_from_host_typed(
              devices[0].ordinal, seed_state, sizeof(seed_state),
              PJRTX_MLX_METAL_DTYPE_U64, state_dims, 1);
      assert(state_a != nullptr);
      assert(state_b != nullptr);
      PjrtxMlxMetalBuffer* next_state_a = nullptr;
      PjrtxMlxMetalBuffer* bits_a = nullptr;
      PjrtxMlxMetalBuffer* next_state_b = nullptr;
      PjrtxMlxMetalBuffer* bits_b = nullptr;
      assert(pjrtx_mlx_metal_buffer_rng_bit_generator(
                 state_a, PJRTX_MLX_METAL_DTYPE_U32, random_dims, 2,
                 &next_state_a, &bits_a) == 1);
      assert(pjrtx_mlx_metal_buffer_rng_bit_generator(
                 state_b, PJRTX_MLX_METAL_DTYPE_U32, random_dims, 2,
                 &next_state_b, &bits_b) == 1);
      assert(next_state_a != nullptr);
      assert(bits_a != nullptr);
      assert(next_state_b != nullptr);
      assert(bits_b != nullptr);
      uint64_t new_state_a[2] = {};
      uint64_t new_state_b[2] = {};
      uint32_t random_a[8] = {};
      uint32_t random_b[8] = {};
      assert(pjrtx_mlx_metal_buffer_copy_to_host(
                 next_state_a, new_state_a, sizeof(new_state_a)) == 1);
      assert(pjrtx_mlx_metal_buffer_copy_to_host(
                 next_state_b, new_state_b, sizeof(new_state_b)) == 1);
      assert(pjrtx_mlx_metal_buffer_copy_to_host(
                 bits_a, random_a, sizeof(random_a)) == 1);
      assert(pjrtx_mlx_metal_buffer_copy_to_host(
                 bits_b, random_b, sizeof(random_b)) == 1);
      assert(new_state_a[0] == new_state_b[0]);
      assert(new_state_a[1] == new_state_b[1]);
      assert(new_state_a[0] == seed_state[0]);
      assert(new_state_a[1] == seed_state[1] + 4);
      const uint32_t expected_random[] = {
          0x1324f11aU, 0xf5ee7bf6U, 0xac0a940eU, 0x9145f7a4U,
          0x2ae1effeU, 0x5af1ca39U, 0x2594e3dfU, 0x326d7bf1U};
      for (int i = 0; i < 8; ++i) {
        assert(random_a[i] == random_b[i]);
        assert(random_a[i] == expected_random[i]);
      }
      pjrtx_mlx_metal_buffer_destroy(bits_b);
      pjrtx_mlx_metal_buffer_destroy(next_state_b);
      pjrtx_mlx_metal_buffer_destroy(bits_a);
      pjrtx_mlx_metal_buffer_destroy(next_state_a);
      pjrtx_mlx_metal_buffer_destroy(state_b);
      pjrtx_mlx_metal_buffer_destroy(state_a);
    }

    {
      const int64_t state_dims[] = {2};
      const int64_t random_dims[] = {2, 4};
      const uint64_t seed_state[] = {0x123456789abcdef0ULL,
                                     0x0fedcba987654321ULL};
      PjrtxMlxMetalBuffer* state =
          pjrtx_mlx_metal_buffer_from_host_typed(
              devices[0].ordinal, seed_state, sizeof(seed_state),
              PJRTX_MLX_METAL_DTYPE_U64, state_dims, 1);
      assert(state != nullptr);
      PjrtxMlxMetalBuffer* next_state = nullptr;
      PjrtxMlxMetalBuffer* bits = nullptr;
      assert(pjrtx_mlx_metal_buffer_rng_bit_generator(
                 state, PJRTX_MLX_METAL_DTYPE_U64, random_dims, 2,
                 &next_state, &bits) == 1);
      assert(next_state != nullptr);
      assert(bits != nullptr);
      uint64_t new_state[2] = {};
      uint64_t random[8] = {};
      assert(pjrtx_mlx_metal_buffer_copy_to_host(
                 next_state, new_state, sizeof(new_state)) == 1);
      assert(pjrtx_mlx_metal_buffer_copy_to_host(
                 bits, random, sizeof(random)) == 1);
      assert(new_state[0] == seed_state[0]);
      assert(new_state[1] == seed_state[1] + 8);
      const uint64_t expected_random[] = {
          0x2ae1effe1324f11aULL, 0x5af1ca39f5ee7bf6ULL,
          0x2594e3dfac0a940eULL, 0x326d7bf19145f7a4ULL,
          0x03866c09756254aeULL, 0x8b8081ab0c7ae7aaULL,
          0x25a131ca49b04f96ULL, 0xbf5aa9785c98b5bdULL};
      for (int i = 0; i < 8; ++i) {
        assert(random[i] == expected_random[i]);
      }
      pjrtx_mlx_metal_buffer_destroy(bits);
      pjrtx_mlx_metal_buffer_destroy(next_state);
      pjrtx_mlx_metal_buffer_destroy(state);
    }

    PjrtxMlxMetalBuffer* f32_rhs_buffer =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, f32_rhs, sizeof(f32_rhs),
            PJRTX_MLX_METAL_DTYPE_F32, f32_dims, 1);
    assert(f32_rhs_buffer != nullptr);
    {
      const int64_t rng_dims[] = {8};
      const float rng_mean = 0.0f;
      const float rng_stddev = 1.0f;
      PjrtxMlxMetalBuffer* rng_mean_buffer =
          pjrtx_mlx_metal_buffer_from_host_typed(
              devices[0].ordinal, &rng_mean, sizeof(rng_mean),
              PJRTX_MLX_METAL_DTYPE_F32, nullptr, 0);
      PjrtxMlxMetalBuffer* rng_stddev_buffer =
          pjrtx_mlx_metal_buffer_from_host_typed(
              devices[0].ordinal, &rng_stddev, sizeof(rng_stddev),
              PJRTX_MLX_METAL_DTYPE_F32, nullptr, 0);
      assert(rng_mean_buffer != nullptr);
      assert(rng_stddev_buffer != nullptr);
      PjrtxMlxMetalBuffer* rng_normal = pjrtx_mlx_metal_buffer_rng(
          rng_mean_buffer, rng_stddev_buffer, PJRTX_MLX_METAL_RNG_NORMAL,
          PJRTX_MLX_METAL_DTYPE_F32, rng_dims, 1);
      assert(rng_normal != nullptr);
      assert(pjrtx_mlx_metal_buffer_has_host_shadow(rng_normal) == 0);
      float rng_output[8] = {};
      assert(pjrtx_mlx_metal_buffer_copy_to_host(
                 rng_normal, rng_output, sizeof(rng_output)) == 1);
      for (int i = 0; i < 8; ++i) {
        assert(std::isfinite(rng_output[i]));
      }
      pjrtx_mlx_metal_buffer_destroy(rng_normal);
      pjrtx_mlx_metal_buffer_destroy(rng_stddev_buffer);
      pjrtx_mlx_metal_buffer_destroy(rng_mean_buffer);
    }
    PjrtxMlxMetalBuffer* f32_sum = pjrtx_mlx_metal_buffer_binary(
        f32_buffer, f32_rhs_buffer, PJRTX_MLX_METAL_U8_BINARY_ADD);
    if (f32_sum != nullptr) {
      assert(pjrtx_mlx_metal_buffer_eval(f32_sum) == 1);
      PjrtxMlxMetalBuffer* eval_many_buffers[] = {f32_sum};
      assert(pjrtx_mlx_metal_buffer_eval_many(eval_many_buffers, 1) == 1);
      assert(pjrtx_mlx_metal_buffer_has_host_shadow(f32_sum) == 0);
      float f32_sum_output[3] = {};
      assert(pjrtx_mlx_metal_buffer_copy_to_host(
                 f32_sum, f32_sum_output, sizeof(f32_sum_output)) == 1);
      for (int i = 0; i < 3; ++i) {
        assert(near(f32_sum_output[i], f32_input[i] + f32_rhs[i]));
      }
      pjrtx_mlx_metal_buffer_destroy(f32_sum);
    }
    PjrtxMlxMetalBuffer* f32_custom_sum =
        pjrtx_mlx_metal_custom_call_binary_add_f32(f32_buffer, f32_rhs_buffer);
    assert(f32_custom_sum != nullptr);
    float f32_custom_sum_output[3] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               f32_custom_sum, f32_custom_sum_output,
               sizeof(f32_custom_sum_output)) == 1);
    for (int i = 0; i < 3; ++i) {
      assert(near(f32_custom_sum_output[i], f32_input[i] + f32_rhs[i]));
    }
    pjrtx_mlx_metal_buffer_destroy(f32_custom_sum);
    {
      const int64_t while_dims[] = {4};
      const float while_state_input[] = {0.0f, 2.0f, 5.0f, 9.0f};
      const float while_limit_input[] = {4.0f, 4.0f, 7.0f, 10.0f};
      const float while_step_input = 1.0f;
      PjrtxMlxMetalBuffer* while_state =
          pjrtx_mlx_metal_buffer_from_host_typed(
              devices[0].ordinal, while_state_input, sizeof(while_state_input),
              PJRTX_MLX_METAL_DTYPE_F32, while_dims, 1);
      PjrtxMlxMetalBuffer* while_limit =
          pjrtx_mlx_metal_buffer_from_host_typed(
              devices[0].ordinal, while_limit_input, sizeof(while_limit_input),
              PJRTX_MLX_METAL_DTYPE_F32, while_dims, 1);
      PjrtxMlxMetalBuffer* while_step =
          pjrtx_mlx_metal_buffer_from_host_typed(
              devices[0].ordinal, &while_step_input, sizeof(while_step_input),
              PJRTX_MLX_METAL_DTYPE_F32, nullptr, 0);
      assert(while_state != nullptr);
      assert(while_limit != nullptr);
      assert(while_step != nullptr);
      PjrtxMlxMetalBuffer* while_output =
          pjrtx_mlx_metal_buffer_while_f32_lt_add(
              while_state, while_limit, while_step, while_dims, 1, 16);
      assert(while_output != nullptr);
      assert(pjrtx_mlx_metal_buffer_has_host_shadow(while_output) == 0);
      float while_result[4] = {};
      assert(pjrtx_mlx_metal_buffer_copy_to_host(
                 while_output, while_result, sizeof(while_result)) == 1);
      const float expected_while[] = {4.0f, 4.0f, 7.0f, 10.0f};
      for (int i = 0; i < 4; ++i) {
        assert(near(while_result[i], expected_while[i]));
      }
      pjrtx_mlx_metal_buffer_destroy(while_output);
      pjrtx_mlx_metal_buffer_destroy(while_step);
      pjrtx_mlx_metal_buffer_destroy(while_limit);
      pjrtx_mlx_metal_buffer_destroy(while_state);

      const float scalar_state_input = 0.0f;
      const float scalar_limit_input = 3.0f;
      PjrtxMlxMetalBuffer* scalar_state =
          pjrtx_mlx_metal_buffer_from_host_typed(
              devices[0].ordinal, &scalar_state_input, sizeof(scalar_state_input),
              PJRTX_MLX_METAL_DTYPE_F32, nullptr, 0);
      PjrtxMlxMetalBuffer* scalar_limit =
          pjrtx_mlx_metal_buffer_from_host_typed(
              devices[0].ordinal, &scalar_limit_input, sizeof(scalar_limit_input),
              PJRTX_MLX_METAL_DTYPE_F32, nullptr, 0);
      PjrtxMlxMetalBuffer* scalar_step =
          pjrtx_mlx_metal_buffer_from_host_typed(
              devices[0].ordinal, &while_step_input, sizeof(while_step_input),
              PJRTX_MLX_METAL_DTYPE_F32, nullptr, 0);
      assert(scalar_state != nullptr);
      assert(scalar_limit != nullptr);
      assert(scalar_step != nullptr);
      PjrtxMlxMetalBuffer* scalar_output =
          pjrtx_mlx_metal_buffer_while_f32_lt_add(
              scalar_state, scalar_limit, scalar_step, nullptr, 0, 8);
      assert(scalar_output != nullptr);
      float scalar_result = 0.0f;
      assert(pjrtx_mlx_metal_buffer_copy_to_host(
                 scalar_output, &scalar_result, sizeof(scalar_result)) == 1);
      assert(near(scalar_result, 3.0f));
      pjrtx_mlx_metal_buffer_destroy(scalar_output);
      pjrtx_mlx_metal_buffer_destroy(scalar_step);
      pjrtx_mlx_metal_buffer_destroy(scalar_limit);
      pjrtx_mlx_metal_buffer_destroy(scalar_state);

      const float down_state_input = 5.0f;
      const float down_limit_input = 0.0f;
      PjrtxMlxMetalBuffer* down_state =
          pjrtx_mlx_metal_buffer_from_host_typed(
              devices[0].ordinal, &down_state_input, sizeof(down_state_input),
              PJRTX_MLX_METAL_DTYPE_F32, nullptr, 0);
      PjrtxMlxMetalBuffer* down_limit =
          pjrtx_mlx_metal_buffer_from_host_typed(
              devices[0].ordinal, &down_limit_input, sizeof(down_limit_input),
              PJRTX_MLX_METAL_DTYPE_F32, nullptr, 0);
      PjrtxMlxMetalBuffer* down_step =
          pjrtx_mlx_metal_buffer_from_host_typed(
              devices[0].ordinal, &while_step_input, sizeof(while_step_input),
              PJRTX_MLX_METAL_DTYPE_F32, nullptr, 0);
      assert(down_state != nullptr);
      assert(down_limit != nullptr);
      assert(down_step != nullptr);
      PjrtxMlxMetalBuffer* down_output =
          pjrtx_mlx_metal_buffer_while_f32_compare_add(
              down_state, down_limit, down_step, PJRTX_MLX_METAL_COMPARE_GT,
              PJRTX_MLX_METAL_U8_BINARY_SUBTRACT, nullptr, 0, 8);
      assert(down_output != nullptr);
      float down_result = -1.0f;
      assert(pjrtx_mlx_metal_buffer_copy_to_host(
                 down_output, &down_result, sizeof(down_result)) == 1);
      assert(near(down_result, 0.0f));
      pjrtx_mlx_metal_buffer_destroy(down_output);
      pjrtx_mlx_metal_buffer_destroy(down_step);
      pjrtx_mlx_metal_buffer_destroy(down_limit);
      pjrtx_mlx_metal_buffer_destroy(down_state);

      const uint16_t bf16_zero = 0x0000;
      const uint16_t bf16_one = 0x3f80;
      const uint16_t bf16_quarter = 0x3e80;
      PjrtxMlxMetalBuffer* bf16_state =
          pjrtx_mlx_metal_buffer_from_host_typed(
              devices[0].ordinal, &bf16_zero, sizeof(bf16_zero),
              PJRTX_MLX_METAL_DTYPE_BF16, nullptr, 0);
      PjrtxMlxMetalBuffer* bf16_limit =
          pjrtx_mlx_metal_buffer_from_host_typed(
              devices[0].ordinal, &bf16_one, sizeof(bf16_one),
              PJRTX_MLX_METAL_DTYPE_BF16, nullptr, 0);
      PjrtxMlxMetalBuffer* bf16_step =
          pjrtx_mlx_metal_buffer_from_host_typed(
              devices[0].ordinal, &bf16_quarter, sizeof(bf16_quarter),
              PJRTX_MLX_METAL_DTYPE_BF16, nullptr, 0);
      assert(bf16_state != nullptr);
      assert(bf16_limit != nullptr);
      assert(bf16_step != nullptr);
      PjrtxMlxMetalBuffer* bf16_output =
          pjrtx_mlx_metal_buffer_while_f32_compare_add(
              bf16_state, bf16_limit, bf16_step, PJRTX_MLX_METAL_COMPARE_LT,
              PJRTX_MLX_METAL_U8_BINARY_ADD, nullptr, 0, 8);
      assert(bf16_output != nullptr);
      uint16_t bf16_result = 0;
      assert(pjrtx_mlx_metal_buffer_copy_to_host(
                 bf16_output, &bf16_result, sizeof(bf16_result)) == 1);
      assert(bf16_result == bf16_one);
      pjrtx_mlx_metal_buffer_destroy(bf16_output);
      pjrtx_mlx_metal_buffer_destroy(bf16_step);
      pjrtx_mlx_metal_buffer_destroy(bf16_limit);
      pjrtx_mlx_metal_buffer_destroy(bf16_state);
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
    PjrtxMlxMetalBuffer* f32_cbrt = pjrtx_mlx_metal_buffer_unary(
        f32_buffer, PJRTX_MLX_METAL_UNARY_CBRT);
    assert(f32_cbrt != nullptr);
    float f32_cbrt_output[3] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               f32_cbrt, f32_cbrt_output, sizeof(f32_cbrt_output)) == 1);
    for (int i = 0; i < 3; ++i) {
      assert(near_relaxed(f32_cbrt_output[i], std::cbrt(f32_input[i])));
    }
    pjrtx_mlx_metal_buffer_destroy(f32_cbrt);
    PjrtxMlxMetalBuffer* f32_round_afz = pjrtx_mlx_metal_buffer_unary(
        f32_buffer, PJRTX_MLX_METAL_UNARY_ROUND_AFZ);
    assert(f32_round_afz != nullptr);
    float f32_round_afz_output[3] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               f32_round_afz, f32_round_afz_output,
               sizeof(f32_round_afz_output)) == 1);
    const float expected_round_afz[] = {2.0f, -2.0f, 4.0f};
    for (int i = 0; i < 3; ++i) {
      assert(near(f32_round_afz_output[i], expected_round_afz[i]));
    }
    pjrtx_mlx_metal_buffer_destroy(f32_round_afz);
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
      assert(setenv("PJRTX_MLX_COMPILE_EVAL", "1", 1) == 0);
      PjrtxMlxMetalBuffer* compile_eval_buffers[] = {sliced};
      assert(pjrtx_mlx_metal_buffer_eval_many(compile_eval_buffers, 1) == 1);
      unsetenv("PJRTX_MLX_COMPILE_EVAL");
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

    const int64_t conv_input_dims[] = {1, 1, 4};
    const int64_t conv_kernel_dims[] = {1, 1, 2};
    const int64_t conv_output_dims[] = {1, 1, 3};
    const float conv_input[] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float conv_kernel[] = {1.0f, 2.0f};
    PjrtxMlxMetalBuffer* conv_lhs =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, conv_input, sizeof(conv_input),
            PJRTX_MLX_METAL_DTYPE_F32, conv_input_dims, 3);
    assert(conv_lhs != nullptr);
    PjrtxMlxMetalBuffer* conv_rhs =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, conv_kernel, sizeof(conv_kernel),
            PJRTX_MLX_METAL_DTYPE_F32, conv_kernel_dims, 3);
    assert(conv_rhs != nullptr);
    const int64_t conv_stride[] = {1};
    const int64_t conv_pad[] = {0};
    const int64_t conv_dilation[] = {1};
    const uint8_t conv_reversal[] = {0};
    PjrtxMlxMetalBuffer* convolved = pjrtx_mlx_metal_buffer_convolution(
        conv_lhs, conv_rhs, conv_stride, conv_pad, conv_pad, conv_dilation,
        conv_dilation, conv_reversal, 1, 1, conv_output_dims, 3);
    assert(convolved != nullptr);
    float conv_output[3] = {};
    assert(pjrtx_mlx_metal_buffer_size(convolved) == sizeof(conv_output));
    assert(pjrtx_mlx_metal_buffer_eval(convolved) == 1);
    assert(pjrtx_mlx_metal_buffer_copy_to_host(convolved, conv_output,
                                               sizeof(conv_output)) == 1);
    const float conv_expected[] = {5.0f, 8.0f, 11.0f};
    for (int i = 0; i < 3; ++i) {
      assert(near(conv_output[i], conv_expected[i]));
    }
    pjrtx_mlx_metal_buffer_destroy(convolved);
    pjrtx_mlx_metal_buffer_destroy(conv_rhs);
    pjrtx_mlx_metal_buffer_destroy(conv_lhs);

    const int64_t conv2d_input_dims[] = {1, 1, 3, 3};
    const int64_t conv2d_kernel_dims[] = {1, 1, 2, 2};
    const int64_t conv2d_output_dims[] = {1, 1, 2, 2};
    const float conv2d_input[] = {1.0f, 2.0f, 3.0f,
                                  4.0f, 5.0f, 6.0f,
                                  7.0f, 8.0f, 9.0f};
    const float conv2d_kernel[] = {1.0f, 1.0f, 1.0f, 1.0f};
    PjrtxMlxMetalBuffer* conv2d_lhs =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, conv2d_input, sizeof(conv2d_input),
            PJRTX_MLX_METAL_DTYPE_F32, conv2d_input_dims, 4);
    assert(conv2d_lhs != nullptr);
    PjrtxMlxMetalBuffer* conv2d_rhs =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, conv2d_kernel, sizeof(conv2d_kernel),
            PJRTX_MLX_METAL_DTYPE_F32, conv2d_kernel_dims, 4);
    assert(conv2d_rhs != nullptr);
    const int64_t conv2d_stride[] = {1, 1};
    const int64_t conv2d_pad[] = {0, 0};
    const int64_t conv2d_dilation[] = {1, 1};
    const uint8_t conv2d_reversal[] = {0, 0};
    PjrtxMlxMetalBuffer* convolved2d = pjrtx_mlx_metal_buffer_convolution(
        conv2d_lhs, conv2d_rhs, conv2d_stride, conv2d_pad, conv2d_pad,
        conv2d_dilation, conv2d_dilation, conv2d_reversal, 2, 1,
        conv2d_output_dims, 4);
    assert(convolved2d != nullptr);
    float conv2d_output[4] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               convolved2d, conv2d_output, sizeof(conv2d_output)) == 1);
    const float conv2d_expected[] = {12.0f, 16.0f, 24.0f, 28.0f};
    for (int i = 0; i < 4; ++i) {
      assert(near(conv2d_output[i], conv2d_expected[i]));
    }
    pjrtx_mlx_metal_buffer_destroy(convolved2d);
    pjrtx_mlx_metal_buffer_destroy(conv2d_rhs);
    pjrtx_mlx_metal_buffer_destroy(conv2d_lhs);

    const int64_t grouped_conv_input_dims[] = {1, 2, 2, 2};
    const int64_t grouped_conv_kernel_dims[] = {2, 1, 1, 1};
    const int64_t grouped_conv_output_dims[] = {1, 2, 2, 2};
    const float grouped_conv_input[] = {1.0f, 2.0f, 3.0f, 4.0f,
                                        5.0f, 6.0f, 7.0f, 8.0f};
    const float grouped_conv_kernel[] = {10.0f, 100.0f};
    PjrtxMlxMetalBuffer* grouped_conv_lhs =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, grouped_conv_input, sizeof(grouped_conv_input),
            PJRTX_MLX_METAL_DTYPE_F32, grouped_conv_input_dims, 4);
    assert(grouped_conv_lhs != nullptr);
    PjrtxMlxMetalBuffer* grouped_conv_rhs =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, grouped_conv_kernel,
            sizeof(grouped_conv_kernel), PJRTX_MLX_METAL_DTYPE_F32,
            grouped_conv_kernel_dims, 4);
    assert(grouped_conv_rhs != nullptr);
    PjrtxMlxMetalBuffer* grouped_convolved =
        pjrtx_mlx_metal_buffer_convolution(
            grouped_conv_lhs, grouped_conv_rhs, conv2d_stride, conv2d_pad,
            conv2d_pad, conv2d_dilation, conv2d_dilation, conv2d_reversal, 2,
            2, grouped_conv_output_dims, 4);
    assert(grouped_convolved != nullptr);
    float grouped_conv_output[8] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               grouped_convolved, grouped_conv_output,
               sizeof(grouped_conv_output)) == 1);
    const float grouped_conv_expected[] = {10.0f,  20.0f,  30.0f,  40.0f,
                                           500.0f, 600.0f, 700.0f, 800.0f};
    for (int i = 0; i < 8; ++i) {
      assert(near(grouped_conv_output[i], grouped_conv_expected[i]));
    }
    pjrtx_mlx_metal_buffer_destroy(grouped_convolved);
    pjrtx_mlx_metal_buffer_destroy(grouped_conv_rhs);
    pjrtx_mlx_metal_buffer_destroy(grouped_conv_lhs);

    const int64_t conv3d_input_dims[] = {1, 1, 3, 3, 3};
    const int64_t conv3d_kernel_dims[] = {1, 1, 2, 2, 2};
    const int64_t conv3d_output_dims[] = {1, 1, 2, 2, 2};
    const float conv3d_input[] = {
        1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,
        10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f,
        19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f, 25.0f, 26.0f, 27.0f,
    };
    const float conv3d_kernel[] = {1.0f, 1.0f, 1.0f, 1.0f,
                                   1.0f, 1.0f, 1.0f, 1.0f};
    PjrtxMlxMetalBuffer* conv3d_lhs =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, conv3d_input, sizeof(conv3d_input),
            PJRTX_MLX_METAL_DTYPE_F32, conv3d_input_dims, 5);
    assert(conv3d_lhs != nullptr);
    PjrtxMlxMetalBuffer* conv3d_rhs =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, conv3d_kernel, sizeof(conv3d_kernel),
            PJRTX_MLX_METAL_DTYPE_F32, conv3d_kernel_dims, 5);
    assert(conv3d_rhs != nullptr);
    const int64_t conv3d_stride[] = {1, 1, 1};
    const int64_t conv3d_pad[] = {0, 0, 0};
    const int64_t conv3d_dilation[] = {1, 1, 1};
    const uint8_t conv3d_reversal[] = {0, 0, 0};
    PjrtxMlxMetalBuffer* convolved3d = pjrtx_mlx_metal_buffer_convolution(
        conv3d_lhs, conv3d_rhs, conv3d_stride, conv3d_pad, conv3d_pad,
        conv3d_dilation, conv3d_dilation, conv3d_reversal, 3, 1,
        conv3d_output_dims, 5);
    assert(convolved3d != nullptr);
    float conv3d_output[8] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               convolved3d, conv3d_output, sizeof(conv3d_output)) == 1);
    const float conv3d_expected[] = {60.0f,  68.0f,  84.0f,  92.0f,
                                     132.0f, 140.0f, 156.0f, 164.0f};
    for (int i = 0; i < 8; ++i) {
      assert(near(conv3d_output[i], conv3d_expected[i]));
    }
    pjrtx_mlx_metal_buffer_destroy(convolved3d);
    pjrtx_mlx_metal_buffer_destroy(conv3d_rhs);
    pjrtx_mlx_metal_buffer_destroy(conv3d_lhs);

    const int64_t rfft_dims[] = {4};
    const int64_t rfft_output_dims[] = {3};
    const int64_t rfft_lengths[] = {4};
    const float rfft_input[] = {1.0f, 2.0f, 3.0f, 4.0f};
    PjrtxMlxMetalBuffer* rfft_buffer =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, rfft_input, sizeof(rfft_input),
            PJRTX_MLX_METAL_DTYPE_F32, rfft_dims, 1);
    assert(rfft_buffer != nullptr);
    PjrtxMlxMetalBuffer* rfft = pjrtx_mlx_metal_buffer_fft(
        rfft_buffer, PJRTX_MLX_METAL_RFFT, rfft_lengths, 1,
        rfft_output_dims, 1);
    assert(rfft != nullptr);
    float rfft_output[6] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               rfft, rfft_output, sizeof(rfft_output)) == 1);
    const float rfft_expected[] = {10.0f, 0.0f, -2.0f, 2.0f, -2.0f, 0.0f};
    for (int i = 0; i < 6; ++i) {
      assert(near_relaxed(rfft_output[i], rfft_expected[i]));
    }
    pjrtx_mlx_metal_buffer_destroy(rfft);
    pjrtx_mlx_metal_buffer_destroy(rfft_buffer);

    const int64_t chol_dims[] = {2, 2};
    const float chol_input[] = {4.0f, 2.0f, 2.0f, 3.0f};
    PjrtxMlxMetalBuffer* chol_buffer =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, chol_input, sizeof(chol_input),
            PJRTX_MLX_METAL_DTYPE_F32, chol_dims, 2);
    assert(chol_buffer != nullptr);
    PjrtxMlxMetalBuffer* chol =
        pjrtx_mlx_metal_buffer_cholesky(chol_buffer, 1, chol_dims, 2);
    assert(chol != nullptr);
    float chol_output[4] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               chol, chol_output, sizeof(chol_output)) == 1);
    const float chol_expected[] = {2.0f, 0.0f, 1.0f, 1.4142135f};
    for (int i = 0; i < 4; ++i) {
      assert(near_relaxed(chol_output[i], chol_expected[i]));
    }
    pjrtx_mlx_metal_buffer_destroy(chol);
    pjrtx_mlx_metal_buffer_destroy(chol_buffer);

    const float tri_a[] = {2.0f, 0.0f, 1.0f, 3.0f};
    const float tri_b[] = {2.0f, 4.0f, 7.0f, 13.0f};
    PjrtxMlxMetalBuffer* tri_a_buffer =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, tri_a, sizeof(tri_a),
            PJRTX_MLX_METAL_DTYPE_F32, chol_dims, 2);
    assert(tri_a_buffer != nullptr);
    PjrtxMlxMetalBuffer* tri_b_buffer =
        pjrtx_mlx_metal_buffer_from_host_typed(
            devices[0].ordinal, tri_b, sizeof(tri_b),
            PJRTX_MLX_METAL_DTYPE_F32, chol_dims, 2);
    assert(tri_b_buffer != nullptr);
    PjrtxMlxMetalBuffer* tri = pjrtx_mlx_metal_buffer_triangular_solve(
        tri_a_buffer, tri_b_buffer, 1, 1, 0, 0, chol_dims, 2);
    assert(tri != nullptr);
    float tri_output[4] = {};
    assert(pjrtx_mlx_metal_buffer_copy_to_host(
               tri, tri_output, sizeof(tri_output)) == 1);
    const float tri_expected[] = {1.0f, 2.0f, 2.0f, 3.6666667f};
    for (int i = 0; i < 4; ++i) {
      assert(near_relaxed(tri_output[i], tri_expected[i]));
    }
    pjrtx_mlx_metal_buffer_destroy(tri);
    pjrtx_mlx_metal_buffer_destroy(tri_b_buffer);
    pjrtx_mlx_metal_buffer_destroy(tri_a_buffer);

    pjrtx_mlx_metal_buffer_destroy(f32_rhs_buffer);
    pjrtx_mlx_metal_buffer_destroy(f32_buffer);
  }

  assert(pjrtx_mlx_metal_copy_devices(nullptr, 8) == 0);
  assert(pjrtx_mlx_metal_copy_devices(devices, 0) == 0);
  assert(pjrtx_mlx_metal_buffer_from_host(0, nullptr, 1) == nullptr);
  assert(pjrtx_mlx_metal_buffer_from_host(0, devices, 0) == nullptr);
  assert(pjrtx_mlx_metal_buffer_eval(nullptr) == 0);
  assert(pjrtx_mlx_metal_buffer_eval_many(nullptr, 1) == 0);
  assert(pjrtx_mlx_metal_buffer_clone(nullptr) == nullptr);
  assert(pjrtx_mlx_metal_buffer_zero_like(nullptr) == nullptr);
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
