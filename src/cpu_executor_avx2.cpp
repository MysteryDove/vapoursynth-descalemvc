#include <dsmvc/engine.hpp>

#include "axis_plan_internal.hpp"
#include "checked_size.hpp"
#include "cpu_packed.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace dsmvc {

namespace {

#if defined(_MSC_VER)
#define DSMVC_FORCE_INLINE __forceinline
#define DSMVC_FLATTEN
#else
#define DSMVC_FORCE_INLINE inline __attribute__((always_inline))
#define DSMVC_FLATTEN __attribute__((flatten))
#endif

struct alignas(32) ScratchVector {
    float lanes[8];
};

struct alignas(32) F64Quad {
    __m256d lanes;
};

struct F64Workspace {
    std::vector<F64Quad> source;
    std::vector<F64Quad> values;
    std::vector<double> scalar_source;
    std::vector<double> scalar_destination;
};

[[nodiscard]] F64Workspace &f64_workspace() {
    thread_local F64Workspace workspace;
    return workspace;
}

[[nodiscard]] DSMVC_FORCE_INLINE F64Quad f64_quad_zero() noexcept {
    return {_mm256_setzero_pd()};
}

[[nodiscard]] DSMVC_FORCE_INLINE F64Quad f64_quad_fma(
    F64Quad value, double coefficient, F64Quad source) noexcept {
    value.lanes = _mm256_add_pd(
        _mm256_mul_pd(_mm256_set1_pd(coefficient), source.lanes),
        value.lanes);
    return value;
}

[[nodiscard]] DSMVC_FORCE_INLINE F64Quad f64_quad_fms(
    F64Quad value, double coefficient, F64Quad source) noexcept {
    value.lanes = _mm256_sub_pd(
        value.lanes,
        _mm256_mul_pd(_mm256_set1_pd(coefficient), source.lanes));
    return value;
}

[[nodiscard]] DSMVC_FORCE_INLINE F64Quad f64_quad_divide(
    F64Quad value, double denominator) noexcept {
    return {_mm256_div_pd(value.lanes, _mm256_set1_pd(denominator))};
}

[[nodiscard]] DSMVC_FORCE_INLINE F64Quad f64_quad_multiply(
    F64Quad value, double coefficient) noexcept {
    return {_mm256_mul_pd(value.lanes, _mm256_set1_pd(coefficient))};
}

template <bool RetainedFloat64, class Loader>
void solve_axis_f64_quad(
    const AxisPlan &plan, Loader &&load_source,
    std::vector<F64Quad> &values) {
    const auto n = plan.destination_size;
    const auto width = static_cast<std::size_t>(n);
    values.resize(width);

    for (std::int32_t i = 0; i < n; ++i) {
        auto value = f64_quad_zero();
        for (auto offset = plan.transpose_offsets[static_cast<std::size_t>(i)];
             offset < plan.transpose_offsets[static_cast<std::size_t>(i) + 1U];
             ++offset) {
            const double weight = RetainedFloat64
                ? plan.transpose_weights_f64[offset]
                : static_cast<double>(plan.transpose_weights[offset]);
            value = f64_quad_fma(
                value, weight, load_source(plan.transpose_indices[offset]));
        }

        const auto available = std::min(plan.half_bandwidth, i);
        for (std::int32_t distance = available; distance >= 1; --distance) {
            const double factor = RetainedFloat64
                ? plan.ldlt_bands_f64[
                      static_cast<std::size_t>(distance) * width
                      + static_cast<std::size_t>(i - distance)]
                : static_cast<double>(plan.lower_ld[
                      static_cast<std::size_t>(distance - 1) * width
                      + static_cast<std::size_t>(i)]);
            value = f64_quad_fms(
                value, factor,
                values[static_cast<std::size_t>(i - distance)]);
        }
        if constexpr (RetainedFloat64) {
            values[static_cast<std::size_t>(i)] = value;
        } else {
            values[static_cast<std::size_t>(i)] = f64_quad_multiply(
                value, static_cast<double>(
                    plan.inverse_diagonal[static_cast<std::size_t>(i)]));
        }
    }

    if constexpr (RetainedFloat64) {
        for (std::int32_t i = 0; i < n; ++i) {
            values[static_cast<std::size_t>(i)] = f64_quad_divide(
                values[static_cast<std::size_t>(i)],
                plan.ldlt_bands_f64[static_cast<std::size_t>(i)]
                    + std::numeric_limits<double>::epsilon());
        }
    }

    for (std::int32_t i = n - 2; i >= 0; --i) {
        auto value = values[static_cast<std::size_t>(i)];
        const auto available = std::min(plan.half_bandwidth, n - i - 1);
        for (std::int32_t distance = available; distance >= 1; --distance) {
            const double factor = RetainedFloat64
                ? plan.ldlt_bands_f64[
                      static_cast<std::size_t>(distance) * width
                      + static_cast<std::size_t>(i)]
                : static_cast<double>(plan.upper_l[
                      static_cast<std::size_t>(distance - 1) * width
                      + static_cast<std::size_t>(i)]);
            value = f64_quad_fms(
                value, factor,
                values[static_cast<std::size_t>(i + distance)]);
        }
        values[static_cast<std::size_t>(i)] = value;
    }
}

template <class Loader>
void solve_axis_f64_quad_dispatch(
    const AxisPlan &plan, Loader &&load_source,
    std::vector<F64Quad> &values) {
    if (plan.requires_float64()) {
        solve_axis_f64_quad<true>(
            plan, std::forward<Loader>(load_source), values);
    } else {
        solve_axis_f64_quad<false>(
            plan, std::forward<Loader>(load_source), values);
    }
}

DSMVC_FORCE_INLINE void transpose4_f64(
    __m256d &row0, __m256d &row1,
    __m256d &row2, __m256d &row3) noexcept {
    const __m256d t0 = _mm256_unpacklo_pd(row0, row1);
    const __m256d t1 = _mm256_unpackhi_pd(row0, row1);
    const __m256d t2 = _mm256_unpacklo_pd(row2, row3);
    const __m256d t3 = _mm256_unpackhi_pd(row2, row3);
    row0 = _mm256_permute2f128_pd(t0, t2, 0x20);
    row1 = _mm256_permute2f128_pd(t1, t3, 0x20);
    row2 = _mm256_permute2f128_pd(t0, t2, 0x31);
    row3 = _mm256_permute2f128_pd(t1, t3, 0x31);
}

void load_rows_f32_quad(
    const AxisPlan &plan, const float *input, std::ptrdiff_t input_stride,
    std::vector<F64Quad> &source) {
    source.resize(static_cast<std::size_t>(plan.source_size));
    for (std::int32_t column = 0; column < plan.source_size; ++column) {
        const __m128 values = _mm_setr_ps(
            input[column], input[input_stride + column],
            input[2 * input_stride + column], input[3 * input_stride + column]);
        source[static_cast<std::size_t>(column)] = {
            _mm256_cvtps_pd(values),
        };
    }
}

void store_rows_f32_quad(
    const AxisPlan &plan, const std::vector<F64Quad> &values,
    float *output, std::ptrdiff_t output_stride) noexcept {
    std::int32_t column = 0;
    for (; column + 4 <= plan.destination_size; column += 4) {
        __m256d x0 = values[static_cast<std::size_t>(column)].lanes;
        __m256d x1 = values[static_cast<std::size_t>(column + 1)].lanes;
        __m256d x2 = values[static_cast<std::size_t>(column + 2)].lanes;
        __m256d x3 = values[static_cast<std::size_t>(column + 3)].lanes;
        transpose4_f64(x0, x1, x2, x3);
        _mm_storeu_ps(output + column, _mm256_cvtpd_ps(x0));
        _mm_storeu_ps(output + output_stride + column, _mm256_cvtpd_ps(x1));
        _mm_storeu_ps(output + 2 * output_stride + column, _mm256_cvtpd_ps(x2));
        _mm_storeu_ps(output + 3 * output_stride + column, _mm256_cvtpd_ps(x3));
    }
    alignas(32) double lanes[4];
    for (; column < plan.destination_size; ++column) {
        _mm256_store_pd(
            lanes, values[static_cast<std::size_t>(column)].lanes);
        output[column] = static_cast<float>(lanes[0]);
        output[output_stride + column] = static_cast<float>(lanes[1]);
        output[2 * output_stride + column] = static_cast<float>(lanes[2]);
        output[3 * output_stride + column] = static_cast<float>(lanes[3]);
    }
}

void store_rows_f64_quad(
    const AxisPlan &plan, const std::vector<F64Quad> &values,
    double *output, std::ptrdiff_t output_stride) noexcept {
    std::int32_t column = 0;
    for (; column + 4 <= plan.destination_size; column += 4) {
        __m256d x0 = values[static_cast<std::size_t>(column)].lanes;
        __m256d x1 = values[static_cast<std::size_t>(column + 1)].lanes;
        __m256d x2 = values[static_cast<std::size_t>(column + 2)].lanes;
        __m256d x3 = values[static_cast<std::size_t>(column + 3)].lanes;
        transpose4_f64(x0, x1, x2, x3);
        _mm256_storeu_pd(output + column, x0);
        _mm256_storeu_pd(output + output_stride + column, x1);
        _mm256_storeu_pd(output + 2 * output_stride + column, x2);
        _mm256_storeu_pd(output + 3 * output_stride + column, x3);
    }
    alignas(32) double lanes[4];
    for (; column < plan.destination_size; ++column) {
        _mm256_store_pd(
            lanes, values[static_cast<std::size_t>(column)].lanes);
        output[column] = lanes[0];
        output[output_stride + column] = lanes[1];
        output[2 * output_stride + column] = lanes[2];
        output[3 * output_stride + column] = lanes[3];
    }
}

template <class Output>
void inverse_rows_f64_avx2_impl(
    const AxisPlan &plan,
    const float *input, std::ptrdiff_t input_row_stride,
    Output *output, std::ptrdiff_t output_row_stride,
    std::int32_t row_count) {
    auto &workspace = f64_workspace();
    std::int32_t row = 0;
    for (; row + 4 <= row_count; row += 4) {
        const auto *source = input
            + static_cast<std::ptrdiff_t>(row) * input_row_stride;
        load_rows_f32_quad(plan, source, input_row_stride, workspace.source);
        solve_axis_f64_quad_dispatch(
            plan,
            [&](std::int32_t index) noexcept {
                return workspace.source[static_cast<std::size_t>(index)];
            },
            workspace.values);
        auto *destination = output
            + static_cast<std::ptrdiff_t>(row) * output_row_stride;
        if constexpr (std::is_same_v<Output, float>) {
            store_rows_f32_quad(
                plan, workspace.values, destination, output_row_stride);
        } else {
            store_rows_f64_quad(
                plan, workspace.values, destination, output_row_stride);
        }
    }

    workspace.scalar_source.resize(static_cast<std::size_t>(plan.source_size));
    workspace.scalar_destination.resize(
        static_cast<std::size_t>(plan.destination_size));
    for (; row < row_count; ++row) {
        const auto *source = input
            + static_cast<std::ptrdiff_t>(row) * input_row_stride;
        for (std::int32_t column = 0; column < plan.source_size; ++column) {
            workspace.scalar_source[static_cast<std::size_t>(column)] =
                static_cast<double>(source[column]);
        }
        detail::inverse_axis_f64_ordered(
            plan, workspace.scalar_source.data(), 1,
            workspace.scalar_destination.data(), 1);
        auto *destination = output
            + static_cast<std::ptrdiff_t>(row) * output_row_stride;
        for (std::int32_t column = 0; column < plan.destination_size; ++column) {
            destination[column] = static_cast<Output>(
                workspace.scalar_destination[static_cast<std::size_t>(column)]);
        }
    }
}

template <class Input>
void inverse_columns_f64_avx2_impl(
    const AxisPlan &plan,
    const Input *input, std::ptrdiff_t input_row_stride,
    float *output, std::ptrdiff_t output_row_stride,
    std::int32_t column_count) {
    auto &workspace = f64_workspace();
    std::int32_t column = 0;
    for (; column + 4 <= column_count; column += 4) {
        solve_axis_f64_quad_dispatch(
            plan,
            [&](std::int32_t index) noexcept {
                const auto *source = input
                    + static_cast<std::ptrdiff_t>(index) * input_row_stride
                    + column;
                if constexpr (std::is_same_v<Input, float>) {
                    return F64Quad{_mm256_cvtps_pd(_mm_loadu_ps(source))};
                } else {
                    return F64Quad{_mm256_loadu_pd(source)};
                }
            },
            workspace.values);
        for (std::int32_t row = 0; row < plan.destination_size; ++row) {
            _mm_storeu_ps(
                output + static_cast<std::ptrdiff_t>(row) * output_row_stride
                    + column,
                _mm256_cvtpd_ps(
                    workspace.values[static_cast<std::size_t>(row)].lanes));
        }
    }

    workspace.scalar_source.resize(static_cast<std::size_t>(plan.source_size));
    workspace.scalar_destination.resize(
        static_cast<std::size_t>(plan.destination_size));
    for (; column < column_count; ++column) {
        for (std::int32_t row = 0; row < plan.source_size; ++row) {
            workspace.scalar_source[static_cast<std::size_t>(row)] =
                static_cast<double>(input[
                    static_cast<std::ptrdiff_t>(row) * input_row_stride
                    + column]);
        }
        detail::inverse_axis_f64_ordered(
            plan, workspace.scalar_source.data(), 1,
            workspace.scalar_destination.data(), 1);
        for (std::int32_t row = 0; row < plan.destination_size; ++row) {
            output[static_cast<std::ptrdiff_t>(row) * output_row_stride
                   + column] = static_cast<float>(
                workspace.scalar_destination[static_cast<std::size_t>(row)]);
        }
    }
}

DSMVC_FORCE_INLINE void transpose8(__m256 &row0, __m256 &row1, __m256 &row2, __m256 &row3,
                __m256 &row4, __m256 &row5, __m256 &row6, __m256 &row7) noexcept {
    const __m256 t0 = _mm256_unpacklo_ps(row0, row1);
    const __m256 t1 = _mm256_unpackhi_ps(row0, row1);
    const __m256 t2 = _mm256_unpacklo_ps(row2, row3);
    const __m256 t3 = _mm256_unpackhi_ps(row2, row3);
    const __m256 t4 = _mm256_unpacklo_ps(row4, row5);
    const __m256 t5 = _mm256_unpackhi_ps(row4, row5);
    const __m256 t6 = _mm256_unpacklo_ps(row6, row7);
    const __m256 t7 = _mm256_unpackhi_ps(row6, row7);
    const __m256 u0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 u1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 u2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 u3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 u4 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 u5 = _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2));
    const __m256 u6 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0));
    const __m256 u7 = _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2));
    row0 = _mm256_permute2f128_ps(u0, u4, 0x20);
    row1 = _mm256_permute2f128_ps(u1, u5, 0x20);
    row2 = _mm256_permute2f128_ps(u2, u6, 0x20);
    row3 = _mm256_permute2f128_ps(u3, u7, 0x20);
    row4 = _mm256_permute2f128_ps(u0, u4, 0x31);
    row5 = _mm256_permute2f128_ps(u1, u5, 0x31);
    row6 = _mm256_permute2f128_ps(u2, u6, 0x31);
    row7 = _mm256_permute2f128_ps(u3, u7, 0x31);
}

void transpose_source(const float *input, std::ptrdiff_t stride,
                      std::int32_t logical_width,
                      std::int32_t padded_width, float *scratch) noexcept {
    for (std::int32_t column = 0; column < padded_width; column += 8) {
        const auto remaining = std::clamp(logical_width - column, 0, 8);
        const __m256i mask = _mm256_setr_epi32(
            remaining > 0 ? -1 : 0, remaining > 1 ? -1 : 0,
            remaining > 2 ? -1 : 0, remaining > 3 ? -1 : 0,
            remaining > 4 ? -1 : 0, remaining > 5 ? -1 : 0,
            remaining > 6 ? -1 : 0, remaining > 7 ? -1 : 0);
        const auto load_row = [&](std::int32_t row) noexcept {
            const auto *source = input
                + static_cast<std::ptrdiff_t>(row) * stride + column;
            return remaining == 8
                ? _mm256_loadu_ps(source)
                : _mm256_maskload_ps(source, mask);
        };
        __m256 x0 = load_row(0);
        __m256 x1 = load_row(1);
        __m256 x2 = load_row(2);
        __m256 x3 = load_row(3);
        __m256 x4 = load_row(4);
        __m256 x5 = load_row(5);
        __m256 x6 = load_row(6);
        __m256 x7 = load_row(7);
        transpose8(x0, x1, x2, x3, x4, x5, x6, x7);
        _mm256_store_ps(scratch + static_cast<std::size_t>(column + 0) * 8U, x0);
        _mm256_store_ps(scratch + static_cast<std::size_t>(column + 1) * 8U, x1);
        _mm256_store_ps(scratch + static_cast<std::size_t>(column + 2) * 8U, x2);
        _mm256_store_ps(scratch + static_cast<std::size_t>(column + 3) * 8U, x3);
        _mm256_store_ps(scratch + static_cast<std::size_t>(column + 4) * 8U, x4);
        _mm256_store_ps(scratch + static_cast<std::size_t>(column + 5) * 8U, x5);
        _mm256_store_ps(scratch + static_cast<std::size_t>(column + 6) * 8U, x6);
        _mm256_store_ps(scratch + static_cast<std::size_t>(column + 7) * 8U, x7);
    }
}

template <class Sample>
void transpose_integer_source(
    const Sample *input, std::ptrdiff_t stride,
    std::int32_t logical_width, std::int32_t padded_width, float input_offset,
    float input_scale, float *scratch) noexcept {
    const __m256 offset = _mm256_set1_ps(input_offset);
    const __m256 scale = _mm256_set1_ps(input_scale);
    for (std::int32_t column = 0; column < padded_width; column += 8) {
        const auto load_row = [&](std::int32_t row) {
            const auto *source = input
                + static_cast<std::ptrdiff_t>(row) * stride + column;
            const auto remaining = std::clamp(logical_width - column, 0, 8);
            std::array<Sample, 8> tail{};
            if (remaining != 8) {
                std::memcpy(
                    tail.data(), source,
                    static_cast<std::size_t>(remaining) * sizeof(Sample));
                source = tail.data();
            }
            __m256i integers;
            if constexpr (std::is_same_v<Sample, std::uint8_t>) {
                integers = _mm256_cvtepu8_epi32(
                    _mm_loadl_epi64(reinterpret_cast<const __m128i *>(source)));
            } else {
                integers = _mm256_cvtepu16_epi32(
                    _mm_loadu_si128(reinterpret_cast<const __m128i *>(source)));
            }
            return _mm256_mul_ps(
                _mm256_sub_ps(_mm256_cvtepi32_ps(integers), offset), scale);
        };
        __m256 x0 = load_row(0);
        __m256 x1 = load_row(1);
        __m256 x2 = load_row(2);
        __m256 x3 = load_row(3);
        __m256 x4 = load_row(4);
        __m256 x5 = load_row(5);
        __m256 x6 = load_row(6);
        __m256 x7 = load_row(7);
        transpose8(x0, x1, x2, x3, x4, x5, x6, x7);
        _mm256_store_ps(scratch + static_cast<std::size_t>(column + 0) * 8U, x0);
        _mm256_store_ps(scratch + static_cast<std::size_t>(column + 1) * 8U, x1);
        _mm256_store_ps(scratch + static_cast<std::size_t>(column + 2) * 8U, x2);
        _mm256_store_ps(scratch + static_cast<std::size_t>(column + 3) * 8U, x3);
        _mm256_store_ps(scratch + static_cast<std::size_t>(column + 4) * 8U, x4);
        _mm256_store_ps(scratch + static_cast<std::size_t>(column + 5) * 8U, x5);
        _mm256_store_ps(scratch + static_cast<std::size_t>(column + 6) * 8U, x6);
        _mm256_store_ps(scratch + static_cast<std::size_t>(column + 7) * 8U, x7);
    }
}

[[nodiscard]] DSMVC_FORCE_INLINE __m256 multiply_transpose(const detail::PackedCpuPlan &packed,
                                        const float *scratch,
                                        std::int32_t row) noexcept {
    __m256 sum = _mm256_setzero_ps();
    const auto left = packed.weights_left[static_cast<std::size_t>(row)];
    const auto right = packed.weights_right[static_cast<std::size_t>(row)];
    const auto base = static_cast<std::size_t>(row)
        * static_cast<std::size_t>(packed.weights_columns);
    for (std::int32_t source = left; source < right; ++source) {
        const __m256 weight = _mm256_set1_ps(
            packed.weights[base + static_cast<std::size_t>(source - left)]);
        sum = _mm256_fmadd_ps(
            weight, _mm256_load_ps(
                        scratch + static_cast<std::size_t>(source) * 8U), sum);
    }
    return sum;
}

[[nodiscard]] DSMVC_FORCE_INLINE __m256 forward_b1(const detail::PackedCpuPlan &packed,
                                std::int32_t i, __m256 value,
                                __m256 previous) noexcept {
    value = _mm256_fnmadd_ps(
        _mm256_set1_ps(packed.lower_ld[static_cast<std::size_t>(i)]),
        previous, value);
    return _mm256_mul_ps(
        value, _mm256_set1_ps(
                   packed.inverse_diagonal[static_cast<std::size_t>(i)]));
}

[[nodiscard]] DSMVC_FORCE_INLINE __m256 backward_b1(const detail::PackedCpuPlan &packed,
                                 std::int32_t i, __m256 value,
                                 __m256 next) noexcept {
    return _mm256_fnmadd_ps(
        _mm256_set1_ps(packed.upper_l[static_cast<std::size_t>(i)]),
        next, value);
}

[[nodiscard]] DSMVC_FORCE_INLINE __m256 forward_b3(const detail::PackedCpuPlan &packed,
                                std::int32_t i, __m256 value,
                                __m256 previous1, __m256 previous2,
                                __m256 previous3) noexcept {
    const auto stride = static_cast<std::size_t>(packed.padded_destination_size);
    const auto index = static_cast<std::size_t>(i);
    value = _mm256_fnmadd_ps(_mm256_set1_ps(packed.lower_ld[2U * stride + index]),
                             previous3, value);
    value = _mm256_fnmadd_ps(_mm256_set1_ps(packed.lower_ld[stride + index]),
                             previous2, value);
    value = _mm256_fnmadd_ps(_mm256_set1_ps(packed.lower_ld[index]),
                             previous1, value);
    return _mm256_mul_ps(value, _mm256_set1_ps(packed.inverse_diagonal[index]));
}

[[nodiscard]] DSMVC_FORCE_INLINE __m256 backward_b3(const detail::PackedCpuPlan &packed,
                                 std::int32_t i, __m256 value,
                                 __m256 next1, __m256 next2,
                                 __m256 next3) noexcept {
    const auto stride = static_cast<std::size_t>(packed.padded_destination_size);
    const auto index = static_cast<std::size_t>(i);
    value = _mm256_fnmadd_ps(_mm256_set1_ps(packed.upper_l[index]), next1, value);
    value = _mm256_fnmadd_ps(_mm256_set1_ps(packed.upper_l[stride + index]),
                             next2, value);
    return _mm256_fnmadd_ps(
        _mm256_set1_ps(packed.upper_l[2U * stride + index]), next3, value);
}

void solve_horizontal_b1(const detail::PackedCpuPlan &packed, const float *scratch,
                         float *output, std::ptrdiff_t stride) noexcept {
    __m256 previous = _mm256_setzero_ps();
    const auto padded = packed.padded_destination_size;
    for (std::int32_t j = 0; j < padded; j += 8) {
        __m256 x0 = forward_b1(packed, j + 0, multiply_transpose(packed, scratch, j + 0), previous);
        __m256 x1 = forward_b1(packed, j + 1, multiply_transpose(packed, scratch, j + 1), x0);
        __m256 x2 = forward_b1(packed, j + 2, multiply_transpose(packed, scratch, j + 2), x1);
        __m256 x3 = forward_b1(packed, j + 3, multiply_transpose(packed, scratch, j + 3), x2);
        __m256 x4 = forward_b1(packed, j + 4, multiply_transpose(packed, scratch, j + 4), x3);
        __m256 x5 = forward_b1(packed, j + 5, multiply_transpose(packed, scratch, j + 5), x4);
        __m256 x6 = forward_b1(packed, j + 6, multiply_transpose(packed, scratch, j + 6), x5);
        __m256 x7 = forward_b1(packed, j + 7, multiply_transpose(packed, scratch, j + 7), x6);
        previous = x7;
        _mm256_storeu_ps(output + j, x0);
        _mm256_storeu_ps(output + stride + j, x1);
        _mm256_storeu_ps(output + 2 * stride + j, x2);
        _mm256_storeu_ps(output + 3 * stride + j, x3);
        _mm256_storeu_ps(output + 4 * stride + j, x4);
        _mm256_storeu_ps(output + 5 * stride + j, x5);
        _mm256_storeu_ps(output + 6 * stride + j, x6);
        _mm256_storeu_ps(output + 7 * stride + j, x7);
    }

    __m256 next = _mm256_setzero_ps();
    for (std::int32_t j = padded - 8; j >= 0; j -= 8) {
        __m256 x0 = _mm256_loadu_ps(output + j);
        __m256 x1 = _mm256_loadu_ps(output + stride + j);
        __m256 x2 = _mm256_loadu_ps(output + 2 * stride + j);
        __m256 x3 = _mm256_loadu_ps(output + 3 * stride + j);
        __m256 x4 = _mm256_loadu_ps(output + 4 * stride + j);
        __m256 x5 = _mm256_loadu_ps(output + 5 * stride + j);
        __m256 x6 = _mm256_loadu_ps(output + 6 * stride + j);
        __m256 x7 = _mm256_loadu_ps(output + 7 * stride + j);
        x7 = backward_b1(packed, j + 7, x7, next);
        x6 = backward_b1(packed, j + 6, x6, x7);
        x5 = backward_b1(packed, j + 5, x5, x6);
        x4 = backward_b1(packed, j + 4, x4, x5);
        x3 = backward_b1(packed, j + 3, x3, x4);
        x2 = backward_b1(packed, j + 2, x2, x3);
        x1 = backward_b1(packed, j + 1, x1, x2);
        x0 = backward_b1(packed, j + 0, x0, x1);
        next = x0;
        transpose8(x0, x1, x2, x3, x4, x5, x6, x7);
        _mm256_storeu_ps(output + j, x0);
        _mm256_storeu_ps(output + stride + j, x1);
        _mm256_storeu_ps(output + 2 * stride + j, x2);
        _mm256_storeu_ps(output + 3 * stride + j, x3);
        _mm256_storeu_ps(output + 4 * stride + j, x4);
        _mm256_storeu_ps(output + 5 * stride + j, x5);
        _mm256_storeu_ps(output + 6 * stride + j, x6);
        _mm256_storeu_ps(output + 7 * stride + j, x7);
    }
}

void solve_horizontal_b3(const detail::PackedCpuPlan &packed, const float *scratch,
                         float *output, std::ptrdiff_t stride) noexcept {
    __m256 previous1 = _mm256_setzero_ps();
    __m256 previous2 = _mm256_setzero_ps();
    __m256 previous3 = _mm256_setzero_ps();
    const auto padded = packed.padded_destination_size;
    for (std::int32_t j = 0; j < padded; j += 8) {
#define DSMVC_FORWARD3(LANE, PREVIOUS1, PREVIOUS2, PREVIOUS3) \
        __m256 x##LANE = forward_b3( \
            packed, j + LANE, multiply_transpose(packed, scratch, j + LANE), \
            PREVIOUS1, PREVIOUS2, PREVIOUS3)
        DSMVC_FORWARD3(0, previous1, previous2, previous3);
        DSMVC_FORWARD3(1, x0, previous1, previous2);
        DSMVC_FORWARD3(2, x1, x0, previous1);
        DSMVC_FORWARD3(3, x2, x1, x0);
        DSMVC_FORWARD3(4, x3, x2, x1);
        DSMVC_FORWARD3(5, x4, x3, x2);
        DSMVC_FORWARD3(6, x5, x4, x3);
        DSMVC_FORWARD3(7, x6, x5, x4);
#undef DSMVC_FORWARD3
        previous1 = x7;
        previous2 = x6;
        previous3 = x5;
        _mm256_storeu_ps(output + j, x0);
        _mm256_storeu_ps(output + stride + j, x1);
        _mm256_storeu_ps(output + 2 * stride + j, x2);
        _mm256_storeu_ps(output + 3 * stride + j, x3);
        _mm256_storeu_ps(output + 4 * stride + j, x4);
        _mm256_storeu_ps(output + 5 * stride + j, x5);
        _mm256_storeu_ps(output + 6 * stride + j, x6);
        _mm256_storeu_ps(output + 7 * stride + j, x7);
    }

    __m256 next1 = _mm256_setzero_ps();
    __m256 next2 = _mm256_setzero_ps();
    __m256 next3 = _mm256_setzero_ps();
    for (std::int32_t j = padded - 8; j >= 0; j -= 8) {
        __m256 x0 = _mm256_loadu_ps(output + j);
        __m256 x1 = _mm256_loadu_ps(output + stride + j);
        __m256 x2 = _mm256_loadu_ps(output + 2 * stride + j);
        __m256 x3 = _mm256_loadu_ps(output + 3 * stride + j);
        __m256 x4 = _mm256_loadu_ps(output + 4 * stride + j);
        __m256 x5 = _mm256_loadu_ps(output + 5 * stride + j);
        __m256 x6 = _mm256_loadu_ps(output + 6 * stride + j);
        __m256 x7 = _mm256_loadu_ps(output + 7 * stride + j);
        x7 = backward_b3(packed, j + 7, x7, next1, next2, next3);
        x6 = backward_b3(packed, j + 6, x6, x7, next1, next2);
        x5 = backward_b3(packed, j + 5, x5, x6, x7, next1);
        x4 = backward_b3(packed, j + 4, x4, x5, x6, x7);
        x3 = backward_b3(packed, j + 3, x3, x4, x5, x6);
        x2 = backward_b3(packed, j + 2, x2, x3, x4, x5);
        x1 = backward_b3(packed, j + 1, x1, x2, x3, x4);
        x0 = backward_b3(packed, j + 0, x0, x1, x2, x3);
        next1 = x0;
        next2 = x1;
        next3 = x2;
        transpose8(x0, x1, x2, x3, x4, x5, x6, x7);
        _mm256_storeu_ps(output + j, x0);
        _mm256_storeu_ps(output + stride + j, x1);
        _mm256_storeu_ps(output + 2 * stride + j, x2);
        _mm256_storeu_ps(output + 3 * stride + j, x3);
        _mm256_storeu_ps(output + 4 * stride + j, x4);
        _mm256_storeu_ps(output + 5 * stride + j, x5);
        _mm256_storeu_ps(output + 6 * stride + j, x6);
        _mm256_storeu_ps(output + 7 * stride + j, x7);
    }
}

[[nodiscard]] float *transposed_output(float *output, std::ptrdiff_t stride,
                                       std::int32_t index) noexcept {
    return output + static_cast<std::ptrdiff_t>(index & 7) * stride
        + static_cast<std::ptrdiff_t>(index & ~7);
}

void solve_horizontal_generic(const AxisPlan &plan,
                              const detail::PackedCpuPlan &packed,
                              const float *scratch, float *output,
                              std::ptrdiff_t stride) noexcept {
    const auto n = plan.destination_size;
    const auto factor_stride = static_cast<std::size_t>(packed.padded_destination_size);
    for (std::int32_t i = 0; i < n; ++i) {
        __m256 value = multiply_transpose(packed, scratch, i);
        const auto available = std::min(plan.half_bandwidth, i);
        for (std::int32_t distance = available; distance >= 1; --distance) {
            value = _mm256_fnmadd_ps(
                _mm256_set1_ps(packed.lower_ld[
                    static_cast<std::size_t>(distance - 1) * factor_stride
                    + static_cast<std::size_t>(i)]),
                _mm256_loadu_ps(transposed_output(output, stride, i - distance)),
                value);
        }
        value = _mm256_mul_ps(
            value, _mm256_set1_ps(
                       packed.inverse_diagonal[static_cast<std::size_t>(i)]));
        _mm256_storeu_ps(transposed_output(output, stride, i), value);
    }
    for (std::int32_t i = n; i < packed.padded_destination_size; ++i) {
        _mm256_storeu_ps(transposed_output(output, stride, i), _mm256_setzero_ps());
    }
    for (std::int32_t i = n - 2; i >= 0; --i) {
        __m256 value = _mm256_loadu_ps(transposed_output(output, stride, i));
        const auto available = std::min(plan.half_bandwidth, n - i - 1);
        for (std::int32_t distance = available; distance >= 1; --distance) {
            value = _mm256_fnmadd_ps(
                _mm256_set1_ps(packed.upper_l[
                    static_cast<std::size_t>(distance - 1) * factor_stride
                    + static_cast<std::size_t>(i)]),
                _mm256_loadu_ps(transposed_output(output, stride, i + distance)),
                value);
        }
        _mm256_storeu_ps(transposed_output(output, stride, i), value);
    }
    for (std::int32_t j = 0; j < packed.padded_destination_size; j += 8) {
        __m256 x0 = _mm256_loadu_ps(output + j);
        __m256 x1 = _mm256_loadu_ps(output + stride + j);
        __m256 x2 = _mm256_loadu_ps(output + 2 * stride + j);
        __m256 x3 = _mm256_loadu_ps(output + 3 * stride + j);
        __m256 x4 = _mm256_loadu_ps(output + 4 * stride + j);
        __m256 x5 = _mm256_loadu_ps(output + 5 * stride + j);
        __m256 x6 = _mm256_loadu_ps(output + 6 * stride + j);
        __m256 x7 = _mm256_loadu_ps(output + 7 * stride + j);
        transpose8(x0, x1, x2, x3, x4, x5, x6, x7);
        _mm256_storeu_ps(output + j, x0);
        _mm256_storeu_ps(output + stride + j, x1);
        _mm256_storeu_ps(output + 2 * stride + j, x2);
        _mm256_storeu_ps(output + 3 * stride + j, x3);
        _mm256_storeu_ps(output + 4 * stride + j, x4);
        _mm256_storeu_ps(output + 5 * stride + j, x5);
        _mm256_storeu_ps(output + 6 * stride + j, x6);
        _mm256_storeu_ps(output + 7 * stride + j, x7);
    }
}

DSMVC_FLATTEN void solve_horizontal_block(
    const AxisPlan &plan, const detail::PackedCpuPlan &packed,
    const float *input, std::ptrdiff_t input_stride,
    float *output, std::ptrdiff_t output_stride,
    float *scratch) noexcept {
    transpose_source(
        input, input_stride, plan.source_size,
        packed.padded_source_size, scratch);
    if (plan.half_bandwidth == 1) {
        solve_horizontal_b1(packed, scratch, output, output_stride);
    } else if (plan.half_bandwidth == 3) {
        solve_horizontal_b3(packed, scratch, output, output_stride);
    } else {
        solve_horizontal_generic(plan, packed, scratch, output, output_stride);
    }
}

template <class Sample>
void solve_horizontal_integer_block(
    const AxisPlan &plan, const detail::PackedCpuPlan &packed,
    const Sample *input, std::ptrdiff_t input_stride,
    float input_offset, float input_scale,
    float *output, std::ptrdiff_t output_stride,
    float *scratch) noexcept {
    transpose_integer_source(
        input, input_stride, plan.source_size, packed.padded_source_size,
        input_offset, input_scale, scratch);
    if (plan.half_bandwidth == 1) {
        solve_horizontal_b1(packed, scratch, output, output_stride);
    } else if (plan.half_bandwidth == 3) {
        solve_horizontal_b3(packed, scratch, output, output_stride);
    } else {
        solve_horizontal_generic(plan, packed, scratch, output, output_stride);
    }
}

DSMVC_FORCE_INLINE void multiply_columns_pair(
    const detail::PackedCpuPlan &packed,
    const float *input, std::ptrdiff_t input_stride,
    std::int32_t row, std::int32_t column,
    __m256 &value0, __m256 &value1) noexcept {
    const auto left = packed.weights_left[static_cast<std::size_t>(row)];
    const auto right = packed.weights_right[static_cast<std::size_t>(row)];
    const auto weight_base = static_cast<std::size_t>(row)
        * static_cast<std::size_t>(packed.weights_columns);
    value0 = _mm256_setzero_ps();
    value1 = _mm256_setzero_ps();

    if (right - left == 2) {
        const auto *source = input + static_cast<std::ptrdiff_t>(left)
            * input_stride + column;
        const auto *weights = packed.weights.data() + weight_base;
#define DSMVC_ACCUMULATE_PAIR_2(TAP) \
        { \
            const __m256 weight = _mm256_set1_ps(weights[TAP]); \
            const auto *tap_source = source \
                + static_cast<std::ptrdiff_t>(TAP) * input_stride; \
            value0 = _mm256_fmadd_ps( \
                weight, _mm256_loadu_ps(tap_source), value0); \
            value1 = _mm256_fmadd_ps( \
                weight, _mm256_loadu_ps(tap_source + 8), value1); \
        }
        DSMVC_ACCUMULATE_PAIR_2(0);
        DSMVC_ACCUMULATE_PAIR_2(1);
#undef DSMVC_ACCUMULATE_PAIR_2
        return;
    }

    if (right - left == 4) {
        const auto *source = input + static_cast<std::ptrdiff_t>(left)
            * input_stride + column;
        const auto *weights = packed.weights.data() + weight_base;
#define DSMVC_ACCUMULATE_PAIR(TAP) \
        { \
            const __m256 weight = _mm256_set1_ps(weights[TAP]); \
            const auto *tap_source = source \
                + static_cast<std::ptrdiff_t>(TAP) * input_stride; \
            value0 = _mm256_fmadd_ps( \
                weight, _mm256_loadu_ps(tap_source), value0); \
            value1 = _mm256_fmadd_ps( \
                weight, _mm256_loadu_ps(tap_source + 8), value1); \
        }
        DSMVC_ACCUMULATE_PAIR(0);
        DSMVC_ACCUMULATE_PAIR(1);
        DSMVC_ACCUMULATE_PAIR(2);
        DSMVC_ACCUMULATE_PAIR(3);
#undef DSMVC_ACCUMULATE_PAIR
        return;
    }

    for (std::int32_t source = left; source < right; ++source) {
        const __m256 weight = _mm256_set1_ps(
            packed.weights[weight_base
                + static_cast<std::size_t>(source - left)]);
        const auto *tap_source = input + static_cast<std::ptrdiff_t>(source)
            * input_stride + column;
        value0 = _mm256_fmadd_ps(
            weight, _mm256_loadu_ps(tap_source), value0);
        value1 = _mm256_fmadd_ps(
            weight, _mm256_loadu_ps(tap_source + 8), value1);
    }
}

DSMVC_FORCE_INLINE __m256 multiply_columns_single(
    const detail::PackedCpuPlan &packed,
    const float *input, std::ptrdiff_t input_stride,
    std::int32_t row, std::int32_t column) noexcept {
    const auto left = packed.weights_left[static_cast<std::size_t>(row)];
    const auto right = packed.weights_right[static_cast<std::size_t>(row)];
    const auto weight_base = static_cast<std::size_t>(row)
        * static_cast<std::size_t>(packed.weights_columns);
    __m256 value = _mm256_setzero_ps();

    if (right - left == 2) {
        const auto *source = input + static_cast<std::ptrdiff_t>(left)
            * input_stride + column;
        const auto *weights = packed.weights.data() + weight_base;
#define DSMVC_ACCUMULATE_SINGLE_2(TAP) \
        value = _mm256_fmadd_ps( \
            _mm256_set1_ps(weights[TAP]), \
            _mm256_loadu_ps(source \
                + static_cast<std::ptrdiff_t>(TAP) * input_stride), value)
        DSMVC_ACCUMULATE_SINGLE_2(0);
        DSMVC_ACCUMULATE_SINGLE_2(1);
#undef DSMVC_ACCUMULATE_SINGLE_2
        return value;
    }

    if (right - left == 4) {
        const auto *source = input + static_cast<std::ptrdiff_t>(left)
            * input_stride + column;
        const auto *weights = packed.weights.data() + weight_base;
#define DSMVC_ACCUMULATE_SINGLE(TAP) \
        value = _mm256_fmadd_ps( \
            _mm256_set1_ps(weights[TAP]), \
            _mm256_loadu_ps(source \
                + static_cast<std::ptrdiff_t>(TAP) * input_stride), value)
        DSMVC_ACCUMULATE_SINGLE(0);
        DSMVC_ACCUMULATE_SINGLE(1);
        DSMVC_ACCUMULATE_SINGLE(2);
        DSMVC_ACCUMULATE_SINGLE(3);
#undef DSMVC_ACCUMULATE_SINGLE
        return value;
    }

    for (std::int32_t source = left; source < right; ++source) {
        value = _mm256_fmadd_ps(
            _mm256_set1_ps(packed.weights[weight_base
                + static_cast<std::size_t>(source - left)]),
            _mm256_loadu_ps(input + static_cast<std::ptrdiff_t>(source)
                * input_stride + column), value);
    }
    return value;
}

template <bool RhsOnly>
void solve_columns_b3_pair(const detail::PackedCpuPlan &packed,
                           const float *input, std::ptrdiff_t input_stride,
                           float *output, std::ptrdiff_t output_stride,
                           std::int32_t column,
                           std::int32_t destination_size) noexcept {
    const auto factor_stride = static_cast<std::size_t>(
        packed.padded_destination_size);
    __m256 previous10 = _mm256_setzero_ps();
    __m256 previous11 = _mm256_setzero_ps();
    __m256 previous20 = _mm256_setzero_ps();
    __m256 previous21 = _mm256_setzero_ps();
    __m256 previous30 = _mm256_setzero_ps();
    __m256 previous31 = _mm256_setzero_ps();

    for (std::int32_t i = 0; i < destination_size; ++i) {
        __m256 value0;
        __m256 value1;
        if constexpr (RhsOnly) {
            const auto *rhs = output + static_cast<std::ptrdiff_t>(i)
                * output_stride + column;
            value0 = _mm256_loadu_ps(rhs);
            value1 = _mm256_loadu_ps(rhs + 8);
        } else {
            multiply_columns_pair(
                packed, input, input_stride, i, column, value0, value1);
        }
        const auto index = static_cast<std::size_t>(i);
        if (i >= 3) {
            const __m256 lower3 = _mm256_set1_ps(
                packed.lower_ld[2U * factor_stride + index]);
            value0 = _mm256_fnmadd_ps(lower3, previous30, value0);
            value1 = _mm256_fnmadd_ps(lower3, previous31, value1);
        }
        if (i >= 2) {
            const __m256 lower2 = _mm256_set1_ps(
                packed.lower_ld[factor_stride + index]);
            value0 = _mm256_fnmadd_ps(lower2, previous20, value0);
            value1 = _mm256_fnmadd_ps(lower2, previous21, value1);
        }
        if (i >= 1) {
            const __m256 lower1 = _mm256_set1_ps(packed.lower_ld[index]);
            value0 = _mm256_fnmadd_ps(lower1, previous10, value0);
            value1 = _mm256_fnmadd_ps(lower1, previous11, value1);
        }
        const __m256 inverse = _mm256_set1_ps(packed.inverse_diagonal[index]);
        value0 = _mm256_mul_ps(value0, inverse);
        value1 = _mm256_mul_ps(value1, inverse);
        auto *destination = output + static_cast<std::ptrdiff_t>(i)
            * output_stride + column;
        _mm256_storeu_ps(destination, value0);
        _mm256_storeu_ps(destination + 8, value1);
        previous30 = previous20;
        previous31 = previous21;
        previous20 = previous10;
        previous21 = previous11;
        previous10 = value0;
        previous11 = value1;
    }

    if (destination_size < 2) return;
    const auto *last = output + static_cast<std::ptrdiff_t>(destination_size - 1)
        * output_stride + column;
    __m256 next10 = _mm256_loadu_ps(last);
    __m256 next11 = _mm256_loadu_ps(last + 8);
    __m256 next20 = _mm256_setzero_ps();
    __m256 next21 = _mm256_setzero_ps();
    __m256 next30 = _mm256_setzero_ps();
    __m256 next31 = _mm256_setzero_ps();
    for (std::int32_t i = destination_size - 2; i >= 0; --i) {
        auto *destination = output + static_cast<std::ptrdiff_t>(i)
            * output_stride + column;
        __m256 value0 = _mm256_loadu_ps(destination);
        __m256 value1 = _mm256_loadu_ps(destination + 8);
        const auto index = static_cast<std::size_t>(i);
        const __m256 upper1 = _mm256_set1_ps(packed.upper_l[index]);
        value0 = _mm256_fnmadd_ps(upper1, next10, value0);
        value1 = _mm256_fnmadd_ps(upper1, next11, value1);
        if (i + 2 < destination_size) {
            const __m256 upper2 = _mm256_set1_ps(
                packed.upper_l[factor_stride + index]);
            value0 = _mm256_fnmadd_ps(upper2, next20, value0);
            value1 = _mm256_fnmadd_ps(upper2, next21, value1);
        }
        if (i + 3 < destination_size) {
            const __m256 upper3 = _mm256_set1_ps(
                packed.upper_l[2U * factor_stride + index]);
            value0 = _mm256_fnmadd_ps(upper3, next30, value0);
            value1 = _mm256_fnmadd_ps(upper3, next31, value1);
        }
        _mm256_storeu_ps(destination, value0);
        _mm256_storeu_ps(destination + 8, value1);
        next30 = next20;
        next31 = next21;
        next20 = next10;
        next21 = next11;
        next10 = value0;
        next11 = value1;
    }
}

template <bool RhsOnly>
void solve_columns_b3_single(const detail::PackedCpuPlan &packed,
                             const float *input, std::ptrdiff_t input_stride,
                             float *output, std::ptrdiff_t output_stride,
                             std::int32_t column,
                             std::int32_t destination_size) noexcept {
    const auto factor_stride = static_cast<std::size_t>(
        packed.padded_destination_size);
    __m256 previous1 = _mm256_setzero_ps();
    __m256 previous2 = _mm256_setzero_ps();
    __m256 previous3 = _mm256_setzero_ps();
    for (std::int32_t i = 0; i < destination_size; ++i) {
        __m256 value;
        if constexpr (RhsOnly) {
            value = _mm256_loadu_ps(
                output + static_cast<std::ptrdiff_t>(i)
                    * output_stride + column);
        } else {
            value = multiply_columns_single(
                packed, input, input_stride, i, column);
        }
        const auto index = static_cast<std::size_t>(i);
        if (i >= 3) {
            value = _mm256_fnmadd_ps(
                _mm256_set1_ps(packed.lower_ld[2U * factor_stride + index]),
                previous3, value);
        }
        if (i >= 2) {
            value = _mm256_fnmadd_ps(
                _mm256_set1_ps(packed.lower_ld[factor_stride + index]),
                previous2, value);
        }
        if (i >= 1) {
            value = _mm256_fnmadd_ps(
                _mm256_set1_ps(packed.lower_ld[index]), previous1, value);
        }
        value = _mm256_mul_ps(
            value, _mm256_set1_ps(packed.inverse_diagonal[index]));
        _mm256_storeu_ps(output + static_cast<std::ptrdiff_t>(i)
                         * output_stride + column, value);
        previous3 = previous2;
        previous2 = previous1;
        previous1 = value;
    }

    if (destination_size < 2) return;
    __m256 next1 = _mm256_loadu_ps(
        output + static_cast<std::ptrdiff_t>(destination_size - 1)
            * output_stride + column);
    __m256 next2 = _mm256_setzero_ps();
    __m256 next3 = _mm256_setzero_ps();
    for (std::int32_t i = destination_size - 2; i >= 0; --i) {
        auto *destination = output + static_cast<std::ptrdiff_t>(i)
            * output_stride + column;
        __m256 value = _mm256_loadu_ps(destination);
        const auto index = static_cast<std::size_t>(i);
        value = _mm256_fnmadd_ps(
            _mm256_set1_ps(packed.upper_l[index]), next1, value);
        if (i + 2 < destination_size) {
            value = _mm256_fnmadd_ps(
                _mm256_set1_ps(packed.upper_l[factor_stride + index]),
                next2, value);
        }
        if (i + 3 < destination_size) {
            value = _mm256_fnmadd_ps(
                _mm256_set1_ps(packed.upper_l[2U * factor_stride + index]),
                next3, value);
        }
        _mm256_storeu_ps(destination, value);
        next3 = next2;
        next2 = next1;
        next1 = value;
    }
}

void solve_columns_b3(const AxisPlan &plan,
                      const detail::PackedCpuPlan &packed,
                      const float *input, std::ptrdiff_t input_stride,
                      float *output, std::ptrdiff_t output_stride,
                      std::int32_t vector_columns) noexcept {
    const auto paired_columns = vector_columns & ~15;
    for (std::int32_t column = 0; column < paired_columns; column += 16) {
        solve_columns_b3_pair<false>(
            packed, input, input_stride, output, output_stride,
            column, plan.destination_size);
    }
    if (paired_columns != vector_columns) {
        solve_columns_b3_single<false>(
            packed, input, input_stride, output, output_stride,
            paired_columns, plan.destination_size);
    }
}

void solve_columns_b1_pair(const detail::PackedCpuPlan &packed,
                           const float *input, std::ptrdiff_t input_stride,
                           float *output, std::ptrdiff_t output_stride,
                           std::int32_t column,
                           std::int32_t destination_size) noexcept {
    __m256 previous0 = _mm256_setzero_ps();
    __m256 previous1 = _mm256_setzero_ps();
    for (std::int32_t i = 0; i < destination_size; ++i) {
        __m256 value0;
        __m256 value1;
        multiply_columns_pair(
            packed, input, input_stride, i, column, value0, value1);
        const auto index = static_cast<std::size_t>(i);
        if (i >= 1) {
            const __m256 lower = _mm256_set1_ps(packed.lower_ld[index]);
            value0 = _mm256_fnmadd_ps(lower, previous0, value0);
            value1 = _mm256_fnmadd_ps(lower, previous1, value1);
        }
        const __m256 inverse = _mm256_set1_ps(packed.inverse_diagonal[index]);
        value0 = _mm256_mul_ps(value0, inverse);
        value1 = _mm256_mul_ps(value1, inverse);
        auto *destination = output + static_cast<std::ptrdiff_t>(i)
            * output_stride + column;
        _mm256_storeu_ps(destination, value0);
        _mm256_storeu_ps(destination + 8, value1);
        previous0 = value0;
        previous1 = value1;
    }

    if (destination_size < 2) return;
    const auto *last = output + static_cast<std::ptrdiff_t>(destination_size - 1)
        * output_stride + column;
    __m256 next0 = _mm256_loadu_ps(last);
    __m256 next1 = _mm256_loadu_ps(last + 8);
    for (std::int32_t i = destination_size - 2; i >= 0; --i) {
        auto *destination = output + static_cast<std::ptrdiff_t>(i)
            * output_stride + column;
        __m256 value0 = _mm256_loadu_ps(destination);
        __m256 value1 = _mm256_loadu_ps(destination + 8);
        const __m256 upper = _mm256_set1_ps(
            packed.upper_l[static_cast<std::size_t>(i)]);
        value0 = _mm256_fnmadd_ps(upper, next0, value0);
        value1 = _mm256_fnmadd_ps(upper, next1, value1);
        _mm256_storeu_ps(destination, value0);
        _mm256_storeu_ps(destination + 8, value1);
        next0 = value0;
        next1 = value1;
    }
}

void solve_columns_b1_single(const detail::PackedCpuPlan &packed,
                             const float *input, std::ptrdiff_t input_stride,
                             float *output, std::ptrdiff_t output_stride,
                             std::int32_t column,
                             std::int32_t destination_size) noexcept {
    __m256 previous = _mm256_setzero_ps();
    for (std::int32_t i = 0; i < destination_size; ++i) {
        __m256 value = multiply_columns_single(
            packed, input, input_stride, i, column);
        if (i >= 1) {
            value = _mm256_fnmadd_ps(
                _mm256_set1_ps(packed.lower_ld[static_cast<std::size_t>(i)]),
                previous, value);
        }
        value = _mm256_mul_ps(
            value, _mm256_set1_ps(
                       packed.inverse_diagonal[static_cast<std::size_t>(i)]));
        _mm256_storeu_ps(output + static_cast<std::ptrdiff_t>(i)
                         * output_stride + column, value);
        previous = value;
    }

    if (destination_size < 2) return;
    __m256 next = _mm256_loadu_ps(
        output + static_cast<std::ptrdiff_t>(destination_size - 1)
            * output_stride + column);
    for (std::int32_t i = destination_size - 2; i >= 0; --i) {
        auto *destination = output + static_cast<std::ptrdiff_t>(i)
            * output_stride + column;
        __m256 value = _mm256_loadu_ps(destination);
        value = _mm256_fnmadd_ps(
            _mm256_set1_ps(packed.upper_l[static_cast<std::size_t>(i)]),
            next, value);
        _mm256_storeu_ps(destination, value);
        next = value;
    }
}

void solve_columns_b1(const AxisPlan &plan,
                      const detail::PackedCpuPlan &packed,
                      const float *input, std::ptrdiff_t input_stride,
                      float *output, std::ptrdiff_t output_stride,
                      std::int32_t vector_columns) noexcept {
    const auto paired_columns = vector_columns & ~15;
    for (std::int32_t column = 0; column < paired_columns; column += 16) {
        solve_columns_b1_pair(
            packed, input, input_stride, output, output_stride,
            column, plan.destination_size);
    }
    if (paired_columns != vector_columns) {
        solve_columns_b1_single(
            packed, input, input_stride, output, output_stride,
            paired_columns, plan.destination_size);
    }
}

template <int FixedBandwidth>
void solve_columns_vector(const AxisPlan &plan,
                          const detail::PackedCpuPlan &packed,
                          const float *input, std::ptrdiff_t input_stride,
                          float *output, std::ptrdiff_t output_stride,
                          std::int32_t vector_columns) noexcept {
    // Keep one input/output column tile in the per-core L2 while the forward
    // and backward recurrences consume it.  A full-frame forward pass leaves
    // the intermediate rows cold before the backward pass at high concurrency.
    constexpr std::int32_t l2_column_tile = 32;
    constexpr std::int32_t frame_parallel_threshold = 1024;
    const auto column_tile = vector_columns >= frame_parallel_threshold
        ? l2_column_tile : vector_columns;
    const auto n = plan.destination_size;
    const auto factor_stride = static_cast<std::size_t>(packed.padded_destination_size);
    const auto bandwidth = FixedBandwidth == 0 ? plan.half_bandwidth : FixedBandwidth;
    for (std::int32_t tile = 0; tile < vector_columns; tile += column_tile) {
        const auto tile_end = std::min(tile + column_tile, vector_columns);
        for (std::int32_t i = 0; i < n; ++i) {
            const auto left = packed.weights_left[static_cast<std::size_t>(i)];
            const auto right = packed.weights_right[static_cast<std::size_t>(i)];
            const auto weight_base = static_cast<std::size_t>(i)
                * static_cast<std::size_t>(packed.weights_columns);
            for (std::int32_t column = tile; column < tile_end; column += 8) {
                __m256 value = _mm256_setzero_ps();
                for (std::int32_t source = left; source < right; ++source) {
                    value = _mm256_fmadd_ps(
                        _mm256_set1_ps(packed.weights[
                            weight_base + static_cast<std::size_t>(source - left)]),
                        _mm256_loadu_ps(input + static_cast<std::ptrdiff_t>(source)
                                        * input_stride + column), value);
                }
                const auto available = std::min(bandwidth, i);
                for (std::int32_t distance = available; distance >= 1; --distance) {
                    value = _mm256_fnmadd_ps(
                        _mm256_set1_ps(packed.lower_ld[
                            static_cast<std::size_t>(distance - 1) * factor_stride
                            + static_cast<std::size_t>(i)]),
                        _mm256_loadu_ps(output + static_cast<std::ptrdiff_t>(i - distance)
                                        * output_stride + column), value);
                }
                value = _mm256_mul_ps(
                    value, _mm256_set1_ps(
                               packed.inverse_diagonal[static_cast<std::size_t>(i)]));
                _mm256_storeu_ps(output + static_cast<std::ptrdiff_t>(i)
                                 * output_stride + column, value);
            }
        }
        for (std::int32_t i = n - 2; i >= 0; --i) {
            const auto available = std::min(bandwidth, n - i - 1);
            for (std::int32_t column = tile; column < tile_end; column += 8) {
                __m256 value = _mm256_loadu_ps(
                    output + static_cast<std::ptrdiff_t>(i) * output_stride + column);
                if constexpr (FixedBandwidth == 3) {
                    for (std::int32_t distance = 1; distance <= available; ++distance) {
                        value = _mm256_fnmadd_ps(
                            _mm256_set1_ps(packed.upper_l[
                                static_cast<std::size_t>(distance - 1) * factor_stride
                                + static_cast<std::size_t>(i)]),
                            _mm256_loadu_ps(output
                                + static_cast<std::ptrdiff_t>(i + distance)
                                    * output_stride + column), value);
                    }
                } else {
                    for (std::int32_t distance = available; distance >= 1; --distance) {
                        value = _mm256_fnmadd_ps(
                            _mm256_set1_ps(packed.upper_l[
                                static_cast<std::size_t>(distance - 1) * factor_stride
                                + static_cast<std::size_t>(i)]),
                            _mm256_loadu_ps(output
                                + static_cast<std::ptrdiff_t>(i + distance)
                                    * output_stride + column), value);
                    }
                }
                _mm256_storeu_ps(output + static_cast<std::ptrdiff_t>(i)
                                 * output_stride + column, value);
            }
        }
    }
}

void solve_columns_pair(const AxisPlan &plan,
                        const detail::PackedCpuPlan &packed,
                        const float *input, std::ptrdiff_t input_stride,
                        float *output, std::ptrdiff_t output_stride,
                        std::int32_t vector_columns) noexcept {
    const auto paired_columns = vector_columns & ~15;
    constexpr std::int32_t l2_column_tile = 32;
    constexpr std::int32_t frame_parallel_threshold = 1024;
    const auto column_tile = paired_columns >= frame_parallel_threshold
        ? l2_column_tile : paired_columns;
    const auto n = plan.destination_size;
    const auto factor_stride = static_cast<std::size_t>(
        packed.padded_destination_size);

    for (std::int32_t tile = 0; tile < paired_columns; tile += column_tile) {
        const auto tile_end = std::min(tile + column_tile, paired_columns);
        for (std::int32_t i = 0; i < n; ++i) {
            const auto index = static_cast<std::size_t>(i);
            const auto available = std::min(plan.half_bandwidth, i);
            for (std::int32_t column = tile; column < tile_end; column += 16) {
                __m256 value0;
                __m256 value1;
                multiply_columns_pair(
                    packed, input, input_stride, i, column, value0, value1);
                for (std::int32_t distance = available;
                     distance >= 1; --distance) {
                    const __m256 lower = _mm256_set1_ps(packed.lower_ld[
                        static_cast<std::size_t>(distance - 1) * factor_stride
                        + index]);
                    const auto *previous = output
                        + static_cast<std::ptrdiff_t>(i - distance)
                            * output_stride + column;
                    value0 = _mm256_fnmadd_ps(
                        lower, _mm256_loadu_ps(previous), value0);
                    value1 = _mm256_fnmadd_ps(
                        lower, _mm256_loadu_ps(previous + 8), value1);
                }
                const __m256 inverse = _mm256_set1_ps(
                    packed.inverse_diagonal[index]);
                value0 = _mm256_mul_ps(value0, inverse);
                value1 = _mm256_mul_ps(value1, inverse);
                auto *destination = output
                    + static_cast<std::ptrdiff_t>(i) * output_stride + column;
                _mm256_storeu_ps(destination, value0);
                _mm256_storeu_ps(destination + 8, value1);
            }
        }

        for (std::int32_t i = n - 2; i >= 0; --i) {
            const auto index = static_cast<std::size_t>(i);
            const auto available = std::min(plan.half_bandwidth, n - i - 1);
            for (std::int32_t column = tile; column < tile_end; column += 16) {
                auto *destination = output
                    + static_cast<std::ptrdiff_t>(i) * output_stride + column;
                __m256 value0 = _mm256_loadu_ps(destination);
                __m256 value1 = _mm256_loadu_ps(destination + 8);
                for (std::int32_t distance = available;
                     distance >= 1; --distance) {
                    const __m256 upper = _mm256_set1_ps(packed.upper_l[
                        static_cast<std::size_t>(distance - 1) * factor_stride
                        + index]);
                    const auto *next = output
                        + static_cast<std::ptrdiff_t>(i + distance)
                            * output_stride + column;
                    value0 = _mm256_fnmadd_ps(
                        upper, _mm256_loadu_ps(next), value0);
                    value1 = _mm256_fnmadd_ps(
                        upper, _mm256_loadu_ps(next + 8), value1);
                }
                _mm256_storeu_ps(destination, value0);
                _mm256_storeu_ps(destination + 8, value1);
            }
        }
    }

    if (paired_columns != vector_columns) {
        solve_columns_vector<0>(
            plan, packed, input + paired_columns, input_stride,
            output + paired_columns, output_stride,
            vector_columns - paired_columns);
    }
}

[[nodiscard]] bool source_reaches_destination_band(
    const detail::PackedCpuPlan &packed, std::int32_t source_row,
    std::int32_t first_destination_row,
    std::int32_t last_destination_row) noexcept {
    const auto begin = packed.source_offsets[static_cast<std::size_t>(source_row)];
    const auto end = packed.source_offsets[static_cast<std::size_t>(source_row) + 1U];
    const auto first = packed.source_destinations.begin()
        + static_cast<std::ptrdiff_t>(begin);
    const auto last = packed.source_destinations.begin()
        + static_cast<std::ptrdiff_t>(end);
    const auto found = std::lower_bound(first, last, first_destination_row);
    return found != last && *found < last_destination_row;
}

void accumulate_horizontal_block(
    const detail::PackedCpuPlan &packed_vertical,
    const float *horizontal_rows, std::ptrdiff_t horizontal_stride,
    std::int32_t block_row, std::int32_t first_source_row,
    std::int32_t last_source_row, float *output,
    std::ptrdiff_t output_stride, std::int32_t vector_columns,
    std::int32_t columns, std::int32_t first_destination_row,
    std::int32_t last_destination_row) noexcept {
    for (std::int32_t source_row = first_source_row;
         source_row < last_source_row; ++source_row) {
        const auto begin = packed_vertical.source_offsets[
            static_cast<std::size_t>(source_row)];
        const auto end = packed_vertical.source_offsets[
            static_cast<std::size_t>(source_row) + 1U];
        auto offset = static_cast<std::uint32_t>(std::lower_bound(
            packed_vertical.source_destinations.begin()
                + static_cast<std::ptrdiff_t>(begin),
            packed_vertical.source_destinations.begin()
                + static_cast<std::ptrdiff_t>(end),
            first_destination_row)
            - packed_vertical.source_destinations.begin());
        const auto *horizontal = horizontal_rows
            + static_cast<std::ptrdiff_t>(source_row - block_row)
                * horizontal_stride;
        for (; offset < end; ++offset) {
            const auto destination_row =
                packed_vertical.source_destinations[offset];
            if (destination_row >= last_destination_row) break;
            auto *rhs = output
                + static_cast<std::ptrdiff_t>(destination_row) * output_stride;
            const __m256 weight = _mm256_set1_ps(
                packed_vertical.source_weights[offset]);
            for (std::int32_t column = 0;
                 column < vector_columns; column += 8) {
                _mm256_storeu_ps(
                    rhs + column,
                    _mm256_fmadd_ps(
                        weight, _mm256_loadu_ps(horizontal + column),
                        _mm256_loadu_ps(rhs + column)));
            }
            const auto scalar_weight = packed_vertical.source_weights[offset];
            for (std::int32_t column = vector_columns;
                 column < columns; ++column) {
                rhs[column] += scalar_weight * horizontal[column];
            }
        }
    }
}

void solve_rhs_vectors(const AxisPlan &plan,
                       const detail::PackedCpuPlan &packed,
                       float *output, std::ptrdiff_t output_stride,
                       std::int32_t vector_columns) noexcept {
    if (plan.half_bandwidth == 3) {
        const auto paired_columns = vector_columns & ~15;
        for (std::int32_t column = 0;
             column < paired_columns; column += 16) {
            solve_columns_b3_pair<true>(
                packed, output, output_stride, output, output_stride,
                column, plan.destination_size);
        }
        if (paired_columns != vector_columns) {
            solve_columns_b3_single<true>(
                packed, output, output_stride, output, output_stride,
                paired_columns, plan.destination_size);
        }
        return;
    }
    const auto paired_columns = vector_columns & ~15;
    constexpr std::int32_t l2_column_tile = 32;
    constexpr std::int32_t frame_parallel_threshold = 1024;
    const auto column_tile = paired_columns >= frame_parallel_threshold
        ? l2_column_tile : std::max(paired_columns, 16);
    const auto n = plan.destination_size;
    const auto factor_stride = static_cast<std::size_t>(
        packed.padded_destination_size);

    for (std::int32_t tile = 0; tile < paired_columns; tile += column_tile) {
        const auto tile_end = std::min(tile + column_tile, paired_columns);
        for (std::int32_t i = 0; i < n; ++i) {
            const auto index = static_cast<std::size_t>(i);
            const auto available = std::min(plan.half_bandwidth, i);
            for (std::int32_t column = tile; column < tile_end; column += 16) {
                auto *destination = output
                    + static_cast<std::ptrdiff_t>(i) * output_stride + column;
                __m256 value0 = _mm256_loadu_ps(destination);
                __m256 value1 = _mm256_loadu_ps(destination + 8);
                for (std::int32_t distance = available;
                     distance >= 1; --distance) {
                    const __m256 lower = _mm256_set1_ps(packed.lower_ld[
                        static_cast<std::size_t>(distance - 1) * factor_stride
                        + index]);
                    const auto *previous = output
                        + static_cast<std::ptrdiff_t>(i - distance)
                            * output_stride + column;
                    value0 = _mm256_fnmadd_ps(
                        lower, _mm256_loadu_ps(previous), value0);
                    value1 = _mm256_fnmadd_ps(
                        lower, _mm256_loadu_ps(previous + 8), value1);
                }
                const __m256 inverse = _mm256_set1_ps(
                    packed.inverse_diagonal[index]);
                _mm256_storeu_ps(destination, _mm256_mul_ps(value0, inverse));
                _mm256_storeu_ps(
                    destination + 8, _mm256_mul_ps(value1, inverse));
            }
        }
        for (std::int32_t i = n - 2; i >= 0; --i) {
            const auto index = static_cast<std::size_t>(i);
            const auto available = std::min(plan.half_bandwidth, n - i - 1);
            for (std::int32_t column = tile; column < tile_end; column += 16) {
                auto *destination = output
                    + static_cast<std::ptrdiff_t>(i) * output_stride + column;
                __m256 value0 = _mm256_loadu_ps(destination);
                __m256 value1 = _mm256_loadu_ps(destination + 8);
                for (std::int32_t distance = available;
                     distance >= 1; --distance) {
                    const __m256 upper = _mm256_set1_ps(packed.upper_l[
                        static_cast<std::size_t>(distance - 1) * factor_stride
                        + index]);
                    const auto *next = output
                        + static_cast<std::ptrdiff_t>(i + distance)
                            * output_stride + column;
                    value0 = _mm256_fnmadd_ps(
                        upper, _mm256_loadu_ps(next), value0);
                    value1 = _mm256_fnmadd_ps(
                        upper, _mm256_loadu_ps(next + 8), value1);
                }
                _mm256_storeu_ps(destination, value0);
                _mm256_storeu_ps(destination + 8, value1);
            }
        }
    }

    if (paired_columns == vector_columns) return;
    const auto column = paired_columns;
    for (std::int32_t i = 0; i < n; ++i) {
        auto *destination = output
            + static_cast<std::ptrdiff_t>(i) * output_stride + column;
        __m256 value = _mm256_loadu_ps(destination);
        const auto index = static_cast<std::size_t>(i);
        const auto available = std::min(plan.half_bandwidth, i);
        for (std::int32_t distance = available; distance >= 1; --distance) {
            value = _mm256_fnmadd_ps(
                _mm256_set1_ps(packed.lower_ld[
                    static_cast<std::size_t>(distance - 1) * factor_stride
                    + index]),
                _mm256_loadu_ps(output
                    + static_cast<std::ptrdiff_t>(i - distance) * output_stride
                    + column), value);
        }
        _mm256_storeu_ps(
            destination,
            _mm256_mul_ps(value, _mm256_set1_ps(
                packed.inverse_diagonal[index])));
    }
    for (std::int32_t i = n - 2; i >= 0; --i) {
        auto *destination = output
            + static_cast<std::ptrdiff_t>(i) * output_stride + column;
        __m256 value = _mm256_loadu_ps(destination);
        const auto index = static_cast<std::size_t>(i);
        const auto available = std::min(plan.half_bandwidth, n - i - 1);
        if (plan.half_bandwidth == 3) {
            for (std::int32_t distance = 1; distance <= available; ++distance) {
                value = _mm256_fnmadd_ps(
                    _mm256_set1_ps(packed.upper_l[
                        static_cast<std::size_t>(distance - 1) * factor_stride
                        + index]),
                    _mm256_loadu_ps(output
                        + static_cast<std::ptrdiff_t>(i + distance)
                            * output_stride + column), value);
            }
        } else {
            for (std::int32_t distance = available; distance >= 1; --distance) {
                value = _mm256_fnmadd_ps(
                    _mm256_set1_ps(packed.upper_l[
                        static_cast<std::size_t>(distance - 1) * factor_stride
                        + index]),
                    _mm256_loadu_ps(output
                        + static_cast<std::ptrdiff_t>(i + distance)
                            * output_stride + column), value);
            }
        }
        _mm256_storeu_ps(destination, value);
    }
}

template <class Sample>
void accumulate_2d_integer_rhs_impl(
    const AxisPlan &horizontal,
    const detail::PackedCpuPlan &packed_horizontal,
    const detail::PackedCpuPlan &packed_vertical,
    const Sample *input, std::ptrdiff_t input_row_stride,
    float input_offset, float input_scale,
    float *output, std::ptrdiff_t output_row_stride,
    std::int32_t first_destination_row,
    std::int32_t last_destination_row) {
    const auto columns = horizontal.destination_size;
    const auto padded_columns = packed_horizontal.padded_destination_size;
    for (std::int32_t row = first_destination_row;
         row < last_destination_row; ++row) {
        std::fill_n(output + static_cast<std::ptrdiff_t>(row) * output_row_stride,
                    padded_columns, 0.0F);
    }

    thread_local std::vector<ScratchVector> transpose_scratch;
    thread_local std::vector<ScratchVector> horizontal_scratch;
    transpose_scratch.resize(
        static_cast<std::size_t>(packed_horizontal.padded_source_size));
    horizontal_scratch.resize(static_cast<std::size_t>(padded_columns));
    auto *transpose_data = transpose_scratch.front().lanes;
    auto *horizontal_rows = horizontal_scratch.front().lanes;

    const auto process_block = [&](std::int32_t block_row,
                                   std::int32_t first_source_row,
                                   std::int32_t last_source_row) {
        bool needed = false;
        for (std::int32_t source_row = first_source_row;
             source_row < last_source_row; ++source_row) {
            if (source_reaches_destination_band(
                    packed_vertical, source_row,
                    first_destination_row, last_destination_row)) {
                needed = true;
                break;
            }
        }
        if (!needed) return;
        solve_horizontal_integer_block(
            horizontal, packed_horizontal,
            input + static_cast<std::ptrdiff_t>(block_row) * input_row_stride,
            input_row_stride, input_offset, input_scale,
            horizontal_rows, padded_columns, transpose_data);
        accumulate_horizontal_block(
            packed_vertical, horizontal_rows, padded_columns,
            block_row, first_source_row, last_source_row,
            output, output_row_stride, padded_columns, columns,
            first_destination_row, last_destination_row);
    };

    const auto source_rows = packed_vertical.axis->source_size;
    const auto complete_rows = source_rows & ~7;
    const auto tail_row = complete_rows == source_rows
        ? source_rows : source_rows - 8;
    for (std::int32_t row = 0; row < complete_rows; row += 8) {
        const auto last_source_row = std::min(row + 8, tail_row);
        if (row < last_source_row) {
            process_block(row, row, last_source_row);
        }
    }
    if (complete_rows != source_rows) {
        process_block(tail_row, tail_row, source_rows);
    }
}

template <class Sample>
DSMVC_FLATTEN void forward_2d_rhs_destination_impl(
    const AxisPlan &horizontal,
    const detail::PackedCpuPlan &packed_horizontal,
    const AxisPlan &vertical,
    const detail::PackedCpuPlan &packed_vertical,
    const Sample *input, std::ptrdiff_t input_row_stride,
    float input_offset, float input_scale,
    float *output, std::ptrdiff_t output_row_stride,
    std::int32_t columns) {
    const auto padded_columns = packed_horizontal.padded_destination_size;
    const auto vector_columns = columns & ~7;
    const auto cache_blocks = static_cast<std::size_t>(
        std::max(packed_vertical.streaming_cache_blocks, 1));

    thread_local std::vector<ScratchVector> transpose_scratch;
    thread_local std::vector<ScratchVector> horizontal_cache;
    thread_local std::vector<std::int32_t> cache_rows;
    thread_local std::vector<std::uint64_t> cache_ages;
    thread_local std::vector<const float *> source_rows;
    transpose_scratch.resize(
        static_cast<std::size_t>(packed_horizontal.padded_source_size));
    horizontal_cache.resize(detail::checked_size_product(
        cache_blocks, static_cast<std::size_t>(padded_columns),
        "AVX2 horizontal cache"));
    cache_rows.assign(cache_blocks, -1);
    cache_ages.assign(cache_blocks, 0U);
    auto *transpose_data = transpose_scratch.front().lanes;
    std::uint64_t age = 0U;

    const auto tail_block = vertical.source_size & 7
        ? vertical.source_size - 8 : vertical.source_size;
    const auto source_block = [&](std::int32_t source) noexcept {
        return source >= tail_block ? tail_block : source & ~7;
    };
    const auto get_source_row = [&](std::int32_t source) -> const float * {
        const auto block_row = source_block(source);
        std::size_t slot = cache_blocks;
        for (std::size_t candidate = 0;
             candidate < cache_blocks; ++candidate) {
            if (cache_rows[candidate] == block_row) {
                slot = candidate;
                break;
            }
        }
        if (slot == cache_blocks) {
            slot = 0U;
            for (std::size_t candidate = 0;
                 candidate < cache_blocks; ++candidate) {
                if (cache_rows[candidate] < 0) {
                    slot = candidate;
                    break;
                }
                if (cache_ages[candidate] < cache_ages[slot]) slot = candidate;
            }
            auto *horizontal_rows = horizontal_cache[
                slot * static_cast<std::size_t>(padded_columns)].lanes;
            if constexpr (std::is_same_v<Sample, float>) {
                solve_horizontal_block(
                    horizontal, packed_horizontal,
                    input + static_cast<std::ptrdiff_t>(block_row)
                        * input_row_stride,
                    input_row_stride, horizontal_rows, padded_columns,
                    transpose_data);
            } else {
                solve_horizontal_integer_block(
                    horizontal, packed_horizontal,
                    input + static_cast<std::ptrdiff_t>(block_row)
                        * input_row_stride,
                    input_row_stride, input_offset, input_scale,
                    horizontal_rows, padded_columns, transpose_data);
            }
            cache_rows[slot] = block_row;
        }
        cache_ages[slot] = ++age;
        return horizontal_cache[
            slot * static_cast<std::size_t>(padded_columns)].lanes
            + static_cast<std::ptrdiff_t>(source - block_row) * padded_columns;
    };

    const auto factor_stride = static_cast<std::size_t>(
        packed_vertical.padded_destination_size);
    for (std::int32_t row = 0; row < vertical.destination_size; ++row) {
        const auto begin = vertical.transpose_offsets[
            static_cast<std::size_t>(row)];
        const auto end = vertical.transpose_offsets[
            static_cast<std::size_t>(row) + 1U];
        source_rows.resize(static_cast<std::size_t>(end - begin));
        for (auto offset = begin; offset < end; ++offset) {
            source_rows[static_cast<std::size_t>(offset - begin)] =
                get_source_row(vertical.transpose_indices[offset]);
        }

        auto *destination = output
            + static_cast<std::ptrdiff_t>(row) * output_row_stride;
        const auto available = std::min(vertical.half_bandwidth, row);
        for (std::int32_t column = 0;
             column < vector_columns; column += 8) {
            __m256 value = _mm256_setzero_ps();
            for (auto offset = begin; offset < end; ++offset) {
                value = _mm256_fmadd_ps(
                    _mm256_set1_ps(vertical.transpose_weights[offset]),
                    _mm256_loadu_ps(source_rows[
                        static_cast<std::size_t>(offset - begin)] + column),
                    value);
            }
            for (std::int32_t distance = available;
                 distance >= 1; --distance) {
                value = _mm256_fnmadd_ps(
                    _mm256_set1_ps(packed_vertical.lower_ld[
                        static_cast<std::size_t>(distance - 1) * factor_stride
                        + static_cast<std::size_t>(row)]),
                    _mm256_loadu_ps(output
                        + static_cast<std::ptrdiff_t>(row - distance)
                            * output_row_stride + column),
                    value);
            }
            _mm256_storeu_ps(
                destination + column,
                _mm256_mul_ps(value, _mm256_set1_ps(
                    packed_vertical.inverse_diagonal[
                        static_cast<std::size_t>(row)])));
        }
        for (std::int32_t column = vector_columns;
             column < columns; ++column) {
            float value = 0.0F;
            for (auto offset = begin; offset < end; ++offset) {
                value += vertical.transpose_weights[offset]
                    * source_rows[static_cast<std::size_t>(offset - begin)][column];
            }
            for (std::int32_t distance = available;
                 distance >= 1; --distance) {
                value -= vertical.lower_ld[
                    static_cast<std::size_t>(distance - 1)
                        * static_cast<std::size_t>(vertical.destination_size)
                    + static_cast<std::size_t>(row)]
                    * output[static_cast<std::ptrdiff_t>(row - distance)
                        * output_row_stride + column];
            }
            destination[column] = value * vertical.inverse_diagonal[
                static_cast<std::size_t>(row)];
        }
    }
}

template <class Sample>
DSMVC_FORCE_INLINE void store_integer_vector(
    __m256 value, Sample *destination,
    const __m256 &scale, const __m256 &offset,
    const __m256 &minimum, const __m256 &maximum) noexcept {
    value = _mm256_add_ps(_mm256_mul_ps(value, scale), offset);
    value = _mm256_min_ps(_mm256_max_ps(value, minimum), maximum);
    const __m256i integers = _mm256_cvtps_epi32(value);
    const __m128i packed = _mm_packus_epi32(
        _mm256_castsi256_si128(integers),
        _mm256_extracti128_si256(integers, 1));
    if constexpr (std::is_same_v<Sample, std::uint8_t>) {
        const __m128i bytes = _mm_packus_epi16(packed, _mm_setzero_si128());
        _mm_storel_epi64(reinterpret_cast<__m128i *>(destination), bytes);
    } else {
        _mm_storeu_si128(reinterpret_cast<__m128i *>(destination), packed);
    }
}

template <class Sample>
DSMVC_FLATTEN void backward_rhs_impl(
    const AxisPlan &plan, const detail::PackedCpuPlan &packed,
    float *input, std::ptrdiff_t input_row_stride,
    Sample *integer_output, std::ptrdiff_t integer_output_row_stride,
    std::int32_t columns, const IntegerConversion *conversion) noexcept {
    const auto vector_columns = columns & ~7;
    const auto n = plan.destination_size;
    const auto factor_stride = static_cast<std::size_t>(
        packed.padded_destination_size);
    const __m256 scale = _mm256_set1_ps(
        conversion ? conversion->output_scale : 1.0F);
    const __m256 offset = _mm256_set1_ps(
        conversion ? conversion->output_offset : 0.0F);
    const __m256 minimum = _mm256_setzero_ps();
    const __m256 maximum = _mm256_set1_ps(static_cast<float>(
        conversion ? conversion->output_maximum : 0U));

    if constexpr (!std::is_same_v<Sample, float>) {
        const auto *last = input
            + static_cast<std::ptrdiff_t>(n - 1) * input_row_stride;
        auto *last_output = integer_output
            + static_cast<std::ptrdiff_t>(n - 1) * integer_output_row_stride;
        for (std::int32_t column = 0;
             column < vector_columns; column += 8) {
            store_integer_vector(
                _mm256_loadu_ps(last + column), last_output + column,
                scale, offset, minimum, maximum);
        }
        for (std::int32_t column = vector_columns;
             column < columns; ++column) {
            const auto scaled = std::clamp(
                last[column] * conversion->output_scale
                    + conversion->output_offset,
                0.0F, static_cast<float>(conversion->output_maximum));
            last_output[column] = static_cast<Sample>(std::nearbyint(scaled));
        }
    }

    for (std::int32_t row = n - 2; row >= 0; --row) {
        auto *destination = input
            + static_cast<std::ptrdiff_t>(row) * input_row_stride;
        auto *integer_destination = [&]() -> Sample * {
            if constexpr (std::is_same_v<Sample, float>) {
                return nullptr;
            } else {
                return integer_output
                    + static_cast<std::ptrdiff_t>(row)
                        * integer_output_row_stride;
            }
        }();
        const auto available = std::min(plan.half_bandwidth, n - row - 1);
        for (std::int32_t column = 0;
             column < vector_columns; column += 8) {
            __m256 value = _mm256_loadu_ps(destination + column);
            if (plan.half_bandwidth == 3) {
                for (std::int32_t distance = 1;
                     distance <= available; ++distance) {
                    value = _mm256_fnmadd_ps(
                        _mm256_set1_ps(packed.upper_l[
                            static_cast<std::size_t>(distance - 1) * factor_stride
                            + static_cast<std::size_t>(row)]),
                        _mm256_loadu_ps(input
                            + static_cast<std::ptrdiff_t>(row + distance)
                                * input_row_stride + column),
                        value);
                }
            } else {
                for (std::int32_t distance = available;
                     distance >= 1; --distance) {
                    value = _mm256_fnmadd_ps(
                        _mm256_set1_ps(packed.upper_l[
                            static_cast<std::size_t>(distance - 1) * factor_stride
                            + static_cast<std::size_t>(row)]),
                        _mm256_loadu_ps(input
                            + static_cast<std::ptrdiff_t>(row + distance)
                                * input_row_stride + column),
                        value);
                }
            }
            _mm256_storeu_ps(destination + column, value);
            if constexpr (!std::is_same_v<Sample, float>) {
                store_integer_vector(
                    value, integer_destination + column,
                    scale, offset, minimum, maximum);
            }
        }
        for (std::int32_t column = vector_columns;
             column < columns; ++column) {
            float sum = 0.0F;
            if (plan.half_bandwidth == 3) {
                for (std::int32_t distance = 1;
                     distance <= available; ++distance) {
                    sum += plan.upper_l[
                        static_cast<std::size_t>(distance - 1)
                            * static_cast<std::size_t>(n)
                        + static_cast<std::size_t>(row)]
                        * input[static_cast<std::ptrdiff_t>(row + distance)
                            * input_row_stride + column];
                }
            } else {
                for (std::int32_t distance = available;
                     distance >= 1; --distance) {
                    sum += plan.upper_l[
                        static_cast<std::size_t>(distance - 1)
                            * static_cast<std::size_t>(n)
                        + static_cast<std::size_t>(row)]
                        * input[static_cast<std::ptrdiff_t>(row + distance)
                            * input_row_stride + column];
                }
            }
            destination[column] -= sum;
            if constexpr (!std::is_same_v<Sample, float>) {
                const auto scaled = std::clamp(
                    destination[column] * conversion->output_scale
                        + conversion->output_offset,
                    0.0F, static_cast<float>(conversion->output_maximum));
                integer_destination[column] =
                    static_cast<Sample>(std::nearbyint(scaled));
            }
        }
    }
}

template <class Sample>
void convert_rhs_to_integer_impl(
    const float *input, std::ptrdiff_t input_row_stride,
    Sample *output, std::ptrdiff_t output_row_stride,
    std::int32_t rows, std::int32_t columns,
    const IntegerConversion &conversion) noexcept {
    const __m256 scale = _mm256_set1_ps(conversion.output_scale);
    const __m256 offset = _mm256_set1_ps(conversion.output_offset);
    const __m256 minimum = _mm256_setzero_ps();
    const __m256 maximum = _mm256_set1_ps(
        static_cast<float>(conversion.output_maximum));
    const auto vector_columns = columns & ~7;
    for (std::int32_t row = 0; row < rows; ++row) {
        const auto *source = input
            + static_cast<std::ptrdiff_t>(row) * input_row_stride;
        auto *destination = output
            + static_cast<std::ptrdiff_t>(row) * output_row_stride;
        for (std::int32_t column = 0;
             column < vector_columns; column += 8) {
            __m256 value = _mm256_add_ps(
                _mm256_mul_ps(_mm256_loadu_ps(source + column), scale),
                offset);
            value = _mm256_min_ps(_mm256_max_ps(value, minimum), maximum);
            const __m256i integers = _mm256_cvtps_epi32(value);
            const __m128i packed = _mm_packus_epi32(
                _mm256_castsi256_si128(integers),
                _mm256_extracti128_si256(integers, 1));
            if constexpr (std::is_same_v<Sample, std::uint8_t>) {
                const __m128i bytes = _mm_packus_epi16(
                    packed, _mm_setzero_si128());
                _mm_storel_epi64(
                    reinterpret_cast<__m128i *>(destination + column), bytes);
            } else {
                _mm_storeu_si128(
                    reinterpret_cast<__m128i *>(destination + column), packed);
            }
        }
        for (std::int32_t column = vector_columns;
             column < columns; ++column) {
            const auto scaled = std::clamp(
                source[column] * conversion.output_scale
                    + conversion.output_offset,
                0.0F, static_cast<float>(conversion.output_maximum));
            destination[column] = static_cast<Sample>(std::nearbyint(scaled));
        }
    }
}

} // namespace

void inverse_rows_f64_avx2(
    const AxisPlan &plan,
    const float *input, std::ptrdiff_t input_row_stride,
    float *output, std::ptrdiff_t output_row_stride,
    std::int32_t row_count) {
    inverse_rows_f64_avx2_impl(
        plan, input, input_row_stride,
        output, output_row_stride, row_count);
}

void inverse_rows_to_f64_avx2(
    const AxisPlan &plan,
    const float *input, std::ptrdiff_t input_row_stride,
    double *output, std::ptrdiff_t output_row_stride,
    std::int32_t row_count) {
    inverse_rows_f64_avx2_impl(
        plan, input, input_row_stride,
        output, output_row_stride, row_count);
}

void inverse_columns_f64_avx2(
    const AxisPlan &plan,
    const float *input, std::ptrdiff_t input_row_stride,
    float *output, std::ptrdiff_t output_row_stride,
    std::int32_t column_count) {
    inverse_columns_f64_avx2_impl(
        plan, input, input_row_stride,
        output, output_row_stride, column_count);
}

void inverse_columns_from_f64_avx2(
    const AxisPlan &plan,
    const double *input, std::ptrdiff_t input_row_stride,
    float *output, std::ptrdiff_t output_row_stride,
    std::int32_t column_count) {
    inverse_columns_f64_avx2_impl(
        plan, input, input_row_stride,
        output, output_row_stride, column_count);
}

void forward_2d_rhs_avx2(
    const AxisPlan &horizontal,
    const detail::PackedCpuPlan &packed_horizontal,
    const AxisPlan &vertical,
    const detail::PackedCpuPlan &packed_vertical,
    const float *input, std::ptrdiff_t input_row_stride,
    float *output, std::ptrdiff_t output_row_stride,
    std::int32_t columns) {
    forward_2d_rhs_destination_impl(
        horizontal, packed_horizontal, vertical, packed_vertical,
        input, input_row_stride, 0.0F, 1.0F,
        output, output_row_stride, columns);
}

void forward_2d_rhs_u8_avx2(
    const AxisPlan &horizontal,
    const detail::PackedCpuPlan &packed_horizontal,
    const AxisPlan &vertical,
    const detail::PackedCpuPlan &packed_vertical,
    const std::uint8_t *input, std::ptrdiff_t input_row_stride,
    const IntegerConversion &conversion,
    float *output, std::ptrdiff_t output_row_stride,
    std::int32_t columns) {
    forward_2d_rhs_destination_impl(
        horizontal, packed_horizontal, vertical, packed_vertical,
        input, input_row_stride,
        conversion.input_offset, conversion.input_scale,
        output, output_row_stride, columns);
}

void forward_2d_rhs_u16_avx2(
    const AxisPlan &horizontal,
    const detail::PackedCpuPlan &packed_horizontal,
    const AxisPlan &vertical,
    const detail::PackedCpuPlan &packed_vertical,
    const std::uint16_t *input, std::ptrdiff_t input_row_stride,
    const IntegerConversion &conversion,
    float *output, std::ptrdiff_t output_row_stride,
    std::int32_t columns) {
    forward_2d_rhs_destination_impl(
        horizontal, packed_horizontal, vertical, packed_vertical,
        input, input_row_stride,
        conversion.input_offset, conversion.input_scale,
        output, output_row_stride, columns);
}

void backward_rhs_avx2(
    const AxisPlan &plan, const detail::PackedCpuPlan &packed,
    float *output, std::ptrdiff_t output_row_stride,
    std::int32_t columns) {
    backward_rhs_impl<float>(
        plan, packed, output, output_row_stride,
        nullptr, 0, columns, nullptr);
}

void backward_rhs_to_u8_avx2(
    const AxisPlan &plan, const detail::PackedCpuPlan &packed,
    float *input, std::ptrdiff_t input_row_stride,
    std::uint8_t *output, std::ptrdiff_t output_row_stride,
    std::int32_t columns, const IntegerConversion &conversion) {
    backward_rhs_impl(
        plan, packed, input, input_row_stride,
        output, output_row_stride, columns, &conversion);
}

void backward_rhs_to_u16_avx2(
    const AxisPlan &plan, const detail::PackedCpuPlan &packed,
    float *input, std::ptrdiff_t input_row_stride,
    std::uint16_t *output, std::ptrdiff_t output_row_stride,
    std::int32_t columns, const IntegerConversion &conversion) {
    backward_rhs_impl(
        plan, packed, input, input_row_stride,
        output, output_row_stride, columns, &conversion);
}

void accumulate_2d_rhs_u8_avx2(
    const AxisPlan &horizontal,
    const detail::PackedCpuPlan &packed_horizontal,
    const detail::PackedCpuPlan &packed_vertical,
    const std::uint8_t *input, std::ptrdiff_t input_row_stride,
    const IntegerConversion &conversion,
    float *output, std::ptrdiff_t output_row_stride,
    std::int32_t first_destination_row,
    std::int32_t last_destination_row) {
    accumulate_2d_integer_rhs_impl(
        horizontal, packed_horizontal, packed_vertical,
        input, input_row_stride,
        conversion.input_offset, conversion.input_scale,
        output, output_row_stride,
        first_destination_row, last_destination_row);
}

void accumulate_2d_rhs_u16_avx2(
    const AxisPlan &horizontal,
    const detail::PackedCpuPlan &packed_horizontal,
    const detail::PackedCpuPlan &packed_vertical,
    const std::uint16_t *input, std::ptrdiff_t input_row_stride,
    const IntegerConversion &conversion,
    float *output, std::ptrdiff_t output_row_stride,
    std::int32_t first_destination_row,
    std::int32_t last_destination_row) {
    accumulate_2d_integer_rhs_impl(
        horizontal, packed_horizontal, packed_vertical,
        input, input_row_stride,
        conversion.input_offset, conversion.input_scale,
        output, output_row_stride,
        first_destination_row, last_destination_row);
}

void convert_rhs_to_u8_avx2(
    const float *input, std::ptrdiff_t input_row_stride,
    std::uint8_t *output, std::ptrdiff_t output_row_stride,
    std::int32_t rows, std::int32_t columns,
    const IntegerConversion &conversion) {
    convert_rhs_to_integer_impl(
        input, input_row_stride, output, output_row_stride,
        rows, columns, conversion);
}

void convert_rhs_to_u16_avx2(
    const float *input, std::ptrdiff_t input_row_stride,
    std::uint16_t *output, std::ptrdiff_t output_row_stride,
    std::int32_t rows, std::int32_t columns,
    const IntegerConversion &conversion) {
    convert_rhs_to_integer_impl(
        input, input_row_stride, output, output_row_stride,
        rows, columns, conversion);
}

void solve_rhs_columns_avx2(
    const AxisPlan &plan, const detail::PackedCpuPlan &packed,
    float *output, std::ptrdiff_t output_row_stride,
    std::int32_t vector_columns) {
    solve_rhs_vectors(
        plan, packed, output, output_row_stride, vector_columns);
}

void inverse_rows_avx2(const AxisPlan &plan,
                       const detail::PackedCpuPlan &packed,
                       const float *input, std::ptrdiff_t input_row_stride,
                       float *output, std::ptrdiff_t output_row_stride,
                       std::int32_t row_count) {
    if (row_count < 8) {
        for (std::int32_t row = 0; row < row_count; ++row) {
            dsmvc::inverse_axis_f32(
                plan, input + static_cast<std::ptrdiff_t>(row) * input_row_stride, 1,
                output + static_cast<std::ptrdiff_t>(row) * output_row_stride, 1);
        }
        return;
    }

    thread_local std::vector<ScratchVector> scratch;
    scratch.resize(static_cast<std::size_t>(packed.padded_source_size));
    auto *scratch_data = scratch.front().lanes;
    const bool use_output_scratch =
        plan.destination_size != packed.padded_destination_size;
    thread_local std::vector<ScratchVector> padded_output;
    if (use_output_scratch) {
        padded_output.resize(
            static_cast<std::size_t>(packed.padded_destination_size));
    }
    const auto solve_block = [&](std::int32_t row) {
        auto *block_output = use_output_scratch
            ? padded_output.front().lanes
            : output + static_cast<std::ptrdiff_t>(row) * output_row_stride;
        const auto block_output_stride = use_output_scratch
            ? static_cast<std::ptrdiff_t>(packed.padded_destination_size)
            : output_row_stride;
        solve_horizontal_block(
            plan, packed,
            input + static_cast<std::ptrdiff_t>(row) * input_row_stride,
            input_row_stride, block_output, block_output_stride, scratch_data);
        if (use_output_scratch) {
            for (std::int32_t lane = 0; lane < 8; ++lane) {
                std::copy_n(
                    block_output
                        + static_cast<std::ptrdiff_t>(lane)
                            * block_output_stride,
                    plan.destination_size,
                    output + static_cast<std::ptrdiff_t>(row + lane)
                        * output_row_stride);
            }
        }
    };
    const auto complete_rows = row_count & ~7;
    for (std::int32_t row = 0; row < complete_rows; row += 8) {
        solve_block(row);
    }
    if (complete_rows != row_count) {
        solve_block(row_count - 8);
    }
}

void inverse_columns_avx2(const AxisPlan &plan,
                          const detail::PackedCpuPlan &packed,
                          const float *input, std::ptrdiff_t input_row_stride,
                          float *output, std::ptrdiff_t output_row_stride,
                          std::int32_t column_count) {
    const auto vector_columns = column_count & ~7;
    if (plan.half_bandwidth == 1) {
        solve_columns_b1(plan, packed, input, input_row_stride, output,
                         output_row_stride, vector_columns);
    } else if (plan.half_bandwidth == 3) {
        solve_columns_b3(plan, packed, input, input_row_stride, output,
                         output_row_stride, vector_columns);
    } else if (plan.half_bandwidth == 5 || plan.half_bandwidth == 7) {
        solve_columns_pair(plan, packed, input, input_row_stride, output,
                           output_row_stride, vector_columns);
    } else {
        solve_columns_vector<0>(plan, packed, input, input_row_stride, output,
                                output_row_stride, vector_columns);
    }
    for (std::int32_t column = vector_columns; column < column_count; ++column) {
        dsmvc::inverse_axis_f32(plan, input + column, input_row_stride,
                               output + column, output_row_stride);
    }
}

} // namespace dsmvc

#undef DSMVC_FORCE_INLINE
#undef DSMVC_FLATTEN
