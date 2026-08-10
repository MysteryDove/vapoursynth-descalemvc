#pragma once

#include "cuda_kernel.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>

namespace dsmvc::cuda_detail::cuda_launch {

[[nodiscard]] cudaError_t transpose(
    const float *source, std::uint32_t width, std::uint32_t height,
    float *destination, cudaStream_t stream);
[[nodiscard]] cudaError_t transpose(
    const std::uint8_t *source, std::uint32_t width, std::uint32_t height,
    cuda_kernel::IntegerConversionDescriptor conversion,
    float *destination, cudaStream_t stream);
[[nodiscard]] cudaError_t transpose(
    const std::uint16_t *source, std::uint32_t width, std::uint32_t height,
    cuda_kernel::IntegerConversionDescriptor conversion,
    float *destination, cudaStream_t stream);

[[nodiscard]] cudaError_t transpose_f64(
    const float *source, std::uint32_t width, std::uint32_t height,
    double *destination, cudaStream_t stream);
[[nodiscard]] cudaError_t transpose_f64(
    const std::uint8_t *source, std::uint32_t width, std::uint32_t height,
    cuda_kernel::IntegerConversionDescriptor conversion,
    double *destination, cudaStream_t stream);
[[nodiscard]] cudaError_t transpose_f64(
    const std::uint16_t *source, std::uint32_t width, std::uint32_t height,
    cuda_kernel::IntegerConversionDescriptor conversion,
    double *destination, cudaStream_t stream);
[[nodiscard]] cudaError_t promote_f64(
    const float *source, std::uint32_t element_count,
    double *destination, cudaStream_t stream);
[[nodiscard]] cudaError_t check_finite_f64(
    const double *source, std::uint32_t element_count,
    std::uint32_t *nonfinite, cudaStream_t stream);

[[nodiscard]] cudaError_t inverse_horizontal(
    const float *source, std::uint32_t vector_count,
    cuda_kernel::AxisPlanDescriptor plan,
    const std::uint32_t *offsets, const std::int32_t *indices,
    const float *weights, const float *lower, const float *upper,
    const float *diagonal, float *output, bool column_major,
    unsigned int threads, unsigned int shared_bytes, cudaStream_t stream);
[[nodiscard]] cudaError_t inverse_vertical(
    const float *source, std::uint32_t source_width,
    cuda_kernel::AxisPlanDescriptor plan,
    const std::uint32_t *offsets, const std::int32_t *indices,
    const float *weights, const float *lower, const float *upper,
    const float *diagonal, float *output, unsigned int threads,
    cudaStream_t stream);

[[nodiscard]] cudaError_t rhs_horizontal(
    const float *source, std::uint32_t vector_count,
    cuda_kernel::AxisPlanDescriptor plan,
    const std::uint32_t *offsets, const std::int32_t *indices,
    const float *weights, float *output, bool column_major,
    cudaStream_t stream);
[[nodiscard]] cudaError_t rhs_vertical(
    const float *source, std::uint32_t source_width,
    cuda_kernel::AxisPlanDescriptor plan,
    const std::uint32_t *offsets, const std::int32_t *indices,
    const float *weights, float *output, cudaStream_t stream);

[[nodiscard]] cudaError_t solve_horizontal(
    std::uint32_t vector_count, cuda_kernel::AxisPlanDescriptor plan,
    const float *lower, const float *upper, const float *diagonal,
    float *output, bool column_major, unsigned int threads,
    unsigned int shared_bytes, cudaStream_t stream);
[[nodiscard]] cudaError_t solve_vertical(
    std::uint32_t source_width, cuda_kernel::AxisPlanDescriptor plan,
    const float *lower, const float *upper, const float *diagonal,
    float *output, unsigned int threads, cudaStream_t stream);

[[nodiscard]] cudaError_t inverse_horizontal_f64(
    const double *source, std::uint32_t vector_count,
    cuda_kernel::AxisPlanDescriptor plan,
    const std::uint32_t *offsets, const std::int32_t *indices,
    const float *weights_f32, const float *lower_f32,
    const float *upper_f32, const float *diagonal_f32,
    const double *weights_f64, const double *bands_f64,
    cuda_kernel::PlanPrecision precision, double *output,
    unsigned int threads, cudaStream_t stream);
[[nodiscard]] cudaError_t inverse_vertical_f64(
    const double *source, std::uint32_t source_width,
    cuda_kernel::AxisPlanDescriptor plan,
    const std::uint32_t *offsets, const std::int32_t *indices,
    const float *weights_f32, const float *lower_f32,
    const float *upper_f32, const float *diagonal_f32,
    const double *weights_f64, const double *bands_f64,
    cuda_kernel::PlanPrecision precision, double *output,
    unsigned int threads, cudaStream_t stream);
[[nodiscard]] cudaError_t rhs_horizontal_f64(
    const double *source, std::uint32_t vector_count,
    cuda_kernel::AxisPlanDescriptor plan,
    const std::uint32_t *offsets, const std::int32_t *indices,
    const float *weights_f32, const double *weights_f64,
    cuda_kernel::PlanPrecision precision, double *output,
    cudaStream_t stream);
[[nodiscard]] cudaError_t rhs_vertical_f64(
    const double *source, std::uint32_t source_width,
    cuda_kernel::AxisPlanDescriptor plan,
    const std::uint32_t *offsets, const std::int32_t *indices,
    const float *weights_f32, const double *weights_f64,
    cuda_kernel::PlanPrecision precision, double *output,
    cudaStream_t stream);
[[nodiscard]] cudaError_t solve_horizontal_f64(
    std::uint32_t vector_count, cuda_kernel::AxisPlanDescriptor plan,
    const float *lower_f32, const float *upper_f32,
    const float *diagonal_f32, const double *bands_f64,
    cuda_kernel::PlanPrecision precision, double *output,
    unsigned int threads, cudaStream_t stream);
[[nodiscard]] cudaError_t solve_vertical_f64(
    std::uint32_t source_width, cuda_kernel::AxisPlanDescriptor plan,
    const float *lower_f32, const float *upper_f32,
    const float *diagonal_f32, const double *bands_f64,
    cuda_kernel::PlanPrecision precision, double *output,
    unsigned int threads, cudaStream_t stream);

[[nodiscard]] cudaError_t convert(
    const float *source, std::uint32_t element_count,
    cuda_kernel::IntegerConversionDescriptor conversion,
    std::uint8_t *output, cudaStream_t stream);
[[nodiscard]] cudaError_t convert(
    const float *source, std::uint32_t element_count,
    cuda_kernel::IntegerConversionDescriptor conversion,
    std::uint16_t *output, cudaStream_t stream);
[[nodiscard]] cudaError_t convert_f64(
    const double *source, std::uint32_t element_count,
    float *output, std::uint32_t *nonfinite, cudaStream_t stream);
[[nodiscard]] cudaError_t convert_f64(
    const double *source, std::uint32_t element_count,
    cuda_kernel::IntegerConversionDescriptor conversion,
    std::uint8_t *output, std::uint32_t *nonfinite, cudaStream_t stream);
[[nodiscard]] cudaError_t convert_f64(
    const double *source, std::uint32_t element_count,
    cuda_kernel::IntegerConversionDescriptor conversion,
    std::uint16_t *output, std::uint32_t *nonfinite, cudaStream_t stream);

} // namespace dsmvc::cuda_detail::cuda_launch
