#include <dsmvc/engine.hpp>

#include "cuda/cuda_executor.hpp"
#include "numerical_conformance.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace {

void require(bool condition, std::string message) {
    if (!condition) throw std::runtime_error(std::move(message));
}

void require_float_agreement(
    const std::vector<float> &expected, const std::vector<float> &actual,
    std::int32_t rows, std::int32_t columns, std::int32_t stride,
    float padding, std::string_view label) {
    std::uint32_t maximum_ulp = 0U;
    for (std::int32_t row = 0; row < rows; ++row) {
        for (std::int32_t column = 0; column < columns; ++column) {
            const auto index = static_cast<std::size_t>(row)
                    * static_cast<std::size_t>(stride)
                + static_cast<std::size_t>(column);
            require(
                std::isfinite(actual[index]),
                std::string(label) + " produced nonfinite output");
            maximum_ulp = std::max(
                maximum_ulp,
                dsmvc::numerical::float_ulp_distance(
                    expected[index], actual[index]));
        }
        for (std::int32_t column = columns; column < stride; ++column) {
            const auto index = static_cast<std::size_t>(row)
                    * static_cast<std::size_t>(stride)
                + static_cast<std::size_t>(column);
            require(
                actual[index] == padding,
                std::string(label) + " overwrote output padding");
        }
    }
    require(
        maximum_ulp <= 1U,
        std::string(label) + " differs from scalar F64 by "
            + std::to_string(maximum_ulp) + " output ULP");
    std::cout << label << " max_ulp=" << maximum_ulp << '\n';
}

std::vector<float> make_padded_float_input(
    std::int32_t rows, std::int32_t columns, std::int32_t stride,
    std::uint32_t seed) {
    std::vector<float> result(
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(stride),
        13.0F);
    const auto packed = dsmvc::numerical::make_normal_input(
        static_cast<std::size_t>(rows) * static_cast<std::size_t>(columns),
        seed);
    for (std::int32_t row = 0; row < rows; ++row) {
        std::copy_n(
            packed.data() + static_cast<std::ptrdiff_t>(row) * columns,
            columns,
            result.data() + static_cast<std::ptrdiff_t>(row) * stride);
    }
    return result;
}

void test_f32_plan_identity() {
    for (const auto &fixture : dsmvc::numerical::axis_fixtures()) {
        const auto plan = dsmvc::build_axis_plan(fixture.request);
        require(
            dsmvc::numerical::f32_plan_hash(fixture.request, plan)
                == fixture.f32_plan_hash,
            std::string(fixture.name) + " F32 plan identity drifted");
    }
}

void test_concurrent_prepare(
    dsmvc::cuda_detail::CudaExecutor &cuda,
    const std::shared_ptr<const dsmvc::AxisPlan> &plan) {
    std::array<std::future<void>, 8> tasks;
    for (auto &task : tasks) {
        task = std::async(std::launch::async, [&] { cuda.prepare(plan); });
    }
    for (auto &task : tasks) task.get();
    cuda.seal();
}

void test_rows(
    dsmvc::cuda_detail::CudaExecutor &cuda,
    const dsmvc::CpuExecutor &scalar,
    const dsmvc::AxisPlan &plan,
    std::string_view label = "cuda-f64-rows-tail") {
    constexpr std::int32_t rows = 5;
    constexpr float padding = -29.0F;
    const auto input_stride = plan.source_size + 3;
    const auto output_stride = plan.destination_size + 5;
    const auto input = make_padded_float_input(
        rows, plan.source_size, input_stride, 0xc0da6401U);
    std::vector<float> expected(
        static_cast<std::size_t>(rows)
            * static_cast<std::size_t>(output_stride),
        padding);
    std::vector<float> actual(expected.size(), padding);
    scalar.inverse_rows(
        plan, input.data(), input_stride,
        expected.data(), output_stride, rows);
    cuda.inverse_rows(
        plan, input.data(), input_stride,
        actual.data(), output_stride, rows, {});
    require_float_agreement(
        expected, actual, rows, plan.destination_size, output_stride,
        padding, label);
}

void test_columns(
    dsmvc::cuda_detail::CudaExecutor &cuda,
    const dsmvc::CpuExecutor &scalar,
    const dsmvc::AxisPlan &plan) {
    constexpr std::int32_t columns = 11;
    constexpr float padding = -31.0F;
    const auto input_stride = columns + 4;
    const auto output_stride = columns + 6;
    const auto input = make_padded_float_input(
        plan.source_size, columns, input_stride, 0xc0da6402U);
    std::vector<float> expected(
        static_cast<std::size_t>(plan.destination_size)
            * static_cast<std::size_t>(output_stride),
        padding);
    std::vector<float> actual(expected.size(), padding);
    scalar.inverse_columns(
        plan, input.data(), input_stride,
        expected.data(), output_stride, columns);
    cuda.inverse_columns(
        plan, input.data(), input_stride,
        actual.data(), output_stride, columns, {});
    require_float_agreement(
        expected, actual, plan.destination_size, columns, output_stride,
        padding, "cuda-f64-columns-tail");
}

void test_2d_float(
    dsmvc::cuda_detail::CudaExecutor &cuda,
    const dsmvc::CpuExecutor &scalar,
    const dsmvc::AxisPlan &horizontal,
    const dsmvc::AxisPlan &vertical,
    std::string_view label,
    std::uint32_t seed) {
    constexpr float padding = -37.0F;
    const auto input_stride = horizontal.source_size + 3;
    const auto output_stride = horizontal.destination_size + 5;
    const auto input = make_padded_float_input(
        vertical.source_size, horizontal.source_size, input_stride, seed);
    std::vector<float> expected(
        static_cast<std::size_t>(vertical.destination_size)
            * static_cast<std::size_t>(output_stride),
        padding);
    std::vector<float> actual(expected.size(), padding);
    scalar.inverse_2d(
        horizontal, vertical, input.data(), input_stride,
        expected.data(), output_stride);
    cuda.inverse_2d(
        horizontal, vertical, input.data(), input_stride,
        actual.data(), output_stride, {});
    require_float_agreement(
        expected, actual, vertical.destination_size,
        horizontal.destination_size, output_stride, padding, label);
}

template <class Sample>
void test_2d_integer(
    dsmvc::cuda_detail::CudaExecutor &cuda,
    const dsmvc::CpuExecutor &scalar,
    const dsmvc::AxisPlan &horizontal,
    const dsmvc::AxisPlan &vertical,
    const dsmvc::IntegerConversion &conversion,
    std::string_view label) {
    constexpr Sample padding = std::numeric_limits<Sample>::max();
    const auto input_stride = horizontal.source_size + 3;
    const auto output_stride = horizontal.destination_size + 5;
    std::vector<Sample> input(
        static_cast<std::size_t>(vertical.source_size)
            * static_cast<std::size_t>(input_stride),
        Sample{});
    std::uint32_t state = 0x64f00d31U;
    for (std::int32_t row = 0; row < vertical.source_size; ++row) {
        for (std::int32_t column = 0;
             column < horizontal.source_size; ++column) {
            state = state * 1664525U + 1013904223U;
            input[static_cast<std::size_t>(row)
                      * static_cast<std::size_t>(input_stride)
                  + static_cast<std::size_t>(column)] =
                static_cast<Sample>(state % (conversion.output_maximum + 1U));
        }
    }
    std::vector<Sample> expected(
        static_cast<std::size_t>(vertical.destination_size)
            * static_cast<std::size_t>(output_stride),
        padding);
    std::vector<Sample> actual(expected.size(), padding);
    if constexpr (std::is_same_v<Sample, std::uint8_t>) {
        scalar.inverse_2d_u8(
            horizontal, vertical, input.data(), input_stride,
            expected.data(), output_stride, conversion);
        cuda.inverse_2d_u8(
            horizontal, vertical, input.data(), input_stride,
            actual.data(), output_stride, conversion, {});
    } else {
        scalar.inverse_2d_u16(
            horizontal, vertical, input.data(), input_stride,
            expected.data(), output_stride, conversion);
        cuda.inverse_2d_u16(
            horizontal, vertical, input.data(), input_stride,
            actual.data(), output_stride, conversion, {});
    }
    std::uint32_t maximum_error = 0U;
    for (std::int32_t row = 0; row < vertical.destination_size; ++row) {
        for (std::int32_t column = 0;
             column < horizontal.destination_size; ++column) {
            const auto index = static_cast<std::size_t>(row)
                    * static_cast<std::size_t>(output_stride)
                + static_cast<std::size_t>(column);
            const auto left = static_cast<std::uint32_t>(expected[index]);
            const auto right = static_cast<std::uint32_t>(actual[index]);
            maximum_error = std::max(
                maximum_error,
                left > right ? left - right : right - left);
        }
        for (std::int32_t column = horizontal.destination_size;
             column < output_stride; ++column) {
            const auto index = static_cast<std::size_t>(row)
                    * static_cast<std::size_t>(output_stride)
                + static_cast<std::size_t>(column);
            require(actual[index] == padding,
                    std::string(label) + " overwrote output padding");
        }
    }
    require(maximum_error <= 1U,
            std::string(label)
                + " differs from scalar F64 by more than one code");
    std::cout << label << " max_code_error=" << maximum_error << '\n';
}

void test_device_conformance() {
    const auto fixtures = dsmvc::numerical::axis_fixtures();
    auto risky = std::make_shared<const dsmvc::AxisPlan>(
        dsmvc::build_axis_plan(fixtures[6].request));
    auto forced = std::make_shared<const dsmvc::AxisPlan>(
        dsmvc::build_axis_plan(fixtures[5].request));
    auto safe = std::make_shared<const dsmvc::AxisPlan>(
        dsmvc::build_axis_plan(dsmvc::numerical::mixed_horizontal_request()));
    require(
        risky->requires_float64() && forced->requires_float64()
            && !safe->requires_float64(),
        "CUDA F64 fixtures selected unexpected precision");

    dsmvc::cuda_detail::CudaExecutor cuda;
    const dsmvc::CpuExecutor scalar(dsmvc::CpuPath::scalar);
    test_concurrent_prepare(cuda, risky);
    test_rows(cuda, scalar, *risky);
    test_columns(cuda, scalar, *forced);

    for (std::size_t index = 0U; index < 4U; ++index) {
        auto request = fixtures[index].request;
        request.f64_mode = dsmvc::F64Mode::float64_only;
        const auto plan = dsmvc::build_axis_plan(request);
        require(
            plan.requires_float64()
                && plan.half_bandwidth
                    == fixtures[index].expected_half_bandwidth,
            std::string(fixtures[index].name)
                + " forced-F64 bandwidth changed");
        test_rows(
            cuda, scalar, plan,
            "cuda-f64-rows-b" + std::to_string(plan.half_bandwidth));
    }

    dsmvc::AxisRequest generic_request;
    generic_request.source_size = 43;
    generic_request.destination_size = 31;
    generic_request.active_length = 30.75;
    generic_request.shift = 0.125;
    generic_request.kernel.kind = dsmvc::KernelKind::custom;
    generic_request.kernel.taps = 5;
    generic_request.border = dsmvc::BorderMode::symmetric;
    generic_request.f64_mode = dsmvc::F64Mode::float64_only;
    const auto generic = dsmvc::build_axis_plan(
        generic_request, [](double x) {
            return std::exp(-0.25 * x * x);
        });
    require(
        generic.requires_float64() && generic.half_bandwidth == 9,
        "custom forced-F64 plan did not select generic B9");
    test_rows(cuda, scalar, generic, "cuda-f64-rows-generic-b9");
    test_2d_float(
        cuda, scalar, *safe, *risky,
        "cuda-f64-2d-safe-risk", 0xc0da6411U);
    test_2d_float(
        cuda, scalar, *risky, *safe,
        "cuda-f64-2d-risk-safe", 0xc0da6412U);
    test_2d_float(
        cuda, scalar, *risky, *forced,
        "cuda-f64-2d-risk-risk", 0xc0da6413U);

    const dsmvc::IntegerConversion u8_conversion{
        16.0F, 1.0F / 219.0F, 219.0F, 16.0F, 255U};
    const dsmvc::IntegerConversion u10_conversion{
        64.0F, 1.0F / 876.0F, 876.0F, 64.0F, 1023U};
    const dsmvc::IntegerConversion u16_conversion{
        0.0F, 1.0F / 65535.0F, 65535.0F, 0.0F, 65535U};
    test_2d_integer<std::uint8_t>(
        cuda, scalar, *safe, *risky, u8_conversion, "cuda-f64-u8");
    test_2d_integer<std::uint16_t>(
        cuda, scalar, *risky, *safe, u10_conversion, "cuda-f64-u10");
    test_2d_integer<std::uint16_t>(
        cuda, scalar, *risky, *forced, u16_conversion, "cuda-f64-u16");
}

} // namespace

int main() {
    try {
        test_f32_plan_identity();
        if (!dsmvc::cuda_detail::backend_available()) {
            std::cout
                << "cuda_f64 conformance skipped: no CUDA device is available\n";
            return EXIT_SUCCESS;
        }
        test_device_conformance();
        std::cout << "cuda_f64 conformance tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "cuda_f64 conformance test failure: "
                  << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
