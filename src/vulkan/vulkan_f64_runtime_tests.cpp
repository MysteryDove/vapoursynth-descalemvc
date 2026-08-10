#include "vulkan_executor.hpp"

#include <dsmvc/engine.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

void require(bool condition, std::string message) {
    if (!condition) throw std::runtime_error(std::move(message));
}

[[nodiscard]] dsmvc::AxisRequest make_request(
    std::int32_t source, std::int32_t destination, double active_length,
    double shift, dsmvc::KernelKind kernel, std::int32_t taps,
    dsmvc::F64Mode mode) {
    dsmvc::AxisRequest request;
    request.source_size = source;
    request.destination_size = destination;
    request.active_length = active_length;
    request.shift = shift;
    request.kernel.kind = kernel;
    request.kernel.taps = taps;
    request.kernel.b = 0.0;
    request.kernel.c = 0.5;
    request.border = dsmvc::BorderMode::symmetric;
    request.f64_mode = mode;
    return request;
}

[[nodiscard]] std::uint32_t ordered_float_bits(float value) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    return (bits & 0x80000000U) != 0U ? ~bits : bits | 0x80000000U;
}

[[nodiscard]] std::uint32_t ulp_distance(float left, float right) {
    const std::uint32_t a = ordered_float_bits(left);
    const std::uint32_t b = ordered_float_bits(right);
    return a > b ? a - b : b - a;
}

void compare_float_matrix(
    std::string_view label, const std::vector<float> &reference,
    const std::vector<float> &candidate, std::int32_t rows,
    std::int32_t columns, std::int32_t stride) {
    std::uint32_t maximum_ulp = 0U;
    for (std::int32_t row = 0; row < rows; ++row) {
        for (std::int32_t column = 0; column < columns; ++column) {
            const std::size_t index = static_cast<std::size_t>(row)
                * static_cast<std::size_t>(stride)
                + static_cast<std::size_t>(column);
            require(std::isfinite(candidate[index]),
                    std::string{label} + " produced a non-finite value");
            maximum_ulp = std::max(
                maximum_ulp, ulp_distance(reference[index], candidate[index]));
        }
    }
    require(maximum_ulp <= 1U,
            std::string{label} + " differs from CPU scalar F64 by "
                + std::to_string(maximum_ulp) + " ULP");
    std::cout << label << ": max_ulp=" << maximum_ulp << '\n';
}

template <class Sample>
void compare_integer_matrix(
    std::string_view label, const std::vector<Sample> &reference,
    const std::vector<Sample> &candidate, std::int32_t rows,
    std::int32_t columns, std::int32_t stride) {
    std::uint32_t maximum_error = 0U;
    for (std::int32_t row = 0; row < rows; ++row) {
        for (std::int32_t column = 0; column < columns; ++column) {
            const std::size_t index = static_cast<std::size_t>(row)
                * static_cast<std::size_t>(stride)
                + static_cast<std::size_t>(column);
            const auto left = static_cast<std::uint32_t>(reference[index]);
            const auto right = static_cast<std::uint32_t>(candidate[index]);
            maximum_error = std::max(
                maximum_error,
                left > right ? left - right : right - left);
        }
    }
    require(maximum_error <= 1U,
            std::string{label}
                + " differs from CPU scalar F64 by more than one code");
    std::cout << label << ": max_code_error=" << maximum_error << '\n';
}

template <class T>
void fill_input(std::vector<T> &values, std::uint32_t seed, std::uint32_t mask) {
    std::uint32_t state = seed;
    for (T &value : values) {
        state ^= state << 13U;
        state ^= state >> 17U;
        state ^= state << 5U;
        if constexpr (std::is_same_v<T, float>) {
            value = static_cast<float>(state & mask)
                / static_cast<float>(mask);
        } else {
            value = static_cast<T>(state & mask);
        }
    }
}

void test_rows_and_columns(
    dsmvc::vulkan_detail::VulkanExecutor &vulkan,
    const dsmvc::CpuExecutor &cpu,
    const std::vector<std::shared_ptr<const dsmvc::AxisPlan>> &plans) {
    constexpr std::int32_t vectors = 5;
    for (std::size_t plan_index = 0U; plan_index < plans.size(); ++plan_index) {
        const auto &plan = *plans[plan_index];
        const std::int32_t input_stride = plan.source_size + 3;
        const std::int32_t output_stride = plan.destination_size + 4;
        auto input = std::make_shared<std::vector<float>>(
            static_cast<std::size_t>(vectors) * input_stride);
        fill_input(*input, 0x64000000U + static_cast<std::uint32_t>(plan_index),
                   0xffffU);
        std::vector<float> reference(
            static_cast<std::size_t>(vectors) * output_stride, -123.0F);
        std::vector<float> candidate(reference.size(), -123.0F);
        std::shared_ptr<const void> lifetime = input;
        cpu.inverse_rows(
            plan, input->data(), input_stride,
            reference.data(), output_stride, vectors);
        vulkan.inverse_rows(
            plan, input->data(), input_stride,
            candidate.data(), output_stride, vectors, lifetime);
        compare_float_matrix(
            "rows-b" + std::to_string(plan.half_bandwidth),
            reference, candidate, vectors, plan.destination_size, output_stride);

        std::fill(candidate.begin(), candidate.end(), -123.0F);
        vulkan.inverse_rows(
            plan, input->data(), input_stride,
            candidate.data(), output_stride, vectors, lifetime);
        compare_float_matrix(
            "rows-cache-b" + std::to_string(plan.half_bandwidth),
            reference, candidate, vectors, plan.destination_size, output_stride);

        const std::int32_t column_stride = vectors + 3;
        std::vector<float> column_input(
            static_cast<std::size_t>(plan.source_size) * column_stride);
        fill_input(column_input,
                   0x64100000U + static_cast<std::uint32_t>(plan_index), 0xffffU);
        std::vector<float> column_reference(
            static_cast<std::size_t>(plan.destination_size) * column_stride,
            -123.0F);
        std::vector<float> column_candidate(column_reference.size(), -123.0F);
        cpu.inverse_columns(
            plan, column_input.data(), column_stride,
            column_reference.data(), column_stride, vectors);
        vulkan.inverse_columns(
            plan, column_input.data(), column_stride,
            column_candidate.data(), column_stride, vectors, {});
        compare_float_matrix(
            "columns-b" + std::to_string(plan.half_bandwidth),
            column_reference, column_candidate,
            plan.destination_size, vectors, column_stride);
    }
}

void test_2d_float(
    std::string_view label,
    dsmvc::vulkan_detail::VulkanExecutor &vulkan,
    const dsmvc::CpuExecutor &cpu,
    const dsmvc::AxisPlan &horizontal, const dsmvc::AxisPlan &vertical) {
    const std::int32_t input_stride = horizontal.source_size + 3;
    const std::int32_t output_stride = horizontal.destination_size + 4;
    auto input = std::make_shared<std::vector<float>>(
        static_cast<std::size_t>(vertical.source_size) * input_stride);
    fill_input(*input, 0x2df64000U, 0xffffU);
    std::vector<float> reference(
        static_cast<std::size_t>(vertical.destination_size) * output_stride,
        -123.0F);
    std::vector<float> candidate(reference.size(), -123.0F);
    cpu.inverse_2d(
        horizontal, vertical, input->data(), input_stride,
        reference.data(), output_stride);
    std::shared_ptr<const void> lifetime = input;
    vulkan.inverse_2d(
        horizontal, vertical, input->data(), input_stride,
        candidate.data(), output_stride, lifetime);
    compare_float_matrix(
        label, reference, candidate, vertical.destination_size,
        horizontal.destination_size, output_stride);

    std::fill(candidate.begin(), candidate.end(), -123.0F);
    vulkan.inverse_2d(
        horizontal, vertical, input->data(), input_stride,
        candidate.data(), output_stride, lifetime);
    compare_float_matrix(
        std::string{label} + "-cache", reference, candidate,
        vertical.destination_size, horizontal.destination_size, output_stride);
}

template <class Sample>
void test_2d_integer(
    std::string_view label,
    dsmvc::vulkan_detail::VulkanExecutor &vulkan,
    const dsmvc::CpuExecutor &cpu,
    const dsmvc::AxisPlan &horizontal, const dsmvc::AxisPlan &vertical,
    const dsmvc::IntegerConversion &conversion, std::uint32_t mask) {
    const std::int32_t input_stride = horizontal.source_size + 3;
    const std::int32_t output_stride = horizontal.destination_size + 4;
    std::vector<Sample> input(
        static_cast<std::size_t>(vertical.source_size) * input_stride);
    fill_input(input, 0x1a7e6400U + mask, mask);
    std::vector<Sample> reference(
        static_cast<std::size_t>(vertical.destination_size) * output_stride,
        std::numeric_limits<Sample>::max());
    std::vector<Sample> candidate(reference.size(),
                                  std::numeric_limits<Sample>::max());
    if constexpr (std::is_same_v<Sample, std::uint8_t>) {
        cpu.inverse_2d_u8(
            horizontal, vertical, input.data(), input_stride,
            reference.data(), output_stride, conversion);
        vulkan.inverse_2d_u8(
            horizontal, vertical, input.data(), input_stride,
            candidate.data(), output_stride, conversion, {});
    } else {
        cpu.inverse_2d_u16(
            horizontal, vertical, input.data(), input_stride,
            reference.data(), output_stride, conversion);
        vulkan.inverse_2d_u16(
            horizontal, vertical, input.data(), input_stride,
            candidate.data(), output_stride, conversion, {});
    }
    compare_integer_matrix(
        label, reference, candidate, vertical.destination_size,
        horizontal.destination_size, output_stride);
}

void test_nonfinite_failure(
    dsmvc::vulkan_detail::VulkanExecutor &vulkan,
    const dsmvc::AxisPlan &plan) {
    std::vector<float> input(static_cast<std::size_t>(plan.source_size), 0.5F);
    std::vector<float> output(static_cast<std::size_t>(plan.destination_size));
    input[input.size() / 2U] = std::numeric_limits<float>::quiet_NaN();
    bool rejected = false;
    try {
        vulkan.inverse_rows(
            plan, input.data(), plan.source_size,
            output.data(), plan.destination_size, 1, {});
    } catch (const std::runtime_error &error) {
        rejected = std::string_view{error.what()}.find("NaN or infinity")
            != std::string_view::npos;
    }
    require(rejected, "Vulkan Float64 did not reject a NaN input");
    std::cout << "nonfinite-failure: rejected=1\n";
}

void test_unsupported_device(
    const dsmvc::vulkan_detail::VulkanFloat64Capabilities &capabilities) {
    require(!capabilities.strict_supported(),
            "unsupported test requires a device missing the strict contract");
    const auto safe = std::make_shared<const dsmvc::AxisPlan>(
        dsmvc::build_axis_plan(make_request(
            16, 16, 16.0, 0.0, dsmvc::KernelKind::bilinear, 3,
            dsmvc::F64Mode::float32_only)));
    const auto retained = std::make_shared<const dsmvc::AxisPlan>(
        dsmvc::build_axis_plan(make_request(
            17, 13, 12.75, 0.125, dsmvc::KernelKind::bicubic, 3,
            dsmvc::F64Mode::float64_only)));
    dsmvc::vulkan_detail::VulkanExecutor vulkan;
    const dsmvc::CpuExecutor cpu(dsmvc::CpuPath::scalar);
    std::vector<float> input(static_cast<std::size_t>(safe->source_size));
    fill_input(input, 0xf3200001U, 0xffffU);
    std::vector<float> reference(static_cast<std::size_t>(safe->destination_size));
    std::vector<float> candidate(reference.size());
    cpu.inverse_rows(
        *safe, input.data(), safe->source_size,
        reference.data(), safe->destination_size, 1);
    vulkan.inverse_rows(
        *safe, input.data(), safe->source_size,
        candidate.data(), safe->destination_size, 1, {});
    compare_float_matrix(
        "unsupported-device-f32", reference, candidate,
        1, safe->destination_size, safe->destination_size);

    bool rejected = false;
    try {
        vulkan.prepare(retained);
    } catch (const std::runtime_error &error) {
        rejected = error.what() == capabilities.requirement_error();
    }
    require(rejected,
            "retained Float64 plan did not report the exact missing capability");
    std::cout << "unsupported-device-f64: error=\""
              << capabilities.requirement_error() << "\"\n";
}

} // namespace

int main(int argc, char **argv) {
    try {
        std::cout << dsmvc::vulkan_detail::selected_float64_capability_report()
                  << '\n';
        const auto capabilities =
            dsmvc::vulkan_detail::selected_float64_capabilities();
        const bool expect_unsupported = argc == 2
            && std::string_view{argv[1]} == "--expect-unsupported";
        if (expect_unsupported) {
            test_unsupported_device(capabilities);
            std::cout << "Vulkan Float64 unsupported-device tests passed\n";
            return 0;
        }
        require(capabilities.strict_supported(), capabilities.requirement_error());

        std::vector<std::shared_ptr<const dsmvc::AxisPlan>> plans;
        for (const auto &request : {
                 make_request(41, 35, 35.0, 0.1875,
                              dsmvc::KernelKind::bilinear, 3,
                              dsmvc::F64Mode::float64_only),
                 make_request(47, 39, 39.0, -0.3125,
                              dsmvc::KernelKind::bicubic, 3,
                              dsmvc::F64Mode::float64_only),
                 make_request(59, 49, 49.0, 0.4375,
                              dsmvc::KernelKind::lanczos, 3,
                              dsmvc::F64Mode::float64_only),
                 make_request(67, 57, 57.0, -0.25,
                              dsmvc::KernelKind::spline64, 3,
                              dsmvc::F64Mode::float64_only),
             }) {
            plans.push_back(std::make_shared<const dsmvc::AxisPlan>(
                dsmvc::build_axis_plan(request)));
        }
        for (std::size_t index = 0U; index < plans.size(); ++index) {
            require(plans[index]->requires_float64(),
                    "forced Float64 plan did not retain Double data");
            require(plans[index]->half_bandwidth
                        == static_cast<std::int32_t>(index * 2U + 1U),
                    "bandwidth fixture drifted");
        }

        auto safe_horizontal = std::make_shared<const dsmvc::AxisPlan>(
            dsmvc::build_axis_plan(make_request(
                17, 13, 12.75, 0.125, dsmvc::KernelKind::bicubic, 3,
                dsmvc::F64Mode::float32_only)));
        auto safe_vertical = std::make_shared<const dsmvc::AxisPlan>(
            dsmvc::build_axis_plan(make_request(
                15, 11, 10.75, -0.25, dsmvc::KernelKind::bicubic, 3,
                dsmvc::F64Mode::float32_only)));
        require(!safe_horizontal->requires_float64()
                    && !safe_vertical->requires_float64(),
                "safe mixed-axis fixture retained Float64 data");

        dsmvc::vulkan_detail::VulkanExecutor vulkan;
        for (const auto &plan : plans) vulkan.prepare(plan);
        vulkan.prepare(safe_horizontal);
        vulkan.prepare(safe_vertical);
        vulkan.seal();
        const dsmvc::CpuExecutor cpu(dsmvc::CpuPath::scalar);

        test_rows_and_columns(vulkan, cpu, plans);
        test_2d_float(
            "2d-risky-risky", vulkan, cpu, *plans[1], *plans[1]);
        test_2d_float(
            "2d-risky-safe", vulkan, cpu, *plans[1], *safe_vertical);
        test_2d_float(
            "2d-safe-risky", vulkan, cpu, *safe_horizontal, *plans[1]);

        test_2d_integer<std::uint8_t>(
            "u8-risky-risky", vulkan, cpu, *plans[1], *plans[1],
            {0.0F, 1.0F / 255.0F, 255.0F, 0.0F, 255U}, 255U);
        test_2d_integer<std::uint16_t>(
            "u10-risky-risky", vulkan, cpu, *plans[1], *plans[1],
            {0.0F, 1.0F / 1023.0F, 1023.0F, 0.0F, 1023U}, 1023U);
        test_2d_integer<std::uint16_t>(
            "u16-risky-risky", vulkan, cpu, *plans[1], *plans[1],
            {0.0F, 1.0F / 65535.0F, 65535.0F, 0.0F, 65535U}, 65535U);
        test_nonfinite_failure(vulkan, *plans[1]);
        std::cout << "Vulkan Float64 runtime tests passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "Vulkan Float64 runtime tests failed: "
                  << error.what() << '\n';
        return 1;
    }
}
