#include "metal_float_executor_apple.hpp"
#include "metal_yuv_executor_apple.hpp"
#include "numerical_conformance.hpp"

#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, std::string message) {
    if (!condition) throw std::runtime_error(std::move(message));
}

[[nodiscard]] std::shared_ptr<const dsmvc::AxisPlan> make_identity_plan() {
    auto request = dsmvc::numerical::make_axis_request(
        5, 5, 5.0, 0.0, dsmvc::KernelKind::bilinear, 0,
        dsmvc::BorderMode::symmetric, dsmvc::F64Mode::float32_only);
    auto plan = std::make_shared<const dsmvc::AxisPlan>(
        dsmvc::build_axis_plan(request));
    require(plan->valid() && !plan->requires_float64(),
            "Metal identity plan is invalid");
    return plan;
}

[[nodiscard]] std::shared_ptr<const dsmvc::AxisPlan> make_generic_plan() {
    auto request = dsmvc::numerical::make_axis_request(
        71, 59, 58.75, 0.1875, dsmvc::KernelKind::custom, 5,
        dsmvc::BorderMode::symmetric, dsmvc::F64Mode::float32_only);
    auto plan = std::make_shared<const dsmvc::AxisPlan>(
        dsmvc::build_axis_plan(request, [](double distance) {
            return std::max(1.0 - distance / 5.0, 0.0);
        }));
    require(plan->valid() && !plan->requires_float64()
                && plan->half_bandwidth == 9,
            "Metal generic plan is invalid");
    return plan;
}

[[nodiscard]] std::shared_ptr<const dsmvc::AxisPlan>
make_single_destination_plan() {
    auto request = dsmvc::numerical::make_axis_request(
        5, 1, 1.0, 0.0, dsmvc::KernelKind::bilinear, 0,
        dsmvc::BorderMode::symmetric, dsmvc::F64Mode::float32_only);
    auto plan = std::make_shared<const dsmvc::AxisPlan>(
        dsmvc::build_axis_plan(request));
    require(plan->valid() && !plan->requires_float64()
                && plan->half_bandwidth == 0
                && plan->lower_ld.empty() && plan->upper_l.empty(),
            "Metal destination-size-one plan is invalid");
    return plan;
}

[[nodiscard]] std::vector<float> ordered_2d(
    const dsmvc::AxisPlan &horizontal, const dsmvc::AxisPlan &vertical,
    const std::vector<float> &input) {
    const auto intermediate_stride = horizontal.destination_size;
    std::vector<float> intermediate(
        static_cast<std::size_t>(vertical.source_size)
        * static_cast<std::size_t>(intermediate_stride));
    for (std::int32_t row = 0; row < vertical.source_size; ++row) {
        dsmvc::detail::inverse_axis_f32_ordered(
            horizontal,
            input.data() + static_cast<std::ptrdiff_t>(row)
                * horizontal.source_size,
            1,
            intermediate.data() + static_cast<std::ptrdiff_t>(row)
                * intermediate_stride,
            1);
    }

    std::vector<float> output(
        static_cast<std::size_t>(vertical.destination_size)
        * static_cast<std::size_t>(horizontal.destination_size));
    for (std::int32_t column = 0;
         column < horizontal.destination_size; ++column) {
        dsmvc::detail::inverse_axis_f32_ordered(
            vertical, intermediate.data() + column, intermediate_stride,
            output.data() + column, horizontal.destination_size);
    }
    return output;
}

void check_metal_route(
    std::string label,
    const std::shared_ptr<const dsmvc::AxisPlan> &horizontal,
    const std::shared_ptr<const dsmvc::AxisPlan> &vertical,
    std::uint32_t seed) {
    const auto input = dsmvc::numerical::make_normal_input(
        static_cast<std::size_t>(horizontal->source_size)
            * static_cast<std::size_t>(vertical->source_size),
        seed);
    const auto expected = ordered_2d(*horizontal, *vertical, input);
    std::vector<float> output(expected.size());

    dsmvc::experimental::MetalFloatExecutor executor(
        horizontal, vertical, 1U);
    std::array frames{
        dsmvc::experimental::MetalFloatFrame{
            input.data(),
            static_cast<std::ptrdiff_t>(horizontal->source_size * sizeof(float)),
            output.data(),
            static_cast<std::ptrdiff_t>(
                horizontal->destination_size * sizeof(float)),
        },
    };
    executor.execute(frames);

    double maximum = 0.0;
    std::uint32_t maximum_ulp = 0U;
    for (std::size_t index = 0; index < output.size(); ++index) {
        require(std::isfinite(output[index]),
                label + " produced a non-finite Metal value");
        maximum = std::max(
            maximum,
            std::abs(static_cast<double>(output[index])
                     - static_cast<double>(expected[index])));
        maximum_ulp = std::max(
            maximum_ulp,
            dsmvc::numerical::float_ulp_distance(
                output[index], expected[index]));
    }
    std::cout << label << " max_abs_vs_ordered=" << maximum
              << " max_ulp_vs_ordered=" << maximum_ulp << '\n';
    require(maximum_ulp == 0U,
            label + " differs from the ordered F32 contract");
}

void check_float64_rejection(
    const std::shared_ptr<const dsmvc::AxisPlan> &identity) {
    const auto retained = std::make_shared<const dsmvc::AxisPlan>(
        dsmvc::build_axis_plan(
            dsmvc::numerical::conditioned_lanczos2_request(
                dsmvc::F64Mode::automatic)));
    require(retained->requires_float64(),
            "Metal rejection fixture did not retain F64 data");
    bool rejected = false;
    try {
        dsmvc::experimental::MetalFloatExecutor executor(
            retained, identity, 1U);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    require(rejected, "direct Metal executor accepted an F64 plan");

    rejected = false;
    try {
        dsmvc::experimental::MetalYuvExecutor executor(
            {retained, identity}, {identity, identity}, 1U, 1U);
    } catch (const std::invalid_argument &) {
        rejected = true;
    }
    require(rejected, "experimental Metal YUV executor accepted an F64 plan");
}

} // namespace

int main() {
    try {
        // Gate on what the executor itself requires (any unified-memory
        // Metal device), not on the "Apple M" device-name prefix: CI
        // runners expose a virtualized unified-memory device whose name
        // does not carry that prefix.
        @autoreleasepool {
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            require(device != nil && device.hasUnifiedMemory,
                    "Metal numerical contract test requires Apple unified memory");
        }
        const auto identity = make_identity_plan();
        const auto fixtures = dsmvc::numerical::axis_fixtures();
        for (std::size_t index = 0; index < 4U; ++index) {
            const auto plan = std::make_shared<const dsmvc::AxisPlan>(
                dsmvc::build_axis_plan(fixtures[index].request));
            const std::string name = fixtures[index].name;
            check_metal_route(
                name + "/horizontal", plan, identity,
                fixtures[index].input_seed);
            check_metal_route(
                name + "/vertical", identity, plan,
                fixtures[index].input_seed ^ 0xa511e9b3U);
        }

        const auto generic = make_generic_plan();
        check_metal_route(
            "generic-b9/horizontal", generic, identity, 0x9e110001U);
        check_metal_route(
            "generic-b9/vertical", identity, generic, 0x9e110002U);
        const auto single_destination = make_single_destination_plan();
        check_metal_route(
            "destination-size-one", single_destination, single_destination,
            0xd3510001U);
        check_float64_rejection(identity);
        std::cout << "dsmvc Metal numerical contract tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "dsmvc Metal numerical contract test failure: "
                  << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
