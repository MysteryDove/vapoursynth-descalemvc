#include <dsmvc/engine.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
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

[[nodiscard]] dsmvc::AxisPlan make_plan(
    std::int32_t source, std::int32_t destination,
    dsmvc::F64Mode precision) {
    dsmvc::AxisRequest request;
    request.source_size = source;
    request.destination_size = destination;
    request.active_length = static_cast<double>(destination);
    request.shift = 0.0;
    request.kernel.kind = dsmvc::KernelKind::bicubic;
    request.kernel.b = 0.0;
    request.kernel.c = 0.5;
    request.border = dsmvc::BorderMode::symmetric;
    request.f64_mode = precision;
    return dsmvc::build_axis_plan(request);
}

[[nodiscard]] std::array<std::uint32_t, 19> edge_palette(
    std::uint32_t maximum) {
    return {
        0U, 1U, 2U, 3U,
        maximum / 16U, maximum / 16U + 1U,
        maximum / 4U - 1U, maximum / 4U, maximum / 4U + 1U,
        maximum / 2U - 1U, maximum / 2U, maximum / 2U + 1U,
        maximum - maximum / 16U - 1U,
        maximum - maximum / 16U,
        maximum - 3U, maximum - 2U, maximum - 1U, maximum,
        (maximum + 1U) / 2U,
    };
}

template <class Sample>
[[nodiscard]] std::vector<Sample> patterned_frame(
    std::int32_t width, std::int32_t height, std::int32_t stride,
    std::uint32_t maximum) {
    std::vector<Sample> result(
        static_cast<std::size_t>(height) * static_cast<std::size_t>(stride),
        std::numeric_limits<Sample>::max());
    for (std::int32_t row = 0; row < height; ++row) {
        for (std::int32_t column = 0; column < width; ++column) {
            const std::uint32_t code = (
                static_cast<std::uint32_t>(column) * 193U
                + static_cast<std::uint32_t>(row) * 389U
                + static_cast<std::uint32_t>(column * row) * 17U)
                % (maximum + 1U);
            result[static_cast<std::size_t>(row)
                       * static_cast<std::size_t>(stride)
                   + static_cast<std::size_t>(column)] =
                static_cast<Sample>(code);
        }
    }
    const auto palette = edge_palette(maximum);
    for (std::size_t index = 0; index < palette.size(); ++index) {
        result[index] = static_cast<Sample>(palette[index]);
    }
    return result;
}

template <class Sample>
void require_padding(
    const std::vector<Sample> &values, std::int32_t rows,
    std::int32_t logical_width, std::int32_t stride, Sample sentinel,
    std::string_view label) {
    for (std::int32_t row = 0; row < rows; ++row) {
        for (std::int32_t column = logical_width; column < stride; ++column) {
            const auto index = static_cast<std::size_t>(row)
                    * static_cast<std::size_t>(stride)
                + static_cast<std::size_t>(column);
            require(values[index] == sentinel,
                    std::string{label} + " modified output padding");
        }
    }
}

template <class Sample>
void require_within_one_code(
    const std::vector<Sample> &reference,
    const std::vector<Sample> &candidate,
    std::int32_t rows, std::int32_t columns, std::int32_t stride,
    std::string_view label) {
    std::uint32_t maximum_error = 0U;
    for (std::int32_t row = 0; row < rows; ++row) {
        for (std::int32_t column = 0; column < columns; ++column) {
            const auto index = static_cast<std::size_t>(row)
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
            std::string{label} + " differs from the same-precision CPU scalar "
                "reference by " + std::to_string(maximum_error) + " codes");
}

template <class ExecutorType, class Sample>
void execute_integer(
    const ExecutorType &executor, bool streamed,
    const dsmvc::AxisPlan &horizontal, const dsmvc::AxisPlan &vertical,
    const std::vector<Sample> &input, std::int32_t input_stride,
    std::vector<Sample> &output, std::int32_t output_stride,
    const dsmvc::IntegerConversion &conversion) {
    if constexpr (std::is_same_v<Sample, std::uint8_t>) {
        if (streamed) {
            executor.inverse_2d_u8_streamed(
                horizontal, vertical, input.data(), input_stride,
                output.data(), output_stride, conversion);
        } else {
            executor.inverse_2d_u8(
                horizontal, vertical, input.data(), input_stride,
                output.data(), output_stride, conversion);
        }
    } else {
        if (streamed) {
            executor.inverse_2d_u16_streamed(
                horizontal, vertical, input.data(), input_stride,
                output.data(), output_stride, conversion);
        } else {
            executor.inverse_2d_u16(
                horizontal, vertical, input.data(), input_stride,
                output.data(), output_stride, conversion);
        }
    }
}

template <class ExecutorType, class Sample>
void check_route(
    const ExecutorType &executor, std::string_view route_name,
    const dsmvc::AxisPlan &horizontal, const dsmvc::AxisPlan &vertical,
    const std::vector<Sample> &input, std::int32_t input_stride,
    const std::vector<Sample> &reference, std::int32_t output_stride,
    const dsmvc::IntegerConversion &conversion, Sample sentinel,
    std::string_view fixture_name) {
    std::vector<Sample> buffered(reference.size(), sentinel);
    std::vector<Sample> streamed(reference.size(), sentinel);
    std::vector<Sample> repeat(reference.size(), sentinel);
    execute_integer(
        executor, false, horizontal, vertical, input, input_stride,
        buffered, output_stride, conversion);
    execute_integer(
        executor, true, horizontal, vertical, input, input_stride,
        streamed, output_stride, conversion);
    execute_integer(
        executor, false, horizontal, vertical, input, input_stride,
        repeat, output_stride, conversion);

    const std::string label = std::string{fixture_name} + "/"
        + std::string{route_name};
    require(buffered == streamed,
            label + " buffered and streamed routes are not bit exact");
    require(buffered == repeat,
            label + " repeated execution is not bit exact");
    require_within_one_code(
        reference, buffered, vertical.destination_size,
        horizontal.destination_size, output_stride, label);
    require_padding(
        buffered, vertical.destination_size, horizontal.destination_size,
        output_stride, sentinel, label);
}

[[nodiscard]] std::vector<dsmvc::BackendKind> available_accelerators(
    const dsmvc::AxisPlan &probe_plan) {
    std::vector<dsmvc::BackendKind> result;
    for (const auto capability : dsmvc::backend_capabilities()) {
        if ((capability.kind != dsmvc::BackendKind::cuda
             && capability.kind != dsmvc::BackendKind::vulkan)
            || !capability.compiled || !capability.device_available) {
            continue;
        }
        try {
            dsmvc::Executor executor(capability.kind, dsmvc::CpuPath::scalar);
            std::vector<float> input(
                static_cast<std::size_t>(probe_plan.source_size), 0.25F);
            std::vector<float> output(
                static_cast<std::size_t>(probe_plan.destination_size));
            executor.inverse_rows(
                probe_plan, input.data(), probe_plan.source_size,
                output.data(), probe_plan.destination_size, 1);
            result.push_back(capability.kind);
        } catch (const std::runtime_error &error) {
            const std::string_view message{error.what()};
            if (capability.kind == dsmvc::BackendKind::vulkan
                && probe_plan.requires_float64()
                && message.find("Vulkan Float64 execution requires")
                    != std::string_view::npos) {
                std::cout << "skipping Vulkan F64 integer contract: "
                          << message << '\n';
                continue;
            }
            throw;
        }
    }
    return result;
}

template <class Sample>
void test_fixture(
    const dsmvc::AxisPlan &horizontal, const dsmvc::AxisPlan &vertical,
    const std::vector<dsmvc::BackendKind> &accelerators,
    const dsmvc::IntegerConversion &conversion, std::string_view label) {
    const std::int32_t input_stride = horizontal.source_size + 3;
    const std::int32_t output_stride = horizontal.destination_size + 5;
    const Sample sentinel = std::is_same_v<Sample, std::uint8_t>
        ? static_cast<Sample>(0xa5U) : static_cast<Sample>(0xa55aU);
    const auto input = patterned_frame<Sample>(
        horizontal.source_size, vertical.source_size, input_stride,
        conversion.output_maximum);
    std::vector<Sample> reference(
        static_cast<std::size_t>(vertical.destination_size)
            * static_cast<std::size_t>(output_stride),
        sentinel);
    const dsmvc::CpuExecutor scalar(dsmvc::CpuPath::scalar);
    execute_integer(
        scalar, false, horizontal, vertical, input, input_stride,
        reference, output_stride, conversion);
    std::vector<Sample> scalar_streamed(reference.size(), sentinel);
    execute_integer(
        scalar, true, horizontal, vertical, input, input_stride,
        scalar_streamed, output_stride, conversion);
    require(reference == scalar_streamed,
            std::string{label}
                + "/scalar buffered and streamed routes are not bit exact");

    check_route(
        scalar, "scalar", horizontal, vertical, input, input_stride,
        reference, output_stride, conversion, sentinel, label);
    if (dsmvc::cpu_avx2_available()) {
        const dsmvc::CpuExecutor avx2(dsmvc::CpuPath::avx2);
        check_route(
            avx2, "avx2", horizontal, vertical, input, input_stride,
            reference, output_stride, conversion, sentinel, label);
    }
    if (dsmvc::cpu_neon_available()) {
        const dsmvc::CpuExecutor neon(dsmvc::CpuPath::neon);
        check_route(
            neon, "neon", horizontal, vertical, input, input_stride,
            reference, output_stride, conversion, sentinel, label);
    }
    for (const auto backend : accelerators) {
        const dsmvc::Executor executor(backend, dsmvc::CpuPath::scalar);
        check_route(
            executor, dsmvc::backend_name(backend), horizontal, vertical,
            input, input_stride, reference, output_stride, conversion,
            sentinel, label);
    }
}

void test_precision(dsmvc::F64Mode precision, std::string_view precision_name) {
    constexpr std::int32_t source_width = 96;
    constexpr std::int32_t source_height = 64;
    constexpr std::int32_t destination_width = 80;
    constexpr std::int32_t destination_height = 48;
    const auto horizontal = make_plan(
        source_width, destination_width, precision);
    const auto vertical = make_plan(
        source_height, destination_height, precision);
    require(horizontal.valid() && vertical.valid(),
            std::string{precision_name} + " integer fixture plan is invalid");
    require((precision == dsmvc::F64Mode::float64_only)
                == (horizontal.requires_float64()
                    && vertical.requires_float64()),
            std::string{precision_name}
                + " integer fixture selected the wrong precision");
    const auto accelerators = available_accelerators(horizontal);

    test_fixture<std::uint8_t>(
        horizontal, vertical, accelerators,
        {0.0F, 1.0F / 255.0F, 255.0F, 0.0F, 255U},
        std::string{precision_name} + "/u8-patterned-96x64-to-80x48");
    test_fixture<std::uint16_t>(
        horizontal, vertical, accelerators,
        {0.0F, 1.0F / 1023.0F, 1023.0F, 0.0F, 1023U},
        std::string{precision_name} + "/u10-patterned-96x64-to-80x48");
    test_fixture<std::uint16_t>(
        horizontal, vertical, accelerators,
        {0.0F, 1.0F / 65535.0F, 65535.0F, 0.0F, 65535U},
        std::string{precision_name} + "/u16-patterned-96x64-to-80x48");
}

} // namespace

int main() {
    try {
        test_precision(dsmvc::F64Mode::float32_only, "f32");
        test_precision(dsmvc::F64Mode::float64_only, "f64");
        std::cout << "dsmvc integer contract tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "dsmvc integer contract test failure: "
                  << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
