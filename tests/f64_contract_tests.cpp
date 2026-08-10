#include "numerical_conformance.hpp"

#include <dsmvc/engine.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
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

template <class Function>
void expect_runtime_error(Function &&function, std::string_view label) {
    try {
        std::forward<Function>(function)();
    } catch (const std::runtime_error &) {
        return;
    }
    throw std::runtime_error(std::string{label}
                             + " did not throw std::runtime_error");
}

[[nodiscard]] dsmvc::AxisRequest make_request(
    std::int32_t source, std::int32_t destination, double active_length,
    double shift, dsmvc::BorderMode border, dsmvc::F64Mode mode) {
    dsmvc::AxisRequest request;
    request.source_size = source;
    request.destination_size = destination;
    request.active_length = active_length;
    request.shift = shift;
    request.kernel.kind = dsmvc::KernelKind::bilinear;
    request.border = border;
    request.f64_mode = mode;
    return request;
}

[[nodiscard]] std::shared_ptr<const dsmvc::AxisPlan> make_plan(
    const dsmvc::AxisRequest &request) {
    auto result = std::make_shared<const dsmvc::AxisPlan>(
        dsmvc::build_axis_plan(request));
    require(result->valid(), "F64 contract plan is invalid");
    return result;
}

[[nodiscard]] std::shared_ptr<const dsmvc::AxisPlan> identity_plan() {
    return make_plan(make_request(
        3, 3, 3.0, 0.0, dsmvc::BorderMode::symmetric,
        dsmvc::F64Mode::float32_only));
}

[[nodiscard]] std::vector<dsmvc::BackendKind> f64_backends(
    const dsmvc::AxisPlan &probe_plan) {
    std::vector<dsmvc::BackendKind> result{dsmvc::BackendKind::cpu};
    const auto capabilities = dsmvc::backend_capabilities();
    for (const auto requested : {
             dsmvc::BackendKind::cuda, dsmvc::BackendKind::vulkan}) {
        const auto found = std::find_if(
            capabilities.begin(), capabilities.end(),
            [requested](const auto &capability) {
                return capability.kind == requested;
            });
        if (found == capabilities.end()
            || !found->compiled || !found->device_available) {
            continue;
        }
        const auto capability = *found;
        try {
            dsmvc::Executor executor(capability.kind);
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
                && message.find("Vulkan Float64 execution requires")
                    != std::string_view::npos) {
                std::cout << "skipping Vulkan F64 contract: " << message << '\n';
                continue;
            }
            throw;
        }
    }
    return result;
}

[[nodiscard]] dsmvc::Executor make_executor(dsmvc::BackendKind backend) {
    return dsmvc::Executor(backend, dsmvc::CpuPath::scalar);
}

void require_float_matrix(
    const std::vector<float> &reference, const std::vector<float> &candidate,
    std::int32_t rows, std::int32_t columns, std::int32_t stride,
    std::string_view label) {
    std::uint32_t maximum_ulp = 0U;
    double maximum_absolute = 0.0;
    std::size_t maximum_index = 0U;
    for (std::int32_t row = 0; row < rows; ++row) {
        for (std::int32_t column = 0; column < columns; ++column) {
            const auto index = static_cast<std::size_t>(row)
                    * static_cast<std::size_t>(stride)
                + static_cast<std::size_t>(column);
            require(std::isfinite(candidate[index]),
                    std::string{label} + " produced a non-finite value");
            const auto ulp = dsmvc::numerical::float_ulp_distance(
                reference[index], candidate[index]);
            if (ulp > maximum_ulp) {
                maximum_ulp = ulp;
                maximum_index = index;
            }
            maximum_absolute = std::max(
                maximum_absolute,
                std::abs(static_cast<double>(reference[index])
                         - static_cast<double>(candidate[index])));
        }
    }
    require(maximum_ulp <= 1U,
            std::string{label} + " differs from CPU scalar by "
                + std::to_string(maximum_ulp) + " ULP (absolute "
                + std::to_string(maximum_absolute) + ", index "
                + std::to_string(maximum_index) + ", reference "
                + std::to_string(reference[maximum_index]) + ", candidate "
                + std::to_string(candidate[maximum_index]) + ")");
}

template <class Sample>
void require_integer_matrix(
    const std::vector<Sample> &reference,
    const std::vector<Sample> &candidate,
    std::int32_t rows, std::int32_t columns, std::int32_t stride,
    std::string_view label) {
    std::uint32_t maximum = 0U;
    for (std::int32_t row = 0; row < rows; ++row) {
        for (std::int32_t column = 0; column < columns; ++column) {
            const auto index = static_cast<std::size_t>(row)
                    * static_cast<std::size_t>(stride)
                + static_cast<std::size_t>(column);
            const auto left = static_cast<std::uint32_t>(reference[index]);
            const auto right = static_cast<std::uint32_t>(candidate[index]);
            maximum = std::max(maximum, left > right ? left - right : right - left);
        }
    }
    require(maximum <= 1U,
            std::string{label} + " differs from CPU scalar by "
                + std::to_string(maximum) + " codes");
}

void test_axis_routes(
    const dsmvc::AxisPlan &plan,
    const std::vector<dsmvc::BackendKind> &backends,
    std::string_view fixture_name) {
    constexpr std::int32_t vectors = 5;
    const std::int32_t row_input_stride = plan.source_size + 2;
    const std::int32_t row_output_stride = plan.destination_size + 3;
    std::vector<float> row_input(
        static_cast<std::size_t>(vectors) * row_input_stride, 7.0F);
    const auto packed = dsmvc::numerical::make_normal_input(
        static_cast<std::size_t>(vectors)
            * static_cast<std::size_t>(plan.source_size),
        0xf6407101U + static_cast<std::uint32_t>(plan.source_size));
    for (std::int32_t row = 0; row < vectors; ++row) {
        std::copy_n(
            packed.data() + static_cast<std::ptrdiff_t>(row) * plan.source_size,
            plan.source_size,
            row_input.data() + static_cast<std::ptrdiff_t>(row)
                * row_input_stride);
    }
    std::vector<float> row_reference(
        static_cast<std::size_t>(vectors) * row_output_stride, -19.0F);
    auto scalar = make_executor(dsmvc::BackendKind::cpu);
    scalar.inverse_rows(
        plan, row_input.data(), row_input_stride,
        row_reference.data(), row_output_stride, vectors);

    const std::int32_t column_stride = vectors + 3;
    std::vector<float> column_input(
        static_cast<std::size_t>(plan.source_size) * column_stride, 5.0F);
    for (std::int32_t row = 0; row < plan.source_size; ++row) {
        for (std::int32_t column = 0; column < vectors; ++column) {
            column_input[static_cast<std::size_t>(row) * column_stride + column] =
                row_input[static_cast<std::size_t>(column) * row_input_stride + row];
        }
    }
    std::vector<float> column_reference(
        static_cast<std::size_t>(plan.destination_size) * column_stride, -21.0F);
    scalar.inverse_columns(
        plan, column_input.data(), column_stride,
        column_reference.data(), column_stride, vectors);

    for (const auto backend : backends) {
        auto executor = make_executor(backend);
        std::vector<float> row_candidate(row_reference.size(), -19.0F);
        std::vector<float> row_repeat(row_reference.size(), -19.0F);
        executor.inverse_rows(
            plan, row_input.data(), row_input_stride,
            row_candidate.data(), row_output_stride, vectors);
        executor.inverse_rows(
            plan, row_input.data(), row_input_stride,
            row_repeat.data(), row_output_stride, vectors);
        require(row_candidate == row_repeat,
                std::string{fixture_name} + "/"
                    + dsmvc::backend_name(backend)
                    + " repeated rows were not bit exact");
        require_float_matrix(
            row_reference, row_candidate, vectors, plan.destination_size,
            row_output_stride,
            std::string{fixture_name} + "/" + dsmvc::backend_name(backend)
                + "/rows");

        std::vector<float> column_candidate(column_reference.size(), -21.0F);
        std::vector<float> column_repeat(column_reference.size(), -21.0F);
        executor.inverse_columns(
            plan, column_input.data(), column_stride,
            column_candidate.data(), column_stride, vectors);
        executor.inverse_columns(
            plan, column_input.data(), column_stride,
            column_repeat.data(), column_stride, vectors);
        require(column_candidate == column_repeat,
                std::string{fixture_name} + "/"
                    + dsmvc::backend_name(backend)
                    + " repeated columns were not bit exact");
        require_float_matrix(
            column_reference, column_candidate, plan.destination_size,
            vectors, column_stride,
            std::string{fixture_name} + "/" + dsmvc::backend_name(backend)
                + "/columns");
    }
}

void test_cpu_simd_tiny(
    const dsmvc::AxisPlan &plan, const dsmvc::AxisPlan &identity) {
    std::vector<dsmvc::CpuPath> paths;
    if (dsmvc::cpu_avx2_available()) paths.push_back(dsmvc::CpuPath::avx2);
    if (dsmvc::cpu_neon_available()) paths.push_back(dsmvc::CpuPath::neon);
    constexpr std::int32_t vectors = 5;
    const std::int32_t row_input_stride = plan.source_size + 2;
    const std::int32_t row_output_stride = plan.destination_size + 3;
    std::vector<float> row_input(
        static_cast<std::size_t>(vectors) * row_input_stride, 0.0F);
    for (std::int32_t row = 0; row < vectors; ++row) {
        row_input[static_cast<std::size_t>(row) * row_input_stride] =
            0.125F + static_cast<float>(row) * 0.1875F;
    }
    const dsmvc::CpuExecutor scalar(dsmvc::CpuPath::scalar);
    std::vector<float> row_reference(
        static_cast<std::size_t>(vectors) * row_output_stride, -7.0F);
    scalar.inverse_rows(
        plan, row_input.data(), row_input_stride,
        row_reference.data(), row_output_stride, vectors);

    const std::int32_t column_stride = vectors + 2;
    std::vector<float> column_input(
        static_cast<std::size_t>(plan.source_size) * column_stride, 0.0F);
    for (std::int32_t column = 0; column < vectors; ++column) {
        column_input[static_cast<std::size_t>(column)] =
            row_input[static_cast<std::size_t>(column) * row_input_stride];
    }
    std::vector<float> column_reference(
        static_cast<std::size_t>(plan.destination_size) * column_stride, -9.0F);
    scalar.inverse_columns(
        plan, column_input.data(), column_stride,
        column_reference.data(), column_stride, vectors);

    const std::int32_t input_2d_stride = plan.source_size + 2;
    const std::int32_t output_2d_stride = plan.destination_size + 3;
    std::vector<float> input_2d(
        static_cast<std::size_t>(identity.source_size) * input_2d_stride, 0.0F);
    for (std::int32_t row = 0; row < identity.source_size; ++row) {
        input_2d[static_cast<std::size_t>(row) * input_2d_stride] =
            0.25F + static_cast<float>(row) * 0.125F;
    }
    std::vector<float> reference_2d(
        static_cast<std::size_t>(identity.destination_size) * output_2d_stride,
        -11.0F);
    scalar.inverse_2d(
        plan, identity, input_2d.data(), input_2d_stride,
        reference_2d.data(), output_2d_stride);

    for (const auto path : paths) {
        const dsmvc::CpuExecutor executor(path);
        std::vector<float> rows(row_reference.size(), -7.0F);
        std::vector<float> columns(column_reference.size(), -9.0F);
        std::vector<float> output_2d(reference_2d.size(), -11.0F);
        executor.inverse_rows(
            plan, row_input.data(), row_input_stride,
            rows.data(), row_output_stride, vectors);
        executor.inverse_columns(
            plan, column_input.data(), column_stride,
            columns.data(), column_stride, vectors);
        executor.inverse_2d(
            plan, identity, input_2d.data(), input_2d_stride,
            output_2d.data(), output_2d_stride);
        const std::string name = path == dsmvc::CpuPath::avx2 ? "AVX2" : "NEON";
        require_float_matrix(
            row_reference, rows, vectors, plan.destination_size,
            row_output_stride, "tiny-pivot/" + name + "/rows");
        require_float_matrix(
            column_reference, columns, plan.destination_size,
            vectors, column_stride, "tiny-pivot/" + name + "/columns");
        require_float_matrix(
            reference_2d, output_2d, identity.destination_size,
            plan.destination_size, output_2d_stride,
            "tiny-pivot/" + name + "/2D");
    }
}

void test_float_2d(
    const dsmvc::AxisPlan &horizontal, const dsmvc::AxisPlan &vertical,
    const std::vector<dsmvc::BackendKind> &backends,
    std::string_view fixture_name, std::uint32_t seed) {
    const std::int32_t input_stride = horizontal.source_size + 2;
    const std::int32_t output_stride = horizontal.destination_size + 3;
    const auto packed = dsmvc::numerical::make_normal_input(
        static_cast<std::size_t>(vertical.source_size)
            * static_cast<std::size_t>(horizontal.source_size),
        seed);
    std::vector<float> input(
        static_cast<std::size_t>(vertical.source_size) * input_stride, 3.0F);
    for (std::int32_t row = 0; row < vertical.source_size; ++row) {
        std::copy_n(
            packed.data() + static_cast<std::ptrdiff_t>(row)
                * horizontal.source_size,
            horizontal.source_size,
            input.data() + static_cast<std::ptrdiff_t>(row) * input_stride);
    }
    std::vector<float> reference(
        static_cast<std::size_t>(vertical.destination_size) * output_stride,
        -31.0F);
    auto scalar = make_executor(dsmvc::BackendKind::cpu);
    scalar.inverse_2d(
        horizontal, vertical, input.data(), input_stride,
        reference.data(), output_stride);
    for (const auto backend : backends) {
        auto executor = make_executor(backend);
        std::vector<float> candidate(reference.size(), -31.0F);
        std::vector<float> repeat(reference.size(), -31.0F);
        executor.inverse_2d(
            horizontal, vertical, input.data(), input_stride,
            candidate.data(), output_stride);
        executor.inverse_2d(
            horizontal, vertical, input.data(), input_stride,
            repeat.data(), output_stride);
        require(candidate == repeat,
                std::string{fixture_name} + "/"
                    + dsmvc::backend_name(backend)
                    + " repeated 2D execution was not bit exact");
        require_float_matrix(
            reference, candidate, vertical.destination_size,
            horizontal.destination_size, output_stride,
            std::string{fixture_name} + "/" + dsmvc::backend_name(backend));
    }
}

template <class Sample>
void test_integer_2d(
    const dsmvc::AxisPlan &horizontal, const dsmvc::AxisPlan &vertical,
    const std::vector<dsmvc::BackendKind> &backends,
    const dsmvc::IntegerConversion &conversion, std::string_view label) {
    const std::int32_t input_stride = horizontal.source_size + 2;
    const std::int32_t output_stride = horizontal.destination_size + 3;
    std::vector<Sample> input(
        static_cast<std::size_t>(vertical.source_size) * input_stride);
    std::uint32_t state = 0x6400c0deU + conversion.output_maximum;
    for (std::int32_t row = 0; row < vertical.source_size; ++row) {
        for (std::int32_t column = 0; column < horizontal.source_size; ++column) {
            state = state * 1664525U + 1013904223U;
            input[static_cast<std::size_t>(row) * input_stride + column] =
                static_cast<Sample>(state % (conversion.output_maximum + 1U));
        }
    }
    std::vector<Sample> reference(
        static_cast<std::size_t>(vertical.destination_size) * output_stride,
        std::numeric_limits<Sample>::max());
    auto scalar = make_executor(dsmvc::BackendKind::cpu);
    if constexpr (std::is_same_v<Sample, std::uint8_t>) {
        scalar.inverse_2d_u8(
            horizontal, vertical, input.data(), input_stride,
            reference.data(), output_stride, conversion);
    } else {
        scalar.inverse_2d_u16(
            horizontal, vertical, input.data(), input_stride,
            reference.data(), output_stride, conversion);
    }
    for (const auto backend : backends) {
        auto executor = make_executor(backend);
        std::vector<Sample> buffered(reference.size(),
                                     std::numeric_limits<Sample>::max());
        std::vector<Sample> streamed(buffered);
        std::vector<Sample> repeat(buffered);
        if constexpr (std::is_same_v<Sample, std::uint8_t>) {
            executor.inverse_2d_u8(
                horizontal, vertical, input.data(), input_stride,
                buffered.data(), output_stride, conversion);
            executor.inverse_2d_u8_streamed(
                horizontal, vertical, input.data(), input_stride,
                streamed.data(), output_stride, conversion);
            executor.inverse_2d_u8(
                horizontal, vertical, input.data(), input_stride,
                repeat.data(), output_stride, conversion);
        } else {
            executor.inverse_2d_u16(
                horizontal, vertical, input.data(), input_stride,
                buffered.data(), output_stride, conversion);
            executor.inverse_2d_u16_streamed(
                horizontal, vertical, input.data(), input_stride,
                streamed.data(), output_stride, conversion);
            executor.inverse_2d_u16(
                horizontal, vertical, input.data(), input_stride,
                repeat.data(), output_stride, conversion);
        }
        require(buffered == streamed && buffered == repeat,
                std::string{label} + "/" + dsmvc::backend_name(backend)
                    + " route or repeat was not bit exact");
        require_integer_matrix(
            reference, buffered, vertical.destination_size,
            horizontal.destination_size, output_stride,
            std::string{label} + "/" + dsmvc::backend_name(backend));
    }
}

[[nodiscard]] dsmvc::AxisPlan explosive_plan(double transpose_weight) {
    dsmvc::AxisPlan plan;
    plan.source_size = 1;
    plan.destination_size = 1;
    plan.support = 1;
    plan.half_bandwidth = 0;
    plan.active_length = 1.0;
    plan.shift = 0.0;
    plan.transpose_offsets = {0U, 1U};
    plan.transpose_indices = {0};
    plan.transpose_weights = {1.0F};
    plan.inverse_diagonal = {1.0F};
    plan.normal_rcond = 1.0;
    plan.normal_inf_norm = 1.0;
    plan.transpose_weights_f64 = {transpose_weight};
    plan.normal_bands_f64 = {1.0};
    plan.ldlt_bands_f64 = {1.0};
    plan.inverse_diagonal_f64 = {1.0};
    require(plan.valid() && plan.requires_float64(),
            "explosive F64 plan is invalid");
    return plan;
}

void test_nonfinite_contract(
    const dsmvc::AxisPlan &retained,
    const std::vector<dsmvc::BackendKind> &backends) {
    for (const auto backend : backends) {
        auto executor = make_executor(backend);
        for (const float value : {
                 std::numeric_limits<float>::quiet_NaN(),
                 std::numeric_limits<float>::infinity(),
                 -std::numeric_limits<float>::infinity()}) {
            std::vector<float> row_input(
                static_cast<std::size_t>(retained.source_size), 0.25F);
            std::vector<float> row_output(
                static_cast<std::size_t>(retained.destination_size));
            row_input[row_input.size() / 2U] = value;
            expect_runtime_error([&] {
                executor.inverse_rows(
                    retained, row_input.data(), retained.source_size,
                    row_output.data(), retained.destination_size, 1);
            }, std::string{dsmvc::backend_name(backend)} + " nonfinite rows");

            constexpr std::int32_t columns = 3;
            std::vector<float> column_input(
                static_cast<std::size_t>(retained.source_size) * columns,
                0.25F);
            std::vector<float> column_output(
                static_cast<std::size_t>(retained.destination_size) * columns);
            column_input[column_input.size() / 2U] = value;
            expect_runtime_error([&] {
                executor.inverse_columns(
                    retained, column_input.data(), columns,
                    column_output.data(), columns, columns);
            }, std::string{dsmvc::backend_name(backend)} + " nonfinite columns");

            const auto identity = identity_plan();
            std::vector<float> input_2d(
                static_cast<std::size_t>(identity->source_size)
                    * retained.source_size,
                0.25F);
            std::vector<float> output_2d(
                static_cast<std::size_t>(identity->destination_size)
                    * retained.destination_size);
            input_2d[input_2d.size() / 2U] = value;
            expect_runtime_error([&] {
                executor.inverse_2d(
                    retained, *identity, input_2d.data(), retained.source_size,
                    output_2d.data(), retained.destination_size);
            }, std::string{dsmvc::backend_name(backend)} + " nonfinite 2D");
        }

        std::vector<float> finite_input(
            static_cast<std::size_t>(retained.source_size), 0.0F);
        finite_input[0] = -0.0F;
        if (finite_input.size() > 1U) {
            finite_input[1] = std::numeric_limits<float>::denorm_min();
        }
        std::vector<float> finite_output(
            static_cast<std::size_t>(retained.destination_size));
        executor.inverse_rows(
            retained, finite_input.data(), retained.source_size,
            finite_output.data(), retained.destination_size, 1);
        require(std::all_of(
                    finite_output.begin(), finite_output.end(),
                    [](float value) { return std::isfinite(value); }),
                std::string{dsmvc::backend_name(backend)}
                    + " rejected signed zero or finite subnormal input");

        const auto cast_overflow = explosive_plan(1.0e308);
        std::vector<float> one_input{1.0F};
        std::vector<float> one_output(1U);
        expect_runtime_error([&] {
            executor.inverse_rows(
                cast_overflow, one_input.data(), 1, one_output.data(), 1, 1);
        }, std::string{dsmvc::backend_name(backend)} + " F64-to-F32 overflow");

        const auto double_overflow = explosive_plan(
            std::numeric_limits<double>::max());
        one_input[0] = 2.0F;
        expect_runtime_error([&] {
            executor.inverse_rows(
                double_overflow, one_input.data(), 1, one_output.data(), 1, 1);
        }, std::string{dsmvc::backend_name(backend)} + " F64 result overflow");

        std::vector<std::uint8_t> integer_input{2U};
        std::vector<std::uint8_t> integer_output(1U);
        expect_runtime_error([&] {
            executor.inverse_2d_u8(
                double_overflow, double_overflow,
                integer_input.data(), 1, integer_output.data(), 1,
                {0.0F, 1.0F, 1.0F, 0.0F, 255U});
        }, std::string{dsmvc::backend_name(backend)}
               + " nonfinite integer conversion");
    }
}

void test_unreferenced_nonfinite_input(
    const std::vector<dsmvc::BackendKind> &backends) {
    auto retained = explosive_plan(1.0);
    retained.source_size = 2;
    retained.active_length = 2.0;
    require(retained.valid(), "unreferenced-input fixture is invalid");
    constexpr std::size_t unused_index = 1U;
    const float nonfinite = std::numeric_limits<float>::quiet_NaN();

    for (const auto backend : backends) {
        auto executor = make_executor(backend);
        const std::string label = dsmvc::backend_name(backend);

        std::vector<float> row_input(
            static_cast<std::size_t>(retained.source_size), 0.25F);
        std::vector<float> row_output(
            static_cast<std::size_t>(retained.destination_size));
        row_input[unused_index] = nonfinite;
        expect_runtime_error([&] {
            executor.inverse_rows(
                retained, row_input.data(), retained.source_size,
                row_output.data(), retained.destination_size, 1);
        }, label + " unreferenced nonfinite row input");

        constexpr std::int32_t columns = 3;
        std::vector<float> column_input(
            static_cast<std::size_t>(retained.source_size) * columns, 0.25F);
        std::vector<float> column_output(
            static_cast<std::size_t>(retained.destination_size) * columns);
        column_input[unused_index * columns + 1U] = nonfinite;
        expect_runtime_error([&] {
            executor.inverse_columns(
                retained, column_input.data(), columns,
                column_output.data(), columns, columns);
        }, label + " unreferenced nonfinite column input");

        const auto identity = identity_plan();
        std::vector<float> input_2d(
            static_cast<std::size_t>(identity->source_size)
                * retained.source_size,
            0.25F);
        std::vector<float> output_2d(
            static_cast<std::size_t>(identity->destination_size)
                * retained.destination_size);
        input_2d[unused_index] = nonfinite;
        expect_runtime_error([&] {
            executor.inverse_2d(
                retained, *identity, input_2d.data(), retained.source_size,
                output_2d.data(), retained.destination_size);
        }, label + " unreferenced nonfinite 2D input");
    }
}

} // namespace

int main() {
    try {
        const auto tiny_forced = make_plan(make_request(
            1, 2, 2.0, 0.0, dsmvc::BorderMode::zero,
            dsmvc::F64Mode::float64_only));
        const auto tiny_automatic = make_plan(make_request(
            1, 2, 2.0, 0.0, dsmvc::BorderMode::zero,
            dsmvc::F64Mode::automatic));
        const auto zero_forced = make_plan(make_request(
            1000, 500, 10.0, 0.0, dsmvc::BorderMode::zero,
            dsmvc::F64Mode::float64_only));
        const auto zero_automatic = make_plan(make_request(
            1000, 500, 10.0, 0.0, dsmvc::BorderMode::zero,
            dsmvc::F64Mode::automatic));
        require(tiny_forced->requires_float64()
                    && tiny_automatic->requires_float64()
                    && zero_forced->requires_float64()
                    && zero_automatic->requires_float64(),
                "tiny/zero pivot fixture did not retain Float64");

        const auto backends = f64_backends(*tiny_forced);
        test_axis_routes(*tiny_forced, backends, "tiny-pivot-forced");
        test_axis_routes(*tiny_automatic, backends, "tiny-pivot-automatic");
        test_axis_routes(*zero_forced, backends, "zero-pivot-forced");
        test_axis_routes(*zero_automatic, backends, "zero-pivot-automatic");

        const auto identity = identity_plan();
        test_cpu_simd_tiny(*tiny_forced, *identity);
        test_float_2d(
            *zero_forced, *identity, backends,
            "zero-pivot-horizontal", 0x64002d01U);
        test_float_2d(
            *zero_automatic, *identity, backends,
            "zero-pivot-horizontal-automatic", 0x64002d04U);
        test_float_2d(
            *identity, *zero_forced, backends,
            "zero-pivot-vertical", 0x64002d02U);
        test_float_2d(
            *tiny_forced, *tiny_forced, backends,
            "tiny-pivot-both-axes", 0x64002d03U);

        test_integer_2d<std::uint8_t>(
            *zero_forced, *identity, backends,
            {0.0F, 1.0F / 255.0F, 255.0F, 0.0F, 255U},
            "zero-pivot-u8");
        test_integer_2d<std::uint8_t>(
            *zero_automatic, *identity, backends,
            {0.0F, 1.0F / 255.0F, 255.0F, 0.0F, 255U},
            "zero-pivot-u8-automatic");
        test_integer_2d<std::uint16_t>(
            *zero_forced, *identity, backends,
            {0.0F, 1.0F / 1023.0F, 1023.0F, 0.0F, 1023U},
            "zero-pivot-u10");
        test_integer_2d<std::uint16_t>(
            *zero_forced, *identity, backends,
            {0.0F, 1.0F / 65535.0F, 65535.0F, 0.0F, 65535U},
            "zero-pivot-u16");
        test_nonfinite_contract(*tiny_forced, backends);
        test_unreferenced_nonfinite_input(backends);
        std::cout << "dsmvc F64 contract tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "dsmvc F64 contract test failure: "
                  << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
