#include "numerical_conformance.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using dsmvc::numerical::AxisFixture;

void require(bool condition, std::string message) {
    if (!condition) throw std::runtime_error(std::move(message));
}

struct FixtureResult {
    dsmvc::AxisPlan plan;
    std::string f32_plan_hash;
    std::string complete_plan_hash;
    std::string ordered_hash;
    std::string ordered_internal_hash;
    std::string production_hash;
    double maximum_f64_absolute = 0.0;
    std::uint32_t maximum_f64_ulp = 0U;
};

[[nodiscard]] FixtureResult evaluate_fixture(const AxisFixture &fixture) {
    FixtureResult result;
    result.plan = dsmvc::build_axis_plan(fixture.request);
    require(result.plan.valid(), std::string(fixture.name) + " plan is invalid");
    require(result.plan.half_bandwidth == fixture.expected_half_bandwidth,
            std::string(fixture.name) + " half-bandwidth drifted");
    result.f32_plan_hash = dsmvc::numerical::f32_plan_hash(
        fixture.request, result.plan);
    result.complete_plan_hash = dsmvc::numerical::complete_plan_hash(
        fixture.request, result.plan);

    const auto input = dsmvc::numerical::make_normal_input(
        static_cast<std::size_t>(result.plan.source_size), fixture.input_seed);
    std::vector<float> production(
        static_cast<std::size_t>(result.plan.destination_size));
    dsmvc::inverse_axis_f32(result.plan, input, production);
    result.production_hash = dsmvc::numerical::output_hash<float>(production);

    if (result.plan.requires_float64()) {
        std::vector<double> source(input.begin(), input.end());
        std::vector<double> ordered(
            static_cast<std::size_t>(result.plan.destination_size));
        dsmvc::detail::inverse_axis_f64_ordered(
            result.plan, source.data(), 1, ordered.data(), 1);
        result.ordered_internal_hash =
            dsmvc::numerical::output_hash<double>(ordered);
        std::vector<float> rounded(ordered.begin(), ordered.end());
        result.ordered_hash = dsmvc::numerical::output_hash<float>(rounded);
        const auto difference = dsmvc::numerical::compare_outputs<float, float>(
            rounded, production);
        require(difference.nonfinite == 0U,
                std::string(fixture.name)
                    + " F64 comparison produced a nonfinite value");
        result.maximum_f64_absolute = difference.maximum_absolute;
        for (std::size_t index = 0; index < rounded.size(); ++index) {
            result.maximum_f64_ulp = std::max(
                result.maximum_f64_ulp,
                dsmvc::numerical::float_ulp_distance(
                    rounded[index], production[index]));
        }
    } else {
        std::vector<float> ordered(
            static_cast<std::size_t>(result.plan.destination_size));
        dsmvc::detail::inverse_axis_f32_ordered(
            result.plan, input.data(), 1, ordered.data(), 1);
        result.ordered_hash = dsmvc::numerical::output_hash<float>(ordered);
    }
    return result;
}

[[nodiscard]] std::size_t expected_storage_bytes(const dsmvc::AxisPlan &plan) {
    return sizeof(plan)
        + plan.transpose_offsets.capacity() * sizeof(std::uint32_t)
        + plan.transpose_indices.capacity() * sizeof(std::int32_t)
        + plan.transpose_weights.capacity() * sizeof(float)
        + plan.lower_ld.capacity() * sizeof(float)
        + plan.upper_l.capacity() * sizeof(float)
        + plan.inverse_diagonal.capacity() * sizeof(float)
        + plan.transpose_weights_f64.capacity() * sizeof(double)
        + plan.normal_bands_f64.capacity() * sizeof(double)
        + plan.ldlt_bands_f64.capacity() * sizeof(double)
        + plan.inverse_diagonal_f64.capacity() * sizeof(double);
}

void check_retained_normal_matrix(
    const AxisFixture &fixture, const dsmvc::AxisPlan &plan) {
    const auto expected_bands =
        (static_cast<std::size_t>(plan.half_bandwidth) + 1U)
        * static_cast<std::size_t>(plan.destination_size);
    require(plan.normal_bands_f64.size() == expected_bands,
            std::string(fixture.name) + " raw normal bands are incomplete");
    require(std::isfinite(plan.normal_inf_norm) && plan.normal_inf_norm > 0.0,
            std::string(fixture.name) + " normal infinity norm is invalid");

    const auto audit = dsmvc::numerical::audit_normal_matrix(plan);
    const long double scale = std::max(
        1.0L, audit.maximum_band_magnitude);
    const long double tolerance = 128.0L
        * static_cast<long double>(std::numeric_limits<double>::epsilon())
        * scale;
    require(audit.maximum_band_error <= tolerance,
            std::string(fixture.name)
                + " retained normal bands differ from long-double reconstruction");
    require(audit.inf_norm_error <= tolerance * 4.0L,
            std::string(fixture.name)
                + " normal infinity norm differs from reconstruction");
}

void test_axis_fixture_goldens(bool emit_goldens) {
    std::size_t fixture_index = 0U;
    std::string complete_plan_drifts;
    for (const AxisFixture &fixture : dsmvc::numerical::axis_fixtures()) {
        const auto result = evaluate_fixture(fixture);
        if (fixture_index < 4U) {
            require(result.plan.normal_rcond >= 1.0e-4,
                    std::string(fixture.name)
                        + " ordinary F32 fixture became ill-conditioned");
        } else if (fixture_index == 4U || fixture_index == 6U) {
            require(result.plan.normal_rcond < 1.0e-4,
                    std::string(fixture.name)
                        + " conditioned fixture no longer crosses the F64 threshold");
        } else {
            require(result.plan.normal_rcond >= 1.0e-4,
                    std::string(fixture.name)
                        + " forced-F64 control is not naturally well-conditioned");
        }
        if (emit_goldens) {
            std::cout << fixture.name
                      << " f32_plan=" << result.f32_plan_hash
                      << " complete_plan=" << result.complete_plan_hash
                      << " rcond=" << result.plan.normal_rcond
                      << " ordered=" << result.ordered_hash
                      << " production=" << result.production_hash;
            if (result.plan.requires_float64()) {
                std::cout << " ordered_internal=" << result.ordered_internal_hash
                          << " max_abs=" << result.maximum_f64_absolute
                          << " max_ulp=" << result.maximum_f64_ulp;
            }
            std::cout << '\n';
        } else {
            require(result.f32_plan_hash == fixture.f32_plan_hash,
                    std::string(fixture.name)
                        + " immutable F32 plan hash drifted: "
                        + result.f32_plan_hash);
            if (result.complete_plan_hash != fixture.complete_plan_hash) {
                if (complete_plan_drifts.empty()) {
                    complete_plan_drifts = "complete plan hashes drifted:";
                }
                complete_plan_drifts += "\n  ";
                complete_plan_drifts += fixture.name;
                complete_plan_drifts += " expected=";
                complete_plan_drifts += fixture.complete_plan_hash;
                complete_plan_drifts += " actual=";
                complete_plan_drifts += result.complete_plan_hash;
            }
            require(result.ordered_hash == fixture.ordered_output_hash,
                    std::string(fixture.name) + " ordered oracle hash drifted: "
                        + result.ordered_hash);
            require(result.production_hash == fixture.production_output_hash,
                    std::string(fixture.name) + " production scalar hash drifted: "
                        + result.production_hash);
        }

        require(result.plan.storage_bytes() == expected_storage_bytes(result.plan),
                std::string(fixture.name)
                    + " plan storage accounting omitted retained data");
        if (result.plan.requires_float64()) {
            check_retained_normal_matrix(fixture, result.plan);
        } else {
            require(result.plan.normal_bands_f64.empty()
                        && result.plan.normal_inf_norm == 0.0,
                    std::string(fixture.name)
                        + " safe F32 plan retained high-precision residual data");
        }
        ++fixture_index;
    }
    require(complete_plan_drifts.empty(), std::move(complete_plan_drifts));
}

void test_cache_accounting() {
    dsmvc::clear_planner_caches();
    const auto request = dsmvc::numerical::conditioned_lanczos2_request(
        dsmvc::F64Mode::automatic);
    const auto plan = dsmvc::get_or_build_axis_plan(request);
    const auto first = dsmvc::planner_cache_stats();
    require(first.plan_entries == 1U
                && first.plan_resident_bytes == plan->storage_bytes(),
            "planner cache did not account for retained normal bands");
    const auto cached = dsmvc::get_or_build_axis_plan(request);
    const auto second = dsmvc::planner_cache_stats();
    require(cached == plan && second.plan_hits == first.plan_hits + 1U,
            "retained normal metadata changed cache identity");
    dsmvc::clear_planner_caches();
}

void test_malformed_retained_metadata() {
    const auto retained = dsmvc::build_axis_plan(
        dsmvc::numerical::conditioned_lanczos2_request(
            dsmvc::F64Mode::automatic));
    const auto safe = dsmvc::build_axis_plan(
        dsmvc::numerical::conditioned_lanczos2_request(
            dsmvc::F64Mode::float32_only));

    const auto rejects = [](dsmvc::AxisPlan plan, std::string_view label) {
        require(!plan.valid(), "malformed numerical metadata accepted: "
                + std::string(label));
    };

    auto malformed = retained;
    malformed.normal_bands_f64.pop_back();
    rejects(std::move(malformed), "truncated raw normal bands");

    malformed = retained;
    malformed.normal_bands_f64.front() =
        std::numeric_limits<double>::quiet_NaN();
    rejects(std::move(malformed), "non-finite raw normal band");

    malformed = retained;
    malformed.normal_inf_norm = std::numeric_limits<double>::infinity();
    rejects(std::move(malformed), "non-finite normal infinity norm");

    malformed = retained;
    malformed.normal_inf_norm = -1.0;
    rejects(std::move(malformed), "negative normal infinity norm");

    malformed = retained;
    malformed.normal_inf_norm = 0.0;
    rejects(std::move(malformed), "zero normal infinity norm");

    malformed = safe;
    malformed.normal_bands_f64.push_back(0.0);
    rejects(std::move(malformed), "raw normal bands on an F32 plan");

    malformed = safe;
    malformed.normal_inf_norm = 1.0;
    rejects(std::move(malformed), "normal norm on an F32 plan");
}

void check_native_tail_control(
    const AxisFixture &fixture, const dsmvc::CpuExecutor &native,
    std::int32_t rows) {
    const auto plan = dsmvc::build_axis_plan(fixture.request);
    const auto padded_source = (plan.source_size + 7) & ~7;
    const auto padded_destination = (plan.destination_size + 7) & ~7;
    const auto input_stride = padded_source + 3;
    const auto output_stride = padded_destination + 3;
    std::vector<float> input(
        static_cast<std::size_t>(rows)
            * static_cast<std::size_t>(input_stride),
        0.0F);
    std::vector<float> expected(
        static_cast<std::size_t>(rows)
            * static_cast<std::size_t>(output_stride),
        -23.0F);
    std::vector<float> output(expected.size(), -23.0F);
    for (std::int32_t row = 0; row < rows; ++row) {
        const auto row_input = dsmvc::numerical::make_normal_input(
            static_cast<std::size_t>(plan.source_size),
            fixture.input_seed + static_cast<std::uint32_t>(row));
        std::copy(row_input.begin(), row_input.end(),
                  input.begin() + static_cast<std::ptrdiff_t>(row)
                      * input_stride);
        dsmvc::detail::inverse_axis_f32_ordered(
            plan,
            input.data() + static_cast<std::ptrdiff_t>(row) * input_stride,
            1,
            expected.data() + static_cast<std::ptrdiff_t>(row) * output_stride,
            1);
    }
    native.inverse_rows(
        plan, input.data(), input_stride, output.data(), output_stride, rows);

    double maximum = 0.0;
    std::uint32_t maximum_ulp = 0U;
    for (std::int32_t row = 0; row < rows; ++row) {
        for (std::int32_t column = 0; column < plan.destination_size;
             ++column) {
            const auto index = static_cast<std::size_t>(row)
                    * static_cast<std::size_t>(output_stride)
                + static_cast<std::size_t>(column);
            require(std::isfinite(output[index]),
                    std::string(fixture.name)
                        + " native tail produced a nonfinite value");
            maximum = std::max(
                maximum,
                std::abs(static_cast<double>(output[index])
                         - static_cast<double>(expected[index])));
            maximum_ulp = std::max(
                maximum_ulp,
                dsmvc::numerical::float_ulp_distance(
                    output[index], expected[index]));
        }
        for (std::int32_t column = plan.destination_size;
             column < padded_destination; ++column) {
            const float tail = output[static_cast<std::size_t>(row)
                                      * static_cast<std::size_t>(output_stride)
                                  + static_cast<std::size_t>(column)];
            require(tail == -23.0F,
                    std::string(fixture.name)
                        + " native path overwrote SIMD tail storage");
        }
        for (std::int32_t column = padded_destination;
             column < output_stride; ++column) {
            require(output[static_cast<std::size_t>(row)
                               * static_cast<std::size_t>(output_stride)
                           + static_cast<std::size_t>(column)] == -23.0F,
                    std::string(fixture.name)
                        + " native path overwrote the output guard");
        }
    }
    require(maximum <= 3.0e-5,
            std::string(fixture.name)
                + " native compatibility path escaped the prework bound");
    if (native.path() == dsmvc::CpuPath::neon) {
        require(maximum_ulp == 0U,
                std::string(fixture.name)
                    + " NEON row path differs from ordered F32");
    }
    std::cout << fixture.name << " native=" << native.name()
              << " rows=" << rows
              << " max_abs_vs_ordered=" << maximum
              << " max_ulp_vs_ordered=" << maximum_ulp << '\n';
}

void test_native_tail_controls() {
    const auto fixtures = dsmvc::numerical::axis_fixtures();
    const dsmvc::CpuExecutor native(dsmvc::CpuPath::automatic);
    // Probe the hardware capability rather than inferring the executed path
    // from CpuExecutor::path(): below the SIMD width every call falls back to
    // scalar, and only rows beyond a full group prove SIMD tail storage.
    std::int32_t lanes = 0;
    if (dsmvc::cpu_avx2_available()) {
        lanes = 8;
    } else if (dsmvc::cpu_neon_available()) {
        lanes = 4;
    }
    for (std::size_t case_index = 0; case_index < 4U; ++case_index) {
        const auto &fixture = fixtures[case_index];
        if (lanes == 0) {
            check_native_tail_control(fixture, native, 5);
            continue;
        }
        check_native_tail_control(fixture, native, lanes - 1);
        check_native_tail_control(fixture, native, lanes);
        check_native_tail_control(fixture, native, lanes + 1);
        check_native_tail_control(fixture, native, lanes * 2);
    }
}

void check_native_f32_columns(
    const AxisFixture &fixture, const dsmvc::CpuExecutor &native,
    std::int32_t columns, std::int32_t stride) {
    const auto plan = dsmvc::build_axis_plan(fixture.request);
    std::vector<float> input(
        static_cast<std::size_t>(plan.source_size)
            * static_cast<std::size_t>(stride),
        0.0F);
    std::vector<float> expected(
        static_cast<std::size_t>(plan.destination_size)
            * static_cast<std::size_t>(stride),
        -23.0F);
    std::vector<float> output(expected.size(), -23.0F);
    for (std::int32_t column = 0; column < columns; ++column) {
        const auto column_input = dsmvc::numerical::make_normal_input(
            static_cast<std::size_t>(plan.source_size),
            fixture.input_seed + static_cast<std::uint32_t>(column));
        for (std::int32_t row = 0; row < plan.source_size; ++row) {
            input[static_cast<std::size_t>(row)
                      * static_cast<std::size_t>(stride)
                  + static_cast<std::size_t>(column)] =
                column_input[static_cast<std::size_t>(row)];
        }
        dsmvc::detail::inverse_axis_f32_ordered(
            plan, input.data() + column, stride,
            expected.data() + column, stride);
    }
    native.inverse_columns(
        plan, input.data(), stride, output.data(), stride, columns);

    double maximum = 0.0;
    std::uint32_t maximum_ulp = 0U;
    for (std::int32_t row = 0; row < plan.destination_size; ++row) {
        for (std::int32_t column = 0; column < columns; ++column) {
            const auto index = static_cast<std::size_t>(row)
                    * static_cast<std::size_t>(stride)
                + static_cast<std::size_t>(column);
            maximum = std::max(
                maximum,
                std::abs(static_cast<double>(output[index])
                         - static_cast<double>(expected[index])));
            maximum_ulp = std::max(
                maximum_ulp,
                dsmvc::numerical::float_ulp_distance(
                    output[index], expected[index]));
        }
    }
    if (native.path() == dsmvc::CpuPath::neon) {
        require(maximum_ulp == 0U,
                std::string(fixture.name)
                    + " NEON column path differs from ordered F32");
    }
    require(maximum <= 3.0e-5,
            std::string(fixture.name)
                + " native column path escaped the F32 absolute-error bound");
    std::cout << fixture.name << " native=" << native.name()
              << " columns=" << columns
              << " max_abs_vs_ordered=" << maximum
              << " max_ulp_vs_ordered=" << maximum_ulp << '\n';
}

void test_native_column_controls() {
    const auto fixtures = dsmvc::numerical::axis_fixtures();
    const dsmvc::CpuExecutor native(dsmvc::CpuPath::automatic);
    for (std::size_t case_index = 0; case_index < 4U; ++case_index) {
        check_native_f32_columns(fixtures[case_index], native, 3, 3);
        check_native_f32_columns(fixtures[case_index], native, 5, 8);
    }
}

void check_native_f64_axes(
    const AxisFixture &fixture, const dsmvc::CpuExecutor &native) {
    constexpr std::int32_t vectors = 5;
    const auto plan = dsmvc::build_axis_plan(fixture.request);
    require(plan.requires_float64(),
            std::string(fixture.name) + " F64 axis fixture lost precision mode");

    const auto row_input_stride = plan.source_size + 3;
    const auto row_output_stride = plan.destination_size + 3;
    std::vector<float> row_input(
        static_cast<std::size_t>(vectors)
            * static_cast<std::size_t>(row_input_stride),
        0.0F);
    std::vector<float> row_expected(
        static_cast<std::size_t>(vectors)
            * static_cast<std::size_t>(row_output_stride),
        -23.0F);
    std::vector<float> row_output(row_expected.size(), -23.0F);

    const auto column_stride = vectors + 3;
    std::vector<float> column_input(
        static_cast<std::size_t>(plan.source_size)
            * static_cast<std::size_t>(column_stride),
        0.0F);
    std::vector<float> column_expected(
        static_cast<std::size_t>(plan.destination_size)
            * static_cast<std::size_t>(column_stride),
        -23.0F);
    std::vector<float> column_output(column_expected.size(), -23.0F);

    std::vector<double> source(static_cast<std::size_t>(plan.source_size));
    std::vector<double> destination(
        static_cast<std::size_t>(plan.destination_size));
    for (std::int32_t vector = 0; vector < vectors; ++vector) {
        const auto vector_input = dsmvc::numerical::make_normal_input(
            static_cast<std::size_t>(plan.source_size),
            fixture.input_seed + static_cast<std::uint32_t>(vector));
        for (std::int32_t index = 0; index < plan.source_size; ++index) {
            const float value = vector_input[static_cast<std::size_t>(index)];
            row_input[static_cast<std::size_t>(vector)
                          * static_cast<std::size_t>(row_input_stride)
                      + static_cast<std::size_t>(index)] = value;
            column_input[static_cast<std::size_t>(index)
                             * static_cast<std::size_t>(column_stride)
                         + static_cast<std::size_t>(vector)] = value;
            source[static_cast<std::size_t>(index)] = static_cast<double>(value);
        }
        dsmvc::detail::inverse_axis_f64_ordered(
            plan, source.data(), 1, destination.data(), 1);
        for (std::int32_t index = 0; index < plan.destination_size; ++index) {
            const float value = static_cast<float>(
                destination[static_cast<std::size_t>(index)]);
            row_expected[static_cast<std::size_t>(vector)
                             * static_cast<std::size_t>(row_output_stride)
                         + static_cast<std::size_t>(index)] = value;
            column_expected[static_cast<std::size_t>(index)
                                * static_cast<std::size_t>(column_stride)
                            + static_cast<std::size_t>(vector)] = value;
        }
    }

    native.inverse_rows(
        plan, row_input.data(), row_input_stride,
        row_output.data(), row_output_stride, vectors);
    native.inverse_columns(
        plan, column_input.data(), column_stride,
        column_output.data(), column_stride, vectors);

    std::uint32_t row_ulp = 0U;
    std::uint32_t column_ulp = 0U;
    for (std::int32_t vector = 0; vector < vectors; ++vector) {
        for (std::int32_t index = 0; index < plan.destination_size; ++index) {
            const auto row_index = static_cast<std::size_t>(vector)
                    * static_cast<std::size_t>(row_output_stride)
                + static_cast<std::size_t>(index);
            const auto column_index = static_cast<std::size_t>(index)
                    * static_cast<std::size_t>(column_stride)
                + static_cast<std::size_t>(vector);
            row_ulp = std::max(
                row_ulp, dsmvc::numerical::float_ulp_distance(
                             row_output[row_index], row_expected[row_index]));
            column_ulp = std::max(
                column_ulp, dsmvc::numerical::float_ulp_distance(
                                column_output[column_index],
                                column_expected[column_index]));
        }
    }
    require(row_ulp <= 1U && column_ulp <= 1U,
            std::string(fixture.name)
                + " native F64 axis path differs by more than one output ULP");
    std::cout << fixture.name << " native=" << native.name()
              << " f64_rows_max_ulp=" << row_ulp
              << " f64_columns_max_ulp=" << column_ulp << '\n';
}

void test_native_f64_axes() {
    const auto fixtures = dsmvc::numerical::axis_fixtures();
    const dsmvc::CpuExecutor native(dsmvc::CpuPath::automatic);
    check_native_f64_axes(fixtures[5], native);
    check_native_f64_axes(fixtures[6], native);
}

struct MixedResult {
    std::string ordered_hash;
    std::string production_hash;
    double maximum_absolute = 0.0;
    std::uint32_t maximum_ulp = 0U;
};

[[nodiscard]] MixedResult evaluate_f64_2d(
    const dsmvc::AxisPlan &horizontal, const dsmvc::AxisPlan &vertical,
    std::uint32_t input_seed, const dsmvc::CpuExecutor &executor) {
    const auto input_stride = horizontal.source_size + 2;
    const auto output_stride = horizontal.destination_size + 3;
    const auto packed_input = dsmvc::numerical::make_normal_input(
        static_cast<std::size_t>(vertical.source_size)
            * static_cast<std::size_t>(horizontal.source_size),
        input_seed);
    std::vector<float> input(
        static_cast<std::size_t>(vertical.source_size)
            * static_cast<std::size_t>(input_stride),
        7.0F);
    for (std::int32_t row = 0; row < vertical.source_size; ++row) {
        std::copy_n(
            packed_input.data()
                + static_cast<std::ptrdiff_t>(row) * horizontal.source_size,
            horizontal.source_size,
            input.data() + static_cast<std::ptrdiff_t>(row) * input_stride);
    }

    const auto intermediate_stride = horizontal.destination_size;
    std::vector<double> intermediate(
        static_cast<std::size_t>(vertical.source_size)
            * static_cast<std::size_t>(intermediate_stride));
    std::vector<double> source(
        static_cast<std::size_t>(horizontal.source_size));
    for (std::int32_t row = 0; row < vertical.source_size; ++row) {
        for (std::int32_t column = 0; column < horizontal.source_size;
             ++column) {
            source[static_cast<std::size_t>(column)] = static_cast<double>(
                input[static_cast<std::ptrdiff_t>(row) * input_stride + column]);
        }
        dsmvc::detail::inverse_axis_f64_ordered(
            horizontal, source.data(), 1,
            intermediate.data()
                + static_cast<std::ptrdiff_t>(row) * intermediate_stride,
            1);
    }

    std::vector<double> ordered_double(
        static_cast<std::size_t>(vertical.destination_size)
            * static_cast<std::size_t>(horizontal.destination_size));
    std::vector<double> column(
        static_cast<std::size_t>(vertical.destination_size));
    for (std::int32_t x = 0; x < horizontal.destination_size; ++x) {
        dsmvc::detail::inverse_axis_f64_ordered(
            vertical, intermediate.data() + x, intermediate_stride,
            column.data(), 1);
        for (std::int32_t y = 0; y < vertical.destination_size; ++y) {
            ordered_double[static_cast<std::size_t>(y)
                               * static_cast<std::size_t>(
                                   horizontal.destination_size)
                           + static_cast<std::size_t>(x)] =
                column[static_cast<std::size_t>(y)];
        }
    }
    std::vector<float> ordered(ordered_double.begin(), ordered_double.end());

    std::vector<float> production(
        static_cast<std::size_t>(vertical.destination_size)
            * static_cast<std::size_t>(output_stride),
        -31.0F);
    executor.inverse_2d(
        horizontal, vertical, input.data(), input_stride,
        production.data(), output_stride);
    std::vector<float> packed_production;
    packed_production.reserve(ordered.size());
    for (std::int32_t row = 0; row < vertical.destination_size; ++row) {
        const auto *begin = production.data()
            + static_cast<std::ptrdiff_t>(row) * output_stride;
        packed_production.insert(
            packed_production.end(), begin,
            begin + horizontal.destination_size);
        for (std::int32_t column_index = horizontal.destination_size;
             column_index < output_stride; ++column_index) {
            require(production[static_cast<std::ptrdiff_t>(row) * output_stride
                               + column_index] == -31.0F,
                    "mixed-axis execution overwrote output padding");
        }
    }
    const auto difference = dsmvc::numerical::compare_outputs<float, float>(
        ordered, packed_production);
    require(difference.nonfinite == 0U,
            "mixed-axis execution produced nonfinite output");
    std::uint32_t maximum_ulp = 0U;
    for (std::size_t index = 0; index < ordered.size(); ++index) {
        maximum_ulp = std::max(
            maximum_ulp,
            dsmvc::numerical::float_ulp_distance(
                ordered[index], packed_production[index]));
    }
    return {
        dsmvc::numerical::output_hash<float>(ordered),
        dsmvc::numerical::output_hash<float>(packed_production),
        difference.maximum_absolute,
        maximum_ulp,
    };
}

void test_mixed_axis(bool emit_goldens) {
    const auto safe = dsmvc::build_axis_plan(
        dsmvc::numerical::mixed_horizontal_request());
    const auto risky = dsmvc::build_axis_plan(
        dsmvc::numerical::conditioned_lanczos2_request(
            dsmvc::F64Mode::automatic));
    require(!safe.requires_float64() && risky.requires_float64(),
            "mixed-axis fixture selected the wrong precision routes");
    const dsmvc::CpuExecutor native(dsmvc::CpuPath::automatic);
    const auto result = evaluate_f64_2d(
        safe, risky, dsmvc::numerical::mixed_input_seed, native);
    if (emit_goldens) {
        std::cout << "mixed-safe-risk-2d ordered=" << result.ordered_hash
                  << " production=" << result.production_hash
                  << " max_abs=" << result.maximum_absolute
                  << " max_ulp=" << result.maximum_ulp << '\n';
    } else {
        require(result.ordered_hash
                    == dsmvc::numerical::mixed_ordered_output_hash,
                "mixed-axis ordered oracle hash drifted: "
                    + result.ordered_hash);
        require(result.production_hash
                    == dsmvc::numerical::mixed_production_output_hash,
                "mixed-axis production hash drifted: "
                    + result.production_hash);
    }
    require(result.maximum_ulp <= 1U,
            "mixed-axis result differs by more than one output ULP");

    const auto reversed = evaluate_f64_2d(
        risky, safe, dsmvc::numerical::mixed_input_seed ^ 0x5a17c3e9U,
        native);
    const auto fixtures = dsmvc::numerical::axis_fixtures();
    const auto forced = dsmvc::build_axis_plan(fixtures[5].request);
    const auto both_f64 = evaluate_f64_2d(
        forced, forced, dsmvc::numerical::mixed_input_seed ^ 0xc64f2d01U,
        native);
    std::cout << "mixed-risk-safe-2d native=" << native.name()
              << " max_abs=" << reversed.maximum_absolute
              << " max_ulp=" << reversed.maximum_ulp << '\n'
              << "forced-f64-f64-2d native=" << native.name()
              << " max_abs=" << both_f64.maximum_absolute
              << " max_ulp=" << both_f64.maximum_ulp << '\n';
    if (native.path() == dsmvc::CpuPath::neon) {
        require(reversed.maximum_ulp == 0U && both_f64.maximum_ulp == 0U,
                "NEON F64 2D path differs from ordered Double");
    } else {
        require(reversed.maximum_ulp <= 1U && both_f64.maximum_ulp <= 1U,
                "F64 2D path differs by more than one output ULP");
    }
}

void test_ordered_f64_qr_anchor() {
    const auto plan = dsmvc::build_axis_plan(
        dsmvc::numerical::conditioned_lanczos2_request(
            dsmvc::F64Mode::automatic));
    std::vector<double> input(static_cast<std::size_t>(plan.source_size));
    std::uint32_t state = 0x9781a5c3U;
    for (double &value : input) {
        state ^= state << 13U;
        state ^= state >> 17U;
        state ^= state << 5U;
        const float sample = static_cast<float>(state & 0xffffU) / 65535.0F;
        value = static_cast<double>(sample);
    }
    std::vector<double> output(static_cast<std::size_t>(plan.destination_size));
    dsmvc::detail::inverse_axis_f64_ordered(
        plan, input.data(), 1, output.data(), 1);
    constexpr double first_qr = 74.22802437585693;
    constexpr double last_qr = -119.91860242539072;
    const double error = std::max(
        std::abs(output.front() - first_qr),
        std::abs(output.back() - last_qr));
    const double condition_scaled_bound = 4.0
        * std::numeric_limits<double>::epsilon()
        / plan.normal_rcond
        * std::max({1.0, std::abs(first_qr), std::abs(last_qr)});
    std::cout << "ordered-f64 QR max_endpoint_error=" << error
              << " condition_scaled_bound=" << condition_scaled_bound << '\n';
    require(error <= condition_scaled_bound,
            "ordered F64 oracle regressed against the independent QR anchor");
}

} // namespace

int main(int argc, char **argv) {
    try {
        bool emit_goldens = false;
        if (argc == 2 && std::string_view(argv[1]) == "--emit-goldens") {
            emit_goldens = true;
        } else if (argc != 1) {
            throw std::invalid_argument(
                "usage: dsmvc_numerical_contract_tests [--emit-goldens]");
        }

        test_axis_fixture_goldens(emit_goldens);
        test_cache_accounting();
        test_malformed_retained_metadata();
        test_native_tail_controls();
        test_native_column_controls();
        test_native_f64_axes();
        test_mixed_axis(emit_goldens);
        test_ordered_f64_qr_anchor();
        std::cout << "dsmvc numerical contract tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception &error) {
        std::cerr << "dsmvc numerical contract test failure: "
                  << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
