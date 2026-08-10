#pragma once

#include <dsmvc/engine.hpp>

#include "axis_plan_internal.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace dsmvc::numerical {

class StableHasher {
public:
    void add_string(std::string_view value) noexcept {
        add_unsigned(static_cast<std::uint64_t>(value.size()), 8U);
        for (const unsigned char byte : value) mix(byte);
    }

    template <class T>
    void add(T value) noexcept {
        if constexpr (std::is_enum_v<T>) {
            add(static_cast<std::underlying_type_t<T>>(value));
        } else if constexpr (std::is_same_v<T, bool>) {
            add_unsigned(value ? 1U : 0U, 1U);
        } else if constexpr (std::is_integral_v<T>) {
            using Unsigned = std::make_unsigned_t<T>;
            add_unsigned(
                static_cast<std::uint64_t>(static_cast<Unsigned>(value)),
                sizeof(T));
        } else if constexpr (std::is_same_v<T, float>) {
            add_unsigned(std::bit_cast<std::uint32_t>(value), sizeof(value));
        } else if constexpr (std::is_same_v<T, double>) {
            add_unsigned(std::bit_cast<std::uint64_t>(value), sizeof(value));
        } else {
            static_assert(std::is_arithmetic_v<T>, "unsupported hash value");
        }
    }

    template <class T>
    void add_span(std::span<const T> values) noexcept {
        add(static_cast<std::uint64_t>(values.size()));
        for (const T value : values) add(value);
    }

    [[nodiscard]] std::uint64_t value() const noexcept { return value_; }

    [[nodiscard]] std::string hex() const {
        std::ostringstream output;
        output << std::hex << std::setfill('0') << std::setw(16) << value_;
        return output.str();
    }

private:
    void mix(std::uint8_t byte) noexcept {
        value_ ^= byte;
        value_ *= 1099511628211ULL;
    }

    void add_unsigned(std::uint64_t value, std::size_t bytes) noexcept {
        for (std::size_t index = 0; index < bytes; ++index) {
            mix(static_cast<std::uint8_t>(value & 0xffU));
            value >>= 8U;
        }
    }

    std::uint64_t value_ = 1469598103934665603ULL;
};

struct AxisFixture {
    const char *name;
    AxisRequest request;
    std::uint32_t input_seed;
    std::int32_t expected_half_bandwidth;
    const char *f32_plan_hash;
    const char *complete_plan_hash;
    const char *ordered_output_hash;
    const char *production_output_hash;
};

[[nodiscard]] inline AxisRequest make_axis_request(
    std::int32_t source, std::int32_t destination, double active_length,
    double shift, KernelKind kernel, std::int32_t taps, BorderMode border,
    F64Mode f64_mode) {
    AxisRequest request;
    request.source_size = source;
    request.destination_size = destination;
    request.active_length = active_length;
    request.shift = shift;
    request.kernel.kind = kernel;
    request.kernel.taps = taps;
    request.kernel.b = 0.0;
    request.kernel.c = 0.5;
    request.border = border;
    request.f64_mode = f64_mode;
    return request;
}

[[nodiscard]] inline AxisRequest conditioned_lanczos2_request(F64Mode mode) {
    return make_axis_request(
        1080, 980, 978.1, 0.95, KernelKind::lanczos, 2,
        BorderMode::symmetric, mode);
}

[[nodiscard]] constexpr const char *complete_plan_golden(
    const char *default_hash, const char *apple_hash = nullptr) noexcept {
#if defined(__APPLE__)
    return apple_hash ? apple_hash : default_hash;
#else
    (void)apple_hash;
    return default_hash;
#endif
}

[[nodiscard]] inline std::array<AxisFixture, 7> axis_fixtures() {
    return {{
        {"b1-symmetric-tail",
         make_axis_request(41, 35, 35.0, 0.1875, KernelKind::bilinear, 3,
                           BorderMode::symmetric, F64Mode::float32_only),
         0xb1001001U, 1, "b7fa7ca018f14328", "61f3f18073fe9e11",
         "71ea8a545feca1b7", "ea1b09290d8d2d47"},
        {"b3-reflect101-tail",
         make_axis_request(47, 39, 39.0, -0.3125, KernelKind::bicubic, 3,
                           BorderMode::reflect101, F64Mode::float32_only),
         0xb3003003U, 3, "9a194da63434e8e1", "a8ee51581e483134",
         "c2ec387c8df33a7b", "7ec44696e0ae2e8c"},
        {"b5-repeat-tail",
         make_axis_request(59, 49, 49.0, 0.4375, KernelKind::lanczos, 3,
                           BorderMode::repeat, F64Mode::float32_only),
         0xb5005005U, 5, "368aafc55d47f98c",
         complete_plan_golden("d7d468d0dcc3798e", "0ed8c2aa402279dd"),
         "0230fd2270a8d447", "a3b5a9429aaac189"},
        {"b7-zero-tail",
         make_axis_request(67, 57, 57.0, -0.25, KernelKind::spline64, 3,
                           BorderMode::zero, F64Mode::float32_only),
         0xb7007007U, 7, "68da8f096af1407f", "dd8b79d05deb000e",
         "19b4db02b77aa1e6", "8333021572f444c1"},
        {"conditioned-f32-control",
         conditioned_lanczos2_request(F64Mode::float32_only),
         0xf3209781U, 3, "5bddb1a9d8c61225", "162056d561d75888",
         "97b0409984eb7606", "90fc1f98834e937e"},
        {"forced-f64-b3",
         make_axis_request(73, 61, 60.5, 0.125, KernelKind::bicubic, 3,
                           BorderMode::symmetric, F64Mode::float64_only),
         0xf6400002U, 3, "8192da268d0b6655", "b9ea390d346a2c8e",
         "d4b9c8ff994a2520", "d4b9c8ff994a2520"},
        {"automatic-risk-f64",
         conditioned_lanczos2_request(F64Mode::automatic),
         0xf6409781U, 3, "88de56a7458f03fd",
         complete_plan_golden("b0ceea661114a07b", "bef07be61c4ea5ed"),
         "1c332683a9d93688", "1c332683a9d93688"},
    }};
}

[[nodiscard]] inline AxisRequest mixed_horizontal_request() {
    return make_axis_request(
        17, 13, 12.75, 0.125, KernelKind::bicubic, 3,
        BorderMode::symmetric, F64Mode::float32_only);
}

inline constexpr std::uint32_t mixed_input_seed = 0x2df64b3aU;
inline constexpr std::string_view mixed_ordered_output_hash =
    "a9dcffe6649bd96e";
inline constexpr std::string_view mixed_production_output_hash =
    "a9dcffe6649bd96e";

[[nodiscard]] inline std::vector<float> make_normal_input(
    std::size_t count, std::uint32_t seed) {
    std::vector<float> result(count);
    std::uint32_t state = seed;
    for (float &value : result) {
        state ^= state << 13U;
        state ^= state >> 17U;
        state ^= state << 5U;
        std::uint32_t bits = 0x3f000000U | (state & 0x007fffffU);
        if ((state & 0x00800000U) != 0U) bits |= 0x80000000U;
        value = std::bit_cast<float>(bits);
    }
    return result;
}

template <class T>
[[nodiscard]] inline std::string output_hash(std::span<const T> output) {
    StableHasher hasher;
    hasher.add_string("dsmvc-ordered-output-v1");
    hasher.add_span(output);
    return hasher.hex();
}

[[nodiscard]] inline std::string f32_plan_hash(
    const AxisRequest &request, const AxisPlan &plan) {
    StableHasher hasher;
    hasher.add_string("dsmvc-f32-axis-plan-v1");
    hasher.add(request.source_size);
    hasher.add(request.destination_size);
    hasher.add(request.active_length);
    hasher.add(request.shift);
    hasher.add(request.kernel.kind);
    hasher.add(request.kernel.taps);
    hasher.add(request.kernel.b);
    hasher.add(request.kernel.c);
    hasher.add(request.border);
    hasher.add(request.f64_mode);
    hasher.add(plan.source_size);
    hasher.add(plan.destination_size);
    hasher.add(plan.support);
    hasher.add(plan.half_bandwidth);
    hasher.add(plan.active_length);
    hasher.add(plan.shift);
    hasher.add(plan.normal_rcond < 1.0e-4);
    hasher.add(plan.requires_float64());
    hasher.add_span<std::uint32_t>(plan.transpose_offsets);
    hasher.add_span<std::int32_t>(plan.transpose_indices);
    hasher.add_span<float>(plan.transpose_weights);
    hasher.add_span<float>(plan.lower_ld);
    hasher.add_span<float>(plan.upper_l);
    hasher.add_span<float>(plan.inverse_diagonal);
    return hasher.hex();
}

[[nodiscard]] inline std::string complete_plan_hash(
    const AxisRequest &request, const AxisPlan &plan) {
    StableHasher hasher;
    hasher.add_string("dsmvc-axis-plan-v1");
    hasher.add(request.source_size);
    hasher.add(request.destination_size);
    hasher.add(request.active_length);
    hasher.add(request.shift);
    hasher.add(request.kernel.kind);
    hasher.add(request.kernel.taps);
    hasher.add(request.kernel.b);
    hasher.add(request.kernel.c);
    hasher.add(request.border);
    hasher.add(request.f64_mode);
    hasher.add(plan.source_size);
    hasher.add(plan.destination_size);
    hasher.add(plan.support);
    hasher.add(plan.half_bandwidth);
    hasher.add(plan.active_length);
    hasher.add(plan.shift);
    hasher.add(plan.normal_rcond);
    hasher.add(plan.normal_inf_norm);
    hasher.add_span<std::uint32_t>(plan.transpose_offsets);
    hasher.add_span<std::int32_t>(plan.transpose_indices);
    hasher.add_span<float>(plan.transpose_weights);
    hasher.add_span<float>(plan.lower_ld);
    hasher.add_span<float>(plan.upper_l);
    hasher.add_span<float>(plan.inverse_diagonal);
    hasher.add_span<double>(plan.transpose_weights_f64);
    hasher.add_span<double>(plan.normal_bands_f64);
    hasher.add_span<double>(plan.ldlt_bands_f64);
    hasher.add_span<double>(plan.inverse_diagonal_f64);
    return hasher.hex();
}

struct NormalMatrixAudit {
    long double maximum_band_error = 0.0L;
    long double maximum_band_magnitude = 0.0L;
    long double reconstructed_inf_norm = 0.0L;
    long double inf_norm_error = 0.0L;
};

[[nodiscard]] inline NormalMatrixAudit audit_normal_matrix(
    const AxisPlan &plan) {
    NormalMatrixAudit result;
    const auto width = static_cast<std::size_t>(plan.destination_size);
    std::vector<long double> row_sums(width, 0.0L);
    for (std::int32_t left_row = 0; left_row < plan.destination_size;
         ++left_row) {
        const auto available = std::min(
            plan.half_bandwidth, plan.destination_size - left_row - 1);
        for (std::int32_t distance = 0; distance <= available; ++distance) {
            const auto right_row = left_row + distance;
            auto left = plan.transpose_offsets[static_cast<std::size_t>(left_row)];
            const auto left_end =
                plan.transpose_offsets[static_cast<std::size_t>(left_row) + 1U];
            auto right = plan.transpose_offsets[static_cast<std::size_t>(right_row)];
            const auto right_end =
                plan.transpose_offsets[static_cast<std::size_t>(right_row) + 1U];
            long double expected = 0.0L;
            while (left < left_end && right < right_end) {
                if (plan.transpose_indices[left] < plan.transpose_indices[right]) {
                    ++left;
                } else if (plan.transpose_indices[right]
                           < plan.transpose_indices[left]) {
                    ++right;
                } else {
                    expected += static_cast<long double>(
                                    plan.transpose_weights_f64[left])
                        * static_cast<long double>(
                                    plan.transpose_weights_f64[right]);
                    ++left;
                    ++right;
                }
            }
            const double retained = plan.normal_bands_f64[
                static_cast<std::size_t>(distance) * width
                + static_cast<std::size_t>(left_row)];
            result.maximum_band_error = std::max(
                result.maximum_band_error,
                std::abs(expected - static_cast<long double>(retained)));
            result.maximum_band_magnitude = std::max(
                result.maximum_band_magnitude, std::abs(expected));
            const long double magnitude = std::abs(expected);
            row_sums[static_cast<std::size_t>(left_row)] += magnitude;
            if (distance != 0) {
                row_sums[static_cast<std::size_t>(right_row)] += magnitude;
            }
        }
    }
    result.reconstructed_inf_norm = *std::max_element(
        row_sums.begin(), row_sums.end());
    result.inf_norm_error = std::abs(
        result.reconstructed_inf_norm
        - static_cast<long double>(plan.normal_inf_norm));
    return result;
}

struct Difference {
    double maximum_absolute = 0.0;
    std::size_t nonfinite = 0U;
};

[[nodiscard]] inline std::uint32_t ordered_float_bits(float value) noexcept {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    return (bits & 0x80000000U) != 0U ? ~bits : bits | 0x80000000U;
}

[[nodiscard]] inline std::uint32_t float_ulp_distance(
    float left, float right) noexcept {
    const auto lhs = ordered_float_bits(left);
    const auto rhs = ordered_float_bits(right);
    return lhs > rhs ? lhs - rhs : rhs - lhs;
}

template <class Left, class Right>
[[nodiscard]] inline Difference compare_outputs(
    std::span<const Left> left, std::span<const Right> right) {
    Difference result;
    if (left.size() != right.size()) {
        result.nonfinite = std::numeric_limits<std::size_t>::max();
        return result;
    }
    for (std::size_t index = 0; index < left.size(); ++index) {
        const double lhs = static_cast<double>(left[index]);
        const double rhs = static_cast<double>(right[index]);
        if (!std::isfinite(lhs) || !std::isfinite(rhs)) {
            ++result.nonfinite;
            continue;
        }
        result.maximum_absolute = std::max(
            result.maximum_absolute, std::abs(lhs - rhs));
    }
    return result;
}

} // namespace dsmvc::numerical
