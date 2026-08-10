#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace dsmvc::detail {

[[nodiscard]] inline std::size_t checked_size_add(
    std::size_t left, std::size_t right, std::string_view label) {
    if (left > std::numeric_limits<std::size_t>::max() - right) {
        throw std::length_error(std::string{label} + " size overflow");
    }
    return left + right;
}

[[nodiscard]] inline std::size_t checked_size_product(
    std::size_t left, std::size_t right, std::string_view label) {
    if (left != 0U
        && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::length_error(std::string{label} + " size overflow");
    }
    return left * right;
}

[[nodiscard]] inline std::size_t checked_size_round_up(
    std::size_t value, std::size_t alignment, std::string_view label) {
    if (alignment == 0U) {
        throw std::invalid_argument("size alignment must be nonzero");
    }
    const auto adjusted = checked_size_add(value, alignment - 1U, label);
    return adjusted / alignment * alignment;
}

[[nodiscard]] inline std::int32_t checked_size_i32(
    std::size_t value, std::string_view label) {
    if (value > static_cast<std::size_t>(
                    std::numeric_limits<std::int32_t>::max())) {
        throw std::length_error(std::string{label}
                                + " exceeds the CPU plan range");
    }
    return static_cast<std::int32_t>(value);
}

} // namespace dsmvc::detail
