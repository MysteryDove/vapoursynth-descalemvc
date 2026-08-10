#include <dsmvc/engine.hpp>

#include "axis_plan_internal.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <numbers>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

// This inverse-only planner follows the descale sampling geometry and banded
// LDLT strategy. It omits forward projection data because dsmvc only executes
// the inverse solve.

namespace dsmvc {
namespace {

constexpr std::size_t plan_cache_entries = 2048U;
constexpr std::size_t plan_cache_bytes = 256U * 1024U * 1024U;
constexpr std::size_t geometry_cache_entries = 4096U;
constexpr std::size_t geometry_cache_bytes = 256U * 1024U * 1024U;

struct PlanKey {
    std::int32_t source_size;
    std::int32_t destination_size;
    std::uint64_t active_length;
    std::uint64_t shift;
    KernelKind kind;
    std::int32_t taps;
    std::uint64_t b;
    std::uint64_t c;
    BorderMode border;
    F64Mode f64_mode;

    friend bool operator==(const PlanKey &, const PlanKey &) = default;
};

struct GeometryKey {
    std::int32_t source_size;
    std::int32_t destination_size;
    std::uint64_t active_length;
    std::uint64_t shift;
    std::int32_t support;
    BorderMode border;

    friend bool operator==(const GeometryKey &, const GeometryKey &) = default;
};

struct KeyHash {
    template <class Key>
    [[nodiscard]] std::size_t operator()(const Key &key) const noexcept {
        std::size_t hash = 1469598103934665603ULL;
        const auto mix = [&hash](std::uint64_t value) {
            hash ^= static_cast<std::size_t>(value);
            hash *= 1099511628211ULL;
        };
        mix(static_cast<std::uint32_t>(key.source_size));
        mix(static_cast<std::uint32_t>(key.destination_size));
        mix(key.active_length);
        mix(key.shift);
        if constexpr (std::is_same_v<Key, PlanKey>) {
            mix(static_cast<std::uint8_t>(key.kind));
            mix(static_cast<std::uint32_t>(key.taps));
            mix(key.b);
            mix(key.c);
            mix(static_cast<std::uint8_t>(key.f64_mode));
        } else {
            mix(static_cast<std::uint32_t>(key.support));
        }
        mix(static_cast<std::uint8_t>(key.border));
        return hash;
    }
};

struct AxisGeometry {
    std::int32_t source_size = 0;
    std::int32_t destination_size = 0;
    std::int32_t support = 0;
    std::vector<double> distances;
    std::vector<std::uint32_t> row_offsets;
    std::vector<std::int32_t> unique_indices;
    std::vector<std::int32_t> tap_slots;

    [[nodiscard]] std::size_t storage_bytes() const noexcept {
        return sizeof(*this)
            + distances.capacity() * sizeof(double)
            + row_offsets.capacity() * sizeof(std::uint32_t)
            + unique_indices.capacity() * sizeof(std::int32_t)
            + tap_slots.capacity() * sizeof(std::int32_t);
    }
};

template <class Key, class Value, class Hash>
class SingleFlightLru {
    enum class State : std::uint8_t { building, ready, failed };

    struct Slot {
        State state = State::building;
        std::shared_ptr<const Value> value;
        std::exception_ptr error;
        std::size_t bytes = 0U;
        bool resident = false;
        typename std::list<Key>::iterator lru;
        std::condition_variable changed;
    };

public:
    SingleFlightLru(std::size_t maximum_entries, std::size_t maximum_bytes)
        : maximum_entries_(maximum_entries), maximum_bytes_(maximum_bytes) {}

    template <class Builder, class SizeFunction>
    [[nodiscard]] std::shared_ptr<const Value> get(
        const Key &key, Builder builder, SizeFunction size_function, bool &hit) {
        std::shared_ptr<Slot> slot;
        {
            std::unique_lock lock(mutex_);
            if (const auto found = entries_.find(key); found != entries_.end()) {
                hit = true;
                slot = found->second;
                slot->changed.wait(lock, [&] { return slot->state != State::building; });
                if (slot->state == State::failed) std::rethrow_exception(slot->error);
                touch(*slot);
                return slot->value;
            }
            hit = false;
            slot = std::make_shared<Slot>();
            entries_.emplace(key, slot);
        }

        std::shared_ptr<const Value> value;
        std::size_t bytes = 0U;
        try {
            value = builder();
            bytes = size_function(*value);
        } catch (...) {
            const auto failure = std::current_exception();
            {
                const std::scoped_lock lock(mutex_);
                slot->state = State::failed;
                slot->error = failure;
                if (const auto found = entries_.find(key);
                    found != entries_.end() && found->second == slot) {
                    entries_.erase(found);
                }
            }
            slot->changed.notify_all();
            std::rethrow_exception(failure);
        }

        {
            const std::scoped_lock lock(mutex_);
            slot->value = value;
            slot->bytes = bytes;
            slot->state = State::ready;
            const auto found = entries_.find(key);
            const bool is_current = found != entries_.end()
                && found->second == slot;
            if (is_current && maximum_entries_ != 0U
                && bytes <= maximum_bytes_) {
                lru_.push_front(key);
                slot->lru = lru_.begin();
                slot->resident = true;
                resident_bytes_ += bytes;
                evict();
            } else if (is_current) {
                entries_.erase(found);
            }
        }
        slot->changed.notify_all();
        return value;
    }

    void clear() {
        const std::scoped_lock lock(mutex_);
        entries_.clear();
        lru_.clear();
        resident_bytes_ = 0U;
    }

    [[nodiscard]] std::pair<std::size_t, std::size_t> snapshot() const {
        const std::scoped_lock lock(mutex_);
        return {lru_.size(), resident_bytes_};
    }

private:
    void touch(Slot &slot) {
        if (!slot.resident) return;
        lru_.splice(lru_.begin(), lru_, slot.lru);
        slot.lru = lru_.begin();
    }

    void evict() {
        while (lru_.size() > maximum_entries_ || resident_bytes_ > maximum_bytes_) {
            const Key victim = lru_.back();
            lru_.pop_back();
            const auto found = entries_.find(victim);
            if (found == entries_.end()) continue;
            const auto &slot = found->second;
            if (slot->state != State::ready || !slot->resident) continue;
            slot->resident = false;
            resident_bytes_ -= slot->bytes;
            entries_.erase(found);
        }
    }

    std::size_t maximum_entries_;
    std::size_t maximum_bytes_;
    mutable std::mutex mutex_;
    std::unordered_map<Key, std::shared_ptr<Slot>, Hash> entries_;
    std::list<Key> lru_;
    std::size_t resident_bytes_ = 0U;
};

struct DoubleCsr {
    std::vector<std::uint32_t> offsets;
    std::vector<std::int32_t> indices;
    std::vector<double> weights;
};

std::atomic<std::uint64_t> plan_hits{};
std::atomic<std::uint64_t> plan_builds{};
std::atomic<std::uint64_t> geometry_hits{};
std::atomic<std::uint64_t> geometry_builds{};

SingleFlightLru<PlanKey, AxisPlan, KeyHash> &plan_cache() {
    static SingleFlightLru<PlanKey, AxisPlan, KeyHash> cache{
        plan_cache_entries, plan_cache_bytes};
    return cache;
}

SingleFlightLru<GeometryKey, AxisGeometry, KeyHash> &geometry_cache() {
    static SingleFlightLru<GeometryKey, AxisGeometry, KeyHash> cache{
        geometry_cache_entries, geometry_cache_bytes};
    return cache;
}

[[nodiscard]] double round_half_up(double value) noexcept {
    return value < 0.0 ? std::floor(value + 0.5)
                       : std::floor(value + 0.49999999999999994);
}

[[nodiscard]] constexpr double square(double value) noexcept {
    return value * value;
}

[[nodiscard]] constexpr double cube(double value) noexcept {
    return value * value * value;
}

[[nodiscard]] double sinc(double value) noexcept {
    if (value == 0.0) return 1.0;
    const double scaled = std::numbers::pi * value;
    return std::sin(scaled) / scaled;
}

[[nodiscard]] double filter_weight(
    const KernelSpec &kernel, double distance) noexcept {
    double x = std::abs(distance);
    switch (kernel.kind) {
    case KernelKind::bilinear:
        return std::max(1.0 - x, 0.0);
    case KernelKind::bicubic:
        if (x < 1.0) {
            return ((12.0 - 9.0 * kernel.b - 6.0 * kernel.c) * cube(x)
                    + (-18.0 + 12.0 * kernel.b + 6.0 * kernel.c) * square(x)
                    + (6.0 - 2.0 * kernel.b)) / 6.0;
        }
        if (x < 2.0) {
            return ((-kernel.b - 6.0 * kernel.c) * cube(x)
                    + (6.0 * kernel.b + 30.0 * kernel.c) * square(x)
                    + (-12.0 * kernel.b - 48.0 * kernel.c) * x
                    + (8.0 * kernel.b + 24.0 * kernel.c)) / 6.0;
        }
        return 0.0;
    case KernelKind::lanczos:
        return kernel.taps > 0 && x < static_cast<double>(kernel.taps)
            ? sinc(x) * sinc(x / static_cast<double>(kernel.taps)) : 0.0;
    case KernelKind::spline16:
        if (x < 1.0) {
            return 1.0 - x / 5.0 - 9.0 * square(x) / 5.0 + cube(x);
        }
        if (x < 2.0) {
            x -= 1.0;
            return -7.0 * x / 15.0 + 4.0 * square(x) / 5.0 - cube(x) / 3.0;
        }
        return 0.0;
    case KernelKind::spline36:
        if (x < 1.0) {
            return 1.0 - 3.0 * x / 209.0 - 453.0 * square(x) / 209.0
                + 13.0 * cube(x) / 11.0;
        }
        if (x < 2.0) {
            x -= 1.0;
            return -156.0 * x / 209.0 + 270.0 * square(x) / 209.0
                - 6.0 * cube(x) / 11.0;
        }
        if (x < 3.0) {
            x -= 2.0;
            return 26.0 * x / 209.0 - 45.0 * square(x) / 209.0
                + cube(x) / 11.0;
        }
        return 0.0;
    case KernelKind::spline64:
        if (x < 1.0) {
            return 1.0 - 3.0 * x / 2911.0 - 6387.0 * square(x) / 2911.0
                + 49.0 * cube(x) / 41.0;
        }
        if (x < 2.0) {
            x -= 1.0;
            return -2328.0 * x / 2911.0 + 4032.0 * square(x) / 2911.0
                - 24.0 * cube(x) / 41.0;
        }
        if (x < 3.0) {
            x -= 2.0;
            return 582.0 * x / 2911.0 - 1008.0 * square(x) / 2911.0
                + 6.0 * cube(x) / 41.0;
        }
        if (x < 4.0) {
            x -= 3.0;
            return -97.0 * x / 2911.0 + 168.0 * square(x) / 2911.0
                - cube(x) / 41.0;
        }
        return 0.0;
    case KernelKind::custom:
        return 0.0;
    }
    return 0.0;
}

[[nodiscard]] std::size_t checked_elements(
    std::int32_t rows, std::int32_t width, const char *name) {
    const auto row_count = static_cast<std::size_t>(rows);
    const auto row_width = static_cast<std::size_t>(width);
    if (row_width != 0U
        && row_count > std::numeric_limits<std::size_t>::max() / row_width) {
        throw std::length_error(std::string{name} + " is too large");
    }
    return row_count * row_width;
}

[[nodiscard]] std::int32_t border_index(
    double pixel_center, std::int32_t size, BorderMode border) {
    const double index = std::floor(pixel_center);
    constexpr double int64_limit = 0x1p63;
    if (!std::isfinite(index) || index < -int64_limit
        || index >= int64_limit) {
        throw std::out_of_range(
            "shift places filter support outside the 64-bit pixel grid");
    }
    return detail::padding_index(
        static_cast<std::int64_t>(index), size, border);
}

[[nodiscard]] std::int32_t filter_support(const AxisRequest &request) {
    if (request.source_size <= 0 || request.destination_size <= 0) {
        throw std::invalid_argument("axis dimensions must be positive");
    }
    if (!(request.active_length > 0.0) || !std::isfinite(request.active_length)
        || !std::isfinite(request.shift)) {
        throw std::invalid_argument(
            "active length and shift must be finite, with positive active length");
    }
    switch (request.border) {
    case BorderMode::zero:
    case BorderMode::repeat:
    case BorderMode::reflect101:
    case BorderMode::symmetric:
    case BorderMode::mirror: break;
    default: throw std::invalid_argument("padding mode is invalid");
    }
    switch (request.f64_mode) {
    case F64Mode::automatic:
    case F64Mode::float32_only:
    case F64Mode::float64_only: break;
    default: throw std::invalid_argument("f64 mode is invalid");
    }
    std::int32_t support = 0;
    switch (request.kernel.kind) {
    case KernelKind::bilinear: support = 1; break;
    case KernelKind::bicubic:
    case KernelKind::spline16: support = 2; break;
    case KernelKind::spline36: support = 3; break;
    case KernelKind::spline64: support = 4; break;
    case KernelKind::lanczos:
    case KernelKind::custom: support = request.kernel.taps; break;
    }
    if (support <= 0
        || support > (std::numeric_limits<std::int32_t>::max() - 1) / 2) {
        throw std::invalid_argument("filter support is invalid or too large");
    }
    return support;
}

[[nodiscard]] PlanKey plan_key(const AxisRequest &request) noexcept {
    const bool bicubic = request.kernel.kind == KernelKind::bicubic;
    const bool lanczos = request.kernel.kind == KernelKind::lanczos;
    return {
        request.source_size,
        request.destination_size,
        std::bit_cast<std::uint64_t>(request.active_length),
        std::bit_cast<std::uint64_t>(request.shift),
        request.kernel.kind,
        lanczos ? request.kernel.taps : 0,
        bicubic ? std::bit_cast<std::uint64_t>(request.kernel.b) : 0U,
        bicubic ? std::bit_cast<std::uint64_t>(request.kernel.c) : 0U,
        request.border,
        request.f64_mode,
    };
}

[[nodiscard]] GeometryKey geometry_key(
    const AxisRequest &request, std::int32_t support) noexcept {
    return {
        request.source_size,
        request.destination_size,
        std::bit_cast<std::uint64_t>(request.active_length),
        std::bit_cast<std::uint64_t>(request.shift),
        support,
        request.border,
    };
}

[[nodiscard]] AxisGeometry build_geometry(
    const AxisRequest &request, std::int32_t support) {
    const std::int32_t tap_count = 2 * support;
    AxisGeometry geometry;
    geometry.source_size = request.source_size;
    geometry.destination_size = request.destination_size;
    geometry.support = support;
    const auto elements = checked_elements(
        request.source_size, tap_count, "inverse geometry");
    geometry.distances.resize(elements);
    geometry.tap_slots.assign(elements, -1);
    geometry.row_offsets.reserve(
        static_cast<std::size_t>(request.source_size) + 1U);
    geometry.unique_indices.reserve(elements);
    geometry.row_offsets.push_back(0U);

    const double ratio = static_cast<double>(request.source_size)
        / request.active_length;
    std::vector<std::int32_t> tap_indices(static_cast<std::size_t>(tap_count));
    std::vector<std::int32_t> unique_indices;
    unique_indices.reserve(static_cast<std::size_t>(tap_count));
    for (std::int32_t row = 0; row < request.source_size; ++row) {
        const double position = (static_cast<double>(row) + 0.5) / ratio
            + request.shift;
        const double begin = round_half_up(
            position - static_cast<double>(support)) + 0.5;
        const std::size_t row_base = static_cast<std::size_t>(row)
            * static_cast<std::size_t>(tap_count);
        unique_indices.clear();
        for (std::int32_t tap = 0; tap < tap_count; ++tap) {
            const double center = begin + static_cast<double>(tap);
            const std::size_t offset = row_base + static_cast<std::size_t>(tap);
            geometry.distances[offset] = center - position;
            const auto index = border_index(
                center, request.destination_size, request.border);
            tap_indices[static_cast<std::size_t>(tap)] = index;
            if (index >= 0
                && std::find(unique_indices.begin(), unique_indices.end(), index)
                    == unique_indices.end()) {
                unique_indices.push_back(index);
            }
        }
        std::sort(unique_indices.begin(), unique_indices.end());
        if (geometry.unique_indices.size() + unique_indices.size()
            > std::numeric_limits<std::uint32_t>::max()) {
            throw std::length_error("inverse geometry coefficient table is too large");
        }
        geometry.unique_indices.insert(
            geometry.unique_indices.end(), unique_indices.begin(), unique_indices.end());
        for (std::int32_t tap = 0; tap < tap_count; ++tap) {
            const auto index = tap_indices[static_cast<std::size_t>(tap)];
            if (index < 0) continue;
            const auto slot = std::lower_bound(
                unique_indices.begin(), unique_indices.end(), index);
            geometry.tap_slots[row_base + static_cast<std::size_t>(tap)] =
                static_cast<std::int32_t>(
                    std::distance(unique_indices.begin(), slot));
        }
        geometry.row_offsets.push_back(static_cast<std::uint32_t>(
            geometry.unique_indices.size()));
    }
    return geometry;
}

[[nodiscard]] std::shared_ptr<const AxisGeometry> get_geometry(
    const AxisRequest &request, std::int32_t support) {
    bool hit = false;
    auto geometry = geometry_cache().get(
        geometry_key(request, support),
        [&] {
            geometry_builds.fetch_add(1U, std::memory_order_relaxed);
            return std::make_shared<const AxisGeometry>(
                build_geometry(request, support));
        },
        [](const AxisGeometry &value) { return value.storage_bytes(); }, hit);
    if (hit) geometry_hits.fetch_add(1U, std::memory_order_relaxed);
    return geometry;
}

[[nodiscard]] DoubleCsr make_descale_matrix(
    const AxisRequest &request, std::int32_t support,
    const AxisGeometry &geometry, const CustomKernel &custom_kernel) {
    const std::int32_t tap_count = 2 * support;
    DoubleCsr result;
    result.offsets.reserve(static_cast<std::size_t>(request.source_size) + 1U);
    result.indices.reserve(geometry.unique_indices.size());
    result.weights.reserve(geometry.unique_indices.size());
    result.offsets.push_back(0U);

    const bool custom = request.kernel.kind == KernelKind::custom;
    std::vector<double> tap_weights(static_cast<std::size_t>(tap_count));
    std::vector<double> coalesced(static_cast<std::size_t>(tap_count));
    std::vector<bool> seen(static_cast<std::size_t>(tap_count));
    for (std::int32_t row = 0; row < request.source_size; ++row) {
        const std::size_t row_base = static_cast<std::size_t>(row)
            * static_cast<std::size_t>(tap_count);
        double total = 0.0;
        for (std::int32_t tap = 0; tap < tap_count; ++tap) {
            const double distance = geometry.distances[
                row_base + static_cast<std::size_t>(tap)];
            const double weight = custom
                ? custom_kernel(std::abs(distance))
                : filter_weight(request.kernel, distance);
            if (!std::isfinite(weight)) {
                throw std::runtime_error("filter returned a non-finite value");
            }
            tap_weights[static_cast<std::size_t>(tap)] = weight;
            total += weight;
        }
        if (!std::isfinite(total) || total == 0.0) {
            throw std::runtime_error("filter produced a zero or non-finite weight sum");
        }

        const auto index_begin = geometry.row_offsets[static_cast<std::size_t>(row)];
        const auto index_end = geometry.row_offsets[static_cast<std::size_t>(row) + 1U];
        const auto unique_count = static_cast<std::size_t>(index_end - index_begin);
        std::fill_n(coalesced.begin(), unique_count, 0.0);
        std::fill_n(seen.begin(), unique_count, false);
        for (std::int32_t tap = 0; tap < tap_count; ++tap) {
            const auto slot = geometry.tap_slots[
                row_base + static_cast<std::size_t>(tap)];
            if (slot < 0) continue;
            const double weight = tap_weights[static_cast<std::size_t>(tap)] / total;
            if (weight == 0.0) continue;
            const auto output_slot = static_cast<std::size_t>(slot);
            coalesced[output_slot] += weight;
            seen[output_slot] = true;
        }
        for (std::size_t slot = 0; slot < unique_count; ++slot) {
            if (!seen[slot]) continue;
            result.indices.push_back(geometry.unique_indices[index_begin + slot]);
            result.weights.push_back(coalesced[slot]);
        }
        if (result.indices.size() > std::numeric_limits<std::uint32_t>::max()) {
            throw std::length_error("inverse coefficient table is too large");
        }
        result.offsets.push_back(
            static_cast<std::uint32_t>(result.indices.size()));
    }
    return result;
}

[[nodiscard]] DoubleCsr transpose_csr(
    const DoubleCsr &input, std::int32_t rows, std::int32_t columns) {
    DoubleCsr result;
    result.offsets.assign(static_cast<std::size_t>(columns) + 1U, 0U);
    for (const auto index : input.indices) {
        ++result.offsets[static_cast<std::size_t>(index) + 1U];
    }
    for (std::int32_t column = 0; column < columns; ++column) {
        result.offsets[static_cast<std::size_t>(column) + 1U] +=
            result.offsets[static_cast<std::size_t>(column)];
    }
    result.indices.resize(input.indices.size());
    result.weights.resize(input.weights.size());
    auto cursor = result.offsets;
    for (std::int32_t row = 0; row < rows; ++row) {
        const auto begin = input.offsets[static_cast<std::size_t>(row)];
        const auto end = input.offsets[static_cast<std::size_t>(row) + 1U];
        for (auto offset = begin; offset < end; ++offset) {
            const auto column = static_cast<std::size_t>(input.indices[offset]);
            const auto target = cursor[column]++;
            result.indices[target] = row;
            result.weights[target] = input.weights[offset];
        }
    }
    return result;
}

[[nodiscard]] std::vector<double> form_normal_bands(
    const DoubleCsr &transpose, std::int32_t columns,
    std::int32_t half_bandwidth) {
    const auto width = static_cast<std::size_t>(columns);
    std::vector<double> bands(
        (static_cast<std::size_t>(half_bandwidth) + 1U) * width, 0.0);
    // Iterate source observations in ascending order for each native-pixel
    // pair. This preserves original descale's Float64 accumulation order while
    // storing and factorizing only the nonzero band.
    for (std::int32_t left_row = 0; left_row < columns; ++left_row) {
        const auto maximum_distance = std::min(
            half_bandwidth, columns - left_row - 1);
        for (std::int32_t distance = 0;
             distance <= maximum_distance; ++distance) {
            const auto right_row = left_row + distance;
            auto left = transpose.offsets[static_cast<std::size_t>(left_row)];
            const auto left_end =
                transpose.offsets[static_cast<std::size_t>(left_row) + 1U];
            auto right = transpose.offsets[static_cast<std::size_t>(right_row)];
            const auto right_end =
                transpose.offsets[static_cast<std::size_t>(right_row) + 1U];
            double sum = 0.0;
            while (left < left_end && right < right_end) {
                const auto left_index = transpose.indices[left];
                const auto right_index = transpose.indices[right];
                if (left_index < right_index) {
                    ++left;
                } else if (right_index < left_index) {
                    ++right;
                } else {
                    sum += transpose.weights[left] * transpose.weights[right];
                    ++left;
                    ++right;
                }
            }
            bands[static_cast<std::size_t>(distance) * width
                  + static_cast<std::size_t>(left_row)] = sum;
        }
    }
    return bands;
}

struct SymmetricBandConditionBounds {
    double one_norm = 0.0;
    double diagonal_dominance_rcond = 0.0;
};

[[nodiscard]] SymmetricBandConditionBounds symmetric_band_condition_bounds(
    const std::vector<double> &bands, std::int32_t n,
    std::int32_t half_bandwidth, std::vector<double> &workspace) {
    const auto width = static_cast<std::size_t>(n);
    workspace.assign(width, 0.0);
    for (std::int32_t column = 0; column < n; ++column) {
        workspace[static_cast<std::size_t>(column)] +=
            std::abs(bands[static_cast<std::size_t>(column)]);
        const auto available = std::min(half_bandwidth, n - column - 1);
        for (std::int32_t distance = 1; distance <= available; ++distance) {
            const double value = std::abs(
                bands[static_cast<std::size_t>(distance) * width
                      + static_cast<std::size_t>(column)]);
            workspace[static_cast<std::size_t>(column)] += value;
            workspace[static_cast<std::size_t>(column + distance)] += value;
        }
    }
    const double one_norm = *std::max_element(
        workspace.begin(), workspace.end());
    double minimum_margin = std::numeric_limits<double>::infinity();
    for (std::int32_t row = 0; row < n; ++row) {
        const double diagonal = std::abs(
            bands[static_cast<std::size_t>(row)]);
        minimum_margin = std::min(
            minimum_margin,
            2.0 * diagonal - workspace[static_cast<std::size_t>(row)]);
    }
    return {
        one_norm,
        minimum_margin > 0.0 && one_norm > 0.0
            ? minimum_margin / one_norm : 0.0,
    };
}

[[nodiscard]] double reciprocal_condition_lower_bound(
    double matrix_one_norm, const std::vector<double> &factors,
    std::int32_t n, std::int32_t half_bandwidth,
    std::vector<double> &workspace) {
    const auto width = static_cast<std::size_t>(n);
    constexpr double epsilon = std::numeric_limits<double>::epsilon();

    // For L = I + E, |L^-1| <= (I - |E|)^-1. Row and column sums of
    // that comparison inverse give conservative infinity/one-norm bounds
    // without running the iterative estimator for clearly safe plans.
    workspace.assign(width, 1.0);
    double inverse_l_infinity_norm = 1.0;
    double inverse_diagonal_norm = 0.0;
    for (std::int32_t row = 0; row < n; ++row) {
        double sum = 1.0;
        const auto available = std::min(half_bandwidth, row);
        for (std::int32_t distance = 1; distance <= available; ++distance) {
            const auto column = row - distance;
            sum += std::abs(
                       factors[static_cast<std::size_t>(distance) * width
                               + static_cast<std::size_t>(column)])
                * workspace[static_cast<std::size_t>(column)];
        }
        workspace[static_cast<std::size_t>(row)] = sum;
        inverse_l_infinity_norm = std::max(inverse_l_infinity_norm, sum);
        inverse_diagonal_norm = std::max(
            inverse_diagonal_norm,
            1.0 / std::abs(
                factors[static_cast<std::size_t>(row)] + epsilon));
    }

    std::fill(workspace.begin(), workspace.end(), 1.0);
    double inverse_l_one_norm = 1.0;
    for (std::int32_t column = n - 1; column >= 0; --column) {
        double sum = 1.0;
        const auto available = std::min(half_bandwidth, n - column - 1);
        for (std::int32_t distance = 1; distance <= available; ++distance) {
            sum += std::abs(
                       factors[static_cast<std::size_t>(distance) * width
                               + static_cast<std::size_t>(column)])
                * workspace[static_cast<std::size_t>(column + distance)];
        }
        workspace[static_cast<std::size_t>(column)] = sum;
        inverse_l_one_norm = std::max(inverse_l_one_norm, sum);
    }

    const double inverse_norm_bound = inverse_l_one_norm
        * inverse_diagonal_norm * inverse_l_infinity_norm;
    if (!(matrix_one_norm > 0.0) || !(inverse_norm_bound > 0.0)
        || !std::isfinite(matrix_one_norm)
        || !std::isfinite(inverse_norm_bound)) {
        return 0.0;
    }
    return std::clamp(
        1.0 / (matrix_one_norm * inverse_norm_bound), 0.0, 1.0);
}

void factor_banded_ldlt(
    std::vector<double> &bands, std::int32_t n,
    std::int32_t half_bandwidth) noexcept {
    const auto width = static_cast<std::size_t>(n);
    constexpr double epsilon = std::numeric_limits<double>::epsilon();
    for (std::int32_t i = 0; i < n; ++i) {
        const auto end = std::min(half_bandwidth + 1, n - i);
        const double pivot = bands[static_cast<std::size_t>(i)] + epsilon;
        for (std::int32_t distance = 1; distance < end; ++distance) {
            const auto upper = static_cast<std::size_t>(distance) * width
                + static_cast<std::size_t>(i);
            const double multiplier = bands[upper] / pivot;
            for (std::int32_t offset = 0; offset < end - distance; ++offset) {
                bands[static_cast<std::size_t>(offset) * width
                      + static_cast<std::size_t>(i + distance)] -= multiplier
                    * bands[static_cast<std::size_t>(distance + offset) * width
                            + static_cast<std::size_t>(i)];
            }
        }
        const double inverse_pivot = 1.0 / pivot;
        for (std::int32_t distance = 1; distance < end; ++distance) {
            bands[static_cast<std::size_t>(distance) * width
                  + static_cast<std::size_t>(i)] *= inverse_pivot;
        }
    }
}

void solve_banded_ldlt(
    const std::vector<double> &factors, std::int32_t n,
    std::int32_t half_bandwidth, double *values,
    std::ptrdiff_t stride) noexcept {
    const auto width = static_cast<std::size_t>(n);
    constexpr double epsilon = std::numeric_limits<double>::epsilon();
    for (std::int32_t i = 0; i < n; ++i) {
        double value = values[static_cast<std::ptrdiff_t>(i) * stride];
        const auto available = std::min(half_bandwidth, i);
        for (std::int32_t distance = available; distance >= 1; --distance) {
            value -= factors[static_cast<std::size_t>(distance) * width
                             + static_cast<std::size_t>(i - distance)]
                * values[static_cast<std::ptrdiff_t>(i - distance) * stride];
        }
        values[static_cast<std::ptrdiff_t>(i) * stride] = value;
    }
    for (std::int32_t i = 0; i < n; ++i) {
        values[static_cast<std::ptrdiff_t>(i) * stride] /=
            factors[static_cast<std::size_t>(i)] + epsilon;
    }
    for (std::int32_t i = n - 2; i >= 0; --i) {
        double value = values[static_cast<std::ptrdiff_t>(i) * stride];
        const auto available = std::min(half_bandwidth, n - i - 1);
        for (std::int32_t distance = available; distance >= 1; --distance) {
            value -= factors[static_cast<std::size_t>(distance) * width
                             + static_cast<std::size_t>(i)]
                * values[static_cast<std::ptrdiff_t>(i + distance) * stride];
        }
        values[static_cast<std::ptrdiff_t>(i) * stride] = value;
    }
}

[[nodiscard]] double estimate_inverse_one_norm(
    const std::vector<double> &factors, std::int32_t n,
    std::int32_t half_bandwidth) {
    const auto size = static_cast<std::size_t>(n);
    std::vector<double> x(size, 1.0 / static_cast<double>(n));
    std::vector<double> y(size);
    std::vector<double> z(size);
    double estimate = 0.0;
    std::size_t previous_index = size;

    for (int iteration = 0; iteration < 8; ++iteration) {
        y = x;
        solve_banded_ldlt(factors, n, half_bandwidth, y.data(), 1);
        const double next_estimate = std::accumulate(
            y.begin(), y.end(), 0.0,
            [](double sum, double value) { return sum + std::abs(value); });
        estimate = std::max(estimate, next_estimate);

        std::transform(y.begin(), y.end(), z.begin(), [](double value) {
            return value < 0.0 ? -1.0 : 1.0;
        });
        solve_banded_ldlt(factors, n, half_bandwidth, z.data(), 1);
        const auto maximum = std::max_element(
            z.begin(), z.end(), [](double left, double right) {
                return std::abs(left) < std::abs(right);
            });
        const auto index = static_cast<std::size_t>(
            std::distance(z.begin(), maximum));
        const double dot = std::inner_product(z.begin(), z.end(), x.begin(), 0.0);
        if (index == previous_index || std::abs(*maximum) <= dot) break;

        std::fill(x.begin(), x.end(), 0.0);
        x[index] = 1.0;
        previous_index = index;
    }
    return estimate;
}

[[nodiscard]] double reciprocal_condition_estimate(
    double matrix_norm, const std::vector<double> &factors, std::int32_t n,
    std::int32_t half_bandwidth) {
    const double inverse_norm = estimate_inverse_one_norm(
        factors, n, half_bandwidth);
    if (!(matrix_norm > 0.0) || !(inverse_norm > 0.0)
        || !std::isfinite(matrix_norm) || !std::isfinite(inverse_norm)) {
        return 0.0;
    }
    return std::clamp(1.0 / (matrix_norm * inverse_norm), 0.0, 1.0);
}

[[nodiscard]] AxisPlan build_axis_plan_impl(
    const AxisRequest &request, const CustomKernel &custom_kernel) {
    const auto support = filter_support(request);
    if (request.kernel.kind == KernelKind::custom && !custom_kernel) {
        throw std::invalid_argument("custom kernel callback is missing");
    }
    const auto geometry = get_geometry(request, support);
    auto matrix = make_descale_matrix(
        request, support, *geometry, custom_kernel);
    auto transpose = transpose_csr(
        matrix, request.source_size, request.destination_size);
    const auto half_bandwidth = std::min(
        2 * support - 1, request.destination_size - 1);
    auto normal_bands = form_normal_bands(
        transpose, request.destination_size, half_bandwidth);
    auto factors = normal_bands;
    factor_banded_ldlt(factors, request.destination_size, half_bandwidth);

    constexpr double float64_rcond_threshold = 1.0e-4;
    std::vector<double> condition_workspace;
    const auto condition_bounds = symmetric_band_condition_bounds(
        normal_bands, request.destination_size, half_bandwidth,
        condition_workspace);
    double normal_rcond = condition_bounds.diagonal_dominance_rcond;
    if (normal_rcond < float64_rcond_threshold) {
        normal_rcond = reciprocal_condition_lower_bound(
            condition_bounds.one_norm, factors, request.destination_size,
            half_bandwidth, condition_workspace);
        if (normal_rcond < float64_rcond_threshold) {
            normal_rcond = reciprocal_condition_estimate(
                condition_bounds.one_norm, factors,
                request.destination_size, half_bandwidth);
        }
    }
    const bool requires_float64 = request.f64_mode == F64Mode::float64_only
        || (request.f64_mode == F64Mode::automatic
            && normal_rcond < float64_rcond_threshold);

    AxisPlan plan;
    plan.source_size = request.source_size;
    plan.destination_size = request.destination_size;
    plan.support = support;
    plan.half_bandwidth = half_bandwidth;
    plan.active_length = request.active_length;
    plan.shift = request.shift;
    plan.normal_rcond = normal_rcond;
    plan.transpose_offsets = std::move(transpose.offsets);
    plan.transpose_indices = std::move(transpose.indices);
    plan.transpose_weights.resize(transpose.weights.size());
    std::transform(
        transpose.weights.begin(), transpose.weights.end(),
        plan.transpose_weights.begin(),
        [](double weight) { return static_cast<float>(weight); });
    if (requires_float64) {
        plan.normal_inf_norm = condition_bounds.one_norm;
        plan.transpose_weights_f64 = transpose.weights;
        plan.normal_bands_f64 = normal_bands;
        plan.ldlt_bands_f64 = factors;
        plan.inverse_diagonal_f64.resize(
            static_cast<std::size_t>(request.destination_size));
        constexpr double epsilon = std::numeric_limits<double>::epsilon();
        for (std::int32_t i = 0; i < request.destination_size; ++i) {
            plan.inverse_diagonal_f64[static_cast<std::size_t>(i)] =
                1.0 / (factors[static_cast<std::size_t>(i)] + epsilon);
        }
    }

    const auto width = static_cast<std::size_t>(request.destination_size);
    const auto factor_count = static_cast<std::size_t>(half_bandwidth) * width;
    plan.lower_ld.assign(factor_count, 0.0F);
    plan.upper_l.assign(factor_count, 0.0F);
    plan.inverse_diagonal.resize(width);
    constexpr double epsilon = std::numeric_limits<double>::epsilon();
    for (std::int32_t i = 0; i < request.destination_size; ++i) {
        const double diagonal = factors[static_cast<std::size_t>(i)];
        plan.inverse_diagonal[static_cast<std::size_t>(i)] =
            static_cast<float>(1.0 / (diagonal + epsilon));
        const auto available = std::min(
            half_bandwidth, request.destination_size - i - 1);
        for (std::int32_t distance = 1; distance <= available; ++distance) {
            const float l = static_cast<float>(
                factors[static_cast<std::size_t>(distance) * width
                        + static_cast<std::size_t>(i)]);
            plan.upper_l[static_cast<std::size_t>(distance - 1) * width
                         + static_cast<std::size_t>(i)] = l;
            const auto row = i + distance;
            plan.lower_ld[static_cast<std::size_t>(distance - 1) * width
                          + static_cast<std::size_t>(row)] =
                static_cast<float>(
                    factors[static_cast<std::size_t>(distance) * width
                            + static_cast<std::size_t>(i)] * diagonal);
        }
    }
    if (!plan.valid()) throw std::runtime_error("failed to build inverse axis plan");
    return plan;
}

template <std::int32_t FixedHalfBandwidth>
void inverse_axis_impl(
    const AxisPlan &plan, const float *input, std::ptrdiff_t input_stride,
    float *output, std::ptrdiff_t output_stride) noexcept {
    const auto n = plan.destination_size;
    const auto band = FixedHalfBandwidth == 0
        ? plan.half_bandwidth : FixedHalfBandwidth;
    const auto width = static_cast<std::size_t>(n);
    for (std::int32_t i = 0; i < n; ++i) {
        float sum = 0.0F;
        for (auto offset = plan.transpose_offsets[static_cast<std::size_t>(i)];
             offset < plan.transpose_offsets[static_cast<std::size_t>(i) + 1U];
             ++offset) {
            sum += plan.transpose_weights[offset]
                * input[static_cast<std::ptrdiff_t>(
                    plan.transpose_indices[offset]) * input_stride];
        }
        const auto available = std::min(band, i);
        for (std::int32_t distance = available; distance >= 1; --distance) {
            sum -= plan.lower_ld[static_cast<std::size_t>(distance - 1) * width
                                 + static_cast<std::size_t>(i)]
                * output[static_cast<std::ptrdiff_t>(i - distance) * output_stride];
        }
        output[static_cast<std::ptrdiff_t>(i) * output_stride] =
            sum * plan.inverse_diagonal[static_cast<std::size_t>(i)];
    }
    for (std::int32_t i = n - 2; i >= 0; --i) {
        float sum = 0.0F;
        const auto available = std::min(band, n - i - 1);
        if constexpr (FixedHalfBandwidth == 3) {
            for (std::int32_t distance = 1; distance <= available; ++distance) {
                sum += plan.upper_l[static_cast<std::size_t>(distance - 1) * width
                                    + static_cast<std::size_t>(i)]
                    * output[static_cast<std::ptrdiff_t>(i + distance)
                             * output_stride];
            }
        } else {
            for (std::int32_t distance = available; distance >= 1; --distance) {
                sum += plan.upper_l[static_cast<std::size_t>(distance - 1) * width
                                    + static_cast<std::size_t>(i)]
                    * output[static_cast<std::ptrdiff_t>(i + distance)
                             * output_stride];
            }
        }
        output[static_cast<std::ptrdiff_t>(i) * output_stride] -= sum;
    }
}

} // namespace

namespace detail {

std::int32_t padding_index(
    std::int64_t index, std::int32_t size, BorderMode mode) {
    if (size <= 0) throw std::invalid_argument("padding size must be positive");
    switch (mode) {
    case BorderMode::zero:
    case BorderMode::repeat:
    case BorderMode::reflect101:
    case BorderMode::symmetric:
    case BorderMode::mirror: break;
    default: throw std::invalid_argument("padding mode is invalid");
    }
    const auto extent = static_cast<std::int64_t>(size);
    if (index >= 0 && index < extent) {
        return static_cast<std::int32_t>(index);
    }

    switch (mode) {
    case BorderMode::zero:
        return -1;
    case BorderMode::repeat:
        return index < 0 ? 0 : size - 1;
    case BorderMode::reflect101: {
        if (size == 1) return 0;
        const auto period = 2 * (extent - 1);
        auto mapped = index % period;
        if (mapped < 0) mapped += period;
        return static_cast<std::int32_t>(
            mapped < extent ? mapped : period - mapped);
    }
    case BorderMode::symmetric: {
        const auto period = 2 * extent;
        auto mapped = index % period;
        if (mapped < 0) mapped += period;
        return static_cast<std::int32_t>(
            mapped < extent ? mapped : period - 1 - mapped);
    }
    case BorderMode::mirror:
        if (index < -extent || index >= 2 * extent) return -1;
        return static_cast<std::int32_t>(
            index < 0 ? -index - 1 : 2 * extent - index - 1);
    }
    throw std::invalid_argument("padding mode is invalid");
}

void inverse_axis_f32_ordered(
    const AxisPlan &plan,
    const float *input, std::ptrdiff_t input_stride,
    float *output, std::ptrdiff_t output_stride) noexcept {
    const auto n = plan.destination_size;
    const auto width = static_cast<std::size_t>(n);
    for (std::int32_t i = 0; i < n; ++i) {
        float value = 0.0F;
        for (auto offset = plan.transpose_offsets[static_cast<std::size_t>(i)];
             offset < plan.transpose_offsets[static_cast<std::size_t>(i) + 1U];
             ++offset) {
            value = std::fma(
                plan.transpose_weights[offset],
                input[static_cast<std::ptrdiff_t>(
                    plan.transpose_indices[offset]) * input_stride],
                value);
        }
        const auto available = std::min(plan.half_bandwidth, i);
        for (std::int32_t distance = available; distance >= 1; --distance) {
            value = std::fma(
                -plan.lower_ld[static_cast<std::size_t>(distance - 1) * width
                               + static_cast<std::size_t>(i)],
                output[static_cast<std::ptrdiff_t>(i - distance) * output_stride],
                value);
        }
        output[static_cast<std::ptrdiff_t>(i) * output_stride] = value
            * plan.inverse_diagonal[static_cast<std::size_t>(i)];
    }
    for (std::int32_t i = n - 2; i >= 0; --i) {
        float value = output[static_cast<std::ptrdiff_t>(i) * output_stride];
        const auto available = std::min(plan.half_bandwidth, n - i - 1);
        if (plan.half_bandwidth == 3) {
            for (std::int32_t distance = 1; distance <= available; ++distance) {
                value = std::fma(
                    -plan.upper_l[
                        static_cast<std::size_t>(distance - 1) * width
                        + static_cast<std::size_t>(i)],
                    output[static_cast<std::ptrdiff_t>(i + distance)
                           * output_stride],
                    value);
            }
        } else {
            for (std::int32_t distance = available; distance >= 1; --distance) {
                value = std::fma(
                    -plan.upper_l[
                        static_cast<std::size_t>(distance - 1) * width
                        + static_cast<std::size_t>(i)],
                    output[static_cast<std::ptrdiff_t>(i + distance)
                           * output_stride],
                    value);
            }
        }
        output[static_cast<std::ptrdiff_t>(i) * output_stride] = value;
    }
}

void inverse_axis_f64_ordered(
    const AxisPlan &plan,
    const double *input, std::ptrdiff_t input_stride,
    double *output, std::ptrdiff_t output_stride) noexcept {
    const auto n = plan.destination_size;
    const auto width = static_cast<std::size_t>(n);
    const bool retained = plan.requires_float64();
    for (std::int32_t i = 0; i < n; ++i) {
        double value = 0.0;
        for (auto offset = plan.transpose_offsets[static_cast<std::size_t>(i)];
             offset < plan.transpose_offsets[static_cast<std::size_t>(i) + 1U];
             ++offset) {
            const double weight = retained
                ? plan.transpose_weights_f64[offset]
                : static_cast<double>(plan.transpose_weights[offset]);
            value = std::fma(
                weight,
                input[static_cast<std::ptrdiff_t>(
                    plan.transpose_indices[offset]) * input_stride],
                value);
        }
        const auto available = std::min(plan.half_bandwidth, i);
        for (std::int32_t distance = available; distance >= 1; --distance) {
            const double factor = retained
                ? plan.ldlt_bands_f64[
                      static_cast<std::size_t>(distance) * width
                      + static_cast<std::size_t>(i - distance)]
                : static_cast<double>(plan.lower_ld[
                      static_cast<std::size_t>(distance - 1) * width
                      + static_cast<std::size_t>(i)]);
            value = std::fma(
                -factor,
                output[static_cast<std::ptrdiff_t>(i - distance) * output_stride],
                value);
        }
        if (retained) {
            output[static_cast<std::ptrdiff_t>(i) * output_stride] = value;
        } else {
            output[static_cast<std::ptrdiff_t>(i) * output_stride] = value
                * static_cast<double>(
                    plan.inverse_diagonal[static_cast<std::size_t>(i)]);
        }
    }
    if (retained) {
        for (std::int32_t i = 0; i < n; ++i) {
            output[static_cast<std::ptrdiff_t>(i) * output_stride] *=
                plan.inverse_diagonal_f64[static_cast<std::size_t>(i)];
        }
    }
    for (std::int32_t i = n - 2; i >= 0; --i) {
        double value = output[static_cast<std::ptrdiff_t>(i) * output_stride];
        const auto available = std::min(plan.half_bandwidth, n - i - 1);
        for (std::int32_t distance = available; distance >= 1; --distance) {
            const double factor = retained
                ? plan.ldlt_bands_f64[
                      static_cast<std::size_t>(distance) * width
                      + static_cast<std::size_t>(i)]
                : static_cast<double>(plan.upper_l[
                      static_cast<std::size_t>(distance - 1) * width
                      + static_cast<std::size_t>(i)]);
            value = std::fma(
                -factor,
                output[static_cast<std::ptrdiff_t>(i + distance) * output_stride],
                value);
        }
        output[static_cast<std::ptrdiff_t>(i) * output_stride] = value;
    }
}

void inverse_axis_f64(
    const AxisPlan &plan,
    const double *input, std::ptrdiff_t input_stride,
    double *output, std::ptrdiff_t output_stride) noexcept {
    const auto n = plan.destination_size;
    const auto width = static_cast<std::size_t>(n);
    if (plan.requires_float64()) {
        for (std::int32_t i = 0; i < n; ++i) {
            double sum = 0.0;
            for (auto offset = plan.transpose_offsets[static_cast<std::size_t>(i)];
                 offset < plan.transpose_offsets[static_cast<std::size_t>(i) + 1U];
                 ++offset) {
                sum += plan.transpose_weights_f64[offset]
                    * input[static_cast<std::ptrdiff_t>(
                        plan.transpose_indices[offset]) * input_stride];
            }
            output[static_cast<std::ptrdiff_t>(i) * output_stride] = sum;
        }
        solve_banded_ldlt(
            plan.ldlt_bands_f64, n, plan.half_bandwidth,
            output, output_stride);
        return;
    }

    for (std::int32_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (auto offset = plan.transpose_offsets[static_cast<std::size_t>(i)];
             offset < plan.transpose_offsets[static_cast<std::size_t>(i) + 1U];
             ++offset) {
            sum += static_cast<double>(plan.transpose_weights[offset])
                * input[static_cast<std::ptrdiff_t>(
                    plan.transpose_indices[offset]) * input_stride];
        }
        const auto available = std::min(plan.half_bandwidth, i);
        for (std::int32_t distance = available; distance >= 1; --distance) {
            sum -= static_cast<double>(plan.lower_ld[
                       static_cast<std::size_t>(distance - 1) * width
                       + static_cast<std::size_t>(i)])
                * output[static_cast<std::ptrdiff_t>(i - distance) * output_stride];
        }
        output[static_cast<std::ptrdiff_t>(i) * output_stride] = sum
            * static_cast<double>(
                plan.inverse_diagonal[static_cast<std::size_t>(i)]);
    }
    for (std::int32_t i = n - 2; i >= 0; --i) {
        double sum = 0.0;
        const auto available = std::min(plan.half_bandwidth, n - i - 1);
        for (std::int32_t distance = available; distance >= 1; --distance) {
            sum += static_cast<double>(plan.upper_l[
                       static_cast<std::size_t>(distance - 1) * width
                       + static_cast<std::size_t>(i)])
                * output[static_cast<std::ptrdiff_t>(i + distance) * output_stride];
        }
        output[static_cast<std::ptrdiff_t>(i) * output_stride] -= sum;
    }
}

} // namespace detail

bool AxisPlan::valid() const noexcept {
    if (source_size <= 0 || destination_size <= 0 || support <= 0
        || half_bandwidth < 0 || half_bandwidth >= destination_size
        || !(active_length > 0.0) || !std::isfinite(active_length)
        || !std::isfinite(shift) || !std::isfinite(normal_rcond)
        || normal_rcond < 0.0 || normal_rcond > 1.0
        || !std::isfinite(normal_inf_norm) || normal_inf_norm < 0.0) {
        return false;
    }
    const auto destination = static_cast<std::size_t>(destination_size);
    const auto factors = static_cast<std::size_t>(half_bandwidth) * destination;
    if (transpose_offsets.size() != destination + 1U
        || transpose_indices.size() != transpose_weights.size()
        || transpose_offsets.empty() || transpose_offsets.front() != 0U
        || transpose_offsets.back() != transpose_indices.size()
        || lower_ld.size() != factors || upper_l.size() != factors
        || inverse_diagonal.size() != destination) {
        return false;
    }

    const bool has_float64 = !ldlt_bands_f64.empty();
    if (has_float64 != !transpose_weights_f64.empty()
        || has_float64 != !normal_bands_f64.empty()
        || has_float64 != !inverse_diagonal_f64.empty()) {
        return false;
    }
    if (has_float64 ? !(normal_inf_norm > 0.0) : normal_inf_norm != 0.0) {
        return false;
    }
    if (has_float64
        && (transpose_weights_f64.size() != transpose_weights.size()
            || normal_bands_f64.size()
                != (static_cast<std::size_t>(half_bandwidth) + 1U)
                    * destination
            || ldlt_bands_f64.size()
                != (static_cast<std::size_t>(half_bandwidth) + 1U)
                    * destination
            || inverse_diagonal_f64.size() != destination)) {
        return false;
    }

    const auto nonzeros = transpose_indices.size();
    for (std::size_t row = 0; row < destination; ++row) {
        const auto begin = static_cast<std::size_t>(transpose_offsets[row]);
        const auto end = static_cast<std::size_t>(transpose_offsets[row + 1U]);
        if (begin > end || end > nonzeros) return false;

        std::int32_t previous = -1;
        for (auto offset = begin; offset < end; ++offset) {
            const auto source = transpose_indices[offset];
            if (source < 0 || source >= source_size || source <= previous
                || !std::isfinite(transpose_weights[offset])) {
                return false;
            }
            previous = source;
        }
    }

    const bool float32_valid =
        std::all_of(lower_ld.begin(), lower_ld.end(), [](float value) {
               return std::isfinite(value);
           })
        && std::all_of(upper_l.begin(), upper_l.end(), [](float value) {
               return std::isfinite(value);
           })
        && std::all_of(
            inverse_diagonal.begin(), inverse_diagonal.end(), [](float value) {
                return std::isfinite(value);
            });
    if (!float32_valid || !has_float64) return float32_valid;
    return std::all_of(
               transpose_weights_f64.begin(), transpose_weights_f64.end(),
               [](double value) { return std::isfinite(value); })
        && std::all_of(
               normal_bands_f64.begin(), normal_bands_f64.end(),
               [](double value) { return std::isfinite(value); })
        && std::all_of(
               ldlt_bands_f64.begin(), ldlt_bands_f64.end(),
               [](double value) { return std::isfinite(value); })
        && std::all_of(
               inverse_diagonal_f64.begin(), inverse_diagonal_f64.end(),
               [](double value) { return std::isfinite(value); });
}

bool AxisPlan::requires_float64() const noexcept {
    return !ldlt_bands_f64.empty();
}

std::size_t AxisPlan::storage_bytes() const noexcept {
    return sizeof(*this)
        + transpose_offsets.capacity() * sizeof(std::uint32_t)
        + transpose_indices.capacity() * sizeof(std::int32_t)
        + transpose_weights.capacity() * sizeof(float)
        + lower_ld.capacity() * sizeof(float)
        + upper_l.capacity() * sizeof(float)
        + inverse_diagonal.capacity() * sizeof(float)
        + transpose_weights_f64.capacity() * sizeof(double)
        + normal_bands_f64.capacity() * sizeof(double)
        + ldlt_bands_f64.capacity() * sizeof(double)
        + inverse_diagonal_f64.capacity() * sizeof(double);
}

AxisPlan build_axis_plan(
    const AxisRequest &request, const CustomKernel &custom_kernel) {
    plan_builds.fetch_add(1U, std::memory_order_relaxed);
    return build_axis_plan_impl(request, custom_kernel);
}

std::shared_ptr<const AxisPlan> get_or_build_axis_plan(
    const AxisRequest &request, const CustomKernel &custom_kernel) {
    if (request.kernel.kind == KernelKind::custom) {
        return std::make_shared<const AxisPlan>(
            build_axis_plan(request, custom_kernel));
    }
    bool hit = false;
    auto plan = plan_cache().get(
        plan_key(request),
        [&] {
            return std::make_shared<const AxisPlan>(
                build_axis_plan(request, custom_kernel));
        },
        [](const AxisPlan &value) { return value.storage_bytes(); }, hit);
    if (hit) plan_hits.fetch_add(1U, std::memory_order_relaxed);
    return plan;
}

PlannerCacheStats planner_cache_stats() {
    const auto [plan_entry_count, plan_bytes] = plan_cache().snapshot();
    const auto [geometry_entry_count, geometry_bytes] = geometry_cache().snapshot();
    return {
        plan_hits.load(std::memory_order_relaxed),
        plan_builds.load(std::memory_order_relaxed),
        geometry_hits.load(std::memory_order_relaxed),
        geometry_builds.load(std::memory_order_relaxed),
        plan_entry_count,
        plan_bytes,
        geometry_entry_count,
        geometry_bytes,
    };
}

void clear_planner_caches() {
    plan_cache().clear();
    geometry_cache().clear();
    plan_hits.store(0U, std::memory_order_relaxed);
    plan_builds.store(0U, std::memory_order_relaxed);
    geometry_hits.store(0U, std::memory_order_relaxed);
    geometry_builds.store(0U, std::memory_order_relaxed);
}

void inverse_axis_f32(
    const AxisPlan &plan, const float *input, std::ptrdiff_t input_stride,
    float *output, std::ptrdiff_t output_stride) {
    if (!plan.valid() || input == nullptr || output == nullptr
        || input_stride == 0 || output_stride == 0) {
        throw std::invalid_argument("invalid inverse axis arguments");
    }
    if (plan.requires_float64()) {
        std::vector<double> source(static_cast<std::size_t>(plan.source_size));
        std::vector<double> destination(
            static_cast<std::size_t>(plan.destination_size));
        for (std::int32_t i = 0; i < plan.source_size; ++i) {
            const float value =
                input[static_cast<std::ptrdiff_t>(i) * input_stride];
            if (!std::isfinite(value)) {
                throw std::runtime_error(
                    "Float64 inverse axis input contains NaN or infinity");
            }
            source[static_cast<std::size_t>(i)] = static_cast<double>(value);
        }
        detail::inverse_axis_f64(
            plan, source.data(), 1, destination.data(), 1);
        for (std::int32_t i = 0; i < plan.destination_size; ++i) {
            const double value = destination[static_cast<std::size_t>(i)];
            const float converted = static_cast<float>(value);
            if (!std::isfinite(value) || !std::isfinite(converted)) {
                throw std::runtime_error(
                    "Float64 inverse axis produced NaN or infinity");
            }
            output[static_cast<std::ptrdiff_t>(i) * output_stride] = converted;
        }
        return;
    }
    if (plan.half_bandwidth == 1) {
        inverse_axis_impl<1>(plan, input, input_stride, output, output_stride);
    } else if (plan.half_bandwidth == 3) {
        inverse_axis_impl<3>(plan, input, input_stride, output, output_stride);
    } else {
        inverse_axis_impl<0>(plan, input, input_stride, output, output_stride);
    }
}

} // namespace dsmvc
