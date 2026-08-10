#include <dsmvc/engine.hpp>

#include "axis_plan_internal.hpp"
#include "checked_size.hpp"
#include "cpu_packed.hpp"

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#elif defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif

#if defined(DSMVC_HAS_NEON_OBJECT) && defined(_WIN32) && defined(_M_ARM64)
#include <windows.h>
#elif defined(DSMVC_HAS_NEON_OBJECT) && defined(__linux__) && defined(__aarch64__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#elif defined(DSMVC_HAS_NEON_OBJECT) && defined(__APPLE__) && defined(__aarch64__)
#include <sys/sysctl.h>
#endif

namespace dsmvc {

namespace {

struct PackingCounters {
    std::atomic<std::uint64_t> executions{0U};
    std::atomic<std::uint64_t> waits{0U};
    std::atomic<std::uint64_t> wait_nanoseconds{0U};
    std::atomic<std::uint64_t> lazy_requests{0U};
    std::atomic<std::uint64_t> lazy_hits{0U};
    std::atomic<std::uint64_t> active{0U};
    std::atomic<std::uint64_t> maximum_active{0U};

    void begin_pack() noexcept {
        executions.fetch_add(1U, std::memory_order_relaxed);
        const auto current = active.fetch_add(1U, std::memory_order_relaxed) + 1U;
        auto maximum = maximum_active.load(std::memory_order_relaxed);
        while (maximum < current
               && !maximum_active.compare_exchange_weak(
                   maximum, current, std::memory_order_relaxed)) {}
    }

    void end_pack() noexcept {
        active.fetch_sub(1U, std::memory_order_relaxed);
    }

    [[nodiscard]] CpuPlanPackingStats snapshot() const noexcept {
        return {
            executions.load(std::memory_order_relaxed),
            waits.load(std::memory_order_relaxed),
            wait_nanoseconds.load(std::memory_order_relaxed),
            lazy_requests.load(std::memory_order_relaxed),
            lazy_hits.load(std::memory_order_relaxed),
            maximum_active.load(std::memory_order_relaxed),
        };
    }
};

class SharedPackedPlan final {
public:
    explicit SharedPackedPlan(std::shared_ptr<const AxisPlan> requested_axis,
                              const AxisPlan *requested_identity = nullptr)
        : axis_(std::move(requested_axis)),
          identity_(requested_identity ? requested_identity : axis_.get()) {}

    [[nodiscard]] const AxisPlan *identity() const noexcept { return identity_; }
    [[nodiscard]] const std::shared_ptr<const AxisPlan> &axis() const noexcept {
        return axis_;
    }

    [[nodiscard]] std::shared_ptr<const detail::PackedCpuPlan> get(
        PackingCounters &counters, bool lazy) {
        if (lazy) counters.lazy_requests.fetch_add(1U, std::memory_order_relaxed);
        bool waited = false;
        for (;;) {
            std::unique_lock lock(mutex_);
            if (packed_) {
                if (lazy) {
                    counters.lazy_hits.fetch_add(1U, std::memory_order_relaxed);
                }
                return packed_;
            }
            if (packing_) {
                if (!waited) {
                    counters.waits.fetch_add(1U, std::memory_order_relaxed);
                    waited = true;
                }
                const auto started = std::chrono::steady_clock::now();
                ready_.wait(lock, [&] { return !packing_; });
                const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - started).count();
                counters.wait_nanoseconds.fetch_add(
                    static_cast<std::uint64_t>(std::max<std::int64_t>(elapsed, 0)),
                    std::memory_order_relaxed);
                continue;
            }
            packing_ = true;
            lock.unlock();

            counters.begin_pack();
            try {
                auto packed = std::make_shared<const detail::PackedCpuPlan>(
                    detail::pack_cpu_plan(axis_, identity_));
                counters.end_pack();
                lock.lock();
                packed_ = std::move(packed);
                packing_ = false;
                auto result = packed_;
                lock.unlock();
                ready_.notify_all();
                return result;
            } catch (...) {
                counters.end_pack();
                lock.lock();
                packing_ = false;
                lock.unlock();
                ready_.notify_all();
                throw;
            }
        }
    }

private:
    std::shared_ptr<const AxisPlan> axis_;
    const AxisPlan *identity_ = nullptr;
    std::mutex mutex_;
    std::condition_variable ready_;
    std::shared_ptr<const detail::PackedCpuPlan> packed_;
    bool packing_ = false;
};

[[nodiscard]] std::shared_ptr<SharedPackedPlan> shared_packed_plan(
    const std::shared_ptr<const AxisPlan> &plan) {
    static std::mutex mutex;
    static std::unordered_map<const AxisPlan *, std::weak_ptr<SharedPackedPlan>> cache;
    const std::scoped_lock lock(mutex);
    if (const auto found = cache.find(plan.get()); found != cache.end()) {
        if (auto entry = found->second.lock(); entry && entry->axis() == plan) {
            return entry;
        }
        cache.erase(found);
    }
    auto entry = std::make_shared<SharedPackedPlan>(plan);
    cache.emplace(plan.get(), entry);
    if (cache.size() > 4096U) {
        std::erase_if(cache, [](const auto &candidate) {
            return candidate.second.expired();
        });
    }
    return entry;
}

class WorkerPool {
    struct JobState {
        explicit JobState(std::size_t count,
                          std::function<void(std::size_t)> function)
            : job(std::move(function)), task_count(count) {}

        std::function<void(std::size_t)> job;
        std::size_t task_count = 0U;
        std::atomic<std::size_t> next_task{1U};
        std::mutex mutex;
        std::exception_ptr error;
    };

public:
    explicit WorkerPool(std::size_t parallelism)
        : parallelism_(std::max<std::size_t>(parallelism, 1U)),
          start_barrier_(static_cast<std::ptrdiff_t>(parallelism_)),
          finish_barrier_(static_cast<std::ptrdiff_t>(parallelism_)) {
        workers_.reserve(parallelism_ - 1U);
        for (std::size_t index = 1; index < parallelism_; ++index) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    ~WorkerPool() {
        stopping_ = true;
        start_barrier_.arrive_and_wait();
        for (auto &worker : workers_) worker.join();
    }

    WorkerPool(const WorkerPool &) = delete;
    WorkerPool &operator=(const WorkerPool &) = delete;

    [[nodiscard]] std::size_t parallelism() const noexcept {
        return parallelism_;
    }

    bool try_run(std::size_t task_count,
                 std::function<void(std::size_t)> function) {
        task_count = std::min(task_count, parallelism_);
        if (task_count < 2U
            || in_use_.test_and_set(std::memory_order_acquire)) {
            return false;
        }

        std::shared_ptr<JobState> state;
        try {
            state = std::make_shared<JobState>(
                task_count, std::move(function));
        } catch (...) {
            in_use_.clear(std::memory_order_release);
            throw;
        }
        current_state_ = state;
        start_barrier_.arrive_and_wait();
        try {
            state->job(0U);
        } catch (...) {
            const std::scoped_lock lock(state->mutex);
            state->error = std::current_exception();
        }
        finish_barrier_.arrive_and_wait();
        current_state_.reset();
        in_use_.clear(std::memory_order_release);
        std::exception_ptr error;
        {
            const std::scoped_lock lock(state->mutex);
            error = state->error;
        }
        if (error) std::rethrow_exception(error);
        return true;
    }

private:
    void worker_loop() {
        for (;;) {
            start_barrier_.arrive_and_wait();
            if (stopping_) return;
            const auto state = current_state_;

            for (;;) {
                const auto task = state->next_task.fetch_add(
                    1U, std::memory_order_relaxed);
                if (task >= state->task_count) break;
                try {
                    state->job(task);
                } catch (...) {
                    const std::scoped_lock lock(state->mutex);
                    if (!state->error) state->error = std::current_exception();
                }
            }
            finish_barrier_.arrive_and_wait();
        }
    }

    std::size_t parallelism_ = 1U;
    std::barrier<> start_barrier_;
    std::barrier<> finish_barrier_;
    std::vector<std::thread> workers_;
    std::shared_ptr<JobState> current_state_;
    std::atomic_flag in_use_ = ATOMIC_FLAG_INIT;
    bool stopping_ = false;
};

[[nodiscard]] std::size_t cpu_parallelism(CpuPath path) noexcept {
#if defined(DSMVC_HAS_NEON_OBJECT)
    if (path != CpuPath::neon) return 1U;
#else
    if (path != CpuPath::avx2) return 1U;
#endif
    const auto hardware = std::max(std::thread::hardware_concurrency(), 1U);
    return std::min<std::size_t>(hardware, 4U);
}

[[nodiscard]] std::shared_ptr<WorkerPool> shared_worker_pool(CpuPath path) {
#if defined(DSMVC_HAS_NEON_OBJECT)
    if (path != CpuPath::neon) return {};
#else
    if (path != CpuPath::avx2) return {};
#endif
    static auto pool = std::make_shared<WorkerPool>(cpu_parallelism(path));
    return pool;
}

} // namespace

#if defined(DSMVC_HAS_AVX2_OBJECT)
void inverse_rows_f64_avx2(
    const AxisPlan &plan,
    const float *input, std::ptrdiff_t input_row_stride,
    float *output, std::ptrdiff_t output_row_stride,
    std::int32_t row_count);
void inverse_rows_to_f64_avx2(
    const AxisPlan &plan,
    const float *input, std::ptrdiff_t input_row_stride,
    double *output, std::ptrdiff_t output_row_stride,
    std::int32_t row_count);
void inverse_columns_f64_avx2(
    const AxisPlan &plan,
    const float *input, std::ptrdiff_t input_row_stride,
    float *output, std::ptrdiff_t output_row_stride,
    std::int32_t column_count);
void inverse_columns_from_f64_avx2(
    const AxisPlan &plan,
    const double *input, std::ptrdiff_t input_row_stride,
    float *output, std::ptrdiff_t output_row_stride,
    std::int32_t column_count);
void inverse_rows_avx2(const AxisPlan &plan,
                       const detail::PackedCpuPlan &packed,
                       const float *input, std::ptrdiff_t input_row_stride,
                       float *output, std::ptrdiff_t output_row_stride,
                       std::int32_t row_count);
void inverse_columns_avx2(const AxisPlan &plan,
                          const detail::PackedCpuPlan &packed,
                          const float *input, std::ptrdiff_t input_row_stride,
                          float *output, std::ptrdiff_t output_row_stride,
                          std::int32_t column_count);
void solve_rhs_columns_avx2(
    const AxisPlan &plan, const detail::PackedCpuPlan &packed,
    float *output, std::ptrdiff_t output_row_stride,
    std::int32_t vector_columns);
void forward_2d_rhs_avx2(
    const AxisPlan &horizontal,
    const detail::PackedCpuPlan &packed_horizontal,
    const AxisPlan &vertical,
    const detail::PackedCpuPlan &packed_vertical,
    const float *input, std::ptrdiff_t input_row_stride,
    float *output, std::ptrdiff_t output_row_stride,
    std::int32_t columns);
void backward_rhs_avx2(
    const AxisPlan &plan, const detail::PackedCpuPlan &packed,
    float *output, std::ptrdiff_t output_row_stride,
    std::int32_t columns);
void accumulate_2d_rhs_u8_avx2(
    const AxisPlan &horizontal,
    const detail::PackedCpuPlan &packed_horizontal,
    const detail::PackedCpuPlan &packed_vertical,
    const std::uint8_t *input, std::ptrdiff_t input_row_stride,
    const IntegerConversion &conversion,
    float *output, std::ptrdiff_t output_row_stride,
    std::int32_t first_destination_row,
    std::int32_t last_destination_row);
void accumulate_2d_rhs_u16_avx2(
    const AxisPlan &horizontal,
    const detail::PackedCpuPlan &packed_horizontal,
    const detail::PackedCpuPlan &packed_vertical,
    const std::uint16_t *input, std::ptrdiff_t input_row_stride,
    const IntegerConversion &conversion,
    float *output, std::ptrdiff_t output_row_stride,
    std::int32_t first_destination_row,
    std::int32_t last_destination_row);
void convert_rhs_to_u8_avx2(
    const float *input, std::ptrdiff_t input_row_stride,
    std::uint8_t *output, std::ptrdiff_t output_row_stride,
    std::int32_t rows, std::int32_t columns,
    const IntegerConversion &conversion);
void convert_rhs_to_u16_avx2(
    const float *input, std::ptrdiff_t input_row_stride,
    std::uint16_t *output, std::ptrdiff_t output_row_stride,
    std::int32_t rows, std::int32_t columns,
    const IntegerConversion &conversion);
void forward_2d_rhs_u8_avx2(
    const AxisPlan &horizontal,
    const detail::PackedCpuPlan &packed_horizontal,
    const AxisPlan &vertical,
    const detail::PackedCpuPlan &packed_vertical,
    const std::uint8_t *input, std::ptrdiff_t input_row_stride,
    const IntegerConversion &conversion,
    float *output, std::ptrdiff_t output_row_stride,
    std::int32_t columns);
void forward_2d_rhs_u16_avx2(
    const AxisPlan &horizontal,
    const detail::PackedCpuPlan &packed_horizontal,
    const AxisPlan &vertical,
    const detail::PackedCpuPlan &packed_vertical,
    const std::uint16_t *input, std::ptrdiff_t input_row_stride,
    const IntegerConversion &conversion,
    float *output, std::ptrdiff_t output_row_stride,
    std::int32_t columns);
void backward_rhs_to_u8_avx2(
    const AxisPlan &plan, const detail::PackedCpuPlan &packed,
    float *input, std::ptrdiff_t input_row_stride,
    std::uint8_t *output, std::ptrdiff_t output_row_stride,
    std::int32_t columns, const IntegerConversion &conversion);
void backward_rhs_to_u16_avx2(
    const AxisPlan &plan, const detail::PackedCpuPlan &packed,
    float *input, std::ptrdiff_t input_row_stride,
    std::uint16_t *output, std::ptrdiff_t output_row_stride,
    std::int32_t columns, const IntegerConversion &conversion);
#endif

#if defined(DSMVC_HAS_NEON_OBJECT)
void inverse_rows_f64_neon(
    const AxisPlan &plan,
    const float *input, std::ptrdiff_t input_row_stride,
    float *output, std::ptrdiff_t output_row_stride,
    std::int32_t row_count);
void inverse_rows_to_f64_neon(
    const AxisPlan &plan,
    const float *input, std::ptrdiff_t input_row_stride,
    double *output, std::ptrdiff_t output_row_stride,
    std::int32_t row_count);
void inverse_columns_f64_neon(
    const AxisPlan &plan,
    const float *input, std::ptrdiff_t input_row_stride,
    float *output, std::ptrdiff_t output_row_stride,
    std::int32_t column_count);
void inverse_columns_from_f64_neon(
    const AxisPlan &plan,
    const double *input, std::ptrdiff_t input_row_stride,
    float *output, std::ptrdiff_t output_row_stride,
    std::int32_t column_count);
void inverse_rows_neon(const AxisPlan &plan,
                       const detail::PackedCpuPlan &packed,
                       const float *input, std::ptrdiff_t input_row_stride,
                       float *output, std::ptrdiff_t output_row_stride,
                       std::int32_t row_count);
void inverse_columns_neon(const AxisPlan &plan,
                          const detail::PackedCpuPlan &packed,
                          const float *input, std::ptrdiff_t input_row_stride,
                          float *output, std::ptrdiff_t output_row_stride,
                          std::int32_t column_count);
void inverse_2d_u8_neon(
    const AxisPlan &horizontal,
    const detail::PackedCpuPlan &packed_horizontal,
    const AxisPlan &vertical,
    const detail::PackedCpuPlan &packed_vertical,
    const std::uint8_t *input, std::ptrdiff_t input_row_stride,
    std::uint8_t *output, std::ptrdiff_t output_row_stride,
    const IntegerConversion &conversion);
void inverse_2d_u16_neon(
    const AxisPlan &horizontal,
    const detail::PackedCpuPlan &packed_horizontal,
    const AxisPlan &vertical,
    const detail::PackedCpuPlan &packed_vertical,
    const std::uint16_t *input, std::ptrdiff_t input_row_stride,
    std::uint16_t *output, std::ptrdiff_t output_row_stride,
    const IntegerConversion &conversion);
void normalize_u8_neon(
    const std::uint8_t *input, std::ptrdiff_t input_row_stride,
    float *output, std::ptrdiff_t output_row_stride,
    std::int32_t rows, std::int32_t columns,
    const IntegerConversion &conversion);
void normalize_u16_neon(
    const std::uint16_t *input, std::ptrdiff_t input_row_stride,
    float *output, std::ptrdiff_t output_row_stride,
    std::int32_t rows, std::int32_t columns,
    const IntegerConversion &conversion);
void convert_to_u8_neon(
    const float *input, std::ptrdiff_t input_row_stride,
    std::uint8_t *output, std::ptrdiff_t output_row_stride,
    std::int32_t rows, std::int32_t columns,
    const IntegerConversion &conversion);
void convert_to_u16_neon(
    const float *input, std::ptrdiff_t input_row_stride,
    std::uint16_t *output, std::ptrdiff_t output_row_stride,
    std::int32_t rows, std::int32_t columns,
    const IntegerConversion &conversion);
#endif

struct CpuExecutor::Impl {
    explicit Impl(CpuPath path)
        : workers(shared_worker_pool(path)) {
        if (!workers) workers = std::make_shared<WorkerPool>(1U);
    }

    mutable std::mutex mutex;
    struct RegisteredPlan {
        std::shared_ptr<SharedPackedPlan> shared;
        bool lazy = false;
    };

    mutable std::vector<RegisteredPlan> plans;
    mutable std::atomic<bool> sealed{false};
    mutable PackingCounters packing;
    std::shared_ptr<WorkerPool> workers;

    [[nodiscard]] auto find(const AxisPlan &plan) const {
        return std::find_if(
            plans.begin(), plans.end(), [&plan](const auto &candidate) {
                return candidate.shared->identity() == &plan;
            });
    }

    [[nodiscard]] RegisteredPlan register_plan(
        const std::shared_ptr<const AxisPlan> &plan, bool lazy) const {
        const std::scoped_lock lock(mutex);
        if (sealed.load(std::memory_order_relaxed)) {
            throw std::logic_error("cannot add an axis to a sealed CPU plan cache");
        }
        const auto found = find(*plan);
        if (found != plans.end()) {
            if (!lazy) found->lazy = false;
            return *found;
        }
        RegisteredPlan registered{shared_packed_plan(plan), lazy};
        plans.push_back(registered);
        return registered;
    }

    [[nodiscard]] std::shared_ptr<const detail::PackedCpuPlan> get(
        const AxisPlan &plan) const {
        if (sealed.load(std::memory_order_acquire)) {
            const auto found = find(plan);
            if (found != plans.end()) {
                return found->shared->get(packing, found->lazy);
            }
        } else {
            const std::scoped_lock lock(mutex);
            const auto found = find(plan);
            if (found != plans.end()) {
                const auto registered = *found;
                return registered.shared->get(packing, registered.lazy);
            }
        }

        // Borrowed plans are intentionally invocation-local. The caller may
        // reassign or destroy one after this call, so its address is not a
        // stable cache key.
        auto owned = std::make_shared<const AxisPlan>(plan);
        auto borrowed = std::make_shared<SharedPackedPlan>(std::move(owned), &plan);
        return borrowed->get(packing, false);
    }
};

namespace detail {

PackedCpuPlan pack_cpu_plan(
    std::shared_ptr<const AxisPlan> axis, const AxisPlan *identity) {
    if (!axis || !axis->valid()) {
        throw std::invalid_argument("cannot pack an invalid CPU axis plan");
    }
    PackedCpuPlan packed;
    packed.axis = std::move(axis);
    packed.identity = identity ? identity : packed.axis.get();
    const auto &plan = *packed.axis;
    const auto n = plan.destination_size;
    packed.padded_source_size = checked_size_i32(
        checked_size_round_up(
            static_cast<std::size_t>(plan.source_size), 8U,
            "CPU packed source"),
        "CPU packed source");
    packed.padded_destination_size = checked_size_i32(
        checked_size_round_up(
            static_cast<std::size_t>(n), 8U,
            "CPU packed destination"),
        "CPU packed destination");
    const auto padded_n = static_cast<std::size_t>(packed.padded_destination_size);
    packed.weights_left.assign(padded_n, 0);
    packed.weights_right.assign(padded_n, 0);
    for (std::int32_t row = 0; row < n; ++row) {
        const auto begin = plan.transpose_offsets[static_cast<std::size_t>(row)];
        const auto end = plan.transpose_offsets[static_cast<std::size_t>(row) + 1U];
        if (begin == end) continue;
        const auto first = plan.transpose_indices[begin];
        const auto last = plan.transpose_indices[end - 1U] + 1;
        packed.weights_left[static_cast<std::size_t>(row)] = first;
        packed.weights_right[static_cast<std::size_t>(row)] = last;
        packed.weights_columns = std::max(packed.weights_columns, last - first);
    }
#if defined(DSMVC_HAS_NEON_OBJECT)
    const auto tail_block = plan.source_size >= 4 && (plan.source_size & 3)
        ? plan.source_size - 4 : plan.source_size;
#else
    const auto tail_block = plan.source_size >= 8 && (plan.source_size & 7)
        ? plan.source_size - 8 : plan.source_size;
#endif
    for (std::int32_t row = 0; row < n; ++row) {
        const auto begin = plan.transpose_offsets[static_cast<std::size_t>(row)];
        const auto end = plan.transpose_offsets[static_cast<std::size_t>(row) + 1U];
        std::int32_t previous_block = -1;
        std::int32_t block_count = 0;
        for (auto offset = begin; offset < end; ++offset) {
            const auto source = plan.transpose_indices[offset];
#if defined(DSMVC_HAS_NEON_OBJECT)
            const auto block = source >= tail_block
                ? tail_block : source & ~3;
#else
            const auto block = source >= tail_block
                ? tail_block : source & ~7;
#endif
            if (block != previous_block) {
                previous_block = block;
                ++block_count;
            }
        }
        packed.streaming_cache_blocks = std::max(
            packed.streaming_cache_blocks, block_count);
    }
    packed.weights.assign(
        checked_size_product(
            padded_n, static_cast<std::size_t>(packed.weights_columns),
            "CPU packed weights"),
        0.0F);
    for (std::int32_t row = 0; row < n; ++row) {
        const auto begin = plan.transpose_offsets[static_cast<std::size_t>(row)];
        const auto end = plan.transpose_offsets[static_cast<std::size_t>(row) + 1U];
        const auto left = packed.weights_left[static_cast<std::size_t>(row)];
        for (auto offset = begin; offset < end; ++offset) {
            const auto column = plan.transpose_indices[offset];
            packed.weights[static_cast<std::size_t>(row)
                               * static_cast<std::size_t>(packed.weights_columns)
                           + static_cast<std::size_t>(column - left)] =
                plan.transpose_weights[offset];
        }
    }

    packed.source_offsets.assign(
        checked_size_add(
            static_cast<std::size_t>(plan.source_size), 1U,
            "CPU packed source offsets"),
        0U);
    for (const auto source : plan.transpose_indices) {
        ++packed.source_offsets[static_cast<std::size_t>(source) + 1U];
    }
    for (std::size_t source = 1; source < packed.source_offsets.size(); ++source) {
        packed.source_offsets[source] += packed.source_offsets[source - 1U];
    }
    packed.source_destinations.resize(plan.transpose_indices.size());
    packed.source_weights.resize(plan.transpose_weights.size());
    auto source_cursors = packed.source_offsets;
    for (std::int32_t row = 0; row < n; ++row) {
        const auto begin = plan.transpose_offsets[static_cast<std::size_t>(row)];
        const auto end = plan.transpose_offsets[static_cast<std::size_t>(row) + 1U];
        for (auto offset = begin; offset < end; ++offset) {
            const auto source = plan.transpose_indices[offset];
            const auto destination = source_cursors[static_cast<std::size_t>(source)]++;
            packed.source_destinations[destination] = row;
            packed.source_weights[destination] = plan.transpose_weights[offset];
        }
    }

    const auto bands = static_cast<std::size_t>(plan.half_bandwidth);
    const auto packed_factors = checked_size_product(
        bands, padded_n, "CPU packed factors");
    packed.lower_ld.assign(packed_factors, 0.0F);
    packed.upper_l.assign(packed_factors, 0.0F);
    packed.inverse_diagonal.assign(padded_n, 0.0F);
    std::copy(plan.inverse_diagonal.begin(), plan.inverse_diagonal.end(),
              packed.inverse_diagonal.begin());
    for (std::size_t band = 0; band < bands; ++band) {
        const auto input_offset = band * static_cast<std::size_t>(n);
        const auto output_offset = band * padded_n;
        std::copy_n(plan.lower_ld.begin() + static_cast<std::ptrdiff_t>(input_offset),
                    n, packed.lower_ld.begin()
                           + static_cast<std::ptrdiff_t>(output_offset));
        std::copy_n(plan.upper_l.begin() + static_cast<std::ptrdiff_t>(input_offset),
                    n, packed.upper_l.begin()
                           + static_cast<std::ptrdiff_t>(output_offset));
    }
    return packed;
}

} // namespace detail

bool cpu_avx2_compiled() noexcept {
#if defined(DSMVC_HAS_AVX2_OBJECT)
    return true;
#else
    return false;
#endif
}

bool cpu_avx2_available() noexcept {
#if defined(DSMVC_HAS_AVX2_OBJECT) && defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
    int registers[4]{};
    __cpuid(registers, 1);
    constexpr int osxsave = 1 << 27;
    constexpr int avx = 1 << 28;
    constexpr int fma = 1 << 12;
    if ((registers[2] & (osxsave | avx | fma)) != (osxsave | avx | fma)) return false;
    const unsigned long long xcr0 = _xgetbv(0);
    if ((xcr0 & 0x6U) != 0x6U) return false;
    __cpuidex(registers, 7, 0);
    return (registers[1] & (1 << 5)) != 0;
#elif defined(DSMVC_HAS_AVX2_OBJECT) && (defined(__x86_64__) || defined(__i386__))
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
    return false;
#endif
}

bool cpu_neon_compiled() noexcept {
#if defined(DSMVC_HAS_NEON_OBJECT)
    return true;
#else
    return false;
#endif
}

bool cpu_neon_available() noexcept {
#if defined(DSMVC_HAS_NEON_OBJECT) && defined(_WIN32) && defined(_M_ARM64)
#if defined(PF_ARM_NEON_INSTRUCTIONS_AVAILABLE)
    return IsProcessorFeaturePresent(PF_ARM_NEON_INSTRUCTIONS_AVAILABLE) != 0;
#else
    return true;
#endif
#elif defined(DSMVC_HAS_NEON_OBJECT) && defined(__linux__) && defined(__aarch64__)
    return (getauxval(AT_HWCAP) & HWCAP_ASIMD) != 0U;
#elif defined(DSMVC_HAS_NEON_OBJECT) && defined(__APPLE__) && defined(__aarch64__)
    int available = 0;
    std::size_t size = sizeof(available);
    if (sysctlbyname("hw.optional.neon", &available, &size, nullptr, 0) == 0) {
        return available != 0;
    }
    return true;
#elif defined(DSMVC_HAS_NEON_OBJECT) \
    && (defined(__aarch64__) || defined(_M_ARM64))
    return true;
#else
    return false;
#endif
}

CpuExecutor::CpuExecutor(CpuPath requested) {
    switch (requested) {
    case CpuPath::automatic:
        if (cpu_avx2_available()) {
            path_ = CpuPath::avx2;
        } else if (cpu_neon_available()) {
            path_ = CpuPath::neon;
        } else {
            path_ = CpuPath::scalar;
        }
        break;
    case CpuPath::scalar:
        path_ = CpuPath::scalar;
        break;
    case CpuPath::avx2:
        if (!cpu_avx2_available()) {
            throw std::runtime_error(
                "the explicit AVX2 path requires compiled AVX2 and FMA support");
        }
        path_ = CpuPath::avx2;
        break;
    case CpuPath::neon:
        if (!cpu_neon_available()) {
            throw std::runtime_error(
                "the explicit NEON path requires compiled AArch64 NEON support");
        }
        path_ = CpuPath::neon;
        break;
    default:
        throw std::invalid_argument("invalid explicit CPU path");
    }
    impl_ = std::make_shared<Impl>(path_);
}

CpuExecutor::~CpuExecutor() = default;

CpuPath CpuExecutor::path() const noexcept { return path_; }

const char *CpuExecutor::name() const noexcept {
    switch (path_) {
    case CpuPath::scalar: return "scalar";
    case CpuPath::avx2: return "avx2-fma";
    case CpuPath::neon: return "neon-fma";
    default: return "invalid";
    }
}

CpuPlanPackingStats CpuExecutor::packing_stats() const noexcept {
    return impl_->packing.snapshot();
}

namespace {

void require_finite_matrix(
    const float *values, std::int32_t rows, std::int32_t columns,
    std::ptrdiff_t row_stride, const char *message) {
    for (std::int32_t row = 0; row < rows; ++row) {
        const auto *source = values
            + static_cast<std::ptrdiff_t>(row) * row_stride;
        for (std::int32_t column = 0; column < columns; ++column) {
            if (!std::isfinite(source[column])) {
                throw std::runtime_error(message);
            }
        }
    }
}

void inverse_rows_f64(
    const AxisPlan &plan,
    const float *input, std::ptrdiff_t input_row_stride,
    float *output, std::ptrdiff_t output_row_stride,
    std::int32_t row_count, WorkerPool &workers);

void inverse_columns_f64(
    const AxisPlan &plan,
    const float *input, std::ptrdiff_t input_row_stride,
    float *output, std::ptrdiff_t output_row_stride,
    std::int32_t column_count, WorkerPool &workers);

} // namespace

void CpuExecutor::prepare(std::shared_ptr<const AxisPlan> plan) const {
    if (!plan || !plan->valid()) {
        throw std::invalid_argument("cannot prepare an invalid CPU axis plan");
    }
    const auto registered = impl_->register_plan(plan, false);
    if (!plan->requires_float64()) {
        (void)registered.shared->get(impl_->packing, false);
    }
}

void CpuExecutor::defer(std::shared_ptr<const AxisPlan> plan) const {
    if (!plan || !plan->valid()) {
        throw std::invalid_argument("cannot defer an invalid CPU axis plan");
    }
    (void)impl_->register_plan(plan, true);
}

void CpuExecutor::seal() const {
    const std::scoped_lock lock(impl_->mutex);
    impl_->sealed.store(true, std::memory_order_release);
}

void CpuExecutor::inverse_rows(const AxisPlan &plan,
                               const float *input, std::ptrdiff_t input_row_stride,
                               float *output, std::ptrdiff_t output_row_stride,
                               std::int32_t row_count) const {
    if (!plan.valid() || !input || !output
        || input_row_stride < plan.source_size
        || output_row_stride < plan.destination_size || row_count < 0) {
        throw std::invalid_argument("invalid row executor arguments");
    }
    if (plan.requires_float64()) {
        require_finite_matrix(
            input, row_count, plan.source_size, input_row_stride,
            "CPU Float64 row input contains NaN or infinity");
        const auto execute = [&] {
#if defined(DSMVC_HAS_AVX2_OBJECT)
        if (path_ == CpuPath::avx2) {
            const auto complete_groups = static_cast<std::size_t>(row_count / 4);
            const auto task_count = std::min(
                impl_->workers->parallelism(), complete_groups);
            const auto enough_work = detail::checked_size_product(
                static_cast<std::size_t>(row_count),
                static_cast<std::size_t>(plan.destination_size),
                "CPU Float64 row work") >= 262144U;
            if (task_count != 0U && enough_work
                && impl_->workers->try_run(
                    task_count, [&](std::size_t task) {
                        const auto first_group =
                            complete_groups * task / task_count;
                        const auto last_group =
                            complete_groups * (task + 1U) / task_count;
                        const auto first_row = static_cast<std::int32_t>(
                            first_group * 4U);
                        const auto task_rows = static_cast<std::int32_t>(
                            (last_group - first_group) * 4U);
                        inverse_rows_f64_avx2(
                            plan,
                            input + static_cast<std::ptrdiff_t>(first_row)
                                * input_row_stride,
                            input_row_stride,
                            output + static_cast<std::ptrdiff_t>(first_row)
                                * output_row_stride,
                            output_row_stride, task_rows);
                    })) {
                const auto complete_rows = static_cast<std::int32_t>(
                    complete_groups * 4U);
                if (complete_rows != row_count) {
                    inverse_rows_f64(
                        plan,
                        input + static_cast<std::ptrdiff_t>(complete_rows)
                            * input_row_stride,
                        input_row_stride,
                        output + static_cast<std::ptrdiff_t>(complete_rows)
                            * output_row_stride,
                        output_row_stride, row_count - complete_rows,
                        *impl_->workers);
                }
                return;
            }
            inverse_rows_f64_avx2(
                plan, input, input_row_stride,
                output, output_row_stride, row_count);
            return;
        }
#elif defined(DSMVC_HAS_NEON_OBJECT)
        if (path_ == CpuPath::neon) {
            const auto complete_groups = static_cast<std::size_t>(row_count / 4);
            const auto task_count = std::min(
                impl_->workers->parallelism(), complete_groups);
            const auto enough_work = detail::checked_size_product(
                static_cast<std::size_t>(row_count),
                static_cast<std::size_t>(plan.destination_size),
                "CPU Float64 row work") >= 262144U;
            if (task_count != 0U && enough_work
                && impl_->workers->try_run(
                    task_count, [&](std::size_t task) {
                        const auto first_group =
                            complete_groups * task / task_count;
                        const auto last_group =
                            complete_groups * (task + 1U) / task_count;
                        const auto first_row = static_cast<std::int32_t>(
                            first_group * 4U);
                        const auto task_rows = static_cast<std::int32_t>(
                            (last_group - first_group) * 4U);
                        inverse_rows_f64_neon(
                            plan,
                            input + static_cast<std::ptrdiff_t>(first_row)
                                * input_row_stride,
                            input_row_stride,
                            output + static_cast<std::ptrdiff_t>(first_row)
                                * output_row_stride,
                            output_row_stride, task_rows);
                    })) {
                const auto complete_rows = static_cast<std::int32_t>(
                    complete_groups * 4U);
                if (complete_rows != row_count) {
                    inverse_rows_f64_neon(
                        plan,
                        input + static_cast<std::ptrdiff_t>(complete_rows)
                            * input_row_stride,
                        input_row_stride,
                        output + static_cast<std::ptrdiff_t>(complete_rows)
                            * output_row_stride,
                        output_row_stride, row_count - complete_rows);
                }
                return;
            }
            inverse_rows_f64_neon(
                plan, input, input_row_stride,
                output, output_row_stride, row_count);
            return;
        }
#endif
        inverse_rows_f64(
            plan, input, input_row_stride, output, output_row_stride,
            row_count, *impl_->workers);
        };
        execute();
        require_finite_matrix(
            output, row_count, plan.destination_size, output_row_stride,
            "CPU Float64 row execution produced NaN or infinity");
        return;
    }
#if defined(DSMVC_HAS_AVX2_OBJECT)
    if (path_ == CpuPath::avx2) {
        const auto packed = impl_->get(plan);
        const auto complete_groups = static_cast<std::size_t>(row_count / 8);
        const auto task_count = std::min(
            impl_->workers->parallelism(), complete_groups);
        const auto enough_work = detail::checked_size_product(
            static_cast<std::size_t>(row_count),
            static_cast<std::size_t>(plan.destination_size),
            "CPU row work") >= 262144U;
        if (enough_work && impl_->workers->try_run(
                task_count, [&](std::size_t task) {
                    const auto first_group = complete_groups * task / task_count;
                    const auto last_group = complete_groups * (task + 1U) / task_count;
                    const auto first_row = static_cast<std::int32_t>(first_group * 8U);
                    const auto task_rows = static_cast<std::int32_t>(
                        (last_group - first_group) * 8U);
                    inverse_rows_avx2(
                        plan, *packed,
                        input + static_cast<std::ptrdiff_t>(first_row) * input_row_stride,
                        input_row_stride,
                        output + static_cast<std::ptrdiff_t>(first_row) * output_row_stride,
                        output_row_stride, task_rows);
                })) {
            if ((row_count & 7) != 0) {
                const auto first_row = row_count - 8;
                inverse_rows_avx2(
                    plan, *packed,
                    input + static_cast<std::ptrdiff_t>(first_row) * input_row_stride,
                    input_row_stride,
                    output + static_cast<std::ptrdiff_t>(first_row) * output_row_stride,
                    output_row_stride, 8);
            }
            return;
        }
        inverse_rows_avx2(plan, *packed, input, input_row_stride, output,
                          output_row_stride, row_count);
        return;
    }
#elif defined(DSMVC_HAS_NEON_OBJECT)
    if (path_ == CpuPath::neon) {
        const auto packed = impl_->get(plan);
        const auto complete_groups = static_cast<std::size_t>(row_count / 4);
        const auto task_count = std::min(
            impl_->workers->parallelism(), complete_groups);
        const auto enough_work = detail::checked_size_product(
            static_cast<std::size_t>(row_count),
            static_cast<std::size_t>(plan.destination_size),
            "CPU row work") >= 262144U;
        if (enough_work && impl_->workers->try_run(
                task_count, [&](std::size_t task) {
                    const auto first_group = complete_groups * task / task_count;
                    const auto last_group = complete_groups * (task + 1U) / task_count;
                    const auto first_row = static_cast<std::int32_t>(first_group * 4U);
                    const auto task_rows = static_cast<std::int32_t>(
                        (last_group - first_group) * 4U);
                    inverse_rows_neon(
                        plan, *packed,
                        input + static_cast<std::ptrdiff_t>(first_row) * input_row_stride,
                        input_row_stride,
                        output + static_cast<std::ptrdiff_t>(first_row) * output_row_stride,
                        output_row_stride, task_rows);
                })) {
            if ((row_count & 3) != 0) {
                const auto first_row = row_count - 4;
                inverse_rows_neon(
                    plan, *packed,
                    input + static_cast<std::ptrdiff_t>(first_row) * input_row_stride,
                    input_row_stride,
                    output + static_cast<std::ptrdiff_t>(first_row) * output_row_stride,
                    output_row_stride, 4);
            }
            return;
        }
        inverse_rows_neon(plan, *packed, input, input_row_stride, output,
                          output_row_stride, row_count);
        return;
    }
#endif
    for (std::int32_t row = 0; row < row_count; ++row) {
        dsmvc::inverse_axis_f32(
            plan, input + static_cast<std::ptrdiff_t>(row) * input_row_stride, 1,
            output + static_cast<std::ptrdiff_t>(row) * output_row_stride, 1);
    }
}

void CpuExecutor::inverse_columns(const AxisPlan &plan,
                                  const float *input, std::ptrdiff_t input_row_stride,
                                  float *output, std::ptrdiff_t output_row_stride,
                                  std::int32_t column_count) const {
    if (!plan.valid() || !input || !output || column_count < 0
        || input_row_stride < column_count
        || output_row_stride < column_count) {
        throw std::invalid_argument("invalid column executor arguments");
    }
    if (plan.requires_float64()) {
        require_finite_matrix(
            input, plan.source_size, column_count, input_row_stride,
            "CPU Float64 column input contains NaN or infinity");
        const auto execute = [&] {
#if defined(DSMVC_HAS_AVX2_OBJECT)
        if (path_ == CpuPath::avx2) {
            const auto complete_groups = static_cast<std::size_t>(
                column_count / 4);
            const auto task_count = std::min(
                impl_->workers->parallelism(), complete_groups);
            const auto enough_work = detail::checked_size_product(
                static_cast<std::size_t>(column_count),
                static_cast<std::size_t>(plan.destination_size),
                "CPU Float64 column work") >= 262144U;
            if (task_count != 0U && enough_work
                && impl_->workers->try_run(
                    task_count, [&](std::size_t task) {
                        const auto first_group =
                            complete_groups * task / task_count;
                        const auto last_group =
                            complete_groups * (task + 1U) / task_count;
                        const auto first_column = static_cast<std::int32_t>(
                            first_group * 4U);
                        const auto task_columns = static_cast<std::int32_t>(
                            (last_group - first_group) * 4U);
                        inverse_columns_f64_avx2(
                            plan, input + first_column, input_row_stride,
                            output + first_column, output_row_stride,
                            task_columns);
                    })) {
                const auto complete_columns = static_cast<std::int32_t>(
                    complete_groups * 4U);
                if (complete_columns != column_count) {
                    inverse_columns_f64(
                        plan, input + complete_columns, input_row_stride,
                        output + complete_columns, output_row_stride,
                        column_count - complete_columns, *impl_->workers);
                }
                return;
            }
            inverse_columns_f64_avx2(
                plan, input, input_row_stride,
                output, output_row_stride, column_count);
            return;
        }
#elif defined(DSMVC_HAS_NEON_OBJECT)
        if (path_ == CpuPath::neon) {
            const auto complete_groups = static_cast<std::size_t>(
                column_count / 4);
            const auto task_count = std::min(
                impl_->workers->parallelism(), complete_groups);
            const auto enough_work = detail::checked_size_product(
                static_cast<std::size_t>(column_count),
                static_cast<std::size_t>(plan.destination_size),
                "CPU Float64 column work") >= 262144U;
            if (task_count != 0U && enough_work
                && impl_->workers->try_run(
                    task_count, [&](std::size_t task) {
                        const auto first_group =
                            complete_groups * task / task_count;
                        const auto last_group =
                            complete_groups * (task + 1U) / task_count;
                        const auto first_column = static_cast<std::int32_t>(
                            first_group * 4U);
                        const auto task_columns = static_cast<std::int32_t>(
                            (last_group - first_group) * 4U);
                        inverse_columns_f64_neon(
                            plan, input + first_column, input_row_stride,
                            output + first_column, output_row_stride,
                            task_columns);
                    })) {
                const auto complete_columns = static_cast<std::int32_t>(
                    complete_groups * 4U);
                if (complete_columns != column_count) {
                    inverse_columns_f64_neon(
                        plan, input + complete_columns, input_row_stride,
                        output + complete_columns, output_row_stride,
                        column_count - complete_columns);
                }
                return;
            }
            inverse_columns_f64_neon(
                plan, input, input_row_stride,
                output, output_row_stride, column_count);
            return;
        }
#endif
        inverse_columns_f64(
            plan, input, input_row_stride, output, output_row_stride,
            column_count, *impl_->workers);
        };
        execute();
        require_finite_matrix(
            output, plan.destination_size, column_count, output_row_stride,
            "CPU Float64 column execution produced NaN or infinity");
        return;
    }
#if defined(DSMVC_HAS_AVX2_OBJECT)
    if (path_ == CpuPath::avx2) {
        const auto packed = impl_->get(plan);
        const auto vector_columns = column_count & ~7;
        const auto column_groups = static_cast<std::size_t>(vector_columns / 8);
        const auto task_count = std::min(
            impl_->workers->parallelism(), column_groups);
        const auto enough_work = detail::checked_size_product(
            static_cast<std::size_t>(column_count),
            static_cast<std::size_t>(plan.destination_size),
            "CPU column work") >= 262144U;
        if (enough_work && impl_->workers->try_run(
                task_count, [&](std::size_t task) {
                    const auto first_group = column_groups * task / task_count;
                    const auto last_group = column_groups * (task + 1U) / task_count;
                    const auto first_column = static_cast<std::int32_t>(first_group * 8U);
                    const auto task_columns = static_cast<std::int32_t>(
                        (last_group - first_group) * 8U);
                    inverse_columns_avx2(
                        plan, *packed, input + first_column, input_row_stride,
                        output + first_column, output_row_stride, task_columns);
                })) {
            for (std::int32_t column = vector_columns;
                 column < column_count; ++column) {
                dsmvc::inverse_axis_f32(
                    plan, input + column, input_row_stride,
                    output + column, output_row_stride);
            }
            return;
        }
        inverse_columns_avx2(plan, *packed, input, input_row_stride, output,
                             output_row_stride, column_count);
        return;
    }
#elif defined(DSMVC_HAS_NEON_OBJECT)
    if (path_ == CpuPath::neon) {
        const auto packed = impl_->get(plan);
        const auto vector_columns = column_count & ~3;
        const auto column_groups = static_cast<std::size_t>(vector_columns / 4);
        const auto task_count = std::min(
            impl_->workers->parallelism(), column_groups);
        const auto enough_work = detail::checked_size_product(
            static_cast<std::size_t>(column_count),
            static_cast<std::size_t>(plan.destination_size),
            "CPU column work") >= 262144U;
        if (enough_work && impl_->workers->try_run(
                task_count, [&](std::size_t task) {
                    const auto first_group = column_groups * task / task_count;
                    const auto last_group = column_groups * (task + 1U) / task_count;
                    const auto first_column = static_cast<std::int32_t>(first_group * 4U);
                    const auto task_columns = static_cast<std::int32_t>(
                        (last_group - first_group) * 4U);
                    inverse_columns_neon(
                        plan, *packed, input + first_column, input_row_stride,
                        output + first_column, output_row_stride, task_columns);
                })) {
            for (std::int32_t column = vector_columns;
                 column < column_count; ++column) {
                dsmvc::inverse_axis_f32(
                    plan, input + column, input_row_stride,
                    output + column, output_row_stride);
            }
            return;
        }
        inverse_columns_neon(plan, *packed, input, input_row_stride, output,
                             output_row_stride, column_count);
        return;
    }
#endif
    for (std::int32_t column = 0; column < column_count; ++column) {
        dsmvc::inverse_axis_f32(plan, input + column, input_row_stride,
                               output + column, output_row_stride);
    }
}

namespace {

template <class Function>
void run_parallel_ranges(
    WorkerPool &workers, std::int32_t count, std::size_t work,
    Function &&function) {
    const auto tasks = std::min(
        workers.parallelism(), static_cast<std::size_t>(count));
    if (work >= 262144U && workers.try_run(tasks, [&](std::size_t task) {
            const auto first = static_cast<std::int32_t>(
                static_cast<std::size_t>(count) * task / tasks);
            const auto last = static_cast<std::int32_t>(
                static_cast<std::size_t>(count) * (task + 1U) / tasks);
            function(first, last);
        })) {
        return;
    }
    function(0, count);
}

void inverse_rows_f64(
    const AxisPlan &plan,
    const float *input, std::ptrdiff_t input_row_stride,
    float *output, std::ptrdiff_t output_row_stride,
    std::int32_t row_count, WorkerPool &workers) {
    const auto work = detail::checked_size_product(
        static_cast<std::size_t>(row_count),
        static_cast<std::size_t>(plan.destination_size),
        "CPU Float64 row work");
    run_parallel_ranges(workers, row_count, work,
        [&](std::int32_t first, std::int32_t last) {
            std::vector<double> source(
                static_cast<std::size_t>(plan.source_size));
            std::vector<double> destination(
                static_cast<std::size_t>(plan.destination_size));
            for (std::int32_t row = first; row < last; ++row) {
                const auto *source_row = input
                    + static_cast<std::ptrdiff_t>(row) * input_row_stride;
                for (std::int32_t column = 0;
                     column < plan.source_size; ++column) {
                    source[static_cast<std::size_t>(column)] =
                        static_cast<double>(source_row[column]);
                }
                detail::inverse_axis_f64(
                    plan, source.data(), 1, destination.data(), 1);
                auto *destination_row = output
                    + static_cast<std::ptrdiff_t>(row) * output_row_stride;
                for (std::int32_t column = 0;
                     column < plan.destination_size; ++column) {
                    destination_row[column] = static_cast<float>(
                        destination[static_cast<std::size_t>(column)]);
                }
            }
        });
}

void inverse_columns_f64(
    const AxisPlan &plan,
    const float *input, std::ptrdiff_t input_row_stride,
    float *output, std::ptrdiff_t output_row_stride,
    std::int32_t column_count, WorkerPool &workers) {
    const auto work = detail::checked_size_product(
        static_cast<std::size_t>(column_count),
        static_cast<std::size_t>(plan.destination_size),
        "CPU Float64 column work");
    run_parallel_ranges(workers, column_count, work,
        [&](std::int32_t first, std::int32_t last) {
            std::vector<double> source(
                static_cast<std::size_t>(plan.source_size));
            std::vector<double> destination(
                static_cast<std::size_t>(plan.destination_size));
            for (std::int32_t column = first; column < last; ++column) {
                for (std::int32_t row = 0; row < plan.source_size; ++row) {
                    source[static_cast<std::size_t>(row)] = static_cast<double>(
                        input[static_cast<std::ptrdiff_t>(row)
                            * input_row_stride + column]);
                }
                detail::inverse_axis_f64(
                    plan, source.data(), 1, destination.data(), 1);
                for (std::int32_t row = 0;
                     row < plan.destination_size; ++row) {
                    output[static_cast<std::ptrdiff_t>(row)
                               * output_row_stride + column] =
                        static_cast<float>(
                            destination[static_cast<std::size_t>(row)]);
                }
            }
        });
}

#if defined(DSMVC_HAS_AVX2_OBJECT)
void inverse_2d_f64_avx2(
    const AxisPlan &horizontal, const AxisPlan &vertical,
    const float *input, std::ptrdiff_t input_row_stride,
    float *output, std::ptrdiff_t output_row_stride,
    WorkerPool &workers) {
    const auto intermediate_stride = horizontal.destination_size;
    std::vector<double> intermediate(detail::checked_size_product(
        static_cast<std::size_t>(vertical.source_size),
        static_cast<std::size_t>(intermediate_stride),
        "CPU Float64 intermediate"));

    const auto horizontal_work = detail::checked_size_product(
        static_cast<std::size_t>(vertical.source_size),
        static_cast<std::size_t>(horizontal.destination_size),
        "CPU Float64 horizontal work");
    run_parallel_ranges(workers, vertical.source_size, horizontal_work,
        [&](std::int32_t first, std::int32_t last) {
            inverse_rows_to_f64_avx2(
                horizontal,
                input + static_cast<std::ptrdiff_t>(first) * input_row_stride,
                input_row_stride,
                intermediate.data()
                    + static_cast<std::ptrdiff_t>(first) * intermediate_stride,
                intermediate_stride, last - first);
        });

    const auto vertical_work = detail::checked_size_product(
        static_cast<std::size_t>(horizontal.destination_size),
        static_cast<std::size_t>(vertical.destination_size),
        "CPU Float64 vertical work");
    run_parallel_ranges(workers, horizontal.destination_size, vertical_work,
        [&](std::int32_t first, std::int32_t last) {
            inverse_columns_from_f64_avx2(
                vertical, intermediate.data() + first,
                intermediate_stride, output + first,
                output_row_stride, last - first);
        });
}
#endif

template <class Sample>
void inverse_2d_f64(
    const AxisPlan &horizontal, const AxisPlan &vertical,
    const Sample *input, std::ptrdiff_t input_row_stride,
    Sample *output, std::ptrdiff_t output_row_stride,
    const IntegerConversion *conversion, WorkerPool &workers,
    bool use_neon) {
    const auto intermediate_stride = horizontal.destination_size;
    std::vector<double> intermediate(detail::checked_size_product(
        static_cast<std::size_t>(vertical.source_size),
        static_cast<std::size_t>(intermediate_stride),
        "CPU Float64 intermediate"));

    const auto horizontal_work = detail::checked_size_product(
        static_cast<std::size_t>(vertical.source_size),
        static_cast<std::size_t>(horizontal.destination_size),
        "CPU Float64 horizontal work");
    run_parallel_ranges(workers, vertical.source_size, horizontal_work,
        [&](std::int32_t first, std::int32_t last) {
#if defined(DSMVC_HAS_NEON_OBJECT)
            if constexpr (std::is_same_v<Sample, float>) {
                if (use_neon) {
                    inverse_rows_to_f64_neon(
                        horizontal,
                        input + static_cast<std::ptrdiff_t>(first)
                            * input_row_stride,
                        input_row_stride,
                        intermediate.data()
                            + static_cast<std::ptrdiff_t>(first)
                                * intermediate_stride,
                        intermediate_stride, last - first);
                    return;
                }
            }
#else
            (void)use_neon;
#endif
            std::vector<double> source(
                static_cast<std::size_t>(horizontal.source_size));
            for (std::int32_t row = first; row < last; ++row) {
                const auto *source_row = input
                    + static_cast<std::ptrdiff_t>(row) * input_row_stride;
                for (std::int32_t column = 0;
                     column < horizontal.source_size; ++column) {
                    if constexpr (std::is_same_v<Sample, float>) {
                        source[static_cast<std::size_t>(column)] =
                            static_cast<double>(source_row[column]);
                    } else {
                        source[static_cast<std::size_t>(column)] =
                            (static_cast<double>(source_row[column])
                             - static_cast<double>(conversion->input_offset))
                            * static_cast<double>(conversion->input_scale);
                    }
                }
                detail::inverse_axis_f64(
                    horizontal, source.data(), 1,
                    intermediate.data()
                        + static_cast<std::ptrdiff_t>(row) * intermediate_stride,
                    1);
            }
        });

    const auto vertical_work = detail::checked_size_product(
        static_cast<std::size_t>(horizontal.destination_size),
        static_cast<std::size_t>(vertical.destination_size),
        "CPU Float64 vertical work");
    std::atomic<bool> nonfinite{false};
    run_parallel_ranges(workers, horizontal.destination_size, vertical_work,
        [&](std::int32_t first, std::int32_t last) {
#if defined(DSMVC_HAS_NEON_OBJECT)
            if constexpr (std::is_same_v<Sample, float>) {
                if (use_neon) {
                    inverse_columns_from_f64_neon(
                        vertical, intermediate.data() + first,
                        intermediate_stride, output + first,
                        output_row_stride, last - first);
                    return;
                }
            }
#endif
            std::vector<double> destination(
                static_cast<std::size_t>(vertical.destination_size));
            for (std::int32_t column = first; column < last; ++column) {
                detail::inverse_axis_f64(
                    vertical, intermediate.data() + column,
                    intermediate_stride, destination.data(), 1);
                for (std::int32_t row = 0;
                     row < vertical.destination_size; ++row) {
                    const double value =
                        destination[static_cast<std::size_t>(row)];
                    if constexpr (std::is_same_v<Sample, float>) {
                        const float converted = static_cast<float>(value);
                        if (!std::isfinite(value)
                            || !std::isfinite(converted)) {
                            nonfinite.store(true, std::memory_order_relaxed);
                        }
                        output[static_cast<std::ptrdiff_t>(row)
                                   * output_row_stride + column] = converted;
                    } else {
                        const double scaled =
                            value * static_cast<double>(conversion->output_scale)
                            + static_cast<double>(conversion->output_offset);
                        if (!std::isfinite(value) || !std::isfinite(scaled)) {
                            nonfinite.store(true, std::memory_order_relaxed);
                            output[static_cast<std::ptrdiff_t>(row)
                                       * output_row_stride + column] = Sample{};
                            continue;
                        }
                        const double converted = std::clamp(
                            scaled,
                            0.0,
                            static_cast<double>(conversion->output_maximum));
                        output[static_cast<std::ptrdiff_t>(row)
                                   * output_row_stride + column] =
                            static_cast<Sample>(std::nearbyint(converted));
                    }
                }
            }
        });
    if (nonfinite.load(std::memory_order_relaxed)) {
        throw std::runtime_error(
            "CPU Float64 2D execution produced NaN or infinity");
    }
}

void solve_rhs_scalar(const AxisPlan &plan, float *values,
                      std::ptrdiff_t stride) noexcept {
    const auto n = plan.destination_size;
    const auto factor_stride = static_cast<std::size_t>(n);
    for (std::int32_t i = 0; i < n; ++i) {
        float value = values[static_cast<std::ptrdiff_t>(i) * stride];
        const auto available = std::min(plan.half_bandwidth, i);
        for (std::int32_t distance = available; distance >= 1; --distance) {
            value -= plan.lower_ld[
                static_cast<std::size_t>(distance - 1) * factor_stride
                + static_cast<std::size_t>(i)]
                * values[static_cast<std::ptrdiff_t>(i - distance) * stride];
        }
        values[static_cast<std::ptrdiff_t>(i) * stride] = value
            * plan.inverse_diagonal[static_cast<std::size_t>(i)];
    }
    for (std::int32_t i = n - 2; i >= 0; --i) {
        float sum = 0.0F;
        const auto available = std::min(plan.half_bandwidth, n - i - 1);
        if (plan.half_bandwidth == 3) {
            for (std::int32_t distance = 1; distance <= available; ++distance) {
                sum += plan.upper_l[
                    static_cast<std::size_t>(distance - 1) * factor_stride
                    + static_cast<std::size_t>(i)]
                    * values[static_cast<std::ptrdiff_t>(i + distance) * stride];
            }
        } else {
            for (std::int32_t distance = available; distance >= 1; --distance) {
                sum += plan.upper_l[
                    static_cast<std::size_t>(distance - 1) * factor_stride
                    + static_cast<std::size_t>(i)]
                    * values[static_cast<std::ptrdiff_t>(i + distance) * stride];
            }
        }
        values[static_cast<std::ptrdiff_t>(i) * stride] -= sum;
    }
}

void backward_rhs_scalar(const AxisPlan &plan, float *values,
                         std::ptrdiff_t stride) noexcept {
    const auto n = plan.destination_size;
    const auto factor_stride = static_cast<std::size_t>(n);
    for (std::int32_t i = n - 2; i >= 0; --i) {
        float sum = 0.0F;
        const auto available = std::min(plan.half_bandwidth, n - i - 1);
        if (plan.half_bandwidth == 3) {
            for (std::int32_t distance = 1; distance <= available; ++distance) {
                sum += plan.upper_l[
                    static_cast<std::size_t>(distance - 1) * factor_stride
                    + static_cast<std::size_t>(i)]
                    * values[static_cast<std::ptrdiff_t>(i + distance) * stride];
            }
        } else {
            for (std::int32_t distance = available; distance >= 1; --distance) {
                sum += plan.upper_l[
                    static_cast<std::size_t>(distance - 1) * factor_stride
                    + static_cast<std::size_t>(i)]
                    * values[static_cast<std::ptrdiff_t>(i + distance) * stride];
            }
        }
        values[static_cast<std::ptrdiff_t>(i) * stride] -= sum;
    }
}

template <class Sample>
void forward_2d_rhs_scalar(
    const AxisPlan &horizontal,
    const detail::PackedCpuPlan &packed_vertical,
    const Sample *input, std::ptrdiff_t input_row_stride,
    const IntegerConversion *conversion,
    float *output, std::ptrdiff_t output_row_stride) {
    const auto columns = horizontal.destination_size;
    const auto cache_rows = static_cast<std::size_t>(
        std::max(packed_vertical.weights_columns, 1));

    thread_local std::vector<float> horizontal_cache;
    thread_local std::vector<float> source_scratch;
    thread_local std::vector<std::int32_t> cache_sources;
    thread_local std::vector<std::uint64_t> cache_ages;
    thread_local std::vector<const float *> source_rows;
    horizontal_cache.resize(detail::checked_size_product(
        cache_rows, static_cast<std::size_t>(columns),
        "CPU horizontal row cache"));
    source_scratch.resize(static_cast<std::size_t>(horizontal.source_size));
    cache_sources.assign(cache_rows, -1);
    cache_ages.assign(cache_rows, 0U);
    std::uint64_t age = 0U;

    const auto get_source_row = [&](std::int32_t source) -> const float * {
        std::size_t slot = cache_rows;
        for (std::size_t candidate = 0; candidate < cache_rows; ++candidate) {
            if (cache_sources[candidate] == source) {
                slot = candidate;
                break;
            }
        }
        if (slot == cache_rows) {
            slot = 0U;
            for (std::size_t candidate = 0; candidate < cache_rows; ++candidate) {
                if (cache_sources[candidate] < 0) {
                    slot = candidate;
                    break;
                }
                if (cache_ages[candidate] < cache_ages[slot]) slot = candidate;
            }
            const float *horizontal_input = nullptr;
            if constexpr (std::is_same_v<Sample, float>) {
                horizontal_input = input
                    + static_cast<std::ptrdiff_t>(source) * input_row_stride;
            } else {
                const auto *integer_row = input
                    + static_cast<std::ptrdiff_t>(source) * input_row_stride;
                for (std::int32_t column = 0;
                     column < horizontal.source_size; ++column) {
                    source_scratch[static_cast<std::size_t>(column)] =
                        (static_cast<float>(integer_row[column])
                         - conversion->input_offset)
                        * conversion->input_scale;
                }
                horizontal_input = source_scratch.data();
            }
            auto *horizontal_output = horizontal_cache.data()
                + slot * static_cast<std::size_t>(columns);
            inverse_axis_f32(
                horizontal, horizontal_input, 1, horizontal_output, 1);
            cache_sources[slot] = source;
        }
        cache_ages[slot] = ++age;
        return horizontal_cache.data()
            + slot * static_cast<std::size_t>(columns);
    };

    const auto &vertical = *packed_vertical.axis;
    const auto factor_stride = static_cast<std::size_t>(
        vertical.destination_size);
    for (std::int32_t row = 0; row < vertical.destination_size; ++row) {
        const auto begin = vertical.transpose_offsets[
            static_cast<std::size_t>(row)];
        const auto end = vertical.transpose_offsets[
            static_cast<std::size_t>(row) + 1U];
        source_rows.resize(static_cast<std::size_t>(end - begin));
        for (auto offset = begin; offset < end; ++offset) {
            source_rows[static_cast<std::size_t>(offset - begin)] =
                get_source_row(vertical.transpose_indices[offset]);
        }
        auto *destination = output
            + static_cast<std::ptrdiff_t>(row) * output_row_stride;
        for (std::int32_t column = 0; column < columns; ++column) {
            float value = 0.0F;
            for (auto offset = begin; offset < end; ++offset) {
                value += vertical.transpose_weights[offset]
                    * source_rows[static_cast<std::size_t>(offset - begin)][column];
            }
            const auto available = std::min(vertical.half_bandwidth, row);
            for (std::int32_t distance = available;
                 distance >= 1; --distance) {
                value -= vertical.lower_ld[
                    static_cast<std::size_t>(distance - 1) * factor_stride
                    + static_cast<std::size_t>(row)]
                    * output[static_cast<std::ptrdiff_t>(row - distance)
                        * output_row_stride + column];
            }
            destination[column] = value * vertical.inverse_diagonal[
                static_cast<std::size_t>(row)];
        }
    }
}

template <class Sample>
void backward_rhs_to_integer_scalar(
    const AxisPlan &plan, float *input, std::ptrdiff_t input_row_stride,
    Sample *output, std::ptrdiff_t output_row_stride,
    std::int32_t columns, const IntegerConversion &conversion) noexcept {
    const auto convert = [&](float value) noexcept {
        const auto scaled = std::clamp(
            value * conversion.output_scale + conversion.output_offset,
            0.0F, static_cast<float>(conversion.output_maximum));
        return static_cast<Sample>(std::nearbyint(scaled));
    };
    const auto n = plan.destination_size;
    const auto factor_stride = static_cast<std::size_t>(n);
    for (std::int32_t column = 0; column < columns; ++column) {
        output[static_cast<std::ptrdiff_t>(n - 1) * output_row_stride + column] =
            convert(input[static_cast<std::ptrdiff_t>(n - 1)
                * input_row_stride + column]);
        for (std::int32_t row = n - 2; row >= 0; --row) {
            auto *value_ptr = input
                + static_cast<std::ptrdiff_t>(row) * input_row_stride + column;
            float sum = 0.0F;
            const auto available = std::min(plan.half_bandwidth, n - row - 1);
            if (plan.half_bandwidth == 3) {
                for (std::int32_t distance = 1;
                     distance <= available; ++distance) {
                    sum += plan.upper_l[
                        static_cast<std::size_t>(distance - 1) * factor_stride
                        + static_cast<std::size_t>(row)]
                        * input[static_cast<std::ptrdiff_t>(row + distance)
                            * input_row_stride + column];
                }
            } else {
                for (std::int32_t distance = available;
                     distance >= 1; --distance) {
                    sum += plan.upper_l[
                        static_cast<std::size_t>(distance - 1) * factor_stride
                        + static_cast<std::size_t>(row)]
                        * input[static_cast<std::ptrdiff_t>(row + distance)
                            * input_row_stride + column];
                }
            }
            *value_ptr -= sum;
            output[static_cast<std::ptrdiff_t>(row) * output_row_stride + column]
                = convert(*value_ptr);
        }
    }
}

template <class Sample>
void accumulate_2d_integer_rhs_scalar(
    const AxisPlan &horizontal,
    const detail::PackedCpuPlan &packed_vertical,
    const Sample *input, std::ptrdiff_t input_row_stride,
    const IntegerConversion &conversion,
    float *output, std::ptrdiff_t output_row_stride) {
    const auto source_columns = horizontal.source_size;
    const auto destination_columns = horizontal.destination_size;
    const auto destination_rows = packed_vertical.axis->destination_size;
    for (std::int32_t row = 0; row < destination_rows; ++row) {
        std::fill_n(output + static_cast<std::ptrdiff_t>(row) * output_row_stride,
                    destination_columns, 0.0F);
    }

    std::vector<float> source_row(static_cast<std::size_t>(source_columns));
    std::vector<float> horizontal_row(
        static_cast<std::size_t>(destination_columns));
    for (std::int32_t source_row_index = 0;
         source_row_index < packed_vertical.axis->source_size;
         ++source_row_index) {
        const auto *integer_row = input
            + static_cast<std::ptrdiff_t>(source_row_index) * input_row_stride;
        for (std::int32_t column = 0; column < source_columns; ++column) {
            source_row[static_cast<std::size_t>(column)] =
                (static_cast<float>(integer_row[column]) - conversion.input_offset)
                * conversion.input_scale;
        }
        inverse_axis_f32(
            horizontal, source_row.data(), 1, horizontal_row.data(), 1);
        const auto begin = packed_vertical.source_offsets[
            static_cast<std::size_t>(source_row_index)];
        const auto end = packed_vertical.source_offsets[
            static_cast<std::size_t>(source_row_index) + 1U];
        for (auto offset = begin; offset < end; ++offset) {
            auto *rhs = output
                + static_cast<std::ptrdiff_t>(
                    packed_vertical.source_destinations[offset])
                    * output_row_stride;
            const auto weight = packed_vertical.source_weights[offset];
            for (std::int32_t column = 0;
                 column < destination_columns; ++column) {
                rhs[column] += weight
                    * horizontal_row[static_cast<std::size_t>(column)];
            }
        }
    }
}

template <class Sample>
void convert_rhs_to_integer_scalar(
    const float *input, std::ptrdiff_t input_row_stride,
    Sample *output, std::ptrdiff_t output_row_stride,
    std::int32_t rows, std::int32_t columns,
    const IntegerConversion &conversion) noexcept {
    for (std::int32_t row = 0; row < rows; ++row) {
        const auto *source = input
            + static_cast<std::ptrdiff_t>(row) * input_row_stride;
        auto *destination = output
            + static_cast<std::ptrdiff_t>(row) * output_row_stride;
        for (std::int32_t column = 0; column < columns; ++column) {
            const auto scaled = std::clamp(
                source[column] * conversion.output_scale
                    + conversion.output_offset,
                0.0F, static_cast<float>(conversion.output_maximum));
            destination[column] = static_cast<Sample>(std::nearbyint(scaled));
        }
    }
}

} // namespace

void CpuExecutor::inverse_2d(
    const AxisPlan &horizontal, const AxisPlan &vertical,
    const float *input, std::ptrdiff_t input_row_stride,
    float *output, std::ptrdiff_t output_row_stride) const {
    if (!horizontal.valid() || !vertical.valid() || !input || !output
        || input_row_stride < horizontal.source_size
        || output_row_stride < horizontal.destination_size
        || vertical.source_size < 1 || vertical.destination_size < 1) {
        throw std::invalid_argument("invalid 2D executor arguments");
    }

    if (horizontal.requires_float64() || vertical.requires_float64()) {
        require_finite_matrix(
            input, vertical.source_size, horizontal.source_size,
            input_row_stride,
            "CPU Float64 2D input contains NaN or infinity");
        const auto execute = [&] {
#if defined(DSMVC_HAS_AVX2_OBJECT)
        if (path_ == CpuPath::avx2) {
            inverse_2d_f64_avx2(
                horizontal, vertical, input, input_row_stride,
                output, output_row_stride, *impl_->workers);
            return;
        }
#endif
        bool use_neon = false;
#if defined(DSMVC_HAS_NEON_OBJECT)
        use_neon = path_ == CpuPath::neon;
#endif
        inverse_2d_f64(
            horizontal, vertical, input, input_row_stride,
            output, output_row_stride,
            static_cast<const IntegerConversion *>(nullptr), *impl_->workers,
            use_neon);
        };
        execute();
        require_finite_matrix(
            output, vertical.destination_size, horizontal.destination_size,
            output_row_stride,
            "CPU Float64 2D execution produced NaN or infinity");
        return;
    }

#if defined(DSMVC_HAS_NEON_OBJECT)
    if (path_ == CpuPath::neon) {
        const auto packed_horizontal = impl_->get(horizontal);
        const auto intermediate_stride =
            packed_horizontal->padded_destination_size;
        thread_local std::vector<float> intermediate;
        intermediate.resize(detail::checked_size_product(
            static_cast<std::size_t>(vertical.source_size),
            static_cast<std::size_t>(intermediate_stride),
            "CPU 2D intermediate"));
        inverse_rows(
            horizontal, input, input_row_stride,
            intermediate.data(), intermediate_stride, vertical.source_size);
        inverse_columns(
            vertical, intermediate.data(), intermediate_stride,
            output, output_row_stride, horizontal.destination_size);
        return;
    }
#endif
    const auto packed_vertical = impl_->get(vertical);
#if defined(DSMVC_HAS_AVX2_OBJECT)
    if (path_ == CpuPath::avx2) {
        const auto packed_horizontal = impl_->get(horizontal);
        if (vertical.source_size >= 8) {
            forward_2d_rhs_avx2(
                horizontal, *packed_horizontal, vertical, *packed_vertical,
                input, input_row_stride, output, output_row_stride,
                horizontal.destination_size);
            backward_rhs_avx2(
                vertical, *packed_vertical, output, output_row_stride,
                horizontal.destination_size);
            return;
        }
    }
#endif

    forward_2d_rhs_scalar(
        horizontal, *packed_vertical, input, input_row_stride,
        static_cast<const IntegerConversion *>(nullptr),
        output, output_row_stride);
    for (std::int32_t column = 0;
         column < horizontal.destination_size; ++column) {
        backward_rhs_scalar(vertical, output + column, output_row_stride);
    }
}

template <class Sample>
void CpuExecutor::inverse_2d_integer(
    const AxisPlan &horizontal, const AxisPlan &vertical,
    const Sample *input, std::ptrdiff_t input_row_stride,
    Sample *output, std::ptrdiff_t output_row_stride,
    const IntegerConversion &conversion) const {
    if (!horizontal.valid() || !vertical.valid() || !input || !output
        || input_row_stride < horizontal.source_size
        || output_row_stride < horizontal.destination_size
        || vertical.source_size < 1 || vertical.destination_size < 1
        || !(conversion.input_scale > 0.0F)
        || !(conversion.output_scale > 0.0F)
        || !std::isfinite(conversion.input_offset)
        || !std::isfinite(conversion.input_scale)
        || !std::isfinite(conversion.output_scale)
        || !std::isfinite(conversion.output_offset)
        || conversion.output_maximum == 0U
        || conversion.output_maximum
            > static_cast<std::uint32_t>(std::numeric_limits<Sample>::max())) {
        throw std::invalid_argument("invalid integer 2D executor arguments");
    }

    if (horizontal.requires_float64() || vertical.requires_float64()) {
        inverse_2d_f64(
            horizontal, vertical, input, input_row_stride,
            output, output_row_stride, &conversion, *impl_->workers, false);
        return;
    }

#if defined(DSMVC_HAS_NEON_OBJECT)
    if (path_ == CpuPath::neon) {
        const auto packed_horizontal = impl_->get(horizontal);
        const auto normalized_stride = packed_horizontal->padded_source_size;
        const auto intermediate_stride =
            packed_horizontal->padded_destination_size;
        thread_local std::vector<float> normalized;
        thread_local std::vector<float> intermediate;
        thread_local std::vector<float> result;
        normalized.resize(detail::checked_size_product(
            static_cast<std::size_t>(vertical.source_size),
            static_cast<std::size_t>(normalized_stride),
            "CPU normalized input"));
        intermediate.resize(detail::checked_size_product(
            static_cast<std::size_t>(vertical.source_size),
            static_cast<std::size_t>(intermediate_stride),
            "CPU integer intermediate"));
        result.resize(detail::checked_size_product(
            static_cast<std::size_t>(vertical.destination_size),
            static_cast<std::size_t>(intermediate_stride),
            "CPU integer result"));
        if constexpr (std::is_same_v<Sample, std::uint8_t>) {
            normalize_u8_neon(
                input, input_row_stride, normalized.data(), normalized_stride,
                vertical.source_size, horizontal.source_size, conversion);
        } else {
            normalize_u16_neon(
                input, input_row_stride, normalized.data(), normalized_stride,
                vertical.source_size, horizontal.source_size, conversion);
        }
        inverse_rows(
            horizontal, normalized.data(), normalized_stride,
            intermediate.data(), intermediate_stride, vertical.source_size);
        inverse_columns(
            vertical, intermediate.data(), intermediate_stride,
            result.data(), intermediate_stride, horizontal.destination_size);
        if constexpr (std::is_same_v<Sample, std::uint8_t>) {
            convert_to_u8_neon(
                result.data(), intermediate_stride, output, output_row_stride,
                vertical.destination_size, horizontal.destination_size,
                conversion);
        } else {
            convert_to_u16_neon(
                result.data(), intermediate_stride, output, output_row_stride,
                vertical.destination_size, horizontal.destination_size,
                conversion);
        }
        return;
    }
#endif
    const auto packed_vertical = impl_->get(vertical);
    const auto padded_columns = detail::checked_size_i32(
        detail::checked_size_round_up(
            static_cast<std::size_t>(horizontal.destination_size), 8U,
            "CPU integer RHS"),
        "CPU integer RHS");
    thread_local std::vector<float> rhs;
    rhs.resize(detail::checked_size_product(
        static_cast<std::size_t>(vertical.destination_size),
        static_cast<std::size_t>(padded_columns), "CPU integer RHS"));
    auto *rhs_data = rhs.data();

#if defined(DSMVC_HAS_AVX2_OBJECT)
    if (path_ == CpuPath::avx2) {
        const auto packed_horizontal = impl_->get(horizontal);
        if (vertical.source_size >= 8) {
            const auto enough_work = detail::checked_size_product(
                static_cast<std::size_t>(horizontal.destination_size),
                static_cast<std::size_t>(vertical.source_size),
                "CPU integer work") >= 262144U;
            const auto row_tasks = std::min<std::size_t>(
                impl_->workers->parallelism(),
                static_cast<std::size_t>(vertical.destination_size));
            const auto accumulate_rows = [&](std::int32_t first_row,
                                             std::int32_t last_row) {
                if constexpr (std::is_same_v<Sample, std::uint8_t>) {
                    accumulate_2d_rhs_u8_avx2(
                        horizontal, *packed_horizontal, *packed_vertical,
                        input, input_row_stride, conversion,
                        rhs_data, padded_columns, first_row, last_row);
                } else {
                    accumulate_2d_rhs_u16_avx2(
                        horizontal, *packed_horizontal, *packed_vertical,
                        input, input_row_stride, conversion,
                        rhs_data, padded_columns, first_row, last_row);
                }
            };
            const auto accumulated_in_parallel = enough_work
                && impl_->workers->try_run(row_tasks, [&](std::size_t task) {
                    const auto first_row = static_cast<std::int32_t>(
                        static_cast<std::size_t>(vertical.destination_size)
                            * task / row_tasks);
                    const auto last_row = static_cast<std::int32_t>(
                        static_cast<std::size_t>(vertical.destination_size)
                            * (task + 1U) / row_tasks);
                    accumulate_rows(first_row, last_row);
                });
            if (!accumulated_in_parallel) {
                accumulate_rows(0, vertical.destination_size);
            }

            const auto column_groups =
                static_cast<std::size_t>(padded_columns / 8);
            const auto column_tasks = std::min(
                impl_->workers->parallelism(), column_groups);
            const auto solved_in_parallel = enough_work
                && impl_->workers->try_run(
                    column_tasks, [&](std::size_t task) {
                        const auto first_group = column_groups * task / column_tasks;
                        const auto last_group =
                            column_groups * (task + 1U) / column_tasks;
                        const auto first_column = static_cast<std::int32_t>(
                            first_group * 8U);
                        const auto task_columns = static_cast<std::int32_t>(
                            (last_group - first_group) * 8U);
                        solve_rhs_columns_avx2(
                            vertical, *packed_vertical,
                            rhs_data + first_column,
                            padded_columns, task_columns);
                    });
            if (!solved_in_parallel) {
                solve_rhs_columns_avx2(
                    vertical, *packed_vertical, rhs_data,
                    padded_columns, padded_columns);
            }

            const auto convert_rows = [&](std::int32_t first_row,
                                          std::int32_t last_row) {
                const auto count = last_row - first_row;
                if constexpr (std::is_same_v<Sample, std::uint8_t>) {
                    convert_rhs_to_u8_avx2(
                        rhs_data + static_cast<std::ptrdiff_t>(first_row)
                            * padded_columns,
                        padded_columns,
                        output + static_cast<std::ptrdiff_t>(first_row)
                            * output_row_stride,
                        output_row_stride, count,
                        horizontal.destination_size, conversion);
                } else {
                    convert_rhs_to_u16_avx2(
                        rhs_data + static_cast<std::ptrdiff_t>(first_row)
                            * padded_columns,
                        padded_columns,
                        output + static_cast<std::ptrdiff_t>(first_row)
                            * output_row_stride,
                        output_row_stride, count,
                        horizontal.destination_size, conversion);
                }
            };
            const auto converted_in_parallel = enough_work
                && impl_->workers->try_run(row_tasks, [&](std::size_t task) {
                    const auto first_row = static_cast<std::int32_t>(
                        static_cast<std::size_t>(vertical.destination_size)
                            * task / row_tasks);
                    const auto last_row = static_cast<std::int32_t>(
                        static_cast<std::size_t>(vertical.destination_size)
                            * (task + 1U) / row_tasks);
                    convert_rows(first_row, last_row);
                });
            if (!converted_in_parallel) {
                convert_rows(0, vertical.destination_size);
            }
            return;
        }
    }
#endif

    accumulate_2d_integer_rhs_scalar(
        horizontal, *packed_vertical, input, input_row_stride,
        conversion, rhs_data, padded_columns);
    for (std::int32_t column = 0;
         column < horizontal.destination_size; ++column) {
        solve_rhs_scalar(vertical, rhs_data + column, padded_columns);
    }
    convert_rhs_to_integer_scalar(
        rhs_data, padded_columns, output, output_row_stride,
        vertical.destination_size, horizontal.destination_size, conversion);
}

template <class Sample>
void CpuExecutor::inverse_2d_integer_streamed(
    const AxisPlan &horizontal, const AxisPlan &vertical,
    const Sample *input, std::ptrdiff_t input_row_stride,
    Sample *output, std::ptrdiff_t output_row_stride,
    const IntegerConversion &conversion) const {
    if (!horizontal.valid() || !vertical.valid() || !input || !output
        || input_row_stride < horizontal.source_size
        || output_row_stride < horizontal.destination_size
        || vertical.source_size < 1 || vertical.destination_size < 1
        || !(conversion.input_scale > 0.0F)
        || !(conversion.output_scale > 0.0F)
        || !std::isfinite(conversion.input_offset)
        || !std::isfinite(conversion.input_scale)
        || !std::isfinite(conversion.output_scale)
        || !std::isfinite(conversion.output_offset)
        || conversion.output_maximum == 0U
        || conversion.output_maximum
            > static_cast<std::uint32_t>(std::numeric_limits<Sample>::max())) {
        throw std::invalid_argument("invalid streamed integer 2D arguments");
    }

    if (horizontal.requires_float64() || vertical.requires_float64()) {
        inverse_2d_f64(
            horizontal, vertical, input, input_row_stride,
            output, output_row_stride, &conversion, *impl_->workers, false);
        return;
    }

#if defined(DSMVC_HAS_NEON_OBJECT)
    if (path_ == CpuPath::neon && vertical.source_size >= 4) {
        const auto packed_horizontal = impl_->get(horizontal);
        const auto packed_vertical = impl_->get(vertical);
        if constexpr (std::is_same_v<Sample, std::uint8_t>) {
            inverse_2d_u8_neon(
                horizontal, *packed_horizontal, vertical, *packed_vertical,
                input, input_row_stride, output, output_row_stride,
                conversion);
        } else {
            inverse_2d_u16_neon(
                horizontal, *packed_horizontal, vertical, *packed_vertical,
                input, input_row_stride, output, output_row_stride,
                conversion);
        }
        return;
    }
#endif
    const auto packed_vertical = impl_->get(vertical);
    const auto padded_columns = detail::checked_size_i32(
        detail::checked_size_round_up(
            static_cast<std::size_t>(horizontal.destination_size), 8U,
            "CPU streamed integer RHS"),
        "CPU streamed integer RHS");
    thread_local std::vector<float> rhs;
    rhs.resize(detail::checked_size_product(
        static_cast<std::size_t>(vertical.destination_size),
        static_cast<std::size_t>(padded_columns),
        "CPU streamed integer RHS"));
    auto *rhs_data = rhs.data();

#if defined(DSMVC_HAS_AVX2_OBJECT)
    if (path_ == CpuPath::avx2) {
        const auto packed_horizontal = impl_->get(horizontal);
        if (vertical.source_size >= 8) {
            if constexpr (std::is_same_v<Sample, std::uint8_t>) {
                forward_2d_rhs_u8_avx2(
                    horizontal, *packed_horizontal,
                    vertical, *packed_vertical,
                    input, input_row_stride, conversion,
                    rhs_data, padded_columns, horizontal.destination_size);
                backward_rhs_to_u8_avx2(
                    vertical, *packed_vertical, rhs_data, padded_columns,
                    output, output_row_stride,
                    horizontal.destination_size, conversion);
            } else {
                forward_2d_rhs_u16_avx2(
                    horizontal, *packed_horizontal,
                    vertical, *packed_vertical,
                    input, input_row_stride, conversion,
                    rhs_data, padded_columns, horizontal.destination_size);
                backward_rhs_to_u16_avx2(
                    vertical, *packed_vertical, rhs_data, padded_columns,
                    output, output_row_stride,
                    horizontal.destination_size, conversion);
            }
            return;
        }
    }
#endif

    forward_2d_rhs_scalar(
        horizontal, *packed_vertical, input, input_row_stride,
        &conversion, rhs_data, padded_columns);
    backward_rhs_to_integer_scalar(
        vertical, rhs_data, padded_columns,
        output, output_row_stride,
        horizontal.destination_size, conversion);
}

void CpuExecutor::inverse_2d_u8(
    const AxisPlan &horizontal, const AxisPlan &vertical,
    const std::uint8_t *input, std::ptrdiff_t input_row_stride,
    std::uint8_t *output, std::ptrdiff_t output_row_stride,
    const IntegerConversion &conversion) const {
    inverse_2d_integer(
        horizontal, vertical, input, input_row_stride,
        output, output_row_stride, conversion);
}

void CpuExecutor::inverse_2d_u16(
    const AxisPlan &horizontal, const AxisPlan &vertical,
    const std::uint16_t *input, std::ptrdiff_t input_row_stride,
    std::uint16_t *output, std::ptrdiff_t output_row_stride,
    const IntegerConversion &conversion) const {
    inverse_2d_integer(
        horizontal, vertical, input, input_row_stride,
        output, output_row_stride, conversion);
}

void CpuExecutor::inverse_2d_u8_streamed(
    const AxisPlan &horizontal, const AxisPlan &vertical,
    const std::uint8_t *input, std::ptrdiff_t input_row_stride,
    std::uint8_t *output, std::ptrdiff_t output_row_stride,
    const IntegerConversion &conversion) const {
    inverse_2d_integer_streamed(
        horizontal, vertical, input, input_row_stride,
        output, output_row_stride, conversion);
}

void CpuExecutor::inverse_2d_u16_streamed(
    const AxisPlan &horizontal, const AxisPlan &vertical,
    const std::uint16_t *input, std::ptrdiff_t input_row_stride,
    std::uint16_t *output, std::ptrdiff_t output_row_stride,
    const IntegerConversion &conversion) const {
    inverse_2d_integer_streamed(
        horizontal, vertical, input, input_row_stride,
        output, output_row_stride, conversion);
}

} // namespace dsmvc
