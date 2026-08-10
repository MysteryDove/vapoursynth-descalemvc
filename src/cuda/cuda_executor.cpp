#include "cuda_executor.hpp"

#include "cuda_launch.hpp"
#include "cuda_kernel.hpp"
#include "cuda_runtime.hpp"
#include "nvtx.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <charconv>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <iterator>
#include <limits>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dsmvc::cuda_detail {
namespace {

constexpr std::size_t maximum_allocation_bytes =
    2ULL * 1024ULL * 1024ULL * 1024ULL - 1ULL;
constexpr unsigned int default_inverse_threads = 64U;

[[nodiscard]] std::size_t checked_product(
    std::size_t left, std::size_t right, const char *label) {
    if (left != 0U && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::length_error(std::string{label} + " size overflow");
    }
    return left * right;
}

[[nodiscard]] std::size_t checked_add(
    std::size_t left, std::size_t right, const char *label) {
    if (right > std::numeric_limits<std::size_t>::max() - left) {
        throw std::length_error(std::string{label} + " size overflow");
    }
    return left + right;
}

[[nodiscard]] std::size_t growing_capacity(
    std::size_t current, std::size_t required) noexcept {
    if (current == 0U) return required;
    const std::size_t headroom = current / 2U;
    const std::size_t grown = current > maximum_allocation_bytes - headroom
        ? maximum_allocation_bytes : current + headroom;
    return std::max(required, grown);
}

[[nodiscard]] std::uint32_t checked_u32(
    std::size_t value, const char *label) {
    if (value > std::numeric_limits<std::uint32_t>::max()) {
        throw std::length_error(std::string{label} + " exceeds CUDA ABI limits");
    }
    return static_cast<std::uint32_t>(value);
}

struct ExecutionSlotConfiguration {
    std::size_t total = 8U;
    std::size_t limited = 4U;
    bool adaptive = true;
};

enum class ExecutionSlotClass : std::uint8_t {
    one_dimensional,
    light_2d,
    host_limited_2d,
    heavy_2d,
};

[[nodiscard]] ExecutionSlotConfiguration
execution_slot_configuration() noexcept {
    constexpr ExecutionSlotConfiguration fallback{};
    const char *text = std::getenv("DSMVC_CUDA_STREAMS");
    if (!text) return fallback;
    std::size_t value = 0U;
    const std::string_view view{text};
    const auto parsed = std::from_chars(
        view.data(), view.data() + view.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != view.data() + view.size()
        || value < 1U || value > 16U) {
        return fallback;
    }
    return {value, value, false};
}

[[nodiscard]] unsigned int inverse_thread_count(
    const char *environment_name,
    unsigned int fallback = default_inverse_threads) noexcept {
    const char *text = std::getenv(environment_name);
    if (!text) return fallback;
    unsigned int value = 0U;
    const std::string_view view{text};
    const auto parsed = std::from_chars(
        view.data(), view.data() + view.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != view.data() + view.size()
        || value < 16U || value > 256U || (value & (value - 1U)) != 0U) {
        return fallback;
    }
    return value;
}

[[nodiscard]] bool environment_flag(
    const char *name, bool fallback) noexcept {
    const char *text = std::getenv(name);
    if (!text) return fallback;
    const std::string_view value{text};
    if (value == "1") return true;
    if (value == "0") return false;
    return fallback;
}

enum class SplitRhsMode : std::uint8_t {
    disabled,
    adaptive,
    forced,
};

enum class HostTransferMode : std::uint8_t {
    staging,
    pageable,
    registered,
    staging_decoupled,
};

[[nodiscard]] HostTransferMode host_transfer_mode() noexcept {
    const char *text = std::getenv("DSMVC_CUDA_HOST_TRANSFER");
    if (text && std::string_view{text} == "staging") {
        return HostTransferMode::staging;
    }
    if (text && std::string_view{text} == "pageable") {
        return HostTransferMode::pageable;
    }
    if (text && std::string_view{text} == "registered") {
        return HostTransferMode::registered;
    }
    if (!text || std::string_view{text} == "staging-decoupled") {
        return HostTransferMode::staging_decoupled;
    }
    return HostTransferMode::staging_decoupled;
}

[[nodiscard]] SplitRhsMode split_rhs_mode() noexcept {
    const char *text = std::getenv("DSMVC_CUDA_SPLIT_RHS");
    if (text && std::string_view{text} == "0") {
        return SplitRhsMode::disabled;
    }
    if (!text || std::string_view{text} == "1"
        || std::string_view{text} == "adaptive") {
        return SplitRhsMode::adaptive;
    }
    if (std::string_view{text} == "force") {
        return SplitRhsMode::forced;
    }
    return SplitRhsMode::disabled;
}

[[nodiscard]] std::size_t plan_cache_capacity_bytes() noexcept {
    // GetNative-style scans create thousands of one-shot plans. A small cache
    // retains hot fixed plans without reserving hundreds of MiB for dead scan
    // candidates; callers with a stable large working set can raise the cap.
    constexpr std::size_t fallback_mb = 16U;
    const char *text = std::getenv("DSMVC_CUDA_PLAN_CACHE_MB");
    if (!text) return fallback_mb * 1024U * 1024U;
    std::size_t value = 0U;
    const std::string_view view{text};
    const auto parsed = std::from_chars(
        view.data(), view.data() + view.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != view.data() + view.size()
        || value < 16U || value > 4096U) {
        return fallback_mb * 1024U * 1024U;
    }
    return value * 1024U * 1024U;
}

[[nodiscard]] std::size_t input_cache_capacity_bytes() noexcept {
    constexpr std::size_t fallback_mb = 64U;
    const char *text = std::getenv("DSMVC_CUDA_INPUT_CACHE_MB");
    if (!text) return fallback_mb * 1024U * 1024U;
    std::size_t value = 0U;
    const std::string_view view{text};
    const auto parsed = std::from_chars(
        view.data(), view.data() + view.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != view.data() + view.size()
        || value > 4096U) {
        return fallback_mb * 1024U * 1024U;
    }
    return value * 1024U * 1024U;
}

class CurrentDeviceGuard {
public:
    explicit CurrentDeviceGuard(int desired) {
        cuda_check(cudaGetDevice(&previous_), "cudaGetDevice");
        if (previous_ != desired) {
            cuda_check(cudaSetDevice(desired), "cudaSetDevice");
            changed_ = true;
        }
    }

    ~CurrentDeviceGuard() {
        if (changed_) (void)cudaSetDevice(previous_);
    }

    CurrentDeviceGuard(const CurrentDeviceGuard &) = delete;
    CurrentDeviceGuard &operator=(const CurrentDeviceGuard &) = delete;

private:
    int previous_ = -1;
    bool changed_ = false;
};

class HostRegistration {
public:
    HostRegistration() = default;

    HostRegistration(
        std::shared_ptr<RuntimeApi> api, int device, const void *pointer,
        std::size_t pitch, std::size_t row_bytes, std::size_t rows)
        : api_(std::move(api)), device_(device) {
        if (!pointer || pitch < row_bytes || row_bytes == 0U || rows == 0U) {
            throw std::invalid_argument("invalid CUDA host registration range");
        }
        bytes_ = checked_add(
            checked_product(rows - 1U, pitch, "CUDA host registration"),
            row_bytes, "CUDA host registration");
        CurrentDeviceGuard current(device_);
        NvtxRange trace{NvtxLabel::host_register, bytes_};
        cuda_check(
            *api_, api_->mem_host_register(
                const_cast<void *>(pointer), bytes_, cudaHostRegisterDefault),
            "cudaHostRegister(frame)");
        pointer_ = const_cast<void *>(pointer);
    }

    ~HostRegistration() { reset(); }
    HostRegistration(const HostRegistration &) = delete;
    HostRegistration &operator=(const HostRegistration &) = delete;
    HostRegistration(HostRegistration &&other) noexcept { swap(other); }
    HostRegistration &operator=(HostRegistration &&other) noexcept {
        if (this != &other) {
            HostRegistration temporary(std::move(other));
            swap(temporary);
        }
        return *this;
    }

    void reset() noexcept {
        if (!pointer_ || !api_) return;
        try {
            CurrentDeviceGuard current(device_);
            NvtxRange trace{NvtxLabel::host_unregister, bytes_};
            (void)api_->mem_host_unregister(pointer_);
        } catch (...) {
        }
        pointer_ = nullptr;
        bytes_ = 0U;
    }

private:
    void swap(HostRegistration &other) noexcept {
        std::swap(api_, other.api_);
        std::swap(device_, other.device_);
        std::swap(pointer_, other.pointer_);
        std::swap(bytes_, other.bytes_);
    }

    std::shared_ptr<RuntimeApi> api_;
    int device_ = -1;
    void *pointer_ = nullptr;
    std::size_t bytes_ = 0U;
};

class DeviceBuffer {
public:
    DeviceBuffer() = default;
    DeviceBuffer(std::shared_ptr<RuntimeApi> api, int device,
                 std::size_t bytes)
        : api_(std::move(api)), device_(device), bytes_(bytes) {
        if (bytes != 0U) {
            CurrentDeviceGuard current(device_);
            cuda_check(*api_, api_->mem_alloc(&pointer_, bytes), "cudaMalloc");
        }
    }

    ~DeviceBuffer() { release(); }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    DeviceBuffer(DeviceBuffer &&other) noexcept { swap(other); }
    DeviceBuffer &operator=(DeviceBuffer &&other) noexcept {
        if (this != &other) {
            DeviceBuffer temporary(std::move(other));
            swap(temporary);
        }
        return *this;
    }

    [[nodiscard]] DevicePointer pointer() const noexcept { return pointer_; }
    [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

    void reserve(const std::shared_ptr<RuntimeApi> &api, int device,
                 std::size_t bytes) {
        if (bytes <= bytes_) return;
        if (bytes > maximum_allocation_bytes) {
            throw std::length_error("CUDA buffer exceeds the 2 GiB allocation guard");
        }
        DeviceBuffer replacement(api, device, growing_capacity(bytes_, bytes));
        swap(replacement);
    }

    void reset() noexcept {
        DeviceBuffer released;
        swap(released);
    }

private:
    void release() noexcept {
        if (!pointer_ || !api_) return;
        int previous = -1;
        const bool queried = cudaGetDevice(&previous) == cudaSuccess;
        const bool changed = queried && previous != device_
            && cudaSetDevice(device_) == cudaSuccess;
        (void)api_->mem_free(pointer_);
        if (changed) (void)cudaSetDevice(previous);
        pointer_ = nullptr;
        bytes_ = 0U;
    }

    void swap(DeviceBuffer &other) noexcept {
        std::swap(api_, other.api_);
        std::swap(device_, other.device_);
        std::swap(pointer_, other.pointer_);
        std::swap(bytes_, other.bytes_);
    }

    std::shared_ptr<RuntimeApi> api_;
    int device_ = -1;
    DevicePointer pointer_ = nullptr;
    std::size_t bytes_ = 0U;
};

class DeviceArena {
private:
    struct Chunk;

public:
    class Allocation {
    public:
        Allocation() = default;
        ~Allocation() { reset(); }
        Allocation(const Allocation &) = delete;
        Allocation &operator=(const Allocation &) = delete;
        Allocation(Allocation &&other) noexcept { swap(other); }
        Allocation &operator=(Allocation &&other) noexcept {
            if (this != &other) {
                Allocation temporary(std::move(other));
                swap(temporary);
            }
            return *this;
        }

        [[nodiscard]] DevicePointer pointer() const noexcept {
            return pointer_;
        }

    private:
        Allocation(
            DeviceArena &arena, Chunk &chunk, std::size_t offset,
            std::size_t bytes, DevicePointer pointer) noexcept
            : arena_(&arena), chunk_(&chunk), pointer_(pointer),
              offset_(offset), bytes_(bytes) {}

        void reset() noexcept {
            if (arena_) arena_->release(*chunk_, offset_, bytes_);
            arena_ = nullptr;
            chunk_ = nullptr;
            pointer_ = nullptr;
            offset_ = 0U;
            bytes_ = 0U;
        }

        void swap(Allocation &other) noexcept {
            std::swap(arena_, other.arena_);
            std::swap(chunk_, other.chunk_);
            std::swap(pointer_, other.pointer_);
            std::swap(offset_, other.offset_);
            std::swap(bytes_, other.bytes_);
        }

        DeviceArena *arena_ = nullptr;
        Chunk *chunk_ = nullptr;
        DevicePointer pointer_ = nullptr;
        std::size_t offset_ = 0U;
        std::size_t bytes_ = 0U;
        friend class DeviceArena;
    };

    [[nodiscard]] Allocation allocate(
        const std::shared_ptr<RuntimeApi> &api, int device,
        std::size_t bytes) {
        constexpr std::size_t alignment = 256U;
        constexpr std::size_t maximum_chunk_bytes = 64U * 1024U * 1024U;
        if (bytes == 0U) return {};
        const std::size_t aligned_bytes = checked_add(
            bytes, alignment - 1U, "CUDA arena allocation")
            & ~(alignment - 1U);
        if (aligned_bytes > maximum_allocation_bytes) {
            throw std::length_error(
                "CUDA arena allocation exceeds the 2 GiB guard");
        }

        const std::scoped_lock lock(mutex_);
        for (const auto &chunk : chunks_) {
            if (auto allocation = try_allocate(*chunk, aligned_bytes)) {
                return std::move(*allocation);
            }
        }

        std::size_t capacity = next_chunk_bytes_;
        while (capacity < aligned_bytes && capacity < maximum_chunk_bytes) {
            capacity *= 2U;
        }
        capacity = std::max(capacity, aligned_bytes);
        auto chunk = std::make_unique<Chunk>(api, device, capacity);
        Chunk &created = *chunk;
        chunks_.push_back(std::move(chunk));
        auto allocation = try_allocate(created, aligned_bytes);
        if (!allocation) {
            throw std::logic_error("new CUDA arena chunk cannot satisfy allocation");
        }
        if (capacity <= maximum_chunk_bytes / 2U) {
            next_chunk_bytes_ = capacity * 2U;
        } else {
            next_chunk_bytes_ = maximum_chunk_bytes;
        }
        return std::move(*allocation);
    }

    void reset() noexcept {
        const std::scoped_lock lock(mutex_);
        chunks_.clear();
        next_chunk_bytes_ = 1U * 1024U * 1024U;
    }

private:
    struct Chunk {
        Chunk(
            const std::shared_ptr<RuntimeApi> &api, int device,
            std::size_t bytes)
            : storage(api, device, bytes) {
            free_ranges.emplace(0U, bytes);
        }

        DeviceBuffer storage;
        std::map<std::size_t, std::size_t> free_ranges;
    };

    [[nodiscard]] std::optional<Allocation> try_allocate(
        Chunk &chunk, std::size_t bytes) {
        for (auto range = chunk.free_ranges.begin();
             range != chunk.free_ranges.end(); ++range) {
            if (range->second < bytes) continue;
            const std::size_t offset = range->first;
            const std::size_t remaining = range->second - bytes;
            chunk.free_ranges.erase(range);
            if (remaining != 0U) {
                chunk.free_ranges.emplace(offset + bytes, remaining);
            }
            return Allocation(
                *this, chunk, offset, bytes,
                chunk.storage.pointer() + offset);
        }
        return std::nullopt;
    }

    void release(
        Chunk &chunk, std::size_t offset, std::size_t bytes) noexcept {
        const std::scoped_lock lock(mutex_);
        auto next = chunk.free_ranges.lower_bound(offset);
        if (next != chunk.free_ranges.begin()) {
            auto previous = std::prev(next);
            if (previous->first + previous->second == offset) {
                offset = previous->first;
                bytes += previous->second;
                chunk.free_ranges.erase(previous);
            }
        }
        if (next != chunk.free_ranges.end()
            && offset + bytes == next->first) {
            bytes += next->second;
            chunk.free_ranges.erase(next);
        }
        chunk.free_ranges.emplace(offset, bytes);
    }

    std::mutex mutex_;
    std::vector<std::unique_ptr<Chunk>> chunks_;
    std::size_t next_chunk_bytes_ = 1U * 1024U * 1024U;
};

class PinnedBuffer {
public:
    PinnedBuffer() = default;
    PinnedBuffer(std::shared_ptr<RuntimeApi> api, int device,
                 std::size_t bytes)
        : api_(std::move(api)), device_(device), bytes_(bytes) {
        if (bytes != 0U) {
            CurrentDeviceGuard current(device_);
            cuda_check(
                *api_, api_->mem_host_alloc(
                    &pointer_, bytes, cudaHostAllocPortable),
                "cudaHostAlloc");
        }
    }

    ~PinnedBuffer() { release(); }
    PinnedBuffer(const PinnedBuffer &) = delete;
    PinnedBuffer &operator=(const PinnedBuffer &) = delete;
    PinnedBuffer(PinnedBuffer &&other) noexcept { swap(other); }
    PinnedBuffer &operator=(PinnedBuffer &&other) noexcept {
        if (this != &other) {
            PinnedBuffer temporary(std::move(other));
            swap(temporary);
        }
        return *this;
    }

    [[nodiscard]] void *data() noexcept { return pointer_; }
    [[nodiscard]] const void *data() const noexcept { return pointer_; }

    void reserve(const std::shared_ptr<RuntimeApi> &api, int device,
                 std::size_t bytes) {
        if (bytes <= bytes_) return;
        if (bytes > maximum_allocation_bytes) {
            throw std::length_error(
                "CUDA pinned buffer exceeds the 2 GiB allocation guard");
        }
        PinnedBuffer replacement(api, device, growing_capacity(bytes_, bytes));
        swap(replacement);
    }

private:
    void release() noexcept {
        if (!pointer_ || !api_) return;
        int previous = -1;
        const bool queried = cudaGetDevice(&previous) == cudaSuccess;
        const bool changed = queried && previous != device_
            && cudaSetDevice(device_) == cudaSuccess;
        (void)api_->mem_free_host(pointer_);
        if (changed) (void)cudaSetDevice(previous);
        pointer_ = nullptr;
        bytes_ = 0U;
    }

    void swap(PinnedBuffer &other) noexcept {
        std::swap(api_, other.api_);
        std::swap(device_, other.device_);
        std::swap(pointer_, other.pointer_);
        std::swap(bytes_, other.bytes_);
    }

    std::shared_ptr<RuntimeApi> api_;
    int device_ = -1;
    void *pointer_ = nullptr;
    std::size_t bytes_ = 0U;
};

class PinnedBlockPool {
public:
    class Allocation {
    public:
        Allocation() = default;
        ~Allocation() { reset(); }
        Allocation(const Allocation &) = delete;
        Allocation &operator=(const Allocation &) = delete;
        Allocation(Allocation &&other) noexcept { swap(other); }
        Allocation &operator=(Allocation &&other) noexcept {
            if (this != &other) {
                Allocation temporary(std::move(other));
                swap(temporary);
            }
            return *this;
        }

        [[nodiscard]] void *data() noexcept { return pointer_; }
        [[nodiscard]] const void *data() const noexcept { return pointer_; }

    private:
        Allocation(PinnedBlockPool &pool, void *pointer,
                   std::size_t block_bytes) noexcept
            : pool_(&pool), pointer_(pointer), block_bytes_(block_bytes) {}

        void reset() noexcept {
            if (pool_) pool_->release(pointer_, block_bytes_);
            pool_ = nullptr;
            pointer_ = nullptr;
            block_bytes_ = 0U;
        }

        void swap(Allocation &other) noexcept {
            std::swap(pool_, other.pool_);
            std::swap(pointer_, other.pointer_);
            std::swap(block_bytes_, other.block_bytes_);
        }

        PinnedBlockPool *pool_ = nullptr;
        void *pointer_ = nullptr;
        std::size_t block_bytes_ = 0U;
        friend class PinnedBlockPool;
    };

    [[nodiscard]] Allocation allocate(
        const std::shared_ptr<RuntimeApi> &api, int device,
        std::size_t bytes) {
        constexpr std::size_t slab_bytes = 8U * 1024U * 1024U;
        if (bytes == 0U) return {};
        if (bytes > maximum_allocation_bytes) {
            throw std::length_error(
                "CUDA pinned block exceeds the 2 GiB guard");
        }
        std::size_t block_bytes = std::bit_ceil(bytes);
        if (block_bytes > maximum_allocation_bytes) block_bytes = bytes;

        const std::scoped_lock lock(mutex_);
        SizeClass &size_class = classes_[block_bytes];
        if (!size_class.available.empty()) {
            void *pointer = size_class.available.back();
            size_class.available.pop_back();
            return Allocation(*this, pointer, block_bytes);
        }

        const std::size_t capacity = std::max(block_bytes, slab_bytes);
        const std::size_t block_count = capacity / block_bytes;
        size_class.available.reserve(
            checked_add(size_class.total_blocks, block_count,
                        "CUDA pinned block pool"));
        auto slab = std::make_unique<PinnedBuffer>(api, device, capacity);
        auto *base = static_cast<std::byte *>(slab->data());
        size_class.slabs.push_back(std::move(slab));
        for (std::size_t index = 1U; index < block_count; ++index) {
            size_class.available.push_back(base + index * block_bytes);
        }
        size_class.total_blocks += block_count;
        return Allocation(*this, base, block_bytes);
    }

    void reset() noexcept {
        decltype(classes_) released;
        {
            const std::scoped_lock lock(mutex_);
            released.swap(classes_);
        }
    }

private:
    struct SizeClass {
        std::vector<std::unique_ptr<PinnedBuffer>> slabs;
        std::vector<void *> available;
        std::size_t total_blocks = 0U;
    };

    void release(void *pointer, std::size_t block_bytes) noexcept {
        const std::scoped_lock lock(mutex_);
        classes_.at(block_bytes).available.push_back(pointer);
    }

    std::mutex mutex_;
    std::map<std::size_t, SizeClass> classes_;
};

class DeviceEvent {
public:
    DeviceEvent() = default;
    DeviceEvent(std::shared_ptr<RuntimeApi> api, int device)
        : api_(std::move(api)), device_(device) {
        CurrentDeviceGuard current(device_);
        cuda_check(
            *api_, api_->event_create(&event_, cudaEventDisableTiming),
            "cudaEventCreateWithFlags");
    }

    ~DeviceEvent() { release(); }
    DeviceEvent(const DeviceEvent &) = delete;
    DeviceEvent &operator=(const DeviceEvent &) = delete;
    DeviceEvent(DeviceEvent &&other) noexcept { swap(other); }
    DeviceEvent &operator=(DeviceEvent &&other) noexcept {
        if (this != &other) {
            DeviceEvent temporary(std::move(other));
            swap(temporary);
        }
        return *this;
    }

    [[nodiscard]] cudaEvent_t event() const noexcept { return event_; }

private:
    void release() noexcept {
        if (!event_ || !api_) return;
        int previous = -1;
        const bool queried = cudaGetDevice(&previous) == cudaSuccess;
        const bool changed = queried && previous != device_
            && cudaSetDevice(device_) == cudaSuccess;
        (void)api_->event_destroy(event_);
        if (changed) (void)cudaSetDevice(previous);
        event_ = nullptr;
    }

    void swap(DeviceEvent &other) noexcept {
        std::swap(api_, other.api_);
        std::swap(device_, other.device_);
        std::swap(event_, other.event_);
    }

    std::shared_ptr<RuntimeApi> api_;
    int device_ = -1;
    cudaEvent_t event_ = nullptr;
};

class DeviceEventPool {
public:
    class Handle {
    public:
        Handle() = default;
        ~Handle() { reset(); }
        Handle(const Handle &) = delete;
        Handle &operator=(const Handle &) = delete;
        Handle(Handle &&other) noexcept { swap(other); }
        Handle &operator=(Handle &&other) noexcept {
            if (this != &other) {
                Handle temporary(std::move(other));
                swap(temporary);
            }
            return *this;
        }

        [[nodiscard]] cudaEvent_t event() const noexcept { return event_; }

    private:
        Handle(DeviceEventPool &pool, cudaEvent_t event) noexcept
            : pool_(&pool), event_(event) {}

        void reset() noexcept {
            if (pool_) pool_->release(event_);
            pool_ = nullptr;
            event_ = nullptr;
        }

        void swap(Handle &other) noexcept {
            std::swap(pool_, other.pool_);
            std::swap(event_, other.event_);
        }

        DeviceEventPool *pool_ = nullptr;
        cudaEvent_t event_ = nullptr;
        friend class DeviceEventPool;
    };

    [[nodiscard]] Handle acquire(
        const std::shared_ptr<RuntimeApi> &api, int device) {
        {
            const std::scoped_lock lock(mutex_);
            if (!api_) {
                api_ = api;
                device_ = device;
            }
            if (!available_.empty()) {
                cudaEvent_t event = available_.back();
                available_.pop_back();
                return Handle(*this, event);
            }
        }
        CurrentDeviceGuard current(device);
        cudaEvent_t event = nullptr;
        cuda_check(
            *api, api->event_create(&event, cudaEventDisableTiming),
            "cudaEventCreateWithFlags(axis plan)");
        return Handle(*this, event);
    }

    void reset() noexcept {
        std::vector<cudaEvent_t> released;
        std::shared_ptr<RuntimeApi> api;
        int device = -1;
        {
            const std::scoped_lock lock(mutex_);
            released.swap(available_);
            api = api_;
            device = device_;
        }
        if (!api || device < 0) return;
        try {
            CurrentDeviceGuard current(device);
            for (cudaEvent_t event : released) {
                (void)api->event_destroy(event);
            }
        } catch (...) {
        }
    }

private:
    void release(cudaEvent_t event) noexcept {
        try {
            const std::scoped_lock lock(mutex_);
            available_.push_back(event);
            return;
        } catch (...) {
        }
        if (!api_ || !event) return;
        try {
            CurrentDeviceGuard current(device_);
            (void)api_->event_destroy(event);
        } catch (...) {
        }
    }

    std::mutex mutex_;
    std::shared_ptr<RuntimeApi> api_;
    int device_ = -1;
    std::vector<cudaEvent_t> available_;
};

enum class CachedInputLayout : std::uint8_t {
    row_major,
    transposed,
};

enum class CachedInputSample : std::uint8_t {
    float32,
    uint8,
    uint16,
};

struct InputCacheKey {
    std::uintptr_t data = 0U;
    std::ptrdiff_t row_stride = 0;
    std::uint32_t width = 0U;
    std::uint32_t height = 0U;
    std::uint32_t input_offset = 0U;
    std::uint32_t input_scale = 0U;
    CachedInputLayout layout = CachedInputLayout::row_major;
    CachedInputSample sample = CachedInputSample::float32;

    bool operator==(const InputCacheKey &) const noexcept = default;
};

struct InputCacheKeyHash {
    [[nodiscard]] std::size_t operator()(
        const InputCacheKey &key) const noexcept {
        std::size_t value = std::hash<std::uintptr_t>{}(key.data);
        const auto mix = [&value](std::size_t item) {
            value ^= item + 0x9e3779b97f4a7c15ULL
                + (value << 6U) + (value >> 2U);
        };
        mix(std::hash<std::ptrdiff_t>{}(key.row_stride));
        mix(key.width);
        mix(key.height);
        mix(key.input_offset);
        mix(key.input_scale);
        mix(static_cast<std::size_t>(key.layout));
        mix(static_cast<std::size_t>(key.sample));
        return value;
    }
};

class CachedInput {
public:
    CachedInput(
        const std::shared_ptr<RuntimeApi> &api, int device,
        DeviceArena &arena, std::size_t bytes,
        std::shared_ptr<const void> requested_lifetime)
        : api_(api), storage_(arena.allocate(api, device, bytes)),
          ready_(api, device), lifetime_(std::move(requested_lifetime)),
          bytes_(bytes) {}

    [[nodiscard]] DevicePointer pointer() const noexcept {
        return storage_.pointer();
    }

    [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

    void publish(cudaStream_t stream) {
        cuda_check(
            *api_, api_->event_record(ready_.event(), stream),
            "cudaEventRecord(shared input)");
        {
            const std::scoped_lock lock(mutex_);
            scheduled_ = true;
        }
        scheduled_ready_.notify_all();
    }

    void fail(std::exception_ptr error) noexcept {
        {
            const std::scoped_lock lock(mutex_);
            error_ = std::move(error);
            scheduled_ = true;
        }
        scheduled_ready_.notify_all();
    }

    void wait_until_scheduled() const {
        std::exception_ptr error;
        {
            std::unique_lock lock(mutex_);
            scheduled_ready_.wait(lock, [&] { return scheduled_; });
            error = error_;
        }
        if (error) std::rethrow_exception(error);
    }

    void wait_on(cudaStream_t stream) const {
        wait_until_scheduled();
        cuda_check(
            *api_, api_->stream_wait_event(stream, ready_.event(), 0U),
            "cudaStreamWaitEvent(shared input)");
    }

private:
    std::shared_ptr<RuntimeApi> api_;
    DeviceArena::Allocation storage_;
    DeviceEvent ready_;
    std::shared_ptr<const void> lifetime_;
    std::size_t bytes_ = 0U;
    mutable std::mutex mutex_;
    mutable std::condition_variable scheduled_ready_;
    bool scheduled_ = false;
    std::exception_ptr error_;
};

class InputCache {
public:
    struct Acquisition {
        std::shared_ptr<CachedInput> input;
        bool producer = false;
    };

    InputCache() : capacity_(input_cache_capacity_bytes()) {}

    [[nodiscard]] bool enabled() const noexcept { return capacity_ != 0U; }

    [[nodiscard]] Acquisition acquire(
        const InputCacheKey &key, const std::shared_ptr<RuntimeApi> &api,
        int device, DeviceArena &arena, std::size_t bytes,
        std::shared_ptr<const void> lifetime) {
        std::vector<std::shared_ptr<CachedInput>> evicted;
        Acquisition result;
        {
            const std::scoped_lock lock(mutex_);
            if (const auto found = entries_.find(key); found != entries_.end()) {
                lru_.splice(lru_.end(), lru_, found->second.lru);
                return {found->second.input, false};
            }

            result.input = std::make_shared<CachedInput>(
                api, device, arena, bytes, std::move(lifetime));
            result.producer = true;
            bytes_ = checked_add(bytes_, bytes, "CUDA input cache");
            lru_.push_back(key);
            entries_.emplace(
                key, Entry{result.input, std::prev(lru_.end())});
            while (bytes_ > capacity_ && entries_.size() > 1U) {
                const InputCacheKey victim_key = lru_.front();
                const auto victim = entries_.find(victim_key);
                bytes_ -= victim->second.input->bytes();
                evicted.push_back(std::move(victim->second.input));
                lru_.pop_front();
                entries_.erase(victim);
            }
        }
        return result;
    }

    void erase(
        const InputCacheKey &key,
        const std::shared_ptr<CachedInput> &expected) noexcept {
        std::shared_ptr<CachedInput> removed;
        {
            const std::scoped_lock lock(mutex_);
            const auto found = entries_.find(key);
            if (found == entries_.end() || found->second.input != expected) return;
            bytes_ -= found->second.input->bytes();
            lru_.erase(found->second.lru);
            removed = std::move(found->second.input);
            entries_.erase(found);
        }
    }

    void reset() noexcept {
        decltype(entries_) removed;
        {
            const std::scoped_lock lock(mutex_);
            removed.swap(entries_);
            lru_.clear();
            bytes_ = 0U;
        }
    }

private:
    struct Entry {
        std::shared_ptr<CachedInput> input;
        std::list<InputCacheKey>::iterator lru;
    };

    std::mutex mutex_;
    std::unordered_map<InputCacheKey, Entry, InputCacheKeyHash> entries_;
    std::list<InputCacheKey> lru_;
    std::size_t bytes_ = 0U;
    const std::size_t capacity_ = 0U;
};

template <class Sample>
[[nodiscard]] InputCacheKey make_input_cache_key(
    const Sample *input, std::ptrdiff_t row_stride,
    std::uint32_t width, std::uint32_t height, CachedInputLayout layout,
    const IntegerConversion *conversion = nullptr) noexcept {
    CachedInputSample sample = CachedInputSample::float32;
    if constexpr (std::is_same_v<Sample, std::uint8_t>) {
        sample = CachedInputSample::uint8;
    } else if constexpr (std::is_same_v<Sample, std::uint16_t>) {
        sample = CachedInputSample::uint16;
    }
    InputCacheKey key{
        reinterpret_cast<std::uintptr_t>(input),
        row_stride,
        width,
        height,
        0U,
        0U,
        layout,
        sample,
    };
    if (conversion) {
        key.input_offset = std::bit_cast<std::uint32_t>(
            conversion->input_offset);
        key.input_scale = std::bit_cast<std::uint32_t>(
            conversion->input_scale);
    }
    return key;
}

struct ExecutionSlot {
    ExecutionSlot(const std::shared_ptr<RuntimeApi> &requested_api, int device)
        : api(requested_api), cuda_device(device) {
        CurrentDeviceGuard current(cuda_device);
        cuda_check(
            *api, api->stream_create(&stream, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags");
    }

    ~ExecutionSlot() {
        if (stream && api) {
            int previous = -1;
            const bool queried = cudaGetDevice(&previous) == cudaSuccess;
            const bool changed = queried && previous != cuda_device
                && cudaSetDevice(cuda_device) == cudaSuccess;
            (void)api->stream_synchronize(stream);
            (void)api->stream_destroy(stream);
            if (changed) (void)cudaSetDevice(previous);
        }
    }

    ExecutionSlot(const ExecutionSlot &) = delete;
    ExecutionSlot &operator=(const ExecutionSlot &) = delete;

    std::shared_ptr<RuntimeApi> api;
    int cuda_device = -1;
    cudaStream_t stream = nullptr;
    PinnedBuffer host_source;
    PinnedBuffer host_destination;
    DeviceBuffer source;
    DeviceBuffer transposed;
    DeviceBuffer intermediate;
    DeviceBuffer destination;
    DeviceBuffer integer_output;
    DeviceBuffer f64_status;
};

class Runtime {
public:
    Runtime() {
        initialize_nvtx();
        horizontal_threads = inverse_thread_count(
            "DSMVC_CUDA_HORIZONTAL_THREADS");
        vertical_threads = inverse_thread_count(
            "DSMVC_CUDA_VERTICAL_THREADS");
        split_horizontal_threads = inverse_thread_count(
            "DSMVC_CUDA_SPLIT_HORIZONTAL_THREADS", 32U);
        split_vertical_threads = inverse_thread_count(
            "DSMVC_CUDA_SPLIT_VERTICAL_THREADS", 32U);
        horizontal_global_transpose = environment_flag(
            "DSMVC_CUDA_HORIZONTAL_GLOBAL_TRANSPOSE", true);
        split_rhs = split_rhs_mode();
        host_transfer = host_transfer_mode();
        api = std::make_shared<RuntimeApi>();

        int count = 0;
        cuda_check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
        if (count < 1) throw std::runtime_error("no CUDA device is available");
        cuda_check(cudaGetDevice(&device_ordinal), "cudaGetDevice");
        CurrentDeviceGuard current(device_ordinal);
        int compute_major = 0;
        cuda_check(
            cudaDeviceGetAttribute(
                &compute_major, cudaDevAttrComputeCapabilityMajor,
                device_ordinal),
            "cudaDeviceGetAttribute(compute capability)");
        native_double_supported = compute_major >= 2;
        cuda_check(cudaFree(nullptr), "CUDA primary-context initialization");
        try {
            cuda_check(
                *api, api->stream_create(
                    &plan_stream, cudaStreamNonBlocking),
                "cudaStreamCreateWithFlags(plan upload)");

            const auto slot_configuration = execution_slot_configuration();
            limited_slot_count = slot_configuration.limited;
            adaptive_slots = slot_configuration.adaptive;
            maximum_slot_count = slot_configuration.total;
            maximum_staging_count = std::min<std::size_t>(
                32U, std::max<std::size_t>(2U, maximum_slot_count * 2U));
            limited_staging_count = std::min(
                maximum_staging_count,
                std::size_t{4U});
            slots.reserve(maximum_slot_count);
            const std::size_t initial_slot_count = adaptive_slots
                ? limited_slot_count : maximum_slot_count;
            for (std::size_t index = 0U;
                 index < initial_slot_count; ++index) {
                slots.push_back(
                    std::make_unique<ExecutionSlot>(api, device_ordinal));
            }
            slot_busy.reserve(maximum_slot_count);
            slot_busy.assign(initial_slot_count, false);
        } catch (...) {
            slots.clear();
            if (plan_stream) {
                (void)api->stream_synchronize(plan_stream);
                (void)api->stream_destroy(plan_stream);
            }
            plan_stream = nullptr;
            throw;
        }
    }

    ~Runtime() {
        if (!api || device_ordinal < 0) return;
        try {
            CurrentDeviceGuard current(device_ordinal);
            slots.clear();
            input_cache.reset();
            input_arena.reset();
            frame_staging_arena.reset();
            if (plan_stream) {
                (void)api->stream_synchronize(plan_stream);
            }
            plan_event_pool.reset();
            plan_staging_arena.reset();
            plan_arena.reset();
            if (plan_stream) {
                (void)api->stream_destroy(plan_stream);
                plan_stream = nullptr;
            }
        } catch (...) {
        }
    }

    [[nodiscard]] std::size_t acquire_slot(ExecutionSlotClass slot_class) {
        std::unique_lock lock(slot_mutex);
        if (adaptive_slots && slot_class == ExecutionSlotClass::heavy_2d
            && slots.size() < maximum_slot_count) {
            CurrentDeviceGuard current(device_ordinal);
            while (slots.size() < maximum_slot_count) {
                slots.push_back(
                    std::make_unique<ExecutionSlot>(api, device_ordinal));
                slot_busy.push_back(false);
            }
        }
        slot_available.wait(lock, [&] {
            const bool limited = adaptive_slots
                && slot_class != ExecutionSlotClass::heavy_2d;
            const std::size_t limit =
                slot_class == ExecutionSlotClass::host_limited_2d
                ? std::min<std::size_t>(2U, limited_slot_count)
                : limited_slot_count;
            return (!limited || limited_slot_busy < limit)
                && std::find(slot_busy.begin(), slot_busy.end(), false)
                    != slot_busy.end();
        });
        const auto found = std::find(slot_busy.begin(), slot_busy.end(), false);
        *found = true;
        if (adaptive_slots && slot_class != ExecutionSlotClass::heavy_2d) {
            ++limited_slot_busy;
        }
        return static_cast<std::size_t>(found - slot_busy.begin());
    }

    void release_slot(
        std::size_t index, ExecutionSlotClass slot_class) noexcept {
        {
            const std::scoped_lock lock(slot_mutex);
            slot_busy[index] = false;
            if (adaptive_slots
                && slot_class != ExecutionSlotClass::heavy_2d) {
                --limited_slot_busy;
            }
        }
        slot_available.notify_one();
    }

    void acquire_staging(bool limited) {
        std::unique_lock lock(staging_mutex);
        staging_available.wait(lock, [&] {
            return staging_busy < maximum_staging_count
                && (!limited || limited_staging_busy < limited_staging_count);
        });
        ++staging_busy;
        if (limited) ++limited_staging_busy;
    }

    void release_staging(bool limited) noexcept {
        {
            const std::scoped_lock lock(staging_mutex);
            --staging_busy;
            if (limited) --limited_staging_busy;
        }
        staging_available.notify_one();
    }

    void attach_executor() {
        const std::scoped_lock lock(executor_mutex);
        ++executor_count;
    }

    void detach_executor() noexcept {
        bool reset_cache = false;
        {
            const std::scoped_lock lock(executor_mutex);
            reset_cache = executor_count != 0U && --executor_count == 0U;
        }
        if (reset_cache) input_cache.reset();
    }

    std::shared_ptr<RuntimeApi> api;
    int device_ordinal = -1;
    cudaStream_t plan_stream = nullptr;
    unsigned int horizontal_threads = default_inverse_threads;
    unsigned int vertical_threads = default_inverse_threads;
    unsigned int split_horizontal_threads = 32U;
    unsigned int split_vertical_threads = 32U;
    bool horizontal_global_transpose = true;
    SplitRhsMode split_rhs = SplitRhsMode::disabled;
    HostTransferMode host_transfer = HostTransferMode::staging;
    DeviceArena plan_arena;
    PinnedBlockPool plan_staging_arena;
    DeviceEventPool plan_event_pool;
    DeviceArena input_arena;
    InputCache input_cache;
    PinnedBlockPool frame_staging_arena;
    std::vector<std::unique_ptr<ExecutionSlot>> slots;
    std::vector<bool> slot_busy;
    std::mutex slot_mutex;
    std::condition_variable slot_available;
    std::size_t maximum_slot_count = 8U;
    std::size_t limited_slot_count = 4U;
    std::size_t limited_slot_busy = 0U;
    bool adaptive_slots = true;
    bool native_double_supported = false;
    std::mutex staging_mutex;
    std::condition_variable staging_available;
    std::size_t maximum_staging_count = 16U;
    std::size_t limited_staging_count = 8U;
    std::size_t staging_busy = 0U;
    std::size_t limited_staging_busy = 0U;
    std::mutex executor_mutex;
    std::size_t executor_count = 0U;
};

[[nodiscard]] std::shared_ptr<Runtime> shared_runtime() {
    static const auto instance = std::make_shared<Runtime>();
    return instance;
}

class SlotRelease {
public:
    SlotRelease(Runtime &runtime, std::size_t index,
                ExecutionSlotClass slot_class) noexcept
        : runtime_(&runtime), index_(index),
          slot_class_(slot_class) {}
    ~SlotRelease() {
        if (!runtime_) return;
        if (!completed_) {
            try {
                CurrentDeviceGuard context(runtime_->device_ordinal);
                ExecutionSlot &slot = *runtime_->slots[index_];
                (void)runtime_->api->stream_synchronize(slot.stream);
            } catch (...) {
            }
        }
        runtime_->release_slot(index_, slot_class_);
    }
    SlotRelease(const SlotRelease &) = delete;
    SlotRelease &operator=(const SlotRelease &) = delete;

    void mark_completed() noexcept { completed_ = true; }

    void release_completed() noexcept {
        if (!runtime_) return;
        completed_ = true;
        runtime_->release_slot(index_, slot_class_);
        runtime_ = nullptr;
    }

private:
    Runtime *runtime_ = nullptr;
    std::size_t index_ = 0U;
    ExecutionSlotClass slot_class_ = ExecutionSlotClass::one_dimensional;
    bool completed_ = false;
};

class StagingRelease {
public:
    StagingRelease(Runtime &runtime, bool limited)
        : runtime_(&runtime), limited_(limited) {
        NvtxRange trace{NvtxLabel::staging_wait};
        runtime_->acquire_staging(limited_);
    }

    ~StagingRelease() {
        if (runtime_) runtime_->release_staging(limited_);
    }
    StagingRelease(const StagingRelease &) = delete;
    StagingRelease &operator=(const StagingRelease &) = delete;

private:
    Runtime *runtime_ = nullptr;
    bool limited_ = false;
};

[[nodiscard]] std::size_t packed_plan_storage_bytes(const AxisPlan &axis) {
    std::size_t total = 0U;
    const auto add_field = [&](const auto &values) {
        using Value = typename std::remove_cvref_t<decltype(values)>::value_type;
        const std::size_t alignment = alignof(Value);
        const std::size_t remainder = total % alignment;
        if (remainder != 0U) {
            total = checked_add(
                total, alignment - remainder, "CUDA packed plan alignment");
        }
        total = checked_add(
            total,
            checked_product(values.size(), sizeof(Value), "CUDA plan field"),
            "CUDA packed plan");
    };
    add_field(axis.transpose_offsets);
    add_field(axis.transpose_indices);
    if (axis.requires_float64()) {
        add_field(axis.transpose_weights_f64);
        add_field(axis.ldlt_bands_f64);
    } else {
        add_field(axis.transpose_weights);
        add_field(axis.lower_ld);
        add_field(axis.upper_l);
        add_field(axis.inverse_diagonal);
    }
    if (total > maximum_allocation_bytes) {
        throw std::length_error("CUDA packed plan exceeds 2 GiB");
    }
    return total;
}

struct PackedPlan {
    PackedPlan(std::shared_ptr<Runtime> requested_runtime,
               std::shared_ptr<const AxisPlan> requested_axis,
               const AxisPlan *requested_identity)
        : runtime(std::move(requested_runtime)), axis(std::move(requested_axis)),
          identity(requested_identity ? requested_identity : axis.get()) {
        if (!axis || !axis->valid()) {
            throw std::invalid_argument("cannot pack an invalid CUDA axis plan");
        }
        descriptor.source_size = static_cast<std::uint32_t>(axis->source_size);
        descriptor.destination_size =
            static_cast<std::uint32_t>(axis->destination_size);
        descriptor.half_bandwidth =
            static_cast<std::uint32_t>(axis->half_bandwidth);
        precision = axis->requires_float64()
            ? cuda_kernel::PlanPrecision::float64
            : cuda_kernel::PlanPrecision::float32;
        if (precision == cuda_kernel::PlanPrecision::float64
            && !runtime->native_double_supported) {
            throw std::runtime_error(
                "CUDA device does not support native Float64 execution");
        }

        storage_bytes = packed_plan_storage_bytes(*axis);
        std::size_t total_bytes = 0U;
        const auto reserve_field = [&](const auto &values) {
            using Value = typename std::remove_cvref_t<decltype(values)>::value_type;
            const std::size_t alignment = alignof(Value);
            const std::size_t remainder = total_bytes % alignment;
            if (remainder != 0U) {
                total_bytes = checked_add(
                    total_bytes, alignment - remainder,
                    "CUDA packed plan alignment");
            }
            const std::size_t bytes = checked_product(
                values.size(), sizeof(Value), "CUDA plan field");
            const std::size_t offset = total_bytes;
            total_bytes = checked_add(total_bytes, bytes, "CUDA packed plan");
            return offset;
        };
        const std::size_t offsets_offset = reserve_field(axis->transpose_offsets);
        const std::size_t indices_offset = reserve_field(axis->transpose_indices);
        std::size_t weights_offset = 0U;
        std::size_t lower_offset = 0U;
        std::size_t upper_offset = 0U;
        std::size_t diagonal_offset = 0U;
        std::size_t weights_f64_offset = 0U;
        std::size_t bands_f64_offset = 0U;
        if (precision == cuda_kernel::PlanPrecision::float64) {
            weights_f64_offset = reserve_field(axis->transpose_weights_f64);
            bands_f64_offset = reserve_field(axis->ldlt_bands_f64);
        } else {
            weights_offset = reserve_field(axis->transpose_weights);
            lower_offset = reserve_field(axis->lower_ld);
            upper_offset = reserve_field(axis->upper_l);
            diagonal_offset = reserve_field(axis->inverse_diagonal);
        }
        if (total_bytes != storage_bytes) {
            throw std::logic_error("CUDA packed plan layout accounting mismatch");
        }
        CurrentDeviceGuard context(runtime->device_ordinal);
        staging = runtime->plan_staging_arena.allocate(
            runtime->api, runtime->device_ordinal, storage_bytes);
        ready = runtime->plan_event_pool.acquire(
            runtime->api, runtime->device_ordinal);
        const auto copy_field = [&](std::size_t offset, const auto &values) {
            using Value = typename std::remove_cvref_t<decltype(values)>::value_type;
            if (!values.empty()) {
                std::memcpy(
                    static_cast<std::byte *>(staging.data())
                        + static_cast<std::ptrdiff_t>(offset),
                    values.data(), values.size() * sizeof(Value));
            }
        };
        {
            NvtxRange trace{NvtxLabel::plan_pack, storage_bytes};
            std::memset(staging.data(), 0, storage_bytes);
            copy_field(offsets_offset, axis->transpose_offsets);
            copy_field(indices_offset, axis->transpose_indices);
            if (precision == cuda_kernel::PlanPrecision::float64) {
                copy_field(weights_f64_offset, axis->transpose_weights_f64);
                copy_field(bands_f64_offset, axis->ldlt_bands_f64);
            } else {
                copy_field(weights_offset, axis->transpose_weights);
                copy_field(lower_offset, axis->lower_ld);
                copy_field(upper_offset, axis->upper_l);
                copy_field(diagonal_offset, axis->inverse_diagonal);
            }
        }

        storage = runtime->plan_arena.allocate(
            runtime->api, runtime->device_ordinal, storage_bytes);
        transpose_offsets = storage.pointer() + offsets_offset;
        transpose_indices = storage.pointer() + indices_offset;
        if (precision == cuda_kernel::PlanPrecision::float64) {
            transpose_weights_f64 = storage.pointer() + weights_f64_offset;
            ldlt_bands_f64 = storage.pointer() + bands_f64_offset;
        } else {
            transpose_weights = storage.pointer() + weights_offset;
            lower_ld = storage.pointer() + lower_offset;
            upper_l = storage.pointer() + upper_offset;
            inverse_diagonal = storage.pointer() + diagonal_offset;
        }
        bool submitted = false;
        {
            NvtxRange trace{NvtxLabel::plan_upload, storage_bytes};
            try {
                cuda_check(
                    *runtime->api,
                    runtime->api->memcpy_htod_async(
                        storage.pointer(), staging.data(), storage_bytes,
                        runtime->plan_stream),
                    "cudaMemcpyAsync(axis plan upload)");
                submitted = true;
                cuda_check(
                    *runtime->api,
                    runtime->api->event_record(
                        ready.event(), runtime->plan_stream),
                    "cudaEventRecord(axis plan)");
            } catch (...) {
                if (submitted) {
                    (void)runtime->api->stream_synchronize(runtime->plan_stream);
                }
                throw;
            }
        }
    }

    ~PackedPlan() {
        if (upload_complete.load(std::memory_order_acquire)) return;
        try {
            const std::scoped_lock lock(upload_mutex);
            if (upload_complete.load(std::memory_order_relaxed)) return;
            CurrentDeviceGuard context(runtime->device_ordinal);
            if (runtime->api->event_synchronize(ready.event()) == cudaSuccess) {
                upload_complete.store(true, std::memory_order_release);
            }
        } catch (...) {
        }
    }

    void wait_on(cudaStream_t stream) const {
        if (upload_complete.load(std::memory_order_acquire)) return;
        PinnedBlockPool::Allocation retired_staging;
        DeviceEventPool::Handle retired_ready;
        const std::scoped_lock lock(upload_mutex);
        if (upload_complete.load(std::memory_order_relaxed)) return;
        const cudaError_t status = runtime->api->event_query(ready.event());
        if (status == cudaSuccess) {
            upload_complete.store(true, std::memory_order_release);
            retired_staging = std::move(staging);
            retired_ready = std::move(ready);
            return;
        }
        if (status != cudaErrorNotReady) {
            cuda_check(*runtime->api, status, "cudaEventQuery(axis plan)");
        }
        cuda_check(
            *runtime->api,
            runtime->api->stream_wait_event(stream, ready.event(), 0U),
            "cudaStreamWaitEvent(axis plan)");
    }

    void mark_upload_complete() const noexcept {
        if (upload_complete.load(std::memory_order_acquire)) return;
        PinnedBlockPool::Allocation retired_staging;
        DeviceEventPool::Handle retired_ready;
        const std::scoped_lock lock(upload_mutex);
        if (upload_complete.load(std::memory_order_relaxed)) return;
        upload_complete.store(true, std::memory_order_release);
        retired_staging = std::move(staging);
        retired_ready = std::move(ready);
    }

    std::shared_ptr<Runtime> runtime;
    std::shared_ptr<const AxisPlan> axis;
    const AxisPlan *identity = nullptr;
    cuda_kernel::AxisPlanDescriptor descriptor{};
    cuda_kernel::PlanPrecision precision =
        cuda_kernel::PlanPrecision::float32;
    DeviceArena::Allocation storage;
    mutable PinnedBlockPool::Allocation staging;
    mutable DeviceEventPool::Handle ready;
    std::size_t storage_bytes = 0U;
    DevicePointer transpose_offsets = nullptr;
    DevicePointer transpose_indices = nullptr;
    DevicePointer transpose_weights = nullptr;
    DevicePointer lower_ld = nullptr;
    DevicePointer upper_l = nullptr;
    DevicePointer inverse_diagonal = nullptr;
    DevicePointer transpose_weights_f64 = nullptr;
    DevicePointer ldlt_bands_f64 = nullptr;
    mutable std::mutex upload_mutex;
    mutable std::atomic<bool> upload_complete{false};
    mutable std::atomic<std::uint32_t> execution_count{0U};
};

class PackedPlanRequest {
public:
    explicit PackedPlanRequest(std::shared_ptr<const AxisPlan> requested_axis)
        : axis_(std::move(requested_axis)) {}

    [[nodiscard]] const std::shared_ptr<const AxisPlan> &axis() const noexcept {
        return axis_;
    }

    void publish(std::shared_ptr<const PackedPlan> packed) noexcept {
        {
            const std::scoped_lock lock(mutex_);
            packed_ = std::move(packed);
            completed_ = true;
        }
        completed_ready_.notify_all();
    }

    void fail(std::exception_ptr error) noexcept {
        {
            const std::scoped_lock lock(mutex_);
            error_ = std::move(error);
            completed_ = true;
        }
        completed_ready_.notify_all();
    }

    [[nodiscard]] std::shared_ptr<const PackedPlan> get() const {
        std::unique_lock lock(mutex_);
        completed_ready_.wait(lock, [&] { return completed_; });
        if (error_) std::rethrow_exception(error_);
        return packed_;
    }

private:
    std::shared_ptr<const AxisPlan> axis_;
    mutable std::mutex mutex_;
    mutable std::condition_variable completed_ready_;
    std::shared_ptr<const PackedPlan> packed_;
    std::exception_ptr error_;
    bool completed_ = false;
};

[[nodiscard]] bool should_split_rhs(
    const Runtime &runtime, const PackedPlan &plan) noexcept {
    if (runtime.split_rhs == SplitRhsMode::disabled
        || plan.descriptor.half_bandwidth < 3U) {
        return false;
    }
    if (runtime.split_rhs == SplitRhsMode::forced) return true;
    return plan.execution_count.fetch_add(1U, std::memory_order_relaxed) != 0U;
}

[[nodiscard]] std::shared_ptr<const PackedPlan> acquire_packed_plan(
    const std::shared_ptr<Runtime> &runtime,
    const std::shared_ptr<const AxisPlan> &plan) {
    struct CacheEntry {
        std::shared_ptr<PackedPlanRequest> request;
        std::list<const AxisPlan *>::iterator lru;
        std::size_t bytes = 0U;
    };
    struct Cache {
        std::mutex mutex;
        std::unordered_map<const AxisPlan *, CacheEntry> entries;
        std::list<const AxisPlan *> lru;
        std::size_t bytes = 0U;
        const std::size_t capacity = plan_cache_capacity_bytes();
    };
    static Cache cache;

    const std::size_t storage_bytes = packed_plan_storage_bytes(*plan);
    std::shared_ptr<PackedPlanRequest> request;
    std::vector<std::shared_ptr<PackedPlanRequest>> evicted;
    bool producer = false;
    {
        NvtxRange trace{NvtxLabel::plan_cache_lookup, storage_bytes};
        const std::scoped_lock lock(cache.mutex);
        if (const auto found = cache.entries.find(plan.get());
            found != cache.entries.end()) {
            if (found->second.request->axis() == plan) {
                cache.lru.splice(
                    cache.lru.end(), cache.lru, found->second.lru);
                request = found->second.request;
            } else {
                cache.bytes -= found->second.bytes;
                cache.lru.erase(found->second.lru);
                evicted.push_back(std::move(found->second.request));
                cache.entries.erase(found);
            }
        }
        if (!request) {
            request = std::make_shared<PackedPlanRequest>(plan);
            producer = true;
            cache.bytes = checked_add(
                cache.bytes, storage_bytes, "CUDA packed-plan cache");
            cache.lru.push_back(plan.get());
            cache.entries.emplace(
                plan.get(), CacheEntry{
                    request, std::prev(cache.lru.end()), storage_bytes});
            while (cache.bytes > cache.capacity && cache.entries.size() > 1U) {
                const AxisPlan *victim_key = cache.lru.front();
                const auto victim = cache.entries.find(victim_key);
                cache.bytes -= victim->second.bytes;
                evicted.push_back(std::move(victim->second.request));
                cache.lru.pop_front();
                cache.entries.erase(victim);
            }
        }
    }

    if (!producer) {
        NvtxRange trace{NvtxLabel::plan_cache_hit, storage_bytes};
        return request->get();
    }
    try {
        NvtxRange trace{NvtxLabel::plan_cache_miss, storage_bytes};
        auto packed = std::make_shared<const PackedPlan>(
            runtime, plan, plan.get());
        request->publish(packed);
        return packed;
    } catch (...) {
        const auto error = std::current_exception();
        request->fail(error);
        std::shared_ptr<PackedPlanRequest> removed;
        {
            const std::scoped_lock lock(cache.mutex);
            const auto found = cache.entries.find(plan.get());
            if (found != cache.entries.end()
                && found->second.request == request) {
                cache.bytes -= found->second.bytes;
                cache.lru.erase(found->second.lru);
                removed = std::move(found->second.request);
                cache.entries.erase(found);
            }
        }
        std::rethrow_exception(error);
    }
}

void pack_host_rows(
    const void *source, std::size_t source_pitch, void *destination,
    std::size_t row_bytes, std::size_t rows) {
    if (source_pitch == row_bytes) {
        std::memcpy(destination, source, row_bytes * rows);
        return;
    }
    const auto *source_bytes = static_cast<const std::byte *>(source);
    auto *destination_bytes = static_cast<std::byte *>(destination);
    for (std::size_t row = 0U; row < rows; ++row) {
        std::memcpy(
            destination_bytes + static_cast<std::ptrdiff_t>(row * row_bytes),
            source_bytes + static_cast<std::ptrdiff_t>(row * source_pitch),
            row_bytes);
    }
}

void unpack_host_rows(
    const void *source, std::size_t row_bytes, void *destination,
    std::size_t destination_pitch, std::size_t rows) {
    if (destination_pitch == row_bytes) {
        std::memcpy(destination, source, row_bytes * rows);
        return;
    }
    const auto *source_bytes = static_cast<const std::byte *>(source);
    auto *destination_bytes = static_cast<std::byte *>(destination);
    for (std::size_t row = 0U; row < rows; ++row) {
        std::memcpy(
            destination_bytes + static_cast<std::ptrdiff_t>(
                row * destination_pitch),
            source_bytes + static_cast<std::ptrdiff_t>(row * row_bytes),
            row_bytes);
    }
}

enum class TransposeSample : std::uint8_t {
    float32,
    uint8,
    uint16,
};

template <class Value>
[[nodiscard]] Value *device_as(DevicePointer pointer) noexcept {
    return reinterpret_cast<Value *>(pointer);
}

void launch_transpose(
    Runtime &runtime, ExecutionSlot &slot, TransposeSample sample,
    std::uint32_t width, std::uint32_t height,
    const cuda_kernel::IntegerConversionDescriptor *conversion,
    DevicePointer requested_source = nullptr,
    DevicePointer requested_destination = nullptr) {
    DevicePointer source = requested_source
        ? requested_source : slot.source.pointer();
    DevicePointer transposed = requested_destination
        ? requested_destination : slot.transposed.pointer();
    cudaError_t status = cudaErrorInvalidValue;
    switch (sample) {
    case TransposeSample::float32:
        status = cuda_launch::transpose(
            device_as<const float>(source), width, height,
            device_as<float>(transposed), slot.stream);
        break;
    case TransposeSample::uint8:
        status = cuda_launch::transpose(
            device_as<const std::uint8_t>(source), width, height, *conversion,
            device_as<float>(transposed), slot.stream);
        break;
    case TransposeSample::uint16:
        status = cuda_launch::transpose(
            device_as<const std::uint16_t>(source), width, height, *conversion,
            device_as<float>(transposed), slot.stream);
        break;
    }
    cuda_check(
        *runtime.api, status, "CUDA transpose kernel launch");
}

void launch_transpose_f64(
    Runtime &runtime, ExecutionSlot &slot, TransposeSample sample,
    std::uint32_t width, std::uint32_t height,
    const cuda_kernel::IntegerConversionDescriptor *conversion,
    DevicePointer requested_source, DevicePointer requested_destination) {
    cudaError_t status = cudaErrorInvalidValue;
    switch (sample) {
    case TransposeSample::float32:
        status = cuda_launch::transpose_f64(
            device_as<const float>(requested_source), width, height,
            device_as<double>(requested_destination), slot.stream);
        break;
    case TransposeSample::uint8:
        status = cuda_launch::transpose_f64(
            device_as<const std::uint8_t>(requested_source), width, height,
            *conversion, device_as<double>(requested_destination), slot.stream);
        break;
    case TransposeSample::uint16:
        status = cuda_launch::transpose_f64(
            device_as<const std::uint16_t>(requested_source), width, height,
            *conversion, device_as<double>(requested_destination), slot.stream);
        break;
    }
    cuda_check(
        *runtime.api, status, "CUDA Float64 transpose kernel launch");
}

void launch_promote_f64(
    Runtime &runtime, ExecutionSlot &slot, DevicePointer source,
    std::uint32_t element_count, DevicePointer destination) {
    cuda_check(
        *runtime.api,
        cuda_launch::promote_f64(
            device_as<const float>(source), element_count,
            device_as<double>(destination), slot.stream),
        "CUDA Float64 promotion kernel launch");
}

void clear_f64_status(Runtime &runtime, ExecutionSlot &slot) {
    cuda_check(
        *runtime.api,
        runtime.api->mem_zero_async(
            slot.f64_status.pointer(), sizeof(std::uint32_t), slot.stream),
        "cudaMemsetAsync(Float64 finite status)");
}

void schedule_f64_finite_check(
    Runtime &runtime, ExecutionSlot &slot, DevicePointer source,
    std::uint32_t element_count) {
    cuda_check(
        *runtime.api,
        cuda_launch::check_finite_f64(
            device_as<const double>(source), element_count,
            device_as<std::uint32_t>(slot.f64_status.pointer()), slot.stream),
        "CUDA Float64 finite-check kernel launch");
}

void schedule_f64_status_download(
    Runtime &runtime, ExecutionSlot &slot,
    std::uint32_t &host_nonfinite) {
    cuda_check(
        *runtime.api,
        runtime.api->memcpy_dtoh_async(
            &host_nonfinite, slot.f64_status.pointer(),
            sizeof(host_nonfinite), slot.stream),
        "cudaMemcpyAsync(Float64 finite status download)");
}

void launch_horizontal(
    Runtime &runtime, ExecutionSlot &slot, const PackedPlan &plan,
    std::uint32_t vector_count, DevicePointer output,
    DevicePointer requested_transposed = nullptr) {
    DevicePointer transposed = requested_transposed
        ? requested_transposed : slot.transposed.pointer();
    const unsigned int shared_bytes = runtime.horizontal_global_transpose
        ? 0U
        : runtime.horizontal_threads * 33U
            * static_cast<unsigned int>(sizeof(float));
    cuda_check(
        *runtime.api,
        cuda_launch::inverse_horizontal(
            device_as<const float>(transposed), vector_count, plan.descriptor,
            device_as<const std::uint32_t>(plan.transpose_offsets),
            device_as<const std::int32_t>(plan.transpose_indices),
            device_as<const float>(plan.transpose_weights),
            device_as<const float>(plan.lower_ld),
            device_as<const float>(plan.upper_l),
            device_as<const float>(plan.inverse_diagonal),
            device_as<float>(output), runtime.horizontal_global_transpose,
            runtime.horizontal_threads, shared_bytes, slot.stream),
        "CUDA inverse-horizontal kernel launch");
}

void launch_horizontal_split(
    Runtime &runtime, ExecutionSlot &slot, const PackedPlan &plan,
    std::uint32_t vector_count, DevicePointer output,
    DevicePointer requested_transposed = nullptr) {
    DevicePointer transposed = requested_transposed
        ? requested_transposed : slot.transposed.pointer();
    cuda_check(
        *runtime.api,
        cuda_launch::rhs_horizontal(
            device_as<const float>(transposed), vector_count, plan.descriptor,
            device_as<const std::uint32_t>(plan.transpose_offsets),
            device_as<const std::int32_t>(plan.transpose_indices),
            device_as<const float>(plan.transpose_weights),
            device_as<float>(output), runtime.horizontal_global_transpose,
            slot.stream),
        "CUDA horizontal-RHS kernel launch");

    const unsigned int shared_bytes = runtime.horizontal_global_transpose
        ? 0U
        : runtime.split_horizontal_threads * 33U
            * static_cast<unsigned int>(sizeof(float));
    cuda_check(
        *runtime.api,
        cuda_launch::solve_horizontal(
            vector_count, plan.descriptor,
            device_as<const float>(plan.lower_ld),
            device_as<const float>(plan.upper_l),
            device_as<const float>(plan.inverse_diagonal),
            device_as<float>(output), runtime.horizontal_global_transpose,
            runtime.split_horizontal_threads, shared_bytes, slot.stream),
        "CUDA horizontal-solve kernel launch");
}

void launch_vertical(
    Runtime &runtime, ExecutionSlot &slot, const PackedPlan &plan,
    DevicePointer source, std::uint32_t source_width, DevicePointer output) {
    cuda_check(
        *runtime.api,
        cuda_launch::inverse_vertical(
            device_as<const float>(source), source_width, plan.descriptor,
            device_as<const std::uint32_t>(plan.transpose_offsets),
            device_as<const std::int32_t>(plan.transpose_indices),
            device_as<const float>(plan.transpose_weights),
            device_as<const float>(plan.lower_ld),
            device_as<const float>(plan.upper_l),
            device_as<const float>(plan.inverse_diagonal),
            device_as<float>(output), runtime.vertical_threads, slot.stream),
        "CUDA inverse-vertical kernel launch");
}

void launch_vertical_split(
    Runtime &runtime, ExecutionSlot &slot, const PackedPlan &plan,
    DevicePointer source, std::uint32_t source_width, DevicePointer output) {
    cuda_check(
        *runtime.api,
        cuda_launch::rhs_vertical(
            device_as<const float>(source), source_width, plan.descriptor,
            device_as<const std::uint32_t>(plan.transpose_offsets),
            device_as<const std::int32_t>(plan.transpose_indices),
            device_as<const float>(plan.transpose_weights),
            device_as<float>(output), slot.stream),
        "CUDA vertical-RHS kernel launch");

    cuda_check(
        *runtime.api,
        cuda_launch::solve_vertical(
            source_width, plan.descriptor,
            device_as<const float>(plan.lower_ld),
            device_as<const float>(plan.upper_l),
            device_as<const float>(plan.inverse_diagonal),
            device_as<float>(output), runtime.split_vertical_threads,
            slot.stream),
        "CUDA vertical-solve kernel launch");
}

void launch_horizontal_f64(
    Runtime &runtime, ExecutionSlot &slot, const PackedPlan &plan,
    std::uint32_t vector_count, DevicePointer source, DevicePointer output) {
    cuda_check(
        *runtime.api,
        cuda_launch::inverse_horizontal_f64(
            device_as<const double>(source), vector_count, plan.descriptor,
            device_as<const std::uint32_t>(plan.transpose_offsets),
            device_as<const std::int32_t>(plan.transpose_indices),
            device_as<const float>(plan.transpose_weights),
            device_as<const float>(plan.lower_ld),
            device_as<const float>(plan.upper_l),
            device_as<const float>(plan.inverse_diagonal),
            device_as<const double>(plan.transpose_weights_f64),
            device_as<const double>(plan.ldlt_bands_f64),
            plan.precision, device_as<double>(output),
            runtime.horizontal_threads, slot.stream),
        "CUDA Float64 inverse-horizontal kernel launch");
}

void launch_horizontal_split_f64(
    Runtime &runtime, ExecutionSlot &slot, const PackedPlan &plan,
    std::uint32_t vector_count, DevicePointer source, DevicePointer output) {
    cuda_check(
        *runtime.api,
        cuda_launch::rhs_horizontal_f64(
            device_as<const double>(source), vector_count, plan.descriptor,
            device_as<const std::uint32_t>(plan.transpose_offsets),
            device_as<const std::int32_t>(plan.transpose_indices),
            device_as<const float>(plan.transpose_weights),
            device_as<const double>(plan.transpose_weights_f64),
            plan.precision, device_as<double>(output), slot.stream),
        "CUDA Float64 horizontal-RHS kernel launch");
    cuda_check(
        *runtime.api,
        cuda_launch::solve_horizontal_f64(
            vector_count, plan.descriptor,
            device_as<const float>(plan.lower_ld),
            device_as<const float>(plan.upper_l),
            device_as<const float>(plan.inverse_diagonal),
            device_as<const double>(plan.ldlt_bands_f64),
            plan.precision, device_as<double>(output),
            runtime.split_horizontal_threads, slot.stream),
        "CUDA Float64 horizontal-solve kernel launch");
}

void launch_vertical_f64(
    Runtime &runtime, ExecutionSlot &slot, const PackedPlan &plan,
    DevicePointer source, std::uint32_t source_width, DevicePointer output) {
    cuda_check(
        *runtime.api,
        cuda_launch::inverse_vertical_f64(
            device_as<const double>(source), source_width, plan.descriptor,
            device_as<const std::uint32_t>(plan.transpose_offsets),
            device_as<const std::int32_t>(plan.transpose_indices),
            device_as<const float>(plan.transpose_weights),
            device_as<const float>(plan.lower_ld),
            device_as<const float>(plan.upper_l),
            device_as<const float>(plan.inverse_diagonal),
            device_as<const double>(plan.transpose_weights_f64),
            device_as<const double>(plan.ldlt_bands_f64),
            plan.precision, device_as<double>(output),
            runtime.vertical_threads, slot.stream),
        "CUDA Float64 inverse-vertical kernel launch");
}

void launch_vertical_split_f64(
    Runtime &runtime, ExecutionSlot &slot, const PackedPlan &plan,
    DevicePointer source, std::uint32_t source_width, DevicePointer output) {
    cuda_check(
        *runtime.api,
        cuda_launch::rhs_vertical_f64(
            device_as<const double>(source), source_width, plan.descriptor,
            device_as<const std::uint32_t>(plan.transpose_offsets),
            device_as<const std::int32_t>(plan.transpose_indices),
            device_as<const float>(plan.transpose_weights),
            device_as<const double>(plan.transpose_weights_f64),
            plan.precision, device_as<double>(output), slot.stream),
        "CUDA Float64 vertical-RHS kernel launch");
    cuda_check(
        *runtime.api,
        cuda_launch::solve_vertical_f64(
            source_width, plan.descriptor,
            device_as<const float>(plan.lower_ld),
            device_as<const float>(plan.upper_l),
            device_as<const float>(plan.inverse_diagonal),
            device_as<const double>(plan.ldlt_bands_f64),
            plan.precision, device_as<double>(output),
            runtime.split_vertical_threads, slot.stream),
        "CUDA Float64 vertical-solve kernel launch");
}

template <class Sample>
void launch_conversion(
    Runtime &runtime, ExecutionSlot &slot, std::uint32_t element_count,
    const cuda_kernel::IntegerConversionDescriptor &conversion) {
    const cudaError_t status = cuda_launch::convert(
        device_as<const float>(slot.destination.pointer()), element_count,
        conversion, device_as<Sample>(slot.integer_output.pointer()),
        slot.stream);
    cuda_check(*runtime.api, status, "CUDA integer-conversion kernel launch");
}

template <class Sample>
void launch_conversion_f64(
    Runtime &runtime, ExecutionSlot &slot, DevicePointer source,
    std::uint32_t element_count,
    const cuda_kernel::IntegerConversionDescriptor *conversion,
    DevicePointer output) {
    cudaError_t status = cudaErrorInvalidValue;
    if constexpr (std::is_same_v<Sample, float>) {
        status = cuda_launch::convert_f64(
            device_as<const double>(source), element_count,
            device_as<float>(output),
            device_as<std::uint32_t>(slot.f64_status.pointer()), slot.stream);
    } else if constexpr (std::is_same_v<Sample, std::uint8_t>) {
        status = cuda_launch::convert_f64(
            device_as<const double>(source), element_count, *conversion,
            device_as<std::uint8_t>(output),
            device_as<std::uint32_t>(slot.f64_status.pointer()), slot.stream);
    } else {
        status = cuda_launch::convert_f64(
            device_as<const double>(source), element_count, *conversion,
            device_as<std::uint16_t>(output),
            device_as<std::uint32_t>(slot.f64_status.pointer()), slot.stream);
    }
    cuda_check(
        *runtime.api, status, "CUDA Float64 output-conversion kernel launch");
}

template <class Sample>
[[nodiscard]] bool valid_conversion(
    const IntegerConversion &conversion) noexcept {
    return conversion.input_scale > 0.0F
        && conversion.output_scale > 0.0F
        && std::isfinite(conversion.input_offset)
        && std::isfinite(conversion.input_scale)
        && std::isfinite(conversion.output_scale)
        && std::isfinite(conversion.output_offset)
        && conversion.output_maximum != 0U
        && conversion.output_maximum
            <= static_cast<std::uint32_t>(std::numeric_limits<Sample>::max());
}

[[nodiscard]] cuda_kernel::IntegerConversionDescriptor conversion_descriptor(
    const IntegerConversion &conversion) noexcept {
    return {
        conversion.input_offset,
        conversion.input_scale,
        conversion.output_scale,
        conversion.output_offset,
        conversion.output_maximum,
    };
}

} // namespace

bool backend_available() noexcept {
    static const bool available = [] {
        int count = 0;
        return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
    }();
    return available;
}

struct CudaExecutor::Impl {
    Impl() : runtime(shared_runtime()) { runtime->attach_executor(); }
    ~Impl() { runtime->detach_executor(); }

    struct PreparedPlan {
        std::shared_ptr<const AxisPlan> axis;
        const AxisPlan *identity = nullptr;
        std::weak_ptr<const PackedPlan> packed;
    };

    [[nodiscard]] auto find(const AxisPlan &plan) const {
        return std::find_if(
            plans.begin(), plans.end(), [&plan](const auto &candidate) {
                return candidate.identity == &plan;
            });
    }

    [[nodiscard]] std::shared_ptr<const PackedPlan> get(
        const AxisPlan &plan) const {
        if (sealed.load(std::memory_order_acquire)) {
            const auto found = find(plan);
            if (found != plans.end()) {
                if (auto packed = found->packed.lock()) return packed;
                return acquire_packed_plan(runtime, found->axis);
            }
        } else {
            const std::scoped_lock lock(mutex);
            const auto found = find(plan);
            if (found != plans.end()) {
                if (auto packed = found->packed.lock()) return packed;
                return acquire_packed_plan(runtime, found->axis);
            }
        }
        auto owned = std::make_shared<const AxisPlan>(plan);
        return std::make_shared<const PackedPlan>(runtime, std::move(owned), &plan);
    }

    template <class Sample>
    void inverse_2d_integer(
        const AxisPlan &horizontal, const AxisPlan &vertical,
        const Sample *input, std::ptrdiff_t input_row_stride,
        Sample *output, std::ptrdiff_t output_row_stride,
        const IntegerConversion &conversion,
        std::shared_ptr<const void> input_lifetime) const {
        if (!horizontal.valid() || !vertical.valid() || !input || !output
            || input_row_stride < horizontal.source_size
            || output_row_stride < horizontal.destination_size
            || vertical.source_size < 1 || vertical.destination_size < 1
            || !valid_conversion<Sample>(conversion)) {
            throw std::invalid_argument("invalid CUDA integer 2D arguments");
        }
        execute_2d(
            horizontal, vertical, input, input_row_stride,
            output, output_row_stride, &conversion,
            std::move(input_lifetime));
    }

    template <class Sample>
    void execute_2d(
        const AxisPlan &horizontal, const AxisPlan &vertical,
        const Sample *input, std::ptrdiff_t input_row_stride,
        Sample *output, std::ptrdiff_t output_row_stride,
        const IntegerConversion *conversion,
        std::shared_ptr<const void> input_lifetime) const {
        NvtxRange execute_trace{NvtxLabel::execute_2d};
        const bool use_float64 =
            horizontal.requires_float64() || vertical.requires_float64();
        if (use_float64 && !runtime->native_double_supported) {
            throw std::runtime_error(
                "CUDA device does not support native Float64 execution");
        }
        std::shared_ptr<const PackedPlan> packed_horizontal;
        std::shared_ptr<const PackedPlan> packed_vertical;
        {
            NvtxRange trace{NvtxLabel::prepared_plan_lookup};
            packed_horizontal = get(horizontal);
            packed_vertical = get(vertical);
        }
        const auto source_width = static_cast<std::uint32_t>(horizontal.source_size);
        const auto source_height = static_cast<std::uint32_t>(vertical.source_size);
        const auto destination_width =
            static_cast<std::uint32_t>(horizontal.destination_size);
        const auto destination_height =
            static_cast<std::uint32_t>(vertical.destination_size);
        const std::size_t source_elements = checked_product(
            source_width, source_height, "CUDA source image");
        const std::size_t intermediate_elements = checked_product(
            destination_width, source_height, "CUDA intermediate image");
        const std::size_t destination_elements = checked_product(
            destination_width, destination_height, "CUDA destination image");
        const std::size_t source_bytes = checked_product(
            source_elements, sizeof(Sample), "CUDA source image");
        const std::size_t transposed_bytes = checked_product(
            source_elements, use_float64 ? sizeof(double) : sizeof(float),
            "CUDA transposed image");
        const std::size_t intermediate_bytes = checked_product(
            intermediate_elements,
            use_float64 ? sizeof(double) : sizeof(float),
            "CUDA intermediate image");
        const std::size_t destination_bytes = checked_product(
            destination_elements,
            use_float64 ? sizeof(double) : sizeof(float),
            "CUDA destination image");
        const std::size_t result_bytes = checked_product(
            destination_elements, sizeof(Sample), "CUDA output image");
        const std::size_t source_row_bytes = checked_product(
            source_width, sizeof(Sample), "CUDA source row");
        const std::size_t source_pitch = checked_product(
            static_cast<std::size_t>(input_row_stride), sizeof(Sample),
            "CUDA source pitch");
        const std::size_t result_row_bytes = checked_product(
            destination_width, sizeof(Sample), "CUDA destination row");
        const std::size_t result_pitch = checked_product(
            static_cast<std::size_t>(output_row_stride), sizeof(Sample),
            "CUDA destination pitch");
        const bool registered_transfer =
            runtime->host_transfer == HostTransferMode::registered;
        const bool direct_transfer = registered_transfer
            || runtime->host_transfer == HostTransferMode::pageable;
        const bool decoupled_transfer =
            runtime->host_transfer == HostTransferMode::staging_decoupled
            && std::is_same_v<Sample, float>;
        const bool cache_requested =
            !use_float64 && input_lifetime && runtime->input_cache.enabled();
        std::shared_ptr<CachedInput> cached_input;
        std::optional<InputCacheKey> cache_key;
        bool cache_producer = false;

        const auto acquire_cached_input = [&] {
            if (!cache_requested) return;
            NvtxRange trace{NvtxLabel::input_cache_lookup, transposed_bytes};
            cache_key = make_input_cache_key(
                input, input_row_stride, source_width, source_height,
                CachedInputLayout::transposed, conversion);
            auto acquired = runtime->input_cache.acquire(
                *cache_key, runtime->api, runtime->device_ordinal,
                runtime->input_arena, transposed_bytes,
                std::move(input_lifetime));
            cached_input = std::move(acquired.input);
            cache_producer = acquired.producer;
        };
        if (direct_transfer || decoupled_transfer) {
            acquire_cached_input();
        }
        const bool reused_input = cached_input && !cache_producer;
        if (reused_input) {
            // Consumers must not occupy every slot while the producer still
            // needs one to record the shared-input event.
            cached_input->wait_until_scheduled();
        }

        std::optional<StagingRelease> staging_release;
        PinnedBlockPool::Allocation staged_source;
        PinnedBlockPool::Allocation staged_destination;
        if (decoupled_transfer) {
            staging_release.emplace(
                *runtime, runtime->adaptive_slots && !reused_input);
            {
                NvtxRange trace{NvtxLabel::buffer_reserve};
                if (!cached_input || cache_producer) {
                    staged_source = runtime->frame_staging_arena.allocate(
                        runtime->api, runtime->device_ordinal, source_bytes);
                }
                staged_destination = runtime->frame_staging_arena.allocate(
                    runtime->api, runtime->device_ordinal, result_bytes);
            }
            if (!cached_input || cache_producer) {
                NvtxRange trace{NvtxLabel::host_pack, source_bytes};
                pack_host_rows(
                    input, source_pitch, staged_source.data(),
                    source_row_bytes, source_height);
            }
        }

        std::optional<HostRegistration> source_registration;
        std::optional<HostRegistration> destination_registration;
        if (registered_transfer && (!cached_input || cache_producer)) {
            source_registration.emplace(
                runtime->api, runtime->device_ordinal, input,
                source_pitch, source_row_bytes, source_height);
        }
        if (registered_transfer) {
            destination_registration.emplace(
                runtime->api, runtime->device_ordinal, output,
                result_pitch, result_row_bytes, destination_height);
        }

        const auto maximum_bandwidth =
            std::max(horizontal.half_bandwidth, vertical.half_bandwidth);
        const ExecutionSlotClass slot_class = decoupled_transfer && !reused_input
            ? ExecutionSlotClass::host_limited_2d
            : maximum_bandwidth >= 7
                ? ExecutionSlotClass::heavy_2d
                : ExecutionSlotClass::light_2d;
        std::size_t slot_index = 0U;
        {
            NvtxRange trace{
                NvtxLabel::slot_wait,
                static_cast<std::uint64_t>(slot_class)};
            slot_index = runtime->acquire_slot(slot_class);
        }
        SlotRelease release(*runtime, slot_index, slot_class);
        CurrentDeviceGuard context(runtime->device_ordinal);
        ExecutionSlot &slot = *runtime->slots[slot_index];
        const std::size_t source_capacity = runtime->horizontal_global_transpose
                && !use_float64
            ? std::max(source_bytes, intermediate_bytes) : source_bytes;
        {
            NvtxRange trace{NvtxLabel::buffer_reserve};
            slot.source.reserve(
                runtime->api, runtime->device_ordinal, source_capacity);
            if (!cache_requested) {
                slot.transposed.reserve(
                    runtime->api, runtime->device_ordinal, transposed_bytes);
            }
            slot.intermediate.reserve(
                runtime->api, runtime->device_ordinal, intermediate_bytes);
            slot.destination.reserve(
                runtime->api, runtime->device_ordinal, destination_bytes);
            if (!direct_transfer && !decoupled_transfer) {
                slot.host_source.reserve(
                    runtime->api, runtime->device_ordinal, source_bytes);
                slot.host_destination.reserve(
                    runtime->api, runtime->device_ordinal, result_bytes);
            }
            if (use_float64 || !std::is_same_v<Sample, float>) {
                slot.integer_output.reserve(
                    runtime->api, runtime->device_ordinal, result_bytes);
            }
            if (use_float64) {
                slot.f64_status.reserve(
                    runtime->api, runtime->device_ordinal,
                    sizeof(std::uint32_t));
            }
        }

        cuda_kernel::IntegerConversionDescriptor converted{};
        TransposeSample transpose = TransposeSample::float32;
        const cuda_kernel::IntegerConversionDescriptor *converted_ptr = nullptr;
        if constexpr (std::is_same_v<Sample, std::uint8_t>) {
            converted = conversion_descriptor(*conversion);
            converted_ptr = &converted;
            transpose = TransposeSample::uint8;
        } else if constexpr (std::is_same_v<Sample, std::uint16_t>) {
            converted = conversion_descriptor(*conversion);
            converted_ptr = &converted;
            transpose = TransposeSample::uint16;
        }

        if (use_float64) {
            clear_f64_status(*runtime, slot);
        }

        if (!direct_transfer && !decoupled_transfer) {
            acquire_cached_input();
        }
        const DevicePointer transposed_input = cached_input
            ? cached_input->pointer() : slot.transposed.pointer();
        const void *transfer_source = decoupled_transfer
            ? staged_source.data() : slot.host_source.data();
        void *transfer_destination = decoupled_transfer
            ? staged_destination.data() : slot.host_destination.data();
        const auto populate_input = [&] {
            if (!direct_transfer && !decoupled_transfer) {
                NvtxRange trace{NvtxLabel::host_pack, source_bytes};
                pack_host_rows(
                    input, source_pitch, slot.host_source.data(),
                    source_row_bytes, source_height);
            }
            {
                NvtxRange trace{NvtxLabel::source_upload, source_bytes};
                const cudaError_t status = direct_transfer
                    ? runtime->api->memcpy_2d_htod_async(
                        slot.source.pointer(), source_row_bytes,
                        input, source_pitch, source_row_bytes, source_height,
                        slot.stream)
                    : runtime->api->memcpy_htod_async(
                        slot.source.pointer(), transfer_source,
                        source_bytes, slot.stream);
                cuda_check(*runtime->api, status, "CUDA source upload");
            }
            {
                NvtxRange trace{NvtxLabel::input_transpose, source_elements};
                if (use_float64) {
                    launch_transpose_f64(
                        *runtime, slot, transpose, source_width, source_height,
                        converted_ptr, slot.source.pointer(), transposed_input);
                } else {
                    launch_transpose(
                        *runtime, slot, transpose, source_width, source_height,
                        converted_ptr, slot.source.pointer(), transposed_input);
                }
            }
        };
        if (!cached_input) {
            populate_input();
        } else if (cache_producer) {
            NvtxRange trace{NvtxLabel::input_cache_miss, transposed_bytes};
            try {
                populate_input();
                cached_input->publish(slot.stream);
            } catch (...) {
                const auto error = std::current_exception();
                cached_input->fail(error);
                runtime->input_cache.erase(*cache_key, cached_input);
                std::rethrow_exception(error);
            }
        } else {
            NvtxRange trace{NvtxLabel::input_cache_hit, transposed_bytes};
            cached_input->wait_on(slot.stream);
        }
        if (use_float64) {
            schedule_f64_finite_check(
                *runtime, slot, transposed_input,
                checked_u32(source_elements, "CUDA source element count"));
        }

        const DevicePointer horizontal_output =
            runtime->horizontal_global_transpose && !use_float64
            ? slot.source.pointer() : slot.intermediate.pointer();
        {
            NvtxRange trace{NvtxLabel::horizontal_plan_wait};
            packed_horizontal->wait_on(slot.stream);
        }
        {
            NvtxRange trace{NvtxLabel::horizontal_stage, source_height};
            if (use_float64
                && should_split_rhs(*runtime, *packed_horizontal)) {
                launch_horizontal_split_f64(
                    *runtime, slot, *packed_horizontal, source_height,
                    transposed_input, horizontal_output);
            } else if (use_float64) {
                launch_horizontal_f64(
                    *runtime, slot, *packed_horizontal, source_height,
                    transposed_input, horizontal_output);
            } else if (should_split_rhs(*runtime, *packed_horizontal)) {
                launch_horizontal_split(
                    *runtime, slot, *packed_horizontal, source_height,
                    horizontal_output, transposed_input);
            } else {
                launch_horizontal(
                    *runtime, slot, *packed_horizontal, source_height,
                    horizontal_output, transposed_input);
            }
        }
        if (runtime->horizontal_global_transpose && !use_float64) {
            NvtxRange trace{
                NvtxLabel::intermediate_transpose, intermediate_elements};
            launch_transpose(
                *runtime, slot, TransposeSample::float32,
                source_height, destination_width, nullptr,
                slot.source.pointer(), slot.intermediate.pointer());
        }
        {
            NvtxRange trace{NvtxLabel::vertical_plan_wait};
            packed_vertical->wait_on(slot.stream);
        }
        {
            NvtxRange trace{NvtxLabel::vertical_stage, destination_width};
            if (use_float64
                && should_split_rhs(*runtime, *packed_vertical)) {
                launch_vertical_split_f64(
                    *runtime, slot, *packed_vertical,
                    slot.intermediate.pointer(), destination_width,
                    slot.destination.pointer());
            } else if (use_float64) {
                launch_vertical_f64(
                    *runtime, slot, *packed_vertical,
                    slot.intermediate.pointer(), destination_width,
                    slot.destination.pointer());
            } else if (should_split_rhs(*runtime, *packed_vertical)) {
                launch_vertical_split(
                    *runtime, slot, *packed_vertical,
                    slot.intermediate.pointer(), destination_width,
                    slot.destination.pointer());
            } else {
                launch_vertical(
                    *runtime, slot, *packed_vertical,
                    slot.intermediate.pointer(), destination_width,
                    slot.destination.pointer());
            }
        }
        std::uint32_t f64_nonfinite = 0U;
        if (use_float64) {
            schedule_f64_finite_check(
                *runtime, slot, slot.destination.pointer(),
                checked_u32(
                    destination_elements, "CUDA destination element count"));
        }
        DevicePointer result = slot.destination.pointer();
        if (use_float64) {
            NvtxRange trace{
                NvtxLabel::output_conversion, destination_elements};
            launch_conversion_f64<Sample>(
                *runtime, slot, slot.destination.pointer(),
                checked_u32(
                    destination_elements, "CUDA destination element count"),
                converted_ptr, slot.integer_output.pointer());
            result = slot.integer_output.pointer();
        } else if constexpr (!std::is_same_v<Sample, float>) {
            NvtxRange trace{
                NvtxLabel::output_conversion, destination_elements};
            launch_conversion<Sample>(
                *runtime, slot, checked_u32(
                    destination_elements, "CUDA destination element count"),
                converted);
            result = slot.integer_output.pointer();
        }
        if (use_float64) {
            schedule_f64_status_download(*runtime, slot, f64_nonfinite);
        }
        {
            NvtxRange trace{NvtxLabel::destination_download, result_bytes};
            const cudaError_t status = direct_transfer
                ? runtime->api->memcpy_2d_dtoh_async(
                    output, result_pitch, result, result_row_bytes,
                    result_row_bytes, destination_height, slot.stream)
                : runtime->api->memcpy_dtoh_async(
                    transfer_destination, result, result_bytes,
                    slot.stream);
            cuda_check(*runtime->api, status, "CUDA destination download");
        }
        {
            NvtxRange trace{NvtxLabel::stream_synchronize, slot_index};
            cuda_check(
                *runtime->api, runtime->api->stream_synchronize(slot.stream),
                "cudaStreamSynchronize");
        }
        packed_horizontal->mark_upload_complete();
        packed_vertical->mark_upload_complete();
        if (direct_transfer || decoupled_transfer) {
            release.release_completed();
        } else {
            release.mark_completed();
        }
        if (f64_nonfinite != 0U) {
            throw std::runtime_error(
                "CUDA Float64 2D execution produced a nonfinite value");
        }
        if (!direct_transfer) {
            NvtxRange trace{NvtxLabel::host_unpack, result_bytes};
            unpack_host_rows(
                transfer_destination, result_row_bytes,
                output, result_pitch, destination_height);
        }
    }

    std::shared_ptr<Runtime> runtime;
    mutable std::mutex mutex;
    mutable std::vector<PreparedPlan> plans;
    mutable std::atomic<bool> sealed{false};
};

CudaExecutor::CudaExecutor() : impl_(std::make_shared<Impl>()) {}
CudaExecutor::~CudaExecutor() = default;

const char *CudaExecutor::name() const noexcept {
    return "cuda";
}

bool CudaExecutor::input_cache_enabled() const noexcept {
    return impl_->runtime->input_cache.enabled();
}

void CudaExecutor::prepare(std::shared_ptr<const AxisPlan> plan) const {
    if (!plan || !plan->valid()) {
        throw std::invalid_argument("cannot prepare an invalid CUDA axis plan");
    }
    const std::scoped_lock lock(impl_->mutex);
    if (impl_->sealed.load(std::memory_order_relaxed)) {
        throw std::logic_error("cannot add an axis to a sealed CUDA plan cache");
    }
    if (impl_->find(*plan) == impl_->plans.end()) {
        auto packed = acquire_packed_plan(impl_->runtime, plan);
        impl_->plans.push_back({plan, plan.get(), packed});
    }
}

void CudaExecutor::seal() const {
    const std::scoped_lock lock(impl_->mutex);
    impl_->sealed.store(true, std::memory_order_release);
}

void CudaExecutor::inverse_rows(
    const AxisPlan &plan,
    const float *input, std::ptrdiff_t input_row_stride,
    float *output, std::ptrdiff_t output_row_stride,
    std::int32_t row_count,
    std::shared_ptr<const void> input_lifetime) const {
    if (!plan.valid() || !input || !output
        || input_row_stride < plan.source_size
        || output_row_stride < plan.destination_size || row_count < 0) {
        throw std::invalid_argument("invalid CUDA row executor arguments");
    }
    if (row_count == 0) return;
    NvtxRange execute_trace{NvtxLabel::execute_rows};
    const bool use_float64 = plan.requires_float64();
    if (use_float64 && !impl_->runtime->native_double_supported) {
        throw std::runtime_error(
            "CUDA device does not support native Float64 execution");
    }
    std::shared_ptr<const PackedPlan> packed;
    {
        NvtxRange trace{NvtxLabel::prepared_plan_lookup};
        packed = impl_->get(plan);
    }
    const auto width = static_cast<std::uint32_t>(plan.source_size);
    const auto height = static_cast<std::uint32_t>(row_count);
    const auto destination_width =
        static_cast<std::uint32_t>(plan.destination_size);
    const std::size_t source_elements = checked_product(
        width, height, "CUDA row source");
    const std::size_t output_elements = checked_product(
        destination_width, height, "CUDA row destination");
    const std::size_t source_bytes = checked_product(
        source_elements, sizeof(float), "CUDA row source");
    const std::size_t result_bytes = checked_product(
        output_elements, sizeof(float), "CUDA row destination");
    const std::size_t transposed_bytes = checked_product(
        source_elements, use_float64 ? sizeof(double) : sizeof(float),
        "CUDA row transposed source");
    const std::size_t output_bytes = checked_product(
        output_elements, use_float64 ? sizeof(double) : sizeof(float),
        "CUDA row workspace destination");

    const bool cache_requested =
        !use_float64 && input_lifetime && impl_->runtime->input_cache.enabled();
    std::shared_ptr<CachedInput> cached_input;
    std::optional<InputCacheKey> cache_key;
    bool cache_producer = false;

    std::size_t slot_index = 0U;
    {
        NvtxRange trace{
            NvtxLabel::slot_wait,
            static_cast<std::uint64_t>(ExecutionSlotClass::one_dimensional)};
        slot_index = impl_->runtime->acquire_slot(
            ExecutionSlotClass::one_dimensional);
    }
    SlotRelease release(
        *impl_->runtime, slot_index, ExecutionSlotClass::one_dimensional);
    CurrentDeviceGuard context(impl_->runtime->device_ordinal);
    ExecutionSlot &slot = *impl_->runtime->slots[slot_index];
    {
        NvtxRange trace{NvtxLabel::buffer_reserve};
        slot.source.reserve(
            impl_->runtime->api, impl_->runtime->device_ordinal, source_bytes);
        if (!cache_requested) {
            slot.transposed.reserve(
                impl_->runtime->api, impl_->runtime->device_ordinal,
                transposed_bytes);
        }
        slot.destination.reserve(
            impl_->runtime->api, impl_->runtime->device_ordinal, output_bytes);
        slot.host_source.reserve(
            impl_->runtime->api, impl_->runtime->device_ordinal, source_bytes);
        slot.host_destination.reserve(
            impl_->runtime->api, impl_->runtime->device_ordinal, result_bytes);
        if (use_float64) {
            slot.integer_output.reserve(
                impl_->runtime->api, impl_->runtime->device_ordinal,
                result_bytes);
            slot.f64_status.reserve(
                impl_->runtime->api, impl_->runtime->device_ordinal,
                sizeof(std::uint32_t));
        }
    }

    if (use_float64) {
        clear_f64_status(*impl_->runtime, slot);
    }

    if (cache_requested) {
        NvtxRange trace{NvtxLabel::input_cache_lookup, source_bytes};
        cache_key = make_input_cache_key(
            input, input_row_stride, width, height,
            CachedInputLayout::transposed);
        auto acquired = impl_->runtime->input_cache.acquire(
            *cache_key, impl_->runtime->api, impl_->runtime->device_ordinal,
            impl_->runtime->input_arena, source_bytes,
            std::move(input_lifetime));
        cached_input = std::move(acquired.input);
        cache_producer = acquired.producer;
    }
    const DevicePointer transposed_input = cached_input
        ? cached_input->pointer() : slot.transposed.pointer();
    const auto populate_input = [&] {
        {
            NvtxRange trace{NvtxLabel::host_pack, source_bytes};
            pack_host_rows(
                input,
                checked_product(
                    static_cast<std::size_t>(input_row_stride), sizeof(float),
                    "CUDA input row pitch"),
                slot.host_source.data(),
                checked_product(width, sizeof(float), "CUDA input row"), height);
        }
        {
            NvtxRange trace{NvtxLabel::source_upload, source_bytes};
            cuda_check(
                *impl_->runtime->api,
                impl_->runtime->api->memcpy_htod_async(
                    slot.source.pointer(), slot.host_source.data(), source_bytes,
                    slot.stream),
                "cudaMemcpyAsync(row source upload)");
        }
        {
            NvtxRange trace{NvtxLabel::input_transpose, source_elements};
            if (use_float64) {
                launch_transpose_f64(
                    *impl_->runtime, slot, TransposeSample::float32,
                    width, height, nullptr,
                    slot.source.pointer(), transposed_input);
            } else {
                launch_transpose(
                    *impl_->runtime, slot, TransposeSample::float32,
                    width, height, nullptr,
                    slot.source.pointer(), transposed_input);
            }
        }
    };
    if (!cached_input) {
        populate_input();
    } else if (cache_producer) {
        NvtxRange trace{NvtxLabel::input_cache_miss, source_bytes};
        try {
            populate_input();
            cached_input->publish(slot.stream);
        } catch (...) {
            const auto error = std::current_exception();
            cached_input->fail(error);
            impl_->runtime->input_cache.erase(*cache_key, cached_input);
            std::rethrow_exception(error);
        }
    } else {
        NvtxRange trace{NvtxLabel::input_cache_hit, source_bytes};
        cached_input->wait_on(slot.stream);
    }
    if (use_float64) {
        schedule_f64_finite_check(
            *impl_->runtime, slot, transposed_input,
            checked_u32(source_elements, "CUDA row source element count"));
    }

    const DevicePointer horizontal_output =
        impl_->runtime->horizontal_global_transpose && !use_float64
        ? slot.source.pointer() : slot.destination.pointer();
    {
        NvtxRange trace{NvtxLabel::horizontal_plan_wait};
        packed->wait_on(slot.stream);
    }
    {
        NvtxRange trace{NvtxLabel::horizontal_stage, height};
        if (use_float64 && should_split_rhs(*impl_->runtime, *packed)) {
            launch_horizontal_split_f64(
                *impl_->runtime, slot, *packed, height,
                transposed_input, horizontal_output);
        } else if (use_float64) {
            launch_horizontal_f64(
                *impl_->runtime, slot, *packed, height,
                transposed_input, horizontal_output);
        } else if (should_split_rhs(*impl_->runtime, *packed)) {
            launch_horizontal_split(
                *impl_->runtime, slot, *packed, height, horizontal_output,
                transposed_input);
        } else {
            launch_horizontal(
                *impl_->runtime, slot, *packed, height, horizontal_output,
                transposed_input);
        }
    }
    if (impl_->runtime->horizontal_global_transpose && !use_float64) {
        NvtxRange trace{NvtxLabel::intermediate_transpose, output_elements};
        launch_transpose(
            *impl_->runtime, slot, TransposeSample::float32,
            height, destination_width, nullptr,
            slot.source.pointer(), slot.destination.pointer());
    }
    std::uint32_t f64_nonfinite = 0U;
    if (use_float64) {
        schedule_f64_finite_check(
            *impl_->runtime, slot, slot.destination.pointer(),
            checked_u32(output_elements, "CUDA row destination element count"));
    }
    DevicePointer result = slot.destination.pointer();
    if (use_float64) {
        NvtxRange trace{NvtxLabel::output_conversion, output_elements};
        launch_conversion_f64<float>(
            *impl_->runtime, slot, slot.destination.pointer(),
            checked_u32(output_elements, "CUDA row destination element count"),
            nullptr, slot.integer_output.pointer());
        result = slot.integer_output.pointer();
    }
    if (use_float64) {
        schedule_f64_status_download(
            *impl_->runtime, slot, f64_nonfinite);
    }
    {
        NvtxRange trace{NvtxLabel::destination_download, result_bytes};
        cuda_check(
            *impl_->runtime->api,
            impl_->runtime->api->memcpy_dtoh_async(
                slot.host_destination.data(), result,
                result_bytes, slot.stream),
            "cudaMemcpyAsync(row destination download)");
    }
    {
        NvtxRange trace{NvtxLabel::stream_synchronize, slot_index};
        cuda_check(
            *impl_->runtime->api,
            impl_->runtime->api->stream_synchronize(slot.stream),
            "cudaStreamSynchronize");
    }
    packed->mark_upload_complete();
    release.mark_completed();
    if (f64_nonfinite != 0U) {
        throw std::runtime_error(
            "CUDA Float64 row execution produced a nonfinite value");
    }
    {
        NvtxRange trace{NvtxLabel::host_unpack, result_bytes};
        unpack_host_rows(
            slot.host_destination.data(),
            checked_product(
                destination_width, sizeof(float), "CUDA output row"),
            output,
            checked_product(
                static_cast<std::size_t>(output_row_stride), sizeof(float),
                "CUDA output row pitch"),
            height);
    }
}

void CudaExecutor::inverse_columns(
    const AxisPlan &plan,
    const float *input, std::ptrdiff_t input_row_stride,
    float *output, std::ptrdiff_t output_row_stride,
    std::int32_t column_count,
    std::shared_ptr<const void> input_lifetime) const {
    if (!plan.valid() || !input || !output
        || input_row_stride < column_count
        || output_row_stride < column_count || column_count < 0) {
        throw std::invalid_argument("invalid CUDA column executor arguments");
    }
    if (column_count == 0) return;
    NvtxRange execute_trace{NvtxLabel::execute_columns};
    const bool use_float64 = plan.requires_float64();
    if (use_float64 && !impl_->runtime->native_double_supported) {
        throw std::runtime_error(
            "CUDA device does not support native Float64 execution");
    }
    std::shared_ptr<const PackedPlan> packed;
    {
        NvtxRange trace{NvtxLabel::prepared_plan_lookup};
        packed = impl_->get(plan);
    }
    const auto width = static_cast<std::uint32_t>(column_count);
    const auto source_height = static_cast<std::uint32_t>(plan.source_size);
    const auto destination_height =
        static_cast<std::uint32_t>(plan.destination_size);
    const std::size_t source_elements = checked_product(
        width, source_height, "CUDA column source");
    const std::size_t output_elements = checked_product(
        width, destination_height, "CUDA column destination");
    const std::size_t source_bytes = checked_product(
        source_elements, sizeof(float), "CUDA column source");
    const std::size_t result_bytes = checked_product(
        output_elements, sizeof(float), "CUDA column destination");
    const std::size_t promoted_bytes = checked_product(
        source_elements, sizeof(double), "CUDA promoted column source");
    const std::size_t output_bytes = checked_product(
        output_elements, use_float64 ? sizeof(double) : sizeof(float),
        "CUDA column workspace destination");

    const bool cache_requested =
        !use_float64 && input_lifetime && impl_->runtime->input_cache.enabled();
    std::shared_ptr<CachedInput> cached_input;
    std::optional<InputCacheKey> cache_key;
    bool cache_producer = false;

    std::size_t slot_index = 0U;
    {
        NvtxRange trace{
            NvtxLabel::slot_wait,
            static_cast<std::uint64_t>(ExecutionSlotClass::one_dimensional)};
        slot_index = impl_->runtime->acquire_slot(
            ExecutionSlotClass::one_dimensional);
    }
    SlotRelease release(
        *impl_->runtime, slot_index, ExecutionSlotClass::one_dimensional);
    CurrentDeviceGuard context(impl_->runtime->device_ordinal);
    ExecutionSlot &slot = *impl_->runtime->slots[slot_index];
    {
        NvtxRange trace{NvtxLabel::buffer_reserve};
        slot.source.reserve(
            impl_->runtime->api, impl_->runtime->device_ordinal, source_bytes);
        slot.destination.reserve(
            impl_->runtime->api, impl_->runtime->device_ordinal, output_bytes);
        if (use_float64) {
            slot.transposed.reserve(
                impl_->runtime->api, impl_->runtime->device_ordinal,
                promoted_bytes);
            slot.integer_output.reserve(
                impl_->runtime->api, impl_->runtime->device_ordinal,
                result_bytes);
            slot.f64_status.reserve(
                impl_->runtime->api, impl_->runtime->device_ordinal,
                sizeof(std::uint32_t));
        }
        slot.host_source.reserve(
            impl_->runtime->api, impl_->runtime->device_ordinal, source_bytes);
        slot.host_destination.reserve(
            impl_->runtime->api, impl_->runtime->device_ordinal, result_bytes);
    }

    if (use_float64) {
        clear_f64_status(*impl_->runtime, slot);
    }

    if (cache_requested) {
        NvtxRange trace{NvtxLabel::input_cache_lookup, source_bytes};
        cache_key = make_input_cache_key(
            input, input_row_stride, width, source_height,
            CachedInputLayout::row_major);
        auto acquired = impl_->runtime->input_cache.acquire(
            *cache_key, impl_->runtime->api, impl_->runtime->device_ordinal,
            impl_->runtime->input_arena, source_bytes,
            std::move(input_lifetime));
        cached_input = std::move(acquired.input);
        cache_producer = acquired.producer;
    }
    const DevicePointer uploaded_input = cached_input
        ? cached_input->pointer() : slot.source.pointer();
    const DevicePointer device_input = use_float64
        ? slot.transposed.pointer() : uploaded_input;
    const auto populate_input = [&] {
        {
            NvtxRange trace{NvtxLabel::host_pack, source_bytes};
            pack_host_rows(
                input,
                checked_product(
                    static_cast<std::size_t>(input_row_stride), sizeof(float),
                    "CUDA input column pitch"),
                slot.host_source.data(),
                checked_product(width, sizeof(float), "CUDA input columns"),
                source_height);
        }
        {
            NvtxRange trace{NvtxLabel::source_upload, source_bytes};
            cuda_check(
                *impl_->runtime->api,
                impl_->runtime->api->memcpy_htod_async(
                    uploaded_input, slot.host_source.data(), source_bytes,
                    slot.stream),
                "cudaMemcpyAsync(column source upload)");
        }
        if (use_float64) {
            NvtxRange trace{NvtxLabel::input_transpose, source_elements};
            launch_promote_f64(
                *impl_->runtime, slot, uploaded_input,
                checked_u32(source_elements, "CUDA column source element count"),
                device_input);
        }
    };
    if (!cached_input) {
        populate_input();
    } else if (cache_producer) {
        NvtxRange trace{NvtxLabel::input_cache_miss, source_bytes};
        try {
            populate_input();
            cached_input->publish(slot.stream);
        } catch (...) {
            const auto error = std::current_exception();
            cached_input->fail(error);
            impl_->runtime->input_cache.erase(*cache_key, cached_input);
            std::rethrow_exception(error);
        }
    } else {
        NvtxRange trace{NvtxLabel::input_cache_hit, source_bytes};
        cached_input->wait_on(slot.stream);
    }
    if (use_float64) {
        schedule_f64_finite_check(
            *impl_->runtime, slot, device_input,
            checked_u32(source_elements, "CUDA column source element count"));
    }

    {
        NvtxRange trace{NvtxLabel::vertical_plan_wait};
        packed->wait_on(slot.stream);
    }
    {
        NvtxRange trace{NvtxLabel::vertical_stage, width};
        if (use_float64 && should_split_rhs(*impl_->runtime, *packed)) {
            launch_vertical_split_f64(
                *impl_->runtime, slot, *packed, device_input, width,
                slot.destination.pointer());
        } else if (use_float64) {
            launch_vertical_f64(
                *impl_->runtime, slot, *packed, device_input, width,
                slot.destination.pointer());
        } else if (should_split_rhs(*impl_->runtime, *packed)) {
            launch_vertical_split(
                *impl_->runtime, slot, *packed, device_input, width,
                slot.destination.pointer());
        } else {
            launch_vertical(
                *impl_->runtime, slot, *packed, device_input, width,
                slot.destination.pointer());
        }
    }
    std::uint32_t f64_nonfinite = 0U;
    if (use_float64) {
        schedule_f64_finite_check(
            *impl_->runtime, slot, slot.destination.pointer(),
            checked_u32(
                output_elements, "CUDA column destination element count"));
    }
    DevicePointer result = slot.destination.pointer();
    if (use_float64) {
        NvtxRange trace{NvtxLabel::output_conversion, output_elements};
        launch_conversion_f64<float>(
            *impl_->runtime, slot, slot.destination.pointer(),
            checked_u32(
                output_elements, "CUDA column destination element count"),
            nullptr, slot.integer_output.pointer());
        result = slot.integer_output.pointer();
    }
    if (use_float64) {
        schedule_f64_status_download(
            *impl_->runtime, slot, f64_nonfinite);
    }
    {
        NvtxRange trace{NvtxLabel::destination_download, result_bytes};
        cuda_check(
            *impl_->runtime->api,
            impl_->runtime->api->memcpy_dtoh_async(
                slot.host_destination.data(), result,
                result_bytes, slot.stream),
            "cudaMemcpyAsync(column destination download)");
    }
    {
        NvtxRange trace{NvtxLabel::stream_synchronize, slot_index};
        cuda_check(
            *impl_->runtime->api,
            impl_->runtime->api->stream_synchronize(slot.stream),
            "cudaStreamSynchronize");
    }
    packed->mark_upload_complete();
    release.mark_completed();
    if (f64_nonfinite != 0U) {
        throw std::runtime_error(
            "CUDA Float64 column execution produced a nonfinite value");
    }
    {
        NvtxRange trace{NvtxLabel::host_unpack, result_bytes};
        unpack_host_rows(
            slot.host_destination.data(),
            checked_product(width, sizeof(float), "CUDA output columns"), output,
            checked_product(
                static_cast<std::size_t>(output_row_stride), sizeof(float),
                "CUDA output column pitch"),
            destination_height);
    }
}

void CudaExecutor::inverse_2d(
    const AxisPlan &horizontal, const AxisPlan &vertical,
    const float *input, std::ptrdiff_t input_row_stride,
    float *output, std::ptrdiff_t output_row_stride,
    std::shared_ptr<const void> input_lifetime) const {
    if (!horizontal.valid() || !vertical.valid() || !input || !output
        || input_row_stride < horizontal.source_size
        || output_row_stride < horizontal.destination_size
        || vertical.source_size < 1 || vertical.destination_size < 1) {
        throw std::invalid_argument("invalid CUDA 2D executor arguments");
    }
    impl_->execute_2d(
        horizontal, vertical, input, input_row_stride,
        output, output_row_stride, nullptr, std::move(input_lifetime));
}

void CudaExecutor::inverse_2d_u8(
    const AxisPlan &horizontal, const AxisPlan &vertical,
    const std::uint8_t *input, std::ptrdiff_t input_row_stride,
    std::uint8_t *output, std::ptrdiff_t output_row_stride,
    const IntegerConversion &conversion,
    std::shared_ptr<const void> input_lifetime) const {
    impl_->inverse_2d_integer(
        horizontal, vertical, input, input_row_stride,
        output, output_row_stride, conversion, std::move(input_lifetime));
}

void CudaExecutor::inverse_2d_u16(
    const AxisPlan &horizontal, const AxisPlan &vertical,
    const std::uint16_t *input, std::ptrdiff_t input_row_stride,
    std::uint16_t *output, std::ptrdiff_t output_row_stride,
    const IntegerConversion &conversion,
    std::shared_ptr<const void> input_lifetime) const {
    impl_->inverse_2d_integer(
        horizontal, vertical, input, input_row_stride,
        output, output_row_stride, conversion, std::move(input_lifetime));
}

} // namespace dsmvc::cuda_detail
