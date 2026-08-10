#pragma once

#include <dsmvc/engine.hpp>

#include <cstddef>
#include <memory>
#include <span>
#include <string>

namespace dsmvc::experimental {

struct MetalFloatFrame {
    const float *source = nullptr;
    std::ptrdiff_t source_stride_bytes = 0;
    float *destination = nullptr;
    std::ptrdiff_t destination_stride_bytes = 0;
};

struct MetalFloatStagingStats {
    std::size_t memcpy_calls = 0U;
    std::size_t copied_bytes = 0U;
};

class MetalFloatExecutor final {
public:
    MetalFloatExecutor(
        std::shared_ptr<const AxisPlan> horizontal,
        std::shared_ptr<const AxisPlan> vertical,
        std::size_t maximum_batch_size,
        std::size_t threads_per_threadgroup = 32U,
        bool profile_signposts = false);
    ~MetalFloatExecutor();

    MetalFloatExecutor(const MetalFloatExecutor &) = delete;
    MetalFloatExecutor &operator=(const MetalFloatExecutor &) = delete;

    void execute(std::span<const MetalFloatFrame> frames);

    [[nodiscard]] const std::string &device_name() const noexcept;
    [[nodiscard]] std::size_t requested_buffer_bytes() const noexcept;
    [[nodiscard]] MetalFloatStagingStats last_staging_stats() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace dsmvc::experimental
